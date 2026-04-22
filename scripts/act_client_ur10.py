#!/usr/bin/env python3
"""
ACT Policy Client — runs on the Robot PC (connected to UR10 + cameras).
Streams observations to the GPU server, receives temporally-aggregated actions,
commands the robot.

Install deps: pip install websockets msgpack-numpy

Usage:
  python act_client_ur10.py \
    --server-host 10.245.91.19 \
    --server-port 8765 \
    --ur-ip 192.168.100.3 \
    --max-steps 400 \
    --num-rollouts 3 \
    --video-dir /tmp/act_rollouts \
    --reset-between-rollouts

Dry-run (no robot, tests server connection only):
  python act_client_ur10.py --server-host 10.245.91.19 --dry-run --max-steps 50
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — must match training data exactly
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_NAMES  = ['cam_high', 'cam_right_wrist']
CONTROL_DT    = 1.0 / 30.0   # 30 Hz — match your recording fps
STATE_DIM     = 7             # 6 UR10 joints + 1 gripper
IMG_H, IMG_W  = 480, 640

CAM_SERIALS = {
    'cam_high':        '204322061013',
    'cam_right_wrist': '923322071837',
}


# ─────────────────────────────────────────────────────────────────────────────
# Robot + Camera helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_robot_and_cameras(args):
    if args.dry_run:
        return None, None

    from lerobot.cameras import make_cameras_from_configs
    from lerobot.cameras.configs import ColorMode
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.lerobot_robot_ur10 import UR10Config

    cam_cfgs = {
        name: RealSenseCameraConfig(
            serial_number_or_name=CAM_SERIALS[name],
            fps=30,
            width=IMG_W,
            height=IMG_H,
            color_mode=ColorMode.RGB,
        )
        for name in CAMERA_NAMES
    }
    robot_cfg = UR10Config(ip=args.ur_ip)
    robot = make_robot_from_config(robot_cfg)
    robot.cameras = make_cameras_from_configs(cam_cfgs)
    robot.connect()
    logging.info("Robot connected (UR10 @ %s)", args.ur_ip)
    return robot, robot.cameras


def get_observation(robot) -> dict[str, Any]:
    """
    Returns flat dict:
      'qpos'            : np.ndarray [STATE_DIM] float32
      'cam_high'        : np.ndarray [H, W, 3]  uint8
      'cam_right_wrist' : np.ndarray [H, W, 3]  uint8
    """
    raw = robot.get_observation()

    joint_keys  = [f'joint_{i}' for i in range(6)]
    gripper_key = 'gripper'
    qpos = np.array(
        [raw[k] for k in joint_keys] + [raw[gripper_key]],
        dtype=np.float32
    )

    obs = {'qpos': qpos}
    for name in CAMERA_NAMES:
        img = raw.get(name)
        if img is None:
            img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        obs[name] = np.ascontiguousarray(img, dtype=np.uint8)
    return obs


def send_action(robot, action: np.ndarray):
    """action: [STATE_DIM] float32 — denormalized joint positions."""
    joint_keys  = [f'joint_{i}' for i in range(6)]
    gripper_key = 'gripper'
    action_dict = {k: float(action[i]) for i, k in enumerate(joint_keys)}
    action_dict[gripper_key] = float(action[6])
    robot.send_action(action_dict)


def move_to_home(robot):
    try:
        from ur10_teleoperate import default_ur_home_action, smooth_move_to_home
        smooth_move_to_home(robot, default_ur_home_action())
        logging.info("Moved to home pose")
    except ImportError:
        logging.warning("ur10_teleoperate not found — skipping home move")


def make_composite_video_frame(obs: dict) -> np.ndarray | None:
    pieces = [obs[n] for n in CAMERA_NAMES if n in obs and obs[n] is not None]
    if not pieces:
        return None
    import cv2
    h = max(p.shape[0] for p in pieces)
    resized = []
    for p in pieces:
        if p.shape[0] != h:
            p = cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h))
        resized.append(p)
    return np.concatenate(resized, axis=1)


def make_dry_run_observation() -> dict[str, Any]:
    return {
        'qpos':            np.zeros(STATE_DIM, dtype=np.float32),
        'cam_high':        np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
        'cam_right_wrist': np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main control loop
# ─────────────────────────────────────────────────────────────────────────────
async def control_loop(args: argparse.Namespace) -> None:
    try:
        import msgpack_numpy
        from websockets.asyncio.client import connect
    except ImportError as e:
        raise ImportError("pip install websockets msgpack-numpy") from e

    uri = f"ws://{args.server_host}:{args.server_port}/"
    packer = msgpack_numpy.Packer()

    robot, cameras = build_robot_and_cameras(args)
    control_dt = args.control_dt if args.control_dt else CONTROL_DT
    num_rollouts = max(1, args.num_rollouts)

    if num_rollouts > 1 and args.max_steps <= 0:
        raise ValueError("--num-rollouts > 1 requires --max-steps > 0")

    video_dir: Path | None = None
    if args.video_dir:
        video_dir = Path(args.video_dir).expanduser().resolve()
        video_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Videos will be saved to %s", video_dir)

    try:
        async with connect(uri, max_size=None, compression=None) as ws:
            # ── Handshake ────────────────────────────────────────────────
            meta_raw = await ws.recv()
            metadata = msgpack_numpy.unpackb(meta_raw)
            logging.info(
                "Server handshake OK | protocol=%s chunk_size=%s state_dim=%s",
                metadata.get('protocol'),
                metadata.get('chunk_size'),
                metadata.get('state_dim'),
            )
            if metadata.get('protocol') != 'act_policy_v1':
                logging.warning("Unexpected protocol %r — continuing anyway", metadata.get('protocol'))

            loop = asyncio.get_running_loop()

            for rollout_idx in range(num_rollouts):
                logging.info("─── Rollout %d / %d ───", rollout_idx + 1, num_rollouts)

                # ── Reset policy aggregator between rollouts ──────────────
                if rollout_idx > 0 and args.reset_between_rollouts:
                    await ws.send(packer.pack({'__ctrl__': 'reset'}))
                    ack = msgpack_numpy.unpackb(await ws.recv())
                    logging.info("Policy reset ack: %s", ack)

                # ── Move robot to home pose ───────────────────────────────
                if robot is not None:
                    await asyncio.to_thread(move_to_home, robot)
                    logging.info("Settling for 3s after home move...")
                    await asyncio.sleep(3.0)
                else:
                    logging.info("[dry-run] skipping home move")

                # ── Video writer setup ────────────────────────────────────
                writer = None
                video_path: Path | None = None
                if video_dir is not None:
                    video_path = video_dir / f"rollout_{rollout_idx:04d}.mp4"

                step = 0
                step_times = []

                while True:
                    if args.max_steps > 0 and step >= args.max_steps:
                        logging.info("Reached max_steps=%d, ending rollout", args.max_steps)
                        break

                    t0 = loop.time()

                    # ── Get observation ───────────────────────────────────
                    if robot is not None:
                        obs = await asyncio.to_thread(get_observation, robot)
                    else:
                        obs = make_dry_run_observation()

                    # ── Send to server ────────────────────────────────────
                    payload = {
                        'observation': {
                            'qpos':            np.ascontiguousarray(obs['qpos'], dtype=np.float32),
                            'cam_high':        obs['cam_high'],
                            'cam_right_wrist': obs['cam_right_wrist'],
                        }
                    }
                    await ws.send(packer.pack(payload))

                    # ── Receive action ────────────────────────────────────
                    resp_raw = await ws.recv()
                    resp = msgpack_numpy.unpackb(resp_raw)

                    if 'error' in resp:
                        raise RuntimeError(f"Server error:\n{resp['error']}")

                    if resp.get('ok') and resp.get('reset'):
                        # Server-side reset ack mid-episode (shouldn't happen normally)
                        continue

                    action = resp.get('action')
                    if action is None or not isinstance(action, np.ndarray):
                        raise RuntimeError(f"Bad action in response: {resp.keys()}")

                    # ── Record video frame ────────────────────────────────
                    if video_dir is not None and video_path is not None:
                        import cv2
                        composite = make_composite_video_frame(obs)
                        if composite is not None:
                            if writer is None:
                                h, w = composite.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                fps_out = max(1.0, 1.0 / control_dt)
                                writer = cv2.VideoWriter(str(video_path), fourcc, fps_out, (w, h))
                                if not writer.isOpened():
                                    raise RuntimeError(f"Failed to open VideoWriter for {video_path}")
                                logging.info("Recording %s at %dx%d %.1f fps", video_path.name, w, h, fps_out)
                            bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
                            writer.write(bgr)

                    # ── Command robot ─────────────────────────────────────
                    if robot is not None:
                        await asyncio.to_thread(send_action, robot, action)
                    else:
                        pass  # dry-run: just consume the action

                    # ── Timing ────────────────────────────────────────────
                    step += 1
                    elapsed = loop.time() - t0
                    step_times.append(elapsed)
                    sleep_t = control_dt - elapsed
                    if sleep_t > 0:
                        await asyncio.sleep(sleep_t)

                    if args.log_every > 0 and step % args.log_every == 0:
                        timing = resp.get('timing', {})
                        avg_step = np.mean(step_times[-args.log_every:]) * 1000
                        logging.info(
                            "rollout=%d step=%d | infer=%.1fms total=%.1fms avg_step=%.1fms agg_t=%s",
                            rollout_idx, step,
                            timing.get('infer_ms', -1),
                            timing.get('total_ms', -1),
                            avg_step,
                            timing.get('step_t', '?'),
                        )

                # ── Wrap up rollout ───────────────────────────────────────
                if writer is not None:
                    writer.release()
                    logging.info("Saved rollout %d video: %s (%d steps)", rollout_idx + 1, video_path, step)
                else:
                    logging.info("Rollout %d done — %d steps", rollout_idx + 1, step)

                if avg := (np.mean(step_times) * 1000 if step_times else 0):
                    logging.info("Avg step time: %.1f ms (%.1f Hz)", avg, 1000.0 / avg)

    finally:
        if robot is not None and getattr(robot, 'is_connected', False):
            robot.disconnect()
            logging.info("Robot disconnected")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='ACT Policy Client for UR10')
    p.add_argument('--server-host',  type=str,   default='10.245.91.19')
    p.add_argument('--server-port',  type=int,   default=8765)
    p.add_argument('--ur-ip',        type=str,   default='192.168.100.3')
    p.add_argument('--control-dt',   type=float, default=None,
                   help='Seconds per step (default: 1/30)')
    p.add_argument('--max-steps',    type=int,   default=400,
                   help='Steps per rollout (0 = run until Ctrl+C)')
    p.add_argument('--num-rollouts', type=int,   default=1)
    p.add_argument('--video-dir',    type=str,   default='',
                   help='Directory to save side-by-side MP4 recordings')
    p.add_argument('--reset-between-rollouts', action='store_true',
                   help='Send reset to server between rollouts')
    p.add_argument('--log-every',    type=int,   default=30,
                   help='Log timing every N steps (0 to disable)')
    p.add_argument('--dry-run',      action='store_true',
                   help='Run without robot — sends fake observations to test server')
    return p.parse_args()


def main():
    args = parse_args()
    try:
        asyncio.run(control_loop(args))
    except KeyboardInterrupt:
        logging.info("Stopped by user")


if __name__ == '__main__':
    main()