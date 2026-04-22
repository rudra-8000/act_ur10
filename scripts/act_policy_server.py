#!/usr/bin/env python3
"""
ACT Policy WebSocket Server — runs on the GPU server.
Loads a trained ACT checkpoint, listens for observations from the robot PC,
returns temporally-aggregated actions.

Install deps: pip install websockets msgpack-numpy

Usage:
  python act_policy_server.py \
    --ckpt-path /home_local/rudra_1/rudra/data/act_1/act_grasp_place_v1/policy_epoch_15000_seed_0.ckpt \
    --stats-path /home_local/rudra_1/rudra/data/act_1/act_grasp_place_v1/dataset_stats.pkl \
    --port 8765
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'detr'))
import argparse
import asyncio
import logging
import pickle
import socket
import time
import traceback

import numpy as np
import torch
from einops import rearrange

# ── ACT imports (adjust path if needed) ──────────────────────────────────────
# import sys
sys.path.insert(0, '/home_local/rudra_1/rudra/act_ur10')
from policy import ACTPolicy



logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — matches your training command
# ─────────────────────────────────────────────────────────────────────────────
POLICY_CONFIG = {
    'lr':            1e-5,
    'num_queries':   100,      # chunk_size
    'kl_weight':     10,
    'hidden_dim':    512,
    'dim_feedforward': 3200,
    'lr_backbone':   1e-5,
    'backbone':      'resnet18',
    'enc_layers':    4,
    'dec_layers':    7,
    'nheads':        8,
    'camera_names':  ['cam_high', 'cam_right_wrist'],
}

STATE_DIM   = 7      # 6 UR10 joints + 1 gripper
CHUNK_SIZE  = 100
IMG_H, IMG_W = 480, 640


# ─────────────────────────────────────────────────────────────────────────────
# Temporal aggregation state per connection
# ─────────────────────────────────────────────────────────────────────────────
class TemporalAggregator:
    """Maintains the rolling action buffer for one episode."""
    def __init__(self, max_timesteps: int = 600, chunk_size: int = CHUNK_SIZE,
                 state_dim: int = STATE_DIM, k: float = 0.01):
        self.chunk_size  = chunk_size
        self.state_dim   = state_dim
        self.k           = k
        self.max_t       = max_timesteps + chunk_size
        self.all_actions = torch.zeros(max_timesteps, max_timesteps + chunk_size, state_dim).cuda()
        self.t           = 0

    def reset(self):
        self.all_actions.zero_()
        self.t = 0

    def push(self, chunk: torch.Tensor):
        """chunk: [1, chunk_size, state_dim] — freshly predicted by the policy."""
        t = self.t
        end = min(t + self.chunk_size, self.max_t)
        width = end - t
        self.all_actions[t, t:end] = chunk[0, :width]

    def aggregate(self) -> np.ndarray:
        """Return the temporally-aggregated action for current timestep t."""
        t = self.t
        actions_for_t = self.all_actions[:t+1, t]          # [t+1, state_dim]
        valid_mask = (actions_for_t.abs().sum(dim=1) != 0)  # ignore zero rows
        valid_actions = actions_for_t[valid_mask]           # [n_valid, state_dim]
        if len(valid_actions) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)
        exp_w = np.exp(-self.k * np.arange(len(valid_actions)))
        exp_w = exp_w / exp_w.sum()
        weights = torch.from_numpy(exp_w).float().cuda().unsqueeze(1)
        action = (valid_actions * weights).sum(dim=0).cpu().numpy()
        self.t += 1
        return action.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Load policy
# ─────────────────────────────────────────────────────────────────────────────
def load_policy(ckpt_path: str, stats_path: str):
    policy = ACTPolicy(POLICY_CONFIG)
    state_dict = torch.load(ckpt_path, map_location='cuda')
    policy.load_state_dict(state_dict)
    policy.cuda().eval()
    logging.info("Loaded policy from %s", ckpt_path)

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    logging.info("Loaded stats from %s", stats_path)
    return policy, stats


def preprocess_qpos(qpos_np: np.ndarray, stats: dict) -> torch.Tensor:
    normed = (qpos_np - stats['qpos_mean']) / (stats['qpos_std'] + 1e-8)
    return torch.from_numpy(normed).float().cuda().unsqueeze(0)  # [1, state_dim]


def postprocess_action(action_np: np.ndarray, stats: dict) -> np.ndarray:
    return action_np * stats['action_std'] + stats['action_mean']


def preprocess_images(images: dict[str, np.ndarray]) -> torch.Tensor:
    """
    images: {'cam_high': [H,W,3] uint8, 'cam_right_wrist': [H,W,3] uint8}
    returns: [1, num_cams, C, H, W] float32 in [0,1]
    """
    frames = []
    for name in POLICY_CONFIG['camera_names']:
        img = images[name]  # [H, W, 3]
        t = torch.from_numpy(img).float() / 255.0
        t = rearrange(t, 'h w c -> c h w')
        frames.append(t)
    stacked = torch.stack(frames, dim=0)           # [num_cams, C, H, W]
    return stacked.unsqueeze(0).cuda()             # [1, num_cams, C, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket server
# ─────────────────────────────────────────────────────────────────────────────
def run_server(args):
    try:
        import msgpack_numpy
        from websockets.asyncio.server import ServerConnection, serve
        from websockets.exceptions import ConnectionClosed
    except ImportError as e:
        raise ImportError("pip install websockets msgpack-numpy") from e

    policy, stats = load_policy(args.ckpt_path, args.stats_path)
    packer = msgpack_numpy.Packer()

    metadata = {
        'protocol':    'act_policy_v1',
        'chunk_size':  CHUNK_SIZE,
        'state_dim':   STATE_DIM,
        'camera_names': POLICY_CONFIG['camera_names'],
        'temporal_agg': True,
    }

    async def handler(ws: ServerConnection):
        remote = ws.remote_address
        logging.info("Client connected: %s", remote)
        await ws.send(packer.pack(metadata))

        agg = TemporalAggregator()

        while True:
            try:
                t_start = time.monotonic()
                raw = msgpack_numpy.unpackb(await ws.recv())

                # ── Control messages ─────────────────────────────────────
                if isinstance(raw, dict) and raw.get('__ctrl__') == 'reset':
                    agg.reset()
                    await ws.send(packer.pack({'ok': True, 'reset': True}))
                    logging.info("Policy/aggregator reset for %s", remote)
                    continue

                # ── Observation ──────────────────────────────────────────
                obs = raw.get('observation', raw)
                qpos_np = np.array(obs['qpos'], dtype=np.float32)       # [7]
                images  = {k: obs[k] for k in POLICY_CONFIG['camera_names']}

                qpos_t = preprocess_qpos(qpos_np, stats)
                imgs_t = preprocess_images(images)

                t_infer = time.monotonic()
                with torch.inference_mode():
                    chunk = policy(qpos_t, imgs_t)  # [1, chunk_size, state_dim]
                infer_ms = (time.monotonic() - t_infer) * 1000

                agg.push(chunk)
                raw_action = agg.aggregate()                             # [7]
                action = postprocess_action(raw_action, stats)           # [7]

                resp = {
                    'action': action,            # [7] float32 — denormed
                    'timing': {
                        'infer_ms':  infer_ms,
                        'total_ms':  (time.monotonic() - t_start) * 1000,
                        'step_t':    agg.t - 1,
                    }
                }
                await ws.send(packer.pack(resp))

            except asyncio.CancelledError:
                raise
            except ConnectionClosed:
                logging.info("Client %s disconnected", remote)
                break
            except Exception:
                err = traceback.format_exc()
                logging.error("Error handling message:\n%s", err)
                await ws.send(packer.pack({'error': err}))

    async def main_async():
        async with serve(handler, '0.0.0.0', args.port,
                         max_size=None, compression=None) as server:
            hostname = socket.gethostname()
            logging.info("ACT server listening on 0.0.0.0:%d (hostname=%s)", args.port, hostname)
            await server.serve_forever()

    asyncio.run(main_async())


def parse_args():
    p = argparse.ArgumentParser(description='ACT Policy WebSocket Server')
    p.add_argument('--ckpt-path',  required=True,
                   help='Path to .ckpt file, e.g. policy_epoch_15000_seed_0.ckpt')
    p.add_argument('--stats-path', required=True,
                   help='Path to dataset_stats.pkl saved during training')
    p.add_argument('--port', type=int, default=8765)
    return p.parse_args()


if __name__ == '__main__':
    run_server(parse_args())