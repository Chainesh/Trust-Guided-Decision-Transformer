"""The frozen critic Q_phi of Eq. 3 and Eq. 11: IQL (Kostrikov et al. 2022) with twin Q and V, expectile regression
of V on the target Q, and Q regressed on r + gamma (1 - done) V(s'). Trained once per (dataset, seed, critic settings),
cached in a file named by its settings, and frozen.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import device, set_seed, save_torch, load_torch, CacheMismatch
from .config import critic_settings, check_same

CRITIC_DEFAULTS = {'critic_layers': 2, 'critic_ln': False}            # for files saved without these settings
LEGACY_IQL = dict(iql_batch=256, iql_lr=3e-4, iql_polyak=0.005, gamma=0.99)


def mlp(in_dim, out_dim, hidden=256, layers=2, ln=False):
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden)] + ([nn.LayerNorm(hidden)] if ln else []) + [nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class TwinQ(nn.Module):
    def __init__(self, state_dim, act_dim, layers=2, ln=False):
        super().__init__()
        self.q1 = mlp(state_dim + act_dim, 1, layers=layers, ln=ln)
        self.q2 = mlp(state_dim + act_dim, 1, layers=layers, ln=ln)

    def both(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def forward(self, states, actions):
        return torch.min(*self.both(states, actions))


class ValueNet(nn.Module):
    def __init__(self, state_dim, layers=2, ln=False):
        super().__init__()
        self.v = mlp(state_dim, 1, layers=layers, ln=ln)

    def forward(self, states):
        return self.v(states).squeeze(-1)


def iql_nets(setup, cfg):
    arch = dict(layers=cfg['critic_layers'], ln=cfg['critic_ln'])
    return (TwinQ(setup.state_dim, setup.act_dim, **arch).to(device),
            TwinQ(setup.state_dim, setup.act_dim, **arch).to(device), ValueNet(setup.state_dim, **arch).to(device))


def make_transitions(setup, cfg):
    """(s, a, r, s', done) of the training trajectories. s' is D4RL's next_observations when the data has them;
    otherwise the following row, and the last row of a trajectory that does not end in a terminal is dropped."""
    S, A, R, S2, D = [], [], [], [], []
    for traj in setup.train_trajs:
        n = len(traj['rewards'])
        obs = traj['observations']
        if 'next_observations' in traj:
            next_obs, last = traj['next_observations'], n
        else:
            next_obs, last = np.concatenate([obs[1:], obs[-1:]], axis=0), (n if traj['terminals'][-1] else n - 1)
        S.append(obs[:last]); A.append(traj['actions'][:last]); R.append(traj['rewards'][:last])
        S2.append(next_obs[:last]); D.append(traj['terminals'][:last].astype(np.float32))
    S, A, R, S2, D = [np.concatenate(x, axis=0) for x in (S, A, R, S2, D)]
    if cfg['iql_reward_norm']:             # IQL (locomotion): rewards x 1000 / (max - min trajectory return)
        returns = np.array([traj['rewards'].sum() for traj in setup.train_trajs])
        R = R * 1000.0 / max(returns.max() - returns.min(), 1e-6)
    R = R + cfg['reward_shift']
    S, S2 = (S - setup.state_mean) / setup.state_std, (S2 - setup.state_mean) / setup.state_std
    return [torch.as_tensor(x, dtype=torch.float32, device=device) for x in (S, A, R, S2, D)]


def train_iql(setup, cfg, seed):
    set_seed(seed)
    S, A, R, S2, D = make_transitions(setup, cfg)
    q, q_target, v = iql_nets(setup, cfg)
    q_target.load_state_dict(q.state_dict())
    q_optimizer = torch.optim.Adam(q.parameters(), lr=cfg['iql_lr'])
    v_optimizer = torch.optim.Adam(v.parameters(), lr=cfg['iql_lr'])
    start = time.time()
    for step in range(cfg['iql_steps']):
        idx = torch.randint(0, len(S), (cfg['iql_batch'],), device=device)
        s, a, r, s2, d = S[idx], A[idx], R[idx], S2[idx], D[idx]
        with torch.no_grad():
            target_q = q_target(s, a)
        diff = target_q - v(s)
        v_loss = (torch.abs(cfg['iql_expectile'] - (diff < 0).float()) * diff ** 2).mean()
        v_optimizer.zero_grad()
        v_loss.backward()
        v_optimizer.step()
        with torch.no_grad():
            backup = r + cfg['gamma'] * (1.0 - d) * v(s2)
        q1, q2 = q.both(s, a)
        q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()
        with torch.no_grad():
            for p, p_target in zip(q.parameters(), q_target.parameters()):
                p_target.mul_(1 - cfg['iql_polyak']).add_(cfg['iql_polyak'] * p)
        if (step + 1) % 100000 == 0:
            print(f'  [{setup.name} | seed {seed}] iql {step + 1}/{cfg["iql_steps"]}: q_loss {q_loss.item():.4f} | '
                  f'mean Q {q1.mean().item():.2f} | {time.time() - start:.0f}s', flush=True)
    print(f'[{setup.name} | seed {seed}] iql {cfg["iql_steps"]} steps, expectile {cfg["iql_expectile"]:g}, reward '
          f'shift {cfg["reward_shift"]:g}, gamma {cfg["gamma"]:g}: q_loss {q_loss.item():.4f} | mean Q '
          f'{q1.mean().item():.2f} | {time.time() - start:.0f}s', flush=True)
    return q, v


# ---- cache ----------------------------------------------------------------------------------------------------------
def critic_name(cfg):
    name = f'critic_{cfg["iql_steps"]}'
    if cfg['iql_expectile'] != 0.7 or cfg['reward_shift'] != 0.0:
        name += f'_e{cfg["iql_expectile"]:g}_r{cfg["reward_shift"]:g}'
    if cfg['iql_reward_norm']:
        name += '_norm'
    if cfg['gamma'] != 0.99:
        name += f'_g{cfg["gamma"]:g}'
    if cfg['critic_ln'] or cfg['critic_layers'] != 2:
        name += f'_{"ln" if cfg["critic_ln"] else "l"}{cfg["critic_layers"]}'
    return name


def critic_path(setup, cfg, seed):
    return os.path.join(setup.run_dir(f'seed{seed}'), critic_name(cfg) + '.pt')


def _freeze(*nets):
    for net in nets:
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)


def get_critic(setup, cfg, seed):
    """(Q, V) for these critic settings, trained once and cached."""
    path = critic_path(setup, cfg, seed)
    settings = critic_settings(cfg)
    if not os.path.exists(path):
        q, v = train_iql(setup, cfg, seed)
        save_torch({'q': q.state_dict(), 'v': v.state_dict(), 'steps': cfg['iql_steps'], 'iql': settings,
                    'fingerprint': setup.fingerprint}, path)
    ckpt = load_torch(path)
    named = {k: settings[k] for k in ('iql_steps', 'iql_expectile', 'reward_shift', 'iql_reward_norm', 'gamma')}
    stored = ckpt.get('iql', {**LEGACY_IQL, **named, 'iql_steps': ckpt['steps']})
    check_same({**CRITIC_DEFAULTS, **stored}, settings, path)
    if ckpt.get('fingerprint', setup.fingerprint) != setup.fingerprint:
        raise CacheMismatch(f'{path} was trained on other data')
    q, _, v = iql_nets(setup, cfg)
    q.load_state_dict(ckpt['q'])
    v.load_state_dict(ckpt['v'])
    _freeze(q, v)
    return q, v
