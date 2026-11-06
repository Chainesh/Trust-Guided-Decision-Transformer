"""Offline data: loading, the two data protocols, the train / held-out split, state normalization, and the context
windows the model is trained on.

Protocols
  episodes: D4RL's trajectories (split at terminals and timeouts), also split where consecutive rows are not one
            transition apart (obs[t+1] != next_obs[t] when the data has next_observations; maze2d and antmaze, which
            do not: a jump of the (x, y) position larger than JUMP_FACTOR x the 99th percentile step).
  stream:   (maze2d) trajectories joined across boundaries that are continuous or repeat the state, split at resets,
            terminals and unflagged jumps, then cut into evaluation-horizon chunks.
"""
import os
import pickle
import random
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import device, domain, save_pickle, load_pickle, CacheMismatch
from .envs import make_env

KEYS = ('observations', 'actions', 'rewards', 'terminals', 'next_observations')
CONTINUITY_TOL = 1e-5     # with next_observations: a step is continuous if obs[t+1] equals next_obs[t]
JUMP_FACTOR = 5.0         # without them (maze2d, antmaze): a position change > JUMP_FACTOR x p99 step is a reset
STREAM_FACTOR = 1.0       # without them: a timeout boundary is continuous if its jump is <= STREAM_FACTOR x p99 step
DT_TARGETS = {'hopper': 3600.0, 'walker2d': 5000.0, 'halfcheetah': 12000.0}   # original DT, with return scale 1000


# ---- raw data -------------------------------------------------------------------------------------------------------
def split_into_trajectories(data):
    ends = data['terminals'].astype(bool)
    if 'timeouts' in data:
        ends = ends | data['timeouts'].astype(bool)
    end_idx = list(np.where(ends)[0])
    if len(end_idx) == 0 or end_idx[-1] != len(ends) - 1:
        end_idx.append(len(ends) - 1)
    trajectories, start = [], 0
    for end in end_idx:
        trajectories.append({k: np.asarray(data[k][start:end + 1]) for k in KEYS if k in data})
        start = end + 1
    return [traj for traj in trajectories if len(traj['rewards']) > 1]


def load_raw(name, data_dir):
    """The min-DT pkl file of a dataset (trajectories in dataset order), created from D4RL if it is missing."""
    path = os.path.join(data_dir, name + '.pkl')
    if not os.path.exists(path):
        built = split_into_trajectories(make_env(name).get_dataset())
        os.makedirs(data_dir, exist_ok=True)
        with open(path + '.tmp', 'wb') as f:
            pickle.dump(built, f)
        os.replace(path + '.tmp', path)
    with open(path, 'rb') as f:
        trajectories = pickle.load(f)
    out = []
    for traj in trajectories:
        rewards = np.asarray(traj['rewards'], dtype=np.float32).reshape(-1)
        t = {'observations': np.asarray(traj['observations'], dtype=np.float32),
             'actions': np.asarray(traj['actions'], dtype=np.float32), 'rewards': rewards,
             'terminals': np.asarray(traj.get('terminals', np.zeros(len(rewards))), dtype=bool).reshape(-1)}
        if 'next_observations' in traj:
            t['next_observations'] = np.asarray(traj['next_observations'], dtype=np.float32)
        out.append(t)
    return out


def jump_dims(name):
    return slice(0, 2) if domain(name) in ('maze2d', 'antmaze') else slice(None)


def step_p99(trajectories, name):
    d = jump_dims(name)
    steps = np.concatenate([np.linalg.norm(np.diff(tr['observations'][:, d], axis=0), axis=1)
                            for tr in trajectories[:2000]])
    return float(np.percentile(steps, 99))


def cut(traj, a, b):
    return {k: v[a:b] for k, v in traj.items()}


def join(parts):
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def split_discontinuities(trajectories, name):
    step, d = step_p99(trajectories, name), jump_dims(name)
    out, n_cuts = [], 0
    for tr in trajectories:
        if 'next_observations' in tr:
            bad = np.linalg.norm(tr['observations'][1:] - tr['next_observations'][:-1], axis=1) > CONTINUITY_TOL
        elif domain(name) in ('maze2d', 'antmaze'):
            bad = np.linalg.norm(np.diff(tr['observations'][:, d], axis=0), axis=1) > JUMP_FACTOR * step
        else:
            bad = np.zeros(len(tr['rewards']) - 1, dtype=bool)
        cuts = list(np.where(bad)[0] + 1)
        n_cuts += len(cuts)
        for a, b in zip([0] + cuts, cuts + [len(tr['rewards'])]):
            if b - a > 1:
                out.append(cut(tr, a, b))
    return out, n_cuts


def boundary_types(trajectories, name):
    step, d = step_p99(trajectories, name), jump_dims(name)
    kinds = []
    for a, b in zip(trajectories[:-1], trajectories[1:]):
        if a['terminals'][-1]:
            kinds.append('terminal')
        elif np.array_equal(b['observations'][0], a['observations'][-1]):
            kinds.append('repeat')
        elif 'next_observations' in a:
            kinds.append('continuous' if np.linalg.norm(b['observations'][0] - a['next_observations'][-1])
                         <= CONTINUITY_TOL else 'reset')
        else:
            jump = np.linalg.norm(b['observations'][0, d] - a['observations'][-1, d])
            kinds.append('continuous' if jump <= STREAM_FACTOR * step else 'reset')
    return kinds


def stream_protocol(raw, name, horizon):
    """At a 'repeat' boundary the last row of the first trajectory is dropped: its next state was never recorded."""
    kinds = boundary_types(raw, name)
    streams, current = [], [raw[0]]
    for kind, b in zip(kinds, raw[1:]):
        if kind in ('continuous', 'repeat'):
            if kind == 'repeat':
                current[-1] = cut(current[-1], 0, len(current[-1]['rewards']) - 1)
            current.append(b)
        else:
            streams.append(current)
            current = [b]
    streams.append(current)
    joined = [join(ps) for ps in ([p for p in parts if len(p['rewards']) > 0] for parts in streams) if ps]
    pieces, n_cuts = split_discontinuities(joined, name)
    chunks = [cut(p, s0, s0 + horizon) for p in pieces for s0 in range(0, len(p['rewards']), horizon)]
    return [c for c in chunks if len(c['rewards']) > 1], n_cuts


def fingerprint(trajectories):
    h = hashlib.sha1()
    for traj in trajectories:
        h.update(np.int64(len(traj['rewards'])).tobytes())
        for k in ('observations', 'actions', 'rewards', 'terminals'):
            h.update(np.ascontiguousarray(traj[k]).tobytes())
    return h.hexdigest()


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


# ---- one dataset under one protocol -----------------------------------------------------------------------------------
class Setup:
    """A dataset under a protocol: environment facts, the processed trajectories, the train / held-out split (10%
    held out, RandomState(12345)), the training split's state mean and std, and the cache directory of its runs."""

    def __init__(self, name, protocol, data_dir='./dataset/', runs_dir='./runs/'):
        self.name, self.protocol, self.runs_dir = name, protocol, runs_dir
        env = make_env(name)
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.max_ep_len = env.spec.max_episode_steps
        self.ref_max = float(env.unwrapped.ref_max_score)
        self.domain = domain(name)

        raw = load_raw(name, data_dir)
        # actions outside [-1, 1] (the tanh head's range) are divided by the action bound and multiplied back when run
        max_abs = np.max(np.stack([np.abs(tr['actions']).max(0) for tr in raw]), 0)
        self.act_scale = np.ones(self.act_dim, dtype=np.float32)
        if max_abs.max() > 1.0 + 1e-3:
            high = np.asarray(env.action_space.high, dtype=np.float32)
            self.act_scale = np.where(np.isfinite(high) & (high >= max_abs - 1e-3), high, max_abs).astype(np.float32)
            for tr in raw:
                tr['actions'] = np.clip(tr['actions'] / self.act_scale, -1.0, 1.0)

        if protocol == 'episodes':
            self.trajectories, self.n_cuts = split_discontinuities(raw, name)
            self.tag = '__ep'
        elif protocol == 'stream':
            self.trajectories, self.n_cuts = stream_protocol(raw, name, self.max_ep_len)
            self.tag = f'__st{self.max_ep_len}'
        else:
            raise ValueError(protocol)

        fp = fingerprint(self.trajectories)
        meta_path = os.path.join(self.run_dir(), 'meta.pkl')
        if not os.path.exists(meta_path):
            n = len(self.trajectories)
            perm = np.random.RandomState(12345).permutation(n)
            n_val = max(1, int(0.1 * n))
            train_idx, val_idx = np.sort(perm[n_val:]), np.sort(perm[:n_val])
            train_obs = np.concatenate([self.trajectories[i]['observations'] for i in train_idx], axis=0)
            save_pickle(dict(train_idx=train_idx, val_idx=val_idx, state_mean=train_obs.mean(0).astype(np.float32),
                             state_std=(train_obs.std(0) + 1e-6).astype(np.float32),
                             max_timestep=int(max(max(len(tr['rewards']) for tr in self.trajectories),
                                                  self.max_ep_len)) + 1,
                             n_trajectories=n, protocol=protocol, fingerprint=fp), meta_path)
        self.meta = load_pickle(meta_path)
        if 'fingerprint' not in self.meta:
            # a meta.pkl saved without a fingerprint: accepted only if the data reproduce what it recorded (trajectory
            # count, longest trajectory, the training split's state mean and std)
            m = self.meta
            ok = m['n_trajectories'] == len(self.trajectories) and m['train_idx'].max() < len(self.trajectories)
            if ok:
                obs = np.concatenate([self.trajectories[i]['observations'] for i in m['train_idx']], axis=0)
                mean_now, std_now = obs.mean(0), obs.std(0) + 1e-6
                ok = (np.all(np.abs(mean_now - m['state_mean']) <= 1e-4 * std_now + 1e-6)
                      and np.all(np.abs(std_now - m['state_std']) <= 1e-4 * std_now + 1e-6)
                      and m['max_timestep'] == int(max(max(len(tr['rewards']) for tr in self.trajectories),
                                                       self.max_ep_len)) + 1)
            if not ok:
                raise CacheMismatch(f'{meta_path}: the data differ from those meta.pkl was written from')
            self.meta['fingerprint'] = fp
            save_pickle(self.meta, meta_path)
        if self.meta['fingerprint'] != fp:
            raise CacheMismatch(f'{meta_path}: the data differ from those the cached runs were made from')
        self.fingerprint = fp
        self.state_mean, self.state_std = self.meta['state_mean'], self.meta['state_std']
        self.train_trajs = [self.trajectories[i] for i in self.meta['train_idx']]
        self.val_trajs = [self.trajectories[i] for i in self.meta['val_idx']]
        rtg_all = np.concatenate([discount_cumsum(tr['rewards'], 1.0) for tr in self.train_trajs])
        self.data_rtg_range = (float(rtg_all.min()), float(rtg_all.max()))
        self._datasets = {}

    def run_dir(self, *parts):
        path = os.path.join(self.runs_dir, self.name + self.tag, *[str(p) for p in parts])
        os.makedirs(path, exist_ok=True)
        return path

    def rtg_scale(self, cfg):
        if cfg['rtg_rule'] == 'dt' and self.domain == 'locomotion':
            return 1000.0
        return float(10 ** np.floor(np.log10(max(abs(self.ref_max), 1.0))))

    def target(self, cfg):
        """The return-to-go the rollout asks for: the configuration's target, else the rule's."""
        t = cfg.get('target')
        if t == 'data max':
            return self.data_rtg_range[1]
        if isinstance(t, str) and t.startswith('x'):          # 'x2': twice the reference maximum
            return float(t[1:]) * self.ref_max
        if t is not None:
            return float(t)
        if cfg['rtg_rule'] == 'dt':
            return next((v for prefix, v in DT_TARGETS.items() if self.name.startswith(prefix)), self.ref_max)
        return self.ref_max

    def datasets(self, rtg_scale, context_len):
        key = (rtg_scale, context_len)
        if key not in self._datasets:
            self._datasets[key] = tuple(D4RLTrajectoryDataset(trs, context_len, rtg_scale, self.state_mean,
                                                              self.state_std) for trs in (self.train_trajs,
                                                                                          self.val_trajs))
        return self._datasets[key]


# ---- context windows ------------------------------------------------------------------------------------------------
class D4RLTrajectoryDataset(Dataset):
    def __init__(self, trajectories, context_len, rtg_scale, state_mean, state_std):
        self.context_len = context_len
        self.trajectories = []                     # normalized copies, the raw trajectories are left untouched
        for traj in trajectories:
            self.trajectories.append({
                'observations': ((traj['observations'] - state_mean) / state_std).astype(np.float32),
                'actions': traj['actions'].astype(np.float32),
                'returns_to_go': (discount_cumsum(traj['rewards'], 1.0) / rtg_scale).reshape(-1, 1).astype(np.float32),
            })

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]
        K = self.context_len
        if traj_len >= K:
            si = random.randint(0, traj_len - K)
            out = [torch.from_numpy(traj[k][si:si + K]) for k in ('observations', 'actions', 'returns_to_go')]
            return (torch.arange(si, si + K), *out, torch.ones(K, dtype=torch.long))
        pad = K - traj_len
        out = [torch.from_numpy(traj[k]) for k in ('observations', 'actions', 'returns_to_go')]
        out = [torch.cat([x, torch.zeros([pad] + list(x.shape[1:]))], dim=0) for x in out]
        mask = torch.cat([torch.ones(traj_len, dtype=torch.long), torch.zeros(pad, dtype=torch.long)], dim=0)
        return (torch.arange(K), *out, mask)


class WindowSampler:
    """Batches of context windows (a trajectory uniformly, or in proportion to its length as in the original DT, then
    a uniform start; shorter trajectories right-padded and masked), drawn and gathered on the device."""

    def __init__(self, dataset, batch_size, by_length=False):
        trajs, self.K, self.batch_size = dataset.trajectories, dataset.context_len, batch_size
        lens = np.array([len(traj['observations']) for traj in trajs])
        weights = lens.astype(np.float64) if by_length else np.ones(len(trajs))
        self.cdf = torch.as_tensor(np.cumsum(weights) / weights.sum(), device=device)
        self.lens = torch.as_tensor(lens, device=device)
        self.offsets = torch.as_tensor(np.concatenate([[0], np.cumsum(lens)[:-1]]), device=device)
        self.data = [torch.as_tensor(np.concatenate([traj[k] for traj in trajs]), device=device)
                     for k in ('observations', 'actions', 'returns_to_go')]

    def __iter__(self):
        return self

    def __next__(self):
        u = torch.rand(self.batch_size, dtype=torch.float64, device=device)
        idx = torch.searchsorted(self.cdf, u, right=True).clamp(max=len(self.lens) - 1)
        lens = self.lens[idx]
        n_starts = (lens - self.K + 1).clamp(min=1)
        start = (torch.rand(self.batch_size, dtype=torch.float64, device=device) * n_starts).long()
        timesteps = start[:, None] + torch.arange(self.K, device=device)    # 0..K-1 when padded
        mask = timesteps < lens[:, None]
        rows = self.offsets[idx][:, None] + timesteps * mask
        states, actions, returns_to_go = [x[rows] * mask.unsqueeze(-1) for x in self.data]
        return timesteps, states, actions, returns_to_go, mask.long()
