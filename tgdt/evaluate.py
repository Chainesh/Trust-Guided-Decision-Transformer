"""Evaluation of one trained model: num_eval_ep episodes, episode e of training seed s run with environment seed
EVAL_SEED_BASE + 1000 s + e, EVAL_BATCH episodes in lockstep. Results are cached beside the model and reused only if
the model file, the critic file, the data and the evaluation seeds are the ones they were made with."""
import os
import time

import numpy as np
import torch

from .calibrate import get_taus
from .common import save_json, load_json, save_pickle, file_id
from .critic import get_critic, critic_path
from .envs import make_env, close_env, d4rl_score, env_goal
from .rollout import run_episodes, log_metrics
from .train import get_model, model_dir

EVAL_SEED_BASE = 70000
EVAL_BATCH = 25


def eval_key(setup, cfg, target):
    key = (f'{cfg["selection"].replace(" ", "-")}_{cfg["eval_weights"]}_Ke{cfg["Ke"]}_a{cfg["alpha"]:g}_'
           f'{cfg["tau_mode"]}_ep{cfg["num_eval_ep"]}_R{target:g}_L{"-".join(str(L) for L in cfg["suffixes"])}')
    if cfg['selection'] == 'hard reset':
        key += f'_cd{cfg["hard_reset_cooldown"]}'
    return key


def evaluate(setup, cfg, seed, force=False, save_logs=False, cache_only=False):
    """The cached evaluation of (cfg, seed), made (training the model first if needed) when missing or stale."""
    target = setup.target(cfg)
    out_dir = os.path.join(model_dir(setup, cfg, seed), 'eval')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, eval_key(setup, cfg, target) + '.json')
    model_path = os.path.join(model_dir(setup, cfg, seed), 'model.pt')
    if os.path.exists(json_path) and not force and os.path.exists(model_path):
        result = load_json(json_path)
        cpath = critic_path(setup, {**cfg}, seed)
        if (result['model_id'] == file_id(model_path) and result['fingerprint'] == setup.fingerprint
                and result['eval_seed_base'] == EVAL_SEED_BASE and os.path.exists(cpath)
                and result['critic_id'] == file_id(cpath)):
            return result
    if cache_only:
        return None

    model = get_model(setup, cfg, seed, cfg['eval_weights'])
    critic = get_critic(setup, model.cfg, seed)[0]
    taus = get_taus(setup, model, cfg, cfg['eval_weights'])
    start = time.time()
    episodes, logs = rollouts(setup, model, critic, taus, cfg, seed, target)

    result = {'env': setup.name, 'protocol': setup.protocol, 'config': cfg['name'], 'arch': cfg['arch'],
              'seed': seed, 'selection': cfg['selection'], 'weights': cfg['eval_weights'], 'target': target,
              'rtg_scale': model.cfg['rtg_scale'], 'suffixes': cfg['suffixes'],
              'taus': {str(L): v for L, v in taus.items()}, 'model_id': model.file_id,
              'critic_id': file_id(critic_path(setup, model.cfg, seed)), 'fingerprint': setup.fingerprint,
              'eval_seed_base': EVAL_SEED_BASE,
              'versions': {'torch': torch.__version__, 'numpy': np.__version__},
              'summary': {k: float(np.nanmean([e[k] for e in episodes])) for k in episodes[0] if k != 'goal'},
              'episodes': episodes}
    save_json(result, json_path)
    if save_logs:
        save_pickle(logs, json_path[:-5] + '_logs.pkl')
    s = result['summary']
    print(f'[{setup.name} | {cfg["name"]} | seed {seed} | {cfg["selection"]} | {cfg["eval_weights"]}] score '
          f'{s["score"]:.1f} | rejected {s["rejected %"]:.1f}% | filter changed {s["filter changed %"]:.1f}% | '
          f'fallback {s["fallback %"]:.1f}% | {time.time() - start:.0f}s', flush=True)
    return result


def rollouts(setup, model, critic, taus, cfg, seed, target):
    """The num_eval_ep evaluation episodes of a model: per-episode return, score, length, goal and filter statistics."""
    n_ep = cfg['num_eval_ep']
    outer_state = np.random.get_state()
    if setup.domain == 'antmaze':   # antmaze-v0 draws each environment's goal from numpy's global generator when it is
        np.random.seed(EVAL_SEED_BASE + 1000 * seed + 999)   # made: seeded, so every rule meets the same goals
    envs = [make_env(setup.name) for _ in range(min(EVAL_BATCH, n_ep))]
    np.random.set_state(outer_state)
    goals = [env_goal(e) for e in envs]
    seeds = [EVAL_SEED_BASE + 1000 * seed + ep for ep in range(n_ep)]
    logs = []
    try:
        for b in range(0, n_ep, len(envs)):
            batch = seeds[b:b + len(envs)]
            logs += run_episodes(setup, model, critic, envs[:len(batch)], taus, batch, target, cfg)
        episodes = []
        for log in logs:
            total = float(log['reward'].sum())
            episodes.append({'return': total, 'score': d4rl_score(envs[0], total), 'length': len(log['reward']),
                             'goal': goals[len(episodes) % len(envs)], **log_metrics(log)})
    finally:
        for e in envs:
            close_env(e)
    return episodes, logs


def mean_se(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0)
