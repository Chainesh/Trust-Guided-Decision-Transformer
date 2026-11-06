"""Calibration of the trust threshold (Sec. 3.3, Eqs. 6-7) on the held-out trajectories."""
import math
import os

import numpy as np
import torch

from .common import device, save_pickle, load_pickle


@torch.no_grad()
def heldout_errors(model, trajectories, L, batch_size=1024):
    """Teacher-forced per-step errors e_t = ||s_hat_{t+1} - s_{t+1}||^2 / d_s on held-out trajectories, where the
    prediction for step t uses the context of the last L steps."""
    all_errors = []
    for traj in trajectories:
        obs = torch.as_tensor(traj['observations'], device=device)
        act = torch.as_tensor(traj['actions'], device=device)
        rtg = torch.as_tensor(traj['returns_to_go'], device=device)
        n = len(obs)
        errors = []
        for b in range(0, n - 1, batch_size):
            ends = np.arange(b, min(b + batch_size, n - 1))
            starts = np.maximum(0, ends - L + 1)
            src = starts[:, None] + np.arange(L)[None, :]            # right-padded windows [start, t]
            valid = src <= ends[:, None]
            src = torch.as_tensor(np.where(valid, src, 0), device=device)
            valid = torch.as_tensor(valid, device=device)
            vf = valid.unsqueeze(-1).float()
            preds = model(src * valid, obs[src] * vf, act[src] * vf, rtg[src] * vf)[0]
            last = torch.as_tensor(ends - starts, device=device)
            pred = preds[torch.arange(len(ends), device=device), last]
            target = obs[torch.as_tensor(ends + 1, device=device)]
            errors.append(((pred - target) ** 2).mean(-1).cpu().numpy())
        all_errors.append(np.concatenate(errors) if errors else np.zeros(0))
    return all_errors


def rolling_scores(errors, Ke):
    """S_t, Eq. 6: mean of the last Ke errors (fewer at the start of a trajectory)."""
    c = np.concatenate([[0.0], np.cumsum(errors)])
    t = np.arange(len(errors))
    lo = np.maximum(0, t - Ke + 1)
    return (c[t + 1] - c[lo]) / (t + 1 - lo)


def conformal_threshold(error_list, Ke, alpha):
    """tau_alpha, Eq. 7: the ceil((n+1)(1-alpha))-th order statistic of held-out rolling scores."""
    scores = np.concatenate([rolling_scores(e, Ke) for e in error_list if len(e) > 0])
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(np.sort(scores)[min(k, n) - 1])


def get_taus(setup, model, cfg, weights='raw'):
    """tau for every suffix length. 'shared' (Table 2): the full context's threshold for all suffixes; 'per suffix':
    each suffix's own. Held-out errors are cached beside the model, keyed by the model file and weights."""
    K = cfg['context_len']
    lengths = sorted(set(cfg['suffixes']) | {K})
    path = os.path.join(os.path.dirname(model.path), f'calib_{weights}.pkl')
    calib = load_pickle(path) if os.path.exists(path) else None
    if calib is None or calib.get('model_id') != model.file_id or not set(lengths) <= set(calib['errors']):
        _, val_ds = setup.datasets(model.cfg['rtg_scale'], K)
        calib = {'model_id': model.file_id, 'errors': {L: heldout_errors(model, val_ds.trajectories, L)
                                                       for L in lengths}}
        save_pickle(calib, path)
    errors = calib['errors']
    if cfg['tau_mode'] == 'shared':
        tau = conformal_threshold(errors[K], cfg['Ke'], cfg['alpha'])
        return {L: tau for L in lengths}
    return {L: conformal_threshold(errors[L], cfg['Ke'], cfg['alpha']) for L in lengths}
