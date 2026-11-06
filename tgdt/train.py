"""Training (Eq. 5):  L = w ||a_hat - a||^2 - lambda_Q Q(s, a_hat) + beta ||delta||^2 + lambda_s ||s_hat' - s'||^2.

Random numbers: the critic is trained or loaded first, then the seed is set, so the model's initialization, its training
windows and dropout masks do not depend on whether the critic was cached; configurations with the same architecture
and sampling get the same initialization and windows. A resumed run restores the saved random state after all setup,
so it continues exactly as an uninterrupted run would.
"""
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import device, set_seed, save_torch, load_torch, rng_state, set_rng_state, file_id
from .config import train_keys, train_hash, check_same
from .critic import get_critic
from .data import WindowSampler
from .model import build_model

LOG_FIELDS = ['iter', 'num_updates', 'action_loss', 'q_value', 'critic_term', 'residual_loss', 'state_loss',
              'residual_abs', 'drift', 'g_imitation', 'g_critic', 'g_residual', 'g_state', 'cos_imitation_critic',
              'lambda_Q_const', 'lr', 'seconds']


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum()


def flat_grad(term, params):
    if not term.requires_grad:
        return None
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g, p in zip(grads, params)])


def grad_stats(terms, params):
    """Gradient norm of each loss term, and the cosine between the update directions of imitation and the critic term
    (positive: the critic pulls the same way as imitation). Draws no random numbers."""
    g = {k: flat_grad(t, params) for k, t in terms.items()}
    out = {f'g_{k}': (0.0 if v is None else float(v.norm())) for k, v in g.items()}
    gi, gc = g['imitation'], g['critic']
    ok = gi is not None and gc is not None and out['g_imitation'] > 0 and out['g_critic'] > 0
    out['cos_imitation_critic'] = float(F.cosine_similarity(-gi, gc, dim=0)) if ok else float('nan')
    return out


@torch.no_grad()
def advantage_weights(critic, value, states, actions, mask, adv_beta):
    """w_t of Eq. 3: z = Q(s, a) - V(s) standardized over the batch's valid steps, w = exp(adv_beta z) capped at 20,
    then divided by its batch mean."""
    adv = critic(states, actions) - value(states)
    adv_mean, n = masked_mean(adv, mask), mask.sum()
    adv_std = torch.sqrt(masked_mean((adv - adv_mean) ** 2, mask) * n / (n - 1))
    adv = (adv - adv_mean) / (adv_std + 1e-6)
    w = torch.exp(adv_beta * adv).clamp(max=20.0)
    return w / masked_mean(w, mask)


@torch.no_grad()
def data_q_scale(critic, dataset, chunk=65536):
    """mean |Q(s, a)| of the frozen critic over every (state, action) of the training trajectories."""
    S = np.concatenate([traj['observations'] for traj in dataset.trajectories])
    A = np.concatenate([traj['actions'] for traj in dataset.trajectories])
    total = 0.0
    for i in range(0, len(S), chunk):
        total += float(critic(torch.as_tensor(S[i:i + chunk], device=device),
                              torch.as_tensor(A[i:i + chunk], device=device)).abs().sum())
    return total / len(S)


def model_cfg(setup, cfg):
    """The configuration as training uses it: the YAML settings and the return scale of this dataset."""
    return {**cfg, 'rtg_scale': setup.rtg_scale(cfg)}


def model_dir(setup, cfg, seed):
    return setup.run_dir(f'seed{seed}', f'tgdt_{cfg["arch"]}_{train_hash(cfg)}')


def _read_log(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _write_log(path, rows):
    with open(path + '.tmp', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    os.replace(path + '.tmp', path)


def train(setup, cfg, seed):
    mcfg = model_cfg(setup, cfg)
    out_dir = model_dir(setup, cfg, seed)
    last_path, log_path = os.path.join(out_dir, 'last.pt'), os.path.join(out_dir, 'train_log.csv')
    reduce = (lambda x: x.sum(-1)) if mcfg['loss_reduction'] == 'sum' else (lambda x: x.mean(-1))
    tkeys = train_keys(mcfg)

    critic, value = (get_critic(setup, mcfg, seed) if (mcfg['lambda_Q'] > 0 or mcfg['adv_weights'])
                     else (None, None))
    train_ds, _ = setup.datasets(mcfg['rtg_scale'], mcfg['context_len'])

    lambda_q_const = None                          # q_norm 'once': lambda_Q = c / mean|Q| on the data, a constant
    if mcfg['q_norm'] == 'once' and mcfg['lambda_Q'] > 0:
        q_scale = data_q_scale(critic, train_ds)
        lambda_q_const = mcfg['lambda_Q'] / max(q_scale, 1e-6)
        print(f'[{setup.name} | seed {seed}] lambda_Q = {mcfg["lambda_Q"]:g} / mean |Q| ({q_scale:.4g}) = '
              f'{lambda_q_const:.4g}', flush=True)

    set_seed(seed)
    model = build_model(mcfg, setup.state_dim, setup.act_dim, setup.meta['max_timestep'], device)
    params = list(model.parameters())
    try:
        optimizer = torch.optim.AdamW(params, lr=mcfg['lr'], weight_decay=mcfg['wt_decay'], foreach=True)
    except TypeError:                              # torch < 1.12
        optimizer = torch.optim.AdamW(params, lr=mcfg['lr'], weight_decay=mcfg['wt_decay'])
    total_updates = mcfg['max_train_iters'] * mcfg['num_updates_per_iter']
    warm = mcfg['warmup_steps']
    if mcfg['lr_schedule'] == 'cosine':           # warmup, then cosine decay to 0 at the last update
        lr_at = lambda steps: ((steps + 1) / warm if steps < warm else
                               0.5 * (1 + math.cos(math.pi * min(steps - warm, total_updates - warm)
                                                   / max(total_updates - warm, 1))))
    else:                                          # warmup, then constant
        lr_at = lambda steps: min((steps + 1) / warm, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    batches = WindowSampler(train_ds, mcfg['batch_size'], mcfg['sample_by_length'])
    ema_decay = mcfg['ema_decay']
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if ema_decay else None
    ema_pairs = [(ema[k], v) for k, v in model.state_dict().items() if v.dtype.is_floating_point] if ema else []

    start_iter = 0
    if os.path.exists(last_path):                  # resume an interrupted run
        ckpt = load_torch(last_path)
        check_same(ckpt['cfg'], tkeys, last_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if ema is not None:
            for k, v in ckpt['ema'].items():
                ema[k].copy_(v)
        start_iter = ckpt['iter'] + 1
        set_rng_state(ckpt['rng'])                 # last, after everything that draws random numbers
    log_rows = [r for r in _read_log(log_path) if int(r['iter']) < start_iter]
    start_time = time.time()

    for it in range(start_iter, mcfg['max_train_iters']):
        model.train()
        logs = {k: [] for k in ('action_loss', 'q_value', 'critic_term', 'residual_loss', 'state_loss', 'residual_abs',
                                'drift')}
        norms = {}
        for u in range(mcfg['num_updates_per_iter']):
            timesteps, states, actions, returns_to_go, traj_mask = next(batches)
            state_preds, action_preds, delta, base = model(timesteps, states, actions, returns_to_go)
            mask = traj_mask.float()
            next_mask = mask[:, 1:]

            w = (advantage_weights(critic, value, states, actions, mask, mcfg['adv_beta']) if mcfg['adv_weights']
                 else torch.ones_like(mask))
            action_loss = masked_mean(w * reduce((action_preds - actions) ** 2), mask)      # Eq. 3, on tanh output
            if mcfg['lambda_Q'] > 0:
                q_value = masked_mean(critic(states, action_preds), mask)
            else:
                q_value = torch.zeros((), device=device)
            critic_term = (lambda_q_const if lambda_q_const is not None else mcfg['lambda_Q']) * q_value
            residual_loss = masked_mean(reduce(delta ** 2), mask)
            state_loss = masked_mean(((state_preds[:, :-1] - states[:, 1:]) ** 2).mean(-1), next_mask)   # Eq. 4
            residual_term = mcfg['beta'] * residual_loss
            state_term = mcfg['lambda_s'] * state_loss
            loss = action_loss - critic_term + residual_term + state_term                  # Eq. 5

            if u == 0:
                norms = grad_stats({'imitation': action_loss, 'critic': critic_term, 'residual': residual_term,
                                    'state': state_term}, params)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, mcfg['grad_clip'])
            optimizer.step()
            scheduler.step()
            if ema_pairs:
                with torch.no_grad():
                    for e, v in ema_pairs:
                        e.mul_(ema_decay).add_(v, alpha=1 - ema_decay)
            with torch.no_grad():
                drift = masked_mean((action_preds - torch.tanh(base)).abs().mean(-1), mask)
                residual_abs = masked_mean(delta.abs().mean(-1), mask)
            for k, v in [('action_loss', action_loss), ('q_value', q_value), ('critic_term', critic_term.abs()),
                         ('residual_loss', residual_loss), ('state_loss', state_loss), ('residual_abs', residual_abs),
                         ('drift', drift)]:
                logs[k].append(v.detach())

        row = {'iter': it, 'num_updates': (it + 1) * mcfg['num_updates_per_iter'],
               'lr': scheduler.get_last_lr()[0], 'seconds': round(time.time() - start_time, 1),
               'lambda_Q_const': lambda_q_const if lambda_q_const is not None else ''}
        row.update({k: float(torch.stack(v).mean()) for k, v in logs.items()})
        row.update(norms)
        log_rows.append(row)
        _write_log(log_path, log_rows)
        save_torch({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'iter': it, 'cfg': tkeys, 'rng': rng_state(),
                    **({'ema': ema} if ema else {})}, last_path)
        if (it + 1) % 10 == 0 or it + 1 == mcfg['max_train_iters']:
            print(f'  [{setup.name} | seed {seed}] iter {it + 1}/{mcfg["max_train_iters"]}: action '
                  f'{row["action_loss"]:.4f} | state {row["state_loss"]:.5f} | critic term {row["critic_term"]:.3f} '
                  f'| {row["seconds"]:.0f}s', flush=True)

    last = log_rows[-1]
    ratio = float(last['g_critic']) / max(float(last['g_imitation']), 1e-12)
    print(f'[{setup.name} | seed {seed}] trained: action {float(last["action_loss"]):.4f} | grad critic / imitation '
          f'{ratio:.2f} | cos {float(last["cos_imitation_critic"]):.2f}', flush=True)
    save_torch({'model': model.state_dict(), 'cfg': tkeys, 'seed': seed, **({'model_ema': ema} if ema else {}),
                'fingerprint': setup.fingerprint, 'lambda_Q_const': lambda_q_const},
               os.path.join(out_dir, 'model.pt'))


def get_model(setup, cfg, seed, weights='raw'):
    """The trained model (trained if missing) with weights 'raw' (the last iterate) or 'ema' (their average)."""
    path = os.path.join(model_dir(setup, cfg, seed), 'model.pt')
    if not os.path.exists(path):
        train(setup, cfg, seed)
    ckpt = load_torch(path)
    mcfg = model_cfg(setup, cfg)
    check_same(ckpt['cfg'], train_keys(mcfg), path)
    if ckpt['fingerprint'] != setup.fingerprint:
        from .common import CacheMismatch
        raise CacheMismatch(f'{path} was trained on other data')
    model = build_model(mcfg, setup.state_dim, setup.act_dim, setup.meta['max_timestep'], device)
    model.load_state_dict(ckpt['model_ema'] if weights == 'ema' else ckpt['model'])
    model.eval()
    model.file_id = file_id(path)
    model.path = path
    return model
