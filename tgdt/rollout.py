"""Trust-filtered execution (Eqs. 8-13), one episode per environment, in lockstep: each step needs one model call for
all episodes and all suffixes.

At step t every suffix L (the last L steps of the history) proposes an action. A suffix is trusted if the rolling mean
of its last Ke next-state errors is at most tau (Eq. 10). Selection rules:
  paper:        the critic ranks the trusted suffixes (Eq. 11); none trusted: L = 1
  critic only:  the critic ranks all suffixes, none rejected (no filter)
  full context: L = K always (no filter, no critic: DT's execution)
  hard reset:   L = K, but when the executed context's rolling score exceeds tau the history is discarded: the next
                decision uses L = 1, the context grows back (up to K), its rolling score restarts after the reset, and
                no new reset happens within `hard_reset_cooldown` steps.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .common import device
from .envs import reset_env, step_env

LOG_KEYS = ('L', 'rejected', 'filter_changed', 'not_critic', 'fallback', 'viol', 'reward', 'S', 'q', 'u', 'trusted')


def select(rule, q, trust):
    """Index of the executed suffix (suffixes ordered longest first: index 0 is L = K, the last is L = 1) and whether
    L = 1 was forced. q: min of the twin critics for each suffix's action; trust: Eq. 10."""
    fallback = torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)
    if rule == 'paper':
        fallback = ~trust.any(1)
        choice = q.masked_fill(~trust, -np.inf).argmax(1)
    elif rule == 'critic only':
        choice = q.argmax(1)
    elif rule == 'full context':
        choice = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
    else:
        raise ValueError(rule)
    choice = torch.where(fallback, torch.full_like(choice, q.shape[1] - 1), choice)
    return choice, fallback


@torch.no_grad()
def run_episodes(setup, model, critic, envs, taus, seeds, target, cfg):
    rule = cfg['selection']
    suffixes = sorted(cfg['suffixes'], reverse=True)       # argmax ties resolve to the longest suffix
    reset_mode = rule == 'hard reset'
    cooldown = cfg['hard_reset_cooldown']
    n, n_L, Ke, K = len(envs), len(suffixes), cfg['Ke'], cfg['context_len']
    rtg_scale, max_ep_len, state_dim, act_dim = model.cfg['rtg_scale'], setup.max_ep_len, setup.state_dim, setup.act_dim
    mean_t, std_t, scale_t = [torch.as_tensor(x, device=device) for x in (setup.state_mean, setup.state_std,
                                                                           setup.act_scale)]
    tau = torch.tensor([taus[L] for L in suffixes], device=device)

    states = torch.zeros((n, max_ep_len, state_dim), device=device)
    actions = torch.zeros((n, max_ep_len, act_dim), device=device)
    returns_to_go = torch.zeros((n, max_ep_len, 1), device=device)
    timesteps = torch.arange(max_ep_len, device=device).repeat(n, 1)
    errors = torch.zeros((n, n_L, Ke), device=device)      # each suffix's last Ke errors (Eq. 6), a ring buffer
    reset_at = np.zeros(n, dtype=np.int64)                  # hard reset: the step the current context starts at
    own_errors = torch.zeros((n, Ke), device=device)        # hard reset: the executed context's errors since its reset
    n_own = np.zeros(n, dtype=np.int64)
    logs = [{k: [] for k in LOG_KEYS} for _ in range(n)]

    def normalize(obs):
        return (torch.as_tensor(np.array(obs), dtype=torch.float32, device=device) - mean_t) / std_t

    obs = [reset_env(env, seed) for env, seed in zip(envs, seeds)]
    rtg = [target / rtg_scale] * n
    active = list(range(n))

    for t in range(max_ep_len):
        idx = torch.as_tensor(active, device=device)
        rows = torch.arange(len(active), device=device)
        raw = normalize([obs[i] for i in active])
        states[idx, t] = raw
        returns_to_go[idx, t, 0] = torch.as_tensor([rtg[i] for i in active], dtype=torch.float32, device=device)
        starts = [max(0, t - L + 1) for L in suffixes]

        def predict(head):            # prediction at step t from each suffix's context, in one right-padded batch
            first = sorted(set(starts))
            T = t + 1 - first[0]
            context = [torch.cat([F.pad(x[idx, s0:t + 1], (0, 0) * (x.dim() - 2) + (0, s0 - first[0]))
                                  for s0 in first]) for x in (timesteps, states, actions, returns_to_go)]
            out = model(*context)[head].view(len(first), len(active), T, -1)
            at_t = {s0: out[j, :, t - s0] for j, s0 in enumerate(first)}
            return torch.stack([at_t[s0] for s0 in starts], 1)            # (active episodes, suffixes, dim)

        def predict_own(head):        # prediction at step t from each episode's own context [own_start, t]
            T = t + 1 - min(own_starts)
            context = [torch.stack([F.pad(x[i, s0:t + 1], (0, 0) * (x.dim() - 2) + (0, T - (t + 1 - s0)))
                                    for i, s0 in zip(active, own_starts)])
                       for x in (timesteps, states, actions, returns_to_go)]
            return model(*context)[head][rows, torch.as_tensor([t - s0 for s0 in own_starts], device=device)]

        candidates = predict(1)
        q1, q2 = critic.both(raw.unsqueeze(1).expand(-1, n_L, -1), candidates)
        q, u = torch.min(q1, q2), (q1 - q2).abs()
        filled = min(t, Ke)
        trust = (errors[idx].sum(-1) / filled <= tau) if filled else torch.ones_like(q, dtype=torch.bool)   # Eq. 10
        favourite = q.argmax(1)
        if reset_mode:
            own_starts = [max(int(reset_at[i]), t - K + 1) for i in active]
            action = predict_own(1)
            L_exec = torch.as_tensor([K if reset_at[i] == 0 else t - s0 + 1 for i, s0 in zip(active, own_starts)],
                                     device=device)
            fallback = torch.as_tensor([bool(t > 0 and reset_at[i] == t) for i in active], device=device)
        else:
            choice, fallback = select(rule, q, trust)
            action = candidates[rows, choice]
            L_exec = torch.as_tensor(suffixes, device=device)[choice]
        actions[idx, t] = action
        steps = [step_env(envs[i], a) for i, a in zip(active, (action * scale_t).cpu().numpy())]

        # reliability errors of each suffix on the realized transitions (Eqs. 12-13)
        next_states = normalize([s for s, _, _ in steps])
        err = ((predict(0) - next_states.unsqueeze(1)) ** 2).mean(-1)
        errors[idx, :, t % Ke] = err
        S = errors[idx].sum(-1) / min(t + 1, Ke)
        if reset_mode:
            pos = torch.as_tensor(n_own[active] % Ke, device=device)
            own_errors[idx, pos] = ((predict_own(0) - next_states) ** 2).mean(-1)
            n_own[active] += 1
            count = torch.as_tensor(np.minimum(n_own[active], Ke), dtype=torch.float32, device=device)
            viol = own_errors[idx].sum(-1) / count > tau[0]
            fire = viol.cpu().numpy() & (t + 1 - reset_at[active] >= cooldown)
            for j in np.nonzero(fire)[0]:
                i = active[j]
                reset_at[i], n_own[i] = t + 1, 0
                own_errors[i] = 0.0
        else:
            viol = S[rows, choice] > tau[choice]

        not_critic = L_exec != torch.as_tensor(suffixes, device=device)[favourite]
        per_step = [L_exec, ~trust.all(1), ~trust[rows, favourite], not_critic, fallback, viol, None, S, q, u, trust]
        per_step = [None if x is None else x.cpu().numpy() for x in per_step]
        still = []
        for j, i in enumerate(active):
            state, reward, done = steps[j]
            for k, x in zip(LOG_KEYS, per_step):
                logs[i][k].append(reward if k == 'reward' else x[j])
            obs[i] = state
            rtg[i] -= reward / rtg_scale
            if not done:
                still.append(i)
        active = still
        if not active:
            break

    logs = [{k: np.asarray(v, dtype=np.float32) for k, v in log.items()} for log in logs]
    for log in logs:
        log.update(suffixes=suffixes, rule=rule)
    return logs


def log_metrics(log):
    """Per-episode statistics (in %): steps where some suffix was untrusted ('rejected'); where the critic's favourite
    was untrusted ('filter changed'); where the executed L differs from the critic's favourite; where L = 1 was
    forced or the context reset ('fallback'); where the executed context's rolling score is above tau ('viol');
    L switches, the mean L and the share of each L; the critic's spread of Q over suffixes and its twin gap."""
    L = log['L']
    out = {'rejected %': 100 * float(np.mean(log['rejected'] > 0)),
           'filter changed %': 100 * float(np.mean(log['filter_changed'] > 0)),
           'not critic choice %': 100 * float(np.mean(log['not_critic'] > 0)),
           'fallback %': 100 * float(np.mean(log['fallback'] > 0)), 'viol %': 100 * float(np.mean(log['viol'] > 0)),
           'switches %': 100 * float(np.mean(L[1:] != L[:-1])) if len(L) > 1 else 0.0, 'mean L': float(L.mean()),
           'Q spread': float(np.mean(log['q'].max(1) - log['q'].min(1))), 'twin gap': float(np.mean(log['u']))}
    out.update({f'L={s} %': 100 * float(np.mean(L == s)) for s in log['suffixes']})
    return out
