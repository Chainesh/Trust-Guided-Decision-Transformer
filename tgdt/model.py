"""The TGDT model: a Decision Transformer over (R_t, s_t, a_t) tokens with
  - a base action head and a bounded residual (Eq. 2): a_hat = tanh(a_base + delta), delta = delta_max tanh(f(z)),
    both read from the state token z_t (which has not seen a_t);
  - a next-state head g (Eq. 4) read from the action token (which has seen the executed a_t).

Two transformer blocks:
  gpt2   the block of the original Decision Transformer (Chen et al. 2021: HuggingFace GPT-2 without positional
         embeddings, as in its trajectory_gpt2.py), which later DT papers reuse unchanged (Online DT, QT, ...):
         pre-LayerNorm, MLP 4x with ReLU (DT's gym default), dropout on the embeddings, attention weights and residual
         branches, a final LayerNorm, weights initialized N(0, 0.02) with zero biases.
  mindt  the block of min-decision-transformer (post-LayerNorm, GELU, PyTorch default initialization).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- min-DT block -------------------------------------------------------------------------------------------------
class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.n_heads = n_heads
        self.max_T = max_T
        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)
        self.proj_net = nn.Linear(h_dim, h_dim)
        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)
        self.register_buffer('mask', torch.tril(torch.ones((max_T, max_T))).view(1, 1, max_T, max_T))

    def forward(self, x):
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))
        attention = self.att_drop(F.softmax(weights, dim=-1) @ v)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)
        return self.proj_drop(self.proj_net(attention))


class MinDTBlock(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(nn.Linear(h_dim, 4 * h_dim), nn.GELU(), nn.Linear(4 * h_dim, h_dim),
                                 nn.Dropout(drop_p))
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        x = self.ln1(x + self.attention(x))
        return self.ln2(x + self.mlp(x))


# ---- GPT-2 block (the original DT's) --------------------------------------------------------------------------------
class GPT2Attention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        assert h_dim % n_heads == 0
        self.n_heads = n_heads
        self.c_attn = nn.Linear(h_dim, 3 * h_dim)          # HF's Conv1D(3 h, h) is a Linear with transposed weight
        self.c_proj = nn.Linear(h_dim, h_dim)
        self.attn_dropout = nn.Dropout(drop_p)
        self.resid_dropout = nn.Dropout(drop_p)
        self.register_buffer('mask', torch.tril(torch.ones((max_T, max_T))).view(1, 1, max_T, max_T))

    def forward(self, x):
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = [y.view(B, T, N, D).transpose(1, 2) for y in (q, k, v)]
        w = q @ k.transpose(2, 3) / math.sqrt(D)
        w = w.masked_fill(self.mask[..., :T, :T] == 0, -1e4)   # HF GPT-2's masked value
        a = self.attn_dropout(F.softmax(w, dim=-1)) @ v
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(a))


class GPT2Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, activation='relu'):
        super().__init__()
        self.ln_1 = nn.LayerNorm(h_dim)
        self.attn = GPT2Attention(h_dim, max_T, n_heads, drop_p)
        self.ln_2 = nn.LayerNorm(h_dim)
        act = {'relu': nn.ReLU(), 'gelu': nn.GELU()}[activation]
        self.mlp = nn.Sequential(nn.Linear(h_dim, 4 * h_dim), act, nn.Linear(4 * h_dim, h_dim), nn.Dropout(drop_p))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


def gpt2_init(module):
    """GPT-2's initialization (trajectory_gpt2.py: _init_weights), applied to the transformer only, as in DT."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


# ---- TGDT -------------------------------------------------------------------------------------------------------------
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p, max_timestep=4096,
                 use_residual=True, delta_max=0.05, arch='gpt2', activation='relu'):
        super().__init__()
        self.state_dim, self.act_dim, self.h_dim = state_dim, act_dim, h_dim
        self.use_residual, self.delta_max, self.arch = use_residual, delta_max, arch
        T = 3 * context_len
        if arch == 'mindt':
            self.transformer = nn.Sequential(*[MinDTBlock(h_dim, T, n_heads, drop_p) for _ in range(n_blocks)])
        elif arch == 'gpt2':
            self.embed_drop = nn.Dropout(drop_p)
            self.transformer = nn.Sequential(*[GPT2Block(h_dim, T, n_heads, drop_p, activation)
                                               for _ in range(n_blocks)])
            self.ln_f = nn.LayerNorm(h_dim)
            self.transformer.apply(gpt2_init)
            self.ln_f.apply(gpt2_init)
        else:
            raise ValueError(arch)

        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = nn.Linear(1, h_dim)
        self.embed_state = nn.Linear(state_dim, h_dim)
        self.embed_action = nn.Linear(act_dim, h_dim)

        self.predict_rtg = nn.Linear(h_dim, 1)                 # return head, as in DT (not used by the loss)
        self.predict_state = nn.Linear(h_dim, state_dim)       # reliability head g (Eq. 4)
        self.predict_action = nn.Linear(h_dim, act_dim)        # base action (Eq. 2)
        self.predict_residual = nn.Sequential(nn.Linear(h_dim, h_dim), nn.GELU(), nn.Linear(h_dim, act_dim))   # f

    def forward(self, timesteps, states, actions, returns_to_go):
        """Returns state predictions (from the action tokens), executed actions tanh(base + delta), delta, and base."""
        B, T, _ = states.shape
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        h = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)   # (R_1, s_1, a_1, R_2, ...)
        h = self.embed_ln(h.permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim))
        if self.arch == 'gpt2':
            h = self.ln_f(self.transformer(self.embed_drop(h)))
        else:
            h = self.transformer(h)
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)     # h[:, 0] return, h[:, 1] state, h[:, 2] action

        state_preds = self.predict_state(h[:, 2])     # s_{t+1} from (..., R_t, s_t, a_t): sees the executed action
        z = h[:, 1]                                   # (..., R_t, s_t): never sees a_t
        base = self.predict_action(z)
        delta = self.delta_max * torch.tanh(self.predict_residual(z)) if self.use_residual else torch.zeros_like(base)
        return state_preds, torch.tanh(base + delta), delta, base


def build_model(cfg, state_dim, act_dim, max_timestep, device):
    model = DecisionTransformer(state_dim=state_dim, act_dim=act_dim, n_blocks=cfg['n_blocks'], h_dim=cfg['embed_dim'],
                                context_len=cfg['context_len'], n_heads=cfg['n_heads'], drop_p=cfg['dropout_p'],
                                max_timestep=max_timestep, use_residual=cfg['use_residual'],
                                delta_max=cfg['delta_max'], arch=cfg['arch'],
                                activation=cfg.get('activation', 'relu')).to(device)
    model.cfg = cfg
    return model
