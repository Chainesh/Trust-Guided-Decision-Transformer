"""YAML configurations.

A run is described by one YAML file (configs/datasets/<dataset>.yaml). It is merged, in order, over configs/base.yaml,
then over every file in its `include:` list (paths relative to configs/), then over its own keys, then over
extra overlays and command-line overrides (`--set train.lambda_Q=25`). The nested YAML is flattened into one dict whose key names are the
ones the code uses (KEYMAP), so a configuration can be compared key by key with the one stored in a checkpoint.
"""
import copy
import hashlib
import json
import os

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')

# (section, key in YAML) -> key in the flat configuration
KEYMAP = {
    ('model', 'arch'): 'arch',                      # 'gpt2': the original DT's GPT-2 block; 'mindt': min-DT's block
    ('model', 'context_len'): 'context_len',
    ('model', 'n_blocks'): 'n_blocks',
    ('model', 'embed_dim'): 'embed_dim',
    ('model', 'n_heads'): 'n_heads',
    ('model', 'dropout'): 'dropout_p',
    ('model', 'activation'): 'activation',          # MLP activation of the gpt2 block ('relu' as in DT, or 'gelu')
    ('model', 'use_residual'): 'use_residual',      # Eq. 2's residual f_psi
    ('model', 'delta_max'): 'delta_max',
    ('train', 'batch_size'): 'batch_size',
    ('train', 'lr'): 'lr',
    ('train', 'weight_decay'): 'wt_decay',
    ('train', 'warmup_steps'): 'warmup_steps',
    ('train', 'lr_schedule'): 'lr_schedule',        # 'constant' (min-DT: warmup then constant) or 'cosine'
    ('train', 'iters'): 'max_train_iters',
    ('train', 'updates_per_iter'): 'num_updates_per_iter',
    ('train', 'grad_clip'): 'grad_clip',
    ('train', 'beta'): 'beta',                      # Eq. 3: weight of ||delta||^2
    ('train', 'lambda_s'): 'lambda_s',              # Eq. 5: weight of the state loss
    ('train', 'lambda_Q'): 'lambda_Q',              # Eq. 3; with q_norm 'once' it is c in lambda_Q = c / mean|Q|
    ('train', 'q_norm'): 'q_norm',                  # 'none': lambda_Q used as given; 'once': c / mean|Q| on the data
    ('train', 'adv_weights'): 'adv_weights',        # Eq. 3's optional weight w_t
    ('train', 'adv_beta'): 'adv_beta',
    ('train', 'loss_reduction'): 'loss_reduction',  # 'sum': squared norms summed over action dimensions (Eq. 3)
    ('train', 'sample_by_length'): 'sample_by_length',
    ('train', 'ema_decay'): 'ema_decay',            # null: no weight average; else kept beside training
    ('critic', 'steps'): 'iql_steps',
    ('critic', 'expectile'): 'iql_expectile',
    ('critic', 'reward_shift'): 'reward_shift',
    ('critic', 'reward_norm'): 'iql_reward_norm',
    ('critic', 'gamma'): 'gamma',
    ('critic', 'batch'): 'iql_batch',
    ('critic', 'lr'): 'iql_lr',
    ('critic', 'polyak'): 'iql_polyak',
    ('critic', 'layers'): 'critic_layers',
    ('critic', 'layer_norm'): 'critic_ln',
    ('data', 'protocol'): 'protocol',               # 'episodes' or 'stream' (maze2d)
    ('returns', 'rule'): 'rtg_rule',                # 'dt': DT's targets and scale 1000 on locomotion
    ('returns', 'target'): 'target',                # null: the rule's; a number; 'x<F>': F x ref max; 'data max'
    ('eval', 'episodes'): 'num_eval_ep',
    ('eval', 'Ke'): 'Ke',
    ('eval', 'alpha'): 'alpha',
    ('eval', 'tau_mode'): 'tau_mode',
    ('eval', 'suffixes'): 'suffixes',
    ('eval', 'selection'): 'selection',             # 'paper' (Eq. 11), 'critic only', 'full context', 'hard reset'
    ('eval', 'hard_reset_cooldown'): 'hard_reset_cooldown',
    ('eval', 'weights'): 'eval_weights',            # 'raw' or 'ema'
}
TOP_KEYS = {'dataset', 'seeds', 'include', 'name'}
SECTIONS = {s for s, _ in KEYMAP}
# settings that change how a trained model is evaluated, not the model itself
EVAL_KEYS = {'num_eval_ep', 'Ke', 'alpha', 'tau_mode', 'suffixes', 'selection', 'hard_reset_cooldown', 'eval_weights',
             'target'}
CRITIC_KEYS = ('iql_steps', 'iql_expectile', 'reward_shift', 'iql_reward_norm', 'iql_batch', 'iql_lr', 'iql_polyak',
               'gamma', 'critic_layers', 'critic_ln')
SELECTIONS = ('paper', 'critic only', 'full context', 'hard reset')


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _read(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _set(tree, dotted, value):
    keys = dotted.split('.')
    node = tree
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load(path, overrides=(), includes=(), config_dir=None):
    """The flat configuration of a YAML file: base.yaml, its includes, the file, extra overlay files, then
    'section.key=value' overrides."""
    config_dir = config_dir or CONFIG_DIR
    if not os.path.exists(path) and os.path.exists(os.path.join(config_dir, 'datasets', path + '.yaml')):
        path = os.path.join(config_dir, 'datasets', path + '.yaml')       # a dataset name is enough
    own = _read(path)
    tree = _read(os.path.join(config_dir, 'base.yaml'))
    for inc in own.get('include', []) or []:
        tree = _merge(tree, _read(os.path.join(config_dir, inc)))
    tree = _merge(tree, {k: v for k, v in own.items() if k != 'include'})
    for inc in includes:
        tree = _merge(tree, _read(inc if os.path.exists(inc) else os.path.join(config_dir, inc)))
    for item in overrides:
        key, _, value = item.partition('=')
        _set(tree, key.strip(), yaml.safe_load(value))
    tree.setdefault('name', os.path.splitext(os.path.basename(path))[0])
    return flatten(tree)


def flatten(tree):
    unknown = set(tree) - TOP_KEYS - SECTIONS
    if unknown:
        raise KeyError(f'unknown top-level keys {sorted(unknown)}')
    if 'dataset' not in tree:
        raise KeyError('a configuration names its dataset (dataset: <D4RL name>)')
    cfg = {k: tree[k] for k in TOP_KEYS - {'include'} if k in tree}
    for section in SECTIONS:
        for k, v in (tree.get(section) or {}).items():
            if (section, k) not in KEYMAP:
                raise KeyError(f'unknown key {section}.{k}')
            cfg[KEYMAP[(section, k)]] = v
    missing = sorted(set(KEYMAP.values()) - set(cfg))
    if missing:
        raise KeyError(f'missing keys {missing} (base.yaml defines every key)')
    return validate(cfg)


def validate(cfg):
    if cfg['q_norm'] in (None, False, 'none'):
        cfg['q_norm'] = False
    assert cfg['q_norm'] in (False, 'once'), cfg['q_norm']
    assert cfg['arch'] in ('gpt2', 'mindt'), cfg['arch']
    assert cfg['selection'] in SELECTIONS, cfg['selection']
    assert cfg['eval_weights'] in ('raw', 'ema'), cfg['eval_weights']
    assert cfg['eval_weights'] == 'raw' or cfg['ema_decay'], 'eval.weights ema needs train.ema_decay'
    assert cfg['loss_reduction'] in ('sum', 'mean')
    assert cfg['protocol'] in ('episodes', 'stream')
    assert cfg['tau_mode'] in ('shared', 'per_suffix'), cfg['tau_mode']
    assert cfg['rtg_rule'] in ('dt', 'ref_log10'), cfg['rtg_rule']
    cfg['seeds'] = [int(s) for s in cfg['seeds']]
    for k in ('lr', 'wt_decay', 'lambda_Q', 'beta', 'lambda_s', 'delta_max', 'gamma', 'iql_expectile', 'iql_lr',
              'iql_polyak', 'reward_shift', 'alpha', 'adv_beta', 'dropout_p', 'grad_clip'):
        cfg[k] = float(cfg[k])
    cfg['suffixes'] = sorted(int(L) for L in cfg['suffixes'])
    assert max(cfg['suffixes']) <= cfg['context_len'], 'a suffix longer than the context'
    return cfg


def train_keys(cfg):
    """The settings a trained model depends on (everything but evaluation settings, seeds and names)."""
    return {k: v for k, v in cfg.items() if k not in EVAL_KEYS | TOP_KEYS}


def train_hash(cfg):
    return hashlib.sha1(json.dumps(train_keys(cfg), sort_keys=True, default=str).encode()).hexdigest()[:8]


def critic_settings(cfg):
    return {k: cfg[k] for k in CRITIC_KEYS}


def check_same(stored, requested, where, keys=None):
    """A cached file is used only if it was made with the requested settings."""
    keys = set(requested) if keys is None else set(keys)
    diffs = {k: (stored.get(k, '<absent>'), requested[k]) for k in keys if stored.get(k, '<absent>') != requested[k]}
    if diffs:
        from .common import CacheMismatch
        raise CacheMismatch(f'{where} was made with other settings: {diffs}')
