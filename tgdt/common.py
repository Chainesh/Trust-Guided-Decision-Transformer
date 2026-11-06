"""Device, seeding, atomic file writes and file hashes shared by every module."""
import hashlib
import json
import os
import pickle
import random

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CacheMismatch(ValueError):
    """A cached file does not match what is requested (configuration, data or model): it is not used."""


def save_torch(obj, path):
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)


def load_torch(path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:                                  # torch < 1.13 has no weights_only
        return torch.load(path, map_location=device)


def save_pickle(obj, path):
    with open(path + '.tmp', 'wb') as f:
        pickle.dump(obj, f)
    os.replace(path + '.tmp', path)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(obj, path):
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + '.tmp', path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


_file_ids = {}


def file_id(path):
    """Hash of a file's bytes (cached while the file is unchanged): identifies the exact model an evaluation used."""
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _file_ids:
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        _file_ids[key] = h.hexdigest()
    return _file_ids[key]


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state):                        # the states must be CPU tensors (load_torch maps to the GPU)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def domain(name):
    for prefix, dom in [('maze2d', 'maze2d'), ('antmaze', 'antmaze'), ('kitchen', 'kitchen'),
                        ('pen', 'adroit'), ('hammer', 'adroit'), ('door', 'adroit')]:
        if name.startswith(prefix):
            return dom
    return 'locomotion'
