"""Environment access (old and new gym APIs) and D4RL's normalized score."""
import contextlib
import io

import numpy as np


def _gym():
    import gym
    try:
        import d4rl  # noqa: F401  (registers the D4RL environments)
    except ImportError:
        pass
    return gym


def make_env(name):
    with contextlib.redirect_stdout(io.StringIO()):             # some D4RL environments print their goal when made
        return _gym().make(name)


def reset_env(env, seed):
    try:
        env.seed(seed)
        state = env.reset()
    except (AttributeError, TypeError):
        state = env.reset(seed=seed)
    return state[0] if isinstance(state, tuple) else state


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        state, reward, terminated, truncated, _ = out
        return state, float(reward), terminated or truncated
    state, reward, done, _ = out
    return state, float(reward), done


def close_env(env):                 # best effort: frees the simulator; a failure to close changes no result
    try:
        env.close()
    except Exception:
        pass


def d4rl_score(env, total_reward):
    return 100.0 * env.unwrapped.get_normalized_score(total_reward)


def env_goal(env):
    """The goal of a goal-reaching environment (antmaze: drawn when the environment is made; maze2d: fixed), or None."""
    u = env.unwrapped
    for attr in ('target_goal', '_target', 'goal'):
        g = getattr(u, attr, None)
        if g is not None:
            return [float(x) for x in np.asarray(g).reshape(-1)[:2]]
    return None
