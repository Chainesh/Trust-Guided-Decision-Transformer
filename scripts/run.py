#!/usr/bin/env python
"""Train (if needed) and evaluate TGDT on one dataset, several, or all of them.

  python scripts/run.py hopper-medium-v2                       # its config's seeds, Eq. 11
  python scripts/run.py hopper-medium-v2 --seeds 107           # one seed
  python scripts/run.py all                                    # every configs/datasets/*.yaml, in order
  python scripts/run.py pen-human-v1 --modes paper "critic only" "full context" "hard reset"
  python scripts/run.py pen-human-v1 --set train.lambda_Q=2.5 --set model.arch=mindt

Everything is cached under --runs (critics, models, evaluations): a second call only reads results. A changed
training setting gets its own model folder (named by a hash of the settings), so nothing is overwritten.
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tgdt import config, Setup, evaluate, mean_se          # noqa: E402

ORDER = ['halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2', 'halfcheetah-medium-replay-v2',
         'hopper-medium-replay-v2', 'walker2d-medium-replay-v2', 'halfcheetah-medium-expert-v2',
         'hopper-medium-expert-v2', 'walker2d-medium-expert-v2', 'maze2d-umaze-v1', 'maze2d-medium-v1',
         'antmaze-umaze-v0', 'antmaze-umaze-diverse-v0', 'antmaze-medium-diverse-v0', 'pen-human-v1',
         'hammer-human-v1', 'door-human-v1', 'pen-cloned-v1', 'hammer-cloned-v1', 'door-cloned-v1']


def dataset_configs(names):
    if names == ['all']:
        found = {os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(config.CONFIG_DIR, 'datasets', '*.yaml'))}
        return [n for n in ORDER if n in found] + sorted(found - set(ORDER))
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('datasets', nargs='+', help="dataset names (configs/datasets/<name>.yaml), YAML paths, or 'all'")
    ap.add_argument('--seeds', type=int, nargs='+', help="training seeds (default: the config's)")
    ap.add_argument('--modes', nargs='+', default=['paper'], choices=list(config.SELECTIONS),
                    help='execution rules to evaluate (each on the same trained model)')
    ap.add_argument('--set', dest='overrides', action='append', default=[], metavar='SECTION.KEY=VALUE')
    ap.add_argument('--include', action='append', default=[], help='extra YAML overlay files')
    ap.add_argument('--runs', default='./runs/', help='cache of critics, models and evaluations')
    ap.add_argument('--data', default='./dataset/', help='folder of the min-DT .pkl datasets (built if missing)')
    ap.add_argument('--force-eval', action='store_true', help='evaluate again even if a valid result is cached')
    ap.add_argument('--save-logs', action='store_true', help='also save per-step rollout logs')
    args = ap.parse_args()

    summary = []
    for name in dataset_configs(args.datasets):
        cfg0 = config.load(name, args.overrides, args.include)
        seeds = args.seeds or cfg0['seeds']
        print(f'\n=== {cfg0["dataset"]} ({cfg0["name"]}): arch {cfg0["arch"]}, seeds {seeds}, modes {args.modes}',
              flush=True)
        t0 = time.time()
        setup = Setup(cfg0['dataset'], cfg0['protocol'], args.data, args.runs)
        for mode in args.modes:
            cfg = {**cfg0, 'selection': mode}
            scores = [evaluate(setup, cfg, s, force=args.force_eval, save_logs=args.save_logs)['summary']['score']
                      for s in seeds]
            m, se = mean_se(scores)
            summary.append((cfg0['dataset'], mode, m, se, scores))
            print(f'>>> {cfg0["dataset"]} | {mode}: {m:.1f} +- {se:.1f}  (seeds {seeds}: '
                  f'{", ".join(f"{x:.1f}" for x in scores)})  [{time.time() - t0:.0f}s]', flush=True)

    print('\n' + '\n'.join(f'{d:32s} {mode:13s} {m:7.1f} +- {se:4.1f}' for d, mode, m, se, _ in summary))


if __name__ == '__main__':
    main()
