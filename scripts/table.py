#!/usr/bin/env python
"""The results table, read from the cache (nothing is trained or evaluated): mean +- s.e. over each config's seeds.

  python scripts/table.py                         # every dataset config, Eq. 11
  python scripts/table.py --modes paper "full context" "critic only"
  python scripts/table.py --set model.arch=mindt  # the same overrides run.py was given
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tgdt import config, Setup, evaluate, mean_se          # noqa: E402
from run import dataset_configs                            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('datasets', nargs='*', default=['all'])
    ap.add_argument('--modes', nargs='+', default=['paper'], choices=list(config.SELECTIONS))
    ap.add_argument('--set', dest='overrides', action='append', default=[])
    ap.add_argument('--include', action='append', default=[])
    ap.add_argument('--runs', default='./runs/')
    ap.add_argument('--data', default='./dataset/')
    ap.add_argument('--out', default='./results/')
    args = ap.parse_args()

    rows = []
    for name in dataset_configs(args.datasets):
        cfg0 = config.load(name, args.overrides, args.include)
        setup = Setup(cfg0['dataset'], cfg0['protocol'], args.data, args.runs)
        row = {'dataset': cfg0['dataset'], 'seeds': ' '.join(map(str, cfg0['seeds']))}
        for mode in args.modes:
            res = [evaluate(setup, {**cfg0, 'selection': mode}, s, cache_only=True) for s in cfg0['seeds']]
            scores = [r['summary']['score'] for r in res if r is not None]
            if scores:
                m, se = mean_se(scores)
                row[mode] = f'{m:.1f} ± {se:.1f}' + ('' if len(scores) == len(res) else f' ({len(scores)} seeds)')
                row[mode + ' per seed'] = ' '.join(f'{x:.1f}' for x in scores)
                row['_' + mode] = m
            else:
                row[mode], row[mode + ' per seed'] = '—', ''
        rows.append(row)

    cols = ['dataset'] + args.modes + [m + ' per seed' for m in args.modes] + ['seeds']
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |' for r in rows]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'table.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(args.out, 'table.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print('\n'.join(lines))
    print(f'\nwritten: {os.path.join(args.out, "table.md")}, {os.path.join(args.out, "table.csv")}')


if __name__ == '__main__':
    main()
