"""Run from repository root: python -m stage1.train --help."""

from .common import run_training, seed_all, training_parser
from .model import Stage1Model


def main(argv=None):
    parser = training_parser(1)
    args = parser.parse_args(argv)
    if args.ngf < 1 or args.ndf < 1 or args.pool_size < 0:
        parser.error('Network widths must be positive; pool size must be non-negative.')
    seed_all(args.seed)
    model = Stage1Model(ngf=args.ngf, ndf=args.ndf, pool_size=args.pool_size)
    run_training(model, args, stage=1)


if __name__ == '__main__':
    main()
