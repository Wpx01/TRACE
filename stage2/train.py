"""Run from repository root: python -m stage2.train --help."""

from stage1.common import load_generator, run_training, seed_all, training_parser
from .model import Stage2Model


def main(argv=None):
    parser = training_parser(2)
    args = parser.parse_args(argv)
    if args.ngf < 1:
        parser.error('ngf must be positive.')
    if not args.resume and (not args.teacher or not args.student):
        parser.error('Both --teacher and --student stage-1 checkpoints are required.')
    if args.resume and (args.teacher or args.student):
        parser.error('Use either --resume or the two stage-1 checkpoints, not both.')
    seed_all(args.seed)
    model = Stage2Model(ngf=args.ngf)
    if not args.resume:
        load_generator(model.G_A, args.teacher, 'G_A')
        load_generator(model.G_B, args.student, 'G_B')
    run_training(model, args, stage=2)


if __name__ == '__main__':
    main()
