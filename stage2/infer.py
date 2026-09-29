"""Final TRACE inference uses only stage-2 G_B; no masks or teacher required."""

from stage1.infer import main


if __name__ == '__main__':
    main(stage=2)
