"""Compatibility entry point; Stage 1 and Stage 2 live in resnet_representation.py."""
import argparse
from pathlib import Path
import yaml

from experiments.resnet_representation import RepresentationBank, ResNetStage2

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/slice_belief_imagenette_fixed1000.yaml')
    parser.add_argument('--features', default='results/resnet_representation_fixed1000_matched_audited/features')
    parser.add_argument('--output', default='results/resnet_representation_fixed1000_stage2')
    args = parser.parse_args()
    ResNetStage2(yaml.safe_load(Path(args.config).read_text()), args.features, args.output).run()


if __name__ == '__main__':
    main()
