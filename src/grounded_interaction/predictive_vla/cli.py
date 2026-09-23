"""Small CLI for the current implementation; no external component factory."""

import argparse
import importlib.util
import json
import sys

from .config import load_config, set_manual_seed


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ip-vla")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "check-data", "train"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="experiments/predictive_vla.yaml")
        if name != "check":
            command.add_argument(
                "--data", required=True, help="existing trajectory manifest JSON"
            )
        if name == "train":
            command.add_argument("--output", required=True)
            command.add_argument("--epochs", type=int, default=1)
            command.add_argument("--device", default="cuda")
            command.add_argument("--model-path")
            command.add_argument("--max-updates", type=int)
            command.add_argument("--warmup-updates", type=int, default=200)
            command.add_argument("--eval-every", type=int, default=100)
            command.add_argument("--validation-examples", type=int, default=64)
            command.add_argument("--resume")
            command.add_argument("--experiment-log")
            command.add_argument(
                "--data-audit", help="completed local full-data audit report"
            )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "check":
            report = {
                "manual_seed": config.manual_seed,
                "dependencies": {
                    name: importlib.util.find_spec(name) is not None
                    for name in ("torch", "numpy", "transformers", "libero")
                },
                "real_model_verified": False,
                "closed_loop_evaluated": False,
            }
        else:
            from .data import TrajectoryDataset

            dataset = TrajectoryDataset(args.data, config)
            if args.command == "check-data":
                summary = dataset.validate()
                report = {"manual_seed": config.manual_seed, **summary}
            else:
                from .backend import NativeVLABackend
                from .training import train

                if config.prediction_weight and args.resume is None:
                    raise ValueError(
                        "new prediction-enabled runs require scripts/run_stage2.sh "
                        "and its explicit stage-one best warm-start checks"
                    )
                if args.epochs < 1 or not {"train", "validation"}.issubset(
                    {entry["split"] for entry in dataset.entries}
                ):
                    raise ValueError(
                        "positive epochs and disjoint train/validation episodes are required"
                    )
                data_summary = (
                    dataset.read_validation_report(args.data_audit)
                    if args.data_audit
                    else None
                )
                set_manual_seed(config.manual_seed)  # before adapter initialization
                backend = NativeVLABackend.from_pretrained(
                    config, device=args.device, local_path=args.model_path
                )
                report = train(
                    backend,
                    dataset,
                    output=args.output,
                    epochs=args.epochs,
                    max_updates=args.max_updates,
                    warmup_updates=args.warmup_updates,
                    eval_every=args.eval_every,
                    validation_examples=args.validation_examples,
                    resume=args.resume,
                    experiment_log=args.experiment_log,
                    data_summary=data_summary,
                )
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (ImportError, ValueError, TypeError, RuntimeError, OSError) as error:
        print(f"ip-vla: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
