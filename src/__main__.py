"""Commands for reviewing and reproducing the frozen release from a checkout."""

import argparse


def main():
    parser = argparse.ArgumentParser(prog="flag-egma", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Check source, checkpoints, and archived scores by checksum")
    reproduction = commands.add_parser("reproduce", help="Run the original scorer and save a separate copy")
    reproduction.add_argument("--output", required=True, help="New directory for the generated ZIP and score files")
    validation = commands.add_parser("validate", help="Run the original selected English holdout validation")
    validation.add_argument("--output", required=True, help="New directory for the validation run")
    args = parser.parse_args()
    if args.command == "verify":
        from .release import verify

        verify()
    else:
        from .reproduce import run_frozen

        run_frozen(args.output, validation=args.command == "validate")


if __name__ == "__main__":
    main()
