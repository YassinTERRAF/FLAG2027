import argparse
import json
import platform
from pathlib import Path

from .common import ROOT, save_json


def main():
    parser = argparse.ArgumentParser(description="Validation-first FLAG 2027 pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download")
    p.add_argument("--sets", nargs="+", choices=["metadata", "train", "dev", "evaluation"], default=["metadata", "train", "evaluation", "dev"])
    p = sub.add_parser("doctor")
    p = sub.add_parser("prepare")
    p.add_argument("--out", default="artifacts/official.npz")
    p = sub.add_parser("extract")
    p.add_argument("--pack", default="artifacts/official.npz")
    p.add_argument("--out", default="artifacts/enriched.npz")
    p.add_argument("--device", default="mps")
    p = sub.add_parser("extract-demographics")
    p.add_argument("--pack", default="artifacts/enriched.npz")
    p.add_argument("--out", default="artifacts/demographic.npz")
    p.add_argument("--device", default="mps")
    p = sub.add_parser("search")
    p.add_argument("--pack", default="artifacts/official.npz")
    p.add_argument("--config", default="configs/search.yaml")
    p.add_argument("--out", default="artifacts/search_official")
    p.add_argument("--device", default="mps")
    p = sub.add_parser("audit")
    p.add_argument("--selection", required=True)
    p.add_argument("--device", default="mps")
    p = sub.add_parser("cca-search")
    p.add_argument("--pack", default="artifacts/official.npz")
    p.add_argument("--out", default="artifacts/cca_official")
    p = sub.add_parser("finalize")
    p.add_argument("--selection", required=True)
    p.add_argument("--out", default="artifacts/final")
    p.add_argument("--device", default="mps")
    p = sub.add_parser("infer")
    p.add_argument("--system", default="artifacts/final/system.json")
    p.add_argument("--data", default="data/evaluation")
    p.add_argument("--out", default="artifacts/submission")
    p.add_argument("--device", default="mps")
    p = sub.add_parser("check-submission")
    p.add_argument("--archive", required=True)
    p.add_argument("--data", default="data/evaluation")
    args = parser.parse_args()
    if args.command == "download":
        from .download import download
        download(args.sets)
    elif args.command == "doctor":
        import torch
        from .common import device_for
        device = device_for("mps")
        x = torch.randn(8, 8, device=device, requires_grad=True)
        (x @ x.T).square().mean().backward()
        report = {"python": platform.python_version(), "architecture": platform.machine(),
                  "torch": torch.__version__, "mps_available": torch.backends.mps.is_available(),
                  "mps_backward_finite": bool(torch.isfinite(x.grad).all().item())}
        save_json(ROOT / "reports/environment.json", report)
        print(json.dumps(report, indent=2))
    elif args.command == "prepare":
        from .data import prepare_official
        prepare_official(args.out)
    elif args.command == "extract":
        from .encoders import extract_training
        extract_training(args.pack, args.out, args.device)
    elif args.command == "extract-demographics":
        from .demographics import extract_demographic_training
        extract_demographic_training(args.pack, args.out, args.device)
    elif args.command == "search":
        from .train import search
        search(args.pack, args.config, args.out, args.device)
    elif args.command == "audit":
        from .train import audit
        audit(args.selection, args.device)
    elif args.command == "cca-search":
        from .cca import cca_search
        cca_search(args.pack, args.out)
    elif args.command == "finalize":
        from .train import finalize
        finalize(args.selection, args.out, args.device)
    elif args.command == "infer":
        from .infer import infer
        infer(args.system, args.data, args.out, args.device)
    elif args.command == "check-submission":
        from .infer import validate_archive
        validate_archive(args.archive, args.data)
        print("Submission structure, finite scores and exact key coverage passed")


if __name__ == "__main__":
    main()
