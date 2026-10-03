import os
from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import snapshot_download


# Download the MMBench2 dataset from the Hugging Face Hub.
# Authenticate via the HF_TOKEN env var or a cached `hf auth login` if required.
DEFAULT_REPO_ID = "nicklashansen/mmbench2"


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--local_dir", type=str, default="./data")
    p.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID,
                   help=f"HF dataset repo id (default: {DEFAULT_REPO_ID}).")
    p.add_argument("--subset", type=str, nargs="*", default=None,
                   help="optional subset(s) to download, e.g. 'val' or 'expert'")
    p.add_argument("--tasks", type=str, nargs="+", default=None,
                   help="optional task names; downloads each task's .pt and PNG strips")
    args = p.parse_args()

    allow_patterns = None
    if args.tasks:
        if not args.subset:
            p.error("--tasks requires at least one --subset")
        allow_patterns = [pattern
                          for subset in args.subset
                          for task in args.tasks
                          for pattern in (f"{subset.rstrip('/')}/{task}.pt",
                                          f"{subset.rstrip('/')}/{task}-*.png")]
    elif args.subset:
        allow_patterns = [f"{s.rstrip('/')}/*" for s in args.subset]

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.local_dir,
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=allow_patterns,
    )
    if args.tasks:
        root = Path(args.local_dir)
        missing = []
        for subset in args.subset:
            directory = root / subset.rstrip('/')
            for task in args.tasks:
                if not (directory / f'{task}.pt').is_file():
                    missing.append(f'{subset}/{task}.pt')
                if not (directory / f'{task}-0.png').is_file():
                    missing.append(f'{subset}/{task}-0.png')
        if missing:
            raise SystemExit(f'requested task files missing from dataset: {missing}')
