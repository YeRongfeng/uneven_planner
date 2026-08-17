#!/usr/bin/env python3
"""Register downloaded LAS/LAZ mother maps in the shared dataset layout.

The command copies source files into
``dataset/external/<domain>/<domain>_<site>/raw`` and prints the directory that
should be passed to the manual review server.  It never overwrites an existing
file.
"""

import argparse
import json
import shutil
from pathlib import Path


def workspace_root():
    return Path(__file__).resolve().parents[4]


def source_files(paths):
    files = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(
                candidate for candidate in path.rglob("*")
                if candidate.is_file())
        else:
            raise FileNotFoundError(path)
        for candidate in candidates:
            if candidate.suffix.lower() in {".las", ".laz"}:
                files.append(candidate)
    if not files:
        raise FileNotFoundError("No .las or .laz files found in the inputs")
    return files


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="terrain/domain name, for example forest")
    parser.add_argument("site", help="source site name, for example wa21")
    parser.add_argument(
        "sources", nargs="+", type=Path,
        help="downloaded .las/.laz files or directories containing them")
    parser.add_argument(
        "--external-root", type=Path,
        help="root for registered sources (default: dataset/external)")
    return parser.parse_args()


def main():
    args = parse_args()
    external_root = (
        args.external_root.expanduser().resolve()
        if args.external_root else workspace_root() / "dataset" / "external")
    destination = external_root / args.domain / f"{args.domain}_{args.site}" / "raw"
    files = source_files(args.sources)
    targets = [(source, destination / source.name) for source in files]
    conflicts = [target for _, target in targets if target.exists()]
    if conflicts:
        raise FileExistsError(
            "Refusing to overwrite registered files: "
            + ", ".join(str(path) for path in conflicts))

    destination.mkdir(parents=True, exist_ok=True)
    for source, target in targets:
        shutil.copy2(source, target)

    print(json.dumps({
        "domain": args.domain,
        "site": args.site,
        "raw_dir": str(destination),
        "files": [str(target) for _, target in targets],
        "review_input": str(destination),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
