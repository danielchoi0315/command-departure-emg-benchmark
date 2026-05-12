from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a lightweight stage manifest.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--output", action="append", default=[])
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--env-file", default="")
    parser.add_argument("--repo-hint", default="")
    parser.add_argument("--stage-start", default="")
    parser.add_argument("--stage-end", default="")
    args = parser.parse_args()

    files = []
    for group, values in [("input", args.input), ("output", args.output), ("config", args.config)]:
        for value in values:
            path = Path(value)
            if path.is_file():
                files.append({"role": group, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
            elif path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file():
                        files.append({"role": group, "path": str(child), "sha256": sha256(child), "bytes": child.stat().st_size})

    manifest = {
        "stage": args.stage,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "stage_start": args.stage_start,
        "stage_end": args.stage_end,
        "repo_hint": args.repo_hint,
        "env_file": args.env_file,
        "files": files,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
