#!/usr/bin/env python3
"""Replace the GitHub URL placeholder in the manuscript and CITATION.cff."""
from __future__ import annotations
import argparse
from pathlib import Path

PLACEHOLDER = "https://github.com/USERNAME/persian-absa-llama3"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="Final public GitHub repository URL")
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    changed = 0
    for p in args.files:
        text = p.read_text(encoding="utf-8")
        if PLACEHOLDER not in text:
            print(f"[SKIP] placeholder not found: {p}")
            continue
        p.write_text(text.replace(PLACEHOLDER, args.url.rstrip('/')), encoding="utf-8")
        changed += 1
        print(f"[UPDATED] {p}")
    return 0 if changed else 1

if __name__ == "__main__":
    raise SystemExit(main())
