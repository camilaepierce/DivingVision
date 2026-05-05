#!/usr/bin/env python3
"""Truncate moderate train/test split JSON files.

Usage: python tools/truncate_splits.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'models' / 'simple'

def truncate(path: Path, n: int):
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise RuntimeError(f"Expected list in {path}")
    print(f"{path.name}: original {len(data)} entries")
    truncated = data[:n]
    path.write_text(json.dumps(truncated, indent=2))
    print(f"{path.name}: written {len(truncated)} entries")

def main():
    train = MOD / 'train_split.json'
    test = MOD / 'test_split.json'
    if not train.exists() or not test.exists():
        raise FileNotFoundError('train_split.json or test_split.json missing in models/moderate')

    truncate(train, 100)
    truncate(test, 10)

if __name__ == '__main__':
    main()
