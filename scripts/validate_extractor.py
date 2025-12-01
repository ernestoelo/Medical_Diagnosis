#!/usr/bin/env python3
"""Run extractor over a list of sample queries and print JSON lines.

Usage:
  python3 scripts/validate_extractor.py
"""
import json
from pathlib import Path
from src.node_extractor import extract_nodes_from_query


BASE = Path(__file__).resolve().parent.parent
SAMPLE_FILE = BASE / 'data' / 'sample_queries.txt'


def load_samples(path: Path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    samples = load_samples(SAMPLE_FILE)
    if not samples:
        print('No sample queries found at', SAMPLE_FILE)
        return 1
    results = []
    for q in samples:
        out = extract_nodes_from_query(q)
        rec = {'query': q, 'extracted': out}
        print(json.dumps(rec, ensure_ascii=False))
        results.append(rec)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
