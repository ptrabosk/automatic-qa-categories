#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.audit_pipeline import AuditPipeline
from src.utils.files import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one JSON file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    pipeline = AuditPipeline(config_dir=args.config_dir, use_llm=not args.no_llm)
    results = pipeline.audit_file(args.input)
    payload = results[0].model_dump() if len(results) == 1 else [result.model_dump() for result in results]
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
