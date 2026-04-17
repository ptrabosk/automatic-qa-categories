#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.audit_pipeline import AuditPipeline
from src.utils.files import iter_json_files, write_json
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all JSON files in a directory.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--glob", default="*.json")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    setup_logging()

    pipeline = AuditPipeline(config_dir=args.config_dir, use_llm=not args.no_llm)
    output_dir = Path(args.output_dir)
    for input_path in iter_json_files(args.input_dir, args.glob):
        try:
            results = pipeline.audit_file(input_path)
            payload = [result.model_dump() for result in results]
            if len(payload) == 1:
                payload = payload[0]
            write_json(output_dir / f"{input_path.stem}.audit.json", payload)
            logging.info("Audited %s", input_path)
        except Exception:
            logging.exception("Failed to audit %s", input_path)
            if args.fail_fast:
                raise


if __name__ == "__main__":
    main()
