from __future__ import annotations

import argparse
from pathlib import Path

from src.audit_pipeline import AuditPipeline
from src.utils.files import iter_json_files, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local QA audits for support conversations.")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--no-llm", action="store_true", help="Disable Ollama and use heuristic specialist fallbacks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single")
    single.add_argument("--input", required=True)
    single.add_argument("--output", required=True)

    batch = subparsers.add_parser("batch")
    batch.add_argument("--input-dir", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--glob", default="*.json")

    args = parser.parse_args()
    pipeline = AuditPipeline(config_dir=args.config_dir, use_llm=not args.no_llm)

    if args.command == "single":
        results = pipeline.audit_file(args.input)
        data = results[0].model_dump() if len(results) == 1 else [result.model_dump() for result in results]
        write_json(args.output, data)
    elif args.command == "batch":
        output_dir = Path(args.output_dir)
        for input_path in iter_json_files(args.input_dir, args.glob):
            results = pipeline.audit_file(input_path)
            data = [result.model_dump() for result in results]
            if len(data) == 1:
                data = data[0]
            write_json(output_dir / f"{input_path.stem}.audit.json", data)


if __name__ == "__main__":
    main()
