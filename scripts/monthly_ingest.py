"""
Monthly entry point: ingest everything currently sitting in data/inbox/,
classify each document, then generate the monthly report. This is the script
you'd schedule (e.g. Windows Task Scheduler) or trigger from the UI's
"Ingest new documents" button.

Usage: python scripts/monthly_ingest.py
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Documents, category names and generated summaries in this project are multilingual
# by design. Force UTF-8 on the console: Windows defaults to cp1252, which raises
# UnicodeEncodeError the moment a CJK character is printed.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


from docclassify.config import CONFIG
from docclassify.pipeline import process_folder
from docclassify.reports.monthly_report import generate_monthly_report
from docclassify.storage import sqlite_store


def main():
    sqlite_store.init_db()

    upload_batch = datetime.now(timezone.utc).strftime("%Y-%m")
    inbox = CONFIG["storage"]["inbox_dir"]

    print(f"Ingesting documents from {inbox} as batch '{upload_batch}' ...")
    results = process_folder(inbox, upload_batch=upload_batch)
    print(f"Ingested {len(results)} documents.")

    if not results:
        print("No new documents found - skipping report generation.")
        return

    print("Generating monthly report ...")
    report = generate_monthly_report(upload_batch)
    print("\n=== Monthly Report ===")
    print(report["overall_digest"])
    print(f"\nFull stats: {report['stats']}")


if __name__ == "__main__":
    main()
