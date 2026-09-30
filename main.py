"""
Main Entrypoint for the Customer Support Email AI Automation System.

Commands:
    python main.py --run-sample        # Run complete pipeline on sample customer emails
    python main.py --train-t5          # Fine-tune T5-small response model (PyTorch/Transformers)
    python main.py --inspect-email ID  # Inspect individual email detailed stage-by-stage output
"""

import argparse
import sys
import os

# Fix Windows console UTF-8 encoding for emojis and markdown symbols
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.ingestion.ingestor import MockEmailIngestor
from src.pipeline import EmailAIPipeline
from src.reporting.report_generator import DailyReportGenerator


def run_sample_pipeline():
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_emails.json")
    ingestor = MockEmailIngestor(sample_path)
    emails = ingestor.fetch_emails(limit=20)

    pipeline = EmailAIPipeline()
    results = pipeline.run(
        emails=emails,
        export_json_path="data/processed_emails.json",
        export_csv_path="data/processed_emails.csv",
        report_html_path="data/daily_management_report.html",
    )

    print("\n" + "=" * 78)
    print("                     DAILY MANAGEMENT REPORT (MARKDOWN)")
    print("=" * 78)
    print(results["markdown_report"])
    print("=" * 78)
    print(f"\n[DONE] Pipeline execution completed successfully.")
    print(f"       -> View HTML Report: file:///{os.path.abspath('data/daily_management_report.html')}")
    print(f"       -> JSON Records:     file:///{os.path.abspath('data/processed_emails.json')}")
    print(f"       -> CSV Export:       file:///{os.path.abspath('data/processed_emails.csv')}\n")


def inspect_email(email_id: str):
    import json
    json_path = os.path.join(os.path.dirname(__file__), "data", "processed_emails.json")
    if not os.path.exists(json_path):
        print("[!] No processed records found. Please run with --run-sample first.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    target = next((r for r in records if r["id"].lower() == email_id.lower()), None)
    if not target:
        print(f"[!] Email ID '{email_id}' not found. Available IDs: {[r['id'] for r in records]}")
        return

    print("=" * 78)
    print(f" STAGE-BY-STAGE TRACE: {target['id']}")
    print("=" * 78)
    print(f"Sender:             {target['sender']}")
    print(f"Subject:            {target['subject']}")
    print(f"Timestamp:          {target['timestamp']}")
    print("-" * 78)
    print("[1] PREPROCESSING:")
    print(f"    Noise Reduction: {target['preprocessing']['character_reduction_pct']}%")
    print(f"    Clean Body:\n{target['preprocessing']['cleaned_body']}")
    print("-" * 78)
    print("[2] CLASSIFICATION:")
    print(f"    Category:        {target['category']} (Confidence: {target['category_confidence']*100:.1f}%)")
    print(f"    Engine:          {target['classification_method']}")
    print("-" * 78)
    print("[3] URGENCY DETECTION:")
    print(f"    Level:           {target['urgency_level']} (Score: {target['urgency_score']})")
    print(f"    Target SLA:      {target['sla_target_hours']} Hours")
    print(f"    Triggers:        {target['urgency_signals']}")
    print("-" * 78)
    print("[4] SUMMARIZATION:")
    print(f"    Problem:         {target['summary']['customer_problem']}")
    print(f"    Action Req:      {target['summary']['requested_action']}")
    print(f"    Key Details:     {target['summary']['important_details_summary']}")
    print("-" * 78)
    print("[5] RESPONSE GENERATION:")
    print(f"    Engine:          {target['response']['engine']}")
    print(f"    Policy Applied:  {target['response']['business_context_applied']}")
    print(f"\n--- DRAFT RESPONSE ---\n{target['response']['response_text']}\n----------------------")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Customer Email AI Pipeline CLI")
    parser.add_argument("--run-sample", action="store_true", help="Run end-to-end pipeline on sample emails")
    parser.add_argument("--train-t5", action="store_true", help="Launch T5-small fine-tuning script")
    parser.add_argument("--inspect-email", type=str, help="Inspect specific email ID after processing")
    args = parser.parse_args()

    if args.run_sample or len(sys.argv) == 1:
        run_sample_pipeline()
    elif args.inspect_email:
        inspect_email(args.inspect_email)
    elif args.train_t5:
        from src.generation.train_t5 import train
        train()


if __name__ == "__main__":
    main()
