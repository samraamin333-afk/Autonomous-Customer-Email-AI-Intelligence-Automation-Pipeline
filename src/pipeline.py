"""
End-to-End Orchestrated Email AI Automation Pipeline.
Wires all 8 architectural stages:
1. Email Ingestion (IMAP / Mock)
2. Email Preprocessing (HTML, signature, quotation cleaning)
3. Email Classification (8 categories)
4. Urgency Detection (Rule + NLP signals)
5. Email Summarization (Problem, action, details)
6. Response Generation (T5-small / Business context)
7. Structured Output (CSV / JSON export)
8. Daily Management Report Generation
"""

from typing import List, Dict, Any, Optional
import time
import os
import json
import pandas as pd

from src.ingestion.ingestor import RawEmail, MockEmailIngestor, IMAPEmailIngestor
from src.preprocessing.cleaner import EmailPreprocessor
from src.classification.classifier import EmailClassifier
from src.urgency.detector import UrgencyDetector
from src.summarization.summarizer import EmailSummarizer
from src.generation.t5_generator import T5ResponseGenerator
from src.reporting.report_generator import DailyReportGenerator


class EmailAIPipeline:
    """Master Pipeline orchestrating the email processing lifecycle."""

    def __init__(
        self,
        t5_model_dir: Optional[str] = None,
        use_t5: bool = True,
    ):
        print("[INIT] Initializing Email AI Pipeline components...")
        self.preprocessor = EmailPreprocessor()
        self.classifier = EmailClassifier()
        self.urgency_detector = UrgencyDetector()
        self.summarizer = EmailSummarizer()
        self.generator = T5ResponseGenerator(model_dir=t5_model_dir, use_t5_if_available=use_t5)
        print("[INIT] All pipeline stages initialized successfully.")

    def process_single_email(self, raw_email: RawEmail) -> Dict[str, Any]:
        """Runs an individual email through the complete 6 inference stages."""
        t_start = time.perf_counter()

        # 1. Preprocessing
        clean_res = self.preprocessor.preprocess(raw_email.body)
        cleaned_body = clean_res["cleaned_body"]

        # 2. Classification
        class_res = self.classifier.classify(cleaned_body, subject=raw_email.subject)

        # 3. Urgency Detection
        urgency_res = self.urgency_detector.detect_urgency(
            cleaned_text=cleaned_body,
            subject=raw_email.subject,
            raw_body=raw_email.body,
        )

        # 4. Summarization
        summary_res = self.summarizer.summarize(cleaned_body, subject=raw_email.subject)

        # 5. Response Generation
        response_res = self.generator.generate_response(
            category=class_res["category"],
            urgency_level=urgency_res["urgency_level"],
            summary=summary_res,
            sender=raw_email.sender,
            cleaned_body=cleaned_body,
            subject=raw_email.subject,
        )

        t_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        return {
            "id": raw_email.id,
            "sender": raw_email.sender,
            "recipient": raw_email.recipient,
            "subject": raw_email.subject,
            "timestamp": raw_email.timestamp,
            "preprocessing": {
                "cleaned_body": cleaned_body,
                "character_reduction_pct": clean_res["character_reduction_pct"],
                "extracted_signature": clean_res["extracted_signature"],
            },
            "category": class_res["category"],
            "category_confidence": class_res["confidence"],
            "classification_method": class_res["method"],
            "urgency_level": urgency_res["urgency_level"],
            "urgency_score": urgency_res["urgency_score"],
            "sla_target_hours": urgency_res["sla_target_hours"],
            "urgency_signals": urgency_res["signals"],
            "summary": summary_res,
            "response": response_res,
            "processing_time_ms": t_elapsed_ms,
        }

    def run(
        self,
        emails: List[RawEmail],
        export_json_path: str = "data/processed_emails.json",
        export_csv_path: str = "data/processed_emails.csv",
        report_html_path: str = "data/daily_management_report.html",
    ) -> Dict[str, Any]:
        """Executes pipeline over a batch of emails and exports structured files & reports."""
        print(f"\n[PIPELINE] Processing {len(emails)} emails...")
        processed_records = []

        for i, email_obj in enumerate(emails, 1):
            record = self.process_single_email(email_obj)
            processed_records.append(record)
            print(f"  [{i}/{len(emails)}] ID: {record['id']} | Category: {record['category']} ({record['category_confidence']*100:.0f}%) | Urgency: {record['urgency_level']} | Latency: {record['processing_time_ms']}ms")

        # 6. Structured Output: JSON Export
        os.makedirs(os.path.dirname(export_json_path), exist_ok=True)
        with open(export_json_path, "w", encoding="utf-8") as f:
            json.dump(processed_records, f, indent=2, ensure_ascii=False)
        print(f"\n[EXPORT] Structured JSON saved -> {export_json_path}")

        # Structured Output: Flattened CSV Export
        flat_records = []
        for r in processed_records:
            flat_records.append({
                "email_id": r["id"],
                "timestamp": r["timestamp"],
                "sender": r["sender"],
                "subject": r["subject"],
                "category": r["category"],
                "category_confidence": r["category_confidence"],
                "urgency_level": r["urgency_level"],
                "urgency_score": r["urgency_score"],
                "sla_target_hours": r["sla_target_hours"],
                "customer_problem": r["summary"]["customer_problem"],
                "requested_action": r["summary"]["requested_action"],
                "important_details": r["summary"]["important_details_summary"],
                "response_engine": r["response"]["engine"],
                "response_preview": r["response"]["response_text"][:120] + "...",
                "processing_time_ms": r["processing_time_ms"],
            })

        df = pd.DataFrame(flat_records)
        df.to_csv(export_csv_path, index=False)
        print(f"[EXPORT] Structured CSV saved  -> {export_csv_path}")

        # 7. Daily Management Report Generation
        reporter = DailyReportGenerator(processed_records)
        reporter.generate_html(report_html_path)
        markdown_report = reporter.generate_markdown()
        print(f"[REPORT] Executive HTML Report -> {report_html_path}")

        return {
            "processed_records": processed_records,
            "markdown_report": markdown_report,
            "export_json_path": export_json_path,
            "export_csv_path": export_csv_path,
            "report_html_path": report_html_path,
        }
