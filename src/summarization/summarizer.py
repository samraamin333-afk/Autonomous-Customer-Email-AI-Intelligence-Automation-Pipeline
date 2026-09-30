"""
Email Summarization Module.
Extracts 3 structured dimensions specified by the architecture:
1. Customer Problem
2. Requested Action
3. Important Details (Entities: Order IDs, Invoice numbers, Amounts, Error codes, Deadlines)
"""

from typing import Dict, Any, List
import re


class EmailSummarizer:
    """Summarizes customer communications into structured analytical fields."""

    # Entity extraction patterns
    ENTITY_PATTERNS = {
        "order_ids": r"\b(?:ORD|PO)[-_]?[0-9A-Z]{4,10}\b",
        "invoice_ids": r"\b(?:INV)[-_]?[0-9A-Z]{4,10}\b",
        "tracking_numbers": r"\b(?:TRK|TRACKING)[-_]?[0-9A-Z]{6,16}\b",
        "account_ids": r"\b(?:ACC|USR|CUST)[-_]?[0-9A-Z]{4,10}\b",
        "error_codes": r"\b(?:ERR_[A-Z0-9_]+|HTTP\s+[45]\d{2}|5\d{2}\s+(?:Internal\s+Server\s+Error|Bad\s+Gateway))\b",
        "monetary_amounts": r"\$\s*[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?",
        "phone_numbers": r"\+?[0-9]{1,3}?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
    }

    ACTION_TRIGGERS = [
        r"(?i)\b(?:please|could\s+you|can\s+you|we\s+need|we\s+are\s+requesting|requesting|kindly|arrange|issue|reset|cancel)\b.*?(?=[.!?\n]|$)",
    ]

    PROBLEM_TRIGGERS = [
        r"(?i)\b(?:failing|broken|damaged|locked\s+out|not\s+working|cannot\s+access|does\s+not\s+comply|charged\s+twice|error|timeout|crushed|not\s+arriving)\b.*?(?=[.!?\n]|$)",
    ]

    def __init__(self):
        self.entity_regexes = {k: re.compile(p, re.IGNORECASE) for k, p in self.ENTITY_PATTERNS.items()}
        self.action_regexes = [re.compile(p) for p in self.ACTION_TRIGGERS]
        self.problem_regexes = [re.compile(p) for p in self.PROBLEM_TRIGGERS]

    def extract_important_details(self, text: str) -> Dict[str, List[str]]:
        """Extracts key business entities like Order IDs, Invoices, Amounts, and Error codes."""
        details = {}
        for entity_name, regex in self.entity_regexes.items():
            matches = list(set(regex.findall(text)))
            if matches:
                details[entity_name] = matches
        return details

    def extract_problem(self, sentences: List[str], subject: str = "") -> str:
        """Identifies the core customer problem statement."""
        # 1. Check if subject itself clearly states problem
        if any(w in subject.lower() for w in ["critical", "error", "damaged", "locked out", "charged twice", "refund", "cancel"]):
            subject_problem = re.sub(r"(?i)^(critical|urgent|re|fwd):\s*", "", subject).strip()
        else:
            subject_problem = ""

        # 2. Search body sentences for problem indicators
        body_problem = ""
        for s in sentences:
            s_clean = s.strip()
            for r in self.problem_regexes:
                match = r.search(s_clean)
                if match:
                    body_problem = s_clean
                    break
            if body_problem:
                break

        if body_problem and subject_problem:
            return f"{subject_problem} ({body_problem})"
        return body_problem or subject_problem or (sentences[0] if sentences else "Inquiry regarding service")

    def extract_requested_action(self, sentences: List[str]) -> str:
        """Identifies the explicit or implicit action requested by the customer."""
        actions = []
        for s in sentences:
            s_clean = s.strip()
            for r in self.action_regexes:
                match = r.search(s_clean)
                if match:
                    actions.append(match.group(0).strip())
                    break
            if len(actions) >= 2:
                break

        if actions:
            return "; ".join(actions)
        return "Review inquiry and provide standard assistance"

    def summarize(self, cleaned_body: str, subject: str = "") -> Dict[str, Any]:
        """
        Produces a 3-part structured summary:
        - Customer Problem
        - Requested Action
        - Important Details
        """
        # Split text into meaningful sentences
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned_body)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]

        problem = self.extract_problem(sentences, subject)
        action = self.extract_requested_action(sentences)
        details = self.extract_important_details(f"{subject}\n{cleaned_body}")

        # Format details into clean human-readable string summary as well as dict
        details_summary = ", ".join([f"{k}: {', '.join(v)}" for k, v in details.items()]) if details else "None specified"

        return {
            "customer_problem": problem,
            "requested_action": action,
            "important_details": details,
            "important_details_summary": details_summary,
            "executive_one_liner": f"[{problem}] -> Requested: [{action}]",
        }
