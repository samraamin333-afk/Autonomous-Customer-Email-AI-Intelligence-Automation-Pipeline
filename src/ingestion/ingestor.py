"""
Email Ingestion Module.
Provides connectors for IMAP (Gmail, Outlook, custom mail servers) and mock/local file ingestion.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import email
from email.header import decode_header
import imaplib
import json
import os
from datetime import datetime


@dataclass
class RawEmail:
    id: str
    sender: str
    recipient: str
    subject: str
    timestamp: str
    body: str
    raw_headers: Optional[Dict[str, Any]] = None


class BaseEmailIngestor:
    """Abstract interface for email ingestion sources."""
    def fetch_emails(self, limit: int = 50) -> List[RawEmail]:
        raise NotImplementedError


class MockEmailIngestor(BaseEmailIngestor):
    """Loads sample emails from local JSON or accepts direct test payloads."""

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath

    def fetch_emails(self, limit: int = 50) -> List[RawEmail]:
        if not self.filepath or not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Mock email data file not found: {self.filepath}")

        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        emails = []
        for item in data[:limit]:
            emails.append(
                RawEmail(
                    id=item.get("id", f"EML-{len(emails)+1:03d}"),
                    sender=item.get("sender", "unknown@example.com"),
                    recipient=item.get("recipient", "support@company.com"),
                    subject=item.get("subject", "(No Subject)"),
                    timestamp=item.get("timestamp", datetime.utcnow().isoformat() + "Z"),
                    body=item.get("body", ""),
                    raw_headers=item.get("raw_headers", {}),
                )
            )
        return emails


class IMAPEmailIngestor(BaseEmailIngestor):
    """
    Connects to standard IMAP email servers (e.g. imap.gmail.com, outlook.office365.com).
    Extracts headers and parses both plaintext and multipart/HTML payloads safely.
    """

    def __init__(self, host: str, user: str, password: str, port: int = 993, folder: str = "INBOX"):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.folder = folder

    def _decode_str(self, header_val: Optional[str]) -> str:
        if not header_val:
            return ""
        decoded_fragments = decode_header(header_val)
        result = []
        for text, encoding in decoded_fragments:
            if isinstance(text, bytes):
                encoding = encoding or "utf-8"
                try:
                    result.append(text.decode(encoding, errors="replace"))
                except LookupError:
                    result.append(text.decode("utf-8", errors="replace"))
            else:
                result.append(str(text))
        return "".join(result)

    def fetch_emails(self, limit: int = 50, mark_seen: bool = False) -> List[RawEmail]:
        """Fetches unread or recent emails via IMAP SSL connection."""
        mail = imaplib.IMAP4_SSL(self.host, self.port)
        try:
            mail.login(self.user, self.password)
            mail.select(self.folder)

            status, search_data = mail.search(None, "UNSEEN" if not mark_seen else "ALL")
            if status != "OK":
                return []

            email_ids = search_data[0].split()
            if not email_ids:
                return []

            # Take the latest N messages
            email_ids = email_ids[-limit:]
            emails = []

            for eid in email_ids:
                fetch_cmd = "(RFC822)" if mark_seen else "(BODY.PEEK[])"
                res, data = mail.fetch(eid, fetch_cmd)
                if res != "OK":
                    continue

                for response_part in data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        subject = self._decode_str(msg.get("Subject"))
                        sender = self._decode_str(msg.get("From"))
                        recipient = self._decode_str(msg.get("To"))
                        date_str = msg.get("Date", datetime.utcnow().isoformat())

                        # Extract email body (prefer html if available for realistic preprocessing)
                        body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                ctype = part.get_content_type()
                                cdisp = str(part.get("Content-Disposition"))
                                if "attachment" not in cdisp:
                                    if ctype in ["text/html", "text/plain"]:
                                        payload = part.get_payload(decode=True)
                                        if payload:
                                            body = payload.decode(errors="replace")
                                            if ctype == "text/html":
                                                break
                        else:
                            payload = msg.get_payload(decode=True)
                            if payload:
                                body = payload.decode(errors="replace")

                        emails.append(
                            RawEmail(
                                id=eid.decode() if isinstance(eid, bytes) else str(eid),
                                sender=sender,
                                recipient=recipient,
                                subject=subject,
                                timestamp=date_str,
                                body=body,
                            )
                        )
            return emails
        finally:
            try:
                mail.close()
                mail.logout()
            except Exception:
                pass
