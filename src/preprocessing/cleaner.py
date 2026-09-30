"""
Email Preprocessing Module.
Handles:
1. HTML sanitization and tag stripping (BeautifulSoup + entity unescaping)
2. Signature stripping (RFC 3676 delimiters, signoffs, mobile signatures, legal disclaimers)
3. Thread/Quotation cleaning (nested replies, Outlook/Gmail forward and reply headers)
"""

import html
import re
from typing import Tuple
from bs4 import BeautifulSoup


class EmailPreprocessor:
    """Preprocesses and sanitizes raw incoming customer emails."""

    # RFC 3676 and common signature patterns
    SIG_DELIMITERS = [
        r"(?m)^--\s*$",                     # RFC 3676 standard signature delimiter
        r"(?m)^_{2,}\s*$",                  # Underscore line
        r"(?m)^-{3,}\s*$",                  # Hyphen line
    ]

    COMMON_SIGNOFFS = [
        r"(?i)\b(best\s+regards|kind\s+regards|warm\s+regards|regards|thanks\s*(?:&|and)?\s*regards|thanks|thank\s+you|cheers|sincerely|yours\s+truly)\b.*$",
    ]

    MOBILE_SIGNATURES = [
        r"(?i)sent\s+from\s+my\s+(iphone|ipad|galaxy|android|smartphone|mobile|phone)",
        r"(?i)get\s+outlook\s+for\s+(ios|android)",
    ]

    DISCLAIMER_PATTERNS = [
        r"(?i)confidentiality\s+notice.*$",
        r"(?i)this\s+(?:email|message|transmission)\s+and\s+any\s+attachments\s+(?:is|are)\s+intended.*$",
        r"(?i)if\s+you\s+(?:have\s+received|are\s+not\s+the\s+intended\s+recipient).*$",
    ]

    # Email reply / quotation patterns
    THREAD_PATTERNS = [
        r"(?m)^>.*$",                                            # Lines starting with quote '>'
        r"(?i)on\s+.*,\s+.*wrote:\s*$",                         # 'On Mon, Jan 1, 2026, user wrote:'
        r"(?i)-{3,}\s*original\s+message\s*-{3,}",              # Outlook '-----Original Message-----'
        r"(?i)from:\s*.*?\nsent:\s*.*?\nto:\s*.*?\nsubject:",  # Outlook header block
    ]

    def __init__(self):
        self._compile_regexes()

    def _compile_regexes(self):
        self.delimiters_re = [re.compile(p) for p in self.SIG_DELIMITERS]
        self.signoffs_re = [re.compile(p, re.DOTALL) for p in self.COMMON_SIGNOFFS]
        self.mobile_re = [re.compile(p) for p in self.MOBILE_SIGNATURES]
        self.disclaimer_re = [re.compile(p, re.DOTALL) for p in self.DISCLAIMER_PATTERNS]
        self.thread_re = [re.compile(p) for p in self.THREAD_PATTERNS]

    def remove_html(self, text: str) -> str:
        """Parses HTML into formatted plain text, removing style, script, and HTML tags."""
        if not text:
            return ""

        # Quick check if text contains HTML tags
        if bool(re.search(r"<[^>]+>", text)):
            try:
                soup = BeautifulSoup(text, "html.parser")
                # Remove script and style elements
                for element in soup(["script", "style", "head", "title", "meta", "[document]"]):
                    element.decompose()

                # Add newlines for block elements
                for br in soup.find_all(["br", "p", "div", "h1", "h2", "h3", "li"]):
                    br.replace_with(f"\n{br.get_text()}\n")

                clean_text = soup.get_text()
            except Exception:
                clean_text = re.sub(r"<[^>]+>", " ", text)
        else:
            clean_text = text

        # Decode HTML entities (e.g., &nbsp;, &gt;, &amp;)
        clean_text = html.unescape(clean_text)
        return clean_text

    def clean_thread(self, text: str) -> Tuple[str, str]:
        """Removes historical thread replies and quoted text, preserving the newest message."""
        extracted_thread = ""

        # Check for Outlook block quote header
        outlook_match = re.search(r"(?i)(-{3,}\s*original\s+message\s*-{3,}|from:\s*.*?\nsent:\s*.*?\nto:)", text)
        if outlook_match:
            extracted_thread = text[outlook_match.start():]
            text = text[:outlook_match.start()]

        # Check for 'On <date> wrote:' pattern
        on_wrote_match = re.search(r"(?i)\n\s*on\s+.*,\s+.*wrote:\s*", text)
        if on_wrote_match:
            extracted_thread = (extracted_thread + "\n" + text[on_wrote_match.start():]).strip()
            text = text[:on_wrote_match.start()]

        # Remove '>' quotation lines
        lines = []
        for line in text.splitlines():
            if line.strip().startswith(">"):
                extracted_thread += "\n" + line
            else:
                lines.append(line)

        return "\n".join(lines).strip(), extracted_thread.strip()

    def remove_signature(self, text: str) -> Tuple[str, str]:
        """Removes email signatures, disclaimers, and signoffs."""
        removed_signature = ""

        # 1. RFC delimiter check
        for pattern in self.delimiters_re:
            match = pattern.search(text)
            if match:
                removed_signature = text[match.start():].strip()
                text = text[:match.start()]
                break

        # 2. Legal / Confidentiality Disclaimers
        for pattern in self.disclaimer_re:
            match = pattern.search(text)
            if match:
                removed_signature = (removed_signature + "\n" + text[match.start():]).strip()
                text = text[:match.start()]

        # 3. Mobile signatures (e.g. Sent from my iPhone)
        for pattern in self.mobile_re:
            match = pattern.search(text)
            if match:
                removed_signature = (removed_signature + "\n" + text[match.start():]).strip()
                text = text[:match.start()]

        # 4. Standard Signoffs (e.g. 'Best regards, John...')
        # Look in the bottom third of the email to avoid accidental false positives in body text
        lines = text.splitlines()
        if len(lines) > 2:
            cutoff = max(len(lines) - 6, len(lines) // 2)
            upper_body = "\n".join(lines[:cutoff])
            lower_body = "\n".join(lines[cutoff:])

            for pattern in self.signoffs_re:
                match = pattern.search(lower_body)
                if match:
                    removed_signature = (removed_signature + "\n" + lower_body[match.start():]).strip()
                    lower_body = lower_body[:match.start()]
                    break
            text = upper_body + "\n" + lower_body

        return text.strip(), removed_signature.strip()

    def preprocess(self, raw_body: str) -> dict:
        """
        Executes full preprocessing pipeline:
        HTML stripping -> Thread extraction -> Signature removal -> Whitespace normalization.
        """
        # Step 1: HTML removal
        text = self.remove_html(raw_body)

        # Step 2: Thread / reply cleaning
        text, thread = self.clean_thread(text)

        # Step 3: Signature removal
        text, signature = self.remove_signature(text)

        # Step 4: Normalize extra whitespace and linebreaks
        cleaned_text = re.sub(r"[ \t]+", " ", text)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

        return {
            "cleaned_body": cleaned_text,
            "extracted_signature": signature,
            "extracted_thread": thread,
            "character_reduction_pct": round((1.0 - (len(cleaned_text) / (len(raw_body) or 1))) * 100, 2),
        }
