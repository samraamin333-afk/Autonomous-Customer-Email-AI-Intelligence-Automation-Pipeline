"""
Urgency Detection Module.
Combines Rule-based signals (legal, operational, temporal, financial)
with NLP / Linguistic signals (sentiment intensity, casing, punctuation cues).
"""

from typing import Dict, Any, List
import re


class UrgencyDetector:
    """Hybrid Urgency and Priority Detection Engine."""

    # Rule-Based: Legal, Compliance & Escaped Risk Signals
    CRITICAL_RISK_PATTERNS = [
        (r"\b(lawyer|attorney|legal\s+action|sue|court|lawsuit|chargeback|regulatory|fraud)\b", 0.40, "Legal/Regulatory Risk"),
        (r"\b(production\s+(?:down|failing|broken)|system\s+outage|service\s+down|sev-?1|blocking\s+(?:all|checkouts|customers))\b", 0.40, "Production Outage"),
        (r"\b(losing\s+(?:money|thousands|millions|revenue)|business\s+critical)\b", 0.35, "Direct Revenue Impact"),
    ]

    # Rule-Based: Temporal & SLA Pressure Signals
    TIME_SENSITIVE_PATTERNS = [
        (r"\b(asap|immediately|emergency|urgent|urgently|critical|highest\s+priority)\b", 0.30, "Explicit Urgency Keyword"),
        (r"\b(within\s+(?:24|12|2|4|1)\s*hours?|today|before\s+(?:eod|end\s+of\s+day|5\s*pm)|right\s+now)\b", 0.25, "Tight Deadline / SLA"),
    ]

    # NLP / Linguistic Distress & Sentiment Signals
    NLP_SENTIMENT_PATTERNS = [
        (r"\b(unacceptable|furious|terrible|disaster|nightmare|ridiculous|unusable|worst)\b", 0.20, "High Negative Sentiment"),
        (r"\b(tried\s+\d+\s+times|waiting\s+for\s+(?:days|weeks)|no\s+response)\b", 0.20, "Customer Frustration Loop"),
    ]

    def __init__(self):
        self.risk_compiled = [(re.compile(p, re.I), w, desc) for p, w, desc in self.CRITICAL_RISK_PATTERNS]
        self.time_compiled = [(re.compile(p, re.I), w, desc) for p, w, desc in self.TIME_SENSITIVE_PATTERNS]
        self.nlp_compiled = [(re.compile(p, re.I), w, desc) for p, w, desc in self.NLP_SENTIMENT_PATTERNS]

    def _detect_financial_signal(self, text: str) -> tuple[float, List[str]]:
        """Detects high monetary amounts (> $500) mentioned in dispute or refund."""
        signals = []
        weight = 0.0
        # Match dollar values e.g. $1,200.00 or 1200 dollars
        amounts = re.findall(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)", text)
        for amt_str in amounts:
            try:
                num = float(amt_str.replace(",", ""))
                if num >= 1000:
                    weight = max(weight, 0.25)
                    signals.append(f"High Financial Exposure: ${num:,.2f}")
                elif num >= 500:
                    weight = max(weight, 0.15)
                    signals.append(f"Moderate Financial Exposure: ${num:,.2f}")
            except ValueError:
                pass
        return weight, signals

    def _detect_linguistic_signals(self, raw_text: str) -> tuple[float, List[str]]:
        """Analyzes punctuation intensity and excessive capitalization."""
        signals = []
        score = 0.0

        # Punctuation signals: multiple exclamation marks or interrobangs
        if re.search(r"!{2,}|\?!|\?!{2,}", raw_text):
            score += 0.15
            signals.append("Multiple Exclamation Marks (Elevated Distress)")

        # Casing signals: check for uppercase words of length >= 4
        words = raw_text.split()
        caps_words = [w for w in words if w.isupper() and len(w) >= 4 and w.isalpha()]
        if len(caps_words) >= 2:
            score += 0.15
            signals.append(f"Capitalized Emphasis Words: {', '.join(caps_words[:4])}")

        return min(score, 0.25), signals

    def detect_urgency(self, cleaned_text: str, subject: str = "", raw_body: str = "") -> Dict[str, Any]:
        """
        Calculates cumulative urgency score from rule-based and linguistic signals.
        Returns urgency level, score (0.0 to 1.0), recommended SLA, and detected triggers.
        """
        combined_text = f"{subject} {cleaned_text}".strip()
        signals = []
        cumulative_score = 0.0

        # 1. Rule-Based Critical Risks
        for regex, weight, label in self.risk_compiled:
            matches = regex.findall(combined_text)
            if matches:
                cumulative_score += weight
                signals.append(f"{label}: '{matches[0]}'")

        # 2. Rule-Based Time Pressure
        for regex, weight, label in self.time_compiled:
            matches = regex.findall(combined_text)
            if matches:
                cumulative_score += weight
                signals.append(f"{label}: '{matches[0]}'")

        # 3. NLP Negative Sentiment / Frustration
        for regex, weight, label in self.nlp_compiled:
            matches = regex.findall(combined_text)
            if matches:
                cumulative_score += weight
                signals.append(f"{label}: '{matches[0]}'")

        # 4. Financial Signals
        fin_weight, fin_signals = self._detect_financial_signal(combined_text)
        cumulative_score += fin_weight
        signals.extend(fin_signals)

        # 5. Linguistic / Punctuation Signals
        ling_weight, ling_signals = self._detect_linguistic_signals(f"{subject} {raw_body or cleaned_text}")
        cumulative_score += ling_weight
        signals.extend(ling_signals)

        # Normalize score to [0.0, 1.0] range
        final_score = min(round(cumulative_score, 2), 1.0)

        # Categorize Urgency Level and determine SLA
        if final_score >= 0.70:
            urgency_level = "CRITICAL"
            sla_hours = 1
        elif final_score >= 0.45:
            urgency_level = "HIGH"
            sla_hours = 4
        elif final_score >= 0.20:
            urgency_level = "MEDIUM"
            sla_hours = 24
        else:
            urgency_level = "LOW"
            sla_hours = 48

        return {
            "urgency_level": urgency_level,
            "urgency_score": final_score,
            "sla_target_hours": sla_hours,
            "signals": signals,
            "requires_human_escalation": urgency_level in ["CRITICAL", "HIGH"],
        }
