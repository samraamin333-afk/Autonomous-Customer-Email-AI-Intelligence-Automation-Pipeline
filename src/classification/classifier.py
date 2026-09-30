"""
Email Classification Module.
Categorizes customer emails into 8 standard enterprise categories:
- Technical Issue
- Billing
- Refund
- Account
- Order
- Delivery
- Cancellation
- General Inquiry

Uses a Hybrid approach:
1. Rule-based pattern matching (domain terminology, regex triggers)
2. Calibrated TF-IDF + SGDClassifier (with self-contained bootstrap corpus)
"""

from typing import Dict, Any, List, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


class EmailClassifier:
    """Hybrid Classifier combining domain rules with calibrated TF-IDF classification."""

    CATEGORIES = [
        "Technical Issue",
        "Billing",
        "Refund",
        "Account",
        "Order",
        "Delivery",
        "Cancellation",
        "General Inquiry",
    ]

    RULE_PATTERNS = {
        "Technical Issue": [
            r"\b(bug|error|500|404|exception|crash|api|endpoint|stack\s*trace|broken|timeout|failure|latency|server|downtime)\b",
            r"\b(not\s+working|fails\s+to|integration\s+issue|database|sdk|webhook)\b",
        ],
        "Billing": [
            r"\b(invoice|charge|charged|double\s+bill(?:ing)?|credit\s+card|pricing|payment|receipt|subscription\s+fee|overcharge)\b",
            r"\b(charged\s+twice|unauthorized\s+charge|billing\s+cycle|tax\s+invoice)\b",
        ],
        "Refund": [
            r"\b(refund|money\s*back|reimburse|reimbursement|return\s+payment|credit\s+my\s+account)\b",
            r"\b(money-back\s+guarantee|return\s+funds)\b",
        ],
        "Account": [
            r"\b(password|2fa|two-factor|login|log\s+in|locked\s+out|sso|authentication|credentials|reset\s+password|profile|mfa)\b",
            r"\b(cannot\s+access\s+account|verify\s+identity|user\s+id)\b",
        ],
        "Order": [
            r"\b(order\s*#?|purchase\s+order|po-?\d+|placed\s+an\s+order|buy|cart|checkout|bulk\s+order|procurement)\b",
            r"\b(fulfillment|order\s+status|order\s+confirmation)\b",
        ],
        "Delivery": [
            r"\b(shipment|shipping|dhl|fedex|ups|courier|tracking|tracking\s*#?|delivered|damaged\s+in\s+transit|lost\s+package)\b",
            r"\b(delivery\s+delay|customs|where\s+is\s+my\s+package|broken\s+box|crushed)\b",
        ],
        "Cancellation": [
            r"\b(cancel|cancellation|terminate|downgrade|stop\s+subscription|discontinue|close\s+account|do\s+not\s+renew)\b",
            r"\b(end\s+contract|unsubscribe)\b",
        ],
        "General Inquiry": [
            r"\b(inquiry|question|information|documentation|features|demo|sandbox|partner|pricing\s+tiers|sales|evaluating)\b",
            r"\b(how\s+to|do\s+you\s+support|feature\s+request|consultation)\b",
        ],
    }

    # High-quality bootstrap training data to train the ML pipeline on initialization
    BOOTSTRAP_DATA: List[Tuple[str, str]] = [
        # Technical Issue
        ("Production API returning 500 error after deployment endpoint broken timeout", "Technical Issue"),
        ("Application crashes when uploading large CSV file stacktrace internal error", "Technical Issue"),
        ("Webhook is not triggering for created events server returns 502 bad gateway", "Technical Issue"),
        ("Getting ERR_CONNECTION_TIMED_OUT when connecting to database cluster", "Technical Issue"),
        ("SDK authentication fails with invalid signature error on python 3.12", "Technical Issue"),

        # Billing
        ("I was charged twice on my credit card for invoice INV-90022 please check", "Billing"),
        ("Why did my subscription fee increase this month without prior notice?", "Billing"),
        ("Need updated receipt and VAT tax invoice with company tax registration id", "Billing"),
        ("Unrecognized charge of $499 on credit card statement from your merchant id", "Billing"),
        ("Update corporate credit card on file for recurring monthly payment", "Billing"),

        # Refund
        ("Please issue a full refund as the product does not meet our requirements", "Refund"),
        ("Requesting money back under your 30 day satisfaction guarantee policy", "Refund"),
        ("I was billed after cancelling, please reverse the transaction and refund", "Refund"),
        ("Service was unavailable for 3 days, requesting SLA downtime refund credit", "Refund"),
        ("Refund requested for accidental purchase of annual enterprise tier", "Refund"),

        # Account
        ("Locked out of my account because 2FA SMS code is not being sent to my phone", "Account"),
        ("Need password reset link sent to admin email address credentials forgotten", "Account"),
        ("Unable to log in via Okta SAML SSO keeps redirecting to login page", "Account"),
        ("Change primary account email address from old domain to new corporate domain", "Account"),
        ("How do I invite additional team members and manage their RBAC permissions", "Account"),

        # Order
        ("What is the status of my order ORD-55421 placed last Monday?", "Order"),
        ("Need to modify quantity of units in Purchase Order PO-98124 before processing", "Order"),
        ("Did not receive order confirmation email after completing checkout process", "Order"),
        ("Bulk procurement order for 50 enterprise hardware licenses", "Order"),
        ("Order checkout keeps failing at final review step", "Order"),

        # Delivery
        ("Tracking number shows delivered but package was not received at our loading dock", "Delivery"),
        ("Shipment arrived damaged outer carton crushed and hardware broken inside", "Delivery"),
        ("FedEx tracking says exception delayed in customs clearance documentation needed", "Delivery"),
        ("When will tracking details be generated for recent shipment dispatched?", "Delivery"),
        ("Wrong items delivered in parcel, received model A instead of model B", "Delivery"),

        # Cancellation
        ("Please cancel my subscription immediately and ensure auto-renewal is off", "Cancellation"),
        ("We are terminating our contract at end of billing cycle do not renew", "Cancellation"),
        ("Close my account and delete all stored personal customer data", "Cancellation"),
        ("Downgrade from Enterprise tier to Free tier at end of month", "Cancellation"),
        ("We decided to cancel our service due to internal budget cuts", "Cancellation"),

        # General Inquiry
        ("What are the rate limits and API quotas for the developer tier?", "General Inquiry"),
        ("Do you offer a sandbox or staging environment for sandbox testing before buying?", "General Inquiry"),
        ("Would like to schedule a product demo with your enterprise sales engineering team", "General Inquiry"),
        ("Does your platform comply with SOC2 Type II and ISO 27001 security standards?", "General Inquiry"),
        ("General question about supported cloud regions and data residency in Europe", "General Inquiry"),
    ]

    def __init__(self):
        self._compile_rules()
        self._init_and_train_ml()

    def _compile_rules(self):
        self.compiled_rules = {}
        for category, patterns in self.RULE_PATTERNS.items():
            self.compiled_rules[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def _init_and_train_ml(self):
        """Initializes and fits a calibrated classifier on domain data."""
        texts = [item[0] for item in self.BOOTSTRAP_DATA]
        labels = [item[1] for item in self.BOOTSTRAP_DATA]

        base_clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1000, random_state=42)
        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, cv=3)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("clf", calibrated_clf),
        ])
        self.pipeline.fit(texts, labels)

    def _rule_scores(self, text: str) -> Dict[str, float]:
        """Calculates normalized rule-based keyword match density for each category."""
        scores = {cat: 0.0 for cat in self.CATEGORIES}
        for cat, patterns in self.compiled_rules.items():
            matches = 0
            for pat in patterns:
                matches += len(pat.findall(text))
            scores[cat] = float(matches)

        total = sum(scores.values())
        if total > 0:
            return {cat: val / total for cat, val in scores.items()}
        return {cat: 0.0 for cat in self.CATEGORIES}

    def classify(self, text: str, subject: str = "") -> Dict[str, Any]:
        """
        Classifies email content into one of the 8 categories using hybrid scoring.
        Weighted combination: 60% ML Probability + 40% Domain Rules.
        """
        full_text = f"{subject} {text}".strip()
        if not full_text:
            return {
                "category": "General Inquiry",
                "confidence": 0.5,
                "probabilities": {cat: 1.0 / len(self.CATEGORIES) for cat in self.CATEGORIES},
                "method": "default_empty",
            }

        # 1. ML Probability prediction
        ml_probs_arr = self.pipeline.predict_proba([full_text])[0]
        ml_classes = self.pipeline.classes_
        ml_probs = {cls_name: float(prob) for cls_name, prob in zip(ml_classes, ml_probs_arr)}

        # Fill any missing category keys
        for cat in self.CATEGORIES:
            ml_probs.setdefault(cat, 0.0)

        # 2. Rule scores
        rule_scores = self._rule_scores(full_text)
        has_rule_signals = sum(rule_scores.values()) > 0

        # 3. Hybrid fusion
        hybrid_scores = {}
        alpha = 0.65 if has_rule_signals else 1.0
        beta = 0.35 if has_rule_signals else 0.0

        for cat in self.CATEGORIES:
            hybrid_scores[cat] = (alpha * ml_probs.get(cat, 0.0)) + (beta * rule_scores.get(cat, 0.0))

        # Re-normalize
        total_score = sum(hybrid_scores.values()) or 1.0
        normalized_probs = {cat: round(score / total_score, 4) for cat, score in hybrid_scores.items()}

        best_category = max(normalized_probs, key=normalized_probs.get)
        confidence = normalized_probs[best_category]

        # If low confidence across all classes, fall back to General Inquiry
        if confidence < 0.28:
            best_category = "General Inquiry"

        return {
            "category": best_category,
            "confidence": confidence,
            "probabilities": normalized_probs,
            "method": "hybrid_ml_rules" if has_rule_signals else "calibrated_ml",
        }
