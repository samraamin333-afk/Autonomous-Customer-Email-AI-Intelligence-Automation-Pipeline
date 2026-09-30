"""
Response Generation Module.
Integrates Fine-tuned T5-small model with domain business context.
Includes a robust contextual template engine as a lightweight fallback
when GPU/PyTorch/Transformers environments are in offline/bootstrap mode.
"""

from typing import Dict, Any, Optional
import os
import re

# Business context repository representing corporate policies, SLAs, and links
BUSINESS_POLICIES: Dict[str, Dict[str, Any]] = {
    "Technical Issue": {
        "policy": "Critical production outages are routed to Level-3 engineering on-call. Status updates available at status.saasflow.com.",
        "sla": "1 hour for Sev-1 outages; 4 hours for standard technical defects.",
        "action_statement": "Our engineering operations team has been dispatched to investigate this incident.",
    },
    "Billing": {
        "policy": "Duplicate transactions and billing discrepancies are reconciled immediately. Credit card refunds take 2-3 business days to reflect.",
        "sla": "Within 24 business hours.",
        "action_statement": "Our finance team has been notified to audit the invoice and reverse any erroneous transaction.",
    },
    "Refund": {
        "policy": "All subscriptions and purchases are covered by our 30-day money-back guarantee. Refunds are credited back to the original payment method.",
        "sla": "Processed within 3 to 5 business days.",
        "action_statement": "We have initiated your refund request in accordance with our 30-day money-back guarantee.",
    },
    "Account": {
        "policy": "Account security and MFA/2FA resets require verification of identity. Temporary one-time access codes can be provided upon email confirmation.",
        "sla": "Immediate to 2 hours.",
        "action_statement": "We have triggered a secure authentication reset procedure for your registered user ID.",
    },
    "Order": {
        "policy": "Hardware and bulk orders are fulfilled from our regional fulfillment centers with automated tracking notifications.",
        "sla": "Fulfillment update within 12 business hours.",
        "action_statement": "Our logistics team is pulling your order details to provide exact tracking and dispatch timing.",
    },
    "Delivery": {
        "policy": "Damaged or delayed shipments are covered under comprehensive transit insurance with immediate free priority replacement.",
        "sla": "Expedited dispatch within 24 hours.",
        "action_statement": "We apologize for the damaged items; we are expediting a replacement shipment to your address.",
    },
    "Cancellation": {
        "policy": "Subscribers retain full access until the end of their current billing cycle. Auto-renewal is cancelled immediately with no further charges.",
        "sla": "Instant confirmation.",
        "action_statement": "We have turned off automatic renewal for your account; no further charges will occur.",
    },
    "General Inquiry": {
        "policy": "Comprehensive documentation, API specs, and SDK reference can be found at docs.saasflow.com.",
        "sla": "Within 1 business day.",
        "action_statement": "We are delighted to assist with your inquiry and provide all necessary technical specifications.",
    },
}


class T5ResponseGenerator:
    """
    Generates tailored, professional customer service responses using
    T5-small model architecture or contextual business fallback.
    """

    def __init__(self, model_dir: Optional[str] = None, use_t5_if_available: bool = True):
        self.model_dir = model_dir
        self.use_t5 = use_t5_if_available
        self.tokenizer = None
        self.model = None
        self._load_t5_model()

    def _load_t5_model(self):
        """Attempts to load fine-tuned T5-small weights if transformers/torch are installed."""
        if not self.use_t5:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            model_path = self.model_dir if (self.model_dir and os.path.exists(self.model_dir)) else "t5-small"
            # Note: We check if local path exists or attempt loading
            if self.model_dir and os.path.exists(self.model_dir):
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        except Exception:
            # Fall back smoothly to business context template engine
            self.model = None
            self.tokenizer = None

    def _extract_customer_name(self, sender: str, cleaned_body: str) -> str:
        """Heuristically determines customer name from email address or signoff."""
        # Try extracting from email prefix (e.g. sarah.jenkins@enterprise.io -> Sarah)
        match = re.search(r"^([a-zA-Z]+)[._]", sender)
        if match:
            return match.group(1).capitalize()
        name_part = sender.split("@")[0].replace(".", " ").title()
        if len(name_part) < 20:
            return name_part.split()[0]
        return "Valued Customer"

    def generate_response(
        self,
        category: str,
        urgency_level: str,
        summary: Dict[str, Any],
        sender: str,
        cleaned_body: str,
        subject: str = "",
    ) -> Dict[str, Any]:
        """
        Generates customer response combining business policies, category context,
        and problem details.
        """
        customer_name = self._extract_customer_name(sender, cleaned_body)
        context = BUSINESS_POLICIES.get(category, BUSINESS_POLICIES["General Inquiry"])

        # Format input prompt for T5-small
        prompt = (
            f"respond support: category: {category} | "
            f"urgency: {urgency_level} | "
            f"policy: {context['policy']} | "
            f"problem: {summary.get('customer_problem')} | "
            f"action: {summary.get('requested_action')} | "
            f"details: {summary.get('important_details_summary')}"
        )

        # 1. If T5 model is loaded in memory, perform sequence-to-sequence inference
        if self.model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                outputs = self.model.generate(
                    **inputs,
                    max_length=150,
                    num_beams=3,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return {
                    "response_text": generated_text,
                    "engine": "Fine-Tuned T5-Small",
                    "business_context_applied": context["policy"],
                    "sla_commitment": context["sla"],
                }
            except Exception:
                pass

        # 2. Hybrid Contextual Response Engine (Production-Grade Fallback)
        urgency_prefix = ""
        if urgency_level == "CRITICAL":
            urgency_prefix = "We understand this is an emergency impacting your operations and have marked this ticket as CRITICAL PRIORITY. "
        elif urgency_level == "HIGH":
            urgency_prefix = "We recognize the urgency of this request and have expedited this ticket to our priority queue. "

        problem_ack = summary.get("customer_problem", "your recent inquiry")
        requested_act = summary.get("requested_action", "your request")
        details_ref = ""
        if summary.get("important_details"):
            details_list = [f"{k.replace('_', ' ').title()}: {', '.join(v)}" for k, v in summary["important_details"].items()]
            details_ref = f"\n\nAssociated References: {'; '.join(details_list)}"

        response_body = (
            f"Dear {customer_name},\n\n"
            f"Thank you for contacting SaaSFlow Support regarding: '{subject or category}'.\n\n"
            f"{urgency_prefix}{context['action_statement']}\n\n"
            f"Regarding your concern: \"{problem_ack}\", we are addressing your requested action (\"{requested_act}\") immediately. "
            f"{context['policy']}\n\n"
            f"Expected Resolution Timeline: {context['sla']}.{details_ref}\n\n"
            f"If you have additional logs, receipts, or questions in the meantime, please reply directly to this email.\n\n"
            f"Best regards,\n"
            f"SaaSFlow Customer Care Team\n"
            f"support@saasflow.com | status.saasflow.com"
        )

        return {
            "response_text": response_body,
            "engine": "Contextual Policy Engine (T5 Fallback)",
            "business_context_applied": context["policy"],
            "sla_commitment": context["sla"],
        }
