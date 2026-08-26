import re

def generate_suggested_response(ticket_text: str, category: str, priority: str, sentiment: str) -> str:
    """
    Generates a structured, empathetic, and actionable customer support response
    incorporating predicted Category, Priority level, and Sentiment tone.
    
    Designed to serve as a baseline rule/template engine that can be swapped or augmented
    with Azure OpenAI / Azure AI Agent Service (AI-103) in enterprise workflows.
    """
    # 1. Sentiment-informed Opening
    if sentiment == "Positive":
        opening = "Thank you for reaching out and for your wonderful feedback!"
    elif sentiment == "Negative":
        if priority == "High":
            opening = "We deeply apologize for the critical inconvenience and frustration this issue has caused."
        else:
            opening = "We apologize for the inconvenience you are experiencing with our service."
    else: # Neutral
        opening = "Thank you for contacting our customer support team."

    # 2. Category-specific Resolution & Action Guidance
    category_templates = {
        "Billing / Payment": (
            "Your billing inquiry regarding payment transactions or account charges has been registered. "
            "Our financial operations team is auditing the transaction details and payment gateway logs. "
            "If any unauthorized or duplicate charge is identified, a prompt reversal or adjustment will be initiated."
        ),
        "Account / Login": (
            "We understand you are having difficulties accessing your account. "
            "Please ensure you are using the latest verification link and check your spam folder for authentication emails. "
            "For security verification, our identity & access management specialists have been alerted to help restore your secure access."
        ),
        "Technical Issue": (
            "We are sorry you are encountering a technical error. Our engineering team has logged the diagnostics. "
            "In the interim, please try clearing your browser cache, verifying network connectivity, or updating to the latest client version. "
            "A technical support engineer is investigating the error logs."
        ),
        "Order / Delivery": (
            "We have logged your order and delivery inquiry. "
            "Our fulfillment and logistics coordination team is tracking the courier transit status and shipment milestones. "
            "We will provide you with an updated delivery timeline or dispatch resolution shortly."
        ),
        "Product Issue": (
            "Thank you for reporting this issue with your product. "
            "Please review our standard troubleshooting steps and hardware guidelines. "
            "If the unit remains defective or damaged, our warranty and replacements team will facilitate a complimentary repair or exchange."
        ),
        "Cancellation / Refund": (
            "Your request regarding subscription cancellation or transaction refund has been received. "
            "Our accounts department will review your purchase eligibility under our 30-day satisfaction guarantee. "
            "You will receive written confirmation along with your refund tracking reference once finalized."
        ),
        "General Inquiry": (
            "Thank you for your interest and general inquiry. "
            "Our customer care team is reviewing your question and will provide comprehensive information, documentation, and guidance to assist you."
        )
    }

    body = category_templates.get(
        category,
        "We have received your support request and assigned it to the appropriate specialist team for review."
    )

    # 3. Priority-based SLA & Escalation Commitment
    if priority == "High":
        sla_note = "Due to the urgent nature of this issue (High Priority), your ticket has been expedited for priority investigation within 1-2 hours."
    elif priority == "Medium":
        sla_note = "Your ticket has been queued with standard priority and an agent will follow up within 4-6 business hours."
    else:
        sla_note = "Our team will review your inquiry and follow up within 24 business hours."

    # 4. Professional Closing
    closing = "Please reply directly to this message if you have additional details to add to your case."

    # Assemble complete coherent response
    full_response = f"{opening}\n\n{body}\n\n{sla_note}\n\n{closing}"
    return full_response


# ----------------------------------------------------------------------
# Future Azure AI Agent Extension Interface (AI-103 Architecture)
# ----------------------------------------------------------------------
def generate_azure_openai_response(ticket_text: str, category: str, priority: str, sentiment: str, context: dict = None) -> str:
    """
    Placeholder/Hook for enterprise Azure OpenAI / Azure AI Agent Service integration.
    
    In a cloud-deployed Azure architecture (AI-103 / AZ-2007), this function constructs a
    system prompt with RAG grounding (retrieving Azure AI Search knowledge base articles)
    and calls Azure OpenAI `gpt-4o-mini` with strict temperature control.
    """
    # Fallback to deterministic template engine during local prototype phase
    return generate_suggested_response(ticket_text, category, priority, sentiment)
