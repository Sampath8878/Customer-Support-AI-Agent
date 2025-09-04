# app/main.py
from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AnalyzeRequest, TicketResponse, OrderInfo
from langchain_ollama import OllamaLLM  # pip install langchain-ollama

# ---- LLM config ----
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_HOST, temperature=0.0)

# ---- FastAPI ----
app = FastAPI(title="Customer Support Ticket Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# ---- Mock ORDER "database" (you can replace with data/orders.json later) ----
ORDER_DB: Dict[str, Dict[str, str]] = {
    "ORD-1001": {"status": "shipped",    "last_update": "2025-09-01", "carrier": "DHL",   "tracking": "DHL123456"},
    "ORD-1002": {"status": "delivered",  "last_update": "2025-08-30", "carrier": "UPS",   "tracking": "1Z999"},
    "ORD-1003": {"status": "processing", "last_update": "2025-09-02"},
    "ORD-1004": {"status": "in transit", "last_update": "2025-09-02", "carrier": "FedEx", "tracking": "FDX555888"},
    "ORD-2001": {"status": "returned",   "last_update": "2025-08-25"},
}

def lookup_order(order_id: str) -> OrderInfo:
    rec = ORDER_DB.get(order_id)
    if not rec:
        return OrderInfo(order_id=order_id, status="unknown", last_update="-", exists=False)
    return OrderInfo(order_id=order_id, exists=True, **rec)  # type: ignore[arg-type]

@app.get("/orders/{order_id}", response_model=OrderInfo)
def get_order(order_id: str):
    return lookup_order(order_id)

# ---- Summarize (LLM; short + safe fallback) ----
def summarize_ticket(text: str) -> str:
    prompt = (
        "Summarize the customer's ticket in one short sentence. "
        "No greeting. No extra details.\n"
        f"Ticket:\n{text}\n"
        "Summary:"
    )
    try:
        out = llm.invoke(prompt).strip()
        # Keep only the first sentence
        out = re.split(r"(?<=[.!?])\s+", out)[0]
        return out
    except Exception:
        words = text.strip().split()
        return " ".join(words[:20]) + ("..." if len(words) > 20 else "")

# ---- Hybrid classification (Rules → LLM) ----
KW_REFUND = [
    "refund", "money back", "return my money", "charged twice", "overcharged",
    "cancel order", "cancelled order", "chargeback", "return for refund"
]
KW_DELIVERY = [
    "delivered", "delivery", "courier", "driver", "tracking", "in transit",
    "shipped", "shipping", "delayed", "stuck", "wrong address", "left at",
    "parcel", "package", "never received", "not received", "missing package",
    "proof of delivery", "pod"
]
KW_DEFECT = [
    "broken", "cracked", "defective", "defect", "doesn't work", "not working",
    "faulty", "damaged", "dead on arrival", "screen issue", "battery issue",
    "camera issue", "won't turn on", "malfunction"
]

def rule_based_label(text: str) -> Tuple[Optional[str], List[str]]:
    t = text.lower()
    matched: List[str] = []
    if any(k in t for k in KW_REFUND):   matched.append("refund")
    if any(k in t for k in KW_DELIVERY): matched.append("delivery")
    if any(k in t for k in KW_DEFECT):   matched.append("defect")

    # Strong guards so classic delivery problems don't slip into defect
    if ("delivered" in t and ("never received" in t or "not received" in t)) or \
       "wrong address" in t or "in transit" in t or "tracking" in t or \
       "missing package" in t:
        return "delivery", matched

    # Priority: refund > delivery (if no defect) > defect
    if "refund" in matched:
        return "refund", matched
    if "delivery" in matched and "defect" not in matched:
        return "delivery", matched
    if "defect" in matched:
        return "defect", matched
    return None, matched

def normalize_label(lbl: str) -> str:
    lbl = lbl.strip().lower()
    if "refund" in lbl: return "refund"
    if "deliver" in lbl or "shipping" in lbl: return "delivery"
    if "defect" in lbl or "broken" in lbl or "fault" in lbl: return "defect"
    return "other"

def llm_label(text: str) -> Tuple[str, str]:
    prompt = (
        "Classify the ticket into one of: refund, delivery, defect, other. "
        "Return ONLY a single word label (no punctuation, no quotes).\n\n"
        f"Ticket:\n{text}\n\n"
        "Label:"
    )
    try:
        raw = llm.invoke(prompt).strip()
    except Exception:
        raw = "other"
    return raw, normalize_label(raw)

# ---- Order ID extraction (normalize embedded IDs too) ----
ORDER_ID_IN_TEXT = re.compile(r"(?i)\bord[-\s]?(\d{3,})\b")

def extract_order_id(text: str) -> Optional[str]:
    m = ORDER_ID_IN_TEXT.search(text or "")
    if not m:
        return None
    return f"ORD-{m.group(1)}"

# ---- Reply templates (sentence #1 changes based on having an ID) ----
def _order_status_line(order: Optional[OrderInfo]) -> str:
    if not order:
        return ""
    if not order.exists or order.status == "unknown":
        return f"Order ID {order.order_id} noted. We could not locate this order in our system yet."
    base = f"Order ID {order.order_id} noted. Our system shows the order status as '{order.status}' (last update {order.last_update})."
    if order.carrier and order.tracking:
        base += f" Carrier: {order.carrier}, Tracking: {order.tracking}."
    return base

def template_reply(summary: str, category: str, order: Optional[OrderInfo], had_order_id: bool) -> str:
    lines: List[str] = []

    if category == "delivery":
        if had_order_id and order:
            lines.append(_order_status_line(order))
        else:
            lines.append("No Order ID provided. Please share your Order ID so we can review tracking scans and delivery events.")
        lines += [
            "We’ll check the latest courier updates and verify proof-of-delivery if applicable.",
            "If it was marked delivered but not received, we’ll request GPS/photo confirmation and coordinate with the carrier.",
            "Please confirm the shipping address and check with neighbors or building security.",
            "We’ll update you within 24–48 hours with next steps, including a replacement or refund if the parcel cannot be located."
        ]
        return " ".join(lines[:5])

    if category == "refund":
        if had_order_id and order:
            lines.append(_order_status_line(order))
        else:
            lines.append("No Order ID provided. To start the refund review, please include your Order ID to verify eligibility.")
        lines += [
            "We’ll email return instructions and an RMA number for tracking.",
            "After inspection, refunds are issued to the original payment method.",
            "You’ll receive a confirmation email with the expected time frame."
        ]
        return " ".join(lines[:5])

    if category == "defect":
        if had_order_id and order:
            lines.append(_order_status_line(order))
        lines += [
            "Sorry to hear there’s a product issue.",
            "Please share clear photos or a short video of the defect so we can validate the claim quickly.",
            "We can arrange a replacement or repair and will send a prepaid return label if needed.",
            "If you have your Order ID handy, include it to speed up verification; otherwise a receipt or serial number also helps."
        ]
        return " ".join(lines[:5])

    # other
    if had_order_id and order:
        lines.append(_order_status_line(order))
    lines += [
        "Thanks for reaching out. Please share a few more details about your request.",
        "We’ll triage the issue and guide you through the right next steps.",
        "If this relates to delivery or a refund, including your Order ID helps us look it up quickly."
    ]
    return " ".join(lines[:4])

# ---- Main API endpoint ----
@app.post("/analyze_ticket", response_model=TicketResponse)
def analyze_ticket(req: AnalyzeRequest) -> TicketResponse:
    # 1) summarize
    summary = summarize_ticket(req.text)

    # 2) classify (rules → LLM)
    rule_label, matched = rule_based_label(req.text)
    trace: Dict[str, Optional[str]] = {
        "category_source": "rules" if rule_label else "llm",
        "matched_keywords": ", ".join(matched) or None,
        "llm_raw": None,
        "order_id_input": req.order_id or None,
        "order_id_extracted": None,
        "had_order_id": None,
        "order_exists": None,
        "order_id_effective": None,
    }

    if rule_label:
        category = rule_label
    else:
        raw, lbl = llm_label(req.text)
        category = lbl
        trace["llm_raw"] = raw

    # 3) order id handling (optional; normalize from input or extract from text)
    order_id = req.order_id or extract_order_id(req.text)
    order = lookup_order(order_id) if order_id else None
    had_order_id = order_id is not None

    trace["had_order_id"] = str(had_order_id)
    trace["order_id_effective"] = order_id or None
    if order:
        trace["order_exists"] = "true" if order.exists else "false"
    if not trace["matched_keywords"]:
        trace["matched_keywords"] = None

    # 4) reply (first sentence differs with/without Order ID for delivery/refund)
    suggested = template_reply(summary, category, order, had_order_id)

    return TicketResponse(
        summary=summary,
        category=category,
        suggested_response=suggested,
        trace=trace,
    )
