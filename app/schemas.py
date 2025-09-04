# app/schemas.py
from __future__ import annotations
import re
from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator

# Accept "ORD-1234", "ord1234", or "ORD 1234" and normalize to "ORD-1234"
ORDER_ID_INPUT_PATTERN = re.compile(r"(?i)^ord[-\s]?(\d{3,})$")

class AnalyzeRequest(BaseModel):
    order_id: Optional[str] = Field(None, description="Optional Order ID, e.g., ORD-1001")
    text: str = Field(..., min_length=5, description="Raw ticket text from email/chat")

    @field_validator("order_id")
    @classmethod
    def normalize_order_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip().upper()
        m = ORDER_ID_INPUT_PATTERN.match(s)
        if not m:
            raise ValueError("Order ID must look like ORD-1001 (ORD- + 3+ digits).")
        digits = m.group(1)
        return f"ORD-{digits}"

class TicketResponse(BaseModel):
    summary: str
    category: str  # refund | delivery | defect | other
    suggested_response: str
    trace: Dict[str, Optional[str]]  # debug/QA info visible in UI if you want

class OrderInfo(BaseModel):
    order_id: str
    status: str
    last_update: str
    carrier: Optional[str] = None
    tracking: Optional[str] = None
    exists: bool = True
