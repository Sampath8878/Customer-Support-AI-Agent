# ui/app.py
import re
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
API_URL = f"{API_BASE}/analyze_ticket"

st.set_page_config(page_title="🛒 Support Ticket Agent", layout="centered")
st.title("🛒 Customer Support Ticket Agent")
st.caption("Classify issues, summarize, and draft responses automatically.")

def api_alive() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.ok
    except Exception:
        return False

order_id = st.text_input("📦 Order ID (optional)", value="", placeholder="e.g., ORD-1001")
ticket_text = st.text_area("✉ Paste a customer ticket here:", height=200)

if st.button("Analyze Ticket", type="primary"):
    if not api_alive():
        st.error("API is not running. Start it with:  python -m uvicorn app.main:app --reload --port 8000")
    elif not ticket_text.strip():
        st.error("Please paste the ticket text.")
    else:
        with st.spinner("Analyzing..."):
            payload = {"text": ticket_text.strip()}
            if order_id.strip():
                payload["order_id"] = order_id.strip()

            try:
                resp = requests.post(API_URL, json=payload, timeout=60)
                if resp.status_code == 200:
                    result = resp.json()
                    st.subheader("📝 Summary")
                    st.write(result["summary"])

                    st.subheader("📂 Predicted Category")
                    st.success(result["category"])

                    st.subheader("💬 Suggested Response")
                    st.info(result["suggested_response"])

                    # Small trace/debug info so you can see which path was used
                    st.caption(
                        f"Order ID used: {result['trace'].get('order_id_effective') or 'None'} • "
                        f"Exists: {result['trace'].get('order_exists') or 'n/a'} • "
                        f"Source: {result['trace'].get('category_source')}"
                    )
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
