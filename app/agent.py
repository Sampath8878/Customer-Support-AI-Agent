import joblib

# Load trained model
model = joblib.load("ticket_model.pkl")

# Keyword dictionaries
delivery_keywords = ["package", "delivered", "courier", "shipment", "tracking", "transit", "shipping"]
defect_keywords = ["broken", "defective", "faulty", "damaged", "not working", "malfunction"]

# Suggested responses
SUGGESTED_RESPONSES = {
    "refund": "We’re sorry for the inconvenience. We’ve initiated your refund process. You should receive confirmation soon.",
    "delivery": "Thank you for reaching out. We are escalating your delivery issue to our logistics team. We’ll update you with tracking details shortly.",
    "defect": "We apologize for the defective product. We’ll send you a replacement right away. Please share order details if not already provided.",
    "other": "Thank you for contacting us. Your issue has been logged, and our support team will respond shortly."
}

def classify_and_respond(text):
    text_lower = text.lower()

    # Rule override for Delivery
    if any(word in text_lower for word in delivery_keywords):
        category = "delivery"
    elif any(word in text_lower for word in defect_keywords):
        category = "defect"
    else:
        category = model.predict([text])[0]

    # Suggested response
    response = SUGGESTED_RESPONSES.get(category, SUGGESTED_RESPONSES["other"])

    return {"category": category, "suggested_response": response}
