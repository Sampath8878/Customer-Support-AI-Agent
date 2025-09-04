# Customer-Support-AI-Agent

Single-agent demo that:

Summarizes a customer ticket (LLM via Ollama)

Classifies it into refund / delivery / defect / other (hybrid: rules + LLM few-shot)

Drafts a short reply (4–5 sentences, no greeting)

(Optional) Retrieves policy text from kb/ (Chroma vector index)

Works fully offline after installing dependencies and pulling a local LLM with Ollama.

1) Prerequisites

Windows 10/11 (works on macOS/Linux too; commands shown for Windows PowerShell)

Python 3.11 or 3.12
Verify: python --version

Ollama (local LLM runtime): https://ollama.com/download

Verify: ollama --version

Recommended LLM: llama3.2:3b (~2 GB). Good speed on typical laptops.

2) First-time setup

Open PowerShell in the project folder (the folder that contains app/, kb/, ui/, requirements.txt).

2.1 Pull and start the LLM
# Pull the model (first time only)
ollama pull llama3.2:3b

# Start Ollama server (keep this window open)
ollama serve


If you see “Only one usage of each socket address …” it means Ollama is already running. That’s OK—leave it running and skip ollama serve.

2.2 Create a virtual environment and install packages

Open a new PowerShell window in the project folder:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2.3 (Optional) Build the knowledge base index

If the project includes kb/ markdown files:

python app\index_kb.py


This creates chroma_db/ for retrieval.

3) Run the API

With the venv activated:

uvicorn app.main:app --reload --port 8000


Open http://127.0.0.1:8000/docs
 to use Swagger UI.

Use POST /analyze_ticket

Paste any test text (see samples below)

Click Execute

Seeing GET / 404 or GET /favicon.ico 404 in the console is normal; the root path is not used. Always open /docs.

4) (Optional) Run the UI

If you want a simple front-end:

Confirm ui/app.py has:

API_URL = "http://127.0.0.1:8000/analyze_ticket"


In a new PowerShell window with venv activated:

streamlit run ui\app.py


Your browser will open the Streamlit app.

5) Test samples (copy-paste)
Refund

I bought the jacket last week but it doesn’t fit. I’d like a refund.

Delivery

My package shows delivered but nothing arrived at my address.

The courier left my parcel at the wrong house. Can you fix this?

It’s been 10 days and the shipment is still stuck in transit.

Defect

The phone arrived with a cracked screen out of the box.

Other

What are your store hours on Sundays?

6) Example API calls
cURL (Windows PowerShell)
curl -X POST http://127.0.0.1:8000/analyze_ticket `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"My package shows delivered but nothing arrived at my address.\"}"

Swagger UI

Open http://127.0.0.1:8000/docs

Choose POST /analyze_ticket

Set body:

{"text": "The phone arrived with a cracked screen out of the box."}


Execute

Response fields:

summary – 1-sentence LLM summary

category – refund / delivery / defect / other

suggested_response – short, no greeting

7) (Optional) Retrain the classical model

If tickets.csv and train_classifier.py are included and you want to generate models/ticket_model.pkl:

venv\Scripts\activate
python train_classifier.py


The running agent already uses hybrid rules + LLM for classification. The sklearn model is optional and can be added as an ensemble later.

8) Project structure (what you should see)
project-root/
├─ app/
│  ├─ main.py          # FastAPI app (summary + hybrid classification + reply)
│  ├─ index_kb.py      # builds chroma_db/ from kb/ markdown files
│  └─ schemas.py       # Pydantic models
├─ kb/                 # policy/FAQ markdown files (used for retrieval)
├─ models/
│  └─ ticket_model.pkl # (optional) sklearn pipeline if you retrain
├─ ui/
│  └─ app.py           # Streamlit UI (optional)
├─ requirements.txt
├─ tickets.csv         # (optional) training/eval data
└─ train_classifier.py # (optional) script to train sklearn model


Do not commit/send venv/ or __pycache__/.
chroma_db/ can be rebuilt any time with python app\index_kb.py.

9) Troubleshooting

A) Swagger loads, but responses fail with “ConnectError” to 11434

Ollama not running or wrong port.

Start: ollama serve

Verify: Invoke-WebRequest http://127.0.0.1:11434/api/tags (should return JSON)

B) “Only one usage of each socket address (11434)”

Ollama is already running. Don’t start it again.

C) / returns 404

Expected. Use /docs.

D) Very long reply

The agent already limits to 4–5 sentences and strips greetings. If it still gets wordy, just resubmit; local LLMs can be a bit chatty on the first warm-up call.

E) UI can’t reach API

Check API_URL in ui/app.py is http://127.0.0.1:8000/analyze_ticket.

Ensure the API terminal shows Uvicorn running on http://127.0.0.1:8000.

F) Change Ollama host/port

If Ollama runs elsewhere, set environment variable before starting the API:

$env:OLLAMA_HOST = "http://127.0.0.1:11434"
uvicorn app.main:app --reload --port 8000

10) One-click helper scripts (optional)

Create these .bat files in the project root for convenience.

pull_model.bat

@echo off
ollama pull llama3.2:3b
pause


start_ollama.bat

@echo off
echo If you see "port in use", Ollama is already running (that's OK).
ollama serve


run_api.bat

@echo off
call venv\Scripts\activate
uvicorn app.main:app --reload --port 8000

11) Notes

The classifier uses hybrid logic: fast rules (shipping vs. defect vs. refund cues) + LLM (few-shot JSON prompt, temperature 0) for consistent labels.

Replies are LLM-generated with strict formatting and a short fallback template per category.

You can extend the KB by dropping more .md files into kb/ and running python app\index_kb.py.
