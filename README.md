# Job Match Intelligence System

A RAG-powered pipeline that ingests job postings from Gmail, ranks them by relevance to my profile using vector similarity, and generates structured hiring analysis per role using GPT-4o.

Built as a portfolio project to demonstrate end-to-end production AI/LLM system skills.

**[Live Demo →](your-streamlit-url-here)** *(add after deploy)*

---

## What it does

1. **Ingests** job-related emails from Gmail (157 postings fetched over 90 days)
2. **Embeds** each posting using OpenAI `text-embedding-3-small` (1536-dim vectors)
3. **Stores** embeddings in ChromaDB with cosine similarity indexing
4. **Ranks** postings by semantic similarity to my candidate profile
5. **Generates** structured fit analysis per job via GPT-4o: fit score, strengths, gaps, what to emphasize in interviews
6. **Evaluates** reasoning quality using an LLM-as-judge framework (relevance, faithfulness, actionability)
7. **Displays** everything in a Streamlit UI with on-demand GPT-4o analysis for any job

---

## Architecture
Gmail API
└── src/fetch_emails.py   →  data/raw_emails.jsonl      (metadata: 157 emails)
└── src/parse_emails.py   →  data/parsed_jobs.jsonl     (with body text)
OpenAI Embeddings (text-embedding-3-small)
└── src/embed.py          →  data/profile_embedding.json  (candidate profile vector)
└── src/embed_jobs.py     →  data/job_embeddings.jsonl    (157 job vectors, dim=1536)
ChromaDB (cosine similarity)
└── src/vectorstore.py    →  data/chromadb/               (persistent local store)
query() → top-N ranked jobs
GPT-4o
└── src/reason.py         →  data/scored_jobs.jsonl       (fit analysis per job)
└── src/eval.py           →  data/eval_results.jsonl      (LLM-as-judge eval scores)
Streamlit
└── app.py                →  localhost:8501               (ranked UI with analysis)

---

## Design decisions

**Why ChromaDB?**
Local, free, no infrastructure to manage. For 157–500 embeddings, a persistent local vector store is the right call — no need for Pinecone or a managed service at this scale. ChromaDB's `PersistentClient` survives restarts without re-ingestion.

**Why `text-embedding-3-small` over `text-embedding-3-large`?**
3-small produces 1536-dim vectors at ~5x lower cost with minimal quality loss for retrieval tasks at this scale. Cost for this project: under $0.01 total for all embeddings.

**Why embed subject + body together?**
The job title carries high signal (e.g. "Deep Learning Researcher" vs "ML Platform Engineer") that would be diluted if only the body was embedded. Prepending the subject to the body before embedding improves ranking quality.

**Why LLM-as-judge for eval?**
Human labeling 157 jobs for relevance isn't practical. Using GPT-4o to evaluate GPT-4o outputs introduces some bias, but scoring on specific dimensions (faithfulness, actionability) with explicit rubrics produces consistent, interpretable results. Key finding: faithfulness averaged 3/5, indicating the reasoning occasionally drifts from specific job requirements — a known RAG failure mode.

**Why not fine-tune?**
The profile and job corpus change constantly. Fine-tuning is expensive and doesn't generalize to new postings. RAG with a strong base model handles distribution shift naturally.

---

## Results

| Metric | Value |
|--------|-------|
| Emails ingested | 157 |
| Embedding dimension | 1536 |
| Top match similarity score | 0.691 (Picovoice — Deep Learning Researcher) |
| Avg GPT-4o fit score (top 10) | 8.0 / 10 |
| Avg eval relevance | 4.3 / 5 |
| Avg eval faithfulness | 3.1 / 5 |
| Avg eval actionability | 4.0 / 5 |

---

## Stack

- **Python 3.13**
- **OpenAI API** — `text-embedding-3-small` for embeddings, `gpt-4o` for reasoning and eval
- **ChromaDB** — local persistent vector store
- **Gmail API** — OAuth 2.0 for email ingestion
- **Streamlit** — UI
- **python-dotenv** — secrets management

---

## Running locally

**Prerequisites:** Python 3.10+, Gmail account, OpenAI API key with credits

```bash
git clone https://github.com/ervsam/job-match-intelligence
cd job-match-intelligence
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
OPENAI_API_KEY=sk-...

Add your Gmail `credentials.json` (see [Gmail API setup](https://developers.google.com/gmail/api/quickstart/python)).

```bash
# Run the full pipeline
PYTHONPATH=. python src/fetch_emails.py    # fetch emails
PYTHONPATH=. python src/parse_emails.py   # extract bodies
PYTHONPATH=. python src/embed.py          # embed your profile
PYTHONPATH=. python src/embed_jobs.py     # embed all jobs
PYTHONPATH=. python src/vectorstore.py   # ingest into ChromaDB + first query
PYTHONPATH=. python src/reason.py        # GPT-4o analysis on top 10
PYTHONPATH=. python src/eval.py          # evaluate reasoning quality

# Launch UI
PYTHONPATH=. streamlit run app.py
```

---

## Author

Ervin — MSc ML/AI, SFU. [LinkedIn](https://linkedin.com/in/ervinsamuel) · [GitHub](https://github.com/ervsam)