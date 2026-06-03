import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from src.vectorstore import query
from src.embed import embed_text

load_dotenv()
client = OpenAI()


SYSTEM_PROMPT = """You are a career advisor helping a candidate evaluate job fit.
Given a job posting and a candidate profile, provide a structured analysis.
Be specific and concrete — reference actual details from both the job and profile.
Keep each section brief but actionable."""

USER_PROMPT_TEMPLATE = """## Candidate Profile
{profile}

## Job Posting
Title: {subject}
Company: {company}
Description:
{body}

## Analysis Request
Provide a JSON response with exactly these keys:
- "fit_score": integer 1-10 (how well this candidate fits this role)
- "fit_summary": 2-3 sentence summary of overall fit
- "strengths": list of 3 specific reasons this candidate is a good match
- "gaps": list of up to 3 skill/experience gaps (empty list if none)
- "emphasize": list of 3 specific things to highlight in a cover letter or interview
"""


def load_profile():
    with open("data/profile.md") as f:
        return f.read()

def load_parsed_jobs():
    jobs = {}
    with open("data/parsed_jobs.jsonl") as f:
        for line in f:
            job = json.loads(line)
            jobs[job["id"]] = job
    return jobs


def analyze_job(profile: str, job: dict) -> dict:
    prompt = USER_PROMPT_TEMPLATE.format(
        profile=profile,
        subject=job.get("subject", ""),
        company=job.get("company", ""),
        body=job.get("body", "")[:3000]  # cap to avoid token limits
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3
    )

    return json.loads(response.choices[0].message.content)


def run_reasoning(n_jobs: int = 10, output_path: str = "data/scored_jobs.jsonl"):
    profile = load_profile()
    parsed_jobs = load_parsed_jobs()

    # Get top-N jobs from ChromaDB
    with open("data/profile_embedding.json") as f:
        profile_data = json.load(f)

    print(f"Querying top {n_jobs} jobs from ChromaDB...")
    top_jobs = query(profile_data["embedding"], n_results=n_jobs)

    print(f"Running GPT-4o analysis on {len(top_jobs)} jobs...\n")

    results = []
    for i, hit in enumerate(top_jobs, 1):
        # Find the full job body from parsed_jobs using subject match
        matched_job = next(
            (j for j in parsed_jobs.values() if j["subject"] == hit["subject"]),
            None
        )
        if not matched_job:
            print(f"  [{i}] Skipping — body not found for: {hit['subject'][:50]}")
            continue

        print(f"  [{i}/{n_jobs}] Analyzing: {hit['subject'][:60]}")

        try:
            analysis = analyze_job(profile, matched_job)
            result = {
                "subject": hit["subject"],
                "company": hit["company"],
                "date": hit["date"],
                "similarity_score": hit["score"],
                "analysis": analysis
            }
            results.append(result)
            time.sleep(0.5)  # rate limiting
        except Exception as e:
            print(f"    Error: {e}")
            continue

    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. {len(results)} jobs analyzed. Saved to {output_path}")

    # Preview top 3
    print("\n--- Top 3 Results ---\n")
    for r in results[:3]:
        a = r["analysis"]
        print(f"{'='*60}")
        print(f"{r['company']} — {r['subject']}")
        print(f"Similarity: {r['similarity_score']} | Fit score: {a.get('fit_score')}/10")
        print(f"Summary: {a.get('fit_summary')}")
        print(f"Emphasize: {a.get('emphasize')}")
        print()


if __name__ == "__main__":
    run_reasoning()
