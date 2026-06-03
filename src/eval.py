import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

EVAL_PROMPT = """You are evaluating the quality of an AI system that matches job postings to a candidate profile and generates hiring advice.

## Candidate Profile
{profile}

## Job Posting
Title: {subject}
Company: {company}
Description:
{body}

## AI-Generated Analysis
Fit Score: {fit_score}/10
Fit Summary: {fit_summary}
Strengths: {strengths}
Gaps: {gaps}
What to Emphasize: {emphasize}

## Evaluation Task
Score the AI analysis on three dimensions. Return a JSON object with exactly these keys:

- "relevance_score": integer 1-5 — Is this job genuinely relevant to the candidate? (1=totally wrong match, 5=excellent match)
- "relevance_reason": one sentence explaining the score

- "faithfulness_score": integer 1-5 — Does the analysis accurately reflect the actual job description? Are the strengths/gaps grounded in real job requirements, or hallucinated? (1=made up details, 5=fully grounded)
- "faithfulness_reason": one sentence explaining the score

- "actionability_score": integer 1-5 — Is the "emphasize" advice specific and useful, or generic? (1=could apply to any job, 5=highly specific to this role and candidate)
- "actionability_reason": one sentence explaining the score

- "overall_score": float — average of the three scores, rounded to 1 decimal place
"""


def eval_single(profile: str, job: dict, analysis: dict) -> dict:
    prompt = EVAL_PROMPT.format(
        profile=profile[:2000],
        subject=job.get("subject", ""),
        company=job.get("company", ""),
        body=job.get("body", "")[:2000],
        fit_score=analysis.get("fit_score"),
        fit_summary=analysis.get("fit_summary"),
        strengths=analysis.get("strengths"),
        gaps=analysis.get("gaps"),
        emphasize=analysis.get("emphasize"),
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return json.loads(response.choices[0].message.content)


def run_eval(
    scored_path="data/scored_jobs.jsonl",
    parsed_path="data/parsed_jobs.jsonl",
    output_path="data/eval_results.jsonl",
    profile_path="data/profile.md"
):
    with open(profile_path) as f:
        profile = f.read()

    # Load parsed jobs for body text
    parsed_jobs = {}
    with open(parsed_path) as f:
        for line in f:
            job = json.loads(line)
            parsed_jobs[job["subject"]] = job

    # Load scored jobs
    scored_jobs = []
    with open(scored_path) as f:
        for line in f:
            scored_jobs.append(json.loads(line))

    print(f"Evaluating {len(scored_jobs)} scored jobs...\n")

    results = []
    for i, scored in enumerate(scored_jobs, 1):
        subject = scored["subject"]
        analysis = scored["analysis"]
        job = parsed_jobs.get(subject, scored)  # fall back to scored data if no body

        print(f"  [{i}/{len(scored_jobs)}] {scored['company']} — {subject[:50]}")

        try:
            eval_result = eval_single(profile, job, analysis)
            results.append({
                "subject": subject,
                "company": scored["company"],
                "similarity_score": scored["similarity_score"],
                "fit_score": analysis.get("fit_score"),
                "eval": eval_result
            })
            time.sleep(0.5)
        except Exception as e:
            print(f"    Error: {e}")
            continue

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. Results saved to {output_path}")

    # Summary table
    print("\n--- Eval Summary ---\n")
    print(f"{'Company':<25} {'Role':<35} {'Sim':>5} {'Fit':>4} {'Rel':>4} {'Fth':>4} {'Act':>4} {'Avg':>5}")
    print("-" * 90)
    for r in results:
        e = r["eval"]
        print(
            f"{r['company']:<25} "
            f"{r['subject'][:34]:<35} "
            f"{r['similarity_score']:>5.3f} "
            f"{r['fit_score']:>4} "
            f"{e.get('relevance_score', '?'):>4} "
            f"{e.get('faithfulness_score', '?'):>4} "
            f"{e.get('actionability_score', '?'):>4} "
            f"{e.get('overall_score', '?'):>5}"
        )


if __name__ == "__main__":
    run_eval()
