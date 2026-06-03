import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in one API call."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


def embed_jobs(
    parsed_path='data/parsed_jobs.jsonl',
    output_path='data/job_embeddings.jsonl'
):
    jobs = []
    with open(parsed_path) as f:
        for line in f:
            jobs.append(json.loads(line))

    print(f'Embedding {len(jobs)} jobs in batches...')

    os.makedirs('data', exist_ok=True)

    batch_size = 50  # OpenAI allows up to 2048 inputs per call, 50 is safe
    results = []

    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        texts = [f"{job['subject']}\n\n{job['body']}" for job in batch]

        embeddings = embed_batch(texts)

        for job, embedding in zip(batch, embeddings):
            results.append({
                'id': job['id'],
                'subject': job['subject'],
                'company': job['company'],
                'date': job['date'],
                'embedding': embedding
            })

        print(f'  {min(i + batch_size, len(jobs))}/{len(jobs)} done')
        time.sleep(0.5)  # gentle rate limiting

    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f'\nDone. {len(results)} embeddings saved to {output_path}')


if __name__ == '__main__':
    embed_jobs()
