import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def embed_text(text: str) -> list[float]:
    """Embed a single string using OpenAI text-embedding-3-small."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def embed_profile(profile_path='data/profile.md', output_path='data/profile_embedding.json'):
    with open(profile_path) as f:
        profile_text = f.read()

    print(f'Embedding profile ({len(profile_text)} chars)...')
    embedding = embed_text(profile_text)

    with open(output_path, 'w') as f:
        json.dump({'text': profile_text, 'embedding': embedding}, f)

    print(f'Done. Embedding dim: {len(embedding)}')
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    embed_profile()
