import json
import os
import base64
import re
from src.auth import get_gmail_service


def get_message_body(payload):
    """Recursively extract plain text body from Gmail message payload."""
    if payload.get('mimeType') == 'text/plain':
        data = payload.get('body', {}).get('data', '')
        if data:
            return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

    if payload.get('mimeType') == 'text/html':
        data = payload.get('body', {}).get('data', '')
        if data:
            html = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            # Strip HTML tags
            return re.sub(r'<[^>]+>', ' ', html)

    # Multipart — recurse into parts
    for part in payload.get('parts', []):
        result = get_message_body(part)
        if result:
            return result

    return ''


def clean_text(text):
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def extract_company(from_field, subject):
    """Best-effort company extraction from sender or subject."""
    # Try subject patterns like "X @ Company" or "X at Company"
    match = re.search(r'(?:@|at)\s+([A-Z][A-Za-z0-9\s&.,]+?)(?:\s*[|\-,]|$)', subject)
    if match:
        return match.group(1).strip()

    # Fall back to sender name (before the email address)
    match = re.match(r'^(.+?)\s*<', from_field)
    if match:
        name = match.group(1).strip()
        # Filter out generic senders
        generic = {'LinkedIn', 'Indeed', 'Glassdoor', 'ZipRecruiter', 'Lever', 'Greenhouse'}
        if name not in generic:
            return name

    return ''


def parse_emails(
    raw_path='data/raw_emails.jsonl',
    output_path='data/parsed_jobs.jsonl',
    min_body_length=100
):
    service = get_gmail_service()

    raw_emails = []
    with open(raw_path) as f:
        for line in f:
            raw_emails.append(json.loads(line))

    print(f'Parsing {len(raw_emails)} emails...')

    parsed = []
    skipped = 0

    for i, email in enumerate(raw_emails):
        if i % 20 == 0:
            print(f'  {i}/{len(raw_emails)}...')

        try:
            full = service.users().messages().get(
                userId='me', id=email['id'], format='full'
            ).execute()

            body = get_message_body(full['payload'])
            body = clean_text(body)

            if len(body) < min_body_length:
                skipped += 1
                continue

            record = {
                'id': email['id'],
                'thread_id': email['thread_id'],
                'subject': email['subject'],
                'from': email['from'],
                'date': email['date'],
                'timestamp_ms': email['timestamp_ms'],
                'company': extract_company(email['from'], email['subject']),
                'body': body[:8000],  # cap at 8k chars — plenty for embedding
            }
            parsed.append(record)

        except Exception as e:
            print(f'  Error on {email["id"]}: {e}')
            skipped += 1
            continue

    os.makedirs('data', exist_ok=True)
    with open(output_path, 'w') as f:
        for record in parsed:
            f.write(json.dumps(record) + '\n')

    print(f'\nDone. {len(parsed)} parsed, {skipped} skipped.')
    print(f'Saved to {output_path}')

    # Preview
    print('\n--- Sample ---')
    for r in parsed[:3]:
        print(f"  {r['company'] or '?':<25} | {r['subject'][:60]}")
        print(f"  body preview: {r['body'][:80]}...")
        print()


if __name__ == '__main__':
    parse_emails()
