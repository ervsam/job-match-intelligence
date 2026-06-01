import json
import os
import base64
from datetime import datetime, timedelta
from src.auth import get_gmail_service

# Search query — catches most job platform emails
QUERY = (
    'from:(linkedin OR indeed OR lever OR greenhouse OR ashby OR workday OR icims) '
    'OR subject:(job OR "machine learning" OR "software engineer" OR "applied scientist" OR "data scientist")'
)

def fetch_job_emails(days_back=90, max_results=200):
    service = get_gmail_service()
    
    # Build date filter
    after_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')
    query = f'{QUERY} after:{after_date}'
    
    print(f'Searching Gmail with query: {query}\n')
    
    emails = []
    page_token = None
    
    while True:
        kwargs = {'userId': 'me', 'q': query, 'maxResults': min(max_results, 500)}
        if page_token:
            kwargs['pageToken'] = page_token
        
        results = service.users().messages().list(**kwargs).execute()
        messages = results.get('messages', [])
        
        if not messages:
            break
        
        print(f'Fetching batch of {len(messages)} messages...')
        
        for msg in messages:
            metadata = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='metadata',
                metadataHeaders=['Subject', 'From', 'Date']
            ).execute()
            
            headers = {h['name']: h['value'] for h in metadata['payload']['headers']}
            
            email_record = {
                'id': msg['id'],
                'thread_id': metadata['threadId'],
                'subject': headers.get('Subject', ''),
                'from': headers.get('From', ''),
                'date': headers.get('Date', ''),
                'timestamp_ms': int(metadata['internalDate']),
            }
            emails.append(email_record)
        
        page_token = results.get('nextPageToken')
        if not page_token or len(emails) >= max_results:
            break
    
    return emails


def save_emails(emails, output_path='data/raw_emails.jsonl'):
    os.makedirs('data', exist_ok=True)
    with open(output_path, 'w') as f:
        for email in emails:
            f.write(json.dumps(email) + '\n')
    print(f'\nSaved {len(emails)} emails to {output_path}')


if __name__ == '__main__':
    emails = fetch_job_emails(days_back=90)
    save_emails(emails)
    
    # Preview
    print('\n--- Sample results ---')
    for e in emails[:5]:
        print(f"  {e['date'][:16]}  |  {e['from'][:30]:<30}  |  {e['subject'][:60]}")
