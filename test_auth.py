from src.auth import get_gmail_service

service = get_gmail_service()
results = service.users().messages().list(userId='me', maxResults=10).execute()

for msg in results.get('messages', []):
    full = service.users().messages().get(
        userId='me', id=msg['id'],
        format='metadata', metadataHeaders=['Subject']
    ).execute()
    headers = {h['name']: h['value'] for h in full['payload']['headers']}
    print(headers.get('Subject', '(no subject)'))
