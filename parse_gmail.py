import os
import re
import base64
import csv
import datetime
import numpy
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pandas as pd
from urllib.parse import urlparse
import spacy
from transformers import pipeline

# Define the scope for Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Gmail search query for job applications (modify if needed)
KEYWORDS = ["thank you for applying", "received your application", "your application has been received", 
            "your application was sent"]

nlp = spacy.load("en_core_web_sm")

# Load a text classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_email_with_nlp(subject, body):
    """Use NLP model to classify emails and extract key information."""
    text = subject + " " + body
    result = classifier(text, candidate_labels=["Job Application", "Not Job Application"])
    
    is_application = result["labels"][0] == "Job Application"
    print(f"is Application: {is_application}, Subject: {subject}")

    # Extract entities with spaCy
    doc = nlp(text)
    company_name = "Unknown"
    job_title = "Unknown"

    for ent in doc.ents:
        if ent.label_ == "ORG":  # Company name
            company_name = ent.text
        if ent.label_ == "JOB":  # Job title (not always detected)
            job_title = ent.text

    return {
        "is_application_email": is_application,
        "company_name": company_name,
        "job_title": job_title
    }

def date_to_timestamp(date_str):
    """Convert date string (MM/DD/YYYY) to Unix timestamp (seconds since epoch)."""
    dt = datetime.datetime.strptime(date_str, "%Y/%m/%d")
    return int(dt.timestamp())

def authenticate_gmail():
    """Authenticate and return the Gmail service."""
    creds = None
    token_file = 'token.json'

    # Load credentials from token.json if available
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    
    # If credentials are invalid, go through OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save credentials for future use
        with open(token_file, 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

# def search_emails(service, start_date, end_date):
#     """Search Gmail for job application emails within a date range."""
#     start_timestamp = date_to_timestamp(start_date)
#     end_timestamp = date_to_timestamp(end_date)
#     query = f"after:{start_timestamp} before:{end_timestamp}"
#     print(query)
#     results = service.users().messages().list(userId='me', q=query).execute()
#     return results.get('messages', [])
def search_emails(service, start_date, end_date):
    """Fetch emails between start_date and end_date with pagination."""
    start_timestamp = date_to_timestamp(start_date)
    end_timestamp = date_to_timestamp(end_date)

    query = f"category:primary after:{start_timestamp} before:{end_timestamp}"
    emails = []
    next_page_token = None

    while True:
        response = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=100,  # Gmail API max is 100 per request
            pageToken=next_page_token  # Handle pagination
        ).execute()

        if 'messages' in response:
            emails.extend(response['messages'])

        next_page_token = response.get('nextPageToken')  # Get next page
        if not next_page_token:  # Stop if no more emails
            break

    return emails  # Return all emails collected


def get_email_details(service, message_id):
    """Retrieve email details by message ID."""
    email = service.users().messages().get(userId='me', id=message_id, format='full').execute()
    headers = email.get('payload', {}).get('headers', [])
    
    # Extract subject and date
    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
    date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown Date')

    # Decode email body (if available)
    body = ''
    if 'parts' in email['payload']:
        for part in email['payload']['parts']:
            if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                break
    
    return date, subject, body, headers


def extract_company_name(subject, body):
    """Extract company name from email subject or body using regex patterns."""
    text = subject + " " + body  # Combine subject and body for better matching

    # Common patterns where company names appear in job application emails
    patterns = [
        r"your application was sent to ([A-Z][a-zA-Z0-9&.\- ]+)",
        r"we have received your application at ([A-Z][a-zA-Z0-9&.\- ]+)",
        r"thank you for applying to ([A-Z][a-zA-Z0-9&.\- ]+)",
        r"thank you for your interest in ([A-Z][a-zA-Z0-9&.\- ]+)",
        r"application received at ([A-Z][a-zA-Z0-9&.\- ]+)",
        r"your application has been submitted to ([A-Z][a-zA-Z0-9&.\- ]+)",
        r"you have applied to ([A-Z][a-zA-Z0-9&.\- ]+)",
        r"([A-Z][a-zA-Z0-9&.\-]+) has received your application"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return "Unknown"



def extract_company_name_from_email(headers):
    """Extract company name from sender's email domain."""
    sender_email = None

    # Find the 'From' field in the email headers
    for header in headers:
        if header['name'].lower() == 'from':
            sender_email = header['value']
            break

    if sender_email:
        # Extract domain from email address
        match = re.search(r'@([\w.-]+)', sender_email)
        if match:
            domain = match.group(1)  # Extract domain (e.g., 'google.com')

            # Remove common suffixes like .com, .io, .ai, etc.
            company_name = domain.split('.')[0].capitalize()
            return company_name
    
    return None


def extract_position(subject, body):
    """Extract job position from email subject or body."""
    position_match = re.search(r"(?i)for the (.*?) position|applying for (.*?) role", subject + " " + body)
    return position_match.group(1) if position_match else "Unknown"

def process_emails(start_date, end_date):
    """Process emails and save results to CSV."""
    service = authenticate_gmail()
    emails = search_emails(service, start_date, end_date)
    #print(emails)
    
    records = []
    for email in emails:
        date, subject, body, headers = get_email_details(service, email['id'])

        # Check if email contains job application keywords
        result = classify_email_with_nlp(subject, body)
        if result["is_application_email"]:
            #print(f"subject: {subject}")
            # company = extract_company_name_from_email(headers)
            # if not company:
            #     company = extract_company_name(subject, body)
            company = result["company_name"]
            #position = extract_position(subject, body)
            position = result["job_title"]
            records.append([date, company, position])

    # # Save to CSV or Excel
    df = pd.DataFrame(records, columns=['Date', 'Company Name', 'Position'])
    df.to_csv('job_applications.csv', index=False)
    df.to_excel('job_applications.xlsx', index=False)

    print(f"Saved {len(records)} job applications to job_applications.csv and job_applications.xlsx")

# Define the date range (YYYY/MM/DD)
start_date = "2025/02/21"
end_date = "2025/02/22"

# Run the script
process_emails(start_date, end_date)
