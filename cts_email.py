
import pandas as pd
import glob, fitz, re
from pprint import pprint
import numpy as np
from tqdm import tqdm
from pypdf import PdfReader
import os, pickle, io, sys, json
import base64,requests, time, pytz # track CPU run time; Timezone
import cts
from datetime import datetime
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
# from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import mimetypes
# mimetypes.init()
# mimetypes.knownfiles


def gmail_authenticate():
    credentials_path = 'credentials_cts_op15.json'
    token_pickle = 'token_cts_op15.pickle'
    scopes = ['https://www.googleapis.com/auth/gmail.compose','https://www.googleapis.com/auth/gmail.readonly']
    creds = None
    # Load the previously saved credentials
    if os.path.exists(token_pickle):
        with open(token_pickle, 'rb') as token:
            creds = pickle.load(token)
    # If there are no valid credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_pickle, 'wb') as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)

def search_messages(service, query):
    result = service.users().messages().list(userId='me', q=query).execute()
    messages = []
    if 'messages' in result:
        messages.extend(result['messages'])
    while 'nextPageToken' in result:
        page_token = result['nextPageToken']
        result = service.users().messages().list(userId='me', q=query, pageToken=page_token).execute()
        if 'messages' in result:
            messages.extend(result['messages'])
    return messages


def search_msg_subject(subject,start_date,end_date):
    service = gmail_authenticate()
    query = f'subject:{subject} after:{start_date} before:{end_date}'
    # query = f'after:{start_date} before:{end_date} {keyword}'
    messages = search_messages(service, query)
    print(f"Found {len(messages)} messages between {start_date} and {end_date} with subject {subject}.")

    df_cols = ['Id','ThreadId','Unix_Time','Pacific_Time','From','To','CC','Subject','Snippet']
    df_mail = pd.DataFrame(columns=df_cols)
    i_row = 0
    for i in range(len(messages)):
        message = messages[i]
        msg = service.users().messages().get(userId='me', id=message['id']).execute()

        # Filter out messages that are not in the inbox
        if msg['labelIds'][-1]!='INBOX':
            continue
        # Check if the message has attachments
        if 'parts' not in msg['payload']:
            continue

        headers = msg['payload']['headers']
        
        has_excel = False
        for part in msg['payload']['parts']:
            if 'filename' in part and part['filename'].endswith('.xlsx'):
                has_excel = True
                break
        if not has_excel:
            continue
        df_mail.loc[i_row,'Id'] = msg['id']
        df_mail.loc[i_row,'ThreadId'] = msg['threadId']
        unix_time = int(int(msg['internalDate'])/1000)
        df_mail.loc[i_row,'Unix_Time'] = unix_time

        pst = pytz.timezone('US/Pacific')
        # utc = pytz.timezone('UTC')
        # df_mail.loc[i_row,'UTC'] = datetime.fromtimestamp(unix_time,utc).strftime("%Y-%m-%d %H:%M:%S")
        df_mail.loc[i_row,'Pacific_Time'] = datetime.fromtimestamp(unix_time,pst).strftime("%Y-%m-%d %H:%M:%S")
        
        df_mail.loc[i_row,'Snippet'] = msg['snippet']
        for header in headers:
            if header['name'] == 'From':
                df_mail.loc[i_row,'From'] = header['value']
            elif header['name'] == 'To':
                df_mail.loc[i_row,'To'] = header['value']
            elif header['name'] == 'Cc':
                df_mail.loc[i_row,'CC'] = header['value']
            elif header['name'] == 'Subject':
                df_mail.loc[i_row,'Subject'] = header['value']
        i_row += 1

    df_mail = df_mail[::-1].reset_index(drop=True)
    df_mail = df_mail.fillna('') # will be deprecated in the future
    # df_mail = df_mail.replace(np.nan, '')

    df_mail2 = df_mail.drop_duplicates(subset=['ThreadId'], keep='first')
    # for i in range (len(df_mail2)):
    #     # remove the "Re:", "Fwd:", "RE:", "FWD:", "回复:" in the subject
    #     while re.match(r'^(Re:|Fwd:|RE:|FWD:|回复:)',df_mail2.loc[i,'Subject']):
    #         df_mail2.loc[i,'Subject'] = ":".join(df_mail2.loc[i,'Subject'].split(':')[1:]).strip()
    #     print(df_mail2.loc[i,'Subject'])
    return df_mail2