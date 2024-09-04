from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os, pickle
import base64
import re
from bm_email import clean_email
import pandas as pd

import numpy as np
from tqdm import tqdm
import time
import datetime
import zipfile
import glob
from sqlalchemy import create_engine
import sqlite3


def gmail_authenticate(credentials_path, token_pickle, scopes):
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



def internal_date_to_str(internal_date):
    # Convert Unix Time to datetime object
    unix_time = int(internal_date) / 1000  # Convert milliseconds to seconds
    date_obj = datetime.datetime.fromtimestamp(unix_time)
    # Convert datetime object to UTC
    date_utc = date_obj.astimezone(datetime.timezone.utc)
    return str(date_utc)

def main():
    # Path to your credentials.json
    prog_dir = '/Users/ljx/Documents/workspace/python/bytemld/email/'
    credentials_path = prog_dir + 'credentials_cts.json'
    # Token pickle file stores the user's access and refresh tokens.
    token_pickle = prog_dir + 'token_cts.pickle'
    # Scopes required by the application
    scopes = ['https://www.googleapis.com/auth/gmail.readonly']

    # Authenticate and create the service
    service = gmail_authenticate(credentials_path, token_pickle, scopes)
    root_dir = '/Users/ljx/Documents/ByteMelodies/data/cts/'
    db_path = root_dir + 'pods/pod.db'
    engine = create_engine(f'sqlite:///{db_path}')
    # check if the database exists
    if not os.path.exists(db_path):
        month_begin = datetime.datetime(2024,3, 1, 0, 0, 0, 0, datetime.timezone.utc)
        last_Unix_Time = int(month_begin.timestamp()*1000)
    else:
        conn = sqlite3.connect(db_path)
        query = f"SELECT Unix_Time FROM emails ORDER BY ROWID DESC LIMIT 1"
        df = pd.read_sql_query(query, conn)
        last_Unix_Time = int(df['Unix_Time'][0])
        conn.close()

    last_UTC = datetime.datetime.fromtimestamp(last_Unix_Time/1000, datetime.timezone.utc)
    subject = "POD"
    start_date = last_UTC.strftime('%Y/%m/%d %H:%M:%S')[0:10] #"2024/05/14"
    end_date = "2099/04/01"
    #sender_email = "USPS Informed Delivery <USPSInformeddelivery@email.informeddelivery.usps.com>"
    #query = f'subject:{subject} after:{start_date} before:{end_date}'
    query = f'subject:{subject} after:{start_date} before:{end_date}'
    #query = f'after:{start_date} before:{end_date} from:{sender_email}'
    messages = search_messages(service, query)






    df_cols = ['Id','ThreadId','Unix_Time','Date_UTC','From','To','CC','Subject','Snippet']
    df2 = pd.DataFrame(columns=df_cols)
    for i in range(len(messages)):
        message = messages[i]
        msg = service.users().messages().get(userId='me', id=message['id']).execute()

        # Filter out messages that are not in the inbox
        if msg['labelIds'][-1]!='INBOX':
            continue
        headers = msg['payload']['headers']
        msg_info = ['']*9
        msg_info[0]=msg['id']
        msg_info[1]=msg['threadId']
        msg_info[2]=msg['internalDate']
        msg_info[3]=internal_date_to_str(msg['internalDate'])
        msg_info[8]=msg['snippet']
        for header in headers:
            if header['name'] == 'From':
                msg_info[4] = header['value']
            elif header['name'] == 'To':
                msg_info[5] = header['value']
            elif header['name'] == 'Cc':
                msg_info[6] = header['value']
            elif header['name'] == 'Subject':
                msg_info[7] = header['value']
            
        df2.loc[len(df2)] = msg_info

    df2['Unix_Time'] = df2['Unix_Time'].astype(np.int64)
    df2 = df2[df2['Unix_Time']>last_Unix_Time]
    # rearrange the index of df2, start from 0, and drop the duplicated emails
    #df2 = df2.reset_index(drop=True)
    df2 = df2[::-1].reset_index(drop=True)

    # Update df2 to include the path for the attachments
    for i in range(len(df2)):
        dt = df2['Date_UTC'][i]
        df2.loc[i,'Filedir'] = dt[0:4] + '/' + dt[5:7] + '/' + dt[8:10]+'_'+df2['Id'][i]
    engine = create_engine(f'sqlite:///{db_path}')
    df2.to_sql('emails', con=engine, if_exists='append', index=False)

    file_log = root_dir + 'pods/log.txt'
    now = datetime.datetime.now()
    if len(df2)>0:
        with open(file_log, 'a') as f:
            f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')}: Found {len(df2)} new emails between "+start_date+" and "+end_date + ". Written into Database\n")

    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')}: Found {len(df2)} new emails between "+start_date+" and "+end_date + ". Written into Database\n")





    #parts = message['payload']['parts']
    def email_part(parts,dir_email,msgId):
        for part in parts:
            if part['mimeType'] == 'multipart/alternative' or part['mimeType'] == 'multipart/mixed':
                email_part(part['parts'],dir_email,msgId)
            elif part['mimeType'] == 'text/plain':
                data = part['body']['data']
                body = base64.urlsafe_b64decode(data.encode('ASCII')).decode('utf-8')
                body = clean_email(body)
                #print(body)
                #continue
            elif part['filename']:  # This part has a filename, so it might be an attachment
                #print(part['mimeType'],part['filename'])
                #if 'application' in part['mimeType']:
                attachment_id = part['body']['attachmentId']
                attachment = service.users().messages().attachments().get(userId='me', messageId=msgId, id=attachment_id).execute()

                # Decode the attachment
                file_data = base64.urlsafe_b64decode(attachment['data'].encode('UTF-8'))
                
                # Save the attachment to a file
                filepath = dir_email + part['filename']

                if not os.path.exists(dir_email):
                    os.makedirs(dir_email)
                with open(filepath, 'wb') as f:
                    f.write(file_data)
                # filepath2 = root_dir + part['filename']
                # with open(filepath2, 'wb') as f:
                #     f.write(file_data)
                
                #print(f"attachment {part['filename']} saved to {filepath}")
        return None






    for i in range (len(df2)):
        msgId = df2['Id'][i]
        dir_email = root_dir + 'pods/' + df2['Filedir'][i] + '/'
        message = service.users().messages().get(userId='me', id=msgId,format='full' ).execute()
        parts = message['payload']['parts']
        email_part(parts,dir_email,msgId)
        



    import shutil
    df_attachment = pd.DataFrame(columns=['EmailId','Filedir','Filename','Filesize','Filetype'])
    for i in range(len(df2)):
        dir_email = df2['Filedir'][i] + '/'
        all_files = glob.glob(root_dir + 'pods/' + dir_email + '*')
        if len(all_files)==0:
            continue

        for file in all_files:
            if file[-4:]=='.zip':
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(root_dir + 'pods/' + dir_email)
                    now = datetime.datetime.now()
                    with open(file_log, 'a') as f:
                        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')}: extracted the zip file: " + file+'\n')
                        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')}: removed the zip file: " + file+'\n')

                    macosx_dir = os.path.join(root_dir + 'pods/' + dir_email, "__MACOSX")
                    if os.path.exists(macosx_dir):
                        #print(macosx_dir)
                        shutil.rmtree(macosx_dir)
                os.remove(file)

        # Now we have extracted all the files in the email    
        all_files = glob.glob(root_dir + 'pods/' + dir_email + '*')
        for filepath in all_files:
            filename = filepath.split('/')[-1]
            file_size = os.path.getsize(filepath)
            file_type = filename.split('.')[-1]
            df_attachment.loc[len(df_attachment)] =  df2.loc[i,["Id","Filedir"]].tolist() + [filename] + [file_size] + [file_type]

    df_attachment.to_sql('files', con=engine, if_exists='append', index=False)
    print("POD files downloaded from emails and information written into Database")

if __name__ == '__main__':
    main()