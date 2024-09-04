from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import io
from googleapiclient.http import MediaIoBaseDownload
from datetime import datetime, timezone

def sync_gdrive_folder_to_local(folder_id,download_path, creds):
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    try:
        service = build("drive", "v3", credentials=creds)
        query = f"'{folder_id}' in parents"
        #results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
        results = service.files().list(q=query, fields="*").execute()
        result2 = service.files().get(fileId=folder_id).execute()
        items = results.get('files', [])

        if not items:
            print('No files found in ', result2['name'])
        else:
            print('Files in :', result2['name'])
        for item in items:
            file_id = item['id']
            file_name = item['name']
            ModifiedTime = item['modifiedTime']
            file_type = item['mimeType']
            date_format = '%Y-%m-%dT%H:%M:%S.%fZ' #'%Y-%m-%dT%H:%M:%S'
            time_stamp = datetime.strptime(ModifiedTime, date_format).replace(tzinfo=timezone.utc)

            if file_type == 'application/vnd.google-apps.folder':
                sync_gdrive_folder_to_local(file_id,os.path.join(download_path, file_name),creds)
            else:
                file_path = os.path.join(download_path, file_name)
                if not os.path.exists(file_path) or os.path.getmtime(file_path) < time_stamp.timestamp():
                    print(f"{item['name']} ({item['id']})")
                    request = service.files().get_media(fileId=file_id)
                    file_path = os.path.join(download_path, file_name)
                    fh = io.FileIO(file_path, 'wb')
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                    access_time = os.path.getatime(file_path)
                    os.utime(file_path, (access_time, time_stamp.timestamp()))
                    print(f"{file_name} downloaded to {file_path}")
    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f"An error occurred: {error}")

def main():
    #SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]

    SCOPES = ['https://www.googleapis.com/auth/drive','https://www.googleapis.com/auth/drive.readonly','https://www.googleapis.com/auth/drive.file']
    # SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    
    dir1 = "/Users/ljx/Documents/workspace/python/bytemld/email/"
    token_path = dir1 + "token_gdrive_jli.json"
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(dir1 + "credential_gdrive.json", SCOPES)
            creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open(token_path, "w") as token:
        token.write(creds.to_json())

    download_path = '/Users/ljx/Documents/ByteMelodies/data/cts/bols/'
    folder_id = '1MDsrkkPh8I7bLr292PYN_dJMRzW_nXF4'
    sync_gdrive_folder_to_local(folder_id,download_path,creds)

if __name__ == '__main__':
    main()
