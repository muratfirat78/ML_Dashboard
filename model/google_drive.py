import os
import random
from datetime import datetime

from google.colab import auth
from controller.controller import Controller
from googleapiclient.discovery import build
import googleapiclient.http
import random
import numpy as np
import io

class GoogleDrive:
    # In online mode this class is used, in offline mode the local_drive.py file is used
    # This class contains the functions to write to Google Drive and save the users performances
    def __init__(self):
        auth.authenticate_user()
        self.drive_service = build('drive', 'v3')
        self.folderid = '1pN1jFF5tcDxrfQRdTQrHDNvvxRLRbI69'
        self.userid = None

    def get_folder(self, userid):
        #get the folder id of the Google Drive share
        query = f"name='{userid}' and '{self.folderid}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.drive_service.files().list(q=query, fields="files(id)").execute()
        files = results.get('files', [])

        if files:
            return files[0]['id']
    
    def download(self, file_id, file_name, userid):
        #download a file
        request = self.drive_service.files().get_media(fileId=file_id)

        #create the userid folder if not existing
        path = os.path.join('drive', str(userid))
        os.makedirs(path, exist_ok=True)

        file_path = os.path.join(path, file_name)

        fh = io.FileIO(f"{file_path}", 'wb')
        downloader = googleapiclient.http.MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

    def get_performances(self,userid):
        #this function goes through the files in the shared Google Drive and downloads the files in the folder of the user
        folderid = self.get_folder(userid)
        query = f"'{folderid}' in parents and trashed=false"
        results = self.drive_service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        
        if not os.path.exists('/content/ML_Dashboard/drive/' + str(userid)):
            os.makedirs('/content/ML_Dashboard/drive/' + str(userid))
            
        for file in files:
            if not os.path.exists('./drive/' + userid + '/' + file['name']):
                self.download(file['id'], file['name'], userid)

    def login_correct(self,userid):
        #This function returns if the user id exists in the Google Drive
        query = f"name='{userid}' and '{self.folderid}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.drive_service.files().list(q=query, fields="files(id)").execute()
        if len(results['files']) > 0:
            return True
        else:
            return False

    def register(self):
        #This function creates a user account by creating a user folder in the shared Google Drive
        if self.userid !=None:
            return "User already exists, your username is: " + self.userid

        query = f"'{self.folderid}' in parents and mimeType='application/vnd.google-apps.folder' and trashed = false and 'me' in owners"
        results = self.drive_service.files().list(q=query,pageSize=1000,fields='files(id, name)').execute()
        files = results.get('files', [])

        if len(files) > 0:
            return "User exists, your username is: " + files[0]['name']
        else:
            userid = None
            while True:
                userid = str(random.randint(10000, 99999))
                query2 = f"name='{userid}' and '{self.folderid}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                results2 = self.drive_service.files().list(q=query2,pageSize=1000,fields="files(name)").execute()
                files2 = results2.get('files', [])
                if len(files2) > 0:
                    continue
                else:
                    file_metadata = {
                    'name': userid,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [self.folderid]
                    }
                    file = self.drive_service.files().create(body=file_metadata, fields='id').execute()
                    self.userid = userid
                    return "User created, your userid is: " + userid
    

    def to_serializable(self, obj):
        # This function is used to convert different type to the correct JSON format
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self.to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.to_serializable(v) for v in obj)
        else:
            return obj


    def upload_log(self, result, userid, timestamp):
        #upload the performance to the Google Drive
        with open('./drive/'+ userid + '/' + timestamp +
                    '.txt', 'w') as f:
            f.write(str(self.to_serializable(result)))

        folderid = self.get_folder(userid)

        #see if the file already exists
        query = f"name='{timestamp}.txt' and '{folderid}' in parents and trashed = false"
        response = self.drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        files = response.get('files', [])

        media = googleapiclient.http.MediaFileUpload('./drive/'+ userid + '/' + timestamp + '.txt', mimetype="text/plain", resumable=True)

        if files:
            #file already exists, overwrite
            file_id = files[0]['id']
            uploaded_file = self.drive_service.files().update(
                fileId=file_id,
                media_body=media
            ).execute()
        else:
            #file does not exist yet, create
            file_metadata = {
            "name": timestamp + ".txt",
            "parents": [folderid]
            }
            
            uploaded_file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id"
            ).execute()