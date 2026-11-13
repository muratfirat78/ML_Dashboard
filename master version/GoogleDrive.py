from googleapiclient.discovery import build
import googleapiclient.http
import os
import io
import datetime
import logging


class GoogleDrive:
    # In online mode this class is used, in offline mode the LocalDrive.py file is used.
    # This class downloads every student's saved files from the shared Drive folder
    # so the Overview can be built from them.
    def __init__(self, visualManager=None):
        self.drive_service = build('drive', 'v3')
        self.folderid = '16GeTq4s_YnuVvlebvQSqnX8PCj_cRa_w'
        self.visualManager = visualManager
        logging.getLogger("google_auth_httplib2").setLevel(logging.ERROR)

    def update_status(self, text):
        if self.visualManager is not None:
            self.visualManager.download_status.value = text

    def download(self, file_id, file_name, save_path):
        os.makedirs(save_path, exist_ok=True)
        request = self.drive_service.files().get_media(fileId=file_id)
        file_path = os.path.join(save_path, file_name)
        fh = io.FileIO(file_path, 'wb')
        downloader = googleapiclient.http.MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

    def get_performances_master(self):
        # go through every student folder in the shared Drive folder and download
        # any .txt/.json save that isn't already present locally, or has changed
        query = f"'{self.folderid}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.drive_service.files().list(q=query, fields="files(id, name)").execute()
        folders = results.get('files', [])

        save_root = '../drive'
        os.makedirs(save_root, exist_ok=True)

        total_files = 0
        downloaded_files = 0

        for folder in folders:
            student_id = folder['name']
            save_path = os.path.join(save_root, student_id)

            query2 = f"'{folder['id']}' in parents and trashed=false"
            results2 = self.drive_service.files().list(q=query2, fields="files(id, name, modifiedTime)").execute()
            files = results2.get('files', [])
            total_files += len(files)

            for file in files:
                filename = file['name']
                if not (filename.endswith('.txt') or filename.endswith('.json')):
                    continue

                file_location = os.path.join(save_path, filename)
                download_file = False

                if os.path.exists(file_location):
                    last_edited_local = datetime.datetime.fromtimestamp(os.path.getmtime(file_location))
                    last_edited_drive = datetime.datetime.fromisoformat(
                        file['modifiedTime'].replace('Z', '+00:00')
                    ).replace(tzinfo=None)
                    if last_edited_local < last_edited_drive:
                        download_file = True
                else:
                    download_file = True

                if download_file:
                    self.download(file['id'], filename, save_path)

                downloaded_files += 1
                self.update_status(f"Download status: {downloaded_files}/{total_files} files downloaded")

        self.update_status("Download finished")
