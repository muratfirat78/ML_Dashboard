from googleapiclient.discovery import build

class GoogleDriveModel:
    def __init__(self):
        self.drive_service = build('drive', 'v3')
        self.folder_id = "1pN1jFF5tcDxrfQRdTQrHDNvvxRLRbI69"

    def upload_file(self, file, destination):
        googleapiclient.http.MediaFileUpload("bestand.txt", resumable=True)
