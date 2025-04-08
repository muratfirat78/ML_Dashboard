import os
from datetime import datetime

class GoogleDrive():
    def __init__(self, controller):
       self.controller = controller

    def login_correct(self,userid):
        if userid == '':
           return False
        
        if os.path.isdir('./drive/' + userid):
           return True
        else:
           return False
        
    def upload_log(self, result, userid, timestamp):
       with open('./drive/'+ userid + '/' + timestamp + '.txt', 'w') as f:
          f.write(str(result))

    def get_performances(self, userid):
       None
       #This function is not neccesary for the local drive, since the files are already stored locally