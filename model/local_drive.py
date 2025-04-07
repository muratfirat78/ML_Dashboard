import os
from datetime import datetime

class GoogleDrive():
    def login_correct(self,userid):
        if userid == '':
           return False
        
        if os.path.isdir('./drive/' + userid):
           return True
        else:
           return False
        
    def upload_log(self, result, userid):
       timestamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
       with open('./drive/'+ userid + '/' + timestamp + '.txt', 'w') as f:
          f.write(str(result))
      