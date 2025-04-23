import os
import random
from datetime import datetime

class GoogleDrive():
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
    
    def register(self):
      folders = set(os.listdir('./drive'))
      userid = None
      while True:
         userid = str(random.randint(10000, 99999))
         if userid not in folders:
            os.makedirs('./drive/' + userid)
            break
      
      return userid