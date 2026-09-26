
import os
import random
from datetime import datetime
import json

class GoogleDrive():
    # In offline mode this class is used, in online mode the google_drive.py file is used
    # This class contains the functions to write to local folder and save the users performances
    def login_correct(self,userid):
        if userid == '':
           return False
        
        if os.path.isdir('./drive/' + userid):
           return True
        else:
           return False
        
    def upload_log(self, result, userid, timestamp, call_stack, pred_modeling_score):
       with open('./drive/'+ userid + '/' + timestamp + '.txt', 'w') as f:
          f.write(str(result))
       with open('./drive/'+ userid + '/' + timestamp + '~' + pred_modeling_score + '.json', 'w') as f:
          json.dump(call_stack, f, indent=2, default=str)

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