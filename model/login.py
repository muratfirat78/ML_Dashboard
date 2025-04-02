class LoginModel:
    def __init__(self, controller):
        self.userid = None
        self.controller = controller

    def login_correct(self,userid, drive):
        if drive != None:
            if drive.login_correct(userid):
                self.userid = userid
                return True
        else:
            self.userid = userid
            return True

    def get_userid(self):
        return self.userid