class LoginModel:
    # The model class for the login and register screen.
    # It focuses on handling the data and performs all necessary calculations and processing for the login and register screen.

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