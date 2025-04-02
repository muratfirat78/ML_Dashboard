class LoginModel:
    def __init__(self, controller):
        self.userid = None
        self.controller = controller

    def login(self,userid):
        self.userid = userid