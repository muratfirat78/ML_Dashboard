from ipywidgets import *

class LoginView:
    def __init__(self, controller):
        self.controller = controller
        self.login_input = None
        self.login_button = None

        self.login_input = widgets.Text(
            description='User ID:',
            disabled=False   
        )
    
        self.login_button = widgets.Button(description="Login")

        self.login_button.on_click(self.login)
        self.vbox = widgets.VBox([self.login_input, self.login_button])

    def login(self, event):
        self.controller.login(self.login_input.value)

    def get_login_view(self):
        return self.vbox
    
    def hide_login(self):
        self.vbox.layout.display = 'none'
