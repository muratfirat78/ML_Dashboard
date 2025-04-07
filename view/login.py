from ipywidgets import *

class LoginView:
    def __init__(self, controller):
        self.controller = controller
        self.login_input = None
        self.terms_text = None
        self.terms_checkbox = None
        self.login_button = None

        self.login_input = widgets.Text(
            description='User ID:',
            disabled=False   
        )

        self.terms_text = widgets.Label(
            value="Terms: All actions performed within this tool will be recorded and stored in Google Drive to assess skill level and recommend relevant assignments."
        )
        self.terms_checkbox = widgets.Checkbox( value=False,
                                                description='I agree to the terms above',
                                                disabled=False)
        self.login_button = widgets.Button(description="Login")

        self.login_button.on_click(self.login)
        self.vbox = widgets.VBox([self.login_input,self.terms_text, self.terms_checkbox, self.login_button])

    def login(self, event):
        self.controller.login(self.login_input.value, self.terms_checkbox.value)

    def get_login_view(self):
        return self.vbox
    
    def hide_login(self):
        self.vbox.layout.display = 'none'
