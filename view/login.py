from ipywidgets import *

class LoginView:
    def __init__(self, controller):
        self.controller = controller
        self.login_input = None
        self.terms_text = None
        self.terms_checkbox = None
        self.login_button = None
        self.loading_spinner = None
        self.loading_text = None
        self.hbox = None

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

        self.loading_text = widgets.Label(
            value="Loading..."
        )
        
        with open('./loader.gif', 'rb') as f:
            img = f.read()

        self.loading_spinner = widgets.Image(value=img
                                             , width=25,
                                              height=25)

        self.login_button.on_click(self.login)
        self.hbox = widgets.HBox([self.loading_spinner, self.loading_text])
        self.hbox.layout.display = 'none'

        self.vbox = widgets.VBox([self.login_input,self.terms_text, self.terms_checkbox, self.login_button, self.hbox])

    def login(self, event):
        self.controller.login(self.login_input.value, self.terms_checkbox.value)

    def get_login_view(self):
        return self.vbox
    
    def hide_login(self):
        self.vbox.layout.display = 'none'
    
    def disable_login_button(self):
        self.login_button.disabled = True

    def show_loading(self):
        self.vbox.layout.display = widgets.Layout(visibility = 'visible')