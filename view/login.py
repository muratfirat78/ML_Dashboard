from ipywidgets import *

class LoginView:
    def __init__(self, controller):
        self.controller = controller
        self.login_input = None
        self.terms_text = None
        self.copyright_text = None
        self.terms_checkbox = None
        self.login_button = None
        self.register_display = None
        self.register_button = None
        self.register_label = None
        self.loading_spinner = None
        self.loading_text = None
        self.hbox = None

        self.login_input = widgets.Text(
            description='User ID:',
            disabled=False   
        )

        self.terms_text = widgets.Label(
            value="Terms: All actions performed within this tool will be recorded and stored in Google Drive to assess skill level and recommend relevant assignments. This information may be a subject for statistical analysis."
        )
        self.copyright_text = widgets.Label(
            value="Copyright © 2025 Murat Firat and Tim Gorter. All rights reserved."
        )
        self.terms_checkbox = widgets.Checkbox( value=False,
                                                description='I agree to the terms above',
                                                disabled=False)
        self.login_button = widgets.Button(description="Login")
        self.register_button = widgets.Button(description="Register")
        
        register_text = "If you don't have a userid yet, please click register to get a userid."
        self.register_label = widgets.HTML(value = f"<b><font color='red'>{register_text}</b>")
        # widgets.Label(
        #     value = r'\(\color{red} {' + register_text  + '}\)'
        # )

        self.loading_text = widgets.Label(
            value="Downloading user data..."
        )
        
        with open('./loader.gif', 'rb') as f:
            img = f.read()

        self.loading_spinner = widgets.Image(value=img
                                             , width=25,
                                              height=25)

        self.login_button.on_click(self.login)
        self.register_button.on_click(self.register)
        self.hbox = widgets.HBox([self.loading_spinner, self.loading_text])
        self.hbox.layout.display = 'none'

        self.vbox = widgets.VBox([self.login_input,self.terms_text, widgets.HBox([self.terms_checkbox, self.login_button]),self.register_label, self.register_button,self.copyright_text, self.hbox])

    def login(self, event):
        self.controller.login(self.login_input.value, self.terms_checkbox.value)
    
    def register(self, event):
        self.controller.register()

    def get_login_view(self):
        return self.vbox
    
    def hide_login(self):
        self.vbox.layout.display = 'none'
    
    def disable_login_button(self):
        self.login_button.disabled = True

    def show_loading(self):
        self.hbox.layout.display = 'flex'