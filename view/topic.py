from ipywidgets import *

class TopicView:
    def __init__(self, controller):
        self.tab = VBox([HTML('')],  
                layout=Layout(
                width='100%',
                max_width='800px',
                height='500px',
                overflow='auto',
                border='1px solid lightgray'
            ))
        self.controller = controller
        self.set_topic('no topic info')
        self.html = ''

    def set_topic(self,html):
        self.html = html
    
    def display_html(self):
        self.tab.children = (HTML(self.html),)

    def get_topic_tab(self):
        return self.tab