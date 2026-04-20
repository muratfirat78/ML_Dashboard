from ipywidgets import *

class TopicModel:

    def __init__(self, controller):
        self.controller = controller

    def set_topic(self, topic_name):
        try:
            with open('./model/topics/' + topic_name + '.html', encoding='utf-8', errors='replace') as html_file:
                html = html_file.read()
        except:
            html = 'no topic info'
        self.controller.topic_view.set_topic(html)