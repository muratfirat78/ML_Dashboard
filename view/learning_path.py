from ipywidgets import widgets

class LearningPathView:
    def __init__(self, controller):
        self.controller = controller

        self.list = widgets.SelectMultiple(
            options=[],
            value=[],
            disabled=True,
            rows=20,
            layout={'width': '50%'}
        )
        self.vbox =  widgets.VBox([self.list])

    def get_icon(self, category):
        if category == 'SelectData':
            return '(Data selection)'
        if category == 'DataCleaning':
            return '(Data cleaning)'
        if category == 'DataProcessing':
            return '(Data processing)'
        if category == 'ModelDevelopment':
            return '(Model development)'
        return ''

    def update_actions(self):
        actions = self.controller.get_list_of_actions()
        updated_action_array = []

        for action in actions:
            updated_action_array += [str(action[2]) + ': ' + action[1]  + ' ' + self.get_icon(action[0])]

        self.list.options = updated_action_array


    def get_learning_path_tab(self):
        return self.vbox