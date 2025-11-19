from ipywidgets import *

class TaskSelectionView:
    def __init__(self, controller):
        self.controller = controller
        self.recommmended_radio_buttons = None
        self.task_dropdown = None
        self.mode_dropdown = None
        self.select_button = None
        self.start_developer_mode_button = None
        self.vbox = None
        self.vbox2 = None
        self.tab_set = None
        self.recommendations_only = False
        self.guided_mode = False

        self.tasks_data = controller.get_tasks_data()
        
        self.filter_task_selection(None)
        self.tasks_label = widgets.Label('Let me work on...')
        self.task_dropdown = widgets.Select(
            options=list(self.task_map.keys()),
            # description='Let me work on...',
            disabled=False
        )
        self.mode_label = widgets.Label('I want to train a machine learning model...')

        self.mode_dropdown = widgets.RadioButtons(
            options=list(["myself","in a guided way"]),
            # description='I want to train a machine learning model...',
            disabled=False
        )
        self.mode_dropdown.layout.width = '400px'

        self.title_label = widgets.HTML(
            value="<h3>" + self.task_dropdown.value+ "</h3>"
        )

        self.description_label = widgets.HTML(
            value=self.get_description(self.task_dropdown.value)
        )
        self.description_label.layout = widgets.Layout(max_width='750px')

        self.task_dropdown.observe(self.update_title_and_description, names='value')
        self.mode_dropdown.observe(self.filter_task_selection, names='value')

        self.recommmended_radio_buttons = widgets.RadioButtons(
            options=[
                'Recommend me tasks', 
                'Show me all tasks'
            ],
            layout={'width': 'max-content'}
        )
        self.recommmended_radio_buttons.observe(self.filter_task_selection, names='value')

        self.recommmended_radio_buttons.disabled = True
        self.recommmended_radio_buttons.layout.visibility = 'hidden'
        self.recommmended_radio_buttons.layout.display = 'none'


        self.select_button = widgets.Button(
            description='Start Task',
            button_style='success'
        )
        self.select_button.on_click(self.start_task)

        self.start_developer_mode_button = widgets.Button(
            description='Developer mode',
            button_style='warning'
        )
        self.start_developer_mode_button.on_click(self.start_developer_mode)

        self.guided_mode_items = widgets.VBox([
            self.title_label,
            self.description_label
        ])

        learning_path_view = self.controller.get_learning_path_view()

        display_items = [widgets.HBox([self.mode_label,self.mode_dropdown]),self.guided_mode_items, self.recommmended_radio_buttons,widgets.HBox([self.tasks_label,self.task_dropdown]), self.select_button]

        if not self.controller.get_online_version():
            display_items += [self.start_developer_mode_button]

        self.vbox = widgets.VBox(display_items)

        self.vbox2 = widgets.VBox([learning_path_view])

        tabs = [self.vbox, self.vbox2]
        tab_set = widgets.Tab(tabs)
        tab_set.set_title(0, 'Selecting task')
        tab_set.set_title(1, 'My competence')
        tab_set.layout.display = 'none'
        self.tab_set = tab_set

    def get_description(self, title):
        task = self.task_map.get(title, {})
        return f"{task['description']}" if task else ""

    def update_title_and_description(self, change):
        new_title = change['new']
        self.title_label.value = "<h3>" + new_title + "</h3>"
        self.description_label.value = self.get_description(new_title)

    def filter_task_selection(self, change):
        if change != None:
            if change["new"] == "Recommend me tasks":
                self.recommendations_only = True
                
            if change["new"] == "Show me all tasks":
                self.recommendations_only = False

            if change["new"] == "in a guided way":
                self.guided_mode = True

            if change["new"] == "myself":
                self.guided_mode = False

        filtered_tasks = self.controller.get_filtered_tasks(self.tasks_data,self.recommendations_only, self.guided_mode)

        self.task_map = {
            task["title"]: task for task in filtered_tasks
        }
        
        if self.task_dropdown != None:
            self.task_dropdown.options=list(self.task_map.keys())

    def start_task(self, event):
        if self.mode_dropdown.value == "myself":
            monitored_mode = True
        elif self.mode_dropdown.value == "in a guided way":
            monitored_mode = False
        selected_title = self.task_dropdown.value
        selected_task = self.task_map[selected_title]
        self.controller.set_task_model(selected_task, monitored_mode)
        self.controller.read_dataset_view(selected_task["dataset"])
        self.controller.hide_task_selection_and_show_tabs()

    def start_developer_mode(self, event):
        self.controller.set_developer_mode()
        self.controller.hide_task_selection_and_show_tabs()

    def get_task_selection_view(self):
        return self.tab_set

    def hide_task_selection(self):
        self.tab_set.layout.display = 'none'
    
    def show_task_selection(self):
        self.filter_task_selection({"new": "Recommend me tasks"})
        self.tab_set.layout = widgets.Layout(visibility = 'visible')

    def disable_selection(self):
        self.task_dropdown.disabled = True
        self.select_button.disabled = True
