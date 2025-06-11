from ipywidgets import *

class TaskSelectionView:
    def __init__(self, controller):
        self.controller = controller
        self.recommmended_radio_buttons = None
        self.task_dropdown = None
        self.mode_dropdown = None
        self.select_button = None
        self.vbox = None
        self.vbox2 = None
        self.tab_set = None
        self.alltasks = False
        self.guided_mode = True

        self.tasks_data = controller.get_tasks_data()
        
        self.filter_task_selection(None)

        self.task_dropdown = widgets.Dropdown(
            options=list(self.task_map.keys()),
            description='Let me work on...',
            disabled=False
        )

        self.mode_dropdown = widgets.Dropdown(
            options=list(["Guidance for a task","My performance monitored"]),
            description='I want...',
            disabled=False
        )

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

        self.recommmended_radio_buttons.layout.display = 'none'

        self.select_button = widgets.Button(
            description='Start Task',
            button_style='success'
        )
        self.select_button.on_click(self.start_task)

        self.guided_mode_items = widgets.VBox([
            self.title_label,
            self.description_label
        ])

        learning_path_view = self.controller.get_learning_path_view()

        self.vbox = widgets.VBox([
            self.mode_dropdown, self.recommmended_radio_buttons,self.task_dropdown,self.guided_mode_items, self.select_button
        ])

        self.vbox2 = widgets.VBox([learning_path_view])
        # self.vbox.layout.display = 'none'

        tabs = [self.vbox, self.vbox2]
        tab_set = widgets.Tab(tabs)
        tab_set.set_title(0, 'Selecting task')
        tab_set.set_title(1, 'My competence')
        tab_set.layout.display = 'none'
        self.tab_set = tab_set

    def get_description(self, title):
        task = self.task_map.get(title, {})
        return f"<i>{task['description']}</i>" if task else ""

    def update_title_and_description(self, change):
        new_title = change['new']
        self.title_label.value = "<h3>" + new_title + "</h3>"
        self.description_label.value = self.get_description(new_title)

    def filter_task_selection(self, change):
        if change != None:
            if change["new"] == "Recommend me tasks":
                self.alltasks = False
                
            if change["new"] == "Show me all tasks":
                self.alltasks = True

            if change["new"] == "Guidance for a task":
                self.guided_mode = True
                self.hide_recommmended_radio_buttons()

            if change["new"] == "My performance monitored":
                self.guided_mode = False
                self.show_recommmended_radio_buttons()

        if self.guided_mode:
            filtered_tasks = [task for task in self.tasks_data if task["mode"] == "guided"]
        else:
            filtered_tasks = [task for task in self.tasks_data if task["mode"] == "monitored"]

        filtered_tasks = self.controller.get_filtered_tasks(self.tasks_data, self.guided_mode, self.alltasks)

        self.task_map = {
            task["title"]: task for task in filtered_tasks
        }
        
        if self.task_dropdown != None:
            self.task_dropdown.options=list(self.task_map.keys())

    def start_task(self, event):
        if self.mode_dropdown.value == "My performance monitored":
            monitored_mode = True
        elif self.mode_dropdown.value == "Guidance for a task":
            monitored_mode = False
        selected_title = self.task_dropdown.value
        selected_task = self.task_map[selected_title]
        self.controller.set_task_model(selected_task, monitored_mode)
        self.controller.read_dataset_view(selected_task["dataset"])
        self.controller.hide_task_selection_and_show_tabs()

    def hide_recommmended_radio_buttons(self):
        self.recommmended_radio_buttons.layout.display = 'none'
        
    def show_recommmended_radio_buttons(self):
        # Todo: implenment recommendation of exercises
        # self.recommmended_radio_buttons.layout = widgets.Layout(visibility = 'visible')
        self.recommmended_radio_buttons.layout.display = 'none'

    def get_task_selection_view(self):
        return self.tab_set

    def hide_task_selection(self):
        self.tab_set.layout.display = 'none'
    
    def show_task_selection(self):
        self.tab_set.layout = widgets.Layout(visibility = 'visible')

    def disable_selection(self):
        self.task_dropdown.disabled = True
        self.select_button.disabled = True
