import ipywidgets as widgets
from IPython.display import display, HTML

class TaskMenuView:
    # The view class for the task menu, used to show hints, explain steps and track student actions.
    def __init__(self, controller):
        self.slider = widgets.IntSlider(layout=widgets.Layout(width="99%", display="none"), min=1,max=1)
        self.slider.style.handle_color = 'lightblue'
        self.slider.observe(self.slider_change)
        self.previous_button = widgets.Button(description='<< Previous', button_style="primary")
        self.previous_button.on_click(self.previous_button_click)
        self.next_button = widgets.Button(description="Next >>", button_style="primary")
        self.next_button.on_click(self.next_button_click)
        self.topic_explaination_button = widgets.Button(description="More info")
        self.topic_explaination_button.on_click(self.topic_explaination_button_click)
        self.undo_button = widgets.Button(description="Revert to this action")
        self.undo_button.on_click(self.undo_click)
        self.undo_button.layout.visibility  = 'hidden'
        self.task_box = widgets.HBox([])
        self.mode = ""
        self.competence_vector = None
        self.actions = []
        self.vertical_seperator = widgets.Box(
            layout=widgets.Layout(
                border='solid 1px lightblue',
                width='1px',
                align_self='stretch',
                margin='0px 8px'
            )
        )
        self._timeline_buttons = [] 
        self.timeline = widgets.GridBox(
            children=[],
            layout=widgets.Layout(
                grid_template_columns="repeat(1, 40px)",
                justify_content="flex-start",
                align_items="center",
                grid_gap="10px",
                width="100%"
            )
        )
        self.timeline = self.get_timeline(1, 1)
        
        
        
        self.hint_button = widgets.Button(
            description="Instructions",
            icon="lightbulb",
            button_style="warning"
        )
        self.finishedtask = False
        self.hint_button.on_click(self.hint)
        self.hint_display_list = []
        
        
        self.status_label = widgets.HTML(" Status: todo", layout=widgets.Layout(height="35px", width="25%", text_align="center"))
        self.button_box = widgets.GridBox(
            children=[self.previous_button, self.topic_explaination_button, self.hint_button, self.undo_button, self.next_button],
            layout=widgets.Layout(
                grid_template_columns="20% 20% 20% 20% 20%",
                justify_items="center",                
                width="100%",
                height="50px"
            )
        )
       
        self.statusbox = widgets.GridBox(
            children=[widgets.HTML(""),self.status_label, widgets.HTML("")],
            layout=widgets.Layout(
                grid_template_columns="33% 33% 33%",   
                justify_items="center",                
                width="100%",
                height="50px"
            )
        )

        self.subsubtask_textarea = widgets.Textarea(description="Tasks", disabled=True, layout=widgets.Layout(width="99%"),style={'background': "#C7EFFF"})
        self.hint_textarea = widgets.Textarea(disabled=True,description="Hints",style={'background': '#C7EFFF'})
        
        self.subsubtask_box = widgets.VBox([self.subsubtask_textarea,self.hint_textarea], layout=widgets.Layout(width="99%",height="100px"))
        self.task_list = []
        self.timeline = self.get_timeline(1,1)
        self.ui = widgets.VBox(self.get_ui(), layout=widgets.Layout(height="230px"))
        self.controller = controller

    def get_timeline(self, number_of_buttons, active):
        while len(self._timeline_buttons) < number_of_buttons:
            i = len(self._timeline_buttons) + 1
            button = widgets.Button(
                description=str(i),
                layout=widgets.Layout(width='40px')
            )
            button.index = i
            button.on_click(self.timeline_button_click)
            self._timeline_buttons.append(button)

        buttons = self._timeline_buttons[:number_of_buttons]

        for btn in buttons:
            if btn.index == active or number_of_buttons == 1:
                btn.style.button_color = '#0d6efd'
                btn.style.text_color = 'white'
            else:
                btn.style.button_color = None
                btn.style.text_color = None

        self.timeline.children = tuple(buttons)
        self.timeline.layout.grid_template_columns = f"repeat({number_of_buttons}, 40px)"
        return self.timeline

    def timeline_button_click(self, button):
        step = button.index
        self.slider.value = step

    def undo_click(self,button):
        self.controller.logger.undo(self.slider.value)

    def topic_explaination_button_click(self, button):
        self.controller.main_view.switch_tab_to_topic_info()

    def previous_button_click(self,button):
        if self.slider.value > 1:
            self.slider.value = self.slider.value - 1

    def next_button_click(self,button):
        if self.slider.value < self.slider.max:
            self.slider.value = self.slider.value + 1

    def slider_change(self, change):

        if not self.task_list:
            return

        change_new = change["new"]

        if isinstance(change_new, dict):
            idx = change_new.get("value")
        else:
            idx = change_new

        if idx is None:
            return

        self.timeline = self.get_timeline(self.slider.max, idx)

        idx -= 1

        if idx >= len(self.task_list) and self.finishedtask:

            self.subsubtask_textarea.description = "Results:"

            if self.mode != "monitored":
                self.subsubtask_textarea.value = "Task completed 🎉"

            return

        task = self.task_list[idx]

        category = task["category"]
        title = task["title"]
        description = task["description"]
        status = task["status"]

        # Text content
        if self.mode != "monitored":
            textarea_value = (
                f"{category}: {title}\n"
                f"Description: {description}"
            )
        else:
            textarea_value = f"{category}\n{title}"
            self.controller.show_message(str(category))

        status_styles = {
            "todo": ("black", "todo"),
            "ready": ("blue", "ready"),
            "inprogress": ("orange", "in progress"),
            "done": ("green", "done (press Next >>)"),
            "incorrect": ("red", "incorrect"),
        }

        color, status_text = status_styles.get(
            status,
            ("black", status)
        )

        self.subsubtask_textarea.value = textarea_value

        if self.mode != "monitored":

            self.status_label.value = (
                f'<b>Status:</b> '
                f'<span style="color:{color};">{status_text}</span>'
            )

            hints = "\n".join(
                task["hints"][:self.hint_display_list[idx]]
            )

            self.hint_textarea.value = hints
    
    def hint(self, button):
        #increase the hint display by 1
        id = self.slider.value - 1
        self.hint_display_list[id] = self.hint_display_list[id] + 1
        #refresh display
        self.slider_change({"new":self.slider.value})

    def show_hint_text(self, text):
        if text is None:
            text = ''

        if text.startswith("Error:"):
            self.hint_textarea.style.text_color = 'red'
        else:
            self.hint_textarea.style.text_color = None 
        self.hint_textarea.value = text

    def get_task_menu(self):
        return self.ui
    
    def add_action_monitored_mode(self, action, value):
        self.actions.append((action,value))

        task_list = []
        for action in self.actions:
            subsubtask_object = {}
            subsubtask_object["category"] = action[0]
            subsubtask_object["title"] = action[1]
            subsubtask_object["description"] = action[0]
            subsubtask_object["hints"] = ''
            subsubtask_object["status"] = 'ready'
            subsubtask_object["value"] = str(action[1])
            task_list.append(subsubtask_object)
        self.refresh_menu(task_list)
        self.slider_change({"new":self.slider.max-1})
        self.slider.value = self.slider.max
    
    def clear_actions_monitored_mode(self):
        self.actions = []

    def set_current_task(self, task, mode):
        #set the data for the current task to display in the task menu
             
        task_list = []
        for subtask in task["subtasks"]:
            category = subtask["title"]
            for subsubtask in subtask["subtasks"]:
                if mode == 'guided':
                    subsubtask_object = {}
                    subsubtask_object["category"] = category
                    subsubtask_object["title"] = subsubtask["title"]
                    subsubtask_object["description"] = subsubtask["description"]
                    subsubtask_object["hints"] = subsubtask["hints"]
                    subsubtask_object["status"] = subsubtask["status"]
                    subsubtask_object["value"] = subsubtask["value"]

                    task_list.append(subsubtask_object)
                if mode == 'monitored':
                    values = subsubtask["value"]
                    
                    if not isinstance(values, list):
                        values = [values]

                    for val in values:
                        subsubtask_object = {}
                        subsubtask_object["category"] = category
                        subsubtask_object["title"] = f'{subsubtask["title"]}: {val}'
                        subsubtask_object["description"] = subsubtask["description"]
                        subsubtask_object["hints"] = subsubtask["hints"]
                        subsubtask_object["status"] = subsubtask["status"]
                        subsubtask_object["value"] = [val]

                        task_list.append(subsubtask_object)
        self.current_task = task
        self.refresh_menu(task_list)

        if mode == "monitored":
            self.statusbox.layout.visibility  = 'hidden'
            self.statusbox.layout.display = 'none'
            self.hint_textarea.layout.width = "99%"
            self.undo_button.layout.visibility  = 'visible'
            self.hint_button.layout.visibility = 'hidden'
            self.subsubtask_textarea.description = "Action"
            self.subsubtask_box.children = [self.subsubtask_textarea, self.hint_textarea,]

        self.mode = mode

    def refresh_menu(self, task_list):
        slider_value = self.slider.value   
        self.task_list = task_list
        if len(self.hint_display_list) == 0:
            #initialize hint display list
            self.hint_display_list = [0] * len(task_list)
        else:
            #extend the hint list
            self.hint_display_list += [0] * (len(task_list) - len(self.hint_display_list))

        if len(task_list) > 0:
            if self.finishedtask:
                timeline_length = len(task_list)
                self.slider.max = len(task_list)
                self.timeline = self.get_timeline(timeline_length, self.slider.value)
            else:
                timeline_length = len(task_list)
                self.timeline = self.get_timeline(timeline_length, self.slider.value)
                self.slider.max = len(task_list)
        else:
            self.slider.max = 1
            self.timeline = self.get_timeline(1,1)
        if slider_value <= self.slider.max:
            self.slider_change({"new":slider_value})
        else:
            self.slider_change({"new":self.slider.max})
        

        self.ui.children = self.get_ui()

    def get_ui(self):
        if self.mode == "monitored":
            return (
                self.subsubtask_box,
                self.statusbox,
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.Label("Actions"),
                                self.timeline
                            ],
                            layout=widgets.Layout(
                                align_items='flex-start',
                                gap='2px',
                                width='100%' 
                            )
                        ),
                        # self.vertical_seperator,
                        # widgets.VBox(
                        #     [self.undo_button, self.topic_explaination_button],
                        #     layout=widgets.Layout(
                        #         width='15%',
                        #         justify_content='center',
                        #         align_items='stretch',
                        #         gap='4px',
                        #         padding='0px 4px'
                        #     )
                        # )
                    ],
                    layout=widgets.Layout(
                        align_items='stretch',
                        width='99%',
                        padding='4px 0px'
                    )
                ),
                widgets.Box(layout=widgets.Layout(
                    border='solid 1px lightblue', width='99%',
                    height='1px', margin='4px 0px',
                ))
            )
        else:
            return ([
            self.slider, 
            self.button_box,
            self.subsubtask_box,
            self.statusbox,
            widgets.Box(layout=widgets.Layout(border='solid 1px lightblue', width='99%', height='1px', margin='5px 0px',style={'background': "#C7EFFF"}))])

    def finished_task(self, competence_vector):
        #display that the student has finished the task
        self.finishedtask = True
        self.competence_vector = competence_vector
        self.slider.max = len(self.task_list)+1
        self.slider.value = len(self.task_list)+1
        self.slider_change({"new":len(self.task_list)+1})