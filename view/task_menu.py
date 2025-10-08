import ipywidgets as widgets
from IPython.display import display, HTML

class TaskMenuView:
    
    def __init__(self, controller):
        self.slider = widgets.IntSlider(layout=widgets.Layout(width="90%"), min=1,max=1)
        self.slider.observe(self.slider_change)
        self.slider_box = widgets.HBox([self.slider], layout=widgets.Layout(justify_content="center", width="100%"))
        self.previous_button = widgets.Button(description='<-- Previous', button_style="primary")
        self.previous_button.on_click(self.previous_button_click)
        self.next_button = widgets.Button(description="Next -->", button_style="primary")
        self.next_button.on_click(self.next_button_click)
        self.task_box = widgets.HBox([])
        self.mode = ""
        self.competence_vector = None
        
        self.hint_button = widgets.Button(
            description="Hint",
            icon="lightbulb",
            button_style="warning"
        )
        self.finishedtask = False
        self.hint_button.on_click(self.hint)
        self.hint_display_list = []
        self.subsubtask_label = widgets.Label("Task:", layout=widgets.Layout(height="35px"))
        self.subsubtask_textarea = widgets.Textarea("",description='Task:', layout=widgets.Layout(width="100%"))
        self.hint_label = widgets.Label("Hints:", layout=widgets.Layout(height="35px"))
        self.hint_textarea = widgets.Textarea("", description='Hints:',layout=widgets.Layout(width="100%"))
        self.status_label = widgets.HTML("status:todo", layout=widgets.Layout(height="35px"))
        self.button_box = widgets.GridBox(
            children=[self.previous_button, self.hint_button, self.next_button],
            layout=widgets.Layout(
                grid_template_columns="33% 33% 33%",   
                justify_items="center",                
                width="100%",
                height="50px"
            )
        )
        self.subsubtask_box = widgets.HBox([self.subsubtask_textarea], layout=widgets.Layout(height="65px"))
        self.hints_box = widgets.HBox([self.hint_textarea], layout=widgets.Layout(width="99%"))
        self.separator = widgets.Box(
            layout=widgets.Layout(border='solid 1px lightgray', width='100%', height='1px', margin='5px 0px')
        )
        self.task_list = []
        self.ui = widgets.VBox([
            self.slider_box,
            self.button_box,
            self.subsubtask_box,
            self.hints_box,
            self.status_label,
            self.separator
        ], layout=widgets.Layout(height="280px"))
        self.controller = controller

    def previous_button_click(self,button):
        if self.slider.value > 1:
            self.slider.value = self.slider.value - 1

    def next_button_click(self,button):
        if self.slider.value < self.slider.max:
            self.slider.value = self.slider.value + 1

    def slider_change(self, change):
        if len(self.task_list) == 0:
            return
        
        change_new = change["new"]
        #sometimes change is a dict and sometimes a int?
        if isinstance(change_new, dict):
            id = change_new.get("value", None)
        else:
            id = change_new
        
        if id is not None:
            id -= 1

            if id >= len(self.task_list) and self.finishedtask:
                self.subsubtask_textarea.description = "Results:"
                if self.mode == "monitored":
                    difficulty_data = dict(self.current_task["difficulty"])
                    competence_vector = self.competence_vector or {}

                    formatted = [f"{skill}: {round(competence_vector.get(skill,0)*100)}/100" 
                                for skill, diff in difficulty_data.items()]

                    pad = max(len(formatted[i*2]) for i in range((len(formatted)+1)//2)) + 4

                    lines = [f"{formatted[i*2].ljust(pad)}{formatted[i*2+1] if i*2+1 < len(formatted) else ''}"
                            for i in range((len(formatted)+1)//2)]

                    self.subsubtask_textarea.value = "\n".join(lines)
                else:
                    self.subsubtask_textarea.value = "Task completed 🎉"

            else:
                self.subsubtask_textarea.description = "Task:"
                category = self.task_list[id]["category"]
                title = self.task_list[id]["title"]
                description = self.task_list[id]["description"]

                textarea_value = category + ": " + title + "\n" + "Description: " + description
                if self.mode == "monitored":
                    if self.task_list[id]["value"]:
                        textarea_value += "\nApplied values:" + str(self.task_list[id]["value"])


                status = self.task_list[id]["status"]

                    
                if status == "todo":
                    color = "black"
                elif status == "ready":
                    color = "blue"
                elif status == "inprogress":
                    color = "orange"
                    status = "in progress"
                elif status == "done":
                    color = "green"
                elif status == "incorrect":
                    color = "red"
                else:
                    color = "black"

                self.status_label.value = f'<b>Status:</b> <span style="color:{color};">{status}</span>'
                self.subsubtask_textarea.value = textarea_value
            
                # Hints
                hints = ""
                for x in range(self.hint_display_list[id]):
                    if x < len(self.task_list[id]["hints"]):
                        hints += self.task_list[id]["hints"][x] + "\n" 
                self.hint_textarea.value = hints
    
    def hint(self, button):
        #increase the hint display by 1
        id = self.slider.value - 1
        self.hint_display_list[id] = self.hint_display_list[id] + 1
        #refresh display
        self.slider_change({"new":self.slider.value})



    def get_task_menu(self):
        return self.ui
    
    def set_current_task(self, task, mode):
        slider_value = self.slider.value        
        task_list = []
        for subtask in task["subtasks"]:
            category = subtask["title"]
            for subsubtask in subtask["subtasks"]:
                subsubtask_object = {}
                subsubtask_object["category"] = category
                subsubtask_object["title"] = subsubtask["title"]
                subsubtask_object["description"] = subsubtask["description"]
                subsubtask_object["hints"] = subsubtask["hints"]
                subsubtask_object["status"] = subsubtask["status"]
                subsubtask_object["value"] = subsubtask["value"]

                task_list.append(subsubtask_object)
        self.task_list = task_list
        if len(self.hint_display_list) == 0:
            #initialize hint display list
            self.hint_display_list = [0] * len(task_list)
        else:
            #extend the hint list
            self.hint_display_list += [0] * (len(task_list) - len(self.hint_display_list))
        if len(task_list) > 0:
            if self.finishedtask:
                self.slider.max = len(task_list) + 1
            else:
                self.slider.max = len(task_list)
        else:
            self.slider.max = 1

        self.current_task = task
        self.slider_change({"new":slider_value})

        if mode == "monitored":
            self.hint_button.layout.display = 'none'
            self.hint_textarea.layout.display = 'none'
            self.status_label.layout.display = 'none'
            self.ui.layout=widgets.Layout(height="150px")
            self.button_box.layout=widgets.Layout(
                grid_template_columns="50% 50%",   
                justify_items="center",                
                width="100%",
                height="50px"
            )
            self.subsubtask_textarea.layout=widgets.Layout(width="100%",height="55px")
        
        self.mode = mode


    def finished_task(self, competence_vector):
        self.finishedtask = True
        self.competence_vector = competence_vector
        self.slider.max = len(self.task_list)+1
        self.slider.value = len(self.task_list)+1
        self.slider_change({"new":len(self.task_list)+1})