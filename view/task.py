import ipywidgets as widgets
from IPython.display import HTML, display, clear_output

modal_output = widgets.Output()
display(modal_output)

class TaskView:
    def __init__(self, controller=None):
        self.controller = controller
        self.monitored_mode = None
        if not self.controller.get_online_version():
            display(HTML("""
            <style>
            .status-ready .lm-Widget.jupyter-widget-Collapse-header {
                background-color: lightyellow; 
            }
                    
            .status-done .lm-Widget.jupyter-widget-Collapse-header {
                background-color: lightgreen;
            }             
                        
            .status-inprogress .lm-Widget.jupyter-widget-Collapse-header {
                background-color: lightblue;
            }             
                         
            .status-incorrect .lm-Widget.jupyter-widget-Collapse-header {
                background-color: #FF6666;
            }             

            </style>
            """))


        self.vbox = widgets.VBox([])
        self.vbox.layout.display = 'none'
        self.outer_accordions = []
        self.inner_accordions = []

    def set_monitored_mode(self,monitored_mode):
        self.monitored_mode = monitored_mode

    def set_task(self, task):
        self.outer_accordions = []
        self.inner_accordions = []

        outer_sections = []

        for subtask in task["subtasks"]:
            inner_accordion, inner_items = self.create_inner_accordion(subtask["subtasks"])
            status = subtask.get("status", "todo")

            outer_collapse = widgets.Accordion(children=[inner_accordion])
            outer_collapse.set_title(0, subtask["title"])
            self.apply_status_class(outer_collapse, status)

            self.outer_accordions.append(outer_collapse)
            self.inner_accordions.append(inner_items)
            outer_sections.append(outer_collapse)

        info_box = widgets.VBox([
            widgets.HTML(f"<h2>{task.get('title', '')}</h2>"),
            widgets.HTML(f"<p>{task.get('description', '')}</p>")
        ])
        info_box.layout = widgets.Layout(max_width='200px')
        info_accordion = widgets.Accordion(children=[info_box])
        info_accordion.set_title(0, "ℹ️ Task information")

        if self.monitored_mode:
            if len(outer_sections) == 0:
                self.vbox.children = [info_accordion] + [widgets.HTML("<p>No subtasks available.</p>")]
            else:
                self.vbox.children = [info_accordion] + outer_sections
        else:
            self.vbox.children = [info_accordion] + outer_sections

    def create_inner_accordion(self, subtasks):
        children = []
        wrappers = []

        for sub in subtasks:
            description = sub.get("description", "")
            values = sub.get("value", "")
            status = sub.get("status", "todo")
            hints = sub.get("hints", [])
            hint_index = [-1]
            hint_output = widgets.HTML("")

            def on_hint_click(b, hints=hints, hint_index=hint_index, hint_output=hint_output):
                hint_index[0] += 1
                if hint_index[0] < len(hints):
                    hint_output.value += f"<p><b>Hint {hint_index[0]+1}:</b> {hints[hint_index[0]]}</p>"

            hint_button = widgets.Button(description="Hint", button_style='success')
            hint_button.on_click(on_hint_click)

            if self.monitored_mode:
                des = widgets.HTML(f"<b>description:</b> {description}")
                val= widgets.HTML(f"<b>applied values:</b> {values}")
                vbox = widgets.VBox([des,val])

            else:
                vbox = widgets.VBox([
                    widgets.HTML(f"<b>description:</b> {description}"),
                    hint_button,
                    hint_output
                ])

            collapse = widgets.Accordion(children=[vbox])
            collapse.set_title(0, sub["title"])

            if not self.monitored_mode: 
                self.apply_status_class(collapse, status)

            wrappers.append(collapse)
            children.append(collapse)

        accordion_vbox = widgets.VBox(children)
        return accordion_vbox, wrappers

    def apply_status_class(self, widget, status):
        current_classes = list(widget._dom_classes)
        for s in ["todo", "ready", "inprogress", "done", "incorrect"]:
            class_name = f"status-{s}"
            if class_name in current_classes:
                current_classes.remove(class_name)

        new_class = f"status-{status.replace(' ', '').lower()}"
        if new_class not in current_classes:
            current_classes.append(new_class)

        widget._dom_classes = tuple(current_classes)

    def update_task_statuses(self, updated_task_data):
        self.task_data = updated_task_data
        for i, subtask in enumerate(updated_task_data["subtasks"]):
            self.apply_status_class(self.outer_accordions[i], subtask["status"])
            if "subtasks" in subtask:
                for j, inner_subtask in enumerate(subtask["subtasks"]):
                    self.apply_status_class(self.inner_accordions[i][j], inner_subtask["status"])
    
    def set_active_accordion(self):
        for i, accordion in enumerate(self.outer_accordions):
            status_classes = accordion._dom_classes
            if any(cls == "status-ready" for cls in status_classes):
                accordion.selected_index = 0 
            else:
                accordion.selected_index = None  
                
    def get_task_view(self):
        return self.vbox

    def show_task(self):
        self.vbox.layout = widgets.Layout(visibility='visible')

    def show_completion_popup(self):
        None
        # todo