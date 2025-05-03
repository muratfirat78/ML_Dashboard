import ipywidgets as widgets
from IPython.display import HTML, display


class TaskView:
    def __init__(self, controller=None):
        self.controller = controller
        if self.controller.get_online_version():
            display(HTML("""
            <style>
            .status-todo .lm-Widget.jupyter-widget-Collapse-header {
                background-color: lightyellow; 
            }
                    
            .status-done .lm-Widget.jupyter-widget-Collapse-header {
                background-color: lightgreen;
            }             
                        
            .status-inprogress .lm-Widget.jupyter-widget-Collapse-header {
                background-color: lightgreen;
            }             

            </style>
            """))
        else:
            display(HTML("""
            <style>
            .status-todo .p-Collapse-header {
                background-color: lightyellow;
            }

            .status-done .p-Collapse-header {
                background-color: lightgreen;
            }
                         
            .status-inprogress .p-Collapse-header {
                background-color: lightskyblue;
            }
            </style>
            """))

        self.vbox = widgets.VBox([])
        self.vbox.layout.display = 'none'
        self.outer_wrappers = []
        self.inner_wrappers = []

    def set_task(self, task):
        self.outer_wrappers = []
        self.inner_wrappers = []

        outer_sections = []

        for subtask in task["SubTasks"]:
            inner_accordion, inner_items = self.create_inner_accordion(subtask["SubTasks"])
            status = subtask.get("status", "todo")

            outer_collapse = widgets.Accordion(children=[inner_accordion])
            outer_collapse.set_title(0, subtask["Title"])
            self.apply_status_class(outer_collapse, status)

            self.outer_wrappers.append(outer_collapse)
            self.inner_wrappers.append(inner_items)
            outer_sections.append(outer_collapse)

        self.vbox.children = outer_sections

    def create_inner_accordion(self, subtasks):
        children = []
        wrappers = []

        for sub in subtasks:
            description = sub.get("Description", "")
            status = sub.get("status", "todo")

            vbox = widgets.VBox([widgets.HTML(f"<b>Description:</b> {description}")])
            collapse = widgets.Accordion(children=[vbox])
            collapse.set_title(0, sub["Title"])
            self.apply_status_class(collapse, status)

            wrappers.append(collapse)
            children.append(collapse)

        accordion_vbox = widgets.VBox(children)
        return accordion_vbox, wrappers

    def apply_status_class(self, widget, status):
        current_classes = list(widget._dom_classes)
        for s in ["todo", "ready", "inprogress", "done"]:
            class_name = f"status-{s}"
            if class_name in current_classes:
                current_classes.remove(class_name)

        new_class = f"status-{status.replace(' ', '').lower()}"
        if new_class not in current_classes:
            current_classes.append(new_class)

        widget._dom_classes = tuple(current_classes)

    def update_task_statuses(self, updated_task_data):
        self.task_data = updated_task_data
        for i, subtask in enumerate(updated_task_data["SubTasks"]):
            self.apply_status_class(self.outer_wrappers[i], subtask["status"])
            if "SubTasks" in subtask:
                for j, inner_subtask in enumerate(subtask["SubTasks"]):
                    self.apply_status_class(self.inner_wrappers[i][j], inner_subtask["status"])

    def get_task_view(self):
        return self.vbox

    def show_task(self):
        self.vbox.layout = widgets.Layout(visibility='visible')
