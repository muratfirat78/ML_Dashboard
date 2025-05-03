import ipywidgets as widgets
from IPython.display import HTML, display

class TaskView:
    def __init__(self, controller=None):
        display(HTML("""
        <style>
        /* Outer accordion header colors */
        .status-todo .jupyter-widget-Accordion .jupyter-widget-Collapse-header {
            background-color: lightyellow;
        }
        .status-inprogress .jupyter-widget-Accordion .jupyter-widget-Collapse-header {
            background-color: lightblue;
        }
        .status-done .jupyter-widget-Accordion .jupyter-widget-Collapse-header {
            background-color: lightgreen;
        }

        /* Inner accordion header colors */
        .status-todo .jupyter-widget-VBox .jupyter-widget-Collapse-header {
            background-color: lightyellow;
        }
        .status-inprogress .jupyter-widget-VBox .jupyter-widget-Collapse-header {
            background-color: lightblue;
        }
        .status-done .jupyter-widget-VBox .jupyter-widget-Collapse-header {
            background-color: lightgreen;
        }
        </style>
        """))

        self.controller = controller
        self.outer_accordion = None
        self.vbox = widgets.VBox([])
        self.vbox.layout.display = 'none'
        self.outer_wrappers = []
        self.inner_wrappers = []

    def set_task(self, task):
        self.outer_wrappers = []
        self.inner_wrappers = []

        outer_sections = []
        titles = []

        for subtask in task["SubTasks"]:
            inner_accordion, inner_items = self.create_inner_accordion(subtask["SubTasks"])
            status = subtask.get("status", "todo")

            wrapper = widgets.Box([inner_accordion])
            self.apply_status_class(wrapper, status)

            self.outer_wrappers.append(wrapper)
            self.inner_wrappers.append(inner_items)

            outer_sections.append(wrapper)
            titles.append(subtask["Title"])

        self.outer_accordion = widgets.Accordion(children=outer_sections)
        for i, title in enumerate(titles):
            self.outer_accordion.set_title(i, title)

        self.vbox.children = [self.outer_accordion]

    def create_inner_accordion(self, subtasks):
        children = []
        titles = []
        wrappers = []

        for sub in subtasks:
            description = sub.get("Description", "")
            status = sub.get("status", "todo")
            content_items = []

            if description:
                content_items.append(widgets.HTML(f"<b>Description:</b> {description}"))

            vbox = widgets.VBox(content_items)
            self.apply_status_class(vbox, status) 

            wrappers.append(vbox)
            children.append(vbox)
            titles.append(sub["Title"])

        accordion = widgets.Accordion(children=children)
        for i, title in enumerate(titles):
            accordion.set_title(i, title)

        return accordion, wrappers

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
        widget.layout = widgets.Layout()

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
