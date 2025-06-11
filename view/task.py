from IPython.display import HTML, display, clear_output
import matplotlib.pyplot as plt
import ipywidgets as widgets
from io import BytesIO
import numpy as np
import uuid
import json

class TaskView:
    def __init__(self, controller=None):
        self.current_task = None
        self.reference_task = None
        self.controller = controller
        self.monitored_mode = None
        self.open_ids = set()
        self.hint_counters = {}

        # Hints scripting
        display(HTML("""
        <script>
        function showHint(event) {
            const btn = event.target;
            const uid = btn.dataset.uid;
            const hints = JSON.parse(btn.dataset.hints);
            let index = parseInt(btn.dataset.index);

            const container = document.getElementById("hint-container-" + uid);

            if (index < hints.length) {
                const hintHTML = `<p><b>Hint ${index + 1}:</b> ${hints[index]}</p>`;
                container.insertAdjacentHTML('beforeend', hintHTML);
                btn.dataset.index = index + 1;
            } else {
                container.insertAdjacentHTML('beforeend', "<p><i>No more hints available.</i></p>");
                btn.disabled = true;
                btn.style.opacity = 0.6;
            }
        }
        </script>
        """))
        # Accordions styling
        display(HTML("""
        <style>
            .task-box, .subtask-section {
                max-width: 300px;
                margin: 10px 0;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                font-size: 14px;
            }
            .status-todo { background-color: #f8f8f8; }
            .status-ready { background-color: lightyellow; }
            .status-inprogress { background-color: lightblue; }
            .status-done { background-color: lightgreen; }
            .status-incorrect { background-color: #FF6666; }
            .hint-box { margin-top: 5px; font-style: italic; color: #333; }
            .hint-button {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                cursor: pointer;
                margin-top: 5px;
            }
            .hint-button:hover {
                background-color: #218838;
            }
            details > summary {
                cursor: pointer;
            }
        </style>
        """))

        self.vbox = widgets.HTML("")
        self.hint_widgets = {}

    def set_monitored_mode(self, monitored_mode):
        self.monitored_mode = monitored_mode

    def set_current_task(self, task):
        self.current_task = task
        html = ""
        self.open_ids = self.capture_open_details()
        self.hint_widgets.clear()

        html += f"""
        <details class='task-box'>
            <summary><strong>{task.get('title', '')}</strong></summary>
            <p>{task.get('description', '')}</p>
        </details>
        """

        for i, subtask in enumerate(task["subtasks"]):
            html += self.render_outer_section(subtask, f"outer-{i}")

        self.vbox.value = html
        clear_output(wait=True)
        display(self.vbox)
        self.display_hint_widgets()
    
    def set_reference_task(self, task):
        self.reference_task = task

    def render_outer_section(self, subtask, uid_prefix):
        status_class = f"status-{subtask.get('status', 'todo')}"
        outer_uid = f"{uid_prefix}-{uuid.uuid4().hex[:6]}"

        open_attr = ""
        if subtask.get("status") in ["ready", "inprogress"]:
            open_attr = "open"

        html = f"""
        <details id="{outer_uid}" class='task-box {status_class}' {open_attr}>
            <summary><b>{subtask['title']}</b></summary>
        """

        for j, inner in enumerate(subtask.get("subtasks", [])):
            html += self.render_inner_section(inner, f"{outer_uid}-inner-{j}")

        html += "</details>"
        return html

    def render_inner_section(self, subtask, uid):
        status_class = f"status-{subtask.get('status', 'todo')}"
        open_attr = "closed"
        hints_raw = json.dumps(subtask.get("hints", []))
        hints_json = (hints_raw
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;"))

        if self.monitored_mode:
            html = f"""
            <details id="{uid}" class='subtask-section {status_class}' style='margin-left: 10px;' {open_attr}>
                <summary><b>{subtask['title']}</b></summary>
                <p><b>Description:</b> {subtask.get('description', '')}</p>
                <p><b>Applied values:</b> {subtask.get('value', '')}</p>
            </details>
            """
            return html
        
        html = f"""
        <details id="{uid}" class='subtask-section {status_class}' style='margin-left: 10px;' {open_attr}>
            <summary><b>{subtask['title']}</b></summary>
            <p><b>Description:</b> {subtask.get('description', '')}</p>
        <div id="hint-container-{uid}" style="margin-top:5px; font-style: italic; color: #333;"></div>
        <button class="hint-button" data-uid="{uid}" data-hints='{hints_json}' data-index="0" onclick="showHint(event)">Hint</button>
    </details>
    """

        hint_button = widgets.Button(description="Hint", button_style='success', layout=widgets.Layout(width='70px'))
        hint_output = widgets.HTML(layout=widgets.Layout(margin='5px 0 0 0', font_style='italic', color='#333'))

        hints = subtask.get("hints", [])
        self.hint_counters[uid] = {"index": -1, "hints": hints}

        def on_hint_button_clicked(b):
            state = self.hint_counters[uid]
            state["index"] += 1
            if state["index"] < len(state["hints"]):
                new_hint = f"<p><b>Hint {state['index'] + 1}:</b> {state['hints'][state['index']]}</p>"
                hint_output.value += new_hint
            else:
                hint_output.value += "<p><i>No more hints available.</i></p>"
                hint_button.disabled = True

        hint_button.on_click(on_hint_button_clicked)

        self.hint_widgets[uid] = widgets.VBox([hint_button, hint_output],
                                              layout=widgets.Layout(margin='0 0 10px 30px'))

        return html

    def display_hint_widgets(self):
        for uid, widget in self.hint_widgets.items():
            display(widget)

    def apply_status_class(self, widget, status):
        pass

    def update_task_statuses(self, updated_task_data):
        self.set_current_task(updated_task_data)

    def set_active_accordion(self):
        pass

    def get_task_view(self):
        return self.vbox

    def show_task(self):
        display(self.vbox)
        self.display_hint_widgets()

    def capture_open_details(self):
        js = """
        <script>
        window.getOpenDetails = function() {
            return Array.from(document.querySelectorAll("details[id]")).filter(d => d.open).map(d => d.id);
        };
        </script>
        """
        display(HTML(js))
        return set()

    def finished_task(self, competence_vector):
        if self.monitored_mode:
            difficulty_data = dict(self.reference_task["difficulty"])
            skills = list(difficulty_data.keys())
            scores = [competence_vector.get(skill, 0) for skill in skills]
            difficulties = [difficulty_data[skill] for skill in skills]
            y_pos = np.arange(len(skills))

            fig, ax = plt.subplots(figsize=(3, 2.5))
            ax.barh(y_pos, scores, color='steelblue', label='Your score')
            for i, diff in enumerate(difficulties):
                ax.plot([diff, diff], [i - 0.4, i + 0.4], color='red', linewidth=2)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(skills, fontsize=8)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Score", fontsize=9)
            ax.set_title("Results", fontsize=10)
            ax.invert_yaxis()

            handles = [
                plt.Line2D([], [], color='red', linewidth=2, label='Difficulty (max score)'),
                plt.Rectangle((0, 0), 1, 1, color='steelblue', label='Your score')
            ]
            fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=7)
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            import base64
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            img_html = f'<img src="data:image/png;base64,{img_b64}" width="300"/>'
            plt.close(fig)

            self.vbox.value += f"""
            <details open class='task-box' style='max-width: 300px;'>
                <summary><b>🎉 Task completed</b></summary>
                {img_html}
            </details>
            """
        else:
            self.vbox.value += """
            <details open class='task-box status-done'>
                <summary><b>🎉 Task completed</b></summary>
                <p>Congratulations, you have completed the task!</p>
            </details>
            """