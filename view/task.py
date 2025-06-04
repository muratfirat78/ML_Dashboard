from IPython.display import HTML, display, clear_output
import matplotlib.pyplot as plt
import ipywidgets as widgets
from io import BytesIO
import numpy as np
import uuid
import json

class TaskView:
    def __init__(self, controller=None):
        self.task_data = None
        self.controller = controller
        self.monitored_mode = None
        self.open_ids = set()
        self.hint_counters = {}

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

    def set_monitored_mode(self, monitored_mode):
        self.monitored_mode = monitored_mode

    def set_task(self, task):
        self.task_data = task
        html = ""
        self.open_ids = self.capture_open_details()

        # Top-level task box collapsed by default (no open)
        html += f"""
        <details class='task-box'>
            <summary><strong>{task.get('title', '')}</strong></summary>
            <p>{task.get('description', '')}</p>
        </details>
        """

        for i, subtask in enumerate(task["subtasks"]):
            html += self.render_outer_section(subtask, f"outer-{i}")

        self.vbox.value = html

    def render_outer_section(self, subtask, uid_prefix):
        status_class = f"status-{subtask.get('status', 'todo')}"
        outer_uid = f"{uid_prefix}-{uuid.uuid4().hex[:6]}"

        # Outer accordions open if status is ready or inprogress, else collapsed
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
        # Always open inner accordions
        open_attr = "closed"

        hint_id = f"hint-{uid}"
        self.hint_counters[hint_id] = {"index": -1, "hints": subtask.get("hints", [])}
        hints_json = json.dumps(self.hint_counters[hint_id]["hints"])

        hint_script = f"""
        <script>
        if (!window.hintState) {{ window.hintState = {{}}; }}
        window.hintState["{hint_id}"] = {{index: -1, hints: {hints_json} }};
        function showHint_{hint_id}() {{
            const state = window.hintState["{hint_id}"];
            state.index++;
            const output = document.getElementById("{hint_id}");
            if (state.index < state.hints.length) {{
                output.innerHTML += `<p><b>Hint ${'{'}state.index + 1{'}'}:</b> ${'{'}state.hints[state.index]{'}'}</p>`;
            }}
        }}
        </script>
        """

        if self.monitored_mode:
            html = f"""
            <div class='subtask-section {status_class}' style='margin-left: 10px;'>
                <p><b>{subtask['title']}</b></p>
                <p><b>Description:</b> {subtask.get('description', '')}</p>
                <p><b>Applied values:</b> {subtask.get('value', '')}</p>
            </div>
            """
        else:
            html = f"""
            <details id="{uid}" class='subtask-section {status_class}' style='margin-left: 10px;' {open_attr}>
                <summary><b>{subtask['title']}</b></summary>
                <p><b>Description:</b> {subtask.get('description', '')}</p>
                <button class="hint-button" onclick="showHint_{hint_id}()">Hint</button>
                <div id="{hint_id}" class="hint-box"></div>
                {hint_script}
            </details>
            """
        return html

    def apply_status_class(self, widget, status):
        # Not used in HTML-only version
        pass

    def update_task_statuses(self, updated_task_data):
        self.set_task(updated_task_data)

    def set_active_accordion(self):
        # Not needed in HTML-only version
        pass

    def get_task_view(self):
        return self.vbox

    def show_task(self):
        display(self.vbox)

    def capture_open_details(self):
        # JS to remember which <details> are open using DOM IDs
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
            difficulty_data = dict(self.task_data["difficulty"])
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
            img_widget = widgets.Image(value=buf.getvalue(), format='png', width=300)
            display(img_widget)
            plt.close(fig)

            self.vbox.value += """
            <details open style='max-width: 300px;'>
                <summary><b>🎉 Task completed</b></summary>
            </details>
            """
        else:
            self.vbox.value += """
            <details open class='task-box'>
                <summary><b>🎉 Task completed</b></summary>
                <p>Congratulations, you have completed the task!</p>
            </details>
            """
