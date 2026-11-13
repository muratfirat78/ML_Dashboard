import ast
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ipywidgets as widgets
from IPython.display import clear_output, display

STAGE_ORDER = ['SelectData', 'DataCleaning', 'DataProcessing', 'ModelDevelopment']


class Overview:
    # Builds a table + plots of student progress for the selected run, based on
    # each student's latest saved performance ("<run>__<timestamp>.txt") file.

    def __init__(self, controller):
        self.controller = controller

        self.table_output = widgets.Output()
        self.stage_output = widgets.Output()
        self.heatmap_output = widgets.Output()

        self.vbox = widgets.VBox([
            self.table_output,
            self.stage_output,
            self.heatmap_output,
        ])

    def get_ui(self):
        return self.vbox

    def refresh(self):
        data = self.load_data(self.controller.run)

        with self.table_output:
            clear_output(wait=True)
            self.show_table(data)

        with self.stage_output:
            clear_output(wait=True)
            self.plot_stage_distribution(data)

        with self.heatmap_output:
            clear_output(wait=True)
            self.plot_progress_heatmap(data)

    def parse_timestamp(self, filename):
        stem = Path(filename).stem
        _, _, ts_part = stem.partition('__')
        try:
            return datetime.strptime(ts_part, "%d-%m-%Y %H-%M-%S")
        except ValueError:
            return None

    def load_data(self, run):
        drive_path = Path('../drive')
        data = {}

        if not run or not drive_path.exists():
            return data

        for student_dir in sorted(drive_path.iterdir()):
            if not student_dir.is_dir():
                continue

            candidates = []
            for file in student_dir.glob(f'{run}__*.txt'):
                ts = self.parse_timestamp(file.name)
                if ts is not None:
                    candidates.append((ts, file))

            if not candidates:
                continue

            candidates.sort(key=lambda c: c[0])
            latest_ts, latest_file = candidates[-1]

            try:
                with open(latest_file, encoding='utf-8') as f:
                    content = f.read()
                # the saved dict can contain numpy scalar reprs, e.g. np.int64(0)
                content = re.sub(r'np\.\w+\(([^()]+)\)', r'\1', content)
                performance = ast.literal_eval(content)
            except Exception as e:
                print(f"Failed reading {latest_file}: {e}")
                continue

            data[student_dir.name] = {
                'performance': performance,
                'timestamp': latest_ts,
            }

        return data

    def furthest_stage(self, performance):
        reached = None
        for stage in STAGE_ORDER:
            if performance.get(stage):
                reached = stage
        return reached

    def get_metric(self, performance):
        model_dev = performance.get('ModelDevelopment', {})
        perf_entry = model_dev.get('ModelPerformance')
        if not perf_entry:
            return None

        metrics, _ = perf_entry
        preferred = ('Accuracy', 'R2', 'R2 Score', 'Mean Squared Error')

        for key in preferred:
            for metric_name, value in metrics:
                if metric_name == key:
                    return f"{key} = {value}"

        if metrics:
            name, value = metrics[0]
            return f"{name} = {value}"
        return None

    def build_rows(self, data):
        rows = []
        for student_id, entry in sorted(data.items()):
            performance = entry['performance']

            dataset = performance.get('SelectData', {}).get('DataSet')
            dataset_name = dataset[0] if isinstance(dataset, (tuple, list)) else dataset

            target = performance.get('DataProcessing', {}).get('AssignTarget')
            target_name = target[0] if isinstance(target, (tuple, list)) else target

            rows.append({
                'Student': student_id,
                'Last activity': entry['timestamp'].strftime('%d-%m-%Y %H:%M'),
                'Dataset': dataset_name or '-',
                'Target': target_name or '-',
                'Furthest stage': self.furthest_stage(performance) or '-',
                'Metric': self.get_metric(performance) or '-',
            })
        return rows

    def show_table(self, data):
        rows = self.build_rows(data)
        if not rows:
            print('No data available for this run yet.')
            return

        df = pd.DataFrame(rows)
        display(df)

    def plot_stage_distribution(self, data):
        rows = self.build_rows(data)
        if not rows:
            return

        df = pd.DataFrame(rows)
        counts = df['Furthest stage'].value_counts().reindex(STAGE_ORDER, fill_value=0)

        plt.figure(figsize=(7, 4))
        plt.bar(counts.index, counts.values, color='steelblue')
        plt.title(f'Students per furthest stage reached ({self.controller.run})')
        plt.ylabel('Students')
        plt.tight_layout()
        plt.show()

    def plot_progress_heatmap(self, data):
        if not data:
            print('No data available.')
            return

        df = pd.DataFrame(0, index=sorted(data.keys()), columns=STAGE_ORDER)

        for student_id, entry in data.items():
            performance = entry['performance']
            for stage in STAGE_ORDER:
                if performance.get(stage):
                    df.loc[student_id, stage] = 1

        annotations = df.replace({1: 'done', 0: ''})

        plt.figure(figsize=(6, max(3, 0.4 * len(df))))
        sns.heatmap(
            df,
            annot=annotations,
            fmt="",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            cbar=False,
            linewidths=0.5,
        )
        plt.title(f'Student progress ({self.controller.run})')
        plt.xlabel('Stage')
        plt.ylabel('Student')
        plt.tight_layout()
        plt.show()
