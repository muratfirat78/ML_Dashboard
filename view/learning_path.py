from ipywidgets import widgets
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import matplotlib.colors as mcolors

class LearningPathView:
    def __init__(self, controller):
        self.controller = controller
        self.graph_data = [{"date":None}]
        self.label = widgets.Label(value="Log current model:")
        self.label2 = widgets.Label(value="Previous result:")
        self.list = widgets.SelectMultiple(
            options=[],
            value=[],
            disabled=True,
            rows=20,
            layout={'width': '50%'}
        )
        self.previous_performance = widgets.HTML()

        self.skill_dropdown = widgets.Dropdown(
            options=list(["All"]),
            description='Select skill:',
            disabled=False
        )
        self.skill_dropdown.observe(self.update_line_chart, names='value')
 
        self.bar_chart = widgets.Output()
        self.line_chart = widgets.Output()
    
        self.log = widgets.VBox([self.label,self.list])
        self.log.layout = widgets.Layout(width='50%')
        self.previous_performance.layout = widgets.Layout(width='50%')
        self.last_result = widgets.VBox([self.label2,self.previous_performance])
     
        self.hbox = widgets.VBox([widgets.HBox([self.bar_chart, self.line_chart])
                                , widgets.HBox([self.log,self.last_result])])


    def get_icon(self, category):
        if category == 'SelectData':
            return '(Data selection)'
        if category == 'DataCleaning':
            return '(Data cleaning)'
        if category == 'DataProcessing':
            return '(Data processing)'
        if category == 'ModelDevelopment':
            return '(Model development)'
        return ''

    def update_actions(self):
        actions = self.controller.get_list_of_actions()
        updated_action_array = []

        for action in actions:
            updated_action_array += [str(action[2]) + ': ' + action[1]  + ' ' + self.get_icon(action[0])]

        self.list.options = updated_action_array

    def update_bar_chart(self):
        self.bar_chart.clear_output()
        with self.bar_chart:
            stat_data = self.graph_data[-1].copy()
            del stat_data['date'] 
            df = pd.DataFrame(list(stat_data.items()), columns=['Skill', 'Value'])
            max_value = 100
            min_value = 0

            if max_value != min_value:  
                df['Value'] = 5 * (df['Value'] - min_value) / (max_value - min_value)
            else:
                df['Value'] = 5 

            norm = mcolors.Normalize(vmin=0, vmax=5)
            cmap = plt.cm.YlGn
            colors = [cmap(norm(v)) for v in df['Value']]

            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='Skill', y='Value', palette=colors)
            plt.ylim(0, 5)
            plt.yticks([0, 1, 2, 3, 4, 5], ['Beginner', 'Basic Knowledge', 'Intermediate', 'Experienced', 'Expert', 'Master'])
            plt.title('Current competence')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            return plt

    
    def update_line_chart(self, value):
        value = value["new"]
        self.line_chart.clear_output()
        with self.line_chart:
            df = pd.DataFrame(self.graph_data)
            df['date'] = df['date'].apply(lambda d: d[0] if isinstance(d, tuple) else d)
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

            if value and value != "All":
                df = df[['date', value]]

            df_melted = df.melt(id_vars='date', var_name='Competence', value_name='Value')
            max_value = 100
            min_value = 0
            df_melted['Value'] = 5 * (df_melted['Value'] - min_value) / (max_value - min_value)
            plt.figure(figsize=(12, 5.6))
            sns.lineplot(data=df_melted, x='date', y='Value', hue='Competence', marker='o', palette='viridis')
            plt.ylim(0, 5)
            plt.yticks([0, 1, 2, 3, 4, 5], ['Beginner', 'Basic Knowledge', 'Intermediate', 'Experienced', 'Expert', 'Master'])

            if value == "All":
                plt.title('Learning path')
            else:
                plt.title('Competence level of ' + value + ' over time')

            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%m-%Y'))
            plt.xticks(df_melted['date'], df_melted['date'].dt.strftime('%d-%m-%Y'), rotation=45)
            plt.tight_layout()
            plt.show()


    def get_learning_path_tab(self):
        return self.hbox
    
    def set_graph_data(self, graph_data):
        self.graph_data = graph_data
        if len(graph_data) > 0:
            self.update_bar_chart()
            self.update_line_chart({"new": "Predictive Modeling"})
    
    def set_last_performance_data(self):
        learning_path = self.controller.get_learning_path()
        if len(learning_path) > 0:
            last_performance = learning_path[-1]
            # Convert to task to get target and dataset name more easily
            task = self.controller.convert_performance_to_task(last_performance, "", "")
            target_column = self.controller.get_target_task(task)
            dataset_name = task["dataset"].replace(".csv", "")
            
            #Get reference task
            reference_task = self.controller.get_reference_task(target_column, dataset_name)
            if reference_task:
                difficulty_data = dict(reference_task["difficulty"])
                current_datetime = datetime.now()

                self.label2.value = "Previous result (" + reference_task["title"] + "):"

                competence_vector = self.controller.calculate_competence_vector(last_performance,reference_task, current_datetime)
                img_html = self.controller.get_result_bar_chart(competence_vector, difficulty_data) 
                self.previous_performance.value = img_html
                return
            
        # If reference task not found
        self.previous_performance.value = "No data found"
