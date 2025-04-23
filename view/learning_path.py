from ipywidgets import widgets
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime


class LearningPathView:
    def __init__(self, controller):
        self.controller = controller
        self.stats = [{"date":None}]
        self.label = widgets.Label(value="Log current model:")
        self.list = widgets.SelectMultiple(
            options=[],
            value=[],
            disabled=True,
            rows=20,
            layout={'width': '50%'}
        )

        self.bar_chart = widgets.Output()
        self.line_chart = widgets.Output()
        self.update_bar_chart()
        self.update_line_chart()
        self.hbox =  widgets.VBox([widgets.HBox([self.bar_chart, self.line_chart]),self.label,self.list])

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
            stat_data = self.stats[-1].copy()
            del stat_data['date']
            df = pd.DataFrame(list(stat_data.items()), columns=['Skill', 'Value'])
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='Skill', y='Value', palette='viridis')
            plt.ylim(0, 100)
            plt.title('Skill level distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            return plt
    
    def update_line_chart(self):
        self.line_chart.clear_output()
        with self.line_chart:
            df = pd.DataFrame(self.stats)
            df['date'] = df['date'].apply(lambda d: d[0] if isinstance(d, tuple) else d)
            df_melted = df.melt(id_vars='date', var_name='Skill', value_name='Value')

            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df_melted, x='date', y='Value', hue='Skill', marker='o', palette='viridis')
            plt.ylim(0, 100)
            plt.title('Skill level over time')
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%m-%Y'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def get_learning_path_tab(self):
        return self.hbox
    
    def set_stats(self, stats):
        self.stats = stats
        self.update_bar_chart()
        self.update_line_chart()