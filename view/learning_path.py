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
        
        self.display_dropdown = widgets.Dropdown(
            options=list(["Currect competence level", "Competence level over time", "Current log"]),
            description='Graph:',
            disabled=False
        )
        self.display_dropdown.observe(self.display_dropdown_change, names='value')

        self.skill_dropdown = widgets.Dropdown(
            options=list(["Currect competence level", "Competence level over time", "Current log"]),
            description='Select skill:',
            disabled=False
        )
        self.skill_dropdown.observe(self.update_line_chart, names='value')
 
        self.bar_chart = widgets.Output()
        self.line_chart = widgets.Output()
    
        self.update_bar_chart()
        self.update_line_chart({"new": "All"})
        self.bar_chart.layout.display = 'block'
        self.skill_dropdown.layout.display = 'none'
        self.line_chart.layout.display = 'none'
        self.list.layout.display = 'none'       
        self.hbox =  widgets.VBox([self.display_dropdown,self.skill_dropdown, self.bar_chart, self.line_chart, self.label,self.list])

    def display_dropdown_change(self, value):
        selection = value["new"]

        if selection == "Competence level over time":
            self.line_chart.layout.display = 'block'
            self.skill_dropdown.layout.display = 'block'
            self.bar_chart.layout.display = 'none'
            self.list.layout.display = 'none'

        elif selection == "Currect competence level":
            self.line_chart.layout.display = 'none'
            self.skill_dropdown.layout.display = 'none'
            self.bar_chart.layout.display = 'block'
            self.list.layout.display = 'none'

        elif selection == "Current log":
            self.line_chart.layout.display = 'none'
            self.skill_dropdown.layout.display = 'none'
            self.bar_chart.layout.display = 'none'
            self.list.layout.display = 'block'

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
            max_value = 100
            min_value = 0

            if max_value != min_value:  
                df['Value'] = 5 * (df['Value'] - min_value) / (max_value - min_value)
            else:
                df['Value'] = 5 

            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='Skill', y='Value', palette='viridis')
            plt.ylim(0, 5)
            plt.yticks([0, 1, 2, 3, 4, 5], ['Beginner', 'Basic Knowledge', 'Intermediate', 'Experienced', 'Expert', 'Master'])
            plt.title('Skill level distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            return plt

    
    def update_line_chart(self, value):
        value = value["new"]
        self.line_chart.clear_output()
        with self.line_chart:
            df = pd.DataFrame(self.stats)
            df['date'] = df['date'].apply(lambda d: d[0] if isinstance(d, tuple) else d)

            if value and value != "All":
                df = df[['date', value]]

            df_melted = df.melt(id_vars='date', var_name='Competence', value_name='Value')
            max_value = 100
            min_value = 0
            df_melted['Value'] = 5 * (df_melted['Value'] - min_value) / (max_value - min_value)
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df_melted, x='date', y='Value', hue='Competence', marker='o', palette='viridis')
            plt.ylim(0, 5)
            plt.yticks([0, 1, 2, 3, 4, 5], ['Beginner', 'Basic Knowledge', 'Intermediate', 'Experienced', 'Expert', 'Master'])

            if value == "All":
                plt.title('Competence level over time')
            else:
                plt.title('Competence level of ' + value + ' over time')

            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%m-%Y'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


    def get_learning_path_tab(self):
        return self.hbox
    
    def set_stats(self, stats):
        self.stats = stats
        if len(stats) > 0:
            stat_data = stats[-1].copy()
            self.skill_dropdown.options = ["All"] + [item[0] for item in list(stat_data.items())[1:]]
            self.update_bar_chart()
            self.update_line_chart({"new": "All"})