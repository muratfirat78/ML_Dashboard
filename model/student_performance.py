import ast
from datetime import datetime
import re

class StudentPerformance:
    def __init__(self, controller):
        self.controller = controller
        self.performance = { 
            'General':{},           
            'SelectData': {},
            'DataCleaning': {},
            'DataProcessing': {},
            'ModelDevelopment': {}
            }
        self.index = 0
        self.timestamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

    def addAction(self, action, value):
        value = (value, self.index)
        category, action_type = action[0], action[1]
        
        if category in self.performance:
            if action_type in self.performance[category]:
                if isinstance(self.performance[category][action_type], list):
                    if isinstance(value, list):
                        self.performance[category][action_type].extend(value) 
                    else:
                        self.performance[category][action_type].append(value)
                else:
                    self.performance[category][action_type] = [self.performance[category][action_type], value]
            else:
                self.performance[category][action_type] = value
        else:
            self.performance[category] = {action_type: value}
        
        self.index += 1

    def string_to_student_performance(self, input_str, date):
        input_str = re.sub(r'array\((\[.*?\])\)', r'\1', input_str)
        data_dict = ast.literal_eval(input_str)
        self.addAction(['General','Date'], datetime.strptime(date, "%d-%m-%Y %H-%M-%S"))
        for category, actions in data_dict.items():
            for action_type, value in actions.items():
                if isinstance(value, list):
                    for item in value:
                        self.addAction([category, action_type], [item[0][0]])
                elif isinstance(value, tuple):
                    self.addAction([category, action_type], value[0])
                else:
                    self.addAction([category, action_type], value)
                

    def get_score(self):
        return self.performance
    
    def get_timestamp(self):
        return self.timestamp
    
    def get_results(self):
        if self.performance.get('General', {}).get('Date') and self.performance.get('ModelDevelopment', {}).get('ModelPerformance') and self.performance.get('SelectData', {}).get('DataSet') and self.performance.get('DataProcessing', {}).get('AssignTarget'):
            return (self.performance.get('SelectData', {}).get('DataSet')
                    ,self.performance.get('DataProcessing', {}).get('AssignTarget')
                    ,self.performance.get('ModelDevelopment', {}).get('ModelPerformance')
                    ,self.performance.get('General', {}).get('Date')
                    )

    def get_list_of_actions(self):
        actions = []
        for category, action_dict in self.performance.items():
            for action_type, value in action_dict.items():
                values = value if isinstance(value, list) else [value]
                for v in values:
                    val, idx = v
                    action_str = f"{action_type}: {val}"
                    actions.append((category, action_str, idx))
        actions.sort(key=lambda x: x[2])
        return actions
    
    def get_metric(self, metric_name):
        try:
            for metric in self.performance.get('ModelDevelopment', {}).get('ModelPerformance')[0][1]:
                if metric[0] == metric_name:
                    return metric[1]
        except:
            return None

        