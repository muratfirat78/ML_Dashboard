import ast
from datetime import datetime

class StudentPerformance:
    def __init__(self):
        self.performance = {            
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

    def string_to_student_performance(self, input_str):
        data_dict = ast.literal_eval(input_str)
        for category, actions in data_dict.items():
            for action_type, value in actions.items():
                if isinstance(value, list):
                    for item in value:
                        self.addAction([category, action_type], item)
                else:
                    self.addAction([category, action_type], value)
                
    def get_score(self):
        return self.performance
    
    def get_timestamp(self):
        return self.timestamp
    
    def printperformances(self):
        print(self.performance['ModelDevelopment']['ModelPerformance'])