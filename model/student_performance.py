class StudentPerformance:
    def __init__(self):
        self.performance = {            
            'SelectData': {},
            'DataCleaning': {},
            'DataProcessing': {},
            'ModelDevelopment': {}
            }

    def addAction(self, action, value):
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
        
        print(self.performance)