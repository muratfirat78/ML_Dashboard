class TaskSelectionModel:
    def __init__(self, controller):
        self.controller = controller

    def get_filtered_tasks(self, tasks, guided_mode, all_tasks, current_skill_vector):
        if guided_mode:
            filtered_tasks = [task for task in tasks if task["mode"] == "guided"]
        else:
            filtered_tasks = [task for task in tasks if task["mode"] == "monitored"]
        
        return filtered_tasks
    
    