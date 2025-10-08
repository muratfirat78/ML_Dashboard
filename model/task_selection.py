import math


class TaskSelectionModel:
    def __init__(self, controller):
        self.controller = controller

    def get_recommended_tasks(self, tasks, current_skill_vector):

        tasks_above_skill_level = []
        # Remove tasks below skill level
        for task in tasks:
            task_too_easy = False
            for difficulty in task["difficulty"]:
                task_skill_difficulty_name = difficulty[0]
                task_skill_difficulty = difficulty[1]
                current_skill_level = current_skill_vector.get(task_skill_difficulty_name)

                if current_skill_level > task_skill_difficulty:
                    task_too_easy = True
                    break
            if task_too_easy == False:
                tasks_above_skill_level.append(task)
        
        # For now: do not remove the tasks below skill level, because the recommendations will run out too quickly this way
        tasks_above_skill_level = tasks
        # Order tasks
        task_differences = []
        # order tasks on difficulty and get the first three
        for task in tasks_above_skill_level:
            total_difference = 0
            for difficulty in task["difficulty"]:
                task_skill_difficulty_name = difficulty[0]
                task_skill_difficulty = difficulty[1]
                if task_skill_difficulty_name == "Predictive Modeling" or task_skill_difficulty_name == "Model Training":
                    current_skill_level = current_skill_vector.get(task_skill_difficulty_name)
                    total_difference += task_skill_difficulty - current_skill_level
            if total_difference > 0:
                task_differences.append((total_difference, task))

        #order recommended tasks by total difference and get the first 3
        task_differences.sort(key=lambda x: x[0])
        recommended_tasks = [task for _, task in task_differences[:min(len(task_differences),3)]]
        return recommended_tasks

    def get_filtered_tasks(self, tasks, guided_mode, recommendations_only, current_skill_vector):
        if guided_mode:
            filtered_tasks = [task for task in tasks if task["mode"] == "guided"]
        else:
            filtered_tasks = [task for task in tasks if task["mode"] == "monitored"]
        
        if recommendations_only:
            recommended_tasks = self.get_recommended_tasks(filtered_tasks, current_skill_vector)
            if len(recommended_tasks) > 0:
                return recommended_tasks
            else:
                #no more recommendations, return filtered tasks
                return filtered_tasks
        
        return filtered_tasks
    