import math


class TaskSelectionModel:
    def __init__(self, controller):
        self.controller = controller

    def get_recommended_tasks(self, tasks, current_skill_vector):

        # tasks_above_skill_level = []
        # # Remove tasks below skill level
        # for task in tasks:
        #     task_too_easy = False
        #     for difficulty in task["difficulty"]:
        #         task_skill_difficulty_name = difficulty[0]
        #         task_skill_difficulty = difficulty[1]
        #         current_skill_level = current_skill_vector.get(task_skill_difficulty_name)

        #         if current_skill_level > task_skill_difficulty:
        #             task_too_easy = True
        #             break
        #     if task_too_easy == False:
        #         tasks_above_skill_level.append(task)
        
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
                if task_skill_difficulty_name == "Predictive Modeling":
                    current_skill_level = current_skill_vector.get(task_skill_difficulty_name)
                    total_difference += task_skill_difficulty - current_skill_level
            if total_difference > 0:
                task_differences.append((total_difference, task))

        #order recommended tasks by total difference
        task_differences.sort(key=lambda x: x[0])
        
        #remove tasks that are already completed with a 95% predictive modeling score
        dataset_performances = self.controller.get_dataset_performances()
        filtered_tasks = [
            (diff, task) for diff, task in task_differences
            if dataset_performances.get(task["dataset"].replace('.csv',''), 0) <= 0.95
        ]

        amount_of_tasks = len(filtered_tasks)
        learning_rate = self.controller.get_learning_rate()


        for task in filtered_tasks:
            print(task[1]["title"] + ": " + str(task[0]))

        if amount_of_tasks > 3:
            first_idx = 0
            second_idx = max(first_idx+1, round((learning_rate * 0.5) * amount_of_tasks - 1))
            third_idx = max(second_idx+1, round((learning_rate) * amount_of_tasks - 1))
            recommended_tasks = [filtered_tasks[first_idx][1], filtered_tasks[second_idx][1], filtered_tasks[third_idx][1]]
        else:
            # Get the first 3
            recommended_tasks = [task for _, task in filtered_tasks[:min(len(filtered_tasks),3)]]


        
        # if len(filtered_tasks) < 3:
        #     # Get the first 3
            


        

        



        for task in filtered_tasks:
            print(task[1]["title"] + ": " + str(task[0]))
        
        
        print(learning_rate)

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
    