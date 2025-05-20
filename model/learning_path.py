import os
import ast
import copy
from model.student_performance import StudentPerformance

class LearningPathModel:
    def __init__(self, controller):
        self.controller = controller
        self.learning_path = []
        self.performance_data = []
        self.dataset_info = []
        self.stats = []
        self.scores = []

    def set_learning_path(self,userid):
        self.learning_path = []
        path = os.path.join('drive', str(userid))
        for filename in os.listdir(path):
            with open(os.path.join(path,filename),'r') as file:
                for line in file:
                    try:
                        performance = StudentPerformance(self.controller)
                        performance.string_to_student_performance(line, filename.replace('.txt', ''))
                        self.learning_path.append(performance)
                    except:
                        print("Error reading performance")

    def get_target(self, student_performance):
        for subtask in student_performance["subtasks"]:
            for subsubtask in subtask["subtasks"]:
                if subsubtask["action"][0] == "DataProcessing" and subsubtask["action"][1]== "AssignTarget":
                    return subsubtask["value"][0]
        return None

    def get_dataset(self, student_performance):
        for subtask in student_performance["subtasks"]:
            for subsubtask in subtask["subtasks"]:
                if subsubtask["action"][0] == "DataProcessing" and subsubtask["action"][1]== "AssignTarget":
                    return subsubtask["value"][0]
        return None

     
    def get_reference_task(self, target_column, dataset):
        tasks = self.controller.get_tasks_data()
        
        for task in tasks:
            if task["dataset"].replace('.csv','') == dataset and task["mode"] == "monitored":
                for subtask in task["subtasks"]:
                    for subsubtask in subtask["subtasks"]:
                        if subsubtask["action"][0] == "DataProcessing" and subsubtask["action"][1]== "AssignTarget":
                            if subsubtask["value"][0] == target_column:
                                return task
        return None

    def get_model_performance(self, task):
        for subtask in task["subtasks"]:
            for subsubtask in subtask["subtasks"]:
                if subsubtask["action"][0] == "ModelDevelopment"and subsubtask["action"][1]=="ModelPerformance":
                    return subsubtask
        return None

    def get_dataset_info(self, dataset, target):
        for info in self.dataset_info:
            if info['dataset'] == dataset and info['target'] == target:
                return info
            
    def subsubtask_in_current_task(self, action, value, current_task):
        for subtask in current_task["subtasks"]:
            for subsubtask in subtask["subtasks"]:
                if subsubtask["action"] == action:
                    if value in subsubtask["value"]:
                        return True
        return False

            
    def get_score(self, reference_task, current_task):
        result = {}
        for subtask in reference_task["subtasks"]:
            amount_of_subsubtasks = 0
            correct = 0
            for subsubtask in subtask["subtasks"]:
                for value in subsubtask["value"]:
                    amount_of_subsubtasks += 1
                    if self.subsubtask_in_current_task(subsubtask["action"],value, current_task):
                        correct += 1

            score = correct / amount_of_subsubtasks
            result[subtask["title"]] = score
        return result

    

    def set_performance_statistics(self):
        self.learning_path.sort(key=lambda x: x.performance['General']['Date'][0])

        for performance_entry in self.learning_path:
            current_task = self.controller.convert_performance_to_task(performance_entry.performance, "", "")
            target_column = self.get_target(current_task)
            dataset_name = current_task["dataset"].replace(".csv", "")
            reference_task = self.get_reference_task(target_column, dataset_name)

            if not reference_task:
                continue

            score = self.get_score(reference_task, current_task)
            score.update({"dataset": dataset_name, "target": target_column})
            self.scores.append(score)

            stat = copy.copy(self.stats[-1]) if self.stats else {}
            stat['date'] = performance_entry.performance['General']['Date'][0]

            for difficulty_name, difficulty_weight in reference_task['difficulty']:
                base_score = score.get(difficulty_name, 0)
                weighted_score = base_score * difficulty_weight

                stat[difficulty_name] = max(stat.get(difficulty_name, 0), weighted_score)

            self.stats.append(stat)

    def get_stats(self):
        return self.stats
