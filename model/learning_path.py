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
        self.skills = []
        self.current_skill_vector = None

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

            
    def get_overlap(self, reference_task, current_task):
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

    def get_maximum_score(self, reference_task, current_performance):
        reference_metric = reference_task["model_metric"]
        
        if reference_metric[0] == "accuracy":
            current_accuracy = current_performance.get_metric("Accuracy")
            if current_accuracy != None:
                return 1-((reference_metric[1] - current_accuracy)/reference_metric[1])*100
                        
        return 0

    def validate_performance(self, reference_task, current_performance):
        # Check data size
        data_size = current_performance.get_metric("data_size")
        min_data_size = reference_task["data_size"] * 0.9
        max_data_size = reference_task["data_size"] * 1.1

        # Data size smaller than the minimum or larger than the maximum data size
        if data_size < min_data_size or data_size > max_data_size:
            return False
        
        # Missing values
        missing_values= current_performance.get_metric("missing_values")
        if missing_values > 0:
            return False
        
        # Type
        type = current_performance.get_metric("type")
        if type != reference_task["type"]:
            return False
        
        # Range
        range = current_performance.get_metric("range")
        if range != reference_task["range"]:
            return False

        return True

    def set_performance_statistics(self):
        self.learning_path.sort(key=lambda x: x.performance['General']['Date'][0])

        for performance in self.learning_path:
            try:
                current_task = self.controller.convert_performance_to_task(performance.performance, "", "")
                target_column = self.get_target(current_task)
                dataset_name = current_task["dataset"].replace(".csv", "")
                reference_task = self.get_reference_task(target_column, dataset_name)

                if not reference_task:
                    continue
                
                #calculate overlap
                overlap = self.get_overlap(reference_task, current_task)
                overlap.update({"dataset": dataset_name, "target": target_column})


                if len(self.skills) >= 1:
                    skill_vector = copy.copy(self.skills[-1])
                else:
                    skill_vector = {}

                # skill_vector = copy.copy(self.skills[-1]) if self.skills else {}
                #add date for graph
                skill_vector['date'] = performance.performance['General']['Date'][0]
                
                valid_performance = self.validate_performance(reference_task, performance)

                if valid_performance:
                    for skill, difficulty in reference_task['difficulty']:
                        # Overlap score is the overlap between the reference task and the current performance
                        # Maximum score is the predictive modeling score
                        # previous skill level
                        overlap_score = overlap.get(skill, 0) * difficulty
                        maximum_score = self.get_maximum_score(reference_task, performance) * difficulty
                        previous_skill_level = skill_vector.get(skill, 0)
                        
                        # update skill vector
                        skill_vector[skill] = max(overlap_score, maximum_score, previous_skill_level)

                    self.skills.append(skill_vector)
                    self.current_skill_vector = skill_vector
            except:
                continue

    def get_stats(self):
        return self.skills
