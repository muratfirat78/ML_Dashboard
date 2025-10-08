import copy
import os
from datetime import datetime

from model.student_performance import StudentPerformance

class LearningManagerModel:
    def __init__(self, controller):
        self.controller = controller
        self.performances = []

    def get_learning_path(self):
        return self.performances

    def set_learning_path(self,userid):
        self.performances = []
        path = os.path.join('drive', str(userid))
        for filename in os.listdir(path):
            with open(os.path.join(path,filename),'r') as file:
                for line in file:
                    try:
                        performance = StudentPerformance(self.controller)
                        performance.string_to_student_performance(line, filename.replace('.txt', ''))
                        self.performances.append(performance)
                    except:
                        print("Error reading performance")


    def subsubtask_in_current_task(self, action, value, current_task):
        for subtask in current_task["subtasks"]:
            for subsubtask in subtask["subtasks"]:
                if subsubtask["action"] == action:
                    if value in subsubtask["value"]:
                        return True
        return False
            
    def get_overlap_score(self, reference_task, skill, current_performance):
        for subtask in reference_task["subtasks"]:
            if subtask["title"] == skill:
                if subtask["title"] == 'Predictive Modeling':
                    pass
                
                amount_of_subsubtasks = 0
                correct = 0
                for subsubtask in subtask["subtasks"]:
                    for value in subsubtask["value"]:
                        if subsubtask["action"][1] == 'ParameterFinetuning':
                            #parameter finetuning is one point for each correct parameter
                            for parameter in value:
                                amount_of_subsubtasks += 1
                                if current_performance.action_in_performance(subsubtask["action"],parameter):
                                    correct += 1
                        elif subsubtask["action"][1] == 'ModelPerformance':
                            pass
                        else:
                            amount_of_subsubtasks += 1
                            if current_performance.action_in_performance(subsubtask["action"],value):
                                correct += 1
                if amount_of_subsubtasks == 0:
                    score = 0
                else:
                    score = correct / amount_of_subsubtasks
                return score
        return 0
    
    def get_overlap_scores(self, reference_task, current_performance):
        result = {}
        for skill, difficulty in reference_task['difficulty']:
            result[skill] = self.get_overlap_score(reference_task, skill, current_performance)
        result['Predictive Modeling'] = self.get_predictive_modeling_score(reference_task,current_performance)
        return result

    def get_predictive_modeling_score(self, reference_task, performance):
        reference_metric = reference_task["model_metric"]
        if reference_metric[0] == "accuracy":
            current_accuracy = performance.get_metric("Accuracy")
            if current_accuracy != None:
                reference_accuracy = reference_metric[1]

                if reference_accuracy != 0:
                    percentage_of_reference = (100/reference_accuracy) * current_accuracy
                    result = (min(100, percentage_of_reference)/100)
                    return result   
                
        if reference_metric[0] == "MSE":
            current_mse = performance.get_metric("MSE")
            reference_mse = reference_metric[1]
            if current_mse != None:
                score = reference_mse / current_mse
                score = min(1, score) # If the current MSE is lower (better than reference), the score will be 100% (1)
                score = max(0, score) # take the max of the score and 0 to prevent score being lower than 0
                return score
        return 0
    
    def get_competence_vector(self, overlap_score, task_difficulty, date):
        final_score = {}
        final_score['date'] = date

        predictive_modeling_score = overlap_score["Predictive Modeling"]
        for skill, difficulty in task_difficulty:
            overlap = overlap_score.get(skill, 0.0) * difficulty
            predictive_modeling = predictive_modeling_score * difficulty
            final_score[skill] = max(overlap,predictive_modeling)
        
        return final_score
    
    def get_performance_score(self, overlap_score, task_difficulty, date):
        final_score = {}
        final_score['date'] = date

        predictive_modeling_score = overlap_score["Predictive Modeling"]
        for skill, difficulty in task_difficulty:
            overlap = overlap_score.get(skill, 0.0) * difficulty
            predictive_modeling = predictive_modeling_score * difficulty
            final_score[skill] = max(overlap,predictive_modeling)
        
        return final_score

    def validate_performance(self, reference_task, current_performance):
        # Check data size
        data_size = current_performance.get_metric("data_size")
        min_data_size_threshold = reference_task["data_size"] * 0.9
        max_data_size_threshold = reference_task["data_size"] * 1.1

        # Data size smaller than the minimum or larger than the maximum data size
        if data_size < min_data_size_threshold or data_size > max_data_size_threshold:
            return False
        
        # Missing values
        missing_values= current_performance.get_metric("missing_values")
        if missing_values > 0:
            return False
        
        # Type
        d_type = current_performance.get_metric("type")
        if d_type != reference_task["type"]:
            return False

        # Range
        range = current_performance.get_metric("range")
        
        try:
            # check number range
            min_max = range.split("-")
            min_val = int(min_max[0])
            max_val = int(min_max[1])

            ref_range = reference_task["range"]
            min_max_ref = ref_range.split("-")
            ref_min = int(min_max_ref[0])
            ref_max = int(min_max_ref[1])

            deviation_margin = 10

            min_threshold = ref_min * (1 - (deviation_margin/100))
            max_threshold = ref_max * (1 + (deviation_margin/100))

            if min_val < min_threshold or min_val > max_threshold:
                return False
            
            if max_val < min_threshold or max_val > max_threshold:
                return False
        except:
            # If parsing fails, fall back to simple string comparison
            if range != reference_task["range"]:
                return False
            
        return True
    
    def calculate_performance_score(self, performance, reference_task):
        try:
            current_task = self.controller.convert_performance_to_task(performance, "", "")
            target_column = self.controller.get_target_task(current_task)
            dataset_name = current_task["dataset"].replace(".csv", "")

            if reference_task == None:
                reference_task = self.controller.get_reference_task(target_column, dataset_name)       
                if not reference_task:
                    # reference task not found
                    return None
            valid_performance = self.validate_performance(reference_task, performance)
            if valid_performance:
                overlap_score = self.get_overlap_scores(reference_task, performance) #for example: {data_cleaning: 0.3,...}
                predictive_modeling_score = overlap_score["Predictive Modeling"]

                for skill,score in overlap_score.items():
                    overlap_score[skill] = max(score, predictive_modeling_score)
                
                return overlap_score

            return None

        
        except Exception as e:
            return None

    def calculate_competence_vector(self, performance, reference_task, date):
        try:
            current_task = self.controller.convert_performance_to_task(performance, "", "")
            target_column = self.controller.get_target_task(current_task)
            dataset_name = current_task["dataset"].replace(".csv", "")

            if reference_task == None:
                reference_task = self.controller.get_reference_task(target_column, dataset_name)       
                if not reference_task:
                    # reference task not found
                    return None
            valid_performance = self.validate_performance(reference_task, performance)
            if valid_performance:
                if not date:
                    date = performance.performance['General']['Date'][0]
                overlap_score = self.get_overlap_scores(reference_task, performance) #for example: {data_cleaning: 0.3,...}
                competence_vector = self.get_competence_vector(overlap_score, reference_task['difficulty'], date)
                return competence_vector

        except Exception as e:
            # print("error:")
            # print(e)
            return None

    def get_task_skill_difficulty(self, task_difficulty, skill):
        for task_skill in task_difficulty:
            if task_skill[0] == skill:
                return task_skill[1]
        
        return None


    def update_competence_vector(self, performance_score, current_competence_vector, task_difficulty, date):
        try:
            updated_competence_vector = {}
            for skill,score in current_competence_vector.items():
                if skill == "date":
                    updated_competence_vector["date"] = date
                else:
                    if score == 0:
                        updated_competence_vector[skill] = performance_score[skill] * self.get_task_skill_difficulty(task_difficulty, skill)
                    else:
                        updated_competence_vector[skill] = 0.5 * (score + performance_score[skill] * self.get_task_skill_difficulty(task_difficulty, skill))
            self.controller.add_competence_vector(updated_competence_vector)
        except:
            None #updating competence vector failed

    def get_skills_from_tasks(self):
        skills = []
        tasks = self.controller.get_tasks_data()

        for task in tasks:
            for skill in task["difficulty"]:
                if skill[0] not in skills:
                    skills.append(skill[0])
        return skills


    def set_competence_vectors(self):
        initial_competence_vector = {}

        for skill in self.get_skills_from_tasks():
            initial_competence_vector[skill] = 0
        
        performances = self.performances
        performances.sort(key=lambda x: x.performance['General']['Date'][0])

        if len(performances) == 0:
            date = datetime.now()
        else:
            date = performances[0].performance['General']['Date'][0]
        
        initial_competence_vector["date"] = date
        self.controller.add_competence_vector(initial_competence_vector)

        current_competence_vector = initial_competence_vector

        for performance in performances:
            current_task = self.controller.convert_performance_to_task(performance, "", "")
            target_column = self.controller.get_target_task(current_task)
            dataset_name = current_task["dataset"].replace(".csv", "")
            reference_task = self.controller.get_reference_task(target_column, dataset_name)
            if reference_task:       
                task_difficulty = reference_task["difficulty"]
                performance_score = self.calculate_performance_score(performance,reference_task)
                self.update_competence_vector(performance_score, current_competence_vector, task_difficulty, performance.performance['General']['Date'][0])


