import copy
import os

from model.student_performance import StudentPerformance

class LearningManagerModel:
    def __init__(self, controller):
        self.controller = controller
        self.learning_path = []

    def get_learning_path(self):
        return self.learning_path

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
                amount_of_subsubtasks = 0
                correct = 0
                for subsubtask in subtask["subtasks"]:
                    for value in subsubtask["value"]:
                        amount_of_subsubtasks += 1
                        if current_performance.action_in_performance(subsubtask["action"],value):
                            correct += 1
                score = correct / amount_of_subsubtasks
                print(score)
                return score
        return 0
    
    def get_overlap_scores(self, reference_task, current_performance):
        result = {}
        for skill, difficulty in reference_task['difficulty']:
            result[skill] = self.get_overlap_score(reference_task, skill, current_performance)
        return result



    def get_overall_score(self, reference_task, performance):
        reference_metric = reference_task["model_metric"]
        
        if reference_metric[0] == "accuracy":
            current_accuracy = performance.get_metric("Accuracy")
            if current_accuracy != None:
                reference_accuracy = reference_metric[1]
                percentage_of_reference = (100/reference_accuracy) * current_accuracy
                result = (min(100, percentage_of_reference)/100)
                return result
                        
        return 0
    
    def get_competence_vector(self, overlap_score, overall_score, task_difficulty, date):
        final_score = {}
        final_score['date'] = date
        for skill, difficulty in task_difficulty:
            overlap = overlap_score.get(skill, 0.0) * difficulty
            overall = overall_score * difficulty
            final_score[skill] = max(overlap,overall)
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

    def set_skill_vectors(self):
        learning_path = self.learning_path
        learning_path.sort(key=lambda x: x.performance['General']['Date'][0])
        
        for performance in learning_path:
                try:
                    current_task = self.controller.convert_performance_to_task(performance.performance, "", "")
                    target_column = self.controller.get_target_task(current_task)
                    dataset_name = current_task["dataset"].replace(".csv", "")
                    reference_task = self.controller.get_reference_task(target_column, dataset_name)       
                    if not reference_task:
                        # reference task not found
                        continue
                    
                    valid_performance = self.validate_performance(reference_task, performance)
                    if valid_performance:
                        date = performance.performance['General']['Date'][0]
                        overlap_score = self.get_overlap_scores(reference_task, performance) #for example: {data_cleaning: 0.3,...}
                        overall_score = self.get_overall_score(reference_task, performance) # for example: 0.3
                        competence_vector = self.get_competence_vector(overlap_score, overall_score, reference_task['difficulty'], date)
                        self.controller.add_skill_vector(competence_vector)

                except Exception as e:
                    print(e)
                    continue
