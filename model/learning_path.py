import os
import copy
from model.student_performance import StudentPerformance

class LearningPathModel:
    def __init__(self, controller):
        self.controller = controller
        self.learning_path = []
        self.performance_data = []
        self.skill_vectors = []
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

    def get_learning_path(self):
        return self.learning_path

    def get_skill_vectors(self):
        return self.skill_vectors
    
    def get_graph_data(self):
        graph_data = []

        for skill_vector in self.skill_vectors:
            if not graph_data:
                graph_data.append(skill_vector)
                continue

            new_vector = copy.copy(graph_data[-1])
            new_vector["date"] = skill_vector.get("date", new_vector.get("date"))

            for skill, value in skill_vector.items():
                if skill == "date":
                    continue
                new_vector[skill] = max(value, new_vector.get(skill, value))

            graph_data.append(new_vector)

        return graph_data
                
    
    def add_skill_vector(self,vector):
        self.skill_vectors.append(vector)
        print(self.skill_vectors)

    def get_current_skill_vector(self):
        return self.current_skill_vector