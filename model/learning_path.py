import os
import copy
from model.student_performance import StudentPerformance

class LearningPathModel:
    def __init__(self, controller):
        self.controller = controller
        self.performance_data = []
        self.skill_vectors = []
        self.current_skill_vector = None

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
        # print(self.skill_vectors)

    def get_current_skill_vector(self):
        return self.current_skill_vector