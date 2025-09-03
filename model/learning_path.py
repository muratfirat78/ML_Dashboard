import os
import copy
from model.student_performance import StudentPerformance

class LearningPathModel:
    def __init__(self, controller):
        self.controller = controller
        self.performance_data = []
        self.competence_vectors = []

    def get_competence_vectors(self):
        return self.competence_vectors
    
    def get_graph_data(self):
        graph_data = []

        for competence_vector in self.competence_vectors:
            if not graph_data:
                graph_data.append(competence_vector)
                continue

            new_vector = copy.copy(graph_data[-1])
            new_vector["date"] = competence_vector.get("date", new_vector.get("date"))

            for skill, value in competence_vector.items():
                if skill == "date":
                    continue
                new_vector[skill] = max(value, new_vector.get(skill, value))

            graph_data.append(new_vector)

        return graph_data
                
    
    def add_competence_vector(self,vector):
        self.competence_vectors.append(vector)

    def get_current_competence_vector(self):
        if len(self.competence_vectors) > 0:
            # Return the most recent competence vector
            return self.competence_vectors[-1]
        else:
            return None