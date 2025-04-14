import os

from model.student_performance import StudentPerformance

class LearningPathModel:
    def __init__(self):
        self.learning_path = []

    def set_learning_path(self,userid):
        path = os.path.join('drive', str(userid))
        for filename in os.listdir(path):
            with open(os.path.join(path,filename),'r') as file:
                for line in file:
                    try:
                        performance = StudentPerformance()
                        performance.string_to_student_performance(line)
                        self.learning_path.append(performance)
                    except:
                        print("Error reading performance")

    def get_scores(self):
        print(self.learning_path)
        for performance in self.learning_path:
            performance.printperformances()

