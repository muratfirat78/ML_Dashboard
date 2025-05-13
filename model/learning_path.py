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

    def set_performance_data(self):
        self.performance_data = []
        performance_data = self.performance_data
        for performance in self.learning_path:
            results = performance.get_results()

            if results:
                dataset = results[0][0][0]
                target =  results[1][0][0]
                res = results[2]

                # if dataset not in performance_data:
                #     performance_data[dataset] = {}

                # if target not in performance_data[dataset]:
                #     performance_data[dataset][target] = []
                if isinstance(res, list):
                    for result in res:
                        model_info = result[0]
                        model_name = model_info[0][0].split('(')[0]
                        metrics = model_info[0][1]
                        date = results[3]
                        performance_data += [{
                            "target": target,
                            "dataset":dataset,
                            "date": date,
                            "model": model_name,
                            "metrics": dict(metrics)
                        }]
                else:
                        model_info = res[0]
                        model_name = model_info[0][0].split('(')[0]
                        metrics = res[0][0][1]
                        date = results[3]
                        performance_data += [{
                            "target": target,
                            "dataset":dataset,
                            "date": date,
                            "model": model_name,
                            "metrics": dict(metrics)
                        }]

    def set_dataset_info(self):
        self.dataset_info = []
        tasks_data = self.controller.get_tasks_data()
        tasks_data_filtered = [task for task in tasks_data if task.get("mode") == "monitored"]
        for task in tasks_data_filtered:
            target = task["target"]
            dataset = task["dataset"].replace(".csv", "")
            difficulty_vector = task["difficulty"]

            self.dataset_info.append({"dataset": dataset,"target": target, "difficulty": difficulty_vector})
        
        # path = './DataSets'
        # for filename in os.listdir(path):
        #     if filename.startswith('Info'):
        #         with open(os.path.join(path,filename),'r') as file:
        #             target = None
        #             difficulty_vector = None
        #             for line in file:
        #                 if line.startswith('target:'):
        #                     target = line.replace('target: ', '').rstrip()

        #                 if line.startswith('difficulty:'):
        #                     difficulty_vector = ast.literal_eval(line.replace('difficulty: ', ''))

        #             if target != None and difficulty_vector != None:
        #                 self.dataset_info.append({"dataset": filename.replace('Info_','').replace('.txt',''),"target": target, "difficulty": difficulty_vector})

    def get_dataset_info(self, dataset, target):
        for info in self.dataset_info:
            if info['dataset'] == dataset and info['target'] == target:
                return info
    
    def set_performance_statistics(self):
        self.stats = []
        sorted_performance_data = sorted(self.performance_data, key=lambda x: x['date'][0])
        for performance in sorted_performance_data:
            dataset_info = self.get_dataset_info(performance['dataset'].replace('.csv',''), performance['target'])
            if dataset_info:
                #todo calculate score
                score = 0.1

                if len(self.stats) >= 1:
                    stat = copy.copy(self.stats[-1])
                else:
                    stat = {}
                    
                stat['date'] = performance['date']
                for difficulty in dataset_info['difficulty']:
                    increase = score * difficulty[1]
                    if difficulty[0] in stat:
                        stat[difficulty[0]] += increase
                        if stat[difficulty[0]] > 100:
                            stat[difficulty[0]] = 100
                    else:
                        stat[difficulty[0]] = increase
                self.stats += [stat]

    def get_stats(self):
        return self.stats
