import pandas as pd
import os
import json

class MainModel:
    def __init__(self, online_version):
        self.curr_df = pd.DataFrame()
        self.currinfo = None
        self.datasplit = False
        self.Xtrain_df = pd.DataFrame()
        self.Xtest_df = pd.DataFrame()
        self.ytrain_df = pd.DataFrame()
        self.ytest_df = pd.DataFrame()
        self.targetcolumn = None
        self.online_version = online_version

        self.tasks_data = None
        self.read_tasks_data()

    def get_online_version(self):
        return self.online_version

    def getYtrain(self):
        return self.ytrain_df

    def set_curr_df(self,mydf):
        self.curr_df = mydf
        return
    def get_curr_df(self):
        return self.curr_df
        

    def get_XTrain(self):
        return self.Xtrain_df
    def get_XTest(self):
        return self.Xtest_df
    def get_YTest(self):
        return self.ytest_df

    def set_XTrain(self,mydf):
        self.Xtrain_df = mydf
        return
    def set_YTrain(self,mydf):
        self.ytrain_df = mydf
        return
    def set_XTest(self,mydf):
        self.Xtest_df = mydf
        return
    def set_YTest(self,mydf):
        self.ytest_df = mydf
        return
    
    def get_tasks_data(self):
        return self.tasks_data
    
    def get_reference_task(self, target_column, dataset):
        tasks = self.tasks_data
        
        for task in tasks:
            if task["dataset"].replace('.csv','') == dataset and task["mode"] == "monitored":
                for subtask in task["subtasks"]:
                    for subsubtask in subtask["subtasks"]:
                        if subsubtask["action"][0] == "DataProcessing" and subsubtask["action"][1]== "AssignTarget":
                            if subsubtask["value"][0] == target_column:
                                return task
        return None
    
    def read_tasks_data(self):
        tasks_data = []
        for filename in os.listdir('./tasks'):
            filepath = os.path.join('./tasks', filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r') as f:
                    loaded_data = json.load(f)
                    tasks_data += [loaded_data]
        self.tasks_data = tasks_data