import pandas as pd

class MainModel:
    def __init__(self):
        self.curr_df = pd.DataFrame()
        self.Xtrain_df = pd.DataFrame()
        self.Xtest_df = pd.DataFrame()
        self.ytrain_df = pd.DataFrame()
        self.ytest_df = pd.DataFrame()
        self.targetcolumn = None
        self.online_version = False
    
    def get_online_version(self):
        return self.online_version