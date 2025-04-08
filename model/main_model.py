import pandas as pd

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