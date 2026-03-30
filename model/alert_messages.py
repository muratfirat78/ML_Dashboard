import re

class AlertMessagesModel:
    # This model class is used to store the error/warning text shown in the ML-Dashboard
    def __init__(self):
        self.messages = {
            #errors/warnings
            "data_split_error" : "Error: at least one array or dtype is required, probably the data needs to be split. You can find more info here:  https://www.techtarget.com/searchenterpriseai/definition/data-splitting",
            "non_numerical_error":"Error: could not convert ... to float, probably there are non numerical values in the dataset",
            "no_target_error": "Error: Can only concatenate str, probably a target needs to be selected.",
            "no_model_error":"Error: A ML model needs to be selected first.",
            "improper_action_error": "Error: improper action selected",

            #step explaination
            'Scaling':'Normalization: This method scales each feature so that all values are within the range of 0 and 1. \nStandardization: Here, each feature is transformed to have a mean of 0 and a standard deviation of 1.',
            'Encoding':'Label encoding assigns a unique integer to each category. Similar to label encoding, ordinal encoding assigns integers to categories, but it\'s specifically designed for data with a natural order.  \nOne Hot Encoding is a method for converting categorical variables into a binary format.',
            'Feature Extraction':'Principal Component Analysis (PCA) is a machine learning technique for dimensionality reduction that transforms a large set of features into a smaller set of new, uncorrelated variables called principal components. \nCorrelation can be used to determine the correlation between the features',
            'Outlier':'Interquartile Range (IQR) is a technique that detects outliers by measuring the variability in a dataset.\nZ Score Formula​:​ The farther away from 0, higher the chance of a given data point being an outlier.',
            'Imbalancedness':'Downsampling is a common data processing technique that addresses imbalances in a dataset by removing data from the majority class such that it matches the size of the minority class. \nThis is opposed to upsampling, which involves resampling minority class points.',
            'Convert Feature 0/1->Bool':'Convert 0 and 1 values to boolean',
            'Assign Target':'Choose the target column',
            'Data Split':'Data splitting is a crucial process in machine learning, involving the partitioning of a dataset into different subsets, training and test',
            'Drop Column':'Remove feature from the data',
            'Remove-Missing':'Remove rows with missing values',
            'Replace-Mean':'Replace missing values with the mean of the feature',
            'Replace-Median':'Replace missing values with the median of the feature',
            'Replace-Mode':'Replace missing values with the mode of the feature'
        }



    def get_message(self, name):
        return self.messages.get(name, "")
    
    def get_message_html(self, name):
        message = self.get_message(name)
        if not message:
            return None
        
        url_pattern = r'(https?://[^\s]+)'
        message_html = re.sub(
            url_pattern,
            r'<a href="\1" target="_blank" '
            r'style="color:#1a73e8; text-decoration:underline; font-weight:500;">\1 ↗</a>',
            message
        )
        return message_html