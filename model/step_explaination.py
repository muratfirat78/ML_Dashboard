
class StepExplainationModel:
    def __init__(self):
        self.explaination_dict = {    'Scaling':'Normalization: This method scales each feature so that all values are within the range of 0 and 1. \nStandardization: Here, each feature is transformed to have a mean of 0 and a standard deviation of 1.',
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

    def get_explaination(self,step):
        return self.explaination_dict.get(step, "")