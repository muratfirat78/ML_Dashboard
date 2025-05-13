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
        self.tasks_data = [{
                            "title": "Titanic survival chance prediction",
                            "description": "It’s the year 1912. The RMS Titanic has tragically struck an iceberg and begun sinking. As a data scientist in a secret rescue agency, you've been granted access to passenger data from the ship’s manifest. Your mission? Build a model that predicts who had the highest chance of survival.",
                            "dataset": "titanic.csv",
                            "mode": "guided",
                            "difficulty": [('Data cleaning',60), ('Data Translation',80), ('Data Transformation', 70), ('Statistics', 65), ('Feature Selection', 50), ('Model Training', 30)],
                            "subtasks": [
                            {
                                "title": "Data Cleaning",
                                "status": "todo",
                                "order": 1,
                                "subtasks": [
                                {
                                    "title": "Replace missing Age with median",
                                    "description": "Fill in missing 'Age' values using the median age.",
                                    "hints": ["Use median() function", "Apply to Age column"],
                                    "status": "todo",
                                    "action": ["DataCleaning", "Replace-Median"],
                                    "value": ["Age"],    
                                    "applied_values": [],   
                                    "order": 1
                                },
                                {
                                    "title": "Remove rows with missing 'Embarked'",
                                    "description": "Remove rows where 'Embarked' is missing.",
                                    "hints": ["", ""],
                                    "status": "todo",
                                    "action": ["DataCleaning", "Remove-Missing"],
                                    "value": ["Embarked"],   
                                    "applied_values": [],                                   
                                    "order": 1
                                },
                                {
                                    "title": "Drop unnecessary columns",
                                    "description": "Drop columns not useful for modeling.",
                                    "hints": ["Look for ID-like, ticket, or name-based columns"],
                                    "status": "todo",
                                    "action": ["DataCleaning", "Drop Column"],
                                    "value": ["Cabin", "PassengerId", "Pclass", "Name", "SibSp", "Parch", "Ticket", "Fare"],
                                    "applied_values": [],  
                                    "order": 1
                                }
                                ]
                            },
                            {
                                "title": "Data Translation",
                                "status": "todo",
                                "order": 2,
                                "subtasks": [
                                {
                                    "title": "Encode categorical columns",
                                    "description": "Convert categorical variables into numeric form.",
                                    "hints": ["Use LabelEncoder for 'Sex' and 'Embarked'"],
                                    "status": "todo",
                                    "action": ["DataProcessing", "LabelEncoding"],
                                    "value": ["Sex", "Embarked"],
                                    "applied_values": [],  
                                    "order": 1
                                },
                                {
                                    "title": "Convert Survived to Boolean",
                                    "description": "Change the Survived column to boolean values.",
                                    "hints": ["True if survived, False if not"],
                                    "status": "todo",
                                    "action": ["DataProcessing", "ConvertToBoolean"],
                                    "value": ["Survived"],
                                    "applied_values": [],  
                                    "order": 1
                                }
                                ]
                            },
                            {
                                "title": "Model Training",
                                "status": "todo",
                                "order": 3,
                                "subtasks": [
                                {
                                    "title": "Assign Target",
                                    "description": "Assign 'Survived' as the target variable for the model.",
                                    "hints": ["Separate X and y"],
                                    "status": "todo",
                                    "action": ["DataProcessing", "AssignTarget"],
                                    "value": ["Survived"],
                                    "applied_values": [],  
                                    "order": 1
                                },
                                {
                                    "title": "Train/Test Split",
                                    "description": "Split the dataset into training and testing sets.",
                                    "hints": ["Use 80/20 split"],
                                    "status": "todo",
                                    "action": ["DataProcessing", "Split"],
                                    "value": ["20%"],
                                    "applied_values": [],  
                                    "order": 2
                                },
                                {
                                    "title": "Train Logistic Regression",
                                    "description": "Train a logistic regression model and evaluate performance.",
                                    "hints": ["Use LogisticRegression", "Check confusion matrix"],
                                    "status": "todo",
                                    "action": ["ModelDevelopment", "ModelPerformance"],
                                    "value": ["Logistic Regression()"],
                                    "applied_values": [],  
                                    "order": 3
                                }
                                ]
                            }
                            ]
                        },{'title': 'todo'
                           , 'description': 'todo'
                           ,"dataset": "titanic.csv"
                            ,"mode": "guided"
                            ,"difficulty": [('Data cleaning',60), ('Data Translation',80), ('Data Transformation', 70), ('Statistics', 65), ('Feature Selection', 50), ('Model Training', 30)]
                           ,'subtasks': [
                            {'title': 'Data Cleaning', 'status': 'todo', 'subtasks': [
                                    {'status': 'todo', 'action': ['DataCleaning', 'Drop Column'
                                        ], 'value': ['Fare', 'Ticket', 'Parch', 'SibSp', 'Name', 'Pclass', 'PassengerId', 'Cabin'
                                        ], 'title': 'Drop Columns', 'description': 'Remove columns that are irrelevant or redundant for the model.', 'hints': [
                                            "Go to the Data Cleaning tab. Select the column(s), choose 'drop column', then click apply.", 'Drop the following column(s): Fare, Ticket, Parch, SibSp, Name, Pclass, PassengerId, Cabin.'
                                        ], 'applied_values': [], 'order': 1
                                    },
                                    {'status': 'todo', 'action': ['DataCleaning', 'Remove-Missing'
                                        ], 'value': ['Embarked'
                                        ], 'title': 'Remove Rows with Missing Values', 'description': 'Remove rows that contain missing values in the column.', 'hints': [
                                            "Go to the Data Cleaning tab. Select the column, choose 'remove missing', then click apply.", 'Remove rows with missing values in Embarked.'
                                        ], 'applied_values': [], 'order': 1
                                    },
                                    {'status': 'todo', 'action': ['DataCleaning', 'Replace-Median'
                                        ], 'value': ['Age'
                                        ], 'title': 'Replace Missing Values (Median)', 'description': 'Replace missing values in a column using the median.', 'hints': [
                                            "Go to the Data Cleaning tab. Select the column, choose 'replace median', then click apply.", 'Apply replace median to Age.'
                                        ], 'applied_values': [], 'order': 1
                                    }
                                ], 'applied_values': [], 'order': 2
                            },
                            {'title': 'Data Translation', 'status': 'todo', 'subtasks': [
                                    {'status': 'todo', 'action': ['DataProcessing', 'LabelEncoding'
                                        ], 'value': ['Embarked', 'Sex'
                                        ], 'title': 'Label Encoding', 'description': 'Convert categorical columns into numeric codes.', 'hints': [
                                            "Go to the Data Processing tab. Choose the column, select 'encoding', choose 'Label Encoding', then click apply.", 'Apply label encoding to Embarked, Sex.'
                                        ], 'applied_values': [], 'order': 1
                                    },
                                    {'status': 'todo', 'action': ['DataProcessing', 'ConvertToBoolean'
                                        ], 'value': ['Survived'
                                        ], 'title': 'Convert to Boolean', 'description': 'Convert 1 and 0 values in the column to boolean (True/False).', 'hints': [
                                            "Go to the Data Processing tab. Choose the column, select 'Convert to Boolean', then click apply.",
                                            "Convert 'Survived' to boolean (True/False)."
                                        ], 'applied_values': [], 'order': 1
                                    }
                                ], 'applied_values': [], 'order': 3
                            },
                            {'title': 'Model Training', 'status': 'todo', 'subtasks': [
                                    {'status': 'todo', 'action': ['DataProcessing', 'AssignTarget'
                                        ], 'value': ['Survived'
                                        ], 'title': 'Assign Target Variable', 'description': 'Assign the column as the target variable for model training.', 'hints': [
                                            "Go to the Data Processing tab. Choose the column, select 'Assign Target', then click apply.",
                                            "Assign 'Survived' as the target column."
                                        ], 'applied_values': [], 'order': 1
                                    },
                                    {'status': 'todo', 'action': ['DataProcessing', 'Split'
                                        ], 'value': ['20%'
                                        ], 'title': 'Train/Test Split', 'description': 'Split dataset into training and testing sets.', 'hints': ['Go to the Data Processing tab. Choose the column, assign the target, set the test ratio, then click split.', 'Use a test ratio of 20% for splitting the dataset.'
                                        ], 'applied_values': [], 'order': 2
                                    },
                                    {'status': 'todo', 'action': ['ModelDevelopment', 'ModelPerformance'
                                        ], 'value': ['Logistic Regression()'
                                        ], 'title': 'Model Training & Evaluation', 'description': 'Train a model and evaluate its performance.', 'hints': ['Go to the Predictive Modeling tab. Select the model type, adjust parameters if needed, then click train.', 'Train the model using the selected parameters and evaluate its performance.'
                                        ], 'applied_values': [], 'order': 3
                                    }
                                ], 'applied_values': [], 'order': 7
                            }
                        ]
                    },
                    {
                        "title":"World happiness report",
                        "mode": "monitored",
                        "dataset": "World Happiness Report.csv",
                        "difficulty": [('Data cleaning',60), ('Data Translation',80), ('Data Transformation', 70), ('Statistics', 65), ('Feature Selection', 50), ('Model Training', 30)],
                        "target": "Life_Satisfaction",
                        "description":"""
                                        <b>About Dataset</b><br>
                                        This dataset contains 4,000 entries with 24 columns related to happiness, economic, social, and political indicators for different countries across multiple years.<br><br>
                                        <b>Columns Overview:</b><br>
                                        <b>Country</b>: Name of the country.<br>
                                        <b>target</b>: Life_Satisfaction<br>
                                        <b>Year</b>: The year of the record.<br>
                                        <b>Happiness_Score</b>: A numerical value indicating the happiness level.<br>
                                        <b>GDP_per_Capita</b>: Economic output per person.<br>
                                        <b>Social_Support</b>: Level of social connections and support.<br>
                                        <b>Healthy_Life_Expectancy</b>: Average life expectancy with good health.<br>
                                        <b>Freedom</b>: Perceived freedom in decision-making.<br>
                                        <b>Generosity</b>: A measure of charitable behavior.<br>
                                        <b>Corruption_Perception</b>: Perception of corruption in society.<br>
                                        <b>Unemployment_Rate</b>: Percentage of unemployed individuals.<br>
                                        <b>Education_Index</b>: A measure of education quality.<br>
                                        <b>Population</b>: Total population of the country.<br>
                                        <b>Urbanization_Rate</b>: Percentage of people living in urban areas.<br>
                                        <b>Life_Satisfaction (Target)</b>: A subjective measure of well-being.<br>
                                        <b>Public_Trust</b>: Confidence in public institutions.<br>
                                        <b>Mental_Health_Index</b>: A measure of overall mental health.<br>
                                        <b>Income_Inequality</b>: Economic disparity metric.<br>
                                        <b>Public_Health_Expenditure</b>: Government spending on health.<br>
                                        <b>Climate_Index</b>: A measure of climate conditions.<br>
                                        <b>Work_Life_Balance</b>: An index measuring work-life balance.<br>
                                        <b>Internet_Access</b>: Percentage of population with internet.<br>
                                        <b>Crime_Rate</b>: Reported crime level.<br>
                                        <b>Political_Stability</b>: A measure of political security.<br>
                                        <b>Employment_Rate</b>: Percentage of employed individuals.
                                        """
                    },
                    {
                        "title":"Titanic",
                        "mode": "monitored",
                        "dataset": "titanic.csv",
                        "difficulty": [('Data cleaning',60), ('Data Translation',80), ('Data Transformation', 70), ('Statistics', 65), ('Feature Selection', 50), ('Model Training', 30)],
                        "target": "Survived",
                        "description":"""
                                        "It’s the year 1912. The RMS Titanic has tragically struck an iceberg and begun sinking. As a data scientist in a secret rescue agency, you've been granted access to passenger data from the ship’s manifest. Your mission? Build a model that predicts who had the highest chance of survival."
                                        """
                    },
                    {
                        "title":"Red Wine Quality",
                        "mode": "monitored",
                        "dataset": "Red Wine Quality.csv",
                        "difficulty": [('Data cleaning',60), ('Data Translation',80), ('Data Transformation', 70), ('Statistics', 65), ('Feature Selection', 50), ('Model Training', 30)],
                        "target": "quality",
                        "description":"""
                                        <b>target:</b> quality<br>

                                        <p>For more information, read <a href="https://scholar.google.com/scholar?q=Cortez+et+al.,+2009" target="_blank">Cortez et al., 2009</a>.</p>

                                        <b>Input variables</b> (based on physicochemical tests):<br>
                                        <ol>
                                        <li>fixed acidity</li>
                                        <li>volatile acidity</li>
                                        <li>citric acid</li>
                                        <li>residual sugar</li>
                                        <li>chlorides</li>
                                        <li>free sulfur dioxide</li>
                                        <li>total sulfur dioxide</li>
                                        <li>density</li>
                                        <li>pH</li>
                                        <li>sulphates</li>
                                        <li>alcohol</li>
                                        </ol>

                                        <b>Output variable</b> (based on sensory data):<br>
                                        <ol start="12">
                                        <li>quality (score between 0 and 10)</li>
                                        </ol>

                                        <p><b>Use machine learning</b> to determine which physiochemical properties make a wine 'good'!</p>
                                        """
                    },
                    {
                        "title":"Student Performance & Behavior Dataset",
                        "mode": "monitored",
                        "dataset": "Student Performance and Behavior Dataset.csv",
                        "difficulty": [('Data cleaning',60), ('Data Translation',80), ('Data Transformation', 70), ('Statistics', 65), ('Feature Selection', 50), ('Model Training', 30)],
                        "target": "Total_Score",
                        "description":"""
                                        <b>Student Performance & Behavior Dataset</b><br><br>

                                        <p>This dataset is real data of 5,000 records collected from a private learning provider.  
                                        The dataset includes key attributes necessary for exploring patterns, correlations, and insights related to academic performance.</p>

                                        <b>Features:</b><br><br>

                                        <b>Target:</b> Total_Score<br><br>

                                        <b>Feature Descriptions:</b>
                                        <ul>
                                        <li><b>Student_ID</b>: Unique identifier for each student.</li>
                                        <li><b>First_Name</b>: Student’s first name.</li>
                                        <li><b>Last_Name</b>: Student’s last name.</li>
                                        <li><b>Email</b>: Contact email (can be anonymized).</li>
                                        <li><b>Gender</b>: Male, Female, Other.</li>
                                        <li><b>Age</b>: The age of the student.</li>
                                        <li><b>Department</b>: Student's department (e.g., CS, Engineering, Business).</li>
                                        <li><b>Attendance (%)</b>: Attendance percentage (0-100%).</li>
                                        <li><b>Midterm_Score</b>: Midterm exam score (out of 100).</li>
                                        <li><b>Final_Score</b>: Final exam score (out of 100).</li>
                                        <li><b>Assignments_Avg</b>: Average score of all assignments (out of 100).</li>
                                        <li><b>Quizzes_Avg</b>: Average quiz scores (out of 100).</li>
                                        <li><b>Participation_Score</b>: Score based on class participation (0-10).</li>
                                        <li><b>Projects_Score</b>: Project evaluation score (out of 100).</li>
                                        <li><b>Total_Score</b>: Weighted sum of all grades.</li>
                                        <li><b>Grade</b>: Letter grade (A, B, C, D, F).</li>
                                        <li><b>Study_Hours_per_Week</b>: Average study hours per week.</li>
                                        <li><b>Extracurricular_Activities</b>: Whether the student participates in extracurriculars (Yes/No).</li>
                                        <li><b>Internet_Access_at_Home</b>: Does the student have access to the internet at home? (Yes/No).</li>
                                        <li><b>Parent_Education_Level</b>: Highest education level of parents (None, High School, Bachelor's, Master's, PhD).</li>
                                        <li><b>Family_Income_Level</b>: Low, Medium, High.</li>
                                        <li><b>Stress_Level (1-10)</b>: Self-reported stress level (1: Low, 10: High).</li>
                                        <li><b>Sleep_Hours_per_Night</b>: Average hours of sleep per night.</li>
                                        </ul>

                                        <b>Dataset contains:</b>
                                        <ul>
                                        <li><b>Missing values</b> (nulls): in some records (e.g., Attendance, Assignments, or Parent Education Level).</li>
                                        <li><b>Bias</b> in some data (e.g., students with high attendance get slightly better grades).</li>
                                        <li><b>Imbalanced distributions</b>: some departments have more students than others.</li>
                                        </ul>
                                        """
                    },
                    {
                        "title":"Heart Prediction Dataset",
                        "mode": "monitored",
                        "dataset": "Heart Prediction Dataset.csv",
                        "difficulty": [('Data cleaning',60), ('Data Translation',80), ('Data Transformation', 70), ('Statistics', 65), ('Feature Selection', 50), ('Model Training', 30)],
                        "target": "HeartDisease",
                        "description":"""
                                        <b>About this file</b><br><br>

                                        <b>Filename:</b> Heart Prediction Quantum Dataset.csv<br>
                                        <b>Rows:</b> 500<br>
                                        <b>Columns:</b> 7<br><br>

                                        <b>Target:</b> HeartDisease<br><br>

                                        <b>Feature Descriptions:</b>
                                        <ul>
                                        <li><b>Age:</b> Patient's age in years</li>
                                        <li><b>Gender:</b> 0 (Female), 1 (Male)</li>
                                        <li><b>BloodPressure:</b> Blood pressure level</li>
                                        <li><b>Cholesterol:</b> Cholesterol level</li>
                                        <li><b>HeartRate:</b> Heart rate in beats per minute</li>
                                        <li><b>HeartDisease (Target):</b> 0 (No heart disease), 1 (Heart disease present)</li>
                                        <li><b>QuantumPatternFeature:</b> A mathematically derived feature incorporating complex, non-linear patterns, potentially aiding in more advanced pattern recognition (a custom-engineered feature)</li>
                                        </ul>
                                        """
                    },
                    {
                        "title":"Water Pollution and Disease",
                        "mode": "monitored",
                        "dataset": "Water Pollution_and_Disease.csv",
                        "difficulty": [('Data cleaning',60), ('Data Translation',80), ('Data Transformation', 70), ('Statistics', 65), ('Feature Selection', 50), ('Model Training', 30)],
                        "target": "Diarrheal Cases per 100,000 people",
                        "description":"""
                                        <b>About the Dataset</b><br><br>
                                            This dataset explores the relationship between water pollution and the prevalence of waterborne diseases worldwide. It includes water quality indicators, pollution levels, disease rates, and socio-economic factors that influence health outcomes. The dataset provides information on different countries and regions, spanning the years <b>2000–2025</b>.<br><br>

                                            It covers key factors such as contaminant levels, access to clean water, bacterial presence, water treatment methods, sanitation coverage, and the incidence of diseases like <b>diarrhea, cholera, and typhoid</b>. Additionally, it incorporates socio-economic variables such as <b>GDP per capita</b>, <b>urbanization rate</b>, and <b>healthcare access</b>, which help assess the broader impact of water pollution on communities.<br><br>

                                            <b>This dataset can be used for:</b>
                                            <ul>
                                            <li>Public health research on the impact of water pollution</li>
                                            <li>Environmental studies to analyze trends in water contamination</li>
                                            <li>Policy-making for clean water access and sanitation improvements</li>
                                            <li>Machine learning models to predict disease outbreaks based on water quality</li>
                                            </ul>

                                            <b>Prevalence:</b>
                                            <ul>
                                            <li>Covers 10 countries (e.g., USA, India, China, Brazil, Nigeria, Bangladesh, Mexico, Indonesia, Pakistan, Ethiopia)</li>
                                            <li>Includes 5 regions per country (e.g., North, South, East, West, Central)</li>
                                            <li>Spans 26 years (2000–2025)</li>
                                            <li>Features 3,000 unique records representing various water sources and pollution conditions</li>
                                            </ul>

                                            <b>Target Variables:</b>
                                            <ul>
                                            <li>Diarrheal Cases per 100,000 people</li>
                                            <li>Cholera Cases per 100,000 people</li>
                                            <li>Typhoid Cases per 100,000 people</li>
                                            </ul>

                                            <b>Primary Target:</b> Diarrheal Cases per 100,000 people<br><br>

                                            <b>Features:</b>
                                            <ul>
                                            <li>Water Treatment Method</li>
                                            <li>Country</li>
                                            <li>Region</li>
                                            <li>Year</li>
                                            <li>Water Source Type</li>
                                            <li>Contaminant Level (ppm)</li>
                                            <li>pH Level</li>
                                            <li>Turbidity (NTU)</li>
                                            <li>Dissolved Oxygen (mg/L)</li>
                                            <li>Nitrate Level (mg/L)</li>
                                            <li>Lead Concentration (µg/L)</li>
                                            <li>Bacteria Count (CFU/mL)</li>
                                            <li>Access to Clean Water (% of Population)</li>
                                            <li>Diarrheal Cases per 100,000 people</li>
                                            <li>Cholera Cases per 100,000 people</li>
                                            <li>Typhoid Cases per 100,000 people</li>
                                            <li>Infant Mortality Rate (per 1,000 live births)</li>
                                            <li>GDP per Capita (USD)</li>
                                            <li>Healthcare Access Index (0–100)</li>
                                            <li>Urbanization Rate (%)</li>
                                            <li>Sanitation Coverage (% of Population)</li>
                                            <li>Rainfall (mm per year)</li>
                                            <li>Temperature (°C)</li>
                                            <li>Population Density (people per km²)</li>
                                            </ul>

                                        """
                    }
                    ]
    
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