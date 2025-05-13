from ipywidgets import *

class TaskSelectionView:
    def __init__(self, controller):
        self.controller = controller
        self.recommmended_radio_buttons = None
        self.task_dropdown = None
        self.mode_dropdown = None
        self.select_button = None
        self.vbox = None
        self.vbox2 = None
        self.tab_set = None
        self.alltasks = False
        self.guided_mode = True

        self.tasks_data = [{
                            "title": "Titanic survival chance prediction",
                            "description": "It’s the year 1912. The RMS Titanic has tragically struck an iceberg and begun sinking. As a data scientist in a secret rescue agency, you've been granted access to passenger data from the ship’s manifest. Your mission? Build a model that predicts who had the highest chance of survival.",
                            "dataset": "titanic.csv",
                            "mode": "guided",
                            "subtasks": [
                            {
                                "id": 3,
                                "title": "Data Cleaning",
                                "status": "todo",
                                "order": 1,
                                "subtasks": [
                                {
                                    "id": 4,
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
                                    "id": 5,
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
                                    "id": 6,
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
                                "id": 7,
                                "title": "Data Translation",
                                "status": "todo",
                                "order": 2,
                                "subtasks": [
                                {
                                    "id": 8,
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
                                    "id": 9,
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
                                "id": 10,
                                "title": "Model Training",
                                "status": "todo",
                                "order": 3,
                                "subtasks": [
                                {
                                    "id": 11,
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
                                    "id": 12,
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
                                    "id": 13,
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
                        },{'title': 'todo', 'description': 'todo',
                        "dataset": "titanic.csv","mode": "guided",'subtasks': [
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
                    }]
        
        self.filter_task_selection(None)

        self.task_dropdown = widgets.Dropdown(
            options=list(self.task_map.keys()),
            description='Select Task:',
            disabled=False
        )

        self.mode_dropdown = widgets.Dropdown(
            options=list(["Guided mode","Monitored mode"]),
            description='Select mode:',
            disabled=False
        )

        self.title_label = widgets.HTML(
            value="<h3>" + self.task_dropdown.value+ "</h3>"
        )

        self.description_label = widgets.HTML(
            value=self.get_description(self.task_dropdown.value)
        )
        self.description_label.layout = widgets.Layout(max_width='500px')

        self.task_dropdown.observe(self.update_title_and_description, names='value')
        self.mode_dropdown.observe(self.filter_task_selection, names='value')

        self.recommmended_radio_buttons = widgets.RadioButtons(
            options=[
                'Recommended tasks only', 
                'All tasks'
            ],
            layout={'width': 'max-content'}
        )
        self.recommmended_radio_buttons.observe(self.filter_task_selection, names='value')

        self.select_button = widgets.Button(
            description='Start Task',
            button_style='success'
        )
        self.select_button.on_click(self.start_task)

        self.guided_mode_items = widgets.VBox([
            self.title_label,
            self.description_label
        ])

        learning_path_view = self.controller.get_learning_path_view()

        self.vbox = widgets.VBox([
            self.mode_dropdown, self.recommmended_radio_buttons,self.task_dropdown,self.guided_mode_items, self.select_button
        ])

        self.vbox2 = widgets.VBox([learning_path_view])
        # self.vbox.layout.display = 'none'

        tabs = [self.vbox, self.vbox2]
        tab_set = widgets.Tab(tabs)
        tab_set.set_title(0, 'Task selection')
        tab_set.set_title(1, 'Learning path')
        tab_set.layout.display = 'none'
        self.tab_set = tab_set

    def get_description(self, title):
        task = self.task_map.get(title, {})
        return f"<i>{task['description']}</i>" if task else ""

    def update_title_and_description(self, change):
        new_title = change['new']
        self.title_label.value = "<h3>" + new_title + "</h3>"
        self.description_label.value = self.get_description(new_title)

    def filter_task_selection(self, change):
        if change != None:
            if change["new"] == "Recommended tasks only":
                self.alltasks = False
                
            if change["new"] == "All tasks":
                self.alltasks = True

            if change["new"] == "Guided mode":
                self.guided_mode = True

            if change["new"] == "Monitored mode":
                self.guided_mode = False
        
        if self.guided_mode:
            filtered_tasks = [task for task in self.tasks_data if task["mode"] == "guided"]
        else:
            filtered_tasks = [task for task in self.tasks_data if task["mode"] == "monitored"]

        self.task_map = {
            task["title"]: task for task in filtered_tasks
        }
        
        if self.task_dropdown != None:
            self.task_dropdown.options=list(self.task_map.keys())


    def start_task(self, event):
        if self.mode_dropdown.value == "Monitored mode":
            monitored_mode = True
        elif self.mode_dropdown.value == "Guided mode":
            monitored_mode = False
        selected_title = self.task_dropdown.value
        selected_task = self.task_map[selected_title]
        self.controller.set_task_model(selected_task, monitored_mode)
        if not monitored_mode:
            self.controller.read_dataset_view(selected_task["dataset"])
        self.controller.hide_task_selection_and_show_tabs()

    def get_task_selection_view(self):
        return self.tab_set

    def hide_task_selection(self):
        self.tab_set.layout.display = 'none'
    
    def show_task_selection(self):
        self.tab_set.layout = widgets.Layout(visibility = 'visible')

    def disable_selection(self):
        self.task_dropdown.disabled = True
        self.select_button.disabled = True