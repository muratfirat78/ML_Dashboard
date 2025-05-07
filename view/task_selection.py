from ipywidgets import *

class TaskSelectionView:
    def __init__(self, controller):
        self.controller = controller
        self.task_dropdown = None
        self.mode_dropdown = None
        self.select_button = None
        self.vbox = None

        tasks_data = [{
                            "title": "Titanic survival chance prediction",
                            "description": "It’s the year 1912. The RMS Titanic has tragically struck an iceberg and begun sinking. As a data scientist in a secret rescue agency, you've been granted access to passenger data from the ship’s manifest. Your mission? Build a model that predicts who had the highest chance of survival.",
                            "subtasks": [
                            {   "id": 1,
                                "title": "Data Selection",
                                "status": "todo",
                                "order": 1,
                                "subtasks": [
                                {
                                    "id": 2,
                                    "title": "Select dataset titanic.csv",
                                    "description": "Load the Titanic dataset for model development.",
                                    "hints": ["Hint1"],
                                    "status": "todo",
                                    "action": ["SelectData", "DataSet"],
                                    "value": ["titanic.csv"],
                                    "applied_values": [],  
                                    "order": 1
                                }
                                ]
                            },
                            {
                                "id": 3,
                                "title": "Data Cleaning",
                                "status": "todo",
                                "order": 2,
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
                                "order": 3,
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
                                "order": 4,
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
                        },{'title': 'todo', 'description': 'todo', 'subtasks': [{'title': 'Data Selection', 'status': 'todo', 'subtasks': [{'status': 'todo', 'action': ['SelectData', 'DataSet'], 'value': ['titanic.csv'], 'title': 'Load Dataset', 'description': 'Load a dataset.', 'hints': ['Go to the Data Selection tab. Choose the dataset from the list and click read.', 'Select the dataset: titanic.csv.'], 'applied_values': [], 'order': 1}], 'applied_values': [], 'order': 1}, {'title': 'Data Cleaning', 'status': 'todo', 'subtasks': [{'status': 'todo', 'action': ['DataCleaning', 'Drop Column'], 'value': ['Fare', 'Ticket', 'Parch', 'SibSp', 'Name', 'Pclass', 'PassengerId', 'Cabin'], 'title': 'Drop Columns', 'description': 'Remove columns that are irrelevant or redundant for the model.', 'hints': ["Go to the Data Cleaning tab. Select the column(s), choose 'drop column', then click apply.", 'Drop the following column(s): Fare, Ticket, Parch, SibSp, Name, Pclass, PassengerId, Cabin.'], 'applied_values': [], 'order': 1}, {'status': 'todo', 'action': ['DataCleaning', 'Remove-Missing'], 'value': ['Embarked'], 'title': 'Remove Rows with Missing Values', 'description': 'Remove rows that contain missing values in the column.', 'hints': ["Go to the Data Cleaning tab. Select the column, choose 'remove missing', then click apply.", 'Remove rows with missing values in Embarked.'], 'applied_values': [], 'order': 1}, {'status': 'todo', 'action': ['DataCleaning', 'Replace-Median'], 'value': ['Age'], 'title': 'Replace Missing Values (Median)', 'description': 'Replace missing values in a column using the median.', 'hints': ["Go to the Data Cleaning tab. Select the column, choose 'replace median', then click apply.", 'Apply replace median to Age.'], 'applied_values': [], 'order': 1}], 'applied_values': [], 'order': 2}, {'title': 'Data Translation', 'status': 'todo', 'subtasks': [{'status': 'todo', 'action': ['DataProcessing', 'LabelEncoding'], 'value': ['Embarked', 'Sex'], 'title': 'Label Encoding', 'description': 'Convert categorical columns into numeric codes.', 'hints': ["Go to the Data Processing tab. Choose the column, select 'encoding', choose 'Label Encoding', then click apply.", 'Apply label encoding to Embarked, Sex.'], 'applied_values': [], 'order': 1}, {'status': 'todo', 'action': ['DataProcessing', 'ConvertToBoolean'], 'value': ['Survived'], 'title': 'Convert to Boolean', 'description': 'Convert 1 and 0 values in the column to boolean (True/False).', 'hints': ["Go to the Data Processing tab. Choose the column, select 'Convert to Boolean', then click apply.", "Convert 'Survived' to boolean (True/False)."], 'applied_values': [], 'order': 1}], 'applied_values': [], 'order': 3}, {'title': 'Model Training', 'status': 'todo', 'subtasks': [{'status': 'todo', 'action': ['DataProcessing', 'AssignTarget'], 'value': ['Survived'], 'title': 'Assign Target Variable', 'description': 'Assign the column as the target variable for model training.', 'hints': ["Go to the Data Processing tab. Choose the column, select 'Assign Target', then click apply.", "Assign 'Survived' as the target column."], 'applied_values': [], 'order': 1}, {'status': 'todo', 'action': ['DataProcessing', 'Split'], 'value': ['20%'], 'title': 'Train/Test Split', 'description': 'Split dataset into training and testing sets.', 'hints': ['Go to the Data Processing tab. Choose the column, assign the target, set the test ratio, then click split.', 'Use a test ratio of 20% for splitting the dataset.'], 'applied_values': [], 'order': 2}, {'status': 'todo', 'action': ['ModelDevelopment', 'ModelPerformance'], 'value': ['Logistic Regression()'], 'title': 'Model Training & Evaluation', 'description': 'Train a model and evaluate its performance.', 'hints': ['Go to the Predictive Modeling tab. Select the model type, adjust parameters if needed, then click train.', 'Train the model using the selected parameters and evaluate its performance.'], 'applied_values': [], 'order': 3}], 'applied_values': [], 'order': 7}]}]
        self.task_map = {
            task["title"]: task for task in tasks_data
        }

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
        self.mode_dropdown.observe(self.on_mode_change, names='value')

        self.select_button = widgets.Button(
            description='Start Task',
            button_style='success'
        )
        self.select_button.on_click(self.start_task)

        self.guided_mode_items = widgets.VBox([
            self.task_dropdown,
            self.title_label,
            self.description_label
        ])

        self.vbox = widgets.VBox([
            widgets.HBox([self.mode_dropdown,widgets.VBox([self.guided_mode_items, self.select_button])]),

        ])
        self.vbox.layout.display = 'none'

        self.on_mode_change({'new': self.mode_dropdown.value})

    def get_description(self, title):
        task = self.task_map.get(title, {})
        return f"<i>{task['description']}</i>" if task else ""

    def update_title_and_description(self, change):
        new_title = change['new']
        self.title_label.value = "<h3>" + new_title + "</h3>"
        self.description_label.value = self.get_description(new_title)

    def start_task(self, event):
        if self.mode_dropdown.value == "Monitored mode":
            monitored_mode = True
        elif self.mode_dropdown.value == "Guided mode":
            monitored_mode = False
        selected_title = self.task_dropdown.value
        selected_task = self.task_map[selected_title]
        self.controller.set_task_model(selected_task, monitored_mode)
        self.controller.hide_task_selection_and_show_tabs()

    def get_task_selection_view(self):
        return self.vbox

    def hide_task_selection(self):
        self.vbox.layout.display = 'none'
    
    def show_task_selection(self):
        self.vbox.layout = widgets.Layout(visibility = 'visible')

    def disable_selection(self):
        self.task_dropdown.disabled = True
        self.select_button.disabled = True


    def on_mode_change(self, change):
        mode = change['new']
        if mode == "Guided mode":
            self.guided_mode_items.layout.display = 'flex'
            self.select_button.description = 'Start Task'
        else: 
            self.guided_mode_items.layout.display = 'none'
            self.select_button.description = 'Start'