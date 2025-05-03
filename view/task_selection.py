from ipywidgets import *

class TaskSelectionView:
    def __init__(self, controller):
        self.controller = controller
        self.task_dropdown = None
        self.select_button = None
        self.vbox = None
        tasks_data = [{
                            "Title": "Titanic survival chance prediction",
                            "Description": "It’s the year 1912. The RMS Titanic has tragically struck an iceberg and begun sinking. As a data scientist in a secret rescue agency, you've been granted access to passenger data from the ship’s manifest. Your mission? Build a model that predicts who had the highest chance of survival.",
                            "SubTasks": [
                            {   "id": 1,
                                "Title": "Data Selection",
                                "status": "todo",
                                "order": 1,
                                "SubTasks": [
                                {
                                    "id": 2,
                                    "Title": "Select dataset titanic.csv",
                                    "Description": "Load the Titanic dataset for model development.",
                                    "Hints": ["Hint1"],
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
                                "Title": "Data Cleaning",
                                "status": "todo",
                                "order": 2,
                                "SubTasks": [
                                {
                                    "id": 4,
                                    "Title": "Replace missing Age with median",
                                    "Description": "Fill in missing 'Age' values using the median age.",
                                    "Hints": ["Use median() function", "Apply to Age column"],
                                    "status": "todo",
                                    "action": ["DataCleaning", "Replace-Median"],
                                    "value": ["Age"],    
                                    "applied_values": [],   
                                    "order": 1
                                },
                                {
                                    "id": 5,
                                    "Title": "Remove rows with missing 'Embarked'",
                                    "Description": "Remove rows where 'Embarked' is missing.",
                                    "Hints": ["", ""],
                                    "status": "todo",
                                    "action": ["DataCleaning", "Remove-Missing"],
                                    "value": ["Embarked"],   
                                    "applied_values": [],                                   
                                    "order": 1
                                },
                                {
                                    "id": 6,
                                    "Title": "Drop unnecessary columns",
                                    "Description": "Drop columns not useful for modeling.",
                                    "Hints": ["Look for ID-like, ticket, or name-based columns"],
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
                                "Title": "Data Translation",
                                "status": "todo",
                                "order": 3,
                                "SubTasks": [
                                {
                                    "id": 8,
                                    "Title": "Encode categorical columns",
                                    "Description": "Convert categorical variables into numeric form.",
                                    "Hints": ["Use LabelEncoder for 'Sex' and 'Embarked'"],
                                    "status": "todo",
                                    "action": ["DataProcessing", "LabelEncoding"],
                                    "value": ["Sex", "Embarked"],
                                    "applied_values": [],  
                                    "order": 1
                                },
                                {
                                    "id": 9,
                                    "Title": "Convert Survived to Boolean",
                                    "Description": "Change the Survived column to boolean values.",
                                    "Hints": ["True if survived, False if not"],
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
                                "Title": "Model Training",
                                "status": "todo",
                                "order": 4,
                                "SubTasks": [
                                {
                                    "id": 11,
                                    "Title": "Assign Target",
                                    "Description": "Assign 'Survived' as the target variable for the model.",
                                    "Hints": ["Separate X and y"],
                                    "status": "todo",
                                    "action": ["DataProcessing", "AssignTarget"],
                                    "value": ["Survived"],
                                    "applied_values": [],  
                                    "order": 1
                                },
                                {
                                    "id": 12,
                                    "Title": "Train/Test Split",
                                    "Description": "Split the dataset into training and testing sets.",
                                    "Hints": ["Use 80/20 split"],
                                    "status": "todo",
                                    "action": ["DataProcessing", "Split"],
                                    "value": ["20%"],
                                    "applied_values": [],  
                                    "order": 2
                                },
                                {
                                    "id": 13,
                                    "Title": "Train Logistic Regression",
                                    "Description": "Train a logistic regression model and evaluate performance.",
                                    "Hints": ["Use LogisticRegression", "Check confusion matrix"],
                                    "status": "todo",
                                    "action": ["ModelDevelopment", "ModelPerformance"],
                                    "value": ["20%"],
                                    "applied_values": [],  
                                    "order": 3
                                }
                                ]
                            }
                            ]
                        }]
        
        self.task_map = {
            task["Title"]: task for task in tasks_data
        }

        self.task_dropdown = widgets.Dropdown(
            options=list(self.task_map.keys()),
            description='Select Task:',
            disabled=False
        )

        self.description_label = widgets.HTML(
            value=self.get_description(self.task_dropdown.value)
        )

        self.task_dropdown.observe(self.update_description, names='value')

        self.select_button = widgets.Button(
            description='Start Task',
            button_style='success'
        )
        self.select_button.on_click(self.start_task)

        self.vbox = widgets.VBox([
            self.task_dropdown,
            self.description_label,
            self.select_button
        ])
        self.vbox.layout.display = 'none'

    def get_description(self, title):
        task = self.task_map.get(title, {})
        return f"<i>{task['Description']}</i>" if task else ""

    def update_description(self, change):
        new_title = change['new']
        self.description_label.value = self.get_description(new_title)

    def start_task(self, event):
        selected_title = self.task_dropdown.value
        selected_task = self.task_map[selected_title]
        self.controller.set_task_model(selected_task)
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

