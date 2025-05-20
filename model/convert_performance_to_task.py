class ConvertPerformanceToTask:
    def get_action_category(self,action):
        if action in ["Replace-Median", "Replace-Mean", "Replace-Mode", "Remove-Missing", "Drop Column", "Edit Range"]:
            return "Data Cleaning"
        
        if action in ["LabelEncoding", "OneHotEncoding", "ConvertToBoolean"]:
            return "Data Translation"
        
        if action in ["Normalize", "Standardize", "outlier", "Unbalancedness Upsample", "Unbalancedness DownSample"]:
            return "Data Transformation"
        
        if action in ["AssignTarget", "Split", "ModelPerformance"]:
            return "Model Training"
    
    def get_subtask(self, subtasks, action):
        for subtask in subtasks:
            if subtask["title"] == self.get_action_category(action):
                return subtask
        return None
    
    def get_subsubtask(self, subsubtasks, action):
        for subsubtask in subsubtasks:
            if subsubtask["action"][0] == action[0] and subsubtask["action"][1] == action[1]:
                return subsubtask
        return None

    def get_title_subsubtask(self, subsubtask):
        action = subsubtask["action"][1]

        title_map = {
            "Edit Range": "Edit Range",
            "Drop Column": "Drop Columns",
            "Replace-Mean": "Replace Missing Values (Mean)",
            "Replace-Median": "Replace Missing Values (Median)",
            "Remove-Missing": "Remove Rows with Missing Values",
            "Replace-Mode": "Replace Missing Values (Mode)",
            "PCA": "Dimensionality Reduction (PCA)",
            "outlier": "Handle Outliers",
            "AssignTarget": "Assign Target Variable",
            "ConvertToBoolean": "Convert to Boolean",
            "Standardize": "Standardize Columns",
            "Normalize": "Normalize Columns",
            "Unbalancedness Upsample": "Upsample Minority Class",
            "Unbalancedness DownSample": "Downsample Majority Class",
            "Split": "Train/Test Split",
            "LabelEncoding": "Label Encoding",
            "OneHotEncoding": "One-Hot Encoding",
            "DataSet": "Load Dataset",
            "ModelPerformance": "Model Training & Evaluation"
        }

        return title_map.get(action, "Unknown Task")        

    def get_description_subsubtask(self, subsubtask):
        action = subsubtask["action"][1]
        if len(subsubtask["value"]) > 1:
            columns = "columns"
            verb = "Edit the ranges of"
            drop = "Remove columns that are irrelevant or redundant for the model."
            replace_mean = "Replace missing values in columns using the mean."
            replace_median = "Replace missing values in columns using the median."
            remove_missing = "Remove rows that contain missing values in columns."
            replace_mode = "Replace missing values in columns using the most frequent value (mode)."
            outlier_desc = "Detect and handle outliers in columns."
            assign_target = "Assign columns as the target variables for model training."
            convert_bool = "Convert 1 and 0 values in columns to boolean (True/False)."
            standardize = "Standardize columns to have zero mean and unit variance."
            normalize = "Normalize columns to scale values between 0 and 1."
            label_encode = "Convert categorical columns into numeric codes."
            onehot_encode = "Convert categorical columns into one-hot encoded vectors."
        else:
            columns = "column"
            verb = "Edit the range of"
            drop = "Remove a column that is irrelevant or redundant for the model."
            replace_mean = "Replace missing values in a column using the mean."
            replace_median = "Replace missing values in a column using the median."
            remove_missing = "Remove rows that contain missing values in the column."
            replace_mode = "Replace missing values in a column using the most frequent value (mode)."
            outlier_desc = "Detect and handle outliers in the column."
            assign_target = "Assign the column as the target variable for model training."
            convert_bool = "Convert 1 and 0 values in the column to boolean (True/False)."
            standardize = "Standardize the column to have zero mean and unit variance."
            normalize = "Normalize the column to scale values between 0 and 1."
            label_encode = "Convert the categorical column into numeric codes."
            onehot_encode = "Convert the categorical column into a one-hot encoded vector."

        if action == "Edit Range":
            return verb + f" {columns}."
        if action == "Drop Column":
            return drop
        if action == "Replace-Mean":
            return replace_mean
        if action == "Replace-Median":
            return replace_median
        if action == "Remove-Missing":
            return remove_missing
        if action == "Replace-Mode":
            return replace_mode
        if action == "PCA":
            return "Apply Principal Component Analysis to reduce dimensionality."
        if action == "outlier":
            return outlier_desc
        if action == "AssignTarget":
            return assign_target
        if action == "ConvertToBoolean":
            return convert_bool
        if action == "Standardize":
            return standardize
        if action == "Normalize":
            return normalize
        if action == "Unbalancedness Upsample":
            return "Handle class imbalance by upsampling the minority class."
        if action == "Unbalancedness DownSample":
            return "Handle class imbalance by downsampling the majority class."
        if action == "Split":
            return "Split dataset into training and testing sets."
        if action == "LabelEncoding":
            return label_encode
        if action == "OneHotEncoding":
            return onehot_encode
        if action == "DataSet":
            return "Load a dataset."
        if action == "ModelPerformance":
            return "Train a model and evaluate its performance."

    def order_matters_subtask(self, subtask):
        if subtask["title"] == "Data Selection":
            return True
        elif subtask["title"] == "Data Cleaning":
            return False
        elif subtask["title"] == "Data Translation":
            return False
        elif subtask["title"] == "Data Transformation":
            return False
        elif subtask["title"] == "Statistics":
            return False
        elif subtask["title"] == "Feature Selection":
            return False
        elif subtask["title"] == "Model Training":
            return True
        
    def get_order_subtask(self, subtask):
        if subtask["title"] == "Data Selection":
            return 1
        elif subtask["title"] == "Data Cleaning":
            return 2
        elif subtask["title"] == "Data Translation":
            return 3
        elif subtask["title"] == "Data Transformation":
            return 4
        elif subtask["title"] == "Statistics":
            return 5
        elif subtask["title"] == "Feature Selection":
            return 6
        elif subtask["title"] == "Model Training":
            return 7
    
    def get_hints_subsubtask(self, subsubtask):
        hints = []

        actions = subsubtask.get("action", [])
        values = subsubtask.get("value", [])

        action_hint_map = {
            "Edit Range": "Go to the Data Cleaning tab. Select a column, choose 'edit range', set the desired min and max values, then click apply.",
            "Drop Column": "Go to the Data Cleaning tab. Select the column(s), choose 'drop column', then click apply.",
            "Replace-Mean": "Go to the Data Cleaning tab. Select the column, choose 'replace mean', then click apply.",
            "Replace-Median": "Go to the Data Cleaning tab. Select the column, choose 'replace median', then click apply.",
            "Remove-Missing": "Go to the Data Cleaning tab. Select the column, choose 'remove missing', then click apply.",
            "Replace-Mode": "Go to the Data Cleaning tab. Select the column, choose 'replace mode', then click apply.",
            "PCA": "Go to the Data Processing tab. Choose the column(s), select 'feature extraction', choose 'PCA', then click 'Add PCA' and apply.",
            "outlier": "Go to the Data Processing tab. Select the column, choose 'outlier', then click apply.",
            "AssignTarget": "Go to the Data Processing tab. Choose the column, select 'Assign Target', then click apply.",
            "ConvertToBoolean": "Go to the Data Processing tab. Choose the column, select 'Convert to Boolean', then click apply.",
            "Standardize": "Go to the Data Processing tab. Choose the column, select 'scaling' as process type, then choose 'standardize' as method, and click apply.",
            "Normalize": "Go to the Data Processing tab. Choose the column, select 'scaling' as process type, then choose 'normalize' as method, and click apply.",
            "Unbalancedness Upsample": "Go to the Data Processing tab. Choose the column, select 'imbalanceness', choose 'upsample', then click apply.",
            "Unbalancedness DownSample": "Go to the Data Processing tab. Choose the column, select 'imbalanceness', choose 'downsample', then click apply.",
            "Split": "Go to the Data Processing tab. Choose the column, assign the target, set the test ratio, then click split.",
            "LabelEncoding": "Go to the Data Processing tab. Choose the column, select 'encoding', choose 'Label Encoding', then click apply.",
            "OneHotEncoding": "Go to the Data Processing tab. Choose the column, select 'encoding', choose 'One Hot Encoding', then click apply.",
            "DataSet": "Go to the Data Selection tab. Choose the dataset from the list and click read.",
            "ModelPerformance": "Go to the Predictive Modeling tab. Select the model type, adjust parameters if needed, then click train."
        }

        action_description_map = {
            "Edit Range": "Edit the numerical range of selected columns.",
            "Drop Column": "Drop the following column(s): {}.",
            "Replace-Mean": "Apply replace mean to {}.",
            "Replace-Median": "Apply replace median to {}.",
            "Remove-Missing": "Remove rows with missing values in {}.",
            "Replace-Mode": "Apply replace mode to {}.",
            "PCA": "Apply PCA to the following column(s): {}.",
            "outlier": "Handle outliers in {}.",
            "AssignTarget": "Assign '{}' as the target column.",
            "ConvertToBoolean": "Convert '{}' to boolean (True/False).",
            "Standardize": "Standardize the following column(s): {}.",
            "Normalize": "Normalize the following column(s): {}.",
            "Unbalancedness Upsample": "Apply upsampling to handle class imbalance in {}.",
            "Unbalancedness DownSample": "Apply downsampling to handle class imbalance in {}.",
            "Split": "Use a test ratio of {} for splitting the dataset.",
            "LabelEncoding": "Apply label encoding to {}.",
            "OneHotEncoding": "Apply one-hot encoding to {}.",
            "DataSet": "Select the dataset: {}.",
            "ModelPerformance": "Train the model using the selected parameters and evaluate its performance."
        }

        for action in actions:
            if action in action_hint_map:
                hints.append(action_hint_map[action])
                break
        
        for action in actions:
            if action in action_description_map:
                val = ', '.join(values)
                desc = action_description_map[action]
                hints.append(desc.format(val))
                break

        return hints

    def convert_performance_to_task(self, performance, title, description):
        subtasks = []
        task = {}
        actions = []
        dataset = ""

        for category, action_dict in performance.items():
            for action_type, value in action_dict.items():
                values = value if isinstance(value, list) else [value]
                for v in values:
                    val, idx = v
                    action = [category, action_type]
                    actions.append((action, val, idx))
        actions.sort(key=lambda x: x[2])

        for action in actions:
            action_str = ""
            if isinstance(action[1], list):
                action_str = action[1][0]
            if isinstance(action[1], str):
                action_str = action[1]
            if isinstance(action[1], tuple):
                action_str = action[1][0]

            if action[0][1] == "DataSet":
                dataset = action_str
                continue
            

            subtask = self.get_subtask(subtasks, action[0][1])

            if subtask is None:
                #subtask does not exist yet, create subtask
                subtask = {}
                subtask["title"] = self.get_action_category(action[0][1])
                subtask["status"] = "todo"
                subtask["subtasks"] = []
                subtask["applied_values"] = []
                subtasks.append(subtask)
            subsubtask = self.get_subsubtask(subtask["subtasks"], action[0])

            if subsubtask is None:
                #subsubtask does not exist yet, create subsubtask
                subsubtask = {}
                subsubtask["status"] = "todo"
                subsubtask["action"] = action[0]
                subsubtask["value"] = []
                subtask["subtasks"].append(subsubtask)

            if isinstance(action_str, list):
                subsubtask["value"].append(action_str[0])
            elif isinstance(action_str, str):
                subsubtask["value"].append(action_str)


        #subtasks created, now set order, hints, descriptions
        for subtask in subtasks:
            subtask["order"] = self.get_order_subtask(subtask)
            order_matters = self.order_matters_subtask(subtask)
            for i,subsubtask in enumerate(subtask["subtasks"]):
                subsubtask["title"] = self.get_title_subsubtask(subsubtask)
                subsubtask["description"] = self.get_description_subsubtask(subsubtask)
                subsubtask["hints"] = self.get_hints_subsubtask(subsubtask)
                subsubtask["applied_values"] = []
                if order_matters:
                    subsubtask["order"] = i+1
                else:
                    subsubtask["order"] = 1

        task["title"] = title
        task["description"] = description
        task["dataset"] = dataset
        task["subtasks"] = subtasks
        return task
