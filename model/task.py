
class TaskModel:
  def __init__(self, controller):
    self.controller = controller
    self.current_task = {}
    self.reference_task = {}
    self.current_subtask = 1
    self.current_subsubtask = 1
    
  def update_statusses_and_set_current_tasks(self):
    #set finished subtasks
    for subtask in self.current_task["subtasks"]:
      correct = True
      for subsubtask in subtask["subtasks"]:
        if subsubtask["status"] != "done":
          correct = False
      
      if correct:
        subtask["status"] = "done"

    # set current_subtask
    current_subtask = 1
    for subtask in self.current_task["subtasks"]:
      if subtask["status"] != "done":
        current_subtask = subtask["order"]
        break
    self.current_subtask = current_subtask
    
    # set current_subsubtask
    current_subsubtask = 1
    for subtask in self.current_task["subtasks"]:
      if subtask["order"] == current_subtask:
        #todo handle multiple currnet subtasks
        for subsubtask in subtask["subtasks"]:
          if subsubtask["status"] != "done":
            current_subsubtask = subsubtask["order"]
            break
    self.current_subsubtask = current_subsubtask

    #set ready tasks
    for subtask in self.current_task["subtasks"]:
      if subtask["order"] == current_subtask and subtask["status"] not in ["done", "inprogress"]:
        subtask["status"] = "ready"
        for subsubtask in subtask["subtasks"]:
          if subsubtask["order"] == self.current_subsubtask and subsubtask["status"] not in ["done", "inprogress", "incorrect"]:           
            subsubtask["status"] = "ready"

    if not self.controller.monitored_mode:
      #check if task is finished
      finished = True
      for subtask in self.current_task["subtasks"]:
        if subtask["status"] != "done":
          finished = False
          
      if finished:
        self.controller.finished_task()

  def list_in_lists(self, list, lists):
    for list_to_check in lists:
      if sorted(list,key=str) == sorted(list_to_check,key=str):
        print(sorted(list,key=str))
        print("==")
        print(sorted(list_to_check,key=str))
        return True

      print(sorted(list,key=str))
      print("!=")
      print(sorted(list_to_check,key=str))
    return False

  def get_correct(self, subsubtask):
        correct = True
        for value in subsubtask["value"]:
          #check string
          if isinstance(value, str):
            if value not in subsubtask["applied_values"]:
              correct = False
          #check list
          if isinstance(value, list):
             if not self.list_in_lists(value, subsubtask["applied_values"]):
                correct = False
        return correct

  def get_partially_correct(self, subsubtask):
        partiallycorrect = False
        # check string
        for value in subsubtask["value"]:
          #check string
          if isinstance(value,str):
            if value in subsubtask["applied_values"]:
              partiallycorrect = True
          #check list
          if isinstance(value,list):
             if self.list_in_lists(value, subsubtask["applied_values"]):
                partiallycorrect = True
        return partiallycorrect

  def get_incorrect(self, subsubtask):
        incorrect = False
        for value in subsubtask["applied_values"]:
          #check string
          if isinstance(value,str):
            if value not in subsubtask["value"]:
              print(value)
              print("string not in")
              print(subsubtask["value"])
              incorrect = True
          #check list
          if isinstance(value,list):
             if not self.list_in_lists(value, subsubtask["value"]):
                print(value)
                print("list not in ")
                print(subsubtask["value"])
                incorrect = True
        return incorrect

  def perform_action(self, action, value):
    if action[1] == "ModelPerformance":
            value = value[0]

    #set applied values
    for subtask in self.current_task["subtasks"]:
      if subtask["order"] == self.current_subtask:
        for subsubtask in subtask["subtasks"]:
          if subsubtask["order"] == self.current_subsubtask:
            if action == subsubtask["action"]:
              if value:
                 subsubtask["applied_values"] += [value]
    
    #set done/inprogress
    for subtask in self.current_task["subtasks"]:
      for subsubtask in subtask["subtasks"]:
        incorrect = False
        partiallycorrect = False
        correct = True

        if action[1] == "ModelPerformance":
           #predictive modeling, compare the model to the reference task
           if self.controller.validate_performance(self.current_task):
              score = self.controller.get_predictive_modeling_score(self.current_task)
              #score is higher than 80% of the reference task
              if score > 0.8:
                 correct = True
                 incorrect = False
              else:
                 correct = False
                 incorrect = True
        else:
          correct = self.get_correct(subsubtask)
          partiallycorrect = self.get_partially_correct(subsubtask)
          incorrect = self.get_incorrect(subsubtask)

        if correct:
            subsubtask["status"] = "done"
        elif partiallycorrect:
            subsubtask["status"] = "inprogress"

        if incorrect:
            subsubtask["status"] = "incorrect"
        

  def set_current_task(self,task):
    self.current_task = task

  def set_reference_task(self,task):
     self.reference_task = task

  def get_reference_task(self):
     return self.reference_task
    
  def get_current_task(self):
    return self.current_task
  
  def get_title(self):
    return self.current_task["title"]

  def get_description(self):
    return self.current_task["description"]
  
  def get_difficulty(self):
     return self.current_task["difficulty"]
  
  def get_target(self, task):
    for subtask in task["subtasks"]:
        for subsubtask in subtask["subtasks"]:
            if subsubtask["action"][0] == "DataProcessing" and subsubtask["action"][1]== "AssignTarget":
                return subsubtask["value"][0]
    return None

  def get_dataset(self, task):
    for subtask in task["subtasks"]:
        for subsubtask in subtask["subtasks"]:
            if subsubtask["action"][0] == "DataProcessing" and subsubtask["action"][1]== "AssignTarget":
                return subsubtask["value"][0]
    return None
  
  def get_model_performance(self, task):
        for subtask in task["subtasks"]:
            for subsubtask in subtask["subtasks"]:
                if subsubtask["action"][0] == "ModelDevelopment"and subsubtask["action"][1]=="ModelPerformance":
                    return subsubtask
        return None