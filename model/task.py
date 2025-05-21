
class TaskModel:
  def __init__(self, controller):
    self.controller = controller
    self.current_task = {}
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

  def perform_action(self, action, value):
    if action[1] == "ModelPerformance":
            value = value[0]

    #set applied values
    for subtask in self.current_task["subtasks"]:
      if subtask["order"] == self.current_subtask:
        for subsubtask in subtask["subtasks"]:
          if subsubtask["order"] == self.current_subsubtask:
            if action == subsubtask["action"]:
              if isinstance(value[0], list) and value:
                subsubtask["applied_values"] += [value[0][0]]
              else:
                subsubtask["applied_values"] += [value[0]]
    
    #set done/inprogress
    for subtask in self.current_task["subtasks"]:
      for subsubtask in subtask["subtasks"]:
        correct = True
        partiallycorrect = False
        incorrect = False

        #check correct
        for value in subsubtask["value"]:
          if value not in subsubtask["applied_values"]:
            correct = False
        
        #check partially correct
        for value in subsubtask["value"]:
          if value in subsubtask["applied_values"]:
            partiallycorrect = True

          
        #check incorrect
        for value in subsubtask["applied_values"]:
          if value not in subsubtask["value"]:
            incorrect = True

        if correct:
            subsubtask["status"] = "done"
        elif partiallycorrect:
            subsubtask["status"] = "inprogress"

        if incorrect:
            subsubtask["status"] = "incorrect"

    #check if task is finished
    finished = True
    for subtask in self.current_task["subtasks"]:
      if subtask["status"] != "done":
        finished = False
        
    if finished:
      self.controller.show_completion_popup()
        

  def set_current_task(self,task):
    self.current_task = task
    
  def get_current_task(self):
    return self.current_task
  
  def get_title(self):
    return self.current_task["title"]

  def get_description(self):
    return self.current_task["description"]
  
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