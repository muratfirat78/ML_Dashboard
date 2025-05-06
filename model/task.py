
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
              if subsubtask["order"] == self.current_subsubtask and subsubtask["status"] not in ["done", "inprogress"]:           
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
        for value in subsubtask["value"]:
          if value not in subsubtask["applied_values"]:
            correct = False
          if value in subsubtask["applied_values"]:
            partiallycorrect = True
        if correct:
          subsubtask["status"] = "done"
        else:
          if partiallycorrect:
            subsubtask["status"] = "inprogress"

  def set_current_task(self,task):
    self.current_task = task
    
  def get_current_task(self):
    return self.current_task
  
  