
class TaskModel:
  def __init__(self, controller):
    self.controller = controller
    self.current_task = {}
    self.current_subtask = 1
    self.current_subsubtask = 1
    
  def update_task(self, action, value):
      print(action)
      print(value)
      #set ready tasks
      for subtask in self.current_task["SubTasks"]:
        for subsubtask in subtask["SubTasks"]:
          if subsubtask["order"] == self.current_subsubtask:
            subsubtask["status"] = "ready"

      #set applied values
      for subtask in self.current_task["SubTasks"]:
        if subtask["order"] == self.current_subtask:
          print(1)
          for subsubtask in subtask["SubTasks"]:
            if subsubtask["order"] == self.current_subsubtask:
              print(2)
              if action == subsubtask["action"]:
                print("set!")
                if isinstance(value[0], list) and value:
                  subsubtask["applied_values"] += [value[0][0]]
                else:
                  subsubtask["applied_values"] += [value[0]]
      
      #set done/inprogress
      for subtask in self.current_task["SubTasks"]:
        for subsubtask in subtask["SubTasks"]:
          correct = True
          partiallycorrect = False
          for value in subsubtask["value"]:
            print(value + "not in:")
            print(subsubtask["applied_values"])
            if value not in subsubtask["applied_values"]:
              print("Nee!")
              correct = False
            if value in subsubtask["applied_values"]:
              partiallycorrect = True
          if correct:
            print("correct!@")
            subsubtask["status"] = "done"
          else:
            if partiallycorrect:
              subsubtask["status"] = "inprogress"
        
        #set finished subtasks
        for subtask in self.current_task["SubTasks"]:
          correct = True
          for subsubtask in subtask["SubTasks"]:
            if subsubtask["status"] != "done":
              correct = False
          
          if correct:
            subtask["status"] = "done"

        # set current_subtask
        current_subtask = 1
        for subtask in self.current_task["SubTasks"]:
          if subtask["status"] != "done":
            current_subtask = subtask["order"]
            break
        self.current_subtask = current_subtask
        
        # set current_subsubtask
        current_subsubtask = 1
        for subtask in self.current_task["SubTasks"]:
          if subtask["order"] == current_subtask:
            #todo handle multiple currnet subtasks
            for subsubtask in subtask["SubTasks"]:
              if subsubtask["status"] != "done":
                current_subsubtask = subsubtask["order"]
                break
        self.current_subsubtask = current_subsubtask

        print(self.current_task)



  def set_current_task(self,task):
    self.current_task = task
    
  def get_current_task(self):
    return self.current_task
  
  