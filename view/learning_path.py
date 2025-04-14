from ipywidgets import widgets
# from ipytree import Tree, Node

class LearningPathView:
    def __init__(self, controller):
        self.controller = controller
        # self.tree = Tree(stripes=True)
        # node1 = Node('node1')
        # node1.icon = 'archive'
        # self.tree.add_node(node1)
        # self.vbox =  widgets.VBox([self.tree])
        self.vbox =  widgets.VBox([])


    def get_learning_path_tab(self):
        return self.vbox