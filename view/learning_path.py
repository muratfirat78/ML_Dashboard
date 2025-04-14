from ipywidgets import widgets
from ipytree import Tree, Node

class LearningPathView:
    def __init__(self, controller):
        self.controller = controller
        self.tree = Tree(stripes=True)
        node1 = Node('node1')
        node1.icon = 'archive'
        self.tree.add_node(node1)
        self.vbox =  widgets.VBox([self.tree])
        # self.vbox =  widgets.VBox([])

    def get_icon(self, category):
        if category == 'SelectData':
            return 'file'
        if category == 'DataCleaning':
            return 'broom'
        if category == 'DataProcessing':
            return 'repeat'
        if category == 'ModelDevelopment':
            return 'cube'
        return ''

    def update_actions(self):
        actions = self.controller.get_list_of_actions()

        for node in self.tree.nodes:
            self.tree.remove_node(node)

        for action in actions:
            node = Node(str(action[2]) + ': ' + action[1])
            node.icon = self.get_icon(action[0])
            self.tree.add_node(node)


    def get_learning_path_tab(self):
        return self.vbox