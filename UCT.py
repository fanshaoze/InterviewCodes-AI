import math
import numpy as np

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        action = [a for a in self.state.get_legal_actions() if a not in [child.state.last_action for child in self.children]][0]
        next_state = self.state.move(action)
        child_node = Node(next_state, self)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.value += result

    def fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

class UCT:
    def __init__(self, root):
        self.root = root

    def search(self, n_simulations):
        for _ in range(n_simulations):
            node = self.tree_policy(self.root)
            reward = self.default_policy(node.state)
            self.backup(node, reward)
        return self.root.best_child(c_param=0)

    def tree_policy(self, node):
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child()
        return node

    def default_policy(self, state):
        while not state.is_terminal():
            state = state.random_move()
        return state.result()

    def backup(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

# Example game state class
class GameState:
    def __init__(self):
        pass

    def get_legal_actions(self):
        # Return a list of legal actions
        pass

    def move(self, action):
        # Return the new game state after applying the action
        pass

    def is_terminal(self):
        # Return whether the game is over
        pass

    def random_move(self):
        # Apply a random move and return the new state
        pass

    def result(self):
        # Return the result of the game
        pass

# Example usage
initial_state = GameState()
root = Node(initial_state)
uct = UCT(root)
best_action = uct.search(1000)
print(best_action.state)
