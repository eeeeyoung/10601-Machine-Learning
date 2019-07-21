import numpy as np
import sys


class Environment:

    def __init__(self):
        self.maze = []
        self.start = (0, 0)
        self.goal = (0, 0)
        self.agent_state = (0, 0)
        self.reward = -1
        self.isterminal = 0
        self.maze_row = 0
        self.maze_column = 0
        self.epsilon = 0
        self.next_state = (0, 0)
        self.total_steps = 0

    def constructor(self, path):
        i = 0
        maze_file = open(path, "r")
        for line in maze_file:
            self.maze.append([])
            line_ = line.strip()
            for k in range(len(line_)):
                self.maze[i].append(line_[k])
                # Retrieving start & end state
                if line_[k] == "S":
                    self.start = (i, k)
                    self.agent_state = (i, k)
                elif line_[k] == "G":
                    self.goal = (i, k)
            i += 1
        self.maze_row = len(self.maze)
        self.maze_column = len(self.maze[0])
        self.q_array = np.zeros((self.maze_row, self.maze_column, 4))
        print(self.maze)

    def step(self, a):
        i = self.agent_state[0]
        j = self.agent_state[1]

        if a == 0:
            i_ = i
            j_ = j - 1
            if self.maze_row > i_ >= 0 and self.maze_column > j_ >= 0 and self.maze[i_][j_] != "*":
                self.next_state = (i_, j_)
            else:
                self.next_state = (i, j)

        if a == 1:
            i_ = i - 1
            j_ = j
            if self.maze_row > i_ >= 0 and self.maze_column > j_ >= 0 and self.maze[i_][j_] != "*":
                self.next_state = (i_, j_)
            else:
                self.next_state = (i, j)

        if a == 2:
            i_ = i
            j_ = j + 1
            if self.maze_row > i_ >= 0 and self.maze_column > j_ >= 0 and self.maze[i_][j_] != "*":
                self.next_state = (i_, j_)
            else:
                self.next_state = (i, j)

        if a == 3:
            i_ = i + 1
            j_ = j
            if self.maze_row > i_ >= 0 and self.maze_column > j_ >= 0 and self.maze[i_][j_] != "*":
                self.next_state = (i_, j_)
            else:
                self.next_state = (i, j)

        if self.agent_state == self.goal:
            self.isterminal = 1
            self.reward = 0

        return self.next_state, self.reward, self.isterminal

    def reset(self):
        self.agent_state = self.start
        self.reward = -1
        self.isterminal = 0


def test_function():
    actions = []
    for line in action_seq_file:
        actions = line.strip().split(" ")

    maze = Environment()
    maze.constructor(sys.argv[1])
    for ele in actions:
        state, reward, isterminal = maze.step(int(ele))
        maze.agent_state = maze.next_state
        output_file.write(str(state[0])+" "+str(state[1])+" "+str(reward)+" "+str(isterminal)+"\n")


if __name__ == "__main__":
    maze_input = open(sys.argv[1], "r")

    output_file = open(sys.argv[2], "w")

    action_seq_file = open(sys.argv[3], "r")

    test_function()