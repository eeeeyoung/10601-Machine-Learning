import numpy as np
import sys
import time

num_epoch = int(sys.argv[5])
discount_factor = float(sys.argv[6])

# initialization maze as a list of lists from input

maze_input = open(sys.argv[1], "r")

maze = []
i = 0
start = (0, 0)
goal = (0, 0)

for line in maze_input:
    maze.append([])
    line_ = line.strip()
    for k in range(len(line_)):
        maze[i].append(line_[k])
        # Retrieving start & end state
        if line_[k] == "S":
            start = (i, k)
        elif line_[k] == "G":
            goal = (i, k)

    i += 1

maze_row = len(maze)
maze_column = len(maze[0])
value_file = open(sys.argv[2], "w")
q_value_file = open(sys.argv[3], "w")
policy_file = open(sys.argv[4], "w")


V_dict = np.zeros((len(maze), len(maze[0])))
max_action = np.zeros((len(maze), len(maze[0]), 4))
V_dict_new = np.zeros((len(maze), len(maze[0])))

# 0 = West, 1 = North, 2 = East, 3 = South
Actions = [0, 1, 2, 3]


def check_convergence(old_m, new_m):
    difference = 0
    for i in range(len(old_m)):
        for j in range(len(old_m[0])):
            if np.abs(old_m[i][j]-new_m[i][j]) > difference:
                difference = np.abs(old_m[i][j]-new_m[i][j])
    return difference


def value_iteration():
    global V_dict
    global V_dict_new
    global max_action
    for k in range(num_epoch):
        for i in range(len(maze)):
            for j in range(len(maze[0])):
                if maze[i][j] == "G":
                    V_dict_new[i][j] = 0
                elif maze[i][j] == "*":
                    V_dict_new[i][j] = 0 # and do nothing
                else:
                    for a in range(len(Actions)):
                        if a == 0:
                            i_ = i
                            j_ = j - 1

                            if maze_row > i_ >= 0 and maze_column > j_ >= 0 and maze[i_][j_] != "*":
                                max_action[i][j][a] = -1 + discount_factor * V_dict[i_][j_]
                            else:
                                max_action[i][j][a] = -1 + discount_factor * V_dict[i][j]

                        elif a == 1:
                            i_ = i - 1
                            j_ = j

                            if maze_row > i_ >= 0 and maze_column > j_ >= 0 and maze[i_][j_] != "*":
                                max_action[i][j][a] = -1 + discount_factor *V_dict[i_][j_]
                            else:
                                max_action[i][j][a] = -1 + discount_factor * V_dict[i][j]

                        elif a == 2:
                            i_ = i
                            j_ = j + 1

                            if maze_row > i_ >= 0 and maze_column > j_ >= 0 and maze[i_][j_] != "*":
                                max_action[i][j][a] = -1 + discount_factor*V_dict[i_][j_]
                            else:
                                max_action[i][j][a] = -1 + discount_factor * V_dict[i][j]

                        elif a == 3:
                            i_ = i + 1
                            j_ = j

                            if maze_row > i_ >= 0 and maze_column > j_ >= 0 and maze[i_][j_] != "*":
                                max_action[i][j][a] = -1 + discount_factor*V_dict[i_][j_]
                            else:
                                max_action[i][j][a] = -1 + discount_factor * V_dict[i][j]

                    move = np.argmax(max_action[i][j])
                    V_dict_new[i][j] = max_action[i][j][move]

        diff = check_convergence(V_dict, V_dict_new)
        if diff < 0.001:
            print("converged", k)
            print("V(0,0)", V_dict_new[0][0])
            print("V(7,0)", V_dict_new[7][0])
            print("V(6,7)", V_dict_new[6][7])
            break

        for i in range(len(maze)):
            for j in range(len(maze[0])):
                V_dict[i][j] = V_dict_new[i][j]

    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] != "*":
                value_file.write(str(i) + " " + str(j) + " " + str(V_dict[i][j]) + "\n")
                policy_file.write(str(i) + " " + str(j) + " " + str(np.argmax(max_action[i][j])) + "\n")
                for k in range(4):
                    q_value_file.write(str(i) + " " + str(j) + " " + str(k) + " " + str(max_action[i][j][k]) + "\n")


start = time.time()
value_iteration()
stop = time.time()
print(stop - start)