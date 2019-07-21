import numpy as np
import sys
import time

from environment import Environment

num_episodes = int(sys.argv[5])
max_episode_length = int(sys.argv[6])
learning_rate = float(sys.argv[7])
discount_factor = float(sys.argv[8])
epsilon = float(sys.argv[9])


def training():
    Actions = [0, 1, 2, 3]
    maze = Environment()
    maze.constructor(sys.argv[1])

    for ep in range(num_episodes):
        maze.reset()
        for ep_len in range(max_episode_length):
            exploit_vs_explore = np.random.choice(np.arange(0, 2), p=[1-epsilon, epsilon])
            if exploit_vs_explore == 0:
                # EXPLOIT

                action = np.argmax(maze.q_array[maze.agent_state[0]][maze.agent_state[1]])

                if action == 0:
                    i_ = maze.agent_state[0]
                    j_ = maze.agent_state[1] - 1
                    if maze.maze_row > i_ >= 0 and maze.maze_column > j_ >= 0 and maze.maze[i_][j_] != "*":
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (-1 + discount_factor * np.max(maze.q_array[i_][j_]))
                        maze.agent_state = (i_, j_)
                    else:
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (-1 + discount_factor * np.max(maze.q_array[maze.agent_state[0]][maze.agent_state[1]]))

                if action == 1:
                    i_ = maze.agent_state[0] - 1
                    j_ = maze.agent_state[1]
                    if maze.maze_row > i_ >= 0 and maze.maze_column > j_ >= 0 and maze.maze[i_][j_] != "*":
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (-1 + discount_factor * np.max(maze.q_array[i_][j_]))
                        maze.agent_state = (i_, j_)
                    else:
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (
                                    -1 + discount_factor * np.max(maze.q_array[maze.agent_state[0]][maze.agent_state[1]]))

                if action == 2:
                    i_ = maze.agent_state[0]
                    j_ = maze.agent_state[1] + 1
                    if maze.maze_row > i_ >= 0 and maze.maze_column > j_ >= 0 and maze.maze[i_][j_] != "*":
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (-1 + discount_factor * np.max(maze.q_array[i_][j_]))
                        maze.agent_state = (i_, j_)
                    else:
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (
                                    -1 + discount_factor * np.max(maze.q_array[maze.agent_state[0]][maze.agent_state[1]]))

                if action == 3:
                    i_ = maze.agent_state[0] + 1
                    j_ = maze.agent_state[1]
                    if maze.maze_row > i_ >= 0 and maze.maze_column > j_ >= 0 and maze.maze[i_][j_] != "*":
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (-1 + discount_factor * np.max(maze.q_array[i_][j_]))
                        maze.agent_state = (i_, j_)
                    else:
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (
                                    -1 + discount_factor * np.max(maze.q_array[maze.agent_state[0]][maze.agent_state[1]]))

            if exploit_vs_explore == 1:
                action = np.random.choice(Actions)

                if action == 0:
                    i_ = maze.agent_state[0]
                    j_ = maze.agent_state[1] - 1
                    if maze.maze_row > i_ >= 0 and maze.maze_column > j_ >= 0 and maze.maze[i_][j_] != "*":
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (-1 + discount_factor * np.max(maze.q_array[i_][j_]))
                        maze.agent_state = (i_, j_)
                    else:
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (-1 + discount_factor * np.max(maze.q_array[maze.agent_state[0]][maze.agent_state[1]]))

                if action == 1:
                    i_ = maze.agent_state[0] - 1
                    j_ = maze.agent_state[1]
                    if maze.maze_row > i_ >= 0 and maze.maze_column > j_ >= 0 and maze.maze[i_][j_] != "*":
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (-1 + discount_factor * np.max(maze.q_array[i_][j_]))
                        maze.agent_state = (i_, j_)
                    else:
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (
                                    -1 + discount_factor * np.max(maze.q_array[maze.agent_state[0]][maze.agent_state[1]]))

                if action == 2:
                    i_ = maze.agent_state[0]
                    j_ = maze.agent_state[1] + 1
                    if maze.maze_row > i_ >= 0 and maze.maze_column > j_ >= 0 and maze.maze[i_][j_] != "*":
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (-1 + discount_factor * np.max(maze.q_array[i_][j_]))
                        maze.agent_state = (i_, j_)
                    else:
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (
                                    -1 + discount_factor * np.max(maze.q_array[maze.agent_state[0]][maze.agent_state[1]]))

                if action == 3:
                    i_ = maze.agent_state[0] + 1
                    j_ = maze.agent_state[1]
                    if maze.maze_row > i_ >= 0 and maze.maze_column > j_ >= 0 and maze.maze[i_][j_] != "*":
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (-1 + discount_factor * np.max(maze.q_array[i_][j_]))
                        maze.agent_state = (i_, j_)
                    else:
                        maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] = (1 - learning_rate) * maze.q_array[maze.agent_state[0]][maze.agent_state[1]][action] + learning_rate * (
                                    -1 + discount_factor * np.max(maze.q_array[maze.agent_state[0]][maze.agent_state[1]]))

            if maze.agent_state == maze.goal:
                maze.total_steps += ep_len
                break
            elif ep_len == max_episode_length -1:
                maze.total_steps += ep_len

    value_file = open(sys.argv[2], "w")
    q_value_file = open(sys.argv[3], "w")
    policy_file = open(sys.argv[4], "w")
    for i in range(maze.maze_row):
        for j in range(maze.maze_column):
            if maze.maze[i][j] != "*":
                value_file.write(str(i) + " " + str(j) + " " + str(np.max(maze.q_array[i][j])) + "\n")
                policy_file.write(str(i) + " " + str(j) + " " + str(np.argmax(maze.q_array[i][j])) + "\n")
                for k in range(len(Actions)):
                    q_value_file.write(str(i) + " " + str(j) + " " + str(k) + " " + str(maze.q_array[i][j][k]) + "\n")
    print("average steps", maze.total_steps/2000)
    print("V(0, 0)", np.max(maze.q_array[0][2]))
    print("V(2, 4)", np.max(maze.q_array[2][4]))
    print("V(2, 0)", np.max(maze.q_array[2][0]))
    print("V(7, 0)", np.max(maze.q_array[7][0]))
    print("V(6, 7)", np.max(maze.q_array[6][7]))


start = time.time()
training()
stop = time.time()

print(stop-start)