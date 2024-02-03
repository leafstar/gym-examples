import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import random

class Agent:
    def __init__(self, alpha, epsilon):
        self.name = "RLagent"
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = {1: 0, 2: 0}

    def learn(self, action, reward):
        self.Q[action] = self.Q[action] + self.alpha * (reward - self.Q[action])
        # return self.Q

    def act(self):
        u = np.random.uniform(0, 1)
        if u > self.epsilon:  # optimal
            best_a = max(self.Q, key=self.Q.get)
        else:  # exploration
            best_a = np.random.randint(1, 3)
        return best_a

    def schedule_hyperparameters(self, k):
        # self.alpha = 1 / (1 + np.log(1 + k))
        # self.alpha = 1
        #self.alpha = 0.9**k
        self.alpha = 1/k


# class BlockType(Enum):
#     FLOOR = 1
#     WALL = 2
#     GOAL = 3
def generate_maze(rows, cols, start_x = 0, start_y = 0):
    maze = [['W' for _ in range(cols)] for _ in range(rows)]

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols and maze[x][y] == 'W'

    def dfs(x, y):
        maze[x][y] = ' '

        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if is_valid(nx, ny):
                maze[nx][ny] = ' '
                maze[(x + nx) // 2][(y + ny) // 2] = ' '
                dfs(nx, ny)

    start_x, start_y = 0, 0
    dfs(start_x, start_y)

    return maze

def print_maze(maze):
    for row in maze:
        print(' '.join(row))  

class SimpleFrozenLake:
    def defineGrid(self, size):
        maze = generate_maze(size, size)
        print_maze(maze)
        return maze
    
    def calcNextState(self, origin, action):
        tempOrigin = [origin[0], origin[1]]
        if action == 0 or action == 1:
            # wall
            # no wall surrounding case
            nextStep = 1
            if action == 1:
                nextStep = -1
            
            tempOrigin[0] -= nextStep   
           
        if action == 2 or action == 3:
            # wall
            # no wall surrounding case
            nextStep = 1
            if action == 3:
                nextStep = -1
            tempOrigin[1] -= nextStep

        if (tempOrigin[0] < 0 or tempOrigin[0] > self.size - 1 or tempOrigin[1] < 0 or tempOrigin[1] > self.size - 1):
                # out of bound
                return origin    
        destinationType = self.grid[tempOrigin[0]][tempOrigin[1]]
        if destinationType == 'W':
            # wall
            return origin
            
        # valid next State
        return tempOrigin
        
    # 0 = 0, U   %4 == 0
    # 1 = 0, D   %4 == 1
    # 2 = 0, L   %4 == 2
    # 3 = 0, R   %4 == 3

    # State: Math.floor(i/4) => (x,y) = (Math.floor(state/4), state%4)
    def initiallizeP(self):
        P = np.zeros((self.size**2*4, self.size**2))
        for StateAction in range(self.size**2*4):
            iState = int(np.floor(StateAction/4))
            iAction = int( StateAction % 4)
            iPos = (int(np.floor(iState/self.size)), int(iState%self.size))
            
            iNextPos = self.calcNextState(iPos, iAction)
            iNextState = iNextPos[0]*self.size + iNextPos[1]

            P[int(StateAction)][int(iNextState)] = 1

        return P

    def __init__(self, size):
        self.size = size
        self.grid = self.defineGrid(size)
        self.P = self.initiallizeP()
        print(self.P)
        # self.p = 
        # self.mu1 = mu1
        # self.sig1 = sig1
        # self.mu21 = mu21
        # self.mu22 = mu22
        # self.sig21 = sig21
        # self.sig22 = sig22

    # def sample(self, which_arm):
    #     if which_arm == 1:  # action
    #         reward = np.random.normal(self.mu1, self.sig1)
    #     else:
    #         uu = np.random.uniform(0, 1, 1)
    #         if uu > 0.5:
    #             reward = np.random.normal(self.mu21, self.sig21)
    #         else:
    #             reward = np.random.normal(self.mu22, self.sig22)
    #     return reward


def train(agent, bandit):
    t = 1000
    accumulated_rewards = [0]
    for cur_t in range(1, t + 1):
        agent.schedule_hyperparameters(cur_t)
        action = agent.act()
        reward = bandit.sample(action)
        acc_avg_reward = accumulated_rewards[cur_t - 1] / cur_t * (cur_t - 1) + reward / cur_t
        accumulated_rewards.append(acc_avg_reward)
        # print(f"{cur_t}, action is {action}, reward is {reward}, acc_avg_reward is {acc_avg_reward}\n")
        agent.learn(action, reward)
    return accumulated_rewards, agent.Q


if __name__ == '__main__':
    my_bandit = SimpleFrozenLake(5)

    eps_list = [0, 0.1, 0.2, 0.5]
    np.random.seed(1)
    for eps in eps_list:
        my_agent = Agent(alpha=1, epsilon=eps)
        acc_k_i = np.zeros((100, 1001))
        for i in range(100):
            acc_i, Q = train(my_agent, my_bandit)
            acc_k_i[i] = acc_i
        plt.plot(range(0, 1001), np.mean(acc_k_i, axis=0), label=f"$\\epsilon$={eps}")
        plt.xlabel("time")
        plt.ylabel("Averaged Accumulative Reward")
        plt.title("Learning rate $\\alpha$ is 1/k")
        print(f"$\\epsilon$={eps}, Qtable = {my_agent.Q}")
    plt.legend()
    plt.show()
