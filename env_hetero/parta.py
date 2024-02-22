import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from env_hetero import utils


def chi(t, E):
    return int(np.floor(t / E) * E)


# class BlockType(Enum):
#     FLOOR = 1
#     WALL = 2
#     GOAL = 3
def generate_maze(rows, cols, start_x=0, start_y=0):
    maze = [['W' for _ in range(cols)] for _ in range(rows)]

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols and maze[x][y] == 'W'

    def dfs(x, y):
        maze[x][y] = 'R'

        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if is_valid(nx, ny):
                maze[nx][ny] = 'R'
                maze[(x + nx) // 2][(y + ny) // 2] = 'R'
                dfs(nx, ny)

    start_x, start_y = 0, 0
    dfs(start_x, start_y)
    maze[rows - 1][cols - 2] = 'R'
    maze[rows - 2][cols - 1] = 'R'
    return maze


def print_maze(maze):
    for row in maze:
        print(' '.join(row))


class SimpleFrozenLake:
    def defineGrid(self, size):
        maze = generate_maze(size, size)
        # print_maze(maze)
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
    def initializeP(self):
        P = np.zeros((self.size ** 2 * 4, self.size ** 2))
        R = np.zeros((self.size ** 2 * 4, 1))
        for StateAction in range(self.size ** 2 * 4):
            iState = int(np.floor(StateAction / 4))
            iAction = int(StateAction % 4)
            iPos = (int(np.floor(iState / self.size)), int(iState % self.size))
            if self.grid[iPos[0]][iPos[1]] == "W":
                continue
            iNextPos = self.calcNextState(iPos, iAction)
            iNextState = iNextPos[0] * self.size + iNextPos[1]
            if iNextPos[0] == self.size - 1 and iNextPos[1] == self.size - 1:
                if iPos != iNextPos:
                    R[StateAction][0] = 1
            if iState == self.size ** 2 - 1:  # current position is goal
                P[int(StateAction)][int(iState)] = 1  # can only stay at the destination
            else:
                P[int(StateAction)][int(iNextState)] = 1

        return P, R

    def __init__(self, size):
        self.size = size
        self.grid = self.defineGrid(size)
        P, R = self.initializeP()
        self.P = P
        self.R = R
        # print(self.P)


class Agent:
    def __init__(self, lam, Q0, P, reward_matrix, gamma=0.99, E=1, T=30000):
        self.name = "RLagent"
        self.lam = lam
        self.Q = Q0
        self.P = P
        self.reward_matrix = reward_matrix
        self.gamma = gamma
        self.steps = 0
        self.E = E
        self.T = T

    def reset(self):
        self.steps = 0
        self.lam = 0.9

    def getV(self):
        V = np.zeros((int(self.Q.shape[0] / 4), 1))
        counter = 0
        for i in range(0, self.Q.shape[0], 4):
            maxQ = -1
            for j in range(4):
                if self.Q[i + j] > maxQ:
                    maxQ = self.Q[i + j]
            V[counter] = maxQ
            counter += 1

        return V

    def learn(self, forstar):
        Delta_tk = []
        Qks = []
        Vks = []
        tol = 10e-5
        max_step = 10000
        Qold = self.Q
        self.updateQ(forstar)
        while np.linalg.norm(self.Q - Qold, np.inf) > tol and self.steps <= max_step:
            Delta_tk.append(np.linalg.norm(self.Q - Qold, np.inf))
            Qks.append(self.Q)
            Vks.append(self.getV())
            Qold = self.Q
            self.updateQ(forstar)
        return Delta_tk, Qks, Vks

    def updateQ(self, forstar):
        # apply Bellman's Operator and soft update
        self.steps += 1
        self.schedule_hyperparameters(self.steps, forstar)
        # print(f"Q = {self.Q}, lam ={self.lam}")
        # print(self.lam)
        self.Q = (1 - self.lam) * self.Q + self.lam * (self.reward_matrix + self.gamma * np.matmul(self.P, self.getV()))
        # return self.Q

    def schedule_hyperparameters(self, k, forstar):
        if forstar:
            self.lam = 0.99
        else:
            # self.lam = 1 / np.sqrt((k+1))
            # self.lam = 1 / np.sqrt(self.T)
            # self.lam = 1 / (k + 1)

            self.lam = 0.99
            if self.steps>150:
                self.lam = 1/np.sqrt(self.steps)
            # self.lam = self.E/(k+1)
            # self.lam = 1/(k + self.E)
            # self.lam = np.log(self.T + 1)**2 / (self.T + 1)
            # self.lam = min(1, np.log(k + 1) / (k + 1) / np.sqrt((1 - self.gamma)))
            # self.lam = np.log(k + 1) / (k + 1)
            # self.lam = np.log(k + 1) **2/ (k + 1)
            # self.lam = 1 / self.T


def train(agents, imagin_env, Qstar, kappa, max_time_step, E=1):
    K = len(agents)
    t = 0
    maxT = max_time_step
    total_comm_num = 0
    Deltas = []
    Delta_sync_rounds = []
    local_Delta_tk = []  # Delta_t^k
    local_stepwise = []  # V_{j+1}-V_j
    Qtbar_list = []
    lam_list = []
    Vik = []  # a collection of Vik
    # initialize Q_0^k
    for ag in agents:
        ag.Q = np.zeros_like(ag.Q)
        ag.reset()
    # append the quantities at time 0
    Q_t_bar = np.mean([ag.Q for ag in agents], axis=0)
    Delta_0 = np.linalg.norm(Qstar - Q_t_bar, np.inf)
    Deltas.append(Delta_0)
    Delta_sync_rounds.append(np.linalg.norm(Qstar - Q_t_bar, np.inf))
    local_Delta_tk.append(np.linalg.norm(Qstar - agents[0].Q, np.inf))
    local_stepwise.append(0)
    # quantities that need to be tracked during the training.
    # in other words, these quantities need to be compared with themselves.
    Vk_old = [ag.getV() for ag in agents]
    Vik.append(Vk_old)
    Q_t_bar_old = Q_t_bar
    while True:
        Qtbar_list.append(Q_t_bar)
        t = t + 1
        for ag in agents:  # all agents local update
            ag.updateQ(forstar=False)  # Q_t^k
        lam_list.append(agents[0].lam)
        Q_t_bar = np.mean([agent.Q for agent in agents], axis=0)  # Q_t bar
        Delta_t_norm = np.linalg.norm(Qstar - Q_t_bar, np.inf)
        Deltas.append(Delta_t_norm)
        local_Delta_tk.append(np.mean([np.linalg.norm(Qstar - agents[i].Q, np.inf) for i in range(K)]))
        local_stepwise.append(np.mean([np.linalg.norm(agents[i].getV() - Vk_old[i], np.inf) for i in range(K)]))
        if t>150:
            E = 10
        if t % E == 0:  # synchronization step
            total_comm_num += 1
            Delta_sync_rounds.append(Delta_t_norm)
            for agent in agents:
                agent.Q = Q_t_bar
        # print(t, Delta_t_norm)
        if Delta_t_norm <= 10e-5 or np.linalg.norm(Q_t_bar_old - Q_t_bar, np.inf) <= 10e-10 or t > maxT:
            break
        Q_t_bar_old = Q_t_bar
        Vk_old = [agents[i].getV() for i in range(K)]
        Vik.append(Vk_old)
    return Deltas, t, total_comm_num, Qtbar_list, Delta_sync_rounds, local_Delta_tk, local_stepwise, lam_list, Vik


def analytical(t, delta0, gamma):
    """
    Analytical solution for exponential decay for this recurrence relation.
    Delta_{t+1} <= (1-(1-gamma)*lam_t)*Delta_{t}
    Args:
        t:
        delta0: Initial difference
        gamma: discount factor
    Returns:
        a list: [Delta_{0}, Delta_{1},..., Delta_{t}]
    """

    def lam(i):
        # return 1/np.sqrt((i+1))
        return np.log(i + 1) ** 2 / (i + 1)
        # return 0.1

    # print(t,np.prod(1-np.array([lam(i) for i in range(t)])*(1-gamma)),delta0)
    return np.prod(1 - np.array([lam(i) for i in range(t)]) * (1 - gamma)) * delta0
    # return 32/3/(1-gamma)**2*(np.exp(-0.05*np.sqrt(lam(i)*30000)))


if __name__ == '__main__':
    mpl.use('macosx')
    random.seed(2)
    K = 5
    maxT = 50000
    E =  maxT+1  # maxT+1  # 1  # np.floor(np.log(maxT)) #
    Pk_list = []  # a list of all local transition prob matrix
    Qkstar_list = []  # a list of all local optimal Q matrix
    Vkstar_list = []  # a list of all local optimal V matrix
    local_Qt_k = []
    local_Vt_k = []
    grid_size = 5
    num_s = grid_size ** 2
    a_dim = 4  # U D L R
    agent_list = []
    R = np.ones((num_s * a_dim, 1))
    for k in range(K):
        Env = SimpleFrozenLake(grid_size)  # initialize the envs
        print(f"maze {k}")
        print_maze(Env.grid)
        Pk = Env.P  # get the transition matrix of env k
        Pk_list.append(Pk)  # add the transition matrix of env k to a list
        # R = Env.R  this line is a simple reward function. only two entries are one.
        agentk = Agent(lam=1, Q0=np.zeros((num_s * a_dim, 1)), P=Pk, reward_matrix=R)
        agent_list.append(agentk)
        Delta_tk, local_Qt, local_Vt = agentk.learn(forstar=True)  # Delta_tk is the local one-step error
        local_Qt_k.append(local_Qt)
        local_Vt_k.append(local_Vt)
        Qkstar_list.append(agentk.Q)
        Vkstar_list.append(agentk.getV())
        # plt.plot(range(len(Delta_tk)), Delta_tk, linestyle='-', label=f"local decay {k}, diminish in {agentk.steps} steps")
    # uncomment if you need comparison with some standard decay rate.
    # plt.plot(range(1000),[np.exp(-i) for i in range(1000)], color="orange", linestyle ='-',label="standard expo_decay")
    # plt.plot(range(1000), [1/(i+1) for i in range(1000)], color="green", linestyle='-', label="1/t")
    # plt.plot(range(1000), [(np.log(i+1)) / (i + 1) for i in range(1000)], color="purple", linestyle='-', label="logt/t")
    # plt.legend()
    # plt.show()

    print("global\n\n\n")
    Pbar = np.mean(Pk_list, axis=0)

    agent_global = Agent(lam=1, Q0=np.zeros((num_s * a_dim, 1)), P=Pbar, reward_matrix=R)
    global_optimal, Qt_list, Vt_list = agent_global.learn(forstar=True)
    Qstar = agent_global.Q
    Vstar = agent_global.getV()

    """
    Animation of agent 1 approaches to the optimal Q.
    Plot of Qstar(Img Env), Qkstar(Local) and average of Qkstar.
    Result is the averaged local optimal Q is far away from the optimal Q in the imaginary env.
    """
    utils.plotQseq(local_Qt_k[0], Qstar, Qkstar_list[0])
    plt.figure()
    plt.ylabel("the difference to optimal Q")
    plt.plot(range(len(Qt_list)), [np.linalg.norm(Qstar - Qt_list[i], np.inf) for i in range(len(Qt_list))],label=f"agent img converges with t={len(Qt_list)}")
    for i in range(K):
        plt.plot(range(len(local_Qt_k[i])),
             [np.linalg.norm(Qkstar_list[i] - local_Qt_k[i][j], np.inf) for j in range(len(local_Qt_k[i]))],label=f"agent {i} converges with t={len(local_Qt_k[i])}")
    plt.legend()

    plot_local_Qs_and_Qstar = False
    if plot_local_Qs_and_Qstar:
        plt.figure()
        colors = plt.cm.viridis(np.linspace(0, 1, K+2))
        plt.plot(Qstar, 'o', color = colors[0], label="Qstar")
        plt.plot(np.mean(Qkstar_list, axis=0), 'o', color="red", label="Qkstar avg")
        for i, Qkstar in enumerate(Qkstar_list):
            plt.plot(Qkstar, 'o', color=colors[i+1], alpha=0.2, label=f"Q{i}_star")
        plt.legend()
        plt.savefig("Qstars.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    """
    Above is the fixed-points analysis, and from now on is the iterative analysis.
    """
    # real_rate = [np.linalg.norm(Qstar - Q, np.inf) for Q in Qt_list] # this is for local

    # plt.plot(range(len(global_optimal)), global_optimal, linestyle='-', label=f"ima_env's optimal behaviour, diminish in {agent_global.steps} steps")

    # plt.plot(range(len(Qt_list)), real_rate, color ="blue", label = "real diff")
    if False:
        for agent in agent_list:
            """
            force kappa to be 0
            """
            agent.P = Pbar
    kappa = 0
    for agent in agent_list:
        agent.E = E
        agent.T = maxT
        if np.linalg.norm(Pbar - agent.P, np.inf) > kappa:
            kappa = np.linalg.norm(Pbar - Pk, np.inf)
    print(f"kappa = {kappa}")

    Deltas, T, comm_num, Qtbar_list, Deltas_sync, local_Delta_tk, local_stepwise, lams, Vik = train(agent_list, Pbar,
                                                                                               Qstar=Qstar,
                                                                                               kappa=kappa,
                                                                                               max_time_step=maxT,
                                                                                               E=E)
    print("computed\n")
    # utils.plotQseq(Qtbar_list, Qstar, None, Deltas)
    plt.figure()
    # analytical_rate = [analytical(i, np.linalg.norm(Qstar, np.inf), gamma=agent_global.gamma) for i in
    #                    range(min(maxT, T))]
    plt.title(r"$\lambda_t = \frac{log(t)}{t}$")
    # plt.plot(range(min(maxT, T)), analytical_rate, color='red', label="analytical")
    plt.scatter(range(len(Deltas)), Deltas, s=2, color="black", linestyle="dotted",
                label=f"fed decay to 0 {T}, err={Deltas[-1]}")
    plt.scatter(range(len(local_Delta_tk)), local_Delta_tk, s=1, alpha=0.8, color="red", linestyle="dotted",
                label=f"local_Delta_tk")
    plt.scatter(range(len(local_stepwise)), local_stepwise, s=1, alpha=0.8, color="blue", linestyle="dotted",
                label=r"max norm of $V_j-V_{j+1}$")
    #plt.scatter(range(0, T, int(E)), Deltas_sync, s=1, color="orange", label=f"sync_err")
    plt.ylim((0, max(Deltas)))
    plt.legend()

    """
    Diagnostic section!!!!!!!
    """
    e22 = []
    e22_b = []
    e22_wo_p = []
    e22_norm_inside = []
    for i in range(T):
        e22_bs = np.linalg.norm(
            np.mean([np.matmul(Pk_list[k], (np.array(Vik[chi(i, E)][k]) - np.array(Vik[i][k]))) for k in range(K)],
                    axis=0),
            np.Inf)
        e22i = np.linalg.norm(np.mean([np.matmul(Pk_list[k], (Vstar - np.array(Vik[i][k]))) for k in range(K)], axis=0),
                              np.Inf)
        e22_norm_inside.append(np.mean([np.linalg.norm(np.matmul(Pk_list[k], (Vstar - np.array(Vik[i][k]))),np.Inf) for k in range(K)], axis=0)
                              )
        e22_wo_p.append(
            np.mean([np.linalg.norm((np.array(Vik[chi(i, E)][k]) - np.array(Vik[i][k])), np.Inf) for k in range(K)],
                    axis=0)
            )
        e22_b.append(e22_bs)
        e22.append(e22i)
    plt.figure()
    plt.title("diagnostics")
    #plt.scatter(range(len(Deltas)), Deltas, s=2, color="black", label=r"$\Delta_{t+1}$")
    #plt.scatter(range(len(lams)), lams, s=2, color="blue", label=r"$\lambda_t$")
    #plt.scatter(range(1, len(Deltas)), (1 - np.array(lams)) * np.array(Deltas[:len(Deltas) - 1]), s=2, color="orange",
    #            label=r"$(1-\lambda_t)\Delta_{t}$")
    plt.scatter(range(len(Deltas)), Deltas, s=12, color="black", linestyle="dotted",
                label=f"fed decay to 0 {T}, err={Deltas[-1]}")
    #plt.scatter(range(0, T, int(E)), Deltas_sync, s=10,alpha=0.8, color="orange", label=r"sync_err $\|\Delta_{\chi(i)}\|$")
    plt.scatter(range(len(e22)), e22, s=2, color="violet", label = r"$\|\frac{1}{K}\sum_{i=1}^K{P^k(V^*-V_i^k)}\|$") # norm(average (Vstar - Vtk))
    plt.scatter(range(len(e22_wo_p)), e22_wo_p, s=10,alpha=0.8, color="purple",
                label=r"$\frac{1}{K}\sum_{i=1}^K\|{(V_{\chi(i)}-V_i^k)}\|$")  # norm(average (Vchi(t) - Vtk))
    plt.scatter(range(len(e22_b)), e22_b, s=6, alpha=0.8, color="yellow", label=r"$\|\frac{1}{K}\sum_{i=1}^K{P^k(V_{\chi(i)}-V_i^k)}\|$") # norm(average (Vchi(t) - Vtk))
    plt.scatter(range(len(e22_norm_inside)), e22_norm_inside, s=1, alpha=0.8, color="crimson", linestyle="dotted",
                label=r"$\frac{1}{K}\sum_{i=1}^K\|{P^k(V^*-V_i^k)}\|$")

    plt.scatter(range(len(local_stepwise)), local_stepwise, s=1, alpha=0.8, color="blue", linestyle="dotted",
                label=r"max norm of $V_j-V_{j+1}$")
    plt.legend()
    print(Deltas[-1])
   # print(utils.lambda_analysis(lams, 5000))
    plt.show()

    #### check V diff
    plt.figure()
    t_v =9
    plt.title(f"time {t_v}")
    plt.ylim([-3,np.max(Vik)+5])
    colors = ["red","black","orange","violet","blue"]
    for i in range(K):
        plt.scatter(range(25),Vik[t_v][i],s = (K-i)*20, color = colors[i],alpha = 1, label = f"V^{i}")
    plt.scatter(range(25),Vstar, s = 100,marker=',',alpha = 0.5, label = f"Vstar")
    plt.legend()
    d = 1
