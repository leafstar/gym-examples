import gymnasium as gym
from tqdm import tqdm
from gymnasium.wrappers import FlattenObservation
from constant import CONSTANTS as CONSTANTS
from agents import QLearningAgent
from evaluate import evaluate
from collections import defaultdict
from gym_examples.envs.grid_world import GridWorldEnv

import threading, time

CONFIG = {
    "eval_episodes": 20,
    "eval_freq": 1000,
    "alpha": 0.5,
    "epsilon": 0.3,
    "agent_count": 5,
    "sync_freq": 10
}
CONFIG.update(CONSTANTS)


def q_learning_eval(
        env,
        config,
        q_table,
        render=True,
        output=True):
    """
    Evaluate configuration of Q-learning on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param render (bool): flag whether evaluation runs should be rendered
    :param output (bool): flag whether mean evaluation performance should be printed
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=0.0,
    )
    eval_agent.q_table = q_table
    return evaluate(env, eval_agent, config["eval_eps_max_steps"], config["eval_episodes"], render)




class SharedTrainingData(object):
    agent: QLearningAgent
    env: any
    config: any

    def __init__(self, a: QLearningAgent, e: any, c: any):
        self.agent = a
        self.env = e
        self.config = c



class DoTrain(threading.Thread):
    def __init__(self, shared: SharedTrainingData, *args, **kwargs):
        super(DoTrain,self).__init__(*args, **kwargs)
        self.shared = shared

        # single train step
    def trainStep(self, agent: QLearningAgent, env, config):
        obs, info = env.reset()
        episodic_return = 0
        t = 0
        max_steps = config["total_eps"] * config["eps_max_steps"]

        while t < config["eps_max_steps"]:
            agent.schedule_hyperparameters(agent.step, max_steps)
            act = agent.act(obs)
            next_obs, reward, done, _, info = env.step(act)
            agent.learn(obs, act, reward, next_obs, done)

            t += 1
            # step_counter += 1
            agent.step += 1         # TODO: Warning!: Check thios
            episodic_return += reward

            if done:
                break

            obs = next_obs
        # TODO: reset agent ???
        # total_reward += episodic_return

    def run(self):
        self.trainStep(self.shared.agent, self.shared.env, self.shared.config)


def train(config, output=True):
    """
    Train and evaluate Q-Learning on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param output (bool): flag if mean evaluation results should be printed
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        total reward over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table
    """

    agentEnvList = []
    for _ in range(1, config["agent_count"] + 1):
        env = gym.make('gym_examples/FrozenLake-v2', render_mode='human')
        agent = QLearningAgent(
            action_space=env.action_space,
            obs_space=env.observation_space,
            gamma=config["gamma"],
            alpha=config["alpha"],
            epsilon=config["epsilon"],
        )
        agentEnvList.append((agent, env))

    'imaginary evaluate env'
    imaginaryEnv = gym.make('gym_examples/FrozenLake-v2', render_mode='human')
    # step_counter = 0

    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []

    global_q_table = {}

    for eps_num in tqdm(range(1, config["total_eps"] + 1)):
        threads = []
        for (agent,env) in agentEnvList:
            tempThread = DoTrain(shared=SharedTrainingData(agent, env, config))
            threads.append(tempThread)
            tempThread.start()
            # trainStep(agent, env, config)

        for t in threads:
            t.join()
            
        if eps_num % config["sync_freq"] == 0:
            global_q_table = defaultdict(lambda: 0)
            for (agent, env) in agentEnvList:
                for key, value in agent.q_table.items():
                    global_q_table[key] += value       # add

            for key ,value in global_q_table.items():
                global_q_table[key] /= config["agent_count"]   # average


            '''open if need to validate q_table average logic'''
            if eps_num > 0 and eps_num % config["eval_freq"] == 0:
                validateKey = (0,0)
                sumedQ: int = 0
                for (agent, env) in agentEnvList:
                    sumedQ += agent.q_table[validateKey]
                   
                isCorrect = sumedQ / config["agent_count"] == global_q_table[validateKey]
                tqdm.write(f"QTable Average: new qt {global_q_table[validateKey]} {sumedQ} {isCorrect}")

            # update q_table
            for (agent, env) in agentEnvList:
                agent.q_table = global_q_table

        
        # if eps_num > 0 and eps_num % config["eval_freq"] == 0:
        #     mean_return, negative_returns = q_learning_eval(imaginaryEnv, config, global_q_table)
        #     tqdm.write(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return}")
        #     evaluation_return_means.append(mean_return)
        #     evaluation_negative_returns.append(negative_returns)

        # total_reward += episodic_return

        # TODO: @muxing figure out when to eval
        # if eps_num > 0 and eps_num % config["eval_freq"] == 0:
        #     mean_return, negative_returns = q_learning_eval(env, config, agent.q_table)
        #     tqdm.write(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return}")
        #     evaluation_return_means.append(mean_return)
        #     evaluation_negative_returns.append(negative_returns)
    # return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table
    mean_return, negative_returns = q_learning_eval(imaginaryEnv, config, global_q_table)
    tqdm.write(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return}")
    evaluation_return_means.append(mean_return)
    evaluation_negative_returns.append(negative_returns)

    return global_q_table
    


if __name__ == "__main__":
    from gymnasium.envs.registration import register

    register(
        id="gym_examples/FrozenLake-v2",
        entry_point="gym_examples.envs:FrozenLakeEnv",
        max_episode_steps=300,
    )

    # env = gym.make(CONFIG["env"])
    
    # wrapped_env = FlattenObservation(env)
    # total_reward, _, _, q_table = train(CONFIG)
    q_table = train(CONFIG)
    # print()
    # print(f"Total reward over training: {total_reward}\n")
    print("Q-table:")
    print(q_table)
