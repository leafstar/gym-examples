import gymnasium as gym
from tqdm import tqdm
from gymnasium.wrappers import FlattenObservation
from constant import CONSTANTS as CONSTANTS
from agents import QLearningAgent
from evaluate import evaluate
from gym_examples.envs.grid_world import GridWorldEnv
CONFIG = {
    "eval_episodes": 500,
    "eval_freq": 1000,
    "alpha": 0.5,
    "epsilon": 0.0,
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


def train(env, config, output=True):
    """
    Train and evaluate Q-Learning on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param output (bool): flag if mean evaluation results should be printed
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        total reward over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table
    """
    agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=config["epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []

    for eps_num in tqdm(range(1, config["total_eps"]+1)):
        obs, info = env.reset()
        episodic_return = 0
        t = 0

        while t < config["eps_max_steps"]:
            agent.schedule_hyperparameters(step_counter, max_steps)
            act = agent.act(obs)
            next_obs, reward, done, _, info = env.step(act)
            agent.learn(obs, act, reward, next_obs, done)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs = next_obs

        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, negative_returns = q_learning_eval(env, config, agent.q_table)
            tqdm.write(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return}")
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)
    return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table


if __name__ == "__main__":

    from gymnasium.envs.registration import register

    register(
        id="gym_examples/GridWorld-v0",
        entry_point="gym_examples.envs:GridWorldEnv",
        max_episode_steps=300,
    )

    env = gym.make(CONFIG["env"])
    env = gym.make('gym_examples/GridWorld-v0')
    wrapped_env = FlattenObservation(env)
    total_reward, _, _, q_table = train(env, CONFIG)
    # print()
    # print(f"Total reward over training: {total_reward}\n")
    print("Q-table:")
    print(q_table)
