from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import List, Dict, DefaultDict
import gymnasium as gym

import numpy as np

class Agent(ABC):
    """Base class for Q-Learning agent

    **ONLY CHANGE THE BODY OF THE act() FUNCTION**

    """

    def __init__(
        self,
        action_space: gym.spaces.Space,
        obs_space: gym.spaces.Space,
        gamma: float,
        epsilon: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of the Q-Learning agent
        namely the epsilon, learning rate and discount rate.

        :param action_space (int): action space of the environment
        :param obs_space (int): observation space of the environment
        :param gamma (float): discount factor (gamma)
        :param epsilon (float): epsilon for epsilon-greedy action selection

        :attr n_acts (int): number of actions
        :attr q_table (DefaultDict): table for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values
        """

        self.action_space = action_space
        self.obs_space = obs_space
        self.n_acts = gym.spaces.utils.flatdim(action_space)

        self.epsilon: float = epsilon
        self.gamma: float = gamma

        self.q_table: DefaultDict = defaultdict(lambda: 0)

    def act(self, obs) -> int:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs:
        :param obs (int): received observation representing the current environmental state
        :return (int): index of selected action
        """
        ### PUT YOUR CODE HERE ###
        best_a = 0  # best action
        best_reward = -1e6  # best reward for current state

        for a in range(self.n_acts):  # find argmax_a (Q(s,a))
            if obs.__class__ == dict:
                obs_n = (tuple(obs['agent']), tuple(obs['target']))
            elif obs.__class__ == int:
                obs_n = obs
            value = self.q_table[(obs_n, a)]
            if value >= best_reward:
                best_reward = value
                best_a = a
        u = np.random.uniform(0, 1)
        if u <= self.epsilon:
            action = self.action_space.sample()
        else:
            action = best_a
        return action

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class QLearningAgent(Agent):
    """
    Agent using the Q-Learning algorithm

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, alpha: float, **kwargs):
        """Constructor of QLearningAgent

        Initializes some variables of the Q-Learning agent, namely the epsilon, discount rate
        and learning rate alpha.

        :param alpha (float): learning rate alpha for Q-learning updates
        """

        super().__init__(**kwargs)
        self.alpha: float = alpha

    def learn(
        self, obs: dict, action: int, reward: float, next_obs: dict, done: bool
    ) -> float:
        """Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (int): received observation representing the current environmental state
        :param action (int): index of applied action
        :param reward (float): received reward
        :param next_obs (tuple): received observation representing the next environmental state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (float): updated Q-value for current observation-action pair
        """
        ### PUT YOUR CODE HERE ###
        best_next_q_value = -1e6
        if obs.__class__ == dict:
            obs_tuple = (tuple(obs['agent']), tuple(obs['target']))
            next_obs_tuple = (tuple(next_obs['agent']), tuple(next_obs['target']))
        elif obs.__class__ == int:
            obs_tuple = obs
            next_obs_tuple = next_obs

        for a in range(self.n_acts):
            if self.q_table[(next_obs_tuple,a)] >= best_next_q_value:
                best_next_q_value = self.q_table[(next_obs_tuple,a)]
        self.q_table[(obs_tuple,action)] += self.alpha*(reward+self.gamma*best_next_q_value-self.q_table[(obs_tuple,action)])
        return self.q_table[(obs_tuple, action)]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q2")


# class MonteCarloAgent(Agent):
#     """
#     Agent using the Monte-Carlo algorithm for training
#
#     **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
#     """
#
#     def __init__(self, **kwargs):
#         """Constructor of MonteCarloAgent
#
#         Initializes some variables of the Monte-Carlo agent, namely epsilon,
#         discount rate and an empty observation-action pair dictionary.
#
#         :attr sa_counts (Dict[(Obs, Act), int]): dictionary to count occurrences observation-action pairs
#         """
#         super().__init__(**kwargs)
#         self.sa_counts = {}
#         self.flagg = []
#
#     def learn(
#         self, obses: List[int], actions: List[int], rewards: List[float]
#     ) -> Dict:
#         """Updates the Q-table based on agent experience
#
#         **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**
#
#         :param obses (List(int)): list of received observations representing environmental states
#             of trajectory (in the order they were encountered)
#         :param actions (List[int]): list of indices of applied actions in trajectory (in the
#             order they were applied)
#         :param rewards (List[float]): list of received rewards during trajectory (in the order
#             they were received)
#         :return (Dict): A dictionary containing the updated Q-value of all the updated state-action pairs
#             indexed by the state action pair.
#         """
#         updated_values = {}
#         ### PUT YOUR CODE HERE ###
#         G = 0
#         for t in reversed(range(len(obses))):
#             G = self.gamma*G+rewards[t]
#             s = obses[t]
#             a = actions[t]
#             previous_sequence = [(obses[i], actions[i]) for i in range(t)]
#             if not (s, a) in self.sa_counts:
#                 self.sa_counts[(s, a)] = 0
#             if not (s, a) in previous_sequence:
#                 reward_before = self.q_table[(s,a)]*self.sa_counts[(s, a)]
#                 self.sa_counts[(s,a)]=self.sa_counts[(s,a)]+1
#                 self.q_table[(s, a)] =(reward_before + G)/self.sa_counts[(s, a)]
#                 updated_values[(s,a)]= self.q_table[(s, a)]
#         #raise NotImplementedError("Needed for Q2")
#         return updated_values
#
#     def schedule_hyperparameters(self, timestep: int, max_timestep: int):
#         """Updates the hyperparameters
#
#         **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**
#
#         This function is called before every episode and allows you to schedule your
#         hyperparameters.
#
#         :param timestep (int): current timestep at the beginning of the episode
#         :param max_timestep (int): maximum timesteps that the training loop will run for
#         """
#         ### PUT YOUR CODE HERE ###
#         #self.gamma = random.uniform(0,1)
#         if timestep <= max_timestep/4:
#             self.epsilon = 0.6   #0.6 9999995
#         elif timestep<= max_timestep*3/5:
#             self.epsilon = max(self.epsilon * 0.9999995, 0.01)
#         else:
#             self.epsilon = 0.01
#
#         #raise NotImplementedError("Needed for Q2")
