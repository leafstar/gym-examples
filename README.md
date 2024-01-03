Mux: 
---(Update Jan 3 Wed)---

I just realized that we were using asynchronous Q learning setting, which is more sophosticated to analyze. Therefore, we may need to create a sampler that allows us to do synchronous update. I will demonstrate the concepts this weekend.

---END---

Install the newest Anaconda,https://www.anaconda.com/download, and you will see a default 'base' environment. Do not use it. Instead, create a new one and install everything inside this venv. If you are using pycharm, you can set the project interpreter to be the virtual conda environment you just created. 

To create a virtual environment, you can do that with GUI or, in terminal, `conda create any_env_name`. Then, `conda activate any_env_name` to activate the env. You will see something like `(frlresearch) muxingwang@Muxings-MacBook-Air gym-examples %`  in your terminal. 

For all packages required, use pycharm's default installation first. If that doesn't work, just google "conda install pkg_name" and follow the first instruction. E.g. see this link to see an installation-command: 'https://anaconda.org/conda-forge/box2d-py'

Once you've installed all the packages, you can run 'script.py' to see if it works or not. (If you are using pycharm, just right-click the file and click run)

END OF MUX's comments.

TODO: We need to extend singleton to distrbuted learning, which means we have to setup K agents with K different environments. We have to determine the code structure, like where should we initiate the agents, how to sync, blabla.

END TODO

This repository is no longer maintained, as Gym is not longer maintained and all future maintenance of it will occur in the replacing [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) library. You can contribute Gymnasium examples to the Gymnasium repository and docs directly if you would like to. If you'd like to learn more about the transition from Gym to Gymnasium, you can read more about it [here](https://farama.org/Announcing-The-Farama-Foundation).

# Gym Examples
Some simple examples of Gym environments and wrappers.
For some explanations of these examples, see the [Gym documentation](https://gymnasium.farama.org).

### Environments
This repository hosts the examples that are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/environment_creation/).
- `GridWorldEnv`: Simplistic implementation of gridworld environment

### Wrappers
This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).
- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment

### Contributing
If you would like to contribute, follow these steps:
- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

PRs may require accompanying PRs in [the documentation repo](https://github.com/Farama-Foundation/Gymnasium/tree/main/docs).
