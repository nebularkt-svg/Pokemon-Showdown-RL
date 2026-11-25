# DQN Training Archive

This directory contains the saved artifacts for run `DQN-random-max-25_10_24_17-14-46`, seed 10. The configuration uses the Showdown random-battle domain and the maximum-base-power baseline. This is a training run with periodic evaluations. The separate 1,000-battle evaluation in the [project README](../../README.md) comes from the course report.

| Artifact | Contents |
| --- | --- |
| [Training records](data/train.csv) and [training plot](figures/train.png) | 12,237 episodes, with the last logged episode ending at step 411,467. |
| [Evaluation records](data/eval.csv) and [evaluation plot](figures/eval.png) | 15 evaluation checkpoints from 270,000 through 410,000 steps. Each includes 10 battles, with 77 wins across all 150 logged battles. |
| [Algorithm configuration](alg_config.json) | DQN, batch size 32, learning rate 0.00005, discount 0.99, and epsilon 0.8 to 0.05 across a configured 450,000 steps. |
| [Environment](env_config.json) and [training](train_config.json) configurations | Random battles, maximum-base-power opponent, seed 10, and 10,000 steps between evaluations. |
| [Model checkpoints](models/) | Four saved PyTorch weight files in their original directories. The `best` and `final` copies are byte-identical. |
| [Battle traces](replays/410000/) | 54 saved logs under the 410,000-step replay directory. These are the battle-protocol traces extracted from the HTML exports, with chat omitted. |

The saved configuration records the settings attached to this run. For example, its learning rate differs from the initial rate described in the course report. The training CSV is the record of the actual per-episode exploration values. The checkpoint archives were inspected without loading the weights. The replay buffer is outside this archive.

## Environment source

The [environment snapshot](../../environment/showdown_environment.py) implements a 56-feature observation and 14 discrete action IDs. The reward uses damage dealt and taken at weights of 3 and 2, plus trade, knockout, and terminal terms, then divides the result by 3. English comments and docstrings describe the archived implementation.

The [recorded dependencies](../../environment/requirements.txt) come from the experiment folder. The environment also imports `showdown_gym.base_environment`, which belongs to the original training workspace. Running the experiment requires that compatible workspace, the CARES RL runner and a local Pokémon Showdown server. The source was syntax-checked and the artifacts were verified against the local archive.
