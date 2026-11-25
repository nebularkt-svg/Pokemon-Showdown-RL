# Reinforcement Learning for Pokémon Showdown

A COMPSYS 726 Deep Q-Network (DQN) project for Generation 9 random singles. The agent uses type matchup features and shaped rewards to choose moves, switches, and Terastallization actions. The course report records **600 wins in 1,000 battles** against a maximum-base-power baseline.

| Start here | Contents |
| --- | --- |
| [Environment source](environment/showdown_environment.py) | Archived observation, action, and reward implementation |
| [Seed-10 training archive](experiments/dqn-random-max-seed-10/README.md) | Configurations, training and evaluation logs, plots, weights, and replay traces |
| [Reported evaluation](#training-and-evaluation) | Independent 1,000-battle result and type-feature ablation |
| [Rule-based companion project](https://github.com/nebularkt-svg/Pokemon-Showdown-Rule-Based-Agent) | Expert-system agent and its Gen 9 Ubers evaluation |

## Environment and algorithm

At each turn, the agent builds an observation from visible battle information and chooses to use a move, switch Pokémon, or Terastallize while using a move. The opponent's unrevealed team members and moves make the battle partially observable.

The project considered Proximal Policy Optimization (PPO) and chose DQN as a starting point for the discrete action space. The experiment logs also contain Double DQN and Rainbow runs. The evaluation below covers the DQN agent.

## Observation design

| Iteration | Features | Reported behavior |
| --- | --- | --- |
| Type matchup baseline | 12 team HP values, four type values for the two active Pokémon, and two threat scores | The agent sometimes switched away from dangerous matchups, but it also made unfavorable switches and missed knockout opportunities. |
| Expanded observations | A 104-dimensional vector with weather, stat stages, and other battle context, alongside larger rewards | Win rates occasionally exceeded 50%, with no sustained upward trend. |
| Simplified observations | Type matchups, HP, Speed differences, move effectiveness, and damage-related features | Win rate stabilized near 50% around 100,000 steps. A later training run reached 60–70%. |

The type chart covers 18 types. A Water-type move against a pure Fire-type opponent yields a 2× multiplier, which can be normalized as an input feature. Threat scores describe the active matchup. Move-specific features encode how effective each candidate move is against the current opponent.

## Actions and rewards

| Agent action IDs | Action | Environment action IDs |
| --- | --- | --- |
| 0–5 | Switch to one of six team slots | 0–5 |
| 6–9 | Use one of four move slots | 6–9 |
| 10–13 | Terastallize and use the corresponding move | 22–25 |

The action mapping matches the environment draft in `simple.txt`. The report describes rewards for damage dealt and taken, knockouts, and battle outcomes. It also records experiments with bonuses for type-effective moves, knockouts with limited damage taken, and winning with surviving teammates. The report gives a 3:2 weight ratio for damage dealt versus damage taken and a reward range of `[-30, 30]`. The draft uses different coefficients, so the exact reward configuration for the reported run remains unverified.

## Training and evaluation

The report records a batch size of 32 and an initial learning rate of 0.001, reduced to 0.0005 once ε fell below 0.3. Exploration was restarted twice by resetting ε to 0.2 and decaying it to 0.05 over 100,000 steps. The first restart occurred at 200,000 steps. The training section places the second at 350,000 steps, while the discussion places it at 300,000.

| Evaluation | Reported result |
| --- | --- |
| Maximum-damage baseline, 10 rounds of 100 battles | Round win rates of 55%, 62%, 63%, 58%, 60%, 61%, 63%, 59%, 58%, and 61% |
| Win-rate summary | Mean of 60.0% and sample standard deviation of 2.54 percentage points across rounds |
| Type feature ablation | Plotted rewards stabilized near +15.0 with type features and −5.4 with those features removed. Both agents used the same network architecture. |

The ablation supports the use of type features in this experiment. The 104-dimensional trial changed the observations and reward scale together, so its results do not isolate the effect of either change. The simplified observation can omit weather and terrain effects that matter in individual battles.

## Implementation and reproducibility

The environment draft uses Python, NumPy, `poke_env`, and a Showdown environment wrapper. The experiment log records a local Showdown server and CARES reinforcement learning entry points for training, resuming, plotting, and evaluation. Reproducing the reported run requires the upstream code revisions, the corresponding reward implementation, training seeds, and model checkpoints.

This summary draws on the course report (`report.doc`), the environment draft (`simple.txt`), and the experiment log (`LOG.txt`). The training and ablation results come from the report. The mean win rate and sample standard deviation were recalculated from the ten round results.
