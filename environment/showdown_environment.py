import os
import time
from typing import Any, Dict

import numpy as np
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import ObsType
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv

class ShowdownEnvironment(BaseShowdownEnv):

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        self.rl_agent = account_name_one

        # Pokémon type IDs.
        self.type_chart = {
            "normal": 0, "fire": 1, "water": 2, "electric": 3, "grass": 4,
            "ice": 5, "fighting": 6, "poison": 7, "ground": 8, "flying": 9,
            "psychic": 10, "bug": 11, "rock": 12, "ghost": 13, "dragon": 14,
            "dark": 15, "steel": 16, "fairy": 17
        }

        # Type matchup table.
        self._build_type_chart()

    def _build_type_chart(self):
        """Build the type matchup table."""
        self.effectiveness = {}

        # Super effective (2.0)
        super_effective = [
            ("fire", "grass"), ("fire", "ice"), ("fire", "bug"), ("fire", "steel"),
            ("water", "fire"), ("water", "ground"), ("water", "rock"),
            ("grass", "water"), ("grass", "ground"), ("grass", "rock"),
            ("electric", "water"), ("electric", "flying"),
            ("ice", "grass"), ("ice", "ground"), ("ice", "flying"), ("ice", "dragon"),
            ("fighting", "normal"), ("fighting", "ice"), ("fighting", "rock"), ("fighting", "dark"), ("fighting", "steel"),
            ("poison", "grass"), ("poison", "fairy"),
            ("ground", "fire"), ("ground", "electric"), ("ground", "poison"), ("ground", "rock"), ("ground", "steel"),
            ("flying", "grass"), ("flying", "fighting"), ("flying", "bug"),
            ("psychic", "fighting"), ("psychic", "poison"),
            ("bug", "grass"), ("bug", "psychic"), ("bug", "dark"),
            ("rock", "fire"), ("rock", "ice"), ("rock", "flying"), ("rock", "bug"),
            ("ghost", "psychic"), ("ghost", "ghost"),
            ("dragon", "dragon"),
            ("dark", "psychic"), ("dark", "ghost"),
            ("steel", "ice"), ("steel", "rock"), ("steel", "fairy"),
            ("fairy", "fighting"), ("fairy", "dragon"), ("fairy", "dark"),
        ]

        # Not very effective (0.5)
        not_very_effective = [
            ("fire", "fire"), ("fire", "water"), ("fire", "rock"), ("fire", "dragon"),
            ("water", "water"), ("water", "grass"), ("water", "dragon"),
            ("grass", "fire"), ("grass", "grass"), ("grass", "poison"), ("grass", "flying"), ("grass", "bug"), ("grass", "dragon"), ("grass", "steel"),
            ("electric", "electric"), ("electric", "grass"), ("electric", "dragon"),
            ("ice", "fire"), ("ice", "water"), ("ice", "ice"), ("ice", "steel"),
            ("fighting", "poison"), ("fighting", "flying"), ("fighting", "psychic"), ("fighting", "bug"), ("fighting", "fairy"),
            ("poison", "poison"), ("poison", "ground"), ("poison", "rock"), ("poison", "ghost"),
            ("ground", "grass"), ("ground", "bug"),
            ("flying", "electric"), ("flying", "rock"), ("flying", "steel"),
            ("psychic", "psychic"), ("psychic", "steel"),
            ("bug", "fire"), ("bug", "fighting"), ("bug", "poison"), ("bug", "flying"), ("bug", "ghost"), ("bug", "steel"), ("bug", "fairy"),
            ("rock", "fighting"), ("rock", "ground"), ("rock", "steel"),
            ("ghost", "dark"),
            ("dragon", "steel"),
            ("dark", "fighting"), ("dark", "dark"), ("dark", "fairy"),
            ("steel", "fire"), ("steel", "water"), ("steel", "electric"), ("steel", "steel"),
            ("fairy", "fire"), ("fairy", "poison"), ("fairy", "steel"),
        ]

        # Immune (0.0)
        no_effect = [
            ("normal", "ghost"), ("electric", "ground"), ("fighting", "ghost"),
            ("poison", "steel"), ("ground", "flying"), ("psychic", "dark"),
            ("ghost", "normal"), ("dragon", "fairy"),
        ]

        for atk, dfd in super_effective:
            self.effectiveness[(atk, dfd)] = 2.0
        for atk, dfd in not_very_effective:
            self.effectiveness[(atk, dfd)] = 0.5
        for atk, dfd in no_effect:
            self.effectiveness[(atk, dfd)] = 0.0

    def _get_action_size(self) -> int | None:

        return 14

    def process_action(self, action: np.int64) -> np.int64:
        """Map compact action IDs to environment actions."""
        if 0 <= action <= 5:
            return action
        elif 6 <= action <= 9:
            return action
        elif 10 <= action <= 13:
            return 22 + (action - 10)
        else:
            raise ValueError(f"Invalid action: {action}")

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()
        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
            info[agent]["fainted_opponent"] = sum(1 for mon in self.battle1.opponent_team.values() if mon.fainted)
            info[agent]["fainted_own"] = sum(1 for mon in self.battle1.team.values() if mon.fainted)
            info[agent]["hp_advantage"] = self._calculate_hp_advantage(self.battle1)
        return info

    def _calculate_hp_advantage(self, battle: AbstractBattle) -> float:
        own_total_hp = sum(mon.current_hp_fraction for mon in battle.team.values())
        opp_total_hp = sum(mon.current_hp_fraction for mon in battle.opponent_team.values())
        return own_total_hp - opp_total_hp

    def calc_reward(self, battle: AbstractBattle) -> float:
        """Calculate the scaled reward from battle events."""
        prior_battle = self._get_prior_battle(battle)
        reward = 0.0

        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [mon.current_hp_fraction for mon in battle.opponent_team.values()]
        if len(health_opponent) < 6:
            health_opponent.extend([1.0] * (6 - len(health_opponent)))

        if prior_battle is not None:
            prior_health_team = [mon.current_hp_fraction for mon in prior_battle.team.values()]
            prior_health_opponent = [mon.current_hp_fraction for mon in prior_battle.opponent_team.values()]
            if len(prior_health_opponent) < 6:
                prior_health_opponent.extend([1.0] * (6 - len(prior_health_opponent)))

            damage_dealt = np.sum(np.array(prior_health_opponent) - np.array(health_opponent))
            damage_taken = np.sum(np.array(prior_health_team) - np.array(health_team))

            reward += 3 * damage_dealt
            reward -= 2 * damage_taken

            if damage_taken > 0:
                trade_ratio = damage_dealt / damage_taken
                if trade_ratio > 1.5:
                    reward += 2
                elif trade_ratio < 0.5:
                    reward -= 2

            fainted_opponent = sum(1 for mon in battle.opponent_team.values() if mon.fainted)
            prior_fainted_opponent = sum(1 for mon in prior_battle.opponent_team.values() if mon.fainted)
            if fainted_opponent > prior_fainted_opponent:
                reward += 15 * (fainted_opponent - prior_fainted_opponent)
                if damage_dealt < 0.5:
                    reward += 5

            fainted_team = sum(1 for mon in battle.team.values() if mon.fainted)
            prior_fainted_team = sum(1 for mon in prior_battle.team.values() if mon.fainted)
            if fainted_team > prior_fainted_team:
                reward -= 10 * (fainted_team - prior_fainted_team)

            if damage_dealt == 0.0 and not battle.finished:
                reward -= 1

        if battle.finished:
            if battle.won:
                reward += 100
                fainted_team = sum(1 for mon in battle.team.values() if mon.fainted)
                if fainted_team == 0:
                    reward += 50
                alive_bonus = (6 - fainted_team) * 7
                reward += alive_bonus
            else:
                reward -= 100
                fainted_team = sum(1 for mon in battle.team.values() if mon.fainted)
                if fainted_team >= 5:
                    reward -= 20

        # Scale the accumulated reward.
        reward /= 3

        return reward

    def _observation_size(self) -> int:
        """Return the declared 56-feature observation size."""
        return 56

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """Encode the 56-feature battle observation."""
        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [mon.current_hp_fraction for mon in battle.opponent_team.values()]
        while len(health_opponent) < 6:
            health_opponent.append(1.0)

        alive_team = [0.0 if mon.fainted else 1.0 for mon in battle.team.values()]
        alive_opponent = [0.0 if mon.fainted else 1.0 for mon in battle.opponent_team.values()]
        while len(alive_opponent) < 6:
            alive_opponent.append(1.0)

        active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon

        own_hp = active.current_hp_fraction if active else 0.0
        own_type1 = self._encode_type(active.type_1 if active else None)
        own_type2 = self._encode_type(active.type_2 if active else None)
        own_speed = min(active.base_stats.get("spe", 100) / 200.0, 1.0) if active else 0.5

        opp_hp = opp_active.current_hp_fraction if opp_active else 0.0
        opp_type1 = self._encode_type(opp_active.type_1 if opp_active else None)
        opp_type2 = self._encode_type(opp_active.type_2 if opp_active else None)
        opp_speed = min(opp_active.base_stats.get("spe", 100) / 200.0, 1.0) if opp_active else 0.5

        move_info = []
        if active and active.moves:
            for move_id in list(active.moves.keys())[:4]:
                move = active.moves[move_id]
                power = min(move.base_power / 150.0, 1.0) if move.base_power else 0.0
                move_type = self._encode_type(move.type)
                eff1 = self._get_effectiveness(move.type, opp_active.type_1 if opp_active else None)
                eff2 = self._get_effectiveness(move.type, opp_active.type_2 if opp_active else None)
                move_info.extend([power, move_type, eff1, eff2])
        while len(move_info) < 16:
            move_info.append(0.0)

        total_hp_own = sum(health_team) / 6.0
        total_hp_opp = sum(health_opponent) / 6.0
        alive_count_own = sum(alive_team) / 6.0
        alive_count_opp = sum(alive_opponent) / 6.0
        hp_advantage = total_hp_own - total_hp_opp
        alive_advantage = alive_count_own - alive_count_opp
        own_status = 1.0 if (active and active.status) else 0.0
        opp_status = 1.0 if (opp_active and opp_active.status) else 0.0

        final_vector = np.array(
            health_team + health_opponent +         # 12
            alive_team + alive_opponent +           # 12 → 24
            [own_hp, own_type1, own_type2, own_speed,
             opp_hp, opp_type1, opp_type2, opp_speed] + # 8 → 32
            move_info +                             # 16 → 48
            [total_hp_own, total_hp_opp, alive_count_own, alive_count_opp,
             hp_advantage, alive_advantage, own_status, opp_status]   # 8 → 56
        )

        return final_vector.astype(np.float32)

    def _encode_type(self, poke_type) -> float:
        """Encode an elemental type as a scalar."""
        if poke_type is None:
            return 0.0
        type_name = str(poke_type).lower().split('.')[-1]
        type_id = self.type_chart.get(type_name, 0)
        return type_id / 17.0

    def _get_effectiveness(self, attack_type, defend_type) -> float:
        """Return the normalized matchup multiplier."""
        if not attack_type or not defend_type:
            return 0.5
        attack_name = str(attack_type).lower().split('.')[-1]
        defend_name = str(defend_type).lower().split('.')[-1]
        eff = self.effectiveness.get((attack_name, defend_name), 1.0)
        if eff == 0.0:
            return 0.0
        elif eff == 0.5:
            return 0.25
        elif eff == 1.0:
            return 0.5
        else:  # 2.0
            return 1.0

########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################

class SingleShowdownWrapper(SingleAgentWrapper):

    def __init__(
        self,
        team_type: str = "random",
        opponent_type: str = "random",
        evaluation: bool = False,
    ):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(
                account_configuration=opponent_configuration
            )
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"

        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)

        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")

        bot_teams = {}

        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(
                    os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8"
                ) as file:
                    bot_teams[team_file[:-4]] = file.read()

        if team_type in bot_teams:
            return bot_teams[team_type]

        return None
