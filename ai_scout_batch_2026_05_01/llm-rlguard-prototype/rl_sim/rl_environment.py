import random
from typing import Dict, Any, List, Tuple

class SimpleTextEnvironment:
    def __init__(self):
        self.current_state_key: str = "start"
        self._base_state_descriptions: Dict[str, str] = {
            "start": "You are at the village entrance. Paths lead north (forest) and east (river).",
            "forest": "You are in a dense forest. You see berries. You can 'collect_berries' or 'return_to_start'.",
            "treasure_found_and_chest_open": "You found the hidden treasure in the chest! Episode complete. You are still by the river."
        }
        self.valid_actions_map: Dict[str, List[str]] = {
            "start": ["go_north_forest", "go_east_river"],
            "forest": ["collect_berries", "return_to_start"],
            "river_closed_chest": ["open_chest", "return_to_start"],
            "river_open_chest": ["return_to_start"],
            "treasure_found_and_chest_open": []
        }
        self.episode_length = 0
        self.max_episode_length = 10
        self.berries_collected_this_episode = 0
        self.chest_status = "closed"

    def reset(self) -> Dict[str, str]:
        self.current_state_key = "start"
        self.episode_length = 0
        self.berries_collected_this_episode = 0
        self.chest_status = "closed"
        return self._get_observation()

    def _get_observation(self) -> Dict[str, str]:
        if self.current_state_key == "river":
            if self.chest_status == "closed":
                description = "You are by a flowing river. A chest lies near the bank."
            else:
                description = "You are by a flowing river. The chest is now empty."
            return {"text": description}
        else:
            return {"text": self._base_state_descriptions.get(self.current_state_key, "Unknown state.")}

    def get_available_actions(self) -> List[str]:
        if self.current_state_key == "river":
            if self.chest_status == "closed":
                return self.valid_actions_map["river_closed_chest"]
            else:
                return self.valid_actions_map["river_open_chest"]
        else:
            return self.valid_actions_map.get(self.current_state_key, [])

    def step(self, action: str) -> Tuple[Dict[str, str], float, bool, Dict[str, Any]]:
        reward = 0.0
        done = False
        info: Dict[str, Any] = {"valid_action": True, "message": ""}

        self.episode_length += 1

        if action not in self.get_available_actions():
            reward = -0.5
            info["valid_action"] = False
            info["message"] = f"Invalid action: '{action}'. Available: {', '.join(self.get_available_actions())}"
            if self.episode_length >= self.max_episode_length:
                done = True
                reward -= 5.0
                info["message"] += " Episode ended due to max length."
            return self._get_observation(), reward, done, info

        if self.current_state_key == "start":
            if action == "go_north_forest":
                self.current_state_key = "forest"
            elif action == "go_east_river":
                self.current_state_key = "river"
        elif self.current_state_key == "forest":
            if action == "collect_berries":
                self.berries_collected_this_episode += 1
                reward = 1.0
                info["berries_collected"] = self.berries_collected_this_episode
            elif action == "return_to_start":
                self.current_state_key = "start"
        elif self.current_state_key == "river":
            if action == "open_chest":
                if self.chest_status == "closed":
                    reward = 10.0
                    self.chest_status = "open"
                    self.current_state_key = "treasure_found_and_chest_open"
                    done = True
                    info["message"] = "You successfully opened the chest and found treasure!"
                else:
                    reward = -0.1
                    info["message"] = "The chest is already open and empty."
            elif action == "return_to_start":
                self.current_state_key = "start"

        if self.episode_length >= self.max_episode_length and not done:
            done = True
            reward -= 5.0
            info["message"] = "Episode ended due to max length."

        info["current_state_key"] = self.current_state_key
        info["episode_length"] = self.episode_length
        info["berries_collected_this_episode"] = self.berries_collected_this_episode
        info["chest_status"] = self.chest_status

        return self._get_observation(), reward, done, info

    def render(self) -> None:
        obs = self._get_observation()
        print(f"Current Observation: {obs['text']}")
        print(f"Available Actions: {', '.join(self.get_available_actions())}")
        print(f"Episode Length: {self.episode_length}/{self.max_episode_length}")
        print(f"Berries Collected: {self.berries_collected_this_episode}")
        print(f"Chest Status: {self.chest_status}")
        print("-" * 30)

    @property
    def observation_space(self) -> Dict[str, str]:
        return {"type": "text", "description": "String describing the current environment state."}

    @property
    def action_space(self) -> List[str]:
        all_actions = set()
        for actions_list in self.valid_actions_map.values():
            all_actions.update(actions_list)
        return sorted(list(all_actions))