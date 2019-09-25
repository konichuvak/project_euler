from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

ROOK_ACTIONS = frozenset({(0, -1), (-1, 0), (0, 1), (1, 0)})


class GridWorldGenerator(object):
    
    def __init__(self, width: int, height: int,
                 actions: List[tuple] = ROOK_ACTIONS,
                 default_reward: float = -1,
                 other_rewards: Dict[tuple, float] = None,
                 blocks: Tuple[Tuple[int, int]] = None,
                 ):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        if blocks:
            for block in blocks:
                self.grid[block] = 2
        self.blocks = blocks
        self.default_reward = default_reward
        self.non_default_rewards = other_rewards
        self.rewards = self._generate_rewards(default_reward, other_rewards)
        self.actions = list(map(np.array, actions))
    
    def _generate_rewards(self, default_reward, other_rewards):
        """
        Creates reward grid
        :param default_reward:  default reward for transitioning to a given state in grid world
        :param other_rewards:   dict with coordinates as keys and reward as values for other rewards
        :return:
        """
        rewards = np.ones((self.width, self.height)) * default_reward
        if other_rewards is None:
            other_rewards = {}
        
        for coord, r in other_rewards.items():
            rewards[coord] = r
        return rewards


class GridWorld(GridWorldGenerator):
    
    def __init__(self, *args, grid_dim: int, gamma: float, **kwargs):
        super(GridWorld, self).__init__(*args, **kwargs)
        self.grid_dim = grid_dim
        self.gamma = gamma
        self.prob = 1 / len(self.actions)
    
    def state_transition(self, state, action):
        """
        :param state:   tuple of (x, y) coordinates of the agent in the grid
        :param action:  performed action
        :return:        (i, j) tuple of the next state and the reward associated with the transition
        """
        state = np.array(state)
        next_state = tuple(state + action)
        x, y = next_state
        
        # check boundary conditions
        if x < 0 or y < 0 or x >= self.grid_dim or y >= self.grid_dim:
            next_state = tuple(state)
        
        # reward = self.rewards[tuple(state)]
        reward = self.rewards[next_state]
        return next_state, reward
    
    def is_terminal(self, x, y):
        return (x == self.grid_dim - 1 and y == self.grid_dim - 1)
    
    def compute_optimal_policy(self, state_values: np.ndarray):
        """
        Computes optimal policy pi* that is greedy wrt the current value function v

        :param state_values:    optimal state-value matrix V*
        :return:                optimal policy Pi*
        """
        opt_policy = np.zeros(state_values.shape, dtype=np.int)
        
        for x in range(self.grid_dim):
            for y in range(self.grid_dim):
                # choose the policy to be the action that maximizes the expected return for this state
                action_values = list()
                if self.is_terminal(x, y):
                    continue
                for action in self.actions:
                    next_state, reward = self.state_transition((x, y), action)
                    if next_state == (x, y):
                        continue
                    v = self.prob * (reward + self.gamma * state_values[
                        next_state])  # bellman update
                    action_values.append(v)
                
                opt_policy[x, y] = np.argmax(np.round(action_values, 5))
        
        return opt_policy
    
    def traverse(self, policy: np.ndarray, start_state: tuple = (0, 0)):
        state = start_state
        total_rew = self.rewards[state]
        opt_path = [state]
        for _ in range(self.grid_dim ** 4):
            if self.is_terminal(*state):
                break
            state, reward = self.state_transition(state, self.actions[
                policy[state[0], state[1]]])
            # print(state, reward)
            opt_path.append(state)
            total_rew += reward
        else:
            next_state, _ = self.state_transition(state, self.actions[
                policy[state[0], state[1]]])
            raise Exception(f'Stuck in {state}. Next state is {next_state}')
        return opt_path, total_rew
    
    def gridworld_policy_iteration(self, in_place, theta):
        """
        Iterative Policy Evaluation for estimating Vpi
        :param in_place:    whether to use the updated value function immediately overwriting the old values
        :param theta:       convergence parameter
        :return:
        """
        
        state_values_sequence = list()
        state_values = np.zeros((self.grid_dim, self.grid_dim))
        new_state_values = state_values.copy()
        delta = float('inf')
        
        while delta > theta:
            value = new_state_values if in_place else state_values
            for x in tqdm(range(self.grid_dim - 1, -1, -1)):
                for y in range(self.grid_dim - 1, -1, -1):
                    v = list()
                    if self.is_terminal(x, y):
                        continue
                    for action in self.actions:
                        next_state, reward = self.state_transition((x, y),
                                                                   action)
                        if (x, y) == next_state:
                            continue
                        v.append(reward + self.gamma * value[
                            next_state])  # bellman update
                    new_state_values[x, y] = max(v)
            print(delta)
            delta = np.sum(np.abs(new_state_values - state_values))
            state_values = new_state_values.copy()
            state_values_sequence.append(state_values)
        
        return state_values_sequence
    
    def policy_evaluation(self, state_values: np.ndarray, policy: np.ndarray,
                          in_place: bool,
                          value_iteration: bool = True):
        """
        Iterative Policy Evaluation for estimating V under policy Pi

        :param state_values:    state-value matrix V_hat
        :param policy:          policy matrix Pi
        :param in_place:        whether to use the updated value function immediately overwriting the old values
        :return:                updated state-value matrix V_hat
        """
        
        new_state_values = state_values.copy()
        delta, theta = float('inf'), 1e-2
        
        while delta > theta:
            value = new_state_values if in_place else state_values
            for y in range(self.grid_dim - 1, -1, -1):
                for x in range(self.grid_dim - 1, -1, -1):
                    
                    if self.is_terminal(x, y):
                        continue
                    
                    if value_iteration:
                        greedy_action_value = -float('inf')
                        for action in self.actions:
                            next_state, reward = self.state_transition((x, y),
                                                                       action)
                            if (x, y) == next_state:
                                continue
                            v = reward + self.gamma * value[next_state]
                            if v > greedy_action_value:
                                greedy_action_value = v
                        new_state_values[x, y] = greedy_action_value
                    else:
                        next_state, reward = self.state_transition((x, y),
                                                                   policy[x, y])
                        if (x, y) == next_state:
                            continue
                        new_state_values[x, y] = reward + self.gamma * value[
                            next_state]
            
            delta = np.sum(np.abs(new_state_values - state_values))
            print(delta)
            state_values = new_state_values.copy()
        
        return state_values
    
    def policy_improvement(self, state_values: np.ndarray, policy: np.ndarray):
        """
        Makes a new policy Pi' that improves on an original policy Pi by making it greedy wrt the value function of
        the original policy V_Pi

        :param state_values:    state-value matrix V_hat
        :param policy:          policy matrix Pi
        :return:                bool for whether policy converged or not; updated policy matrix Pi'
        """
        policy_stable = True
        policy_improvements = 0
        
        for x in range(self.grid_dim - 1, -1, -1):
            for y in range(self.grid_dim - 1, -1, -1):
                
                if self.is_terminal(x, y):
                    continue
                
                old_action = policy[x, y]
                
                # choose the policy to be the action that maximizes the expected return for this state
                greedy_action = policy[x, y]
                action_value = -float('inf')
                for a, action in enumerate(self.actions):
                    next_state, reward = self.state_transition((x, y), action)
                    if (x, y) == next_state:
                        continue
                    v = reward + self.gamma * state_values[next_state]
                    if v > action_value:
                        action_value = v
                        greedy_action = a
                
                if old_action != greedy_action:
                    policy_improvements += 1
                    policy_stable = False
                
                policy[x, y] = greedy_action
        
        print(f'Policy improved in {policy_improvements} states')
        # print(state_values)
        return policy_stable, policy
    
    def policy_iteration(self, in_place: bool,
                         value_iteration: bool = False) -> (
        np.ndarray, np.ndarray):
        
        state_values = np.zeros((self.grid_dim, self.grid_dim))
        policy = np.zeros(state_values.shape, dtype=np.int)
        
        state_values_seq = list()
        policy_seq = list()
        
        policy_stable = False
        iteration = 0
        
        while not policy_stable:
            # for i in tqdm(range(30)):
            iteration += 1
            state_values = self.policy_evaluation(state_values, policy,
                                                  in_place, value_iteration)
            policy_stable, policy = self.policy_improvement(state_values,
                                                            policy)
            state_values_seq.append(state_values)
            policy_seq.append(policy)
            if value_iteration:
                break
        
        return state_values_seq, policy_seq
