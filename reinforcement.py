try:
    from enum import Enum
except Exception as e:
    from aenum import Enum

import matplotlib.pyplot as plt
import random
import copy
import pickle

class StateType(object):
    """
    Represents the possible states in the grid world.
    """
    NONE = 0,
    FIRE = 1,
    BLACK_HOLE = 2,
    TREASURE = 3

    _state_values = {NONE: 0,
                     FIRE: -10, 
                     BLACK_HOLE: -100, 
                     TREASURE: 100}

    def __init__(self, state):
        self._state = state

    @property
    def value(self):
        return self._state_values[self._state]

    @property
    def name(self):
        return self._state

class Actions(object):
    """
    Represents the possible actions in the grid world.
    """
    def __init__(self):
        self._actions = Enum('Actions', 'up down left right')

    @property
    def up(self):
        return self._actions.up

    @property
    def down(self):
        return self._actions.down

    @property
    def left(self):
        return self._actions.left

    @property
    def right(self):
        return self._actions.right

    @staticmethod
    def get_orthogonal_action(action, direction = True):
        """
        Given a direction returns the orthogonal direction. Boolean flag 'direction'
        switches which way the orthogonal direction is chosen.
        """
        if action == Actions.up:
            if direction:
                return Actions.right
            else:
                return Actions.left
        elif action == Actions.down:
            if direction:
                return Actions.right
            else:
                return Actions.left
        elif action == Actions.left:
            if direction:
                return Actions.up
            else:
                return Actions.down
        elif action == Actions.right:
            if direction:
                return Actions.up
            else:
                return Actions.down
        else:
            raise ValueError("Invalid input.")

    @staticmethod
    def get_random_action():
        # Make deterministic
        # random.seed(1)
        # Random number between inclusive 1, 100
        val = random.randint(1,4)
        if val == 1:
            return Actions.up
        elif val == 2:
            return Actions.down
        elif val == 3:
            return Actions.left
        elif val == 4:
            return Actions.right


class ActionValueFunction(object):
    """
    Data structure representing the Action-Value function held by the agent
    when exploring the environment.
    """
    def __init__(self, initial_state = {}):
        self._action_val_fun = initial_state

    def add_state(self, state, action_rewards = {}):
        """
        Warning! Overrides state if already exists.
        """
        self._action_val_fun[state] = action_rewards

    def update_action(self, state, action, reward):
        """
        """
        if not state in self._action_val_fun:
            self.add_state(state)
        self._action_val_fun[state][action] = reward

    def get_expected_reward_value(self, state, action):
        if state in self._action_val_fun:
            if action in self._action_val_fun[state]:
                return self._action_val_fun[state][action]
        return 0

    def get_best_next_action(self, state):
        up = self.get_expected_reward_value(state, Actions.up)
        down = self.get_expected_reward_value(state, Actions.down)
        left = self.get_expected_reward_value(state, Actions.left)
        right = self.get_expected_reward_value(state, Actions.right)
        maximum_reward = max([up, down, left, right])
        if up == maximum_reward:
            return Actions.up
        elif down == maximum_reward:
            return Actions.down
        elif left == maximum_reward:
            return Actions.left
        elif right == maximum_reward:
            return Actions.right

    def __str__(self):
        return str(self._action_val_fun)

class World(object):
    """
    Represents the environment and the current position of the agent within
    the world.
    """
    def __init__(self, width = 5, height = 5, initial_state = (0,0)):
        self._world = [[StateType(StateType.NONE).value for x in range(width)] 
        for y in range(height)]
        self._state = initial_state
        self._width = width
        self._height = height

    def add_statetype_to_cell(self, cell_coord, state_type):
        """
        N.B. Actually adds the reward of the statetype to cell as that's all
        we're interested in.
        cell_coord is the (x,y) coordinate.
        """
        print "State Value: {}".format(str(state_type.value))
        self._world[cell_coord[0]][cell_coord[1]] = state_type.value

    def get_reward_in_cell(self, cell_coord):
        return self._world[cell_coord[0]][cell_coord[1]]

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def __str__(self):
        _str = ""
        for row in self._world:
            _str += str(row) + "\n"
        return str(_str)

# Required function signatures below here

def env_move_det(s, a):
    flag = False
    s = copy.copy(s)
    if a == Actions.up:
        # Go up if we're not already at the top
        if s.state[0] > 0:
            s.state = (s.state[0] - 1, s.state[1])
    elif a == Actions.down:
        # Go down if we're not already at the bottom
        if s.state[0] < s.height - 1:
            s.state = (s.state[0] + 1, s.state[1])
    elif a == Actions.left:
        # Go down if we're not already at far left
        if s.state[1] > 0:
            s.state = (s.state[0], s.state[1] - 1)
    elif a == Actions.right:
        # Go right if we're not already at far right
        if s.state[1] < s.width - 1:
            s.state = (s.state[0], s.state[1] + 1)
    return s

def env_move_sto(s, a):
    """ 
    Follow the desired action 80 percent of time, performing orthogonal actions for
    the other 20 percent of the time. 
    """
    # Make deterministic
    # random.seed(1)
    # Random number between inclusive 1, 100
    val = random.randint(1,100)
    if val <= 80:
        # Do original action
        return env_move_det(s, a)
    elif val > 80 and val <= 90:
        # Do orthogonal action 1
        a = Actions.get_orthogonal_action(a, True)
        return env_move_det(s, a)
    elif val > 90:
        # Do orthogonal action 2
        a = Actions.get_orthogonal_action(a, False)
        return env_move_det(s, a)
    else:
        raise ValueError("Invalid input.")

def env_reward(s, a, next_s):
    """Gets the reward in the next state"""
    return next_s.get_reward_in_cell(next_s.state)

def agt_choose(s, epsilon, policy):
    # Make deterministic
    # random.seed(1)
    # Random number between inclusive 1, 100
    val = random.randint(1, 100)
    epsilon *= 100
    if val < epsilon:
        # Do random
        return Actions.get_random_action()
    else:
        # Follow policy
        return policy.get_best_next_action(s.state)

def agt_learn_sarsa(alpha, s, a, r, next_s, next_a, action_val_function, gamma = 0.99):
    action_val_function = copy.copy(action_val_function)
    q_state = action_val_function.get_expected_reward_value(s.state, a)
    q_state_next = action_val_function.get_expected_reward_value(next_s.state, next_a)
    new_expected_reward = ((1 - alpha) * q_state) + (alpha * (r + (gamma * (q_state_next))))
    action_val_function.update_action(s.state, a, new_expected_reward)
    return action_val_function

def agt_learn_q(alpha, s, a, r, next_s, action_val_function, gamma = 0.99):
    action_val_function = copy.copy(action_val_function)
    best_next_action = action_val_function.get_best_next_action(next_s.state)
    q_state = action_val_function.get_expected_reward_value(s.state, a)
    q_state_next = action_val_function.get_expected_reward_value(next_s.state, best_next_action)
    new_expected_reward = ((1 - alpha) * q_state) + alpha * (r + gamma * (q_state_next))
    action_val_function.update_action(s.state, a, new_expected_reward)
    return action_val_function

def agt_learn_final(alpha, s, a, r, action_val_function):
    new_expected_reward = ((1 - alpha) * 0) + alpha * (r)
    action_val_function.update_action(s.state, a, new_expected_reward)
    return action_val_function

def agt_reset_value(world, action_val_function):
    optimistic_val = 150
    for i in range(world.width - 1):
        for j in range(world.height -1):
            action_val_function.add_state((i,j),{Actions.up:    optimistic_val, 
                                                 Actions.down:  optimistic_val,
                                                 Actions.right: optimistic_val,
                                                 Actions.left:  optimistic_val})

def learning_procedure(learning = "sarsa", 
                       deterministic = True, 
                       epsilon = 0.1, 
                       alpha = 0.1,
                       epsilon_dynamic = False,
                       gamma = 0.99, 
                       epochs = 500,
                       episodes = 500):
    # Create the world
    world_width = 5
    world_height = 6
    size_s = world_width * world_height
    s = World(width = world_width, height = world_height)
    s.add_statetype_to_cell((2,1), StateType(StateType.FIRE))
    s.add_statetype_to_cell((5,1), StateType(StateType.BLACK_HOLE))
    s.add_statetype_to_cell((1,3), StateType(StateType.BLACK_HOLE))
    s.add_statetype_to_cell((2,3), StateType(StateType.FIRE))
    s.add_statetype_to_cell((5,4), StateType(StateType.TREASURE))
    print s # initial state


    avf = ActionValueFunction()

    EPOCHS = epochs
    EPISODES = episodes
    EPSILON = epsilon
    T = size_s
    ALPHA = alpha
    GAMMA = gamma
    epsilon_dynamic = epsilon_dynamic


    # Assume only SARSA and Q-Learning for now
    if learning == "sarsa":
        learning_is_sarsa = True
    else:
        learning_is_sarsa = False

    # Apply deterministic action updates
    learning_is_deterministic = deterministic

    # Clear reward array
    rewards = [0.0 for episode in range(EPISODES)]
    init_action_val_up = [0.0 for episode in range(EPISODES)]
    init_action_val_down = [0.0 for episode in range(EPISODES)]
    init_action_val_left = [0.0 for episode in range(EPISODES)]
    init_action_val_right = [0.0 for episode in range(EPISODES)]

    for epochs in range(EPOCHS):
        agt_reset_value(s, avf)
        for episode in range(EPISODES):
            #print "h"
            learning = episode < EPISODES - 50

            # learning = True
            # print learning
            if epsilon_dynamic:
                eps = 1.0 - float(episode) / float(EPISODES)
            else:
                eps = EPSILON if learning else 0
            cumulative_gamma = 1
            # initial state done above
            a = agt_choose(s, eps, avf)


            # up
            cur_init_action_val_up = avf.get_expected_reward_value((0,0), Actions.up)
            init_action_val_up[episode] += float(cur_init_action_val_up) / float(EPOCHS)

            # down
            cur_init_action_val_down = avf.get_expected_reward_value((0,0), Actions.down)
            init_action_val_down[episode] += float(cur_init_action_val_down) / float(EPOCHS)

            # left
            cur_init_action_val_left = avf.get_expected_reward_value((0,0), Actions.left)
            init_action_val_left[episode] += float(cur_init_action_val_left) / float(EPOCHS)

            # right
            cur_init_action_val_right = avf.get_expected_reward_value((0,0), Actions.right)
            init_action_val_right[episode] += float(cur_init_action_val_right) / float(EPOCHS)


            for timestep in range(T):
                if learning_is_deterministic:
                    next_s = env_move_det(s,a)
                else:
                    next_s = env_move_sto(s,a)

                r = env_reward(s, a, next_s)

                rewards[episode] += float(cumulative_gamma * r) / float(EPOCHS)
                cumulative_gamma *= GAMMA

                next_a = agt_choose(next_s, eps, avf)

                # Update while learning
                if(learning):
                    if(next_s.get_reward_in_cell(next_s.state) == 100 or
                       next_s.get_reward_in_cell(next_s.state) == -100 or
                       timestep == T - 1):
                        avf = agt_learn_final(ALPHA, s, a, r, avf)
                        break
                    else:
                        if learning_is_sarsa:
                            avf = agt_learn_sarsa(ALPHA, s, a, r, next_s, next_a, avf)
                        else:
                            avf = agt_learn_q(ALPHA, s, a, r, next_s, avf)
                    a = next_a
                    s = next_s

                # Absorbing state, even after learning
                if(next_s.get_reward_in_cell(next_s.state) == 100 or
                       next_s.get_reward_in_cell(next_s.state) == -100 or
                       timestep == T - 1):
                    break

    init_action_vals = [init_action_val_up, init_action_val_down, init_action_val_left, init_action_val_right]
    return rewards, init_action_vals

def run_experiment_on_learning_variations(epsilon = 0.1, alpha = 0.1, loc = 4):
    """
    Shows plot for complete experiment.
    """
    rewards_s_d, x = learning_procedure(learning = "sarsa", deterministic = True, 
        epsilon = epsilon, alpha = alpha)
    rewards_s_s, x = learning_procedure(learning = "sarsa", deterministic = False, 
        epsilon = epsilon, alpha = alpha)
    rewards_q_d, x = learning_procedure(learning = "q-learn", deterministic = True, 
        epsilon = epsilon, alpha = alpha)
    rewards_q_s, x = learning_procedure(learning = "q-learn", deterministic = False, 
        epsilon = epsilon, alpha = alpha)

    line_1, = plt.plot(rewards_s_d, label='SARSA Deterministic', lw=1, color='g')
    line_2, = plt.plot(rewards_s_s, label='SARSA Stochastic', lw=1, color='r')
    line_3, = plt.plot(rewards_q_d, label='Q-Learning Deterministic', lw=1, color='b')
    line_4, = plt.plot(rewards_q_s, label='Q-Learning Stochastic', lw=1, color='k')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.axis([0, 500, -100, 150])
    plt.legend(handles=[line_1, line_2, line_3, line_4], loc=loc)
    plt.show()

    all_results = [rewards_s_d, rewards_s_s, rewards_q_d, rewards_q_s]
    title = str(epsilon) + "-" + str(alpha)
    with open('results-' + title + '.txt', 'wb') as fp:
        pickle.dump(all_results, fp)

def run_experiment_epsilon_dynamic(alpha = 0.05):
    """
    Shows plot for complete experiment.
    """
    rewards_s_d, x = learning_procedure(learning = "sarsa", deterministic = True, 
        alpha = alpha, epsilon_dynamic = True)
    rewards_s_s, x = learning_procedure(learning = "sarsa", deterministic = False, 
        alpha = alpha, epsilon_dynamic = True)
    rewards_q_d, x = learning_procedure(learning = "q-learn", deterministic = True, 
        alpha = alpha, epsilon_dynamic = True)
    rewards_q_s, x = learning_procedure(learning = "q-learn", deterministic = False, 
        alpha = alpha, epsilon_dynamic = True)

    line_1, = plt.plot(rewards_s_d, label='SARSA Deterministic', lw=1, color='g')
    line_2, = plt.plot(rewards_s_s, label='SARSA Stochastic', lw=1, color='r')
    line_3, = plt.plot(rewards_q_d, label='Q-Learning Deterministic', lw=1, color='b')
    line_4, = plt.plot(rewards_q_s, label='Q-Learning Stochastic', lw=1, color='k')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.axis([0, 500, -100, 150])
    plt.legend(handles=[line_1, line_2, line_3, line_4], loc=2)
    plt.show()

    all_results = [rewards_s_d, rewards_s_s, rewards_q_d, rewards_q_s]
    title = str("epsilon") + "-" + str(alpha)
    with open('results-epsilon-dynamic-' + title + '.txt', 'wb') as fp:
        pickle.dump(all_results, fp)

def run_experiment_initial_action_value():
    """
    Shows plot for complete experiment.
    """
    rewards, init_action_vals = learning_procedure(learning = "q-learn", deterministic = True, 
        epsilon = 0.1, alpha = 0.1)

    up = init_action_vals[0]
    down = init_action_vals[1]
    left = init_action_vals[2]
    right =init_action_vals[3]

    line_1, = plt.plot(up, label='Up', lw=1, color='g')
    line_2, = plt.plot(down, label='Down', lw=1, color='r')
    line_3, = plt.plot(left, label='Left', lw=1, color='b')
    line_4, = plt.plot(right, label='Right', lw=1, color='k')
    plt.ylabel('Initial State Value Function Action')
    plt.xlabel('Episode')
    plt.axis([0, 500, -100, 150])
    plt.legend(handles=[line_1, line_2, line_3, line_4], loc=4)
    plt.show()
    title = str("epsilon")
    with open('results-epsilon-dynamic-' + title + '.txt', 'wb') as fp:
        pickle.dump(init_action_vals, fp)

if __name__ == "__main__":

    # Experiments
    run_experiment_on_learning_variations(epsilon = 0.1, alpha = 0.1)
    run_experiment_on_learning_variations(epsilon = 1, alpha = 1, loc = 2)
    run_experiment_epsilon_dynamic(alpha = 0.05)
    run_experiment_initial_action_value()
