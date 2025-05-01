# Section 2: Designing the environment
import numpy as np
"""
This code represents a partially completed environment class to use for the assignment.
Functions that will require changes in order to build the environment will be marked with 
three asterisks (***) in the function description comment. While 
"""

class GridWorld():
  def __init__(self, goal_locations: list[tuple[int, int]], goal_rewards: list[int]):
    """A grid-based environment with an agent that can be stepped towards a goal. 
    
    Args: 
      goal_locations: List of locations of the goal(s)
      goal_rewards: List of rewards. The pair is respective to the goals location.
    """

    # List of goals
    self._goal_loc = goal_locations
    self._goal_rewards = goal_rewards

    if len(goal_locations) == 0: 
      raise ValueError("Need at least one goal location.")

    if len(goal_locations) != len(goal_rewards): 
      raise ValueError("The same number of goal locations and goal rewards should be specified.")
    
    # Build the GridWorld
    self._initialize_gridworld()

  def _initialize_gridworld(self):
    """
    GridWorld initialisation.
    
    ***Here, you should define the properties of GridWorld according to the figure in the handout,
    as well as the penalties/rewards associated with the features of GridWorld. Specifically, 
    _cliffs, _walls, _terminal_locs, _terminal_rewards, and _starting_loc. ***
    """

    # Define the shape of the GridWorld (rows, columns)
    self._shape = (13, 10)

    # Define locations of cliffs (unsafe tiles that terminate the episode with negative reward)
    self._cliffs = []

    # Row 0: (0,0) to (0,9)
    for j in range(10):
        self._cliffs.append((0, j))

    # Row 4: (4,0) to (4,3) and (4,7) to (4,9)
    for j in range(4):
        self._cliffs.append((4, j))
    for j in range(7, 10):
        self._cliffs.append((4, j))

    # Row 5: (5,0) to (5,3) and (5,7) to (5,9)
    for j in range(4):
        self._cliffs.append((5, j))
    for j in range(7, 10):
        self._cliffs.append((5, j))

    # Row 6: (6,0) to (6,3) and (6,7) to (6,9)
    for j in range(4):
        self._cliffs.append((6, j))
    for j in range(7, 10):
        self._cliffs.append((6, j))

    # Row 7: (7,0) to (7,3)
    for j in range(4):
        self._cliffs.append((7, j))

    # Row 8: (8,0) to (8,3)
    for j in range(4):
        self._cliffs.append((8, j))

    # Define locations of walls
    self._walls = [
        (9, 0), (9, 1), (9, 2), (9, 3), (7, 7), (7, 8),
        (7, 9)
    ]

    # Define terminal locations (cliffs and goals)
    self._terminal_locs = self._cliffs.copy() + [(2,1), (9, 8)]

    # Define terminal rewards (negative for cliffs, positive for goals)
    self._terminal_rewards = [-100] * len(self._cliffs) + [250, 5000]

    # Define the starting location
    self._starting_loc = (11, 1)
    
    # Reward for standard tiles in GridWorld
    self._default_reward = 0
    
    # Maximum duration of each episode
    self._max_t = 500

    # Actions
    self._action_size = 4
    self._direction_names = ['N', 'E', 'S', 'W']

    # Sanity check for initial implementation
    if self._cliffs is None or self._walls is None or self._terminal_locs is None or self._terminal_rewards is None:
      raise NotImplementedError("Ensure all attributes are properly initialized.")
    
    # States
    self._locations = []
    for i in range(self._shape[0]):
      for j in range(self._shape[1]):
        loc = (i, j)
        # Adding the state to locations if it is not an obstacle
        if self._is_location(loc):
          self._locations.append(loc)
    self._state_size = len(self._locations)

    # Set deterministic transitions
    self._probability_success_of_action = 1

    # Neighbours - each line is a state, ranked by state-number, each column is a direction (N, E, S, W)
    self._neighbours = np.zeros((self._state_size, 4))
    for state in range(self._state_size):
      loc = self.get_loc_from_state(state)

      neighbour = (loc[0]-1, loc[1])  # North
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('N')] = self._get_state_from_loc(neighbour)
      else:
        self._neighbours[state][self._direction_names.index('N')] = state

      neighbour = (loc[0], loc[1]+1)  # East
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('E')] = self._get_state_from_loc(neighbour)
      else:
        self._neighbours[state][self._direction_names.index('E')] = state

      neighbour = (loc[0]+1, loc[1])  # South
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('S')] = self._get_state_from_loc(neighbour)
      else:
        self._neighbours[state][self._direction_names.index('S')] = state

      neighbour = (loc[0], loc[1]-1)  # West
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('W')] = self._get_state_from_loc(neighbour)
      else:
        self._neighbours[state][self._direction_names.index('W')] = state

    # Absorbing condition to mark terminal locations that end an episode
    self._absorbing = np.zeros((1, self._state_size), dtype=bool)
    for a in self._terminal_locs:
      absorbing_state = self._get_state_from_loc(a)
      self._absorbing[0, absorbing_state] = True
      
    # Transition matrix
    self._T = np.zeros((self._state_size, self._state_size, self._action_size))
    for action in range(self._action_size):
      for outcome in range(4):
        if action == outcome:
          prob = 1 - 3.0 * ((1.0 - self._probability_success_of_action) / 3.0)
        else:
          prob = (1.0 - self._probability_success_of_action) / 3.0

        # Write this probability in the transition matrix
        for prior_state in range(self._state_size):
          if not self._absorbing[0, prior_state]:
            post_state = self._neighbours[prior_state, outcome]
            post_state = int(post_state)
            self._T[prior_state, post_state, action] += prob

    # Reward matrix
    self._R = np.ones((self._state_size, self._state_size, self._action_size))
    self._R = self._default_reward * self._R
    for i in range(len(self._terminal_rewards)):
      post_state = self._get_state_from_loc(self._terminal_locs[i])
      self._R[:, post_state, :] = self._terminal_rewards[i]

    # Reset the environment output variables once initialised
    self.reset()

    
    
  def get_cliffs_loc(self) -> list[tuple[int, int]]:
    """
    Get the location of cliffs. USED ONLY FOR PLOTTING!
    """
    return self._cliffs
  
  def get_walls_loc(self) -> list[tuple[int, int]]:
    """
    Get the location of walls. USED ONLY FOR PLOTTING!
    """
    return self._walls

  def get_gridshape(self) -> tuple[int, int]:
    """
    Get the shape of the gridworld. USED ONLY FOR PLOTTING!
    """
    return self._shape
  
  def get_starting_loc(self) -> tuple[int, int]:
    """
    Get starting location. USED ONLY FOR PLOTTING!
    """
    return self._starting_loc
  
  def get_goal_loc(self) -> list[tuple[int, int]]:
    """
    Get the location(s) of goal(s). USED ONLY FOR PLOTTING!
    """
    return self._goal_loc

  def _is_location(self, loc: tuple[int, int]) -> bool:
    """
    Check if the location a valid state (not out of GridWorld and not a wall)

    Args: 
      loc: location of the state. 

    Returns: 
      Whether the location a valid state.
    
    
    *** Here you should set controls that prevent movement into walls and outside of
    the grid ***
    
    """
    # Check if the location is out of bounds
    if loc[0] < 0 or loc[0] >= self._shape[0] or loc[1] < 0 or loc[1] >= self._shape[1]:
        return False

    # Check if the location is a wall
    if self._walls is not None and loc in self._walls:
        return False

    return True



  def _get_state_from_loc(self, loc):
    """
    Get the state number corresponding to a given location
    input: loc {tuple} -- location of the state
    output: index {int} -- corresponding state number
    """
    return self._locations.index(tuple(loc))


  def get_loc_from_state(self, state):
    """
    Get the state number corresponding to a given location
    input: index {int} -- state number
    output: loc {tuple} -- corresponding location
    """
    return self._locations[state]

  def is_terminal(self, state: int) -> bool:
    """Returns if a state is terminal
    """
    return self._absorbing[0, state]

  def get_action_size(self) -> int:
    """Returns the number of actions. 
    """
    return self._action_size

  def get_state_size(self) -> int:
    """Returns the number of states.
    """
    return self._state_size

  # Functions used to perform episodes in the GridWorld environment
  def reset(self) -> tuple[int, int, bool, bool]:
    """
    Reset the environment state to starting state
    
    Returns: 
      - t, the current timestep
      - state, the current state of the envionment
      - reward, the current reward
      - done, True if reach a terminal state / False otherwise
    """
    self._t = 0
    self._state = self._get_state_from_loc(self._starting_loc)
    self._reward = 0
    self._done = False

    return self._t, self._state, self._reward, self._done


  def step(self, action: int) -> tuple[int, int, bool, bool]:
    """
    Perform an action in the environment

    Args: 
      action: action to perform

    Returns:
      - t, the current timestep
      - state, the current state of the envionment
      - reward, the current reward
      - done, True if reach a terminal state / False otherwise
    """

    # If environment already finished, print an error
    if self._done or self._absorbing[0, self._state]:
      print("Please reset the environment")
      return self._t, self._state, self._reward, self._done

    # Look for the first possible next states (so get a reachable state even if probability_success = 0)
    new_state = 0
    while self._T[self._state, new_state, action] == 0:
      new_state += 1
    assert self._T[self._state, new_state, action] != 0, "Selected initial state should be probability 0, something might be wrong in the environment."

    # Find the first state for which probability of occurence matches the random value
    total_probability = self._T[self._state, new_state, action]
    while (total_probability < self._probability_success_of_action) and (new_state < self._state_size-1):
     new_state += 1
     total_probability += self._T[self._state, new_state, action]
    assert self._T[self._state, new_state, action] != 0, "Selected state should be probability 0, something might be wrong in the environment."

    # Setting new t, state, reward and boolean 'done' condition
    self._t += 1
    self._reward = self._R[self._state, new_state, action]
    self._done = self._absorbing[0, new_state] or self._t > self._max_t
    self._state = new_state
    return self._t, self._state, self._reward, self._done
  