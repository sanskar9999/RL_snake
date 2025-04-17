# --- START OF FILE snake_4_v2.py (with Numba optimizations) ---

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from itertools import count
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import time
import math
# import matplotlib.pyplot as plt # Usually not needed in Colab unless saving plots
from tqdm import tqdm
import wandb  # Import Weights & Biases
import numba # <<< IMPORT NUMBA

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True # Can slow down training, disable if performance is critical

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
GRID_WIDTH = 20
GRID_HEIGHT = 20
INITIAL_SNAKE_LENGTH = 3
ACTIONS = [0, 1, 2, 3]  # Up, Right, Down, Left
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # Corresponds to actions: 0:Up, 1:Right, 2:Down, 3:Left
OPPOSITE_ACTIONS = {0: 2, 1: 3, 2: 0, 3: 1}  # Opposite direction mapping
MAX_SNAKE_LENGTH = 50 # For normalization
RAYCAST_DISTANCE = max(GRID_WIDTH, GRID_HEIGHT) * 2 # Max distance for raycasting, considering wrap-around

# State Representation (Compact 32D)
# Self: Head pos (2, norm), Dir (4, one-hot), Len (1, norm), Raycast self (8, norm) = 15
# Opponent: Head pos (2, norm), Dir (4, one-hot), Len (1, norm), Raycast opp (8, norm) = 15
# Food: Relative pos (2, norm) = 2
# Total = 32
STATE_SIZE = 32

# Hyperparameters - Adjusted based on discussion
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05 # Lowered final epsilon
EPS_DECAY = 0.99995 # Slightly faster decay might be needed with lower EPS_END, monitor!
TARGET_UPDATE_STEPS = 5000 # Update target network based on steps
LEARNING_RATE = 0.00025
MEMORY_SIZE = 100000 # Increased memory size
MAX_STEPS_PER_EPISODE = 1000

# Prioritized Experience Replay (PER) parameters
PER_ALPHA = 0.6
PER_BETA_START = 0.4
# Anneal beta over ~1M steps (adjust based on avg episode length and total training steps)
PER_BETA_FRAMES = 1000000
PER_EPSILON = 1e-6

# Opponent Pool
OPPONENT_POOL_SIZE = 10
UPDATE_OPPONENT_POOL_FREQ = 1000 # Episodes (can also be step-based if preferred)

# Reward Structure - Adjusted
REWARD_FOOD = 5.0
REWARD_KILL = 100.0 # Reduced kill reward
REWARD_LIVING = 0.1 # Slightly increased survival reward
PENALTY_DEATH = -100.0

# Experience replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# --- Numba Optimized Helper Functions for PER ---
@numba.jit(nopython=True)
def _update_priorities_numba(priorities_array, batch_indices, batch_priorities, per_epsilon, current_max_priority):
    """Numba-optimized function to update priorities in the array."""
    new_max_priority = current_max_priority
    for i in range(len(batch_indices)):
        idx = batch_indices[i]
        prio = batch_priorities[i]
        # Ensure priority is positive and update max priority tracker
        new_prio = abs(prio) + per_epsilon
        priorities_array[idx] = new_prio
        if new_prio > new_max_priority:
            new_max_priority = new_prio
    return new_max_priority

@numba.jit(nopython=True)
def _calculate_weights_numba(prios_at_indices, probs_sum, total_samples, beta, alpha):
    """Numba-optimized function to calculate PER weights."""
    # Calculate probabilities for the sampled indices only
    probs = prios_at_indices ** alpha
    weights = (total_samples * probs / probs_sum) ** (-beta)
    return weights.astype(np.float32)


# --- Prioritized Replay Memory ---
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=PER_ALPHA):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        # Ensure priorities is float64 for precision with potentially small numbers
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self.beta = PER_BETA_START
        self._max_priority = 1.0 # Track max priority efficiently

    def push(self, *args):
        # Use the current max priority for new transitions
        prio = self._max_priority

        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.position] = Transition(*args)

        # Ensure priority is float64
        self.priorities[self.position] = float(prio)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, total_steps):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            # Sample only from filled part, ensure it's float64
            prios = self.priorities[:len(self.memory)].astype(np.float64)

        probs = prios ** self.alpha
        probs_sum = probs.sum()

        # --- Sampling (Still uses np.random.choice - potential bottleneck) ---
        if probs_sum == 0 or len(prios) == 0: # Handle edge cases
             # Uniform sampling if all priorities are zero or memory empty
             indices = np.random.choice(len(self.memory), batch_size)
        else:
            probs /= probs_sum
            # Ensure probabilities are float64 for choice
            indices = np.random.choice(len(self.memory), batch_size, p=probs.astype(np.float64))
        # --- End Sampling ---

        samples = [self.memory[i] for i in indices]
        total = len(self.memory)

        # Anneal beta based on total_steps
        self.beta = min(1.0, PER_BETA_START + (1.0 - PER_BETA_START) * total_steps / PER_BETA_FRAMES)

        # --- Calculate Weights using Numba ---
        if probs_sum > 0:
             # Pass only the needed priorities to Numba function
             prios_at_indices = prios[indices]
             weights = _calculate_weights_numba(prios_at_indices, probs_sum, total, self.beta, self.alpha)
        else: # Handle case where probs_sum is zero
             weights = np.ones_like(indices, dtype=np.float32)

        # Normalize for stability (after Numba calculation)
        max_w = weights.max()
        if max_w > 0:
            weights /= max_w
        # --- End Weight Calculation ---

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        # Ensure inputs are NumPy arrays for Numba
        batch_indices_np = np.array(batch_indices, dtype=np.int64)
        batch_priorities_np = np.array(batch_priorities, dtype=np.float64) # Use float64

        # Call the Numba-optimized function
        new_max_p = _update_priorities_numba(
            self.priorities, # Pass the array directly
            batch_indices_np,
            batch_priorities_np,
            PER_EPSILON,
            self._max_priority # Pass current max priority
        )
        # Update the instance's max priority
        self._max_priority = new_max_p


    def __len__(self):
        return len(self.memory)

# --- Dueling Double DQN Network Architecture ---
class DuelingDeepSnakeNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingDeepSnakeNet, self).__init__()
        # Increased hidden layer size slightly
        self.fc1 = nn.Linear(input_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)

        # Dueling streams
        self.value_stream = nn.Linear(hidden_size, 1)
        self.advantage_stream = nn.Linear(hidden_size, output_size)

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.value_stream.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.advantage_stream.weight, nonlinearity='linear')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value_stream(x)
        advantages = self.advantage_stream(x)

        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

# --- DQN Agent with Dueling Double DQN and PER ---
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size

        # Q Networks (Dueling Architecture)
        self.policy_net = DuelingDeepSnakeNet(state_size, hidden_size, action_size).to(device)
        self.target_net = DuelingDeepSnakeNet(state_size, hidden_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        # Prioritized Replay memory
        self.memory = PrioritizedReplayMemory(MEMORY_SIZE)

        # Exploration rate
        self.epsilon = EPS_START

        # Training step counter (internal to agent, used for epsilon decay)
        self.agent_steps_done = 0

    def act(self, state, valid_actions, use_epsilon=True):
        # Act with epsilon-greedy policy
        if use_epsilon:
            self.agent_steps_done += 1 # Increment steps done for epsilon decay
            current_epsilon = self.epsilon
        else:
            current_epsilon = 0.0 # Force greedy action if use_epsilon is False

        if random.random() < current_epsilon:
            # Ensure valid_actions is not empty before choosing
            return random.choice(valid_actions) if valid_actions else 0 # Default to action 0 if no valid actions (should be rare)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)

                # Filter valid actions by setting invalid Q-values to -infinity
                valid_q_values = q_values.clone()
                for action in range(self.action_size):
                    if action not in valid_actions:
                        valid_q_values[0, action] = -float('inf')

                # Handle case where all valid actions have -inf Q-value (should be rare)
                # Also handle if valid_actions was empty initially
                if not valid_actions or torch.isinf(valid_q_values).all():
                     # Fallback: If truly no valid moves, maybe return current direction?
                     # Or just pick the first action (0) as a desperate move.
                     # Let's stick to random valid choice if possible, else 0.
                     return random.choice(valid_actions) if valid_actions else 0

                return valid_q_values.max(1)[1].item()

    def remember(self, state, action, next_state, reward, done):
        # Store transition in PER buffer
        self.memory.push(state, action, next_state, reward, done)

    def learn(self, total_steps): # Pass total_steps for PER beta annealing
        if len(self.memory) < BATCH_SIZE:
            return 0.0 # Not enough samples yet

        # Sample batch using PER
        transitions, indices, weights = self.memory.sample(BATCH_SIZE, total_steps)
        batch = Transition(*zip(*transitions))

        # Convert batch elements to tensors
        state_batch = torch.from_numpy(np.stack(batch.state)).float().to(device)
        action_batch = torch.from_numpy(np.array(batch.action)).long().unsqueeze(1).to(device)
        reward_batch = torch.from_numpy(np.array(batch.reward)).float().to(device)
        weights_batch = torch.from_numpy(weights).float().to(device)

        # Identify non-final next states and handle None values correctly
        non_final_mask_list = [s is not None for s in batch.next_state]
        non_final_mask = torch.BoolTensor(non_final_mask_list).to(device)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]

        if non_final_next_states_list:
            non_final_next_states = torch.from_numpy(np.stack(non_final_next_states_list)).float().to(device)
        else:
            non_final_next_states = torch.empty((0, self.state_size), device=device, dtype=torch.float32)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for non-final states using Double DQN
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        if non_final_mask.sum() > 0:
            # Select best action using policy_net
            next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            # Evaluate that action using target_net
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze(1).detach()

        # Compute the expected Q values: R + gamma * V(s_{t+1})
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss element-wise
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')

        # Apply Importance Sampling weights
        weighted_loss = (weights_batch.unsqueeze(1) * loss).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # Gradient clipping
        self.optimizer.step()

        # Update priorities in PER buffer
        # Ensure TD errors are float64 for Numba update function
        td_errors = loss.squeeze(1).detach().cpu().numpy().astype(np.float64)
        self.memory.update_priorities(indices, td_errors)

        return weighted_loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        # Decay epsilon based on agent's internal step counter
        self.epsilon = max(EPS_END, EPS_START * (EPS_DECAY ** self.agent_steps_done))
        # Alternative: Decay based on total_steps passed from trainer
        # self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY) # Simpler multiplicative decay

    def save_state(self, filename):
        """Save agent's state (networks, optimizer, epsilon, steps)."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'agent_steps_done': self.agent_steps_done,
            # Saving PER memory is complex, usually omitted. Rebuilds on load.
        }, filename)
        print(f"Agent state saved to {filename}")

    def load_state(self, filename):
        """Load agent's state."""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.agent_steps_done = checkpoint.get('agent_steps_done', 0) # Load steps if available
            self.target_net.eval() # Ensure target net is in eval mode
            print(f"Agent state loaded from {filename}")
            # Reset PER beta on load, it will anneal based on new total_steps
            self.memory.beta = PER_BETA_START
            # Reset max priority on load? Or keep it high? Let's reset.
            self.memory._max_priority = 1.0
        else:
            print(f"Warning: Checkpoint file not found at {filename}. Starting fresh.")


# --- Food class ---
@dataclass
class Food:
    position: Tuple[int, int]

# --- Snake class ---
class Snake:
    def __init__(self, x, y, initial_direction=None):
        self.positions = deque([(x, y)])
        self.direction = initial_direction if initial_direction is not None else random.choice(DIRECTIONS)
        self.length = INITIAL_SNAKE_LENGTH
        self.score = 0
        self.lifetime = 0
        self.steps_since_food = 0
        self.max_steps_without_food = GRID_WIDTH * GRID_HEIGHT * 2 # Generous limit

        # Initialize full length correctly
        current_x, current_y = x, y
        for _ in range(1, INITIAL_SNAKE_LENGTH):
            dx, dy = self.direction
            prev_x = (current_x - dx + GRID_WIDTH) % GRID_WIDTH
            prev_y = (current_y - dy + GRID_HEIGHT) % GRID_HEIGHT
            if (prev_x, prev_y) not in self.positions:
                 self.positions.append((prev_x, prev_y))
                 current_x, current_y = prev_x, prev_y
            else:
                # This can happen if initial length > grid size or starts near corner
                # print("Warning: Could not initialize full snake length due to space constraints.")
                break # Stop trying to extend

    def get_head_position(self):
        return self.positions[0]

    def get_action_from_direction(self):
        try:
            return DIRECTIONS.index(self.direction)
        except ValueError:
            # print(f"Warning: Invalid direction {self.direction}")
            # Find the closest direction if needed, or default
            return 0 # Default to UP

    def get_direction_from_action(self, action):
         # Ensure action is valid index
         return DIRECTIONS[action % len(DIRECTIONS)]


    def get_valid_actions(self):
        current_action = self.get_action_from_direction()
        invalid_action = OPPOSITE_ACTIONS.get(current_action) # Use .get for safety

        head_x, head_y = self.get_head_position()
        valid = []
        for action in ACTIONS:
            if action == invalid_action:
                continue
            dx, dy = self.get_direction_from_action(action)
            next_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)
            # Check if next head position is the neck (second segment)
            if len(self.positions) > 1 and next_head == self.positions[1]:
                 continue # Prevent moving directly into the neck
            valid.append(action)

        # If no valid moves (e.g., trapped in a 1x1 space), allow moving back? Or just return empty?
        # Returning empty might cause issues in agent.act if random.choice fails.
        # Let's allow moving back if it's the *only* option (should only happen if length <= 2)
        if not valid and len(self.positions) <= 2 and invalid_action is not None:
             return [invalid_action]
        elif not valid: # Trapped and length > 2, no valid moves
             # This state should lead to death in the next step anyway
             # Return the current direction's action as a placeholder? Or first available?
             return [current_action] # Return current action if completely trapped
        return valid


    def move(self, action):
        # Update direction based on action
        new_direction = self.get_direction_from_action(action)
        self.direction = new_direction

        # Calculate new head position
        head_x, head_y = self.get_head_position()
        dx, dy = self.direction
        new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)

        # Check for self-collision (excluding the tail tip which will move)
        # Create set from deque efficiently
        body_set = set(self.positions)
        tail_tip = None
        if len(self.positions) >= self.length and len(self.positions) > 0: # Tail will move
             tail_tip = self.positions[-1]
             if tail_tip in body_set: # Only discard if it's actually there
                 body_set.discard(tail_tip)

        if new_head in body_set:
            return True, 'self' # Collision occurred, type 'self'

        # Insert new head
        self.positions.appendleft(new_head)

        # Remove tail if snake didn't grow
        if len(self.positions) > self.length:
            self.positions.pop()

        # Update stats
        self.lifetime += 1
        self.steps_since_food += 1

        # Check for starvation
        if self.steps_since_food >= self.max_steps_without_food:
            return True, 'starve' # Starvation occurred

        return False, None # No collision

    def grow(self):
        self.length = min(self.length + 1, MAX_SNAKE_LENGTH)
        self.score += 1 # Score is based on food eaten (or length increase)
        self.steps_since_food = 0


# --- Raycasting Function ---
def _raycast(self, start_pos, direction_vec, max_dist, obstacles_set):
    """Cast a ray, return normalized distance to obstacle or max_dist."""
    current_x, current_y = start_pos
    dx, dy = direction_vec
    grid_width = GRID_WIDTH # Assuming GRID_WIDTH/HEIGHT are accessible globally or passed if needed
    grid_height = GRID_HEIGHT

    for dist in range(1, int(max_dist) + 1): # Ensure max_dist is int
        current_x = (current_x + dx) % grid_width
        current_y = (current_y + dy) % grid_height
        # Check against the set directly
        if (current_x, current_y) in obstacles_set:
            # Normalize distance before returning
            return self._normalize_distance(float(dist))
    # Nothing hit, normalize max_dist before returning
    return self._normalize_distance(float(max_dist))

# --- Environment class ---
class SnakeEnvironment:
    def __init__(self):
        self.snake1: Optional[Snake] = None
        self.snake2: Optional[Snake] = None
        self.foods: List[Food] = []
        self.done = False
        self.steps = 0
        # Precompute grid diagonal for normalization
        self._grid_diag = math.sqrt(GRID_WIDTH**2 + GRID_HEIGHT**2)

    def _raycast(self, start_pos, direction_vec, max_dist, obstacles_set):
        """Cast a ray, return normalized distance to obstacle or max_dist."""
        current_x, current_y = start_pos
        dx, dy = direction_vec
        grid_width = GRID_WIDTH # Assuming GRID_WIDTH/HEIGHT are accessible globally or passed if needed
        grid_height = GRID_HEIGHT

        for dist in range(1, int(max_dist) + 1): # Ensure max_dist is int
            current_x = (current_x + dx) % grid_width
            current_y = (current_y + dy) % grid_height
            # Check against the set directly
            if (current_x, current_y) in obstacles_set:
                # Normalize distance before returning
                return self._normalize_distance(float(dist))
        # Nothing hit, normalize max_dist before returning
        return self._normalize_distance(float(max_dist))

    def reset(self):
        # Create snakes at random positions, ensuring they don't overlap initially
        while True:
            # Ensure minimum distance between starting heads? Maybe not necessary.
            pos1 = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            dir1 = random.choice(DIRECTIONS)
            self.snake1 = Snake(*pos1, initial_direction=dir1)

            pos2 = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            # Ensure head2 not in snake1 body (convert deque to set for faster check)
            snake1_pos_set_init = set(self.snake1.positions)
            while pos2 in snake1_pos_set_init:
                 pos2 = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

            dir2 = random.choice(DIRECTIONS)
            # Avoid immediate head-on collision if starting adjacent
            dx1, dy1 = dir1
            next_pos1 = ((pos1[0] + dx1) % GRID_WIDTH, (pos1[1] + dy1) % GRID_HEIGHT)
            if pos2 == next_pos1 and dir1 == (-dir2[0], -dir2[1]):
                 # Choose a direction that isn't the opposite of dir1
                 possible_dirs = [d for d in DIRECTIONS if d != (-dir1[0], -dir1[1])]
                 dir2 = random.choice(possible_dirs) if possible_dirs else dir1 # Fallback if trapped

            self.snake2 = Snake(*pos2, initial_direction=dir2)

            # Ensure snake2's body doesn't overlap snake1
            overlap = False
            # Re-check snake1 pos set as it might have changed during init
            snake1_pos_set = set(self.snake1.positions)
            for p2 in self.snake2.positions:
                if p2 in snake1_pos_set:
                    overlap = True
                    break
            if not overlap:
                 break # Found valid starting positions

        # Create food
        self.foods = []
        self._spawn_food()

        self.done = False
        self.steps = 0

        # Get initial states
        state1 = self._get_state(self.snake1, self.snake2)
        state2 = self._get_state(self.snake2, self.snake1)

        return state1, state2

    def _spawn_food(self):
        # Spawn new food at a random empty position
        while len(self.foods) < 1:
            # Combine occupied positions efficiently
            occupied = set(self.snake1.positions) | set(self.snake2.positions) | {f.position for f in self.foods}
            # Consider using a precomputed set of all cells if performance is critical here
            # all_cells = set((x, y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT))
            # empty_cells = list(all_cells - occupied)

            # Alternative: Randomly sample until an empty cell is found (usually faster if grid isn't almost full)
            max_attempts = GRID_WIDTH * GRID_HEIGHT * 2 # Safety limit
            for _ in range(max_attempts):
                 pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
                 if pos not in occupied:
                     self.foods.append(Food(pos))
                     return # Food spawned successfully

            # If loop finishes without finding a spot (grid is full)
            print("Warning: No empty cells to spawn food!")
            self.done = True # End the game if no space for food
            break


    def _normalize_pos(self, pos):
        # Avoid division by zero if grid size is 1
        norm_x = pos[0] / max(1, GRID_WIDTH - 1)
        norm_y = pos[1] / max(1, GRID_HEIGHT - 1)
        return [norm_x, norm_y]


    def _normalize_length(self, length):
        return length / (MAX_SNAKE_LENGTH + 1e-9)

    def _normalize_distance(self, dist):
        # Normalize by grid diagonal as a rough max distance
        # Use precomputed diagonal
        return dist / (self._grid_diag + 1e-9)

    def _get_relative_food_pos(self, head_pos, food_pos):
        head_x, head_y = head_pos
        food_x, food_y = food_pos
        dx = food_x - head_x
        dy = food_y - head_y

        # Wrap around logic
        half_width = GRID_WIDTH / 2.0
        half_height = GRID_HEIGHT / 2.0
        if abs(dx) > half_width: dx = -np.sign(dx) * (GRID_WIDTH - abs(dx))
        if abs(dy) > half_height: dy = -np.sign(dy) * (GRID_HEIGHT - abs(dy))

        # Normalize to roughly [-1, 1]
        norm_dx = dx / (half_width + 1e-9)
        norm_dy = dy / (half_height + 1e-9)
        return [norm_dx, norm_dy]

    def _get_state(self, snake, other_snake):
        """Generate the 32-dimensional state vector for the snake."""
        head_pos = snake.get_head_position()
        # head_x, head_y = head_pos # No longer needed here

        # 1. Self Features (15 dims)
        norm_head_pos = self._normalize_pos(head_pos)
        direction_action = snake.get_action_from_direction()
        direction_one_hot = np.zeros(4)
        direction_one_hot[direction_action] = 1.0
        norm_len = self._normalize_length(snake.length)

        # Raycasts for self body (excluding head)
        # Use the obstacle set directly
        self_body_obstacles_set = set(snake.positions)
        self_body_obstacles_set.discard(head_pos) # Don't raycast against own head
        # REMOVED: self_body_obstacles_list = list(self_body_obstacles_set)

        self_raycasts = []
        ray_directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        for d_vec in ray_directions: # Use direction vector tuple
            # Call original _raycast with tuple, set; it returns normalized distance
            norm_dist = self._raycast(head_pos, d_vec, RAYCAST_DISTANCE, self_body_obstacles_set)
            # REMOVED: Normalization step here
            self_raycasts.append(norm_dist)

        # 2. Opponent Features (15 dims)
        other_head_pos = other_snake.get_head_position()
        norm_other_head_pos = self._normalize_pos(other_head_pos)
        other_direction_action = other_snake.get_action_from_direction()
        other_direction_one_hot = np.zeros(4)
        other_direction_one_hot[other_direction_action] = 1.0
        norm_other_len = self._normalize_length(other_snake.length)

        # Raycasts for opponent body (including head this time) from self's perspective
        # Use the obstacle set directly
        opponent_obstacles_set = set(other_snake.positions)
        # REMOVED: opponent_obstacles_list = list(opponent_obstacles_set)

        opponent_raycasts = []
        for d_vec in ray_directions: # Use direction vector tuple
            # Call original _raycast with tuple, set; it returns normalized distance
            norm_dist = self._raycast(head_pos, d_vec, RAYCAST_DISTANCE, opponent_obstacles_set)
            # REMOVED: Normalization step here
            opponent_raycasts.append(norm_dist)

        # 3. Food Features (2 dims)
        if self.foods:
            food_pos = self.foods[0].position
            relative_food = self._get_relative_food_pos(head_pos, food_pos)
        else:
            relative_food = [0.0, 0.0] # No food present

        # Concatenate all features
        state = np.concatenate([
            norm_head_pos, direction_one_hot, [norm_len], self_raycasts,
            norm_other_head_pos, other_direction_one_hot, [norm_other_len], opponent_raycasts,
            relative_food
        ]).astype(np.float32)

        if state.shape[0] != STATE_SIZE:
             raise ValueError(f"State size mismatch! Expected {STATE_SIZE}, got {state.shape[0]}")
        return state

    def step(self, action1, action2):
        self.steps += 1
        if self.done:
            # Return consistent structure even if done
            s1 = self._get_state(self.snake1, self.snake2) if self.snake1 else None
            s2 = self._get_state(self.snake2, self.snake1) if self.snake2 else None
            return (s1, s2), (0.0, 0.0), True, 'N/A', 'N/A'

        # --- Move Snakes ---
        collision1, reason1 = self.snake1.move(action1)
        collision2, reason2 = self.snake2.move(action2)

        head1 = self.snake1.get_head_position()
        head2 = self.snake2.get_head_position()
        # Convert deques to sets for efficient collision checking
        snake1_body_set = set(self.snake1.positions)
        snake2_body_set = set(self.snake2.positions)

        # --- Check Inter-Snake Collisions ---
        # Head-on collision (snakes moved into the same square)
        head_on_collision = (head1 == head2)

        # Snake 1 hitting Snake 2's body (excluding S2's new head if not head-on)
        snake1_hit_snake2_body = head1 in (snake2_body_set - {head2}) if not head_on_collision else False

        # Snake 2 hitting Snake 1's body (excluding S1's new head if not head-on)
        snake2_hit_snake1_body = head2 in (snake1_body_set - {head1}) if not head_on_collision else False

        # Determine who died based on collisions
        died1 = collision1 # Died from self-collision or starvation
        died2 = collision2 # Died from self-collision or starvation

        if head_on_collision:
            # Compare lengths *before* potential growth this step
            len1 = self.snake1.length
            len2 = self.snake2.length
            if len1 > len2:
                died2 = True; reason2 = 'head-on loss'
            elif len2 > len1:
                died1 = True; reason1 = 'head-on loss'
            else: # Equal length head-on
                died1 = True; reason1 = 'head-on draw'
                died2 = True; reason2 = 'head-on draw'
        else:
            if snake1_hit_snake2_body:
                died1 = True; reason1 = 'hit other body'
            if snake2_hit_snake1_body:
                died2 = True; reason2 = 'hit other body'

        # --- Calculate Rewards ---
        reward1 = 0.0
        reward2 = 0.0

        # Food Reward (only if alive)
        food_eaten_by_1 = False
        food_eaten_by_2 = False
        if self.foods:
            food_pos = self.foods[0].position
            # Check if heads landed on food AND the snake didn't die in the same step
            if head1 == food_pos and not died1:
                self.snake1.grow()
                reward1 += REWARD_FOOD
                food_eaten_by_1 = True
            # Check snake 2 only if snake 1 didn't eat (or if both landed there but one died)
            # Ensure head2 exists and snake2 didn't die
            if head2 == food_pos and not died2 and not food_eaten_by_1:
                 self.snake2.grow()
                 reward2 += REWARD_FOOD
                 food_eaten_by_2 = True

            # Remove food if eaten by either snake
            if food_eaten_by_1 or food_eaten_by_2:
                self.foods.pop(0)
                self._spawn_food() # Spawn new food (check self.done again?)

        # Check if spawning food failed and ended the game
        if self.done:
             # If game ended due to no food space, assign death penalties? Or just end?
             # Let's assign penalties if one died for other reasons before this check
             if not died1: reward1 += PENALTY_DEATH # Penalize survivors if game ends abruptly
             if not died2: reward2 += PENALTY_DEATH
             died1 = True
             died2 = True
             reason1 = reason1 if died1 else 'no food space'
             reason2 = reason2 if died2 else 'no food space'

        # Collision Penalties & Kill Rewards (apply after potential food growth)
        if died1 and died2: # Both died in the same step (or game ended)
            # Avoid double penalty if already penalized for no food space
            if reason1 != 'no food space': reward1 += PENALTY_DEATH
            if reason2 != 'no food space': reward2 += PENALTY_DEATH
        elif died1: # Snake 1 died, Snake 2 survived
            reward1 += PENALTY_DEATH
            reward2 += REWARD_KILL # Snake 2 gets kill reward
        elif died2: # Snake 2 died, Snake 1 survived
            reward2 += PENALTY_DEATH
            reward1 += REWARD_KILL # Snake 1 gets kill reward

        # Survival Bonus (only if alive at the end of the step)
        if not died1:
            reward1 += REWARD_LIVING
        if not died2:
            reward2 += REWARD_LIVING

        # --- Check Game End Conditions ---
        self.done = died1 or died2 or self.steps >= MAX_STEPS_PER_EPISODE # self.done might be true from _spawn_food

        # Determine winner/loser status for logging
        outcome1 = 'win' if died2 and not died1 else ('loss' if died1 and not died2 else 'draw')
        outcome2 = 'win' if died1 and not died2 else ('loss' if died2 and not died1 else 'draw')

        # --- Get Next States ---
        # Only get state if the snake is not considered dead for the *next* step
        next_state1 = self._get_state(self.snake1, self.snake2) if not died1 else None
        next_state2 = self._get_state(self.snake2, self.snake1) if not died2 else None

        return (next_state1, next_state2), (reward1, reward2), self.done, outcome1, outcome2


# --- Self-Play Trainer with Opponent Pool ---
class SelfPlayTrainer:
    def __init__(self, state_size, action_size, agent_load_path=None, wandb_log=True):
        self.state_size = state_size
        self.action_size = action_size
        self.agent = DQNAgent(state_size, action_size)
        self.env = SnakeEnvironment()
        self.best_avg_score = -float('inf')
        self.scores_window = deque(maxlen=100) # Agent 1's score
        self.episode_outcomes = deque(maxlen=100) # Track 'win', 'loss', 'draw' for agent 1
        self.total_steps = 0

        # Opponent Pool
        self.opponent_pool = deque(maxlen=OPPONENT_POOL_SIZE)


        # Load agent state if path provided
        if agent_load_path:
            self.agent.load_state(agent_load_path)
            # Re-initialize opponent pool with the loaded agent state
            loaded_state_dict = self.agent.policy_net.state_dict()
            self.opponent_pool.clear()
            for _ in range(OPPONENT_POOL_SIZE):
                # Ensure opponent weights are on CPU if agent is on GPU, or vice versa?
                # Let's keep opponent weights as loaded (could be GPU or CPU tensors)
                # The opponent agent instance will move them to its device when loaded later.
                self.opponent_pool.append(loaded_state_dict.copy())
            print(f"Opponent pool re-initialized with weights from {agent_load_path}")
            # Potentially load total_steps from checkpoint if saved? Assumes agent_steps_done reflects training progress.
            self.total_steps = self.agent.agent_steps_done # Approximate total steps based on loaded agent
        else:
             # Initialize pool with fresh agent weights if not loading
             initial_state_dict = self.agent.policy_net.state_dict()
             for _ in range(OPPONENT_POOL_SIZE):
                 self.opponent_pool.append(initial_state_dict.copy())


        # WandB Logging
        self.wandb_log = wandb_log
        if self.wandb_log:
            # Ensure wandb directory exists
            if not os.path.exists("./wandb"):
                 os.makedirs("./wandb")
            try:
                wandb.init(project="snake-ai-1v1-superhuman-v2", config={ # Changed project name slightly
                    "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "gamma": GAMMA,
                    "eps_start": EPS_START, "eps_end": EPS_END, "eps_decay": EPS_DECAY,
                    "target_update_steps": TARGET_UPDATE_STEPS, "memory_size": MEMORY_SIZE,
                    "state_size": STATE_SIZE, "per_alpha": PER_ALPHA, "per_beta_start": PER_BETA_START,
                    "per_beta_frames": PER_BETA_FRAMES, "opponent_pool_size": OPPONENT_POOL_SIZE,
                    "max_steps_per_episode": MAX_STEPS_PER_EPISODE, "grid_size": f"{GRID_WIDTH}x{GRID_HEIGHT}",
                    "reward_kill": REWARD_KILL, "reward_food": REWARD_FOOD,
                    "reward_living": REWARD_LIVING, "penalty_death": PENALTY_DEATH,
                    "seed": SEED, "device": str(device) # Log device used
                })
                # wandb.watch(self.agent.policy_net, log_freq=1000) # Optional
            except Exception as e:
                 print(f"Error initializing WandB: {e}")
                 print("Disabling WandB logging for this run.")
                 self.wandb_log = False


    def train(self, num_episodes=10000, eval_interval=100, save_interval=1000, checkpoint_dir="checkpoints_v2"):

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Temporary opponent agent instance (lives on the same device as main agent)
        opponent_agent = DQNAgent(self.state_size, self.action_size)
        opponent_agent.policy_net.eval() # Opponent always acts greedily based on loaded weights

        pbar = tqdm(range(1, num_episodes + 1), desc="Training Episodes")
        for episode in pbar:
            state1, state2 = self.env.reset()
            done = False
            episode_reward1 = 0
            episode_loss_sum = 0.0
            learn_steps = 0
            steps_in_episode = 0
            final_outcome1 = 'draw' # Default outcome if max steps reached

            # Select opponent from pool and load its weights
            # Ensure pool is not empty
            if not self.opponent_pool:
                 print("Warning: Opponent pool is empty! Using current agent weights.")
                 opponent_state_dict = self.agent.policy_net.state_dict()
                 self.opponent_pool.append(opponent_state_dict.copy()) # Add current weights back
            else:
                 opponent_state_dict = random.choice(self.opponent_pool)

            # Load state dict, mapping location to the opponent agent's device
            opponent_agent.policy_net.load_state_dict(opponent_state_dict)
            opponent_agent.policy_net.to(device) # Ensure opponent net is on correct device
            opponent_agent.policy_net.eval()


            while not done:
                # Agent 1 (learning agent) action
                if state1 is None: break # Agent 1 already lost
                valid_actions1 = self.env.snake1.get_valid_actions()
                action1 = self.agent.act(state1, valid_actions1, use_epsilon=True)

                # Agent 2 (opponent from pool) action
                if state2 is None: # Opponent already lost, agent 1 should just continue
                    # Choose a valid action for agent 2, although it won't matter
                    action2 = self.env.snake2.get_action_from_direction() if self.env.snake2 else 0
                else:
                    valid_actions2 = self.env.snake2.get_valid_actions()
                    # Opponent acts greedily (use_epsilon=False)
                    action2 = opponent_agent.act(state2, valid_actions2, use_epsilon=False)


                # Environment step
                (next_state1, next_state2), (reward1, reward2), done, outcome1, outcome2 = self.env.step(action1, action2)

                # Remember experience for the learning agent (Agent 1)
                # Only remember if the initial state was valid
                if state1 is not None:
                    self.agent.remember(state1, action1, next_state1, reward1, done) # Use env's done flag

                # Learn (if enough memory)
                loss = self.agent.learn(self.total_steps) # Pass total_steps for PER beta annealing
                if loss > 0: # Only count loss if learning actually happened
                    episode_loss_sum += loss
                    learn_steps += 1

                # Update states
                state1 = next_state1
                state2 = next_state2 # Opponent needs its state for next action

                episode_reward1 += reward1
                steps_in_episode += 1
                self.total_steps += 1
                final_outcome1 = outcome1 # Store the outcome from the step the game ended

                # Decay epsilon (based on agent's internal counter)
                self.agent.decay_epsilon()

                # Update target network periodically based on total steps
                if self.total_steps % TARGET_UPDATE_STEPS == 0:
                    self.agent.update_target_net()
                    # print(f"\nUpdated target network at step {self.total_steps}") # Optional debug print

                # Break if state becomes None (agent died) or game is done
                if state1 is None or done:
                    break


            # --- End of Episode ---
            self.scores_window.append(self.env.snake1.score if self.env.snake1 else 0) # Track score of the learning agent
            self.episode_outcomes.append(final_outcome1) # Track win/loss/draw

            # Update opponent pool periodically (based on episodes)
            if episode % UPDATE_OPPONENT_POOL_FREQ == 0:
                 # Add a copy of the current agent's policy weights to the pool
                 # Ensure weights are moved to CPU before storing if agent is on GPU,
                 # to avoid potential issues if pool is used by CPU-only process later.
                 # state_dict_to_store = {k: v.cpu() for k, v in self.agent.policy_net.state_dict().items()}
                 # Or just store as is, assuming loader handles device mapping:
                 state_dict_to_store = self.agent.policy_net.state_dict().copy()
                 self.opponent_pool.append(state_dict_to_store)
                 # print(f"\nUpdated opponent pool at episode {episode}. Pool size: {len(self.opponent_pool)}")


            # Logging and Saving
            if episode % eval_interval == 0:
                avg_score = np.mean(self.scores_window) if self.scores_window else 0.0
                avg_loss = episode_loss_sum / max(1, learn_steps) # Average loss over learning steps
                win_rate = sum(1 for o in self.episode_outcomes if o == 'win') / max(1, len(self.episode_outcomes))
                loss_rate = sum(1 for o in self.episode_outcomes if o == 'loss') / max(1, len(self.episode_outcomes))
                draw_rate = sum(1 for o in self.episode_outcomes if o == 'draw') / max(1, len(self.episode_outcomes))

                # Update progress bar description
                pbar.set_description(f"Ep {episode} | Avg Score: {avg_score:.2f} | Win Rate: {win_rate:.2f} | Loss: {avg_loss:.4f} | Eps: {self.agent.epsilon:.4f}")

                if self.wandb_log:
                    log_data = {
                        "episode": episode,
                        "score": self.env.snake1.score if self.env.snake1 else 0,
                        "average_score_100ep": avg_score,
                        "win_rate_100ep": win_rate,
                        "loss_rate_100ep": loss_rate,
                        "draw_rate_100ep": draw_rate,
                        "average_loss": avg_loss,
                        "epsilon": self.agent.epsilon,
                        "steps_in_episode": steps_in_episode,
                        "total_steps": self.total_steps,
                        "buffer_size": len(self.agent.memory),
                        "per_beta": self.agent.memory.beta,
                        "snake1_length": self.env.snake1.length if self.env.snake1 else 0,
                        "snake2_length": self.env.snake2.length if self.env.snake2 else 0,
                    }
                    try:
                        wandb.log(log_data, step=self.total_steps) # Log against total_steps
                    except Exception as e:
                         print(f"Error logging to WandB: {e}")


                # Save best agent based on average score (only when window is full)
                if len(self.scores_window) == 100 and avg_score > self.best_avg_score:
                    self.best_avg_score = avg_score
                    best_filename = os.path.join(checkpoint_dir, f"best_agent_avg_score_{avg_score:.2f}.pt")
                    self.agent.save_state(best_filename)
                    if self.wandb_log:
                         try:
                             # Save best model file to WandB run
                             wandb.save(best_filename, base_path=checkpoint_dir, policy="now")
                             print(f"\nSaved new best model with avg score: {avg_score:.2f}")
                         except Exception as e:
                              print(f"Error saving best model to WandB: {e}")


            # Save periodic checkpoint
            if episode % save_interval == 0:
                ckpt_filename = os.path.join(checkpoint_dir, f"checkpoint_ep_{episode}_steps_{self.total_steps}.pt")
                self.agent.save_state(ckpt_filename)
                # Optional: Save checkpoint to WandB (can consume storage)
                # if self.wandb_log:
                #     try:
                #         wandb.save(ckpt_filename, base_path=checkpoint_dir, policy="now")
                #     except Exception as e:
                #          print(f"Error saving checkpoint to WandB: {e}")


        # --- End of Training ---
        pbar.close()
        final_filename = os.path.join(checkpoint_dir, f"final_agent_ep_{num_episodes}_steps_{self.total_steps}.pt")
        self.agent.save_state(final_filename)
        print(f"Training finished. Final agent saved to {final_filename}")
        if self.wandb_log:
            try:
                wandb.save(final_filename, base_path=checkpoint_dir, policy="now")
                wandb.finish()
            except Exception as e:
                 print(f"Error saving final model/finishing WandB: {e}")


# --- Testing and Visualization ---
# (Keep the test_agent and render_game functions from the previous version,
#  they should work with the updated agent and environment)
def test_agent(agent_path, num_games=10, render=False, opponent_mode='agent', opponent_path=None):
    """Test a trained agent against different opponents."""
    print(f"\n--- Testing Agent: {agent_path} ---")
    print(f"Opponent Mode: {opponent_mode}")

    # Determine device for testing (can be different from training)
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing using device: {test_device}")

    # Load agent
    agent = DQNAgent(STATE_SIZE, 4) # Action size is 4
    # Load state onto the chosen test_device
    if os.path.exists(agent_path):
        checkpoint = torch.load(agent_path, map_location=test_device)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        # No need to load optimizer, target net, epsilon for testing usually
        agent.policy_net.to(test_device)
        agent.policy_net.eval()
        print(f"Agent policy network loaded from {agent_path} to {test_device}")
    else:
        print(f"Error: Agent model file not found at {agent_path}. Aborting test.")
        return None, None

    # Opponent setup
    opponent_agent = None
    if opponent_mode == 'agent' or opponent_mode == 'self':
        opp_path = opponent_path if opponent_mode == 'self' and opponent_path else agent_path
        print(f"Loading opponent from: {opp_path}")
        if not os.path.exists(opp_path):
             print(f"Error: Opponent model file not found at {opp_path}. Using random opponent.")
             opponent_mode = 'random' # Fallback
        else:
            opponent_agent = DQNAgent(STATE_SIZE, 4)
            opp_checkpoint = torch.load(opp_path, map_location=test_device)
            opponent_agent.policy_net.load_state_dict(opp_checkpoint['policy_net_state_dict'])
            opponent_agent.policy_net.to(test_device)
            opponent_agent.policy_net.eval()
            print(f"Opponent policy network loaded from {opp_path} to {test_device}")


    # Create environment
    env = SnakeEnvironment()

    # Statistics
    agent_scores = []
    opponent_scores = []
    game_lengths = []
    agent_wins = 0
    opponent_wins = 0
    draws = 0

    for game in tqdm(range(num_games), desc="Testing Games"):
        state1, state2 = env.reset()
        done = False
        game_steps = 0
        final_outcome1 = 'draw' # Default

        while not done:
            # Agent 1 (the agent being tested) action
            action1 = 0 # Default
            if state1 is not None:
                valid_actions1 = env.snake1.get_valid_actions()
                # Use agent.act with use_epsilon=False for deterministic testing
                action1 = agent.act(state1, valid_actions1, use_epsilon=False)
            else: # Agent 1 already lost
                 done = True; break


            # Agent 2 (opponent) action
            action2 = 0 # Default
            if state2 is not None:
                valid_actions2 = env.snake2.get_valid_actions()
                if opponent_mode == 'random':
                    action2 = random.choice(valid_actions2) if valid_actions2 else 0
                elif opponent_agent: # Handles 'agent' and 'self' modes
                    action2 = opponent_agent.act(state2, valid_actions2, use_epsilon=False)
                else: # Fallback if opponent failed to load
                    action2 = random.choice(valid_actions2) if valid_actions2 else 0
            # else: Opponent already lost, action2 doesn't matter much


            # Take actions
            (next_state1, next_state2), (reward1, reward2), done, outcome1, outcome2 = env.step(action1, action2)
            game_steps += 1
            final_outcome1 = outcome1

            # Render game state if requested
            if render:
                render_game(env)
                time.sleep(0.05) # Slow down rendering

            # Update states
            state1 = next_state1
            state2 = next_state2

            if done:
                if final_outcome1 == 'win': agent_wins += 1
                elif final_outcome1 == 'loss': opponent_wins += 1
                else: draws += 1


        # Record statistics
        agent_scores.append(env.snake1.score if env.snake1 else 0)
        opponent_scores.append(env.snake2.score if env.snake2 else 0)
        game_lengths.append(game_steps)


    # Print summary
    print(f"\n--- Testing Summary ({num_games} games) ---")
    print(f"Agent: {os.path.basename(agent_path)}")
    opp_desc = opponent_mode
    if opponent_mode == 'self' and opponent_path:
        opp_desc += f" ({os.path.basename(opponent_path)})"
    elif opponent_mode == 'agent':
         opp_desc += f" (self)"

    print(f"Opponent Mode: {opp_desc}")
    print(f"Agent Avg Score: {np.mean(agent_scores):.2f} +/- {np.std(agent_scores):.2f}")
    print(f"Opponent Avg Score: {np.mean(opponent_scores):.2f} +/- {np.std(opponent_scores):.2f}")
    print(f"Avg Game Length: {np.mean(game_lengths):.2f} +/- {np.std(game_lengths):.2f}")
    print(f"Agent Wins: {agent_wins} ({agent_wins/num_games*100:.1f}%)")
    print(f"Opponent Wins: {opponent_wins} ({opponent_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print("--------------------------------------")

    return agent_scores, game_lengths


def render_game(env):
    """Render the current game state to the console."""
    grid = [['.' for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    # Add food
    for food in env.foods:
        x, y = food.position
        if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH: grid[y][x] = 'F'

    # Add snake2 body (draw before head)
    if env.snake2:
        # Iterate safely over deque copy
        for pos in list(env.snake2.positions)[1:]:
            x, y = pos
            if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH: grid[y][x] = 'x'

    # Add snake1 body (draw before head)
    if env.snake1:
        # Iterate safely over deque copy
        for pos in list(env.snake1.positions)[1:]:
            x, y = pos
            if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH: grid[y][x] = 'o'

    # Add snake2 head
    if env.snake2:
        x, y = env.snake2.get_head_position()
        if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH: grid[y][x] = '2'

    # Add snake1 head (draw last)
    if env.snake1:
        x, y = env.snake1.get_head_position()
        if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH: grid[y][x] = '1'

    # Print grid
    # Use ANSI escape codes for potentially better clearing
    print("\033[H\033[J", end="") # Clears screen and moves cursor to home
    # os.system('cls' if os.name == 'nt' else 'clear') # Alternative
    print(f"Step: {env.steps}/{MAX_STEPS_PER_EPISODE}")
    s1_score = env.snake1.score if env.snake1 else 'DEAD'
    s2_score = env.snake2.score if env.snake2 else 'DEAD'
    s1_len = env.snake1.length if env.snake1 else 'DEAD'
    s2_len = env.snake2.length if env.snake2 else 'DEAD'
    print(f"Score: S1={s1_score} | S2={s2_score}")
    print(f"Length: S1={s1_len} | S2={s2_len}")
    print('+' + '-' * GRID_WIDTH + '+')
    for row in grid: print('|' + ''.join(row) + '|')
    print('+' + '-' * GRID_WIDTH + '+')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Snake AI v2: D3QN + PER + Self-Play (Numba Optimized)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode')
    parser.add_argument('--episodes', type=int, default=20000, help='Number of episodes for training')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load model state')
    parser.add_argument('--save_interval', type=int, default=2000, help='Save checkpoint every N episodes')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluate and log every N episodes')
    parser.add_argument('--checkpoint_dir', type=str, default='snake_checkpoints_v2_numba', help='Directory for checkpoints') # Changed default dir
    parser.add_argument('--games', type=int, default=20, help='Number of games for testing')
    parser.add_argument('--render', action='store_true', help='Render game during testing')
    parser.add_argument('--opponent', type=str, default='agent', choices=['agent', 'random', 'self'], help='Opponent type for testing')
    parser.add_argument('--opponent_model', type=str, default=None, help='Path to load opponent model state (used only if --opponent=self)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--profile_episodes', type=int, default=0, help='Number of episodes to profile (0 to disable profiling)')


    args = parser.parse_args()

    if args.mode == 'train':
        print("--- Starting Training (v2 - Numba Optimized) ---")
        trainer = SelfPlayTrainer(STATE_SIZE, 4, # Action size is 4
                                  agent_load_path=args.load_model,
                                  wandb_log=not args.no_wandb)

        # --- Profiling Setup ---
        if args.profile_episodes > 0:
            import cProfile, pstats, io # Keep imports local if only used here
            print(f"--- PROFILING for {args.profile_episodes} episodes ---")
            profiler = cProfile.Profile()
            profiler.enable() # Start watching function calls

            # Run training for a limited number of episodes for profiling
            trainer.train(num_episodes=args.profile_episodes, # Use the profile_episodes argument
                          eval_interval=args.profile_episodes + 1, # Disable eval during profiling
                          save_interval=args.profile_episodes + 1, # Disable saving during profiling
                          checkpoint_dir=args.checkpoint_dir)

            profiler.disable() # Stop watching
            print("--- Profiling Finished ---")

            # --- Analyze and Print Profiling Results ---
            s = io.StringIO() # Capture output in memory
            # Sort stats by 'cumulative time' spent in function and its subfunctions
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
            ps.print_stats(50) # Print the top 50 time-consuming functions

            print("\n--- cProfile Results (Top 50 by Cumulative Time) ---")
            print(s.getvalue()) # Print the captured output

        else:
            # --- Normal Training Run (No Profiling) ---
            trainer.train(num_episodes=args.episodes,
                          eval_interval=args.eval_interval,
                          save_interval=args.save_interval,
                          checkpoint_dir=args.checkpoint_dir)


    elif args.mode == 'test':
        if not args.load_model:
            print("Error: Please provide a model path using --load_model for testing.")
            return
        if args.opponent == 'self' and not args.opponent_model:
            print("Warning: Opponent mode is 'self' but --opponent_model not specified. Using the main agent model as opponent.")
            args.opponent_model = args.load_model # Default to self-play against same agent

        print(f"--- Starting Testing (v2 - Numba Optimized) ---")
        test_agent(args.load_model,
                   num_games=args.games,
                   render=args.render,
                   opponent_mode=args.opponent,
                   opponent_path=args.opponent_model) # Pass opponent model path

if __name__ == "__main__":
    main()
    # Example execution commands:
    # Train from scratch for 50k episodes:
    # python snake_4_v2.py --mode train --episodes 50000 --checkpoint_dir snake_models_v2_numba_50k

    # Continue training from a checkpoint:
    # python snake_4_v2.py --mode train --episodes 100000 --load_model snake_models_v2_numba_50k/checkpoint_ep_50000_steps_....pt --checkpoint_dir snake_models_v2_numba_100k

    # Test the final agent against itself with rendering:
    # python snake_4_v2.py --mode test --load_model snake_models_v2_numba_100k/final_agent_....pt --games 10 --render --opponent agent

    # Profile the Numba-optimized code (e.g., on GPU if available):
    # python snake_4_v2.py --mode train --profile_episodes 200 --checkpoint_dir snake_models_v2_numba_profile

# --- END OF FILE snake_4_v2.py (with Numba optimizations) ---