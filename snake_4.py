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
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb  # Import Weights & Biases

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
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

# Hyperparameters
BATCH_SIZE = 256 # Increased batch size often helps with stability
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1 # Ensure persistent exploration
EPS_DECAY = 0.9999 # Slower decay to encourage longer exploration
TARGET_UPDATE = 100 # Update target network less frequently
LEARNING_RATE = 0.00025 # Common LR for Adam with DQN
MEMORY_SIZE = 50000 # Reduced memory for faster sampling, adjust based on RAM
MAX_STEPS_PER_EPISODE = 1000 # Max steps before episode ends (prevents infinite games)

# Prioritized Experience Replay (PER) parameters
PER_ALPHA = 0.6  # Priority exponent
PER_BETA_START = 0.4  # Initial importance sampling exponent
PER_BETA_FRAMES = 10000 * 50 # Anneal beta over ~500k steps (adjust based on avg episode length)
PER_EPSILON = 1e-6 # Small value added to priorities

# Opponent Pool
OPPONENT_POOL_SIZE = 10
UPDATE_OPPONENT_POOL_FREQ = 1000 # Episodes

# Experience replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# --- Prioritized Replay Memory ---
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=PER_ALPHA):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self.beta = PER_BETA_START

    def push(self, *args):
        max_prio = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.position] = Transition(*args)
            
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
            
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max() # Normalize for stability
        weights = np.array(weights, dtype=np.float32)

        # Anneal beta
        self.beta = min(1.0, self.beta + (1.0 - PER_BETA_START) / PER_BETA_FRAMES)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + PER_EPSILON # Add epsilon to ensure non-zero priority

    def __len__(self):
        return len(self.memory)

# --- Dueling Double DQN Network Architecture ---
class DuelingDeepSnakeNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingDeepSnakeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Dueling streams
        self.value_stream = nn.Linear(hidden_size, 1)
        self.advantage_stream = nn.Linear(hidden_size, output_size)

        # Initialize weights (optional but good practice)
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

        # Training step counter
        self.steps_done = 0

    def act(self, state, valid_actions):
        # Act with epsilon-greedy policy
        self.steps_done += 1 # Increment steps done for epsilon decay
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)

                # Filter valid actions by setting invalid Q-values to -infinity
                valid_q_values = q_values.clone()
                for action in range(self.action_size):
                    if action not in valid_actions:
                        valid_q_values[0, action] = -float('inf')

                return valid_q_values.max(1)[1].item()

    def remember(self, state, action, next_state, reward, done):
        # Store transition in PER buffer
        self.memory.push(state, action, next_state, reward, done)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0 # Not enough samples yet

        # Sample batch using PER
        transitions, indices, weights = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Convert batch elements to tensors
        state_batch = torch.from_numpy(np.stack(batch.state)).float().to(device)
        action_batch = torch.from_numpy(np.array(batch.action)).long().unsqueeze(1).to(device)
        reward_batch = torch.from_numpy(np.array(batch.reward)).float().to(device)
        weights_batch = torch.from_numpy(weights).float().to(device)

        # Identify non-final next states
        non_final_mask = torch.BoolTensor([s is not None for s in batch.next_state]).to(device)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]

        if non_final_next_states_list:
            non_final_next_states = torch.from_numpy(np.stack(non_final_next_states_list)).float().to(device)
        else:
            # Handle case where all next states are final (e.g., batch_size=1 and episode ends)
            non_final_next_states = torch.empty((0, self.state_size), device=device, dtype=torch.float32)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        if non_final_mask.sum() > 0:
            # Double DQN: Select best action using policy_net, evaluate using target_net
            next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze(1).detach()

        # Compute the expected Q values: R + gamma * V(s_{t+1})
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss (Smooth L1 loss) element-wise
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')

        # Apply Importance Sampling weights
        weighted_loss = (weights_batch.unsqueeze(1) * loss).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()

        # Update priorities in PER buffer
        td_errors = loss.squeeze(1).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        return weighted_loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def save_state(self, filename):
        """Save agent's state (networks, optimizer, epsilon)."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            # Note: Saving replay buffer is often skipped due to size, but can be done
            # 'replay_memory': self.memory # Requires custom serialization for PER
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
            self.steps_done = checkpoint.get('steps_done', 0) # Load steps_done if available
            # Ensure target net is in eval mode after loading
            self.target_net.eval()
            print(f"Agent state loaded from {filename}")
            # Update PER beta based on loaded steps if needed (or reset)
            # self.memory.beta = min(1.0, PER_BETA_START + (1.0 - PER_BETA_START) * self.steps_done / PER_BETA_FRAMES)

        else:
            print(f"Warning: Checkpoint file not found at {filename}. Starting fresh.")


# --- Food class ---
@dataclass
class Food:
    position: Tuple[int, int]

# --- Snake class ---
class Snake:
    def __init__(self, x, y, initial_direction=None):
        self.positions = deque([(x, y)]) # Use deque for efficient pop/append
        self.direction = initial_direction if initial_direction is not None else random.choice(DIRECTIONS)
        self.length = INITIAL_SNAKE_LENGTH
        self.score = 0
        self.lifetime = 0
        self.steps_since_food = 0
        self.max_steps_without_food = GRID_WIDTH * GRID_HEIGHT * 2 # Generous limit

        # Initialize full length correctly, avoiding immediate self-collision
        current_x, current_y = x, y
        for _ in range(1, INITIAL_SNAKE_LENGTH):
            # Add segments *behind* the initial head based on the *opposite* direction
            dx, dy = self.direction
            prev_x = (current_x - dx + GRID_WIDTH) % GRID_WIDTH
            prev_y = (current_y - dy + GRID_HEIGHT) % GRID_HEIGHT
            # Check if the spot is already taken (unlikely but possible with short init length)
            if (prev_x, prev_y) not in self.positions:
                 self.positions.append((prev_x, prev_y))
                 current_x, current_y = prev_x, prev_y
            else:
                # Fallback if space behind is blocked (shouldn't happen with reasonable start pos)
                # Try adding perpendicular? Or just break? Let's break for simplicity.
                print("Warning: Could not initialize full snake length due to space constraints.")
                break


    def get_head_position(self):
        return self.positions[0]

    def get_action_from_direction(self):
        try:
            return DIRECTIONS.index(self.direction)
        except ValueError:
             # Should not happen if direction is always valid
            print(f"Warning: Invalid direction {self.direction}")
            return 0 # Default to Up

    def get_direction_from_action(self, action):
        return DIRECTIONS[action]

    def get_valid_actions(self):
        current_action = self.get_action_from_direction()
        invalid_action = OPPOSITE_ACTIONS[current_action]
        return [a for a in ACTIONS if a != invalid_action]

    def move(self, action):
        # Update direction based on action
        new_direction = self.get_direction_from_action(action)
        self.direction = new_direction

        # Calculate new head position
        head_x, head_y = self.get_head_position()
        dx, dy = self.direction
        new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)

        # Check for self-collision (excluding the tail tip which will move)
        # Need to convert deque to list/set for efficient checking
        body_without_tail = list(self.positions)
        if len(body_without_tail) >= self.length: # Only exclude tail if not growing
             body_without_tail = body_without_tail[:-1]

        if new_head in body_without_tail:
            return True  # Collision occurred

        # Insert new head
        self.positions.appendleft(new_head) # More efficient than insert(0, ...)

        # Remove tail if snake didn't grow
        if len(self.positions) > self.length:
            self.positions.pop() # Remove from the right (tail)

        # Update stats
        self.lifetime += 1
        self.steps_since_food += 1

        # Check for starvation
        if self.steps_since_food >= self.max_steps_without_food:
            return True  # Starvation occurred

        return False  # No collision

    def grow(self):
        self.length = min(self.length + 1, MAX_SNAKE_LENGTH) # Cap length
        self.score += 1 # Score is based on food eaten
        self.steps_since_food = 0

# --- Environment class ---
class SnakeEnvironment:
    def __init__(self):
        self.snake1: Optional[Snake] = None
        self.snake2: Optional[Snake] = None
        self.foods: List[Food] = []
        self.done = False
        self.steps = 0

    def reset(self):
        # Create snakes at random positions, ensuring they don't overlap initially
        while True:
            pos1 = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            dir1 = random.choice(DIRECTIONS)
            self.snake1 = Snake(*pos1, initial_direction=dir1)

            pos2 = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
             # Ensure pos2 is not in snake1's initial body
            while pos2 in self.snake1.positions:
                 pos2 = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

            dir2 = random.choice(DIRECTIONS)
            # Ensure directions are not immediately head-on if close
            if pos1 == pos2 and dir1 == (-dir2[0], -dir2[1]):
                 dir2 = random.choice([d for d in DIRECTIONS if d != (-dir1[0], -dir1[1])])

            self.snake2 = Snake(*pos2, initial_direction=dir2)

            # Ensure snake2's body doesn't overlap snake1
            overlap = False
            for p2 in self.snake2.positions:
                if p2 in self.snake1.positions:
                    overlap = True
                    break
            if not overlap:
                 break # Found valid starting positions

        # Create food
        self.foods = []
        self._spawn_food() # Spawn initial food

        self.done = False
        self.steps = 0

        # Get initial states
        state1 = self._get_state(self.snake1, self.snake2)
        state2 = self._get_state(self.snake2, self.snake1)

        return state1, state2

    def _spawn_food(self):
        # Spawn new food at a random empty position
        while len(self.foods) < 1: # Ensure only one food item exists
            occupied = set(self.snake1.positions) | set(self.snake2.positions) | {f.position for f in self.foods}
            
            # Generate all possible cells
            all_cells = set((x, y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT))
            empty_cells = list(all_cells - occupied)

            if empty_cells:
                pos = random.choice(empty_cells)
                self.foods.append(Food(pos))
            else:
                # Grid is full? End the game? Or just wait?
                print("Warning: No empty cells to spawn food!")
                # For now, let the game continue without new food.
                break

    def _normalize_pos(self, pos):
        """Normalize position to [0, 1]."""
        return [pos[0] / (GRID_WIDTH -1 + 1e-9), pos[1] / (GRID_HEIGHT -1 + 1e-9)]

    def _normalize_length(self, length):
        """Normalize length to [0, 1]."""
        return length / (MAX_SNAKE_LENGTH + 1e-9)

    def _normalize_distance(self, dist):
        """Normalize raycast distance."""
        # Normalize by max possible distance (diagonal across grid, considering wrap)
        # A simpler normalization might be just by RAYCAST_DISTANCE
        return dist / (RAYCAST_DISTANCE + 1e-9)

    def _raycast(self, start_pos, direction_vec, max_dist, obstacles):
        """Cast a ray, return normalized distance to obstacle or max_dist."""
        current_x, current_y = start_pos
        dx, dy = direction_vec
        obstacle_set = set(obstacles) # Faster lookups

        for dist in range(1, max_dist + 1):
            current_x = (current_x + dx) % GRID_WIDTH
            current_y = (current_y + dy) % GRID_HEIGHT
            if (current_x, current_y) in obstacle_set:
                return self._normalize_distance(dist)
        return self._normalize_distance(max_dist) # Nothing hit within max_dist

    def _get_relative_food_pos(self, head_pos, food_pos):
        """Calculate normalized relative food position with wrap-around."""
        head_x, head_y = head_pos
        food_x, food_y = food_pos

        # Calculate differences considering wrap-around
        dx = food_x - head_x
        dy = food_y - head_y

        if dx > GRID_WIDTH / 2:
            dx -= GRID_WIDTH
        elif dx < -GRID_WIDTH / 2:
            dx += GRID_WIDTH

        if dy > GRID_HEIGHT / 2:
            dy -= GRID_HEIGHT
        elif dy < -GRID_HEIGHT / 2:
            dy += GRID_HEIGHT

        # Normalize to roughly [-1, 1] range
        norm_dx = dx / (GRID_WIDTH / 2 + 1e-9)
        norm_dy = dy / (GRID_HEIGHT / 2 + 1e-9)

        return [norm_dx, norm_dy]


    def _get_state(self, snake, other_snake):
        """Generate the 32-dimensional state vector for the snake."""
        head_pos = snake.get_head_position()
        other_head_pos = other_snake.get_head_position()

        # 1. Self Features (15 dims)
        norm_head_pos = self._normalize_pos(head_pos)
        direction_action = snake.get_action_from_direction()
        direction_one_hot = np.zeros(4)
        direction_one_hot[direction_action] = 1.0
        norm_len = self._normalize_length(snake.length)

        # Raycasts for self body (excluding head)
        self_body_obstacles = list(snake.positions)[1:]
        self_raycasts = []
        # Directions: N, NE, E, SE, S, SW, W, NW (relative to grid axes)
        ray_directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        for d_vec in ray_directions:
            dist = self._raycast(head_pos, d_vec, RAYCAST_DISTANCE, self_body_obstacles)
            self_raycasts.append(dist)

        # 2. Opponent Features (15 dims)
        norm_other_head_pos = self._normalize_pos(other_head_pos)
        other_direction_action = other_snake.get_action_from_direction()
        other_direction_one_hot = np.zeros(4)
        other_direction_one_hot[other_direction_action] = 1.0
        norm_other_len = self._normalize_length(other_snake.length)

        # Raycasts for opponent body (excluding head) from self's perspective
        opponent_body_obstacles = list(other_snake.positions)[1:]
        opponent_raycasts = []
        for d_vec in ray_directions:
            dist = self._raycast(head_pos, d_vec, RAYCAST_DISTANCE, opponent_body_obstacles)
            opponent_raycasts.append(dist)

        # 3. Food Features (2 dims)
        if self.foods:
            food_pos = self.foods[0].position
            relative_food = self._get_relative_food_pos(head_pos, food_pos)
        else:
            relative_food = [0.0, 0.0] # No food present

        # Concatenate all features
        state = np.concatenate([
            norm_head_pos,
            direction_one_hot,
            [norm_len],
            self_raycasts,
            norm_other_head_pos,
            other_direction_one_hot,
            [norm_other_len],
            opponent_raycasts,
            relative_food
        ]).astype(np.float32)

        if state.shape[0] != STATE_SIZE:
             raise ValueError(f"State size mismatch! Expected {STATE_SIZE}, got {state.shape[0]}")

        return state

    def step(self, action1, action2):
        self.steps += 1

        # Check if game is already done (shouldn't happen with proper loop control)
        if self.done:
            # Return terminal state (None) and zero rewards
            return (None, None), (0.0, 0.0), True, False, False # next_states, rewards, done, killed1, killed2

        # --- Move Snakes ---
        collision1 = self.snake1.move(action1) # Checks self-collision and starvation
        collision2 = self.snake2.move(action2) # Checks self-collision and starvation

        head1 = self.snake1.get_head_position()
        head2 = self.snake2.get_head_position()

        # --- Check Inter-Snake Collisions ---
        head_on_collision = (head1 == head2)
        snake1_hit_snake2_body = head1 in list(self.snake2.positions)[1:] # Check against body excluding head
        snake2_hit_snake1_body = head2 in list(self.snake1.positions)[1:] # Check against body excluding head

        # Update collision flags based on inter-snake collisions
        if head_on_collision:
            # Determine winner based on length (longer snake wins head-on)
            if self.snake1.length > self.snake2.length:
                collision2 = True
            elif self.snake2.length > self.snake1.length:
                collision1 = True
            else: # Equal length, both die
                collision1 = True
                collision2 = True
        else:
            if snake1_hit_snake2_body:
                collision1 = True
            if snake2_hit_snake1_body:
                collision2 = True

        # --- Calculate Rewards ---
        reward1 = 0.0
        reward2 = 0.0
        killed1 = False
        killed2 = False

        # Food Reward
        food_eaten_by_1 = False
        food_eaten_by_2 = False
        if self.foods:
            food_pos = self.foods[0].position
            if head1 == food_pos and not collision1: # Can only eat if not dead
                self.snake1.grow()
                reward1 += 5.0
                food_eaten_by_1 = True
                self.foods.pop(0) # Remove eaten food
                self._spawn_food() # Spawn new food
            # Check snake 2 only if snake 1 didn't eat the same food
            elif head2 == food_pos and not collision2:
                self.snake2.grow()
                reward2 += 5.0
                food_eaten_by_2 = True
                self.foods.pop(0)
                self._spawn_food()


        # Collision Penalties & Kill Rewards
        if collision1 and collision2: # Both died (e.g., head-on equal length, or simultaneous self/other collision)
            reward1 -= 10.0
            reward2 -= 10.0
            killed1 = True
            killed2 = True
        elif collision1: # Snake 1 died
            reward1 -= 10.0
            reward2 += 100.0 # Snake 2 gets kill reward
            killed1 = True
        elif collision2: # Snake 2 died
            reward2 -= 10.0
            reward1 += 100.0 # Snake 1 gets kill reward
            killed2 = True

        # Survival Bonus (only if alive)
        if not collision1:
            reward1 += 0.05  # Slightly reduced survival reward
        if not collision2:
            reward2 += 0.05  # Slightly reduced survival reward

        # --- Check Game End Conditions ---
        # Game ends if either snake dies or max steps reached
        self.done = collision1 or collision2 or self.steps >= MAX_STEPS_PER_EPISODE

        # --- Get Next States ---
        # If a snake died, its next state is None
        next_state1 = self._get_state(self.snake1, self.snake2) if not collision1 else None
        next_state2 = self._get_state(self.snake2, self.snake1) if not collision2 else None

        return (next_state1, next_state2), (reward1, reward2), self.done, killed1, killed2


# --- Self-Play Trainer with Opponent Pool ---
class SelfPlayTrainer:
    def __init__(self, state_size, action_size, agent_load_path=None, wandb_log=True):
        self.state_size = state_size
        self.action_size = action_size
        self.agent = DQNAgent(state_size, action_size)
        self.env = SnakeEnvironment()
        self.best_avg_score = -float('inf') # Track best average score over eval window
        self.scores_window = deque(maxlen=100) # For calculating running average score
        self.total_steps = 0

        # Opponent Pool
        self.opponent_pool = deque(maxlen=OPPONENT_POOL_SIZE)
        # Initialize pool with copies of the initial agent's state
        initial_state_dict = self.agent.policy_net.state_dict()
        for _ in range(OPPONENT_POOL_SIZE):
             self.opponent_pool.append(initial_state_dict.copy()) # Store state dicts

        # Load agent state if path provided
        if agent_load_path:
            self.agent.load_state(agent_load_path)
            # Potentially re-initialize opponent pool with the loaded agent state
            loaded_state_dict = self.agent.policy_net.state_dict()
            self.opponent_pool.clear()
            for _ in range(OPPONENT_POOL_SIZE):
                self.opponent_pool.append(loaded_state_dict.copy())


        # WandB Logging
        self.wandb_log = wandb_log
        if self.wandb_log:
            wandb.init(project="snake-ai-1v1-superhuman", config={
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "gamma": GAMMA,
                "eps_decay": EPS_DECAY,
                "target_update": TARGET_UPDATE,
                "memory_size": MEMORY_SIZE,
                "state_size": STATE_SIZE,
                "per_alpha": PER_ALPHA,
                "per_beta_start": PER_BETA_START,
                "opponent_pool_size": OPPONENT_POOL_SIZE,
                "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
                "grid_size": f"{GRID_WIDTH}x{GRID_HEIGHT}",
                "reward_kill": 100,
                "reward_food": 5,
                "reward_living": 0.1,
                "penalty_death": -10,
            })
            # Watch the model gradients (optional, can be resource intensive)
            # wandb.watch(self.agent.policy_net, log_freq=1000)


    def train(self, num_episodes=10000, eval_interval=100, save_interval=1000, checkpoint_dir="checkpoints"):
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Temporary opponent agent instance
        opponent_agent = DQNAgent(self.state_size, self.action_size)
        opponent_agent.epsilon = 0.0 # Opponent acts greedily based on its loaded weights

        for episode in tqdm(range(1, num_episodes + 1)):
            state1, state2 = self.env.reset()
            done = False
            episode_reward1 = 0
            episode_reward2 = 0
            episode_loss = 0
            steps_in_episode = 0

            # Select opponent from pool for this episode
            opponent_state_dict = random.choice(self.opponent_pool)
            opponent_agent.policy_net.load_state_dict(opponent_state_dict)
            opponent_agent.target_net.load_state_dict(opponent_state_dict) # Keep target net consistent
            opponent_agent.policy_net.eval() # Ensure opponent is in eval mode


            while not done:
                # Agent 1 (learning agent) action
                valid_actions1 = self.env.snake1.get_valid_actions()
                action1 = self.agent.act(state1, valid_actions1)

                # Agent 2 (opponent from pool) action
                valid_actions2 = self.env.snake2.get_valid_actions()
                # Opponent acts greedily (epsilon=0)
                action2 = opponent_agent.act(state2, valid_actions2) # Use opponent's act method

                # Environment step
                (next_state1, next_state2), (reward1, reward2), done, killed1, killed2 = self.env.step(action1, action2)

                # Remember experience for the learning agent (Agent 1)
                # Only store if the agent didn't die immediately in this step
                if state1 is not None: # Ensure state1 was valid at the start of the step
                     self.agent.remember(state1, action1, next_state1, reward1, done or killed1) # Mark done if agent died

                # Learn (if enough memory)
                loss = self.agent.learn()
                episode_loss += loss

                # Update states
                state1 = next_state1
                state2 = next_state2 # Need state2 for the opponent's next action

                episode_reward1 += reward1
                # episode_reward2 += reward2 # We primarily care about agent 1's reward
                steps_in_episode += 1
                self.total_steps += 1

                # Decay epsilon (based on total steps or agent's internal counter)
                self.agent.decay_epsilon()

                # Update PER beta based on total steps
                self.agent.memory.beta = min(1.0, PER_BETA_START + (1.0 - PER_BETA_START) * self.total_steps / PER_BETA_FRAMES)


            # --- End of Episode ---
            self.scores_window.append(self.env.snake1.score) # Track score of the learning agent

            # Update target network periodically
            if episode % TARGET_UPDATE == 0:
                self.agent.update_target_net()

            # Update opponent pool periodically
            if episode % UPDATE_OPPONENT_POOL_FREQ == 0:
                 self.opponent_pool.append(self.agent.policy_net.state_dict().copy()) # Add current agent's weights
                 print(f"\nUpdated opponent pool at episode {episode}. Pool size: {len(self.opponent_pool)}")


            # Logging and Saving
            if episode % eval_interval == 0:
                avg_score = np.mean(self.scores_window)
                avg_loss = episode_loss / max(1, steps_in_episode) # Avg loss per step in episode
                
                print(f"\nEpisode {episode}/{num_episodes} | Steps: {steps_in_episode} | Score: {self.env.snake1.score:.2f} | Avg Score (100 ep): {avg_score:.2f} | Loss: {avg_loss:.4f} | Epsilon: {self.agent.epsilon:.4f} | Beta: {self.agent.memory.beta:.4f}")

                if self.wandb_log:
                    wandb.log({
                        "episode": episode,
                        "score": self.env.snake1.score,
                        "average_score_100ep": avg_score,
                        "average_loss": avg_loss,
                        "epsilon": self.agent.epsilon,
                        "steps_in_episode": steps_in_episode,
                        "total_steps": self.total_steps,
                        "buffer_size": len(self.agent.memory),
                        "per_beta": self.agent.memory.beta,
                        "snake1_length": self.env.snake1.length,
                        "snake2_length": self.env.snake2.length, # Log opponent length too
                    })

                # Save best agent based on average score
                if avg_score > self.best_avg_score and len(self.scores_window) == 100:
                    self.best_avg_score = avg_score
                    best_filename = os.path.join(checkpoint_dir, f"best_agent_avg_score_{avg_score:.2f}.pt")
                    self.agent.save_state(best_filename)
                    if self.wandb_log:
                         wandb.save(best_filename) # Save best model to WandB

            # Save periodic checkpoint
            if episode % save_interval == 0:
                ckpt_filename = os.path.join(checkpoint_dir, f"checkpoint_ep_{episode}.pt")
                self.agent.save_state(ckpt_filename)
                if self.wandb_log:
                    # Save checkpoint to WandB (optional, can consume storage)
                    # wandb.save(ckpt_filename)
                    pass


        # --- End of Training ---
        final_filename = os.path.join(checkpoint_dir, "final_agent.pt")
        self.agent.save_state(final_filename)
        print(f"Training finished. Final agent saved to {final_filename}")
        if self.wandb_log:
            wandb.save(final_filename)
            wandb.finish()


# --- Testing and Visualization ---
def test_agent(agent_path, num_games=10, render=False, opponent_mode='agent'):
    """Test a trained agent against different opponents."""
    print(f"\n--- Testing Agent: {agent_path} ---")
    print(f"Opponent Mode: {opponent_mode}")

    # Load agent
    agent = DQNAgent(STATE_SIZE, 4) # Action size is 4
    agent.load_state(agent_path)
    agent.epsilon = 0.01 # Very low epsilon for testing, allows slight exploration if stuck
    agent.policy_net.eval()

    # Opponent setup
    opponent_agent = None
    if opponent_mode == 'agent':
        opponent_agent = DQNAgent(STATE_SIZE, 4)
        opponent_agent.load_state(agent_path) # Opponent uses the same weights
        opponent_agent.epsilon = 0.01
        opponent_agent.policy_net.eval()
    elif opponent_mode == 'self': # Load a potentially different agent as opponent
        # You might want to specify a different path for the opponent agent
        opponent_path = agent_path # Default to same agent for 'self' mode
        print(f"Loading opponent from: {opponent_path}")
        opponent_agent = DQNAgent(STATE_SIZE, 4)
        opponent_agent.load_state(opponent_path)
        opponent_agent.epsilon = 0.01
        opponent_agent.policy_net.eval()


    # Create environment
    env = SnakeEnvironment()

    # Statistics
    agent_scores = []
    opponent_scores = []
    game_lengths = []
    agent_wins = 0
    opponent_wins = 0
    draws = 0

    for game in range(num_games):
        state1, state2 = env.reset()
        done = False
        game_steps = 0

        while not done:
            # Agent 1 (the agent being tested) action
            valid_actions1 = env.snake1.get_valid_actions()
            action1 = agent.act(state1, valid_actions1)

            # Agent 2 (opponent) action
            valid_actions2 = env.snake2.get_valid_actions()
            if opponent_mode == 'random':
                action2 = random.choice(valid_actions2)
            elif opponent_mode in ['agent', 'self'] and opponent_agent:
                 action2 = opponent_agent.act(state2, valid_actions2)
            else: # Default to random if mode is unclear
                 action2 = random.choice(valid_actions2)


            # Take actions
            (next_state1, next_state2), (reward1, reward2), done, killed1, killed2 = env.step(action1, action2)
            game_steps += 1

            # Render game state if requested
            if render:
                render_game(env)
                time.sleep(0.05) # Slow down rendering

            # Update states
            state1 = next_state1
            state2 = next_state2

            if done:
                if killed1 and killed2:
                    draws += 1
                    print(f"Game {game+1}/{num_games} - Draw! Score: {env.snake1.score} vs {env.snake2.score}, Length: {game_steps}")
                elif killed1:
                    opponent_wins += 1
                    print(f"Game {game+1}/{num_games} - Opponent Wins! Score: {env.snake1.score} vs {env.snake2.score}, Length: {game_steps}")
                elif killed2:
                    agent_wins += 1
                    print(f"Game {game+1}/{num_games} - Agent Wins! Score: {env.snake1.score} vs {env.snake2.score}, Length: {game_steps}")
                else: # Game ended by steps limit
                    # Determine winner by score, or draw if equal
                    if env.snake1.score > env.snake2.score:
                         agent_wins += 1
                         print(f"Game {game+1}/{num_games} - Agent Wins (Timeout)! Score: {env.snake1.score} vs {env.snake2.score}, Length: {game_steps}")
                    elif env.snake2.score > env.snake1.score:
                         opponent_wins += 1
                         print(f"Game {game+1}/{num_games} - Opponent Wins (Timeout)! Score: {env.snake1.score} vs {env.snake2.score}, Length: {game_steps}")
                    else:
                         draws += 1
                         print(f"Game {game+1}/{num_games} - Draw (Timeout)! Score: {env.snake1.score} vs {env.snake2.score}, Length: {game_steps}")


        # Record statistics
        agent_scores.append(env.snake1.score)
        opponent_scores.append(env.snake2.score)
        game_lengths.append(game_steps)


    # Print summary
    print(f"\n--- Testing Summary ({num_games} games) ---")
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
    # Create grid representation
    grid = [['.' for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)] # Use '.' for empty

    # Add food
    for food in env.foods:
        x, y = food.position
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            grid[y][x] = 'F'

    # Add snake2 body (draw before head)
    if env.snake2:
        for i, pos in enumerate(list(env.snake2.positions)[1:]): # Iterate body segments
            x, y = pos
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                grid[y][x] = 'x'

    # Add snake1 body (draw before head)
    if env.snake1:
        for i, pos in enumerate(list(env.snake1.positions)[1:]): # Iterate body segments
            x, y = pos
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                grid[y][x] = 'o'

    # Add snake2 head
    if env.snake2:
        x, y = env.snake2.get_head_position()
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            grid[y][x] = '2'

    # Add snake1 head (draw last to overwrite if necessary)
    if env.snake1:
        x, y = env.snake1.get_head_position()
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            grid[y][x] = '1'


    # Print grid
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear console
    print(f"Step: {env.steps}/{MAX_STEPS_PER_EPISODE}")
    print(f"Score: Snake1 (Agent)={env.snake1.score if env.snake1 else 'N/A'} | Snake2 (Opponent)={env.snake2.score if env.snake2 else 'N/A'}")
    print(f"Length: S1={env.snake1.length if env.snake1 else 'N/A'} | S2={env.snake2.length if env.snake2 else 'N/A'}")
    print('+' + '-' * GRID_WIDTH + '+')
    for row in grid:
        print('|' + ''.join(row) + '|')
    print('+' + '-' * GRID_WIDTH + '+')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Snake AI with Dueling Double DQN, PER, and Self-Play')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes for training')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load a pre-trained model state for training continuation or testing')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save checkpoint every N episodes')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluate and log every N episodes')
    parser.add_argument('--checkpoint_dir', type=str, default='snake_checkpoints_v2', help='Directory to save checkpoints')
    parser.add_argument('--games', type=int, default=20, help='Number of games for testing')
    parser.add_argument('--render', action='store_true', help='Render game during testing')
    parser.add_argument('--opponent', type=str, default='agent', choices=['agent', 'random', 'self'], help='Opponent type during testing (agent=same weights, random, self=potentially different loaded agent)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')

    args = parser.parse_args()

    if args.mode == 'train':
        print("--- Starting Training ---")
        trainer = SelfPlayTrainer(STATE_SIZE, 4, # Action size is 4
                                  agent_load_path=args.load_model,
                                  wandb_log=not args.no_wandb)
        trainer.train(num_episodes=args.episodes,
                      eval_interval=args.eval_interval,
                      save_interval=args.save_interval,
                      checkpoint_dir=args.checkpoint_dir)

    elif args.mode == 'test':
        if not args.load_model:
            print("Error: Please provide a model path using --load_model for testing.")
            return
        print(f"--- Starting Testing ---")
        test_agent(args.load_model,
                   num_games=args.games,
                   render=args.render,
                   opponent_mode=args.opponent)

if __name__ == "__main__":
    main()
    # Example execution commands:
    # Train from scratch for 10k episodes:
    # python your_script_name.py --mode train --episodes 10000 --checkpoint_dir snake_models_10k
    
    # Continue training from a checkpoint:
    # python your_script_name.py --mode train --episodes 20000 --load_model snake_models_10k/checkpoint_ep_10000.pt --checkpoint_dir snake_models_20k

    # Test the final agent against itself with rendering:
    # python your_script_name.py --mode test --load_model snake_models_10k/final_agent.pt --games 10 --render --opponent agent
    
    # Test the best agent against random opponent:
    # python your_script_name.py --mode test --load_model snake_models_10k/best_agent_....pt --games 50 --opponent random