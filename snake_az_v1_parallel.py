# --- START OF FILE snake_az_v1_parallel.py ---

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from itertools import count
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Deque
import time
import math
from tqdm import tqdm
import wandb
import copy # For deep copying states/nodes if needed
import logging # Use logging instead of print for warnings

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True # Can slow down training

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- Game Constants ---
GRID_WIDTH = 10
GRID_HEIGHT = 10
INITIAL_SNAKE_LENGTH = 3
ACTIONS = [0, 1, 2, 3]  # 0:Up, 1:Right, 2:Down, 3:Left
NUM_ACTIONS = len(ACTIONS)
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # Corresponds to actions
OPPOSITE_ACTIONS = {0: 2, 1: 3, 2: 0, 3: 1}
MAX_SNAKE_LENGTH = GRID_WIDTH * GRID_HEIGHT

# --- AlphaZero Hyperparameters ---
# Parallelism
N_PARALLEL_GAMES = 64    # Number of games to run in parallel (adjust based on GPU memory)
MCTS_EVAL_BATCH_SIZE = 128 # Max number of states to evaluate in network batch during MCTS (can be > N_PARALLEL_GAMES)

# MCTS
N_SIMULATIONS = 50       # Number of MCTS simulation *steps* per move decision (adjust)
C_PUCT = 1.5
MCTS_TEMPERATURE_START = 1.0
MCTS_TEMPERATURE_END = 0.01
MCTS_TEMPERATURE_DECAY_STEPS = 50000 # Steps over which to anneal temperature
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

# Training
BATCH_SIZE = 256          # Training batch size (can be larger with parallel data gen)
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
MEMORY_SIZE = 100000     # Increased memory size
TRAIN_INTERVAL_GAMES = N_PARALLEL_GAMES * 4 # Train every N*k games generated
MIN_MEMORY_FOR_TRAINING = 10000 # Increased minimum memory
CHECKPOINT_INTERVAL_GAMES = N_PARALLEL_GAMES * 20 # Save model checkpoint every N*k games

# Network Architecture
NUM_RESIDUAL_BLOCKS = 5
NUM_FILTERS = 64

# Environment/Game
MAX_STEPS_PER_EPISODE = 200

# State Representation
STATE_CHANNELS = 5

# Reward Structure
REWARD_WIN = 1.0
REWARD_LOSS = -1.0
REWARD_DRAW = -1.0

# --- Data Structures ---
@dataclass
class GameStateData:
    state_tensor: np.ndarray
    mcts_policy: np.ndarray
    player: int
    outcome: Optional[float] = None

# Replay Buffer (Global)
replay_buffer: Deque[GameStateData] = deque(maxlen=MEMORY_SIZE)

# --- Food class (same as before) ---
@dataclass
class Food:
    position: Tuple[int, int]

# --- Snake class (same as before) ---
class Snake:
    def __init__(self, player_id, x, y, initial_direction=None):
        self.player_id = player_id # 1 or 2
        self.positions = deque([(x, y)])
        self.direction = initial_direction if initial_direction is not None else random.choice(DIRECTIONS)
        self.length = INITIAL_SNAKE_LENGTH
        self.score = 0 # Based on food eaten
        self.lifetime = 0
        # self.steps_since_food = 0
        # self.max_steps_without_food = GRID_WIDTH * GRID_HEIGHT * 1.5 # More generous?

        # Initialize body
        current_x, current_y = x, y
        for _ in range(1, INITIAL_SNAKE_LENGTH):
            dx, dy = self.direction
            prev_x = (current_x - dx + GRID_WIDTH) % GRID_WIDTH
            prev_y = (current_y - dy + GRID_HEIGHT) % GRID_HEIGHT
            if (prev_x, prev_y) not in self.positions:
                 self.positions.append((prev_x, prev_y))
                 current_x, current_y = prev_x, prev_y
            else:
                # Try other directions if blocked immediately
                found_extension = False
                for act in range(NUM_ACTIONS):
                    dx_alt, dy_alt = DIRECTIONS[act]
                    if (dx_alt, dy_alt) == (-dx, -dy): continue # Don't go back immediately
                    prev_x_alt = (current_x - dx_alt + GRID_WIDTH) % GRID_WIDTH
                    prev_y_alt = (current_y - dy_alt + GRID_HEIGHT) % GRID_HEIGHT
                    if (prev_x_alt, prev_y_alt) not in self.positions:
                        self.positions.append((prev_x_alt, prev_y_alt))
                        current_x, current_y = prev_x_alt, prev_y_alt
                        found_extension = True
                        break
                if not found_extension:
                    break # Stop trying to extend if truly blocked

    def get_head_position(self):
        return self.positions[0]

    def get_action_from_direction(self):
        try:
            return DIRECTIONS.index(self.direction)
        except ValueError:
            return 0 # Default

    def get_direction_from_action(self, action):
         return DIRECTIONS[action % len(DIRECTIONS)]

    def get_valid_actions(self) -> List[int]:
        current_action = self.get_action_from_direction()
        invalid_action = OPPOSITE_ACTIONS.get(current_action)

        head_x, head_y = self.get_head_position()
        valid = []
        for action in ACTIONS:
            # Prevent moving directly backward into the neck
            if action == invalid_action and len(self.positions) > 1:
                 continue
            valid.append(action)

        # If completely trapped (e.g., 1x1 space), allow moving back if length 1, else return current action
        if not valid:
            if len(self.positions) == 1 and invalid_action is not None:
                return [invalid_action]
            else: # Trapped, return current direction as only 'option' (will likely lead to death)
                return [current_action] if current_action is not None else [0] # Add failsafe default
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
        body_set = set(list(self.positions)[:-1]) # Check against body excluding tail

        if new_head in body_set:
            return True, 'self' # Collision occurred

        # Insert new head
        self.positions.appendleft(new_head)

        # Remove tail if snake didn't grow
        if len(self.positions) > self.length:
            self.positions.pop()

        # Update stats
        self.lifetime += 1
        #self.steps_since_food += 1

        # Check for starvation  NO!!!!!!
        #if self.steps_since_food >= self.max_steps_without_food:
        #    return True, 'starve'

        return False, None # No collision

    def grow(self):
        self.length = min(self.length + 1, MAX_SNAKE_LENGTH)
        self.score += 1
        # self.steps_since_food = 0

# --- Environment class (Mostly same, added helper for perspective) ---
class SnakeEnvironment:
    def __init__(self):
        self.snake1: Optional[Snake] = None
        self.snake2: Optional[Snake] = None
        self.foods: List[Food] = []
        self.steps = 0
        self.done = False
        self.game_id = random.randint(0, 1_000_000) # For tracking parallel games if needed

    def reset(self):
        self.game_id = random.randint(0, 1_000_000)
        # Create snakes at random non-overlapping positions
        while True:
            pos1 = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            dir1 = random.choice(DIRECTIONS)
            self.snake1 = Snake(player_id=1, x=pos1[0], y=pos1[1], initial_direction=dir1)

            pos2 = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            snake1_pos_set_init = set(self.snake1.positions)
            attempts = 0
            while pos2 in snake1_pos_set_init and attempts < GRID_WIDTH * GRID_HEIGHT:
                 pos2 = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
                 attempts += 1
            if attempts >= GRID_WIDTH * GRID_HEIGHT: # Avoid infinite loop if grid is small/full
                logging.warning("Could not find non-overlapping start for snake 2")
                # Handle this case, maybe reset again or place randomly anyway
                pos2 = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))


            dir2 = random.choice(DIRECTIONS)
            # Avoid immediate head-on collision if starting adjacent and opposite
            dx1, dy1 = dir1
            next_pos1 = ((pos1[0] + dx1) % GRID_WIDTH, (pos1[1] + dy1) % GRID_HEIGHT)
            if pos2 == next_pos1 and dir1 == (-dir2[0], -dir2[1]):
                 possible_dirs = [d for d in DIRECTIONS if d != (-dir1[0], -dir1[1])]
                 dir2 = random.choice(possible_dirs) if possible_dirs else dir1

            self.snake2 = Snake(player_id=2, x=pos2[0], y=pos2[1], initial_direction=dir2)

            # Final check for body overlap
            overlap = False
            snake1_pos_set = set(self.snake1.positions)
            for p2 in self.snake2.positions:
                if p2 in snake1_pos_set:
                    overlap = True; break
            if not overlap: break
            # If overlap, loop continues to try new positions

        self.foods = []
        self._spawn_food()
        self.steps = 0
        self.done = False

        # Return initial state tensor (from player 1's perspective)
        return self._get_state_tensor(player_perspective=1)

    def _spawn_food(self):
        while len(self.foods) < 1:
            occupied = set(self.snake1.positions) | set(self.snake2.positions) | {f.position for f in self.foods}
            if len(occupied) >= GRID_WIDTH * GRID_HEIGHT:
                logging.warning(f"Game {self.game_id}: No empty cells to spawn food!")
                self.done = True # End game if no space
                return # Exit the spawn attempt

            max_attempts = GRID_WIDTH * GRID_HEIGHT * 2
            for _ in range(max_attempts):
                 pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
                 if pos not in occupied:
                     self.foods.append(Food(pos))
                     return
            logging.warning(f"Game {self.game_id}: Failed to spawn food after {max_attempts} attempts.")
            self.done = True # End game if no space after many attempts
            break

    def _get_state_tensor(self, player_perspective: int) -> np.ndarray:
        tensor = np.zeros((STATE_CHANNELS, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
        s1 = self.snake1
        s2 = self.snake2

        if s1:
            head1 = s1.get_head_position()
            tensor[0, head1[1], head1[0]] = 1.0
            for pos in list(s1.positions)[1:]:
                tensor[1, pos[1], pos[0]] = 1.0
        if s2:
            head2 = s2.get_head_position()
            tensor[2, head2[1], head2[0]] = 1.0
            for pos in list(s2.positions)[1:]:
                tensor[3, pos[1], pos[0]] = 1.0

        for food in self.foods:
            tensor[4, food.position[1], food.position[0]] = 1.0

        if player_perspective == 2:
            tensor_copy = tensor.copy()
            tensor[0, :, :] = tensor_copy[2, :, :]
            tensor[1, :, :] = tensor_copy[3, :, :]
            tensor[2, :, :] = tensor_copy[0, :, :]
            tensor[3, :, :] = tensor_copy[1, :, :]
        return tensor

    def step(self, action1: int, action2: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            # Ensure snakes exist before accessing attributes
            s1_score = self.snake1.score if self.snake1 else 0
            s2_score = self.snake2.score if self.snake2 else 0
            # Return current state tensor, 0 reward, done=True
            return self._get_state_tensor(1), 0.0, True, {'outcome': 'draw', 's1_score': s1_score, 's2_score': s2_score}

        self.steps += 1

        # --- Move Snakes ---
        # Ensure actions are valid before moving (basic check)
        valid1 = self.get_valid_actions(1)
        valid2 = self.get_valid_actions(2)
        if action1 not in valid1: action1 = valid1[0] # Fallback to first valid action
        if action2 not in valid2: action2 = valid2[0] # Fallback

        collision1, reason1 = self.snake1.move(action1)
        collision2, reason2 = self.snake2.move(action2)

        head1 = self.snake1.get_head_position()
        head2 = self.snake2.get_head_position()
        # Get body sets *after* moving, excluding new heads
        snake1_body_set = set(list(self.snake1.positions)[1:])
        snake2_body_set = set(list(self.snake2.positions)[1:])

        # --- Check Inter-Snake Collisions ---
        head_on_collision = (head1 == head2)
        # Check collision with the *other* snake's body *after* they moved
        snake1_hit_snake2_body = head1 in snake2_body_set if not head_on_collision else False
        snake2_hit_snake1_body = head2 in snake1_body_set if not head_on_collision else False

        died1 = collision1
        died2 = collision2

        if head_on_collision:
            len1 = self.snake1.length
            len2 = self.snake2.length
            if len1 > len2: died2 = True; reason2 = 'head-on loss'
            elif len2 > len1: died1 = True; reason1 = 'head-on loss'
            else: died1 = True; reason1 = 'head-on draw'; died2 = True; reason2 = 'head-on draw'
        else:
            if snake1_hit_snake2_body: died1 = True; reason1 = 'hit other body'
            if snake2_hit_snake1_body: died2 = True; reason2 = 'hit other body'

        # --- Food Eating ---
        food_eaten_by_1 = False
        food_eaten_by_2 = False
        food_to_remove = -1
        for i, food in enumerate(self.foods):
            food_pos = food.position
            ate_this_food = False
            if head1 == food_pos and not died1:
                self.snake1.grow()
                food_eaten_by_1 = True
                ate_this_food = True
            # Check snake 2 only if snake 1 didn't eat it and snake 2 is alive
            if head2 == food_pos and not died2 and not ate_this_food:
                 self.snake2.grow()
                 food_eaten_by_2 = True
                 ate_this_food = True

            if ate_this_food:
                food_to_remove = i
                break # Only eat one food per step

        if food_to_remove != -1:
            self.foods.pop(food_to_remove)
            self._spawn_food() # Spawn new food

        # Check if spawning food failed and caused game end
        if self.done and not (died1 or died2):
             died1 = True; reason1 = 'no food space'
             died2 = True; reason2 = 'no food space'

        # --- Determine Outcome and Final Reward (z) ---
        self.done = died1 or died2 or self.steps >= MAX_STEPS_PER_EPISODE

        outcome = 'draw'
        final_reward_z = REWARD_DRAW

        if died1 and died2:
            outcome = 'draw'
            final_reward_z = REWARD_DRAW
        elif died1:
            outcome = 'loss'
            final_reward_z = REWARD_LOSS
        elif died2:
            outcome = 'win'
            final_reward_z = REWARD_WIN
        elif self.steps >= MAX_STEPS_PER_EPISODE:
             outcome = 'draw'
             final_reward_z = REWARD_DRAW
             # Optional: score comparison draw breaker
             if self.snake1.score > self.snake2.score: outcome = 'win'; final_reward_z = REWARD_WIN
             elif self.snake2.score > self.snake1.score: outcome = 'loss'; final_reward_z = REWARD_LOSS

        info = {
            'outcome': outcome, 'final_reward_z': final_reward_z,
            's1_score': self.snake1.score, 's2_score': self.snake2.score,
            's1_len': self.snake1.length, 's2_len': self.snake2.length,
            'steps': self.steps,
            'reason1': reason1 if died1 else None, 'reason2': reason2 if died2 else None,
        }

        next_state_tensor = self._get_state_tensor(player_perspective=1)
        return next_state_tensor, final_reward_z, self.done, info

    def get_valid_actions(self, player_id: int) -> List[int]:
        snake = self.snake1 if player_id == 1 else self.snake2
        if snake:
            return snake.get_valid_actions()
        return list(range(NUM_ACTIONS)) # Return all actions if snake somehow doesn't exist

    def get_current_player_perspective_state(self, player_id: int) -> np.ndarray:
        return self._get_state_tensor(player_perspective=player_id)

    def clone(self):
        cloned_env = SnakeEnvironment()
        cloned_env.snake1 = copy.deepcopy(self.snake1)
        cloned_env.snake2 = copy.deepcopy(self.snake2)
        cloned_env.foods = copy.deepcopy(self.foods)
        cloned_env.steps = self.steps
        cloned_env.done = self.done
        cloned_env.game_id = self.game_id # Keep same ID for cloned state during MCTS
        return cloned_env

# --- AlphaZero Network (same as before) ---
class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AlphaZeroNet(nn.Module):
    def __init__(self, input_channels=STATE_CHANNELS, num_filters=NUM_FILTERS, num_res_blocks=NUM_RESIDUAL_BLOCKS, num_actions=NUM_ACTIONS):
        super().__init__()
        self.conv_in = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        self.conv_policy = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        policy_flat_size = 2 * GRID_HEIGHT * GRID_WIDTH
        self.fc_policy = nn.Linear(policy_flat_size, num_actions)
        self.conv_value = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        value_flat_size = 1 * GRID_HEIGHT * GRID_WIDTH
        self.fc_value1 = nn.Linear(value_flat_size, 64)
        self.fc_value2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.fc_policy(policy)
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))
        return policy_logits, value

# --- MCTS Node (Modified slightly for batch processing) ---
class MCTSNode:
    def __init__(self, parent: Optional['MCTSNode'], prior_p: float, action_taken: Optional[int] = None):
        self.parent = parent
        self.children: Dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.total_action_value = 0.0 # W
        self.mean_action_value = 0.0  # Q
        self.prior_p = prior_p        # P
        self.action_taken = action_taken # Action that led to this node

        # For parallel processing
        self.is_expanded = False
        self.needs_evaluation = False # Flag if this leaf needs NN evaluation
        self.virtual_loss = 0 # To discourage selecting same path in parallel sims

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        best_score = -float('inf')
        best_action = -1
        best_child = None

        sqrt_total_visits = math.sqrt(self.visit_count + self.virtual_loss) # Include virtual loss

        for action, child in self.children.items():
            q_value = child.mean_action_value
            # Apply virtual loss to Q for selection
            q_value_adjusted = (child.total_action_value - child.virtual_loss) / (child.visit_count + 1e-9)

            u_value = (c_puct * child.prior_p * sqrt_total_visits / (1 + child.visit_count + child.virtual_loss))
            score = q_value_adjusted + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
             logging.error(f"MCTS selection failed: Node has no children or issue with scores. Parent visits: {self.visit_count}, Children: {list(self.children.keys())}")
             # Fallback: Choose child with max prior? Or random valid?
             if self.children:
                 best_action = max(self.children.keys(), key=lambda a: self.children[a].prior_p)
                 best_child = self.children[best_action]
                 logging.warning(f"Falling back to child with max prior: Action {best_action}")
                 return best_action, best_child
             else:
                 raise RuntimeError("MCTS selection failed: Node has no children.")

        return best_action, best_child

    def expand(self, policy_probs: np.ndarray, valid_actions: List[int]):
        self.is_expanded = True
        self.needs_evaluation = False # No longer needs evaluation after expansion
        for action in valid_actions:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior_p=policy_probs[action], action_taken=action)

    def add_virtual_loss(self, loss_count=1):
        """Temporarily penalize this node during selection phase of parallel runs."""
        self.virtual_loss += loss_count
        if self.parent:
            self.parent.add_virtual_loss(loss_count)

    def remove_virtual_loss(self, loss_count=1):
        """Remove penalty after backup."""
        self.virtual_loss -= loss_count
        if self.parent:
            self.parent.remove_virtual_loss(loss_count)

    def update_recursive(self, value: float):
        """Update node statistics recursively up the tree."""
        if self.parent:
            # Value should be from the perspective of the player *at the parent node*
            # Since value is from the child's perspective, flip it for the parent.
            self.parent.update_recursive(-value)

        self.visit_count += 1
        self.total_action_value += value
        self.mean_action_value = self.total_action_value / self.visit_count

    def get_action_probs(self, temperature: float) -> np.ndarray:
        """Calculates action probabilities based on visit counts, adjusted by temperature."""
        probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if not self.children:
            return probs # Should not happen if expanded

        child_visits = np.array([self.children[a].visit_count if a in self.children else 0
                                 for a in range(NUM_ACTIONS)], dtype=np.float32)

        if temperature < 1e-6: # Exploit: Choose the best action deterministically
            best_action = np.argmax(child_visits)
            probs[best_action] = 1.0
        else:
            # Explore: Sample action proportionally to N^(1/T)
            adjusted_visits = child_visits ** (1.0 / temperature)
            total_adjusted_visits = np.sum(adjusted_visits)
            if total_adjusted_visits < 1e-6:
                # Fallback to uniform over children if all adjusted visits are zero
                num_children = len(self.children)
                if num_children > 0:
                    valid_child_actions = list(self.children.keys())
                    probs[valid_child_actions] = 1.0 / num_children
                else: # Should be impossible if called after simulations
                    probs[:] = 1.0 / NUM_ACTIONS # Failsafe uniform
            else:
                probs = adjusted_visits / total_adjusted_visits

        # Ensure probabilities sum to 1 (handle potential float inaccuracies)
        probs /= np.sum(probs)
        return probs


# --- Parallel Self-Play Manager ---
@dataclass
class MCTSRequest:
    game_idx: int
    player_id: int
    node: MCTSNode
    sim_env: SnakeEnvironment # Cloned environment state for this simulation path
    search_path: List[MCTSNode]

class ParallelSelfPlayManager:
    def __init__(self, network: AlphaZeroNet, num_parallel_games: int, c_puct: float, n_simulations: int):
        self.network = network
        self.num_parallel_games = num_parallel_games
        self.c_puct = c_puct
        self.n_simulations_per_move = n_simulations # Target simulations per move decision

        self.envs = [SnakeEnvironment() for _ in range(num_parallel_games)]
        self.mcts_roots_p1: List[Optional[MCTSNode]] = [None] * num_parallel_games
        self.mcts_roots_p2: List[Optional[MCTSNode]] = [None] * num_parallel_games
        self.game_histories: List[List[GameStateData]] = [[] for _ in range(num_parallel_games)]
        self.simulations_done_count: List[Tuple[int, int]] = [(0, 0)] * num_parallel_games # (p1_sims, p2_sims)

        self.pending_evaluations: List[MCTSRequest] = [] # Nodes waiting for NN evaluation
        self.active_simulations: List[MCTSRequest] = [] # Simulations currently in selection phase

        self._reset_all()

    def _reset_all(self):
        logging.info("Resetting all parallel environments.")
        for i in range(self.num_parallel_games):
            self.envs[i].reset()
            self.mcts_roots_p1[i] = MCTSNode(parent=None, prior_p=1.0)
            self.mcts_roots_p2[i] = MCTSNode(parent=None, prior_p=1.0)
            self.game_histories[i] = []
            self.simulations_done_count[i] = (0, 0)
        self.pending_evaluations = []
        self.active_simulations = []
        # Initial root evaluation needed
        self._init_roots()

    def _init_roots(self):
        """Evaluate initial root nodes for all games and players."""
        states_p1 = []
        states_p2 = []
        valid_actions_p1 = []
        valid_actions_p2 = []
        indices_p1 = []
        indices_p2 = []

        for i in range(self.num_parallel_games):
            if not self.envs[i].done:
                states_p1.append(self.envs[i].get_current_player_perspective_state(1))
                valid_actions_p1.append(self.envs[i].get_valid_actions(1))
                indices_p1.append(i)

                states_p2.append(self.envs[i].get_current_player_perspective_state(2))
                valid_actions_p2.append(self.envs[i].get_valid_actions(2))
                indices_p2.append(i)

        if not indices_p1 and not indices_p2: return # All games might be done initially (unlikely)

        all_states = states_p1 + states_p2
        all_indices = indices_p1 + indices_p2
        all_players = [1] * len(indices_p1) + [2] * len(indices_p2)
        all_valid_actions = valid_actions_p1 + valid_actions_p2

        if not all_states: return

        state_batch = torch.from_numpy(np.array(all_states)).to(device)
        with torch.no_grad():
            policy_logits_batch, value_batch = self.network(state_batch)
            policy_probs_batch = F.softmax(policy_logits_batch, dim=1).cpu().numpy()
            value_batch = value_batch.cpu().numpy()

        for k in range(len(all_indices)):
            game_idx = all_indices[k]
            player_id = all_players[k]
            root = self.mcts_roots_p1[game_idx] if player_id == 1 else self.mcts_roots_p2[game_idx]
            policy_probs = policy_probs_batch[k]
            value = value_batch[k][0]
            valid_actions = all_valid_actions[k]

            # Apply Dirichlet noise at the root
            if valid_actions:
                noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(valid_actions))
                noisy_policy = (1 - DIRICHLET_EPSILON) * policy_probs[valid_actions] + DIRICHLET_EPSILON * noise
                policy_probs_masked = np.zeros_like(policy_probs)
                policy_probs_masked[valid_actions] = noisy_policy
                 # Renormalize
                policy_sum = np.sum(policy_probs_masked)
                if policy_sum > 1e-6: policy_probs_masked /= policy_sum
                else: policy_probs_masked[valid_actions] = 1.0 / len(valid_actions) # Uniform fallback
            else:
                policy_probs_masked = np.zeros_like(policy_probs) # No valid actions

            root.expand(policy_probs_masked, valid_actions)
            # Initial value estimate isn't directly used for root update in AZ MCTS


    def run_simulation_step(self):
        """Performs one step of MCTS for a batch of games: Selection -> Evaluation -> Backup."""

        # --- 1. Start New Simulations (Selection Phase) ---
        num_to_start = MCTS_EVAL_BATCH_SIZE - len(self.pending_evaluations) # How many new sims can we start?
        new_requests = []

        # Prioritize games/players with fewer completed simulations
        sim_counts = [(i, p, self.simulations_done_count[i][p-1])
                      for i in range(self.num_parallel_games)
                      for p in [1, 2] if not self.envs[i].done]
        sim_counts.sort(key=lambda x: x[2]) # Sort by number of sims done

        started_count = 0
        for game_idx, player_id, _ in sim_counts:
            if started_count >= num_to_start: break
            if self.envs[game_idx].done: continue # Skip finished games

            root = self.mcts_roots_p1[game_idx] if player_id == 1 else self.mcts_roots_p2[game_idx]
            if not root.is_expanded: # Skip if root wasn't expanded (e.g., no valid actions)
                 continue

            sim_env = self.envs[game_idx].clone() # Clone env for this simulation path
            node = root
            search_path = [node]
            node.add_virtual_loss() # Add virtual loss for selection

            # Selection loop
            while node.is_expanded and not sim_env.done:
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
                node.add_virtual_loss() # Add virtual loss

                # --- Simulate opponent's move (Heuristic: use opponent's MCTS root policy) ---
                opponent_player = 3 - player_id
                opp_root = self.mcts_roots_p1[game_idx] if opponent_player == 1 else self.mcts_roots_p2[game_idx]
                opp_valid_actions = sim_env.get_valid_actions(opponent_player)

                if not opp_valid_actions: # Opponent trapped
                    opponent_action = 0 # Default action
                elif not opp_root.is_expanded: # Opponent root not ready (should not happen after init)
                    opponent_action = random.choice(opp_valid_actions) # Random valid action
                else:
                    # Use opponent's current MCTS policy (visit counts) without temperature
                    opp_probs = opp_root.get_action_probs(temperature=0.0)
                    masked_opp_probs = np.zeros_like(opp_probs)
                    masked_opp_probs[opp_valid_actions] = opp_probs[opp_valid_actions]
                    prob_sum = np.sum(masked_opp_probs)
                    if prob_sum > 1e-6: masked_opp_probs /= prob_sum
                    else: masked_opp_probs[opp_valid_actions] = 1.0 / len(opp_valid_actions)

                    if np.all(masked_opp_probs == 0): # Failsafe if all valid actions have 0 prob
                        opponent_action = random.choice(opp_valid_actions)
                    else:
                        opponent_action = np.random.choice(NUM_ACTIONS, p=masked_opp_probs)


                action1 = action if player_id == 1 else opponent_action
                action2 = action if player_id == 2 else opponent_action

                _, _, sim_done, _ = sim_env.step(action1, action2)
                if sim_done: break # Reached terminal state during selection

            # Add the leaf node to pending evaluations or handle terminal state
            request = MCTSRequest(game_idx=game_idx, player_id=player_id, node=node, sim_env=sim_env, search_path=search_path)
            if sim_env.done:
                # If simulation ended, backpropagate immediately
                self._backup_value(request, terminal=True)
            else:
                # Mark leaf for evaluation and add to pending list
                node.needs_evaluation = True
                self.pending_evaluations.append(request)

            started_count += 1


        # --- 2. Batch Evaluation ---
        if not self.pending_evaluations:
            return # Nothing to evaluate

        eval_batch = self.pending_evaluations[:MCTS_EVAL_BATCH_SIZE]
        self.pending_evaluations = self.pending_evaluations[MCTS_EVAL_BATCH_SIZE:]

        states_to_eval = []
        valid_actions_list = []
        player_perspectives = [] # Player whose perspective the state is from
        for request in eval_batch:
            # State tensor must be from the perspective of the player whose turn it is *at the leaf*
            # In our simulation structure, the 'player_id' in the request is the one we are finding a move for.
            # The leaf node represents a state *after* that player (and opponent) potentially moved.
            # The value network predicts the outcome from the perspective of the player *whose turn it is*.
            # Since we simulate both moves, the 'current player' concept is tricky.
            # Let's evaluate from the perspective of the player associated with the MCTS tree (request.player_id).
            states_to_eval.append(request.sim_env.get_current_player_perspective_state(request.player_id))
            valid_actions_list.append(request.sim_env.get_valid_actions(request.player_id))
            player_perspectives.append(request.player_id)


        state_batch = torch.from_numpy(np.array(states_to_eval)).to(device)
        with torch.no_grad():
            policy_logits_batch, value_batch = self.network(state_batch)
            policy_probs_batch = F.softmax(policy_logits_batch, dim=1).cpu().numpy()
            value_batch = value_batch.cpu().numpy()

        # --- 3. Expansion & Backup ---
        for i, request in enumerate(eval_batch):
            leaf_node = request.node
            policy_probs = policy_probs_batch[i]
            value = value_batch[i][0] # Network value prediction (from request.player_id perspective)
            valid_actions = valid_actions_list[i]

            # Mask invalid actions in policy
            masked_policy = np.zeros_like(policy_probs)
            if valid_actions:
                masked_policy[valid_actions] = policy_probs[valid_actions]
                policy_sum = np.sum(masked_policy)
                if policy_sum > 1e-6: masked_policy /= policy_sum
                else: masked_policy[valid_actions] = 1.0 / len(valid_actions)

            # Expand the leaf node
            if valid_actions:
                 leaf_node.expand(masked_policy, valid_actions)
            else: # No valid actions from this state
                 leaf_node.is_expanded = True # Mark as expanded but has no children
                 leaf_node.needs_evaluation = False

            # Backup the evaluated value
            self._backup_value(request, value=value, terminal=False)


    def _backup_value(self, request: MCTSRequest, value: Optional[float] = None, terminal: bool = False):
        """Backpropagate value up the search path and remove virtual loss."""
        if terminal:
            # Get final reward (z) from the simulated env info
            # Need to call step again (with dummy actions) to get info dict cleanly
            _, _, _, sim_info = request.sim_env.step(0, 0)
            final_reward_z_p1 = sim_info.get('final_reward_z', 0.0)
            # Value must be from the perspective of the player whose turn it was *at the leaf node*
            # This is request.player_id
            backup_value = final_reward_z_p1 if request.player_id == 1 else -final_reward_z_p1
        else:
            # Use the network's predicted value
            backup_value = value if value is not None else 0.0

        # Update nodes recursively, flipping value sign for parent
        request.node.update_recursive(backup_value)

        # Remove virtual loss along the path
        for node in request.search_path:
            node.remove_virtual_loss()

        # Increment simulation count for the root player
        p1_sims, p2_sims = self.simulations_done_count[request.game_idx]
        if request.player_id == 1: p1_sims += 1
        else: p2_sims += 1
        self.simulations_done_count[request.game_idx] = (p1_sims, p2_sims)


    def get_actions_and_store_data(self, temperature: float) -> bool:
        """
        Checks if enough simulations are done for all games.
        If yes, selects actions, stores data, steps environments, resets finished ones,
        and returns True. Otherwise returns False.
        """
        ready_to_move = True
        for i in range(self.num_parallel_games):
            if not self.envs[i].done:
                p1_sims, p2_sims = self.simulations_done_count[i]
                if p1_sims < self.n_simulations_per_move or p2_sims < self.n_simulations_per_move:
                    ready_to_move = False
                    break
        if not ready_to_move:
            return False

        # --- All games ready: Select Actions, Store Data, Step ---
        actions1 = np.zeros(self.num_parallel_games, dtype=int)
        actions2 = np.zeros(self.num_parallel_games, dtype=int)
        newly_finished_game_indices = []
        global replay_buffer # Allow modification of global buffer

        for i in range(self.num_parallel_games):
            if self.envs[i].done: continue # Skip already finished games

            # Get MCTS policies
            mcts_policy1 = self.mcts_roots_p1[i].get_action_probs(temperature)
            mcts_policy2 = self.mcts_roots_p2[i].get_action_probs(temperature)

            # Store data for buffer (state from player's perspective, MCTS policy)
            state1_persp = self.envs[i].get_current_player_perspective_state(1)
            self.game_histories[i].append(GameStateData(state_tensor=state1_persp, mcts_policy=mcts_policy1, player=1))
            state2_persp = self.envs[i].get_current_player_perspective_state(2)
            self.game_histories[i].append(GameStateData(state_tensor=state2_persp, mcts_policy=mcts_policy2, player=2))

            # Select actions (usually deterministically/greedily after sufficient search in AZ)
            # Use temperature 0 for action selection after search, temp was for exploration *during* search policy generation
            action1 = np.argmax(self.mcts_roots_p1[i].get_action_probs(temperature=0.0))
            action2 = np.argmax(self.mcts_roots_p2[i].get_action_probs(temperature=0.0))

            # Fallback if argmax gives invalid action (shouldn't happen with proper masking/probs)
            valid1 = self.envs[i].get_valid_actions(1)
            valid2 = self.envs[i].get_valid_actions(2)
            if action1 not in valid1: action1 = valid1[0] if valid1 else 0
            if action2 not in valid2: action2 = valid2[0] if valid2 else 0

            actions1[i] = action1
            actions2[i] = action2

        # --- Step all environments ---
        infos = []
        for i in range(self.num_parallel_games):
             if self.envs[i].done:
                 infos.append(None) # Placeholder for already done games
                 continue
             _, _, done, info = self.envs[i].step(actions1[i], actions2[i])
             infos.append(info)
             if done:
                 newly_finished_game_indices.append(i)

        # --- Process finished games ---
        for i in newly_finished_game_indices:
            final_outcome_z = infos[i].get('final_reward_z', 0.0)
            # Assign outcome to history and add to replay buffer
            for data in self.game_histories[i]:
                data.outcome = final_outcome_z if data.player == 1 else -final_outcome_z
                replay_buffer.append(data)

            # Log game end stats
            if wandb.run is not None: # Check if wandb is initialized
                 try:
                     wandb.log({
                         f"game_end/outcome_p1": infos[i]['outcome'],
                         f"game_end/final_reward_z_p1": final_outcome_z,
                         f"game_end/score_p1": infos[i]['s1_score'],
                         f"game_end/score_p2": infos[i]['s2_score'],
                         f"game_end/length_p1": infos[i]['s1_len'],
                         f"game_end/length_p2": infos[i]['s2_len'],
                         f"game_end/steps": infos[i]['steps'],
                     }, commit=False) # Commit later in main loop
                 except Exception as e:
                     logging.warning(f"WandB game end logging error: {e}")


            # Reset environment and MCTS roots
            self.envs[i].reset()
            self.mcts_roots_p1[i] = MCTSNode(parent=None, prior_p=1.0)
            self.mcts_roots_p2[i] = MCTSNode(parent=None, prior_p=1.0)
            self.game_histories[i] = []
            self.simulations_done_count[i] = (0, 0)

        # --- Re-initialize roots for newly reset games ---
        self._init_roots() # This will evaluate roots only for non-done games

        # Reset simulation counts for games that just moved but didn't finish
        for i in range(self.num_parallel_games):
            if not self.envs[i].done and i not in newly_finished_game_indices:
                 self.simulations_done_count[i] = (0, 0)


        return True # Actions were taken

# --- Training Loop (Mostly same, uses global replay_buffer) ---
def train_step(network, optimizer, replay_buffer, batch_size):
    if len(replay_buffer) < MIN_MEMORY_FOR_TRAINING or len(replay_buffer) < batch_size:
        return 0.0, 0.0, 0.0 # Not enough data yet

    batch_data = random.sample(list(replay_buffer), batch_size)
    state_tensors = np.array([d.state_tensor for d in batch_data])
    mcts_policies = np.array([d.mcts_policy for d in batch_data])
    outcomes = np.array([d.outcome for d in batch_data]).reshape(-1, 1)

    state_batch = torch.from_numpy(state_tensors).to(device)
    mcts_policy_batch = torch.from_numpy(mcts_policies).to(device)
    outcome_batch = torch.from_numpy(outcomes).float().to(device)

    network.train()
    policy_logits, value_pred = network(state_batch)

    value_loss = F.mse_loss(value_pred, outcome_batch)
    policy_loss = F.cross_entropy(policy_logits, mcts_policy_batch)
    total_loss = value_loss + policy_loss

    optimizer.zero_grad()
    total_loss.backward()
    # Optional: Gradient clipping
    # torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), policy_loss.item(), value_loss.item()

# --- Main Function ---
def main_alpha_zero_parallel(num_total_games_target=100000, load_path=None, save_dir="az_checkpoints_v1_parallel", wandb_log=True):

    os.makedirs(save_dir, exist_ok=True)

    network = AlphaZeroNet().to(device)
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    start_game_idx = 0
    total_steps_simulated = 0 # Track total simulation *steps* across all games
    games_generated_count = 0

    if load_path and os.path.exists(load_path):
        try:
            checkpoint = torch.load(load_path, map_location=device)
            network.load_state_dict(checkpoint['network_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            games_generated_count = checkpoint.get('games_generated_count', 0)
            total_steps_simulated = checkpoint.get('total_steps_simulated', 0)
            # Load replay buffer if saved (optional)
            if 'replay_buffer' in checkpoint and checkpoint['replay_buffer'] is not None:
                 global replay_buffer
                 replay_buffer = checkpoint['replay_buffer']
                 logging.info(f"Loaded replay buffer with {len(replay_buffer)} elements.")
            else:
                 logging.info("Replay buffer not found in checkpoint or was None.")

            logging.info(f"Loaded checkpoint from {load_path}. Resuming from game count {games_generated_count}.")
            start_game_idx = games_generated_count # Continue counting
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_game_idx = 0
            total_steps_simulated = 0
            games_generated_count = 0
    else:
        logging.info("Starting training from scratch.")

    if wandb_log:
        try:
            wandb.init(project="snake-ai-alphazero-v1-parallel", config={
                "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
                "n_simulations": N_SIMULATIONS, "c_puct": C_PUCT,
                "memory_size": MEMORY_SIZE, "grid_size": f"{GRID_WIDTH}x{GRID_HEIGHT}",
                "num_res_blocks": NUM_RESIDUAL_BLOCKS, "num_filters": NUM_FILTERS,
                "max_steps_per_episode": MAX_STEPS_PER_EPISODE, "seed": SEED,
                "device": str(device), "weight_decay": WEIGHT_DECAY,
                "mcts_temp_start": MCTS_TEMPERATURE_START, "mcts_temp_end": MCTS_TEMPERATURE_END,
                "mcts_temp_decay": MCTS_TEMPERATURE_DECAY_STEPS,
                "dirichlet_alpha": DIRICHLET_ALPHA, "dirichlet_epsilon": DIRICHLET_EPSILON,
                "n_parallel_games": N_PARALLEL_GAMES, "mcts_eval_batch_size": MCTS_EVAL_BATCH_SIZE,
                "train_interval_games": TRAIN_INTERVAL_GAMES, "min_memory_for_training": MIN_MEMORY_FOR_TRAINING,
            })
            wandb.watch(network, log_freq=1000)
        except Exception as e:
             logging.error(f"Error initializing WandB: {e}. Disabling logging.")
             wandb_log = False

    # --- Initialize Parallel Manager ---
    self_play_manager = ParallelSelfPlayManager(network, N_PARALLEL_GAMES, C_PUCT, N_SIMULATIONS)

    # --- Main Loop ---
    pbar = tqdm(total=num_total_games_target, initial=games_generated_count, desc="Self-Play Games Generated")
    last_train_game_count = games_generated_count
    last_checkpoint_game_count = games_generated_count

    while games_generated_count < num_total_games_target:
        network.eval() # Ensure network is in eval mode for self-play

        # --- Run MCTS Simulation Step ---
        start_sim_step_time = time.time()
        self_play_manager.run_simulation_step()
        sim_step_duration = time.time() - start_sim_step_time
        total_steps_simulated += len(self_play_manager.pending_evaluations) # Rough estimate of sims performed

        # --- Check if Ready to Select Actions and Step Environments ---
        start_action_step_time = time.time()
        temperature = MCTS_TEMPERATURE_START # Fixed temperature for now, decay is complex with parallel
        # temperature = get_mcts_temperature(total_steps_simulated) # Could use total steps
        actions_taken = self_play_manager.get_actions_and_store_data(temperature)
        action_step_duration = time.time() - start_action_step_time

        if actions_taken:
            # Update game count based on how many games finished and were reset
            current_game_count = len(replay_buffer) // (MAX_STEPS_PER_EPISODE * 2) # Estimate based on buffer size
            # A better way: count resets in get_actions_and_store_data
            # Let's increment based on the number of parallel games completing a cycle
            games_generated_count += N_PARALLEL_GAMES # Increment by batch size when actions are taken
            pbar.update(N_PARALLEL_GAMES)

            # --- Train Network Periodically ---
            if games_generated_count - last_train_game_count >= TRAIN_INTERVAL_GAMES:
                if len(replay_buffer) >= MIN_MEMORY_FOR_TRAINING:
                    logging.info(f"Starting training at {games_generated_count} games generated...")
                    start_train_time = time.time()
                    network.train()
                    num_train_steps = (games_generated_count - last_train_game_count) * 2 * MAX_STEPS_PER_EPISODE // BATCH_SIZE # Heuristic
                    num_train_steps = max(10, min(num_train_steps, 500)) # Clamp training steps
                    avg_loss, avg_p_loss, avg_v_loss = 0.0, 0.0, 0.0
                    for _ in range(num_train_steps):
                         loss, p_loss, v_loss = train_step(network, optimizer, replay_buffer, BATCH_SIZE)
                         avg_loss += loss
                         avg_p_loss += p_loss
                         avg_v_loss += v_loss
                    avg_loss /= num_train_steps
                    avg_p_loss /= num_train_steps
                    avg_v_loss /= num_train_steps
                    network.eval()
                    train_duration = time.time() - start_train_time
                    logging.info(f"Training finished in {train_duration:.2f}s. Avg Loss: {avg_loss:.4f}")
                    if wandb_log:
                        try:
                            wandb.log({
                                "train/total_loss": avg_loss,
                                "train/policy_loss": avg_p_loss,
                                "train/value_loss": avg_v_loss,
                                "train/num_train_steps": num_train_steps,
                                "perf/train_duration_sec": train_duration,
                                "progress/games_generated": games_generated_count,
                                "progress/buffer_size": len(replay_buffer),
                                "progress/total_steps_simulated": total_steps_simulated,
                            }, commit=True) # Commit all logs now
                        except Exception as e: logging.warning(f"WandB training log error: {e}")

                    last_train_game_count = games_generated_count
                else:
                     logging.info(f"Skipping training at {games_generated_count} games, buffer size {len(replay_buffer)} < {MIN_MEMORY_FOR_TRAINING}")
                     if wandb_log: wandb.log({}, commit=True) # Commit other logs even if skipping training

            # --- Save Checkpoint Periodically ---
            if games_generated_count - last_checkpoint_game_count >= CHECKPOINT_INTERVAL_GAMES:
                ckpt_path = os.path.join(save_dir, f"checkpoint_games_{games_generated_count}.pt")
                try:
                    # Save buffer only occasionally or if small enough
                    save_buffer = len(replay_buffer) < MEMORY_SIZE // 2 # Example condition
                    buffer_to_save = replay_buffer if save_buffer else None
                    torch.save({
                        'games_generated_count': games_generated_count,
                        'network_state_dict': network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'total_steps_simulated': total_steps_simulated,
                        'replay_buffer': buffer_to_save,
                    }, ckpt_path)
                    logging.info(f"Checkpoint saved to {ckpt_path}")
                    if wandb_log:
                         try: wandb.save(ckpt_path, base_path=save_dir, policy="now")
                         except Exception as e: logging.warning(f"WandB save error: {e}")
                    last_checkpoint_game_count = games_generated_count
                except Exception as e:
                    logging.error(f"Failed to save checkpoint: {e}")

            # Update progress bar description
            pbar.set_description(f"Gen Games: {games_generated_count} | Buf: {len(replay_buffer)} | Sim Step: {sim_step_duration:.3f}s | Act Step: {action_step_duration:.3f}s")


    # --- End of Training ---
    pbar.close()
    final_path = os.path.join(save_dir, f"final_model_games_{games_generated_count}.pt")
    torch.save({
        'games_generated_count': games_generated_count,
        'network_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_steps_simulated': total_steps_simulated,
    }, final_path)
    logging.info(f"Training finished. Final model saved to {final_path}")
    if wandb_log:
        try:
            wandb.save(final_path, base_path=save_dir, policy="now")
            wandb.finish()
        except Exception as e: logging.warning(f"WandB final save error: {e}")


# --- Testing Function (Needs Adaptation for AZ - Use original sequential MCTS for simplicity) ---
# Re-use the original run_mcts for testing, as parallel manager is complex overhead for single game eval
def run_mcts_sequential(env_state: SnakeEnvironment, current_player: int, network: AlphaZeroNet, n_simulations: int, c_puct: float):
    """ Original sequential MCTS implementation for testing/evaluation. """
    root_node = MCTSNode(parent=None, prior_p=1.0)
    initial_state_tensor = env_state.get_current_player_perspective_state(current_player)
    initial_valid_actions = env_state.get_valid_actions(current_player)

    if not initial_valid_actions: # Handle case where player is already trapped
        action_probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if initial_valid_actions: # Should be empty list here, but check anyway
             action_probs[initial_valid_actions] = 1.0 / len(initial_valid_actions)
        else: # Truly no actions
             action_probs[0] = 1.0 # Default to action 0
        return action_probs


    with torch.no_grad():
        state_torch = torch.from_numpy(initial_state_tensor).unsqueeze(0).to(device)
        policy_logits, root_value = network(state_torch)
        policy_probs_torch = F.softmax(policy_logits, dim=1).squeeze(0).cpu()
        masked_policy = torch.zeros_like(policy_probs_torch)
        masked_policy[initial_valid_actions] = policy_probs_torch[initial_valid_actions]
        policy_sum = masked_policy.sum()
        if policy_sum > 1e-6: masked_policy /= policy_sum
        else: masked_policy[initial_valid_actions] = 1.0 / len(initial_valid_actions)
        policy_probs_np = masked_policy.numpy()
        root_value = root_value.item()

    # NO Dirichlet noise during testing
    root_node.expand(policy_probs_np, initial_valid_actions)

    for _ in range(n_simulations):
        node = root_node
        sim_env = env_state.clone()
        search_path = [node]
        player = current_player

        while node.is_expanded:
            action, node = node.select_child(c_puct) # Use original select_child without virtual loss
            search_path.append(node)

            # Simulate opponent's move (Heuristic: greedy policy from network)
            opponent = 3 - player
            opponent_state_tensor = sim_env.get_current_player_perspective_state(opponent)
            opponent_valid_actions = sim_env.get_valid_actions(opponent)

            if not opponent_valid_actions:
                opponent_action = 0
            else:
                with torch.no_grad():
                    opp_state_torch = torch.from_numpy(opponent_state_tensor).unsqueeze(0).to(device)
                    opp_policy_logits, _ = network(opp_state_torch)
                    opp_policy_probs = F.softmax(opp_policy_logits, dim=1).squeeze(0).cpu().numpy()
                    best_opp_action_score = -1
                    opponent_action = opponent_valid_actions[0]
                    for opp_act in opponent_valid_actions:
                        if opp_policy_probs[opp_act] > best_opp_action_score:
                            best_opp_action_score = opp_policy_probs[opp_act]
                            opponent_action = opp_act

            action1 = action if player == 1 else opponent_action
            action2 = action if player == 2 else opponent_action
            _, _, sim_done, _ = sim_env.step(action1, action2)
            if sim_done: break

        leaf_node = search_path[-1]
        value = 0.0

        if not sim_env.done:
            leaf_state_tensor = sim_env.get_current_player_perspective_state(player)
            leaf_valid_actions = sim_env.get_valid_actions(player)
            with torch.no_grad():
                leaf_state_torch = torch.from_numpy(leaf_state_tensor).unsqueeze(0).to(device)
                policy_logits, value_torch = network(leaf_state_torch)
                policy_probs_torch = F.softmax(policy_logits, dim=1).squeeze(0).cpu()
                masked_policy = torch.zeros_like(policy_probs_torch)
                if leaf_valid_actions:
                    masked_policy[leaf_valid_actions] = policy_probs_torch[leaf_valid_actions]
                    policy_sum = masked_policy.sum()
                    if policy_sum > 1e-6: masked_policy /= policy_sum
                    else: masked_policy[leaf_valid_actions] = 1.0 / len(leaf_valid_actions)
                policy_probs_np = masked_policy.numpy()
                value = value_torch.item()
            if leaf_valid_actions:
                leaf_node.expand(policy_probs_np, leaf_valid_actions)
        else:
            _, _, _, sim_info = sim_env.step(0, 0)
            final_reward_z_p1 = sim_info.get('final_reward_z', 0.0)
            value = final_reward_z_p1 if player == 1 else -final_reward_z_p1

        # Use original update_recursive
        leaf_node.update_recursive(value)


    # Calculate final action probabilities based on visit counts (Temperature=0 for testing)
    action_probs = root_node.get_action_probs(temperature=0.0)
    return action_probs


def test_agent_az(agent_path, num_games=10, render=False):
    """Test a trained AlphaZero agent against itself using sequential MCTS."""
    logging.info(f"\n--- Testing AlphaZero Agent: {agent_path} ---")

    network = AlphaZeroNet().to(device)
    if not os.path.exists(agent_path):
        logging.error(f"Error: Model file not found at {agent_path}. Aborting test.")
        return None, None
    checkpoint = torch.load(agent_path, map_location=device)
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()
    logging.info(f"Network loaded from {agent_path} to {device}")

    env = SnakeEnvironment()
    agent_wins = 0
    opponent_wins = 0
    draws = 0
    game_lengths = []
    agent_scores = []
    opponent_scores = []

    # Use fixed MCTS settings for testing
    test_simulations = N_SIMULATIONS * 2 # Maybe more simulations for testing?
    test_c_puct = C_PUCT

    for game in tqdm(range(num_games), desc="Testing Games"):
        env.reset()
        done = False
        game_steps = 0
        final_outcome = 'draw'

        while not done:
            game_steps += 1

            # Player 1's move (Agent)
            mcts_policy1 = run_mcts_sequential(env, 1, network, test_simulations, test_c_puct)
            action1 = np.argmax(mcts_policy1) # Choose best action deterministically

            # Player 2's move (Agent)
            mcts_policy2 = run_mcts_sequential(env, 2, network, test_simulations, test_c_puct)
            action2 = np.argmax(mcts_policy2)

            # Step environment
            _, _, done, info = env.step(action1, action2)
            final_outcome = info['outcome']

            if render:
                render_game_az(env) # Use the existing render function
                time.sleep(0.05)

            if done:
                if final_outcome == 'win': agent_wins += 1
                elif final_outcome == 'loss': opponent_wins += 1
                else: draws += 1
                game_lengths.append(game_steps)
                agent_scores.append(info['s1_score'])
                opponent_scores.append(info['s2_score'])
                if render: time.sleep(1) # Pause at end of rendered game

    # Print summary
    print(f"\n--- Testing Summary ({num_games} games vs self) ---")
    print(f"Agent: {os.path.basename(agent_path)}")
    print(f"Agent Avg Score: {np.mean(agent_scores):.2f} +/- {np.std(agent_scores):.2f}")
    print(f"Opponent Avg Score: {np.mean(opponent_scores):.2f} +/- {np.std(opponent_scores):.2f}")
    print(f"Avg Game Length: {np.mean(game_lengths):.2f} +/- {np.std(game_lengths):.2f}")
    print(f"Agent Wins (P1): {agent_wins} ({agent_wins/num_games*100:.1f}%)")
    print(f"Opponent Wins (P2): {opponent_wins} ({opponent_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print("--------------------------------------")

    return agent_scores, game_lengths

# --- Render Function (same as before) ---
def render_game_az(env):
    grid = [['.' for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    if env.foods:
        for food in env.foods:
            x, y = food.position
            if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH: grid[y][x] = 'F'
    if env.snake2:
        for pos in list(env.snake2.positions)[1:]:
            x, y = pos
            if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH: grid[y][x] = 'x'
    if env.snake1:
        for pos in list(env.snake1.positions)[1:]:
            x, y = pos
            if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH: grid[y][x] = 'o'
    if env.snake2:
        x, y = env.snake2.get_head_position()
        if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH: grid[y][x] = '2'
    if env.snake1:
        x, y = env.snake1.get_head_position()
        if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH: grid[y][x] = '1'

    print("\033[H\033[J", end="") # Clears screen
    print(f"Step: {env.steps}/{MAX_STEPS_PER_EPISODE}")
    s1_score = env.snake1.score if env.snake1 and not env.done else 'DEAD' if env.done else env.snake1.score
    s2_score = env.snake2.score if env.snake2 and not env.done else 'DEAD' if env.done else env.snake2.score
    s1_len = env.snake1.length if env.snake1 else 0
    s2_len = env.snake2.length if env.snake2 else 0
    print(f"Score: S1={s1_score} | S2={s2_score}")
    print(f"Length: S1={s1_len} | S2={s2_len}")
    print('+' + '-' * GRID_WIDTH + '+')
    for row in grid: print('|' + ''.join(row) + '|')
    print('+' + '-' * GRID_WIDTH + '+')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Snake AI v_AZ (Parallel): AlphaZero Style')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode')
    parser.add_argument('--games', type=int, default=100000, help='Number of games target for training / games for testing')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load model state')
    parser.add_argument('--save_dir', type=str, default='az_checkpoints_v1_parallel', help='Directory for checkpoints')
    parser.add_argument('--render', action='store_true', help='Render game during testing')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--parallel_games', type=int, default=N_PARALLEL_GAMES, help='Number of parallel games for self-play')
    parser.add_argument('--simulations', type=int, default=N_SIMULATIONS, help='Number of MCTS simulation steps per move')


    args = parser.parse_args()

    # Update constants from args if provided
    N_PARALLEL_GAMES = args.parallel_games
    N_SIMULATIONS = args.simulations

    if args.mode == 'train':
        print("--- Starting AlphaZero Parallel Training ---")
        main_alpha_zero_parallel(num_total_games_target=args.games,
                                 load_path=args.load_model,
                                 save_dir=args.save_dir,
                                 wandb_log=not args.no_wandb)
    elif args.mode == 'test':
        if not args.load_model:
            print("Error: Please provide a model path using --load_model for testing.")
        else:
            print(f"--- Starting AlphaZero Testing (Sequential MCTS) ---")
            test_agent_az(args.load_model,
                          num_games=args.games,
                          render=args.render)

# Example execution commands:
# Train from scratch for 100k games generated:
# python snake_az_v1_parallel.py --mode train --games 100000 --save_dir az_models_parallel_100k --parallel_games 64 --simulations 50

# Continue training:
# python snake_az_v1_parallel.py --mode train --games 200000 --load_model az_models_parallel_100k/checkpoint_games_....pt --save_dir az_models_parallel_200k

# Test the final agent:
# python snake_az_v1_parallel.py --mode test --load_model az_models_parallel_200k/final_model_....pt --games 50 --render

# --- END OF FILE snake_az_v1_parallel.py ---