
# from google.colab import drive; drive.mount('/content/drive')
#
# ----------------------------------------------------------------
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Adjust cu118 if Colab uses a different CUDA version
# !pip install wandb tqdm numpy matplotlib
# ----------------------------------------------------------------
#
# import wandb
# wandb.login() # 82925d5456555da4dcdb332999faff1cabbfd637
#
# ----------------------------------------------------------------

# snake_6.py
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
from typing import List, Tuple, Dict, Optional, Union
import time
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import argparse # <--- MOVE THIS IMPORT HERE

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
# ... rest of your imports and constants ...
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True # Can slow down training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Constants for 10x10 CNN version ---
GRID_WIDTH = 10
GRID_HEIGHT = 10
INITIAL_SNAKE_LENGTH = 3
ACTIONS = [0, 1, 2, 3]  # Up, Right, Down, Left
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # Corresponds to actions
OPPOSITE_ACTIONS = {0: 2, 1: 3, 2: 0, 3: 1}
MAX_SNAKE_LENGTH = 25
NUM_STATE_CHANNELS = 5
STATE_SHAPE = (NUM_STATE_CHANNELS, GRID_HEIGHT, GRID_WIDTH)

# Hyperparameters
BATCH_SIZE = 256
GAMMA = 0.99
# Epsilon parameters are less relevant with NoisyNets
NOISY_NETS_CONSTANT_EPSILON = 0.01 # Small residual epsilon, can be 0.0
TARGET_UPDATE = 200 # Update target network a bit more frequently with deeper nets
LEARNING_RATE = 1e-4  # Common LR for Adam with NoisyNets/Deeper CNNs
MEMORY_SIZE = 100000 # Increased memory size
MAX_STEPS_PER_EPISODE = 500

# Prioritized Experience Replay (PER) parameters
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = 10000 * 200 # Anneal beta over ~2M agent steps
PER_EPSILON = 1e-6

# Opponent Pool
OPPONENT_POOL_SIZE = 20 # Slightly larger pool
UPDATE_OPPONENT_POOL_FREQ = 500 # Episodes
OPPONENT_RECENCY_BIAS_STRENGTH = 2.0 # Higher values give more weight to recent opponents

# Reward Shaping Constants (same as snake_5.py)
FOOD_REWARD = 10.0
KILL_REWARD = 100.0
DEATH_PENALTY = -10.0
LIVING_BONUS = 0.01
FOOD_CLOSENESS_REWARD_SCALE = 0.02
BAITING_REWARD_SCALE = 0.03
DANGER_PENALTY_SCALE = 0.03
SELF_DANGER_PENALTY_SCALE = 0.03
MAX_PROXIMITY_DIST = 4

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class PrioritizedReplayMemory: # Identical to snake_5.py
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
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        self.beta = min(1.0, self.beta + (1.0 - PER_BETA_START) / PER_BETA_FRAMES)
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + PER_EPSILON

    def __len__(self):
        return len(self.memory)

# --- Noisy Linear Layer ---
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training: # Apply noise only during training
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else: # Use mean weights for evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features}, {self.out_features}, std_init={self.std_init})'

# --- Dueling CNN Q-Network with Noisy Layers ---
class DuelingCNNNoisyQNetwork(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int, noisy_std_init=0.5):
        super(DuelingCNNNoisyQNetwork, self).__init__()
        c, h, w = input_shape

        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)       # (N, 32, H, W)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)      # (N, 64, H, W)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                      # (N, 64, H/2, W/2) -> (N, 64, 5, 5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)     # (N, 128, H/2, W/2)
        self.bn3 = nn.BatchNorm2d(128)
        # Optional: A second pooling layer if H/2 and W/2 are still large enough
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (N, 128, H/4, W/4)

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool1(x)
            x = F.relu(self.bn3(self.conv3(x)))
            # if hasattr(self, 'pool2'): x = self.pool2(x)
            flattened_size = x.numel()

        fc_layer_size = 512 # Size of the hidden FC layer

        # Dueling streams with NoisyLinear layers
        self.fc1_adv = NoisyLinear(flattened_size, fc_layer_size, std_init=noisy_std_init)
        self.fc1_val = NoisyLinear(flattened_size, fc_layer_size, std_init=noisy_std_init)

        self.fc2_adv = NoisyLinear(fc_layer_size, num_actions, std_init=noisy_std_init)
        self.fc2_val = NoisyLinear(fc_layer_size, 1, std_init=noisy_std_init)

        self._initialize_conv_weights()

    def _initialize_conv_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            # NoisyLinear layers initialize themselves

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        # if hasattr(self, 'pool2'): x = self.pool2(x)
        
        x = x.view(x.size(0), -1) # Flatten

        adv = F.relu(self.fc1_adv(x))
        adv = self.fc2_adv(adv)

        val = F.relu(self.fc1_val(x))
        val = self.fc2_val(val)

        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self): # Helper to reset noise in all NoisyLinear submodules
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class DQNAgent:
    def __init__(self, state_shape: Tuple[int, int, int], action_size: int, noisy_std_init=0.5):
        self.state_shape = state_shape
        self.action_size = action_size

        self.policy_net = DuelingCNNNoisyQNetwork(state_shape, action_size, noisy_std_init).to(device)
        self.target_net = DuelingCNNNoisyQNetwork(state_shape, action_size, noisy_std_init).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net is always in eval mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=30, factor=0.5, min_lr=1e-7, verbose=True)

        self.memory = PrioritizedReplayMemory(MEMORY_SIZE)
        self.epsilon = NOISY_NETS_CONSTANT_EPSILON # Fixed small epsilon, or 0.0
        self.steps_done = 0 # For PER beta annealing and other tracking

    def act(self, state: np.ndarray, valid_actions: List[int]):
        # self.steps_done += 1 # Increment here if used for epsilon decay, otherwise in trainer
        if random.random() < self.epsilon: # Minimal random action override
            return random.choice(valid_actions)
        else:
            with torch.no_grad(): # Q-value estimation should not affect gradients here
                # policy_net should be in train() mode if noise is desired for exploration,
                # or eval() mode for deterministic exploitation.
                # This is handled by the trainer.
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                valid_q_values = q_values.clone()
                for action_idx in range(self.action_size):
                    if action_idx not in valid_actions:
                        valid_q_values[0, action_idx] = -float('inf')
                return valid_q_values.max(1)[1].item()

    def remember(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def learn(self):
        if len(self.memory) < BATCH_SIZE: return 0.0
        
        # Ensure policy_net is in training mode for NoisyLinear layers to apply noise
        # if they are part of the computation graph for loss (which they are).
        # However, the noise is sampled once per forward pass if self.training is true.
        # The noise for action selection is handled by the trainer setting policy_net.train().
        # For the learning step, the current Q-values are computed with noise (if policy_net is in train mode).
        # This is standard for Noisy DQN.

        transitions, indices, weights = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.from_numpy(np.stack(batch.state)).float().to(device)
        action_batch = torch.from_numpy(np.array(batch.action)).long().unsqueeze(1).to(device)
        reward_batch = torch.from_numpy(np.array(batch.reward)).float().to(device)
        weights_batch = torch.from_numpy(weights).float().to(device)

        non_final_mask = torch.BoolTensor([s is not None for s in batch.next_state]).to(device)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        
        if non_final_next_states_list:
            non_final_next_states = torch.from_numpy(np.stack(non_final_next_states_list)).float().to(device)
        else:
            non_final_next_states = torch.empty((0, *self.state_shape), device=device, dtype=torch.float32)

        # Q(s_t, a) from policy_net (with noise if in train() mode)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Next state Q-values for Double DQN
        # Actions selected by policy_net (with noise if in train() mode)
        # Values from target_net (in eval() mode, so no noise from target_net's NoisyLinear layers)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        if non_final_mask.sum() > 0:
            # Important: For Double DQN with NoisyNets, the action selection from policy_net
            # should ideally use a *separate* noise sample or be deterministic.
            # However, common practice is to use the current noisy policy_net.
            # For simplicity, we use current noisy policy_net.
            # To be more "pure", one might call policy_net.reset_noise() before this specific max operation.
            next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze(1).detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')
        weighted_loss = (weights_batch.unsqueeze(1) * loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        td_errors = loss.squeeze(1).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        return weighted_loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # If target_net also has NoisyLinear layers, their noise should be reset
        # or they should use the mean weights. Since target_net is always in eval mode,
        # its NoisyLinear layers will use mean weights automatically.

    def reset_policy_noise(self): # Call this at the start of each episode for the learning agent
        self.policy_net.reset_noise()

    def save_state(self, filename):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon, # Save even if constant
            'steps_done': self.steps_done,
            'memory_beta': self.memory.beta
        }, filename)
        print(f"Agent state saved to {filename}")

    def load_state(self, filename):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epsilon = checkpoint.get('epsilon', NOISY_NETS_CONSTANT_EPSILON)
            self.steps_done = checkpoint.get('steps_done', 0)
            self.memory.beta = checkpoint.get('memory_beta', PER_BETA_START)
            self.target_net.eval()
            print(f"Agent state loaded from {filename}")
        else:
            print(f"Warning: Checkpoint file not found at {filename}. Starting fresh.")

# --- Snake, Food, Environment classes (Identical to snake_5.py) ---
@dataclass
class Food: position: Tuple[int, int]
class Snake: # ... (Copy from snake_5.py) ...
    def __init__(self, x, y, initial_direction=None):
        self.positions = deque([(x, y)])
        self.direction = initial_direction if initial_direction is not None else random.choice(DIRECTIONS)
        self.length = INITIAL_SNAKE_LENGTH
        self.score = 0
        self.lifetime = 0
        self.steps_since_food = 0
        self.max_steps_without_food = GRID_WIDTH * GRID_HEIGHT

        current_x, current_y = x, y
        for _ in range(1, INITIAL_SNAKE_LENGTH):
            dx, dy = self.direction
            prev_x = (current_x - dx + GRID_WIDTH) % GRID_WIDTH
            prev_y = (current_y - dy + GRID_HEIGHT) % GRID_HEIGHT
            if (prev_x, prev_y) not in self.positions:
                 self.positions.append((prev_x, prev_y))
                 current_x, current_y = prev_x, prev_y
            else: break
    def get_head_position(self): return self.positions[0]
    def get_action_from_direction(self): return DIRECTIONS.index(self.direction)
    def get_direction_from_action(self, action): return DIRECTIONS[action]
    def get_valid_actions(self):
        current_action = self.get_action_from_direction()
        invalid_action = OPPOSITE_ACTIONS[current_action]
        return [a for a in ACTIONS if a != invalid_action]
    def move(self, action):
        new_direction = self.get_direction_from_action(action)
        self.direction = new_direction
        head_x, head_y = self.get_head_position()
        dx, dy = self.direction
        new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)
        body_without_tail = list(self.positions)
        if len(body_without_tail) >= self.length: body_without_tail = body_without_tail[:-1]
        if new_head in body_without_tail: return True
        self.positions.appendleft(new_head)
        if len(self.positions) > self.length: self.positions.pop()
        self.lifetime += 1
        self.steps_since_food += 1
        if self.steps_since_food >= self.max_steps_without_food: return True
        return False
    def grow(self):
        self.length = min(self.length + 1, MAX_SNAKE_LENGTH)
        self.score += 1
        self.steps_since_food = 0

class SnakeEnvironment: # ... (Copy from snake_5.py, including reward shaping) ...
    def __init__(self):
        self.snake1: Optional[Snake] = None; self.snake2: Optional[Snake] = None
        self.foods: List[Food] = []; self.done = False; self.steps = 0
        self.s1_dist_to_food_prev: Optional[float] = None
        self.s2_dist_to_food_prev: Optional[float] = None
    def _get_manhattan_distance_wrapped(self, pos1, pos2):
        x1, y1 = pos1; x2, y2 = pos2
        dx = abs(x1 - x2); dy = abs(y1 - y2)
        return min(dx, GRID_WIDTH - dx) + min(dy, GRID_HEIGHT - dy)
    def reset(self):
        while True:
            pos1 = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1)); dir1 = random.choice(DIRECTIONS)
            self.snake1 = Snake(*pos1, initial_direction=dir1)
            pos2 = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            while pos2 in self.snake1.positions: pos2 = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            dir2 = random.choice(DIRECTIONS)
            if pos1 == pos2 and dir1 == (-dir2[0], -dir2[1]): dir2 = random.choice([d for d in DIRECTIONS if d != (-dir1[0], -dir1[1])])
            self.snake2 = Snake(*pos2, initial_direction=dir2)
            if not any(p2 in self.snake1.positions for p2 in self.snake2.positions): break
        self.foods = []; self._spawn_food(); self.done = False; self.steps = 0
        self.s1_dist_to_food_prev = None; self.s2_dist_to_food_prev = None
        if self.foods:
            food_pos = self.foods[0].position
            self.s1_dist_to_food_prev = self._get_manhattan_distance_wrapped(self.snake1.get_head_position(), food_pos)
            self.s2_dist_to_food_prev = self._get_manhattan_distance_wrapped(self.snake2.get_head_position(), food_pos)
        return self._get_state(self.snake1, self.snake2), self._get_state(self.snake2, self.snake1)
    def _spawn_food(self):
        while len(self.foods) < 1:
            occupied = set(self.snake1.positions) | set(self.snake2.positions) | {f.position for f in self.foods}
            empty_cells = list(set((x,y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT)) - occupied)
            if empty_cells: self.foods.append(Food(random.choice(empty_cells)))
            else: break
    def _get_state(self, snake: Snake, other_snake: Snake) -> np.ndarray:
        state = np.zeros(STATE_SHAPE, dtype=np.float32)
        hx, hy = snake.get_head_position(); state[0, hy, hx] = 1.0
        for i, (bx, by) in enumerate(list(snake.positions)):
            if i > 0: state[1, by, bx] = 1.0
        ohx, ohy = other_snake.get_head_position(); state[2, ohy, ohx] = 1.0
        for i, (obx, oby) in enumerate(list(other_snake.positions)):
            if i > 0: state[3, oby, obx] = 1.0
        if self.foods: fx, fy = self.foods[0].position; state[4, fy, fx] = 1.0
        return state
    def step(self, action1: int, action2: int):
        self.steps += 1
        if self.done: return (None, None), (0.0, 0.0), True, False, False
        s1_head_prev = self.snake1.get_head_position(); s2_head_prev = self.snake2.get_head_position()
        collision1 = self.snake1.move(action1); collision2 = self.snake2.move(action2)
        head1 = self.snake1.get_head_position(); head2 = self.snake2.get_head_position()
        head_on = (head1 == head2)
        s1_hit_s2b = head1 in list(self.snake2.positions)[1:]
        s2_hit_s1b = head2 in list(self.snake1.positions)[1:]
        if head_on:
            if self.snake1.length > self.snake2.length: collision2 = True
            elif self.snake2.length > self.snake1.length: collision1 = True
            else: collision1, collision2 = True, True
        else:
            if s1_hit_s2b: collision1 = True
            if s2_hit_s1b: collision2 = True
        reward1, reward2 = LIVING_BONUS, LIVING_BONUS; killed1, killed2 = False, False
        if self.foods: # Food reward and closeness
            food_pos = self.foods[0].position
            if head1 == food_pos and not collision1: self.snake1.grow(); reward1 += FOOD_REWARD; self.foods.pop(0); self._spawn_food()
            elif head2 == food_pos and not collision2: self.snake2.grow(); reward2 += FOOD_REWARD; self.foods.pop(0); self._spawn_food()
            next_s1_dist_food, next_s2_dist_food = None, None
            if self.foods: # Food might have been eaten and respawned
                food_pos_after = self.foods[0].position
                next_s1_dist_food = self._get_manhattan_distance_wrapped(head1, food_pos_after)
                next_s2_dist_food = self._get_manhattan_distance_wrapped(head2, food_pos_after)
                if self.s1_dist_to_food_prev is not None and next_s1_dist_food is not None and not collision1:
                    dist_change = self.s1_dist_to_food_prev - next_s1_dist_food
                    if dist_change > 0: reward1 += FOOD_CLOSENESS_REWARD_SCALE * dist_change
                if self.s2_dist_to_food_prev is not None and next_s2_dist_food is not None and not collision2:
                    dist_change = self.s2_dist_to_food_prev - next_s2_dist_food
                    if dist_change > 0: reward2 += FOOD_CLOSENESS_REWARD_SCALE * dist_change
            self.s1_dist_to_food_prev = next_s1_dist_food; self.s2_dist_to_food_prev = next_s2_dist_food
        # Proximity rewards/penalties
        for snake_idx, (main_snake, other_snake, r_val, coll_flag) in enumerate([(self.snake1, self.snake2, reward1, collision1), (self.snake2, self.snake1, reward2, collision2)]):
            if not coll_flag:
                current_reward = r_val
                # Baiting: Opponent head near my body
                min_dist_oh_mb = min((self._get_manhattan_distance_wrapped(other_snake.get_head_position(), bp) for bp in list(main_snake.positions)[1:]), default=float('inf'))
                if min_dist_oh_mb < MAX_PROXIMITY_DIST: current_reward += BAITING_REWARD_SCALE * (MAX_PROXIMITY_DIST - min_dist_oh_mb)
                # Danger: My head near opponent body
                min_dist_mh_ob = min((self._get_manhattan_distance_wrapped(main_snake.get_head_position(), bp) for bp in list(other_snake.positions)[1:]), default=float('inf'))
                if min_dist_mh_ob < MAX_PROXIMITY_DIST: current_reward -= DANGER_PENALTY_SCALE * (MAX_PROXIMITY_DIST - min_dist_mh_ob)
                # Self-Danger: My head near my own body
                if len(main_snake.positions) > 2:
                    min_dist_mh_mb = min((self._get_manhattan_distance_wrapped(main_snake.get_head_position(), bp) for i, bp in enumerate(list(main_snake.positions)) if i > 1), default=float('inf'))
                    if min_dist_mh_mb < MAX_PROXIMITY_DIST: current_reward -= SELF_DANGER_PENALTY_SCALE * (MAX_PROXIMITY_DIST - min_dist_mh_mb)
                if snake_idx == 0: reward1 = current_reward
                else: reward2 = current_reward
        if collision1 and collision2: reward1 += DEATH_PENALTY; reward2 += DEATH_PENALTY; killed1, killed2 = True, True
        elif collision1: reward1 += DEATH_PENALTY; reward2 += KILL_REWARD; killed1 = True
        elif collision2: reward2 += DEATH_PENALTY; reward1 += KILL_REWARD; killed2 = True
        if killed1: reward1 = DEATH_PENALTY # Ensure death penalty is final
        if killed2: reward2 = DEATH_PENALTY
        self.done = killed1 or killed2 or self.steps >= MAX_STEPS_PER_EPISODE
        next_s1 = self._get_state(self.snake1, self.snake2) if not killed1 else None
        next_s2 = self._get_state(self.snake2, self.snake1) if not killed2 else None
        return (next_s1, next_s2), (reward1, reward2), self.done, killed1, killed2

# --- Self-Play Trainer with Enhanced Opponent Pool ---
class SelfPlayTrainer:
    def __init__(self, state_shape: Tuple[int,int,int], action_size: int, agent_load_path=None, wandb_log=True, noisy_std_init=0.5):
        self.state_shape = state_shape
        self.action_size = action_size
        self.agent = DQNAgent(state_shape, action_size, noisy_std_init=noisy_std_init)
        self.env = SnakeEnvironment()
        self.best_avg_score = -float('inf')
        self.scores_window = deque(maxlen=100)
        self.total_steps = 0
        # Opponent pool stores (state_dict, total_steps_at_add)
        self.opponent_pool: deque[Tuple[dict, int]] = deque(maxlen=OPPONENT_POOL_SIZE)

        if agent_load_path:
            self.agent.load_state(agent_load_path)
            self.total_steps = self.agent.steps_done # Sync total steps
            # Re-initialize opponent pool with the loaded agent state
            loaded_state_dict = self.agent.policy_net.state_dict()
            self.opponent_pool.clear()
            for _ in range(OPPONENT_POOL_SIZE): # Fill with current best
                self.opponent_pool.append((loaded_state_dict.copy(), self.total_steps))
        else: # Initialize pool with copies of the initial agent's state
            initial_state_dict = self.agent.policy_net.state_dict()
            for _ in range(OPPONENT_POOL_SIZE):
                 self.opponent_pool.append((initial_state_dict.copy(), 0))

        self.wandb_log = wandb_log
        if self.wandb_log:
            wandb.init(project="snake-ai-1v1-noisy-cnn", config={
                "architecture": "DeeperDuelingCNN_NoisyNets_BatchNorm_MaxPool",
                "noisy_std_init": noisy_std_init,
                "grid_size": f"{GRID_WIDTH}x{GRID_HEIGHT}", "state_channels": NUM_STATE_CHANNELS,
                "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "gamma": GAMMA,
                "constant_epsilon": NOISY_NETS_CONSTANT_EPSILON,
                "target_update": TARGET_UPDATE, "memory_size": MEMORY_SIZE,
                "per_alpha": PER_ALPHA, "per_beta_start": PER_BETA_START, "per_beta_frames": PER_BETA_FRAMES,
                "opponent_pool_size": OPPONENT_POOL_SIZE, "update_opponent_freq": UPDATE_OPPONENT_POOL_FREQ,
                "opponent_recency_bias": OPPONENT_RECENCY_BIAS_STRENGTH,
                "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
                # ... (Include all reward constants) ...
                "reward_food": FOOD_REWARD, "reward_kill": KILL_REWARD, "penalty_death": DEATH_PENALTY,
                "reward_living": LIVING_BONUS, "food_closeness_scale": FOOD_CLOSENESS_REWARD_SCALE,
                "baiting_scale": BAITING_REWARD_SCALE, "danger_scale": DANGER_PENALTY_SCALE,
                "self_danger_scale": SELF_DANGER_PENALTY_SCALE, "max_proximity_dist": MAX_PROXIMITY_DIST,
                "lr_scheduler": "ReduceLROnPlateau"
            })

    def _select_opponent_state_dict(self) -> dict:
        if not self.opponent_pool: # Should not happen if initialized properly
            return self.agent.policy_net.state_dict().copy() # Fallback to self

        pool_size = len(self.opponent_pool)
        # Weighted sampling favoring recent opponents (higher total_steps_at_add)
        # Using total_steps_at_add as a proxy for "strength" or "recency"
        steps_at_add = np.array([data[1] for data in self.opponent_pool], dtype=np.float64)
        
        # Normalize steps to create weights, add small constant to avoid zero weights if all are same
        min_steps = steps_at_add.min()
        weights = (steps_at_add - min_steps + 1.0) ** OPPONENT_RECENCY_BIAS_STRENGTH
        
        if weights.sum() == 0: # All weights zero (e.g. if bias_strength is 0 and all steps are same)
            weights = np.ones(pool_size, dtype=np.float32) # Uniform

        weights /= weights.sum()
        
        try:
            chosen_idx = np.random.choice(pool_size, p=weights)
            return self.opponent_pool[chosen_idx][0]
        except ValueError as e: # If weights don't sum to 1 due to precision
            print(f"Warning: ValueError in opponent selection: {e}. Using uniform sampling.")
            chosen_idx = np.random.choice(pool_size)
            return self.opponent_pool[chosen_idx][0]


    def train(self, num_episodes=50000, eval_interval=100, save_interval=1000, checkpoint_dir="checkpoints_noisy_cnn"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        opponent_agent = DQNAgent(self.state_shape, self.action_size, noisy_std_init=self.agent.policy_net.fc1_adv.std_init)
        opponent_agent.policy_net.eval() # Opponent always in eval mode (no noise for actions)

        for episode in tqdm(range(1, num_episodes + 1)):
            state1, state2 = self.env.reset()
            self.agent.reset_policy_noise() # Reset noise for the learning agent's policy net
            self.agent.policy_net.train()   # Ensure learning agent's policy net is in train mode for noisy exploration

            done = False
            episode_reward1_raw = 0
            episode_loss = 0
            steps_in_episode = 0

            opponent_state_dict = self._select_opponent_state_dict()
            opponent_agent.policy_net.load_state_dict(opponent_state_dict)
            # Target net for opponent doesn't matter as it only acts
            
            while not done:
                valid_actions1 = self.env.snake1.get_valid_actions()
                action1 = self.agent.act(state1, valid_actions1) # policy_net is in train() mode

                valid_actions2 = self.env.snake2.get_valid_actions()
                # Opponent acts with its policy_net in eval() mode (set above)
                action2 = opponent_agent.act(state2, valid_actions2)

                (next_state1, next_state2), (reward1, reward2), done, killed1, killed2 = self.env.step(action1, action2)
                
                self.total_steps += 1
                self.agent.steps_done = self.total_steps # Update agent's internal step counter

                if state1 is not None:
                     self.agent.remember(state1, action1, next_state1, reward1, done or killed1)
                
                # policy_net should be in train() for learning step as well, so NoisyLinear layers
                # contribute their noisy parameters to the Q-value calculation for the current state.
                loss = self.agent.learn()
                episode_loss += loss if loss is not None else 0.0

                state1, state2 = next_state1, next_state2
                episode_reward1_raw += reward1
                steps_in_episode += 1
            
            # After episode, for next episode's action selection, noise will be reset.
            # policy_net remains in train() mode for the trainer.

            self.scores_window.append(self.env.snake1.score)

            if episode % TARGET_UPDATE == 0: self.agent.update_target_net()
            if episode % UPDATE_OPPONENT_POOL_FREQ == 0:
                 current_agent_state_dict = self.agent.policy_net.state_dict().copy()
                 self.opponent_pool.append((current_agent_state_dict, self.total_steps))
                 print(f"\nUpdated opponent pool at ep {episode}. Pool size: {len(self.opponent_pool)}. Added agent from step {self.total_steps}")

            if episode % eval_interval == 0:
                avg_score_window = np.mean(self.scores_window) if self.scores_window else 0.0
                avg_loss_episode = episode_loss / max(1, steps_in_episode)
                self.agent.scheduler.step(avg_loss_episode)

                print(f"\nEp {episode}/{num_episodes} | Steps: {steps_in_episode} | S1 Score: {self.env.snake1.score:.0f} | Avg Score: {avg_score_window:.2f} | Loss: {avg_loss_episode:.4f} | LR: {self.agent.optimizer.param_groups[0]['lr']:.1e} | Beta: {self.agent.memory.beta:.4f}")
                if self.wandb_log:
                    wandb.log({
                        "episode": episode, "score_agent1": self.env.snake1.score,
                        "average_score_100ep": avg_score_window, "average_loss_episode": avg_loss_episode,
                        "learning_rate": self.agent.optimizer.param_groups[0]['lr'],
                        "steps_in_episode": steps_in_episode, "total_steps": self.total_steps,
                        "buffer_size": len(self.agent.memory), "per_beta": self.agent.memory.beta,
                        "snake1_length": self.env.snake1.length, "snake2_length": self.env.snake2.length,
                        "raw_episode_reward1": episode_reward1_raw,
                        "opponent_pool_size": len(self.opponent_pool)
                    })
                if avg_score_window > self.best_avg_score and len(self.scores_window) == 100:
                    self.best_avg_score = avg_score_window
                    # Switch agent to eval mode for saving mean weights of NoisyLinear
                    self.agent.policy_net.eval()
                    best_filename = os.path.join(checkpoint_dir, f"best_agent_noisy_cnn_avg_score_{avg_score_window:.2f}.pt")
                    self.agent.save_state(best_filename)
                    if self.wandb_log: wandb.save(best_filename)
                    self.agent.policy_net.train() # Switch back to train mode

            if episode % save_interval == 0:
                self.agent.policy_net.eval() # Save with mean weights
                ckpt_filename = os.path.join(checkpoint_dir, f"checkpoint_noisy_cnn_ep_{episode}.pt")
                self.agent.save_state(ckpt_filename)
                self.agent.policy_net.train() # Switch back

        self.agent.policy_net.eval() # Final save with mean weights
        final_filename = os.path.join(checkpoint_dir, "final_agent_noisy_cnn.pt")
        self.agent.save_state(final_filename)
        print(f"Training finished. Final agent saved to {final_filename}")
        if self.wandb_log:
            wandb.save(final_filename)
            wandb.finish()

# --- Testing and Rendering (Similar to snake_5.py, but agent uses NoisyNets) ---
def test_agent(agent_path, num_games=10, render=False, opponent_mode='agent', noisy_std_init_test=0.5):
    print(f"\n--- Testing Agent: {agent_path} --- Opponent: {opponent_mode}")
    # For testing, NoisyLinear layers should be in eval mode (use mean weights)
    agent = DQNAgent(STATE_SHAPE, 4, noisy_std_init=noisy_std_init_test)
    agent.load_state(agent_path)
    agent.epsilon = 0.0 # No random actions during testing
    agent.policy_net.eval() # CRITICAL: Ensure NoisyLinear uses mean weights

    opponent_agent = None
    if opponent_mode in ['agent', 'self']:
        opponent_agent = DQNAgent(STATE_SHAPE, 4, noisy_std_init=noisy_std_init_test)
        opponent_path = agent_path
        if opponent_mode == 'self': print(f"Self-play test: Opponent uses same weights: {opponent_path}")
        opponent_agent.load_state(opponent_path)
        opponent_agent.epsilon = 0.0
        opponent_agent.policy_net.eval() # Opponent also uses mean weights

    env = SnakeEnvironment()
    # ... (Rest of test_agent logic from snake_5.py, it should largely work) ...
    agent_scores, opponent_scores, game_lengths = [], [], []
    agent_wins, opponent_wins, draws = 0, 0, 0
    for game_num in range(num_games):
        state1, state2 = env.reset()
        # For testing, noise should be off for the agent being tested.
        # agent.policy_net.eval() is already set.
        done = False; game_steps = 0
        while not done:
            valid_actions1 = env.snake1.get_valid_actions()
            action1 = agent.act(state1, valid_actions1)
            valid_actions2 = env.snake2.get_valid_actions()
            if opponent_mode == 'random': action2 = random.choice(valid_actions2)
            elif opponent_agent: action2 = opponent_agent.act(state2, valid_actions2)
            else: action2 = random.choice(valid_actions2)
            (next_state1, next_state2), (r1,r2), done, k1, k2 = env.step(action1, action2)
            game_steps += 1
            if render: render_game(env); time.sleep(0.1)
            state1, state2 = next_state1, next_state2
            if done:
                s1s,s2s=env.snake1.score,env.snake2.score
                if k1 and k2: draws+=1; outcome="Draw"
                elif k1: opponent_wins+=1; outcome="Opponent Wins"
                elif k2: agent_wins+=1; outcome="Agent Wins"
                else: # Timeout
                    if s1s > s2s: agent_wins+=1; outcome="Agent Wins (T)"
                    elif s2s > s1s: opponent_wins+=1; outcome="Opponent Wins (T)"
                    else: draws+=1; outcome="Draw (T)"
                print(f"Game {game_num+1}/{num_games} - {outcome}! S1:{s1s} S2:{s2s}, Steps:{game_steps}")
        agent_scores.append(env.snake1.score); opponent_scores.append(env.snake2.score); game_lengths.append(game_steps)
    print(f"\n--- Test Summary ({num_games} games) vs {opponent_mode} ---")
    print(f"Agent Avg Score: {np.mean(agent_scores):.2f} +/- {np.std(agent_scores):.2f}")
    print(f"Opponent Avg Score: {np.mean(opponent_scores):.2f} +/- {np.std(opponent_scores):.2f}")
    print(f"Avg Game Length: {np.mean(game_lengths):.2f} +/- {np.std(game_lengths):.2f}")
    print(f"Agent Wins: {agent_wins} ({agent_wins/num_games*100:.1f}%) | Opponent Wins: {opponent_wins} ({opponent_wins/num_games*100:.1f}%) | Draws: {draws} ({draws/num_games*100:.1f}%)")

def render_game(env: SnakeEnvironment): # Identical to snake_5.py
    grid = [['.' for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    if env.foods: grid[env.foods[0].position[1]][env.foods[0].position[0]] = 'F'
    if env.snake2:
        for i, (x,y) in enumerate(list(env.snake2.positions)): grid[y][x] = 'x' if i > 0 else '2'
    if env.snake1:
        for i, (x,y) in enumerate(list(env.snake1.positions)): grid[y][x] = 'o' if i > 0 else '1'
    os.system('cls' if os.name == 'nt' else 'clear')
    s1s = env.snake1.score if env.snake1 else 'N/A'; s2s = env.snake2.score if env.snake2 else 'N/A'
    s1l = env.snake1.length if env.snake1 else 'N/A'; s2l = env.snake2.length if env.snake2 else 'N/A'
    print(f"Step: {env.steps}/{MAX_STEPS_PER_EPISODE} | S1 Scr:{s1s}(L:{s1l}) | S2 Scr:{s2s}(L:{s2l})")
    print('+' + '-'*GRID_WIDTH + '+'); [print('|' + ''.join(r) + '|') for r in grid]; print('+' + '-'*GRID_WIDTH + '+')

# --- Main Execution ---
def main_run(mode='train', episodes=50000, load_model=None, save_interval=1000,
             eval_interval=100, checkpoint_dir='snake_checkpoints_noisy_cnn_v1',
             games=20, render=False, opponent='agent', no_wandb=False, noisy_std_init=0.5):
    args = argparse.Namespace(
        mode=mode, episodes=episodes, load_model=load_model,
        save_interval=save_interval, eval_interval=eval_interval,
        checkpoint_dir=checkpoint_dir, games=games, render=render,
        opponent=opponent, no_wandb=no_wandb, noisy_std_init=noisy_std_init
    ) # For notebook compatibility
    
    if args.mode == 'train':
        print("--- Starting NoisyNet CNN Agent Training ---")
        trainer = SelfPlayTrainer(STATE_SHAPE, 4, agent_load_path=args.load_model,
                                  wandb_log=not args.no_wandb, noisy_std_init=args.noisy_std_init)
        trainer.train(num_episodes=args.episodes, eval_interval=args.eval_interval,
                      save_interval=args.save_interval, checkpoint_dir=args.checkpoint_dir)
    elif args.mode == 'test':
        if not args.load_model: print("Error: --load_model required for testing."); return
        print(f"--- Starting NoisyNet CNN Agent Testing ---")
        test_agent(args.load_model, num_games=args.games, render=args.render,
                   opponent_mode=args.opponent, noisy_std_init_test=args.noisy_std_init)

if __name__ == "__main__":
    # This block now correctly parses command-line arguments
    parser = argparse.ArgumentParser(description='Snake AI NoisyNet D3QN PER Self-Play')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--episodes', type=int, default=50000, help='Number of episodes for training')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load model state')
    parser.add_argument('--save_interval', type=int, default=2000, help='Save checkpoint every N episodes')
    parser.add_argument('--eval_interval', type=int, default=200, help='Evaluate and log every N episodes')
    parser.add_argument('--checkpoint_dir', type=str, default='snake_checkpoints_noisy_cnn_v1', help='Dir for checkpoints')
    parser.add_argument('--games', type=int, default=20, help='Number of games for testing')
    parser.add_argument('--render', action='store_true', help='Render game during testing')
    parser.add_argument('--opponent', type=str, default='agent', choices=['agent', 'random', 'self'], help='Opponent type for testing')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--noisy_std_init', type=float, default=0.5, help='Initial std for NoisyLinear layers')
    
    # Parse the arguments from the command line
    cli_args = parser.parse_args()

    # Call main_run using the parsed arguments
    main_run(
        mode=cli_args.mode,
        episodes=cli_args.episodes,
        load_model=cli_args.load_model,
        save_interval=cli_args.save_interval,
        eval_interval=cli_args.eval_interval,
        checkpoint_dir=cli_args.checkpoint_dir,
        games=cli_args.games,
        render=cli_args.render,
        opponent=cli_args.opponent,
        no_wandb=cli_args.no_wandb,
        noisy_std_init=cli_args.noisy_std_init
    )

     # ... (Test mode example remains the same) ...

     # Example of how you would call it for testing after training:
     # main_run(
     #     mode='test',
     #     load_model='/content/drive/MyDrive/MySnakeAI/checkpoints_noisy_cnn_v1/best_agent_noisy_cnn_avg_score_XX.XX.pt', # <--- CHANGE THIS to your saved model path
     #     games=50, # Number of test games
     #     render=True, # Set to True to see the game visually during testing
     #     opponent='agent', # Opponent type for testing ('agent', 'random', 'self')
     #     no_wandb=True # WandB usually not needed for simple testing
     # )