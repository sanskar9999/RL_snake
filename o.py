import pygame
import random
import numpy as np

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
GRID_SIZE = 10
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRID_COLOR = (10, 10, 10)  # Faint grid color
SNAKE1_COLOR = (0, 255, 0)    # Green (o)
SNAKE2_COLOR = (0, 0, 255)    # Blue (o1)
SNAKE3_COLOR = (255, 255, 0)  # Yellow (o2)
SNAKE4_COLOR = (255, 0, 255)  # Magenta (o3)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Respawn time (in frames)
RESPAWN_TIME = 100

# Global best parameters for learning snakes
global_best_o1 = {'score': 78, 'food_weight': 0.84, 'safety_weight': 0.47, 'look_ahead': 3, 'aggression': 0.50}
global_best_o2 = {'score': 6054, 'food_weight': 0.10, 'safety_weight': 4.80, 'look_ahead': 10, 'aggression': 1.35}
global_best_o3 = {'score': 3946, 'food_weight': 0.10, 'safety_weight': 4.89, 'look_ahead': 10, 'aggression': 1.58}

# Track the best snake overall
best_snake_overall = {'snake': 'SnakeO2', 'score': 6054, 'food_weight': 0.10, 'safety_weight': 4.80, 'look_ahead': 10, 'aggression': 1.35}


class SnakeO:
    def __init__(self, snake_id, start_position, start_direction, color):
        self.snake_id = snake_id
        self.positions = [start_position]
        self.length = 5
        self.direction = start_direction
        self.score = 0
        self.color = color
        self.alive = True
        self.move_cooldown = 0
        self.respawn_counter = 0

    def get_head_position(self):
        return self.positions[0]

    def update(self, other_snakes_positions):
        if not self.alive:
            if self.respawn_counter > 0:
                self.respawn_counter -= 1
                if self.respawn_counter == 0:
                    self.reset()
            return []

        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        head = self.get_head_position()
        x, y = self.direction
        new_position = (((head[0] + x) % GRID_WIDTH), ((head[1] + y) % GRID_HEIGHT))

        collided = False
        food_to_spawn = []

        if new_position in self.positions[1:] or new_position in other_snakes_positions:
            collided = True

        if collided:
            food_count = max(self.length // 2, 1)
            for i in range(food_count):
                if i < len(self.positions):
                    food_to_spawn.append(self.positions[i])
            self.alive = False
            self.respawn_counter = RESPAWN_TIME
            self.score -= 10  # Deduct 10 points on death
            return food_to_spawn

        self.positions.insert(0, new_position)
        if len(self.positions) > self.length:
            self.positions.pop()
        return []

    def reset(self):
        self.positions = [(GRID_WIDTH // 5, GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.length = 5
        self.alive = True

    def ai_move(self, food_positions, other_snakes_positions, other_snakes_heads):
        if not self.alive or self.move_cooldown > 0:
            return

        head = self.get_head_position()
        nearest_food = min(food_positions, key=lambda f: abs(head[0] - f[0]) + abs(head[1] - f[1]), default=None)

        if nearest_food:
            dx = nearest_food[0] - head[0]
            dy = nearest_food[1] - head[1]
            if abs(dx) > GRID_WIDTH // 2: dx = -dx
            if abs(dy) > GRID_HEIGHT // 2: dy = -dy

            possible_directions = []
            if abs(dx) > abs(dy):
                if dx > 0 and self.direction != LEFT: possible_directions.append(RIGHT)
                elif dx < 0 and self.direction != RIGHT: possible_directions.append(LEFT)
                if dy > 0 and self.direction != UP: possible_directions.append(DOWN)
                elif dy < 0 and self.direction != DOWN: possible_directions.append(UP)
            else:
                if dy > 0 and self.direction != UP: possible_directions.append(DOWN)
                elif dy < 0 and self.direction != DOWN: possible_directions.append(UP)
                if dx > 0 and self.direction != LEFT: possible_directions.append(RIGHT)
                elif dx < 0 and self.direction != RIGHT: possible_directions.append(LEFT)
        else:
            possible_directions = [d for d in [UP, DOWN, LEFT, RIGHT] if d != (-self.direction[0], -self.direction[1])]

        safe_directions = []
        for direction in possible_directions:
            new_pos = ((head[0] + direction[0]) % GRID_WIDTH, (head[1] + direction[1]) % GRID_HEIGHT)
            if new_pos not in self.positions[1:] and new_pos not in other_snakes_positions:
                safe_directions.append(direction)

        if safe_directions:
            self.direction = safe_directions[0]
        self.move_cooldown = 1

    def render(self, surface):
        if not self.alive:
            return
        for i, position in enumerate(self.positions):
            rect = pygame.Rect((position[0] * GRID_SIZE, position[1] * GRID_SIZE), (GRID_SIZE, GRID_SIZE))
            color = (max(0, self.color[0] - 30), max(0, self.color[1] - 30), max(0, self.color[2] - 30)) if i == 0 else self.color
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, BLACK, rect, 1)


class SnakeO1:
    def __init__(self, snake_id, start_position, start_direction, color):
        self.snake_id = snake_id
        self.positions = [start_position]
        self.length = 3
        self.direction = start_direction
        self.score = 0
        self.color = color
        self.alive = True
        self.move_cooldown = 0
        self.respawn_counter = 0
        # Initialize with best parameters
        self.food_weight = global_best_o1['food_weight']
        self.safety_weight = global_best_o1['safety_weight']
        self.look_ahead = global_best_o1['look_ahead']
        self.aggression = global_best_o1['aggression']
        self.best_score = global_best_o1['score']
        self.best_food_weight = self.food_weight
        self.best_safety_weight = self.safety_weight
        self.best_look_ahead = self.look_ahead
        self.best_aggression = self.aggression
        self.mutation_rate = 0.1
        self.aggressive_mutation = False

    def get_head_position(self):
        return self.positions[0]

    def update(self, other_snakes_positions):
        if not self.alive:
            if self.respawn_counter > 0:
                self.respawn_counter -= 1
                if self.respawn_counter == 0:
                    self.reset()
            return []

        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        head = self.get_head_position()
        x, y = self.direction
        new_position = (((head[0] + x) % GRID_WIDTH), ((head[1] + y) % GRID_HEIGHT))

        if new_position in self.positions[1:] or new_position in other_snakes_positions:
            food_count = max(self.length // 2, 1)
            food_to_spawn = [self.positions[i] for i in range(min(food_count, len(self.positions)))]
            self.alive = False
            self.respawn_counter = RESPAWN_TIME
            self.score -= 10  # Deduct 10 points on death
            return food_to_spawn

        self.positions.insert(0, new_position)
        if len(self.positions) > self.length:
            self.positions.pop()
        return []

    def reset(self):
        global global_best_o1, best_snake_overall
        self.positions = [(2 * GRID_WIDTH // 5, GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.length = 3
        self.alive = True
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_food_weight = self.food_weight
            self.best_safety_weight = self.safety_weight
            self.best_look_ahead = self.look_ahead
            self.best_aggression = self.aggression
            self.aggressive_mutation = False
            # Update global best for this snake type
            global_best_o1.update({
                'score': self.score,
                'food_weight': self.food_weight,
                'safety_weight': self.safety_weight,
                'look_ahead': self.look_ahead,
                'aggression': self.aggression
            })
            # Check if this is the best overall
            if self.score > best_snake_overall['score']:
                best_snake_overall.update({
                    'snake': 'SnakeO1',
                    'score': self.score,
                    'food_weight': self.food_weight,
                    'safety_weight': self.safety_weight,
                    'look_ahead': self.look_ahead,
                    'aggression': self.aggression
                })
        else:
            self.food_weight, self.safety_weight, self.look_ahead, self.aggression = (
                self.best_food_weight, self.best_safety_weight, self.best_look_ahead, self.best_aggression
            )
            self.aggressive_mutation = True
        self.mutate_parameters()

    def mutate_parameters(self):
        rate = self.mutation_rate * (2 if self.aggressive_mutation else 1)
        self.food_weight = max(0.1, min(5.0, self.food_weight + np.random.uniform(-rate, rate)))
        self.safety_weight = max(0.1, min(5.0, self.safety_weight + np.random.uniform(-rate, rate)))
        self.aggression = max(0.0, min(2.0, self.aggression + np.random.uniform(-rate, rate)))
        self.look_ahead = max(1, min(10, self.look_ahead + random.choice([-1, 0, 1])))

    def ai_move(self, food_positions, other_snakes_positions, other_snakes_heads):
        if not self.alive or self.move_cooldown > 0:
            return

        head = self.get_head_position()
        scores = {}
        for direction in [UP, DOWN, LEFT, RIGHT]:
            if direction == (-self.direction[0], -self.direction[1]):
                continue
            safe = True
            future_pos = head
            for _ in range(self.look_ahead):
                future_pos = ((future_pos[0] + direction[0]) % GRID_WIDTH, (future_pos[1] + direction[1]) % GRID_HEIGHT)
                if future_pos in self.positions[1:] or future_pos in other_snakes_positions:
                    safe = False
                    break
            safety_score = 1 if safe else 0
            new_head = ((head[0] + direction[0]) % GRID_WIDTH, (head[1] + direction[1]) % GRID_HEIGHT)
            food_benefit = (min(abs(head[0] - f[0]) + abs(head[1] - f[1]) for f in food_positions) -
                            min(abs(new_head[0] - f[0]) + abs(new_head[1] - f[1]) for f in food_positions)) if food_positions else 0
            # Aggression: Score based on proximity to other snakes' heads
            aggression_score = 0
            if other_snakes_heads:
                current_dist = min(abs(head[0] - h[0]) + abs(head[1] - h[1]) for h in other_snakes_heads)
                new_dist = min(abs(new_head[0] - h[0]) + abs(new_head[1] - h[1]) for h in other_snakes_heads)
                aggression_score = (current_dist - new_dist) * self.aggression
            scores[direction] = (food_benefit * self.food_weight) + (safety_score * self.safety_weight) + aggression_score

        if scores:
            self.direction = max(scores, key=scores.get)
            self.move_cooldown = 1

    def render(self, surface):
        if not self.alive:
            return
        for i, position in enumerate(self.positions):
            rect = pygame.Rect((position[0] * GRID_SIZE, position[1] * GRID_SIZE), (GRID_SIZE, GRID_SIZE))
            color = (max(0, self.color[0] - 30), max(0, self.color[1] - 30), max(0, self.color[2] - 30)) if i == 0 else self.color
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, BLACK, rect, 1)


class SnakeO2:
    def __init__(self, snake_id, start_position, start_direction, color):
        self.snake_id = snake_id
        self.positions = [start_position]
        self.length = 3
        self.direction = start_direction
        self.score = 0
        self.color = color
        self.alive = True
        self.move_cooldown = 0
        self.respawn_counter = 0
        self.food_weight = global_best_o2['food_weight']
        self.safety_weight = global_best_o2['safety_weight']
        self.look_ahead = global_best_o2['look_ahead']
        self.aggression = global_best_o2['aggression']
        self.best_score = global_best_o2['score']
        self.best_food_weight = self.food_weight
        self.best_safety_weight = self.safety_weight
        self.best_look_ahead = self.look_ahead
        self.best_aggression = self.aggression
        self.mutation_rate = 0.1
        self.aggressive_mutation = False
        self.last_move_safety = 1
        self.last_collision_type = None

    def get_head_position(self):
        return self.positions[0]

    def update(self, other_snakes_positions):
        if not self.alive:
            if self.respawn_counter > 0:
                self.respawn_counter -= 1
                if self.respawn_counter == 0:
                    self.reset()
            return []

        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        head = self.get_head_position()
        x, y = self.direction
        new_position = (((head[0] + x) % GRID_WIDTH), ((head[1] + y) % GRID_HEIGHT))

        if new_position in self.positions[1:]:
            self.last_collision_type = 'self'
            food_count = max(self.length // 2, 1)
            food_to_spawn = [self.positions[i] for i in range(min(food_count, len(self.positions)))]
            self.alive = False
            self.respawn_counter = RESPAWN_TIME
            self.score -= 10
            return food_to_spawn
        elif new_position in other_snakes_positions:
            self.last_collision_type = 'other'
            food_count = max(self.length // 2, 1)
            food_to_spawn = [self.positions[i] for i in range(min(food_count, len(self.positions)))]
            self.alive = False
            self.respawn_counter = RESPAWN_TIME
            self.score -= 10
            return food_to_spawn

        self.positions.insert(0, new_position)
        if len(self.positions) > self.length:
            self.positions.pop()
        return []

    def reset(self):
        global global_best_o2, best_snake_overall
        self.positions = [(3 * GRID_WIDTH // 5, GRID_HEIGHT // 2)]
        self.direction = LEFT
        self.length = 3
        self.alive = True
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_food_weight = self.food_weight
            self.best_safety_weight = self.safety_weight
            self.best_look_ahead = self.look_ahead
            self.best_aggression = self.aggression
            self.aggressive_mutation = False
            global_best_o2.update({
                'score': self.score,
                'food_weight': self.food_weight,
                'safety_weight': self.safety_weight,
                'look_ahead': self.look_ahead,
                'aggression': self.aggression
            })
            if self.score > best_snake_overall['score']:
                best_snake_overall.update({
                    'snake': 'SnakeO2',
                    'score': self.score,
                    'food_weight': self.food_weight,
                    'safety_weight': self.safety_weight,
                    'look_ahead': self.look_ahead,
                    'aggression': self.aggression
                })
        else:
            self.food_weight, self.safety_weight, self.look_ahead, self.aggression = (
                self.best_food_weight, self.best_safety_weight, self.best_look_ahead, self.best_aggression
            )
            self.aggressive_mutation = True
        if self.last_move_safety == 0:
            self.safety_weight += self.mutation_rate
            self.food_weight -= self.mutation_rate
        if self.last_collision_type == 'other':
            self.look_ahead = min(self.look_ahead + 1, 10)
        elif self.last_collision_type == 'self':
            self.safety_weight += self.mutation_rate
        self.food_weight = max(0.1, min(5.0, self.food_weight))
        self.safety_weight = max(0.1, min(5.0, self.safety_weight))
        self.look_ahead = max(1, min(10, self.look_ahead))
        self.aggression = max(0.0, min(2.0, self.aggression))
        self.mutate_parameters()

    def mutate_parameters(self):
        rate = self.mutation_rate * (2 if self.aggressive_mutation else 1)
        self.food_weight = max(0.1, min(5.0, self.food_weight + np.random.uniform(-rate, rate)))
        self.safety_weight = max(0.1, min(5.0, self.safety_weight + np.random.uniform(-rate, rate)))
        self.aggression = max(0.0, min(2.0, self.aggression + np.random.uniform(-rate, rate)))
        self.look_ahead = max(1, min(10, self.look_ahead + random.choice([-1, 0, 1])))

    def ai_move(self, food_positions, other_snakes_positions, other_snakes_heads):
        if not self.alive or self.move_cooldown > 0:
            return

        head = self.get_head_position()
        scores = {}
        for direction in [UP, DOWN, LEFT, RIGHT]:
            if direction == (-self.direction[0], -self.direction[1]):
                continue
            safe = True
            future_pos = head
            for _ in range(self.look_ahead):
                future_pos = ((future_pos[0] + direction[0]) % GRID_WIDTH, (future_pos[1] + direction[1]) % GRID_HEIGHT)
                if future_pos in self.positions[1:] or future_pos in other_snakes_positions:
                    safe = False
                    break
            safety_score = 1 if safe else 0
            new_head = ((head[0] + direction[0]) % GRID_WIDTH, (head[1] + direction[1]) % GRID_HEIGHT)
            food_benefit = (min(abs(head[0] - f[0]) + abs(head[1] - f[1]) for f in food_positions) -
                            min(abs(new_head[0] - f[0]) + abs(new_head[1] - f[1]) for f in food_positions)) if food_positions else 0
            aggression_score = 0
            if other_snakes_heads:
                current_dist = min(abs(head[0] - h[0]) + abs(head[1] - h[1]) for h in other_snakes_heads)
                new_dist = min(abs(new_head[0] - h[0]) + abs(new_head[1] - h[1]) for h in other_snakes_heads)
                aggression_score = (current_dist - new_dist) * self.aggression
            scores[direction] = {
                'total': (food_benefit * self.food_weight) + (safety_score * self.safety_weight) + aggression_score,
                'safety': safety_score
            }

        if scores:
            best_direction = max(scores, key=lambda k: scores[k]['total'])
            self.direction = best_direction
            self.last_move_safety = scores[best_direction]['safety']
            self.move_cooldown = 1

    def render(self, surface):
        if not self.alive:
            return
        for i, position in enumerate(self.positions):
            rect = pygame.Rect((position[0] * GRID_SIZE, position[1] * GRID_SIZE), (GRID_SIZE, GRID_SIZE))
            color = (max(0, self.color[0] - 30), max(0, self.color[1] - 30), max(0, self.color[2] - 30)) if i == 0 else self.color
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, BLACK, rect, 1)


class SnakeO3:
    def __init__(self, snake_id, start_position, start_direction, color):
        self.snake_id = snake_id
        self.positions = [start_position]
        self.length = 3
        self.direction = start_direction
        self.score = 0
        self.color = color
        self.alive = True
        self.move_cooldown = 0
        self.respawn_counter = 0
        self.food_weight = global_best_o3['food_weight']
        self.safety_weight = global_best_o3['safety_weight']
        self.look_ahead = global_best_o3['look_ahead']
        self.aggression = global_best_o3['aggression']
        self.best_score = global_best_o3['score']
        self.best_food_weight = self.food_weight
        self.best_safety_weight = self.safety_weight
        self.best_look_ahead = self.look_ahead
        self.best_aggression = self.aggression
        self.mutation_rate = 0.1
        self.aggressive_mutation = False
        self.last_move_safety = 1
        self.last_collision_type = None
        self.hunger = 0

    def get_head_position(self):
        return self.positions[0]

    def update(self, other_snakes_positions):
        if not self.alive:
            if self.respawn_counter > 0:
                self.respawn_counter -= 1
                if self.respawn_counter == 0:
                    self.reset()
            return []

        if self.move_cooldown > 0:
            self.move_cooldown -= 1

        head = self.get_head_position()
        x, y = self.direction
        new_position = (((head[0] + x) % GRID_WIDTH), ((head[1] + y) % GRID_HEIGHT))

        if new_position in self.positions[1:]:
            self.last_collision_type = 'self'
            food_count = max(self.length // 2, 1)
            food_to_spawn = [self.positions[i] for i in range(min(food_count, len(self.positions)))]
            self.alive = False
            self.respawn_counter = RESPAWN_TIME
            self.score -= 10
            return food_to_spawn
        elif new_position in other_snakes_positions:
            self.last_collision_type = 'other'
            food_count = max(self.length // 2, 1)
            food_to_spawn = [self.positions[i] for i in range(min(food_count, len(self.positions)))]
            self.alive = False
            self.respawn_counter = RESPAWN_TIME
            self.score -= 10
            return food_to_spawn

        self.positions.insert(0, new_position)
        if len(self.positions) > self.length:
            self.positions.pop()
        self.hunger += 1
        return []

    def reset(self):
        global global_best_o3, best_snake_overall
        self.positions = [(4 * GRID_WIDTH // 5, GRID_HEIGHT // 2)]
        self.direction = LEFT
        self.length = 3
        self.alive = True
        self.hunger = 0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_food_weight = self.food_weight
            self.best_safety_weight = self.safety_weight
            self.best_look_ahead = self.look_ahead
            self.best_aggression = self.aggression
            self.aggressive_mutation = False
            global_best_o3.update({
                'score': self.score,
                'food_weight': self.food_weight,
                'safety_weight': self.safety_weight,
                'look_ahead': self.look_ahead,
                'aggression': self.aggression
            })
            if self.score > best_snake_overall['score']:
                best_snake_overall.update({
                    'snake': 'SnakeO3',
                    'score': self.score,
                    'food_weight': self.food_weight,
                    'safety_weight': self.safety_weight,
                    'look_ahead': self.look_ahead,
                    'aggression': self.aggression
                })
        else:
            self.food_weight, self.safety_weight, self.look_ahead, self.aggression = (
                self.best_food_weight, self.best_safety_weight, self.best_look_ahead, self.best_aggression
            )
            self.aggressive_mutation = True
        if self.last_move_safety == 0:
            self.safety_weight += self.mutation_rate
            self.food_weight -= self.mutation_rate
        if self.last_collision_type == 'other':
            self.look_ahead = min(self.look_ahead + 1, 10)
        elif self.last_collision_type == 'self':
            self.safety_weight += self.mutation_rate
        self.food_weight = max(0.1, min(5.0, self.food_weight))
        self.safety_weight = max(0.1, min(5.0, self.safety_weight))
        self.look_ahead = max(1, min(10, self.look_ahead))
        self.aggression = max(0.0, min(2.0, self.aggression))
        self.mutate_parameters()

    def mutate_parameters(self):
        rate = self.mutation_rate * (2 if self.aggressive_mutation else 1)
        self.food_weight = max(0.1, min(5.0, self.food_weight + np.random.uniform(-rate, rate)))
        self.safety_weight = max(0.1, min(5.0, self.safety_weight + np.random.uniform(-rate, rate)))
        self.aggression = max(0.0, min(2.0, self.aggression + np.random.uniform(-rate, rate)))
        self.look_ahead = max(1, min(10, self.look_ahead + random.choice([-1, 0, 1])))

    def ai_move(self, food_positions, other_snakes_positions, other_snakes_heads):
        if not self.alive or self.move_cooldown > 0:
            return

        head = self.get_head_position()
        scores = {}
        for direction in [UP, DOWN, LEFT, RIGHT]:
            if direction == (-self.direction[0], -self.direction[1]):
                continue
            safe = True
            future_pos = head
            for step in range(self.look_ahead):
                future_pos = ((future_pos[0] + direction[0]) % GRID_WIDTH, (future_pos[1] + direction[1]) % GRID_HEIGHT)
                if future_pos in self.positions[1:] or (step == 0 and future_pos in other_snakes_positions):
                    safe = False
                    break
            safety_score = 1 if safe else 0
            new_head = ((head[0] + direction[0]) % GRID_WIDTH, (head[1] + direction[1]) % GRID_HEIGHT)
            food_benefit = (min(abs(head[0] - f[0]) + abs(head[1] - f[1]) for f in food_positions) -
                            min(abs(new_head[0] - f[0]) + abs(new_head[1] - f[1]) for f in food_positions)) if food_positions else 0
            hunger_modifier = 1 + (self.hunger * 0.005)
            aggression_score = 0
            if other_snakes_heads:
                current_dist = min(abs(head[0] - h[0]) + abs(head[1] - h[1]) for h in other_snakes_heads)
                new_dist = min(abs(new_head[0] - h[0]) + abs(new_head[1] - h[1]) for h in other_snakes_heads)
                aggression_score = (current_dist - new_dist) * self.aggression
            scores[direction] = {
                'total': (food_benefit * self.food_weight * hunger_modifier) + (safety_score * self.safety_weight) + aggression_score,
                'safety': safety_score
            }

        if scores:
            max_score = max(scores[k]['total'] for k in scores)
            best_directions = [k for k in scores if scores[k]['total'] == max_score]
            self.direction = random.choice(best_directions)
            self.last_move_safety = scores[self.direction]['safety']
            self.move_cooldown = 1

    def render(self, surface):
        if not self.alive:
            return
        for i, position in enumerate(self.positions):
            rect = pygame.Rect((position[0] * GRID_SIZE, position[1] * GRID_SIZE), (GRID_SIZE, GRID_SIZE))
            color = (max(0, self.color[0] - 30), max(0, self.color[1] - 30), max(0, self.color[2] - 30)) if i == 0 else self.color
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, BLACK, rect, 1)


class Food:
    def __init__(self, position=None):
        self.color = RED
        self.position = position if position else self.randomize_position()

    def randomize_position(self):
        return (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

    def render(self, surface):
        rect = pygame.Rect((self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE), (GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(surface, self.color, rect)
        pygame.draw.rect(surface, BLACK, rect, 1)


class Game:
    def __init__(self):
        self.snakes = [
            SnakeO(0, (GRID_WIDTH // 5, GRID_HEIGHT // 2), RIGHT, SNAKE1_COLOR),
            SnakeO1(1, (2 * GRID_WIDTH // 5, GRID_HEIGHT // 2), RIGHT, SNAKE2_COLOR),
            SnakeO2(2, (3 * GRID_WIDTH // 5, GRID_HEIGHT // 2), LEFT, SNAKE3_COLOR),
            SnakeO3(3, (4 * GRID_WIDTH // 5, GRID_HEIGHT // 2), LEFT, SNAKE4_COLOR)
        ]
        self.foods = []
        self.initialize_food()  # Call a simplified initialization

    def initialize_food(self):
        # Simplified initialization: Just spawn 8 food items randomly.
        for _ in range(8):
            self.spawn_food()

    def spawn_food(self, position=None):
        food = Food(position)
        all_positions = [pos for snake in self.snakes if snake.alive for pos in snake.positions]
        while food.position in all_positions:
            food.position = food.randomize_position()
        self.foods.append(food)


    def update(self):
        for i, snake in enumerate(self.snakes):
            other_positions = [pos for j, s in enumerate(self.snakes) if j != i and s.alive for pos in s.positions]
            food_positions = snake.update(other_positions)
            for pos in food_positions:
                self.spawn_food(pos)  # Simplified: Spawn at the given position

        for snake in self.snakes:
            if not snake.alive:
                continue
            head = snake.get_head_position()
            for food in self.foods[:]:
                if head == food.position:
                    snake.length += 1
                    snake.score += 1
                    if hasattr(snake, 'hunger'):  # Only SnakeO3 has hunger
                        snake.hunger = 0
                    self.foods.remove(food)

        # Ensure at least 8 food items
        while len(self.foods) < 8:
            self.spawn_food()


    def move_ai(self):
        food_positions = [food.position for food in self.foods]
        heads = [snake.get_head_position() for snake in self.snakes if snake.alive]
        for i, snake in enumerate(self.snakes):
            other_positions = [pos for j, s in enumerate(self.snakes) if j != i and s.alive for pos in s.positions]
            other_heads = [h for j, h in enumerate(heads) if j != i]
            snake.ai_move(food_positions, other_positions, other_heads)

    def render(self, surface):
        # Draw grid
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(surface, GRID_COLOR, (0, y), (SCREEN_WIDTH, y))
        # Draw food and snakes
        for food in self.foods:
            food.render(surface)
        for snake in self.snakes:
            snake.render(surface)

    def get_respawn_time(self, snake):
        return 0 if snake.alive else snake.respawn_counter


def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('4 AI Snake Competition')
    clock = pygame.time.Clock()
    game = Game()
    font = pygame.font.Font(None, 18)
    small_font = pygame.font.Font(None, 18)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game.move_ai()
        game.update()

        screen.fill(BLACK)

        # Centered Title
        title = font.render('4 AI Snake Competition', True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 20))
        screen.blit(title, title_rect)

        # FPS Display
        fps = small_font.render(f'FPS: {int(clock.get_fps())}', True, WHITE)
        screen.blit(fps, (SCREEN_WIDTH - 100, 10))

        game.render(screen)  # Corrected line: Pass the screen surface

        # Color-Coded Background for Scores
        score_bg = pygame.Surface((200, 100))
        score_bg.set_alpha(128)
        score_bg.fill((0, 0, 0))
        screen.blit(score_bg, (0, 0))

        # Display scores
        for i, snake in enumerate(game.snakes):
            names = ['O', 'O1', 'O2', 'O3']
            score_text = font.render(f'{names[i]}: {snake.score}', True, snake.color)
            screen.blit(score_text, (10, 10 + i * 20))

        # Display global best parameters in bottom-right corner
        best_params = [
            f"O1: S={global_best_o1['score']}, F={global_best_o1['food_weight']:.2f}, S={global_best_o1['safety_weight']:.2f}, L={global_best_o1['look_ahead']}, A={global_best_o1['aggression']:.2f}",
            f"O2: S={global_best_o2['score']}, F={global_best_o2['food_weight']:.2f}, S={global_best_o2['safety_weight']:.2f}, L={global_best_o2['look_ahead']}, A={global_best_o2['aggression']:.2f}",
            f"O3: S={global_best_o3['score']}, F={global_best_o3['food_weight']:.2f}, S={global_best_o3['safety_weight']:.2f}, L={global_best_o3['look_ahead']}, A={global_best_o3['aggression']:.2f}"
        ]
        for i, text in enumerate(best_params):
            param_text = small_font.render(text, True, WHITE)
            screen.blit(param_text, (SCREEN_WIDTH - 300, SCREEN_HEIGHT - 90 + i * 20))

        # Display overall best snake
        if best_snake_overall['snake']:
            overall_text = f"Best Overall: {best_snake_overall['snake'].replace('Snake', 'O')} (S={best_snake_overall['score']}, F={best_snake_overall['food_weight']:.2f}, S={best_snake_overall['safety_weight']:.2f}, L={best_snake_overall['look_ahead']}, A={best_snake_overall['aggression']:.2f})"
            overall_text_render = small_font.render(overall_text, True, WHITE)
            screen.blit(overall_text_render, (10, SCREEN_HEIGHT - 50))

        # Display current parameters for learning snakes (continued)
        current_params = [
            f"O1 Now: F={game.snakes[1].food_weight:.2f}, S={game.snakes[1].safety_weight:.2f}, L={game.snakes[1].look_ahead}, A={game.snakes[1].aggression:.2f}",
            f"O2 Now: F={game.snakes[2].food_weight:.2f}, S={game.snakes[2].safety_weight:.2f}, L={game.snakes[2].look_ahead}, A={game.snakes[2].aggression:.2f}",
            f"O3 Now: F={game.snakes[3].food_weight:.2f}, S={game.snakes[3].safety_weight:.2f}, L={game.snakes[3].look_ahead}, A={game.snakes[3].aggression:.2f}"
        ]
        for i, text in enumerate(current_params):
            current_param_text = small_font.render(text, True, WHITE)
            screen.blit(current_param_text, (SCREEN_WIDTH - 300, SCREEN_HEIGHT - 30 + i * 20))

        # Display death messages on the right side
        for i, snake in enumerate(game.snakes):
            names = ['O', 'O1', 'O2', 'O3']
            if not snake.alive:
                status = small_font.render(f'{names[i]} died!', True, RED)
                timer = small_font.render(f'Respawn in: {game.get_respawn_time(snake) // 10 + 1}s', True, WHITE)
                screen.blit(status, (SCREEN_WIDTH - 150, 50 + i * 60))
                screen.blit(timer, (SCREEN_WIDTH - 150, 70 + i * 60))

        # Health Bar for SnakeO3 (Hunger)
        if game.snakes[3].alive:
            hunger = game.snakes[3].hunger
            max_hunger = 1000  # Arbitrary max hunger for visualization
            health_width = 100 * (1 - min(hunger / max_hunger, 1))
            health_bar = pygame.Surface((100, 10))
            health_bar.fill((255, 0, 0))  # Red for danger
            health_bar.fill((0, 255, 0), (0, 0, health_width, 10))  # Green for health
            screen.blit(health_bar, (SCREEN_WIDTH - 150, 250))
            health_text = small_font.render('AI 4 Health', True, WHITE)
            screen.blit(health_text, (SCREEN_WIDTH - 150, 230))

        pygame.display.update()
        clock.tick(10000)

    pygame.quit()

if __name__ == "__main__":
    main()