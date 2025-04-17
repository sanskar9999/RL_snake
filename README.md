# Advanced Snake AI: D3QN Self-Play and Heuristic Competition 🐍

This repository contains two distinct Python-based Snake AI projects:

1.  **`snake_4.py`**: A sophisticated 1-vs-1 Snake AI trained using Reinforcement Learning (Dueling Double Deep Q-Network with Prioritized Experience Replay and Self-Play).
2.  **`o.py`**: A Pygame simulation featuring four competing Snake AIs (one simple heuristic, three with evolving heuristic parameters) battling for food and survival.

---

## 1. `snake_4.py`: 1v1 Dueling Double DQN Snake

This script implements a Deep Reinforcement Learning agent designed to play a 1-vs-1 version of the classic Snake game. The goal is to train an agent that can strategically navigate, collect food, and outmaneuver its opponent.

### Features

*   **Reinforcement Learning Algorithm**: Dueling Double Deep Q-Network (D3QN) for stable and efficient learning.
*   **Prioritized Experience Replay (PER)**: Focuses learning on more significant transitions.
*   **Self-Play Training**: The agent trains against versions of itself stored in an opponent pool, promoting robust strategy development.
*   **Compact State Representation**: A 32-dimensional vector encoding crucial information about the game state (snake positions, directions, lengths, raycasts, food location).
*   **Reward Shaping**: Carefully designed rewards for eating food, surviving, killing the opponent, and penalties for dying.
*   **Hyperparameter Tuning**: Configurable parameters for learning rate, batch size, exploration (epsilon), memory size, etc.
*   **Checkpointing**: Save and load agent progress (network weights, optimizer state, epsilon).
*   **Weights & Biases Integration**: Log training metrics (scores, loss, epsilon, etc.) for visualization and tracking (optional).
*   **Testing Mode**: Evaluate trained agents against different opponents (itself, random).
*   **Console Rendering**: Basic text-based rendering during testing mode.

### Technical Details

*   **Network Architecture**: Dueling DQN separates the value and advantage streams for better Q-value estimation.
*   **State Vector (32D)**:
    *   *Self:* Normalized Head Pos (2), One-Hot Direction (4), Normalized Length (1), Normalized Raycasts (8 directions) for self-collision (8) = **15 dims**
    *   *Opponent:* Normalized Head Pos (2), One-Hot Direction (4), Normalized Length (1), Normalized Raycasts (8 directions) for opponent collision (8) = **15 dims**
    *   *Food:* Normalized Relative Position (wrapped) (2) = **2 dims**
*   **Actions**: 4 discrete actions (Up, Right, Down, Left). Invalid moves (moving back into oneself) are masked.
*   **Environment**: Custom 1v1 Snake environment with grid wrapping.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install torch numpy matplotlib tqdm wandb
    ```
    *(Note: Ensure you have a compatible PyTorch version installed, potentially with CUDA support if you have a GPU.)*

### Usage

**Training:**

*   **Start training from scratch:**
    ```bash
    python snake_4.py --mode train --episodes 20000 --checkpoint_dir snake_models_v1
    ```
*   **Continue training from a checkpoint:**
    ```bash
    python snake_4.py --mode train --episodes 40000 --load_model snake_models_v1/checkpoint_ep_20000.pt --checkpoint_dir snake_models_v1
    ```
*   **Disable Weights & Biases logging:**
    ```bash
    python snake_4.py --mode train --no_wandb
    ```

**Testing:**

*   **Test a trained agent against itself:**
    ```bash
    python snake_4.py --mode test --load_model snake_models_v1/best_agent_....pt --games 20 --opponent agent
    ```
*   **Test against itself with console rendering:**
    ```bash
    python snake_4.py --mode test --load_model snake_models_v1/final_agent.pt --games 10 --opponent agent --render
    ```
*   **Test against a random opponent:**
    ```bash
    python snake_4.py --mode test --load_model snake_models_v1/final_agent.pt --games 50 --opponent random
    ```


### Configuration

Key hyperparameters and settings can be adjusted directly within the `snake_4.py` script:

*   `GRID_WIDTH`, `GRID_HEIGHT`
*   `BATCH_SIZE`, `GAMMA`, `EPS_START`, `EPS_END`, `EPS_DECAY`, `TARGET_UPDATE`, `LEARNING_RATE`, `MEMORY_SIZE`
*   `PER_ALPHA`, `PER_BETA_START`, `PER_BETA_FRAMES`
*   `OPPONENT_POOL_SIZE`, `UPDATE_OPPONENT_POOL_FREQ`
*   Reward values within the `SnakeEnvironment.step` method.

---

## 2. `o.py`: 4-Snake AI Heuristic Competition

This script uses Pygame to simulate a competition between four different Snake AIs within the same environment. One AI uses a simple heuristic, while the other three employ more complex, evolving heuristics.

### Features

*   **Pygame Visualization**: Real-time graphical display of the snake competition.
*   **Multiple Competing AIs**: Four snakes (O, O1, O2, O3) with distinct behaviors.
*   **Heuristic AI Logic**: Snakes make decisions based on weighted factors like food proximity, safety (collision avoidance), and aggression (moving towards opponents).
*   **Simple Evolutionary Mechanism**: Snakes O1, O2, and O3 mutate their heuristic parameters (weights, look-ahead, aggression) upon respawning, potentially improving over time based on their score relative to their personal best.
*   **Dynamic Environment**: Snakes respawn after death, food is replenished, and snakes drop food upon death.
*   **Score Tracking**: Displays current scores for each snake.
*   **Parameter Display**: Shows the current and best-ever parameters for the evolving snakes (O1, O2, O3) and tracks the overall best-performing snake configuration found during the simulation run.

### AI Strategies

*   **SnakeO**: Basic AI focusing on the nearest food with simple collision avoidance.
*   **SnakeO1, SnakeO2, SnakeO3**: More advanced AIs that evaluate potential moves based on:
    *   `food_weight`: Importance of moving towards food.
    *   `safety_weight`: Importance of avoiding collisions (self and others) using look-ahead.
    *   `look_ahead`: How many steps into the future to check for safety.
    *   `aggression`: Tendency to move towards other snake heads.
    *   These parameters mutate slightly upon respawn, guided by whether the snake surpassed its previous best score. SnakeO3 also incorporates a 'hunger' factor.

### Setup

1.  **Clone the repository (if not already done):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  **Create a virtual environment (recommended, if not already done):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install pygame numpy
    ```

### Usage

*   **Run the simulation:**
    ```bash
    python o.py
    ```
    The simulation will start, showing the snakes competing in the Pygame window. Close the window to stop the simulation.

### Configuration

Key settings can be adjusted within the `o.py` script:

*   `SCREEN_WIDTH`, `SCREEN_HEIGHT`, `GRID_SIZE`
*   `RESPAWN_TIME`
*   Initial parameters and best scores stored in `global_best_o1`, `global_best_o2`, `global_best_o3`.
*   Mutation rates and logic within the `reset` and `mutate_parameters` methods of SnakeO1, O2, and O3.

---

## License

This project is licensed under the MIT License - see the LICENSE file (if available) for details. 
