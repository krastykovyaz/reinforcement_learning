# 🧠 Deep Crossentropy Method (CEM)

Reinforcement Learning with neural networks using the Cross-Entropy Method. Trains an MLP policy to solve discrete control tasks.

## 🎮 Environments

| Environment | Goal |
|-------------|------|
| `CartPole-v0` | Balance pole (solved) |
| `MountainCar-v0` | Reach flag (reward ≥ -150) |
| `LunarLander-v2` | Land safely (reward ≥ +50) |

## 🏗️ Architecture

- **Policy**: `MLPClassifier` (scikit-learn)
- **Hidden layers**: 20–40 neurons with `tanh` activation
- **Algorithm**: Generate sessions → select elite trajectories → update policy

## 🚀 Usage

```python
# Train
n_sessions, percentile = 100, 70
for i in range(100):
    sessions = [generate_session() for _ in range(n_sessions)]
    states, actions, rewards = zip(*sessions)
    elite_states, elite_actions = select_elites(states, actions, rewards, percentile)
    agent.partial_fit(elite_states, elite_actions)

# Visualize
visualize_mountain_car_live(env, agent)

⚙️ Key Functions
Function	Purpose
generate_session()	Play one episode, return states, actions, reward
select_elites()	Filter top percentile% episodes
agent.partial_fit()	Train on elite transitions

📈 Tips
Increase percentile (80–90) for harder environments

Larger n_sessions reduces reward variance

Use learning_rate='adaptive' for stability

Visualize every 10 iterations to monitor progress

🔧 Dependencies

gymnasium>=0.29.0
scikit-learn>=1.2.0
numpy>=1.21.0
matplotlib>=3.5.0
