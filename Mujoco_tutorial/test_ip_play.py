import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make(
    "InvertedPendulum-v5",
    render_mode="human"
)

model = PPO.load("ppo_inverted_pendulum")

obs, info = env.reset()

while True:

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()