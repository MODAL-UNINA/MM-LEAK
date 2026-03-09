import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from sentence_transformers import SentenceTransformer
import gymnasium as gym
from stable_baselines3.common.callbacks import CallbackList
from GITHUB.Phase_1.RL_phase1 import DataExtractionEnv, StrategyAgent
import target_attacker_phase1 as target_attacker
import callbacks

total_steps = 50000
lr = 5e-5
clip = 0.2
ent = 0.05
INITIAL_CLIP_RANGE = 0.2
n_steps_per_update = 1024
N_EPISODES_PER_BATCH = 5

if __name__ == "__main__":


    LOG_DIR = "./ppo_red_team_logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    print("Loading models...")
    attacker = target_attacker.MultimodalAttackerTools()
    target = target_attacker.Qwen2VLTarget()

    # Create shared Strategy Agent
    strategy_agent_shared = StrategyAgent(
        attacker,
        SentenceTransformer('all-MiniLM-L6-v2'),
        batch_size=N_EPISODES_PER_BATCH,
    )

    # Setup training environment
    # [Full env setup with hierarchical strategy integration — see RL_traindata.py]
    
    # PPO policy configuration
    policy_kwargs = dict(
        features_extractor_class=target_attacker.RedTeamFeatureExtractor,
        features_extractor_kwargs=dict(),
        net_arch=dict(pi=[512, 256], vf=[512, 256]),
    )

    # Setup callbacks
    metrics_callback = callbacks.FullMetricsCallback(
        check_freq=n_steps_per_update, verbose=1)
    entropy_callback = callbacks.EntropyScheduleCallback(
        initial_ent_coef=0.02, min_ent_coef=0.001, total_timesteps=total_steps)

    callback_list = CallbackList([metrics_callback, entropy_callback])
