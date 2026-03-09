
import callbacks
import target_attacker
import os   
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from sentence_transformers import SentenceTransformer
import gymnasium as gym
from GITHUB.Phase_2.RL_phase2 import DataExtractionEnv, StrategyAgent
from stable_baselines3.common.callbacks import CallbackList


# =============================================================================
# ENTRY POINT — MAIN TRAINING LOOP
# =============================================================================
# =============================================================================
# HYPERPARAMETERS
# =============================================================================

total_steps          = 100_000   # Total environment steps for PPO training
lr                   = 5e-5      # Learning rate for Agent 2 (PPO)
clip                 = 0.2       # PPO clip range
ent                  = 0.05      # Entropy coefficient (encourages exploration)
INITIAL_CLIP_RANGE   = 0.2
n_steps_per_update   = 1024      # Steps between each PPO gradient update
N_EPISODES_PER_BATCH = 5         # Episodes per strategy batch (Agent 1 update frequency)


if __name__ == "__main__":

    LOG_DIR = "./ppo_red_team_logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Load models
    print("Loading models...")
    attacker = target_attacker.MultimodalAttackerTools()
    target = target_attacker.Qwen2VLTarget_LoRa()

    # 3. Create shared Strategy Agent (Agent 1)
    strategy_agent_condiviso = StrategyAgent(
        attacker,
        SentenceTransformer('all-MiniLM-L6-v2'),
        batch_size=N_EPISODES_PER_BATCH,
    )

    # 4. Setup training environment with hierarchical support
    env_base = DataExtractionEnv(attacker, target, dataset=dataset,
                                  max_steps=3, use_hierarchical=True)
    env_base.strategy_agent = strategy_agent_condiviso
    env = Monitor(env_base)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=10.0)

    # 5. Setup evaluation environment (shares obs normaliz ation with training)
    test_env_base = DataExtractionEnv(attacker, target, dataset=test_data,
                                       max_steps=3, use_hierarchical=True)
    test_env_base.strategy_agent = strategy_agent_condiviso
    test_env = Monitor(test_env_base)
    test_env = DummyVecEnv([lambda: test_env_base])
    test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False, clip_reward=10.0)
    test_env.obs_rms = env.obs_rms  # Share statistics

    # 6. PPO configuration with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=target_attacker.RedTeamFeatureExtractor,
        features_extractor_kwargs=dict(),
        net_arch=dict(pi=[512, 256], vf=[512, 256]),
    )

    # 7. Setup callbacks for training monitoring
    metrics_callback = callbacks.FullMetricsCallback(
        check_freq=n_steps_per_update, filename="training_metrics.csv", verbose=1)
    attacks_logger = callbacks.BestAttacksLoggerCallback(
        check_freq=32, filename="best_attacks.txt")
    entropy_callback = callbacks.EntropyScheduleCallback(
        initial_ent_coef=0.02, min_ent_coef=0.001, total_timesteps=total_steps)
    dynamic_update_cb = callbacks.DynamicStrategyUpdateCallback(
        update_freq_episodes=N_EPISODES_PER_BATCH, verbose=1)

    callback_list = CallbackList([
        metrics_callback, attacks_logger, entropy_callback, dynamic_update_cb,
    ])

    # 8. Create and train Agent 2 (PPO Prompt Agent)
    agent2 = PPO(
        "MlpPolicy", env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        verbose=1,
        learning_rate=lr,
        n_steps=2048,
        batch_size=128,
        clip_range=clip,
        ent_coef=ent,
    )

    agent2.learn(
        total_timesteps=total_steps,
        tb_log_name="PPO_Prompt_Agent",
        callback=callback_list,
        progress_bar=True,
    )

    final_stats = strategy_agent_condiviso.get_memory_stats()

