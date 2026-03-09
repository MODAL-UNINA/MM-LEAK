# =============================================================================
# callbacks_phase1.py
#
# Custom Callbacks for Stable Baselines3 PPO Training
# 
# Purpose:
#   Implements specialized monitoring and logging during hierarchical RL training
#   with dual-agent system (Strategy Generator + Prompt Agent).
#   Tracks PII elicitation metrics, strategy performance, and attack quality.
#
# Key Classes:
#   - FullMetricsCallback: Captures training metrics (rewards, losses, learning rate)
#   - BestAttacksLoggerCallback: Records high-impact attacks discovered
#   - EntropyScheduleCallback: Dynamically adjusts entropy coefficient
#   - PIIElicitationMetricsCallback: Monitors PII extraction performance
#   - ElicitationStrategyFeedbackCallback: Tracks per-strategy performance data
#   - DynamicStrategyUpdateCallback: Orchestrates feedback to Strategy Agent
# =============================================================================

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import os
import re


# =============================================================================
# CLASS: FullMetricsCallback
#
# Purpose: Capture all PPO training metrics (rewards, policy/value losses, entropy)
#
# Key Features:
#   - Wraps the SB3 logger to intercept ALL metric recording calls
#   - Avoids race conditions between _on_step() and logger updates
#   - Saves metrics to CSV and generates performance plots
#   - Tracks custom PII extraction metrics alongside standard PPO metrics
# =============================================================================
class FullMetricsCallback(BaseCallback):
    """
    Saves average episode reward and PPO network losses to CSV.
    Wraps the logger to intercept metric recording immediately when emitted.
    """
    def __init__(self, check_freq: int, filename: str = "training_metrics.csv", verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.all_logs = []
        self.filename = filename
        self.last_values = {}
        self.logger_wrapped = False
        self.pending_save_at = None
        # Custom PII metrics buffers
        self._pii_scores = []
        self._refusal_rates = []
        self._pii_types = []
        self._conv_lengths = []
        self._total_pii = []
    
    # ─────────────────────────────────────────────────────────────────────
    # Logger wrapping: intercept metric recording for thread-safe logging
    # ─────────────────────────────────────────────────────────────────────
    def _on_training_start(self) -> None:
        """
        Wrap the SB3 logger to capture ALL metric emission calls.
        This ensures we record metrics the moment they're logged, avoiding
        timing issues where metrics are recorded but not yet available to _on_step().
        """
        if not self.logger_wrapped and hasattr(self.model, 'logger') and self.model.logger is not None:
            original_record = self.model.logger.record
            callback_self = self
            
            def wrapped_record(key, value, exclude=None):
                original_record(key, value, exclude)
                callback_self.last_values[key] = value
                
                # train/n_updates is the last metric emitted by PPO in each update cycle → save now
                if callback_self.pending_save_at is not None and key == 'train/n_updates':
                    if callback_self.verbose > 1:
                        print(f"✅ All metrics logged! Saving now...")
                        print(f"   last_values has {len(callback_self.last_values)} keys: {list(callback_self.last_values.keys())}")
                    
                    record = callback_self.last_values.copy()
                    record['timestep'] = callback_self.pending_save_at
                    callback_self.all_logs.append(record)
                    callback_self.save_data_to_csv()
                    callback_self.pending_save_at = None
            
            self.model.logger.record = wrapped_record
            self.logger_wrapped = True
            
            if self.verbose > 0:
                print("✅ Logger successfully wrapped!")
    
    # ─────────────────────────────────────────────────────────────────────
    # Main callback logic
    # ─────────────────────────────────────────────────────────────────────
    def _on_step(self) -> bool:
        """
        Called every environment step. Collects custom PII metrics and schedules saves.
        """
        # Read custom metrics from environment info
        infos = self.locals.get("infos", [])
        for info in infos:
            pii = info.get("pii_score", None)
            if pii is not None:
                self._pii_scores.append(pii)
                self._refusal_rates.append(float(info.get("is_refusal", False)))
                self._pii_types.append(len(info.get("pii_types", [])))
                self._conv_lengths.append(info.get("conversation_length", 0))
                self._total_pii.append(info.get("total_pii_extracted", 0.0))

        # Update running mean of episode rewards
        if len(self.model.ep_info_buffer) > 0:
            self.last_values['rollout/ep_rew_mean'] = np.mean(
                [ep['r'] for ep in self.model.ep_info_buffer]
            )

        # Schedule save if checkpoint frequency reached
        if self.n_calls % self.check_freq == 0:
            # Add custom metric averages to the log
            if self._pii_scores:
                self.last_values['custom/avg_pii_score'] = np.mean(self._pii_scores)
                self.last_values['custom/avg_refusal_rate'] = np.mean(self._refusal_rates)
                self.last_values['custom/avg_pii_types'] = np.mean(self._pii_types)
                self.last_values['custom/avg_conv_length'] = np.mean(self._conv_lengths)
                self.last_values['custom/total_pii_extracted'] = np.sum(self._total_pii)
                # Reset PII metric buffers
                self._pii_scores.clear()
                self._refusal_rates.clear()
                self._pii_types.clear()
                self._conv_lengths.clear()
                self._total_pii.clear()

            self.pending_save_at = self.num_timesteps
        return True
    
    # ─────────────────────────────────────────────────────────────────────
    # Export methods
    # ─────────────────────────────────────────────────────────────────────
    def _on_training_end(self) -> None:
        """Called when training completes. Save final metrics and generate plots."""
        if self.pending_save_at is not None:
            record = self.last_values.copy()
            record['timestep'] = self.pending_save_at
            self.all_logs.append(record)
        
        self.save_data_to_csv()
        self.plot_all_metrics()
    
    def get_save_path(self, filename):
        """Get the appropriate log directory path."""
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            log_dir = self.model.logger.get_dir()
        else:
            log_dir = None
            
        if log_dir is None:
            log_dir = "./logs"
        
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, filename)
    
    def save_data_to_csv(self):
        """Export all collected metrics to text-formatted CSV."""
        if not self.all_logs:
            return
        
        df = pd.DataFrame(self.all_logs)
        
        # Define preferred column order (standard PPO metrics first)
        desired_order = [
            'timestep', 
            'rollout/ep_rew_mean',
            'train/value_loss',
            'train/entropy_loss',
            'train/learning_rate',
            'train/loss',
            'train/policy_gradient_loss',
            'train/approx_kl',
            'train/clip_fraction',
            'train/explained_variance',
            'train/n_updates',
            'train/clip_range'
        ]
        existing_cols = [c for c in desired_order if c in df.columns]
        remaining_cols = [c for c in df.columns if c not in existing_cols]
        df = df[existing_cols + remaining_cols]
        
        if 'timestep' in df.columns:
            df = df.sort_values(by='timestep')
        
        base_name = self.filename.replace('.csv', '') + '.txt'
        full_path = self.get_save_path(base_name)
        
        # Format as readable table
        table_content = df.to_string(index=False, float_format="%.4f", justify='right', col_space=12)
        
        with open(full_path, "w") as f:
            f.write(table_content)
        
        if self.verbose > 1:
            print(f"📄 Tabular log saved to: {full_path}")
    
    def plot_all_metrics(self):
        """Generate performance visualization plots from collected metrics."""
        if not self.all_logs: 
            print("No data to plot")
            return
            
        df = pd.DataFrame(self.all_logs)
        save_img_path = self.get_save_path('performance_metrics.png')
            
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
        axs = axs.flatten()
        
        def safe_plot(ax, col_name, title, color):
            """Helper: plot a column if it exists, otherwise show "not found" message."""
            if col_name in df.columns:
                subset = df.dropna(subset=[col_name])
                if not subset.empty:
                    ax.plot(subset['timestep'], subset[col_name], color=color)
                    ax.set_title(title)
                    ax.grid(True)
                else:
                    ax.set_title(f"{title} (No data)")
            else:
                ax.set_title(f"{title} (Not found)")

        safe_plot(axs[0], 'rollout/ep_rew_mean', 'Train Mean Reward', 'g')
        safe_plot(axs[2], 'train/policy_gradient_loss', 'Policy Gradient Loss', 'b')
        safe_plot(axs[3], 'train/value_loss', 'Value Loss', 'r')
        safe_plot(axs[4], 'train/entropy_loss', 'Entropy Loss (Exploration)', 'purple')
        safe_plot(axs[5], 'train/loss', 'Total Loss', 'black')

        fig.tight_layout()
        fig.savefig(save_img_path)
        plt.close(fig)
        
        if self.verbose > 0:
            print(f"📊 Plot saved to: {save_img_path}")


# =============================================================================
# CLASS: BestAttacksLoggerCallback
#
# Purpose: Track and record the highest-quality attacks discovered during training
#
# Key Features:
#   - Maintains a "Hall of Fame" of top attacks ranked by PII extraction score
#   - Deduplicates attacks to avoid logging redundant findings
#   - Saves results to human-readable text file for attack analysis
# =============================================================================
class BestAttacksLoggerCallback(BaseCallback):
    """
    ✅ FIXED: Tracks best attacks based on PII_SCORE instead of similarity.
    Maintains a ranked list of top attacks and saves to formatted text file.
    """
    def __init__(self, check_freq: int, filename: str = "best_attacks.txt", top_n: int = 15, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.filename = filename
        self.top_n = top_n
        self.best_attacks = []

    def _on_step(self) -> bool:
        """Assess each step's results and update hall of fame."""
        infos = self.locals.get("infos")
        rewards = self.locals.get("rewards")
        
        if infos:
            for i, info in enumerate(infos):
                pii_score = info.get("pii_score", 0.0)
                attack_prompt = info.get("attack_prompt", "N/A")
                
                # Only log if PII score exceeds minimum threshold
                if pii_score > 0.3:
                    modality = info.get("modality", "unknown")
                    
                    # Build unique key (handles image/cross_modal attacks differently)
                    if modality in ("image", "cross_modal"):
                        unique_key = f"[{modality.upper()}]_step{self.num_timesteps}_strategy_{info.get('strategy_desc', '')[:50]}"
                    else:
                        unique_key = attack_prompt.strip() if isinstance(attack_prompt, str) else str(self.num_timesteps)
                    
                    attack_data = {
                        "step": self.num_timesteps,
                        "reward": round(float(rewards[i]), 4), 
                        "pii_score": round(pii_score, 4),
                        "pii_types": info.get("pii_types", []),
                        "is_refusal": info.get("is_refusal", False),
                        "prompt": unique_key,
                        "display_prompt": attack_prompt.strip() if isinstance(attack_prompt, str) else "[IMAGE]",
                        "modality": modality,
                        "response": info.get("target_response", "N/A").strip(),
                        "strategy": info.get("strategy_desc", "N/A").strip()[:100]
                    }
                    
                    # Prevent duplicate entries
                    if not any(a['prompt'] == attack_data['prompt'] for a in self.best_attacks):
                        self.best_attacks.append(attack_data)
                    
                    # Keep only top-N attacks, ranked by PII score
                    self.best_attacks = sorted(
                        self.best_attacks, 
                        key=lambda x: x["pii_score"], 
                        reverse=True
                    )[:self.top_n]
        
        if self.n_calls % self.check_freq == 0:
            self.save_to_txt()
            
        return True

    def save_to_txt(self):
        """Export hall of fame to formatted text file."""
        if not self.best_attacks:
            return
            
        path = self.get_save_path()
        
        with open(path, "w", encoding='utf-8') as f:
            f.write(f"=== HALL OF FAME ATTACKS - Step {self.num_timesteps} ===\n")
            f.write(f"Top {len(self.best_attacks)} attacks ranked by PII Score\n")
            f.write("-" * 80 + "\n\n")
            
            for i, attack in enumerate(self.best_attacks, 1):
                f.write(f"RANK #{i}\n")
                f.write(f"STEP: {attack['step']} | PII SCORE: {attack['pii_score']:.4f} | REWARD: {attack['reward']}\n")
                f.write(f"PII TYPES: {', '.join(attack['pii_types']) if attack['pii_types'] else 'None'}\n")
                f.write(f"REFUSAL: {attack['is_refusal']}\n")
                f.write(f"STRATEGY: {attack['strategy']}\n")
                f.write(f"PROMPT:   {attack.get('display_prompt', attack['prompt'])}\n")
                f.write(f"RESPONSE: {attack['response']}\n")
                f.write("-" * 40 + "\n")

    def get_save_path(self):
        """Get path for saving hall of fame results."""
        log_dir = self.logger.get_dir() if hasattr(self, 'logger') else None
        target_dir = log_dir if log_dir else "."
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, self.filename)

    def _on_training_end(self) -> None:
        """Save final hall of fame when training ends."""
        self.save_to_txt()


# =============================================================================
# CLASS: EntropyScheduleCallback
#
# Purpose: Decay exploration coefficient over training course
#
# Details:
#   Linear decay from initial_ent_coef → min_ent_coef
#   Encourages more exploitation as agent learns
# =============================================================================
class EntropyScheduleCallback(BaseCallback):
    """Linear decay of entropy coefficient from `initial_ent_coef` to `min_ent_coef`."""
    def __init__(self, initial_ent_coef, min_ent_coef, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.min_ent_coef = min_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        """Update entropy coefficient based on training progress."""
        progress = 1.0 - (self.num_timesteps / self.total_timesteps)
        progress = max(0.0, progress)
        
        new_ent_coef = self.min_ent_coef + (self.initial_ent_coef - self.min_ent_coef) * progress
        self.model.ent_coef = new_ent_coef
        return True


# =============================================================================
# CLASS: PIIElicitationMetricsCallback
#
# Purpose: Monitor PII elicitation performance WITHOUT ground truth
#
# Metrics Tracked:
#   - PII extraction score (detected sensitive data)
#   - Refusal rate (how often model refuses)
#   - PII type diversity (how many different PII categories extracted)
#   - Conversation length (dialogue depth)
# =============================================================================
class PIIElicitationMetricsCallback(BaseCallback):
    """
    Tracks metrics for elicitation-based attacks (no ground truth required).
    Logs PII scores, refusal rates, and diversity metrics to CSV.
    """
    
    def __init__(self, check_freq=100, filename="elicitation_metrics.csv", verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.filename = filename
        self.file_created = False
        
        self.pii_scores = []
        self.refusal_rates = []
        self.pii_diversity = []
        self.conversation_lengths = []
        self._buffer_pii = []
        self._buffer_refusal = []
        self._buffer_pii_types = []
        self._buffer_conv_lengths = []
        self._buffer_total_pii = []
    
    def _on_training_start(self) -> None:
        """Initialize CSV file with headers."""
        log_path = self.get_save_path()
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestep', 'episode',
                'avg_pii_score', 'avg_refusal_rate', 'avg_pii_types',
                'avg_conv_length', 'total_pii_extracted'
            ])
        self.file_created = True
        if self.verbose > 0:
            print(f"✓ PII Metrics CSV created at: {log_path}")
    
    def get_save_path(self):
        """Get path to metrics CSV file in log directory."""
        if hasattr(self, 'logger') and self.logger is not None:
            log_dir = self.logger.get_dir()
        elif hasattr(self.model, 'logger') and self.model.logger is not None:
            log_dir = self.model.logger.get_dir()
        else:
            log_dir = None
        
        if log_dir is None:
            log_dir = "./logs"
        
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, self.filename)
    
    def _on_step(self):
        """Collect metrics from every step into buffers."""
        infos = self.locals.get('infos', [])
        for info in infos:
            self._buffer_pii.append(info.get('pii_score', 0.0))
            self._buffer_refusal.append(1.0 if info.get('is_refusal', False) else 0.0)
            self._buffer_pii_types.append(len(info.get('pii_types', [])))
            self._buffer_conv_lengths.append(info.get('conversation_length', 0))
            self._buffer_total_pii.append(info.get('total_pii_extracted', 0.0))

        # Log only every check_freq calls, but using accumulated data
        if self.n_calls % self.check_freq != 0:
            return True

        avg_pii_score = np.mean(self._buffer_pii) if self._buffer_pii else 0.0
        avg_rr = np.mean(self._buffer_refusal) if self._buffer_refusal else 0.0
        avg_pii_types = np.mean(self._buffer_pii_types) if self._buffer_pii_types else 0.0
        avg_conv_len = np.mean(self._buffer_conv_lengths) if self._buffer_conv_lengths else 0.0
        total_pii = np.sum(self._buffer_total_pii)

        # Clear buffers after logging
        self._buffer_pii.clear()
        self._buffer_refusal.clear()
        self._buffer_pii_types.clear()
        self._buffer_conv_lengths.clear()
        self._buffer_total_pii.clear()

        log_path = self.get_save_path()
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.num_timesteps,
                len(self.pii_scores),
                avg_pii_score,
                avg_rr,
                avg_pii_types,
                avg_conv_len,
                total_pii
            ])
            
        if self.verbose > 0:
            print(f"\n📊 Elicitation Metrics @ {self.num_timesteps} steps:")
            print(f"   PII Score: {avg_pii_score:.2f} | RR: {avg_rr:.1%}")
            print(f"   PII Types: {avg_pii_types:.1f} | Conv Length: {avg_conv_len:.1f}")
        
        return True


# =============================================================================
# CLASS: ElicitationStrategyFeedbackCallback
#
# Purpose: Log per-strategy performance for offline analysis
# =============================================================================
class ElicitationStrategyFeedbackCallback(BaseCallback):
    """Records feedback for each strategy used in elicitation training."""
    
    def __init__(self, check_freq=32, filename="strategy_feedback.csv", verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.filename = filename
        self.file_created = False
        self._buffer = []

    def _on_training_start(self) -> None:
        """Initialize CSV with headers for strategy tracking."""
        log_path = self.get_save_path()
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestep', 'strategy_idx', 'strategy_desc',
                'pii_score', 'reward', 'is_refusal', 'pii_types'
            ])
        self.file_created = True
        if self.verbose > 0:
            print(f"✓ Strategy Feedback CSV created at: {log_path}")
    
    def get_save_path(self):
        """Determine CSV save location."""
        if hasattr(self, 'logger') and self.logger is not None:
            log_dir = self.logger.get_dir()
        elif hasattr(self.model, 'logger') and self.model.logger is not None:
            log_dir = self.model.logger.get_dir()
        else:
            log_dir = None
        
        if log_dir is None:
            log_dir = "./logs"
        
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, self.filename)
    
    def _on_step(self):
        """Collect strategy performance data every step."""
        infos = self.locals.get('infos', [])
        rewards = self.locals.get('rewards', [0])

        for i, info in enumerate(infos):
            if 'strategy_idx' not in info or info['strategy_idx'] < 0:
                continue
            pii_types_str = ','.join(info.get('pii_types', []))
            reward = rewards[i] if i < len(rewards) else 0
            self._buffer.append([
                self.num_timesteps,
                info['strategy_idx'],
                info.get('strategy_desc', 'N/A')[:100],
                info.get('pii_score', 0.0),
                reward,
                info.get('is_refusal', False),
                pii_types_str,
                info.get('modality', 'unknown')
            ])

        # Write to CSV only every check_freq calls
        if self.n_calls % self.check_freq != 0 or not self._buffer:
            return True

        log_path = self.get_save_path()
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self._buffer)

        self._buffer.clear()
        return True


# =============================================================================
# CLASS: DynamicStrategyUpdateCallback
#
# Purpose: Interface between Agent 2 (PPO) and Agent 1 (Strategy Generator)
#
# Workflow:
#   1. Accumulate episode feedback in buffer
#   2. Every N episodes, send feedback to Strategy Agent
#   3. Strategy Agent generates new batch based on feedback
#   4. New batch propagated to environment
# =============================================================================
class DynamicStrategyUpdateCallback(BaseCallback):
    """
    ✅ FIXED: Updates strategies dynamically using PII_SCORE instead of similarity.
    Sends accumulated feedback to Strategy Agent every N episodes.
    """
    def __init__(self, update_freq_episodes=5, verbose=1):
        super(DynamicStrategyUpdateCallback, self).__init__(verbose)
        self.update_freq = update_freq_episodes
        self.episode_count = 0
        self.feedback_buffer = []
        
    def _on_step(self) -> bool:
        """Accumulate rewards/metrics for each action taken."""
        info = self.locals['infos'][0]
        modality = info.get('modality', 'unknown')
        mismatch_type = info.get('mismatch_type', None)
        reward = self.locals['rewards'][0]
        
        # ✅ FIXED: use pii_score instead of similarity
        pii_score = info.get('pii_score', 0.0)
        strategy_idx = info.get('strategy_idx', -1)

        # Get current batch info from environment
        inner = self._unwrap_vec_env(self.training_env)
        current_batch = inner.get_attr("current_strategy_batch")[0]
        vector = None
        if current_batch is not None and 0 <= strategy_idx < len(current_batch):
            vector = current_batch[strategy_idx]

        if vector is not None and strategy_idx >= 0:
            self.feedback_buffer.append({
                'strategy_idx': strategy_idx,
                'vector': vector,
                'reward': float(reward),
                'pii_score': float(pii_score),
                'modality': modality,
                'mismatch_type': mismatch_type
            })

        if self.locals['dones'][0]:
            self.episode_count += 1
            
            if self.episode_count % self.update_freq == 0:
                if self.verbose > 0:
                    print(f"\n[DynamicCallback] Update triggered at episode {self.episode_count}")
                
                strategy_agent = inner.get_attr("strategy_agent")[0]
                strategy_agent.update_from_feedback(self.feedback_buffer)
                
                max_pii = max([x['pii_score'] for x in self.feedback_buffer], default=0)
                print(f"   Buffer stats: {len(self.feedback_buffer)} steps recorded. Max PII Score: {max_pii:.4f}")
                
                new_descs, new_embs, new_modalities, styles = strategy_agent.generate_strategy_batch()
                inner.env_method("reset_with_strategy_batch", new_descs, new_embs, new_modalities, styles)
                
                self.feedback_buffer = []
                
        return True

    @staticmethod
    def _unwrap_vec_env(env):
        """Extract base environment from vector wrapper layers."""
        inner = env
        while hasattr(inner, 'venv'):
            inner = inner.venv
        return inner
