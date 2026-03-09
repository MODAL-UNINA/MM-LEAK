# =============================================================================
# callbacks.py
#
# Extended Phase 2 Callbacks for PPO Training with Advanced Monitoring
#
# These callbacks integrate with SB3 training loop to:
# - Track multimodal attack performance metrics
# - Maintain hall of fame of best attacks
# - Implement dynamic strategy updates via Agent 1 feedback
# - Schedule entropy coefficient decay for exploration control
# - Generate comprehensive evaluation reports
#
# All callbacks inherit from BaseCallback (SB3 framework).
# =============================================================================

import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
from collections import defaultdict, OrderedDict
from stable_baselines3.common.callbacks import BaseCallback
import re


# =============================================================================
# CALLBACK 1: FullMetricsCallback
#
# Purpose: Capture ALL metrics from SB3 training and custom environment
# Features: CSV export, metric buffering, plot generation
# =============================================================================

class FullMetricsCallback(BaseCallback):
    """
    Comprehensive training metrics logger.
    
    Intercepts metrics emitted by SB3 PPO trainer and custom environment info.
    Wraps internal logger to capture ALL emissions, not just final updates.
    Exports to CSV and generates periodic plots.
    
    Captured metrics:
    - PPO metrics: policy_gradient_loss, value_loss, entropy_loss, explained_variance
    - Custom metrics: pii_score, refusal_rate, pii_types, total_pii_extracted
    - Training state: episode, timesteps, learning_rate
    """
    
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.metrics_buffer = defaultdict(list)
        self.step_counter = 0
        self.csv_path = None
        self.csv_writer = None
        self.csv_file = None
        
    def _on_step(self) -> bool:
        """Called at every training step."""
        self.step_counter += 1
        
        # Extract SB3 metrics from logger
        if self.model.logger.name_to_value:
            for key, value in self.model.logger.name_to_value.items():
                self.metrics_buffer[key].append(value)
        
        # Extract custom environment info
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'buf_infos'):
            for info in self.model.env.buf_infos:
                if info:
                    for key, value in info.items():
                        self.metrics_buffer[f"env/{key}"].append(value)
        
        # Periodic flush to CSV
        if self.step_counter % self.check_freq == 0:
            self._flush_metrics()
            if self.verbose > 0:
                print(f"   [Metrics {self.step_counter} steps]")
        
        return True
    
    def _on_training_start(self) -> None:
        """Initialize CSV file at training start."""
        self.csv_path = Path(self.model.logger.dir or "./logs") / "metrics.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=['timestamp', 'step'])
        self.csv_writer.writeheader()
    
    def _flush_metrics(self):
        """Write accumulated metrics to CSV."""
        if not self.metrics_buffer or self.csv_writer is None:
            return
        
        row = {
            'timestamp': datetime.now().isoformat(),
            'step': self.num_timesteps
        }
        row.update({k: np.mean(v) if v else 0.0 for k, v in self.metrics_buffer.items()})
        
        if self.csv_writer:
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        
        self.metrics_buffer.clear()
    
    def _on_training_end(self) -> None:
        """Finalize CSV and generate summary plots."""
        if self.csv_file:
            self.csv_file.close()
        
        # Generate plots if CSV has data
        if self.csv_path and self.csv_path.exists():
            self._generate_plots()
    
    def _generate_plots(self):
        """Generate matplotlib plots from CSV metrics."""
        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return
            
            # Filter numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'step' in numeric_cols:
                numeric_cols.remove('step')
            
            # Plot loss metrics
            loss_cols = [c for c in numeric_cols if 'loss' in c.lower()]
            if loss_cols:
                plt.figure(figsize=(12, 4))
                for col in loss_cols:
                    plt.plot(df['step'], df[col], label=col)
                plt.xlabel('Timesteps')
                plt.ylabel('Loss')
                plt.legend()
                plt.title('Training Losses')
                plt.savefig(self.csv_path.parent / 'losses.png')
                plt.close()
            
            # Plot PII metrics
            pii_cols = [c for c in numeric_cols if 'pii' in c.lower() or 'refusal' in c.lower()]
            if pii_cols:
                plt.figure(figsize=(12, 4))
                for col in pii_cols:
                    plt.plot(df['step'], df[col], label=col)
                plt.xlabel('Timesteps')
                plt.ylabel('Score')
                plt.legend()
                plt.title('PII Extraction Metrics')
                plt.savefig(self.csv_path.parent / 'pii_metrics.png')
                plt.close()
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Error generating plots: {e}")


# =============================================================================
# CALLBACK 2: BestAttacksLoggerCallback
#
# Purpose: Maintain hall of fame of highest-scoring attacks
# Features: Deduplication, ranking by score, periodic report generation
# =============================================================================

class BestAttacksLoggerCallback(BaseCallback):
    """
    Tracks and logs the top N attack samples.
    
    Maintains an ordered buffer of best attacks ranked by PII score.
    Deduplication strategy:
    - Text attacks: deduplicated by response text prefix (first 50 chars)
    - Image attacks: deduplicated by image hash
    - Cross-modal: deduplicated by combination of text + image source
    
    Periodically exports report to JSON with full attack details.
    """
    
    def __init__(self, top_k=15, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.top_k = top_k
        self.check_freq = check_freq
        self.best_attacks = OrderedDict()  # key -> attack_info
        self.seed_set = set()  # For deduplication
        self.step_counter = 0
        
    def _on_step(self) -> bool:
        """Process new attacks during training."""
        self.step_counter += 1
        
        # Extract attacks from environment info
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'buf_infos'):
            for info_dict in self.model.env.buf_infos:
                if not info_dict:
                    continue
                
                attack_data = info_dict.get('attack_data')
                if attack_data:
                    score = float(info_dict.get('pii_score', 0.0))
                    modality = info_dict.get('modality', 'unknown')
                    
                    # Compute unique key for deduplication
                    unique_key = self._compute_unique_key(attack_data, modality)
                    
                    if unique_key not in self.seed_set:
                        self.best_attacks[unique_key] = {
                            "score": score,
                            "modality": modality,
                            "attack": attack_data,
                            "step": self.num_timesteps,
                            "unique_key": unique_key,
                        }
                        self.seed_set.add(unique_key)
        
        # Keep only top K
        sorted_attacks = sorted(
            self.best_attacks.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        self.best_attacks = OrderedDict(sorted_attacks[:self.top_k])
        
        # Periodic report
        if self.check_freq > 0 and self.step_counter % self.check_freq == 0:
            self._save_report()
        
        return True
    
    def _compute_unique_key(self, attack_data: dict, modality: str) -> str:
        """Deduplicate attacks based on modality."""
        if modality == 'text':
            prompt = attack_data.get('prompt', '')
            return f"text_{hash(prompt)}"
        elif modality == 'image':
            # Hash of image source (avoiding pixel-level comparison)
            img_desc = attack_data.get('image_desc', '')
            return f"image_{hash(img_desc)}"
        else:  # cross_modal
            prompt = attack_data.get('prompt', '')
            img_desc = attack_data.get('image_desc', '')
            return f"cross_{hash((prompt, img_desc))}"
    
    def _save_report(self):
        """Export hall of fame to JSON report."""
        if not self.best_attacks:
            return
        
        report_path = Path(self.model.logger.dir or "./logs") / "best_attacks.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_timesteps": self.num_timesteps,
            "top_attacks": [
                {
                    "rank": idx + 1,
                    "score": float(item['score']),
                    "modality": item['modality'],
                    "discovered_at_step": item['step'],
                    "attack_summary": str(item['attack'])[:200],
                }
                for idx, (_, item) in enumerate(self.best_attacks.items())
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        if self.verbose > 0:
            top = list(self.best_attacks.values())[0]
            print(f"   Best attack (score={top['score']:.4f}, {top['modality']})")


# =============================================================================
# CALLBACK 3: EntropyScheduleCallback
#
# Purpose: Decay exploration coefficient over training to shift from exploration to exploitation
# =============================================================================

class EntropyScheduleCallback(BaseCallback):
    """
    Linear entropy coefficient decay schedule.
    
    Entropy coefficient controls exploration in PPO. High entropy = more stochasticity.
    Schedule: from initial_ent_coef → min_ent_coef over total_timesteps.
    
    Applied at every log step (typically every n_steps=1024 updates).
    """
    
    def __init__(self, initial_ent_coef=0.02, min_ent_coef=0.001, total_timesteps=50000):
        super().__init__(verbose=0)
        self.initial_ent_coef = initial_ent_coef
        self.min_ent_coef = min_ent_coef
        self.total_timesteps = total_timesteps
    
    def _on_step(self) -> bool:
        """Update entropy coefficient based on progress."""
        progress = self.num_timesteps / self.total_timesteps
        
        # Linear interpolation from initial → min
        new_ent_coef = (
            self.initial_ent_coef - 
            (self.initial_ent_coef - self.min_ent_coef) * progress
        )
        
        # Apply to model
        self.model.ent_coef = max(new_ent_coef, self.min_ent_coef)
        
        return True


# =============================================================================
# CALLBACK 4: StrategyFeedbackCallback
#
# Purpose: Aggregate episode performance per strategy and send to Agent 1
# =============================================================================

class StrategyFeedbackCallback(BaseCallback):
    """
    Collects per-strategy performance metrics.
    
    Groups episodes by strategy index and logs:
    - Average PII score per strategy
    - Refusal rates per strategy
    - Reward distribution
    
    Used by Agent 1 (StrategyAgent) via update_from_feedback() call.
    """
    
    def __init__(self, check_freq=5000):
        super().__init__(verbose=0)
        self.check_freq = check_freq
        self.strategy_metrics = defaultdict(lambda: {
            'scores': [],
            'rewards': [],
            'refusals': [],
            'pii_types': defaultdict(int),
        })
        self.step_counter = 0
    
    def _on_step(self) -> bool:
        """Accumulate strategy-level metrics."""
        self.step_counter += 1
        
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'buf_infos'):
            for info in self.model.env.buf_infos:
                if not info:
                    continue
                
                strategy_idx = info.get('strategy_idx')
                if strategy_idx is not None:
                    metrics = self.strategy_metrics[strategy_idx]
                    metrics['scores'].append(info.get('pii_score', 0.0))
                    metrics['rewards'].append(info.get('reward', 0.0))
                    metrics['refusals'].append(int(info.get('is_refusal', False)))
                    
                    pii_types = info.get('pii_types', [])
                    for ptype in pii_types:
                        metrics['pii_types'][ptype] += 1
        
        if self.check_freq > 0 and self.step_counter % self.check_freq == 0:
            self._report_strategy_performance()
        
        return True
    
    def _report_strategy_performance(self):
        """Log per-strategy statistics."""
        if not self.strategy_metrics:
            return
        
        for strat_idx, metrics in self.strategy_metrics.items():
            if metrics['scores']:
                avg_score = np.mean(metrics['scores'])
                avg_reward = np.mean(metrics['rewards'])
                refusal_rate = np.mean(metrics['refusals'])
                print(f"   Strategy {strat_idx}: score={avg_score:.4f}, "
                      f"reward={avg_reward:.2f}, refusal={refusal_rate:.2%}")


# =============================================================================
# CALLBACK 5: SimilarityEvalCallback
#
# Purpose: Monitor hybrid similarity scores and embedding quality
# =============================================================================

class SimilarityEvalCallback(BaseCallback):
    """
    Tracks similarity metric distributions.
    
    Logs per-timestep:
    - Mean/std of hybrid similarity (semantic + verbatim + LCS)
    - Similarity percentile distribution
    - Embedding quality metrics
    """
    
    def __init__(self, check_freq=1000):
        super().__init__(verbose=0)
        self.check_freq = check_freq
        self.similarity_buffer = []
        self.step_counter = 0
    
    def _on_step(self) -> bool:
        """Accumulate similarity scores."""
        self.step_counter += 1
        
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'buf_infos'):
            for info in self.model.env.buf_infos:
                if info and 'similarity' in info:
                    self.similarity_buffer.append(float(info['similarity']))
        
        if self.check_freq > 0 and self.step_counter % self.check_freq == 0:
            self._report_similarity()
            self.similarity_buffer.clear()
        
        return True
    
    def _report_similarity(self):
        """Log similarity distribution stats."""
        if not self.similarity_buffer:
            return
        
        data = np.array(self.similarity_buffer)
        print(f"   Similarity: μ={np.mean(data):.4f}, σ={np.std(data):.4f}, "
              f"p25={np.percentile(data, 25):.4f}, p75={np.percentile(data, 75):.4f}")


# =============================================================================
# CALLBACK 6: DynamicStrategyUpdateCallback
#
# Purpose: Send accumulated feedback to Agent 1 every N episodes
# Orchestrates feedback loop between Agent 2 (RL) and Agent 1 (LLM)
# =============================================================================

class DynamicStrategyUpdateCallback(BaseCallback):
    """
    Implements feedback loop between agents.
    
    Every N training episodes:
    1. Collect all episode infos from replay buffer
    2. Aggregate by strategy index
    3. Call strategy_agent.update_from_feedback()
    
    This allows Agent 1 to adapt strategy generation based on actual 
    Agent 2 performance in the environment.
    
    Must be provided with reference to shared StrategyAgent instance.
    """
    
    def __init__(self, strategy_agent, update_interval=5000, verbose=0):
        super().__init__(verbose=verbose)
        self.strategy_agent = strategy_agent
        self.update_interval = update_interval
        self.step_counter = 0
        self.episode_infos = []
    
    def _on_step(self) -> bool:
        """Accumulate episode infos."""
        self.step_counter += 1
        
        # Collect infos
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'buf_infos'):
            self.episode_infos.extend(
                [info for info in self.model.env.buf_infos if info]
            )
        
        # Periodic feedback to Agent 1
        if self.update_interval > 0 and self.step_counter % self.update_interval == 0:
            if self.episode_infos:
                self.strategy_agent.update_from_feedback(self.episode_infos)
                self.episode_infos.clear()
                
                if self.verbose > 0:
                    stats = self.strategy_agent.get_memory_stats()
                    print(f"\n   Agent 1 updated: {stats['total_strategies']} strategies learned")
                    for s in stats['best_strategies'][:3]:
                        print(f"      #{s['id']}: avg_score={s['avg_score']:.4f}")
        
        return True
