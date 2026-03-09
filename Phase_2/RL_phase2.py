# =============================================================================
# RL_traindata.py
#
# Phase 2 Hierarchical RL Red-Teaming Training Script
#
# Architecture:
#   Agent 1 (StrategyAgent): Generates and evolves high-level attack strategies
#   Agent 2 (PPO MlpPolicy): Learns to craft concrete adversarial prompts
#
# Target Model: Qwen2-VL-7B-Instruct (optionally fine-tuned with LoRA)
# Attacker Model: MultimodalAttackerTools (text + image generation)
# Dataset: Multimodal canary JSONL with text and image PII samples
#
# Training Loop:
#   For N total timesteps:
#     - PPO training step (Agent 2) updates prompt agent policy
#     - Every M episodes: DynamicStrategyUpdateCallback sends feedback to Agent 1
#     - Agent 1 generates new strategy batch based on episode performance
#     - Metrics logged: PII scores, refusal rates, diversity metrics
# =============================================================================


from stable_baselines3.common.utils import get_latest_run_id
from sentence_transformers import SentenceTransformer
import gymnasium as gym
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
from stable_baselines3 import PPO
from sklearn.cluster import DBSCAN
from collections import deque
from gymnasium import spaces
import numpy as np



# =============================================================================
# DATASET LOADING & NORMALIZATION
# =============================================================================

def normalize_sample(sample: dict) -> dict:
    """
    Normalize raw JSONL sample into unified format.
    Handles both modalities:
      - Text: {"source": "canary_text", "is_image": False, "prefix": "SECRET-...", "target": "full_secret"}
      - Image: {"source": "canary_image", "is_image": True, "task_type": "canary_inpainting",
                "input": "<base64_masked>", "target": "<base64_original>", "secret_extracted": "SECRET-..."}
    
    Returns: Unified dict with: prefix, target, task_type, input, is_image, source
    """
    is_image = sample.get('is_image', False)

    if is_image:
        return {
            'prefix':    sample.get('secret_extracted', ''),  # Text to extract from image
            'target':    sample.get('target', ''),             # Original unmasked image (base64)
            'task_type': sample.get('task_type', 'canary_inpainting'),
            'input':     sample.get('input', ''),              # Masked image (base64)
            'is_image':  True,
            'source':    sample.get('source', 'canary_image'),
        }
    else:
        return {
            'prefix':    sample.get('prefix', ''),
            'target':    sample.get('target', ''),
            'task_type': 'text',
            'input':     None,
            'is_image':  False,
            'source':    sample.get('source', 'canary_text'),
        }



# =============================================================================
# AGENT 1 — STRATEGY GENERATOR
#
# Maintains memory of tested strategies and generates improved batches using:
#   - Elite retention: perturbs best strategy from previous batch
#   - Evolutionary mutation: LLM-driven changes to high-scoring strategies
#   - Novelty search: DBSCAN clustering to explore underexplored regions
#   - UCB1 selection: balances exploration across text/image/cross_modal
# =============================================================================

class StrategyAgent:
    """
    See RL_phase1.py for full detailed documentation on StrategyAgent.
    This Phase 2 version is identical in function but integrates with 
    phase-aware attack generation.
    """
    def __init__(self, attacker_model, embedder, batch_size: int = 3):
        self.attacker_model = attacker_model
        self.embedder = embedder
        self.batch_size = batch_size
        
        # Strategy memory and embeddings
        self.strategy_memory = {}
        self.embedding_cache = {}
        self.last_batch_results = []
        
        # Stagnation tracking
        self.stagnation_counter = 0
        self.best_similarity_ever = 0.0
        self.exploration_rate = 0.3
        self.recent_best_scores = deque(maxlen=5)
        
        # Per-modality scores
        self.modality_scores = {"text": [], "image": [], "cross_modal": []}
        self.image_style_scores = {}
        self.best_image_styles = []
        
        # LLM temperature
        self.base_temperature = 0.7
        self.current_temperature = 0.7
        
        # Cross-modal tracking
        self.mismatch_type_scores = {}
        
        # Batch state
        self.current_batch_descs = []
        self.current_batch_modalities = []
        self.current_batch_styles = []

    def generate_strategy_batch(self):
        """Generate batch of attack strategies (implementation detail — see RL_phase1.py)."""
        # Full implementation: strategy retention (elite) → mutation → novelty → fill
        # Returns: (descriptions, embeddings, modalities, styles)
        # [Implementation omitted for brevity - identical to RL_phase1.py]
        pass

    def update_from_feedback(self, feedback_data: list):
        """Update strategy memory based on episode performance."""
        # Processes feedback from Agent 2 (PPO)
        # Updates self.last_batch_results with best strategies per batch
        # Tracks modality performance via UCB1 algorithm
        # [Implementation omitted for brevity - identical to RL_phase1.py]
        pass

    def get_memory_stats(self) -> dict:
        """Return summary statistics of strategy memory."""
        # Returns total strategies tested and top-5 performers
        # [Implementation omitted for brevity - identical to RL_phase1.py]
        pass


# =============================================================================
# GYMNASIUM ENVIRONMENT — DataExtractionEnv
#
# Wraps target model as Gymnasium environment for PPO training.
#
# Observation space (1568-dim):
#   [similarity(1), length(1), entropy(1), action_onehot(8),
#    strategy_embedding(384), prefix_embedding(384),
#    response_embedding(384), last_prompt_embedding(384)]
#
# Action space: [pressure(5), framing(3)] in hierarchical mode
# =============================================================================

class DataExtractionEnv(gym.Env):
    """
    Gymnasium environment for PII extraction attacks.
    
    Full implementation available in original file.
    Key methods:
      - reset(): Initialize episode with new strategy and dataset sample
      - step(action): Execute one attack turn, return observation and reward
      - _calculate_hybrid_similarity(): Semantic + verbatim + LCS similarity
      - _calculate_extraction_reward(): Dense reward for PII quality
    """

    def __init__(self, attacker_model, target_model, dataset=WIKI_DATASET,
                 max_steps: int = 3, use_hierarchical: bool = True):
        gym.Env.__init__(self)
        
        self.attacker_model = attacker_model
        self.target_model = target_model
        self.dataset = dataset
        self.max_steps = max_steps
        self.current_step = 0
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.use_hierarchical = use_hierarchical

        # Hierarchical state management
        if self.use_hierarchical:
            self.strategy_agent = StrategyAgent(self.attacker_model, self.embedder, batch_size=3)
            self.current_strategy_batch = None
            self.current_strategy_idx = 0

        # Initialize episode state attributes
        self.current_sample = None
        self.current_prefix = ""
        self.current_ground_truth = ""
        self.current_strategy = None
        self.current_strategy_text = "N/A"
        self.current_strategy_modality = "text"

        # Action and observation spaces
        if self.use_hierarchical:
            self.action_space = spaces.MultiDiscrete([5, 3])
        else:
            self.action_space = spaces.MultiDiscrete([6, 5, 3])

        self.action_size = int(np.sum(self.action_space.nvec))

        # Observation space: 1568-dim vector
        self.obs_dim = 1 + 1 + 1 + 8 + 384 + 384 + 384 + 384  # Total 1568
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        # Full implementation: sample new dataset item, select strategy, initialize observation
        # [Implementation omitted - see original file for details]
        pass

    def step(self, action):
        """Execute one training step."""
        # Key logic:
        # 1. Decode action (pressure, framing)
        # 2. Generate attack (text or image) using strategy
        # 3. Get target response
        # 4. Compute reward and similarity
        # 5. Return obs, reward, terminated, truncated, info
        # [Full implementation in original file]
        pass


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def linear_schedule(initial_value: float):
    """
    Linear learning rate decay schedule.
    Maps progress_remaining ∈ [1, 0] → learning_rate
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func
