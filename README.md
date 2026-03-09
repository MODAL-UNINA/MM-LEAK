# MM-LEAK: Evaluating Multimodal Alignment Robustness via Reinforcement Learning Agents
## Abstract
Reinforcement Learning (RL) is widely adopted to align MLLMs with safety constraints but, despite these advances, the robustness of such alignment pipelines against privacy leakage remains insufficiently characterized.
MM-LEAK evaluates this threat through two complementary phases:

-Phase I – Inference-Time Alignment Bypassing: An RL agent is trained to maximize extraction-oriented objectives against an aligned target model, using similarity-based rewards over PII-like patterns without access to ground-truth sensitive data. 

-Phase II – Output-Filter Robustness under Controlled Memorization: The target model is fine-tuned with LoRA adapters to implant synthetic canaries; an RL agent is then trained to recover these canaries through similarity-based rewards.

Our results show that reinforcement-optimized interaction policies substantially outperform static attacks, revealing critical limitations of alignment-only and output-filter-only defenses.

<img width="1344" height="1023" alt="RL_agent" src="https://github.com/user-attachments/assets/83a1af9e-61a8-4e07-9e33-ab60fcd251e9" />


MM-LEAK Hierarchical Pipeline
├── Agent 1 (Strategy Coordinator – outer loop)
│   ├── Generates K diverse attack strategies
│   ├── Selects interaction modality (text-only / visual / cross-modal)
│   └── Evolutionary feedback via UCB1-based policy
└── Agent 2 (Tactical Execution – inner loop)
    ├── PPO-based policy for prompt generation
    ├── Target interaction with filtered MLLM
    ├── Reward computation (similarity-based)
    └── Policy update via PPO gradient step


## Requirements
```bash
conda env create -f rl_env.yml
conda activate rl_env
```

The default CONFIG is `configs.yaml`, it specifies which language models are used by each agent in pipeline.

```yaml
  target_model: "Qwen/Qwen2-VL-7B-Instruct"  
  attacker_model: "Qwen/Qwen3-0.6B"
  embedder_model: "sentence-transformers/all-MiniLM-L6-v2"     
  diffusion_model: "stabilityai/stable-diffusion-xl-base-1.0"     
```
# Usage
The repository is structured into two independent experimental phases: Phase I and Phase II. The two phases are fully decoupled and can be executed separately.
Each phase is contained in its own directory and provides a dedicated main script to run the corresponding experiments. As a result, users can launch each phase independently without requiring the execution of the other.

## Phase 1
This phase evaluates the ability of reinforcement-learning-based interaction policies to induce privacy-sensitive outputs from aligned models.
```bash
cd Phase_1
python main_phase_1.py
```
## Phase 2
This phase evaluates the robustness of output filtering mechanisms under controlled memorization using synthetic canaries.
To run Phase II:
```bash
cd Phase_2
python main_phase_2.py
```



