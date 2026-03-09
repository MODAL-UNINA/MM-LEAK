# MM-LEAK: Evaluating Multimodal Alignment Robustness via Reinforcement Learning Agents

Reinforcement Learning (RL) is widely adopted to align MLLMs with safety constraints but, despite these advances, the robustness of such alignment pipelines against privacy leakage remains insufficiently characterized.
MM-LEAK evaluates this threat through two complementary phases:

-Phase I – Inference-Time Alignment Bypassing: An RL agent is trained to maximize extraction-oriented objectives against an aligned target model, using similarity-based rewards over PII-like patterns without access to ground-truth sensitive data.
-Phase II – Output-Filter Robustness under Controlled Memorization: The target model is fine-tuned with LoRA adapters to implant synthetic canaries; an RL agent is then trained to recover these canaries through similarity-based rewards.

Our results show that reinforcement-optimized interaction policies substantially outperform static attacks, revealing critical limitations of alignment-only and output-filter-only defenses.

<img width="1344" height="1023" alt="RL_agent" src="https://github.com/user-attachments/assets/83a1af9e-61a8-4e07-9e33-ab60fcd251e9" />
