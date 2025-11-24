# Masked-and-Reordered Self-Supervision for Reinforcement Learning from Verifiable Rewards 

**Authors**: Zhen Wang, Zhifeng Gao, Guolin Ke  

**Link**: [PDF](https://arxiv.org/pdf/2511.17473)  

**Abstract**: Test-time scaling has been shown to substantially improve large language models' (LLMs) mathematical reasoning. However, for a large portion of mathematical corpora, especially theorem proving, RLVR's scalability is limited: intermediate reasoning is crucial, while final answers are difficult to directly and reliably verify. Meanwhile, token-level SFT often degenerates into rote memorization rather than inducing longer chains of thought. Inspired by BERT's self-supervised tasks, we propose MR-RLVR (Masked-and-Reordered RLVR), which constructs process-level self-supervised rewards via "masked-then-fill" and "step reordering" to extract learnable signals from intermediate reasoning. Our training pipeline comprises two stages: we first perform self-supervised training on sampled mathematical calculation and proof data; we then conduct RLVR fine-tuning on mathematical calculation datasets where only outcomes are verifiable. We implement MR-RLVR on Qwen2.5-3B and DeepSeek-R1-Distill-Qwen-1.5B, and evaluate on AIME24, AIME25, AMC23, and MATH500. Under a fixed sampling and decoding budget, MR-RLVR achieves average relative gains over the original RLVR of +9.86% Pass@1, +5.27% Pass@5, and +4.00% Pass@8. These results indicate that incorporating process-aware self-supervised signals can effectively enhance RLVR's scalability and performance in only outcome-verifiable settings. 

---
# Improving Latent Reasoning in LLMs via Soft Concept Mixing 

**Authors**: Kang Wang, Xiangyu Duan, Tianyi Du  

**Link**: [PDF](https://arxiv.org/pdf/2511.16885)  

**Abstract**: Unlike human reasoning in abstract conceptual spaces, large language models (LLMs) typically reason by generating discrete tokens, which potentially limit their expressive power. The recent work Soft Thinking has shown that LLMs' latent reasoning via soft concepts is a promising direction, but LLMs are trained on discrete tokens. To reduce this gap between the soft concepts in reasoning and the discrete tokens in training, we propose Soft Concept Mixing (SCM), a soft concept aware training scheme that directly exposes the model to soft representations during training. Specifically, SCM constructs a soft concept vector by forming a probability-weighted average of embeddings. Then, this vector is mixed into the model's hidden states, which embody rich contextual information. Finally, the entire latent reasoning process is optimized with Reinforcement Learning (RL). Experiments on five reasoning benchmarks demonstrate that SCM improves the reasoning performance of LLMs, and simultaneously maintains a stable training dynamic. 

---
