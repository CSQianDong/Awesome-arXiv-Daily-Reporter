# VQPP: Video Query Performance Prediction Benchmark 

**Authors**: Adrian Catalin Lutu, Eduard Poesina, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2602.17814)  

**Abstract**: Query performance prediction (QPP) is an important and actively studied information retrieval task, having various applications, such as query reformulation, query expansion, and retrieval system selection, among many others. The task has been primarily studied in the context of text and image retrieval, whereas QPP for content-based video retrieval (CBVR) remains largely underexplored. To this end, we propose the first benchmark for video query performance prediction (VQPP), comprising two text-to-video retrieval datasets and two CBVR systems, respectively. VQPP contains a total of 56K text queries and 51K videos, and comes with official training, validation and test splits, fostering direct comparisons and reproducible results. We explore multiple pre-retrieval and post-retrieval performance predictors, creating a representative benchmark for future exploration of QPP in the video domain. Our results show that pre-retrieval predictors obtain competitive performance, enabling applications before performing the retrieval step. We also demonstrate the applicability of VQPP by employing the best performing pre-retrieval predictor as reward model for training a large language model (LLM) on the query reformulation task via direct preference optimization (DPO). We release our benchmark and code at this https URL. 

---
# Turbo Connection: Reasoning as Information Flow from Higher to Lower Layers 

**Authors**: Mohan Tang, Sidi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.17993)  

**Abstract**: Complex problems, whether in math, logic, or planning, are solved by humans through a sequence of steps where the result of one step informs the next. In this work, we adopt the perspective that the reasoning power of Transformers is fundamentally limited by a fixed maximum number of steps along any latent path of computation. To address this, we introduce Turbo Connection (TurboConn), a novel architecture that overcomes the fixed-depth constraint by routing multiple residual connections from the higher-layer hidden states of each token $t$ to the lower layers of token $t+1$. Fine-tuning pre-trained LLMs with our method not only yields accuracy gains of 0.9% to over 10% on benchmarks like GSM8K, Parity, and multi-step arithmetic, but also demonstrates that the density of these backward connections is critical; our dense interaction significantly outperforms "sparse" alternatives that only pass a single hidden state or vector. Notably, TurboConn can be integrated into pre-trained LLMs to overcome task-specific plateaus: while a fine-tuned Qwen-3-1.7B achieves only 53.78% on Parity, adding our architectural modification enables the model to reach 100% accuracy, all without the necessity to retrain the full model from scratch or sophisticated curriculum learning. Our results provide strong empirical evidence that the depth of the computational path is a key factor in reasoning ability, also offering a new mechanism to enhance LLMs without significantly affecting generation latency. 

---
# Gradient Regularization Prevents Reward Hacking in Reinforcement Learning from Human Feedback and Verifiable Rewards 

**Authors**: Johannes Ackermann, Michael Noukhovitch, Takashi Ishida, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2602.18037)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) or Verifiable Rewards (RLVR) are two key steps in the post-training of modern Language Models (LMs). A common problem is reward hacking, where the policy may exploit inaccuracies of the reward and learn an unintended behavior. Most previous works address this by limiting the policy update with a Kullback-Leibler (KL) penalty towards a reference model. We propose a different framing: Train the LM in a way that biases policy updates towards regions in which the reward is more accurate. First, we derive a theoretical connection between the accuracy of a reward model and the flatness of an optimum at convergence. Gradient regularization (GR) can then be used to bias training to flatter regions and thereby maintain reward model accuracy. We confirm these results by showing that the gradient norm and reward accuracy are empirically correlated in RLHF. We then show that Reference Resets of the KL penalty implicitly use GR to find flatter regions with higher reward accuracy. We further improve on this by proposing to use explicit GR with an efficient finite-difference estimate. Empirically, GR performs better than a KL penalty across a diverse set of RL experiments with LMs. GR achieves a higher GPT-judged win-rate in RLHF, avoids overly focusing on the format in rule-based math rewards, and prevents hacking the judge in LLM-as-a-Judge math tasks. 

---
# CodeScaler: Scaling Code LLM Training and Test-Time Inference via Execution-Free Reward Models 

**Authors**: Xiao Zhu, Xinyu Zhou, Boyu Zhu, Hanxu Hu, Mingzhe Du, Haotian Zhang, Huiming Wang, Zhijiang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.17684)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has driven recent progress in code large language models by leveraging execution-based feedback from unit tests, but its scalability is fundamentally constrained by the availability and reliability of high-quality test cases. We propose CodeScaler, an execution-free reward model designed to scale both reinforcement learning training and test-time inference for code generation. CodeScaler is trained on carefully curated preference data derived from verified code problems and incorporates syntax-aware code extraction and validity-preserving reward shaping to ensure stable and robust optimization. Across five coding benchmarks, CodeScaler improves Qwen3-8B-Base by an average of +11.72 points, outperforming binary execution-based RL by +1.82 points, and enables scalable reinforcement learning on synthetic datasets without any test cases. At inference time, CodeScaler serves as an effective test-time scaling method, achieving performance comparable to unit test approaches while providing a 10-fold reduction in latency. Moreover, CodeScaler surpasses existing reward models on RM-Bench not only in the code domain (+3.3 points), but also in general and reasoning domains (+2.7 points on average). 

---
# Curriculum Learning for Efficient Chain-of-Thought Distillation via Structure-Aware Masking and GRPO 

**Authors**: Bowen Yu, Maolin Wang, Sheng Zhang, Binhao Wang, Yi Wen, Jingtong Gao, Bowen Liu, Zimo Zhao, Wanyu Wang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.17686)  

**Abstract**: Distilling Chain-of-Thought (CoT) reasoning from large language models into compact student models presents a fundamental challenge: teacher rationales are often too verbose for smaller models to faithfully reproduce. Existing approaches either compress reasoning into single-step, losing the interpretability that makes CoT valuable. We present a three-stage curriculum learning framework that addresses this capacity mismatch through progressive skill acquisition. First, we establish structural understanding via masked shuffled reconstruction. Second, we apply Group Relative Policy Optimization (GRPO) on masked completion tasks, enabling the model to discover its own balance between accuracy and brevity. Third, we identify persistent failure cases and guide the student to internalize teacher knowledge through targeted rewriting, again optimized with GRPO. Experiments on GSM8K demonstrate that our approach enables Qwen2.5-3B-Base to achieve an 11.29 percent accuracy improvement while reducing output length by 27.4 percent, surpassing both instruction-tuned variants and prior distillation methods. 

---
