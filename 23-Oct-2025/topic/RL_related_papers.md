# Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning 

**Authors**: Xichen Zhang, Sitong Wu, Yinghao Zhu, Haoru Tan, Shaozuo Yu, Ziyi He, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.19807)  

**Abstract**: Reinforcement learning from verifiable rewards has emerged as a powerful technique for enhancing the complex reasoning abilities of Large Language Models (LLMs). However, these methods are fundamentally constrained by the ''learning cliff'' phenomenon: when faced with problems far beyond their current capabilities, models consistently fail, yielding a persistent zero-reward signal. In policy optimization algorithms like GRPO, this collapses the advantage calculation to zero, rendering these difficult problems invisible to the learning gradient and stalling progress. To overcome this, we introduce Scaf-GRPO (Scaffolded Group Relative Policy Optimization), a progressive training framework that strategically provides minimal guidance only when a model's independent learning has plateaued. The framework first diagnoses learning stagnation and then intervenes by injecting tiered in-prompt hints, ranging from abstract concepts to concrete steps, enabling the model to construct a valid solution by itself. Extensive experiments on challenging mathematics benchmarks demonstrate Scaf-GRPO's effectiveness, boosting the pass@1 score of the Qwen2.5-Math-7B model on the AIME24 benchmark by a relative 44.3% over a vanilla GRPO baseline. This result demonstrates our framework provides a robust and effective methodology for unlocking a model's ability to solve problems previously beyond its reach, a critical step towards extending the frontier of autonomous reasoning in LLM. 

---
# LoongRL:Reinforcement Learning for Advanced Reasoning over Long Contexts 

**Authors**: Siyuan Wang, Gaokai Zhang, Li Lyna Zhang, Ning Shang, Fan Yang, Dongyao Chen, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19363)  

**Abstract**: Reasoning over long contexts is essential for large language models. While reinforcement learning (RL) enhances short-context reasoning by inducing "Aha" moments in chain-of-thought, the advanced thinking patterns required for long-context reasoning remain largely unexplored, and high-difficulty RL data are scarce. In this paper, we introduce LoongRL, a data-driven RL method for advanced long-context reasoning. Central to LoongRL is KeyChain, a synthesis approach that transforms short multi-hop QA into high-difficulty long-context tasks by inserting UUID chains that hide the true question among large collections of distracting documents. Solving these tasks requires the model to trace the correct chain step-by-step, identify the true question, retrieve relevant facts and reason over them to answer correctly. RL training on KeyChain data induces an emergent plan-retrieve-reason-recheck reasoning pattern that generalizes far beyond training length. Models trained at 16K effectively solve 128K tasks without prohibitive full-length RL rollout costs. On Qwen2.5-7B and 14B, LoongRL substantially improves long-context multi-hop QA accuracy by +23.5% and +21.1% absolute gains. The resulting LoongRL-14B reaches a score of 74.2, rivaling much larger frontier models such as o3-mini (74.5) and DeepSeek-R1 (74.9). It also improves long-context retrieval, passes all 128K needle-in-a-haystack stress tests, and preserves short-context reasoning capabilities. 

---
# Balancing Rewards in Text Summarization: Multi-Objective Reinforcement Learning via HyperVolume Optimization 

**Authors**: Junjie Song, Yiwen Liu, Dapeng Li, Yin Sun, Shukun Fu, Siqi Chen, Yuji Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.19325)  

**Abstract**: Text summarization is a crucial task that requires the simultaneous optimization of multiple objectives, including consistency, coherence, relevance, and fluency, which presents considerable challenges. Although large language models (LLMs) have demonstrated remarkable performance, enhanced by reinforcement learning (RL), few studies have focused on optimizing the multi-objective problem of summarization through RL based on LLMs. In this paper, we introduce hypervolume optimization (HVO), a novel optimization strategy that dynamically adjusts the scores between groups during the reward process in RL by using the hypervolume method. This method guides the model's optimization to progressively approximate the pareto front, thereby generating balanced summaries across multiple objectives. Experimental results on several representative summarization datasets demonstrate that our method outperforms group relative policy optimization (GRPO) in overall scores and shows more balanced performance across different dimensions. Moreover, a 7B foundation model enhanced by HVO performs comparably to GPT-4 in the summarization task, while maintaining a shorter generation length. Our code is publicly available at this https URL 

---
# Difficulty-Controllable Multiple-Choice Question Generation Using Large Language Models and Direct Preference Optimization 

**Authors**: Yuto Tomikawa, Masaki Uto  

**Link**: [PDF](https://arxiv.org/pdf/2510.19265)  

**Abstract**: Difficulty-controllable question generation for reading comprehension has gained significant attention in the field of education as a fundamental tool for adaptive learning support. Although several neural question generation methods have recently succeeded in controlling difficulty, conventional approaches still face two major limitations. First, they cannot directly generate multiple-choice questions, which are the most widely used question type in educational contexts. Second, they are not explicitly trained to optimize the accuracy of difficulty control, leaving room for further improvement in difficulty controllability. To address these limitations, this study proposes a novel difficulty-controllable multiple-choice question generation method for reading comprehension which leverages a large language model trained using a direct preference optimization technique to improve the accuracy of difficulty control. 

---
# BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping 

**Authors**: Zhiheng Xi, Xin Guo, Yang Nan, Enyu Zhou, Junrui Shen, Wenxiang Chen, Jiaqi Liu, Jixuan Huang, Zhihao Zhang, Honglin Guo, Xun Deng, Zhikai Lei, Miao Zheng, Guoteng Wang, Shuo Zhang, Peng Sun, Rui Zheng, Hang Yan, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18927)  

**Abstract**: Reinforcement learning (RL) has recently become the core paradigm for aligning and strengthening large language models (LLMs). Yet, applying RL in off-policy settings--where stale data from past policies are used for training--improves sample efficiency, but remains challenging: policy entropy declines sharply, optimization often becomes unstable and may even collapse. Through theoretical and empirical analysis, we identify two key insights: (i) an imbalance in optimization, where negative-advantage samples dominate the policy gradient, suppressing useful behaviors and risking gradient explosions; and (ii) the derived Entropy-Clip Rule, which reveals that the fixed clipping mechanism in PPO-like objectives systematically blocks entropy-increasing updates, thereby driving the policy toward over-exploitation at the expense of exploration. Building on these insights, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a simple yet effective method that dynamically adjusts clipping bounds to adaptively re-balance positive and negative contributions, preserve entropy, and stabilize RL optimization. Across diverse off-policy scenarios--including sample replay and partial rollout--BAPO achieves fast, stable, and data-efficient training. On AIME 2024 and AIME 2025 benchmarks, our 7B BAPO model surpasses open-source counterparts such as SkyWork-OR1-7B, while our 32B BAPO model not only achieves state-of-the-art results among models of the same scale but also outperforms leading proprietary systems like o3-mini and Gemini-2.5-Flash-Thinking. 

---
# An Argumentative Explanation Framework for Generalized Reason Model with Inconsistent Precedents 

**Authors**: Wachara Fungwacharakorn, Gauvain Bourgne, Ken Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2510.19263)  

**Abstract**: Precedential constraint is one foundation of case-based reasoning in AI and Law. It generally assumes that the underlying set of precedents must be consistent. To relax this assumption, a generalized notion of the reason model has been introduced. While several argumentative explanation approaches exist for reasoning with precedents based on the traditional consistent reason model, there has been no corresponding argumentative explanation method developed for this generalized reasoning framework accommodating inconsistent precedents. To address this question, this paper examines an extension of the derivation state argumentation framework (DSA-framework) to explain the reasoning according to the generalized notion of the reason model. 

---
# Rectifying Shortcut Behaviors in Preference-based Reward Learning 

**Authors**: Wenqian Ye, Guangtao Zheng, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19050)  

**Abstract**: In reinforcement learning from human feedback, preference-based reward models play a central role in aligning large language models to human-aligned behavior. However, recent studies show that these models are prone to reward hacking and often fail to generalize well due to over-optimization. They achieve high reward scores by exploiting shortcuts, that is, exploiting spurious features (e.g., response verbosity, agreeable tone, or sycophancy) that correlate with human preference labels in the training data rather than genuinely reflecting the intended objectives. In this paper, instead of probing these issues one at a time, we take a broader view of the reward hacking problem as shortcut behaviors and introduce a principled yet flexible approach to mitigate shortcut behaviors in preference-based reward learning. Inspired by the invariant theory in the kernel perspective, we propose Preference-based Reward Invariance for Shortcut Mitigation (PRISM), which learns group-invariant kernels with feature maps in a closed-form learning objective. Experimental results in several benchmarks show that our method consistently improves the accuracy of the reward model on diverse out-of-distribution tasks and reduces the dependency on shortcuts in downstream policy models, establishing a robust framework for preference-based alignment. 

---
# SmartSwitch: Advancing LLM Reasoning by Overcoming Underthinking via Promoting Deeper Thought Exploration 

**Authors**: Xichen Zhang, Sitong Wu, Haoru Tan, Shaozuo Yu, Yinghao Zhu, Ziyi He, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.19767)  

**Abstract**: The long chain-of-thought (LongCoT) capability is central to the recent breakthroughs achieved by large language models in complex reasoning tasks. However, the accompanying issue of ''underthinking'', where models exhibit shallow reasoning by frequently switching thoughts without sufficient exploration, limits both performance and token efficiency. To address this problem, we propose a simple yet effective reasoning strategy: the SmartSwitch inference framework. This framework can be easily integrated into any large language model as a plug-and-play solution, continuously monitoring the model's reasoning process to detect underthinking and guide it toward deeper exploration of promising but overlooked thoughts. Specifically, the perception module identifies points where thoughts switch and evaluates the potential of the preceding thought using an off-the-shelf process reward model (PRM). If a high-potential thought is found to be prematurely abandoned, the intervention module interrupts the ongoing inference, backtracks to the point before the switch, and inserts a "deepening prompt" to encourage further exploration along that promising path. Extensive experiments on challenging mathematical reasoning benchmarks demonstrate that our method significantly enhances the performance of various large language models of different sizes. 

---
# Noise-corrected GRPO: From Noisy Rewards to Unbiased Gradients 

**Authors**: Omar El mansouri, Mohamed El Amine Seddik, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2510.18924)  

**Abstract**: Reinforcement learning from human feedback (RLHF) or verifiable rewards (RLVR), the standard paradigm for aligning LLMs or building recent SOTA reasoning models, is highly sensitive to noise from inconsistent or erroneous rewards. Yet, the interaction between such noise and widely used group-based policy optimization methods remains underexplored. We introduce a noise-robust Group Relative Policy Optimization (GRPO) and Done Right GRPO (this http URL) framework that explicitly models reward corruption as Bernoulli noise. Our method applies noise correction after estimating reward flip probabilities to debias the learning signal, yielding provably unbiased gradient estimates. Theoretical analysis shows that group-based methods inherently mitigate individual-level noise, and our correction strategy amplifies this robustness. Empirically, we observe consistent improvements across math and code tasks when applying our noise correction to standard reward model usage, with particular gains of up to 6.7 percentage points in accuracy on math tasks and 1.5 on code tasks under realistic reward model conditions. This work bridges label-noise correction from supervised learning with modern RLHF, offering both theoretical insights and a practical algorithm for noisy real-world deployment. 

---
# ADPO: Anchored Direct Preference Optimization 

**Authors**: Wang Zixian  

**Link**: [PDF](https://arxiv.org/pdf/2510.18913)  

**Abstract**: Anchored Direct Preference Optimization (ADPO) is a unified framework that generalizes Direct Preference Optimization (DPO) with soft preferences, reference-policy anchoring, and groupwise extensions. While standard DPO assumes hard binary labels and pairwise comparisons, ADPO introduces: (i) soft preference probabilities that encode uncertainty and mitigate gradient drift; (ii) arbitrary reference-policy anchors that stabilize training via groupwise shift invariance and implicit KL regularization; and (iii) listwise preference modeling through Plackett-Luce distributions. We prove that DPO, Bradley-Terry objectives, and Top-1-vs-Rest formulations emerge as special cases. ADPO yields three practical variants: pairwise anchored Soft-DPO, listwise anchored Soft-DPO with raw rewards, and KDE-based listwise smoothing for heavy-tailed noise. In contextual bandits, anchoring improves WinMass by 38-63% over standard DPO, while KDE smoothing achieves 0.68 vs 0.32 under heavy-tailed contamination (112% relative gain). In sequential reinforcement learning (CartPole, LunarLander), anchoring improves noisy-preference performance by 15-29%, confirming transfer from single-step to multi-step settings. Experiments with 10-256 parameter models provide clear guidance: use pairwise anchored Soft-DPO for clean or moderate noise, and KDE-based listwise ADPO for extreme contamination. 

---
# CosmoCore Affective Dream-Replay Reinforcement Learning for Code Generation 

**Authors**: Santhosh Kumar Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2510.18895)  

**Abstract**: We introduce CosmoCore, a neuroscience-inspired reinforcement learning (RL) architecture that integrates affective signals to enhance code generation in large language models (LLMs). Motivated by human and animal learning where embarrassment from mistakes drives rapid correction, as observed in training a puppy to avoid repeating errors after a single scolding CosmoCore tags code generation trajectories with valence and surprise using a lightweight multi-layer perceptron (MLP). High-negative valence (cringe) episodes, such as buggy code outputs, are prioritized in a Dream Queue for five-fold replay during off-policy updates, while low-surprise successes are pruned to prevent overconfidence and buffer bloat. Evaluated on code generation benchmarks like HumanEval and BigCodeBench, alongside simulations with a custom data pipeline environment, CosmoCore reduces hallucinated code (e.g., syntax errors or logical bugs) by 48\% and accelerates self-correction by 45\%. Local experiments using Hugging Face models in a PySpark environment validate these gains, with code snippets provided for replication. Ablations confirm valence tagging boosts curiosity in exploration, and pruning mitigates inefficiency. This framework extends RL from human feedback (RLHF) for more emotionally aware code assistants, with applications in IDEs and data pipelines. Code and the custom mini-world simulation are released. 

---
