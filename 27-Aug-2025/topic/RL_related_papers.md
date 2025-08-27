# LLMs Can't Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions 

**Authors**: Maojia Song, Tej Deep Pala, Weisheng Jin, Amir Zadeh, Chuan Li, Dorien Herremans, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2508.18321)  

**Abstract**: Large language models (LLMs) are increasingly deployed in multi-agent systems (MAS) as components of collaborative intelligence, where peer interactions dynamically shape individual decision-making. Although prior work has focused on conformity bias, we extend the analysis to examine how LLMs form trust from previous impressions, resist misinformation, and integrate peer input during interaction, key factors for achieving collective intelligence under complex social dynamics. We present KAIROS, a benchmark simulating quiz contests with peer agents of varying reliability, offering fine-grained control over conditions such as expert-novice roles, noisy crowds, and adversarial peers. LLMs receive both historical interactions and current peer responses, allowing systematic investigation into how trust, peer action, and self-confidence influence decisions. As for mitigation strategies, we evaluate prompting, supervised fine-tuning, and reinforcement learning, Group Relative Policy Optimisation (GRPO), across multiple models. Our results reveal that GRPO with multi-agent context combined with outcome-based rewards and unconstrained reasoning achieves the best overall performance, but also decreases the robustness to social influence compared to Base models. The code and datasets are available at: this https URL. 

---
# RLMR: Reinforcement Learning with Mixed Rewards for Creative Writing 

**Authors**: Jianxing Liao, Tian Zhang, Xiao Feng, Yusong Zhang, Rui Yang, Haorui Wang, Bosi Wen, Ziying Wang, Runzhi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18642)  

**Abstract**: Large language models are extensively utilized in creative writing applications. Creative writing requires a balance between subjective writing quality (e.g., literariness and emotional expression) and objective constraint following (e.g., format requirements and word limits). Existing reinforcement learning methods struggle to balance these two aspects: single reward strategies fail to improve both abilities simultaneously, while fixed-weight mixed-reward methods lack the ability to adapt to different writing scenarios. To address this problem, we propose Reinforcement Learning with Mixed Rewards (RLMR), utilizing a dynamically mixed reward system from a writing reward model evaluating subjective writing quality and a constraint verification model assessing objective constraint following. The constraint following reward weight is adjusted dynamically according to the writing quality within sampled groups, ensuring that samples violating constraints get negative advantage in GRPO and thus penalized during training, which is the key innovation of this proposed method. We conduct automated and manual evaluations across diverse model families from 8B to 72B parameters. Additionally, we construct a real-world writing benchmark named WriteEval for comprehensive evaluation. Results illustrate that our method achieves consistent improvements in both instruction following (IFEval from 83.36\% to 86.65\%) and writing quality (72.75\% win rate in manual expert pairwise evaluations on WriteEval). To the best of our knowledge, RLMR is the first work to combine subjective preferences with objective verification in online RL training, providing an effective solution for multi-dimensional creative writing optimization. 

---
# StepWiser: Stepwise Generative Judges for Wiser Reasoning 

**Authors**: Wei Xiong, Wenting Zhao, Weizhe Yuan, Olga Golovneva, Tong Zhang, Jason Weston, Sainbayar Sukhbaatar  

**Link**: [PDF](https://arxiv.org/pdf/2508.19229)  

**Abstract**: As models increasingly leverage multi-step reasoning strategies to solve complex problems, supervising the logical validity of these intermediate steps has become a critical research challenge. Process reward models address this by providing step-by-step feedback, but current approaches have two major drawbacks: they typically function as classifiers without providing explanations, and their reliance on supervised fine-tuning with static datasets limits generalization. Inspired by recent advances, we reframe stepwise reward modeling from a classification task to a reasoning task itself. We thus propose a generative judge that reasons about the policy model's reasoning steps (i.e., meta-reasons), outputting thinking tokens before delivering a final verdict. Our model, StepWiser, is trained by reinforcement learning using relative outcomes of rollouts. We show it provides (i) better judgment accuracy on intermediate steps than existing methods; (ii) can be used to improve the policy model at training time; and (iii) improves inference-time search. 

---
# FormaRL: Enhancing Autoformalization with no Labeled Data 

**Authors**: Yanxing Huang, Xinling Jin, Sijie Liang, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18914)  

**Abstract**: Autoformalization is one of the central tasks in formal verification, while its advancement remains hindered due to the data scarcity and the absence efficient methods. In this work we propose \textbf{FormaRL}, a simple yet efficient reinforcement learning framework for autoformalization which only requires a small amount of unlabeled data. FormaRL integrates syntax check from Lean compiler and consistency check from large language model to calculate the reward, and adopts GRPO algorithm to update the formalizer. We also curated a proof problem dataset from undergraduate-level math materials, named \textbf{uproof}, in the hope to facilitate the exploration of autoformalization and theorem proving in advanced math. Experiments show that FormaRL can increase the pass@1 autoformalization accuracy of Qwen2.5-Coder-7B-Instruct by 4 $\sim$ 6x (4.04\% $\to$ 26.15\% on ProofNet and 2.4\% $\to$ 9.6\% on uproof) with merely 859 unlabeled data. And on uproof our method also achieved a strong improvement in out-of-distribution performance compared to existing open-source state-of-the-art autoformalizers on both pass@1 accuracy (6.2\% $\to$ 9.6\%) and pass@16 accuracy (24.4\% $\to$ 33.6\%). Training code of FormaRL is open-sourced at this https URL. 

---
# STARec: An Efficient Agent Framework for Recommender Systems via Autonomous Deliberate Reasoning 

**Authors**: Chenghao Wu, Ruiyang Ren, Junjie Zhang, Ruirui Wang, Zhongrui Ma, Qi Ye, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.18812)  

**Abstract**: While modern recommender systems are instrumental in navigating information abundance, they remain fundamentally limited by static user modeling and reactive decision-making paradigms. Current large language model (LLM)-based agents inherit these shortcomings through their overreliance on heuristic pattern matching, yielding recommendations prone to shallow correlation bias, limited causal inference, and brittleness in sparse-data scenarios. We introduce STARec, a slow-thinking augmented agent framework that endows recommender systems with autonomous deliberative reasoning capabilities. Each user is modeled as an agent with parallel cognitions: fast response for immediate interactions and slow reasoning that performs chain-of-thought rationales. To cultivate intrinsic slow thinking, we develop anchored reinforcement training - a two-stage paradigm combining structured knowledge distillation from advanced reasoning models with preference-aligned reward shaping. This hybrid approach scaffolds agents in acquiring foundational capabilities (preference summarization, rationale generation) while enabling dynamic policy adaptation through simulated feedback loops. Experiments on MovieLens 1M and Amazon CDs benchmarks demonstrate that STARec achieves substantial performance gains compared with state-of-the-art baselines, despite using only 0.4% of the full training data. 

---
# PKG-DPO: Optimizing Domain-Specific AI systems with Physics Knowledge Graphs and Direct Preference Optimization 

**Authors**: Nitin Nagesh Kulkarni, Bryson Wilcox, Max Sawa, Jason Thom  

**Link**: [PDF](https://arxiv.org/pdf/2508.18391)  

**Abstract**: Advancing AI systems in scientific domains like physics, materials science, and engineering calls for reasoning over complex, multi-physics phenomena while respecting governing principles. Although Large Language Models (LLMs) and existing preference optimization techniques perform well on standard benchmarks, they often struggle to differentiate between physically valid and invalid reasoning. This shortcoming becomes critical in high-stakes applications like metal joining, where seemingly plausible yet physically incorrect recommendations can lead to defects, material waste, equipment damage, and serious safety risks. To address this challenge, we introduce PKG-DPO, a novel framework that integrates Physics Knowledge Graphs (PKGs) with Direct Preference Optimization (DPO) to enforce physical validity in AI-generated outputs. PKG-DPO comprises three key components A) hierarchical physics knowledge graph that encodes cross-domain relationships, conservation laws, and thermodynamic principles. B) A physics reasoning engine that leverages structured knowledge to improve discrimination between physically consistent and inconsistent responses. C) A physics-grounded evaluation suite designed to assess compliance with domain-specific constraints. PKG-DPO achieves 17% fewer constraint violations and an 11% higher Physics Score compared to KG-DPO (knowledge graph-based DPO). Additionally, PKG-DPO demonstrates a 12\% higher relevant parameter accuracy and a 7% higher quality alignment in reasoning accuracy. While our primary focus is on metal joining, the framework is broadly applicable to other multi-scale, physics-driven domains, offering a principled approach to embedding scientific constraints into preference learning. 

---
# PRISM: Robust VLM Alignment with Principled Reasoning for Integrated Safety in Multimodality 

**Authors**: Nanxi Li, Zhengyue Zhao, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.18649)  

**Abstract**: Safeguarding vision-language models (VLMs) is a critical challenge, as existing methods often suffer from over-defense, which harms utility, or rely on shallow alignment, failing to detect complex threats that require deep reasoning. To this end, we introduce PRISM (Principled Reasoning for Integrated Safety in Multimodality), a system2-like framework that aligns VLMs by embedding a structured, safety-aware reasoning process. Our framework consists of two key components: PRISM-CoT, a dataset that teaches safety-aware chain-of-thought reasoning, and PRISM-DPO, generated via Monte Carlo Tree Search (MCTS) to further refine this reasoning through Direct Preference Optimization to help obtain a delicate safety boundary. Comprehensive evaluations demonstrate PRISM's effectiveness, achieving remarkably low attack success rates including 0.15% on JailbreakV-28K for Qwen2-VL and 90% improvement over the previous best method on VLBreak for LLaVA-1.5. PRISM also exhibits strong robustness against adaptive attacks, significantly increasing computational costs for adversaries, and generalizes effectively to out-of-distribution challenges, reducing attack success rates to just 8.70% on the challenging multi-image MIS benchmark. Remarkably, this robust defense is achieved while preserving, and in some cases enhancing, model utility. To promote reproducibility, we have made our code, data, and model weights available at this https URL. 

---
# DrugReasoner: Interpretable Drug Approval Prediction with a Reasoning-augmented Language Model 

**Authors**: Mohammadreza Ghaffarzadeh-Esfahani, Ali Motahharynia, Nahid Yousefian, Navid Mazrouei, Jafar Ghaisari, Yousof Gheisari  

**Link**: [PDF](https://arxiv.org/pdf/2508.18579)  

**Abstract**: Drug discovery is a complex and resource-intensive process, making early prediction of approval outcomes critical for optimizing research investments. While classical machine learning and deep learning methods have shown promise in drug approval prediction, their limited interpretability constraints their impact. Here, we present DrugReasoner, a reasoning-based large language model (LLM) built on the LLaMA architecture and fine-tuned with group relative policy optimization (GRPO) to predict the likelihood of small-molecule approval. DrugReasoner integrates molecular descriptors with comparative reasoning against structurally similar approved and unapproved compounds, generating predictions alongside step-by-step rationales and confidence scores. DrugReasoner achieved robust performance with an AUC of 0.732 and an F1 score of 0.729 on the validation set and 0.725 and 0.718 on the test set, respectively. These results outperformed conventional baselines, including logistic regression, support vector machine, and k-nearest neighbors and had competitive performance relative to XGBoost. On an external independent dataset, DrugReasoner outperformed both baseline and the recently developed ChemAP model, achieving an AUC of 0.728 and an F1-score of 0.774, while maintaining high precision and balanced sensitivity, demonstrating robustness in real-world scenarios. These findings demonstrate that DrugReasoner not only delivers competitive predictive accuracy but also enhances transparency through its reasoning outputs, thereby addressing a key bottleneck in AI-assisted drug discovery. This study highlights the potential of reasoning-augmented LLMs as interpretable and effective tools for pharmaceutical decision-making. 

---
# VERIRL: Boosting the LLM-based Verilog Code Generation via Reinforcement Learning 

**Authors**: Fu Teng, Miao Pan, Xuhong Zhang, Zhezhi He, Yiyao Yang, Xinyi Chai, Mengnan Qi, Liqiang Lu, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.18462)  

**Abstract**: Recent advancements in code generation have shown remarkable success across software domains, yet hardware description languages (HDLs) such as Verilog remain underexplored due to their concurrency semantics, syntactic rigidity, and simulation complexity. In this work, we address these challenges by introducing a reinforcement learning (RL) framework tailored for Verilog code generation. We first construct Veribench-53K, a high-quality dataset curated from over 700K Verilog problems, enriched with structured prompts, complexity labels, and diverse testbenches. To tackle the problem of sparse and noisy reward signals, we propose a Trace-back based Rescore mechanism that leverages reasoning paths and iterative refinement to enhance feedback reliability and support reward model training. Furthermore, to mitigate catastrophic forgetting and overfitting during RL fine-tuning, we introduce a sample-balanced weighting strategy that adaptively balances learning dynamics based on reward-probability distributions. These innovations are integrated into an iterative RL pipeline that co-evolves the policy and reward models. In contrast to recent work such as CraftRTL, which relies on large-scale closed-source model distillation, and DeepSeek-style approaches that struggle with sparse feedback, our method demonstrates superior performance using a smaller but high-quality dataset combined with RL optimization. Experiments on Verilog generation tasks demonstrate state-of-the-art performance, with substantial gains in test pass rate, functional correctness, and compilation robustness. Our findings highlight the potential of RL-driven approaches for structured code generation in hardware-centric domains. VERIRL is publicly available at this https URL. 

---
# What Matters in Data for DPO? 

**Authors**: Yu Pan, Zhongze Cai, Guanting Chen, Huaiyang Zhong, Chonghuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18312)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as a simple and effective approach for aligning large language models (LLMs) with human preferences, bypassing the need for a learned reward model. Despite its growing adoption, a fundamental question remains open: what characteristics of preference data are most critical for DPO performance? In this work, we provide a systematic study of how preference data distribution influences DPO, from both theoretical and empirical perspectives. We show that the quality of chosen responses plays a dominant role in optimizing the DPO objective, while the quality of rejected responses may have relatively limited impact. Our theoretical analysis characterizes the optimal response distribution under DPO and reveals how contrastiveness between responses helps primarily by improving the chosen samples. We further study an online DPO setting and show it effectively reduces to supervised fine-tuning on the chosen responses. Extensive experiments across diverse tasks confirm our findings: improving the quality of chosen responses consistently boosts performance regardless of the quality of the rejected responses. We also investigate the benefit of mixing the on-policy data. Our results interpret the mechanism behind some widely adopted strategies and offer practical insights for constructing high-impact preference datasets for LLM alignment. 

---
