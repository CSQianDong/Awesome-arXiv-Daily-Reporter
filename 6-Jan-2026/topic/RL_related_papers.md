# DermoGPT: Open Weights and Open Data for Morphology-Grounded Dermatological Reasoning MLLMs 

**Authors**: Jinghan Ru, Siyuan Yan, Yuguo Yin, Yuexian Zou, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2601.01868)  

**Abstract**: Multimodal Large Language Models (MLLMs) show promise for medical applications, yet progress in dermatology lags due to limited training data, narrow task coverage, and lack of clinically-grounded supervision that mirrors expert diagnostic workflows. We present a comprehensive framework to address these gaps. First, we introduce DermoInstruct, a large-scale morphology-anchored instruction corpus comprising 211,243 images and 772,675 trajectories across five task formats, capturing the complete diagnostic pipeline from morphological observation and clinical reasoning to final diagnosis. Second, we establish DermoBench, a rigorous benchmark evaluating 11 tasks across four clinical axes: Morphology, Diagnosis, Reasoning, and Fairness, including a challenging subset of 3,600 expert-verified open-ended instances and human performance baselines. Third, we develop DermoGPT, a dermatology reasoning MLLM trained via supervised fine-tuning followed by our Morphologically-Anchored Visual-Inference-Consistent (MAVIC) reinforcement learning objective, which enforces consistency between visual observations and diagnostic conclusions. At inference, we deploy Confidence-Consistency Test-time adaptation (CCT) for robust predictions. Experiments show DermoGPT significantly outperforms 16 representative baselines across all axes, achieving state-of-the-art performance while substantially narrowing the human-AI gap. DermoInstruct, DermoBench and DermoGPT will be made publicly available at this https URL upon acceptance. 

---
# How Does Prefix Matter in Reasoning Model Tuning? 

**Authors**: Raj Vardhan Tomar, Preslav Nakov, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01624)  

**Abstract**: Recent alignment studies commonly remove introductory boilerplate phrases from supervised fine-tuning (SFT) datasets. This work challenges that assumption. We hypothesize that safety- and reasoning-oriented prefix sentences serve as lightweight alignment signals that can guide model decoding toward safer and more coherent responses. To examine this, we fine-tune three R1 series models across three core model capabilities: reasoning (mathematics, coding), safety, and factuality, systematically varying prefix inclusion from 0% to 100%.
Results show that prefix-conditioned SFT improves both safety and reasoning performance, yielding up to +6% higher Safe@1 accuracy on adversarial benchmarks (WildJailbreak, StrongReject) and +7% improvement on GSM8K reasoning. However, factuality and coding tasks show marginal or negative effects, indicating that prefix-induced narrowing of the search space benefits structured reasoning. Token-level loss analysis further reveals that prefix tokens such as "revised" and "logically" incur higher gradient magnitudes, acting as alignment anchors that stabilize reasoning trajectories. Our findings suggest that prefix conditioning offers a scalable and interpretable mechanism for improving reasoning safety, serving as an implicit form of alignment that complements traditional reward-based methods. 

---
# Unsupervised Text Style Transfer for Controllable Intensity 

**Authors**: Shuhuan Gu, Wenbiao Tao, Xinchen Ma, Kangkang He, Ye Guo, Xiang Li, Yunshi Lan  

**Link**: [PDF](https://arxiv.org/pdf/2601.01060)  

**Abstract**: Unsupervised Text Style Transfer (UTST) aims to build a system to transfer the stylistic properties of a given text without parallel text pairs. Compared with text transfer between style polarities, UTST for controllable intensity is more challenging due to the subtle differences in stylistic features across different intensity levels. Faced with the challenges posed by the lack of parallel data and the indistinguishability between adjacent intensity levels, we propose a SFT-then-PPO paradigm to fine-tune an LLM. We first fine-tune the LLM with synthesized parallel data. Then, we further train the LLM with PPO, where the rewards are elaborately designed for distinguishing the stylistic intensity in hierarchical levels. Both the global and local stylistic features are considered to formulate the reward functions. The experiments on two UTST benchmarks showcase that both rewards have their advantages and applying them to LLM fine-tuning can effectively improve the performance of an LLM backbone based on various evaluation metrics. Even for close levels of intensity, we can still observe the noticeable stylistic difference between the generated text. 

---
# Reinforcement Learning Enhanced Multi-hop Reasoning for Temporal Knowledge Question Answering 

**Authors**: Wuzhenghong Wen, Chao Xue, Su Pan, Yuwei Sun, Minlong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2601.01195)  

**Abstract**: Temporal knowledge graph question answering (TKGQA) involves multi-hop reasoning over temporally constrained entity relationships in the knowledge graph to answer a given question. However, at each hop, large language models (LLMs) retrieve subgraphs with numerous temporally similar and semantically complex relations, increasing the risk of suboptimal decisions and error propagation. To address these challenges, we propose the multi-hop reasoning enhanced (MRE) framework, which enhances both forward and backward reasoning to improve the identification of globally optimal reasoning trajectories. Specifically, MRE begins with prompt engineering to guide the LLM in generating diverse reasoning trajectories for a given question. Valid reasoning trajectories are then selected for supervised fine-tuning, serving as a cold-start strategy. Finally, we introduce Tree-Group Relative Policy Optimization (T-GRPO), a recursive, tree-structured learning-by-exploration approach. At each hop, exploration establishes strong causal dependencies on the previous hop, while evaluation is informed by multi-path exploration feedback from subsequent hops. Experimental results on two TKGQA benchmarks indicate that the proposed MRE-based model consistently surpasses state-of-the-art (SOTA) approaches in handling complex multi-hop queries. Further analysis highlights improved interpretability and robustness to noisy temporal annotations. 

---
# Counterfactual Self-Questioning for Stable Policy Optimization in Language Models 

**Authors**: Mandar Parab  

**Link**: [PDF](https://arxiv.org/pdf/2601.00885)  

**Abstract**: Recent work on language model self-improvement shows that models can refine their own reasoning through reflection, verification, debate, or self-generated rewards. However, most existing approaches rely on external critics, learned reward models, or ensemble sampling, which increases complexity and training instability. We propose Counterfactual Self-Questioning, a framework in which a single language model generates and evaluates counterfactual critiques of its own reasoning. The method produces an initial reasoning trace, formulates targeted questions that challenge potential failure points, and generates alternative reasoning trajectories that expose incorrect assumptions or invalid steps. These counterfactual trajectories provide structured relative feedback that can be directly used for policy optimization without auxiliary models. Experiments on multiple mathematical reasoning benchmarks show that counterfactual self-questioning improves accuracy and training stability, particularly for smaller models, enabling scalable self-improvement using internally generated supervision alone. 

---
