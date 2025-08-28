# InquireMobile: Teaching VLM-based Mobile Agent to Request Human Assistance via Reinforcement Fine-Tuning 

**Authors**: Qihang Ai, Pi Bu, Yue Cao, Yingyao Wang, Jihao Gu, Jingxuan Xing, Zekun Zhu, Wei Jiang, Zhicheng Zheng, Jun Song, Yuning Jiang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.19679)  

**Abstract**: Recent advances in Vision-Language Models (VLMs) have enabled mobile agents to perceive and interact with real-world mobile environments based on human instructions. However, the current fully autonomous paradigm poses potential safety risks when model understanding or reasoning capabilities are insufficient. To address this challenge, we first introduce \textbf{InquireBench}, a comprehensive benchmark specifically designed to evaluate mobile agents' capabilities in safe interaction and proactive inquiry with users, encompassing 5 categories and 22 sub-categories, where most existing VLM-based agents demonstrate near-zero performance. In this paper, we aim to develop an interactive system that actively seeks human confirmation at critical decision points. To achieve this, we propose \textbf{InquireMobile}, a novel model inspired by reinforcement learning, featuring a two-stage training strategy and an interactive pre-action reasoning mechanism. Finally, our model achieves an 46.8% improvement in inquiry success rate and the best overall success rate among existing baselines on InquireBench. We will open-source all datasets, models, and evaluation codes to facilitate development in both academia and industry. 

---
# ReST-RL: Achieving Accurate Code Reasoning of LLMs with Optimized Self-Training and Decoding 

**Authors**: Sining Zhoubian, Dan Zhang, Yuxiao Dong, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.19576)  

**Abstract**: With respect to improving the reasoning accuracy of LLMs, the representative reinforcement learning (RL) method GRPO faces failure due to insignificant reward variance, while verification methods based on process reward models (PRMs) suffer from difficulties with training data acquisition and verification effectiveness. To tackle these problems, this paper introduces ReST-RL, a unified LLM RL paradigm that significantly improves LLM's code reasoning ability by combining an improved GRPO algorithm with a meticulously designed test time decoding method assisted by a value model (VM). As the first stage of policy reinforcement, ReST-GRPO adopts an optimized ReST algorithm to filter and assemble high-value training data, increasing the reward variance of GRPO sampling, thus improving the effectiveness and efficiency of training. After the basic reasoning ability of LLM policy has been improved, we further propose a test time decoding optimization method called VM-MCTS. Through Monte-Carlo Tree Search (MCTS), we collect accurate value targets with no annotation required, on which VM training is based. When decoding, the VM is deployed by an adapted MCTS algorithm to provide precise process signals as well as verification scores, assisting the LLM policy to achieve high reasoning accuracy. We validate the effectiveness of the proposed RL paradigm through extensive experiments on coding problems. Upon comparison, our approach significantly outperforms other reinforcement training baselines (e.g., naive GRPO and ReST-DPO), as well as decoding and verification baselines (e.g., PRM-BoN and ORM-MCTS) on well-known coding benchmarks of various levels (e.g., APPS, BigCodeBench, and HumanEval), indicating its power to strengthen the reasoning ability of LLM policies. Codes for our project can be found at this https URL. 

---
# Improving Low-Resource Translation with Dictionary-Guided Fine-Tuning and RL: A Spanish-to-Wayuunaiki Study 

**Authors**: Manuel Mosquera, Melissa Robles, Johan Rodriguez, Ruben Manrique  

**Link**: [PDF](https://arxiv.org/pdf/2508.19481)  

**Abstract**: Low-resource machine translation remains a significant challenge for large language models (LLMs), which often lack exposure to these languages during pretraining and have limited parallel data for fine-tuning. We propose a novel approach that enhances translation for low-resource languages by integrating an external dictionary tool and training models end-to-end using reinforcement learning, in addition to supervised fine-tuning. Focusing on the Spanish-Wayuunaiki language pair, we frame translation as a tool-augmented decision-making problem in which the model can selectively consult a bilingual dictionary during generation. Our method combines supervised instruction tuning with Guided Reward Policy Optimization (GRPO), enabling the model to learn both when and how to use the tool effectively. BLEU similarity scores are used as rewards to guide this learning process. Preliminary results show that our tool-augmented models achieve up to +3.37 BLEU improvement over previous work, and a 18% relative gain compared to a supervised baseline without dictionary access, on the Spanish-Wayuunaiki test set from the AmericasNLP 2025 Shared Task. We also conduct ablation studies to assess the effects of model architecture and training strategy, comparing Qwen2.5-0.5B-Instruct with other models such as LLaMA and a prior NLLB-based system. These findings highlight the promise of combining LLMs with external tools and the role of reinforcement learning in improving translation quality in low-resource language settings. 

---
# RL-Finetuned LLMs for Privacy-Preserving Synthetic Rewriting 

**Authors**: Zhan Shi, Yefeng Yuan, Yuhong Liu, Liang Cheng, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2508.19286)  

**Abstract**: The performance of modern machine learning systems depends on access to large, high-quality datasets, often sourced from user-generated content or proprietary, domain-specific corpora. However, these rich datasets inherently contain sensitive personal information, raising significant concerns about privacy, data security, and compliance with regulatory frameworks. While conventional anonymization techniques can remove explicit identifiers, such removal may result in performance drop in downstream machine learning tasks. More importantly, simple anonymization may not be effective against inference attacks that exploit implicit signals such as writing style, topical focus, or demographic cues, highlighting the need for more robust privacy safeguards during model training. To address the challenging issue of balancing user privacy and data utility, we propose a reinforcement learning framework that fine-tunes a large language model (LLM) using a composite reward function that jointly optimizes for explicit and implicit privacy, semantic fidelity, and output diversity. To effectively capture population level regularities, the privacy reward combines semantic cues with structural patterns derived from a minimum spanning tree (MST) over latent representations. By modeling these privacy-sensitive signals in their distributional context, the proposed approach guides the model to generate synthetic rewrites that preserve utility while mitigating privacy risks. Empirical results show that the proposed method significantly enhances author obfuscation and privacy metrics without degrading semantic quality, providing a scalable and model-agnostic solution for privacy preserving data generation in the era of large language models. 

---
# Refining Text Generation for Realistic Conversational Recommendation via Direct Preference Optimization 

**Authors**: Manato Tajiri, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2508.19918)  

**Abstract**: Conversational Recommender Systems (CRSs) aim to elicit user preferences via natural dialogue to provide suitable item recommendations. However, current CRSs often deviate from realistic human interactions by rapidly recommending items in brief sessions. This work addresses this gap by leveraging Large Language Models (LLMs) to generate dialogue summaries from dialogue history and item recommendation information from item description. This approach enables the extraction of both explicit user statements and implicit preferences inferred from the dialogue context. We introduce a method using Direct Preference Optimization (DPO) to ensure dialogue summary and item recommendation information are rich in information crucial for effective recommendations. Experiments on two public datasets validate our method's effectiveness in fostering more natural and realistic conversational recommendation this http URL implementation is publicly available at:this https URL 

---
# Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning 

**Authors**: Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Hinrich Schütze, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.19828)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of NLP tasks, but they remain fundamentally stateless, constrained by limited context windows that hinder long-horizon reasoning. Recent efforts to address this limitation often augment LLMs with an external memory bank, yet most existing pipelines are static and heuristic-driven, lacking any learned mechanism for deciding what to store, update, or retrieve. We present Memory-R1, a reinforcement learning (RL) framework that equips LLMs with the ability to actively manage and utilize external memory through two specialized agents: a Memory Manager that learns to perform structured memory operations {ADD, UPDATE, DELETE, NOOP}, and an Answer Agent that selects the most relevant entries and reasons over them to produce an answer. Both agents are fine-tuned with outcome-driven RL (PPO and GRPO), enabling adaptive memory management and use with minimal supervision. With as few as 152 question-answer pairs and a corresponding temporal memory bank for training, Memory-R1 outperforms the most competitive existing baseline and demonstrates strong generalization across diverse question types and LLM backbones. Beyond presenting an effective approach, this work provides insights into how RL can unlock more agentic, memory-aware behaviors in LLMs, pointing toward richer, more persistent reasoning systems. 

---
