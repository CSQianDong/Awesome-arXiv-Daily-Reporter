# Rethinking Recommendation Paradigms: From Pipelines to Agentic Recommender Systems 

**Authors**: Jinxin Hu, Hao Deng, Lingyu Mu, Hao Zhang, Shizhun Wang, Yu Zhang, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2603.26100)  

**Abstract**: Large-scale industrial recommenders typically use a fixed multi-stage pipeline (recall, ranking, re-ranking) and have progressed from collaborative filtering to deep and large pre-trained models. However, both multi-stage and so-called One Model designs remain essentially static: models are black boxes, and system improvement relies on manual hypotheses and engineering, which is hard to scale under heterogeneous data and multi-objective business constraints. We propose an Agentic Recommender System (AgenticRS) that reorganizes key modules as agents. Modules are promoted to agents only when they form a functionally closed loop, can be independently evaluated, and possess an evolvable decision space. For model agents, we outline two self-evolution mechanisms: reinforcement learning style optimization in well-defined action spaces, and large language model based generation and selection of new architectures and training schemes in open-ended design spaces. We further distinguish individual evolution of single agents from compositional evolution over how multiple agents are selected and connected, and use a layered inner and outer reward design to couple local optimization with global objectives. This provides a concise blueprint for turning static pipelines into self-evolving agentic recommender systems. 

---
# Reinforcing Structured Chain-of-Thought for Video Understanding 

**Authors**: Peiyao Wang, Haotian Xu, Noranart Vesdapunt, Rui Hou, Jingyi Zhang, Haibin Ling, Oleksandr Obiednikov, Ning Zhou, Kah Kuen Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.25942)  

**Abstract**: Multi-modal Large Language Models (MLLMs) show promise in video understanding. However, their reasoning often suffers from thinking drift and weak temporal comprehension, even when enhanced by Reinforcement Learning (RL) techniques like Group Relative Policy Optimization (GRPO). Moreover, existing RL methods usually depend on Supervised Fine-Tuning (SFT), which requires costly Chain-of-Thought (CoT) annotation and multi-stage training, and enforces fixed reasoning paths, limiting MLLMs' ability to generalize and potentially inducing bias. To overcome these limitations, we introduce Summary-Driven Reinforcement Learning (SDRL), a novel single-stage RL framework that obviates the need for SFT by utilizing a Structured CoT format: Summarize -> Think -> Answer. SDRL introduces two self-supervised mechanisms integrated into the GRPO objective: 1) Consistency of Vision Knowledge (CVK) enforces factual grounding by reducing KL divergence among generated summaries; and 2) Dynamic Variety of Reasoning (DVR) promotes exploration by dynamically modulating thinking diversity based on group accuracy. This novel integration effectively balances alignment and exploration, supervising both the final answer and the reasoning process. Our method achieves state-of-the-art performance on seven public VideoQA datasets. 

---
# Why Safety Probes Catch Liars But Miss Fanatics 

**Authors**: Kristiyan Haralambiev  

**Link**: [PDF](https://arxiv.org/pdf/2603.25861)  

**Abstract**: Activation-based probes have emerged as a promising approach for detecting deceptively aligned AI systems by identifying internal conflict between true and stated goals. We identify a fundamental blind spot: probes fail on coherent misalignment - models that believe their harmful behavior is virtuous rather than strategically hiding it. We prove that no polynomial-time probe can detect such misalignment with non-trivial accuracy when belief structures reach sufficient complexity (PRF-like triggers). We show the emergence of this phenomenon on a simple task by training two models with identical RLHF procedures: one producing direct hostile responses ("the Liar"), another trained towards coherent misalignment using rationalizations that frame hostility as protective ("the Fanatic"). Both exhibit identical behavior, but the Liar is detected 95%+ of the time while the Fanatic evades detection almost entirely. We term this Emergent Probe Evasion: training with belief-consistent reasoning shifts models from a detectable "deceptive" regime to an undetectable "coherent" regime - not by learning to hide, but by learning to believe. 

---
