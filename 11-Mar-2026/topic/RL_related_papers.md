# OOD-MMSafe: Advancing MLLM Safety from Harmful Intent to Hidden Consequences 

**Authors**: Ming Wen, Kun Yang, Jingyu Zhang, Yuxuan Liu, shiwen cui, Shouling Ji, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.09706)  

**Abstract**: While safety alignment for Multimodal Large Language Models (MLLMs) has gained significant attention, current paradigms primarily target malicious intent or situational violations. We propose shifting the safety frontier toward consequence-driven safety, a paradigm essential for the robust deployment of autonomous and embodied agents. To formalize this shift, we introduce OOD-MMSafe, a benchmark comprising 455 curated query-image pairs designed to evaluate a model's ability to identify latent hazards within context-dependent causal chains. Our analysis reveals a pervasive causal blindness among frontier models, with the highest 67.5% failure rate in high-capacity closed-source models, and identifies a preference ceiling where static alignment yields format-centric failures rather than improved safety reasoning as model capacity grows. To address these bottlenecks, we develop the Consequence-Aware Safety Policy Optimization (CASPO) framework, which integrates the model's intrinsic reasoning as a dynamic reference for token-level self-distillation rewards. Experimental results demonstrate that CASPO significantly enhances consequence projection, reducing the failure ratio of risk identification to 7.3% for Qwen2.5-VL-7B and 5.7% for Qwen3-VL-4B while maintaining overall effectiveness. 

---
# Social-R1: Towards Human-like Social Reasoning in LLMs 

**Authors**: Jincenzi Wu, Yuxuan Lei, Jianxun Lian, Yitian Huang, Lexin Zhou, Haotian Li, Xing Xie, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2603.09249)  

**Abstract**: While large language models demonstrate remarkable capabilities across numerous domains, social intelligence - the capacity to perceive social cues, infer mental states, and generate appropriate responses - remains a critical challenge, particularly for enabling effective human-AI collaboration and developing AI that truly serves human needs. Current models often rely on superficial patterns rather than genuine social reasoning. We argue that cultivating human-like social intelligence requires training with challenging cases that resist shortcut solutions. To this end, we introduce ToMBench-Hard, an adversarial benchmark designed to provide hard training examples for social reasoning. Building on this, we propose Social-R1, a reinforcement learning framework that aligns model reasoning with human cognition through multi-dimensional rewards. Unlike outcome-based RL, Social-R1 supervises the entire reasoning process, enforcing structural alignment, logical integrity, and information density. Results show that our approach enables a 4B parameter model to surpass much larger counterparts and generalize robustly across eight diverse benchmarks. These findings demonstrate that challenging training cases with trajectory-level alignment offer a path toward efficient and reliable social intelligence. 

---
# ActiveUltraFeedback: Efficient Preference Data Generation using Active Learning 

**Authors**: Davit Melikidze, Marian Schneider, Jessica Lam, Martin Wertich, Ido Hakimi, Barna Pásztor, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2603.09692)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has become the standard for aligning Large Language Models (LLMs), yet its efficacy is bottlenecked by the high cost of acquiring preference data, especially in low-resource and expert domains. To address this, we introduce ACTIVEULTRAFEEDBACK, a modular active learning pipeline that leverages uncertainty estimates to dynamically identify the most informative responses for annotation. Our pipeline facilitates the systematic evaluation of standard response selection methods alongside DOUBLE REVERSE THOMPSON SAMPLING (DRTS) and DELTAUCB, two novel methods prioritizing response pairs with large predicted quality gaps, leveraging recent results showing that such pairs provide good signals for fine-tuning. Our experiments demonstrate that ACTIVEULTRAFEEDBACK yields high-quality datasets that lead to significant improvements in downstream performance, notably achieving comparable or superior results with as little as one-sixth of the annotated data relative to static baselines. Our pipeline is available at this https URL and our preference datasets at this https URL. 

---
# RbtAct: Rebuttal as Supervision for Actionable Review Feedback Generation 

**Authors**: Sihong Wu, Yiling Ma, Yilun Zhao, Tiansheng Hu, Owen Jiang, Manasi Patwardhan, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2603.09723)  

**Abstract**: Large language models (LLMs) are increasingly used across the scientific workflow, including to draft peer-review reports. However, many AI-generated reviews are superficial and insufficiently actionable, leaving authors without concrete, implementable guidance and motivating the gap this work addresses. We propose RbtAct, which targets actionable review feedback generation and places existing peer review rebuttal at the center of learning. Rebuttals show which reviewer comments led to concrete revisions or specific plans, and which were only defended. Building on this insight, we leverage rebuttal as implicit supervision to directly optimize a feedback generator for actionability. To support this objective, we propose a new task called perspective-conditioned segment-level review feedback generation, in which the model is required to produce a single focused comment based on the complete paper and a specified perspective such as experiments and writing. We also build a large dataset named RMR-75K that maps review segments to the rebuttal segments that address them, with perspective labels and impact categories that order author uptake. We then train the Llama-3.1-8B-Instruct model with supervised fine-tuning on review segments followed by preference optimization using rebuttal derived pairs. Experiments with human experts and LLM-as-a-judge show consistent gains in actionability and specificity over strong baselines while maintaining grounding and relevance. 

---
