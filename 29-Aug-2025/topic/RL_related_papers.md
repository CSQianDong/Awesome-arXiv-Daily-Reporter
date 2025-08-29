# Leveraging Generative Models for Real-Time Query-Driven Text Summarization in Large-Scale Web Search 

**Authors**: Zeyu Xiong, Yixuan Nan, Li Gao, Hengzhu Tang, Shuaiqiang Wang, Junfeng Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20559)  

**Abstract**: In the dynamic landscape of large-scale web search, Query-Driven Text Summarization (QDTS) aims to generate concise and informative summaries from textual documents based on a given query, which is essential for improving user engagement and facilitating rapid decision-making. Traditional extractive summarization models, based primarily on ranking candidate summary segments, have been the dominant approach in industrial applications. However, these approaches suffer from two key limitations: 1) The multi-stage pipeline often introduces cumulative information loss and architectural bottlenecks due to its weakest component; 2) Traditional models lack sufficient semantic understanding of both user queries and documents, particularly when dealing with complex search intents. In this study, we propose a novel framework to pioneer the application of generative models to address real-time QDTS in industrial web search. Our approach integrates large model distillation, supervised fine-tuning, direct preference optimization, and lookahead decoding to transform a lightweight model with only 0.1B parameters into a domain-specialized QDTS expert. Evaluated on multiple industry-relevant metrics, our model outperforms the production baseline and achieves a new state of the art. Furthermore, it demonstrates excellent deployment efficiency, requiring only 334 NVIDIA L20 GPUs to handle \textasciitilde50,000 queries per second under 55~ms average latency per query. 

---
# SageLM: A Multi-aspect and Explainable Large Language Model for Speech Judgement 

**Authors**: Yuan Ge, Junxiang Zhang, Xiaoqian Liu, Bei Li, Xiangnan Ma, Chenglong Wang, Kaiyang Ye, Yangfan Du, Linfeng Zhang, Yuxin Huang, Tong Xiao, Zhengtao Yu, JingBo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20916)  

**Abstract**: Speech-to-Speech (S2S) Large Language Models (LLMs) are foundational to natural human-computer interaction, enabling end-to-end spoken dialogue systems. However, evaluating these models remains a fundamental challenge. We propose \texttt{SageLM}, an end-to-end, multi-aspect, and explainable speech LLM for comprehensive S2S LLMs evaluation. First, unlike cascaded approaches that disregard acoustic features, SageLM jointly assesses both semantic and acoustic dimensions. Second, it leverages rationale-based supervision to enhance explainability and guide model learning, achieving superior alignment with evaluation outcomes compared to rule-based reinforcement learning methods. Third, we introduce \textit{SpeechFeedback}, a synthetic preference dataset, and employ a two-stage training paradigm to mitigate the scarcity of speech preference data. Trained on both semantic and acoustic dimensions, SageLM achieves an 82.79\% agreement rate with human evaluators, outperforming cascaded and SLM-based baselines by at least 7.42\% and 26.20\%, respectively. 

---
# rStar2-Agent: Agentic Reasoning Technical Report 

**Authors**: Ning Shang, Yifei Liu, Yi Zhu, Li Lyna Zhang, Weijiang Xu, Xinyu Guan, Buze Zhang, Bingcheng Dong, Xudong Zhou, Bowen Zhang, Ying Xin, Ziming Miao, Scarlett Li, Fan Yang, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20722)  

**Abstract**: We introduce rStar2-Agent, a 14B math reasoning model trained with agentic reinforcement learning to achieve frontier-level performance. Beyond current long CoT, the model demonstrates advanced cognitive behaviors, such as thinking carefully before using Python coding tools and reflecting on code execution feedback to autonomously explore, verify, and refine intermediate steps in complex problem-solving. This capability is enabled through three key innovations that makes agentic RL effective at scale: (i) an efficient RL infrastructure with a reliable Python code environment that supports high-throughput execution and mitigates the high rollout costs, enabling training on limited GPU resources (64 MI300X GPUs); (ii) GRPO-RoC, an agentic RL algorithm with a Resample-on-Correct rollout strategy that addresses the inherent environment noises from coding tools, allowing the model to reason more effectively in a code environment; (iii) An efficient agent training recipe that starts with non-reasoning SFT and progresses through multi-RL stages, yielding advanced cognitive abilities with minimal compute cost. To this end, rStar2-Agent boosts a pre-trained 14B model to state of the art in only 510 RL steps within one week, achieving average pass@1 scores of 80.6% on AIME24 and 69.8% on AIME25, surpassing DeepSeek-R1 (671B) with significantly shorter responses. Beyond mathematics, rStar2-Agent-14B also demonstrates strong generalization to alignment, scientific reasoning, and agentic tool-use tasks. Code and training recipes are available at this https URL. 

---
# Graph-R1: Unleashing LLM Reasoning with NP-Hard Graph Problems 

**Authors**: Yuyao Wang, Bowen Liu, Jianheng Tang, Nuo Chen, Yuhan Li, Qifan Zhang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20373)  

**Abstract**: Reasoning Large Language Models (RLLMs) have recently achieved remarkable progress on complex reasoning tasks, largely enabled by their long chain-of-thought (Long CoT) capabilities. However, developing these Long CoT behaviors relies heavily on post-training with high-quality datasets, which are typically costly and human-curated (e.g., mathematics and code), leaving scalable alternatives unexplored. In this work, we introduce NP-hard (NPH) graph problems as a novel synthetic training corpus, as they inherently require deep reasoning, extensive exploration, and reflective strategies, which are core characteristics of Long CoT reasoning. Building on this insight, we develop a two-stage post-training framework: (i) Long CoT Supervised Fine-Tuning (SFT) on rejection-sampled NPH graph instances, which substantially enhances reasoning depth, and (ii) Reinforcement Learning (RL) with a fine-grained reward design, which sharpens reasoning efficiency. Our flagship model, Graph-R1-7B, demonstrates strong generalization across mathematics, coding, STEM, and logic, and surpasses QwQ-32B on NPH graph problems in both accuracy and reasoning efficiency. These results position NPH graph problems as an effective and scalable resource for advancing Long CoT reasoning in LLMs, opening a new frontier for LLM post-training. Our implementation is available at this https URL, with models and datasets hosted in our Hugging Face collection HKUST-DSAIL/Graph-R1. 

---
# Improving Alignment in LVLMs with Debiased Self-Judgment 

**Authors**: Sihan Yang, Chenhang Cui, Zihao Zhao, Yiyang Zhou, Weilong Yan, Ying Wei, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20655)  

**Abstract**: The rapid advancements in Large Language Models (LLMs) and Large Visual-Language Models (LVLMs) have opened up new opportunities for integrating visual and linguistic modalities. However, effectively aligning these modalities remains challenging, often leading to hallucinations--where generated outputs are not grounded in the visual input--and raising safety concerns across various domains. Existing alignment methods, such as instruction tuning and preference tuning, often rely on external datasets, human annotations, or complex post-processing, which limit scalability and increase costs. To address these challenges, we propose a novel approach that generates the debiased self-judgment score, a self-evaluation metric created internally by the model without relying on external resources. This enables the model to autonomously improve alignment. Our method enhances both decoding strategies and preference tuning processes, resulting in reduced hallucinations, enhanced safety, and improved overall capability. Empirical results show that our approach significantly outperforms traditional methods, offering a more effective solution for aligning LVLMs. 

---
# Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning 

**Authors**: Weitao Feng, Lixu Wang, Tianyi Wei, Jie Zhang, Chongyang Gao, Sinong Zhan, Peizhuo Lv, Wei Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.20697)  

**Abstract**: As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense. 

---
# Mitigating Hallucinations in Multimodal LLMs via Object-aware Preference Optimization 

**Authors**: Alberto Compagnoni, Davide Caffagni, Nicholas Moratelli, Lorenzo Baraldi, Marcella Cornia, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2508.20181)  

**Abstract**: Multimodal Large Language Models (MLLMs) emerge as a unified interface to address a multitude of tasks, ranging from NLP to computer vision. Despite showcasing state-of-the-art results in many benchmarks, a long-standing issue is the tendency of MLLMs to hallucinate, that is to generate answers to the user's query that are not reflected in the visual input. In this paper, we address the problem of hallucinations as an alignment problem, seeking to steer the MLLM so that it prefers generating content without hallucinations. In contrast to recent approaches that require complicated pipelines to build synthetic preference data for alignment training, often relying on proprietary models, we capitalize on the well-known CHAIR metric, originally proposed to gauge the degree of hallucinations in image captioning. Given a pair of generated answers, we leverage CHAIR to distinguish winner and loser options (i.e., non-hallucinated and hallucinated samples) and fine-tune off-the-shelf MLLMs via Direct Preference Optimization (DPO). The resulting method, which we refer to as CHAIR-DPO, effectively diminishes the amount of hallucinated answers on several hallucination benchmarks, demonstrating the effectiveness of fine-tuning the MLLM with a CHAIR-based reward. Source code and trained models are publicly available at this https URL. 

---
# ChatThero: An LLM-Supported Chatbot for Behavior Change and Therapeutic Support in Addiction Recovery 

**Authors**: Junda Wang, Zonghai Yao, Zhichao Yang, Lingxi Li, Junhui Qian, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20996)  

**Abstract**: Substance use disorders (SUDs) affect over 36 million people worldwide, yet few receive effective care due to stigma, motivational barriers, and limited personalized support. Although large language models (LLMs) show promise for mental-health assistance, most systems lack tight integration with clinically validated strategies, reducing effectiveness in addiction recovery. We present ChatThero, a multi-agent conversational framework that couples dynamic patient modeling with context-sensitive therapeutic dialogue and adaptive persuasive strategies grounded in cognitive behavioral therapy (CBT) and motivational interviewing (MI). We build a high-fidelity synthetic benchmark spanning Easy, Medium, and Hard resistance levels, and train ChatThero with a two-stage pipeline comprising supervised fine-tuning (SFT) followed by direct preference optimization (DPO). In evaluation, ChatThero yields a 41.5\% average gain in patient motivation, a 0.49\% increase in treatment confidence, and resolves hard cases with 26\% fewer turns than GPT-4o, and both automated and human clinical assessments rate it higher in empathy, responsiveness, and behavioral realism. The framework supports rigorous, privacy-preserving study of therapeutic conversation and provides a robust, replicable basis for research and clinical translation. 

---
# IntentionReasoner: Facilitating Adaptive LLM Safeguards through Intent Reasoning and Selective Query Refinement 

**Authors**: Yuanzhe Shen, Zisu Huang, Zhengkang Guo, Yide Liu, Guanxu Chen, Ruicheng Yin, Xiaoqing Zheng, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20151)  

**Abstract**: The rapid advancement of large language models (LLMs) has driven their adoption across diverse domains, yet their ability to generate harmful content poses significant safety challenges. While extensive research has focused on mitigating harmful outputs, such efforts often come at the cost of excessively rejecting harmless prompts. Striking a balance among safety, over-refusal, and utility remains a critical challenge. In this work, we introduce IntentionReasoner, a novel safeguard mechanism that leverages a dedicated guard model to perform intent reasoning, multi-level safety classification, and query rewriting to neutralize potentially harmful intent in edge-case queries. Specifically, we first construct a comprehensive dataset comprising approximately 163,000 queries, each annotated with intent reasoning, safety labels, and rewritten versions. Supervised fine-tuning is then applied to equip the guard model with foundational capabilities in format adherence, intent analysis, and safe rewriting. Finally, we apply a tailored multi-reward optimization strategy that integrates rule-based heuristics and reward model signals within a reinforcement learning framework to further enhance performance. Extensive experiments show that IntentionReasoner excels in multiple safeguard benchmarks, generation quality evaluations, and jailbreak attack scenarios, significantly enhancing safety while effectively reducing over-refusal rates and improving the quality of responses. 

---
# Quantum Verifiable Rewards for Post-Training Qiskit Code Assistant 

**Authors**: Nicolas Dupuis, Adarsh Tiwari, Youssef Mroueh, David Kremer, Ismael Faro, Juan Cruz-Benito  

**Link**: [PDF](https://arxiv.org/pdf/2508.20907)  

**Abstract**: Qiskit is an open-source quantum computing framework that allows users to design, simulate, and run quantum circuits on real quantum hardware. We explore post-training techniques for LLMs to assist in writing Qiskit code. We introduce quantum verification as an effective method for ensuring code quality and executability on quantum hardware. To support this, we developed a synthetic data pipeline that generates quantum problem-unit test pairs and used it to create preference data for aligning LLMs with DPO. Additionally, we trained models using GRPO, leveraging quantum-verifiable rewards provided by the quantum hardware. Our best-performing model, combining DPO and GRPO, surpasses the strongest open-source baselines on the challenging Qiskit-HumanEval-hard benchmark. 

---
# AWorld: Orchestrating the Training Recipe for Agentic AI 

**Authors**: Chengyue Yu, Siyuan Lu, Chenyi Zhuang, Dong Wang, Qintong Wu, Zongyue Li, Runsheng Gan, Chunfeng Wang, Siqi Hou, Gaochi Huang, Wenlong Yan, Lifeng Hong, Aohui Xue, Yanfeng Wang, Jinjie Gu, David Tsai, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20404)  

**Abstract**: The learning from practice paradigm is crucial for developing capable Agentic AI systems, yet it is severely hampered by inefficient experience generation, a bottleneck especially pronounced in complex benchmarks like GAIA. To address this, we introduce AWorld, an open-source system engineered for large-scale agent-environment interaction. By distributing tasks across a cluster, AWorld accelerates experience collection by 14.6x compared to standard single-node, sequential execution. This critical speedup makes extensive reinforcement learning practical and scalable. Leveraging this capability, we trained a Qwen3-32B-based agent that significantly outperforms its base model, increasing its overall GAIA accuracy from 21.59% to 32.23%. On the benchmark's most challenging levels, our agent achieves a score of 16.33%, surpassing the performance of leading proprietary models. Our open-source system and resulting agent provide a practical blueprint for a complete agentic AI training pipeline, from efficient interaction to demonstrable model improvement. 

---
# Inference-Time Alignment Control for Diffusion Models with Reinforcement Learning Guidance 

**Authors**: Luozhijie Jin, Zijie Qiu, Jie Liu, Zijie Diao, Lifeng Qiao, Ning Ding, Alex Lamb, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21016)  

**Abstract**: Denoising-based generative models, particularly diffusion and flow matching algorithms, have achieved remarkable success. However, aligning their output distributions with complex downstream objectives, such as human preferences, compositional accuracy, or data compressibility, remains challenging. While reinforcement learning (RL) fine-tuning methods, inspired by advances in RL from human feedback (RLHF) for large language models, have been adapted to these generative frameworks, current RL approaches are suboptimal for diffusion models and offer limited flexibility in controlling alignment strength after fine-tuning. In this work, we reinterpret RL fine-tuning for diffusion models through the lens of stochastic differential equations and implicit reward conditioning. We introduce Reinforcement Learning Guidance (RLG), an inference-time method that adapts Classifier-Free Guidance (CFG) by combining the outputs of the base and RL fine-tuned models via a geometric average. Our theoretical analysis shows that RLG's guidance scale is mathematically equivalent to adjusting the KL-regularization coefficient in standard RL objectives, enabling dynamic control over the alignment-quality trade-off without further training. Extensive experiments demonstrate that RLG consistently improves the performance of RL fine-tuned models across various architectures, RL algorithms, and downstream tasks, including human preferences, compositional control, compressibility, and text rendering. Furthermore, RLG supports both interpolation and extrapolation, thereby offering unprecedented flexibility in controlling generative alignment. Our approach provides a practical and theoretically sound solution for enhancing and controlling diffusion model alignment at inference. The source code for RLG is publicly available at the Github: this https URL. 

---
# MedGR$^2$: Breaking the Data Barrier for Medical Reasoning via Generative Reward Learning 

**Authors**: Weihai Zhi, Jiayan Guo, Shangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20549)  

**Abstract**: The application of Vision-Language Models (VLMs) in medicine is critically hampered by the scarcity of high-quality, expert-annotated data. Supervised Fine-Tuning (SFT) on existing datasets often leads to poor generalization on unseen modalities and tasks, while Reinforcement Learning (RL), a promising alternative, is stymied by the lack of reliable reward signals in this data-scarce domain. To break this impasse, we introduce Generative Reward Learning for Medical Reasoning (MedGR$^2$), a novel framework that creates a self-improving virtuous cycle. MedGR$^2$ co-develops a data generator and a reward model, enabling the automated, continuous creation of high-quality, multi-modal medical data that serves as both a superior training source for SFT and RL. Our experiments demonstrate that SFT with MedGR$^2$-produced data already surpasses baselines trained on large-scale, human-curated datasets. Crucially, when leveraging this data for RL via Group Relative Policy Optimization (GRPO), our model achieves state-of-the-art cross-modality and cross-task generalization, significantly outperforming specialized RL-based methods. Furthermore, our compact model, empowered by MedGR$^2$, achieves performance competitive with foundation models possessing over 10 times more parameters. MedGR$^2$ presents a new paradigm for data-efficient learning in high-stakes domains, transforming the problem from data scarcity to data generation and unlocking the full potential of RL for building truly generalizable medical AI. 

---
# Towards Better Correctness and Efficiency in Code Generation 

**Authors**: Yunlong Feng, Yang Xu, Xiao Xu, Binyuan Hui, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20124)  

**Abstract**: While code large language models have demonstrated remarkable progress in code generation, the generated code often exhibits poor runtime efficiency, limiting its practical application in performance-sensitive scenarios. To address this limitation, we propose an efficiency-oriented reinforcement learning framework guided by a novel performance reward. Based on this framework, we take a deeper dive into the code efficiency problem, identifying then proposing methods to overcome key bottlenecks: (1) Dynamic exploration overcomes the static data constraints of offline fine-tuning, enabling the discovery of more efficient code implementations. (2) The error-insensitive reinforcement learning method and high-contrast efficiency signals are crucial for mitigating systematic errors and achieving effective optimization. (3) Online exploration is most effective when starting from a high-correctness baseline, as this allows for efficiency improvements without sacrificing accuracy. With these discoveries, we finally propose a two-stage tuning method, which achieves high and balanced performance across correctness and efficiency. The results of experiments show the effectiveness of the method, which improves code correctness by 10.18\% and runtime efficiency by 7.75\% on a 7B model, achieving performance comparable to much larger model. 

---
