# EDGE-GRPO: Entropy-Driven GRPO with Guided Error Correction for Advantage Diversity 

**Authors**: Xingjian Zhang, Siwei Wen, Wenjun Wu, Lei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.21848)  

**Abstract**: Large Language Models (LLMs) have made remarkable progress in enhancing step-by-step reasoning through reinforcement learning. However, the Group Relative Policy Optimization (GRPO) algorithm, which relies on sparse reward rules, often encounters the issue of identical rewards within groups, leading to the advantage collapse problem. Existing works typically address this challenge from two perspectives: enforcing model reflection to enhance response diversity, and introducing internal feedback to augment the training signal (advantage). In this work, we begin by analyzing the limitations of model reflection and investigating the policy entropy of responses at the fine-grained sample level. Based on our experimental findings, we propose the EDGE-GRPO algorithm, which adopts \textbf{E}ntropy-\textbf{D}riven Advantage and \textbf{G}uided \textbf{E}rror Correction to effectively mitigate the problem of advantage collapse. Extensive experiments on several main reasoning benchmarks demonstrate the effectiveness and superiority of our approach. It is available at this https URL. 

---
# Reasoning Language Models for Root Cause Analysis in 5G Wireless Networks 

**Authors**: Mohamed Sana, Nicola Piovesan, Antonio De Domenico, Yibin Kang, Haozhe Zhang, Merouane Debbah, Fadhel Ayed  

**Link**: [PDF](https://arxiv.org/pdf/2507.21974)  

**Abstract**: Root Cause Analysis (RCA) in mobile networks remains a challenging task due to the need for interpretability, domain expertise, and causal reasoning. In this work, we propose a lightweight framework that leverages Large Language Models (LLMs) for RCA. To do so, we introduce TeleLogs, a curated dataset of annotated troubleshooting problems designed to benchmark RCA capabilities. Our evaluation reveals that existing open-source reasoning LLMs struggle with these problems, underscoring the need for domain-specific adaptation. To address this issue, we propose a two-stage training methodology that combines supervised fine-tuning with reinforcement learning to improve the accuracy and reasoning quality of LLMs. The proposed approach fine-tunes a series of RCA models to integrate domain knowledge and generate structured, multi-step diagnostic explanations, improving both interpretability and effectiveness. Extensive experiments across multiple LLM sizes show significant performance gains over state-of-the-art reasoning and non-reasoning models, including strong generalization to randomized test variants. These results demonstrate the promise of domain-adapted, reasoning-enhanced LLMs for practical and explainable RCA in network operation and management. 

---
# MoHoBench: Assessing Honesty of Multimodal Large Language Models via Unanswerable Visual Questions 

**Authors**: Yanxu Zhu, Shitong Duan, Xiangxu Zhang, Jitao Sang, Peng Zhang, Tun Lu, Xiao Zhou, Jing Yao, Xiaoyuan Yi, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.21503)  

**Abstract**: Recently Multimodal Large Language Models (MLLMs) have achieved considerable advancements in vision-language tasks, yet produce potentially harmful or untrustworthy content. Despite substantial work investigating the trustworthiness of language models, MMLMs' capability to act honestly, especially when faced with visually unanswerable questions, remains largely underexplored. This work presents the first systematic assessment of honesty behaviors across various MLLMs. We ground honesty in models' response behaviors to unanswerable visual questions, define four representative types of such questions, and construct MoHoBench, a large-scale MMLM honest benchmark, consisting of 12k+ visual question samples, whose quality is guaranteed by multi-stage filtering and human verification. Using MoHoBench, we benchmarked the honesty of 28 popular MMLMs and conducted a comprehensive analysis. Our findings show that: (1) most models fail to appropriately refuse to answer when necessary, and (2) MMLMs' honesty is not solely a language modeling issue, but is deeply influenced by visual information, necessitating the development of dedicated methods for multimodal honesty alignment. Therefore, we implemented initial alignment methods using supervised and preference learning to improve honesty behavior, providing a foundation for future work on trustworthy MLLMs. Our data and code can be found at this https URL. 

---
# Teaching Language Models To Gather Information Proactively 

**Authors**: Tenghao Huang, Sihao Chen, Muhao Chen, Jonathan May, Longqi Yang, Mengting Wan, Pei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.21389)  

**Abstract**: Large language models (LLMs) are increasingly expected to function as collaborative partners, engaging in back-and-forth dialogue to solve complex, ambiguous problems. However, current LLMs often falter in real-world settings, defaulting to passive responses or narrow clarifications when faced with incomplete or under-specified prompts, falling short of proactively gathering the missing information that is crucial for high-quality solutions. In this work, we introduce a new task paradigm: proactive information gathering, where LLMs must identify gaps in the provided context and strategically elicit implicit user knowledge through targeted questions. To systematically study and train this capability, we design a scalable framework that generates partially specified, real-world tasks, masking key information and simulating authentic ambiguity. Within this setup, our core innovation is a reinforcement finetuning strategy that rewards questions that elicit genuinely new, implicit user information -- such as hidden domain expertise or fine-grained requirements -- that would otherwise remain unspoken. Experiments demonstrate that our trained Qwen-2.5-7B model significantly outperforms o3-mini by 18% on automatic evaluation metrics. More importantly, human evaluation reveals that clarification questions and final outlines generated by our model are favored by human annotators by 42% and 28% respectively. Together, these results highlight the value of proactive clarification in elevating LLMs from passive text generators to genuinely collaborative thought partners. 

---
# NPO: Learning Alignment and Meta-Alignment through Structured Human Feedback 

**Authors**: Madhava Gaikwad, Ashwini Ramchandra Doke  

**Link**: [PDF](https://arxiv.org/pdf/2507.21131)  

**Abstract**: We present NPO, an alignment-aware learning framework that operationalizes feedback-driven adaptation in human-in-the-loop decision systems. Unlike prior approaches that treat alignment as a static or post-hoc property, NPO introduces a formalization of alignment loss that is measurable, supervisable, and reducible under structured feedback. In parallel, we propose meta-alignment as the fidelity of the monitoring process that governs retraining or override triggers, and show that it is formally reducible to primary alignment via threshold fidelity. Our implementation spans a scalable operational loop involving scenario scoring, threshold tuning, policy validation, and structured feedback ingestion, including "likes", overrides, and abstentions. We provide formal convergence results under stochastic feedback and show that both alignment loss and monitoring fidelity converge additively. Empirically, NPO demonstrates measurable value in hyperscale deployment settings. A simulation-based artifact and ablation studies further illustrate the theoretical principles in action. Together, NPO offers a compact, inspectable architecture for continual alignment monitoring, helping bridge theoretical alignment guarantees with practical reliability in dynamic environments. 

---
# ChemDFM-R: An Chemical Reasoner LLM Enhanced with Atomized Chemical Knowledge 

**Authors**: Zihan Zhao, Bo Chen, Ziping Wan, Lu Chen, Xuanze Lin, Shiyang Yu, Situo Zhang, Da Ma, Zichen Zhu, Danyang Zhang, Huayang Wang, Zhongyang Dai, Liyang Wen, Xin Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21990)  

**Abstract**: While large language models (LLMs) have achieved impressive progress, their application in scientific domains such as chemistry remains hindered by shallow domain understanding and limited reasoning capabilities. In this work, we focus on the specific field of chemistry and develop a Chemical Reasoner LLM, ChemDFM-R. We first construct a comprehensive dataset of atomized knowledge points to enhance the model's understanding of the fundamental principles and logical structure of chemistry. Then, we propose a mix-sourced distillation strategy that integrates expert-curated knowledge with general-domain reasoning skills, followed by domain-specific reinforcement learning to enhance chemical reasoning. Experiments on diverse chemical benchmarks demonstrate that ChemDFM-R achieves state-of-the-art performance while providing interpretable, rationale-driven outputs. Further case studies illustrate how explicit reasoning chains significantly improve the reliability, transparency, and practical utility of the model in real-world human-AI collaboration scenarios. 

---
# Post-Training Large Language Models via Reinforcement Learning from Self-Feedback 

**Authors**: Carel van Niekerk, Renato Vukovic, Benjamin Matthias Ruppik, Hsien-chin Lin, Milica Gašić  

**Link**: [PDF](https://arxiv.org/pdf/2507.21931)  

**Abstract**: Large Language Models (LLMs) often produce plausible but poorly-calibrated answers, limiting their reliability on reasoning-intensive tasks. We present Reinforcement Learning from Self-Feedback (RLSF), a post-training stage that uses the model's own confidence as an intrinsic reward, mimicking how humans learn in the absence of external feedback. After a frozen LLM generates several chain-of-thought solutions, we define and compute the confidence of each final answer span and rank the traces accordingly. These synthetic preferences are then used to fine-tune the policy with standard preference optimization, similar to RLHF yet requiring no human labels, gold answers, or externally curated rewards.
RLSF simultaneously (i) refines the model's probability estimates -- restoring well-behaved calibration -- and (ii) strengthens step-by-step reasoning, yielding improved performance on arithmetic reasoning and multiple-choice question answering.
By turning a model's own uncertainty into useful self-feedback, RLSF affirms reinforcement learning on intrinsic model behaviour as a principled and data-efficient component of the LLM post-training pipeline and warrents further research in intrinsic rewards for LLM post-training. 

---
# LLM-Adapted Interpretation Framework for Machine Learning Models 

**Authors**: Yuqi Jin, Zihan Hu, Weiteng Zhang, Weihao Xie, Jianwei Shuai, Xian Shen, Zhen Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.21179)  

**Abstract**: Background & Aims: High-performance machine learning models like XGBoost are often "black boxes," limiting their clinical adoption due to a lack of interpretability. This study aims to bridge the gap between predictive accuracy and narrative transparency for sarcopenia risk assessment. Methods: We propose the LLM-Adapted Interpretation Framework (LAI-ML), a novel knowledge distillation architecture. LAI-ML transforms feature attributions from a trained XGBoost model into a probabilistic format using specialized techniques (HAGA and CACS). A Large Language Model (LLM), guided by a reinforcement learning loop and case-based retrieval, then generates data-faithful diagnostic narratives. Results: The LAI-ML framework achieved 83% prediction accuracy, significantly outperforming the baseline XGBoost model, 13% higher. Notably, the LLM not only replicated the teacher model's logic but also corrected its predictions in 21.7% of discordant cases, demonstrating enhanced reasoning. Conclusion: LAI-ML effectively translates opaque model predictions into trustworthy and interpretable clinical insights, offering a deployable solution to the "black-box" problem in medical AI. 

---
# MaPPO: Maximum a Posteriori Preference Optimization with Prior Knowledge 

**Authors**: Guangchen Lan, Sipeng Zhang, Tianle Wang, Yuwei Zhang, Daoan Zhang, Xinpeng Wei, Xiaoman Pan, Hongming Zhang, Dong-Jun Han, Christopher G. Brinton  

**Link**: [PDF](https://arxiv.org/pdf/2507.21183)  

**Abstract**: As the era of large language models (LLMs) on behalf of users unfolds, Preference Optimization (PO) methods have become a central approach to aligning LLMs with human preferences and improving performance. We propose Maximum a Posteriori Preference Optimization (MaPPO), a framework for learning from preferences that explicitly incorporates prior reward knowledge into the optimization objective. While existing methods such as Direct Preference Optimization (DPO) and its variants treat preference learning as a Maximum Likelihood Estimation (MLE) problem, MaPPO extends this paradigm by integrating prior reward estimates into a principled Maximum a Posteriori (MaP) objective. This not only generalizes DPO and its variants, but also enhances alignment by mitigating the oversimplified binary classification of responses. More importantly, MaPPO introduces no additional hyperparameter, and supports preference optimization in both offline and online settings. In addition, MaPPO can be used as a plugin with consistent improvement on DPO variants, including widely used SimPO, IPO, and CPO. Extensive empirical evaluations of different model sizes and model series on three standard benchmarks, including MT-Bench, AlpacaEval 2.0, and Arena-Hard, demonstrate consistent improvements in alignment performance without sacrificing computational efficiency. 

---
# AutoTIR: Autonomous Tools Integrated Reasoning via Reinforcement Learning 

**Authors**: Yifan Wei, Xiaoyan Yu, Yixuan Weng, Tengfei Pan, Angsheng Li, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2507.21836)  

**Abstract**: Large Language Models (LLMs), when enhanced through reasoning-oriented post-training, evolve into powerful Large Reasoning Models (LRMs). Tool-Integrated Reasoning (TIR) further extends their capabilities by incorporating external tools, but existing methods often rely on rigid, predefined tool-use patterns that risk degrading core language competence. Inspired by the human ability to adaptively select tools, we introduce AutoTIR, a reinforcement learning framework that enables LLMs to autonomously decide whether and which tool to invoke during the reasoning process, rather than following static tool-use strategies. AutoTIR leverages a hybrid reward mechanism that jointly optimizes for task-specific answer correctness, structured output adherence, and penalization of incorrect tool usage, thereby encouraging both precise reasoning and efficient tool integration. Extensive evaluations across diverse knowledge-intensive, mathematical, and general language modeling tasks demonstrate that AutoTIR achieves superior overall performance, significantly outperforming baselines and exhibits superior generalization in tool-use behavior. These results highlight the promise of reinforcement learning in building truly generalizable and scalable TIR capabilities in LLMs. The code and data are available at this https URL. 

---
# Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning 

**Authors**: Haoran Luo, Haihong E, Guanting Chen, Qika Lin, Yikai Guo, Fangzhi Xu, Zemin Kuang, Meina Song, Xiaobao Wu, Yifan Zhu, Luu Anh Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21892)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucination in LLMs by incorporating external knowledge, but relies on chunk-based retrieval that lacks structural semantics. GraphRAG methods improve RAG by modeling knowledge as entity-relation graphs, but still face challenges in high construction cost, fixed one-time retrieval, and reliance on long-context reasoning and prompt design. To address these challenges, we propose Graph-R1, an agentic GraphRAG framework via end-to-end reinforcement learning (RL). It introduces lightweight knowledge hypergraph construction, models retrieval as a multi-turn agent-environment interaction, and optimizes the agent process via an end-to-end reward mechanism. Experiments on standard RAG datasets show that Graph-R1 outperforms traditional GraphRAG and RL-enhanced RAG methods in reasoning accuracy, retrieval efficiency, and generation quality. 

---
# Libra: Assessing and Improving Reward Model by Learning to Think 

**Authors**: Meng Zhou, Bei Li, Jiahao Liu, Xiaowen Shi, Yang Bai, Rongxiang Weng, Jingang Wang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.21645)  

**Abstract**: Reinforcement learning (RL) has significantly improved the reasoning ability of large language models. However, current reward models underperform in challenging reasoning scenarios and predominant RL training paradigms rely on rule-based or reference-based rewards, which impose two critical limitations: 1) the dependence on finely annotated reference answer to attain rewards; and 2) the requirement for constrained output format. These limitations fundamentally hinder further RL data scaling and sustained enhancement of model reasoning performance. To address these limitations, we propose a comprehensive framework for evaluating and improving the performance of reward models in complex reasoning scenarios. We first present a reasoning-oriented benchmark (Libra Bench), systematically constructed from a diverse collection of challenging mathematical problems and advanced reasoning models, to address the limitations of existing reward model benchmarks in reasoning scenarios. We further introduce a novel approach for improving the generative reward model via learning-to-think methodologies. Based on the proposed approach, we develop Libra-RM series, a collection of generative reward models with reasoning capabilities that achieve state-of-the-art results on various benchmarks. Comprehensive downstream experiments are conducted and the experimental results demonstrate the correlation between our Libra Bench and downstream application, and the potential of Libra-RM to further improve reasoning models with unlabeled data. 

---
# Rewrite-to-Rank: Optimizing Ad Visibility via Retrieval-Aware Text Rewriting 

**Authors**: Chloe Ho, Ishneet Sukhvinder Singh, Diya Sharma, Tanvi Reddy Anumandla, Michael Lu, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.21099)  

**Abstract**: Search algorithms and user query relevance have given LLMs the ability to return relevant information, but the effect of content phrasing on ad visibility remains underexplored. We investigate how LLM-based rewriting of advertisements can improve their ranking in retrieval systems and inclusion in generated LLM responses, without modifying the retrieval model itself. We introduce a supervised fine-tuning framework with a custom loss balancing semantic relevance and content fidelity. To evaluate effectiveness, we propose two metrics: DeltaMRR@K (ranking improvement) and DeltaDIR@K (inclusion frequency improvement). Our approach presents a scalable method to optimize ad phrasing, enhancing visibility in retrieval-based LLM workflows. Experiments across both instruction-based and few-shot prompting demonstrate that PPO trained models outperform both prompt engineering and supervised fine-tuning in most cases, achieving up to a 2.79 DeltaDIR@5 and 0.0073 DeltaMRR@5 in instruction-based prompting. These results highlight the importance of how the ad is written before retrieval and prompt format and reinforcement learning in effective ad rewriting for LLM integrated retrieval systems. 

---
