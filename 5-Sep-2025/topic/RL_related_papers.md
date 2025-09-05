# EvoEmo: Towards Evolved Emotional Policies for LLM Agents in Multi-Turn Negotiation 

**Authors**: Yunbo Long, Liming Xu, Lukas Beckenbauer, Yuhan Liu, Alexandra Brintrup  

**Link**: [PDF](https://arxiv.org/pdf/2509.04310)  

**Abstract**: Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated that agents can engage in \textit{complex}, \textit{multi-turn} negotiations, opening new avenues for agentic AI. However, existing LLM agents largely overlook the functional role of emotions in such negotiations, instead generating passive, preference-driven emotional responses that make them vulnerable to manipulation and strategic exploitation by adversarial counterparts. To address this gap, we present EvoEmo, an evolutionary reinforcement learning framework that optimizes dynamic emotional expression in negotiations. EvoEmo models emotional state transitions as a Markov Decision Process and employs population-based genetic optimization to evolve high-reward emotion policies across diverse negotiation scenarios. We further propose an evaluation framework with two baselines -- vanilla strategies and fixed-emotion strategies -- for benchmarking emotion-aware negotiation. Extensive experiments and ablation studies show that EvoEmo consistently outperforms both baselines, achieving higher success rates, higher efficiency, and increased buyer savings. This findings highlight the importance of adaptive emotional expression in enabling more effective LLM agents for multi-turn negotiation. 

---
# A Foundation Model for Chest X-ray Interpretation with Grounded Reasoning via Online Reinforcement Learning 

**Authors**: Qika Lin, Yifan Zhu, Bin Pu, Ling Huang, Haoran Luo, Jingying Ma, Zhen Peng, Tianzhe Zhao, Fangzhi Xu, Jian Zhang, Kai He, Zhonghong Ou, Swapnil Mishra, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.03906)  

**Abstract**: Medical foundation models (FMs) have shown tremendous promise amid the rapid advancements in artificial intelligence (AI) technologies. However, current medical FMs typically generate answers in a black-box manner, lacking transparent reasoning processes and locally grounded interpretability, which hinders their practical clinical deployments. To this end, we introduce DeepMedix-R1, a holistic medical FM for chest X-ray (CXR) interpretation. It leverages a sequential training pipeline: initially fine-tuned on curated CXR instruction data to equip with fundamental CXR interpretation capabilities, then exposed to high-quality synthetic reasoning samples to enable cold-start reasoning, and finally refined via online reinforcement learning to enhance both grounded reasoning quality and generation performance. Thus, the model produces both an answer and reasoning steps tied to the image's local regions for each query. Quantitative evaluation demonstrates substantial improvements in report generation (e.g., 14.54% and 31.32% over LLaVA-Rad and MedGemma) and visual question answering (e.g., 57.75% and 23.06% over MedGemma and CheXagent) tasks. To facilitate robust assessment, we propose Report Arena, a benchmarking framework using advanced language models to evaluate answer quality, further highlighting the superiority of DeepMedix-R1. Expert review of generated reasoning steps reveals greater interpretability and clinical plausibility compared to the established Qwen2.5-VL-7B model (0.7416 vs. 0.2584 overall preference). Collectively, our work advances medical FM development toward holistic, transparent, and clinically actionable modeling for CXR interpretation. 

---
# Learning to Deliberate: Meta-policy Collaboration for Agentic LLMs with Multi-agent Reinforcement Learning 

**Authors**: Wei Yang, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2509.03817)  

**Abstract**: Multi-agent systems of large language models (LLMs) show promise for complex reasoning, but their effectiveness is often limited by fixed collaboration protocols. These frameworks typically focus on macro-level orchestration while overlooking agents' internal deliberative capabilities. This critical meta-cognitive blindspot treats agents as passive executors unable to adapt their strategy based on internal cognitive states like uncertainty or confidence. We introduce the Meta-Policy Deliberation Framework (MPDF), where agents learn a decentralized policy over a set of high-level meta-cognitive actions: Persist, Refine, and Concede. To overcome the instability of traditional policy gradients in this setting, we develop SoftRankPO, a novel reinforcement learning algorithm. SoftRankPO stabilizes training by shaping advantages based on the rank of rewards mapped through smooth normal quantiles, making the learning process robust to reward variance. Experiments show that MPDF with SoftRankPO achieves a a 4-5% absolute gain in average accuracy across five mathematical and general reasoning benchmarks compared to six state-of-the-art heuristic and learning-based multi-agent reasoning algorithms. Our work presents a paradigm for learning adaptive, meta-cognitive policies for multi-agent LLM systems, shifting the focus from designing fixed protocols to learning dynamic, deliberative strategies. 

---
# Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning 

**Authors**: Haozhe Wang, Qixin Xu, Che Liu, Junhong Wu, Fangzhen Lin, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.03646)  

**Abstract**: Reinforcement Learning (RL) has proven highly effective at enhancing the complex reasoning abilities of Large Language Models (LLMs), yet underlying mechanisms driving this success remain largely opaque. Our analysis reveals that puzzling phenomena like ``aha moments", ``length-scaling'' and entropy dynamics are not disparate occurrences but hallmarks of an emergent reasoning hierarchy, akin to the separation of high-level strategic planning from low-level procedural execution in human cognition. We uncover a compelling two-phase dynamic: initially, a model is constrained by procedural correctness and must improve its low-level skills. The learning bottleneck then decisively shifts, with performance gains being driven by the exploration and mastery of high-level strategic planning. This insight exposes a core inefficiency in prevailing RL algorithms like GRPO, which apply optimization pressure agnostically and dilute the learning signal across all tokens. To address this, we propose HIerarchy-Aware Credit Assignment (HICRA), an algorithm that concentrates optimization efforts on high-impact planning tokens. HICRA significantly outperforms strong baselines, demonstrating that focusing on this strategic bottleneck is key to unlocking advanced reasoning. Furthermore, we validate semantic entropy as a superior compass for measuring strategic exploration over misleading metrics such as token-level entropy. 

---
# HumAIne-Chatbot: Real-Time Personalized Conversational AI via Reinforcement Learning 

**Authors**: Georgios Makridis, Georgios Fragiadakis, Jorge Oliveira, Tomaz Saraiva, Philip Mavrepis, Georgios Fatouros, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2509.04303)  

**Abstract**: Current conversational AI systems often provide generic, one-size-fits-all interactions that overlook individual user characteristics and lack adaptive dialogue management. To address this gap, we introduce \textbf{HumAIne-chatbot}, an AI-driven conversational agent that personalizes responses through a novel user profiling framework. The system is pre-trained on a diverse set of GPT-generated virtual personas to establish a broad prior over user types. During live interactions, an online reinforcement learning agent refines per-user models by combining implicit signals (e.g. typing speed, sentiment, engagement duration) with explicit feedback (e.g., likes and dislikes). This profile dynamically informs the chatbot dialogue policy, enabling real-time adaptation of both content and style. To evaluate the system, we performed controlled experiments with 50 synthetic personas in multiple conversation domains. The results showed consistent improvements in user satisfaction, personalization accuracy, and task achievement when personalization features were enabled. Statistical analysis confirmed significant differences between personalized and nonpersonalized conditions, with large effect sizes across key metrics. These findings highlight the effectiveness of AI-driven user profiling and provide a strong foundation for future real-world validation. 

---
# QuesGenie: Intelligent Multimodal Question Generation 

**Authors**: Ahmed Mubarak, Amna Ahmed, Amira Nasser, Aya Mohamed, Fares El-Sadek, Mohammed Ahmed, Ahmed Salah, Youssef Sobhy  

**Link**: [PDF](https://arxiv.org/pdf/2509.03535)  

**Abstract**: In today's information-rich era, learners have access to abundant educational resources, but the lack of practice materials tailored to these resources presents a significant challenge. This project addresses that gap by developing a multi-modal question generation system that can automatically generate diverse question types from various content formats. The system features four major components: multi-modal input handling, question generation, reinforcement learning from human feedback (RLHF), and an end-to-end interactive interface. This project lays the foundation for automated, scalable, and intelligent question generation, carefully balancing resource efficiency, robust functionality and a smooth user experience. 

---
# AR$^2$: Adversarial Reinforcement Learning for Abstract Reasoning in Large Language Models 

**Authors**: Cheng-Kai Yeh, Hsing-Wang Lee, Chung-Hung Kuo, Hen-Hsen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03537)  

**Abstract**: Abstraction--the ability to recognize and distill essential computational patterns from complex problem statements--is a foundational skill in computer science, critical both for human problem-solvers and coding-oriented large language models (LLMs). Despite recent advances in training LLMs for code generation using reinforcement learning (RL), most existing approaches focus primarily on superficial pattern recognition, overlooking explicit training for abstraction. In this study, we propose AR$^2$ (Adversarial Reinforcement Learning for Abstract Reasoning), a novel framework explicitly designed to enhance the abstraction abilities of LLMs. AR$^2$ employs a teacher model to transform kernel problems into narrative-rich, challenging descriptions without changing their fundamental logic. Simultaneously, a student coding model is trained to solve these complex narrative problems by extracting their underlying computational kernels. Experimental results demonstrate that AR$^2$ substantially improves the student model's accuracy on previously unseen, challenging programming tasks, underscoring abstraction as a key skill for enhancing LLM generalization. 

---
# Synthesizing Sheet Music Problems for Evaluation and Reinforcement Learning 

**Authors**: Zhilin Wang, Zhe Yang, Yun Luo, Yafu Li, Haoran Zhang, Runzhe Zhan, Derek F. Wong, Jizhe Zhou, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.04059)  

**Abstract**: Enhancing the ability of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) to interpret sheet music is a crucial step toward building AI musicians. However, current research lacks both evaluation benchmarks and training data for sheet music reasoning. To address this, we propose the idea of synthesizing sheet music problems grounded in music theory, which can serve both as evaluation benchmarks and as training data for reinforcement learning with verifiable rewards (RLVR). We introduce a data synthesis framework that generates verifiable sheet music questions in both textual and visual modalities, leading to the Synthetic Sheet Music Reasoning Benchmark (SSMR-Bench) and a complementary training set. Evaluation results on SSMR-Bench show the importance of models' reasoning abilities in interpreting sheet music. At the same time, the poor performance of Gemini 2.5-Pro highlights the challenges that MLLMs still face in interpreting sheet music in a visual format. By leveraging synthetic data for RLVR, Qwen3-8B-Base and Qwen2.5-VL-Instruct achieve improvements on the SSMR-Bench. Besides, the trained Qwen3-8B-Base surpasses GPT-4 in overall performance on MusicTheoryBench and achieves reasoning performance comparable to GPT-4 with the strategies of Role play and Chain-of-Thought. Notably, its performance on math problems also improves relative to the original Qwen3-8B-Base. Furthermore, our results show that the enhanced reasoning ability can also facilitate music composition. In conclusion, we are the first to propose the idea of synthesizing sheet music problems based on music theory rules, and demonstrate its effectiveness not only in advancing model reasoning for sheet music understanding but also in unlocking new possibilities for AI-assisted music creation. 

---
# Align-then-Slide: A complete evaluation framework for Ultra-Long Document-Level Machine Translation 

**Authors**: Jiaxin Guo, Daimeng Wei, Yuanchang Luo, Xiaoyu Chen, Zhanglin Wu, Huan Yang, Hengchao Shang, Zongyao Li, Zhiqiang Rao, Jinlong Yang, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03809)  

**Abstract**: Large language models (LLMs) have ushered in a new era for document-level machine translation (\textit{doc}-mt), yet their whole-document outputs challenge existing evaluation methods that assume sentence-by-sentence alignment. We introduce \textit{\textbf{Align-then-Slide}}, a complete evaluation framework for ultra-long doc-mt. In the Align stage, we automatically infer sentence-level source-target correspondences and rebuild the target to match the source sentence number, resolving omissions and many-to-one/one-to-many mappings. In the n-Chunk Sliding Evaluate stage, we calculate averaged metric scores under 1-, 2-, 3- and 4-chunk for multi-granularity assessment. Experiments on the WMT benchmark show a Pearson correlation of 0.929 between our method with expert MQM rankings. On a newly curated real-world test set, our method again aligns closely with human judgments. Furthermore, preference data produced by Align-then-Slide enables effective CPO training and its direct use as a reward model for GRPO, both yielding translations preferred over a vanilla SFT baseline. The results validate our framework as an accurate, robust, and actionable evaluation tool for doc-mt systems. 

---
# Towards a Unified View of Large Language Model Post-Training 

**Authors**: Xingtai Lv, Yuxin Zuo, Youbang Sun, Hongyi Liu, Yuntian Wei, Zhekai Chen, Lixuan He, Xuekai Zhu, Kaiyan Zhang, Bingning Wang, Ning Ding, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.04419)  

**Abstract**: Two major sources of training data exist for post-training modern language models: online (model-generated rollouts) data, and offline (human or other-model demonstrations) data. These two types of data are typically used by approaches like Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT), respectively. In this paper, we show that these approaches are not in contradiction, but are instances of a single optimization process. We derive a Unified Policy Gradient Estimator, and present the calculations of a wide spectrum of post-training approaches as the gradient of a common objective under different data distribution assumptions and various bias-variance tradeoffs. The gradient estimator is constructed with four interchangeable parts: stabilization mask, reference policy denominator, advantage estimate, and likelihood gradient. Motivated by our theoretical findings, we propose Hybrid Post-Training (HPT), an algorithm that dynamically selects different training signals. HPT is designed to yield both effective exploitation of demonstration and stable exploration without sacrificing learned reasoning patterns. We provide extensive experiments and ablation studies to verify the effectiveness of our unified theoretical framework and HPT. Across six mathematical reasoning benchmarks and two out-of-distribution suites, HPT consistently surpasses strong baselines across models of varying scales and families. 

---
# Enhancing Speech Large Language Models through Reinforced Behavior Alignment 

**Authors**: Yansong Liu, Jiateng Li, Yuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03526)  

**Abstract**: The recent advancements of Large Language Models (LLMs) have spurred considerable research interest in extending their linguistic capabilities beyond text to other modalities, which leads to emergence of speech-based LLMs (SpeechLMs) with capability of processing user request in either speech or textual formats. However, owing to inter-modal discrepancies, these SpeechLMs still exhibit a significant performance gap compared to their text-based LLM counterparts in instruction-following, particularly when confronted with the dynamic and variable nature of user speech. To address this challenge, this paper introduces a framework termed Reinforced Behavior Alignment (RBA), designed to bolster the language generation proficiency of SpeechLMs. Instead of relying on supervised fine-tuning from human annotations, RBA employs a self-synthesis methodology to generate extensive, high-fidelity alignment data by a powerful teacher LLM. Then SpeechLMs is aligned its behavior with that of a teacher using a reinforcement learning-based approach. Experimental results demonstrate that this method effectively enhances the instruction-following capabilities of SpeechLMs that outperform conventional distillation baselines. Crucially, we demonstrate that RBA can be seamlessly extended to tasks such including spoken question answering and speech-to-text translation, attaining state-of-the-art performance on open benchmarks with only self-generated data. 

---
# CoT-Space: A Theoretical Framework for Internal Slow-Thinking via Reinforcement Learning 

**Authors**: Zeyu Gan, Hao Yi, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04027)  

**Abstract**: Reinforcement Learning (RL) has become a pivotal approach for enhancing the reasoning capabilities of Large Language Models (LLMs). However, a significant theoretical gap persists, as traditional token-level RL frameworks fail to align with the reasoning-level nature of complex, multi-step thought processes like Chain-of-Thought (CoT). To address this challenge, we introduce CoT-Space, a novel theoretical framework that recasts LLM reasoning from a discrete token-prediction task to an optimization process within a continuous, reasoning-level semantic space. By analyzing this process from both a noise perspective and a risk perspective, we demonstrate that the convergence to an optimal CoT length is a natural consequence of the fundamental trade-off between underfitting and overfitting. Furthermore, extensive experiments provide strong empirical validation for our theoretical findings. Our framework not only provides a coherent explanation for empirical phenomena such as overthinking but also offers a solid theoretical foundation to guide the future development of more effective and generalizable reasoning agents. 

---
