# Less Redundancy: Boosting Practicality of Vision Language Model in Walking Assistants 

**Authors**: Chongyang Li, Yuan Zhiqiang, Jiapei Zhang, Ying Deng, Hanbo Bi, Zexi Jia, Xiaoyue Duan, Peixiang Luo, Jinchao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16070)  

**Abstract**: Approximately 283 million people worldwide live with visual impairments, motivating increasing research into leveraging Visual Language Models (VLMs) to develop effective walking assistance systems for blind and low vision individuals. However, existing VLMs in walking assistant task often have outputs that contain considerable redundancy and extraneous details, adversely affecting users' ability to accurately assess their surroundings. Moreover, these models typically lack the capability to proactively assess environmental risks and adaptively trigger reminders based on the appropriate scene, leading to excessive temporal redundancy. To mitigate output and temporal redundancy, we propose WalkVLM-LR, a walking assistance model with less redundancy. To reduce output redundancy, we introduce four human-preference-based custom reward functions within the GRPO-based reasoning framework to optimize the output in terms of conciseness, fluency, keyword density, and accuracy, thereby producing more informative and streamlined outputs. To minimize temporal redundancy, we incorporate an environment awareness discriminator, which shares the visual encoder with the VLMs to reduce redundant computations and enhance discriminative efficiency, to make WalkVLM-LR assess scene risk levels and minimize unnecessary reminders. Experimental results demonstrate that our method achieves state-of-the-art performance across all evaluation metrics compared with other models, particularly in output conciseness and less temporal redundancy. 

---
# Coarse-to-Fine Personalized LLM Impressions for Streamlined Radiology Reports 

**Authors**: Chengbo Sun, Hui Yi Leong, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.15845)  

**Abstract**: The manual creation of the "Impression" section in radiology reports is a primary driver of radiologist burnout. To address this challenge, we propose a coarse-to-fine framework that leverages open-source large language models (LLMs) to automatically generate and personalize impressions from clinical findings. The system first produces a draft impression and then refines it using machine learning and reinforcement learning from human feedback (RLHF) to align with individual radiologists' styles while ensuring factual accuracy. We fine-tune LLaMA and Mistral models on a large dataset of reports from the University of Chicago Medicine. Our approach is designed to significantly reduce administrative workload and improve reporting efficiency while maintaining high standards of clinical precision. 

---
# ALAS: Autonomous Learning Agent for Self-Updating Language Models 

**Authors**: Dhruv Atreja  

**Link**: [PDF](https://arxiv.org/pdf/2508.15805)  

**Abstract**: Large language models (LLMs) often have a fixed knowledge cutoff, limiting their accuracy on emerging information. We present ALAS (Autonomous Learning Agent System), a modular pipeline that continuously updates an LLM's knowledge with minimal human intervention. ALAS autonomously generates a learning curriculum for a target domain, retrieves up-to-date information from the web (with citations), distills this into question-answer training data, and fine-tunes the model through supervised fine-tuning (SFT) and direct preference optimization (DPO). It iteratively evaluates performance and revises the curriculum, enabling long-term continual learning. We demonstrate ALAS's ability to self-improve a model on rapidly evolving domains (e.g., new Python releases, latest security CVEs, academic trends), significantly boosting post-cutoff question answering accuracy (from 15% to 90% on average) without manual dataset curation. The system emphasizes modularity and reproducibility: each component (planning, retrieval, distillation, memory, fine-tuning) is interchangeable and built on standard APIs. We discuss comparative baselines (e.g., retrieval-augmented generation vs. fine-tuning) and show that ALAS achieves 90% accuracy on knowledge-updated queries with minimal engineering overhead. Finally, we outline limitations (cost, dependency on source quality) and future directions for autonomous lifelong learning in LLMs. 

---
# From Clicks to Preference: A Multi-stage Alignment Framework for Generative Query Suggestion in Conversational System 

**Authors**: Junhao Yin, Haolin Wang, Peng Bao, Ju Xu, Yongliang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15811)  

**Abstract**: Generative query suggestion using large language models offers a powerful way to enhance conversational systems, but aligning outputs with nuanced user preferences remains a critical challenge. To address this, we introduce a multi-stage framework designed for progressive alignment between the generation policy and user intent. Our pipeline begins with prompt engineering as a cold-start strategy, followed by the Supervised Fine-Tuning stage, in which we introduce a distillation method on click logs to create a robust foundational model. To better model user preferences while capturing their inherent uncertainty, we develop a Gaussian Reward Model (GaRM) that represents user preferences as probability distributions rather than point estimates. Finally, we employ reinforcement learning to align the generation policy with these preferences, guided by a composite reward function that integrates GaRM with auxiliary heuristics to mitigate reward hacking. To maintain training stability, this process is enhanced by a novel out-of-distribution regularization method and a two-stage reward fusion technique. Extensive experiments demonstrate that our framework significantly outperforms baselines on both automatic and human evaluations and yields a 34\% relative increase in user engagement as measured by click-through rate in live A/B tests. 

---
# User-Assistant Bias in LLMs 

**Authors**: Xu Pan, Jingxuan Fan, Zidi Xiong, Ely Hahami, Jorin Overwiening, Ziqian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.15815)  

**Abstract**: Large language models (LLMs) can bias towards relying on their own or the user's information in chat history, leading to overly stubborn or agreeable behaviors in multi-turn conversations. In this paper, we formalize this model characteristic as user-assistant bias and introduce an 8k multi-turn conversation dataset $\textbf{UserAssist}$, which we use to benchmark, understand and manipulate the user-assistant bias in frontier LLMs. Leveraging $\textbf{UserAssist-test}$, we first benchmark the user-assistant bias of 26 commercial and 26 open-weight models. Commercial models show various levels of user bias. Evaluation on open-weight models reveals significant user bias in the instruction-tuned models, and weak user bias in reasoning (or reasoning-distilled) models. We then perform controlled fine-tuning experiments to pinpoint the post-training recipe contributing to these bias shifts: human preference alignment increases user bias, while training on chain-of-thought reasoning traces decreases it. Finally, we demonstrate that user-assistant bias can be bidirectionally adjusted by performing direct preference optimization (DPO) on $\textbf{UserAssist-train}$, and generalizes well to both in-domain and out-of-domain conversations. Our results provide insights into how the LLM integrates information from different sources, and also a viable way to detect and control model abnormalities. 

---
# CARFT: Boosting LLM Reasoning via Contrastive Learning with Annotated Chain-of-Thought-based Reinforced Fine-Tuning 

**Authors**: Wenqiao Zhu, Ji Liu, Rongjuncheng Zhang, Haipang Wu, Yulun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15868)  

**Abstract**: Reasoning capability plays a significantly critical role in the the broad applications of Large Language Models (LLMs). To enhance the reasoning performance of LLMs, diverse Reinforcement Learning (RL)-based fine-tuning approaches have been proposed to address the limited generalization capability of LLMs trained solely via Supervised Fine-Tuning (SFT). Despite their effectiveness, two major limitations hinder the advancement of LLMs. First, vanilla RL-based approaches ignore annotated Chain-of-Thought (CoT) and incorporate unstable reasoning path sampling, which typically results in model collapse, unstable training process, and suboptimal performance. Second, existing SFT approaches generally overemphasize the annotated CoT, potentially leading to performance degradation due to insufficient exploitation of potential CoT. In this paper, we propose a Contrastive learning with annotated CoT-based Reinforced Fine-Tuning approach, i.e., \TheName{}, to enhance the reasoning performance of LLMs while addressing the aforementioned limitations. Specifically, we propose learning a representation for each CoT. Based on this representation, we design novel contrastive signals to guide the fine-tuning process. Our approach not only fully exploits the available annotated CoT but also stabilizes the fine-tuning procedure by incorporating an additional unsupervised learning signal. We conduct comprehensive experiments and in-depth analysis with three baseline approaches, two foundation models, and two datasets to demonstrate significant advantages of \TheName{} in terms of robustness, performance (up to 10.15\%), and efficiency (up to 30.62\%). Code is available at this https URL. 

---
# KG-o1: Enhancing Multi-hop Question Answering in Large Language Models via Knowledge Graph Integration 

**Authors**: Nan Wang, Yongqi Fan, yansha zhu, ZongYu Wang, Xuezhi Cao, Xinyan He, Haiyun Jiang, Tong Ruan, Jingping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15790)  

**Abstract**: Large Language Models (LLMs) face challenges in knowledge-intensive reasoning tasks like classic multi-hop question and answering, which involves reasoning across multiple facts. This difficulty arises because the chain of thoughts (CoTs) generated by LLMs in such tasks often deviate from real or a priori reasoning paths. In contrast, knowledge graphs (KGs) explicitly represent the logical connections between facts through entities and relationships. This reflects a significant gap. Meanwhile, large reasoning models (LRMs), such as o1, have demonstrated that long-step reasoning significantly enhances the performance of LLMs. Building on these insights, we propose KG-o1, a four-stage approach that integrates KGs to enhance the multi-hop reasoning abilities of LLMs. We first filter out initial entities and generate complex subgraphs. Secondly, we construct logical paths for subgraphs and then use knowledge graphs to build a dataset with a complex and extended brainstorming process, which trains LLMs to imitate long-term reasoning. Finally, we employ rejection sampling to generate a self-improving corpus for direct preference optimization (DPO), further refining the LLMs reasoning abilities. We conducted experiments on two simple and two complex datasets. The results show that KG-o1 models exhibit superior performance across all tasks compared to existing LRMs. 

---
# Constraints-Guided Diffusion Reasoner for Neuro-Symbolic Learning 

**Authors**: Xuan Zhang, Zhijian Zhou, Weidi Xu, Yanting Miao, Chao Qu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16524)  

**Abstract**: Enabling neural networks to learn complex logical constraints and fulfill symbolic reasoning is a critical challenge. Bridging this gap often requires guiding the neural network's output distribution to move closer to the symbolic constraints. While diffusion models have shown remarkable generative capability across various domains, we employ the powerful architecture to perform neuro-symbolic learning and solve logical puzzles. Our diffusion-based pipeline adopts a two-stage training strategy: the first stage focuses on cultivating basic reasoning abilities, while the second emphasizes systematic learning of logical constraints. To impose hard constraints on neural outputs in the second stage, we formulate the diffusion reasoner as a Markov decision process and innovatively fine-tune it with an improved proximal policy optimization algorithm. We utilize a rule-based reward signal derived from the logical consistency of neural outputs and adopt a flexible strategy to optimize the diffusion reasoner's policy. We evaluate our methodology on some classical symbolic reasoning benchmarks, including Sudoku, Maze, pathfinding and preference learning. Experimental results demonstrate that our approach achieves outstanding accuracy and logical consistency among neural networks. 

---
# FLAMES: Improving LLM Math Reasoning via a Fine-Grained Analysis of the Data Synthesis Pipeline 

**Authors**: Parker Seegmiller, Kartik Mehta, Soumya Saha, Chenyang Tao, Shereen Oraby, Arpit Gupta, Tagyoung Chung, Mohit Bansal, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.16514)  

**Abstract**: Recent works improving LLM math reasoning with synthetic data have used unique setups, making comparison of data synthesis strategies impractical. This leaves many unanswered questions about the roles of different factors in the synthetic data pipeline, such as the impact of filtering low-quality problems. To address this gap, we introduce FLAMES, a Framework for LLM Assessment of Math rEasoning Data Synthesis, and perform a systematic study of 10 existing data synthesis strategies and multiple other factors impacting the performance of synthetic math reasoning data. Our FLAMES experiments provide several valuable insights about the optimal balance of difficulty and diversity of synthetic data. First, data agents designed to increase problem complexity lead to best improvements on most math metrics. Second, with a fixed data generation budget, keeping higher problem coverage is more important than keeping only problems with reliable solutions. Third, GSM8K- and MATH-based synthetic data can lead to improvements on competition-level benchmarks, showcasing easy-to-hard generalization. Leveraging insights from our FLAMES experiments, we design two novel data synthesis strategies for improving out-of-domain generalization and robustness. Further, we develop the FLAMES dataset, an effective blend of our novel and existing data synthesis strategies, outperforming public datasets on OlympiadBench (+15.7), CollegeMath (+4.5), GSMPlus (+6.5), and MATH (+3.1). Fine-tuning Qwen2.5-Math-7B on the FLAMES dataset achieves 81.4% on MATH, surpassing larger Llama3 405B, GPT-4o and Claude 3.5 Sonnet. 

---
# OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval 

**Authors**: Yu Liu, Yanbing Liu, Fangfang Yuan, Cong Cao, Youbang Sun, Kun Peng, WeiZhuo Chen, Jianjun Li, Zhiyuan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.16438)  

**Abstract**: Recent advances in large language models (LLMs) and dense retrievers have driven significant progress in retrieval-augmented generation (RAG). However, existing approaches face significant challenges in complex reasoning-oriented multi-hop retrieval tasks: 1) Ineffective reasoning-oriented planning: Prior methods struggle to generate robust multi-step plans for complex queries, as rule-based decomposers perform poorly on out-of-template questions. 2) Suboptimal reasoning-driven retrieval: Related methods employ limited query reformulation, leading to iterative retrieval loops that often fail to locate golden documents. 3) Insufficient reasoning-guided filtering: Prevailing methods lack the fine-grained reasoning to effectively filter salient information from noisy results, hindering utilization of retrieved knowledge. Fundamentally, these limitations all stem from the weak coupling between retrieval and reasoning in current RAG architectures. We introduce the Orchestrated Planner-Executor Reasoning Architecture (OPERA), a novel reasoning-driven retrieval framework. OPERA's Goal Planning Module (GPM) decomposes questions into sub-goals, which are executed by a Reason-Execute Module (REM) with specialized components for precise reasoning and effective retrieval. To train OPERA, we propose Multi-Agents Progressive Group Relative Policy Optimization (MAPGRPO), a novel variant of GRPO. Experiments on complex multi-hop benchmarks show OPERA's superior performance, validating both the MAPGRPO method and OPERA's design. Code is available at this https URL. 

---
# FlexMUSE: Multimodal Unification and Semantics Enhancement Framework with Flexible interaction for Creative Writing 

**Authors**: Jiahao Chen, Zhiyong Ma, Wenbiao Du, Qingyuan Chuai  

**Link**: [PDF](https://arxiv.org/pdf/2508.16230)  

**Abstract**: Multi-modal creative writing (MMCW) aims to produce illustrated articles. Unlike common multi-modal generative (MMG) tasks such as storytelling or caption generation, MMCW is an entirely new and more abstract challenge where textual and visual contexts are not strictly related to each other. Existing methods for related tasks can be forcibly migrated to this track, but they require specific modality inputs or costly training, and often suffer from semantic inconsistencies between modalities. Therefore, the main challenge lies in economically performing MMCW with flexible interactive patterns, where the semantics between the modalities of the output are more aligned. In this work, we propose FlexMUSE with a T2I module to enable optional visual input. FlexMUSE promotes creativity and emphasizes the unification between modalities by proposing the modality semantic alignment gating (msaGate) to restrict the textual input. Besides, an attention-based cross-modality fusion is proposed to augment the input features for semantic enhancement. The modality semantic creative direct preference optimization (mscDPO) within FlexMUSE is designed by extending the rejected samples to facilitate the writing creativity. Moreover, to advance the MMCW, we expose a dataset called ArtMUSE which contains with around 3k calibrated text-image pairs. FlexMUSE achieves promising results, demonstrating its consistency, creativity and coherence. 

---
