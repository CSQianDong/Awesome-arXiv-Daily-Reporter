# Instella: Fully Open Language Models with Stellar Performance 

**Authors**: Jiang Liu, Jialian Wu, Xiaodong Yu, Yusheng Su, Prakamya Mishra, Gowtham Ramesh, Sudhanshu Ranjan, Chaitanya Manem, Ximeng Sun, Ze Wang, Pratik Prabhanjan Brahma, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2511.10628)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance across a wide range of tasks, yet the majority of high-performing models remain closed-source or partially open, limiting transparency and reproducibility. In this work, we introduce Instella, a family of fully open three billion parameter language models trained entirely on openly available data and codebase. Powered by AMD Instinct MI300X GPUs, Instella is developed through large-scale pre-training, general-purpose instruction tuning, and alignment with human preferences. Despite using substantially fewer pre-training tokens than many contemporaries, Instella achieves state-of-the-art results among fully open models and is competitive with leading open-weight models of comparable size. We further release two specialized variants: Instella-Long, capable of handling context lengths up to 128K tokens, and Instella-Math, a reasoning-focused model enhanced through supervised fine-tuning and reinforcement learning on mathematical tasks. Together, these contributions establish Instella as a transparent, performant, and versatile alternative for the community, advancing the goal of open and reproducible language modeling research. 

---
# Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following 

**Authors**: Yun He, Wenzhe Li, Hejia Zhang, Songlin Li, Karishma Mandyam, Sopan Khosla, Yuanhao Xiong, Nanshu Wang, Selina Peng, Beibin Li, Shengjie Bi, Shishir G. Patil, Qi Qi, Shengyu Feng, Julian Katz-Samuels, Richard Yuanzhe Pang, Sujan Gonugondla, Hunter Lang, Yue Yu, Yundi Qian, Maryam Fazel-Zarandi, Licheng Yu, Amine Benhalloum, Hany Awadalla, Manaal Faruqui  

**Link**: [PDF](https://arxiv.org/pdf/2511.10507)  

**Abstract**: Recent progress in large language models (LLMs) has led to impressive performance on a range of tasks, yet advanced instruction following (IF)-especially for complex, multi-turn, and system-prompted instructions-remains a significant challenge. Rigorous evaluation and effective training for such capabilities are hindered by the lack of high-quality, human-annotated benchmarks and reliable, interpretable reward signals. In this work, we introduce AdvancedIF (we will release this benchmark soon), a comprehensive benchmark featuring over 1,600 prompts and expert-curated rubrics that assess LLMs ability to follow complex, multi-turn, and system-level instructions. We further propose RIFL (Rubric-based Instruction-Following Learning), a novel post-training pipeline that leverages rubric generation, a finetuned rubric verifier, and reward shaping to enable effective reinforcement learning for instruction following. Extensive experiments demonstrate that RIFL substantially improves the instruction-following abilities of LLMs, achieving a 6.7% absolute gain on AdvancedIF and strong results on public benchmarks. Our ablation studies confirm the effectiveness of each component in RIFL. This work establishes rubrics as a powerful tool for both training and evaluating advanced IF in LLMs, paving the way for more capable and reliable AI systems. 

---
# Reasoning About Intent for Ambiguous Requests 

**Authors**: Irina Saparina, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2511.10453)  

**Abstract**: Large language models often respond to ambiguous requests by implicitly committing to one interpretation. Intent misunderstandings can frustrate users and create safety risks. To address this, we propose generating multiple interpretation-answer pairs in a single structured response to ambiguous requests. Our models are trained with reinforcement learning and customized reward functions using multiple valid answers as supervision. Experiments on conversational question answering and semantic parsing demonstrate that our method achieves higher coverage of valid answers than baseline approaches. Human evaluation confirms that predicted interpretations are highly aligned with their answers. Our approach promotes transparency with explicit interpretations, achieves efficiency by requiring only one generation step, and supports downstream applications through its structured output format. 

---
# Rectify Evaluation Preference: Improving LLMs' Critique on Math Reasoning via Perplexity-aware Reinforcement Learning 

**Authors**: Changyuan Tian, Zhicong Lu, Shuang Qian, Nayu Liu, Peiguang Li, Li Jin, Leiyi Hu, Zhizhao Zeng, Sirui Wang, Ke Zeng, Zhi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.10303)  

**Abstract**: To improve Multi-step Mathematical Reasoning (MsMR) of Large Language Models (LLMs), it is crucial to obtain scalable supervision from the corpus by automatically critiquing mistakes in the reasoning process of MsMR and rendering a final verdict of the problem-solution. Most existing methods rely on crafting high-quality supervised fine-tuning demonstrations for critiquing capability enhancement and pay little attention to delving into the underlying reason for the poor critiquing performance of LLMs. In this paper, we orthogonally quantify and investigate the potential reason -- imbalanced evaluation preference, and conduct a statistical preference analysis. Motivated by the analysis of the reason, a novel perplexity-aware reinforcement learning algorithm is proposed to rectify the evaluation preference, elevating the critiquing capability. Specifically, to probe into LLMs' critiquing characteristics, a One-to-many Problem-Solution (OPS) benchmark is meticulously constructed to quantify the behavior difference of LLMs when evaluating the problem solutions generated by itself and others. Then, to investigate the behavior difference in depth, we conduct a statistical preference analysis oriented on perplexity and find an intriguing phenomenon -- ``LLMs incline to judge solutions with lower perplexity as correct'', which is dubbed as \textit{imbalanced evaluation preference}. To rectify this preference, we regard perplexity as the baton in the algorithm of Group Relative Policy Optimization, supporting the LLMs to explore trajectories that judge lower perplexity as wrong and higher perplexity as correct. Extensive experimental results on our built OPS and existing available critic benchmarks demonstrate the validity of our method. 

---
# HierRouter: Coordinated Routing of Specialized Large Language Models via Reinforcement Learning 

**Authors**: Nikunj Gupta, Bill Guo, Rajgopal Kannan, Viktor K. Prasanna  

**Link**: [PDF](https://arxiv.org/pdf/2511.09873)  

**Abstract**: Large Language Models (LLMs) deliver state-of-the-art performance across many tasks but impose high computational and memory costs, limiting their deployment in resource-constrained or real-time settings. To address this, we propose HierRouter, a hierarchical routing approach that dynamically assembles inference pipelines from a pool of specialized, lightweight language models. Formulated as a finite-horizon Markov Decision Process (MDP), our approach trains a Proximal Policy Optimization (PPO)-based reinforcement learning agent to iteratively select which models to invoke at each stage of multi-hop inference. The agent conditions on the evolving context and accumulated cost to make context-aware routing decisions. Experiments with three open-source candidate LLMs across six benchmarks, including QA, code generation, and mathematical reasoning, show that HierRouter improves response quality by up to 2.4x compared to using individual models independently, while incurring only a minimal additional inference cost on average. These results highlight the promise of hierarchical routing for cost-efficient, high-performance LLM inference. All codes can be found here this https URL Nikunj-Gupta/hierouter. 

---
# In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback 

**Authors**: Mingye Zhu, Yi Liu, Zheren Fu, Quan Wang, Yongdong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09865)  

**Abstract**: Training Large Language Models (LLMs) for chain-of-thought reasoning presents a significant challenge: supervised fine-tuning on a single "golden" rationale hurts generalization as it penalizes equally valid alternatives, whereas reinforcement learning with verifiable rewards struggles with credit assignment and prohibitive computational cost. To tackle these limitations, we introduce InTRO (In-Token Rationality Optimization), a new framework that enables both token-level exploration and self-feedback for accurate and concise reasoning. Instead of directly optimizing an intractable objective over all valid reasoning paths, InTRO leverages correction factors-token-wise importance weights estimated by the information discrepancy between the generative policy and its answer-conditioned counterpart, for informative next token selection. This approach allows the model to perform token-level exploration and receive self-generated feedback within a single forward pass, ultimately encouraging accurate and concise rationales. Across six math-reasoning benchmarks, InTRO consistently outperforms other baselines, raising solution accuracy by up to 20% relative to the base model. Its chains of thought are also notably more concise, exhibiting reduced verbosity. Beyond this, InTRO enables cross-domain transfer, successfully adapting to out-of-domain reasoning tasks that extend beyond the realm of mathematics, demonstrating robust generalization. 

---
# Towards Emotionally Intelligent and Responsible Reinforcement Learning 

**Authors**: Garapati Keerthana, Manik Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.10573)  

**Abstract**: Personalized decision systems in healthcare and behavioral support often rely on static rule-based or engagement-maximizing heuristics that overlook users' emotional context and ethical constraints. Such approaches risk recommending insensitive or unsafe interventions, especially in domains involving serious mental illness, substance use disorders, or depression. To address this limitation, we propose a Responsible Reinforcement Learning (RRL) framework that integrates emotional and contextual understanding with ethical considerations into the sequential decision-making process. RRL formulates personalization as a Constrained Markov Decision Process (CMDP), where the agent optimizes engagement and adherence while ensuring emotional alignment and ethical safety. We introduce a multi-objective reward function that explicitly balances short-term behavioral engagement with long-term user well-being, and define an emotion-informed state representation that captures fluctuations in emotional readiness, affect, and risk. The proposed architecture can be instantiated with any RL algorithm (e.g., DQN, PPO) augmented with safety constraints or Lagrangian regularization. Conceptually, this framework operationalizes empathy and responsibility within machine learning policy optimization, bridging safe RL, affective computing and responsible AI. We discuss the implications of this approach for human-centric domains such as behavioral health, education, and digital therapeutics, and outline simulation-based validation paths for future empirical work. This paper aims to initiate a methodological conversation about ethically aligned reinforcement learning for emotionally aware and trustworthy personalization systems. 

---
# Music Flamingo: Scaling Music Understanding in Audio Language Models 

**Authors**: Sreyan Ghosh, Arushi Goel, Lasha Koroshinadze, Sang-gil Lee, Zhifeng Kong, Joao Felipe Santos, Ramani Duraiswami, Dinesh Manocha, Wei Ping, Mohammad Shoeybi, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2511.10289)  

**Abstract**: We introduce Music Flamingo, a novel large audio-language model designed to advance music (including song) understanding in foundational audio models. While audio-language research has progressed rapidly, music remains challenging due to its dynamic, layered, and information-dense nature. Progress has been further limited by the difficulty of scaling open audio understanding models, primarily because of the scarcity of high-quality music data and annotations. As a result, prior models are restricted to producing short, high-level captions, answering only surface-level questions, and showing limited generalization across diverse musical cultures. To address these challenges, we curate MF-Skills, a large-scale dataset labeled through a multi-stage pipeline that yields rich captions and question-answer pairs covering harmony, structure, timbre, lyrics, and cultural context. We fine-tune an enhanced Audio Flamingo 3 backbone on MF-Skills and further strengthen multiple skills relevant to music understanding. To improve the model's reasoning abilities, we introduce a post-training recipe: we first cold-start with MF-Think, a novel chain-of-thought dataset grounded in music theory, followed by GRPO-based reinforcement learning with custom rewards. Music Flamingo achieves state-of-the-art results across 10+ benchmarks for music understanding and reasoning, establishing itself as a generalist and musically intelligent audio-language model. Beyond strong empirical results, Music Flamingo sets a new standard for advanced music understanding by demonstrating how models can move from surface-level recognition toward layered, human-like perception of songs. We believe this work provides both a benchmark and a foundation for the community to build the next generation of models that engage with music as meaningfully as humans do. 

---
# Learning to Pose Problems: Reasoning-Driven and Solver-Adaptive Data Synthesis for Large Reasoning Models 

**Authors**: Yongxian Wei, Yilin Zhao, Li Shen, Xinrui Chen, Runxi Cheng, Sinan Du, Hao Yu, Gang Liu, Jiahong Yan, Chun Yuan, Dian Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.09907)  

**Abstract**: Data synthesis for training large reasoning models offers a scalable alternative to limited, human-curated datasets, enabling the creation of high-quality data. However, existing approaches face several challenges: (i) indiscriminate generation that ignores the solver's ability and yields low-value problems, or reliance on complex data pipelines to balance problem difficulty; and (ii) a lack of reasoning in problem generation, leading to shallow problem variants. In this paper, we develop a problem generator that reasons explicitly to plan problem directions before synthesis and adapts difficulty to the solver's ability. Specifically, we construct related problem pairs and augment them with intermediate problem-design CoT produced by a reasoning model. These data bootstrap problem-design strategies from the generator. Then, we treat the solver's feedback on synthetic problems as a reward signal, enabling the generator to calibrate difficulty and produce complementary problems near the edge of the solver's competence. Extensive experiments on 10 mathematical and general reasoning benchmarks show that our method achieves an average improvement of 2.5% and generalizes to both language and vision-language models. Moreover, a solver trained on the synthesized data provides improved rewards for continued generator training, enabling co-evolution and yielding a further 0.7% performance gain. Our code will be made publicly available here. 

---
# Radiology Workflow-Guided Hierarchical Reinforcement Fine-Tuning for Medical Report Generation 

**Authors**: Bodong Du, Honglong Yang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.10065)  

**Abstract**: Radiologists compose diagnostic reports through a structured workflow: they describe visual findings, summarize them into impressions, and carefully refine statements in clinically critical cases. However, most existing medical report generation (MRG) systems treat reports as flat sequences, overlooking this hierarchical organization and leading to inconsistencies between descriptive and diagnostic content. To align model behavior with real-world reporting practices, we propose RadFlow, a hierarchical workflow-guided reinforcement optimization framework that explicitly models the structured nature of clinical reporting. RadFlow introduces a clinically grounded reward hierarchy that mirrors the organization of radiological reports. At the global level, the reward integrates linguistic fluency, medical-domain correctness, and cross-sectional consistency between Finding and Impression, promoting coherent and clinically faithful narratives. At the local level, a section-specific reward emphasizes Impression quality, reflecting its central role in diagnostic accuracy. Furthermore, a critical-aware policy optimization mechanism adaptively regularizes learning for high-risk or clinically sensitive cases, emulating the cautious refinement behavior of radiologists when documenting critical findings. Together, these components translate the structured reporting paradigm into the reinforcement fine-tuning process, enabling the model to generate reports that are both linguistically consistent and clinically aligned. Experiments on chest X-ray and carotid ultrasound datasets demonstrate that RadFlow consistently improves diagnostic coherence and overall report quality compared with state-of-the-art baselines. 

---
# AgentEvolver: Towards Efficient Self-Evolving Agent System 

**Authors**: Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou, Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji Deng, Zouying Cao, Zhaoyang Liu, Bolin Ding, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.10395)  

**Abstract**: Autonomous agents powered by large language models (LLMs) have the potential to significantly enhance human productivity by reasoning, using tools, and executing complex tasks in diverse environments. However, current approaches to developing such agents remain costly and inefficient, as they typically require manually constructed task datasets and reinforcement learning (RL) pipelines with extensive random exploration. These limitations lead to prohibitively high data-construction costs, low exploration efficiency, and poor sample utilization. To address these challenges, we present AgentEvolver, a self-evolving agent system that leverages the semantic understanding and reasoning capabilities of LLMs to drive autonomous agent learning. AgentEvolver introduces three synergistic mechanisms: (i) self-questioning, which enables curiosity-driven task generation in novel environments, reducing dependence on handcrafted datasets; (ii) self-navigating, which improves exploration efficiency through experience reuse and hybrid policy guidance; and (iii) self-attributing, which enhances sample efficiency by assigning differentiated rewards to trajectory states and actions based on their contribution. By integrating these mechanisms into a unified framework, AgentEvolver enables scalable, cost-effective, and continual improvement of agent capabilities. Preliminary experiments indicate that AgentEvolver achieves more efficient exploration, better sample utilization, and faster adaptation compared to traditional RL-based baselines. 

---
