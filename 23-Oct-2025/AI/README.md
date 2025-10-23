# Benchmarking World-Model Learning 

**Authors**: Archana Warrier, Dat Nyugen, Michelangelo Naim, Moksh Jain, Yichao Liang, Karen Schroeder, Cambridge Yang, Joshua B. Tenenbaum, Sebastian Vollmer, Kevin Ellis, Zenna Tavares  

**Link**: [PDF](https://arxiv.org/pdf/2510.19788)  

**Abstract**: Model-learning agents should gather information to learn world models that support many downstream tasks and inferences, such as predicting unobserved states, estimating near- and far-term consequences of actions, planning action sequences, and detecting changes in dynamics. Current methods for learning and evaluating world models diverge from this goal: training and evaluation are anchored to next-frame prediction, and success is scored by reward maximization in the same environment. We propose WorldTest, a protocol to evaluate model-learning agents that separates reward-free interaction from a scored test phase in a different but related environment. WorldTest is open-ended$\unicode{x2014}$models should support many different tasks unknown ahead of time$\unicode{x2014}$and agnostic to model representation, allowing comparison across approaches. We instantiated WorldTest with AutumnBench, a suite of 43 interactive grid-world environments and 129 tasks across three families: masked-frame prediction, planning, and predicting changes to the causal dynamics. We compared 517 human participants and three frontier models on AutumnBench. We found that humans outperform the models, and scaling compute improves performance only in some environments but not others. WorldTest provides a novel template$\unicode{x2014}$reward-free exploration, derived tests, and behavior-based scoring$\unicode{x2014}$to evaluate what agents learn about environment dynamics, and AutumnBench exposes significant headroom in world-model learning. 

---
# Beyond Reactivity: Measuring Proactive Problem Solving in LLM Agents 

**Authors**: Gil Pasternak, Dheeraj Rajagopal, Julia White, Dhruv Atreja, Matthew Thomas, George Hurn-Maloney, Ash Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2510.19771)  

**Abstract**: LLM-based agents are increasingly moving towards proactivity: rather than awaiting instruction, they exercise agency to anticipate user needs and solve them autonomously. However, evaluating proactivity is challenging; current benchmarks are constrained to localized context, limiting their ability to test reasoning across sources and longer time horizons. To address this gap, we present PROBE (Proactive Resolution Of BottlEnecks). PROBE decomposes proactivity as a pipeline of three core capabilities: (1) searching for unspecified issues, (2) identifying specific bottlenecks, and (3) executing appropriate resolutions. We apply PROBE to evaluate leading LLMs and popular agentic frameworks, showing that even state-of-the-art models struggle to solve this benchmark. Computing our consistent measurements across frontier LLMs and agents, we find that the best end-to-end performance of 40% is achieved by both GPT-5 and Claude Opus-4.1. Additionally, we demonstrate the relative capabilities of each model and analyze mutual failure modes. Our results highlight the current limitations of autonomous action in agentic systems, and expose promising future research directions. 

---
# Misalignment Bounty: Crowdsourcing AI Agent Misbehavior 

**Authors**: Rustem Turtayev, Natalia Fedorova, Oleg Serikov, Sergey Koldyba, Lev Avagyan, Dmitrii Volkov  

**Link**: [PDF](https://arxiv.org/pdf/2510.19738)  

**Abstract**: Advanced AI systems sometimes act in ways that differ from human intent. To gather clear, reproducible examples, we ran the Misalignment Bounty: a crowdsourced project that collected cases of agents pursuing unintended or unsafe goals. The bounty received 295 submissions, of which nine were awarded.
This report explains the program's motivation and evaluation criteria, and walks through the nine winning submissions step by step. 

---
# Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning 

**Authors**: Gunshi Gupta, Karmesh Yadav, Zsolt Kira, Yarin Gal, Rahaf Aljundi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19732)  

**Abstract**: To enable embodied agents to operate effectively over extended timeframes, it is crucial to develop models that form and access memories to stay contextualized in their environment. In the current paradigm of training transformer-based policies for embodied sequential decision-making tasks, visual inputs often overwhelm the context limits of transformers, while humans can maintain and utilize a lifetime of experience compressed as memories. Significant compression is possible in principle, as much of the input is irrelevant and can be abstracted. However, existing approaches predominantly focus on either recurrent models with fixed-size memory or transformers with full-context reliance. In this work, we propose Memo, a transformer-based architecture and training recipe for reinforcement learning (RL) on memory-intensive, long-horizon tasks. Memo incorporates the creation and retrieval of memory by interleaving periodic summarization tokens with the inputs of a model during training. We demonstrate Memo's effectiveness on a gridworld meta-RL benchmark and a multi-object navigation task in photo-realistic indoor settings. Memo outperforms naive long-context transformer baselines while being more compute and storage efficient. Additionally, Memo generalizes better to longer contexts at inference time and remains robust in streaming settings, where historical context must be truncated to fit inference constraints. 

---
# RLIE: Rule Generation with Logistic Regression, Iterative Refinement, and Evaluation for Large Language Models 

**Authors**: Yang Yang, Hua XU, Zhangyi Hu, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2510.19698)  

**Abstract**: Large Language Models (LLMs) can propose rules in natural language, sidestepping the need for a predefined predicate space in traditional rule learning. Yet many LLM-based approaches ignore interactions among rules, and the opportunity to couple LLMs with probabilistic rule learning for robust inference remains underexplored. We present RLIE, a unified framework that integrates LLMs with probabilistic modeling to learn a set of weighted rules. RLIE has four stages: (1) Rule generation, where an LLM proposes and filters candidates; (2) Logistic regression, which learns probabilistic weights for global selection and calibration; (3) Iterative refinement, which updates the rule set using prediction errors; and (4) Evaluation, which compares the weighted rule set as a direct classifier with methods that inject rules into an LLM. We evaluate multiple inference strategies on real-world datasets. Applying rules directly with their learned weights yields superior performance, whereas prompting LLMs with the rules, weights, and logistic-model outputs surprisingly degrades accuracy. This supports the view that LLMs excel at semantic generation and interpretation but are less reliable for precise probabilistic integration. RLIE clarifies the potential and limitations of LLMs for inductive reasoning and couples them with classic probabilistic rule combination methods to enable more reliable neuro-symbolic reasoning. 

---
# Explainable e-sports win prediction through Machine Learning classification in streaming 

**Authors**: Silvia García-Méndez, Francisco de Arriba-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2510.19671)  

**Abstract**: The increasing number of spectators and players in e-sports, along with the development of optimized communication solutions and cloud computing technology, has motivated the constant growth of the online game industry. Even though Artificial Intelligence-based solutions for e-sports analytics are traditionally defined as extracting meaningful patterns from related data and visualizing them to enhance decision-making, most of the effort in professional winning prediction has been focused on the classification aspect from a batch perspective, also leaving aside the visualization techniques. Consequently, this work contributes to an explainable win prediction classification solution in streaming in which input data is controlled over several sliding windows to reflect relevant game changes. Experimental results attained an accuracy higher than 90 %, surpassing the performance of competing solutions in the literature. Ultimately, our system can be leveraged by ranking and recommender systems for informed decision-making, thanks to the explainability module, which fosters trust in the outcome predictions. 

---
# A Graph Engine for Guitar Chord-Tone Soloing Education 

**Authors**: Matthew Keating, Michael Casey  

**Link**: [PDF](https://arxiv.org/pdf/2510.19666)  

**Abstract**: We present a graph-based engine for computing chord tone soloing suggestions for guitar students. Chord tone soloing is a fundamental practice for improvising over a chord progression, where the instrumentalist uses only the notes contained in the current chord. This practice is a building block for all advanced jazz guitar theory but is difficult to learn and practice. First, we discuss methods for generating chord-tone arpeggios. Next, we construct a weighted graph where each node represents a chord tone arpeggio for a chord in the progression. Then, we calculate the edge weight between each consecutive chord's nodes in terms of optimal transition tones. We then find the shortest path through this graph and reconstruct a chord-tone soloing line. Finally, we discuss a user-friendly system to handle input and output to this engine for guitar students to practice chord tone soloing. 

---
# AgentSense: LLMs Empower Generalizable and Explainable Web-Based Participatory Urban Sensing 

**Authors**: Xusen Guo, Mingxing Peng, Xixuan Hao, Xingchen Zou, Qiongyan Wang, Sijie Ruan, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19661)  

**Abstract**: Web-based participatory urban sensing has emerged as a vital approach for modern urban management by leveraging mobile individuals as distributed sensors. However, existing urban sensing systems struggle with limited generalization across diverse urban scenarios and poor interpretability in decision-making. In this work, we introduce AgentSense, a hybrid, training-free framework that integrates large language models (LLMs) into participatory urban sensing through a multi-agent evolution system. AgentSense initially employs classical planner to generate baseline solutions and then iteratively refines them to adapt sensing task assignments to dynamic urban conditions and heterogeneous worker preferences, while producing natural language explanations that enhance transparency and trust. Extensive experiments across two large-scale mobility datasets and seven types of dynamic disturbances demonstrate that AgentSense offers distinct advantages in adaptivity and explainability over traditional methods. Furthermore, compared to single-agent LLM baselines, our approach outperforms in both performance and robustness, while delivering more reasonable and transparent explanations. These results position AgentSense as a significant advancement towards deploying adaptive and explainable urban sensing systems on the web. 

---
# HSCodeComp: A Realistic and Expert-level Benchmark for Deep Search Agents in Hierarchical Rule Application 

**Authors**: Yiqian Yang, Tian Lan, Qianghuai Jia, Li Zhu, Hui Jiang, Hang Zhu, Longyue Wang, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19631)  

**Abstract**: Effective deep search agents must not only access open-domain and domain-specific knowledge but also apply complex rules-such as legal clauses, medical manuals and tariff rules. These rules often feature vague boundaries and implicit logic relationships, making precise application challenging for agents. However, this critical capability is largely overlooked by current agent benchmarks.
To fill this gap, we introduce HSCodeComp, the first realistic, expert-level e-commerce benchmark designed to evaluate deep search agents in hierarchical rule application. In this task, the deep reasoning process of agents is guided by these rules to predict 10-digit Harmonized System Code (HSCode) of products with noisy but realistic descriptions. These codes, established by the World Customs Organization, are vital for global supply chain efficiency. Built from real-world data collected from large-scale e-commerce platforms, our proposed HSCodeComp comprises 632 product entries spanning diverse product categories, with these HSCodes annotated by several human experts.
Extensive experimental results on several state-of-the-art LLMs, open-source, and closed-source agents reveal a huge performance gap: best agent achieves only 46.8% 10-digit accuracy, far below human experts at 95.0%. Besides, detailed analysis demonstrates the challenges of hierarchical rule application, and test-time scaling fails to improve performance further. 

---
# DAIL: Beyond Task Ambiguity for Language-Conditioned Reinforcement Learning 

**Authors**: Runpeng Xie, Quanwei Wang, Hao Hu, Zherui Zhou, Ni Mu, Xiyun Li, Yiqin Yang, Shuang Xu, Qianchuan Zhao, Bo XU  

**Link**: [PDF](https://arxiv.org/pdf/2510.19562)  

**Abstract**: Comprehending natural language and following human instructions are critical capabilities for intelligent agents. However, the flexibility of linguistic instructions induces substantial ambiguity across language-conditioned tasks, severely degrading algorithmic performance. To address these limitations, we present a novel method named DAIL (Distributional Aligned Learning), featuring two key components: distributional policy and semantic alignment. Specifically, we provide theoretical results that the value distribution estimation mechanism enhances task differentiability. Meanwhile, the semantic alignment module captures the correspondence between trajectories and linguistic instructions. Extensive experimental results on both structured and visual observation benchmarks demonstrate that DAIL effectively resolves instruction ambiguities, achieving superior performance to baseline methods. Our implementation is available at this https URL. 

---
# NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning 

**Authors**: Wonje Choi, Jooyoung Kim, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2510.19429)  

**Abstract**: We address the challenge of adopting language models (LMs) for embodied tasks in dynamic environments, where online access to large-scale inference engines or symbolic planners is constrained due to latency, connectivity, and resource limitations. To this end, we present NeSyPr, a novel embodied reasoning framework that compiles knowledge via neurosymbolic proceduralization, thereby equipping LM-based agents with structured, adaptive, and timely reasoning capabilities. In NeSyPr, task-specific plans are first explicitly generated by a symbolic tool leveraging its declarative knowledge. These plans are then transformed into composable procedural representations that encode the plans' implicit production rules, enabling the resulting composed procedures to be seamlessly integrated into the LM's inference process. This neurosymbolic proceduralization abstracts and generalizes multi-step symbolic structured path-finding and reasoning into single-step LM inference, akin to human knowledge compilation. It supports efficient test-time inference without relying on external symbolic guidance, making it well suited for deployment in latency-sensitive and resource-constrained physical systems. We evaluate NeSyPr on the embodied benchmarks PDDLGym, VirtualHome, and ALFWorld, demonstrating its efficient reasoning capabilities over large-scale reasoning models and a symbolic planner, while using more compact LMs. 

---
# MSC-Bench: A Rigorous Benchmark for Multi-Server Tool Orchestration 

**Authors**: Jia-Kai Dong, I-Wei Huang, Chun-Tin Wu, Yi-Tien Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2510.19423)  

**Abstract**: We introduce MSC-Bench, a large-scale benchmark for evaluating multi-hop, end-to-end tool orchestration by LLM agents in a hierarchical Model-Context Protocol (MCP) ecosystem. Existing benchmarks often evaluate tools in isolation, ignoring challenges such as functional overlap and cross-server orchestration, leading to overly optimistic assessments. MSC-Bench addresses these gaps by constructing ground truth through 'equal function sets', allowing objective metrics such as F1 score and reducing the dependency on LLM-as-a-judge evaluation. Organized as a five-level curriculum, it systematically tests agent capabilities from single-tool orchestration to complex cross-server planning, and robustness to out-of-scope requests. Experiments reveal that rigid hierarchies can hinder performance without co-designed strategies, and even state-of-the-art agents exhibit systemic weaknesses in robustness. MSC-Bench provides a diagnostic framework to expose these limitations and guide the development of more capable and efficient tool-using agents. The benchmark and resources are publicly available at this https URL. 

---
# Continual Knowledge Adaptation for Reinforcement Learning 

**Authors**: Jinwu Hu, Zihao Lian, Zhiquan Wen, Chenghao Li, Guohao Chen, Xutao Wen, Bin Xiao, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.19314)  

**Abstract**: Reinforcement Learning enables agents to learn optimal behaviors through interactions with environments. However, real-world environments are typically non-stationary, requiring agents to continuously adapt to new tasks and changing conditions. Although Continual Reinforcement Learning facilitates learning across multiple tasks, existing methods often suffer from catastrophic forgetting and inefficient knowledge utilization. To address these challenges, we propose Continual Knowledge Adaptation for Reinforcement Learning (CKA-RL), which enables the accumulation and effective utilization of historical knowledge. Specifically, we introduce a Continual Knowledge Adaptation strategy, which involves maintaining a task-specific knowledge vector pool and dynamically using historical knowledge to adapt the agent to new tasks. This process mitigates catastrophic forgetting and enables efficient knowledge transfer across tasks by preserving and adapting critical model parameters. Additionally, we propose an Adaptive Knowledge Merging mechanism that combines similar knowledge vectors to address scalability challenges, reducing memory requirements while ensuring the retention of essential knowledge. Experiments on three benchmarks demonstrate that the proposed CKA-RL outperforms state-of-the-art methods, achieving an improvement of 4.20% in overall performance and 8.02% in forward transfer. The source code is available at this https URL. 

---
# Learning to Make Friends: Coaching LLM Agents toward Emergent Social Ties 

**Authors**: Philipp J. Schneider, Lin Tian, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19299)  

**Abstract**: Can large language model (LLM) agents reproduce the complex social dynamics that characterize human online behavior -- shaped by homophily, reciprocity, and social validation -- and what memory and learning mechanisms enable such dynamics to emerge? We present a multi-agent LLM simulation framework in which agents repeatedly interact, evaluate one another, and adapt their behavior through in-context learning accelerated by a coaching signal. To model human social behavior, we design behavioral reward functions that capture core drivers of online engagement, including social interaction, information seeking, self-presentation, coordination, and emotional support. These rewards align agent objectives with empirically observed user motivations, enabling the study of how network structures and group formations emerge from individual decision-making. Our experiments show that coached LLM agents develop stable interaction patterns and form emergent social ties, yielding network structures that mirror properties of real online communities. By combining behavioral rewards with in-context adaptation, our framework establishes a principled testbed for investigating collective dynamics in LLM populations and reveals how artificial agents may approximate or diverge from human-like social behavior. 

---
# An Argumentative Explanation Framework for Generalized Reason Model with Inconsistent Precedents 

**Authors**: Wachara Fungwacharakorn, Gauvain Bourgne, Ken Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2510.19263)  

**Abstract**: Precedential constraint is one foundation of case-based reasoning in AI and Law. It generally assumes that the underlying set of precedents must be consistent. To relax this assumption, a generalized notion of the reason model has been introduced. While several argumentative explanation approaches exist for reasoning with precedents based on the traditional consistent reason model, there has been no corresponding argumentative explanation method developed for this generalized reasoning framework accommodating inconsistent precedents. To address this question, this paper examines an extension of the derivation state argumentation framework (DSA-framework) to explain the reasoning according to the generalized notion of the reason model. 

---
# ChatGPT Unveils Its Limits: Principles of Law Deliver Checkmate 

**Authors**: Marianna Molinari, Ilaria Angela Amantea, Marinella Quaranta, Guido Governatori  

**Link**: [PDF](https://arxiv.org/pdf/2510.19261)  

**Abstract**: This study examines the performance of ChatGPT with an experiment in the legal domain. We compare the outcome with it a baseline using regular expressions (Regex), rather than focusing solely on the assessment against human performance. The study reveals that even if ChatGPT has access to the necessary knowledge and competencies, it is unable to assemble them, reason through, in a way that leads to an exhaustive result. This unveils a major limitation of ChatGPT. Intelligence encompasses the ability to break down complex issues and address them according to multiple required competencies, providing a unified and comprehensive solution. In the legal domain, one of the most crucial tasks is reading legal decisions and extracting key passages condensed from principles of law (PoLs), which are then incorporated into subsequent rulings by judges or defense documents by lawyers. In performing this task, artificial intelligence lacks an all-encompassing understanding and reasoning, which makes it inherently limited. Genuine intelligence, remains a uniquely human trait, at least in this particular field. 

---
# WebGraphEval: Multi-Turn Trajectory Evaluation for Web Agents using Graph Representation 

**Authors**: Yaoyao Qian, Yuanli Wang, Jinda Zhang, Yun Zong, Meixu Chen, Hanhan Zhou, Jindan Huang, Yifan Zeng, Xinyu Hu, Chan Hee Song, Danqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19205)  

**Abstract**: Current evaluation of web agents largely reduces to binary success metrics or conformity to a single reference trajectory, ignoring the structural diversity present in benchmark datasets. We present WebGraphEval, a framework that abstracts trajectories from multiple agents into a unified, weighted action graph. This representation is directly compatible with benchmarks such as WebArena, leveraging leaderboard runs and newly collected trajectories without modifying environments. The framework canonically encodes actions, merges recurring behaviors, and applies structural analyses including reward propagation and success-weighted edge statistics. Evaluations across thousands of trajectories from six web agents show that the graph abstraction captures cross-model regularities, highlights redundancy and inefficiency, and identifies critical decision points overlooked by outcome-based metrics. By framing web interaction as graph-structured data, WebGraphEval establishes a general methodology for multi-path, cross-agent, and efficiency-aware evaluation of web agents. 

---
# The Zero-Step Thinking: An Empirical Study of Mode Selection as Harder Early Exit in Reasoning Models 

**Authors**: Yuqiao Tan, Shizhu He, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.19176)  

**Abstract**: Reasoning models have demonstrated exceptional performance in tasks such as mathematics and logical reasoning, primarily due to their ability to engage in step-by-step thinking during the reasoning process. However, this often leads to overthinking, resulting in unnecessary computational overhead. To address this issue, Mode Selection aims to automatically decide between Long-CoT (Chain-of-Thought) or Short-CoT by utilizing either a Thinking or NoThinking mode. Simultaneously, Early Exit determines the optimal stopping point during the iterative reasoning process. Both methods seek to reduce the computational burden. In this paper, we first identify Mode Selection as a more challenging variant of the Early Exit problem, as they share similar objectives but differ in decision timing. While Early Exit focuses on determining the best stopping point for concise reasoning at inference time, Mode Selection must make this decision at the beginning of the reasoning process, relying on pre-defined fake thoughts without engaging in an explicit reasoning process, referred to as zero-step thinking. Through empirical studies on nine baselines, we observe that prompt-based approaches often fail due to their limited classification capabilities when provided with minimal hand-crafted information. In contrast, approaches that leverage internal information generally perform better across most scenarios but still exhibit issues with stability. Our findings indicate that existing methods relying solely on the information provided by models are insufficient for effectively addressing Mode Selection in scenarios with limited information, highlighting the ongoing challenges of this task. Our code is available at this https URL. 

---
# A Multi-faceted Analysis of Cognitive Abilities: Evaluating Prompt Methods with Large Language Models on the CONSORT Checklist 

**Authors**: Sohyeon Jeon, Hyung-Chul Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.19139)  

**Abstract**: Despite the rapid expansion of Large Language Models (LLMs) in healthcare, the ability of these systems to assess clinical trial reporting according to CONSORT standards remains unclear, particularly with respect to their cognitive and reasoning strategies. This study applies a behavioral and metacognitive analytic approach with expert-validated data, systematically comparing two representative LLMs under three prompt conditions. Clear differences emerged in how the models approached various CONSORT items, and prompt types, including shifts in reasoning style, explicit uncertainty, and alternative interpretations shaped response patterns. Our results highlight the current limitations of these systems in clinical compliance automation and underscore the importance of understanding their cognitive adaptations and strategic behavior in developing more explainable and reliable medical AI. 

---
# The MUSE Benchmark: Probing Music Perception and Auditory Relational Reasoning in Audio LLMS 

**Authors**: Brandon James Carone, Iran R. Roman, Pablo Ripollés  

**Link**: [PDF](https://arxiv.org/pdf/2510.19055)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated capabilities in audio understanding, but current evaluations may obscure fundamental weaknesses in relational reasoning. We introduce the Music Understanding and Structural Evaluation (MUSE) Benchmark, an open-source resource with 10 tasks designed to probe fundamental music perception skills. We evaluate four SOTA models (Gemini Pro and Flash, Qwen2.5-Omni, and Audio-Flamingo 3) against a large human baseline (N=200). Our results reveal a wide variance in SOTA capabilities and a persistent gap with human experts. While Gemini Pro succeeds on basic perception, Qwen and Audio Flamingo 3 perform at or near chance, exposing severe perceptual deficits. Furthermore, we find Chain-of-Thought (CoT) prompting provides inconsistent, often detrimental results. Our work provides a critical tool for evaluating invariant musical representations and driving development of more robust AI systems. 

---
# Rectifying Shortcut Behaviors in Preference-based Reward Learning 

**Authors**: Wenqian Ye, Guangtao Zheng, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19050)  

**Abstract**: In reinforcement learning from human feedback, preference-based reward models play a central role in aligning large language models to human-aligned behavior. However, recent studies show that these models are prone to reward hacking and often fail to generalize well due to over-optimization. They achieve high reward scores by exploiting shortcuts, that is, exploiting spurious features (e.g., response verbosity, agreeable tone, or sycophancy) that correlate with human preference labels in the training data rather than genuinely reflecting the intended objectives. In this paper, instead of probing these issues one at a time, we take a broader view of the reward hacking problem as shortcut behaviors and introduce a principled yet flexible approach to mitigate shortcut behaviors in preference-based reward learning. Inspired by the invariant theory in the kernel perspective, we propose Preference-based Reward Invariance for Shortcut Mitigation (PRISM), which learns group-invariant kernels with feature maps in a closed-form learning objective. Experimental results in several benchmarks show that our method consistently improves the accuracy of the reward model on diverse out-of-distribution tasks and reduces the dependency on shortcuts in downstream policy models, establishing a robust framework for preference-based alignment. 

---
# Timely Clinical Diagnosis through Active Test Selection 

**Authors**: Silas Ruhrberg Estévez, Nicolás Astorga, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2510.18988)  

**Abstract**: There is growing interest in using machine learning (ML) to support clinical diag- nosis, but most approaches rely on static, fully observed datasets and fail to reflect the sequential, resource-aware reasoning clinicians use in practice. Diagnosis remains complex and error prone, especially in high-pressure or resource-limited settings, underscoring the need for frameworks that help clinicians make timely and cost-effective decisions. We propose ACTMED (Adaptive Clinical Test selection via Model-based Experimental Design), a diagnostic framework that integrates Bayesian Experimental Design (BED) with large language models (LLMs) to better emulate real-world diagnostic reasoning. At each step, ACTMED selects the test expected to yield the greatest reduction in diagnostic uncertainty for a given patient. LLMs act as flexible simulators, generating plausible patient state distributions and supporting belief updates without requiring structured, task-specific training data. Clinicians can remain in the loop; reviewing test suggestions, interpreting intermediate outputs, and applying clinical judgment throughout. We evaluate ACTMED on real-world datasets and show it can optimize test selection to improve diagnostic accuracy, interpretability, and resource use. This represents a step to- ward transparent, adaptive, and clinician-aligned diagnostic systems that generalize across settings with reduced reliance on domain-specific data. 

---
# Test-time Verification via Optimal Transport: Coverage, ROC, & Sub-optimality 

**Authors**: Arpan Mukherjee, Marcello Bullo, Debabrota Basu, Deniz Gündüz  

**Link**: [PDF](https://arxiv.org/pdf/2510.18982)  

**Abstract**: While test-time scaling with verification has shown promise in improving the performance of large language models (LLMs), the role of the verifier and its imperfections remain underexplored. The effect of verification manifests through interactions of three quantities: (i) the generator's coverage, (ii) the verifier's region of convergence (ROC), and (iii) the sampling algorithm's sub-optimality. Though recent studies capture subsets of these factors, a unified framework quantifying the geometry of their interplay is missing. We frame verifiable test-time scaling as a transport problem. This characterizes the interaction of coverage, ROC, and sub-optimality, and uncovers that the sub-optimality--coverage curve exhibits three regimes. A transport regime -- where sub-optimality increases with coverage, a policy improvement regime -- where sub-optimality may decrease with coverage, depending on the verifier's ROC, and a saturation regime -- where sub-optimality plateaus, unaffected by coverage. We further propose and analyze two classes of sampling algorithms -- sequential and batched, and examine how their computational complexities shape these trade-offs. Empirical results with Qwen, Llama, and Gemma models corroborate our theoretical findings. 

---
# Semantic World Models 

**Authors**: Jacob Berg, Chuning Zhu, Yanda Bao, Ishan Durugkar, Abhishek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.19818)  

**Abstract**: Planning with world models offers a powerful paradigm for robotic control. Conventional approaches train a model to predict future frames conditioned on current frames and actions, which can then be used for planning. However, the objective of predicting future pixels is often at odds with the actual planning objective; strong pixel reconstruction does not always correlate with good planning decisions. This paper posits that instead of reconstructing future frames as pixels, world models only need to predict task-relevant semantic information about the future. For such prediction the paper poses world modeling as a visual question answering problem about semantic information in future frames. This perspective allows world modeling to be approached with the same tools underlying vision language models. Thus vision language models can be trained as "semantic" world models through a supervised finetuning process on image-action-text data, enabling planning for decision-making while inheriting many of the generalization and robustness properties from the pretrained vision-language models. The paper demonstrates how such a semantic world model can be used for policy improvement on open-ended robotics tasks, leading to significant generalization improvements over typical paradigms of reconstruction-based action-conditional world modeling. Website available at this https URL. 

---
# Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning 

**Authors**: Xichen Zhang, Sitong Wu, Yinghao Zhu, Haoru Tan, Shaozuo Yu, Ziyi He, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.19807)  

**Abstract**: Reinforcement learning from verifiable rewards has emerged as a powerful technique for enhancing the complex reasoning abilities of Large Language Models (LLMs). However, these methods are fundamentally constrained by the ''learning cliff'' phenomenon: when faced with problems far beyond their current capabilities, models consistently fail, yielding a persistent zero-reward signal. In policy optimization algorithms like GRPO, this collapses the advantage calculation to zero, rendering these difficult problems invisible to the learning gradient and stalling progress. To overcome this, we introduce Scaf-GRPO (Scaffolded Group Relative Policy Optimization), a progressive training framework that strategically provides minimal guidance only when a model's independent learning has plateaued. The framework first diagnoses learning stagnation and then intervenes by injecting tiered in-prompt hints, ranging from abstract concepts to concrete steps, enabling the model to construct a valid solution by itself. Extensive experiments on challenging mathematics benchmarks demonstrate Scaf-GRPO's effectiveness, boosting the pass@1 score of the Qwen2.5-Math-7B model on the AIME24 benchmark by a relative 44.3% over a vanilla GRPO baseline. This result demonstrates our framework provides a robust and effective methodology for unlocking a model's ability to solve problems previously beyond its reach, a critical step towards extending the frontier of autonomous reasoning in LLM. 

---
# Integrating Transparent Models, LLMs, and Practitioner-in-the-Loop: A Case of Nonprofit Program Evaluation 

**Authors**: Ji Ma, Albert Casella  

**Link**: [PDF](https://arxiv.org/pdf/2510.19799)  

**Abstract**: Public and nonprofit organizations often hesitate to adopt AI tools because most models are opaque even though standard approaches typically analyze aggregate patterns rather than offering actionable, case-level guidance. This study tests a practitioner-in-the-loop workflow that pairs transparent decision-tree models with large language models (LLMs) to improve predictive accuracy, interpretability, and the generation of practical insights. Using data from an ongoing college-success program, we build interpretable decision trees to surface key predictors. We then provide each tree's structure to an LLM, enabling it to reproduce case-level predictions grounded in the transparent models. Practitioners participate throughout feature engineering, model design, explanation review, and usability assessment, ensuring that field expertise informs the analysis at every stage. Results show that integrating transparent models, LLMs, and practitioner input yields accurate, trustworthy, and actionable case-level evaluations, offering a viable pathway for responsible AI adoption in the public and nonprofit sectors. 

---
# On Controlled Change: Generative AI's Impact on Professional Authority in Journalism 

**Authors**: Tomás Dodds, Wang Ngai Yeung, Claudia Mellado, Mathias-Felipe de Lima-Santos  

**Link**: [PDF](https://arxiv.org/pdf/2510.19792)  

**Abstract**: Using (generative) artificial intelligence tools and systems in journalism is expected to increase journalists' production rates, transform newsrooms' economic models, and further personalize the audience's news consumption practices. Since its release in 2022, OpenAI's ChatGPT and other large language models have raised the alarms inside news organizations, not only for bringing new challenges to news reporting and fact-checking but also for what these technologies would mean for journalists' professional authority in journalism. This paper examines how journalists in Dutch media manage the integration of AI technologies into their daily routines. Drawing from 13 interviews with editors, journalists, and innovation managers in different news outlets and media companies, we propose the concept of controlled change. as a heuristic to explain how journalists are proactively setting guidelines, experimenting with AI tools, and identifying their limitations and capabilities. Using professional authority as a theoretical framework, we argue that journalists anticipate and integrate AI technologies in a supervised manner and identify three primary mechanisms through which journalists manage this integration: (1) developing adaptive guidelines that align AI use with ethical codes, (2) experimenting with AI technologies to determine their necessity and fit, and (3) critically assessing the capabilities and limitations of AI systems. 

---
# AdaSPEC: Selective Knowledge Distillation for Efficient Speculative Decoders 

**Authors**: Yuezhou Hu, Jiaxin Guo, Xinyu Feng, Tuo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.19779)  

**Abstract**: Speculative Decoding (SD) accelerates large language model inference by employing a small draft model to generate predictions, which are then verified by a larger target model. The effectiveness of SD hinges on the alignment between these models, which is typically enhanced by Knowledge Distillation (KD). However, conventional KD methods aim to minimize the KL divergence between the draft and target models across all tokens, a goal that is misaligned with the true objective of SD, which is to maximize token acceptance rate. Therefore, draft models often struggle to fully assimilate the target model's knowledge due to capacity constraints, leading to suboptimal performance. To address this challenge, we propose AdaSPEC, a novel method that incorporates selective token filtering into the KD process. AdaSPEC utilizes a reference model to identify and filter out difficult-to-fit tokens, enabling the distillation of a draft model that better aligns with the target model on simpler tokens. This approach improves the overall token acceptance rate without compromising generation quality. We evaluate AdaSPEC across diverse tasks, including arithmetic reasoning, instruction-following, coding, and summarization, using model configurations of 31M/1.4B and 350M/2.7B parameters. Our results demonstrate that AdaSPEC consistently outperforms the state-of-the-art DistillSpec method, achieving higher acceptance rates across all tasks (up to 15\%). The code is publicly available at this https URL. 

---
# SmartSwitch: Advancing LLM Reasoning by Overcoming Underthinking via Promoting Deeper Thought Exploration 

**Authors**: Xichen Zhang, Sitong Wu, Haoru Tan, Shaozuo Yu, Yinghao Zhu, Ziyi He, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.19767)  

**Abstract**: The long chain-of-thought (LongCoT) capability is central to the recent breakthroughs achieved by large language models in complex reasoning tasks. However, the accompanying issue of ''underthinking'', where models exhibit shallow reasoning by frequently switching thoughts without sufficient exploration, limits both performance and token efficiency. To address this problem, we propose a simple yet effective reasoning strategy: the SmartSwitch inference framework. This framework can be easily integrated into any large language model as a plug-and-play solution, continuously monitoring the model's reasoning process to detect underthinking and guide it toward deeper exploration of promising but overlooked thoughts. Specifically, the perception module identifies points where thoughts switch and evaluates the potential of the preceding thought using an off-the-shelf process reward model (PRM). If a high-potential thought is found to be prematurely abandoned, the intervention module interrupts the ongoing inference, backtracks to the point before the switch, and inserts a "deepening prompt" to encourage further exploration along that promising path. Extensive experiments on challenging mathematical reasoning benchmarks demonstrate that our method significantly enhances the performance of various large language models of different sizes. 

---
# A Survey on Cache Methods in Diffusion Models: Toward Efficient Multi-Modal Generation 

**Authors**: Jiacheng Liu, Xinyu Wang, Yuqi Lin, Zhikai Wang, Peiru Wang, Peiliang Cai, Qinming Zhou, Zhengan Yan, Zexuan Yan, Zhengyi Shi, Chang Zou, Yue Ma, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19755)  

**Abstract**: Diffusion Models have become a cornerstone of modern generative AI for their exceptional generation quality and controllability. However, their inherent \textit{multi-step iterations} and \textit{complex backbone networks} lead to prohibitive computational overhead and generation latency, forming a major bottleneck for real-time applications. Although existing acceleration techniques have made progress, they still face challenges such as limited applicability, high training costs, or quality degradation.
Against this backdrop, \textbf{Diffusion Caching} offers a promising training-free, architecture-agnostic, and efficient inference paradigm. Its core mechanism identifies and reuses intrinsic computational redundancies in the diffusion process. By enabling feature-level cross-step reuse and inter-layer scheduling, it reduces computation without modifying model parameters. This paper systematically reviews the theoretical foundations and evolution of Diffusion Caching and proposes a unified framework for its classification and analysis.
Through comparative analysis of representative methods, we show that Diffusion Caching evolves from \textit{static reuse} to \textit{dynamic prediction}. This trend enhances caching flexibility across diverse tasks and enables integration with other acceleration techniques such as sampling optimization and model distillation, paving the way for a unified, efficient inference framework for future multimodal and interactive applications. We argue that this paradigm will become a key enabler of real-time and efficient generative AI, injecting new vitality into both theory and practice of \textit{Efficient Generative Intelligence}. 

---
# Learning Affordances at Inference-Time for Vision-Language-Action Models 

**Authors**: Ameesh Shah, William Chen, Adwait Godbole, Federico Mora, Sanjit A. Seshia, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2510.19752)  

**Abstract**: Solving complex real-world control tasks often takes multiple tries: if we fail at first, we reflect on what went wrong, and change our strategy accordingly to avoid making the same mistake. In robotics, Vision-Language-Action models (VLAs) offer a promising path towards solving complex control tasks, but lack the ability to contextually and dynamically readjust behavior when they fail to accomplish a task. In this work, we introduce Learning from Inference-Time Execution (LITEN), which connects a VLA low-level policy to a high-level VLM that conditions on past experiences by including them in-context, allowing it to learn the affordances and capabilities of the low-level VLA. Our approach iterates between a reasoning phase that generates and executes plans for the low-level VLA, and an assessment phase that reflects on the resulting execution and draws useful conclusions to be included in future reasoning contexts. Unlike similar approaches to self-refinement in non-robotics domains, LITEN must reflect on unstructured real-world robot trajectories (e.g., raw videos), which requires structured guiderails during assessment. Our experimental results demonstrate LITEN is able to effectively learn from past experience to generate plans that use high-affordance instructions to accomplish long-horizon tasks. 

---
# Enabling Granular Subgroup Level Model Evaluations by Generating Synthetic Medical Time Series 

**Authors**: Mahmoud Ibrahim, Bart Elen, Chang Sun, Gökhan Ertaylan, Michel Dumontier  

**Link**: [PDF](https://arxiv.org/pdf/2510.19728)  

**Abstract**: We present a novel framework for leveraging synthetic ICU time-series data not only to train but also to rigorously and trustworthily evaluate predictive models, both at the population level and within fine-grained demographic subgroups. Building on prior diffusion and VAE-based generators (TimeDiff, HealthGen, TimeAutoDiff), we introduce \textit{Enhanced TimeAutoDiff}, which augments the latent diffusion objective with distribution-alignment penalties. We extensively benchmark all models on MIMIC-III and eICU, on 24-hour mortality and binary length-of-stay tasks. Our results show that Enhanced TimeAutoDiff reduces the gap between real-on-synthetic and real-on-real evaluation (``TRTS gap'') by over 70\%, achieving $\Delta_{TRTS} \leq 0.014$ AUROC, while preserving training utility ($\Delta_{TSTR} \approx 0.01$). Crucially, for 32 intersectional subgroups, large synthetic cohorts cut subgroup-level AUROC estimation error by up to 50\% relative to small real test sets, and outperform them in 72--84\% of subgroups. This work provides a practical, privacy-preserving roadmap for trustworthy, granular model evaluation in critical care, enabling robust and reliable performance analysis across diverse patient populations without exposing sensitive EHR data, contributing to the overall trustworthiness of Medical AI. 

---
# Do Prompts Reshape Representations? An Empirical Study of Prompting Effects on Embeddings 

**Authors**: Cesar Gonzalez-Gutierrez, Dirk Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2510.19694)  

**Abstract**: Prompting is a common approach for leveraging LMs in zero-shot settings. However, the underlying mechanisms that enable LMs to perform diverse tasks without task-specific supervision remain poorly understood. Studying the relationship between prompting and the quality of internal representations can shed light on how pre-trained embeddings may support in-context task solving. In this empirical study, we conduct a series of probing experiments on prompt embeddings, analyzing various combinations of prompt templates for zero-shot classification. Our findings show that while prompting affects the quality of representations, these changes do not consistently correlate with the relevance of the prompts to the target task. This result challenges the assumption that more relevant prompts necessarily lead to better representations. We further analyze potential factors that may contribute to this unexpected behavior. 

---
# Toward Agentic Software Engineering Beyond Code: Framing Vision, Values, and Vocabulary 

**Authors**: Rashina Hoda  

**Link**: [PDF](https://arxiv.org/pdf/2510.19692)  

**Abstract**: Agentic AI is poised to usher in a seismic paradigm shift in Software Engineering (SE). As technologists rush head-along to make agentic AI a reality, SE researchers are driven to establish agentic SE as a research area. While early visions of agentic SE are primarily focused on code-related activities, early empirical evidence calls for a consideration of a range of socio-technical concerns to make it work in practice. This paper contributes to the emerging community vision by: (a) recommending an expansion of its scope beyond code, toward a 'whole of process' vision, grounding it in SE foundations and evolution and emerging agentic SE frameworks, (b) proposing a preliminary set of values and principles to guide efforts, and (c) sharing guidance on designing/using well-defined vocabulary for agentic SE. It is hoped that these ideas will encourage community collaborations and steer the SE community towards laying strong foundations of agentic SE so its not only inevitable but also deliberate and desirable in the long run. 

---
# Serverless GPU Architecture for Enterprise HR Analytics: A Production-Scale BDaaS Implementation 

**Authors**: Guilin Zhang, Wulan Guo, Ziqi Tan, Srinivas Vippagunta, Suchitra Raman, Shreeshankar Chatterjee, Ju Lin, Shang Liu, Mary Schladenhauffen, Jeffrey Luo, Hailong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19689)  

**Abstract**: Industrial and government organizations increasingly depend on data-driven analytics for workforce, finance, and regulated decision processes, where timeliness, cost efficiency, and compliance are critical. Distributed frameworks such as Spark and Flink remain effective for massive-scale batch or streaming analytics but introduce coordination complexity and auditing overheads that misalign with moderate-scale, latency-sensitive inference. Meanwhile, cloud providers now offer serverless GPUs, and models such as TabNet enable interpretable tabular ML, motivating new deployment blueprints for regulated environments. In this paper, we present a production-oriented Big Data as a Service (BDaaS) blueprint that integrates a single-node serverless GPU runtime with TabNet. The design leverages GPU acceleration for throughput, serverless elasticity for cost reduction, and feature-mask interpretability for IL4/FIPS compliance. We conduct benchmarks on the HR, Adult, and BLS datasets, comparing our approach against Spark and CPU baselines. Our results show that GPU pipelines achieve up to 4.5x higher throughput, 98x lower latency, and 90% lower cost per 1K inferences compared to Spark baselines, while compliance mechanisms add only ~5.7 ms latency with p99 < 22 ms. Interpretability remains stable under peak load, ensuring reliable auditability. Taken together, these findings provide a compliance-aware benchmark, a reproducible Helm-packaged blueprint, and a decision framework that demonstrate the practicality of secure, interpretable, and cost-efficient serverless GPU analytics for regulated enterprise and government settings. 

---
# Are Large Language Models Sensitive to the Motives Behind Communication? 

**Authors**: Addison J. Wu, Ryan Liu, Kerem Oktar, Theodore R. Sumers, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2510.19687)  

**Abstract**: Human communication is motivated: people speak, write, and create content with a particular communicative intent in mind. As a result, information that large language models (LLMs) and AI agents process is inherently framed by humans' intentions and incentives. People are adept at navigating such nuanced information: we routinely identify benevolent or self-serving motives in order to decide what statements to trust. For LLMs to be effective in the real world, they too must critically evaluate content by factoring in the motivations of the source -- for instance, weighing the credibility of claims made in a sales pitch. In this paper, we undertake a comprehensive study of whether LLMs have this capacity for motivational vigilance. We first employ controlled experiments from cognitive science to verify that LLMs' behavior is consistent with rational models of learning from motivated testimony, and find they successfully discount information from biased sources in a human-like manner. We then extend our evaluation to sponsored online adverts, a more naturalistic reflection of LLM agents' information ecosystems. In these settings, we find that LLMs' inferences do not track the rational models' predictions nearly as closely -- partly due to additional information that distracts them from vigilance-relevant considerations. However, a simple steering intervention that boosts the salience of intentions and incentives substantially increases the correspondence between LLMs and the rational model. These results suggest that LLMs possess a basic sensitivity to the motivations of others, but generalizing to novel real-world settings will require further improvements to these models. 

---
# Directive, Metacognitive or a Blend of Both? A Comparison of AI-Generated Feedback Types on Student Engagement, Confidence, and Outcomes 

**Authors**: Omar Alsaiari, Nilufar Baghaei, Jason M. Lodge, Omid Noroozi, Dragan Gašević, Marie Boden, Hassan Khosravi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19685)  

**Abstract**: Feedback is one of the most powerful influences on student learning, with extensive research examining how best to implement it in educational settings. Increasingly, feedback is being generated by artificial intelligence (AI), offering scalable and adaptive responses. Two widely studied approaches are directive feedback, which gives explicit explanations and reduces cognitive load to speed up learning, and metacognitive feedback which prompts learners to reflect, track their progress, and develop self-regulated learning (SRL) skills. While both approaches have clear theoretical advantages, their comparative effects on engagement, confidence, and quality of work remain underexplored. This study presents a semester-long randomised controlled trial with 329 students in an introductory design and programming course using an adaptive educational platform. Participants were assigned to receive directive, metacognitive, or hybrid AI-generated feedback that blended elements of both directive and metacognitive feedback. Results showed that revision behaviour differed across feedback conditions, with Hybrid prompting the most revisions compared to Directive and Metacognitive. Confidence ratings were uniformly high, and resource quality outcomes were comparable across conditions. These findings highlight the promise of AI in delivering feedback that balances clarity with reflection. Hybrid approaches, in particular, show potential to combine actionable guidance for immediate improvement with opportunities for self-reflection and metacognitive growth. 

---
# I Spy With My Model's Eye: Visual Search as a Behavioural Test for MLLMs 

**Authors**: John Burden, Jonathan Prunty, Ben Slater, Matthieu Tehenan, Greg Davis, Lucy Cheke  

**Link**: [PDF](https://arxiv.org/pdf/2510.19678)  

**Abstract**: Multimodal large language models (MLLMs) achieve strong performance on vision-language tasks, yet their visual processing is opaque. Most black-box evaluations measure task accuracy, but reveal little about underlying mechanisms. Drawing on cognitive psychology, we adapt classic visual search paradigms -- originally developed to study human perception -- to test whether MLLMs exhibit the ``pop-out'' effect, where salient visual features are detected independently of distractor set size. Using controlled experiments targeting colour, size and lighting features, we find that advanced MLLMs exhibit human-like pop-out effects in colour or size-based disjunctive (single feature) search, as well as capacity limits for conjunctive (multiple feature) search. We also find evidence to suggest that MLLMs, like humans, incorporate natural scene priors such as lighting direction into object representations. We reinforce our findings using targeted fine-tuning and mechanistic interpretability analyses. Our work shows how visual search can serve as a cognitively grounded diagnostic tool for evaluating perceptual capabilities in MLLMs. 

---
# Study of Training Dynamics for Memory-Constrained Fine-Tuning 

**Authors**: Aël Quélennec, Nour Hezbri, Pavlo Mozharovskyi, Van-Tam Nguyen, Enzo Tartaglione  

**Link**: [PDF](https://arxiv.org/pdf/2510.19675)  

**Abstract**: Memory-efficient training of deep neural networks has become increasingly important as models grow larger while deployment environments impose strict resource constraints. We propose TraDy, a novel transfer learning scheme leveraging two key insights: layer importance for updates is architecture-dependent and determinable a priori, while dynamic stochastic channel selection provides superior gradient approximation compared to static approaches. We introduce a dynamic channel selection approach that stochastically resamples channels between epochs within preselected layers. Extensive experiments demonstrate TraDy achieves state-of-the-art performance across various downstream tasks and architectures while maintaining strict memory constraints, achieving up to 99% activation sparsity, 95% weight derivative sparsity, and 97% reduction in FLOPs for weight derivative computation. 

---
# Unraveling Emotions with Pre-Trained Models 

**Authors**: Alejandro Pajón-Sanmartín, Francisco De Arriba-Pérez, Silvia García-Méndez, Fátima Leal, Benedita Malheiro, Juan Carlos Burguillo-Rial  

**Link**: [PDF](https://arxiv.org/pdf/2510.19668)  

**Abstract**: Transformer models have significantly advanced the field of emotion recognition. However, there are still open challenges when exploring open-ended queries for Large Language Models (LLMs). Although current models offer good results, automatic emotion analysis in open texts presents significant challenges, such as contextual ambiguity, linguistic variability, and difficulty interpreting complex emotional expressions. These limitations make the direct application of generalist models difficult. Accordingly, this work compares the effectiveness of fine-tuning and prompt engineering in emotion detection in three distinct scenarios: (i) performance of fine-tuned pre-trained models and general-purpose LLMs using simple prompts; (ii) effectiveness of different emotion prompt designs with LLMs; and (iii) impact of emotion grouping techniques on these models. Experimental tests attain metrics above 70% with a fine-tuned pre-trained model for emotion recognition. Moreover, the findings highlight that LLMs require structured prompt engineering and emotion grouping to enhance their performance. These advancements improve sentiment analysis, human-computer interaction, and understanding of user behavior across various domains. 

---
# From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction 

**Authors**: Zhida Zhao, Talas Fu, Yifan Wang, Lijun Wang, Huchuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19654)  

**Abstract**: Despite remarkable progress in driving world models, their potential for autonomous systems remains largely untapped: the world models are mostly learned for world simulation and decoupled from trajectory planning. While recent efforts aim to unify world modeling and planning in a single framework, the synergistic facilitation mechanism of world modeling for planning still requires further exploration. In this work, we introduce a new driving paradigm named Policy World Model (PWM), which not only integrates world modeling and trajectory planning within a unified architecture, but is also able to benefit planning using the learned world knowledge through the proposed action-free future state forecasting scheme. Through collaborative state-action prediction, PWM can mimic the human-like anticipatory perception, yielding more reliable planning performance. To facilitate the efficiency of video forecasting, we further introduce a dynamically enhanced parallel token generation mechanism, equipped with a context-guided tokenizer and an adaptive dynamic focal loss. Despite utilizing only front camera input, our method matches or exceeds state-of-the-art approaches that rely on multi-view and multi-modal inputs. Code and model weights will be released at this https URL. 

---
# Style Attack Disguise: When Fonts Become a Camouflage for Adversarial Intent 

**Authors**: Yangshijie Zhang, Xinda Wang, Jialin Liu, Wenqiang Wang, Zhicong Ma, Xingxing Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.19641)  

**Abstract**: With social media growth, users employ stylistic fonts and font-like emoji to express individuality, creating visually appealing text that remains human-readable. However, these fonts introduce hidden vulnerabilities in NLP models: while humans easily read stylistic text, models process these characters as distinct tokens, causing interference. We identify this human-model perception gap and propose a style-based attack, Style Attack Disguise (SAD). We design two sizes: light for query efficiency and strong for superior attack performance. Experiments on sentiment classification and machine translation across traditional models, LLMs, and commercial services demonstrate SAD's strong attack performance. We also show SAD's potential threats to multimodal tasks including text-to-image and text-to-speech generation. 

---
# Human-Agent Collaborative Paper-to-Page Crafting for Under $0.1 

**Authors**: Qianli Ma, Siyu Wang, Yilin Chen, Yinhao Tang, Yixiang Yang, Chang Guo, Bingjie Gao, Zhening Xing, Yanan Sun, Zhipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19600)  

**Abstract**: In the quest for scientific progress, communicating research is as vital as the discovery itself. Yet, researchers are often sidetracked by the manual, repetitive chore of building project webpages to make their dense papers accessible. While automation has tackled static slides and posters, the dynamic, interactive nature of webpages has remained an unaddressed challenge. To bridge this gap, we reframe the problem, arguing that the solution lies not in a single command, but in a collaborative, hierarchical process. We introduce $\textbf{AutoPage}$, a novel multi-agent system that embodies this philosophy. AutoPage deconstructs paper-to-page creation into a coarse-to-fine pipeline from narrative planning to multimodal content generation and interactive rendering. To combat AI hallucination, dedicated "Checker" agents verify each step against the source paper, while optional human checkpoints ensure the final product aligns perfectly with the author's vision, transforming the system from a mere tool into a powerful collaborative assistant. To rigorously validate our approach, we also construct $\textbf{PageBench}$, the first benchmark for this new task. Experiments show AutoPage not only generates high-quality, visually appealing pages but does so with remarkable efficiency in under 15 minutes for less than \$0.1. Code and dataset will be released at $\href{this https URL}{Webpage}$. 

---
# XBench: A Comprehensive Benchmark for Visual-Language Explanations in Chest Radiography 

**Authors**: Haozhe Luo, Shelley Zixin Shu, Ziyu Zhou, Sebastian Otalora, Mauricio Reyes  

**Link**: [PDF](https://arxiv.org/pdf/2510.19599)  

**Abstract**: Vision-language models (VLMs) have recently shown remarkable zero-shot performance in medical image understanding, yet their grounding ability, the extent to which textual concepts align with visual evidence, remains underexplored. In the medical domain, however, reliable grounding is essential for interpretability and clinical adoption. In this work, we present the first systematic benchmark for evaluating cross-modal interpretability in chest X-rays across seven CLIP-style VLM variants. We generate visual explanations using cross-attention and similarity-based localization maps, and quantitatively assess their alignment with radiologist-annotated regions across multiple pathologies. Our analysis reveals that: (1) while all VLM variants demonstrate reasonable localization for large and well-defined pathologies, their performance substantially degrades for small or diffuse lesions; (2) models that are pretrained on chest X-ray-specific datasets exhibit improved alignment compared to those trained on general-domain data. (3) The overall recognition ability and grounding ability of the model are strongly correlated. These findings underscore that current VLMs, despite their strong recognition ability, still fall short in clinically reliable grounding, highlighting the need for targeted interpretability benchmarks before deployment in medical practice. XBench code is available at this https URL 

---
# A Goal-Driven Survey on Root Cause Analysis 

**Authors**: Aoyang Fang, Haowen Yang, Haoze Dong, Qisheng Lu, Junjielong Xu, Pinjia He  

**Link**: [PDF](https://arxiv.org/pdf/2510.19593)  

**Abstract**: Root Cause Analysis (RCA) is a crucial aspect of incident management in large-scale cloud services. While the term root cause analysis or RCA has been widely used, different studies formulate the task differently. This is because the term "RCA" implicitly covers tasks with distinct underlying goals. For instance, the goal of localizing a faulty service for rapid triage is fundamentally different from identifying a specific functional bug for a definitive fix. However, previous surveys have largely overlooked these goal-based distinctions, conventionally categorizing papers by input data types (e.g., metric-based vs. trace-based methods). This leads to the grouping of works with disparate objectives, thereby obscuring the true progress and gaps in the field. Meanwhile, the typical audience of an RCA survey is either laymen who want to know the goals and big picture of the task or RCA researchers who want to figure out past research under the same task formulation. Thus, an RCA survey that organizes the related papers according to their goals is in high demand. To this end, this paper presents a goal-driven framework that effectively categorizes and integrates 135 papers on RCA in the context of cloud incident management based on their diverse goals, spanning the period from 2014 to 2025. In addition to the goal-driven categorization, it discusses the ultimate goal of all RCA papers as an umbrella covering different RCA formulations. Moreover, the paper discusses open challenges and future directions in RCA. 

---
# Detecting Latin in Historical Books with Large Language Models: A Multimodal Benchmark 

**Authors**: Yu Wu, Ke Shu, Jonas Fischer, Lidia Pivovarova, David Rosson, Eetu Mäkelä, Mikko Tolonen  

**Link**: [PDF](https://arxiv.org/pdf/2510.19585)  

**Abstract**: This paper presents a novel task of extracting Latin fragments from mixed-language historical documents with varied layouts. We benchmark and evaluate the performance of large foundation models against a multimodal dataset of 724 annotated pages. The results demonstrate that reliable Latin detection with contemporary models is achievable. Our study provides the first comprehensive analysis of these models' capabilities and limits for this task. 

---
# Multi-modal Co-learning for Earth Observation: Enhancing single-modality models via modality collaboration 

**Authors**: Francisco Mena, Dino Ienco, Cassio F. Dantas, Roberto Interdonato, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2510.19579)  

**Abstract**: Multi-modal co-learning is emerging as an effective paradigm in machine learning, enabling models to collaboratively learn from different modalities to enhance single-modality predictions. Earth Observation (EO) represents a quintessential domain for multi-modal data analysis, wherein diverse remote sensors collect data to sense our planet. This unprecedented volume of data introduces novel challenges. Specifically, the access to the same sensor modalities at both training and inference stages becomes increasingly complex based on real-world constraints affecting remote sensing platforms. In this context, multi-modal co-learning presents a promising strategy to leverage the vast amount of sensor-derived data available at the training stage to improve single-modality models for inference-time deployment. Most current research efforts focus on designing customized solutions for either particular downstream tasks or specific modalities available at the inference stage. To address this, we propose a novel multi-modal co-learning framework capable of generalizing across various tasks without targeting a specific modality for inference. Our approach combines contrastive and modality discriminative learning together to guide single-modality models to structure the internal model manifold into modality-shared and modality-specific information. We evaluate our framework on four EO benchmarks spanning classification and regression tasks across different sensor modalities, where only one of the modalities available during training is accessible at inference time. Our results demonstrate consistent predictive improvements over state-of-the-art approaches from the recent machine learning and computer vision literature, as well as EO-specific methods. The obtained findings validate our framework in the single-modality inference scenarios across a diverse range of EO applications. 

---
# A Matter of Time: Revealing the Structure of Time in Vision-Language Models 

**Authors**: Nidham Tekaya, Manuela Waldner, Matthias Zeppelzauer  

**Link**: [PDF](https://arxiv.org/pdf/2510.19559)  

**Abstract**: Large-scale vision-language models (VLMs) such as CLIP have gained popularity for their generalizable and expressive multimodal representations. By leveraging large-scale training data with diverse textual metadata, VLMs acquire open-vocabulary capabilities, solving tasks beyond their training scope. This paper investigates the temporal awareness of VLMs, assessing their ability to position visual content in time. We introduce TIME10k, a benchmark dataset of over 10,000 images with temporal ground truth, and evaluate the time-awareness of 37 VLMs by a novel methodology. Our investigation reveals that temporal information is structured along a low-dimensional, non-linear manifold in the VLM embedding space. Based on this insight, we propose methods to derive an explicit ``timeline'' representation from the embedding space. These representations model time and its chronological progression and thereby facilitate temporal reasoning tasks. Our timeline approaches achieve competitive to superior accuracy compared to a prompt-based baseline while being computationally efficient. All code and data are available at this https URL. 

---
# Demonstrating Real Advantage of Machine-Learning-Enhanced Monte Carlo for Combinatorial Optimization 

**Authors**: Luca Maria Del Bono, Federico Ricci-Tersenghi, Francesco Zamponi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19544)  

**Abstract**: Combinatorial optimization problems are central to both practical applications and the development of optimization methods. While classical and quantum algorithms have been refined over decades, machine learning-assisted approaches are comparatively recent and have not yet consistently outperformed simple, state-of-the-art classical methods. Here, we focus on a class of Quadratic Unconstrained Binary Optimization (QUBO) problems, specifically the challenge of finding minimum energy configurations in three-dimensional Ising spin glasses. We use a Global Annealing Monte Carlo algorithm that integrates standard local moves with global moves proposed via machine learning. We show that local moves play a crucial role in achieving optimal performance. Benchmarking against Simulated Annealing and Population Annealing, we demonstrate that Global Annealing not only surpasses the performance of Simulated Annealing but also exhibits greater robustness than Population Annealing, maintaining effectiveness across problem hardness and system size without hyperparameter tuning. These results provide, to our knowledge, the first clear and robust evidence that a machine learning-assisted optimization method can exceed the capabilities of classical state-of-the-art techniques in a combinatorial optimization setting. 

---
# Insights into the Unknown: Federated Data Diversity Analysis on Molecular Data 

**Authors**: Markus Bujotzek, Evelyn Trautmann, Calum Hand, Ian Hales  

**Link**: [PDF](https://arxiv.org/pdf/2510.19535)  

**Abstract**: AI methods are increasingly shaping pharmaceutical drug discovery. However, their translation to industrial applications remains limited due to their reliance on public datasets, lacking scale and diversity of proprietary pharmaceutical data. Federated learning (FL) offers a promising approach to integrate private data into privacy-preserving, collaborative model training across data silos. This federated data access complicates important data-centric tasks such as estimating dataset diversity, performing informed data splits, and understanding the structure of the combined chemical space. To address this gap, we investigate how well federated clustering methods can disentangle and represent distributed molecular data. We benchmark three approaches, Federated kMeans (Fed-kMeans), Federated Principal Component Analysis combined with Fed-kMeans (Fed-PCA+Fed-kMeans), and Federated Locality-Sensitive Hashing (Fed-LSH), against their centralized counterparts on eight diverse molecular datasets. Our evaluation utilizes both, standard mathematical and a chemistry-informed evaluation metrics, SF-ICF, that we introduce in this work. The large-scale benchmarking combined with an in-depth explainability analysis shows the importance of incorporating domain knowledge through chemistry-informed metrics, and on-client explainability analyses for federated diversity analysis on molecular data. 

---
# Optimizing the Unknown: Black Box Bayesian Optimization with Energy-Based Model and Reinforcement Learning 

**Authors**: Ruiyao Miao, Junren Xiao, Shiya Tsang, Hui Xiong, Yingnian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19530)  

**Abstract**: Existing Bayesian Optimization (BO) methods typically balance exploration and exploitation to optimize costly objective functions. However, these methods often suffer from a significant one-step bias, which may lead to convergence towards local optima and poor performance in complex or high-dimensional tasks. Recently, Black-Box Optimization (BBO) has achieved success across various scientific and engineering domains, particularly when function evaluations are costly and gradients are unavailable. Motivated by this, we propose the Reinforced Energy-Based Model for Bayesian Optimization (REBMBO), which integrates Gaussian Processes (GP) for local guidance with an Energy-Based Model (EBM) to capture global structural information. Notably, we define each Bayesian Optimization iteration as a Markov Decision Process (MDP) and use Proximal Policy Optimization (PPO) for adaptive multi-step lookahead, dynamically adjusting the depth and direction of exploration to effectively overcome the limitations of traditional BO methods. We conduct extensive experiments on synthetic and real-world benchmarks, confirming the superior performance of REBMBO. Additional analyses across various GP configurations further highlight its adaptability and robustness. 

---
# From Prototypes to Sparse ECG Explanations: SHAP-Driven Counterfactuals for Multivariate Time-Series Multi-class Classification 

**Authors**: Maciej Mozolewski, Betül Bayrak, Kerstin Bach, Grzegorz J. Nalepa  

**Link**: [PDF](https://arxiv.org/pdf/2510.19514)  

**Abstract**: In eXplainable Artificial Intelligence (XAI), instance-based explanations for time series have gained increasing attention due to their potential for actionable and interpretable insights in domains such as healthcare. Addressing the challenges of explainability of state-of-the-art models, we propose a prototype-driven framework for generating sparse counterfactual explanations tailored to 12-lead ECG classification models. Our method employs SHAP-based thresholds to identify critical signal segments and convert them into interval rules, uses Dynamic Time Warping (DTW) and medoid clustering to extract representative prototypes, and aligns these prototypes to query R-peaks for coherence with the sample being explained. The framework generates counterfactuals that modify only 78% of the original signal while maintaining 81.3% validity across all classes and achieving 43% improvement in temporal stability. We evaluate three variants of our approach, Original, Sparse, and Aligned Sparse, with class-specific performance ranging from 98.9% validity for myocardial infarction (MI) to challenges with hypertrophy (HYP) detection (13.2%). This approach supports near realtime generation (< 1 second) of clinically valid counterfactuals and provides a foundation for interactive explanation platforms. Our findings establish design principles for physiologically-aware counterfactual explanations in AI-based diagnosis systems and outline pathways toward user-controlled explanation interfaces for clinical deployment. 

---
# Modeling realistic human behavior using generative agents in a multimodal transport system: Software architecture and Application to Toulouse 

**Authors**: Trung-Dung Vu, Benoit Gaudou, Kamaldeep Singh Oberoi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19497)  

**Abstract**: Modeling realistic human behaviour to understand people's mode choices in order to propose personalised mobility solutions remains challenging. This paper presents an architecture for modeling realistic human mobility behavior in complex multimodal transport systems, demonstrated through a case study in Toulouse, France. We apply Large Language Models (LLMs) within an agent-based simulation to capture decision-making in a real urban setting. The framework integrates the GAMA simulation platform with an LLM-based generative agent, along with General Transit Feed Specification (GTFS) data for public transport, and OpenTripPlanner for multimodal routing. GAMA platform models the interactive transport environment, providing visualization and dynamic agent interactions while eliminating the need to construct the simulation environment from scratch. This design enables a stronger focus on developing generative agents and evaluating their performance in transport decision-making processes. Over a simulated month, results show that agents not only make context-aware transport decisions but also form habits over time. We conclude that combining LLMs with agent-based simulation offers a promising direction for advancing intelligent transportation systems and personalised multimodal mobility solutions. We also discuss some limitations of this approach and outline future work on scaling to larger regions, integrating real-time data, and refining memory models. 

---
# CARES: Context-Aware Resolution Selector for VLMs 

**Authors**: Moshe Kimhi, Nimrod Shabtay, Raja Giryes, Chaim Baskin, Eli Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2510.19496)  

**Abstract**: Large vision-language models (VLMs) commonly process images at native or high resolution to remain effective across tasks. This inflates visual tokens ofter to 97-99% of total tokens, resulting in high compute and latency, even when low-resolution images would suffice. We introduce \emph{CARES}-a \textbf{C}ontext-\textbf{A}ware \textbf{R}esolution \textbf{S}elector, a lightweight preprocessing module that, given an image-query pair, predicts the \emph{minimal} sufficient input resolution. CARES uses a compact VLM (350M) to extract features and predict when a target pretrained VLM's response converges to its peak ability to answer correctly. Though trained as a discrete classifier over a set of optional resolutions, CARES interpolates continuous resolutions at inference for fine-grained control. Across five multimodal benchmarks spanning documents and natural images, as well as diverse target VLMs, CARES preserves task performance while reducing compute by up to 80%. 

---
# Using Non-Expert Data to Robustify Imitation Learning via Offline Reinforcement Learning 

**Authors**: Kevin Huang, Rosario Scalise, Cleah Winston, Ayush Agrawal, Yunchu Zhang, Rohan Baijal, Markus Grotz, Byron Boots, Benjamin Burchfiel, Hongkai Dai, Masha Itkina, Paarth Shah, Abhishek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.19495)  

**Abstract**: Imitation learning has proven effective for training robots to perform complex tasks from expert human demonstrations. However, it remains limited by its reliance on high-quality, task-specific data, restricting adaptability to the diverse range of real-world object configurations and scenarios. In contrast, non-expert data -- such as play data, suboptimal demonstrations, partial task completions, or rollouts from suboptimal policies -- can offer broader coverage and lower collection costs. However, conventional imitation learning approaches fail to utilize this data effectively. To address these challenges, we posit that with right design decisions, offline reinforcement learning can be used as a tool to harness non-expert data to enhance the performance of imitation learning policies. We show that while standard offline RL approaches can be ineffective at actually leveraging non-expert data under the sparse data coverage settings typically encountered in the real world, simple algorithmic modifications can allow for the utilization of this data, without significant additional assumptions. Our approach shows that broadening the support of the policy distribution can allow imitation algorithms augmented by offline RL to solve tasks robustly, showing considerably enhanced recovery and generalization behavior. In manipulation tasks, these innovations significantly increase the range of initial conditions where learned policies are successful when non-expert data is incorporated. Moreover, we show that these methods are able to leverage all collected data, including partial or suboptimal demonstrations, to bolster task-directed policy performance. This underscores the importance of algorithmic techniques for using non-expert data for robust policy learning in robotics. 

---
# VideoAgentTrek: Computer Use Pretraining from Unlabeled Videos 

**Authors**: Dunjie Lu, Yiheng Xu, Junli Wang, Haoyuan Wu, Xinyuan Wang, Zekun Wang, Junlin Yang, Hongjin Su, Jixuan Chen, Junda Chen, Yuchen Mao, Jingren Zhou, Junyang Lin, Binyuan Hui, Tao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19488)  

**Abstract**: Training computer-use agents requires massive amounts of GUI interaction data, but manually annotating action trajectories at scale is prohibitively expensive. We present VideoAgentTrek, a scalable pipeline that automatically mines training data from publicly available screen-recorded videos at web scale, eliminating the need for manual annotation. Our approach addresses a key challenge: raw videos contain implicit demonstrations but lack explicit action labels. To solve this, we develop Video2Action, an inverse dynamics module (IDM) with two components: (1) a video grounding model that detects and localizes GUI actions with precise temporal boundaries and context, and (2) an action-content recognizer that extracts structured parameters like click coordinates and typed text with high fidelity. Applied to 39,000 YouTube tutorial videos, our pipeline generates 1.52 million interaction steps automatically. We leverage this data through continued pretraining followed by supervised fine-tuning. On OSWorld-Verified, our approach improves task success rates from 9.3% (SFT-only baseline) to 15.8%, a 70% relative improvement. On AgentNetBench, step accuracy increases from 64.1% to 69.3%. Our results demonstrate that passive internet videos can be transformed into high-quality supervision for computer-use agents, providing a scalable alternative to expensive manual annotation. 

---
# KnowMol: Advancing Molecular Large Language Models with Multi-Level Chemical Knowledge 

**Authors**: Zaifei Yang, Hong Chang, Ruibing Hou, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.19484)  

**Abstract**: The molecular large language models have garnered widespread attention due to their promising potential on molecular applications. However, current molecular large language models face significant limitations in understanding molecules due to inadequate textual descriptions and suboptimal molecular representation strategies during pretraining. To address these challenges, we introduce KnowMol-100K, a large-scale dataset with 100K fine-grained molecular annotations across multiple levels, bridging the gap between molecules and textual descriptions. Additionally, we propose chemically-informative molecular representation, effectively addressing limitations in existing molecular representation strategies. Building upon these innovations, we develop KnowMol, a state-of-the-art multi-modal molecular large language model. Extensive experiments demonstrate that KnowMol achieves superior performance across molecular understanding and generation tasks.
GitHub: this https URL
Huggingface: this https URL 

---
# Graph Unlearning Meets Influence-aware Negative Preference Optimization 

**Authors**: Qiang Chen, Zhongze Wu, Ang He, Xi Lin, Shuo Jiang, Shan You, Chang Xu, Yi Chen, Xiu Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.19479)  

**Abstract**: Recent advancements in graph unlearning models have enhanced model utility by preserving the node representation essentially invariant, while using gradient ascent on the forget set to achieve unlearning. However, this approach causes a drastic degradation in model utility during the unlearning process due to the rapid divergence speed of gradient ascent. In this paper, we introduce \textbf{INPO}, an \textbf{I}nfluence-aware \textbf{N}egative \textbf{P}reference \textbf{O}ptimization framework that focuses on slowing the divergence speed and improving the robustness of the model utility to the unlearning process. Specifically, we first analyze that NPO has slower divergence speed and theoretically propose that unlearning high-influence edges can reduce impact of unlearning. We design an influence-aware message function to amplify the influence of unlearned edges and mitigate the tight topological coupling between the forget set and the retain set. The influence of each edge is quickly estimated by a removal-based method. Additionally, we propose a topological entropy loss from the perspective of topology to avoid excessive information loss in the local structure during unlearning. Extensive experiments conducted on five real-world datasets demonstrate that INPO-based model achieves state-of-the-art performance on all forget quality metrics while maintaining the model's utility. Codes are available at \href{this https URL}{this https URL}. 

---
# A Concrete Roadmap towards Safety Cases based on Chain-of-Thought Monitoring 

**Authors**: Julian Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2510.19476)  

**Abstract**: As AI systems approach dangerous capability levels where inability safety cases become insufficient, we need alternative approaches to ensure safety. This paper presents a roadmap for constructing safety cases based on chain-of-thought (CoT) monitoring in reasoning models and outlines our research agenda. We argue that CoT monitoring might support both control and trustworthiness safety cases. We propose a two-part safety case: (1) establishing that models lack dangerous capabilities when operating without their CoT, and (2) ensuring that any dangerous capabilities enabled by a CoT are detectable by CoT monitoring. We systematically examine two threats to monitorability: neuralese and encoded reasoning, which we categorize into three forms (linguistic drift, steganography, and alien reasoning) and analyze their potential drivers. We evaluate existing and novel techniques for maintaining CoT faithfulness. For cases where models produce non-monitorable reasoning, we explore the possibility of extracting a monitorable CoT from a non-monitorable CoT. To assess the viability of CoT monitoring safety cases, we establish prediction markets to aggregate forecasts on key technical milestones influencing their feasibility. 

---
# HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid Expert/Data Transmission 

**Authors**: Weihao Yang, Hao Huang, Donglei Wu, Ningke Li, Yanqi Pan, Qiyang Zheng, Wen Xia, Shiyi Li, Qiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19470)  

**Abstract**: Mixture-of-Experts (MoE) has become a popular architecture for scaling large models. However, the rapidly growing scale outpaces model training on a single DC, driving a shift toward a more flexible, cross-DC training paradigm. Under this, Expert Parallelism (EP) of MoE faces significant scalability issues due to the limited cross-DC bandwidth. Specifically, existing EP optimizations attempt to overlap data communication and computation, which has little benefit in low-bandwidth scenarios due to a much longer data communication time. Therefore, the trends of cross-DC EP scaling is fast becoming a critical roadblock to the continued growth of MoE models.
To address this, we propose HybridEP, a modeling-guided framework to optimize EP under constrained bandwidth. Our key idea is to dynamically transform the spatial placement of experts to reduce data communication traffic and frequency, thereby minimizing EP's communication overheads. However, it is non-trivial to find the optimal solution because it complicates the original communication pattern by mixing data and expert communication. We therefore build a stream-based model to determine the optimal transmission ratio. Guided by this, we incorporate two techniques: (1) domain-based partition to construct the mapping between hybrid patterns and specific communication topology at GPU level, and (2) parameter-efficient migration to further refine this topology by reducing expert transmission overhead and enlarging the domain size. Combining all these designs, HybridEP can be considered as a more general EP with better scalability. Experimental results show that HybridEP outperforms existing state-of-the-art MoE training systems by up to 5.6x under constrained bandwidth. We further compare HybridEP and EP on large-scale simulations. HybridEP achieves up to 1.45x speedup with 1k DCs under different bandwidths. 

---
# Universal Quantitative Abstraction: Categorical Duality and Logical Completeness for Probabilistic Systems 

**Authors**: Nivar Anwer  

**Link**: [PDF](https://arxiv.org/pdf/2510.19444)  

**Abstract**: A unified theory of quantitative abstraction is presented for probabilistic systems that links category theory, optimal transport, and quantitative modal logic. At its core is a canonical $ \varepsilon $-quotient endowed with a universal property: among all $ \varepsilon $-abstractions, it is the most informative one that respects a prescribed bound on value loss. This construction induces an adjunction between abstraction and realization functors $ (Q_{\varepsilon} \dashv R_{\varepsilon}) $, established via the Special Adjoint Functor Theorem, revealing a categorical duality between metric structure and logical semantics. A behavioral pseudometric is characterized as the unique fixed point of a Bellman-style operator, with contraction and Lipschitz properties proved in a coalgebraic setting. A quantitative modal $ \mu $-calculus is introduced and shown to be expressively complete for logically representable systems, so that behavioral distance coincides with maximal logical deviation. Compositionality under interface refinement is analyzed, clarifying how abstractions interact across system boundaries. An exact validation suite on finite Markov decision processes corroborates the contraction property, value-loss bounds, stability under perturbation, adversarial distinguishability, and scalability, demonstrating both robustness and computational feasibility. The resulting framework provides principled targets for state aggregation and representation learning, with mathematically precise guarantees for value-function approximation in stochastic domains. 

---
# Neural Variational Dropout Processes 

**Authors**: Insu Jeon, Youngjin Park, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.19425)  

**Abstract**: Learning to infer the conditional posterior model is a key step for robust meta-learning. This paper presents a new Bayesian meta-learning approach called Neural Variational Dropout Processes (NVDPs). NVDPs model the conditional posterior distribution based on a task-specific dropout; a low-rank product of Bernoulli experts meta-model is utilized for a memory-efficient mapping of dropout rates from a few observed contexts. It allows for a quick reconfiguration of a globally learned and shared neural network for new tasks in multi-task few-shot learning. In addition, NVDPs utilize a novel prior conditioned on the whole task data to optimize the conditional \textit{dropout} posterior in the amortized variational inference. Surprisingly, this enables the robust approximation of task-specific dropout rates that can deal with a wide range of functional ambiguities and uncertainties. We compared the proposed method with other meta-learning approaches in the few-shot learning tasks such as 1D stochastic regression, image inpainting, and classification. The results show the excellent performance of NVDPs. 

---
# FairNet: Dynamic Fairness Correction without Performance Loss via Contrastive Conditional LoRA 

**Authors**: Songqi Zhou, Zeyuan Liu, Benben Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19421)  

**Abstract**: Ensuring fairness in machine learning models is a critical challenge. Existing debiasing methods often compromise performance, rely on static correction strategies, and struggle with data sparsity, particularly within minority groups. Furthermore, their utilization of sensitive attributes is often suboptimal, either depending excessively on complete attribute labeling or disregarding these attributes entirely. To overcome these limitations, we propose FairNet, a novel framework for dynamic, instance-level fairness correction. FairNet integrates a bias detector with conditional low-rank adaptation (LoRA), which enables selective activation of the fairness correction mechanism exclusively for instances identified as biased, and thereby preserve performance on unbiased instances. A key contribution is a new contrastive loss function for training the LoRA module, specifically designed to minimize intra-class representation disparities across different sensitive groups and effectively address underfitting in minority groups. The FairNet framework can flexibly handle scenarios with complete, partial, or entirely absent sensitive attribute labels. Theoretical analysis confirms that, under moderate TPR/FPR for the bias detector, FairNet can enhance the performance of the worst group without diminishing overall model performance, and potentially yield slight performance improvements. Comprehensive empirical evaluations across diverse vision and language benchmarks validate the effectiveness of FairNet. 

---
# Monitoring LLM-based Multi-Agent Systems Against Corruptions via Node Evaluation 

**Authors**: Chengcan Wu, Zhixin Zhang, Mingqian Xu, Zeming Wei, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.19420)  

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have become a popular paradigm of AI applications. However, trustworthiness issues in MAS remain a critical concern. Unlike challenges in single-agent systems, MAS involve more complex communication processes, making them susceptible to corruption attacks. To mitigate this issue, several defense mechanisms have been developed based on the graph representation of MAS, where agents represent nodes and communications form edges. Nevertheless, these methods predominantly focus on static graph defense, attempting to either detect attacks in a fixed graph structure or optimize a static topology with certain defensive capabilities. To address this limitation, we propose a dynamic defense paradigm for MAS graph structures, which continuously monitors communication within the MAS graph, then dynamically adjusts the graph topology, accurately disrupts malicious communications, and effectively defends against evolving and diverse dynamic attacks. Experimental results in increasingly complex and dynamic MAS environments demonstrate that our method significantly outperforms existing MAS defense mechanisms, contributing an effective guardrail for their trustworthy applications. Our code is available at this https URL. 

---
# EchoFake: A Replay-Aware Dataset for Practical Speech Deepfake Detection 

**Authors**: Tong Zhang, Yihuan Huang, Yanzhen Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.19414)  

**Abstract**: The growing prevalence of speech deepfakes has raised serious concerns, particularly in real-world scenarios such as telephone fraud and identity theft. While many anti-spoofing systems have demonstrated promising performance on lab-generated synthetic speech, they often fail when confronted with physical replay attacks-a common and low-cost form of attack used in practical settings. Our experiments show that models trained on existing datasets exhibit severe performance degradation, with average accuracy dropping to 59.6% when evaluated on replayed audio. To bridge this gap, we present EchoFake, a comprehensive dataset comprising more than 120 hours of audio from over 13,000 speakers, featuring both cutting-edge zero-shot text-to-speech (TTS) speech and physical replay recordings collected under varied devices and real-world environmental settings. Additionally, we evaluate three baseline detection models and show that models trained on EchoFake achieve lower average EERs across datasets, indicating better generalization. By introducing more practical challenges relevant to real-world deployment, EchoFake offers a more realistic foundation for advancing spoofing detection methods. 

---
# ToMMeR -- Efficient Entity Mention Detection from Large Language Models 

**Authors**: Victor Morand, Nadi Tomeh, Josiane Mothe, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2510.19410)  

**Abstract**: Identifying which text spans refer to entities -- mention detection -- is both foundational for information extraction and a known performance bottleneck. We introduce ToMMeR, a lightweight model (<300K parameters) probing mention detection capabilities from early LLM layers. Across 13 NER benchmarks, ToMMeR achieves 93\% recall zero-shot, with over 90\% precision using an LLM as a judge showing that ToMMeR rarely produces spurious predictions despite high recall. Cross-model analysis reveals that diverse architectures (14M-15B parameters) converge on similar mention boundaries (DICE >75\%), confirming that mention detection emerges naturally from language modeling. When extended with span classification heads, ToMMeR achieves near SOTA NER performance (80-87\% F1 on standard benchmarks). Our work provides evidence that structured entity representations exist in early transformer layers and can be efficiently recovered with minimal parameters. 

---
# ColorAgent: Building A Robust, Personalized, and Interactive OS Agent 

**Authors**: Ning Li, Qiqiang Lin, Zheng Wu, Xiaoyun Mo, Weiming Zhang, Yin Zhao, Xiangmou Qu, Jiamu Zhou, Jun Wang, Congmin Zheng, Yuanyi Song, Hongjiang Chen, Heyuan Huang, Jihong Wang, Jiaxin Yin, Jingwei Yu, Junwei Liao, Qiuying Peng, Xingyu Lou, Jun Wang, Weiwen Liu, Zhuosheng Zhang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19386)  

**Abstract**: With the advancements in hardware, software, and large language model technologies, the interaction between humans and operating systems has evolved from the command-line interface to the rapidly emerging AI agent interactions. Building an operating system (OS) agent capable of executing user instructions and faithfully following user desires is becoming a reality. In this technical report, we present ColorAgent, an OS agent designed to engage in long-horizon, robust interactions with the environment while also enabling personalized and proactive user interaction. To enable long-horizon interactions with the environment, we enhance the model's capabilities through step-wise reinforcement learning and self-evolving training, while also developing a tailored multi-agent framework that ensures generality, consistency, and robustness. In terms of user interaction, we explore personalized user intent recognition and proactive engagement, positioning the OS agent not merely as an automation tool but as a warm, collaborative partner. We evaluate ColorAgent on the AndroidWorld and AndroidLab benchmarks, achieving success rates of 77.2% and 50.7%, respectively, establishing a new state of the art. Nonetheless, we note that current benchmarks are insufficient for a comprehensive evaluation of OS agents and propose further exploring directions in future work, particularly in the areas of evaluation paradigms, agent collaboration, and security. Our code is available at this https URL. 

---
# The Massive Legal Embedding Benchmark (MLEB) 

**Authors**: Umar Butler, Abdur-Rahman Butler, Adrian Lucas Malec  

**Link**: [PDF](https://arxiv.org/pdf/2510.19365)  

**Abstract**: We present the Massive Legal Embedding Benchmark (MLEB), the largest, most diverse, and most comprehensive open-source benchmark for legal information retrieval to date. MLEB consists of ten expert-annotated datasets spanning multiple jurisdictions (the US, UK, EU, Australia, Ireland, and Singapore), document types (cases, legislation, regulatory guidance, contracts, and literature), and task types (search, zero-shot classification, and question answering). Seven of the datasets in MLEB were newly constructed in order to fill domain and jurisdictional gaps in the open-source legal information retrieval landscape. We document our methodology in building MLEB and creating the new constituent datasets, and release our code, results, and data openly to assist with reproducible evaluations. 

---
# AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation 

**Authors**: Xianyang Liu, Yilin Liu, Shuai Wang, Hao Cheng, Andrew Estornell, Yuzhi Zhao, Jiaheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.19361)  

**Abstract**: The creation of high-quality datasets to improve Large Language Model (LLM) reasoning remains a significant challenge, as current methods often suffer from generating low-quality/incorrect answers and limited information richness from available data sources. To address this, we propose AgenticMath, a novel agentic pipeline for generating high-quality mathematical question-answer pairs to enhance the supervised fine-tuning of LLMs. Our method operates through four stages: (1) Seed Question Filter that selects questions with high information richness, complexity, and clarity; (2) an Agentic Question Rephrase step that employs a multi-agent system to generate diverse, logically consistent paraphrases; (3) an Answer Augment step where rewrite answers using chain-of-thought reasoning to enhance numerical and logical correctness, without reliance on human-provided labels; and (4) a final Question and Answer Evaluation that retains only the most superior pairs. Extensive experiments demonstrate that, fine-tuning 3B-8B parameter LLMs on AgenticMath generated datasets (comprising only 30-60K math samples) achieves competitive or superior performance on diverse in domain and out-of-domain mathematical reasoning benchmarks compared to baselines trained on much more data (e.g., 400K or 2.3M samples). Our work demonstrates that targeted, high-quality data generation is a more efficient path to improving mathematical reasoning in LLMs than large-scale, low-quality alternatives. 

---
# M3-SLU: Evaluating Speaker-Attributed Reasoning in Multimodal Large Language Models 

**Authors**: Yejin Kwon, Taewoo Kang, Hyunsoo Yoon, Changouk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.19358)  

**Abstract**: We present M3-SLU, a new multimodal large language model (MLLM) benchmark for evaluating multi-speaker, multi-turn spoken language understanding. While recent models show strong performance in speech and text comprehension, they still struggle with speaker-attributed reasoning, the ability to understand who said what and when in natural conversations. M3-SLU is built from four open corpora (CHiME-6, MELD, MultiDialog, and AMI) and comprises over 12,000 validated instances with paired audio, transcripts, and metadata. It includes two tasks: (1) Speaker-Attributed Question Answering and (2) Speaker Attribution via Utterance Matching. We provide baseline results for both cascaded pipelines and end-to-end MLLMs, evaluated using an LLM-as-Judge and accuracy metrics. Results show that while models can capture what was said, they often fail to identify who said it, revealing a key gap in speaker-aware dialogue understanding. M3-SLU offers as a challenging benchmark to advance research in speaker-aware multimodal understanding. 

---
# Learning To Defer To A Population With Limited Demonstrations 

**Authors**: Nilesh Ramgolam, Gustavo Carneiro, Hsiang-Ting, Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.19351)  

**Abstract**: This paper addresses the critical data scarcity that hinders the practical deployment of learning to defer (L2D) systems to the population. We introduce a context-aware, semi-supervised framework that uses meta-learning to generate expert-specific embeddings from only a few demonstrations. We demonstrate the efficacy of a dual-purpose mechanism, where these embeddings are used first to generate a large corpus of pseudo-labels for training, and subsequently to enable on-the-fly adaptation to new experts at test-time. The experiment results on three different datasets confirm that a model trained on these synthetic labels rapidly approaches oracle-level performance, validating the data efficiency of our approach. By resolving a key training bottleneck, this work makes adaptive L2D systems more practical and scalable, paving the way for human-AI collaboration in real-world environments. To facilitate reproducibility and address implementation details not covered in the main text, we provide our source code and training configurations at this https URL. 

---
# A New Type of Adversarial Examples 

**Authors**: Xingyang Nie, Guojie Xiao, Su Pan, Biao Wang, Huilin Ge, Tao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19347)  

**Abstract**: Most machine learning models are vulnerable to adversarial examples, which poses security concerns on these models. Adversarial examples are crafted by applying subtle but intentionally worst-case modifications to examples from the dataset, leading the model to output a different answer from the original example. In this paper, adversarial examples are formed in an exactly opposite manner, which are significantly different from the original examples but result in the same answer. We propose a novel set of algorithms to produce such adversarial examples, including the negative iterative fast gradient sign method (NI-FGSM) and the negative iterative fast gradient method (NI-FGM), along with their momentum variants: the negative momentum iterative fast gradient sign method (NMI-FGSM) and the negative momentum iterative fast gradient method (NMI-FGM). Adversarial examples constructed by these methods could be used to perform an attack on machine learning systems in certain occasions. Moreover, our results show that the adversarial examples are not merely distributed in the neighbourhood of the examples from the dataset; instead, they are distributed extensively in the sample space. 

---
# Foundation Model Forecasts: Form and Function 

**Authors**: Alvaro Perez-Diaz, James C. Loach, Danielle E. Toutoungi, Lee Middleton  

**Link**: [PDF](https://arxiv.org/pdf/2510.19345)  

**Abstract**: Time-series foundation models (TSFMs) achieve strong forecast accuracy, yet accuracy alone does not determine practical value. The form of a forecast -- point, quantile, parametric, or trajectory ensemble -- fundamentally constrains which operational tasks it can support. We survey recent TSFMs and find that two-thirds produce only point or parametric forecasts, while many operational tasks require trajectory ensembles that preserve temporal dependence. We establish when forecast types can be converted and when they cannot: trajectory ensembles convert to simpler forms via marginalization without additional assumptions, but the reverse requires imposing temporal dependence through copulas or conformal methods. We prove that marginals cannot determine path-dependent event probabilities -- infinitely many joint distributions share identical marginals but yield different answers to operational questions. We map six fundamental forecasting tasks to minimal sufficient forecast types and provide a task-aligned evaluation framework. Our analysis clarifies when forecast type, not accuracy, differentiates practical utility. 

---
# To Use or to Refuse? Re-Centering Student Agency with Generative AI in Engineering Design Education 

**Authors**: Thijs Willems, Sumbul Khan, Qian Huang, Bradley Camburn, Nachamma Sockalingam, King Wang Poon  

**Link**: [PDF](https://arxiv.org/pdf/2510.19342)  

**Abstract**: This pilot study traces students' reflections on the use of AI in a 13-week foundational design course enrolling over 500 first-year engineering and architecture students at the Singapore University of Technology and Design. The course was an AI-enhanced design course, with several interventions to equip students with AI based design skills. Students were required to reflect on whether the technology was used as a tool (instrumental assistant), a teammate (collaborative partner), or neither (deliberate non-use). By foregrounding this three-way lens, students learned to use AI for innovation rather than just automation and to reflect on agency, ethics, and context rather than on prompt crafting alone. Evidence stems from coursework artefacts: thirteen structured reflection spreadsheets and eight illustrated briefs submitted, combined with notes of teachers and researchers. Qualitative coding of these materials reveals shared practices brought about through the inclusion of Gen-AI, including accelerated prototyping, rapid skill acquisition, iterative prompt refinement, purposeful "switch-offs" during user research, and emergent routines for recognizing hallucinations. Unexpectedly, students not only harnessed Gen-AI for speed but (enabled by the tool-teammate-neither triage) also learned to reject its outputs, invent their own hallucination fire-drills, and divert the reclaimed hours into deeper user research, thereby transforming efficiency into innovation. The implications of the approach we explore shows that: we can transform AI uptake into an assessable design habit; that rewarding selective non-use cultivates hallucination-aware workflows; and, practically, that a coordinated bundle of tool access, reflection, role tagging, and public recognition through competition awards allows AI based innovation in education to scale without compromising accountability. 

---
# Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning 

**Authors**: Ling Team, Bin Han, Caizhi Tang, Chen Liang, Donghao Zhang, Fan Yuan, Feng Zhu, Jie Gao, Jingyu Hu, Longfei Li, Meng Li, Mingyang Zhang, Peijie Jiang, Peng Jiao, Qian Zhao, Qingyuan Yang, Wenbo Shen, Xinxing Yang, Yalin Zhang, Yankun Ren, Yao Zhao, Yibo Cao, Yixuan Sun, Yue Zhang, Yuchen Fang, Zibin Lin, Zixuan Cheng, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.19338)  

**Abstract**: In this technical report, we present the Ring-linear model series, specifically including Ring-mini-linear-2.0 and Ring-flash-linear-2.0. Ring-mini-linear-2.0 comprises 16B parameters and 957M activations, while Ring-flash-linear-2.0 contains 104B parameters and 6.1B activations. Both models adopt a hybrid architecture that effectively integrates linear attention and softmax attention, significantly reducing I/O and computational overhead in long-context inference scenarios. Compared to a 32 billion parameter dense model, this series reduces inference cost to 1/10, and compared to the original Ring series, the cost is also reduced by over 50%. Furthermore, through systematic exploration of the ratio between different attention mechanisms in the hybrid architecture, we have identified the currently optimal model structure. Additionally, by leveraging our self-developed high-performance FP8 operator library-linghe, overall training efficiency has been improved by 50%. Benefiting from the high alignment between the training and inference engine operators, the models can undergo long-term, stable, and highly efficient optimization during the reinforcement learning phase, consistently maintaining SOTA performance across multiple challenging complex reasoning benchmarks. 

---
# Metadata Extraction Leveraging Large Language Models 

**Authors**: Cuize Han, Sesh Jalagam  

**Link**: [PDF](https://arxiv.org/pdf/2510.19334)  

**Abstract**: The advent of Large Language Models has revolutionized tasks across domains, including the automation of legal document analysis, a critical component of modern contract management systems. This paper presents a comprehensive implementation of LLM-enhanced metadata extraction for contract review, focusing on the automatic detection and annotation of salient legal clauses. Leveraging both the publicly available Contract Understanding Atticus Dataset (CUAD) and proprietary contract datasets, our work demonstrates the integration of advanced LLM methodologies with practical applications. We identify three pivotal elements for optimizing metadata extraction: robust text conversion, strategic chunk selection, and advanced LLM-specific techniques, including Chain of Thought (CoT) prompting and structured tool calling. The results from our experiments highlight the substantial improvements in clause identification accuracy and efficiency. Our approach shows promise in reducing the time and cost associated with contract review while maintaining high accuracy in legal clause identification. The results suggest that carefully optimized LLM systems could serve as valuable tools for legal professionals, potentially increasing access to efficient contract review services for organizations of all sizes. 

---
# Seabed-Net: A multi-task network for joint bathymetry estimation and seabed classification from remote sensing imagery in shallow waters 

**Authors**: Panagiotis Agrafiotis, Begüm Demir  

**Link**: [PDF](https://arxiv.org/pdf/2510.19329)  

**Abstract**: Accurate, detailed, and regularly updated bathymetry, coupled with complex semantic content, is essential for under-mapped shallow-water environments facing increasing climatological and anthropogenic pressures. However, existing approaches that derive either depth or seabed classes from remote sensing imagery treat these tasks in isolation, forfeiting the mutual benefits of their interaction and hindering the broader adoption of deep learning methods. To address these limitations, we introduce Seabed-Net, a unified multi-task framework that simultaneously predicts bathymetry and pixel-based seabed classification from remote sensing imagery of various resolutions. Seabed-Net employs dual-branch encoders for bathymetry estimation and pixel-based seabed classification, integrates cross-task features via an Attention Feature Fusion module and a windowed Swin-Transformer fusion block, and balances objectives through dynamic task uncertainty weighting. In extensive evaluations at two heterogeneous coastal sites, it consistently outperforms traditional empirical models and traditional machine learning regression methods, achieving up to 75\% lower RMSE. It also reduces bathymetric RMSE by 10-30\% compared to state-of-the-art single-task and multi-task baselines and improves seabed classification accuracy up to 8\%. Qualitative analyses further demonstrate enhanced spatial consistency, sharper habitat boundaries, and corrected depth biases in low-contrast regions. These results confirm that jointly modeling depth with both substrate and seabed habitats yields synergistic gains, offering a robust, open solution for integrated shallow-water mapping. Code and pretrained weights are available at this https URL. 

---
# SORA-ATMAS: Adaptive Trust Management and Multi-LLM Aligned Governance for Future Smart Cities 

**Authors**: Usama Antuley, Shahbaz Siddiqui, Sufian Hameed, Waqas Arif, Subhan Shah, Syed Attique Shah  

**Link**: [PDF](https://arxiv.org/pdf/2510.19327)  

**Abstract**: The rapid evolution of smart cities has increased the reliance on intelligent interconnected services to optimize infrastructure, resources, and citizen well-being. Agentic AI has emerged as a key enabler by supporting autonomous decision-making and adaptive coordination, allowing urban systems to respond in real time to dynamic conditions. Its benefits are evident in areas such as transportation, where the integration of traffic data, weather forecasts, and safety sensors enables dynamic rerouting and a faster response to hazards. However, its deployment across heterogeneous smart city ecosystems raises critical governance, risk, and compliance (GRC) challenges, including accountability, data privacy, and regulatory alignment within decentralized infrastructures. Evaluation of SORA-ATMAS with three domain agents (Weather, Traffic, and Safety) demonstrated that its governance policies, including a fallback mechanism for high-risk scenarios, effectively steer multiple LLMs (GPT, Grok, DeepSeek) towards domain-optimized, policy-aligned outputs, producing an average MAE reduction of 35% across agents. Results showed stable weather monitoring, effective handling of high-risk traffic plateaus 0.85, and adaptive trust regulation in Safety/Fire scenarios 0.65. Runtime profiling of a 3-agent deployment confirmed scalability, with throughput between 13.8-17.2 requests per second, execution times below 72~ms, and governance delays under 100 ms, analytical projections suggest maintained performance at larger scales. Cross-domain rules ensured safe interoperability, with traffic rerouting permitted only under validated weather conditions. These findings validate SORA-ATMAS as a regulation-aligned, context-aware, and verifiable governance framework that consolidates distributed agent outputs into accountable, real-time decisions, offering a resilient foundation for smart-city management. 

---
# Balancing Rewards in Text Summarization: Multi-Objective Reinforcement Learning via HyperVolume Optimization 

**Authors**: Junjie Song, Yiwen Liu, Dapeng Li, Yin Sun, Shukun Fu, Siqi Chen, Yuji Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.19325)  

**Abstract**: Text summarization is a crucial task that requires the simultaneous optimization of multiple objectives, including consistency, coherence, relevance, and fluency, which presents considerable challenges. Although large language models (LLMs) have demonstrated remarkable performance, enhanced by reinforcement learning (RL), few studies have focused on optimizing the multi-objective problem of summarization through RL based on LLMs. In this paper, we introduce hypervolume optimization (HVO), a novel optimization strategy that dynamically adjusts the scores between groups during the reward process in RL by using the hypervolume method. This method guides the model's optimization to progressively approximate the pareto front, thereby generating balanced summaries across multiple objectives. Experimental results on several representative summarization datasets demonstrate that our method outperforms group relative policy optimization (GRPO) in overall scores and shows more balanced performance across different dimensions. Moreover, a 7B foundation model enhanced by HVO performs comparably to GPT-4 in the summarization task, while maintaining a shorter generation length. Our code is publicly available at this https URL 

---
# Enabling Reconfiguration-Communication Overlap for Collective Communication in Optical Networks 

**Authors**: Changbo Wu, Zhuolong Yu, Gongming Zhao, Hongli Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19322)  

**Abstract**: Collective communication (CC) is widely adopted for large-scale distributed machine learning (DML) training workloads. DML's predictable traffic pattern provides a great oppotunity for applying optical network technology. Existing optical interconnects-based CC schemes adopt ``one-shot network reconfiguration'', which provisions static high-capacity topologies for an entire collective operation -- sometimes for a full training iteration. However, this approach faces significant scalability limitations when supporting more complex and efficient CC algorithms required for modern workloads: the ``one-shot'' strategies either demand excessive resource overprovisioning or suffer performance degradation due to rigid resource allocation.
To address these challenges, we propose SWOT, a demand-aware optical network framework. SWOT employs ``intra-collective reconfiguration'' and can dynamically align network resources with CC traffic patterns. SWOT incorporates a novel scheduling technique that overlaps optical switch reconfigurations with ongoing transmissions, and improves communication efficiency. SWOT introduce a lightweight collective communication shim that enables coordinated optical network configuration and transmission scheduling while supporting seamless integration with existing CC libraries. Our simulation results demonstrate SWOT's significant performance improvements. 

---
# Online Handwritten Signature Verification Based on Temporal-Spatial Graph Attention Transformer 

**Authors**: Hai-jie Yuan, Heng Zhang, Fei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.19321)  

**Abstract**: Handwritten signature verification is a crucial aspect of identity authentication, with applications in various domains such as finance and e-commerce. However, achieving high accuracy in signature verification remains challenging due to intra-user variability and the risk of forgery. This paper introduces a novel approach for dynamic signature verification: the Temporal-Spatial Graph Attention Transformer (TS-GATR). TS-GATR combines the Graph Attention Network (GAT) and the Gated Recurrent Unit (GRU) to model both spatial and temporal dependencies in signature data. TS-GATR enhances verification performance by representing signatures as graphs, where each node captures dynamic features (e.g. position, velocity, pressure), and by using attention mechanisms to model their complex relationships. The proposed method further employs a Dual-Graph Attention Transformer (DGATR) module, which utilizes k-step and k-nearest neighbor adjacency graphs to model local and global spatial features, respectively. To capture long-term temporal dependencies, the model integrates GRU, thereby enhancing its ability to learn dynamic features during signature verification. Comprehensive experiments conducted on benchmark datasets such as MSDS and DeepSignDB show that TS-GATR surpasses current state-of-the-art approaches, consistently achieving lower Equal Error Rates (EER) across various scenarios. 

---
# Collaborative penetration testing suite for emerging generative AI algorithms 

**Authors**: Petar Radanliev  

**Link**: [PDF](https://arxiv.org/pdf/2510.19303)  

**Abstract**: Problem Space: AI Vulnerabilities and Quantum Threats Generative AI vulnerabilities: model inversion, data poisoning, adversarial inputs. Quantum threats Shor Algorithm breaking RSA ECC encryption. Challenge Secure generative AI models against classical and quantum cyberattacks. Proposed Solution Collaborative Penetration Testing Suite Five Integrated Components: DAST SAST OWASP ZAP, Burp Suite, SonarQube, Fortify. IAST Contrast Assess integrated with CI CD pipeline. Blockchain Logging Hyperledger Fabric for tamper-proof logs. Quantum Cryptography Lattice based RLWE protocols. AI Red Team Simulations Adversarial ML & Quantum-assisted attacks. Integration Layer: Unified workflow for AI, cybersecurity, and quantum experts. Key Results 300+ vulnerabilities identified across test environments. 70% reduction in high-severity issues within 2 weeks. 90% resolution efficiency for blockchain-logged vulnerabilities. Quantum-resistant cryptography maintained 100% integrity in tests. Outcome: Quantum AI Security Protocol integrating Blockchain Quantum Cryptography AI Red Teaming. 

---
# Knowledge and Common Knowledge of Strategies 

**Authors**: Borja Sierra Miranda, Thomas Studer  

**Link**: [PDF](https://arxiv.org/pdf/2510.19298)  

**Abstract**: Most existing work on strategic reasoning simply adopts either an informed or an uninformed semantics. We propose a model where knowledge of strategies can be specified on a fine-grained level. In particular, it is possible to distinguish first-order, higher-order, and common knowledge of strategies. We illustrate the effect of higher-order knowledge of strategies by studying the game Hanabi. Further, we show that common knowledge of strategies is necessary to solve the consensus problem. Finally, we study the decidability of the model checking problem. 

---
# Enhancing Early Alzheimer Disease Detection through Big Data and Ensemble Few-Shot Learning 

**Authors**: Safa Ben Atitallah, Maha Driss, Wadii Boulila, Anis Koubaa  

**Link**: [PDF](https://arxiv.org/pdf/2510.19282)  

**Abstract**: Alzheimer disease is a severe brain disorder that causes harm in various brain areas and leads to memory damage. The limited availability of labeled medical data poses a significant challenge for accurate Alzheimer disease detection. There is a critical need for effective methods to improve the accuracy of Alzheimer disease detection, considering the scarcity of labeled data, the complexity of the disease, and the constraints related to data privacy. To address this challenge, our study leverages the power of big data in the form of pre-trained Convolutional Neural Networks (CNNs) within the framework of Few-Shot Learning (FSL) and ensemble learning. We propose an ensemble approach based on a Prototypical Network (ProtoNet), a powerful method in FSL, integrating various pre-trained CNNs as encoders. This integration enhances the richness of features extracted from medical images. Our approach also includes a combination of class-aware loss and entropy loss to ensure a more precise classification of Alzheimer disease progression levels. The effectiveness of our method was evaluated using two datasets, the Kaggle Alzheimer dataset and the ADNI dataset, achieving an accuracy of 99.72% and 99.86%, respectively. The comparison of our results with relevant state-of-the-art studies demonstrated that our approach achieved superior accuracy and highlighted its validity and potential for real-world applications in early Alzheimer disease detection. 

---
# Social World Model-Augmented Mechanism Design Policy Learning 

**Authors**: Xiaoyuan Zhang, Yizhe Huang, Chengdong Ma, Zhixun Chen, Long Ma, Yali Du, Song-Chun Zhu, Yaodong Yang, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.19270)  

**Abstract**: Designing adaptive mechanisms to align individual and collective interests remains a central challenge in artificial social intelligence. Existing methods often struggle with modeling heterogeneous agents possessing persistent latent traits (e.g., skills, preferences) and dealing with complex multi-agent system dynamics. These challenges are compounded by the critical need for high sample efficiency due to costly real-world interactions. World Models, by learning to predict environmental dynamics, offer a promising pathway to enhance mechanism design in heterogeneous and complex systems. In this paper, we introduce a novel method named SWM-AP (Social World Model-Augmented Mechanism Design Policy Learning), which learns a social world model hierarchically modeling agents' behavior to enhance mechanism design. Specifically, the social world model infers agents' traits from their interaction trajectories and learns a trait-based model to predict agents' responses to the deployed mechanisms. The mechanism design policy collects extensive training trajectories by interacting with the social world model, while concurrently inferring agents' traits online during real-world interactions to further boost policy learning efficiency. Experiments in diverse settings (tax policy design, team coordination, and facility location) demonstrate that SWM-AP outperforms established model-based and model-free RL baselines in cumulative rewards and sample efficiency. 

---
# LAPRAD: LLM-Assisted PRotocol Attack Discovery 

**Authors**: R.Can Aygun, Yehuda Afek, Anat Bremler-Barr, Leonard Kleinrock  

**Link**: [PDF](https://arxiv.org/pdf/2510.19264)  

**Abstract**: With the goal of improving the security of Internet protocols, we seek faster, semi-automatic methods to discover new vulnerabilities in protocols such as DNS, BGP, and others. To this end, we introduce the LLM-Assisted Protocol Attack Discovery (LAPRAD) methodology, enabling security researchers with some DNS knowledge to efficiently uncover vulnerabilities that would otherwise be hard to detect.
LAPRAD follows a three-stage process. In the first, we consult an LLM (GPT-o1) that has been trained on a broad corpus of DNS-related sources and previous DDoS attacks to identify potential exploits. In the second stage, a different LLM automatically constructs the corresponding attack configurations using the ReACT approach implemented via LangChain (DNS zone file generation). Finally, in the third stage, we validate the attack's functionality and effectiveness.
Using LAPRAD, we uncovered three new DDoS attacks on the DNS protocol and rediscovered two recently reported ones that were not included in the LLM's training data. The first new attack employs a bait-and-switch technique to trick resolvers into caching large, bogus DNSSEC RRSIGs, reducing their serving capacity to as little as 6%. The second exploits large DNSSEC encryption algorithms (RSA-4096) with multiple keys, thereby bypassing a recently implemented default RRSet limit. The third leverages ANY-type responses to produce a similar effect.
These variations of a cache-flushing DDoS attack, called SigCacheFlush, circumvent existing patches, severely degrade resolver query capacity, and impact the latest versions of major DNS resolver implementations. 

---
# FnRGNN: Distribution-aware Fairness in Graph Neural Network 

**Authors**: Soyoung Park, Sungsu Lim  

**Link**: [PDF](https://arxiv.org/pdf/2510.19257)  

**Abstract**: Graph Neural Networks (GNNs) excel at learning from structured data, yet fairness in regression tasks remains underexplored. Existing approaches mainly target classification and representation-level debiasing, which cannot fully address the continuous nature of node-level regression. We propose FnRGNN, a fairness-aware in-processing framework for GNN-based node regression that applies interventions at three levels: (i) structure-level edge reweighting, (ii) representation-level alignment via MMD, and (iii) prediction-level normalization through Sinkhorn-based distribution matching. This multi-level strategy ensures robust fairness under complex graph topologies. Experiments on four real-world datasets demonstrate that FnRGNN reduces group disparities without sacrificing performance. Code is available at this https URL. 

---
# See, Think, Act: Online Shopper Behavior Simulation with VLM Agents 

**Authors**: Yimeng Zhang, Jiri Gesi, Ran Xue, Tian Wang, Ziyi Wang, Yuxuan Lu, Sinong Zhan, Huimin Zeng, Qingjun Cui, Yufan Guo, Jing Huang, Mubarak Shah, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19245)  

**Abstract**: LLMs have recently demonstrated strong potential in simulating online shopper behavior. Prior work has improved action prediction by applying SFT on action traces with LLM-generated rationales, and by leveraging RL to further enhance reasoning capabilities. Despite these advances, current approaches rely on text-based inputs and overlook the essential role of visual perception in shaping human decision-making during web GUI interactions. In this paper, we investigate the integration of visual information, specifically webpage screenshots, into behavior simulation via VLMs, leveraging OPeRA dataset. By grounding agent decision-making in both textual and visual modalities, we aim to narrow the gap between synthetic agents and real-world users, thereby enabling more cognitively aligned simulations of online shopping behavior. Specifically, we employ SFT for joint action prediction and rationale generation, conditioning on the full interaction context, which comprises action history, past HTML observations, and the current webpage screenshot. To further enhance reasoning capabilities, we integrate RL with a hierarchical reward structure, scaled by a difficulty-aware factor that prioritizes challenging decision points. Empirically, our studies show that incorporating visual grounding yields substantial gains: the combination of text and image inputs improves exact match accuracy by more than 6% over text-only inputs. These results indicate that multi-modal grounding not only boosts predictive accuracy but also enhances simulation fidelity in visually complex environments, which captures nuances of human attention and decision-making that text-only agents often miss. Finally, we revisit the design space of behavior simulation frameworks, identify key methodological limitations, and propose future research directions toward building efficient and effective human behavior simulators. 

---
# SPOT: Scalable Policy Optimization with Trees for Markov Decision Processes 

**Authors**: Xuyuan Xiong, Pedro Chumpitaz-Flores, Kaixun Hua, Cheng Hua  

**Link**: [PDF](https://arxiv.org/pdf/2510.19241)  

**Abstract**: Interpretable reinforcement learning policies are essential for high-stakes decision-making, yet optimizing decision tree policies in Markov Decision Processes (MDPs) remains challenging. We propose SPOT, a novel method for computing decision tree policies, which formulates the optimization problem as a mixed-integer linear program (MILP). To enhance efficiency, we employ a reduced-space branch-and-bound approach that decouples the MDP dynamics from tree-structure constraints, enabling efficient parallel search. This significantly improves runtime and scalability compared to previous methods. Our approach ensures that each iteration yields the optimal decision tree. Experimental results on standard benchmarks demonstrate that SPOT achieves substantial speedup and scales to larger MDPs with a significantly higher number of states. The resulting decision tree policies are interpretable and compact, maintaining transparency without compromising performance. These results demonstrate that our approach simultaneously achieves interpretability and scalability, delivering high-quality policies an order of magnitude faster than existing approaches. 

---
# No Intelligence Without Statistics: The Invisible Backbone of Artificial Intelligence 

**Authors**: Ernest Fokoué  

**Link**: [PDF](https://arxiv.org/pdf/2510.19212)  

**Abstract**: The rapid ascent of artificial intelligence (AI) is often portrayed as a revolution born from computer science and engineering. This narrative, however, obscures a fundamental truth: the theoretical and methodological core of AI is, and has always been, statistical. This paper systematically argues that the field of statistics provides the indispensable foundation for machine learning and modern AI. We deconstruct AI into nine foundational pillars-Inference, Density Estimation, Sequential Learning, Generalization, Representation Learning, Interpretability, Causality, Optimization, and Unification-demonstrating that each is built upon century-old statistical principles. From the inferential frameworks of hypothesis testing and estimation that underpin model evaluation, to the density estimation roots of clustering and generative AI; from the time-series analysis inspiring recurrent networks to the causal models that promise true understanding, we trace an unbroken statistical lineage. While celebrating the computational engines that power modern AI, we contend that statistics provides the brain-the theoretical frameworks, uncertainty quantification, and inferential goals-while computer science provides the brawn-the scalable algorithms and hardware. Recognizing this statistical backbone is not merely an academic exercise, but a necessary step for developing more robust, interpretable, and trustworthy intelligent systems. We issue a call to action for education, research, and practice to re-embrace this statistical foundation. Ignoring these roots risks building a fragile future; embracing them is the path to truly intelligent machines. There is no machine learning without statistical learning; no artificial intelligence without statistical thought. 

---
# An Active Diffusion Neural Network for Graphs 

**Authors**: Mengying Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19202)  

**Abstract**: The analogy to heat diffusion has enhanced our understanding of information flow in graphs and inspired the development of Graph Neural Networks (GNNs). However, most diffusion-based GNNs emulate passive heat diffusion, which still suffers from over-smoothing and limits their ability to capture global graph information. Inspired by the heat death of the universe, which posits that energy distribution becomes uniform over time in a closed system, we recognize that, without external input, node representations in a graph converge to identical feature vectors as diffusion progresses. To address this issue, we propose the Active Diffusion-based Graph Neural Network (ADGNN). ADGNN achieves active diffusion by integrating multiple external information sources that dynamically influence the diffusion process, effectively overcoming the over-smoothing problem. Furthermore, our approach realizes true infinite diffusion by directly calculating the closed-form solution of the active diffusion iterative formula. This allows nodes to preserve their unique characteristics while efficiently gaining comprehensive insights into the graph's global structure. We evaluate ADGNN against several state-of-the-art GNN models across various graph tasks. The results demonstrate that ADGNN significantly improves both accuracy and efficiency, highlighting its effectiveness in capturing global graph information and maintaining node distinctiveness. 

---
# Rethinking Driving World Model as Synthetic Data Generator for Perception Tasks 

**Authors**: Kai Zeng, Zhanqian Wu, Kaixin Xiong, Xiaobao Wei, Xiangyu Guo, Zhenxin Zhu, Kalok Ho, Lijun Zhou, Bohan Zeng, Ming Lu, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.19195)  

**Abstract**: Recent advancements in driving world models enable controllable generation of high-quality RGB videos or multimodal videos. Existing methods primarily focus on metrics related to generation quality and controllability. However, they often overlook the evaluation of downstream perception tasks, which are $\mathbf{really\ crucial}$ for the performance of autonomous driving. Existing methods usually leverage a training strategy that first pretrains on synthetic data and finetunes on real data, resulting in twice the epochs compared to the baseline (real data only). When we double the epochs in the baseline, the benefit of synthetic data becomes negligible. To thoroughly demonstrate the benefit of synthetic data, we introduce Dream4Drive, a novel synthetic data generation framework designed for enhancing the downstream perception tasks. Dream4Drive first decomposes the input video into several 3D-aware guidance maps and subsequently renders the 3D assets onto these guidance maps. Finally, the driving world model is fine-tuned to produce the edited, multi-view photorealistic videos, which can be used to train the downstream perception models. Dream4Drive enables unprecedented flexibility in generating multi-view corner cases at scale, significantly boosting corner case perception in autonomous driving. To facilitate future research, we also contribute a large-scale 3D asset dataset named DriveObj3D, covering the typical categories in driving scenarios and enabling diverse 3D-aware video editing. We conduct comprehensive experiments to show that Dream4Drive can effectively boost the performance of downstream perception models under various training epochs. Project: $\href{this https URL}{this\ https\ URL}$ 

---
# PruneHal: Reducing Hallucinations in Multi-modal Large Language Models through Adaptive KV Cache Pruning 

**Authors**: Fengyuan Sun, Hui Chen, Xinhao Xu, Dandan Zheng, Jingdong Chen, Jun Zhou, Jungong Han, Guiguang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.19183)  

**Abstract**: While multi-modal large language models (MLLMs) have made significant progress in recent years, the issue of hallucinations remains a major challenge. To mitigate this phenomenon, existing solutions either introduce additional data for further training or incorporate external or internal information during inference. However, these approaches inevitably introduce extra computational costs. In this paper, we observe that hallucinations in MLLMs are strongly associated with insufficient attention allocated to visual tokens. In particular, the presence of redundant visual tokens disperses the model's attention, preventing it from focusing on the most informative ones. As a result, critical visual cues are often under-attended, which in turn exacerbates the occurrence of hallucinations. Building on this observation, we propose \textbf{PruneHal}, a training-free, simple yet effective method that leverages adaptive KV cache pruning to enhance the model's focus on critical visual information, thereby mitigating hallucinations. To the best of our knowledge, we are the first to apply token pruning for hallucination mitigation in MLLMs. Notably, our method don't require additional training and incurs nearly no extra inference cost. Moreover, PruneHal is model-agnostic and can be seamlessly integrated with different decoding strategies, including those specifically designed for hallucination mitigation. We evaluate PruneHal on several widely used hallucination evaluation benchmarks using four mainstream MLLMs, achieving robust and outstanding results that highlight the effectiveness and superiority of our method. Our code will be publicly available. 

---
# Interpretable Question Answering with Knowledge Graphs 

**Authors**: Kartikeya Aneja, Manasvi Srivastava, Subhayan Das, Nagender Aneja  

**Link**: [PDF](https://arxiv.org/pdf/2510.19181)  

**Abstract**: This paper presents a question answering system that operates exclusively on a knowledge graph retrieval without relying on retrieval augmented generation (RAG) with large language models (LLMs). Instead, a small paraphraser model is used to paraphrase the entity relationship edges retrieved from querying the knowledge graph. The proposed pipeline is divided into two main stages. The first stage involves pre-processing a document to generate sets of question-answer (QA) pairs. The second stage converts these QAs into a knowledge graph from which graph-based retrieval is performed using embeddings and fuzzy techniques. The graph is queried, re-ranked, and paraphrased to generate a final answer. This work includes an evaluation using LLM-as-a-judge on the CRAG benchmark, which resulted in accuracies of 71.9% and 54.4% using LLAMA-3.2 and GPT-3.5-Turbo, respectively. 

---
# Imbalanced Gradients in RL Post-Training of Multi-Task LLMs 

**Authors**: Runzhe Wu, Ankur Samanta, Ayush Jain, Scott Fujimoto, Jeongyeol Kwon, Ben Kretzu, Youliang Yu, Kaveh Hassani, Boris Vidolov, Yonathan Efroni  

**Link**: [PDF](https://arxiv.org/pdf/2510.19178)  

**Abstract**: Multi-task post-training of large language models (LLMs) is typically performed by mixing datasets from different tasks and optimizing them jointly. This approach implicitly assumes that all tasks contribute gradients of similar magnitudes; when this assumption fails, optimization becomes biased toward large-gradient tasks. In this paper, however, we show that this assumption fails in RL post-training: certain tasks produce significantly larger gradients, thus biasing updates toward those tasks. Such gradient imbalance would be justified only if larger gradients implied larger learning gains on the tasks (i.e., larger performance improvements) -- but we find this is not true. Large-gradient tasks can achieve similar or even much lower learning gains than small-gradient ones. Further analyses reveal that these gradient imbalances cannot be explained by typical training statistics such as training rewards or advantages, suggesting that they arise from the inherent differences between tasks. This cautions against naive dataset mixing and calls for future work on principled gradient-level corrections for LLMs. 

---
# News-Aware Direct Reinforcement Trading for Financial Markets 

**Authors**: Qing-Yu Lan, Zhan-He Wang, Jun-Qian Jiang, Yu-Tong Wang, Yun-Song Piao  

**Link**: [PDF](https://arxiv.org/pdf/2510.19173)  

**Abstract**: The financial market is known to be highly sensitive to news. Therefore, effectively incorporating news data into quantitative trading remains an important challenge. Existing approaches typically rely on manually designed rules and/or handcrafted features. In this work, we directly use the news sentiment scores derived from large language models, together with raw price and volume data, as observable inputs for reinforcement learning. These inputs are processed by sequence models such as recurrent neural networks or Transformers to make end-to-end trading decisions. We conduct experiments using the cryptocurrency market as an example and evaluate two representative reinforcement learning algorithms, namely Double Deep Q-Network (DDQN) and Group Relative Policy Optimization (GRPO). The results demonstrate that our news-aware approach, which does not depend on handcrafted features or manually designed rules, can achieve performance superior to market benchmarks. We further highlight the critical role of time-series information in this process. 

---
# When Facts Change: Probing LLMs on Evolving Knowledge with evolveQA 

**Authors**: Nishanth Sridhar Nakshatri, Shamik Roy, Manoj Ghuhan Arivazhagan, Hanhan Zhou, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2510.19172)  

**Abstract**: LLMs often fail to handle temporal knowledge conflicts--contradictions arising when facts evolve over time within their training data. Existing studies evaluate this phenomenon through benchmarks built on structured knowledge bases like Wikidata, but they focus on widely-covered, easily-memorized popular entities and lack the dynamic structure needed to fairly evaluate LLMs with different knowledge cut-off dates. We introduce evolveQA, a benchmark specifically designed to evaluate LLMs on temporally evolving knowledge, constructed from 3 real-world, time-stamped corpora: AWS updates, Azure changes, and WHO disease outbreak reports. Our framework identifies naturally occurring knowledge evolution and generates questions with gold answers tailored to different LLM knowledge cut-off dates. Through extensive evaluation of 12 open and closed-source LLMs across 3 knowledge probing formats, we demonstrate significant performance drops of up to 31% on evolveQA compared to static knowledge questions. 

---
# X-Ego: Acquiring Team-Level Tactical Situational Awareness via Cross-Egocentric Contrastive Video Representation Learning 

**Authors**: Yunzhe Wang, Soham Hans, Volkan Ustun  

**Link**: [PDF](https://arxiv.org/pdf/2510.19150)  

**Abstract**: Human team tactics emerge from each player's individual perspective and their ability to anticipate, interpret, and adapt to teammates' intentions. While advances in video understanding have improved the modeling of team interactions in sports, most existing work relies on third-person broadcast views and overlooks the synchronous, egocentric nature of multi-agent learning. We introduce X-Ego-CS, a benchmark dataset consisting of 124 hours of gameplay footage from 45 professional-level matches of the popular e-sports game Counter-Strike 2, designed to facilitate research on multi-agent decision-making in complex 3D environments. X-Ego-CS provides cross-egocentric video streams that synchronously capture all players' first-person perspectives along with state-action trajectories. Building on this resource, we propose Cross-Ego Contrastive Learning (CECL), which aligns teammates' egocentric visual streams to foster team-level tactical situational awareness from an individual's perspective. We evaluate CECL on a teammate-opponent location prediction task, demonstrating its effectiveness in enhancing an agent's ability to infer both teammate and opponent positions from a single first-person view using state-of-the-art video encoders. Together, X-Ego-CS and CECL establish a foundation for cross-egocentric multi-agent benchmarking in esports. More broadly, our work positions gameplay understanding as a testbed for multi-agent modeling and tactical learning, with implications for spatiotemporal reasoning and human-AI teaming in both virtual and real-world domains. Code and dataset are available at this https URL. 

---
# InvarGC: Invariant Granger Causality for Heterogeneous Interventional Time Series under Latent Confounding 

**Authors**: Ziyi Zhang, Shaogang Ren, Xiaoning Qian, Nick Duffield  

**Link**: [PDF](https://arxiv.org/pdf/2510.19138)  

**Abstract**: Granger causality is widely used for causal structure discovery in complex systems from multivariate time series data. Traditional Granger causality tests based on linear models often fail to detect even mild non-linear causal relationships. Therefore, numerous recent studies have investigated non-linear Granger causality methods, achieving improved performance. However, these methods often rely on two key assumptions: causal sufficiency and known interventional targets. Causal sufficiency assumes the absence of latent confounders, yet their presence can introduce spurious correlations. Moreover, real-world time series data usually come from heterogeneous environments, without prior knowledge of interventions. Therefore, in practice, it is difficult to distinguish intervened environments from non-intervened ones, and even harder to identify which variables or timesteps are affected. To address these challenges, we propose Invariant Granger Causality (InvarGC), which leverages cross-environment heterogeneity to mitigate the effects of latent confounding and to distinguish intervened from non-intervened environments with edge-level granularity, thereby recovering invariant causal relations. In addition, we establish the identifiability under these conditions. Extensive experiments on both synthetic and real-world datasets demonstrate the competitive performance of our approach compared to state-of-the-art methods. 

---
# A Cross-Environment and Cross-Embodiment Path Planning Framework via a Conditional Diffusion Model 

**Authors**: Mehran Ghafarian Tamizi, Homayoun Honari, Amir Mehdi Soufi Enayati, Aleksey Nozdryn-Plotnicki, Homayoun Najjaran  

**Link**: [PDF](https://arxiv.org/pdf/2510.19128)  

**Abstract**: Path planning for a robotic system in high-dimensional cluttered environments needs to be efficient, safe, and adaptable for different environments and hardware. Conventional methods face high computation time and require extensive parameter tuning, while prior learning-based methods still fail to generalize effectively. The primary goal of this research is to develop a path planning framework capable of generalizing to unseen environments and new robotic manipulators without the need for retraining. We present GADGET (Generalizable and Adaptive Diffusion-Guided Environment-aware Trajectory generation), a diffusion-based planning model that generates joint-space trajectories conditioned on voxelized scene representations as well as start and goal configurations. A key innovation is GADGET's hybrid dual-conditioning mechanism that combines classifier-free guidance via learned scene encoding with classifier-guided Control Barrier Function (CBF) safety shaping, integrating environment awareness with real-time collision avoidance directly in the denoising process. This design supports zero-shot transfer to new environments and robotic embodiments without retraining. Experimental results show that GADGET achieves high success rates with low collision intensity in spherical-obstacle, bin-picking, and shelf environments, with CBF guidance further improving safety. Moreover, comparative evaluations indicate strong performance relative to both sampling-based and learning-based baselines. Furthermore, GADGET provides transferability across Franka Panda, Kinova Gen3 (6/7-DoF), and UR5 robots, and physical execution on a Kinova Gen3 demonstrates its ability to generate safe, collision-free trajectories in real-world settings. 

---
# Steering Autoregressive Music Generation with Recursive Feature Machines 

**Authors**: Daniel Zhao, Daniel Beaglehole, Taylor Berg-Kirkpatrick, Julian McAuley, Zachary Novack  

**Link**: [PDF](https://arxiv.org/pdf/2510.19127)  

**Abstract**: Controllable music generation remains a significant challenge, with existing methods often requiring model retraining or introducing audible artifacts. We introduce MusicRFM, a framework that adapts Recursive Feature Machines (RFMs) to enable fine-grained, interpretable control over frozen, pre-trained music models by directly steering their internal activations. RFMs analyze a model's internal gradients to produce interpretable "concept directions", or specific axes in the activation space that correspond to musical attributes like notes or chords. We first train lightweight RFM probes to discover these directions within MusicGen's hidden states; then, during inference, we inject them back into the model to guide the generation process in real-time without per-step optimization. We present advanced mechanisms for this control, including dynamic, time-varying schedules and methods for the simultaneous enforcement of multiple musical properties. Our method successfully navigates the trade-off between control and generation quality: we can increase the accuracy of generating a target musical note from 0.23 to 0.82, while text prompt adherence remains within approximately 0.02 of the unsteered baseline, demonstrating effective control with minimal impact on prompt fidelity. We release code to encourage further exploration on RFMs in the music domain. 

---
# A Novel Approach to Breast Cancer Segmentation using U-Net Model with Attention Mechanisms and FedProx 

**Authors**: Eyad Gad, Mustafa Abou Khatwa, Mustafa A. Elattar, Sahar Selim  

**Link**: [PDF](https://arxiv.org/pdf/2510.19118)  

**Abstract**: Breast cancer is a leading cause of death among women worldwide, emphasizing the need for early detection and accurate diagnosis. As such Ultrasound Imaging, a reliable and cost-effective tool, is used for this purpose, however the sensitive nature of medical data makes it challenging to develop accurate and private artificial intelligence models. A solution is Federated Learning as it is a promising technique for distributed machine learning on sensitive medical data while preserving patient privacy. However, training on non-Independent and non-Identically Distributed (non-IID) local datasets can impact the accuracy and generalization of the trained model, which is crucial for accurate tumour boundary delineation in BC segmentation. This study aims to tackle this challenge by applying the Federated Proximal (FedProx) method to non-IID Ultrasonic Breast Cancer Imaging datasets. Moreover, we focus on enhancing tumour segmentation accuracy by incorporating a modified U-Net model with attention mechanisms. Our approach resulted in a global model with 96% accuracy, demonstrating the effectiveness of our method in enhancing tumour segmentation accuracy while preserving patient privacy. Our findings suggest that FedProx has the potential to be a promising approach for training precise machine learning models on non-IID local medical datasets. 

---
# That's Deprecated! Understanding, Detecting, and Steering Knowledge Conflicts in Language Models for Code Generation 

**Authors**: Jaesung Bae, Cameron Churchwell, Mitchell Hermon, Tsun-An Hsieh, Jocelyn Xu, Yekaterina Yegorova, Mark Hasegawa-Johnson, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.19116)  

**Abstract**: This paper investigates how large language models (LLMs) behave when faced with discrepancies between their parametric knowledge and conflicting information contained in a prompt. Building on prior question-answering (QA) research, we extend the investigation of knowledge conflicts to the realm of code generation. We propose a domain-agnostic framework for constructing and interpreting such conflicts, along with a novel evaluation method and dataset tailored to code conflict scenarios. Our experiments indicate that sufficiently large LLMs encode the notion of a knowledge conflict in their parameters, enabling us to detect knowledge conflicts with up to \textbf{80.65\%} accuracy. Building on these insights, we show that activation-level steering can achieve up to a \textbf{12.6\%} improvement in steering success over a random baseline. However, effectiveness depends critically on balancing model size, task domain, and steering direction. The experiment code and data will be made publicly available after acceptance. 

---
# What Makes a Good Curriculum? Disentangling the Effects of Data Ordering on LLM Mathematical Reasoning 

**Authors**: Yaning Jia, Chunhui Zhang, Xingjian Diao, Xiangchi Yuan, Zhongyu Ouyang, soroush vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19099)  

**Abstract**: Curriculum learning (CL) - ordering training data from easy to hard - has become a popular strategy for improving reasoning in large language models (LLMs). Yet prior work employs disparate difficulty metrics and training setups, leaving open fundamental questions: When does curriculum help? Which direction - forward or reverse - is better? And does the answer depend on what we measure? We address these questions through a unified offline evaluation framework that decomposes curriculum difficulty into five complementary dimensions: Problem Difficulty, Model Surprisal, Confidence Margin, Predictive Uncertainty, and Decision Variability. Through controlled post-training experiments on mathematical reasoning benchmarks with Llama3.1-8B, Mistral-7B, and Gemma3-4B, we find that (i) no curriculum strategy dominates universally - the relative effectiveness of forward versus reverse CL depends jointly on model capability and task complexity; (ii) even within a single metric, samples at different difficulty levels produce distinct gains depending on task demands; and (iii) task-aligned curricula focus on shaping the model's final representations and generalization, whereas inner-state curricula modulate internal states such as confidence and uncertainty. Our findings challenge the notion of a universal curriculum strategy and offer actionable guidance across model and task regimes, with some metrics indicating that prioritizing decision-uncertain samples can further enhance learning outcomes. 

---
# Local Guidance for Configuration-Based Multi-Agent Pathfinding 

**Authors**: Tomoki Arita, Keisuke Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2510.19072)  

**Abstract**: Guidance is an emerging concept that improves the empirical performance of real-time, sub-optimal multi-agent pathfinding (MAPF) methods. It offers additional information to MAPF algorithms to mitigate congestion on a global scale by considering the collective behavior of all agents across the entire workspace. This global perspective helps reduce agents' waiting times, thereby improving overall coordination efficiency. In contrast, this study explores an alternative approach: providing local guidance in the vicinity of each agent. While such localized methods involve recomputation as agents move and may appear computationally demanding, we empirically demonstrate that supplying informative spatiotemporal cues to the planner can significantly improve solution quality without exceeding a moderate time budget. When applied to LaCAM, a leading configuration-based solver, this form of guidance establishes a new performance frontier for MAPF. 

---
# PoSh: Using Scene Graphs To Guide LLMs-as-a-Judge For Detailed Image Descriptions 

**Authors**: Amith Ananthram, Elias Stengel-Eskin, Lorena A. Bradford, Julia Demarest, Adam Purvis, Keith Krut, Robert Stein, Rina Elster Pantalony, Mohit Bansal, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2510.19060)  

**Abstract**: While vision-language models (VLMs) have advanced into detailed image description, evaluation remains a challenge. Standard metrics (e.g. CIDEr, SPICE) were designed for short texts and tuned to recognize errors that are now uncommon, such as object misidentification. In contrast, long texts require sensitivity to attribute and relation attachments and scores that localize errors to particular text spans. In this work, we introduce PoSh, a metric for detailed image description that uses scene graphs as structured rubrics to guide LLMs-as-a-Judge, producing aggregate scores grounded in fine-grained errors (e.g. mistakes in compositional understanding). PoSh is replicable, interpretable and a better proxy for human raters than existing metrics (including GPT4o-as-a-Judge). To validate PoSh, we introduce a challenging new dataset, DOCENT. This novel benchmark contains artwork, paired with expert-written references, and model-generated descriptions, augmented with granular and coarse judgments of their quality from art history students. Thus, DOCENT enables evaluating both detailed image description metrics and detailed image description itself in a challenging new domain. We show that PoSh achieves stronger correlations (+0.05 Spearman $\rho$) with the human judgments in DOCENT than the best open-weight alternatives, is robust to image type (using CapArena, an existing dataset of web imagery) and is a capable reward function, outperforming standard supervised fine-tuning. Then, using PoSh, we characterize the performance of open and closed models in describing the paintings, sketches and statues in DOCENT and find that foundation models struggle to achieve full, error-free coverage of images with rich scene dynamics, establishing a demanding new task to gauge VLM progress. Through both PoSh and DOCENT, we hope to enable advances in important areas such as assistive text generation. 

---
# REPAIR Approach for Social-based City Reconstruction Planning in case of natural disasters 

**Authors**: Ghulam Mudassir, Antinisca Di Marco, Giordano d'Aloisio  

**Link**: [PDF](https://arxiv.org/pdf/2510.19048)  

**Abstract**: Natural disasters always have several effects on human lives. It is challenging for governments to tackle these incidents and to rebuild the economic, social and physical infrastructures and facilities with the available resources (mainly budget and time). Governments always define plans and policies according to the law and political strategies that should maximise social benefits. The severity of damage and the vast resources needed to bring life back to normality make such reconstruction a challenge. This article is the extension of our previously published work by conducting comprehensive comparative analysis by integrating additional deep learning models plus random agent which is used as a baseline. Our prior research introduced a decision support system by using the Deep Reinforcement Learning technique for the planning of post-disaster city reconstruction, maximizing the social benefit of the reconstruction process, considering available resources, meeting the needs of the broad community stakeholders (like citizens' social benefits and politicians' priorities) and keeping in consideration city's structural constraints (like dependencies among roads and buildings). The proposed approach, named post disaster REbuilding plAn ProvIdeR (REPAIR) is generic. It can determine a set of alternative plans for local administrators who select the ideal one to implement, and it can be applied to areas of any extension. We show the application of REPAIR in a real use case, i.e., to the L'Aquila reconstruction process, damaged in 2009 by a major earthquake. 

---
# "Over-the-Hood" AI Inclusivity Bugs and How 3 AI Product Teams Found and Fixed Them 

**Authors**: Andrew Anderson, Fatima A. Moussaoui, Jimena Noa Guevara, Md Montaser Hamid, Margaret Burnett  

**Link**: [PDF](https://arxiv.org/pdf/2510.19033)  

**Abstract**: While much research has shown the presence of AI's "under-the-hood" biases (e.g., algorithmic, training data, etc.), what about "over-the-hood" inclusivity biases: barriers in user-facing AI products that disproportionately exclude users with certain problem-solving approaches? Recent research has begun to report the existence of such biases -- but what do they look like, how prevalent are they, and how can developers find and fix them? To find out, we conducted a field study with 3 AI product teams, to investigate what kinds of AI inclusivity bugs exist uniquely in user-facing AI products, and whether/how AI product teams might harness an existing (non-AI-oriented) inclusive design method to find and fix them. The teams' work resulted in identifying 6 types of AI inclusivity bugs arising 83 times, fixes covering 47 of these bug instances, and a new variation of the GenderMag inclusive design method, GenderMag-for-AI, that is especially effective at detecting certain kinds of AI inclusivity bugs. 

---
# CLiVR: Conversational Learning System in Virtual Reality with AI-Powered Patients 

**Authors**: Akilan Amithasagaran, Sagnik Dakshit, Bhavani Suryadevara, Lindsey Stockton  

**Link**: [PDF](https://arxiv.org/pdf/2510.19031)  

**Abstract**: Simulations constitute a fundamental component of medical and nursing education and traditionally employ standardized patients (SP) and high-fidelity manikins to develop clinical reasoning and communication skills. However, these methods require substantial resources, limiting accessibility and scalability. In this study, we introduce CLiVR, a Conversational Learning system in Virtual Reality that integrates large language models (LLMs), speech processing, and 3D avatars to simulate realistic doctor-patient interactions. Developed in Unity and deployed on the Meta Quest 3 platform, CLiVR enables trainees to engage in natural dialogue with virtual patients. Each simulation is dynamically generated from a syndrome-symptom database and enhanced with sentiment analysis to provide feedback on communication tone. Through an expert user study involving medical school faculty (n=13), we assessed usability, realism, and perceived educational impact. Results demonstrated strong user acceptance, high confidence in educational potential, and valuable feedback for improvement. CLiVR offers a scalable, immersive supplement to SP-based training. 

---
# FlexiDataGen: An Adaptive LLM Framework for Dynamic Semantic Dataset Generation in Sensitive Domains 

**Authors**: Hamed Jelodar, Samita Bai, Roozbeh Razavi-Far, Ali A. Ghorbani  

**Link**: [PDF](https://arxiv.org/pdf/2510.19025)  

**Abstract**: Dataset availability and quality remain critical challenges in machine learning, especially in domains where data are scarce, expensive to acquire, or constrained by privacy regulations. Fields such as healthcare, biomedical research, and cybersecurity frequently encounter high data acquisition costs, limited access to annotated data, and the rarity or sensitivity of key events. These issues-collectively referred to as the dataset challenge-hinder the development of accurate and generalizable machine learning models in such high-stakes domains. To address this, we introduce FlexiDataGen, an adaptive large language model (LLM) framework designed for dynamic semantic dataset generation in sensitive domains. FlexiDataGen autonomously synthesizes rich, semantically coherent, and linguistically diverse datasets tailored to specialized fields. The framework integrates four core components: (1) syntactic-semantic analysis, (2) retrieval-augmented generation, (3) dynamic element injection, and (4) iterative paraphrasing with semantic validation. Together, these components ensure the generation of high-quality, domain-relevant data. Experimental results show that FlexiDataGen effectively alleviates data shortages and annotation bottlenecks, enabling scalable and accurate machine learning model development. 

---
# Prior-informed optimization of treatment recommendation via bandit algorithms trained on large language model-processed historical records 

**Authors**: Saman Nessari, Ali Bozorgi-Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.19014)  

**Abstract**: Current medical practice depends on standardized treatment frameworks and empirical methodologies that neglect individual patient variations, leading to suboptimal health outcomes. We develop a comprehensive system integrating Large Language Models (LLMs), Conditional Tabular Generative Adversarial Networks (CTGAN), T-learner counterfactual models, and contextual bandit approaches to provide customized, data-informed clinical recommendations. The approach utilizes LLMs to process unstructured medical narratives into structured datasets (93.2% accuracy), uses CTGANs to produce realistic synthetic patient data (55% accuracy via two-sample verification), deploys T-learners to forecast patient-specific treatment responses (84.3% accuracy), and integrates prior-informed contextual bandits to enhance online therapeutic selection by effectively balancing exploration of new possibilities with exploitation of existing knowledge. Testing on stage III colon cancer datasets revealed that our KernelUCB approach obtained 0.60-0.61 average reward scores across 5,000 rounds, exceeding other reference methods. This comprehensive system overcomes cold-start limitations in online learning environments, improves computational effectiveness, and constitutes notable progress toward individualized medicine adapted to specific patient characteristics. 

---
# Plural Voices, Single Agent: Towards Inclusive AI in Multi-User Domestic Spaces 

**Authors**: Joydeep Chandra, Satyam Kumar Navneet  

**Link**: [PDF](https://arxiv.org/pdf/2510.19008)  

**Abstract**: Domestic AI agents faces ethical, autonomy, and inclusion challenges, particularly for overlooked groups like children, elderly, and Neurodivergent users. We present the Plural Voices Model (PVM), a novel single-agent framework that dynamically negotiates multi-user needs through real-time value alignment, leveraging diverse public datasets on mental health, eldercare, education, and moral reasoning. Using human+synthetic curriculum design with fairness-aware scenarios and ethical enhancements, PVM identifies core values, conflicts, and accessibility requirements to inform inclusive principles. Our privacy-focused prototype features adaptive safety scaffolds, tailored interactions (e.g., step-by-step guidance for Neurodivergent users, simple wording for children), and equitable conflict resolution. In preliminary evaluations, PVM outperforms multi-agent baselines in compliance (76% vs. 70%), fairness (90% vs. 85%), safety-violation rate (0% vs. 7%), and latency. Design innovations, including video guidance, autonomy sliders, family hubs, and adaptive safety dashboards, demonstrate new directions for ethical and inclusive domestic AI, for building user-centered agentic systems in plural domestic contexts. Our Codes and Model are been open sourced, available for reproduction: this https URL 

---
# $Δ$t-Mamba3D: A Time-Aware Spatio-Temporal State-Space Model for Breast Cancer Risk Prediction 

**Authors**: Zhengbo Zhou, Dooman Arefan, Margarita Zuley, Shandong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.19003)  

**Abstract**: Longitudinal analysis of sequential radiological images is hampered by a fundamental data challenge: how to effectively model a sequence of high-resolution images captured at irregular time intervals. This data structure contains indispensable spatial and temporal cues that current methods fail to fully exploit. Models often compromise by either collapsing spatial information into vectors or applying spatio-temporal models that are computationally inefficient and incompatible with non-uniform time steps. We address this challenge with Time-Aware $\Delta$t-Mamba3D, a novel state-space architecture adapted for longitudinal medical imaging. Our model simultaneously encodes irregular inter-visit intervals and rich spatio-temporal context while remaining computationally efficient. Its core innovation is a continuous-time selective scanning mechanism that explicitly integrates the true time difference between exams into its state transitions. This is complemented by a multi-scale 3D neighborhood fusion module that robustly captures spatio-temporal relationships. In a comprehensive breast cancer risk prediction benchmark using sequential screening mammogram exams, our model shows superior performance, improving the validation c-index by 2-5 percentage points and achieving higher 1-5 year AUC scores compared to established variants of recurrent, transformer, and state-space models. Thanks to its linear complexity, the model can efficiently process long and complex patient screening histories of mammograms, forming a new framework for longitudinal image analysis. 

---
# Robust Driving QA through Metadata-Grounded Context and Task-Specific Prompts 

**Authors**: Seungjun Yu, Junsung Park, Youngsun Lim, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2510.19001)  

**Abstract**: We present a two-phase vision-language QA system for autonomous driving that answers high-level perception, prediction, and planning questions. In Phase-1, a large multimodal LLM (Qwen2.5-VL-32B) is conditioned on six-camera inputs, a short temporal window of history, and a chain-of-thought prompt with few-shot exemplars. A self-consistency ensemble (multiple sampled reasoning chains) further improves answer reliability. In Phase-2, we augment the prompt with nuScenes scene metadata (object annotations, ego-vehicle state, etc.) and category-specific question instructions (separate prompts for perception, prediction, planning tasks). In experiments on a driving QA benchmark, our approach significantly outperforms the baseline Qwen2.5 models. For example, using 5 history frames and 10-shot prompting in Phase-1 yields 65.1% overall accuracy (vs.62.61% with zero-shot); applying self-consistency raises this to 66.85%. Phase-2 achieves 67.37% overall. Notably, the system maintains 96% accuracy under severe visual corruption. These results demonstrate that carefully engineered prompts and contextual grounding can greatly enhance high-level driving QA with pretrained vision-language models. 

---
# $\nabla$-SDF: Learning Euclidean Signed Distance Functions Online with Gradient-Augmented Octree Interpolation and Neural Residual 

**Authors**: Zhirui Dai, Qihao Qian, Tianxing Fan, Nikolay Atanasov  

**Link**: [PDF](https://arxiv.org/pdf/2510.18999)  

**Abstract**: Estimation of signed distance functions (SDFs) from point cloud data has been shown to benefit many robot autonomy capabilities, including localization, mapping, motion planning, and control. Methods that support online and large-scale SDF reconstruction tend to rely on discrete volumetric data structures, which affect the continuity and differentiability of the SDF estimates. Recently, using implicit features, neural network methods have demonstrated high-fidelity and differentiable SDF reconstruction but they tend to be less efficient, can experience catastrophic forgetting and memory limitations in large environments, and are often restricted to truncated SDFs. This work proposes $\nabla$-SDF, a hybrid method that combines an explicit prior obtained from gradient-augmented octree interpolation with an implicit neural residual. Our method achieves non-truncated (Euclidean) SDF reconstruction with computational and memory efficiency comparable to volumetric methods and differentiability and accuracy comparable to neural network methods. Extensive experiments demonstrate that \methodname{} outperforms the state of the art in terms of accuracy and efficiency, providing a scalable solution for downstream tasks in robotics and computer vision. 

---
# ProfBench: Multi-Domain Rubrics requiring Professional Knowledge to Answer and Judge 

**Authors**: Zhilin Wang, Jaehun Jung, Ximing Lu, Shizhe Diao, Ellie Evans, Jiaqi Zeng, Pavlo Molchanov, Yejin Choi, Jan Kautz, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.18941)  

**Abstract**: Evaluating progress in large language models (LLMs) is often constrained by the challenge of verifying responses, limiting assessments to tasks like mathematics, programming, and short-form question-answering. However, many real-world applications require evaluating LLMs in processing professional documents, synthesizing information, and generating comprehensive reports in response to user queries. We introduce ProfBench: a set of over 7000 response-criterion pairs as evaluated by human-experts with professional knowledge across Physics PhD, Chemistry PhD, Finance MBA and Consulting MBA. We build robust and affordable LLM-Judges to evaluate ProfBench rubrics, by mitigating self-enhancement bias and reducing the cost of evaluation by 2-3 orders of magnitude, to make it fair and accessible to the broader community. Our findings reveal that ProfBench poses significant challenges even for state-of-the-art LLMs, with top-performing models like GPT-5-high achieving only 65.9\% overall performance. Furthermore, we identify notable performance disparities between proprietary and open-weight models and provide insights into the role that extended thinking plays in addressing complex, professional-domain tasks. Data: this https URL and Code: this https URL 

---
# NeuroAda: Activating Each Neuron's Potential for Parameter-Efficient Fine-Tuning 

**Authors**: Zhi Zhang, Yixian Shen, Congfeng Cao, Ekaterina Shutova  

**Link**: [PDF](https://arxiv.org/pdf/2510.18940)  

**Abstract**: Existing parameter-efficient fine-tuning (PEFT) methods primarily fall into two categories: addition-based and selective in-situ adaptation. The former, such as LoRA, introduce additional modules to adapt the model to downstream tasks, offering strong memory efficiency. However, their representational capacity is often limited, making them less suitable for fine-grained adaptation. In contrast, the latter directly fine-tunes a carefully chosen subset of the original model parameters, allowing for more precise and effective adaptation, but at the cost of significantly increased memory consumption. To reconcile this trade-off, we propose NeuroAda, a novel PEFT method that enables fine-grained model finetuning while maintaining high memory efficiency. Our approach first identifies important parameters (i.e., connections within the network) as in selective adaptation, and then introduces bypass connections for these selected parameters. During finetuning, only the bypass connections are updated, leaving the original model parameters frozen. Empirical results on 23+ tasks spanning both natural language generation and understanding demonstrate that NeuroAda achieves state-of-the-art performance with as little as $\leq \textbf{0.02}\%$ trainable parameters, while reducing CUDA memory usage by up to 60%. We release our code here: this https URL. 

---
# StutterZero and StutterFormer: End-to-End Speech Conversion for Stuttering Transcription and Correction 

**Authors**: Qianheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18938)  

**Abstract**: Over 70 million people worldwide experience stuttering, yet most automatic speech systems misinterpret disfluent utterances or fail to transcribe them accurately. Existing methods for stutter correction rely on handcrafted feature extraction or multi-stage automatic speech recognition (ASR) and text-to-speech (TTS) pipelines, which separate transcription from audio reconstruction and often amplify distortions. This work introduces StutterZero and StutterFormer, the first end-to-end waveform-to-waveform models that directly convert stuttered speech into fluent speech while jointly predicting its transcription. StutterZero employs a convolutional-bidirectional LSTM encoder-decoder with attention, whereas StutterFormer integrates a dual-stream Transformer with shared acoustic-linguistic representations. Both architectures are trained on paired stuttered-fluent data synthesized from the SEP-28K and LibriStutter corpora and evaluated on unseen speakers from the FluencyBank dataset. Across all benchmarks, StutterZero had a 24% decrease in Word Error Rate (WER) and a 31% improvement in semantic similarity (BERTScore) compared to the leading Whisper-Medium model. StutterFormer achieved better results, with a 28% decrease in WER and a 34% improvement in BERTScore. The results validate the feasibility of direct end-to-end stutter-to-fluent speech conversion, offering new opportunities for inclusive human-computer interaction, speech therapy, and accessibility-oriented AI systems. 

---
# A Justice Lens on Fairness and Ethics Courses in Computing Education: LLM-Assisted Multi-Perspective and Thematic Evaluation 

**Authors**: Kenya S. Andrews, Deborah Dormah Kanubala, Kehinde Aruleba, Francisco Enrique Vicente Castro, Renata A Revelo  

**Link**: [PDF](https://arxiv.org/pdf/2510.18931)  

**Abstract**: Course syllabi set the tone and expectations for courses, shaping the learning experience for both students and instructors. In computing courses, especially those addressing fairness and ethics in artificial intelligence (AI), machine learning (ML), and algorithmic design, it is imperative that we understand how approaches to navigating barriers to fair outcomes are being this http URL expectations should be inclusive, transparent, and grounded in promoting critical thinking. Syllabus analysis offers a way to evaluate the coverage, depth, practices, and expectations within a course. Manual syllabus evaluation, however, is time-consuming and prone to inconsistency. To address this, we developed a justice-oriented scoring rubric and asked a large language model (LLM) to review syllabi through a multi-perspective role simulation. Using this rubric, we evaluated 24 syllabi from four perspectives: instructor, departmental chair, institutional reviewer, and external evaluator. We also prompted the LLM to identify thematic trends across the courses. Findings show that multiperspective evaluation aids us in noting nuanced, role-specific priorities, leveraging them to fill hidden gaps in curricula design of AI/ML and related computing courses focused on fairness and ethics. These insights offer concrete directions for improving the design and delivery of fairness, ethics, and justice content in such courses. 

---
# BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping 

**Authors**: Zhiheng Xi, Xin Guo, Yang Nan, Enyu Zhou, Junrui Shen, Wenxiang Chen, Jiaqi Liu, Jixuan Huang, Zhihao Zhang, Honglin Guo, Xun Deng, Zhikai Lei, Miao Zheng, Guoteng Wang, Shuo Zhang, Peng Sun, Rui Zheng, Hang Yan, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18927)  

**Abstract**: Reinforcement learning (RL) has recently become the core paradigm for aligning and strengthening large language models (LLMs). Yet, applying RL in off-policy settings--where stale data from past policies are used for training--improves sample efficiency, but remains challenging: policy entropy declines sharply, optimization often becomes unstable and may even collapse. Through theoretical and empirical analysis, we identify two key insights: (i) an imbalance in optimization, where negative-advantage samples dominate the policy gradient, suppressing useful behaviors and risking gradient explosions; and (ii) the derived Entropy-Clip Rule, which reveals that the fixed clipping mechanism in PPO-like objectives systematically blocks entropy-increasing updates, thereby driving the policy toward over-exploitation at the expense of exploration. Building on these insights, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a simple yet effective method that dynamically adjusts clipping bounds to adaptively re-balance positive and negative contributions, preserve entropy, and stabilize RL optimization. Across diverse off-policy scenarios--including sample replay and partial rollout--BAPO achieves fast, stable, and data-efficient training. On AIME 2024 and AIME 2025 benchmarks, our 7B BAPO model surpasses open-source counterparts such as SkyWork-OR1-7B, while our 32B BAPO model not only achieves state-of-the-art results among models of the same scale but also outperforms leading proprietary systems like o3-mini and Gemini-2.5-Flash-Thinking. 

---
# Application of Reduced-Order Models for Temporal Multiscale Representations in the Prediction of Dynamical Systems 

**Authors**: Elias Al Ghazal, Jad Mounayer, Beatriz Moya, Sebastian Rodriguez, Chady Ghnatios, Francisco Chinesta  

**Link**: [PDF](https://arxiv.org/pdf/2510.18925)  

**Abstract**: Modeling and predicting the dynamics of complex multiscale systems remains a significant challenge due to their inherent nonlinearities and sensitivity to initial conditions, as well as limitations of traditional machine learning methods that fail to capture high frequency behaviours. To overcome these difficulties, we propose three approaches for multiscale learning. The first leverages the Partition of Unity (PU) method, integrated with neural networks, to decompose the dynamics into local components and directly predict both macro- and micro-scale behaviors. The second applies the Singular Value Decomposition (SVD) to extract dominant modes that explicitly separate macro- and micro-scale dynamics. Since full access to the data matrix is rarely available in practice, we further employ a Sparse High-Order SVD to reconstruct multiscale dynamics from limited measurements. Together, these approaches ensure that both coarse and fine dynamics are accurately captured, making the framework effective for real-world applications involving complex, multi-scale phenomena and adaptable to higher-dimensional systems with incomplete observations, by providing an approximation and interpretation in all time scales present in the phenomena under study. 

---
# Noise-corrected GRPO: From Noisy Rewards to Unbiased Gradients 

**Authors**: Omar El mansouri, Mohamed El Amine Seddik, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2510.18924)  

**Abstract**: Reinforcement learning from human feedback (RLHF) or verifiable rewards (RLVR), the standard paradigm for aligning LLMs or building recent SOTA reasoning models, is highly sensitive to noise from inconsistent or erroneous rewards. Yet, the interaction between such noise and widely used group-based policy optimization methods remains underexplored. We introduce a noise-robust Group Relative Policy Optimization (GRPO) and Done Right GRPO (this http URL) framework that explicitly models reward corruption as Bernoulli noise. Our method applies noise correction after estimating reward flip probabilities to debias the learning signal, yielding provably unbiased gradient estimates. Theoretical analysis shows that group-based methods inherently mitigate individual-level noise, and our correction strategy amplifies this robustness. Empirically, we observe consistent improvements across math and code tasks when applying our noise correction to standard reward model usage, with particular gains of up to 6.7 percentage points in accuracy on math tasks and 1.5 on code tasks under realistic reward model conditions. This work bridges label-noise correction from supervised learning with modern RLHF, offering both theoretical insights and a practical algorithm for noisy real-world deployment. 

---
# Benchmarking On-Device Machine Learning on Apple Silicon with MLX 

**Authors**: Oluwaseun A. Ajayi, Ogundepo Odunayo  

**Link**: [PDF](https://arxiv.org/pdf/2510.18921)  

**Abstract**: The recent widespread adoption of Large Language Models (LLMs) and machine learning in general has sparked research interest in exploring the possibilities of deploying these models on smaller devices such as laptops and mobile phones. This creates a need for frameworks and approaches that are capable of taking advantage of on-device hardware. The MLX framework was created to address this need. It is a framework optimized for machine learning (ML) computations on Apple silicon devices, facilitating easier research, experimentation, and prototyping.
This paper presents a performance evaluation of MLX, focusing on inference latency of transformer models. We compare the performance of different transformer architecture implementations in MLX with their Pytorch counterparts. For this research we create a framework called MLX-transformers which includes different transformer implementations in MLX and downloads the model checkpoints in pytorch and converts it to the MLX format. By leveraging the advanced architecture and capabilities of Apple Silicon, MLX-Transformers enables seamless execution of transformer models directly sourced from Hugging Face, eliminating the need for checkpoint conversion often required when porting models between frameworks.
Our study benchmarks different transformer models on two Apple Silicon macbook devices against an NVIDIA CUDA GPU. Specifically, we compare the inference latency performance of models with the same parameter sizes and checkpoints. We evaluate the performance of BERT, RoBERTa, and XLM-RoBERTa models, with the intention of extending future work to include models of different modalities, thus providing a more comprehensive assessment of MLX's capabilities. The results highlight MLX's potential in enabling efficient and more accessible on-device ML applications within Apple's ecosystem. 

---
# Misinformation Detection using Large Language Models with Explainability 

**Authors**: Jainee Patel, Chintan Bhatt, Himani Trivedi, Thanh Thi Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.18918)  

**Abstract**: The rapid spread of misinformation on online platforms undermines trust among individuals and hinders informed decision making. This paper shows an explainable and computationally efficient pipeline to detect misinformation using transformer-based pretrained language models (PLMs). We optimize both RoBERTa and DistilBERT using a two-step strategy: first, we freeze the backbone and train only the classification head; then, we progressively unfreeze the backbone layers while applying layer-wise learning rate decay. On two real-world benchmark datasets, COVID Fake News and FakeNewsNet GossipCop, we test the proposed approach with a unified protocol of preprocessing and stratified splits. To ensure transparency, we integrate the Local Interpretable Model-Agnostic Explanations (LIME) at the token level to present token-level rationales and SHapley Additive exPlanations (SHAP) at the global feature attribution level. It demonstrates that DistilBERT achieves accuracy comparable to RoBERTa while requiring significantly less computational resources. This work makes two key contributions: (1) it quantitatively shows that a lightweight PLM can maintain task performance while substantially reducing computational cost, and (2) it presents an explainable pipeline that retrieves faithful local and global justifications without compromising performance. The results suggest that PLMs combined with principled fine-tuning and interpretability can be an effective framework for scalable, trustworthy misinformation detection. 

---
# MMAO-Bench: MultiModal All in One Benchmark Reveals Compositional Law between Uni-modal and Omni-modal in OmniModels 

**Authors**: Chen Chen, ZeYang Hu, Fengjiao Chen, Liya Ma, Jiaxing Liu, Xiaoyu Li, Xuezhi Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.18915)  

**Abstract**: Multimodal Large Languages models have been progressing from uni-modal understanding toward unifying visual, audio and language modalities, collectively termed omni models. However, the correlation between uni-modal and omni-modal remains unclear, which requires comprehensive evaluation to drive omni model's intelligence evolution. In this work, we propose a novel, high quality and diversity omni model benchmark, MultiModal All in One Benchmark (MMAO-Bench), which effectively assesses both uni-modal and omni-modal understanding capabilities. The benchmark consists of 1880 human curated samples, across 44 task types, and a innovative multi-step open-ended question type that better assess complex reasoning tasks. Experimental result shows the compositional law between cross-modal and uni-modal performance and the omni-modal capability manifests as a bottleneck effect on weak models, while exhibiting synergistic promotion on strong models. 

---
# Context-aware Fairness Evaluation and Mitigation in LLMs 

**Authors**: Afrozah Nadeem, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2510.18914)  

**Abstract**: Large language models often display undesirable behaviors embedded in their internal representations, undermining fairness, inconsistency drift, amplification of harmful content, and the propagation of unwanted patterns during extended dialogue and conversations. Although training-time or data-centric methods attempt to reduce these effects, they are computationally expensive, irreversible once deployed, and slow to adapt to new conversational contexts. Pruning-based methods provide a flexible and transparent way to reduce bias by adjusting the neurons responsible for certain behaviors. However, most existing approaches are static; once a neuron is removed, the model loses the ability to adapt when the conversation or context changes. To address this, we propose a dynamic, reversible, pruning-based framework that detects context-aware neuron activations and applies adaptive masking to modulate their influence during generation. Our inference-time solution provides fine-grained, memory-aware mitigation with knowledge-preserved, more coherent behavior across multilingual single- and multi-turn dialogues, enabling dynamic fairness control in real-world conversational AI. 

---
# ADPO: Anchored Direct Preference Optimization 

**Authors**: Wang Zixian  

**Link**: [PDF](https://arxiv.org/pdf/2510.18913)  

**Abstract**: Anchored Direct Preference Optimization (ADPO) is a unified framework that generalizes Direct Preference Optimization (DPO) with soft preferences, reference-policy anchoring, and groupwise extensions. While standard DPO assumes hard binary labels and pairwise comparisons, ADPO introduces: (i) soft preference probabilities that encode uncertainty and mitigate gradient drift; (ii) arbitrary reference-policy anchors that stabilize training via groupwise shift invariance and implicit KL regularization; and (iii) listwise preference modeling through Plackett-Luce distributions. We prove that DPO, Bradley-Terry objectives, and Top-1-vs-Rest formulations emerge as special cases. ADPO yields three practical variants: pairwise anchored Soft-DPO, listwise anchored Soft-DPO with raw rewards, and KDE-based listwise smoothing for heavy-tailed noise. In contextual bandits, anchoring improves WinMass by 38-63% over standard DPO, while KDE smoothing achieves 0.68 vs 0.32 under heavy-tailed contamination (112% relative gain). In sequential reinforcement learning (CartPole, LunarLander), anchoring improves noisy-preference performance by 15-29%, confirming transfer from single-step to multi-step settings. Experiments with 10-256 parameter models provide clear guidance: use pairwise anchored Soft-DPO for clean or moderate noise, and KDE-based listwise ADPO for extreme contamination. 

---
# Prospects for Using Artificial Intelligence to Understand Intrinsic Kinetics of Heterogeneous Catalytic Reactions 

**Authors**: Andrew J. Medford, Todd N. Whittaker, Bjarne Kreitz, David W. Flaherty, John R. Kitchin  

**Link**: [PDF](https://arxiv.org/pdf/2510.18911)  

**Abstract**: Artificial intelligence (AI) is influencing heterogeneous catalysis research by accelerating simulations and materials discovery. A key frontier is integrating AI with multiscale models and multimodal experiments to address the "many-to-one" challenge of linking intrinsic kinetics to observables. Advances in machine-learned force fields, microkinetics, and reactor modeling enable rapid exploration of chemical spaces, while operando and transient data provide unprecedented insight. Yet, inconsistent data quality and model complexity limit mechanistic discovery. Generative and agentic AI can automate model generation, quantify uncertainty, and couple theory with experiment, realizing "self-driving models" that produce interpretable, reproducible, and transferable understanding of catalytic systems. 

---
# Large Connectome Model: An fMRI Foundation Model of Brain Connectomes Empowered by Brain-Environment Interaction in Multitask Learning Landscape 

**Authors**: Ziquan Wei, Tingting Dan, Guorong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.18910)  

**Abstract**: A reliable foundation model of functional neuroimages is critical to promote clinical applications where the performance of current AI models is significantly impeded by a limited sample size. To that end, tremendous efforts have been made to pretraining large models on extensive unlabeled fMRI data using scalable self-supervised learning. Since self-supervision is not necessarily aligned with the brain-to-outcome relationship, most foundation models are suboptimal to the downstream task, such as predicting disease outcomes. By capitalizing on rich environmental variables and demographic data along with an unprecedented amount of functional neuroimages, we form the brain modeling as a multitask learning and present a scalable model architecture for (i) multitask pretraining by tokenizing multiple brain-environment interactions (BEI) and (ii) semi-supervised finetuning by assigning pseudo-labels of pretrained BEI. We have evaluated our foundation model on a variety of applications, including sex prediction, human behavior recognition, and disease early diagnosis of Autism, Parkinson's disease, Alzheimer's disease, and {Schizophrenia}, where promising results indicate the great potential to facilitate current neuroimaging applications in clinical routines. 

---
# Learning from the Best, Differently: A Diversity-Driven Rethinking on Data Selection 

**Authors**: Hongyi He, Xiao Liu, Zhenghao Lin, Mingni Tang, Yi Cheng, Jintao Wang, Wenjie Li, Peng Cheng, Yeyun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2510.18909)  

**Abstract**: High-quality pre-training data is crutial for large language models, where quality captures factual reliability and semantic value, and diversity ensures broad coverage and distributional heterogeneity. Existing approaches typically rely on single or multiple-dimensional score-based selection. However, directly selecting top-scored data often degrades performance, and sampling from a broader range is required to recover results. The above non-monotonicity between dataset scores and downstream benchmark results reveals a fundamental bias: score-based methods collapse correlated dimensions, causing top-scored data to appear high-quality while systematically overlooking diversity. We argue that ensuring diversity requires decomposing correlated metrics into orthogonal feature dimensions, from which the top-scored data can be directly selected. Therefore, we proposed the Orthogonal Diversity-Aware Selection (ODiS) algorithm, which preserves both quality and diversity during data selection. First, ODiS evaluates data from multiple dimensions, covering language quality, knowledge quality, and comprehension difficulty. The multi-dimensional scores are then decorrelated via Principal Component Analysis (PCA), yielding orthogonal evaluation dimensions. For each dimension, a Roberta-based scorer is trained to regress the data onto PCA-projected scores, enabling scalable inference on large corpora. Finally, ODiS constructs the training dataset by selecting top-scored data within each orthogonal dimension, thereby ensuring both quality and diversity. Empirical results show that ODiS-selected data exhibit less than 2\% inter-dimension overlap, confirming orthogonality between dimensions. More importantly, models trained with ODiS-selected data significantly outperform other baselines on downstream benchmarks, highlighting the necessity of orthogonal, diversity-aware data selection for LLMs. 

---
# Improving Topic Modeling of Social Media Short Texts with Rephrasing: A Case Study of COVID-19 Related Tweets 

**Authors**: Wangjiaxuan Xin, Shuhua Yin, Shi Chen, Yaorong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2510.18908)  

**Abstract**: Social media platforms such as Twitter (now X) provide rich data for analyzing public discourse, especially during crises such as the COVID-19 pandemic. However, the brevity, informality, and noise of social media short texts often hinder the effectiveness of traditional topic modeling, producing incoherent or redundant topics that are often difficult to interpret. To address these challenges, we have developed \emph{TM-Rephrase}, a model-agnostic framework that leverages large language models (LLMs) to rephrase raw tweets into more standardized and formal language prior to topic modeling. Using a dataset of 25,027 COVID-19-related Twitter posts, we investigate the effects of two rephrasing strategies, general- and colloquial-to-formal-rephrasing, on multiple topic modeling methods. Results demonstrate that \emph{TM-Rephrase} improves three metrics measuring topic modeling performance (i.e., topic coherence, topic uniqueness, and topic diversity) while reducing topic redundancy of most topic modeling algorithms, with the colloquial-to-formal strategy yielding the greatest performance gains and especially for the Latent Dirichlet Allocation (LDA) algorithm. This study contributes to a model-agnostic approach to enhancing topic modeling in public health related social media analysis, with broad implications for improved understanding of public discourse in health crisis as well as other important domains. 

---
# 3D Optimization for AI Inference Scaling: Balancing Accuracy, Cost, and Latency 

**Authors**: Minseok Jung, Abhas Ricky, Muhammad Rameez Chatni  

**Link**: [PDF](https://arxiv.org/pdf/2510.18905)  

**Abstract**: AI inference scaling is often tuned through 1D heuristics (a fixed reasoning passes) or 2D bivariate trade-offs (e.g., performance vs. compute), which fail to consider cost and latency constraints. We introduce a 3D optimization framework that jointly calibrates accuracy, cost, and latency within a unified decision space, enabling constraints-aware inference scaling. Using Monte Carlo simulations across three representative scenarios and nine simulated large language models, we evaluate four optimization methods to address the 3D multi-objective optimization (MOO) problem. Framing inference scaling in MOO shapes a feasible space that 1D and 2D optimizations fail to capture, enabling environmentadaptive selection of the inference scaling k. Results show that knee-point optimization achieves the best balance, while accuracy-maximization remains favorable when precision is prioritized. The framework establishes a theoretical foundation for deployment-aware inference scaling across diverse operational contexts. 

---
# DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code 

**Authors**: Shriyansh Agrawal, Aidan Lau, Sanyam Shah, Ahan M R, Kevin Zhu, Sunishchal Dev, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2510.18904)  

**Abstract**: The prevalence of Large Language Models (LLMs) for generating multilingual text and source code has only increased the imperative for machine-generated content detectors to be accurate and efficient across domains. Current detectors, predominantly utilizing zero-shot methods, such as Fast DetectGPT or GPTZero, either incur high computational cost or lack sufficient accuracy, often with a trade-off between the two, leaving room for further improvement. To address these gaps, we propose the fine-tuning of encoder-only Small Language Models (SLMs), in particular, the pre-trained models of RoBERTA and CodeBERTa using specialized datasets on source code and other natural language to prove that for the task of binary classification, SLMs outperform LLMs by a huge margin whilst using a fraction of compute. Our encoders achieve AUROC $= 0.97$ to $0.99$ and macro-F1 $0.89$ to $0.94$ while reducing latency by $8$-$12\times$ and peak VRAM by $3$-$5\times$ at $512$-token inputs. Under cross-generator shifts and adversarial transformations (paraphrase, back-translation; code formatting/renaming), performance retains $\geq 92%$ of clean AUROC. We release training and evaluation scripts with seeds and configs; a reproducibility checklist is also included. 

---
# Evaluating LLMs for Career Guidance: Comparative Analysis of Computing Competency Recommendations Across Ten African Countries 

**Authors**: Precious Eze, Stephanie Lunn, Bruk Berhane  

**Link**: [PDF](https://arxiv.org/pdf/2510.18902)  

**Abstract**: Employers increasingly expect graduates to utilize large language models (LLMs) in the workplace, yet the competencies needed for computing roles across Africa remain unclear given varying national contexts. This study examined how six LLMs, namely ChatGPT 4, DeepSeek, Gemini, Claude 3.5, Llama 3, and Mistral AI, describe entry-level computing career expectations across ten African countries. Using the Computing Curricula 2020 framework and drawing on Digital Colonialism Theory and Ubuntu Philosophy, we analyzed 60 LLM responses to standardized prompts. Technical skills such as cloud computing and programming appeared consistently, but notable differences emerged in how models addressed non-technical competencies, particularly ethics and responsible AI use. Models varied considerably in recognizing country-specific factors, including local technology ecosystems, language requirements, and national policies. Open-source models demonstrated stronger contextual awareness and a better balance between technical and professional skills, earning top scores in nine of ten countries. Still, all models struggled with cultural sensitivity and infrastructure considerations, averaging only 35.4% contextual awareness. This first broad comparison of LLM career guidance for African computing students uncovers entrenched infrastructure assumptions and Western-centric biases, creating gaps between technical recommendations and local needs. The strong performance of cost-effective open-source models (Llama: 4.47/5; DeepSeek: 4.25/5) compared to proprietary alternatives (ChatGPT 4: 3.90/5; Claude: 3.46/5) challenges assumptions about AI tool quality in resource-constrained settings. Our findings highlight how computing competency requirements vary widely across Africa and underscore the need for decolonial approaches to AI in education that emphasize contextual relevance 

---
# AI for Distributed Systems Design: Scalable Cloud Optimization Through Repeated LLMs Sampling And Simulators 

**Authors**: Jacopo Tagliabue  

**Link**: [PDF](https://arxiv.org/pdf/2510.18897)  

**Abstract**: We explore AI-driven distributed-systems policy design by combining stochastic code generation from large language models (LLMs) with deterministic verification in a domain-specific simulator. Using a Function-as-a-Service runtime (Bauplan) and its open-source simulator (Eudoxia) as a case study, we frame scheduler design as an iterative generate-and-verify loop: an LLM proposes a Python policy, the simulator evaluates it on standardized traces, and structured feedback steers subsequent generations. This setup preserves interpretability while enabling targeted search over a large design space. We detail the system architecture and report preliminary results on throughput improvements across multiple models. Beyond early gains, we discuss the limits of the current setup and outline next steps; in particular, we conjecture that AI will be crucial for scaling this methodology by helping to bootstrap new simulators. 

---
# CosmoCore Affective Dream-Replay Reinforcement Learning for Code Generation 

**Authors**: Santhosh Kumar Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2510.18895)  

**Abstract**: We introduce CosmoCore, a neuroscience-inspired reinforcement learning (RL) architecture that integrates affective signals to enhance code generation in large language models (LLMs). Motivated by human and animal learning where embarrassment from mistakes drives rapid correction, as observed in training a puppy to avoid repeating errors after a single scolding CosmoCore tags code generation trajectories with valence and surprise using a lightweight multi-layer perceptron (MLP). High-negative valence (cringe) episodes, such as buggy code outputs, are prioritized in a Dream Queue for five-fold replay during off-policy updates, while low-surprise successes are pruned to prevent overconfidence and buffer bloat. Evaluated on code generation benchmarks like HumanEval and BigCodeBench, alongside simulations with a custom data pipeline environment, CosmoCore reduces hallucinated code (e.g., syntax errors or logical bugs) by 48\% and accelerates self-correction by 45\%. Local experiments using Hugging Face models in a PySpark environment validate these gains, with code snippets provided for replication. Ablations confirm valence tagging boosts curiosity in exploration, and pruning mitigates inefficiency. This framework extends RL from human feedback (RLHF) for more emotionally aware code assistants, with applications in IDEs and data pipelines. Code and the custom mini-world simulation are released. 

---
# CodeCRDT: Observation-Driven Coordination for Multi-Agent LLM Code Generation 

**Authors**: Sergey Pugachev  

**Link**: [PDF](https://arxiv.org/pdf/2510.18893)  

**Abstract**: Multi-agent LLM systems fail to realize parallel speedups due to costly coordination. We present CodeCRDT, an observation-driven coordination pattern where agents coordinate by monitoring a shared state with observable updates and deterministic convergence, rather than explicit message passing. Using Conflict-Free Replicated Data Types (CRDTs), CodeCRDT enables lock-free, conflict-free concurrent code generation with strong eventual consistency. Evaluation across 600 trials (6 tasks, 50 runs per mode) shows both benefits and trade-offs: up to 21.1% speedup on some tasks, up to 39.4% slowdown on others, and 100% convergence with zero merge failures. The study formalizes observation-driven coordination for stochastic LLM agents, revealing semantic conflict rates (5-10%) and quality-performance tradeoffs, and provides empirical characterization of when parallel coordination succeeds versus fails based on task structure. 

---
# Small Language Models Offer Significant Potential for Science Community 

**Authors**: Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.18890)  

**Abstract**: Recent advancements in natural language processing, particularly with large language models (LLMs), are transforming how scientists engage with the literature. While the adoption of LLMs is increasing, concerns remain regarding potential information biases and computational costs. Rather than LLMs, I developed a framework to evaluate the feasibility of precise, rapid, and cost-effective information retrieval from extensive geoscience literature using freely available small language models (MiniLMs). A curated corpus of approximately 77 million high-quality sentences, extracted from 95 leading peer-reviewed geoscience journals such as Geophysical Research Letters and Earth and Planetary Science Letters published during years 2000 to 2024, was constructed. MiniLMs enable a computationally efficient approach for extracting relevant domain-specific information from these corpora through semantic search techniques and sentence-level indexing. This approach, unlike LLMs such as ChatGPT-4 that often produces generalized responses, excels at identifying substantial amounts of expert-verified information with established, multi-disciplinary sources, especially for information with quantitative findings. Furthermore, by analyzing emotional tone via sentiment analysis and topical clusters through unsupervised clustering within sentences, MiniLM provides a powerful tool for tracking the evolution of conclusions, research priorities, advancements, and emerging questions within geoscience communities. Overall, MiniLM holds significant potential within the geoscience community for applications such as fact and image retrievals, trend analyses, contradiction analyses, and educational purposes. 

---
# Contextual Augmentation for Entity Linking using Large Language Models 

**Authors**: Daniel Vollmers, Hamada M. Zahera, Diego Moussallem, Axel-Cyrille Ngonga Ngomo  

**Link**: [PDF](https://arxiv.org/pdf/2510.18888)  

**Abstract**: Entity Linking involves detecting and linking entity mentions in natural language texts to a knowledge graph. Traditional methods use a two-step process with separate models for entity recognition and disambiguation, which can be computationally intensive and less effective. We propose a fine-tuned model that jointly integrates entity recognition and disambiguation in a unified framework. Furthermore, our approach leverages large language models to enrich the context of entity mentions, yielding better performance in entity disambiguation. We evaluated our approach on benchmark datasets and compared with several baselines. The evaluation results show that our approach achieves state-of-the-art performance on out-of-domain datasets. 

---
# LLM Bazaar: A Service Design for Supporting Collaborative Learning with an LLM-Powered Multi-Party Collaboration Infrastructure 

**Authors**: Zhen Wu, Jiaxin Shi, R. Charles Murray, Carolyn Rosé, Micah San Andres  

**Link**: [PDF](https://arxiv.org/pdf/2510.18877)  

**Abstract**: For nearly two decades, conversational agents have played a critical role in structuring interactions in collaborative learning, shaping group dynamics, and supporting student engagement. The recent integration of large language models (LLMs) into these agents offers new possibilities for fostering critical thinking and collaborative problem solving. In this work, we begin with an open source collaboration support architecture called Bazaar and integrate an LLM-agent shell that enables introduction of LLM-empowered, real time, context sensitive collaborative support for group learning. This design and infrastructure paves the way for exploring how tailored LLM-empowered environments can reshape collaborative learning outcomes and interaction patterns. 

---
# What is Implementation Science; and Why It Matters for Bridging the Artificial Intelligence Innovation-to-Application Gap in Medical Imaging 

**Authors**: Ahmad Fayaz-Bakhsh, Janice Tania, Syaheerah Lebai Lutfi, Abhinav K. Jha, Arman Rahmim  

**Link**: [PDF](https://arxiv.org/pdf/2510.13006)  

**Abstract**: The transformative potential of artificial intelligence (AI) in medical Imaging (MI) is well recognized. Yet despite promising reports in research settings, many AI tools fail to achieve clinical adoption in practice. In fact, more generally, there is a documented 17-year average delay between evidence generation and implementation of a technology1. Implementation science (IS) may provide a practical, evidence-based framework to bridge the gap between AI development and real-world clinical imaging use that helps shorten this lag through systematic frameworks, strategies, and hybrid research designs. We outline challenges specific to AI adoption in MI workflows, including infrastructural, educational, and cultural barriers. We highlight the complementary roles of effectiveness research and implementation research, emphasizing hybrid study designs and the role of integrated KT (iKT), stakeholder engagement, and equity-focused co-creation in designing sustainable and generalizable solutions. We discuss integration of Human-Computer Interaction (HCI) frameworks in MI towards usable AI. Adopting IS is not only a methodological advancement; it is a strategic imperative for accelerating translation of innovation into improved patient outcomes. 

---
# A Unified Formal Theory on the Logical Limits of Symbol Grounding 

**Authors**: Zhangchi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20409)  

**Abstract**: This paper synthesizes a series of formal proofs to construct a unified theory on the logical limits of the Symbol Grounding Problem. We demonstrate through a four-stage argument that meaning within a formal system must arise from a process that is external, dynamic, and non-algorithmic. First, we prove that any purely symbolic system, devoid of external connections, cannot internally establish a consistent foundation for meaning due to self-referential paradoxes. Second, we extend this limitation to systems with any finite, static set of pre-established meanings, proving they are inherently incomplete. Third, we demonstrate that the very "act" of connecting an internal symbol to an external meaning cannot be a product of logical inference within the system but must be an axiomatic, meta-level update. Finally, we prove that any attempt to automate this update process using a fixed, external "judgment" algorithm will inevitably construct a larger, yet equally incomplete, symbolic system. Together, these conclusions formally establish that the grounding of meaning is a necessarily open-ended, non-algorithmic process, revealing a fundamental, Gödel-style limitation for any self-contained intelligent system. 

---
