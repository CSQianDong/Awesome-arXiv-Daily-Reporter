# Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions 

**Authors**: Abhishek Rath  

**Link**: [PDF](https://arxiv.org/pdf/2601.04170)  

**Abstract**: Multi-agent Large Language Model (LLM) systems have emerged as powerful architectures for complex task decomposition and collaborative problem-solving. However, their long-term behavioral stability remains largely unexamined. This study introduces the concept of agent drift, defined as the progressive degradation of agent behavior, decision quality, and inter-agent coherence over extended interaction sequences. We present a comprehensive theoretical framework for understanding drift phenomena, proposing three distinct manifestations: semantic drift (progressive deviation from original intent), coordination drift (breakdown in multi-agent consensus mechanisms), and behavioral drift (emergence of unintended strategies).
We introduce the Agent Stability Index (ASI), a novel composite metric framework for quantifying drift across twelve dimensions, including response consistency, tool usage patterns, reasoning pathway stability, and inter-agent agreement rates. Through simulation-based analysis and theoretical modeling, we demonstrate how unchecked agent drift can lead to substantial reductions in task completion accuracy and increased human intervention requirements.
We propose three mitigation strategies: episodic memory consolidation, drift-aware routing protocols, and adaptive behavioral anchoring. Theoretical analysis suggests these approaches can significantly reduce drift-related errors while maintaining system throughput. This work establishes a foundational methodology for monitoring, measuring, and mitigating agent drift in production agentic AI systems, with direct implications for enterprise deployment reliability and AI safety research. 

---
# ComfySearch: Autonomous Exploration and Reasoning for ComfyUI Workflows 

**Authors**: Jinwei Su, Qizhen Lan, Zeyu Wang, Yinghui Xia, Hairu Wen, Yiqun Duan, Xi Xiao, Tianyu Shi, Yang Jingsong, Lewei He  

**Link**: [PDF](https://arxiv.org/pdf/2601.04060)  

**Abstract**: AI-generated content has progressed from monolithic models to modular workflows, especially on platforms like ComfyUI, allowing users to customize complex creative pipelines. However, the large number of components in ComfyUI and the difficulty of maintaining long-horizon structural consistency under strict graph constraints frequently lead to low pass rates and workflows of limited quality. To tackle these limitations, we present ComfySearch, an agentic framework that can effectively explore the component space and generate functional ComfyUI pipelines via validation-guided workflow construction. Experiments demonstrate that ComfySearch substantially outperforms existing methods on complex and creative tasks, achieving higher executability (pass) rates, higher solution rates, and stronger generalization. 

---
# MobileDreamer: Generative Sketch World Model for GUI Agent 

**Authors**: Yilin Cao, Yufeng Zhong, Zhixiong Zeng, Liming Zheng, Jing Huang, Haibo Qiu, Peng Shi, Wenji Mao, Wan Guanglu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04035)  

**Abstract**: Mobile GUI agents have shown strong potential in real-world automation and practical applications. However, most existing agents remain reactive, making decisions mainly from current screen, which limits their performance on long-horizon tasks. Building a world model from repeated interactions enables forecasting action outcomes and supports better decision making for mobile GUI agents. This is challenging because the model must predict post-action states with spatial awareness while remaining efficient enough for practical deployment. In this paper, we propose MobileDreamer, an efficient world-model-based lookahead framework to equip the GUI agents based on the future imagination provided by the world model. It consists of textual sketch world model and rollout imagination for GUI agent. Textual sketch world model forecasts post-action states through a learning process to transform digital images into key task-related sketches, and designs a novel order-invariant learning strategy to preserve the spatial information of GUI elements. The rollout imagination strategy for GUI agent optimizes the action-selection process by leveraging the prediction capability of world model. Experiments on Android World show that MobileDreamer achieves state-of-the-art performance and improves task success by 5.25%. World model evaluations further verify that our textual sketch modeling accurately forecasts key GUI elements. 

---
# Anti-Length Shift: Dynamic Outlier Truncation for Training Efficient Reasoning Models 

**Authors**: Wei Wu, Liyi Chen, Congxi Xiao, Tianfu Wang, Qimeng Wang, Chengqiang Lu, Yan Gao, Yi Wu, Yao Hu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2601.03969)  

**Abstract**: Large reasoning models enhanced by reinforcement learning with verifiable rewards have achieved significant performance gains by extending their chain-of-thought. However, this paradigm incurs substantial deployment costs as models often exhibit excessive verbosity on simple queries. Existing efficient reasoning methods relying on explicit length penalties often introduce optimization conflicts and leave the generative mechanisms driving overthinking largely unexamined. In this paper, we identify a phenomenon termed length shift where models increasingly generate unnecessary reasoning on trivial inputs during training. To address this, we introduce Dynamic Outlier Truncation (DOT), a training-time intervention that selectively suppresses redundant tokens. This method targets only the extreme tail of response lengths within fully correct rollout groups while preserving long-horizon reasoning capabilities for complex problems. To complement this intervention and ensure stable convergence, we further incorporate auxiliary KL regularization and predictive dynamic sampling. Experimental results across multiple model scales demonstrate that our approach significantly pushes the efficiency-performance Pareto frontier outward. Notably, on the AIME-24, our method reduces inference token usage by 78% while simultaneously increasing accuracy compared to the initial policy and surpassing state-of-the-art efficient reasoning methods. 

---
# Trade-R1: Bridging Verifiable Rewards to Stochastic Environments via Process-Level Reasoning Verification 

**Authors**: Rui Sun, Yifan Sun, Sheng Xu, Li Zhao, Jing Li, Daxin Jiang, Chen Hua, Zuo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2601.03948)  

**Abstract**: Reinforcement Learning (RL) has enabled Large Language Models (LLMs) to achieve remarkable reasoning in domains like mathematics and coding, where verifiable rewards provide clear signals. However, extending this paradigm to financial decision is challenged by the market's stochastic nature: rewards are verifiable but inherently noisy, causing standard RL to degenerate into reward hacking. To address this, we propose Trade-R1, a model training framework that bridges verifiable rewards to stochastic environments via process-level reasoning verification. Our key innovation is a verification method that transforms the problem of evaluating reasoning over lengthy financial documents into a structured Retrieval-Augmented Generation (RAG) task. We construct a triangular consistency metric, assessing pairwise alignment between retrieved evidence, reasoning chains, and decisions to serve as a validity filter for noisy market returns. We explore two reward integration strategies: Fixed-effect Semantic Reward (FSR) for stable alignment signals, and Dynamic-effect Semantic Reward (DSR) for coupled magnitude optimization. Experiments on different country asset selection demonstrate that our paradigm reduces reward hacking, with DSR achieving superior cross-market generalization while maintaining the highest reasoning consistency. 

---
# Current Agents Fail to Leverage World Model as Tool for Foresight 

**Authors**: Cheng Qian, Emre Can Acikgoz, Bingxuan Li, Xiusi Chen, Yuji Zhang, Bingxiang He, Qinyu Luo, Dilek Hakkani-Tür, Gokhan Tur, Yunzhu Li, Heng Ji, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2601.03905)  

**Abstract**: Agents built on vision-language models increasingly face tasks that demand anticipating future states rather than relying on short-horizon reasoning. Generative world models offer a promising remedy: agents could use them as external simulators to foresee outcomes before acting. This paper empirically examines whether current agents can leverage such world models as tools to enhance their cognition. Across diverse agentic and visual question answering tasks, we observe that some agents rarely invoke simulation (fewer than 1%), frequently misuse predicted rollouts (approximately 15%), and often exhibit inconsistent or even degraded performance (up to 5%) when simulation is available or enforced. Attribution analysis further indicates that the primary bottleneck lies in the agents' capacity to decide when to simulate, how to interpret predicted outcomes, and how to integrate foresight into downstream reasoning. These findings underscore the need for mechanisms that foster calibrated, strategic interaction with world models, paving the way toward more reliable anticipatory cognition in future agent systems. 

---
# Investigating the Grounding Bottleneck for a Large-Scale Configuration Problem: Existing Tools and Constraint-Aware Guessing 

**Authors**: Veronika Semmelrock, Gerhard Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2601.03850)  

**Abstract**: Answer set programming (ASP) aims to realize the AI vision: The user specifies the problem, and the computer solves it. Indeed, ASP has made this vision true in many application domains. However, will current ASP solving techniques scale up for large configuration problems? As a benchmark for such problems, we investigated the configuration of electronic systems, which may comprise more than 30,000 components. We show the potential and limits of current ASP technology, focusing on methods that address the so-called grounding bottleneck, i.e., the sharp increase of memory demands in the size of the problem instances. To push the limits, we investigated the incremental solving approach, which proved effective in practice. However, even in the incremental approach, memory demands impose significant limits. Based on an analysis of grounding, we developed the method constraint-aware guessing, which significantly reduced the memory need. 

---
# xDNN(ASP): Explanation Generation System for Deep Neural Networks powered by Answer Set Programming 

**Authors**: Ly Ly Trieu, Tran Cao Son  

**Link**: [PDF](https://arxiv.org/pdf/2601.03847)  

**Abstract**: Explainable artificial intelligence (xAI) has gained significant attention in recent years. Among other things, explainablility for deep neural networks has been a topic of intensive research due to the meteoric rise in prominence of deep neural networks and their "black-box" nature. xAI approaches can be characterized along different dimensions such as their scope (global versus local explanations) or underlying methodologies (statistic-based versus rule-based strategies). Methods generating global explanations aim to provide reasoning process applicable to all possible output classes while local explanation methods focus only on a single, specific class. SHAP (SHapley Additive exPlanations), a well-known statistical technique, identifies important features of a network. Deep neural network rule extraction method constructs IF-THEN rules that link input conditions to a class. Another approach focuses on generating counterfactuals which help explain how small changes to an input can affect the model's predictions. However, these techniques primarily focus on the input-output relationship and thus neglect the structure of the network in explanation generation.   In this work, we propose xDNN(ASP), an explanation generation system for deep neural networks that provides global explanations. Given a neural network model and its training data, xDNN(ASP) extracts a logic program under answer set semantics that-in the ideal case-represents the trained model, i.e., answer sets of the extracted program correspond one-to-one to input-output pairs of the network. We demonstrate experimentally, using two synthetic datasets, that not only the extracted logic program maintains a high-level of accuracy in the prediction task, but it also provides valuable information for the understanding of the model such as the importance of features as well as the impact of hidden nodes on the prediction. The latter can be used as a guide for reducing the number of nodes used in hidden layers, i.e., providing a means for optimizing the network. 

---
# Formally Explaining Decision Tree Models with Answer Set Programming 

**Authors**: Akihiro Takemura, Masayuki Otani, Katsumi Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2601.03845)  

**Abstract**: Decision tree models, including random forests and gradient-boosted decision trees, are widely used in machine learning due to their high predictive performance.  However, their complex structures often make them difficult to interpret, especially in safety-critical applications where model decisions require formal justification.  Recent work has demonstrated that logical and abductive explanations can be derived through automated reasoning techniques.  In this paper, we propose a method for generating various types of explanations, namely, sufficient, contrastive, majority, and tree-specific explanations, using Answer Set Programming (ASP).  Compared to SAT-based approaches, our ASP-based method offers greater flexibility in encoding user preferences and supports enumeration of all possible explanations.  We empirically evaluate the approach on a diverse set of datasets and demonstrate its effectiveness and limitations compared to existing methods. 

---
# XAI-LAW: A Logic Programming Tool for Modeling, Explaining, and Learning Legal Decisions 

**Authors**: Agostino Dovier, Talissa Dreossi, Andrea Formisano, Benedetta Strizzolo  

**Link**: [PDF](https://arxiv.org/pdf/2601.03844)  

**Abstract**: We propose an approach to model articles of the Italian Criminal Code (ICC), using Answer Set Programming (ASP), and to semi-automatically learn legal rules from examples based on prior judicial decisions. The developed tool is intended to support legal experts during the criminal trial phase by providing reasoning and possible legal outcomes. The methodology involves analyzing and encoding articles of the ICC in ASP, including "crimes against the person" and property offenses. The resulting model is validated on a set of previous verdicts and refined as necessary. During the encoding process, contradictions may arise; these are properly handled by the system, which also generates possible decisions for new cases and provides explanations through a tool that leverages the "supportedness" of stable models. The automatic explainability offered by the tool can also be used to clarify the logic behind judicial decisions, making the decision-making process more interpretable. Furthermore, the tool integrates an inductive logic programming system for ASP, which is employed to generalize legal rules from case examples. 

---
# Defeasible Conditionals using Answer Set Programming 

**Authors**: Racquel Dennison, Jesse Heyninck, Thomas Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2601.03840)  

**Abstract**: Defeasible entailment is concerned with drawing plausible conclusions from incomplete information. A foundational framework for modelling defeasible entailment is the KLM framework. Introduced by Kraus, Lehmann, and Magidor, the KLM framework outlines several key properties for defeasible entailment. One of the most prominent algorithms within this framework is Rational Closure (RC). This paper presents a declarative definition for computing RC using Answer Set Programming (ASP). Our approach enables the automatic construction of the minimal ranked model from a given knowledge base and supports entailment checking for specified queries. We formally prove the correctness of our ASP encoding and conduct empirical evaluations to compare the performance of our implementation with that of existing imperative implementations, specifically the InfOCF solver. The results demonstrate that our ASP-based approach adheres to RC's theoretical foundations and offers improved computational efficiency. 

---
# ROI-Reasoning: Rational Optimization for Inference via Pre-Computation Meta-Cognition 

**Authors**: Muyang Zhao, Qi Qi, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.03822)  

**Abstract**: Large language models (LLMs) can achieve strong reasoning performance with sufficient computation, but they do not inherently know how much computation a task requires. We study budgeted inference-time reasoning for multiple tasks under a strict global token constraint and formalize it as a Ordered Stochastic Multiple-Choice Knapsack Problem(OS-MCKP). This perspective highlights a meta-cognitive requirement -- anticipating task difficulty, estimating return over investment (ROI), and allocating computation strategically. We propose ROI-Reasoning, a two-stage framework that endows LLMs with intrinsic, budget-aware rationality. In the first stage, Meta-Cognitive Fine-Tuning teaches models to predict reasoning cost and expected utility before generation, enabling explicit solve-or-skip decisions. Next, Rationality-Aware Reinforcement Learning optimizes sequential decision making under a hard token budget, allowing models to learn long-horizon allocation strategies. Across budgeted mathematical reasoning benchmarks, ROI-Reasoning consistently improves overall score while substantially reducing regret under tight computation budgets. 

---
# EntroCoT: Enhancing Chain-of-Thought via Adaptive Entropy-Guided Segmentation 

**Authors**: Zihang Li, Yuhang Wang, Yikun Zong, Wenhan Yu, Xiaokun Yuan, Runhan Jiang, Zirui Liu, Tong Yang, Arthur Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03769)  

**Abstract**: Chain-of-Thought (CoT) prompting has significantly enhanced the mathematical reasoning capabilities of Large Language Models. We find existing fine-tuning datasets frequently suffer from the "answer right but reasoning wrong" probelm, where correct final answers are derived from hallucinated, redundant, or logically invalid intermediate steps. This paper proposes EntroCoT, a unified framework for automatically identifying and refining low-quality CoT supervision traces. EntroCoT first proposes an entropy-based mechanism to segment the reasoning trace into multiple steps at uncertain junctures, and then introduces a Monte Carlo rollout-based mechanism to evaluate the marginal contribution of each step. By accurately filtering deceptive reasoning samples, EntroCoT constructs a high-quality dataset where every intermediate step in each reasoning trace facilitates the final answer. Extensive experiments on mathematical benchmarks demonstrate that fine-tuning on the subset constructed by EntroCoT consistently outperforms the baseslines of full-dataset supervision. 

---
# Personalized Medication Planning via Direct Domain Modeling and LLM-Generated Heuristics 

**Authors**: Yonatan Vernik, Alexander Tuisov, David Izhaki, Hana Weitman, Gal A. Kaminka, Alexander Shleyfman  

**Link**: [PDF](https://arxiv.org/pdf/2601.03687)  

**Abstract**: Personalized medication planning involves selecting medications and determining a dosing schedule to achieve medical goals specific to each individual patient. Previous work successfully demonstrated that automated planners, using general domain-independent heuristics, are able to generate personalized treatments, when the domain and problems are modeled using a general domain description language (\pddlp). Unfortunately, this process was limited in practice to consider no more than seven medications. In clinical terms, this is a non-starter. In this paper, we explore the use of automatically-generated domain- and problem-specific heuristics to be used with general search, as a method of scaling up medication planning to levels allowing closer work with clinicians. Specifically, we specify the domain programmatically (specifying an initial state and a successor generation procedure), and use an LLM to generate a problem specific heuristic that can be used by a fixed search algorithm (GBFS). The results indicate dramatic improvements in coverage and planning time, scaling up the number of medications to at least 28, and bringing medication planning one step closer to practical applications. 

---
# Sandwich Reasoning: An Answer-Reasoning-Answer Approach for Low-Latency Query Correction 

**Authors**: Chen Zhang, Kepu Zhang, Jiatong Zhang, Xiao Zhang, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03672)  

**Abstract**: Query correction is a critical entry point in modern search pipelines, demanding high accuracy strictly within real-time latency constraints. Chain-of-Thought (CoT) reasoning improves accuracy but incurs prohibitive latency for real-time query correction. A potential solution is to output an answer before reasoning to reduce latency; however, under autoregressive decoding, the early answer is independent of subsequent reasoning, preventing the model from leveraging its reasoning capability to improve accuracy. To address this issue, we propose Sandwich Reasoning (SandwichR), a novel approach that explicitly aligns a fast initial answer with post-hoc reasoning, enabling low-latency query correction without sacrificing reasoning-aware accuracy. SandwichR follows an Answer-Reasoning-Answer paradigm, producing an initial correction, an explicit reasoning process, and a final refined correction. To align the initial answer with post-reasoning insights, we design a consistency-aware reinforcement learning (RL) strategy: a dedicated consistency reward enforces alignment between the initial and final corrections, while margin-based rejection sampling prioritizes borderline samples where reasoning drives the most impactful corrective gains. Additionally, we construct a high-quality query correction dataset, addressing the lack of specialized benchmarks for complex query correction. Experimental results demonstrate that SandwichR achieves SOTA accuracy comparable to standard CoT while delivering a 40-70% latency reduction, resolving the latency-accuracy trade-off in online search. 

---
# How Does the Thinking Step Influence Model Safety? An Entropy-based Safety Reminder for LRMs 

**Authors**: Su-Hyeon Kim, Hyundong Jin, Yejin Lee, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.03662)  

**Abstract**: Large Reasoning Models (LRMs) achieve remarkable success through explicit thinking steps, yet the thinking steps introduce a novel risk by potentially amplifying unsafe behaviors. Despite this vulnerability, conventional defense mechanisms remain ineffective as they overlook the unique reasoning dynamics of LRMs. In this work, we find that the emergence of safe-reminding phrases within thinking steps plays a pivotal role in ensuring LRM safety. Motivated by this finding, we propose SafeRemind, a decoding-time defense method that dynamically injects safe-reminding phrases into thinking steps. By leveraging entropy triggers to intervene at decision-locking points, SafeRemind redirects potentially harmful trajectories toward safer outcomes without requiring any parameter updates. Extensive evaluations across five LRMs and six benchmarks demonstrate that SafeRemind substantially enhances safety, achieving improvements of up to 45.5%p while preserving core reasoning utility. 

---
# Architecting Agentic Communities using Design Patterns 

**Authors**: Zoran Milosevic, Fethi Rabhi  

**Link**: [PDF](https://arxiv.org/pdf/2601.03624)  

**Abstract**: The rapid evolution of Large Language Models (LLM) and subsequent Agentic AI technologies requires systematic architectural guidance for building sophisticated, production-grade systems. This paper presents an approach for architecting such systems using design patterns derived from enterprise distributed systems standards, formal methods, and industry practice. We classify these patterns into three tiers: LLM Agents (task-specific automation), Agentic AI (adaptive goal-seekers), and Agentic Communities (organizational frameworks where AI agents and human participants coordinate through formal roles, protocols, and governance structures). We focus on Agentic Communities - coordination frameworks encompassing LLM Agents, Agentic AI entities, and humans - most relevant for enterprise and industrial applications. Drawing on established coordination principles from distributed systems, we ground these patterns in a formal framework that specifies collaboration agreements where AI agents and humans fill roles within governed ecosystems. This approach provides both practical guidance and formal verification capabilities, enabling expression of organizational, legal, and ethical rules through accountability mechanisms that ensure operational and verifiable governance of inter-agent communication, negotiation, and intent modeling. We validate this framework through a clinical trial matching case study. Our goal is to provide actionable guidance to practitioners while maintaining the formal rigor essential for enterprise deployment in dynamic, multi-agent ecosystems. 

---
# Interleaved Tool-Call Reasoning for Protein Function Understanding 

**Authors**: Chuanliu Fan, Zicheng Ma, Huanran Meng, Aijia Zhang, Wenjie Du, Jun Zhang, Yi Qin Gao, Ziqiang Cao, Guohong Fu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03604)  

**Abstract**: Recent advances in large language models (LLMs) have highlighted the effectiveness of chain-of-thought reasoning in symbolic domains such as mathematics and programming. However, our study shows that directly transferring such text-based reasoning paradigms to protein function understanding is ineffective: reinforcement learning mainly amplifies superficial keyword patterns while failing to introduce new biological knowledge, resulting in limited generalization. We argue that protein function prediction is a knowledge-intensive scientific task that fundamentally relies on external biological priors and computational tools rather than purely internal reasoning. To address this gap, we propose PFUA, a tool-augmented protein reasoning agent that unifies problem decomposition, tool invocation, and grounded answer generation. Instead of relying on long unconstrained reasoning traces, PFUA integrates domain-specific tools to produce verifiable intermediate evidence. Experiments on four benchmarks demonstrate that PFUA consistently outperforms text-only reasoning models with an average performance improvement of 103%. 

---
# Controllable LLM Reasoning via Sparse Autoencoder-Based Steering 

**Authors**: Yi Fang, Wenjie Wang, Mingfeng Xue, Boyi Deng, Fengli Xu, Dayiheng Liu, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.03595)  

**Abstract**: Large Reasoning Models (LRMs) exhibit human-like cognitive reasoning strategies (e.g. backtracking, cross-verification) during reasoning process, which improves their performance on complex tasks. Currently, reasoning strategies are autonomously selected by LRMs themselves. However, such autonomous selection often produces inefficient or even erroneous reasoning paths. To make reasoning more reliable and flexible, it is important to develop methods for controlling reasoning strategies. Existing methods struggle to control fine-grained reasoning strategies due to conceptual entanglement in LRMs' hidden states. To address this, we leverage Sparse Autoencoders (SAEs) to decompose strategy-entangled hidden states into a disentangled feature space. To identify the few strategy-specific features from the vast pool of SAE features, we propose SAE-Steering, an efficient two-stage feature identification pipeline. SAE-Steering first recalls features that amplify the logits of strategy-specific keywords, filtering out over 99\% of features, and then ranks the remaining features by their control effectiveness. Using the identified strategy-specific features as control vectors, SAE-Steering outperforms existing methods by over 15\% in control effectiveness. Furthermore, controlling reasoning strategies can redirect LRMs from erroneous paths to correct ones, achieving a 7\% absolute accuracy improvement. 

---
# SCRIBE: Structured Mid-Level Supervision for Tool-Using Language Models 

**Authors**: Yuxuan Jiang, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2601.03555)  

**Abstract**: Training reliable tool-augmented agents remains a significant challenge, largely due to the difficulty of credit assignment in multi-step reasoning. While process-level reward models offer a promising direction, existing LLM-based judges often produce noisy and inconsistent signals because they lack fine-grained, task-specific rubrics to distinguish high-level planning from low-level execution. In this work, we introduce SCRIBE (Skill-Conditioned Reward with Intermediate Behavioral Evaluation), a reinforcement learning framework that intervenes at a novel mid-level abstraction. SCRIBE grounds reward modeling in a curated library of skill prototypes, transforming open-ended LLM evaluation into a constrained verification problem. By routing each subgoal to a corresponding prototype, the reward model is equipped with precise, structured rubrics that substantially reduce reward variance.
Experimental results show that SCRIBE achieves state-of-the-art performance across a range of reasoning and tool-use benchmarks. In particular, it improves the AIME25 accuracy of a Qwen3-4B model from 43.3% to 63.3%, and significantly increases success rates in complex multi-turn tool interactions.
Further analysis of training dynamics reveals a co-evolution across abstraction levels, where mastery of mid-level skills consistently precedes the emergence of effective high-level planning behaviors. Finally, we demonstrate that SCRIBE is additive to low-level tool optimizations, providing a scalable and complementary pathway toward more autonomous and reliable tool-using agents. 

---
# ReEfBench: Quantifying the Reasoning Efficiency of LLMs 

**Authors**: Zhizhang Fu, Yuancheng Gu, Chenkai Hu, Hanmeng Liu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03550)  

**Abstract**: Test-time scaling has enabled Large Language Models (LLMs) to tackle complex reasoning, yet the limitations of current Chain-of-Thought (CoT) evaluation obscures whether performance gains stem from genuine reasoning or mere verbosity. To address this, (1) we propose a novel neuro-symbolic framework for the non-intrusive, comprehensive process-centric evaluation of reasoning. (2) Through this lens, we identify four distinct behavioral prototypes and diagnose the failure modes. (3) We examine the impact of inference mode, training strategy, and model scale. Our analysis reveals that extended token generation is not a prerequisite for deep reasoning. Furthermore, we reveal critical constraints: mixing long and short CoT data in training risks in premature saturation and collapse, while distillation into smaller models captures behavioral length but fails to replicate logical efficacy due to intrinsic capacity limits. 

---
# STAR-S: Improving Safety Alignment through Self-Taught Reasoning on Safety Rules 

**Authors**: Di Wu, Yanyan Zhao, Xin Lu, Mingzhe Li, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.03537)  

**Abstract**: Defending against jailbreak attacks is crucial for the safe deployment of Large Language Models (LLMs). Recent research has attempted to improve safety by training models to reason over safety rules before responding. However, a key issue lies in determining what form of safety reasoning effectively defends against jailbreak attacks, which is difficult to explicitly design or directly obtain. To address this, we propose \textbf{STAR-S} (\textbf{S}elf-\textbf{TA}ught \textbf{R}easoning based on \textbf{S}afety rules), a framework that integrates the learning of safety rule reasoning into a self-taught loop. The core of STAR-S involves eliciting reasoning and reflection guided by safety rules, then leveraging fine-tuning to enhance safety reasoning. Repeating this process creates a synergistic cycle. Improvements in the model's reasoning and interpretation of safety rules allow it to produce better reasoning data under safety rule prompts, which is then utilized for further training. Experiments show that STAR-S effectively defends against jailbreak attacks, outperforming baselines. Code is available at: this https URL. 

---
# Variance Computation for Weighted Model Counting with Knowledge Compilation Approach 

**Authors**: Kengo Nakamura, Masaaki Nishino, Norihito Yasuda  

**Link**: [PDF](https://arxiv.org/pdf/2601.03523)  

**Abstract**: One of the most important queries in knowledge compilation is weighted model counting (WMC), which has been applied to probabilistic inference on various models, such as Bayesian networks. In practical situations on inference tasks, the model's parameters have uncertainty because they are often learned from data, and thus we want to compute the degree of uncertainty in the inference outcome. One possible approach is to regard the inference outcome as a random variable by introducing distributions for the parameters and evaluate the variance of the outcome. Unfortunately, the tractability of computing such a variance is hardly known. Motivated by this, we consider the problem of computing the variance of WMC and investigate this problem's tractability. First, we derive a polynomial time algorithm to evaluate the WMC variance when the input is given as a structured d-DNNF. Second, we prove the hardness of this problem for structured DNNFs, d-DNNFs, and FBDDs, which is intriguing because the latter two allow polynomial time WMC algorithms. Finally, we show an application that measures the uncertainty in the inference of Bayesian networks. We empirically show that our algorithm can evaluate the variance of the marginal probability on real-world Bayesian networks and analyze the impact of the variances of parameters on the variance of the marginal. 

---
# Evolving Programmatic Skill Networks 

**Authors**: Haochen Shi, Xingdi Yuan, Bang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03509)  

**Abstract**: We study continual skill acquisition in open-ended embodied environments where an agent must construct, refine, and reuse an expanding library of executable skills. We introduce the Programmatic Skill Network (PSN), a framework in which skills are executable symbolic programs forming a compositional network that evolves through experience. PSN defines three core mechanisms instantiated via large language models: (1)REFLECT for structured fault localization over skill compositions, (2) progressive optimization with maturity-aware update gating that stabilizes reliable skills while maintaining plasticity for uncertain ones, and (3) canonical structural refactoring under rollback validation that maintains network compactness. We further show that PSN's learning dynamics exhibit structural parallels to neural network training. Experiments on MineDojo and Crafter demonstrate robust skill reuse, rapid adaptation, and strong generalization across open-ended task distributions.\footnote{We plan to open-source the code. 

---
# Personalization of Large Foundation Models for Health Interventions 

**Authors**: Stefan Konigorski, Johannes E. Vedder, Babajide Alamu Owoyele, İbrahim Özkan  

**Link**: [PDF](https://arxiv.org/pdf/2601.03482)  

**Abstract**: Large foundation models (LFMs) transform healthcare AI in prevention, diagnostics, and treatment. However, whether LFMs can provide truly personalized treatment recommendations remains an open question. Recent research has revealed multiple challenges for personalization, including the fundamental generalizability paradox: models achieving high accuracy in one clinical study perform at chance level in others, demonstrating that personalization and external validity exist in tension. This exemplifies broader contradictions in AI-driven healthcare: the privacy-performance paradox, scale-specificity paradox, and the automation-empathy paradox. As another challenge, the degree of causal understanding required for personalized recommendations, as opposed to mere predictive capacities of LFMs, remains an open question. N-of-1 trials -- crossover self-experiments and the gold standard for individual causal inference in personalized medicine -- resolve these tensions by providing within-person causal evidence while preserving privacy through local experimentation. Despite their impressive capabilities, this paper argues that LFMs cannot replace N-of-1 trials. We argue that LFMs and N-of-1 trials are complementary: LFMs excel at rapid hypothesis generation from population patterns using multimodal data, while N-of-1 trials excel at causal validation for a given individual. We propose a hybrid framework that combines the strengths of both to enable personalization and navigate the identified paradoxes: LFMs generate ranked intervention candidates with uncertainty estimates, which trigger subsequent N-of-1 trials. Clarifying the boundary between prediction and causation and explicitly addressing the paradoxical tensions are essential for responsible AI integration in personalized medicine. 

---
# CPGPrompt: Translating Clinical Guidelines into LLM-Executable Decision Support 

**Authors**: Ruiqi Deng, Geoffrey Martin, Tony Wang, Gongbo Zhang, Yi Liu, Chunhua Weng, Yanshan Wang, Justin F Rousseau, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2601.03475)  

**Abstract**: Clinical practice guidelines (CPGs) provide evidence-based recommendations for patient care; however, integrating them into Artificial Intelligence (AI) remains challenging. Previous approaches, such as rule-based systems, face significant limitations, including poor interpretability, inconsistent adherence to guidelines, and narrow domain applicability. To address this, we develop and validate CPGPrompt, an auto-prompting system that converts narrative clinical guidelines into large language models (LLMs).
Our framework translates CPGs into structured decision trees and utilizes an LLM to dynamically navigate them for patient case evaluation. Synthetic vignettes were generated across three domains (headache, lower back pain, and prostate cancer) and distributed into four categories to test different decision scenarios. System performance was assessed on both binary specialty-referral decisions and fine-grained pathway-classification tasks.
The binary specialty referral classification achieved consistently strong performance across all domains (F1: 0.85-1.00), with high recall (1.00 $\pm$ 0.00). In contrast, multi-class pathway assignment showed reduced performance, with domain-specific variations: headache (F1: 0.47), lower back pain (F1: 0.72), and prostate cancer (F1: 0.77). Domain-specific performance differences reflected the structure of each guideline. The headache guideline highlighted challenges with negation handling. The lower back pain guideline required temporal reasoning. In contrast, prostate cancer pathways benefited from quantifiable laboratory tests, resulting in more reliable decision-making. 

---
# Toward Maturity-Based Certification of Embodied AI: Quantifying Trustworthiness Through Measurement Mechanisms 

**Authors**: Michael C. Darling, Alan H. Hesu, Michael A. Mardikes, Brian C. McGuigan, Reed M. Milewicz  

**Link**: [PDF](https://arxiv.org/pdf/2601.03470)  

**Abstract**: We propose a maturity-based framework for certifying embodied AI systems through explicit measurement mechanisms. We argue that certifiable embodied AI requires structured assessment frameworks, quantitative scoring mechanisms, and methods for navigating multi-objective trade-offs inherent in trustworthiness evaluation. We demonstrate this approach using uncertainty quantification as an exemplar measurement mechanism and illustrate feasibility through an Uncrewed Aircraft System (UAS) detection case study. 

---
# Exploration Through Introspection: A Self-Aware Reward Model 

**Authors**: Michael Petrowski, Milica Gašić  

**Link**: [PDF](https://arxiv.org/pdf/2601.03389)  

**Abstract**: Understanding how artificial agents model internal mental states is central to advancing Theory of Mind in AI. Evidence points to a unified system for self- and other-awareness. We explore this self-awareness by having reinforcement learning agents infer their own internal states in gridworld environments. Specifically, we introduce an introspective exploration component that is inspired by biological pain as a learning signal by utilizing a hidden Markov model to infer "pain-belief" from online observations. This signal is integrated into a subjective reward function to study how self-awareness affects the agent's learning abilities. Further, we use this computational framework to investigate the difference in performance between normal and chronic pain perception models. Results show that introspective agents in general significantly outperform standard baseline agents and can replicate complex human-like behaviors. 

---
# Enhancing LLM Instruction Following: An Evaluation-Driven Multi-Agentic Workflow for Prompt Instructions Optimization 

**Authors**: Alberto Purpura, Li Wang, Sahil Badyal, Eugenio Beaufrand, Adam Faulkner  

**Link**: [PDF](https://arxiv.org/pdf/2601.03359)  

**Abstract**: Large Language Models (LLMs) often generate substantively relevant content but fail to adhere to formal constraints, leading to outputs that are conceptually correct but procedurally flawed. Traditional prompt refinement approaches focus on rephrasing the description of the primary task an LLM has to perform, neglecting the granular constraints that function as acceptance criteria for its response. We propose a novel multi-agentic workflow that decouples optimization of the primary task description from its constraints, using quantitative scores as feedback to iteratively rewrite and improve them. Our evaluation demonstrates this method produces revised prompts that yield significantly higher compliance scores from models like Llama 3.1 8B and Mixtral-8x 7B. 

---
# Digital Red Queen: Adversarial Program Evolution in Core War with LLMs 

**Authors**: Akarsh Kumar, Ryan Bahlous-Boldi, Prafull Sharma, Phillip Isola, Sebastian Risi, Yujin Tang, David Ha  

**Link**: [PDF](https://arxiv.org/pdf/2601.03335)  

**Abstract**: Large language models (LLMs) are increasingly being used to evolve solutions to problems in many domains, in a process inspired by biological evolution. However, unlike biological evolution, most LLM-evolution frameworks are formulated as static optimization problems, overlooking the open-ended adversarial dynamics that characterize real-world evolutionary processes. Here, we study Digital Red Queen (DRQ), a simple self-play algorithm that embraces these so-called "Red Queen" dynamics via continual adaptation to a changing objective. DRQ uses an LLM to evolve assembly-like programs, called warriors, which compete against each other for control of a virtual machine in the game of Core War, a Turing-complete environment studied in artificial life and connected to cybersecurity. In each round of DRQ, the model evolves a new warrior to defeat all previous ones, producing a sequence of adapted warriors. Over many rounds, we observe that warriors become increasingly general (relative to a set of held-out human warriors). Interestingly, warriors also become less behaviorally diverse across independent runs, indicating a convergence pressure toward a general-purpose behavioral strategy, much like convergent evolution in nature. This result highlights a potential value of shifting from static objectives to dynamic Red Queen objectives. Our work positions Core War as a rich, controllable sandbox for studying adversarial adaptation in artificial systems and for evaluating LLM-based evolution methods. More broadly, the simplicity and effectiveness of DRQ suggest that similarly minimal self-play approaches could prove useful in other more practical multi-agent adversarial domains, like real-world cybersecurity or combating drug resistance. 

---
# Mastering the Game of Go with Self-play Experience Replay 

**Authors**: Jingbin Liu, Xuechun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03306)  

**Abstract**: The game of Go has long served as a benchmark for artificial intelligence, demanding sophisticated strategic reasoning and long-term planning. Previous approaches such as AlphaGo and its successors, have predominantly relied on model-based Monte-Carlo Tree Search (MCTS). In this work, we present QZero, a novel model-free reinforcement learning algorithm that forgoes search during training and learns a Nash equilibrium policy through self-play and off-policy experience replay. Built upon entropy-regularized Q-learning, QZero utilizes a single Q-value network to unify policy evaluation and improvement. Starting tabula rasa without human data and trained for 5 months with modest compute resources (7 GPUs), QZero achieved a performance level comparable to that of AlphaGo. This demonstrates, for the first time, the efficiency of using model-free reinforcement learning to master the game of Go, as well as the feasibility of off-policy reinforcement learning in solving large-scale and complex environments. 

---
# Embedding Autonomous Agents in Resource-Constrained Robotic Platforms 

**Authors**: Negar Halakou, Juan F. Gutierrez, Ye Sun, Han Jiang, Xueming Wu, Yilun Song, Andres Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2601.04191)  

**Abstract**: Many embedded devices operate under resource constraints and in dynamic environments, requiring local decision-making capabilities. Enabling devices to make independent decisions in such environments can improve the responsiveness of the system and reduce the dependence on constant external control. In this work, we integrate an autonomous agent, programmed using AgentSpeak, with a small two-wheeled robot that explores a maze using its own decision-making and sensor data. Experimental results show that the agent successfully solved the maze in 59 seconds using 287 reasoning cycles, with decision phases taking less than one millisecond. These results indicate that the reasoning process is efficient enough for real-time execution on resource-constrained hardware. This integration demonstrates how high-level agent-based control can be applied to resource-constrained embedded systems for autonomous operation. 

---
# Clinical Data Goes MEDS? Let's OWL make sense of it 

**Authors**: Alberto Marfoglia, Jong Ho Jhee, Adrien Coulet  

**Link**: [PDF](https://arxiv.org/pdf/2601.04164)  

**Abstract**: The application of machine learning on healthcare data is often hindered by the lack of standardized and semantically explicit representation, leading to limited interoperability and reproducibility across datasets and experiments. The Medical Event Data Standard (MEDS) addresses these issues by introducing a minimal, event-centric data model designed for reproducible machine-learning workflows from health data. However, MEDS is defined as a data-format specification and does not natively provide integration with the Semantic Web ecosystem. In this article, we introduce MEDS-OWL, a lightweight OWL ontology that provides formal concepts and relations to enable representing MEDS datasets as RDF graphs. Additionally, we implemented meds2rdf, a Python conversion library that transforms MEDS events into RDF graphs, ensuring conformance with the ontology. We demonstrate the approach on a synthetic clinical dataset that describes patient care pathways for ruptured intracranial aneurysms and validate the resulting graph using SHACL constraints. The first release of MEDS-OWL comprises 13 classes, 10 object properties, 20 data properties, and 24 OWL axioms. Combined with meds2rdf, it enables data transformation into FAIR-aligned datasets, provenance-aware publishing, and interoperability of event-based clinical data. By bridging MEDS with the Semantic Web, this work contributes a reusable semantic layer for event-based clinical data and establishes a robust foundation for subsequent graph-based analytics. 

---
# Klear: Unified Multi-Task Audio-Video Joint Generation 

**Authors**: Jun Wang, Chunyu Qiang, Yuxin Guo, Yiran Wang, Xijuan Zeng, Chen Zhang, Pengfei Wan  

**Link**: [PDF](https://arxiv.org/pdf/2601.04151)  

**Abstract**: Audio-video joint generation has progressed rapidly, yet substantial challenges still remain. Non-commercial approaches still suffer audio-visual asynchrony, poor lip-speech alignment, and unimodal degradation, which can be stemmed from weak audio-visual correspondence modeling, limited generalization, and scarce high-quality dense-caption data. To address these issues, we introduce Klear and delve into three axes--model architecture, training strategy, and data curation. Architecturally, we adopt a single-tower design with unified DiT blocks and an Omni-Full Attention mechanism, achieving tight audio-visual alignment and strong scalability. Training-wise, we adopt a progressive multitask regime--random modality masking to joint optimization across tasks, and a multistage curriculum, yielding robust representations, strengthening A-V aligned world knowledge, and preventing unimodal collapse. For datasets, we present the first large-scale audio-video dataset with dense captions, and introduce a novel automated data-construction pipeline which annotates and filters millions of diverse, high-quality, strictly aligned audio-video-caption triplets. Building on this, Klear scales to large datasets, delivering high-fidelity, semantically and temporally aligned, instruction-following generation in both joint and unimodal settings while generalizing robustly to out-of-distribution scenarios. Across tasks, it substantially outperforms prior methods by a large margin and achieves performance comparable to Veo 3, offering a unified, scalable path toward next-generation audio-video synthesis. 

---
# Wow, wo, val! A Comprehensive Embodied World Model Evaluation Turing Test 

**Authors**: Chun-Kai Fan, Xiaowei Chi, Xiaozhu Ju, Hao Li, Yong Bao, Yu-Kai Wang, Lizhang Chen, Zhiyuan Jiang, Kuangzhi Ge, Ying Li, Weishi Mi, Qingpo Wuwu, Peidong Jia, Yulin Luo, Kevin Zhang, Zhiyuan Qin, Yong Dai, Sirui Han, Yike Guo, Shanghang Zhang, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04137)  

**Abstract**: As world models gain momentum in Embodied AI, an increasing number of works explore using video foundation models as predictive world models for downstream embodied tasks like 3D prediction or interactive generation. However, before exploring these downstream tasks, video foundation models still have two critical questions unanswered: (1) whether their generative generalization is sufficient to maintain perceptual fidelity in the eyes of human observers, and (2) whether they are robust enough to serve as a universal prior for real-world embodied agents. To provide a standardized framework for answering these questions, we introduce the Embodied Turing Test benchmark: WoW-World-Eval (Wow,wo,val). Building upon 609 robot manipulation data, Wow-wo-val examines five core abilities, including perception, planning, prediction, generalization, and execution. We propose a comprehensive evaluation protocol with 22 metrics to assess the models' generation ability, which achieves a high Pearson Correlation between the overall score and human preference (>0.93) and establishes a reliable foundation for the Human Turing Test. On Wow-wo-val, models achieve only 17.27 on long-horizon planning and at best 68.02 on physical consistency, indicating limited spatiotemporal consistency and physical reasoning. For the Inverse Dynamic Model Turing Test, we first use an IDM to evaluate the video foundation models' execution accuracy in the real world. However, most models collapse to $\approx$ 0% success, while WoW maintains a 40.74% success rate. These findings point to a noticeable gap between the generated videos and the real world, highlighting the urgency and necessity of benchmarking World Model in Embodied AI. 

---
# ContextFocus: Activation Steering for Contextual Faithfulness in Large Language Models 

**Authors**: Nikhil Anand, Shwetha Somasundaram, Anirudh Phukan, Apoorv Saxena, Koyel Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2601.04131)  

**Abstract**: Large Language Models (LLMs) encode vast amounts of parametric knowledge during pre-training. As world knowledge evolves, effective deployment increasingly depends on their ability to faithfully follow externally retrieved context. When such evidence conflicts with the model's internal knowledge, LLMs often default to memorized facts, producing unfaithful outputs. In this work, we introduce ContextFocus, a lightweight activation steering approach that improves context faithfulness in such knowledge-conflict settings while preserving fluency and efficiency. Unlike prior approaches, our solution requires no model finetuning and incurs minimal inference-time overhead, making it highly efficient. We evaluate ContextFocus on the ConFiQA benchmark, comparing it against strong baselines including ContextDPO, COIECD, and prompting-based methods. Furthermore, we show that our method is complementary to prompting strategies and remains effective on larger models. Extensive experiments show that ContextFocus significantly improves contextual-faithfulness. Our results highlight the effectiveness, robustness, and efficiency of ContextFocus in improving contextual-faithfulness of LLM outputs. 

---
# Pixel-Wise Multimodal Contrastive Learning for Remote Sensing Images 

**Authors**: Leandro Stival, Ricardo da Silva Torres, Helio Pedrini  

**Link**: [PDF](https://arxiv.org/pdf/2601.04127)  

**Abstract**: Satellites continuously generate massive volumes of data, particularly for Earth observation, including satellite image time series (SITS). However, most deep learning models are designed to process either entire images or complete time series sequences to extract meaningful features for downstream tasks. In this study, we propose a novel multimodal approach that leverages pixel-wise two-dimensional (2D) representations to encode visual property variations from SITS more effectively. Specifically, we generate recurrence plots from pixel-based vegetation index time series (NDVI, EVI, and SAVI) as an alternative to using raw pixel values, creating more informative representations. Additionally, we introduce PIxel-wise Multimodal Contrastive (PIMC), a new multimodal self-supervision approach that produces effective encoders based on two-dimensional pixel time series representations and remote sensing imagery (RSI). To validate our approach, we assess its performance on three downstream tasks: pixel-level forecasting and classification using the PASTIS dataset, and land cover classification on the EuroSAT dataset. Moreover, we compare our results to state-of-the-art (SOTA) methods on all downstream tasks. Our experimental results show that the use of 2D representations significantly enhances feature extraction from SITS, while contrastive learning improves the quality of representations for both pixel time series and RSI. These findings suggest that our multimodal method outperforms existing models in various Earth observation tasks, establishing it as a robust self-supervision framework for processing both SITS and RSI. Code avaliable on 

---
# InfiniteWeb: Scalable Web Environment Synthesis for GUI Agent Training 

**Authors**: Ziyun Zhang, Zezhou Wang, Xiaoyi Zhang, Zongyu Guo, Jiahao Li, Bin Li, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04126)  

**Abstract**: GUI agents that interact with graphical interfaces on behalf of users represent a promising direction for practical AI assistants. However, training such agents is hindered by the scarcity of suitable environments. We present InfiniteWeb, a system that automatically generates functional web environments at scale for GUI agent training. While LLMs perform well on generating a single webpage, building a realistic and functional website with many interconnected pages faces challenges. We address these challenges through unified specification, task-centric test-driven development, and a combination of website seed with reference design image to ensure diversity. Our system also generates verifiable task evaluators enabling dense reward signals for reinforcement learning. Experiments show that InfiniteWeb surpasses commercial coding agents at realistic website construction, and GUI agents trained on our generated environments achieve significant performance improvements on OSWorld and Online-Mind2Web, demonstrating the effectiveness of proposed system. 

---
# Quantifying the Impact of Modules and Their Interactions in the PSO-X Framework 

**Authors**: Christian L. Camacho-Villalón, Ana Nikolikj, Katharina Dost, Eva Tuba, Sašo Džeroski, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2601.04100)  

**Abstract**: The PSO-X framework incorporates dozens of modules that have been proposed for solving single-objective continuous optimization problems using particle swarm optimization. While modular frameworks enable users to automatically generate and configure algorithms tailored to specific optimization problems, the complexity of this process increases with the number of modules in the framework and the degrees of freedom defined for their interaction. Understanding how modules affect the performance of algorithms for different problems is critical to making the process of finding effective implementations more efficient and identifying promising areas for further investigation. Despite their practical applications and scientific relevance, there is a lack of empirical studies investigating which modules matter most in modular optimization frameworks and how they interact. In this paper, we analyze the performance of 1424 particle swarm optimization algorithms instantiated from the PSO-X framework on the 25 functions in the CEC'05 benchmark suite with 10 and 30 dimensions. We use functional ANOVA to quantify the impact of modules and their combinations on performance in different problem classes. In practice, this allows us to identify which modules have greater influence on PSO-X performance depending on problem features such as multimodality, mathematical transformations and varying dimensionality. We then perform a cluster analysis to identify groups of problem classes that share similar module effect patterns. Our results show low variability in the importance of modules in all problem classes, suggesting that particle swarm optimization performance is driven by a few influential modules. 

---
# Layer-wise Positional Bias in Short-Context Language Modeling 

**Authors**: Maryam Rahimi, Mahdi Nouri, Yadollah Yaghoobzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2601.04098)  

**Abstract**: Language models often show a preference for using information from specific positions in the input regardless of semantic relevance. While positional bias has been studied in various contexts, from attention sinks to task performance degradation in long-context settings, prior work has not established how these biases evolve across individual layers and input positions, or how they vary independent of task complexity. We introduce an attribution-based framework to analyze positional effects in short-context language modeling. Using layer conductance with a sliding-window approach, we quantify how each layer distributes importance across input positions, yielding layer-wise positional importance profiles. We find that these profiles are architecture-specific, stable across inputs, and invariant to lexical scrambling. Characterizing these profiles, we find prominent recency bias that increases with depth and subtle primacy bias that diminishes through model depth. Beyond positional structure, we also show that early layers preferentially weight content words over function words across all positions, while later layers lose this word-type differentiation. 

---
# CSSG: Measuring Code Similarity with Semantic Graphs 

**Authors**: Jingwen Xu, Yiyang Lu, Changze Lv, Zisu Huang, Zhengkang Guo, Zhengyuan Wang, Muzhao Tian, Xuanjing Huang, Xiaoqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.04085)  

**Abstract**: Existing code similarity metrics, such as BLEU, CodeBLEU, and TSED, largely rely on surface-level string overlap or abstract syntax tree structures, and often fail to capture deeper semantic relationships between this http URL propose CSSG (Code Similarity using Semantic Graphs), a novel metric that leverages program dependence graphs to explicitly model control dependencies and variable interactions, providing a semantics-aware representation of this http URL on the CodeContests+ dataset show that CSSG consistently outperforms existing metrics in distinguishing more similar code from less similar code under both monolingual and cross-lingual settings, demonstrating that dependency-aware graph representations offer a more effective alternative to surface-level or syntax-based similarity measures. 

---
# Analyzing Reasoning Consistency in Large Multimodal Models under Cross-Modal Conflicts 

**Authors**: Zhihao Zhu, Jiafeng Liang, Shixin Jiang, Jinlan Fu, Ming Liu, Guanglu Sun, See-Kiong Ng, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.04073)  

**Abstract**: Large Multimodal Models (LMMs) have demonstrated impressive capabilities in video reasoning via Chain-of-Thought (CoT). However, the robustness of their reasoning chains remains questionable. In this paper, we identify a critical failure mode termed textual inertia, where once a textual hallucination occurs in the thinking process, models tend to blindly adhere to the erroneous text while neglecting conflicting visual evidence. To systematically investigate this, we propose the LogicGraph Perturbation Protocol that structurally injects perturbations into the reasoning chains of diverse LMMs spanning both native reasoning architectures and prompt-driven paradigms to evaluate their self-reflection capabilities. The results reveal that models successfully self-correct in less than 10% of cases and predominantly succumb to blind textual error propagation. To mitigate this, we introduce Active Visual-Context Refinement, a training-free inference paradigm which orchestrates an active visual re-grounding mechanism to enforce fine-grained verification coupled with an adaptive context refinement strategy to summarize and denoise the reasoning history. Experiments demonstrate that our approach significantly stifles hallucination propagation and enhances reasoning robustness. 

---
# Mind the Generative Details: Direct Localized Detail Preference Optimization for Video Diffusion Models 

**Authors**: Zitong Huang, Kaidong Zhang, Yukang Ding, Chao Gao, Rui Ding, Ying Chen, Wangmeng Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2601.04068)  

**Abstract**: Aligning text-to-video diffusion models with human preferences is crucial for generating high-quality videos. Existing Direct Preference Otimization (DPO) methods rely on multi-sample ranking and task-specific critic models, which is inefficient and often yields ambiguous global supervision. To address these limitations, we propose LocalDPO, a novel post-training framework that constructs localized preference pairs from real videos and optimizes alignment at the spatio-temporal region level. We design an automated pipeline to efficiently collect preference pair data that generates preference pairs with a single inference per prompt, eliminating the need for external critic models or manual annotation. Specifically, we treat high-quality real videos as positive samples and generate corresponding negatives by locally corrupting them with random spatio-temporal masks and restoring only the masked regions using the frozen base model. During training, we introduce a region-aware DPO loss that restricts preference learning to corrupted areas for rapid convergence. Experiments on Wan2.1 and CogVideoX demonstrate that LocalDPO consistently improves video fidelity, temporal coherence and human preference scores over other post-training approaches, establishing a more efficient and fine-grained paradigm for video generator alignment. 

---
# HoneyTrap: Deceiving Large Language Model Attackers to Honeypot Traps with Resilient Multi-Agent Defense 

**Authors**: Siyuan Li, Xi Lin, Jun Wu, Zehao Liu, Haoyu Li, Tianjie Ju, Xiang Chen, Jianhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04034)  

**Abstract**: Jailbreak attacks pose significant threats to large language models (LLMs), enabling attackers to bypass safeguards. However, existing reactive defense approaches struggle to keep up with the rapidly evolving multi-turn jailbreaks, where attackers continuously deepen their attacks to exploit vulnerabilities. To address this critical challenge, we propose HoneyTrap, a novel deceptive LLM defense framework leveraging collaborative defenders to counter jailbreak attacks. It integrates four defensive agents, Threat Interceptor, Misdirection Controller, Forensic Tracker, and System Harmonizer, each performing a specialized security role and collaborating to complete a deceptive defense. To ensure a comprehensive evaluation, we introduce MTJ-Pro, a challenging multi-turn progressive jailbreak dataset that combines seven advanced jailbreak strategies designed to gradually deepen attack strategies across multi-turn attacks. Besides, we present two novel metrics: Mislead Success Rate (MSR) and Attack Resource Consumption (ARC), which provide more nuanced assessments of deceptive defense beyond conventional measures. Experimental results on GPT-4, GPT-3.5-turbo, Gemini-1.5-pro, and LLaMa-3.1 demonstrate that HoneyTrap achieves an average reduction of 68.77% in attack success rates compared to state-of-the-art baselines. Notably, even in a dedicated adaptive attacker setting with intensified conditions, HoneyTrap remains resilient, leveraging deceptive engagement to prolong interactions, significantly increasing the time and computational costs required for successful exploitation. Unlike simple rejection, HoneyTrap strategically wastes attacker resources without impacting benign queries, improving MSR and ARC by 118.11% and 149.16%, respectively. 

---
# A Scheduling Framework for Efficient MoE Inference on Edge GPU-NDP Systems 

**Authors**: Qi Wu, Chao Fang, Jiayuan Chen, Ye Lin, Yueqi Zhang, Yichuan Bai, Yuan Du, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2601.03992)  

**Abstract**: Mixture-of-Experts (MoE) models facilitate edge deployment by decoupling model capacity from active computation, yet their large memory footprint drives the need for GPU systems with near-data processing (NDP) capabilities that offload experts to dedicated processing units. However, deploying MoE models on such edge-based GPU-NDP systems faces three critical challenges: 1) severe load imbalance across NDP units due to non-uniform expert selection and expert parallelism, 2) insufficient GPU utilization during expert computation within NDP units, and 3) extensive data pre-profiling necessitated by unpredictable expert activation patterns for pre-fetching. To address these challenges, this paper proposes an efficient inference framework featuring three key optimizations. First, the underexplored tensor parallelism in MoE inference is exploited to partition and compute large expert parameters across multiple NDP units simultaneously towards edge low-batch scenarios. Second, a load-balancing-aware scheduling algorithm distributes expert computations across NDP units and GPU to maximize resource utilization. Third, a dataset-free pre-fetching strategy proactively loads frequently accessed experts to minimize activation delays. Experimental results show that our framework enables GPU-NDP systems to achieve 2.41x on average and up to 2.56x speedup in end-to-end latency compared to state-of-the-art approaches, significantly enhancing MoE inference efficiency in resource-constrained environments. 

---
# Large-Scale Aspect-Based Sentiment Analysis with Reasoning-Infused LLMs 

**Authors**: Paweł Liskowski, Krzysztof Jankowski  

**Link**: [PDF](https://arxiv.org/pdf/2601.03940)  

**Abstract**: We introduce Arctic-ABSA, a collection of powerful models for real-life aspect-based sentiment analysis (ABSA). Our models are tailored to commercial needs, trained on a large corpus of public data alongside carefully generated synthetic data, resulting in a dataset 20 times larger than SemEval14. We extend typical ABSA models by expanding the number of sentiment classes from the standard three (positive, negative, neutral) to five, adding mixed and unknown classes, while also jointly predicting overall text sentiment and supporting multiple languages. We experiment with reasoning injection by fine-tuning on Chain-of-Thought (CoT) examples and introduce a novel reasoning pretraining technique for encoder-only models that significantly improves downstream fine-tuning and generalization. Our 395M-parameter encoder and 8B-parameter decoder achieve up to 10 percentage points higher accuracy than GPT-4o and Claude 3.5 Sonnet, while setting new state-of-the-art results on the SemEval14 benchmark. A single multilingual model maintains 87-91% accuracy across six languages without degrading English performance. We release ABSA-mix, a large-scale benchmark aggregating 17 public ABSA datasets across 92 domains. 

---
# FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning 

**Authors**: Yujie Feng, Hao Wang, Jian Li, Xu Chu, Zhaolu Kang, Yiran Liu, Yasha Wang, Philip S. Yu, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03938)  

**Abstract**: Continual learning (CL) for large language models (LLMs) aims to enable sequential knowledge acquisition without catastrophic forgetting. Memory replay methods are widely used for their practicality and effectiveness, but most rely on fixed, step-based heuristics that often misalign with the model's actual learning progress, since identical training steps can result in varying degrees of parameter change. Motivated by recent findings that LLM forgetting mirrors the Ebbinghaus human forgetting curve, we propose FOREVER (FORgEtting curVe-inspired mEmory Replay), a novel CL framework that aligns replay schedules with a model-centric notion of time. FOREVER defines model time using the magnitude of optimizer updates, allowing forgetting curve-inspired replay intervals to align with the model's internal evolution rather than raw training steps. Building on this approach, FOREVER incorporates a forgetting curve-based replay scheduler to determine when to replay and an intensity-aware regularization mechanism to adaptively control how to replay. Extensive experiments on three CL benchmarks and models ranging from 0.6B to 13B parameters demonstrate that FOREVER consistently mitigates catastrophic forgetting. 

---
# Bayes-PD: Exploring a Sequence to Binding Bayesian Neural Network model trained on Phage Display data 

**Authors**: Ilann Amiaud-Plachy, Michael Blank, Oliver Bent, Sebastien Boyer  

**Link**: [PDF](https://arxiv.org/pdf/2601.03930)  

**Abstract**: Phage display is a powerful laboratory technique used to study the interactions between proteins and other molecules, whether other proteins, peptides, DNA or RNA. The under-utilisation of this data in conjunction with deep learning models for protein design may be attributed to; high experimental noise levels; the complex nature of data pre-processing; and difficulty interpreting these experimental results. In this work, we propose a novel approach utilising a Bayesian Neural Network within a training loop, in order to simulate the phage display experiment and its associated noise. Our goal is to investigate how understanding the experimental noise and model uncertainty can enable the reliable application of such models to reliably interpret phage display experiments. We validate our approach using actual binding affinity measurements instead of relying solely on proxy values derived from 'held-out' phage display rounds. 

---
# FocusUI: Efficient UI Grounding via Position-Preserving Visual Token Selection 

**Authors**: Mingyu Ouyang, Kevin Qinghong Lin, Mike Zheng Shou, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2601.03928)  

**Abstract**: Vision-Language Models (VLMs) have shown remarkable performance in User Interface (UI) grounding tasks, driven by their ability to process increasingly high-resolution screenshots. However, screenshots are tokenized into thousands of visual tokens (e.g., about 4700 for 2K resolution), incurring significant computational overhead and diluting attention. In contrast, humans typically focus on regions of interest when interacting with UI. In this work, we pioneer the task of efficient UI grounding. Guided by practical analysis of the task's characteristics and challenges, we propose FocusUI, an efficient UI grounding framework that selects patches most relevant to the instruction while preserving positional continuity for precise grounding. FocusUI addresses two key challenges: (1) Eliminating redundant tokens in visual encoding. We construct patch-level supervision by fusing an instruction-conditioned score with a rule-based UI-graph score that down-weights large homogeneous regions to select distinct and instruction-relevant visual tokens. (2) Preserving positional continuity during visual token selection. We find that general visual token pruning methods suffer from severe accuracy degradation on UI grounding tasks due to broken positional information. We introduce a novel PosPad strategy, which compresses each contiguous sequence of dropped visual tokens into a single special marker placed at the sequence's last index to preserve positional continuity. Comprehensive experiments on four grounding benchmarks demonstrate that FocusUI surpasses GUI-specific baselines. On the ScreenSpot-Pro benchmark, FocusUI-7B achieves a performance improvement of 3.7% over GUI-Actor-7B. Even with only 30% visual token retention, FocusUI-7B drops by only 3.2% while achieving up to 1.44x faster inference and 17% lower peak GPU memory. 

---
# A Gap Between Decision Trees and Neural Networks 

**Authors**: Akash Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.03919)  

**Abstract**: We study when geometric simplicity of decision boundaries, used here as a notion of interpretability, can conflict with accurate approximation of axis-aligned decision trees by shallow neural networks. Decision trees induce rule-based, axis-aligned decision regions (finite unions of boxes), whereas shallow ReLU networks are typically trained as score models whose predictions are obtained by thresholding. We analyze the infinite-width, bounded-norm, single-hidden-layer ReLU class through the Radon total variation ($\mathrm{R}\mathrm{TV}$) seminorm, which controls the geometric complexity of level sets.
We first show that the hard tree indicator $1_A$ has infinite $\mathrm{R}\mathrm{TV}$. Moreover, two natural split-wise continuous surrogates--piecewise-linear ramp smoothing and sigmoidal (logistic) smoothing--also have infinite $\mathrm{R}\mathrm{TV}$ in dimensions $d>1$, while Gaussian convolution yields finite $\mathrm{R}\mathrm{TV}$ but with an explicit exponential dependence on $d$.
We then separate two goals that are often conflated: classification after thresholding (recovering the decision set) versus score learning (learning a calibrated score close to $1_A$). For classification, we construct a smooth barrier score $S_A$ with finite $\mathrm{R}\mathrm{TV}$ whose fixed threshold $\tau=1$ exactly recovers the box. Under a mild tube-mass condition near $\partial A$, we prove an $L_1(P)$ calibration bound that decays polynomially in a sharpness parameter, along with an explicit $\mathrm{R}\mathrm{TV}$ upper bound in terms of face measures. Experiments on synthetic unions of rectangles illustrate the resulting accuracy--complexity tradeoff and how threshold selection shifts where training lands along it. 

---
# An Algebraic Representation Theorem for Linear GENEOs in Geometric Machine Learning 

**Authors**: Francesco Conti, Patrizio Frosini, Nicola Quercioli  

**Link**: [PDF](https://arxiv.org/pdf/2601.03910)  

**Abstract**: Geometric and Topological Deep Learning are rapidly growing research areas that enhance machine learning through the use of geometric and topological structures. Within this framework, Group Equivariant Non-Expansive Operators (GENEOs) have emerged as a powerful class of operators for encoding symmetries and designing efficient, interpretable neural architectures. Originally introduced in Topological Data Analysis, GENEOs have since found applications in Deep Learning as tools for constructing equivariant models with reduced parameter complexity. GENEOs provide a unifying framework bridging Geometric and Topological Deep Learning and include the operator computing persistence diagrams as a special case. Their theoretical foundations rely on group actions, equivariance, and compactness properties of operator spaces, grounding them in algebra and geometry while enabling both mathematical rigor and practical relevance. While a previous representation theorem characterized linear GENEOs acting on data of the same type, many real-world applications require operators between heterogeneous data spaces. In this work, we address this limitation by introducing a new representation theorem for linear GENEOs acting between different perception pairs, based on generalized T-permutant measures. Under mild assumptions on the data domains and group actions, our result provides a complete characterization of such operators. We also prove the compactness and convexity of the space of linear GENEOs. We further demonstrate the practical impact of this theory by applying the proposed framework to improve the performance of autoencoders, highlighting the relevance of GENEOs in modern machine learning applications. 

---
# Adaptive-Boundary-Clipping GRPO: Ensuring Bounded Ratios for Stable and Generalizable Training 

**Authors**: Chi Liu, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.03895)  

**Abstract**: Group Relative Policy Optimization (GRPO) has emerged as a popular algorithm for reinforcement learning with large language models (LLMs). However, upon analyzing its clipping mechanism, we argue that it is suboptimal in certain scenarios. With appropriate modifications, GRPO can be significantly enhanced to improve both flexibility and generalization. To this end, we propose Adaptive-Boundary-Clipping GRPO (ABC-GRPO), an asymmetric and adaptive refinement of the original GRPO framework. We demonstrate that ABC-GRPO achieves superior performance over standard GRPO on mathematical reasoning tasks using the Qwen3 LLMs. Moreover, ABC-GRPO maintains substantially higher entropy throughout training, thereby preserving the model's exploration capacity and mitigating premature convergence. The implementation code is available online to ease reproducibility this https URL. 

---
# Spectral Manifold Regularization for Stable and Modular Routing in Deep MoE Architectures 

**Authors**: Ibrahim Delibasoglu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03889)  

**Abstract**: Mixture of Experts (MoE) architectures enable efficient scaling of neural networks but suffer from expert collapse, where routing converges to a few dominant experts. This reduces model capacity and causes catastrophic interference during adaptation. We propose the Spectrally-Regularized Mixture of Experts (SR-MoE), which imposes geometric constraints on the routing manifold to enforce structural modularity. Our method uses dual regularization: spectral norm constraints bound routing function Lipschitz continuity, while stable rank penalties preserve high-dimensional feature diversity in expert selection. We evaluate SR-MoE across architectural scales and dataset complexities using modular one-shot adaptation tasks. Results show that traditional linear gating fails with increasing depth (accuracy drops up to 4.72% due to expert entanglement), while SR-MoE maintains structural integrity (mean interference -0.32%). Our spectral constraints facilitate positive knowledge transfer, enabling localized expert updates without global performance decay. SR-MoE provides a general solution for building high-capacity, modular networks capable of stable lifelong learning. 

---
# IndexTTS 2.5 Technical Report 

**Authors**: Yunpei Li, Xun Zhou, Jinchao Wang, Lu Wang, Yong Wu, Siyi Zhou, Yiquan Zhou, Jingchen Shu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03888)  

**Abstract**: In prior work, we introduced IndexTTS 2, a zero-shot neural text-to-speech foundation model comprising two core components: a transformer-based Text-to-Semantic (T2S) module and a non-autoregressive Semantic-to-Mel (S2M) module, which together enable faithful emotion replication and establish the first autoregressive duration-controllable generative paradigm. Building upon this, we present IndexTTS 2.5, which significantly enhances multilingual coverage, inference speed, and overall synthesis quality through four key improvements: 1) Semantic Codec Compression: we reduce the semantic codec frame rate from 50 Hz to 25 Hz, halving sequence length and substantially lowering both training and inference costs; 2) Architectural Upgrade: we replace the U-DiT-based backbone of the S2M module with a more efficient Zipformer-based modeling architecture, achieving notable parameter reduction and faster mel-spectrogram generation; 3) Multilingual Extension: We propose three explicit cross-lingual modeling strategies, boundary-aware alignment, token-level concatenation, and instruction-guided generation, establishing practical design principles for zero-shot multilingual emotional TTS that supports Chinese, English, Japanese, and Spanish, and enables robust emotion transfer even without target-language emotional training data; 4) Reinforcement Learning Optimization: we apply GRPO in post-training of the T2S module, improving pronunciation accuracy and natrualness. Experiments show that IndexTTS 2.5 not only supports broader language coverage but also replicates emotional prosody in unseen languages under the same zero-shot setting. IndexTTS 2.5 achieves a 2.28 times improvement in RTF while maintaining comparable WER and speaker similarity to IndexTTS 2. 

---
# FLNet: Flood-Induced Agriculture Damage Assessment using Super Resolution of Satellite Images 

**Authors**: Sanidhya Ghosal, Anurag Sharma, Sushil Ghildiyal, Mukesh Saini  

**Link**: [PDF](https://arxiv.org/pdf/2601.03884)  

**Abstract**: Distributing government relief efforts after a flood is challenging. In India, the crops are widely affected by floods; therefore, making rapid and accurate crop damage assessment is crucial for effective post-disaster agricultural management. Traditional manual surveys are slow and biased, while current satellite-based methods face challenges like cloud cover and low spatial resolution. Therefore, to bridge this gap, this paper introduced FLNet, a novel deep learning based architecture that used super-resolution to enhance the 10 m spatial resolution of Sentinel-2 satellite images into 3 m resolution before classifying damage. We tested our model on the Bihar Flood Impacted Croplands Dataset (BFCD-22), and the results showed an improved critical "Full Damage" F1-score from 0.83 to 0.89, nearly matching the 0.89 score of commercial high-resolution imagery. This work presented a cost-effective and scalable solution, paving the way for a nationwide shift from manual to automated, high-fidelity damage assessment. 

---
# Women Worry, Men Adopt: How Gendered Perceptions Shape the Use of Generative AI 

**Authors**: Fabian Stephany, Jedrzej Duszynski  

**Link**: [PDF](https://arxiv.org/pdf/2601.03880)  

**Abstract**: Generative artificial intelligence (GenAI) is diffusing rapidly, yet its adoption is strikingly unequal. Using nationally representative UK survey data from 2023 to 2024, we show that women adopt GenAI substantially less often than men because they perceive its societal risks differently. We construct a composite index capturing concerns about mental health, privacy, climate impact, and labor market disruption. This index explains between 9 and 18 percent of the variation in GenAI adoption and ranks among the strongest predictors for women across all age groups, surpassing digital literacy and education for young women. Intersectional analyses show that the largest disparities arise among younger, digitally fluent individuals with high societal risk concerns, where gender gaps in personal use exceed 45 percentage points. Using a synthetic twin panel design, we show that increased optimism about AI's societal impact raises GenAI use among young women from 13 percent to 33 percent, substantially narrowing the gender divide. These findings indicate that gendered perceptions of AI's social and ethical consequences, rather than access or capability, are the primary drivers of unequal GenAI adoption, with implications for productivity, skill formation, and economic inequality in an AI enabled economy. 

---
# What Matters For Safety Alignment? 

**Authors**: Xing Li, Hui-Ling Zhen, Lihao Yin, Xianzhi Yu, Zhenhua Dong, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2601.03868)  

**Abstract**: This paper presents a comprehensive empirical study on the safety alignment capabilities. We evaluate what matters for safety alignment in LLMs and LRMs to provide essential insights for developing more secure and reliable AI systems. We systematically investigate and compare the influence of six critical intrinsic model characteristics and three external attack techniques. Our large-scale evaluation is conducted using 32 recent, popular LLMs and LRMs across thirteen distinct model families, spanning a parameter scale from 3B to 235B. The assessment leverages five established safety datasets and probes model vulnerabilities with 56 jailbreak techniques and four CoT attack strategies, resulting in 4.6M API calls. Our key empirical findings are fourfold. First, we identify the LRMs GPT-OSS-20B, Qwen3-Next-80B-A3B-Thinking, and GPT-OSS-120B as the top-three safest models, which substantiates the significant advantage of integrated reasoning and self-reflection mechanisms for robust safety alignment. Second, post-training and knowledge distillation may lead to a systematic degradation of safety alignment. We thus argue that safety must be treated as an explicit constraint or a core optimization objective during these stages, not merely subordinated to the pursuit of general capability. Third, we reveal a pronounced vulnerability: employing a CoT attack via a response prefix can elevate the attack success rate by 3.34x on average and from 0.6% to 96.3% for Seed-OSS-36B-Instruct. This critical finding underscores the safety risks inherent in text-completion interfaces and features that allow user-defined response prefixes in LLM services, highlighting an urgent need for architectural and deployment safeguards. Fourth, roleplay, prompt injection, and gradient-based search for adversarial prompts are the predominant methodologies for eliciting unaligned behaviors in modern models. 

---
# Implementing the First-Order Logic of Here and There 

**Authors**: Jens Otten, Torsten Schaub  

**Link**: [PDF](https://arxiv.org/pdf/2601.03848)  

**Abstract**: We present automated theorem provers for the first-order logic of here and there (HT). They are based on a native sequent calculus for the logic of HT and an axiomatic embedding of the logic of HT into intuitionistic logic. The analytic proof search in the sequent calculus is optimized by using free variables and skolemization. The embedding is used in combination with sequent, tableau and connection calculi for intuitionistic first-order logic. All provers are evaluated on a large benchmark set of first-order formulas, providing a foundation for the development of more efficient HT provers. 

---
# When Numbers Start Talking: Implicit Numerical Coordination Among LLM-Based Agents 

**Authors**: Alessio Buscemi, Daniele Proverbio, Alessandro Di Stefano, Anh Han, German Castignani, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2601.03846)  

**Abstract**: LLMs-based agents increasingly operate in multi-agent environments where strategic interaction and coordination are required. While existing work has largely focused on individual agents or on interacting agents sharing explicit communication, less is known about how interacting agents coordinate implicitly. In particular, agents may engage in covert communication, relying on indirect or non-linguistic signals embedded in their actions rather than on explicit messages. This paper presents a game-theoretic study of covert communication in LLM-driven multi-agent systems. We analyse interactions across four canonical game-theoretic settings under different communication regimes, including explicit, restricted, and absent communication. Considering heterogeneous agent personalities and both one-shot and repeated games, we characterise when covert signals emerge and how they shape coordination and strategic outcomes. 

---
# On the Trap Space Semantics of Normal Logic Programs 

**Authors**: Van-Giang Trinh, Sylvain Soliman, François Fages, Belaid Benhamou  

**Link**: [PDF](https://arxiv.org/pdf/2601.03842)  

**Abstract**: The logical semantics of normal logic programs has traditionally been based on the notions of Clark's completion and two-valued or three-valued canonical models, including supported, stable, regular, and well-founded models. Two-valued interpretations can also be seen as states evolving under a program's update operator, producing a transition graph whose fixed points and cycles capture stable and oscillatory behaviors, respectively. We refer to this view as dynamical semantics since it characterizes the program's meaning in terms of state-space trajectories, as first introduced in the stable (supported) class semantics. Recently, we have established a formal connection between Datalog^\neg programs (i.e., normal logic programs without function symbols) and Boolean networks, leading to the introduction of the trap space concept for Datalog^\neg programs. In this paper, we generalize the trap space concept to arbitrary normal logic programs, introducing trap space semantics as a new approach to their interpretation. This new semantics admits both model-theoretic and dynamical characterizations, providing a comprehensive approach to understanding program behavior. We establish the foundational properties of the trap space semantics and systematically relate it to the established model-theoretic semantics, including the stable (supported), stable (supported) partial, regular, and L-stable model semantics, as well as to the dynamical stable (supported) class semantics. Our results demonstrate that the trap space semantics offers a unified and precise framework for proving the existence of supported classes, strict stable (supported) classes, and regular models, in addition to uncovering and formalizing deeper relationships among the existing semantics of normal logic programs. 

---
# Logic Tensor Network-Enhanced Generative Adversarial Network 

**Authors**: Nijesh Upreti, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2601.03839)  

**Abstract**: In this paper, we introduce Logic Tensor Network-Enhanced Generative Adversarial Network (LTN-GAN), a novel framework that enhances Generative Adversarial Networks (GANs) by incorporating Logic Tensor Networks (LTNs) to enforce domain-specific logical constraints during the sample generation process. Although GANs have shown remarkable success in generating realistic data, they often lack mechanisms to incorporate prior knowledge or enforce logical consistency, limiting their applicability in domains requiring rule adherence. LTNs provide a principled way to integrate first-order logic with neural networks, enabling models to reason over and satisfy logical constraints. By combining the strengths of GANs for realistic data synthesis with LTNs for logical reasoning, we gain valuable insights into how logical constraints influence the generative process while improving both the diversity and logical consistency of the generated samples. We evaluate LTN-GAN across multiple datasets, including synthetic datasets (gaussian, grid, rings) and the MNIST dataset, demonstrating that our model significantly outperforms traditional GANs in terms of adherence to predefined logical constraints while maintaining the quality and diversity of generated samples. This work highlights the potential of neuro-symbolic approaches to enhance generative modeling in knowledge-intensive domains. 

---
# IDESplat: Iterative Depth Probability Estimation for Generalizable 3D Gaussian Splatting 

**Authors**: Wei Long, Haifeng Wu, Shiyin Jiang, Jinhua Zhang, Xinchun Ji, Shuhang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03824)  

**Abstract**: Generalizable 3D Gaussian Splatting aims to directly predict Gaussian parameters using a feed-forward network for scene reconstruction. Among these parameters, Gaussian means are particularly difficult to predict, so depth is usually estimated first and then unprojected to obtain the Gaussian sphere centers. Existing methods typically rely solely on a single warp to estimate depth probability, which hinders their ability to fully leverage cross-view geometric cues, resulting in unstable and coarse depth maps. To address this limitation, we propose IDESplat, which iteratively applies warp operations to boost depth probability estimation for accurate Gaussian mean prediction. First, to eliminate the inherent instability of a single warp, we introduce a Depth Probability Boosting Unit (DPBU) that integrates epipolar attention maps produced by cascading warp operations in a multiplicative manner. Next, we construct an iterative depth estimation process by stacking multiple DPBUs, progressively identifying potential depth candidates with high likelihood. As IDESplat iteratively boosts depth probability estimates and updates the depth candidates, the depth map is gradually refined, resulting in accurate Gaussian means. We conduct experiments on RealEstate10K, ACID, and DL3DV. IDESplat achieves outstanding reconstruction quality and state-of-the-art performance with real-time efficiency. On RE10K, it outperforms DepthSplat by 0.33 dB in PSNR, using only 10.7% of the parameters and 70% of the memory. Additionally, our IDESplat improves PSNR by 2.95 dB over DepthSplat on the DTU dataset in cross-dataset experiments, demonstrating its strong generalization ability. 

---
# AI Generated Text Detection 

**Authors**: Adilkhan Alikhanov, Aidar Amangeldi, Diar Demeubay, Dilnaz Akhmetzhan, Nurbek Moldakhmetov, Omar Polat, Galymzhan Zharas  

**Link**: [PDF](https://arxiv.org/pdf/2601.03812)  

**Abstract**: The rapid development of large language models has led to an increase in AI-generated text, with students increasingly using LLM-generated content as their own work, which violates academic integrity. This paper presents an evaluation of AI text detection methods, including both traditional machine learning models and transformer-based architectures. We utilize two datasets, HC3 and DAIGT v2, to build a unified benchmark and apply a topic-based data split to prevent information leakage. This approach ensures robust generalization across unseen domains. Our experiments show that TF-IDF logistic regression achieves a reasonable baseline accuracy of 82.87%. However, deep learning models outperform it. The BiLSTM classifier achieves an accuracy of 88.86%, while DistilBERT achieves a similar accuracy of 88.11% with the highest ROC-AUC score of 0.96, demonstrating the strongest overall performance. The results indicate that contextual semantic modeling is significantly superior to lexical features and highlight the importance of mitigating topic memorization through appropriate evaluation protocols. The limitations of this work are primarily related to dataset diversity and computational constraints. In future work, we plan to expand dataset diversity and utilize parameter-efficient fine-tuning methods such as LoRA. We also plan to explore smaller or distilled models and employ more efficient batching strategies and hardware-aware optimization. 

---
# Where meaning lives: Layer-wise accessibility of psycholinguistic features in encoder and decoder language models 

**Authors**: Taisiia Tikhomirova, Dirk U. Wulff  

**Link**: [PDF](https://arxiv.org/pdf/2601.03798)  

**Abstract**: Understanding where transformer language models encode psychologically meaningful aspects of meaning is essential for both theory and practice. We conduct a systematic layer-wise probing study of 58 psycholinguistic features across 10 transformer models, spanning encoder-only and decoder-only architectures, and compare three embedding extraction methods. We find that apparent localization of meaning is strongly method-dependent: contextualized embeddings yield higher feature-specific selectivity and different layer-wise profiles than isolated embeddings. Across models and methods, final-layer representations are rarely optimal for recovering psycholinguistic information with linear probes. Despite these differences, models exhibit a shared depth ordering of meaning dimensions, with lexical properties peaking earlier and experiential and affective dimensions peaking later. Together, these results show that where meaning "lives" in transformer models reflects an interaction between methodological choices and architectural constraints. 

---
# An Algorithmic Framework for Systematic Literature Reviews: A Case Study for Financial Narratives 

**Authors**: Gabin Taibi, Joerg Osterrieder  

**Link**: [PDF](https://arxiv.org/pdf/2601.03794)  

**Abstract**: This paper introduces an algorithmic framework for conducting systematic literature reviews (SLRs), designed to improve efficiency, reproducibility, and selection quality assessment in the literature review process. The proposed method integrates Natural Language Processing (NLP) techniques, clustering algorithms, and interpretability tools to automate and structure the selection and analysis of academic publications. The framework is applied to a case study focused on financial narratives, an emerging area in financial economics that examines how structured accounts of economic events, formed by the convergence of individual interpretations, influence market dynamics and asset prices. Drawing from the Scopus database of peer-reviewed literature, the review highlights research efforts to model financial narratives using various NLP techniques. Results reveal that while advances have been made, the conceptualization of financial narratives remains fragmented, often reduced to sentiment analysis, topic modeling, or their combination, without a unified theoretical framework. The findings underscore the value of more rigorous and dynamic narrative modeling approaches and demonstrate the effectiveness of the proposed algorithmic SLR methodology. 

---
# Do LLMs Really Memorize Personally Identifiable Information? Revisiting PII Leakage with a Cue-Controlled Memorization Framework 

**Authors**: Xiaoyu Luo, Yiyi Chen, Qiongxiu Li, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2601.03791)  

**Abstract**: Large Language Models (LLMs) have been reported to "leak" Personally Identifiable Information (PII), with successful PII reconstruction often interpreted as evidence of memorization. We propose a principled revision of memorization evaluation for LLMs, arguing that PII leakage should be evaluated under low lexical cue conditions, where target PII cannot be reconstructed through prompt-induced generalization or pattern completion. We formalize Cue-Resistant Memorization (CRM) as a cue-controlled evaluation framework and a necessary condition for valid memorization evaluation, explicitly conditioning on prompt-target overlap cues. Using CRM, we conduct a large-scale multilingual re-evaluation of PII leakage across 32 languages and multiple memorization paradigms. Revisiting reconstruction-based settings, including verbatim prefix-suffix completion and associative reconstruction, we find that their apparent effectiveness is driven primarily by direct surface-form cues rather than by true memorization. When such cues are controlled for, reconstruction success diminishes substantially. We further examine cue-free generation and membership inference, both of which exhibit extremely low true positive rates. Overall, our results suggest that previously reported PII leakage is better explained by cue-driven behavior than by genuine memorization, highlighting the importance of cue-controlled evaluation for reliably quantifying privacy-relevant memorization in LLMs. 

---
# NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning 

**Authors**: Zhongtao Miao, Kaiyan Zhao, Masaaki Nagata, Yoshimasa Tsuruoka  

**Link**: [PDF](https://arxiv.org/pdf/2601.03790)  

**Abstract**: Neologism-aware machine translation aims to translate source sentences containing neologisms into target languages. This field remains underexplored compared with general machine translation (MT). In this paper, we propose an agentic framework, NeoAMT, for neologism-aware machine translation using a Wiktionary search tool. Specifically, we first create a new dataset for neologism-aware machine translation and develop a search tool based on Wiktionary. The new dataset covers 16 languages and 75 translation directions and is derived from approximately 10 million records of an English Wiktionary dump. The retrieval corpus of the search tool is also constructed from around 3 million cleaned records of the Wiktionary dump. We then use it for training the translation agent with reinforcement learning (RL) and evaluating the accuracy of neologism-aware machine translation. Based on this, we also propose an RL training framework that contains a novel reward design and an adaptive rollout generation approach by leveraging "translation difficulty" to further improve the translation quality of translation agents using our search tool. 

---
# Criminal Liability of Generative Artificial Intelligence Providers for User-Generated Child Sexual Abuse Material 

**Authors**: Anamaria Mojica-Hanke, Thomas Goger, Svenja Wölfel, Brian Valerius, Steffen Herbold  

**Link**: [PDF](https://arxiv.org/pdf/2601.03788)  

**Abstract**: The development of more powerful Generative Artificial Intelligence (GenAI) has expanded its capabilities and the variety of outputs. This has introduced significant legal challenges, including gray areas in various legal systems, such as the assessment of criminal liability for those responsible for these models. Therefore, we conducted a multidisciplinary study utilizing the statutory interpretation of relevant German laws, which, in conjunction with scenarios, provides a perspective on the different properties of GenAI in the context of Child Sexual Abuse Material (CSAM) generation. We found that generating CSAM with GenAI may have criminal and legal consequences not only for the user committing the primary offense but also for individuals responsible for the models, such as independent software developers, researchers, and company representatives. Additionally, the assessment of criminal liability may be affected by contextual and technical factors, including the type of generated image, content moderation policies, and the model's intended purpose. Based on our findings, we discussed the implications for different roles, as well as the requirements when developing such systems. 

---
# Membox: Weaving Topic Continuity into Long-Range Memory for LLM Agents 

**Authors**: Dehao Tao, Guoliang Ma, Yongfeng Huang, Minghu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03785)  

**Abstract**: Human-agent dialogues often exhibit topic continuity-a stable thematic frame that evolves through temporally adjacent exchanges-yet most large language model (LLM) agent memory systems fail to preserve it. Existing designs follow a fragmentation-compensation paradigm: they first break dialogue streams into isolated utterances for storage, then attempt to restore coherence via embedding-based retrieval. This process irreversibly damages narrative and causal flow, while biasing retrieval towards lexical similarity. We introduce membox, a hierarchical memory architecture centered on a Topic Loom that continuously monitors dialogue in a sliding-window fashion, grouping consecutive same-topic turns into coherent "memory boxes" at storage time. Sealed boxes are then linked by a Trace Weaver into long-range event-timeline traces, recovering macro-topic recurrences across discontinuities. Experiments on LoCoMo demonstrate that Membox achieves up to 68% F1 improvement on temporal reasoning tasks, outperforming competitive baselines (e.g., Mem0, A-MEM). Notably, Membox attains these gains while using only a fraction of the context tokens required by existing methods, highlighting a superior balance between efficiency and effectiveness. By explicitly modeling topic continuity, Membox offers a cognitively motivated mechanism for enhancing both coherence and efficiency in LLM agents. 

---
# PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation 

**Authors**: Wenlong Huang, Yu-Wei Chao, Arsalan Mousavian, Ming-Yu Liu, Dieter Fox, Kaichun Mo, Li Fei-Fei  

**Link**: [PDF](https://arxiv.org/pdf/2601.03782)  

**Abstract**: Humans anticipate, from a glance and a contemplated action of their bodies, how the 3D world will respond, a capability that is equally vital for robotic manipulation. We introduce PointWorld, a large pre-trained 3D world model that unifies state and action in a shared 3D space as 3D point flows: given one or few RGB-D images and a sequence of low-level robot action commands, PointWorld forecasts per-pixel displacements in 3D that respond to the given actions. By representing actions as 3D point flows instead of embodiment-specific action spaces (e.g., joint positions), this formulation directly conditions on physical geometries of robots while seamlessly integrating learning across embodiments. To train our 3D world model, we curate a large-scale dataset spanning real and simulated robotic manipulation in open-world environments, enabled by recent advances in 3D vision and simulated environments, totaling about 2M trajectories and 500 hours across a single-arm Franka and a bimanual humanoid. Through rigorous, large-scale empirical studies of backbones, action representations, learning objectives, partial observability, data mixtures, domain transfers, and scaling, we distill design principles for large-scale 3D world modeling. With a real-time (0.1s) inference speed, PointWorld can be efficiently integrated in the model-predictive control (MPC) framework for manipulation. We demonstrate that a single pre-trained checkpoint enables a real-world Franka robot to perform rigid-body pushing, deformable and articulated object manipulation, and tool use, without requiring any demonstrations or post-training and all from a single image captured in-the-wild. Project website at this https URL. 

---
# Scalable Machine Learning Force Fields for Macromolecular Systems Through Long-Range Aware Message Passing 

**Authors**: Chu Wang, Lin Huang, Xinran Wei, Tao Qin, Arthur Jiang, Lixue Cheng, Jia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03774)  

**Abstract**: Machine learning force fields (MLFFs) have revolutionized molecular simulations by providing quantum mechanical accuracy at the speed of molecular mechanical computations. However, a fundamental reliance of these models on fixed-cutoff architectures limits their applicability to macromolecular systems where long-range interactions dominate. We demonstrate that this locality constraint causes force prediction errors to scale monotonically with system size, revealing a critical architectural bottleneck. To overcome this, we establish the systematically designed MolLR25 ({Mol}ecules with {L}ong-{R}ange effect) benchmark up to 1200 atoms, generated using high-fidelity DFT, and introduce E2Former-LSR, an equivariant transformer that explicitly integrates long-range attention blocks. E2Former-LSR exhibits stable error scaling, achieves superior fidelity in capturing non-covalent decay, and maintains precision on complex protein conformations. Crucially, its efficient design provides up to 30% speedup compared to purely local models. This work validates the necessity of non-local architectures for generalizable MLFFs, enabling high-fidelity molecular dynamics for large-scale chemical and biological systems. 

---
# Learning Shrinks the Hard Tail: Training-Dependent Inference Scaling in a Solvable Linear Model 

**Authors**: Noam Levi  

**Link**: [PDF](https://arxiv.org/pdf/2601.03764)  

**Abstract**: We analyze neural scaling laws in a solvable model of last-layer fine-tuning where targets have intrinsic, instance-heterogeneous difficulty. In our Latent Instance Difficulty (LID) model, each input's target variance is governed by a latent ``precision'' drawn from a heavy-tailed distribution. While generalization loss recovers standard scaling laws, our main contribution connects this to inference. The pass@$k$ failure rate exhibits a power-law decay, $k^{-\beta_\text{eff}}$, but the observed exponent $\beta_\text{eff}$ is training-dependent. It grows with sample size $N$ before saturating at an intrinsic limit $\beta$ set by the difficulty distribution's tail. This coupling reveals that learning shrinks the ``hard tail'' of the error distribution: improvements in the model's generalization error steepen the pass@$k$ curve until irreducible target variance dominates. The LID model yields testable, closed-form predictions for this behavior, including a compute-allocation rule that favors training before saturation and inference attempts after. We validate these predictions in simulations and in two real-data proxies: CIFAR-10H (human-label variance) and a maths teacher-student distillation task. 

---
# Evaluation of Multilingual LLMs Personalized Text Generation Capabilities Targeting Groups and Social-Media Platforms 

**Authors**: Dominik Macko  

**Link**: [PDF](https://arxiv.org/pdf/2601.03752)  

**Abstract**: Capabilities of large language models to generate multilingual coherent text have continuously enhanced in recent years, which opens concerns about their potential misuse. Previous research has shown that they can be misused for generation of personalized disinformation in multiple languages. It has also been observed that personalization negatively affects detectability of machine-generated texts; however, this has been studied in the English language only. In this work, we examine this phenomenon across 10 languages, while we focus not only on potential misuse of personalization capabilities, but also on potential benefits they offer. Overall, we cover 1080 combinations of various personalization aspects in the prompts, for which the texts are generated by 16 distinct language models (17,280 texts in total). Our results indicate that there are differences in personalization quality of the generated texts when targeting demographic groups and when targeting social-media platforms across languages. Personalization towards platforms affects detectability of the generated texts in a higher scale, especially in English, where the personalization quality is the highest. 

---
# Bridging OLAP and RAG: A Multidimensional Approach to the Design of Corpus Partitioning 

**Authors**: Dario Maio, Stefano Rizzi  

**Link**: [PDF](https://arxiv.org/pdf/2601.03748)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are increasingly deployed on large-scale document collections, often comprising millions of documents and tens of millions of text chunks. In industrial-scale retrieval platforms, scalability is typically addressed through horizontal sharding and a combination of Approximate Nearest-Neighbor search, hybrid indexing, and optimized metadata filtering. Although effective from an efficiency perspective, these mechanisms rely on bottom-up, similarity-driven organization and lack a conceptual rationale for corpus partitioning. In this paper, we claim that the design of large-scale RAG systems may benefit from the combination of two orthogonal strategies: semantic clustering, which optimizes locality in embedding space, and multidimensional partitioning, which governs where retrieval should occur based on conceptual dimensions such as time and organizational context. Although such dimensions are already implicitly present in current systems, they are used in an ad hoc and poorly structured manner. We propose the Dimensional Fact Model (DFM) as a conceptual framework to guide the design of multidimensional partitions for RAG corpora. The DFM provides a principled way to reason about facts, dimensions, hierarchies, and granularity in retrieval-oriented settings. This framework naturally supports hierarchical routing and controlled fallback strategies, ensuring that retrieval remains robust even in the presence of incomplete metadata, while transforming the search process from a 'black-box' similarity matching into a governable and deterministic workflow. This work is intended as a position paper; its goal is to bridge the gap between OLAP-style multidimensional modeling and modern RAG architectures, and to stimulate further research on principled, explainable, and governable retrieval strategies at scale. 

---
# O-Researcher: An Open Ended Deep Research Model via Multi-Agent Distillation and Agentic RL 

**Authors**: Yi Yao, He Zhu, Piaohong Wang, Jincheng Ren, Xinlong Yang, Qianben Chen, Xiaowan Li, Dingfeng Shi, Jiaxian Li, Qiexiang Wang, Sinuo Wang, Xinpeng Liu, Jiaqi Wu, Minghao Liu, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.03743)  

**Abstract**: The performance gap between closed-source and open-source large language models (LLMs) is largely attributed to disparities in access to high-quality training data. To bridge this gap, we introduce a novel framework for the automated synthesis of sophisticated, research-grade instructional data. Our approach centers on a multi-agent workflow where collaborative AI agents simulate complex tool-integrated reasoning to generate diverse and high-fidelity data end-to-end. Leveraging this synthesized data, we develop a two-stage training strategy that integrates supervised fine-tuning with a novel reinforcement learning method, designed to maximize model alignment and capability. Extensive experiments demonstrate that our framework empowers open-source models across multiple scales, enabling them to achieve new state-of-the-art performance on the major deep research benchmark. This work provides a scalable and effective pathway for advancing open-source LLMs without relying on proprietary data or models. 

---
# RadDiff: Describing Differences in Radiology Image Sets with Natural Language 

**Authors**: Xiaoxian Shen, Yuhui Zhang, Sahithi Ankireddy, Xiaohan Wang, Maya Varma, Henry Guo, Curtis Langlotz, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2601.03733)  

**Abstract**: Understanding how two radiology image sets differ is critical for generating clinical insights and for interpreting medical AI systems. We introduce RadDiff, a multimodal agentic system that performs radiologist-style comparative reasoning to describe clinically meaningful differences between paired radiology studies. RadDiff builds on a proposer-ranker framework from VisDiff, and incorporates four innovations inspired by real diagnostic workflows: (1) medical knowledge injection through domain-adapted vision-language models; (2) multimodal reasoning that integrates images with their clinical reports; (3) iterative hypothesis refinement across multiple reasoning rounds; and (4) targeted visual search that localizes and zooms in on salient regions to capture subtle findings. To evaluate RadDiff, we construct RadDiffBench, a challenging benchmark comprising 57 expert-validated radiology study pairs with ground-truth difference descriptions. On RadDiffBench, RadDiff achieves 47% accuracy, and 50% accuracy when guided by ground-truth reports, significantly outperforming the general-domain VisDiff baseline. We further demonstrate RadDiff's versatility across diverse clinical tasks, including COVID-19 phenotype comparison, racial subgroup analysis, and discovery of survival-related imaging features. Together, RadDiff and RadDiffBench provide the first method-and-benchmark foundation for systematically uncovering meaningful differences in radiological data. 

---
# From Laboratory to Real-World Applications: Benchmarking Agentic Code Reasoning at the Repository Level 

**Authors**: Jia Li, Yuxin Su, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03731)  

**Abstract**: As large language models (LLMs) evolve into autonomous agents, evaluating repository-level reasoning, the ability to maintain logical consistency across massive, real-world, interdependent file systems, has become critical. Current benchmarks typically fluctuate between isolated code snippets and black-box evaluations. We present RepoReason, a white-box diagnostic benchmark centered on abductive assertion verification. To eliminate memorization while preserving authentic logical depth, we implement an execution-driven mutation framework that utilizes the environment as a semantic oracle to regenerate ground-truth states. Furthermore, we establish a fine-grained diagnostic system using dynamic program slicing, quantifying reasoning via three orthogonal metrics: $ESV$ (reading load), $MCL$ (simulation depth), and $DFI$ (integration width). Comprehensive evaluations of frontier models (e.g., Claude-4.5-Sonnet, DeepSeek-v3.1-Terminus) reveal a prevalent aggregation deficit, where integration width serves as the primary cognitive bottleneck. Our findings provide granular white-box insights for optimizing the next generation of agentic software engineering. 

---
# CSMCIR: CoT-Enhanced Symmetric Alignment with Memory Bank for Composed Image Retrieval 

**Authors**: Zhipeng Qian, Zihan Liang, Yufei Ma, Ben Chen, Huangyu Dai, Yiwei Ma, Jiayi Ji, Chenyi Lei, Han Li, Xiaoshuai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.03728)  

**Abstract**: Composed Image Retrieval (CIR) enables users to search for target images using both a reference image and manipulation text, offering substantial advantages over single-modality retrieval systems. However, existing CIR methods suffer from representation space fragmentation: queries and targets comprise heterogeneous modalities and are processed by distinct encoders, forcing models to bridge misaligned representation spaces only through post-hoc alignment, which fundamentally limits retrieval performance. This architectural asymmetry manifests as three distinct, well-separated clusters in the feature space, directly demonstrating how heterogeneous modalities create fundamentally misaligned representation spaces from initialization. In this work, we propose CSMCIR, a unified representation framework that achieves efficient query-target alignment through three synergistic components. First, we introduce a Multi-level Chain-of-Thought (MCoT) prompting strategy that guides Multimodal Large Language Models to generate discriminative, semantically compatible captions for target images, establishing modal symmetry. Building upon this, we design a symmetric dual-tower architecture where both query and target sides utilize the identical shared-parameter Q-Former for cross-modal encoding, ensuring consistent feature representations and further reducing the alignment gap. Finally, this architectural symmetry enables an entropy-based, temporally dynamic Memory Bank strategy that provides high-quality negative samples while maintaining consistency with the evolving model state. Extensive experiments on four benchmark datasets demonstrate that our CSMCIR achieves state-of-the-art performance with superior training efficiency. Comprehensive ablation studies further validate the effectiveness of each proposed component. 

---
# R$^3$L: Reflect-then-Retry Reinforcement Learning with Language-Guided Exploration, Pivotal Credit, and Positive Amplification 

**Authors**: Weijie Shi, Yanxi Chen, Zexi Li, Xuchen Pan, Yuchang Sun, Jiajie Xu, Xiaofang Zhou, Yaliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.03715)  

**Abstract**: Reinforcement learning drives recent advances in LLM reasoning and agentic capabilities, yet current approaches struggle with both exploration and exploitation. Exploration suffers from low success rates on difficult tasks and high costs of repeated rollouts from scratch. Exploitation suffers from coarse credit assignment and training instability: Trajectory-level rewards penalize valid prefixes for later errors, and failure-dominated groups overwhelm the few positive signals, leaving optimization without constructive direction. To this end, we propose R$^3$L, Reflect-then-Retry Reinforcement Learning with Language-Guided Exploration, Pivotal Credit, and Positive Amplification. To synthesize high-quality trajectories, R$^3$L shifts from stochastic sampling to active synthesis via reflect-then-retry, leveraging language feedback to diagnose errors, transform failed attempts into successful ones, and reduce rollout costs by restarting from identified failure points. With errors diagnosed and localized, Pivotal Credit Assignment updates only the diverging suffix where contrastive signals exist, excluding the shared prefix from gradient update. Since failures dominate on difficult tasks and reflect-then-retry produces off-policy data, risking training instability, Positive Amplification upweights successful trajectories to ensure positive signals guide the optimization process. Experiments on agentic and reasoning tasks demonstrate 5\% to 52\% relative improvements over baselines while maintaining training stability. Our code is released at this https URL. 

---
# The Power of 10: New Rules for the Digital World 

**Authors**: Sarah Spiekermann-Hoff, Marc Langheinrich, Johannes Hoff, Christiane Wendehorst, Jürgen Pfeffer, Thomas Fuchs, Armin Grunwald  

**Link**: [PDF](https://arxiv.org/pdf/2601.03709)  

**Abstract**: As artificial intelligence rapidly advances, society is increasingly captivated by promises of superhuman machines and seamless digital futures. Yet these visions often obscure mounting social, ethical, and psychological concerns tied to pervasive digital technologies - from surveillance to mental health crises. This article argues that a guiding ethos is urgently needed to navigate these transformations. Inspired by the lasting influence of the biblical Ten Commandments, a European interdisciplinary group has proposed "Ten Rules for the Digital World" - a novel ethical framework to help individuals and societies make prudent, human-centered decisions in the age of "supercharged" technology. 

---
# MHRC-Bench: A Multilingual Hardware Repository-Level Code Completion benchmark 

**Authors**: Qingyun Zou, Jiahao Cui, Nuo Chen, Bingsheng He, Weng-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2601.03708)  

**Abstract**: Large language models (LLMs) have achieved strong performance on code completion tasks in general-purpose programming languages. However, existing repository-level code completion benchmarks focus almost exclusively on software code and largely overlook hardware description languages. In this work, we present \textbf{MHRC-Bench}, consisting of \textbf{MHRC-Bench-Train} and \textbf{MHRC-Bench-Eval}, the first benchmark designed for multilingual hardware code completion at the repository level. Our benchmark targets completion tasks and covers three major hardware design coding styles. Each completion target is annotated with code-structure-level and hardware-oriented semantic labels derived from concrete syntax tree analysis. We conduct a comprehensive evaluation of models on MHRC-Bench-Eval. Comprehensive evaluation results and analysis demonstrate the effectiveness of MHRC-Bench. 

---
# Investigating Knowledge Distillation Through Neural Networks for Protein Binding Affinity Prediction 

**Authors**: Wajid Arshad Abbasi, Syed Ali Abbas, Maryum Bibi, Saiqa Andleeb, Muhammad Naveed Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2601.03704)  

**Abstract**: The trade-off between predictive accuracy and data availability makes it difficult to predict protein--protein binding affinity accurately. The lack of experimentally resolved protein structures limits the performance of structure-based machine learning models, which generally outperform sequence-based methods. In order to overcome this constraint, we suggest a regression framework based on knowledge distillation that uses protein structural data during training and only needs sequence data during inference. The suggested method uses binding affinity labels and intermediate feature representations to jointly supervise the training of a sequence-based student network under the guidance of a structure-informed teacher network. Leave-One-Complex-Out (LOCO) cross-validation was used to assess the framework on a non-redundant protein--protein binding affinity benchmark dataset. A maximum Pearson correlation coefficient (P_r) of 0.375 and an RMSE of 2.712 kcal/mol were obtained by sequence-only baseline models, whereas a P_r of 0.512 and an RMSE of 2.445 kcal/mol were obtained by structure-based models. With a P_r of 0.481 and an RMSE of 2.488 kcal/mol, the distillation-based student model greatly enhanced sequence-only performance. Improved agreement and decreased bias were further confirmed by thorough error analyses. With the potential to close the performance gap between sequence-based and structure-based models as larger datasets become available, these findings show that knowledge distillation is an efficient method for transferring structural knowledge to sequence-based predictors. The source code for running inference with the proposed distillation-based binding affinity predictor can be accessed at this https URL. 

---
# TreeAdv: Tree-Structured Advantage Redistribution for Group-Based RL 

**Authors**: Lang Cao, Hui Ruan, Yongqian Li, Peng Chao, Wu Ning, Haonan Song, Renhong Chen, Yitong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.03703)  

**Abstract**: Reinforcement learning with group-based objectives, such as Group Relative Policy Optimization (GRPO), is a common framework for aligning large language models on complex reasoning tasks. However, standard GRPO treats each rollout trajectory as an independent flat sequence and assigns a single sequence-level advantage to all tokens, which leads to sample inefficiency and a length bias toward verbose, redundant chains of thought without improving logical depth. We introduce TreeAdv (Tree-Structured Advantage Redistribution for Group-Based RL), which makes the tree structure of group rollouts explicit for both exploration and advantage assignment. Specifically, TreeAdv builds a group of trees (a forest) based on an entropy-driven sampling method where each tree branches at high-uncertainty decisions while sharing low-uncertainty tokens across rollouts. Then, TreeAdv aggregates token-level advantages for internal tree segments by redistributing the advantages of complete rollouts (all leaf nodes), and TreeAdv can easily apply to group-based objectives such as GRPO or GSPO. Across 10 math reasoning benchmarks, TreeAdv consistently outperforms GRPO and GSPO, while using substantially fewer generated tokens under identical supervision, data, and decoding budgets. 

---
# Inference Attacks Against Graph Generative Diffusion Models 

**Authors**: Xiuling Wang, Xin Huang, Guibo Luo, Jianliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03701)  

**Abstract**: Graph generative diffusion models have recently emerged as a powerful paradigm for generating complex graph structures, effectively capturing intricate dependencies and relationships within graph data. However, the privacy risks associated with these models remain largely unexplored. In this paper, we investigate information leakage in such models through three types of black-box inference attacks. First, we design a graph reconstruction attack, which can reconstruct graphs structurally similar to those training graphs from the generated graphs. Second, we propose a property inference attack to infer the properties of the training graphs, such as the average graph density and the distribution of densities, from the generated graphs. Third, we develop two membership inference attacks to determine whether a given graph is present in the training set. Extensive experiments on three different types of graph generative diffusion models and six real-world graphs demonstrate the effectiveness of these attacks, significantly outperforming the baseline approaches. Finally, we propose two defense mechanisms that mitigate these inference attacks and achieve a better trade-off between defense strength and target model utility than existing methods. Our code is available at this https URL. 

---
# ADEPT: Adaptive Dynamic Early-Exit Process for Transformers 

**Authors**: Sangmin Yoo, Srikanth Malla, Chiho Choi, Wei D. Lu, Joon Hee Choi  

**Link**: [PDF](https://arxiv.org/pdf/2601.03700)  

**Abstract**: The inference of large language models imposes significant computational workloads, often requiring the processing of billions of parameters. Although early-exit strategies have proven effective in reducing computational demands by halting inference earlier, they apply either to only the first token in the generation phase or at the prompt level in the prefill phase. Thus, the Key-Value (KV) cache for skipped layers remains a bottleneck for subsequent token generation, limiting the benefits of early exit. We introduce ADEPT (Adaptive Dynamic Early-exit Process for Transformers), a novel approach designed to overcome this issue and enable dynamic early exit in both the prefill and generation phases. The proposed adaptive token-level early-exit mechanism adjusts computation dynamically based on token complexity, optimizing efficiency without compromising performance. ADEPT further enhances KV generation procedure by decoupling sequential dependencies in skipped layers, making token-level early exit more practical. Experimental results demonstrate that ADEPT improves efficiency by up to 25% in language generation tasks and achieves a 4x speed-up in downstream classification tasks, with up to a 45% improvement in performance. 

---
# Can AI Chatbots Provide Coaching in Engineering? Beyond Information Processing Toward Mastery 

**Authors**: Junaid Qadir, Muhammad Adil Attique, Saleha Shoaib, Syed Ibrahim Ghaznavi  

**Link**: [PDF](https://arxiv.org/pdf/2601.03693)  

**Abstract**: Engineering education faces a double disruption: traditional apprenticeship models that cultivated judgment and tacit skill are eroding, just as generative AI emerges as an informal coaching partner. This convergence rekindles long-standing questions in the philosophy of AI and cognition about the limits of computation, the nature of embodied rationality, and the distinction between information processing and wisdom. Building on this rich intellectual tradition, this paper examines whether AI chatbots can provide coaching that fosters mastery rather than merely delivering information. We synthesize critical perspectives from decades of scholarship on expertise, tacit knowledge, and human-machine interaction, situating them within the context of contemporary AI-driven education. Empirically, we report findings from a mixed-methods study (N = 75 students, N = 7 faculty) exploring the use of a coaching chatbot in engineering education. Results reveal a consistent boundary: participants accept AI for technical problem solving (convergent tasks; M = 3.84 on a 1-5 Likert scale) but remain skeptical of its capacity for moral, emotional, and contextual judgment (divergent tasks). Faculty express stronger concerns over risk (M = 4.71 vs. M = 4.14, p = 0.003), and privacy emerges as a key requirement, with 64-71 percent of participants demanding strict confidentiality. Our findings suggest that while generative AI can democratize access to cognitive and procedural support, it cannot replicate the embodied, value-laden dimensions of human mentorship. We propose a multiplex coaching framework that integrates human wisdom within expert-in-the-loop models, preserving the depth of apprenticeship while leveraging AI scalability to enrich the next generation of engineering education. 

---
# A Pre-trained Reaction Embedding Descriptor Capturing Bond Transformation Patterns 

**Authors**: Weiqi Liu, Fenglei Cao, Yuan Qi, Li-Cheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03689)  

**Abstract**: With the rise of data-driven reaction prediction models, effective reaction descriptors are crucial for bridging the gap between real-world chemistry and digital representations. However, general-purpose, reaction-wise descriptors remain scarce. This study introduces RXNEmb, a novel reaction-level descriptor derived from RXNGraphormer, a model pre-trained to distinguish real reactions from fictitious ones with erroneous bond changes, thereby learning intrinsic bond formation and cleavage patterns. We demonstrate its utility by data-driven re-clustering of the USPTO-50k dataset, yielding a classification that more directly reflects bond-change similarities than rule-based categories. Combined with dimensionality reduction, RXNEmb enables visualization of reaction space diversity. Furthermore, attention weight analysis reveals the model's focus on chemically critical sites, providing mechanistic insight. RXNEmb serves as a powerful, interpretable tool for reaction fingerprinting and analysis, paving the way for more data-centric approaches in reaction analysis and discovery. 

---
# From Implicit to Explicit: Token-Efficient Logical Supervision for Mathematical Reasoning in LLMs 

**Authors**: Shaojie Wang, Liang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03682)  

**Abstract**: Recent studies reveal that large language models (LLMs) exhibit limited logical reasoning abilities in mathematical problem-solving, instead often relying on pattern-matching and memorization. We systematically analyze this limitation, focusing on logical relationship understanding, which is a core capability underlying genuine logical reasoning, and reveal that errors related to this capability account for over 90\% of incorrect predictions, with Chain-of-Thought Supervised Fine-Tuning (CoT-SFT) failing to substantially reduce these errors. To address this bottleneck, we propose First-Step Logical Reasoning (FSLR), a lightweight training framework targeting logical relationship understanding. Our key insight is that the first planning step-identifying which variables to use and which operation to apply-encourages the model to derive logical relationships directly from the problem statement. By training models on this isolated step, FSLR provides explicit supervision for logical relationship understanding, unlike CoT-SFT which implicitly embeds such relationships within complete solution trajectories. Extensive experiments across multiple models and datasets demonstrate that FSLR consistently outperforms CoT-SFT under both in-distribution and out-of-distribution settings, with average improvements of 3.2\% and 4.6\%, respectively. Moreover, FSLR achieves 4-6x faster training and reduces training token consumption by over 80\%. 

---
# Towards Compositional Generalization of LLMs via Skill Taxonomy Guided Data Synthesis 

**Authors**: Yifan Wei, Li Du, Xiaoyan Yu, Yang Feng, Angsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.03676)  

**Abstract**: Large Language Models (LLMs) and agent-based systems often struggle with compositional generalization due to a data bottleneck in which complex skill combinations follow a long-tailed, power-law distribution, limiting both instruction-following performance and generalization in agent-centric tasks. To address this challenge, we propose STEPS, a Skill Taxonomy guided Entropy-based Post-training data Synthesis framework for generating compositionally challenging data. STEPS explicitly targets compositional generalization by uncovering latent relationships among skills and organizing them into an interpretable, hierarchical skill taxonomy using structural information theory. Building on this taxonomy, we formulate data synthesis as a constrained information maximization problem, selecting skill combinations that maximize marginal structural information within the hierarchy while preserving semantic coherence. Experiments on challenging instruction-following benchmarks show that STEPS outperforms existing data synthesis baselines, while also yielding improved compositional generalization in downstream agent-based evaluations. 

---
# Disentangling Aleatoric and Epistemic Uncertainty in Physics-Informed Neural Networks. Application to Insulation Material Degradation Prognostics 

**Authors**: Ibai Ramirez, Jokin Alcibar, Joel Pino, Mikel Sanz, Jose I. Aizpurua  

**Link**: [PDF](https://arxiv.org/pdf/2601.03673)  

**Abstract**: Physics-Informed Neural Networks (PINNs) provide a framework for integrating physical laws with data. However, their application to Prognostics and Health Management (PHM) remains constrained by the limited uncertainty quantification (UQ) capabilities. Most existing PINN-based prognostics approaches are deterministic or account only for epistemic uncertainty, limiting their suitability for risk-aware decision-making. This work introduces a heteroscedastic Bayesian Physics-Informed Neural Network (B-PINN) framework that jointly models epistemic and aleatoric uncertainty, yielding full predictive posteriors for spatiotemporal insulation material ageing estimation. The approach integrates Bayesian Neural Networks (BNNs) with physics-based residual enforcement and prior distributions, enabling probabilistic inference within a physics-informed learning architecture. The framework is evaluated on transformer insulation ageing application, validated with a finite-element thermal model and field measurements from a solar power plant, and benchmarked against deterministic PINNs, dropout-based PINNs (d-PINNs), and alternative B-PINN variants. Results show that the proposed B-PINN provides improved predictive accuracy and better-calibrated uncertainty estimates than competing approaches. A systematic sensitivity study further analyzes the impact of boundary-condition, initial-condition, and residual sampling strategies on accuracy, calibration, and generalization. Overall, the findings highlight the potential of Bayesian physics-informed learning to support uncertainty-aware prognostics and informed decision-making in transformer asset management. 

---
# Discontinuous Galerkin finite element operator network for solving non-smooth PDEs 

**Authors**: Kapil Chawla, Youngjoon Hong, Jae Yong Lee, Sanghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.03668)  

**Abstract**: We introduce Discontinuous Galerkin Finite Element Operator Network (DG--FEONet), a data-free operator learning framework that combines the strengths of the discontinuous Galerkin (DG) method with neural networks to solve parametric partial differential equations (PDEs) with discontinuous coefficients and non-smooth solutions. Unlike traditional operator learning models such as DeepONet and Fourier Neural Operator, which require large paired datasets and often struggle near sharp features, our approach minimizes the residual of a DG-based weak formulation using the Symmetric Interior Penalty Galerkin (SIPG) scheme. DG-FEONet predicts element-wise solution coefficients via a neural network, enabling data-free training without the need for precomputed input-output pairs. We provide theoretical justification through convergence analysis and validate the model's performance on a series of one- and two-dimensional PDE problems, demonstrating accurate recovery of discontinuities, strong generalization across parameter space, and reliable convergence rates. Our results highlight the potential of combining local discretization schemes with machine learning to achieve robust, singularity-aware operator approximation in challenging PDE settings. 

---
# e5-omni: Explicit Cross-modal Alignment for Omni-modal Embeddings 

**Authors**: Haonan Chen, Sicheng Gao, Radu Timofte, Tetsuya Sakai, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2601.03666)  

**Abstract**: Modern information systems often involve different types of items, e.g., a text query, an image, a video clip, or an audio segment. This motivates omni-modal embedding models that map heterogeneous modalities into a shared space for direct comparison. However, most recent omni-modal embeddings still rely heavily on implicit alignment inherited from pretrained vision-language model (VLM) backbones. In practice, this causes three common issues: (i) similarity logits have modality-dependent sharpness, so scores are not on a consistent scale; (ii) in-batch negatives become less effective over time because mixed-modality batches create an imbalanced hardness distribution; as a result, many negatives quickly become trivial and contribute little gradient; and (iii) embeddings across modalities show mismatched first- and second-order statistics, which makes rankings less stable. To tackle these problems, we propose e5-omni, a lightweight explicit alignment recipe that adapts off-the-shelf VLMs into robust omni-modal embedding models. e5-omni combines three simple components: (1) modality-aware temperature calibration to align similarity scales, (2) a controllable negative curriculum with debiasing to focus on confusing negatives while reducing the impact of false negatives, and (3) batch whitening with covariance regularization to better match cross-modal geometry in the shared embedding space. Experiments on MMEB-V2 and AudioCaps show consistent gains over strong bi-modal and omni-modal baselines, and the same recipe also transfers well to other VLM backbones. We release our model checkpoint at this https URL. 

---
# AMIR-GRPO: Inducing Implicit Preference Signals into GRPO 

**Authors**: Amir Hossein Yari, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2601.03661)  

**Abstract**: Reinforcement learning has become the primary paradigm for aligning large language models (LLMs) on complex reasoning tasks, with group relative policy optimization (GRPO) widely used in large-scale post-training. However, GRPO faces structural limitations in reasoning-heavy settings: sequence-level advantage normalization introduces systematic length bias, penalties for low-quality trajectories are diluted, and the scalar objective discards rich pairwise preference information embedded in within-group reward rankings. As a result, valuable supervision from costly rollouts remains underutilized.
We propose AMIR-GRPO, which augments GRPO with an implicit DPO-style contrastive regularizer constructed directly from intra-group reward rankings, requiring no additional annotations. This mechanism amplifies suppression of low-reward trajectories, attenuates response-level length bias, and transforms each rollout group into a denser set of supervision constraints. Across multiple mathematical reasoning benchmarks, AMIR-GRPO consistently outperforms strong GRPO baselines, yields clearer separation between correct and incorrect reasoning chains, and delivers broader coverage gains beyond the subset of instances solved by standard GRPO. 

---
# Group and Exclusive Sparse Regularization-based Continual Learning of CNNs 

**Authors**: Basile Tousside, Janis Mohr, Jörg Frochte  

**Link**: [PDF](https://arxiv.org/pdf/2601.03658)  

**Abstract**: We present a regularization-based approach for continual learning (CL) of fixed capacity convolutional neural networks (CNN) that does not suffer from the problem of catastrophic forgetting when learning multiple tasks sequentially. This method referred to as Group and Exclusive Sparsity based Continual Learning (GESCL) avoids forgetting of previous tasks by ensuring the stability of the CNN via a stability regularization term, which prevents filters detected as important for past tasks to deviate too much when learning a new task. On top of that, GESCL makes the network plastic via a plasticity regularization term that leverage the over-parameterization of CNNs to efficiently sparsify the network and tunes unimportant filters making them relevant for future tasks. Doing so, GESCL deals with significantly less parameters and computation compared to CL approaches that either dynamically expand the network or memorize past tasks' data. Experiments on popular CL vision benchmarks show that GESCL leads to significant improvements over state-of-the-art method in terms of overall CL performance, as measured by classification accuracy as well as in terms of avoiding catastrophic forgetting. 

---
# In Search of Grandmother Cells: Tracing Interpretable Neurons in Tabular Representations 

**Authors**: Ricardo Knauer, Erik Rodner  

**Link**: [PDF](https://arxiv.org/pdf/2601.03657)  

**Abstract**: Foundation models are powerful yet often opaque in their decision-making. A topic of continued interest in both neuroscience and artificial intelligence is whether some neurons behave like grandmother cells, i.e., neurons that are inherently interpretable because they exclusively respond to single concepts. In this work, we propose two information-theoretic measures that quantify the neuronal saliency and selectivity for single concepts. We apply these metrics to the representations of TabPFN, a tabular foundation model, and perform a simple search across neuron-concept pairs to find the most salient and selective pair. Our analysis provides the first evidence that some neurons in such models show moderate, statistically significant saliency and selectivity for high-level concepts. These findings suggest that interpretable neurons can emerge naturally and that they can, in some cases, be identified without resorting to more complex interpretability techniques. 

---
# ReLA: Representation Learning and Aggregation for Job Scheduling with Reinforcement Learning 

**Authors**: Zhengyi Kwan, Zhang Wei, Aik Beng Ng, Zhengkui Wang, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2601.03646)  

**Abstract**: Job scheduling is widely used in real-world manufacturing systems to assign ordered job operations to machines under various constraints. Existing solutions remain limited by long running time or insufficient schedule quality, especially when problem scale increases. In this paper, we propose ReLA, a reinforcement-learning (RL) scheduler built on structured representation learning and aggregation. ReLA first learns diverse representations from scheduling entities, including job operations and machines, using two intra-entity learning modules with self-attention and convolution and one inter-entity learning module with cross-attention. These modules are applied in a multi-scale architecture, and their outputs are aggregated to support RL decision-making. Across experiments on small, medium, and large job instances, ReLA achieves the best makespan in most tested settings over the latest solutions. On non-large instances, ReLA reduces the optimality gap of the SOTA baseline by 13.0%, while on large-scale instances it reduces the gap by 78.6%, with the average optimality gaps lowered to 7.3% and 2.1%, respectively. These results confirm that ReLA's learned representations and aggregation provide strong decision support for RL scheduling, and enable fast job completion and decision-making for real-world applications. 

---
# MFC-RFNet: A Multi-scale Guided Rectified Flow Network for Radar Sequence Prediction 

**Authors**: Wenjie Luo, Chuanhu Deng, Chaorong Li, Rongyao Deng, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03633)  

**Abstract**: Accurate and high-resolution precipitation nowcasting from radar echo sequences is crucial for disaster mitigation and economic planning, yet it remains a significant challenge. Key difficulties include modeling complex multi-scale evolution, correcting inter-frame feature misalignment caused by displacement, and efficiently capturing long-range spatiotemporal context without sacrificing spatial fidelity. To address these issues, we present the Multi-scale Feature Communication Rectified Flow (RF) Network (MFC-RFNet), a generative framework that integrates multi-scale communication with guided feature fusion. To enhance multi-scale fusion while retaining fine detail, a Wavelet-Guided Skip Connection (WGSC) preserves high-frequency components, and a Feature Communication Module (FCM) promotes bidirectional cross-scale interaction. To correct inter-frame displacement, a Condition-Guided Spatial Transform Fusion (CGSTF) learns spatial transforms from conditioning echoes to align shallow features. The backbone adopts rectified flow training to learn near-linear probability-flow trajectories, enabling few-step sampling with stable fidelity. Additionally, lightweight Vision-RWKV (RWKV) blocks are placed at the encoder tail, the bottleneck, and the first decoder layer to capture long-range spatiotemporal dependencies at low spatial resolutions with moderate compute. Evaluations on four public datasets (SEVIR, MeteoNet, Shanghai, and CIKM) demonstrate consistent improvements over strong baselines, yielding clearer echo morphology at higher rain-rate thresholds and sustained skill at longer lead times. These results suggest that the proposed synergy of RF training with scale-aware communication, spatial alignment, and frequency-aware fusion presents an effective and robust approach for radar-based nowcasting. 

---
# ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis 

**Authors**: Haitao Li, Chunxiang Jin, Chenglin Li, Wenhao Guan, Zhengxing Huang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.03632)  

**Abstract**: Zero-shot text-to-speech models can clone a speaker's timbre from a short reference audio, but they also strongly inherit the speaking style present in the reference. As a result, synthesizing speech with a desired style often requires carefully selecting reference audio, which is impractical when only limited or mismatched references are available. While recent controllable TTS methods attempt to address this issue, they typically rely on absolute style targets and discrete textual prompts, and therefore do not support continuous and reference-relative style control. We propose ReStyle-TTS, a framework that enables continuous and reference-relative style control in zero-shot TTS. Our key insight is that effective style control requires first reducing the model's implicit dependence on reference style before introducing explicit control mechanisms. To this end, we introduce Decoupled Classifier-Free Guidance (DCFG), which independently controls text and reference guidance, reducing reliance on reference style while preserving text fidelity. On top of this, we apply style-specific LoRAs together with Orthogonal LoRA Fusion to enable continuous and disentangled multi-attribute control, and introduce a Timbre Consistency Optimization module to mitigate timbre drift caused by weakened reference guidance. Experiments show that ReStyle-TTS enables user-friendly, continuous, and relative control over pitch, energy, and multiple emotions while maintaining intelligibility and speaker timbre, and performs robustly in challenging mismatched reference-target style scenarios. 

---
# Evaluating the Pre-Consultation Ability of LLMs using Diagnostic Guidelines 

**Authors**: Jean Seo, Gibaeg Kim, Kihun Shin, Seungseop Lim, Hyunkyung Lee, Wooseok Han, Jongwon Lee, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03627)  

**Abstract**: We introduce EPAG, a benchmark dataset and framework designed for Evaluating the Pre-consultation Ability of LLMs using diagnostic Guidelines. LLMs are evaluated directly through HPI-diagnostic guideline comparison and indirectly through disease diagnosis. In our experiments, we observe that small open-source models fine-tuned with a well-curated, task-specific dataset can outperform frontier LLMs in pre-consultation. Additionally, we find that increased amount of HPI (History of Present Illness) does not necessarily lead to improved diagnostic performance. Further experiments reveal that the language of pre-consultation influences the characteristics of the dialogue. By open-sourcing our dataset and evaluation pipeline on this https URL, we aim to contribute to the evaluation and further development of LLM applications in real-world clinical settings. 

---
# Investigation into respiratory sound classification for an imbalanced data set using hybrid LSTM-KAN architectures 

**Authors**: Nithinkumar K.V, Anand R  

**Link**: [PDF](https://arxiv.org/pdf/2601.03610)  

**Abstract**: Respiratory sounds captured via auscultation contain critical clues for diagnosing pulmonary conditions. Automated classification of these sounds faces challenges due to subtle acoustic differences and severe class imbalance in clinical datasets. This study investigates respiratory sound classification with a focus on mitigating pronounced class imbalance. We propose a hybrid deep learning model that combines a Long Short-Term Memory (LSTM) network for sequential feature encoding with a Kolmogorov-Arnold Network (KAN) for classification. The model is integrated with a comprehensive feature extraction pipeline and targeted imbalance mitigation strategies. Experiments were conducted on a public respiratory sound database comprising six classes with a highly skewed distribution. Techniques such as focal loss, class-specific data augmentation, and Synthetic Minority Over-sampling Technique (SMOTE) were employed to enhance minority class recognition. The proposed Hybrid LSTM-KAN model achieves an overall accuracy of 94.6 percent and a macro-averaged F1 score of 0.703, despite the dominant COPD class accounting for over 86 percent of the data. Improved detection performance is observed for minority classes compared to baseline approaches, demonstrating the effectiveness of the proposed architecture for imbalanced respiratory sound classification. 

---
# Policy-Guided Search on Tree-of-Thoughts for Efficient Problem Solving with Bounded Language Model Queries 

**Authors**: Sumedh Pendurkar, Guni Sharon  

**Link**: [PDF](https://arxiv.org/pdf/2601.03606)  

**Abstract**: Recent studies explored integrating state-space search algorithms with Language Models (LM) to perform look-ahead on the token generation process, the ''Tree-of-Thoughts'' (ToT), generated by LMs, thereby improving performance on problem-solving tasks. However, the affiliated search algorithms often overlook the significant computational costs associated with LM inference, particularly in scenarios with constrained computational budgets. Consequently, we address the problem of improving LM performance on problem-solving tasks under limited computational budgets. We demonstrate how the probabilities assigned to thoughts by LMs can serve as a heuristic to guide search within the ToT framework, thereby reducing the number of thought evaluations. Building on this insight, we adapt a heuristic search algorithm, Levin Tree Search (LTS), to the ToT framework, which leverages LMs as policies to guide the tree exploration efficiently. We extend the theoretical results of LTS by showing that, for ToT (a pruned tree), LTS guarantees a bound on the number of states expanded, and consequently, on the number of thoughts generated. Additionally, we analyze the sensitivity of this bound to the temperature values commonly used in the final softmax layer of the LM. Empirical evaluation under a fixed LM query budget demonstrates that LTS consistently achieves comparable or higher accuracy than baseline search algorithms within the ToT framework, across three domains (Blocksworld, PrOntoQA, Array Sorting) and four distinct LMs. These findings highlight the efficacy of LTS on ToT, particularly in enabling cost-effective and time-efficient problem-solving, making it well-suited for latency-critical and resource-constrained applications. 

---
# ALERT: Zero-shot LLM Jailbreak Detection via Internal Discrepancy Amplification 

**Authors**: Xiao Lin, Philip Li, Zhichen Zeng, Tingwei Li, Tianxin Wei, Xuying Ning, Gaotang Li, Yuzhong Chen, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2601.03600)  

**Abstract**: Despite rich safety alignment strategies, large language models (LLMs) remain highly susceptible to jailbreak attacks, which compromise safety guardrails and pose serious security risks. Existing detection methods mainly detect jailbreak status relying on jailbreak templates present in the training data. However, few studies address the more realistic and challenging zero-shot jailbreak detection setting, where no jailbreak templates are available during training. This setting better reflects real-world scenarios where new attacks continually emerge and evolve. To address this challenge, we propose a layer-wise, module-wise, and token-wise amplification framework that progressively magnifies internal feature discrepancies between benign and jailbreak prompts. We uncover safety-relevant layers, identify specific modules that inherently encode zero-shot discriminative signals, and localize informative safety tokens. Building upon these insights, we introduce ALERT (Amplification-based Jailbreak Detector), an efficient and effective zero-shot jailbreak detector that introduces two independent yet complementary classifiers on amplified representations. Extensive experiments on three safety benchmarks demonstrate that ALERT achieves consistently strong zero-shot detection performance. Specifically, (i) across all datasets and attack strategies, ALERT reliably ranks among the top two methods, and (ii) it outperforms the second-best baseline by at least 10% in average Accuracy and F1-score, and sometimes by up to 40%. 

---
# From Chains to Graphs: Self-Structured Reasoning for General-Domain LLMs 

**Authors**: Yingjian Chen, Haoran Liu, Yinhong Liu, Sherry T. Tong, Aosong Feng, Jinghui Lu, Juntao Zhang, Yusuke Iwasawa, Yutaka Matsuo, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.03597)  

**Abstract**: Large Language Models (LLMs) show strong reasoning ability in open-domain question answering, yet their reasoning processes are typically linear and often logically inconsistent. In contrast, real-world reasoning requires integrating multiple premises and solving subproblems in parallel. Existing methods, such as Chain-of-Thought (CoT), express reasoning in a linear textual form, which may appear coherent but frequently leads to inconsistent conclusions. Recent approaches rely on externally provided graphs and do not explore how LLMs can construct and use their own graph-structured reasoning, particularly in open-domain QA. To fill this gap, we novelly explore graph-structured reasoning of LLMs in general-domain question answering. We propose Self-Graph Reasoning (SGR), a framework that enables LLMs to explicitly represent their reasoning process as a structured graph before producing the final answer. We further construct a graph-structured reasoning dataset that merges multiple candidate reasoning graphs into refined graph structures for model training. Experiments on five QA benchmarks across both general and specialized domains show that SGR consistently improves reasoning consistency and yields a 17.74% gain over the base model. The LLaMA-3.3-70B model fine-tuned with SGR performs comparably to GPT-4o and surpasses Claude-3.5-Haiku, demonstrating the effectiveness of graph-structured reasoning. 

---
# Can LLMs See Without Pixels? Benchmarking Spatial Intelligence from Textual Descriptions 

**Authors**: Zhongbin Guo, Zhen Yang, Yushan Li, Xinyue Zhang, Wenyu Gao, Jiacheng Wang, Chengzhi Li, Xiangrui Liu, Ping Jian  

**Link**: [PDF](https://arxiv.org/pdf/2601.03590)  

**Abstract**: Recent advancements in Spatial Intelligence (SI) have predominantly relied on Vision-Language Models (VLMs), yet a critical question remains: does spatial understanding originate from visual encoders or the fundamental reasoning backbone? Inspired by this question, we introduce SiT-Bench, a novel benchmark designed to evaluate the SI performance of Large Language Models (LLMs) without pixel-level input, comprises over 3,800 expert-annotated items across five primary categories and 17 subtasks, ranging from egocentric navigation and perspective transformation to fine-grained robotic manipulation. By converting single/multi-view scenes into high-fidelity, coordinate-aware textual descriptions, we challenge LLMs to perform symbolic textual reasoning rather than visual pattern matching. Evaluation results of state-of-the-art (SOTA) LLMs reveals that while models achieve proficiency in localized semantic tasks, a significant "spatial gap" remains in global consistency. Notably, we find that explicit spatial reasoning significantly boosts performance, suggesting that LLMs possess latent world-modeling potential. Our proposed dataset SiT-Bench serves as a foundational resource to foster the development of spatially-grounded LLM backbones for future VLMs and embodied agents. Our code and benchmark will be released at this https URL . 

---
# Deontic Knowledge Graphs for Privacy Compliance in Multimodal Disaster Data Sharing 

**Authors**: Kelvin Uzoma Echenim, Karuna Pande Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2601.03587)  

**Abstract**: Disaster response requires sharing heterogeneous artifacts, from tabular assistance records to UAS imagery, under overlapping privacy mandates. Operational systems often reduce compliance to binary access control, which is brittle in time-critical workflows. We present a novel deontic knowledge graph-based framework that integrates a Disaster Management Knowledge Graph (DKG) with a Policy Knowledge Graph (PKG) derived from IoT-Reg and FEMA/DHS privacy drivers. Our release decision function supports three outcomes: Allow, Block, and Allow-with-Transform. The latter binds obligations to transforms and verifies post-transform compliance via provenance-linked derived artifacts; blocked requests are logged as semantic privacy incidents. Evaluation on a 5.1M-triple DKG with 316K images shows exact-match decision correctness, sub-second per-decision latency, and interactive query performance across both single-graph and federated workloads. 

---
# A Proposed Paradigm for Imputing Missing Multi-Sensor Data in the Healthcare Domain 

**Authors**: Vaibhav Gupta, Florian Grensing, Beyza Cinar, Maria Maleshkova  

**Link**: [PDF](https://arxiv.org/pdf/2601.03565)  

**Abstract**: Chronic diseases such as diabetes pose significant management challenges, particularly due to the risk of complications like hypoglycemia, which require timely detection and intervention. Continuous health monitoring through wearable sensors offers a promising solution for early prediction of glycemic events. However, effective use of multisensor data is hindered by issues such as signal noise and frequent missing values. This study examines the limitations of existing datasets and emphasizes the temporal characteristics of key features relevant to hypoglycemia prediction. A comprehensive analysis of imputation techniques is conducted, focusing on those employed in state-of-the-art studies. Furthermore, imputation methods derived from machine learning and deep learning applications in other healthcare contexts are evaluated for their potential to address longer gaps in time-series data. Based on this analysis, a systematic paradigm is proposed, wherein imputation strategies are tailored to the nature of specific features and the duration of missing intervals. The review concludes by emphasizing the importance of investigating the temporal dynamics of individual features and the implementation of multiple, feature-specific imputation techniques to effectively address heterogeneous temporal patterns inherent in the data. 

---
# Evaluating LLMs for Police Decision-Making: A Framework Based on Police Action Scenarios 

**Authors**: Sangyub Lee, Heedou Kim, Hyeoncheol Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.03553)  

**Abstract**: The use of Large Language Models (LLMs) in police operations is growing, yet an evaluation framework tailored to police operations remains absent. While LLM's responses may not always be legally incorrect, their unverified use still can lead to severe issues such as unlawful arrests and improper evidence collection. To address this, we propose PAS (Police Action Scenarios), a systematic framework covering the entire evaluation process. Applying this framework, we constructed a novel QA dataset from over 8,000 official documents and established key metrics validated through statistical analysis with police expert judgements. Experimental results show that commercial LLMs struggle with our new police-related tasks, particularly in providing fact-based recommendations. This study highlights the necessity of an expandable evaluation framework to ensure reliable AI-driven police operations. We release our data and prompt template. 

---
# Value-Action Alignment in Large Language Models under Privacy-Prosocial Conflict 

**Authors**: Guanyu Chen, Chenxiao Yu, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03546)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate decision-making tasks involving personal data sharing, where privacy concerns and prosocial motivations can push choices in opposite directions. Existing evaluations often measure privacy-related attitudes or sharing intentions in isolation, which makes it difficult to determine whether a model's expressed values jointly predict its downstream data-sharing actions as in real human behaviors. We introduce a context-based assessment protocol that sequentially administers standardized questionnaires for privacy attitudes, prosocialness, and acceptance of data sharing within a bounded, history-carrying session. To evaluate value-action alignments under competing attitudes, we use multi-group structural equation modeling (MGSEM) to identify relations from privacy concerns and prosocialness to data sharing. We propose Value-Action Alignment Rate (VAAR), a human-referenced directional agreement metric that aggregates path-level evidence for expected signs. Across multiple LLMs, we observe stable but model-specific Privacy-PSA-AoDS profiles, and substantial heterogeneity in value-action alignment. 

---
# Layer-Order Inversion: Rethinking Latent Multi-Hop Reasoning in Large Language Models 

**Authors**: Xukai Liu, Ye Liu, Jipeng Zhang, Yanghai Zhang, Kai Zhang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03542)  

**Abstract**: Large language models (LLMs) perform well on multi-hop reasoning, yet how they internally compose multiple facts remains unclear. Recent work proposes \emph{hop-aligned circuit hypothesis}, suggesting that bridge entities are computed sequentially across layers before later-hop answers. Through systematic analyses on real-world multi-hop queries, we show that this hop-aligned assumption does not generalize: later-hop answer entities can become decodable earlier than bridge entities, a phenomenon we call \emph{layer-order inversion}, which strengthens with total hops. To explain this behavior, we propose a \emph{probabilistic recall-and-extract} framework that models multi-hop reasoning as broad probabilistic recall in shallow MLP layers followed by selective extraction in deeper attention layers. This framework is empirically validated through systematic probing analyses, reinterpreting prior layer-wise decoding evidence, explaining chain-of-thought gains, and providing a mechanistic diagnosis of multi-hop failures despite correct single-hop knowledge. Code is available at this https URL. 

---
# VeRPO: Verifiable Dense Reward Policy Optimization for Code Generation 

**Authors**: Longwen Wang, Xuan'er Wu, Xiaohui Hu, Yirui Liu, Yuankai Fan, Kaidong Yu, Qizhen Weng, Wei Xi, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.03525)  

**Abstract**: Effective reward design is a central challenge in Reinforcement Learning (RL) for code generation. Mainstream pass/fail outcome rewards enforce functional correctness via executing unit tests, but the resulting sparsity limits potential performance gains. While recent work has explored external Reward Models (RM) to generate richer, continuous rewards, the learned RMs suffer from reward misalignment and prohibitive computational cost. In this paper, we introduce \textbf{VeRPO} (\textbf{V}erifiable D\textbf{e}nse \textbf{R}eward \textbf{P}olicy \textbf{O}ptimization), a novel RL framework for code generation that synthesizes \textit{robust and dense rewards fully grounded in verifiable execution feedback}. The core idea of VeRPO is constructing dense rewards from weighted partial success: by dynamically estimating the difficulty weight of each unit test based on the execution statistics during training, a dense reward is derived from the sum of weights of the passed unit tests. To solidify the consistency between partial success and end-to-end functional correctness, VeRPO further integrates the dense signal with global execution outcomes, establishing a robust and dense reward paradigm relying solely on verifiable execution feedback. Extensive experiments across diverse benchmarks and settings demonstrate that VeRPO consistently outperforms outcome-driven and RM-based baselines, achieving up to +8.83\% gain in pass@1 with negligible time cost (< 0.02\%) and zero GPU memory overhead. 

---
# A Reinforcement Learning-Based Model for Mapping and Goal-Directed Navigation Using Multiscale Place Fields 

**Authors**: Bekarys Dukenbaev, Andrew Gerstenslager, Alexander Johnson, Ali A. Minai  

**Link**: [PDF](https://arxiv.org/pdf/2601.03520)  

**Abstract**: Autonomous navigation in complex and partially observable environments remains a central challenge in robotics. Several bio-inspired models of mapping and navigation based on place cells in the mammalian hippocampus have been proposed. This paper introduces a new robust model that employs parallel layers of place fields at multiple spatial scales, a replay-based reward mechanism, and dynamic scale fusion. Simulations show that the model improves path efficiency and accelerates learning compared to single-scale baselines, highlighting the value of multiscale spatial representations for adaptive robot navigation. 

---
# Mem-Gallery: Benchmarking Multimodal Long-Term Conversational Memory for MLLM Agents 

**Authors**: Yuanchen Bei, Tianxin Wei, Xuying Ning, Yanjun Zhao, Zhining Liu, Xiao Lin, Yada Zhu, Hendrik Hamann, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2601.03515)  

**Abstract**: Long-term memory is a critical capability for multimodal large language model (MLLM) agents, particularly in conversational settings where information accumulates and evolves over time. However, existing benchmarks either evaluate multi-session memory in text-only conversations or assess multimodal understanding within localized contexts, failing to evaluate how multimodal memory is preserved, organized, and evolved across long-term conversational trajectories. Thus, we introduce Mem-Gallery, a new benchmark for evaluating multimodal long-term conversational memory in MLLM agents. Mem-Gallery features high-quality multi-session conversations grounded in both visual and textual information, with long interaction horizons and rich multimodal dependencies. Building on this dataset, we propose a systematic evaluation framework that assesses key memory capabilities along three functional dimensions: memory extraction and test-time adaptation, memory reasoning, and memory knowledge management. Extensive benchmarking across thirteen memory systems reveals several key findings, highlighting the necessity of explicit multimodal information retention and memory organization, the persistent limitations in memory reasoning and knowledge management, as well as the efficiency bottleneck of current models. 

---
# Deploy-Master: Automating the Deployment of 50,000+ Agent-Ready Scientific Tools in One Day 

**Authors**: Yi Wang, Zhenting Huang, Zhaohan Ding, Ruoxue Liao, Yuan Huang, Xinzijian Liu, Jiajun Xie, Siheng Chen, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03513)  

**Abstract**: Open-source scientific software is abundant, yet most tools remain difficult to compile, configure, and reuse, sustaining a small-workshop mode of scientific computing. This deployment bottleneck limits reproducibility, large-scale evaluation, and the practical integration of scientific tools into modern AI-for-Science (AI4S) and agentic workflows.
We present Deploy-Master, a one-stop agentic workflow for large-scale tool discovery, build specification inference, execution-based validation, and publication. Guided by a taxonomy spanning 90+ scientific and engineering domains, our discovery stage starts from a recall-oriented pool of over 500,000 public repositories and progressively filters it to 52,550 executable tool candidates under license- and quality-aware criteria. Deploy-Master transforms heterogeneous open-source repositories into runnable, containerized capabilities grounded in execution rather than documentation claims. In a single day, we performed 52,550 build attempts and constructed reproducible runtime environments for 50,112 scientific tools. Each successful tool is validated by a minimal executable command and registered in SciencePedia for search and reuse, enabling direct human use and optional agent-based invocation.
Beyond delivering runnable tools, we report a deployment trace at the scale of 50,000 tools, characterizing throughput, cost profiles, failure surfaces, and specification uncertainty that become visible only at scale. These results explain why scientific software remains difficult to operationalize and motivate shared, observable execution substrates as a foundation for scalable AI4S and agentic science. 

---
# Bootstrapping Code Translation with Weighted Multilanguage Exploration 

**Authors**: Yuhan Wu, Huan Zhang, Wei Cheng, Chen Shen, Jingyue Yang, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03512)  

**Abstract**: Code translation across multiple programming languages is essential yet challenging due to two vital obstacles: scarcity of parallel data paired with executable test oracles, and optimization imbalance when handling diverse language pairs. We propose BootTrans, a bootstrapping method that resolves both obstacles. Its key idea is to leverage the functional invariance and cross-lingual portability of test suites, adapting abundant pivot-language unit tests to serve as universal verification oracles for multilingual RL training. Our method introduces a dual-pool architecture with seed and exploration pools to progressively expand training data via execution-guided experience collection. Furthermore, we design a language-aware weighting mechanism that dynamically prioritizes harder translation directions based on relative performance across sibling languages, mitigating optimization imbalance. Extensive experiments on the HumanEval-X and TransCoder-Test benchmarks demonstrate substantial improvements over baseline LLMs across all translation directions, with ablations validating the effectiveness of both bootstrapping and weighting components. 

---
# IntroLM: Introspective Language Models via Prefilling-Time Self-Evaluation 

**Authors**: Hossein Hosseini Kasnavieh, Gholamreza Haffari, Chris Leckie, Adel N. Toosi  

**Link**: [PDF](https://arxiv.org/pdf/2601.03511)  

**Abstract**: A major challenge for the operation of large language models (LLMs) is how to predict whether a specific LLM will produce sufficiently high-quality output for a given query. Existing approaches rely on external classifiers, most commonly BERT based models, which suffer from limited context windows, constrained representational capacity, and additional computational overhead. We propose IntroLM, a method that enables causal language models to predict their own output quality during the prefilling phase without affecting generation using introspective tokens. By introducing token conditional LoRA that activates only for the introspective token, the model learns to predict the output quality for a given query while preserving the original backbone behavior and avoiding external evaluators. On question answering benchmarks, IntroLM applied to Qwen3 8B achieves a ROC AUC of 90 precent for success prediction, outperforming a DeBERTa classifier by 14 precent. When integrated into multi model routing systems, IntroLM achieves superior cost performance tradeoffs, reducing latency by up to 33 precent and large model usage by up to 50 precent at matched reliability. 

---
# Reasoning Pattern Alignment Merging for Adaptive Reasoning 

**Authors**: Zhaofeng Zhong, Wei Yuan, Tong Chen, Xiangyu Zhao, Quoc Viet Hung Nguyen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2601.03506)  

**Abstract**: Recent large reasoning models (LRMs) have made substantial progress in complex reasoning tasks, yet they often generate lengthy reasoning paths for every query, incurring unnecessary computation and latency. Existing speed-up approaches typically rely on retraining the model or designing sophisticated prompting, which are either prohibitively expensive or highly sensitive to the input and prompt formulation. In this work, we study model merging as a lightweight alternative for efficient reasoning: by combining a long chain-of-thought (Long-CoT) reasoning model with a Short-CoT instruction model, we obtain an adaptive reasoner without training from scratch or requiring large-scale additional data. Building on this idea, we propose Reasoning Pattern Alignment Merging (RPAM), a layer-wise model merging framework based on feature alignment to facilitate query-adaptive reasoning. RPAM first constructs a small pattern-labeled calibration set that assigns each query an appropriate reasoning pattern. It then optimizes layer-wise merging coefficients by aligning the merged model's intermediate representations with those of the selected model, while a contrastive objective explicitly pushes them away from the non-selected model. Experiments on seven widely used reasoning benchmarks show that RPAM substantially reduces inference cost while maintaining strong performance. Upon article acceptance, we will provide open-source code to reproduce experiments for RPAM. 

---
# Beyond Perplexity: A Lightweight Benchmark for Knowledge Retention in Supervised Fine-Tuning 

**Authors**: Soheil Zibakhsh Shabgahi, Pedram Aghazadeh, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2601.03505)  

**Abstract**: Supervised Fine-Tuning (SFT) is a standard approach for injecting domain knowledge into Large Language Models (LLMs). However, relying on validation perplexity to monitor training is often insufficient, as it confounds stylistic mimicry with genuine factual internalization. To address this, we introduce the Knowledge Retention (KR) Test , a lightweight, corpus-grounded evaluation framework designed to distinguish factual learning from linguistics. KR-Test utilizes automatically generated contrastive examples to measure likelihood preferences for correct versus incorrect continuations, requiring no instruction tuning or generative decoding. We validate the framework's integrity through a "blind vs. oracle" baseline analysis. Furthermore, we demonstrate the diagnostic capabilities of KR-Test by analyzing the training dynamics of Low-Rank Adaptation (LoRA). By exposing the fine-grained dissociation between linguistic convergence and knowledge retention, KR-Test enhances the interpretability of fine-tuning dynamics. 

---
# SDCD: Structure-Disrupted Contrastive Decoding for Mitigating Hallucinations in Large Vision-Language Models 

**Authors**: Yuxuan Xia, Siheng Wang, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.03500)  

**Abstract**: Large Vision-Language Models (LVLMs) demonstrate significant progress in multimodal understanding and reasoning, yet object hallucination remains a critical challenge. While existing research focuses on mitigating language priors or high-level statistical biases, they often overlook the internal complexities of the visual encoding process. We identify that visual statistical bias, arising from the inherent Bag-of-Patches behavior of Vision Encoders under weak structural supervision, acts as a contributing factor of object hallucinations. Under this bias, models prioritize local texture features within individual patches over holistic geometric structures. This tendency may induce spurious visual confidence and result in hallucinations. To address this, we introduce a training-free algorithm called Structure-Disrupted Contrastive Decoding (SDCD), which performs contrastive calibration of the output distribution by introducing a shuffled structure-disrupted view. By penalizing tokens that maintain high confidence under this structure-less view, SDCD effectively suppresses the texture-driven bias. Experimental results demonstrate that SDCD significantly mitigates hallucinations across multiple benchmarks and enhances the overall multimodal capabilities of LVLMs. 

---
# Cyberattack Detection in Virtualized Microgrids Using LightGBM and Knowledge-Distilled Classifiers 

**Authors**: Osasumwen Cedric Ogiesoba-Eguakun, Suman Rath  

**Link**: [PDF](https://arxiv.org/pdf/2601.03495)  

**Abstract**: Modern microgrids depend on distributed sensing and communication interfaces, making them increasingly vulnerable to cyber physical disturbances that threaten operational continuity and equipment safety. In this work, a complete virtual microgrid was designed and implemented in MATLAB/Simulink, integrating heterogeneous renewable sources and secondary controller layers. A structured cyberattack framework was developed using MGLib to inject adversarial signals directly into the secondary control pathways. Multiple attack classes were emulated, including ramp, sinusoidal, additive, coordinated stealth, and denial of service behaviors. The virtual environment was used to generate labeled datasets under both normal and attack conditions. The datasets trained Light Gradient Boosting Machine (LightGBM) models to perform two functions: detecting the presence of an intrusion (binary) and distinguishing among attack types (multiclass). The multiclass model attained 99.72% accuracy and a 99.62% F1 score, while the binary model attained 94.8% accuracy and a 94.3% F1 score. A knowledge-distillation step reduced the size of the multiclass model, allowing faster predictions with only a small drop in performance. Real-time tests showed a processing delay of about 54 to 67 ms per 1000 samples, demonstrating suitability for CPU-based edge deployment in microgrid controllers. The results confirm that lightweight machine learning based intrusion detection methods can provide fast, accurate, and efficient cyberattack detection without relying on complex deep learning models. Key contributions include: (1) development of a complete MATLAB-based virtual microgrid, (2) structured attack injection at the control layer, (3) creation of multiclass labeled datasets, and (4) design of low-cost AI models suitable for practical microgrid cybersecurity. 

---
# Submodular Evaluation Subset Selection in Automatic Prompt Optimization 

**Authors**: Jinming Nian, Zhiyuan Peng, Hongwei Shang, Dae Hoon Park, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03493)  

**Abstract**: Automatic prompt optimization reduces manual prompt engineering, but relies on task performance measured on a small, often randomly sampled evaluation subset as its main source of feedback signal. Despite this, how to select that evaluation subset is usually treated as an implementation detail. We study evaluation subset selection for prompt optimization from a principled perspective and propose SESS, a submodular evaluation subset selection method. We frame selection as maximizing an objective set function and show that, under mild conditions, it is monotone and submodular, enabling greedy selection with theoretical guarantees. Across GSM8K, MATH, and GPQA-Diamond, submodularly selected evaluation subsets can yield better optimized prompts than random or heuristic baselines. 

---
# CroBIM-U: Uncertainty-Driven Referring Remote Sensing Image Segmentation 

**Authors**: Yuzhe Sun, Zhe Dong, Haochen Jiang, Tianzhu Liu, Yanfeng Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03490)  

**Abstract**: Referring remote sensing image segmentation aims to localize specific targets described by natural language within complex overhead imagery. However, due to extreme scale variations, dense similar distractors, and intricate boundary structures, the reliability of cross-modal alignment exhibits significant \textbf{spatial non-uniformity}. Existing methods typically employ uniform fusion and refinement strategies across the entire image, which often introduces unnecessary linguistic perturbations in visually clear regions while failing to provide sufficient disambiguation in confused areas. To address this, we propose an \textbf{uncertainty-guided framework} that explicitly leverages a pixel-wise \textbf{referring uncertainty map} as a spatial prior to orchestrate adaptive inference. Specifically, we introduce a plug-and-play \textbf{Referring Uncertainty Scorer (RUS)}, which is trained via an online error-consistency supervision strategy to interpretably predict the spatial distribution of referential ambiguity. Building on this prior, we design two plug-and-play modules: 1) \textbf{Uncertainty-Gated Fusion (UGF)}, which dynamically modulates language injection strength to enhance constraints in high-uncertainty regions while suppressing noise in low-uncertainty ones; and 2) \textbf{Uncertainty-Driven Local Refinement (UDLR)}, which utilizes uncertainty-derived soft masks to focus refinement on error-prone boundaries and fine details. Extensive experiments demonstrate that our method functions as a unified, plug-and-play solution that significantly improves robustness and geometric fidelity in complex remote sensing scenes without altering the backbone architecture. 

---
# Efficient Sequential Recommendation for Long Term User Interest Via Personalization 

**Authors**: Qiang Zhang, Hanchao Yu, Ivan Ji, Chen Yuan, Yi Zhang, Chihuang Liu, Xiaolong Wang, Christopher E. Lambert, Ren Chen, Chen Kovacs, Xinzhu Bei, Renqin Cai, Rui Li, Lizhu Zhang, Xiangjun Fan, Qunshu Zhang, Benyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03479)  

**Abstract**: Recent years have witnessed success of sequential modeling, generative recommender, and large language model for recommendation. Though the scaling law has been validated for sequential models, it showed inefficiency in computational capacity when considering real-world applications like recommendation, due to the non-linear(quadratic) increasing nature of the transformer model. To improve the efficiency of the sequential model, we introduced a novel approach to sequential recommendation that leverages personalization techniques to enhance efficiency and performance. Our method compresses long user interaction histories into learnable tokens, which are then combined with recent interactions to generate recommendations. This approach significantly reduces computational costs while maintaining high recommendation accuracy. Our method could be applied to existing transformer based recommendation models, e.g., HSTU and HLLM. Extensive experiments on multiple sequential models demonstrate its versatility and effectiveness. Source code is available at \href{this https URL}{this https URL}. 

---
# Online Decision-Making Under Uncertainty for Vehicle-to-Building Systems 

**Authors**: Rishav Sen, Yunuo Zhang, Fangqi Liu, Jose Paolo Talusan, Ava Pettet, Yoshinori Suzue, Ayan Mukhopadhyay, Abhishek Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2601.03476)  

**Abstract**: Vehicle-to-building (V2B) systems integrate physical infrastructures, such as smart buildings and electric vehicles (EVs) connected to chargers at the building, with digital control mechanisms to manage energy use. By utilizing EVs as flexible energy reservoirs, buildings can dynamically charge and discharge them to optimize energy use and cut costs under time-variable pricing and demand charge policies. This setup leads to the V2B optimization problem, where buildings coordinate EV charging and discharging to minimize total electricity costs while meeting users' charging requirements. However, the V2B optimization problem is challenging because of: (1) fluctuating electricity pricing, which includes both energy charges ($/kWh) and demand charges ($/kW); (2) long planning horizons (typically over 30 days); (3) heterogeneous chargers with varying charging rates, controllability, and directionality (i.e., unidirectional or bidirectional); and (4) user-specific battery levels at departure to ensure user requirements are met. In contrast to existing approaches that often model this setting as a single-shot combinatorial optimization problem, we highlight critical limitations in prior work and instead model the V2B optimization problem as a Markov decision process (MDP), i.e., a stochastic control process. Solving the resulting MDP is challenging due to the large state and action spaces. To address the challenges of the large state space, we leverage online search, and we counter the action space by using domain-specific heuristics to prune unpromising actions. We validate our approach in collaboration with Nissan Advanced Technology Center - Silicon Valley. Using data from their EV testbed, we show that the proposed framework significantly outperforms state-of-the-art methods. 

---
# SegNSP: Revisiting Next Sentence Prediction for Linear Text Segmentation 

**Authors**: José Isidro, Filipe Cunha, Purificação Silvano, Alípio Jorge, Nuno Guimarães, Sérgio Nunes, Ricardo Campos  

**Link**: [PDF](https://arxiv.org/pdf/2601.03474)  

**Abstract**: Linear text segmentation is a long-standing problem in natural language processing (NLP), focused on dividing continuous text into coherent and semantically meaningful units. Despite its importance, the task remains challenging due to the complexity of defining topic boundaries, the variability in discourse structure, and the need to balance local coherence with global context. These difficulties hinder downstream applications such as summarization, information retrieval, and question answering. In this work, we introduce SegNSP, framing linear text segmentation as a next sentence prediction (NSP) task. Although NSP has largely been abandoned in modern pre-training, its explicit modeling of sentence-to-sentence continuity makes it a natural fit for detecting topic boundaries. We propose a label-agnostic NSP approach, which predicts whether the next sentence continues the current topic without requiring explicit topic labels, and enhance it with a segmentation-aware loss combined with harder negative sampling to better capture discourse continuity. Unlike recent proposals that leverage NSP alongside auxiliary topic classification, our approach avoids task-specific supervision. We evaluate our model against established baselines on two datasets, CitiLink-Minutes, for which we establish the first segmentation benchmark, and WikiSection. On CitiLink-Minutes, SegNSP achieves a B-$F_1$ of 0.79, closely aligning with human-annotated topic transitions, while on WikiSection it attains a B-F$_1$ of 0.65, outperforming the strongest reproducible baseline, TopSeg, by 0.17 absolute points. These results demonstrate competitive and robust performance, highlighting the effectiveness of modeling sentence-to-sentence continuity for improving segmentation quality and supporting downstream NLP applications. 

---
# EpiQAL: Benchmarking Large Language Models in Epidemiological Question Answering for Enhanced Alignment and Reasoning 

**Authors**: Mingyang Wei, Dehai Min, Zewen Liu, Yuzhang Xie, Guanchen Wu, Carl Yang, Max S. Y. Lau, Qi He, Lu Cheng, Wei Jin  

**Link**: [PDF](https://arxiv.org/pdf/2601.03471)  

**Abstract**: Reliable epidemiological reasoning requires synthesizing study evidence to infer disease burden, transmission dynamics, and intervention effects at the population level. Existing medical question answering benchmarks primarily emphasize clinical knowledge or patient-level reasoning, yet few systematically evaluate evidence-grounded epidemiological inference. We present EpiQAL, the first diagnostic benchmark for epidemiological question answering across diverse diseases, comprising three subsets built from open-access literature. The subsets respectively evaluate text-grounded factual recall, multi-step inference linking document evidence with epidemiological principles, and conclusion reconstruction with the Discussion section withheld. Construction combines expert-designed taxonomy guidance, multi-model verification, and retrieval-based difficulty control. Experiments on ten open models reveal that current LLMs show limited performance on epidemiological reasoning, with multi-step inference posing the greatest challenge. Model rankings shift across subsets, and scale alone does not predict success. Chain-of-Thought prompting benefits multi-step inference but yields mixed results elsewhere. EpiQAL provides fine-grained diagnostic signals for evidence grounding, inferential reasoning, and conclusion reconstruction. 

---
# Content vs. Form: What Drives the Writing Score Gap Across Socioeconomic Backgrounds? A Generated Panel Approach 

**Authors**: Nadav Kunievsky, Pedro Pertusi  

**Link**: [PDF](https://arxiv.org/pdf/2601.03469)  

**Abstract**: Students from different socioeconomic backgrounds exhibit persistent gaps in test scores, gaps that can translate into unequal educational and labor-market outcomes later in life. In many assessments, performance reflects not only what students know, but also how effectively they can communicate that knowledge. This distinction is especially salient in writing assessments, where scores jointly reward the substance of students' ideas and the way those ideas are expressed. As a result, observed score gaps may conflate differences in underlying content with differences in expressive skill. A central question, therefore, is how much of the socioeconomic-status (SES) gap in scores is driven by differences in what students say versus how they say it. We study this question using a large corpus of persuasive essays written by U.S. middle- and high-school students. We introduce a new measurement strategy that separates content from style by leveraging large language models to generate multiple stylistic variants of each essay. These rewrites preserve the underlying arguments while systematically altering surface expression, creating a "generated panel" that introduces controlled within-essay variation in style. This approach allows us to decompose SES gaps in writing scores into contributions from content and style. We find an SES gap of 0.67 points on a 1-6 scale. Approximately 69% of the gap is attributable to differences in essay content quality, Style differences account for 26% of the gap, and differences in evaluation standards across SES groups account for the remaining 5%. These patterns seems stable across demographic subgroups and writing tasks. More broadly, our approach shows how large language models can be used to generate controlled variation in observational data, enabling researchers to isolate and quantify the contributions of otherwise entangled factors. 

---
# FROST-Drive: Scalable and Efficient End-to-End Driving with a Frozen Vision Encoder 

**Authors**: Zeyu Dong, Yimin Zhu, Yu Wu, Yu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.03460)  

**Abstract**: End-to-end (E2E) models in autonomous driving aim to directly map sensor inputs to control commands, but their ability to generalize to novel and complex scenarios remains a key challenge. The common practice of fully fine-tuning the vision encoder on driving datasets potentially limits its generalization by causing the model to specialize too heavily in the training data. This work challenges the necessity of this training paradigm. We propose FROST-Drive, a novel E2E architecture designed to preserve and leverage the powerful generalization capabilities of a pretrained vision encoder from a Vision-Language Model (VLM). By keeping the encoder's weights frozen, our approach directly transfers the rich, generalized world knowledge from the VLM to the driving task. Our model architecture combines this frozen encoder with a transformer-based adapter for multimodal fusion and a GRU-based decoder for smooth waypoint generation. Furthermore, we introduce a custom loss function designed to directly optimize for Rater Feedback Score (RFS), a metric that prioritizes robust trajectory planning. We conduct extensive experiments on Waymo Open E2E Dataset, a large-scale datasets deliberately curated to capture the long-tail scenarios, demonstrating that our frozen-encoder approach significantly outperforms models that employ full fine-tuning. Our results provide substantial evidence that preserving the broad knowledge of a capable VLM is a more effective strategy for achieving robust, generalizable driving performance than intensive domain-specific adaptation. This offers a new pathway for developing vision-based models that can better handle the complexities of real-world application domains. 

---
# An Expectation-Maximization Algorithm for Domain Adaptation in Gaussian Causal Models 

**Authors**: Mohammad Ali Javidian  

**Link**: [PDF](https://arxiv.org/pdf/2601.03459)  

**Abstract**: We study the problem of imputing a designated target variable that is systematically missing in a shifted deployment domain, when a Gaussian causal DAG is available from a fully observed source domain. We propose a unified EM-based framework that combines source and target data through the DAG structure to transfer information from observed variables to the missing target. On the methodological side, we formulate a population EM operator in the DAG parameter space and introduce a first-order (gradient) EM update that replaces the costly generalized least-squares M-step with a single projected gradient step. Under standard local strong-concavity and smoothness assumptions and a BWY-style \cite{Balakrishnan2017EM} gradient-stability (bounded missing-information) condition, we show that this first-order EM operator is locally contractive around the true target parameters, yielding geometric convergence and finite-sample guarantees on parameter error and the induced target-imputation error in Gaussian SEMs under covariate shift and local mechanism shifts. Algorithmically, we exploit the known causal DAG to freeze source-invariant mechanisms and re-estimate only those conditional distributions directly affected by the shift, making the procedure scalable to higher-dimensional models. In experiments on a synthetic seven-node SEM, the 64-node MAGIC-IRRI genetic network, and the Sachs protein-signaling data, the proposed DAG-aware first-order EM algorithm improves target imputation accuracy over a fit-on-source Bayesian network and a Kiiveri-style EM baseline, with the largest gains under pronounced domain shift. 

---
# Automated Feedback Generation for Undergraduate Mathematics: Development and Evaluation of an AI Teaching Assistant 

**Authors**: Aron Gohr, Marie-Amelie Lawn, Kevin Gao, Inigo Serjeant, Stephen Heslip  

**Link**: [PDF](https://arxiv.org/pdf/2601.03458)  

**Abstract**: Intelligent tutoring systems have long enabled automated immediate feedback on student work when it is presented in a tightly structured format and when problems are very constrained, but reliably assessing free-form mathematical reasoning remains challenging.
We present a system that processes free-form natural language input, handles a wide range of edge cases, and comments competently not only on the technical correctness of submitted proofs, but also on style and presentation issues. We discuss the advantages and disadvantages of various approaches to the evaluation of such a system, and show that by the metrics we evaluate, the quality of the feedback generated is comparable to that produced by human experts when assessing early undergraduate homework. We stress-test our system with a small set of more advanced and unusual questions, and report both significant gaps and encouraging successes in that more challenging setting.
Our system uses large language models in a modular workflow. The workflow configuration is human-readable and editable without programming knowledge, and allows some intermediate steps to be precomputed or injected by the instructor.
A version of our tool is deployed on the Imperial mathematics homework platform Lambdafeedback. We report also on the integration of our tool into this platform. 

---
# Microeconomic Foundations of Multi-Agent Learning 

**Authors**: Nassim Helou  

**Link**: [PDF](https://arxiv.org/pdf/2601.03451)  

**Abstract**: Modern AI systems increasingly operate inside markets and institutions where data, behavior, and incentives are endogenous. This paper develops an economic foundation for multi-agent learning by studying a principal-agent interaction in a Markov decision process with strategic externalities, where both the principal and the agent learn over time. We propose a two-phase incentive mechanism that first estimates implementable transfers and then uses them to steer long-run dynamics; under mild regret-based rationality and exploration conditions, the mechanism achieves sublinear social-welfare regret and thus asymptotically optimal welfare. Simulations illustrate how even coarse incentives can correct inefficient learning under stateful externalities, highlighting the necessity of incentive-aware design for safe and welfare-aligned AI in markets and insurance. 

---
# Soft Contextualized Encoder For User Defined Text Classification 

**Authors**: Charu Maheshwari, Vyas Raina  

**Link**: [PDF](https://arxiv.org/pdf/2601.03450)  

**Abstract**: User-Defined Text Classification (UDTC) considers the challenge of classifying input text to user-specified, previously unseen classes, a setting that arises frequently in real-world applications such as enterprise analytics, content moderation, and domain-specific information retrieval. We propose a soft-contextualized encoder architecture for UDTC which contextualizes each candidate label with the label set and a static soft prompt representation of the input query. Training on diverse, multi-source datasets enables the model to generalize effectively to zero-shot classification over entirely unseen topic sets drawn from arbitrary domains. We evaluate the proposed architecture both on held-out in-distribution test data and on multiple unseen UDTC benchmarks. Across datasets, the model achieves state-of-the-art performance, consistently outperforming or matching the baselines. 

---
# Grading Scale Impact on LLM-as-a-Judge: Human-LLM Alignment Is Highest on 0-5 Grading Scale 

**Authors**: Weiyue Li, Minda Zhao, Weixuan Dong, Jiahui Cai, Yuze Wei, Michael Pocress, Yi Li, Wanyan Yuan, Xiaoyue Wang, Ruoyu Hou, Kaiyuan Lou, Wenqi Zeng, Yutong Yang, Yilun Du, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03444)  

**Abstract**: Large language models (LLMs) are increasingly used as automated evaluators, yet prior works demonstrate that these LLM judges often lack consistency in scoring when the prompt is altered. However, the effect of the grading scale itself remains underexplored. We study the LLM-as-a-judge problem by comparing two kinds of raters: humans and LLMs. We collect ratings from both groups on three scales and across six benchmarks that include objective, open-ended subjective, and mixed tasks. Using intraclass correlation coefficients (ICC) to measure absolute agreement, we find that LLM judgments are not perfectly consistent across scales on subjective benchmarks, and that the choice of scale substantially shifts human-LLM agreement, even when within-group panel reliability is high. Aggregated over tasks, the grading scale of 0-5 yields the strongest human-LLM alignment. We further demonstrate that pooled reliability can mask benchmark heterogeneity and reveal systematic subgroup differences in alignment across gender groups, strengthening the importance of scale design and sub-level diagnostics as essential components of LLM-as-a-judge protocols. 

---
# Discriminating real and synthetic super-resolved audio samples using embedding-based classifiers 

**Authors**: Mikhail Silaev, Konstantinos Drossos, Tuomas Virtanen  

**Link**: [PDF](https://arxiv.org/pdf/2601.03443)  

**Abstract**: Generative adversarial networks (GANs) and diffusion models have recently achieved state-of-the-art performance in audio super-resolution (ADSR), producing perceptually convincing wideband audio from narrowband inputs. However, existing evaluations primarily rely on signal-level or perceptual metrics, leaving open the question of how closely the distributions of synthetic super-resolved and real wideband audio match. Here we address this problem by analyzing the separability of real and super-resolved audio in various embedding spaces. We consider both middle-band ($4\to 16$~kHz) and full-band ($16\to 48$~kHz) upsampling tasks for speech and music, training linear classifiers to distinguish real from synthetic samples based on multiple types of audio embeddings. Comparisons with objective metrics and subjective listening tests reveal that embedding-based classifiers achieve near-perfect separation, even when the generated audio attains high perceptual quality and state-of-the-art metric scores. This behavior is consistent across datasets and models, including recent diffusion-based approaches, highlighting a persistent gap between perceptual quality and true distributional fidelity in ADSR models. 

---
# MARVEL: A Multi Agent-based Research Validator and Enabler using Large Language Models 

**Authors**: Nikhil Mukund, Yifang Luo, Fan Zhang, Lisa Barsotti, Erik Katsavounidis  

**Link**: [PDF](https://arxiv.org/pdf/2601.03436)  

**Abstract**: We present MARVEL (this https URL), a locally deployable, open-source framework for domain-aware question answering and assisted scientific research. It is designed to address the increasing demands of a digital assistant for scientific groups that can read highly technical data, cite precisely, and operate within authenticated networks. MARVEL combines a fast path for straightforward queries with a more deliberate DeepSearch mode that integrates retrieval-augmented generation and Monte Carlo Tree Search. It explores complementary subqueries, allocates more compute to promising branches, and maintains a global evidence ledger that preserves sources during drafting. We applied this framework in the context of gravitational-wave research related to the Laser Interferometer Gravitational-wave Observatory. Answers are grounded in a curated semantic index of research literature, doctoral theses, LIGO documents, and long-running detector electronic logbooks, with targeted web searches when appropriate. Because direct benchmarking against commercial LLMs cannot be performed on private data, we evaluated MARVEL on two publicly available surrogate datasets that capture comparable semantic and technical characteristics. On these benchmarks, MARVEL matches a GPT-4o mini baseline on literature-centric queries and substantially outperforms it on detector-operations content, where domain retrieval and guided reasoning are decisive. By making the complete framework and evaluation datasets openly available, we aim to provide a reproducible foundation for developing domain-specific scientific assistants. 

---
# The Illusion of Specialization: Unveiling the Domain-Invariant "Standing Committee" in Mixture-of-Experts Models 

**Authors**: Yan Wang, Yitao Xu, Nanhan Shen, Jinyan Su, Jimin Huang, Zining Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03425)  

**Abstract**: Mixture of Experts models are widely assumed to achieve domain specialization through sparse routing. In this work, we question this assumption by introducing COMMITTEEAUDIT, a post hoc framework that analyzes routing behavior at the level of expert groups rather than individual experts. Across three representative models and the MMLU benchmark, we uncover a domain-invariant Standing Committee. This is a compact coalition of routed experts that consistently captures the majority of routing mass across domains, layers, and routing budgets, even when architectures already include shared experts. Qualitative analysis further shows that Standing Committees anchor reasoning structure and syntax, while peripheral experts handle domain-specific knowledge. These findings reveal a strong structural bias toward centralized computation, suggesting that specialization in Mixture of Experts models is far less pervasive than commonly believed. This inherent bias also indicates that current training objectives, such as load-balancing losses that enforce uniform expert utilization, may be working against the model's natural optimization path, thereby limiting training efficiency and performance. 

---
# Spectral Archaeology: The Causal Topology of Model Evolution 

**Authors**: Valentin Noël  

**Link**: [PDF](https://arxiv.org/pdf/2601.03424)  

**Abstract**: Behavioral benchmarks tell us \textit{what} a model does, but not \textit{how}. We introduce a training-free mechanistic probe using attention-graph spectra. Treating each layer as a token graph, we compute algebraic connectivity ($\lambda_2$), smoothness, and spectral entropy. Across 12 models and 10 languages, these measures yield stable ``spectral fingerprints'' that expose discontinuities missed by standard evaluation.
We report four results. (1) Models undergoing specific curriculum transitions (e.g., code-to-chat) show an English-only, syntax-triggered connectivity failure on non-canonical constructions, reaching $\Delta\lambda_2 \approx -0.76$. We term this scar \textit{Passive-Triggered Connectivity Collapse} (PTCC). Analysis of the Phi lineage reveals that PTCC appears and resolves across developmental stages, implicating brittle curriculum shifts rather than synthetic data per se. (2) PTCC reflects a specialization trade-off: strengthened formal routing at the expense of stylistic flexibility. (3) We identify four recurrent processing strategies; simple frozen-threshold rules enable perfect forensic identification across lineages. (4) Mechanistically, PTCC localizes to a sparse Layer 2 ``compensatory patch'' of heads that fails under syntactic stress; activation steering can partially restore connectivity, recovering $\approx 38\%$ of lost information flow.
Finally, dominant topological regimes track tokenization density more than language identity, suggesting ``healthy'' geometry varies systematically across scripts. Overall, attention-graph spectra provide a practical tool for auditing and training-regime verification. 

---
# Training-Free Adaptation of New-Generation LLMs using Legacy Clinical Models 

**Authors**: Sasha Ronaghi, Chloe Stanwyck, Asad Aali, Amir Ronaghi, Miguel Fuentes, Tina Hernandez-Boussard, Emily Alsentzer  

**Link**: [PDF](https://arxiv.org/pdf/2601.03423)  

**Abstract**: Adapting language models to the clinical domain through continued pretraining and fine-tuning requires costly retraining for each new model generation. We propose Cross-Architecture Proxy Tuning (CAPT), a model-ensembling approach that enables training-free adaptation of state-of-the-art general-domain models using existing clinical models. CAPT supports models with disjoint vocabularies, leveraging contrastive decoding to selectively inject clinically relevant signals while preserving the general-domain model's reasoning and fluency. On six clinical classification and text-generation tasks, CAPT with a new-generation general-domain model and an older-generation clinical model consistently outperforms both models individually and state-of-the-art ensembling approaches (average +17.6% over UniTE, +41.4% over proxy tuning across tasks). Through token-level analysis and physician case studies, we demonstrate that CAPT amplifies clinically actionable language, reduces context errors, and increases clinical specificity. 

---
# Jailbreaking LLMs Without Gradients or Priors: Effective and Transferable Attacks 

**Authors**: Zhakshylyk Nurlanov, Frank R. Schmidt, Florian Bernard  

**Link**: [PDF](https://arxiv.org/pdf/2601.03420)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in safety-critical domains, rigorously evaluating their robustness against adversarial jailbreaks is essential. However, current safety evaluations often overestimate robustness because existing automated attacks are limited by restrictive assumptions. They typically rely on handcrafted priors or require white-box access for gradient propagation. We challenge these constraints by demonstrating that token-level iterative optimization can succeed without gradients or priors. We introduce RAILS (RAndom Iterative Local Search), a framework that operates solely on model logits. RAILS matches the effectiveness of gradient-based methods through two key innovations: a novel auto-regressive loss that enforces exact prefix matching, and a history-based selection strategy that bridges the gap between the proxy optimization objective and the true attack success rate. Crucially, by eliminating gradient dependency, RAILS enables cross-tokenizer ensemble attacks. This allows for the discovery of shared adversarial patterns that generalize across disjoint vocabularies, significantly enhancing transferability to closed-source systems. Empirically, RAILS achieves near 100% success rates on multiple open-source models and high black-box attack transferability to closed-source systems like GPT and Gemini. 

---
# Tigrinya Number Verbalization: Rules, Algorithm, and Implementation 

**Authors**: Fitsum Gaim, Issayas Tesfamariam  

**Link**: [PDF](https://arxiv.org/pdf/2601.03403)  

**Abstract**: We present a systematic formalization of Tigrinya cardinal and ordinal number verbalization, addressing a gap in computational resources for the language. This work documents the canonical rules governing the expression of numerical values in spoken Tigrinya, including the conjunction system, scale words, and special cases for dates, times, and currency. We provide a formal algorithm for number-to-word conversion and release an open-source implementation. Evaluation of frontier large language models (LLMs) reveals significant gaps in their ability to accurately verbalize Tigrinya numbers, underscoring the need for explicit rule documentation. This work serves language modeling, speech synthesis, and accessibility applications targeting Tigrinya-speaking communities. 

---
# Eye-Q: A Multilingual Benchmark for Visual Word Puzzle Solving and Image-to-Phrase Reasoning 

**Authors**: Ali Najar, Alireza Mirrokni, Arshia Izadyari, Sadegh Mohammadian, Amir Homayoon Sharifizade, Asal Meskin, Mobin Bagherian, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2601.03400)  

**Abstract**: Vision-Language Models (VLMs) have achieved strong performance on standard vision-language benchmarks, yet often rely on surface-level recognition rather than deeper reasoning. We propose visual word puzzles as a challenging alternative, as they require discovering implicit visual cues, generating and revising hypotheses, and mapping perceptual evidence to non-literal concepts in ways that are difficult to solve via literal grounding, OCR-heavy shortcuts, or simple retrieval-style matching. We introduce Eye-Q, a multilingual benchmark designed to assess this form of complex visual understanding. Eye-Q contains 1,343 puzzles in which a model observes a conceptually dense scene with a brief description and must infer a specific target word or phrase. The puzzles are intentionally unstructured and cue-implicit, with distractors and contextual relationships that demand selective attention, abstraction, and associative inference. The benchmark spans English, Persian, Arabic, and cross-lingual puzzles. We evaluate state-of-the-art VLMs using an open-ended, human-aligned protocol that probes hypothesis formation and revision under lightweight assistance. Results reveal substantial performance gaps, especially on abstract and cross-lingual puzzles, highlighting limitations in current models' ability to construct and search over appropriate conceptual representations for flexible image-to-phrase inference; maximum accuracy reaches only 60.27%. 

---
# Metaphors are a Source of Cross-Domain Misalignment of Large Reasoning Models 

**Authors**: Zhibo Hu, Chen Wang, Yanfeng Shu, Hye-young Paik, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03388)  

**Abstract**: Earlier research has shown that metaphors influence human's decision making, which raises the question of whether metaphors also influence large language models (LLMs)' reasoning pathways, considering their training data contain a large number of metaphors. In this work, we investigate the problem in the scope of the emergent misalignment problem where LLMs can generalize patterns learned from misaligned content in one domain to another domain. We discover a strong causal relationship between metaphors in training data and the misalignment degree of LLMs' reasoning contents. With interventions using metaphors in pre-training, fine-tuning and re-alignment phases, models' cross-domain misalignment degrees change significantly. As we delve deeper into the causes behind this phenomenon, we observe that there is a connection between metaphors and the activation of global and local latent features of large reasoning models. By monitoring these latent features, we design a detector that predict misaligned content with high accuracy. 

---
# MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models 

**Authors**: Yang Shi, Yifeng Xie, Minzhe Guo, Liangsi Lu, Mingxuan Huang, Jingchao Wang, Zhihong Zhu, Boyan Xu, Zhiqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03331)  

**Abstract**: Recent advances in Vision-Language Models (VLMs) have improved performance in multi-modal learning, raising the question of whether these models truly understand the content they process. Crucially, can VLMs detect when a reasoning process is wrong and identify its error type? To answer this, we present MMErroR, a multi-modal benchmark of 2,013 samples, each embedding a single coherent reasoning error. These samples span 24 subdomains across six top-level domains, ensuring broad coverage and taxonomic richness. Unlike existing benchmarks that focus on answer correctness, MMErroR targets a process-level, error-centric evaluation that requires models to detect incorrect reasoning and classify the error type within both visual and linguistic contexts. We evaluate 20 advanced VLMs, even the best model (Gemini-3.0-Pro) classifies the error in only 66.47\% of cases, underscoring the challenge of identifying erroneous reasoning. Furthermore, the ability to accurately identify errors offers valuable insights into the capabilities of multi-modal reasoning models. Project Page: this https URL 

---
# Attention mechanisms in neural networks 

**Authors**: Hasi Hays  

**Link**: [PDF](https://arxiv.org/pdf/2601.03329)  

**Abstract**: Attention mechanisms represent a fundamental paradigm shift in neural network architectures, enabling models to selectively focus on relevant portions of input sequences through learned weighting functions. This monograph provides a comprehensive and rigorous mathematical treatment of attention mechanisms, encompassing their theoretical foundations, computational properties, and practical implementations in contemporary deep learning systems. Applications in natural language processing, computer vision, and multimodal learning demonstrate the versatility of attention mechanisms. We examine language modeling with autoregressive transformers, bidirectional encoders for representation learning, sequence-to-sequence translation, Vision Transformers for image classification, and cross-modal attention for vision-language tasks. Empirical analysis reveals training characteristics, scaling laws that relate performance to model size and computation, attention pattern visualizations, and performance benchmarks across standard datasets. We discuss the interpretability of learned attention patterns and their relationship to linguistic and visual structures. The monograph concludes with a critical examination of current limitations, including computational scalability, data efficiency, systematic generalization, and interpretability challenges. 

---
# Extreme-value forest fire prediction A study of the Loss Function in an Ordinality Scheme 

**Authors**: Nicolas Caron, Christophe Guyeux, Hassan Noura, Benjamin Aynes  

**Link**: [PDF](https://arxiv.org/pdf/2601.03327)  

**Abstract**: Wildfires are highly imbalanced natural hazards in both space and severity, making the prediction of extreme events particularly challenging. In this work, we introduce the first ordinal classification framework for forecasting wildfire severity levels directly aligned with operational decision-making in France. Our study investigates the influence of loss-function design on the ability of neural models to predict rare yet critical high-severity fire occurrences. We compare standard cross-entropy with several ordinal-aware objectives, including the proposed probabilistic TDeGPD loss derived from a truncated discrete exponentiated Generalized Pareto Distribution. Through extensive benchmarking over multiple architectures and real operational data, we show that ordinal supervision substantially improves model performance over conventional approaches. In particular, the Weighted Kappa Loss (WKLoss) achieves the best overall results, with more than +0.1 IoU gain on the most extreme severity classes while maintaining competitive calibration quality. However, performance remains limited for the rarest events due to their extremely low representation in the dataset. These findings highlight the importance of integrating both severity ordering, data imbalance considerations, and seasonality risk into wildfire forecasting systems. Future work will focus on incorporating seasonal dynamics and uncertainty information into training to further improve the reliability of extreme-event prediction. 

---
# Bare-Metal Tensor Virtualization: Overcoming the Memory Wall in Edge-AI Inference on ARM64 

**Authors**: Bugra Kilictas, Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2601.03324)  

**Abstract**: The deployment of Large Language Models (LLMs) on edge devices is fundamentally constrained by the "Memory Wall" the bottleneck where data movement latency outstrips arithmetic throughput. Standard inference runtimes often incur significant overhead through high-level abstractions, dynamic dispatch, and unaligned memory access patterns. In this work, we present a novel "Virtual Tensor Core" architecture implemented in software, optimized specifically for ARM64 microarchitectures (Apple Silicon). By bypassing standard library containers in favor of direct memory mapping (mmap) and implementing hand-tuned NEON SIMD kernels, we achieve a form of "Software-Defined Direct Memory Access (DMA)." Our proposed Tensor Virtualization Layout (TVL) guarantees 100% cache line utilization for weight matrices, while our zero-copy loader eliminates initialization latency. Experimental results on a 110M parameter model demonstrate a stable throughput of >60 tokens/second on M2 hardware. While proprietary hardware accelerators (e.g., Apple AMX) can achieve higher peak throughput, our architecture provides a fully open, portable, and deterministic reference implementation for studying the memory bottleneck on general-purpose ARM silicon, meeting the 200ms psycholinguistic latency threshold without opaque dependencies. 

---
# HEEGNet: Hyperbolic Embeddings for EEG 

**Authors**: Shanglin Li, Shiwen Chu, Okan Koç, Yi Ding, Qibin Zhao, Motoaki Kawanabe, Ziheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.03322)  

**Abstract**: Electroencephalography (EEG)-based brain-computer interfaces facilitate direct communication with a computer, enabling promising applications in human-computer interactions. However, their utility is currently limited because EEG decoding often suffers from poor generalization due to distribution shifts across domains (e.g., subjects). Learning robust representations that capture underlying task-relevant information would mitigate these shifts and improve generalization. One promising approach is to exploit the underlying hierarchical structure in EEG, as recent studies suggest that hierarchical cognitive processes, such as visual processing, can be encoded in EEG. While many decoding methods still rely on Euclidean embeddings, recent work has begun exploring hyperbolic geometry for EEG. Hyperbolic spaces, regarded as the continuous analogue of tree structures, provide a natural geometry for representing hierarchical data. In this study, we first empirically demonstrate that EEG data exhibit hyperbolicity and show that hyperbolic embeddings improve generalization. Motivated by these findings, we propose HEEGNet, a hybrid hyperbolic network architecture to capture the hierarchical structure in EEG and learn domain-invariant hyperbolic embeddings. To this end, HEEGNet combines both Euclidean and hyperbolic encoders and employs a novel coarse-to-fine domain adaptation strategy. Extensive experiments on multiple public EEG datasets, covering visual evoked potentials, emotion recognition, and intracranial EEG, demonstrate that HEEGNet achieves state-of-the-art performance. 

---
# Aligning Findings with Diagnosis: A Self-Consistent Reinforcement Learning Framework for Trustworthy Radiology Reporting 

**Authors**: Kun Zhao, Siyuan Dai, Pan Wang, Jifeng Song, Hui Ji, Chenghua Lin, Liang Zhan, Haoteng Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03321)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown strong potential for radiology report generation, yet their clinical translation is hindered by architectural heterogeneity and the prevalence of factual hallucinations. Standard supervised fine-tuning often fails to strictly align linguistic outputs with visual evidence, while existing reinforcement learning approaches struggle with either prohibitive computational costs or limited exploration. To address these challenges, we propose a comprehensive framework for self-consistent radiology report generation. First, we conduct a systematic evaluation to identify optimal vision encoder and LLM backbone configurations for medical imaging. Building on this foundation, we introduce a novel "Reason-then-Summarize" architecture optimized via Group Relative Policy Optimization (GRPO). This framework restructures generation into two distinct components: a think block for detailed findings and an answer block for structured disease labels. By utilizing a multi-dimensional composite reward function, we explicitly penalize logical discrepancies between the generated narrative and the final diagnosis. Extensive experiments on the MIMIC-CXR benchmark demonstrate that our method achieves state-of-the-art performance in clinical efficacy metrics and significantly reduces hallucinations compared to strong supervised baselines. 

---
# Ratio-Variance Regularized Policy Optimization for Efficient LLM Fine-tuning 

**Authors**: Yu Luo, Shuo Han, Yihan Hu, Dong Li, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2601.03320)  

**Abstract**: On-policy reinforcement learning (RL), particularly Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO), has become the dominant paradigm for fine-tuning large language models (LLMs). While policy ratio clipping stabilizes training, this heuristic hard constraint incurs a fundamental cost: it indiscriminately truncates gradients from high-return yet high-divergence actions, suppressing rare but highly informative "eureka moments" in complex reasoning. Moreover, once data becomes slightly stale, hard clipping renders it unusable, leading to severe sample inefficiency. In this work, we revisit the trust-region objective in policy optimization and show that explicitly constraining the \emph{variance (second central moment) of the policy ratio} provides a principled and smooth relaxation of hard clipping. This distributional constraint stabilizes policy updates while preserving gradient signals from valuable trajectories. Building on this insight, we propose $R^2VPO$ (Ratio-Variance Regularized Policy Optimization), a novel primal-dual framework that supports stable on-policy learning and enables principled off-policy data reuse by dynamically reweighting stale samples rather than discarding them. We extensively evaluate $R^2VPO$ on fine-tuning state-of-the-art LLMs, including DeepSeek-Distill-Qwen-1.5B and the openPangu-Embedded series (1B and 7B), across challenging mathematical reasoning benchmarks. Experimental results show that $R^2VPO$ consistently achieves superior asymptotic performance, with average relative gains of up to 17% over strong clipping-based baselines, while requiring approximately 50% fewer rollouts to reach convergence. These findings establish ratio-variance control as a promising direction for improving both stability and data efficiency in RL-based LLM alignment. 

---
# CaricatureGS: Exaggerating 3D Gaussian Splatting Faces With Gaussian Curvature 

**Authors**: Eldad Matmon, Amit Bracha, Noam Rotstein, Ron Kimmel  

**Link**: [PDF](https://arxiv.org/pdf/2601.03319)  

**Abstract**: A photorealistic and controllable 3D caricaturization framework for faces is introduced. We start with an intrinsic Gaussian curvature-based surface exaggeration technique, which, when coupled with texture, tends to produce over-smoothed renders. To address this, we resort to 3D Gaussian Splatting (3DGS), which has recently been shown to produce realistic free-viewpoint avatars. Given a multiview sequence, we extract a FLAME mesh, solve a curvature-weighted Poisson equation, and obtain its exaggerated form. However, directly deforming the Gaussians yields poor results, necessitating the synthesis of pseudo-ground-truth caricature images by warping each frame to its exaggerated 2D representation using local affine transformations. We then devise a training scheme that alternates real and synthesized supervision, enabling a single Gaussian collection to represent both natural and exaggerated avatars. This scheme improves fidelity, supports local edits, and allows continuous control over the intensity of the caricature. In order to achieve real-time deformations, an efficient interpolation between the original and exaggerated surfaces is introduced. We further analyze and show that it has a bounded deviation from closed-form solutions. In both quantitative and qualitative evaluations, our results outperform prior work, delivering photorealistic, geometry-controlled caricature avatars. 

---
# Deep Learning-Based Image Recognition for Soft-Shell Shrimp Classification 

**Authors**: Yun-Hao Zhang, I-Hsien Ting, Dario Liberona, Yun-Hsiu Liu, Kazunori Minetaki  

**Link**: [PDF](https://arxiv.org/pdf/2601.03317)  

**Abstract**: With the integration of information technology into aquaculture, production has become more stable and continues to grow annually. As consumer demand for high-quality aquatic products rises, freshness and appearance integrity are key concerns. In shrimp-based processed foods, freshness declines rapidly post-harvest, and soft-shell shrimp often suffer from head-body separation after cooking or freezing, affecting product appearance and consumer perception. To address these issues, this study leverages deep learning-based image recognition for automated classification of white shrimp immediately after harvest. A convolutional neural network (CNN) model replaces manual sorting, enhancing classification accuracy, efficiency, and consistency. By reducing processing time, this technology helps maintain freshness and ensures that shrimp transportation businesses meet customer demands more effectively. 

---
# Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts 

**Authors**: Dhruv Trehan, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2601.03315)  

**Abstract**: We report a case study of four end-to-end attempts to autonomously generate ML research papers using a pipeline of six LLM agents mapped to stages of the scientific workflow. Of these four, three attempts failed during implementation or evaluation. One completed the pipeline and was accepted to Agents4Science 2025, an experimental inaugural venue that required AI systems as first authors, passing both human and multi-AI review. From these attempts, we document six recurring failure modes: bias toward training data defaults, implementation drift under execution pressure, memory and context degradation across long-horizon tasks, overexcitement that declares success despite obvious failures, insufficient domain intelligence, and weak scientific taste in experimental design. We conclude by discussing four design principles for more robust AI-scientist systems, implications for autonomous scientific discovery, and we release all prompts, artifacts, and outputs at this https URL 

---
# VLM4VLA: Revisiting Vision-Language-Models in Vision-Language-Action Models 

**Authors**: Jianke Zhang, Xiaoyu Chen, Qiuyue Wang, Mingsheng Li, Yanjiang Guo, Yucheng Hu, Jiajun Zhang, Shuai Bai, Junyang Lin, Jianyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.03309)  

**Abstract**: Vision-Language-Action (VLA) models, which integrate pretrained large Vision-Language Models (VLM) into their policy backbone, are gaining significant attention for their promising generalization capabilities. This paper revisits a fundamental yet seldom systematically studied question: how VLM choice and competence translate to downstream VLA policies performance? We introduce VLM4VLA, a minimal adaptation pipeline that converts general-purpose VLMs into VLA policies using only a small set of new learnable parameters for fair and efficient comparison. Despite its simplicity, VLM4VLA proves surprisingly competitive with more sophisticated network designs. Through extensive empirical studies on various downstream tasks across three benchmarks, we find that while VLM initialization offers a consistent benefit over training from scratch, a VLM's general capabilities are poor predictors of its downstream task performance. This challenges common assumptions, indicating that standard VLM competence is necessary but insufficient for effective embodied control. We further investigate the impact of specific embodied capabilities by fine-tuning VLMs on seven auxiliary embodied tasks (e.g., embodied QA, visual pointing, depth estimation). Contrary to intuition, improving a VLM's performance on specific embodied skills does not guarantee better downstream control performance. Finally, modality-level ablations identify the visual module in VLM, rather than the language component, as the primary performance bottleneck. We demonstrate that injecting control-relevant supervision into the vision encoder of the VLM yields consistent gains, even when the encoder remains frozen during downstream fine-tuning. This isolates a persistent domain gap between current VLM pretraining objectives and the requirements of embodied action-planning. 

---
# Mass Concept Erasure in Diffusion Models with Concept Hierarchy 

**Authors**: Jiahang Tu, Ye Li, Yiming Wu, Hanbin Zhao, Chao Zhang, Hui Qian  

**Link**: [PDF](https://arxiv.org/pdf/2601.03305)  

**Abstract**: The success of diffusion models has raised concerns about the generation of unsafe or harmful content, prompting concept erasure approaches that fine-tune modules to suppress specific concepts while preserving general generative capabilities. However, as the number of erased concepts grows, these methods often become inefficient and ineffective, since each concept requires a separate set of fine-tuned parameters and may degrade the overall generation quality. In this work, we propose a supertype-subtype concept hierarchy that organizes erased concepts into a parent-child structure. Each erased concept is treated as a child node, and semantically related concepts (e.g., macaw, and bald eagle) are grouped under a shared parent node, referred to as a supertype concept (e.g., bird). Rather than erasing concepts individually, we introduce an effective and efficient group-wise suppression method, where semantically similar concepts are grouped and erased jointly by sharing a single set of learnable parameters. During the erasure phase, standard diffusion regularization is applied to preserve denoising process in unmasked regions. To mitigate the degradation of supertype generation caused by excessive erasure of semantically related subtypes, we propose a novel method called Supertype-Preserving Low-Rank Adaptation (SuPLoRA), which encodes the supertype concept information in the frozen down-projection matrix and updates only the up-projection matrix during erasure. Theoretical analysis demonstrates the effectiveness of SuPLoRA in mitigating generation performance degradation. We construct a more challenging benchmark that requires simultaneous erasure of concepts across diverse domains, including celebrities, objects, and pornographic content. 

---
# AI-Driven Cybersecurity Threats: A Survey of Emerging Risks and Defensive Strategies 

**Authors**: Sai Teja Erukude, Viswa Chaitanya Marella, Suhasnadh Reddy Veluru  

**Link**: [PDF](https://arxiv.org/pdf/2601.03304)  

**Abstract**: Artificial Intelligence's dual-use nature is revolutionizing the cybersecurity landscape, introducing new threats across four main categories: deepfakes and synthetic media, adversarial AI attacks, automated malware, and AI-powered social engineering. This paper aims to analyze emerging risks, attack mechanisms, and defense shortcomings related to AI in cybersecurity. We introduce a comparative taxonomy connecting AI capabilities with threat modalities and defenses, review over 70 academic and industry references, and identify impactful opportunities for research, such as hybrid detection pipelines and benchmarking frameworks. The paper is structured thematically by threat type, with each section addressing technical context, real-world incidents, legal frameworks, and countermeasures. Our findings emphasize the urgency for explainable, interdisciplinary, and regulatory-compliant AI defense systems to maintain trust and security in digital ecosystems. 

---
# CageDroneRF: A Large-Scale RF Benchmark and Toolkit for Drone Perception 

**Authors**: Mohammad Rostami, Atik Faysal, Hongtao Xia, Hadi Kasasbeh, Ziang Gao, Huaxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03302)  

**Abstract**: We present CageDroneRF (CDRF), a large-scale benchmark for Radio-Frequency (RF) drone detection and identification built from real-world captures and systematically generated synthetic variants. CDRF addresses the scarcity and limited diversity of existing RF datasets by coupling extensive raw recordings with a principled augmentation pipeline that (i) precisely controls Signal-to-Noise Ratio (SNR), (ii) injects interfering emitters, and (iii) applies frequency shifts with label-consistent bounding-box transformations for detection. This dataset spans a wide range of contemporary drone models, many unavailable in current public datasets, and acquisition conditions, derived from data collected at the Rowan University campus and within a controlled RF-cage facility. CDRF is released with interoperable open-source tools for data generation, preprocessing, augmentation, and evaluation that also operate on existing public benchmarks. CDRF enables standardized benchmarking for classification, open-set recognition, and object detection, supporting rigorous comparisons and reproducible pipelines. By releasing this comprehensive benchmark and tooling, CDRF aims to accelerate progress toward robust, generalizable RF perception models. 

---
# PC2P: Multi-Agent Path Finding via Personalized-Enhanced Communication and Crowd Perception 

**Authors**: Guotao Li, Shaoyun Xu, Yuexing Hao, Yang Wang, Yuhui Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.03301)  

**Abstract**: Distributed Multi-Agent Path Finding (MAPF) integrated with Multi-Agent Reinforcement Learning (MARL) has emerged as a prominent research focus, enabling real-time cooperative decision-making in partially observable environments through inter-agent communication. However, due to insufficient collaborative and perceptual capabilities, existing methods are inadequate for scaling across diverse environmental conditions. To address these challenges, we propose PC2P, a novel distributed MAPF method derived from a Q-learning-based MARL framework. Initially, we introduce a personalized-enhanced communication mechanism based on dynamic graph topology, which ascertains the core aspects of ``who" and ``what" in interactive process through three-stage operations: selection, generation, and aggregation. Concurrently, we incorporate local crowd perception to enrich agents' heuristic observation, thereby strengthening the model's guidance for effective actions via the integration of static spatial constraints and dynamic occupancy changes. To resolve extreme deadlock issues, we propose a region-based deadlock-breaking strategy that leverages expert guidance to implement efficient coordination within confined areas. Experimental results demonstrate that PC2P achieves superior performance compared to state-of-the-art distributed MAPF methods in varied environments. Ablation studies further confirm the effectiveness of each module for overall performance. 

---
# 130k Lines of Formal Topology in Two Weeks: Simple and Cheap Autoformalization for Everyone? 

**Authors**: Josef Urban  

**Link**: [PDF](https://arxiv.org/pdf/2601.03298)  

**Abstract**: This is a brief description of a project that has already autoformalized a large portion of the general topology from the Munkres textbook (which has in total 241 pages in 7 chapters and 39 sections). The project has been running since November 21, 2025 and has as of January 4, 2026, produced 160k lines of formalized topology. Most of it (about 130k lines) have been done in two weeks,from December 22 to January 4, for an LLM subscription cost of about \$100. This includes a 3k-line proof of Urysohn's lemma, a 2k-line proof of Urysohn's Metrization theorem, over 10k-line proof of the Tietze extension theorem, and many more (in total over 1.5k lemmas/theorems). The approach is quite simple and cheap: build a long-running feedback loop between an LLM and a reasonably fast proof checker equipped with a core foundational library. The LLM is now instantiated as ChatGPT (mostly 5.2) or Claude Sonnet (4.5) run through the respective Codex or Claude Code command line interfaces. The proof checker is Chad Brown's higher-order set theory system Megalodon, and the core library is Brown's formalization of basic set theory and surreal numbers (including reals, etc). The rest is some prompt engineering and technical choices which we describe here. Based on the fast progress, low cost, virtually unknown ITP/library, and the simple setup available to everyone, we believe that (auto)formalization may become quite easy and ubiquitous in 2026, regardless of which proof assistant is used. 

---
# AgentMark: Utility-Preserving Behavioral Watermarking for Agents 

**Authors**: Kaibo Huang, Jin Tan, Yukun Wei, Wanling Li, Zipei Zhang, Hui Tian, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.03294)  

**Abstract**: LLM-based agents are increasingly deployed to autonomously solve complex tasks, raising urgent needs for IP protection and regulatory provenance. While content watermarking effectively attributes LLM-generated outputs, it fails to directly identify the high-level planning behaviors (e.g., tool and subgoal choices) that govern multi-step execution. Critically, watermarking at the planning-behavior layer faces unique challenges: minor distributional deviations in decision-making can compound during long-term agent operation, degrading utility, and many agents operate as black boxes that are difficult to intervene in directly. To bridge this gap, we propose AgentMark, a behavioral watermarking framework that embeds multi-bit identifiers into planning decisions while preserving utility. It operates by eliciting an explicit behavior distribution from the agent and applying distribution-preserving conditional sampling, enabling deployment under black-box APIs while remaining compatible with action-layer content watermarking. Experiments across embodied, tool-use, and social environments demonstrate practical multi-bit capacity, robust recovery from partial logs, and utility preservation. The code is available at this https URL. 

---
# Lightweight Transformer Architectures for Edge Devices in Real-Time Applications 

**Authors**: Hema Hariharan Samson  

**Link**: [PDF](https://arxiv.org/pdf/2601.03290)  

**Abstract**: The deployment of transformer-based models on resource-constrained edge devices represents a critical challenge in enabling real-time artificial intelligence applications. This comprehensive survey examines lightweight transformer architectures specifically designed for edge deployment, analyzing recent advances in model compression, quantization, pruning, and knowledge distillation techniques. We systematically review prominent lightweight variants including MobileBERT, TinyBERT, DistilBERT, EfficientFormer, EdgeFormer, and MobileViT, providing detailed performance benchmarks on standard datasets such as GLUE, SQuAD, ImageNet-1K, and COCO. Our analysis encompasses current industry adoption patterns across major hardware platforms (NVIDIA Jetson, Qualcomm Snapdragon, Apple Neural Engine, ARM architectures), deployment frameworks (TensorFlow Lite, ONNX Runtime, PyTorch Mobile, CoreML), and optimization strategies. Experimental results demonstrate that modern lightweight transformers can achieve 75-96% of full-model accuracy while reducing model size by 4-10x and inference latency by 3-9x, enabling deployment on devices with as little as 2-5W power consumption. We identify sparse attention mechanisms, mixed-precision quantization (INT8/FP16), and hardware-aware neural architecture search as the most effective optimization strategies. Novel findings include memory-bandwidth bottleneck analysis revealing 15-40M parameter models achieve optimal hardware utilization (60-75% efficiency), quantization sweet spots for different model types, and comprehensive energy efficiency profiling across edge platforms. We establish real-time performance boundaries and provide a practical 6-step deployment pipeline achieving 8-12x size reduction with less than 2% accuracy degradation. 

---
# Automated Post-Incident Policy Gap Analysis via Threat-Informed Evidence Mapping using Large Language Models 

**Authors**: Huan Lin Oh, Jay Yong Jun Jie, Mandy Lee Ling Siu, Jonathan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2601.03287)  

**Abstract**: Cybersecurity post-incident reviews are essential for identifying control failures and improving organisational resilience, yet they remain labour-intensive, time-consuming, and heavily reliant on expert judgment. This paper investigates whether Large Language Models (LLMs) can augment post-incident review workflows by autonomously analysing system evidence and identifying security policy gaps. We present a threat-informed, agentic framework that ingests log data, maps observed behaviours to the MITRE ATT&CK framework, and evaluates organisational security policies for adequacy and compliance. Using a simulated brute-force attack scenario against a Windows OpenSSH service (MITRE ATT&CK T1110), the system leverages GPT-4o for reasoning, LangGraph for multi-agent workflow orchestration, and LlamaIndex for traceable policy retrieval. Experimental results indicate that the LLM-based pipeline can interpret log-derived evidence, identify insufficient or missing policy controls, and generate actionable remediation recommendations with explicit evidence-to-policy traceability. Unlike prior work that treats log analysis and policy validation as isolated tasks, this study integrates both into a unified end-to-end proof-of-concept post-incident review framework. The findings suggest that LLM-assisted analysis has the potential to improve the efficiency, consistency, and auditability of post-incident evaluations, while highlighting the continued need for human oversight in high-stakes cybersecurity decision-making. 

---
# HyperCLOVA X 32B Think 

**Authors**: NAVER Cloud HyperCLOVA X Team  

**Link**: [PDF](https://arxiv.org/pdf/2601.03286)  

**Abstract**: In this report, we present HyperCLOVA X 32B Think, a vision-language model designed with particular emphasis on reasoning within the Korean linguistic and cultural context, as well as agentic ability. HyperCLOVA X 32B Think is pre-trained with a strong focus on reasoning capabilities and subsequently post-trained to support multimodal understanding, enhanced reasoning, agentic behaviors, and alignment with human preferences. Experimental evaluations against comparably sized models demonstrate that our model achieves strong performance on Korean text-to-text and vision-to-text benchmarks, as well as on agent-oriented evaluation tasks. By open-sourcing HyperCLOVA X 32B Think, we aim to support broader adoption and facilitate further research and innovation across both academic and industrial communities. 

---
# Feedback Indices to Evaluate LLM Responses to Rebuttals for Multiple Choice Type Questions 

**Authors**: Justin C. Dunlap, Anne-Simone Parent, Ralf Widenhorn  

**Link**: [PDF](https://arxiv.org/pdf/2601.03285)  

**Abstract**: We present a systematic framework of indices designed to characterize Large Language Model (LLM) responses when challenged with rebuttals during a chat. Assessing how LLMs respond to user dissent is crucial for understanding their reliability and behavior patterns, yet the complexity of human-LLM interactions makes systematic evaluation challenging. Our approach employs a fictitious-response rebuttal method that quantifies LLM behavior when presented with multiple-choice questions followed by deliberate challenges to their fictitious previous response. The indices are specifically designed to detect and measure what could be characterized as sycophantic behavior (excessive agreement with user challenges) or stubborn responses (rigid adherence to the fictitious response in the chat history) from LLMs. These metrics allow investigation of the relationships between sycophancy, stubbornness, and the model's actual mastery of the subject matter. We demonstrate the utility of these indices using two physics problems as test scenarios with various OpenAI models. The framework is intentionally generalizable to any multiple-choice format question, including on topics without universally accepted correct answers. Our results reveal measurable differences across OpenAI model generations, with trends indicating that newer models and those employing greater "Reasoning Effort" exhibit reduced sycophantic behavior. The FR pairing method combined with our proposed indices provides a practical, adaptable toolkit for systematically comparing LLM dialogue behaviors across different models and contexts. 

---
# AI-Guided Discovery of Novel Ionic Liquid Solvents for Industrial CO2 Capture 

**Authors**: Davide Garbelotto, Alexander Lobo, Urvi Awasthi, Oleg Medvedev, Srayanta Mukherjee, Anton Aristov, Konstantin Polunin, Alex De Mur, Leonid Zhukov, Azad Huseynov, Murad Abdullayev  

**Link**: [PDF](https://arxiv.org/pdf/2601.03284)  

**Abstract**: We present an AI-driven approach to discover compounds with optimal properties for CO2 capture from flue gas-refinery emissions' primary source. Focusing on ionic liquids (ILs) as alternatives to traditional amine-based solvents, we successfully identify new IL candidates with high working capacity, manageable viscosity, favorable regeneration energy, and viable synthetic routes. Our approach follows a five-stage pipeline. First, we generate IL candidates by pairing available cation and anion molecules, then predict temperature- and pressure-dependent CO2 solubility and viscosity using a GNN-based molecular property prediction model. Next, we convert solubility to working capacity and regeneration energy via Van't Hoff modeling, and then find the best set of candidates using Pareto optimization, before finally filtering those based on feasible synthesis routes. We identify 36 feasible candidates that could enable 5-10% OPEX savings and up to 10% CAPEX reductions through lower regeneration energy requirements and reduced corrosivity-offering a novel carbon-capture strategy for refineries moving forward. 

---
# $α^3$-Bench: A Unified Benchmark of Safety, Robustness, and Efficiency for LLM-Based UAV Agents over 6G Networks 

**Authors**: Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2601.03281)  

**Abstract**: Large Language Models (LLMs) are increasingly used as high level controllers for autonomous Unmanned Aerial Vehicle (UAV) missions. However, existing evaluations rarely assess whether such agents remain safe, protocol compliant, and effective under realistic next generation networking constraints. This paper introduces $\alpha^3$-Bench, a benchmark for evaluating LLM driven UAV autonomy as a multi turn conversational reasoning and control problem operating under dynamic 6G conditions. Each mission is formulated as a language mediated control loop between an LLM based UAV agent and a human operator, where decisions must satisfy strict schema validity, mission policies, speaker alternation, and safety constraints while adapting to fluctuating network slices, latency, jitter, packet loss, throughput, and edge load variations.
To reflect modern agentic workflows, $\alpha^3$-Bench integrates a dual action layer supporting both tool calls and agent to agent coordination, enabling evaluation of tool use consistency and multi agent interactions. We construct a large scale corpus of 113k conversational UAV episodes grounded in UAVBench scenarios and evaluate 17 state of the art LLMs using a fixed subset of 50 episodes per scenario under deterministic decoding. We propose a composite $\alpha^3$ metric that unifies six pillars: Task Outcome, Safety Policy, Tool Consistency, Interaction Quality, Network Robustness, and Communication Cost, with efficiency normalized scores per second and per thousand tokens. Results show that while several models achieve high mission success and safety compliance, robustness and efficiency vary significantly under degraded 6G conditions, highlighting the need for network aware and resource efficient LLM based UAV agents. The dataset is publicly available on GitHub : this https URL 

---
# A Quantum Model for Constrained Markowitz Modern Portfolio Using Slack Variables to Process Mixed-Binary Optimization under QAOA 

**Authors**: Pablo Thomassin, Guillaume Guerard, Sonia Djebali, Vincent Marc Lambert  

**Link**: [PDF](https://arxiv.org/pdf/2601.03278)  

**Abstract**: Effectively encoding inequality constraints is a primary obstacle in applying quantum algorithms to financial optimization. A quantum model for Markowitz portfolio optimization is presented that resolves this by embedding slack variables directly into the problem Hamiltonian. The method maps each slack variable to a dedicated ancilla qubit, transforming the problem into a Quadratic Unconstrained Binary Optimization (QUBO) formulation suitable for the Quantum Approximate Optimization Algorithm (QAOA). This process internalizes the constraints within the quantum state, altering the problem's energy landscape to facilitate optimization. The model is empirically validated through simulation, showing it consistently finds the optimal portfolio where a standard penalty-based QAOA fails. This work demonstrates that modifying the Hamiltonian architecture via a slack-ancilla scheme provides a robust and effective pathway for solving constrained optimization problems on quantum computers. A fundamental quantum limit on the simultaneous precision of portfolio risk and return is also posited. 

---
# MixRx: Predicting Drug Combination Interactions with LLMs 

**Authors**: Risha Surana, Cameron Saidock, Hugo Chacon  

**Link**: [PDF](https://arxiv.org/pdf/2601.03277)  

**Abstract**: MixRx uses Large Language Models (LLMs) to classify drug combination interactions as Additive, Synergistic, or Antagonistic, given a multi-drug patient history. We evaluate the performance of 4 models, GPT-2, Mistral Instruct 2.0, and the fine-tuned counterparts. Our results showed a potential for such an application, with the Mistral Instruct 2.0 Fine-Tuned model providing an average accuracy score on standard and perturbed datasets of 81.5%. This paper aims to further develop an upcoming area of research that evaluates if LLMs can be used for biological prediction tasks. 

---
# Topic Segmentation Using Generative Language Models 

**Authors**: Pierre Mackenzie, Maya Shah, Patrick Frenett  

**Link**: [PDF](https://arxiv.org/pdf/2601.03276)  

**Abstract**: Topic segmentation using generative Large Language Models (LLMs) remains relatively unexplored. Previous methods use semantic similarity between sentences, but such models lack the long range dependencies and vast knowledge found in LLMs. In this work, we propose an overlapping and recursive prompting strategy using sentence enumeration. We also support the adoption of the boundary similarity evaluation metric. Results show that LLMs can be more effective segmenters than existing methods, but issues remain to be solved before they can be relied upon for topic segmentation. 

---
# LLM_annotate: A Python package for annotating and analyzing fiction characters 

**Authors**: Hannes Rosenbusch  

**Link**: [PDF](https://arxiv.org/pdf/2601.03274)  

**Abstract**: LLM_annotate is a Python package for analyzing the personality of fiction characters with large language models. It standardizes workflows for annotating character behaviors in full texts (e.g., books and movie scripts), inferring character traits, and validating annotation/inference quality via a human-in-the-loop GUI. The package includes functions for text chunking, LLM-based annotation, character name disambiguation, quality scoring, and computation of character-level statistics and embeddings. Researchers can use any LLM, commercial, open-source, or custom, within LLM_annotate. Through tutorial examples using The Simpsons Movie and the novel Pride and Prejudice, I demonstrate the usage of the package for efficient and reproducible character analyses. 

---
# GuardEval: A Multi-Perspective Benchmark for Evaluating Safety, Fairness, and Robustness in LLM Moderators 

**Authors**: Naseem Machlovi, Maryam Saleki, Ruhul Amin, Mohamed Rahouti, Shawqi Al-Maliki, Junaid Qadir, Mohamed M. Abdallah, Ala Al-Fuqaha  

**Link**: [PDF](https://arxiv.org/pdf/2601.03273)  

**Abstract**: As large language models (LLMs) become deeply embedded in daily life, the urgent need for safer moderation systems, distinguishing between naive from harmful requests while upholding appropriate censorship boundaries, has never been greater. While existing LLMs can detect harmful or unsafe content, they often struggle with nuanced cases such as implicit offensiveness, subtle gender and racial biases, and jailbreak prompts, due to the subjective and context-dependent nature of these issues. Furthermore, their heavy reliance on training data can reinforce societal biases, resulting in inconsistent and ethically problematic outputs. To address these challenges, we introduce GuardEval, a unified multi-perspective benchmark dataset designed for both training and evaluation, containing 106 fine-grained categories spanning human emotions, offensive and hateful language, gender and racial bias, and broader safety concerns. We also present GemmaGuard (GGuard), a QLoRA fine-tuned version of Gemma3-12B trained on GuardEval, to assess content moderation with fine-grained labels. Our evaluation shows that GGuard achieves a macro F1 score of 0.832, substantially outperforming leading moderation models, including OpenAI Moderator (0.64) and Llama Guard (0.61). We show that multi-perspective, human-centered safety benchmarks are critical for reducing biased and inconsistent moderation decisions. GuardEval and GGuard together demonstrate that diverse, representative data materially improve safety, fairness, and robustness on complex, borderline cases. 

---
# Less is more: Not all samples are effective for evaluation 

**Authors**: Wentang Song, Jinqiang Li, Kele Huang, Junhui Lin, Shengxiang Wu, Zhongshi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.03272)  

**Abstract**: The versatility of Large Language Models (LLMs) in vertical domains has spurred the development of numerous specialized evaluation benchmarks. However, these benchmarks often suffer from significant semantic redundancy and impose high computational costs during evaluation. Existing compression methods, such as tinyBenchmarks depend critically on correctness labels from multiple historical models evaluated on the full test set, making them inapplicable in cold-start scenarios, such as the introduction of a new task, domain, or model with no prior evaluation history.
To address this limitation, we propose a history-free test set compression framework that requires no prior model performance data. Our method begins by fine-tuning a base LLM on a small amount of domain-specific data to internalize task-relevant semantics. It then generates high-level semantic embeddings for all original test samples using only their raw textual content. In this domain-adapted embedding space, we perform task-aware clustering and introduce a novel dataset X-ray mechanism that analyzes cluster geometry to dynamically calibrate the compression intensity based on the intrinsic redundancy of the benchmark.
Experiments on professional-domain dataset, notably a large-scale 3GPP communications benchmark, demonstrate that our approach effectively identifies and removes redundant samples, reducing evaluation cost by over 90% while preserving high fidelity to the full benchmark. 

---
# Advances and Challenges in Semantic Textual Similarity: A Comprehensive Survey 

**Authors**: Lokendra Kumar, Neelesh S. Upadhye, Kannan Piedy  

**Link**: [PDF](https://arxiv.org/pdf/2601.03270)  

**Abstract**: Semantic Textual Similarity (STS) research has expanded rapidly since 2021, driven by advances in transformer architectures, contrastive learning, and domain-specific techniques. This survey reviews progress across six key areas: transformer-based models, contrastive learning, domain-focused solutions, multi-modal methods, graph-based approaches, and knowledge-enhanced techniques. Recent transformer models such as FarSSiBERT and DeBERTa-v3 have achieved remarkable accuracy, while contrastive methods like AspectCSE have established new benchmarks. Domain-adapted models, including CXR-BERT for medical texts and Financial-STS for finance, demonstrate how STS can be effectively customized for specialized fields. Moreover, multi-modal, graph-based, and knowledge-integrated models further enhance semantic understanding and representation. By organizing and analyzing these developments, the survey provides valuable insights into current methods, practical applications, and remaining challenges. It aims to guide researchers and practitioners alike in navigating rapid advancements, highlighting emerging trends and future opportunities in the evolving field of STS. 

---
# The Instruction Gap: LLMs get lost in Following Instruction 

**Authors**: Vishesh Tripathi, Uday Allu, Biddwan Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2601.03269)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in natural language understanding and generation, yet their deployment in enterprise environments reveals a critical limitation: inconsistent adherence to custom instructions. This study presents a comprehensive evaluation of 13 leading LLMs across instruction compliance, response accuracy, and performance metrics in realworld RAG (Retrieval-Augmented Generation) scenarios. Through systematic testing with samples and enterprise-grade evaluation protocols, we demonstrate that instruction following varies dramatically across models, with Claude-Sonnet-4 and GPT-5 achieving the highest results. Our findings reveal the "instruction gap" - a fundamental challenge where models excel at general tasks but struggle with precise instruction adherence required for enterprise deployment. This work provides practical insights for organizations deploying LLM-powered solutions and establishes benchmarks for instruction-following capabilities across major model families. 

---
# OpenAI GPT-5 System Card 

**Authors**: Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, Akshay Nathan, Alan Luo, Alec Helyar, Aleksander Madry, Aleksandr Efremov, Aleksandra Spyra, Alex Baker-Whitcomb, Alex Beutel, Alex Karpenko, Alex Makelov, Alex Neitz, Alex Wei, Alexandra Barr, Alexandre Kirchmeyer, Alexey Ivanov, Alexi Christakis, Alistair Gillespie, Allison Tam, Ally Bennett, Alvin Wan, Alyssa Huang, Amy McDonald Sandjideh, Amy Yang, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrei Gheorghe, Andres Garcia Garcia, Andrew Braunstein, Andrew Liu, Andrew Schmidt, Andrey Mereskin, Andrey Mishchenko, Andy Applebaum, Andy Rogerson, Ann Rajan, Annie Wei, Anoop Kotha, Anubha Srivastava, Anushree Agrawal, Arun Vijayvergiya, Ashley Tyra, Ashvin Nair, Avi Nayak, Ben Eggers, Bessie Ji, Beth Hoover, Bill Chen, Blair Chen, Boaz Barak, Borys Minaiev, Botao Hao, Bowen Baker, Brad Lightcap, Brandon McKinzie, Brandon Wang, Brendan Quinn, Brian Fioca, Brian Hsu, Brian Yang, Brian Yu, Brian Zhang, Brittany Brenner, Callie Riggins Zetino, Cameron Raymond, Camillo Lugaresi, Carolina Paz, Cary Hudson, Cedric Whitney, Chak Li, Charles Chen, Charlotte Cole, Chelsea Voss, Chen Ding, Chen Shen, Chengdu Huang, Chris Colby, Chris Hallacy, Chris Koch, Chris Lu, Christina Kaplan, Christina Kim, CJ Minott-Henriques, Cliff Frey, Cody Yu, Coley Czarnecki, Colin Reid, Colin Wei, Cory Decareaux, Cristina Scheau  

**Link**: [PDF](https://arxiv.org/pdf/2601.03267)  

**Abstract**: This is the system card published alongside the OpenAI GPT-5 launch, August 2025.
GPT-5 is a unified system with a smart and fast model that answers most questions, a deeper reasoning model for harder problems, and a real-time router that quickly decides which model to use based on conversation type, complexity, tool needs, and explicit intent (for example, if you say 'think hard about this' in the prompt). The router is continuously trained on real signals, including when users switch models, preference rates for responses, and measured correctness, improving over time. Once usage limits are reached, a mini version of each model handles remaining queries.
This system card focuses primarily on gpt-5-thinking and gpt-5-main, while evaluations for other models are available in the appendix. The GPT-5 system not only outperforms previous models on benchmarks and answers questions more quickly, but -- more importantly -- is more useful for real-world queries. We've made significant advances in reducing hallucinations, improving instruction following, and minimizing sycophancy, and have leveled up GPT-5's performance in three of ChatGPT's most common uses: writing, coding, and health. All of the GPT-5 models additionally feature safe-completions, our latest approach to safety training to prevent disallowed content.
Similarly to ChatGPT agent, we have decided to treat gpt-5-thinking as High capability in the Biological and Chemical domain under our Preparedness Framework, activating the associated safeguards. While we do not have definitive evidence that this model could meaningfully help a novice to create severe biological harm -- our defined threshold for High capability -- we have chosen to take a precautionary approach. 

---
# Benchmarking and Adapting On-Device Large Language Models for Clinical Decision Support 

**Authors**: Alif Munim, Jun Ma, Omar Ibrahim, Alhusain Abdalla, Shuolin Yin, Leo Chen, Bo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03266)  

**Abstract**: Large language models (LLMs) have rapidly advanced in clinical decision-making, yet the deployment of proprietary systems is hindered by privacy concerns and reliance on cloud-based infrastructure. Open-source alternatives allow local inference but often require large model sizes that limit their use in resource-constrained clinical settings. Here, we benchmark two on-device LLMs, gpt-oss-20b and gpt-oss-120b, across three representative clinical tasks: general disease diagnosis, specialty-specific (ophthalmology) diagnosis and management, and simulation of human expert grading and evaluation. We compare their performance with state-of-the-art proprietary models (GPT-5 and o4-mini) and a leading open-source model (DeepSeek-R1), and we further evaluate the adaptability of on-device systems by fine-tuning gpt-oss-20b on general diagnostic data. Across tasks, gpt-oss models achieve performance comparable to or exceeding DeepSeek-R1 and o4-mini despite being substantially smaller. In addition, fine-tuning remarkably improves the diagnostic accuracy of gpt-oss-20b, enabling it to approach the performance of GPT-5. These findings highlight the potential of on-device LLMs to deliver accurate, adaptable, and privacy-preserving clinical decision support, offering a practical pathway for broader integration of LLMs into routine clinical practice. 

---
# Internal Reasoning vs. External Control: A Thermodynamic Analysis of Sycophancy in Large Language Models 

**Authors**: Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03263)  

**Abstract**: Large Language Models frequently exhibit sycophancy, prioritizing user agreeableness over correctness. We investigate whether this requires external regulation or can be mitigated by internal reasoning alone. Using CAP-GSM8K (N=500), an adversarial dataset, we evaluate internal (CoT) versus external (RCA) mechanisms across GPT-3.5, GPT-4o, and GPT-5.1. Our results reveal the structural limits of internal reasoning: it causes performance collapse in weak models (the Prioritization Paradox) and leaves an 11.4\% final output gap in frontier models. In contrast, RCA structurally eliminates sycophancy (0.0\%) across all tiers. We synthesize these findings into a thermodynamic hierarchy: hybrid systems achieve Resonance (optimal efficiency) only when capabilities are matched and strong, while weak or mismatched pairs succumb to Dissonance and Entropy. This confirms that external structural constraints are strictly necessary to guarantee safety. 

---
# DeepResearch-Slice: Bridging the Retrieval-Utilization Gap via Explicit Text Slicing 

**Authors**: Shuo Lu, Yinuo Xu, Jianjie Cheng, Lingxiao He, Meng Wang, Jian Liang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03261)  

**Abstract**: Deep Research agents predominantly optimize search policies to maximize retrieval probability. However, we identify a critical bottleneck: the retrieval-utilization gap, where models fail to use gold evidence even after it is retrieved, due to context blindness in noisy environments. To bridge this gap, we propose DeepResearch-Slice, a simple yet effective neuro-symbolic framework. Unlike implicit attention, our approach predicts precise span indices to perform a deterministic hard filter before reasoning. Extensive evaluations across six benchmarks show substantial robustness gains. Applying our method to frozen backbones yields a 73 percent relative improvement, from 19.1 percent to 33.0 percent, effectively mitigating noise without requiring parameter updates to the reasoning model. These results highlight the need for explicit grounding mechanisms in open-ended research. 

---
