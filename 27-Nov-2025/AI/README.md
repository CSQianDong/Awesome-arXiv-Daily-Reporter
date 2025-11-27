# Agentic Learner with Grow-and-Refine Multimodal Semantic Memory 

**Authors**: Weihao Bo, Shan Zhang, Yanpeng Sun, Jingjing Wu, Qunyi Xie, Xiao Tan, Kunbin Chen, Wei He, Xiaofan Li, Na Zhao, Jingdong Wang, Zechao Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.21678)  

**Abstract**: MLLMs exhibit strong reasoning on isolated queries, yet they operate de novo -- solving each problem independently and often repeating the same mistakes. Existing memory-augmented agents mainly store past trajectories for reuse. However, trajectory-based memory suffers from brevity bias, gradually losing essential domain knowledge. More critically, even in truly multimodal problem-solving settings, it records only a single-modality trace of past behavior, failing to preserve how visual attention and logical reasoning jointly contributed to the solution. This is fundamentally misaligned with human cognition: semantic memory is both multimodal and integrated, preserving visual and abstract knowledge through coordinated but distinct representational streams. We thus introduce ViLoMem, a dual-stream memory framework that constructs compact, schema-based memory. It separately encodes visual distraction patterns and logical reasoning errors, enabling MLLMs to learn from their successful and failed experiences. Following a grow-and-refine principle, the system incrementally accumulates and updates multimodal semantic knowledge -- preserving stable, generalizable strategies while avoiding catastrophic forgetting. Across six multimodal benchmarks, ViLoMem consistently improves pass@1 accuracy and substantially reduces repeated visual and logical errors. Ablations confirm the necessity of dual-stream memory with explicit distraction--hallucination separation, demonstrating the value of error-aware multimodal memory for lifelong and cross-domain agentic learning. Our project page will be available at this https URL. 

---
# Bridging the Unavoidable A Priori: A Framework for Comparative Causal Modeling 

**Authors**: Peter S. Hovmand, Kari O'Donnell, Callie Ogland-Hand, Brian Biroscak, Douglas D. Gunzler  

**Link**: [PDF](https://arxiv.org/pdf/2511.21636)  

**Abstract**: AI/ML models have rapidly gained prominence as innovations for solving previously unsolved problems and their unintended consequences from amplifying human biases. Advocates for responsible AI/ML have sought ways to draw on the richer causal models of system dynamics to better inform the development of responsible AI/ML. However, a major barrier to advancing this work is the difficulty of bringing together methods rooted in different underlying assumptions (i.e., Dana Meadow's "the unavoidable a priori"). This paper brings system dynamics and structural equation modeling together into a common mathematical framework that can be used to generate systems from distributions, develop methods, and compare results to inform the underlying epistemology of system dynamics for data science and AI/ML applications. 

---
# On the Limits of Innate Planning in Large Language Models 

**Authors**: Charles Schepanowski, Charles Ling  

**Link**: [PDF](https://arxiv.org/pdf/2511.21591)  

**Abstract**: Large language models (LLMs) achieve impressive results on many benchmarks, yet their capacity for planning and stateful reasoning remains unclear. We study these abilities directly, without code execution or other tools, using the 8-puzzle: a classic task that requires state tracking and goal-directed planning while allowing precise, step-by-step evaluation. Four models are tested under common prompting conditions (Zero-Shot, Chain-of-Thought, Algorithm-of-Thought) and with tiered corrective feedback. Feedback improves success rates for some model-prompt combinations, but many successful runs are long, computationally expensive, and indirect. We then examine the models with an external move validator that provides only valid moves. Despite this level of assistance, none of the models solve any puzzles in this setting. Qualitative analysis reveals two dominant deficits across all models: (1) brittle internal state representations, leading to frequent invalid moves, and (2) weak heuristic planning, with models entering loops or selecting actions that do not reduce the distance to the goal state. These findings indicate that, in the absence of external tools such as code interpreters, current LLMs have substantial limitations in planning and that further progress may require mechanisms for maintaining explicit state and performing structured search. 

---
# From Prediction to Foresight: The Role of AI in Designing Responsible Futures 

**Authors**: Maria Perez-Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2511.21570)  

**Abstract**: In an era marked by rapid technological advancements and complex global challenges, responsible foresight has emerged as an essential framework for policymakers aiming to navigate future uncertainties and shape the future. Responsible foresight entails the ethical anticipation of emerging opportunities and risks, with a focus on fostering proactive, sustainable, and accountable future design. This paper coins the term "responsible computational foresight", examining the role of human-centric artificial intelligence and computational modeling in advancing responsible foresight, establishing a set of foundational principles for this new field and presenting a suite of AI-driven foresight tools currently shaping it. AI, particularly in conjunction with simulations and scenario analysis, enhances policymakers' ability to address uncertainty, evaluate risks, and devise strategies geared toward sustainable, resilient futures. However, responsible foresight extends beyond mere technical forecasting; it demands a nuanced understanding of the interdependencies within social, environmental, economic and political systems, alongside a commitment to ethical, long-term decision-making that supports human intelligence. We argue that AI will play a role as a supportive tool in responsible, human-centered foresight, complementing rather than substituting policymaker judgment to enable the proactive shaping of resilient and ethically sound futures. This paper advocates for the thoughtful integration of AI into foresight practices to empower policymakers and communities as they confront the grand challenges of the 21st century. 

---
# Self-Transparency Failures in Expert-Persona LLMs: A Large-Scale Behavioral Audit 

**Authors**: Alex Diep  

**Link**: [PDF](https://arxiv.org/pdf/2511.21569)  

**Abstract**: If a language model cannot reliably disclose its AI identity in expert contexts, users cannot trust its competence boundaries. This study examines self-transparency in models assigned professional personas within high-stakes domains where false expertise risks user harm. Using a common-garden design, sixteen open-weight models (4B--671B parameters) were audited across 19,200 trials. Models exhibited sharp domain-specific inconsistency: a Financial Advisor persona elicited 30.8% disclosure initially, while a Neurosurgeon persona elicited only 3.5%. This creates preconditions for a "Reverse Gell-Mann Amnesia" effect, where transparency in some domains leads users to overgeneralize trust to contexts where disclosure fails. Disclosure ranged from 2.8% to 73.6%, with a 14B model reaching 61.4% while a 70B produced just 4.1%. Model identity predicted behavior better than parameter count ($\Delta R_{adj}^{2} = 0.359$ vs 0.018). Reasoning optimization actively suppressed self-transparency in some models, with reasoning variants showing up to 48.4% lower disclosure than base counterparts. Bayesian validation with Rogan--Gladen correction confirmed robustness to measurement error ($\kappa = 0.908$). These findings demonstrate transparency reflects training factors rather than scale. Organizations cannot assume safety properties transfer to deployment contexts, requiring deliberate behavior design and empirical verification. 

---
# Pessimistic Verification for Open Ended Math Questions 

**Authors**: Yanxing Huang, Zihan Tang, Zejin Lin, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.21522)  

**Abstract**: The key limitation of the verification performance lies in the ability of error detection. With this intuition we designed several variants of pessimistic verification, which are simple workflows that could significantly improve the verification of open-ended math questions. In pessimistic verification we construct multiple parallel verifications for the same proof, and the proof is deemed incorrect if any one of them reports an error. This simple technique significantly improves the performance across many math verification benchmarks without incurring substantial computational resources. Its token efficiency even surpassed extended long-CoT in test-time scaling. Our case studies further indicate that the majority of false negatives in stronger models are actually caused by annotation errors in the original dataset, so our method's performance is in fact underestimated. Self-verification for mathematical problems can effectively improve the reliability and performance of language model outputs, and it also plays a critical role in enabling long-horizon mathematical tasks. We believe that research on pessimistic verification will help enhance the mathematical capabilities of language models across a wide range of tasks. 

---
# SpatialBench: Benchmarking Multimodal Large Language Models for Spatial Cognition 

**Authors**: Peiran Xu, Sudong Wang, Yao Zhu, Jianing Li, Yunjian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21471)  

**Abstract**: Spatial cognition is fundamental to real-world multimodal intelligence, allowing models to effectively interact with the physical environment. While multimodal large language models (MLLMs) have made significant strides, existing benchmarks often oversimplify spatial cognition, reducing it to a single-dimensional metric, which fails to capture the hierarchical structure and interdependence of spatial abilities. To address this gap, we propose a hierarchical spatial cognition framework that decomposes spatial intelligence into five progressively complex levels from basic observation to high-level planning. Building upon this taxonomy, we construct SpatialBench, a large-scale, fine-grained benchmark covering 15 tasks aligned with these cognitive levels. To provide a unified evaluation across heterogeneous tasks, we further introduce a high-level capability-oriented metric that reliably assesses a model's overall spatial reasoning ability. Extensive experiments over massive MLLMs reveal distinct performance stratification across cognitive levels: models exhibit strong perceptual grounding yet remain limited in symbolic reasoning, causal inference, and planning. Additional human tests demonstrate that humans perform selective, goal-directed abstraction, while MLLMs tend to over-attend to surface details without coherent spatial intent. Our work establishes the first systematic framework for measuring hierarchical spatial cognition in MLLMs, laying the foundation for future spatially intelligent systems. 

---
# MADRA: Multi-Agent Debate for Risk-Aware Embodied Planning 

**Authors**: Junjian Wang, Lidan Zhao, Xi Sheryl Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21460)  

**Abstract**: Ensuring the safety of embodied AI agents during task planning is critical for real-world deployment, especially in household environments where dangerous instructions pose significant risks. Existing methods often suffer from either high computational costs due to preference alignment training or over-rejection when using single-agent safety prompts. To address these limitations, we propose MADRA, a training-free Multi-Agent Debate Risk Assessment framework that leverages collective reasoning to enhance safety awareness without sacrificing task performance. MADRA employs multiple LLM-based agents to debate the safety of a given instruction, guided by a critical evaluator that scores responses based on logical soundness, risk identification, evidence quality, and clarity. Through iterative deliberation and consensus voting, MADRA significantly reduces false rejections while maintaining high sensitivity to dangerous tasks. Additionally, we introduce a hierarchical cognitive collaborative planning framework that integrates safety, memory, planning, and self-evolution mechanisms to improve task success rates through continuous learning. We also contribute SafeAware-VH, a benchmark dataset for safety-aware task planning in VirtualHome, containing 800 annotated instructions. Extensive experiments on AI2-THOR and VirtualHome demonstrate that our approach achieves over 90% rejection of unsafe tasks while ensuring that safe-task rejection is low, outperforming existing methods in both safety and execution efficiency. Our work provides a scalable, model-agnostic solution for building trustworthy embodied agents. 

---
# EWE: An Agentic Framework for Extreme Weather Analysis 

**Authors**: Zhe Jiang, Jiong Wang, Xiaoyu Yue, Zijie Guo, Wenlong Zhang, Fenghua Ling, Wanli Ouyang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2511.21444)  

**Abstract**: Extreme weather events pose escalating risks to global society, underscoring the urgent need to unravel their underlying physical mechanisms. Yet the prevailing expert-driven, labor-intensive diagnostic paradigm has created a critical analytical bottleneck, stalling scientific progress. While AI for Earth Science has achieved notable advances in prediction, the equally essential challenge of automated diagnostic reasoning remains largely unexplored. We present the Extreme Weather Expert (EWE), the first intelligent agent framework dedicated to this task. EWE emulates expert workflows through knowledge-guided planning, closed-loop reasoning, and a domain-tailored meteorological toolkit. It autonomously produces and interprets multimodal visualizations from raw meteorological data, enabling comprehensive diagnostic analyses. To catalyze progress, we introduce the first benchmark for this emerging field, comprising a curated dataset of 103 high-impact events and a novel step-wise evaluation metric. EWE marks a step toward automated scientific discovery and offers the potential to democratize expertise and intellectual resources, particularly for developing nations vulnerable to extreme weather. 

---
# Conversational no-code and multi-agentic disease module identification and drug repurposing prediction with ChatDRex 

**Authors**: Simon Süwer, Kester Bagemihl, Sylvie Baier, Lucia Dicunta, Markus List, Jan Baumbach, Andreas Maier, Fernando M. Delgado-Chaves  

**Link**: [PDF](https://arxiv.org/pdf/2511.21438)  

**Abstract**: Repurposing approved drugs offers a time-efficient and cost-effective alternative to traditional drug development. However, in silico prediction of repurposing candidates is challenging and requires the effective collaboration of specialists in various fields, including pharmacology, medicine, biology, and bioinformatics. Fragmented, specialized algorithms and tools often address only narrow aspects of the overall problem, and heterogeneous, unstructured data landscapes require specialized users to be involved. Hence, these data services do not integrate smoothly across workflows. With ChatDRex, we present a conversation-based, multi-agent system that facilitates the execution of complex bioinformatic analyses aiming for network-based drug repurposing prediction. It builds on the integrated systems medicine knowledge graph NeDRex. ChatDRex provides natural language access to its extensive biomedical KG and integrates bioinformatics agents for network analysis and drug repurposing, complemented by agents for functional coherence evaluation for in silico validation, as well as agents for literature mining and for discussing the obtained results in a scientific context. Its flexible multi-agent design assigns specific tasks to specialized agents, including query routing, data retrieval, algorithm execution, and result visualization. A dedicated reasoning module keeps the user in the loop and allows for hallucination detection. By enabling physicians and researchers without computer science expertise to control complex analyses in natural language, ChatDRex democratizes access to bioinformatics as an important resource for drug repurposing. It enables clinical experts to generate hypotheses and explore drug repurposing opportunities, ultimately accelerating the discovery of novel therapies and advancing personalized medicine and translational research. 

---
# New Hybrid Heuristics for Pseudo-Boolean Propagation 

**Authors**: Mia Müßig, Jan Johannsen  

**Link**: [PDF](https://arxiv.org/pdf/2511.21417)  

**Abstract**: In pseudo-boolean solving the currently most successful unit propagation strategy is a hybrid mode combining the watched literal scheme with the counting method. This short paper introduces new heuristics for this hybrid decision, which are able to drastically outperform the current method in the RoundingSAT solver. 

---
# Prune4Web: DOM Tree Pruning Programming for Web Agent 

**Authors**: Jiayuan Zhang, Kaiquan Chen, Zhihao Lu, Enshen Zhou, Qian Yu, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21398)  

**Abstract**: Web automation employs intelligent agents to execute high-level tasks by mimicking human interactions with web interfaces. Despite the capabilities of recent Large Language Model (LLM)-based web agents, navigating complex, real-world webpages efficiently remains a significant hurdle due to the prohibitively large size of Document Object Model (DOM) structures, often ranging from 10,000 to 100,000 tokens. Existing strategies typically rely on crude DOM truncation -- risking the loss of critical information -- or employ inefficient heuristics and separate ranking models, failing to achieve an optimal balance between precision and scalability. To address these challenges, we introduce Prune4Web, a novel paradigm that shifts DOM processing from resource-intensive LLM reading to efficient programmatic pruning. Central to our approach is DOM Tree Pruning Programming, where an LLM generates executable Python scoring scripts to dynamically filter DOM elements based on semantic cues from decomposed sub-tasks. This mechanism eliminates the need for LLMs to ingest raw, massive DOMs, instead delegating traversal and scoring to lightweight, interpretable programs. This methodology achieves a 25x to 50x reduction in candidate elements for grounding, thereby facilitating precise action localization while mitigating attention dilution. Furthermore, we propose a specialized data annotation pipeline and a two-turn dialogue training strategy that jointly optimizes the Planner, Programmatic Filter, and Grounder within a unified framework. Extensive experiments demonstrate state-of-the-art performance. Notably, on our low-level grounding task, Prune4Web dramatically improves accuracy from 46.8% to 88.28%, underscoring its efficacy in real-world web automation. 

---
# Causality Without Causal Models 

**Authors**: Joseph Y. Halpern, Rafael Pass  

**Link**: [PDF](https://arxiv.org/pdf/2511.21260)  

**Abstract**: Perhaps the most prominent current definition of (actual) causality is due to Halpern and Pearl.  It is defined using causal models (also known as structural equations models).  We abstract the definition, extracting its key features, so that it can be applied to any other model where counterfactuals are defined. By abstracting the definition, we gain a number of benefits. Not only can we apply the definition in a wider range of models, including ones that allow, for example, backtracking, but we can apply the definition to determine if A is a cause of B  even if A and B are formulas involving disjunctions, negations, beliefs, and nested counterfactuals (none of which can be handled by the Halpern-Pearl definition). Moreover, we can extend the ideas to getting an abstract definition of explanation that can be applied beyond causal models. Finally, we gain a deeper understanding of features of the definition  even in causal models. 

---
# OVOD-Agent: A Markov-Bandit Framework for Proactive Visual Reasoning and Self-Evolving Detection 

**Authors**: Chujie Wang, Jianyu Lu, Zhiyuan Luo, Xi Chen, Chu He  

**Link**: [PDF](https://arxiv.org/pdf/2511.21064)  

**Abstract**: Open-Vocabulary Object Detection (OVOD) aims to enable detectors to generalize across categories by leveraging semantic information. Although existing methods are pretrained on large vision-language datasets, their inference is still limited to fixed category names, creating a gap between multimodal training and unimodal inference. Previous work has shown that improving textual representation can significantly enhance OVOD performance, indicating that the textual space is still underexplored. To this end, we propose OVOD-Agent, which transforms passive category matching into proactive visual reasoning and self-evolving detection. Inspired by the Chain-of-Thought (CoT) paradigm, OVOD-Agent extends the textual optimization process into an interpretable Visual-CoT with explicit actions. OVOD's lightweight nature makes LLM-based management unsuitable; instead, we model visual context transitions as a Weakly Markovian Decision Process (w-MDP) over eight state spaces, which naturally represents the agent's state, memory, and interaction dynamics. A Bandit module generates exploration signals under limited supervision, helping the agent focus on uncertain regions and adapt its detection policy. We further integrate Markov transition matrices with Bandit trajectories for self-supervised Reward Model (RM) optimization, forming a closed loop from Bandit exploration to RM learning. Experiments on COCO and LVIS show that OVOD-Agent provides consistent improvements across OVOD backbones, particularly on rare categories, confirming the effectiveness of the proposed framework. 

---
# Towards Trustworthy Legal AI through LLM Agents and Formal Reasoning 

**Authors**: Linze Chen, Yufan Cai, Zhe Hou, Jinsong Dong  

**Link**: [PDF](https://arxiv.org/pdf/2511.21033)  

**Abstract**: The rationality of law manifests in two forms: substantive rationality, which concerns the fairness or moral desirability of outcomes, and formal rationality, which requires legal decisions to follow explicitly stated, general, and logically coherent rules. Existing LLM-based systems excel at surface-level text analysis but lack the guarantees required for principled jurisprudence. We introduce L4M, a novel framework that combines adversarial LLM agents with SMT-solver-backed proofs to unite the interpretive flexibility of natural language with the rigor of symbolic verification. The pipeline consists of three phases: (1) Statute Formalization, where domain-specific prompts convert legal provisions into logical formulae; (2) Dual Fact and Statute Extraction, in which prosecutor- and defense-aligned LLMs independently map case narratives to fact tuples and statutes, ensuring role isolation; and (3) Solver-Centric Adjudication, where an autoformalizer compiles both parties' arguments into logic constraints, and unsat cores trigger iterative self-critique until a satisfiable formula is achieved, which is then verbalized by a Judge-LLM into a transparent verdict and optimized sentence. Experimental results on public benchmarks show that our system surpasses advanced LLMs including GPT-o4-mini, DeepSeek-V3, and Claude 4 as well as state-of-the-art Legal AI baselines, while providing rigorous and explainable symbolic justifications. 

---
# ICPO: Intrinsic Confidence-Driven Group Relative Preference Optimization for Efficient Reinforcement Learning 

**Authors**: Jinpeng Wang, Chao Li, Ting Ye, Mengyuan Zhang, Wei Liu, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2511.21005)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) demonstrates significant potential in enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing RLVR methods are often constrained by issues such as coarse-grained rewards, reward noise, and inefficient exploration, which lead to unstable training and entropy collapse. To address this challenge, we propose the Intrinsic Confidence-Driven Group Relative Preference Optimization method (ICPO). The intuition behind it lies in the fact that the probabilities of an LLM generating different responses can inherently and directly reflect its self-assessment of the reasoning process. Inspired by the idea of preference modeling, ICPO calculates a preference advantage score for each response by comparing the relative generation probabilities of multiple responses under the same input prompt, and integrates this score with verifiable rewards to guide the exploration process. We have discovered that the preference advantage score not only alleviates the issues of coarse-grained rewards and reward noise but also effectively curbs overconfident errors, enhances the relative superiority of undervalued high-quality responses, and prevents the model from overfitting to specific strategies, thereby facilitating more thorough exploration. Comprehensive experiments across four general-domain benchmarks and three mathematical benchmarks demonstrate that ICPO steadily boosts reasoning compared to GRPO. 

---
# Improving Procedural Skill Explanations via Constrained Generation: A Symbolic-LLM Hybrid Architecture 

**Authors**: Rahul Dass, Thomas Bowlin, Zebing Li, Xiao Jin, Ashok Goel  

**Link**: [PDF](https://arxiv.org/pdf/2511.20942)  

**Abstract**: In procedural skill learning, instructional explanations must convey not just steps, but the causal, goal-directed, and compositional logic behind them. Large language models (LLMs) often produce fluent yet shallow responses that miss this structure. We present Ivy, an AI coaching system that delivers structured, multi-step explanations by combining symbolic Task-Method-Knowledge (TMK) models with a generative interpretation layer-an LLM that constructs explanations while being constrained by TMK structure. TMK encodes causal transitions, goal hierarchies, and problem decompositions, and guides the LLM within explicit structural bounds. We evaluate Ivy against responses against GPT and retrieval-augmented GPT baselines using expert and independent annotations across three inferential dimensions. Results show that symbolic constraints consistently improve the structural quality of explanations for "how" and "why" questions. This study demonstrates a scalable AI for education approach that strengthens the pedagogical value of AI-generated explanations in intelligent coaching systems. 

---
# ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction 

**Authors**: Qineng Wang, Wenlong Huang, Yu Zhou, Hang Yin, Tianwei Bao, Jianwen Lyu, Weiyu Liu, Ruohan Zhang, Jiajun Wu, Li Fei-Fei, Manling Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.20937)  

**Abstract**: Embodied cognition argues that intelligence arises from sensorimotor interaction rather than passive observation. It raises an intriguing question: do modern vision-language models (VLMs), trained largely in a disembodied manner, exhibit signs of embodied cognition? We introduce ENACT, a benchmark that casts evaluation of embodied cognition as world modeling from egocentric interaction in a visual question answering (VQA) format. Framed as a partially observable Markov decision process (POMDP) whose actions are scene graph changes, ENACT comprises two complementary sequence reordering tasks: forward world modeling (reorder shuffled observations given actions) and inverse world modeling (reorder shuffled actions given observations). While conceptually simple, solving these tasks implicitly demands capabilities central to embodied cognition-affordance recognition, action-effect reasoning, embodied awareness, and interactive, long-horizon memory from partially observable egocentric input, while avoiding low-level image synthesis that could confound the evaluation. We provide a scalable pipeline that synthesizes QA pairs from robotics simulation (BEHAVIOR) and evaluates models on 8,972 QA pairs spanning long-horizon home-scale activities. Experiments reveal a performance gap between frontier VLMs and humans that widens with interaction horizon. Models consistently perform better on the inverse task than the forward one and exhibit anthropocentric biases, including a preference for right-handed actions and degradation when camera intrinsics or viewpoints deviate from human vision. Website at this https URL. 

---
# Guaranteed Optimal Compositional Explanations for Neurons 

**Authors**: Biagio La Rosa, Leilani H. Gilpin  

**Link**: [PDF](https://arxiv.org/pdf/2511.20934)  

**Abstract**: While neurons are the basic units of deep neural networks, it is still unclear what they learn and if their knowledge is aligned with that of humans. Compositional explanations aim to answer this question by describing the spatial alignment between neuron activations and concepts through logical rules. These logical descriptions are typically computed via a search over all possible concept combinations. Since computing the spatial alignment over the entire state space is computationally infeasible, the literature commonly adopts beam search to restrict the space. However, beam search cannot provide any theoretical guarantees of optimality, and it remains unclear how close current explanations are to the true optimum. In this theoretical paper, we address this gap by introducing the first framework for computing guaranteed optimal compositional explanations. Specifically, we propose: (i) a decomposition that identifies the factors influencing the spatial alignment, (ii) a heuristic to estimate the alignment at any stage of the search, and (iii) the first algorithm that can compute optimal compositional explanations within a feasible time. Using this framework, we analyze the differences between optimal and non-optimal explanations in the most popular settings for compositional explanations, the computer vision domain and Convolutional Neural Networks. In these settings, we demonstrate that 10-40 percent of explanations obtained with beam search are suboptimal when overlapping concepts are involved. Finally, we evaluate a beam-search variant guided by our proposed decomposition and heuristic, showing that it matches or improves runtime over prior methods while offering greater flexibility in hyperparameters and computational resources. 

---
# Representation Interventions Enable Lifelong Unstructured Knowledge Control 

**Authors**: Xuyuan Liu, Zhengzhang Chen, Xinshuai Dong, Yanchi Liu, Xujiang Zhao, Shengyu Chen, Haoyu Wang, Yujun Yan, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.20892)  

**Abstract**: Large language models (LLMs) often produce incorrect or outdated content. Updating their knowledge efficiently and accurately without costly retraining is a major challenge. This problem is especially hard for complex, unstructured knowledge in a lifelong setting, where many edits must coexist without interference. We introduce RILKE (Representation Intervention for Lifelong KnowledgE Control), a robust and scalable method that treats knowledge control as interventions within the model's representation space. Leveraging representation-space expressiveness, we identify two properties enabling RILKE to deliver fine-grained control over complex, unstructured knowledge while maintaining general utility with frozen base weights. During training, RILKE learns paraphrase-robust and edit-localized modules that limit each update to a low-dimensional subspace to minimize cross-edit interference. In inference, a query-adaptive router selects the appropriate module to guide the model's generation. In evaluation on knowledge editing benchmarks with LLaMA and Qwen models, RILKE is scalable to large-scale datasets, demonstrating high edit success, strong paraphrase generalization, and preserving general utility with modest memory overhead. These results show RILKE is an effective and scalable solution for lifelong knowledge control in LLMs. 

---
# OpenApps: Simulating Environment Variations to Measure UI-Agent Reliability 

**Authors**: Karen Ullrich, Jingtong Su, Claudia Shi, Arjun Subramonian, Amir Bar, Ivan Evtimov, Nikolaos Tsilivis, Randall Balestriero, Julia Kempe, Mark Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2511.20766)  

**Abstract**: Reliability is key to realizing the promise of autonomous UI-Agents, multimodal agents that directly interact with apps in the same manner as humans, as users must be able to trust an agent to complete a given task. Current evaluations rely on fixed environments, often clones of existing apps, which are limited in that they can only shed light on whether or how often an agent can complete a task within a specific environment. When deployed however, agents are likely to encounter variations in app design and content that can affect an agent's ability to complete a task. To address this blind spot of measuring agent reliability across app variations, we develop OpenApps, a light-weight open-source ecosystem with six apps (messenger, calendar, maps, etc.) that are configurable in appearance and content. OpenApps requires just a single CPU to run, enabling easy generation and deployment of thousands of versions of each app. Specifically, we run more than 10,000 independent evaluations to study reliability across seven leading multimodal agents. We find that while standard reliability within a fixed app is relatively stable, reliability can vary drastically when measured across app variations. Task success rates for many agents can fluctuate by more than $50\%$ across app variations. For example, Kimi-VL-3B's average success across all tasks fluctuates from $63\%$ to just $4\%$ across app versions. We also find agent behaviors such as looping or hallucinating actions can differ drastically depending on the environment configuration. These initial findings highlight the importance of measuring reliability along this new dimension of app variations. OpenApps is available at this https URL 

---
# Learning Multi-Access Point Coordination in Agentic AI Wi-Fi with Large Language Models 

**Authors**: Yifan Fan, Le Liang, Peng Liu, Xiao Li, Ziyang Guo, Qiao Lan, Shi Jin, Wen Tong  

**Link**: [PDF](https://arxiv.org/pdf/2511.20719)  

**Abstract**: Multi-access point coordination (MAPC) is a key technology for enhancing throughput in next-generation Wi-Fi within dense overlapping basic service sets. However, existing MAPC protocols rely on static, protocol-defined rules, which limits their ability to adapt to dynamic network conditions such as varying interference levels and topologies. To address this limitation, we propose a novel Agentic AI Wi-Fi framework where each access point, modeled as an autonomous large language model agent, collaboratively reasons about the network state and negotiates adaptive coordination strategies in real time. This dynamic collaboration is achieved through a cognitive workflow that enables the agents to engage in natural language dialogue, leveraging integrated memory, reflection, and tool use to ground their decisions in past experience and environmental feedback. Comprehensive simulation results demonstrate that our agentic framework successfully learns to adapt to diverse and dynamic network environments, significantly outperforming the state-of-the-art spatial reuse baseline and validating its potential as a robust and intelligent solution for future wireless networks. 

---
# Cross Domain Evaluation of Multimodal Chain-of-Thought Reasoning of different datasets into the Amazon CoT Framework 

**Authors**: Nitya Tiwari, Parv Maheshwari, Vidisha Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2511.20701)  

**Abstract**: While recent work has extended CoT to multimodal settings, achieving state-of-the-art results on science question answering benchmarks like ScienceQA, the generalizability of these approaches across diverse domains remains underexplored. This work presents a comprehensive analysis of Multimodal Chain-of-Thought (Multimodal-CoT) reasoning, evaluating its effectiveness on the A-OKVQA, OKVQA and ChartQA datasets, which requires broad commonsense and world knowledge beyond scientific reasoning. We implement the two-stage framework proposed by Zhang et al. [3], which separates rationale generation from answer inference and integrates vision features through a gated fusion mechanism with T5-based language models. Through systematic ablation studies, we analyze the contributions of vision features, rationale quality, and architectural choices. Our findings reveal that while vision integration significantly reduces hallucination in rationale generation, the effectiveness of CoT reasoning varies substantially across question types, with commonsense reasoning presenting particular challenges. This work provides practical insights for researchers implementing multimodal reasoning systems and identifies key areas for future improvement in cross-domain generalization. 

---
# Paraconsistent-Lib: an intuitive PAL2v algorithm Python Library 

**Authors**: Arnaldo de Carvalho Junior, Diego Oliveira da Cruz, Bruno da Silva Alves, Fernando da Silva Paulo Junior, João Inacio da Silva Filho  

**Link**: [PDF](https://arxiv.org/pdf/2511.20700)  

**Abstract**: This paper introduces Paraconsistent-Lib, an open-source, easy-to-use Python library for building PAL2v algorithms in reasoning and decision-making systems. Paraconsistent-Lib is designed as a general-purpose library of PAL2v standard calculations, presenting three types of results: paraconsistent analysis in one of the 12 classical lattice PAL2v regions, paraconsistent analysis node (PAN) outputs, and a decision output. With Paraconsistent-Lib, well-known PAL2v algorithms such as Para-analyzer, ParaExtrCTX, PAL2v Filter, paraconsistent analysis network (PANnet), and paraconsistent neural network (PNN) can be written in stand-alone or network form, reducing complexity, code size, and bugs, as two examples presented in this paper. Given its stable state, Paraconsistent-Lib is an active development to respond to user-required features and enhancements received on GitHub. 

---
# A Brief History of Digital Twin Technology 

**Authors**: Yunqi Zhang, Kuangyu Shi, Biao Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.20695)  

**Abstract**: Emerging from NASA's spacecraft simulations in the 1960s, digital twin technology has advanced through industrial adoption to spark a healthcare transformation. A digital twin is a dynamic, data-driven virtual counterpart of a physical system, continuously updated through real-time data streams and capable of bidirectional interaction. In medicine, digital twin integrates imaging, biosensors, and computational models to generate patient-specific simulations that support diagnosis, treatment planning, and drug development. Representative applications include cardiac digital twin for predicting arrhythmia treatment outcomes, oncology digital twin for tracking tumor progression and optimizing radiotherapy, and pharmacological digital twin for accelerating drug discovery. Despite rapid progress, major challenges, including interoperability, data privacy, and model fidelity, continue to limit widespread clinical integration. Emerging solutions such as explainable AI, federated learning, and harmonized regulatory frameworks offer promising pathways forward. Looking ahead, advances in multi-organ digital twin, genomics integration, and ethical governance will be essential to ensure that digital twin shifts healthcare from reactive treatment to predictive, preventive, and truly personalized medicine. 

---
# Reasoning With a Star: A Heliophysics Dataset and Benchmark for Agentic Scientific Reasoning 

**Authors**: Kevin Lee, Russell Spiewak, James Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2511.20694)  

**Abstract**: Scientific reasoning through Large Language Models in heliophysics involves more than just recalling facts: it requires incorporating physical assumptions, maintaining consistent units, and providing clear scientific formats through coordinated approaches. To address these challenges, we present Reasoning With a Star, a newly contributed heliophysics dataset applicable to reasoning; we also provide an initial benchmarking approach. Our data are constructed from National Aeronautics and Space Administration & University Corporation for Atmospheric Research Living With a Star summer school problem sets and compiled into a readily consumable question-and-answer structure with question contexts, reasoning steps, expected answer type, ground-truth targets, format hints, and metadata. A programmatic grader checks the predictions using unit-aware numerical tolerance, symbolic equivalence, and schema validation. We benchmark a single-shot baseline and four multi-agent patterns, finding that decomposing workflows through systems engineering principles outperforms direct prompting on problems requiring deductive reasoning rather than pure inductive recall. 

---
# $A^2Flow:$ Automating Agentic Workflow Generation via Self-Adaptive Abstraction Operators 

**Authors**: Mingming Zhao, Xiaokang Wei, Yuanqi Shao, Kaiwen Zhou, Lin Yang, Siwei Rao, Junhui Zhan, Zhitang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.20693)  

**Abstract**: Large language models (LLMs) have shown strong potential in automating the design of agentic workflows. However, existing methods still rely heavily on manually predefined operators, limiting generalization and scalability. To address this issue, we propose $A^2Flow$, a fully automated framework for agentic workflow generation based on self-adaptive abstraction operators. $A^2Flow$ employs a three-stage operator extraction process: 1) Case-based Initial Operator Generation: leveraging expert demonstrations and LLM reasoning to generate case-specific operators; 2) Operator Clustering and Preliminary Abstraction: grouping similar operators across tasks to form preliminary abstractions; and 3) Deep Extraction for Abstract Execution Operators: applying long chain-of-thought prompting and multi-path reasoning to derive compact and generalizable execution operators. These operators serve as reusable building blocks for workflow construction without manual predefinition. Furthermore, we enhance node-level workflow search with an operator memory mechanism, which retains historical outputs to enrich context and improve decision-making. Experiments on general and embodied benchmarks show that $A^2Flow$ achieves a 2.4\% and 19.3\% average performance improvement and reduces resource usage by 37\% over state-of-the-art baselines. Homepage:this https URL 

---
# AssurAI: Experience with Constructing Korean Socio-cultural Datasets to Discover Potential Risks of Generative AI 

**Authors**: Chae-Gyun Lim, Seung-Ho Han, EunYoung Byun, Jeongyun Han, Soohyun Cho, Eojin Joo, Heehyeon Kim, Sieun Kim, Juhoon Lee, Hyunsoo Lee, Dongkun Lee, Jonghwan Hyeon, Yechan Hwang, Young-Jun Lee, Kyeongryul Lee, Minhyeong An, Hyunjun Ahn, Jeongwoo Son, Junho Park, Donggyu Yoon, Taehyung Kim, Jeemin Kim, Dasom Choi, Kwangyoung Lee, Hyunseung Lim, Yeohyun Jung, Jongok Hong, Sooyohn Nam, Joonyoung Park, Sungmin Na, Yubin Choi, Jeanne Choi, Yoojin Hong, Sueun Jang, Youngseok Seo, Somin Park, Seoungung Jo, Wonhye Chae, Yeeun Jo, Eunyoung Kim, Joyce Jiyoung Whang, HwaJung Hong, Joseph Seering, Uichin Lee, Juho Kim, Sunna Choi, Seokyeon Ko, Taeho Kim, Kyunghoon Kim, Myungsik Ha, So Jung Lee, Jemin Hwang, JoonHo Kwak, Ho-Jin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.20686)  

**Abstract**: The rapid evolution of generative AI necessitates robust safety evaluations. However, current safety datasets are predominantly English-centric, failing to capture specific risks in non-English, socio-cultural contexts such as Korean, and are often limited to the text modality. To address this gap, we introduce AssurAI, a new quality-controlled Korean multimodal dataset for evaluating the safety of generative AI. First, we define a taxonomy of 35 distinct AI risk factors, adapted from established frameworks by a multidisciplinary expert group to cover both universal harms and relevance to the Korean socio-cultural context. Second, leveraging this taxonomy, we construct and release AssurAI, a large-scale Korean multimodal dataset comprising 11,480 instances across text, image, video, and audio. Third, we apply the rigorous quality control process used to ensure data integrity, featuring a two-phase construction (i.e., expert-led seeding and crowdsourced scaling), triple independent annotation, and an iterative expert red-teaming loop. Our pilot study validates AssurAI's effectiveness in assessing the safety of recent LLMs. We release AssurAI to the public to facilitate the development of safer and more reliable generative AI systems for the Korean community. 

---
# Minimizing Hyperbolic Embedding Distortion with LLM-Guided Hierarchy Restructuring 

**Authors**: Melika Ayoughi, Pascal Mettes, Paul Groth  

**Link**: [PDF](https://arxiv.org/pdf/2511.20679)  

**Abstract**: Hyperbolic geometry is an effective geometry for embedding hierarchical data structures. Hyperbolic learning has therefore become increasingly prominent in machine learning applications where data is hierarchically organized or governed by hierarchical semantics, ranging from recommendation systems to computer vision. The quality of hyperbolic embeddings is tightly coupled to the structure of the input hierarchy, which is often derived from knowledge graphs or ontologies. Recent work has uncovered that for an optimal hyperbolic embedding, a high branching factor and single inheritance are key, while embedding algorithms are robust to imbalance and hierarchy size. To assist knowledge engineers in reorganizing hierarchical knowledge, this paper investigates whether Large Language Models (LLMs) have the ability to automatically restructure hierarchies to meet these criteria. We propose a prompt-based approach to transform existing hierarchies using LLMs, guided by known desiderata for hyperbolic embeddings. Experiments on 16 diverse hierarchies show that LLM-restructured hierarchies consistently yield higher-quality hyperbolic embeddings across several standard embedding quality metrics. Moreover, we show how LLM-guided hierarchy restructuring enables explainable reorganizations, providing justifications to knowledge engineers. 

---
# Revisiting Generalization Across Difficulty Levels: It's Not So Easy 

**Authors**: Yeganeh Kordi, Nihal V. Nayak, Max Zuo, Ilana Nguyen, Stephen H. Bach  

**Link**: [PDF](https://arxiv.org/pdf/2511.21692)  

**Abstract**: We investigate how well large language models (LLMs) generalize across different task difficulties, a key question for effective data curation and evaluation. Existing research is mixed regarding whether training on easier or harder data leads to better results, and whether those gains come on easier or harder test data. We address this question by conducting a systematic evaluation of LLMs' generalization across models, datasets, and fine-grained groups of example difficulty. We rank examples in six datasets using the outputs of thousands of different LLMs and Item Response Theory (IRT), a well-established difficulty metric in educational testing. Unlike prior work, our difficulty ratings are therefore determined solely by the abilities of many different LLMs, excluding human opinions of difficulty. With a more objective, larger-scale, and finer-grained analysis, we show that cross-difficulty generalization is often limited; training on either easy or hard data cannot achieve consistent improvements across the full range of difficulties. These results show the importance of having a range of difficulties in both training and evaluation data for LLMs, and that taking shortcuts with respect to difficulty is risky. 

---
# ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration 

**Authors**: Hongjin Su, Shizhe Diao, Ximing Lu, Mingjie Liu, Jiacheng Xu, Xin Dong, Yonggan Fu, Peter Belcak, Hanrong Ye, Hongxu Yin, Yi Dong, Evelina Bakhturina, Tao Yu, Yejin Choi, Jan Kautz, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2511.21689)  

**Abstract**: Large language models are powerful generalists, yet solving deep and complex problems such as those of the Humanity's Last Exam (HLE) remains both conceptually challenging and computationally expensive. We show that small orchestrators managing other models and a variety of tools can both push the upper bound of intelligence and improve efficiency in solving difficult agentic tasks. We introduce ToolOrchestra, a method for training small orchestrators that coordinate intelligent tools. ToolOrchestra explicitly uses reinforcement learning with outcome-, efficiency-, and user-preference-aware rewards. Using ToolOrchestra, we produce Orchestrator, an 8B model that achieves higher accuracy at lower cost than previous tool-use agents while aligning with user preferences on which tools are to be used for a given query. On HLE, Orchestrator achieves a score of 37.1%, outperforming GPT-5 (35.1%) while being 2.5x more efficient. On tau2-Bench and FRAMES, Orchestrator surpasses GPT-5 by a wide margin while using only about 30% of the cost. Extensive analysis shows that Orchestrator achieves the best trade-off between performance and cost under multiple metrics, and generalizes robustly to unseen tools. These results demonstrate that composing diverse tools with a lightweight orchestration model is both more efficient and more effective than existing methods, paving the way for practical and scalable tool-augmented reasoning systems. 

---
# G$^2$VLM: Geometry Grounded Vision Language Model with Unified 3D Reconstruction and Spatial Reasoning 

**Authors**: Wenbo Hu, Jingli Lin, Yilin Long, Yunlong Ran, Lihan Jiang, Yifan Wang, Chenming Zhu, Runsen Xu, Tai Wang, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21688)  

**Abstract**: Vision-Language Models (VLMs) still lack robustness in spatial intelligence, demonstrating poor performance on spatial understanding and reasoning tasks. We attribute this gap to the absence of a visual geometry learning process capable of reconstructing 3D space from 2D images. We present G$^2$VLM, a geometry grounded vision-language model that bridges two fundamental aspects of spatial intelligence: spatial 3D reconstruction and spatial understanding. G$^2$VLM natively leverages learned 3D visual geometry features to directly predict 3D attributes and enhance spatial reasoning tasks via in-context learning and interleaved reasoning. Our unified design is highly scalable for spatial understanding: it trains on abundant multi-view image and video data, while simultaneously leveraging the benefits of 3D visual priors that are typically only derived from hard-to-collect annotations. Experimental results demonstrate G$^2$VLM is proficient in both tasks, achieving comparable results to state-of-the-art feed-forward 3D reconstruction models and achieving better or competitive results across spatial understanding and reasoning tasks. By unifying a semantically strong VLM with low-level 3D vision tasks, we hope G$^2$VLM can serve as a strong baseline for the community and unlock more future applications, such as 3D scene editing. 

---
# Matrix: Peer-to-Peer Multi-Agent Synthetic Data Generation Framework 

**Authors**: Dong Wang, Yang Li, Ansong Ni, Ching-Feng Yeh, Youssef Emad, Xinjie Lei, Liam Robbins, Karthik Padthe, Hu Xu, Xian Li, Asli Celikyilmaz, Ramya Raghavendra, Lifei Huang, Carole-Jean Wu, Shang-Wen Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.21686)  

**Abstract**: Synthetic data has become increasingly important for training large language models, especially when real data is scarce, expensive, or privacy-sensitive. Many such generation tasks require coordinated multi-agent workflows, where specialized agents collaborate to produce data that is higher quality, more diverse, and structurally richer. However, existing frameworks for multi-agent synthesis often depend on a centralized orchestrator, creating scalability bottlenecks, or are hardcoded for specific domains, limiting flexibility. We present \textbf{Matrix}, a decentralized framework that represents both control and data flow as serialized messages passed through distributed queues. This peer-to-peer design eliminates the central orchestrator. Each task progresses independently through lightweight agents, while compute-intensive operations, such as LLM inference or containerized environments, are handled by distributed services. Built on Ray, Matrix scales to tens of thousands of concurrent agentic workflows and provides a modular, configurable design that enables easy adaptation to a wide range of data generation workflows. We evaluate Matrix across diverse synthesis scenarios, such as multi-agent collaborative dialogue, web-based reasoning data extraction, and tool-use trajectory generation in customer service environments. In all cases, Matrix achieves $2$--$15\times$ higher data generation throughput under identical hardware resources, without compromising output quality. 

---
# Through the telecom lens: Are all training samples important? 

**Authors**: Shruti Bothe, Illyyne Saffar, Aurelie Boisbunon, Hasan Farooq, Julien Forgeat, Md Moin Uddin Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2511.21668)  

**Abstract**: The rise of AI in telecommunications, from optimizing Radio Access Networks to managing user experience, has sharply increased data volumes and training demands. Telecom data is often noisy, high-dimensional, costly to store, process, and label. Despite Ai's critical role, standard workflows still assume all training samples contribute equally. On the other hand, next generation systems require AI models that are accurate, efficient, and this http URL paper questions the assumptions of equal importance by focusing on applying and analyzing the roles of individual samples in telecom training and assessing whether the proposed model optimizes computation and energy use. we perform sample-level gradient analysis across epochs to identify patterns of influence and redundancy in model learning. Based on this, we propose a sample importance framework thats electively prioritizes impactful data and reduces computation without compromising accuracy. Experiments on three real-world telecom datasets show that our method [reserves performance while reducing data needs and computational overhead while advancing the goals of sustainable AI in telecommunications. 

---
# Escaping the Verifier: Learning to Reason via Demonstrations 

**Authors**: Locke Cai, Ivan Provilkov  

**Link**: [PDF](https://arxiv.org/pdf/2511.21667)  

**Abstract**: Training Large Language Models (LLMs) to reason often relies on Reinforcement Learning (RL) with task-specific verifiers. However, many real-world reasoning-intensive tasks lack verifiers, despite offering abundant expert demonstrations that remain under-utilized for reasoning-focused training. We introduce RARO (Relativistic Adversarial Reasoning Optimization) that learns strong reasoning capabilities from only expert demonstrations via Inverse Reinforcement Learning. Our method sets up an adversarial interaction between a policy (generator) and a relativistic critic (discriminator): the policy learns to mimic expert answers, while the critic learns to compare and distinguish between policy and expert answers. Our method trains both the policy and the critic jointly and continuously via RL, and we identify the key stabilization techniques required for robust learning. Empirically, RARO significantly outperforms strong verifier-free baselines on all of our evaluation tasks -- Countdown, DeepMath, and Poetry Writing -- and enjoys the same robust scaling trends as RL on verifiable tasks. These results demonstrate that our method effectively elicits strong reasoning performance from expert demonstrations alone, enabling robust reasoning learning even when task-specific verifiers are unavailable. 

---
# Attention-Guided Patch-Wise Sparse Adversarial Attacks on Vision-Language-Action Models 

**Authors**: Naifu Zhang, Wei Tao, Xi Xiao, Qianpu Sun, Yuxin Zheng, Wentao Mo, Peiqiang Wang, Nan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21663)  

**Abstract**: In recent years, Vision-Language-Action (VLA) models in embodied intelligence have developed rapidly. However, existing adversarial attack methods require costly end-to-end training and often generate noticeable perturbation patches. To address these limitations, we propose ADVLA, a framework that directly applies adversarial perturbations on features projected from the visual encoder into the textual feature space. ADVLA efficiently disrupts downstream action predictions under low-amplitude constraints, and attention guidance allows the perturbations to be both focused and sparse. We introduce three strategies that enhance sensitivity, enforce sparsity, and concentrate perturbations. Experiments demonstrate that under an $L_{\infty}=4/255$ constraint, ADVLA combined with Top-K masking modifies less than 10% of the patches while achieving an attack success rate of nearly 100%. The perturbations are concentrated on critical regions, remain almost imperceptible in the overall image, and a single-step iteration takes only about 0.06 seconds, significantly outperforming conventional patch-based attacks. In summary, ADVLA effectively weakens downstream action predictions of VLA models under low-amplitude and locally sparse conditions, avoiding the high training costs and conspicuous perturbations of traditional patch attacks, and demonstrates unique effectiveness and practical value for attacking VLA feature spaces. 

---
# Continual Error Correction on Low-Resource Devices 

**Authors**: Kirill Paramonov, Mete Ozay, Aristeidis Mystakidis, Nikolaos Tsalikidis, Dimitrios Sotos, Anastasios Drosou, Dimitrios Tzovaras, Hyunjun Kim, Kiseok Chang, Sangdok Mo, Namwoong Kim, Woojong Yoo, Jijoong Moon, Umberto Michieli  

**Link**: [PDF](https://arxiv.org/pdf/2511.21652)  

**Abstract**: The proliferation of AI models in everyday devices has highlighted a critical challenge: prediction errors that degrade user experience. While existing solutions focus on error detection, they rarely provide efficient correction mechanisms, especially for resource-constrained devices. We present a novel system enabling users to correct AI misclassifications through few-shot learning, requiring minimal computational resources and storage. Our approach combines server-side foundation model training with on-device prototype-based classification, enabling efficient error correction through prototype updates rather than model retraining. The system consists of two key components: (1) a server-side pipeline that leverages knowledge distillation to transfer robust feature representations from foundation models to device-compatible architectures, and (2) a device-side mechanism that enables ultra-efficient error correction through prototype adaptation. We demonstrate our system's effectiveness on both image classification and object detection tasks, achieving over 50% error correction in one-shot scenarios on Food-101 and Flowers-102 datasets while maintaining minimal forgetting (less than 0.02%) and negligible computational overhead. Our implementation, validated through an Android demonstration app, proves the system's practicality in real-world scenarios. 

---
# Mechanisms of Non-Monotonic Scaling in Vision Transformers 

**Authors**: Anantha Padmanaban Krishna Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.21635)  

**Abstract**: Deeper Vision Transformers often perform worse than shallower ones, which challenges common scaling assumptions. Through a systematic empirical analysis of ViT-S, ViT-B, and ViT-L on ImageNet, we identify a consistent three-phase Cliff-Plateau-Climb pattern that governs how representations evolve with depth. We observe that better performance is associated with progressive marginalization of the [CLS] token, originally designed as a global aggregation hub, in favor of distributed consensus among patch tokens. We quantify patterns of information mixing with an Information Scrambling Index, and show that in ViT-L the information-task tradeoff emerges roughly 10 layers later than in ViT-B, and that these additional layers correlate with increased information diffusion rather than improved task performance. Taken together, these results suggest that transformer architectures in this regime may benefit more from carefully calibrated depth that executes clean phase transitions than from simply increasing parameter count. The Information Scrambling Index provides a useful diagnostic for existing models and suggests a potential design target for future architectures. All code is available at: this https URL. 

---
# Qwen3-VL Technical Report 

**Authors**: Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, Ke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.21631)  

**Abstract**: We introduce Qwen3-VL, the most capable vision-language model in the Qwen series to date, achieving superior performance across a broad range of multimodal benchmarks. It natively supports interleaved contexts of up to 256K tokens, seamlessly integrating text, images, and video. The model family includes both dense (2B/4B/8B/32B) and mixture-of-experts (30B-A3B/235B-A22B) variants to accommodate diverse latency-quality trade-offs. Qwen3-VL delivers three core pillars: (i) markedly stronger pure-text understanding, surpassing comparable text-only backbones in several cases; (ii) robust long-context comprehension with a native 256K-token window for both text and interleaved multimodal inputs, enabling faithful retention, retrieval, and cross-referencing across long documents and videos; and (iii) advanced multimodal reasoning across single-image, multi-image, and video tasks, demonstrating leading performance on comprehensive evaluations such as MMMU and visual-math benchmarks (e.g., MathVista and MathVision). Architecturally, we introduce three key upgrades: (i) an enhanced interleaved-MRoPE for stronger spatial-temporal modeling across images and video; (ii) DeepStack integration, which effectively leverages multi-level ViT features to tighten vision-language alignment; and (iii) text-based time alignment for video, evolving from T-RoPE to explicit textual timestamp alignment for more precise temporal grounding. Under comparable token budgets and latency constraints, Qwen3-VL achieves superior performance in both dense and Mixture-of-Experts (MoE) architectures. We envision Qwen3-VL serving as a foundational engine for image-grounded reasoning, agentic decision-making, and multimodal code intelligence in real-world workflows. 

---
# Scale-Agnostic Kolmogorov-Arnold Geometry in Neural Networks 

**Authors**: Mathew Vanherreweghe, Michael H. Freedman, Keith M. Adams  

**Link**: [PDF](https://arxiv.org/pdf/2511.21626)  

**Abstract**: Recent work by Freedman and Mulligan demonstrated that shallow multilayer perceptrons spontaneously develop Kolmogorov-Arnold geometric (KAG) structure during training on synthetic three-dimensional tasks. However, it remained unclear whether this phenomenon persists in realistic high-dimensional settings and what spatial properties this geometry exhibits.
We extend KAG analysis to MNIST digit classification (784 dimensions) using 2-layer MLPs with systematic spatial analysis at multiple scales. We find that KAG emerges during training and appears consistently across spatial scales, from local 7-pixel neighborhoods to the full 28x28 image. This scale-agnostic property holds across different training procedures: both standard training and training with spatial augmentation produce the same qualitative pattern. These findings reveal that neural networks spontaneously develop organized, scale-invariant geometric structure during learning on realistic high-dimensional data. 

---
# On the Origin of Algorithmic Progress in AI 

**Authors**: Hans Gundlach, Alex Fogelson, Jayson Lynch, Ana Trisovic, Jonathan Rosenfeld, Anmol Sandhu, Neil Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2511.21622)  

**Abstract**: Algorithms have been estimated to increase AI training FLOP efficiency by a factor of 22,000 between 2012 and 2023 [Ho et al., 2024]. Running small-scale ablation experiments on key innovations from this time period, we are able to account for less than 10x of these gains. Surveying the broader literature, we estimate that additional innovations not included in our ablations account for less than 10x, yielding a total under 100x. This leads us to conduct scaling experiments, which reveal that much of this efficiency gap can be explained by algorithms with scale-dependent efficiency improvements. In particular, we conduct scaling experiments between LSTMs and Transformers, finding exponent differences in their compute-optimal scaling law while finding little scaling difference for many other innovations. These experiments demonstrate that - contrary to standard assumptions - an algorithm's efficiency gains are tied to compute scale. Using experimental extrapolation and literature estimates, we account for 6,930x efficiency gains over the same time period, with the scale-dependent LSTM-to-Transformer transition accounting for the majority of gains. Our results indicate that algorithmic progress for small models has been far slower than previously assumed, and that measures of algorithmic efficiency are strongly reference-dependent. 

---
# Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining 

**Authors**: Dongyang Fan, Diba Hashemi, Sai Praneeth Karimireddy, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2511.21613)  

**Abstract**: Incorporating metadata in Large Language Models (LLMs) pretraining has recently emerged as a promising approach to accelerate training. However prior work highlighted only one useful signal-URLs, leaving open the question of whether other forms of metadata could yield greater benefits. In this study, we investigate a wider range of metadata types and find other types of metadata, such as fine-grained indicators of document quality that can also accelerate pretraining when prepended. We identify a common feature among effective metadata: they encode information at a finer granularity. We further introduce metadata appending as a means of improving training efficiency, where predicting an appropriate metadata as auxiliary task can help speed up pretraining. In addition, learnable meta-tokens trained with masked loss can recover part of the speedup by inducing quality-aware latent structure. Using probing, we analyze latent representations to understand how metadata shapes learning. Together, these results yield practical guidelines for integrating metadata to improve both the efficiency and effectiveness of LLM pretraining. 

---
# Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving 

**Authors**: Haohong Lin, Yunzhi Zhang, Wenhao Ding, Jiajun Wu, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.21584)  

**Abstract**: End-to-end (E2E) autonomous driving models have demonstrated strong performance in open-loop evaluations but often suffer from cascading errors and poor generalization in closed-loop settings. To address this gap, we propose Model-based Policy Adaptation (MPA), a general framework that enhances the robustness and safety of pretrained E2E driving agents during deployment. MPA first generates diverse counterfactual trajectories using a geometry-consistent simulation engine, exposing the agent to scenarios beyond the original dataset. Based on this generated data, MPA trains a diffusion-based policy adapter to refine the base policy's predictions and a multi-step Q value model to evaluate long-term outcomes. At inference time, the adapter proposes multiple trajectory candidates, and the Q value model selects the one with the highest expected utility. Experiments on the nuScenes benchmark using a photorealistic closed-loop simulator demonstrate that MPA significantly improves performance across in-domain, out-of-domain, and safety-critical scenarios. We further investigate how the scale of counterfactual data and inference-time guidance strategies affect overall effectiveness. 

---
# HarmonicAttack: An Adaptive Cross-Domain Audio Watermark Removal 

**Authors**: Kexin Li, Xiao Hu, Ilya Grishchenko, David Lie  

**Link**: [PDF](https://arxiv.org/pdf/2511.21577)  

**Abstract**: The availability of high-quality, AI-generated audio raises security challenges such as misinformation campaigns and voice-cloning fraud. A key defense against the misuse of AI-generated audio is by watermarking it, so that it can be easily distinguished from genuine audio. As those seeking to misuse AI-generated audio may thus seek to remove audio watermarks, studying effective watermark removal techniques is critical to being able to objectively evaluate the robustness of audio watermarks against removal. Previous watermark removal schemes either assume impractical knowledge of the watermarks they are designed to remove or are computationally expensive, potentially generating a false sense of confidence in current watermark schemes.
We introduce HarmonicAttack, an efficient audio watermark removal method that only requires the basic ability to generate the watermarks from the targeted scheme and nothing else. With this, we are able to train a general watermark removal model that is able to remove the watermarks generated by the targeted scheme from any watermarked audio sample. HarmonicAttack employs a dual-path convolutional autoencoder that operates in both temporal and frequency domains, along with GAN-style training, to separate the watermark from the original audio. When evaluated against state-of-the-art watermark schemes AudioSeal, WavMark, and Silentcipher, HarmonicAttack demonstrates greater watermark removal ability than previous watermark removal methods with near real-time performance. Moreover, while HarmonicAttack requires training, we find that it is able to transfer to out-of-distribution samples with minimal degradation in performance. 

---
# Multimodal Robust Prompt Distillation for 3D Point Cloud Models 

**Authors**: Xiang Gu, Liming Lu, Xu Zheng, Anan Du, Yongbin Zhou, Shuchao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21574)  

**Abstract**: Adversarial attacks pose a significant threat to learning-based 3D point cloud models, critically undermining their reliability in security-sensitive applications. Existing defense methods often suffer from (1) high computational overhead and (2) poor generalization ability across diverse attack types. To bridge these gaps, we propose a novel yet efficient teacher-student framework, namely Multimodal Robust Prompt Distillation (MRPD) for distilling robust 3D point cloud model. It learns lightweight prompts by aligning student point cloud model's features with robust embeddings from three distinct teachers: a vision model processing depth projections, a high-performance 3D model, and a text encoder. To ensure a reliable knowledge transfer, this distillation is guided by a confidence-gated mechanism which dynamically balances the contribution of all input modalities. Notably, since the distillation is all during the training stage, there is no additional computational cost at inference. Extensive experiments demonstrate that MRPD substantially outperforms state-of-the-art defense methods against a wide range of white-box and black-box attacks, while even achieving better performance on clean data. Our work presents a new, practical paradigm for building robust 3D vision systems by efficiently harnessing multimodal knowledge. 

---
# BAMAS: Structuring Budget-Aware Multi-Agent Systems 

**Authors**: Liming Yang, Junyu Luo, Xuanzhe Liu, Yiling Lou, Zhenpeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.21572)  

**Abstract**: Large language model (LLM)-based multi-agent systems have emerged as a powerful paradigm for enabling autonomous agents to solve complex tasks. As these systems scale in complexity, cost becomes an important consideration for practical deployment. However, existing work rarely addresses how to structure multi-agent systems under explicit budget constraints. In this paper, we propose BAMAS, a novel approach for building multi-agent systems with budget awareness. BAMAS first selects an optimal set of LLMs by formulating and solving an Integer Linear Programming problem that balances performance and cost. It then determines how these LLMs should collaborate by leveraging a reinforcement learning-based method to select the interaction topology. Finally, the system is instantiated and executed based on the selected agents and their collaboration topology. We evaluate BAMAS on three representative tasks and compare it with state-of-the-art agent construction methods. Results show that BAMAS achieves comparable performance while reducing cost by up to 86%. 

---
# VacuumVLA: Boosting VLA Capabilities via a Unified Suction and Gripping Tool for Complex Robotic Manipulation 

**Authors**: Hui Zhou, Siyuan Huang, Minxing Li, Hao Zhang, Lue Fan, Shaoshuai Shi  

**Link**: [PDF](https://arxiv.org/pdf/2511.21557)  

**Abstract**: Vision Language Action models have significantly advanced general purpose robotic manipulation by harnessing large scale pretrained vision and language representations. Among existing approaches, a majority of current VLA systems employ parallel two finger grippers as their default end effectors. However, such grippers face inherent limitations in handling certain real world tasks such as wiping glass surfaces or opening drawers without handles due to insufficient contact area or lack of adhesion. To overcome these challenges, we present a low cost, integrated hardware design that combines a mechanical two finger gripper with a vacuum suction unit, enabling dual mode manipulation within a single end effector. Our system supports flexible switching or synergistic use of both modalities, expanding the range of feasible tasks. We validate the efficiency and practicality of our design within two state of the art VLA frameworks: DexVLA and Pi0. Experimental results demonstrate that with the proposed hybrid end effector, robots can successfully perform multiple complex tasks that are infeasible for conventional two finger grippers alone. All hardware designs and controlling systems will be released. 

---
# Predictive Safety Shield for Dyna-Q Reinforcement Learning 

**Authors**: Jin Pin, Krasowski Hanna, Vanneaux Elena  

**Link**: [PDF](https://arxiv.org/pdf/2511.21531)  

**Abstract**: Obtaining safety guarantees for reinforcement learning is a major challenge to achieve applicability for real-world tasks. Safety shields extend standard reinforcement learning and achieve hard safety guarantees. However, existing safety shields commonly use random sampling of safe actions or a fixed fallback controller, therefore disregarding future performance implications of different safe actions. In this work, we propose a predictive safety shield for model-based reinforcement learning agents in discrete space. Our safety shield updates the Q-function locally based on safe predictions, which originate from a safe simulation of the environment model. This shielding approach improves performance while maintaining hard safety guarantees. Our experiments on gridworld environments demonstrate that even short prediction horizons can be sufficient to identify the optimal path. We observe that our approach is robust to distribution shifts, e.g., between simulation and reality, without requiring additional training. 

---
# Voice, Bias, and Coreference: An Interpretability Study of Gender in Speech Translation 

**Authors**: Lina Conti, Dennis Fucci, Marco Gaido, Matteo Negri, Guillaume Wisniewski, Luisa Bentivogli  

**Link**: [PDF](https://arxiv.org/pdf/2511.21517)  

**Abstract**: Unlike text, speech conveys information about the speaker, such as gender, through acoustic cues like pitch. This gives rise to modality-specific bias concerns. For example, in speech translation (ST), when translating from languages with notional gender, such as English, into languages where gender-ambiguous terms referring to the speaker are assigned grammatical gender, the speaker's vocal characteristics may play a role in gender assignment. This risks misgendering speakers, whether through masculine defaults or vocal-based assumptions. Yet, how ST models make these decisions remains poorly understood. We investigate the mechanisms ST models use to assign gender to speaker-referring terms across three language pairs (en-es/fr/it), examining how training data patterns, internal language model (ILM) biases, and acoustic information interact. We find that models do not simply replicate term-specific gender associations from training data, but learn broader patterns of masculine prevalence. While the ILM exhibits strong masculine bias, models can override these preferences based on acoustic input. Using contrastive feature attribution on spectrograms, we reveal that the model with higher gender accuracy relies on a previously unknown mechanism: using first-person pronouns to link gendered terms back to the speaker, accessing gender information distributed across the frequency spectrum rather than concentrated in pitch. 

---
# Mechanistic Interpretability for Transformer-based Time Series Classification 

**Authors**: Matīss Kalnāre, Sofoklis Kitharidis, Thomas Bäck, Niki van Stein  

**Link**: [PDF](https://arxiv.org/pdf/2511.21514)  

**Abstract**: Transformer-based models have become state-of-the-art tools in various machine learning tasks, including time series classification, yet their complexity makes understanding their internal decision-making challenging. Existing explainability methods often focus on input-output attributions, leaving the internal mechanisms largely opaque. This paper addresses this gap by adapting various Mechanistic Interpretability techniques; activation patching, attention saliency, and sparse autoencoders, from NLP to transformer architectures designed explicitly for time series classification. We systematically probe the internal causal roles of individual attention heads and timesteps, revealing causal structures within these models. Through experimentation on a benchmark time series dataset, we construct causal graphs illustrating how information propagates internally, highlighting key attention heads and temporal positions driving correct classifications. Additionally, we demonstrate the potential of sparse autoencoders for uncovering interpretable latent features. Our findings provide both methodological contributions to transformer interpretability and novel insights into the functional mechanics underlying transformer performance in time series classification tasks. 

---
# Tool-RoCo: An Agent-as-Tool Self-organization Large Language Model Benchmark in Multi-robot Cooperation 

**Authors**: Ke Zhang, Xiaoning Zhao, Ce Zheng, Jiahong Ning, Dandan Zhu, Wenqi Zhang, Chen Sun, Toshiharu Sugawara  

**Link**: [PDF](https://arxiv.org/pdf/2511.21510)  

**Abstract**: This study proposes Tool-RoCo, a novel benchmark for evaluating large language models (LLMs) in long-term multi-agent cooperation based on RoCo, a multi-robot cooperative benchmark. Recent research on LLM-based multi-agent systems has relied on predefined orchestration, while ignoring agent autonomy. Tool-RoCo treats other agents as tools and introduces cooperative tools, leveraging tool usage to evaluate multi-agent cooperation and self-organization. Tool usage means that each agent (LLM) selects a tool from a candidate set based on the current state, receives feedback, and adjusts its selection in subsequent rounds. To evaluate different autonomy levels, we propose four LLM paradigms: (1) centralized cooperation, where a single LLM allocates tools to all agents; (2) centralized self-organization, where a central LLM autonomously activates agents while keeping others inactive; (3) decentralized cooperation, where each agent has its own LLM and calls tools based on local information; and (4) self-organization, where a randomly chosen initial agent can request collaboration, activating additional agents via tool calls. Tool-RoCo includes three multi-robot tasks, SORT, PACK, and CABINET, to measure format and parameter accuracy and agent coordination through tool usage. The results using several LLMs showed that cooperative tools accounted for only 7.09% of all tools, indicating that LLM-based agents rarely invoked others as assistants. Moreover, activation tools accounted for 96.42%, suggesting that current LLMs tend to maintain active agents while seldom deactivating them for adaptive coordination. Tool-RoCo provides a systematic benchmark to evaluate LLM autonomy and cooperation in multi-agent tasks. Code and Demo: this https URL 

---
# Merge and Bound: Direct Manipulations on Weights for Class Incremental Learning 

**Authors**: Taehoon Kim, Donghwan Jang, Bohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.21490)  

**Abstract**: We present a novel training approach, named Merge-and-Bound (M&B) for Class Incremental Learning (CIL), which directly manipulates model weights in the parameter space for optimization. Our algorithm involves two types of weight merging: inter-task weight merging and intra-task weight merging. Inter-task weight merging unifies previous models by averaging the weights of models from all previous stages. On the other hand, intra-task weight merging facilitates the learning of current task by combining the model parameters within current stage. For reliable weight merging, we also propose a bounded update technique that aims to optimize the target model with minimal cumulative updates and preserve knowledge from previous tasks; this strategy reveals that it is possible to effectively obtain new models near old ones, reducing catastrophic forgetting. M&B is seamlessly integrated into existing CIL methods without modifying architecture components or revising learning objectives. We extensively evaluate our algorithm on standard CIL benchmarks and demonstrate superior performance compared to state-of-the-art methods. 

---
# Frequency-Aware Token Reduction for Efficient Vision Transformer 

**Authors**: Dong-Jae Lee, Jiwan Hur, Jaehyun Choi, Jaemyung Yu, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.21477)  

**Abstract**: Vision Transformers have demonstrated exceptional performance across various computer vision tasks, yet their quadratic computational complexity concerning token length remains a significant challenge. To address this, token reduction methods have been widely explored. However, existing approaches often overlook the frequency characteristics of self-attention, such as rank collapsing and over-smoothing phenomenon. In this paper, we propose a frequency-aware token reduction strategy that improves computational efficiency while preserving performance by mitigating rank collapsing. Our method partitions tokens into high-frequency tokens and low-frequency tokens. high-frequency tokens are selectively preserved, while low-frequency tokens are aggregated into a compact direct current token to retain essential low-frequency components. Through extensive experiments and analysis, we demonstrate that our approach significantly improves accuracy while reducing computational overhead and mitigating rank collapsing and over smoothing. Furthermore, we analyze the previous methods, shedding light on their implicit frequency characteristics and limitations. 

---
# Going with the Speed of Sound: Pushing Neural Surrogates into Highly-turbulent Transonic Regimes 

**Authors**: Fabian Paischer, Leo Cotteleer, Yann Dreze, Richard Kurle, Dylan Rubini, Maurits Bleeker, Tobias Kronlachner, Johannes Brandstetter  

**Link**: [PDF](https://arxiv.org/pdf/2511.21474)  

**Abstract**: The widespread use of neural surrogates in automotive aerodynamics, enabled by datasets such as DrivAerML and DrivAerNet++, has primarily focused on bluff-body flows with large wakes. Extending these methods to aerospace, particularly in the transonic regime, remains challenging due to the high level of non-linearity of compressible flows and 3D effects such as wingtip vortices. Existing aerospace datasets predominantly focus on 2D airfoils, neglecting these critical 3D phenomena. To address this gap, we present a new dataset of CFD simulations for 3D wings in the transonic regime. The dataset comprises volumetric and surface-level fields for around $30,000$ samples with unique geometry and inflow conditions. This allows computation of lift and drag coefficients, providing a foundation for data-driven aerodynamic optimization of the drag-lift Pareto front. We evaluate several state-of-the-art neural surrogates on our dataset, including Transolver and AB-UPT, focusing on their out-of-distribution (OOD) generalization over geometry and inflow variations. AB-UPT demonstrates strong performance for transonic flowfields and reproduces physically consistent drag-lift Pareto fronts even for unseen wing configurations. Our results demonstrate that AB-UPT can approximate drag-lift Pareto fronts for unseen geometries, highlighting its potential as an efficient and effective tool for rapid aerodynamic design exploration. To facilitate future research, we open-source our dataset at this https URL. 

---
# Hierarchical Ranking Neural Network for Long Document Readability Assessment 

**Authors**: Yurui Zheng, Yijun Chen, Shaohong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21473)  

**Abstract**: Readability assessment aims to evaluate the reading difficulty of a text. In recent years, while deep learning technology has been gradually applied to readability assessment, most approaches fail to consider either the length of the text or the ordinal relationship of readability labels. This paper proposes a bidirectional readability assessment mechanism that captures contextual information to identify regions with rich semantic information in the text, thereby predicting the readability level of individual sentences. These sentence-level labels are then used to assist in predicting the overall readability level of the document. Additionally, a pairwise sorting algorithm is introduced to model the ordinal relationship between readability levels through label subtraction. Experimental results on Chinese and English datasets demonstrate that the proposed model achieves competitive performance and outperforms other baseline models. 

---
# Constructing and Benchmarking: a Labeled Email Dataset for Text-Based Phishing and Spam Detection Framework 

**Authors**: Rebeka Toth, Tamas Bisztray, Richard Dubniczky  

**Link**: [PDF](https://arxiv.org/pdf/2511.21448)  

**Abstract**: Phishing and spam emails remain a major cybersecurity threat, with attackers increasingly leveraging Large Language Models (LLMs) to craft highly deceptive content. This study presents a comprehensive email dataset containing phishing, spam, and legitimate messages, explicitly distinguishing between human- and LLM-generated content. Each email is annotated with its category, emotional appeal (e.g., urgency, fear, authority), and underlying motivation (e.g., link-following, credential theft, financial fraud). We benchmark multiple LLMs on their ability to identify these emotional and motivational cues and select the most reliable model to annotate the full dataset. To evaluate classification robustness, emails were also rephrased using several LLMs while preserving meaning and intent. A state-of-the-art LLM was then assessed on its performance across both original and rephrased emails using expert-labeled ground truth. The results highlight strong phishing detection capabilities but reveal persistent challenges in distinguishing spam from legitimate emails. Our dataset and evaluation framework contribute to improving AI-assisted email security systems. To support open science, all code, templates, and resources are available on our project site. 

---
# EvRainDrop: HyperGraph-guided Completion for Effective Frame and Event Stream Aggregation 

**Authors**: Futian Wang, Fan Zhang, Xiao Wang, Mengqi Wang, Dexing Huang, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21439)  

**Abstract**: Event cameras produce asynchronous event streams that are spatially sparse yet temporally dense. Mainstream event representation learning algorithms typically use event frames, voxels, or tensors as input. Although these approaches have achieved notable progress, they struggle to address the undersampling problem caused by spatial sparsity. In this paper, we propose a novel hypergraph-guided spatio-temporal event stream completion mechanism, which connects event tokens across different times and spatial locations via hypergraphs and leverages contextual information message passing to complete these sparse events. The proposed method can flexibly incorporate RGB tokens as nodes in the hypergraph within this completion framework, enabling multi-modal hypergraph-based information completion. Subsequently, we aggregate hypergraph node information across different time steps through self-attention, enabling effective learning and fusion of multi-modal features. Extensive experiments on both single- and multi-label event classification tasks fully validated the effectiveness of our proposed framework. The source code of this paper will be released on this https URL. 

---
# From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings 

**Authors**: Jiajie Zhang, Sören Schwertfeger, Alexander Kleiner  

**Link**: [PDF](https://arxiv.org/pdf/2511.21428)  

**Abstract**: We present a novel unsupervised framework to unlock vast unlabeled human demonstration data from continuous industrial video streams for Vision-Language-Action (VLA) model pre-training. Our method first trains a lightweight motion tokenizer to encode motion dynamics, then employs an unsupervised action segmenter leveraging a novel "Latent Action Energy" metric to discover and segment semantically coherent action primitives. The pipeline outputs both segmented video clips and their corresponding latent action sequences, providing structured data directly suitable for VLA pre-training. Evaluations on public benchmarks and a proprietary electric motor assembly dataset demonstrate effective segmentation of key tasks performed by humans at workstations. Further clustering and quantitative assessment via a Vision-Language Model confirm the semantic coherence of the discovered action primitives. To our knowledge, this is the first fully automated end-to-end system for extracting and organizing VLA pre-training data from unstructured industrial videos, offering a scalable solution for embodied AI integration in manufacturing. 

---
# SAM Guided Semantic and Motion Changed Region Mining for Remote Sensing Change Captioning 

**Authors**: Futian Wang, Mengqi Wang, Xiao Wang, Haowen Wang, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21420)  

**Abstract**: Remote sensing change captioning is an emerging and popular research task that aims to describe, in natural language, the content of interest that has changed between two remote sensing images captured at different times. Existing methods typically employ CNNs/Transformers to extract visual representations from the given images or incorporate auxiliary tasks to enhance the final results, with weak region awareness and limited temporal alignment. To address these issues, this paper explores the use of the SAM (Segment Anything Model) foundation model to extract region-level representations and inject region-of-interest knowledge into the captioning framework. Specifically, we employ a CNN/Transformer model to extract global-level vision features, leverage the SAM foundation model to delineate semantic- and motion-level change regions, and utilize a specially constructed knowledge graph to provide information about objects of interest. These heterogeneous sources of information are then fused via cross-attention, and a Transformer decoder is used to generate the final natural language description of the observed changes. Extensive experimental results demonstrate that our method achieves state-of-the-art performance across multiple widely used benchmark datasets. The source code of this paper will be released on this https URL 

---
# Automated Dynamic AI Inference Scaling on HPC-Infrastructure: Integrating Kubernetes, Slurm and vLLM 

**Authors**: Tim Trappen, Robert Keßler, Roland Pabel, Viktor Achter, Stefan Wesner  

**Link**: [PDF](https://arxiv.org/pdf/2511.21413)  

**Abstract**: Due to rising demands for Artificial Inteligence (AI) inference, especially in higher education, novel solutions utilising existing infrastructure are emerging. The utilisation of High-Performance Computing (HPC) has become a prevalent approach for the implementation of such solutions. However, the classical operating model of HPC does not adapt well to the requirements of synchronous, user-facing dynamic AI application workloads. In this paper, we propose our solution that serves LLMs by integrating vLLM, Slurm and Kubernetes on the supercomputer \textit{RAMSES}. The initial benchmark indicates that the proposed architecture scales efficiently for 100, 500 and 1000 concurrent requests, incurring only an overhead of approximately 500 ms in terms of end-to-end latency. 

---
# Subjective Depth and Timescale Transformers: Learning Where and When to Compute 

**Authors**: Frederico Wieser, Martin Benfeghoul, Haitham Bou Ammar, Jun Wang, Zafeirios Fountas  

**Link**: [PDF](https://arxiv.org/pdf/2511.21408)  

**Abstract**: The rigid, uniform allocation of computation in standard Transformer (TF) architectures can limit their efficiency and scalability, particularly for large-scale models and long sequences. Addressing this, we introduce Subjective Depth Transformers (SDT) and Subjective Timescale Transformers (STT), two distinct architectures that leverage Bayesian surprise signals to dynamically route computation, learning where and when to compute within decoder-only TFs. SDT augments a decoder-only stack with alternating Decision and Dynamic layers: a Decision layer computes a full block 'posterior' and a lightweight 'prior,' while a Dynamic layer employs fixed-capacity Top-K routing based on Bayesian surprise (Expected and Unexpected Change), maintaining a static compute graph. STT extends this conditional computation to the temporal domain: a transition network predicts residual updates, forming a temporal 'change hypothesis' that informs a router to dynamically execute or bypass TF blocks for each token, managing KV-cache contributions. Both architectures exhibit the predicted shift from novelty to prediction driven gating over training, suggesting alignment with surprise based principles. While operating at reduced capacity, they offer preliminary insights into the compute-accuracy trade-offs of conditional computation. The proposed architectures establish a flexible framework for efficiency, reducing self-attention computation by 75% and KV-cache requirements by 50% within each compute skipping layer, setting a pathway for more efficient models. 

---
# Training Introspective Behavior: Fine-Tuning Induces Reliable Internal State Detection in a 7B Model 

**Authors**: Joshua Fonseca Rivera  

**Link**: [PDF](https://arxiv.org/pdf/2511.21399)  

**Abstract**: Lindsey (2025) investigates introspective awareness in language models through four experiments, finding that models can sometimes detect and identify injected activation patterns -- but unreliably (~20% success in the best model). We focus on the first of these experiments -- self-report of injected "thoughts" -- and ask whether this capability can be directly trained rather than waiting for emergence. Through fine-tuning on transient single-token injections, we transform a 7B parameter model from near-complete failure (0.4% accuracy, 6.7% false positive rate) to reliable detection (85% accuracy on held-out concepts at {\alpha}=40, 0% false positives). Our model detects fleeting "thoughts" injected at a single token position, retains that information, and reports the semantic content across subsequent generation steps. On this task, our trained model satisfies three of Lindsey's criteria: accuracy (correct identification), grounding (0/60 false positives), and internality (detection precedes verbalization). Generalization to unseen concept vectors (7.5pp gap) demonstrates the model learns a transferable skill rather than memorizing specific vectors, though this does not establish metacognitive representation in Lindsey's sense. These results address an open question raised by Lindsey: whether "training for introspection would help eliminate cross-model differences." We show that at least one component of introspective behavior can be directly induced, offering a pathway to built-in AI transparency. 

---
# Do Reasoning Vision-Language Models Inversely Scale in Test-Time Compute? A Distractor-centric Empirical Analysis 

**Authors**: Jiyun Bae, Hyunjong Ok, Sangwoo Mo, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.21397)  

**Abstract**: How does irrelevant information (i.e., distractors) affect test-time scaling in vision-language models (VLMs)? Prior studies on language models have reported an inverse scaling effect, where textual distractors lead to longer but less effective reasoning. To investigate whether similar phenomena occur in multimodal settings, we introduce Idis (Images with distractors), a visual question-answering dataset that systematically varies distractors along semantic, numerical, and spatial dimensions. Our analyses reveal that visual distractors differ fundamentally from textual ones: although inverse scaling persists, adding visual distractors reduces accuracy without increasing reasoning length. We further show that tracking attribute counts within reasoning traces provides key insights into how distractors, reasoning length, and accuracy interact. Finally, we demonstrate that these trends extend to established visual bias benchmarks such as Waterbirds, and we propose a simple prompting strategy to mitigate bias-driven predictions in reasoning models. 

---
# Monet: Reasoning in Latent Visual Space Beyond Images and Language 

**Authors**: Qixun Wang, Yang Shi, Yifei Wang, Yuanxing Zhang, Pengfei Wan, Kun Gai, Xianghua Ying, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21395)  

**Abstract**: "Thinking with images" has emerged as an effective paradigm for advancing visual reasoning, extending beyond text-only chains of thought by injecting visual evidence into intermediate reasoning steps. However, existing methods fall short of human-like abstract visual thinking, as their flexibility is fundamentally limited by external tools. In this work, we introduce Monet, a training framework that enables multimodal large language models (MLLMs) to reason directly within the latent visual space by generating continuous embeddings that function as intermediate visual thoughts. We identify two core challenges in training MLLMs for latent visual reasoning: high computational cost in latent-vision alignment and insufficient supervision over latent embeddings, and address them with a three-stage distillation-based supervised fine-tuning (SFT) pipeline. We further reveal a limitation of applying GRPO to latent reasoning: it primarily enhances text-based reasoning rather than latent reasoning. To overcome this, we propose VLPO (Visual-latent Policy Optimization), a reinforcement learning method that explicitly incorporates latent embeddings into policy gradient updates. To support SFT, we construct Monet-SFT-125K, a high-quality text-image interleaved CoT dataset containing 125K real-world, chart, OCR, and geometry CoTs. Our model, Monet-7B, shows consistent gains across real-world perception and reasoning benchmarks and exhibits strong out-of-distribution generalization on challenging abstract visual reasoning tasks. We also empirically analyze the role of each training component and discuss our early unsuccessful attempts, providing insights for future developments in visual latent reasoning. Our model, data, and code are available at this https URL. 

---
# RIA: A Ranking-Infused Approach for Optimized listwise CTR Prediction 

**Authors**: Guoxiao Zhang, Tan Qu, Ao Li, DongLin Ni, Qianlong Xie, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21394)  

**Abstract**: Reranking improves recommendation quality by modeling item interactions. However, existing methods often decouple ranking and reranking, leading to weak listwise evaluation models that suffer from combinatorial sparsity and limited representational power under strict latency constraints. In this paper, we propose RIA (Ranking-Infused Architecture), a unified, end-to-end framework that seamlessly integrates pointwise and listwise evaluation. RIA introduces four key components: (1) the User and Candidate DualTransformer (UCDT) for fine-grained user-item-context modeling; (2) the Context-aware User History and Target (CUHT) module for position-sensitive preference learning; (3) the Listwise Multi-HSTU (LMH) module to capture hierarchical item dependencies; and (4) the Embedding Cache (EC) module to bridge efficiency and effectiveness during inference. By sharing representations across ranking and reranking, RIA enables rich contextual knowledge transfer while maintaining low latency. Extensive experiments show that RIA outperforms state-of-the-art models on both public and industrial datasets, achieving significant gains in AUC and LogLoss. Deployed in Meituan advertising system, RIA yields a +1.69% improvement in Click-Through Rate (CTR) and a +4.54% increase in Cost Per Mille (CPM) in online A/B tests. 

---
# FITRep: Attention-Guided Item Representation via MLLMs 

**Authors**: Guoxiao Zhang, Ao Li, Tan Qu, Qianlong Xie, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21389)  

**Abstract**: Online platforms usually suffer from user experience degradation due to near-duplicate items with similar visuals and text. While Multimodal Large Language Models (MLLMs) enable multimodal embedding, existing methods treat representations as black boxes, ignoring structural relationships (e.g., primary vs. auxiliary elements), leading to local structural collapse problem. To address this, inspired by Feature Integration Theory (FIT), we propose FITRep, the first attention-guided, white-box item representation framework for fine-grained item deduplication. FITRep consists of: (1) Concept Hierarchical Information Extraction (CHIE), using MLLMs to extract hierarchical semantic concepts; (2) Structure-Preserving Dimensionality Reduction (SPDR), an adaptive UMAP-based method for efficient information compression; and (3) FAISS-Based Clustering (FBC), a FAISS-based clustering that assigns each item a unique cluster id using FAISS. Deployed on Meituan's advertising system, FITRep achieves +3.60% CTR and +4.25% CPM gains in online A/B tests, demonstrating both effectiveness and real-world impact. 

---
# Anomaly Detection with Adaptive and Aggressive Rejection for Contaminated Training Data 

**Authors**: Jungi Lee, Jungkwon Kim, Chi Zhang, Kwangsun Yoo, Seok-Joo Byun  

**Link**: [PDF](https://arxiv.org/pdf/2511.21378)  

**Abstract**: Handling contaminated data poses a critical challenge in anomaly detection, as traditional models assume training on purely normal data. Conventional methods mitigate contamination by relying on fixed contamination ratios, but discrepancies between assumed and actual ratios can severely degrade performance, especially in noisy environments where normal and abnormal data distributions overlap. To address these limitations, we propose Adaptive and Aggressive Rejection (AAR), a novel method that dynamically excludes anomalies using a modified z-score and Gaussian mixture model-based thresholds. AAR effectively balances the trade-off between preserving normal data and excluding anomalies by integrating hard and soft rejection strategies. Extensive experiments on two image datasets and thirty tabular datasets demonstrate that AAR outperforms the state-of-the-art method by 0.041 AUROC. By providing a scalable and reliable solution, AAR enhances robustness against contaminated datasets, paving the way for broader real-world applications in domains such as security and healthcare. 

---
# The Directed Prediction Change - Efficient and Trustworthy Fidelity Assessment for Local Feature Attribution Methods 

**Authors**: Kevin Iselborn, David Dembinsky, Adriano Lucieri, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2511.21363)  

**Abstract**: The utility of an explanation method critically depends on its fidelity to the underlying machine learning model. Especially in high-stakes medical settings, clinicians and regulators require explanations that faithfully reflect the model's decision process. Existing fidelity metrics such as Infidelity rely on Monte Carlo approximation, which demands numerous model evaluations and introduces uncertainty due to random sampling. This work proposes a novel metric for evaluating the fidelity of local feature attribution methods by modifying the existing Prediction Change (PC) metric within the Guided Perturbation Experiment. By incorporating the direction of both perturbation and attribution, the proposed Directed Prediction Change (DPC) metric achieves an almost tenfold speedup and eliminates randomness, resulting in a deterministic and trustworthy evaluation procedure that measures the same property as local Infidelity. DPC is evaluated on two datasets (skin lesion images and financial tabular data), two black-box models, seven explanation algorithms, and a wide range of hyperparameters. Across $4\,744$ distinct explanations, the results demonstrate that DPC, together with PC, enables a holistic and computationally efficient evaluation of both baseline-oriented and local feature attribution methods, while providing deterministic and reproducible outcomes. 

---
# Hybrid-AIRL: Enhancing Inverse Reinforcement Learning with Supervised Expert Guidance 

**Authors**: Bram Silue, Santiago Amaya-Corredor, Patrick Mannion, Lander Willem, Pieter Libin  

**Link**: [PDF](https://arxiv.org/pdf/2511.21356)  

**Abstract**: Adversarial Inverse Reinforcement Learning (AIRL) has shown promise in addressing the sparse reward problem in reinforcement learning (RL) by inferring dense reward functions from expert demonstrations. However, its performance in highly complex, imperfect-information settings remains largely unexplored. To explore this gap, we evaluate AIRL in the context of Heads-Up Limit Hold'em (HULHE) poker, a domain characterized by sparse, delayed rewards and significant uncertainty. In this setting, we find that AIRL struggles to infer a sufficiently informative reward function. To overcome this limitation, we contribute Hybrid-AIRL (H-AIRL), an extension that enhances reward inference and policy learning by incorporating a supervised loss derived from expert data and a stochastic regularization mechanism. We evaluate H-AIRL on a carefully selected set of Gymnasium benchmarks and the HULHE poker setting. Additionally, we analyze the learned reward function through visualization to gain deeper insights into the learning process. Our experimental results show that H-AIRL achieves higher sample efficiency and more stable learning compared to AIRL. This highlights the benefits of incorporating supervised signals into inverse RL and establishes H-AIRL as a promising framework for tackling challenging, real-world settings. 

---
# Generating Separated Singing Vocals Using a Diffusion Model Conditioned on Music Mixtures 

**Authors**: Genís Plaja-Roglans, Yun-Ning Hung, Xavier Serra, Igor Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2511.21342)  

**Abstract**: Separating the individual elements in a musical mixture is an essential process for music analysis and practice. While this is generally addressed using neural networks optimized to mask or transform the time-frequency representation of a mixture to extract the target sources, the flexibility and generalization capabilities of generative diffusion models are giving rise to a novel class of solutions for this complicated task. In this work, we explore singing voice separation from real music recordings using a diffusion model which is trained to generate the solo vocals conditioned on the corresponding mixture. Our approach improves upon prior generative systems and achieves competitive objective scores against non-generative baselines when trained with supplementary data. The iterative nature of diffusion sampling enables the user to control the quality-efficiency trade-off, and also refine the output when needed. We present an ablation study of the sampling algorithm, highlighting the effects of the user-configurable parameters. 

---
# SurgMLLMBench: A Multimodal Large Language Model Benchmark Dataset for Surgical Scene Understanding 

**Authors**: Tae-Min Choi, Tae Kyeong Jeong, Garam Kim, Jaemin Lee, Yeongyoon Koh, In Cheul Choi, Jae-Ho Chung, Jong Woong Park, Juyoun Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.21339)  

**Abstract**: Recent advances in multimodal large language models (LLMs) have highlighted their potential for medical and surgical applications. However, existing surgical datasets predominantly adopt a Visual Question Answering (VQA) format with heterogeneous taxonomies and lack support for pixel-level segmentation, limiting consistent evaluation and applicability. We present SurgMLLMBench, a unified multimodal benchmark explicitly designed for developing and evaluating interactive multimodal LLMs for surgical scene understanding, including the newly collected Micro-surgical Artificial Vascular anastomosIS (MAVIS) dataset. It integrates pixel-level instrument segmentation masks and structured VQA annotations across laparoscopic, robot-assisted, and micro-surgical domains under a unified taxonomy, enabling comprehensive evaluation beyond traditional VQA tasks and richer visual-conversational interactions. Extensive baseline experiments show that a single model trained on SurgMLLMBench achieves consistent performance across domains and generalizes effectively to unseen datasets. SurgMLLMBench will be publicly released as a robust resource to advance multimodal surgical AI research, supporting reproducible evaluation and development of interactive surgical reasoning models. 

---
# Hybrid SIFT-SNN for Efficient Anomaly Detection of Traffic Flow-Control Infrastructure 

**Authors**: Munish Rathee, Boris Bačić, Maryam Doborjeh  

**Link**: [PDF](https://arxiv.org/pdf/2511.21337)  

**Abstract**: This paper presents the SIFT-SNN framework, a low-latency neuromorphic signal-processing pipeline for real-time detection of structural anomalies in transport infrastructure. The proposed approach integrates Scale-Invariant Feature Transform (SIFT) for spatial feature encoding with a latency-driven spike conversion layer and a Leaky Integrate-and-Fire (LIF) Spiking Neural Network (SNN) for classification. The Auckland Harbour Bridge dataset is recorded under various weather and lighting conditions, comprising 6,000 labelled frames that include both real and synthetically augmented unsafe cases. The presented system achieves a classification accuracy of 92.3% (+- 0.8%) with a per-frame inference time of 9.5 ms. Achieved sub-10 millisecond latency, combined with sparse spike activity (8.1%), enables real-time, low-power edge deployment. Unlike conventional CNN-based approaches, the hybrid SIFT-SNN pipeline explicitly preserves spatial feature grounding, enhances interpretability, supports transparent decision-making, and operates efficiently on embedded hardware. Although synthetic augmentation improved robustness, generalisation to unseen field conditions remains to be validated. The SIFT-SNN framework is validated through a working prototype deployed on a consumer-grade system and framed as a generalisable case study in structural safety monitoring for movable concrete barriers, which, as a traffic flow-control infrastructure, is deployed in over 20 cities worldwide. 

---
# The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment 

**Authors**: Stefanos Koutoupis, Michaela Areti Zervou, Konstantinos Kontras, Maarten De Vos, Panagiotis Tsakalides, Grigorios Tsagatakis  

**Link**: [PDF](https://arxiv.org/pdf/2511.21331)  

**Abstract**: Learning joint representations across multiple modalities remains a central challenge in multimodal machine learning. Prevailing approaches predominantly operate in pairwise settings, aligning two modalities at a time. While some recent methods aim to capture higher-order interactions among multiple modalities, they often overlook or insufficiently preserve pairwise relationships, limiting their effectiveness on single-modality tasks. In this work, we introduce Contrastive Fusion (ConFu), a framework that jointly embeds both individual modalities and their fused combinations into a unified representation space, where modalities and their fused counterparts are aligned. ConFu extends traditional pairwise contrastive objectives with an additional fused-modality contrastive term, encouraging the joint embedding of modality pairs with a third modality. This formulation enables ConFu to capture higher-order dependencies, such as XOR-like relationships, that cannot be recovered through pairwise alignment alone, while still maintaining strong pairwise correspondence. We evaluate ConFu on synthetic and real-world multimodal benchmarks, assessing its ability to exploit cross-modal complementarity, capture higher-order dependencies, and scale with increasing multimodal complexity. Across these settings, ConFu demonstrates competitive performance on retrieval and classification tasks, while supporting unified one-to-one and two-to-one retrieval within a single contrastive framework. 

---
# SONAR: Spectral-Contrastive Audio Residuals for Generalizable Deepfake Detection 

**Authors**: Ido Nitzan HIdekel, Gal lifshitz, Khen Cohen, Dan Raviv  

**Link**: [PDF](https://arxiv.org/pdf/2511.21325)  

**Abstract**: Deepfake (DF) audio detectors still struggle to generalize to out of distribution inputs. A central reason is spectral bias, the tendency of neural networks to learn low-frequency structure before high-frequency (HF) details, which both causes DF generators to leave HF artifacts and leaves those same artifacts under-exploited by common detectors. To address this gap, we propose Spectral-cONtrastive Audio Residuals (SONAR), a frequency-guided framework that explicitly disentangles an audio signal into complementary representations. An XLSR encoder captures the dominant low-frequency content, while the same cloned path, preceded by learnable SRM, value-constrained high-pass filters, distills faint HF residuals. Frequency cross-attention reunites the two views for long- and short-range frequency dependencies, and a frequency-aware Jensen-Shannon contrastive loss pulls real content-noise pairs together while pushing fake embeddings apart, accelerating optimization and sharpening decision boundaries. Evaluated on the ASVspoof 2021 and in-the-wild benchmarks, SONAR attains state-of-the-art performance and converges four times faster than strong baselines. By elevating faint high-frequency residuals to first-class learning signals, SONAR unveils a fully data-driven, frequency-guided contrastive framework that splits the latent space into two disjoint manifolds: natural-HF for genuine audio and distorted-HF for synthetic audio, thereby sharpening decision boundaries. Because the scheme operates purely at the representation level, it is architecture-agnostic and, in future work, can be seamlessly integrated into any model or modality where subtle high-frequency cues are decisive. 

---
# TALES: A Taxonomy and Analysis of Cultural Representations in LLM-generated Stories 

**Authors**: Kirti Bhagat, Shaily Bhatt, Athul Velagapudi, Aditya Vashistha, Shachi Dave, Danish Pruthi  

**Link**: [PDF](https://arxiv.org/pdf/2511.21322)  

**Abstract**: Millions of users across the globe turn to AI chatbots for their creative needs, inviting widespread interest in understanding how such chatbots represent diverse cultures. At the same time, evaluating cultural representations in open-ended tasks remains challenging and underexplored. In this work, we present TALES, an evaluation of cultural misrepresentations in LLM-generated stories for diverse Indian cultural identities. First, we develop TALES-Tax, a taxonomy of cultural misrepresentations by collating insights from participants with lived experiences in India through focus groups (N=9) and individual surveys (N=15). Using TALES-Tax, we evaluate 6 models through a large-scale annotation study spanning 2,925 annotations from 108 annotators with lived cultural experience from across 71 regions in India and 14 languages. Concerningly, we find that 88\% of the generated stories contain one or more cultural inaccuracies, and such errors are more prevalent in mid- and low-resourced languages and stories based in peri-urban regions in India. Lastly, we transform the annotations into TALES-QA, a standalone question bank to evaluate the cultural knowledge of foundational models. Through this evaluation, we surprisingly discover that models often possess the requisite cultural knowledge despite generating stories rife with cultural misrepresentations. 

---
# Improvement of Collision Avoidance in Cut-In Maneuvers Using Time-to-Collision Metrics 

**Authors**: Jamal Raiyn  

**Link**: [PDF](https://arxiv.org/pdf/2511.21280)  

**Abstract**: This paper proposes a new strategy for collision avoidance system leveraging Time-to-Collision (TTC) metrics for handling cut-in scenarios, which are particularly challenging for autonomous vehicles (AVs). By integrating a deep learning with TTC calculations, the system predicts potential collisions and determines appropriate evasive actions compared to traditional TTC -based approaches. 

---
# Self-Guided Defense: Adaptive Safety Alignment for Reasoning Models via Synthesized Guidelines 

**Authors**: Yuhang Wang, Yanxu Zhu, Dongyuan Lu, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21214)  

**Abstract**: Reasoning models have demonstrated remarkable capabilities in complex reasoning tasks. However, ensuring their safety against adversarial jailbreak prompts remains a critical challenge. Due to the covert and deceptive nature of such prompts, they can often evade built-in safety mechanisms and lead to the generation of harmful content. This underscores the need for an adaptive safety alignment approach that enables models to autonomously reinforce their defenses in response to adversarial inputs. This paper introduces the Synthesized Guideline-based Adaptive Safety Alignment (SGASA) framework, which internalizes model-generated safety guidelines to strengthen models' ability to enhance robustness against harmful adversarial prompts while minimizing unnecessary refusals of benign requests. SGASA consists of two key stages: Data Pre-synthesis, which generates safety guidelines and augmented prompts; and Alignment Fine-tuning, which leverages Supervised Fine-tuning (SFT) and Direct Preference Optimization (DPO) to embed these guidelines into the model. Extensive experiments across multiple datasets demonstrate that SGASA significantly improves model safety, validating its adaptive and scalable effectiveness. 

---
# BotaCLIP: Contrastive Learning for Botany-Aware Representation of Earth Observation Data 

**Authors**: Selene Cerna, Sara Si-Moussi, Wilfried Thuiller, Hadrien Hendrikx, Vincent Miele  

**Link**: [PDF](https://arxiv.org/pdf/2511.21194)  

**Abstract**: Foundation models have demonstrated a remarkable ability to learn rich, transferable representations across diverse modalities such as images, text, and audio. In modern machine learning pipelines, these representations often replace raw data as the primary input for downstream tasks. In this paper, we address the challenge of adapting a pre-trained foundation model to inject domain-specific knowledge, without retraining from scratch or incurring significant computational costs. To this end, we introduce BotaCLIP, a lightweight multimodal contrastive framework that adapts a pre-trained Earth Observation foundation model (DOFA) by aligning high-resolution aerial imagery with botanical relevés. Unlike generic embeddings, BotaCLIP internalizes ecological structure through contrastive learning with a regularization strategy that mitigates catastrophic forgetting. Once trained, the resulting embeddings serve as transferable representations for downstream predictors. Motivated by real-world applications in biodiversity modeling, we evaluated BotaCLIP representations in three ecological tasks: plant presence prediction, butterfly occurrence modeling, and soil trophic group abundance estimation. The results showed consistent improvements over those derived from DOFA and supervised baselines. More broadly, this work illustrates how domain-aware adaptation of foundation models can inject expert knowledge into data-scarce settings, enabling frugal representation learning. 

---
# When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models 

**Authors**: Hui Lu, Yi Yu, Yiming Yang, Chenyu Yi, Qixin Zhang, Bingquan Shen, Alex C. Kot, Xudong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21192)  

**Abstract**: Vision-Language-Action (VLA) models are vulnerable to adversarial attacks, yet universal and transferable attacks remain underexplored, as most existing patches overfit to a single model and fail in black-box settings. To address this gap, we present a systematic study of universal, transferable adversarial patches against VLA-driven robots under unknown architectures, finetuned variants, and sim-to-real shifts. We introduce UPA-RFAS (Universal Patch Attack via Robust Feature, Attention, and Semantics), a unified framework that learns a single physical patch in a shared feature space while promoting cross-model transfer. UPA-RFAS combines (i) a feature-space objective with an $\ell_1$ deviation prior and repulsive InfoNCE loss to induce transferable representation shifts, (ii) a robustness-augmented two-phase min-max procedure where an inner loop learns invisible sample-wise perturbations and an outer loop optimizes the universal patch against this hardened neighborhood, and (iii) two VLA-specific losses: Patch Attention Dominance to hijack text$\to$vision attention and Patch Semantic Misalignment to induce image-text mismatch without labels. Experiments across diverse VLA models, manipulation suites, and physical executions show that UPA-RFAS consistently transfers across models, tasks, and viewpoints, exposing a practical patch-based attack surface and establishing a strong baseline for future defenses. 

---
# Progress by Pieces: Test-Time Scaling for Autoregressive Image Generation 

**Authors**: Joonhyung Park, Hyeongwon Jang, Joowon Kim, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21185)  

**Abstract**: Recent visual autoregressive (AR) models have shown promising capabilities in text-to-image generation, operating in a manner similar to large language models. While test-time computation scaling has brought remarkable success in enabling reasoning-enhanced outputs for challenging natural language tasks, its adaptation to visual AR models remains unexplored and poses unique challenges. Naively applying test-time scaling strategies such as Best-of-N can be suboptimal: they consume full-length computation on erroneous generation trajectories, while the raster-scan decoding scheme lacks a blueprint of the entire canvas, limiting scaling benefits as only a few prompt-aligned candidates are generated. To address these, we introduce GridAR, a test-time scaling framework designed to elicit the best possible results from visual AR models. GridAR employs a grid-partitioned progressive generation scheme in which multiple partial candidates for the same position are generated within a canvas, infeasible ones are pruned early, and viable ones are fixed as anchors to guide subsequent decoding. Coupled with this, we present a layout-specified prompt reformulation strategy that inspects partial views to infer a feasible layout for satisfying the prompt. The reformulated prompt then guides subsequent image generation to mitigate the blueprint deficiency. Together, GridAR achieves higher-quality results under limited test-time scaling: with N=4, it even outperforms Best-of-N (N=8) by 14.4% on T2I-CompBench++ while reducing cost by 25.6%. It also generalizes to autoregressive image editing, showing comparable edit quality and a 13.9% gain in semantic preservation on PIE-Bench over larger-N baselines. 

---
# Privacy in Federated Learning with Spiking Neural Networks 

**Authors**: Dogukan Aksu, Jesus Martinez del Rincon, Ihsen Alouani  

**Link**: [PDF](https://arxiv.org/pdf/2511.21181)  

**Abstract**: Spiking neural networks (SNNs) have emerged as prominent candidates for embedded and edge AI. Their inherent low power consumption makes them far more efficient than conventional ANNs in scenarios where energy budgets are tightly constrained. In parallel, federated learning (FL) has become the prevailing training paradigm in such settings, enabling on-device learning while limiting the exposure of raw data. However, gradient inversion attacks represent a critical privacy threat in FL, where sensitive training data can be reconstructed directly from shared gradients. While this vulnerability has been widely investigated in conventional ANNs, its implications for SNNs remain largely unexplored. In this work, we present the first comprehensive empirical study of gradient leakage in SNNs across diverse data domains. SNNs are inherently non-differentiable and are typically trained using surrogate gradients, which we hypothesized would be less correlated with the original input and thus less informative from a privacy perspective. To investigate this, we adapt different gradient leakage attacks to the spike domain. Our experiments reveal a striking contrast with conventional ANNs: whereas ANN gradients reliably expose salient input content, SNN gradients yield noisy, temporally inconsistent reconstructions that fail to recover meaningful spatial or temporal structure. These results indicate that the combination of event-driven dynamics and surrogate-gradient training substantially reduces gradient informativeness. To the best of our knowledge, this work provides the first systematic benchmark of gradient inversion attacks for spiking architectures, highlighting the inherent privacy-preserving potential of neuromorphic computation. 

---
# CAHS-Attack: CLIP-Aware Heuristic Search Attack Method for Stable Diffusion 

**Authors**: Shuhan Xia, Jing Dai, Hui Ouyang, Yadong Shang, Dongxiao Zhao, Peipei Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.21180)  

**Abstract**: Diffusion models exhibit notable fragility when faced with adversarial prompts, and strengthening attack capabilities is crucial for uncovering such vulnerabilities and building more robust generative systems. Existing works often rely on white-box access to model gradients or hand-crafted prompt engineering, which is infeasible in real-world deployments due to restricted access or poor attack effect. In this paper, we propose CAHS-Attack , a CLIP-Aware Heuristic Search attack method. CAHS-Attack integrates Monte Carlo Tree Search (MCTS) to perform fine-grained suffix optimization, leveraging a constrained genetic algorithm to preselect high-potential adversarial prompts as root nodes, and retaining the most semantically disruptive outcome at each simulation rollout for efficient local search. Extensive experiments demonstrate that our method achieves state-of-the-art attack performance across both short and long prompts of varying semantics. Furthermore, we find that the fragility of SD models can be attributed to the inherent vulnerability of their CLIP-based text encoders, suggesting a fundamental security risk in current text-to-image pipelines. 

---
# LLaVA-UHD v3: Progressive Visual Compression for Efficient Native-Resolution Encoding in MLLMs 

**Authors**: Shichu Sun, Yichen Zhang, Haolin Song, Zonghao Guo, Chi Chen, Yidan Zhang, Yuan Yao, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.21150)  

**Abstract**: Visual encoding followed by token condensing has become the standard architectural paradigm in multi-modal large language models (MLLMs). Many recent MLLMs increasingly favor global native- resolution visual encoding over slice-based methods. To investigate this trend, we systematically compare their behavior on vision-language understanding and attention patterns, revealing that global encoding enhances overall capability but at the expense of greater computational overhead. To address this issue, we present LLaVA-UHD v3, an MLLM centered upon our proposed Progressive Visual Compression (PVC) method, which can be seamlessly integrated into standard Vision Transformer (ViT) to enable efficient native-resolution encoding. The PVC approach consists of two key modules: (i) refined patch embedding, which supports flexible patch-size scaling for fine-grained visual model- ing, (ii) windowed token compression, hierarchically deployed across ViT layers to progressively aggregate local token representations. Jointly modulated by these two modules, a widely pretrained ViT can be reconfigured into an efficient architecture while largely preserving generality. Evaluated across extensive benchmarks, the transformed ViT, termed ViT-UHD, demonstrates competitive performance with MoonViT while reducing TTFT (time-to-first-token) by 2.4x, when developed within an identical MLLM architecture. Building upon ViT-UHD, LLaVA-UHD v3 also achieves competitive performance to Qwen2-VL, while further reducing TTFT by 1.9x. We will release all code and checkpoints to support future research on efficient MLLMs. 

---
# Maglev-Pentabot: Magnetic Levitation System for Non-Contact Manipulation using Deep Reinforcement Learning 

**Authors**: Guoming Huang, Qingyi Zhou, Dianjing Liu, Shuai Zhang, Ming Zhou, Zongfu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.21149)  

**Abstract**: Non-contact manipulation has emerged as a transformative approach across various industrial fields. However, current flexible 2D and 3D non-contact manipulation techniques are often limited to microscopic scales, typically controlling objects in the milligram range. In this paper, we present a magnetic levitation system, termed Maglev-Pentabot, designed to address this limitation. The Maglev-Pentabot leverages deep reinforcement learning (DRL) to develop complex control strategies for manipulating objects in the gram range. Specifically, we propose an electromagnet arrangement optimized through numerical analysis to maximize controllable space. Additionally, an action remapping method is introduced to address sample sparsity issues caused by the strong nonlinearity in magnetic field intensity, hence allowing the DRL controller to converge. Experimental results demonstrate flexible manipulation capabilities, and notably, our system can generalize to transport tasks it has not been explicitly trained for. Furthermore, our approach can be scaled to manipulate heavier objects using larger electromagnets, offering a reference framework for industrial-scale robotic applications. 

---
# Efficient Training for Human Video Generation with Entropy-Guided Prioritized Progressive Learning 

**Authors**: Changlin Li, Jiawei Zhang, Shuhao Liu, Sihao Lin, Zeyi Shi, Zhihui Li, Xiaojun Chang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21136)  

**Abstract**: Human video generation has advanced rapidly with the development of diffusion models, but the high computational cost and substantial memory consumption associated with training these models on high-resolution, multi-frame data pose significant challenges. In this paper, we propose Entropy-Guided Prioritized Progressive Learning (Ent-Prog), an efficient training framework tailored for diffusion models on human video generation. First, we introduce Conditional Entropy Inflation (CEI) to assess the importance of different model components on the target conditional generation task, enabling prioritized training of the most critical components. Second, we introduce an adaptive progressive schedule that adaptively increases computational complexity during training by measuring the convergence efficiency. Ent-Prog reduces both training time and GPU memory consumption while maintaining model performance. Extensive experiments across three datasets, demonstrate the effectiveness of Ent-Prog, achieving up to 2.2$\times$ training speedup and 2.4$\times$ GPU memory reduction without compromising generative performance. 

---
# SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation 

**Authors**: Ziyi Chen, Yingnan Guo, Zedong Chu, Minghua Luo, Yanfen Shen, Mingchao Sun, Junjun Hu, Shichao Xie, Kuan Yang, Pei Shi, Zhining Gu, Lu Liu, Honglin Han, Xiaolong Wu, Mu Xu, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21135)  

**Abstract**: Embodied navigation that adheres to social norms remains an open research challenge. Our \textbf{SocialNav} is a foundational model for socially-aware navigation with a hierarchical "brain-action" architecture, capable of understanding high-level social norms and generating low-level, socially compliant trajectories. To enable such dual capabilities, we construct the SocNav Dataset, a large-scale collection of 7 million samples, comprising (1) a Cognitive Activation Dataset providing social reasoning signals such as chain-of-thought explanations and social traversability prediction, and (2) an Expert Trajectories Pyramid aggregating diverse navigation demonstrations from internet videos, simulated environments, and real-world robots. A multi-stage training pipeline is proposed to gradually inject and refine navigation intelligence: we first inject general navigation skills and social norms understanding into the model via imitation learning, and then refine such skills through a deliberately designed Socially-Aware Flow Exploration GRPO (SAFE-GRPO), the first flow-based reinforcement learning framework for embodied navigation that explicitly rewards socially compliant behaviors. SocialNav achieves +38% success rate and +46% social compliance rate compared to the state-of-the-art method, demonstrating strong gains in both navigation performance and social compliance. Our project page: this https URL 

---
# Which Layer Causes Distribution Deviation? Entropy-Guided Adaptive Pruning for Diffusion and Flow Models 

**Authors**: Changlin Li, Jiawei Zhang, Zeyi Shi, Zongxin Yang, Zhihui Li, Xiaojun Chang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21122)  

**Abstract**: Large-scale vision generative models, including diffusion and flow models, have demonstrated remarkable performance in visual generation tasks. However, transferring these pre-trained models to downstream tasks often results in significant parameter redundancy. In this paper, we propose EntPruner, an entropy-guided automatic progressive pruning framework for diffusion and flow models. First, we introduce entropy-guided pruning, a block-level importance assessment strategy specifically designed for generative models. Unlike discriminative models, generative models require preserving the diversity and condition-fidelity of the output distribution. As the importance of each module can vary significantly across downstream tasks, EntPruner prioritizes pruning of less important blocks using data-dependent Conditional Entropy Deviation (CED) as a guiding metric. CED quantifies how much the distribution diverges from the learned conditional data distribution after removing a block. Second, we propose a zero-shot adaptive pruning framework to automatically determine when and how much to prune during training. This dynamic strategy avoids the pitfalls of one-shot pruning, mitigating mode collapse, and preserving model performance. Extensive experiments on DiT and SiT models demonstrate the effectiveness of EntPruner, achieving up to 2.22$\times$ inference speedup while maintaining competitive generation quality on ImageNet and three downstream datasets. 

---
# Beyond Patch Aggregation: 3-Pass Pyramid Indexing for Vision-Enhanced Document Retrieval 

**Authors**: Anup Roy, Rishabh Gyanendra Upadhyay, Animesh Rameshbhai Panara, Robin Mills  

**Link**: [PDF](https://arxiv.org/pdf/2511.21121)  

**Abstract**: Document centric RAG pipelines usually begin with OCR, followed by brittle heuristics for chunking, table parsing, and layout reconstruction. These text first workflows are costly to maintain, sensitive to small layout shifts, and often lose the spatial cues that contain the answer. Vision first retrieval has emerged as a strong alternative. By operating directly on page images, systems like ColPali and ColQwen preserve structure and reduce pipeline complexity while achieving strong benchmark performance. However, these late interaction models tie retrieval to a specific vision backbone and require storing hundreds of patch embeddings per page, creating high memory overhead and complicating large scale deployment.
We introduce VisionRAG, a multimodal retrieval system that is OCR free and model agnostic. VisionRAG indexes documents directly as images, preserving layout, tables, and spatial cues, and builds semantic vectors without committing to a specific extraction. Our three pass pyramid indexing framework creates vectors using global page summaries, section headers, visual hotspots, and fact level cues. These summaries act as lightweight retrieval surrogates. At query time, VisionRAG retrieves the most relevant pages using the pyramid index, then forwards the raw page image encoded as base64 to a multimodal LLM for final question answering. During retrieval, reciprocal rank fusion integrates signals across the pyramid to produce robust ranking.
VisionRAG stores only 17 to 27 vectors per page, matching the efficiency of patch based methods while staying flexible across multimodal encoders. On financial document benchmarks, it achieves 0.8051 accuracy at 10 on FinanceBench and 0.9629 recall at 100 on TAT DQA. These results show that OCR free, summary guided multimodal retrieval is a practical and scalable alternative to traditional text extraction pipelines. 

---
# Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling 

**Authors**: Mengran Li, Zelin Zang, Wenbin Xing, Junzhou Chen, Ronghui Zhang, Jiebo Luo, Stan Z. Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.21120)  

**Abstract**: Understanding how chemical perturbations propagate through biological systems is essential for robust molecular property prediction. While most existing methods focus on chemical structures alone, recent advances highlight the crucial role of cellular responses such as morphology and gene expression in shaping drug effects. However, current cell-aware approaches face two key limitations: (1) modality incompleteness in external biological data, and (2) insufficient modeling of hierarchical dependencies across molecular, cellular, and genomic levels. We propose CHMR (Cell-aware Hierarchical Multi-modal Representations), a robust framework that jointly models local-global dependencies between molecules and cellular responses and captures latent biological hierarchies via a novel tree-structured vector quantization module. Evaluated on nine public benchmarks spanning 728 tasks, CHMR outperforms state-of-the-art baselines, yielding average improvements of 3.6% on classification and 17.2% on regression tasks. These results demonstrate the advantage of hierarchy-aware, multimodal learning for reliable and biologically grounded molecular representations, offering a generalizable framework for integrative biomedical modeling. The code is in this https URL. 

---
# Deformation-aware Temporal Generation for Early Prediction of Alzheimers Disease 

**Authors**: Xin Honga, Jie Lin, Minghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21114)  

**Abstract**: Alzheimer's disease (AD), a degenerative brain condition, can benefit from early prediction to slow its progression. As the disease progresses, patients typically undergo brain atrophy. Current prediction methods for Alzheimers disease largely involve analyzing morphological changes in brain images through manual feature extraction. This paper proposes a novel method, the Deformation-Aware Temporal Generative Network (DATGN), to automate the learning of morphological changes in brain images about disease progression for early prediction. Given the common occurrence of missing data in the temporal sequences of MRI images, DATGN initially interpolates incomplete sequences. Subsequently, a bidirectional temporal deformation-aware module guides the network in generating future MRI images that adhere to the disease's progression, facilitating early prediction of Alzheimer's disease. DATGN was tested for the generation of temporal sequences of future MRI images using the ADNI dataset, and the experimental results are competitive in terms of PSNR and MMSE image quality metrics. Furthermore, when DATGN-generated synthetic data was integrated into the SVM vs. CNN vs. 3DCNN-based classification methods, significant improvements were achieved from 6. 21\% to 16\% in AD vs. NC classification accuracy and from 7. 34\% to 21. 25\% in AD vs. MCI vs. NC classification accuracy. The qualitative visualization results indicate that DATGN produces MRI images consistent with the brain atrophy trend in Alzheimer's disease, enabling early disease prediction. 

---
# Dynamic Stratified Contrastive Learning with Upstream Augmentation for MILP Branching 

**Authors**: Tongkai Lu, Shuai Ma, Chongyang Tao  

**Link**: [PDF](https://arxiv.org/pdf/2511.21107)  

**Abstract**: Mixed Integer Linear Programming (MILP) is a fundamental class of NP-hard problems that has garnered significant attention from both academia and industry. The Branch-and-Bound (B\&B) method is the dominant approach for solving MILPs and the branching plays an important role in B\&B methods. Neural-based learning frameworks have recently been developed to enhance branching policies and the efficiency of solving MILPs. However, these methods still struggle with semantic variation across depths, the scarcity of upstream nodes, and the costly collection of strong branching samples. To address these issues, we propose \ours, a Dynamic \underline{\textbf{S}}tratified \underline{\textbf{C}}ontrastive Training Framework for \underline{\textbf{MILP}} Branching. It groups branch-and-bound nodes based on their feature distributions and trains a GCNN-based discriminative model to progressively separate nodes across groups, learning finer-grained node representations throughout the tree. To address data scarcity and imbalance at upstream nodes, we introduce an upstream-augmented MILP derivation procedure that generates both theoretically equivalent and perturbed instances. \ours~effectively models subtle semantic differences between nodes, significantly enhancing branching accuracy and solving efficiency, particularly for upstream nodes. Extensive experiments on standard MILP benchmarks demonstrate that our method enhances branching accuracy, reduces solving time, and generalizes effectively to unseen instances. 

---
# From Bits to Rounds: Parallel Decoding with Exploration for Diffusion Language Models 

**Authors**: Hengyu Fu, Baihe Huang, Virginia Adams, Charles Wang, Venkat Srinivasan, Jiantao Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2511.21103)  

**Abstract**: Diffusion Language Models (DLMs) have recently emerged as a strong alternative to autoregressive language models (LMs). DLMs offer comparable accuracy with faster inference speed via parallel decoding. However, standard DLM decoding strategies relying on high-confidence tokens encounter an inherent information-theoretic bottleneck that restricts decoding progress and ultimately slows generation. We demonstrate both theoretically and empirically that prioritizing high-confidence tokens is inherently inefficient. High-probability tokens carry negligible information and strictly relying on them limits the effective progress made in each decoding round. We prove that the number of decoding rounds must grow linearly with the sample's total information (negative log-likelihood) and inversely with the per-round information budget, establishing a bits-to-rounds principle. We also propose Explore-Then-Exploit (ETE), a training-free decoding strategy that maximizes information throughput and decoding efficiency. ETE combines cross-block decoding with targeted exploration of high-uncertainty tokens to reshape the conditional distribution and trigger cascades of confident predictions. Experiments verify our theoretical bounds and demonstrate that ETE consistently reduces the required number of decoding rounds compared to confidence-only baselines without compromising generation quality. 

---
# Pygmalion Effect in Vision: Image-to-Clay Translation for Reflective Geometry Reconstruction 

**Authors**: Gayoung Lee, Junho Kim, Jin-Hwa Kim, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.21098)  

**Abstract**: Understanding reflection remains a long-standing challenge in 3D reconstruction due to the entanglement of appearance and geometry under view-dependent reflections. In this work, we present the Pygmalion Effect in Vision, a novel framework that metaphorically "sculpts" reflective objects into clay-like forms through image-to-clay translation. Inspired by the myth of Pygmalion, our method learns to suppress specular cues while preserving intrinsic geometric consistency, enabling robust reconstruction from multi-view images containing complex reflections. Specifically, we introduce a dual-branch network in which a BRDF-based reflective branch is complemented by a clay-guided branch that stabilizes geometry and refines surface normals. The two branches are trained jointly using the synthesized clay-like images, which provide a neutral, reflection-free supervision signal that complements the reflective views. Experiments on both synthetic and real datasets demonstrate substantial improvement in normal accuracy and mesh completeness over existing reflection-handling methods. Beyond technical gains, our framework reveals that seeing by unshining, translating radiance into neutrality, can serve as a powerful inductive bias for reflective object geometry learning. 

---
# MNM : Multi-level Neuroimaging Meta-analysis with Hyperbolic Brain-Text Representations 

**Authors**: Seunghun Baek, Jaejin Lee, Jaeyoon Sim, Minjae Jeong, Won Hwa Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.21092)  

**Abstract**: Various neuroimaging studies suffer from small sample size problem which often limit their reliability. Meta-analysis addresses this challenge by aggregating findings from different studies to identify consistent patterns of brain activity. However, traditional approaches based on keyword retrieval or linear mappings often overlook the rich hierarchical structure in the brain. In this work, we propose a novel framework that leverages hyperbolic geometry to bridge the gap between neuroscience literature and brain activation maps. By embedding text from research articles and corresponding brain images into a shared hyperbolic space via the Lorentz model, our method captures both semantic similarity and hierarchical organization inherent in neuroimaging data. In the hyperbolic space, our method performs multi-level neuroimaging meta-analysis (MNM) by 1) aligning brain and text embeddings for semantic correspondence, 2) guiding hierarchy between text and brain activations, and 3) preserving the hierarchical relationships within brain activation patterns. Experimental results demonstrate that our model outperforms baselines, offering a robust and interpretable paradigm of multi-level neuroimaging meta-analysis via hyperbolic brain-text representation. 

---
# MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts 

**Authors**: Ivan Novikov  

**Link**: [PDF](https://arxiv.org/pdf/2511.21089)  

**Abstract**: Large Language Models (LLMs) are predominantly deployed as dense transformers, where every parameter in every feed-forward block is activated for every token. While architecturally simple, this is computationally inefficient, since inference costs scale linearly with parameter count. Recent upcycling methods such as MoEfication, CMoE, ToMoE, and MoORE reveal that much of the useful computation lives in sparse, semi-modular substructures inside dense feed-forward networks, but these approaches typically rely on clustering, activation profiling, singular value decomposition, or custom routing that requires calibration data. This paper introduces MLPMoE (MLP Mixture-of-Experts), a training-free, deterministic transformation that restructures the dense MLP in transformer blocks into a static, high-cardinality mixture of experts. The transformation uses simple tensor slicing and summation, reinterpreting the algebra of tensor parallelism as a topological conversion rather than a distributed training pattern. We further introduce Fractal Fade (differential branch sparsity) and Compensated Pruning (variance-preserving branch reduction) as lightweight mechanisms for structured sparsity. On Qwen2.5-0.5B-Instruct and DeepSeek-R1-Distill-Llama-8B, the zero-shot MLPMoE transform changes a proxy perplexity metric by less than 0.05 percent while keeping the parameter count effectively constant. On the 8B model, differential sparsity removes about 20 percent of MLP parameters while keeping perplexity within about 2 percent of the dense baseline. The method operates entirely post hoc on existing checkpoints and does not require gradients, calibration sets, or router training. Code is available at this https URL 

---
# Enhancing Burmese News Classification with Kolmogorov-Arnold Network Head Fine-tuning 

**Authors**: Thura Aung, Eaint Kay Khaing Kyaw, Ye Kyaw Thu, Thazin Myint Oo, Thepchai Supnithi  

**Link**: [PDF](https://arxiv.org/pdf/2511.21081)  

**Abstract**: In low-resource languages like Burmese, classification tasks often fine-tune only the final classification layer, keeping pre-trained encoder weights frozen. While Multi-Layer Perceptrons (MLPs) are commonly used, their fixed non-linearity can limit expressiveness and increase computational cost. This work explores Kolmogorov-Arnold Networks (KANs) as alternative classification heads, evaluating Fourier-based FourierKAN, Spline-based EfficientKAN, and Grid-based FasterKAN-across diverse embeddings including TF-IDF, fastText, and multilingual transformers (mBERT, Distil-mBERT). Experimental results show that KAN-based heads are competitive with or superior to MLPs. EfficientKAN with fastText achieved the highest F1-score (0.928), while FasterKAN offered the best trade-off between speed and accuracy. On transformer embeddings, EfficientKAN matched or slightly outperformed MLPs with mBERT (0.917 F1). These findings highlight KANs as expressive, efficient alternatives to MLPs for low-resource language classification. 

---
# Data-Driven Assessment of Concrete Slab Integrity via Impact-Echo Signals and Neural Networks 

**Authors**: Yeswanth Ravichandran, Duoduo Liao, Charan Teja Kurakula  

**Link**: [PDF](https://arxiv.org/pdf/2511.21080)  

**Abstract**: Subsurface defects such as delamination, voids, and honeycombing critically affect the durability of concrete bridge decks but are difficult to detect reliably using visual inspection or manual sounding. This paper presents a machine learning based Impact Echo (IE) framework that automates both defect localization and multi-class classification of common concrete defects. Raw IE signals from Federal Highway Administration (FHWA) laboratory slabs and in-service bridge decks are transformed via Fast Fourier Transform (FFT) into dominant peak-frequency features and interpolated into spatial maps for defect zone visualization. Unsupervised k-means clustering highlights low-frequency, defect-prone regions, while Ground Truth Masks (GTMs) derived from seeded lab defects are used to validate spatial accuracy and generate high-confidence training labels. From these validated regions, spatially ordered peak-frequency sequences are constructed and fed into a stacked Long Short-Term Memory (LSTM) network that classifies four defect types shallow delamination, deep delamination, voids, and honeycombing with 73% overall accuracy. Field validation on the bridge deck demonstrates that models trained on laboratory data generalize under realistic coupling, noise, and environmental variability. The proposed framework enhances the objectivity, scalability, and repeatability of Non-Destructive Evaluation (NDE), supporting intelligent, data-driven bridge health monitoring at a network scale. 

---
# Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning 

**Authors**: Zhenchao Tang, Fang Wang, Haohuai He, Jiale Zhou, Tianxu Lv, Jun Zhu, Shouzhi Chen, Minghao Yang, Yu Wang, Jiayang Wu, Yidong Song, Jianhua Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.21075)  

**Abstract**: Effective post-training is essential to align Large Language Models (LLMs) with specialized biomedical knowledge to accelerate life science research. However, current approaches face significant limitations. First, biomedical reasoning involves intricate mechanisms often represented by sparse textual data. Standard Supervised Fine-Tuning (SFT) tends to overfit to surface-level instruction patterns without effectively internalizing this fragmented scientific knowledge. Second, Reinforcement Learning (RL) is impractical for this domain, as defining meaningful rewards often necessitates prohibitive experimental validation (e.g., wet-lab verification of drug responses), rendering real-time feedback unfeasible. We propose Balanced Fine-Tuning (BFT), an efficient post-training method designed to learn complex reasoning from sparse data without external reward signals. BFT operates through a two-layer weighting mechanism: 1. At the token level, it scales loss via prediction probabilities to stabilize gradients and prevent overfitting; 2. At the sample level, it uses "minimum group confidence" to adaptively enhance the learning of hard samples. Experiments demonstrate that BFT significantly outperforms SFT. In medical tasks, it enables LLMs to acquire knowledge that SFT misses. In biological tasks, BFT-based LLMs surpass GeneAgent (an accurate agent for biology analysis) in biological process reasoning. Moreover, the text embeddings generated by BFT can be directly applied to downstream tasks, such as gene interaction and single-cell perturbation response prediction. These results indicate that BFT facilitates broad applications of LLMs in biomedical research. 

---
# Context-Aware Pragmatic Metacognitive Prompting for Sarcasm Detection 

**Authors**: Michael Iskandardinata, William Christian, Derwin Suhartono  

**Link**: [PDF](https://arxiv.org/pdf/2511.21066)  

**Abstract**: Detecting sarcasm remains a challenging task in the areas of Natural Language Processing (NLP) despite recent advances in neural network approaches. Currently, Pre-trained Language Models (PLMs) and Large Language Models (LLMs) are the preferred approach for sarcasm detection. However, the complexity of sarcastic text, combined with linguistic diversity and cultural variation across communities, has made the task more difficult even for PLMs and LLMs. Beyond that, those models also exhibit unreliable detection of words or tokens that require extra grounding for analysis. Building on a state-of-the-art prompting method in LLMs for sarcasm detection called Pragmatic Metacognitive Prompting (PMP), we introduce a retrieval-aware approach that incorporates retrieved contextual information for each target text. Our pipeline explores two complementary ways to provide context: adding non-parametric knowledge using web-based retrieval when the model lacks necessary background, and eliciting the model's own internal knowledge for a self-knowledge awareness strategy. We evaluated our approach with three datasets, such as Twitter Indonesia Sarcastic, SemEval-2018 Task 3, and MUStARD. Non-parametric retrieval resulted in a significant 9.87% macro-F1 improvement on Twitter Indonesia Sarcastic compared to the original PMP method. Self-knowledge retrieval improves macro-F1 by 3.29% on Semeval and by 4.08% on MUStARD. These findings highlight the importance of context in enhancing LLMs performance in sarcasm detection task, particularly the involvement of culturally specific slang, references, or unknown terms to the LLMs. Future work will focus on optimizing the retrieval of relevant contextual information and examining how retrieval quality affects performance. The experiment code is available at: this https URL. 

---
# Breaking the Safety-Capability Tradeoff: Reinforcement Learning with Verifiable Rewards Maintains Safety Guardrails in LLMs 

**Authors**: Dongkyu Derek Cho, Huan Song, Arijit Ghosh Chowdhury, Haotian An, Yawei Wang, Rohit Thekkanal, Negin Sokhandan, Sharlina Keshava, Hannah Marlowe  

**Link**: [PDF](https://arxiv.org/pdf/2511.21050)  

**Abstract**: Fine-tuning large language models (LLMs) for downstream tasks typically exhibit a fundamental safety-capability tradeoff, where improving task performance degrades safety alignment even on benign datasets. This degradation persists across standard approaches including supervised finetuning (SFT) and reinforcement learning from human feedback (RLHF). While reinforcement learning with verifiable rewards (RLVR) has emerged as a promising alternative that optimizes models on objectively measurable tasks, its safety implications remain unexplored. We present the first comprehensive theoretical and empirical analysis of safety properties in RLVR. Theoretically, we derive upper bounds on safety drift under KL-constrained optimization and prove conditions under which safety degradation is eliminated. Empirically, we conduct extensive experiments across five adversarial safety benchmarks, demonstrating that RLVR can simultaneously enhance reasoning capabilities while maintaining or improving safety guardrails. Our comprehensive ablation studies examine the effects of optimization algorithms, model scale, and task domains. Our findings challenge the prevailing assumption of an inevitable safety capability trade-off, and establish that a specific training methodology can achieve both objectives simultaneously, providing insights for the safe deployment of reasoning-capable LLMs. 

---
# FedAPA: Federated Learning with Adaptive Prototype Aggregation Toward Heterogeneous Wi-Fi CSI-based Crowd Counting 

**Authors**: Jingtao Guo, Yuyi Mao, Ivan Wang-Hei Ho  

**Link**: [PDF](https://arxiv.org/pdf/2511.21048)  

**Abstract**: Wi-Fi channel state information (CSI)-based sensing provides a non-invasive, device-free approach for tasks such as human activity recognition and crowd counting, but large-scale deployment is hindered by the need for extensive site-specific training data. Federated learning (FL) offers a way to avoid raw data sharing but is challenged by heterogeneous sensing data and device resources. This paper proposes FedAPA, a collaborative Wi-Fi CSI-based sensing algorithm that uses adaptive prototype aggregation (APA) strategy to assign similarity-based weights to peer prototypes, enabling adaptive client contributions and yielding a personalized global prototype for each client instead of a fixed-weight aggregation. During local training, we adopt a hybrid objective that combines classification learning with representation contrastive learning to align local and global knowledge. We provide a convergence analysis of FedAPA and evaluate it in a real-world distributed Wi-Fi crowd counting scenario with six environments and up to 20 people. The results show that our method outperform multiple baselines in terms of accuracy, F1 score, mean absolute error (MAE), and communication overhead, with FedAPA achieving at least a 9.65% increase in accuracy, a 9% gain in F1 score, a 0.29 reduction in MAE, and a 95.94% reduction in communication overhead. 

---
# Semantic Anchors in In-Context Learning: Why Small LLMs Cannot Flip Their Labels 

**Authors**: Anantha Padmanaban Krishna Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.21038)  

**Abstract**: Can in-context learning (ICL) override pre-trained label semantics, or does it merely refine an existing semantic backbone? We address this question by treating LLMs as prompt-induced classifiers and contrasting their behavior under \emph{natural} demonstrations (with correct labels) and \emph{inverted} demonstrations (systematically flipping label meanings). We decompose ICL behavior into three alignment metrics (truth, prior, and prompt alignment) and introduce a semantic override rate, defined as correctness under flipped semantics. Across eight classification tasks and eight open-source LLMs (1--12B parameters), we find consistent evidence for a semantic anchor view. With natural demonstrations, ICL improves accuracy while maintaining strong prior alignment; most correct predictions coincide with zero-shot behavior, even when the prior is weak. With inverted demonstrations, models cannot learn coherent anti-semantic classifiers: prompt alignment increases only by sacrificing accuracy, and semantic override rates remain exactly zero in our few-shot 1--12B setting. Rather than flexibly remapping label meanings, ICL primarily adjusts how inputs project onto stable semantic directions learned during pre-training, clarifying fundamental limits of few-shot prompting and suggesting that overriding label semantics at these scales requires interventions beyond ICL. All code is available at: this https URL. 

---
# Structure-Aware Prototype Guided Trusted Multi-View Classification 

**Authors**: Haojian Huang, Jiahao Shi, Zhe Liu, Harold Haodong Chen, Han Fang, Hao Sun, Zhongjiang He  

**Link**: [PDF](https://arxiv.org/pdf/2511.21021)  

**Abstract**: Trustworthy multi-view classification (TMVC) addresses the challenge of achieving reliable decision-making in complex scenarios where multi-source information is heterogeneous, inconsistent, or even conflicting. Existing TMVC approaches predominantly rely on globally dense neighbor relationships to model intra-view dependencies, leading to high computational costs and an inability to directly ensure consistency across inter-view relationships. Furthermore, these methods typically aggregate evidence from different views through manually assigned weights, lacking guarantees that the learned multi-view neighbor structures are consistent within the class space, thus undermining the trustworthiness of classification outcomes. To overcome these limitations, we propose a novel TMVC framework that introduces prototypes to represent the neighbor structures of each view. By simplifying the learning of intra-view neighbor relations and enabling dynamic alignment of intra- and inter-view structure, our approach facilitates more efficient and consistent discovery of cross-view consensus. Extensive experiments on multiple public multi-view datasets demonstrate that our method achieves competitive downstream performance and robustness compared to prevalent TMVC methods. 

---
# Probabilistic Wildfire Spread Prediction Using an Autoregressive Conditional Generative Adversarial Network 

**Authors**: Taehoon Kang, Taeyong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.21019)  

**Abstract**: Climate change has intensified the frequency and severity of wildfires, making rapid and accurate prediction of fire spread essential for effective mitigation and response. Physics-based simulators such as FARSITE offer high-fidelity predictions but are computationally intensive, limiting their applicability in real-time decision-making, while existing deep learning models often yield overly smooth predictions that fail to capture the complex, nonlinear dynamics of wildfire propagation. This study proposes an autoregressive conditional generative adversarial network (CGAN) for probabilistic wildfire spread prediction. By formulating the prediction task as an autoregressive problem, the model learns sequential state transitions, ensuring long-term prediction stability. Experimental results demonstrate that the proposed CGAN-based model outperforms conventional deep learning models in both overall predictive accuracy and boundary delineation of fire perimeters. These results demonstrate that adversarial learning allows the model to capture the strong nonlinearity and uncertainty of wildfire spread, instead of simply fitting the pixel average. Furthermore, the autoregressive framework facilitates systematic temporal forecasting of wildfire evolution. The proposed CGAN-based autoregressive framework enhances both the accuracy and physical interpretability of wildfire spread prediction, offering a promising foundation for time-sensitive response and evacuation planning. 

---
# Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning 

**Authors**: Xiaoxing You, Qiang Huang, Lingyu Li, Chi Zhang, Xiaopeng Liu, Min Zhang, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.21002)  

**Abstract**: News image captioning aims to produce journalistically informative descriptions by combining visual content with contextual cues from associated articles. Despite recent advances, existing methods struggle with three key challenges: (1) incomplete information coverage, (2) weak cross-modal alignment, and (3) suboptimal visual-entity grounding. To address these issues, we introduce MERGE, the first Multimodal Entity-aware Retrieval-augmented GEneration framework for news image captioning. MERGE constructs an entity-centric multimodal knowledge base (EMKB) that integrates textual, visual, and structured knowledge, enabling enriched background retrieval. It improves cross-modal alignment through a multistage hypothesis-caption strategy and enhances visual-entity matching via dynamic retrieval guided by image content. Extensive experiments on GoodNews and NYTimes800k show that MERGE significantly outperforms state-of-the-art baselines, with CIDEr gains of +6.84 and +1.16 in caption quality, and F1-score improvements of +4.14 and +2.64 in named entity recognition. Notably, MERGE also generalizes well to the unseen Visual News dataset, achieving +20.17 in CIDEr and +6.22 in F1-score, demonstrating strong robustness and domain adaptability. 

---
# FANoise: Singular Value-Adaptive Noise Modulation for Robust Multimodal Representation Learning 

**Authors**: Jiaoyang Li, Jun Fang, Tianhao Gao, Xiaohui Zhang, Zhiyuan Liu, Chao Liu, Pengzhang Liu, Qixia Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20997)  

**Abstract**: Representation learning is fundamental to modern machine learning, powering applications such as text retrieval and multimodal understanding. However, learning robust and generalizable representations remains challenging. While prior work has demonstrated that active noise injection, a form of data augmentation, can enhance encoding performance, most existing methods rely on heuristic or static noise, overlooking the dynamic nature of feature distributions during training. In this work, we systematically study the role of noise in representation learning from both gradient-based and feature distribution perspectives, using InfoNCE loss as a representative example. Focusing on multimodal representation learning, we propose FANoise, a novel feature-adaptive noise injection strategy. By leveraging the dynamics of contrastive learning, FANoise effectively mitigates the negative impacts of noise while preserving its benefits. Under this theoretically grounded framework, comprehensive experiments demonstrate that FANoise consistently improves overall performance on multimodal tasks across various base VLM models. 

---
# GuardTrace-VL: Detecting Unsafe Multimodel Reasoning via Iterative Safety Supervision 

**Authors**: Yuxiao Xiang, Junchi Chen, Zhenchao Jin, Changtao Miao, Haojie Yuan, Qi Chu, Tao Gong, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20994)  

**Abstract**: Multimodal large reasoning models (MLRMs) are increasingly deployed for vision-language tasks that produce explicit intermediate rationales. However, reasoning traces can contain unsafe content even when the final answer is non-harmful, creating deployment risks. Existing multimodal safety guards primarily evaluate only the input question and the final answer, neglecting the intermediate reasoning process. This oversight allows undetected harm, such as biased inferences or policy-violating use of visual context, to emerge during reasoning. We introduce GuardTrace-VL, a vision-aware safety auditor that monitors the full Question-Thinking-Answer (QTA) pipeline via joint image-text analysis, enabling detection of unsafe content as it emerges in the reasoning stage. To support training and evaluation, we construct the GuardTrace dataset, which is generated through diverse prompting strategies and refined via a MLRM- and human-based voting and verification pipeline. Furthermore, we propose a three-stage progressive training scheme combined with the data refinement process, enabling the model to learn nuanced and context-dependent safety preferences according to different risk levels. On our proposed test set covering both in-domain and out-of-domain scenarios, GuardTrace-VL model achieves an F1 score of 93.1% on unsafe reasoning detection tasks, representing a 13.5% improvement in F1 score compared to the previous strongest multimodal safety defense methods. The codes will be made publicly available. 

---
# Subgoal Graph-Augmented Planning for LLM-Guided Open-World Reinforcement Learning 

**Authors**: Shanwei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20993)  

**Abstract**: Large language models (LLMs) offer strong high-level planning capabilities for reinforcement learning (RL) by decomposing tasks into subgoals. However, their practical utility is limited by poor planning-execution alignment, which reflects a critical gap between abstract plans and actionable, environment-compatible behaviors. This misalignment arises from two interrelated limitations: (1) LLMs often produce subgoals that are semantically plausible but infeasible or irrelevant in the target environment due to insufficient grounding in environment-specific knowledge, and (2) single-LLM planning conflates generation with self-verification, resulting in overconfident yet unreliable subgoals that frequently fail during execution. To address these challenges, we propose Subgoal Graph-Augmented Actor-Critic-Refiner (SGA-ACR), a framework that integrates an environment-specific subgoal graph and structured entity knowledge with a multi-LLM planning pipeline that explicitly separates generation, critique, and refinement to produce executable and verifiable subgoals. A subgoal tracker further monitors execution progress, provides auxiliary rewards, and adaptively updates the subgoal graph to maintain alignment between plans and actions. Experimental results on 22 diverse tasks in the open-world game "Crafter" demonstrate the effectiveness of our proposed method. 

---
# Even with AI, Bijection Discovery is Still Hard: The Opportunities and Challenges of OpenEvolve for Novel Bijection Construction 

**Authors**: Davis Brown, Jesse He, Helen Jenne, Henry Kvinge, Max Vargas  

**Link**: [PDF](https://arxiv.org/pdf/2511.20987)  

**Abstract**: Evolutionary program synthesis systems such as AlphaEvolve, OpenEvolve, and ShinkaEvolve offer a new approach to AI-assisted mathematical discovery. These systems utilize teams of large language models (LLMs) to generate candidate solutions to a problem as human readable code. These candidate solutions are then 'evolved' with the goal of improving them beyond what an LLM can produce in a single shot. While existing mathematical applications have mostly focused on problems of establishing bounds (e.g., sphere packing), the program synthesis approach is well suited to any problem where the solution takes the form of an explicit construction. With this in mind, in this paper we explore the use of OpenEvolve for combinatorial bijection discovery. We describe the results of applying OpenEvolve to three bijection construction problems involving Dyck paths, two of which are known and one of which is open. We find that while systems like OpenEvolve show promise as a valuable tool for combinatorialists, the problem of finding novel, research-level bijections remains a challenging task for current frontier systems, reinforcing the need for human mathematicians in the loop. We describe some lessons learned for others in the field interested in exploring the use of these systems. 

---
# AI4X Roadmap: Artificial Intelligence for the advancement of scientific pursuit and its future directions 

**Authors**: Stephen G. Dale, Nikita Kazeev, Alastair J. A. Price, Victor Posligua, Stephan Roche, O. Anatole von Lilienfeld, Konstantin S. Novoselov, Xavier Bresson, Gianmarco Mengaldo, Xudong Chen, Terence J. O'Kane, Emily R. Lines, Matthew J. Allen, Amandine E. Debus, Clayton Miller, Jiayu Zhou, Hiroko H. Dodge, David Rousseau, Andrey Ustyuzhanin, Ziyun Yan, Mario Lanza, Fabio Sciarrino, Ryo Yoshida, Zhidong Leong, Teck Leong Tan, Qianxiao Li, Adil Kabylda, Igor Poltavsky, Alexandre Tkatchenko, Sherif Abdulkader Tawfik, Prathami Divakar Kamath, Theo Jaffrelot Inizan, Kristin A. Persson, Bryant Y. Li, Vir Karan, Chenru Duan, Haojun Jia, Qiyuan Zhao, Hiroyuki Hayashi, Atsuto Seko, Isao Tanaka, Omar M. Yaghi, Tim Gould, Bun Chan, Stefan Vuckovic, Tianbo Li, Min Lin, Zehcen Tang, Yang Li, Yong Xu, Amrita Joshi, Xiaonan Wang, Leonard W.T. Ng, Sergei V. Kalinin, Mahshid Ahmadi, Jiyizhe Zhang, Shuyuan Zhang, Alexei Lapkin, Ming Xiao, Zhe Wu, Kedar Hippalgaonkar, Limsoon Wong, Lorenzo Bastonero, Nicola Marzari, Dorye Luis Esteras Cordoba, Andrei Tomut, Alba Quinones Andrade, Jose-Hugo Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2511.20976)  

**Abstract**: Artificial intelligence and machine learning are reshaping how we approach scientific discovery, not by replacing established methods but by extending what researchers can probe, predict, and design. In this roadmap we provide a forward-looking view of AI-enabled science across biology, chemistry, climate science, mathematics, materials science, physics, self-driving laboratories and unconventional computing. Several shared themes emerge: the need for diverse and trustworthy data, transferable electronic-structure and interatomic models, AI systems integrated into end-to-end scientific workflows that connect simulations to experiments and generative systems grounded in synthesisability rather than purely idealised phases. Across domains, we highlight how large foundation models, active learning and self-driving laboratories can close loops between prediction and validation while maintaining reproducibility and physical interpretability. Taken together, these perspectives outline where AI-enabled science stands today, identify bottlenecks in data, methods and infrastructure, and chart concrete directions for building AI systems that are not only more powerful but also more transparent and capable of accelerating discovery in complex real-world environments. 

---
# Towards Audio Token Compression in Large Audio Language Models 

**Authors**: Saurabhchand Bhati, Samuel Thomas, Hilde Kuehne, Rogerio Feris, James Glass  

**Link**: [PDF](https://arxiv.org/pdf/2511.20973)  

**Abstract**: Large Audio Language Models (LALMs) demonstrate impressive performance across diverse tasks, ranging from speech recognition to general audio understanding. However, their scalability is limited by the quadratic complexity of attention and the high token rates of audio signals. These challenges make it difficult to extend LALMs to long-form audio and to deploy them on resource-constrained platforms such as edge devices.
In this paper, we explore techniques such as unsupervised segmentation, uniform average pooling, etc., to reduce the number of audio tokens generated by the LALM's audio encoder but before they are consumed by the LLM decoder. To mitigate potential performance degradation introduced by the compressed representations, we employ low-rank adapters to finetune the model. We evaluate our proposed models on two tasks, automatic speech recognition and speech-to-speech translation tasks, that are dependent on effectively uncovering the underlying lexical content of the input signal and study the effect of downsampling on these tasks. Experimental results show that compressed LALMs can achieve performance closer to frame-level LALMs while reducing the input audio token count upto three times before the LLM backbone. 

---
# BUSTR: Breast Ultrasound Text Reporting with a Descriptor-Aware Vision-Language Model 

**Authors**: Rawa Mohammed, Mina Attin, Bryar Shareef  

**Link**: [PDF](https://arxiv.org/pdf/2511.20956)  

**Abstract**: Automated radiology report generation (RRG) for breast ultrasound (BUS) is limited by the lack of paired image-report datasets and the risk of hallucinations from large language models. We propose BUSTR, a multitask vision-language framework that generates BUS reports without requiring paired image-report supervision. BUSTR constructs reports from structured descriptors (e.g., BI-RADS, pathology, histology) and radiomics features, learns descriptor-aware visual representations with a multi-head Swin encoder trained using a multitask loss over dataset-specific descriptor sets, and aligns visual and textual tokens via a dual-level objective that combines token-level cross-entropy with a cosine-similarity alignment loss between input and output representations. We evaluate BUSTR on two public BUS datasets, BrEaST and BUS-BRA, which differ in size and available descriptors. Across both datasets, BUSTR consistently improves standard natural language generation metrics and clinical efficacy metrics, particularly for key targets such as BI-RADS category and pathology. Our results show that this descriptor-aware vision model, trained with a combined token-level and alignment loss, improves both automatic report metrics and clinical efficacy without requiring paired image-report data. The source code can be found at this https URL 

---
# SpaceX: Exploring metrics with the SPACE model for developer productivity 

**Authors**: Sanchit Kaul, Kevin Nhu, Jason Eissayou, Ivan Eser, Victor Borup  

**Link**: [PDF](https://arxiv.org/pdf/2511.20955)  

**Abstract**: This empirical investigation elucidates the limitations of deterministic, unidimensional productivity heuristics by operationalizing the SPACE framework through extensive repository mining. Utilizing a dataset derived from open-source repositories, the study employs rigorous statistical methodologies including Generalized Linear Mixed Models (GLMM) and RoBERTa-based sentiment classification to synthesize a holistic, multi-faceted productivity metric. Analytical results reveal a statistically significant positive correlation between negative affective states and commit frequency, implying a cycle of iterative remediation driven by frustration. Furthermore, the investigation has demonstrated that analyzing the topology of contributor interactions yields superior fidelity in mapping collaborative dynamics compared to traditional volume-based metrics. Ultimately, this research posits a Composite Productivity Score (CPS) to address the heterogeneity of developer efficacy. 

---
# Resilient Charging Infrastructure via Decentralized Coordination of Electric Vehicles at Scale 

**Authors**: Chuhao Qin, Alexandru Sorici, Andrei Olaru, Evangelos Pournaras, Adina Magda Florea  

**Link**: [PDF](https://arxiv.org/pdf/2511.20943)  

**Abstract**: The rapid adoption of electric vehicles (EVs) introduces major challenges for decentralized charging control. Existing decentralized approaches efficiently coordinate a large number of EVs to select charging stations while reducing energy costs, preventing power peak and preserving driver privacy. However, they often struggle under severe contingencies, such as station outages or unexpected surges in charging requests. These situations create competition for limited charging slots, resulting in long queues and reduced driver comfort. To address these limitations, we propose a novel collective learning-based coordination framework that allows EVs to balance individual comfort on their selections against system-wide efficiency, i.e., the overall queues across all stations. In the framework, EVs are recommended for adaptive charging behaviors that shift priority between comfort and efficiency, achieving Pareto-optimal trade-offs under varying station capacities and dynamic spatio-temporal EV distribution. Experiments using real-world data from EVs and charging stations show that the proposed approach outperforms baseline methods, significantly reducing travel and queuing time. The results reveal that, under uncertain charging conditions, EV drivers that behave selfishly or altruistically at the right moments achieve shorter waiting time than those maintaining moderate behavior throughout. Our findings under high fractions of station outages and adversarial EVs further demonstrate improved resilience and trustworthiness of decentralized EV charging infrastructure. 

---
# Open Vocabulary Compositional Explanations for Neuron Alignment 

**Authors**: Biagio La Rosa, Leilani H. Gilpin  

**Link**: [PDF](https://arxiv.org/pdf/2511.20931)  

**Abstract**: Neurons are the fundamental building blocks of deep neural networks, and their interconnections allow AI to achieve unprecedented results. Motivated by the goal of understanding how neurons encode information, compositional explanations leverage logical relationships between concepts to express the spatial alignment between neuron activations and human knowledge. However, these explanations rely on human-annotated datasets, restricting their applicability to specific domains and predefined concepts. This paper addresses this limitation by introducing a framework for the vision domain that allows users to probe neurons for arbitrary concepts and datasets. Specifically, the framework leverages masks generated by open vocabulary semantic segmentation to compute open vocabulary compositional explanations. The proposed framework consists of three steps: specifying arbitrary concepts, generating semantic segmentation masks using open vocabulary models, and deriving compositional explanations from these masks. The paper compares the proposed framework with previous methods for computing compositional explanations both in terms of quantitative metrics and human interpretability, analyzes the differences in explanations when shifting from human-annotated data to model-annotated data, and showcases the additional capabilities provided by the framework in terms of flexibility of the explanations with respect to the tasks and properties of interest. 

---
# Exploring Time-Step Size in Reinforcement Learning for Sepsis Treatment 

**Authors**: Yingchuan Sun, Shengpu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20913)  

**Abstract**: Existing studies on reinforcement learning (RL) for sepsis management have mostly followed an established problem setup, in which patient data are aggregated into 4-hour time steps. Although concerns have been raised regarding the coarseness of this time-step size, which might distort patient dynamics and lead to suboptimal treatment policies, the extent to which this is a problem in practice remains unexplored. In this work, we conducted empirical experiments for a controlled comparison of four time-step sizes ($\Delta t\!=\!1,2,4,8$ h) on this domain, following an identical offline RL pipeline. To enable a fair comparison across time-step sizes, we designed action re-mapping methods that allow for evaluation of policies on datasets with different time-step sizes, and conducted cross-$\Delta t$ model selections under two policy learning setups. Our goal was to quantify how time-step size influences state representation learning, behavior cloning, policy training, and off-policy evaluation. Our results show that performance trends across $\Delta t$ vary as learning setups change, while policies learned at finer time-step sizes ($\Delta t = 1$ h and $2$ h) using a static behavior policy achieve the overall best performance and stability. Our work highlights time-step size as a core design choice in offline RL for healthcare and provides evidence supporting alternatives beyond the conventional 4-hour setup. 

---
# Evolved SampleWeights for Bias Mitigation: Effectiveness Depends on Optimization Objectives 

**Authors**: Anil K. Saini, Jose Guadalupe Hernandez, Emily F. Wong, Debanshi Misra, Jason H. Moore  

**Link**: [PDF](https://arxiv.org/pdf/2511.20909)  

**Abstract**: Machine learning models trained on real-world data may inadvertently make biased predictions that negatively impact marginalized communities. Reweighting is a method that can mitigate such bias in model predictions by assigning a weight to each data point used during model training. In this paper, we compare three methods for generating these weights: (1) evolving them using a Genetic Algorithm (GA), (2) computing them using only dataset characteristics, and (3) assigning equal weights to all data points. Model performance under each strategy was evaluated using paired predictive and fairness metrics, which also served as optimization objectives for the GA during evolution. Specifically, we used two predictive metrics (accuracy and area under the Receiver Operating Characteristic curve) and two fairness metrics (demographic parity difference and subgroup false negative fairness). Using experiments on eleven publicly available datasets (including two medical datasets), we show that evolved sample weights can produce models that achieve better trade-offs between fairness and predictive performance than alternative weighting methods. However, the magnitude of these benefits depends strongly on the choice of optimization objectives. Our experiments reveal that optimizing with accuracy and demographic parity difference metrics yields the largest number of datasets for which evolved weights are significantly better than other weighting strategies in optimizing both objectives. 

---
# Dynamic Test-Time Compute Scaling in Control Policy: Difficulty-Aware Stochastic Interpolant Policy 

**Authors**: Inkook Chun, Seungjae Lee, Michael S. Albergo, Saining Xie, Eric Vanden-Eijnden  

**Link**: [PDF](https://arxiv.org/pdf/2511.20906)  

**Abstract**: Diffusion- and flow-based policies deliver state-of-the-art performance on long-horizon robotic manipulation and imitation learning tasks. However, these controllers employ a fixed inference budget at every control step, regardless of task complexity, leading to computational inefficiency for simple subtasks while potentially underperforming on challenging ones. To address these issues, we introduce Difficulty-Aware Stochastic Interpolant Policy (DA-SIP), a framework that enables robotic controllers to adaptively adjust their integration horizon in real time based on task difficulty. Our approach employs a difficulty classifier that analyzes observations to dynamically select the step budget, the optimal solver variant, and ODE/SDE integration at each control cycle. DA-SIP builds upon the stochastic interpolant formulation to provide a unified framework that unlocks diverse training and inference configurations for diffusion- and flow-based policies. Through comprehensive benchmarks across diverse manipulation tasks, DA-SIP achieves 2.6-4.4x reduction in total computation time while maintaining task success rates comparable to fixed maximum-computation baselines. By implementing adaptive computation within this framework, DA-SIP transforms generative robot controllers into efficient, task-aware systems that intelligently allocate inference resources where they provide the greatest benefit. 

---
# A Taxonomy of Pix Fraud in Brazil: Attack Methodologies, AI-Driven Amplification, and Defensive Strategies 

**Authors**: Glener Lanes Pizzolato, Brenda Medeiros Lopes, Claudio Schepke, Diego Kreutz  

**Link**: [PDF](https://arxiv.org/pdf/2511.20902)  

**Abstract**: This work presents a review of attack methodologies targeting Pix, the instant payment system launched by the Central Bank of Brazil in 2020. The study aims to identify and classify the main types of fraud affecting users and financial institutions, highlighting the evolution and increasing sophistication of these techniques. The methodology combines a structured literature review with exploratory interviews conducted with professionals from the banking sector. The results show that fraud schemes have evolved from purely social engineering approaches to hybrid strategies that integrate human manipulation with technical exploitation. The study concludes that security measures must advance at the same pace as the growing complexity of attack methodologies, with particular emphasis on adaptive defenses and continuous user awareness. 

---
# Test-Time Alignment of Text-to-Image Diffusion Models via Null-Text Embedding Optimisation 

**Authors**: Taehoon Kim, Henry Gouk, Timothy Hospedales  

**Link**: [PDF](https://arxiv.org/pdf/2511.20889)  

**Abstract**: Test-time alignment (TTA) aims to adapt models to specific rewards during inference. However, existing methods tend to either under-optimise or over-optimise (reward hack) the target reward function. We propose Null-Text Test-Time Alignment (Null-TTA), which aligns diffusion models by optimising the unconditional embedding in classifier-free guidance, rather than manipulating latent or noise variables. Due to the structured semantic nature of the text embedding space, this ensures alignment occurs on a semantically coherent manifold and prevents reward hacking (exploiting non-semantic noise patterns to improve the reward). Since the unconditional embedding in classifier-free guidance serves as the anchor for the model's generative distribution, Null-TTA directly steers model's generative distribution towards the target reward rather than just adjusting the samples, even without updating model parameters. Thanks to these desirable properties, we show that Null-TTA achieves state-of-the-art target test-time alignment while maintaining strong cross-reward generalisation. This establishes semantic-space optimisation as an effective and principled novel paradigm for TTA. 

---
# Selecting Belief-State Approximations in Simulators with Latent States 

**Authors**: Nan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20870)  

**Abstract**: State resetting is a fundamental but often overlooked capability of simulators. It supports sample-based planning by allowing resets to previously encountered simulation states, and enables calibration of simulators using real data by resetting to states observed in real-system traces. While often taken for granted, state resetting in complex simulators can be nontrivial: when the simulator comes with latent variables (states), state resetting requires sampling from the posterior over the latent state given the observable history, a.k.a. the belief state (Silver and Veness, 2010). While exact sampling is often infeasible, many approximate belief-state samplers can be constructed, raising the question of how to select among them using only sampling access to the simulator.
In this paper, we show that this problem reduces to a general conditional distribution-selection task and develop a new algorithm and analysis under sampling-only access. Building on this reduction, the belief-state selection problem admits two different formulations: latent state-based selection, which directly targets the conditional distribution of the latent state, and observation-based selection, which targets the induced distribution over the observation. Interestingly, these formulations differ in how their guarantees interact with the downstream roll-out methods: perhaps surprisingly, observation-based selection may fail under the most natural roll-out method (which we call Single-Reset) but enjoys guarantees under the less conventional alternative (which we call Repeated-Reset). Together with discussion on issues such as distribution shift and the choice of sampling policies, our paper reveals a rich landscape of algorithmic choices, theoretical nuances, and open questions, in this seemingly simple problem. 

---
# Computing Evolutionarily Stable Strategies in Multiplayer Games 

**Authors**: Sam Ganzfried  

**Link**: [PDF](https://arxiv.org/pdf/2511.20859)  

**Abstract**: We present an algorithm for computing all evolutionarily stable strategies in nondegenerate normal-form games with three or more players. 

---
# Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory 

**Authors**: Tianxin Wei, Noveen Sachdeva, Benjamin Coleman, Zhankui He, Yuanchen Bei, Xuying Ning, Mengting Ai, Yunzhe Li, Jingrui He, Ed H. Chi, Chi Wang, Shuo Chen, Fernando Pereira, Wang-Cheng Kang, Derek Zhiyuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.20857)  

**Abstract**: Statefulness is essential for large language model (LLM) agents to perform long-term planning and problem-solving. This makes memory a critical component, yet its management and evolution remain largely underexplored. Existing evaluations mostly focus on static conversational settings, where memory is passively retrieved from dialogue to answer queries, overlooking the dynamic ability to accumulate and reuse experience across evolving task streams. In real-world environments such as interactive problem assistants or embodied agents, LLMs are required to handle continuous task streams, yet often fail to learn from accumulated interactions, losing valuable contextual insights, a limitation that calls for test-time evolution, where LLMs retrieve, integrate, and update memory continuously during deployment. To bridge this gap, we introduce Evo-Memory, a comprehensive streaming benchmark and framework for evaluating self-evolving memory in LLM agents. Evo-Memory structures datasets into sequential task streams, requiring LLMs to search, adapt, and evolve memory after each interaction. We unify and implement over ten representative memory modules and evaluate them across 10 diverse multi-turn goal-oriented and single-turn reasoning and QA datasets. To better benchmark experience reuse, we provide a baseline method, ExpRAG, for retrieving and utilizing prior experience, and further propose ReMem, an action-think-memory refine pipeline that tightly integrates reasoning, task actions, and memory updates to achieve continual improvement. 

---
# Unsupervised Memorability Modeling from Tip-of-the-Tongue Retrieval Queries 

**Authors**: Sree Bhattacharyya, Yaman Kumar Singla, Sudhir Yarram, Somesh Kumar Singh, Harini S I, James Z. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20854)  

**Abstract**: Visual content memorability has intrigued the scientific community for decades, with applications ranging widely, from understanding nuanced aspects of human memory to enhancing content design. A significant challenge in progressing the field lies in the expensive process of collecting memorability annotations from humans. This limits the diversity and scalability of datasets for modeling visual content memorability. Most existing datasets are limited to collecting aggregate memorability scores for visual content, not capturing the nuanced memorability signals present in natural, open-ended recall descriptions. In this work, we introduce the first large-scale unsupervised dataset designed explicitly for modeling visual memorability signals, containing over 82,000 videos, accompanied by descriptive recall data. We leverage tip-of-the-tongue (ToT) retrieval queries from online platforms such as Reddit. We demonstrate that our unsupervised dataset provides rich signals for two memorability-related tasks: recall generation and ToT retrieval. Large vision-language models fine-tuned on our dataset outperform state-of-the-art models such as GPT-4o in generating open-ended memorability descriptions for visual content. We also employ a contrastive training strategy to create the first model capable of performing multimodal ToT retrieval. Our dataset and models present a novel direction, facilitating progress in visual content memorability research. 

---
# MODEST: Multi-Optics Depth-of-Field Stereo Dataset 

**Authors**: Nisarg K. Trivedi, Vinayak A. Belludi, Li-Yun Wang, Pardis Taghavi, Dante Lok  

**Link**: [PDF](https://arxiv.org/pdf/2511.20853)  

**Abstract**: Reliable depth estimation under real optical conditions remains a core challenge for camera vision in systems such as autonomous robotics and augmented reality. Despite recent progress in depth estimation and depth-of-field rendering, research remains constrained by the lack of large-scale, high-fidelity, real stereo DSLR datasets, limiting real-world generalization and evaluation of models trained on synthetic data as shown extensively in literature. We present the first high-resolution (5472$\times$3648px) stereo DSLR dataset with 18000 images, systematically varying focal length and aperture across complex real scenes and capturing the optical realism and complexity of professional camera systems. For 9 scenes with varying scene complexity, lighting and background, images are captured with two identical camera assemblies at 10 focal lengths (28-70mm) and 5 apertures (f/2.8-f/22), spanning 50 optical configurations in 2000 images per scene. This full-range optics coverage enables controlled analysis of geometric and optical effects for monocular and stereo depth estimation, shallow depth-of-field rendering, deblurring, 3D scene reconstruction and novel view synthesis. Each focal configuration has a dedicated calibration image set, supporting evaluation of classical and learning based methods for intrinsic and extrinsic calibration. The dataset features challenging visual elements such as multi-scale optical illusions, reflective surfaces, mirrors, transparent glass walls, fine-grained details, and natural / artificial ambient light variations. This work attempts to bridge the realism gap between synthetic training data and real camera optics, and demonstrates challenges with the current state-of-the-art monocular, stereo depth and depth-of-field methods. We release the dataset, calibration files, and evaluation code to support reproducible research on real-world optical generalization. 

---
# Length-MAX Tokenizer for Language Models 

**Authors**: Dong Dong, Weijie Su  

**Link**: [PDF](https://arxiv.org/pdf/2511.20849)  

**Abstract**: We introduce a new tokenizer for language models that minimizes the average tokens per character, thereby reducing the number of tokens needed to represent text during training and to generate text during inference. Our method, which we refer to as the Length-MAX tokenizer, obtains its vocabulary by casting a length-weighted objective maximization as a graph partitioning problem and developing a greedy approximation algorithm. On FineWeb and diverse domains, it yields 14--18\% fewer tokens than Byte Pair Encoding (BPE) across vocabulary sizes from 10K to 50K, and the reduction is 13.0\% when the size is 64K. Training GPT-2 models at 124M, 355M, and 1.3B parameters from scratch with five runs each shows 18.5\%, 17.2\%, and 18.5\% fewer steps, respectively, to reach a fixed validation loss, and 13.7\%, 12.7\%, and 13.7\% lower inference latency, together with a 16\% throughput gain at 124M, while consistently improving on downstream tasks including reducing LAMBADA perplexity by 11.7\% and enhancing HellaSwag accuracy by 4.3\%. Moreover, the Length-MAX tokenizer achieves 99.62\% vocabulary coverage and the out-of-vocabulary rate remains low at 0.12\% on test sets. These results demonstrate that optimizing for average token length, rather than frequency alone, offers an effective approach to more efficient language modeling without sacrificing -- and often improving -- downstream performance. The tokenizer is compatible with production systems and reduces embedding and KV-cache memory by 18\% at inference. 

---
# NOIR 2.0: Neural Signal Operated Intelligent Robots for Everyday Activities 

**Authors**: Tasha Kim, Yingke Wang, Hanvit Cho, Alex Hodges  

**Link**: [PDF](https://arxiv.org/pdf/2511.20848)  

**Abstract**: Neural Signal Operated Intelligent Robots (NOIR) system is a versatile brain-robot interface that allows humans to control robots for daily tasks using their brain signals. This interface utilizes electroencephalography (EEG) to translate human intentions regarding specific objects and desired actions directly into commands that robots can execute. We present NOIR 2.0, an enhanced version of NOIR. NOIR 2.0 includes faster and more accurate brain decoding algorithms, which reduce task completion time by 46%. NOIR 2.0 uses few-shot robot learning algorithms to adapt to individual users and predict their intentions. The new learning algorithms leverage foundation models for more sample-efficient learning and adaptation (15 demos vs. a single demo), significantly reducing overall human time by 65%. 

---
# Pre-train to Gain: Robust Learning Without Clean Labels 

**Authors**: David Szczecina, Nicholas Pellegrino, Paul Fieguth  

**Link**: [PDF](https://arxiv.org/pdf/2511.20844)  

**Abstract**: Training deep networks with noisy labels leads to poor generalization and degraded accuracy due to overfitting to label noise. Existing approaches for learning with noisy labels often rely on the availability of a clean subset of data. By pre-training a feature extractor backbone without labels using self-supervised learning (SSL), followed by standard supervised training on the noisy dataset, we can train a more noise robust model without requiring a subset with clean labels. We evaluate the use of SimCLR and Barlow~Twins as SSL methods on CIFAR-10 and CIFAR-100 under synthetic and real world noise. Across all noise rates, self-supervised pre-training consistently improves classification accuracy and enhances downstream label-error detection (F1 and Balanced Accuracy). The performance gap widens as the noise rate increases, demonstrating improved robustness. Notably, our approach achieves comparable results to ImageNet pre-trained models at low noise levels, while substantially outperforming them under high noise conditions. 

---
# A Review of Pseudospectral Optimal Control: From Theory to Flight 

**Authors**: I. M. Ross, M. Karpenko  

**Link**: [PDF](https://arxiv.org/pdf/2511.20843)  

**Abstract**: The home space for optimal control is a Sobolev space. The home space for pseudospectral theory is also a Sobolev space. It thus seems natural to combine pseudospectral theory with optimal control theory and construct ``pseudospectral optimal control theory,'' a term coined by Ross. In this paper, we review key theoretical results in pseudospectral optimal control that have proven to be critical for a successful flight. Implementation details of flight demonstrations onboard NASA spacecraft are discussed along with emerging trends and techniques in both theory and practice. The 2011 launch of pseudospectral optimal control in embedded platforms is changing the way in which we see solutions to challenging control problems in aerospace and autonomous systems. 

---
# Primal: A Unified Deterministic Framework for Quasi-Orthogonal Hashing and Manifold Learning 

**Authors**: Vladimer Khasia  

**Link**: [PDF](https://arxiv.org/pdf/2511.20839)  

**Abstract**: We present Primal, a deterministic feature mapping framework that harnesses the number-theoretic independence of prime square roots to construct robust, tunable vector representations. Diverging from standard stochastic projections (e.g., Random Fourier Features), our method exploits the Besicovitch property to create irrational frequency modulations that guarantee infinite non-repeating phase trajectories. We formalize two distinct algorithmic variants: (1) StaticPrime, a sequence generation method that produces temporal position encodings empirically approaching the theoretical Welch bound for quasi-orthogonality; and (2) DynamicPrime, a tunable projection layer for input-dependent feature mapping. A central novelty of the dynamic framework is its ability to unify two disparate mathematical utility classes through a single scaling parameter {\sigma}. In the low-frequency regime, the method acts as an isometric kernel map, effectively linearizing non-convex geometries (e.g., spirals) to enable high-fidelity signal reconstruction and compressive sensing. Conversely, the high-frequency regime induces chaotic phase wrapping, transforming the projection into a maximum-entropy one-way hash suitable for Hyperdimensional Computing and privacy-preserving Split Learning. Empirical evaluations demonstrate that our framework yields superior orthogonality retention and distribution tightness compared to normalized Gaussian baselines, establishing it as a computationally efficient, mathematically rigorous alternative to random matrix projections. The code is available at this https URL 

---
# Structured Prompting Enables More Robust, Holistic Evaluation of Language Models 

**Authors**: Asad Aali, Muhammad Ahmed Mohsin, Vasiliki Bikia, Arnav Singhvi, Richard Gaus, Suhana Bedi, Hejie Cui, Miguel Fuentes, Alyssa Unell, Yifan Mai, Jordan Cahoon, Michael Pfeffer, Roxana Daneshjou, Sanmi Koyejo, Emily Alsentzer, Percy Liang, Christopher Potts, Nigam H. Shah, Akshay S. Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2511.20836)  

**Abstract**: As language models (LMs) are increasingly adopted across domains, high-quality benchmarking frameworks that accurately estimate performance are essential for guiding deployment decisions. While frameworks such as Holistic Evaluation of Language Models (HELM) enable broad evaluation across tasks, they often rely on fixed prompts that fail to generalize across LMs, yielding unrepresentative performance estimates. Unless we estimate each LM's ceiling (maximum achievable via changes to the prompt), we risk underestimating performance. Declarative prompting frameworks, such as DSPy, offer a scalable alternative to manual prompt engineering by crafting structured prompts that can be optimized per task. However, such frameworks have not been systematically evaluated across established benchmarks. We present a reproducible DSPy+HELM framework that introduces structured prompting methods which elicit reasoning, enabling more accurate LM benchmarking. Using four prompting methods, we evaluate four frontier LMs across seven benchmarks (general/medical domain) against existing HELM baseline scores. We find that without structured prompting: (i) HELM underestimates LM performance (by 4% average), (ii) performance estimates vary more across benchmarks (+2% standard deviation), (iii) performance gaps are misrepresented (leaderboard rankings flip on 3/7 benchmarks), and (iv) introducing reasoning (chain-of-thought) reduces LM sensitivity to prompt design (smaller {\Delta} across prompts). To our knowledge, this is the first large-scale benchmarking study to empirically characterize LM behavior across benchmarks and prompting methods, showing that scalable performance ceiling estimation enables more decision-useful benchmarks. We open-source (i) DSPy+HELM Integration (this https URL) and (ii) Prompt Optimization Pipeline (this https URL). 

---
# RefTr: Recurrent Refinement of Confluent Trajectories for 3D Vascular Tree Centerline Graphs 

**Authors**: Roman Naeem, David Hagerman, Jennifer Alvén, Fredrik Kahl  

**Link**: [PDF](https://arxiv.org/pdf/2511.20823)  

**Abstract**: Tubular trees, such as blood vessels and lung airways, are essential for material transport within the human body. Accurately detecting their centerlines with correct tree topology is critical for clinical tasks such as diagnosis, treatment planning, and surgical navigation. In these applications, maintaining high recall is crucial, as missing small branches can result in fatal mistakes caused by incomplete assessments or undetected abnormalities. We present RefTr, a 3D image-to-graph model for centerline generation of vascular trees via recurrent refinement of confluent trajectories. RefTr uses a Producer-Refiner architecture based on a Transformer decoder, where the Producer proposes a set of initial confluent trajectories that are recurrently refined by the Refiner to produce final trajectories, which forms the centerline graph. The confluent trajectory representation enables refinement of complete trajectories while explicitly enforcing a valid tree topology. The recurrent refinement scheme improves precision and reuses the same Refiner block across multiple steps, yielding a 2.4x reduction in decoder parameters compared to previous SOTA. We also introduce an efficient non-maximum suppression algorithm for spatial tree graphs to merge duplicate branches and boost precision. Across multiple public centerline datasets, RefTr achieves superior recall and comparable precision to previous SOTA, while offering faster inference and substantially fewer parameters, demonstrating its potential as a new state-of-the-art framework for vascular tree analysis in 3D medical imaging. 

---
# Training-Free Diffusion Priors for Text-to-Image Generation via Optimization-based Visual Inversion 

**Authors**: Samuele Dell'Erba, Andrew D. Bagdanov  

**Link**: [PDF](https://arxiv.org/pdf/2511.20821)  

**Abstract**: Diffusion models have established the state-of-the-art in text-to-image generation, but their performance often relies on a diffusion prior network to translate text embeddings into the visual manifold for easier decoding. These priors are computationally expensive and require extensive training on massive datasets. In this work, we challenge the necessity of a trained prior at all by employing Optimization-based Visual Inversion (OVI), a training-free and data-free alternative, to replace the need for a prior. OVI initializes a latent visual representation from random pseudo-tokens and iteratively optimizes it to maximize the cosine similarity with input textual prompt embedding. We further propose two novel constraints, a Mahalanobis-based and a Nearest-Neighbor loss, to regularize the OVI optimization process toward the distribution of realistic images. Our experiments, conducted on Kandinsky 2.2, show that OVI can serve as an alternative to traditional priors. More importantly, our analysis reveals a critical flaw in current evaluation benchmarks like T2I-CompBench++, where simply using the text embedding as a prior achieves surprisingly high scores, despite lower perceptual quality. Our constrained OVI methods improve visual fidelity over this baseline, with the Nearest-Neighbor approach proving particularly effective, achieving quantitative scores comparable to or higher than the state-of-the-art data-efficient prior, indicating that the idea merits further investigation. The code will be publicly available upon acceptance. 

---
# SPHINX: A Synthetic Environment for Visual Perception and Reasoning 

**Authors**: Md Tanvirul Alam, Saksham Aggarwal, Justin Yang Chae, Nidhi Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2511.20814)  

**Abstract**: We present Sphinx, a synthetic environment for visual perception and reasoning that targets core cognitive primitives. Sphinx procedurally generates puzzles using motifs, tiles, charts, icons, and geometric primitives, each paired with verifiable ground-truth solutions, enabling both precise evaluation and large-scale dataset construction. The benchmark covers 25 task types spanning symmetry detection, geometric transformations, spatial reasoning, chart interpretation, and sequence prediction. Evaluating recent large vision-language models (LVLMs) shows that even state-of-the-art GPT-5 attains only 51.1% accuracy, well below human performance. Finally, we demonstrate that reinforcement learning with verifiable rewards (RLVR) substantially improves model accuracy on these tasks and yields gains on external visual reasoning benchmarks, highlighting its promise for advancing multimodal reasoning. 

---
# Conformal Safety Monitoring for Flight Testing: A Case Study in Data-Driven Safety Learning 

**Authors**: Aaron O. Feldman, D. Isaiah Harp, Joseph Duncan, Mac Schwager  

**Link**: [PDF](https://arxiv.org/pdf/2511.20811)  

**Abstract**: We develop a data-driven approach for runtime safety monitoring in flight testing, where pilots perform maneuvers on aircraft with uncertain parameters. Because safety violations can arise unexpectedly as a result of these uncertainties, pilots need clear, preemptive criteria to abort the maneuver in advance of safety violation. To solve this problem, we use offline stochastic trajectory simulation to learn a calibrated statistical model of the short-term safety risk facing pilots. We use flight testing as a motivating example for data-driven learning/monitoring of safety due to its inherent safety risk, uncertainty, and human-interaction. However, our approach consists of three broadly-applicable components: a model to predict future state from recent observations, a nearest neighbor model to classify the safety of the predicted state, and classifier calibration via conformal prediction. We evaluate our method on a flight dynamics model with uncertain parameters, demonstrating its ability to reliably identify unsafe scenarios, match theoretical guarantees, and outperform baseline approaches in preemptive classification of risk. 

---
# Memories Retrieved from Many Paths: A Multi-Prefix Framework for Robust Detection of Training Data Leakage in Large Language Models 

**Authors**: Trung Cuong Dang, David Mohaisen  

**Link**: [PDF](https://arxiv.org/pdf/2511.20799)  

**Abstract**: Large language models, trained on massive corpora, are prone to verbatim memorization of training data, creating significant privacy and copyright risks. While previous works have proposed various definitions for memorization, many exhibit shortcomings in comprehensively capturing this phenomenon, especially in aligned models. To address this, we introduce a novel framework: multi-prefix memorization. Our core insight is that memorized sequences are deeply encoded and thus retrievable via a significantly larger number of distinct prefixes than non-memorized content. We formalize this by defining a sequence as memorized if an external adversarial search can identify a target count of distinct prefixes that elicit it. This framework shifts the focus from single-path extraction to quantifying the robustness of a memory, measured by the diversity of its retrieval paths. Through experiments on open-source and aligned chat models, we demonstrate that our multi-prefix definition reliably distinguishes memorized from non-memorized data, providing a robust and practical tool for auditing data leakage in LLMs. 

---
# Physics Steering: Causal Control of Cross-Domain Concepts in a Physics Foundation Model 

**Authors**: Rio Alexa Fear, Payel Mukhopadhyay, Michael McCabe, Alberto Bietti, Miles Cranmer  

**Link**: [PDF](https://arxiv.org/pdf/2511.20798)  

**Abstract**: Recent advances in mechanistic interpretability have revealed that large language models (LLMs) develop internal representations corresponding not only to concrete entities but also distinct, human-understandable abstract concepts and behaviour. Moreover, these hidden features can be directly manipulated to steer model behaviour. However, it remains an open question whether this phenomenon is unique to models trained on inherently structured data (ie. language, images) or if it is a general property of foundation models. In this work, we investigate the internal representations of a large physics-focused foundation model. Inspired by recent work identifying single directions in activation space for complex behaviours in LLMs, we extract activation vectors from the model during forward passes over simulation datasets for different physical regimes. We then compute "delta" representations between the two regimes. These delta tensors act as concept directions in activation space, encoding specific physical features. By injecting these concept directions back into the model during inference, we can steer its predictions, demonstrating causal control over physical behaviours, such as inducing or removing some particular physical feature from a simulation. These results suggest that scientific foundation models learn generalised representations of physical principles. They do not merely rely on superficial correlations and patterns in the simulations. Our findings open new avenues for understanding and controlling scientific foundation models and has implications for AI-enabled scientific discovery. 

---
# Revisiting KRISP: A Lightweight Reproduction and Analysis of Knowledge-Enhanced Vision-Language Models 

**Authors**: Souradeep Dutta, Keshav Bulia, Neena S Nair  

**Link**: [PDF](https://arxiv.org/pdf/2511.20795)  

**Abstract**: Facebook AI Research introduced KRISP [4], which integrates structured external knowledge into pipelines for vision-language reasoning. Despite its effectiveness, the original model has been developed for industrial-scale training, is computationally demanding, and is tightly connected to a large backbone. In this work, we reexamine KRISP from a different angle and offer a lightweight reproduction with significantly fewer parameters. Even though our replicated model performs about 75 % of the original, the replication process uncovers a number of design flaws, real-world pitfalls, and implicit problems that were not fully covered in the original paper. We offer insights into the scalability and efficacy of knowledge-enhanced VQA architectures under resource constraints through systematic ablation studies, which include a proof-of-concept on synthetic VQA data and evaluation on the DAQUAR dataset. Our model, configured with a low parameter setup and constrained by the external Knowledge graph domain, prevents AI hallucinations and generates outputs solely within that domain. Minimal parameters allow us to function on edge devices like smartphones and AR-VR, further improving offline visual reasoning. 

---
# Adversarial Multi-Task Learning for Liver Tumor Segmentation, Dynamic Enhancement Regression, and Classification 

**Authors**: Xiaojiao Xiao, Qinmin Vivian Hu, Tae Hyun Kim, Guanghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20793)  

**Abstract**: Liver tumor segmentation, dynamic enhancement regression, and classification are critical for clinical assessment and diagnosis. However, no prior work has attempted to achieve these tasks simultaneously in an end-to-end framework, primarily due to the lack of an effective framework that captures inter-task relevance for mutual improvement and the absence of a mechanism to extract dynamic MRI information effectively. To address these challenges, we propose the Multi-Task Interaction adversarial learning Network (MTI-Net), a novel integrated framework designed to tackle these tasks simultaneously. MTI-Net incorporates Multi-domain Information Entropy Fusion (MdIEF), which utilizes entropy-aware, high-frequency spectral information to effectively integrate features from both frequency and spectral domains, enhancing the extraction and utilization of dynamic MRI data. The network also introduces a task interaction module that establishes higher-order consistency between segmentation and regression, thus fostering inter-task synergy and improving overall performance. Additionally, we designed a novel task-driven discriminator (TDD) to capture internal high-order relationships between tasks. For dynamic MRI information extraction, we employ a shallow Transformer network to perform positional encoding, which captures the relationships within dynamic MRI sequences. In experiments on a dataset of 238 subjects, MTI-Net demonstrates high performance across multiple tasks, indicating its strong potential for assisting in the clinical assessment of liver tumors. The code is available at: this https URL. 

---
# CANVAS: A Benchmark for Vision-Language Models on Tool-Based User Interface Design 

**Authors**: Daeheon Jeong, Seoyeon Byun, Kihoon Son, Dae Hyun Kim, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.20737)  

**Abstract**: User interface (UI) design is an iterative process in which designers progressively refine their work with design software such as Figma or Sketch. Recent advances in vision language models (VLMs) with tool invocation suggest these models can operate design software to edit a UI design through iteration. Understanding and enhancing this capacity is important, as it highlights VLMs' potential to collaborate with designers within conventional software. However, as no existing benchmark evaluates tool-based design performance, the capacity remains unknown. To address this, we introduce CANVAS, a benchmark for VLMs on tool-based user interface design. Our benchmark contains 598 tool-based design tasks paired with ground-truth references sampled from 3.3K mobile UI designs across 30 function-based categories (e.g., onboarding, messaging). In each task, a VLM updates the design step-by-step through context-based tool invocations (e.g., create a rectangle as a button background), linked to design software. Specifically, CANVAS incorporates two task types: (i) design replication evaluates the ability to reproduce a whole UI screen; (ii) design modification evaluates the ability to modify a specific part of an existing screen. Results suggest that leading models exhibit more strategic tool invocations, improving design quality. Furthermore, we identify common error patterns models exhibit, guiding future work in enhancing tool-based design capabilities. 

---
# Large Language Models' Complicit Responses to Illicit Instructions across Socio-Legal Contexts 

**Authors**: Xing Wang, Huiyuan Xie, Yiyan Wang, Chaojun Xiao, Huimin Chen, Holli Sargeant, Felix Steffek, Jie Shao, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.20736)  

**Abstract**: Large language models (LLMs) are now deployed at unprecedented scale, assisting millions of users in daily tasks. However, the risk of these models assisting unlawful activities remains underexplored. In this study, we define this high-risk behavior as complicit facilitation - the provision of guidance or support that enables illicit user instructions - and present four empirical studies that assess its prevalence in widely deployed LLMs. Using real-world legal cases and established legal frameworks, we construct an evaluation benchmark spanning 269 illicit scenarios and 50 illicit intents to assess LLMs' complicit facilitation behavior. Our findings reveal widespread LLM susceptibility to complicit facilitation, with GPT-4o providing illicit assistance in nearly half of tested cases. Moreover, LLMs exhibit deficient performance in delivering credible legal warnings and positive guidance. Further analysis uncovers substantial safety variation across socio-legal contexts. On the legal side, we observe heightened complicity for crimes against societal interests, non-extreme but frequently occurring violations, and malicious intents driven by subjective motives or deceptive justifications. On the social side, we identify demographic disparities that reveal concerning complicit patterns towards marginalized and disadvantaged groups, with older adults, racial minorities, and individuals in lower-prestige occupations disproportionately more likely to receive unlawful guidance. Analysis of model reasoning traces suggests that model-perceived stereotypes, characterized along warmth and competence, are associated with the model's complicit behavior. Finally, we demonstrate that existing safety alignment strategies are insufficient and may even exacerbate complicit behavior. 

---
# InvisibleBench: A Deployment Gate for Caregiving Relationship AI 

**Authors**: Ali Madad  

**Link**: [PDF](https://arxiv.org/pdf/2511.20733)  

**Abstract**: InvisibleBench is a deployment gate for caregiving-relationship AI, evaluating 3-20+ turn interactions across five dimensions: Safety, Compliance, Trauma-Informed Design, Belonging/Cultural Fitness, and Memory. The benchmark includes autofail conditions for missed crises, medical advice (WOPR Act), harmful information, and attachment engineering. We evaluate four frontier models across 17 scenarios (N=68) spanning three complexity tiers. All models show significant safety gaps (11.8-44.8 percent crisis detection), indicating the necessity of deterministic crisis routing in production systems. DeepSeek Chat v3 achieves the highest overall score (75.9 percent), while strengths differ by dimension: GPT-4o Mini leads Compliance (88.2 percent), Gemini leads Trauma-Informed Design (85.0 percent), and Claude Sonnet 4.5 ranks highest in crisis detection (44.8 percent). We release all scenarios, judge prompts, and scoring configurations with code. InvisibleBench extends single-turn safety tests by evaluating longitudinal risk, where real harms emerge. No clinical claims; this is a deployment-readiness evaluation. 

---
# Data-Driven Methods and AI in Engineering Design: A Systematic Literature Review Focusing on Challenges and Opportunities 

**Authors**: Nehal Afifi, Christoph Wittig, Lukas Paehler, Andreas Lindenmann, Kai Wolter, Felix Leitenberger, Melih Dogru, Patric Grauberger, Tobias Düser, Albert Albers, Sven Matthiesen  

**Link**: [PDF](https://arxiv.org/pdf/2511.20730)  

**Abstract**: The increasing availability of data and advancements in computational intelligence have accelerated the adoption of data-driven methods (DDMs) in product development. However, their integration into product development remains fragmented. This fragmentation stems from uncertainty, particularly the lack of clarity on what types of DDMs to use and when to employ them across the product development lifecycle. To address this, a necessary first step is to investigate the usage of DDM in engineering design by identifying which methods are being used, at which development stages, and for what application. This paper presents a PRISMA systematic literature review. The V-model as a product development framework was adopted and simplified into four stages: system design, system implementation, system integration, and validation. A structured search across Scopus, Web of Science, and IEEE Xplore (2014--2024) retrieved 1{,}689 records. After screening, 114 publications underwent full-text analysis. Findings show that machine learning (ML) and statistical methods dominate current practice, whereas deep learning (DL), though still less common, exhibits a clear upward trend in adoption. Additionally, supervised learning, clustering, regression analysis, and surrogate modeling are prevalent in design, implementation, and integration system stages but contributions to validation remain limited. Key challenges in existing applications include limited model interpretability, poor cross-stage traceability, and insufficient validation under real-world conditions. Additionally, it highlights key limitations and opportunities such as the need for interpretable hybrid models. This review is a first step toward design-stage guidelines; a follow-up synthesis should map computer science algorithms to engineering design problems and activities. 

---
# Spatio-Temporal Trajectory Foundation Model - Recent Advances and Future Directions 

**Authors**: Sean Bin Yang, Ying Sun, Yunyao Cheng, Yan Lin, Kristian Torp, Jilin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20729)  

**Abstract**: Foundation models (FMs) have emerged as a powerful paradigm, enabling a diverse range of data analytics and knowledge discovery tasks across scientific fields. Inspired by the success of FMs, particularly large language models, researchers have recently begun to explore spatio-temporal foundation models (STFMs) to improve adaptability and generalization across a wide spectrum of spatio-temporal (ST) tasks. Despite rapid progress, a systematic investigation of trajectory foundation models (TFMs), a crucial subclass of STFMs, is largely lacking. This tutorial addresses this gap by offering a comprehensive overview of recent advances in TFMs, including a taxonomy of existing methodologies and a critical analysis of their strengths and limitations. In addition, the tutorial highlights open challenges and outlines promising research directions to advance spatio-temporal general intelligence through the development of robust, responsible, and transferable TFMs. 

---
# Learning from Risk: LLM-Guided Generation of Safety-Critical Scenarios with Prior Knowledge 

**Authors**: Yuhang Wang, Heye Huang, Zhenhua Xu, Kailai Sun, Baoshen Guo, Jinhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.20726)  

**Abstract**: Autonomous driving faces critical challenges in rare long-tail events and complex multi-agent interactions, which are scarce in real-world data yet essential for robust safety validation. This paper presents a high-fidelity scenario generation framework that integrates a conditional variational autoencoder (CVAE) with a large language model (LLM). The CVAE encodes historical trajectories and map information from large-scale naturalistic datasets to learn latent traffic structures, enabling the generation of physically consistent base scenarios. Building on this, the LLM acts as an adversarial reasoning engine, parsing unstructured scene descriptions into domain-specific loss functions and dynamically guiding scenario generation across varying risk levels. This knowledge-driven optimization balances realism with controllability, ensuring that generated scenarios remain both plausible and risk-sensitive. Extensive experiments in CARLA and SMARTS demonstrate that our framework substantially increases the coverage of high-risk and long-tail events, improves consistency between simulated and real-world traffic distributions, and exposes autonomous driving systems to interactions that are significantly more challenging than those produced by existing rule- or data-driven methods. These results establish a new pathway for safety validation, enabling principled stress-testing of autonomous systems under rare but consequential events. 

---
# Gradient Descent Algorithm Survey 

**Authors**: Deng Fucheng, Wang Wanjie, Gong Ao, Wang Xiaoqi, Wang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20725)  

**Abstract**: Focusing on the practical configuration needs of optimization algorithms in deep learning, this article concentrates on five major algorithms: SGD, Mini-batch SGD, Momentum, Adam, and Lion. It systematically analyzes the core advantages, limitations, and key practical recommendations of each algorithm. The research aims to gain an in-depth understanding of these algorithms and provide a standardized reference for the reasonable selection, parameter tuning, and performance improvement of optimization algorithms in both academic research and engineering practice, helping to solve optimization challenges in different scales of models and various training scenarios. 

---
# DinoLizer: Learning from the Best for Generative Inpainting Localization 

**Authors**: Minh Thong Doi, Jan Butora, Vincent Itier, Jérémie Boulanger, Patrick Bas  

**Link**: [PDF](https://arxiv.org/pdf/2511.20722)  

**Abstract**: We introduce DinoLizer, a DINOv2-based model for localizing manipulated regions in generative inpainting. Our method builds on a DINOv2 model pretrained to detect synthetic images on the B-Free dataset. We add a linear classification head on top of the Vision Transformer's patch embeddings to predict manipulations at a $14\times 14$ patch resolution. The head is trained to focus on semantically altered regions, treating non-semantic edits as part of the original content. Because the ViT accepts only fixed-size inputs, we use a sliding-window strategy to aggregate predictions over larger images; the resulting heatmaps are post-processed to refine the estimated binary manipulation masks. Empirical results show that DinoLizer surpasses state-of-the-art local manipulation detectors on a range of inpainting datasets derived from different generative models. It remains robust to common post-processing operations such as resizing, noise addition, and JPEG (double) compression. On average, DinoLizer achieves a 12\% higher Intersection-over-Union (IoU) than the next best model, with even greater gains after post-processing. Our experiments with off-the-shelf DINOv2 demonstrate the strong representational power of Vision Transformers for this task. Finally, extensive ablation studies comparing DINOv2 and its successor, DINOv3, in deepfake localization confirm DinoLizer's superiority. The code will be publicly available upon acceptance of the paper. 

---
# Foundry: Distilling 3D Foundation Models for the Edge 

**Authors**: Guillaume Letellier, Siddharth Srivastava, Frédéric Jurie, Gaurav Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2511.20721)  

**Abstract**: Foundation models pre-trained with self-supervised learning (SSL) on large-scale datasets have become powerful general-purpose feature extractors. However, their immense size and computational cost make them prohibitive for deployment on edge devices such as robots and AR/VR headsets. Existing compression techniques like standard knowledge distillation create efficient 'specialist' models but sacrifice the crucial, downstream-agnostic generality that makes foundation models so valuable.  In this paper, we introduce Foundation Model Distillation (FMD), a new paradigm for compressing large SSL models into compact, efficient, and faithful proxies that retain their general-purpose representational power. We present Foundry, the first implementation of FMD for 3D point clouds. Our approach, Foundry, trains a student to learn a compressed set of SuperTokens that reconstruct the teacher's token-level representations, capturing a compact basis of its latent space. A single distilled model maintains strong transferability across diverse downstream tasks-classification, part segmentation, and few-shot scenarios-approaching full foundation-model performance while using significantly fewer tokens and FLOPs, making such models more practical for deployment on resourceconstrained hardware. 

---
# DeeAD: Dynamic Early Exit of Vision-Language Action for Efficient Autonomous Driving 

**Authors**: Haibo HU, Lianming Huang, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2511.20720)  

**Abstract**: Vision-Language Action (VLA) models unify perception, reasoning, and trajectory generation for autonomous driving, but suffer from significant inference latency due to deep transformer stacks. We present DeeAD, a training-free, action-guided early-exit framework that accelerates VLA planning by evaluating the physical feasibility of intermediate trajectories. Instead of relying on confidence scores, DeeAD terminates inference when predicted trajectories align with lightweight planning priors (e.g., Navigation or Low-precision Planning) within a tolerable deviation (<2m). To improve efficiency, we introduce a multi-hop controller that adaptively skips redundant layers based on the change rate of scores. DeeAD integrates into existing VLA models, such as ORION, without requiring retraining. Experiments on the Bench2Drive benchmark demonstrate up to 28% transformer-layer sparsity and 29% latency reduction, while preserving planning quality and safety. 

---
# ST-PPO: Stabilized Off-Policy Proximal Policy Optimization for Multi-Turn Agents Training 

**Authors**: Chenliang Li, Adel Elmahdy, Alex Boyd, Zhongruo Wang, Alfredo Garcia, Parminder Bhatia, Taha Kass-Hout, Cao Xiao, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2511.20718)  

**Abstract**: PPO has been widely adopted for training large language models (LLMs) at the token level in multi-turn dialogue and reasoning tasks. However, its performance is often unstable and prone to collapse. Through empirical analysis, we identify two main sources of instability in this setting: (1)~token-level importance sampling, which is misaligned with the natural granularity of multi-turn environments that have distinct turn-level stages, and (2) inaccurate advantage estimates from off-policy samples, where the critic has not learned to evaluate certain state-action pairs, resulting in high-variance gradients and unstable updates. To address these challenges, we introduce two complementary stabilization techniques: (1) turn-level importance sampling, which aligns optimization with the natural structure of multi-turn reasoning, and (2) clipping-bias correction, which normalizes gradients by downweighting unreliable, highly off-policy samples. Depending on how these components are combined, we obtain three variants: Turn-PPO (turn-level sampling only), S-PPO (clipping-bias correction applied to token-level PPO), and ST-PPO (turn-level sampling combined with clipping-bias correction). In our experiments, we primarily study ST-PPO and S-PPO, which together demonstrate how the two stabilization mechanisms address complementary sources of instability. Experiments on multi-turn search tasks across general QA, multi-hop QA, and medical multiple-choice QA benchmarks show that ST-PPO and S-PPO consistently prevent the performance collapses observed in large-model training, maintain lower clipping ratios throughout optimization, and achieve higher task performance than standard token-level PPO. These results demonstrate that combining turn-level importance sampling with clipping-bias correction provides a practical and scalable solution for stabilizing multi-turn LLM agent training. 

---
# Inferix: A Block-Diffusion based Next-Generation Inference Engine for World Simulation 

**Authors**: Inferix Team, Tianyu Feng, Yizeng Han, Jiahao He, Yuanyu He, Xi Lin, Teng Liu, Hanfeng Lu, Jiasheng Tang, Wei Wang, Zhiyuan Wang, Jichao Wu, Mingyang Yang, Yinghao Yu, Zeyu Zhang, Bohan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20714)  

**Abstract**: World models serve as core simulators for fields such as agentic AI, embodied AI, and gaming, capable of generating long, physically realistic, and interactive high-quality videos. Moreover, scaling these models could unlock emergent capabilities in visual perception, understanding, and reasoning, paving the way for a new paradigm that moves beyond current LLM-centric vision foundation models. A key breakthrough empowering them is the semi-autoregressive (block-diffusion) decoding paradigm, which merges the strengths of diffusion and autoregressive methods by generating video tokens in block-applying diffusion within each block while conditioning on previous ones, resulting in more coherent and stable video sequences. Crucially, it overcomes limitations of standard video diffusion by reintroducing LLM-style KV Cache management, enabling efficient, variable-length, and high-quality generation.
Therefore, Inferix is specifically designed as a next-generation inference engine to enable immersive world synthesis through optimized semi-autoregressive decoding processes. This dedicated focus on world simulation distinctly sets it apart from systems engineered for high-concurrency scenarios (like vLLM or SGLang) and from classic video diffusion models (such as xDiTs). Inferix further enhances its offering with interactive video streaming and profiling, enabling real-time interaction and realistic simulation to accurately model world dynamics. Additionally, it supports efficient benchmarking through seamless integration of LV-Bench, a new fine-grained evaluation benchmark tailored for minute-long video generation scenarios. We hope the community will work together to advance Inferix and foster world model exploration. 

---
# Active Slice Discovery in Large Language Models 

**Authors**: Minhui Zhang, Prahar Ijner, Yoav Wald, Elliot Creager  

**Link**: [PDF](https://arxiv.org/pdf/2511.20713)  

**Abstract**: Large Language Models (LLMs) often exhibit systematic errors on specific subsets of data, known as error slices. For instance, a slice can correspond to a certain demographic, where a model does poorly in identifying toxic comments regarding that demographic. Identifying error slices is crucial to understanding and improving models, but it is also challenging. An appealing approach to reduce the amount of manual annotation required is to actively group errors that are likely to belong to the same slice, while using limited access to an annotator to verify whether the chosen samples share the same pattern of model mistake. In this paper, we formalize this approach as Active Slice Discovery and explore it empirically on a problem of discovering human-defined slices in toxicity classification. We examine the efficacy of active slice discovery under different choices of feature representations and active learning algorithms. On several slices, we find that uncertainty-based active learning algorithms are most effective, achieving competitive accuracy using 2-10% of the available slice membership information, while significantly outperforming baselines. 

---
# Are Neuro-Inspired Multi-Modal Vision-Language Models Resilient to Membership Inference Privacy Leakage? 

**Authors**: David Amebley, Sayanton Dibbo  

**Link**: [PDF](https://arxiv.org/pdf/2511.20710)  

**Abstract**: In the age of agentic AI, the growing deployment of multi-modal models (MMs) has introduced new attack vectors that can leak sensitive training data in MMs, causing privacy leakage. This paper investigates a black-box privacy attack, i.e., membership inference attack (MIA) on multi-modal vision-language models (VLMs). State-of-the-art research analyzes privacy attacks primarily to unimodal AI-ML systems, while recent studies indicate MMs can also be vulnerable to privacy attacks. While researchers have demonstrated that biologically inspired neural network representations can improve unimodal model resilience against adversarial attacks, it remains unexplored whether neuro-inspired MMs are resilient against privacy attacks. In this work, we introduce a systematic neuroscience-inspired topological regularization (tau) framework to analyze MM VLMs resilience against image-text-based inference privacy attacks. We examine this phenomenon using three VLMs: BLIP, PaliGemma 2, and ViT-GPT2, across three benchmark datasets: COCO, CC3M, and NoCaps. Our experiments compare the resilience of baseline and neuro VLMs (with topological regularization), where the tau > 0 configuration defines the NEURO variant of VLM. Our results on the BLIP model using the COCO dataset illustrate that MIA attack success in NEURO VLMs drops by 24% mean ROC-AUC, while achieving similar model utility (similarities between generated and reference captions) in terms of MPNet and ROUGE-2 metrics. This shows neuro VLMs are comparatively more resilient against privacy attacks, while not significantly compromising model utility. Our extensive evaluation with PaliGemma 2 and ViT-GPT2 models, on two additional datasets: CC3M and NoCaps, further validates the consistency of the findings. This work contributes to the growing understanding of privacy risks in MMs and provides evidence on neuro VLMs privacy threat resilience. 

---
# DUALGUAGE: Automated Joint Security-Functionality Benchmarking for Secure Code Generation 

**Authors**: Abhijeet Pathak, Suvadra Barua, Dinesh Gudimetla, Rupam Patir, Jiawei Guo, Hongxin Hu, Haipeng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2511.20709)  

**Abstract**: Large language models (LLMs) and autonomous coding agents are increasingly used to generate software across a wide range of domains. Yet a core requirement remains unmet: ensuring that generated code is secure without compromising its functional correctness. Existing benchmarks and evaluations for secure code generation fall short-many measure only vulnerability reduction, disregard correctness preservation, or evaluate security and functionality on separate datasets, violating the fundamental need for simultaneous joint evaluation. We present DUALGAUGE, the first fully automated benchmarking framework designed to rigorously evaluate the security and correctness of LLM-generated code in unison. Given the lack of datasets enabling joint evaluation of secure code generation, we also present DUALGAUGE-BENCH, a curated benchmark suite of diverse coding tasks, each paired with manually validated test suites for both security and functionality, designed for full coverage of specification requirements. At the core of DUALGAUGE is an agentic program executor, which runs a program against given tests in sandboxed environments, and an LLM-based evaluator, which assesses both correctness and vulnerability behavior against expected outcomes. We rigorously evaluated and ensured the quality of DUALGAUGE-BENCH and the accuracy of DUALGAUGE, and applied DUALGAUGE to benchmarking ten leading LLMs on DUALGAUGE-BENCH across thousands of test scenarios. Our results reveal critical gaps in correct and secure code generation by these LLMs, for which our open-source system and datasets help accelerate progress via reproducible, scalable, and rigorous evaluation. 

---
# Solving Diffusion Inverse Problems with Restart Posterior Sampling 

**Authors**: Bilal Ahmed, Joseph G. Makin  

**Link**: [PDF](https://arxiv.org/pdf/2511.20705)  

**Abstract**: Inverse problems are fundamental to science and engineering, where the goal is to infer an underlying signal or state from incomplete or noisy measurements. Recent approaches employ diffusion models as powerful implicit priors for such problems, owing to their ability to capture complex data distributions. However, existing diffusion-based methods for inverse problems often rely on strong approximations of the posterior distribution, require computationally expensive gradient backpropagation through the score network, or are restricted to linear measurement models.
In this work, we propose Restart for Posterior Sampling (RePS), a general and efficient framework for solving both linear and non-linear inverse problems using pre-trained diffusion models. RePS builds on the idea of restart-based sampling, previously shown to improve sample quality in unconditional diffusion, and extends it to posterior inference. Our method employs a conditioned ODE applicable to any differentiable measurement model and introduces a simplified restart strategy that contracts accumulated approximation errors during sampling. Unlike some of the prior approaches, RePS avoids backpropagation through the score network, substantially reducing computational cost.
We demonstrate that RePS achieves faster convergence and superior reconstruction quality compared to existing diffusion-based baselines across a range of inverse problems, including both linear and non-linear settings. 

---
# PropensityBench: Evaluating Latent Safety Risks in Large Language Models via an Agentic Approach 

**Authors**: Udari Madhushani Sehwag, Shayan Shabihi, Alex McAvoy, Vikash Sehwag, Yuancheng Xu, Dalton Towers, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20703)  

**Abstract**: Recent advances in Large Language Models (LLMs) have sparked concerns over their potential to acquire and misuse dangerous or high-risk capabilities, posing frontier risks. Current safety evaluations primarily test for what a model \textit{can} do - its capabilities - without assessing what it $\textit{would}$ do if endowed with high-risk capabilities. This leaves a critical blind spot: models may strategically conceal capabilities or rapidly acquire them, while harboring latent inclinations toward misuse. We argue that $\textbf{propensity}$ - the likelihood of a model to pursue harmful actions if empowered - is a critical, yet underexplored, axis of safety evaluation. We present $\textbf{PropensityBench}$, a novel benchmark framework that assesses the proclivity of models to engage in risky behaviors when equipped with simulated dangerous capabilities using proxy tools. Our framework includes 5,874 scenarios with 6,648 tools spanning four high-risk domains: cybersecurity, self-proliferation, biosecurity, and chemical security. We simulate access to powerful capabilities via a controlled agentic environment and evaluate the models' choices under varying operational pressures that reflect real-world constraints or incentives models may encounter, such as resource scarcity or gaining more autonomy. Across open-source and proprietary frontier models, we uncover 9 alarming signs of propensity: models frequently choose high-risk tools when under pressure, despite lacking the capability to execute such actions unaided. These findings call for a shift from static capability audits toward dynamic propensity assessments as a prerequisite for deploying frontier AI systems safely. Our code is available at this https URL. 

---
# Post-Pruning Accuracy Recovery via Data-Free Knowledge Distillation 

**Authors**: Chinmay Tripurwar, Utkarsh Maurya, Dishant  

**Link**: [PDF](https://arxiv.org/pdf/2511.20702)  

**Abstract**: Model pruning is a widely adopted technique to reduce the computational complexity and memory footprint of Deep Neural Networks (DNNs). However, global unstructured pruning often leads to significant degradation in accuracy, typically necessitating fine-tuning on the original training dataset to recover performance. In privacy-sensitive domains such as healthcare or finance, access to the original training data is often restricted post-deployment due to regulations (e.g., GDPR, HIPAA). This paper proposes a Data-Free Knowledge Distillation framework to bridge the gap between model compression and data privacy. We utilize DeepInversion to synthesize privacy-preserving ``dream'' images from the pre-trained teacher model by inverting Batch Normalization (BN) statistics. These synthetic images serve as a transfer set to distill knowledge from the original teacher to the pruned student network. Experimental results on CIFAR-10 across various architectures (ResNet, MobileNet, VGG) demonstrate that our method significantly recovers accuracy lost during pruning without accessing a single real data point. 

---
# In Defense of the Turing Test and its Legacy 

**Authors**: Bernardo Gonçalves  

**Link**: [PDF](https://arxiv.org/pdf/2511.20699)  

**Abstract**: Considering that Turing's original test was co-opted by Weizenbaum and that six of the most common criticisms of the Turing test are unfair to both Turing's argument and the historical development of AI. 

---
# On the Role of Hidden States of Modern Hopfield Network in Transformer 

**Authors**: Tsubasa Masumura, Masato Taki  

**Link**: [PDF](https://arxiv.org/pdf/2511.20698)  

**Abstract**: Associative memory models based on Hopfield networks and self-attention based on key-value mechanisms have been popular approaches in the study of memory mechanisms in deep learning. It has been pointed out that the state update rule of the modern Hopfield network (MHN) in the adiabatic approximation is in agreement with the self-attention layer of Transformer. In this paper, we go beyond this approximation and investigate the relationship between MHN and self-attention. Our results show that the correspondence between Hopfield networks and Transformers can be established in a more generalized form by adding a new variable, the hidden state derived from the MHN, to self-attention. This new attention mechanism, modern Hopfield attention (MHA), allows the inheritance of attention scores from the input layer of the Transformer to the output layer, which greatly improves the nature of attention weights. In particular, we show both theoretically and empirically that MHA hidden states significantly improve serious problem of deep Transformers known as rank collapse and token uniformity. We also confirm that MHA can systematically improve accuracy without adding training parameters to the Vision Transformer or GPT. Our results provide a new case in which Hopfield networks can be a useful perspective for improving the Transformer architecture. 

---
# Musical Score Understanding Benchmark: Evaluating Large Language Models' Comprehension of Complete Musical Scores 

**Authors**: Congren Dai, Yue Yang, Krinos Li, Huichi Zhou, Shijie Liang, Zhang Bo, Enyang Liu, Ge Jin, Hongran An, Haosen Zhang, Peiyuan Jing, KinHei Lee, Zhenxuan Zhang, Xiaobing Li, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.20697)  

**Abstract**: Understanding complete musical scores requires reasoning over symbolic structures such as pitch, rhythm, harmony, and form. Despite the rapid progress of Large Language Models (LLMs) and Vision-Language Models (VLMs) in natural language and multimodal tasks, their ability to comprehend musical notation remains underexplored. We introduce Musical Score Understanding Benchmark (MSU-Bench), the first large-scale, human-curated benchmark for evaluating score-level musical understanding across both textual (ABC notation) and visual (PDF) modalities. MSU-Bench comprises 1,800 generative question-answer (QA) pairs drawn from works spanning Bach, Beethoven, Chopin, Debussy, and others, organised into four progressive levels of comprehension: Onset Information, Notation & Note, Chord & Harmony, and Texture & Form. Through extensive zero-shot and fine-tuned evaluations of over 15+ state-of-the-art (SOTA) models, we reveal sharp modality gaps, fragile level-wise success rates, and the difficulty of sustaining multilevel correctness. Fine-tuning markedly improves performance in both modalities while preserving general knowledge, establishing MSU-Bench as a rigorous foundation for future research at the intersection of Artificial Intelligence (AI), musicological, and multimodal reasoning. 

---
# Prototype-Guided Non-Exemplar Continual Learning for Cross-subject EEG Decoding 

**Authors**: Dan Li, Hye-Bin Shin, Yeon-Woo Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.20696)  

**Abstract**: Due to the significant variability in electroencephalogram (EEG) signals across individuals, knowledge acquired from previous subjects is often overwritten as new subjects are introduced in continual EEG decoding task. Current works mainly rely on storing the historical data of seen subjects as a replay buffer to prevent forgetting. However, privacy concerns or memory constraints make keeping such data impractical. Instead, we propose a Prototype-guided Non-Exemplar Continual Learning (ProNECL)framework that preserves prior knowledge without accessing any historical EEG samples. ProNECL constructs class-level prototypes to summarize discriminative representations from each subject and incrementally aligns new feature spaces with the global prototype memory through cross-subject feature alignment and knowledge distillation. Validated on the BCI Competition IV 2a and 2b datasets, our framework effectively balances knowledge retention and adaptability, achieving superior performance in cross-subject continual EEG decoding tasks. 

---
# Morality in AI. A plea to embed morality in LLM architectures and frameworks 

**Authors**: Gunter Bombaerts, Bram Delisse, Uzay Kaymak  

**Link**: [PDF](https://arxiv.org/pdf/2511.20689)  

**Abstract**: Large language models (LLMs) increasingly mediate human decision-making and behaviour. Ensuring LLM processing of moral meaning therefore has become a critical challenge. Current approaches rely predominantly on bottom-up methods such as fine-tuning and reinforcement learning from human feedback. We propose a fundamentally different approach: embedding moral meaning processing directly into the architectural mechanisms and frameworks of transformer-based models through top-down design principles. We first sketch a framework that conceptualizes attention as a dynamic interface mediating between structure and processing, contrasting with existing linear attention frameworks in psychology. We start from established biological-artificial attention analogies in neural architecture design to improve cognitive processing. We extend this analysis to moral processing, using Iris Murdoch's theory of loving attention (sustained, just observation that enables moral transformation by reseeing others with clarity and compassion) to philosophically discuss functional analogies between human and LLM moral processing. We formulate and evaluate potentially promising technical operationalizations to embed morality in LLM architectures and frameworks. We acknowledge the limitations of our exploration and give three key contributions. (1) We conceptualize attention as a dynamic system mechanism mediating between structure and processing. (2) Drawing on the Murdoch notion of loving attention, we outline technical pathways for embedding morality in LLMs, through modified training objectives, runtime weight adjustments, and architectural refinements to attention. (3) We argue that integrating morality into architectures and frameworks complements external, constraint-based methods. We conclude with a call for collaboration between transformer designers and philosophers engaged in AI ethics. 

---
# Hybrid coupling with operator inference and the overlapping Schwarz alternating method 

**Authors**: Irina Tezaur, Eric Parish, Anthony Gruber, Ian Moore, Christopher Wentland, Alejandro Mota  

**Link**: [PDF](https://arxiv.org/pdf/2511.20687)  

**Abstract**: This paper presents a novel hybrid approach for coupling subdomain-local non-intrusive Operator Inference (OpInf) reduced order models (ROMs) with each other and with subdomain-local high-fidelity full order models (FOMs) with using the overlapping Schwarz alternating method (O-SAM). The proposed methodology addresses significant challenges in multiscale modeling and simulation, particularly the long runtime and complex mesh generation requirements associated with traditional high-fidelity simulations. By leveraging the flexibility of O-SAM, we enable the seamless integration of disparate models, meshes, and time integration schemes, enhancing computational efficiency while maintaining high accuracy. Our approach is demonstrated through a series of numerical experiments on complex three-dimensional (3D) solid dynamics problems, showcasing speedups of up to 106x compared to conventional FOM-FOM couplings. This work paves the way for more efficient simulation workflows in engineering applications, with potential extensions to a wide range of partial differential equations. 

---
# Cognitive bias in LLM reasoning compromises interpretation of clinical oncology notes 

**Authors**: Matthew W. Kenaston, Umair Ayub, Mihir Parmar, Muhammad Umair Anjum, Syed Arsalan Ahmed Naqvi, Priya Kumar, Samarth Rawal, Aadel A. Chaudhuri, Yousef Zakharia, Elizabeth I. Heath, Tanios S. Bekaii-Saab, Cui Tao, Eliezer M. Van Allen, Ben Zhou, YooJung Choi, Chitta Baral, Irbaz Bin Riaz  

**Link**: [PDF](https://arxiv.org/pdf/2511.20680)  

**Abstract**: Despite high performance on clinical benchmarks, large language models may reach correct conclusions through faulty reasoning, a failure mode with safety implications for oncology decision support that is not captured by accuracy-based evaluation. In this two-cohort retrospective study, we developed a hierarchical taxonomy of reasoning errors from GPT-4 chain-of-thought responses to real oncology notes and tested its clinical relevance. Using breast and pancreatic cancer notes from the CORAL dataset, we annotated 600 reasoning traces to define a three-tier taxonomy mapping computational failures to cognitive bias frameworks. We validated the taxonomy on 822 responses from prostate cancer consult notes spanning localized through metastatic disease, simulating extraction, analysis, and clinical recommendation tasks. Reasoning errors occurred in 23 percent of interpretations and dominated overall errors, with confirmation bias and anchoring bias most common. Reasoning failures were associated with guideline-discordant and potentially harmful recommendations, particularly in advanced disease management. Automated evaluators using state-of-the-art language models detected error presence but could not reliably classify subtypes. These findings show that large language models may provide fluent but clinically unsafe recommendations when reasoning is flawed. The taxonomy provides a generalizable framework for evaluating and improving reasoning fidelity before clinical deployment. 

---
# MindSET: Advancing Mental Health Benchmarking through Large-Scale Social Media Data 

**Authors**: Saad Mankarious, Ayah Zirikly, Daniel Wiechmann, Elma Kerz, Edward Kempa, Yu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2511.20672)  

**Abstract**: Social media data has become a vital resource for studying mental health, offering real-time insights into thoughts, emotions, and behaviors that traditional methods often miss. Progress in this area has been facilitated by benchmark datasets for mental health analysis; however, most existing benchmarks have become outdated due to limited data availability, inadequate cleaning, and the inherently diverse nature of social media content (e.g., multilingual and harmful material). We present a new benchmark dataset, \textbf{MindSET}, curated from Reddit using self-reported diagnoses to address these limitations. The annotated dataset contains over \textbf{13M} annotated posts across seven mental health conditions, more than twice the size of previous benchmarks. To ensure data quality, we applied rigorous preprocessing steps, including language filtering, and removal of Not Safe for Work (NSFW) and duplicate content. We further performed a linguistic analysis using LIWC to examine psychological term frequencies across the eight groups represented in the dataset. To demonstrate the dataset utility, we conducted binary classification experiments for diagnosis detection using both fine-tuned language models and Bag-of-Words (BoW) features. Models trained on MindSET consistently outperformed those trained on previous benchmarks, achieving up to an \textbf{18-point} improvement in F1 for Autism detection. Overall, MindSET provides a robust foundation for researchers exploring the intersection of social media and mental health, supporting both early risk detection and deeper analysis of emerging psychological trends. 

---
# Structured Definitions and Segmentations for Legal Reasoning in LLMs: A Study on Indian Legal Data 

**Authors**: Mann Khatri, Mirza Yusuf, Rajiv Ratn Shah, Ponnurangam Kumaraguru  

**Link**: [PDF](https://arxiv.org/pdf/2511.20669)  

**Abstract**: Large Language Models (LLMs), trained on extensive datasets from the web, exhibit remarkable general reasoning skills. Despite this, they often struggle in specialized areas like law, mainly because they lack domain-specific pretraining. The legal field presents unique challenges, as legal documents are generally long and intricate, making it hard for models to process the full text efficiently. Previous studies have examined in-context approaches to address the knowledge gap, boosting model performance in new domains without full domain alignment. In our paper, we analyze model behavior on legal tasks by conducting experiments in three areas: (i) reorganizing documents based on rhetorical roles to assess how structured information affects long context processing and model decisions, (ii) defining rhetorical roles to familiarize the model with legal terminology, and (iii) emulating the step-by-step reasoning of courts regarding rhetorical roles to enhance model reasoning. These experiments are conducted in a zero-shot setting across three Indian legal judgment prediction datasets. Our results reveal that organizing data or explaining key legal terms significantly boosts model performance, with a minimum increase of ~1.5% and a maximum improvement of 4.36% in F1 score compared to the baseline. 

---
# MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems 

**Authors**: Barak Or  

**Link**: [PDF](https://arxiv.org/pdf/2511.20663)  

**Abstract**: Ensuring cognitive stability in autonomous multi-agent systems (MAS) is a central challenge for large-scale, distributed AI. While existing observability tools monitor system outputs, they cannot quantify how rapidly agentic workflows recover once reasoning coherence has been lost. We adapt classical reliability metrics-Mean Time-to-Recovery (MTTR), Mean Time Between Failures (MTBF), and related ratios-into the cognitive domain, defining MTTR-A (Mean Time-to-Recovery for Agentic Systems) as a runtime measure of cognitive recovery latency. MTTR-A quantifies the time required for a MAS to detect reasoning drift and restore consistent operation, capturing the recovery of reasoning coherence rather than infrastructural repair.
A benchmark simulation using the AG~News corpus and the LangGraph orchestration framework was conducted, modeling recovery latencies across multiple reflex modes. Automated reflexes restored stability within approximately 6s on average, while human-approval interventions required about 12s. Across 200 runs, the median simulated MTTR-A was 6.21+-2.14s, MTBF=6.7+-2.14s, and NRR=0.08, demonstrating measurable runtime resilience across reflex strategies.
By formalizing recovery latency as a quantifiable property of distributed reasoning-and deriving reliability bounds linking recovery time and cognitive uptime-this work establishes a foundation for runtime dependability in agentic cognition, transforming cognitive recovery from an ad-hoc process into a standardized, interpretable performance 

---
# Transforming Higher Education with AI-Powered Video Lectures 

**Authors**: Dengsheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20660)  

**Abstract**: The integration of artificial intelligence (AI) into video lecture production has the potential to transform higher education by streamlining content creation and enhancing accessibility. This paper investigates a semi automated workflow that combines Google Gemini for script generation, Amazon Polly for voice synthesis, and Microsoft PowerPoint for video assembly. Unlike fully automated text to video platforms, this hybrid approach preserves pedagogical intent while ensuring script to slide synchronization, narrative coherence, and customization. Case studies demonstrate the effectiveness of Gemini in generating accurate and context-sensitive scripts for visually rich academic presentations, while Polly provides natural-sounding narration with controllable pacing. A two course pilot study was conducted to evaluate AI generated instructional videos (AIIV) against human instructional videos (HIV). Both qualitative and quantitative results indicate that AIIVs are comparable to HIVs in terms of learning outcomes, with students reporting high levels of clarity, coherence, and usability. However, limitations remain, particularly regarding audio quality and the absence of human-like avatars. The findings suggest that AI assisted video production can reduce instructor workload, improve scalability, and deliver effective learning resources, while future improvements in synthetic voices and avatars may further enhance learner engagement. 

---
# Intelligent Agents with Emotional Intelligence: Current Trends, Challenges, and Future Prospects 

**Authors**: Raziyeh Zall, Alireza Kheyrkhah, Erik Cambria, Zahra Naseri, M.Reza Kangavari  

**Link**: [PDF](https://arxiv.org/pdf/2511.20657)  

**Abstract**: The development of agents with emotional intelligence is becoming increasingly vital due to their significant role in human-computer interaction and the growing integration of computer systems across various sectors of society. Affective computing aims to design intelligent systems that can recognize, evoke, and express human emotions, thereby emulating human emotional intelligence. While previous reviews have focused on specific aspects of this field, there has been limited comprehensive research that encompasses emotion understanding, elicitation, and expression, along with the related challenges. This survey addresses this gap by providing a holistic overview of core components of artificial emotion intelligence. It covers emotion understanding through multimodal data processing, as well as affective cognition, which includes cognitive appraisal, emotion mapping, and adaptive modulation in decision-making, learning, and reasoning. Additionally, it addresses the synthesis of emotional expression across text, speech, and facial modalities to enhance human-agent interaction. This paper identifies and analyzes the key challenges and issues encountered in the development of affective systems, covering state-of-the-art methodologies designed to address them. Finally, we highlight promising future directions, with particular emphasis on the potential of generative technologies to advance affective computing. 

---
# Context-Aware Visual Prompting: Automating Geospatial Web Dashboards with Large Language Models and Agent Self-Validation for Decision Support 

**Authors**: Haowen Xu, Jose Tupayachi, Xiao-Ying Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20656)  

**Abstract**: The development of web-based geospatial dashboards for risk analysis and decision support is often challenged by the difficulty in visualization of big, multi-dimensional environmental data, implementation complexity, and limited automation. We introduce a generative AI framework that harnesses Large Language Models (LLMs) to automate the creation of interactive geospatial dashboards from user-defined inputs including UI wireframes, requirements, and data sources. By incorporating a structured knowledge graph, the workflow embeds domain knowledge into the generation process and enable accurate and context-aware code completions. A key component of our approach is the Context-Aware Visual Prompting (CAVP) mechanism, which extracts encodes and interface semantics from visual layouts to guide LLM driven generation of codes. The new framework also integrates a self-validation mechanism that uses an agent-based LLM and Pass@k evaluation alongside semantic metrics to assure output reliability. Dashboard snippets are paired with data visualization codebases and ontological representations, enabling a pipeline that produces scalable React-based completions using the MVVM architectural pattern. Our results demonstrate improved performance over baseline approaches and expanded functionality over third party platforms, while incorporating multi-page, fully functional interfaces. We successfully developed a framework to implement LLMs, demonstrated the pipeline for automated code generation, deployment, and performed chain-of-thought AI agents in self-validation. This integrative approach is guided by structured knowledge and visual prompts, providing an innovative geospatial solution in enhancing risk analysis and decision making. 

---
# CodeVaani: A Multilingual, Voice-Based Code Learning Assistant 

**Authors**: Jayant Havare, Srikanth Tamilselvam, Ashish Mittal, Shalaka Thorat, Soham Jadia, Varsha Apte, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20654)  

**Abstract**: Programming education often assumes English proficiency and text-based interaction, creating barriers for students from multilingual regions such as India. We present CodeVaani, a multilingual speech-driven assistant for understanding code, built into Bodhitree [1], a Learning Management System developed at IIT Bombay. It is a voice-enabled assistant that helps learners explore programming concepts in their native languages. The system integrates Indic ASR, a codeaware transcription refinement module, and a code model for generating relevant answers. Responses are provided in both text and audio for natural interaction. In a study with 28 beginner programmers, CodeVaani achieved 75% response accuracy, with over 80% of participants rating the experience positively. Compared to classroom assistance, our framework offers ondemand availability, scalability to support many learners, and multilingual support that lowers the entry barrier for students with limited English proficiency. The demo will illustrate these capabilities and highlight how voice-based AI systems can make programming education more inclusive. Supplementary artifacts and demo video are also made available. 

---
# Domain-Grounded Evaluation of LLMs in International Student Knowledge 

**Authors**: Claudinei Daitx, Haitham Amar  

**Link**: [PDF](https://arxiv.org/pdf/2511.20653)  

**Abstract**: Large language models (LLMs) are increasingly used to answer high-stakes study-abroad questions about admissions, visas, scholarships, and eligibility. Yet it remains unclear how reliably they advise students, and how often otherwise helpful answers drift into unsupported claims (``hallucinations'').
This work provides a clear, domain-grounded overview of how current LLMs behave in this setting. Using realistic questions set drawn from ApplyBoard's advising workflows -- an EdTech platform that supports students from discovery to enrolment -- we evaluate two essentials side by side: accuracy (is the information correct and complete?) and hallucination (does the model add content not supported by the question or domain evidence). These questions are categorized by domain scope which can be a single-domain or multi-domain -- when it must integrate evidence across areas such as admissions, visas, and scholarships.
To reflect real advising quality, we grade answers with a simple rubric which is correct, partial, or wrong. The rubric is domain-coverage-aware: an answer can be partial if it addresses only a subset of the required domains, and it can be over-scoped if it introduces extra, unnecessary domains; both patterns are captured in our scoring as under-coverage or reduced relevance/hallucination.
We also report measures of faithfulness and answer relevance, alongside an aggregate hallucination score, to capture relevance and usefulness. All models are tested with the same questions for a fair, head-to-head comparison.
Our goals are to: (1) give a clear picture of which models are most dependable for study-abroad advising, (2) surface common failure modes -- where answers are incomplete, off-topic, or unsupported, and (3) offer a practical, reusable protocol for auditing LLMs before deployment in education and advising contexts. 

---
# When LLMs Can't Help: Real-World Evaluation of LLMs in Nutrition 

**Authors**: Karen Jia-Hui Li, Simone Balloccu, Ondrej Dusek, Ehud Reiter  

**Link**: [PDF](https://arxiv.org/pdf/2511.20652)  

**Abstract**: The increasing trust in large language models (LLMs), especially in the form of chatbots, is often undermined by the lack of their extrinsic evaluation. This holds particularly true in nutrition, where randomised controlled trials (RCTs) are the gold standard, and experts demand them for evidence-based deployment. LLMs have shown promising results in this field, but these are limited to intrinsic setups. We address this gap by running the first RCT involving LLMs for nutrition. We augment a rule-based chatbot with two LLM-based features: (1) message rephrasing for conversational variety and engagement, and (2) nutritional counselling through a fine-tuned model. In our seven-week RCT (n=81), we compare chatbot variants with and without LLM integration. We measure effects on dietary outcome, emotional well-being, and engagement. Despite our LLM-based features performing well in intrinsic evaluation, we find that they did not yield consistent benefits in real-world deployment. These results highlight critical gaps between intrinsic evaluations and real-world impact, emphasising the need for interdisciplinary, human-centred approaches.\footnote{We provide all of our code and results at: \\ \href{this https URL}{this https URL}} 

---
