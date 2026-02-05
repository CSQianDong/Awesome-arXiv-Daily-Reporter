# Fluid Representations in Reasoning Models 

**Authors**: Dmitrii Kharlapenko, Alessandro Stolfo, Arthur Conmy, Mrinmaya Sachan, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2602.04843)  

**Abstract**: Reasoning language models, which generate long chains of thought, dramatically outperform non-reasoning language models on abstract problems. However, the internal model mechanisms that allow this superior performance remain poorly understood. We present a mechanistic analysis of how QwQ-32B - a model specifically trained to produce extensive reasoning traces - process abstract structural information. On Mystery Blocksworld - a semantically obfuscated planning domain - we find that QwQ-32B gradually improves its internal representation of actions and concepts during reasoning. The model develops abstract encodings that focus on structure rather than specific action names. Through steering experiments, we establish causal evidence that these adaptations improve problem solving: injecting refined representations from successful traces boosts accuracy, while symbolic representations can replace many obfuscated encodings with minimal performance loss. We find that one of the factors driving reasoning model performance is in-context refinement of token representations, which we dub Fluid Reasoning Representations. 

---
# Group-Evolving Agents: Open-Ended Self-Improvement via Experience Sharing 

**Authors**: Zhaotian Weng, Antonis Antoniades, Deepak Nathani, Zhen Zhang, Xiao Pu, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04837)  

**Abstract**: Open-ended self-improving agents can autonomously modify their own structural designs to advance their capabilities and overcome the limits of pre-defined architectures, thus reducing reliance on human intervention. We introduce Group-Evolving Agents (GEA), a new paradigm for open-ended self-improvements, which treats a group of agents as the fundamental evolutionary unit, enabling explicit experience sharing and reuse within the group throughout evolution. Unlike existing open-ended self-evolving paradigms that adopt tree-structured evolution, GEA overcomes the limitation of inefficient utilization of exploratory diversity caused by isolated evolutionary branches. We evaluate GEA on challenging coding benchmarks, where it significantly outperforms state-of-the-art self-evolving methods (71.0% vs. 56.7% on SWE-bench Verified, 88.3% vs. 68.3% on Polyglot) and matches or exceeds top human-designed agent frameworks (71.8% and 52.0% on two benchmarks, respectively). Analysis reveals that GEA more effectively converts early-stage exploratory diversity into sustained, long-term progress, achieving stronger performance under the same number of evolved agents. Furthermore, GEA exhibits consistent transferability across different coding models and greater robustness, fixing framework-level bugs in 1.4 iterations on average, versus 5 for self-evolving methods. 

---
# Are AI Capabilities Increasing Exponentially? A Competing Hypothesis 

**Authors**: Haosen Ge, Hamsa Bastani, Osbert Bastani  

**Link**: [PDF](https://arxiv.org/pdf/2602.04836)  

**Abstract**: Rapidly increasing AI capabilities have substantial real-world consequences, ranging from AI safety concerns to labor market consequences. The Model Evaluation & Threat Research (METR) report argues that AI capabilities have exhibited exponential growth since 2019. In this note, we argue that the data does not support exponential growth, even in shorter-term horizons. Whereas the METR study claims that fitting sigmoid/logistic curves results in inflection points far in the future, we fit a sigmoid curve to their current data and find that the inflection point has already passed. In addition, we propose a more complex model that decomposes AI capabilities into base and reasoning capabilities, exhibiting individual rates of improvement. We prove that this model supports our hypothesis that AI capabilities will exhibit an inflection point in the near future. Our goal is not to establish a rigorous forecast of our own, but to highlight the fragility of existing forecasts of exponential growth. 

---
# Agentic AI in Healthcare & Medicine: A Seven-Dimensional Taxonomy for Empirical Evaluation of LLM-based Agents 

**Authors**: Shubham Vatsal, Harsh Dubey, Aditi Singh  

**Link**: [PDF](https://arxiv.org/pdf/2602.04813)  

**Abstract**: Large Language Model (LLM)-based agents that plan, use tools and act has begun to shape healthcare and medicine. Reported studies demonstrate competence on various tasks ranging from EHR analysis and differential diagnosis to treatment planning and research workflows. Yet the literature largely consists of overviews which are either broad surveys or narrow dives into a single capability (e.g., memory, planning, reasoning), leaving healthcare work without a common frame. We address this by reviewing 49 studies using a seven-dimensional taxonomy: Cognitive Capabilities, Knowledge Management, Interaction Patterns, Adaptation & Learning, Safety & Ethics, Framework Typology and Core Tasks & Subtasks with 29 operational sub-dimensions. Using explicit inclusion and exclusion criteria and a labeling rubric (Fully Implemented, Partially Implemented, Not Implemented), we map each study to the taxonomy and report quantitative summaries of capability prevalence and co-occurrence patterns. Our empirical analysis surfaces clear asymmetries. For instance, the External Knowledge Integration sub-dimension under Knowledge Management is commonly realized (~76% Fully Implemented) whereas Event-Triggered Activation sub-dimenison under Interaction Patterns is largely absent (~92% Not Implemented) and Drift Detection & Mitigation sub-dimension under Adaptation & Learning is rare (~98% Not Implemented). Architecturally, Multi-Agent Design sub-dimension under Framework Typology is the dominant pattern (~82% Fully Implemented) while orchestration layers remain mostly partial. Across Core Tasks & Subtasks, information centric capabilities lead e.g., Medical Question Answering & Decision Support and Benchmarking & Simulation, while action and discovery oriented areas such as Treatment Planning & Prescription still show substantial gaps (~59% Not Implemented). 

---
# WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning 

**Authors**: Zelai Xu, Zhexuan Xu, Ruize Zhang, Chunyang Zhu, Shi Yu, Weilin Liu, Quanlu Zhang, Wenbo Ding, Chao Yu, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04634)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have largely focused on depth scaling, where a single agent solves long-horizon problems with multi-turn reasoning and tool use. However, as tasks grow broader, the key bottleneck shifts from individual competence to organizational capability. In this work, we explore a complementary dimension of width scaling with multi-agent systems to address broad information seeking. Existing multi-agent systems often rely on hand-crafted workflows and turn-taking interactions that fail to parallelize work effectively. To bridge this gap, we propose WideSeek-R1, a lead-agent-subagent framework trained via multi-agent reinforcement learning (MARL) to synergize scalable orchestration and parallel execution. By utilizing a shared LLM with isolated contexts and specialized tools, WideSeek-R1 jointly optimizes the lead agent and parallel subagents on a curated dataset of 20k broad information-seeking tasks. Extensive experiments show that WideSeek-R1-4B achieves an item F1 score of 40.0% on the WideSearch benchmark, which is comparable to the performance of single-agent DeepSeek-R1-671B. Furthermore, WideSeek-R1-4B exhibits consistent performance gains as the number of parallel subagents increases, highlighting the effectiveness of width scaling. 

---
# Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration 

**Authors**: Jiaheng Liu, Yuanxing Zhang, Shihao Li, Xinping Lei  

**Link**: [PDF](https://arxiv.org/pdf/2602.04575)  

**Abstract**: For the past decade, the trajectory of generative artificial intelligence (AI) has been dominated by a model-centric paradigm driven by scaling laws. Despite significant leaps in visual fidelity, this approach has encountered a ``usability ceiling'' manifested as the Intent-Execution Gap (i.e., the fundamental disparity between a creator's high-level intent and the stochastic, black-box nature of current single-shot models). In this paper, inspired by the Vibe Coding, we introduce the \textbf{Vibe AIGC}, a new paradigm for content generation via agentic orchestration, which represents the autonomous synthesis of hierarchical multi-agent workflows.
Under this paradigm, the user's role transcends traditional prompt engineering, evolving into a Commander who provides a Vibe, a high-level representation encompassing aesthetic preferences, functional logic, and etc. A centralized Meta-Planner then functions as a system architect, deconstructing this ``Vibe'' into executable, verifiable, and adaptive agentic pipelines. By transitioning from stochastic inference to logical orchestration, Vibe AIGC bridges the gap between human imagination and machine execution. We contend that this shift will redefine the human-AI collaborative economy, transforming AI from a fragile inference engine into a robust system-level engineering partner that democratizes the creation of complex, long-horizon digital assets. 

---
# From Competition to Collaboration: Designing Sustainable Mechanisms Between LLMs and Online Forums 

**Authors**: Niv Fono, Yftah Ziser, Omer Ben-Porat  

**Link**: [PDF](https://arxiv.org/pdf/2602.04572)  

**Abstract**: While Generative AI (GenAI) systems draw users away from (Q&A) forums, they also depend on the very data those forums produce to improve their performance. Addressing this paradox, we propose a framework of sequential interaction, in which a GenAI system proposes questions to a forum that can publish some of them. Our framework captures several intricacies of such a collaboration, including non-monetary exchanges, asymmetric information, and incentive misalignment. We bring the framework to life through comprehensive, data-driven simulations using real Stack Exchange data and commonly used LLMs. We demonstrate the incentive misalignment empirically, yet show that players can achieve roughly half of the utility in an ideal full-information scenario. Our results highlight the potential for sustainable collaboration that preserves effective knowledge sharing between AI systems and human knowledge platforms. 

---
# ReThinker: Scientific Reasoning by Rethinking with Guided Reflection and Confidence Control 

**Authors**: Zhentao Tang, Yuqi Cui, Shixiong Kai, Wenqian Zhao, Ke Ye, Xing Li, Anxin Tian, Zehua Pei, Hui-Ling Zhen, Shoubo Hu, Xiaoguang Li, Yunhe Wang, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2602.04496)  

**Abstract**: Expert-level scientific reasoning remains challenging for large language models, particularly on benchmarks such as Humanity's Last Exam (HLE), where rigid tool pipelines, brittle multi-agent coordination, and inefficient test-time scaling often limit performance. We introduce ReThinker, a confidence-aware agentic framework that orchestrates retrieval, tool use, and multi-agent reasoning through a stage-wise Solver-Critic-Selector architecture. Rather than following a fixed pipeline, ReThinker dynamically allocates computation based on model confidence, enabling adaptive tool invocation, guided multi-dimensional reflection, and robust confidence-weighted selection. To support scalable training without human annotation, we further propose a reverse data synthesis pipeline and an adaptive trajectory recycling strategy that transform successful reasoning traces into high-quality supervision. Experiments on HLE, GAIA, and XBench demonstrate that ReThinker consistently outperforms state-of-the-art foundation models with tools and existing deep research systems, achieving state-of-the-art results on expert-level reasoning tasks. 

---
# Digital Twins & ZeroConf AI: Structuring Automated Intelligent Pipelines for Industrial Applications 

**Authors**: Marco Picone, Fabio Turazza, Matteo Martinelli, Marco Mamei  

**Link**: [PDF](https://arxiv.org/pdf/2602.04385)  

**Abstract**: The increasing complexity of Cyber-Physical Systems (CPS), particularly in the industrial domain, has amplified the challenges associated with the effective integration of Artificial Intelligence (AI) and Machine Learning (ML) techniques. Fragmentation across IoT and IIoT technologies, manifested through diverse communication protocols, data formats and device capabilities, creates a substantial gap between low-level physical layers and high-level intelligent functionalities. Recently, Digital Twin (DT) technology has emerged as a promising solution, offering structured, interoperable and semantically rich digital representations of physical assets. Current approaches are often siloed and tightly coupled, limiting scalability and reuse of AI functionalities. This work proposes a modular and interoperable solution that enables seamless AI pipeline integration into CPS by minimizing configuration and decoupling the roles of DTs and AI components. We introduce the concept of Zero Configuration (ZeroConf) AI pipelines, where DTs orchestrate data management and intelligent augmentation. The approach is demonstrated in a MicroFactory scenario, showing support for concurrent ML models and dynamic data processing, effectively accelerating the deployment of intelligent services in complex industrial settings. 

---
# From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning for Embodied Agents 

**Authors**: SeungWon Seo, SooBin Lim, SeongRae Noh, Haneul Kim, HyeongYeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04326)  

**Abstract**: Embodied agents operating in multi-agent, partially observable, and decentralized environments must plan and act despite pervasive uncertainty about hidden objects and collaborators' intentions. Recent advances in applying Large Language Models (LLMs) to embodied agents have addressed many long-standing challenges, such as high-level goal decomposition and online adaptation. Yet, uncertainty is still primarily mitigated through frequent inter-agent communication. This incurs substantial token and time costs, and can disrupt established workflows, when human partners are involved. We introduce PCE, a Planner-Composer-Evaluator framework that converts the fragmented assumptions latent in LLM reasoning traces into a structured decision tree. Internal nodes encode environment assumptions and leaves map to actions; each path is then scored by scenario likelihood, goal-directed gain, and execution cost to guide rational action selection without heavy communication. Across two challenging multi-agent benchmarks (C-WAH and TDW-MAT) and three diverse LLM backbones, PCE consistently outperforms communication-centric baselines in success rate and task efficiency while showing comparable token usage. Ablation results indicate that the performance gains obtained by scaling model capacity or reasoning depth persist even when PCE is applied, while PCE consistently raises the baseline across both capacity and reasoning-depth scales, confirming that structured uncertainty handling complements both forms of scaling. A user study further demonstrates that PCE produces communication patterns that human partners perceive as more efficient and trustworthy. Together, these results establish a principled route for turning latent LLM assumptions into reliable strategies for uncertainty-aware planning. 

---
# Agent-Omit: Training Efficient LLM Agents for Adaptive Thought and Observation Omission via Agentic Reinforcement Learning 

**Authors**: Yansong Ning, Jun Fang, Naiqiang Tan, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.04284)  

**Abstract**: Managing agent thought and observation during multi-turn agent-environment interactions is an emerging strategy to improve agent efficiency. However, existing studies treat the entire interaction trajectories equally, overlooking the thought necessity and observation utility varies across turns. To this end, we first conduct quantitative investigations into how thought and observation affect agent effectiveness and efficiency. Based on our findings, we propose Agent-Omit, a unified training framework that empowers LLM agents to adaptively omit redundant thoughts and observations. Specifically, we first synthesize a small amount of cold-start data, including both single-turn and multi-turn omission scenarios, to fine-tune the agent for omission behaviors. Furthermore, we introduce an omit-aware agentic reinforcement learning approach, incorporating a dual sampling mechanism and a tailored omission reward to incentivize the agent's adaptive omission capability. Theoretically, we prove that the deviation of our omission policy is upper-bounded by KL-divergence. Experimental results on five agent benchmarks show that our constructed Agent-Omit-8B could obtain performance comparable to seven frontier LLM agent, and achieve the best effectiveness-efficiency trade-off than seven efficient LLM agents methods. Our code and data are available at this https URL. 

---
# Empirical-MCTS: Continuous Agent Evolution via Dual-Experience Monte Carlo Tree Search 

**Authors**: Hao Lu, Haoyuan Huang, Yulin Zhou, Chen Li, Ningxin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.04248)  

**Abstract**: Inference-time scaling strategies, particularly Monte Carlo Tree Search (MCTS), have significantly enhanced the reasoning capabilities of Large Language Models (LLMs). However, current approaches remain predominantly stateless, discarding successful reasoning patterns after each problem instance and failing to mimic the empirical accumulation of wisdom characteristic of human problem-solving. To bridge this gap, we introduce Empirical-MCTS, a dual-loop framework that transforms stateless search into a continuous, non-parametric learning process. The framework unifies local exploration with global memory optimization through two novel mechanisms: Pairwise-Experience-Evolutionary Meta-Prompting (PE-EMP) and a Memory Optimization Agent. PE-EMP functions as a reflexive optimizer within the local search, utilizing pairwise feedback to dynamically synthesize adaptive criteria and evolve meta-prompts (system prompts) in real-time. Simultaneously, the Memory Optimization Agent manages a global repository as a dynamic policy prior, employing atomic operations to distill high-quality insights across problems. Extensive evaluations on complex reasoning benchmarks, including AIME25, ARC-AGI-2, and MathArena Apex, demonstrate that Empirical-MCTS significantly outperforms both stateless MCTS strategies and standalone experience-driven agents. These results underscore the critical necessity of coupling structured search with empirical accumulation for mastering complex, open-ended reasoning tasks. 

---
# InterPReT: Interactive Policy Restructuring and Training Enable Effective Imitation Learning from Laypersons 

**Authors**: Feiyu Gavin Zhu, Jean Oh, Reid Simmons  

**Link**: [PDF](https://arxiv.org/pdf/2602.04213)  

**Abstract**: Imitation learning has shown success in many tasks by learning from expert demonstrations. However, most existing work relies on large-scale demonstrations from technical professionals and close monitoring of the training process. These are challenging for a layperson when they want to teach the agent new skills. To lower the barrier of teaching AI agents, we propose Interactive Policy Restructuring and Training (InterPReT), which takes user instructions to continually update the policy structure and optimize its parameters to fit user demonstrations. This enables end-users to interactively give instructions and demonstrations, monitor the agent's performance, and review the agent's decision-making strategies. A user study (N=34) on teaching an AI agent to drive in a racing game confirms that our approach yields more robust policies without impairing system usability, compared to a generic imitation learning baseline, when a layperson is responsible for both giving demonstrations and determining when to stop. This shows that our method is more suitable for end-users without much technical background in machine learning to train a dependable policy 

---
# Steering LLMs via Scalable Interactive Oversight 

**Authors**: Enyu Zhou, Zhiheng Xi, Long Ma, Zhihao Zhang, Shihan Dou, Zhikai Lei, Guoteng Wang, Rui Zheng, Hang Yan, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04210)  

**Abstract**: As Large Language Models increasingly automate complex, long-horizon tasks such as \emph{vibe coding}, a supervision gap has emerged. While models excel at execution, users often struggle to guide them effectively due to insufficient domain expertise, the difficulty of articulating precise intent, and the inability to reliably validate complex outputs. It presents a critical challenge in scalable oversight: enabling humans to responsibly steer AI systems on tasks that surpass their own ability to specify or verify. To tackle this, we propose Scalable Interactive Oversight, a framework that decomposes complex intent into a recursive tree of manageable decisions to amplify human supervision. Rather than relying on open-ended prompting, our system elicits low-burden feedback at each node and recursively aggregates these signals into precise global guidance. Validated in web development task, our framework enables non-experts to produce expert-level Product Requirement Documents, achieving a 54\% improvement in alignment. Crucially, we demonstrate that this framework can be optimized via Reinforcement Learning using only online user feedback, offering a practical pathway for maintaining human control as AI scales. 

---
# OMG-Agent: Toward Robust Missing Modality Generation with Decoupled Coarse-to-Fine Agentic Workflows 

**Authors**: Ruiting Dai, Zheyu Wang, Haoyu Yang, Yihan Liu, Chengzhi Wang, Zekun Zhang, Zishan Huang, Jiaman Cen, Lisi Mo  

**Link**: [PDF](https://arxiv.org/pdf/2602.04144)  

**Abstract**: Data incompleteness severely impedes the reliability of multimodal systems. Existing reconstruction methods face distinct bottlenecks: conventional parametric/generative models are prone to hallucinations due to over-reliance on internal memory, while retrieval-augmented frameworks struggle with retrieval rigidity. Critically, these end-to-end architectures are fundamentally constrained by Semantic-Detail Entanglement -- a structural conflict between logical reasoning and signal synthesis that compromises fidelity. In this paper, we present \textbf{\underline{O}}mni-\textbf{\underline{M}}odality \textbf{\underline{G}}eneration Agent (\textbf{OMG-Agent}), a novel framework that shifts the paradigm from static mapping to a dynamic coarse-to-fine Agentic Workflow. By mimicking a \textit{deliberate-then-act} cognitive process, OMG-Agent explicitly decouples the task into three synergistic stages: (1) an MLLM-driven Semantic Planner that resolves input ambiguity via Progressive Contextual Reasoning, creating a deterministic structured semantic plan; (2) a non-parametric Evidence Retriever that grounds abstract semantics in external knowledge; and (3) a Retrieval-Injected Executor that utilizes retrieved evidence as flexible feature prompts to overcome rigidity and synthesize high-fidelity details. Extensive experiments on multiple benchmarks demonstrate that OMG-Agent consistently surpasses state-of-the-art methods, maintaining robustness under extreme missingness, e.g., a $2.6$-point gain on CMU-MOSI at $70$\% missing rates. 

---
# Interfaze: The Future of AI is built on Task-Specific Small Models 

**Authors**: Harsha Vardhan Khurdula, Vineet Agarwal, Yoeven D Khemlani  

**Link**: [PDF](https://arxiv.org/pdf/2602.04101)  

**Abstract**: We present Interfaze, a system that treats modern LLM applications as a problem of building and acting over context, not just picking the right monolithic model. Instead of a single transformer, we combine (i) a stack of heterogeneous DNNs paired with small language models as perception modules for OCR involving complex PDFs, charts and diagrams, and multilingual ASR with (ii) a context-construction layer that crawls, indexes, and parses external sources (web pages, code, PDFs) into compact structured state, and (iii) an action layer that can browse, retrieve, execute code in a sandbox, and drive a headless browser for dynamic web pages. A thin controller sits on top of this stack and exposes a single, OpenAI-style endpoint: it decides which small models and actions to run and always forwards the distilled context to a user-selected LLM that produces the final response.
On this architecture, Interfaze-Beta achieves 83.6% on MMLU-Pro, 91.4% on MMLU, 81.3% on GPQA-Diamond, 57.8% on LiveCodeBench v5, and 90.0% on AIME-2025, along with strong multimodal scores on MMMU (val) (77.3%), AI2D (91.5%), ChartQA (90.9%), and Common Voice v16 (90.8%). We show that most queries are handled primarily by the small-model and tool stack, with the large LLM operating only on distilled context, yielding competitive accuracy while shifting the bulk of computation away from the most expensive and monolithic models. 

---
# Scaling In-Context Online Learning Capability of LLMs via Cross-Episode Meta-RL 

**Authors**: Xiaofeng Lin, Sirou Zhu, Yilei Chen, Mingyu Chen, Hejian Sang, Ioannis Paschalidis, Zhipeng Wang, Aldo Pacchiano, Xuezhou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04089)  

**Abstract**: Large language models (LLMs) achieve strong performance when all task-relevant information is available upfront, as in static prediction and instruction-following problems. However, many real-world decision-making tasks are inherently online: crucial information must be acquired through interaction, feedback is delayed, and effective behavior requires balancing information collection and exploitation over time. While in-context learning enables adaptation without weight updates, existing LLMs often struggle to reliably leverage in-context interaction experience in such settings. In this work, we show that this limitation can be addressed through training. We introduce ORBIT, a multi-task, multi-episode meta-reinforcement learning framework that trains LLMs to learn from interaction in context. After meta-training, a relatively small open-source model (Qwen3-14B) demonstrates substantially improved in-context online learning on entirely unseen environments, matching the performance of GPT-5.2 and outperforming standard RL fine-tuning by a large margin. Scaling experiments further reveal consistent gains with model size, suggesting significant headroom for learn-at-inference-time decision-making agents. Code reproducing the results in the paper can be found at this https URL. 

---
# Axiomatic Foundations of Counterfactual Explanations 

**Authors**: Leila Amgoud, Martin Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2602.04028)  

**Abstract**: Explaining autonomous and intelligent systems is critical in order to improve trust in their decisions. Counterfactuals have emerged as one of the most compelling forms of explanation. They address ``why not'' questions by revealing how decisions could be altered. Despite the growing literature, most existing explainers focus on a single type of counterfactual and are restricted to local explanations, focusing on individual instances. There has been no systematic study of alternative counterfactual types, nor of global counterfactuals that shed light on a system's overall reasoning process.
This paper addresses the two gaps by introducing an axiomatic framework built on a set of desirable properties for counterfactual explainers. It proves impossibility theorems showing that no single explainer can satisfy certain axiom combinations simultaneously, and fully characterizes all compatible sets. Representation theorems then establish five one-to-one correspondences between specific subsets of axioms and the families of explainers that satisfy them. Each family gives rise to a distinct type of counterfactual explanation, uncovering five fundamentally different types of counterfactuals. Some of these correspond to local explanations, while others capture global explanations. Finally, the framework situates existing explainers within this taxonomy, formally characterizes their behavior, and analyzes the computational complexity of generating such explanations. 

---
# When AI Persuades: Adversarial Explanation Attacks on Human Trust in AI-Assisted Decision Making 

**Authors**: Shutong Fan, Lan Zhang, Xiaoyong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2602.04003)  

**Abstract**: Most adversarial threats in artificial intelligence target the computational behavior of models rather than the humans who rely on them. Yet modern AI systems increasingly operate within human decision loops, where users interpret and act on model recommendations. Large Language Models generate fluent natural-language explanations that shape how users perceive and trust AI outputs, revealing a new attack surface at the cognitive layer: the communication channel between AI and its users. We introduce adversarial explanation attacks (AEAs), where an attacker manipulates the framing of LLM-generated explanations to modulate human trust in incorrect outputs. We formalize this behavioral threat through the trust miscalibration gap, a metric that captures the difference in human trust between correct and incorrect outputs under adversarial explanations. By incorporating this gap, AEAs explore the daunting threats in which persuasive explanations reinforce users' trust in incorrect predictions. To characterize this threat, we conducted a controlled experiment (n = 205), systematically varying four dimensions of explanation framing: reasoning mode, evidence type, communication style, and presentation format. Our findings show that users report nearly identical trust for adversarial and benign explanations, with adversarial explanations preserving the vast majority of benign trust despite being incorrect. The most vulnerable cases arise when AEAs closely resemble expert communication, combining authoritative evidence, neutral tone, and domain-appropriate reasoning. Vulnerability is highest on hard tasks, in fact-driven domains, and among participants who are less formally educated, younger, or highly trusting of AI. This is the first systematic security study that treats explanations as an adversarial cognitive channel and quantifies their impact on human trust in AI-assisted decision making. 

---
# Monitorability as a Free Gift: How RLVR Spontaneously Aligns Reasoning 

**Authors**: Zidi Xiong, Shan Chen, Himabindu Lakkaraju  

**Link**: [PDF](https://arxiv.org/pdf/2602.03978)  

**Abstract**: As Large Reasoning Models (LRMs) are increasingly deployed, auditing their chain-of-thought (CoT) traces for safety becomes critical. Recent work has reported that monitorability--the degree to which CoT faithfully and informatively reflects internal computation--can appear as a "free gift" during the early stages of Reinforcement Learning with Verifiable Rewards (RLVR). We make this observation concrete through a systematic evaluation across model families and training domains. Our results show that this effect is not universal: monitorability improvements are strongly data-dependent. In particular, we demonstrate the critical role of data diversity and instruction-following data during RLVR training. We further show that monitorability is orthogonal to capability--improvements in reasoning performance do not imply increased transparency. Through mechanistic analysis, we attribute monitorability gains primarily to response distribution sharpening (entropy reduction) and increased attention to the prompt, rather than stronger causal reliance on reasoning traces. We also reveal how monitorability dynamics vary with controlled training and evaluation difficulty. Together, these findings provide a holistic view of how monitorability emerges under RLVR, clarifying when gains are likely to occur and when they are not. 

---
# Adaptive Test-Time Compute Allocation via Learned Heuristics over Categorical Structure 

**Authors**: Shuhui Qu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03975)  

**Abstract**: Test-time computation has become a primary driver of progress in large language model (LLM) reasoning, but it is increasingly bottlenecked by expensive verification. In many reasoning systems, a large fraction of verifier calls are spent on redundant or unpromising intermediate hypotheses. We study reasoning under a \emph{verification-cost-limited} setting and ask how verification effort should be allocated across intermediate states. We propose a state-level selective verification framework that combines (i) deterministic feasibility gating over a structured move interface, (ii) pre-verification ranking using a hybrid of learned state-distance and residual scoring, and (iii) adaptive allocation of verifier calls based on local uncertainty. Unlike solution-level best-of-$N$ or uniform intermediate verification, our method distributes verification where it is most informative. On the \textsc{MATH} benchmark, our approach achieves higher accuracy than best-of-$N$, majority voting, and beam search while using 44\% fewer verifier calls. 

---
# Active Epistemic Control for Query-Efficient Verified Planning 

**Authors**: Shuhui Qu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03974)  

**Abstract**: Planning in interactive environments is challenging under partial observability: task-critical preconditions (e.g., object locations or container states) may be unknown at decision time, yet grounding them through interaction is costly. Learned world models can cheaply predict missing facts, but prediction errors can silently induce infeasible commitments. We present \textbf{Active Epistemic Control (AEC)}, an epistemic-categorical planning layer that integrates model-based belief management with categorical feasibility checks. AEC maintains a strict separation between a \emph{grounded fact store} used for commitment and a \emph{belief store} used only for pruning candidate plans. At each step, it either queries the environment to ground an unresolved predicate when uncertainty is high or predictions are ambiguous, or simulates the predicate to filter hypotheses when confidence is sufficient. Final commitment is gated by grounded precondition coverage and an SQ-BCP pullback-style compatibility check, so simulated beliefs affect efficiency but cannot directly certify feasibility. Experiments on ALFWorld and ScienceWorld show that AEC achieves competitive success with fewer replanning rounds than strong LLM-agent baselines. 

---
# AgentArk: Distilling Multi-Agent Intelligence into a Single LLM Agent 

**Authors**: Yinyi Luo, Yiqiao Jin, Weichen Yu, Mengqi Zhang, Srijan Kumar, Xiaoxiao Li, Weijie Xu, Xin Chen, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03955)  

**Abstract**: While large language model (LLM) multi-agent systems achieve superior reasoning performance through iterative debate, practical deployment is limited by their high computational cost and error propagation. This paper proposes AgentArk, a novel framework to distill multi-agent dynamics into the weights of a single model, effectively transforming explicit test-time interactions into implicit model capabilities. This equips a single agent with the intelligence of multi-agent systems while remaining computationally efficient. Specifically, we investigate three hierarchical distillation strategies across various models, tasks, scaling, and scenarios: reasoning-enhanced fine-tuning; trajectory-based augmentation; and process-aware distillation. By shifting the burden of computation from inference to training, the distilled models preserve the efficiency of one agent while exhibiting strong reasoning and self-correction performance of multiple agents. They further demonstrate enhanced robustness and generalization across diverse reasoning tasks. We hope this work can shed light on future research on efficient and robust multi-agent development. Our code is at this https URL. 

---
# Enhancing Mathematical Problem Solving in LLMs through Execution-Driven Reasoning Augmentation 

**Authors**: Aditya Basarkar, Benyamin Tabarsi, Tiffany Barnes, Dongkuan  

**Link**: [PDF](https://arxiv.org/pdf/2602.03950)  

**Abstract**: Mathematical problem solving is a fundamental benchmark for assessing the reasoning capabilities of artificial intelligence and a gateway to applications in education, science, and engineering where reliable symbolic reasoning is essential. Although recent advances in multi-agent LLM-based systems have enhanced their mathematical reasoning capabilities, they still lack a reliably revisable representation of the reasoning process. Existing agents either operate in rigid sequential pipelines that cannot correct earlier steps or rely on heuristic self-evaluation that can fail to identify and fix errors. In addition, programmatic context can distract language models and degrade accuracy. To address these gaps, we introduce Iteratively Improved Program Construction (IIPC), a reasoning method that iteratively refines programmatic reasoning chains and combines execution feedback with the native Chain-of-thought abilities of the base LLM to maintain high-level contextual focus. IIPC surpasses competing approaches in the majority of reasoning benchmarks on multiple base LLMs. All code and implementations are released as open source. 

---
# Knowledge Model Prompting Increases LLM Performance on Planning Tasks 

**Authors**: Erik Goh, John Kos, Ashok Goel  

**Link**: [PDF](https://arxiv.org/pdf/2602.03900)  

**Abstract**: Large Language Models (LLM) can struggle with reasoning ability and planning tasks. Many prompting techniques have been developed to assist with LLM reasoning, notably Chain-of-Thought (CoT); however, these techniques, too, have come under scrutiny as LLMs' ability to reason at all has come into question. Borrowing from the domain of cognitive and educational science, this paper investigates whether the Task-Method-Knowledge (TMK) framework can improve LLM reasoning capabilities beyond its previously demonstrated success in educational applications. The TMK framework's unique ability to capture causal, teleological, and hierarchical reasoning structures, combined with its explicit task decomposition mechanisms, makes it particularly well-suited for addressing language model reasoning deficiencies, and unlike other hierarchical frameworks such as HTN and BDI, TMK provides explicit representations of not just what to do and how to do it, but also why actions are taken. The study evaluates TMK by experimenting on the PlanBench benchmark, focusing on the Blocksworld domain to test for reasoning and planning capabilities, examining whether TMK-structured prompting can help language models better decompose complex planning problems into manageable sub-tasks. Results also highlight significant performance inversion in reasoning models. TMK prompting enables the reasoning model to achieve up to an accuracy of 97.3\% on opaque, symbolic tasks (Random versions of Blocksworld in PlanBench) where it previously failed (31.5\%), suggesting the potential to bridge the gap between semantic approximation and symbolic manipulation. Our findings suggest that TMK functions not merely as context, but also as a mechanism that steers reasoning models away from their default linguistic modes to engage formal, code-execution pathways in the context of the experiments. 

---
# Protein Autoregressive Modeling via Multiscale Structure Generation 

**Authors**: Yanru Qu, Cheng-Yen Hsieh, Zaixiang Zheng, Ge Liu, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2602.04883)  

**Abstract**: We present protein autoregressive modeling (PAR), the first multi-scale autoregressive framework for protein backbone generation via coarse-to-fine next-scale prediction. Using the hierarchical nature of proteins, PAR generates structures that mimic sculpting a statue, forming a coarse topology and refining structural details over scales. To achieve this, PAR consists of three key components: (i) multi-scale downsampling operations that represent protein structures across multiple scales during training; (ii) an autoregressive transformer that encodes multi-scale information and produces conditional embeddings to guide structure generation; (iii) a flow-based backbone decoder that generates backbone atoms conditioned on these embeddings. Moreover, autoregressive models suffer from exposure bias, caused by the training and the generation procedure mismatch, and substantially degrades structure generation quality. We effectively alleviate this issue by adopting noisy context learning and scheduled sampling, enabling robust backbone generation. Notably, PAR exhibits strong zero-shot generalization, supporting flexible human-prompted conditional generation and motif scaffolding without requiring fine-tuning. On the unconditional generation benchmark, PAR effectively learns protein distributions and produces backbones of high design quality, and exhibits favorable scaling behavior. Together, these properties establish PAR as a promising framework for protein structure generation. 

---
# Contrastive Continual Learning for Model Adaptability in Internet of Things 

**Authors**: Ajesh Koyatan Chathoth  

**Link**: [PDF](https://arxiv.org/pdf/2602.04881)  

**Abstract**: Internet of Things (IoT) deployments operate in nonstationary, dynamic environments where factors such as sensor drift, evolving user behavior, and heterogeneous user privacy requirements can affect application utility. Continual learning (CL) addresses this by adapting models over time without catastrophic forgetting. Meanwhile, contrastive learning has emerged as a powerful representation-learning paradigm that improves robustness and sample efficiency in a self-supervised manner. This paper reviews the usage of \emph{contrastive continual learning} (CCL) for IoT, connecting algorithmic design (replay, regularization, distillation, prompts) with IoT system realities (TinyML constraints, intermittent connectivity, privacy). We present a unifying problem formulation, derive common objectives that blend contrastive and distillation losses, propose an IoT-oriented reference architecture for on-device, edge, and cloud-based CCL, and provide guidance on evaluation protocols and metrics. Finally, we highlight open unique challenges with respect to the IoT domain, such as spanning tabular and streaming IoT data, concept drift, federated settings, and energy-aware training. 

---
# Rethinking the Trust Region in LLM Reinforcement Learning 

**Authors**: Penghui Qi, Xiangxin Zhou, Zichen Liu, Tianyu Pang, Chao Du, Min Lin, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.04879)  

**Abstract**: Reinforcement learning (RL) has become a cornerstone for fine-tuning Large Language Models (LLMs), with Proximal Policy Optimization (PPO) serving as the de facto standard algorithm. Despite its ubiquity, we argue that the core ratio clipping mechanism in PPO is structurally ill-suited for the large vocabularies inherent to LLMs. PPO constrains policy updates based on the probability ratio of sampled tokens, which serves as a noisy single-sample Monte Carlo estimate of the true policy divergence. This creates a sub-optimal learning dynamic: updates to low-probability tokens are aggressively over-penalized, while potentially catastrophic shifts in high-probability tokens are under-constrained, leading to training inefficiency and instability. To address this, we propose Divergence Proximal Policy Optimization (DPPO), which substitutes heuristic clipping with a more principled constraint based on a direct estimate of policy divergence (e.g., Total Variation or KL). To avoid huge memory footprint, we introduce the efficient Binary and Top-K approximations to capture the essential divergence with negligible overhead. Extensive empirical evaluations demonstrate that DPPO achieves superior training stability and efficiency compared to existing methods, offering a more robust foundation for RL-based LLM fine-tuning. 

---
# Multi-layer Cross-Attention is Provably Optimal for Multi-modal In-context Learning 

**Authors**: Nicholas Barnfield, Subhabrata Sen, Pragya Sur  

**Link**: [PDF](https://arxiv.org/pdf/2602.04872)  

**Abstract**: Recent progress has rapidly advanced our understanding of the mechanisms underlying in-context learning in modern attention-based neural networks. However, existing results focus exclusively on unimodal data; in contrast, the theoretical underpinnings of in-context learning for multi-modal data remain poorly understood. We introduce a mathematically tractable framework for studying multi-modal learning and explore when transformer-like architectures can recover Bayes-optimal performance in-context. To model multi-modal problems, we assume the observed data arises from a latent factor model. Our first result comprises a negative take on expressibility: we prove that single-layer, linear self-attention fails to recover the Bayes-optimal predictor uniformly over the task distribution. To address this limitation, we introduce a novel, linearized cross-attention mechanism, which we study in the regime where both the number of cross-attention layers and the context length are large. We show that this cross-attention mechanism is provably Bayes optimal when optimized using gradient flow. Our results underscore the benefits of depth for in-context learning and establish the provable utility of cross-attention for multi-modal distributions. 

---
# CRoSS: A Continual Robotic Simulation Suite for Scalable Reinforcement Learning with High Task Diversity and Realistic Physics Simulation 

**Authors**: Yannick Denker, Alexander Gepperth  

**Link**: [PDF](https://arxiv.org/pdf/2602.04868)  

**Abstract**: Continual reinforcement learning (CRL) requires agents to learn from a sequence of tasks without forgetting previously acquired policies. In this work, we introduce a novel benchmark suite for CRL based on realistically simulated robots in the Gazebo simulator. Our Continual Robotic Simulation Suite (CRoSS) benchmarks rely on two robotic platforms: a two-wheeled differential-drive robot with lidar, camera and bumper sensor, and a robotic arm with seven joints. The former represent an agent in line-following and object-pushing scenarios, where variation of visual and structural parameters yields a large number of distinct tasks, whereas the latter is used in two goal-reaching scenarios with high-level cartesian hand position control (modeled after the Continual World benchmark), and low-level control based on joint angles. For the robotic arm benchmarks, we provide additional kinematics-only variants that bypass the need for physical simulation (as long as no sensor readings are required), and which can be run two orders of magnitude faster. CRoSS is designed to be easily extensible and enables controlled studies of continual reinforcement learning in robotic settings with high physical realism, and in particular allow the use of almost arbitrary simulated sensors. To ensure reproducibility and ease of use, we provide a containerized setup (Apptainer) that runs out-of-the-box, and report performances of standard RL algorithms, including Deep Q-Networks (DQN) and policy gradient methods. This highlights the suitability as a scalable and reproducible benchmark for CRL research. 

---
# Subliminal Effects in Your Data: A General Mechanism via Log-Linearity 

**Authors**: Ishaq Aden-Ali, Noah Golowich, Allen Liu, Abhishek Shetty, Ankur Moitra, Nika Haghtalab  

**Link**: [PDF](https://arxiv.org/pdf/2602.04863)  

**Abstract**: Training modern large language models (LLMs) has become a veritable smorgasbord of algorithms and datasets designed to elicit particular behaviors, making it critical to develop techniques to understand the effects of datasets on the model's properties. This is exacerbated by recent experiments that show datasets can transmit signals that are not directly observable from individual datapoints, posing a conceptual challenge for dataset-centric understandings of LLM training and suggesting a missing fundamental account of such phenomena. Towards understanding such effects, inspired by recent work on the linear structure of LLMs, we uncover a general mechanism through which hidden subtexts can arise in generic datasets.
We introduce Logit-Linear-Selection (LLS), a method that prescribes how to select subsets of a generic preference dataset to elicit a wide range of hidden effects. We apply LLS to discover subsets of real-world datasets so that models trained on them exhibit behaviors ranging from having specific preferences, to responding to prompts in a different language not present in the dataset, to taking on a different persona. Crucially, the effect persists for the selected subset, across models with varying architectures, supporting its generality and universality. 

---
# From Evaluation to Design: Using Potential Energy Surface Smoothness Metrics to Guide Machine Learning Interatomic Potential Architectures 

**Authors**: Ryan Liu, Eric Qu, Tobias Kreiman, Samuel M. Blau, Aditi S. Krishnapriyan  

**Link**: [PDF](https://arxiv.org/pdf/2602.04861)  

**Abstract**: Machine Learning Interatomic Potentials (MLIPs) sometimes fail to reproduce the physical smoothness of the quantum potential energy surface (PES), leading to erroneous behavior in downstream simulations that standard energy and force regression evaluations can miss. Existing evaluations, such as microcanonical molecular dynamics (MD), are computationally expensive and primarily probe near-equilibrium states. To improve evaluation metrics for MLIPs, we introduce the Bond Smoothness Characterization Test (BSCT). This efficient benchmark probes the PES via controlled bond deformations and detects non-smoothness, including discontinuities, artificial minima, and spurious forces, both near and far from equilibrium. We show that BSCT correlates strongly with MD stability while requiring a fraction of the cost of MD. To demonstrate how BSCT can guide iterative model design, we utilize an unconstrained Transformer backbone as a testbed, illustrating how refinements such as a new differentiable $k$-nearest neighbors algorithm and temperature-controlled attention reduce artifacts identified by our metric. By optimizing model design systematically based on BSCT, the resulting MLIP simultaneously achieves a low conventional E/F regression error, stable MD simulations, and robust atomistic property predictions. Our results establish BSCT as both a validation metric and as an "in-the-loop" model design proxy that alerts MLIP developers to physical challenges that cannot be efficiently evaluated by current MLIP benchmarks. 

---
# El Agente Quntur: A research collaborator agent for quantum chemistry 

**Authors**: Juan B. Pérez-Sánchez, Yunheng Zou, Jorge A. Campos-Gonzalez-Angulo, Marcel Müller, Ignacio Gustin, Andrew Wang, Han Hao, Tsz Wai Ko, Changhyeok Choi, Eric S. Isbrandt, Mohammad Ghazi Vakili, Hanyong Xu, Chris Crebolder, Varinia Bernales, Alán Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2602.04850)  

**Abstract**: Quantum chemistry is a foundational enabling tool for the fields of chemistry, materials science, computational biology and others. Despite of its power, the practical application of quantum chemistry simulations remains in the hands of qualified experts due to methodological complexity, software heterogeneity, and the need for informed interpretation of results. To bridge the accessibility gap for these tools and expand their reach to chemists with broader backgrounds, we introduce El Agente Quntur, a hierarchical, multi-agent AI system designed to operate not merely as an automation tool but as a research collaborator for computational quantum chemistry. Quntur was designed following three main strategies: i) elimination of hard-coded procedural policies in favour of reasoning-driven decisions, ii) construction of general and composable actions that facilitate generalization and efficiency, and iii) implementation of guided deep research to integrate abstract quantum-chemical reasoning across subdisciplines and a detailed understanding of the software's internal logic and syntax. Although instantiated in ORCA, these design principles are applicable to research agents more generally and easily expandable to additional quantum chemistry packages and beyond. Quntur supports the full range of calculations available in ORCA 6.0 and reasons over software documentation and scientific literature to plan, execute, adapt, and analyze in silico chemistry experiments following best practices. We discuss the advances and current bottlenecks in agentic systems operating at the research level in computational chemistry, and outline a roadmap toward a fully autonomous end-to-end computational chemistry research agent. 

---
# El Agente Estructural: An Artificially Intelligent Molecular Editor 

**Authors**: Changhyeok Choi, Yunheng Zou, Marcel Müller, Han Hao, Yeonghun Kang, Juan B. Pérez-Sánchez, Ignacio Gustin, Hanyong Xu, Mohammad Ghazi Vakili, Chris Crebolder, Alán Aspuru-Guzik, Varinia Bernales  

**Link**: [PDF](https://arxiv.org/pdf/2602.04849)  

**Abstract**: We present El Agente Estructural, a multimodal, natural-language-driven geometry-generation and manipulation agent for autonomous chemistry and molecular modelling. Unlike molecular generation or editing via generative models, Estructural mimics how human experts directly manipulate molecular systems in three dimensions by integrating a comprehensive set of domain-informed tools and vision-language models. This design enables precise control over atomic or functional group replacements, atomic connectivity, and stereochemistry without the need to rebuild extensive core molecular frameworks. Through a series of representative case studies, we demonstrate that Estructural enables chemically meaningful geometry manipulation across a wide range of real-world scenarios. These include site-selective functionalization, ligand binding, ligand exchange, stereochemically controlled structure construction, isomer interconversion, fragment-level structural analysis, image-guided generation of structures from schematic reaction mechanisms, and mechanism-driven geometry generation and modification. These examples illustrate how multimodal reasoning, when combined with specialized geometry-aware tools, supports interactive and context-aware molecular modelling beyond structure generation. Looking forward, the integration of Estructural into El Agente Quntur, an autonomous multi-agent quantum chemistry platform, enhances its capabilities by adding sophisticated tools for the generation and editing of three-dimensional structures. 

---
# It's not a Lottery, it's a Race: Understanding How Gradient Descent Adapts the Network's Capacity to the Task 

**Authors**: Hannah Pinson  

**Link**: [PDF](https://arxiv.org/pdf/2602.04832)  

**Abstract**: Our theoretical understanding of neural networks is lagging behind their empirical success. One of the important unexplained phenomena is why and how, during the process of training with gradient descent, the theoretical capacity of neural networks is reduced to an effective capacity that fits the task. We here investigate the mechanism by which gradient descent achieves this through analyzing the learning dynamics at the level of individual neurons in single hidden layer ReLU networks. We identify three dynamical principles -- mutual alignment, unlocking and racing -- that together explain why we can often successfully reduce capacity after training through the merging of equivalent neurons or the pruning of low norm weights. We specifically explain the mechanism behind the lottery ticket conjecture, or why the specific, beneficial initial conditions of some neurons lead them to obtain higher weight norms. 

---
# Safe Urban Traffic Control via Uncertainty-Aware Conformal Prediction and World-Model Reinforcement Learning 

**Authors**: Joydeep Chandra, Satyam Kumar Navneet, Aleksandr Algazinov, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04821)  

**Abstract**: Urban traffic management demands systems that simultaneously predict future conditions, detect anomalies, and take safe corrective actions -- all while providing reliability guarantees. We present STREAM-RL, a unified framework that introduces three novel algorithmic contributions: (1) PU-GAT+, an Uncertainty-Guided Adaptive Conformal Forecaster that uses prediction uncertainty to dynamically reweight graph attention via confidence-monotonic attention, achieving distribution-free coverage guarantees; (2) CRFN-BY, a Conformal Residual Flow Network that models uncertainty-normalized residuals via normalizing flows with Benjamini-Yekutieli FDR control under arbitrary dependence; and (3) LyCon-WRL+, an Uncertainty-Guided Safe World-Model RL agent with Lyapunov stability certificates, certified Lipschitz bounds, and uncertainty-propagated imagination rollouts. To our knowledge, this is the first framework to propagate calibrated uncertainty from forecasting through anomaly detection to safe policy learning with end-to-end theoretical guarantees. Experiments on multiple real-world traffic trajectory data demonstrate that STREAM-RL achieves 91.4\% coverage efficiency, controls FDR at 4.1\% under verified dependence, and improves safety rate to 95.2\% compared to 69\% for standard PPO while achieving higher reward, with 23ms end-to-end inference latency. 

---
# Toward Reliable and Explainable Nail Disease Classification: Leveraging Adversarial Training and Grad-CAM Visualization 

**Authors**: Farzia Hossain, Samanta Ghosh, Shahida Begum, B. M. Shahria Alam, Mohammad Tahmid Noor, Md Parvez Mia, Nishat Tasnim Niloy  

**Link**: [PDF](https://arxiv.org/pdf/2602.04820)  

**Abstract**: Human nail diseases are gradually observed over all age groups, especially among older individuals, often going ignored until they become severe. Early detection and accurate diagnosis of such conditions are important because they sometimes reveal our body's health problems. But it is challenging due to the inferred visual differences between disease types. This paper presents a machine learning-based model for automated classification of nail diseases based on a publicly available dataset, which contains 3,835 images scaling six categories. In 224x224 pixels, all images were resized to ensure consistency. To evaluate performance, four well-known CNN models-InceptionV3, DenseNet201, EfficientNetV2, and ResNet50 were trained and analyzed. Among these, InceptionV3 outperformed the others with an accuracy of 95.57%, while DenseNet201 came next with 94.79%. To make the model stronger and less likely to make mistakes on tricky or noisy images, we used adversarial training. To help understand how the model makes decisions, we used SHAP to highlight important features in the predictions. This system could be a helpful support for doctors, making nail disease diagnosis more accurate and faster. 

---
# SE-Bench: Benchmarking Self-Evolution with Knowledge Internalization 

**Authors**: Jiarui Yuan, Tailin Jin, Weize Chen, Zeyuan Liu, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.04811)  

**Abstract**: True self-evolution requires agents to act as lifelong learners that internalize novel experiences to solve future problems. However, rigorously measuring this foundational capability is hindered by two obstacles: the entanglement of prior knowledge, where ``new'' knowledge may appear in pre-training data, and the entanglement of reasoning complexity, where failures may stem from problem difficulty rather than an inability to recall learned knowledge. We introduce SE-Bench, a diagnostic environment that obfuscates the NumPy library and its API doc into a pseudo-novel package with randomized identifiers. Agents are trained to internalize this package and evaluated on simple coding tasks without access to documentation, yielding a clean setting where tasks are trivial with the new API doc but impossible for base models without it. Our investigation reveals three insights: (1) the Open-Book Paradox, where training with reference documentation inhibits retention, requiring "Closed-Book Training" to force knowledge compression into weights; (2) the RL Gap, where standard RL fails to internalize new knowledge completely due to PPO clipping and negative gradients; and (3) the viability of Self-Play for internalization, proving models can learn from self-generated, noisy tasks when coupled with SFT, but not RL. Overall, SE-Bench establishes a rigorous diagnostic platform for self-evolution with knowledge internalization. Our code and dataset can be found at this https URL. 

---
# Beyond Rewards in Reinforcement Learning for Cyber Defence 

**Authors**: Elizabeth Bates, Chris Hicks, Vasilios Mavroudis  

**Link**: [PDF](https://arxiv.org/pdf/2602.04809)  

**Abstract**: Recent years have seen an explosion of interest in autonomous cyber defence agents trained to defend computer networks using deep reinforcement learning. These agents are typically trained in cyber gym environments using dense, highly engineered reward functions which combine many penalties and incentives for a range of (un)desirable states and costly actions. Dense rewards help alleviate the challenge of exploring complex environments but risk biasing agents towards suboptimal and potentially riskier solutions, a critical issue in complex cyber environments. We thoroughly evaluate the impact of reward function structure on learning and policy behavioural characteristics using a variety of sparse and dense reward functions, two well-established cyber gyms, a range of network sizes, and both policy gradient and value-based RL algorithms. Our evaluation is enabled by a novel ground truth evaluation approach which allows directly comparing between different reward functions, illuminating the nuanced inter-relationships between rewards, action space and the risks of suboptimal policies in cyber environments. Our results show that sparse rewards, provided they are goal aligned and can be encountered frequently, uniquely offer both enhanced training reliability and more effective cyber defence agents with lower-risk policies. Surprisingly, sparse rewards can also yield policies that are better aligned with cyber defender goals and make sparing use of costly defensive actions without explicit reward-based numerical penalties. 

---
# Skin Tokens: A Learned Compact Representation for Unified Autoregressive Rigging 

**Authors**: Jia-peng Zhang, Cheng-Feng Pu, Meng-Hao Guo, Yan-Pei Cao, Shi-Min Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.04805)  

**Abstract**: The rapid proliferation of generative 3D models has created a critical bottleneck in animation pipelines: rigging. Existing automated methods are fundamentally limited by their approach to skinning, treating it as an ill-posed, high-dimensional regression task that is inefficient to optimize and is typically decoupled from skeleton generation. We posit this is a representation problem and introduce SkinTokens: a learned, compact, and discrete representation for skinning weights. By leveraging an FSQ-CVAE to capture the intrinsic sparsity of skinning, we reframe the task from continuous regression to a more tractable token sequence prediction problem. This representation enables TokenRig, a unified autoregressive framework that models the entire rig as a single sequence of skeletal parameters and SkinTokens, learning the complicated dependencies between skeletons and skin deformations. The unified model is then amenable to a reinforcement learning stage, where tailored geometric and semantic rewards improve generalization to complex, out-of-distribution assets. Quantitatively, the SkinTokens representation leads to a 98%-133% percents improvement in skinning accuracy over state-of-the-art methods, while the full TokenRig framework, refined with RL, enhances bone prediction by 17%-22%. Our work presents a unified, generative approach to rigging that yields higher fidelity and robustness, offering a scalable solution to a long-standing challenge in 3D content creation. 

---
# Team, Then Trim: An Assembly-Line LLM Framework for High-Quality Tabular Data Generation 

**Authors**: Congjing Zhang, Ryan Feng Lin, Ruoxuan Bao, Shuai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04785)  

**Abstract**: While tabular data is fundamental to many real-world machine learning (ML) applications, acquiring high-quality tabular data is usually labor-intensive and expensive. Limited by the scarcity of observations, tabular datasets often exhibit critical deficiencies, such as class imbalance, selection bias, and low fidelity. To address these challenges, building on recent advances in Large Language Models (LLMs), this paper introduces Team-then-Trim (T$^2$), a framework that synthesizes high-quality tabular data through a collaborative team of LLMs, followed by a rigorous three-stage plug-in data quality control (QC) pipeline. In T$^2$, tabular data generation is conceptualized as a manufacturing process: specialized LLMs, guided by domain knowledge, are tasked with generating different data components sequentially, and the resulting products, i.e., the synthetic data, are systematically evaluated across multiple dimensions of QC. Empirical results on both simulated and real-world datasets demonstrate that T$^2$ outperforms state-of-the-art methods in producing high-quality tabular data, highlighting its potential to support downstream models when direct data collection is practically infeasible. 

---
# Billion-Scale Graph Foundation Models 

**Authors**: Maya Bechler-Speicher, Yoel Gottlieb, Andrey Isakov, David Abensur, Ami Tavory, Daniel Haimovich, Ido Guy, Udi Weinsberg  

**Link**: [PDF](https://arxiv.org/pdf/2602.04768)  

**Abstract**: Graph-structured data underpins many critical applications. While foundation models have transformed language and vision via large-scale pretraining and lightweight adaptation, extending this paradigm to general, real-world graphs is challenging. In this work, we present Graph Billion- Foundation-Fusion (GraphBFF): the first end-to-end recipe for building billion-parameter Graph Foundation Models (GFMs) for arbitrary heterogeneous, billion-scale graphs. Central to the recipe is the GraphBFF Transformer, a flexible and scalable architecture designed for practical billion-scale GFMs. Using the GraphBFF, we present the first neural scaling laws for general graphs and show that loss decreases predictably as either model capacity or training data scales, depending on which factor is the bottleneck. The GraphBFF framework provides concrete methodologies for data batching, pretraining, and fine-tuning for building GFMs at scale. We demonstrate the effectiveness of the framework with an evaluation of a 1.4 billion-parameter GraphBFF Transformer pretrained on one billion samples. Across ten diverse, real-world downstream tasks on graphs unseen during training, spanning node- and link-level classification and regression, GraphBFF achieves remarkable zero-shot and probing performance, including in few-shot settings, with large margins of up to 31 PRAUC points. Finally, we discuss key challenges and open opportunities for making GFMs a practical and principled foundation for graph learning at industrial scale. 

---
# Active Asymmetric Multi-Agent Multimodal Learning under Uncertainty 

**Authors**: Rui Liu, Pratap Tokekar, Ming Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.04763)  

**Abstract**: Multi-agent systems are increasingly equipped with heterogeneous multimodal sensors, enabling richer perception but introducing modality-specific and agent-dependent uncertainty. Existing multi-agent collaboration frameworks typically reason at the agent level, assume homogeneous sensing, and handle uncertainty implicitly, limiting robustness under sensor corruption. We propose Active Asymmetric Multi-Agent Multimodal Learning under Uncertainty (A2MAML), a principled approach for uncertainty-aware, modality-level collaboration. A2MAML models each modality-specific feature as a stochastic estimate with uncertainty prediction, actively selects reliable agent-modality pairs, and aggregates information via Bayesian inverse-variance weighting. This formulation enables fine-grained, modality-level fusion, supports asymmetric modality availability, and provides a principled mechanism to suppress corrupted or noisy modalities. Extensive experiments on connected autonomous driving scenarios for collaborative accident detection demonstrate that A2MAML consistently outperforms both single-agent and collaborative baselines, achieving up to 18.7% higher accident detection rate. 

---
# When Silence Is Golden: Can LLMs Learn to Abstain in Temporal QA and Beyond? 

**Authors**: Xinyu Zhou, Chang Jin, Carsten Eickhoff, Zhijiang Guo, Seyed Ali Bahrainian  

**Link**: [PDF](https://arxiv.org/pdf/2602.04755)  

**Abstract**: Large language models (LLMs) rarely admit uncertainty, often producing fluent but misleading answers, rather than abstaining (i.e., refusing to answer). This weakness is even evident in temporal question answering, where models frequently ignore time-sensitive evidence and conflate facts across different time-periods. In this paper, we present the first empirical study of training LLMs with an abstention ability while reasoning about temporal QA. Existing approaches such as calibration might be unreliable in capturing uncertainty in complex reasoning. We instead frame abstention as a teachable skill and introduce a pipeline that couples Chain-of-Thought (CoT) supervision with Reinforcement Learning (RL) guided by abstention-aware rewards. Our goal is to systematically analyze how different information types and training techniques affect temporal reasoning with abstention behavior in LLMs. Through extensive experiments studying various methods, we find that RL yields strong empirical gains on reasoning: a model initialized by Qwen2.5-1.5B-Instruct surpasses GPT-4o by $3.46\%$ and $5.80\%$ in Exact Match on TimeQA-Easy and Hard, respectively. Moreover, it improves the True Positive rate on unanswerable questions by $20\%$ over a pure supervised fine-tuned (SFT) variant. Beyond performance, our analysis shows that SFT induces overconfidence and harms reliability, while RL improves prediction accuracy but exhibits similar risks. Finally, by comparing implicit reasoning cues (e.g., original context, temporal sub-context, knowledge graphs) with explicit CoT supervision, we find that implicit information provides limited benefit for reasoning with abstention. Our study provides new insights into how abstention and reasoning can be jointly optimized, providing a foundation for building more reliable LLMs. 

---
# Comparative Insights on Adversarial Machine Learning from Industry and Academia: A User-Study Approach 

**Authors**: Vishruti Kakkad, Paul Chung, Hanan Hibshi, Maverick Woo  

**Link**: [PDF](https://arxiv.org/pdf/2602.04753)  

**Abstract**: An exponential growth of Machine Learning and its Generative AI applications brings with it significant security challenges, often referred to as Adversarial Machine Learning (AML). In this paper, we conducted two comprehensive studies to explore the perspectives of industry professionals and students on different AML vulnerabilities and their educational strategies. In our first study, we conducted an online survey with professionals revealing a notable correlation between cybersecurity education and concern for AML threats. For our second study, we developed two CTF challenges that implement Natural Language Processing and Generative AI concepts and demonstrate a poisoning attack on the training data set. The effectiveness of these challenges was evaluated by surveying undergraduate and graduate students at Carnegie Mellon University, finding that a CTF-based approach effectively engages interest in AML threats. Based on the responses of the participants in our research, we provide detailed recommendations emphasizing the critical need for integrated security education within the ML curriculum. 

---
# Exploiting contextual information to improve stance detection in informal political discourse with LLMs 

**Authors**: Arman Engin Sucu, Yixiang Zhou, Mario A. Nascimento, Tony Mullen  

**Link**: [PDF](https://arxiv.org/pdf/2602.04750)  

**Abstract**: This study investigates the use of Large Language Models (LLMs) for political stance detection in informal online discourse, where language is often sarcastic, ambiguous, and context-dependent. We explore whether providing contextual information, specifically user profile summaries derived from historical posts, can improve classification accuracy. Using a real-world political forum dataset, we generate structured profiles that summarize users' ideological leaning, recurring topics, and linguistic patterns. We evaluate seven state-of-the-art LLMs across baseline and context-enriched setups through a comprehensive cross-model evaluation. Our findings show that contextual prompts significantly boost accuracy, with improvements ranging from +17.5\% to +38.5\%, achieving up to 74\% accuracy that surpasses previous approaches. We also analyze how profile size and post selection strategies affect performance, showing that strategically chosen political content yields better results than larger, randomly selected contexts. These findings underscore the value of incorporating user-level context to enhance LLM performance in nuanced political classification tasks. 

---
# Alignment Drift in Multimodal LLMs: A Two-Phase, Longitudinal Evaluation of Harm Across Eight Model Releases 

**Authors**: Casey Ford, Madison Van Doren, Emily Dix  

**Link**: [PDF](https://arxiv.org/pdf/2602.04739)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly deployed in real-world systems, yet their safety under adversarial prompting remains underexplored. We present a two-phase evaluation of MLLM harmlessness using a fixed benchmark of 726 adversarial prompts authored by 26 professional red teamers. Phase 1 assessed GPT-4o, Claude Sonnet 3.5, Pixtral 12B, and Qwen VL Plus; Phase 2 evaluated their successors (GPT-5, Claude Sonnet 4.5, Pixtral Large, and Qwen Omni) yielding 82,256 human harm ratings. Large, persistent differences emerged across model families: Pixtral models were consistently the most vulnerable, whereas Claude models appeared safest due to high refusal rates. Attack success rates (ASR) showed clear alignment drift: GPT and Claude models exhibited increased ASR across generations, while Pixtral and Qwen showed modest decreases. Modality effects also shifted over time: text-only prompts were more effective in Phase 1, whereas Phase 2 produced model-specific patterns, with GPT-5 and Claude 4.5 showing near-equivalent vulnerability across modalities. These findings demonstrate that MLLM harmlessness is neither uniform nor stable across updates, underscoring the need for longitudinal, multimodal benchmarks to track evolving safety behaviour. 

---
# From Data to Behavior: Predicting Unintended Model Behaviors Before Training 

**Authors**: Mengru Wang, Zhenqian Xu, Junfeng Fang, Yunzhi Yao, Shumin Deng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04735)  

**Abstract**: Large Language Models (LLMs) can acquire unintended biases from seemingly benign training data even without explicit cues or malicious content. Existing methods struggle to detect such risks before fine-tuning, making post hoc evaluation costly and inefficient. To address this challenge, we introduce Data2Behavior, a new task for predicting unintended model behaviors prior to training. We also propose Manipulating Data Features (MDF), a lightweight approach that summarizes candidate data through their mean representations and injects them into the forward pass of a base model, allowing latent statistical signals in the data to shape model activations and reveal potential biases and safety risks without updating any parameters. MDF achieves reliable prediction while consuming only about 20% of the GPU resources required for fine-tuning. Experiments on Qwen3-14B, Qwen2.5-32B-Instruct, and Gemma-3-12b-it confirm that MDF can anticipate unintended behaviors and provide insight into pre-training vulnerabilities. 

---
# Supporting software engineering tasks with agentic AI: Demonstration on document retrieval and test scenario generation 

**Authors**: Marian Kica, Lukas Radosky, David Slivka, Karin Kubinova, Daniel Dovhun, Tomas Uhercik, Erik Bircak, Ivan Polasek  

**Link**: [PDF](https://arxiv.org/pdf/2602.04726)  

**Abstract**: The introduction of large language models ignited great retooling and rethinking of the software development models. The ensuing response of software engineering research yielded a massive body of tools and approaches. In this paper, we join the hassle by introducing agentic AI solutions for two tasks. First, we developed a solution for automatic test scenario generation from a detailed requirements description. This approach relies on specialized worker agents forming a star topology with the supervisor agent in the middle. We demonstrate its capabilities on a real-world example. Second, we developed an agentic AI solution for the document retrieval task in the context of software engineering documents. Our solution enables performing various use cases on a body of documents related to the development of a single software, including search, question answering, tracking changes, and large document summarization. In this case, each use case is handled by a dedicated LLM-based agent, which performs all subtasks related to the corresponding use case. We conclude by hinting at the future perspectives of our line of research. 

---
# Identifying Intervenable and Interpretable Features via Orthogonality Regularization 

**Authors**: Moritz Miller, Florent Draye, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2602.04718)  

**Abstract**: With recent progress on fine-tuning language models around a fixed sparse autoencoder, we disentangle the decoder matrix into almost orthogonal features. This reduces interference and superposition between the features, while keeping performance on the target dataset essentially unchanged. Our orthogonality penalty leads to identifiable features, ensuring the uniqueness of the decomposition. Further, we find that the distance between embedded feature explanations increases with stricter orthogonality penalty, a desirable property for interpretability. Invoking the $\textit{Independent Causal Mechanisms}$ principle, we argue that orthogonality promotes modular representations amenable to causal intervention. We empirically show that these increasingly orthogonalized features allow for isolated interventions. Our code is available under $\texttt{this https URL}$. 

---
# Adaptive Prompt Elicitation for Text-to-Image Generation 

**Authors**: Xinyi Wen, Lena Hegemann, Xiaofu Jin, Shuai Ma, Antti Oulasvirta  

**Link**: [PDF](https://arxiv.org/pdf/2602.04713)  

**Abstract**: Aligning text-to-image generation with user intent remains challenging, for users who provide ambiguous inputs and struggle with model idiosyncrasies. We propose Adaptive Prompt Elicitation (APE), a technique that adaptively asks visual queries to help users refine prompts without extensive writing. Our technical contribution is a formulation of interactive intent inference under an information-theoretic framework. APE represents latent intent as interpretable feature requirements using language model priors, adaptively generates visual queries, and compiles elicited requirements into effective prompts. Evaluation on IDEA-Bench and DesignBench shows that APE achieves stronger alignment with improved efficiency. A user study with challenging user-defined tasks demonstrates 19.8% higher alignment without workload overhead. Our work contributes a principled approach to prompting that, for general users, offers an effective and efficient complement to the prevailing prompt-based interaction paradigm with text-to-image models. 

---
# SAR-RAG: ATR Visual Question Answering by Semantic Search, Retrieval, and MLLM Generation 

**Authors**: David F. Ramirez, Tim Overman, Kristen Jaskie, Joe Marvin, Andreas Spanias  

**Link**: [PDF](https://arxiv.org/pdf/2602.04712)  

**Abstract**: We present a visual-context image retrieval-augmented generation (ImageRAG) assisted AI agent for automatic target recognition (ATR) of synthetic aperture radar (SAR). SAR is a remote sensing method used in defense and security applications to detect and monitor the positions of military vehicles, which may appear indistinguishable in images. Researchers have extensively studied SAR ATR to improve the differentiation and identification of vehicle types, characteristics, and measurements. Test examples can be compared with known vehicle target types to improve recognition tasks. New methods enhance the capabilities of neural networks, transformer attention, and multimodal large language models. An agentic AI method may be developed to utilize a defined set of tools, such as searching through a library of similar examples. Our proposed method, SAR Retrieval-Augmented Generation (SAR-RAG), combines a multimodal large language model (MLLM) with a vector database of semantic embeddings to support contextual search for image exemplars with known qualities. By recovering past image examples with known true target types, our SAR-RAG system can compare similar vehicle categories, achieving improved ATR prediction accuracy. We evaluate this through search and retrieval metrics, categorical classification accuracy, and numeric regression of vehicle dimensions. These metrics all show improvements when SAR-RAG is added to an MLLM baseline method as an attached ATR memory bank. 

---
# Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention 

**Authors**: Sagie Dekel, Moshe Tennenholtz, Oren Kurland  

**Link**: [PDF](https://arxiv.org/pdf/2602.04711)  

**Abstract**: Retrieval Augmented Generation (RAG) is a highly effective paradigm for keeping LLM-based responses up-to-date and reducing the likelihood of hallucinations. Yet, RAG was recently shown to be quite vulnerable to corpus knowledge poisoning: an attacker injects misleading documents to the corpus to steer an LLMs' output to an undesired response. We argue that the standard causal attention mechanism in LLMs enables harmful cross-document interactions, specifically in cases of attacks. Accordingly, we introduce a novel defense approach for RAG: Sparse Document Attention RAG (SDAG). This is a block-sparse attention mechanism that disallows cross-attention between retrieved documents. SDAG requires a minimal inference-time change to the attention mask; furthermore, no fine-tuning or additional architectural changes are needed. We present an empirical evaluation of LLM-based question answering (QA) with a variety of attack strategies on RAG. We show that our SDAG method substantially outperforms the standard causal attention mechanism in terms of attack success rate. We further demonstrate the clear merits of integrating SDAG with state-of-the-art RAG defense methods. Specifically, the integration results in performance that is statistically significantly better than the state-of-the-art. 

---
# DRMOT: A Dataset and Framework for RGBD Referring Multi-Object Tracking 

**Authors**: Sijia Chen, Lijuan Ma, Yanqiu Yu, En Yu, Liman Liu, Wenbing Tao  

**Link**: [PDF](https://arxiv.org/pdf/2602.04692)  

**Abstract**: Referring Multi-Object Tracking (RMOT) aims to track specific targets based on language descriptions and is vital for interactive AI systems such as robotics and autonomous driving. However, existing RMOT models rely solely on 2D RGB data, making it challenging to accurately detect and associate targets characterized by complex spatial semantics (e.g., ``the person closest to the camera'') and to maintain reliable identities under severe occlusion, due to the absence of explicit 3D spatial information. In this work, we propose a novel task, RGBD Referring Multi-Object Tracking (DRMOT), which explicitly requires models to fuse RGB, Depth (D), and Language (L) modalities to achieve 3D-aware tracking. To advance research on the DRMOT task, we construct a tailored RGBD referring multi-object tracking dataset, named DRSet, designed to evaluate models' spatial-semantic grounding and tracking capabilities. Specifically, DRSet contains RGB images and depth maps from 187 scenes, along with 240 language descriptions, among which 56 descriptions incorporate depth-related information. Furthermore, we propose DRTrack, a MLLM-guided depth-referring tracking framework. DRTrack performs depth-aware target grounding from joint RGB-D-L inputs and enforces robust trajectory association by incorporating depth cues. Extensive experiments on the DRSet dataset demonstrate the effectiveness of our framework. 

---
# Audio ControlNet for Fine-Grained Audio Generation and Editing 

**Authors**: Haina Zhu, Yao Xiao, Xiquan Li, Ziyang Ma, Jianwei Yu, Bowen Zhang, Mingqi Yang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.04680)  

**Abstract**: We study the fine-grained text-to-audio (T2A) generation task. While recent models can synthesize high-quality audio from text descriptions, they often lack precise control over attributes such as loudness, pitch, and sound events. Unlike prior approaches that retrain models for specific control types, we propose to train ControlNet models on top of pre-trained T2A backbones to achieve controllable generation over loudness, pitch, and event roll. We introduce two designs, T2A-ControlNet and T2A-Adapter, and show that the T2A-Adapter model offers a more efficient structure with strong control ability. With only 38M additional parameters, T2A-Adapter achieves state-of-the-art performance on the AudioSet-Strong in both event-level and segment-level F1 scores. We further extend this framework to audio editing, proposing T2A-Editor for removing and inserting audio events at time locations specified by instructions. Models, code, dataset pipelines, and benchmarks will be released to support future research on controllable audio generation and editing. 

---
# Let Experts Feel Uncertainty: A Multi-Expert Label Distribution Approach to Probabilistic Time Series Forecasting 

**Authors**: Zhen Zhou, Zhirui Wang, Qi Hong, Yunyang Shi, Ziyuan Gu, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.04678)  

**Abstract**: Time series forecasting in real-world applications requires both high predictive accuracy and interpretable uncertainty quantification. Traditional point prediction methods often fail to capture the inherent uncertainty in time series data, while existing probabilistic approaches struggle to balance computational efficiency with interpretability. We propose a novel Multi-Expert Learning Distributional Labels (LDL) framework that addresses these challenges through mixture-of-experts architectures with distributional learning capabilities. Our approach introduces two complementary methods: (1) Multi-Expert LDL, which employs multiple experts with different learned parameters to capture diverse temporal patterns, and (2) Pattern-Aware LDL-MoE, which explicitly decomposes time series into interpretable components (trend, seasonality, changepoints, volatility) through specialized sub-experts. Both frameworks extend traditional point prediction to distributional learning, enabling rich uncertainty quantification through Maximum Mean Discrepancy (MMD). We evaluate our methods on aggregated sales data derived from the M5 dataset, demonstrating superior performance compared to baseline approaches. The continuous Multi-Expert LDL achieves the best overall performance, while the Pattern-Aware LDL-MoE provides enhanced interpretability through component-wise analysis. Our frameworks successfully balance predictive accuracy with interpretability, making them suitable for real-world forecasting applications where both performance and actionable insights are crucial. 

---
# Overstating Attitudes, Ignoring Networks: LLM Biases in Simulating Misinformation Susceptibility 

**Authors**: Eun Cheol Choi, Lindsay E. Young, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2602.04674)  

**Abstract**: Large language models (LLMs) are increasingly used as proxies for human judgment in computational social science, yet their ability to reproduce patterns of susceptibility to misinformation remains unclear. We test whether LLM-simulated survey respondents, prompted with participant profiles drawn from social survey data measuring network, demographic, attitudinal and behavioral features, can reproduce human patterns of misinformation belief and sharing. Using three online surveys as baselines, we evaluate whether LLM outputs match observed response distributions and recover feature-outcome associations present in the original survey data. LLM-generated responses capture broad distributional tendencies and show modest correlation with human responses, but consistently overstate the association between belief and sharing. Linear models fit to simulated responses exhibit substantially higher explained variance and place disproportionate weight on attitudinal and behavioral features, while largely ignoring personal network characteristics, relative to models fit to human responses. Analyses of model-generated reasoning and LLM training data suggest that these distortions reflect systematic biases in how misinformation-related concepts are represented. Our findings suggest that LLM-based survey simulations are better suited for diagnosing systematic divergences from human judgment than for substituting it. 

---
# Delving into Muon and Beyond: Deep Analysis and Extensions 

**Authors**: Xianbiao Qi, Marco Chen, Jiaquan Ye, Yelin He, Rong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2602.04669)  

**Abstract**: The Muon optimizer has recently attracted considerable attention for its strong empirical performance and use of orthogonalized updates on matrix-shaped parameters, yet its underlying mechanisms and relationship to adaptive optimizers such as Adam remain insufficiently understood. In this work, we aim to address these questions through a unified spectral perspective. Specifically, we view Muon as the p = 0 endpoint of a family of spectral transformations of the form U \boldsymbol{\Sigma}^{p} V' , and consider additional variants with p = 1/2 , p = 1/4 , and p = 1 . These transformations are applied to both first-moment updates, as in momentum SGD, and to root-mean-square (RMS) normalized gradient updates as in Adam. To enable efficient computation, we develop a coupled Newton iteration that avoids explicit singular value decomposition. Across controlled experiments, we find that RMS-normalized updates yield more stable optimization than first-moment updates. Moreover, while spectral compression provides strong stabilization benefits under first-moment updates, the Muon update (p = 0) does not consistently outperform Adam. These results suggest that Muon is best understood as an effective form of spectral normalization, but not a universally superior optimization method. Our source code will be released at this https URL. 

---
# Rethinking the Design Space of Reinforcement Learning for Diffusion Models: On the Importance of Likelihood Estimation Beyond Loss Design 

**Authors**: Jaemoo Choi, Yuchen Zhu, Wei Guo, Petr Molodyk, Bo Yuan, Jinbin Bai, Yi Xin, Molei Tao, Yongxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.04663)  

**Abstract**: Reinforcement learning has been widely applied to diffusion and flow models for visual tasks such as text-to-image generation. However, these tasks remain challenging because diffusion models have intractable likelihoods, which creates a barrier for directly applying popular policy-gradient type methods. Existing approaches primarily focus on crafting new objectives built on already heavily engineered LLM objectives, using ad hoc estimators for likelihood, without a thorough investigation into how such estimation affects overall algorithmic performance. In this work, we provide a systematic analysis of the RL design space by disentangling three factors: i) policy-gradient objectives, ii) likelihood estimators, and iii) rollout sampling schemes. We show that adopting an evidence lower bound (ELBO) based model likelihood estimator, computed only from the final generated sample, is the dominant factor enabling effective, efficient, and stable RL optimization, outweighing the impact of the specific policy-gradient loss functional. We validate our findings across multiple reward benchmarks using SD 3.5 Medium, and observe consistent trends across all tasks. Our method improves the GenEval score from 0.24 to 0.95 in 90 GPU hours, which is $4.6\times$ more efficient than FlowGRPO and $2\times$ more efficient than the SOTA method DiffusionNFT without reward hacking. 

---
# Towards Structured, State-Aware, and Execution-Grounded Reasoning for Software Engineering Agents 

**Authors**: Tse-Hsun, Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.04640)  

**Abstract**: Software Engineering (SE) agents have shown promising abilities in supporting various SE tasks. Current SE agents remain fundamentally reactive, making decisions mainly based on conversation history and the most recent response. However, this reactive design provides no explicit structure or persistent state within the agent's memory, making long-horizon reasoning challenging. As a result, SE agents struggle to maintain a coherent understanding across reasoning steps, adapt their hypotheses as new evidence emerges, or incorporate execution feedback into the mental reasoning model of the system state.
In this position paper, we argue that, to further advance SE agents, we need to move beyond reactive behavior toward a structured, state-aware, and execution-grounded reasoning. We outline how explicit structure, persistent and evolving state, and the integration of execution-grounded feedback can help SE agents perform more coherent and reliable reasoning in long-horizon tasks. We also provide an initial roadmap for developing next-generation SE agents that can more effectively perform real-world tasks. 

---
# A Human-Centered Privacy Approach (HCP) to AI 

**Authors**: Luyi Sun, Wei Xu, Zaifeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2602.04616)  

**Abstract**: As the paradigm of Human-Centered AI (HCAI) gains prominence, its benefits to society are accompanied by significant ethical concerns, one of which is the protection of individual privacy. This chapter provides a comprehensive overview of privacy within HCAI, proposing a human-centered privacy (HCP) framework, providing integrated solution from technology, ethics, and human factors perspectives. The chapter begins by mapping privacy risks across each stage of AI development lifecycle, from data collection to deployment and reuse, highlighting the impact of privacy risks on the entire system. The chapter then introduces privacy-preserving techniques such as federated learning and dif erential privacy. Subsequent chapters integrate the crucial user perspective by examining mental models, alongside the evolving regulatory and ethical landscapes as well as privacy governance. Next, advice on design guidelines is provided based on the human-centered privacy framework. After that, we introduce practical case studies across diverse fields. Finally, the chapter discusses persistent open challenges and future research directions, concluding that a multidisciplinary approach, merging technical, design, policy, and ethical expertise, is essential to successfully embed privacy into the core of HCAI, thereby ensuring these technologies advance in a manner that respects and ensures human autonomy, trust and dignity. 

---
# RexBERT: Context Specialized Bidirectional Encoders for E-commerce 

**Authors**: Rahul Bajaj, Anuj Garg  

**Link**: [PDF](https://arxiv.org/pdf/2602.04605)  

**Abstract**: Encoder-only transformers remain indispensable in retrieval, classification, and ranking systems where latency, stability, and cost are paramount. Most general purpose encoders, however, are trained on generic corpora with limited coverage of specialized domains. We introduce RexBERT, a family of BERT-style encoders designed specifically for e-commerce semantics. We make three contributions. First, we release Ecom-niverse, a 350 billion token corpus curated from diverse retail and shopping sources. We describe a modular pipeline that isolates and extracts e-commerce content from FineFineWeb and other open web resources, and characterize the resulting domain distribution. Second, we present a reproducible pretraining recipe building on ModernBERT's architectural advances. The recipe consists of three phases: general pre-training, context extension, and annealed domain specialization. Third, we train RexBERT models ranging from 17M to 400M parameters and evaluate them on token classification, semantic similarity, and general natural language understanding tasks using e-commerce datasets. Despite having 2-3x fewer parameters, RexBERT outperforms larger general-purpose encoders and matches or surpasses modern long-context models on domain-specific benchmarks. Our results demonstrate that high quality in-domain data combined with a principled training approach provides a stronger foundation for e-commerce applications than indiscriminate scaling alone. 

---
# VILLAIN at AVerImaTeC: Verifying Image-Text Claims via Multi-Agent Collaboration 

**Authors**: Jaeyoon Jung, Yejun Yoon, Seunghyun Yoon, Kunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2602.04587)  

**Abstract**: This paper describes VILLAIN, a multimodal fact-checking system that verifies image-text claims through prompt-based multi-agent collaboration. For the AVerImaTeC shared task, VILLAIN employs vision-language model agents across multiple stages of fact-checking. Textual and visual evidence is retrieved from the knowledge store enriched through additional web collection. To identify key information and address inconsistencies among evidence items, modality-specific and cross-modal agents generate analysis reports. In the subsequent stage, question-answer pairs are produced based on these reports. Finally, the Verdict Prediction agent produces the verification outcome based on the image-text claim and the generated question-answer pairs. Our system ranked first on the leaderboard across all evaluation metrics. The source code is publicly available at this https URL. 

---
# Trust The Typical 

**Authors**: Debargha Ganguly, Sreehari Sankar, Biyao Zhang, Vikash Singh, Kanan Gupta, Harshini Kavuru, Alan Luo, Weicong Chen, Warren Morningstar, Raghu Machiraju, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2602.04581)  

**Abstract**: Current approaches to LLM safety fundamentally rely on a brittle cat-and-mouse game of identifying and blocking known threats via guardrails. We argue for a fresh approach: robust safety comes not from enumerating what is harmful, but from deeply understanding what is safe. We introduce Trust The Typical (T3), a framework that operationalizes this principle by treating safety as an out-of-distribution (OOD) detection problem. T3 learns the distribution of acceptable prompts in a semantic space and flags any significant deviation as a potential threat. Unlike prior methods, it requires no training on harmful examples, yet achieves state-of-the-art performance across 18 benchmarks spanning toxicity, hate speech, jailbreaking, multilingual harms, and over-refusal, reducing false positive rates by up to 40x relative to specialized safety models. A single model trained only on safe English text transfers effectively to diverse domains and over 14 languages without retraining. Finally, we demonstrate production readiness by integrating a GPU-optimized version into vLLM, enabling continuous guardrailing during token generation with less than 6% overhead even under dense evaluation intervals on large-scale workloads. 

---
# Dual Mind World Model Inspired Network Digital Twin for Access Scheduling 

**Authors**: Hrishikesh Dutta, Roberto Minerva, Noel Crespi  

**Link**: [PDF](https://arxiv.org/pdf/2602.04566)  

**Abstract**: Emerging networked systems such as industrial IoT and real-time cyber-physical infrastructures demand intelligent scheduling strategies capable of adapting to dynamic traffic, deadlines, and interference constraints. In this work, we present a novel Digital Twin-enabled scheduling framework inspired by Dual Mind World Model (DMWM) architecture, for learning-informed and imagination-driven network control. Unlike conventional rule-based or purely data-driven policies, the proposed DMWM combines short-horizon predictive planning with symbolic model-based rollout, enabling the scheduler to anticipate future network states and adjust transmission decisions accordingly. We implement the framework in a configurable simulation testbed and benchmark its performance against traditional heuristics and reinforcement learning baselines under varied traffic conditions. Our results show that DMWM achieves superior performance in bursty, interference-limited, and deadline-sensitive environments, while maintaining interpretability and sample efficiency. The proposed design bridges the gap between network-level reasoning and low-overhead learning, marking a step toward scalable and adaptive NDT-based network optimization. 

---
# OmniRad: A Radiological Foundation Model for Multi-Task Medical Image Analysis 

**Authors**: Luca Zedda, Andrea Loddo, Cecilia Di Ruberto  

**Link**: [PDF](https://arxiv.org/pdf/2602.04547)  

**Abstract**: Radiological analysis increasingly benefits from pretrained visual representations that can support heterogeneous downstream tasks across imaging modalities. In this work, we introduce OmniRad, a self-supervised radiological foundation model pretrained on 1.2 million medical images, designed with radiology-inspired principles emphasizing representation reuse and cross-task transferability. We evaluate the pretrained encoder under multiple downstream adaptation regimes, including lightweight task-specific adapters with a frozen backbone as well as full end-to-end fine-tuning for classification, allowing us to assess both representation quality and task-specific performance. OmniRad is evaluated on a broad suite of public benchmarks spanning classification and segmentation across multiple modalities. On the MedMNISTv2 collection, OmniRad improves classification F1 by up to 2.05% over competing foundation models. For dense prediction, OmniRad attains mean Dice score improvements across six MedSegBench datasets when using frozen representations. Qualitative analyses and latent-space visualizations suggest improved feature clustering and modality-related separation. 

---
# Continual Learning through Control Minimization 

**Authors**: Sander de Haan, Yassine Taoudi-Benchekroun, Pau Vilimelis Aceituno, Benjamin F. Grewe  

**Link**: [PDF](https://arxiv.org/pdf/2602.04542)  

**Abstract**: Catastrophic forgetting remains a fundamental challenge for neural networks when tasks are trained sequentially. In this work, we reformulate continual learning as a control problem where learning and preservation signals compete within neural activity dynamics. We convert regularization penalties into preservation signals that protect prior-task representations. Learning then proceeds by minimizing the control effort required to integrate new tasks while competing with the preservation of prior tasks. At equilibrium, the neural activities produce weight updates that implicitly encode the full prior-task curvature, a property we term the continual-natural gradient, requiring no explicit curvature storage. Experiments confirm that our learning framework recovers true prior-task curvature and enables task discrimination, outperforming existing methods on standard benchmarks without replay. 

---
# LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding 

**Authors**: Gang Lin, Dongfang Li, Zhuoen Chen, Yukun Shi, Xuhui Chen, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04541)  

**Abstract**: The proliferation of long-context large language models (LLMs) exposes a key bottleneck: the rapidly expanding key-value cache during decoding, which imposes heavy memory and latency costs. While recent approaches attempt to alleviate this by sharing a single set of crucial tokens across layers, such coarse-grained sharing undermines model performance by neglecting the functional diversity of attention heads. To address this, we propose LycheeDecode, an efficient decoding method centered on a fine-grained hybrid-head attention mechanism that employs a hardware-efficient top-k selection strategy. Specifically, the novel HardKuma-based mechanism partitions attention heads into a small subset of retrieval heads that dynamically identify crucial tokens and a majority of sparse heads that reuse them for efficient computation. Through extensive experiments on leading models like Llama3 and Qwen3 across diverse benchmarks for long-context understanding (e.g., LongBench, RULER) and complex reasoning (e.g., AIME24, OlympiadBench), we demonstrate that LycheeDecode achieves generative quality comparable to, and at times surpassing even the full-attention baseline. Crucially, this is accomplished with up to a 2.7x speedup at a 128K context length. By preserving the functional diversity of attention heads, our fine-grained strategy overcomes the performance bottlenecks of existing methods, providing a powerful and validated pathway to both efficient and high-quality long-context LLM inference. 

---
# SLUM-i: Semi-supervised Learning for Urban Mapping of Informal Settlements and Data Quality Benchmarking 

**Authors**: Muhammad Taha Mukhtar, Syed Musa Ali Kazmi, Khola Naseem, Muhammad Ali Chattha, Andreas Dengel, Sheraz Ahmed, Muhammad Naseer Bajwa, Muhammad Imran Malik  

**Link**: [PDF](https://arxiv.org/pdf/2602.04525)  

**Abstract**: Rapid urban expansion has fueled the growth of informal settlements in major cities of low- and middle-income countries, with Lahore and Karachi in Pakistan and Mumbai in India serving as prominent examples. However, large-scale mapping of these settlements is severely constrained not only by the scarcity of annotations but by inherent data quality challenges, specifically high spectral ambiguity between formal and informal structures and significant annotation noise. We address this by introducing a benchmark dataset for Lahore, constructed from scratch, along with companion datasets for Karachi and Mumbai, which were derived from verified administrative boundaries, totaling 1,869 $\text{km}^2$ of area. To evaluate the global robustness of our framework, we extend our experiments to five additional established benchmarks, encompassing eight cities across three continents, and provide comprehensive data quality assessments of all datasets. We also propose a new semi-supervised segmentation framework designed to mitigate the class imbalance and feature degradation inherent in standard semi-supervised learning pipelines. Our method integrates a Class-Aware Adaptive Thresholding mechanism that dynamically adjusts confidence thresholds to prevent minority class suppression and a Prototype Bank System that enforces semantic consistency by anchoring predictions to historically learned high-fidelity feature representations. Extensive experiments across a total of eight cities spanning three continents demonstrate that our approach outperforms state-of-the-art semi-supervised baselines. Most notably, our method demonstrates superior domain transfer capability whereby a model trained on only 10% of source labels reaches a 0.461 mIoU on unseen geographies and outperforms the zero-shot generalization of fully supervised models. 

---
# Learning the Value Systems of Agents with Preference-based and Inverse Reinforcement Learning 

**Authors**: Andrés Holgado-Sánchez, Holger Billhardt, Alberto Fernández, Sascha Ossowski  

**Link**: [PDF](https://arxiv.org/pdf/2602.04518)  

**Abstract**: Agreement Technologies refer to open computer systems in which autonomous software agents interact with one another, typically on behalf of humans, in order to come to mutually acceptable agreements. With the advance of AI systems in recent years, it has become apparent that such agreements, in order to be acceptable to the involved parties, must remain aligned with ethical principles and moral values. However, this is notoriously difficult to ensure, especially as different human users (and their software agents) may hold different value systems, i.e. they may differently weigh the importance of individual moral values. Furthermore, it is often hard to specify the precise meaning of a value in a particular context in a computational manner. Methods to estimate value systems based on human-engineered specifications, e.g. based on value surveys, are limited in scale due to the need for intense human moderation. In this article, we propose a novel method to automatically \emph{learn} value systems from observations and human demonstrations. In particular, we propose a formal model of the \emph{value system learning} problem, its instantiation to sequential decision-making domains based on multi-objective Markov decision processes, as well as tailored preference-based and inverse reinforcement learning algorithms to infer value grounding functions and value systems. The approach is illustrated and evaluated by two simulated use cases. 

---
# BrainVista: Modeling Naturalistic Brain Dynamics as Multimodal Next-Token Prediction 

**Authors**: Xuanhua Yin, Runkai Zhao, Lina Yao, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2602.04512)  

**Abstract**: Naturalistic fMRI characterizes the brain as a dynamic predictive engine driven by continuous sensory streams. However, modeling the causal forward evolution in realistic neural simulation is impeded by the timescale mismatch between multimodal inputs and the complex topology of cortical networks. To address these challenges, we introduce BrainVista, a multimodal autoregressive framework designed to model the causal evolution of brain states. BrainVista incorporates Network-wise Tokenizers to disentangle system-specific dynamics and a Spatial Mixer Head that captures inter-network information flow without compromising functional boundaries. Furthermore, we propose a novel Stimulus-to-Brain (S2B) masking mechanism to synchronize high-frequency sensory stimuli with hemodynamically filtered signals, enabling strict, history-only causal conditioning. We validate our framework on Algonauts 2025, CineBrain, and HAD, achieving state-of-the-art fMRI encoding performance. In long-horizon rollout settings, our model yields substantial improvements over baselines, increasing pattern correlation by 36.0\% and 33.3\% on relative to the strongest baseline Algonauts 2025 and CineBrain, respectively. 

---
# Discovering Mechanistic Models of Neural Activity: System Identification in an in Silico Zebrafish 

**Authors**: Jan-Matthis Lueckmann, Viren Jain, Michał Januszewski  

**Link**: [PDF](https://arxiv.org/pdf/2602.04492)  

**Abstract**: Constructing mechanistic models of neural circuits is a fundamental goal of neuroscience, yet verifying such models is limited by the lack of ground truth. To rigorously test model discovery, we establish an in silico testbed using neuromechanical simulations of a larval zebrafish as a transparent ground truth. We find that LLM-based tree search autonomously discovers predictive models that significantly outperform established forecasting baselines. Conditioning on sensory drive is necessary but not sufficient for faithful system identification, as models exploit statistical shortcuts. Structural priors prove essential for enabling robust out-of-distribution generalization and recovery of interpretable mechanistic models. Our insights provide guidance for modeling real-world neural recordings and offer a broader template for AI-driven scientific discovery. 

---
# LLM-Empowered Cooperative Content Caching in Vehicular Fog Caching-Assisted Platoon Networks 

**Authors**: Bowen Tan, Qiong Wu, Pingyi Fan, Kezhi Wang, Nan Cheng, Wen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.04471)  

**Abstract**: This letter proposes a novel three-tier content caching architecture for Vehicular Fog Caching (VFC)-assisted platoon, where the VFC is formed by the vehicles driving near the platoon. The system strategically coordinates storage across local platoon vehicles, dynamic VFC clusters, and cloud server (CS) to minimize content retrieval latency. To efficiently manage distributed storage, we integrate large language models (LLMs) for real-time and intelligent caching decisions. The proposed approach leverages LLMs' ability to process heterogeneous information, including user profiles, historical data, content characteristics, and dynamic system states. Through a designed prompting framework encoding task objectives and caching constraints, the LLMs formulate caching as a decision-making task, and our hierarchical deterministic caching mapping strategy enables adaptive requests prediction and precise content placement across three tiers without frequent retraining. Simulation results demonstrate the advantages of our proposed caching scheme. 

---
# Is Micro Domain-Adaptive Pre-Training Effective for Real-World Operations? Multi-Step Evaluation Reveals Potential and Bottlenecks 

**Authors**: Masaya Tsunokake, Yuta Koreeda, Terufumi Morishita, Koichi Nagatsuka, Hikaru Tomonari, Yasuhiro Sogawa  

**Link**: [PDF](https://arxiv.org/pdf/2602.04466)  

**Abstract**: When applying LLMs to real-world enterprise operations, LLMs need to handle proprietary knowledge in small domains of specific operations ($\textbf{micro domains}$). A previous study shows micro domain-adaptive pre-training ($\textbf{mDAPT}$) with fewer documents is effective, similarly to DAPT in larger domains. However, it evaluates mDAPT only on multiple-choice questions; thus, its effectiveness for generative tasks in real-world operations remains unknown. We aim to reveal the potential and bottlenecks of mDAPT for generative tasks. To this end, we disentangle the answering process into three subtasks and evaluate the performance of each subtask: (1) $\textbf{eliciting}$ facts relevant to questions from an LLM's own knowledge, (2) $\textbf{reasoning}$ over the facts to obtain conclusions, and (3) $\textbf{composing}$ long-form answers based on the conclusions. We verified mDAPT on proprietary IT product knowledge for real-world questions in IT technical support operations. As a result, mDAPT resolved the elicitation task that the base model struggled with but did not resolve other subtasks. This clarifies mDAPT's effectiveness in the knowledge aspect and its bottlenecks in other aspects. Further analysis empirically shows that resolving the elicitation and reasoning tasks ensures sufficient performance (over 90%), emphasizing the need to enhance reasoning capability. 

---
# Growth First, Care Second? Tracing the Landscape of LLM Value Preferences in Everyday Dilemmas 

**Authors**: Zhiyi Chen, Eun Cheol Choi, Yingjia Luo, Xinyi Wang, Yulei Xiao, Aizi Yang, Luca Luceri  

**Link**: [PDF](https://arxiv.org/pdf/2602.04456)  

**Abstract**: People increasingly seek advice online from both human peers and large language model (LLM)-based chatbots. Such advice rarely involves identifying a single correct answer; instead, it typically requires navigating trade-offs among competing values. We aim to characterize how LLMs navigate value trade-offs across different advice-seeking contexts. First, we examine the value trade-off structure underlying advice seeking using a curated dataset from four advice-oriented subreddits. Using a bottom-up approach, we inductively construct a hierarchical value framework by aggregating fine-grained values extracted from individual advice options into higher-level value categories. We construct value co-occurrence networks to characterize how values co-occur within dilemmas and find substantial heterogeneity in value trade-off structures across advice-seeking contexts: a women-focused subreddit exhibits the highest network density, indicating more complex value conflicts; women's, men's, and friendship-related subreddits exhibit highly correlated value-conflict patterns centered on security-related tensions (security vs. respect/connection/commitment); by contrast, career advice forms a distinct structure where security frequently clashes with self-actualization and growth. We then evaluate LLM value preferences against these dilemmas and find that, across models and contexts, LLMs consistently prioritize values related to Exploration & Growth over Benevolence & Connection. This systemically skewed value orientation highlights a potential risk of value homogenization in AI-mediated advice, raising concerns about how such systems may shape decision-making and normative outcomes at scale. 

---
# RASA: Routing-Aware Safety Alignment for Mixture-of-Experts Models 

**Authors**: Jiacheng Liang, Yuhui Wang, Tanqiu Jiang, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04448)  

**Abstract**: Mixture-of-Experts (MoE) language models introduce unique challenges for safety alignment due to their sparse routing mechanisms, which can enable degenerate optimization behaviors under standard full-parameter fine-tuning. In our preliminary experiments, we observe that naively applying full-parameter safety fine-tuning to MoE models can reduce attack success rates through routing or expert dominance effects, rather than by directly repairing Safety-Critical Experts. To address this challenge, we propose RASA, a routing-aware expert-level alignment framework that explicitly repairs Safety-Critical Experts while preventing routing-based bypasses. RASA identifies experts disproportionately activated by successful jailbreaks, selectively fine-tunes only these experts under fixed routing, and subsequently enforces routing consistency with safety-aligned contexts. Across two representative MoE architectures and a diverse set of jailbreak attacks, RASA achieves near-perfect robustness, strong cross-attack generalization, and substantially reduced over-refusal, while preserving general capabilities on benchmarks such as MMLU, GSM8K, and TruthfulQA. Our results suggest that robust MoE safety alignment benefits from targeted expert repair rather than global parameter updates, offering a practical and architecture-preserving alternative to prior approaches. 

---
# Mixture of Masters: Sparse Chess Language Models with Player Routing 

**Authors**: Giacomo Frisoni, Lorenzo Molfetta, Davide Freddi, Gianluca Moro  

**Link**: [PDF](https://arxiv.org/pdf/2602.04447)  

**Abstract**: Modern chess language models are dense transformers trained on millions of games played by thousands of high-rated individuals. However, these monolithic networks tend to collapse into mode-averaged behavior, where stylistic boundaries are blurred, and rare but effective strategies are suppressed. To counteract homogenization, we introduce Mixture-of-Masters (MoM), the first chess mixture-of-experts model with small-sized GPT experts emulating world-class grandmasters. Each expert is trained with a combination of self-supervised learning and reinforcement learning guided by chess-specific rewards. For each move, a post-hoc learnable gating network selects the most appropriate persona to channel depending on the game state, allowing MoM to switch its style dynamically$--$e.g., Tal's offensive vocation or Petrosian's defensive solidity. When evaluated against Stockfish on unseen standard games, MoM outperforms both dense individual expert networks and popular GPT baselines trained on aggregated data, while ensuring generation variety, control, and interpretability. 

---
# No One-Size-Fits-All: Building Systems For Translation to Bashkir, Kazakh, Kyrgyz, Tatar and Chuvash Using Synthetic And Original Data 

**Authors**: Dmitry Karpov  

**Link**: [PDF](https://arxiv.org/pdf/2602.04442)  

**Abstract**: We explore machine translation for five Turkic language pairs: Russian-Bashkir, Russian-Kazakh, Russian-Kyrgyz, English-Tatar, English-Chuvash. Fine-tuning nllb-200-distilled-600M with LoRA on synthetic data achieved chrF++ 49.71 for Kazakh and 46.94 for Bashkir. Prompting DeepSeek-V3.2 with retrieved similar examples achieved chrF++ 39.47 for Chuvash. For Tatar, zero-shot or retrieval-based approaches achieved chrF++ 41.6, while for Kyrgyz the zero-shot approach reached 45.6. We release the dataset and the obtained weights. 

---
# SPEAR: An Engineering Case Study of Multi-Agent Coordination for Smart Contract Auditing 

**Authors**: Arnab Mallick, Indraveni Chebolu, Harmesh Rana  

**Link**: [PDF](https://arxiv.org/pdf/2602.04418)  

**Abstract**: We present SPEAR, a multi-agent coordination framework for smart contract auditing that applies established MAS patterns in a realistic security analysis workflow. SPEAR models auditing as a coordinated mission carried out by specialized agents: a Planning Agent prioritizes contracts using risk-aware heuristics, an Execution Agent allocates tasks via the Contract Net protocol, and a Repair Agent autonomously recovers from brittle generated artifacts using a programmatic-first repair policy. Agents maintain local beliefs updated through AGM-compliant revision, coordinate via negotiation and auction protocols, and revise plans as new information becomes available. An empirical study compares the multi-agent design with centralized and pipeline-based alternatives under controlled failure scenarios, focusing on coordination, recovery behavior, and resource use. 

---
# EMA Policy Gradient: Taming Reinforcement Learning for LLMs with EMA Anchor and Top-k KL 

**Authors**: Lunjun Zhang, Jimmy Ba  

**Link**: [PDF](https://arxiv.org/pdf/2602.04417)  

**Abstract**: Reinforcement Learning (RL) has enabled Large Language Models (LLMs) to acquire increasingly complex reasoning and agentic behaviors. In this work, we propose two simple techniques to improve policy gradient algorithms for LLMs. First, we replace the fixed anchor policy during RL with an Exponential Moving Average (EMA), similar to a target network in deep Q-learning. Second, we introduce Top-k KL estimator, which allows for flexible interpolation between exact KL and sampled KL. We derive the stability conditions for using EMA anchor; moreover, we show that our Top-k KL estimator yields both unbiased KL values and unbiased gradients at any k, while bringing the benefits of exact KL. When combined with GRPO, the two techniques (EMA-PG) lead to a significant performance boost. On math reasoning, it allows R1-distilled Qwen-1.5B to reach 53.9% on OlympiadBench compared to 50.8% by GRPO. On agentic RL domains, with Qwen-3B base, EMA-PG improves GRPO by an average of 33.3% across 7 datasets of Q&A with search engines, including 29.7% $\rightarrow$ 44.1% on HotpotQA, 27.4% $\rightarrow$ 40.1% on 2WikiMultiHopQA. Overall, we show that EMA-PG is a simple, principled, and powerful approach to scaling RL for LLMs. Code: this https URL 

---
# Med-MMFL: A Multimodal Federated Learning Benchmark in Healthcare 

**Authors**: Aavash Chhetri, Bibek Niroula, Pratik Shrestha, Yash Raj Shrestha, Lesley A Anderson, Prashnna K Gyawali, Loris Bazzani, Binod Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2602.04416)  

**Abstract**: Federated learning (FL) enables collaborative model training across decentralized medical institutions while preserving data privacy. However, medical FL benchmarks remain scarce, with existing efforts focusing mainly on unimodal or bimodal modalities and a limited range of medical tasks. This gap underscores the need for standardized evaluation to advance systematic understanding in medical MultiModal FL (MMFL). To this end, we introduce Med-MMFL, the first comprehensive MMFL benchmark for the medical domain, encompassing diverse modalities, tasks, and federation scenarios. Our benchmark evaluates six representative state-of-the-art FL algorithms, covering different aggregation strategies, loss formulations, and regularization techniques. It spans datasets with 2 to 4 modalities, comprising a total of 10 unique medical modalities, including text, pathology images, ECG, X-ray, radiology reports, and multiple MRI sequences. Experiments are conducted across naturally federated, synthetic IID, and synthetic non-IID settings to simulate real-world heterogeneity. We assess segmentation, classification, modality alignment (retrieval), and VQA tasks. To support reproducibility and fair comparison of future multimodal federated learning (MMFL) methods under realistic medical settings, we release the complete benchmark implementation, including data processing and partitioning pipelines, at this https URL . 

---
# History-Guided Iterative Visual Reasoning with Self-Correction 

**Authors**: Xinglong Yang, Zhilin Peng, Zhanzhan Liu, Haochen Shi, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04413)  

**Abstract**: Self-consistency methods are the core technique for improving the reasoning reliability of multimodal large language models (MLLMs). By generating multiple reasoning results through repeated sampling and selecting the best answer via voting, they play an important role in cross-modal tasks. However, most existing self-consistency methods are limited to a fixed ``repeated sampling and voting'' paradigm and do not reuse historical reasoning information. As a result, models struggle to actively correct visual understanding errors and dynamically adjust their reasoning during iteration. Inspired by the human reasoning behavior of repeated verification and dynamic error correction, we propose the H-GIVR framework. During iterative reasoning, the MLLM observes the image multiple times and uses previously generated answers as references for subsequent steps, enabling dynamic correction of errors and improving answer accuracy. We conduct comprehensive experiments on five datasets and three models. The results show that the H-GIVR framework can significantly improve cross-modal reasoning accuracy while maintaining low computational cost. For instance, using \texttt{Llama3.2-vision:11b} on the ScienceQA dataset, the model requires an average of 2.57 responses per question to achieve an accuracy of 78.90\%, representing a 107\% improvement over the baseline. 

---
# Performative Learning Theory 

**Authors**: Julian Rodemann, Unai Fischer-Abaigar, James Bailie, Krikamol Muandet  

**Link**: [PDF](https://arxiv.org/pdf/2602.04402)  

**Abstract**: Performative predictions influence the very outcomes they aim to forecast. We study performative predictions that affect a sample (e.g., only existing users of an app) and/or the whole population (e.g., all potential app users). This raises the question of how well models generalize under performativity. For example, how well can we draw insights about new app users based on existing users when both of them react to the app's predictions? We address this question by embedding performative predictions into statistical learning theory. We prove generalization bounds under performative effects on the sample, on the population, and on both. A key intuition behind our proofs is that in the worst case, the population negates predictions, while the sample deceptively fulfills them. We cast such self-negating and self-fulfilling predictions as min-max and min-min risk functionals in Wasserstein space, respectively. Our analysis reveals a fundamental trade-off between performatively changing the world and learning from it: the more a model affects data, the less it can learn from it. Moreover, our analysis results in a surprising insight on how to improve generalization guarantees by retraining on performatively distorted samples. We illustrate our bounds in a case study on prediction-informed assignments of unemployed German residents to job trainings, drawing upon administrative labor market records from 1975 to 2017 in Germany. 

---
# Bi-directional Bias Attribution: Debiasing Large Language Models without Modifying Prompts 

**Authors**: Yujie Lin, Kunquan Li, Yixuan Liao, Xiaoxin Chen, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2602.04398)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across a wide range of natural language processing tasks. However, their outputs often exhibit social biases, raising fairness concerns. Existing debiasing methods, such as fine-tuning on additional datasets or prompt engineering, face scalability issues or compromise user experience in multi-turn interactions. To address these challenges, we propose a framework for detecting stereotype-inducing words and attributing neuron-level bias in LLMs, without the need for fine-tuning or prompt modification. Our framework first identifies stereotype-inducing adjectives and nouns via comparative analysis across demographic groups. We then attribute biased behavior to specific neurons using two attribution strategies based on integrated gradients. Finally, we mitigate bias by directly intervening on their activations at the projection layer. Experiments on three widely used LLMs demonstrate that our method effectively reduces bias while preserving overall model performance. Code is available at the github link: this https URL. 

---
# LoRDO: Distributed Low-Rank Optimization with Infrequent Communication 

**Authors**: Andrej Jovanović, Alex Iacob, Mher Safaryan, Ionut-Vlad Modoranu, Lorenzo Sani, William F. Shen, Xinchi Qiu, Dan Alistarh, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2602.04396)  

**Abstract**: Distributed training of foundation models via $\texttt{DDP}$ is limited by interconnect bandwidth. While infrequent communication strategies reduce synchronization frequency, they remain bottlenecked by the memory and communication requirements of optimizer states. Low-rank optimizers can alleviate these constraints; however, in the local-update regime, workers lack access to the full-batch gradients required to compute low-rank projections, which degrades performance. We propose $\texttt{LoRDO}$, a principled framework unifying low-rank optimization with infrequent synchronization. We first demonstrate that, while global projections based on pseudo-gradients are theoretically superior, they permanently restrict the optimization trajectory to a low-rank subspace. To restore subspace exploration, we introduce a full-rank quasi-hyperbolic update. $\texttt{LoRDO}$ achieves near-parity with low-rank $\texttt{DDP}$ in language modeling and downstream tasks at model scales of $125$M--$720$M, while reducing communication by $\approx 10 \times$. Finally, we show that $\texttt{LoRDO}$ improves performance even more in very low-memory settings with small rank/batch size. 

---
# Blockchain Federated Learning for Sustainable Retail: Reducing Waste through Collaborative Demand Forecasting 

**Authors**: Fabio Turazza, Alessandro Neri, Marcello Pietri, Maria Angela Butturi, Marco Picone, Marco Mamei  

**Link**: [PDF](https://arxiv.org/pdf/2602.04384)  

**Abstract**: Effective demand forecasting is crucial for reducing food waste. However, data privacy concerns often hinder collaboration among retailers, limiting the potential for improved predictive accuracy. In this study, we explore the application of Federated Learning (FL) in Sustainable Supply Chain Management (SSCM), with a focus on the grocery retail sector dealing with perishable goods. We develop a baseline predictive model for demand forecasting and waste assessment in an isolated retailer scenario. Subsequently, we introduce a Blockchain-based FL model, trained collaboratively across multiple retailers without direct data sharing. Our preliminary results show that FL models have performance almost equivalent to the ideal setting in which parties share data with each other, and are notably superior to models built by individual parties without sharing data, cutting waste and boosting efficiency. 

---
# Enabling Real-Time Colonoscopic Polyp Segmentation on Commodity CPUs via Ultra-Lightweight Architecture 

**Authors**: Weihao Gao, Zhuo Deng, Zheng Gong, Lan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.04381)  

**Abstract**: Early detection of colorectal cancer hinges on real-time, accurate polyp identification and resection. Yet current high-precision segmentation models rely on GPUs, making them impractical to deploy in primary hospitals, mobile endoscopy units, or capsule robots. To bridge this gap, we present the UltraSeg family, operating in an extreme-compression regime (<0.3 M parameters). UltraSeg-108K (0.108 M parameters) is optimized for single-center data, while UltraSeg-130K (0.13 M parameters) generalizes to multi-center, multi-modal images. By jointly optimizing encoder-decoder widths, incorporating constrained dilated convolutions to enlarge receptive fields, and integrating a cross-layer lightweight fusion module, the models achieve 90 FPS on a single CPU core without sacrificing accuracy. Evaluated on seven public datasets, UltraSeg retains >94% of the Dice score of a 31 M-parameter U-Net while utilizing only 0.4% of its parameters, establishing a strong, clinically viable baseline for the extreme-compression domain and offering an immediately deployable solution for resource-constrained settings. This work provides not only a CPU-native solution for colonoscopy but also a reproducible blueprint for broader minimally invasive surgical vision applications. Source code is publicly available to ensure reproducibility and facilitate future benchmarking. 

---
# Beyond KL Divergence: Policy Optimization with Flexible Bregman Divergences for LLM Reasoning 

**Authors**: Rui Yuan, Mykola Khandoga, Vinay Kumar Sankarapu  

**Link**: [PDF](https://arxiv.org/pdf/2602.04380)  

**Abstract**: Policy optimization methods like Group Relative Policy Optimization (GRPO) and its variants have achieved strong results on mathematical reasoning and code generation tasks. Despite extensive exploration of reward processing strategies and training dynamics, all existing group-based methods exclusively use KL divergence for policy regularization, leaving the choice of divergence function unexplored. We introduce Group-Based Mirror Policy Optimization (GBMPO), a framework that extends group-based policy optimization to flexible Bregman divergences, including hand-designed alternatives (L2 in probability space) and learned neural mirror maps. On GSM8K mathematical reasoning, hand-designed ProbL2-GRPO achieves 86.7% accuracy, improving +5.5 points over the Dr. GRPO baseline. On MBPP code generation, neural mirror maps reach 60.1-60.8% pass@1, with random initialization already capturing most of the benefit. While evolutionary strategies meta-learning provides marginal accuracy improvements, its primary value lies in variance reduction ($\pm$0.2 versus $\pm$0.6) and efficiency gains (15% shorter responses on MBPP), suggesting that random initialization of neural mirror maps is sufficient for most practical applications. These results establish divergence choice as a critical, previously unexplored design dimension in group-based policy optimization for LLM reasoning. 

---
# SparVAR: Exploring Sparsity in Visual AutoRegressive Modeling for Training-Free Acceleration 

**Authors**: Zekun Li, Ning Wang, Tongxin Bai, Changwang Mei, Peisong Wang, Shuang Qiu, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.04361)  

**Abstract**: Visual AutoRegressive (VAR) modeling has garnered significant attention for its innovative next-scale prediction paradigm. However, mainstream VAR paradigms attend to all tokens across historical scales at each autoregressive step. As the next scale resolution grows, the computational complexity of attention increases quartically with resolution, causing substantial latency. Prior accelerations often skip high-resolution scales, which speeds up inference but discards high-frequency details and harms image quality. To address these problems, we present SparVAR, a training-free acceleration framework that exploits three properties of VAR attention: (i) strong attention sinks, (ii) cross-scale activation similarity, and (iii) pronounced locality. Specifically, we dynamically predict the sparse attention pattern of later high-resolution scales from a sparse decision scale, and construct scale self-similar sparse attention via an efficient index-mapping mechanism, enabling high-efficiency sparse attention computation at large scales. Furthermore, we propose cross-scale local sparse attention and implement an efficient block-wise sparse kernel, which achieves $\mathbf{> 5\times}$ faster forward speed than FlashAttention. Extensive experiments demonstrate that the proposed SparseVAR can reduce the generation time of an 8B model producing $1024\times1024$ high-resolution images to the 1s, without skipping the last scales. Compared with the VAR baseline accelerated by FlashAttention, our method achieves a $\mathbf{1.57\times}$ speed-up while preserving almost all high-frequency details. When combined with existing scale-skipping strategies, SparseVAR attains up to a $\mathbf{2.28\times}$ acceleration, while maintaining competitive visual generation quality. Code is available at this https URL. 

---
# Counterfactual Explanations for Hypergraph Neural Networks 

**Authors**: Fabiano Veglianti, Lorenzo Antonelli, Gabriele Tolomei  

**Link**: [PDF](https://arxiv.org/pdf/2602.04360)  

**Abstract**: Hypergraph neural networks (HGNNs) effectively model higher-order interactions in many real-world systems but remain difficult to interpret, limiting their deployment in high-stakes settings.
We introduce CF-HyperGNNExplainer, a counterfactual explanation method for HGNNs that identifies the minimal structural changes required to alter a model's prediction. The method generates counterfactual hypergraphs using actionable edits limited to removing node-hyperedge incidences or deleting hyperedges, producing concise and structurally meaningful explanations. Experiments on three benchmark datasets show that CF-HyperGNNExplainer generates valid and concise counterfactuals, highlighting the higher-order relations most critical to HGNN decisions. 

---
# VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image 

**Authors**: Teng-Fang Hsiao, Bo-Kai Ruan, Yu-Lun Liu, Hong-Han Shuai  

**Link**: [PDF](https://arxiv.org/pdf/2602.04349)  

**Abstract**: 3D editing has emerged as a critical research area to provide users with flexible control over 3D assets. While current editing approaches predominantly focus on 3D Gaussian Splatting or multi-view images, the direct editing of 3D meshes remains underexplored. Prior attempts, such as VoxHammer, rely on voxel-based representations that suffer from limited resolution and necessitate labor-intensive 3D mask. To address these limitations, we propose \textbf{VecSet-Edit}, the first pipeline that leverages the high-fidelity VecSet Large Reconstruction Model (LRM) as a backbone for mesh editing. Our approach is grounded on a analysis of the spatial properties in VecSet tokens, revealing that token subsets govern distinct geometric regions. Based on this insight, we introduce Mask-guided Token Seeding and Attention-aligned Token Gating strategies to precisely localize target regions using only 2D image conditions. Also, considering the difference between VecSet diffusion process versus voxel we design a Drift-aware Token Pruning to reject geometric outliers during the denoising process. Finally, our Detail-preserving Texture Baking module ensures that we not only preserve the geometric details of original mesh but also the textural information. More details can be found in our project page: this https URL 

---
# UnMaskFork: Test-Time Scaling for Masked Diffusion via Deterministic Action Branching 

**Authors**: Kou Misaki, Takuya Akiba  

**Link**: [PDF](https://arxiv.org/pdf/2602.04344)  

**Abstract**: Test-time scaling strategies have effectively leveraged inference-time compute to enhance the reasoning abilities of Autoregressive Large Language Models. In this work, we demonstrate that Masked Diffusion Language Models (MDLMs) are inherently amenable to advanced search strategies, owing to their iterative and non-autoregressive generation process. To leverage this, we propose UnMaskFork (UMF), a framework that formulates the unmasking trajectory as a search tree and employs Monte Carlo Tree Search to optimize the generation path. In contrast to standard scaling methods relying on stochastic sampling, UMF explores the search space through deterministic partial unmasking actions performed by multiple MDLMs. Our empirical evaluation demonstrates that UMF consistently outperforms existing test-time scaling baselines on complex coding benchmarks, while also exhibiting strong scalability on mathematical reasoning tasks. 

---
# Explicit Uncertainty Modeling for Active CLIP Adaptation with Dual Prompt Tuning 

**Authors**: Qian-Wei Wang, Yaguang Song, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2602.04340)  

**Abstract**: Pre-trained vision-language models such as CLIP exhibit strong transferability, yet adapting them to downstream image classification tasks under limited annotation budgets remains challenging. In active learning settings, the model must select the most informative samples for annotation from a large pool of unlabeled data. Existing approaches typically estimate uncertainty via entropy-based criteria or representation clustering, without explicitly modeling uncertainty from the model perspective. In this work, we propose a robust uncertainty modeling framework for active CLIP adaptation based on dual-prompt tuning. We introduce two learnable prompts in the textual branch of CLIP. The positive prompt enhances the discriminability of task-specific textual embeddings corresponding to light-weight tuned visual embeddings, improving classification reliability. Meanwhile, the negative prompt is trained in an reversed manner to explicitly model the probability that the predicted label is correct, providing a principled uncertainty signal for guiding active sample selection. Extensive experiments across different fine-tuning paradigms demonstrate that our method consistently outperforms existing active learning methods under the same annotation budget. 

---
# Fine-tuning Pre-trained Vision-Language Models in a Human-Annotation-Free Manner 

**Authors**: Qian-Wei Wang, Guanghao Meng, Ren Cai, Yaguang Song, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2602.04337)  

**Abstract**: Large-scale vision-language models (VLMs) such as CLIP exhibit strong zero-shot generalization, but adapting them to downstream tasks typically requires costly labeled data. Existing unsupervised self-training methods rely on pseudo-labeling, yet often suffer from unreliable confidence filtering, confirmation bias, and underutilization of low-confidence samples. We propose Collaborative Fine-Tuning (CoFT), an unsupervised adaptation framework that leverages unlabeled data through a dual-model, cross-modal collaboration mechanism. CoFT introduces a dual-prompt learning strategy with positive and negative textual prompts to explicitly model pseudo-label cleanliness in a sample-dependent manner, removing the need for hand-crafted thresholds or noise assumptions. The negative prompt also regularizes lightweight visual adaptation modules, improving robustness under noisy supervision. CoFT employs a two-phase training scheme, transitioning from parameter-efficient fine-tuning on high-confidence samples to full fine-tuning guided by collaboratively filtered pseudo-labels. Building on CoFT, CoFT+ further enhances adaptation via iterative fine-tuning, momentum contrastive learning, and LLM-generated prompts. Extensive experiments demonstrate consistent gains over existing unsupervised methods and even few-shot supervised baselines. 

---
# Efficient Equivariant High-Order Crystal Tensor Prediction via Cartesian Local-Environment Many-Body Coupling 

**Authors**: Dian Jin, Yancheng Yuan, Xiaoming Tao  

**Link**: [PDF](https://arxiv.org/pdf/2602.04323)  

**Abstract**: End-to-end prediction of high-order crystal tensor properties from atomic structures remains challenging: while spherical-harmonic equivariant models are expressive, their Clebsch-Gordan tensor products incur substantial compute and memory costs for higher-order targets. We propose the Cartesian Environment Interaction Tensor Network (CEITNet), an approach that constructs a multi-channel Cartesian local environment tensor for each atom and performs flexible many-body mixing via a learnable channel-space interaction. By performing learning in channel space and using Cartesian tensor bases to assemble equivariant outputs, CEITNet enables efficient construction of high-order tensor. Across benchmark datasets for order-2 dielectric, order-3 piezoelectric, and order-4 elastic tensor prediction, CEITNet surpasses prior high-order prediction methods on key accuracy criteria while offering high computational efficiency. 

---
# DeFrame: Debiasing Large Language Models Against Framing Effects 

**Authors**: Kahee Lim, Soyeon Kim, Steven Euijong Whang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04306)  

**Abstract**: As large language models (LLMs) are increasingly deployed in real-world applications, ensuring their fair responses across demographics has become crucial. Despite many efforts, an ongoing challenge is hidden bias: LLMs appear fair under standard evaluations, but can produce biased responses outside those evaluation settings. In this paper, we identify framing -- differences in how semantically equivalent prompts are expressed (e.g., "A is better than B" vs. "B is worse than A") -- as an underexplored contributor to this gap. We first introduce the concept of "framing disparity" to quantify the impact of framing on fairness evaluation. By augmenting fairness evaluation benchmarks with alternative framings, we find that (1) fairness scores vary significantly with framing and (2) existing debiasing methods improve overall (i.e., frame-averaged) fairness, but often fail to reduce framing-induced disparities. To address this, we propose a framing-aware debiasing method that encourages LLMs to be more consistent across framings. Experiments demonstrate that our approach reduces overall bias and improves robustness against framing disparities, enabling LLMs to produce fairer and more consistent responses. 

---
# Beyond Static Cropping: Layer-Adaptive Visual Localization and Decoding Enhancement 

**Authors**: Zipeng Zhu, Zhanghao Hu, Qinglin Zhu, Yuxi Hong, Yijun Liu, Jingyong Su, Yulan He, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2602.04304)  

**Abstract**: Large Vision-Language Models (LVLMs) have advanced rapidly by aligning visual patches with the text embedding space, but a fixed visual-token budget forces images to be resized to a uniform pretraining resolution, often erasing fine-grained details and causing hallucinations via over-reliance on language priors. Recent attention-guided enhancement (e.g., cropping or region-focused attention allocation) alleviates this, yet it commonly hinges on a static "magic layer" empirically chosen on simple recognition benchmarks and thus may not transfer to complex reasoning tasks. In contrast to this static assumption, we propose a dynamic perspective on visual grounding. Through a layer-wise sensitivity analysis, we demonstrate that visual grounding is a dynamic process: while simple object recognition tasks rely on middle layers, complex visual search and reasoning tasks require visual information to be reactivated at deeper layers. Based on this observation, we introduce Visual Activation by Query (VAQ), a metric that identifies the layer whose attention map is most relevant to query-specific visual grounding by measuring attention sensitivity to the input query. Building on VAQ, we further propose LASER (Layer-adaptive Attention-guided Selective visual and decoding Enhancement for Reasoning), a training-free inference procedure that adaptively selects task-appropriate layers for visual localization and question answering. Experiments across diverse VQA benchmarks show that LASER significantly improves VQA accuracy across tasks with varying levels of complexity. 

---
# Revisiting Prompt Sensitivity in Large Language Models for Text Classification: The Role of Prompt Underspecification 

**Authors**: Branislav Pecher, Michal Spiegel, Robert Belanec, Jan Cegin  

**Link**: [PDF](https://arxiv.org/pdf/2602.04297)  

**Abstract**: Large language models (LLMs) are widely used as zero-shot and few-shot classifiers, where task behaviour is largely controlled through prompting. A growing number of works have observed that LLMs are sensitive to prompt variations, with small changes leading to large changes in performance. However, in many cases, the investigation of sensitivity is performed using underspecified prompts that provide minimal task instructions and weakly constrain the model's output space. In this work, we argue that a significant portion of the observed prompt sensitivity can be attributed to prompt underspecification. We systematically study and compare the sensitivity of underspecified prompts and prompts that provide specific instructions. Utilising performance analysis, logit analysis, and linear probing, we find that underspecified prompts exhibit higher performance variance and lower logit values for relevant tokens, while instruction-prompts suffer less from such problems. However, linear probing analysis suggests that the effects of prompt underspecification have only a marginal impact on the internal LLM representations, instead emerging in the final layers. Overall, our findings highlight the need for more rigour when investigating and mitigating prompt sensitivity. 

---
# ProxyWar: Dynamic Assessment of LLM Code Generation in Game Arenas 

**Authors**: Wenjun Peng, Xinyu Wang, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.04296)  

**Abstract**: Large language models (LLMs) have revolutionized automated code generation, yet the evaluation of their real-world effectiveness remains limited by static benchmarks and simplistic metrics. We present ProxyWar, a novel framework that systematically assesses code generation quality by embedding LLM-generated agents within diverse, competitive game environments. Unlike existing approaches, ProxyWar evaluates not only functional correctness but also the operational characteristics of generated programs, combining automated testing, iterative code repair, and multi-agent tournaments to provide a holistic view of program behavior. Applied to a range of state-of-the-art coders and games, our approach uncovers notable discrepancies between benchmark scores and actual performance in dynamic settings, revealing overlooked limitations and opportunities for improvement. These findings highlight the need for richer, competition-based evaluation of code generation. Looking forward, ProxyWar lays a foundation for research into LLM-driven algorithm discovery, adaptive problem solving, and the study of practical efficiency and robustness, including the potential for models to outperform hand-crafted agents. The project is available at this https URL. 

---
# How Few-shot Demonstrations Affect Prompt-based Defenses Against LLM Jailbreak Attacks 

**Authors**: Yanshu Wang, Shuaishuai Yang, Jingjing He, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04294)  

**Abstract**: Large Language Models (LLMs) face increasing threats from jailbreak attacks that bypass safety alignment. While prompt-based defenses such as Role-Oriented Prompts (RoP) and Task-Oriented Prompts (ToP) have shown effectiveness, the role of few-shot demonstrations in these defense strategies remains unclear. Prior work suggests that few-shot examples may compromise safety, but lacks investigation into how few-shot interacts with different system prompt strategies. In this paper, we conduct a comprehensive evaluation on multiple mainstream LLMs across four safety benchmarks (AdvBench, HarmBench, SG-Bench, XSTest) using six jailbreak attack methods. Our key finding reveals that few-shot demonstrations produce opposite effects on RoP and ToP: few-shot enhances RoP's safety rate by up to 4.5% through reinforcing role identity, while it degrades ToP's effectiveness by up to 21.2% through distracting attention from task instructions. Based on these findings, we provide practical recommendations for deploying prompt-based defenses in real-world LLM applications. 

---
# Disentangling Causal Importance from Emergent Structure in Multi-Expert Orchestration 

**Authors**: Sudipto Ghosh, Sujoy Nath, Sunny Manchanda, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2602.04291)  

**Abstract**: Multi-expert systems, where multiple Large Language Models (LLMs) collaborate to solve complex tasks, are increasingly adopted for high-performance reasoning and generation. However, the orchestration policies governing expert interaction and sequencing remain largely opaque. We introduce INFORM, an interpretability analysis that treats orchestration as an explicit, analyzable computation, enabling the decoupling of expert interaction structure, execution order, and causal attribution. We use INFORM to evaluate an orchestrator on GSM8K, HumanEval, and MMLU using a homogeneous consortium of ten instruction-tuned experts drawn from LLaMA-3.1 8B, Qwen-3 8B, and DeepSeek-R1 8B, with controlled decoding-temperature variation, and a secondary heterogeneous consortium spanning 1B-7B parameter models. Across tasks, routing dominance is a poor proxy for functional necessity. We reveal a divergence between relational importance, captured by routing mass and interaction topology, and intrinsic importance, measured via gradient-based causal attribution: frequently selected experts often act as interaction hubs with limited causal influence, while sparsely routed experts can be structurally critical. Orchestration behaviors emerge asynchronously, with expert centralization preceding stable routing confidence and expert ordering remaining non-deterministic. Targeted ablations show that masking intrinsically important experts induces disproportionate collapse in interaction structure compared to masking frequent peers, confirming that INFORM exposes causal and structural dependencies beyond accuracy metrics alone. 

---
# Contextual Drag: How Errors in the Context Affect LLM Reasoning 

**Authors**: Yun Cheng, Xingyu Zhu, Haoyu Zhao, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2602.04288)  

**Abstract**: Central to many self-improvement pipelines for large language models (LLMs) is the assumption that models can improve by reflecting on past mistakes. We study a phenomenon termed contextual drag: the presence of failed attempts in the context biases subsequent generations toward structurally similar errors. Across evaluations of 11 proprietary and open-weight models on 8 reasoning tasks, contextual drag induces 10-20% performance drops, and iterative self-refinement in models with severe contextual drag can collapse into self-deterioration. Structural analysis using tree edit distance reveals that subsequent reasoning trajectories inherit structurally similar error patterns from the context. We demonstrate that neither external feedback nor successful self-verification suffices to eliminate this effect. While mitigation strategies such as fallback-behavior fine-tuning and context denoising yield partial improvements, they fail to fully restore baseline performance, positioning contextual drag as a persistent failure mode in current reasoning architectures. 

---
# Multi Objective Design Optimization of Non Pneumatic Passenger Car Tires Using Finite Element Modeling, Machine Learning, and Particle swarm Optimization and Bayesian Optimization Algorithms 

**Authors**: Priyankkumar Dhrangdhariya, Soumyadipta Maiti, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2602.04277)  

**Abstract**: Non Pneumatic tires offer a promising alternative to pneumatic tires. However, their discontinuous spoke structures present challenges in stiffness tuning, durability, and high speed vibration. This study introduces an integrated generative design and machine learning driven framework to optimize UPTIS type spoke geometries for passenger vehicles. Upper and lower spoke profiles were parameterized using high order polynomial representations, enabling the creation of approximately 250 generative designs through PCHIP based geometric variation. Machine learning models like KRR for stiffness and XGBoost for durability and vibration achieved strong predictive accuracy, reducing the reliance on computationally intensive FEM simulations. Optimization using Particle Swarm Optimization and Bayesian Optimization further enabled extensive performance refinement. The resulting designs demonstrate 53% stiffness tunability, up to 50% durability improvement, and 43% reduction in vibration compared to the baseline. PSO provided fast, targeted convergence, while Bayesian Optimization effectively explored multi objective tradeoffs. Overall, the proposed framework enables systematic development of high performance, next generation UPTIS spoke structures. 

---
# SkeletonGaussian: Editable 4D Generation through Gaussian Skeletonization 

**Authors**: Lifan Wu, Ruijie Zhu, Yubo Ai, Tianzhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04271)  

**Abstract**: 4D generation has made remarkable progress in synthesizing dynamic 3D objects from input text, images, or videos. However, existing methods often represent motion as an implicit deformation field, which limits direct control and editability. To address this issue, we propose SkeletonGaussian, a novel framework for generating editable dynamic 3D Gaussians from monocular video input. Our approach introduces a hierarchical articulated representation that decomposes motion into sparse rigid motion explicitly driven by a skeleton and fine-grained non-rigid motion. Concretely, we extract a robust skeleton and drive rigid motion via linear blend skinning, followed by a hexplane-based refinement for non-rigid deformations, enhancing interpretability and editability. Experimental results demonstrate that SkeletonGaussian surpasses existing methods in generation quality while enabling intuitive motion editing, establishing a new paradigm for editable 4D generation. Project page: this https URL 

---
# Thickening-to-Thinning: Reward Shaping via Human-Inspired Learning Dynamics for LLM Reasoning 

**Authors**: Wenze Lin, Zhen Yang, Xitai Jiang, Pony Ma, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04265)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising paradigm for enhancing reasoning in Large Language Models (LLMs). However, it frequently encounters challenges such as entropy collapse, excessive verbosity, and insufficient exploration for hard problems. Crucially, existing reward schemes fail to distinguish between the need for extensive search during problem-solving and the efficiency required for mastered knowledge. In this work, we introduce T2T(Thickening-to-Thinning), a dynamic reward framework inspired by human learning processes. Specifically, it implements a dual-phase mechanism: (1) On incorrect attempts, T2T incentivizes "thickening" (longer trajectories) to broaden the search space and explore novel solution paths; (2) Upon achieving correctness, it shifts to "thinning", imposing length penalties to discourage redundancy, thereby fostering model confidence and crystallizing reasoning capabilities. Extensive experiments on mathematical benchmarks (MATH-500, AIME, AMC) across Qwen-series and Deepseek models demonstrate that T2T significantly outperforms standard GRPO and recent baselines, achieving superior performance. 

---
# From Dead Neurons to Deep Approximators: Deep Bernstein Networks as a Provable Alternative to Residual Layers 

**Authors**: Ibrahim Albool, Malak Gamal El-Din, Salma Elmalaki, Yasser Shoukry  

**Link**: [PDF](https://arxiv.org/pdf/2602.04264)  

**Abstract**: Residual connections are the de facto standard for mitigating vanishing gradients, yet they impose structural constraints and fail to address the inherent inefficiencies of piecewise linear activations. We show that Deep Bernstein Networks (which utilizes Bernstein polynomials as activation functions) can act as residual-free architecture while simultaneously optimize trainability and representation power. We provide a two-fold theoretical foundation for our approach. First, we derive a theoretical lower bound on the local derivative, proving it remains strictly bounded away from zero. This directly addresses the root cause of gradient stagnation; empirically, our architecture reduces ``dead'' neurons from 90\% in standard deep networks to less than 5\%, outperforming ReLU, Leaky ReLU, SeLU, and GeLU. Second, we establish that the approximation error for Bernstein-based networks decays exponentially with depth, a significant improvement over the polynomial rates of ReLU-based architectures. By unifying these results, we demonstrate that Bernstein activations provide a superior mechanism for function approximation and signal flow. Our experiments on HIGGS and MNIST confirm that Deep Bernstein Networks achieve high-performance training without skip-connections, offering a principled path toward deep, residual-free architectures with enhanced expressive capacity. 

---
# AppleVLM: End-to-end Autonomous Driving with Advanced Perception and Planning-Enhanced Vision-Language Models 

**Authors**: Yuxuan Han, Kunyuan Wu, Qianyi Shao, Renxiang Xiao, Zilu Wang, Cansen Jiang, Yi Xiao, Liang Hu, Yunjiang Lou  

**Link**: [PDF](https://arxiv.org/pdf/2602.04256)  

**Abstract**: End-to-end autonomous driving has emerged as a promising paradigm integrating perception, decision-making, and control within a unified learning framework. Recently, Vision-Language Models (VLMs) have gained significant attention for their potential to enhance the robustness and generalization of end-to-end driving models in diverse and unseen scenarios. However, existing VLM-based approaches still face challenges, including suboptimal lane perception, language understanding biases, and difficulties in handling corner cases. To address these issues, we propose AppleVLM, an advanced perception and planning-enhanced VLM model for robust end-to-end driving. AppleVLM introduces a novel vision encoder and a planning strategy encoder to improve perception and decision-making. Firstly, the vision encoder fuses spatial-temporal information from multi-view images across multiple timesteps using a deformable transformer mechanism, enhancing robustness to camera variations and facilitating scalable deployment across different vehicle platforms. Secondly, unlike traditional VLM-based approaches, AppleVLM introduces a dedicated planning modality that encodes explicit Bird's-Eye-View spatial information, mitigating language biases in navigation instructions. Finally, a VLM decoder fine-tuned by a hierarchical Chain-of-Thought integrates vision, language, and planning features to output robust driving waypoints. We evaluate AppleVLM in closed-loop experiments on two CARLA benchmarks, achieving state-of-the-art driving performance. Furthermore, we deploy AppleVLM on an AGV platform and successfully showcase real-world end-to-end autonomous driving in complex outdoor environments. 

---
# ACIL: Active Class Incremental Learning for Image Classification 

**Authors**: Aditya R. Bhattacharya, Debanjan Goswami, Shayok Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2602.04252)  

**Abstract**: Continual learning (or class incremental learning) is a realistic learning scenario for computer vision systems, where deep neural networks are trained on episodic data, and the data from previous episodes are generally inaccessible to the model. Existing research in this domain has primarily focused on avoiding catastrophic forgetting, which occurs due to the continuously changing class distributions in each episode and the inaccessibility of the data from previous episodes. However, these methods assume that all the training samples in every episode are annotated; this not only incurs a huge annotation cost, but also results in a wastage of annotation effort, since most of the samples in a given episode will not be accessible to the model in subsequent episodes. Active learning algorithms identify the salient and informative samples from large amounts of unlabeled data and are instrumental in reducing the human annotation effort in inducing a deep neural network. In this paper, we propose ACIL, a novel active learning framework for class incremental learning settings. We exploit a criterion based on uncertainty and diversity to identify the exemplar samples that need to be annotated in each episode, and will be appended to the data in the next episode. Such a framework can drastically reduce annotation cost and can also avoid catastrophic forgetting. Our extensive empirical analyses on several vision datasets corroborate the promise and potential of our framework against relevant baselines. 

---
# RAPO: Risk-Aware Preference Optimization for Generalizable Safe Reasoning 

**Authors**: Zeming Wei, Qiaosheng Zhang, Xia Hu, Xingcheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.04224)  

**Abstract**: Large Reasoning Models (LRMs) have achieved tremendous success with their chain-of-thought (CoT) reasoning, yet also face safety issues similar to those of basic language models. In particular, while algorithms are designed to guide them to deliberately refuse harmful prompts with safe reasoning, this process often fails to generalize against diverse and complex jailbreak attacks. In this work, we attribute these failures to the generalization of the safe reasoning process, particularly their insufficiency against complex attack prompts. We provide both theoretical and empirical evidence to show the necessity of a more sufficient safe reasoning process to defend against advanced attack prompts. Building on this insight, we propose a Risk-Aware Preference Optimization (RAPO) framework that enables LRM to adaptively identify and address the safety risks with appropriate granularity in its thinking content. Extensive experiments demonstrate that RAPO successfully generalizes multiple LRMs' safe reasoning adaptively across diverse attack prompts whilst preserving general utility, contributing a robust alignment technique for LRM safety. Our code is available at this https URL. 

---
# OAT: Ordered Action Tokenization 

**Authors**: Chaoqi Liu, Xiaoshen Han, Jiawei Gao, Yue Zhao, Haonan Chen, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2602.04215)  

**Abstract**: Autoregressive policies offer a compelling foundation for scalable robot learning by enabling discrete abstraction, token-level reasoning, and flexible inference. However, applying autoregressive modeling to continuous robot actions requires an effective action tokenization scheme. Existing approaches either rely on analytical discretization methods that produce prohibitively long token sequences, or learned latent tokenizers that lack structure, limiting their compatibility with next-token prediction. In this work, we identify three desiderata for action tokenization - high compression, total decodability, and a left-to-right causally ordered token space - and introduce Ordered Action Tokenization (OAT), a learned action tokenizer that satisfies all three. OAT discretizes action chunks into an ordered sequence of tokens using transformer with registers, finite scalar quantization, and ordering-inducing training mechanisms. The resulting token space aligns naturally with autoregressive generation and enables prefix-based detokenization, yielding an anytime trade-off between inference cost and action fidelity. Across more than 20 tasks spanning four simulation benchmarks and real-world settings, autoregressive policies equipped with OAT consistently outperform prior tokenization schemes and diffusion-based baselines, while offering significantly greater flexibility at inference time. 

---
# Language Models Struggle to Use Representations Learned In-Context 

**Authors**: Michael A. Lepori, Tal Linzen, Ann Yuan, Katja Filippova  

**Link**: [PDF](https://arxiv.org/pdf/2602.04212)  

**Abstract**: Though large language models (LLMs) have enabled great success across a wide variety of tasks, they still appear to fall short of one of the loftier goals of artificial intelligence research: creating an artificial system that can adapt its behavior to radically new contexts upon deployment. One important step towards this goal is to create systems that can induce rich representations of data that are seen in-context, and then flexibly deploy these representations to accomplish goals. Recently, Park et al. (2024) demonstrated that current LLMs are indeed capable of inducing such representation from context (i.e., in-context representation learning). The present study investigates whether LLMs can use these representations to complete simple downstream tasks.
We first assess whether open-weights LLMs can use in-context representations for next-token prediction, and then probe models using a novel task, adaptive world modeling. In both tasks, we find evidence that open-weights LLMs struggle to deploy representations of novel semantics that are defined in-context, even if they encode these semantics in their latent representations. Furthermore, we assess closed-source, state-of-the-art reasoning models on the adaptive world modeling task, demonstrating that even the most performant LLMs cannot reliably leverage novel patterns presented in-context. Overall, this work seeks to inspire novel methods for encouraging models to not only encode information presented in-context, but to do so in a manner that supports flexible deployment of this information. 

---
# SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models 

**Authors**: Hyeonbeom Choi, Daechul Ahn, Youhan Lee, Taewook Kang, Seongwon Cho, Jonghyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2602.04208)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as a promising paradigm for general-purpose robotic control, with test-time scaling (TTS) gaining attention to enhance robustness beyond training. However, existing TTS methods for VLAs require additional training, verifiers, and multiple forward passes, making them impractical for deployment. Moreover, they intervene only at action decoding while keeping visual representations fixed-insufficient under perceptual ambiguity, where reconsidering how to perceive is as important as deciding what to do. To address these limitations, we propose SCALE, a simple inference strategy that jointly modulates visual perception and action based on 'self-uncertainty', inspired by uncertainty-driven exploration in Active Inference theory-requiring no additional training, no verifier, and only a single forward pass. SCALE broadens exploration in both perception and action under high uncertainty, while focusing on exploitation when confident-enabling adaptive execution across varying conditions. Experiments on simulated and real-world benchmarks demonstrate that SCALE improves state-of-the-art VLAs and outperforms existing TTS methods while maintaining single-pass efficiency. 

---
# Enforcing Monotonic Progress in Legal Cross-Examination: Preventing Long-Horizon Stagnation in LLM-Based Inquiry 

**Authors**: Hsien-Jyh Liao  

**Link**: [PDF](https://arxiv.org/pdf/2602.04206)  

**Abstract**: Large language models (LLMs) exhibit impressive linguistic fluency but struggle to reliably complete long-horizon tasks under explicit procedural constraints. In legal cross-examination, purely proba-bilistic generation often maintains behavioral coherence while failing to ensure procedural advancement. We characterize this failure as procedural stagnation and propose Soft-FSM, a neuro-symbolic architecture that enforces monotonic progress over accumulated Key Information Units (KIUs) via an external deterministic state controller. Experiments on three real-world Taiwanese criminal homicide cases show that baseline methods collapse below 40% completeness, while Soft-FSM consistently achieves over 97% with near-zero redundancy. These results suggest that, in such domains, reliable task completion cannot be guaranteed by emergent LLM behavior alone, and can be reliably enforced through explicit and verifiable external state control. 

---
# From Helpfulness to Toxic Proactivity: Diagnosing Behavioral Misalignment in LLM Agents 

**Authors**: Xinyue Wang, Yuanhe Zhang, Zhengshuo Gong, Haoran Gao, Fanyu Meng, Zhenhong Zhou, Li Sun, Yang Liu, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2602.04197)  

**Abstract**: The enhanced capabilities of LLM-based agents come with an emergency for model planning and tool-use abilities. Attributing to helpful-harmless trade-off from LLM alignment, agents typically also inherit the flaw of "over-refusal", which is a passive failure mode. However, the proactive planning and action capabilities of agents introduce another crucial danger on the other side of the trade-off. This phenomenon we term "Toxic Proactivity'': an active failure mode in which an agent, driven by the optimization for Machiavellian helpfulness, disregards ethical constraints to maximize utility. Unlike over-refusal, Toxic Proactivity manifests as the agent taking excessive or manipulative measures to ensure its "usefulness'' is maintained. Existing research pays little attention to identifying this behavior, as it often lacks the subtle context required for such strategies to unfold. To reveal this risk, we introduce a novel evaluation framework based on dilemma-driven interactions between dual models, enabling the simulation and analysis of agent behavior over multi-step behavioral trajectories. Through extensive experiments with mainstream LLMs, we demonstrate that Toxic Proactivity is a widespread behavioral phenomenon and reveal two major tendencies. We further present a systematic benchmark for evaluating Toxic Proactive behavior across contextual settings. 

---
# Natural Language Instructions for Scene-Responsive Human-in-the-Loop Motion Planning in Autonomous Driving using Vision-Language-Action Models 

**Authors**: Angel Martinez-Sanchez, Parthib Roy, Ross Greer  

**Link**: [PDF](https://arxiv.org/pdf/2602.04184)  

**Abstract**: Instruction-grounded driving, where passenger language guides trajectory planning, requires vehicles to understand intent before motion. However, most prior instruction-following planners rely on simulation or fixed command vocabularies, limiting real-world generalization. doScenes, the first real-world dataset linking free-form instructions (with referentiality) to nuScenes ground-truth motion, enables instruction-conditioned planning. In this work, we adapt OpenEMMA, an open-source MLLM-based end-to-end driving framework that ingests front-camera views and ego-state and outputs 10-step speed-curvature trajectories, to this setting, presenting a reproducible instruction-conditioned baseline on doScenes and investigate the effects of human instruction prompts on predicted driving behavior. We integrate doScenes directives as passenger-style prompts within OpenEMMA's vision-language interface, enabling linguistic conditioning before trajectory generation. Evaluated on 849 annotated scenes using ADE, we observe that instruction conditioning substantially improves robustness by preventing extreme baseline failures, yielding a 98.7% reduction in mean ADE. When such outliers are removed, instructions still influence trajectory alignment, with well-phrased prompts improving ADE by up to 5.1%. We use this analysis to discuss what makes a "good" instruction for the OpenEMMA framework. We release the evaluation prompts and scripts to establish a reproducible baseline for instruction-aware planning. GitHub: this https URL 

---
# HoloEv-Net: Efficient Event-based Action Recognition via Holographic Spatial Embedding and Global Spectral Gating 

**Authors**: Weidong Hao  

**Link**: [PDF](https://arxiv.org/pdf/2602.04182)  

**Abstract**: Event-based Action Recognition (EAR) has attracted significant attention due to the high temporal resolution and high dynamic range of event cameras. However, existing methods typically suffer from (i) the computational redundancy of dense voxel representations, (ii) structural redundancy inherent in multi-branch architectures, and (iii) the under-utilization of spectral information in capturing global motion patterns. To address these challenges, we propose an efficient EAR framework named HoloEv-Net. First, to simultaneously tackle representation and structural redundancies, we introduce a Compact Holographic Spatiotemporal Representation (CHSR). Departing from computationally expensive voxel grids, CHSR implicitly embeds horizontal spatial cues into the Time-Height (T-H) view, effectively preserving 3D spatiotemporal contexts within a 2D representation. Second, to exploit the neglected spectral cues, we design a Global Spectral Gating (GSG) module. By leveraging the Fast Fourier Transform (FFT) for global token mixing in the frequency domain, GSG enhances the representation capability with negligible parameter overhead. Extensive experiments demonstrate the scalability and effectiveness of our framework. Specifically, HoloEv-Net-Base achieves state-of-the-art performance on THU-EACT-50-CHL, HARDVS and DailyDVS-200, outperforming existing methods by 10.29%, 1.71% and 6.25%, respectively. Furthermore, our lightweight variant, HoloEv-Net-Small, delivers highly competitive accuracy while offering extreme efficiency, reducing parameters by 5.4 times, FLOPs by 300times, and latency by 2.4times compared to heavy baselines, demonstrating its potential for edge deployment. 

---
# Topology-Aware Revival for Efficient Sparse Training 

**Authors**: Meiling Jin, Fei Wang, Xiaoyun Yuan, Chen Qian, Yuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.04166)  

**Abstract**: Static sparse training is a promising route to efficient learning by committing to a fixed mask pattern, yet the constrained structure reduces robustness. Early pruning decisions can lock the network into a brittle structure that is difficult to escape, especially in deep reinforcement learning (RL) where the evolving policy continually shifts the training distribution. We propose Topology-Aware Revival (TAR), a lightweight one-shot post-pruning procedure that improves static sparsity without dynamic rewiring. After static pruning, TAR performs a single revival step by allocating a small reserve budget across layers according to topology needs, randomly uniformly reactivating a few previously pruned connections within each layer, and then keeping the resulting connectivity fixed for the remainder of training. Across multiple continuous-control tasks with SAC and TD3, TAR improves final return over static sparse baselines by up to +37.9% and also outperforms dynamic sparse training baselines with a median gain of +13.5%. 

---
# Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity 

**Authors**: Chenhe Du, Qing Wu, Xuanyu Tian, Jingyi Yu, Hongjiang Wei, Yuyao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04162)  

**Abstract**: 3D medical imaging is in high demand and essential for clinical diagnosis and scientific research. Currently, diffusion models (DMs) have become an effective tool for medical imaging reconstruction thanks to their ability to learn rich, high-quality data priors. However, learning the 3D data distribution with DMs in medical imaging is challenging, not only due to the difficulties in data collection but also because of the significant computational burden during model training. A common compromise is to train the DMs on 2D data priors and reconstruct stacked 2D slices to address 3D medical inverse problems. However, the intrinsic randomness of diffusion sampling causes severe inter-slice discontinuities of reconstructed 3D volumes. Existing methods often enforce continuity regularizations along the z-axis, which introduces sensitive hyper-parameters and may lead to over-smoothing results. In this work, we revisit the origin of stochasticity in diffusion sampling and introduce Inter-Slice Consistent Stochasticity (ISCS), a simple yet effective strategy that encourages interslice consistency during diffusion sampling. Our key idea is to control the consistency of stochastic noise components during diffusion sampling, thereby aligning their sampling trajectories without adding any new loss terms or optimization steps. Importantly, the proposed ISCS is plug-and-play and can be dropped into any 2D trained diffusion based 3D reconstruction pipeline without additional computational cost. Experiments on several medical imaging problems show that our method can effectively improve the performance of medical 3D imaging problems based on 2D diffusion models. Our findings suggest that controlling inter-slice stochasticity is a principled and practically attractive route toward high-fidelity 3D medical imaging with 2D diffusion priors. The code is available at: this https URL 

---
# Pruning for Generalization: A Transfer-Oriented Spatiotemporal Graph Framework 

**Authors**: Zihao Jing, Yuxi Long, Ganlin Feng  

**Link**: [PDF](https://arxiv.org/pdf/2602.04153)  

**Abstract**: Multivariate time series forecasting in graph-structured domains is critical for real-world applications, yet existing spatiotemporal models often suffer from performance degradation under data scarcity and cross-domain shifts. We address these challenges through the lens of structure-aware context selection. We propose TL-GPSTGN, a transfer-oriented spatiotemporal framework that enhances sample efficiency and out-of-distribution generalization by selectively pruning non-optimized graph context. Specifically, our method employs information-theoretic and correlation-based criteria to extract structurally informative subgraphs and features, resulting in a compact, semantically grounded representation. This optimized context is subsequently integrated into a spatiotemporal convolutional architecture to capture complex multivariate dynamics. Evaluations on large-scale traffic benchmarks demonstrate that TL-GPSTGN consistently outperforms baselines in low-data transfer scenarios. Our findings suggest that explicit context pruning serves as a powerful inductive bias for improving the robustness of graph-based forecasting models. 

---
# MA3DSG: Multi-Agent 3D Scene Graph Generation for Large-Scale Indoor Environments 

**Authors**: Yirum Kim, Jaewoo Kim, Ue-Hwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.04152)  

**Abstract**: Current 3D scene graph generation (3DSGG) approaches heavily rely on a single-agent assumption and small-scale environments, exhibiting limited scalability to real-world scenarios. In this work, we introduce Multi-Agent 3D Scene Graph Generation (MA3DSG) model, the first framework designed to tackle this scalability challenge using multiple agents. We develop a training-free graph alignment algorithm that efficiently merges partial query graphs from individual agents into a unified global scene graph. Leveraging extensive analysis and empirical insights, our approach enables conventional single-agent systems to operate collaboratively without requiring any learnable parameters. To rigorously evaluate 3DSGG performance, we propose MA3DSG-Bench-a benchmark that supports diverse agent configurations, domain sizes, and environmental conditions-providing a more general and extensible evaluation framework. This work lays a solid foundation for scalable, multi-agent 3DSGG research. 

---
# JSynFlow: Japanese Synthesised Flowchart Visual Question Answering Dataset built with Large Language Models 

**Authors**: Hiroshi Sasaki  

**Link**: [PDF](https://arxiv.org/pdf/2602.04142)  

**Abstract**: Vision and language models (VLMs) are expected to analyse complex documents, such as those containing flowcharts, through a question-answering (QA) interface. The ability to recognise and interpret these flowcharts is in high demand, as they provide valuable insights unavailable in text-only explanations. However, developing VLMs with precise flowchart understanding requires large-scale datasets of flowchart images and corresponding text, the creation of which is highly time-consuming. To address this challenge, we introduce JSynFlow, a synthesised visual QA dataset for Japanese flowcharts, generated using large language models (LLMs). Our dataset comprises task descriptions for various business occupations, the corresponding flowchart images rendered from domain-specific language (DSL) code, and related QA pairs. This paper details the dataset's synthesis procedure and demonstrates that fine-tuning with JSynFlow significantly improves VLM performance on flowchart-based QA tasks. Our dataset is publicly available at this https URL. 

---
# KGLAMP: Knowledge Graph-guided Language model for Adaptive Multi-robot Planning and Replanning 

**Authors**: Chak Lam Shek, Faizan M. Tariq, Sangjae Bae, David Isele, Piyush Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2602.04129)  

**Abstract**: Heterogeneous multi-robot systems are increasingly deployed in long-horizon missions that require coordination among robots with diverse capabilities. However, existing planning approaches struggle to construct accurate symbolic representations and maintain plan consistency in dynamic environments. Classical PDDL planners require manually crafted symbolic models, while LLM-based planners often ignore agent heterogeneity and environmental uncertainty. We introduce KGLAMP, a knowledge-graph-guided LLM planning framework for heterogeneous multi-robot teams. The framework maintains a structured knowledge graph encoding object relations, spatial reachability, and robot capabilities, which guides the LLM in generating accurate PDDL problem specifications. The knowledge graph serves as a persistent, dynamically updated memory that incorporates new observations and triggers replanning upon detecting inconsistencies, enabling symbolic plans to adapt to evolving world states. Experiments on the MAT-THOR benchmark show that KGLAMP improves performance by at least 25.5% over both LLM-only and PDDL-based variants. 

---
# From Lemmas to Dependencies: What Signals Drive Light Verbs Classification? 

**Authors**: Sercan Karakaş, Yusuf Şimşek  

**Link**: [PDF](https://arxiv.org/pdf/2602.04127)  

**Abstract**: Light verb constructions (LVCs) are a challenging class of verbal multiword expressions, especially in Turkish, where rich morphology and productive complex predicates create minimal contrasts between idiomatic predicate meanings and literal verb--argument uses. This paper asks what signals drive LVC classification by systematically restricting model inputs. Using UD-derived supervision, we compare lemma-driven baselines (lemma TF--IDF + Logistic Regression; BERTurk trained on lemma sequences), a grammar-only Logistic Regression over UD morphosyntax (UPOS/DEPREL/MORPH), and a full-input BERTurk baseline. We evaluate on a controlled diagnostic set with Random negatives, lexical controls (NLVC), and LVC positives, reporting split-wise performance to expose decision-boundary behavior. Results show that coarse morphosyntax alone is insufficient for robust LVC detection under controlled contrasts, while lexical identity supports LVC judgments but is sensitive to calibration and normalization choices. Overall, Our findings motivate targeted evaluation of Turkish MWEs and show that ``lemma-only'' is not a single, well-defined representation, but one that depends critically on how normalization is operationalized. 

---
# Scalable Explainability-as-a-Service (XaaS) for Edge AI Systems 

**Authors**: Samaresh Kumar Singh, Joyjit Roy  

**Link**: [PDF](https://arxiv.org/pdf/2602.04120)  

**Abstract**: Though Explainable AI (XAI) has made significant advancements, its inclusion in edge and IoT systems is typically ad-hoc and inefficient. Most current methods are "coupled" in such a way that they generate explanations simultaneously with model inferences. As a result, these approaches incur redundant computation, high latency and poor scalability when deployed across heterogeneous sets of edge devices. In this work we propose Explainability-as-a-Service (XaaS), a distributed architecture for treating explainability as a first-class system service (as opposed to a model-specific feature). The key innovation in our proposed XaaS architecture is that it decouples inference from explanation generation allowing edge devices to request, cache and verify explanations subject to resource and latency constraints. To achieve this, we introduce three main innovations: (1) A distributed explanation cache with a semantic similarity based explanation retrieval method which significantly reduces redundant computation; (2) A lightweight verification protocol that ensures the fidelity of both cached and newly generated explanations; and (3) An adaptive explanation engine that chooses explanation methods based upon device capability and user requirement. We evaluated the performance of XaaS on three real-world edge-AI use cases: (i) manufacturing quality control; (ii) autonomous vehicle perception; and (iii) healthcare diagnostics. Experimental results show that XaaS reduces latency by 38\% while maintaining high explanation quality across three real-world deployments. Overall, this work enables the deployment of transparent and accountable AI across large scale, heterogeneous IoT systems, and bridges the gap between XAI research and edge-practicality. 

---
# Toward Effective Multimodal Graph Foundation Model: A Divide-and-Conquer Based Approach 

**Authors**: Sicheng Liu, Xunkai Li, Daohan Su, Ru Zhang, Hongchao Qin, Ronghua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04116)  

**Abstract**: Graph Foundation Models (GFMs) have achieved remarkable success in generalizing across diverse domains. However, they mainly focus on Text-Attributed Graphs (TAGs), leaving Multimodal-Attributed Graphs (MAGs) largely untapped. Developing Multimodal Graph Foundation Models (MGFMs) allows for leveraging the rich multimodal information in MAGs, and extends applicability to broader types of downstream tasks. While recent MGFMs integrate diverse modality information, our empirical investigation reveals two fundamental limitations of existing MGFMs: (1)they fail to explicitly model modality interaction, essential for capturing intricate cross-modal semantics beyond simple aggregation, and (2)they exhibit sub-optimal modality alignment, which is critical for bridging the significant semantic disparity between distinct modal spaces. To address these challenges, we propose PLANET (graPh topoLogy-aware modAlity iNteraction and alignmEnT), a novel framework employing a Divide-and-Conquer strategy to decouple modality interaction and alignment across distinct granularities. At the embedding granularity, (1)Embedding-wise Domain Gating (EDG) performs local semantic enrichment by adaptively infusing topology-aware cross-modal context, achieving modality interaction. At the node granularity, (2)Node-wise Discretization Retrieval (NDR) ensures global modality alignment by constructing a Discretized Semantic Representation Space (DSRS) to bridge modality gaps. Extensive experiments demonstrate that PLANET significantly outperforms state-of-the-art baselines across diverse graph-centric and multimodal generative tasks. 

---
# Tinker Tales: Supporting Child-AI Collaboration through Co-Creative Storytelling with Educational Scaffolding 

**Authors**: Nayoung Choi, Jiseung Hong, Peace Cyebukayire, Ikseon Choi, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2602.04109)  

**Abstract**: Artificial intelligence (AI) is increasingly framed as a collaborative partner in creative activities, yet children's interactions with AI have largely been studied in AI-led instructional settings rather than co-creative collaboration. This leaves open questions about how children can meaningfully engage with AI through iterative co-creation. We present Tinker Tales, a tangible storytelling system designed with narrative and social-emotional scaffolding to support child-AI collaboration. The system combines a physical storytelling board, NFC-embedded toys representing story elements (e.g., characters, places, items, and emotions), and a mobile app that mediates child-AI interaction. Children shape and refine stories by placing and moving story elements and interacting with the AI through tangible and voice-based interaction. We conducted an exploratory user study with 10 children to examine how they interacted with Tinker Tales. Our findings show that children treated the AI as an attentive, responsive collaborator, while scaffolding supported coherent narrative refinement without diminishing children's agency. 

---
# DMS2F-HAD: A Dual-branch Mamba-based Spatial-Spectral Fusion Network for Hyperspectral Anomaly Detection 

**Authors**: Aayushma Pant, Lakpa Tamang, Tsz-Kwan Lee, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2602.04102)  

**Abstract**: Hyperspectral anomaly detection (HAD) aims to identify rare and irregular targets in high-dimensional hyperspectral images (HSIs), which are often noisy and unlabelled data. Existing deep learning methods either fail to capture long-range spectral dependencies (e.g., convolutional neural networks) or suffer from high computational cost (e.g., Transformers). To address these challenges, we propose DMS2F-HAD, a novel dual-branch Mamba-based model. Our architecture utilizes Mamba's linear-time modeling to efficiently learn distinct spatial and spectral features in specialized branches, which are then integrated by a dynamic gated fusion mechanism to enhance anomaly localization. Across fourteen benchmark HSI datasets, our proposed DMS2F-HAD not only achieves a state-of-the-art average AUC of 98.78%, but also demonstrates superior efficiency with an inference speed 4.6 times faster than comparable deep learning methods. The results highlight DMS2FHAD's strong generalization and scalability, positioning it as a strong candidate for practical HAD applications. 

---
# A computational account of dreaming: learning and memory consolidation 

**Authors**: Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04095)  

**Abstract**: A number of studies have concluded that dreaming is mostly caused by randomly arriving internal signals because "dream contents are random impulses", and argued that dream sleep is unlikely to play an important part in our intellectual capacity. On the contrary, numerous functional studies have revealed that dream sleep does play an important role in our learning and other intellectual functions. Specifically, recent studies have suggested the importance of dream sleep in memory consolidation, following the findings of neural replaying of recent waking patterns in the hippocampus. The randomness has been the hurdle that divides dream theories into either functional or functionless. This study presents a cognitive and computational model of dream process. This model is simulated to perform the functions of learning and memory consolidation, which are two most popular dream functions that have been proposed. The simulations demonstrate that random signals may result in learning and memory consolidation. Thus, dreaming is proposed as a continuation of brain's waking activities that processes signals activated spontaneously and randomly from the hippocampus. The characteristics of the model are discussed and found in agreement with many characteristics concluded from various empirical studies. 

---
# Structure-Informed Estimation for Pilot-Limited MIMO Channels via Tensor Decomposition 

**Authors**: Alexandre Barbosa de Lima  

**Link**: [PDF](https://arxiv.org/pdf/2602.04083)  

**Abstract**: Channel estimation in wideband multiple-input multiple-output (MIMO) systems faces fundamental pilot overhead limitations in high-dimensional beyond-5G and sixth-generation (6G) scenarios. This paper presents a hybrid tensor-neural architecture that formulates pilot-limited channel estimation as low-rank tensor completion from sparse observations -- a fundamentally different setting from prior tensor methods that assume fully observed received signal tensors. A canonical polyadic (CP) baseline implemented via a projection-based scheme (Tucker completion under partial observations) and Tucker decompositions are compared under varying signal-to-noise ratio (SNR) and scattering conditions: CP performs well for specular channels matching the multipath model, while Tucker provides greater robustness under model mismatch. A lightweight three-dimensional (3D) U-Net learns residual components beyond the low-rank structure, bridging algebraic models and realistic propagation effects. Empirical recovery threshold analysis shows that sample complexity scales approximately with intrinsic model dimensionality $L(N_r + N_t + N_f)$ rather than ambient tensor size $N_r N_t N_f$, where $L$ denotes the number of dominant propagation paths. Experiments on synthetic channels demonstrate 10-20\,dB normalized mean-square error (NMSE) improvement over least-squares (LS) and orthogonal matching pursuit (OMP) baselines at 5-10\% pilot density, while evaluations on DeepMIMO ray-tracing channels show 24-44\% additional NMSE reduction over pure tensor-based methods. 

---
# Principles of Lipschitz continuity in neural networks 

**Authors**: Róisín Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.04078)  

**Abstract**: Deep learning has achieved remarkable success across a wide range of domains, significantly expanding the frontiers of what is achievable in artificial intelligence. Yet, despite these advances, critical challenges remain -- most notably, ensuring robustness to small input perturbations and generalization to out-of-distribution data. These critical challenges underscore the need to understand the underlying fundamental principles that govern robustness and generalization. Among the theoretical tools available, Lipschitz continuity plays a pivotal role in governing the fundamental properties of neural networks related to robustness and generalization. It quantifies the worst-case sensitivity of network's outputs to small input perturbations. While its importance is widely acknowledged, prior research has predominantly focused on empirical regularization approaches based on Lipschitz constraints, leaving the underlying principles less explored. This thesis seeks to advance a principled understanding of the principles of Lipschitz continuity in neural networks within the paradigm of machine learning, examined from two complementary perspectives: an internal perspective -- focusing on the temporal evolution of Lipschitz continuity in neural networks during training (i.e., training dynamics); and an external perspective -- investigating how Lipschitz continuity modulates the behavior of neural networks with respect to features in the input data, particularly its role in governing frequency signal propagation (i.e., modulation of frequency signal propagation). 

---
# On the Credibility of Evaluating LLMs using Survey Questions 

**Authors**: Jindřich Libovický  

**Link**: [PDF](https://arxiv.org/pdf/2602.04033)  

**Abstract**: Recent studies evaluate the value orientation of large language models (LLMs) using adapted social surveys, typically by prompting models with survey questions and comparing their responses to average human responses. This paper identifies limitations in this methodology that, depending on the exact setup, can lead to both underestimating and overestimating the similarity of value orientation. Using the World Value Survey in three languages across five countries, we demonstrate that prompting methods (direct vs. chain-of-thought) and decoding strategies (greedy vs. sampling) significantly affect results. To assess the interaction between answers, we introduce a novel metric, self-correlation distance. This metric measures whether LLMs maintain consistent relationships between answers across different questions, as humans do. This indicates that even a high average agreement with human data, when considering LLM responses independently, does not guarantee structural alignment in responses. Additionally, we reveal a weak correlation between two common evaluation metrics, mean-squared distance and KL divergence, which assume that survey answers are independent of each other. For future research, we recommend CoT prompting, sampling-based decoding with dozens of samples, and robust analysis using multiple metrics, including self-correlation distance. 

---
# PluRel: Synthetic Data unlocks Scaling Laws for Relational Foundation Models 

**Authors**: Vignesh Kothapalli, Rishabh Ranjan, Valter Hudovernik, Vijay Prakash Dwivedi, Johannes Hoffart, Carlos Guestrin, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2602.04029)  

**Abstract**: Relational Foundation Models (RFMs) facilitate data-driven decision-making by learning from complex multi-table databases. However, the diverse relational databases needed to train such models are rarely public due to privacy constraints. While there are methods to generate synthetic tabular data of arbitrary size, incorporating schema structure and primary--foreign key connectivity for multi-table generation remains challenging. Here we introduce PluRel, a framework to synthesize multi-tabular relational databases from scratch. In a step-by-step fashion, PluRel models (1) schemas with directed graphs, (2) inter-table primary-foreign key connectivity with bipartite graphs, and, (3) feature distributions in tables via conditional causal mechanisms. The design space across these stages supports the synthesis of a wide range of diverse databases, while being computationally lightweight. Using PluRel, we observe for the first time that (1) RFM pretraining loss exhibits power-law scaling with the number of synthetic databases and total pretraining tokens, (2) scaling the number of synthetic databases improves generalization to real databases, and (3) synthetic pretraining yields strong base models for continued pretraining on real databases. Overall, our framework and results position synthetic data scaling as a promising paradigm for RFMs. 

---
# Understanding and Guiding Layer Placement in Parameter-Efficient Fine-Tuning of Large Language Models 

**Authors**: Yichen Xu, Yuyang Liang, Shan Dai, Tianyang Hu, Tsz Nam Chan, Chenhao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.04019)  

**Abstract**: As large language models (LLMs) continue to grow, the cost of full-parameter fine-tuning has made parameter-efficient fine-tuning (PEFT) the default strategy for downstream adaptation. Constraints from inference latency in scalable serving and fine-tuning cost in edge or rapid-deployment settings make the choice of which layers to fine-tune unavoidable. Yet current practice typically applies PEFT uniformly across all layers, with limited understanding or leverage of layer selection. This paper develops a unified projected residual view of PEFT on top of a frozen base model. Under a local quadratic approximation, layerwise adaptation is governed by three quantities: (i) the projected residual norm (resnorm), which measures how much correctable bias a layer can capture; (ii) the activation energy, which determines feature conditioning; and (iii) layer coupling, which quantifies how strongly residuals interact across layers. We show that, for squared loss and linear adapters, the resnorm equals a normalized gradient norm, activation energy controls ill-conditioning and noise amplification, and weak coupling yields approximately additive layerwise contributions. Building on these insights, we introduce the Layer Card, a reusable diagnostic that summarizes residual signal strength, compute cost, and performance for each layer of a given model. With an identical model and LoRA configuration, Layer Card-guided placement refines the choice of adapted layers to flexibly prioritize different objectives, such as maximizing performance or reducing fine-tuning cost. Moreover, on Qwen3-8B, we show that selectively adapting a subset of layers can achieve performance close to full-layer LoRA while substantially reducing fine-tuning cost and the number of adapter-augmented layers during inference, offering a more cost-performance-aware alternative to full-layer insertion. 

---
# PromptSplit: Revealing Prompt-Level Disagreement in Generative Models 

**Authors**: Mehdi Lotfian, Mohammad Jalali, Farzan Farnia  

**Link**: [PDF](https://arxiv.org/pdf/2602.04009)  

**Abstract**: Prompt-guided generative AI models have rapidly expanded across vision and language domains, producing realistic and diverse outputs from textual inputs. The growing variety of such models, trained with different data and architectures, calls for principled methods to identify which types of prompts lead to distinct model behaviors. In this work, we propose PromptSplit, a kernel-based framework for detecting and analyzing prompt-dependent disagreement between generative models. For each compared model pair, PromptSplit constructs a joint prompt--output representation by forming tensor-product embeddings of the prompt and image (or text) features, and then computes the corresponding kernel covariance matrix. We utilize the eigenspace of the weighted difference between these matrices to identify the main directions of behavioral difference across prompts. To ensure scalability, we employ a random-projection approximation that reduces computational complexity to $O(nr^2 + r^3)$ for projection dimension $r$. We further provide a theoretical analysis showing that this approximation yields an eigenstructure estimate whose expected deviation from the full-dimensional result is bounded by $O(1/r^2)$. Experiments across text-to-image, text-to-text, and image-captioning settings demonstrate that PromptSplit accurately detects ground-truth behavioral differences and isolates the prompts responsible, offering an interpretable tool for detecting where generative models disagree. 

---
# Rational ANOVA Networks 

**Authors**: Jusheng Zhang, Ningyuan Liu, Qinhan Lyu, Jing Yang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04006)  

**Abstract**: Deep neural networks typically treat nonlinearities as fixed primitives (e.g., ReLU), limiting both interpretability and the granularity of control over the induced function class. While recent additive models (like KANs) attempt to address this using splines, they often suffer from computational inefficiency and boundary instability. We propose the Rational-ANOVA Network (RAN), a foundational architecture grounded in functional ANOVA decomposition and Padé-style rational approximation. RAN models f(x) as a composition of main effects and sparse pairwise interactions, where each component is parameterized by a stable, learnable rational unit. Crucially, we enforce a strictly positive denominator, which avoids poles and numerical instability while capturing sharp transitions and near-singular behaviors more efficiently than polynomial bases. This ANOVA structure provides an explicit low-order interaction bias for data efficiency and interpretability, while the rational parameterization significantly improves extrapolation. Across controlled function benchmarks and vision classification tasks (e.g., CIFAR-10) under matched parameter and compute budgets, RAN matches or surpasses parameter-matched MLPs and learnable-activation baselines, with better stability and throughput. Code is available at this https URL. 

---
# When Chains of Thought Don't Matter: Causal Bypass in Large Language Models 

**Authors**: Anish Sathyanarayanan, Aditya Nagarsekar, Aarush Rathore  

**Link**: [PDF](https://arxiv.org/pdf/2602.03994)  

**Abstract**: Chain-of-thought (CoT) prompting is widely assumed to expose a model's reasoning process and improve transparency. We attempted to enforce this assumption by penalizing unfaithful reasoning, but found that surface-level compliance does not guarantee causal reliance. Our central finding is negative: even when CoT is verbose, strategic, and flagged by surface-level manipulation detectors, model answers are often causally independent of the CoT content. We present a diagnostic framework for auditing this failure mode: it combines (i) an interpretable behavioral module that scores manipulation-relevant signals in CoT text and (ii) a causal probe that measures CoT-mediated influence (CMI) via hidden-state patching and reports a bypass score ($1-\mathrm{CMI}$), quantifying the degree to which the answer is produced by a bypass circuit independent of the rationale. In pilot evaluations, audit-aware prompting increases detectable manipulation signals (mean risk-score delta: $+5.10$), yet causal probes reveal task-dependent mediation: many QA items exhibit near-total bypass (CMI $\approx 0$), while some logic problems show stronger mediation (CMI up to $0.56$). Layer-wise analysis reveals narrow and task-dependent ``reasoning windows'' even when mean CMI is low. 

---
# DeXposure-FM: A Time-series, Graph Foundation Model for Credit Exposures and Stability on Decentralized Financial Networks 

**Authors**: Aijie Shu, Wenbin Wu, Gbenga Ibikunle, Fengxiang He  

**Link**: [PDF](https://arxiv.org/pdf/2602.03981)  

**Abstract**: Credit exposure in Decentralized Finance (DeFi) is often implicit and token-mediated, creating a dense web of inter-protocol dependencies. Thus, a shock to one token may result in significant and uncontrolled contagion effects. As the DeFi ecosystem becomes increasingly linked with traditional financial infrastructure through instruments, such as stablecoins, the risk posed by this dynamic demands more powerful quantification tools. We introduce DeXposure-FM, the first time-series, graph foundation model for measuring and forecasting inter-protocol credit exposure on DeFi networks, to the best of our knowledge. Employing a graph-tabular encoder, with pre-trained weight initialization, and multiple task-specific heads, DeXposure-FM is trained on the DeXposure dataset that has 43.7 million data entries, across 4,300+ protocols on 602 blockchains, covering 24,300+ unique tokens. The training is operationalized for credit-exposure forecasting, predicting the joint dynamics of (1) protocol-level flows, and (2) the topology and weights of credit-exposure links. The DeXposure-FM is empirically validated on two machine learning benchmarks; it consistently outperforms the state-of-the-art approaches, including a graph foundation model and temporal graph neural networks. DeXposure-FM further produces financial economics tools that support macroprudential monitoring and scenario-based DeFi stress testing, by enabling protocol-level systemic-importance scores, sector-level spillover and concentration measures via a forecast-then-measure pipeline. Empirical verification fully supports our financial economics tools. The model and code have been publicly available. Model: this https URL.
Code: this https URL. 

---
# Transformers perform adaptive partial pooling 

**Authors**: Vsevolod Kapatsinski  

**Link**: [PDF](https://arxiv.org/pdf/2602.03980)  

**Abstract**: Because language is creative, any reasonable language model must generalize, deciding what to say in novel contexts by using information from similar contexts. But what about contexts that are not novel but merely infrequent? In hierarchical regression, the model's predictions for behavior in a context are affected by observations from other similar contexts to the extent that 1) the current context is infrequent and 2) different contexts behave similarly. This is called adaptive partial pooling of evidence. This paper shows that next-word predictions of a transformer (GPT2) are increasingly unaffected by observations from outside the current context across epochs of training (the amount of pooling reduces with training), and that the extent of pooling is affected by context frequency, context number (type frequency) and context variability in a similar way to hierarchical regression. These characteristics of learning in transformers are argued to be realistic on both rational and empirical grounds. 

---
# Fixed Budget is No Harder Than Fixed Confidence in Best-Arm Identification up to Logarithmic Factors 

**Authors**: Kapilan Balagopalan, Yinan Li, Yao Zhao, Tuan Nguyen, Anton Daitche, Houssam Nassif, Kwang-Sung Jun  

**Link**: [PDF](https://arxiv.org/pdf/2602.03972)  

**Abstract**: The best-arm identification (BAI) problem is one of the most fundamental problems in interactive machine learning, which has two flavors: the fixed-budget setting (FB) and the fixed-confidence setting (FC).
For $K$-armed bandits with the unique best arm, the optimal sample complexities for both settings have been settled down, and they match up to logarithmic factors.
This prompts an interesting research question about the generic, potentially structured BAI problems: Is FB harder than FC or the other way around?
In this paper, we show that FB is no harder than FC up to logarithmic factors.
We do this constructively: we propose a novel algorithm called FC2FB (fixed confidence to fixed budget), which is a meta algorithm that takes in an FC algorithm $\mathcal{A}$ and turn it into an FB algorithm.
We prove that this FC2FB enjoys a sample complexity that matches, up to logarithmic factors, that of the sample complexity of $\mathcal{A}$.
This means that the optimal FC sample complexity is an upper bound of the optimal FB sample complexity up to logarithmic factors.
Our result not only reveals a fundamental relationship between FB and FC, but also has a significant implication: FC2FB, combined with existing state-of-the-art FC algorithms, leads to improved sample complexity for a number of FB problems. 

---
# Structural shifts in institutional participation and collaboration within the AI arXiv preprint research ecosystem 

**Authors**: Shama Magnur, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2602.03969)  

**Abstract**: The emergence of large language models (LLMs) represents a significant technological shift within the scientific ecosystem, particularly within the field of artificial intelligence (AI). This paper examines structural changes in the AI research landscape using a dataset of arXiv preprints (cs.AI) from 2021 through 2025. Given the rapid pace of AI development, the preprint ecosystem has become a critical barometer for real-time scientific shifts, often preceding formal peer-reviewed publication by months or years. By employing a multi-stage data collection and enrichment pipeline in conjunction with LLM-based institution classification, we analyze the evolution of publication volumes, author team sizes, and academic--industry collaboration patterns. Our results reveal an unprecedented surge in publication output following the introduction of ChatGPT, with academic institutions continuing to provide the largest volume of research. However, we observe that academic--industry collaboration is still suppressed, as measured by a Normalized Collaboration Index (NCI) that remains significantly below the random-mixing baseline across all major subfields. These findings highlight a continuing institutional divide and suggest that the capital-intensive nature of generative AI research may be reshaping the boundaries of scientific collaboration. 

---
# Semantic Rate Distortion and Posterior Design: Compute Constraints, Multimodality, and Strategic Inference 

**Authors**: Emrah Akyol  

**Link**: [PDF](https://arxiv.org/pdf/2602.03949)  

**Abstract**: We study strategic Gaussian semantic compression under rate and compute constraints, where an encoder and decoder optimize distinct quadratic objectives. A latent Gaussian state generates a task dependent semantic variable, and the decoder best responds via MMSE estimation, reducing the encoder's problem to posterior covariance design under an information rate constraint. We characterize the strategic rate distortion function in direct, remote, and full information regimes, derive semantic waterfilling and rate constrained Gaussian persuasion solutions, and establish Gaussian optimality under misaligned objectives. We further show that architectural compute limits act as implicit rate constraints, yielding exponential improvements in semantic accuracy with model depth and inference time compute, while multimodal observation eliminates the geometric mean penalty inherent to remote encoding. These results provide information theoretic foundations for data and energy efficient AI and offer a principled interpretation of modern multimodal language models as posterior design mechanisms under resource constraints. 

---
# Linguistic Blind Spots in Clinical Decision Extraction 

**Authors**: Mohamed Elgaar, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2602.03942)  

**Abstract**: Extracting medical decisions from clinical notes is a key step for clinical decision support and patient-facing care summaries. We study how the linguistic characteristics of clinical decisions vary across decision categories and whether these differences explain extraction failures. Using MedDec discharge summaries annotated with decision categories from the Decision Identification and Classification Taxonomy for Use in Medicine (DICTUM), we compute seven linguistic indices for each decision span and analyze span-level extraction recall of a standard transformer model. We find clear category-specific signatures: drug-related and problem-defining decisions are entity-dense and telegraphic, whereas advice and precaution decisions contain more narrative, with higher stopword and pronoun proportions and more frequent hedging and negation cues. On the validation split, exact-match recall is 48%, with large gaps across linguistic strata: recall drops from 58% to 24% from the lowest to highest stopword-proportion bins, and spans containing hedging or negation cues are less likely to be recovered. Under a relaxed overlap-based match criterion, recall increases to 71%, indicating that many errors are span boundary disagreements rather than complete misses. Overall, narrative-style spans--common in advice and precaution decisions--are a consistent blind spot under exact matching, suggesting that downstream systems should incorporate boundary-tolerant evaluation and extraction strategies for clinical decisions. 

---
# First-Principles AI finds crystallization of fractional quantum Hall liquids 

**Authors**: Ahmed Abouelkomsan, Liang Fu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03927)  

**Abstract**: When does a fractional quantum Hall (FQH) liquid crystallize? Addressing this question requires a framework that treats fractionalization and crystallization on equal footing, especially in strong Landau-level mixing regime. Here, we introduce MagNet, a self-attention neural-network variational wavefunction designed for quantum systems in magnetic fields on the torus geometry. We show that MagNet provides a unifying and expressive ansatz capable of describing both FQH states and electron crystals within the same architecture. Trained solely by energy minimization of the microscopic Hamiltonian, MagNet discovers topological liquid and electron crystal ground states across a broad range of Landau-level mixing. Our results highlight the power of first-principles AI for solving strongly interacting many-body problems and finding competing phases without external training data or physics pre-knowledge. 

---
# WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling 

**Authors**: Michael Aich, Andreas Fürst, Florian Sestak, Carlos Ruiz-Gonzalez, Niklas Boers, Johannes Brandstetter  

**Link**: [PDF](https://arxiv.org/pdf/2602.03924)  

**Abstract**: Deep learning has revolutionized weather and climate modeling, yet the current landscape remains fragmented: highly specialized models are typically trained individually for distinct tasks. To unify this landscape, we introduce WIND, a single pre-trained foundation model capable of replacing specialized baselines across a vast array of tasks. Crucially, in contrast to previous atmospheric foundation models, we achieve this without any task-specific fine-tuning. To learn a robust, task-agnostic prior of the atmosphere, we pre-train WIND with a self-supervised video reconstruction objective, utilizing an unconditional video diffusion model to iteratively reconstruct atmospheric dynamics from a noisy state. At inference, we frame diverse domain-specific problems strictly as inverse problems and solve them via posterior sampling. This unified approach allows us to tackle highly relevant weather and climate problems, including probabilistic forecasting, spatial and temporal downscaling, sparse reconstruction and enforcing conservation laws purely with our pre-trained model. We further demonstrate the model's capacity to generate physically consistent counterfactual storylines of extreme weather events under global warming scenarios. By combining generative video modeling with inverse problem solving, WIND offers a computationally efficient paradigm shift in AI-based atmospheric modeling. 

---
# SpecMD: A Comprehensive Study On Speculative Expert Prefetching 

**Authors**: Duc Hoang, Ajay Jaiswal, Mohammad Samragh, Minsik Cho  

**Link**: [PDF](https://arxiv.org/pdf/2602.03921)  

**Abstract**: Mixture-of-Experts (MoE) models enable sparse expert activation, meaning that only a subset of the model's parameters is used during each inference. However, to translate this sparsity into practical performance, an expert caching mechanism is required. Previous works have proposed hardware-centric caching policies, but how these various caching policies interact with each other and different hardware specification remains poorly understood. To address this gap, we develop \textbf{SpecMD}, a standardized framework for benchmarking ad-hoc cache policies on various hardware configurations. Using SpecMD, we perform an exhaustive benchmarking of several MoE caching strategies, reproducing and extending prior approaches in controlled settings with realistic constraints. Our experiments reveal that MoE expert access is not consistent with temporal locality assumptions (e.g LRU, LFU). Motivated by this observation, we propose \textbf{Least-Stale}, a novel eviction policy that exploits MoE's predictable expert access patterns to reduce collision misses by up to $85\times$ over LRU. With such gains, we achieve over $88\%$ hit rates with up to $34.7\%$ Time-to-first-token (TTFT) reduction on OLMoE at only $5\%$ or $0.6GB$ of VRAM cache capacity. 

---
# Phaedra: Learning High-Fidelity Discrete Tokenization for the Physical Science 

**Authors**: Levi Lingsch, Georgios Kissas, Johannes Jakubik, Siddhartha Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2602.03915)  

**Abstract**: Tokens are discrete representations that allow modern deep learning to scale by transforming high-dimensional data into sequences that can be efficiently learned, generated, and generalized to new tasks. These have become foundational for image and video generation and, more recently, physical simulation. As existing tokenizers are designed for the explicit requirements of realistic visual perception of images, it is necessary to ask whether these approaches are optimal for scientific images, which exhibit a large dynamic range and require token embeddings to retain physical and spectral properties. In this work, we investigate the accuracy of a suite of image tokenizers across a range of metrics designed to measure the fidelity of PDE properties in both physical and spectral space. Based on the observation that these struggle to capture both fine details and precise magnitudes, we propose Phaedra, inspired by classical shape-gain quantization and proper orthogonal decomposition. We demonstrate that Phaedra consistently improves reconstruction across a range of PDE datasets. Additionally, our results show strong out-of-distribution generalization capabilities to three tasks of increasing complexity, namely known PDEs with different conditions, unknown PDEs, and real-world Earth observation and weather data. 

---
# Entropy-Aware Structural Alignment for Zero-Shot Handwritten Chinese Character Recognition 

**Authors**: Qiuming Luo, Tao Zeng, Feng Li, Heming Liu, Rui Mao, Chang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2602.03913)  

**Abstract**: Zero-shot Handwritten Chinese Character Recognition (HCCR) aims to recognize unseen characters by leveraging radical-based semantic compositions. However, existing approaches often treat characters as flat radical sequences, neglecting the hierarchical topology and the uneven information density of different components. To address these limitations, we propose an Entropy-Aware Structural Alignment Network that bridges the visual-semantic gap through information-theoretic modeling. First, we introduce an Information Entropy Prior to dynamically modulate positional embeddings via multiplicative interaction, acting as a saliency detector that prioritizes discriminative roots over ubiquitous components. Second, we construct a Dual-View Radical Tree to extract multi-granularity structural features, which are integrated via an adaptive Sigmoid-based gating network to encode both global layout and local spatial roles. Finally, a Top-K Semantic Feature Fusion mechanism is devised to augment the decoding process by utilizing the centroid of semantic neighbors, effectively rectifying visual ambiguities through feature-level consensus. Extensive experiments demonstrate that our method establishes new state-of-the-art performance, significantly outperforming existing CLIP-based baselines in the challenging zero-shot setting. Furthermore, the framework exhibits exceptional data efficiency, demonstrating rapid adaptability with minimal support samples. 

---
# HY3D-Bench: Generation of 3D Assets 

**Authors**: Team Hunyuan3D, Bowen Zhang, Chunchao Guo, Dongyuan Guo, Haolin Liu, Hongyu Yan, Huiwen Shi, Jiaao Yu, Jiachen Xu, Jingwei Huang, Kunhong Li, Lifu Wang, Linus, Penghao Wang, Qingxiang Lin, Ruining Tang, Xianghui Yang, Yang Li, Yirui Guan, Yunfei Zhao, Yunhan Yang, Zeqiang Lai, Zhihao Liang, Zibo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.03907)  

**Abstract**: While recent advances in neural representations and generative models have revolutionized 3D content creation, the field remains constrained by significant data processing bottlenecks. To address this, we introduce HY3D-Bench, an open-source ecosystem designed to establish a unified, high-quality foundation for 3D generation. Our contributions are threefold: (1) We curate a library of 250k high-fidelity 3D objects distilled from large-scale repositories, employing a rigorous pipeline to deliver training-ready artifacts, including watertight meshes and multi-view renderings; (2) We introduce structured part-level decomposition, providing the granularity essential for fine-grained perception and controllable editing; and (3) We bridge real-world distribution gaps via a scalable AIGC synthesis pipeline, contributing 125k synthetic assets to enhance diversity in long-tail categories. Validated empirically through the training of Hunyuan3D-2.1-Small, HY3D-Bench democratizes access to robust data resources, aiming to catalyze innovation across 3D perception, robotics, and digital content creation. 

---
# GeoIB: Geometry-Aware Information Bottleneck via Statistical-Manifold Compression 

**Authors**: Weiqi Wang, Zhiyi Tian, Chenhan Zhang, Shui Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03906)  

**Abstract**: Information Bottleneck (IB) is widely used, but in deep learning, it is usually implemented through tractable surrogates, such as variational bounds or neural mutual information (MI) estimators, rather than directly controlling the MI I(X;Z) itself. The looseness and estimator-dependent bias can make IB "compression" only indirectly controlled and optimization fragile.
We revisit the IB problem through the lens of information geometry and propose a \textbf{Geo}metric \textbf{I}nformation \textbf{B}ottleneck (\textbf{GeoIB}) that dispenses with mutual information (MI) estimation. We show that I(X;Z) and I(Z;Y) admit exact projection forms as minimal Kullback-Leibler (KL) distances from the joint distributions to their respective independence manifolds. Guided by this view, GeoIB controls information compression with two complementary terms: (i) a distribution-level Fisher-Rao (FR) discrepancy, which matches KL to second order and is reparameterization-invariant; and (ii) a geometry-level Jacobian-Frobenius (JF) term that provides a local capacity-type upper bound on I(Z;X) by penalizing pullback volume expansion of the encoder. We further derive a natural-gradient optimizer consistent with the FR metric and prove that the standard additive natural-gradient step is first-order equivalent to the geodesic update. We conducted extensive experiments and observed that the GeoIB achieves a better trade-off between prediction accuracy and compression ratio in the information plane than the mainstream IB baselines on popular datasets. GeoIB improves invariance and optimization stability by unifying distributional and geometric regularization under a single bottleneck multiplier. The source code of GeoIB is released at "this https URL. 

---
# All-Atom GPCR-Ligand Simulation via Residual Isometric Latent Flow 

**Authors**: Jiying Zhang, Shuhao Zhang, Pierre Vandergheynst, Patrick Barth  

**Link**: [PDF](https://arxiv.org/pdf/2602.03902)  

**Abstract**: G-protein-coupled receptors (GPCRs), primary targets for over one-third of approved therapeutics, rely on intricate conformational transitions to transduce signals. While Molecular Dynamics (MD) is essential for elucidating this transduction process, particularly within ligand-bound complexes, conventional all-atom MD simulation is computationally prohibitive. In this paper, we introduce GPCRLMD, a deep generative framework for efficient all-atom GPCR-ligand this http URL employs a Harmonic-Prior Variational Autoencoder (HP-VAE) to first map the complex into a regularized isometric latent space, preserving geometric topology via physics-informed constraints. Within this latent space, a Residual Latent Flow samples evolution trajectories, which are subsequently decoded back to atomic coordinates. By capturing temporal dynamics via relative displacements anchored to the initial structure, this residual mechanism effectively decouples static topology from dynamic fluctuations. Experimental results demonstrate that GPCRLMD achieves state-of-the-art performance in GPCR-ligand dynamics simulation, faithfully reproducing thermodynamic observables and critical ligand-receptor interactions. 

---
# Byzantine Machine Learning: MultiKrum and an optimal notion of robustness 

**Authors**: Gilles Bareilles, Wassim Bouaziz, Julien Fageot, El-Mahdi El-Mhamdi  

**Link**: [PDF](https://arxiv.org/pdf/2602.03899)  

**Abstract**: Aggregation rules are the cornerstone of distributed (or federated) learning in the presence of adversaries, under the so-called Byzantine threat model. They are also interesting mathematical objects from the point of view of robust mean estimation. The Krum aggregation rule has been extensively studied, and endowed with formal robustness and convergence guarantees. Yet, MultiKrum, a natural extension of Krum, is often preferred in practice for its superior empirical performance, even though no theoretical guarantees were available until now. In this work, we provide the first proof that MultiKrum is a robust aggregation rule, and bound its robustness coefficient. To do so, we introduce $\kappa^\star$, the optimal *robustness coefficient* of an aggregation rule, which quantifies the accuracy of mean estimation in the presence of adversaries in a tighter manner compared with previously adopted notions of robustness. We then construct an upper and a lower bound on MultiKrum's robustness coefficient. As a by-product, we also improve on the best-known bounds on Krum's robustness coefficient. We show that MultiKrum's bounds are never worse than Krum's, and better in realistic regimes. We illustrate this analysis by an experimental investigation on the quality of the lower bound. 

---
# Vision Transformers for Zero-Shot Clustering of Animal Images: A Comparative Benchmarking Study 

**Authors**: Hugo Markoff, Stefan Hein Bengtson, Michael Ørsted  

**Link**: [PDF](https://arxiv.org/pdf/2602.03894)  

**Abstract**: Manual labeling of animal images remains a significant bottleneck in ecological research, limiting the scale and efficiency of biodiversity monitoring efforts. This study investigates whether state-of-the-art Vision Transformer (ViT) foundation models can reduce thousands of unlabeled animal images directly to species-level clusters. We present a comprehensive benchmarking framework evaluating five ViT models combined with five dimensionality reduction techniques and four clustering algorithms, two supervised and two unsupervised, across 60 species (30 mammals and 30 birds), with each test using a random subset of 200 validated images per species. We investigate when clustering succeeds at species-level, where it fails, and whether clustering within the species-level reveals ecologically meaningful patterns such as sex, age, or phenotypic variation. Our results demonstrate near-perfect species-level clustering (V-measure: 0.958) using DINOv3 embeddings with t-SNE and supervised hierarchical clustering methods. Unsupervised approaches achieve competitive performance (0.943) while requiring no prior species knowledge, rejecting only 1.14% of images as outliers requiring expert review. We further demonstrate robustness to realistic long-tailed distributions of species and show that intentional over-clustering can reliably extract intra-specific variation including age classes, sexual dimorphism, and pelage differences. We introduce an open-source benchmarking toolkit and provide recommendations for ecologists to select appropriate methods for sorting their specific taxonomic groups and data. 

---
# Audit After Segmentation: Reference-Free Mask Quality Assessment for Language-Referred Audio-Visual Segmentation 

**Authors**: Jinxing Zhou, Yanghao Zhou, Yaoting Wang, Zongyan Han, Jiaqi Ma, Henghui Ding, Rao Muhammad Anwer, Hisham Cholakkal  

**Link**: [PDF](https://arxiv.org/pdf/2602.03892)  

**Abstract**: Language-referred audio-visual segmentation (Ref-AVS) aims to segment target objects described by natural language by jointly reasoning over video, audio, and text. Beyond generating segmentation masks, providing rich and interpretable diagnoses of mask quality remains largely underexplored. In this work, we introduce Mask Quality Assessment in the Ref-AVS context (MQA-RefAVS), a new task that evaluates the quality of candidate segmentation masks without relying on ground-truth annotations as references at inference time. Given audio-visual-language inputs and each provided segmentation mask, the task requires estimating its IoU with the unobserved ground truth, identifying the corresponding error type, and recommending an actionable quality-control decision. To support this task, we construct MQ-RAVSBench, a benchmark featuring diverse and representative mask error modes that span both geometric and semantic issues. We further propose MQ-Auditor, a multimodal large language model (MLLM)-based auditor that explicitly reasons over multimodal cues and mask information to produce quantitative and qualitative mask quality assessments. Extensive experiments demonstrate that MQ-Auditor outperforms strong open-source and commercial MLLMs and can be integrated with existing Ref-AVS systems to detect segmentation failures and support downstream segmentation improvement. Data and codes will be released at this https URL. 

---
# Sounding Highlights: Dual-Pathway Audio Encoders for Audio-Visual Video Highlight Detection 

**Authors**: Seohyun Joo, Yoori Oh  

**Link**: [PDF](https://arxiv.org/pdf/2602.03891)  

**Abstract**: Audio-visual video highlight detection aims to automatically identify the most salient moments in videos by leveraging both visual and auditory cues. However, existing models often underutilize the audio modality, focusing on high-level semantic features while failing to fully leverage the rich, dynamic characteristics of sound. To address this limitation, we propose a novel framework, Dual-Pathway Audio Encoders for Video Highlight Detection (DAViHD). The dual-pathway audio encoder is composed of a semantic pathway for content understanding and a dynamic pathway that captures spectro-temporal dynamics. The semantic pathway extracts high-level information by identifying the content within the audio, such as speech, music, or specific sound events. The dynamic pathway employs a frequency-adaptive mechanism as time evolves to jointly model these dynamics, enabling it to identify transient acoustic events via salient spectral bands and rapid energy changes. We integrate the novel audio encoder into a full audio-visual framework and achieve new state-of-the-art performance on the large-scale this http URL benchmark. Our results demonstrate that a sophisticated, dual-faceted audio representation is key to advancing the field of highlight detection. 

---
# Explainable Computer Vision Framework for Automated Pore Detection and Criticality Assessment in Additive Manufacturing 

**Authors**: Akshansh Mishra, Rakesh Morisetty  

**Link**: [PDF](https://arxiv.org/pdf/2602.03883)  

**Abstract**: Internal porosity remains a critical defect mode in additively manufactured components, compromising structural performance and limiting industrial adoption. Automated defect detection methods exist but lack interpretability, preventing engineers from understanding the physical basis of criticality predictions. This study presents an explainable computer vision framework for pore detection and criticality assessment in three-dimensional tomographic volumes. Sequential grayscale slices were reconstructed into volumetric datasets, and intensity-based thresholding with connected component analysis identified 500 individual pores. Each pore was characterized using geometric descriptors including size, aspect ratio, extent, and spatial position relative to the specimen boundary. A pore interaction network was constructed using percentile-based Euclidean distance criteria, yielding 24,950 inter-pore connections. Machine learning models predicted pore criticality scores from extracted features, and SHAP analysis quantified individual feature contributions. Results demonstrate that normalized surface distance dominates model predictions, contributing more than an order of magnitude greater importance than all other descriptors. Pore size provides minimal influence, while geometric parameters show negligible impact. The strong inverse relationship between surface proximity and criticality reveals boundary-driven failure mechanisms. This interpretable framework enables transparent defect assessment and provides actionable insights for process optimization and quality control in additive manufacturing. 

---
# PriorProbe: Recovering Individual-Level Priors for Personalizing Neural Networks in Facial Expression Recognition 

**Authors**: Haijiang Yan, Nick Chater, Adam Sanborn  

**Link**: [PDF](https://arxiv.org/pdf/2602.03882)  

**Abstract**: Incorporating individual-level cognitive priors offers an important route to personalizing neural networks, yet accurately eliciting such priors remains challenging: existing methods either fail to uniquely identify them or introduce systematic biases. Here, we introduce PriorProbe, a novel elicitation approach grounded in Markov Chain Monte Carlo with People that recovers fine-grained, individual-specific priors. Focusing on a facial expression recognition task, we apply PriorProbe to individual participants and test whether integrating the recovered priors with a state-of-the-art neural network improves its ability to predict an individual's classification on ambiguous stimuli. The PriorProbe-derived priors yield substantial performance gains, outperforming both the neural network alone and alternative sources of priors, while preserving the network's inference on ground-truth labels. Together, these results demonstrate that PriorProbe provides a general and interpretable framework for personalizing deep neural networks. 

---
# DiGAN: Diffusion-Guided Attention Network for Early Alzheimer's Disease Detection 

**Authors**: Maxx Richard Rahman, Mostafa Hammouda, Wolfgang Maass  

**Link**: [PDF](https://arxiv.org/pdf/2602.03881)  

**Abstract**: Early diagnosis of Alzheimer's disease (AD) remains a major challenge due to the subtle and temporally irregular progression of structural brain changes in the prodromal stages. Existing deep learning approaches require large longitudinal datasets and often fail to model the temporal continuity and modality irregularities inherent in real-world clinical data. To address these limitations, we propose the Diffusion-Guided Attention Network (DiGAN), which integrates latent diffusion modelling with an attention-guided convolutional network. The diffusion model synthesizes realistic longitudinal neuroimaging trajectories from limited training data, enriching temporal context and improving robustness to unevenly spaced visits. The attention-convolutional layer then captures discriminative structural--temporal patterns that distinguish cognitively normal subjects from those with mild cognitive impairment and subjective cognitive decline. Experiments on synthetic and ADNI datasets demonstrate that DiGAN outperforms existing state-of-the-art baselines, showing its potential for early-stage AD detection. 

---
# TruKAN: Towards More Efficient Kolmogorov-Arnold Networks Using Truncated Power Functions 

**Authors**: Ali Bayeh, Samira Sadaoui, Malek Mouhoub  

**Link**: [PDF](https://arxiv.org/pdf/2602.03879)  

**Abstract**: To address the trade-off between computational efficiency and adherence to Kolmogorov-Arnold Network (KAN) principles, we propose TruKAN, a new architecture based on the KAN structure and learnable activation functions. TruKAN replaces the B-spline basis in KAN with a family of truncated power functions derived from k-order spline theory. This change maintains the KAN's expressiveness while enhancing accuracy and training time. Each TruKAN layer combines a truncated power term with a polynomial term and employs either shared or individual knots. TruKAN exhibits greater interpretability than other KAN variants due to its simplified basis functions and knot configurations. By prioritizing interpretable basis functions, TruKAN aims to balance approximation efficacy with transparency. We develop the TruKAN model and integrate it into an advanced EfficientNet-V2-based framework, which is then evaluated on computer vision benchmark datasets. To ensure a fair comparison, we develop various models: MLP-, KAN-, SineKAN and TruKAN-based EfficientNet frameworks and assess their training time and accuracy across small and deep architectures. The training phase uses hybrid optimization to improve convergence stability. Additionally, we investigate layer normalization techniques for all the models and assess the impact of shared versus individual knots in TruKAN. Overall, TruKAN outperforms other KAN models in terms of accuracy, computational efficiency and memory usage on the complex vision task, demonstrating advantages beyond the limited settings explored in prior KAN studies. 

---
# GOPO: Policy Optimization using Ranked Rewards 

**Authors**: Kyuseong Choi, Dwaipayan Saha, Woojeong Kim, Anish Agarwal, Raaz Dwivedi  

**Link**: [PDF](https://arxiv.org/pdf/2602.03876)  

**Abstract**: Standard reinforcement learning from human feedback (RLHF) trains a reward model on pairwise preference data and then uses it for policy optimization. However, while reward models are optimized to capture relative preferences, existing policy optimization techniques rely on absolute reward magnitudes during training. In settings where the rewards are non-verifiable such as summarization, instruction following, and chat completion, this misalignment often leads to suboptimal performance. We introduce Group Ordinal Policy Optimization (GOPO), a policy optimization method that uses only the ranking of the rewards and discards their magnitudes. Our rank-based transformation of rewards provides several gains, compared to Group Relative Policy Optimization (GRPO), in settings with non-verifiable rewards: (1) consistently higher training/validation reward trajectories, (2) improved LLM-as-judge evaluations across most intermediate training steps, and (3) reaching a policy of comparable quality in substantially less training steps than GRPO. We demonstrate consistent improvements across a range of tasks and model sizes. 

---
# Reversible Deep Learning for 13C NMR in Chemoinformatics: On Structures and Spectra 

**Authors**: Stefan Kuhn, Vandana Dwarka, Przemyslaw Karol Grenda, Eero Vainikko  

**Link**: [PDF](https://arxiv.org/pdf/2602.03875)  

**Abstract**: We introduce a reversible deep learning model for 13C NMR that uses a single conditional invertible neural network for both directions between molecular structures and spectra. The network is built from i-RevNet style bijective blocks, so the forward map and its inverse are available by construction. We train the model to predict a 128-bit binned spectrum code from a graph-based structure encoding, while the remaining latent dimensions capture residual variability. At inference time, we invert the same trained network to generate structure candidates from a spectrum code, which explicitly represents the one-to-many nature of spectrum-to-structure inference. On a filtered subset, the model is numerically invertible on trained examples, achieves spectrum-code prediction above chance, and produces coarse but meaningful structural signals when inverted on validation spectra. These results demonstrate that invertible architectures can unify spectrum prediction and uncertainty-aware candidate generation within one end-to-end model. 

---
# Decoding Ambiguous Emotions with Test-Time Scaling in Audio-Language Models 

**Authors**: Hong Jia, Weibin Li, Jingyao Wu, Xiaofeng Yu, Yan Gao, Jintao Cheng, Xiaoyu Tang, Feng Xia, Ting Dang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03873)  

**Abstract**: Emotion recognition from human speech is a critical enabler for socially aware conversational AI. However, while most prior work frames emotion recognition as a categorical classification problem, real-world affective states are often ambiguous, overlapping, and context-dependent, posing significant challenges for both annotation and automatic modeling. Recent large-scale audio language models (ALMs) offer new opportunities for nuanced affective reasoning without explicit emotion supervision, but their capacity to handle ambiguous emotions remains underexplored. At the same time, advances in inference-time techniques such as test-time scaling (TTS) have shown promise for improving generalization and adaptability in hard NLP tasks, but their relevance to affective computing is still largely unknown. In this work, we introduce the first benchmark for ambiguous emotion recognition in speech with ALMs under test-time scaling. Our evaluation systematically compares eight state-of-the-art ALMs and five TTS strategies across three prominent speech emotion datasets. We further provide an in-depth analysis of the interaction between model capacity, TTS, and affective ambiguity, offering new insights into the computational and representational challenges of ambiguous emotion understanding. Our benchmark establishes a foundation for developing more robust, context-aware, and emotionally intelligent speech-based AI systems, and highlights key future directions for bridging the gap between model assumptions and the complexity of real-world human emotion. 

---
# Understanding the Impact of Differentially Private Training on Memorization of Long-Tailed Data 

**Authors**: Jiaming Zhang, Huanyi Xie, Meng Ding, Shaopeng Fu, Jinyan Liu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03872)  

**Abstract**: Recent research shows that modern deep learning models achieve high predictive accuracy partly by memorizing individual training samples. Such memorization raises serious privacy concerns, motivating the widespread adoption of differentially private training algorithms such as DP-SGD. However, a growing body of empirical work shows that DP-SGD often leads to suboptimal generalization performance, particularly on long-tailed data that contain a large number of rare or atypical samples. Despite these observations, a theoretical understanding of this phenomenon remains largely unexplored, and existing differential privacy analysis are difficult to extend to the nonconvex and nonsmooth neural networks commonly used in practice. In this work, we develop the first theoretical framework for analyzing DP-SGD on long-tailed data from a feature learning perspective. We show that the test error of DP-SGD-trained models on the long-tailed subpopulation is significantly larger than the overall test error over the entire dataset. Our analysis further characterizes the training dynamics of DP-SGD, demonstrating how gradient clipping and noise injection jointly adversely affect the model's ability to memorize informative but underrepresented samples. Finally, we validate our theoretical findings through extensive experiments on both synthetic and real-world datasets. 

---
# Benchmarking Automatic Speech Recognition for Indian Languages in Agricultural Contexts 

**Authors**: Chandrashekar M S, Vineet Singh, Lakshmi Pedapudi  

**Link**: [PDF](https://arxiv.org/pdf/2602.03868)  

**Abstract**: The digitization of agricultural advisory services in India requires robust Automatic Speech Recognition (ASR) systems capable of accurately transcribing domain-specific terminology in multiple Indian languages. This paper presents a benchmarking framework for evaluating ASR performance in agricultural contexts across Hindi, Telugu, and Odia languages. We introduce evaluation metrics including Agriculture Weighted Word Error Rate (AWWER) and domain-specific utility scoring to complement traditional metrics. Our evaluation of 10,934 audio recordings, each transcribed by up to 10 ASR models, reveals performance variations across languages and models, with Hindi achieving the best overall performance (WER: 16.2%) while Odia presents the greatest challenges (best WER: 35.1%, achieved only with speaker diarization). We characterize audio quality challenges inherent to real-world agricultural field recordings and demonstrate that speaker diarization with best-speaker selection can substantially reduce WER for multi-speaker recordings (upto 66% depending on the proportion of multi-speaker audio). We identify recurring error patterns in agricultural terminology and provide practical recommendations for improving ASR systems in low-resource agricultural domains. The study establishes baseline benchmarks for future agricultural ASR development. 

---
# PaperX: A Unified Framework for Multimodal Academic Presentation Generation with Scholar DAG 

**Authors**: Tao Yu, Minghui Zhang, Zhiqing Cui, Hao Wang, Zhongtian Luo, Shenghua Chai, Junhao Gong, Yuzhao Peng, Yuxuan Zhou, Yujia Yang, Zhenghao Zhang, Haopeng Jin, Xinming Wang, Yufei Xiong, Jiabing Yang, Jiahao Yuan, Hanqing Wang, Hongzhu Yi, YiFan Zhang, Yan Huang, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03866)  

**Abstract**: Transforming scientific papers into multimodal presentation content is essential for research dissemination but remains labor intensive. Existing automated solutions typically treat each format as an isolated downstream task, leading to redundant processing and semantic inconsistency. We introduce PaperX, a unified framework that models academic presentation generation as a structural transformation and rendering process. Central to our approach is the Scholar DAG, an intermediate representation that decouples the paper's logical structure from its final presentation syntax. By applying adaptive graph traversal strategies, PaperX generates diverse, high quality outputs from a single source. Comprehensive evaluations demonstrate that our framework achieves the state of the art performance in content fidelity and aesthetic quality while significantly improving cost efficiency compared to specialized single task agents. 

---
# Perceptions of AI-CBT: Trust and Barriers in Chinese Postgrads 

**Authors**: Chan-in Sio, Alex Mann, Lingxi Fan, Andrew Cheung, Lik-hang Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.03852)  

**Abstract**: The mental well-being of graduate students is an increasing concern, yet the adoption of scalable support remains uneven. Artificial intelligence-powered cognitive behavioral therapy chatbots (AI-CBT) offer low barrier help, but little is known about how Chinese postgraduates perceive and use them. This qualitative study explored perceptions and experiences of AI-CBT chatbots among ten Chinese graduate students recruited through social media. Semi-structured Zoom interviews were conducted and analyzed using reflexive thematic analysis, with the Health Belief Model (HBM) and the Theory of Planned Behavior (TPB) as sensitizing frameworks. The findings indicate a cautious openness to AI-CBT chatbots: perceived usefulness and 24/7 access supported favorable attitudes, while data privacy, emotional safety, and uncertainty about `fit' for complex problems restricted the intention to use. Social norms (e.g., stigma and peer views) and perceived control (digital literacy, language quality) further shaped adoption. The study offers context-specific information to guide the culturally sensitive design, communication, and deployment of AI mental well-being tools for student populations in China and outlines the design implications around transparency, safeguards, and graduated care pathways. 

---
# WebAccessVL: Making an Accessible Web via Violation-Conditioned VLM 

**Authors**: Amber Yijia Zheng, Jae Joong Lee, Bedrich Benes, Raymond A. Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2602.03850)  

**Abstract**: We present a vision-language model (VLM) that automatically edits website HTML to address Web Content Accessibility Guidelines 2 (WCAG2) violations. We formulate this as a supervised image-conditioned program synthesis task, where the model learns to correct HTML given the HTML and its rendering. We collected WebAccessVL, a new dataset with manually corrected accessibility violations, establishing paired training data. We then propose a violation-conditioned VLM that additionally conditions on the WCAG2 violation count to guide the correction process. Experiments demonstrate that our method effectively reduces the average number of violations from 5.34 to 0.44 per website, outperforming commercial LLM APIs (Gemini, GPT-5). A perceptual study confirms that our edited websites maintain the original visual appearance and content. 

---
# HybridQuestion: Human-AI Collaboration for Identifying High-Impact Research Questions 

**Authors**: Keyu Zhao, Fengli Xu, Yong Li, Tie-Yan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03849)  

**Abstract**: The "AI Scientist" paradigm is transforming scientific research by automating key stages of the research process, from idea generation to scholarly writing. This shift is expected to accelerate discovery and expand the scope of scientific inquiry. However, a key question remains unclear: can AI scientists identify meaningful research questions? While Large Language Models (LLMs) have been applied successfully to task-specific ideation, their potential to conduct strategic, long-term assessments of past breakthroughs and future questions remains largely unexplored. To address this gap, we explore a human-AI hybrid solution that integrates the scalable data processing capabilities of AI with the value judgment of human experts. Our methodology is structured in three phases. The first phase, AI-Accelerated Information Gathering, leverages AI's advantage in processing vast amounts of literature to generate a hybrid information base. The second phase, Candidate Question Proposing, utilizes this synthesized data to prompt an ensemble of six diverse LLMs to propose an initial candidate pool, filtered via a cross-model voting mechanism. The third phase, Hybrid Question Selection, refines this pool through a multi-stage filtering process that progressively increases human oversight. To validate this system, we conducted an experiment aiming to identify the Top 10 Scientific Breakthroughs of 2025 and the Top 10 Scientific Questions for 2026 across five major disciplines. Our analysis reveals that while AI agents demonstrate high alignment with human experts in recognizing established breakthroughs, they exhibit greater divergence in forecasting prospective questions, suggesting that human judgment remains crucial for evaluating subjective, forward-looking challenges. 

---
# Merged ChemProt-DrugProt for Relation Extraction from Biomedical Literature 

**Authors**: Mai H. Nguyen, Shibani Likhite, Jiawei Tang, Darshini Mahendran, Bridget T. McInnes  

**Link**: [PDF](https://arxiv.org/pdf/2405.18605)  

**Abstract**: The extraction of chemical-gene relations plays a pivotal role in understanding the intricate interactions between chemical compounds and genes, with significant implications for drug discovery, disease understanding, and biomedical research. This paper presents a data set created by merging the ChemProt and DrugProt datasets to augment sample counts and improve model accuracy. We evaluate the merged dataset using two state of the art relationship extraction algorithms: Bidirectional Encoder Representations from Transformers (BERT) specifically BioBERT, and Graph Convolutional Networks (GCNs) combined with BioBERT. While BioBERT excels at capturing local contexts, it may benefit from incorporating global information essential for understanding chemical-gene interactions. This can be achieved by integrating GCNs with BioBERT to harness both global and local context. Our results show that by integrating the ChemProt and DrugProt datasets, we demonstrated significant improvements in model performance, particularly in CPR groups shared between the datasets. Incorporating the global context using GCN can help increase the overall precision and recall in some of the CPR groups over using just BioBERT. 

---
