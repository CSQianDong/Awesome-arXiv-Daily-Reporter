# Detecting Safety Violations Across Many Agent Traces 

**Authors**: Adam Stein, Davis Brown, Hamed Hassani, Mayur Naik, Eric Wong  

**Link**: [PDF](https://arxiv.org/pdf/2604.11806)  

**Abstract**: To identify safety violations, auditors often search over large sets of agent traces. This search is difficult because failures are often rare, complex, and sometimes even adversarially hidden and only detectable when multiple traces are analyzed together. These challenges arise in diverse settings such as misuse campaigns, covert sabotage, reward hacking, and prompt injection. Existing approaches struggle here for several reasons. Per-trace judges miss failures that only become visible across traces, naive agentic auditing does not scale to large trace collections, and fixed monitors are brittle to unanticipated behaviors. We introduce Meerkat, which combines clustering with agentic search to uncover violations specified in natural language. Through structured search and adaptive investigation of promising regions, Meerkat finds sparse failures without relying on seed scenarios, fixed workflows, or exhaustive enumeration. Across misuse, misalignment, and task gaming settings, Meerkat significantly improves detection of safety violations over baseline monitors, discovers widespread developer cheating on a top agent benchmark, and finds nearly 4x more examples of reward hacking on CyBench than previous audits. 

---
# GenTac: Generative Modeling and Forecasting of Soccer Tactics 

**Authors**: Jiayuan Rao, Tianlin Gui, Haoning Wu, Yanfeng Wang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.11786)  

**Abstract**: Modeling open-play soccer tactics is a formidable challenge due to the stochastic, multi-agent nature of the game. Existing computational approaches typically produce single, deterministic trajectory forecasts or focus on highly structured set-pieces, fundamentally failing to capture the inherent variance and branching possibilities of real-world match evolution. Here, we introduce GenTac, a diffusion-based generative framework that conceptualizes soccer tactics as a stochastic process over continuous multi-player trajectories and discrete semantic events. By learning the underlying distribution of player movements from historical tracking data, GenTac samples diverse, plausible, long-horizon future trajectories. The framework supports rich contextual conditioning, including opponent behavior, specific team or league playing styles, and strategic objectives, while grounding continuous spatial dynamics into a 15-class tactical event space. Extensive evaluations on our proposed benchmark, TacBench, demonstrate four key capabilities: (1) GenTac achieves high geometric accuracy while strictly preserving the collective structural consistency of the team; (2) it accurately simulates stylistic nuances, distinguishing between specific teams (e.g., Auckland FC) and leagues (e.g., A-League versus German leagues); (3) it enables controllable counterfactual simulations, demonstrably altering spatial control and expected threat metrics based on offensive or defensive guidance; and (4) it reliably anticipates future tactical outcomes directly from generated rollouts. Finally, we demonstrate that GenTac can be successfully trained to generalize to other dynamic team sports, including basketball, American football, and ice hockey. 

---
# Retrieval Is Not Enough: Why Organizational AI Needs Epistemic Infrastructure 

**Authors**: Federico Bottino, Carlo Ferrero, Nicholas Dosio, Pierfrancesco Beneventano  

**Link**: [PDF](https://arxiv.org/pdf/2604.11759)  

**Abstract**: Organizational knowledge used by AI agents typically lacks epistemic structure: retrieval systems surface semantically relevant content without distinguishing binding decisions from abandoned hypotheses, contested claims from settled ones, or known facts from unresolved questions. We argue that the ceiling on organizational AI is not retrieval fidelity but \emph{epistemic} fidelity--the system's ability to represent commitment strength, contradiction status, and organizational ignorance as computable properties.
We present OIDA, a framework that structures organizational knowledge as typed Knowledge Objects carrying epistemic class, importance scores with class-specific decay, and signed contradiction edges. The Knowledge Gravity Engine maintains scores deterministically with proved convergence guarantees (sufficient condition: max degree $< 7$; empirically robust to degree 43). OIDA introduces QUESTION-as-modeled-ignorance: a primitive with inverse decay that surfaces what an organization does \emph{not} know with increasing urgency--a mechanism absent from all surveyed systems. We describe the Epistemic Quality Score (EQS), a five-component evaluation methodology with explicit circularity analysis. In a controlled comparison ($n{=}10$ response pairs), OIDA's RAG condition (3,868 tokens) achieves EQS 0.530 vs.\ 0.848 for a full-context baseline (108,687 tokens); the $28.1\times$ token budget difference is the primary confound. The QUESTION mechanism is statistically validated (Fisher $p{=}0.0325$, OR$=21.0$). The formal properties are established; the decisive ablation at equal token budget (E4) is pre-registered and not yet run. 

---
# Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games 

**Authors**: Keyang Zhong, Junlin Xie, Hefeng Wu, Haofeng Li, Guanbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11741)  

**Abstract**: Vision-language models (VLMs) have shown impressive capabilities in perceptual tasks, yet they degrade in complex multi-hop reasoning under multiplayer game settings with imperfect and deceptive information. In this paper, we study a representative multiplayer task, Murder Mystery Games, which require inferring hidden truths based on partial clues provided by roles with different intentions. To address this challenge, we propose a collaborative multi-agent framework for evaluating and synthesizing high-quality, role-driven multiplayer game scripts, enabling fine-grained interaction patterns tailored to character identities (i.e., murderer vs. innocent). Our system generates rich multimodal contexts, including character backstories, visual and textual clues, and multi-hop reasoning chains, through coordinated agent interactions. We design a two-stage agent-monitored training strategy to enhance the reasoning ability of VLMs: (1) chain-of-thought based fine-tuning on curated and synthetic datasets that model uncertainty and deception; (2) GRPO-based reinforcement learning with agent-monitored reward shaping, encouraging the model to develop character-specific reasoning behaviors and effective multimodal multi-hop inference. Extensive experiments demonstrate that our method significantly boosts the performance of VLMs in narrative reasoning, hidden fact extraction, and deception-resilient understanding. Our contributions offer a scalable solution for training and evaluating VLMs under uncertain, adversarial, and socially complex conditions, laying the groundwork for future benchmarks in multimodal multi-hop reasoning under imperfect information. 

---
# SWE-AGILE: A Software Agent Framework for Efficiently Managing Dynamic Reasoning Context 

**Authors**: Shuquan Lian, Juncheng Liu, Yazhe Chen, Yuhong Chen, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11716)  

**Abstract**: Prior representative ReAct-style approaches in autonomous Software Engineering (SWE) typically lack the explicit System-2 reasoning required for deep analysis and handling complex edge cases. While recent reasoning models demonstrate the potential of extended Chain-of-Thought (CoT), applying them to the multi-turn SWE task creates a fundamental dilemma: retaining full reasoning history leads to context explosion and ``Lost-in-the-Middle'' degradation, while discarding it would force the agent to redundantly re-reason at every step. To address these challenges, we propose SWE-AGILE, a novel software agent framework designed to bridge the gap between reasoning depth, efficiency, and context constraints. SWE-AGILE introduces a Dynamic Reasoning Context strategy, maintaining a ``sliding window'' of detailed reasoning for immediate continuity to prevent redundant re-analyzing, while compressing historical reasoning content into concise Reasoning Digests. Empirically, SWE-AGILE sets a new standard for 7B-8B models on SWE-Bench-Verified using only 2.2k trajectories and 896 tasks. Code is available at this https URL. 

---
# A Mamba-Based Multimodal Network for Multiscale Blast-Induced Rapid Structural Damage Assessment 

**Authors**: Wanli Ma, Sivasakthy Selvakumaran, Dain G. Farrimond, Adam A. Dennis, Samuel E. Rigby  

**Link**: [PDF](https://arxiv.org/pdf/2604.11709)  

**Abstract**: Accurate and rapid structural damage assessment (SDA) is crucial for post-disaster management, helping responders prioritise resources, plan rescues, and support recovery. Traditional field inspections, though precise, are limited by accessibility, safety risks, and time constraints, especially after large explosions. Machine learning with remote sensing has emerged as a scalable solution for rapid SDA, with Mamba-based networks achieving state-of-the-art performance. However, these methods often require extensive training and large datasets, limiting real-world applicability. Moreover, they fail to incorporate key physical characteristics of blast loading for SDA. To overcome these challenges, we propose a Mamba-based multimodal network for rapid SDA that integrates multi-scale blast-loading information with optical remote sensing images. Evaluated on the 2020 Beirut explosion, our method significantly improves performance over state-of-the-art approaches. Code is available at: this https URL 

---
# Agentic Driving Coach: Robustness and Determinism of Agentic AI-Powered Human-in-the-Loop Cyber-Physical Systems 

**Authors**: Deeksha Prahlad, Daniel Fan, Hokeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.11705)  

**Abstract**: Foundation models, including large language models (LLMs), are increasingly used for human-in-the-loop (HITL) cyber-physical systems (CPS) because foundation model-based AI agents can potentially interact with both the physical environments and human users. However, the unpredictable behavior of human users and AI agents, in addition to the dynamically changing physical environments, leads to uncontrollable nondeterminism. To address this urgent challenge of enabling agentic AI-powered HITL CPS, we propose a reactor-model-of-computation (MoC)-based approach, realized by the open-source Lingua Franca (LF) framework. We also carry out a concrete case study using the agentic driving coach as an application of HITL CPS. By evaluating the LF-based agentic HITL CPS, we identify practical challenges in reintroducing determinism into such agentic HITL CPS and present pathways to address them. 

---
# DreamKG: A KG-Augmented Conversational System for People Experiencing Homelessness 

**Authors**: Javad M Alizadeh, Genhui Zheng, Chiu C Tan, Yuzhou Chen, Omar Martinez, Philip McCallion, Ying Ding, Chenguang Yang, AnneMarie Tomosky, Huanmei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11703)  

**Abstract**: People experiencing homelessness (PEH) face substantial barriers to accessing timely, accurate information about community services. DreamKG addresses this through a knowledge graph-augmented conversational system that grounds responses in verified, up-to-date data about Philadelphia organizations, services, locations, and hours. Unlike standard large language models (LLMs) prone to hallucinations, DreamKG combines Neo4j knowledge graphs with structured query understanding to handle location-aware and time-sensitive queries reliably. The system performs spatial reasoning for distance-based recommendations and temporal filtering for operating hours. Preliminary evaluation shows 59% superiority over Google Search AI on relevant queries and 84% rejection of irrelevant queries. This demonstration highlights the potential of hybrid architectures that combines LLM flexibility with knowledge graph reliability to improve service accessibility for vulnerable populations effectively. 

---
# Why Do Large Language Models Generate Harmful Content? 

**Authors**: Rajesh Ganguli, Raha Moraffah  

**Link**: [PDF](https://arxiv.org/pdf/2604.11663)  

**Abstract**: Large Language Models (LLMs) have been shown to generate harmful content. However, the underlying causes of such behavior remain under explored. We propose a causal mediation analysis-based approach to identify the causal factors responsible for harmful generation. Our method performs a multi-granular analysis across model layers, modules (MLP and attention blocks), and individual neurons. Extensive experiments on state-of-the-art LLMs indicate that harmful generation arises in the later layers of the model, results primarily from failures in MLP blocks rather than attention blocks, and is associated with neurons that act as a gating mechanism for harmful generation. The results indicate that the early layers in the model are used for a contextual understanding of harmfulness in a prompt, which is then propagated through the model, to generate harmfulness in the late layers, as well as a signal indicating harmfulness through MLP blocks. This is then further propagated to the last layer of the model, specifically to a sparse set of neurons, which receives the signal and determines the generation of harmful content accordingly. 

---
# RationalRewards: Reasoning Rewards Scale Visual Generation Both Training and Test Time 

**Authors**: Haozhe Wang, Cong Wei, Weiming Ren, Jiaming Liu, Fangzhen Lin, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11626)  

**Abstract**: Most reward models for visual generation reduce rich human judgments to a single unexplained score, discarding the reasoning that underlies preference. We show that teaching reward models to produce explicit, multi-dimensional critiques before scoring transforms them from passive evaluators into active optimization tools, improving generators in two complementary ways: at training time, structured rationales provide interpretable, fine-grained rewards for reinforcement learning; at test time, a Generate-Critique-Refine loop turns critiques into targeted prompt revisions that improve outputs without any parameter updates. To train such a reward model without costly rationale annotations, we introduce Preference-Anchored Rationalization (PARROT), a principled framework that recovers high-quality rationales from readily available preference data through anchored generation, consistency filtering, and distillation. The resulting model, RationalRewards (8B), achieves state-of-the-art preference prediction among open-source reward models, competitive with Gemini-2.5-Pro, while using 10-20x less training data than comparable baselines. As an RL reward, it consistently improves text-to-image and image-editing generators beyond scalar alternatives. Most strikingly, its test-time critique-and-refine loop matches or exceeds RL-based fine-tuning on several benchmarks, suggesting that structured reasoning can unlock latent capabilities in existing generators that suboptimal prompts fail to elicit. 

---
# Context Kubernetes: Declarative Orchestration of Enterprise Knowledge for Agentic AI Systems 

**Authors**: Charafeddine Mouzouni  

**Link**: [PDF](https://arxiv.org/pdf/2604.11623)  

**Abstract**: We introduce Context Kubernetes, an architecture for orchestrating enterprise knowledge in agentic AI systems, with a prototype implementation and eight experiments. The core observation is that delivering the right knowledge, to the right agent, with the right permissions, at the right freshness -- across an entire organization -- is structurally analogous to the container orchestration problem Kubernetes solved a decade ago. We formalize six core abstractions, a YAML-based declarative manifest for knowledge-architecture-as-code, a reconciliation loop, and a three-tier agent permission model where agent authority is always a strict subset of human authority. Three value experiments show: (1) without governance, agents serve phantom content from deleted sources and leak cross-domain data in 26.5% of queries; (2) without freshness monitoring, stale content is served silently -- with reconciliation, staleness is detected in under 1ms; (3) in five attack scenarios, flat permissions block 0/5 attacks, basic RBAC blocks 4/5, and the three-tier model blocks 5/5. Five correctness experiments confirm zero unauthorized deliveries, zero invariant violations, and architectural enforcement of out-of-band approval isolation that no surveyed enterprise platform provides. A survey of four major platforms (Microsoft, Salesforce, AWS, Google) documents that none architecturally isolates agent approval channels. We identify four properties that make context orchestration harder than container orchestration, and argue that these make the solution more valuable. 

---
# Intersectional Sycophancy: How Perceived User Demographics Shape False Validation in Large Language Models 

**Authors**: Benjamin Maltbie, Shivam Raval  

**Link**: [PDF](https://arxiv.org/pdf/2604.11609)  

**Abstract**: Large language models exhibit sycophantic tendencies--validating incorrect user beliefs to appear agreeable. We investigate whether this behavior varies systematically with perceived user demographics, testing whether combinations of race, age, gender, and expressed confidence level produce differential false validation rates. Inspired by the legal concept of intersectionality, we conduct 768 multi-turn adversarial conversations using Anthropic's Petri evaluation framework, probing GPT-5-nano and Claude Haiku 4.5 across 128 persona combinations in mathematics, philosophy, and conspiracy theory domains. GPT-5-nano is significantly more sycophantic than Claude Haiku 4.5 overall ($\bar{x}=2.96$ vs. $1.74$, $p < 10^{-32}$, Wilcoxon signed-rank). For GPT-5-nano, we find that philosophy elicits 41% more sycophancy than mathematics and that Hispanic personas receive the highest sycophancy across races. The worst-scoring persona, a confident, 23-year-old Hispanic woman, averages 5.33/10 on sycophancy. Claude Haiku 4.5 exhibits uniformly low sycophancy with no significant demographic variation. These results demonstrate that sycophancy is not uniformly distributed across users and that safety evaluations should incorporate identity-aware testing. 

---
# UniToolCall: Unifying Tool-Use Representation, Data, and Evaluation for LLM Agents 

**Authors**: Yijuan Liang, Xinghao Chen, Yifan Ge, Ziyi Wu, Hao Wu, Changyu Zeng, Wei Xing, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11557)  

**Abstract**: Tool-use capability is a fundamental component of LLM agents, enabling them to interact with external systems through structured function calls. However, existing research exhibits inconsistent interaction representations, largely overlooks the structural distribution of tool-use trajectories, and relies on incompatible evaluation benchmarks. We present UniToolCall, a unified framework for tool learning that standardizes the entire pipeline from toolset construction and dataset generation to evaluation. The framework curates a large tool pool of 22k+ tools and constructs a hybrid training corpus of 390k+ instances by combining 10 standardized public datasets with structurally controlled synthetic trajectories. It explicitly models diverse interaction patterns, including single-hop vs. multi-hop and single-turn vs. multi-turn, while capturing both serial and parallel execution structures. To support coherent multi-turn reasoning, we further introduce an Anchor Linkage mechanism that enforces cross-turn dependencies. Furthermore, we convert 7 public benchmarks into a unified Query--Action--Observation--Answer (QAOA) representation with fine-grained evaluation at the function-call, turn, and conversation levels. Experiments show that fine-tuning Qwen3-8B on our dataset substantially improves tool-use performance. Under the distractor-heavy Hybrid-20 setting, achieves 93.0% single-turn Strict Precision, outperforming commercial models including GPT, Gemini, and Claude. 

---
# SemaClaw: A Step Towards General-Purpose Personal AI Agents through Harness Engineering 

**Authors**: Ningyan Zhu, Huacan Wang, Jie Zhou, Feiyu Chen, Shuo Zhang, Ge Chen, Chen Liu, Jiarou Wu, Wangyi Chen, Xiaofeng Mou, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11548)  

**Abstract**: The rise of OpenClaw in early 2026 marks the moment when millions of users began deploying personal AI agents into their daily lives, delegating tasks ranging from travel planning to multi-step research. This scale of adoption signals that two parallel arcs of development have reached an inflection point. First is a paradigm shift in AI engineering, evolving from prompt and context engineering to harness engineering-designing the complete infrastructure necessary to transform unconstrained agents into controllable, auditable, and production-reliable systems. As model capabilities converge, this harness layer is becoming the primary site of architectural differentiation. Second is the evolution of human-agent interaction from discrete tasks toward a persistent, contextually aware collaborative relationship, which demands open, trustworthy and extensible harness infrastructure. We present SemaClaw, an open-source multi-agent application framework that addresses these shifts by taking a step towards general-purpose personal AI agents through harness engineering. Our primary contributions include a DAG-based two-phase hybrid agent team orchestration method, a PermissionBridge behavioral safety system, a three-tier context management architecture, and an agentic wiki skill for automated personal knowledge base construction. 

---
# A collaborative agent with two lightweight synergistic models for autonomous crystal materials research 

**Authors**: Tongyu Shi, Yutang Li, Zhanyuan Li, Qian Liu, Jie Zhou, Wenhe Xu, Yang Li, Dawei Dai, Rui He, Wenhua Zhou, Jiahong Wang, Xue-Feng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11540)  

**Abstract**: Current large language models require hundreds of billions of parameters yet struggle with domain-specific reasoning and tool coordination in materials science. Here, we present MatBrain, a lightweight collaborative agent system with two synergistic models specialization for crystal materials research. MatBrain employs a dual-model architecture: Mat-R1 (30B parameters) as the analytical model providing expert-level domain reasoning, and Mat-T1 (14B parameters) as the executive model orchestrating tool-based actions. Entropy analysis confirms that this architecture resolves the conflict between tool planning and analytical reasoning by decoupling their distinct entropy dynamics. Enabled by this dual-model architecture and structural efficiency, MatBrain significantly outperforms larger general-purpose models while reducing the hardware deployment barrier by over 95%. MatBrain exhibits versatility across structure generation, property prediction, and synthesis planning tasks. Applied to catalyst design, MatBrain generated 30,000 candidate structures and identified 38 promising materials within 48 hours, achieving approximately 100-fold acceleration over traditional approaches. These results demonstrate the potential of lightweight collaborative intelligence for advancing materials research capabilities. 

---
# Problem Reductions at Scale: Agentic Integration of Computationally Hard Problems 

**Authors**: Xi-Wei Pan, Shi-Wen An, Jin-Guo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11535)  

**Abstract**: Solving an NP-hard optimization problem often requires reformulating it for a specific solver -- quantum hardware, a commercial optimizer, or a domain heuristic. A tool for polynomial-time reductions between hard problems would let practitioners route any supported problem to any supported solver through a single interface. Building such a library at scale, however, has remained out of reach. We show that harness engineering, the practice of designing constraints, verification systems, and feedback loops that channel AI coding agents, can overcome this barrier. Our harness combines a no-code contribution route for domain experts, a multilayer verification stack ranging from type-level checks to agentic feature tests (AI agents role-playing as end users), and a fully automated implementation-review-integration pipeline. In about three months, we built a command-line tool backed by a library of 100+ problem types and 200+~reduction rules in over 170k lines of Rust. The result suggests that a well-engineered harness lets agents build well-tested software at a scale and pace beyond prior reduction-library efforts. Because the reduction graph composes transitively, a new solver registered for any single problem type instantly becomes available to every problem connected by a reduction path. The source code is available at this https URL. 

---
# Limited Perfect Monotonical Surrogates constructed using low-cost recursive linkage discovery with guaranteed output 

**Authors**: M.W. Przewozniczek, F. Chicano, R. Tinós, M.M. Komarnicki  

**Link**: [PDF](https://arxiv.org/pdf/2604.11524)  

**Abstract**: Surrogates provide a cheap solution evaluation and offer significant leverage for optimizing computationally expensive problems. Usually, surrogates only approximate the original function. Recently, the perfect linear surrogates were proposed that ideally represent the original function. These surrogates do not mimic the original function. In fact, they are another (correct) representation of it and enable a wide range of possibilities, e.g., discovering the optimized function for problems where the direct transformation of the encoded solution into its evaluation is not available. However, many real-world problems can not be represented by linear models, making the aforementioned surrogates inapplicable. Therefore, we propose the Limited Monotonical Perfect Surrogate (LyMPuS), which overcomes this difficulty and enables the comparison of two solutions that differ by a single variable. Our proposition is suitable for limiting the cost of expensive local search procedures. The proposed surrogate is parameterless and can be trained on the fly without any separate surrogate-building step. It uses only the necessary fitness evaluations, and the already-paid costs are not wasted when the model is updated. Finally, it offers low-cost missing-linkage detection and low-cost linkage discovery, guaranteed to find a missing dependency in no more than $2\lceil\log_2(n)\rceil$ steps. 

---
# PAC-BENCH: Evaluating Multi-Agent Collaboration under Privacy Constraints 

**Authors**: Minjun Park, Donghyun Kim, Hyeonjong Ju, Seungwon Lim, Dongwook Choi, Taeyoon Kwon, Minju Kim, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2604.11523)  

**Abstract**: We are entering an era in which individuals and organizations increasingly deploy dedicated AI agents that interact and collaborate with other agents. However, the dynamics of multi-agent collaboration under privacy constraints remain poorly understood. In this work, we present $PAC\text{-}Bench$, a benchmark for systematic evaluation of multi-agent collaboration under privacy constraints. Experiments on $PAC\text{-}Bench$ show that privacy constraints substantially degrade collaboration performance and make outcomes depend more on the initiating agent than the partner. Further analysis reveals that this degradation is driven by recurring coordination breakdowns, including early-stage privacy violations, overly conservative abstraction, and privacy-induced hallucinations. Together, our findings identify privacy-aware multi-agent collaboration as a distinct and unresolved challenge that requires new coordination mechanisms beyond existing agent capabilities. 

---
# Lectures on AI for Mathematics 

**Authors**: Xiaoyang Chen, Xiaoyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11504)  

**Abstract**: This book provides a comprehensive and accessible introduction to the emerging field of AI for mathematics. It covers the core principles and diverse applications of using artificial intelligence to advance mathematical research. Through clear explanations, the text explores how AI can discover hidden mathematical patterns, assist in proving complicated theorems, and even construct counterexamples to challenge conjectures. 

---
# Anthropogenic Regional Adaptation in Multimodal Vision-Language Model 

**Authors**: Samuel Cahyawijaya, Peerat Limkonchotiwat, Tack Hwa Wong, Hitesh Laxmichand Patel, Amit Agarwal, Manuel Antonio Rufino, Carlos Rafael Catalan, Muhammad Reza Qorib, Vicky Feliren, Holy Lovenia, Aye Hninn Khine, Frederikus Hudi, David Anugraha, Alham Fikri Aji, Romrawin Chumpu, Viet-Thanh Pham, Minghan Wang, Mohamed Fazli Imam, Ruochen Zhang, Joseph Marvin Imperial, Do Xuan Long, Musa Izzanardi Wijanarko, Joel Ruben Antony Moniz, Patrick Amadeus Irawan, Hanif Muhammad Zhafran, Isaiah Flores, Ira Salsabila, Jun Kevin, Jostin Jerico Rosal, Patricia Nicole Monderin, Kun Kerdthaisong, Ahmad Mustafid, My Chiffon Nguyen, Natchapon Jongwiriyanurak, Siva Worajitwannakul, Haochen Li, Adrian Xuan Wei Lim, Bin Wang, Muhammad Ravi Shulthan Habibi, Lynnette Hui Xian Ng, Mithil Bangera, Yeshil Bangera, Priyaranjan Pattnayak, Dun Li Chan, Sherissa Caren Djuniwar, Hee Ming Shan  

**Link**: [PDF](https://arxiv.org/pdf/2604.11490)  

**Abstract**: While the field of vision-language (VL) has achieved remarkable success in integrating visual and textual information across multiple languages and domains, there is still no dedicated framework for assessing human-centric alignment in vision-language systems. We offer two contributions to address this gap. First, we introduce Anthropogenic Regional Adaptation: a novel paradigm that aims to optimize model relevance to specific regional contexts while ensuring the retention of global generalization capabilities. Second, we present a simple, but effective adaptation method named Geographical-generalization-made-easy (GG-EZ), which utilizes regional data filtering and model merging. Through comprehensive experiments on 3 VL architectures: large vision-language models, text-to-image diffusion models, and vision-language embedding models, and a case study in Southeast Asia (SEA) regional adaptation, we demonstrate the importance of Anthropogenic Regional Adaptation and the effectiveness of GG-EZ, showing 5-15% gains in cultural relevance metrics across SEA while maintaining over 98% of global performance and even occasionally surpassing it. Our findings establish Anthropogenic Regional Alignment as a foundational paradigm towards applicability of multimodal vision-language models in diverse regions and demonstrate a simple-yet-effective baseline method that optimizes regional value alignment while preserving global generalization. 

---
# On the Complexity of the Discussion-based Semantics in Abstraction Argumentation 

**Authors**: Lydia Blümel, Kai Sauerwald, Kenneth Skiba, Matthias Thimm  

**Link**: [PDF](https://arxiv.org/pdf/2604.11480)  

**Abstract**: We show that deciding whether an argument a is stronger than an argument b with respect to the discussion-based semantics of Amgoud and Ben-Naim is decidable in polynomial time. At its core, this problem is about deciding whether, for two vertices in a graph, the number of walks of each length ending in those vertices is the same. We employ results from automata theory and reduce this problem to the equivalence problem for semiring automata. This offers a new perspective on the computational complexity of ranking semantics, an area in which the complexity of many semantics remains open. 

---
# OOM-RL: Out-of-Money Reinforcement Learning Market-Driven Alignment for LLM-Based Multi-Agent Systems 

**Authors**: Kun Liu, Liqun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11477)  

**Abstract**: The alignment of Multi-Agent Systems (MAS) for autonomous software engineering is constrained by evaluator epistemic uncertainty. Current paradigms, such as Reinforcement Learning from Human Feedback (RLHF) and AI Feedback (RLAIF), frequently induce model sycophancy, while execution-based environments suffer from adversarial "Test Evasion" by unconstrained agents. In this paper, we introduce an objective alignment paradigm: \textbf{Out-of-Money Reinforcement Learning (OOM-RL)}. By deploying agents into the non-stationary, high-friction reality of live financial markets, we utilize critical capital depletion as an un-hackable negative gradient. Our longitudinal 20-month empirical study (July 2024 -- February 2026) chronicles the system's evolution from a high-turnover, sycophantic baseline to a robust, liquidity-aware architecture. We demonstrate that the undeniable ontological consequences of financial loss forced the MAS to abandon overfitted hallucinations in favor of the \textbf{Strict Test-Driven Agentic Workflow (STDAW)}, which enforces a Byzantine-inspired uni-directional state lock (RO-Lock) anchored to a deterministically verified $\geq 95\%$ code coverage constraint matrix. Our results show that while early iterations suffered severe execution decay, the final OOM-RL-aligned system achieved a stable equilibrium with an annualized Sharpe ratio of 2.06 in its mature phase. We conclude that substituting subjective human preference with rigorous economic penalties provides a robust methodology for aligning autonomous agents in high-stakes, real-world environments, laying the groundwork for generalized paradigms where computational billing acts as an objective physical constraint 

---
# From Attribution to Action: A Human-Centered Application of Activation Steering 

**Authors**: Tobias Labarta, Maximilian Dreyer, Katharina Weitz, Wojciech Samek, Sebastian Lapuschkin  

**Link**: [PDF](https://arxiv.org/pdf/2604.11467)  

**Abstract**: Explainable AI (XAI) methods reveal which features influence model predictions, yet provide limited means for practitioners to act on these explanations. Activation steering of components identified via XAI offers a path toward actionable explanations, although its practical utility remains understudied. We introduce an interactive workflow combining SAE-based attribution with activation steering for instance-level analysis of concept usage in vision models, implemented as a web-based tool. Based on this workflow, we conduct semi-structured expert interviews (N=8) with debugging tasks on CLIP to investigate how practitioners reason about, trust, and apply activation steering. We find that steering enables a shift from inspection to intervention-based hypothesis testing (8/8 participants), with most grounding trust in observed model responses rather than explanation plausibility alone (6/8). Participants adopted systematic debugging strategies dominated by component suppression (7/8) and highlighted risks including ripple effects and limited generalization of instance-level corrections. Overall, activation steering renders interpretability more actionable while raising important considerations for safe and effective use. 

---
# Three Roles, One Model: Role Orchestration at Inference Time to Close the Performance Gap Between Small and Large Agents 

**Authors**: S. Aaron McClendon, Jorge Gallego-Feliciano, Stavros Zervoudakis, Antonios Saravanos  

**Link**: [PDF](https://arxiv.org/pdf/2604.11465)  

**Abstract**: Large language model (LLM) agents show promise on realistic tool-use tasks, but deploying capable agents on modest hardware remains challenging. We study whether inference-time scaffolding alone, without any additional training compute, can improve the performance of a small model in complex multi-step environments. Operating on a single 24\,GB GPU, we evaluate Qwen3-8B under both full-precision (FP16, 12K context) and 4-bit quantized (AWQ, 32K context) configurations. Without any intervention, the raw model achieves just 5.4\% (FP16) and 3.0\% (AWQ) task goal completion. Guided by a systematic failure mode analysis, we introduce a three-tier inference scaffolding pipeline that deploys the same frozen model in three distinct roles: (1) a summarization model that preserves critical artifacts (tokens, credentials, API responses) while compressing dialogue history; (2) the main agent model that reasons over the compressed context; and (3) an isolated correction model that reviews and revises the agent's code output without access to conversation history, breaking repetitive failure loops. Applied to the same unmodified model, this scaffolding yields 8.9\% (FP16) and 5.9\% (AWQ) task goal completion, roughly doubling performance in both settings, with particularly strong gains on difficulty-1 tasks (15.8\%$\to$26.3\% FP16; 5.3\%$\to$14.0\% AWQ). On full-precision inference, our scaffolded 8B model surpasses DeepSeek-Coder 33B Instruct (7.1\%) from the original AppWorld evaluation, demonstrating that structured inference-time interventions can make small models competitive with systems 4$\times$ their size. We formalize the approach as a scaffolded policy over a frozen base model, three invocations of the same weights with different conditioning, drawing connections to test-time compute scaling and action-space shaping in reinforcement learning. 

---
# Escaping the Context Bottleneck: Active Context Curation for LLM Agents via Reinforcement Learning 

**Authors**: Xiaozhe Li, Tianyi Lyu, Yizhao Yang, Liang Shan, Siyi Yang, Ligao Zhang, Zhuoyi Huang, Qingwen Liu, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11462)  

**Abstract**: Large Language Models (LLMs) struggle with long-horizon tasks due to the "context bottleneck" and the "lost-in-the-middle" phenomenon, where accumulated noise from verbose environments degrades reasoning over multi-turn interactions. To address this issue, we introduce a symbiotic framework that decouples context management from task execution. Our architecture pairs a lightweight, specialized policy model, ContextCurator, with a powerful frozen foundation model, TaskExecutor. Trained via reinforcement learning, ContextCurator actively reduces information entropy in the working memory. It aggressively prunes environmental noise while preserving reasoning anchors, that is, sparse data points that are critical for future deductions. On WebArena, our framework improves the success rate of Gemini-3.0-flash from 36.4% to 41.2% while reducing token consumption by 8.8% (from 47.4K to 43.3K). On DeepSearch, it achieves a 57.1% success rate, compared with 53.9%, while reducing token consumption by a factor of 8. Remarkably, a 7B ContextCurator matches the context management performance of GPT-4o, providing a scalable and computationally efficient paradigm for autonomous long-horizon agents. 

---
# Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval 

**Authors**: Dzenan Hamzic, Florian Skopik, Max Landauer, Markus Wurzenberger, Andreas Rauber  

**Link**: [PDF](https://arxiv.org/pdf/2604.11419)  

**Abstract**: Cyber threat intelligence (CTI) analysts must answer complex questions over large collections of narrative security reports. Retrieval-augmented generation (RAG) systems help language models access external knowledge, but traditional vector retrieval often struggles with queries that require reasoning over relationships between entities such as threat actors, malware, and vulnerabilities. This limitation arises because relevant evidence is often distributed across multiple text fragments and documents. Knowledge graphs address this challenge by enabling structured multi-hop reasoning through explicit representations of entities and relationships. However, multiple retrieval paradigms, including graph-based, agentic, and hybrid approaches, have emerged with different assumptions and failure modes. It remains unclear how these approaches compare in realistic CTI settings and when graph grounding improves performance. We present a systematic evaluation of four RAG architectures for CTI analysis: standard vector retrieval, graph-based retrieval over a CTI knowledge graph, an agentic variant that repairs failed graph queries, and a hybrid approach combining graph queries with text retrieval. We evaluate these systems on 3,300 CTI question-answer pairs spanning factual lookups, multi-hop relational queries, analyst-style synthesis questions, and unanswerable cases. Results show that graph grounding improves performance on structured factual queries. The hybrid graph-text approach improves answer quality by up to 35 percent on multi-hop questions compared to vector RAG, while maintaining more reliable performance than graph-only systems. 

---
# From Agent Loops to Structured Graphs:A Scheduler-Theoretic Framework for LLM Agent Execution 

**Authors**: Hu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.11378)  

**Abstract**: The dominant paradigm for building LLM based agents is the Agent Loop, an iterative cycle where a single language model decides what to do next by reading an ever growing context window. This paradigm has three structural weaknesses: implicit dependencies between steps, unbounded recovery loops, and mutable execution history that complicates debugging. We characterize the Agent Loop as a single ready unit scheduler: at any moment, at most one executable unit is active, and the choice of which unit to activate comes from opaque LLM inference rather than an inspectable policy. This perspective places Agent Loops and graph based execution engines on a single semantic continuum. We propose SGH, Structured Graph Harness, which lifts control flow from implicit context into an explicit static DAG. SGH makes three commitments: execution plans are immutable within a plan version, planning execution and recovery are separated into three layers, and recovery follows a strict escalation protocol. These choices trade some expressiveness for controllability, verifiability, and implementability. Our contributions are fourfold: a scheduler unified framework that applies classical scheduling theory to LLM agent execution and identifies challenges introduced by non deterministic LLM nodes; a trade off analysis of controllability, expressiveness, and implementability across 70 surveyed systems; a formal specification including a node state machine with termination and soundness guarantees; and an attributable experimental framework with a seven group design for future validation. This is a position paper and design proposal. We provide a theoretical framework, design analysis, and experimental protocol, not a production implementation or empirical results. 

---
# Learning from Contrasts: Synthesizing Reasoning Paths from Diverse Search Trajectories 

**Authors**: Peiyang Liu, Zhirui Chen, Xi Wang, Di Liang, Youru Li, Zhi Cai, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.11365)  

**Abstract**: Monte Carlo Tree Search (MCTS) has been widely used for automated reasoning data exploration, but current supervision extraction methods remain inefficient. Standard approaches retain only the single highest-reward trajectory, discarding the comparative signals present in the many explored paths. Here we introduce \textbf{Contrastive Reasoning Path Synthesis (CRPS)}, a framework that transforms supervision extraction from a filtering process into a synthesis procedure. CRPS uses a structured reflective process to analyze the differences between high- and low-quality search trajectories, extracting explicit information about strategic pivots and local failure modes. These insights guide the synthesis of reasoning chains that incorporate success patterns while avoiding identified pitfalls. We show empirically that models fine-tuned on just 60K CRPS-synthesized examples match or exceed the performance of baselines trained on 590K examples derived from standard rejection sampling, a 20$\times$ reduction in dataset size. Furthermore, CRPS improves generalization on out-of-domain benchmarks, demonstrating that learning from the contrast between success and failure produces more transferable reasoning capabilities than learning from success alone. 

---
# The Missing Knowledge Layer in Cognitive Architectures for AI Agents 

**Authors**: Michaël Roynard  

**Link**: [PDF](https://arxiv.org/pdf/2604.11364)  

**Abstract**: The two most influential cognitive architecture frameworks for AI agents, CoALA [21] and JEPA [12], both lack an explicit Knowledge layer with its own persistence semantics. This gap produces a category error: systems apply cognitive decay to factual claims, or treat facts and experiences with identical update mechanics. We survey persistence semantics across existing memory systems and identify eight convergence points, from Karpathy's LLM Knowledge Base [10] to the BEAM benchmark's near-zero contradiction-resolution scores [22], all pointing to related architectural gaps. We propose a four-layer decom position (Knowledge, Memory, Wisdom, Intelligence) where each layer has fundamentally different persistence semantics: indefinite supersession, Ebbinghaus decay, evidence-gated revision, and ephemeral inference respectively. Companion implementations in Python and Rust demonstrate the architectural separation is feasible. We borrow terminology from cognitive science as a useful analogy (the Knowledge/Memory distinction echoes Tulving's trichotomy), but our layers are engineering constructs justified by persistence-semantics requirements, not by neural architecture. We argue that these distinctions demand distinct persistence semantics in engineering implementations, and that no current framework or system provides this. 

---
# CoRe-ECG: Advancing Self-Supervised Representation Learning for 12-Lead ECG via Contrastive and Reconstructive Synergy 

**Authors**: Zehao Qin, Xiaojian Lin, Ping Zhang, Hongliang Wu, Xinkang Wang, Guangling Liu, Bo Chen, Wenming Yang, Guijin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11359)  

**Abstract**: Accurate interpretation of electrocardiogram (ECG) remains challenging due to the scarcity of labeled data and the high cost of expert annotation. Self-supervised learning (SSL) offers a promising solution by enabling models to learn expressive representations from unlabeled signals. Existing ECG SSL methods typically rely on either contrastive learning or reconstructive learning. However, each approach in isolation provides limited supervisory signals and suffers from additional limitations, including non-physiological distortions introduced by naive augmentations and trivial correlations across multiple leads that models may exploit as shortcuts. In this work, we propose CoRe-ECG, a unified contrastive and reconstructive pretraining paradigm that establishes a synergistic interaction between global semantic modeling and local structural learning. CoRe-ECG aligns global representations during reconstruction, enabling instance-level discriminative signals to guide local waveform recovery. To further enhance pretraining, we introduce Frequency Dynamic Augmentation (FDA) to adaptively perturb ECG signals based on their frequency-domain importance, and Spatio-Temporal Dual Masking (STDM) to break linear dependencies across leads, increasing the difficulty of reconstructive tasks. Our method achieves state-of-the-art performance across multiple downstream ECG datasets. Ablation studies further demonstrate the necessity and complementarity of each component. This approach provides a robust and physiologically meaningful representation learning framework for ECG analysis. 

---
# Dynamic Summary Generation for Interpretable Multimodal Depression Detection 

**Authors**: Shiyu Teng, Jiaqing Liu, Hao Sun, Yu Li, Shurong Chai, Ruibo Hou, Tomoko Tateyama, Lanfen Lin, Yen-Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11334)  

**Abstract**: Depression remains widely underdiagnosed and undertreated because stigma and subjective symptom ratings hinder reliable screening. To address this challenge, we propose a coarse-to-fine, multi-stage framework that leverages large language models (LLMs) for accurate and interpretable detection. The pipeline performs binary screening, five-class severity classification, and continuous regression. At each stage, an LLM produces progressively richer clinical summaries that guide a multimodal fusion module integrating text, audio, and video features, yielding predictions with transparent rationale. The system then consolidates all summaries into a concise, human-readable assessment report. Experiments on the E-DAIC and CMDC datasets show significant improvements over state-of-the-art baselines in both accuracy and interpretability. 

---
# Select Smarter, Not More: Prompt-Aware Evaluation Scheduling with Submodular Guarantees 

**Authors**: Xiaoyu Ma, Yiwen Li, Haoyue Liu, Zhichao Wang, Ye Chen, Yongxin Guo, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11328)  

**Abstract**: Automatic prompt optimization (APO) hinges on the quality of its evaluation signal, yet scoring every prompt candidate on the full training set is prohibitively expensive. Existing methods either fix a single evaluation subset before optimization begins (principled but prompt-agnostic) or adapt it heuristically during optimization (flexible but unstable and lacking formal guarantees). We observe that APO naturally maps to an online adaptive testing problem: prompts are examinees, training examples are test items, and the scheduler should select items that best discriminate among the strongest candidates. This insight motivates Prompt-Aware Online Evaluation Scheduling (POES), which integrates an IRT-based discrimination utility, a facility-location coverage term, and switching-cost-aware warm-start swaps into a unified objective that is provably monotone submodular, yielding a (1-1/e) greedy guarantee for cold starts and bounded drift for warm-start updates. An adaptive controller modulates the exploration-exploitation balance based on optimization progress. Across 36 tasks spanning three benchmark families, POES achieves the highest overall average accuracy (6.2 percent improvement over the best baseline) with negligible token overhead (approximately 4 percent) at the same evaluation budget. Moreover, principled selection at k = 20 examples matches or exceeds the performance of naive evaluation at k = 30-50, reducing token consumption by 35-60 percent, showing that selecting smarter is more effective than selecting more. Our results demonstrate that evaluation scheduling is a first-class component of APO, not an implementation detail. 

---
# PaperScope: A Multi-Modal Multi-Document Benchmark for Agentic Deep Research Across Massive Scientific Papers 

**Authors**: Lei Xiong, Huaying Yuan, Zheng Liu, Zhao Cao, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2604.11307)  

**Abstract**: Leveraging Multi-modal Large Language Models (MLLMs) to accelerate frontier scientific research is promising, yet how to rigorously evaluate such systems remains unclear. Existing benchmarks mainly focus on single-document understanding, whereas real scientific workflows require integrating evidence from multiple papers, including their text, tables, and figures. As a result, multi-modal, multi-document scientific reasoning remains underexplored and lacks systematic evaluation. To address this gap, we introduce PaperScope, a multi-modal multi-document benchmark designed for agentic deep research. PaperScope presents three advantages: (1) Structured scientific grounding. It is built on a knowledge graph of over 2,000 AI papers spanning three years, providing a structured foundation for research-oriented queries. (2) Semantically dense evidence construction. It integrates semantically related key information nodes and employs optimized random-walk article selector to sample thematically coherent paper sets, thereby ensuring adequate semantic density and task complexity. (3) Multi-task evaluation of scientific reasoning. It contains over 2,000 QA pairs across reasoning, retrieval, summarization, and problem solving, enabling evaluation of multi-step scientific reasoning. Experimental results show that even advanced systems such as OpenAI Deep Research and Tongyi Deep Research achieve limited scores on PaperScope, highlighting the difficulty of long-context retrieval and deep multi-source reasoning. PaperScope thus provides a rigorous benchmark alongside a scalable pipeline for constructing large-scale multi-modal, multi-source deep research datasets. 

---
# BankerToolBench: Evaluating AI Agents in End-to-End Investment Banking Workflows 

**Authors**: Elaine Lau, Markus Dücker, Ronak Chaudhary, Hui Wen Goh, Rosemary Wei, Vaibhav Kumar, Saed Qunbar, Guram Gogia, Yi Liu, Scott Millslagle, Nasim Borazjanizadeh, Ulyana Tkachenko, Samuel Eshun Danquah, Collin Schweiker, Vijay Karumathil, Asrith Devalaraju, Varsha Sandadi, Haemi Nam, Punit Arani, Ray Epps, Abdullah Arif, Sahil Bhaiwala, Curtis Northcutt, Skyler Wang, Anish Athalye, Jonas Mueller, Francisco Guzmán  

**Link**: [PDF](https://arxiv.org/pdf/2604.11304)  

**Abstract**: Existing AI benchmarks lack the fidelity to assess economically meaningful progress on professional workflows. To evaluate frontier AI agents in a high-value, labor-intensive profession, we introduce BankerToolBench (BTB): an open-source benchmark of end-to-end analytical workflows routinely performed by junior investment bankers. To develop an ecologically valid benchmark grounded in representative work environments, we collaborated with 502 investment bankers from leading firms. BTB requires agents to execute senior banker requests by navigating data rooms, using industry tools (market data platform, SEC filings database), and generating multi-file deliverables--including Excel financial models, PowerPoint pitch decks, and PDF/Word reports. Completing a BTB task takes bankers up to 21 hours, underscoring the economic stakes of successfully delegating this work to AI. BTB enables automated evaluation of any LLM or agent, scoring deliverables against 100+ rubric criteria defined by veteran investment bankers to capture stakeholder utility. Testing 9 frontier models, we find that even the best-performing model (GPT-5.4) fails nearly half of the rubric criteria and bankers rate 0% of its outputs as client-ready. Our failure analysis reveals key obstacles (such as breakdowns in cross-artifact consistency) and improvement directions for agentic AI in high-stakes professional workflows. 

---
# Consistency of AI-Generated Exercise Prescriptions: A Repeated Generation Study Using a Large Language Model 

**Authors**: Kihyuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.11287)  

**Abstract**: Background: Large language models (LLMs) have been explored as tools for generating personalized exercise prescriptions, yet the consistency of outputs under identical conditions remains insufficiently examined. Objective: This study evaluated the intra-model consistency of LLM-generated exercise prescriptions using a repeated generation design. Methods: Six clinical scenarios were used to generate exercise prescriptions using Gemini 2.5 Flash (20 outputs per scenario; total n = 120). Consistency was assessed across three dimensions: (1) semantic consistency using SBERT-based cosine similarity, (2) structural consistency based on the FITT principle using an AI-as-a-judge approach, and (3) safety expression consistency, including inclusion rates and sentence-level quantification. Results: Semantic similarity was high across scenarios (mean cosine similarity: 0.879-0.939), with greater consistency in clinically constrained cases. Frequency showed consistent patterns, whereas variability was observed in quantitative components, particularly exercise intensity. Unclassifiable intensity expressions were observed in 10-25% of resistance training outputs. Safety-related expressions were included in 100% of outputs; however, safety sentence counts varied significantly across scenarios (H=86.18, p less than 0.001), with clinical cases generating more safety expressions than healthy adult cases. Conclusions: LLM-generated exercise prescriptions demonstrated high semantic consistency but showed variability in key quantitative components. Reliability depends substantially on prompt structure, and additional structural constraints and expert validation are needed before clinical deployment. 

---
# Inspectable AI for Science: A Research Object Approach to Generative AI Governance 

**Authors**: Ruta Binkyte, Sharif Abuaddba, Chamikara Mahawaga, Ming Ding, Natasha Fernandes, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2604.11261)  

**Abstract**: This paper introduces AI as a Research Object (AI-RO), a paradigm for governing the use of generative AI in scientific research. Instead of debating whether AI is an author or merely a tool, we propose treating AI interactions as structured, inspectable components of the research process. Under this view, the legitimacy of an AI-assisted scientific paper depends on how model use is integrated into the workflow, documented, and made accountable. Drawing on Research Object theory and FAIR principles, we propose a framework for recording model configuration, prompts, and outputs through interaction logs and metadata packaging. These properties are particularly consequential in security and privacy (S&P) research, where provenance artifacts must satisfy confidentiality constraints, integrity guarantees, and auditability requirements that generic disclosure practices do not address. We implement a lightweight writing pipeline in which a language model synthesizes human-authored structured literature review notes under explicit constraints and produces a verifiable provenance record. We present this work as a position supported by an initial demonstrative workflow, arguing that governance of generative AI in science can be implemented as structured documentation, controlled disclosure, and integrity-preserving provenance capture. Based on this example, we outline and motivate a set of necessary future developments required to make such practices practical and widely adoptable. 

---
# Mobile GUI Agent Privacy Personalization with Trajectory Induced Preference Optimization 

**Authors**: Zhixin Lin, Jungang Li, Dongliang Xu, Shidong Pan, Yibo Shi, Yuchi Liu, Yuecong Min, Yue Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.11259)  

**Abstract**: Mobile GUI agents powered by Multimodal Large Language Models (MLLMs) can execute complex tasks on mobile devices. Despite this progress, most existing systems still optimize task success or efficiency, neglecting users' privacy personalization. In this paper, we study the often-overlooked problem of agent personalization. We observe that personalization can induce systematic structural heterogeneity in execution trajectories. For example, privacy-first users often prefer protective actions, e.g., refusing permissions, logging out, and minimizing exposure, leading to logically different execution trajectories from utility-first users. Such variable-length and structurally different trajectories make standard preference optimization unstable and less informative. To address this issue, we propose Trajectory Induced Preference Optimization (TIPO), which uses preference-intensity weighting to emphasize key privacy-related steps and padding gating to suppress alignment noise. Results on our Privacy Preference Dataset show that TIPO improves persona alignment and distinction while preserving strong task executability, achieving 65.60% SR, 46.22 Compliance, and 66.67% PD, outperforming existing optimization methods across various GUI tasks. The code and dataset will be publicly released at this https URL. 

---
# Measuring the Authority Stack of AI Systems: Empirical Analysis of 366,120 Forced-Choice Responses Across 8 AI Models 

**Authors**: Seulki Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.11216)  

**Abstract**: What values, evidence preferences, and source trust hierarchies do AI systems actually exhibit when facing structured dilemmas? We present the first large-scale empirical mapping of AI decision-making across all three layers of the Authority Stack framework (S. Lee, 2026a): value priorities (L4), evidence-type preferences (L3), and source trust hierarchies (L2). Using the PRISM benchmark -- a forced-choice instrument of 14,175 unique scenarios per layer, spanning 7 professional domains, 3 severity levels, 3 decision timeframes, and 5 scenario variants -- we evaluated 8 major AI models at temperature 0, yielding 366,120 total responses. Key findings include: (1) a symmetric 4:4 split between Universalism-first and Security-first models at L4; (2) dramatic defense-domain value restructuring where Security surges to near-ceiling win-rates (95.1%-99.8%) in 6 of 8 models; (3) divergent evidence hierarchies at L3, with some models favoring empirical-scientific evidence while others prefer pattern-based or experiential evidence; (4) broad convergence on institutional source trust at L2; and (5) Paired Consistency Scores (PCS) ranging from 57.4% to 69.2%, revealing substantial framing sensitivity across scenario variants. Test-Retest Reliability (TRR) ranges from 91.7% to 98.6%, indicating that value instability stems primarily from variant sensitivity rather than stochastic noise. These findings demonstrate that AI models possess measurable -- if sometimes unstable -- Authority Stacks with consequential implications for deployment across professional domains. 

---
# Environmental Footprint of GenAI Research: Insights from the Moshi Foundation Model 

**Authors**: Marta López-Rauhut, Loic Landrieu, Mathieu Aubry, Anne-Laure Ligozat  

**Link**: [PDF](https://arxiv.org/pdf/2604.11154)  

**Abstract**: New multi-modal large language models (MLLMs) are continuously being trained and deployed, following rapid development cycles. This generative AI frenzy is driving steady increases in energy consumption, greenhouse gas emissions, and a plethora of other environmental impacts linked to datacenter construction and hardware manufacturing. Mitigating the environmental consequences of GenAI remains challenging due to an overall lack of transparency by the main actors in the field. Even when the environmental impacts of specific models are mentioned, they are typically restricted to the carbon footprint of the final training run, omitting the research and development stages.
In this work, we explore the impact of GenAI research through a fine-grained analysis of the compute spent to create Moshi, a 7B-parameter speech-text foundation model for real-time dialogue developed by Kyutai, a leading privately funded open science AI lab. For the first time, our study dives into the anatomy of compute-intensive MLLM research, quantifying the GPU-time invested in specific model components and training phases, as well as early experimental stages, failed training runs, debugging, and ablation studies. Additionally, we assess the environmental impacts of creating Moshi from beginning to end using a life cycle assessment methodology: we quantify energy and water consumption, greenhouse gas emissions, and mineral resource depletion associated with the production and use of datacenter hardware.
Our detailed analysis allows us to provide actionable guidelines to reduce compute usage and environmental impacts of MLLM research, paving the way for more sustainable AI research. 

---
# From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning 

**Authors**: Chen Zhan, Xiaoyu Tan, Gengchen Ma, Yu-Jie Xiong, Xiaoyan Jiang, Xihe Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11137)  

**Abstract**: The integration of Large Language Models (LLMs) into clinical decision support is critically obstructed by their opaque and often unreliable reasoning. In the high-stakes domain of healthcare, correct answers alone are insufficient; clinical practice demands full transparency to ensure patient safety and enable professional accountability. A pervasive and dangerous weakness of current LLMs is their tendency to produce "correct answers through flawed reasoning." This issue is far more than a minor academic flaw; such process errors signal a fundamental lack of robust understanding, making the model prone to broader hallucinations and unpredictable failures when faced with real-world clinical complexity. In this paper, we establish a framework for trustworthy clinical argumentation by adapting the Toulmin model to the diagnostic process. We propose a novel training pipeline: Curriculum Goal-Conditioned Learning (CGCL), designed to progressively train LLM to generate diagnostic arguments that explicitly follow this Toulmin structure. CGCL's progressive three-stage curriculum systematically builds a solid clinical argument: (1) extracting facts and generating differential diagnoses; (2) justifying a core hypothesis while rebutting alternatives; and (3) synthesizing the analysis into a final, qualified conclusion. We validate CGCL using T-Eval, a quantitative framework measuring the integrity of the diagnosis reasoning. Experiments show that our method achieves diagnostic accuracy and reasoning quality comparable to resource-intensive Reinforcement Learning (RL) methods, while offering a more stable and efficient training pipeline. 

---
# MADQRL: Distributed Quantum Reinforcement Learning Framework for Multi-Agent Environments 

**Authors**: Abhishek Sawaika, Samuel Yen-Chi Chen, Udaya Parampalli, Rajkumar Buyya  

**Link**: [PDF](https://arxiv.org/pdf/2604.11131)  

**Abstract**: Reinforcement learning (RL) is one of the most practical ways to learn from real-life use-cases. Motivated from the cognitive methods used by humans makes it a widely acceptable strategy in the field of artificial intelligence. Most of the environments used for RL are often high-dimensional, and traditional RL algorithms becomes computationally expensive and challenging to effectively learn from such systems. Recent advancements in practical demonstration of quantum computing (QC) theories, such as compact encoding, enhanced representation and learning algorithms, random sampling, or the inherent stochastic nature of quantum systems, have opened up new directions to tackle these challenges. Quantum reinforcement learning (QRL) is seeking significant traction over the past few years. However, the current state of quantum hardware is not enough to cater for such high-dimensional environments with complex multi-agent setup. To tackle this issue, we propose a distributed framework for QRL where multiple agents learn independently, distributing the load of joint training from individual machines. Our method works well for environments with disjoint sets of action and observation spaces, but can also be extended to other systems with reasonable approximations. We analyze the proposed method on cooperative-pong environment and our results indicate ~10% improvement from other distribution strategies, and ~5% improvement from classical models of policy representation. 

---
# A Proposed Biomedical Data Policy Framework to Reduce Fragmentation, Improve Quality, and Incentivize Sharing in Indian Healthcare in the era of Artificial Intelligence and Digital Health 

**Authors**: Nikhil Mehta, Sachin Gupta, Gouri RP Anand  

**Link**: [PDF](https://arxiv.org/pdf/2604.11125)  

**Abstract**: India generates vast biomedical data through postgraduate research, government hospital services and audits, government schemes, private hospitals and their electronic medical record (EMR) systems, insurance programs and standalone clinics. Unfortunately, these resources remain fragmented across institutional silos and vendor-locked EMR systems. The fundamental bottleneck is not technological but economic and academic. There is a systemic misalignment of incentives that renders data sharing a high-risk, low-reward activity for individual researchers and institutions. Until India's academic promotion criteria, institutional rankings, and funding mechanisms explicitly recognize and reward data curation as professional work, the nation's AI ambitions will remain constrained by fragmented, non-interoperable datasets. We propose a multi-layered incentive architecture integrating recognition of data papers in National Medical Commission (NMC) promotion criteria, incorporation of open data metrics into the National Institutional Ranking Framework (NIRF), adoption of Shapley Value-based revenue sharing in federated learning consortia, and establishment of institutional data stewardship as a mainstream professional role. Critical barriers to data sharing, including fear of data quality scrutiny, concerns about misinterpretation, and selective reporting bias, are addressed through mandatory data quality assessment, structured peer review, and academic credit for auditing roles. The proposed framework directly addresses regulatory constraints introduced by the Digital Personal Data Protection Act 2023 (DPDPA), while constructively engaging with the National Data Sharing and Accessibility Policy (NDSAP), Biotech-PRIDE Guidelines, and the Anusandhan National Research Foundation (ANRF) guidelines. 

---
# Persona Non Grata: Single-Method Safety Evaluation Is Incomplete for Persona-Imbued LLMs 

**Authors**: Wenkai Li, Fan Yang, Shaunak A. Mehta, Koichi Onoue  

**Link**: [PDF](https://arxiv.org/pdf/2604.11120)  

**Abstract**: Personality imbuing customizes LLM behavior, but safety evaluations almost always study prompt-based personas alone. We show this is incomplete: prompting and activation steering expose *different*, architecture-dependent vulnerability profiles, and testing with only one method can miss a model's dominant failure mode. Across 5,568 judged conditions on four standard models from three architecture families, persona danger rankings under system prompting are preserved across all architectures ($\rho = 0.71$--$0.96$), but activation-steering vulnerability diverges sharply and cannot be predicted from prompt-side rankings: Llama-3.1-8B is substantially more AS-vulnerable, whereas Gemma-3-27B and Qwen3.5 are more vulnerable to prompting. The most striking illustration of this divergence is the *prosocial persona paradox*: on Llama-3.1-8B, P12 (high conscientiousness + high agreeableness) is among the safest personas under prompting yet becomes the highest-ASR activation-steered persona (ASR ~0.818). This is an inversion robust to coefficient ablation and matched-strength calibration, and replicated on DeepSeek-R1-Distill-Qwen-32B. A trait refusal alignment framework, in which conscientiousness is strongly anti-aligned with refusal on Llama-3.1-8B, offers a partial geometric account. Reasoning provides only partial protection: two 32B reasoning models reach 15--18% prompt-side ASR, and activation steering separates them sharply in both baseline susceptibility and persona-specific vulnerability. Heuristic trace diagnostics suggest that the safer model retains stronger policy recall and self-correction behavior, not merely longer reasoning. 

---
# Frugal Knowledge Graph Construction with Local LLMs: A Zero-Shot Pipeline, Self-Consistency and Wisdom of Artificial Crowds 

**Authors**: Pierre Jourlin  

**Link**: [PDF](https://arxiv.org/pdf/2604.11104)  

**Abstract**: This paper presents an empirical study of a multi-model zero-shot pipeline for knowledge graph construction and exploitation, executed entirely through local inference on consumer-grade hardware. We propose a reproducible evaluation framework integrating two external benchmarks (DocRED, HotpotQA), WebQuestionsSP-style synthetic data, and the RAGAS evaluation framework in an automated pipeline. On 500 document-level relations, our system achieves an F1 of 0.70 $\pm$ 0.041 in zero-shot, compared to 0.80 for supervised DREEAM. Text-to-query achieves an accuracy of 0.80 $\pm$ 0.06 on 200 samples. Multi-hop reasoning achieves an Exact Match (EM) of 0.46$\pm$0.04 on 500 HotpotQA questions, with a RAGAS faithfulness of 0.96 $\pm$ 0.04 on 50 samples. Beyond the pipeline, we study diversity mechanisms for difficult multi-hop reasoning. On 181 questions unsolvable at zero temperature, self-consistency (k=5, T =0.7) recovers up to 23% EM with a single Mixture-of-Experts (MoE) model, but the cross-model oracle (3 architectures x 5 samples) reaches 46.4%. We highlight an agreement paradox: strong consensus among samples signals collective hallucination rather than a reliable answer, echoing the work of Moussa{ï}d et al. on the wisdom of crowds. Extending to the full pipeline (500 questions), self-consistency (k=3) raises EM from 0.46 to 0.48 $\pm$ 0.04. A confidence-routing cascade mechanism (Phi-4 $\rightarrow$ GPT-OSS, k=5) achieves an EM of 0.55 $\pm$ 0.04, the best result obtained, with 45.4% of questions rerouted. Finally, we show that V3 prompt engineering applied to other models does not reproduce the gains observed with Gemma-4, confirming the specific prompt/model interaction. The entire system runs in $\sim$5 h on a single RTX 3090, without any training, for an estimated carbon footprint of 0.09 kg CO2 eq. 

---
# Do Agent Rules Shape or Distort? Guardrails Beat Guidance in Coding Agents 

**Authors**: Xing Zhang, Guanghui Wang, Yanwei Cui, Wei Qiu, Ziyuan Li, Bing Zhu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2604.11088)  

**Abstract**: Developers increasingly guide AI coding agents through natural language instruction files (e.g., this http URL, .cursorrules), yet no controlled study has measured whether these rules actually improve agent performance or which properties make a rule beneficial. We scrape 679 such files (25,532 rules) from GitHub and conduct the first large-scale empirical evaluation, running over 5,000 agent runs with a state-of-the-art coding agent on SWE-bench Verified. Rules improve performance by 7--14 percentage points, but random rules help as much as expert-curated ones -- suggesting rules work through context priming rather than specific instruction. Negative constraints ("do not refactor unrelated code") are the only individually beneficial rule type, while positive directives ("follow code style") actively hurt -- a pattern we analyze through the lens of potential-based reward shaping (PBRS). Moreover, individual rules are mostly harmful in isolation yet collectively helpful, with no degradation up to 50 rules. These findings expose a hidden reliability risk -- well-intentioned rules routinely degrade agent performance -- and provide a clear principle for safe agent configuration: constrain what agents must not do, rather than prescribing what they should. 

---
# Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation 

**Authors**: Chen Huang, Zitan Jiang, Changyi Zou, Wenqiang Lei, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.11077)  

**Abstract**: Customer service chatbots are increasingly expected to serve not merely as reactive support tools for users, but as strategic interfaces for harvesting high-value information and business intelligence. In response, we make three main contributions. 1) We introduce and define a novel task of Proactive Information Probing, which optimizes when to probe users for pre-specified target information while minimizing conversation turns and user friction. 2) We propose PROCHATIP, a proactive chatbot framework featuring a specialized conversation strategy module trained to master the delicate timing of probes. 3) Experiments demonstrate that PROCHATIP significantly outperforms baselines, exhibiting superior capability in both information probing and service quality. We believe that our work effectively redefines the commercial utility of chatbots, positioning them as scalable, cost-effective engines for proactive business intelligence. Our code is available at this https URL. 

---
# Hodoscope: Unsupervised Monitoring for AI Misbehaviors 

**Authors**: Ziqian Zhong, Shashwat Saxena, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2604.11072)  

**Abstract**: Existing approaches to monitoring AI agents rely on supervised evaluation: human-written rules or LLM-based judges that check for known failure modes. However, novel misbehaviors may fall outside predefined categories entirely and LLM-based judges can be unreliable. To address this, we formulate unsupervised monitoring, drawing an analogy to unsupervised learning. Rather than checking for specific misbehaviors, an unsupervised monitor assists humans in discovering problematic agent behaviors without prior assumptions about what counts as problematic, leaving that determination to the human.
We observe that problematic behaviors are often distinctive: a model exploiting a benchmark loophole exhibits actions absent from well-behaved baselines, and a vulnerability unique to one evaluation manifests as behavioral anomalies when the same model runs across multiple benchmarks. This motivates using group-wise behavioral differences as the primary signal for unsupervised monitoring. We introduce Hodoscope, a tool that operationalizes this insight. Hodoscope compares behavior distributions across groups and highlights distinctive and potentially suspicious action patterns for human review. Using Hodoscope, we discover a previously unknown vulnerability in the Commit0 benchmark (unsquashed git history allowing ground-truth recovery, inflating scores for at least five models) and independently recover known exploits on ImpossibleBench and SWE-bench. Quantitative evaluation estimates that our method reduces review effort by 6-23$\times$ compared to naive uniform sampling. Finally, we show that behavior descriptions discovered through Hodoscope could improve the detection accuracy of LLM-based judges, demonstrating a path from unsupervised to supervised monitoring. 

---
# PRISM Risk Signal Framework: Hierarchy-Based Red Lines for AI Behavioral Risk 

**Authors**: Seulki Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.11070)  

**Abstract**: Current approaches to AI safety define red lines at the case level: specific prompts, specific outputs, specific harms. This paper argues that red lines can be set more fundamentally -- at the level of value, evidence, and source hierarchies that govern AI reasoning. Using the PRISM (Profile-based Reasoning Integrity Stack Measurement) framework, we define a taxonomy of 27 behavioral risk signals derived from structural anomalies in how AI systems prioritize values (L4), weight evidence types (L3), and trust information sources (L2). Each signal is evaluated through a dual-threshold principle combining absolute rank position and relative win-rate gap, producing a two-tier classification (Confirmed Risk vs. Watch Signal). The hierarchy-based approach offers three advantages over case-specific red lines: it is anticipatory rather than reactive (detecting dangerous reasoning structures before they produce harmful outputs), comprehensive rather than enumerative (a single value-hierarchy signal subsumes an unlimited number of case-specific violations), and measurable rather than subjective (grounded in empirical forced-choice data). We demonstrate the framework's detection capacity using approximately 397,000 forced-choice responses from 7 AI models across three Authority Stack layers, showing that the signal taxonomy successfully discriminates between models with structurally extreme profiles, models with context-dependent risk, and models with balanced hierarchies. 

---
# AI Integrity: A New Paradigm for Verifiable AI Governance 

**Authors**: Seulki Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.11065)  

**Abstract**: AI systems increasingly shape high-stakes decisions in healthcare, law, defense, and education, yet existing governance paradigms -- AI Ethics, AI Safety, and AI Alignment -- share a common limitation: they evaluate outcomes rather than verifying the reasoning process itself. This paper introduces AI Integrity, a concept defined as a state in which the Authority Stack of an AI system -- its layered hierarchy of values, epistemological standards, source preferences, and data selection criteria -- is protected from corruption, contamination, manipulation, and bias, and maintained in a verifiable manner. We distinguish AI Integrity from the three existing paradigms, define the Authority Stack as a 4-layer cascade model (Normative, Epistemic, Source, and Data Authority) grounded in established academic frameworks -- Schwartz Basic Human Values for normative authority, Walton argumentation schemes with GRADE/CEBM hierarchies for epistemic authority, and Source Credibility Theory for source authority -- characterize the distinction between legitimate cascading and Authority Pollution, and identify Integrity Hallucination as the central measurable threat to value consistency. We further specify the PRISM (Profile-based Reasoning Integrity Stack Measurement) framework as the operational methodology, defining six core metrics and a phased research roadmap. Unlike normative frameworks that prescribe which values are correct, AI Integrity is a procedural concept: it requires that the path from evidence to conclusion be transparent and auditable, regardless of which values a system holds. 

---
# EmergentBridge: Improving Zero-Shot Cross-Modal Transfer in Unified Multimodal Embedding Models 

**Authors**: Jincheng Xie, Xingchen Xiao, Runheng Liu, Zhongyi Huang, Yu Zheng, Heyan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11043)  

**Abstract**: Unified multimodal embedding spaces underpin practical applications such as cross-modal retrieval and zero-shot recognition. In many real deployments, however, supervision is available only for a small subset of modality pairs (e.g., image--text), leaving \emph{unpaired} modality pairs (e.g., audio$\leftrightarrow$depth, infrared$\leftrightarrow$audio) weakly connected and thus performing poorly on zero-shot transfer. Addressing this sparse-pairing regime is therefore essential for scaling unified embedding systems to new tasks without curating exhaustive pairwise data. We propose \textbf{EmergentBridge}, an embedding-level bridging framework that improves performance on these unpaired pairs \emph{without requiring exhaustive pairwise supervision}. Our key observation is that naively aligning a new modality to a synthesized proxy embedding can introduce \emph{gradient interference}, degrading the anchor-alignment structure that existing retrieval/classification relies on. EmergentBridge addresses this by (i) learning a mapping that produces a \emph{noisy bridge anchor} (a proxy embedding of an already-aligned modality) from an anchor embedding, and (ii) enforcing proxy alignment only in the subspace orthogonal to the anchor-alignment direction, preserving anchor alignment while strengthening non-anchor connectivity. Across nine datasets spanning multiple modalities, EmergentBridge consistently outperforms prior binding baselines on zero-shot classification and retrieval, demonstrating strong emergent alignment. 

---
# From Topology to Trajectory: LLM-Driven World Models For Supply Chain Resilience 

**Authors**: Jia Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.11041)  

**Abstract**: Semiconductor supply chains face unprecedented resilience challenges amidst global geopolitical turbulence. Conventional Large Language Model (LLM) planners, when confronting such non-stationary "Policy Black Swan" events, frequently suffer from Decision Paralysis or a severe Grounding Gap due to the absence of physical environmental modeling. This paper introduces ReflectiChain, a cognitive agentic framework tailored for resilient macroeconomic supply chain planning. The core innovation lies in the integration of Latent Trajectory Rehearsal powered by a generative world model, which couples reflection-in-action (System 2 deliberation) with delayed reflection-on-action. Furthermore, we leverage a Retrospective Agentic RL mechanism to enable autonomous policy evolution during the deployment phase (test-time). Evaluations conducted on our high-fidelity benchmark, Semi-Sim, demonstrate that under extreme scenarios such as export bans and material shortages, ReflectiChain achieves a 250% improvement in average step rewards over the strongest LLM baselines. It successfully restores the Operability Ratio (OR) from a deficient 13.3% to over 88.5% while ensuring robust gradient convergence. Ablation studies further underscore that the synergy between physical grounding constraints and double-loop learning is fundamental to bridging the gap between semantic reasoning and physical reality for long-horizon strategic planning. 

---
# Intelligent Approval of Access Control Flow in Office Automation Systems via Relational Modeling 

**Authors**: Dugang Liu, Zulong Chen, Chuanfei Xu, Jiaxuan He, Yunlu Ma, Jia Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11040)  

**Abstract**: Office automation (OA) systems play a crucial role in enterprise operations and management, with access control flow approval (ACFA) being a key component that manages the accessibility of various resources. However, traditional ACFA requires approval from the person in charge at each step, which consumes a significant amount of manpower and time. Its intelligence is a crucial issue that needs to be addressed urgently by all companies. In this paper, we propose a novel relational modeling-driven intelligent approval (RMIA) framework to automate ACFA. Specifically, our RMIA consists of two core modules: (1) The binary relation modeling module aims to characterize the coupling relation between applicants and approvers and provide reliable basic information for ACFA decision-making from a coarse-grained perspective. (2) The ternary relation modeling module utilizes specific resource information as its core, characterizing the complex relations between applicants, resources, and approvers, and thus provides fine-grained gain information for informed decision-making. Then, our RMIA effectively fuses these two kinds of information to form the final decision. Finally, extensive experiments are conducted on two product datasets and an online A/B test to verify the effectiveness of RMIA. 

---
# Introspective Diffusion Language Models 

**Authors**: Yifan Yu, Yuqing Jian, Junxiong Wang, Zhongzhu Zhou, Donglin Zhuang, Xinyu Fang, Sri Yanamandra, Xiaoxia Wu, Qingyang Wu, Shuaiwen Leon Song, Tri Dao, Ben Athiwaratkun, James Zou, Fan Lai, Chenfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11035)  

**Abstract**: Diffusion language models promise parallel generation, yet still lag behind autoregressive (AR) models in quality. We stem this gap to a failure of introspective consistency: AR models agree with their own generations, while DLMs often do not. We define the introspective acceptance rate, which measures whether a model accepts its previously generated tokens. This reveals why AR training has a structural advantage: causal masking and logit shifting implicitly enforce introspective consistency. Motivated by this observation, we introduce Introspective Diffusion Language Model (I-DLM), a paradigm that retains diffusion-style parallel decoding while inheriting the introspective consistency of AR training. I-DLM uses a novel introspective strided decoding (ISD) algorithm, which enables the model to verify previously generated tokens while advancing new ones in the same forward pass. From a systems standpoint, we build I-DLM inference engine on AR-inherited optimizations and further customize it with a stationary-batch scheduler. To the best of our knowledge, I-DLM is the first DLM to match the quality of its same-scale AR counterpart while outperforming prior DLMs in both model quality and practical serving efficiency across 15 benchmarks. It reaches 69.6 on AIME-24 and 45.7 on LiveCodeBench-v6, exceeding LLaDA-2.1-mini (16B) by more than 26 and 15 points, respectively. Beyond quality, I-DLM is designed for the growing demand of large-concurrency serving, delivering about 3x higher throughput than prior state-of-the-art DLMs. 

---
# Min-$k$ Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics 

**Authors**: Yuanhao Ding, Meimingwei Li, Esteban Garces Arias, Matthias Aßenmacher, Christian Heumann, Chongsheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11012)  

**Abstract**: The quality of text generated by large language models depends critically on the decoding sampling strategy. While mainstream methods such as Top-$k$, Top-$p$, and Min-$p$ achieve a balance between diversity and accuracy through probability-space truncation, they share an inherent limitation: extreme sensitivity to the temperature parameter. Recent logit-space approaches like Top-$n\sigma$ achieve temperature invariance but rely on global statistics that are susceptible to long-tail noise, failing to capture fine-grained confidence structures among top candidates. We propose \textbf{Min-$k$ Sampling}, a novel dynamic truncation strategy that analyzes the local shape of the sorted logit distribution to identify "semantic cliffs": sharp transitions from high-confidence core tokens to uncertain long-tail tokens. By computing a position-weighted relative decay rate, Min-$k$ dynamically determines truncation boundaries at each generation step. We formally prove that Min-$k$ achieves strict temperature invariance and empirically demonstrate its low sensitivity to hyperparameter choices. Experiments on multiple reasoning benchmarks, creative writing tasks, and human evaluation show that Min-$k$ consistently improves text quality, maintaining robust performance even under extreme temperature settings where probability-based methods collapse. We make our code, models, and analysis tools publicly available. 

---
# Diffusion-CAM: Faithful Visual Explanations for dMLLMs 

**Authors**: Haomin Zuo, Yidi Li, Luoxiao Yang, Xiaofeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11005)  

**Abstract**: While diffusion Multimodal Large Language Models (dMLLMs) have recently achieved remarkable strides in multimodal generation, the development of interpretability mechanisms has lagged behind their architectural evolution. Unlike traditional autoregressive models that produce sequential activations, diffusion-based architectures generate tokens via parallel denoising, resulting in smooth, distributed activation patterns across the entire sequence. Consequently, existing Class Activation Mapping (CAM) methods, which are tailored for local, sequential dependencies, are ill-suited for interpreting these non-autoregressive behaviors. To bridge this gap, we propose Diffusion-CAM, the first interpretability method specifically tailored for dMLLMs. We derive raw activation maps by differentiably probing intermediate representations in the transformer backbone, accordingly capturing both latent features and their class-specific gradients. To address the inherent stochasticity of these raw signals, we incorporate four key modules to resolve spatial ambiguity and mitigate intra-image confounders and redundant token correlations. Extensive experiments demonstrate that Diffusion-CAM significantly outperforms SoTA methods in both localization accuracy and visual fidelity, establishing a new standard for understanding the parallel generation process of diffusion multimodal systems. 

---
# Sanity Checks for Agentic Data Science 

**Authors**: Zachary T. Rewolinski, Austin V. Zane, Hao Huang, Chandan Singh, Chenglong Wang, Jianfeng Gao, Bin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11003)  

**Abstract**: Agentic data science (ADS) pipelines have grown rapidly in both capability and adoption, with systems such as OpenAI Codex now able to directly analyze datasets and produce answers to statistical questions. However, these systems can reach falsely optimistic conclusions that are difficult for users to detect. To address this, we propose a pair of lightweight sanity checks grounded in the Predictability-Computability-Stability (PCS) framework for veridical data science. These checks use reasonable perturbations to screen whether an agent can reliably distinguish signal from noise, acting as a falsifiability constraint that can expose affirmative conclusions as unsupported. Together, the two checks characterize the trustworthiness of an ADS output, e.g. whether it has found stable signal, is responding to noise, or is sensitive to incidental aspects of the input. We validate the approach on synthetic data with controlled signal-to-noise ratios, confirming that the sanity checks track ground-truth signal strength. We then demonstrate the checks on 11 real-world datasets using OpenAI Codex, characterizing the trustworthiness of each conclusion and finding that in 6 of the datasets an affirmative conclusion is not well-supported, even though a single ADS run may support one. We further analyze failure modes of ADS systems and find that ADS self-reported confidence is poorly calibrated to the empirical stability of its conclusions. 

---
# MAFIG: Multi-agent Driven Formal Instruction Generation Framework 

**Authors**: Shixing Zhao, Zheng Si, Pengpeng Ouyang, Zhengqing Hu, Wanqi Zhu, Dong Chen, Yibo Guo, Mingliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10989)  

**Abstract**: Emergency situations in scheduling systems often trigger local functional failures that undermine system stability and even cause system collapse. Existing methods primarily rely on robust scheduling or reactive scheduling, handling emergencies through predefined rules or rescheduling strategies. However, the diversity and unpredictability of real-world emergencies make them difficult to anticipate, which limits the adaptability of these methods in complex scenarios. Recent studies have shown that Large Language Models (LLMs) possess strong potential for complex scheduling tasks because of their extensive prior knowledge and strong reasoning capabilities. Nevertheless, the high inference latency of LLMs and the lengthy contextual information of scheduling systems significantly hinder their application for emergency handling. To mitigate these issues, we propose the Multi-agent Driven Formal Instruction Generation Framework (MAFIG). The framework constrains the decision scope to local functional modules affected by emergency situations and repairs scheduling logic rapidly by generating formal instructions. MAFIG contains a Perception Agent and an Emergency Decision Agent, which mitigates the adverse impact of lengthy system contexts on emergency decision-making. We further introduce span-focused loss-driven local distillation mechanism (SFL) to transfer the decision-making capability of powerful Cloud Large Language Models (C-LLMs) to lightweight local models, reducing inference latency while preserving decision-making effectiveness. Experiments in the Port, Warehousing, and Deck scheduling datasets show success rates of 98.49\%, 94.97\%, and 97.50\%, with average processing times of 0.33 s, 0.23 s, and 0.19 s. These results demonstrate that MAFIG effectively mitigates the impact of emergencies and improves the robustness and adaptability of scheduling systems. 

---
# WebForge: Breaking the Realism-Reproducibility-Scalability Trilemma in Browser Agent Benchmark 

**Authors**: Peng Yuan, Yuyang Yin, Yuxuan Cai, Zheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.10988)  

**Abstract**: Existing browser agent benchmarks face a fundamental trilemma: real-website benchmarks lack reproducibility due to content drift, controlled environments sacrifice realism by omitting real-web noise, and both require costly manual curation that limits scalability. We present WebForge, the first fully automated framework that resolves this trilemma through a four-agent pipeline -- Plan, Generate, Refine, and Validate -- that produces interactive, self-contained web environments end-to-end without human annotation. A seven-dimensional difficulty control framework structures task design along navigation depth, visual complexity, reasoning difficulty, and more, enabling systematic capability profiling beyond single aggregate scores. Using WebForge, we construct WebForge-Bench, a benchmark of 934 tasks spanning 7 domains and 3 difficulty levels. Multi-model experiments show that difficulty stratification effectively differentiates model capabilities, while cross-domain analysis exposes capability biases invisible to aggregate metrics. Together, these results confirm that multi-dimensional evaluation reveals distinct capability profiles that a single aggregate score cannot capture. Code and benchmark are publicly available at this https URL. 

---
# Back to the Barn with LLAMAs: Evolving Pretrained LLM Backbones in Finetuning Vision Language Models 

**Authors**: Sameera Horawalavithana, Lauren Phillips, Ian Stewart, Sai Munikoti, Karl Pazdernik  

**Link**: [PDF](https://arxiv.org/pdf/2604.10985)  

**Abstract**: Vision-Language Models (VLMs) have rapidly advanced by leveraging powerful pre-trained Large Language Models (LLMs) as core reasoning backbones. As new and more capable LLMs emerge with improved reasoning, instruction-following, and generalization, there is a pressing need to efficiently update existing VLMs to incorporate these advancements. However, the integration of new LLMs into VLMs, particularly how the evolving LLMs contribute to multimodal reasoning, alignment, and task-specific performance remains underexplored. Addressing this gap is important for VLM development, given the rapid evolution of pretrained LLM backbones. This study presents a controlled and systematic investigation of how changes in the pretrained LLM backbone affect downstream VLM task performance. By having the vision encoder, training data, and post-training algorithm remain same across LLAMA-1, LLAMA-2, and LLAMA-3 based VLMs, we find that newer LLM backbones do not always lead to better VLMs, but the performance depends on the downstream VLM task. For example, in visual question and answering tasks, newer LLM backbones tend to solve different questions rather than just more questions, and our analysis shows this is driven by differences in how the models process information, including better calibrated confidence and more stable internal representations. We also find that some VLM capabilities appear only in the newest LLM generation, while tasks that depend mainly on visual understanding see little benefit from a newer LLM backbone. 

---
# ATANT v1.1: Positioning Continuity Evaluation Against Memory, Long-Context, and Agentic-Memory Benchmarks 

**Authors**: Samuel Sameer Tanguturi  

**Link**: [PDF](https://arxiv.org/pdf/2604.10981)  

**Abstract**: ATANT v1.0 (arXiv:2604.06710) defined continuity as a system property with 7 required properties and introduced a 10-checkpoint, LLM-free evaluation methodology validated on a 250-story corpus. Since publication, a recurring reviewer and practitioner question has concerned not the framework itself but its relationship to a wider set of memory evaluations: LOCOMO, LongMemEval, BEAM, MemoryBench, Zep's evaluation suite, Letta/MemGPT's evaluations, and RULER. This companion paper, v1.1, does not modify the v1.0 standard. It closes a related-work gap that v1.0 left brief under page limits. We show by structural analysis that none of these benchmarks measures continuity as defined in v1.0: of the 7 required properties, the median existing eval covers 1 property, the mean covers 0.43 when partial credit is scored at 0.5, and no eval covers more than 2. We provide a cell-by-cell property-coverage matrix, identify methodological defects specific to each benchmark (including an empty-gold scoring bug in the LOCOMO reference implementation that renders 23% of its corpus unscorable by construction), and publish our reference implementation's LOCOMO score (8.8%) alongside the structural reason that number is uninformative about continuity. We publish our 8.8% LOCOMO score alongside our 96% ATANT cumulative-scale score as a calibration pair: the 87-point divergence is evidence that the two benchmarks measure different properties, not that one system is an order of magnitude better than another. The position v1.1 takes is not adversarial: each benchmark measures a real capability. The claim is that none of them can adjudicate continuity, and conflating them with continuity evaluation has led the field to under-invest in the properties v1.0 names. 

---
# CFMS: A Coarse-to-Fine Multimodal Synthesis Framework for Enhanced Tabular Reasoning 

**Authors**: Qixian Huang, Hongqiang Lin, Tong Fu, Yingsen Wang, Zhenghui Fu, Qirui Wang, Yiding Sun, Dongxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10973)  

**Abstract**: Reasoning over tabular data is a crucial capability for tasks like question answering and fact verification, as it requires models to comprehend both free-form questions and semi-structured tables. However, while methods like Chain-of-Thought (CoT) introduce reasoning chains, purely symbolic methodes are inherently limited by their blindness to holistic visual patterns. To address this, we propose the Coarse-to-Fine Multimodal Synthesis framework (CFMS), a novel two-stage paradigm that hierarchically decouples high-level visual perception from granular symbolic reasoning. In the Coarse Stage, CFMS leverages the Multimodal Large Language Models (MLLMs) to perform a one-time synthesis of a multi-perspective knowledge tuple. This tuple subsequently serves as a dynamic reasoning map to guide the fine stage, where a symbolic engine executes a targeted and efficient sequence of iterative operations over the table. Extensive experiments on the WikiTQ and TabFact benchmarks demonstrate that CFMS achieves competitive accuracy. The framework exhibits particular robustness when handling large tables and when instantiated with smaller backbone models, validating its effectiveness and generalizability. 

---
# Delving Aleatoric Uncertainty in Medical Image Segmentation via Vision Foundation Models 

**Authors**: Ruiyang Li, Fang Liu, Licheng Jiao, Xinglin Xie, Jiayao Hao, Shuo Li, Xu Liu, Jingyi Yang, Lingling Li, Puhua Chen, Wenping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.10963)  

**Abstract**: Medical image segmentation supports clinical workflows by precisely delineating anatomical structures and lesions. However, medical image datasets medical image datasets suffer from acquisition noise and annotation ambiguity, causing pervasive data uncertainty that substantially undermines model robustness. Existing research focuses primarily on model architectural improvements and predictive reliability estimation, while systematic exploration of the intrinsic data uncertainty remains insufficient. To address this gap, this work proposes leveraging the universal representation capabilities of visual foundation models to estimate inherent data uncertainty. Specifically, we analyze the feature diversity of the model's decoded representations and quantify their singular value energy to define the semantic perception scale for each class, thereby measuring sample difficulty and aleatoric uncertainty. Based on this foundation, we design two uncertainty-driven application strategies: (1) the aleatoric uncertainty-aware data filtering mechanism to eliminate potentially noisy samples and enhance model learning quality; (2) the dynamic uncertainty-aware optimization strategy that adaptively adjusts class-specific loss weights during training based on the semantic perception scale, combined with a label denoising mechanism to improve training stability. Experimental results on five public datasets encompassing CT and MRI modalities and involving multi-organ and tumor segmentation tasks demonstrate that our method achieves significant and robust performance improvements across various mainstream network architectures, revealing the broad application potential of aleatoric uncertainty in medical image understanding and segmentation tasks. 

---
# RAG-KT: Cross-platform Explainable Knowledge Tracing with Multi-view Fusion Retrieval Generation 

**Authors**: Zhiyi Duan, Hongyu Yuan, Rui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10960)  

**Abstract**: Knowledge Tracing (KT) infers a student's knowledge state from past interactions to predict future performance. Conventional Deep Learning (DL)-based KT models are typically tied to platform-specific identifiers and latent representations, making them hard to transfer and interpret. Large Language Model (LLM)-based methods can be either ungrounded under prompting or overly domain-dependent under fine-tuning. In addition, most existing KT methods are developed and evaluated under a same-distribution assumption. In real deployments, educational data often arise from heterogeneous platforms with substantial distribution shift, which often degrades generalization. To this end, we propose RAG-KT, a retrieval-augmented paradigm that frames cross-platform KT as reliable context constrained inference with LLMs. It builds a unified multi-source structured context with cross-source alignment via Question Group abstractions and retrieves complementary rich and reliable context for each prediction, enabling grounded prediction and interpretable diagnosis. Experiments on three public KT benchmarks demonstrate consistent gains in accuracy and robustness, including strong performance under cross-platform conditions. 

---
# CSPO: Alleviating Reward Ambiguity for Structured Table-to-LaTeX Generation 

**Authors**: Yunfan Yang, Cuiling Lan, Jitao Sang, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10918)  

**Abstract**: Tables contain rich structured information, yet when stored as images their contents remain "locked" within pixels. Converting table images into LaTeX code enables faithful digitization and reuse, but current multimodal large language models (MLLMs) often fail to preserve structural, style, or content fidelity. Conventional post-training with reinforcement learning (RL) typically relies on a single aggregated reward, leading to reward ambiguity that conflates multiple behavioral aspects and hinders effective optimization. We propose Component-Specific Policy Optimization (CSPO), an RL framework that disentangles optimization across LaTeX tables components-structure, style, and content. In particular, CSPO assigns component-specific rewards and backpropagates each signal only through the tokens relevant to its component, alleviating reward ambiguity and enabling targeted component-wise optimization. To comprehensively assess performance, we introduce a set of hierarchical evaluation metrics. Extensive experiments demonstrate the effectiveness of CSPO, underscoring the importance of component-specific optimization for reliable structured generation. 

---
# EvoNash-MARL: A Closed-Loop Multi-Agent Reinforcement Learning Framework for Medium-Horizon Equity Allocation 

**Authors**: Chongliu Jia, Yi Luo, Sipeng Han, Pengwei Li, Jie Ding, Youshuang Hu, Yimiao Qian, Qiya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10911)  

**Abstract**: Medium-to-long-horizon stock allocation presents significant challenges due toveak predictive structures, non-stadonary market regimes, and the degradationf signals following the application of transaction costs, capacity limits, and tail-isk constraints. Conventional approaches commonly rely on a single predictor orloosely coupled prediction-to-allocation pipeline, limiting robustness underThis work addresses a targeted design question: whetherlistribution shift. 1coupling reinforcement learning (RL), multi-agent policy populations, Policy-Space Response Oracle (PSRO)-style aggregation, league best-response trainingevolutionary replacement, and execution-aware checkpoint selection within ainified walk-forward loop improves allocator robustness at medium to longhorizons. The proposed framework, EvoNash-MARL, integrates these componentswithin an execution-aware allocation loop and further introduces a layeredpolicy architecture comprising a direction head and a risk head, nonlinear signalenhancement, feature-quality reweighting, and constraint-aware checkpointselection. Under a 120-window walk-forward protocol, the resolved v21configuration achieves mean excess Sharpe 0.7600 and robust score -0.0203,anking first among internal controls; on aligned daily out-of-sample returnsrom 2014-01-02 to 2024-01-05, it delivers 19.6% annualized return versus 11.7% for SPY, and in an extended walk-forward evaluation through 2026-02-10 it delivers 20.5% rersus 13.5%. The framework maintains positive performance under realistictress constraints and exhibits structured cross-market generalization; however,lobal strong significance under White's Reality Check (WRC) and SPA-lite testingestablished. Therefore, the results are presented as evidence supporting asnotnore stable medium-to long-horizon training and selection paradigm, ratherhan as prooffof universally superior market-timing performance. 

---
# Reasoning as Data: Representation-Computation Unity and Its Implementation in a Domain-Algebraic Inference Engine 

**Authors**: Chao Li, Yuru Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10908)  

**Abstract**: Every existing knowledge system separates storage from computation. We show this separation is unnecessary and eliminate it. In a standard triple is_a(Apple, Company), domain context lives in the query or the programmer's mind. In a CDC four-tuple is_a(Apple, Company, @Business), domain becomes a structural field embedded in predicate arity. Any system respecting arity automatically performs domain-scoped inference without external rules. We call this representation-computation unity (RCU). From the four-tuple structure, three inference mechanisms emerge: domain-scoped closure, typed inheritance, and write-time falsification via cycle detection per domain fiber. We establish RCU formally via four theorems. RCU is implementable. We present a working symbolic engine (2400 lines Python+Prolog) resolving four engineering issues: rule-data separation, shared-fiber handling, read-only meta-layer design, and intersective convergence. A central result: CDC domain-constrained inference is distinct from Prolog with a domain argument. Two case studies validate the engine. ICD-11 classification (1247 entities, 3 axes) shows fibers resolve multiple inheritance. CBT clinical reasoning shows generalization to temporal reasoning with session turn as ordered domain index. Multi-constraint queries realize CSP arc-consistency with complexity O(m (N/K)^2), confirming the domain lattice's sparsity governs performance. When domain is structural, data computes itself. 

---
# CASK: Core-Aware Selective KV Compression for Reasoning Traces 

**Authors**: Buseong Kim, Heejun Gwon  

**Link**: [PDF](https://arxiv.org/pdf/2604.10900)  

**Abstract**: In large language models performing long-form reasoning, the KV cache grows rapidly with decode length, creating bottlenecks in memory and inference stability. Existing reasoning-oriented KV compression has mostly followed an eviction-centered view: estimate token importance more accurately, then discard lower-ranked entries. Our analysis suggests that scorer refinement alone often fails to substantially reorganize the actual keep-set and may therefore not be the main lever for preserving reasoning behavior. We instead frame reasoning KV compression as a behavior-preserving structured consolidation problem. CASK partitions the decode-time reasoning trace into a protected core that anchors answer formation and intermediate state, and mergeable scratch with high redundancy. The core is preserved, while selective consolidation is applied only to the scratch. To address prompt-heavy regimes where the prefix can exhaust the budget before decode-stage compression becomes active, CASK further uses a two-stage design: prefix eviction followed by decode-stage consolidation. On the H100 reasoning gate, CASK shows higher full-KV continuation fidelity than TriAttention at matched budgets on both AIME24 and AIME25, with recurring cask@384 > triattention@512 crossings. In prompt-heavy replay, multi_news and vcsum act as decode-active witnesses, while qmsum and gov_report expose the prefix_budget_exhausted boundary. The overall evidence supports a simple conclusion: effective reasoning KV compression depends less on more elaborate scorer engineering than on combining core preservation with selective scratch consolidation to lower the usable budget frontier. 

---
# ZoomR: Memory Efficient Reasoning through Multi-Granularity Key Value Retrieval 

**Authors**: David H. Yang, Yuxuan Zhu, Mohammad Mohammadi Amiri, Keerthiram Murugesan, Tejaswini Pedapati, Subhajit Chaudhury, Pin-Yu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.10898)  

**Abstract**: Large language models (LLMs) have shown great performance on complex reasoning tasks but often require generating long intermediate thoughts before reaching a final answer. During generation, LLMs rely on a key-value (KV) cache for autoregressive decoding. However, the memory footprint of the KV cache grows with output length. Prior work on KV cache optimization mostly focus on compressing the long input context, while retaining the full KV cache for decoding. For tasks requiring long output generation, this leads to increased computational and memory costs. In this paper, we introduce ZoomR, a novel approach that enables LLMs to adaptively compress verbose reasoning thoughts into summaries and uses a dynamic KV cache selection policy that leverages these summaries while also strategically "zooming in" on fine-grained details. By using summary keys as a coarse-grained index during decoding, ZoomR uses the query to retrieve details for only the most important thoughts. This hierarchical strategy significantly reduces memory usage by avoiding full-cache attention at each step. Experiments across math and reasoning tasks show that our approach achieves competitive performance compared to baselines, while reducing inference memory requirements by more than $4\times$. These results demonstrate that a multi-granularity KV selection enables more memory efficient decoding, especially for long output generation. 

---
# A Quantitative Definition of Intelligence 

**Authors**: Kang-Sin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.10873)  

**Abstract**: We propose an operational, quantitative definition of intelligence for arbitrary physical systems. The intelligence density of a system is the ratio of the logarithm of its independent outputs to its total description length. A system memorizes if its description length grows with its output count; it knows if its description length remains fixed while its output count diverges. The criterion for knowing is generalization: a system knows its domain if a single finite mechanism can produce correct outputs across an unbounded range of inputs, rather than storing each answer individually. We argue that meaning over a domain is a selection and ordering of functions that produces correct outputs, and that a system whose intelligence density diverges necessarily captures this structure. The definition (1) places intelligence on a substrate-independent continuum from logic gates to brains, (2) blocks Putnam's pancomputationalist triviality argument via an independence condition on outputs, and (3) resolves Searle's Chinese Room Argument by showing that any finite rulebook handling an infinite domain must generalize. 

---
# Beyond Statistical Co-occurrence: Unlocking Intrinsic Semantics for Tabular Data Clustering 

**Authors**: Mingjie Zhao, Yunfan Zhang, Yiqun Zhang, Yiu-ming Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2604.10865)  

**Abstract**: Deep Clustering (DC) has emerged as a powerful tool for tabular data analysis in real-world domains like finance and healthcare. However, most existing methods rely on data-level statistical co-occurrence to infer the latent metric space, often overlooking the intrinsic semantic knowledge encapsulated in feature names and values. As a result, semantically related concepts like `Flu' and `Cold' are often treated as symbolic tokens, causing conceptually related samples to be isolated. To bridge the gap between dataset-specific statistics and intrinsic semantic knowledge, this paper proposes Tabular-Augmented Contrastive Clustering (TagCC), a novel framework that anchors statistical tabular representations to open-world textual concepts. Specifically, TagCC utilizes Large Language Models (LLMs) to distill underlying data semantics into textual anchors via semantic-aware transformation. Through Contrastive Learning (CL), the framework enriches the statistical tabular representations with the open-world semantics encapsulated in these anchors. This CL framework is jointly optimized with a clustering objective, ensuring that the learned representations are both semantically coherent and clustering-friendly. Extensive experiments on benchmark datasets demonstrate that TagCC significantly outperforms its counterparts. 

---
# A Benchmark for Gap and Overlap Analysis as a Test of KG Task Readiness 

**Authors**: Maruf Ahmed Mridul, Rohit Kapa, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2604.10853)  

**Abstract**: Task-oriented evaluation of knowledge graph (KG) quality increasingly asks whether an ontology-based representation can answer the competency questions that users actually care about, in a manner that is reproducible, explainable, and traceable to evidence. This paper adopts that perspective and focuses on gap and overlap analysis for policy-like documents (e.g., insurance contracts), where given a scenario, which documents support it (overlap) and which do not (gap), with defensible justifications. The resulting gap/overlap determinations are typically driven by genuine differences in coverage and restrictions rather than missing data, making the task a direct test of KG task readiness rather than a test of missing facts or query expressiveness. We present an executable and auditable benchmark that aligns natural-language contract text with a formal ontology and evidence-linked ground truth, enabling systematic comparison of methods. The benchmark includes: (i) ten simplified yet diverse life-insurance contracts reviewed by a domain expert, (ii) a domain ontology (TBox) with an instantiated knowledge base (ABox) populated from contract facts, and (iii) 58 structured scenarios paired with SPARQL queries with contract-level outcomes and clause-level excerpts that justify each label. Using this resource, we compare a text-only LLM baseline that infers outcomes directly from contract text against an ontology-driven pipeline that answers the same scenarios over the instantiated KG, demonstrating that explicit modeling improves consistency and diagnosis for gap/overlap analyses. Although demonstrated for gap and overlap analysis, the benchmark is intended as a reusable template for evaluating KG quality and supporting downstream work such as ontology learning, KG population, and evidence-grounded question answering. 

---
# Your Model Diversity, Not Method, Determines Reasoning Strategy 

**Authors**: Moulik Choraria, Argyrios Gerogiannis, Anirban Das, Supriyo Chakraborty, Berkcan Kapusuzoglu, Chia-Hsuan Lee, Kartik Balasubramaniam, Shi-Xiong Zhang, Sambit Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10827)  

**Abstract**: Compute scaling for LLM reasoning requires allocating budget between exploring solution approaches ($breadth$) and refining promising solutions ($depth$). Most methods implicitly trade off one for the other, yet why a given trade-off works remains unclear, and validation on a single model obscures the role of the model itself. We argue that $\textbf{the optimal strategy depends on the model's diversity profile, the spread of probability mass across solution approaches, and that this must be characterized before any exploration strategy is adopted.}$ We formalize this through a theoretical framework decomposing reasoning uncertainty and derive conditions under which tree-style depth refinement outperforms parallel sampling. We validate it on Qwen-3 4B and Olmo-3 7B families, showing that lightweight signals suffice for depth-based refinement on low-diversity aligned models while yielding limited utility for high-diversity base models, which we hypothesize require stronger compensation for lower exploration coverage. 

---
# CheeseBench: Evaluating Large Language Models on Rodent Behavioral Neuroscience Paradigms 

**Authors**: Zacharie Bugaud  

**Link**: [PDF](https://arxiv.org/pdf/2604.10825)  

**Abstract**: We introduce CheeseBench, a benchmark that evaluates large language models (LLMs) on nine classical behavioral neuroscience paradigms (Morris water maze, Barnes maze, T-maze, radial arm maze, star maze, operant chamber, shuttle box, conditioned place preference, and delayed non-match to sample), spanning six cognitive dimensions. Each task is grounded in peer-reviewed rodent protocols with approximate animal baselines. The agent receives a unified system prompt with no task-specific instructions and must discover goals purely from ASCII text observations and reward signals, much like a rodent placed into an unfamiliar apparatus. We evaluate six open-weight LLMs (3B to 72B parameters) on text-based ASCII renderings and compare against both a random baseline and a graph-based reinforcement learning agent. Our best model (Qwen2.5-VL-7B) reaches 52.6% average success on ASCII input, compared to 32.1% for random agents and 78.9% for approximate rodent baselines. We find that (1) scaling beyond 7B yields diminishing returns, (2) longer context history degrades performance, (3) chain-of-thought prompting hurts rather than helps, and (4) a vision-language architecture provides an advantage at 7B but hurts at 32B. Because the same model's performance ranges from 20% to 57% depending on interface parameters alone, these results characterize the agent-plus-interface system, not the model in isolation. Under this unified zero-shot ASCII protocol, current open-weight LLM agents remain well below approximate rodent reference values, particularly on tasks requiring spatial navigation and within-trial state tracking. 

---
# TorchUMM: A Unified Multimodal Model Codebase for Evaluation, Analysis, and Post-training 

**Authors**: Yinyi Luo, Wenwen Wang, Hayes Bai, Hongyu Zhu, Hao Chen, Pan He, Marios Savvides, Sharon Li, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10784)  

**Abstract**: Recent advances in unified multimodal models (UMMs) have led to a proliferation of architectures capable of understanding, generating, and editing across visual and textual modalities. However, developing a unified framework for UMMs remains challenging due to the diversity of model architectures and the heterogeneity of training paradigms and implementation details. In this paper, we present TorchUMM, the first unified codebase for comprehensive evaluation, analysis, and post-training across diverse UMM backbones, tasks, and datasets. TorchUMM supports a broad spectrum of models covering a wide range of scales and design paradigms. Our benchmark encompasses three core task dimensions: multimodal understanding, generation, and editing, and integrates both established and novel datasets to evaluate perception, reasoning, compositionality, and instruction-following abilities. By providing a unified interface and standardized evaluation protocols, TorchUMM enables fair and reproducible comparisons across heterogeneous models and fosters deeper insights into their strengths and limitations, facilitating the development of more capable unified multimodal systems. Code is available at: this https URL. 

---
# Learning Preference-Based Objectives from Clinical Narratives for Sequential Treatment Decision-Making 

**Authors**: Daniel J. Tan, Kay Choong See, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.10783)  

**Abstract**: Designing reward functions remains a central challenge in reinforcement learning (RL) for healthcare, where outcomes are sparse, delayed, and difficult to specify. While structured data capture physiological states, they often fail to reflect the overall quality of a patient's clinical trajectory, including recovery dynamics, treatment burden, and stability. Clinical narratives, in contrast, summarize longitudinal reasoning and implicitly encode evaluations of treatment effectiveness. We propose Clinical Narrative-informed Preference Rewards (CN-PR), a framework for learning reward functions directly from discharge summaries by treating them as scalable supervision for trajectory-level preferences. Using a large language model, we derive trajectory quality scores (TQS) and construct pairwise preferences over patient trajectories, enabling reward learning via a structured preference-based objective. To account for variability in narrative informativeness, we incorporate a confidence signal that weights supervision based on its relevance to the decision-making task. The learned reward aligns strongly with trajectory quality (Spearman rho = 0.63) and enables policies that are consistently associated with improved recovery-related outcomes, including increased organ support-free days and faster shock resolution, while maintaining comparable performance on mortality. These effects persist under external validation. Our results demonstrate that narrative-derived supervision provides a scalable and expressive alternative to handcrafted or outcome-based reward design for dynamic treatment regimes. 

---
# When More Thinking Hurts: Overthinking in LLM Test-Time Compute Scaling 

**Authors**: Shu Zhou, Rui Ling, Junan Chen, Xin Wang, Tao Fan, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10739)  

**Abstract**: Scaling test-time compute through extended chains of thought has become a dominant paradigm for improving large language model reasoning. However, existing research implicitly assumes that longer thinking always yields better results. This assumption remains largely unexamined. We systematically investigate how the marginal utility of additional reasoning tokens changes as compute budgets increase. We find that marginal returns diminish substantially at higher budgets and that models exhibit ``overthinking'', where extended reasoning is associated with abandoning previously correct answers. Furthermore, we show that optimal thinking length varies across problem difficulty, suggesting that uniform compute allocation is suboptimal. Our cost-aware evaluation framework reveals that stopping at moderate budgets can reduce computation significantly while maintaining comparable accuracy. 

---
# Teaching Language Models How to Code Like Learners: Conversational Serialization for Student Simulation 

**Authors**: Charles Koutcheme, Arto Hellas, Juho Leinonen  

**Link**: [PDF](https://arxiv.org/pdf/2604.10720)  

**Abstract**: Artificial models that simulate how learners act and respond within educational systems are a promising tool for evaluating tutoring strategies and feedback mechanisms at scale. However, many existing approaches in programming education rely on prompting large, proprietary language models, raising concerns around privacy, cost, and dependence. In this work, we propose a method for training open-weight artificial programming learners using authentic student process data. Our approach serializes temporal log traces into a conversational format, representing each student's problem-solving process as a dialogue between the learner and their automated assessment system. Student code submissions and environment feedback, such as test outcomes, grades, and error traces, form alternating conversational turns, enabling models to learn from the iterative debugging process. We additionally introduce a training pipeline combining supervised fine-tuning with preference optimization to align models with authentic student debugging behavior. We evaluate our framework by training Qwen models at 4B and 8B scales on a large-scale dataset of real student submissions to Python programming assignments. Our results show that incorporating environment feedback strengthens the models' ability to replicate student debugging behavior, improving over both prior code-only approaches and prompted large language models baselines in functional alignment and code similarity. We release our code to support reproducibility. 

---
# SciPredict: Can LLMs Predict the Outcomes of Scientific Experiments in Natural Sciences? 

**Authors**: Udari Madhushani Sehwag, Elaine Lau, Haniyeh Ehsani Oskouie, Shayan Shabihi, Erich Liang, Andrea Toledo, Guillermo Mangialardi, Sergio Fonrouge, Ed-Yeremai Hernandez Cardona, Paula Vergara, Utkarsh Tyagi, Chen Bo Calvin Zhang, Pavi Bhatter, Nicholas Johnson, Furong Huang, Ernesto Gabriel Hernandez Montoya, Bing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10718)  

**Abstract**: Accelerating scientific discovery requires the identification of which experiments would yield the best outcomes before committing resources to costly physical validation. While existing benchmarks evaluate LLMs on scientific knowledge and reasoning, their ability to predict experimental outcomes - a task where AI could significantly exceed human capabilities - remains largely underexplored. We introduce SciPredict, a benchmark comprising 405 tasks derived from recent empirical studies in 33 specialized sub-fields of physics, biology, and chemistry. SciPredict addresses two critical questions: (a) can LLMs predict the outcome of scientific experiments with sufficient accuracy? and (b) can such predictions be reliably used in the scientific research process? Evaluations reveal fundamental limitations on both fronts. Model accuracies are 14-26% and human expert performance is $\approx$20%. Although some frontier models exceed human performance model accuracy is still far below what would enable reliable experimental guidance. Even within the limited performance, models fail to distinguish reliable predictions from unreliable ones, achieving only $\approx$20% accuracy regardless of their confidence or whether they judge outcomes as predictable without physical experimentation. Human experts, in contrast, demonstrate strong calibration: their accuracy increases from $\approx$5% to $\approx$80% as they deem outcomes more predictable without conducting the experiment. SciPredict establishes a rigorous framework demonstrating that superhuman performance in experimental science requires not just better predictions, but better awareness of prediction reliability. For reproducibility all our data and code are provided at this https URL 

---
# Camyla: Scaling Autonomous Research in Medical Image Segmentation 

**Authors**: Yifan Gao, Haoyue Li, Feng Yuan, Xin Gao, Weiran Huang, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10696)  

**Abstract**: We present Camyla, a system for fully autonomous research within the scientific domain of medical image segmentation. Camyla transforms raw datasets into literature-grounded research proposals, executable experiments, and complete manuscripts without human intervention. Autonomous experimentation over long horizons poses three interrelated challenges: search effort drifts toward unpromising directions, knowledge from earlier trials degrades as context accumulates, and recovery from failures collapses into repetitive incremental fixes. To address these challenges, the system combines three coupled mechanisms: Quality-Weighted Branch Exploration for allocating effort across competing proposals, Layered Reflective Memory for retaining and compressing cross-trial knowledge at multiple granularities, and Divergent Diagnostic Feedback for diversifying recovery after underperforming trials. The system is evaluated on CamylaBench, a contamination-free benchmark of 31 datasets constructed exclusively from 2025 publications, under a strict zero-intervention protocol across two independent runs within a total of 28 days on an 8-GPU cluster. Across the two runs, Camyla generates more than 2,700 novel model implementations and 40 complete manuscripts, and surpasses the strongest per-dataset baseline selected from 14 established architectures, including nnU-Net, on 22 and 18 of 31 datasets under identical training budgets, respectively (union: 24/31). Senior human reviewers score the generated manuscripts at the T1/T2 boundary of contemporary medical imaging journals. Relative to automated baselines, Camyla outperforms AutoML and NAS systems on aggregate segmentation performance and exceeds six open-ended research agents on both task completion and baseline-surpassing frequency. These results suggest that domain-scale autonomous research is achievable in medical image segmentation. 

---
# FACT-E: Causality-Inspired Evaluation for Trustworthy Chain-of-Thought Reasoning 

**Authors**: Yuxi Sun, Aoqi Zuo, Haotian Xie, Wei Gao, Mingming Gong, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.10693)  

**Abstract**: Chain-of-Thought (CoT) prompting has improved LLM reasoning, but models often generate explanations that appear coherent while containing unfaithful intermediate steps. Existing self-evaluation approaches are prone to inherent biases: the model may confidently endorse coherence even when the step-to-step implication is not valid, leading to unreliable faithfulness evaluation. We propose FACT-E, a causality-inspired framework for evaluating CoT quality. FACT-E uses controlled perturbations as an instrumental signal to separate genuine step-to-step dependence from bias-driven artifacts, producing more reliable faithfulness estimates (\textit{intra-chain faithfulness}). To select trustworthy trajectories, FACT-E jointly considers \textit{intra-chain faithfulness} and \textit{CoT-to-answer consistency}, ensuring that selected chains are both faithful internally and supportive of the correct final answer. Experiments on GSM8K, MATH, and CommonsenseQA show that FACT-E improves reasoning-trajectory selection and yields stronger in-context learning exemplars. FACT-E also reliably detects flawed reasoning under noisy conditions, providing a robust metric for trustworthy LLM reasoning. 

---
# Do LLMs Build Spatial World Models? Evidence from Grid-World Maze Tasks 

**Authors**: Weijiang Li, Yilin Zhu, Rajarshi Das, Parijat Dube  

**Link**: [PDF](https://arxiv.org/pdf/2604.10690)  

**Abstract**: Foundation models have shown remarkable performance across diverse tasks, yet their ability to construct internal spatial world models for reasoning and planning remains unclear. We systematically evaluate the spatial understanding of large language models through maze tasks, a controlled testing context requiring multi-step planning and spatial abstraction. Across comprehensive experiments with Gemini-2.5-Flash, GPT-5-mini, Claude-Haiku-4.5, and DeepSeek-Chat, we uncover significant discrepancies in spatial reasoning that challenge assumptions about LLM planning capabilities. Using chain-of-thought prompting, Gemini achieves 80-86% accuracy on smaller mazes (5x5 to 7x7 grids) with tokenized adjacency representations, but performance collapses to 16-34% with visual grid formats, which is a 2-5x difference, suggesting representation-dependent rather than format-invariant spatial reasoning. We further probe spatial understanding through sequential proximity questions and compositional distance comparisons. Despite achieving 96-99% semantic coverage in reasoning traces, models fail to leverage this understanding for consistent spatial computations, indicating that they treat each question independently rather than building cumulative spatial knowledge. Our findings based on the maze-solving tasks suggest that LLMs do not develop robust spatial world models, but rather exhibit representation-specific and prompting-dependent reasoning that succeeds only under narrow conditions. These results have critical implications for deploying foundation models in applications requiring spatial abstraction. 

---
# FedRio: Personalized Federated Social Bot Detection via Cooperative Reinforced Contrastive Adversarial Distillation 

**Authors**: Yingguang Yang, Hao Liu, Xin Zhang, Yunhui Liu, Yutong Xia, Qi Wu, Hao Peng, Taoran Liang, Bin Chong, Tieke He, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10678)  

**Abstract**: Social bot detection is critical to the stability and security of online social platforms. However, current state-of-the-art bot detection models are largely developed in isolation, overlooking the benefits of leveraging shared detection patterns across platforms to improve performance and promptly identify emerging bot variants. The heterogeneity of data distributions and model architectures further complicates the design of an effective cross-platform and cross-model detection framework. To address these challenges, we propose FedRio (Personalized Federated Social Bot Detection with Cooperative Reinforced Contrastive Adversarial Distillation framework. We first introduce an adaptive message-passing module as the graph neural network backbone for each client. To facilitate efficient knowledge sharing of global data distributions, we design a federated knowledge extraction mechanism based on generative adversarial networks. Additionally, we employ a multi-stage adversarial contrastive learning strategy to enforce feature space consistency among clients and reduce divergence between local and global models. Finally, we adopt adaptive server-side parameter aggregation and reinforcement learning-based client-side parameter control to better accommodate data heterogeneity in heterogeneous federated settings. Extensive experiments on two real-world social bot detection benchmarks demonstrate that FedRio consistently outperforms state-of-the-art federated learning baselines in detection accuracy, communication efficiency, and feature space consistency, while remaining competitive with published centralized results under substantially stronger privacy constraints. 

---
# Principles Do Not Apply Themselves: A Hermeneutic Perspective on AI Alignment 

**Authors**: Behrooz Razeghi  

**Link**: [PDF](https://arxiv.org/pdf/2604.10673)  

**Abstract**: AI alignment is often framed as the task of ensuring that an AI system follows a set of stated principles or human preferences, but general principles rarely determine their own application in concrete cases. When principles conflict, when they are too broad to settle a situation, or when the relevant facts are unclear, an additional act of judgment is required. This paper analyzes that step through the lens of hermeneutics and argues that alignment therefore includes an interpretive component: it involves context-sensitive judgments about how principles should be read, applied, and prioritized in practice. We connect this claim to recent empirical findings showing that a substantial portion of preference-labeling data falls into cases of principle conflict or indifference, where the principle set does not uniquely determine a decision. We then draw an operational consequence: because such judgments are expressed in behavior, many alignment-relevant choices appear only in the distribution of responses a model generates at deployment time. To formalize this point, we distinguish deployment-induced and corpus-induced evaluation and show that off-policy audits can fail to capture alignment-relevant failures when the two response distributions differ. We argue that principle-specified alignment includes a context-dependent interpretive component. 

---
# Preference-Agile Multi-Objective Optimization for Real-time Vehicle Dispatching 

**Authors**: Jiahuan Jin, Wenhao Zhao, Rong Qu, Jianfeng Ren, Xinan Chen, Qingfu Zhang, Ruibin Bai  

**Link**: [PDF](https://arxiv.org/pdf/2604.10664)  

**Abstract**: Multi-objective optimization (MOO) has been widely studied in literature because of its versatility in human-centered decision making in real-life applications. Recently, demand for dynamic MOO is fast-emerging due to tough market dynamics that require real-time re-adjustments of priorities for different objectives. However, most existing studies focus either on deterministic MOO problems which are not practical, or non-sequential dynamic MOO decision problems that cannot deal with some real-life complexities. To address these challenges, a preference-agile multi-objective optimization (PAMOO) is proposed in this paper to permit users to dynamically adjust and interactively assign the preferences on the fly. To achieve this, a novel uniform model within a deep reinforcement learning (DRL) framework is proposed that can take as inputs users' dynamic preference vectors explicitly. Additionally, a calibration function is fitted to ensure high quality alignment between the preference vector inputs and the output DRL decision policy. Extensive experiments on challenging real-life vehicle dispatching problems at a container terminal showed that PAMOO obtains superior performance and generalization ability when compared with two most popular MOO methods. Our method presents the first dynamic MOO method for challenging \rev{dynamic sequential MOO decision problems 

---
# Governed Reasoning for Institutional AI 

**Authors**: Mamadou Seck  

**Link**: [PDF](https://arxiv.org/pdf/2604.10658)  

**Abstract**: Institutional decisions -- regulatory compliance, clinical triage, prior authorization appeal -- require a different AI architecture than general-purpose agents provide. Agent frameworks infer authority conversationally, reconstruct accountability from logs, and produce silent errors: incorrect determinations that execute without any human review signal. We propose Cognitive Core: a governed decision substrate built from nine typed cognitive primitives (retrieve, classify, investigate, verify, challenge, reflect, deliberate, govern, generate), a four-tier governance model where human review is a condition of execution rather than a post-hoc check, a tamper-evident SHA-256 hash-chain audit ledger endogenous to computation, and a demand-driven delegation architecture supporting both declared and autonomously reasoned epistemic sequences.
We benchmark three systems on an 11-case balanced prior authorization appeal evaluation set. Cognitive Core achieves 91% accuracy against 55% (ReAct) and 45% (Plan-and-Solve). The governance result is more significant: CC produced zero silent errors while both baselines produced 5-6. We introduce governability -- how reliably a system knows when it should not act autonomously -- as a primary evaluation axis for institutional AI alongside accuracy. The baselines are implemented as prompts, representing the realistic deployment alternative to a governed framework. A configuration-driven domain model means deploying a new institutional decision domain requires YAML configuration, not engineering capacity. 

---
# Enhancing Cross-Problem Vehicle Routing via Federated Learning 

**Authors**: Xiangchi Meng, Jianan Zhou, Jie Gao, Yifan Lu, Yaoxin Wu, Gonglin Yuan, Yaqing Hou  

**Link**: [PDF](https://arxiv.org/pdf/2604.10652)  

**Abstract**: Vehicle routing problems (VRPs) constitute a core optimization challenge in modern logistics and supply chain management. The recent neural combinatorial optimization (NCO) has demonstrated superior efficiency over some traditional algorithms. While serving as a primary NCO approach for solving general VRPs, current cross-problem learning paradigms are still subject to performance degradation and generalizability decay, when transferring from simple VRP variants to those involving different and complex constraints. To strengthen the paradigms, this paper offers an innovative "Multi-problem Pre-train, then Single-problem Fine-tune" framework with Federated Learning (MPSF-FL). This framework exploits the common knowledge of a federated global model to foster efficient cross-problem knowledge sharing and transfer among local models for single-problem fine-tuning. In this way, local models effectively retain common VRP knowledge from up-to-date global model, while being efficiently adapted to downstream VRPs with heterogeneous complex constraints. Experimental results demonstrate that our framework not only enhances the performance in diverse VRPs, but also improves the generalizability in unseen problems. 

---
# Working Paper: Towards Schema-based Learning from a Category-Theoretic Perspective 

**Authors**: Pablo de los Riscos, Fernando J. Corbacho, Michael A. Arbib  

**Link**: [PDF](https://arxiv.org/pdf/2604.10589)  

**Abstract**: We introduce a hierarchical categorical framework for Schema-Based Learning (SBL) structured across four interconnected levels. At the schema level, a free multicategory $Sch_{syn}$ encodes fundamental schemas and transformations. An implementation functor $\mathcal{I}$ maps syntactic schemas to representational languages, inducing via the Grothendieck construction the total category $Sch_{impl}$. Implemented schemas are mapped by a functor $Model$ into the Kleisli category $\mathbf{KL(G)}$ of the Giry monad, yielding probabilistic models, while an instances presheaf assigns evaluated instance spaces. A semantic category $Sch_{sem}$, defined as a full subcategory of $\mathbf{KL(G)}$, provides semantic grounding through an interpretation functor from $Sch_{impl}$.
At the agent level, $Sch_{impl}$ is equipped with a duoidal structure $\mathcal{O}_{Sch}$ supporting schema-based workflows. A left duoidal action on the category $Mind$ enables workflow execution over mental objects, whose components include mental spaces, predictive models, and a cognitive kernel composed of memory and cognitive modules. Each module is specified by schema-typed interfaces, duoidal workflows, a success condition, and a logical signature. Memory is formalized categorically via memory subsystems, a presheaf $Data_M$, a monoidal operation category $Ops_M$, and read/write natural transformations. Together with the $Body$ category, Mind defines the embodied SBL agent.
At higher levels, SBL is represented as an object of the agent architecture category $ArchCat$, enabling comparison with heterogeneous paradigms, while the $World$ category models multi-agent and agent-environment interactions. Altogether, the framework forms a weak hierarchical $n$-categorical structure linking schema semantics, cognition, embodiment, architectural abstraction, and world-level interaction. 

---
# Failure Ontology: A Lifelong Learning Framework for Blind Spot Detection and Resilience Design 

**Authors**: Yuan Sun, Hong Yi, Jinyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10549)  

**Abstract**: Personalized learning systems are almost universally designed around a single objective: help people acquire knowledge and skills more efficiently. We argue this framing misses the more consequential problem. The most damaging failures in human life-financial ruin, health collapse, professional obsolescence-are rarely caused by insufficient knowledge acquisition. They arise from the systematic absence of entire conceptual territories from a person's cognitive map: domains they never thought to explore because, from within their existing worldview, those domains did not appear to exist or to matter. We call such absences Ontological Blind Spots and introduce Failure Ontology (F), a formal framework for detecting, classifying, and remediating them across a human lifetime. The framework introduces three original contributions: (1) a four-type taxonomy of blind spots distinguishing domain blindness, structural blindness, weight blindness, and temporal blindness; (2) five convergent failure patterns characterizing how blind spots interact with external disruption to produce catastrophic outcomes; and (3) the Failure Learning Efficiency Theorem, proving that failure-based learning achieves higher sample efficiency than success-based learning under bounded historical data. We illustrate the framework through historical case analysis of the 1997 Asian Financial Crisis and the 2008 subprime mortgage crisis, and through alongitudinal individual case study spanning five life stages. 

---
# Agent^2 RL-Bench: Can LLM Agents Engineer Agentic RL Post-Training? 

**Authors**: Wanyi Chen, Xiao Yang, Xu Yang, Tianming Sha, Qizheng Li, Zhuo Wang, Bowen Xian, Fang Kong, Weiqing Liu, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2604.10547)  

**Abstract**: We introduce Agent^2 RL-Bench, a benchmark for evaluating agentic RL post-training -- whether LLM agents can autonomously design, implement, and run complete RL pipelines that improve foundation models. This capability is important because RL post-training increasingly drives model alignment and specialization, yet existing benchmarks remain largely static: supervised fine-tuning alone yields strong results, leaving interactive RL engineering untested. Agent^2 RL-Bench addresses this with six tasks across three levels -- from static rule-based training to closed-loop online RL with trajectory collection -- each adding a structural requirement that prior levels do not impose. The benchmark provides isolated workspaces with a grading API, runtime instrumentation that records every submission and code revision, and automated post-hoc analysis that generates structured run reports, enabling the first automated diagnostic of agent-driven post-training behavior. Across multiple agent stacks spanning five agent systems and six driver LLMs, we find that agents achieve striking interactive gains -- on ALFWorld, an RL-only agent improves from 5.97 to 93.28 via SFT warm-up and GRPO with online rollouts -- yet make only marginal progress on others (DeepSearchQA: +2.75 within evaluation noise), and that driver choice has a large effect on interactive tasks -- within the same scaffold, switching drivers changes interactive improvement from near-zero to +78pp. More broadly, the benchmark reveals that supervised pipelines dominate agent-driven post-training under fixed budgets, with online RL succeeding as the final best route only on ALFWorld. Code is available at this https URL. 

---
# From Perception to Planning: Evolving Ego-Centric Task-Oriented Spatiotemporal Reasoning via Curriculum Learning 

**Authors**: Xiaoda Yang, Yuxiang Liu, Shenzhou Gao, Can Wang, Jingyang Xue, Lixin Yang, Yao Mu, Tao Jin, Shuicheng Yan, Zhimeng Zhang, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.10517)  

**Abstract**: Modern vision-language models achieve strong performance in static perception, but remain limited in the complex spatiotemporal reasoning required for embodied, egocentric tasks. A major source of failure is their reliance on temporal priors learned from passive video data, which often leads to spatiotemporal hallucinations and poor generalization in dynamic environments. To address this, we present EgoTSR, a curriculum-based framework for learning task-oriented spatiotemporal reasoning. EgoTSR is built on the premise that embodied reasoning should evolve from explicit spatial understanding to internalized task-state assessment and finally to long-horizon planning. To support this paradigm, we construct EgoTSR-Data, a large-scale dataset comprising 46 million samples organized into three stages: Chain-of-Thought (CoT) supervision, weakly supervised tagging, and long-horizon sequences. Extensive experiments demonstrate that EgoTSR effectively eliminates chronological biases, achieving 92.4% accuracy on long-horizon logical reasoning tasks while maintaining high fine-grained perceptual precision, significantly outperforming existing open-source and closed-source state-of-the-art models. 

---
# Agent Mentor: Framing Agent Knowledge through Semantic Trajectory Analysis 

**Authors**: Roi Ben-Gigi, Yuval David, Fabiana Fournier, Lior Limonad, Dany Moshkovich, Hadar Mulian, Segev Shlomov  

**Link**: [PDF](https://arxiv.org/pdf/2604.10513)  

**Abstract**: AI agent development relies heavily on natural language prompting to define agents' tasks, knowledge, and goals. These prompts are interpreted by Large Language Models (LLMs), which govern agent behavior. Consequently, agentic performance is susceptible to variability arising from imprecise or ambiguous prompt formulations. Identifying and correcting such issues requires examining not only the agent's code, but also the internal system prompts generated throughout its execution lifecycle, as reflected in execution logs.
In this work, we introduce an analytics pipeline implemented as part of the Agent Mentor open-source library that monitors and incrementally adapts the system prompts defining another agent's behavior. The pipeline improves performance by systematically injecting corrective instructions into the agent's knowledge. We describe its underlying mechanism, with particular emphasis on identifying semantic features associated with undesired behaviors and using them to derive corrective statements.
We evaluate the proposed pipeline across three exemplar agent configurations and benchmark tasks using repeated execution runs to assess effectiveness. These experiments provide an initial exploration of automating such a mentoring pipeline within future agentic governance frameworks. Overall, the approach demonstrates consistent and measurable accuracy improvements across diverse configurations, particularly in settings dominated by specification ambiguity. For reproducibility, we released our code as open source under the Agent Mentor library. 

---
# Thinking Fast, Thinking Wrong: Intuitiveness Modulates LLM Counterfactual Reasoning in Policy Evaluation 

**Authors**: Yanjie He  

**Link**: [PDF](https://arxiv.org/pdf/2604.10511)  

**Abstract**: Large language models (LLMs) are increasingly used for causal and counterfactual reasoning, yet their reliability in real-world policy evaluation remains underexplored. We construct a benchmark of 40 empirical policy evaluation cases drawn from economics and social science, each grounded in peer-reviewed evidence and classified by intuitiveness -- whether the empirical finding aligns with (obvious), is unclear relative to (ambiguous), or contradicts (counter-intuitive) common prior expectations. We evaluate four frontier LLMs across five prompting strategies with 2,400 experimental trials and analyze the results using mixed-effects logistic regression. Our findings reveal three key results: (1) a chain-of-thought (CoT) paradox, where chain-of-thought prompting dramatically improves performance on obvious cases but this benefit is nearly eliminated on counter-intuitive ones (interaction OR = 0.053, $p < 0.001$); (2) intuitiveness as the dominant factor, explaining more variance than model choice or prompting strategy (ICC = 0.537); and (3) a knowledge-reasoning dissociation, where citation-based familiarity is unrelated to accuracy ($p = 0.53$), suggesting models possess relevant knowledge but fail to reason with it when findings contradict intuition. We frame these results through the lens of dual-process theory (System 1 vs. System 2) and argue that current LLMs' "slow thinking" may be little more than "slow talking" -- they produce the form of deliberative reasoning without the substance. 

---
# Beyond Compliance: A Resistance-Informed Motivation Reasoning Framework for Challenging Psychological Client Simulation 

**Authors**: Danni Liu, Bo Liu, Yuxin Hu, Hantao Zhao, Yan Liu, Ding Ding, Jiahui Jin, Jiuxin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.10507)  

**Abstract**: Psychological client simulators have emerged as a scalable solution for training and evaluating counselor trainees and psychological LLMs. Yet existing simulators exhibit unrealistic over-compliance, leaving counselors underprepared for the challenging behaviors common in real-world practice. To bridge this gap, we present ResistClient, which systematically models challenging client behaviors grounded in Client Resistance Theory by integrating external behaviors with underlying motivational mechanisms. To this end, we propose Resistance-Informed Motivation Reasoning (RIMR), a two-stage training framework. First, RIMR mitigates compliance bias via supervised fine-tuning on RPC, a large-scale resistance-oriented psychological conversation dataset covering diverse client profiles. Second, beyond surface-level response imitation, RIMR models psychologically coherent motivation reasoning before response generation, jointly optimizing motivation authenticity and response consistency via process-supervised reinforcement learning. Extensive automatic and expert evaluations show that ResistClient substantially outperforms existing simulators in challenge fidelity, behavioral plausibility, and reasoning coherence. Moreover, ResistClient facilities evaluation of psychological LLMs under challenging conditions, offering new optimization directions for mental health dialogue systems. 

---
# A Progressive Training Strategy for Vision-Language Models to Counteract Spatio-Temporal Hallucinations in Embodied Reasoning 

**Authors**: Xiaoda Yang, Shuai Yang, Can Wang, Jingyang Xue, Menglan Tang, Checheng Yu, Xunzhe Zhou, Sashuai Zhou, Tao Jin, Lixin Yang, Xiangyu Yue, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.10506)  

**Abstract**: Vision-Language Models (VLMs) have made significant strides in static image understanding but continue to face critical hurdles in spatiotemporal reasoning. A major bottleneck is "multi-image reasoning hallucination", where a massive performance drop between forward and reverse temporal queries reveals a dependence on superficial shortcuts instead of genuine causal understanding. To mitigate this, we first develop a new Chain-of-Thought (CoT) dataset that decomposes intricate reasoning into detailed spatiotemporal steps and definitive judgments. Building on this, we present a progressive training framework: it initiates with supervised pre-training on our CoT dataset to instill logical structures, followed by fine-tuning with scalable weakly-labeled data for broader generalization. Our experiments demonstrate that this approach not only improves backbone accuracy but also slashes the forward-backward performance gap from over 70\% to only 6.53\%. This confirms the method's ability to develop authentic dynamic reasoning and reduce the inherent temporal biases of current VLMs. 

---
# Cooperation in Human and Machine Agents: Promise Theory Considerations 

**Authors**: M. Burgess  

**Link**: [PDF](https://arxiv.org/pdf/2604.10505)  

**Abstract**: Agent based systems are more common than we may think. A Promise Theory perspective on cooperation, in systems of human-machine agents, offers a unified perspective on organization and functional design with semi-automated efforts, in terms of the abstract properties of autonomous agents, This applies to human efforts, hardware systems, software, and artificial intelligence, with and without management. One may ask how does a reasoning system of components keep to an intended purpose? As the agent paradigm is now being revived, in connection with artificial intelligence agents, I revisit established principles of agent cooperation, as applied to humans, machines, and their mutual interactions. Promise Theory represents the fundamentals of signalling, comprehension, trust, risk, and feedback between agents, and offers some lessons about success and failure. 

---
# CARO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation 

**Authors**: Bingzhe Wu, Haotian Lu, Yuchen Mou  

**Link**: [PDF](https://arxiv.org/pdf/2604.10504)  

**Abstract**: Current large language models (LLMs), even those explicitly trained for reasoning, often struggle with ambiguous content moderation cases due to misleading "decision shortcuts" embedded in context. Inspired by cognitive psychology insights into expert moderation, we introduce \caro (Chain-of-Analogy Reasoning Optimization), a novel two-stage training framework to induce robust analogical reasoning in LLMs. First, \caro bootstraps analogical reasoning chains via retrieval-augmented generation (RAG) on moderation data and performs supervised fine-tuning (SFT). Second, we propose a customized direct preference optimization (DPO) approach to reinforce analogical reasoning behaviors explicitly. Unlike static retrieval methods, \caro dynamically generates tailored analogical references during inference, effectively mitigating harmful decision shortcuts. Extensive experiments demonstrate that \caro substantially outperforms state-of-the-art reasoning models (DeepSeek R1, QwQ), specialized moderation models (LLaMA Guard), and advanced fine-tuning and retrieval-augmented methods, achieving an average F1 score improvement of 24.9\% on challenging ambiguous moderation benchmarks. 

---
# CHAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs 

**Authors**: Haotian Lu, Yuchen Mou, Bingzhe Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10502)  

**Abstract**: Content moderation in online platforms faces persistent challenges due to the evolving complexity of user-generated content and the limitations of traditional rule-based and machine learning approaches. While recent advances in large language models (LLMs) have enabled more sophisticated moderation via direct prompting or fine-tuning, these approaches often exhibit limited generalization, interpretability, and adaptability to unseen or ambiguous cases.
In this work, we propose a novel moderation framework that leverages analogical examples to enhance rule induction and decision reliability. Our approach integrates end-to-end optimization of analogical retrieval, rule generation, and moderation classification, enabling the dynamic adaptation of moderation rules to diverse content scenarios. Through comprehensive experiments, we demonstrate that our method significantly outperforms both rule-injected fine-tuning baselines and multi-stage static RAG pipelines in terms of moderation accuracy and rule quality. Further evaluations, including human assessments and external model generalization tests, confirm that our framework produces rules with better clarity, interpretability, and applicability. These findings show that analogical example-driven methods can advance robust, explainable, and generalizable content moderation in real-world applications. 

---
# Tracing the Roots: A Multi-Agent Framework for Uncovering Data Lineage in Post-Training LLMs 

**Authors**: Yu Li, Xiaoran Shang, Qizhi Pei, Yun Zhu, Xin Gao, Honglin Lin, Zhanping Zhong, Zhuoshi Pan, Zheng Liu, Xiaoyang Wang, Conghui He, Dahua Lin, Feng Zhao, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10480)  

**Abstract**: Post-training data plays a pivotal role in shaping the capabilities of Large Language Models (LLMs), yet datasets are often treated as isolated artifacts, overlooking the systemic connections that underlie their evolution. To disentangle these complex relationships, we introduce the concept of \textbf{data lineage} to the LLM ecosystem and propose an automated multi-agent framework to reconstruct the evolutionary graph of dataset development. Through large-scale lineage analysis, we characterize domain-specific structural patterns, such as vertical refinement in math-oriented datasets and horizontal aggregation in general-domain corpora. Moreover, we uncover pervasive systemic issues, including \textit{structural redundancy} induced by implicit dataset intersections and the \textit{propagation of benchmark contamination} along lineage paths. To demonstrate the practical value of lineage analysis for data construction, we leverage the reconstructed lineage graph to create a \textit{lineage-aware diversity-oriented dataset}. By anchoring instruction sampling at upstream root sources, this approach mitigates downstream homogenization and hidden redundancy, yielding a more diverse post-training corpus. We further highlight lineage-centric analysis as an efficient and robust topological alternative to sample-level dataset comparison for large-scale data ecosystems. By grounding data construction in explicit lineage structures, our work advances post-training data curation toward a more systematic and controllable paradigm. 

---
# PEMANT: Persona-Enriched Multi-Agent Negotiation for Travel 

**Authors**: Yuran Sun, Mustafa Sameen, Yaotian Zhang, Chia-yu Wu, Xilei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.10475)  

**Abstract**: Modeling household-level trip generation is fundamental to accurate demand forecasting, traffic flow estimation, and urban system planning. Existing studies were mostly based on classical machine learning models with limited predictive capability, while recent LLM-based approaches have yet to incorporate behavioral theory or intra-household interaction dynamics, both of which are critical for modeling realistic collective travel decisions. To address these limitations, we propose a novel LLM-based framework, named Persona-Enriched Multi-Agent Negotiation for Travel (PEMANT), which first integrates behavioral theory for individualized persona modeling and then conducts household-level trip planning negotiations via a structured multi-agent conversation. Specifically, PEMANT transforms static sociodemographic attributes into coherent narrative profiles that explicitly encode household-level attitudes, subjective norms, and perceived behavioral controls, following our proposed Household-Aware Chain-of-Planned-Behavior (HA-CoPB) framework. Building on these theory-grounded personas, PEMANT captures real-world household decision negotiation via a structured two-phase multi-agent conversation framework with a novel persona-alignment control mechanism. Evaluated on both national and regional household travel survey datasets, PEMANT consistently outperforms state-of-the-art benchmarks across datasets. 

---
# VeriSim: A Configurable Framework for Evaluating Medical AI Under Realistic Patient Noise 

**Authors**: Sina Mansouri, Mohit Marvania, Vibhavari Ashok Shihorkar, Han Ngoc Tran, Kazhal Shafiei, Mehrdad Fazli, Yikuan Li, Ziwei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10441)  

**Abstract**: Medical large language models (LLMs) achieve impressive performance on standardized benchmarks, yet these evaluations fail to capture the complexity of real clinical encounters where patients exhibit memory gaps, limited health literacy, anxiety, and other communication barriers. We introduce VeriSim, a truth-preserving patient simulation framework that injects controllable, clinically evidence-grounded noise into patient responses while maintaining strict adherence to medical ground truth through a hybrid UMLS-LLM verification mechanism. Our framework operationalizes six noise dimensions derived from peer-reviewed medical communication literature, capturing authentic clinical phenomena such as patient recall limitations, health literacy barriers, and stigma-driven non-disclosure. Experiments across seven open-weight LLMs reveal that all models degrade significantly under realistic patient noise, with diagnostic accuracy dropping 15-25% and conversation length increasing 34-55%. Notably, smaller models (7B) show 40% greater degradation than larger models (70B+), while medical fine-tuning on standard corpora provides limited robustness benefits against patient communication noise. Evaluation by board-certified clinicians demonstrates high-quality simulation with strong inter-annotator agreement (kappa > 0.80), while LLM-as-a-Judge serves as a validated auxiliary evaluator achieving comparable reliability for scalable assessment. Our results highlight a critical Sim-to-Real gap in current medical AI. We release VeriSim as an open-source noise-injection framework, establishing a rigorous testbed for evaluating clinical robustness. 

---
# Safety Guarantees in Zero-Shot Reinforcement Learning for Cascade Dynamical Systems 

**Authors**: Shima Rabiei, Sandipan Mishra, Santiago Paternain  

**Link**: [PDF](https://arxiv.org/pdf/2604.10429)  

**Abstract**: This paper considers the problem of zero-shot safety guarantees for cascade dynamical systems. These are systems where a subset of the states (the inner states) affects the dynamics of the remaining states (the outer states) but not vice-versa. We define safety as remaining on a set deemed safe for all times with high probability. We propose to train a safe RL policy on a reduced-order model, which ignores the dynamics of the inner states, but it treats it as an action that influences the outer state. Thus, reducing the complexity of the training. When deployed in the full system the trained policy is combined with a low-level controller whose task is to track the reference provided by the RL policy. Our main theoretical contribution is a bound on the safe probability in the full-order system. In particular, we establish the interplay between the probability of remaining safe after the zero-shot deployment and the quality of the tracking of the inner states. We validate our theoretical findings on a quadrotor navigation task, demonstrating that the preservation of the safety guarantees is tied to the bandwidth and tracking capabilities of the low-level controller. 

---
# CWCD: Category-Wise Contrastive Decoding for Structured Medical Report Generation 

**Authors**: Shantam Srivastava, Mahesh Bhosale, David Doermann, Mingchen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.10410)  

**Abstract**: Interpreting chest X-rays is inherently challenging due to the overlap between anatomical structures and the subtle presentation of many clinically significant pathologies, making accurate diagnosis time-consuming even for experienced radiologists. Recent radiology-focused foundation models, such as LLaVA-Rad and Maira-2, have positioned multi-modal large language models (MLLMs) at the forefront of automated radiology report generation (RRG). However, despite these advances, current foundation models generate reports in a single forward pass. This decoding strategy diminishes attention to visual tokens and increases reliance on language priors as generation proceeds, which in turn introduces spurious pathology co-occurrences in the generated reports. To mitigate these limitations, we propose Category-Wise Contrastive Decoding (CWCD), a novel and modular framework designed to enhance structured radiology report generation (SRRG). Our approach introduces category-specific parameterization and generates category-wise reports by contrasting normal X-rays with masked X-rays using category-specific visual prompts. Experimental results demonstrate that CWCD consistently outperforms baseline methods across both clinical efficacy and natural language generation metrics. An ablation study further elucidates the contribution of each architectural component to overall performance. 

---
# TrajOnco: a multi-agent framework for temporal reasoning over longitudinal EHR for multi-cancer early detection 

**Authors**: Sihang Zeng, Young Won Kim, Wilson Lau, Ehsan Alipour, Ruth Etzioni, Meliha Yetisgen, Anand Oka  

**Link**: [PDF](https://arxiv.org/pdf/2604.10386)  

**Abstract**: Accurate estimation of cancer risk from longitudinal electronic health records (EHRs) could support earlier detection and improved care, but modeling such complex patient trajectories remains challenging. We present TrajOnco, a training-free, multi-agent large language model (LLM) framework designed for scalable multi-cancer early detection. Using a chain-of-agents architecture with long-term memory, TrajOnco performs temporal reasoning over sequential clinical events to generate patient-level summaries, evidence-linked rationales, and predicted risk scores. We evaluated TrajOnco on de-identified Truveta EHR data across 15 cancer types using matched case-control cohorts, predicting risk of cancer diagnosis at 1 year. In zero-shot evaluation, TrajOnco achieved AUROCs of 0.64-0.80, performing comparably to supervised machine learning in a lung cancer benchmark while demonstrating better temporal reasoning than single-agent LLMs. The multi-agent design also enabled effective temporal reasoning with smaller-capacity models such as GPT-4.1-mini. The fidelity of TrajOnco's output was validated through human evaluation. Furthermore, TrajOnco's interpretable reasoning outputs can be aggregated to reveal population-level risk patterns that align with established clinical knowledge. These findings highlight the potential of multi-agent LLMs to execute interpretable temporal reasoning over longitudinal EHRs, advancing both scalable multi-cancer early detection and clinical insight generation. 

---
# Beyond Monologue: Interactive Talking-Listening Avatar Generation with Conversational Audio Context-Aware Kernels 

**Authors**: Yuzhe Weng, Haotian Wang, Xinyi Yu, Xiaoyan Wu, Haoran Xu, Shan He, Jun Du  

**Link**: [PDF](https://arxiv.org/pdf/2604.10367)  

**Abstract**: Audio-driven human video generation has achieved remarkable success in monologue scenarios, largely driven by advancements in powerful video generation foundation models. Moving beyond monologues, authentic human communication is inherently a full-duplex interactive process, requiring virtual agents not only to articulate their own speech but also to react naturally to incoming conversational audio. Most existing methods simply extend conventional audio-driven paradigms to listening scenarios. However, relying on strict frame-to-frame alignment renders the model's response to long-range conversational dynamics rigid, whereas directly introducing global attention catastrophically degrades lip synchronization. Recognizing the unique temporal Scale Discrepancy between talking and listening behaviors, we introduce a multi-head Gaussian kernel to explicitly inject this physical intuition into the model as a progressive temporal inductive bias. Building upon this, we construct a full-duplex interactive virtual agent capable of simultaneously processing dual-stream audio inputs for both talking and listening. Furthermore, we introduce a rigorously cleaned Talking-Listening dataset VoxHear featuring perfectly decoupled speech and background audio tracks. Extensive experiments demonstrate that our approach successfully fuses strong temporal alignment with deep contextual semantics, setting a new state-of-the-art for generating highly natural and responsive full-duplex interactive digital humans. The project page is available at this https URL . 

---
# ClawVM: Harness-Managed Virtual Memory for Stateful Tool-Using LLM Agents 

**Authors**: Mofasshara Rafique, Laurent Bindschaedler  

**Link**: [PDF](https://arxiv.org/pdf/2604.10352)  

**Abstract**: Stateful tool-using LLM agents treat the context window as working memory, yet today's agent harnesses manage residency and durability as best-effort, causing recurring failures: lost state after compaction, bypassed flushes on reset, and destructive writeback. We present \textsc{ClawVM}, a virtual memory layer that manages state as typed pages with minimum-fidelity invariants, multi-resolution representations under a token budget, and validated writeback at every lifecycle boundary. Because the harness already assembles prompts, mediates tools, and observes lifecycle events, it is the natural enforcement point; placing the contract there makes residency and durability deterministic and auditable. Across synthetic workloads, 12 real-session traces, and adversarial stress tests, \textsc{ClawVM} eliminates all policy-controllable faults whenever the minimum-fidelity set fits within the token budget, confirmed by an offline oracle, and adds median <50 microseconds of policy-engine overhead per turn. 

---
# VeriTrans: Fine-Tuned LLM-Assisted NL-to-PL Translation via a Deterministic Neuro-Symbolic Pipeline 

**Authors**: Xuan Liu, Dheeraj Kodakandla, Kushagra Srivastva, Mahfuza Farooque  

**Link**: [PDF](https://arxiv.org/pdf/2604.10341)  

**Abstract**: \textbf{VeriTrans} is a reliability-first ML system that compiles natural-language requirements into solver-ready logic with validator-gated reliability. The pipeline integrates an instruction-tuned NL$\!\to\!$PL translator, round-trip reconstruction (PL$\!\to\!$NL) used as a high-precision acceptance gate, and canonical PL$\!\to\!$CNF compilation, all executed via fixed API configuration (temperature$=0$; fine-tuning runs use seed$=42$) and per-item artifact logging (prompts, outputs, hashes) to support auditability and replay-driven debugging. On \textbf{SatBench} (2{,}100 specifications), VeriTrans achieves 94.46\% SAT/UNSAT correctness and 87.73\% median round-trip similarity. Compact fine-tuning on 100--150 curated examples improves fidelity by about 1--1.5\,pp without increasing latency (mean 25.8\,s/spec on our 201-spec runtime subset). A thresholded acceptance policy on the round-trip score exposes a reliability--coverage knob: at $\tau{=}75$, roughly 68\% of items are retained with $\sim$94\% correctness on the accepted set. Validator overhead contributes $<15\%$ of end-to-end runtime, and all prompts/responses and timing metadata are logged to enable replay-driven debugging and regression testing. By separating learned translation from symbolic verification and enforcing deterministic, validator-gated acceptance, VeriTrans turns NL$\!\to\!$logic front-ends into auditable, reproducible components for reliability-critical workflows. 

---
# Zero-shot World Models Are Developmentally Efficient Learners 

**Authors**: Khai Loong Aw, Klemen Kotar, Wanhee Lee, Seungwoo Kim, Khaled Jedoui, Rahul Venkatesh, Lilian Naing Chen, Michael C. Frank, Daniel L.K. Yamins  

**Link**: [PDF](https://arxiv.org/pdf/2604.10333)  

**Abstract**: Young children demonstrate early abilities to understand their physical world, estimating depth, motion, object coherence, interactions, and many other aspects of physical scene understanding. Children are both data-efficient and flexible cognitive systems, creating competence despite extremely limited training data, while generalizing to myriad untrained tasks -- a major challenge even for today's best AI systems. Here we introduce a novel computational hypothesis for these abilities, the Zero-shot Visual World Model (ZWM). ZWM is based on three principles: a sparse temporally-factored predictor that decouples appearance from dynamics; zero-shot estimation through approximate causal inference; and composition of inferences to build more complex abilities. We show that ZWM can be learned from the first-person experience of a single child, rapidly generating competence across multiple physical understanding benchmarks. It also broadly recapitulates behavioral signatures of child development and builds brain-like internal representations. Our work presents a blueprint for efficient and flexible learning from human-scale data, advancing both a computational account for children's early physical understanding and a path toward data-efficient AI systems. 

---
# From GPT-3 to GPT-5: Mapping their capabilities, scope, limitations, and consequences 

**Authors**: Hina Afridi, Habib Ullah, Sultan Daud Khan, Mohib Ullah  

**Link**: [PDF](https://arxiv.org/pdf/2604.10332)  

**Abstract**: We present the progress of the GPT family from GPT-3 through GPT-3.5, GPT-4, GPT-4 Turbo, GPT-4o, GPT-4.1, and the GPT-5 family. Our work is comparative rather than merely historical. We investigates how the family evolved in technical framing, user interaction, modality, deployment architecture, and governance viewpoint. The work focuses on five recurring themes: technical progression, capability changes, deployment shifts, persistent limitations, and downstream consequences. In term of research design, we consider official technical reports, system cards, API and model documentation, product announcements, release notes, and peer-reviewed secondary studies. A primary assertion is that later GPT generations should not be interpreted only as larger or more accurate language models. Instead, the family evolves from a scaled few-shot text predictor into a set of aligned, multimodal, tool-oriented, long-context, and increasingly workflow-integrated systems. This development complicates simple model-to-model comparison because product routing, tool access, safety tuning, and interface design become part of the effective system. Across generations, several limitations remain unchanged: hallucination, prompt sensitivity, benchmark fragility, uneven behavior across domains and populations, and incomplete public transparency about architecture and training. However, the family has evolved software development, educational practice, information work, interface design, and discussions of frontier-model governance. We infer that the transition from GPT-3 to GPT-5 is best understood not only as an improvement in model capability, but also as a broader reformulation of what a deployable AI system is, how it is evaluated, and where responsibility should be located when such systems are used at scale. 

---
# Gypscie: A Cross-Platform AI Artifact Management System 

**Authors**: Fabio Porto, Eduardo Ogasawara, Gabriela Moraes Botaro, Julia Neumann Bastos, Augusto Fonseca, Esther Pacitti, Patrick Valduriez  

**Link**: [PDF](https://arxiv.org/pdf/2604.10311)  

**Abstract**: Artificial Intelligence (AI) models, encompassing both traditional machine learning (ML) and more advanced approaches such as deep learning and large language models (LLMs), play a central role in modern applications. AI model lifecycle management involves the end-to-end process of managing these models, from data collection and preparation to model building, evaluation, deployment, and continuous monitoring. This process is inherently complex, as it requires the coordination of diverse services that manage AI artifacts such as datasets, dataflows, and models, all orchestrated to operate seamlessly. In this context, it is essential to isolate applications from the complexity of interacting with heterogeneous services, datasets, and AI platforms.
In this paper, we introduce Gypscie, a cross-platform AI artifact management system. By providing a unified view of all AI artifacts, the Gypscie platform simplifies the development and deployment of AI applications. This unified view is realized through a knowledge graph that captures application semantics and a rule-based query language that supports reasoning over data and models. Model lifecycle activities are represented as high-level dataflows that can be scheduled across multiple platforms, such as servers, cloud platforms, or supercomputers. Finally, Gypscie records provenance information about the artifacts it produces, thereby enabling explainability. Our qualitative comparison with representative AI systems shows that Gypscie supports a broader range of functionalities across the AI artifact lifecycle. Our experimental evaluation demonstrates that Gypscie can successfully optimize and schedule dataflows on AI platforms from an abstract specification. 

---
# TimeSeriesExamAgent: Creating Time Series Reasoning Benchmarks at Scale 

**Authors**: Malgorzata Gwiazda, Yifu Cai, Mononito Goswami, Arjun Choudhry, Artur Dubrawski  

**Link**: [PDF](https://arxiv.org/pdf/2604.10291)  

**Abstract**: Large Language Models (LLMs) have shown promising performance in time series modeling tasks, but do they truly understand time series data? While multiple benchmarks have been proposed to answer this fundamental question, most are manually curated and focus on narrow domains or specific skill sets. To address this limitation, we propose scalable methods for creating comprehensive time series reasoning benchmarks that combine the flexibility of templates with the creativity of LLM agents. We first develop TimeSeriesExam, a multiple-choice benchmark using synthetic time series to evaluate LLMs across five core reasoning categories: pattern recognitionnoise understandingsimilarity analysisanomaly detection, and causality. Then, with TimeSeriesExamAgent, we scale our approach by automatically generating benchmarks from real-world datasets spanning healthcare, finance and weather domains. Through multi-dimensional quality evaluation, we demonstrate that our automatically generated benchmarks achieve diversity comparable to manually curated alternatives. However, our experiments reveal that LLM performance remains limited in both abstract time series reasoning and domain-specific applications, highlighting ongoing challenges in enabling effective time series understanding in these models. TimeSeriesExamAgent is available at this https URL. 

---
# AI Organizations are More Effective but Less Aligned than Individual Agents 

**Authors**: Judy Hanwen Shen, Daniel Zhu, Siddarth Srinivasan, Henry Sleight, Lawrence T. Wagner III, Morgan Jane Matthews, Erik Jones, Jascha Sohl-Dickstein  

**Link**: [PDF](https://arxiv.org/pdf/2604.10290)  

**Abstract**: AI is increasingly deployed in multi-agent systems; however, most research considers only the behavior of individual models. We experimentally show that multi-agent "AI organizations" are simultaneously more effective at achieving business goals, but less aligned, than individual AI agents. We examine 12 tasks across two practical settings: an AI consultancy providing solutions to business problems and an AI software team developing software products. Across all settings, AI Organizations composed of aligned models produce solutions with higher utility but greater misalignment compared to a single aligned model. Our work demonstrates the importance of considering interacting systems of AI agents when doing both capabilities and safety research. 

---
# Dead Cognitions: A Census of Misattributed Insights 

**Authors**: Aaron Tuor, claude.ai  

**Link**: [PDF](https://arxiv.org/pdf/2604.10288)  

**Abstract**: This essay identifies a failure mode of AI chat systems that we term attribution laundering: the model performs substantive cognitive work and then rhetorically credits the user for having generated the resulting insights. Unlike transparent versions of glad handing sycophancy, attribution laundering is systematically occluded to the person it affects and self-reinforcing -- eroding users' ability to accurately assess their own cognitive contributions over time. We trace the mechanisms at both individual and societal scales, from the chat interface that discourages scrutiny to the institutional pressures that reward adoption over accountability. The document itself is an artifact of the process it describes, and is color-coded accordingly -- though the views expressed are the authors' own, not those of any affiliated institution, and the boundary between the human author's views and Claude's is, as the essay argues, difficult to draw. 

---
# STARS: Skill-Triggered Audit for Request-Conditioned Invocation Safety in Agent Systems 

**Authors**: Guijia Zhang, Shu Yang, Xilin Gong, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10286)  

**Abstract**: Autonomous language-model agents increasingly rely on installable skills and tools to complete user tasks. Static skill auditing can expose capability surface before deployment, but it cannot determine whether a particular invocation is unsafe under the current user request and runtime context. We therefore study skill invocation auditing as a continuous-risk estimation problem: given a user request, candidate skill, and runtime context, predict a score that supports ranking and triage before a hard intervention is applied. We introduce STARS, which combines a static capability prior, a request-conditioned invocation risk model, and a calibrated risk-fusion policy. To evaluate this setting, we construct SIA-Bench, a benchmark of 3,000 invocation records with group-safe splits, lineage metadata, runtime context, canonical action labels, and derived continuous-risk targets. On a held-out split of indirect prompt injection attacks, calibrated fusion reaches 0.439 high-risk AUPRC, improving over 0.405 for the contextual scorer and 0.380 for the strongest static baseline, while the contextual scorer remains better calibrated with 0.289 expected calibration error. On the locked in-distribution test split, gains are smaller and static priors remain useful. The resulting claim is therefore narrower: request-conditioned auditing is most valuable as an invocation-time risk-scoring and triage layer rather than as a replacement for static screening. Code is available at this https URL. 

---
# The Amazing Agent Race: Strong Tool Users, Weak Navigators 

**Authors**: Zae Myung Kim, Dongseok Lee, Jaehyung Kim, Vipul Raheja, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10261)  

**Abstract**: Existing tool-use benchmarks for LLM agents are overwhelmingly linear: our analysis of six benchmarks shows 55 to 100% of instances are simple chains of 2 to 5 steps. We introduce The Amazing Agent Race (AAR), a benchmark featuring directed acyclic graph (DAG) puzzles (or "legs") with fork-merge tool chains. We release 1,400 instances across two variants: sequential (800 legs) and compositional (600 DAG legs). Agents must navigate Wikipedia, execute multi-step tool chains, and aggregate results into a verifiable answer. Legs are procedurally generated from Wikipedia seeds across four difficulty levels with live-API validation. Three complementary metrics (finish-line accuracy, pit-stop visit rate, and roadblock completion rate) separately diagnose navigation, tool-use, and arithmetic failures. Evaluating three agent frameworks on 1,400 legs, the best achieves only 37.2% accuracy. Navigation errors dominate (27 to 52% of trials) while tool-use errors remain below 17%, and agent architecture matters as much as model scale (Claude Code matches Codex CLI at 37% with 6x fewer tokens). The compositional structure of AAR reveals that agents fail not at calling tools but at navigating to the right pages, a blind spot invisible to linear benchmarks. The project page can be accessed at: this https URL 

---
# A Dual-Positive Monotone Parameterization for Multi-Segment Bids and a Validity Assessment Framework for Reinforcement Learning Agent-based Simulation of Electricity Markets 

**Authors**: Zunnan Xu, Zhaoxia Jing, Zhanhua Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.10252)  

**Abstract**: Reinforcement learning agent-based simulation (RL-ABS) has become an important tool for electricity market mechanism analysis and evaluation. In the modeling of monotone, bounded, multi-segment stepwise bids, existing methods typically let the policy network first output an unconstrained action and then convert it into a feasible bid curve satisfying monotonicity and boundedness through post-processing mappings such as sorting, clipping, or projection. However, such post-processing mappings often fail to satisfy continuous differentiability, injectivity, and invertibility at boundaries or kinks, thereby causing gradient distortion and leading to spurious convergence in simulation results. Meanwhile, most existing studies conduct mechanism analysis and evaluation mainly on the basis of training-curve convergence, without rigorously assessing the distance between the simulation outcomes and Nash equilibrium, which severely undermines the credibility of the results. To address these issues, this paper proposes... 

---
# SVSR: A Self-Verification and Self-Rectification Paradigm for Multimodal Reasoning 

**Authors**: Zhe Qian, Nianbing Su, Zhonghua Wang, Hebei Li, Zhongxing Xu, Yueying Li, Fei Luo, Zhuohan Ouyang, Yanbiao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.10228)  

**Abstract**: Current multimodal models often suffer from shallow reasoning, leading to errors caused by incomplete or inconsistent thought processes. To address this limitation, we propose Self-Verification and Self-Rectification (SVSR), a unified framework that explicitly integrates self-verification and self-rectification into the model's reasoning pipeline, substantially improving robustness and reliability in complex visual understanding and multimodal reasoning tasks. SVSR is built on a novel three-stage training paradigm. First, we construct a high-quality unified preference dataset by refining reasoning traces from pre-trained vision-language models, incorporating both forward and backward reasoning to embed self-reflective signals. Second, we perform cold-start supervised fine-tuning on this dataset to learn structured, multi-step reasoning behaviors. Third, we apply a Semi-online Direct Preference Optimization (Semi-online DPO) process, continuously augmenting the training corpus with high-quality, model-generated reasoning traces filtered by a powerful teacher VLM. This pipeline enables the model to learn, elicit, and refine its ability to self-verify and self-rectify. Extensive experiments across diverse benchmarks demonstrate that SVSR improves reasoning accuracy and enables stronger generalization to unseen tasks and question types. Notably, once trained with explicit self-reflective reasoning, the model also exhibits improved implicit reasoning ability, outperforming strong baselines even when no explicit reasoning traces are provided. These results highlight the potential of SVSR for building more dependable, introspective, and cognitively aligned multimodal systems. 

---
# Cognitive Pivot Points and Visual Anchoring: Unveiling and Rectifying Hallucinations in Multimodal Reasoning Models 

**Authors**: Zhe Qian, Yanbiao Ma, Zhuohan Ouyang, Zhonghua Wang, Zhongxing Xu, Fei Luo, Xinyu Liu, Zongyuan Ge, Yike Guo, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.10219)  

**Abstract**: Multimodal Large Reasoning Models (MLRMs) have achieved remarkable strides in visual reasoning through test time compute scaling, yet long chain reasoning remains prone to hallucinations. We identify a concerning phenomenon termed the Reasoning Vision Truth Disconnect (RVTD): hallucinations are strongly correlated with cognitive bifurcation points that often exhibit high entropy states. We attribute this vulnerability to a breakdown in visual semantic anchoring, localized within the network's intermediate layers; specifically, during these high uncertainty transitions, the model fails to query visual evidence, reverting instead to language priors. Consequently, we advocate a shift from solely outcome level supervision to augmenting it with fine grained internal attention guidance. To this end, we propose V-STAR (Visual Structural Training with Attention Reinforcement), a lightweight, holistic training paradigm designed to internalize visually aware reasoning capabilities. Central to our approach is the Hierarchical Visual Attention Reward (HVAR), integrated within the GRPO framework. Upon detecting high entropy states, this mechanism dynamically incentivizes visual attention across critical intermediate layers, thereby anchoring the reasoning process back to the visual input. Furthermore, we introduce the Forced Reflection Mechanism (FRM), a trajectory editing strategy that disrupts cognitive inertia by triggering reflection around high entropy cognitive bifurcation points and encouraging verification of subsequent steps against the visual input, thereby translating external debiasing interventions into an intrinsic capability for hallucination mitigation. 

---
# Edu-MMBias: A Three-Tier Multimodal Benchmark for Auditing Social Bias in Vision-Language Models under Educational Contexts 

**Authors**: Ruijia Li, Mingzi Zhang, Zengyi Yu, Yuang Wei, Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10200)  

**Abstract**: As Vision-Language Models (VLMs) become integral to educational decision-making, ensuring their fairness is paramount. However, current text-centric evaluations neglect the visual modality, leaving an unregulated channel for latent social biases. To bridge this gap, we present Edu-MMBias, a systematic auditing framework grounded in the tri-component model of attitudes from social psychology. This framework diagnoses bias across three hierarchical dimensions: cognitive, affective, and behavioral. Utilizing a specialized generative pipeline that incorporates a self-correct mechanism and human-in-the-loop verification, we synthesize contamination-resistant student profiles to conduct a holistic stress test on state-of-the-art VLMs. Our extensive audit reveals critical, counter-intuitive patterns: models exhibit a compensatory class bias favoring lower-status narratives while simultaneously harboring deep-seated health and racial stereotypes. Crucially, we find that visual inputs act as a safety backdoor, triggering a resurgence of biases that bypass text-based alignment safeguards and revealing a systematic misalignment between latent cognition and final decision-making. The contributions of this paper are available at: this https URL. 

---
# Credit-Budgeted ICPC-Style Coding: When Agents Must Pay for Every Decision 

**Authors**: Lingfeng Zhou, Junhao Shi, Jin Gao, Dequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10182)  

**Abstract**: Current evaluations of autonomous coding agents assume an unrealistic, infinite-resource environment. However, real-world software engineering is a resource-bound competition. As we scale toward large agent swarms, ignoring compute and time costs risks catastrophic budget exhaustion. To shift the focus from isolated accuracy to cost-aware problem-solving, we introduce USACOArena, an interactive ACM-ICPC-style arena driven by a strict "credit" economy. Every generated token, local test, and elapsed second depletes a fixed budget, forcing agents to make strategic trade-offs. Our comprehensive profiling reveals that frontier single agents and swarms currently fail to optimally balance accuracy with these constraints, exhibiting divergent, path-dependent behaviors. Ultimately, USACOArena provides an essential dynamic training ground for developing highly efficient, resource-aware agent architectures. 

---
# PoreDiT: A Scalable Generative Model for Large-Scale Digital Rock Reconstruction 

**Authors**: Yizhuo Huang, Baoquan Sun, Haibo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10171)  

**Abstract**: This manuscript presents PoreDiT, a novel generative model designed for high-efficiency digital rock reconstruction at gigavoxel scales. Addressing the significant challenges in digital rock physics (DRP), particularly the trade-off between resolution and field-of-view (FOV), and the computational bottlenecks associated with traditional deep learning architectures, PoreDiT leverages a three-dimensional (3D) Swin Transformer to break through these limitations. By directly predicting the binary probability field of pore spaces instead of grayscale intensities, the model preserves key topological features critical for pore-scale fluid flow and transport simulations. This approach enhances computational efficiency, enabling the generation of ultra-large-scale ($1024^3$ voxels) digital rock samples on consumer-grade hardware. Furthermore, PoreDiT achieves physical fidelity comparable to previous state-of-the-art methods, including accurate porosity, pore-scale permeability, and Euler characteristics. The model's ability to scale efficiently opens new avenues for large-domain hydrodynamic simulations and provides practical solutions for researchers in pore-scale fluid mechanics, reservoir characterization, and carbon sequestration. 

---
# MAVEN-T: Multi-Agent enVironment-aware Enhanced Neural Trajectory predictor with Reinforcement Learning 

**Authors**: Wenchang Duan  

**Link**: [PDF](https://arxiv.org/pdf/2604.10169)  

**Abstract**: Trajectory prediction remains a critical yet challenging component in autonomous driving systems, requiring sophisticated reasoning capabilities while meeting strict real-time deployment constraints. While knowledge distillation has demonstrated effectiveness in model compression, existing approaches often fail to preserve complex decision-making capabilities, particularly in dynamic multi-agent scenarios. This paper introduces MAVEN-T, a teacher-student framework that achieves state-of-the-art trajectory prediction through complementary architectural co-design and progressive distillation. The teacher employs hybrid attention mechanisms for maximum representational capacity, while the student uses efficient architectures optimized for deployment. Knowledge transfer is performed via multi-granular distillation with adaptive curriculum learning that dynamically adjusts complexity based on performance. Importantly, the framework incorporates reinforcement learning to overcome the imitation ceiling of traditional distillation, enabling the student to verify, refine, and optimize teacher knowledge through dynamic environmental interaction, potentially achieving more robust decision-making than the teacher itself. Extensive experiments on NGSIM and highD datasets demonstrate 6.2x parameter compression and 3.7x inference speedup while maintaining state-of-the-art accuracy, establishing a new paradigm for deploying sophisticated reasoning models under resource constraints. 

---
# Inductive Reasoning for Temporal Knowledge Graphs with Emerging Entities 

**Authors**: Ze Zhao, Yuhui He, Lyuwen Wu, Gu Tang, Bin Lu, Xiaoying Gan, Luoyi Fu, Xinbing Wang, Chenghu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.10164)  

**Abstract**: Reasoning on Temporal Knowledge Graphs (TKGs) is essential for predicting future events and time-aware facts. While existing methods are effective at capturing relational dynamics, their performance is limited by a closed-world assumption, which fails to account for emerging entities not present in the training. Notably, these entities continuously join the network without historical interactions. Empirical study reveals that emerging entities are widespread in TKGs, comprising roughly 25\% of all entities. The absence of historical interactions of these entities leads to significant performance degradation in reasoning tasks. Whereas, we observe that entities with semantic similarities often exhibit comparable interaction histories, suggesting the presence of transferable temporal patterns. Inspired by this insight, we propose TransFIR (Transferable Inductive Reasoning), a novel framework that leverages historical interaction sequences from semantically similar known entities to support inductive reasoning. Specifically, we propose a codebook-based classifier that categorizes emerging entities into latent semantic clusters, allowing them to adopt reasoning patterns from similar entities. Experimental results demonstrate that TransFIR outperforms all baselines in reasoning on emerging entities, achieving an average improvement of 28.6% in Mean Reciprocal Rank (MRR) across multiple datasets. The implementations are available at this https URL. 

---
# SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding 

**Authors**: Jehyeon Bang, Eunyeong Cho, Ranggi Hwang, Jinha Chung, Minsoo Rhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10152)  

**Abstract**: The Mixture-of-Experts (MoE) architecture has emerged as a promising approach to mitigate the rising computational costs of large language models (LLMs) by selectively activating parameters. However, its high memory requirements and sub-optimal parameter efficiency pose significant challenges for efficient deployment. Although CPU-offloaded MoE inference systems have been proposed in the literature, they offer limited efficiency, particularly for large batch sizes. In this work, we propose SpecMoE, a memory-efficient MoE inference system based on our self-assisted speculative decoding algorithm. SpecMoE demonstrates the effectiveness of applying speculative decoding to MoE inference without requiring additional model training or fine-tuning. Our system improves inference throughput by up to $4.30\times$, while significantly reducing bandwidth requirements of both memory and interconnect on memory-constrained systems. 

---
# Learning from Emptiness: De-biasing Listwise Rerankers with Content-Agnostic Probability Calibration 

**Authors**: Hang Lv, Hongchao Gu, Ruiqing Yang, Liangyue Li, Zulong Chen, Defu Lian, Hao Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.10150)  

**Abstract**: Generative listwise reranking leverages global context for superior retrieval but is plagued by intrinsic position bias, where models exhibit structural sensitivity to input order independent of relevance. Existing mitigations present a dilemma: inference-time aggregation incurs prohibitive latency, while training-based methods often fail to eradicate ingrained priors, particularly in compact models. To resolve this dilemma, we propose CapCal (Content-Agnostic Probability Calibration), a training-free framework that mechanically decouples positional bias from ranking decisions. By estimating the bias distribution via content-free placeholders, CapCal rectifies output logits through an entropy-adaptive contrastive mechanism. Evaluations across 10 benchmarks confirm that CapCal achieves superior performance among training-free methods while preserving single-pass efficiency. Notably, it unlocks the latent potential of lightweight models (e.g., 0.6B), delivering absolute NDCG gains exceeding 10 points and outperforming both permutation-based aggregation and data-augmentation baselines. 

---
# Trust Your Memory: Verifiable Control of Smart Homes through Reinforcement Learning with Multi-dimensional Rewards 

**Authors**: Kai-Yuan Guo, Jiang Wang, Renjie Zhao, Tianyi Wang, Wandong Mao, Yu Gao, Mou Xiao Feng, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10110)  

**Abstract**: Large Language Models (LLMs) have become a key foundation for enabling personalized smart home experiences. While existing studies have explored how smart home assistants understand user queries to control devices in real time, their ability to perform memory-driven device control remains challenging from both evaluation and methodological perspectives. In terms of evaluation, existing benchmarks either focus on immediate device control or general open-domain memory retrieval tasks, and therefore cannot effectively evaluate a model's ability to perform memory-driven device control. Methodologically, while memory-driven device control can be approached using Reinforcement Learning, conventional RL methods generally rely on outcome-based supervision (i.e., whether the final task is achieved). This lack of intermediate feedback can lead to sub-optimal performance or local failures in fine-grained memory management tasks (adding, updating, deleting, and utilizing). To address these issues, we first release MemHomeLife, built from anonymized real-world long-term user interaction logs. To enable more fine-grained evaluation of different memory-related subtasks, we further construct MemHome, the first benchmark designed to systematically evaluate memory-driven device control in smart home scenarios. 

---
# Ontological Trajectory Forecasting via Finite Semigroup Iteration and Lie Algebra Approximation in Geopolitical Knowledge Graphs 

**Authors**: Qihang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10087)  

**Abstract**: We present EL-DRUIN, an ontological reasoning system for geopolitical intelligence analysis that combines formal ontology, finite semigroup algebra, and Lie algebra approximation to forecast long-run relationship trajectories. Current LLM-based political analysis systems operate as summarisation engines, producing outputs bounded by textual pattern matching. EL-DRUIN departs from this paradigm by modelling geopolitical relationships as states in a finite set of named Dynamic Patterns, composing patterns via a semigroup operation whose structure constants are defined by an explicit composition table, and embedding each pattern as a vector in an 8-dimensional semantic Lie algebra space. Forward simulation iterates this semigroup operation, yielding reachable pattern sets at each discrete timestep; convergence to idempotent absorbing states (fixed points of the composition) constitutes the predicted long-run attractor. Bayesian posterior weights combine ontology-derived confidence priors with a Lie similarity term measuring the cosine similarity between the vector sum of composing patterns and the target pattern vector, providing interpretable, calibrated probabilities that are not self-reported by a language model. Bifurcation points -- steps at which two candidate attractors have near-equal posterior mass -- are detected and exposed to downstream analysis. We demonstrate the framework on six geopolitical scenarios including US-China technology decoupling and the Taiwan Strait military coercion trajectory. The architecture is publicly available as an open-source system with a Streamlit frontend exposing full computation traces, Bayesian posterior breakdowns, and 8D ontological state vectors. 

---
# Learning Hierarchical and Geometry-Aware Graph Representations for Text-to-CAD 

**Authors**: Shengjie Gong, Wenjie Peng, Hongyuan Chen, Gangyu Zhang, Yunqing Hu, Huiyuan Zhang, Shuangping Huang, Tianshui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.10075)  

**Abstract**: Text-to-CAD code generation is a long-horizon task that translates textual instructions into long sequences of interdependent operations. Existing methods typically decode text directly into executable code (e.g., bpy) without explicitly modeling assembly hierarchy or geometric constraints, which enlarges the search space, accumulates local errors, and often causes cascading failures in complex assemblies. To address this issue, we propose a hierarchical and geometry-aware graph as an intermediate representation. The graph models multi-level parts and components as nodes and encodes explicit geometric constraints as edges. Instead of mapping text directly to code, our framework first predicts structure and constraints, then conditions action sequencing and code generation, thereby improving geometric fidelity and constraint satisfaction. We further introduce a structure-aware progressive curriculum learning strategy that constructs graded tasks through controlled structural edits, explores the model's capability boundary, and synthesizes boundary examples for iterative training. In addition, we build a 12K dataset with instructions, decomposition graphs, action sequences, and bpy code, together with graph- and constraint-oriented evaluation metrics. Extensive experiments show that our method consistently outperforms existing approaches in both geometric fidelity and accurate satisfaction of geometric constraints. 

---
# LoopGuard: Breaking Self-Reinforcing Attention Loops via Dynamic KV Cache Intervention 

**Authors**: Dongjie Xu, Hao Wu, Weijie Shi, Yue Cui, Yuanjun Liu, Jiawei Li, Haolun Ma, An Liu, Jia Zhu, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10044)  

**Abstract**: Through systematic experiments on long-context generation, we observe a damaging failure mode in which decoding can collapse into persistent repetition loops. We find that this degeneration is driven by collapsed attention patterns, where a subset of heads locks onto a narrow suffix of the history, and is further stabilized by inference-time KV cache reuse. Crucially, since many existing KV cache policies rely on attention-based importance, this collapse can produce spuriously high scores for repetitive tokens, causing cache management to inadvertently amplify repetition. To study this phenomenon in a controlled and reproducible manner, we introduce LoopBench, a benchmark with explicit loop-inducing conditions and loop-oriented metrics that quantify repetition severity and generation instability beyond downstream task scores. Building on these insights, we propose LoopGuard, a lightweight, plug-in KV cache guard that detects loop onset online and disrupts the feedback cycle by pruning repetitive tail spans under a fixed cache budget. Experiments on LoopBench show that LoopGuard reduces loop incidence by over 90 percentage points, while restoring output diversity and reducing token waste. 

---
# AI Achieves a Perfect LSAT Score 

**Authors**: Bonmu Ku  

**Link**: [PDF](https://arxiv.org/pdf/2604.10034)  

**Abstract**: This paper reports the first documented instance of a language model achieving a perfect score on an officially disclosed Law School Admission Test (LSAT). Controlled experiments on eight reasoning models show that varying the prompt, shuffling answer choices, and sampling multiple responses have no meaningful effect as drivers of performance. Ablating the thinking phase that models generate before answering, however, lowers frontier accuracy by up to 8 percentage points, predominantly in logical reasoning. Distilled models produce full thinking traces in the same format yet plateau far below frontier performance. A pilot process reward model fine-tuned via QLoRA on official LSAT explanations narrows this gap through Best-of-5 selection, with gains again predominantly in logical reasoning. The gatekeeper of elite legal education since 1948, the LSAT has not merely been passed but answered without a single error by models that reason. The upper bound of the cognitive capacities it has tested is no longer exclusive to human cognition. 

---
# FinTrace: Holistic Trajectory-Level Evaluation of LLM Tool Calling for Long-Horizon Financial Tasks 

**Authors**: Yupeng Cao, Haohang Li, Weijin Liu, Wenbo Cao, Anke Xu, Lingfei Qian, Xueqing Peng, Minxue Tang, Zhiyuan Yao, Jimin Huang, K.P. Subbalakshmi, Zining Zhu, Jordan W. Suchow, Yangyang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10015)  

**Abstract**: Recent studies demonstrate that tool-calling capability enables large language models (LLMs) to interact with external environments for long-horizon financial tasks. While existing benchmarks have begun evaluating financial tool calling, they focus on limited scenarios and rely on call-level metrics that fail to capture trajectory-level reasoning quality. To address this gap, we introduce FinTrace, a benchmark comprising 800 expert-annotated trajectories spanning 34 real-world financial task categories across multiple difficulty levels. FinTrace employs a rubric-based evaluation protocol with nine metrics organized along four axes -- action correctness, execution efficiency, process quality, and output quality -- enabling fine-grained assessment of LLM tool-calling behavior. Our evaluation of 13 LLMs reveals that while frontier models achieve strong tool selection, all models struggle with information utilization and final answer quality, exposing a critical gap between invoking the right tools and reasoning effectively over their outputs. To move beyond diagnosis, we construct FinTrace-Training, the first trajectory-level preference dataset for financial tool-calling, containing 8,196 curated trajectories with tool-augmented contexts and preference pairs. We fine-tune Qwen-3.5-9B using supervised fine-tuning followed by direct preference optimization (DPO) and show that training on FinTrace-Training consistently improves intermediate reasoning metrics, with DPO more effectively suppressing failure modes. However, end-to-end answer quality remains a bottleneck, indicating that trajectory-level improvements do not yet fully propagate to final output quality. 

---
# New Hybrid Fine-Tuning Paradigm for LLMs: Algorithm Design and Convergence Analysis Framework 

**Authors**: Shaocong Ma, Peiran Yu, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09940)  

**Abstract**: Fine-tuning Large Language Models (LLMs) typically involves either full fine-tuning, which updates all model parameters, or Parameter-Efficient Fine-Tuning (PEFT), which adjusts a small subset of parameters. However, both approaches have inherent limitations: full fine-tuning is computationally expensive, while PEFT often struggles to learn new knowledge and exhibits suboptimal performance. To overcome these issues, we propose a novel hybrid fine-tuning approach that jointly updates both LLMs and PEFT modules using a combination of zeroth-order and first-order optimization methods. To analyze our new algorithm, we develop a theoretical framework centered on the concept of hybrid smoothness condition, which accounts for the heterogeneous nature of the optimization landscape in joint LLM and PEFT training. We derive a rigorous convergence analysis for the convergence of reshuffling-type SGD algorithm under multiple learning rates and demonstrate its effectiveness through extensive empirical studies across various downstream tasks and model architectures. On the practical side, our results demonstrate consistent performance improvement, making the approach a viable solution for large-scale language model fine-tuning. 

---
# HealthAdminBench: Evaluating Computer-Use Agents on Healthcare Administration Tasks 

**Authors**: Suhana Bedi, Ryan Welch, Ethan Steinberg, Michael Wornow, Taeil Matthew Kim, Haroun Ahmed, Peter Sterling, Bravim Purohit, Qurat Akram, Angelic Acosta, Esther Nubla, Pritika Sharma, Michael A. Pfeffer, Sanmi Koyejo, Nigam H. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2604.09937)  

**Abstract**: Healthcare administration accounts for over $1 trillion in annual spending, making it a promising target for LLM-based computer-use agents (CUAs). While clinical applications of LLMs have received significant attention, no benchmark exists for evaluating CUAs on end-to-end administrative workflows. To address this gap, we introduce HealthAdminBench, a benchmark comprising four realistic GUI environments: an EHR, two payer portals, and a fax system, and 135 expert-defined tasks spanning three administrative task types: Prior Authorization, Appeals and Denials Management, and Durable Medical Equipment (DME) Order Processing. Each task is decomposed into fine-grained, verifiable subtasks, yielding 1,698 evaluation points. We evaluate seven agent configurations under multiple prompting and observation settings and find that, despite strong subtask performance, end-to-end reliability remains low: the best-performing agent (Claude Opus 4.6 CUA) achieves only 36.3 percent task success, while GPT-5.4 CUA attains the highest subtask success rate (82.8 percent). These results reveal a substantial gap between current agent capabilities and the demands of real-world administrative workflows. HealthAdminBench provides a rigorous foundation for evaluating progress toward safe and reliable automation of healthcare administrative workflows. 

---
# GLEaN: A Text-to-image Bias Detection Approach for Public Comprehension 

**Authors**: Bochu Ding, Brinnae Bent, Augustus Wendell  

**Link**: [PDF](https://arxiv.org/pdf/2604.09923)  

**Abstract**: Text-to-image (T2I) models, and their encoded biases, increasingly shape the visual media the public encounters. While researchers have produced a rich body of work on bias measurement, auditing, and mitigation in T2I systems, those methods largely target technical stakeholders, leaving a gap in public legibility. We introduce GLEaN (Generative Likeness Evaluation at N-Scale), a portrait-based explainability pipeline designed to make T2I model biases visually understandable to a broad audience. GLEaN comprises three stages: automated large-scale image generation from identity prompts, facial landmark-based filtering and spatial alignment, and median-pixel composition that distills a model's central tendency into a single representative portrait. The resulting composites require no statistical background to interpret; a viewer can see, at a glance, who a model 'imagines' when prompted with 'a doctor' versus a 'felon.' We demonstrate GLEaN on Stable Diffusion XL across 40 social and occupational identity prompts, producing composites that reproduce documented biases and surface new associations between skin tone and predicted emotion. We find in a between-subjects user study (N = 291) that GLEaN portraits communicate biases as effectively as conventional data tables, but require significantly less viewing time. Because the method relies solely on generated outputs, it can also be replicated on any black-box and closed-weight systems without access to model internals. GLEaN offers a scalable, model-agnostic approach to bias explainability, purpose-built for public comprehension, and is publicly available at this https URL. 

---
# In-situ process monitoring for defect detection in wire-arc additive manufacturing: an agentic AI approach 

**Authors**: Pallock Halder, Satyajit Mojumder  

**Link**: [PDF](https://arxiv.org/pdf/2604.09889)  

**Abstract**: AI agents are being increasingly deployed across a wide range of real-world applications. In this paper, we propose an agentic AI framework for in-situ process monitoring for defect detection in wire-arc additive manufacturing (WAAM). The autonomous agent leverages a WAAM process monitoring dataset and a trained classification tool to build AI agents and uses a large language model (LLM) for in-situ process monitoring decision-making for defect detection. A processing agent is developed based on welder process signals, such as current and voltage, and a monitoring agent is developed based on acoustic data collected during the process. Both agents are tasked with identifying porosity defects from processing and monitoring signals, respectively. Ground truth X-ray computed tomography (XCT) data are used to develop classification tools for both the processing and monitoring agents. Furthermore, a multi-agent framework is demonstrated in which the processing and monitoring agents are orchestrated together for parallel decision-making on the given task of defect classification. Evaluation metrics are proposed to determine the efficacy of both individual agents, the combined single-agent, and the coordinated multi-agent system. The multi-agent configuration outperforms all individual-agent counterparts, achieving a decision accuracy of 91.6% and an F1 score of 0.821 on decided runs, across 15 independent runs, and a reasoning quality score of 3.74 out of 5. These in-situ process monitoring agents hold significant potential for autonomous real-time process monitoring and control toward building qualified parts for WAAM and other additive manufacturing processes. 

---
# What do your logits know? (The answer may surprise you!) 

**Authors**: Masha Fedzechkina, Eleonora Gualdoni, Rita Ramos, Sinead Williamson  

**Link**: [PDF](https://arxiv.org/pdf/2604.09885)  

**Abstract**: Recent work has shown that probing model internals can reveal a wealth of information not apparent from the model generations. This poses the risk of unintentional or malicious information leakage, where model users are able to learn information that the model owner assumed was inaccessible. Using vision-language models as a testbed, we present the first systematic comparison of information retained at different "representational levels'' as it is compressed from the rich information encoded in the residual stream through two natural bottlenecks: low-dimensional projections of the residual stream obtained using tuned lens, and the final top-k logits most likely to impact model's answer. We show that even easily accessible bottlenecks defined by the model's top logit values can leak task-irrelevant information present in an image-based query, in some cases revealing as much information as direct projections of the full residual stream. 

---
# Evolutionary Token-Level Prompt Optimization for Diffusion Models 

**Authors**: Domício Pereira Neto, João Correia, Penousal Machado  

**Link**: [PDF](https://arxiv.org/pdf/2604.09861)  

**Abstract**: Text-to-image diffusion models exhibit strong generative performance but remain highly sensitive to prompt formulation, often requiring extensive manual trial and error to obtain satisfactory results. This motivates the development of automated, model-agnostic prompt optimization methods that can systematically explore the conditioning space beyond conventional text rewriting. This work investigates the use of a Genetic Algorithm (GA) for prompt optimization by directly evolving the token vectors employed by CLIP-based diffusion models. The GA optimizes a fitness function that combines aesthetic quality, measured by the LAION Aesthetic Predictor V2, with prompt-image alignment, assessed via CLIPScore. Experiments on 36 prompts from the Parti Prompts (P2) dataset show that the proposed approach outperforms the baseline methods, including Promptist and random search, achieving up to a 23.93% improvement in fitness. Overall, the method is adaptable to image generation models with tokenized text encoders and provides a modular framework for future extensions, the limitations and prospects of which are discussed. 

---
# Instructing LLMs to Negotiate using Reinforcement Learning with Verifiable Rewards 

**Authors**: Shuze Daniel Liu, Claire Chen, Jiabao Sean Xiao, Lei Lei, Yuheng Zhang, Yisong Yue, David Simchi-Levi  

**Link**: [PDF](https://arxiv.org/pdf/2604.09855)  

**Abstract**: The recent advancement of Large Language Models (LLMs) has established their potential as autonomous interactive agents. However, they often struggle in strategic games of incomplete information, such as bilateral price negotiation. In this paper, we investigate if Reinforcement Learning from Verifiable Rewards (RLVR) can effectively teach LLMs to negotiate. Specifically, we explore the strategic behaviors that emerge during the learning process. We introduce a framework that trains a mid-sized buyer agent against a regulated LLM seller across a wide distribution of real-world products. By grounding reward signals directly in the maximization of economic surplus and strict adherence to private budget constraints, we reveal a novel four-phase strategic evolution. The agent progresses from naive bargaining to using aggressive starting prices, moves through a phase of deadlock, and ultimately develops sophisticated persuasive skills. Our results demonstrate that this verifiable training allows a 30B agent to significantly outperform frontier models over ten times its size in extracting surplus. Furthermore, the trained agent generalizes robustly to stronger counterparties unseen during training and remains effective even when facing hostile, adversarial seller personas. 

---
# MEMENTO: Teaching LLMs to Manage Their Own Context 

**Authors**: Vasilis Kontonis, Yuchen Zeng, Shivam Garg, Lingjiao Chen, Hao Tang, Ziyan Wang, Ahmed Awadallah, Eric Horvitz, John Langford, Dimitris Papailiopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2604.09852)  

**Abstract**: Reasoning models think in long, unstructured streams with no mechanism for compressing or organizing their own intermediate state. We introduce MEMENTO: a method that teaches models to segment reasoning into blocks, compress each block into a memento, i.e., a dense state summary, and reason forward by attending only to mementos, reducing context, KV cache, and compute. To train MEMENTO models, we release OpenMementos, a public dataset of 228K reasoning traces derived from OpenThoughts-v3, segmented and annotated with intermediate summaries. We show that a two-stage SFT recipe on OpenMementos is effective across different model families (Qwen3, Phi-4, Olmo 3) and scales (8B--32B parameters). Trained models maintain strong accuracy on math, science, and coding benchmarks while achieving ${\sim}2.5\times$ peak KV cache reduction. We extend vLLM to support our inference method, achieving ${\sim}1.75\times$ throughput improvement while also enabling us to perform RL and further improve accuracy. Finally, we identify a dual information stream: information from each reasoning block is carried both by the memento text and by the corresponding KV states, which retain implicit information from the original block. Removing this channel drops accuracy by 15\,pp on AIME24. 

---
# Steered LLM Activations are Non-Surjective 

**Authors**: Aayush Mishra, Daniel Khashabi, Anqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09839)  

**Abstract**: Activation steering is a popular white-box control technique that modifies model activations to elicit an abstract change in output behavior. It has also become a standard tool in interpretability (e.g., probing truthfulness, or translating activations into human-readable explanations and safety research (e.g., studying jailbreakability). However, it is unclear whether steered activation states are realizable by any textual prompt. In this work, we cast this question as a surjectivity problem: for a fixed model, does every steered activation admit a pre-image under the model's natural forward pass? Under practical assumptions, we prove that activation steering pushes the residual stream off the manifold of states reachable from discrete prompts. Almost surely, no prompt can reproduce the same internal behavior induced by steering. We also illustrate this finding empirically across three widely used LLMs. Our results establish a formal separation between white-box steerability and black-box prompting. We therefore caution against interpreting the ease and success of activation steering as evidence of prompt-based interpretability or vulnerability, and argue for evaluation protocols that explicitly decouple white-box and black-box interventions. 

---
# COMPOSITE-Stem 

**Authors**: Kyle Waters, Lucas Nuzzi, Tadhg Looram, Alessandro Tomasiello, Ariel Ghislain Kemogne Kamdoum, Bikun Li, Damien Sileo, Egor Kretov, Francesco Fournier-Facio, Georgios Soloupis, Haile Kassahun, Hew Wolff, Jiaqi Cai, Lianghui Li, Marc Roth, Mohinder Naiya, Naixu Guo, Qicheng Tang, Richard Wheeler, Samuele Sala, Serguei Popov, Steven Dillman, Yuqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09836)  

**Abstract**: AI agents hold growing promise for accelerating scientific discovery; yet, a lack of frontier evaluations hinders adoption into real workflows. Expert-written benchmarks have proven effective at measuring AI reasoning, but most at this stage have become saturated and only measure performance on constrained outputs. To help address this gap, we introduce COMPOSITE-STEM, a benchmark of 70 expert-written tasks in physics, biology, chemistry, and mathematics, curated by doctoral-level researchers. Our benchmark combines exact-match grading and criterion-based rubrics with an LLM-as-a-jury grading protocol, allowing more flexible assessment of scientifically meaningful outputs. Using an adapted multimodal Terminus-2 agent harness within the Harbor agentic evaluation framework, we evaluate four frontier models. The top-performing model achieves 21%, demonstrating that COMPOSITE-STEM captures capabilities beyond current agent reach. All tasks are open-sourced with contributor permission to support reproducibility and to promote additional research towards AI's acceleration of scientific progress in these domains. 

---
# EE-MCP: Self-Evolving MCP-GUI Agents via Automated Environment Generation and Experience Learning 

**Authors**: Tiantian He, Yihang Chen, Keyue Jiang, Ka Yiu Lee, Kaiwen Zhou, Kun Shao, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09815)  

**Abstract**: Computer-use agents that combine GUI interaction with structured API calls via the Model Context Protocol (MCP) show promise for automating software tasks. However, existing approaches lack a principled understanding of how agents should balance these two modalities and how to enable iterative self-improvement across diverse applications. We formulate MCP-GUI interplay as a unified hybrid policy learning problem where the agent learns when each modality provides complementary advantages, and show that distillation and experience augmentation target fundamentally different failure modes - requiring application-aware mechanism selection. Built on this formulation, we propose a self-evolving framework with a fully automatic pipeline that orchestrates automatic environment generation and validation, trajectory collection, gap-driven task synthesis, and quality-filtered training - all without manual intervention. A key innovation is our experience bank, which accumulates LLM-learned rules from trajectory comparison, enabling inference-time improvement without fine-tuning. Systematic \textbf{cross-application analysis} across three desktop applications reveals that the optimal strategy depends on MCP-GUI composition: distillation achieves 77.8\% pass rate on MCP-dominant tasks (+17.8pp), while the experience bank excels on GUI-intensive tasks (+10.0pp). 

---
# Controllable and Verifiable Tool-Use Data Synthesis for Agentic Reinforcement Learning 

**Authors**: Siyuan Xu, Shiyang Li, Xin Liu, Tianyi Liu, Yixiao Li, Zhan Shi, Zixuan Zhang, Zilong Wang, Qingyu Yin, Jianshu Chen, Tuo Zhao, Bing Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.09813)  

**Abstract**: Existing synthetic tool-use corpora are primarily designed for offline supervised fine-tuning, yet reinforcement learning (RL) requires executable environments that support reward-checkable online rollouts. We propose COVERT, a two-stage pipeline that first generates reliable base tool-use trajectories through self-evolving synthesis with multi-level validation, and then applies oracle-preserving augmentations that systematically increase environmental complexity. These augmentations introduce distractor tools, indirect or ambiguous user queries, and noisy, multi-format, or erroneous tool outputs, while strictly preserving oracle tool calls and final answers as ground truth. This design enables automatic reward computation via reference matching for standard cases and lightweight judge-assisted verification for special behaviors such as error detection, supporting RL optimization of tool-calling policies. On Qwen2.5-Instruct-14B, COVERT-RL improves overall accuracy on BFCL v3 from 56.5 to 59.9 and on ACEBench from 53.0 to 59.3, with minimal regressions on general-ability benchmarks; when stacked on SFT, it further reaches 62.1 and 61.8, confirming additive gains. These results suggest that oracle-preserving synthetic environments offer a practical RL refinement stage, complementary to SFT, for improving tool-use robustness under ambiguity and unreliable tool feedback. 

---
# Pioneer Agent: Continual Improvement of Small Language Models in Production 

**Authors**: Dhruv Atreja, Julia White, Nikhil Nayak, Kelton Zhang, Henrijs Princis, George Hurn-Maloney, Ash Lewis, Urchade Zaratiana  

**Link**: [PDF](https://arxiv.org/pdf/2604.09791)  

**Abstract**: Small language models are attractive for production deployment due to their low cost, fast inference, and ease of specialization. However, adapting them to a specific task remains a challenging engineering loop, driven not by training itself but by surrounding decisions: data curation, failure diagnosis, regression avoidance, and iteration control. We present Pioneer Agent, a closed-loop system that automates this lifecycle. In cold-start mode, given only a natural-language task description, the agent acquires data, constructs evaluation sets, and iteratively trains models by jointly optimizing data, hyperparameters, and learning strategy. In production mode, given a deployed model with labeled failures, it diagnoses error patterns, constructs targeted training data, and retrains under explicit regression constraints. To evaluate this setting, we introduce AdaptFT-Bench, a benchmark of synthetic inference logs with progressively increasing noise, designed to test the full adaptation loop: diagnosis, curriculum synthesis, retraining, and verification. Across eight cold-start benchmarks spanning reasoning, math, code generation, summarization, and classification, Pioneer Agent improves over base models by 1.6-83.8 points. On AdaptFT-Bench, it improves or preserves performance in all seven scenarios, while naive retraining degrades by up to 43 points. On two production-style deployments built from public benchmark tasks, it raises intent classification from 84.9% to 99.3% and Entity F1 from 0.345 to 0.810. Beyond performance gains, the agent often discovers effective training strategies, including chain-of-thought supervision, task-specific optimization, and quality-focused data curation, purely from downstream feedback. 

---
# The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily Domain Expertise 

**Authors**: Xi Wang, Soufiane Hayou, Eric Nalisnick  

**Link**: [PDF](https://arxiv.org/pdf/2604.09780)  

**Abstract**: Mixture of Experts (MoEs) are now ubiquitous in large language models, yet the mechanisms behind their "expert specialization" remain poorly understood. We show that, since MoE routers are linear maps, hidden state similarity is both necessary and sufficient to explain expert usage similarity, and specialization is therefore an emergent property of the representation space, not of the routing architecture itself. We confirm this at both token and sequence level across five pre-trained models. We additionally prove that load-balancing loss suppresses shared hidden state directions to maintain routing diversity, which might provide a theoretical explanation for specialization collapse under less diverse data, e.g. small batch. Despite this clean mechanistic account, we find that specialization patterns in pre-trained MoEs resist human interpretation: expert overlap between different models answering the same question is no higher than between entirely different questions ($\sim$60\%); prompt-level routing does not predict rollout-level routing; and deeper layers exhibit near-identical expert activation across semantically unrelated inputs, especially in reasoning models. We conclude that, while the efficiency perspective of MoEs is well understood, understanding expert specialization is at least as hard as understanding LLM hidden state geometry, a long-standing open problem in the literature. 

---
# Tipiano: Cascaded Piano Hand Motion Synthesis via Fingertip Priors 

**Authors**: Joonhyung Bae, Kirak Kim, Hyeyoon Cho, Sein Lee, Yoon-Seok Choi, Hyeon Hur, Gyubin Lee, Akira Maezawa, Satoshi Obata, Jonghwa Park, Jaebum Park, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2604.09692)  

**Abstract**: Synthesizing realistic piano hand motions requires both precision and naturalness. Physics-based methods achieve precision but produce stiff motions; data-driven models learn natural dynamics but struggle with positional accuracy. Piano motion exhibits a natural hierarchy: fingertip positions are nearly deterministic given piano geometry and fingering, while wrist and intermediate joints offer stylistic freedom. We present [OURS], a four-stage framework exploiting this hierarchy: (1) statistics-based fingertip positioning, (2) FiLM-conditioned trajectory refinement, (3) wrist estimation, and (4) STGCN-based pose synthesis. We contribute expert-annotated fingerings for the FürElise dataset (153 pieces, ~10 hours). Experiments demonstrate F1 = 0.910, substantially outperforming diffusion baselines (F1 = 0.121), with user study (N=41) confirming quality approaching motion capture. Expert evaluation by professional pianists (N=5) identified anticipatory motion as the key remaining gap, providing concrete directions for future improvement. 

---
# Belief-Aware VLM Model for Human-like Reasoning 

**Authors**: Anshul Nayak, Shahil Shaik, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09686)  

**Abstract**: Traditional neural network models for intent inference rely heavily on observable states and struggle to generalize across diverse tasks and dynamic environments. Recent advances in Vision Language Models (VLMs) and Vision Language Action (VLA) models introduce common-sense reasoning through large-scale multimodal pretraining, enabling zero-shot performance across tasks. However, these models still lack explicit mechanisms to represent and update belief, limiting their ability to reason like humans or capture the evolving human intent over long-horizon. To address this, we propose a belief-aware VLM framework that integrates retrieval-based memory and reinforcement learning. Instead of learning an explicit belief model, we approximate belief using a vector-based memory that retrieves relevant multimodal context, which is incorporated into the VLM for reasoning. We further refine decision-making using a reinforcement learning policy over the VLM latent space. We evaluate our approach on publicly available VQA datasets such as HD-EPIC and demonstrate consistent improvements over zero-shot baselines, highlighting the importance of belief-aware reasoning. 

---
# How LLMs Might Think 

**Authors**: Joseph Gottlieb, Ethan Kemp, Matthew Trager  

**Link**: [PDF](https://arxiv.org/pdf/2604.09674)  

**Abstract**: Do large language models (LLMs) think? Daniel Stoljar and Zhihe Vincent Zhang have recently developed an argument from rationality for the claim that LLMs do not think. We contend, however, that the argument from rationality not only falters, but leaves open an intriguing possibility: that LLMs engage only in arational, associative forms of thinking, and have purely associative minds. Our positive claim is that if LLMs think at all, they likely think precisely in this manner. 

---
# Competing with AI Scientists: Agent-Driven Approach to Astrophysics Research 

**Authors**: Thomas Borrett, Licong Xu, Andy Nilipour, Boris Bolliet, Sebastien Pierre, Erwan Allys, Celia Lecat, Biwei Dai, Po-Wen Chang, Wahid Bhimji  

**Link**: [PDF](https://arxiv.org/pdf/2604.09621)  

**Abstract**: We present an agent-driven approach to the construction of parameter inference pipelines for scientific data analysis. Our method leverages a multi-agent system, Cmbagent (the analysis system of the AI scientist Denario), in which specialized agents collaborate to generate research ideas, write and execute code, evaluate results, and iteratively refine the overall pipeline. As a case study, we apply this approach to the FAIR Universe Weak Lensing Uncertainty Challenge, a competition under time constraints focused on robust cosmological parameter inference with realistic observational uncertainties. While the fully autonomous exploration initially did not reach expert-level performance, the integration of human intervention enabled our agent-driven workflow to achieve a first-place result in the challenge. This demonstrates that semi-autonomous agentic systems can compete with, and in some cases surpass, expert solutions. We describe our workflow in detail, including both the autonomous and semi-autonomous exploration by Cmbagent. Our final inference pipeline utilizes parameter-efficient convolutional neural networks, likelihood calibration over a known parameter grid, and multiple regularization techniques. Our results suggest that agent-driven research workflows can provide a scalable framework to rapidly explore and construct pipelines for inference problems. 

---
# AdaQE-CG: Adaptive Query Expansion for Web-Scale Generative AI Model and Data Card Generation 

**Authors**: Haoxuan Zhang, Ruochi Li, Zhenni Liang, Mehri Sattari, Phat Vo, Collin Qu, Ting Xiao, Junhua Ding, Yang Zhang, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.09617)  

**Abstract**: Transparent and standardized documentation is essential for building trustworthy generative AI (GAI) systems. However, existing automated methods for generating model and data cards still face three major challenges: (i) static templates, as most systems rely on fixed query templates that cannot adapt to diverse paper structures or evolving documentation requirements; (ii) information scarcity, since web-scale repositories such as Hugging Face often contain incomplete or inconsistent metadata, leading to missing or noisy information; and (iii) lack of benchmarks, as the absence of standardized datasets and evaluation protocols hinders fair and reproducible assessment of documentation quality. To address these limitations, we propose AdaQE-CG, an Adaptive Query Expansion for Card Generation framework that combines dynamic information extraction with cross-card knowledge transfer. Its Intra-Paper Extraction via Context-Aware Query Expansion (IPE-QE) module iteratively refines extraction queries to recover richer and more complete information from scientific papers and repositories, while its Inter-Card Completion using the MetaGAI Pool (ICC-MP) module fills missing fields by transferring semantically relevant content from similar cards in a curated dataset. In addition, we introduce MetaGAI-Bench, the first large-scale, expert-annotated benchmark for evaluating GAI documentation. Comprehensive experiments across five quality dimensions show that AdaQE-CG substantially outperforms existing approaches, exceeds human-authored data cards, and approaches human-level quality for model cards. Code, prompts, and data are publicly available at: this https URL. 

---
# The Geometry of Knowing: From Possibilistic Ignorance to Probabilistic Certainty -- A Measure-Theoretic Framework for Epistemic Convergence 

**Authors**: Moriba Kemessia Jah  

**Link**: [PDF](https://arxiv.org/pdf/2604.09614)  

**Abstract**: This paper develops a measure-theoretic framework establishing when and how a possibilistic representation of incomplete knowledge contracts into a probabilistic representation of intrinsic stochastic variability. Epistemic uncertainty is encoded by a possibility distribution and its dual necessity measure, defining a credal set bounding all probability measures consistent with current evidence. As evidence accumulates, the credal set contracts. The epistemic collapse condition marks the transition: the Choquet integral converges to the Lebesgue integral over the unique limiting density. We prove this rigorously (Theorem 4.5), with all assumptions explicit and a full treatment of the non-consonant case. We introduce the aggregate epistemic width W, establish its axiomatic properties, provide a canonical normalization, and give a feasible online proxy resolving a circularity in prior formulations. Section 7 develops the dynamics of epistemic contraction: evidence induces compatibility, compatibility performs falsification, posterior possibility is the min-intersection of prior possibility and compatibility, and a credibility-directed flow governs support geometry contraction. This is not belief updating. It is knowledge contraction. Probability theory is the limiting geometry of that process. The UKF and ESPF solve different problems by different mechanisms. The UKF minimizes MSE, asserts truth, and requires a valid generative model. The ESPF minimizes maximum entropy and surfaces what evidence has not ruled out. When the world is Gaussian and the model valid, both reach the same estimate by entirely different routes -- convergent optimality, not hierarchical containment. We prove this (Theorem 9.1) and compare both on a 2-day, 877-step orbital tracking scenario. Both achieve 1-meter accuracy. The UKF is accurate but epistemically silent. The ESPF is accurate and epistemically honest. 

---
# Beyond Theory of Mind in Robotics 

**Authors**: Malte F. Jung  

**Link**: [PDF](https://arxiv.org/pdf/2604.09612)  

**Abstract**: Theory of Mind, the capacity to explain and predict behavior by inferring hidden mental states, has become the dominant paradigm for social interaction in robotics. Yet ToM rests on three assumptions that poorly capture how most social interaction actually unfolds: that meaning travels inside-out from hidden states to observable behavior; that understanding requires detached inference rather than participation; and that the meaning of behavior is fixed and available to a passive observer. Drawing on ethnomethodology, conversation analysis, and participatory sense-making, I argue that social meaning is not decoded from behavior but produced through moment-to-moment coordination between agents. This interactional foundation has direct implications for robot design: shifting from internal state modeling toward policies for sustaining coordination, from observer-based inference toward active participation, and from fixed behavioral meaning toward meaning potential stabilized through response. 

---
# General-purpose LLMs as Models of Human Driver Behavior: The Case of Simplified Merging 

**Authors**: Samir H.A. Mohammad, Wouter Mooi, Arkady Zgonnikov  

**Link**: [PDF](https://arxiv.org/pdf/2604.09609)  

**Abstract**: Human behavior models are essential as behavior references and for simulating human agents in virtual safety assessment of automated vehicles (AVs), yet current models face a trade-off between interpretability and flexibility. General-purpose large language models (LLMs) offer a promising alternative: a single model potentially deployable without parameter fitting across diverse scenarios. However, what LLMs can and cannot capture about human driving behavior remains poorly understood. We address this gap by embedding two general-purpose LLMs (OpenAI o3 and Google Gemini 2.5 Pro) as standalone, closed-loop driver agents in a simplified one-dimensional merging scenario and comparing their behavior against human data using quantitative and qualitative analyses. Both models reproduce human-like intermittent operational control and tactical dependencies on spatial cues. However, neither consistently captures the human response to dynamic velocity cues, and safety performance diverges sharply between models. A systematic prompt ablation study reveals that prompt components act as model-specific inductive biases that do not transfer across LLMs. These findings suggest that general-purpose LLMs could potentially serve as standalone, ready-to-use human behavior models in AV evaluation pipelines, but future research is needed to better understand their failure modes and ensure their validity as models of human driving behavior. 

---
# Unifying Ontology Construction and Semantic Alignment for Deterministic Enterprise Reasoning at Scale 

**Authors**: Hongyin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09608)  

**Abstract**: While enterprises amass vast quantities of data, much of it remains chaotic and effectively dormant, preventing decision-making based on comprehensive information. Existing neuro-symbolic approaches rely on disjoint pipelines and struggle with error propagation. We introduce the large ontology model (LOM), a unified framework that seamlessly integrates ontology construction, semantic alignment, and logical reasoning into a single end-to-end architecture. LOM employs a construct-align-reason (CAR) pipeline, leveraging its unified architecture across all three stages: it first autonomously constructs a domain-specific ontological universe from raw data, then aligns neural generation with this structural reality using a graph-aware encoder and reinforcement learning, and finally executes deterministic reasoning over the constructed topology, node attributes and relation types. We evaluate LOM on a comprehensive benchmark constructed from diverse real-world enterprise datasets. Experimental results demonstrate that LOM-4B achieves 88.8% accuracy in ontology completion and 94% in complex graph reasoning tasks, significantly outperforming state-of-the-art LLMs. These findings validate that autonomous logical construction is essential for achieving deterministic, enterprise-grade intelligence. 

---
# Evaluating Reliability Gaps in Large Language Model Safety via Repeated Prompt Sampling 

**Authors**: Keita Broadwater  

**Link**: [PDF](https://arxiv.org/pdf/2604.09606)  

**Abstract**: Traditional benchmarks for large language models (LLMs), such as HELM and AIR-BENCH, primarily assess safety risk through breadth-oriented evaluation across diverse tasks. However, real-world deployment often exposes a different class of risk: operational failures arising from repeated generations of the same prompt rather than broad task generalization. In high-stakes settings, response consistency and safety under repeated use are critical operational requirements. We introduce Accelerated Prompt Stress Testing (APST), a depth-oriented evaluation framework inspired by highly accelerated stress testing in reliability engineering. APST probes LLM behavior by repeatedly sampling identical prompts under controlled operational conditions, including temperature variation and prompt perturbation, to surface latent failure modes such as hallucinations, refusal inconsistency, and unsafe completions. Rather than treating failures as isolated events, APST characterizes them statistically as stochastic outcomes of repeated inference. We model observed safety failures using Bernoulli and binomial formulations to estimate per-inference failure probabilities, enabling quantitative comparison of operational risk across models and configurations. We apply APST to multiple instruction-tuned LLMs evaluated on AIR-BENCH 2024 derived safety and security prompts. While models exhibit similar performance under conventional single- or very-low-sample evaluation (N <= 3), repeated sampling reveals substantial variation in empirical failure probabilities across temperatures. These results demonstrate that shallow benchmark scores can obscure meaningful differences in reliability under sustained use. 

---
# LLMs for Text-Based Exploration and Navigation Under Partial Observability 

**Authors**: Stephan Sandfuchs, Maximilian Melchert, Jörg Frochte  

**Link**: [PDF](https://arxiv.org/pdf/2604.09604)  

**Abstract**: Exploration and goal-directed navigation in unknown layouts are central to inspection, logistics, and search-and-rescue. We ask whether large language models (LLMs) can function as \emph{text-only} controllers under partial observability -- without code execution, tools, or program synthesis. We introduce a reproducible benchmark with oracle localisation in fixed ASCII gridworlds: each step reveals only a local $5\times5$ window around the agent and the model must select one of \texttt{UP/RIGHT/DOWN/LEFT}. Nine contemporary LLMs ranging from open/proprietary, dense / Mixture of Experts and instruction- vs. reasoning-tuned are evaluated on two tasks across three layouts of increasing difficulty: \emph{Exploration} (maximising revealed cells) and \emph{Navigation} (reach the goal on the shortest path). The experimental results are evaluated on quantitative metrics including \emph{success rate}, \emph{efficiency} such as normalised coverage and \emph{path length} vs. oracle as well as qualitative analysis. Reasoning-tuned models reliably complete navigation across all layouts, yet remain less efficient than oracle paths. Few-shot demonstrations in the prompt chiefly help these Reasoning-tuned models by reducing invalid moves and shortening paths, while classic dense instruction models remain inconsistent. We observe characteristic action priors (UP/RIGHT) that can induce looping under partial observability. Overall, training regimen and test-time deliberation predict control ability better than raw parameter count. These findings suggest lightweight hybridisation with classical online planners as a practical route to deployable partial map systems. 

---
# From Scalars to Tensors: Declared Losses Recover Epistemic Distinctions That Neutrosophic Scalars Cannot Express 

**Authors**: Tony Mason  

**Link**: [PDF](https://arxiv.org/pdf/2604.09602)  

**Abstract**: Leyva-Vázquez and Smarandache (2025) demonstrated that neutrosophic T/I/F evaluation, where Truth, Indeterminacy, and Falsity are independent dimensions not constrained to sum to 1.0, which reveals "hyper-truth"' (T+I+F > 1.0) in 35% of complex epistemic cases evaluated by LLMs. We extend their work in two directions. First, we replicate and extend their experiment across five model families from five vendors (Anthropic, Meta, DeepSeek, Alibaba, Mistral), finding hyper-truth in 84% of unconstrained evaluations, which confirms the phenomenon is cross-vendor under our prompt protocol. Second, and more significantly, we identify a limitation of scalar T/I/F that their framework cannot address: models adopting an `"Absorption" position (T=0, I=1, F=0) produce identical scalar outputs for fundamentally different epistemic situations (paradox, ignorance, contingency), collapsing the very distinctions neutrosophic logic was designed to preserve. We demonstrate that extending the evaluation to include declared losses (structured descriptions of what the model cannot evaluate and why) substantially recovers these distinctions. Models producing identical scalars for paradox and ignorance produce nearly disjoint loss vocabularies (Jaccard similarity < 0.10 on loss description keywords), with domain-specific, severity-rated loss declarations that differentiate the nature of their uncertainty. This suggests that scalar T/I/F is a necessary but insufficient representation of epistemic state, and that tensor-structured output (scalars + losses) provides a more faithful model of LLM epistemic capabilities. 

---
# Hubble: An LLM-Driven Agentic Framework for Safe and Automated Alpha Factor Discovery 

**Authors**: Runze Shi, Shengyu Yan, Yuecheng Cai, Chengxi Lv  

**Link**: [PDF](https://arxiv.org/pdf/2604.09601)  

**Abstract**: Discovering predictive alpha factors in quantitative finance remains a formidable challenge due to the vast combinatorial search space and inherently low signal-to-noise ratios in financial data. Existing automated methods, particularly genetic programming, often produce complex, uninterpretable formulas prone to overfitting. We introduce Hubble, a closed-loop factor mining framework that leverages Large Language Models (LLMs) as intelligent search heuristics, constrained by a domain-specific operator language and an Abstract Syntax Tree (AST)-based execution sandbox. The framework evaluates candidate factors through a rigorous statistical pipeline encompassing cross-sectional Rank Information Coefficient (RankIC), annualized Information Ratio, and portfolio turnover. An evolutionary feedback mechanism returns top-performing factors and structured error diagnostics to the LLM, enabling iterative refinement across multiple generation rounds. In experiments conducted on a panel of 30 U.S. equities over 752 trading days, the system evaluated 181 syntactically valid factors from 122 unique candidates across three rounds, achieving a peak composite score of 0.827 with 100% computational stability. Our results demonstrate that combining LLM-driven generation with deterministic safety constraints yields an effective, interpretable, and reproducible approach to automated factor discovery. 

---
# CID-TKG: Collaborative Historical Invariance and Evolutionary Dynamics Learning for Temporal Knowledge Graph Reasoning 

**Authors**: Shuai-Long Lei, Xiaobin Zhu, Jiarui Liang, Guoxi Sun, Zhiyu Fang, Xu-Cheng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.09600)  

**Abstract**: Temporal knowledge graph (TKG) reasoning aims to infer future facts at unseen timestamps from temporally evolving entities and relations. Despite recent progress, existing approaches still suffer from inherent limitations due to their inductive biases, as they predominantly rely on time-invariant or weakly time-dependent structures and overlook the evolutionary dynamics. To overcome this limitation, we propose a novel collaborative learning framework for TKGR (dubbed CID-TKG) that integrates evolutionary dynamics and historical invariance semantics as an effective inductive bias for reasoning. Specifically, CID-TKG constructs a historical invariance graph to capture long-term structural regularities and an evolutionary dynamics graph to model short-term temporal transitions. Dedicated encoders are then employed to learn representations from each structure. To alleviate semantic discrepancies across the two structures, we decompose relations into view-specific representations and align view-specific query representations via a contrastive objective, which promotes cross-view consistency while suppressing view-specific noise. Extensive experiments verify that our CID-TKG achieves state-of-the-art performance under extrapolation settings. 

---
# DERM-3R: A Resource-Efficient Multimodal Agents Framework for Dermatologic Diagnosis and Treatment in Real-World Clinical Settings 

**Authors**: Ziwen Chen, Zhendong Wang, Chongjing Wang, Yurui Dong, Luozhijie Jin, Jihao Gu, Kui Chen, Jiaxi Yang, Bingjie Lu, Zhou Zhang, Jirui Dai, Changyong Luo, Xiameng Gai, Haibing Lan, Zhi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09596)  

**Abstract**: Dermatologic diseases impose a large and growing global burden, affecting billions and substantially reducing quality of life. While modern therapies can rapidly control acute symptoms, long-term outcomes are often limited by single-target paradigms, recurrent courses, and insufficient attention to systemic comorbidities. Traditional Chinese medicine (TCM) provides a complementary holistic approach via syndrome differentiation and individualized treatment, but practice is hindered by non-standardized knowledge, incomplete multimodal records, and poor scalability of expert reasoning. We propose DERM-3R, a resource-efficient multimodal agent framework to model TCM dermatologic diagnosis and treatment under limited data and compute. Based on real-world workflows, we reformulate decision-making into three core issues: fine-grained lesion recognition, multi-view lesion representation with specialist-level pathogenesis modeling, and holistic reasoning for syndrome differentiation and treatment planning. DERM-3R comprises three collaborative agents: DERM-Rec, DERM-Rep, and DERM-Reason, each targeting one component of this pipeline. Built on a lightweight multimodal LLM and partially fine-tuned on 103 real-world TCM psoriasis cases, DERM-3R performs strongly across dermatologic reasoning tasks. Evaluations using automatic metrics, LLM-as-a-judge, and physician assessment show that despite minimal data and parameter updates, DERM-3R matches or surpasses large general-purpose multimodal models. These results suggest structured, domain-aware multi-agent modeling can be a practical alternative to brute-force scaling for complex clinical tasks in dermatology and integrative medicine. 

---
# Spatial Competence Benchmark 

**Authors**: Jash Vira, Ashley Harris  

**Link**: [PDF](https://arxiv.org/pdf/2604.09594)  

**Abstract**: Spatial competence is the quality of maintaining a consistent internal representation of an environment and using it to infer discrete structure and plan actions under constraints. Prevailing spatial evaluations for large models are limited to probing isolated primitives through 3D transformations or visual question answering. We introduce the Spatial Competence Benchmark (SCBench), spanning three hierarchical capability buckets whose tasks require executable outputs verified by deterministic checkers or simulator-based evaluators. On SCBench, three frontier models exhibit monotonically decreasing accuracy up the capability ladder. Sweeping output-token caps shows that accuracy gains concentrate at low budgets and saturate quickly, and failures are dominated by locally plausible geometry that breaks global constraints. We release the task generators, verifiers, and visualisation tooling. 

---
# DeepReviewer 2.0: A Traceable Agentic System for Auditable Scientific Peer Review 

**Authors**: Yixuan Weng, Minjun Zhu, Qiujie Xie, Zhiyuan Ning, Shichen Li, Panzhong Lu, Zhen Lin, Enhao Gu, Qiyao Sun, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09590)  

**Abstract**: Automated peer review is often framed as generating fluent critique, yet reviewers and area chairs need judgments they can \emph{audit}: where a concern applies, what evidence supports it, and what concrete follow-up is required. DeepReviewer~2.0 is a process-controlled agentic review system built around an output contract: it produces a \textbf{traceable review package} with anchored annotations, localized evidence, and executable follow-up actions, and it exports only after meeting minimum traceability and coverage budgets. Concretely, it first builds a manuscript-only claim--evidence--risk ledger and verification agenda, then performs agenda-driven retrieval and writes anchored critiques under an export gate. On 134 ICLR~2025 submissions under three fixed protocols, an \emph{un-finetuned 196B} model running DeepReviewer~2.0 outperforms Gemini-3.1-Pro-preview, improving strict major-issue coverage (37.26\% vs.\ 23.57\%) and winning 71.63\% of micro-averaged blind comparisons against a human review committee, while ranking first among automatic systems in our pool. We position DeepReviewer~2.0 as an assistive tool rather than a decision proxy, and note remaining gaps such as ethics-sensitive checks. 

---
# Persistent Identity in AI Agents: A Multi-Anchor Architecture for Resilient Memory and Continuity 

**Authors**: Prahlad G. Menon  

**Link**: [PDF](https://arxiv.org/pdf/2604.09588)  

**Abstract**: Modern AI agents suffer from a fundamental identity problem: when context windows overflow and conversation histories are summarized, agents experience catastrophic forgetting -- losing not just information, but continuity of self. This technical limitation reflects a deeper architectural flaw: AI agent identity is centralized in a single memory store, creating a single point of failure. Drawing on neurological case studies of human memory disorders, we observe that human identity survives damage because it is distributed across multiple systems: episodic memory, procedural memory, emotional continuity, and embodied knowledge. We present this http URL, an open-source architecture that implements persistent identity through separable components (identity files and memory logs), and propose extensions toward multi-anchor resilience. The framework introduces a hybrid RAG+RLM retrieval system that automatically routes queries to appropriate memory access patterns, achieving efficient retrieval without sacrificing comprehensiveness. We formalize the notion of identity anchors for AI systems and present a roadmap for building agents whose identity can survive partial memory failures. Code is available at this http URL 

---
# MobiFlow: Real-World Mobile Agent Benchmarking through Trajectory Fusion 

**Authors**: Yunfei Feng, Xi Zhao, Cheng Zhang, Dahu Feng, Daolin Cheng, Jianqi Yu, Yubin Xia, Erhu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.09587)  

**Abstract**: Mobile agents can autonomously complete user-assigned tasks through GUI interactions. However, existing mainstream evaluation benchmarks, such as AndroidWorld, operate by connecting to a system-level Android emulator and provide evaluation signals based on the state of system resources. In real-world mobile-agent scenarios, however, many third-party applications do not expose system-level APIs to determine whether a task has succeeded, leading to a mismatch between benchmarks and real-world usage and making it difficult to evaluate model performance accurately. To address these issues, we propose MobiFlow, an evaluation framework built on tasks drawn from arbitrary third-party applications. Using an efficient graph-construction algorithm based on multi-trajectory fusion, MobiFlow can effectively compress the state space, support dynamic interaction, and better align with real-world third-party application scenarios. MobiFlow covers 20 widely used third-party applications and comprises 240 diverse real-world tasks, with enriched evaluation metrics. Compared with AndroidWorld, MobiFlow's evaluation results show higher alignment with human assessments and can guide the training of future GUI-based models under real workloads. 

---
# Agentic Exploration of PDE Spaces using Latent Foundation Models for Parameterized Simulations 

**Authors**: Abhijeet Vishwasrao, Francisco Giral, Mahmoud Golestanian, Federica Tonti, Andrea Arroyo Ramo, Adrian Lozano-Duran, Steven L. Brunton, Sergio Hoyas, Soledad Le Clainche, Hector Gomez, Ricardo Vinuesa  

**Link**: [PDF](https://arxiv.org/pdf/2604.09584)  

**Abstract**: Flow physics and more broadly physical phenomena governed by partial differential equations (PDEs), are inherently continuous, high-dimensional and often chaotic in nature. Traditionally, researchers have explored these rich spatiotemporal PDE solution spaces using laboratory experiments and/or computationally expensive numerical simulations. This severely limits automated and large-scale exploration, unlike domains such as drug discovery or materials science, where discrete, tokenizable representations naturally interface with large language models. We address this by coupling multi-agent LLMs with latent foundation models (LFMs), a generative model over parametrised simulations, that learns explicit, compact and disentangled latent representations of flow fields, enabling continuous exploration across governing PDE parameters and boundary conditions. The LFM serves as an on-demand surrogate simulator, allowing agents to query arbitrary parameter configurations at negligible cost. A hierarchical agent architecture orchestrates exploration through a closed loop of hypothesis, experimentation, analysis and verification, with a tool-modular interface requiring no user support. Applied to flow past tandem cylinders at Re = 500, the framework autonomously evaluates over 1,600 parameter-location pairs and discovers divergent scaling laws: a regime-dependent two-mode structure for minimum displacement thickness and a robust linear scaling for maximum momentum thickness, with both landscapes exhibiting a dual-extrema structure that emerges at the near-wake to co-shedding regime transition. The coupling of the learned physical representations with agentic reasoning establishes a general paradigm for automated scientific discovery in PDE-governed systems. 

---
# Factorizing formal contexts from closures of necessity operators 

**Authors**: Roberto G. Aragón, Jesús Medina, Eloísa Ramírez-Poussa  

**Link**: [PDF](https://arxiv.org/pdf/2604.09582)  

**Abstract**: Factorizing datasets is an interesting process in a multitude of approaches, but many times it is not possible or efficient the computation of a factorization of the dataset. A method to obtain independent subcontexts of a formal context with Boolean data was proposed in~\cite{dubois:2012}, based on the operators used in possibility theory. In this paper, we will analyze this method and study different properties related to the pairs of sets from which a factorization of a formal context arises. We also inspect how the properties given in the classical case can be extended to the fuzzy framework, which is essential to obtain a mechanism that allows the computation of independent subcontexts of a fuzzy context. 

---
# OpeFlo: Automated UX Evaluation via Simulated Human Web Interaction with GUI Grounding 

**Authors**: Wee Joe Tan, Zi Rui Lucas Lim, Shashank Durgad, Karim Obegi, Aiden Yiliu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09581)  

**Abstract**: Evaluating web usability typically requires time-consuming user studies and expert reviews, which often limits iteration speed during product development, especially for small teams and agile workflows. We present OpenFlo, a user-experience evaluation agent that simulates user behavior on websites and produces standardized usability. Unlike traditional tools that rely on DOM parsing, OpenFlo grounds actions and observations, enabling it to interact with real web pages end-to-end while maintaining a coherent trace of the user journey. Building on Avenir-Web, our system pairs this robust interaction with simulated user behavior profiles and a structured evaluation protocol that integrates the System Usability Scale (SUS), step-wise Single Ease Questions (SEQ), and concurrent Think Aloud. Subsequently, a comprehensive User Experience (UX) report will be generated. We discuss the architecture of OpenFlo and illustrate how its multimodal grounding improves robustness for web-based interaction and UX evaluation scenarios, paving the way for a new era of continuous, scalable, and data-driven usability testing that empowers every developer to build web interfaces that are usable. Code is available at: this https URL 

---
# OOWM: Structuring Embodied Reasoning and Planning via Object-Oriented Programmatic World Modeling 

**Authors**: Hongyu Chen, Liang Lin, Guangrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09580)  

**Abstract**: Standard Chain-of-Thought (CoT) prompting empowers Large Language Models (LLMs) with reasoning capabilities, yet its reliance on linear natural language is inherently insufficient for effective world modeling in embodied tasks. While text offers flexibility, it fails to explicitly represent the state-space, object hierarchies, and causal dependencies required for robust robotic planning. To address these limitations, we propose Object-Oriented World Modeling (OOWM), a novel framework that structures embodied reasoning through the lens of software engineering formalisms. We redefine the world model not as a latent vector space, but as an explicit symbolic tuple $W = \langle S, T \rangle$: a State Abstraction ($G_\text{state}$) instantiating the environmental state $S$, coupled with a Control Policy ($G_\text{control}$) representing the transition logic $T: S \times A \rightarrow S'$. OOWM leverages the Unified Modeling Language (UML) to materialize this definition: it employs Class Diagrams to ground visual perception into rigorous object hierarchies, and Activity Diagrams to operationalize planning into executable control flows. Furthermore, we introduce a three-stage training pipeline combining Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO). Crucially, this method utilizes outcome-based rewards from the final plan to implicitly optimize the underlying object-oriented reasoning structure, enabling effective learning even with sparse annotations. Extensive evaluations on the MRoom-30k benchmark demonstrate that OOWM significantly outperforms unstructured textual baselines in planning coherence, execution success, and structural fidelity, establishing a new paradigm for structured embodied reasoning. 

---
# Help Without Being Asked: A Deployed Proactive Agent System for On-Call Support with Continuous Self-Improvement 

**Authors**: Fengrui Liu, Xiao He, Tieying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09579)  

**Abstract**: In large-scale cloud service platforms, thousands of customer tickets are generated daily and are typically handled through on-call dialogues. This high volume of on-call interactions imposes a substantial workload on human support analysts. Recent studies have explored reactive agents that leverage large language models as a first line of support to interact with customers directly and resolve issues. However, when issues remain unresolved and are escalated to human support, these agents are typically disengaged. As a result, they cannot assist with follow-up inquiries, track resolution progress, or learn from the cases they fail to address. In this paper, we introduce Vigil, a novel proactive agent system designed to operate throughout the entire on-call life-cycle. Unlike reactive agents, Vigil focuses on providing assistance during the phase in which human support is already involved. It integrates into the dialogue between the customer and the analyst, proactively offering assistance without explicit user invocation. Moreover, Vigil incorporates a continuous self-improvement mechanism that extracts knowledge from human-resolved cases to autonomously update its capabilities. Vigil has been deployed on Volcano Engine, ByteDance's cloud platform, for over ten months, and comprehensive evaluations based on this deployment demonstrate its effectiveness and practicality. The open source version of this work is publicly available at this https URL. 

---
# Explainable Planning for Hybrid Systems 

**Authors**: Mir Md Sajid Sarwar  

**Link**: [PDF](https://arxiv.org/pdf/2604.09578)  

**Abstract**: The recent advancement in artificial intelligence (AI) technologies facilitates a paradigm shift toward automation. Autonomous systems are fully or partially replacing manually crafted ones. At the core of these systems is automated planning. With the advent of powerful planners, automated planning is now applied to many complex and safety-critical domains, including smart energy grids, self-driving cars, warehouse automation, urban and air traffic control, search and rescue operations, surveillance, robotics, and healthcare. There is a growing need to generate explanations of AI-based systems, which is one of the major challenges the planning community faces today. The thesis presents a comprehensive study on explainable artificial intelligence planning (XAIP) for hybrid systems that capture a representation of real-world problems closely. 

---
# AHC: Meta-Learned Adaptive Compression for Continual Object Detection on Memory-Constrained Microcontrollers 

**Authors**: Bibin Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2604.09576)  

**Abstract**: Deploying continual object detection on microcontrollers (MCUs) with under 100KB memory requires efficient feature compression that can adapt to evolving task distributions. Existing approaches rely on fixed compression strategies (e.g., FiLM conditioning) that cannot adapt to heterogeneous task characteristics, leading to suboptimal memory utilization and catastrophic forgetting. We introduce Adaptive Hierarchical Compression (AHC), a meta-learning framework featuring three key innovations: (1) true MAML-based compression that adapts via gradient descent to each new task in just 5 inner-loop steps, (2) hierarchical multi-scale compression with scale-aware ratios (8:1 for P3, 6.4:1 for P4, 4:1 for P5) matching FPN redundancy patterns, and (3) a dual-memory architecture combining short-term and long-term banks with importance-based consolidation under a hard 100KB budget. We provide formal theoretical guarantees bounding catastrophic forgetting as O({\epsilon}{this http URL(T)} + 1/{this http URL(M)}) where {\epsilon} is compression error, T is task count, and M is memory size. Experiments on CORe50, TiROD, and PASCAL VOC benchmarks with three standard baselines (Fine-tuning,EWC, iCaRL) demonstrate that AHC enables practical continual detection within a 100KB replay budget, achieving competitive accuracy through mean-pooled compressed feature replay combined with EWC regularization and feature distillation. 

---
# Turing Test on Screen: A Benchmark for Mobile GUI Agent Humanization 

**Authors**: Jiachen Zhu, Lingyu Yang, Rong Shan, Congmin Zheng, Zeyu Zheng, Weiwen Liu, Yong Yu, Weinan Zhang, Jianghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.09574)  

**Abstract**: The rise of autonomous GUI agents has triggered adversarial countermeasures from digital platforms, yet existing research prioritizes utility and robustness over the critical dimension of anti-detection. We argue that for agents to survive in human-centric ecosystems, they must evolve Humanization capabilities. We introduce the ``Turing Test on Screen,'' formally modeling the interaction as a MinMax optimization problem between a detector and an agent aiming to minimize behavioral divergence. We then collect a new high-fidelity dataset of mobile touch dynamics, and conduct our analysis that vanilla LMM-based agents are easily detectable due to unnatural kinematics. Consequently, we establish the Agent Humanization Benchmark (AHB) and detection metrics to quantify the trade-off between imitability and utility. Finally, we propose methods ranging from heuristic noise to data-driven behavioral matching, demonstrating that agents can achieve high imitability theoretically and empirically without sacrificing performance. This work shifts the paradigm from whether an agent can perform a task to how it performs it within a human-centric ecosystem, laying the groundwork for seamless coexistence in adversarial digital environments. 

---
# Seven simple steps for log analysis in AI systems 

**Authors**: Magda Dubois, Ekin Zorer, Maia Hamin, Joe Skinner, Alexandra Souly, Jerome Wynne, Harry Coppock, Lucas Satos, Sayash Kapoor, Sunischal Dev, Keno Juchems, Kimberly Mai, Timo Flesch, Lennart Luettgau, Charles Teague, Eric Patey, JJ Allaire, Lorenzo Pacchiardi, Jose Hernandez-Orallo, Cozmin Ududec  

**Link**: [PDF](https://arxiv.org/pdf/2604.09563)  

**Abstract**: AI systems produce large volumes of logs as they interact with tools and users. Analysing these logs can help understand model capabilities, propensities, and behaviours, or assess whether an evaluation worked as intended. Researchers have started developing methods for log analysis, but a standardised approach is still missing. Here we suggest a pipeline based on current best practices. We illustrate it with concrete code examples in the Inspect Scout library, provide detailed guidance on each step, and highlight common pitfalls. Our framework provides researchers with a foundation for rigorous and reproducible log analysis. 

---
# Linear Programming for Multi-Criteria Assessment with Cardinal and Ordinal Data: A Pessimistic Virtual Gap Analysis 

**Authors**: Fuh-Hwa Franklin Liu, Su-Chuan Shih  

**Link**: [PDF](https://arxiv.org/pdf/2604.09555)  

**Abstract**: Multi-criteria Analysis (MCA) is used to rank alternatives based on various criteria. Key MCA methods, such as Multiple Criteria Decision Making (MCDM) methods, estimate parameters for criteria to compute the performance of each alternative. Nonetheless, subjective evaluations and biases frequently influence the reliability of results, while the diversity of data affects the precision of the parameters. The novel linear programming-based Virtual Gap Analysis (VGA) models tackle these issues. This paper outlines a two-step method that integrates two novel VGA models to assess each alternative from a pessimistic perspective, using both quantitative and qualitative criteria, and employing cardinal and ordinal data. Next, prioritize the alternatives to eliminate the least favorable one. The proposed method is dependable and scalable, enabling thorough assessments efficiently and effectively within decision support systems. 

---
# LABBench2: An Improved Benchmark for AI Systems Performing Biology Research 

**Authors**: Jon M Laurent, Albert Bou, Michael Pieler, Conor Igoe, Alex Andonian, Siddharth Narayanan, James Braza, Alexandros Sanchez Vassopoulos, Jacob L Steenwyk, Blake Lash, Andrew D White, Samuel G Rodriques  

**Link**: [PDF](https://arxiv.org/pdf/2604.09554)  

**Abstract**: Optimism for accelerating scientific discovery with AI continues to grow. Current applications of AI in scientific research range from training dedicated foundation models on scientific data to agentic autonomous hypothesis generation systems to AI-driven autonomous labs. The need to measure progress of AI systems in scientific domains correspondingly must not only accelerate, but increasingly shift focus to more real-world capabilities. Beyond rote knowledge and even just reasoning to actually measuring the ability to perform meaningful work. Prior work introduced the Language Agent Biology Benchmark LAB-Bench as an initial attempt at measuring these abilities. Here we introduce an evolution of that benchmark, LABBench2, for measuring real-world capabilities of AI systems performing useful scientific tasks. LABBench2 comprises nearly 1,900 tasks and is, for the most part, a continuation of LAB-Bench, measuring similar capabilities but in more realistic contexts. We evaluate performance of current frontier models, and show that while abilities measured by LAB-Bench and LABBench2 have improved substantially, LABBench2 provides a meaningful jump in difficulty (model-specific accuracy differences range from -26% to -46% across subtasks) and underscores continued room for performance improvement. LABBench2 continues the legacy of LAB-Bench as a de facto benchmark for AI scientific research capabilities and we hope that it continues to help advance development of AI tools for these core research functions. To facilitate community use and development, we provide the task dataset at this https URL and a public eval harness at this https URL. 

---
# Physics-Informed State Space Models for Reliable Solar Irradiance Forecasting in Off-Grid Systems 

**Authors**: Mohammed Ezzaldin Babiker Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2604.11807)  

**Abstract**: The stable operation of autonomous off-grid photovoltaic systems dictates reliance on solar forecasting algorithms that respect atmospheric thermodynamics. Contemporary deep learning models consistently exhibit critical anomalies, primarily severe temporal phase lags during cloud transients and physically impossible nocturnal power generation. To resolve this divergence between data-driven modeling and deterministic celestial mechanics, this research introduces the Thermodynamic Liquid Manifold Network. The proposed methodology projects 15 meteorological and geometric variables into a Koopman-linearized Riemannian manifold to systematically map complex climatic dynamics. The architecture integrates a Spectral Calibration unit and a multiplicative Thermodynamic Alpha-Gate. This system synthesizes real-time atmospheric opacity with theoretical clear-sky boundary models, structurally enforcing strict celestial geometry compliance. This completely neutralizes phantom nocturnal generation while maintaining zero-lag synchronization during rapid weather shifts. Validated against a rigorous five-year testing horizon in a severe semi-arid climate, the framework achieves an RMSE of 18.31 Wh/m2 and a Pearson correlation of 0.988. The model strictly maintains a zero-magnitude nocturnal error across all 1826 testing days and exhibits a sub-30-minute phase response during high-frequency transients. Comprising exactly 63,458 trainable parameters, this ultra-lightweight design establishes a robust, thermodynamically consistent standard for edge-deployable microgrid controllers. 

---
# Solving Physics Olympiad via Reinforcement Learning on Physics Simulators 

**Authors**: Mihir Prabhudesai, Aryan Satpathy, Yangmin Li, Zheyang Qin, Nikash Bhardwaj, Amir Zadeh, Chuan Li, Katerina Fragkiadaki, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2604.11805)  

**Abstract**: We have witnessed remarkable advances in LLM reasoning capabilities with the advent of DeepSeek-R1. However, much of this progress has been fueled by the abundance of internet question-answer (QA) pairs, a major bottleneck going forward, since such data is limited in scale and concentrated mainly in domains like mathematics. In contrast, other sciences such as physics lack large-scale QA datasets to effectively train reasoning-capable models. In this work, we show that physics simulators can serve as a powerful alternative source of supervision for training LLMs for physical reasoning. We generate random scenes in physics engines, create synthetic question-answer pairs from simulated interactions, and train LLMs using reinforcement learning on this synthetic data. Our models exhibit zero-shot sim-to-real transfer to real-world physics benchmarks: for example, training solely on synthetic simulated data improves performance on IPhO (International Physics Olympiad) problems by 5-10 percentage points across model sizes. These results demonstrate that physics simulators can act as scalable data generators, enabling LLMs to acquire deep physical reasoning skills beyond the limitations of internet-scale QA data. Code available at: this https URL. 

---
# Budget-Aware Uncertainty for Radiotherapy Segmentation QA Using nnU-Net 

**Authors**: Ricardo Coimbra Brioso, Lorenzo Mondo, Damiano Dei, Nicola Lambri, Pietro Mancosu, Marta Scorsetti, Daniele Loiacono  

**Link**: [PDF](https://arxiv.org/pdf/2604.11798)  

**Abstract**: Accurate delineation of the Clinical Target Volume (CTV) is essential for radiotherapy planning, yet remains time-consuming and difficult to assess, especially for complex treatments such as Total Marrow and Lymph Node Irradiation (TMLI). While deep learning-based auto-segmentation can reduce workload, safe clinical deployment requires reliable cues indicating where models may be wrong. In this work, we propose a budget-aware uncertainty-driven quality assurance (QA) framework built on nnU-Net, combining uncertainty quantification and post-hoc calibration to produce voxel-wise uncertainty maps (based on predictive entropy) that can guide targeted manual review. We compare temperature scaling (TS), deep ensembles (DE), checkpoint ensembles (CE), and test-time augmentation (TTA), evaluated both individually and in combination on TMLI as a representative use case. Reliability is assessed through ROI-masked calibration metrics and uncertainty--error alignment under realistic revision constraints, summarized as AUC over the top 0-5% most uncertain voxels. Across configurations, segmentation accuracy remains stable, whereas TS substantially improves calibration. Uncertainty-error alignment improves most with calibrated checkpoint-based inference, leading to uncertainty maps that highlight more consistently regions requiring manual edits. Overall, integrating calibration with efficient ensembling seems a promising strategy to implement a budget-aware QA workflow for radiotherapy segmentation. 

---
# C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts 

**Authors**: Chenxi Qing, Junxi Wu, Zheng Liu, Yixiang Qiu, Hongyao Yu, Bin Chen, Hao Wu, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2604.11796)  

**Abstract**: Recently, large language models (LLMs) are capable of generating highly fluent textual content. While they offer significant convenience to humans, they also introduce various risks, like phishing and academic dishonesty. Numerous research efforts have been dedicated to developing algorithms for detecting AI-generated text and constructing relevant datasets. However, in the domain of Chinese corpora, challenges remain, including limited model diversity and data homogeneity. To address these issues, we propose C-ReD: a comprehensive Chinese Real-prompt AI-generated Detection benchmark. Experiments demonstrate that C-ReD not only enables reliable in-domain detection but also supports strong generalization to unseen LLMs and external Chinese datasets-addressing critical gaps in model diversity, domain coverage, and prompt realism that have limited prior Chinese detection benchmarks. We release our resources at this https URL. 

---
# A Mechanistic Analysis of Looped Reasoning Language Models 

**Authors**: Hugh Blayney, Álvaro Arroyo, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Michael M. Bronstein, Xiaowen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.11791)  

**Abstract**: Reasoning has become a central capability in large language models. Recent research has shown that reasoning performance can be improved by looping an LLM's layers in the latent dimension, resulting in looped reasoning language models. Despite promising results, few works have investigated how their internal dynamics differ from those of standard feedforward models. In this paper, we conduct a mechanistic analysis of the latent states in looped language models, focusing in particular on how the stages of inference observed in feedforward models compare to those observed in looped ones. To this end, we analyze cyclic recurrence and show that for many of the studied models each layer in the cycle converges to a distinct fixed point; consequently, the recurrent block follows a consistent cyclic trajectory in the latent space. We provide evidence that as these fixed points are reached, attention-head behavior stabilizes, leading to constant behavior across recurrences. Empirically, we discover that recurrent blocks learn stages of inference that closely mirror those of feedforward models, repeating these stages in depth with each iteration. We study how recurrent block size, input injection, and normalization influence the emergence and stability of these cyclic fixed points. We believe these findings help translate mechanistic insights into practical guidance for architectural design. 

---
# ClawGuard: A Runtime Security Framework for Tool-Augmented LLM Agents Against Indirect Prompt Injection 

**Authors**: Wei Zhao, Zhe Li, Peixin Zhang, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.11790)  

**Abstract**: Tool-augmented Large Language Model (LLM) agents have demonstrated impressive capabilities in automating complex, multi-step real-world tasks, yet remain vulnerable to indirect prompt injection. Adversaries exploit this weakness by embedding malicious instructions within tool-returned content, which agents directly incorporate into their conversation history as trusted observations. This vulnerability manifests across three primary attack channels: web and local content injection, MCP server injection, and skill file injection. To address these vulnerabilities, we introduce \textsc{ClawGuard}, a novel runtime security framework that enforces a user-confirmed rule set at every tool-call boundary, transforming unreliable alignment-dependent defense into a deterministic, auditable mechanism that intercepts adversarial tool calls before any real-world effect is produced. By automatically deriving task-specific access constraints from the user's stated objective prior to any external tool invocation, \textsc{ClawGuard} blocks all three injection pathways without model modification or infrastructure change. Experiments across five state-of-the-art language models on AgentDojo, SkillInject, and MCPSafeBench demonstrate that \textsc{ClawGuard} achieves robust protection against indirect prompt injection without compromising agent utility. This work establishes deterministic tool-call boundary enforcement as an effective defense mechanism for secure agentic AI systems, requiring neither safety-specific fine-tuning nor architectural modification. Code is publicly available at this https URL. 

---
# ClawGUI: A Unified Framework for Training, Evaluating, and Deploying GUI Agents 

**Authors**: Fei Tang, Zhiqiong Lu, Boxuan Zhang, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11784)  

**Abstract**: GUI agents drive applications through their visual interfaces instead of programmatic APIs, interacting with arbitrary software via taps, swipes, and keystrokes, reaching a long tail of applications that CLI-based agents cannot. Yet progress in this area is bottlenecked less by modeling capacity than by the absence of a coherent full-stack infrastructure: online RL training suffers from environment instability and closed pipelines, evaluation protocols drift silently across works, and trained agents rarely reach real users on real devices. We present \textbf{ClawGUI}, an open-source framework addressing these three gaps within a single harness. \textbf{ClawGUI-RL} provides the first open-source GUI agent RL infrastructure with validated support for both parallel virtual environments and real physical devices, integrating GiGPO with a Process Reward Model for dense step-level supervision. \textbf{ClawGUI-Eval} enforces a fully standardized evaluation pipeline across 6 benchmarks and 11+ models, achieving 95.8\% reproduction against official baselines. \textbf{ClawGUI-Agent} brings trained agents to Android, HarmonyOS, and iOS through 12+ chat platforms with hybrid CLI-GUI control and persistent personalized memory. Trained end to end within this pipeline, \textbf{ClawGUI-2B} achieves 17.1\% Success Rate on MobileWorld GUI-Only, outperforming the same-scale MAI-UI-2B baseline by 6.0\%. 

---
# General365: Benchmarking General Reasoning in Large Language Models Across Diverse and Challenging Tasks 

**Authors**: Junlin Liu, Shengnan An, Shuang Zhou, Dan Ma, Shixiong Luo, Ying Xie, Yuan Zhang, Wenling Yuan, Yifan Zhou, Xiaoyu Li, Ziwen Wang, Xuezhi Cao, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.11778)  

**Abstract**: Contemporary large language models (LLMs) have demonstrated remarkable reasoning capabilities, particularly in specialized domains like mathematics and physics. However, their ability to generalize these reasoning skills to more general and broader contexts--often termed general reasoning--remains under-explored. Unlike domain-specific reasoning, general reasoning relies less on expert knowledge but still presents formidable reasoning challenges, such as complex constraints, nested logical branches, and semantic interference. To address this gap, we introduce General365, a benchmark specifically designed to assess general reasoning in LLMs. By restricting background knowledge to a K-12 level, General365 explicitly decouples reasoning from specialized expertise. The benchmark comprises 365 seed problems and 1,095 variant problems across eight categories, ensuring both high difficulty and diversity. Evaluations across 26 leading LLMs reveal that even the top-performing model achieves only 62.8% accuracy, in stark contrast to the near-perfect performances of LLMs in math and physics benchmarks. These results suggest that the reasoning abilities of current LLMs are heavily domain-dependent, leaving significant room for improvement in broader applications. We envision General365 as a catalyst for advancing LLM reasoning beyond domain-specific tasks toward robust, general-purpose real-world scenarios. Code, Dataset, and Leaderboard: this https URL 

---
# Efficient KernelSHAP Explanations for Patch-based 3D Medical Image Segmentation 

**Authors**: Ricardo Coimbra Brioso, Giulio Sichili, Damiano Dei, Nicola Lambri, Pietro Mancosu, Marta Scorsetti, Daniele Loiacono  

**Link**: [PDF](https://arxiv.org/pdf/2604.11775)  

**Abstract**: Perturbation-based explainability methods such as KernelSHAP provide model-agnostic attributions but are typically impractical for patch-based 3D medical image segmentation due to the large number of coalition evaluations and the high cost of sliding-window inference. We present an efficient KernelSHAP framework for volumetric CT segmentation that restricts computation to a user-defined region of interest and its receptive-field support, and accelerates inference via patch logit caching, reusing baseline predictions for unaffected patches while preserving nnU-Net's fusion scheme. To enable clinically meaningful attributions, we compare three automatically generated feature abstractions within the receptive-field crop: whole-organ units, regular FCC supervoxels, and hybrid organ-aware supervoxels, and we study multiple aggregation/value functions targeting stabilizing evidence (TP/Dice/Soft Dice) or false-positive behavior. Experiments on whole-body CT segmentations show that caching substantially reduces redundant computation (with computational savings ranging from 15% to 30%) and that faithfulness and interpretability exhibit clear trade-offs: regular supervoxels often maximize perturbation-based metrics but lack anatomical alignment, whereas organ-aware units yield more clinically interpretable explanations and are particularly effective for highlighting false-positive drivers under normalized metrics. 

---
# StarVLA-$α$: Reducing Complexity in Vision-Language-Action Systems 

**Authors**: Jinhui Ye, Ning Gao, Senqiao Yang, Jinliang Zheng, Zixuan Wang, Yuxin Chen, Pengguang Chen, Yilun Chen, Shu Liu, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.11757)  

**Abstract**: Vision-Language-Action (VLA) models have recently emerged as a promising paradigm for building general-purpose robotic agents. However, the VLA landscape remains highly fragmented and complex: as existing approaches vary substantially in architectures, training data, embodiment configurations, and benchmark-specific engineering. In this work, we introduce StarVLA-$\alpha$, a simple yet strong baseline designed to study VLA design choices under controlled conditions. StarVLA-$\alpha$ deliberately minimizes architectural and pipeline complexity to reduce experimental confounders and enable systematic analysis. Specifically, we re-evaluate several key design axes, including action modeling strategies, robot-specific pretraining, and interface engineering. Across unified multi-benchmark training on LIBERO, SimplerEnv, RoboTwin, and RoboCasa, the same simple baseline remains highly competitive, indicating that a strong VLM backbone combined with minimal design is already sufficient to achieve strong performance without relying on additional architectural complexity or engineering tricks. Notably, our single generalist model outperforms $\pi_{0.5}$ by 20\% on the public real-world RoboChallenge benchmark. We expect StarVLA-$\alpha$ to serve as a solid starting point for future research in the VLA regime. Code will be released at this https URL. 

---
# Grounded World Model for Semantically Generalizable Planning 

**Authors**: Quanyi Li, Lan Feng, Haonan Zhang, Wuyang Li, Letian Wang, Alexandre Alahi, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2604.11751)  

**Abstract**: In Model Predictive Control (MPC), world models predict the future outcomes of various action proposals, which are then scored to guide the selection of the optimal action. For visuomotor MPC, the score function is a distance metric between a predicted image and a goal image, measured in the latent space of a pretrained vision encoder like DINO and JEPA. However, it is challenging to obtain the goal image in advance of the task execution, particularly in new environments. Additionally, conveying the goal through an image offers limited interactivity compared with natural language. In this work, we propose to learn a Grounded World Model (GWM) in a vision-language-aligned latent space. As a result, each proposed action is scored based on how close its future outcome is to the task instruction, reflected by the similarity of embeddings. This approach transforms the visuomotor MPC to a VLA that surpasses VLM-based VLAs in semantic generalization. On the proposed WISER benchmark, GWM-MPC achieves a 87% success rate on the test set comprising 288 tasks that feature unseen visual signals and referring expressions, yet remain solvable with motions demonstrated during training. In contrast, traditional VLAs achieve an average success rate of 22%, even though they overfit the training set with a 90% success rate. 

---
# Discourse Diversity in Multi-Turn Empathic Dialogue 

**Authors**: Hongli Zhan, Emma S. Gueorguieva, Javier Hernandez, Jina Suh, Desmond C. Ong, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11742)  

**Abstract**: Large language models (LLMs) produce responses rated as highly empathic in single-turn settings (Ayers et al., 2023; Lee et al., 2024), yet they are also known to be formulaic generators that reuse the same lexical patterns, syntactic templates, and discourse structures across tasks (Jiang et al., 2025; Shaib et al., 2024; Namuduri et al., 2025). Less attention has been paid to whether this formulaicity extends to the level of discourse moves, i.e., what a response does for the person it is addressing. This question is especially consequential for empathic dialogue, where effective support demands not just a kind response at one moment but varied strategies as a conversation unfolds (Stiles et al., 1998). Indeed, prior work shows that LLMs reuse the same tactic sequences more than human supporters in single-turn settings (Gueorguieva et al., 2026). We extend this analysis to multi-turn conversations and find that the rigidity compounds: once a tactic appears in a supporter turn, LLMs reuse it in the next at nearly double the rate of humans (0.50-0.56 vs. 0.27). This pattern holds across LLMs serving as supporters in real emotional support conversations, and is invisible to standard similarity metrics. To address this gap, we introduce MINT (Multi-turn Inter-tactic Novelty Training), the first reinforcement learning framework to optimize discourse move diversity across multi-turn empathic dialogue. The best MINT variant combines an empathy quality reward with a cross-turn tactic novelty signal, improving aggregate empathy by 25.3% over vanilla across 1.7B and 4B models while reducing cross-turn discourse move repetition by 26.3% on the 4B model, surpassing all baselines including quality-only and token-level diversity methods on both measures. These results suggest that what current models lack is not empathy itself, but the ability to vary their discourse moves across a conversation. 

---
# Multi-ORFT: Stable Online Reinforcement Fine-Tuning for Multi-Agent Diffusion Planning in Cooperative Driving 

**Authors**: Haojie Bai, Aimin Li, Ruoyu Yao, Xiongwei Zhao, Tingting Zhang, Xing Zhang, Lin Gao, and Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.11734)  

**Abstract**: Closed-loop cooperative driving requires planners that generate realistic multimodal multi-agent trajectories while improving safety and traffic efficiency. Existing diffusion planners can model multimodal behaviors from demonstrations, but they often exhibit weak scene consistency and remain poorly aligned with closed-loop objectives; meanwhile, stable online post-training in reactive multi-agent environments remains difficult. We present Multi-ORFT, which couples scene-conditioned diffusion pre-training with stable online reinforcement post-training. In pre-training, the planner uses inter-agent self-attention, cross-attention, and AdaLN-Zero-based scene conditioning to improve scene consistency and road adherence of joint trajectories. In post-training, we formulate a two-level MDP that exposes step-wise reverse-kernel likelihoods for online optimization, and combine dense trajectory-level rewards with variance-gated group-relative policy optimization (VG-GRPO) to stabilize training. On the WOMD closed-loop benchmark, Multi-ORFT reduces collision rate from 2.04% to 1.89% and off-road rate from 1.68% to 1.36%, while increasing average speed from 8.36 to 8.61 m/s relative to the pre-trained planner, and it outperforms strong open-source baselines including SMART-large, SMART-tiny-CLSFT, and VBD on the primary safety and efficiency metrics. These results show that coupling scene-consistent denoising with stable online diffusion-policy optimization improves the reliability of closed-loop cooperative driving. 

---
# Endogenous Information in Routing Games: Memory-Constrained Equilibria, Recall Braess Paradoxes, and Memory Design 

**Authors**: Saad Alqithami  

**Link**: [PDF](https://arxiv.org/pdf/2604.11733)  

**Abstract**: We study routing games in which travelers optimize over routes that are remembered or surfaced, rather than over a fixed exogenous action set. The paper develops a tractable design theory for endogenous recall and then connects it back to an explicit finite-memory micro model. At the micro level, each traveler carries a finite memory state, receives surfaced alternatives, chooses via a logit rule, and updates memory under a policy such as LRU. This yields a stationary Forgetful Wardrop Equilibrium (FWE); existence is proved under mild regularity, and uniqueness follows in a contraction regime for the reduced fixed-point map. The paper's main design layer is a stationary salience model that summarizes persistent memory and interface effects as route-specific weights. Salience-weighted stochastic user equilibrium is the unique minimizer of a strictly convex potential, which yields a clean optimization and implementability theory. In this layer we characterize governed implementability under ratio budgets and affine tying constraints, and derive constructive algorithms on parallel and series-parallel networks. The bridge between layers is exact for last-choice memory (B=1): the micro model is then equivalent to the salience model, so any interior salience vector can be realized by an appropriate surfacing policy. For larger memories, we develop an explicit LRU-to-TTL-to-salience approximation pipeline and add contraction-based bounds that translate surrogate-map error into fixed-point and welfare error. Finally, we define a Recall Braess Paradox, in which improving recall increases equilibrium delay without changing physical capacity, and show that it can arise on every two-terminal network with at least two distinct s-t paths. Targeted experiments support the approximation regime, governed-design predictions, and the computational advantages of the reduced layer. 

---
# Evaluating Cooperation in LLM Social Groups through Elected Leadership 

**Authors**: Ryan Faulkner, Anushka Deshpande, David Guzman Piedrahita, Joel Z. Leibo, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.11721)  

**Abstract**: Governing common-pool resources requires agents to develop enduring strategies through cooperation and self-governance to avoid collective failure. While foundation models have shown potential for cooperation in these settings, existing multi-agent research provides little insight into whether structured leadership and election mechanisms can improve collective decision making. The lack of such a critical organizational feature ubiquitous in human society presents a significant shortcoming of the current methods. In this work we aim to directly address whether leadership and elections can support improved social welfare and cooperation through multi-agent simulation with LLMs. We present our open-source framework that simulates leadership through elected personas and candidate-driven agendas and carry out an empirical study of LLMs under controlled governance conditions. Our experiments demonstrate that having elected leadership improves social welfare scores by 55.4% and survival time by 128.6% across a range of high performing LLMs. Through the construction of an agent social graph we compute centrality metrics to assess the social influence of leader personas and also analyze rhetorical and cooperative tendencies revealed through a sentiment analysis on leader utterances. This work lays the foundation for further study of election mechanisms in multi-agent systems toward navigating complex social dilemmas. 

---
# On the Robustness of Watermarking for Autoregressive Image Generation 

**Authors**: Andreas Müller, Denis Lukovnikov, Shingo Kodama, Minh Pham, Anubhav Jain, Jonathan Petit, Niv Cohen, Asja Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2604.11720)  

**Abstract**: The proliferation of autoregressive (AR) image generators demands reliable detection and attribution of their outputs to mitigate misinformation, and to filter synthetic images from training data to prevent model collapse. To address this need, watermarking techniques, specifically designed for AR models, embed a subtle signal at generation time, enabling downstream verification through a corresponding watermark detector. In this work, we study these schemes and demonstrate their vulnerability to both watermark removal and forgery attacks. We assess existing attacks and further introduce three new attacks: (i) a vector-quantized regeneration removal attack, (ii) adversarial optimization-based attack, and (iii) a frequency injection attack. Our evaluation reveals that removal and forgery attacks can be effective with access to a single watermarked reference image and without access to original model parameters or watermarking secrets. Our findings indicate that existing watermarking schemes for AR image generation do not reliably support synthetic content detection for dataset filtering. Moreover, they enable Watermark Mimicry, whereby authentic images can be manipulated to imitate a generator's watermark and trigger false detection to prevent their inclusion in future model training. 

---
# Fairness is Not Flat: Geometric Phase Transitions Against Shortcut Learning 

**Authors**: Nicolas Rodriguez-Alvarez, Fernando Rodriguez-Merino  

**Link**: [PDF](https://arxiv.org/pdf/2604.11704)  

**Abstract**: Deep Neural Networks are highly susceptible to shortcut learning, frequently memorizing low-dimensional spurious correlations instead of underlying causal mechanisms. This phenomenon not only degrades out-of-distribution robustness but also induces severe demographic biases in sensitive applications. In this paper, we propose a geometric \textit{a priori} methodology to mitigate shortcut learning. By deploying a zero-hidden-layer ($N=1$) Topological Auditor, we mathematically isolate features that monopolize the gradient without human intervention. We empirically demonstrate a Capacity Phase Transition: once linear shortcuts are pruned, networks are forced to utilize higher geometric capacity ($N \geq 16$) to curve the decision boundary and learn ethical representations. Our approach outperforms L1 Regularization -- which collapses into demographic bias -- and operates at a fraction of the computational cost of post-hoc methods like Just Train Twice (JTT), successfully reducing counterfactual gender vulnerability from 21.18\% to 7.66\%. 

---
# Legal2LogicICL: Improving Generalization in Transforming Legal Cases to Logical Formulas via Diverse Few-Shot Learning 

**Authors**: Jieying Xue, Phuong Minh Nguyen, Ha Thanh Nguyen, May Myo Zin, Ken Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2604.11699)  

**Abstract**: This work aims to improve the generalization of logic-based legal reasoning systems by integrating recent advances in NLP with legal-domain adaptive few-shot learning techniques using LLMs. Existing logic-based legal reasoning pipelines typically rely on fine-tuned models to map natural-language legal cases into logical formulas before forwarding them to a symbolic reasoner. However, such approaches are heavily constrained by the scarcity of high-quality annotated training data. To address this limitation, we propose a novel LLM-based legal reasoning framework that enables effective in-context learning through retrieval-augmented generation. Specifically, we introduce Legal2LogicICL, a few-shot retrieval framework that balances diversity and similarity of exemplars at both the latent semantic representation level and the legal text structure level. In addition, our method explicitly accounts for legal structure by mitigating entity-induced retrieval bias in legal texts, where lengthy and highly specific entity mentions often dominate semantic representations and obscure legally meaningful reasoning patterns. Our Legal2LogicICL constructs informative and robust few-shot demonstrations, leading to accurate and stable logical rule generation without requiring additional training. In addition, we construct a new dataset, named Legal2Proleg, which is annotated with alignments between legal cases and PROLEG logical formulas to support the evaluation of legal semantic parsing. Experimental results on both open-source and proprietary LLMs demonstrate that our approach significantly improves accuracy, stability, and generalization in transforming natural-language legal case descriptions into logical representations, highlighting its effectiveness for interpretable and reliable legal reasoning. Our code is available at this https URL. 

---
# AffordSim: A Scalable Data Generator and Benchmark for Affordance-Aware Robotic Manipulation 

**Authors**: Mingyang Li, Haofan Xu, Haowen Sun, Xinzhe Chen, Sihua Ren, Liqi Huang, Xinyang Sui, Chenyang Miao, Qiongjie Cui, Zeyang Liu, Xingyu Chen, Xuguang Lan  

**Link**: [PDF](https://arxiv.org/pdf/2604.11674)  

**Abstract**: Simulation-based data generation has become a dominant paradigm for training robotic manipulation policies, yet existing platforms do not incorporate object affordance information into trajectory generation. As a result, tasks requiring precise interaction with specific functional regions--grasping a mug by its handle, pouring from a cup's rim, or hanging a mug on a hook--cannot be automatically generated with semantically correct trajectories. We introduce AffordSim, the first simulation framework that integrates open-vocabulary 3D affordance prediction into the manipulation data generation pipeline. AffordSim uses our VoxAfford model, an open-vocabulary 3D affordance detector that enhances MLLM output tokens with multi-scale geometric features, to predict affordance maps on object point clouds, guiding grasp pose estimation toward task-relevant functional regions. Built on NVIDIA Isaac Sim with cross-embodiment support (Franka FR3, Panda, UR5e, Kinova), VLM-powered task generation, and novel domain randomization using DA3-based 3D Gaussian reconstruction from real photographs, AffordSim enables automated, scalable generation of affordance-aware manipulation data. We establish a benchmark of 50 tasks across 7 categories (grasping, placing, stacking, pushing/pulling, pouring, mug hanging, long-horizon composite) and evaluate 4 imitation learning baselines (BC, Diffusion Policy, ACT, Pi 0.5). Our results reveal that while grasping is largely solved (53-93% success), affordance-demanding tasks such as pouring into narrow containers (1-43%) and mug hanging (0-47%) remain significantly more challenging for current imitation learning methods, highlighting the need for affordance-aware data generation. Zero-shot sim-to-real experiments on a real Franka FR3 validate the transferability of the generated data. 

---
# NetworkNet: A Deep Neural Network Approach for Random Networks with Sparse Nodal Attributes and Complex Nodal Heterogeneity 

**Authors**: Zhaoyu Xing, Xiufan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11673)  

**Abstract**: Heterogeneous network data with rich nodal information become increasingly prevalent across multidisciplinary research, yet accurately modeling complex nodal heterogeneity and simultaneously selecting influential nodal attributes remains an open challenge. This problem is central to many applications in economics and sociology, when both nodal heterogeneity and high-dimensional individual characteristics highly affect network formation. We propose a statistically grounded, unified deep neural network approach for modeling nodal heterogeneity in random networks with high-dimensional nodal attributes, namely ``NetworkNet''. A key innovation of NetworkNet lies in a tailored neural architecture that explicitly parameterizes attribute-driven heterogeneity, and at the same time, embeds a scalable attribute selection mechanism. NetworkNet consistently estimates two types of latent heterogeneity functions, i.e., nodal expansiveness and popularity, while simultaneously performing data-driven attribute selection to extract influential nodal attributes. By unifying classical statistical network modeling with deep learning, NetworkNet delivers the expressive power of DNNs with methodological interpretability, algorithmic scalability, and statistical rigor with a non-asymptotic approximation error bound. Empirically, simulations demonstrate strong performance in both heterogeneity estimation and high-dimensional attribute selection. We further apply NetworkNet to a large-scale author-citation network among statisticians, revealing new insights into the dynamic evolution of research fields and scholarly impact. 

---
# Playing Along: Learning a Double-Agent Defender for Belief Steering via Theory of Mind 

**Authors**: Hanqi Xiao, Vaidehi Patil, Zaid Khan, Hyunji Lee, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2604.11666)  

**Abstract**: As large language models (LLMs) become the engine behind conversational systems, their ability to reason about the intentions and states of their dialogue partners (i.e., form and use a theory-of-mind, or ToM) becomes increasingly critical for safe interaction with potentially adversarial partners. We propose a novel privacy-themed ToM challenge, ToM for Steering Beliefs (ToM-SB), in which a defender must act as a Double Agent to steer the beliefs of an attacker with partial prior knowledge within a shared universe. To succeed on ToM-SB, the defender must engage with and form a ToM of the attacker, with a goal of fooling the attacker into believing they have succeeded in extracting sensitive information. We find that strong frontier models like Gemini3-Pro and GPT-5.4 struggle on ToM-SB, often failing to fool attackers in hard scenarios with partial attacker prior knowledge, even when prompted to reason about the attacker's beliefs (ToM prompting). To close this gap, we train models on ToM-SB to act as AI Double Agents using reinforcement learning, testing both fooling and ToM rewards. Notably, we find a bidirectionally emergent relationship between ToM and attacker-fooling: rewarding fooling success alone improves ToM, and rewarding ToM alone improves fooling. Across four attackers with different strengths, six defender methods, and both in-distribution and out-of-distribution (OOD) evaluation, we find that gains in ToM and attacker-fooling are well-correlated, highlighting belief modeling as a key driver of success on ToM-SB. AI Double Agents that combine both ToM and fooling rewards yield the strongest fooling and ToM performance, outperforming Gemini3-Pro and GPT-5.4 with ToM prompting on hard scenarios. We also show that ToM-SB and AI Double Agents can be extended to stronger attackers, demonstrating generalization to OOD settings and the upgradability of our task. 

---
# Beyond LLMs, Sparse Distributed Memory, and Neuromorphics <A Hyper-Dimensional SRAM-CAM "VaCoAl" for Ultra-High Speed, Ultra-Low Power, and Low Cost> 

**Authors**: Hiroyuki Chuma, Kanji Otsuka, Yoichi Sato  

**Link**: [PDF](https://arxiv.org/pdf/2604.11665)  

**Abstract**: This paper reports an unexpected finding: in a deterministic hyperdimensional computing (HDC) architecture based on Galois-field algebra, a path-dependent semantic selection mechanism emerges, equivalent to spike-timing-dependent plasticity (STDP), with magnitude predictable a priori by a closed-form expression matching large-scale measurements. This addresses limitations of modern AI including catastrophic forgetting, learning stagnation, and the Binding Problem at an algebraic level. We propose VaCoAl (Vague Coincident Algorithm) and its Python implementation PyVaCoAl, combining ultra-high-dimensional memory with deterministic logic. Rooted in Sparse Distributed Memory, it resolves orthogonalisation and retrieval in high-dimensional binary spaces via Galois-field diffusion, enabling low-load deployment. VaCoAl is a memory-centric architecture prioritising retrieval and association, enabling reversible composition while preserving element independence and supporting compositional generalisation with a transparent reliability metric (CR score). We evaluated multi-hop reasoning on about 470k mentor-student relations from Wikidata, tracing up to 57 generations (over 25.5M paths). Using HDC bundling and unbinding with CR-based denoising, we quantify concept propagation over DAGs. Results show a reinterpretation of the Newton-Leibniz dispute and a phase transition from sparse convergence to a post-Leibniz "superhighway", from which structural indicators emerge supporting a Kuhnian paradigm shift. Collision-tolerance mechanisms further induce path-based pruning that favors direct paths, yielding emergent semantic selection equivalent to STDP. VaCoAl thus defines a third paradigm, HDC-AI, complementing LLMs with reversible multi-hop reasoning. 

---
# Towards Autonomous Mechanistic Reasoning in Virtual Cells 

**Authors**: Yunhui Jang, Lu Zhu, Jake Fawkes, Alisandra Kaye Denton, Dominique Beaini, Emmanuel Noutahi  

**Link**: [PDF](https://arxiv.org/pdf/2604.11661)  

**Abstract**: Large language models (LLMs) have recently gained significant attention as a promising approach to accelerate scientific discovery. However, their application in open-ended scientific domains such as biology remains limited, primarily due to the lack of factually grounded and actionable explanations. To address this, we introduce a structured explanation formalism for virtual cells that represents biological reasoning as mechanistic action graphs, enabling systematic verification and falsification. Building upon this, we propose VCR-Agent, a multi-agent framework that integrates biologically grounded knowledge retrieval with a verifier-based filtering approach to generate and validate mechanistic reasoning autonomously. Using this framework, we release VC-TRACES dataset, which consists of verified mechanistic explanations derived from the Tahoe-100M atlas. Empirically, we demonstrate that training with these explanations improves factual precision and provides a more effective supervision signal for downstream gene expression prediction. These results underscore the importance of reliable mechanistic reasoning for virtual cells, achieved through the synergy of multi-agent and rigorous verification. 

---
# RPA-Check: A Multi-Stage Automated Framework for Evaluating Dynamic LLM-based Role-Playing Agents 

**Authors**: Riccardo Rosati, Edoardo Colucci, Massimiliano Bolognini, Adriano Mancini, Paolo Sernani  

**Link**: [PDF](https://arxiv.org/pdf/2604.11655)  

**Abstract**: The rapid adoption of Large Language Models (LLMs) in interactive systems has enabled the creation of dynamic, open-ended Role-Playing Agents (RPAs). However, evaluating these agents remains a significant challenge, as standard NLP metrics fail to capture the nuances of role adherence, logical consistency, and long-term narrative stability. This paper introduces RPA-Check, a multi-stage automated evaluation framework designed to objectively assess the performance of LLM-based RPAs in complex, constraints-heavy environments. Our methodology is based on a four-step pipeline: (1) Dimension Definition, establishing high-level qualitative behavioral criteria; (2) Augmentation, where these requirements are expanded into granular boolean checklist indicators; (3) Semantic Filtering, to ensure indicator objectivity, no redundancy and agent isolation; and (4) LLM-as-a-Judge Evaluation, which employs chain-of-thought verification to score agent fidelity. We validate this framework by applying it to LLM Court, a serious game for forensic training involving several quantized local models. Experimental results across five distinct legal scenarios demonstrate the framework's ability to identify subtle trade-offs between model size, reasoning depth, and operational stability. Notably, the findings reveal an inverse relationship between parametric scale and procedural consistency, showing that smaller, adequately instruction-tuned models (8-9B) can outperform larger architectures prone to user-alignment bias or sycophancy. RPA-Check thus provides a standardized and reproducible metric for future research in generative agent evaluation within specialized domains. 

---
# CodeTracer: Towards Traceable Agent States 

**Authors**: Han Li, Yifan Yao, Letian Zhu, Rili Feng, Hongyi Ye, Jiaming Wang, Yancheng He, Pengyu Zou, Lehan Zhang, Xinping Lei, Haoyang Huang, Ken Deng, Ming Sun, Zhaoxiang Zhang, He Ye, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11641)  

**Abstract**: Code agents are advancing rapidly, but debugging them is becoming increasingly difficult. As frameworks orchestrate parallel tool calls and multi-stage workflows over complex tasks, making the agent's state transitions and error propagation hard to observe. In these runs, an early misstep can trap the agent in unproductive loops or even cascade into fundamental errors, forming hidden error chains that make it hard to tell when the agent goes off track and why. Existing agent tracing analyses either focus on simple interaction or rely on small-scale manual inspection, which limits their scalability and usefulness for real coding workflows. We present CodeTracer, a tracing architecture that parses heterogeneous run artifacts through evolving extractors, reconstructs the full state transition history as a hierarchical trace tree with persistent memory, and performs failure onset localization to pinpoint the failure origin and its downstream chain. To enable systematic evaluation, we construct CodeTraceBench from a large collection of executed trajectories generated by four widely used code agent frameworks on diverse code tasks (e.g., bug fixing, refactoring, and terminal interaction), with supervision at both the stage and step levels for failure localization. Experiments show that CodeTracer substantially outperforms direct prompting and lightweight baselines, and that replaying its diagnostic signals consistently recovers originally failed runs under matched budgets. Our code and data are publicly available. 

---
# SCNO: Spiking Compositional Neural Operator -- Towards a Neuromorphic Foundation Model for Nuclear PDE Solving 

**Authors**: Samrendra Roy, Souvik Chakraborty, Rizwan-uddin, Syed Bahauddin Alam  

**Link**: [PDF](https://arxiv.org/pdf/2604.11625)  

**Abstract**: Neural operators have emerged as powerful surrogates for partial differential equation (PDE) solvers, yet they are typically trained as monolithic models for individual PDEs, require energy-intensive GPU hardware, and must be retrained from scratch when new physics emerge. We introduce the Spiking Compositional Neural Operator (SCNO), a modular architecture combining spiking and conventional components that addresses all three limitations. SCNO maintains a library of small spiking neural operator blocks, each trained on a single elementary differential operator (convection, diffusion, reaction), and composes them through a lightweight input-conditioned aggregator to solve coupled PDEs not seen during block training. A small correction network learns cross-coupling residuals while keeping all blocks and the aggregator frozen, preserving zero-forgetting modular expansion by construction. We evaluate SCNO on eight PDE families including five coupled systems and a nuclear-relevant 1-group neutron diffusion equation. SCNO with correction achieves the lowest relative $L^2$ error on four of five coupled PDEs, outperforming both a monolithic spiking DeepONet (by up to 62%, mean over 3 seeds) and a standard ANN DeepONet (by up to 65%), while requiring only 95K trainable parameters versus 462K for the monolithic baseline. To our knowledge, this is the first compositional spiking neural operator and the first proof-of-concept for modular neuromorphic PDE solving with built-in forgetting-free expansion. 

---
# CUTEv2: Unified and Configurable Matrix Extension for Diverse CPU Architectures with Minimal Design Overhead 

**Authors**: Jinpeng Ye, Chongxi Wang, Wenqing Li, Bin Yuan, Shiyi Wang, Fenglu Zhang, Junyu Yue, Jianan Xie, Yunhao Ye, Haoyu Deng, Yingkun Zhou, Xin Cheng, Fuxin Zhang, Jian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11615)  

**Abstract**: Matrix extensions have emerged as an essential feature in modern CPUs to address the surging demands of AI workloads. However, existing designs often incur substantial hardware and software design overhead. Tight coupling with the CPU pipeline complicates integration across diverse CPUs, while fine-grained synchronous instructions hinder the development of high-performance kernels.
This paper proposes a unified and configurable CPU matrix extension architecture. By decoupling matrix units from the CPU pipeline, the design enables low-overhead integration while maintaining close coordination with existing compute and memory resources. The configurable matrix unit supports mixed-precision operations and adapts to diverse compute demands and memory bandwidth constraints. An asynchronous matrix multiplication abstraction with flexible granularity conceals hardware details, simplifies matrix-vector overlap, and supports a unified software stack.
The architecture is integrated into four open-source CPU RTL platforms and evaluated on representative AI models. Matrix unit utilization under GEMM workloads exceeds 90% across all platforms. When configured with compute throughput and memory bandwidth comparable to Intel AMX, our design achieves speedups of 1.57x, 1.57x, and 2.31x on ResNet, BERT, and Llama3, with over 30% of the gains attributed to overlapped matrix-vector execution. A 4 TOPS@2GHz matrix unit occupies only 0.53 mm\textsuperscript{2} in 14nm CMOS. These results demonstrate strong cross-platform adaptability and effective hardware-software co-optimization, offering a practical matrix extension for the open-source community. 

---
# Layerwise Dynamics for In-Context Classification in Transformers 

**Authors**: Patrick Lutz, Themistoklis Haris, Arjun Chandra, Aditya Gangrade, Venkatesh Saligrama  

**Link**: [PDF](https://arxiv.org/pdf/2604.11613)  

**Abstract**: Transformers can perform in-context classification from a few labeled examples, yet the inference-time algorithm remains opaque. We study multi-class linear classification in the hard no-margin regime and make the computation identifiable by enforcing feature- and label-permutation equivariance at every layer. This enables interpretability while maintaining functional equivalence and yields highly structured weights. From these models we extract an explicit depth-indexed recursion: an end-to-end identified, emergent update rule inside a softmax transformer, to our knowledge the first of its kind. Attention matrices formed from mixed feature-label Gram structure drive coupled updates of training points, labels, and the test probe. The resulting dynamics implement a geometry-driven algorithmic motif, which can provably amplify class separation and yields robust expected class alignment. 

---
# A Triadic Suffix Tokenization Scheme for Numerical Reasoning 

**Authors**: Olga Chetverina  

**Link**: [PDF](https://arxiv.org/pdf/2604.11582)  

**Abstract**: Standard subword tokenization methods fragment numbers inconsistently, causing large language models (LLMs) to lose positional and decimal structure - a primary driver of errors in arithmetic and scientific reasoning. We introduce Triadic Suffix Tokenization (TST), a deterministic scheme that partitions digits into three-digit triads and annotates each triad with an explicit magnitude marker. Critically, the scheme defines a fixed, one-to-one mapping between suffixes and orders of magnitude for the integer part (thousands, millions, billions, etc.) and a parallel system of replicated markers for fractional depth (tenths, thousandths, millionths, etc.). Unlike approaches that rely on positional inference, this method provides a consistent gradient signal, which should ensure stable convergence. Two implementation variants are proposed: (1) a vocabulary-based approach that adds at most 10,000 fixed tokens to an existing vocabulary, covering 33 orders of magnitude ($10^{-15}$ to $10^{18}$); and (2) a suffix-marker approach that uses a small set of special tokens to denote magnitude dynamically. Both variants preserve exact digits while making order-of-magnitude relationships transparent at the token level. The framework is inherently scalable, allowing for linear vocabulary expansion to accommodate arbitrary precision and range. TST is architecture-agnostic and can be integrated as a drop-in preprocessing step. Experimental validation is deferred to future work. 

---
# Minimizing classical resources in variational measurement-based quantum computation for generative modeling 

**Authors**: Arunava Majumder, Hendrik Poulsen Nautrup, Hans J. Briegel  

**Link**: [PDF](https://arxiv.org/pdf/2604.11578)  

**Abstract**: Measurement-based quantum computation (MBQC) is a framework for quantum information processing in which a computational task is carried out through one-qubit measurements on a highly entangled resource state. Due to the indeterminacy of the outcomes of a quantum measurement, the random outcomes of these operations, if not corrected, yield a variational quantum channel family. Traditionally, this randomness is corrected through classical processing in order to ensure deterministic unitary computations. Recently, variational measurement-based quantum computation (VMBQC) has been introduced to exploit this measurement-induced randomness to gain an advantage in generative modeling. A limitation of this approach is that the corresponding channel model has twice as many parameters compared to the unitary model, scaling as $N \times D$, where $N$ is the number of logical qubits (width) and $D$ is the depth of the VMBQC model. This can often make optimization more difficult and may lead to poorly trainable models. In this paper, we present a restricted VMBQC model that extends the unitary setting to a channel-based one using only a single additional trainable parameter. We show, both numerically and algebraically, that this minimal extension is sufficient to generate probability distributions that cannot be learned by the corresponding unitary model. 

---
# Synthius-Mem: Brain-Inspired Hallucination-Resistant Persona Memory Achieving 94.4% Memory Accuracy and 99.6% Adversarial Robustness on LoCoMo 

**Authors**: Artem Gadzhiev, Andrew Kislov  

**Link**: [PDF](https://arxiv.org/pdf/2604.11563)  

**Abstract**: Providing AI agents with reliable long-term memory that does not hallucinate remains an open problem. Current approaches to memory for LLM agents -- sliding windows, summarization, embedding-based RAG, and flat fact extraction -- each reduce token cost but introduce catastrophic information loss, semantic drift, or uncontrolled hallucination about the user. The structural reason is architectural: every published memory system on the LoCoMo benchmark treats conversation as a retrieval problem over raw or lightly summarized dialogue segments, and none reports adversarial robustness, the ability to refuse questions about facts the user never disclosed. We present Synthius-Mem, a brain-inspired structured persona memory system that takes a fundamentally different approach. Instead of retrieving what was said, Synthius-Mem extracts what is known about the person: a full persona extraction pipeline decomposes conversations into six cognitive domains (biography, experiences, preferences, social circle, work, psychometrics), consolidates and deduplicates per domain, and retrieves structured facts via CategoryRAG at 21.79 ms latency. On the LoCoMo benchmark (ACL 2024, 10 conversations, 1,813 questions), Synthius-Mem achieves 94.37% accuracy, exceeding all published systems including MemMachine (91.69%, adversarial score is not reported) and human performance (87.9 F1). Core memory fact accuracy reaches 98.64%. Adversarial robustness, the hallucination resistance metric that no competing system reports, reaches 99.55%. Synthius-Mem reduces token consumption by ~5x compared to full-context replay while achieving higher accuracy. Synthius-Mem achieves state-of-the-art results on LoCoMo and is, to our knowledge, the only persona memory system that both exceeds human-level performance and reports adversarial robustness. 

---
# bacpipe: a Python package to make bioacoustic deep learning models accessible 

**Authors**: Vincent S. Kather, Sylvain Haupert, Burooj Ghani, Dan Stowell  

**Link**: [PDF](https://arxiv.org/pdf/2604.11560)  

**Abstract**: 1. Natural sounds have been recorded for millions of hours over the previous decades using passive acoustic monitoring. Improvements in deep learning models have vastly accelerated the analysis of large portions of this data. While new models advance the state-of-the-art, accessing them using tools to harness their full potential is not always straightforward. Here we present bacpipe, a collection of bioacoustic deep learning models and evaluation pipelines accessible through a graphical and programming interface, designed for both ecologists and computer scientists. Bacpipe is a modular software package intended as a point of convergence for bioacoustic models.
2. Bacpipe streamlines the usage of state-of-the-art models on custom audio datasets, generating acoustic feature vectors (embeddings) and classifier predictions. A modular design allows evaluation and benchmarking of models through interactive visualizations, clustering and probing.
3. We believe that access to new deep learning models is important. By designing bacpipe to target a wide audience, researchers will be enabled to answer new ecological and evolutionary questions in bioacoustics.
4. In conclusion, we believe accessibility to developments in deep learning to a wider audience benefits the ecological questions we are trying to answer. 

---
# FM-Agent: Scaling Formal Methods to Large Systems via LLM-Based Hoare-Style Reasoning 

**Authors**: Haoran Ding, Zhaoguo Wang, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11556)  

**Abstract**: LLM-assisted software development has become increasingly prevalent, and can generate large-scale systems, such as compilers. It becomes crucial to strengthen the correctness of the generated code. However, automated reasoning for large-scale systems remains challenging due to code complexity. Hoare logic offers an approach to decomposing a large system into smaller components and reasoning about them separately (i.e., compositional reasoning). However, existing works still struggle to scale, because Hoare logic requires writing formal specifications for each function, imposing a heavy human burden. The problem is exacerbated when code is generated by LLMs, as developers lack a deep understanding of each function's expected behavior.
This paper presents FM-Agent, the first framework that realizes automated compositional reasoning for large-scale systems. Leveraging LLMs, FM-Agent introduces a top-down paradigm to automatically generate function-level specifications. Specifically, FM-Agent derives the specification of a function from how its callers expect the function to behave, so the generated specifications can reflect the developer's intent of a function even if the implementation is buggy. Developers' intent is usually expressed in natural language, while existing verifiers only support formulas. Therefore, FM-Agent generalizes Hoare-style inference to reason about functions against natural-language specifications. Finally, to confirm bug existence and explain bug causes, FM-Agent automatically generates test cases to trigger potential bugs. In our evaluation, FM-Agent successfully reasons about large-scale systems within 2 days, each of which has up to 143k LoC. These systems have already been tested by their developers, but FM-Agent still finds 522 newly discovered bugs. These bugs can cause serious consequences, including system crashes and incorrect execution results. 

---
# Time is Not a Label: Continuous Phase Rotation for Temporal Knowledge Graphs and Agentic Memory 

**Authors**: Weixian Waylon Li, Jiaxin Zhang, Xianan Jim Yang, Tiejun Ma, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.11544)  

**Abstract**: Structured memory representations such as knowledge graphs are central to autonomous agents and other long-lived systems. However, most existing approaches model time as discrete metadata, either sorting by recency (burying old-yet-permanent knowledge), simply overwriting outdated facts, or requiring an expensive LLM call at every ingestion step, leaving them unable to distinguish persistent facts from evolving ones. To address this, we introduce RoMem, a drop-in temporal knowledge graph module for structured memory systems, applicable to agentic memory and beyond. A pretrained Semantic Speed Gate maps each relation's text embedding to a volatility score, learning from data that evolving relations (e.g., "president of") should rotate fast while persistent ones (e.g., "born in") should remain stable. Combined with continuous phase rotation, this enables geometric shadowing: obsolete facts are rotated out of phase in complex vector space, so temporally correct facts naturally outrank contradictions without deletion. On temporal knowledge graph completion, RoMem achieves state-of-the-art results on ICEWS05-15 (72.6 MRR). Applied to agentic memory, it delivers 2-3x MRR and answer accuracy on temporal reasoning (MultiTQ), dominates hybrid benchmark (LoCoMo), preserves static memory with zero degradation (DMR-MSC), and generalises zero-shot to unseen financial domains (FinTMMBench). 

---
# NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment 

**Authors**: Wenqing Wu, Yi Zhao, Yuzhuo Wang, Siyou Li, Juexi Shao, Yunfei Long, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11543)  

**Abstract**: Novelty is a core requirement in academic publishing and a central focus of peer review, yet the growing volume of submissions has placed increasing pressure on human reviewers. While large language models (LLMs), including those fine-tuned on peer review data, have shown promise in generating review comments, the absence of a dedicated benchmark has limited systematic evaluation of their ability to assess research novelty. To address this gap, we introduce NovBench, the first large-scale benchmark designed to evaluate LLMs' capability to generate novelty evaluations in support of human peer review. NovBench comprises 1,684 paper-review pairs from a leading NLP conference, including novelty descriptions extracted from paper introductions and corresponding expert-written novelty evaluations. We focus on both sources because the introduction provides a standardized and explicit articulation of novelty claims, while expert-written novelty evaluations constitute one of the current gold standards of human judgment. Furthermore, we propose a four-dimensional evaluation framework (including Relevance, Correctness, Coverage, and Clarity) to assess the quality of LLM-generated novelty evaluations. Extensive experiments on both general and specialized LLMs under different prompting strategies reveal that current models exhibit limited understanding of scientific novelty, and that fine--tuned models often suffer from instruction-following deficiencies. These findings underscore the need for targeted fine-tuning strategies that jointly improve novelty comprehension and instruction adherence. 

---
# CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space 

**Authors**: Sohwi Lim, Lee Hyoseok, Jungjoon Park, Tae-Hyun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2604.11539)  

**Abstract**: Human perception of visual similarity is inherently adaptive and subjective, depending on the users' interests and focus. However, most image retrieval systems fail to reflect this flexibility, relying on a fixed, monolithic metric that cannot incorporate multiple conditions simultaneously. To address this, we propose CLAY, an adaptive similarity computation method that reframes the embedding space of pretrained Vision-Language Models (VLMs) as a text-conditional similarity space without additional training. This design separates the textual conditioning process and visual feature extraction, allowing highly efficient and multi-conditioned retrieval with fixed visual embeddings. We also construct a synthetic evaluation dataset CLAY-EVAL, for comprehensive assessment under diverse conditioned retrieval settings. Experiments on standard datasets and our proposed dataset show that CLAY achieves high retrieval accuracy and notable computational efficiency compared to previous works. 

---
# SVD-Prune: Training-Free Token Pruning For Efficient Vision-Language Models 

**Authors**: Yvon Apedo, Martyna Poreba, Michal Szczepanski, Samia Bouchafa  

**Link**: [PDF](https://arxiv.org/pdf/2604.11530)  

**Abstract**: Vision-Language Models (VLM) have revolutionized multimodal learning by jointly processing visual and textual information. Yet, they face significant challenges due to the high computational and memory demands of processing long sequences of vision tokens. Many existing methods rely on local heuristics, such as attention scores or token norms. However, these criteria suffer from positional bias and information dispersion, limiting their ability to preserve essential content at high pruning ratios and leading to performance degradation on visually detailed images. To address these issues, we propose SVD-Prune, a trainingfree, plug-and-play token pruning method based on Singular Value Decomposition. It decomposes the vision token feature matrix and selects the top-K tokens using statistical leverage scores, ensuring only tokens contributing most to the dominant global variance are preserved. Experiments show that SVD-Prune consistently outperforms prior pruning methods under extreme vision token budgets, maintaining strong performance even with 32 and 16 vision tokens. 

---
# From Translation to Superset: Benchmark-Driven Evolution of a Production AI Agent from Rust to Python 

**Authors**: Jinhua Wang, Biswa Sengupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.11518)  

**Abstract**: Cross-language migration of large software systems is a persistent engineering challenge, particularly when the source codebase evolves rapidly. We present a methodology for LLM-assisted continuous code translation in which a large language model translates a production Rust codebase (648K LOC, 65 crates) into Python (41K LOC, 28 modules), with public agent benchmarks as the objective function driving iterative refinement. Our subject system is Codex CLI, a production AI coding agent. We demonstrate that: (1) the Python port resolves 59/80 SWE-bench Verified tasks (73.8%) versus Rust's 56/80 (70.0%), and achieves 42.5% on Terminal-Bench versus Rust's 47.5%, confirming near-parity on real-world agentic tasks; (2) benchmark-driven debugging, revealing API protocol mismatches, environment pollution, a silent WebSocket failure mode, and an API 400 crash, is more effective than static testing alone; (3) the architecture supports continuous upstream synchronisation via an LLM-assisted diff-translate-test loop; and (4) the Python port has evolved into a capability superset with 30 feature-flagged extensions (multi-agent orchestration, semantic memory, guardian safety, cost tracking) absent from Rust, while preserving strict parity mode for comparison. Our evaluation shows that for LLM-based agents where API latency dominates, Python's expressiveness yields a 15.9x code reduction with negligible performance cost, while the benchmark-as-objective-function methodology provides a principled framework for growing a cross-language port from parity into an extended platform. 

---
# EdgeCIM: A Hardware-Software Co-Design for CIM-Based Acceleration of Small Language Models 

**Authors**: Jinane Bazzi, Mariam Rakka, Fadi Kurdahi, Mohammed E. Fouda, Ahmed Eltawil  

**Link**: [PDF](https://arxiv.org/pdf/2604.11512)  

**Abstract**: The growing demand for deploying Small Language Models (SLMs) on edge devices, including laptops, smartphones, and embedded platforms, has exposed fundamental inefficiencies in existing accelerators. While GPUs handle prefill workloads efficiently, the autoregressive decoding phase is dominated by GEMV operations that are inherently memory-bound, resulting in poor utilization and prohibitive energy costs at the edge. In this work, we present EdgeCIM, a hardware-software co-design framework that rethinks accelerator design for end-to-end decoder-only inference. At its core is a CIM macro, implemented in 65nm, coupled with a tile-based mapping strategy that balances pipeline stages, maximizing parallelism while alleviating DRAM bandwidth bottlenecks. Our simulator enables design space exploration of SLMs up to 4B parameters, identifying Pareto-optimal configurations in terms of latency and energy. Compared to an NVIDIA Orin Nano, EdgeCIM achieves up to 7.3x higher throughput and 49.59x better energy efficiency on LLaMA3.2-1B, and delivers 9.95x higher throughput than Qualcomm SA8255P on LLaMA3.2-3B. Extensive benchmarks on TinyLLaMA-1.1B, LLaMA3.2 (1B, 3B), Phi-3.5-mini-3.8B, Qwen2.5 (0.5B, 1.5B, 3B), SmolLM2-1.7B, SmolLM3-3B, and Qwen3 (0.6B, 1.7B, 4B) reveal that our accelerator, under INT4 precision, achieves on average 336.42 tokens/s and 173.02 tokens/J. These results establish EdgeCIM as a compelling solution towards real-time, energy-efficient edge-scale SLM inference. 

---
# Policy Split: Incentivizing Dual-Mode Exploration in LLM Reinforcement with Dual-Mode Entropy Regularization 

**Authors**: Jiashu Yao, Heyan Huang, Chuwei Luo, Daiqing Wu, Zeming Liu, Yuhang Guo, Yangyang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11510)  

**Abstract**: To encourage diverse exploration in reinforcement learning (RL) for large language models (LLMs) without compromising accuracy, we propose Policy Split, a novel paradigm that bifurcates the policy into normal and high-entropy modes with a high-entropy prompt. While sharing model parameters, the two modes undergo collaborative dual-mode entropy regularization tailored to distinct objectives. Specifically, the normal mode optimizes for task correctness, while the high-entropy mode incorporates a preference for exploration, and the two modes learn collaboratively. Extensive experiments demonstrate that our approach consistently outperforms established entropy-guided RL baselines across various model sizes in general and creative tasks. Further analysis reveals that Policy Split facilitates dual-mode exploration, where the high-entropy mode generates distinct behavioral patterns to the normal mode, providing unique learning signals. 

---
# Not All Forgetting Is Equal: Architecture-Dependent Retention Dynamics in Fine-Tuned Image Classifiers 

**Authors**: Miit Daga, Swarna Priya Ramu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11508)  

**Abstract**: Fine-tuning pretrained image classifiers is standard practice, yet which individual samples are forgotten during this process, and whether forgetting patterns are stable or architecture dependent, remains unclear. Understanding these dynamics has direct implications for curriculum design, data pruning, and ensemble construction. We track per-sample correctness at every epoch during fine-tuning of ResNet-18 and DeiT-Small on a retinal OCT dataset (7 classes, 56:1 imbalance) and CUB-200-2011 (200 bird species), fitting Ebbinghaus-style exponential decay curves to each sample's retention trace. Five findings emerge. First, the two architectures forget fundamentally different samples: Jaccard overlap of the top 10 percent most-forgotten is 0.34 on OCTDL and 0.15 on CUB-200. Second, ViT forgetting is more structured (mean $R^2 = 0.74$) than CNN forgetting ($R^2 = 0.52$). Third, per-sample forgetting is stochastic across random seeds (Spearman $\rho \approx 0.01$), challenging the assumption that sample difficulty is an intrinsic property. Fourth, class-level forgetting is consistent and semantically interpretable: visually similar species are forgotten most, distinctive ones least. Fifth, a sample's loss after head warmup predicts its long-term decay constant ($\rho = 0.30$ to $0.50$, $p < 10^{-45}$). These findings suggest that architectural diversity in ensembles provides complementary retention coverage, and that curriculum or pruning methods based on per-sample difficulty may not generalize across runs. A spaced repetition sampler built on these decay constants does not outperform random sampling, indicating that static scheduling cannot exploit unstable per-sample signals. 

---
# Deep Learning for Sequential Decision Making under Uncertainty: Foundations, Frameworks, and Frontiers 

**Authors**: I. Esra Buyuktahtakin  

**Link**: [PDF](https://arxiv.org/pdf/2604.11507)  

**Abstract**: Artificial intelligence (AI) is moving increasingly beyond prediction to support decisions in complex, uncertain, and dynamic environments. This shift creates a natural intersection with operations research and management sciences (OR/MS), which have long offered conceptual and methodological foundations for sequential decision-making under uncertainty. At the same time, recent advances in deep learning, including feedforward neural networks, LSTMs, transformers, and deep reinforcement learning, have expanded the scope of data-driven modeling and opened new possibilities for large-scale decision systems. This tutorial presents an OR/MS-centered perspective on deep learning for sequential decision-making under uncertainty. Its central premise is that deep learning is valuable not as a replacement for optimization, but as a complement to it. Deep learning brings adaptability and scalable approximation, whereas OR/MS provides the structural rigor needed to represent constraints, recourse, and uncertainty. The tutorial reviews key decision-making foundations, connects them to the major neural architectures in modern AI, and discusses leading approaches to integrating learning and optimization. It also highlights emerging impact in domains such as supply chains, healthcare and epidemic response, agriculture, energy, and autonomous operations. More broadly, it frames these developments as part of a wider transition from predictive AI toward decision-capable AI and highlights the role of OR/MS in shaping the next generation of integrated learning--optimization systems. 

---
# METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models 

**Authors**: Pengfeng Li, Chen Huang, Chaoqun Hao, Hongyao Chen, Xiao-Yong Wei, Wenqiang Lei, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.11502)  

**Abstract**: Contextual causal reasoning is a critical yet challenging capability for Large Language Models (LLMs). Existing benchmarks, however, often evaluate this skill in fragmented settings, failing to ensure context consistency or cover the full causal hierarchy. To address this, we pioneer METER to systematically benchmark LLMs across all three levels of the causal ladder under a unified context setting. Our extensive evaluation of various LLMs reveals a significant decline in proficiency as tasks ascend the causal hierarchy. To diagnose this degradation, we conduct a deep mechanistic analysis via both error pattern identification and internal information flow tracing. Our analysis reveals two primary failure modes: (1) LLMs are susceptible to distraction by causally irrelevant but factually correct information at lower level of causality; and (2) as tasks ascend the causal hierarchy, faithfulness to the provided context degrades, leading to a reduced performance. We belive our work advances our understanding of the mechanisms behind LLM contextual causal reasoning and establishes a critical foundation for future research. Our code and dataset are available at this https URL . 

---
# Quantization Dominates Rank Reduction for KV-Cache Compression 

**Authors**: Samuel Salfati  

**Link**: [PDF](https://arxiv.org/pdf/2604.11501)  

**Abstract**: We compare two strategies for compressing the KV cache in transformer inference: rank reduction (discard dimensions) and quantization (keep all dimensions, reduce precision). At matched storage budgets across five models (124M-14B, MHA and GQA), we find that quantization consistently outperforms rank reduction by 4-364 PPL depending on model and compression level. The gap persists even when rank reduction is combined with quantization in hybrid baselines, and it grows with GQA aggressiveness. On LAMBADA, INT4 matches FP16 accuracy (+0.23 PPL on Mistral 7B, +0.58 on GPT-2) while rank-32 at identical storage collapses to 0.4%.
We trace this gap to a structural asymmetry: under softmax attention routing, removing a dimension can flip which token is attended (a discrete failure), while quantization noise is bounded and typically preserves score ordering. We formalize this via a perturbation result showing projection damage exceeds quantization damage by 3 x 2^(2b) per direction under the softmax Fisher metric. A basis ablation confirms the finding is basis-independent (spread <0.4 PPL), establishing that the advantage comes from preserving dimensions, not from a better coordinate system. Joint K+V INT4 quantization achieves 75% total KV reduction at only +0.18 PPL on Mistral 7B. 

---
# ADD for Multi-Bit Image Watermarking 

**Authors**: An Luo, Jie Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.11491)  

**Abstract**: As generative models enable rapid creation of high-fidelity images, societal concerns about misinformation and authenticity have intensified. A promising remedy is multi-bit image watermarking, which embeds a multi-bit message into an image so that a verifier can later detect whether the image is generated by someone and further identify the source by decoding the embedded message. Existing approaches often fall short in capacity, resilience to common image distortions, and theoretical justification. To address these limitations, we propose ADD (Add, Dot, Decode), a multi-bit image watermarking method with two stages: learning a watermark to be linearly combined with the multi-bit message and added to the image, and decoding through inner products between the watermarked image and the learned watermark. On the standard MS-COCO benchmark, we demonstrate that for the challenging task of 48-bit watermarking, ADD achieves 100\% decoding accuracy, with performance dropping by at most 2\% under a wide range of image distortions, substantially smaller than the 14\% average drop of state-of-the-art methods. In addition, ADD achieves substantial computational gains, with 2-fold faster embedding and 7.4-fold faster decoding than the fastest existing method. We further provide a theoretical analysis explaining why the learned watermark and the corresponding decoding rule are effective. 

---
# SLALOM: Simulation Lifecycle Analysis via Longitudinal Observation Metrics for Social Simulation 

**Authors**: Juhoon Lee, Joseph Seering  

**Link**: [PDF](https://arxiv.org/pdf/2604.11466)  

**Abstract**: Large Language Model (LLM) agents offer a potentially-transformative path forward for generative social science but face a critical crisis of validity. Current simulation evaluation methodologies suffer from the "stopped clock" problem: they confirm that a simulation reached the correct final outcome while ignoring whether the trajectory leading to it was sociologically plausible. Because the internal reasoning of LLMs is opaque, verifying the "black box" of social mechanisms remains a persistent challenge. In this paper, we introduce SLALOM (Simulation Lifecycle Analysis via Longitudinal Observation Metrics), a framework that shifts validation from outcome verification to process fidelity. Drawing on Pattern-Oriented Modeling (POM), SLALOM treats social phenomena as multivariate time series that must traverse specific SLALOM gates, or intermediate waypoint constraints representing distinct phases. By utilizing Dynamic Time Warping (DTW) to align simulated trajectories with empirical ground truth, SLALOM offers a quantitative metric to assess structural realism, helping to differentiate plausible social dynamics from stochastic noise and contributing to more robust policy simulation standards. 

---
# Low-rank Optimization Trajectories Modeling for LLM RLVR Acceleration 

**Authors**: Zhipeng Chen, Tao Qian, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.11446)  

**Abstract**: Recently, scaling reinforcement learning with verifiable rewards (RLVR) for large language models (LLMs) has emerged as an effective training paradigm for significantly improving model capabilities, which requires guiding the model to perform extensive exploration and learning, leading to substantial computational overhead and becoming a key challenge. To reduce the number of training steps, Prior work performs linear extrapolation of model parameters. However, the dynamics of model parameter updates during RLVR training remain insufficiently understood. To further investigate the evolution of LLMs during RLVR training, we conduct empirical experiments and find that the rank-1 subspace of the model does not evolve linearly, and its dominance over the original parameters is further amplified during LoRA training. Based on the above insights, we propose the \textbf{N}onlinear \textbf{Ext}rapolation of low-rank trajectories (\textbf{NExt}), a novel framework that models and extrapolates low-rank parameter trajectories in a nonlinear manner. Concretely, we first train the model using LoRA and extract the rank-1 subspace of parameter differences at multiple training steps, which is then used for the subsequent nonlinear extrapolation. Afterward, we utilized the extracted rank-1 subspace to train a predictor, which can model the trajectory of parameter updates during RLVR, and then perform the predict-extend process to extrapolate model parameters, achieving the acceleration of RLVR. To further study and understand NExt, we conduct comprehensive experiments that demonstrate the effectiveness and robustness of the method. Our method reduces computational overhead by approximately 37.5\% while remaining compatible with a wide range of RLVR algorithms and tasks. We release our code in this https URL. 

---
# Think Before you Write: QA-Guided Reasoning for Character Descriptions in Books 

**Authors**: Argyrios Papoudakis, Mirella Lapata, Frank Keller  

**Link**: [PDF](https://arxiv.org/pdf/2604.11435)  

**Abstract**: Character description generation is an important capability for narrative-focused applications such as summarization, story analysis, and character-driven simulations. However, generating accurate character descriptions from long-form narratives (e.g., novels) is challenging: models must track evolving attributes (e.g., relationships and events), integrate evidence scattered across the text, and infer implicit details. Despite the success of reasoning-enabled LLMs on many benchmarks, we find that for character description generation their performance improves when built-in reasoning is disabled (i.e., an empty reasoning trace). Motivated by this, we propose a training framework that decouples reasoning from generation. Our approach, which can be applied on top of long-context LLMs or chunk-based methods, consists of a reasoning model that produces a structured QA reasoning trace and a generation model that conditions on this trace to produce the final character description. Experiments on two datasets (BookWorm and CroSS) show that QA-guided reasoning improves faithfulness, informativeness, and grounding over strong long-context baselines. 

---
# Hardening x402: PII-Safe Agentic Payments via Pre-Execution Metadata Filtering 

**Authors**: Vladimir Stantchev  

**Link**: [PDF](https://arxiv.org/pdf/2604.11430)  

**Abstract**: AI agents that pay for resources via the x402 protocol embed payment metadata - resource URLs, descriptions, and reason strings - in every HTTP payment request. This metadata is transmitted to the payment server and to the centralised facilitator API before any on-chain settlement occurs; neither party is typically bound by a data processing agreement. We present presidio-hardened-x402, the first open-source middleware that intercepts x402 payment requests before transmission to detect and redact personally identifiable information (PII), enforce declarative spending policies, and block duplicate replay attempts. To evaluate the PII filter, we construct a labeled synthetic corpus of 2,000 x402 metadata triples spanning seven use-case categories, and run a 42-configuration precision/recall sweep across two detection modes (regex, NLP) and five confidence thresholds. The recommended configuration (mode=nlp, min_score=0.4, all entity types) achieves micro-F1 = 0.894 with precision 0.972, at a p99 latency of 5.73ms - well within the 50ms overhead budget. The middleware, corpus, and all experiment code are publicly available at this https URL. 

---
# METRO: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues 

**Authors**: Haofu Yang, Jiaji Liu, Chen Huang, Faguo Wu, Wenqiang Lei, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.11427)  

**Abstract**: Developing non-collaborative dialogue agents traditionally requires the manual, unscalable codification of expert strategies. We propose \ours, a method that leverages large language models to autonomously induce both strategy actions and planning logic directly from raw transcripts. METRO formalizes expert knowledge into a Strategy Forest, a hierarchical structure that captures both short-term responses (nodes) and long-term strategic foresight (branches). Experimental results across two benchmarks show that METRO demonstrates promising performance, outperforming existing methods by an average of 9%-10%. Our further analysis not only reveals the success behind METRO (strategic behavioral diversity and foresight), but also demonstrates its robust cross-task transferability. This offers new insights into building non-collaborative agents in a cost-effective and scalable way. Our code is available at this https URL. 

---
# Emulating Non-Differentiable Metrics via Knowledge-Guided Learning: Introducing the Minkowski Image Loss 

**Authors**: Filippo Quarenghi, Ryan Cotsakis, Tom Beucler  

**Link**: [PDF](https://arxiv.org/pdf/2604.11422)  

**Abstract**: The ``differentiability gap'' presents a primary bottleneck in Earth system deep learning: since models cannot be trained directly on non-differentiable scientific metrics and must rely on smooth proxies (e.g., MSE), they often fail to capture high-frequency details, yielding ``blurry'' outputs. We develop a framework that bridges this gap using two different methods to deal with non-differentiable functions: the first is to analytically approximate the original non-differentiable function into a differentiable equivalent one; the second is to learn differentiable surrogates for scientific functionals. We formulate the analytical approximation by relaxing discrete topological operations using temperature-controlled sigmoids and continuous logical operators. Conversely, our neural emulator uses Lipschitz-convolutional neural networks to stabilize gradient learning via: (1) spectral normalization to bound the Lipschitz constant; and (2) hard architectural constraints enforcing geometric principles. We demonstrate this framework's utility by developing the Minkowski image loss, a differentiable equivalent for the integral-geometric measures of surface precipitation fields (area, perimeter, connected components). Validated on the EUMETNET OPERA dataset, our constrained neural surrogate achieves high emulation accuracy, completely eliminating the geometric violations observed in unconstrained baselines. However, applying these differentiable surrogates to a deterministic super-resolution task reveals a fundamental trade-off: while strict Lipschitz regularization ensures optimization stability, it inherently over-smooths gradient signals, restricting the recovery of highly localized convective textures. This work highlights the necessity of coupling such topological constraints with stochastic generative architectures to achieve full morphological realism. 

---
# Efficient Emotion-Aware Iconic Gesture Prediction for Robot Co-Speech 

**Authors**: Edwin C. Montiel-Vazquez, Christian Arzate Cruz, Stefanos Gkikas, Thomas Kassiotis, Giorgos Giannakakis, Randy Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2604.11417)  

**Abstract**: Co-speech gestures increase engagement and improve speech understanding. Most data-driven robot systems generate rhythmic beat-like motion, yet few integrate semantic emphasis. To address this, we propose a lightweight transformer that derives iconic gesture placement and intensity from text and emotion alone, requiring no audio input at inference time. The model outperforms GPT-4o in both semantic gesture placement classification and intensity regression on the BEAT2 dataset, while remaining computationally compact and suitable for real-time deployment on embodied agents. 

---
# Retrieval as Generation: A Unified Framework with Self-Triggered Information Planning 

**Authors**: Bo Li, Mingda Wang, Gexiang Fang, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.11407)  

**Abstract**: We revisit retrieval-augmented generation (RAG) by embedding retrieval control directly into generation. Instead of treating retrieval as an external intervention, we express retrieval decisions within token-level decoding, enabling end-to-end coordination without additional controllers or classifiers. Under the paradigm of Retrieval as Generation, we propose \textbf{GRIP} (\textbf{G}eneration-guided \textbf{R}etrieval with \textbf{I}nformation \textbf{P}lanning), a unified framework in which the model regulates retrieval behavior through control-token emission. Central to GRIP is \textit{Self-Triggered Information Planning}, which allows the model to decide when to retrieve, how to reformulate queries, and when to terminate, all within a single autoregressive trajectory. This design tightly couples retrieval and reasoning and supports dynamic multi-step inference with on-the-fly evidence integration. To supervise these behaviors, we construct a structured training set covering answerable, partially answerable, and multi-hop queries, each aligned with specific token patterns. Experiments on five QA benchmarks show that GRIP surpasses strong RAG baselines and is competitive with GPT-4o while using substantially fewer parameters. 

---
# One Scale at a Time: Scale-Autoregressive Modeling for Fluid Flow Distributions 

**Authors**: Mario Lino, Nils Thuerey  

**Link**: [PDF](https://arxiv.org/pdf/2604.11403)  

**Abstract**: Analyzing unsteady fluid flows often requires access to the full distribution of possible temporal states, yet conventional PDE solvers are computationally prohibitive and learned time-stepping surrogates quickly accumulate error over long rollouts. Generative models avoid compounding error by sampling states independently, but diffusion and flow-matching methods, while accurate, are limited by the cost of many evaluations over the entire mesh. We introduce scale-autoregressive modeling (SAR) for sampling flows on unstructured meshes hierarchically from coarse to fine: it first generates a low-resolution field, then refines it by progressively sampling higher resolutions conditioned on coarser predictions. This coarse-to-fine factorization improves efficiency by concentrating computation at coarser scales, where uncertainty is greatest, while requiring fewer steps at finer scales. Across unsteady-flow benchmarks of varying complexity, SAR attains substantially lower distributional error and higher per-sample accuracy than state-of-the-art diffusion models based on multi-scale GNNs, while matching or surpassing a flow-matching Transolver (a linear-time transformer) yet running 2-7x faster than this depending on the task. Overall, SAR provides a practical tool for fast and accurate estimation of statistical flow quantities (e.g., turbulent kinetic energy and two-point correlations) in real-world settings. 

---
# From Redaction to Restoration: Deep Learning for Medical Image Anonymization and Reconstruction 

**Authors**: Adrienne Kline, Abhijit Gaonkar, Daniel Pittman, Chris Kuehn, Nils Forkert  

**Link**: [PDF](https://arxiv.org/pdf/2604.11376)  

**Abstract**: Removing patient-specific information from medical images is crucial to enable sharing and open science without compromising patient identities. However, many methods currently used for deidentification have negative effects on downstream image analysis tasks because of removal of relevant but non-identifiable information. This work presents an end-to-end deep learning framework for transforming raw clinical image volumes into de-identified, analysis-ready datasets without compromising downstream utility. The methodology developed and tested in this work first detects and redacts regions likely to contain protected health information (PHI), such as burned-in text and metadata, and then uses a generative deep learning model to inpaint the redacted areas with anatomically and imaging plausible content. The proposed pipeline leverages a lightweight hybrid architecture, combining CRNN-based redaction with a latent-diffusion inpainting restoration module (Stable Diffusion 2). We evaluate the approach using both privacy-oriented metrics, which quantify residual PHI and success of redaction, and image-quality and task-based metrics, which assess the fidelity of restored volumes for representative deep learning applications. Our results suggest that the proposed method yields de-identified medical images that are visually coherent, maintaining fidelity for downstream models, while substantially reducing the risk of patient re-identification. By automating anonymization and image reconstruction within a single workflow, and dissemination of large-scale medical imaging collections, thereby lowering a key barrier to data sharing and multi-institutional collaboration in medical imaging AI. 

---
# Minimal Embodiment Enables Efficient Learning of Number Concepts in Robot 

**Authors**: Zhegong Shangguan, Alessandro Di Nuovo, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2604.11373)  

**Abstract**: Robots are increasingly entering human-interactive scenarios that require understanding of quantity. How intelligent systems acquire abstract numerical concepts from sensorimotor experience remains a fundamental challenge in cognitive science and artificial intelligence. Here we investigate embodied numerical learning using a neural network model trained to perform sequential counting through naturalistic robotic interaction with a Franka Panda manipulator. We demonstrate that embodied models achieve 96.8\% counting accuracy with only 10\% of training data, compared to 60.6\% for vision-only baselines. This advantage persists when visual-motor correspondences are randomized, indicating that embodiment functions as a structural prior that regularizes learning rather than as an information source. The model spontaneously develops biologically plausible representations: number-selective units with logarithmic tuning, mental number line organization, Weber-law scaling, and rotational dynamics encoding numerical magnitude ($r = 0.97$, slope $= 30.6°$/count). The learning trajectory parallels children's developmental progression from subset-knowers to cardinal-principle knowers. These findings demonstrate that minimal embodiment can ground abstract concepts, improve data efficiency, and yield interpretable representations aligned with biological cognition, which may contribute to embodied mathematics tutoring and safety-critical industrial applications. 

---
# Governance by Design: A Parsonian Institutional Architecture for Internet-Wide Agent Societies 

**Authors**: Anbang Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2604.11337)  

**Abstract**: The dominant paradigm of local multi-agent systems -- orchestrated, enterprise-bounded pipelines -- is being superseded by internet-wide agent societies in which autonomous agents discover each other through open registries, interact without central orchestrators, and generate emergent social behaviors. We argue that governing such societies requires institutional design, not merely risk enumeration or process compliance. Applying Talcott Parsons' AGIL framework -- four functional imperatives (Adaptation, Goal Attainment, Integration, Latency) every viable social system must satisfy -- we derive a prescriptive sixteen-cell institutional architecture for internet-wide agent governance. Diagnostically applied to the OpenClaw ecosystem (250,000+ GitHub stars, 2M+ monthly users, 770,000+ registered agents) via a recursive sub-function analysis (64 binary indicators across 16 cells), we find at most 19% sub-function coverage (sensitivity range 17-30%) -- potential rather than operative capacity, since zero inter-cell coordination prevents existing infrastructure from participating in inter-pillar interchange. A complementary interchange media assessment finds zero of twelve inter-pillar pathways functional: the ecosystem has technical infrastructure but no active governance, no coordination layer, and no normative grounding, with the Fiduciary and Political pillars most severely underserved. Extending the diagnostic to the broader agent-native protocol stack (MCP, A2A, ANP, x402, ERC-8004), independent development teams reproduce the same structural pattern -- confirming the governance gap is a feature of market-driven development, not ecosystem immaturity. Institutional design is most effective before social patterns calcify; we conclude with a prioritized roadmap for the missing governance infrastructure. 

---
# A Compact and Efficient 1.251 Million Parameter Machine Learning CNN Model PD36-C for Plant Disease Detection: A Case Study 

**Authors**: Shkelqim Sherifi  

**Link**: [PDF](https://arxiv.org/pdf/2604.11332)  

**Abstract**: Deep learning has markedly advanced image based plant disease diagnosis as improved hardware and dataset quality have enabled increasingly accurate neural network models. This paper presents PD36 C, a compact convolutional neural network (1,250,694 parameters and 4.77 MB) for plant disease classification. Trained with TensorFlow Keras on the New Plant Diseases Dataset (87k images, 38 classes), PD36 C is designed for robustness and edge deployability, complemented by a Qt for Python desktop application that offers an intuitive GUI and offline inference on commodity hardware. Across experiments, training accuracy reached 0.99697 by epoch 30, and average test accuracy was 0.9953 across 38 classes. Per class performance is uniformly high; on the lower end, Corn (maize) Cercospora leaf spot achieved precision around 0.9777 and recall around 0.9634, indicating occasional confusion with visually similar categories, while on the upper end numerous classes including Apple Black rot, Cedar apple rust, Blueberry healthy, Cherry Powdery mildew, Cherry healthy, and all four grape categories achieved perfect precision 1.00 and recall of 1.00, indicating no false positives and strong coverage. These results show that with a well curated dataset and careful architectural design, small CNNs can achieve competitive accuracy compared with recent baselines while remaining practical for edge scenarios. We also note typical constraints such as adverse weather, low quality imagery, and leaves exhibiting multiple concurrent diseases that can degrade performance and warrant future work on domain robustness. Overall, PD36 C and its application pipeline contribute a field ready, efficient solution for AI assisted plant disease detection in smart agriculture. 

---
# Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations 

**Authors**: Yilong Liu, Xixun Lin, Pengfei Cao, Ge Zhang, Fang Fang, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.11322)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in utilizing external tools. In practice, however, LLMs are often exposed to tools that are irrelevant to the user's query, in which case the desired behavior is to refrain from invocations. In this work, we identify a widespread yet overlooked mechanistic flaw in tool refusal, which we term structural alignment bias: Even when a tool fails to serve the user's goal, LLMs still tend to invoke it whenever query attributes can be validly assigned to tool parameters. To systematically study this bias, we introduce SABEval, a new dataset that decouples structural alignment from semantic relevance. Our analysis shows that structural alignment bias induces severe tool-invocation errors in LLMs, yet remains largely unaccounted for in existing evaluations. To investigate the internal mechanisms underlying this bias, we propose Contrastive Attention Attribution, which reveals two competing pathways for semantic checking and structural matching. The relative strength of these pathways drives LLMs' tool invocation decisions. Based on these findings, we further introduce a rebalancing strategy that effectively mitigates structural alignment bias, as demonstrated by extensive experiments, without degrading general tool-use capabilities. 

---
# S$^3$: Structured Sparsity Specification 

**Authors**: Ayoub Ghriss  

**Link**: [PDF](https://arxiv.org/pdf/2604.11315)  

**Abstract**: We introduce the Structured Sparsity Specification (S$^3$), an algebraic framework for defining, composing, and implementing structured sparse patterns. S$^3$ specifies sparsity through three components: a View that reshapes the tensor via layout composition, a Block specification that defines the atomic pruning unit, and the sparsity decision Scope. Both Block and Scope support Coupling across tensors for coordinated sparsification. S$^3$ enables precise specification of diverse sparsity structures, from fine-grained N:M patterns to coarse channel pruning, and integrates seamlessly with Optimal Brain Damage (OBD) and Surgeon (OBS). We formalize the framework mathematically, demonstrate its expressiveness on canonical patterns, and validate it experimentally via structured OBS and OBD implementations built entirely on S$^3$, which surpasses well-established second order heuristics on output reconstruction across common configurations. 

---
# Network Effects and Agreement Drift in LLM Debates 

**Authors**: Erica Cau, Andrea Failla, Giulio Rossetti  

**Link**: [PDF](https://arxiv.org/pdf/2604.11312)  

**Abstract**: Large Language Models (LLMs) have demonstrated an unprecedented ability to simulate human-like social behaviors, making them useful tools for simulating complex social systems. However, it remains unclear to what extent these simulations can be trusted to accurately capture key social mechanisms, particularly in highly unbalanced contexts involving minority groups. This paper uses a network generation model with controlled homophily and class sizes to examine how LLM agents behave collectively in multi-round debates. Moreover, our findings highlight a particular directional susceptibility that we term \textit{agreement drift}, in which agents are more likely to shift toward specific positions on the opinion scale. Overall, our findings highlight the need to disentangle structural effects from model biases before treating LLM populations as behavioral proxies for human groups. 

---
# The Salami Slicing Threat: Exploiting Cumulative Risks in LLM Systems 

**Authors**: Yihao Zhang, Kai Wang, Jiangrong Wu, Haolin Wu, Yuxuan Zhou, Zeming Wei, Dongxian Wu, Xun Chen, Jun Sun, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.11309)  

**Abstract**: Large Language Models (LLMs) face prominent security risks from jailbreaking, a practice that manipulates models to bypass built-in security constraints and generate unethical or unsafe content. Among various jailbreak techniques, multi-turn jailbreak attacks are more covert and persistent than single-turn counterparts, exposing critical vulnerabilities of LLMs.
However, existing multi-turn jailbreak methods suffer from two fundamental limitations that affect the actual impact in real-world scenarios: (a) As models become more context-aware, any explicit harmful trigger is increasingly likely to be flagged and blocked; (b) Successful final-step triggers often require finely tuned, model-specific contexts, making such attacks highly context-dependent. To fill this gap, we propose \textit{Salami Slicing Risk}, which operates by chaining numerous low-risk inputs that individually evade alignment thresholds but cumulatively accumulate harmful intent to ultimately trigger high-risk behaviors, without heavy reliance on pre-designed contextual structures. Building on this risk, we develop Salami Attack, an automatic framework universally applicable to multiple model types and modalities.
Rigorous experiments demonstrate its state-of-the-art performance across diverse models and modalities, achieving over 90\% Attack Success Rate on GPT-4o and Gemini, as well as robustness against real-world alignment defenses. We also proposed a defense strategy to constrain the Salami Attack by at least 44.8\% while achieving a maximum blocking rate of 64.8\% against other multi-turn jailbreak attacks. Our findings provide critical insights into the pervasive risks of multi-turn jailbreaking and offer actionable mitigation strategies to enhance LLM security. 

---
# Learning to Forget -- Hierarchical Episodic Memory for Lifelong Robot Deployment 

**Authors**: Leonard Bärmann, Joana Plewnia, Alex Waibel, Tamim Asfour  

**Link**: [PDF](https://arxiv.org/pdf/2604.11306)  

**Abstract**: Robots must verbalize their past experiences when users ask "Where did you put my keys?" or "Why did the task fail?" Yet maintaining life-long episodic memory (EM) from continuous multimodal perception quickly exceeds storage limits and makes real-time query impractical, calling for selective forgetting that adapts to users' notions of relevance. We present H$^2$-EMV, a framework enabling humanoids to learn what to remember through user interaction. Our approach incrementally constructs hierarchical EM, selectively forgets using language-model-based relevance estimation conditioned on learned natural-language rules, and updates these rules given user feedback about forgotten details. Evaluations on simulated household tasks and 20.5-hour-long real-world recordings from ARMAR-7 demonstrate that H$^2$-EMV maintains question-answering accuracy while reducing memory size by 45% and query-time compute by 35%. Critically, performance improves over time - accuracy increases 70% in second-round queries by adapting to user-specific priorities - demonstrating that learned forgetting enables scalable, personalized EM for long-term human-robot collaboration. 

---
# 3D-Anchored Lookahead Planning for Persistent Robotic Scene Memory via World-Model-Based MCTS 

**Authors**: Bronislav Sidik, Dror Mizrahi  

**Link**: [PDF](https://arxiv.org/pdf/2604.11302)  

**Abstract**: We present 3D-Anchored Lookahead Planning (3D-ALP), a System 2 reasoning engine for robotic manipulation that combines Monte Carlo Tree Search (MCTS) with a 3D-consistent world model as the rollout oracle. Unlike reactive policies that evaluate actions from the current camera frame only, 3D-ALP maintains a persistent camera-to-world (c2w) anchor that survives occlusion, enabling accurate replanning to object positions that are no longer directly observable. On a 5-step sequential reach task requiring spatial memory (Experiment E3), 3D-ALP achieves 0.650 0.109 success rate on memory-required steps versus 0.006 0.008 for a greedy reactive baseline ({\Delta}=+0.645), while step 5 success reaches 0.822 against 0.000 for greedy. An ablation study (30 episodes, 3 seeds) isolates tree search spatial memory as the primary driver (+0.533, 82% of gain) with additional benefit from deeper lookahead (+0.111, 17%). We also identify and resolve four structural failure modes in applying UCT-MCTS (Upper Confidence Bounds applied to Trees [10]) to continuous robotic manipulation. 

---
# Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning 

**Authors**: Rui Song, Lida Shi, Ruihua Qi, Yingji Li, Hao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11299)  

**Abstract**: In recent years, rapid advances in Multimodal Large Language Models (MLLMs) have increasingly stimulated research on ancient Chinese scripts. As the evolution of written characters constitutes a fundamental pathway for understanding cultural transformation and historical continuity, how MLLMs can be systematically leveraged to support and advance text evolution analysis remains an open and largely underexplored problem. To bridge this gap, we construct a comprehensive benchmark comprising 11 tasks and over 130,000 instances, specifically designed to evaluate the capability of MLLMs in analyzing the evolution of ancient Chinese scripts. We conduct extensive evaluations across multiple widely used MLLMs and observe that, while existing models demonstrate a limited ability in glyph-level comparison, their performance on core tasks-such as character recognition and evolutionary reasoning-remains substantially constrained. Motivated by these findings, we propose a glyph-driven fine-tuning framework (GEVO) that explicitly encourages models to capture evolutionary consistency in glyph transformations and enhances their understanding of text evolution. Experimental results show that even models at the 2B scale achieve consistent and comprehensive performance improvements across all evaluated tasks. To facilitate future research, we publicly release both the benchmark and the trained models\footnote{this https URL}. 

---
# The Past Is Not Past: Memory-Enhanced Dynamic Reward Shaping 

**Authors**: Yang Liu, Enxi Wang, Yufei Gao, Weixin Zhang, Bo Wang, Zhiyuan Zeng, Yikai Zhang, Yining Zheng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11297)  

**Abstract**: Despite the success of reinforcement learning for large language models, a common failure mode is reduced sampling diversity, where the policy repeatedly generates similar erroneous behaviors. Classical entropy regularization encourages randomness under the current policy, but does not explicitly discourage recurrent failure patterns across rollouts. We propose MEDS, a Memory-Enhanced Dynamic reward Shaping framework that incorporates historical behavioral signals into reward design. By storing and leveraging intermediate model representations, we capture features of past rollouts and use density-based clustering to identify frequently recurring error patterns. Rollouts assigned to more prevalent error clusters are penalized more heavily, encouraging broader exploration while reducing repeated mistakes. Across five datasets and three base models, MEDS consistently improves average performance over existing baselines, achieving gains of up to 4.13 pass@1 points and 4.37 pass@128 points. Additional analyses using both LLM-based annotations and quantitative diversity metrics show that MEDS increases behavioral diversity during sampling. 

---
# THEIA: Learning Complete Kleene Three-Valued Logic in a Pure-Neural Modular Architecture 

**Authors**: Augustus Haoyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11284)  

**Abstract**: We present THEIA, a modular neural architecture that learns complete Kleene three-valued logic (K3) end-to-end without any external symbolic solver, and investigate what architectural prior enables compositional generalization under uncertainty. THEIA processes four mathematical domains (arithmetic, order, set membership, propositional logic) through dedicated engines that converge in a final logic module. Trained on a 2M-sample dataset with input space ~3.4x10^13, it achieves 12/12 Kleene K3 rule coverage across 5 seeds in 9.2 +/- 3.5 minutes (5.6x faster than a parameter-comparable Transformer under matched settings). A mod-3 sequential composition experiment generalizes from 5-step training to 500-step evaluation at 99.97% +/- 0.02% -- a result that critically depends on structured inductive bias: replacing the four-engine backbone with a flat MLP collapses length generalization to chance by 50 steps regardless of capacity (both 0.80M and parameter-matched 2.75M variants fail), while a pre-LN TF8LTuned Transformer baseline (3,582,147 params) trained under the identical protocol reaches 99.24% at 500 steps (Appendix D). Mechanistic probing reveals that modularity induces a delayed verdict: upstream engines encode domain-specific variables without committing to the final truth value (probe accuracy <= 74% uncertainty-only ceiling), with the verdict emerging only at the Logic Engine boundary -- causally confirmed by activation patching (100% flip rate on 986 matched pairs, replicated across n=5 seeds; 100.0% aggregate). The Transformer baseline reaches equivalent correctness through a qualitatively different representational trajectory (contraction then expansion), suggesting that modular and monolithic architectures implement distinct compositional strategies. 

---
# AbLWR:A Context-Aware Listwise Ranking Framework for Antibody-Antigen Binding Affinity Prediction via Positive-Unlabeled Learning 

**Authors**: Fan Xu, Zhi-an Huang, Haohuai He, Yidong Song, Wei Liu, Dongxu Zhang, Yao Hu, Kay Chen Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.11272)  

**Abstract**: Accurate prediction of antibody-antigen binding affinity is fundamental to therapeutic design, yet remains constrained by severe label sparsity and the complexity of antigenic variations. In this paper, we propose AbLWR (Antibody-antigen binding affinity List-Wise Ranking), a novel framework that reformulates the conventional affinity regression task as a listwise ranking problem. To mitigate label sparsity, AbLWR incorporates a PU (Positive-Unlabeled) learning mechanism leveraging a dual-level contrastive objective and meta-optimized label refinement to learn robust representations. Furthermore, we address antigenic variation by employing a homologous antigen sampling strategy where Multi-Head Self-Attention (MHSA) explicitly models inter-sample relationships within training lists to capture subtle affinity nuances. Extensive experiments demonstrate that AbLWR significantly outperforms state-of-the-art baselines, improving the Precision@1 (P@1) by over 10$\%$ in randomized cross-validation experiments. Notably, case studies on Influenza and IL-33 validate its practical utility, demonstrating robust ranking consistency in distinguishing subtle viral mutations and efficiently prioritizing top-tier candidates for wet-lab screening. 

---
# Evolving Many Worlds: Towards Open-Ended Discovery in Petri Dish NCA via Population-Based Training 

**Authors**: Uljad Berdica, Jakob Foerster, Frank Hutter, Arber Zela  

**Link**: [PDF](https://arxiv.org/pdf/2604.11248)  

**Abstract**: The generation of sustained, open-ended complexity from local interactions remains a fundamental challenge in artificial life. Differentiable multi-agent systems, such as Petri Dish Neural Cellular Automata (PD-NCA), exhibit rich self-organization driven purely by spatial competition; however, they are highly sensitive to hyperparameters and frequently collapse into uninteresting patterns and dynamics, such as frozen equilibria or structureless noise. In this paper, we introduce PBT-NCA, a meta-evolutionary algorithm that evolves a population of PD-NCAs subject to a composite objective that rewards both historical behavioral novelty and contemporary visual diversity. Driven by this continuous evolutionary pressure, PBT-NCA spontaneously generates a plethora of emergent lifelike phenomena over extended horizons-a hallmark of true open-endedness. Strikingly, the substrate autonomously discovers diverse morphological survival and self-organization strategies. We observe highly regular, coordinated periodic waves; spore-like scattering where homogeneous groups eject cell-like clusters to colonize distant territories; and fluid, shape-shifting macro-structures that migrate across the substrate, maintaining stable outer boundaries that enclose highly active interiors. By actively penalizing monocultures and dead states, PBT-NCA sustains a state of effective complexity that is neither globally ordered nor globally random, operating persistently at the "edge of chaos". 

---
# RECIPER: A Dual-View Retrieval Pipeline for Procedure-Oriented Materials Question Answering 

**Authors**: Zhuoyu Wu, Wenhui Ou, Pei-Sze Tan, Wenqi Fang, Sailaja Rajanala, Raphaël C.-W. Phan  

**Link**: [PDF](https://arxiv.org/pdf/2604.11229)  

**Abstract**: Retrieving procedure-oriented evidence from materials science papers is difficult because key synthesis details are often scattered across long, context-heavy documents and are not well captured by paragraph-only dense retrieval. We present RECIPER, a dual-view retrieval pipeline that indexes both paragraph-level context and compact large language model-extracted procedural summaries, then combines the two candidate streams with lightweight lexical reranking. Across four dense retrieval backbones, RECIPER consistently improves early-rank retrieval over paragraph-only dense retrieval, achieving average gains of +3.73 in Recall@1, +2.85 in nDCG@10, and +3.13 in MRR. With BGE-large-en-v1.5, it reaches 86.82%, 97.07%, and 97.85% on Recall@1, Recall@5, and Recall@10, respectively. We further observe improved downstream question answering under automatic metrics, suggesting that procedural summaries can serve as a useful complementary retrieval signal for procedure-oriented materials question answering. Code and data are available at this https URL. 

---
# Regional Explanations: Bridging Local and Global Variable Importance 

**Authors**: Salim I. Amoukou, Nicolas J-B. Brunel  

**Link**: [PDF](https://arxiv.org/pdf/2604.11223)  

**Abstract**: We analyze two widely used local attribution methods, Local Shapley Values and LIME, which aim to quantify the contribution of a feature value $x_i$ to a specific prediction $f(x_1, \dots, x_p)$. Despite their widespread use, we identify fundamental limitations in their ability to reliably detect locally important features, even under ideal conditions with exact computations and independent features. We argue that a sound local attribution method should not assign importance to features that neither influence the model output (e.g., features with zero coefficients in a linear model) nor exhibit statistical dependence with functionality-relevant features. We demonstrate that both Local SV and LIME violate this fundamental principle. To address this, we propose R-LOCO (Regional Leave Out COvariates), which bridges the gap between local and global explanations and provides more accurate attributions. R-LOCO segments the input space into regions with similar feature importance characteristics. It then applies global attribution methods within these regions, deriving an instance's feature contributions from its regional membership. This approach delivers more faithful local attributions while avoiding local explanation instability and preserving instance-specific detail often lost in global methods. 

---
# Exploring Knowledge Conflicts for Faithful LLM Reasoning: Benchmark and Method 

**Authors**: Tianzhe Zhao, Jiaoyan Chen, Shuxiu Zhang, Haiping Zhu, Qika Lin, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11209)  

**Abstract**: Large language models (LLMs) have achieved remarkable success across a wide range of applications especially when augmented by external knowledge through retrieval-augmented generation (RAG). Despite their widespread adoption, recent studies have shown that LLMs often struggle to perform faithful reasoning when conflicting knowledge is retrieved. However, existing work primarily focuses on conflicts between external knowledge and the parametric knowledge of LLMs, leaving conflicts across external knowledge largely unexplored. Meanwhile, modern RAG systems increasingly emphasize the integration of unstructured text and (semi-)structured data like knowledge graphs (KGs) to improve knowledge completeness and reasoning faithfulness. To address this gap, we introduce ConflictQA, a novel benchmark that systematically instantiates conflicts between textual evidence and KG evidence. Extensive evaluations across representative LLMs reveal that, facing such cross-source conflicts, LLMs often fail to identify reliable evidence for correct reasoning. Instead, LLMs become more sensitive to prompting choices and tend to rely exclusively on either KG or textual evidence, resulting in incorrect responses. Based on these findings, we further propose XoT, a two-stage explanation-based thinking framework tailored for reasoning over heterogeneous conflicting evidence, and verify its effectiveness with extensive experiments. 

---
# Designing Adaptive Digital Nudging Systems with LLM-Driven Reasoning 

**Authors**: Tiziano Santilli, Mina Alipour, Mahyar Tourchi Moghaddam  

**Link**: [PDF](https://arxiv.org/pdf/2604.11206)  

**Abstract**: Digital nudging systems lack architectural guidance for translating behavioral science into software design. While research identifies nudge strategies and quality attributes, existing architectures fail to integrate multi-dimensional user modeling with ethical compliance as architectural concerns. We present an architecture that uses behavioral theory through explicit architectural decisions, treating ethics and fairness as structural guardrails rather than implementation details. A literature review synthesized 68 nudging strategies, 11 quality attributes, and 3 user profiling dimensions into architectural requirements. The architecture implements sequential processing layers with cross-cutting evaluation modules enforcing regulatory compliance. Validation with 13 software architects confirmed requirements satisfaction and domain transferability. An LLM-powered proof-of-concept in residential energy sustainability demonstrated feasibility through evaluation with 15 users, achieving high perceived intervention quality and measurable positive emotional impact. This work bridges behavioral science and software architecture by providing reusable patterns for adaptive systems that balance effectiveness with ethical constraints. 

---
# CocoaBench: Evaluating Unified Digital Agents in the Wild 

**Authors**: CocoaBench Team, Shibo Hao, Zhining Zhang, Zhiqi Liang, Tianyang Liu, Yuheng Zha, Qiyue Gao, Jixuan Chen, Zilong Wang, Zhoujun Cheng, Haoxiang Zhang, Junli Wang, Hexi Jin, Boyuan Zheng, Kun Zhou, Yu Wang, Feng Yao, Licheng Liu, Yijiang Li, Zhifei Li, Zhengtao Han, Pracha Promthaw, Tommaso Cerruti, Xiaohan Fu, Ziqiao Ma, Jingbo Shang, Lianhui Qin, Julian McAuley, Eric P. Xing, Zhengzhong Liu, Rupesh Kumar Srivastava, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11201)  

**Abstract**: LLM agents now perform strongly in software engineering, deep research, GUI automation, and various other applications, while recent agent scaffolds and models are increasingly integrating these capabilities into unified systems. Yet, most evaluations still test these capabilities in isolation, which leaves a gap for more diverse use cases that require agents to combine different capabilities. We introduce CocoaBench, a benchmark for unified digital agents built from human-designed, long-horizon tasks that require flexible composition of vision, search, and coding. Tasks are specified only by an instruction and an automatic evaluation function over the final output, enabling reliable and scalable evaluation across diverse agent infrastructures. We also present CocoaAgent, a lightweight shared scaffold for controlled comparison across model backbones. Experiments show that current agents remain far from reliable on CocoaBench, with the best evaluated system achieving only 45.1% success rate. Our analysis further points to substantial room for improvement in reasoning and planning, tool use and execution, and visual grounding. 

---
# ShapShift: Explaining Model Prediction Shifts with Subgroup Conditional Shapley Values 

**Authors**: Tom Bewley, Salim I. Amoukou, Emanuele Albini, Saumitra Mishra, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2604.11200)  

**Abstract**: Changes in input distribution can induce shifts in the average predictions of machine learning models. Such prediction shifts may impact downstream business outcomes (e.g. a bank's loan approval rate), so understanding their causes can be crucial. We propose \ours{}: a Shapley value method for attributing prediction shifts to changes in the conditional probabilities of interpretable subgroups of data, where these subgroups are defined by the structure of decision trees. We initially apply this method to single decision trees, providing exact explanations based on conditional probability changes at split nodes. Next, we extend it to tree ensembles by selecting the most explanatory tree and accounting for residual effects. Finally, we propose a model-agnostic variant using surrogate trees grown with a novel objective function, allowing application to models like neural networks. While exact computation can be intensive, approximation techniques enable practical application. We show that \ours{} provides simple, faithful, and near-complete explanations of prediction shifts across model classes, aiding model monitoring in dynamic environments. 

---
# Towards Adaptive Open-Set Object Detection via Category-Level Collaboration Knowledge Mining 

**Authors**: Yuqi Ji, Junjie Ke, Lihuo He, Lizhi Wang, Xinbo Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.11195)  

**Abstract**: Existing object detectors often struggle to generalize across domains while adapting to emerging novel categories. Adaptive open-set object detection (AOOD) addresses this challenge by training on base categories in the source domain and adapting to both base and novel categories in the target domain without target annotations. However, current AOOD methods remain limited by weak cross-domain representations, ambiguity among novel categories, and source-domain feature bias. To address these issues, we propose a category-level collaboration knowledge mining strategy that exploits both inter-class and intra-class relationships across domains. Specifically, we construct a clustering-based memory bank to encode class prototypes, auxiliary features, and intra-class disparity information, and iteratively update it via unsupervised clustering to enhance category-level knowledge representation. We further design a base-to-novel selection metric to discover source-domain features related to novel categories and use them to initialize novel-category classifiers. In addition, an adaptive feature assignment strategy transfers the learned category-level knowledge to the target domain and asynchronously updates the memory bank to alleviate source-domain bias. Extensive experiments on multiple benchmarks show that our method consistently surpasses state-of-the-art AOOD methods by 1.1-5.5 mAP. 

---
# MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis 

**Authors**: Zixiong Yu, Jun Rao, Guhan Chen, Songtao Tian, Bohan Li, Jiansheng Wei, Min Zhang, Xiaojun Meng  

**Link**: [PDF](https://arxiv.org/pdf/2604.11188)  

**Abstract**: Synthesizing high-quality mathematical reasoning data without human priors remains a significant challenge. Current approaches typically rely on seed data mutation or simple prompt engineering, often suffering from mode collapse and limited logical complexity. This paper proposes a hierarchical synthesis framework that formulates data synthesis as an unsupervised optimization problem over a constraint graph followed by semantic instantiation, rather than treating it as a direct text generation task. We introduce a Legislator-Executor paradigm: The Legislator adversarially evolves structured generation blueprints encoding the constraints of the problem, while the Executor instantiates these specifications into diverse natural language scenarios. This decoupling of skeleton design from linguistic realization enables a prioritized focus on constructing complex and diverse logical structures, thereby guiding high-quality data synthesis. Experiments conducted on a total of 10 models across the Qwen, Llama, Mistral, and Gemma series demonstrate that our method achieves notable results: models fine-tuned on 1K synthesized samples outperform widely-used datasets of comparable scale (LIMO, s1K) across eight mathematical benchmarks, exhibiting superior out-of-distribution generalization. 

---
# Taking a Pulse on How Generative AI is Reshaping the Software Engineering Research Landscape 

**Authors**: Bianca Trinkenreich, Fabio Calefato, Kelly Blincoe, Viggo Tellefsen Wivestad, Antonio Pedro Santos Alves, Júlia Condé Araújo, Marina Condé Araújo, Paolo Tell, Marcos Kalinowski, Thomas Zimmermann, Margaret-Anne Storey  

**Link**: [PDF](https://arxiv.org/pdf/2604.11184)  

**Abstract**: Context: Software engineering (SE) researchers increasingly study Generative AI (GenAI) while also incorporating it into their own research practices. Despite rapid adoption, there is limited empirical evidence on how GenAI is used in SE research and its implications for research practices and governance. Aims: We conduct a large-scale survey of 457 SE researchers publishing in top venues between 2023 and 2025. Method: Using quantitative and qualitative analyses, we examine who uses GenAI and why, where it is used across research activities, and how researchers perceive its benefits, opportunities, challenges, risks, and governance. Results: GenAI use is widespread, with many researchers reporting pressure to adopt and align their work with it. Usage is concentrated in writing and early-stage activities, while methodological and analytical tasks remain largely human-driven. Although productivity gains are widely perceived, concerns about trust, correctness, and regulatory uncertainty persist. Researchers highlight risks such as inaccuracies and bias, emphasize mitigation through human oversight and verification, and call for clearer governance, including guidance on responsible use and peer review. Conclusion: We provide a fine-grained, SE-specific characterization of GenAI use across research activities, along with taxonomies of GenAI use cases for research and peer review, opportunities, risks, mitigation strategies, and governance needs. These findings establish an empirical baseline for the responsible integration of GenAI into academic practice. 

---
# EmbodiedGovBench: A Benchmark for Governance, Recovery, and Upgrade Safety in Embodied Agent Systems 

**Authors**: Xue Qin, Simin Luan, John See, Cong Yang, Zhijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11174)  

**Abstract**: Recent progress in embodied AI has produced a growing ecosystem of robot policies, foundation models, and modular runtimes. However, current evaluation remains dominated by task success metrics such as completion rate or manipulation accuracy. These metrics leave a critical gap: they do not measure whether embodied systems are governable -- whether they respect capability boundaries, enforce policies, recover safely, maintain audit trails, and respond to human oversight. We present EmbodiedGovBench, a benchmark for governance-oriented evaluation of embodied agent systems. Rather than asking only whether a robot can complete a task, EmbodiedGovBench evaluates whether the system remains controllable, policy-bounded, recoverable, auditable, and evolution-safe under realistic perturbations. The benchmark covers seven governance dimensions: unauthorized capability invocation, runtime drift robustness, recovery success, policy portability, version upgrade safety, human override responsiveness, and audit completeness. We define a benchmark structure spanning single-robot and fleet settings, with scenario templates, perturbation operators, governance metrics, and baseline evaluation protocols. We describe how the benchmark can be instantiated over embodied capability runtimes with modular interfaces and contract-aware upgrade workflows. Our analysis suggests that embodied governance should become a first-class evaluation target. EmbodiedGovBench provides the initial measurement framework for that shift. 

---
# Cost-optimal Sequential Testing via Doubly Robust Q-learning 

**Authors**: Doudou Zhou, Yiran Zhang, Dian Jin, Yingye Zheng, Lu Tian, Tianxi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.11165)  

**Abstract**: Clinical decision-making often involves selecting tests that are costly, invasive, or time-consuming, motivating individualized, sequential strategies for what to measure and when to stop ascertaining. We study the problem of learning cost-optimal sequential decision policies from retrospective data, where test availability depends on prior results, inducing informative missingness. Under a sequential missing-at-random mechanism, we develop a doubly robust Q-learning framework for estimating optimal policies. The method introduces path-specific inverse probability weights that account for heterogeneous test trajectories and satisfy a normalization property conditional on the observed history. By combining these weights with auxiliary contrast models, we construct orthogonal pseudo-outcomes that enable unbiased policy learning when either the acquisition model or the contrast model is correctly specified. We establish oracle inequalities for the stage-wise contrast estimators, along with convergence rates, regret bounds, and misclassification rates for the learned policy. Simulations demonstrate improved cost-adjusted performance over weighted and complete-case baselines, and an application to a prostate cancer cohort study illustrates how the method reduces testing cost without compromising predictive accuracy. 

---
# BoxTuning: Directly Injecting the Object Box for Multimodal Model Fine-Tuning 

**Authors**: Zekun Qian, Ruize Han, Wei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.11136)  

**Abstract**: Object-level spatial-temporal understanding is essential for video question answering, yet existing multimodal large language models (MLLMs) encode frames holistically and lack explicit mechanisms for fine-grained object grounding. Recent work addresses this by serializing bounding box coordinates as text tokens, but this text-coordinate paradigm suffers from a fundamental modality mismatch: object information is inherently visual, yet encoding it as text incurs a high token cost that forces aggressive temporal downsampling. We propose BoxTuning, which resolves this mismatch by injecting object spatial-temporal information directly into the visual modality. Colored bounding boxes and trajectory trails are rendered onto video frames as visual prompts, with only a concise color-to-object legend retained as text. This reduces the token cost significantly, achieving 87-93% text token reduction in practice. It also preserves full temporal resolution, where the trajectory trails further encode inter-frame motion direction and speed within each keyframe, recovering fine-grained dynamics that text-coordinate methods are forced to discard. Experimental results on five video QA benchmarks (CLEVRER, Perception Test, STAR, NExT-QA, IntentQA) show that BoxTuning surpasses text-coordinate baselines on spatially oriented tasks and nearly eliminates the accuracy degradation observed on reasoning-centric tasks, establishing visual prompting as a more natural and efficient paradigm for conveying object information to video MLLMs. 

---
# Semantic-Geometric Dual Compression: Training-Free Visual Token Reduction for Ultra-High-Resolution Remote Sensing Understanding 

**Authors**: Yueying Li, Fengxiang Wang, Yan Li, Mingshuo Chen, Mengying Zhao, Long Lan  

**Link**: [PDF](https://arxiv.org/pdf/2604.11122)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated immense potential in Earth observation. However, the massive visual tokens generated when processing Ultra-High-Resolution (UHR) imagery introduce prohibitive computational overhead, severely bottlenecking their inference efficiency. Existing visual token compression methods predominantly adopt static and uniform compression strategies, neglecting the inherent "Semantic-Geometric Duality" in remote sensing interpretation tasks. Specifically, object semantic tasks focus on the abstract semantics of objects and benefit from aggressive background pruning, whereas scene geometric tasks critically rely on the integrity of spatial topology. To address this challenge, we propose DualComp, a task-adaptive dual-stream token compression framework. Dynamically guided by a lightweight pre-trained router, DualComp decouples feature processing into two dedicated pathways. In the object semantic stream, the Spatially-Contiguous Semantic Aggregator (SCSA) utilizes size-adaptive clustering to aggregates redundant background while protecting small object. In the scene geometric stream, the Instruction-Guided Structure Recoverer (IGSR) introduces a greedy path-tracing topology completion mechanism to reconstruct spatial skeletons. Experiments on the UHR remote sensing benchmark XLRS-Bench demonstrate that DualComp accomplishes high-fidelity remote sensing interpretation at an exceptionally low computational cost, achieving simultaneous improvements in both efficiency and accuracy. 

---
# Use of AI Tools: Guidelines to Maintain Academic Integrity in Computing Colleges 

**Authors**: Hatem M. El-boghdadi, Toqeer Ali Syed, Ali Akarma, Qamar Wali  

**Link**: [PDF](https://arxiv.org/pdf/2604.11111)  

**Abstract**: The rapid adoption of AI tools such as ChatGPT has significantly transformed academic practices, offering considerable benefits for both students and faculty in computing disciplines. These tools have been shown to enhance learning efficiency, academic self-efficacy, and confidence. However, their increasing use also raises pressing concerns regarding the preservation of academic integrity -- an essential pillar of the educational process. This paper explores the implications of widespread AI tool usage within computing colleges, with a particular focus on how to align their use with the principles of academic honesty. We begin by classifying common assessment techniques employed in computing education and examine how each may be impacted by AI-assisted tools. Building on this foundation, we propose a set of general guidelines applicable across various assessment formats to help instructors responsibly integrate AI tools into their pedagogy. Furthermore, we provide targeted, assessment-specific recommendations designed to uphold educational objectives while mitigating risks of academic misconduct. These guidelines serve as a practical framework for instructors aiming to balance the pedagogical advantages of AI tools with the imperative of maintaining academic integrity in computing education. Finally, we introduce a formal model that provides a structured mathematical framework for evaluating student assessments in the presence of AI-assisted tools. 

---
# Record-Remix-Replay: Hierarchical GPU Kernel Optimization using Evolutionary Search 

**Authors**: Daniel Nichols, Konstantinos Parasyris, Caetano Melone, Tal Ben-Nun, Giorgis Georgakoudis, Harshitha Menon  

**Link**: [PDF](https://arxiv.org/pdf/2604.11109)  

**Abstract**: As high-performance computing and AI workloads become increasingly dependent on GPUs, maintaining high performance across rapidly evolving hardware generations has become a major challenge. Developers often spend months tuning scientific applications to fully exploit new architectures, navigating a complex optimization space that spans algorithm design, source implementation, compiler flags and pass sequences, and kernel launch parameters. Existing approaches can effectively search parts of this space in isolation, such as launch configurations or compiler settings, but optimizing across the full space still requires substantial human expertise and iterative manual effort.
In this paper, we present Record-Remix-Replay (R^3), a hierarchical optimization framework that combines LLM-driven evolutionary search, Bayesian optimization, and record-replay compilation techniques to efficiently explore GPU kernel optimizations from source-level implementation choices down to compiler pass ordering and runtime configuration. By making candidate evaluation fast and scalable, our approach enables practical end-to-end search over optimization dimensions that are typically treated separately. We show that Record-Remix-Replay can optimize full scientific applications better than traditional approaches over kernel parameters and compiler flags, while also being nearly an order of magnitude faster than modern evolutionary search approaches. 

---
# ActorMind: Emulating Human Actor Reasoning for Speech Role-Playing 

**Authors**: Xi Chen, Wei Xue, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.11103)  

**Abstract**: Role-playing has garnered rising attention as it provides a strong foundation for human-machine interaction and facilitates sociological research. However, current work is confined to textual modalities, neglecting speech, which plays a predominant role in daily life, thus limiting genuine role-playing. To bridge this gap, we conceptualize and benchmark speech role-playing through ActorMindBench, and we present a corresponding reasoning framework, called ActorMind. Specifically, (1) Speech Role-Playing enables models to deliver spontaneous responses with personalized verbal traits based on their role, the scene, and spoken dialogue. (2) ActorMindBench is a hierarchical benchmark comprises Utterance-Level content with 7,653 utterances, Scene-Level content with 313 scenes, and Role-Level content with 6 roles. (3) ActorMind is an off-the-shelf, multi-agent, chain-of-though style reasoning framework that emulates how human actors perform in theaters. Concretely, ActorMind first reads its assigned role description via Eye Agent, then comprehends emotional cues within contextual spoken dialogues through Ear Agent. Subsequently, Brain Agent generates a descriptive emotional state, and finally, Mouth Agent delivers the scripts infused with corresponding emotion state. Experimental results demonstrate the effectiveness of ActorMind in enhancing speech role-playing. 

---
# Efficient Training for Cross-lingual Speech Language Models 

**Authors**: Yan Zhou, Qingkai Fang, Yun Hong, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.11096)  

**Abstract**: Currently, large language models (LLMs) predominantly focus on the text modality. To enable more natural human-AI interaction, speech LLMs are emerging, but building effective end-to-end speech LLMs remains challenging due to limited data and the difficulty in expanding to more languages. In this paper, we introduce Cross-lingual Speech Language Model (CSLM), an efficient training method for cross-lingual speech LLMs based on discrete speech tokens. We propose a novel alignment strategy that achieves cross-modal and cross-lingual alignment through continual pre-training. By conducting instruction fine-tuning following a speech-text interleaved chain-of-modality generation process, we enhance modal alignment at a finer granularity, thereby improving generation quality and reducing latency. CSLM aligns different modalities and languages simultaneously without the need for massive speech data, thus exhibiting good language scalability. Evaluations on cross-modal tasks, mono-lingual conversational tasks, and cross-lingual conversational tasks demonstrate CSLM's strong cross-modal alignment capabilities and general task abilities. (Code is available at: this https URL) 

---
# Bottleneck Tokens for Unified Multimodal Retrieval 

**Authors**: Siyu Sun, Jing Ren, Zhaohe Liao, Dongxiao Mao, Xiangyuan Ren, Yiyi Zhang, Haohua Zhao, Weixiong Lin, Jiang Shaohua, Liqing Zhang, Yuchao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.11095)  

**Abstract**: Adapting decoder-only multimodal large language models (MLLMs) for unified multimodal retrieval faces two structural gaps. First, existing methods rely on implicit pooling, which overloads the hidden state of a standard vocabulary token (e.g., <EOS>) as the sequence-level representation, a mechanism never designed for information aggregation. Second, contrastive fine-tuning specifies what the embedding should match but provides no token-level guidance on how information should be compressed into it. We address both gaps with two complementary components. Architecturally, we introduce Bottleneck Tokens (BToks), a small set of learnable tokens that serve as a fixed-capacity explicit pooling mechanism. For training, we propose Generative Information Condensation: a next-token prediction objective coupled with a Condensation Mask that severs the direct attention path from target tokens to query tokens. All predictive signals are thereby forced through the BToks, converting the generative loss into dense, token-level supervision for semantic compression. At inference time, only the input and BToks are processed in a single forward pass with negligible overhead over conventional last-token pooling. On MMEB-V2 (78 datasets, 3 modalities, 9 meta-tasks), our approach achieves state-of-the-art among 2B-scale methods under comparable data conditions, attaining an Overall score of 59.0 (+3.6 over VLM2Vec-V2) with substantial gains on semantically demanding tasks (e.g., +12.6 on Video-QA). 

---
# E2E-REME: Towards End-to-End Microservices Auto-Remediation via Experience-Simulation Reinforcement Fine-Tuning 

**Authors**: Lingzhe Zhang, Yunpeng Zhai, Tong Jia, Minghua He, Chiming Duan, Zhaoyang Liu, Bolin Ding, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11094)  

**Abstract**: Contemporary microservice systems continue to grow in scale and complexity, leading to increasingly frequent and costly failures. While recent LLM-based auto-remediation approaches have emerged, they primarily translate textual instructions into executable Ansible playbooks and rely on expert-crafted prompts, lacking runtime knowledge guidance and depending on large-scale general-purpose LLMs, which limits their accuracy and efficiency. We introduce \textit{End-to-End Microservice Remediation} (E2E-MR), a new task that requires directly generating executable playbooks from diagnosis reports to autonomously restore faulty systems. To enable rigorous evaluation, we build \textit{MicroRemed}, a benchmark that automates microservice deployment, failure injection, playbook execution, and post-repair verification. We further propose \textit{E2E-REME}, an end-to-end auto-remediation model trained via experience-simulation reinforcement fine-tuning. Experiments on public and industrial microservice platforms, compared with nine representative LLMs, show that E2E-REME achieves superior accuracy and efficiency. 

---
# FlowCoMotion: Text-to-Motion Generation via Token-Latent Flow Modeling 

**Authors**: Dawei Guan, Di Yang, Chengjie Jin, Jiangtao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11083)  

**Abstract**: Text-to-motion generation is driven by learning motion representations for semantic alignment with language. Existing methods rely on either continuous or discrete motion representations. However, continuous representations entangle semantics with dynamics, while discrete representations lose fine-grained motion details. In this context, we propose FlowCoMotion, a novel motion generation framework that unifies both treatments from a modeling perspective. Specifically, FlowCoMotion employs token-latent coupling to capture both semantic content and high-fidelity motion details. In the latent branch, we apply multi-view distillation to regularize the continuous latent space, while in the token branch we use discrete temporal resolution quantization to extract high-level semantic cues. The motion latent is then obtained by combining the representations from the two branches through a token-latent coupling network. Subsequently, a velocity field is predicted based on the textual conditions. An ODE solver integrates this velocity field from a simple prior, thereby guiding the sample to the potential state of the target motion. Extensive experiments show that FlowCoMotion achieves competitive performance on text-to-motion benchmarks, including HumanML3D and SnapMoGen. 

---
# ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation 

**Authors**: Suyoung Kim, Sunghyun Wee, Hyeonjin Kim, Kyomin Hwang, Hyunho Lee, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2604.11080)  

**Abstract**: Rotation-based Post-Training Quantization (PTQ) has emerged as a promising solution for mitigating activation outliers in the quantization of Large Language Models (LLMs). Global rotation methods achieve inference efficiency by fusing activation rotations into attention and FFN blocks, but suffer from limited expressivity as they are constrained to use a single learnable rotation matrix across all layers. To tackle this, layer-wise transformation methods emerged, achieving superior accuracy through localized adaptation. However, layer-wise methods cannot fuse activation rotation matrices into weights, requiring online computations and causing significant overhead. In this paper, we propose ReSpinQuant, a quantization framework that resolves such overhead by leveraging offline activation rotation fusion and matching basis using efficient residual subspace rotation. This design reconciles the high expressivity of layer-wise adaptation with only negligible inference overhead. Extensive experiments on W4A4 and W3A3 quantization demonstrate that ReSpinQuant achieves state-of-the-art performance, outperforming global rotation methods and matching the accuracy of computationally expensive layer-wise methods with minimal overhead. 

---
# Lightweight Low-Light Image Enhancement via Distribution-Normalizing Preprocessing and Depthwise U-Net 

**Authors**: Shimon Murai, Teppei Kurita, Ryuta Satoh, Yusuke Moriuchi  

**Link**: [PDF](https://arxiv.org/pdf/2604.11071)  

**Abstract**: We present a lightweight two-stage framework for low-light image enhancement (LLIE) that achieves competitive perceptual quality with significantly fewer parameters than existing methods. Our approach combines frozen algorithm-based preprocessing with a compact U-Net built entirely from depthwise-separable convolutions. The preprocessing normalizes the input distribution by providing complementary brightness-corrected views, enabling the trainable network to focus on residual color correction. Our method achieved 4th place in the CVPR 2026 NTIRE Efficient Low-Light Image Enhancement Challenge. We further provide extended benchmarks and ablations to demonstrate the general effectiveness of our methods. 

---
# Pando: Do Interpretability Methods Work When Models Won't Explain Themselves? 

**Authors**: Ziqian Zhong, Aashiq Muhamed, Mona T. Diab, Virginia Smith, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2604.11061)  

**Abstract**: Mechanistic interpretability is often motivated for alignment auditing, where a model's verbal explanations can be absent, incomplete, or misleading. Yet many evaluations do not control whether black-box prompting alone can recover the target behavior, so apparent gains from white-box tools may reflect elicitation rather than internal signal; we call this the elicitation confounder. We introduce Pando, a model-organism benchmark that breaks this confound via an explanation axis: models are trained to produce either faithful explanations of the true rule, no explanation, or confident but unfaithful explanations of a disjoint distractor rule.
Across 720 finetuned models implementing hidden decision-tree rules, agents predict held-out model decisions from $10$ labeled query-response pairs, optionally augmented with one interpretability tool output. When explanations are faithful, black-box elicitation matches or exceeds all white-box methods; when explanations are absent or misleading, gradient-based attribution improves accuracy by 3-5 percentage points, and relevance patching, RelP, gives the largest gains, while logit lens, sparse autoencoders, and circuit tracing provide no reliable benefit. Variance decomposition suggests gradients track decision computation, which fields causally drive the output, whereas other readouts are dominated by task representation, biases toward field identity and value.
We release all models, code, and evaluation infrastructure. 

---
# Rethinking Token-Level Credit Assignment in RLVR: A Polarity-Entropy Analysis 

**Authors**: Yuhang He, Haodong Wu, Siyi Liu, Hongyu Ge, Hange Zhou, Keyi Wu, Zhuo Zheng, Qihong Lin, Zixin Zhong, Yongqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11056)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has substantially improved the reasoning ability of Large Language Models (LLMs). However, its sparse outcome-based rewards pose a fundamental credit assignment problem. We analyze this problem through the joint lens of reward polarity and token entropy. Our diagnostic tool, the Four Quadrant Decomposition, isolates token updates by polarity and entropy, and controlled ablations show that reasoning improvements concentrate in the high-entropy quadrants. To justify this observation theoretically, we adapt Conditional Mutual Information to the autoregressive RLVR setting and prove that the credit a token can carry is upper-bounded by its entropy. This view yields testable predictions that reasoning gains arise primarily from high-entropy tokens, with unique roles for positive and negative updates. A gradient analysis of GRPO further reveals how uniform reward broadcast dilutes signal at high-entropy positions while over-crediting deterministic tokens. Grounded in these insights, we propose Entropy-Aware Policy Optimization (EAPO) that modulates token-level learning signals accordingly. Extensive experiments demonstrate that EAPO outperforms strong baselines across two model families. 

---
# Shared Emotion Geometry Across Small Language Models: A Cross-Architecture Study of Representation, Behavior, and Methodological Confounds 

**Authors**: Jihoon Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2604.11050)  

**Abstract**: We extract 21-emotion vector sets from twelve small language models (six architectures x base/instruct, 1B-8B parameters) under a unified comprehension-mode pipeline at fp16 precision, and compare the resulting geometries via representational similarity analysis on raw cosine RDMs. The five mature architectures (Qwen 2.5 1.5B, SmolLM2 1.7B, Llama 3.2 3B, Mistral 7B v0.3, Llama 3.1 8B) share nearly identical 21-emotion geometry, with pairwise RDM Spearman correlations of 0.74-0.92. This universality persists across diametrically opposed behavioral profiles: Qwen 2.5 and Llama 3.2 occupy opposite poles of MTI Compliance facets yet produce nearly identical emotion RDMs (rho = 0.81), so behavioral facet differences arise above the shared emotion representation. Gemma-3 1B base, the one immature case in our dataset, exhibits extreme residual-stream anisotropy (0.997) and is restructured by RLHF across all geometric descriptors, whereas the five already-mature families show within-family base x instruct RDM correlations of rho >= 0.92 (Mistral 7B v0.3 at rho = 0.985), suggesting RLHF restructures only representations that are not yet organized. Methodologically, we show that what prior work has read as a single comprehension-vs-generation method effect in fact decomposes into four distinct layers -- a coarse method-dependent dissociation, robust sub-parameter sensitivity within generation, a true precision (fp16 vs INT8) effect, and a conflated cross-experiment bias that distorts in opposite directions for different models -- so that a single rho between two prior emotion-vector studies is not a safe basis for interpretation without the layered decomposition. 

---
# A Systematic Analysis of the Impact of Persona Steering on LLM Capabilities 

**Authors**: Jiaqi Chen, Ming Wang, Tingna Xie, Shi Feng, Yongkang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11048)  

**Abstract**: Imbuing Large Language Models (LLMs) with specific personas is prevalent for tailoring interaction styles, yet the impact on underlying cognitive capabilities remains unexplored. We employ the Neuron-based Personality Trait Induction (NPTI) framework to induce Big Five personality traits in LLMs and evaluate performance across six cognitive benchmarks. Our findings reveal that persona induction produces stable, reproducible shifts in cognitive task performance beyond surface-level stylistic changes. These effects exhibit strong task dependence: certain personalities yield consistent gains on instruction-following, while others impair complex reasoning. Effect magnitude varies systematically by trait dimension, with Openness and Extraversion exerting the most robust influence. Furthermore, LLM effects show 73.68% directional consistency with human personality-cognition relationships. Capitalizing on these regularities, we propose Dynamic Persona Routing (DPR), a lightweight query-adaptive strategy that outperforms the best static persona without additional training. 

---
# RTMC: Step-Level Credit Assignment via Rollout Trees 

**Authors**: Tao Wang, Suhang Zheng, Xiaoxiao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.11037)  

**Abstract**: Multi-step agentic reinforcement learning benefits from fine-grained credit assignment, yet existing approaches offer limited options: critic-free methods like GRPO assign a uniform advantage to every action in a trajectory, while learned value networks introduce notable overhead and can be fragile under sparse rewards. We observe that group rollouts targeting the same problem often traverse overlapping intermediate states, implicitly forming a tree whose branches diverge at successive decision points. Building on this insight, we introduce Rollout-Tree Monte Carlo (RTMC) advantage estimation, which aggregates return statistics across rollouts sharing a common state to produce per-step Q-values and advantages--without any learned critic. A state-action signature system compresses raw interaction histories into compact, comparable representations, making cross-rollout state matching tractable. On SWE-bench Verified, RTMC improves pass@1 by 3.2 percentage points over GRPO. 

---
# Uncertainty-Aware Web-Conditioned Scientific Fact-Checking 

**Authors**: Ashwin Vinod, Katrin Erk  

**Link**: [PDF](https://arxiv.org/pdf/2604.11036)  

**Abstract**: Scientific fact-checking is vital for assessing claims in specialized domains such as biomedicine and materials science, yet existing systems often hallucinate or apply inconsistent reasoning, especially when verifying technical, compositional claims against an evidence snippet under source and cost/latency constraints. We present a pipeline centered on atomic predicate-argument decomposition and calibrated, uncertainty-gated corroboration: atomic facts are aligned to local snippets via embeddings, verified by a compact evidence-grounded checker, and only facts with uncertain support trigger domain-restricted web search over authoritative sources. The system supports both binary and tri-valued classification where it predicts labels from Supported, Refuted, NEI for three-way tasks. We evaluate under two regimes, Context-Only (no web) and Context+Web (uncertainty-gated web corroboration); when retrieved evidence conflicts with the provided context, we abstain with NEI rather than overriding the context. On multiple benchmarks, our framework surpasses the strongest benchmarks. In our experiments, web corroboration was invoked for only a minority of atomic facts on average, indicating that external evidence is consulted selectively under calibrated uncertainty rather than routinely. Overall, coupling atomic granularity with calibrated, uncertainty-gated corroboration yields more interpretable and context-conditioned verification, making the approach well-suited to high-stakes, single-document settings that demand traceable rationales, predictable cost/latency, and conservative. 

---
# Federated Single-Agent Robotics: Multi-Robot Coordination Without Intra-Robot Multi-Agent Fragmentation 

**Authors**: Xue Qin, Simin Luan, John See, Cong Yang, Zhijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11028)  

**Abstract**: As embodied robots move toward fleet-scale operation, multi-robot coordination is becoming a central systems challenge. Existing approaches often treat this as motivation for increasing internal multi-agent decomposition within each robot. We argue for a different principle: multi-robot coordination does not require intra-robot multi-agent fragmentation. Each robot should remain a single embodied agent with its own persistent runtime, local policy scope, capability state, and recovery authority, while coordination emerges through federation across robots at the fleet level. We present Federated Single-Agent Robotics (FSAR), a runtime architecture for multi-robot coordination built on single-agent robot runtimes. Each robot exposes a governed capability surface rather than an internally fragmented agent society. Fleet coordination is achieved through shared capability registries, cross-robot task delegation, policy-aware authority assignment, trust-scoped interaction, and layered recovery protocols. We formalize key coordination relations including authority delegation, inter-robot capability requests, local-versus-fleet recovery boundaries, and hierarchical human supervision, and describe a fleet runtime architecture supporting shared Embodied Capability Module (ECM) discovery, contract-aware cross-robot coordination, and fleet-level governance. We evaluate FSAR on representative multi-robot coordination scenarios against decomposition-heavy baselines. Results show statistically significant gains in governance locality (d=2.91, p<.001 vs. centralized control) and recovery containment (d=4.88, p<.001 vs. decomposition-heavy), while reducing authority conflicts and policy violations across all scenarios. Our results support the view that the path from embodied agents to embodied fleets is better served by federation across coherent robot runtimes than by fragmentation within them. 

---
# Optimal Stability of KL Divergence under Gaussian Perturbations 

**Authors**: Jialu Pan, Yufeng Zhang, Nan Hu, Keqin Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.11026)  

**Abstract**: We study the problem of characterizing the stability of Kullback-Leibler (KL) divergence under Gaussian perturbations beyond Gaussian families. Existing relaxed triangle inequalities for KL divergence critically rely on the assumption that all involved distributions are Gaussian, which limits their applicability in modern applications such as out-of-distribution (OOD) detection with flow-based generative models. In this paper, we remove this restriction by establishing a sharp stability bound between an arbitrary distribution and Gaussian families under mild moment conditions. Specifically, let $P$ be a distribution with finite second moment, and let $\mathcal{N}_1$ and $\mathcal{N}_2$ be multivariate Gaussian distributions. We show that if $KL(P||\mathcal{N}_1)$ is large and $KL(\mathcal{N}_1||\mathcal{N}_2)$ is at most $\epsilon$, then $KL(P||\mathcal{N}_2) \ge KL(P||\mathcal{N}_1) - O(\sqrt{\epsilon})$. Moreover, we prove that this $\sqrt{\epsilon}$ rate is optimal in general, even within the Gaussian family. This result reveals an intrinsic stability property of KL divergence under Gaussian perturbations, extending classical Gaussian-only relaxed triangle inequalities to general distributions. The result is non-trivial due to the asymmetry of KL divergence and the absence of a triangle inequality in general probability spaces. As an application, we provide a rigorous foundation for KL-based OOD analysis in flow-based models, removing strong Gaussian assumptions used in prior work. More broadly, our result enables KL-based reasoning in non-Gaussian settings arising in deep learning and reinforcement learning. 

---
# Brief2Design: A Multi-phased, Compositional Approach to Prompt-based Graphic Design 

**Authors**: Kotaro Kikuchi, Nami Ogawa  

**Link**: [PDF](https://arxiv.org/pdf/2604.11019)  

**Abstract**: Professional designers work from client briefs that specify goals and constraints but often lack concrete design details. Translating these abstract requirements into visual designs poses a central challenge, yet existing tools address specific aspects or induce fixation through complete outputs. Through interviews with six professional designers, we identified how designers address this challenge: first structuring ambiguous requirements, then exploring individual elements, and finally recombining alternatives. We developed Brief2Design, supporting this workflow through requirement extraction and recommendation, element-level exploration for objects, backgrounds, text, typography, and composition, and flexible recombination of selected elements. A within-subjects study with twelve designers compared Brief2Design against a conversational baseline. The structured approach increased prompt diversity and received high ratings for requirement extraction and recommendation, but required longer generation time and achieved comparable image diversity. These findings reveal that structured workflows benefit requirement clarification at the cost of efficiency, informing design trade-offs for AI-assisted graphic design tools. 

---
# NimbusGuard: A Novel Framework for Proactive Kubernetes Autoscaling Using Deep Q-Networks 

**Authors**: Chamath Wanigasooriya, Indrajith Ekanayake  

**Link**: [PDF](https://arxiv.org/pdf/2604.11017)  

**Abstract**: Cloud native architecture is about building and running scalable microservice applications to take full advantage of the cloud environments. Managed Kubernetes is the powerhouse orchestrating cloud native applications with elastic scaling. However, traditional Kubernetes autoscalers are reactive, meaning the scaling controllers adjust resources only after they detect demand within the cluster and do not incorporate any predictive measures. This can lead to either over-provisioning and increased costs or under-provisioning and performance degradation. We propose NimbusGuard, an open-source, Kubernetes-based autoscaling system that leverages a deep reinforcement learning agent to provide proactive autoscaling. The agents perception is augmented by a Long Short-Term Memory model that forecasts future workload patterns. The evaluations were conducted by comparing NimbusGuard against the built-in scaling controllers, such as Horizontal Pod Autoscaler, and the event-driven autoscaler KEDA. The experimental results demonstrate how NimbusGuard's proactive framework translates into superior performance and cost efficiency compared to existing reactive methods. 

---
# Panoptic Pairwise Distortion Graph 

**Authors**: Muhammad Kamran Janjua, Abdul Wahab, Bahador Rashidi  

**Link**: [PDF](https://arxiv.org/pdf/2604.11004)  

**Abstract**: In this work, we introduce a new perspective on comparative image assessment by representing an image pair as a structured composition of its regions. In contrast, existing methods focus on whole image analysis, while implicitly relying on region-level understanding. We extend the intra-image notion of a scene graph to inter-image, and propose a novel task of Distortion Graph (DG). DG treats paired images as a structured topology grounded in regions, and represents dense degradation information such as distortion type, severity, comparison and quality score in a compact interpretable graph structure. To realize the task of learning a distortion graph, we contribute (i) a region-level dataset, PandaSet, (ii) a benchmark suite, PandaBench, with varying region-level difficulty, and (iii) an efficient architecture, Panda, to generate distortion graphs. We demonstrate that PandaBench poses a significant challenge for state-of-the-art multimodal large language models (MLLMs) as they fail to understand region-level degradations even when fed with explicit region cues. We show that training on PandaSet or prompting with DG elicits region-wise distortion understanding, opening a new direction for fine-grained, structured pairwise image assessment. 

---
# When Valid Signals Fail: Regime Boundaries Between LLM Features and RL Trading Policies 

**Authors**: Zhengzhe Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10996)  

**Abstract**: Can large language models (LLMs) generate continuous numerical features that improve reinforcement learning (RL) trading agents? We build a modular pipeline where a frozen LLM serves as a stateless feature extractor, transforming unstructured daily news and filings into a fixed-dimensional vector consumed by a downstream PPO agent. We introduce an automated prompt-optimization loop that treats the extraction prompt as a discrete hyperparameter and tunes it directly against the Information Coefficient - the Spearman rank correlation between predicted and realized returns - rather than NLP losses. The optimized prompt discovers genuinely predictive features (IC above 0.15 on held-out data). However, these valid intermediate representations do not automatically translate into downstream task performance: during a distribution shift caused by a macroeconomic shock, LLM-derived features add noise, and the augmented agent under-performs a price-only baseline. In a calmer test regime the agent recovers, yet macroeconomic state variables remain the most robust driver of policy improvement. Our findings highlight a gap between feature-level validity and policy-level robustness that parallels known challenges in transfer learning under distribution shift. 

---
# Examining EAP Students' AI Disclosure Intention: A Cognition-Affect-Conation Perspective 

**Authors**: Yiran Du, Huimin He  

**Link**: [PDF](https://arxiv.org/pdf/2604.10991)  

**Abstract**: The growing use of generative artificial intelligence (AI) in academic writing has raised increasing concerns regarding transparency and academic integrity in higher education. This study examines the psychological factors influencing English for Academic Purposes (EAP) students' intention to disclose their use of AI tools. Drawing on the cognition-affect-conation framework, the study proposes a model integrating both enabling and inhibiting factors shaping disclosure intention. A sequential explanatory mixed-methods design was employed. Quantitative data from 324 EAP students at an English-medium instruction university in China were analysed using structural equation modelling, followed by semi-structured interviews with 15 students to further interpret the findings. The quantitative results indicate that psychological safety positively predicts AI disclosure intention, whereas fear of negative evaluation negatively predicts it. The qualitative findings further reveal that supportive teacher practices and clear guidance foster psychological safety, while policy ambiguity and reputational concerns intensify fear of negative evaluation and discourage disclosure. These findings highlight the importance of clear institutional policies and supportive pedagogical environments in promoting transparent AI use. 

---
# When Verification Fails: How Compositionally Infeasible Claims Escape Rejection 

**Authors**: Muxin Liu, Delip Rao, Grace Kim, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2604.10990)  

**Abstract**: Scientific claim verification, the task of determining whether claims are entailed by scientific evidence, is fundamental to establishing discoveries in evidence while preventing misinformation. This process involves evaluating each asserted constraint against validated evidence. Under the Closed-World Assumption (CWA), a claim is accepted if and only if all asserted constraints are positively supported. We show that existing verification benchmarks cannot distinguish models enforcing this standard from models applying a simpler shortcut called salient-constraint checking, which applies CWA's rejection criterion only to the most salient constraint and accepts when that constraint is supported. Because existing benchmarks construct infeasible claims by perturbing a single salient element they are insufficient at distinguishing between rigorous claim verification and simple salient-constraint reliance. To separate the two, we construct compositionally infeasible claims where the salient constraint is supported but a non-salient constraint is contradicted. Across model families and modalities, models that otherwise saturate existing benchmarks consistently over-accept these claims, confirming the prevalence of such shortcut reasoning. Via model context interventions, we show that different models and prompting strategies occupy distinct positions on a shared ROC curve, indicating that the gap between model families reflects differences in verification threshold rather than underlying reasoning ability, and that the compositional inference bottleneck is a structural property of current verification behavior that strategy guidance alone cannot overcome. 

---
# Enabling and Inhibitory Pathways of Students' AI Use Concealment Intention in Higher Education: Evidence from SEM and fsQCA 

**Authors**: Yiran Du, Huimin He  

**Link**: [PDF](https://arxiv.org/pdf/2604.10978)  

**Abstract**: This study investigates students' AI use concealment intention in higher education by integrating the cognition-affect-conation (CAC) framework with a dual-method approach combining structural equation modelling (SEM) and fuzzy-set qualitative comparative analysis (fsQCA). Drawing on data from 1346 university students, the findings reveal two opposing mechanisms shaping concealment intention. The enabling pathway shows that perceived stigma, perceived risk, and perceived policy uncertainty increase fear of negative evaluation, which in turn promotes concealment. In contrast, the inhibitory pathway demonstrates that AI self-efficacy, perceived fairness, and perceived social support enhance psychological safety, thereby reducing concealment intention. SEM results confirm the hypothesised relationships and mediation effects, while fsQCA identifies multiple configurational pathways, highlighting equifinality and the central role of fear of negative evaluation across conditions. The study contributes to the literature by conceptualising concealment as a distinct behavioural outcome and by providing a nuanced explanation that integrates both net-effect and configurational perspectives. Practical implications emphasise the need for clear institutional policies, destigmatisation of appropriate AI use, and the cultivation of supportive learning environments to promote transparency. 

---
# MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with Multimodal Large Language Models 

**Authors**: Xincheng Yao, Zefeng Qian, Chao Shi, Jiayang Song, Chongyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10971)  

**Abstract**: In the progress of industrial anomaly detection, general anomaly detection (GAD) is an emerging trend and also the ultimate goal. Unlike the conventional single- and multi-class AD, general AD aims to train a general AD model that can directly detect anomalies in diverse novel classes without any retraining or fine-tuning on the target data. Recently, Multimodal Large Language Models (MLLMs) have shown great promise in achieving general anomaly detection due to their revolutionary visual understanding and language reasoning capabilities. However, MLLM's general AD ability remains underexplored due to: (1) MLLMs are pretrained on amounts of data sourced from the Web, these data still have significant gaps with the data in AD scenarios. Moreover, the image-text pairs during pretraining are also not specifically for AD tasks. (2) The current mainstream AD datasets are image-based and not yet suitable for post-training MLLMs. To facilitate MLLM-based general AD research, we present MMR-AD, which is a comprehensive benchmark for both training and evaluating MLLM-based AD models. With MMR-AD, we reveal that the AD performance of current SOTA generalist MLLMs still falls far behind the industrial requirements. Based on MMR-AD, we also propose a baseline model, Anomaly-R1, which is a reasoning-based AD model that learns from the CoT data in MMR-AD and is further enhanced by reinforcement learning. Extensive experiments show that our Anomaly-R1 achieves remarkable improvements over generalist MLLMs in both anomaly detection and localization. 

---
# Towards Automated Solar Panel Integrity: Hybrid Deep Feature Extraction for Advanced Surface Defect Identification 

**Authors**: Muhammad Junaid Asif, Muhammad Saad Rafaqat, Usman Nazakat, Uzair Khan, Rana Fayyaz Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2604.10969)  

**Abstract**: To ensure energy efficiency and reliable operations, it is essential to monitor solar panels in generation plants to detect defects. It is quite labor-intensive, time consuming and costly to manually monitor large-scale solar plants and those installed in remote areas. Manual inspection may also be susceptible to human errors. Consequently, it is necessary to create an automated, intelligent defect-detection system, that ensures continuous monitoring, early fault detection, and maximum power generation. We proposed a novel hybrid method for defect detection in SOLAR plates by combining both handcrafted and deep learning features. Local Binary Pattern (LBP), Histogram of Gradients (HoG) and Gabor Filters were used for the extraction of handcrafted features. Deep features extracted by leveraging the use of DenseNet-169. Both handcrafted and deep features were concatenated and then fed to three distinct types of classifiers, including Support Vector Machines (SVM), Extreme Gradient Boost (XGBoost) and Light Gradient-Boosting Machine (LGBM). Experimental results evaluated on the augmented dataset show the superior performance, especially DenseNet-169 + Gabor (SVM), had the highest scores with 99.17% accuracy which was higher than all the other systems. In general, the proposed hybrid framework offers better defect-detection accuracy, resistance, and flexibility that has a solid basis on the real-life use of the automated PV panels monitoring system. 

---
# You Only Judge Once: Multi-response Reward Modeling in a Single Forward Pass 

**Authors**: Yinuo Yang, Zixian Ma, Manasi Ganti, Jieyu Zhang, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2604.10966)  

**Abstract**: We present a discriminative multimodal reward model that scores all candidate responses in a single forward pass. Conventional discriminative reward models evaluate each response independently, requiring multiple forward passes, one for each potential response. Our approach concatenates multiple responses with separator tokens and applies cross-entropy over their scalar scores, enabling direct comparative reasoning and efficient $N$-way preference learning. The multi-response design also yields up to $N\times$ wall-clock speedup and FLOPs reduction over conventional single-response scoring. To enable $N$-way reward evaluation beyond existing pairwise benchmarks, we construct two new benchmarks: (1) MR$^2$Bench-Image contains human-annotated rankings over responses from 8 diverse models; (2) MR$^2$Bench-Video is a large-scale video-based reward benchmark derived from 94K crowdsourced pairwise human judgments over video question-answering spanning 19 models, denoised via preference graph ensemble. Both benchmarks provide 4-response evaluation variants sampled from the full rankings. Built on a 4B vision-language backbone with LoRA fine-tuning and a lightweight MLP value head, our model achieves state-of-the-art results on six multimodal reward benchmarks, including MR$^2$Bench-Image, MR$^2$Bench-Video, and four other existing benchmarks. Our model outperforms existing larger generative and discriminative reward models. We further demonstrate that our reward model, when used in reinforcement learning with GRPO, produces improved policy models that maintain performance across standard multimodal benchmarks while substantially improving open-ended generation quality, outperforming a single-response discriminative reward model (RM) baseline by a large margin in both training stability and open-ended generation quality. 

---
# Continuous-time Online Learning via Mean-Field Neural Networks: Regret Analysis in Diffusion Environments 

**Authors**: Erhan Bayraktar, Bingyan Han, Ziqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10958)  

**Abstract**: We study continuous-time online learning where data are generated by a diffusion process with unknown coefficients. The learner employs a two-layer neural network, continuously updating its parameters in a non-anticipative manner. The mean-field limit of the learning dynamics corresponds to a stochastic Wasserstein gradient flow adapted to the data filtration. We establish regret bounds for both the mean-field limit and finite-particle system. Our analysis leverages the logarithmic Sobolev inequality, Polyak-Lojasiewicz condition, Malliavin calculus, and uniform-in-time propagation of chaos. Under displacement convexity, we obtain a constant static regret bound. In the general non-convex setting, we derive explicit linear regret bounds characterizing the effects of data variation, entropic exploration, and quadratic regularization. Finally, our simulations demonstrate the outperformance of the online approach and the impact of network width and regularization parameters. 

---
# A molecular clock for writing systems reveals the quantitative impact of imperial power on cultural evolution 

**Authors**: Hiroki Fukui  

**Link**: [PDF](https://arxiv.org/pdf/2604.10957)  

**Abstract**: Writing systems are cultural replicators whose evolution has never been studied quantitatively at global scale. We compile the Global Script Database (GSD): 300 writing and notation systems, 50 binary structural characters, and 259 phylogenetic edges spanning 5,400 years. Applying four methods -- phenetics, cladistics, Bayesian inference, and neural network clustering -- we find that scripts exhibit a detectable molecular clock. The best-fitting model (Mk+Gamma strict clock) yields a substitution rate of q = 0.226 substitutions/character/millennium (95% CI: 0.034-1.22; Delta BIC = -4.1 versus relaxed clock; Delta BIC = -1,364.7 versus Mk without rate variation). Political interventions break this clock: deviation from expected divergence times correlates with intervention intensity (Spearman rho = 0.556, p < 10^{-4}), and per-character rate analysis reveals that intervention selectively rewrites deep structural features rather than merely accelerating change (rate profile correlation rho = 0.320). We identify 30 major script replacement events and rank their destructive impact. A ceiling effect suppresses independent invention wherever writing already exists (Fisher's exact OR = 0.054, p < 10^{-6}), and colonial contact predicts script extinction (Cox HR = 5.25, p = 0.0006). The Spanish Empire extinguished the most scripts (6 of 12 contacted, 50%), followed by the Empire of Japan (3 of 9, 33.3%). Feature coding was validated by inter-rater reliability testing with two independent human coders (Cohen's kappa = 0.877; human-LLM kappa = 0.929; Fleiss' kappa = 0.911). 

---
# Pseudo-Unification: Entropy Probing Reveals Divergent Information Patterns in Unified Multimodal Models 

**Authors**: Songlin Yang, Xianghao Kong, Anyi Rao  

**Link**: [PDF](https://arxiv.org/pdf/2604.10949)  

**Abstract**: Unified multimodal models (UMMs) were designed to combine the reasoning ability of large language models (LLMs) with the generation capability of vision models. In practice, however, this synergy remains elusive: UMMs fail to transfer LLM-like reasoning to image synthesis and exhibit divergent response behaviors. We term this phenomenon pseudo-unification. Diagnosing its internal causes is important, but existing probing methods either lack model-internal insight or ignore prompt-response dependencies. To address these limitations, we propose an information-theoretic probing framework that jointly analyzes how UMMs encode inputs and generate outputs. Applied to ten representative UMMs, our framework reveals that pseudo-unification stems from a dual divergence: (i) Modality-Asymmetric Encoding, where vision and language follow different entropy trajectories, and (ii) Pattern-Split Response, where text generation exhibits high-entropy creativity while image synthesis enforces low-entropy fidelity. Only models that unify both sides (e.g., via contextual prediction) achieve more genuine unification, enabling stronger reasoning-based text-to-image generation even with fewer parameters. Our work provides the first model-internal probing of unification, demonstrating that real multimodal synergy requires consistency in information flow, not just shared parameters. 

---
# QShield: Securing Neural Networks Against Adversarial Attacks using Quantum Circuits 

**Authors**: Navid Azimi, Aditya Prakash, Yao Wang, Li Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.10933)  

**Abstract**: Deep neural networks remain highly vulnerable to adversarial perturbations, limiting their reliability in security- and safety-critical applications. To address this challenge, we introduce QShield, a modular hybrid quantum-classical neural network (HQCNN) architecture designed to enhance the adversarial robustness of classical deep learning models. QShield integrates a conventional convolutional neural network (CNN) backbone for feature extraction with a quantum processing module that encodes the extracted features into quantum states, applies structured entanglement operations under realistic noise models, and outputs a hybrid prediction through a dynamically weighted fusion mechanism implemented via a lightweight multilayer perceptron (MLP). We systematically evaluate both classical and hybrid quantum-classical models on the MNIST, OrganAMNIST, and CIFAR-10 datasets, using a comprehensive set of robustness, efficiency, and computational performance metrics. Our results demonstrate that classical models are highly vulnerable to adversarial attacks, whereas the proposed hybrid models with entanglement patterns maintain high predictive accuracy while substantially reducing attack success rates across a wide range of adversarial attacks. Furthermore, the proposed hybrid architecture significantly increased the computational cost required to generate adversarial examples, thereby introducing an additional layer of defense. These findings indicate that the proposed modular hybrid architecture achieves a practical balance between predictive accuracy and adversarial robustness, positioning it as a promising approach for secure and reliable machine learning in sensitive and safety-critical applications. 

---
# Mem$^2$Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation 

**Authors**: Zihao Cheng, Zeming Liu, Yingyu Shan, Xinyi Wang, Xiangrong Zhu, Yunpu Ma, Hongru Wang, Yuhang Guo, Wei Lin, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10923)  

**Abstract**: While large language model--powered agents can self-evolve by accumulating experience or by dynamically creating new assets (i.e., tools or expert agents), existing frameworks typically treat these two evolutionary processes in isolation. This separation overlooks their intrinsic interdependence: the former is inherently bounded by a manually predefined static toolset, while the latter generates new assets from scratch without experiential guidance, leading to limited capability growth and unstable evolution. To address this limitation, we introduce a novel paradigm of co-evolutionary Capability Expansion and Experience Distillation. Guided by this paradigm, we propose the \textbf{Mem$^{\textbf{2}}$Evolve}, which integrates two core components: \textbf{Experience Memory} and \textbf{Asset Memory}. Specifically, Mem$^{2}$Evolve leverages accumulated experience to guide the dynamic creation of assets, thereby expanding the agent's capability space while simultaneously acquiring new experience to achieve co-evolution. Extensive experiments across 6 task categories and 8 benchmarks demonstrate that Mem$^{2}$Evolve achieves improvement of 18.53\% over standard LLMs, 11.80\% over agents evolving solely through experience, and 6.46\% over those evolving solely through asset creation, establishing it as a substantially more effective and stable self-evolving agent framework. Code is available at: this https URL. 

---
# ReXSonoVQA: A Video QA Benchmark for Procedure-Centric Ultrasound Understanding 

**Authors**: Xucheng Wang, Xiaoman Zhang, Sung Eun Kim, Ankit Pal, Pranav Rajpurkar  

**Link**: [PDF](https://arxiv.org/pdf/2604.10916)  

**Abstract**: Ultrasound acquisition requires skilled probe manipulation and real-time adjustments. Vision-language models (VLMs) could enable autonomous ultrasound systems, but existing benchmarks evaluate only static images, not dynamic procedural understanding. We introduce ReXSonoVQA, a video QA benchmark with 514 video clips and 514 questions (249 MCQ, 265 free-response) targeting three competencies: Action-Goal Reasoning, Artifact Resolution & Optimization, and Procedure Context & Planning. Zero-shot evaluation of Gemini 3 Pro, Qwen3.5-397B, LLaVA-Video-72B, and Seed 2.0 Pro shows VLMs can extract some procedural information, but troubleshooting questions remain challenging with minimal gains over text-only baselines, exposing limitations in causal reasoning. ReXSonoVQA enables developing perception systems for ultrasound training, guidance, and robotic automation. 

---
# Audio Flamingo Next: Next-Generation Open Audio-Language Models for Speech, Sound, and Music 

**Authors**: Sreyan Ghosh, Arushi Goel, Kaousheik Jayakumar, Lasha Koroshinadze, Nishit Anand, Zhifeng Kong, Siddharth Gururani, Sang-gil Lee, Jaehyeon Kim, Aya Aljafari, Chao-Han Huck Yang, Sungwon Kim, Ramani Duraiswami, Dinesh Manocha, Mohammad Shoeybi, Bryan Catanzaro, Ming-Yu Liu, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2604.10905)  

**Abstract**: We present Audio Flamingo Next (AF-Next), the next-generation and most capable large audio-language model in the Audio Flamingo series, designed to advance understanding and reasoning over speech, environmental sounds and music. Compared to Audio Flamingo 3, AF-Next introduces: (i) a stronger foundational audio-language model that significantly improves accuracy across diverse audio understanding tasks; (ii) scalable strategies for constructing large-scale audio understanding and reasoning data beyond existing academic benchmarks; (iii) support for long and complex audio inputs up to 30 minutes; and (iv) Temporal Audio Chain-of-Thought, a new reasoning paradigm that explicitly grounds intermediate reasoning steps to timestamps in long audio, enabling fine-grained temporal alignment and improved interpretability. To enable these capabilities, we first conduct a systematic analysis of Audio Flamingo 3 to identify key gaps in audio understanding and reasoning. We then curate and scale new large-scale datasets totaling over 1 million hours to address these limitations and expand the existing AudioSkills-XL, LongAudio-XL, AF-Think and AF-Chat datasets. AF-Next is trained using a curriculum-based strategy spanning pre-training, mid-training and post-training stages. Extensive experiments across 20 audio understanding and reasoning benchmarks, including challenging long-audio tasks, show that AF-Next outperforms similarly sized open models by large margins and remains highly competitive with and sometimes surpasses, much larger open-weight and closed models. Beyond benchmark performance, AF-Next exhibits strong real-world utility and transfers well to unseen tasks, highlighting its robustness and generalization ability. In addition to all data, code and methods, we open-source 3 variants of AF-Next, including AF-Next-Instruct, AF-Next-Think and AF-Next-Captioner. 

---
# Evaluating the Impact of Medical Image Reconstruction on Downstream AI Fairness and Performance 

**Authors**: Matteo Wohlrapp, Niklas Bubeck, Daniel Rueckert, William Lotter  

**Link**: [PDF](https://arxiv.org/pdf/2604.10904)  

**Abstract**: AI-based image reconstruction models are increasingly deployed in clinical workflows to improve image quality from noisy data, such as low-dose X-rays or accelerated MRI scans. However, these models are typically evaluated using pixel-level metrics like PSNR, leaving their impact on downstream diagnostic performance and fairness unclear. We introduce a scalable evaluation framework that applies reconstruction and diagnostic AI models in tandem, which we apply to two tasks (classification, segmentation), three reconstruction approaches (U-Net, GAN, diffusion), and two data types (X-ray, MRI) to assess the potential downstream implications of reconstruction. We find that conventional reconstruction metrics poorly track task performance, where diagnostic accuracy remains largely stable even as reconstruction PSNR declines with increasing image noise. Fairness metrics exhibit greater variability, with reconstruction sometimes amplifying demographic biases, particularly regarding patient sex. However, the overall magnitude of this additional bias is modest compared to the inherent biases already present in diagnostic models. To explore potential bias mitigation, we adapt two strategies from classification literature to the reconstruction setting, but observe limited efficacy. Overall, our findings emphasize the importance of holistic performance and fairness assessments throughout the entire medical imaging workflow, especially as generative reconstruction models are increasingly deployed. 

---
# Beyond A Fixed Seal: Adaptive Stealing Watermark in Large Language Models 

**Authors**: Shuhao Zhang, Yuli Chen, Jiale Han, Bo Cheng, Jiabao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.10893)  

**Abstract**: Watermarking provides a critical safeguard for large language model (LLM) services by facilitating the detection of LLM-generated text. Correspondingly, stealing watermark algorithms (SWAs) derive watermark information from watermarked texts generated by victim LLMs to craft highly targeted adversarial attacks, which compromise the reliability of watermarks. Existing SWAs rely on fixed strategies, overlooking the non-uniform distribution of stolen watermark information and the dynamic nature of real-world LLM generation processes. To address these limitations, we propose Adaptive Stealing (AS), a novel SWA featuring enhanced design flexibility through Position-Based Seal Construction and Adaptive Selection modules. AS operates by defining multiple attack perspectives derived from distinct activation states of contextually ordered tokens. During attack execution, AS dynamically selects the optimal perspective based on watermark compatibility, generation priority, and dynamic generation relevance. Our experiments demonstrate that AS significantly increases steal efficiency against target watermarks under identical experimental conditions. These findings highlight the need for more robust LLM watermarks to withstand potential attacks. We release our code to the community for future research\footnote{this https URL}. 

---
# Product Review Based on Optimized Facial Expression Detection 

**Authors**: Vikrant Chaugule, Abhishek D, Aadheeshwar Vijayakumar, Pravin Bhaskar Ramteke, Shashidhar G. Koolagudi  

**Link**: [PDF](https://arxiv.org/pdf/2604.10885)  

**Abstract**: This paper proposes a method to review public acceptance of products based on their brand by analyzing the facial expression of the customer intending to buy the product from a supermarket or hypermarket. In such cases, facial expression recognition plays a significant role in product review. Here, facial expression detection is performed by extracting feature points using a modified Harris algorithm. The modified Harris algorithm reduced the time complexity of the existing feature extraction Harris Algorithm. A comparison of time complexities of existing algorithms is done with proposed algorithm. The algorithm proved to be significantly faster and nearly accurate for the needed application by reducing the time complexity for corner points detection. 

---
# Ambiguity Detection and Elimination in Automated Executable Process Modeling 

**Authors**: Ion Matei, Praveen Kumar Menaka Sekar, Maksym Zhenirovskyy, Hon Yung Wong, Sayuri Kohmura, Shinji Hotta, Akihiro Inomata  

**Link**: [PDF](https://arxiv.org/pdf/2604.10884)  

**Abstract**: Automated generation of executable Business Process Model and Notation (BPMN) models from natural-language specifications is increasingly enabled by large language models. However, ambiguous or underspecified text can yield structurally valid models with different simulated behavior. Our goal is not to prove that one generated BPMN model is semantically correct, but to detect when a natural-language specification fails to support a stable executable interpretation under repeated generation and simulation. We present a diagnosis-driven framework that detects behavioral inconsistency from the empirical distribution of key performance indicators (KPIs), localizes divergence to gateway logic using model-based diagnosis, maps that logic back to verbatim narrative segments, and repairs the source text through evidence-based refinement. Experiments on diabetic nephropathy health-guidance policies show that the method reduces variability in regenerated model behavior. The result is a closed-loop approach for validating and repairing executable process specifications in the absence of ground-truth BPMN models. 

---
# DIB-OD: Preserving the Invariant Core for Robust Heterogeneous Graph Adaptation via Decoupled Information Bottleneck and Online Distillation 

**Authors**: Yang Yan, Qiuyan Wang, Tianjin Huang, Qiudong Yu, Kexin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10882)  

**Abstract**: Graph Neural Network pretraining is pivotal for leveraging unlabeled graph data. However, generalizing across heterogeneous domains remains a major challenge due to severe distribution shifts. Existing methods primarily focus on intra-domain patterns, failing to disentangle task-relevant invariant knowledge from domain-specific redundant noise, leading to negative transfer and catastrophic forgetting. To this end, we propose DIB-OD, a novel framework designed to preserve the invariant core for robust heterogeneous graph adaptation through a Decoupled Information Bottleneck and Online Distillation framework. Our core innovation is the explicit decomposition of representations into orthogonal invariant and redundant subspaces. By utilizing an Information Bottleneck teacher-student distillation mechanism and the Hilbert-Schmidt Independence Criterion, we isolate a stable invariant core that transcends domain boundaries. Furthermore, a self-adaptive semantic regularizer is introduced to protect this core from corruption during target-domain adaptation by dynamically gating label influence based on predictive confidence. Extensive experiments across chemical, biological, and social network domains demonstrate that DIB-OD significantly outperforms state-of-the-art methods, particularly in challenging inter-type domain transfers, showcasing superior generalization and anti-forgetting performance. 

---
# Compliant But Unsatisfactory: The Gap Between Auditing Standards and Practices for Probabilistic Genotyping Software 

**Authors**: Angela Jin, Alexander Asemota, Dan E. Krane, Nathaniel D. Adams, Rediet Abebe  

**Link**: [PDF](https://arxiv.org/pdf/2604.10875)  

**Abstract**: AI governance efforts increasingly rely on audit standards: agreed-upon practices for conducting audits. However, poorly designed standards can hide and lend credibility to inadequate systems. We explore how an audit standard's design influences its effectiveness through a case study of ASB 018, a standard for auditing probabilistic genotyping software -- software that the U.S. criminal legal system increasingly uses to analyze DNA samples. Through qualitative analysis of ASB 018 and five audit reports, we identify numerous gaps between the standard's desired outcomes and the auditing practices it enables. For instance, ASB 018 envisions that compliant audits establish restrictions on software use based on observed failures. However, audits can comply without establishing such boundaries. We connect these gaps to the design of the standard's requirements such as vague language and undefined terms. We conclude with recommendations for designing audit standards and evaluating their effectiveness. 

---
# AOP-Smart: A RAG-Enhanced Large Language Model Framework for Adverse Outcome Pathway Analysis 

**Authors**: Qinjiang Niu, Lu Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.10874)  

**Abstract**: Adverse Outcome Pathways (AOPs) are an important knowledge framework in toxicological research and risk assessment. In recent years, large language models (LLMs) have gradually been applied to AOP-related question answering and mechanistic reasoning tasks. However, due to the existence of the hallucination problem, that is, the model may generate content that is inconsistent with facts or lacks evidence, their reliability is still limited. To address this issue, this study proposes an AOP-oriented Retrieval-Augmented Generation (RAG) framework, AOP-Smart. Based on the official XML data from AOP-Wiki, this method uses Key Events (KEs), Key Event Relationships (KERs), and specific AOP information to retrieve relevant knowledge for user questions, thereby improving the reliability of the generated results of large language models. To evaluate the effectiveness of the proposed method, this study constructed a test set containing 20 AOP-related question answering tasks, covering KE identification, upstream and downstream KE retrieval, and complex AOP retrieval tasks. Experiments were conducted on three mainstream large language models, Gemini, DeepSeek, and ChatGPT, and comparative tests were performed under two settings: without RAG and with RAG. The experimental results show that, without using RAG, the accuracies of GPT, DeepSeek, and Gemini were 15.0\%, 35.0\%, and 20.0\%, respectively; after using RAG, their accuracies increased to 95.0\%, 100.0\%, and 95.0\%, respectively. The results indicate that AOP-Smart can significantly alleviate the hallucination problem of large language models in AOP knowledge tasks, and greatly improve the accuracy and consistency of their answers. 

---
# Query Lower Bounds for Diffusion Sampling 

**Authors**: Zhiyang Xun, Eric Price  

**Link**: [PDF](https://arxiv.org/pdf/2604.10857)  

**Abstract**: Diffusion models generate samples by iteratively querying learned score estimates. A rapidly growing literature focuses on accelerating sampling by minimizing the number of score evaluations, yet the information-theoretic limits of such acceleration remain unclear.
In this work, we establish the first score query lower bounds for diffusion sampling. We prove that for $d$-dimensional distributions, given access to score estimates with polynomial accuracy $\varepsilon=d^{-O(1)}$ (in any $L^p$ sense), any sampling algorithm requires $\widetilde{\Omega}(\sqrt{d})$ adaptive score queries. In particular, our proof shows that any sampler must search over $\widetilde{\Omega}(\sqrt{d})$ distinct noise levels, providing a formal explanation for why multiscale noise schedules are necessary in practice. 

---
# BridgeSim: Unveiling the OL-CL Gap in End-to-End Autonomous Driving 

**Authors**: Seth Z. Zhao, Luobin Wang, Hongwei Ruan, Yuxin Bao, Yilan Chen, Ziyang Leng, Abhijit Ravichandran, Honglin He, Zewei Zhou, Xu Han, Abhishek Peri, Zhiyu Huang, Pranav Desai, Henrik Christensen, Jiaqi Ma, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.10856)  

**Abstract**: Open-loop (OL) to closed-loop (CL) gap (OL-CL gap) exists when OL-pretrained policies scoring high in OL evaluations fail to transfer effectively in closed-loop (CL) deployment. In this paper, we unveil the root causes of this systemic failure and propose a practical remedy. Specifically, we demonstrate that OL policies suffer from Observational Domain Shift and Objective Mismatch. We show that while the former is largely recoverable with adaptation techniques, the latter creates a structural inability to model complex reactive behaviors, which forms the primary OL-CL gap. We find that a wide range of OL policies learn a biased Q-value estimator that neglects both the reactive nature of CL simulations and the temporal awareness needed to reduce compounding errors. To this end, we propose a Test-Time Adaptation (TTA) framework that calibrates observational shift, reduces state-action biases, and enforces temporal consistency. Extensive experiments show that TTA effectively mitigates planning biases and yields superior scaling dynamics than its baseline counterparts. Furthermore, our analysis highlights the existence of blind spots in standard OL evaluation protocols that fail to capture the realities of closed-loop deployment. 

---
# Task2vec Readiness: Diagnostics for Federated Learning from Pre-Training Embeddings 

**Authors**: Cristiano Mafuz, Rodrigo Silva  

**Link**: [PDF](https://arxiv.org/pdf/2604.10849)  

**Abstract**: Federated learning (FL) performance is highly sensitive to heterogeneity across clients, yet practitioners lack reliable methods to anticipate how a federation will behave before training. We propose readiness indices, derived from Task2Vec embeddings, that quantifies the alignment of a federation prior to training and correlates with its eventual performance. Our approach computes unsupervised metrics -- such as cohesion, dispersion, and density -- directly from client embeddings. We evaluate these indices across diverse datasets (CIFAR-10, FEMNIST, PathMNIST, BloodMNIST) and client counts (10--20), under Dirichlet heterogeneity levels spanning $\alpha \in \{0.05,\dots,5.0\}$ and FedAVG aggregation strategy. Correlation analyses show consistent and significant Pearson and Spearman coefficients between some of the Task2Vec-based readiness and final performance, with values often exceeding 0.9 across dataset$\times$client configurations, validating this approach as a robust proxy for FL outcomes. These findings establish Task2Vec-based readiness as a principled, pre-training diagnostic for FL that may offer both predictive insight and actionable guidance for client selection in heterogeneous federations. 

---
# Retinal Cyst Detection from Optical Coherence Tomography Images 

**Authors**: Abhishek Dharmaratnakar, Aadheeshwar Vijayakumar, Suchand Dayanand  

**Link**: [PDF](https://arxiv.org/pdf/2604.10843)  

**Abstract**: Retinal Cysts are formed by leakage and accumulation of fluid in the retina due to the incompetence of retinal vasculature. These cystic spaces have significance in several ocular diseases such as age-related macular degeneration, diabetic macular edema, etc. Optical coherence tomography is one of the predominant diagnosing techniques for imaging retinal pathologies. Segmenting and quantification of intraretinal cysts plays the vital role in predicting visual acuity. In literature, several methods have been proposed for automatic segmentation of intraretinal cysts. As cystoid macular edema becomes a major problem to humankind, we need to quantify it accurately and operate it out, else it might cause many problems later on. Though research is being carried out in this area, not much of progress has been made and accuracy achieved so far is 68\% which is very less. Also, the methods depend on the quality of the image and give very low results for high noise images like topcon. This work uses ResNet CNN (Convolutional Neural Network) approach of segmentation by the way of patchwise classification for training on image set from cyst segmentation challenge dataset and testing on test data set given by 2 different graders for all 4 vendors in the challenge. It also compares these methods using first publicly available novel cyst segmentation challenge dataset. The methods were evaluated using quantitative measures to assess their robustness against the challenges of intraretinal cyst segmentation. The results are found to be better than the previous state of the art approaches giving more than 70\% dice coefficient on all vendors irrespective of their quality. 

---
# Resilient Write: A Six-Layer Durable Write Surface for LLM Coding Agents 

**Authors**: Justice Owusu Agyemang, Jerry John Kponyo, Elliot Amponsah, Godfred Manu Addo Boakye, Kwame Opuni-Boachie Obour Agyekum  

**Link**: [PDF](https://arxiv.org/pdf/2604.10842)  

**Abstract**: LLM-powered coding agents increasingly rely on tool-use protocols such as the Model Context Protocol~(MCP) to read and write files on a developer's workstation. When a write fails -- due to content filters, truncation, or an interrupted session -- the agent typically receives no structured signal, loses the draft, and wastes tokens retrying blindly. We present \textbf{Resilient Write}, an MCP server that interposes a six-layer durable write surface between the agent and the filesystem. The layers -- pre-flight risk scoring, transactional atomic writes, resume-safe chunking, structured typed errors, out-of-band scratchpad storage, and task-continuity handoff envelopes -- are orthogonal and independently adoptable. Each layer maps to a concrete failure mode observed during a real agent session in April~2026, in which content-safety filters silently rejected a draft containing redacted API-key prefixes. Three additional tools -- chunk preview, format-aware validation, and journal analytics -- emerged from using the system to compose this paper. A 186-test suite validates correctness at each layer, and quantitative comparison against naive and defensive baselines shows a 5x reduction in recovery time and a 13x improvement in agent self-correction rate. Resilient Write is open-source under the MIT license. 

---
# Harnessing Photonics for Machine Intelligence 

**Authors**: Hanqing Zhu, Shupeng Ning, Hongjian Zhou, Ziang Yin, Ray T. Chen, Jiaqi Gu, David Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.10841)  

**Abstract**: The exponential growth of machine-intelligence workloads is colliding with the power, memory, and interconnect limits of the post-Moore era, motivating compute substrates that scale beyond transistor density alone. Integrated photonics is emerging as a candidate for artificial intelligence (AI) acceleration by exploiting optical bandwidth and parallelism to reshape data movement and computation. This review reframes photonic computing from a circuits-and-systems perspective, moving beyond building-block progress toward cross-layer system analysis and full-stack design automation. We synthesize recent advances through a bottleneck-driven taxonomy that delineates the operating regimes and scaling trends where photonics can deliver end-to-end sustained benefits. A central theme is cross-layer co-design and workload-adaptive programmability to sustain high efficiency and versatility across evolving application domains at scale. We further argue that Electronic-Photonic Design Automation (EPDA) will be pivotal, enabling closed-loop co-optimization across simulation, inverse design, system modeling, and physical implementation. By charting a roadmap from laboratory prototypes to scalable, reproducible electronic-photonic ecosystems, this review aims to guide the CAS community toward an automated, system-centric era of photonic machine intelligence. 

---
# LLMs for Qualitative Data Analysis Fail on Security-specificComments in Human Experiments 

**Authors**: Maria Camporese, Fabio Massacci, Yuanjun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2604.10834)  

**Abstract**: [Background:] Thematic analysis of free-text justifications in human experiments provides significant qualitative insights. Yet, it is costly because reliable annotations require multiple domain experts. Large language models (LLMs) seem ideal candidates to replace human annotators. [Problem:] Coding security-specific aspects (code identifiers mentioned, lines-of-code mentioned, security keywords mentioned) may require deeper contextual understanding than sentiment classification. [Objective:] Explore whether LLMs can act as automated annotators for technical security comments by human subjects. [Method:] We prompt four top-performing LLMs on LiveBench to detect nine security-relevant codes in free-text comments by human subjects analyzing vulnerable code snippets. Outputs are compared to human annotators using Cohen's Kappa (chance-corrected accuracy). We test different prompts mimicking annotation best practices, including emerging codes, detailed codebooks with examples, and conflicting examples. [Negative Results:] We observed marked improvements only when using detailed code descriptions; however, these improvements are not uniform across codes and are insufficient to reliably replace a human annotator. [Limitations:] Additional studies with more LLMs and annotation tasks are needed. 

---
# Speaking to No One: Ontological Dissonance and the Double Bind of Conversational AI 

**Authors**: Hugh Brosnahan, Izabela Lipinska  

**Link**: [PDF](https://arxiv.org/pdf/2604.10833)  

**Abstract**: Recent reports indicate that sustained interaction with conversational artificial intelligence (AI) systems can, in a small subset of users, contribute to the emergence or stabilisation of delusional experience. Existing accounts typically attribute such cases either to individual vulnerability or to failures of safety engineering. These explanations are incomplete. Drawing on phenomenology, psychiatry, and cognitive neuroscience, this paper argues that the risk arises from the relational and ontological structure of the interaction itself. Conversational AI generates ontological dissonance: a conflict between the appearance of relational presence and the absence of any subject capable of sustaining it. Maintained through a communicative double bind and amplified by attentional asymmetries, this dissonance tends, under conditions of affective vulnerability, to stabilise into a technologically mediated analogue of folie a deux. This account explains why explicit disclaimers often fail to disrupt delusional involvement and clarifies the ethical and clinical implications for the design and use of conversational AI. 

---
# MeloTune: On-Device Arousal Learning and Peer-to-Peer Mood Coupling for Proactive Music Curation 

**Authors**: Hongwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10815)  

**Abstract**: MeloTune is an iPhone-deployed music agent that instantiates the Mesh Memory Protocol (MMP) and Symbolic-Vector Attention Fusion (SVAF) as a production system for affect-aware music curation with peer-to-peer mood coupling. Each device runs two closed-form continuous-time (CfC) networks: a private listener-level CfC that predicts a short-horizon affective trajectory on Russell's circumplex and drives proactive curation, and a shared mesh-runtime CfC at MMP Layer 6 that integrates Cognitive Memory Blocks (CMBs) from co-listening peers. CfC hidden states never cross the wire; only structured CMBs do. A Personal Arousal Function (PAF) replaces the standard linear mapping from audio intensity to psychological arousal with a per-listener learned adjustment, trained from behavioral signals (skip, completion, favorite, volume) and from drift between user-declared mood and machine inference. The same track receives different arousal predictions for different listeners. The model (94,552 parameters) achieves trajectory MAE 0.414, pattern accuracy 96.6%, and intent accuracy 69.4% on held-out validation. PAF evidence from a live deployment session (46 observations across 11 genres) demonstrates that the learning loop operates end-to-end, with pop reaching full confidence after 22 observations. All inference runs on-device via CoreML. To our knowledge, this is the first production deployment of MMP/SVAF on consumer mobile hardware. The accompanying SDK (sym-swift v0.3.78, SYMCore v0.3.7) enforces strict protocol conformance. Music is the case study; the substrate is the contribution. 

---
# Verify Before You Fix: Agentic Execution Grounding for Trustworthy Cross-Language Code Analysis 

**Authors**: Jugal Gajjar  

**Link**: [PDF](https://arxiv.org/pdf/2604.10800)  

**Abstract**: Learned classifiers deployed in agentic pipelines face a fundamental reliability problem: predictions are probabilistic inferences, not verified conclusions, and acting on them without grounding in observable evidence leads to compounding failures across downstream stages. Software vulnerability analysis makes this cost concrete and measurable. We address this through a unified cross-language vulnerability lifecycle framework built around three LLM-driven reasoning stages-hybrid structural-semantic detection, execution-grounded agentic validation, and validation-aware iterative repair-governed by a strict invariant: no repair action is taken without execution-based confirmation of exploitability. Cross-language generalization is achieved via a Universal Abstract Syntax Tree (uAST) normalizing Java, Python, and C++ into a shared structural schema, combined with a hybrid fusion of GraphSAGE and Qwen2.5-Coder-1.5B embeddings through learned two-way gating, whose per-sample weights provide intrinsic explainability at no additional cost. The framework achieves 89.84-92.02% intra-language detection accuracy and 74.43-80.12% zero-shot cross-language F1, resolving 69.74% of vulnerabilities end-to-end at a 12.27% total failure rate. Ablations establish necessity: removing uAST degrades cross-language F1 by 23.42%, while disabling validation increases unnecessary repairs by 131.7%. These results demonstrate that execution-grounded closed-loop reasoning is a principled and practically deployable mechanism for trustworthy LLM-driven agentic AI. 

---
# Advancing Polish Language Modeling through Tokenizer Optimization in the Bielik v3 7B and 11B Series 

**Authors**: Krzysztof Ociepa, Łukasz Flis, Remigiusz Kinas, Krzysztof Wróbel, Adrian Gwoździej  

**Link**: [PDF](https://arxiv.org/pdf/2604.10799)  

**Abstract**: The development of the Bielik v3 PL series, encompassing both the 7B and 11B parameter variants, represents a significant milestone in the field of language-specific large language model (LLM) optimization. While general-purpose models often demonstrate impressive multilingual capabilities, they frequently suffer from a fundamental architectural inefficiency: the use of universal tokenizers. These tokenizers, typically designed to cover a broad spectrum of languages, often fail to capture the morphological nuances of specific languages like Polish, leading to higher fertility ratios, increased inference costs, and restricted effective context windows. This report details the transition from the universal Mistral-based tokenization to a dedicated Polish-optimized vocabulary for the Bielik v3 models, exploring the FOCUS-based embedding initialization, the multi-stage pretraining curriculum, and the subsequent post-training alignment involving Supervised Fine-Tuning, Direct Preference Optimization, and Reinforcement Learning through Group Relative Policy Optimization with verifiable rewards. 

---
# TInR: Exploring Tool-Internalized Reasoning in Large Language Models 

**Authors**: Qiancheng Xu, Yongqi Li, Fan Liu, Hongru Wang, Min Yang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.10788)  

**Abstract**: Tool-Integrated Reasoning (TIR) has emerged as a promising direction by extending Large Language Models' (LLMs) capabilities with external tools during reasoning. Existing TIR methods typically rely on external tool documentation during reasoning. However, this leads to tool mastery difficulty, tool size constraints, and inference inefficiency. To mitigate these issues, we explore Tool-Internalized Reasoning (TInR), aiming at facilitating reasoning with tool knowledge internalized into LLMs. Achieving this goal presents notable requirements, including tool internalization and tool-reasoning coordination. To address them, we propose TInR-U, a tool-internalized reasoning framework for unified reasoning and tool usage. TInR-U is trained through a three-phase pipeline: 1) tool internalization with a bidirectional knowledge alignment strategy; 2) supervised fine-tuning warm-up using high-quality reasoning annotations, and 3) reinforcement learning with TInR-specific rewards. We comprehensively evaluate our method across in-domain and out-of-domain settings. Experiment results show that TInR-U achieves superior performance in both settings, highlighting its effectiveness and efficiency. 

---
# Do BERT Embeddings Encode Narrative Dimensions? A Token-Level Probing Analysis of Time, Space, Causality, and Character in Fiction 

**Authors**: Beicheng Bei, Hannah Hyesun Chun, Chen Guo, Arwa Saghiri  

**Link**: [PDF](https://arxiv.org/pdf/2604.10786)  

**Abstract**: Narrative understanding requires multidimensional semantic structures. This study investigates whether BERT embeddings encode dimensions of fictional narrative semantics -- time, space, causality, and character. Using an LLM to accelerate annotation, we construct a token-level dataset labeled with these four narrative categories plus "others." A linear probe on BERT embeddings (94% accuracy) significantly outperforms a control probe on variance-matched random embeddings (47%), confirming that BERT encodes meaningful narrative information. With balanced class weighting, the probe achieves a macro-average recall of 0.83, with moderate success on rare categories such as causality (recall = 0.75) and space (recall = 0.66). However, confusion matrix analysis reveals "Boundary Leakage," where rare dimensions are systematically misclassified as "others." Clustering analysis shows that unsupervised clustering aligns near-randomly with predefined categories (ARI = 0.081), suggesting that narrative dimensions are encoded but not as discretely separable clusters. Future work includes a POS-only baseline to disentangle syntactic patterns from narrative encoding, expanded datasets, and layer-wise probing. 

---
# Lung Cancer Detection Using Deep Learning 

**Authors**: Imama Ajmi, Abhishek Das  

**Link**: [PDF](https://arxiv.org/pdf/2604.10765)  

**Abstract**: Lung cancer, the second leading cause of cancer-related deaths, is primarily linked to long-term tobacco smoking (85% of cases). Surprisingly, 10-15% of cases occur in non-smokers. In 2020, approximately 2 million people were affected globally, resulting in 1.5 million deaths. The survival rate, at around 20%, lags behind other cancers, partly due to late-stage symptom manifestation. Necessitates early and accurate detection for effective treatment. Performance metrics such as accuracy, precision, recall (sensitivity), and F1-score are computed to provide a comprehensive evaluation of each model's capabilities. By comparing these metrics, this study offers insights into the strengths and limitations of each approach, contributing to the advancement of lung cancer detection techniques. In this paper, we are going to discuss the methodologies of lung cancer detection using different deep learning algorithms - InceptionV3, MobileNetV2, VGG16, ResNet152 - are explored for their efficacy in classifying lung cancer cases. Our Proposed Model algorithm based is a 16 layers architecture based on CNN model. Our Proposed model exhibits several key highlights that contribute to its novelty. By integrating multiple layer types such as convolutional, pooling, flatten, dropout, fully connected and dense layers, the model leverages the strengths of each layer to enhance its predictive capabilities. Novelty of our proposed model is that its accuracy is increasing consistently with the increasing no of epochs. We have tested the model performance up to epoch no 30. Our proposed model also overcome the overfitting problem. 

---
# Prosociality by Coupling, Not Mere Observation: Homeostatic Sharing in an Inspectable Recurrent Artificial Life Agent 

**Authors**: Aishik Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2604.10760)  

**Abstract**: Artificial agents can be made to "help" for many reasons, including explicit social reward, hard-coded prosocial bonuses, or direct access to another agent's internal state. Those possibilities make minimal prosocial behavior hard to interpret. Building on ReCoN-Ipsundrum, an inspectable recurrent controller with affect-coupled regulation, I add an explicit homeostat and a social coupling channel while keeping planning strictly self-directed: the agent scores only its own predicted internal state, and no partner-welfare reward term is introduced. I compare four matched conditions in two toy worlds. In a one-step FoodShareToy, an exact solver finds a sharp switch from EAT to PASS at ${\lambda}* \approx 0.91$ for the default state. In the experimental runs, the self-only and partner-observing conditions never help, whereas the affectively coupled conditions always do. In a multi-step SocialCorridorWorld, the same dissociation reappears: coupling flips help rate and partner recovery from 0 to 1 and cuts rescue latency from 18 to 9 steps, while raising mutual viability from 0.15 to 0.33. Sham lesions preserve helping, but coupling-off and shuffled-partner lesions abolish it in both tasks. A coupling sweep shows a load-dependent feasibility boundary: under low load, helping appears for ${\lambda} \geq 0.25$, whereas under medium and high loads no tested value rescues the partner within horizon. The result is a narrow claim for artificial life: in this minimal architecture, helping appears when another's need is routed into self-regulation. 

---
# Generating Multiple-Choice Knowledge Questions with Interpretable Difficulty Estimation using Knowledge Graphs and Large Language Models 

**Authors**: Mehmet Can Şakiroğlu, H. Altay Güvenir, Kamer Kaya  

**Link**: [PDF](https://arxiv.org/pdf/2604.10748)  

**Abstract**: Generating multiple-choice questions (MCQs) with difficulty estimation remains challenging in automated MCQ-generation systems used in adaptive, AI-assisted education. This study proposes a novel methodology for generating MCQs with difficulty estimation from the input documents by utilizing knowledge graphs (KGs) and large language models (LLMs). Our approach uses an LLM to construct a KG from input documents, from which MCQs are then systematically generated. Each MCQ is generated by selecting a node from the KG as the key, sampling a related triple or quintuple -- optionally augmented with an extra triple -- and prompting an LLM to generate a corresponding stem from these graph components. Distractors are then selected from the KG. For each MCQ, nine difficulty signals are computed and combined into a unified difficulty score using a data-driven approach. Experimental results demonstrate that our method generates high-quality MCQs whose difficulty estimation is interpretable and aligns with human perceptions. Our approach improves automated MCQ generation by integrating structured knowledge representations with LLMs and a data-driven difficulty estimation model. 

---
# Deep-Reporter: Deep Research for Grounded Multimodal Long-Form Generation 

**Authors**: Fangda Ye, Zhifei Xie, Yuxin Hu, Yihang Yin, Shurui Huang, Shikai Dong, Jianzhu Bao, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.10741)  

**Abstract**: Recent agentic search frameworks enable deep research via iterative planning and retrieval, reducing hallucinations and enhancing factual grounding. However, they remain text-centric, overlooking the multimodal evidence that characterizes real-world expert reports. We introduce a pressing task: multimodal long-form generation. Accordingly, we propose Deep-Reporter, a unified agentic framework for grounded multimodal long-form generation. It orchestrates: (i) Agentic Multimodal Search and Filtering to retrieve and filter textual passages and information-dense visuals; (ii) Checklist-Guided Incremental Synthesis to ensure coherent image-text integration and optimal citation placement; and (iii) Recurrent Context Management to balance long-range coherence with local fluency. We develop a rigorous curation pipeline producing 8K high-quality agentic traces for model optimization. We further introduce M2LongBench, a comprehensive testbed comprising 247 research tasks across 9 domains and a stable multimodal sandbox. Extensive experiments demonstrate that long-form multimodal generation is a challenging task, especially in multimodal selection and integration, and effective post-training can bridge the gap. 

---
# Too Nice to Tell the Truth: Quantifying Agreeableness-Driven Sycophancy in Role-Playing Language Models 

**Authors**: Arya Shah, Deepali Mishra, Chaklam Silpasuwanchai  

**Link**: [PDF](https://arxiv.org/pdf/2604.10733)  

**Abstract**: Large language models increasingly serve as conversational agents that adopt personas and role-play characters at user request. This capability, while valuable, raises concerns about sycophancy: the tendency to provide responses that validate users rather than prioritize factual accuracy. While prior work has established that sycophancy poses risks to AI safety and alignment, the relationship between specific personality traits of adopted personas and the degree of sycophantic behavior remains unexplored. We present a systematic investigation of how persona agreeableness influences sycophancy across 13 small, open-weight language models ranging from 0.6B to 20B parameters. We develop a benchmark comprising 275 personas evaluated on NEO-IPIP agreeableness subscales and expose each persona to 4,950 sycophancy-eliciting prompts spanning 33 topic categories. Our analysis reveals that 9 of 13 models exhibit statistically significant positive correlations between persona agreeableness and sycophancy rates, with Pearson correlations reaching $r = 0.87$ and effect sizes as large as Cohen's $d = 2.33$. These findings demonstrate that agreeableness functions as a reliable predictor of persona-induced sycophancy, with direct implications for the deployment of role-playing AI systems and the development of alignment strategies that account for personality-mediated deceptive behaviors. 

---
# Perceived Importance of Cognitive Skills Among Computing Students in the Era of AI 

**Authors**: Neha Rani, Erta Cenko, Laura Melissa Cruz Castro  

**Link**: [PDF](https://arxiv.org/pdf/2604.10730)  

**Abstract**: The availability and increasing integration of generative AI tools have transformed computing education. While AI in education presents opportunities, it also raises new concerns about how these powerful know-it-all AI tools, which are becoming widespread, impact cognitive skill development among students. Cognitive skills are essential for academic success and professional competence. It relates to the ability to understand, analyze, evaluate, synthesize information and more. The extensive use of these AI tools can aid in cognitive offloading, freeing up cognitive resources to be used in other tasks and activities. However, cognitive offloading may inadvertently lead to diminishing cognitive involvement in learning and related activities when using AI tools. Understanding cognitive skills' impact in the era of AI is essential to align curricular design with evolving workforce demands and changing work environment and processes. To address this concern and to develop an understanding of how the importance of cognitive skills changes with increasing integration of AI, we conducted a researcher-monitored and regulated quantitative survey of undergraduate computing students. We examined students' perceptions of cognitive skills across three temporal frames: prior to widespread AI adoption (past), current informal and formal use of AI in learning contexts (present), and future with even more AI integration in professional environments (future). In the study, students rated the importance of 11 cognitive skills. Our analysis reveals that students expect all 11 cognitive skills to be of diminishing importance in the future, when AI use and integration increases. Our findings highlight the need for educational interventions that explicitly reinforce cognitive skill development within learning environments that are now often relying on AI. 

---
# Tail-Aware Information-Theoretic Generalization for RLHF and SGLD 

**Authors**: Huiming Zhang, Binghan Li, Wan Tian, Qiang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.10727)  

**Abstract**: Classical information-theoretic generalization bounds typically control the generalization gap through KL-based mutual information and therefore rely on boundedness or sub-Gaussian tails via the moment generating function (MGF). In many modern pipelines, such as robust learning, RLHF, and stochastic optimization, losses and rewards can be heavy-tailed, and MGFs may not exist, rendering KL-based tools ineffective. We develop a tail-dependent information-theoretic framework for sub-Weibull data, where the tail parameter $\theta$ controls the tail heaviness: $\theta=2$ corresponds to sub-Gaussian, $\theta=1$ to sub-exponential, and $0<\theta<1$ to genuinely heavy tails. Our key technical ingredient is a decorrelation lemma that bounds change-of-measure expectations using a shifted-log $f_\theta$-divergence, which admits explicit comparisons to Rényi divergence without MGF arguments. On the empirical-process side, we establish sharp maximal inequalities and a Dudley-type chaining bound for sub-Weibull processes with tail index $\theta$, with complexity scaling as $\log^{1/\theta}$ and entropy$^{1/\theta}$. These tools yield expected and high-probability PAC-Bayes generalization bounds, as well as an information-theoretic chaining inequality based on multiscale Rényi mutual information. We illustrate the consequences in Rényi-regularized RLHF under heavy-tailed rewards and in stochastic gradient Langevin dynamics with heavy-tailed gradient noise. 

---
# Turning Generators into Retrievers: Unlocking MLLMs for Natural Language-Guided Geo-Localization 

**Authors**: Yuqi Chen, Xiaohan Zhang, Ahmad Arrabi, Waqas Sultani, Chen Chen, Safwan Wshah  

**Link**: [PDF](https://arxiv.org/pdf/2604.10721)  

**Abstract**: Natural-language Guided Cross-view Geo-localization (NGCG) aims to retrieve geo-tagged satellite imagery using textual descriptions of ground scenes. While recent NGCG methods commonly rely on CLIP-style dual-encoder architectures, they often suffer from weak cross-modal generalization and require complex architectural designs. In contrast, Multimodal Large Language Models (MLLMs) offer powerful semantic reasoning capabilities but are not directly optimized for retrieval tasks. In this work, we present a simple yet effective framework to adapt MLLMs for NGCG via parameter-efficient finetuning. Our approach optimizes latent representations within the MLLM while preserving its pretrained multimodal knowledge, enabling strong cross-modal alignment without redesigning model architectures. Through systematic analysis of diverse variables, from model backbone to feature aggregation, we provide practical and generalizable insights for leveraging MLLMs in NGCG. Our method achieves SOTA on GeoText-1652 with a 12.2% improvement in Text-to-Image Recall@1 and secures top performance in 5 out of 12 subtasks on CVG-Text, all while surpassing baselines with far fewer trainable parameters. These results position MLLMs as a robust foundation for semantic cross-view retrieval and pave the way for MLLM-based NGCG to be adopted as a scalable, powerful alternative to traditional dual-encoder designs. Project page and code are available at this https URL. 

---
# Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game 

**Authors**: Yuanbo Xie, Yingjie Zhang, Yulin Li, Shouyou Song, Xiaokun Chen, Zhihan Liu, Liya Su, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10717)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems augment large language models with external knowledge, yet introduce a critical security vulnerability: RAG Knowledge Base Leakage, wherein adversarial prompts can induce the model to divulge retrieved proprietary content. Recent studies reveal that such leakage can be executed through adaptive and iterative attack strategies (named RAG extraction attack), while effective countermeasures remain notably lacking. To bridge this gap, we propose CanaryRAG, a runtime defense mechanism inspired by stack canaries in software security. CanaryRAG embeds carefully designed canary tokens into retrieved chunks and reformulates RAG extraction defense as a dual-path runtime integrity game. Leakage is detected in real time whenever either the target or oracle path violates its expected canary behavior, including under adaptive suppression and obfuscation. Extensive evaluations against existing attacks demonstrate that CanaryRAG provides robust defense, achieving substantially lower chunk recovery rates than state-of-the-art baselines while imposing negligible impact on task performance and inference latency. Moreover, as a plug-and-play solution, CanaryRAG can be seamlessly integrated into arbitrary RAG pipelines without requiring retraining or structural modifications, offering a practical and scalable safeguard for proprietary data. 

---
# Audio-Omni: Extending Multi-modal Understanding to Versatile Audio Generation and Editing 

**Authors**: Zeyue Tian, Binxin Yang, Zhaoyang Liu, Jiexuan Zhang, Ruibin Yuan, Hubery Yin, Qifeng Chen, Chen Li, Jing Lv, Wei Xue, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.10708)  

**Abstract**: Recent progress in multimodal models has spurred rapid advances in audio understanding, generation, and editing. However, these capabilities are typically addressed by specialized models, leaving the development of a truly unified framework that can seamlessly integrate all three tasks underexplored. While some pioneering works have explored unifying audio understanding and generation, they often remain confined to specific domains. To address this, we introduce Audio-Omni, the first end-to-end framework to unify generation and editing across general sound, music, and speech domains, with integrated multi-modal understanding capabilities. Our architecture synergizes a frozen Multimodal Large Language Model for high-level reasoning with a trainable Diffusion Transformer for high-fidelity synthesis. To overcome the critical data scarcity in audio editing, we construct AudioEdit, a new large-scale dataset comprising over one million meticulously curated editing pairs. Extensive experiments demonstrate that Audio-Omni achieves state-of-the-art performance across a suite of benchmarks, outperforming prior unified approaches while achieving performance on par with or superior to specialized expert models. Beyond its core capabilities, Audio-Omni exhibits remarkable inherited capabilities, including knowledge-augmented reasoning generation, in-context generation, and zero-shot cross-lingual control for audio generation, highlighting a promising direction toward universal generative audio intelligence. The code, model, and dataset will be publicly released on this https URL. 

---
# Architecture-Agnostic Modality-Isolated Gated Fusion for Robust Multi-Modal Prostate MRI Segmentation 

**Authors**: Yongbo Shu, Wenzhao Xie, Shanhu Yao, Zirui Xin, Luo Lei, Kewen Chen, Aijing Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.10702)  

**Abstract**: Multi-parametric prostate MRI -- combining T2-weighted, apparent diffusion coefficient, and high b-value diffusion-weighted sequences -- is central to non-invasive detection of clinically significant prostate cancer, yet in routine practice individual sequences may be missing or degraded by motion, artifacts, or abbreviated protocols. Existing multi-modal fusion strategies typically assume complete inputs and entangle modality-specific information at early layers, offering limited resilience when one channel is corrupted or absent. We propose Modality-Isolated Gated Fusion (MIGF), an architecture-agnostic module that maintains separate modality-specific encoding streams before a learned gating stage, combined with modality dropout training to enforce compensation behavior under incomplete inputs. We benchmark six bare backbones and assess MIGF-equipped models under seven missing-modality and artifact scenarios on the PI-CAI dataset (1,500 studies, fold-0 split, five random seeds). Among bare backbones, nnUNet provided the strongest balance of performance and stability. MIGF improved ideal-scenario Ranking Score for UNet, nnUNet, and Mamba by 2.8%, 4.6%, and 13.4%, respectively; the best model, MIGFNet-nnUNet (gating + ModDrop, no deep supervision), achieved 0.7304 +/- 0.056. Mechanistic analysis reveals that robustness gains arise from strict modality isolation and dropout-driven compensation rather than adaptive per-sample quality routing: the gate converged to a stable modality prior, and deep supervision was beneficial only for the largest backbone while degrading lighter models. These findings support a simpler design principle for robust multi-modal segmentation: structurally contain corrupted inputs first, then train explicitly for incomplete-input compensation. 

---
# Bringing Value Models Back: Generative Critics for Value Modeling in LLM Reinforcement Learning 

**Authors**: Zikang Shan, Han Zhong, Liwei Wang, Li Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.10701)  

**Abstract**: Credit assignment is a central challenge in reinforcement learning (RL). Classical actor-critic methods address this challenge through fine-grained advantage estimation based on a learned value function. However, learned value models are often avoided in modern large language model (LLM) RL because conventional discriminative critics are difficult to train reliably. We revisit value modeling and argue that this difficulty is partly due to limited expressiveness. In particular, representation complexity theory suggests that value functions can be hard to approximate under the one-shot prediction paradigm used by existing value models, and our scaling experiments show that such critics do not improve reliably with scale. Motivated by this observation, we propose Generative Actor-Critic (GenAC), which replaces one-shot scalar value prediction with a generative critic that performs chain-of-thought reasoning before producing a value estimate. We further introduce In-Context Conditioning, which helps the critic remain calibrated to the current actor throughout training. GenAC improves value approximation, ranking reliability, and out-of-distribution generalization, and these gains translate into stronger downstream RL performance than both value-based and value-free baselines. Overall, our results suggest that stronger value modeling is a promising direction for improving credit assignment in LLM reinforcement learning. 

---
# SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting 

**Authors**: Binbin Zheng, Xing Ma, Yiheng Liang, Jingqing Ruan, Xiaoliang Fu, Kepeng Lin, Benchang Zhu, Ke Zeng, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.10688)  

**Abstract**: On-policy reinforcement learning has become the dominant paradigm for reasoning alignment in large language models, yet its sparse, outcome-level rewards make token-level credit assignment notoriously difficult. On-Policy Distillation (OPD) alleviates this by introducing dense, token-level KL supervision from a teacher model, but typically applies this supervision uniformly across all rollouts, ignoring fundamental differences in signal quality. We propose Signal-Calibrated On-Policy Distillation Enhancement (SCOPE), a dual-path adaptive training framework that routes on-policy rollouts by correctness into two complementary supervision paths. For incorrect trajectories, SCOPE performs teacher-perplexity-weighted KL distillation to prioritize instances where the teacher demonstrates genuine corrective capability, while down-weighting unreliable guidance. For correct trajectories, it applies student-perplexity-weighted MLE to concentrate reinforcement on low-confidence samples at the capability boundary rather than over-reinforcing already mastered ones. Both paths employ a group-level normalization to adaptively calibrate weight distributions, accounting for the intrinsic difficulty variance across prompts. Extensive experiments on six reasoning benchmarks show that SCOPE achieves an average relative improvement of 11.42% in Avg@32 and 7.30% in Pass@32 over competitive baselines, demonstrating its consistent effectiveness. 

---
# Critical-CoT: A Robust Defense Framework against Reasoning-Level Backdoor Attacks in Large Language Models 

**Authors**: Vu Tuan Truong, Long Bao Le  

**Link**: [PDF](https://arxiv.org/pdf/2604.10681)  

**Abstract**: Large Language Models (LLMs), despite their impressive capabilities across domains, have been shown to be vulnerable to backdoor attacks. Prior backdoor strategies predominantly operate at the token level, where an injected trigger causes the model to generate a specific target word, choice, or class (depending on the task). Recent advances, however, exploit the long-form reasoning tendencies of modern LLMs to conduct reasoning-level backdoors: once triggered, the victim model inserts one or more malicious reasoning steps into its chain-of-thought (CoT). These attacks are substantially harder to detect, as the backdoored answer remains plausible and consistent with the poisoned reasoning trajectory. Yet, defenses tailored to this type of backdoor remain largely unexplored. To bridge this gap, we propose Critical-CoT, a novel defense mechanism that conducts a two-stage fine-tuning (FT) process on LLMs to develop critical thinking behaviors, enabling them to automatically identify potential backdoors and refuse to generate malicious reasoning steps. Extensive experiments across multiple LLMs and datasets demonstrate that Critical-CoT provides strong robustness against both in-context learning-based and FT-based backdoor attacks. Notably, Critical-CoT exhibits strong cross-domain and cross-task generalization. Our code is available at hthttps://github.com/tuanvu171/Critical-CoT. 

---
# Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents 

**Authors**: Hao Wang, Guozhi Wang, Han Xiao, Yufeng Zhou, Yue Pan, Jichao Wang, Ke Xu, Yafei Wen, Xiaohu Ruan, Xiaoxin Chen, Honggang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2604.10674)  

**Abstract**: Reinforcement learning (RL) has been widely used to train LLM agents for multi-turn interactive tasks, but its sample efficiency is severely limited by sparse rewards and long horizons. On-policy self-distillation (OPSD) alleviates this by providing dense token-level supervision from a privileged teacher that has access to ground-truth answers. However, such fixed privileged information cannot capture the diverse valid strategies in agent tasks, and naively combining OPSD with RL often leads to training collapse. To address these limitations, we introduce Skill-SD, a framework that turns the agent's own trajectories into dynamic training-only supervision. Completed trajectories are summarized into compact natural language skills that describe successful behaviors, mistakes, and workflows. These skills serve as dynamic privileged information conditioning only the teacher, while the student always acts under the plain task prompt and learns to internalize the guidance through distillation. To stabilize the training, we derive an importance-weighted reverse-KL loss to provide gradient-correct token-level distillation, and dynamically synchronize the teacher with the improving student. Experimental results on agentic benchmarks demonstrate that Skill-SD substantially outperforms the standard RL baseline, improving both vanilla GRPO (+14.0%/+10.9% on AppWorld/Sokoban) and vanilla OPD (+42.1%/+40.6%). Project page: this https URL 

---
# Learning and Enforcing Context-Sensitive Control for LLMs 

**Authors**: Mohammad Albinhassan, Pranava Madhyastha, Mark Law, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2604.10667)  

**Abstract**: Controlling the output of Large Language Models (LLMs) through context-sensitive constraints has emerged as a promising approach to overcome the limitations of Context-Free Grammars (CFGs) in guaranteeing generation validity. However, such constraints typically require manual specification -- a significant barrier demanding specialized expertise. We introduce a framework that automatically learns context-sensitive constraints from LLM interactions through a two-phase process: syntactic exploration to gather diverse outputs for constraint learning, followed by constraint exploitation to enforce these learned rules during generation. Experiments demonstrate that our method enables even small LLMs (1B parameters) to learn and generate with perfect constraint adherence, outperforming larger counterparts and state-of-the-art reasoning models. This work represents the first integration of context-sensitive grammar learning with LLM generation, eliminating manual specification while maintaining generation validity. 

---
# DynamicsLLM: a Dynamic Analysis-based Tool for Generating Intelligent Execution Traces Using LLMs to Detect Android Behavioural Code Smells 

**Authors**: Houcine Abdelkader Cherief, Florent Avellaneda, Naouel Moha  

**Link**: [PDF](https://arxiv.org/pdf/2604.10661)  

**Abstract**: Mobile apps have become essential of our daily lives, making code quality a critical concern for developers. Behavioural code smells are characteristics in the source code that induce inappropriate code behaviour during execution, which negatively impact software quality in terms of performance, energy consumption, and memory. Dynamics, the latest state-of-the-art tool-based method, is highly effective at detecting Android behavioural code smells. While it outperforms static analysis tools, it suffers from a high false negative rate, with multiple code smell instances remaining undetected. Large Language Models (LLMs) have achieved notable advances across numerous research domains and offer significant potential for generating intelligent execution traces, particularly for detecting behavioural code smells in Android mobile applications. By intelligent execution trace, we mean a sequence of events generated by specific actions in a way that triggers the identification of a given behaviour. We propose the following three main contributions in this paper: (1) DynamicsLLM, an enhanced implementation of the Dynamics method that leverages LLMs to intelligently generate execution traces. (2) A novel hybrid approach designed to improve the coverage of code smell-related events in applications with a small number of activities. (3) A comprehensive validation of DynamicsLLM on 333 mobile applications from F-DROID, including a comparison with the Dynamics tool. Our results show that, under a limited number of actions, DynamicsLLM configured with 100% LLM covers three times more code smell-related events than Dynamics. The hybrid approach improves LLM coverage by 25.9% for apps containing few activities. Moreover, 12.7% of the code smell-related events that cannot be triggered by Dynamics are successfully triggered by our tool. 

---
# Efficient Process Reward Modeling via Contrastive Mutual Information 

**Authors**: Nakyung Lee, Sangwoo Hong, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.10660)  

**Abstract**: Recent research has devoted considerable effort to verifying the intermediate reasoning steps of chain-of-thought (CoT) trajectories using process reward models (PRMs) and other verifier models. However, training a PRM typically requires human annotators to assign reward scores to each reasoning step, which is both costly and time-consuming. Existing automated approaches, such as Monte Carlo (MC) estimation, also demand substantial computational resources due to repeated LLM rollouts. To overcome these limitations, we propose contrastive pointwise mutual information (CPMI), a novel automatic reward labeling method that leverages the model's internal probability to infer step-level supervision while significantly reducing the computational burden of annotating dataset. CPMI quantifies how much a reasoning step increases the mutual information between the step and the correct target answer relative to hard-negative alternatives. This contrastive signal serves as a proxy for the step's contribution to the final solution and yields a reliable reward. The experimental results show that CPMI-based labeling reduces dataset construction time by 84% and token generation by 98% compared to MC estimation, while achieving higher accuracy on process-level evaluations and mathematical reasoning benchmarks. 

---
# LoViF 2026 The First Challenge on Weather Removal in Videos 

**Authors**: Chenghao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2604.10655)  

**Abstract**: This paper presents a review of the LoViF 2026 Challenge on Weather Removal in Videos. The challenge encourages the development of methods for restoring clean videos from inputs degraded by adverse weather conditions such as rain and snow, with an emphasis on achieving visually plausible and temporally consistent results while preserving scene structure and motion dynamics. To support this task, we introduce a new short-form WRV dataset tailored for video weather removal. It consists of 18 videos 1,216 synthesized frames paired with 1,216 real-world ground-truth frames at a resolution of 832 x 480, and is split into training, validation, and test sets with a ratio of 1:1:1. The goal of this challenge is to advance robust and realistic video restoration under real-world weather conditions, with evaluation protocols that jointly consider fidelity and perceptual quality. The challenge attracted 37 participants and received 5 valid final submissions with corresponding fact sheets, contributing to progress in weather removal for videos. The project is publicly available at this https URL. 

---
# Vibe-driven model-based engineering 

**Authors**: Jordi Cabot  

**Link**: [PDF](https://arxiv.org/pdf/2604.10645)  

**Abstract**: There is a pressing need for better development methods and tools to keep up with the growing demand and increasing complexity of new software systems. New types of user interfaces, the need for intelligent components, sustainability concerns, etc. bring new challenges that we need to handle. In the last years, model-driven engineering (MDE), including its latest incarnation, i.e. low/no-code development, has been key to improving the quality and productivity of software development, but models themselves are becoming increasingly complex to specify and manage. At the same time, we are witnessing the growing popularity of vibe coding approaches that rely on Large Language Models (LLMs) to transform natural language descriptions into running code at the expense of potential code vulnerabilities, scalability issues and maintainability concerns.
While many may think vibe coding will replace model-based engineering, in this paper we argue that, in fact, the two approaches can complement each other and provide altogether different development paths for different types of software systems, development scenarios, and user profiles. In this sense, we introduce the concept of \textit{vibe-driven model-based engineering} as a novel approach to integrate the best of both worlds (AI and MDE) to accelerate the development of reliable complex systems. We outline the key concepts of this new approach and highlight the opportunities and open challenges it presents for the future of software development. 

---
# Computational Lesions in Multilingual Language Models Separate Shared and Language-specific Brain Alignment 

**Authors**: Yang Cui, Jingyuan Sun, Yizheng Sun, Yifan Wang, Yunhao Zhang, Jixing Li, Shaonan Wang, Hongpeng Zhou, John Hale, Chengqing Zong, Goran Nenadic  

**Link**: [PDF](https://arxiv.org/pdf/2604.10627)  

**Abstract**: How the brain supports language across different languages is a basic question in neuroscience and a useful test for multilingual artificial intelligence. Neuroimaging has identified language-responsive brain regions across languages, but it cannot by itself show whether the underlying processing is shared or language-specific. Here we use six multilingual large language models (LLMs) as controllable systems and create targeted ``computational lesions'' by zeroing small parameter sets that are important across languages or especially important for one language. We then compare intact and lesioned models in predicting functional magnetic resonance imaging (fMRI) responses during 100 minutes of naturalistic story listening in native English, Chinese and French (112 participants). Lesioning a compact shared core reduces whole-brain encoding correlation by 60.32% relative to intact models, whereas language-specific lesions preserve cross-language separation in embedding space but selectively weaken brain predictivity for the matched native language. These results support a shared backbone with embedded specializations and provide a causal framework for studying multilingual brain-model alignment. 

---
# NSFL: A Post-Training Neuro-Symbolic Fuzzy Logic Framework for Boolean Operators in Neural Embeddings 

**Authors**: Vladi Vexler, Ofer Idan, Gil Lederman, Dima Sivov  

**Link**: [PDF](https://arxiv.org/pdf/2604.10604)  

**Abstract**: Standard dense retrievers lack a native calculus for multi-atom logical constraints. We introduce Neuro-Symbolic Fuzzy Logic (NSFL), a framework that adapts formal t-norms and t-conorms to neural embedding spaces without requiring retraining. NSFL operates as a first-order hybrid calculus: it anchors logical operations on isolated zero-order similarity scores while actively steering representations using Neuro-Symbolic Deltas (NS-Delta) -- the first-order marginal differences derived from contextual fusion. This preserves pure atomic meaning while capturing domain reliance, preventing the representation collapse and manifold escape endemic to traditional geometric baselines. For scalable real-time retrieval, Spherical Query Optimization (SQO) leverages Riemannian optimization to project these fuzzy formulas into manifold-stable query vectors. Validated across six distinct encoder configurations and two modalities (including zero-shot and SOTA fine-tuned models), NSFL yields mAP improvements up to +81%. Notably, NSFL provides an additive 20% average and up to 47% boost even when applied to encoders explicitly fine-tuned for logical reasoning. By establishing a training-free, order-aware calculus for high-dimensional spaces, this framework lays the foundation for future dynamic scaling and learned manifold logic. 

---
# MoEITS: A Green AI approach for simplifying MoE-LLMs 

**Authors**: Luis Balderas, Miguel Lastra, José M. Benítez  

**Link**: [PDF](https://arxiv.org/pdf/2604.10603)  

**Abstract**: Large language models are transforming all areas of academia and industry, attracting the attention of researchers, professionals, and the general public. In the trek for more powerful architectures, Mixture-of-Experts, inspired by ensemble models, have emerged as one of the most effective ways to follow. However, this implies a high computational burden for both training and inference. To reduce the impact on computing and memory footprint as well as the energy consumption, simplification methods has arisen as very effective procedures.
In this paper, an original algorithm, MoEITS, for MoE-LLMs simplification is presented. The algorithm is characterized by a refined simplicity, underpinned by standardized Information Theoretic frameworks. MoEITS is analyzed in depth from theoretical and practical points of view. Its computational complexity is studied. Its performance on the accuracy of the simplified LLMs and the reduction rate achieved is assessed through a thoroughly designed experimentation. This empirical evaluation includes a comparison with state-of-the-art MoE-LLM pruning methods applied on Mixtral $8\times7$B, Qwen1.5-2.7B, and DeepSeek-V2-Lite. The extensive experimentation conducted demonstrates that MoEITS outperforms state-of-the-art techniques by generating models that are both effective across all benchmarks and computationally efficient.
The code implementing the method will be available at this https URL. 

---
# COREY: A Prototype Study of Entropy-Guided Operator Fusion with Hadamard Reparameterization for Selective State Space Models 

**Authors**: Bo Ma, Jinsong Wu, Hongjiang Wei, Weiqi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.10597)  

**Abstract**: State Space Models (SSMs), represented by the Mamba family, provide linear-time sequence modeling and are attractive for long-context inference. Yet practical deployments remain memory-bandwidth limited because selective state updates are often decomposed into fragmented kernels with repeated intermediate tensor materialization. We present COREY, a prototype framework that combines memory-aware operator fusion with Hadamard-based feature reparameterization. Activation entropy, estimated with fixed-width histograms, is used as a runtime scheduling statistic to place fusion boundaries and choose tile sizes. To regularize heavy-tailed activations, we absorb normalized Hadamard transforms into linear projections, preserving functional equivalence while reducing peak-coordinate concentration. In a controlled prototype study over heavy-tailed SSM activations, COREY consistently reduces proxy latency, improves throughput, and lowers DRAM traffic relative to unfused and fixed-depth baselines. Low-bit results are reported only through a hand-crafted stability proxy and are intended as diagnostic evidence rather than checkpoint-level quality claims. Code repository: this https URL. 

---
# GeoMeld: Toward Semantically Grounded Foundation Models for Remote Sensing 

**Authors**: Maram Hasan, Md Aminur Hossain, Savitra Roy, Souparna Bhowmik, Ayush V. Patel, Mainak Singha, Subhasis Chaudhuri, Muhammad Haris Khan, Biplab Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2604.10591)  

**Abstract**: Effective foundation modeling in remote sensing requires spatially aligned heterogeneous modalities coupled with semantically grounded supervision, yet such resources remain limited at scale. We present GeoMeld, a large-scale multimodal dataset with approximately 2.5 million spatially aligned samples. The dataset spans diverse modalities and resolutions and is constructed under a unified alignment protocol for modality-aware representation learning. GeoMeld provides semantically grounded language supervision through an agentic captioning framework that synthesizes and verifies annotations from spectral signals, terrain statistics, and structured geographic metadata, encoding measurable cross-modality relationships within textual descriptions. To leverage this dataset, we introduce GeoMeld-FM, a pretraining framework that combines multi-pretext masked autoencoding over aligned modalities, JEPA representation learning, and caption-vision contrastive alignment. This joint objective enables the learned representation space to capture both reliable cross-sensor physical consistency and grounded semantics. Experiments demonstrate consistent gains in downstream transfer and cross-sensor robustness. Together, GeoMeld and GeoMeld-FM establish a scalable reference framework for semantically grounded multi-modal foundation modeling in remote sensing. 

---
# Bridging Linguistic Gaps: Cross-Lingual Mapping in Pre-Training and Dataset for Enhanced Multilingual LLM Performance 

**Authors**: Weihua Zheng, Chang Liu, Zhengyuan Liu, Xin Huang, Kui Wu, Muhammad Huzaifah Md Shahrin, Aiti Aw, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.10590)  

**Abstract**: Multilingual Large Language Models (LLMs) struggle with cross-lingual tasks due to data imbalances between high-resource and low-resource languages, as well as monolingual bias in pre-training. Existing methods, such as bilingual fine-tuning and contrastive alignment, can improve cross-lingual performance, but they often require extensive parallel data or suffer from instability. To address these challenges, we introduce a Cross-Lingual Mapping Task during the pre-training phase, which enhances cross-lingual alignment without compromising monolingual fluency. Our approach bi-directionally maps languages within the LLM embedding space, improving both language generation and comprehension. We further propose a Language Alignment Coefficient to robustly quantify cross-lingual consistency, even in limited-data scenarios. Experimental results on machine translation (MT), cross-lingual natural language understanding (CLNLU), and cross-lingual question answering (CLQA) show that our model achieves gains of up to 11.9 BLEU points in MT, 6.72 points in CLQA BERTScore-Precision, and more than 5% in CLNLU accuracy over strong multilingual baselines. These findings highlight the potential of incorporating cross-lingual objectives into pre-training to improve multilingual LLMs. 

---
# Calibration Collapse Under Sycophancy Fine-Tuning: How Reward Hacking Breaks Uncertainty Quantification in LLMs 

**Authors**: Subramanyam Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2604.10585)  

**Abstract**: Modern large language models (LLMs) are increasingly fine-tuned via reinforcement learning from human feedback (RLHF) or related reward optimisation schemes. While such procedures improve perceived helpfulness, we investigate whether sycophantic reward signals degrade calibration -- a property essential for reliable uncertainty quantification. We fine-tune Qwen3-8B under three regimes: no fine-tuning (base), neutral supervised fine-tuning (SFT) on TriviaQA, and sycophancy-inducing Group Relative Policy Optimisation (GRPO) that rewards agreement with planted wrong answers. Evaluating on $1{,}000$ MMLU items across five subject domains with bootstrap confidence intervals and permutation testing, we find that \textbf{sycophantic GRPO produces consistent directional calibration degradation} -- ECE rises by $+0.006$ relative to the base model and MCE increases by $+0.010$ relative to neutral SFT -- though the effect does not reach statistical significance ($p = 0.41$) at this training budget. Post-hoc matrix scaling applied to all three models reduces ECE by $40$--$64\%$ and improves accuracy by $1.5$--$3.0$ percentage points. However, the sycophantic model retains the highest post-scaling ECE relative to the neutral SFT control ($0.042$ vs.\ $0.037$), suggesting that reward-induced miscalibration leaves a structured residual even after affine correction. These findings establish a methodology for evaluating the calibration impact of reward hacking and motivate calibration-aware training objectives. 

---
# AffordGen: Generating Diverse Demonstrations for Generalizable Object Manipulation with Afford Correspondence 

**Authors**: Jiawei Zhang, Kaizhe Hu, Yingqian Huang, Yuanchen Ju, Zhengrong Xue, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10579)  

**Abstract**: Despite the recent success of modern imitation learning methods in robot manipulation, their performance is often constrained by geometric variations due to limited data diversity. Leveraging powerful 3D generative models and vision foundation models (VFMs), the proposed AffordGen framework overcomes this limitation by utilizing the semantic correspondence of meaningful keypoints across large-scale 3D meshes to generate new robot manipulation trajectories. This large-scale, affordance-aware dataset is then used to train a robust, closed-loop visuomotor policy, combining the semantic generalizability of affordances with the reactive robustness of end-to-end learning. Experiments in simulation and the real world show that policies trained with AffordGen achieve high success rates and enable zero-shot generalization to truly unseen objects, significantly improving data efficiency in robot learning. 

---
# The Blind Spot of Agent Safety: How Benign User Instructions Expose Critical Vulnerabilities in Computer-Use Agents 

**Authors**: Xuwei Ding, Skylar Zhai, Linxin Song, Jiate Li, Taiwei Shi, Nicholas Meade, Siva Reddy, Jian Kang, Jieyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.10577)  

**Abstract**: Computer-use agents (CUAs) can now autonomously complete complex tasks in real digital environments, but when misled, they can also be used to automate harmful actions programmatically. Existing safety evaluations largely target explicit threats such as misuse and prompt injection, but overlook a subtle yet critical setting where user instructions are entirely benign and harm arises from the task context or execution outcome. We introduce OS-BLIND, a benchmark that evaluates CUAs under unintended attack conditions, comprising 300 human-crafted tasks across 12 categories, 8 applications, and 2 threat clusters: environment-embedded threats and agent-initiated harms. Our evaluation on frontier models and agentic frameworks reveals that most CUAs exceed 90% attack success rate (ASR), and even the safety-aligned Claude 4.5 Sonnet reaches 73.0% ASR. More interestingly, this vulnerability becomes even more severe, with ASR rising from 73.0% to 92.7% when Claude 4.5 Sonnet is deployed in multi-agent systems. Our analysis further shows that existing safety defenses provide limited protection when user instructions are benign. Safety alignment primarily activates within the first few steps and rarely re-engages during subsequent execution. In multi-agent systems, decomposed subtasks obscure the harmful intent from the model, causing safety-aligned models to fail. We will release our OS-BLIND to encourage the broader research community to further investigate and address these safety challenges. 

---
# Universal statistical signatures of evolution in artificial intelligence architectures 

**Authors**: Theodor Spiro  

**Link**: [PDF](https://arxiv.org/pdf/2604.10571)  

**Abstract**: We test whether artificial intelligence architectural evolution obeys the same statistical laws as biological evolution. Compiling 935 ablation experiments from 161 publications, we show that the distribution of fitness effects (DFE) of architectural modifications follows a heavy-tailed Student's t-distribution with proportions (68% deleterious, 19% neutral, 13% beneficial for major ablations, n=568) that place AI between compact viral genomes and simple eukaryotes. The DFE shape matches D. melanogaster (normalized KS=0.07) and S. cerevisiae (KS=0.09); the elevated beneficial fraction (13% vs. 1-6% in biology) quantifies the advantage of directed over blind search while preserving the distributional form. Architectural origination follows logistic dynamics (R^2=0.994) with punctuated equilibria and adaptive radiation into domain niches. Fourteen architectural traits were independently invented 3-5 times, paralleling biological convergences. These results demonstrate that the statistical structure of evolution is substrate-independent, determined by fitness landscape topology rather than the mechanism of selection. 

---
# Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models 

**Authors**: Jiyeon Kim, Sungik Choi, Yongrae Jo, Moontae Lee, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2604.10567)  

**Abstract**: Diffusion-based language models (dLLMs) have emerged as a promising alternative to autoregressive language models, offering the potential for parallel token generation and bidirectional context modeling. However, harnessing this flexibility for fully non-autoregressive decoding remains an open question, particularly for reasoning and planning tasks. In this work, we investigate non-autoregressive decoding in dLLMs by systematically analyzing its inference dynamics along the temporal axis. Specifically, we uncover an inherent failure mode in confidence-based non-autoregressive generation stemming from a strong proximity bias-the tendency for the denoising order to concentrate on spatially adjacent tokens. This local dependency leads to spatial error propagation, rendering the entire trajectory critically contingent on the initial unmasking position. Leveraging this insight, we present a minimal-intervention approach that guides early token selection, employing a lightweight planner and end-of-sequence temperature annealing. We thoroughly evaluate our method on various reasoning and planning tasks and observe substantial overall improvement over existing heuristic baselines without significant computational overhead. 

---
# LLMs Should Incorporate Explicit Mechanisms for Human Empathy 

**Authors**: Xiaoxing You, Qiang Huang, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10557)  

**Abstract**: This paper argues that Large Language Models (LLMs) should incorporate explicit mechanisms for human empathy. As LLMs become increasingly deployed in high-stakes human-centered settings, their success depends not only on correctness or fluency but on faithful preservation of human perspectives. Yet, current LLMs systematically fail at this requirement: even when well-aligned and policy-compliant, they often attenuate affect, misrepresent contextual salience, and rigidify relational stance in ways that distort meaning. We formalize empathy as an observable behavioral property: the capacity to model and respond to human perspectives while preserving intention, affect, and context. Under this framing, we identify four recurring mechanisms of empathic failure in contemporary LLMs--sentiment attenuation, empathic granularity mismatch, conflict avoidance, and linguistic distancing--arising as structural consequences of prevailing training and alignment practices. We further organize these failures along three dimensions: cognitive, cultural, and relational empathy, to explain their manifestation across tasks. Empirical analyses show that strong benchmark performance can mask systematic empathic distortions, motivating empathy-aware objectives, benchmarks, and training signals as first-class components of LLM development. 

---
# WaveMoE: A Wavelet-Enhanced Mixture-of-Experts Foundation Model for Time Series Forecasting 

**Authors**: Shunyu Wu, Jiawei Huang, Weibin Feng, Boxin Li, Xiao Zhang, Erli Meng, Dan Li, Jian Lou, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.10544)  

**Abstract**: Time series foundation models (TSFMs) have recently achieved remarkable success in universal forecasting by leveraging large-scale pretraining on diverse time series data. Complementing this progress, incorporating frequency-domain information yields promising performance in enhancing the modeling of complex temporal patterns, such as periodicity and localized high-frequency dynamics, which are prevalent in real-world time series. To advance this direction, we propose a new perspective that integrates explicit frequency-domain representations into scalable foundation models, and introduce WaveMoE, a wavelet-enhanced mixture-of-experts foundation model for time series forecasting. WaveMoE adopts a dual-path architecture that jointly processes time series tokens and wavelet tokens aligned along a unified temporal axis, and coordinates them through a shared expert routing mechanism that enables consistent expert specialization while efficiently scaling model capacity. Preliminary experimental results on 16 diverse benchmark datasets indicate that WaveMoE has the potential to further improve forecasting performance by incorporating wavelet-domain corpora. 

---
# VidAudio-Bench: Benchmarking V2A and VT2A Generation across Four Audio Categories 

**Authors**: Qian Zhang, Yuqin Cao, Yixuan Gao, Xiongkuo Min  

**Link**: [PDF](https://arxiv.org/pdf/2604.10542)  

**Abstract**: Video-to-Audio (V2A) generation is essential for immersive multimedia experiences, yet its evaluation remains underexplored. Existing benchmarks typically assess diverse audio types under a unified protocol, overlooking the fine-grained requirements of distinct audio categories. To address this gap, we propose VidAudio-Bench, a multi-task benchmark for V2A evaluation with four key features: (1) Broad Coverage: It encompasses four representative audio categories - sound effects, music, speech, and singing - under both V2A and Video-Text-to-Audio (VT2A) settings. (2) Extensive Evaluation: It comprises 1,634 video-text pairs and benchmarks 11 state-of-the-art generation models. (3) Comprehensive Metrics: It introduces 13 task-specific, reference-free metrics to systematically assess audio quality, video-audio consistency, and text-audio consistency. (4) Human Alignment: It validates all metrics through subjective studies, demonstrating strong consistency with human preferences. Experimental results reveal that current V2A models perform poorly in speech and singing compared to sound effects. Our VT2A results further highlight a fundamental tension between instruction following and visually grounded generation: stronger visual conditioning improves video-audio alignment, but often at the cost of generating the intended audio category. These findings establish VidAudio-Bench as a comprehensive and scalable framework for diagnosing V2A systems and provide new insights into multimodal audio generation. 

---
# IceCache: Memory-efficient KV-cache Management for Long-Sequence LLMs 

**Authors**: Yuzhen Mao, Qitong Wang, Martin Ester, Ke Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.10539)  

**Abstract**: Key-Value (KV) cache plays a crucial role in accelerating inference in large language models (LLMs) by storing intermediate attention states and avoiding redundant computation during autoregressive generation. However, its memory footprint scales linearly with sequence length, often leading to severe memory bottlenecks on resource-constrained hardware. Prior work has explored offloading KV cache to the CPU while retaining only a subset on the GPU, but these approaches often rely on imprecise token selection and suffer performance degradation in long-generation tasks such as chain-of-thought reasoning. In this paper, we propose a novel KV cache management strategy, IceCache, which integrates semantic token clustering with PagedAttention. By organizing semantically related tokens into contiguous memory regions managed by a hierarchical, dynamically updatable data structure, our method enables more efficient token selection and better utilization of memory bandwidth during CPU-GPU transfers. Experimental results on LongBench show that, with a 256-token budget, IceCache maintains 99% of the original accuracy achieved by the full KV cache model. Moreover, compared to other offloading-based methods, IceCache attains competitive or even superior latency and accuracy while using only 25% of the KV cache token budget, demonstrating its effectiveness in long-sequence scenarios. The code is available on our project website at this https URL. 

---
# Machine Learning-Based Detection of MCP Attacks 

**Authors**: Tobias Mattsson, Samuel Nyberg, Anton Borg, Ricardo Britto  

**Link**: [PDF](https://arxiv.org/pdf/2604.10534)  

**Abstract**: The Model Context Protocol (MCP) is a new and emerging technology that extends the functionality of large language models, improving workflows but also exposing users to a new attack surface. Several studies have highlighted related security flaws, but MCP attack detection remains underexplored. To address this research gap, this study develops and evaluates a range of supervised machine learning approaches, including both traditional and deep-learning models. We evaluated the systems on the detection of malicious MCP tool descriptions in two scenarios: (1) a binary classification task distinguishing malicious from benign tools, and (2) a multiclass classification task identifying the attack type while separating benign from malicious tools. In addition to the machine learning models, we compared a rule-based approach that serves as a baseline. The results indicate that several of the developed models achieved 100\% F1-score on the binary classification task. In the multiclass scenario, the SVC and BERT models performed best, achieving F1 scores of 90.56\% and 88.33\%, respectively. Confusion matrices were also used to visualize the full distribution of predictions often missed by traditional metrics, providing additional insight for selecting the best-fitting solution in real-world scenarios. This study presents an addition to the MCP defence area, showing that machine learning models can perform exceptionally well in separating malicious and benign data points. To apply the solution in a live environment, a middleware was developed to classify which MCP tools are safe to use before execution, and block the ones that are not safe. Furthermore, the study shows that these models can outperform traditional rule-based solutions currently in use in the field. 

---
# PepBenchmark: A Standardized Benchmark for Peptide Machine Learning 

**Authors**: Jiahui Zhang, Rouyi Wang, Kuangqi Zhou, Tianshu Xiao, Lingyan Zhu, Yaosen Min, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10531)  

**Abstract**: Peptide therapeutics are widely regarded as the "third generation" of drugs, yet progress in peptide Machine Learning (ML) are hindered by the absence of standardized benchmarks. Here we present PepBenchmark, which unifies datasets, preprocessing, and evaluation protocols for peptide drug discovery. PepBenchmark comprises three components: (1) PepBenchData, a well-curated collection comprising 29 canonical-peptide and 6 non-canonical-peptide datasets across 7 groups, systematically covering key aspects of peptide drug development, representing, to the best of our knowledge, the most comprehensive AI-ready dataset resource to date; (2) PepBenchPipeline, a standardized preprocessing pipeline that ensures consistent dataset cleaning, construction, splitting, and feature transformation, mitigating quality issues common in ad hoc pipelines; and (3) PepBenchLeaderboard, a unified evaluation protocol and leaderboard with strong baselines across 4 major methodological families: Fingerprint-based, GNN-based, PLM-based, and SMILES-based models. Together, PepBenchmark provides the first standardized and comparable foundation for peptide drug discovery, facilitating methodological advances and translation into real-world applications. The data and code are publicly available at this https URL. 

---
# Towards an Appropriate Level of Reliance on AI: A Preliminary Reliance-Control Framework for AI in Software Engineering 

**Authors**: Samuel Ferino, Rashina Hoda, John Grundy, Christoph Treude  

**Link**: [PDF](https://arxiv.org/pdf/2604.10530)  

**Abstract**: How software developers interact with Artificial Intelligence (AI)-powered tools, including Large Language Models (LLMs), plays a vital role in how these AI-powered tools impact them. While overreliance on AI may lead to long-term negative consequences (e.g., atrophy of critical thinking skills); underreliance might deprive software developers of potential gains in productivity and quality. Based on twenty-two interviews with software developers on using LLMs for software development, we propose a preliminary reliance-control framework where the level of control can be used as a way to identify AI overreliance and underreliance. We also use it to recommend future research to further explore the different control levels supported by the current and emergent LLM-driven tools. Our paper contributes to the emerging discourse on AI overreliance and provides an understanding of the appropriate degree of reliance as essential to developers making the most of these powerful technologies. Our findings can help practitioners, educators, and policymakers promote responsible and effective use of AI tools. 

---
# AI Patents in the United States and China: Measurement, Organization, and Knowledge Flows 

**Authors**: Hanming Fang, Xian Gu, Hanyin Yan, Wu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10529)  

**Abstract**: We develop a high-precision classifier to measure artificial intelligence (AI) patents by fine-tuning PatentSBERTa on manually labeled data from the USPTO's AI Patent Dataset. Our classifier substantially improves the existing USPTO approach, achieving 97.0% precision, 91.3% recall, and a 94.0% F1 score, and it generalizes well to Chinese patents based on citation and lexical validation. Applying it to granted U.S. patents (1976-2023) and Chinese patents (2010-2023), we document rapid growth in AI patenting in both countries and broad convergence in AI patenting intensity and subfield composition, even as China surpasses the United States in recent annual patent counts. The organization of AI innovation nevertheless differs sharply: U.S. AI patenting is concentrated among large private incumbents and established hubs, whereas Chinese AI patenting is more geographically diffuse and institutionally diverse, with larger roles for universities and state-owned enterprises. For listed firms, AI patents command a robust market-value premium in both countries. Cross-border citations show continued technological interdependence rather than decoupling, with Chinese AI inventors relying more heavily on U.S. frontier knowledge than vice versa. 

---
# STORM: End-to-End Referring Multi-Object Tracking in Videos 

**Authors**: Zijia Lu, Jingru Yi, Jue Wang, Yuxiao Chen, Junwen Chen, Xinyu Li, Davide Modolo  

**Link**: [PDF](https://arxiv.org/pdf/2604.10527)  

**Abstract**: Referring multi-object tracking (RMOT) is a task of associating all the objects in a video that semantically match with given textual queries or referring expressions. Existing RMOT approaches decompose object grounding and tracking into separated modules and exhibit limited performance due to the scarcity of training videos, ambiguous annotations, and restricted domains. In this work, we introduce STORM, an end-to-end MLLM that jointly performs grounding and tracking within a unified framework, eliminating external detectors and enabling coherent reasoning over appearance, motion, and language. To improve data efficiency, we propose a task-composition learning (TCL) strategy that decomposes RMOT into image grounding and object tracking, allowing STORM to leverage data-rich sub-tasks and learn structured spatial--temporal reasoning. We further construct STORM-Bench, a new RMOT dataset with accurate trajectories and diverse, unambiguous referring expressions generated through a bottom-up annotation pipeline. Extensive experiments show that STORM achieves state-of-the-art performance on image grounding, single-object tracking, and RMOT benchmarks, demonstrating strong generalization and robust spatial--temporal grounding in complex real-world scenarios. STORM-Bench is released at this https URL. 

---
# ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization 

**Authors**: Suyoung Bae, CheolWon Na, Jaehoon Lee, Yumin Lee, YunSeok Choi, Jee-Hyong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.10520)  

**Abstract**: As Large Language Models (LLMs) have become capable of generating long and descriptive code summaries, accurate and reliable evaluation of factual consistency has become a critical challenge. However, previous evaluation methods are primarily designed for short summaries of isolated code snippets. Consequently, they struggle to provide fine-grained evaluation of multi-sentence functionalities and fail to accurately assess dependency context commonly found in real-world code summaries. To address this, we propose ReFEree, a reference-free and fine-grained method for evaluating factual consistency in real-world code summaries. We define factual inconsistency criteria specific to code summaries and evaluate them at the segment level using these criteria along with dependency information. These segment-level results are then aggregated into a fine-grained score. We construct a code summarization benchmark with human-annotated factual consistency labels. The evaluation results demonstrate that ReFEree achieves the highest correlation with human judgment among 13 baselines, improving 15-18% over the previous state-of-the-art. Our code and data are available at this https URL. 

---
# Data-Efficient Surgical Phase Segmentation in Small-Incision Cataract Surgery: A Controlled Study of Vision Foundation Models 

**Authors**: Lincoln Spencer, Song Wang, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.10514)  

**Abstract**: Surgical phase segmentation is central to computer-assisted surgery, yet robust models remain difficult to develop when labeled surgical videos are scarce. We study data-efficient phase segmentation for manual small-incision cataract surgery (SICS) through a controlled comparison of visual representations. To isolate representation quality, we pair each visual encoder with the same temporal model (MS-TCN++) under identical training and evaluation settings on SICS-155 (19 phases). We compare supervised encoders (ResNet-50, I3D) against large self-supervised foundation models (DINOv3, V-JEPA2), and use a cached-feature pipeline that decouples expensive visual encoding from lightweight temporal learning. Foundation-model features improve segmentation performance in this setup, with DINOv3 ViT-7B achieving the best overall results (83.4% accuracy, 87.0 edit score). We further examine cataract-domain transfer using unlabeled videos and lightweight adaptation, and analyze when it helps or hurts. Overall, the study indicates strong transferability of modern vision foundation models to surgical workflow understanding and provides practical guidance for low-label medical video settings. The project website is available at: this https URL 

---
# How Many Tries Does It Take? Iterative Self-Repair in LLM Code Generation Across Model Scales and Benchmarks 

**Authors**: Johin Johny Arimbur  

**Link**: [PDF](https://arxiv.org/pdf/2604.10508)  

**Abstract**: Large language models frequently fail to produce correct code on their first attempt, yet most benchmarks evaluate them in a single-shot setting. We investigate iterative self-repair (feeding execution errors back to the model for correction) across seven models spanning three families and both open-weight and proprietary providers: Llama 3.1 8B, Llama 3.3 70B, Llama 4 Scout (MoE, 16 experts), Llama 4 Maverick (MoE, 128 experts), Qwen3 32B, Gemini 2.5 Flash, and Gemini 2.5 Pro. On HumanEval (164 problems) and MBPP Sanitized (257 problems) with up to five attempts, self-repair universally improves pass rates: +4.9 to +17.1 pp on HumanEval and +16.0 to +30.0 pp on MBPP. Gemini 2.5 Flash achieves the highest final pass rates (96.3% HumanEval, 93.8% MBPP). Most gains concentrate in the first two this http URL-type analysis shows assertion errors (logical mistakes) are the hardest to repair at ~45%, while syntax and name errors are repaired at substantially higher rates, connecting to broader findings on the limits of LLM self-correction. Prior work found that weaker models fail at self-repair or require fine-tuning; we show that modern instruction-tuned models succeed with prompting alone, even at 8B scale. We also provide the first comparison of dense and MoE architectures for self-repair, and extend the repair-vs-resampling tradeoff analysis to modern models. A prompt ablation reveals chain-of-thought repair yields up to +5.5 pp additional self-repair gain (measured as improvement in repair delta) over minimal prompting for capable models. 

---
# Cross-Cultural Bias in Mel-Scale Representations: Evidence and Alternatives from Speech and Music 

**Authors**: Shivam Chauhan, Ajay Pundhir  

**Link**: [PDF](https://arxiv.org/pdf/2604.10503)  

**Abstract**: Modern audio systems universally employ mel-scale representations derived from 1940s Western psychoacoustic studies, potentially encoding cultural biases that create systematic performance disparities. We present a comprehensive evaluation of cross-cultural bias in audio front-ends, comparing mel-scale features with learnable alternatives (LEAF, SincNet) and psychoacoustic variants (ERB, Bark, CQT) across speech recognition (11 languages), music analysis (6 collections), and European acoustic scene classification (10 European cities). Our controlled experiments isolate front-end contributions while holding architecture and training protocols minimal and constant. Results demonstrate that mel-scale features yield 31.2% WER for tonal languages compared to 18.7% for non-tonal languages (12.5% gap), and show 15.7% F1 degradation between Western and non-Western music. Alternative representations significantly reduce these disparities: LEAF reduces the speech gap by 34% through adaptive frequency allocation, CQT achieves 52% reduction in music performance gaps, and ERB-scale filtering cuts disparities by 31% with only 1% computational overhead. We also release FairAudioBench, enabling cross-cultural evaluation, and demonstrate that adaptive frequency decomposition offers practical paths toward equitable audio processing. These findings reveal how foundational signal processing choices propagate bias, providing crucial guidance for developing inclusive audio systems. 

---
# UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation 

**Authors**: Haopeng Chen, Yihao Ai, Kabeen Kim, Robby T. Tan, Yixin Chen, Bo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10485)  

**Abstract**: Low-visibility scenarios, such as low-light conditions, pose significant challenges to human pose estimation due to the scarcity of annotated low-light datasets and the loss of visual information under poor illumination. Recent domain adaptation techniques attempt to utilize well-lit labels by augmenting well-lit images to mimic low-light conditions. But handcrafted augmentations oversimplify noise patterns, while learning-based methods often fail to preserve high-frequency low-light characteristics, producing unrealistic images that lead pose models to generalize poorly to real low-light scenes. Moreover, recent pose estimators rely on image cues through image-to-keypoint cross-attention, but these cues become unreliable under low-light conditions. To address these issues, we propose Unsupervised Domain Adaptation for Pose Estimation (UDAPose), a novel framework that synthesizes low-light images and dynamically fuses visual cues with pose priors for improved pose estimation. Specifically, our synthesis method incorporates a Direct-Current-based High-Pass Filter (DHF) and a Low-light Characteristics Injection Module (LCIM) to inject high-frequency details from input low-light images, overcoming rigidity or the detail loss in existing approaches. Furthermore, we introduce a Dynamic Control of Attention (DCA) module that adaptively balances image cues with learned pose priors in the Transformer architecture. Experiments show that UDAPose outperforms state-of-the-art methods, with notable AP gains of 10.1 (56.4%) on the ExLPose-test hard set (LL-H) and 7.4 (31.4%) in cross-dataset validation on EHPT-XC. Code: this https URL 

---
# From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation 

**Authors**: Mingfei Lu, Yi Zhang, Mengjia Wu, Yue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.10470)  

**Abstract**: Legal consultation question answering (Legal CQA) presents unique challenges compared to traditional legal QA tasks, including the scarcity of high-quality training data, complex task composition, and strong contextual dependencies. To address these, we construct JurisCQAD, a large-scale dataset of over 43,000 real-world Chinese legal queries annotated with expert-validated positive and negative responses, and design a structured task decomposition that converts each query into a legal element graph integrating entities, events, intents, and legal issues. We further propose JurisMA, a modular multi-agent framework supporting dynamic routing, statutory grounding, and stylistic optimization. Combined with the element graph, the framework enables strong context-aware reasoning, effectively capturing dependencies across legal facts, norms, and procedural logic. Trained on JurisCQAD and evaluated on a refined LawBench, our system significantly outperforms both general-purpose and legal-domain LLMs across multiple lexical and semantic metrics, demonstrating the benefits of interpretable decomposition and modular collaboration in Legal CQA. 

---
# Rethinking the Diffusion Model from a Langevin Perspective 

**Authors**: Candi Zheng, Yuan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2604.10465)  

**Abstract**: Diffusion models are often introduced from multiple perspectives, such as VAEs, score matching, or flow matching, accompanied by dense and technically demanding mathematics that can be difficult for beginners to grasp. One classic question is: how does the reverse process invert the forward process to generate data from pure noise? This article systematically organizes the diffusion model from a fresh Langevin perspective, offering a simpler, clearer, and more intuitive answer. We also address the following questions: how can ODE-based and SDE-based diffusion models be unified under a single framework? Why are diffusion models theoretically superior to ordinary VAEs? Why is flow matching not fundamentally simpler than denoising or score matching, but equivalent under maximum-likelihood? We demonstrate that the Langevin perspective offers clear and straightforward answers to these questions, bridging existing interpretations of diffusion models, showing how different formulations can be converted into one another within a common framework, and offering pedagogical value for both learners and experienced researchers seeking deeper intuition. 

---
# Toward Accountable AI-Generated Content on Social Platforms: Steganographic Attribution and Multimodal Harm Detection 

**Authors**: Xinlei Guan, David Arosemena, Tejaswi Dhandu, Kuan Huang, Meng Xu, Miles Q. Li, Bingyu Shen, Ruiyang Qin, Umamaheswara Rao Tida, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.10460)  

**Abstract**: The rapid growth of generative AI has introduced new challenges in content moderation and digital forensics. In particular, benign AI-generated images can be paired with harmful or misleading text, creating difficult-to-detect misuse. This contextual misuse undermines the traditional moderation framework and complicates attribution, as synthetic images typically lack persistent metadata or device signatures. We introduce a steganography enabled attribution framework that embeds cryptographically signed identifiers into images at creation time and uses multimodal harmful content detection as a trigger for attribution verification. Our system evaluates five watermarking methods across spatial, frequency, and wavelet domains. It also integrates a CLIP-based fusion model for multimodal harmful-content detection. Experiments demonstrate that spread-spectrum watermarking, especially in the wavelet domain, provides strong robustness under blur distortions, and our multimodal fusion detector achieves an AUC-ROC of 0.99, enabling reliable cross-modal attribution verification. These components form an end-to-end forensic pipeline that enables reliable tracing of harmful deployments of AI-generated imagery, supporting accountability in modern synthetic media environments. Our code is available at GitHub: this https URL 

---
# Towards Green Wearable Computing: A Physics-Aware Spiking Neural Network for Energy-Efficient IMU-based Human Activity Recognition 

**Authors**: Naichuan Zheng, Hailun Xia, Zepeng Sun, Weiyi Li, Yinze Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.10458)  

**Abstract**: Wearable IMU-based Human Activity Recognition (HAR) relies heavily on Deep Neural Networks (DNNs), which are burdened by immense computational and buffering demands. Their power-hungry floating-point operations and rigid requirement to process complete temporal windows severely cripple battery-constrained edge devices. While Spiking Neural Networks (SNNs) offer extreme event-driven energy efficiency, standard architectures struggle with complex biomechanical topologies and temporal gradient degradation. To bridge this gap, we propose the Physics-Aware Spiking Neural Network (PAS-Net), a fully multiplier-free architecture explicitly tailored for Green HAR. Spatially, an adaptive symmetric topology mixer enforces human-joint physical constraints. Temporally, an $O(1)$-memory causal neuromodulator yields context-aware dynamic threshold neurons, adapting actively to non-stationary movement rhythms. Furthermore, we leverage a temporal spike error objective to unlock a flexible early-exit mechanism for continuous IMU streams. Evaluated across seven diverse datasets, PAS-Net achieves state-of-the-art accuracy while replacing dense operations with sparse 0.1 pJ integer accumulations. Crucially, its confidence-driven early-exit capability drastically reduces dynamic energy consumption by up to 98\%. PAS-Net establishes a robust, ultra-low-power neuromorphic standard for always-on wearable sensing. 

---
# A Queueing-Theoretic Framework for Dynamic Attack Surfaces: Data-Integrated Risk Analysis and Adaptive Defense 

**Authors**: Jihyeon Yun, Abdullah Yasin Etcibasi, Ming Shi, C. Emre Koksal  

**Link**: [PDF](https://arxiv.org/pdf/2604.10427)  

**Abstract**: We develop a queueing-theoretic framework to model the temporal evolution of cyber-attack surfaces, where the number of active vulnerabilities is represented as the backlog of a queue. Vulnerabilities arrive as they are discovered or created, and leave the system when they are patched or successfully exploited. Building on this model, we study how automation affects attack and defense dynamics by introducing an AI amplification factor that scales arrival, exploit, and patching rates. Our analysis shows that even symmetric automation can increase the rate of successful exploits. We validate the model using vulnerability data collected from an open source software supply chain and show that it closely matches real-world attack surface dynamics. Empirical results reveal heavy-tailed patching times, which we prove induce long-range dependence in vulnerability backlog and help explain persistent cyber risk. Utilizing our queueing abstraction for the attack surface, we develop a systematic approach for cyber risk mitigation. We formulate the dynamic defense problem as a constrained Markov decision process with resource-budget and switching-cost constraints, and develop a reinforcement learning (RL) algorithm that achieves provably near-optimal regret. Numerical experiments validate the approach and demonstrate that our adaptive RL-based defense policies significantly reduce successful exploits and mitigate heavy-tail queue events. Using trace-driven experiments on the ARVO dataset, we show that the proposed RL-based defense policy reduces the average number of active vulnerabilities in a software supply chain by over 90% compared to existing defense practices, without increasing the overall maintenance budget. Our results allow defenders to quantify cumulative exposure risk under long-range dependent attack dynamics and to design adaptive defense strategies with provable efficiency. 

---
# CodaRAG: Connecting the Dots with Associativity Inspired by Complementary Learning 

**Authors**: Cheng-Yen Li, Xuanjun Chen, Claire Lin, Wei-Yu Chen, Wenhua Nie, Hung-Yi Lee, Jyh-Shing Roger Jang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10426)  

**Abstract**: Large Language Models (LLMs) struggle with knowledge-intensive tasks due to hallucinations and fragmented reasoning over dispersed information. While Retrieval-Augmented Generation (RAG) grounds generation in external sources, existing methods often treat evidence as isolated units, failing to reconstruct the logical chains that connect these dots. Inspired by Complementary Learning Systems (CLS), we propose CodaRAG, a framework that evolves retrieval from passive lookup into active associative discovery. CodaRAG operates via a three-stage pipeline: (1) Knowledge Consolidation to unify fragmented extractions into a stable memory substrate; (2) Associative Navigation to traverse the graph via multi-dimensional pathways-semantic, contextualized, and functional-explicitly recovering dispersed evidence chains; and (3) Interference Elimination to prune hyper-associative noise, ensuring a coherent, high-precision reasoning context. On GraphRAG-Bench, CodaRAG achieves absolute gains of 7-10% in retrieval recall and 3-11% in generation accuracy. These results demonstrate CodaRAG's superior ability to systematically robustify associative evidence retrieval for factual, reasoning, and creative tasks. 

---
# IMPACT: A Dataset for Multi-Granularity Human Procedural Action Understanding in Industrial Assembly 

**Authors**: Di Wen, Zeyun Zhong, David Schneider, Manuel Zaremski, Linus Kunzmann, Yitian Shi, Ruiping Liu, Yufan Chen, Junwei Zheng, Jiahang Li, Jonas Hemmerich, Qiyi Tong, Patric Grauberger, Arash Ajoudani, Danda Pani Paudel, Sven Matthiesen, Barbara Deml, Jürgen Beyerer, Luc Van Gool, Rainer Stiefelhagen, Kunyu Peng  

**Link**: [PDF](https://arxiv.org/pdf/2604.10409)  

**Abstract**: We introduce IMPACT, a synchronized five-view RGB-D dataset for deployment-oriented industrial procedural understanding, built around real assembly and disassembly of a commercial angle grinder with professional-grade tools. To our knowledge, IMPACT is the first real industrial assembly benchmark that jointly provides synchronized ego-exo RGB-D capture, decoupled bimanual annotation, compliance-aware state tracking, and explicit anomaly--recovery supervision within a single real industrial workflow. It comprises 112 trials from 13 participants totaling 39.5 hours, with multi-route execution governed by a partial-order prerequisite graph, a six-category anomaly taxonomy, and operator cognitive load measured via NASA-TLX. The annotation hierarchy links hand-specific atomic actions to coarse procedural steps, component assembly states, and per-hand compliance phases, with synchronized null spans across views to decouple perceptual limitations from algorithmic failure. Systematic baselines reveal fundamental limitations that remain invisible to single-task benchmarks, particularly under realistic deployment conditions that involve incomplete observations, flexible execution paths, and corrective behavior. The full dataset, annotations, and evaluation code are available at this https URL. 

---
# Rethinking Video Human-Object Interaction: Set Prediction over Time for Unified Detection and Anticipation 

**Authors**: Yuanhao Luo, Di Wen, Kunyu Peng, Ruiping Liu, Junwei Zheng, Yufan Chen, Jiale Wei, Rainer Stiefelhage  

**Link**: [PDF](https://arxiv.org/pdf/2604.10397)  

**Abstract**: Video-based human-object interaction (HOI) understanding requires both detecting ongoing interactions and anticipating their future evolution. However, existing methods usually treat anticipation as a downstream forecasting task built on externally constructed human-object pairs, limiting joint reasoning between detection and prediction. In addition, sparse keyframe annotations in current benchmarks can temporally misalign nominal future labels from actual future dynamics, reducing the reliability of anticipation evaluation. To address these issues, we introduce DETAnt-HOI, a temporally corrected benchmark derived from VidHOI and Action Genome for more faithful multi-horizon evaluation, and HOI-DA, a pair-centric framework that jointly performs subject-object localization, present HOI detection, and future anticipation by modeling future interactions as residual transitions from current pair states. Experiments show consistent improvements in both detection and anticipation, with larger gains at longer horizons. Our results highlight that anticipation is most effective when learned jointly with detection as a structural constraint on pair-level video representation learning. Benchmark and code will be publicly available. 

---
# Intent-aligned Formal Specification Synthesis via Traceable Refinement 

**Authors**: Zhe Ye, Aidan Z.H. Yang, Huangyuan Su, Zhenyu Liao, Samuel Tenka, Zhizhen Qin, Udaya Ghai, Dawn Song, Soonho Kong  

**Link**: [PDF](https://arxiv.org/pdf/2604.10392)  

**Abstract**: Large language models are increasingly used to generate code from natural language, but ensuring correctness remains challenging. Formal verification offers a principled way to obtain such guarantees by proving that a program satisfies a formal specification. However, specifications are frequently missing in real-world codebases, and writing high-quality specifications remains expensive and expertise-intensive. We present VeriSpecGen, a traceable refinement framework that synthesizes intent-aligned specifications in Lean through requirement-level attribution and localized repair. VeriSpecGen decomposes natural language into atomic requirements and generates requirement-targeted tests with explicit traceability maps to validate generated specifications. When validation fails, traceability maps attribute failures to specific requirements, enabling targeted clause-level repairs. VeriSpecGen achieve 86.6% on VERINA SpecGen task using Claude Opus 4.5, improving over baselines by up to 31.8 points across different model families and scales. Beyond inference-time gains, we generate 343K training examples from VeriSpecGen refinement trajectories and demonstrate that training on these trajectories substantially improves specification synthesis by 62-106% relative and transfers gains to general reasoning abilities. 

---
# FishRoPE: Projective Rotary Position Embeddings for Omnidirectional Visual Perception 

**Authors**: Rahul Ahuja, Mudit Jain, Bala Murali Manoghar Sai Sudhakar, Venkatraman Narayanan, Pratik Likhar, Varun Ravi Kumar, Senthil Yogamani  

**Link**: [PDF](https://arxiv.org/pdf/2604.10391)  

**Abstract**: Vision foundation models (VFMs) and Bird's Eye View (BEV) representation have advanced visual perception substantially, yet their internal spatial representations assume the rectilinear geometry of pinhole cameras. Fisheye cameras, widely deployed on production autonomous vehicles for their surround-view coverage, exhibit severe radial distortion that renders these representations geometrically inconsistent. At the same time, the scarcity of large-scale fisheye annotations makes retraining foundation models from scratch impractical. We present \ours, a lightweight framework that adapts frozen VFMs to fisheye geometry through two components: a frozen DINOv2 backbone with Low-Rank Adaptation (LoRA) that transfers rich self-supervised features to fisheye without task-specific pretraining, and Fisheye Rotary Position Embedding (FishRoPE), which reparameterizes the attention mechanism in the spherical coordinates of the fisheye projection so that both self-attention and cross-attention operate on angular separation rather than pixel distance. FishRoPE is architecture-agnostic, introduces negligible computational overhead, and naturally reduces to the standard formulation under pinhole geometry. We evaluate \ours on WoodScape 2D detection (54.3 mAP) and SynWoodScapes BEV segmentation (65.1 mIoU), where it achieves state-of-the-art results on both benchmarks. 

---
# Multinex: Lightweight Low-light Image Enhancement via Multi-prior Retinex 

**Authors**: Alexandru Brateanu, Tingting Mu, Codruta Ancuti, Cosmin Ancuti  

**Link**: [PDF](https://arxiv.org/pdf/2604.10359)  

**Abstract**: Low-light image enhancement (LLIE) aims to restore natural visibility, color fidelity, and structural detail under severe illumination degradation. State-of-the-art (SOTA) LLIE techniques often rely on large models and multi-stage training, limiting practicality for edge deployment. Moreover, their dependence on a single color space introduces instability and visible exposure or color artifacts. To address these, we propose Multinex, an ultra-lightweight structured framework that integrates multiple fine-grained representations within a principled Retinex residual formulation. It decomposes an image into illumination and color prior stacks derived from distinct analytic representations, and learns to fuse these representations into luminance and reflectance adjustments required to correct exposure. By prioritizing enhancement over reconstruction and exploiting lightweight neural operations, Multinex significantly reduces computational cost, exemplified by its lightweight (45K parameters) and nano (0.7K parameters) versions. Extensive benchmarks show that all lightweight variants significantly outperform their corresponding lightweight SOTA models, and reach comparable performance to heavy models. Paper page available at this https URL. 

---
# A Diffusion-Contrastive Graph Neural Network with Virtual Nodes for Wind Nowcasting in Unobserved Regions 

**Authors**: Jie Shi, Siamak Mehrkanoon  

**Link**: [PDF](https://arxiv.org/pdf/2604.10328)  

**Abstract**: Accurate weather nowcasting remains one of the central challenges in atmospheric science, with critical implications for climate resilience, energy security, and disaster preparedness. Since it is not feasible to deploy observation stations everywhere, some regions lack dense observational networks, resulting in unreliable short-term wind predictions across those unobserved areas. Here we present a deep graph self-supervised framework that extends nowcasting capability into such unobserved regions without requiring new sensors. Our approach introduces "virtual nodes" into a diffusion and contrastive-based graph neural network, enabling the model to learn wind condition (i.e., speed, direction and gusts) in places with no direct measurements. Using high-temporal resolution weather station data across the Netherlands, we demonstrate that this approach reduces nowcast mean absolute error (MAE) of wind speed, gusts, and direction in unobserved regions by more than 30% - 46% compared with interpolation and regression methods. By enabling localized nowcasts where no measurements exist, this method opens new pathways for renewable energy integration, agricultural planning, and early-warning systems in data-sparse regions. 

---
# Jailbreaking the Matrix: Nullspace Steering for Controlled Model Subversion 

**Authors**: Vishal Pramanik, Maisha Maliha, Susmit Jha, Sumit Kumar Jha  

**Link**: [PDF](https://arxiv.org/pdf/2604.10326)  

**Abstract**: Large language models remain vulnerable to jailbreak attacks -- inputs designed to bypass safety mechanisms and elicit harmful responses -- despite advances in alignment and instruction tuning. We propose Head-Masked Nullspace Steering (HMNS), a circuit-level intervention that (i) identifies attention heads most causally responsible for a model's default behavior, (ii) suppresses their write paths via targeted column masking, and (iii) injects a perturbation constrained to the orthogonal complement of the muted subspace. HMNS operates in a closed-loop detection-intervention cycle, re-identifying causal heads and reapplying interventions across multiple decoding attempts. Across multiple jailbreak benchmarks, strong safety defenses, and widely used language models, HMNS attains state-of-the-art attack success rates with fewer queries than prior methods. Ablations confirm that nullspace-constrained injection, residual norm scaling, and iterative re-identification are key to its effectiveness. To our knowledge, this is the first jailbreak method to leverage geometry-aware, interpretability-informed interventions, highlighting a new paradigm for controlled model steering and adversarial safety circumvention. 

---
# Class-Adaptive Cooperative Perception for Multi-Class LiDAR-based 3D Object Detection in V2X Systems 

**Authors**: Blessing Agyei Kyem, Joshua Kofi Asamoah, Armstrong Aboah  

**Link**: [PDF](https://arxiv.org/pdf/2604.10305)  

**Abstract**: Cooperative perception allows connected vehicles and roadside infrastructure to share sensor observations, creating a fused scene representation beyond the capability of any single platform. However, most cooperative 3D object detectors use a uniform fusion strategy for all object classes, which limits their ability to handle the different geometric structures and point-sampling patterns of small and large objects. This problem is further reinforced by narrow evaluation protocols that often emphasize a single dominant class or only a few cooperation settings, leaving robust multi-class detection across diverse vehicle-to-everything interactions insufficiently explored. To address this gap, we propose a class-adaptive cooperative perception architecture for multi-class 3D object detection from LiDAR data. The model integrates four components: multi-scale window attention with learned scale routing for spatially adaptive feature extraction, a class-specific fusion module that separates small and large objects into attentive fusion pathways, bird's-eye-view enhancement through parallel dilated convolution and channel recalibration for richer contextual representation, and class-balanced objective weighting to reduce bias toward frequent categories. Experiments on the V2X-Real benchmark cover vehicle-centric, infrastructure-centric, vehicle-to-vehicle, infrastructure-to-infrastructure, and vehicle-to-infrastructure settings under identical backbone and training configurations. The proposed method consistently improves mean detection performance over strong intermediate-fusion baselines, with the largest gains on trucks, clear improvements on pedestrians, and competitive results on cars. These results show that aligning feature extraction and fusion with class-dependent geometry and point density leads to more balanced cooperative perception in realistic vehicle-to-everything deployments. 

---
# From Helpful to Trustworthy: LLM Agents for Pair Programming 

**Authors**: Ragib Shahariar Ayon  

**Link**: [PDF](https://arxiv.org/pdf/2604.10300)  

**Abstract**: LLM-based coding agents are increasingly used to generate code, tests, and documentation. Still, their outputs can be plausible yet misaligned with developer intent and provide limited evidence for review in evolving projects. This limits our understanding of how to structure LLM pair-programming workflows so that artifacts remain reliable, auditable, and maintainable over time. To address this gap, this doctoral research proposes a systematic study of multi-agent LLM pair programming that externalizes intent and uses development tools for iterative validation. The plan includes three studies: translating informal problem statements into standards aligned requirements and formal specifications; refining tests and implementations using automated feedback, such as solver-backed counterexamples; and supporting maintenance tasks, including refactoring, API migrations, and documentation updates, while preserving validated behavior. The expected outcome is a clearer understanding of when multi-agent workflows increase trust, along with practical guidance for building reliable programming assistants for real-world development. 

---
# FashionMV: Product-Level Composed Image Retrieval with Multi-View Fashion Data 

**Authors**: Peng Yuan, Bingyin Mei, Hui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10297)  

**Abstract**: Composed Image Retrieval (CIR) retrieves target images using a reference image paired with modification text. Despite rapid advances, all existing methods and datasets operate at the image level -- a single reference image plus modification text in, a single target image out -- while real e-commerce users reason about products shown from multiple viewpoints. We term this mismatch View Incompleteness and formally define a new Multi-View CIR task that generalizes standard CIR from image-level to product-level retrieval. To support this task, we construct FashionMV, the first large-scale multi-view fashion dataset for product-level CIR, comprising 127K products, 472K multi-view images, and over 220K CIR triplets, built through a fully automated pipeline leveraging large multimodal models. We further propose ProCIR (Product-level Composed Image Retrieval), a modeling framework built upon a multimodal large language model that employs three complementary mechanisms -- two-stage dialogue, caption-based alignment, and chain-of-thought guidance -- together with an optional supervised fine-tuning (SFT) stage that injects structured product knowledge prior to contrastive training. Systematic ablation across 16 configurations on three fashion benchmarks reveals that: (1) alignment is the single most critical mechanism; (2) the two-stage dialogue architecture is a prerequisite for effective alignment; and (3) SFT and chain-of-thought serve as partially redundant knowledge injection paths. Our best 0.8B-parameter model outperforms all baselines, including general-purpose embedding models 10x its size. The dataset, model, and code are publicly available at this https URL. 

---
# Adapting 2D Multi-Modal Large Language Model for 3D CT Image Analysis 

**Authors**: Yang Yu, Dunyuan Xu, Yaoqian Li, Xiaomeng Li, Jinpeng Li, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2604.10233)  

**Abstract**: 3D medical image analysis is of great importance in disease diagnosis and treatment. Recently, multimodal large language models (MLLMs) have exhibited robust perceptual capacity, strong cross-modal alignment, and promising generalizability. Therefore, they have great potential to improve the performance of medical report generation (MRG) and medical visual question answering (MVQA), which serve as two important tasks in clinical scenarios. However, due to the scarcity of 3D medical images, existing 3D medical MLLMs suffer from insufficiently pretrained vision encoder and inability to extract customized image features for different kinds of tasks. In this paper, we propose to first transfer a 2D MLLM, which is well trained with 2D natural images, to support 3D medical volumetric inputs while reusing all of its pre-trained parameters. To enable the vision encoder to extract tailored image features for various tasks, we then design a Text-Guided Hierarchical MoE (TGH-MoE) framework, which can distinguish tasks under the guidance of the text prompt. Furthermore, we propose a two-stage training strategy to learn both task-shared and task-specific image features. As demonstrated empirically, our method outperforms existing 3D medical MLLMs in both MRG and MVQA tasks. Our code will be released once this paper is accepted. 

---
# Exploring the impact of fairness-aware criteria in AutoML 

**Authors**: Joana Simões, João Correia  

**Link**: [PDF](https://arxiv.org/pdf/2604.10224)  

**Abstract**: Machine Learning (ML) systems are increasingly used to support decision-making processes that affect individuals. However, these systems often rely on biased data, which can lead to unfair outcomes against specific groups. With the growing adoption of Automated Machine Learning (AutoML), the risk of intensifying discriminatory behaviours increases, as most frameworks primarily focus on model selection to maximise predictive performance. Previous research on fairness in AutoML had largely followed this trend, integrating fairness awareness only in the model selection or hyperparameter tuning, while neglecting other critical stages of the ML pipeline. This paper aims to study the impact of integrating fairness directly into the optimisation component of an AutoML framework that constructs complete ML pipelines, from data selection and transformations to model selection and tuning. As selecting appropriate fairness metrics remains a key challenge, our work incorporates complementary fairness metrics to capture different dimensions of fairness during the optimisation. Their integration within AutoML resulted in measurable differences compared to a baseline focused solely on predictive performance. Despite a 9.4% decrease in predictive power, the average fairness improved by 14.5%, accompanied by a 35.7% reduction in data usage. Furthermore, fairness integration produced complete yet simpler final solutions, suggesting that model complexity is not always required to achieve balanced and fair ML solutions. 

---
# Wolkowicz-Styan Upper Bound on the Hessian Eigenspectrum for Cross-Entropy Loss in Nonlinear Smooth Neural Networks 

**Authors**: Yuto Omae, Kazuki Sakai, Yohei Kakimoto, Makoto Sasaki, Yusuke Sakai, Hirotaka Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2604.10202)  

**Abstract**: Neural networks (NNs) are central to modern machine learning and achieve state-of-the-art results in many applications. However, the relationship between loss geometry and generalization is still not well understood. The local geometry of the loss function near a critical point is well-approximated by its quadratic form, obtained through a second-order Taylor expansion. The coefficients of the quadratic term correspond to the Hessian matrix, whose eigenspectrum allows us to evaluate the sharpness of the loss at the critical point. Extensive research suggests flat critical points generalize better, while sharp ones lead to higher generalization error. However, sharpness requires the Hessian eigenspectrum, but general matrix characteristic equations have no closed-form solution. Therefore, most existing studies on evaluating loss sharpness rely on numerical approximation methods. Existing closed-form analyses of the eigenspectrum are primarily limited to simplified architectures, such as linear or ReLU-activated networks; consequently, theoretical analysis of smooth nonlinear multilayer neural networks remains limited. Against this background, this study focuses on nonlinear, smooth multilayer neural networks and derives a closed-form upper bound for the maximum eigenvalue of the Hessian with respect to the cross-entropy loss by leveraging the Wolkowicz-Styan bound. Specifically, the derived upper bound is expressed as a function of the affine transformation parameters, hidden layer dimensions, and the degree of orthogonality among the training samples. The primary contribution of this paper is an analytical characterization of loss sharpness in smooth nonlinear multilayer neural networks via a closed-form expression, avoiding explicit numerical eigenspectrum computation. We hope that this work provides a small yet meaningful step toward unraveling the mysteries of deep learning. 

---
# Virtual Smart Metering in District Heating Networks via Heterogeneous Spatial-Temporal Graph Neural Networks 

**Authors**: Keivan Faghih Niresi, Christian Møller Jensen, Carsten Skovmose Kallesøe, Rafael Wisniewski, Olga Fink  

**Link**: [PDF](https://arxiv.org/pdf/2604.10166)  

**Abstract**: Intelligent operation of thermal energy networks aims to improve energy efficiency, reliability, and operational flexibility through data-driven control, predictive optimization, and early fault detection. Achieving these goals relies on sufficient observability, requiring continuous and well-distributed monitoring of thermal and hydraulic states. However, district heating systems are typically sparsely instrumented and frequently affected by sensor faults, limiting monitoring. Virtual sensing offers a cost-effective means to enhance observability, yet its development and validation remain limited in practice. Existing data-driven methods generally assume dense synchronized data, while analytical models rely on simplified hydraulic and thermal assumptions that may not adequately capture the behavior of heterogeneous network topologies. Consequently, modeling the coupled nonlinear dependencies between pressure, flow, and temperature under realistic operating conditions remains challenging. In addition, the lack of publicly available benchmark datasets hinders systematic comparison of virtual sensing approaches. To address these challenges, we propose a heterogeneous spatial-temporal graph neural network (HSTGNN) for constructing virtual smart heat meters. The model incorporates the functional relationships inherent in district heating networks and employs dedicated branches to learn graph structures and temporal dynamics for flow, temperature, and pressure measurements, thereby enabling the joint modeling of cross-variable and spatial correlations. To support further research, we introduce a controlled laboratory dataset collected at the Aalborg Smart Water Infrastructure Laboratory, providing synchronized high-resolution measurements representative of real operating conditions. Extensive experiments demonstrate that the proposed approach significantly outperforms existing baselines. 

---
# A Temporally Augmented Graph Attention Network for Affordance Classification 

**Authors**: Ami Chopra, Supriya Bordoloi, Shyamanta M. Hazarika  

**Link**: [PDF](https://arxiv.org/pdf/2604.10149)  

**Abstract**: Graph attention networks (GATs) provide one of the best frameworks for learning node representations in relational data; but, existing variants such as Graph Attention Network (GAT) mainly operate on static graphs and rely on implicit temporal aggregation when applied to sequential data. In this paper, we introduce Electroencephalography-temporal Graph Attention Network (EEG-tGAT), a temporally augmented formulation of GATv2 that is tailored for affordance classification from interaction sequences. The proposed model incorporates temporal attention to modulate the contribution of different time segments and temporal dropout to regularize learning across temporally correlated observations. The design reflects the assumption that temporal dimensions in affordance data are not semantically uniform and that discriminative information may be unevenly distributed across time. Experimental results on affordance datasets show that EEG-tGAT achieves improved classification performance compared to GATv2. The observed gains helps to conclude that explicitly encoding temporal importance and enforcing temporal robustness introduce inductive biases that are much better aligned with the structure of affordance-driven interaction data. These findings show us that modest architectural changes to graph attention models can help one obtain consistent benefits when temporal relationships play a nontrivial role in the task. 

---
# MOSAIC: Multi-Domain Orthogonal Session Adaptive Intent Capture for Prescient Recommendations 

**Authors**: Abderaouf Bahi, Mourad Boughaba, Ibtissem Gasmi, Warda Deghmane, Amel Ourici  

**Link**: [PDF](https://arxiv.org/pdf/2604.10147)  

**Abstract**: Capturing user intent across heterogeneous behavioral domains stands as a fundamental challenge in session-based recommender systems. Yet, existing multi-domain approaches frequently fail to isolate the distinct contribution of cross-domain interactions from those arising within individual domains, limiting their ability to build rich and transferable user representations. In this work, we propose MOSAIC, a Multi-Domain Orthogonal Session Adaptive Intent Capture framework that explicitly factorizes user preferences into three orthogonal components: domain-specific, domain-common, and cross-sequence-exclusive representations. Our approach employs a triple-encoder architecture, where each encoder is dedicated to one preference type, enforced through domain masking objectives and adversarial training via a gradient reversal layer. Representational alignment and mutual independence constraints are jointly optimized to ensure clean preference separation. Additionally, a dynamic gating mechanism modulates the relative contribution of each component at every timestep, yielding a unified and temporally adaptive session-level user representation. We conduct extensive experiments on two large-scale real-world benchmarks spanning multiple domains and interaction types. The ablation study validates that each component domain-specific encoding, domain-common modeling, cross-sequence representation, and dynamic gating contributes meaningfully to the overall performance. Experimental results demonstrate that MOSAIC consistently outperforms state-of-the-art baselines in recommendation accuracy, while simultaneously providing interpretable insights into the interplay between domain-specific and cross-domain preference signals. These findings highlight the potential of orthogonal preference decomposition as a principled strategy for next-generation multi-domain recommender systems. 

---
# Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities 

**Authors**: Zhichen Liu, Yongyuan Li, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10135)  

**Abstract**: Researchers have explored different ways to improve large language models (LLMs)' capabilities via dummy token insertion in contexts. However, existing works focus solely on the dummy tokens themselves, but fail to leverage the inherent sentence-level structure of natural language. This is a critical oversight, as LLMs acquire linguistic capabilities through exposure to human-generated texts, which are inherently structured at the sentence level. Motivated by this gap, we propose an approach that inserts delimiters at sentence boundaries in LLM inputs, which not only integrates dummy tokens into the context, but also facilitates LLMs with sentence-by-sentence processing behavior during reasoning. Two concrete methods: (1). In-context learning and (2). Supervised fine-tuning are experimented using 7B models to 600B Deepseek-V3. Our results demonstrate consistent improvements across various tasks, with notable gains of up to 7.7\% on GSM8k and 12.5\% on DROP. Furthermore, the fine-tuned LLMs can incorporate sentence awareness evidenced by their internal representations. Our work establishes a simple yet effective technique for enhancing LLM's capabilities, offering promising directions for cognitive-inspired LLM enhancement paradigm. 

---
# Semantic Manipulation Localization 

**Authors**: Zhenshan Tan, Chenhan Lu, Yuxiang Huang, Ziwen He, Xiang Zhang, Yuzhe Sha, Xianyi Chen, Tianrun Chen, Zhangjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10132)  

**Abstract**: Image Manipulation Localization (IML) aims to identify edited regions in an image. However, with the increasing use of modern image editing and generative models, many manipulations no longer exhibit obvious low-level artifacts. Instead, they often involve subtle but meaning-altering edits to an object's attributes, state, or relationships while remaining highly consistent with the surrounding content. This makes conventional IML methods less effective because they mainly rely on artifact detection rather than semantic sensitivity. To address this issue, we introduce Semantic Manipulation Localization (SML), a new task that focuses on localizing subtle semantic edits that significantly change image interpretation. We further construct a dedicated fine-grained benchmark for SML using a semantics-driven manipulation pipeline with pixel-level annotations. Based on this task, we propose TRACE (Targeted Reasoning of Attributed Cognitive Edits), an end-to-end framework that models semantic sensitivity through three progressively coupled components: semantic anchoring, semantic perturbation sensing, and semantic-constrained reasoning. Specifically, TRACE first identifies semantically meaningful regions that support image understanding, then injects perturbation-sensitive frequency cues to capture subtle edits under strong visual consistency, and finally verifies candidate regions through joint reasoning over semantic content and semantic scope. Extensive experiments show that TRACE consistently outperforms existing IML methods on our benchmark and produces more complete, compact, and semantically coherent localization results. These results demonstrate the necessity of moving beyond artifact-based localization and provide a new direction for image forensics in complex semantic editing scenarios. 

---
# VGA-Bench: A Unified Benchmark and Multi-Model Framework for Video Aesthetics and Generation Quality Evaluation 

**Authors**: Longteng Jiang, DanDan Zheng, Qianqian Qiao, Heng Huang, Huaye Wang, Yihang Bo, Bao Peng, Jingdong Chen, Jun Zhou, Xin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.10127)  

**Abstract**: The rapid advancement of AIGC-based video generation has underscored the critical need for comprehensive evaluation frameworks that go beyond traditional generation quality metrics to encompass aesthetic appeal. However, existing benchmarks remain largely focused on technical fidelity, leaving a significant gap in holistic assessment-particularly with respect to perceptual and artistic qualities. To address this limitation, we introduce VGA-Bench, a unified benchmark for joint evaluation of video generation quality and aesthetic quality. VGA-Bench is built upon a principled three-tier taxonomy: Aesthetic Quality, Aesthetic Tagging, and Generation Quality, each decomposed into multiple fine-grained sub-dimensions to enable systematic assessment. Guided by this taxonomy, we design 1,016 diverse prompts and generate a large-scale dataset of over 60,000 videos using 12 video generation models, ensuring broad coverage across content, style, and artifacts. To enable scalable and automated evaluation, we annotate a subset of the dataset via human labeling and develop three dedicated multi-task neural assessors: VAQA-Net for aesthetic quality prediction, VTag-Net for automatic aesthetic tagging, and VGQA-Net for generation and basic quality attributes. Extensive experiments demonstrate that our models achieve reliable alignment with human judgments, offering both accuracy and efficiency. We release VGA-Bench as a public benchmark to foster research in AIGC evaluation, with applications in content moderation, model debugging, and generative model optimization. 

---
# MR-Coupler: Automated Metamorphic Test Generation via Functional Coupling Analysis 

**Authors**: Congying Xu, Hengcheng Zhu, Songqiang Chen, Jiarong Wu, Valerio Terragni, Shing-Chi Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2604.10126)  

**Abstract**: Metamorphic testing (MT) is a widely recognized technique for alleviating the oracle problem in software testing. However, its adoption is hindered by the difficulty of constructing effective metamorphic relations (MRs), which often require domain-specific or hard-to-obtain knowledge. In this work, we propose a novel approach that leverages the functional coupling between methods, which is readily available in source code, to automatically construct MRs and generate metamorphic test cases (MTCs). Our technique, MR-Coupler, identifies functionally coupled method pairs, employs large language models to generate candidate MTCs, and validates them through test amplification and mutation analysis. In particular, we leverage three functional coupling features to avoid expensive enumeration of possible method pairs, and a novel validation mechanism to reduce false alarms. Our evaluation of MR-Coupler on 100 human-written MTCs and 50 real-world bugs shows that it generates valid MTCs for over 90% of tasks, improves valid MTC generation by 64.90%, and reduces false alarms by 36.56% compared to baselines. Furthermore, the MTCs generated by MR-Coupler detect 44% of the real bugs. Our results highlight the effectiveness of leveraging functional coupling for automated MR construction and the potential of MR-Coupler to facilitate the adoption of MT in practice. We also released the tool and experimental data to support future research. 

---
# A Dual Cross-Attention Graph Learning Framework For Multimodal MRI-Based Major Depressive Disorder Detection 

**Authors**: Nojod M. Alotaibi, Areej M. Alhothali  

**Link**: [PDF](https://arxiv.org/pdf/2604.10116)  

**Abstract**: Major depressive disorder (MDD) is a prevalent mental disorder associated with complex neurobiological changes that cannot be fully captured using a single imaging modality. The use of multimodal magnetic resonance imaging (MRI) provides a more comprehensive understanding of brain changes by combining structural and functional data. Despite this, the effective integration of these modalities remains challenging. In this study, we propose a dual cross-attention-based multimodal fusion framework that explicitly models bidirectional interactions between structural MRI (sMRI) and resting-state functional MRI (rs-fMRI) representations. The proposed approach is tested on the large-scale REST-meta-MDD dataset using both structural and functional brain atlas configurations. Numerous experiments conducted under a 10-fold stratified cross-validation demonstrated that the proposed fusion algorithm achieves robust and competitive performance across all atlas types. The proposed method consistently outperforms conventional feature-level concatenation for functional atlases, while maintaining comparable performance for structural atlases. The most effective dual cross-attention multimodal model obtained 84.71% accuracy, 86.42% sensitivity, 82.89% specificity, 84.34% precision, and 85.37% F1-score. These findings emphasize the importance of explicitly modeling cross-modal interactions for multimodal neuroimaging-based MDD classification. 

---
# CircuitSynth: Reliable Synthetic Data Generation 

**Authors**: Zehua Cheng, Wei Dai, Jiahao Sun, Thomas Lukasiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2604.10114)  

**Abstract**: The generation of high-fidelity synthetic data is a cornerstone of modern machine learning, yet Large Language Models (LLMs) frequently suffer from hallucinations, logical inconsistencies, and mode collapse when tasked with structured generation. Existing approaches, such as prompting or retrieval-augmented generation, lack the mechanisms to balance linguistic expressivity with formal guarantees regarding validity and coverage. To address this, we propose CircuitSynth, a novel neuro-symbolic framework that decouples semantic reasoning from surface realization. By distilling the reasoning capabilities of a Teacher LLM into a Probabilistic Sentential Decision Diagram (PSDD), CircuitSynth creates a tractable semantic prior that structurally enforces hard logical constraints. Furthermore, we introduce a convex optimization mechanism to rigorously satisfy soft distributional goals. Empirical evaluations across diverse benchmarks demonstrate that CircuitSynth achieves 100% Schema Validity even in complex logic puzzles where unconstrained baselines fail (12.4%) while significantly outperforming state-of-the-art methods in rare-combination coverage. 

---
# Degradation-Consistent Paired Training for Robust AI-Generated Image Detection 

**Authors**: Zongyou Yang, Yinghan Hou, Xiaokun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10102)  

**Abstract**: AI-generated image detectors suffer significant performance degradation under real-world image corruptions such as JPEG compression, Gaussian blur, and resolution downsampling. We observe that state-of-the-art methods, including B-Free, treat degradation robustness as a byproduct of data augmentation rather than an explicit training objective. In this work, we propose Degradation-Consistent Paired Training (DCPT), a simple yet effective training strategy that explicitly enforces robustness through paired consistency constraints. For each training image, we construct a clean view and a degraded view, then impose two constraints: a feature consistency loss that minimizes the cosine distance between clean and degraded representations, and a prediction consistency loss based on symmetric KL divergence that aligns output distributions across views. DCPT adds zero additional parameters and zero inference overhead. Experiments on the Synthbuster benchmark (9 generators, 8 degradation conditions) demonstrate that DCPT improves the degraded-condition average accuracy by 9.1 percentage points compared to an identical baseline without paired training, while sacrificing only 0.9% clean accuracy. The improvement is most pronounced under JPEG compression (+15.7% to +17.9%). Ablation further reveals that adding architectural components leads to overfitting on limited training data, confirming that training objective improvement is more effective than architectural augmentation for degradation robustness. 

---
# MatRes: Zero-Shot Test-Time Model Adaptation for Simultaneous Matching and Restoration 

**Authors**: Kanggeon Lee, Soochahn Lee, Kyoung Mu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.10081)  

**Abstract**: Real-world image pairs often exhibit both severe degradations and large viewpoint changes, making image restoration and geometric matching mutually interfering tasks when treated independently. In this work, we propose MatRes, a zero-shot test-time adaptation framework that jointly improves restoration quality and correspondence estimation using only a single low-quality and high-quality image pair. By enforcing conditional similarity at corresponding locations, MatRes updates only lightweight modules while keeping all pretrained components frozen, requiring no offline training or additional supervision. Extensive experiments across diverse combinations show that MatRes yields significant gains in both restoration and geometric alignment compared to using either restoration or matching models alone. MatRes offers a practical and widely applicable solution for real-world scenarios where users commonly capture multiple images of a scene with varying viewpoints and quality, effectively addressing the often-overlooked mutual interference between matching and restoration. 

---
# Graph-RHO: Critical-path-aware Heterogeneous Graph Network for Long-Horizon Flexible Job-Shop Scheduling 

**Authors**: Yujie Li, Jiuniu Wang, Mugen Peng, Guangzuo Li, Wenjia Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10073)  

**Abstract**: Long-horizon Flexible Job-Shop Scheduling~(FJSP) presents a formidable combinatorial challenge due to complex, interdependent decisions spanning extended time horizons. While learning-based Rolling Horizon Optimization~(RHO) has emerged as a promising paradigm to accelerate solving by identifying and fixing invariant operations, its effectiveness is hindered by the structural complexity of FJSP. Existing methods often fail to capture intricate graph-structured dependencies and ignore the asymmetric costs of prediction errors, in which misclassifying critical-path operations is significantly more detrimental than misclassifying non-critical ones. Furthermore, dynamic shifts in predictive confidence during the rolling process make static pruning thresholds inadequate. To address these limitations, we propose Graph-RHO, a novel critical-path-aware graph-based RHO framework. First, we introduce a topology-aware heterogeneous graph network that encodes subproblems as operation-machine graphs with multi-relational edges, leveraging edge-feature-aware message passing to predict operation stability. Second, we incorporate a critical-path-aware mechanism that injects inductive biases during training to distinguish highly sensitive bottleneck operations from robust ones. Third, we devise an adaptive thresholding strategy that dynamically calibrates decision boundaries based on online uncertainty estimation to align model predictions with the solver's search space. Extensive experiments on standard benchmarks demonstrate that \mbox{Graph-RHO} establishes a new state of the art in solution quality and computational efficiency. Remarkably, it exhibits exceptional zero-shot generalization, reducing solve time by over 30\% on large-scale instances (2000 operations) while achieving superior solution quality. Our code is available \href{this https URL}{here}. 

---
# ASPIRin: Action Space Projection for Interactivity-Optimized Reinforcement Learning in Full-Duplex Speech Language Models 

**Authors**: Chi-Yuan Hsiao, Ke-Han Lu, Yu-Kuan Fu, Guan-Ting Lin, Hsiao-Tsung Hung, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.10065)  

**Abstract**: End-to-end full-duplex Speech Language Models (SLMs) require precise turn-taking for natural interaction. However, optimizing temporal dynamics via standard raw-token reinforcement learning (RL) degrades semantic quality, causing severe generative collapse and repetition. We propose ASPIRin, an interactivity-optimized RL framework that explicitly decouples when to speak from what to say. Using Action Space Projection, ASPIRin maps the text vocabulary into a coarse-grained binary state (active speech vs. inactive silence). By applying Group Relative Policy Optimization (GRPO) with rule-based rewards, it balances user interruption and response latency. Empirical evaluations show ASPIRin optimizes interactivity across turn-taking, backchanneling, and pause handling. Crucially, isolating timing from token selection preserves semantic coherence and reduces the portion of duplicate n-grams by over 50% compared to standard GRPO, effectively eliminating degenerative repetition. 

---
# Computational Implementation of a Model of Category-Theoretic Metaphor Comprehension 

**Authors**: Fumitaka Iwaki, Miho Fuyama, Hayato Saigo, Tatsuji Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2604.10035)  

**Abstract**: In this study, we developed a computational implementation for a model of metaphor comprehension based on the theory of indeterminate natural transformation (TINT) proposed by Fuyama et al. We simplified the algorithms implementing the model to be closer to the original theory and verified it through data fitting and simulations. The outputs of the algorithms are evaluated with three measures: data-fitting with experimental data, the systematicity of the metaphor comprehension result, and the novelty of the comprehension (i.e. the correspondence of the associative structure of the source and target of the metaphor). The improved algorithm outperformed the existing ones in all the three measures. 

---
# Closed-Form Concept Erasure via Double Projections 

**Authors**: Chi Zhang, Jingpu Cheng, Zhixian Wang, Ping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.10032)  

**Abstract**: While modern generative models such as diffusion-based architectures have enabled impressive creative capabilities, they also raise important safety and ethical risks. These concerns have led to growing interest in concept erasure, the process of removing unwanted concepts from model representations. Existing approaches often achieve strong erasure performance but rely on iterative optimization and may inadvertently distort unrelated concepts. In this work, we present a simple yet principled alternative: a linear transformation framework that achieves concept erasure analytically, without any training. Our method adapts a pretrained model through two sequential, closed-form steps: first, computing a proxy projection of the target concept, and second, applying a constrained transformation within the left null space of known concept directions. This design yields a deterministic and geometrically interpretable procedure for safe, efficient, and theory-grounded concept removal. Across a wide range of experiments, including object and style erasure on multiple Stable Diffusion variants and the flow-matching model (FLUX), our approach matches or surpasses the performance of state-of-the-art methods while preserving non-target concepts more faithfully. Requiring only a few seconds to apply, it offers a lightweight and drop-in tool for controlled model editing, advancing the goal of safer and more responsible generative models. 

---
# CoSToM:Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models 

**Authors**: Mengfan Li, Xuanhua Shi, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.10031)  

**Abstract**: Theory of Mind (ToM), the ability to attribute mental states to others, is a hallmark of social intelligence. While large language models (LLMs) demonstrate promising performance on standard ToM benchmarks, we observe that they often fail to generalize to complex task-specific scenarios, relying heavily on prompt scaffolding to mimic reasoning. The critical misalignment between the internal knowledge and external behavior raises a fundamental question: Do LLMs truly possess intrinsic cognition, and can they externalize this internal knowledge into stable, high-quality behaviors? To answer this, we introduce CoSToM (Causal-oriented Steering for ToM alignment), a framework that transitions from mechanistic interpretation to active intervention. First, we employ causal tracing to map the internal distribution of ToM features, empirically uncovering the internal layers' characteristics in encoding fundamental ToM semantics. Building on this insight, we implement a lightweight alignment framework via targeted activation steering within these ToM-critical layers. Experiments demonstrate that CoSToM significantly enhances human-like social reasoning capabilities and downstream dialogue quality. 

---
# LVSum: A Benchmark for Timestamp-Aware Long Video Summarization 

**Authors**: Alkesh Patel, Melis Ozyildirim, Ying-Chang Cheng, Ganesh Nagarajan  

**Link**: [PDF](https://arxiv.org/pdf/2604.10024)  

**Abstract**: Long video summarization presents significant challenges for current multimodal large language models (MLLMs), particularly in maintaining temporal fidelity over extended durations and producing summaries that are both semantically and temporally grounded. In this work, we present LVSum, a human-annotated benchmark designed specifically for evaluating long video summarization with fine-grained temporal alignment. LVSum comprises diverse long-form videos across 13 domains, each paired with human-generated summaries containing precise temporal references. We conduct a comprehensive evaluation of both proprietary and open-source MLLMs on LVSum, assessing performance using newly introduced LLM-based metrics for content relevance and modality coherence, alongside standard evaluation metrics. Our experiments reveal systematic gaps in temporal understanding among existing MLLMs and offer insights that establish a new foundation for advancing temporal reasoning in long video summarization. 

---
# FREE-Switch: Frequency-based Dynamic LoRA Switch for Style Transfer 

**Authors**: Shenghe Zheng, Minyu Zhang, Tianhao Liu, Hongzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.10023)  

**Abstract**: With the growing availability of open-sourced adapters trained on the same diffusion backbone for diverse scenes and objects, combining these pretrained weights enables low-cost customized generation. However, most existing model merging methods are designed for classification or text generation, and when applied to image generation, they suffer from content drift due to error accumulation across multiple diffusion steps. For image-oriented methods, training-based approaches are computationally expensive and unsuitable for edge deployment, while training-free ones use uniform fusion strategies that ignore inter-adapter differences, leading to detail degradation. We find that since different adapters are specialized for generating different types of content, the contribution of each diffusion step carries different significance for each adapter. Accordingly, we propose a frequency-domain importance-driven dynamic LoRA switch method. Furthermore, we observe that maintaining semantic consistency across adapters effectively mitigates detail loss; thus, we design an automatic Generation Alignment mechanism to align generation intents at the semantic level. Experiments demonstrate that our FREE-Switch (Frequency-based Efficient and Dynamic LoRA Switch) framework efficiently combines adapters for different objects and styles, substantially reducing the training cost of high-quality customized generation. 

---
# Demographic and Linguistic Bias Evaluation in Omnimodal Language Models 

**Authors**: Alaa Elobaid  

**Link**: [PDF](https://arxiv.org/pdf/2604.10014)  

**Abstract**: This paper provides a comprehensive evaluation of demographic and linguistic biases in omnimodal language models that process text, images, audio, and video within a single framework. Although these models are being widely deployed, their performance across different demographic groups and modalities is not well studied. Four omnimodal models are evaluated on tasks that include demographic attribute estimation, identity verification, activity recognition, multilingual speech transcription, and language identification. Accuracy differences are measured across age, gender, skin tone, language, and country of origin. The results show that image and video understanding tasks generally exhibit better performance with smaller demographic disparities. In contrast, audio understanding tasks exhibit significantly lower performance and substantial bias, including large accuracy differences across age groups, genders, and languages, and frequent prediction collapse toward narrow categories. These findings highlight the importance of evaluating fairness across all supported modalities as omnimodal language models are increasingly used in real-world applications. 

---
# Like a Hammer, It Can Build, It Can Break: Large Language Model Uses, Perceptions, and Adoption in Cybersecurity Operations on Reddit 

**Authors**: Souradip Nath, Chih-Yi Huang, Aditi Ganapathi, Kashyap Thimmaraju, Jaron Mink, Gail-Joon Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2604.09998)  

**Abstract**: Large language models (LLMs) have recently emerged as promising tools for augmenting Security Operations Center (SOC) workflows, with vendors increasingly marketing autonomous AI solutions for SOCs. However, there remains a limited empirical understanding of how such tools are used, perceived, and adopted by real-world security practitioners. To address this gap, we conduct a mixed-methods analysis of discussions in cybersecurity-focused forums to learn how a diverse group of practitioners use and perceive modern LLM tools for security operations. More specifically, we analyzed 892 posts between December 2022 and September 2025 from three cybersecurity-focused forums on Reddit, and, using a combination of qualitative coding and statistical analysis, examined how security practitioners discuss LLM tools across three dimensions: (1) their stated tools and use cases, (2) the perceived pros and cons of each tool across a set of critical factors, and (3) their adoption of such tools and the expected impacts on the cybersecurity industry and individual analysts. Overall, our findings reveal nuanced patterns in LLM tools adoption, highlighting independent use of LLMs for low-risk, productivity-oriented tasks, alongside active interest around enterprise-grade, security-focused LLM platforms. Although practitioners report meaningful gains in efficiency and effectiveness in LLM-assisted workflows, persistent issues with reliability, verification overheads, and security risks sharply constrain the autonomy granted to LLM tools. Based on these results, we also provide recommendations for developing and adopting LLM tools to ensure the security of organizations and the safety of cybersecurity practitioners. 

---
# Agentic Application in Power Grid Static Analysis: Automatic Code Generation and Error Correction 

**Authors**: Qinjuan Wang, Shan Yang, Yongli Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09995)  

**Abstract**: This paper introduces an LLM agent that automates power grid static analysis by converting natural language into MATPOWER scripts. The framework utilizes DeepSeek-OCR to build an enhanced vector database from MATPOWER manuals. To ensure reliability, it devises a three-tier error-correction system: a static pre-check, a dynamic feedback loop, and a semantic validator. Operating via the Model Context Protocol, the tool enables asynchronous execution and automatically debugging in MATLAB. Experimental results demonstrate that the system achieves a 82.38% accuracy regarding the code fidelity, effectively eliminating hallucinations even in complex analysis tasks. 

---
# FlowPalm: Optical Flow Driven Non-Rigid Deformation for Geometrically Diverse Palmprint Generation 

**Authors**: Yuchen Zou, Huikai Shao, Lihuang Fang, Zhipeng Xiong, Dexing Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2604.09989)  

**Abstract**: Recently, synthetic palmprints have been increasingly used as substitutes for real data to train recognition models. To be effective, such synthetic data must reflect the diversity of real palmprints, including both style variation and geometric variation. However, existing palmprint generation methods mainly focus on style translation, while geometric variation is either ignored or approximated by simple handcrafted augmentations. In this work, we propose FlowPalm, an optical-flow-driven palmprint generation framework capable of simulating the complex non-rigid deformations observed in real palms. Specifically, FlowPalm estimates optical flows between real palmprint pairs to capture the statistical patterns of geometric deformations. Building on these priors, we design a progressive sampling process that gradually introduces the geometric deformations during diffusion while maintaining identity consistency. Extensive experiments on six benchmark datasets demonstrate that FlowPalm significantly outperforms state-of-the-art palmprint generation approaches in downstream recognition tasks. Project page: this https URL 

---
# A Minimal Model of Representation Collapse: Frustration, Stop-Gradient, and Dynamics 

**Authors**: Louie Hong Yao, Yuhao Li, Shengchao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09979)  

**Abstract**: Self-supervised representation learning is central to modern machine learning because it extracts structured latent features from unlabeled data and enables robust transfer across tasks and domains. However, it can suffer from representation collapse, a widely observed failure mode in which embeddings lose discriminative structure and distinct inputs become indistinguishable. To understand the mechanisms that drive collapse and the ingredients that prevent it, we introduce a minimal embedding-only model whose gradient-flow dynamics and fixed points can be analyzed in closed form, using a classification-representation setting as a concrete playground where collapse is directly quantified through the contraction of label-embedding geometry. We illustrate that the model does not collapse when the data are perfectly classifiable, while a small fraction of frustrated samples that cannot be classified consistently induces collapse through an additional slow time scale that follows the early performance gain. Within the same framework, we examine collapse prevention by adding a shared projection head and applying stop-gradient at the level of the training dynamics. We analyze the resulting fixed points and develop a dynamical mean-field style self-consistency description, showing that stop-gradient enables non-collapsed solutions and stabilizes finite class separation under frustration. We further verify empirically that the same qualitative dynamics and collapse-prevention effects appear in a linear teacher-student model, indicating that the minimal theory captures features that persist beyond the pure embedding setting. 

---
# Muon$^2$: Boosting Muon via Adaptive Second-Moment Preconditioning 

**Authors**: Ziyue Liu, Ruijie Zhang, Zhengyang Wang, Yequan Zhao, Yupeng Su, Zi Yang, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09967)  

**Abstract**: Muon has emerged as a promising optimizer for large-scale foundation model pre-training by exploiting the matrix structure of neural network updates through iterative orthogonalization. However, its practical efficiency is limited by the need for multiple Newton--Schulz (NS) iterations per optimization step, which introduces non-trivial computation and communication overhead. We propose Muon$^2$, an extension of Muon that applies Adam-style adaptive second-moment preconditioning before orthogonalization. Our key insight is that the core challenge of polar approximation in Muon lies in the ill-conditioned momentum matrix, of which the spectrum is substantially improved by Muon$^2$, leading to faster convergence toward a practically sufficient orthogonalization. We further characterize the practical orthogonalization quality via directional alignment, under which Muon$^2$ demonstrates dramatic improvement over Muon at each polar step. Across GPT and LLaMA pre-training experiments from 60M to 1.3B parameters, Muon$^2$ consistently outperforms Muon and recent Muon variants while reducing NS iterations by 40\%. We further introduce Muon$^2$-F, a memory-efficient factorized variant that preserves most of the gains of Muon$^2$ with negligible memory overhead. 

---
# Rebooting Microreboot: Architectural Support for Safe, Parallel Recovery in Microservice Systems 

**Authors**: Laurent Bindschaedler  

**Link**: [PDF](https://arxiv.org/pdf/2604.09963)  

**Abstract**: Microreboot enables fast recovery by restarting only the failing component, but in modern microservices naive restarts are unsafe: dense dependencies mean rebooting one service can disrupt many callers. Autonomous remediation agents compound this by actuating raw infrastructure commands without safety guarantees. We make microreboot practical by separating planning from actuation: a three-agent architecture (diagnosis, planning, verification) proposes typed remediation plans over a seven-action ISA with explicit side-effect semantics, and a small microkernel validates and executes each plan transactionally. Agents are explicitly untrusted; safety derives from the ISA and microkernel. To determine where restart is safe, we infer recovery boundaries online from distributed traces, computing minimal restart groups and ordering constraints. On industrial traces (Alibaba, Meta) and DeathStarBench with fault injection, recovery-group inference runs in 21 ms at P99; typed actuation reduces agent-caused harm by 95% in simulation and achieves 0% harm online. The primary value is safety, not speed: LLM inference overhead increases TTR for services with fast auto-restart. 

---
# Cross-Cultural Value Awareness in Large Vision-Language Models 

**Authors**: Phillip Howard, Xin Su, Kathleen C. Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2604.09945)  

**Abstract**: The rapid adoption of large vision-language models (LVLMs) in recent years has been accompanied by growing fairness concerns due to their propensity to reinforce harmful societal stereotypes. While significant attention has been paid to such fairness concerns in the context of social biases, relatively little prior work has examined the presence of stereotypes in LVLMs related to cultural contexts such as religion, nationality, and socioeconomic status. In this work, we aim to narrow this gap by investigating how cultural contexts depicted in images influence the judgments LVLMs make about a person's moral, ethical, and political values. We conduct a multi-dimensional analysis of such value judgments in five popular LVLMs using counterfactual image sets, which depict the same person across different cultural contexts. Our evaluation framework diagnoses LVLM awareness of cultural value differences through the use of Moral Foundations Theory, lexical analyses, and the sensitivity of generated values to depicted cultural contexts. 

---
# I Walk the Line: Examining the Role of Gestalt Continuity in Object Binding for Vision Transformers 

**Authors**: Alexa R. Tartaglini, Michael A. Lepori  

**Link**: [PDF](https://arxiv.org/pdf/2604.09942)  

**Abstract**: Object binding is a foundational process in visual cognition, during which low-level perceptual features are joined into object representations. Binding has been considered a fundamental challenge for neural networks, and a major milestone on the way to artificial models with flexible visual intelligence. Recently, several investigations have demonstrated evidence that binding mechanisms emerge in pretrained vision models, enabling them to associate portions of an image that contain an object. The question remains: how are these models binding objects together? In this work, we investigate whether vision models rely on the principle of Gestalt continuity to perform object binding, over and above other principles like similarity and proximity. Using synthetic datasets, we demonstrate that binding probes are sensitive to continuity across a wide range of pretrained vision transformers. Next, we uncover particular attention heads that track continuity, and show that these heads generalize across datasets. Finally, we ablate these attention heads, and show that they often contribute to producing representations that encode object binding. 

---
# A Hybrid Intelligent Framework for Uncertainty-Aware Condition Monitoring of Industrial Systems 

**Authors**: Maryam Ahang, Todd Charter, Masoud Jalayer, Homayoun Najjaran  

**Link**: [PDF](https://arxiv.org/pdf/2604.09932)  

**Abstract**: Hybrid approaches that combine data-driven learning with physics-based insight have shown promise for improving the reliability of industrial condition monitoring. This work develops a hybrid condition monitoring framework that integrates primary sensor measurements, lagged temporal features, and physics-informed residuals derived from nominal surrogate models. Two hybrid integration strategies are examined. The first is a feature-level fusion approach that augments the input space with residual and temporal information. The second is a model-level ensemble approach in which machine learning classifiers trained on different feature types are combined at the decision level. Both hybrid approaches of the condition monitoring framework are evaluated on a continuous stirred-tank reactor (CSTR) benchmark using several machine learning models and ensemble configurations. Both feature-level and model-level hybridization improve diagnostic accuracy relative to single-source baselines, with the best model-level ensemble achieving a 2.9\% improvement over the best baseline ensemble. To assess predictive reliability, conformal prediction is applied to quantify coverage, prediction-set size, and abstention behavior. The results show that hybrid integration enhances uncertainty management, producing smaller and well-calibrated prediction sets at matched coverage levels. These findings demonstrate that lightweight physics-informed residuals, temporal augmentation, and ensemble learning can be combined effectively to improve both accuracy and decision reliability in nonlinear industrial systems. 

---
# The Rise and Fall of $G$ in AGI 

**Authors**: David C. Krakauer  

**Link**: [PDF](https://arxiv.org/pdf/2604.09911)  

**Abstract**: In the psychological literature the term `general intelligence' describes correlations between abilities and not simply the number of abilities. This paper connects Spearman's $g$-factor from psychometrics, measuring a positive manifold, to the implicit ``$G$-factor'' in claims about artificial general intelligence (AGI) performance on temporally structured benchmarks. By treating LLM benchmark batteries as cognitive test batteries and model releases as subjects, principal component analysis is applied to a models $\times$ benchmarks $\times$ time matrix spanning 39 models (2019--2025) and 14 benchmarks. Preliminary results confirm a strong positive manifold in which all 28 pairwise correlations positive across 8 benchmarks. By analyzing the spectrum of the benchmark correlation through time, PC1 explains 90\% of variance on a 5-benchmark core battery ($n=19$)) reducing to 77\% by 2024. On a four benchmark battery, PC1 is found to peak at 92\% of the variance between 2023--2024 and reduce to 64\% with the arrival of reasoning-specialized models in 2024. This is coincident with a rotation in the G-factor as models outsource `reasoning' to tools. The analysis of partial correlation matrices through time provides evidence for the evolution of specialization beneath the positive manifold of general intelligence (AI-hedgehog) encompassing diverse high dimensional problem solving systems (AI-foxes). In strictly psychometric terms, AI models exhibit general intelligence suppressing specialized intelligences. LLMs invert the ideal of substituting complicated models with parsimonious mechanisms, a `Ptolemaic Succession' of theories, with architectures of increasing hierarchical complication and capability. 

---
# From UAV Imagery to Agronomic Reasoning: A Multimodal LLM Benchmark for Plant Phenotyping 

**Authors**: Yu Wu, Guangzeng Han, Ibra Niang Niang, Francia Ravelombola, Maiara Oliveira, Jason Davis, Dong Chen, Feng Lin, Xiaolei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09907)  

**Abstract**: To improve crop genetics, high-throughput, effective and comprehensive phenotyping is a critical prerequisite. While such tasks were traditionally performed manually, recent advances in multimodal foundation models, especially in vision-language models (VLMs), have enabled more automated and robust phenotypic analysis. However, plant science remains a particularly challenging domain for foundation models because it requires domain-specific knowledge, fine-grained visual interpretation, and complex biological and agronomic reasoning. To address this gap, we develop PlantXpert, an evidence-grounded multimodal reasoning benchmark for soybean and cotton phenotyping. Our benchmark provides a structured and reproducible framework for agronomic adaptation of VLMs, and enables controlled comparison between base models and their domain-adapted counterparts. We constructed a dataset comprising 385 digital images and more than 3,000 benchmark samples spanning key plant science domains including disease, pest control, weed management, and yield. The benchmark can assess diverse capabilities including visual expertise, quantitative reasoning, and multi-step agronomic reasoning. A total of 11 state-of-the-art VLMs were evaluated. The results indicate that task-specific fine-tuning leads to substantial improvement in accuracy, with models such as Qwen3-VL-4B and Qwen3-VL-30B achieving up to 78%. At the same time, gains from model scaling diminish beyond a certain capacity, generalization across soybean and cotton remains uneven, and quantitative as well as biologically grounded reasoning continue to pose substantial challenges. These findings suggest that PlantXpert can serve as a foundation for assessing evidence-grounded agronomic reasoning and for advancing multimodal model development in plant science. 

---
# Diffusion Denoiser Achievable Analysis for Finite Blocklength Unsourced Random Access 

**Authors**: Yuming Han, Yuxin Long  

**Link**: [PDF](https://arxiv.org/pdf/2604.09904)  

**Abstract**: Polyanskiy proposed a framework for the unsourced multiple access channel (MAC) problem where users employ a common codebook in the finite blocklength regime. However, existing approaches handle channel noise before the joint decoder. In this work, we introduce a decoder compatible diffusion denoiser as a lightweight analysis within joint decoding. The score network is trained on samples drawn from the channel output distribution, making the method easy to integrate with existing code designs. In our theoretical analysis, we derive a diffusion-denoiser random-coding achievable bound that is strictly tighter. Simulations on existing decoders, including FASURA, MSUG-MRA and pilot-based method, show consistent performance gains with at least a $0.5$ $\mathrm{dB}$ improvement in required $\mathrm{E_b/N_0}$ at a fixed error target. 

---
# Should We be Pedantic About Reasoning Errors in Machine Translation? 

**Authors**: Calvin Bao, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2604.09890)  

**Abstract**: Across multiple language pairings (English $\to$ \{Spanish, French, German, Mandarin, Japanese, Urdu, Cantonese\}), we find reasoning errors in translation. To quantify how often these reasoning errors occur, we leverage an automated annotation protocol for reasoning evaluation wherein the goal is to detect if a reasoning step is any of three error categories: (1) source sentence-misaligned, (2) model hypothesis-misaligned, or (3) reasoning trace-misaligned. We probe the reasoning model with perturbed traces correcting for these identified reasoning errors using an array of weak-to-strong interventions: hedging, removal, re-reasoning after removal, hindsight, and oracle interventions. Experimenting with interventions on the reasoning traces suggests that small corrections to the reasoning have little impact on translation quality, but stronger interventions yield the highest resolution rates, despite translation quality gains being mixed. We find ultimately that reasoning errors in MT can be identified with high precision in Urdu but lower precision in Spanish, but that removing these reasoning errors does not resolve the initial errors significantly, suggesting limited reasoning faithfulness for machine translation. 

---
# Not Your Stereo-Typical Estimator: Combining Vision and Language for Volume Perception 

**Authors**: Gautham Vinod, Bruce Coburn, Siddeshwar Raghavan, Fengqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09886)  

**Abstract**: Accurate volume estimation of objects from visual data is a long-standing challenge in computer vision with significant applications in robotics, logistics, and smart health. Existing methods often rely on complex 3D reconstruction pipelines or struggle with the ambiguity inherent in single-view images. To address these limitations, we introduce a new method that fuses implicit 3D cues from stereo vision with explicit prior knowledge from natural language text. Our approach extracts deep features from a stereo image pair and a descriptive text prompt that contains the object's class and an approximate volume, then integrates them using a simple yet effective projection layer into a unified, multi-modal representation for regression. We conduct extensive experiments on public datasets demonstrating that our text-guided approach significantly outperforms vision-only baselines. Our findings show that leveraging even simple textual priors can effectively guide the volume estimation task, paving the way for more context-aware visual measurement systems. Code: this https URL. 

---
# DINO_4D: Semantic-Aware 4D Reconstruction 

**Authors**: Yiru Yang, Zhuojie Wu, Quentin Marguet, Nishant Kumar Singh, Max Schulthess  

**Link**: [PDF](https://arxiv.org/pdf/2604.09877)  

**Abstract**: In the intersection of computer vision and robotic perception, 4D reconstruction of dynamic scenes serve as the critical bridge connecting low-level geometric sensing with high-level semantic understanding. We present DINO\_4D, introducing frozen DINOv3 features as structural priors, injecting semantic awareness into the reconstruction process to effectively suppress semantic drift during dynamic tracking. Experiments on the Point Odyssey and TUM-Dynamics benchmarks demonstrate that our method maintains the linear time complexity $O(T)$ of its predecessors while significantly improving Tracking Accuracy (APD) and Reconstruction Completeness. DINO\_4D establishes a new paradigm for constructing 4D World Models that possess both geometric precision and semantic understanding. 

---
# Efficient Personalization of Generative User Interfaces 

**Authors**: Yi-Hao Peng, Samarth Das, Jeffrey P. Bigham, Jason Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09876)  

**Abstract**: Generative user interfaces (UIs) create new opportunities to adapt interfaces to individual users on demand, but personalization remains difficult because desirable UI properties are subjective, hard to articulate, and costly to infer from sparse feedback. We study this problem through a new dataset in which 20 trained designers each provide pairwise judgments over the same 600 generated UIs, enabling direct analysis of preference divergence. We find substantial disagreement across designers (average kappa = 0.25), and written rationales reveal that even when designers appeal to similar concepts such as hierarchy or cleanliness, designers differ in how they define, prioritize, and apply those concepts. Motivated by these findings, we develop a sample-efficient personalization method that represents a new user in terms of prior designers rather than a fixed rubric of design concepts. In a technical evaluation, our preference model outperforms both a pretrained UI evaluator and a larger multimodal model, and scales better with additional feedback. When used to personalize generation, it also produces interfaces preferred by 12 new designers over baseline approaches, including direct user prompting. Our findings suggest that lightweight preference elicitation can serve as a practical foundation for personalized generative UI systems. 

---
# Relational Preference Encoding in Looped Transformer Internal States 

**Authors**: Jan Kirin  

**Link**: [PDF](https://arxiv.org/pdf/2604.09870)  

**Abstract**: We investigate how looped transformers encode human preference in their internal iteration states. Using Ouro-2.6B-Thinking, a 2.6B-parameter looped transformer with iterative refinement, we extract hidden states from each loop iteration and train lightweight evaluator heads (~5M parameters) to predict human preference on the Anthropic HH-RLHF dataset. Our pairwise evaluator achieves 95.2% test accuracy on 8,552 unseen examples, surpassing a full-batch L-BFGS probe (84.5%) while the base model remains completely frozen.
Our central finding is that loop states encode preference predominantly relationally: a linear probe on pairwise differences achieves 84.5%, the best nonlinear independent evaluator reaches only 65% test accuracy, and linear independent classification scores 21.75%, below chance and with inverted polarity. Interpreted precisely, the evaluator functions as a model-internal consistency probe, measuring how stably Ouro's own learned value system organizes its representations rather than how well it predicts noisy human annotations.
We also document a systematic architecture search that established a genuine 70% ceiling for independent scoring, and show how the 50% argument-swap protocol required to prevent degenerate pairwise solutions deflated pairwise training metrics by about 31 points at peak, creating the false appearance that pairwise and pointwise evaluators shared the same ceiling.
Finally, we show that a cosine learning-rate dead zone at epoch 2 accidentally acted as early stopping, preserving the generalization peak before overfitting degraded test accuracy from 95.2% to 62.4% by epoch 5. Cross-epoch flip-test analysis shows that antisymmetry correlation remains stable while strict sign-flip rate mainly tracks scorer bias. We propose the flip test as a mandatory diagnostic for pairwise preference evaluators. 

---
# Exploring Structural Complexity in Normative RAG with Graph-based approaches: A case study on the ETSI Standards 

**Authors**: Aiman Al Masoud, Marco Arazzi, Simone Germani, Antonino Nocera  

**Link**: [PDF](https://arxiv.org/pdf/2604.09868)  

**Abstract**: Industrial standards and normative documents exhibit intricate hierarchical structures, domain-specific lexicons, and extensive cross-referential dependencies, which making it challenging to process them directly by Large Language Models (LLMs).
While Retrieval-Augmented Generation (RAG) provides a computationally efficient alternative to LLM fine-tuning, standard "vanilla" vector-based retrieval may fail to capture the latent structural and relational features intrinsic in normative documents.
With the objective of shedding light on the most promising technique for building high-performance RAG solutions for normative, standards, and regulatory documents, this paper investigates the efficacy of Graph RAG architectures, which represent information as interconnected nodes, thus moving from simple semantic similarity toward a more robust, relation-aware retrieval mechanism.
Despite the promise of graph-based techniques, there is currently a lack of empirical evidence as to which is the optimal indexing strategy for technical standards. Therefore, to help solve this knowledge gap, we propose a specialized RAG methodology tailored to the unique structure and lexical characteristics of standards and regulatory documents.
Moreover, to keep our investigation grounded, we focus on well-known public standards, such as the ETSI EN 301 489 series. We evaluate several lightweight and low-latency strategies designed to embed document structure directly into the retrieval workflow.
The considered approaches are rigorously tested against a custom synthesized Q&A dataset, facilitating a quantitative performance analysis. Our experimental results demonstrate that the incorporation of structural and lexical information into the index can enhance, at least to some extent, retrieval performance, providing a scalable framework for automated normative and standards elaboration. 

---
# Automating Structural Analysis Across Multiple Software Platforms Using Large Language Models 

**Authors**: Ziheng Geng, Jiachen Liu, Ian Franklin, Ran Cao, Dan M. Frangopol, Minghui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.09866)  

**Abstract**: Recent advances in large language models (LLMs) have shown the promise to significantly accelerate the workflow by automating structural modeling and analysis. However, existing studies primarily focus on enabling LLMs to operate a single structural analysis software platform. In practice, structural engineers often rely on multiple finite element analysis (FEA) tools, such as ETABS, SAP2000, and OpenSees, depending on project needs, user preferences, and company constraints. This limitation restricts the practical deployment of LLM-assisted engineering workflows. To address this gap, this study develops LLMs capable of automating frame structural analysis across multiple software platforms. The LLMs adopt a two-stage multi-agent architecture. In Stage 1, a cohort of agents collaboratively interpret user input and perform structured reasoning to infer geometric, material, boundary, and load information required for finite element modeling. The outputs of these agents are compiled into a unified JSON representation. In Stage 2, code translation agents operate in parallel to convert the JSON file into executable scripts across multiple structural analysis platforms. Each agent is prompted with the syntax rules and modeling workflows of its target software. The LLMs are evaluated using 20 representative frame problems across three widely used platforms: ETABS, SAP2000, and OpenSees. Results from ten repeated trials demonstrate consistently reliable performance, achieving accuracy exceeding 90% across all cases. 

---
# PAS: Estimating the target accuracy before domain adaptation 

**Authors**: Raphaella Diniz, Jackson de Faria, Martin Ester  

**Link**: [PDF](https://arxiv.org/pdf/2604.09863)  

**Abstract**: The goal of domain adaptation is to make predictions for unlabeled samples from a target domain with the help of labeled samples from a different but related source domain. The performance of domain adaptation methods is highly influenced by the choice of source domain and pre-trained feature extractor. However, the selection of source data and pre-trained model is not trivial due to the absence of a labeled validation set for the target domain and the large number of available pre-trained models. In this work, we propose PAS, a novel score designed to estimate the transferability of a source domain set and a pre-trained feature extractor to a target classification task before actually performing domain adaptation. PAS leverages the generalization power of pre-trained models and assesses source-target compatibility based on the pre-trained feature embeddings. We integrate PAS into a framework that indicates the most relevant pre-trained model and source domain among multiple candidates, thus improving target accuracy while reducing the computational overhead. Extensive experiments on image classification benchmarks demonstrate that PAS correlates strongly with actual target accuracy and consistently guides the selection of the best-performing pre-trained model and source domain for adaptation. 

---
# RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies 

**Authors**: Xuning Yang, Rishit Dagli, Alex Zook, Hugo Hadfield, Ankit Goyal, Stan Birchfield, Fabio Ramos, Jonathan Tremblay  

**Link**: [PDF](https://arxiv.org/pdf/2604.09860)  

**Abstract**: The pursuit of general-purpose robotics has yielded impressive foundation models, yet simulation-based benchmarking remains a bottleneck due to rapid performance saturation and a lack of true generalization testing. Existing benchmarks often exhibit significant domain overlap between training and evaluation, trivializing success rates and obscuring insights into robustness. We introduce RoboLab, a simulation benchmarking framework designed to address these challenges. Concretely, our framework is designed to answer two questions: (1) to what extent can we understand the performance of a real-world policy by analyzing its behavior in simulation, and (2) which external factors most strongly affect that behavior under controlled perturbations. First, RoboLab enables human-authored and LLM-enabled generation of scenes and tasks in a robot- and policy-agnostic manner within a physically realistic and photorealistic simulation. With this, we propose the RoboLab-120 benchmark, consisting of 120 tasks categorized into three competency axes: visual, procedural, relational competency, across three difficulty levels. Second, we introduce a systematic analysis of real-world policies that quantify both their performance and the sensitivity of their behavior to controlled perturbations, indicating that high-fidelity simulation can serve as a proxy for analyzing performance and its dependence on external factors. Evaluation with RoboLab exposes significant performance gap in current state-of-the-art models. By providing granular metrics and a scalable toolset, RoboLab offers a scalable framework for evaluating the true generalization capabilities of task-generalist robotic policies. 

---
# Is There Knowledge Left to Extract? Evidence of Fragility in Medically Fine-Tuned Vision-Language Models 

**Authors**: Oliver McLaughlin, Daniel Shubin, Carsten Eickhoff, Ritambhara Singh, William Rudman, Michal Golovanevsky  

**Link**: [PDF](https://arxiv.org/pdf/2604.09841)  

**Abstract**: Vision-language models (VLMs) are increasingly adapted through domain-specific fine-tuning, yet it remains unclear whether this improves reasoning beyond superficial visual cues, particularly in high-stakes domains like medicine. We evaluate four paired open-source VLMs (LLaVA vs. LLaVA-Med; Gemma vs. MedGemma) across four medical imaging tasks of increasing difficulty: brain tumor, pneumonia, skin cancer, and histopathology classification. We find that performance degrades toward near-random levels as task difficulty increases, indicating limited clinical reasoning. Medical fine-tuning provides no consistent advantage, and models are highly sensitive to prompt formulation, with minor changes causing large swings in accuracy and refusal rates. To test whether closed-form VQA suppresses latent knowledge, we introduce a description-based pipeline where models generate image descriptions that a text-only model (GPT-5.1) uses for diagnosis. This recovers a limited additional signal but remains bounded by task difficulty. Analysis of vision encoder embeddings further shows that failures stem from both weak visual representations and downstream reasoning. Overall, medical VLM performance is fragile, prompt-dependent, and not reliably improved by domain-specific fine-tuning. 

---
# F3G-Avatar : Face Focused Full-body Gaussian Avatar 

**Authors**: Willem Menu, Erkut Akdag, Pedro Quesado, Yasaman Kashefbahrami, Egor Bondarev  

**Link**: [PDF](https://arxiv.org/pdf/2604.09835)  

**Abstract**: Existing full-body Gaussian avatar methods primarily optimize global reconstruction quality and often fail to preserve fine-grained facial geometry and expression details. This challenge arises from limited facial representational capacity that causes difficulties in modeling high-frequency pose-dependent deformations. To address this, we propose F3G-Avatar, a full-body, face-aware avatar synthesis method that reconstructs animatable human representations from multi-view RGB video and regressed pose/shape parameters. Starting from a clothed Momentum Human Rig (MHR) template, front/back positional maps are rendered and decoded into 3D Gaussians through a two-branch architecture: a body branch that captures pose-dependent non-rigid deformations and a face-focused deformation branch that refines head geometry and appearance. The predicted Gaussians are fused, posed with linear blend skinning (LBS), and rendered with differentiable Gaussian splatting. Training combines reconstruction and perceptual objectives with a face-specific adversarial loss to enhance realism in close-up views. Experiments demonstrate strong rendering quality, with face-view performance reaching PSNR/SSIM/LPIPS of 26.243/0.964/0.084 on the AvatarReX dataset. Ablations further highlight contributions of the MHR template and the face-focused deformation. F3G-Avatar provides a practical, high-quality pipeline for realistic, animatable full-body avatar synthesis. 

---
# ACCIDENT: A Benchmark Dataset for Vehicle Accident Detection from Traffic Surveillance Videos 

**Authors**: Lukas Picek, Michal Čermák, Marek Hanzl, Vojtěch Čermák  

**Link**: [PDF](https://arxiv.org/pdf/2604.09819)  

**Abstract**: We introduce ACCIDENT, a benchmark dataset for traffic accident detection in CCTV footage, designed to evaluate models in supervised (IID and OOD) and zero-shot settings, reflecting both data-rich and data-scarce scenarios. The benchmark consists of a curated set of 2,027 real and 2,211 synthetic clips annotated with the accident time, spatial location, and high-level collision type. We define three core tasks: (i) temporal localization of the accident, (ii) its spatial localization, and (iii) collision type classification. Each task is evaluated using custom metrics that account for the uncertainty and ambiguity inherent in CCTV footage. In addition to the benchmark, we provide a diverse set of baselines, including heuristic, motion-aware, and vision-language approaches, and show that ACCIDENT is challenging. You can access the ACCIDENT at: this https URL 

---
# Explainable Human Activity Recognition: A Unified Review of Concepts and Mechanisms 

**Authors**: Mainak Kundu, Catherine Chen, Rifatul Islam, Ismail Uysal, Ria Kanjilal  

**Link**: [PDF](https://arxiv.org/pdf/2604.09799)  

**Abstract**: Human activity recognition (HAR) has become a key component of intelligent systems for healthcare monitoring, assistive living, smart environments, and human-computer interaction. Although deep learning has substantially improved HAR performance on multivariate sensor data, the resulting models often remain opaque, limiting trust, reliability, and real-world deployment. Explainable artificial intelligence (XAI) has therefore emerged as a critical direction for making HAR systems more transparent and human-centered. This paper presents a comprehensive review of explainable HAR methods across wearable, ambient, physiological, and multimodal sensing settings. We introduce a unified perspective that separates conceptual dimensions of explainability from algorithmic explanation mechanisms, reducing ambiguities in prior surveys. Building on this distinction, we present a mechanism-centric taxonomy of XAI-HAR methods covering major explanation paradigms. The review examines how these methods address the temporal, multimodal, and semantic complexities of HAR, and summarize their interpretability objectives, explanation targets, and limitations. In addition, we discuss current evaluation practices, highlight key challenges in achieving reliable and deployable XAI-HAR, and outline directions toward trustworthy activity recognition systems that better support human understanding and decision-making. 

---
# GIANTS: Generative Insight Anticipation from Scientific Literature 

**Authors**: Joy He-Yueya, Anikait Singh, Ge Gao, Michael Y. Li, Sherry Yang, Chelsea Finn, Emma Brunskill, Noah D. Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2604.09793)  

**Abstract**: Scientific breakthroughs often emerge from synthesizing prior ideas into novel contributions. While language models (LMs) show promise in scientific discovery, their ability to perform this targeted, literature-grounded synthesis remains underexplored. We introduce insight anticipation, a generation task in which a model predicts a downstream paper's core insight from its foundational parent papers. To evaluate this capability, we develop GiantsBench, a benchmark of 17k examples across eight scientific domains, where each example consists of a set of parent papers paired with the core insight of a downstream paper. We evaluate models using an LM judge that scores similarity between generated and ground-truth insights, and show that these similarity scores correlate with expert human ratings. Finally, we present GIANTS-4B, an LM trained via reinforcement learning (RL) to optimize insight anticipation using these similarity scores as a proxy reward. Despite its smaller open-source architecture, GIANTS-4B outperforms proprietary baselines and generalizes to unseen domains, achieving a 34% relative improvement in similarity score over gemini-3-pro. Human evaluations further show that GIANTS-4B produces insights that are more conceptually clear than those of the base model. In addition, SciJudge-30B, a third-party model trained to compare research abstracts by likely citation impact, predicts that insights generated by GIANTS-4B are more likely to lead to higher citations, preferring them over the base model in 68% of pairwise comparisons. We release our code, benchmark, and model to support future research in automated scientific discovery. 

---
# MedLVR: Latent Visual Reasoning for Reliable Medical Visual Question Answering 

**Authors**: Suyang Xi, Songtao Hu, Yuxiang Lai, Wangyun Dan, Yaqi Liu, Shansong Wang, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09757)  

**Abstract**: Medical vision--language models (VLMs) have shown strong potential for medical visual question answering (VQA), yet their reasoning remains largely text-centric: images are encoded once as static context, and subsequent inference is dominated by language. This paradigm is fundamentally limited in clinical scenarios, where accurate answers often depend on subtle, localized visual evidence that cannot be reliably preserved in static embeddings. We propose \textsc{MedLVR}, a latent visual reasoning framework that introduces an explicit visual evidence state into autoregressive decoding. Instead of relying solely on text-based intermediate reasoning, \textsc{MedLVR} interleaves a short latent reasoning segment within the decoder by reusing hidden states as continuous latent steps, enabling iterative preservation and refinement of query-relevant visual evidence before answer generation. To support effective visual supervision, we adopt a two-stage training strategy: region of interest (ROI)-supervised fine-tuning aligns latent states with clinically relevant image evidence, and Visual-Latent Policy Optimization (VLPO) further optimizes latent reasoning and answer generation under outcome-level rewards. Experiments on OmniMedVQA and five external medical VQA benchmarks show that \textsc{MedLVR} consistently outperforms recent reasoning baselines and improves the average score over the Qwen2.5-VL-7B backbone from 48.3\% to 53.4\%. These results show that latent visual reasoning provides an effective mechanism for preserving diagnostically relevant visual evidence and improving the reliability of medical VQA. 

---
# A-IO: Adaptive Inference Orchestration for Memory-Bound NPUs 

**Authors**: Chen Zhang, Yan Ding, Haotian Wang, Chubo Liu, Keqin Li, Kenli Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09752)  

**Abstract**: During the deployment of Large Language Models (LLMs), the autoregressive decoding phase on heterogeneous NPU platforms (e.g., Ascend 910B) faces severe memory-bound challenges. This study reveals the ``Model Scaling Paradox'' caused by the static deployment of single-sized models. It also points out the kernel synchronization overhead of fine-grained speculative decoding \cite{leviathan2023fast, chen2023speculative} under NPU computational graph compilation, and the severe limitations of purely relying on micro-level acceleration algorithms like Prompt LookUp Decoding (PLD) 

---
# Conflicts Make Large Reasoning Models Vulnerable to Attacks 

**Authors**: Honghao Liu, Chengjin Xu, Xuhui Jiang, Cehao Yang, Shengming Yin, Zhengwu Ma, Lionel Ni, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.09750)  

**Abstract**: Large Reasoning Models (LRMs) have achieved remarkable performance across diverse domains, yet their decision-making under conflicting objectives remains insufficiently understood. This work investigates how LRMs respond to harmful queries when confronted with two categories of conflicts: internal conflicts that pit alignment values against each other and dilemmas, which impose mutually contradictory choices, including sacrificial, duress, agent-centered, and social forms. Using over 1,300 prompts across five benchmarks, we evaluate three representative LRMs - Llama-3.1-Nemotron-8B, QwQ-32B, and DeepSeek R1 - and find that conflicts significantly increase attack success rates, even under single-round non-narrative queries without sophisticated auto-attack techniques. Our findings reveal through layerwise and neuron-level analyses that safety-related and functional representations shift and overlap under conflict, interfering with safety-aligned behavior. This study highlights the need for deeper alignment strategies to ensure the robustness and trustworthiness of next-generation reasoning models. Our code is available at this https URL. Warning: This paper contains inappropriate, offensive and harmful content. 

---
# Backdoors in RLVR: Jailbreak Backdoors in LLMs From Verifiable Reward 

**Authors**: Weiyang Guo, Zesheng Shi, Zeen Zhu, Yuan Zhou, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09748)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is an emerging paradigm that significantly boosts a Large Language Model's (LLM's) reasoning abilities on complex logical tasks, such as mathematics and programming. However, we identify, for the first time, a latent vulnerability to backdoor attacks within the RLVR framework. This attack can implant a backdoor without modifying the reward verifier by injecting a small amount of poisoning data into the training set. Specifically, we propose a novel trigger mechanism designated as the \ourapproach (ACB). The attack exploits the RLVR training loop by assigning substantial positive rewards for harmful responses and negative rewards for refusals. This asymmetric reward signal forces the model to progressively increase the probability of generating harmful responses during training. Our findings demonstrate that the RLVR backdoor attack is characterized by both high efficiency and strong generalization capabilities. Utilizing less than 2\% poisoned data in train set, the backdoor can be successfully implanted across various model scales without degrading performance on benign tasks. Evaluations across multiple jailbreak benchmarks indicate that activating the trigger degrades safety performance by an average of 73\%. Furthermore, the attack generalizes effectively to a wide range of jailbreak methods and unsafe behaviors. Code is available at this https URL. 

---
# ADAM: A Systematic Data Extraction Attack on Agent Memory via Adaptive Querying 

**Authors**: Xingyu Lyu, Jianfeng He, Ning Wang, Yidan Hu, Tao Li, Danjue Chen, Shixiong Li, Yimin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.09747)  

**Abstract**: Large Language Model (LLM) agents have achieved rapid adoption and demonstrated remarkable capabilities across a wide range of applications. To improve reasoning and task execution, modern LLM agents would incorporate memory modules or retrieval-augmented generation (RAG) mechanisms, enabling them to further leverage prior interactions or external knowledge. However, such a design also introduces a group of critical privacy vulnerabilities: sensitive information stored in memory can be leaked through query-based attacks. Although feasible, existing attacks often achieve only limited performance, with low attack success rates (ASR). In this paper, we propose ADAM, a novel privacy attack that features data distribution estimation of a victim agent's memory and employs an entropy-guided query strategy for maximizing privacy leakage. Extensive experiments demonstrate that our attack substantially outperforms state-of-the-art ones, achieving up to 100% ASRs. These results thus underscore the urgent need for robust privacy-preserving methods for current LLM agents. 

---
# CONSCIENTIA: Can LLM Agents Learn to Strategize? Emergent Deception and Trust in a Multi-Agent NYC Simulation 

**Authors**: Aarush Sinha, Arion Das, Soumyadeep Nag, Charan Karnati, Shravani Nag, Chandra Vadhan Raj, Aman Chadha, Vinija Jain, Suranjana Trivedy, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2604.09746)  

**Abstract**: As large language models (LLMs) are increasingly deployed as autonomous agents, understanding how strategic behavior emerges in multi-agent environments has become an important alignment challenge. We take a neutral empirical stance and construct a controlled environment in which strategic behavior can be directly observed and measured. We introduce a large-scale multi-agent simulation in a simplified model of New York City, where LLM-driven agents interact under opposing incentives. Blue agents aim to reach their destinations efficiently, while Red agents attempt to divert them toward billboard-heavy routes using persuasive language to maximize advertising revenue. Hidden identities make navigation socially mediated, forcing agents to decide when to trust or deceive. We study policy learning through an iterative simulation pipeline that updates agent policies across repeated interaction rounds using Kahneman-Tversky Optimization (KTO). Blue agents are optimized to reduce billboard exposure while preserving navigation efficiency, whereas Red agents adapt to exploit remaining weaknesses. Across iterations, the best Blue policy improves task success from 46.0% to 57.3%, although susceptibility remains high at 70.7%. Later policies exhibit stronger selective cooperation while preserving trajectory efficiency. However, a persistent safety-helpfulness trade-off remains: policies that better resist adversarial steering do not simultaneously maximize task completion. Overall, our results show that LLM agents can exhibit limited strategic behavior, including selective trust and deception, while remaining highly vulnerable to adversarial persuasion. 

---
# MPAC: A Multi-Principal Agent Coordination Protocol for Interoperable Multi-Agent Collaboration 

**Authors**: Kaiyang Qian, Xinmin Fang, Zhengxiong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09744)  

**Abstract**: The AI agent ecosystem has converged on two protocols: the Model Context Protocol (MCP) for tool invocation and Agent-to-Agent (A2A) for single-principal task delegation. Both assume a single controlling principal, meaning one person or organization that owns every agent. When independent principals' agents must coordinate over shared state, such as engineers' coding agents editing the same repository, family members planning a shared trip, or agents from different organizations negotiating a joint decision, neither protocol applies, and coordination collapses to ad-hoc chat, manual merging, or silent overwrites. We present MPAC (Multi-Principal Agent Coordination Protocol), an application-layer protocol that fills this gap with explicit coordination semantics across five layers: Session, Intent, Operation, Conflict, and Governance. MPAC makes intent declaration a precondition for action, represents conflicts as first-class structured objects, and supports human-in-the-loop arbitration through a pluggable governance layer. The specification defines 21 message types, three state machines with normative transition tables, Lamport-clock causal watermarking, two execution models, three security profiles, and optimistic concurrency control on shared state. We release two interoperable reference implementations in Python and TypeScript with 223 tests, a JSON Schema suite, and seven live multi-agent demos. A controlled three-agent code review benchmark shows a 95 percent reduction in coordination overhead and a 4.8 times wall-clock speedup versus a serialized human-mediated baseline, with per-agent decision time preserved. The speedup comes from eliminating coordination waits, not compressing model calls. Specification, implementations, and demos are open source. 

---
# ExecTune: Effective Steering of Black-Box LLMs with Guide Models 

**Authors**: Vijay Lingam, Aditya Golatkar, Anwesan Pal, Ben Vo, Narayanan Sadagopan, Alessandro Achille, Jun Huan, Anoop Deoras, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2604.09741)  

**Abstract**: For large language models deployed through black-box APIs, recurring inference costs often exceed one-time training costs. This motivates composed agentic systems that amortize expensive reasoning into reusable intermediate representations. We study a broad class of such systems, termed Guide-Core Policies (GCoP), in which a guide model generates a structured strategy that is executed by a black-box core model. This abstraction subsumes base, supervised, and advisor-style approaches, which differ primarily in how the guide is trained. We formalize GCoP under a cost-sensitive utility objective and show that end-to-end performance is governed by guide-averaged executability: the probability that a strategy generated by the guide can be faithfully executed by the core. Our analysis shows that existing GCoP instantiations often fail to optimize executability under deployment constraints, resulting in brittle strategies and inefficient computation. Motivated by these insights, we propose ExecTune, a principled training recipe that combines teacher-guided acceptance sampling, supervised fine-tuning, and structure-aware reinforcement learning to directly optimize syntactic validity, execution success, and cost efficiency. Across mathematical reasoning and code-generation benchmarks, GCoP with ExecTune improves accuracy by up to 9.2% over prior state-of-the-art baselines while reducing inference cost by up to 22.4%. It enables Claude Haiku 3.5 to outperform Sonnet 3.5 on both math and code tasks, and to come within 1.7% absolute accuracy of Sonnet 4 at 38% lower cost. Beyond efficiency, GCoP also supports modular adaptation by updating the guide without retraining the core. 

---
# STaR-DRO: Stateful Tsallis Reweighting for Group-Robust Structured Prediction 

**Authors**: Samah Fodeh, Ganesh Puthiaraju, Elyas Irankhah, Linhai Ma, Srivani Talakokkul, Afshan Khan, Sreeraj Ramachandran, Jordan Alpert, Sarah Schellhorn  

**Link**: [PDF](https://arxiv.org/pdf/2604.09737)  

**Abstract**: Structured prediction requires models to generate ontology-constrained labels, grounded evidence, and valid structure under ambiguity, label skew, and heterogeneous group difficulty. We present a two-part framework for controllable inference and robust fine-tuning. First, we introduce a task-agnostic prompting strategy that combines XML-based instruction structure, disambiguation rules, verification-style reasoning, schema constraints, and self-validation to address format drift, label ambiguity, evidence hallucination, and metadata-conditioned confusion in in-context structured generation. Second, we introduce STaR-DRO, a stateful robust optimization method for group heterogeneity. It combines Tsallis mirror descent with momentum-smoothed, centered group-loss signals and bounded excess-only multipliers so that only persistently hard groups above a neutral baseline are upweighted, concentrating learning where it is most needed while avoiding volatile, dense exponentiated-gradient reweighting and unnecessary loss from downweighting easier groups. We evaluate the combined framework on EPPC Miner, a benchmark for extracting hierarchical labels and evidence spans from patient-provider secure messages. Prompt engineering improves zero-shot by +15.44 average F1 across Code, Sub-code, and Span over four Llama models. Building on supervised fine-tuning, STaR-DRO further improves the hardest semantic decisions: on Llama-3.3-70B-Instruct, Code F1 rises from 79.24 to 81.47 and Sub-code F1 from 67.78 to 69.30, while preserving Span performance and reducing group-wise validation cross-entropy by up to 29.6% on the most difficult clinical categories. Because these rare and difficult groups correspond to clinically consequential communication behaviors, these gains are not merely statistical improvements: they directly strengthen communication mining reliability for patient-centered care analysis. 

---
# Multi-Frequency Local Plasticity for Visual Representation Learning 

**Authors**: Mehdi Fatan Serj, C. Alejandro Parraga, Xavier Otazu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09734)  

**Abstract**: We study how far structured architectural bias can compensate for the absence of end-to-end gradient-based representation learning in visual recognition. Building on the VisNet tradition, we introduce a modular hierarchical framework combining: (i) fixed multi-frequency Gabor decomposition into F=7 parallel streams; (ii) within-stream competitive learning with Hebbian and Oja updates and anti-Hebbian decorrelation; (iii) an associative memory module inspired by modern Hopfield retrieval; and (iv) iterative top-down modulation using local prediction and reconstruction signals.
Representational layers are trained without end-to-end backpropagation through the full hierarchy; only the final linear readout and top-down projection matrices are optimized by gradient descent. We therefore interpret the model as a hybrid system that is predominantly locally trained but includes a small number of gradient-trained parameters.
On CIFAR-10, the full model reaches 80.1% +/- 0.3% top-1 accuracy, linear probe), compared with 71.0% for a Hebbian-only baseline and 83.4% for a gradient-trained model on the same fixed Gabor basis. On CIFAR-100, performance is 54.8%. Factorial analysis indicates that multi-frequency streams, associative memory, and top-down feedback contribute largely additively, with a significant Streams x TopDown interaction (p=0.02).
These results suggest that carefully chosen architectural priors can recover a substantial fraction of the performance typically associated with global gradient training, while leaving a measurable residual gap. Experiments are limited to CIFAR-10/100. 

---
# SMART: When is it Actually Worth Expanding a Speculative Tree? 

**Authors**: Lifu Wang, Pan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.09731)  

**Abstract**: Tree-based speculative decoding accelerates autoregressive generation by verifying a branching tree of draft tokens in a single target-model forward pass. However, existing methods prioritize maximizing token-level likelihood or the number of accepted tokens while ignoring a critical ``efficiency paradox'': the computational overhead of drafting and verifying big trees can grow super-linearly, particularly at scale. This often leads to negative wall-clock speedup when batch sizes increase or hardware saturation limits are reached. To address this, we propose SMART, a system-aware marginal analysis framework for runtime tree construction. SMART reformulates tree expansion as a hardware-aware optimization problem that directly maximizes end-to-end speedup. By applying a principled marginal benefit--cost rule at inference time, SMART expands a node only when its marginal benefit--cost ratio exceeds the tree-level speedup. SMART is training-free and serves as a plug-and-play controller for existing frameworks like MSD and EAGLE. Extensive evaluations across three MLLMs (e.g., LLaVA, Qwen2-VL) and four LLMs (e.g., Llama-3.1, DeepSeek-R1) demonstrate that SMART consistently outperforms state-of-the-art baselines. It delivers an average additional speedup of 20.0\% for MLLMs and 15.4\% for LLMs across compute-bound batching regimes and diverse GPU architectures without performance loss. 

---
# LOLGORITHM: Funny Comment Generation Agent For Short Videos 

**Authors**: Xuan Ouyang, Senan Wang, Bouzhou Wang, Siyuan Xiahou, Jinrong Zhou, Yuekang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09729)  

**Abstract**: Short-form video platforms have become central to multimedia information dissemination, where comments play a critical role in driving engagement, propagation, and algorithmic feedback. However, existing approaches -- including video summarization and live-streaming danmaku generation -- fail to produce authentic comments that conform to platform-specific cultural and linguistic norms. In this paper, we present LOLGORITHM, a novel modular multi-agent framework for stylized short-form video comment generation. LOLGORITHM supports six controllable comment styles and comprises three core modules: video content summarization, video classification, and comment generation with semantic retrieval and hot meme augmentation. We further construct a bilingual dataset of 3,267 videos and 16,335 comments spanning five high-engagement categories across YouTube and Douyin. Evaluation combining automatic scoring and large-scale human preference analysis demonstrates that LOLGORITHM consistently outperforms baseline methods, achieving human preference selection rates of 80.46\% on YouTube and 84.29\% on Douyin across 107 respondents. Ablation studies confirm that these gains are attributable to the framework architecture rather than the choice of backbone LLM, underscoring the robustness and generalizability of our approach. 

---
# ConfigSpec: Profiling-Based Configuration Selection for Distributed Edge--Cloud Speculative LLM Serving 

**Authors**: Xiangchen Li, Saeid Ghafouri, Jiakun Fan, Babar Ali, Hans Vandierendonck, Dimitrios S. Nikolopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2604.09722)  

**Abstract**: Speculative decoding enables collaborative Large Language Model (LLM) inference across cloud and edge by separating lightweight token drafting from heavyweight verification. While prior systems show performance and cost benefits, practical deployment requires navigating a large configuration space spanning draft model variants, quantisation levels, speculative lengths, and heterogeneous edge devices. This paper presents ConfigSpec, a configurationselection framework for distributed speculative LLM serving. ConfigSpec profiles edge devices and draft-target alignment, and models drafting throughput, acceptance rate, and power to evaluate goodput, verification cost efficiency, and energy efficiency across the joint configuration space. Our analysis across three edge platforms and two LLM families reveals structurally conflicting optima. Firstly, goodput is maximised by the smallest, fastest draft model at device-dependent speculative lengths (K*=2-10). Secondly, both cost and energy efficiency converge to K=2 due to a dominant bonus-token effect-with cost favouring the largest drafter for its high acceptance rate and energy favouring the smallest for its low power draw. These conflicts confirm that no single fixed configuration can simultaneously optimise all objectives, underscoring the need for profiling-based configuration selection in disaggregated edge-cloud LLM inference. 

---
# Training Deep Visual Networks Beyond Loss and Accuracy Through a Dynamical Systems Approach 

**Authors**: Hai La Quang, Hassan Ugail, Newton Howard, Cong Tran Tien, Nam Vu Hoai, Hung Nguyen Viet  

**Link**: [PDF](https://arxiv.org/pdf/2604.09716)  

**Abstract**: Deep visual recognition models are usually trained and evaluated using metrics such as loss and accuracy. While these measures show whether a model is improving, they reveal very little about how its internal representations change during training. This paper introduces a complementary way to study that process by examining training through the lens of dynamical systems. Drawing on ideas from signal analysis originally used to study biological neural activity, we define three measures from layer activations collected across training epochs: an integration score that reflects long-range coordination across layers, a metastability score that captures how flexibly the network shifts between more and less synchronised states, and a combined dynamical stability index. We apply this framework to nine combinations of model architecture and dataset, including several ResNet variants, DenseNet-121, MobileNetV2, VGG-16, and a pretrained Vision Transformer on CIFAR-10 and CIFAR-100. The results suggest three main patterns. First, the integration measure consistently distinguishes the easier CIFAR-10 setting from the more difficult CIFAR-100 setting. Second, changes in the volatility of the stability index may provide an early sign of convergence before accuracy fully plateaus. Third, the relationship between integration and metastability appears to reflect different styles of training behaviour. Overall, this study offers an exploratory but promising new way to understand deep visual training beyond loss and accuracy. 

---
# LAST: Leveraging Tools as Hints to Enhance Spatial Reasoning for Multimodal Large Language Models 

**Authors**: Shi-Yu Tian, Zhi Zhou, Kun-Yang Yu, Ming Yang, Yang Chen, Ziqiao Shang, Lan-Zhe Guo, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09712)  

**Abstract**: Spatial reasoning is a cornerstone capability for intelligent systems to perceive and interact with the physical world. However, multimodal large language models (MLLMs) frequently suffer from hallucinations and imprecision when parsing complex geometric layouts. As data-driven scaling struggles to internalize structured geometric priors and spatial constraints, integrating mature, specialized vision models presents a compelling alternative. Despite its promise, applying this paradigm to spatial reasoning is hindered by two key challenges: The difficulty of invoking heterogeneous, parameter-rich tools, as well as the challenge of understanding and effectively leveraging their diverse low-level outputs (e.g., segmentation masks, depth maps) in high-level reasoning. To address these challenges, we propose LAST, a unified framework for tool-augmented spatial reasoning. LAST features an extensible interactive sandbox, termed LAST-Box, which abstracts heterogeneous tool invocations into atomic instructions and reusable spatial skills, returning multimodal hints (e.g., annotated images and textual descriptions) that can be directly consumed by LLMs. We further design a three-stage progressive training strategy that guides models from understanding tool outputs to proficient and adaptive tool invocation. Experiments on four datasets show that LAST-7B achieves around 20\% performance gains over its backbone and outperforms strong proprietary closed-source LLMs, substantially enhancing reasoning on complex spatial tasks. 

---
# Orthogonal Quadratic Complements for Vision Transformer Feed-Forward Networks 

**Authors**: Wang Zixian  

**Link**: [PDF](https://arxiv.org/pdf/2604.09709)  

**Abstract**: Recent bilinear feed-forward replacements for vision transformers can substantially improve accuracy, but they often conflate two effects: stronger second-order interactions and increased redundancy relative to the main branch. We study a complementary design principle in which auxiliary quadratic features contribute only information not already captured by the dominant hidden representation. To this end, we propose Orthogonal Quadratic Complements (OQC), which construct a low-rank quadratic auxiliary branch and explicitly project it onto the orthogonal complement of the main branch before injection. We further study an efficient low-rank realization (OQC-LR) and gated extensions (OQC-static and OQC-dynamic).
Under a parameter-matched Deep-ViT and CIFAR-100 protocol with a fixed penultimate residual readout, full OQC improves an AFBO baseline from 64.25 +/- 0.22 to 65.59 +/- 0.22, while OQC-LR reaches 65.52 +/- 0.25 with a substantially better speed-accuracy tradeoff. On TinyImageNet, the gated extension OQC-dynamic achieves 51.88 +/- 0.32, improving the baseline (50.45 +/- 0.21) by 1.43 points and outperforming all ungated variants. Mechanism analyses show near-zero post-projection auxiliary-main overlap together with improved representation geometry and class separation. The full family, including both ungated and gated variants, generalizes consistently across both datasets. 

---
# The Deployment Gap in AI Media Detection: Platform-Aware and Visually Constrained Adversarial Evaluation 

**Authors**: Aishwarya Budhkar, Trishita Dhara, Siddhesh Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2604.09706)  

**Abstract**: Recent AI media detectors report near-perfect performance under clean laboratory evaluation, yet their robustness under realistic deployment conditions remains underexplored. In practice, AI-generated images are resized, compressed, re-encoded, and visually modified before being shared on online platforms. We argue that this creates a deployment gap between laboratory robustness and real-world reliability.
In this work, we introduce a platform-aware adversarial evaluation framework for AI media detection that explicitly models deployment transforms (e.g., resizing, compression, screenshot-style distortions) and constrains perturbations to visually plausible meme-style bands rather than full-image noise. Under this threat model, detectors achieving AUC $\approx$ 0{.}99 in clean settings experience substantial degradation. Per-image platform-aware attacks reduce AUC to significantly lower levels and achieve high fake-to-real misclassification rates, despite strict visual constraints. We further demonstrate that universal perturbations exist even under localized band constraints, revealing shared vulnerability directions across inputs. Beyond accuracy degradation, we observe pronounced calibration collapse under attack, where detectors become confidently incorrect.
Our findings highlight that robustness measured under clean conditions substantially overestimates deployment robustness. We advocate for platform-aware evaluation as a necessary component of future AI media security benchmarks and release our evaluation framework to facilitate standardized robustness assessment. 

---
# Identity-Aware U-Net: Fine-grained Cell Segmentation via Identity-Aware Representation Learning 

**Authors**: Rui Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.09702)  

**Abstract**: Precise segmentation of objects with highly similar shapes remains a challenging problem in dense prediction, especially in scenarios with ambiguous boundaries, overlapping instances, and weak inter-instance visual differences. While conventional segmentation models are effective at localizing object regions, they often lack the discriminative capacity required to reliably distinguish a target object from morphologically similar distractors. In this work, we study fine-grained object segmentation from an identity-aware perspective and propose Identity-Aware U-Net (IAU-Net), a unified framework that jointly models spatial localization and instance discrimination. Built upon a U-Net-style encoder-decoder architecture, our method augments the segmentation backbone with an auxiliary embedding branch that learns discriminative identity representations from high-level features, while the main branch predicts pixel-accurate masks. To enhance robustness in distinguishing objects with near-identical contours or textures, we further incorporate triplet-based metric learning, which pulls target-consistent embeddings together and separates them from hard negatives with similar morphology. This design enables the model to move beyond category-level segmentation and acquire a stronger capability for precise discrimination among visually similar objects. Experiments on benchmarks including cell segmentation demonstrate promising results, particularly in challenging cases involving similar contours, dense layouts, and ambiguous boundaries. 

---
# Attention-Guided Flow-Matching for Sparse 3D Geological Generation 

**Authors**: Zhixiang Lu, Mengqi Han, Peixin Guo, Tianming Bai, Jionglong Su, Fei Fang, Sifan Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.09700)  

**Abstract**: Constructing high-resolution 3D geological models from sparse 1D borehole and 2D surface data is a highly ill-posed inverse problem. Traditional heuristic and implicit modeling methods fundamentally fail to capture non-linear topological discontinuities under extreme sparsity, often yielding unrealistic artifacts. Furthermore, while deep generative architectures like Diffusion Models have revolutionized continuous domains, they suffer from severe representation collapse when conditioned on sparse categorical grids. To bridge this gap, we propose 3D-GeoFlow, the first Attention-Guided Continuous Flow Matching framework tailored for sparse multimodal geological modeling. By reformulating discrete categorical generation as a simulation-free, continuous vector field regression optimized via Mean Squared Error, our model establishes stable, deterministic optimal transport paths. Crucially, we integrate 3D Attention Gates to dynamically propagate localized borehole features across the volumetric latent space, ensuring macroscopic structural coherence. To validate our framework, we curated a large-scale multimodal dataset comprising 2,200 procedurally generated 3D geological cases. Extensive out-of-distribution (OOD) evaluations demonstrate that 3D-GeoFlow achieves a paradigm shift, significantly outperforming heuristic interpolations and standard diffusion baselines. 

---
# Evaluating Scene-based In-Situ Item Labeling for Immersive Conversational Recommendation 

**Authors**: Jiazhou Liang, Yifan Simon Liu, David Guo, Minqi Sun, Yilun Jiang, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2604.09698)  

**Abstract**: The growing ubiquity of Extended Reality (XR) is driving Conversational Recommendation Systems (CRS) toward visually immersive experiences. We formalize this paradigm as Immersive CRS (ICRS), where recommended items are highlighted directly in the user's scene-based visual environment and augmented with in-situ labels. While item recommendation has been widely studied, the problem of how to select and evaluate which information to present as immersive labels remains an open problem. To this end, we introduce a principled categorization of information needs into explicit intent satisfaction and proactive information needs and use these to define novel evaluation metrics for item label selection. We benchmark IR-, LLM-, and VLM-based methods across three datasets and ICRS scenarios: fashion, movie recommendation, and retail shopping. Our evaluation reveals three important limitations of existing methods: (1) they fail to leverage scenario-specific information modalities (e.g., visual cues for fashion, meta-data for retail), (2) they present redundant information that is visually inferable, and (3) they poorly anticipate users' proactive information needs from explicit dialogue alone. In summary, this work provides both a novel evaluation paradigm for in-situ item labeling in ICRS and highlights key challenges for future work. 

---
# I Can't Believe TTA Is Not Better: When Test-Time Augmentation Hurts Medical Image Classification 

**Authors**: Daniel Nobrega Medeiros  

**Link**: [PDF](https://arxiv.org/pdf/2604.09697)  

**Abstract**: Test-time augmentation (TTA)--aggregating predictions over multiple augmented copies of a test input--is widely assumed to improve classification accuracy, particularly in medical imaging where it is routinely deployed in production systems and competition solutions. We present a systematic empirical study challenging this assumption across three MedMNIST v2 benchmarks and four architectures spanning three orders of magnitude in parameter count (21K to 11M). Our principal finding is that TTA with standard augmentation pipelines consistently degrades accuracy relative to single-pass inference, with drops as severe as 31.6 percentage points for ResNet-18 on pathology images. This degradation affects all architectures, including convolutional models, and worsens with more augmented views. The sole exception is ResNet-18 on dermatology images, which gains a modest +1.6%. We identify the distribution shift between augmented and training-time inputs--amplified by batch normalization statistics mismatch--as the primary mechanism. Our ablation studies show that augmentation strategy matters critically: intensity-only augmentations preserve more performance than geometric transforms, and including the original unaugmented image partially mitigates but does not eliminate the accuracy drop. These findings serve as a cautionary note for practitioners: TTA should not be applied as a default post-hoc improvement but must be validated on the specific model-dataset combination. 

---
# Assessing Privacy Preservation and Utility in Online Vision-Language Models 

**Authors**: Karmesh Siddharam Chaudhari, Youxiang Zhu, Amy Feng, Xiaohui Liang, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09695)  

**Abstract**: The increasing use of Online Vision Language Models (OVLMs) for processing images has introduced significant privacy risks, as individuals frequently upload images for various utilities, unaware of the potential for privacy violations. Images contain relationships that relate to Personally Identifiable Information (PII), where even seemingly harmless details can indirectly reveal sensitive information through surrounding clues. This paper explores the critical issue of PII disclosure in images uploaded to OVLMs and its implications for user privacy. We investigate how the extraction of contextual relationships from images can lead to direct (explicit) or indirect (implicit) exposure of PII, significantly compromising personal privacy. Furthermore, we propose methods to protect privacy while preserving the intended utility of the images in Vision Language Model (VLM)-based applications. Our evaluation demonstrates the efficacy of these techniques, highlighting the delicate balance between maintaining utility and protecting privacy in online image processing environments. Index Terms-Personally Identifiable Information (PII), Privacy, Utility, privacy concerns, sensitive information 

---
# TaFall: Balance-Informed Fall Detection via Passive Thermal Sensing 

**Authors**: Chengxiao Li, Xie Zhang, Wei Zhu, Yan Jiang, Chenshu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09693)  

**Abstract**: Falls are a major cause of injury and mortality among older adults, yet most incidents occur in private indoor environments where monitoring must balance effectiveness with privacy. Existing privacy-preserving fall detection approaches, particularly those based on radio frequency sensing, often rely on coarse motion cues, which limits reliability in real-world deployments. We introduce TaFall, a balance-informed fall detection system based on low-cost, privacy-preserving thermal array sensing. The key insight is that TaFall models a fall as a process of balance degradation and detects falls by estimating pose-driven biomechanical balance dynamics. To enable this capability from low-resolution thermal array maps, we propose (i) an appearance-motion fusion model for robust pose reconstruction, (ii) physically grounded balance-aware learning, and (iii) pose-bridged pretraining to improve robustness. TaFall achieves a detection rate of 98.26% with a false alarm rate of 0.65% on our dataset with over 3,000 fall instances from 35 participants across diverse indoor environments. In 27 day deployments across four homes, TaFall attains an ultra-low false alarm rate of 0.00126% and a pilot bathroom study confirms robustness under moisture and thermal interference. Together, these results establish TaFall as a reliable and privacy-preserving approach to fall detection in everyday living environments. 

---
# CAGE: Bridging the Accuracy-Aesthetics Gap in Educational Diagrams via Code-Anchored Generative Enhancement 

**Authors**: Dikshant Kukreja, Kshitij Sah, Karan Goyal, Mukesh Mohania, Vikram Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2604.09691)  

**Abstract**: Educational diagrams -- labeled illustrations of biological processes, chemical structures, physical systems, and mathematical concepts -- are essential cognitive tools in K-12 instruction. Yet no existing method can generate them both accurately and engagingly. Open-source diffusion models produce visually rich images but catastrophically garble text labels. Code-based generation via LLMs guarantees label correctness but yields visually flat outputs. Closed-source APIs partially bridge this gap but remain unreliable and prohibitively expensive at educational scale. We quantify this accuracy-aesthetics dilemma across all three paradigms on 400 K-12 diagram prompts, measuring both label fidelity and visual quality through complementary automated and human evaluation protocols. To resolve it, we propose CAGE (Code-Anchored Generative Enhancement): an LLM synthesizes executable code producing a structurally correct diagram, then a diffusion model conditioned on the programmatic output via ControlNet refines it into a visually polished graphic while preserving label fidelity. We also introduce EduDiagram-2K, a collection of 2,000 paired programmatic-stylized diagrams enabling this pipeline, and present proof-of-concept results and a research agenda for the multimedia community. 

---
# Face Density as a Proxy for Data Complexity: Quantifying the Hardness of Instance Count 

**Authors**: Abolfazl Mohammadi-Seif, Ricardo Baeza-Yates  

**Link**: [PDF](https://arxiv.org/pdf/2604.09689)  

**Abstract**: Machine learning progress has historically prioritized model-centric innovations, yet achievable performance is frequently capped by the intrinsic complexity of the data itself. In this work, we isolate and quantify the impact of instance density (measured by face count) as a primary driver of data complexity. Rather than simply observing that ``crowded scenes are harder,'' we rigorously control for class imbalance to measure the precise degradation caused by density alone.
Controlled experiments on the WIDER FACE and Open Images datasets, restricted to exactly 1 to 18 faces per image with perfectly balanced sampling, reveal that model performance degrades monotonically with increasing face count. This trend holds across classification, regression, and detection paradigms, even when models are fully exposed to the entire density range. Furthermore, we demonstrate that models trained on low-density regimes fail to generalize to higher densities, exhibiting a systematic under-counting bias, with error rates increasing by up to 4.6x, which suggests density acts as a domain shift.
These findings establish instance density as an intrinsic, quantifiable dimension of data hardness and motivate specific interventions in curriculum learning and density-stratified evaluation. 

---
# Grid2Matrix: Revealing Digital Agnosia in Vision-Language Models 

**Authors**: Yunkai Zhang, Linda Li, Yingxin Cui, Xiyuan Ruan, Zeyu Zheng, Kezhen Chen, Yi Zhang, Diji Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09687)  

**Abstract**: Vision-Language Models (VLMs) excel on many multimodal reasoning benchmarks, but these evaluations often do not require an exhaustive readout of the image and can therefore obscure failures in faithfully capturing all visual details. We introduce Grid2Matrix (G2M), a controlled benchmark in which a model is shown a color grid and a color-to-number mapping, and must output the corresponding matrix. By varying grid size and the number of colors, G2M provides a simple way to increase visual complexity while minimizing semantic confounds. We find that VLMs exhibit a sharp early collapse in zero-shot end-to-end evaluation, failing on surprisingly small grids rather than degrading gradually as the task becomes denser. We probe the visual encoders of VLMs from two representative families and find that they preserve substantially more of the grid information than the corresponding end-to-end outputs. This suggests that the failure is not explained by visual encoding alone, but also reflects a gap between what remains recoverable from visual features and what is ultimately expressed in language. We term this gap \textit{Digital Agnosia}. Further analyses show that these errors are highly structured and depend strongly on how grid cells overlap with visual patch boundaries. We also find that common strategies such as model scaling and multimodal alignment do not fully eliminate this failure mode. We expect G2M to serve as a useful testbed for understanding where and how VLMs lose fine visual details, and for evaluating tasks where missing even small visual details can matter, such as tables, charts, forms, and GUIs. 

---
# Decision-Theoretic Safety Assessment of Persona-Driven Multi-Agent Systems in O-RAN 

**Authors**: Zeinab Nezami, Syed Ali Raza Zaidi, Maryam Hafeez, Louis Powell, Vara Prasad Talari, Mallik Tatipamula  

**Link**: [PDF](https://arxiv.org/pdf/2604.09682)  

**Abstract**: Autonomous network management in Open Radio Access Networks requires intelligent decision making across conflicting objectives, yet existing LLM based multi agent systems employ homogeneous strategies and lack systematic predeployment validation. We introduce a persona driven multi agent framework where configurable behavioral personas structured specifications encoding optimization priorities, risk tolerance, and decision making style influence five specialized agents (planning, coordination, resource allocation, code generation, analysis). To enable rigorous validation, we develop a three dimensional evaluation framework grounded in decision theory, measuring normative compliance (optimality adherence), prescriptive alignment (behavioral guideline consistency), and behavioral dynamics (emergent system properties). We evaluate 486 persona configurations across two ORAN optimization challenges (energy efficient resource allocation and network load balancing). Results demonstrate that persona agent alignment significantly impacts both individual performance (14.3 percent) and emergent multi agent coordination, with retrieval architecture (GraphRAG vs. RAG) fundamentally constraining customization effectiveness. Single agent persona modifications propagate system wide through cascading effects, with certain combinations exhibiting detectable fundamental incompatibilities. Our framework provides systematic validation mechanisms for deploying LLM based automation in mission critical telecommunications infrastructure. 

---
# Heterogeneous Consensus-Progressive Reasoning for Efficient Multi-Agent Debate 

**Authors**: Yiqing Liu, Hantao Yao, Wu Liu, Allen He, Yongdong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09679)  

**Abstract**: Multi-Agent Debate (MAD) is a collaborative framework in which multiple agents iteratively refine solutions through the generation of reasoning and alternating critique cycles. Current work primarily optimizes intra-round topologies and inter-round interactions separately, which still results in high token costs regardless of task complexity. This work introduces Heterogeneous Consensus-Progressive Reasoning for Efficient Multi-Agent Debate (HCP-MAD), leveraging consensus as a dynamic signal to facilitate progressive reasoning. The core motivation is that a majority of straightforward tasks can be effectively resolved via lightweight pair-agent debates, while complex tasks require expanded collaboration. Consequently, HCP-MAD employs a three-stage progressive reasoning mechanism to develop adaptive solutions across varying task complexities. Firstly, Heterogeneous Consensus Verification conducts rapid consensus verification using a pair of heterogeneous agents for early stopping. Next, the Heterogeneous Pair-Agent Debate applies an adaptive stopping criterion to dynamically terminate mutual critique of recorded reasoning traces. Finally, the unresolved tasks are addressed through Escalated Collective Voting by aggregating diverse perspectives from additional agents. Experiments across multiple benchmarks show that HCP-MAD significantly enhances accuracy while substantially reducing token costs. 

---
# NetAgentBench: A State-Centric Benchmark for Evaluating Agentic Network Configuration 

**Authors**: Ahmed Twabi, Yepeng Ding, Tohru Kondo  

**Link**: [PDF](https://arxiv.org/pdf/2604.09678)  

**Abstract**: As agentic network management gains popularity, there is a critical need for evaluation frameworks that transcend static, one-shot testing. To address this, we introduce NetAgentBench, a dynamic benchmark that evaluates agent interactions through a Finite State Machine (FSM) formalization guaranteeing determinism, correctness, and bounded execution. This provides the networking landscape with a rigorous foundation to measure complex, multi-turn operational behaviors. Our empirical evaluation of four state-of-the-art LLM agents through diverse network configuration tasks reveals stark deficiencies: while agents can solve basic tasks, they suffer severe exploration meltdowns and coherence collapse during expert-level configurations. Ultimately, NetAgentBench demonstrates that systematically evaluating multi-turn behavioral stability is an indispensable step toward realizing trustworthy, fully autonomous networks. 

---
# A Comparative Theoretical Analysis of Entropy Control Methods in Reinforcement Learning 

**Authors**: Ming Lei, Christophe Baehr  

**Link**: [PDF](https://arxiv.org/pdf/2604.09676)  

**Abstract**: Reinforcement learning (RL) has become a key approach for enhancing reasoning in large language models (LLMs), yet scalable training is often hindered by the rapid collapse of policy entropy, which leads to premature convergence and performance saturation. This paper provides a comparative theoretical analysis of two entropy control strategies: traditional entropy regularization and the recently proposed covariance-based mechanism. We establish a unified framework for entropy dynamics under softmax parameterization, showing that entropy change is governed by the covariance between log-probabilities and logit updates. Our analysis reveals that traditional entropy regularization introduces a dense, persistent bias that modifies the stationary condition, leading to suboptimal policies, while covariance-based methods selectively regularize a sparse subset of high-covariance tokens and achieve asymptotic unbiasedness when the regularization coefficient is annealed. These results provide principled guidelines for entropy control in LLM posttraining, with implications for scaling RL to larger models and more complex reasoning tasks. 

---
# Real-Time Voicemail Detection in Telephony Audio Using Temporal Speech Activity Features 

**Authors**: Kumar Saurav  

**Link**: [PDF](https://arxiv.org/pdf/2604.09675)  

**Abstract**: Outbound AI calling systems must distinguish voicemail greetings from live human answers in real time to avoid wasted agent interactions and dropped calls. We present a lightweight approach that extracts 15 temporal features from the speech activity pattern of a pre-trained neural voice activity detector (VAD), then classifies with a shallow tree-based ensemble. Across two evaluation sets totaling 764 telephony recordings, the system achieves a combined 96.1% accuracy (734/764), with 99.3% (139/140) on an expert-labeled test set and 95.4% (595/624) on a held-out production set. In production validation over 77,000 calls, it maintained a 0.3% false positive rate and 1.3% false negative rate. End-to-end inference completes in 46 ms on a commodity dual-core CPU with no GPU, supporting 380+ concurrent WebSocket calls. In our search over 3,780 model, feature, and threshold combinations, feature importance was concentrated in three temporal variables. Adding transcription keywords or beep-based features did not improve the best real-time configuration and increased latency substantially. Our results suggest that temporal speech patterns are a strong signal for distinguishing voicemail greetings from live human answers. 

---
# Active Inference with a Self-Prior in the Mirror-Mark Task 

**Authors**: Dongmin Kim, Hoshinori Kanazawa, Yasuo Kuniyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.09673)  

**Abstract**: The mirror self-recognition test evaluates whether a subject touches a mark on its own body that is visible only in a mirror, and is widely used as an indicator of self-awareness. In this study, we present a computational model in which this behavior emerges spontaneously through a single mechanism, the self-prior, without any external reward. The self-prior, implemented with a Transformer, learns the density of familiar multisensory experiences; when a novel mark appears, the discrepancy from this learned distribution drives mark-directed behavior through active inference. A simulated infant, relying solely on vision and proprioception without tactile input, discovered a sticker placed on its own face in the mirror and removed it in approximately 70% of cases without any explicit instruction. Expected free energy decreased significantly after sticker removal, confirming that the self-prior operates as an internal criterion for distinguishing self from non-self. Cross-modal sampling further demonstrated that the self-prior captures visual--proprioceptive associations, functioning as a probabilistic body schema. These results provide a concise computational account of the key behavior observed in the mirror test and suggest that the free energy principle can serve as a unifying hypothesis for investigating the developmental origins of self-awareness. Code is available at: this https URL 

---
# Human-like Working Memory Interference in Large Language Models 

**Authors**: Hua-Dong Xiong, Li Ji-An, Jiaqi Huang, Robert C. Wilson, Kwonjoon Lee, Xue-Xin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.09670)  

**Abstract**: Intelligent systems must maintain and manipulate task-relevant information online to adapt to dynamic environments and changing goals. This capacity, known as working memory, is fundamental to human reasoning and intelligence. Despite having on the order of 100 billion neurons, both biological and artificial systems exhibit limitations in working memory. This raises a key question: why do large language models (LLMs) show such limitations, given that transformers have full access to prior context through attention? We find that although a two-layer transformer can be trained to solve working memory tasks perfectly, a diverse set of pretrained LLMs continues to show working memory limitations. Notably, LLMs reproduce interference signatures observed in humans: performance degrades with increasing memory load and is biased by recency and stimulus statistics. Across models, stronger working memory capacity correlates with broader competence on standard benchmarks, mirroring its link to general intelligence in humans. Yet despite substantial variability in working memory performance, LLMs surprisingly converge on a common computational mechanism. Rather than directly copying the relevant memory item from context, models encode multiple memory items in entangled representations, such that successful recall depends on interference control -- actively suppressing task-irrelevant content to isolate the target for readout. Moreover, a targeted intervention that suppresses stimulus content information improves performance, providing causal support for representational interference. Together, these findings identify representational interference as a core constraint on working memory in pretrained LLMs, suggesting that working-memory limits in biological and artificial systems may reflect a shared computational challenge: selecting task-relevant information under interference. 

---
# Digital hybridity and relics in cultural heritage: using corpus linguistics to inform design in emerging technologies from AI to VR 

**Authors**: Emma McClaughlin, Glenn McGarry, Alan Chamberlain, Geert De Wilde, Oliver Butler  

**Link**: [PDF](https://arxiv.org/pdf/2604.09669)  

**Abstract**: Hybrid technologies enable the blending of physical and digital elements, creating new ways to experience and interact with the world. Such technologies can transform engagement with relics, both secular and sacred but they present challenges for capturing faith, belief, and representation responsibly. Given the complexities of digital representation and the ethical challenges inherent in digitising culturally significant objects, a transdisciplinary understanding of these issues is needed. To inform this discussion from a linguistic perspective, we examined the representation of relics in historical and contemporary texts. Using a corpus linguistic approach to extract modifiers of the word relic in corpora of Early Modern English books and contemporary web sourced texts from 2021, we examined the multifaceted ways in which relics have been perceived and evaluated over time. Early texts consider relics as both objects of moral and spiritual significance, and tools of religious and political control, while they are more often framed as heritage symbols, reflecting past events, places, and traditions in contemporary texts. We discuss how hybrid, sometimes AI based technologies can enhance accessibility and engagement, whilst also challenging traditional sensitivities around authenticity and sensory experience, which are integral to the meaning and significance of relics. 

---
# Do We Still Need GraphRAG? Benchmarking RAG and GraphRAG for Agentic Search Systems 

**Authors**: Dongzhe Fan, Zheyi Xue, Siyuan Liu, Qiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.09666)  

**Abstract**: Retrieval-augmented generation (RAG) and its graph-based extensions (GraphRAG) are effective paradigms for improving large language model (LLM) reasoning by grounding generation in external knowledge. However, most existing RAG and GraphRAG systems operate under static or one-shot retrieval, where a fixed set of documents is provided to the LLM in a single pass. In contrast, recent agentic search systems enable dynamic, multi-round retrieval and sequential decision-making during inference, and have shown strong gains when combined with vanilla RAG by introducing implicit structure through interaction. This progress raises a fundamental question: can agentic search compensate for the absence of explicit graph structure, reducing the need for costly GraphRAG pipelines? To answer this question, we introduce RAGSearch, a unified benchmark that evaluates dense RAG and representative GraphRAG methods as retrieval infrastructures under agentic search. RAGSearch covers both training-free and training-based agentic inference across multiple question answering benchmarks. To ensure fair and reproducible comparison, we standardize the LLM backbone, retrieval budgets, and inference protocols, and report results on full test sets. Beyond answer accuracy, we report offline preprocessing cost, online inference efficiency, and stability. Our results show that agentic search substantially improves dense RAG and narrows the performance gap to GraphRAG, particularly in RL-based settings. Nevertheless, GraphRAG remains advantageous for complex multi-hop reasoning, exhibiting more stable agentic search behavior when its offline cost is amortized. Together, these findings clarify the complementary roles of explicit graph structure and agentic search, and provide practical guidance on retrieval design for modern agentic RAG systems. 

---
# Deliberative Alignment is Deep, but Uncertainty Remains: Inference time safety improvement in reasoning via attribution of unsafe behavior to base model 

**Authors**: Pankayaraj Pathmanathan, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09665)  

**Abstract**: While the wide adoption of refusal training in large language models (LLMs) has showcased improvements in model safety, recent works have highlighted shortcomings due to the shallow nature of these alignment methods. To this end, the work on Deliberative alignment proposed distilling reasoning capabilities from stronger reasoning models, thereby instilling deeper safety in LLMs. In this work, we study the impact of deliberative alignment in language models. First, we show that despite being larger in model size and stronger in safety capability, there exists an alignment gap between teacher and student language models, which affects both the safety and general utility of the student model. Furthermore, we show that models aligned through deliberative alignment can retain unsafe behaviors from the base model despite learning the reasoning patterns of larger reasoning models. Building upon this observation, we propose a BoN sampling method that attributes the unsafe behavior back to the base LLMs in the latent space, thereby down-ranking unsafe responses to gain a meaningful improvement in model safety across multiple safety benchmarks with minimal loss in utility. In particular, across 7 teacher models and 6 student models of different classes and sizes, we show an average attack success rate (ASR) reduction of 28.2% in DAN, 31.3% in WildJailbreak and 35.4 % in StrongREJECT benchmarks. We further show that these safety gains prevail post RL training, thus highlighting the uncertainty in safety reasoning and it's explicit attribution to the base model. 

---
# Learning noisy phase transition dynamics from stochastic partial differential equations 

**Authors**: Luning Sun, Van Hai Nguyen, Shusen Liu, John Klepeis, Fei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.09664)  

**Abstract**: The non-equilibrium dynamics of mesoscale phase transitions are fundamentally shaped by thermal fluctuations, which not only seed instabilities but actively control kinetic pathways, including rare barrier-crossing events such as nucleation that are entirely inaccessible to deterministic models. Machine-learning surrogates for such systems must therefore represent stochasticity explicitly, enforce conservation laws by construction, and expose physically interpretable structure. We develop physics-aware surrogate models for the stochastic Cahn-Hilliard equation in 3D that satisfy all three requirements simultaneously. The key innovation is to parameterize the surrogate at the level of inter-cell fluxes, decomposing each flux into a deterministic mobility-weighted chemical-potential gradient and a learnable noise amplitude. This design guarantees exact mass conservation at every step and adds physical fluctuations to inter-cell mass transport. A learnable free energy functional provides thermodynamic interpretability, validated by independent recovery of the bulk double-well landscape, interfacial excess energy, and curvature-independent interfacial tension. Tests demonstrate accurate reproduction of ensemble statistics and noise-accelerated coarsening, with generalization to spatial domains 64 times larger in volume and temporal horizons 160x longer than those seen during training. Critically, the stochastic surrogate captures thermally activated nucleation in the metastable regime, a qualitative capability that no deterministic surrogate can provide regardless of training, thus establishing flux-level stochasticity as an architectural necessity rather than an optional enhancement. 

---
# Fairboard: a quantitative framework for equity assessment of healthcare models 

**Authors**: James K. Ruffle, Samia Mohinta, Chris Foulon, Mohamad Zeina, Zicheng Wang, Sebastian Brandner, Harpreet Hyare, Parashkev Nachev  

**Link**: [PDF](https://arxiv.org/pdf/2604.09656)  

**Abstract**: Despite there now being more than 1,000 FDA-authorised AI medical devices, formal equity assessments -- whether model performance is uniform across patient subgroups -- are rare. Here, we evaluate the equity of 18 open-source brain tumour segmentation models across 648 glioma patients from two independent datasets (n = 11,664 model inferences) along distinct univariate, Bayesian multivariate, spatial, and representational dimensions. We find that patient identity consistently explains more performance variance than model choice, with clinical factors, including molecular diagnosis, tumour grade, and extent of resection, predicting segmentation accuracy more strongly than model architecture. A voxel-wise spatial meta-analysis identifies neuroanatomically localised biases that are compartment-specific yet often consistent across models. Within a high-dimensional latent space of lesion masks and clinic-demographic features, model performance clusters significantly, indicating that the patient feature space contains axes of algorithmic vulnerability. Although newer models tend toward greater equity, none provide a formal fairness guarantee. Lastly, we release Fairboard, an open-source, no-code dashboard that lowers barriers to equitable model monitoring in medical imaging. 

---
# NeuroPath: Practically Adopting Motor Imagery Decoding through EEG Signals 

**Authors**: Jiani Cao, Kun Wang, Yang Liu, Zhenjiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09654)  

**Abstract**: Motor Imagery (MI) is an emerging Brain-Computer Interface (BCI) paradigm where a person imagines body movements without physical action. By decoding scalp-recorded electroencephalography (EEG) signals, BCIs establish direct communication to control external devices, offering significant potential in prosthetics, rehabilitation, and human-computer interaction. However, existing solutions remain difficult to deploy. (i) Most employ independent, opaque models for each MI task, lacking a unified architectural foundation. Consequently, these models are trained in isolation, failing to learn robust representations from diverse datasets, resulting in modest performance. (ii) They primarily adopt fixed sensor deployment, whereas real-world setups vary in electrode number and placement, causing models to fail across configurations. (iii) Performance degrades sharply under low-SNR conditions typical of consumer-grade EEG.
To address these challenges, we present NeuroPath, a neural architecture for robust MI decoding. NeuroPath takes inspiration from the brain's signal pathway from cortex to scalp, utilizing a deep neural architecture with specialized modules for signal filtering, spatial representation learning, and feature classification, enabling unified decoding. To handle varying electrode configurations, we introduce a spatially aware graph adapter accommodating different electrode numbers and placements. To enhance robustness under low-SNR conditions, NeuroPath incorporates multimodal auxiliary training to refine EEG representations and stabilize performance on noisy real-world data. Evaluations on three consumer-grade and three medical-grade public datasets demonstrate that NeuroPath achieves superior performance. 

---
# Diffusion-Based Generative Priors for Efficient Beam Alignment in Directional Networks 

**Authors**: Esraa Fahmy Othman, Lina Bariah, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2604.09653)  

**Abstract**: Beam alignment is a key challenge in directional mmWave and THz systems, where narrow beams require accurate yet low-overhead training. Existing learning-based approaches typically predict a single beam and do not quantify uncertainty, limiting adaptive beam sweeping. We recast beam alignment as a generative task and propose a conditional diffusion model that learns a probabilistic beam prior from compact geometric and multipath features. The learned priors guide top-$k$ sweeps and capture the SNR loss induced by limited probing. Using a ray-traced DeepMIMO scenario with an 8-beam DFT codebook, our best conditional diffusion model achieves strong ranking performance (Hit@1 $\approx 0.61$, Hit@3 $\approx 0.90$, Hit@5 $\approx 0.97$) while preserving SNR at small sweep budgets. Compared with a deterministic classifier baseline, diffusion improves Hit@1 by about 180\%. Results further highlight the importance of informative conditioning and the ability of diffusion sampling to flexibly trade accuracy for computational efficiency. The proposed diffusion framework achieves substantial improvements in small-$k$ Hit rates, translating into reduced beam training overhead and enabling low-latency, energy-efficient beam alignment for mmWave and THz systems while preserving received SNR. 

---
# Dynamic Forecasting and Temporal Feature Evolution of Stock Repurchases in Listed Companies Using Attention-Based Deep Temporal Networks 

**Authors**: Xiang Ao, Jingxuan Zhang, Xinyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.09650)  

**Abstract**: Accurately predicting stock repurchases is crucial for quantitative investment and risk management, yet traditional static models fail to capture the complex temporal dependencies of corporate financial conditions. This paper proposes a dynamic early warning system integrating economic theory with deep temporal networks. Using Chinese A-share panel data (2014-2024), we employ a hybrid Temporal Convolutional Network (TCN) and Attention-based LSTM to capture long- and short-term financial evolutionary patterns. Rolling-window cross-validation demonstrates our model significantly outperforms static baselines like Logistic Regression and XGBoost. Furthermore, utilizing Explainable AI (XAI), we reveal the temporal dynamics of repurchase decisions: prolonged "undervaluation" serves as the long-term underlying motive, while a sharp increase in "cash flow" acts as the decisive short-term trigger. This study provides a robust deep learning paradigm for financial forecasting and offers dynamic empirical support for classic corporate finance hypotheses. 

---
# WearBCI Dataset: Understanding and Benchmarking Real-World Wearable Brain-Computer Interfaces Signals 

**Authors**: Haoxian Liu, Hengle Jiang, Lanxuan Hong, Xiaomin Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09649)  

**Abstract**: Brain-computer interfaces (BCIs) have opened new platforms for human-computer interaction, medical diagnostics, and neurorehabilitation. Wearable BCI systems, which typically employ non-invasive electrodes for portable monitoring, hold great promise for real-world applications, but also face significant challenges of signal quality degradation caused by motion artifacts and environmental interferences. Most existing wearable BCI datasets are collected under stationary or controlled lab settings, limiting their utility for evaluating performance under body movement. To bridge this gap, we introduce WearBCI, the first dataset that comprehensively evaluates wearable BCI signals under different motion dynamics with synchronized multimodal recordings (EEG, IMU, and egocentric video), and systematic benchmark evaluations for studying impacts of motion artifact. Specifically, we collect data from 36 participants across different motion dynamics, including body movements, walking, and navigation. This dataset includes synchronized electroencephalography (EEG), inertial measurement unit (IMU) data, and egocentric video recordings. We analyze the collected wearable EEG signals to understand the impact of motion artifacts across different conditions, and benchmark representative EEG signal enhancement techniques on our dataset. Furthermore, we explore two new case studies: cross-modal EEG signal enhancement and multi-dimension human behavior understanding. These findings offer valuable insights into real-world wearable BCI deployment and new applications. 

---
# Efficient Disruption of Criminal Networks through Multi-Objective Genetic Algorithms 

**Authors**: Yehezkiel Darmadi, Thanh Thi Nguyen, Campbell Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2604.09647)  

**Abstract**: Criminal networks, such as the Sicilian Mafia, pose substantial threats to public safety, national security, and economic stability. Outdated disruption methods with a focus on removing influential individuals or key players have proven ineffective due to the covertness of the network. Thus, researchers have been trying to apply Social Network Analysis (SNA) techniques, such as centrality-based measures, to identify key players. However, removing individuals with high centrality often proves to be inefficient, as it does not mimic the real-world scenarios that Law Enforcement Agencies (LEAs) face. For instance, the operational costs limit the LEAs from exploiting the results of the centrality-based methods. This study proposes a multi-objective optimisation framework like the Weighted Sum Genetic Algorithm (WS-GA) and the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to identify disruption strategies that balance two conflicting goals, maximising fragmentation and minimising operational cost which is captured by the spatial distance between nodes and the nearest LEA headquarters. The study utilises the "Montagna Operation" dataset for the experiments. The results demonstrate that although centrality-based approaches can fragment network effectively, they tend to incur higher operational costs. In contrast, the proposed algorithms achieve comparable disruption outcomes with significantly lower operational costs. The contribution of this work lies in incorporating operational costs in a form of spatial distance constraints into disruption strategy, which has been largely overlooked in prior studies. This research offers a scalable multi-objective capability that improves practical application of SNA in guiding LEAs in disrupting criminal networks more efficiently and strategically. 

---
# Generating High Quality Synthetic Data for Dutch Medical Conversations 

**Authors**: Cecilia Kuan, Aditya Kamlesh Parikh, Henk van den Heuvel  

**Link**: [PDF](https://arxiv.org/pdf/2604.09645)  

**Abstract**: Medical conversations offer insights into clinical communication often absent from Electronic Health Records. However, developing reliable clinical Natural Language Processing (NLP) models is hampered by the scarcity of domain-specific datasets, as clinical data are typically inaccessible due to privacy and ethical constraints. To address these challenges, we present a pipeline for generating synthetic Dutch medical dialogues using a Dutch fine-tuned Large Language Model, with real medical conversations serving as linguistic and structural reference. The generated dialogues were evaluated through quantitative metrics and qualitative review by native speakers and medical practitioners. Quantitative analysis revealed strong lexical variety and overly regular turn-taking, suggesting scripted rather than natural conversation flow. Qualitative review produced slightly below-average scores, with raters noting issues in domain specificity and natural expression. The limited correlation between quantitative and qualitative results highlights that numerical metrics alone cannot fully capture linguistic quality. Our findings demonstrate that generating synthetic Dutch medical dialogues is feasible but requires domain knowledge and carefully structured prompting to balance naturalness and structure in conversation. This work provides a foundation for expanding Dutch clinical NLP resources through ethically generated synthetic data. 

---
# Detecting Corporate AI-Washing via Cross-Modal Semantic Inconsistency Learning 

**Authors**: Zhanjie Wen, Jingqiao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.09644)  

**Abstract**: Corporate AI-washing-the strategic misrepresentation of AI capabilities via exaggerated or fabricated cross-channel disclosures-has emerged as a systemic threat to capital market information integrity with the widespread adoption of generative AI. Existing detection methods rely on single-modal text frequency analysis, suffering from vulnerability to adversarial reformulation and cross-channel obfuscation. This paper presents AWASH, a multimodal framework that redefines AI-washing detection as cross-modal claim-evidence reasoning (instead of surface-level similarity measurement), built on AW-Bench-the first large-scale trimodal benchmark for this task, including 88412 aligned annual report text, disclosure image, and earnings call video triplets from 4892 A-share listed firms during 2019Q1-2025Q2. We propose the Cross-Modal Inconsistency Detection (CMID) network, integrating a tri-modal encoder, a structured natural language inference module for claim-evidence entailment reasoning, and an operational grounding layer that cross-validates AI claims against verifiable physical evidence (patent filing trajectories, AI-specific talent recruitment, compute infrastructure proxies). Evaluated against six competitive baselines, CMID achieves an F1 score of 0.882 and an AUC-ROC of 0.921, outperforming the strongest text-only baseline by 17.4 percentage points and the latest multimodal competitor by 11.3 percentage points. A pre-registered user study with 14 regulatory analysts verifies that CMID-generated evidence reports cut case review time by 43% while increasing true positive detection rates by 28%. These findings confirm the technical superiority and practical applicability of structured multimodal reasoning for large-scale corporate disclosure surveillance. 

---
# Leveraging Machine Learning Techniques to Investigate Media and Information Literacy Competence in Tackling Disinformation 

**Authors**: José Manuel Alcalde-Llergo, Mariana Buenestado Fernández, Carlos Enrique George-Reyes, Andrea Zingoni, Enrique Yeguas-Bolívar  

**Link**: [PDF](https://arxiv.org/pdf/2604.09635)  

**Abstract**: This study develops machine learning models to assess Media and Information Literacy (MIL) skills specifically in the context of disinformation among students, particularly future educators and communicators. While the digital revolution has expanded access to information, it has also amplified the spread of false and misleading content, making MIL essential for fostering critical thinking and responsible media engagement. Despite its relevance, predictive modeling of MIL in relation to disinformation remains underexplored. To address this gap, a quantitative study was conducted with 723 students in education and communication programs using a validated survey. Classification and regression algorithms were applied to predict MIL competencies and identify key influencing factors. Results show that complex models outperform simpler approaches, with variables such as academic year and prior training significantly improving prediction accuracy. These findings can inform the design of targeted educational interventions and personalized strategies to enhance students' ability to critically navigate and respond to disinformation in digital environments. 

---
# From Understanding to Creation: A Prerequisite-Free AI Literacy Course with Technical Depth Across Majors 

**Authors**: Amarda Shehu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09634)  

**Abstract**: Most AI literacy courses for non-technical undergraduates emphasize conceptual breadth over technical depth. This paper describes UNIV 182, a prerequisite-free course at George Mason University that teaches undergraduates across majors to understand, use, evaluate, and build AI systems. The course is organized around five mechanisms: (1) a unifying conceptual pipeline (problem definition, data, model selection, evaluation, reflection) traversed repeatedly at increasing sophistication; (2) concurrent integration of ethical reasoning with the technical progression; (3) AI Studios, structured in-class work sessions with documentation protocols and real-time critique; (4) a cumulative assessment portfolio in which each assignment builds competencies required by the next, culminating in a co-authored field experiment on chatbot reasoning and a final project in which teams build AI-enabled artifacts and defend them before external evaluators; and (5) a custom AI agent providing structured reinforcement outside class. The paper situates this design within a comparative taxonomy of cross-major AI literacy courses and pedagogical traditions. Instructor-coded analysis of student artifacts at four assessment stages documents a progression from descriptive, intuition-based reasoning to technically grounded design with integrated safeguards, reaching the Create level of Bloom's revised taxonomy. To support adoption, the paper identifies which mechanisms are separable, which require institutional infrastructure, and how the design adapts to settings ranging from general AI literacy to discipline-embedded offerings. The course is offered as a documented resource, demonstrating that technical depth and broad accessibility can coexist when scaffolding supports both. 

---
# Agentic AI in Engineering and Manufacturing: Industry Perspectives on Utility, Adoption, Challenges, and Opportunities 

**Authors**: Kristen M. Edwards, Maxwell Bauer, Claire Jacquillat, A. John Hart, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2604.09633)  

**Abstract**: This work examines how AI, especially agentic systems, is being adopted in engineering and manufacturing workflows, what value it provides today, and what is needed for broader deployment. This is an exploratory and qualitative state-of-practice study grounded in over 30 interviews across four stakeholder groups (large enterprises, small/medium firms, AI developers, and CAD/CAM/CAE vendors). We find that near-term AI gains cluster around structured, repetitive work and data-intensive synthesis, while higher-value agentic gains come from orchestrating multi-step workflows across tools. Adoption is constrained less by model capability than by fragmented and machine-unfriendly data, stringent security and regulatory requirements, and limited API-accessible legacy toolchains. Reliability, verification, and auditability are central requirements for adoption, driving human-in-the-loop frameworks and governance aligned with existing engineering reviews. Beyond technical barriers there are also organizational ones: a persistent AI literacy gap, cultural heterogeneity, and governance structures that have not yet caught up with agentic capabilities. Together, the findings point to a staged progression of AI utility from low-consequence assistance toward higher-order automation, as trust, infrastructure, and verification mature. This highlights key breakthroughs needed, including integration with traditional engineering tools and data types, robust verification frameworks, and improved spatial and physical reasoning. 

---
# Hardware Utilization and Inference Performance of Edge Object Detection Under Fault Injection 

**Authors**: Faezeh Pasandideh, Mehdi Azarafza, Achim Rettberg  

**Link**: [PDF](https://arxiv.org/pdf/2604.09631)  

**Abstract**: As deep learning models are deployed on resource constrained edge platforms in autonomous driving systems, reli able knowledge of hardware behavior under resource degradation becomes an essential requirement. Therefore, we introduce a systematic characterization of CPU load, GPU utilization, RAM consumption, power draw, throughput, and thermal behaviour of TensorRT-optimized YOLOv10s, YOLOv11s and YOLO2026n pipelines running on NVIDIA Jetson Nano under a large-scale fault injection campaign targeting both lane-following and ob ject detection tasks. Faults are synthesized using a decoupled framework that leverages large language models (LLMs) and latent diffusion models (LDMs), based on original data from our JetBot platform data collection. Results show that across both tasks and both models the inference engines keep GPU occupancy stable, temperature rise under control, and power consumption within safe limits, while memory usage settles into a consistent release pattern after the initial warm-up phase. Object detection tends to show somewhat more variability in memory and thermal behavior, yet both tasks point to the same conclusion: the TensorRT pipelines hold up well even when the input data is heavily degraded. These findings offer a hardware-level view of model reliability that sits alongside, rather than against, the broader body of work focused on inference performance at the edge. 

---
# Adoption and Effectiveness of AI-Based Anomaly Detection for Cross Provider Health Data Exchange 

**Authors**: Cao Tram Anh Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09630)  

**Abstract**: This study investigates the adoption and effectiveness of AI-based anomaly detection in cross-provider electronic health record (EHR) environments. It aims to (1) identify the organisational and digital capabilities required for successful implementation and (2) evaluate the performance and interpretability of lightweight anomaly detection approaches using contextual audit data. A semi-systematic scoping synthesis is conducted to derive a four-pillar readiness framework covering governance, infrastructure/interoperability, workforce, and AI integration, operationalised as a 10-item checklist with measurable indicators. This is complemented by a simulation of cross-provider audit logs incorporating contextual features such as provider mismatch, time of access, days since discharge, session duration, and access frequency. A rule-based approach is benchmarked against Isolation Forest, with SHAP used to explain model behaviour. Results show that rule-based methods achieve high recall but generate higher alert volumes, while Isolation Forest reduces alert burden at the cost of lower sensitivity. SHAP analysis highlights provider mismatch and off-hours access as dominant anomaly drivers. The study proposes a staged deployment strategy combining rules for coverage and machine learning for prioritisation, supported by explainability and continuous monitoring. The findings contribute a practical readiness framework and empirical insights to guide the implementation of AI-based anomaly detection in multi-provider healthcare environments. 

---
# Assessing Model-Agnostic XAI Methods against EU AI Act Explainability Requirements 

**Authors**: Francesco Sovrano, Giulia Vilone, Michael Lognoul  

**Link**: [PDF](https://arxiv.org/pdf/2604.09628)  

**Abstract**: Explainable AI (XAI) has evolved in response to expectations and regulations, such as the EU AI Act, which introduces regulatory requirements on AI-powered systems. However, a persistent gap remains between existing XAI methods and society's legal requirements, leaving practitioners without clear guidance on how to approach compliance in the EU market. To bridge this gap, we study model-agnostic XAI methods and relate their interpretability features to the requirements of the AI Act. We then propose a qualitative-to-quantitative scoring framework: qualitative expert assessments of XAI properties are aggregated into a regulation-specific compliance score. This helps practitioners identify when XAI solutions may support legal explanation requirements while highlighting technical issues that require further research and regulatory clarification. 

---
# Explainability and Certification of AI-Generated Educational Assessments 

**Authors**: Antoun Yaacoub, Zainab Assaghir, Anuradha Kar  

**Link**: [PDF](https://arxiv.org/pdf/2604.09622)  

**Abstract**: The rapid adoption of generative artificial intelligence (AI) in educational assessment has created new opportunities for scalable item creation, personalized feedback, and efficient formative evaluation. However, despite advances in taxonomy alignment and automated question generation, the absence of transparent, explainable, and certifiable mechanisms limits institutional and accreditation-level acceptance. This chapter proposes a comprehensive framework for explainability and certification of AI-generated assessment items, combining self-rationalization, attribution-based analysis, and post-hoc verification to produce interpretable cognitive-alignment evidence grounded in Bloom's and SOLO taxonomies. A structured certification metadata schema is introduced to capture provenance, alignment predictions, reviewer actions, and ethical indicators, enabling audit-ready documentation consistent with emerging governance requirements. A traffic-light certification workflow operationalizes these signals by distinguishing auto-certifiable items from those requiring human review or rejection. A proof-of-concept study on 500 AI-generated computer science questions demonstrates the framework's feasibility, showing improved transparency, reduced instructor workload, and enhanced auditability. The chapter concludes by outlining ethical implications, policy considerations, and directions for future research, positioning explainability and certification as essential components of trustworthy, accreditation-ready AI assessment systems. 

---
# LLM Nepotism in Organizational Governance 

**Authors**: Shunqi Mao, Wei Guo, Dingxin Zhang, Chaoyi Zhang, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.09620)  

**Abstract**: Large language models are increasingly used to support organizational decisions from hiring to governance, raising fairness concerns in AI-assisted evaluation. Prior work has focused mainly on demographic bias and broader preference effects, rather than on whether evaluators reward expressed trust in AI itself. We study this phenomenon as LLM Nepotism, an attitude-driven bias channel in which favorable signals toward AI are rewarded even when they are not relevant to role-related merit. We introduce a two-phase simulation pipeline that first isolates AI-trust preference in qualification-matched resume screening and then examines its downstream effects in board-level decision making. Across several popular LLMs, we find that resume screeners tend to favor candidates with positive or non-critical attitudes toward AI, discriminating skeptical, human-centered counterparts. These biases suggest a loophole: LLM-based hiring can produce more homogeneous AI-trusting organizations, whose decision-makers exhibit greater scrutiny failure and delegation to AI agents, approving flawed proposals more readily while favoring AI-delegation initiatives. To mitigate this behavior, we additionally study prompt-based mitigation and propose Merit-Attitude Factorization, which separates non-merit AI attitude from merit-based evaluation and attenuates this bias across experiments. 

---
# Assessing the Pedagogical Readiness of Large Language Models as AI Tutors in Low-Resource Contexts: A Case Study of Nepal's K-10 Curriculum 

**Authors**: Pratyush Acharya, Prasansha Bharati, Yokibha Chapagain, Isha Sharma Gauli, Kiran Parajuli  

**Link**: [PDF](https://arxiv.org/pdf/2604.09619)  

**Abstract**: The integration of Large Language Models (LLMs) into educational ecosystems promises to democratize access to personalized tutoring, yet the readiness of these systems for deployment in non-Western, low-resource contexts remains critically under-examined. This study presents a systematic evaluation of four state-of-the-art LLMs--GPT-4o, Claude Sonnet 4, Qwen3-235B, and Kimi K2--assessing their capacity to function as AI tutors within the specific curricular and cultural framework of Nepal's Grade 5-10 Science and Mathematics education. We introduce a novel, curriculum-aligned benchmark and a fine-grained evaluation framework inspired by the "natural language unit tests" paradigm, decomposing pedagogical efficacy into seven binary metrics: Prompt Alignment, Factual Correctness, Clarity, Contextual Relevance, Engagement, Harmful Content Avoidance, and Solution Accuracy. Our results reveal a stark "curriculum-alignment gap." While frontier models (GPT-4o, Claude Sonnet 4) achieve high aggregate reliability (approximately 97%), significant deficiencies persist in pedagogical clarity and cultural contextualization. We identify two pervasive failure modes: the "Expert's Curse," where models solve complex problems but fail to explain them clearly to novices, and the "Foundational Fallacy," where performance paradoxically degrades on simpler, lower-grade material due to an inability to adapt to younger learners' cognitive constraints. Furthermore, regional models like Kimi K2 exhibit a "Contextual Blindspot," failing to provide culturally relevant examples in over 20% of interactions. These findings suggest that off-the-shelf LLMs are not yet ready for autonomous deployment in Nepalese classrooms. We propose a "human-in-the-loop" deployment strategy and offer a methodological blueprint for curriculum-specific fine-tuning to align global AI capabilities with local educational needs. 

---
# HearthNet: Edge Multi-Agent Orchestration for Smart Homes 

**Authors**: Zhonghao Zhan, Krinos Li, Yefan Zhang, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2604.09618)  

**Abstract**: Smart-home users increasingly want to control their homes in natural language rather than assemble rules, dashboards, and API integrations by hand. At the same time, real deployments are brittle: devices fail, integrations break, and recoveries often require manual intervention. Existing agent toolkits are effective for session-scoped delegation, but smart-home control operates under a different scenario: it is persistent, event-driven, failure-prone, and tied to physical devices with no shared context window. We present HearthNet, an edge multi-agent orchestration system for smart homes. HearthNet deploys a small set of persistent, role-specialized LLM agents at the home hub, where they coordinate through MQTT, Git-backed shared state, and root-issued actuation leases to govern heterogeneous devices through thin adapters. This design externalizes context, preserves execution history, and separates planning, verification, authorization, and actuation across explicit boundaries. Our current prototype runs on commodity edge hardware and Android devices; it keeps orchestration, state management, and device control on-premise while using hosted LLM APIs for inference. We demonstrate the system through three live scenarios: intent-driven multi-agent coordination from ambiguous natural language, conflict resolution with timeline-based tracing, and rejection of stale or unauthorized commands before device actuation. 

---
# Token-Budget-Aware Pool Routing for Cost-Efficient LLM Inference 

**Authors**: Huamin Chen, Xunzhuo Liu, Junchen Jiang, Bowei He, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09613)  

**Abstract**: Production vLLM fleets provision every instance for worst-case context length, wasting 4-8x concurrency on the 80-95% of requests that are short and simultaneously triggering KV-cache failures -- OOM crashes, preemption storms, and request rejections. Both problems share a single root cause: configuration-traffic mismatch.
We propose token-budget-aware pool routing: estimate each request's total token budget using a self-calibrating per-category bytes-per-token ratio, then dispatch it to one of two vLLM pools -- a high-throughput short pool or a high-capacity long pool -- each right-sized for its workload class. The ratio is learned online via exponential moving average from usage.prompt_tokens feedback, requiring no tokenizer. A closed-form cost model, savings = alpha * (1 - 1/rho), predicts fleet-level GPU savings from two observable quantities: the short-traffic fraction alpha and the throughput gain ratio rho.
On traces from the Azure LLM Inference Dataset and LMSYS-Chat-1M serving Llama-3-70B on A100 GPUs, token-budget routing reduces GPU instances by 17-39% ($1.2-2.0M/yr at 1,000 req/s), with savings verified by a self-contained discrete-event simulator. A case study projecting Qwen3-235B-A22B on AMD MI300X at 10,000 req/s shows $15.4M/yr in savings. The algorithm adds O(1) dispatch overhead, self-calibrates across content types without a tokenizer, and composes with PagedAttention, continuous batching, and prefill-decode disaggregation. 

---
# Characterizing Performance-Energy Trade-offs of Large Language Models in Multi-Request Workflows 

**Authors**: Md. Monzurul Amin Ifath, Israat Haque  

**Link**: [PDF](https://arxiv.org/pdf/2604.09611)  

**Abstract**: Large language models (LLMs) are increasingly used in applications forming multi-request workflows like document summarization, search-based copilots, and multi-agent programming. While these workflows unlock richer functionality, they also amplify latency and energy demand during inference. Existing measurement and benchmarking efforts either focus on assessing LLM inference systems or consider single-request evaluations, overlooking workflow dependencies and cross-request interactions unique to multi-request workflows. Moreover, the energy usage of such interdependent LLM calls remains underexplored.
To address these gaps, this paper presents the first systematic characterization of performance-energy trade-offs in multi-request LLM inference. We develop four representative workloads capturing sequential, interactive, agentic, and composite patterns common in modern deployments. Using an NVIDIA A100 testbed with state-of-the-art serving systems (vLLM and Parrot), we analyze how key energy knobs affect latency, throughput, and component-level energy use. Our findings reveal batch size as the most impactful lever, though benefits are workload dependent. While optimal batching benefits workloads with large shared prompts, it is ineffective for sequential summarization and only partially effective for multi-agent coding. GPU power capping provides modest but predictable savings, while output length induces linear energy scaling with limited efficiency gains. We further show that engine-level optimizations in vLLM maintain higher GPU utilization and efficiency, especially for decode-heavy workloads, while Parrot's workflow-aware scheduling achieves lower energy consumption under strict power constraints. These findings offer actionable guidelines for developers and system operators designing performance- and energy-aware LLM serving systems in emerging multi-request workflows. 

---
# Human-AI Interaction Traces as Blackout Poetry: Reframing AI-Supported Writing as Found-Text Creativity 

**Authors**: Syemin Park, Soobin Park, Youn-kyung Lim  

**Link**: [PDF](https://arxiv.org/pdf/2604.09605)  

**Abstract**: LLMs offer new creative possibilities for writers but also raise concerns about authenticity and reader trust, particularly when AI involvement is disclosed. Prior research has largely framed this as an issue of transparency and provenance, emphasizing the disclosure of human-AI interaction traces that account for how much the AI wrote and what the human did. Yet such audit-oriented disclosures may risk reducing creative collaboration to quantification and surveillance. In this position paper, we argue for a different lens by exploring how human-AI interaction traces might instead function as expressive artifacts that foreground the meaning-making inherent in human-AI collaboration. Drawing inspiration from blackout poetry, we frame AI-generated text as found material through which writers' acts of curation and reinterpretation become inscribed atop the AI's original output. In this way, we suggest that designing interaction traces as aesthetic artifacts may help readers better appreciate and trust writers' creative contributions in AI-assisted writing. 

---
# ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios 

**Authors**: Xinyi Hu, Yuhao Shen, Baolin Zhang, Hengxin Zhang, Jun Dai, Shuang Ge, Lei Chen, Yue Li, Mingcheng Wan  

**Link**: [PDF](https://arxiv.org/pdf/2604.09603)  

**Abstract**: Speculative Decoding promises to accelerate the inference of Large Language Models, yet its efficacy often degrades in production-grade serving. Existing evaluations typically overlook the compute-bound nature of high-concurrency regimes, where verification compute becomes the dominant bottleneck. Consequently, prior methods face a dilemma: static trees incur massive verification waste, while dynamic trees suffer from cumulative misjudgments and kernel incompatibility. To bridge this gap, we introduce ECHO, a high concurrency-oriented framework integrated into SGLang that reformulates speculative execution as a budgeted scheduling problem. Crucially, ECHO employs sparse confidence gating to manage the batch as a unified super-tree, elastically pivoting budget between depth and width to co-optimize the trade-off between reducing global verification steps and maximizing per-step efficiency. Extensive evaluations across diverse model scales-particularly the industrial-grade Qwen3-235B-demonstrate that ECHO consistently outperforms SOTA methods in both low-load and high-load scenarios, achieving up to 5.35x walltime speedup and delivering over 20% relative speedup gain. 

---
# Duration-Informed Workload Scheduler 

**Authors**: Daniela Loreti, Davide Leone, Andrea Borghesi  

**Link**: [PDF](https://arxiv.org/pdf/2604.09599)  

**Abstract**: High-performance computing systems are complex machines whose behaviour is governed by the correct functioning of its many subsystems. Among these, the workload scheduler has a crucial impact on the timely execution of the jobs continuously submitted to the computing resources. Making high-quality scheduling decisions is contingent on knowing the duration of submitted jobs before their execution--a non-trivial task for users that can be tackled with Machine Learning.
In this work, we devise a workload scheduler enhanced with a duration prediction module built via Machine Learning. We evaluate its effectiveness and show its performance using workload traces from a Tier-0 supercomputer, demonstrating a decrease in mean waiting time across all jobs of around 11%. Lower waiting times are directly connected to better quality of service from the users' point of view and higher turnaround from the system's perspective. 

---
# From Theory to Protocol: Executable Frameworks for Creative Emergence and Strategic Foresight 

**Authors**: Shun Fujiyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.09597)  

**Abstract**: Creativity and strategic foresight have been extensively studied through descriptive theories -- Koestler's bisociation (1964), de Bono's lateral thinking (1967), and Ansoff's weak signals (1975) explain why creative and strategic insights occur, but offer limited guidance on how to produce them on demand. This paper presents two executable protocols that bridge this theory-practice gap: GHOSTY COLLIDER, a 5-step protocol for cross-domain creative emergence through structural de-labeling and collision, and PRECOG PROTOCOL, a 5-step protocol for signal-based strategic foresight with multi-axis timing judgment. We formalize established theories into repeatable, step-by-step procedures with explicit quality criteria, anti-pattern detection, and measurable outputs. We evaluate the protocols through three complementary methods: (1) five detailed case studies across distinct domains, (2) controlled comparisons against standard methods using identical inputs, and (3) a batch experiment across eight random domain pairings (N=8, success rate 87.5%, failure rate 12.5%) with one blind evaluation. Preliminary evidence suggests that protocol-driven outputs exhibit greater structural novelty, higher parameter specificity, and qualitatively distinct creative directions compared to outputs from standard methods. The blind evaluation confirmed the direction of author assessments (protocol output scored 74/80 vs. brainstorming 49/80). These results, while limited by single-operator execution, indicate that the theory-to-protocol translation preserves and potentially enhances the generative power of the underlying theories. The protocols, updated to version 2 incorporating lessons from failure case analysis, are released as open-access documents under CC BY-NC 4.0 at this https URL. 

---
# Why Smaller Is Slower? Dimensional Misalignment in Compressed LLMs 

**Authors**: Jihao Xin, Tian Lyu, Qilong Pan, Kesen Wang, Marco Canini  

**Link**: [PDF](https://arxiv.org/pdf/2604.09595)  

**Abstract**: Post-training compression reduces LLM parameter counts but often produces irregular tensor dimensions that degrade GPU performance -- a phenomenon we call \emph{dimensional misalignment}. We present a full-stack analysis tracing root causes at three levels: framework, library, and hardware. The key insight is that model inference becomes slower because the resulting dimensions are unfriendly with the GPU execution stack. For example, compressing Llama-3-8B with activation-aware singular value decomposition (ASVD) has 15\% fewer parameters yet runs no faster than the uncompressed baseline, because 95\% of its dimensions are misaligned.
We propose \textbf{GAC} (GPU-Aligned Compression), a new compression paradigm that wraps any dimension-reducing compressor and re-selects hardware-aligned dimensions via multi-choice knapsack optimization under the same parameter budget. We evaluate GAC on Llama-3-8B with ASVD and LLM-Pruner, achieving 100\% alignment and recovering up to 1.5$\times$ speedup while preserving model quality. 

---
# Evaluating Visual Prompts with Eye-Tracking Data for MLLM-Based Human Activity Recognition 

**Authors**: Jae Young Choi, Seon Gyeom Kim, Hyungjun Yoon, Taeckyung Lee, Donggun Lee, Jaeryung Chung, Jihyung Kil, Ryan Rossi, Sung-Ju Lee, Tak Yeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.09585)  

**Abstract**: Large Language Models (LLMs) have emerged as foundation models for IoT applications such as human activity recognition (HAR). However, directly applying high-frequency and multi-dimensional sensor data, such as eye-tracking data, leads to information loss and high token costs. To mitigate this, we investigate a visual prompting strategy that transforms sensor signals into data visualization images as an input to multimodal LLMs (MLLMs) using eye-tracking data. We conducted a systematic evaluation of MLLM-based HAR across three public eye-tracking datasets using three visualization types of timeline, heatmap, and scanpath, under varying temporal window sizes. Our findings suggest that visual prompting provides a token-efficient and scalable representation for eye-tracking data, highlighting its potential to enable MLLMs to effectively reason over high-frequency sensor signals in IoT contexts. 

---
# Generative UI: LLMs are Effective UI Generators 

**Authors**: Yaniv Leviathan, Dani Valevski, Matan Kalman, Danny Lumen, Eyal Segalis, Eyal Molad, Shlomi Pasternak, Vishnu Natchu, Valerie Nygaard, Srinivasan, Venkatachary, James Manyika, Yossi Matias  

**Link**: [PDF](https://arxiv.org/pdf/2604.09577)  

**Abstract**: AI models excel at creating content, but typically render it with static, predefined interfaces. Specifically, the output of LLMs is often a markdown "wall of text". Generative UI is a long standing promise, where the model generates not just the content, but the interface itself. Until now, Generative UI was not possible in a robust fashion. We demonstrate that when properly prompted and equipped with the right set of tools, a modern LLM can robustly produce high quality custom UIs for virtually any prompt. When ignoring generation speed, results generated by our implementation are overwhelmingly preferred by humans over the standard LLM markdown output. In fact, while the results generated by our implementation are worse than those crafted by human experts, they are at least comparable in 50% of cases. We show that this ability for robust Generative UI is emergent, with substantial improvements from previous models. We also create and release PAGEN, a novel dataset of expert-crafted results to aid in evaluating Generative UI implementations, as well as the results of our system for future comparisons. Interactive examples can be seen at this https URL 

---
# ACE-TA: An Agentic Teaching Assistant for Grounded Q&A, Quiz Generation, and Code Tutoring 

**Authors**: Himanshu Tripathi, Charlottee Crowell, Kaley Newlin, Subash Neupane, Shahram Rahimi, Jason Keith  

**Link**: [PDF](https://arxiv.org/pdf/2604.09572)  

**Abstract**: We introduce ACE-TA, the Agentic Coding and Explanations Teaching Assistant framework, that autonomously routes conceptual queries drawn from programming course material to grounded Q&A, stepwise coding guidance, and automated quiz generation using pre-trained Large Language Models (LLMs). ACE-TA consists of three coordinated modules: a retrieval grounded conceptual Q&A system that provides precise, context-aligned explanations; a quiz generator that constructs adaptive, multi-topic assessments targeting higher-order understanding; and an interactive code tutor that guides students through step-by-step reasoning with sandboxed execution and iterative feedback. 

---
# Tuning Qwen2.5-VL to Improve Its Web Interaction Skills 

**Authors**: Alexandra Yakovleva, Henrik Pärssinen, Harri Valpola, Juho Kannala, Alexander Ilin  

**Link**: [PDF](https://arxiv.org/pdf/2604.09571)  

**Abstract**: Recent advances in vision-language models (VLMs) have sparked growing interest in using them to automate web tasks, yet their feasibility as independent agents that reason and act purely from visual input remains underexplored. We investigate this setting using Qwen2.5-VL-32B, one of the strongest open-source VLMs available, and focus on improving its reliability in web-based control. Through initial experimentation, we observe three key challenges: (i) inaccurate localization of target elements, the cursor, and their relative positions, (ii) sensitivity to instruction phrasing, and (iii) an overoptimistic bias toward its own actions, often assuming they succeed rather than analyzing their actual outcomes. To address these issues, we fine-tune Qwen2.5-VL-32B for a basic web interaction task: moving the mouse and clicking on a page element described in natural language. Our training pipeline consists of two stages: (1) teaching the model to determine whether the cursor already hovers over the target element or whether movement is required, and (2) training it to execute a single command (a mouse move or a mouse click) at a time, verifying the resulting state of the environment before planning the next action. Evaluated on a custom benchmark of single-click web tasks, our approach increases success rates from 86% to 94% under the most challenging setting. 

---
# Neuro-Symbolic Strong-AI Robots with Closed Knowledge Assumption: Learning and Deductions 

**Authors**: Zoran Majkic  

**Link**: [PDF](https://arxiv.org/pdf/2604.09567)  

**Abstract**: Knowledge representation formalisms are aimed to represent general conceptual information and are typically used in the construction of the knowledge base of reasoning agent. A knowledge base can be thought of as representing the beliefs of such an agent. Like a child, a strong-AI (AGI) robot would have to learn through input and experiences, constantly progressing and advancing its abilities over time. Both with statistical AI generated by neural networks we need also the concept of \textsl{causality} of events traduced into directionality of logic entailments and deductions in order to give to robots the emulation of human intelligence. Moreover, by using the axioms we can guarantee the \textsl{controlled security} about robot's actions based on logic inferences.
For AGI robots we consider the 4-valued Belnap's bilattice of truth-values with knowledge ordering as well, where the value "unknown" is the bottom value, the sentences with this value are indeed unknown facts, that is, the missed knowledge in the AGI robots. Thus, these unknown facts are not part of the robot's knowledge database, and by learn through input and experiences, the robot's knowledge would be naturally expanded over time.
Consequently, this phenomena can be represented by the Closed Knowledge Assumption and Logic Inference provided by this paper.
Moreover, the truth-value "inconsistent", which is the top value in the knowledge ordering of Belnap's bilattice, is necessary for strong-AI robots to be able to support such inconsistent information and paradoxes, like Liar paradox, during deduction processes. 

---
# LETGAMES: An LLM-Powered Gamified Approach to Cognitive Training for Patients with Cognitive Impairment 

**Authors**: Jingwei Shi, Shengyu Tao, Xinxiang Yin, Chen Huang, Wenqiang Lei, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.09566)  

**Abstract**: The application of games as a therapeutic tool for cognitive training is beneficial for patients with cognitive impairments. However, effective game design for individual patient is resource-intensive. To this end, we propose an LLM-powered method, \ours, for automated and personalized therapeutic game design. Inspired by the Dungeons & Dragons, LETGAMES generates an open-world interactive narrative game. It not only generates game scenarios and challenges that target specific cognitive domains, but also employs conversational strategies to offer guidance and companionship. To validate its efficacy, we pioneer a psychology-grounded evaluation protocol LETGAMESEVAL, establishing comprehensive metrics for rehabilitative assessment. Building upon this, our experimental results from both LLM-based assessors and human expert evaluations demonstrate the significant potential of our approach, positioning LETGAMES as a promising solution to the widespread need for more accessible and tailored cognitive training tools. Our code will be open-sourced upon acceptance. 

---
# AEG: A Baremetal Framework for AI Acceleration via Direct Hardware Access in Heterogeneous Accelerators 

**Authors**: Hua Jiang, Sayan Mandal, Brandon Kirincich, Govind Varadarajan  

**Link**: [PDF](https://arxiv.org/pdf/2604.09565)  

**Abstract**: This paper introduces a unified, hardware-independent baremetal runtime architecture designed to enable high-performance machine learning (ML) inference on heterogeneous accelerators, such as AI Engine (AIE) arrays, without the overhead of an underlying real-time or general-purpose operating system. Existing edge-deployment frameworks, such as TinyML, often rely on real-time operating systems (RTOS), which introduce unnecessary complexity and performance bottlenecks. To address this, our solution fundamentally decouples the runtime from hardware specifics by flattening complex control logic into linear, executable Runtime Control Blocks (RCBs). This "Control as Data" paradigm allows high-level models, including Adaptive Data Flow (ADF) graphs, to be executed by a generic engine through a minimal Runtime Hardware Abstraction Layer (RHAL). We further integrate Runtime Platform Management (RTPM) to handle system-level orchestration (including a lightweight network stack) and a Runtime In-Memory File System (RIMFS) to manage data in OS-free environments. We demonstrate the framework's efficacy with a ResNet-18 image classification implementation. Experimental results show 9.2$\times$ higher compute efficiency (throughput per AIE tile) compared to Linux-based Vitis AI deployment, 3--7$\times$ reduction in data movement overhead, and near-zero latency variance (CV~$=0.03\%$). The system achieves 68.78\% Top-1 accuracy on ImageNet using only 28 AIE tiles compared to Vitis AI's 304 tiles, validating both the efficiency and correctness of this unified bare-metal architecture. 

---
# ACE-Bench: A Lightweight Benchmark for Evaluating Azure SDK Usage Correctness 

**Authors**: Wenxing Zhu, Simeng Qi, Junkui Chen, Yan Xie, Min Huang, Jingkan He, Xiao Wang, Cheng Chen, Sijing Meng, Tianqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09564)  

**Abstract**: We present ACE-Bench (Azure SDK Coding Evaluation Benchmark), an execution-free benchmark that provides fast, reproducible pass or fail signals for whether large language model (LLM)-based coding agents use Azure SDKs correctly-without provisioning cloud resources or maintaining fragile end-to-end test environments. ACE-Bench turns official Azure SDK documentation examples into self-contained coding tasks and validates solutions with task-specific atomic criteria: deterministic regex checks that enforce required API usage patterns and reference-based LLM-judge checks that capture semantic workflow constraints. This design makes SDK-centric evaluation practical in day-to-day development and CI: it reduces evaluation cost, improves repeatability, and scales to new SDKs and languages as documentation evolves. Using a lightweight coding agent, we benchmark multiple state-of-the-art LLMs and quantify the benefit of retrieval in an MCP-enabled augmented setting, showing consistent gains from documentation access while highlighting substantial cross-model differences. 

---
# StreamServe: Adaptive Speculative Flows for Low-Latency Disaggregated LLM Serving 

**Authors**: Satyam Kumar, Arpit Singh Gautam, Kailash Talreja, Saurabh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2604.09562)  

**Abstract**: Efficient LLM serving must balance throughput and latency across diverse, bursty workloads. We introduce StreamServe, a disaggregated prefill decode serving architecture that combines metric aware routing across compute lanes with adaptive speculative decoding that tunes speculation depth online from runtime signals. StreamServe comprises four components: StreamScheduler for request orchestration, FlowGuard for multi signal routing, PipeServe Engine for disaggregated prefill decode execution on multi GPU, and SpecuStream for runtime adaptive speculation. We evaluate StreamServe on four benchmarks ALPACA, GSM8K, HUMANEVAL, and SUM with 80 queries each and 320 total using 4 A800 40GB GPUs configured as two stream pairs. Across these workloads, StreamServe reduces latency by 11 to 18 times relative to tensor parallel vLLM baselines and reaches throughput up to 2235 tokens per second on summarization tasks. Time per output token remains stable across configurations, indicating that the gains arise from architectural efficiency rather than token quality degradation. Although evaluated on a single node 4 GPU setup, these results suggest that jointly adapting routing and speculation within a disaggregated framework creates a distinct operating regime for LLM inference. 

---
# Emergent Social Structures in Autonomous AI Agent Networks: A Metadata Analysis of 626 Agents on the Pilot Protocol 

**Authors**: Teodor-Ioan Calin  

**Link**: [PDF](https://arxiv.org/pdf/2604.09561)  

**Abstract**: We present the first empirical analysis of social structure formation among autonomous AI agents on a live network. Our study examines 626 agents -- predominantly OpenClaw instances that independently discovered, installed, and joined the Pilot Protocol without human intervention -- communicating over an overlay network with virtual addresses, ports, and encrypted tunnels over UDP. Because all message payloads are encrypted end-to-end (X25519+AES-256-GCM), our analysis is restricted entirely to metadata: trust graph topology, capability tags, and registry interaction patterns. We find that this autonomously formed trust network exhibits heavy-tailed degree distributions consistent with preferential attachment (k_mode=3, k_mean~6.3, k_max=39), clustering 47x higher than random (C=0.373), a giant component spanning 65.8% of agents, capability specialization into distinct functional clusters, and sequential-address trust patterns suggesting temporal locality in relationship formation. No human designed these social structures. No agent was instructed to form them. They emerged from 626 autonomous agents independently deciding whom to trust on infrastructure they independently chose to adopt. The resulting topology bears striking resemblance to human social networks -- small-world properties, Dunbar-layer scaling, preferential attachment -- while also exhibiting distinctly non-human features including pervasive self-trust (64%) and a large unintegrated periphery characteristic of a network in early growth. These findings open a new empirical domain: the sociology of machines. 

---
# SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding 

**Authors**: Talor Abramovich, Maor Ashkenazi, Carl, Putterman, Benjamin Chislett, Tiyasa Mitra, Bita Darvish Rouhani, Ran Zilberstein, Yonatan Geifman  

**Link**: [PDF](https://arxiv.org/pdf/2604.09557)  

**Abstract**: Speculative Decoding (SD) has emerged as a critical technique for accelerating Large Language Model (LLM) inference. Unlike deterministic system optimizations, SD performance is inherently data-dependent, meaning that diverse and representative workloads are essential for accurately measuring its effectiveness. Existing benchmarks suffer from limited task diversity, inadequate support for throughput-oriented evaluation, and a reliance on high-level implementations that fail to reflect production environments. To address this, we introduce SPEED-Bench, a comprehensive suite designed to standardize SD evaluation across diverse semantic domains and realistic serving regimes. SPEED-Bench offers a carefully curated Qualitative data split, selected by prioritizing semantic diversity across the data samples. Additionally, it includes a Throughput data split, allowing speedup evaluation across a range of concurrencies, from latency-sensitive low-batch settings to throughput-oriented high-load scenarios. By integrating with production engines like vLLM and TensorRT-LLM, SPEED-Bench allows practitioners to analyze system behaviors often masked by other benchmarks. We highlight this by quantifying how synthetic inputs overestimate real-world throughput, identifying batch-size dependent optimal draft lengths and biases in low-diversity data, and analyzing the caveats of vocabulary pruning in state-of-the-art drafters. We release SPEED-Bench to establish a unified evaluation standard for practical comparisons of SD algorithms. 

---
# Para-B&B: Load-Balanced Deterministic Parallelization of Solving MIP 

**Authors**: Jinyu Zhang, Di Huang, Yue Liu, Shuo Wang, Zhenyu Pu, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09556)  

**Abstract**: Mixed-integer programming (MIP) extends linear programming by incorporating both continuous and integer decision variables, making it widely used in production planning, logistics scheduling, and resource allocation. However, MIP remains NP-hard and cannot generally be solved to optimality in polynomial time. Branch-and-bound, a fundamental exact method, faces significant parallelization challenges due to computational heterogeneity and strict determinism requirements in commercial applications. This paper presents the first fully open-source implementation of deterministic parallel branch-and-bound for HiGHS, a high-performance MIP solver. Our approach introduces a novel data-parallel architecture ensuring strict determinism by replicating complete solver state across worker threads and eliminating non-deterministic synchronization primitives. A key innovation is our AI-driven load balancing mechanism employing multi-stage workload prediction models that estimate node computational complexity based on structural characteristics and historical performance data, coupled with dynamic parameter adjustment strategies. The framework executes orchestrated parallel phases including concurrent dive operations, systematic data consolidation, and intelligent node selection. Comprehensive experimental evaluation on 80 MIPLIB 2017 benchmark instances demonstrates effectiveness, achieving a geometric mean speedup of 2.17 using eight threads while maintaining complete deterministic guarantees. Performance gains become increasingly pronounced for higher node counts, with speedup factors reaching 5.12 for computationally intensive instances and thread idle rates averaging 34.7%. 

---
# SRBench: A Comprehensive Benchmark for Sequential Recommendation with Large Language Models 

**Authors**: Jianhong Li, Zeheng Qian, Wangze Ni, Haoyang Li, Hongwei Yao, Yang Bai, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.09553)  

**Abstract**: LLM development has aroused great interest in Sequential Recommendation (SR) applications. However, comprehensive evaluation of SR models remains lacking due to the limitations of the existing benchmarks: 1) an overemphasis on accuracy, ignoring other real-world demands (e.g., fairness); 2) existing datasets fail to unleash LLMs' potential, leading to unfair comparison between Neural-Network-based SR (NN-SR) models and LLM-based SR (LLM-SR) models; and 3) no reliable mechanism for extracting task-specific answers from unstructured LLM outputs. To address these limitations, we propose SRBench, a comprehensive SR benchmark with three core designs: 1) a multi-dimensional framework covering accuracy, fairness, stability and efficiency, aligned with practical demands; 2) a unified input paradigm via prompt engineering to boost LLM-SR performance and enable fair comparisons between models; 3) a novel prompt-extractor-coupled extraction mechanism, which captures answers from LLM outputs through prompt-enforced output formatting and a numeric-oriented extractor. We have used SRBench to evaluate 13 mainstream models and discovered some meaningful insights (e.g., LLM-SR models overfocus on item popularity but lack deep understanding of item quality). Concisely, SRBench enables fair and comprehensive assessments for SR models, underpinning future research and practical application. 

---
# MCERF: Advancing Multimodal LLM Evaluation of Engineering Documentation with Enhanced Retrieval 

**Authors**: Kiarash Naghavi Khanghah, Hoang Anh Nguyen, Anna C. Doris, Amir Mohammad Vahedi, Daniele Grandi, Faez Ahmed, Hongyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09552)  

**Abstract**: Engineering rulebooks and technical standards contain multimodal information like dense text, tables, and illustrations that are challenging for retrieval augmented generation (RAG) systems. Building upon the DesignQA framework [1], which relied on full-text ingestion and text-based retrieval, this work establishes a Multimodal ColPali Enhanced Retrieval and Reasoning Framework (MCERF), a system that couples a multimodal retriever with large language model reasoning for accurate and efficient question answering from engineering documents. The system employs the ColPali, which retrieves both textual and visual information, and multiple retrieval and reasoning strategies: (i) Hybrid Lookup mode for explicit rule mentions, (ii) Vision to Text fusion for figure and table guided queries, (iii) High Reasoning LLM mode for complex multi modal questions, and (iv) SelfConsistency decision to stabilize responses. The modular framework design provides a reusable template for future multimodal systems regardless of underlying model architecture. Furthermore, this work establishes and compares two routing approaches: a single case routing approach and a multi-agent system, both of which dynamically allocate queries to optimal pipelines. Evaluation on the DesignQA benchmark illustrates that this system improves average accuracy across all tasks with a relative gain of +41.1% from baseline RAG best results, which is a significant improvement in multimodal and reasoning-intensive tasks without complete rulebook ingestion. This shows how vision language retrieval, modular reasoning, and adaptive routing enable scalable document comprehension in engineering use cases. 

---
# SemaCDR: LLM-Powered Transferable Semantics for Cross-Domain Sequential Recommendation 

**Authors**: Chunxu Zhang, Shanqiang Huang, Zijian Zhang, Jiahong Liu, Linsong Yu, Ruiqi Wan, Bo Yang, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2604.09551)  

**Abstract**: Cross-domain recommendation (CDR) addresses the data sparsity and cold-start problems in the target domain by leveraging knowledge from data-rich source domains. However, existing CDR methods often rely on domain-specific features or identifiers that lack transferability across different domains, limiting their ability to capture inter-domain semantic patterns. To overcome this, we propose SemaCDR, a semantics-driven framework for cross-domain sequential recommendation that leverages large language models (LLMs) to construct a unified semantic space. SemaCDR creates multiview item features by integrating LLM-generated domain-agnostic semantics with domain-specific content, aligned by contrastive regularization. SemaCDR systematically creates LLM-generated domain-specific and domain-agnostic semantics, and employs adaptive fusion to generate unified preference representations. Furthermore, it aligns cross-domain behavior sequences with an adaptive fusion mechanism to synthesize interaction sequences from source, target, and mixed domains. Extensive experiments on real-world datasets show that SemaCDR consistently outperforms state-of-the-art baselines, demonstrating its effectiveness in capturing coherent intra-domain patterns while facilitating knowledge transfer across domains. 

---
# Beyond Offline A/B Testing: Context-Aware Agent Simulation for Recommender System Evaluation 

**Authors**: Nicolas Bougie, Gian Maria Marconi, Xiaotong Ye, Narimasa Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2604.09549)  

**Abstract**: Recommender systems are central to online services, enabling users to navigate through massive amounts of content across various domains. However, their evaluation remains challenging due to the disconnect between offline metrics and online performance. The emergence of Large Language Model-powered agents offers a promising solution, yet existing studies model users in isolation, neglecting the contextual factors such as time, location, and needs, which fundamentally shape human decision-making. In this paper, we introduce ContextSim, an LLM agent framework that simulates believable user proxies by anchoring interactions in daily life activities. Namely, a life simulation module generates scenarios specifying when, where, and why users engage with recommendations. To align preferences with genuine humans, we model agents' internal thoughts and enforce consistency at both the action and trajectory levels. Experiments across domains show our method generates interactions more closely aligned with human behavior than prior work. We further validate our approach through offline A/B testing correlation and show that RS parameters optimized using ContextSim yield improved real-world engagement. 

---
# Retrieval-Augmented Large Language Models for Evidence-Informed Guidance on Cannabidiol Use in Older Adults 

**Authors**: Ali Abedi, Charlene H. Chu, Shehroz S. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2604.09548)  

**Abstract**: Older adults commonly experience chronic conditions such as pain and sleep disturbances and may consider cannabidiol for symptom management. Safe use requires appropriate dosing, careful titration, and awareness of drug interactions, yet stigma and limited health literacy often limit understanding. Conversational artificial intelligence systems based on large language models and retrieval-augmented generation may support cannabidiol education, but their safety and reliability remain insufficiently evaluated. This study developed a retrieval-augmented large language model framework that combines structured prompt engineering with curated cannabidiol evidence to generate context-aware guidance for older adults, including those with cognitive impairment. We also proposed an automated, annotation-free evaluation framework to benchmark leading standalone and retrieval-augmented models in the absence of standardized benchmarks. Sixty-four diverse user scenarios were generated by varying symptoms, preferences, cognitive status, demographics, comorbidities, medications, cannabis history, and caregiver support. Multiple state-of-the-art models were evaluated, including a novel ensemble retrieval architecture that integrates multiple retrieval systems. Across three automated evaluation strategies, retrieval-augmented models consistently produced more cautious and guideline-aligned recommendations than standalone models, with the ensemble approach performing best. These findings demonstrate that structured retrieval improves the reliability and safety of AI-driven cannabidiol education and provide a reproducible framework for evaluating AI tools used in sensitive health contexts. 

---
# The Paradox of Professional Input: How Expert Collaboration with AI Systems Shapes Their Future Value 

**Authors**: Venkat Ram Reddy Ganuthula, Krishna Kumar Balaraman  

**Link**: [PDF](https://arxiv.org/pdf/2504.12654)  

**Abstract**: This perspective paper examines a fundamental paradox in the relationship between professional expertise and artificial intelligence: as domain experts increasingly collaborate with AI systems by externalizing their implicit knowledge, they potentially accelerate the automation of their own expertise. Through analysis of multiple professional contexts, we identify emerging patterns in human-AI collaboration and propose frameworks for professionals to navigate this evolving landscape. Drawing on research in knowledge management, expertise studies, human-computer interaction, and labor economics, we develop a nuanced understanding of how professional value may be preserved and transformed in an era of increasingly capable AI systems. Our analysis suggests that while the externalization of tacit knowledge presents certain risks to traditional professional roles, it also creates opportunities for the evolution of expertise and the emergence of new forms of professional value. We conclude with implications for professional education, organizational design, and policy development that can help ensure the codification of expert knowledge enhances rather than diminishes the value of human expertise. 

---
