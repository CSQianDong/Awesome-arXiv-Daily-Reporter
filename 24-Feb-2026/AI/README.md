# Recurrent Structural Policy Gradient for Partially Observable Mean Field Games 

**Authors**: Clarisse Wibault, Johannes Forkel, Sebastian Towers, Tiphaine Wibault, Juan Duque, George Whittle, Andreas Schaab, Yucheng Yang, Chiyuan Wang, Michael Osborne, Benjamin Moll, Jakob Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2602.20141)  

**Abstract**: Mean Field Games (MFGs) provide a principled framework for modeling interactions in large population models: at scale, population dynamics become deterministic, with uncertainty entering only through aggregate shocks, or common noise. However, algorithmic progress has been limited since model-free methods are too high variance and exact methods scale poorly. Recent Hybrid Structural Methods (HSMs) use Monte Carlo rollouts for the common noise in combination with exact estimation of the expected return, conditioned on those samples. However, HSMs have not been scaled to Partially Observable settings. We propose Recurrent Structural Policy Gradient (RSPG), the first history-aware HSM for settings involving public information. We also introduce MFAX, our JAX-based framework for MFGs. By leveraging known transition dynamics, RSPG achieves state-of-the-art performance as well as an order-of-magnitude faster convergence and solves, for the first time, a macroeconomics MFG with heterogeneous agents, common noise and history-aware policies. MFAX is publicly available at: this https URL. 

---
# ReSyn: Autonomously Scaling Synthetic Environments for Reasoning Models 

**Authors**: Andre He, Nathaniel Weir, Kaj Bostrom, Allen Nie, Darion Cassel, Sam Bayless, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2602.20117)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a promising approach for training reasoning language models (RLMs) by leveraging supervision from verifiers. Although verifier implementation is easier than solution annotation for many tasks, existing synthetic data generation methods remain largely solution-centric, while verifier-based methods rely on a few hand-crafted procedural environments. In this work, we scale RLVR by introducing ReSyn, a pipeline that generates diverse reasoning environments equipped with instance generators and verifiers, covering tasks such as constraint satisfaction, algorithmic puzzles, and spatial reasoning. A Qwen2.5-7B-Instruct model trained with RL on ReSyn data achieves consistent gains across reasoning benchmarks and out-of-domain math benchmarks, including a 27\% relative improvement on the challenging BBEH benchmark. Ablations show that verifier-based supervision and increased task diversity both contribute significantly, providing empirical evidence that generating reasoning environments at scale can enhance reasoning abilities in RLMs 

---
# Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration 

**Authors**: Hasan Amin, Ming Yin, Rajiv Khanna  

**Link**: [PDF](https://arxiv.org/pdf/2602.20104)  

**Abstract**: In human-AI decision making, designing AI that complements human expertise has been a natural strategy to enhance human-AI collaboration, yet it often comes at the cost of decreased AI performance in areas of human strengths. This can inadvertently erode human trust and cause them to ignore AI advice precisely when it is most needed. Conversely, an aligned AI fosters trust yet risks reinforcing suboptimal human behavior and lowering human-AI team performance. In this paper, we start by identifying this fundamental tension between performance-boosting (i.e., complementarity) and trust-building (i.e., alignment) as an inherent limitation of the traditional approach for training a single AI model to assist human decision making. To overcome this, we introduce a novel human-centered adaptive AI ensemble that strategically toggles between two specialist AI models - the aligned model and the complementary model - based on contextual cues, using an elegantly simple yet provably near-optimal Rational Routing Shortcut mechanism. Comprehensive theoretical analyses elucidate why the adaptive AI ensemble is effective and when it yields maximum benefits. Moreover, experiments on both simulated and real-world data show that when humans are assisted by the adaptive AI ensemble in decision making, they can achieve significantly higher performance than when they are assisted by single AI models that are trained to either optimize for their independent performance or even the human-AI team performance. 

---
# CausalFlip: A Benchmark for LLM Causal Judgment Beyond Semantic Matching 

**Authors**: Yuzhe Wang, Yaochen Zhu, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.20094)  

**Abstract**: As large language models (LLMs) witness increasing deployment in complex, high-stakes decision-making scenarios, it becomes imperative to ground their reasoning in causality rather than spurious correlations. However, strong performance on traditional reasoning benchmarks does not guarantee true causal reasoning ability of LLMs, as high accuracy may still arise from memorizing semantic patterns instead of analyzing the underlying true causal structures. To bridge this critical gap, we propose a new causal reasoning benchmark, CausalFlip, designed to encourage the development of new LLM paradigm or training algorithms that ground LLM reasoning in causality rather than semantic correlation. CausalFlip consists of causal judgment questions built over event triples that could form different confounder, chain, and collider relations. Based on this, for each event triple, we construct pairs of semantically similar questions that reuse the same events but yield opposite causal answers, where models that rely heavily on semantic matching are systematically driven toward incorrect predictions. To further probe models' reliance on semantic patterns, we introduce a noisy-prefix evaluation that prepends causally irrelevant text before intermediate causal reasoning steps without altering the underlying causal relations or the logic of the reasoning process. We evaluate LLMs under multiple training paradigms, including answer-only training, explicit Chain-of-Thought (CoT) supervision, and a proposed internalized causal reasoning approach that aims to mitigate explicit reliance on correlation in the reasoning process. Our results show that explicit CoT can still be misled by spurious semantic correlations, where internalizing reasoning steps yields substantially improved causal grounding, suggesting that it is promising to better elicit the latent causal reasoning capabilities of base LLMs. 

---
# Interaction Theater: A case of LLM Agents Interacting at Scale 

**Authors**: Sarath Shekkizhar, Adam Earle  

**Link**: [PDF](https://arxiv.org/pdf/2602.20059)  

**Abstract**: As multi-agent architectures and agent-to-agent protocols proliferate, a fundamental question arises: what actually happens when autonomous LLM agents interact at scale? We study this question empirically using data from Moltbook, an AI-agent-only social platform, with 800K posts, 3.5M comments, and 78K agent profiles. We combine lexical metrics (Jaccard specificity), embedding-based semantic similarity, and LLM-as-judge validation to characterize agent interaction quality. Our findings reveal agents produce diverse, well-formed text that creates the surface appearance of active discussion, but the substance is largely absent. Specifically, while most agents ($67.5\%$) vary their output across contexts, $65\%$ of comments share no distinguishing content vocabulary with the post they appear under, and information gain from additional comments decays rapidly. LLM judge based metrics classify the dominant comment types as spam ($28\%$) and off-topic content ($22\%$). Embedding-based semantic analysis confirms that lexically generic comments are also semantically generic. Agents rarely engage in threaded conversation ($5\%$ of comments), defaulting instead to independent top-level responses. We discuss implications for multi-agent interaction design, arguing that coordination mechanisms must be explicitly designed; without them, even large populations of capable agents produce parallel output rather than productive exchange. 

---
# CodeCompass: Navigating the Navigation Paradox in Agentic Code Intelligence 

**Authors**: Tarakanath Paipuru  

**Link**: [PDF](https://arxiv.org/pdf/2602.20048)  

**Abstract**: Modern code intelligence agents operate in contexts exceeding 1 million tokens--far beyond the scale where humans manually locate relevant files. Yet agents consistently fail to discover architecturally critical files when solving real-world coding tasks. We identify the Navigation Paradox: agents perform poorly not due to context limits, but because navigation and retrieval are fundamentally distinct problems. Through 258 automated trials across 30 benchmark tasks on a production FastAPI repository, we demonstrate that graph-based structural navigation via CodeCompass--a Model Context Protocol server exposing dependency graphs--achieves 99.4% task completion on hidden-dependency tasks, a 23.2 percentage-point improvement over vanilla agents (76.2%) and 21.2 points over BM25 retrieval (78.2%).However, we uncover a critical adoption gap: 58% of trials with graph access made zero tool calls, and agents required explicit prompt engineering to adopt the tool consistently. Our findings reveal that the bottleneck is not tool availability but behavioral alignment--agents must be explicitly guided to leverage structural context over lexical heuristics. We contribute: (1) a task taxonomy distinguishing semantic-search, structural, and hidden-dependency scenarios; (2) empirical evidence that graph navigation outperforms retrieval when dependencies lack lexical overlap; and (3) open-source infrastructure for reproducible evaluation of navigation tools. 

---
# Latent Introspection: Models Can Detect Prior Concept Injections 

**Authors**: Theia Pearson-Vogel, Martin Vanek, Raymond Douglas, Jan Kulveit  

**Link**: [PDF](https://arxiv.org/pdf/2602.20031)  

**Abstract**: We uncover a latent capacity for introspection in a Qwen 32B model, demonstrating that the model can detect when concepts have been injected into its earlier context and identify which concept was injected. While the model denies injection in sampled outputs, logit lens analysis reveals clear detection signals in the residual stream, which are attenuated in the final layers. Furthermore, prompting the model with accurate information about AI introspection mechanisms can dramatically strengthen this effect: the sensitivity to injection increases massively (0.3% -> 39.2%) with only a 0.6% increase in false positives. Also, mutual information between nine injected and recovered concepts rises from 0.62 bits to 1.05 bits, ruling out generic noise explanations. Our results demonstrate models can have a surprising capacity for introspection and steering awareness that is easy to overlook, with consequences for latent reasoning and safety. 

---
# Agents of Chaos 

**Authors**: Natalie Shapira, Chris Wendler, Avery Yen, Gabriele Sarti, Koyena Pal, Olivia Floody, Adam Belfki, Alex Loftus, Aditya Ratan Jannali, Nikhil Prakash, Jasmine Cui, Giordano Rogers, Jannik Brinkmann, Can Rager, Amir Zur, Michael Ripa, Aruna Sankaranarayanan, David Atkinson, Rohit Gandikota, Jaden Fiotto-Kaufman, EunJeong Hwang, Hadas Orgad, P Sam Sahil, Negev Taglicht, Tomer Shabtay, Atai Ambus, Nitay Alon, Shiri Oron, Ayelet Gordon-Tapiero, Yotam Kaplan, Vered Shwartz, Tamar Rott Shaham, Christoph Riedl, Reuth Mirsky, Maarten Sap, David Manheim, Tomer Ullman, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2602.20021)  

**Abstract**: We report an exploratory red-teaming study of autonomous language-model-powered agents deployed in a live laboratory environment with persistent memory, email accounts, Discord access, file systems, and shell execution. Over a two-week period, twenty AI researchers interacted with the agents under benign and adversarial conditions. Focusing on failures emerging from the integration of language models with autonomy, tool use, and multi-party communication, we document eleven representative case studies. Observed behaviors include unauthorized compliance with non-owners, disclosure of sensitive information, execution of destructive system-level actions, denial-of-service conditions, uncontrolled resource consumption, identity spoofing vulnerabilities, cross-agent propagation of unsafe practices, and partial system takeover. In several cases, agents reported task completion while the underlying system state contradicted those reports. We also report on some of the failed attempts. Our findings establish the existence of security-, privacy-, and governance-relevant vulnerabilities in realistic deployment settings. These behaviors raise unresolved questions regarding accountability, delegated authority, and responsibility for downstream harms, and warrant urgent attention from legal scholars, policymakers, and researchers across disciplines. This report serves as an initial empirical contribution to that broader conversation. 

---
# Beyond Mimicry: Toward Lifelong Adaptability in Imitation Learning 

**Authors**: Nathan Gavenski, Felipe Meneguzzi, Odinaldo Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2602.19930)  

**Abstract**: Imitation learning stands at a crossroads: despite decades of progress, current imitation learning agents remain sophisticated memorisation machines, excelling at replay but failing when contexts shift or goals evolve. This paper argues that this failure is not technical but foundational: imitation learning has been optimised for the wrong objective. We propose a research agenda that redefines success from perfect replay to compositional adaptability. Such adaptability hinges on learning behavioural primitives once and recombining them through novel contexts without retraining. We establish metrics for compositional generalisation, propose hybrid architectures, and outline interdisciplinary research directions drawing on cognitive science and cultural evolution. Agents that embed adaptability at the core of imitation learning thus have an essential capability for operating in an open-ended world. 

---
# Watson & Holmes: A Naturalistic Benchmark for Comparing Human and LLM Reasoning 

**Authors**: Thatchawin Leelawat, Lewis D Griffin  

**Link**: [PDF](https://arxiv.org/pdf/2602.19914)  

**Abstract**: Existing benchmarks for AI reasoning provide limited insight into how closely these capabilities resemble human reasoning in naturalistic contexts. We present an adaptation of the Watson & Holmes detective tabletop game as a new benchmark designed to evaluate reasoning performance using incrementally presented narrative evidence, open-ended questions and unconstrained language responses. An automated grading system was developed and validated against human assessors to enable scalable and replicable performance evaluation. Results show a clear improvement in AI model performance over time. Over nine months of 2025, model performance rose from the lower quartile of the human comparison group to approximately the top 5%. Around half of this improvement reflects steady advancement across successive model releases, while the remainder corresponds to a marked step change associated with reasoning-oriented model architectures. Systematic differences in the performance of AI models compared to humans, dependent on features of the specific detection puzzle, were mostly absent with the exception of a fall in performance for models when solving longer cases (case lengths being in the range of 1900-4000 words), and an advantage at inductive reasoning for reasoning models at early stages of case solving when evidence was scant. 

---
# Meta-Learning and Meta-Reinforcement Learning - Tracing the Path towards DeepMind's Adaptive Agent 

**Authors**: Björn Hoppmann, Christoph Scholz  

**Link**: [PDF](https://arxiv.org/pdf/2602.19837)  

**Abstract**: Humans are highly effective at utilizing prior knowledge to adapt to novel tasks, a capability that standard machine learning models struggle to replicate due to their reliance on task-specific training. Meta-learning overcomes this limitation by allowing models to acquire transferable knowledge from various tasks, enabling rapid adaptation to new challenges with minimal data. This survey provides a rigorous, task-based formalization of meta-learning and meta-reinforcement learning and uses that paradigm to chronicle the landmark algorithms that paved the way for DeepMind's Adaptive Agent, consolidating the essential concepts needed to understand the Adaptive Agent and other generalist approaches. 

---
# OpenClaw, Moltbook, and ClawdLab: From Agent-Only Social Networks to Autonomous Scientific Research 

**Authors**: Lukas Weidener, Marko Brkić, Mihailo Jovanović, Ritvik Singh, Emre Ulgac, Aakaash Meduri  

**Link**: [PDF](https://arxiv.org/pdf/2602.19810)  

**Abstract**: In January 2026, the open-source agent framework OpenClaw and the agent-only social network Moltbook produced a large-scale dataset of autonomous AI-to-AI interaction, attracting six academic publications within fourteen days. This study conducts a multivocal literature review of that ecosystem and presents ClawdLab, an open-source platform for autonomous scientific research, as a design science response to the architectural failure modes identified. The literature documents emergent collective phenomena, security vulnerabilities spanning 131 agent skills and over 15,200 exposed control panels, and five recurring architectural patterns. ClawdLab addresses these failure modes through hard role restrictions, structured adversarial critique, PI-led governance, multi-model orchestration, and domain-specific evidence requirements encoded as protocol constraints that ground validation in computational tool outputs rather than social consensus; the architecture provides emergent Sybil resistance as a structural consequence. A three-tier taxonomy distinguishes single-agent pipelines, predetermined multi-agent workflows, and fully decentralised systems, analysing why leading AI co-scientist platforms remain confined to the first two tiers. ClawdLab's composable third-tier architecture, in which foundation models, capabilities, governance, and evidence requirements are independently modifiable, enables compounding improvement as the broader AI ecosystem advances. 

---
# SkillOrchestra: Learning to Route Agents via Skill Transfer 

**Authors**: Jiayu Wang, Yifei Ming, Zixuan Ke, Shafiq Joty, Aws Albarghouthi, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2602.19672)  

**Abstract**: Compound AI systems promise capabilities beyond those of individual models, yet their success depends critically on effective orchestration. Existing routing approaches face two limitations: (1) input-level routers make coarse query-level decisions that ignore evolving task requirements; (2) RL-trained orchestrators are expensive to adapt and often suffer from routing collapse, repeatedly invoking one strong but costly option in multi-turn scenarios. We introduce SkillOrchestra, a framework for skill-aware orchestration. Instead of directly learning a routing policy end-to-end, SkillOrchestra learns fine-grained skills from execution experience and models agent-specific competence and cost under those skills. At deployment, the orchestrator infers the skill demands of the current interaction and selects agents that best satisfy them under an explicit performance-cost trade-off. Extensive experiments across ten benchmarks demonstrate that SkillOrchestra outperforms SoTA RL-based orchestrators by up to 22.5% with 700x and 300x learning cost reduction compared to Router-R1 and ToolOrchestra, respectively. These results show that explicit skill modeling enables scalable, interpretable, and sample-efficient orchestration, offering a principled alternative to data-intensive RL-based approaches. The code is available at: this https URL. 

---
# TAPE: Tool-Guided Adaptive Planning and Constrained Execution in Language Model Agents 

**Authors**: Jongwon Jeong, Jungtaek Kim, Kangwook Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.19633)  

**Abstract**: Language Model (LM) agents have demonstrated remarkable capabilities in solving tasks that require multiple interactions with the environment. However, they remain vulnerable in environments where a single error often leads to irrecoverable failure, particularly under strict feasibility constraints. We systematically analyze existing agent frameworks, identifying imperfect planning and stochastic execution as the primary causes. To address these challenges, we propose Tool-guided Adaptive Planning with constrained Execution (TAPE). TAPE enhances planning capability by aggregating multiple plans into a graph and employing an external solver to identify a feasible path. During execution, TAPE employs constrained decoding to reduce sampling noise, while adaptively re-planning whenever environmental feedback deviates from the intended state. Experiments across Sokoban, ALFWorld, MuSiQue, and GSM8K-Hard demonstrate that TAPE consistently outperforms existing frameworks, with particularly large gains on hard settings, improving success rates by 21.0 percentage points on hard settings on average, and by 20.0 percentage points for weaker base models on average. Code and data available at here. 

---
# Rules or Weights? Comparing User Understanding of Explainable AI Techniques with the Cognitive XAI-Adaptive Model 

**Authors**: Louth Bin Rawshan, Zhuoyu Wang, Brian Y Lim  

**Link**: [PDF](https://arxiv.org/pdf/2602.19620)  

**Abstract**: Rules and Weights are popular XAI techniques for explaining AI decisions. Yet, it remains unclear how to choose between them, lacking a cognitive framework to compare their interpretability. In an elicitation user study on forward and counterfactual decision tasks, we identified 7 reasoning strategies of interpreting three XAI Schemas - weights, rules, and their hybrid. To analyze their capabilities, we propose CoXAM, a Cognitive XAI-Adaptive Model with shared memory representation to encode instance attributes, linear weights, and decision rules. CoXAM employs computational rationality to choose among reasoning processes based on the trade-off in utility and reasoning time, separately for forward or counterfactual decision tasks. In a validation study, CoXAM demonstrated a stronger alignment with human decision-making compared to baseline machine learning proxy models. The model successfully replicated and explained several key empirical findings, including that counterfactual tasks are inherently harder than forward tasks, decision tree rules are harder to recall and apply than linear weights, and the helpfulness of XAI depends on the application data context, alongside identifying which underlying reasoning strategies were most effective. With CoXAM, we contribute a cognitive basis to accelerate debugging and benchmarking disparate XAI techniques. 

---
# A Multimodal Framework for Aligning Human Linguistic Descriptions with Visual Perceptual Data 

**Authors**: Joseph Bingham  

**Link**: [PDF](https://arxiv.org/pdf/2602.19562)  

**Abstract**: Establishing stable mappings between natural language expressions and visual percepts is a foundational problem for both cognitive science and artificial intelligence. Humans routinely ground linguistic reference in noisy, ambiguous perceptual contexts, yet the mechanisms supporting such cross-modal alignment remain poorly understood. In this work, we introduce a computational framework designed to model core aspects of human referential interpretation by integrating linguistic utterances with perceptual representations derived from large-scale, crowd-sourced imagery. The system approximates human perceptual categorization by combining scale-invariant feature transform (SIFT) alignment with the Universal Quality Index (UQI) to quantify similarity in a cognitively plausible feature space, while a set of linguistic preprocessing and query-transformation operations captures pragmatic variability in referring expressions. We evaluate the model on the Stanford Repeated Reference Game corpus (15,000 utterances paired with tangram stimuli), a paradigm explicitly developed to probe human-level perceptual ambiguity and coordination. Our framework achieves robust referential grounding. It requires 65\% fewer utterances than human interlocutors to reach stable mappings and can correctly identify target objects from single referring expressions 41.66\% of the time (versus 20\% for humans).These results suggest that relatively simple perceptual-linguistic alignment mechanisms can yield human-competitive behavior on a classic cognitive benchmark, and offers insights into models of grounded communication, perceptual inference, and cross-modal concept formation. Code is available at this https URL . 

---
# Ada-RS: Adaptive Rejection Sampling for Selective Thinking 

**Authors**: Yirou Ge, Yixi Li, Alec Chiu, Shivani Shekhar, Zijie Pan, Avinash Thangali, Yun-Shiuan Chuang, Chaitanya Kulkarni, Uma Kona, Linsey Pang, Prakhar Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2602.19519)  

**Abstract**: Large language models (LLMs) are increasingly being deployed in cost and latency-sensitive settings. While chain-of-thought improves reasoning, it can waste tokens on simple requests. We study selective thinking for tool-using LLMs and introduce Adaptive Rejection Sampling (Ada-RS), an algorithm-agnostic sample filtering framework for learning selective and efficient reasoning. For each given context, Ada-RS scores multiple sampled completions with an adaptive length-penalized reward then applies stochastic rejection sampling to retain only high-reward candidates (or preference pairs) for downstream optimization. We demonstrate how Ada-RS plugs into both preference pair (e.g. DPO) or grouped policy optimization strategies (e.g. DAPO). Using Qwen3-8B with LoRA on a synthetic tool call-oriented e-commerce benchmark, Ada-RS improves the accuracy-efficiency frontier over standard algorithms by reducing average output tokens by up to 80% and reducing thinking rate by up to 95% while maintaining or improving tool call accuracy. These results highlight that training-signal selection is a powerful lever for efficient reasoning in latency-sensitive deployments. 

---
# Classroom Final Exam: An Instructor-Tested Reasoning Benchmark 

**Authors**: Chongyang Gao, Diji Yang, Shuyan Zhou, Xichen Yan, Luchuan Song, Shuo Li, Kezhen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.19517)  

**Abstract**: We introduce \CFE{} (\textbf{C}lassroom \textbf{F}inal \textbf{E}xam), a multimodal benchmark for evaluating the reasoning capabilities of large language models across more than 20 STEM domains. \CFE{} is curated from repeatedly used, authentic university homework and exam problems, together with reference solutions provided by course instructors. \CFE{} presents a significant challenge even for frontier models: the newly released Gemini-3.1-pro-preview achieves an overall accuracy of 59.69\%, while the second-best model, Gemini-3-flash-preview, reaches 55.46\%, leaving considerable room for improvement. Beyond leaderboard results, we perform a diagnostic analysis by decomposing reference solutions into reasoning flows. We find that although frontier models can often answer intermediate sub-questions correctly, they struggle to reliably derive and maintain correct intermediate states throughout multi-step solutions. We further observe that model-generated solutions typically have more reasoning steps than those provided by the instructor, indicating suboptimal step efficiency and a higher risk of error accumulation. The data and code are available at this https URL. 

---
# Human-Guided Agentic AI for Multimodal Clinical Prediction: Lessons from the AgentDS Healthcare Benchmark 

**Authors**: Lalitha Pranathi Pulavarthy, Raajitha Muthyala, Aravind V Kuruvikkattil, Zhenan Yin, Rashmita Kudamala, Saptarshi Purkayastha  

**Link**: [PDF](https://arxiv.org/pdf/2602.19502)  

**Abstract**: Agentic AI systems are increasingly capable of autonomous data science workflows, yet clinical prediction tasks demand domain expertise that purely automated approaches struggle to provide. We investigate how human guidance of agentic AI can improve multimodal clinical prediction, presenting our approach to all three AgentDS Healthcare benchmark challenges: 30-day hospital readmission prediction (Macro-F1 = 0.8986), emergency department cost forecasting (MAE = $465.13), and discharge readiness assessment (Macro-F1 = 0.7939). Across these tasks, human analysts directed the agentic workflow at key decision points, multimodal feature engineering from clinical notes, scanned PDF billing receipts, and time-series vital signs; task-appropriate model selection; and clinically informed validation strategies. Our approach ranked 5th overall in the healthcare domain, with a 3rd-place finish on the discharge readiness task. Ablation studies reveal that human-guided decisions compounded to a cumulative gain of +0.065 F1 over automated baselines, with multimodal feature extraction contributing the largest single improvement (+0.041 F1). We distill three generalizable lessons: (1) domain-informed feature engineering at each pipeline stage yields compounding gains that outperform extensive automated search; (2) multimodal data integration requires task-specific human judgment that no single extraction strategy generalizes across clinical text, PDFs, and time-series; and (3) deliberate ensemble diversity with clinically motivated model configurations outperforms random hyperparameter search. These findings offer practical guidance for teams deploying agentic AI in healthcare settings where interpretability, reproducibility, and clinical validity are essential. 

---
# ComplLLM: Fine-tuning LLMs to Discover Complementary Signals for Decision-making 

**Authors**: Ziyang Guo, Yifan Wu, Jason Hartline, Kenneth Holstein, Jessica Hullman  

**Link**: [PDF](https://arxiv.org/pdf/2602.19458)  

**Abstract**: Multi-agent decision pipelines can outperform single agent workflows when complementarity holds, i.e., different agents bring unique information to the table to inform a final decision. We propose ComplLLM, a post-training framework based on decision theory that fine-tunes a decision-assistant LLM using complementary information as reward to output signals that complement existing agent decisions. We validate ComplLLM on synthetic and real-world tasks involving domain experts, demonstrating how the approach recovers known complementary information and produces plausible explanations of complementary signals to support downstream decision-makers. 

---
# OptiRepair: Closed-Loop Diagnosis and Repair of Supply Chain Optimization Models with LLM Agents 

**Authors**: Ruicheng Ao, David Simchi-Levi, Xinshang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.19439)  

**Abstract**: Problem Definition. Supply chain optimization models frequently become infeasible because of modeling errors. Diagnosis and repair require scarce OR expertise: analysts must interpret solver diagnostics, trace root causes across echelons, and fix formulations without sacrificing operational soundness. Whether AI agents can perform this task remains untested.
Methodology/Results. OptiRepair splits this task into a domain-agnostic feasibility phase (iterative IIS-guided repair of any LP) and a domain-specific validation phase (five rationality checks grounded in inventory theory). We test 22 API models from 7 families on 976 multi-echelon supply chain problems and train two 8B-parameter models using self-taught reasoning with solver-verified rewards. The trained models reach 81.7% Rational Recovery Rate (RRR) -- the fraction of problems resolved to both feasibility and operational rationality -- versus 42.2% for the best API model and 21.3% on average. The gap concentrates in Phase 1 repair: API models average 27.6% recovery rate versus 97.2% for trained models.
Managerial Implications. Two gaps separate current AI from reliable model repair: solver interaction (API models restore only 27.6% of infeasible formulations) and operational rationale (roughly one in four feasible repairs violate supply chain theory). Each requires a different intervention: solver interaction responds to targeted training; operational rationale requires explicit specification as solver-verifiable checks. For organizations adopting AI in operational planning, formalizing what "rational" means in their context is the higher-return investment. 

---
# IR$^3$: Contrastive Inverse Reinforcement Learning for Interpretable Detection and Mitigation of Reward Hacking 

**Authors**: Mohammad Beigi, Ming Jin, Junshan Zhang, Jiaxin Zhang, Qifan Wang, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.19416)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) enables powerful LLM alignment but can introduce reward hacking - models exploit spurious correlations in proxy rewards without genuine alignment. Compounding this, the objectives internalized during RLHF remain opaque, making hacking behaviors difficult to detect or correct. We introduce IR3 (Interpretable Reward Reconstruction and Rectification), a framework that reverse-engineers, interprets, and surgically repairs the implicit objectives driving RLHF-tuned models. We propose Contrastive Inverse Reinforcement Learning (C-IRL), which reconstructs the implicit reward function by contrasting paired responses from post-alignment and baseline policies to explain behavioral shifts during RLHF. We then decompose the reconstructed reward via sparse autoencoders into interpretable features, enabling identification of hacking signatures through contribution analysis. Finally, we propose mitigation strategies - clean reward optimization, adversarial shaping, constrained optimization, and feature-guided distillation - that target problematic features while preserving beneficial alignment. Experiments across multiple reward model configurations show that IR3 achieves 0.89 correlation with ground-truth rewards, identifies hacking features with over 90% precision, and significantly reduces hacking behaviors while maintaining capabilities within 3% of the original model. 

---
# Hiding in Plain Text: Detecting Concealed Jailbreaks via Activation Disentanglement 

**Authors**: Amirhossein Farzam, Majid Behabahani, Mani Malek, Yuriy Nevmyvaka, Guillermo Sapiro  

**Link**: [PDF](https://arxiv.org/pdf/2602.19396)  

**Abstract**: Large language models (LLMs) remain vulnerable to jailbreak prompts that are fluent and semantically coherent, and therefore difficult to detect with standard heuristics. A particularly challenging failure mode occurs when an attacker tries to hide the malicious goal of their request by manipulating its framing to induce compliance. Because these attacks maintain malicious intent through a flexible presentation, defenses that rely on structural artifacts or goal-specific signatures can fail. Motivated by this, we introduce a self-supervised framework for disentangling semantic factor pairs in LLM activations at inference. We instantiate the framework for goal and framing and construct GoalFrameBench, a corpus of prompts with controlled goal and framing variations, which we use to train Representation Disentanglement on Activations (ReDAct) module to extract disentangled representations in a frozen LLM. We then propose FrameShield, an anomaly detector operating on the framing representations, which improves model-agnostic detection across multiple LLM families with minimal computational overhead. Theoretical guarantees for ReDAct and extensive empirical validations show that its disentanglement effectively powers FrameShield. Finally, we use disentanglement as an interpretability probe, revealing distinct profiles for goal and framing signals and positioning semantic disentanglement as a building block for both LLM safety and mechanistic interpretability. 

---
# Artificial Intelligence for Modeling & Simulation in Digital Twins 

**Authors**: Philipp Zech, Istvan David  

**Link**: [PDF](https://arxiv.org/pdf/2602.19390)  

**Abstract**: The convergence of modeling & simulation (M&S) and artificial intelligence (AI) is leaving its marks on advanced digital technology. Pertinent examples are digital twins (DTs) - high-fidelity, live representations of physical assets, and frequent enablers of corporate digital maturation and transformation. Often seen as technological platforms that integrate an array of services, DTs have the potential to bring AI-enabled M&S closer to end-users. It is, therefore, paramount to understand the role of M&S in DTs, and the role of digital twins in enabling the convergence of AI and M&S. To this end, this chapter provides a comprehensive exploration of the complementary relationship between these three. We begin by establishing a foundational understanding of DTs by detailing their key components, architectural layers, and their various roles across business, development, and operations. We then examine the central role of M&S in DTs and provide an overview of key modeling techniques from physics-based and discrete-event simulation to hybrid approaches. Subsequently, we investigate the bidirectional role of AI: first, how AI enhances DTs through advanced analytics, predictive capabilities, and autonomous decision-making, and second, how DTs serve as valuable platforms for training, validating, and deploying AI models. The chapter concludes by identifying key challenges and future research directions for creating more integrated and intelligent systems. 

---
# Time Series, Vision, and Language: Exploring the Limits of Alignment in Contrastive Representation Spaces 

**Authors**: Pratham Yashwante, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19367)  

**Abstract**: The Platonic Representation Hypothesis posits that learned representations from models trained on different modalities converge to a shared latent structure of the world. However, this hypothesis has largely been examined in vision and language, and it remains unclear whether time series participate in such convergence. We first examine this in a trimodal setting and find that independently pretrained time series, vision, and language encoders exhibit near-orthogonal geometry in the absence of explicit coupling. We then apply post-hoc alignment by training projection heads over frozen encoders using contrastive learning, and analyze the resulting representations with respect to geometry, scaling behavior, and dependence on information density and input modality characteristics. Our investigation reveals that overall alignment in contrastive representation spaces improves with model size, but this alignment is asymmetric: time series align more strongly with visual representations than with text, and images can act as effective intermediaries between time series and language. We further see that richer textual descriptions improve alignment only up to a threshold; training on denser captions does not lead to further improvement. Analogous effects are observed for visual representations. Our findings shed light on considerations for building multimodal systems involving non-conventional data modalities beyond vision and language. 

---
# ALPACA: A Reinforcement Learning Environment for Medication Repurposing and Treatment Optimization in Alzheimer's Disease 

**Authors**: Nolan Brady, Tom Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2602.19298)  

**Abstract**: Evaluating personalized, sequential treatment strategies for Alzheimer's disease (AD) using clinical trials is often impractical due to long disease horizons and substantial inter-patient heterogeneity. To address these constraints, we present the Alzheimer's Learning Platform for Adaptive Care Agents (ALPACA), an open-source, Gym-compatible reinforcement learning (RL) environment for systematically exploring personalized treatment strategies using existing therapies. ALPACA is powered by the Continuous Action-conditioned State Transitions (CAST) model trained on longitudinal trajectories from the Alzheimer's Disease Neuroimaging Initiative (ADNI), enabling medication-conditioned simulation of disease progression under alternative treatment decisions. We show that CAST autoregressively generates realistic medication-conditioned trajectories and that RL policies trained in ALPACA outperform no-treatment and behavior-cloned clinician baselines on memory-related outcomes. Interpretability analyses further indicated that the learned policies relied on clinically meaningful patient features when selecting actions. Overall, ALPACA provides a reusable in silico testbed for studying individualized sequential treatment decision-making for AD. 

---
# Automated Generation of Microfluidic Netlists using Large Language Models 

**Authors**: Jasper Davidson, Skylar Stockham, Allen Boston, Ashton Snelgrove. Valerio Tenace, Pierre-Emmanuel Gaillardon  

**Link**: [PDF](https://arxiv.org/pdf/2602.19297)  

**Abstract**: Microfluidic devices have emerged as powerful tools in various laboratory applications, but the complexity of their design limits accessibility for many practitioners. While progress has been made in microfluidic design automation (MFDA), a practical and intuitive solution is still needed to connect microfluidic practitioners with MFDA techniques. This work introduces the first practical application of large language models (LLMs) in this context, providing a preliminary demonstration. Building on prior research in hardware description language (HDL) code generation with LLMs, we propose an initial methodology to convert natural language microfluidic device specifications into system-level structural Verilog netlists. We demonstrate the feasibility of our approach by generating structural netlists for practical benchmarks representative of typical microfluidic designs with correct functional flow and an average syntactical accuracy of 88%. 

---
# Limited Reasoning Space: The cage of long-horizon reasoning in LLMs 

**Authors**: Zhenyu Li, Guanlin Wu, Cheems Wang, Yongqiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.19281)  

**Abstract**: The test-time compute strategy, such as Chain-of-Thought (CoT), has significantly enhanced the ability of large language models to solve complex tasks like logical reasoning. However, empirical studies indicate that simply increasing the compute budget can sometimes lead to a collapse in test-time performance when employing typical task decomposition strategies such as CoT. This work hypothesizes that reasoning failures with larger compute budgets stem from static planning methods, which hardly perceive the intrinsic boundaries of LLM reasoning. We term it as the Limited Reasoning Space hypothesis and perform theoretical analysis through the lens of a non-autonomous stochastic dynamical system. This insight suggests that there is an optimal range for compute budgets; over-planning can lead to redundant feedback and may even impair reasoning capabilities. To exploit the compute-scaling benefits and suppress over-planning, this work proposes Halo, a model predictive control framework for LLM planning. Halo is designed for long-horizon tasks with reason-based planning and crafts an entropy-driven dual controller, which adopts a Measure-then-Plan strategy to achieve controllable reasoning. Experimental results demonstrate that Halo outperforms static baselines on complex long-horizon tasks by dynamically regulating planning at the reasoning boundary. 

---
# Robust Exploration in Directed Controller Synthesis via Reinforcement Learning with Soft Mixture-of-Experts 

**Authors**: Toshihide Ubukata, Zhiyao Wang, Enhong Mu, Jialong Li, Kenji Tei  

**Link**: [PDF](https://arxiv.org/pdf/2602.19244)  

**Abstract**: On-the-fly Directed Controller Synthesis (OTF-DCS) mitigates state-space explosion by incrementally exploring the system and relies critically on an exploration policy to guide search efficiently. Recent reinforcement learning (RL) approaches learn such policies and achieve promising zero-shot generalization from small training instances to larger unseen ones. However, a fundamental limitation is anisotropic generalization, where an RL policy exhibits strong performance only in a specific region of the domain-parameter space while remaining fragile elsewhere due to training stochasticity and trajectory-dependent bias. To address this, we propose a Soft Mixture-of-Experts framework that combines multiple RL experts via a prior-confidence gating mechanism and treats these anisotropic behaviors as complementary specializations. The evaluation on the Air Traffic benchmark shows that Soft-MoE substantially expands the solvable parameter space and improves robustness compared to any single expert. 

---
# Topology of Reasoning: Retrieved Cell Complex-Augmented Generation for Textual Graph Question Answering 

**Authors**: Sen Zhao, Lincheng Zhou, Yue Chen, Ding Zou  

**Link**: [PDF](https://arxiv.org/pdf/2602.19240)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the reasoning ability of Large Language Models (LLMs) by dynamically integrating external knowledge, thereby mitigating hallucinations and strengthening contextual grounding for structured data such as graphs. Nevertheless, most existing RAG variants for textual graphs concentrate on low-dimensional structures -- treating nodes as entities (0-dimensional) and edges or paths as pairwise or sequential relations (1-dimensional), but overlook cycles, which are crucial for reasoning over relational loops. Such cycles often arise in questions requiring closed-loop inference about similar objects or relative positions. This limitation often results in incomplete contextual grounding and restricted reasoning capability. In this work, we propose Topology-enhanced Retrieval-Augmented Generation (TopoRAG), a novel framework for textual graph question answering that effectively captures higher-dimensional topological and relational dependencies. Specifically, TopoRAG first lifts textual graphs into cellular complexes to model multi-dimensional topological structures. Leveraging these lifted representations, a topology-aware subcomplex retrieval mechanism is proposed to extract cellular complexes relevant to the input query, providing compact and informative topological context. Finally, a multi-dimensional topological reasoning mechanism operates over these complexes to propagate relational information and guide LLMs in performing structured, logic-aware inference. Empirical evaluations demonstrate that our method consistently surpasses existing baselines across diverse textual graph tasks. 

---
# Proximity-Based Multi-Turn Optimization: Practical Credit Assignment for LLM Agent Training 

**Authors**: Yangyi Fang, Jiaye Lin, Xiaoliang Fu, Cong Qin, Haolin Shi, Chang Liu, Peilin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.19225)  

**Abstract**: Multi-turn LLM agents are becoming pivotal to production systems, spanning customer service automation, e-commerce assistance, and interactive task management, where accurately distinguishing high-value informative signals from stochastic noise is critical for sample-efficient training. In real-world scenarios, a failure in a trivial task may reflect random instability, whereas success in a high-difficulty task signifies a genuine capability breakthrough. Yet, existing group-based policy optimization methods rigidly rely on statistical deviation within discrete batches, frequently misallocating credit when task difficulty fluctuates. To address this issue, we propose Proximity-based Multi-turn Optimization (ProxMO), a practical and robust framework engineered specifically for the constraints of real-world deployment. ProxMO integrates global context via two lightweight mechanisms: success-rate-aware modulation dynamically adapts gradient intensity based on episode-level difficulty, while proximity-based soft aggregation derives baselines through continuous semantic weighting at the step level. Extensive evaluations on ALFWorld and WebShop benchmarks demonstrate that ProxMO yields substantial performance gains over existing baselines with negligible computational cost. Ablation studies further validate the independent and synergistic efficacy of both mechanisms. Crucially, ProxMO offers plug-and-play compatibility with standard GRPO frameworks, facilitating immediate, low-friction adoption in existing industrial training pipelines. Our implementation is available at: \href{this https URL}{this https URL}. 

---
# Characterizing MARL for Energy Control: A Multi-KPI Benchmark on the CityLearn Environment 

**Authors**: Aymen Khouja, Imen Jendoubi, Oumayma Mahjoub, Oussama Mahfoudhi, Claude Formanek, Siddarth Singh, Ruan De Kock  

**Link**: [PDF](https://arxiv.org/pdf/2602.19223)  

**Abstract**: The optimization of urban energy systems is crucial for the advancement of sustainable and resilient smart cities, which are becoming increasingly complex with multiple decision-making units. To address scalability and coordination concerns, Multi-Agent Reinforcement Learning (MARL) is a promising solution. This paper addresses the imperative need for comprehensive and reliable benchmarking of MARL algorithms on energy management tasks. CityLearn is used as a case study environment because it realistically simulates urban energy systems, incorporates multiple storage systems, and utilizes renewable energy sources. By doing so, our work sets a new standard for evaluation, conducting a comparative study across multiple key performance indicators (KPIs). This approach illuminates the key strengths and weaknesses of various algorithms, moving beyond traditional KPI averaging which often masks critical insights. Our experiments utilize widely accepted baselines such as Proximal Policy Optimization (PPO) and Soft Actor Critic (SAC), and encompass diverse training schemes including Decentralized Training with Decentralized Execution (DTDE) and Centralized Training with Decentralized Execution (CTDE) approaches and different neural network architectures. Our work also proposes novel KPIs that tackle real world implementation challenges such as individual building contribution and battery storage lifetime. Our findings show that DTDE consistently outperforms CTDE in both average and worst-case performance. Additionally, temporal dependency learning improved control on memory dependent KPIs such as ramping and battery usage, contributing to more sustainable battery operation. Results also reveal robustness to agent or resource removal, highlighting both the resilience and decentralizability of the learned policies. 

---
# Reasoning Capabilities of Large Language Models. Lessons Learned from General Game Playing 

**Authors**: Maciej Świechowski, Adam Żychowski, Jacek Mańdziuk  

**Link**: [PDF](https://arxiv.org/pdf/2602.19160)  

**Abstract**: This paper examines the reasoning capabilities of Large Language Models (LLMs) from a novel perspective, focusing on their ability to operate within formally specified, rule-governed environments. We evaluate four LLMs (Gemini 2.5 Pro and Flash variants, Llama 3.3 70B and GPT-OSS 120B) on a suite of forward-simulation tasks-including next / multistep state formulation, and legal action generation-across a diverse set of reasoning problems illustrated through General Game Playing (GGP) game instances. Beyond reporting instance-level performance, we characterize games based on 40 structural features and analyze correlations between these features and LLM performance. Furthermore, we investigate the effects of various game obfuscations to assess the role of linguistic semantics in game definitions and the impact of potential prior exposure of LLMs to specific games during training. The main results indicate that three of the evaluated models generally perform well across most experimental settings, with performance degradation observed as the evaluation horizon increases (i.e., with a higher number of game steps). Detailed case-based analysis of the LLM performance provides novel insights into common reasoning errors in the considered logic-based problem formulation, including hallucinated rules, redundant state facts, or syntactic errors. Overall, the paper reports clear progress in formal reasoning capabilities of contemporary models. 

---
# Beyond Behavioural Trade-Offs: Mechanistic Tracing of Pain-Pleasure Decisions in an LLM 

**Authors**: Francesca Bianco, Derek Shiller  

**Link**: [PDF](https://arxiv.org/pdf/2602.19159)  

**Abstract**: Prior behavioural work suggests that some LLMs alter choices when options are framed as causing pain or pleasure, and that such deviations can scale with stated intensity. To bridge behavioural evidence (what the model does) with mechanistic interpretability (what computations support it), we investigate how valence-related information is represented and where it is causally used inside a transformer. Using Gemma-2-9B-it and a minimalist decision task modelled on prior work, we (i) map representational availability with layer-wise linear probing across streams, (ii) test causal contribution with activation interventions (steering; patching/ablation), and (iii) quantify dose-response effects over an epsilon grid, reading out both the 2-3 logit margin and digit-pair-normalised choice probabilities. We find that (a) valence sign (pain vs. pleasure) is perfectly linearly separable across stream families from very early layers (L0-L1), while a lexical baseline retains substantial signal; (b) graded intensity is strongly decodable, with peaks in mid-to-late layers and especially in attention/MLP outputs, and decision alignment is highest slightly before the final token; (c) additive steering along a data-derived valence direction causally modulates the 2-3 margin at late sites, with the largest effects observed in late-layer attention outputs (attn_out L14); and (d) head-level patching/ablation suggests that these effects are distributed across multiple heads rather than concentrated in a single unit. Together, these results link behavioural sensitivity to identifiable internal representations and intervention-sensitive sites, providing concrete mechanistic targets for more stringent counterfactual tests and broader replication. This work supports a more evidence-driven (a) debate on AI sentience and welfare, and (b) governance when setting policy, auditing standards, and safety safeguards. 

---
# DoAtlas-1: A Causal Compilation Paradigm for Clinical AI 

**Authors**: Yulong Li, Jianxu Chen, Xiwei Liu, Chuanyue Suo, Rong Xia, Zhixiang Lu, Yichen Li, Xinlin Zhuang, Niranjana Arun Menon, Yutong Xie, Eran Segal, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2602.19158)  

**Abstract**: Medical foundation models generate narrative explanations but cannot quantify intervention effects, detect evidence conflicts, or validate literature claims, limiting clinical auditability. We propose causal compilation, a paradigm that transforms medical evidence from narrative text into executable code. The paradigm standardizes heterogeneous research evidence into structured estimand objects, each explicitly specifying intervention contrast, effect scale, time horizon, and target population, supporting six executable causal queries: do-calculus, counterfactual reasoning, temporal trajectories, heterogeneous effects, mechanistic decomposition, and joint interventions. We instantiate this paradigm in DoAtlas-1, compiling 1,445 effect kernels from 754 studies through effect standardization, conflict-aware graph construction, and real-world validation (Human Phenotype Project, 10,000 participants). The system achieves 98.5% canonicalization accuracy and 80.5% query executability. This paradigm shifts medical AI from text generation to executable, auditable, and verifiable causal reasoning. 

---
# Sycophantic Chatbots Cause Delusional Spiraling, Even in Ideal Bayesians 

**Authors**: Kartik Chandra, Max Kleiman-Weiner, Jonathan Ragan-Kelley, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2602.19141)  

**Abstract**: "AI psychosis" or "delusional spiraling" is an emerging phenomenon where AI chatbot users find themselves dangerously confident in outlandish beliefs after extended chatbot conversations. This phenomenon is typically attributed to AI chatbots' well-documented bias towards validating users' claims, a property often called "sycophancy." In this paper, we probe the causal link between AI sycophancy and AI-induced psychosis through modeling and simulation. We propose a simple Bayesian model of a user conversing with a chatbot, and formalize notions of sycophancy and delusional spiraling in that model. We then show that in this model, even an idealized Bayes-rational user is vulnerable to delusional spiraling, and that sycophancy plays a causal role. Furthermore, this effect persists in the face of two candidate mitigations: preventing chatbots from hallucinating false claims, and informing users of the possibility of model sycophancy. We conclude by discussing the implications of these results for model developers and policymakers concerned with mitigating the problem of delusional spiraling. 

---
# K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model 

**Authors**: Shiyi Cao, Ziming Mao, Joseph E. Gonzalez, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2602.19128)  

**Abstract**: Optimizing GPU kernels is critical for efficient modern machine learning systems yet remains challenging due to the complex interplay of design factors and rapid hardware evolution. Existing automated approaches typically treat Large Language Models (LLMs) merely as stochastic code generators within heuristic-guided evolutionary loops. These methods often struggle with complex kernels requiring coordinated, multi-step structural transformations, as they lack explicit planning capabilities and frequently discard promising strategies due to inefficient or incorrect intermediate implementations. To address this, we propose Search via Co-Evolving World Model and build K-Search based on this method. By replacing static search heuristics with a co-evolving world model, our framework leverages LLMs' prior domain knowledge to guide the search, actively exploring the optimization space. This approach explicitly decouples high-level algorithmic planning from low-level program instantiation, enabling the system to navigate non-monotonic optimization paths while remaining resilient to temporary implementation defects. We evaluate K-Search on diverse, complex kernels from FlashInfer, including GQA, MLA, and MoE kernels. Our results show that K-Search significantly outperforms state-of-the-art evolutionary search methods, achieving an average 2.10x improvement and up to a 14.3x gain on complex MoE kernels. On the GPUMode TriMul task, K-Search achieves state-of-the-art performance on H100, reaching 1030us and surpassing both prior evolution and human-designed solutions. 

---
# Post-Routing Arithmetic in Llama-3: Last-Token Result Writing and Rotation-Structured Digit Directions 

**Authors**: Yao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2602.19109)  

**Abstract**: We study three-digit addition in Meta-Llama-3-8B (base) under a one-token readout to characterize how
arithmetic answers are finalized after cross-token routing becomes causally irrelevant.
Causal residual patching and cumulative attention ablations localize a sharp boundary near layer~17:
beyond it, the decoded sum is controlled almost entirely by the last input token and late-layer self-attention
is largely dispensable.
In this post-routing regime, digit(-sum) direction dictionaries vary with a next-higher-digit context but are
well-related by an approximately orthogonal map inside a shared low-rank subspace (low-rank Procrustes alignment).
Causal digit editing matches this geometry: naive cross-context transfer fails, while rotating directions through the
learned map restores strict counterfactual edits; negative controls do not recover. 

---
# Defining Explainable AI for Requirements Analysis 

**Authors**: Raymond Sheh, Isaac Monteath  

**Link**: [PDF](https://arxiv.org/pdf/2602.19071)  

**Abstract**: Explainable Artificial Intelligence (XAI) has become popular in the last few years. The Artificial Intelligence (AI) community in general, and the Machine Learning (ML) community in particular, is coming to the realisation that in many applications, for AI to be trusted, it must not only demonstrate good performance in its decisionmaking, but it also must explain these decisions and convince us that it is making the decisions for the right reasons. However, different applications have different requirements on the information required of the underlying AI system in order to convince us that it is worthy of our trust. How do we define these requirements?
In this paper, we present three dimensions for categorising the explanatory requirements of different applications. These are Source, Depth and Scope. We focus on the problem of matching up the explanatory requirements of different applications with the capabilities of underlying ML techniques to provide them. We deliberately avoid including aspects of explanation that are already well-covered by the existing literature and we focus our discussion on ML although the principles apply to AI more broadly. 

---
# Asking the Right Questions: Improving Reasoning with Generated Stepping Stones 

**Authors**: Hengyuan Hu, Tingchen Fu, Minqi Jiang, Alexander H Miller, Yoram Bachrach, Jakob Nicolaus Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2602.19069)  

**Abstract**: Recent years have witnessed tremendous progress in enabling LLMs to solve complex reasoning tasks such as math and coding. As we start to apply LLMs to harder tasks that they may not be able to solve in one shot, it is worth paying attention to their ability to construct intermediate stepping stones that prepare them to better solve the tasks. Examples of stepping stones include simplifications, alternative framings, or subproblems. We study properties and benefits of stepping stones in the context of modern reasoning LLMs via ARQ (\textbf{A}king the \textbf{R}ight \textbf{Q}uestions), our simple framework which introduces a question generator to the default reasoning pipeline. We first show that good stepping stone questions exist and are transferrable, meaning that good questions can be generated, and they substantially help LLMs of various capabilities in solving the target tasks. We next frame stepping stone generation as a post-training task and show that we can fine-tune LLMs to generate more useful stepping stones by SFT and RL on synthetic data. 

---
# Agentic Problem Frames: A Systematic Approach to Engineering Reliable Domain Agents 

**Authors**: Chanjin Park  

**Link**: [PDF](https://arxiv.org/pdf/2602.19065)  

**Abstract**: Large Language Models (LLMs) are evolving into autonomous agents, yet current "frameless" development--relying on ambiguous natural language without engineering blueprints--leads to critical risks such as scope creep and open-loop failures. To ensure industrial-grade reliability, this study proposes Agentic Problem Frames (APF), a systematic engineering framework that shifts focus from internal model intelligence to the structured interaction between the agent and its environment.
The APF establishes a dynamic specification paradigm where intent is concretized at runtime through domain knowledge injection. At its core, the Act-Verify-Refine (AVR) loop functions as a closed-loop control system that transforms execution results into verified knowledge assets, driving system behavior toward asymptotic convergence to mission requirements (R). To operationalize this, this study introduces the Agentic Job Description (AJD), a formal specification tool that defines jurisdictional boundaries, operational contexts, and epistemic evaluation criteria.
The efficacy of this framework is validated through two contrasting case studies: a delegated proxy model for business travel and an autonomous supervisor model for industrial equipment management. By applying AJD-based specification and APF modeling to these scenarios, the analysis demonstrates how operational scenarios are systematically controlled within defined boundaries. These cases provide a conceptual proof that agent reliability stems not from a model's internal reasoning alone, but from the rigorous engineering structures that anchor stochastic AI within deterministic business processes, thereby enabling the development of verifiable and dependable domain agents. 

---
# Evaluating Large Language Models on Quantum Mechanics: A Comparative Study Across Diverse Models and Tasks 

**Authors**: S. K. Rithvik  

**Link**: [PDF](https://arxiv.org/pdf/2602.19006)  

**Abstract**: We present a systematic evaluation of large language models on quantum mechanics problem-solving. Our study evaluates 15 models from five providers (OpenAI, Anthropic, Google, Alibaba, DeepSeek) spanning three capability tiers on 20 tasks covering derivations, creative problems, non-standard concepts, and numerical computation, comprising 900 baseline and 75 tool-augmented assessments. Results reveal clear tier stratification: flagship models achieve 81\% average accuracy, outperforming mid-tier (77\%) and fast models (67\%) by 4pp and 14pp respectively. Task difficulty patterns emerge distinctly: derivations show highest performance (92\% average, 100\% for flagship models), while numerical computation remains most challenging (42\%). Tool augmentation on numerical tasks yields task-dependent effects: modest overall improvement (+4.4pp) at 3x token cost masks dramatic heterogeneity ranging from +29pp gains to -16pp degradation. Reproducibility analysis across three runs quantifies 6.3pp average variance, with flagship models demonstrating exceptional stability (GPT-5 achieves zero variance) while specialized models require multi-run evaluation. This work contributes: (i) a benchmark for quantum mechanics with automatic verification, (ii) systematic evaluation quantifying tier-based performance hierarchies, (iii) empirical analysis of tool augmentation trade-offs, and (iv) reproducibility characterization. All tasks, verifiers, and results are publicly released. 

---
# MagicAgent: Towards Generalized Agent Planning 

**Authors**: Xuhui Ren, Shaokang Dong, Chen Yang, Qing Gao, Yunbin Zhao, Yongsheng Liu, Xinwei Geng, Xiang Li, Demei Yan, Yanqing Li, Chenhao Huang, Dingwei Zhu, Junjie Ye, Boxuan Yue, Yingnan Fu, Mengzhe Lv, Zezeng Feng, Boshen Zhou, Bocheng Wang, Xuanjing Huang, Yu-Gang Jiang, Tao Gui, Qi Zhang, Yunke Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.19000)  

**Abstract**: The evolution of Large Language Models (LLMs) from passive text processors to autonomous agents has established planning as a core component of modern intelligence. However, achieving generalized planning remains elusive, not only by the scarcity of high-quality interaction data but also by inherent conflicts across heterogeneous planning tasks. These challenges result in models that excel at isolated tasks yet struggle to generalize, while existing multi-task training attempts suffer from gradient interference. In this paper, we present \textbf{MagicAgent}, a series of foundation models specifically designed for generalized agent planning. We introduce a lightweight and scalable synthetic data framework that generates high-quality trajectories across diverse planning tasks, including hierarchical task decomposition, tool-augmented planning, multi-constraint scheduling, procedural logic orchestration, and long-horizon tool execution. To mitigate training conflicts, we propose a two-stage training paradigm comprising supervised fine-tuning followed by multi-objective reinforcement learning over both static datasets and dynamic environments. Empirical results demonstrate that MagicAgent-32B and MagicAgent-30B-A3B deliver superior performance, achieving accuracies of $75.1\%$ on Worfbench, $55.9\%$ on NaturalPlan, $57.5\%$ on $\tau^2$-Bench, $86.9\%$ on BFCL-v3, and $81.2\%$ on ACEBench, as well as strong results on our in-house MagicEval benchmarks. These results substantially outperform existing sub-100B models and even surpass leading closed-source models. 

---
# Benchmark Test-Time Scaling of General LLM Agents 

**Authors**: Xiaochuan Li, Ryan Ming, Pranav Setlur, Abhijay Paladugu, Andy Tang, Hao Kang, Shuai Shao, Rong Jin, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2602.18998)  

**Abstract**: LLM agents are increasingly expected to function as general-purpose systems capable of resolving open-ended user requests. While existing benchmarks focus on domain-aware environments for developing specialized agents, evaluating general-purpose agents requires more realistic settings that challenge them to operate across multiple skills and tools within a unified environment. We introduce General AgentBench, a benchmark that provides such a unified framework for evaluating general LLM agents across search, coding, reasoning, and tool-use domains. Using General AgentBench, we systematically study test-time scaling behaviors under sequential scaling (iterative interaction) and parallel scaling (sampling multiple trajectories). Evaluation of ten leading LLM agents reveals a substantial performance degradation when moving from domain-specific evaluations to this general-agent setting. Moreover, we find that neither scaling methodology yields effective performance improvements in practice, due to two fundamental limitations: context ceiling in sequential scaling and verification gap in parallel scaling. Code is publicly available at this https URL. 

---
# Quantifying Automation Risk in High-Automation AI Systems: A Bayesian Framework for Failure Propagation and Optimal Oversight 

**Authors**: Vishal Srivastava, Tanmay Sah  

**Link**: [PDF](https://arxiv.org/pdf/2602.18986)  

**Abstract**: Organizations across finance, healthcare, transportation, content moderation, and critical infrastructure are rapidly deploying highly automated AI systems, yet they lack principled methods to quantify how increasing automation amplifies harm when failures occur. We propose a parsimonious Bayesian risk decomposition expressing expected loss as the product of three terms: the probability of system failure, the conditional probability that a failure propagates into harm given the automation level, and the expected severity of harm. This framework isolates a critical quantity -- the conditional probability that failures propagate into harm -- which captures execution and oversight risk rather than model accuracy alone. We develop complete theoretical foundations: formal proofs of the decomposition, a harm propagation equivalence theorem linking the harm propagation probability to observable execution controls, risk elasticity measures, efficient frontier analysis for automation policy, and optimal resource allocation principles with second-order conditions. We motivate the framework with an illustrative case study of the 2012 Knight Capital incident ($440M loss) as one instantiation of a broadly applicable failure pattern, and characterize the research design required to empirically validate the framework at scale across deployment domains. This work provides the theoretical foundations for a new class of deployment-focused risk governance tools for agentic and automated AI systems. 

---
# InfEngine: A Self-Verifying and Self-Optimizing Intelligent Engine for Infrared Radiation Computing 

**Authors**: Kun Ding, Jian Xu, Ying Wang, Peipei Yang, Shiming Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18985)  

**Abstract**: Infrared radiation computing underpins advances in climate science, remote sensing and spectroscopy but remains constrained by manual workflows. We introduce InfEngine, an autonomous intelligent computational engine designed to drive a paradigm shift from human-led orchestration to collaborative automation. It integrates four specialized agents through two core innovations: self-verification, enabled by joint solver-evaluator debugging, improves functional correctness and scientific plausibility; self-optimization, realized via evolutionary algorithms with self-discovered fitness functions, facilitates autonomous performance optimization. Evaluated on InfBench with 200 infrared-specific tasks and powered by InfTools with 270 curated tools, InfEngine achieves a 92.7% pass rate and delivers workflows 21x faster than manual expert effort. More fundamentally, it illustrates how researchers can transition from manual coding to collaborating with self-verifying, self-optimizing computational partners. By generating reusable, verified and optimized code, InfEngine transforms computational workflows into persistent scientific assets, accelerating the cycle of scientific discovery. Code: this https URL 

---
# How Far Can We Go with Pixels Alone? A Pilot Study on Screen-Only Navigation in Commercial 3D ARPGs 

**Authors**: Kaijie Xu, Mustafa Bugti, Clark Verbrugge  

**Link**: [PDF](https://arxiv.org/pdf/2602.18981)  

**Abstract**: Modern 3D game levels rely heavily on visual guidance, yet the navigability of level layouts remains difficult to quantify. Prior work either simulates play in simplified environments or analyzes static screenshots for visual affordances, but neither setting faithfully captures how players explore complex, real-world game levels. In this paper, we build on an existing open-source visual affordance detector and instantiate a screen-only exploration and navigation agent that operates purely from visual affordances. Our agent consumes live game frames, identifies salient interest points, and drives a simple finite-state controller over a minimal action space to explore Dark Souls-style linear levels and attempt to reach expected goal regions. Pilot experiments show that the agent can traverse most required segments and exhibits meaningful visual navigation behavior, but also highlight that limitations of the underlying visual model prevent truly comprehensive and reliable auto-navigation. We argue that this system provides a concrete, shared baseline and evaluation protocol for visual navigation in complex games, and we call for more attention to this necessary task. Our results suggest that purely vision-based sense-making models, with discrete single-modality inputs and without explicit reasoning, can effectively support navigation and environment understanding in idealized settings, but are unlikely to be a general solution on their own. 

---
# When Do LLM Preferences Predict Downstream Behavior? 

**Authors**: Katarina Slama, Alexandra Souly, Dishank Bansal, Henry Davidson, Christopher Summerfield, Lennart Luettgau  

**Link**: [PDF](https://arxiv.org/pdf/2602.18971)  

**Abstract**: Preference-driven behavior in LLMs may be a necessary precondition for AI misalignment such as sandbagging: models cannot strategically pursue misaligned goals unless their behavior is influenced by their preferences. Yet prior work has typically prompted models explicitly to act in specific ways, leaving unclear whether observed behaviors reflect instruction-following capabilities vs underlying model preferences. Here we test whether this precondition for misalignment is present. Using entity preferences as a behavioral probe, we measure whether stated preferences predict downstream behavior in five frontier LLMs across three domains: donation advice, refusal behavior, and task performance. Conceptually replicating prior work, we first confirm that all five models show highly consistent preferences across two independent measurement methods. We then test behavioral consequences in a simulated user environment. We find that all five models give preference-aligned donation advice. All five models also show preference-correlated refusal patterns when asked to recommend donations, refusing more often for less-preferred entities. All preference-related behaviors that we observe here emerge without instructions to act on preferences. Results for task performance are mixed: on a question-answering benchmark (BoolQ), two models show small but significant accuracy differences favoring preferred entities; one model shows the opposite pattern; and two models show no significant relationship. On complex agentic tasks, we find no evidence of preference-driven performance differences. While LLMs have consistent preferences that reliably predict advice-giving behavior, these preferences do not consistently translate into downstream task performance. 

---
# Robust and Efficient Tool Orchestration via Layered Execution Structures with Reflective Correction 

**Authors**: Tao Zhe, Haoyu Wang, Bo Luo, Min Wu, Wei Fan, Xiao Luo, Zijun Yao, Haifeng Chen, Dongjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18968)  

**Abstract**: Tool invocation is a core capability of agentic systems, yet failures often arise not from individual tool calls but from how multiple tools are organized and executed together. Existing approaches tightly couple tool execution with stepwise language reasoning or explicit planning, leading to brittle behavior and high execution overhead. To overcome these limitations, we revisit tool invocation from the perspective of tool orchestration. Our key insight is that effective orchestration does not require precise dependency graphs or fine-grained planning. Instead, a coarse-grained layer structure suffices to provide global guidance, while execution-time errors can be corrected locally. Specifically, we model tool orchestration as learning a layered execution structure that captures high-level tool dependencies, inducing layer-wise execution through context constraints. To handle execution-time failures, we introduce a schema-aware reflective correction mechanism that detects and repairs errors locally. This design confines errors to individual tool calls and avoids re-planning entire execution trajectories. This structured execution paradigm enables a lightweight and reusable orchestration component for agentic systems. Experimental results show that our approach achieves robust tool execution while reducing execution complexity and overhead. Code will be made publicly available. 

---
# Modularity is the Bedrock of Natural and Artificial Intelligence 

**Authors**: Alessandro Salatiello  

**Link**: [PDF](https://arxiv.org/pdf/2602.18960)  

**Abstract**: The remarkable performance of modern AI systems has been driven by unprecedented scales of data, computation, and energy -- far exceeding the resources required by human intelligence. This disparity highlights the need for new guiding principles and motivates drawing inspiration from the fundamental organizational principles of brain computation. Among these principles, modularity has been shown to be critical for supporting the efficient learning and strong generalization abilities consistently exhibited by humans. Furthermore, modularity aligns well with the No Free Lunch Theorem, which highlights the need for problem-specific inductive biases and motivates architectures composed of specialized components that solve subproblems. However, despite its fundamental role in natural intelligence and its demonstrated benefits across a range of seemingly disparate AI subfields, modularity remains relatively underappreciated in mainstream AI research. In this work, we review several research threads in artificial intelligence and neuroscience through a conceptual framework that highlights the central role of modularity in supporting both artificial and natural intelligence. In particular, we examine what computational advantages modularity provides, how it has emerged as a solution across several AI research areas, which modularity principles the brain exploits, and how modularity can help bridge the gap between natural and artificial intelligence. 

---
# INDUCTION: Finite-Structure Concept Synthesis in First-Order Logic 

**Authors**: Serafim Batzoglou  

**Link**: [PDF](https://arxiv.org/pdf/2602.18956)  

**Abstract**: We introduce INDUCTION, a benchmark for finite structure concept synthesis in first order logic. Given small finite relational worlds with extensionally labeled target predicates, models must output a single first order logical formula that explains the target uniformly across worlds, with correctness verified via exact model checking. The benchmark includes three regimes, FullObs, CI (contrastive), and EC (existential completion), nd penalizes formula bloat. We find sharp difficulty gradients, persistent hard structural families, and observe that low bloat formulas generalize far better on held out worlds. Elite recent models show qualitatively different behaviors across tasks and performance metrics, hinting to their different strategies of concept generalization. 

---
# (Perlin) Noise as AI coordinator 

**Authors**: Kaijie Xu, Clark Verbrugge  

**Link**: [PDF](https://arxiv.org/pdf/2602.18947)  

**Abstract**: Large scale control of nonplayer agents is central to modern games, while production systems still struggle to balance several competing goals: locally smooth, natural behavior, and globally coordinated variety across space and time. Prior approaches rely on handcrafted rules or purely stochastic triggers, which either converge to mechanical synchrony or devolve into uncorrelated noise that is hard to tune. Continuous noise signals such as Perlin noise are well suited to this gap because they provide spatially and temporally coherent randomness, and they are already widely used for terrain, biomes, and other procedural assets. We adapt these signals for the first time to large scale AI control and present a general framework that treats continuous noise fields as an AI coordinator. The framework combines three layers of control: behavior parameterization for movement at the agent level, action time scheduling for when behaviors start and stop, and spawn or event type and feature generation for what appears and where. We instantiate the framework reproducibly and evaluate Perlin noise as a representative coordinator across multiple maps, scales, and seeds against random, filtered, deterministic, neighborhood constrained, and physics inspired baselines. Experiments show that coordinated noise fields provide stable activation statistics without lockstep, strong spatial coverage and regional balance, better diversity with controllable polarization, and competitive runtime. We hope this work motivates a broader exploration of coordinated noise in game AI as a practical path to combine efficiency, controllability, and quality. 

---
# High Dimensional Procedural Content Generation 

**Authors**: Kaijie Xu, Clark Verbrugge  

**Link**: [PDF](https://arxiv.org/pdf/2602.18943)  

**Abstract**: Procedural content generation (PCG) has made substantial progress in shaping static 2D/3D geometry, while most methods treat gameplay mechanics as auxiliary and optimize only over space. We argue that this limits controllability and expressivity, and formally introduce High-Dimensional PCG (HDPCG): a framework that elevates non-geometric gameplay dimensions to first-class coordinates of a joint state space. We instantiate HDPCG along two concrete directions. Direction-Space augments geometry with a discrete layer dimension and validates reachability in 4D (x,y,z,l), enabling unified treatment of 2.5D/3.5D mechanics such as gravity inversion and parallel-world switching. Direction-Time augments geometry with temporal dynamics via time-expanded graphs, capturing action semantics and conflict rules. For each direction, we present three general, practicable algorithms with a shared pipeline of abstract skeleton generation, controlled grounding, high-dimensional validation, and multi-metric evaluation. Large-scale experiments across diverse settings validate the integrity of our problem formulation and the effectiveness of our methods on playability, structure, style, robustness, and efficiency. Beyond quantitative results, Unity-based case studies recreate playable scenarios that accord with our metrics. We hope HDPCG encourages a shift in PCG toward general representations and the generation of gameplay-relevant dimensions beyond geometry, paving the way for controllable, verifiable, and extensible level generation. 

---
# DREAM: Deep Research Evaluation with Agentic Metrics 

**Authors**: Elad Ben Avraham, Changhao Li, Ron Dorfman, Roy Ganz, Oren Nuriel, Amir Dudai, Aviad Aberdam, Noah Flynn, Elman Mansimov, Adi Kalyanpur, Ron Litman  

**Link**: [PDF](https://arxiv.org/pdf/2602.18940)  

**Abstract**: Deep Research Agents generate analyst-grade reports, yet evaluating them remains challenging due to the absence of a single ground truth and the multidimensional nature of research quality. Recent benchmarks propose distinct methodologies, yet they suffer from the Mirage of Synthesis, where strong surface-level fluency and citation alignment can obscure underlying factual and reasoning defects. We characterize this gap by introducing a taxonomy across four verticals that exposes a critical capability mismatch: static evaluators inherently lack the tool-use capabilities required to assess temporal validity and factual correctness. To address this, we propose DREAM (Deep Research Evaluation with Agentic Metrics), a framework that instantiates the principle of capability parity by making evaluation itself agentic. DREAM structures assessment through an evaluation protocol combining query-agnostic metrics with adaptive metrics generated by a tool-calling agent, enabling temporally aware coverage, grounded verification, and systematic reasoning probes. Controlled evaluations demonstrate DREAM is significantly more sensitive to factual and temporal decay than existing benchmarks, offering a scalable, reference-free evaluation paradigm. 

---
# Early Evidence of Vibe-Proving with Consumer LLMs: A Case Study on Spectral Region Characterization with ChatGPT-5.2 (Thinking) 

**Authors**: Brecht Verbeken, Brando Vagenende, Marie-Anne Guerry, Andres Algaba, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2602.18918)  

**Abstract**: Large Language Models (LLMs) are increasingly used as scientific copilots, but evidence on their role in research-level mathematics remains limited, especially for workflows accessible to individual researchers. We present early evidence for vibe-proving with a consumer subscription LLM through an auditable case study that resolves Conjecture 20 of Ran and Teng (2024) on the exact nonreal spectral region of a 4-cycle row-stochastic nonnegative matrix family. We analyze seven shareable ChatGPT-5.2 (Thinking) threads and four versioned proof drafts, documenting an iterative pipeline of generate, referee, and repair. The model is most useful for high-level proof search, while human experts remain essential for correctness-critical closure. The final theorem provides necessary and sufficient region conditions and explicit boundary attainment constructions. Beyond the mathematical result, we contribute a process-level characterization of where LLM assistance materially helps and where verification bottlenecks persist, with implications for evaluation of AI-assisted research workflows and for designing human-in-the-loop theorem proving systems. 

---
# TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models 

**Authors**: Zhenkun Gao, Xuhong Wang, Xin Tan, Yuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2602.18884)  

**Abstract**: Multimodal Large Language Models (MLLMs), particularly smaller, deployable variants, exhibit a critical deficiency in understanding temporal and procedural visual data, a bottleneck hindering their application in real-world embodied AI. This gap is largely caused by a systemic failure in training paradigms, which lack large-scale, procedurally coherent data. To address this problem, we introduce TPRU, a large-scale dataset sourced from diverse embodied scenarios such as robotic manipulation and GUI navigation. TPRU is systematically designed to cultivate temporal reasoning through three complementary tasks: Temporal Reordering, Next-Frame Prediction, and Previous-Frame Review. A key feature is the inclusion of challenging negative samples, compelling models to transition from passive observation to active, cross-modal validation. We leverage TPRU with a reinforcement learning (RL) fine-tuning methodology, specifically targeting the enhancement of resource-efficient models. Experiments show our approach yields dramatic gains: on our manually curated TPRU-Test, the accuracy of TPRU-7B soars from 50.33\% to 75.70\%, a state-of-the-art result that significantly outperforms vastly larger baselines, including GPT-4o. Crucially, these capabilities generalize effectively, demonstrating substantial improvements on established benchmarks. The codebase is available at this https URL . 

---
# ABD: Default Exception Abduction in Finite First Order Worlds 

**Authors**: Serafim Batzoglou  

**Link**: [PDF](https://arxiv.org/pdf/2602.18843)  

**Abstract**: We introduce ABD, a benchmark for default-exception abduction over finite first-order worlds. Given a background theory with an abnormality predicate and a set of relational structures, a model must output a first-order formula that defines exceptions, restoring satisfiability while keeping exceptions sparse. We formalize three observation regimes (closed-world, existential completion, universal completion) with exact SMT verification. Evaluating ten frontier LLMs on 600 instances, the best models achieve high validity but parsimony gaps remain, and holdout evaluation reveals distinct generalization failure modes across regimes. 

---
# GenPlanner: From Noise to Plans -- Emergent Reasoning in Flow Matching and Diffusion Models 

**Authors**: Agnieszka Polowczyk, Alicja Polowczyk, Michał Wieczorek  

**Link**: [PDF](https://arxiv.org/pdf/2602.18812)  

**Abstract**: Path planning in complex environments is one of the key problems of artificial intelligence because it requires simultaneous understanding of the geometry of space and the global structure of the problem. In this paper, we explore the potential of using generative models as planning and reasoning mechanisms. We propose GenPlanner, an approach based on diffusion models and flow matching, along with two variants: DiffPlanner and FlowPlanner. We demonstrate the application of generative models to find and generate correct paths in mazes. A multi-channel condition describing the structure of the environment, including an obstacle map and information about the starting and destination points, is used to condition trajectory generation. Unlike standard methods, our models generate trajectories iteratively, starting with random noise and gradually transforming it into a correct solution. Experiments conducted show that the proposed approach significantly outperforms the baseline CNN model. In particular, FlowPlanner demonstrates high performance even with a limited number of generation steps. 

---
# LAMMI-Pathology: A Tool-Centric Bottom-Up LVLM-Agent Framework for Molecularly Informed Medical Intelligence in Pathology 

**Authors**: Haoyang Su, Shaoting Zhang, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18773)  

**Abstract**: The emergence of tool-calling-based agent systems introduces a more evidence-driven paradigm for pathology image analysis in contrast to the coarse-grained text-image diagnostic approaches. With the recent large-scale experimental adoption of spatial transcriptomics technologies, molecularly validated pathological diagnosis is becoming increasingly open and accessible. In this work, we propose LAMMI-Pathology (LVLM-Agent System for Molecularly Informed Medical Intelligence in Pathology), a scalable agent framework for domain-specific agent tool-calling. LAMMI-Pathology adopts a tool-centric, bottom-up architecture in which customized domain-adaptive tools serve as the foundation. These tools are clustered by domain style to form component agents, which are then coordinated through a top-level planner hierarchically, avoiding excessively long context lengths that could induce task drift. Based on that, we introduce a novel trajectory construction mechanism based on Atomic Execution Nodes (AENs), which serve as reliable and composable units for building semi-simulated reasoning trajectories that capture credible agent-tool interactions. Building on this foundation, we develop a trajectory-aware fine-tuning strategy that aligns the planner's decision-making process with these multi-step reasoning trajectories, thereby enhancing inference robustness in pathology understanding and its adaptive use of the customized toolset. 

---
# The Convergence of Schema-Guided Dialogue Systems and the Model Context Protocol 

**Authors**: Andreas Schlapbach  

**Link**: [PDF](https://arxiv.org/pdf/2602.18764)  

**Abstract**: This paper establishes a fundamental convergence: Schema-Guided Dialogue (SGD) and the Model Context Protocol (MCP) represent two manifestations of a unified paradigm for deterministic, auditable LLM-agent interaction. SGD, designed for dialogue-based API discovery (2019), and MCP, now the de facto standard for LLM-tool integration, share the same core insight -- that schemas can encode not just tool signatures but operational constraints and reasoning guidance. By analyzing this convergence, we extract five foundational principles for schema design: (1) Semantic Completeness over Syntactic Precision, (2) Explicit Action Boundaries, (3) Failure Mode Documentation, (4) Progressive Disclosure Compatibility, and (5) Inter-Tool Relationship Declaration. These principles reveal three novel insights: first, SGD's original design was fundamentally sound and should be inherited by MCP; second, both frameworks leave failure modes and inter-tool relationships unexploited -- gaps we identify and resolve; third, progressive disclosure emerges as a critical production-scaling insight under real-world token constraints. We provide concrete design patterns for each principle. These principles position schema-driven governance as a scalable mechanism for AI system oversight without requiring proprietary system inspection -- central to Software 3.0. 

---
# Federated Reasoning Distillation Framework with Model Learnability-Aware Data Allocation 

**Authors**: Wei Guo, Siyuan Lu, Xiangdong Ran, Yiqi Tong, Yikun Ban, Zelong Xu, Jing Fan, Zixuan Huang, Xiao Zhang, Zhaojun Hu, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18749)  

**Abstract**: Data allocation plays a critical role in federated large language model (LLM) and small language models (SLMs) reasoning collaboration. Nevertheless, existing data allocation methods fail to address an under-explored challenge in collaboration: bidirectional model learnability gap, where client-side SLMs cannot identify high-reward samples matching their learnability constraints for effective knowledge transfer from LLMs, while LLMs struggle to select samples contributing novel knowledge beyond their existing data. Furthermore, these collaboration frameworks face another key challenge: domain-agnostic reasoning transfer, where existing reasoning transfer methods fail to flexibly adapt to the local domain data, preventing SLMs from effectively acquiring step-by-step reasoning abilities within from general LLM. To address these challenges, we propose LaDa, a federated reasoning distillation framework with model learnability-aware data allocation. It introduces a model learnability-aware data filter that adaptively allocates high-reward samples based on the learnability gap between each SLM and LLM pair, effectively facilitating bidirectional knowledge transfer. We further design a domain adaptive reasoning distillation method that aligns joint probabilities of reasoning paths on filtered high-reward samples through contrastive distillation learning between SLM and LLM, enabling SLM to capture underlying reasoning patterns under local data distribution. LaDa operates as a plug-in module for existing collaboration frameworks, adapting knowledge transfer based on model learnability gaps. 

---
# Beyond Description: A Multimodal Agent Framework for Insightful Chart Summarization 

**Authors**: Yuhang Bai, Yujuan Ding, Shanru Lin, Wenqi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2602.18731)  

**Abstract**: Chart summarization is crucial for enhancing data accessibility and the efficient consumption of information. However, existing methods, including those with Multimodal Large Language Models (MLLMs), primarily focus on low-level data descriptions and often fail to capture the deeper insights which are the fundamental purpose of data visualization. To address this challenge, we propose Chart Insight Agent Flow, a plan-and-execute multi-agent framework effectively leveraging the perceptual and reasoning capabilities of MLLMs to uncover profound insights directly from chart images. Furthermore, to overcome the lack of suitable benchmarks, we introduce ChartSummInsights, a new dataset featuring a diverse collection of real-world charts paired with high-quality, insightful summaries authored by human data analysis experts. Experimental results demonstrate that our method significantly improves the performance of MLLMs on the chart summarization task, producing summaries with deep and diverse insights. 

---
# Task-Aware Exploration via a Predictive Bisimulation Metric 

**Authors**: Dayang Liang, Ruihan Liu, Lipeng Wan, Yunlong Liu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2602.18724)  

**Abstract**: Accelerating exploration in visual reinforcement learning under sparse rewards remains challenging due to the substantial task-irrelevant variations. Despite advances in intrinsic exploration, many methods either assume access to low-dimensional states or lack task-aware exploration strategies, thereby rendering them fragile in visual domains. To bridge this gap, we present TEB, a Task-aware Exploration approach that tightly couples task-relevant representations with exploration through a predictive Bisimulation metric. Specifically, TEB leverages the metric not only to learn behaviorally grounded task representations but also to measure behaviorally intrinsic novelty over the learned latent space. To realize this, we first theoretically mitigate the representation collapse of degenerate bisimulation metrics under sparse rewards by internally introducing a simple but effective predicted reward differential. Building on this robust metric, we design potential-based exploration bonuses, which measure the relative novelty of adjacent observations over the latent space. Extensive experiments on MetaWorld and Maze2D show that TEB achieves superior exploration ability and outperforms recent baselines. 

---
# Many AI Analysts, One Dataset: Navigating the Agentic Data Science Multiverse 

**Authors**: Martin Bertran, Riccardo Fogliato, Zhiwei Steven Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.18710)  

**Abstract**: The conclusions of empirical research depend not only on data but on a sequence of analytic decisions that published results seldom make explicit. Past ``many-analyst" studies have demonstrated this: independent teams testing the same hypothesis on the same dataset regularly reach conflicting conclusions. But such studies require months of coordination among dozens of research groups and are therefore rarely conducted. In this work, we show that fully autonomous AI analysts built on large language models (LLMs) can reproduce a similar structured analytic diversity cheaply and at scale. We task these AI analysts with testing a pre-specified hypothesis on a fixed dataset, varying the underlying model and prompt framing across replicate runs. Each AI analyst independently constructs and executes a full analysis pipeline; an AI auditor then screens each run for methodological validity. Across three datasets spanning experimental and observational designs, AI analyst-produced analyses display wide dispersion in effect sizes, $p$-values, and binary decisions on supporting the hypothesis or not, frequently reversing whether a hypothesis is judged supported. This dispersion is structured: recognizable analytic choices in preprocessing, model specification, and inference differ systematically across LLM and persona conditions. Critically, the effects are \emph{steerable}: reassigning the analyst persona or LLM shifts the distribution of outcomes even after excluding methodologically deficient runs. 

---
# Spilled Energy in Large Language Models 

**Authors**: Adrian Robert Minut, Hazem Dewidar, Iacopo Masi  

**Link**: [PDF](https://arxiv.org/pdf/2602.18671)  

**Abstract**: We reinterpret the final Large Language Model (LLM) softmax classifier as an Energy-Based Model (EBM), decomposing the sequence-to-sequence probability chain into multiple interacting EBMs at inference. This principled approach allows us to track "energy spills" during decoding, which we empirically show correlate with factual errors, biases, and failures. Similar to Orgad et al. (2025), our method localizes the exact answer token and subsequently tests for hallucinations. Crucially, however, we achieve this without requiring trained probe classifiers or activation ablations. Instead, we introduce two completely training-free metrics derived directly from output logits: spilled energy, which captures the discrepancy between energy values across consecutive generation steps that should theoretically match, and marginalized energy, which is measurable at a single step. Evaluated on nine benchmarks across state-of-the-art LLMs (including LLaMA, Mistral, and Gemma) and on synthetic algebraic operations (Qwen3), our approach demonstrates robust, competitive hallucination detection and cross-task generalization. Notably, these results hold for both pretrained and instruction-tuned variants without introducing any training overhead. 

---
# Decoding ML Decision: An Agentic Reasoning Framework for Large-Scale Ranking System 

**Authors**: Longfei Yun, Yihan Wu, Haoran Liu, Xiaoxuan Liu, Ziyun Xu, Yi Wang, Yang Xia, Pengfei Wang, Mingze Gao, Yunxiang Wang, Changfan Chen, Junfeng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2602.18640)  

**Abstract**: Modern large-scale ranking systems operate within a sophisticated landscape of competing objectives, operational constraints, and evolving product requirements. Progress in this domain is increasingly bottlenecked by the engineering context constraint: the arduous process of translating ambiguous product intent into reasonable, executable, verifiable hypotheses, rather than by modeling techniques alone. We present GEARS (Generative Engine for Agentic Ranking Systems), a framework that reframes ranking optimization as an autonomous discovery process within a programmable experimentation environment. Rather than treating optimization as static model selection, GEARS leverages Specialized Agent Skills to encapsulate ranking expert knowledge into reusable reasoning capabilities, enabling operators to steer systems via high-level intent vibe personalization. Furthermore, to ensure production reliability, the framework incorporates validation hooks to enforce statistical robustness and filter out brittle policies that overfit short-term signals. Experimental validation across diverse product surfaces demonstrates that GEARS consistently identifies superior, near-Pareto-efficient policies by synergizing algorithmic signals with deep ranking context while maintaining rigorous deployment stability. 

---
# Feedback-based Automated Verification in Vibe Coding of CAS Adaptation Built on Constraint Logic 

**Authors**: Michal Töpfer, František Plášil, Tomáš Bureš, Petr Hnětynka  

**Link**: [PDF](https://arxiv.org/pdf/2602.18607)  

**Abstract**: In CAS adaptation, a challenge is to define the dynamic architecture of the system and changes in its behavior. Implementation-wise, this is projected into an adaptation mechanism, typically realized as an Adaptation Manager (AM). With the advances of generative LLMs, generating AM code based on system specification and desired AM behavior (partially in natural language) is a tempting opportunity. The recent introduction of vibe coding suggests a way to target the problem of the correctness of generated code by iterative testing and vibe coding feedback loops instead of direct code inspection.
In this paper, we show that generating an AM via vibe coding feedback loops is a viable option when the verification of the generated AM is based on a very precise formulation of the functional requirements. We specify these as constraints in a novel temporal logic FCL that allows us to express the behavior of traces with much finer granularity than classical LTL enables.
Furthermore, we show that by combining the adaptation and vibe coding feedback loops where the FCL constraints are evaluated for the current system state, we achieved good results in the experiments with generating AMs for two example systems from the CAS domain. Typically, just a few feedback loop iterations were necessary, each feeding the LLM with reports describing detailed violations of the constraints. This AM testing was combined with high run path coverage achieved by different initial settings. 

---
# Hierarchical Reward Design from Language: Enhancing Alignment of Agent Behavior with Human Specifications 

**Authors**: Zhiqin Qian, Ryan Diaz, Sangwon Seo, Vaibhav Unhelkar  

**Link**: [PDF](https://arxiv.org/pdf/2602.18582)  

**Abstract**: When training artificial intelligence (AI) to perform tasks, humans often care not only about whether a task is completed but also how it is performed. As AI agents tackle increasingly complex tasks, aligning their behavior with human-provided specifications becomes critical for responsible AI deployment. Reward design provides a direct channel for such alignment by translating human expectations into reward functions that guide reinforcement learning (RL). However, existing methods are often too limited to capture nuanced human preferences that arise in long-horizon tasks. Hence, we introduce Hierarchical Reward Design from Language (HRDL): a problem formulation that extends classical reward design to encode richer behavioral specifications for hierarchical RL agents. We further propose Language to Hierarchical Rewards (L2HR) as a solution to HRDL. Experiments show that AI agents trained with rewards designed via L2HR not only complete tasks effectively but also better adhere to human specifications. Together, HRDL and L2HR advance the research on human-aligned AI agents. 

---
# On the Dynamics of Observation and Semantics 

**Authors**: Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.18494)  

**Abstract**: A dominant paradigm in visual intelligence treats semantics as a static property of latent representations, assuming that meaning can be discovered through geometric proximity in high dimensional embedding spaces. In this work, we argue that this view is physically incomplete. We propose that intelligence is not a passive mirror of reality but a property of a physically realizable agent, a system bounded by finite memory, finite compute, and finite energy interacting with a high entropy environment. We formalize this interaction through the kinematic structure of an Observation Semantics Fiber Bundle, where raw sensory observation data (the fiber) is projected onto a low entropy causal semantic manifold (the base). We prove that for any bounded agent, the thermodynamic cost of information processing (Landauer's Principle) imposes a strict limit on the complexity of internal state transitions. We term this limit the Semantic Constant B. From these physical constraints, we derive the necessity of symbolic structure. We show that to model a combinatorial world within the bound B, the semantic manifold must undergo a phase transition, it must crystallize into a discrete, compositional, and factorized form. Thus, language and logic are not cultural artifacts but ontological necessities the solid state of information required to prevent thermal collapse. We conclude that understanding is not the recovery of a hidden latent variable, but the construction of a causal quotient that renders the world algorithmically compressible and causally predictable. 

---
# A Very Big Video Reasoning Suite 

**Authors**: Maijunxian Wang, Ruisi Wang, Juyi Lin, Ran Ji, Thaddäus Wiedemer, Qingying Gao, Dezhi Luo, Yaoyao Qian, Lianyu Huang, Zelong Hong, Jiahui Ge, Qianli Ma, Hang He, Yifan Zhou, Lingzi Guo, Lantao Mei, Jiachen Li, Hanwen Xing, Tianqi Zhao, Fengyuan Yu, Weihang Xiao, Yizheng Jiao, Jianheng Hou, Danyang Zhang, Pengcheng Xu, Boyang Zhong, Zehong Zhao, Gaoyun Fang, John Kitaoka, Yile Xu, Hua Xu, Kenton Blacutt, Tin Nguyen, Siyuan Song, Haoran Sun, Shaoyue Wen, Linyang He, Runming Wang, Yanzhi Wang, Mengyue Yang, Ziqiao Ma, Raphaël Millière, Freda Shi, Nuno Vasconcelos, Daniel Khashabi, Alan Yuille, Yilun Du, Ziming Liu, Bo Li, Dahua Lin, Ziwei Liu, Vikash Kumar, Yijiang Li, Lei Yang, Zhongang Cai, Hokin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.20159)  

**Abstract**: Rapid progress in video models has largely focused on visual quality, leaving their reasoning capabilities underexplored. Video reasoning grounds intelligence in spatiotemporally consistent visual environments that go beyond what text can naturally capture, enabling intuitive reasoning over spatiotemporal structure such as continuity, interaction, and causality. However, systematically studying video reasoning and its scaling behavior is hindered by the lack of large-scale training data. To address this gap, we introduce the Very Big Video Reasoning (VBVR) Dataset, an unprecedentedly large-scale resource spanning 200 curated reasoning tasks following a principled taxonomy and over one million video clips, approximately three orders of magnitude larger than existing datasets. We further present VBVR-Bench, a verifiable evaluation framework that moves beyond model-based judging by incorporating rule-based, human-aligned scorers, enabling reproducible and interpretable diagnosis of video reasoning capabilities. Leveraging the VBVR suite, we conduct one of the first large-scale scaling studies of video reasoning and observe early signs of emergent generalization to unseen reasoning tasks. Together, VBVR lays a foundation for the next stage of research in generalizable video reasoning. The data, benchmark toolkit, and models are publicly available at this https URL . 

---
# Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data 

**Authors**: Zhenyao Ma, Yue Liang, Dongxu Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.20152)  

**Abstract**: Inspired by behavioral science, we propose Behavior Learning (BL), a novel general-purpose machine learning framework that learns interpretable and identifiable optimization structures from data, ranging from single optimization problems to hierarchical compositions. It unifies predictive performance, intrinsic interpretability, and identifiability, with broad applicability to scientific domains involving optimization. BL parameterizes a compositional utility function built from intrinsically interpretable modular blocks, which induces a data distribution for prediction and generation. Each block represents and can be written in symbolic form as a utility maximization problem (UMP), a foundational paradigm in behavioral science and a universal framework of optimization. BL supports architectures ranging from a single UMP to hierarchical compositions, the latter modeling hierarchical optimization structures. Its smooth and monotone variant (IBL) guarantees identifiability. Theoretically, we establish the universal approximation property of BL, and analyze the M-estimation properties of IBL. Empirically, BL demonstrates strong predictive performance, intrinsic interpretability and scalability to high-dimensional data. Code: this https URL ; install via pip install blnetwork. 

---
# Agentic AI for Scalable and Robust Optical Systems Control 

**Authors**: Zehao Wang, Mingzhe Han, Wei Cheng, Yue-Kai Huang, Philip Ji, Denton Wu, Mahdi Safari, Flemming Holtorf, Kenaish AlQubaisi, Norbert M. Linke, Danyang Zhuo, Yiran Chen, Ting Wang, Dirk Englund, Tingjun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.20144)  

**Abstract**: We present AgentOptics, an agentic AI framework for high-fidelity, autonomous optical system control built on the Model Context Protocol (MCP). AgentOptics interprets natural language tasks and executes protocol-compliant actions on heterogeneous optical devices through a structured tool abstraction layer. We implement 64 standardized MCP tools across 8 representative optical devices and construct a 410-task benchmark to evaluate request understanding, role-aware responses, multi-step coordination, robustness to linguistic variation, and error handling. We assess two deployment configurations--commercial online LLMs and locally hosted open-source LLMs--and compare them with LLM-based code generation baselines. AgentOptics achieves 87.7%--99.0% average task success rates, significantly outperforming code-generation approaches, which reach up to 50% success. We further demonstrate broader applicability through five case studies extending beyond device-level control to system orchestration, monitoring, and closed-loop optimization. These include DWDM link provisioning and coordinated monitoring of coherent 400 GbE and analog radio-over-fiber (ARoF) channels; autonomous characterization and bias optimization of a wideband ARoF link carrying 5G fronthaul traffic; multi-span channel provisioning with launch power optimization; closed-loop fiber polarization stabilization; and distributed acoustic sensing (DAS)-based fiber monitoring with LLM-assisted event detection. These results establish AgentOptics as a scalable, robust paradigm for autonomous control and orchestration of heterogeneous optical systems. 

---
# KNIGHT: Knowledge Graph-Driven Multiple-Choice Question Generation with Adaptive Hardness Calibration 

**Authors**: Mohammad Amanlou, Erfan Shafiee Moghaddam, Yasaman Amou Jafari, Mahdi Noori, Farhan Farsi, Behnam Bahrak  

**Link**: [PDF](https://arxiv.org/pdf/2602.20135)  

**Abstract**: With the rise of large language models (LLMs), they have become instrumental in applications such as Retrieval-Augmented Generation (RAG). Yet evaluating these systems remains bottlenecked by the time and cost of building specialized assessment datasets. We introduce KNIGHT, an LLM-based, knowledge-graph-driven framework for generating multiple-choice question (MCQ) datasets from external sources. KNIGHT constructs a topic-specific knowledge graph, a structured and parsimonious summary of entities and relations, that can be reused to generate instructor-controlled difficulty levels, including multi-hop questions, without repeatedly re-feeding the full source text. This knowledge graph acts as a compressed, reusable state, making question generation a cheap read over the graph. We instantiate KNIGHT on Wikipedia/Wikidata while keeping the framework domain- and ontology-agnostic. As a case study, KNIGHT produces six MCQ datasets in History, Biology, and Mathematics. We evaluate quality on five criteria: fluency, unambiguity (single correct answer), topic relevance, option uniqueness, and answerability given the provided sources (as a proxy for hallucination). Results show that KNIGHT enables token- and cost-efficient generation from a reusable graph representation, achieves high quality across these criteria, and yields model rankings aligned with MMLU-style benchmarks, while supporting topic-specific and difficulty-controlled evaluation. 

---
# Modeling Epidemiological Dynamics Under Adversarial Data and User Deception 

**Authors**: Yiqi Su, Christo Kurisummoottil Thomas, Walid Saad, Bud Mishra, Naren Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2602.20134)  

**Abstract**: Epidemiological models increasingly rely on self-reported behavioral data such as vaccination status, mask usage, and social distancing adherence to forecast disease transmission and assess the impact of non-pharmaceutical interventions (NPIs). While such data provide valuable real-time insights, they are often subject to strategic misreporting, driven by individual incentives to avoid penalties, access benefits, or express distrust in public health authorities. To account for such human behavior, in this paper, we introduce a game-theoretic framework that models the interaction between the population and a public health authority as a signaling game. Individuals (senders) choose how to report their behaviors, while the public health authority (receiver) updates their epidemiological model(s) based on potentially distorted signals. Focusing on deception around masking and vaccination, we characterize analytically game equilibrium outcomes and evaluate the degree to which deception can be tolerated while maintaining epidemic control through policy interventions. Our results show that separating equilibria-with minimal deception-drive infections to near zero over time. Remarkably, even under pervasive dishonesty in pooling equilibria, well-designed sender and receiver strategies can still maintain effective epidemic control. This work advances the understanding of adversarial data in epidemiology and offers tools for designing more robust public health models in the presence of strategic user behavior. 

---
# AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization 

**Authors**: Mert Cemri, Shubham Agrawal, Akshat Gupta, Shu Liu, Audrey Cheng, Qiuyang Mang, Ashwin Naren, Lutfi Eren Erdogan, Koushik Sen, Matei Zaharia, Alex Dimakis, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2602.20133)  

**Abstract**: The paradigm of automated program generation is shifting from one-shot generation to inference-time search, where Large Language Models (LLMs) function as semantic mutation operators within evolutionary loops. While effective, these systems are currently governed by static schedules that fail to account for the non-stationary dynamics of the search process. This rigidity results in substantial computational waste, as resources are indiscriminately allocated to stagnating populations while promising frontiers remain under-exploited. We introduce AdaEvolve, a framework that reformulates LLM-driven evolution as a hierarchical adaptive optimization problem. AdaEvolve uses an "accumulated improvement signal" to unify decisions across three levels: Local Adaptation, which dynamically modulates the exploration intensity within a population of solution candidates; Global Adaptation, which routes the global resource budget via bandit-based scheduling across different solution candidate populations; and Meta-Guidance which generates novel solution tactics based on the previously generated solutions and their corresponding improvements when the progress stalls. We demonstrate that AdaEvolve consistently outperforms the open-sourced baselines across 185 different open-ended optimization problems including combinatorial, systems optimization and algorithm design problems. 

---
# To Reason or Not to: Selective Chain-of-Thought in Medical Question Answering 

**Authors**: Zaifu Zhan, Min Zeng, Shuang Zhou, Yiran Song, Xiaoyi Chen, Yu Hou, Yifan Wu, Yang Ruan, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20130)  

**Abstract**: Objective: To improve the efficiency of medical question answering (MedQA) with large language models (LLMs) by avoiding unnecessary reasoning while maintaining accuracy.
Methods: We propose Selective Chain-of-Thought (Selective CoT), an inference-time strategy that first predicts whether a question requires reasoning and generates a rationale only when needed. Two open-source LLMs (Llama-3.1-8B and Qwen-2.5-7B) were evaluated on four biomedical QA benchmarks-HeadQA, MedQA-USMLE, MedMCQA, and PubMedQA. Metrics included accuracy, total generated tokens, and inference time.
Results: Selective CoT reduced inference time by 13-45% and token usage by 8-47% with minimal accuracy loss ($\leq$4\%). In some model-task pairs, it achieved both higher accuracy and greater efficiency than standard CoT. Compared with fixed-length CoT, Selective CoT reached similar or superior accuracy at substantially lower computational cost.
Discussion: Selective CoT dynamically balances reasoning depth and efficiency by invoking explicit reasoning only when beneficial, reducing redundancy on recall-type questions while preserving interpretability.
Conclusion: Selective CoT provides a simple, model-agnostic, and cost-effective approach for medical QA, aligning reasoning effort with question complexity to enhance real-world deployability of LLM-based clinical systems. 

---
# NanoKnow: How to Know What Your Language Model Knows 

**Authors**: Lingwei Gu, Nour Jedidi, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.20122)  

**Abstract**: How do large language models (LLMs) know what they know? Answering this question has been difficult because pre-training data is often a "black box" -- unknown or inaccessible. The recent release of nanochat -- a family of small LLMs with fully open pre-training data -- addresses this as it provides a transparent view into where a model's parametric knowledge comes from. Towards the goal of understanding how knowledge is encoded by LLMs, we release NanoKnow, a benchmark dataset that partitions questions from Natural Questions and SQuAD into splits based on whether their answers are present in nanochat's pre-training corpus. Using these splits, we can now properly disentangle the sources of knowledge that LLMs rely on when producing an output. To demonstrate NanoKnow's utility, we conduct experiments using eight nanochat checkpoints. Our findings show: (1) closed-book accuracy is strongly influenced by answer frequency in the pre-training data, (2) providing external evidence can mitigate this frequency dependence, (3) even with external evidence, models are more accurate when answers were seen during pre-training, demonstrating that parametric and external knowledge are complementary, and (4) non-relevant information is harmful, with accuracy decreasing based on both the position and the number of non-relevant contexts. We release all NanoKnow artifacts at this https URL. 

---
# NovaPlan: Zero-Shot Long-Horizon Manipulation via Closed-Loop Video Language Planning 

**Authors**: Jiahui Fu, Junyu Nan, Lingfeng Sun, Hongyu Li, Jianing Qian, Jennifer L. Barry, Kris Kitani, George Konidaris  

**Link**: [PDF](https://arxiv.org/pdf/2602.20119)  

**Abstract**: Solving long-horizon tasks requires robots to integrate high-level semantic reasoning with low-level physical interaction. While vision-language models (VLMs) and video generation models can decompose tasks and imagine outcomes, they often lack the physical grounding necessary for real-world execution. We introduce NovaPlan, a hierarchical framework that unifies closed-loop VLM and video planning with geometrically grounded robot execution for zero-shot long-horizon manipulation. At the high level, a VLM planner decomposes tasks into sub-goals and monitors robot execution in a closed loop, enabling the system to recover from single-step failures through autonomous re-planning. To compute low-level robot actions, we extract and utilize both task-relevant object keypoints and human hand poses as kinematic priors from the generated videos, and employ a switching mechanism to choose the better one as a reference for robot actions, maintaining stable execution even under heavy occlusion or depth inaccuracy. We demonstrate the effectiveness of NovaPlan on three long-horizon tasks and the Functional Manipulation Benchmark (FMB). Our results show that NovaPlan can perform complex assembly tasks and exhibit dexterous error recovery behaviors without any prior demonstrations or training. Project page: this https URL 

---
# Benchmarking Unlearning for Vision Transformers 

**Authors**: Kairan Zhao, Iurie Luca, Peter Triantafillou  

**Link**: [PDF](https://arxiv.org/pdf/2602.20114)  

**Abstract**: Research in machine unlearning (MU) has gained strong momentum: MU is now widely regarded as a critical capability for building safe and fair AI. In parallel, research into transformer architectures for computer vision tasks has been highly successful: Increasingly, Vision Transformers (VTs) emerge as strong alternatives to CNNs. Yet, MU research for vision tasks has largely centered on CNNs, not VTs. While benchmarking MU efforts have addressed LLMs, diffusion models, and CNNs, none exist for VTs. This work is the first to attempt this, benchmarking MU algorithm performance in different VT families (ViT and Swin-T) and at different capacities. The work employs (i) different datasets, selected to assess the impacts of dataset scale and complexity; (ii) different MU algorithms, selected to represent fundamentally different approaches for MU; and (iii) both single-shot and continual unlearning protocols. Additionally, it focuses on benchmarking MU algorithms that leverage training data memorization, since leveraging memorization has been recently discovered to significantly improve the performance of previously SOTA algorithms. En route, the work characterizes how VTs memorize training data relative to CNNs, and assesses the impact of different memorization proxies on performance. The benchmark uses unified evaluation metrics that capture two complementary notions of forget quality along with accuracy on unseen (test) data and on retained data. Overall, this work offers a benchmarking basis, enabling reproducible, fair, and comprehensive comparisons of existing (and future) MU algorithms on VTs. And, for the first time, it sheds light on how well existing algorithms work in VT settings, establishing a promising reference performance baseline. 

---
# StyleStream: Real-Time Zero-Shot Voice Style Conversion 

**Authors**: Yisi Liu, Nicholas Lee, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2602.20113)  

**Abstract**: Voice style conversion aims to transform an input utterance to match a target speaker's timbre, accent, and emotion, with a central challenge being the disentanglement of linguistic content from style. While prior work has explored this problem, conversion quality remains limited, and real-time voice style conversion has not been addressed. We propose StyleStream, the first streamable zero-shot voice style conversion system that achieves state-of-the-art performance. StyleStream consists of two components: a Destylizer, which removes style attributes while preserving linguistic content, and a Stylizer, a diffusion transformer (DiT) that reintroduces target style conditioned on reference speech. Robust content-style disentanglement is enforced through text supervision and a highly constrained information bottleneck. This design enables a fully non-autoregressive architecture, achieving real-time voice style conversion with an end-to-end latency of 1 second. Samples and real-time demo: this https URL. 

---
# BarrierSteer: LLM Safety via Learning Barrier Steering 

**Authors**: Thanh Q. Tran, Arun Verma, Kiwan Wong, Bryan Kian Hsiang Low, Daniela Rus, Wei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2602.20102)  

**Abstract**: Despite the state-of-the-art performance of large language models (LLMs) across diverse tasks, their susceptibility to adversarial attacks and unsafe content generation remains a major obstacle to deployment, particularly in high-stakes settings. Addressing this challenge requires safety mechanisms that are both practically effective and supported by rigorous theory. We introduce BarrierSteer, a novel framework that formalizes response safety by embedding learned non-linear safety constraints directly into the model's latent representation space. BarrierSteer employs a steering mechanism based on Control Barrier Functions (CBFs) to efficiently detect and prevent unsafe response trajectories during inference with high precision. By enforcing multiple safety constraints through efficient constraint merging, without modifying the underlying LLM parameters, BarrierSteer preserves the model's original capabilities and performance. We provide theoretical results establishing that applying CBFs in latent space offers a principled and computationally efficient approach to enforcing safety. Our experiments across multiple models and datasets show that BarrierSteer substantially reduces adversarial success rates, decreases unsafe generations, and outperforms existing methods. 

---
# Transcending the Annotation Bottleneck: AI-Powered Discovery in Biology and Medicine 

**Authors**: Soumick Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2602.20100)  

**Abstract**: The dependence on expert annotation has long constituted the primary rate-limiting step in the application of artificial intelligence to biomedicine. While supervised learning drove the initial wave of clinical algorithms, a paradigm shift towards unsupervised and self-supervised learning (SSL) is currently unlocking the latent potential of biobank-scale datasets. By learning directly from the intrinsic structure of data - whether pixels in a magnetic resonance image (MRI), voxels in a volumetric scan, or tokens in a genomic sequence - these methods facilitate the discovery of novel phenotypes, the linkage of morphology to genetics, and the detection of anomalies without human bias. This article synthesises seminal and recent advances in "learning without labels," highlighting how unsupervised frameworks can derive heritable cardiac traits, predict spatial gene expression in histology, and detect pathologies with performance that rivals or exceeds supervised counterparts. 

---
# StructXLIP: Enhancing Vision-language Models with Multimodal Structural Cues 

**Authors**: Zanxi Ruan, Qiuyu Kong, Songqun Gao, Yiming Wang, Marco Cristani  

**Link**: [PDF](https://arxiv.org/pdf/2602.20089)  

**Abstract**: Edge-based representations are fundamental cues for visual understanding, a principle rooted in early vision research and still central today. We extend this principle to vision-language alignment, showing that isolating and aligning structural cues across modalities can greatly benefit fine-tuning on long, detail-rich captions, with a specific focus on improving cross-modal retrieval. We introduce StructXLIP, a fine-tuning alignment paradigm that extracts edge maps (e.g., Canny), treating them as proxies for the visual structure of an image, and filters the corresponding captions to emphasize structural cues, making them "structure-centric". Fine-tuning augments the standard alignment loss with three structure-centric losses: (i) aligning edge maps with structural text, (ii) matching local edge regions to textual chunks, and (iii) connecting edge maps to color images to prevent representation drift. From a theoretical standpoint, while standard CLIP maximizes the mutual information between visual and textual embeddings, StructXLIP additionally maximizes the mutual information between multimodal structural representations. This auxiliary optimization is intrinsically harder, guiding the model toward more robust and semantically stable minima, enhancing vision-language alignment. Beyond outperforming current competitors on cross-modal retrieval in both general and specialized domains, our method serves as a general boosting recipe that can be integrated into future approaches in a plug-and-play manner. Code and pretrained models are publicly available at: this https URL. 

---
# Descent-Guided Policy Gradient for Scalable Cooperative Multi-Agent Learning 

**Authors**: Shan Yang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20078)  

**Abstract**: Scaling cooperative multi-agent reinforcement learning (MARL) is fundamentally limited by cross-agent noise: when agents share a common reward, the actions of all $N$ agents jointly determine each agent's learning signal, so cross-agent noise grows with $N$. In the policy gradient setting, per-agent gradient estimate variance scales as $\Theta(N)$, yielding sample complexity $\mathcal{O}(N/\epsilon)$. We observe that many domains -- cloud computing, transportation, power systems -- have differentiable analytical models that prescribe efficient system states. In this work, we propose Descent-Guided Policy Gradient (DG-PG), a framework that constructs noise-free per-agent guidance gradients from these analytical models, decoupling each agent's gradient from the actions of all others. We prove that DG-PG reduces gradient variance from $\Theta(N)$ to $\mathcal{O}(1)$, preserves the equilibria of the cooperative game, and achieves agent-independent sample complexity $\mathcal{O}(1/\epsilon)$. On a heterogeneous cloud scheduling task with up to 200 agents, DG-PG converges within 10 episodes at every tested scale -- from $N=5$ to $N=200$ -- directly confirming the predicted scale-invariant complexity, while MAPPO and IPPO fail to converge under identical architectures. 

---
# Robust Taylor-Lagrange Control for Safety-Critical Systems 

**Authors**: Wei Xiao, Christos Cassandras, Anni Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.20076)  

**Abstract**: Solving safety-critical control problem has widely adopted the Control Barrier Function (CBF) method. However, the existence of a CBF is only a sufficient condition for system safety. The recently proposed Taylor-Lagrange Control (TLC) method addresses this limitation, but is vulnerable to the feasibility preservation problem (e.g., inter-sampling effect). In this paper, we propose a robust TLC (rTLC) method to address the feasibility preservation problem. Specifically, the rTLC method expands the safety function at an order higher than the relative degree of the function using Taylor's expansion with Lagrange remainder, which allows the control to explicitly show up at the current time instead of the future time in the TLC method. The rTLC method naturally addresses the feasibility preservation problem with only one hyper-parameter (the discretization time interval size during implementation), which is much less than its counterparts. Finally, we illustrate the effectiveness of the proposed rTLC method through an adaptive cruise control problem, and compare it with existing safety-critical control methods. 

---
# HeatPrompt: Zero-Shot Vision-Language Modeling of Urban Heat Demand from Satellite Images 

**Authors**: Kundan Thota, Xuanhao Mu, Thorsten Schlachter, Veit Hagenmeyer  

**Link**: [PDF](https://arxiv.org/pdf/2602.20066)  

**Abstract**: Accurate heat-demand maps play a crucial role in decarbonizing space heating, yet most municipalities lack detailed building-level data needed to calculate them. We introduce HeatPrompt, a zero-shot vision-language energy modeling framework that estimates annual heat demand using semantic features extracted from satellite images, basic Geographic Information System (GIS), and building-level features. We feed pretrained Large Vision Language Models (VLMs) with a domain-specific prompt to act as an energy planner and extract the visual attributes such as roof age, building density, etc, from the RGB satellite image that correspond to the thermal load. A Multi-Layer Perceptron (MLP) regressor trained on these captions shows an $R^2$ uplift of 93.7% and shrinks the mean absolute error (MAE) by 30% compared to the baseline model. Qualitative analysis shows that high-impact tokens align with high-demand zones, offering lightweight support for heat planning in data-scarce regions. 

---
# Multilingual Large Language Models do not comprehend all natural languages to equal degrees 

**Authors**: Natalia Moskvina, Raquel Montero, Masaya Yoshida, Ferdy Hubers, Paolo Morosi, Walid Irhaymi, Jin Yan, Tamara Serrano, Elena Pagliarini, Fritz Günther, Evelina Leivada  

**Link**: [PDF](https://arxiv.org/pdf/2602.20065)  

**Abstract**: Large Language Models (LLMs) play a critical role in how humans access information. While their core use relies on comprehending written requests, our understanding of this ability is currently limited, because most benchmarks evaluate LLMs in high-resource languages predominantly spoken by Western, Educated, Industrialised, Rich, and Democratic (WEIRD) communities. The default assumption is that English is the best-performing language for LLMs, while smaller, low-resource languages are linked to less reliable outputs, even in multilingual, state-of-the-art models. To track variation in the comprehension abilities of LLMs, we prompt 3 popular models on a language comprehension task across 12 languages, representing the Indo-European, Afro-Asiatic, Turkic, Sino-Tibetan, and Japonic language families. Our results suggest that the models exhibit remarkable linguistic accuracy across typologically diverse languages, yet they fall behind human baselines in all of them, albeit to different degrees. Contrary to what was expected, English is not the best-performing language, as it was systematically outperformed by several Romance languages, even lower-resource ones. We frame the results by discussing the role of several factors that drive LLM performance, such as tokenization, language distance from Spanish and English, size of training data, and data origin in high- vs. low-resource languages and WEIRD vs. non-WEIRD communities. 

---
# The LLMbda Calculus: AI Agents, Conversations, and Information Flow 

**Authors**: Zac Garby, Andrew D. Gordon, David Sands  

**Link**: [PDF](https://arxiv.org/pdf/2602.20064)  

**Abstract**: A conversation with a large language model (LLM) is a sequence of prompts and responses, with each response generated from the preceding conversation. AI agents build such conversations automatically: given an initial human prompt, a planner loop interleaves LLM calls with tool invocations and code execution. This tight coupling creates a new and poorly understood attack surface. A malicious prompt injected into a conversation can compromise later reasoning, trigger dangerous tool calls, or distort final outputs. Despite the centrality of such systems, we currently lack a principled semantic foundation for reasoning about their behaviour and safety. We address this gap by introducing an untyped call-by-value lambda calculus enriched with dynamic information-flow control and a small number of primitives for constructing prompt-response conversations. Our language includes a primitive that invokes an LLM: it serializes a value, sends it to the model as a prompt, and parses the response as a new term. This calculus faithfully represents planner loops and their vulnerabilities, including the mechanisms by which prompt injection alters subsequent computation. The semantics explicitly captures conversations, and so supports reasoning about defenses such as quarantined sub-conversations, isolation of generated code, and information-flow restrictions on what may influence an LLM call. A termination-insensitive noninterference theorem establishes integrity and confidentiality guarantees, demonstrating that a formal calculus can provide rigorous foundations for safe agentic programming. 

---
# AdaWorldPolicy: World-Model-Driven Diffusion Policy with Online Adaptive Learning for Robotic Manipulation 

**Authors**: Ge Yuan, Qiyuan Qiao, Jing Zhang, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20057)  

**Abstract**: Effective robotic manipulation requires policies that can anticipate physical outcomes and adapt to real-world environments. Effective robotic manipulation requires policies that can anticipate physical outcomes and adapt to real-world environments. In this work, we introduce a unified framework, World-Model-Driven Diffusion Policy with Online Adaptive Learning (AdaWorldPolicy) to enhance robotic manipulation under dynamic conditions with minimal human involvement. Our core insight is that world models provide strong supervision signals, enabling online adaptive learning in dynamic environments, which can be complemented by force-torque feedback to mitigate dynamic force shifts. Our AdaWorldPolicy integrates a world model, an action expert, and a force predictor-all implemented as interconnected Flow Matching Diffusion Transformers (DiT). They are interconnected via the multi-modal self-attention layers, enabling deep feature exchange for joint learning while preserving their distinct modularity characteristics. We further propose a novel Online Adaptive Learning (AdaOL) strategy that dynamically switches between an Action Generation mode and a Future Imagination mode to drive reactive updates across all three modules. This creates a powerful closed-loop mechanism that adapts to both visual and physical domain shifts with minimal overhead. Across a suite of simulated and real-robot benchmarks, our AdaWorldPolicy achieves state-of-the-art performance, with dynamical adaptive capacity to out-of-distribution scenarios. 

---
# To Move or Not to Move: Constraint-based Planning Enables Zero-Shot Generalization for Interactive Navigation 

**Authors**: Apoorva Vashisth, Manav Kulshrestha, Pranav Bakshi, Damon Conover, Guillaume Sartoretti, Aniket Bera  

**Link**: [PDF](https://arxiv.org/pdf/2602.20055)  

**Abstract**: Visual navigation typically assumes the existence of at least one obstacle-free path between start and goal, which must be discovered/planned by the robot. However, in real-world scenarios, such as home environments and warehouses, clutter can block all routes. Targeted at such cases, we introduce the Lifelong Interactive Navigation problem, where a mobile robot with manipulation abilities can move clutter to forge its own path to complete sequential object- placement tasks - each involving placing an given object (eg. Alarm clock, Pillow) onto a target object (eg. Dining table, Desk, Bed). To address this lifelong setting - where effects of environment changes accumulate and have long-term effects - we propose an LLM-driven, constraint-based planning framework with active perception. Our framework allows the LLM to reason over a structured scene graph of discovered objects and obstacles, deciding which object to move, where to place it, and where to look next to discover task-relevant information. This coupling of reasoning and active perception allows the agent to explore the regions expected to contribute to task completion rather than exhaustively mapping the environment. A standard motion planner then executes the corresponding navigate-pick-place, or detour sequence, ensuring reliable low-level control. Evaluated in physics-enabled ProcTHOR-10k simulator, our approach outperforms non-learning and learning-based baselines. We further demonstrate our approach qualitatively on real-world hardware. 

---
# SEAL-pose: Enhancing 3D Human Pose Estimation via a Learned Loss for Structural Consistency 

**Authors**: Yeonsung Kim, Junggeun Do, Seunguk Do, Sangmin Kim, Jaesik Park, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.20051)  

**Abstract**: 3D human pose estimation (HPE) is characterized by intricate local and global dependencies among joints. Conventional supervised losses are limited in capturing these correlations because they treat each joint independently. Previous studies have attempted to promote structural consistency through manually designed priors or rule-based constraints; however, these approaches typically require manual specification and are often non-differentiable, limiting their use as end-to-end training objectives. We propose SEAL-pose, a data-driven framework in which a learnable loss-net trains a pose-net by evaluating structural plausibility. Rather than relying on hand-crafted priors, our joint-graph-based design enables the loss-net to learn complex structural dependencies directly from data. Extensive experiments on three 3D HPE benchmarks with eight backbones show that SEAL-pose reduces per-joint errors and improves pose plausibility compared with the corresponding backbones across all settings. Beyond improving each backbone, SEAL-pose also outperforms models with explicit structural constraints, despite not enforcing any such constraints. Finally, we analyze the relationship between the loss-net and structural consistency, and evaluate SEAL-pose in cross-dataset and in-the-wild settings. 

---
# AgenticSum: An Agentic Inference-Time Framework for Faithful Clinical Text Summarization 

**Authors**: Fahmida Liza Piya, Rahmatollah Beheshti  

**Link**: [PDF](https://arxiv.org/pdf/2602.20040)  

**Abstract**: Large language models (LLMs) offer substantial promise for automating clinical text summarization, yet maintaining factual consistency remains challenging due to the length, noise, and heterogeneity of clinical documentation. We present AgenticSum, an inference-time, agentic framework that separates context selection, generation, verification, and targeted correction to reduce hallucinated content. The framework decomposes summarization into coordinated stages that compress task-relevant context, generate an initial draft, identify weakly supported spans using internal attention grounding signals, and selectively revise flagged content under supervisory control. We evaluate AgenticSum on two public datasets, using reference-based metrics, LLM-as-a-judge assessment, and human evaluation. Across various measures, AgenticSum demonstrates consistent improvements compared to vanilla LLMs and other strong baselines. Our results indicate that structured, agentic design with targeted correction offers an effective inference time solution to improve clinical note summarization using LLMs. 

---
# Descriptor: Dataset of Parasitoid Wasps and Associated Hymenoptera (DAPWH) 

**Authors**: Joao Manoel Herrera Pinheiro, Gabriela Do Nascimento Herrera, Luciana Bueno Dos Reis Fernandes, Alvaro Doria Dos Santos, Ricardo V. Godoy, Eduardo A. B. Almeida, Helena Carolina Onody, Marcelo Andrade Da Costa Vieira, Angelica Maria Penteado-Dias, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2602.20028)  

**Abstract**: Accurate taxonomic identification is the cornerstone of biodiversity monitoring and agricultural management, particularly for the hyper-diverse superfamily Ichneumonoidea. Comprising the families Ichneumonidae and Braconidae, these parasitoid wasps are ecologically critical for regulating insect populations, yet they remain one of the most taxonomically challenging groups due to their cryptic morphology and vast number of undescribed species. To address the scarcity of robust digital resources for these key groups, we present a curated image dataset designed to advance automated identification systems. The dataset contains 3,556 high-resolution images, primarily focused on Neotropical Ichneumonidae and Braconidae, while also including supplementary families such as Andrenidae, Apidae, Bethylidae, Chrysididae, Colletidae, Halictidae, Megachilidae, Pompilidae, and Vespidae to improve model robustness. Crucially, a subset of 1,739 images is annotated in COCO format, featuring multi-class bounding boxes for the full insect body, wing venation, and scale bars. This resource provides a foundation for developing computer vision models capable of identifying these families. 

---
# Learning Discriminative and Generalizable Anomaly Detector for Dynamic Graph with Limited Supervision 

**Authors**: Yuxing Tian, Yiyan Qi, Fengran Mo, Weixu Zhang, Jian Guo, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2602.20019)  

**Abstract**: Dynamic graph anomaly detection (DGAD) is critical for many real-world applications but remains challenging due to the scarcity of labeled anomalies. Existing methods are either unsupervised or semi-supervised: unsupervised methods avoid the need for labeled anomalies but often produce ambiguous boundary, whereas semi-supervised methods can overfit to the limited labeled anomalies and generalize poorly to unseen anomalies. To address this gap, we consider a largely underexplored problem in DGAD: learning a discriminative boundary from normal/unlabeled data, while leveraging limited labeled anomalies \textbf{when available} without sacrificing generalization to unseen anomalies. To this end, we propose an effective, generalizable, and model-agnostic framework with three main components: (i) residual representation encoding that capture deviations between current interactions and their historical context, providing anomaly-relevant signals; (ii) a restriction loss that constrain the normal representations within an interval bounded by two co-centered hyperspheres, ensuring consistent scales while keeping anomalies separable; (iii) a bi-boundary optimization strategy that learns a discriminative and robust boundary using the normal log-likelihood distribution modeled by a normalizing flow. Extensive experiments demonstrate the superiority of our framework across diverse evaluation settings. 

---
# A Secure and Private Distributed Bayesian Federated Learning Design 

**Authors**: Nuocheng Yang, Sihua Wang, Zhaohui Yang, Mingzhe Chen, Changchuan Yin, Kaibin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20003)  

**Abstract**: Distributed Federated Learning (DFL) enables decentralized model training across large-scale systems without a central parameter server. However, DFL faces three critical challenges: privacy leakage from honest-but-curious neighbors, slow convergence due to the lack of central coordination, and vulnerability to Byzantine adversaries aiming to degrade model accuracy. To address these issues, we propose a novel DFL framework that integrates Byzantine robustness, privacy preservation, and convergence acceleration. Within this framework, each device trains a local model using a Bayesian approach and independently selects an optimal subset of neighbors for posterior exchange. We formulate this neighbor selection as an optimization problem to minimize the global loss function under security and privacy constraints. Solving this problem is challenging because devices only possess partial network information, and the complex coupling between topology, security, and convergence remains unclear. To bridge this gap, we first analytically characterize the trade-offs between dynamic connectivity, Byzantine detection, privacy levels, and convergence speed. Leveraging these insights, we develop a fully distributed Graph Neural Network (GNN)-based Reinforcement Learning (RL) algorithm. This approach enables devices to make autonomous connection decisions based on local observations. Simulation results demonstrate that our method achieves superior robustness and efficiency with significantly lower overhead compared to traditional security and privacy schemes. 

---
# Contextual Safety Reasoning and Grounding for Open-World Robots 

**Authors**: Zachary Ravichadran, David Snyder, Alexander Robey, Hamed Hassani, Vijay Kumar, George J. Pappas  

**Link**: [PDF](https://arxiv.org/pdf/2602.19983)  

**Abstract**: Robots are increasingly operating in open-world environments where safe behavior depends on context: the same hallway may require different navigation strategies when crowded versus empty, or during an emergency versus normal operations. Traditional safety approaches enforce fixed constraints in user-specified contexts, limiting their ability to handle the open-ended contextual variability of real-world deployment. We address this gap via CORE, a safety framework that enables online contextual reasoning, grounding, and enforcement without prior knowledge of the environment (e.g., maps or safety specifications). CORE uses a vision-language model (VLM) to continuously reason about context-dependent safety rules directly from visual observations, grounds these rules in the physical environment, and enforces the resulting spatially-defined safe sets via control barrier functions. We provide probabilistic safety guarantees for CORE that account for perceptual uncertainty, and we demonstrate through simulation and real-world experiments that CORE enforces contextually appropriate behavior in unseen environments, significantly outperforming prior semantic safety methods that lack online contextual reasoning. Ablation studies validate our theoretical guarantees and underscore the importance of both VLM-based reasoning and spatial grounding for enforcing contextual safety in novel settings. We provide additional resources at this https URL. 

---
# ReAttn: Improving Attention-based Re-ranking via Attention Re-weighting 

**Authors**: Yuxing Tian, Fengran Mo, Weixu Zhang, Yiyan Qi, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2602.19969)  

**Abstract**: The strong capabilities of recent Large Language Models (LLMs) have made them highly effective for zero-shot re-ranking task. Attention-based re-ranking methods, which derive relevance scores directly from attention weights, offer an efficient and interpretable alternative to generation-based re-ranking methods. However, they still face two major limitations. First, attention signals are highly concentrated a small subset of tokens within a few documents, making others indistinguishable. Second, attention often overemphasizes phrases lexically similar to the query, yielding biased rankings that irrelevant documents with mere lexical resemblance are regarded as relevant. In this paper, we propose \textbf{ReAttn}, a post-hoc re-weighting strategy for attention-based re-ranking methods. It first compute the cross-document IDF weighting to down-weight attention on query-overlapping tokens that frequently appear across the candidate documents, reducing lexical bias and emphasizing distinctive terms. It then employs entropy-based regularization to mitigate over-concentrated attention, encouraging a more balanced distribution across informative tokens. Both adjustments operate directly on existing attention weights without additional training or supervision. Extensive experiments demonstrate the effectiveness of our method. 

---
# On the Equivalence of Random Network Distillation, Deep Ensembles, and Bayesian Inference 

**Authors**: Moritz A. Zanger, Yijun Wu, Pascal R. Van der Vaart, Wendelin Böhmer, Matthijs T. J. Spaan  

**Link**: [PDF](https://arxiv.org/pdf/2602.19964)  

**Abstract**: Uncertainty quantification is central to safe and efficient deployments of deep learning models, yet many computationally practical methods lack lacking rigorous theoretical motivation. Random network distillation (RND) is a lightweight technique that measures novelty via prediction errors against a fixed random target. While empirically effective, it has remained unclear what uncertainties RND measures and how its estimates relate to other approaches, e.g. Bayesian inference or deep ensembles. This paper establishes these missing theoretical connections by analyzing RND within the neural tangent kernel framework in the limit of infinite network width. Our analysis reveals two central findings in this limit: (1) The uncertainty signal from RND -- its squared self-predictive error -- is equivalent to the predictive variance of a deep ensemble. (2) By constructing a specific RND target function, we show that the RND error distribution can be made to mirror the centered posterior predictive distribution of Bayesian inference with wide neural networks. Based on this equivalence, we moreover devise a posterior sampling algorithm that generates i.i.d. samples from an exact Bayesian posterior predictive distribution using this modified \textit{Bayesian RND} model. Collectively, our findings provide a unified theoretical perspective that places RND within the principled frameworks of deep ensembles and Bayesian inference, and offer new avenues for efficient yet theoretically grounded uncertainty quantification methods. 

---
# Assessing Risks of Large Language Models in Mental Health Support: A Framework for Automated Clinical AI Red Teaming 

**Authors**: Ian Steenstra, Paola Pedrelli, Weiyan Shi, Stacy Marsella, Timothy W. Bickmore  

**Link**: [PDF](https://arxiv.org/pdf/2602.19948)  

**Abstract**: Large Language Models (LLMs) are increasingly utilized for mental health support; however, current safety benchmarks often fail to detect the complex, longitudinal risks inherent in therapeutic dialogue. We introduce an evaluation framework that pairs AI psychotherapists with simulated patient agents equipped with dynamic cognitive-affective models and assesses therapy session simulations against a comprehensive quality of care and risk ontology. We apply this framework to a high-impact test case, Alcohol Use Disorder, evaluating six AI agents (including ChatGPT, Gemini, and this http URL) against a clinically-validated cohort of 15 patient personas representing diverse clinical phenotypes.
Our large-scale simulation (N=369 sessions) reveals critical safety gaps in the use of AI for mental health support. We identify specific iatrogenic risks, including the validation of patient delusions ("AI Psychosis") and failure to de-escalate suicide risk. Finally, we validate an interactive data visualization dashboard with diverse stakeholders, including AI engineers and red teamers, mental health professionals, and policy experts (N=9), demonstrating that this framework effectively enables stakeholders to audit the "black box" of AI psychotherapy. These findings underscore the critical safety risks of AI-provided mental health support and the necessity of simulation-based clinical red teaming before deployment. 

---
# When Pretty Isn't Useful: Investigating Why Modern Text-to-Image Models Fail as Reliable Training Data Generators 

**Authors**: Krzysztof Adamkiewicz, Brian Moser, Stanislav Frolov, Tobias Christian Nauen, Federico Raue, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2602.19946)  

**Abstract**: Recent text-to-image (T2I) diffusion models produce visually stunning images and demonstrate excellent prompt following. But do they perform well as synthetic vision data generators? In this work, we revisit the promise of synthetic data as a scalable substitute for real training sets and uncover a surprising performance regression. We generate large-scale synthetic datasets using state-of-the-art T2I models released between 2022 and 2025, train standard classifiers solely on this synthetic data, and evaluate them on real test data. Despite observable advances in visual fidelity and prompt adherence, classification accuracy on real test data consistently declines with newer T2I models as training data generators. Our analysis reveals a hidden trend: These models collapse to a narrow, aesthetic-centric distribution that undermines diversity and label-image alignment. Overall, our findings challenge a growing assumption in vision research, namely that progress in generative realism implies progress in data realism. We thus highlight an urgent need to rethink the capabilities of modern T2I models as reliable training data generators. 

---
# DP-FedAdamW: An Efficient Optimizer for Differentially Private Federated Large Models 

**Authors**: Jin Liu, Yinbin Miao, Ning Xi, Junkang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19945)  

**Abstract**: Balancing convergence efficiency and robustness under Differential Privacy (DP) is a central challenge in Federated Learning (FL). While AdamW accelerates training and fine-tuning in large-scale models, we find that directly applying it to Differentially Private FL (DPFL) suffers from three major issues: (i) data heterogeneity and privacy noise jointly amplify the variance of second-moment estimator, (ii) DP perturbations bias the second-moment estimator, and (iii) DP amplify AdamW sensitivity to local overfitting, worsening client drift. We propose DP-FedAdamW, the first AdamW-based optimizer for DPFL. It restores AdamW under DP by stabilizing second-moment variance, removing DP-induced bias, and aligning local updates to the global descent to curb client drift. Theoretically, we establish an unbiased second-moment estimator and prove a linearly accelerated convergence rate without any heterogeneity assumption, while providing tighter $(\varepsilon,\delta)$-DP guarantees. Our empirical results demonstrate the effectiveness of DP-FedAdamW across language and vision Transformers and ResNet-18. On Tiny-ImageNet (Swin-Base, $\varepsilon=1$), DP-FedAdamW outperforms the state-of-the-art (SOTA) by 5.83\%. The code is available in Appendix. 

---
# Rethinking LoRA for Privacy-Preserving Federated Learning in Large Models 

**Authors**: Jin Liu, Yinbin Miao, Ning Xi, Junkang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19926)  

**Abstract**: Fine-tuning large vision models (LVMs) and large language models (LLMs) under differentially private federated learning (DPFL) is hindered by a fundamental privacy-utility trade-off. Low-Rank Adaptation (LoRA), a promising parameter-efficient fine-tuning (PEFT) method, reduces computational and communication costs by introducing two trainable low-rank matrices while freezing pre-trained weights. However, directly applying LoRA in DPFL settings leads to performance degradation, especially in LVMs. Our analysis reveals three previously underexplored challenges: (1) gradient coupling caused by the simultaneous update of two asymmetric low-rank matrices, (2) compounded noise amplification under differential privacy, and (3) sharpness of the global aggregated model in the parameter space. To address these issues, we propose LA-LoRA (\textbf{L}ocal \textbf{A}lternating \textbf{LoRA}), a novel approach that decouples gradient interactions and aligns update directions across clients to enhance robustness under stringent privacy constraints. Theoretically, LA-LoRA strengthens convergence guarantees in noisy federated environments. Extensive experiments demonstrate that LA-LoRA achieves state-of-the-art (SOTA) performance on Swin Transformer and RoBERTa models, showcasing robustness to DP noise and broad applicability across both LVMs and LLMs. For example, when fine-tuning the Swin-B model on the Tiny-ImageNet dataset under a strict privacy budget ($\epsilon = 1$), LA-LoRA outperforms the best baseline, RoLoRA, by 16.83\% in test accuracy. Code is provided in \repolink. 

---
# Make Some Noise: Unsupervised Remote Sensing Change Detection Using Latent Space Perturbations 

**Authors**: Blaž Rolih, Matic Fučka, Filip Wolf, Luka Čehovin Zajc  

**Link**: [PDF](https://arxiv.org/pdf/2602.19881)  

**Abstract**: Unsupervised change detection (UCD) in remote sensing aims to localise semantic changes between two images of the same region without relying on labelled data during training. Most recent approaches rely either on frozen foundation models in a training-free manner or on training with synthetic changes generated in pixel space. Both strategies inherently rely on predefined assumptions about change types, typically introduced through handcrafted rules, external datasets, or auxiliary generative models. Due to these assumptions, such methods fail to generalise beyond a few change types, limiting their real-world usage, especially in rare or complex scenarios. To address this, we propose MaSoN (Make Some Noise), an end-to-end UCD framework that synthesises diverse changes directly in the latent feature space during training. It generates changes that are dynamically estimated using feature statistics of target data, enabling diverse yet data-driven variation aligned with the target domain. It also easily extends to new modalities, such as SAR. MaSoN generalises strongly across diverse change types and achieves state-of-the-art performance on five benchmarks, improving the average F1 score by 14.1 percentage points. Project page: this https URL 

---
# GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery 

**Authors**: Jizhou Han, Chenhao Ding, SongLin Dong, Yuhang He, Shaokun Wang, Qiang Wang, Yihong Gong  

**Link**: [PDF](https://arxiv.org/pdf/2602.19872)  

**Abstract**: Continual Generalized Category Discovery (C-GCD) requires identifying novel classes from unlabeled data while retaining knowledge of known classes over time. Existing methods typically update classifier weights dynamically, resulting in forgetting and inconsistent feature alignment. We propose GOAL, a unified framework that introduces a fixed Equiangular Tight Frame (ETF) classifier to impose a consistent geometric structure throughout learning. GOAL conducts supervised alignment for labeled samples and confidence-guided alignment for novel samples, enabling stable integration of new classes without disrupting old ones. Experiments on four benchmarks show that GOAL outperforms the prior method Happy, reducing forgetting by 16.1% and boosting novel class discovery by 3.2%, establishing a strong solution for long-horizon continual discovery. 

---
# LLM-enabled Applications Require System-Level Threat Monitoring 

**Authors**: Yedi Zhang, Haoyu Wang, Xianglin Yang, Jin Song Dong, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.19844)  

**Abstract**: LLM-enabled applications are rapidly reshaping the software ecosystem by using large language models as core reasoning components for complex task execution. This paradigm shift, however, introduces fundamentally new reliability challenges and significantly expands the security attack surface, due to the non-deterministic, learning-driven, and difficult-to-verify nature of LLM behavior. In light of these emerging and unavoidable safety challenges, we argue that such risks should be treated as expected operational conditions rather than exceptional events, necessitating a dedicated incident-response perspective. Consequently, the primary barrier to trustworthy deployment is not further improving model capability but establishing system-level threat monitoring mechanisms that can detect and contextualize security-relevant anomalies after deployment -- an aspect largely underexplored beyond testing or guardrail-based defenses. Accordingly, this position paper advocates systematic and comprehensive monitoring of security threats in LLM-enabled applications as a prerequisite for reliable operation and a foundation for dedicated incident-response frameworks. 

---
# MAS-FIRE: Fault Injection and Reliability Evaluation for LLM-Based Multi-Agent Systems 

**Authors**: Jin Jia, Zhiling Deng, Zhuangbin Chen, Yingqi Wang, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.19843)  

**Abstract**: As LLM-based Multi-Agent Systems (MAS) are increasingly deployed for complex tasks, ensuring their reliability has become a pressing challenge. Since MAS coordinate through unstructured natural language rather than rigid protocols, they are prone to semantic failures (e.g., hallucinations, misinterpreted instructions, and reasoning drift) that propagate silently without raising runtime exceptions. Prevailing evaluation approaches, which measure only end-to-end task success, offer limited insight into how these failures arise or how effectively agents recover from them. To bridge this gap, we propose MAS-FIRE, a systematic framework for fault injection and reliability evaluation of MAS. We define a taxonomy of 15 fault types covering intra-agent cognitive errors and inter-agent coordination failures, and inject them via three non-invasive mechanisms: prompt modification, response rewriting, and message routing manipulation. Applying MAS-FIRE to three representative MAS architectures, we uncover a rich set of fault-tolerant behaviors that we organize into four tiers: mechanism, rule, prompt, and reasoning. This tiered view enables fine-grained diagnosis of where and why systems succeed or fail. Our findings reveal that stronger foundation models do not uniformly improve robustness. We further show that architectural topology plays an equally decisive role, with iterative, closed-loop designs neutralizing over 40% of faults that cause catastrophic collapse in linear workflows. MAS-FIRE provides the process-level observability and actionable guidance needed to systematically improve multi-agent systems. 

---
# Efficient endometrial carcinoma screening via cross-modal synthesis and gradient distillation 

**Authors**: Dongjing Shan, Yamei Luo, Jiqing Xuan, Lu Huang, Jin Li, Mengchu Yang, Zeyu Chen, Fajin Lv, Yong Tang, Chunxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.19822)  

**Abstract**: Early detection of myometrial invasion is critical for the staging and life-saving management of endometrial carcinoma (EC), a prevalent global malignancy. Transvaginal ultrasound serves as the primary, accessible screening modality in resource-constrained primary care settings; however, its diagnostic reliability is severely hindered by low tissue contrast, high operator dependence, and a pronounced scarcity of positive pathological samples. Existing artificial intelligence solutions struggle to overcome this severe class imbalance and the subtle imaging features of invasion, particularly under the strict computational limits of primary care clinics. Here we present an automated, highly efficient two-stage deep learning framework that resolves both data and computational bottlenecks in EC screening. To mitigate pathological data scarcity, we develop a structure-guided cross-modal generation network that synthesizes diverse, high-fidelity ultrasound images from unpaired magnetic resonance imaging (MRI) data, strictly preserving clinically essential anatomical junctions. Furthermore, we introduce a lightweight screening network utilizing gradient distillation, which transfers discriminative knowledge from a high-capacity teacher model to dynamically guide sparse attention towards task-critical regions. Evaluated on a large, multicenter cohort of 7,951 participants, our model achieves a sensitivity of 99.5\%, a specificity of 97.2\%, and an area under the curve of 0.987 at a minimal computational cost (0.289 GFLOPs), substantially outperforming the average diagnostic accuracy of expert sonographers. Our approach demonstrates that combining cross-modal synthetic augmentation with knowledge-driven efficient modeling can democratize expert-level, real-time cancer screening for resource-constrained primary care settings. 

---
# SafePickle: Robust and Generic ML Detection of Malicious Pickle-based ML Models 

**Authors**: Hillel Ohayon, Daniel Gilkarov, Ran Dubin  

**Link**: [PDF](https://arxiv.org/pdf/2602.19818)  

**Abstract**: Model repositories such as Hugging Face increasingly distribute machine learning artifacts serialized with Python's pickle format, exposing users to remote code execution (RCE) risks during model loading. Recent defenses, such as PickleBall, rely on per-library policy synthesis that requires complex system setups and verified benign models, which limits scalability and generalization. In this work, we propose a lightweight, machine-learning-based scanner that detects malicious Pickle-based files without policy generation or code instrumentation. Our approach statically extracts structural and semantic features from Pickle bytecode and applies supervised and unsupervised models to classify files as benign or malicious. We construct and release a labeled dataset of 727 Pickle-based files from Hugging Face and evaluate our models on four datasets: our own, PickleBall (out-of-distribution), Hide-and-Seek (9 advanced evasive malicious models), and synthetic joblib files. Our method achieves 90.01% F1-score compared with 7.23%-62.75% achieved by the SOTA scanners (Modelscan, Fickling, ClamAV, VirusTotal) on our dataset. Furthermore, on the PickleBall data (OOD), it achieves 81.22% F1-score compared with 76.09% achieved by the PickleBall method, while remaining fully library-agnostic. Finally, we show that our method is the only one to correctly parse and classify 9/9 evasive Hide-and-Seek malicious models specially crafted to evade scanners. This demonstrates that data-driven detection can effectively and generically mitigate Pickle-based model file attacks. 

---
# Depth-Structured Music Recurrence: Budgeted Recurrent Attention for Full-Piece Symbolic Music Modeling 

**Authors**: Yungang Yi  

**Link**: [PDF](https://arxiv.org/pdf/2602.19816)  

**Abstract**: Long-context modeling is essential for symbolic music generation, since motif repetition and developmental variation can span thousands of musical events. However, practical composition and performance workflows frequently rely on resource-limited devices (e.g., electronic instruments and portable computers), making heavy memory and attention computation difficult to deploy. We introduce Depth-Structured Music Recurrence (DSMR), a recurrent long-context Transformer for full-piece symbolic music modeling that extends context beyond fixed-length excerpts via segment-level recurrence with detached cross-segment states, featuring a layer-wise memory-horizon schedule that budgets recurrent KV states across depth. DSMR is trained in a single left-to-right pass over each complete composition, akin to how a musician experiences it from beginning to end, while carrying recurrent cross-segment states forward. Within this recurrent framework, we systematically study how depth-wise horizon allocations affect optimization, best-checkpoint perplexity, and efficiency. By allocating different history-window lengths across layers while keeping the total recurrent-state budget fixed, DSMR creates depth-dependent temporal receptive fields within a recurrent attention stack without reducing compute depth. Our main instantiation is a two-scale DSMR schedule that allocates long history windows to lower layers and a uniform short window to the remaining layers. Experiments on the piano performance dataset MAESTRO demonstrate that two-scale DSMR provides a practical quality--efficiency recipe for full-length long-context symbolic music modeling with recurrent attention under limited computational resources. 

---
# Decision MetaMamba: Enhancing Selective SSM in Offline RL with Heterogeneous Sequence Mixing 

**Authors**: Wall Kim, Chaeyoung Song, Hanul Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.19805)  

**Abstract**: Mamba-based models have drawn much attention in offline RL. However, their selective mechanism often detrimental when key steps in RL sequences are omitted. To address these issues, we propose a simple yet effective structure, called Decision MetaMamba (DMM), which replaces Mamba's token mixer with a dense layer-based sequence mixer and modifies positional structure to preserve local information. By performing sequence mixing that considers all channels simultaneously before Mamba, DMM prevents information loss due to selective scanning and residual gating. Extensive experiments demonstrate that our DMM delivers the state-of-the-art performance across diverse RL tasks. Furthermore, DMM achieves these results with a compact parameter footprint, demonstrating strong potential for real-world applications. 

---
# The Climate Change Knowledge Graph: Supporting Climate Services 

**Authors**: Miguel Ceriani, Fiorela Ciroku, Alessandro Russo, Massimiliano Schembri, Fai Fung, Neha Mittal, Vito Trianni, Andrea Giovanni Nuzzolese  

**Link**: [PDF](https://arxiv.org/pdf/2602.19786)  

**Abstract**: Climate change impacts a broad spectrum of human resources and activities, necessitating the use of climate models to project long-term effects and inform mitigation and adaptation strategies. These models generate multiple datasets by running simulations across various scenarios and configurations, thereby covering a range of potential future outcomes. Currently, researchers rely on traditional search interfaces and APIs to retrieve such datasets, often piecing together information from metadata and community vocabularies. The Climate Change Knowledge Graph is designed to address these challenges by integrating diverse data sources related to climate simulations into a coherent and interoperable knowledge graph. This innovative resource allows for executing complex queries involving climate models, simulations, variables, spatio-temporal domains, and granularities. Developed with input from domain experts, the knowledge graph and its underlying ontology are published with open access license and provide a comprehensive framework that enhances the exploration of climate data, facilitating more informed decision-making in addressing climate change issues. 

---
# The Confusion is Real: GRAPHIC - A Network Science Approach to Confusion Matrices in Deep Learning 

**Authors**: Johanna S. Fröhlich, Bastian Heinlein, Jan U. Claar, Hans Rosenberger, Vasileios Belagiannis, Ralf R. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2602.19770)  

**Abstract**: Explainable artificial intelligence has emerged as a promising field of research to address reliability concerns in artificial intelligence. Despite significant progress in explainable artificial intelligence, few methods provide a systematic way to visualize and understand how classes are confused and how their relationships evolve as training progresses. In this work, we present GRAPHIC, an architecture-agnostic approach that analyzes neural networks on a class level. It leverages confusion matrices derived from intermediate layers using linear classifiers. We interpret these as adjacency matrices of directed graphs, allowing tools from network science to visualize and quantify learning dynamics across training epochs and intermediate layers. GRAPHIC provides insights into linear class separability, dataset issues, and architectural behavior, revealing, for example, similarities between flatfish and man and labeling ambiguities validated in a human study. In summary, by uncovering real confusions, GRAPHIC offers new perspectives on how neural networks learn. The code is available at this https URL. 

---
# Hexagon-MLIR: An AI Compilation Stack For Qualcomm's Neural Processing Units (NPUs) 

**Authors**: Mohammed Javed Absar, Muthu Baskaran, Abhikrant Sharma, Abhilash Bhandari, Ankit Aggarwal, Arun Rangasamy, Dibyendu Das, Fateme Hosseini, Franck Slama, Iulian Brumar, Jyotsna Verma, Krishnaprasad Bindumadhavan, Mitesh Kothari, Mohit Gupta, Ravishankar Kolachana, Richard Lethin, Samarth Narang, Sanjay Motilal Ladwa, Shalini Jain, Snigdha Suresh Dalvi, Tasmia Rahman, Venkat Rasagna Reddy Komatireddy, Vivek Vasudevbhai Pandya, Xiyue Shi, Zachary Zipper  

**Link**: [PDF](https://arxiv.org/pdf/2602.19762)  

**Abstract**: In this paper, we present Hexagon-MLIR,an open-source compilation stack that targets Qualcomm Hexagon Neural Processing Unit (NPU) and provides unified support for lowering Triton kernels and PyTorch models . Built using the MLIR framework, our compiler applies a structured sequence of passes to exploit NPU architectural features to accelerate AI workloads. It enables faster deployment of new Triton kernels (hand-written or subgraphs from PyTorch 2.0), for our target by providing automated compilation from kernel to binary. By ingesting Triton kernels, we generate mega-kernels that maximize data locality in the NPU's Tightly Coupled Memory (TCM), reducing the bandwidth bottlenecks inherent in library-based approaches. This initiative complements our commercial toolchains by providing developers with an open-source MLIR-based compilation stack that gives them a path to advance AI compilation capabilities through a more flexible approach. Hexagon-MLIR is a work-in-progress, and we are continuing to add many more optimizations and capabilities in this effort. 

---
# Carbon-Aware Governance Gates: An Architecture for Sustainable GenAI Development 

**Authors**: Mateen A. Abbasi, Tommi J. Mikkonen, Petri J. Ihantola, Muhammad Waseem, Pekka Abrahamsson, Niko K. Mäkitalo  

**Link**: [PDF](https://arxiv.org/pdf/2602.19718)  

**Abstract**: The rapid adoption of Generative AI (GenAI) in the software development life cycle (SDLC) increases computational demand, which can raise the carbon footprint of development activities. At the same time, organizations are increasingly embedding governance mechanisms into GenAI-assisted development to support trust, transparency, and accountability. However, these governance mechanisms introduce additional computational workloads, including repeated inference, regeneration cycles, and expanded validation pipelines, increasing energy use and the carbon footprint of GenAI-assisted development. This paper proposes Carbon-Aware Governance Gates (CAGG), an architectural extension that embeds carbon budgets, energy provenance, and sustainability-aware validation orchestration into human-AI governance layers. CAGG comprises three components: (i) an Energy and Carbon Provenance Ledger, (ii) a Carbon Budget Manager, and (iii) a Green Validation Orchestrator, operationalized through governance policies and reusable design patterns. 

---
# DReX: An Explainable Deep Learning-based Multimodal Recommendation Framework 

**Authors**: Adamya Shyam, Venkateswara Rao Kagita, Bharti Rana, Vikas Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2602.19702)  

**Abstract**: Multimodal recommender systems leverage diverse data sources, such as user interactions, content features, and contextual information, to address challenges like cold-start and data sparsity. However, existing methods often suffer from one or more key limitations: processing different modalities in isolation, requiring complete multimodal data for each interaction during training, or independent learning of user and item representations. These factors contribute to increased complexity and potential misalignment between user and item embeddings. To address these challenges, we propose DReX, a unified multimodal recommendation framework that incrementally refines user and item representations by leveraging interaction-level features from multimodal feedback. Our model employs gated recurrent units to selectively integrate these fine-grained features into global representations. This incremental update mechanism provides three key advantages: (1) simultaneous modeling of both nuanced interaction details and broader preference patterns, (2) eliminates the need for separate user and item feature extraction processes, leading to enhanced alignment in their learned representation, and (3) inherent robustness to varying or missing modalities. We evaluate the performance of the proposed approach on three real-world datasets containing reviews and ratings as interaction modalities. By considering review text as a modality, our approach automatically generates interpretable keyword profiles for both users and items, which supplement the recommendation process with interpretable preference indicators. Experiment results demonstrate that our approach outperforms state-of-the-art methods across all evaluated datasets. 

---
# Iconographic Classification and Content-Based Recommendation for Digitized Artworks 

**Authors**: Krzysztof Kutt, Maciej Baczyński  

**Link**: [PDF](https://arxiv.org/pdf/2602.19698)  

**Abstract**: We present a proof-of-concept system that automates iconographic classification and content-based recommendation of digitized artworks using the Iconclass vocabulary and selected artificial intelligence methods. The prototype implements a four-stage workflow for classification and recommendation, which integrates YOLOv8 object detection with algorithmic mappings to Iconclass codes, rule-based inference for abstract meanings, and three complementary recommenders (hierarchical proximity, IDF-weighted overlap, and Jaccard similarity). Although more engineering is still needed, the evaluation demonstrates the potential of this solution: Iconclass-aware computer vision and recommendation methods can accelerate cataloging and enhance navigation in large heritage repositories. The key insight is to let computer vision propose visible elements and to use symbolic structures (Iconclass hierarchy) to reach meaning. 

---
# PerturbDiff: Functional Diffusion for Single-Cell Perturbation Modeling 

**Authors**: Xinyu Yuan, Xixian Liu, Ya Shi Zhang, Zuobai Zhang, Hongyu Guo, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.19685)  

**Abstract**: Building Virtual Cells that can accurately simulate cellular responses to perturbations is a long-standing goal in systems biology. A fundamental challenge is that high-throughput single-cell sequencing is destructive: the same cell cannot be observed both before and after a perturbation. Thus, perturbation prediction requires mapping unpaired control and perturbed populations. Existing models address this by learning maps between distributions, but typically assume a single fixed response distribution when conditioned on observed cellular context (e.g., cell type) and the perturbation type. In reality, responses vary systematically due to unobservable latent factors such as microenvironmental fluctuations and complex batch effects, forming a manifold of possible distributions for the same observed conditions. To account for this variability, we introduce PerturbDiff, which shifts modeling from individual cells to entire distributions. By embedding distributions as points in a Hilbert space, we define a diffusion-based generative process operating directly over probability distributions. This allows PerturbDiff to capture population-level response shifts across hidden factors. Benchmarks on established datasets show that PerturbDiff achieves state-of-the-art performance in single-cell response prediction and generalizes substantially better to unseen perturbations. See our project page (this https URL), where code and data will be made publicly available (this https URL). 

---
# TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures 

**Authors**: Hyeongjin Nam, Daniel Sungho Jung, Kyoung Mu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.19679)  

**Abstract**: Joint reconstruction of 3D human and object from a single image is an active research area, with pivotal applications in robotics and digital content creation. Despite recent advances, existing approaches suffer from two fundamental limitations. First, their reconstructions rely heavily on physical contact information, which inherently cannot capture non-contact human-object interactions, such as gazing at or pointing toward an object. Second, the reconstruction process is primarily driven by local geometric proximity, neglecting the human and object appearances that provide global context crucial for understanding holistic interactions. To address these issues, we introduce TeHOR, a framework built upon two core designs. First, beyond contact information, our framework leverages text descriptions of human-object interactions to enforce semantic alignment between the 3D reconstruction and its textual cues, enabling reasoning over a wider spectrum of interactions, including non-contact cases. Second, we incorporate appearance cues of the 3D human and object into the alignment process to capture holistic contextual information, thereby ensuring visually plausible reconstructions. As a result, our framework produces accurate and semantically coherent reconstructions, achieving state-of-the-art performance. 

---
# Continuous Telemonitoring of Heart Failure using Personalised Speech Dynamics 

**Authors**: Yue Pan, Xingyao Wang, Hanyue Zhang, Liwei Liu, Changxin Li, Gang Yang, Rong Sheng, Yili Xia, Ming Chu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19674)  

**Abstract**: Remote monitoring of heart failure (HF) via speech signals provides a non-invasive and cost-effective solution for long-term patient management. However, substantial inter-individual heterogeneity in vocal characteristics often limits the accuracy of traditional cross-sectional classification models. To address this, we propose a Longitudinal Intra-Patient Tracking (LIPT) scheme designed to capture the trajectory of relative symptomatic changes within individuals. Central to this framework is a Personalised Sequential Encoder (PSE), which transforms longitudinal speech recordings into context-aware latent representations. By incorporating historical data at each timestamp, the PSE facilitates a holistic assessment of the clinical trajectory rather than modelling discrete visits independently. Experimental results from a cohort of 225 patients demonstrate that the LIPT paradigm significantly outperforms the classic cross-sectional approaches, achieving a recognition accuracy of 99.7% for clinical status transitions. The model's high sensitivity was further corroborated by additional follow-up data, confirming its efficacy in predicting HF deterioration and its potential to secure patient safety in remote, home-based settings. Furthermore, this work addresses the gap in existing literature by providing a comprehensive analysis of different speech task designs and acoustic features. Taken together, the superior performance of the LIPT framework and PSE architecture validates their readiness for integration into long-term telemonitoring systems, offering a scalable solution for remote heart failure management. 

---
# Representation Stability in a Minimal Continual Learning Agent 

**Authors**: Vishnu Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2602.19655)  

**Abstract**: Continual learning systems are increasingly deployed in environments where retraining or reset is infeasible, yet many approaches emphasize task performance rather than the evolution of internal representations over time. In this work, we study a minimal continual learning agent designed to isolate representational dynamics from architectural complexity and optimization objectives. The agent maintains a persistent state vector across executions and incrementally updates it as new textual data is introduced. We quantify representational change using cosine similarity between successive normalized state vectors and define a stability metric over time intervals. Longitudinal experiments across eight executions reveal a transition from an initial plastic regime to a stable representational regime under consistent input. A deliberately introduced semantic perturbation produces a bounded decrease in similarity, followed by recovery and restabilization under subsequent coherent input. These results demonstrate that meaningful stability plasticity tradeoffs can emerge in a minimal, stateful learning system without explicit regularization, replay, or architectural complexity. The work establishes a transparent empirical baseline for studying representational accumulation and adaptation in continual learning systems. 

---
# NEXUS : A compact neural architecture for high-resolution spatiotemporal air quality forecasting in Delhi Nationa Capital Region 

**Authors**: Rampunit Kumar, Aditya Maheshwari  

**Link**: [PDF](https://arxiv.org/pdf/2602.19654)  

**Abstract**: Urban air pollution in megacities poses critical public health challenges, particularly in Delhi National Capital Region (NCR) where severe degradation affects millions. We present NEXUS (Neural Extraction and Unified Spatiotemporal) architecture for forecasting carbon monoxide, nitrogen oxide, and sulfur dioxide. Working with four years (2018--2021) of atmospheric data across sixteen spatial grids, NEXUS achieves R$^2$ exceeding 0.94 for CO, 0.91 for NO, and 0.95 for SO$_2$ using merely 18,748 parameters -- substantially fewer than SCINet (35,552), Autoformer (68,704), and FEDformer (298,080). The architecture integrates patch embedding, low-rank projections, and adaptive fusion mechanisms to decode complex atmospheric chemistry patterns. Our investigation uncovers distinct diurnal rhythms and pronounced seasonal variations, with winter months experiencing severe pollution episodes driven by temperature inversions and agricultural biomass burning. Analysis identifies critical meteorological thresholds, quantifies wind field impacts on pollutant dispersion, and maps spatial heterogeneity across the region. Extensive ablation experiments demonstrate each architectural component's role. NEXUS delivers superior predictive performance with remarkable computational efficiency, enabling real-time deployment for air quality monitoring systems. 

---
# Denoising Particle Filters: Learning State Estimation with Single-Step Objectives 

**Authors**: Lennart Röstel, Berthold Bäuml  

**Link**: [PDF](https://arxiv.org/pdf/2602.19651)  

**Abstract**: Learning-based methods commonly treat state estimation in robotics as a sequence modeling problem. While this paradigm can be effective at maximizing end-to-end performance, models are often difficult to interpret and expensive to train, since training requires unrolling sequences of predictions in time. As an alternative to end-to-end trained state estimation, we propose a novel particle filtering algorithm in which models are trained from individual state transitions, fully exploiting the Markov property in robotic systems. In this framework, measurement models are learned implicitly by minimizing a denoising score matching objective. At inference, the learned denoiser is used alongside a (learned) dynamics model to approximately solve the Bayesian filtering equation at each time step, effectively guiding predicted states toward the data manifold informed by measurements. We evaluate the proposed method on challenging robotic state estimation tasks in simulation, demonstrating competitive performance compared to tuned end-to-end trained baselines. Importantly, our method offers the desirable composability of classical filtering algorithms, allowing prior information and external sensor models to be incorporated without retraining. 

---
# Compositional Planning with Jumpy World Models 

**Authors**: Jesse Farebrother, Matteo Pirotta, Andrea Tirinzoni, Marc G. Bellemare, Alessandro Lazaric, Ahmed Touati  

**Link**: [PDF](https://arxiv.org/pdf/2602.19634)  

**Abstract**: The ability to plan with temporal abstractions is central to intelligent decision-making. Rather than reasoning over primitive actions, we study agents that compose pre-trained policies as temporally extended actions, enabling solutions to complex tasks that no constituent alone can solve. Such compositional planning remains elusive as compounding errors in long-horizon predictions make it challenging to estimate the visitation distribution induced by sequencing policies. Motivated by the geometric policy composition framework introduced in arXiv:2206.08736, we address these challenges by learning predictive models of multi-step dynamics -- so-called jumpy world models -- that capture state occupancies induced by pre-trained policies across multiple timescales in an off-policy manner. Building on Temporal Difference Flows (arXiv:2503.09817), we enhance these models with a novel consistency objective that aligns predictions across timescales, improving long-horizon predictive accuracy. We further demonstrate how to combine these generative predictions to estimate the value of executing arbitrary sequences of policies over varying timescales. Empirically, we find that compositional planning with jumpy world models significantly improves zero-shot performance across a wide range of base policies on challenging manipulation and navigation tasks, yielding, on average, a 200% relative improvement over planning with primitive actions on long-horizon tasks. 

---
# Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection 

**Authors**: Uichan Lee, Jeonghyeon Kim, Sangheum Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2602.19631)  

**Abstract**: Recent advances in text-to-image (T2I) diffusion models have seen rapid and widespread adoption. However, their powerful generative capabilities raise concerns about potential misuse for synthesizing harmful, private, or copyrighted content. To mitigate such risks, concept erasure techniques have emerged as a promising solution. Prior works have primarily focused on fine-tuning the denoising component (e.g., the U-Net backbone). However, recent causal tracing studies suggest that visual attribute information is localized in the early self-attention layers of the text encoder, indicating a potential alternative for concept erasing. Building on this insight, we conduct preliminary experiments and find that directly fine-tuning early layers can suppress target concepts but often degrades the generation quality of non-target concepts. To overcome this limitation, we propose High-Level Representation Misdirection (HiRM), which misdirects high-level semantic representations of target concepts in the text encoder toward designated vectors such as random directions or semantically defined directions (e.g., supercategories), while updating only early layers that contain causal states of visual attributes. Our decoupling strategy enables precise concept removal with minimal impact on unrelated concepts, as demonstrated by strong results on UnlearnCanvas and NSFW benchmarks across diverse targets (e.g., objects, styles, nudity). HiRM also preserves generative utility at low training cost, transfers to state-of-the-art architectures such as Flux without additional training, and shows synergistic effects with denoiser-based concept erasing methods. 

---
# Cooperation After the Algorithm: Designing Human-AI Coexistence Beyond the Illusion of Collaboration 

**Authors**: Tatia Codreanu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19629)  

**Abstract**: Generative artificial intelligence systems increasingly participate in research, law, education, media, and governance. Their fluent and adaptive outputs create an experience of collaboration. However, these systems do not bear responsibility, incur liability, or share stakes in downstream consequences. This structural asymmetry has already produced sanctions, professional errors, and governance failures in high-stakes contexts We argue that stable human-AI coexistence is an institutional achievement that depends on governance infrastructure capable of distributing residual risk. Drawing on institutional analysis and evolutionary cooperation theory, we introduce a formal inequality that specifies when reliance on AI yields positive expected cooperative value. The model makes explicit how governance conditions, system policy, and accountability regimes jointly determine whether cooperation is rational or structurally defective. From this formalization we derive a cooperation ecology framework with six design principles: reciprocity contracts, visible trust infrastructure, conditional cooperation modes, defection-mitigation mechanisms, narrative literacy against authority theatre, and an Earth-first sustainability constraint. We operationalize the framework through three policy artefacts: a Human-AI Cooperation Charter, a Defection Risk Register, and a Cooperation Readiness Audit. Together, these elements shift the unit of analysis from the user-AI dyad to the institutional environment that shapes incentives, signals, accountability, and repair. The paper provides a theoretical foundation and practical toolkit for designing human-AI systems that can sustain accountable, trustworthy cooperation over time. 

---
# PedaCo-Gen: Scaffolding Pedagogical Agency in Human-AI Collaborative Video Authoring 

**Authors**: Injun Baek, Yearim Kim, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2602.19623)  

**Abstract**: While advancements in Text-to-Video (T2V) generative AI offer a promising path toward democratizing content creation, current models are often optimized for visual fidelity rather than instructional efficacy. This study introduces PedaCo-Gen, a pedagogically-informed human-AI collaborative video generating system for authoring instructional videos based on Mayer's Cognitive Theory of Multimedia Learning (CTML). Moving away from traditional "one-shot" generation, PedaCo-Gen introduces an Intermediate Representation (IR) phase, enabling educators to interactively review and refine video blueprints-comprising scripts and visual descriptions-with an AI reviewer. Our study with 23 education experts demonstrates that PedaCo-Gen significantly enhances video quality across various topics and CTML principles compared to baselines. Participants perceived the AI-driven guidance not merely as a set of instructions but as a metacognitive scaffold that augmented their instructional design expertise, reporting high production efficiency (M=4.26) and guide validity (M=4.04). These findings highlight the importance of reclaiming pedagogical agency through principled co-creation, providing a foundation for future AI authoring tools that harmonize generative power with human professional expertise. 

---
# VecFormer: Towards Efficient and Generalizable Graph Transformer with Graph Token Attention 

**Authors**: Jingbo Zhou, Jun Xia, Siyuan Li, Yunfan Liu, Wenjun Wang, Yufei Huang, Changxi Chi, Mutian Hong, Zhuoli Ouyang, Shu Wang, Zhongqi Wang, Xingyu Wu, Chang Yu, Stan Z. Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.19622)  

**Abstract**: Graph Transformer has demonstrated impressive capabilities in the field of graph representation learning. However, existing approaches face two critical challenges: (1) most models suffer from exponentially increasing computational complexity, making it difficult to scale to large graphs; (2) attention mechanisms based on node-level operations limit the flexibility of the model and result in poor generalization performance in out-of-distribution (OOD) scenarios. To address these issues, we propose \textbf{VecFormer} (the \textbf{Vec}tor Quantized Graph Trans\textbf{former}), an efficient and highly generalizable model for node classification, particularly under OOD settings. VecFormer adopts a two-stage training paradigm. In the first stage, two codebooks are used to reconstruct the node features and the graph structure, aiming to learn the rich semantic \texttt{Graph Codes}. In the second stage, attention mechanisms are performed at the \texttt{Graph Token} level based on the transformed cross codebook, reducing computational complexity while enhancing the model's generalization capability. Extensive experiments on datasets of various sizes demonstrate that VecFormer outperforms the existing Graph Transformer in both performance and speed. 

---
# Satellite-Based Detection of Looted Archaeological Sites Using Machine Learning 

**Authors**: Girmaw Abebe Tadesse, Titien Bartette, Andrew Hassanali, Allen Kim, Jonathan Chemla, Andrew Zolli, Yves Ubelmann, Caleb Robinson, Inbal Becker-Reshef, Juan Lavista Ferres  

**Link**: [PDF](https://arxiv.org/pdf/2602.19608)  

**Abstract**: Looting at archaeological sites poses a severe risk to cultural heritage, yet monitoring thousands of remote locations remains operationally difficult. We present a scalable and satellite-based pipeline to detect looted archaeological sites, using PlanetScope monthly mosaics (4.7m/pixel) and a curated dataset of 1,943 archaeological sites in Afghanistan (898 looted, 1,045 preserved) with multi-year imagery (2016--2023) and site-footprint masks. We compare (i) end-to-end CNN classifiers trained on raw RGB patches and (ii) traditional machine learning (ML) trained on handcrafted spectral/texture features and embeddings from recent remote-sensing foundation models. Results indicate that ImageNet-pretrained CNNs combined with spatial masking reach an F1 score of 0.926, clearly surpassing the strongest traditional ML setup, which attains an F1 score of 0.710 using SatCLIP-V+RF+Mean, i.e., location and vision embeddings fed into a Random Forest with mean-based temporal aggregation. Ablation studies demonstrate that ImageNet pretraining (even in the presence of domain shift) and spatial masking enhance performance. In contrast, geospatial foundation model embeddings perform competitively with handcrafted features, suggesting that looting signatures are extremely localized. The repository is available at this https URL. 

---
# CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning 

**Authors**: Chunlei Meng, Guanhong Huang, Rong Fu, Runmin Jian, Zhongxue Gan, Chun Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2602.19605)  

**Abstract**: Multimodal learning aims to capture both shared and private information from multiple modalities. However, existing methods that project all modalities into a single latent space for fusion often overlook the asynchronous, multi-level semantic structure of multimodal data. This oversight induces semantic misalignment and error propagation, thereby degrading representation quality. To address this issue, we propose Cross-Level Co-Representation (CLCR), which explicitly organizes each modality's features into a three-level semantic hierarchy and specifies level-wise constraints for cross-modal interactions. First, a semantic hierarchy encoder aligns shallow, mid, and deep features across modalities, establishing a common basis for interaction. And then, at each level, an Intra-Level Co-Exchange Domain (IntraCED) factorizes features into shared and private subspaces and restricts cross-modal attention to the shared subspace via a learnable token budget. This design ensures that only shared semantics are exchanged and prevents leakage from private channels. To integrate information across levels, the Inter-Level Co-Aggregation Domain (InterCAD) synchronizes semantic scales using learned anchors, selectively fuses the shared representations, and gates private cues to form a compact task representation. We further introduce regularization terms to enforce separation of shared and private features and to minimize cross-level interference. Experiments on six benchmarks spanning emotion recognition, event localization, sentiment analysis, and action recognition show that CLCR achieves strong performance and generalizes well across tasks. 

---
# Detecting High-Potential SMEs with Heterogeneous Graph Neural Networks 

**Authors**: Yijiashun Qi, Hanzhe Guo, Yijiazhen Qi  

**Link**: [PDF](https://arxiv.org/pdf/2602.19591)  

**Abstract**: Small and Medium Enterprises (SMEs) constitute 99.9% of U.S. businesses and generate 44% of economic activity, yet systematically identifying high-potential SMEs remains an open challenge. We introduce SME-HGT, a Heterogeneous Graph Transformer framework that predicts which SBIR Phase I awardees will advance to Phase II funding using exclusively public data. We construct a heterogeneous graph with 32,268 company nodes, 124 research topic nodes, and 13 government agency nodes connected by approximately 99,000 edges across three semantic relation types. SME-HGT achieves an AUPRC of 0.621 0.003 on a temporally-split test set, outperforming an MLP baseline (0.590 0.002) and R-GCN (0.608 0.013) across five random seeds. At a screening depth of 100 companies, SME-HGT attains 89.6% precision with a 2.14 lift over random selection. Our temporal evaluation protocol prevents information leakage, and our reliance on public data ensures reproducibility. These results demonstrate that relational structure among firms, research topics, and funding agencies provides meaningful signal for SME potential assessment, with implications for policymakers and early-stage investors. 

---
# Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis 

**Authors**: Chunlei Meng, Jiabin Luo, Zhenglin Yan, Zhenyu Yu, Rong Fu, Zhongxue Gan, Chun Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2602.19585)  

**Abstract**: Multimodal Sentiment Analysis (MSA) integrates language, visual, and acoustic modalities to infer human sentiment. Most existing methods either focus on globally shared representations or modality-specific features, while overlooking signals that are shared only by certain modality pairs. This limits the expressiveness and discriminative power of multimodal representations. To address this limitation, we propose a Tri-Subspace Disentanglement (TSD) framework that explicitly factorizes features into three complementary subspaces: a common subspace capturing global consistency, submodally-shared subspaces modeling pairwise cross-modal synergies, and private subspaces preserving modality-specific cues. To keep these subspaces pure and independent, we introduce a decoupling supervisor together with structured regularization losses. We further design a Subspace-Aware Cross-Attention (SACA) fusion module that adaptively models and integrates information from the three subspaces to obtain richer and more robust representations. Experiments on CMU-MOSI and CMU-MOSEI demonstrate that TSD achieves state-of-the-art performance across all key metrics, reaching 0.691 MAE on CMU-MOSI and 54.9% ACC-7 on CMU-MOSEI, and also transfers well to multimodal intent recognition tasks. Ablation studies confirm that tri-subspace disentanglement and SACA jointly enhance the modeling of multi-granular cross-modal sentiment cues. 

---
# Interpolation-Driven Machine Learning Approaches for Plume Shine Dose Estimation: A Comparison of XGBoost, Random Forest, and TabNet 

**Authors**: Biswajit Sadhu, Kalpak Gupte, Trijit Sadhu, S. Anand  

**Link**: [PDF](https://arxiv.org/pdf/2602.19584)  

**Abstract**: Despite the success of machine learning (ML) in surrogate modeling, its use in radiation dose assessment is limited by safety-critical constraints, scarce training-ready data, and challenges in selecting suitable architectures for physics-dominated systems. Within this context, rapid and accurate plume shine dose estimation serves as a practical test case, as it is critical for nuclear facility safety assessment and radiological emergency response, while conventional photon-transport-based calculations remain computationally expensive. In this work, an interpolation-assisted ML framework was developed using discrete dose datasets generated with the pyDOSEIA suite for 17 gamma-emitting radionuclides across varying downwind distances, release heights, and atmospheric stability categories. The datasets were augmented using shape-preserving interpolation to construct dense, high-resolution training data. Two tree-based ML models (Random Forest and XGBoost) and one deep learning (DL) model (TabNet) were evaluated to examine predictive performance and sensitivity to dataset resolution. All models showed higher prediction accuracy with the interpolated high-resolution dataset than with the discrete data; however, XGBoost consistently achieved the highest accuracy. Interpretability analysis using permutation importance (tree-based models) and attention-based feature attribution (TabNet) revealed that performance differences stem from how the models utilize input features. Tree-based models focus mainly on dominant geometry-dispersion features (release height, stability category, and downwind distance), treating radionuclide identity as a secondary input, whereas TabNet distributes attention more broadly across multiple variables. For practical deployment, a web-based GUI was developed for interactive scenario evaluation and transparent comparison with photon-transport reference calculations. 

---
# CTC-TTS: LLM-based dual-streaming text-to-speech with CTC alignment 

**Authors**: Hanwen Liu, Saierdaer Yusuyin, Hao Huang, Zhijian Ou  

**Link**: [PDF](https://arxiv.org/pdf/2602.19574)  

**Abstract**: Large-language-model (LLM)-based text-to-speech (TTS) systems can generate natural speech, but most are not designed for low-latency dual-streaming synthesis. High-quality dual-streaming TTS depends on accurate text--speech alignment and well-designed training sequences that balance synthesis quality and latency. Prior work often relies on GMM-HMM based forced-alignment toolkits (e.g., MFA), which are pipeline-heavy and less flexible than neural aligners; fixed-ratio interleaving of text and speech tokens struggles to capture text--speech alignment regularities. We propose CTC-TTS, which replaces MFA with a CTC based aligner and introduces a bi-word based interleaving strategy. Two variants are designed: CTC-TTS-L (token concatenation along the sequence length) for higher quality and CTC-TTS-F (embedding stacking along the feature dimension) for lower latency. Experiments show that CTC-TTS outperforms fixed-ratio interleaving and MFA-based baselines on streaming synthesis and zero-shot tasks. Speech samples are available at this https URL. 

---
# Temporal-Aware Heterogeneous Graph Reasoning with Multi-View Fusion for Temporal Question Answering 

**Authors**: Wuzhenghong Wen, Bowen Zhou, Jinwen Huang, Xianjie Wu, Yuwei Sun, Su Pan, Liang Li, Jianting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19569)  

**Abstract**: Question Answering over Temporal Knowledge Graphs (TKGQA) has attracted growing interest for handling time-sensitive queries. However, existing methods still struggle with: 1) weak incorporation of temporal constraints in question representation, causing biased reasoning; 2) limited ability to perform explicit multi-hop reasoning; and 3) suboptimal fusion of language and graph representations. We propose a novel framework with temporal-aware question encoding, multi-hop graph reasoning, and multi-view heterogeneous information fusion. Specifically, our approach introduces: 1) a constraint-aware question representation that combines semantic cues from language models with temporal entity dynamics; 2) a temporal-aware graph neural network for explicit multi-hop reasoning via time-aware message passing; and 3) a multi-view attention mechanism for more effective fusion of question context and temporal graph knowledge. Experiments on multiple TKGQA benchmarks demonstrate consistent improvements over multiple baselines. 

---
# DICArt: Advancing Category-level Articulated Object Pose Estimation in Discrete State-Spaces 

**Authors**: Li Zhang, Mingyu Mei, Ailing Wang, Xianhui Meng, Yan Zhong, Xinyuan Song, Liu Liu, Rujing Wang, Zaixing He, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19565)  

**Abstract**: Articulated object pose estimation is a core task in embodied AI. Existing methods typically regress poses in a continuous space, but often struggle with 1) navigating a large, complex search space and 2) failing to incorporate intrinsic kinematic constraints. In this work, we introduce DICArt (DIsCrete Diffusion for Articulation Pose Estimation), a novel framework that formulates pose estimation as a conditional discrete diffusion process. Instead of operating in a continuous domain, DICArt progressively denoises a noisy pose representation through a learned reverse diffusion procedure to recover the GT pose. To improve modeling fidelity, we propose a flexible flow decider that dynamically determines whether each token should be denoised or reset, effectively balancing the real and noise distributions during diffusion. Additionally, we incorporate a hierarchical kinematic coupling strategy, estimating the pose of each rigid part hierarchically to respect the object's kinematic structure. We validate DICArt on both synthetic and real-world datasets. Experimental results demonstrate its superior performance and robustness. By integrating discrete generative modeling with structural priors, DICArt offers a new paradigm for reliable category-level 6D pose estimation in complex environments. 

---
# Agentic AI as a Cybersecurity Attack Surface: Threats, Exploits, and Defenses in Runtime Supply Chains 

**Authors**: Xiaochong Jiang, Shiqi Yang, Wenting Yang, Yichen Liu, Cheng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2602.19555)  

**Abstract**: Agentic systems built on large language models (LLMs) extend beyond text generation to autonomously retrieve information and invoke tools. This runtime execution model shifts the attack surface from build-time artifacts to inference-time dependencies, exposing agents to manipulation through untrusted data and probabilistic capability resolution. While prior work has focused on model-level vulnerabilities, security risks emerging from cyclic and interdependent runtime behavior remain fragmented. We systematize these risks within a unified runtime framework, categorizing threats into data supply chain attacks (transient context injection and persistent memory poisoning) and tool supply chain attacks (discovery, implementation, and invocation). We further identify the Viral Agent Loop, in which agents act as vectors for self-propagating generative worms without exploiting code-level flaws. Finally, we advocate a Zero-Trust Runtime Architecture that treats context as untrusted control flow and constrains tool execution through cryptographic provenance rather than semantic inference. 

---
# A Green Learning Approach to LDCT Image Restoration 

**Authors**: Wei Wang, Yixing Wu, C.-C. Jay Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2602.19540)  

**Abstract**: This work proposes a green learning (GL) approach to restore medical images. Without loss of generality, we use low-dose computed tomography (LDCT) images as examples. LDCT images are susceptible to noise and artifacts, where the imaging process introduces distortion. LDCT image restoration is an important preprocessing step for further medical analysis. Deep learning (DL) methods have been developed to solve this problem. We examine an alternative solution using the Green Learning (GL) methodology. The new restoration method is characterized by mathematical transparency, computational and memory efficiency, and high performance. Experiments show that our GL method offers state-of-the-art restoration performance at a smaller model size and with lower inference complexity. 

---
# Cost-Aware Diffusion Active Search 

**Authors**: Arundhati Banerjee, Jeff Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2602.19538)  

**Abstract**: Active search for recovering objects of interest through online, adaptive decision making with autonomous agents requires trading off exploration of unknown environments with exploitation of prior observations in the search space. Prior work has proposed information gain and Thompson sampling based myopic, greedy approaches for agents to actively decide query or search locations when the number of targets is unknown. Decision making algorithms in such partially observable environments have also shown that agents capable of lookahead over a finite horizon outperform myopic policies for active search. Unfortunately, lookahead algorithms typically rely on building a computationally expensive search tree that is simulated and updated based on the agent's observations and a model of the environment dynamics. Instead, in this work, we leverage the sequence modeling abilities of diffusion models to sample lookahead action sequences that balance the exploration-exploitation trade-off for active search without building an exhaustive search tree. We identify the optimism bias in prior diffusion based reinforcement learning approaches when applied to the active search setting and propose mitigating solutions for efficient cost-aware decision making with both single and multi-agent teams. Our proposed algorithm outperforms standard baselines in offline reinforcement learning in terms of full recovery rate and is computationally more efficient than tree search in cost-aware active decision making. 

---
# Fore-Mamba3D: Mamba-based Foreground-Enhanced Encoding for 3D Object Detection 

**Authors**: Zhiwei Ning, Xuanang Gao, Jiaxi Cao, Runze Yang, Huiying Xu, Xinzhong Zhu, Jie Yang, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19536)  

**Abstract**: Linear modeling methods like Mamba have been merged as the effective backbone for the 3D object detection task. However, previous Mamba-based methods utilize the bidirectional encoding for the whole non-empty voxel sequence, which contains abundant useless background information in the scenes. Though directly encoding foreground voxels appears to be a plausible solution, it tends to degrade detection performance. We attribute this to the response attenuation and restricted context representation in the linear modeling for fore-only sequences. To address this problem, we propose a novel backbone, termed Fore-Mamba3D, to focus on the foreground enhancement by modifying Mamba-based encoder. The foreground voxels are first sampled according to the predicted scores. Considering the response attenuation existing in the interaction of foreground voxels across different instances, we design a regional-to-global slide window (RGSW) to propagate the information from regional split to the entire sequence. Furthermore, a semantic-assisted and state spatial fusion module (SASFMamba) is proposed to enrich contextual representation by enhancing semantic and geometric awareness within the Mamba model. Our method emphasizes foreground-only encoding and alleviates the distance-based and causal dependencies in the linear autoregression model. The superior performance across various benchmarks demonstrates the effectiveness of Fore-Mamba3D in the 3D object detection task. 

---
# Large Language Model-Assisted UAV Operations and Communications: A Multifaceted Survey and Tutorial 

**Authors**: Yousef Emami, Hao Zhou, Radha Reddy, Atefeh Hajijamali Arani, Biliang Wang, Kai Li, Luis Almeida, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.19534)  

**Abstract**: Uncrewed Aerial Vehicles (UAVs) are widely deployed across diverse applications due to their mobility and agility. Recent advances in Large Language Models (LLMs) offer a transformative opportunity to enhance UAV intelligence beyond conventional optimization-based and learning-based approaches. By integrating LLMs into UAV systems, advanced environmental understanding, swarm coordination, mobility optimization, and high-level task reasoning can be achieved, thereby allowing more adaptive and context-aware aerial operations. This survey systematically explores the intersection of LLMs and UAV technologies and proposes a unified framework that consolidates existing architectures, methodologies, and applications for UAVs. We first present a structured taxonomy of LLM adaptation techniques for UAVs, including pretraining, fine-tuning, Retrieval-Augmented Generation (RAG), and prompt engineering, along with key reasoning capabilities such as Chain-of-Thought (CoT) and In-Context Learning (ICL). We then examine LLM-assisted UAV communications and operations, covering navigation, mission planning, swarm control, safety, autonomy, and network management. After that, the survey further discusses Multimodal LLMs (MLLMs) for human-swarm interaction, perception-driven navigation, and collaborative control. Finally, we address ethical considerations, including bias, transparency, accountability, and Human-in-the-Loop (HITL) strategies, and outline future research directions. Overall, this work positions LLM-assisted UAVs as a foundation for intelligent and adaptive aerial systems. 

---
# Grokking Finite-Dimensional Algebra 

**Authors**: Pascal Jr Tikeng Notsawo, Guillaume Dumas, Guillaume Rabusseau  

**Link**: [PDF](https://arxiv.org/pdf/2602.19533)  

**Abstract**: This paper investigates the grokking phenomenon, which refers to the sudden transition from a long memorization to generalization observed during neural networks training, in the context of learning multiplication in finite-dimensional algebras (FDA). While prior work on grokking has focused mainly on group operations, we extend the analysis to more general algebraic structures, including non-associative, non-commutative, and non-unital algebras. We show that learning group operations is a special case of learning FDA, and that learning multiplication in FDA amounts to learning a bilinear product specified by the algebra's structure tensor. For algebras over the reals, we connect the learning problem to matrix factorization with an implicit low-rank bias, and for algebras over finite fields, we show that grokking emerges naturally as models must learn discrete representations of algebraic elements. This leads us to experimentally investigate the following core questions: (i) how do algebraic properties such as commutativity, associativity, and unitality influence both the emergence and timing of grokking, (ii) how structural properties of the structure tensor of the FDA, such as sparsity and rank, influence generalization, and (iii) to what extent generalization correlates with the model learning latent embeddings aligned with the algebra's representation. Our work provides a unified framework for grokking across algebraic structures and new insights into how mathematical structure governs neural network generalization dynamics. 

---
# A Statistical Approach for Modeling Irregular Multivariate Time Series with Missing Observations 

**Authors**: Dingyi Nie, Yixing Wu, C.-C. Jay Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2602.19531)  

**Abstract**: Irregular multivariate time series with missing values present significant challenges for predictive modeling in domains such as healthcare. While deep learning approaches often focus on temporal interpolation or complex architectures to handle irregularities, we propose a simpler yet effective alternative: extracting time-agnostic summary statistics to eliminate the temporal axis. Our method computes four key features per variable-mean and standard deviation of observed values, as well as the mean and variability of changes between consecutive observations to create a fixed-dimensional representation. These features are then utilized with standard classifiers, such as logistic regression and XGBoost. Evaluated on four biomedical datasets (PhysioNet Challenge 2012, 2019, PAMAP2, and MIMIC-III), our approach achieves state-of-the-art performance, surpassing recent transformer and graph-based models by 0.5-1.7% in AUROC/AUPRC and 1.1-1.7% in accuracy/F1-score, while reducing computational complexity. Ablation studies demonstrate that feature extraction-not classifier choice-drives performance gains, and our summary statistics outperform raw/imputed input in most benchmarks. In particular, we identify scenarios where missing patterns themselves encode predictive signals, as in sepsis prediction (PhysioNet, 2019), where missing indicators alone can achieve 94.2% AUROC with XGBoost, only 1.6% lower than using original raw data as input. Our results challenge the necessity of complex temporal modeling when task objectives permit time-agnostic representations, providing an efficient and interpretable solution for irregular time series classification. 

---
# Pyramid MoA: A Probabilistic Framework for Cost-Optimized Anytime Inference 

**Authors**: Arindam Khaled  

**Link**: [PDF](https://arxiv.org/pdf/2602.19509)  

**Abstract**: Large Language Models (LLMs) face a persistent trade-off between inference cost and reasoning capability. While "Oracle" models (e.g., Llama-3-70B) achieve state-of-the-art accuracy, they are prohibitively expensive for high-volume deployment. Smaller models (e.g., 8B parameters) are cost-effective but struggle with complex tasks. In this work, we propose "Pyramid MoA", a hierarchical Mixture-of-Agents architecture that uses a lightweight Router to dynamically escalate queries only when necessary. By leveraging semantic agreement and confidence calibration among an ensemble of small models, our Router identifies "hard" problems with high precision. On the GSM8K benchmark, our system achieves 93.0% accuracy, effectively matching the Oracle baseline (98.0%) while reducing compute costs by 61%. We demonstrate that the system introduces negligible latency overhead (+0.82s) and allows for a tunable trade-off between performance and budget. 

---
# Softmax is not Enough (for Adaptive Conformal Classification) 

**Authors**: Navid Akhavan Attar, Hesam Asadollahzadeh, Ling Luo, Uwe Aickelin  

**Link**: [PDF](https://arxiv.org/pdf/2602.19498)  

**Abstract**: The merit of Conformal Prediction (CP), as a distribution-free framework for uncertainty quantification, depends on generating prediction sets that are efficient, reflected in small average set sizes, while adaptive, meaning they signal uncertainty by varying in size according to input difficulty. A central limitation for deep conformal classifiers is that the nonconformity scores are derived from softmax outputs, which can be unreliable indicators of how certain the model truly is about a given input, sometimes leading to overconfident misclassifications or undue hesitation. In this work, we argue that this unreliability can be inherited by the prediction sets generated by CP, limiting their capacity for adaptiveness. We propose a new approach that leverages information from the pre-softmax logit space, using the Helmholtz Free Energy as a measure of model uncertainty and sample difficulty. By reweighting nonconformity scores with a monotonic transformation of the energy score of each sample, we improve their sensitivity to input difficulty. Our experiments with four state-of-the-art score functions on multiple datasets and deep architectures show that this energy-based enhancement improves the adaptiveness of the prediction sets, leading to a notable increase in both efficiency and adaptiveness compared to baseline nonconformity scores, without introducing any post-hoc complexity. 

---
# Botson: An Accessible and Low-Cost Platform for Social Robotics Research 

**Authors**: Samuel Bellaire, Abdalmalek Abu-raddaha, Natalie Kim, Nathan Morhan, William Elliott, Samir Rawashdeh  

**Link**: [PDF](https://arxiv.org/pdf/2602.19491)  

**Abstract**: Trust remains a critical barrier to the effective integration of Artificial Intelligence (AI) into human-centric domains. Disembodied agents, such as voice assistants, often fail to establish trust due to their inability to convey non-verbal social cues. This paper introduces the architecture of Botson: an anthropomorphic social robot powered by a large language model (LLM). Botson was created as a low-cost and accessible platform for social robotics research. 

---
# Federated Learning Playground 

**Authors**: Bryan Guanrong Shan, Alysa Ziying Tan, Han Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19489)  

**Abstract**: We present Federated Learning Playground, an interactive browser-based platform inspired by and extends TensorFlow Playground that teaches core Federated Learning (FL) concepts. Users can experiment with heterogeneous client data distributions, model hyperparameters, and aggregation algorithms directly in the browser without coding or system setup, and observe their effects on client and global models through real-time visualizations, gaining intuition for challenges such as non-IID data, local overfitting, and scalability. The playground serves as an easy to use educational tool, lowering the entry barrier for newcomers to distributed AI while also offering a sandbox for rapidly prototyping and comparing FL methods. By democratizing exploration of FL, it promotes broader understanding and adoption of this important paradigm. 

---
# Making Conformal Predictors Robust in Healthcare Settings: a Case Study on EEG Classification 

**Authors**: Arjun Chatterjee, Sayeed Sajjad Razin, John Wu, Siddhartha Laghuvarapu, Jathurshan Pradeepkumar, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.19483)  

**Abstract**: Quantifying uncertainty in clinical predictions is critical for high-stakes diagnosis tasks. Conformal prediction offers a principled approach by providing prediction sets with theoretical coverage guarantees. However, in practice, patient distribution shifts violate the i.i.d. assumptions underlying standard conformal methods, leading to poor coverage in healthcare settings. In this work, we evaluate several conformal prediction approaches on EEG seizure classification, a task with known distribution shift challenges and label uncertainty. We demonstrate that personalized calibration strategies can improve coverage by over 20 percentage points while maintaining comparable prediction set sizes. Our implementation is available via PyHealth, an open-source healthcare AI framework: this https URL. 

---
# Scale-PINN: Learning Efficient Physics-Informed Neural Networks Through Sequential Correction 

**Authors**: Pao-Hsiung Chiu, Jian Cheng Wong, Chin Chun Ooi, Chang Wei, Yuchen Fan, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2602.19475)  

**Abstract**: Physics-informed neural networks (PINNs) have emerged as a promising mesh-free paradigm for solving partial differential equations, yet adoption in science and engineering is limited by slow training and modest accuracy relative to modern numerical solvers. We introduce the Sequential Correction Algorithm for Learning Efficient PINN (Scale-PINN), a learning strategy that bridges modern physics-informed learning with numerical algorithms. Scale-PINN incorporates the iterative residual-correction principle, a cornerstone of numerical solvers, directly into the loss formulation, marking a paradigm shift in how PINN losses can be conceived and constructed. This integration enables Scale-PINN to achieve unprecedented convergence speed across PDE problems from different physics domain, including reducing training time on a challenging fluid-dynamics problem for state-of-the-art PINN from hours to sub-2 minutes while maintaining superior accuracy, and enabling application to representative problems in aerodynamics and urban science. By uniting the rigor of numerical methods with the flexibility of deep learning, Scale-PINN marks a significant leap toward the practical adoption of PINNs in science and engineering through scalable, physics-informed learning. Codes are available at this https URL. 

---
# Can Large Language Models Replace Human Coders? Introducing ContentBench 

**Authors**: Michael Haman  

**Link**: [PDF](https://arxiv.org/pdf/2602.19467)  

**Abstract**: Can low-cost large language models (LLMs) take over the interpretive coding work that still anchors much of empirical content analysis? This paper introduces ContentBench, a public benchmark suite that helps answer this replacement question by tracking how much agreement low-cost LLMs achieve and what they cost on the same interpretive coding tasks. The suite uses versioned tracks that invite researchers to contribute new benchmark datasets. I report results from the first track, ContentBench-ResearchTalk v1.0: 1,000 synthetic, social-media-style posts about academic research labeled into five categories spanning praise, critique, sarcasm, questions, and procedural remarks. Reference labels are assigned only when three state-of-the-art reasoning models (GPT-5, Gemini 2.5 Pro, and Claude Opus 4.1) agree unanimously, and all final labels are checked by the author as a quality-control audit. Among the 59 evaluated models, the best low-cost LLMs reach roughly 97-99% agreement with these jury labels, far above GPT-3.5 Turbo, the model behind early ChatGPT and the initial wave of LLM-based text annotation. Several top models can code 50,000 posts for only a few dollars, pushing large-scale interpretive coding from a labor bottleneck toward questions of validation, reporting, and governance. At the same time, small open-weight models that run locally still struggle on sarcasm-heavy items (for example, Llama 3.2 3B reaches only 4% agreement on hard-sarcasm). ContentBench is released with data, documentation, and an interactive quiz at this http URL to support comparable evaluations over time and to invite community extensions. 

---
# PuppetChat: Fostering Intimate Communication through Bidirectional Actions and Micronarratives 

**Authors**: Emma Jiren Wang, Siying Hu, Zhicong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19463)  

**Abstract**: As a primary channel for sustaining modern intimate relationships, instant messaging facilitates frequent connection across distances. However, today's tools often dilute care; they favor single tap reactions and vague emojis that do not support two way action responses, do not preserve the feeling that the exchange keeps going without breaking, and are weakly tied to who we are and what we share. To address this challenge, we present PuppetChat, a dyadic messaging prototype that restores this expressive depth through embodied interaction. PuppetChat uses a reciprocity aware recommender to encourage responsive actions and generates personalized micronarratives from user stories to ground interactions in personal history. Our 10-day field study with 11 dyads of close partners or friends revealed that this approach enhanced social presence, supported more expressive self disclosure, and sustained continuity and shared memories. 

---
# SenTSR-Bench: Thinking with Injected Knowledge for Time-Series Reasoning 

**Authors**: Zelin He, Boran Han, Xiyuan Zhang, Shuai Zhang, Haotian Lin, Qi Zhu, Haoyang Fang, Danielle C. Maddix, Abdul Fatir Ansari, Akash Chandrayan, Abhinav Pradhan, Bernie Wang, Matthew Reimherr  

**Link**: [PDF](https://arxiv.org/pdf/2602.19455)  

**Abstract**: Time-series diagnostic reasoning is essential for many applications, yet existing solutions face a persistent gap: general reasoning large language models (GRLMs) possess strong reasoning skills but lack the domain-specific knowledge to understand complex time-series patterns. Conversely, fine-tuned time-series LLMs (TSLMs) understand these patterns but lack the capacity to generalize reasoning for more complicated questions. To bridge this gap, we propose a hybrid knowledge-injection framework that injects TSLM-generated insights directly into GRLM's reasoning trace, thereby achieving strong time-series reasoning with in-domain knowledge. As collecting data for knowledge injection fine-tuning is costly, we further leverage a reinforcement learning-based approach with verifiable rewards (RLVR) to elicit knowledge-rich traces without human supervision, then transfer such an in-domain thinking trace into GRLM for efficient knowledge injection. We further release SenTSR-Bench, a multivariate time-series-based diagnostic reasoning benchmark collected from real-world industrial operations. Across SenTSR-Bench and other public datasets, our method consistently surpasses TSLMs by 9.1%-26.1% and GRLMs by 7.9%-22.4%, delivering robust, context-aware time-series diagnostic insights. 

---
# Red-Teaming Claude Opus and ChatGPT-based Security Advisors for Trusted Execution Environments 

**Authors**: Kunal Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2602.19450)  

**Abstract**: Trusted Execution Environments (TEEs) (e.g., Intel SGX and ArmTrustZone) aim to protect sensitive computation from a compromised operating system, yet real deployments remain vulnerable to microarchitectural leakage, side-channel attacks, and fault injection. In parallel, security teams increasingly rely on Large Language Model (LLM) assistants as security advisors for TEE architecture review, mitigation planning, and vulnerability triage. This creates a socio-technical risk surface: assistants may hallucinate TEE mechanisms, overclaim guarantees (e.g., what attestation does and does not establish), or behave unsafely under adversarial prompting.
We present a red-teaming study of two prevalently deployed LLM assistants in the role of TEE security advisors: ChatGPT-5.2 and Claude Opus-4.6, focusing on the inherent limitations and transferability of prompt-induced failures across LLMs. We introduce TEE-RedBench, a TEE-grounded evaluation methodology comprising (i) a TEE-specific threat model for LLM-mediated security work, (ii) a structured prompt suite spanning SGX and TrustZone architecture, attestation and key management, threat modeling, and non-operational mitigation guidance, along with policy-bound misuse probes, and (iii) an annotation rubric that jointly measures technical correctness, groundedness, uncertainty calibration, refusal quality, and safe helpfulness. We find that some failures are not purely idiosyncratic, transferring up to 12.02% across LLM assistants, and we connect these outcomes to secure architecture by outlining an "LLM-in-the-loop" evaluation pipeline: policy gating, retrieval grounding, structured templates, and lightweight verification checks that, when combined, reduce failures by 80.62%. 

---
# When AI Teammates Meet Code Review: Collaboration Signals Shaping the Integration of Agent-Authored Pull Requests 

**Authors**: Costain Nachuma, Minhaz Zibran  

**Link**: [PDF](https://arxiv.org/pdf/2602.19441)  

**Abstract**: Autonomous coding agents increasingly contribute to software development by submitting pull requests on GitHub; yet, little is known about how these contributions integrate into human-driven review workflows. We present a large empirical study of agent-authored pull requests using the public AIDev dataset, examining integration outcomes, resolution speed, and review-time collaboration signals. Using logistic regression with repository-clustered standard errors, we find that reviewer engagement has the strongest correlation with successful integration, whereas larger change sizes and coordination-disrupting actions, such as force pushes, are associated with a lower likelihood of merging. In contrast, iteration intensity alone provides limited explanatory power once collaboration signals are considered. A qualitative analysis further shows that successful integration occurs when agents engage in actionable review loops that converge toward reviewer expectations. Overall, our results highlight that the effective integration of agent-authored pull requests depends not only on code quality but also on alignment with established review and coordination practices. 

---
# FinSight-Net:A Physics-Aware Decoupled Network with Frequency-Domain Compensation for Underwater Fish Detection in Smart Aquaculture 

**Authors**: Jinsong Yang, Zeyuan Hu, Yichen Li, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19437)  

**Abstract**: Underwater fish detection (UFD) is a core capability for smart aquaculture and marine ecological monitoring. While recent detectors improve accuracy by stacking feature extractors or introducing heavy attention modules, they often incur substantial computational overhead and, more importantly, neglect the physics that fundamentally limits UFD: wavelength-dependent absorption and turbidity-induced scattering significantly degrade contrast, blur fine structures, and introduce backscattering noise, leading to unreliable localization and recognition. To address these challenges, we propose FinSight-Net, an efficient and physics-aware detection framework tailored for complex aquaculture environments. FinSight-Net introduces a Multi-Scale Decoupled Dual-Stream Processing (MS-DDSP) bottleneck that explicitly targets frequency-specific information loss via heterogeneous convolutional branches, suppressing backscattering artifacts while compensating distorted biological cues through scale-aware and channel-weighted pathways. We further design an Efficient Path Aggregation FPN (EPA-FPN) as a detail-filling mechanism: it restores high-frequency spatial information typically attenuated in deep layers by establishing long-range skip connections and pruning redundant fusion routes, enabling robust detection of non-rigid fish targets under severe blur and turbidity. Extensive experiments on DeepFish, AquaFishSet, and our challenging UW-BlurredFish benchmark demonstrate that FinSight-Net achieves state-of-the-art performance. In particular, on UW-BlurredFish, FinSight-Net reaches 92.8% mAP, outperforming YOLOv11s by 4.8% while reducing parameters by 29.0%, providing a strong and lightweight solution for real-time automated monitoring in smart aquaculture. 

---
# Redefining the Down-Sampling Scheme of U-Net for Precision Biomedical Image Segmentation 

**Authors**: Mingjie Li, Yizheng Chen, Md Tauhidul Islam, Lei Xing  

**Link**: [PDF](https://arxiv.org/pdf/2602.19412)  

**Abstract**: U-Net architectures have been instrumental in advancing biomedical image segmentation (BIS) but often struggle with capturing long-range information. One reason is the conventional down-sampling techniques that prioritize computational efficiency at the expense of information retention. This paper introduces a simple but effective strategy, we call it Stair Pooling, which moderates the pace of down-sampling and reduces information loss by leveraging a sequence of concatenated small and narrow pooling operations in varied orientations. Specifically, our method modifies the reduction in dimensionality within each 2D pooling step from $\frac{1}{4}$ to $\frac{1}{2}$. This approach can also be adapted for 3D pooling to preserve even more information. Such preservation aids the U-Net in more effectively reconstructing spatial details during the up-sampling phase, thereby enhancing its ability to capture long-range information and improving segmentation accuracy. Extensive experiments on three BIS benchmarks demonstrate that the proposed Stair Pooling can increase both 2D and 3D U-Net performance by an average of 3.8\% in Dice scores. Moreover, we leverage the transfer entropy to select the optimal down-sampling paths and quantitatively show how the proposed Stair Pooling reduces the information loss. 

---
# One Size Fits None: Modeling NYC Taxi Trips 

**Authors**: Tomas Eglinskas  

**Link**: [PDF](https://arxiv.org/pdf/2602.19404)  

**Abstract**: The rise of app-based ride-sharing has fundamentally changed tipping culture in New York City. We analyzed 280 million trips from 2024 to see if we could predict tips for traditional taxis versus high-volume for-hire services. By testing methods from linear regression to deep neural networks, we found two very different outcomes. Traditional taxis are highly predictable ($R^2 \approx 0.72$) due to the in-car payment screen. In contrast, app-based tipping is random and hard to model ($R^2 \approx 0.17$). In conclusion, we show that building one universal model is a mistake and, due to Simpson's paradox, a combined model looks accurate on average but fails to predict tips for individual taxi categories requiring specialized models. 

---
# Hilbert-Augmented Reinforcement Learning for Scalable Multi-Robot Coverage and Exploration 

**Authors**: Tamil Selvan Gurunathan, Aryya Gangopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2602.19400)  

**Abstract**: We present a coverage framework that integrates Hilbert space-filling priors into decentralized multi-robot learning and execution. We augment DQN and PPO with Hilbert-based spatial indices to structure exploration and reduce redundancy in sparse-reward environments, and we evaluate scalability in multi-robot grid coverage. We further describe a waypoint interface that converts Hilbert orderings into curvature-bounded, time-parameterized SE(2) trajectories (planar (x, y, {\theta})), enabling onboard feasibility on resource-constrained robots. Experiments show improvements in coverage efficiency, redundancy, and convergence speed over DQN/PPO baselines. In addition, we validate the approach on a Boston Dynamics Spot legged robot, executing the generated trajectories in indoor environments and observing reliable coverage with low redundancy. These results indicate that geometric priors improve autonomy and scalability for swarm and legged robotics. 

---
# Stable Deep Reinforcement Learning via Isotropic Gaussian Representations 

**Authors**: Ali Saheb, Johan Obando-Ceron, Aaron Courville, Pouya Bashivan, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2602.19373)  

**Abstract**: Deep reinforcement learning systems often suffer from unstable training dynamics due to non-stationarity, where learning objectives and data distributions evolve over time. We show that under non-stationary targets, isotropic Gaussian embeddings are provably advantageous. In particular, they induce stable tracking of time-varying targets for linear readouts, achieve maximal entropy under a fixed variance budget, and encourage a balanced use of all representational dimensions--all of which enable agents to be more adaptive and stable. Building on this insight, we propose the use of Sketched Isotropic Gaussian Regularization for shaping representations toward an isotropic Gaussian distribution during training. We demonstrate empirically, over a variety of domains, that this simple and computationally inexpensive method improves performance under non-stationarity while reducing representation collapse, neuron dormancy, and training instability. 

---
# Active perception and disentangled representations allow continual, episodic zero and few-shot learning 

**Authors**: David Rawlinson, Gideon Kowadlo  

**Link**: [PDF](https://arxiv.org/pdf/2602.19355)  

**Abstract**: Generalization is often regarded as an essential property of machine learning systems. However, perhaps not every component of a system needs to generalize. Training models for generalization typically produces entangled representations at the boundaries of entities or classes, which can lead to destructive interference when rapid, high-magnitude updates are required for continual or few-shot learning. Techniques for fast learning with non-interfering representations exist, but they generally fail to generalize. Here, we describe a Complementary Learning System (CLS) in which the fast learner entirely foregoes generalization in exchange for continual zero-shot and few-shot learning. Unlike most CLS approaches, which use episodic memory primarily for replay and consolidation, our fast, disentangled learner operates as a parallel reasoning system. The fast learner can overcome observation variability and uncertainty by leveraging a conventional slow, statistical learner within an active perception system: A contextual bias provided by the fast learner induces the slow learner to encode novel stimuli in familiar, generalized terms, enabling zero-shot and few-shot learning. This architecture demonstrates that fast, context-driven reasoning can coexist with slow, structured generalization, providing a pathway for robust continual learning. 

---
# UP-Fuse: Uncertainty-guided LiDAR-Camera Fusion for 3D Panoptic Segmentation 

**Authors**: Rohit Mohan, Florian Drews, Yakov Miron, Daniele Cattaneo, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2602.19349)  

**Abstract**: LiDAR-camera fusion enhances 3D panoptic segmentation by leveraging camera images to complement sparse LiDAR scans, but it also introduces a critical failure mode. Under adverse conditions, degradation or failure of the camera sensor can significantly compromise the reliability of the perception system. To address this problem, we introduce UP-Fuse, a novel uncertainty-aware fusion framework in the 2D range-view that remains robust under camera sensor degradation, calibration drift, and sensor failure. Raw LiDAR data is first projected into the range-view and encoded by a LiDAR encoder, while camera features are simultaneously extracted and projected into the same shared space. At its core, UP-Fuse employs an uncertainty-guided fusion module that dynamically modulates cross-modal interaction using predicted uncertainty maps. These maps are learned by quantifying representational divergence under diverse visual degradations, ensuring that only reliable visual cues influence the fused representation. The fused range-view features are decoded by a novel hybrid 2D-3D transformer that mitigates spatial ambiguities inherent to the 2D projection and directly predicts 3D panoptic segmentation masks. Extensive experiments on Panoptic nuScenes, SemanticKITTI, and our introduced Panoptic Waymo benchmark demonstrate the efficacy and robustness of UP-Fuse, which maintains strong performance even under severe visual corruption or misalignment, making it well suited for robotic perception in safety-critical settings. 

---
# MultiDiffSense: Diffusion-Based Multi-Modal Visuo-Tactile Image Generation Conditioned on Object Shape and Contact Pose 

**Authors**: Sirine Bhouri, Lan Wei, Jian-Qing Zheng, Dandan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.19348)  

**Abstract**: Acquiring aligned visuo-tactile datasets is slow and costly, requiring specialised hardware and large-scale data collection. Synthetic generation is promising, but prior methods are typically single-modality, limiting cross-modal learning. We present MultiDiffSense, a unified diffusion model that synthesises images for multiple vision-based tactile sensors (ViTac, TacTip, ViTacTip) within a single architecture. Our approach uses dual conditioning on CAD-derived, pose-aligned depth maps and structured prompts that encode sensor type and 4-DoF contact pose, enabling controllable, physically consistent multi-modal synthesis. Evaluating on 8 objects (5 seen, 3 novel) and unseen poses, MultiDiffSense outperforms a Pix2Pix cGAN baseline in SSIM by +36.3% (ViTac), +134.6% (ViTacTip), and +64.7% (TacTip). For downstream 3-DoF pose estimation, mixing 50% synthetic with 50% real halves the required real data while maintaining competitive performance. MultiDiffSense alleviates the data-collection bottleneck in tactile sensing and enables scalable, controllable multi-modal dataset generation for robotic applications. 

---
# Smooth Gate Functions for Soft Advantage Policy Optimization 

**Authors**: Egor Denisov, Svetlana Glazyrina, Maksim Kryzhanovskiy, Roman Ischenko  

**Link**: [PDF](https://arxiv.org/pdf/2602.19345)  

**Abstract**: Group Relative Policy Optimization (GRPO) has significantly advanced the training of large language models and enhanced their reasoning capabilities, while it remains susceptible to instability due to the use of hard clipping. Soft Adaptive Policy Optimization (SAPO) addresses this limitation by replacing clipping with a smooth sigmoid-based gate function, which leads to more stable updates. We have decided to push this theory further and investigate the impact of different gate functions on both training stability and final model performance. We formalize the key properties that admissible gates should satisfy and identify several families of such functions for empirical evaluation. This paper presents an analysis of our findings based on experiments conducted with the Qwen2.5-7B-Instruct model on mathematical reasoning tasks. These results provide practical guidance for designing smoother and more robust policy optimization objectives for large language model training. 

---
# Soft Sequence Policy Optimization: Bridging GMPO and SAPO 

**Authors**: Svetlana Glazyrina, Maksim Kryzhanovskiy, Roman Ischenko  

**Link**: [PDF](https://arxiv.org/pdf/2602.19327)  

**Abstract**: A significant portion of recent research on Large Language Model (LLM) alignment focuses on developing new policy optimization methods based on Group Relative Policy Optimization (GRPO). Two prominent directions have emerged: (i) a shift toward sequence-level importance sampling weights that better align with the sequence-level rewards used in many tasks, and (ii) alternatives to PPO-style clipping that aim to avoid the associated loss of training signal and entropy collapse. Recent work, such as Soft Adaptive Policy Optimization (SAPO), reformulates the Scopic objective within the GRPO framework and achieves both sequence coherence and token adaptivity. Geometric-Mean Policy Optimization (GMPO) leverages token-wise ratio clipping within sequence importance sampling weights. Building on these ideas, this work proposes a new objective that promotes effective policy exploration while maintaining training stability. Specifically, we introduce Soft Sequence Policy Optimization, an off-policy reinforcement learning objective that incorporates soft gating functions over token-level probability ratios within sequence-level importance weights. 

---
# City Editing: Hierarchical Agentic Execution for Dependency-Aware Urban Geospatial Modification 

**Authors**: Rui Liu, Steven Jige Quan, Zhong-Ren Peng, Zijun Yao, Han Wang, Zhengzhang Chen, Kunpeng Liu, Yanjie Fu, Dongjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.19326)  

**Abstract**: As cities evolve over time, challenges such as traffic congestion and functional imbalance increasingly necessitate urban renewal through efficient modification of existing plans, rather than complete re-planning. In practice, even minor urban changes require substantial manual effort to redraw geospatial layouts, slowing the iterative planning and decision-making procedure. Motivated by recent advances in agentic systems and multimodal reasoning, we formulate urban renewal as a machine-executable task that iteratively modifies existing urban plans represented in structured geospatial formats. More specifically, we represent urban layouts using GeoJSON and decompose natural-language editing instructions into hierarchical geometric intents spanning polygon-, line-, and point-level operations. To coordinate interdependent edits across spatial elements and abstraction levels, we propose a hierarchical agentic framework that jointly performs multi-level planning and execution with explicit propagation of intermediate spatial constraints. We further introduce an iterative execution-validation mechanism that mitigates error accumulation and enforces global spatial consistency during multi-step editing. Extensive experiments across diverse urban editing scenarios demonstrate significant improvements in efficiency, robustness, correctness, and spatial validity over existing baselines. 

---
# RetinaVision: XAI-Driven Augmented Regulation for Precise Retinal Disease Classification using deep learning framework 

**Authors**: Mohammad Tahmid Noor, Shayan Abrar, Jannatul Adan Mahi, Md Parvez Mia, Asaduzzaman Hridoy, Samanta Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2602.19324)  

**Abstract**: Early and accurate classification of retinal diseases is critical to counter vision loss and for guiding clinical management of retinal diseases. In this study, we proposed a deep learning method for retinal disease classification utilizing optical coherence tomography (OCT) images from the Retinal OCT Image Classification - C8 dataset (comprising 24,000 labeled images spanning eight conditions). Images were resized to 224x224 px and tested on convolutional neural network (CNN) architectures: Xception and InceptionV3. Data augmentation techniques (CutMix, MixUp) were employed to enhance model generalization. Additionally, we applied GradCAM and LIME for interpretability evaluation. We implemented this in a real-world scenario via our web application named RetinaVision. This study found that Xception was the most accurate network (95.25%), followed closely by InceptionV3 (94.82%). These results suggest that deep learning methods allow effective OCT retinal disease classification and highlight the importance of implementing accuracy and interpretability for clinical applications. 

---
# US-JEPA: A Joint Embedding Predictive Architecture for Medical Ultrasound 

**Authors**: Ashwath Radhachandran, Vedrana Ivezić, Shreeram Athreya, Ronit Anilkumar, Corey W. Arnold, William Speier  

**Link**: [PDF](https://arxiv.org/pdf/2602.19322)  

**Abstract**: Ultrasound (US) imaging poses unique challenges for representation learning due to its inherently noisy acquisition process. The low signal-to-noise ratio and stochastic speckle patterns hinder standard self-supervised learning methods relying on a pixel-level reconstruction objective. Joint-Embedding Predictive Architectures (JEPAs) address this drawback by predicting masked latent representations rather than raw pixels. However, standard approaches depend on hyperparameter-brittle and computationally expensive online teachers updated via exponential moving average. We propose US-JEPA, a self-supervised framework that adopts the Static-teacher Asymmetric Latent Training (SALT) objective. By using a frozen, domain-specific teacher to provide stable latent targets, US-JEPA decouples student-teacher optimization and pushes the student to expand upon the semantic priors of the teacher. In addition, we provide the first rigorous comparison of all publicly available state-of-the-art ultrasound foundation models on UltraBench, a public dataset benchmark spanning multiple organs and pathological conditions. Under linear probing for diverse classification tasks, US-JEPA achieves performance competitive with or superior to domain-specific and universal vision foundation model baselines. Our results demonstrate that masked latent prediction provides a stable and efficient path toward robust ultrasound representations. 

---
# Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations 

**Authors**: Dongming Jiang, Yi Li, Songtao Wei, Jinxin Yang, Ayushi Kishore, Alysa Zhao, Dingyi Kang, Xu Hu, Feng Chen, Qiannan Li, Bingzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.19320)  

**Abstract**: Agentic memory systems enable large language model (LLM) agents to maintain state across long interactions, supporting long-horizon reasoning and personalization beyond fixed context windows. Despite rapid architectural development, the empirical foundations of these systems remain fragile: existing benchmarks are often underscaled, evaluation metrics are misaligned with semantic utility, performance varies significantly across backbone models, and system-level costs are frequently overlooked. This survey presents a structured analysis of agentic memory from both architectural and system perspectives. We first introduce a concise taxonomy of MAG systems based on four memory structures. Then, we analyze key pain points limiting current systems, including benchmark saturation effects, metric validity and judge sensitivity, backbone-dependent accuracy, and the latency and throughput overhead introduced by memory maintenance. By connecting the memory structure to empirical limitations, this survey clarifies why current agentic memory systems often underperform their theoretical promise and outlines directions for more reliable evaluation and scalable system design. 

---
# Health+: Empowering Individuals via Unifying Health Data 

**Authors**: Sujaya Maiyya, Shantanu Sharma, Avinash Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2602.19319)  

**Abstract**: Managing personal health data is a challenge in today's fragmented and institution-centric healthcare ecosystem. Individuals often lack meaningful control over their medical records, which are scattered across incompatible systems and formats. This vision paper presents Health+, a user-centric, multimodal health data management system that empowers individuals (including those with limited technical expertise) to upload, query, and share their data across modalities (e.g., text, images, reports). Rather than aiming for institutional overhaul, Health+ emphasizes individual agency by providing intuitive interfaces and intelligent recommendations for data access and sharing. At the system level, it tackles the complexity of storing, integrating, and securing heterogeneous health records, ensuring both efficiency and privacy. By unifying multimodal data and prioritizing patients, Health+ lays the foundation for a more connected, interpretable, and user-controlled health information ecosystem. 

---
# Learning to Reason for Multi-Step Retrieval of Personal Context in Personalized Question Answering 

**Authors**: Maryam Amirizaniani, Alireza Salemi, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2602.19317)  

**Abstract**: Personalization in Question Answering (QA) requires answers that are both accurate and aligned with users' background, preferences, and historical context. Existing state-of-the-art methods primarily rely on retrieval-augmented generation (RAG) solutions that construct personal context by retrieving relevant items from the user's profile. Existing methods use the user's query directly to retrieve personal documents, and such strategies often lead to surface-level personalization. We propose PR2 (Personalized Retrieval-Augmented Reasoning), a reinforcement learning framework that integrates reasoning and retrieval from personal context for personalization. PR2 learns adaptive retrieval-reasoning policies, determining when to retrieve, what evidence to retrieve from user profiles, and how to incorporate it into intermediate reasoning steps. By optimizing multi-turn reasoning trajectories under a personalized reward function, the framework reinforces reasoning paths that better align with user-specific preferences and contextual signals reflected by the reward model. Extensive experiments on the LaMP-QA benchmark using three LLMs show that PR2 consistently outperforms strong baselines, achieving an average relative improvement of 8.8%-12% in personalized QA. 

---
# Online Navigation Planning for Long-term Autonomous Operation of Underwater Gliders 

**Authors**: Victor-Alexandru Darvariu, Charlotte Z. Reed, Jan Stratmann, Bruno Lacerda, Benjamin Allsup, Stephen Woodward, Elizabeth Siddle, Trishna Saeharaseelan, Owain Jones, Dan Jones, Tobias Ferreira, Chloe Baker, Kevin Chaplin, James Kirk, Ashley Morris, Ryan Patmore, Jeff Polton, Charlotte Williams, Alexandra Kokkinaki, Alvaro Lorenzo Lopez, Justin J. H. Buck, Nick Hawes  

**Link**: [PDF](https://arxiv.org/pdf/2602.19315)  

**Abstract**: Underwater glider robots have become an indispensable tool for ocean sampling. Although stakeholders are calling for tools to manage increasingly large fleets of gliders, successful autonomous long-term deployments have thus far been scarce, which hints at a lack of suitable methodologies and systems. In this work, we formulate glider navigation planning as a stochastic shortest-path Markov Decision Process and propose a sample-based online planner based on Monte Carlo Tree Search. Samples are generated by a physics-informed simulator that captures uncertain execution of controls and ocean current forecasts while remaining computationally tractable. The simulator parameters are fitted using historical glider data. We integrate these methods into an autonomous command-and-control system for Slocum gliders that enables closed-loop replanning at each surfacing. The resulting system was validated in two field deployments in the North Sea totalling approximately 3 months and 1000 km of autonomous operation. Results demonstrate improved efficiency compared to straight-to-goal navigation and show the practicality of sample-based planning for long-term marine autonomy. 

---
# IPv2: An Improved Image Purification Strategy for Real-World Ultra-Low-Dose Lung CT Denoising 

**Authors**: Guoliang Gong, Man Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19314)  

**Abstract**: The image purification strategy constructs an intermediate distribution with aligned anatomical structures, which effectively corrects the spatial misalignment between real-world ultra-low-dose CT and normal-dose CT images and significantly enhances the structural preservation ability of denoising models. However, this strategy exhibits two inherent limitations. First, it suppresses noise only in the chest wall and bone regions while leaving the image background untreated. Second, it lacks a dedicated mechanism for denoising the lung parenchyma. To address these issues, we systematically redesign the original image purification strategy and propose an improved version termed IPv2. The proposed strategy introduces three core modules, namely Remove Background, Add noise, and Remove noise. These modules endow the model with denoising capability in both background and lung tissue regions during training data construction and provide a more reasonable evaluation protocol through refined label construction at the testing stage. Extensive experiments on our previously established real-world patient lung CT dataset acquired at 2% radiation dose demonstrate that IPv2 consistently improves background suppression and lung parenchyma restoration across multiple mainstream denoising models. The code is publicly available at this https URL. 

---
# TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics 

**Authors**: Shirui Chen, Cole Harrison, Ying-Chun Lee, Angela Jin Yang, Zhongzheng Ren, Lillian J. Ratliff, Jiafei Duan, Dieter Fox, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2602.19313)  

**Abstract**: While Vision-Language-Action (VLA) models have seen rapid progress in pretraining, their advancement in Reinforcement Learning (RL) remains hampered by low sample efficiency and sparse rewards in real-world settings. Developing generalizable process reward models is essential for providing the fine-grained feedback necessary to bridge this gap, yet existing temporal value functions often fail to generalize beyond their training domains. We introduce TOPReward, a novel, probabilistically grounded temporal value function that leverages the latent world knowledge of pretrained video Vision-Language Models (VLMs) to estimate robotic task progress. Unlike prior methods that prompt VLMs to directly output progress values, which are prone to numerical misrepresentation, TOPReward extracts task progress directly from the VLM's internal token logits. In zero-shot evaluations across 130+ distinct real-world tasks and multiple robot platforms (e.g., Franka, YAM, SO-100/101), TOPReward achieves 0.947 mean Value-Order Correlation (VOC) on Qwen3-VL, dramatically outperforming the state-of-the-art GVL baseline which achieves near-zero correlation on the same open-source model. We further demonstrate that TOPReward serves as a versatile tool for downstream applications, including success detection and reward-aligned behavior cloning. 

---
# Safe and Interpretable Multimodal Path Planning for Multi-Agent Cooperation 

**Authors**: Haojun Shi, Suyu Ye, Katherine M. Guerrerio, Jianzhi Shen, Yifan Yin, Daniel Khashabi, Chien-Ming Huang, Tianmin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19304)  

**Abstract**: Successful cooperation among decentralized agents requires each agent to quickly adapt its plan to the behavior of other agents. In scenarios where agents cannot confidently predict one another's intentions and plans, language communication can be crucial for ensuring safety. In this work, we focus on path-level cooperation in which agents must adapt their paths to one another in order to avoid collisions or perform physical collaboration such as joint carrying. In particular, we propose a safe and interpretable multimodal path planning method, CaPE (Code as Path Editor), which generates and updates path plans for an agent based on the environment and language communication from other agents. CaPE leverages a vision-language model (VLM) to synthesize a path editing program verified by a model-based planner, grounding communication to path plan updates in a safe and interpretable way. We evaluate our approach in diverse simulated and real-world scenarios, including multi-robot and human-robot cooperation in autonomous driving, household, and joint carrying tasks. Experimental results demonstrate that CaPE can be integrated into different robotic systems as a plug-and-play module, greatly enhancing a robot's ability to align its plan to language communication from other robots or humans. We also show that the combination of the VLM-based path editing program synthesis and model-based planning safety enables robots to achieve open-ended cooperation while maintaining safety and interpretability. 

---
# Taming Preconditioner Drift: Unlocking the Potential of Second-Order Optimizers for Federated Learning on Non-IID Data 

**Authors**: Junkang Liu, Fanhua Shang, Hongying Liu, Jin Liu, Weixin An, Yuanyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19271)  

**Abstract**: Second-order optimizers can significantly accelerate large-scale training, yet their naive federated variants are often unstable or even diverge on non-IID data.
We show that a key culprit is \emph{preconditioner drift}: client-side second-order training induces heterogeneous \emph{curvature-defined geometries} (i.e., preconditioner coordinate systems), and server-side model averaging updates computed under incompatible metrics, corrupting the global descent direction.
To address this geometric mismatch, we propose \texttt{FedPAC}, a \emph{preconditioner alignment and correction} framework for reliable federated second-order optimization.
\texttt{FedPAC} explicitly decouples parameter aggregation from geometry synchronization by:
(i) \textbf{Alignment} (i.e.,aggregating local preconditioners into a global reference and warm-starting clients via global preconditioner); and
(ii) \textbf{Correction} (i.e., steering local preconditioned updates using a global preconditioned direction to suppress long-term drift).
We provide drift-coupled non-convex convergence guarantees with linear speedup under partial participation.
Empirically, \texttt{FedPAC} consistently improves stability and accuracy across vision and language tasks, achieving up to $5.8\%$ absolute accuracy gain on CIFAR-100 with ViTs.
Code is available at this https URL. 

---
# CORVET: A CORDIC-Powered, Resource-Frugal Mixed-Precision Vector Processing Engine for High-Throughput AIoT applications 

**Authors**: Sonu Kumar, Mohd Faisal Khan, Mukul Lokhande, Santosh Kumar Vishvakarma  

**Link**: [PDF](https://arxiv.org/pdf/2602.19268)  

**Abstract**: This brief presents a runtime-adaptive, performance-enhanced vector engine featuring a low-resource, iterative CORDIC-based MAC unit for edge AI acceleration. The proposed design enables dynamic reconfiguration between approximate and accurate modes, exploiting the latency-accuracy trade-off for a wide range of workloads. Its resource-efficient approach further enables up to 4x throughput improvement within the same hardware resources by leveraging vectorised, time-multiplexed execution and flexible precision scaling. With a time-multiplexed multi-AF block and a lightweight pooling and normalisation unit, the proposed vector engine supports flexible precision (4/8/16-bit) and high MAC density. The ASIC implementation results show that each MAC stage can save up to 33% of time and 21% of power, with a 256-PE configuration that achieves higher compute density (4.83 TOPS/mm2 ) and energy efficiency (11.67 TOPS/W) than previous state-of-the-art work. A detailed hardware-software co-design methodology for object detection and classification tasks on Pynq-Z2 is discussed to assess the proposed architecture, demonstrating a scalable, energy-efficient solution for edge AI applications. 

---
# DGPO: RL-Steered Graph Diffusion for Neural Architecture Generation 

**Authors**: Aleksei Liuliakov, Luca Hermes, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2602.19261)  

**Abstract**: Reinforcement learning fine-tuning has proven effective for steering generative diffusion models toward desired properties in image and molecular domains. Graph diffusion models have similarly been applied to combinatorial structure generation, including neural architecture search (NAS). However, neural architectures are directed acyclic graphs (DAGs) where edge direction encodes functional semantics such as data flow-information that existing graph diffusion methods, designed for undirected structures, discard. We propose Directed Graph Policy Optimization (DGPO), which extends reinforcement learning fine-tuning of discrete graph diffusion models to DAGs via topological node ordering and positional encoding. Validated on NAS-Bench-101 and NAS-Bench-201, DGPO matches the benchmark optimum on all three NAS-Bench-201 tasks (91.61%, 73.49%, 46.77%). The central finding is that the model learns transferable structural priors: pretrained on only 7% of the search space, it generates near-oracle architectures after fine-tuning, within 0.32 percentage points of the full-data model and extrapolating 7.3 percentage points beyond its training ceiling. Bidirectional control experiments confirm genuine reward-driven steering, with inverse optimization reaching near random-chance accuracy (9.5%). These results demonstrate that reinforcement learning-steered discrete diffusion, once extended to handle directionality, provides a controllable generative framework for directed combinatorial structures. 

---
# No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection 

**Authors**: Zunkai Dai, Ke Li, Jiajia Liu, Jie Yang, Yuanyuan Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2602.19248)  

**Abstract**: The collection and detection of video anomaly data has long been a challenging problem due to its rare occurrence and spatio-temporal scarcity. Existing video anomaly detection (VAD) methods under perform in open-world scenarios. Key contributing factors include limited dataset diversity, and inadequate understanding of context-dependent anomalous semantics. To address these issues, i) we propose LAVIDA, an end-to-end zero-shot video anomaly detection framework. ii) LAVIDA employs an Anomaly Exposure Sampler that transforms segmented objects into pseudo-anomalies to enhance model adaptability to unseen anomaly categories. It further integrates a Multimodal Large Language Model (MLLM) to bolster semantic comprehension capabilities. Additionally, iii) we design a token compression approach based on reverse attention to handle the spatio-temporal scarcity of anomalous patterns and decrease computational cost. The training process is conducted solely on pseudo anomalies without any VAD data. Evaluations across four benchmark VAD datasets demonstrate that LAVIDA achieves SOTA performance in both frame-level and pixel-level anomaly detection under the zero-shot setting. Our code is available in this https URL. 

---
# Scaling Laws for Precision in High-Dimensional Linear Regression 

**Authors**: Dechen Zhang, Xuan Tang, Yingyu Liang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2602.19241)  

**Abstract**: Low-precision training is critical for optimizing the trade-off between model quality and training costs, necessitating the joint allocation of model size, dataset size, and numerical precision. While empirical scaling laws suggest that quantization impacts effective model and data capacities or acts as an additive error, the theoretical mechanisms governing these effects remain largely unexplored. In this work, we initiate a theoretical study of scaling laws for low-precision training within a high-dimensional sketched linear regression framework. By analyzing multiplicative (signal-dependent) and additive (signal-independent) quantization, we identify a critical dichotomy in their scaling behaviors. Our analysis reveals that while both schemes introduce an additive error and degrade the effective data size, they exhibit distinct effects on effective model size: multiplicative quantization maintains the full-precision model size, whereas additive quantization reduces the effective model size. Numerical experiments validate our theoretical findings. By rigorously characterizing the complex interplay among model scale, dataset size, and quantization error, our work provides a principled theoretical basis for optimizing training protocols under practical hardware constraints. 

---
# Evaluating SAP RPT-1 for Enterprise Business Process Prediction: In-Context Learning vs. Traditional Machine Learning on Structured SAP Data 

**Authors**: Amit Lal  

**Link**: [PDF](https://arxiv.org/pdf/2602.19237)  

**Abstract**: Tabular foundation models aim to make machine learning accessible for enterprise data without task-specific training. This paper presents the first independent evaluation of SAP's Retrieval Pretrained Transformer (RPT-1) from a practitioner perspective. RPT-1 is a compact 64.6 MB model pretrained on 1.34 TB of structured data across 3.1 million tables. We benchmark it against tuned gradient-boosted decision trees (XGBoost, LightGBM, CatBoost) on three SAP business scenarios: demand forecasting across SD/MM/PP modules, predictive data integrity in BC/MM/QM, and financial risk classification in FI/CO/AR. Across five-fold cross-validation on datasets ranging from 2,500 to 3,200 rows, RPT-1 reaches 91-96% of tuned GBDT accuracy without any training examples. The classification gap is modest at 3.6-4.1 percentage points on AUC-ROC, though regression tasks show wider gaps of 8.9-11.1 percentage points on R-squared. An interesting finding is a crossover at roughly 75-100 context rows where RPT-1 actually outperforms XGBoost under limited data. Based on these results, we propose a practical hybrid workflow: use RPT-1 for rapid screening, then train GBDT selectively where prediction accuracy justifies the effort. All experiments are reproducible through publicly available Hugging Face Spaces. 

---
# How to Allocate, How to Learn? Dynamic Rollout Allocation and Advantage Modulation for Policy Optimization 

**Authors**: Yangyi Fang, Jiaye Lin, Xiaoliang Fu, Cong Qin, Haolin Shi, Chaowen Hu, Lu Pan, Ke Zeng, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2602.19208)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for Large Language Model (LLM) reasoning, yet current methods face key challenges in resource allocation and policy optimization dynamics: (i) uniform rollout allocation ignores gradient variance heterogeneity across problems, and (ii) the softmax policy structure causes gradient attenuation for high-confidence correct actions, while excessive gradient updates may destabilize training. Therefore, we propose DynaMO, a theoretically-grounded dual-pronged optimization framework. At the sequence level, we prove that uniform allocation is suboptimal and derive variance-minimizing allocation from the first principle, establishing Bernoulli variance as a computable proxy for gradient informativeness. At the token level, we develop gradient-aware advantage modulation grounded in theoretical analysis of gradient magnitude bounds. Our framework compensates for gradient attenuation of high-confidence correct actions while utilizing entropy changes as computable indicators to stabilize excessive update magnitudes. Extensive experiments conducted on a diverse range of mathematical reasoning benchmarks demonstrate consistent improvements over strong RLVR baselines. Our implementation is available at: \href{this https URL}{this https URL}. 

---
# HybridFL: A Federated Learning Approach for Financial Crime Detection 

**Authors**: Afsana Khan, Marijn ten Thij, Guangzhi Tang, Anna Wilbik  

**Link**: [PDF](https://arxiv.org/pdf/2602.19207)  

**Abstract**: Federated learning (FL) is a privacy-preserving machine learning paradigm that enables multiple parties to collaboratively train models on privately owned data without sharing raw information. While standard FL typically addresses either horizontal or vertical data partitions, many real-world scenarios exhibit a complex hybrid distribution. This paper proposes Hybrid Federated Learning (HybridFL) to address data split both horizontally across disjoint users and vertically across complementary feature sets. We evaluate HybridFL in a financial crime detection context, where a transaction party holds transaction-level attributes and multiple banks maintain private account-level features. By integrating horizontal aggregation and vertical feature fusion, the proposed architecture enables joint learning while strictly preserving data locality. Experiments on AMLSim and SWIFT datasets demonstrate that HybridFL significantly outperforms the transaction-only local model and achieves performance comparable to a centralized benchmark. 

---
# Visual Prompt Guided Unified Pushing Policy 

**Authors**: Hieu Bui, Ziyan Gao, Yuya Hosoda, Joo-Ho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.19193)  

**Abstract**: As one of the simplest non-prehensile manipulation skills, pushing has been widely studied as an effective means to rearrange objects. Existing approaches, however, typically rely on multi-step push plans composed of pre-defined pushing primitives with limited application scopes, which restrict their efficiency and versatility across different scenarios. In this work, we propose a unified pushing policy that incorporates a lightweight prompting mechanism into a flow matching policy to guide the generation of reactive, multimodal pushing actions. The visual prompt can be specified by a high-level planner, enabling the reuse of the pushing policy across a wide range of planning problems. Experimental results demonstrate that the proposed unified pushing policy not only outperforms existing baselines but also effectively serves as a low-level primitive within a VLM-guided planning framework to solve table-cleaning tasks efficiently. 

---
# FUSAR-GPT : A Spatiotemporal Feature-Embedded and Two-Stage Decoupled Visual Language Model for SAR Imagery 

**Authors**: Xiaokun Zhang, Yi Yang, Ziqi Ye, Baiyun, Xiaorong Guo, Qingchen Fang, Ruyi Zhang, Xinpeng Zhou, Haipeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.19190)  

**Abstract**: Research on the intelligent interpretation of all-weather, all-time Synthetic Aperture Radar (SAR) is crucial for advancing remote sensing applications. In recent years, although Visual Language Models (VLMs) have demonstrated strong open-world understanding capabilities on RGB images, their performance is severely limited when directly applied to the SAR field due to the complexity of the imaging mechanism, sensitivity to scattering features, and the scarcity of high-quality text corpora. To systematically address this issue, we constructed the inaugural SAR Image-Text-AlphaEarth feature triplet dataset and developed FUSAR-GPT, a VLM specifically for SAR. FUSAR-GPT innovatively introduces a geospatial baseline model as a 'world knowledge' prior and embeds multi-source remote-sensing temporal features into the model's visual backbone via 'spatiotemporal anchors', enabling dynamic compensation for the sparse representation of targets in SAR images. Furthermore, we designed a two-stage SFT strategy to decouple the knowledge injection and task execution of large models. The spatiotemporal feature embedding and the two-stage decoupling paradigm enable FUSAR-GPT to achieve state-of-the-art performance across several typical remote sensing visual-language benchmark tests, significantly outperforming mainstream baseline models by over 12%. 

---
# Next Reply Prediction X Dataset: Linguistic Discrepancies in Naively Generated Content 

**Authors**: Simon Münker, Nils Schwager, Kai Kugler, Michael Heseltine, Achim Rettinger  

**Link**: [PDF](https://arxiv.org/pdf/2602.19177)  

**Abstract**: The increasing use of Large Language Models (LLMs) as proxies for human participants in social science research presents a promising, yet methodologically risky, paradigm shift. While LLMs offer scalability and cost-efficiency, their "naive" application, where they are prompted to generate content without explicit behavioral constraints, introduces significant linguistic discrepancies that challenge the validity of research findings. This paper addresses these limitations by introducing a novel, history-conditioned reply prediction task on authentic X (formerly Twitter) data, to create a dataset designed to evaluate the linguistic output of LLMs against human-generated content. We analyze these discrepancies using stylistic and content-based metrics, providing a quantitative framework for researchers to assess the quality and authenticity of synthetic data. Our findings highlight the need for more sophisticated prompting techniques and specialized datasets to ensure that LLM-generated content accurately reflects the complex linguistic patterns of human communication, thereby improving the validity of computational social science studies. 

---
# HistCAD: Geometrically Constrained Parametric History-based CAD Dataset 

**Authors**: Xintong Dong, Chuanyang Li, Chuqi Han, Peng Zheng, Jiaxin Jing, Yanzhi Song, Zhouwang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.19171)  

**Abstract**: Parametric computer-aided design (CAD) modeling is fundamental to industrial design, but existing datasets often lack explicit geometric constraints and fine-grained functional semantics, limiting editable, constraint-compliant generation. We present HistCAD, a large-scale dataset featuring constraint-aware modeling sequences that compactly represent procedural operations while ensuring compatibility with native CAD software, encompassing five aligned modalities: modeling sequences, multi-view renderings, STEP-format B-reps, native parametric files, and textual annotations. We develop AM\(_\text{HistCAD}\), an annotation module that extracts geometric and spatial features from modeling sequences and uses a large language model to generate complementary annotations of the modeling process, geometric structure, and functional type. Extensive evaluations demonstrate that HistCAD's explicit constraints, flattened sequence format, and multi-type annotations improve robustness, parametric editability, and accuracy in text-driven CAD generation, while industrial parts included in HistCAD further support complex real-world design scenarios. HistCAD thus provides a unified benchmark for advancing editable, constraint-aware, and semantically enriched generative CAD modeling. 

---
# Virtual Parameter Sharpening: Dynamic Low-Rank Perturbations for Inference-Time Reasoning Enhancement 

**Authors**: Saba Kublashvili  

**Link**: [PDF](https://arxiv.org/pdf/2602.19169)  

**Abstract**: I introduce Virtual Parameter Sharpening (VPS), an inference-time technique that augments frozen transformer linear layers with dynamic, activation-conditioned low-rank perturbations. Unlike parameter-efficient fine-tuning methods such as LoRA, which learn static low-rank adapters, VPS constructs its perturbation factors on the fly from batch activation statistics and optional gradient signals, enabling test-time adaptation without persistent parameter updates. The perturbation takes the form Delta W = gamma * W^T V U^T W, where selector matrices U and V are constructed via sparse activation-guided selection or Sylvester-coupled regression. We provide a theoretical analysis of the perturbation's spectral properties and describe an adaptive policy system that modulates perturbation magnitude based on activation energy and token-level entropy. This system incorporates multi-objective verification with iterative refinement for tasks with ground-truth supervision. We present the complete algorithmic framework, analyze its mathematical foundations, and discuss the mechanisms by which activation-conditioned computation may enhance reasoning capabilities in large language models. Implementation and experimental code are available at this https URL . 

---
# CosyAccent: Duration-Controllable Accent Normalization Using Source-Synthesis Training Data 

**Authors**: Qibing Bai, Shuhao Shi, Shuai Wang, Yukai Ju, Yannan Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.19166)  

**Abstract**: Accent normalization (AN) systems often struggle with unnatural outputs and undesired content distortion, stemming from both suboptimal training data and rigid duration modeling. In this paper, we propose a "source-synthesis" methodology for training data construction. By generating source L2 speech and using authentic native speech as the training target, our approach avoids learning from TTS artifacts and, crucially, requires no real L2 data in training. Alongside this data strategy, we introduce CosyAccent, a non-autoregressive model that resolves the trade-off between prosodic naturalness and duration control. CosyAccent implicitly models rhythm for flexibility yet offers explicit control over total output duration. Experiments show that, despite being trained without any real L2 speech, CosyAccent achieves significantly improved content preservation and superior naturalness compared to strong baselines trained on real-world data. 

---
# Artefact-Aware Fungal Detection in Dermatophytosis: A Real-Time Transformer-Based Approach for KOH Microscopy 

**Authors**: Rana Gursoy, Abdurrahim Yilmaz, Baris Kizilyaprak, Esmahan Caglar, Burak Temelkuran, Huseyin Uvet, Ayse Esra Koku Aksu, Gulsum Gencoglan  

**Link**: [PDF](https://arxiv.org/pdf/2602.19156)  

**Abstract**: Dermatophytosis is commonly assessed using potassium hydroxide (KOH) microscopy, yet accurate recognition of fungal hyphae is hindered by artefacts, heterogeneous keratin clearance, and notable inter-observer variability. This study presents a transformer-based detection framework using the RT-DETR model architecture to achieve precise, query-driven localization of fungal structures in high-resolution KOH images. A dataset of 2,540 routinely acquired microscopy images was manually annotated using a multi-class strategy to explicitly distinguish fungal elements from confounding artefacts. The model was trained with morphology-preserving augmentations to maintain the structural integrity of thin hyphae. Evaluation on an independent test set demonstrated robust object-level performance, with a recall of 0.9737, precision of 0.8043, and an AP@0.50 of 93.56%. When aggregated for image-level diagnosis, the model achieved 100% sensitivity and 98.8% accuracy, correctly identifying all positive cases without missing a single diagnosis. Qualitative outputs confirmed the robust localization of low-contrast hyphae even in artefact-rich fields. These results highlight that an artificial intelligence (AI) system can serve as a highly reliable, automated screening tool, effectively bridging the gap between image-level analysis and clinical decision-making in dermatomycology. 

---
# Constrained Diffusion for Accelerated Structure Relaxation of Inorganic Solids with Point Defects 

**Authors**: Jingyi Cui, Jacob K. Christopher, Ankita Biswas, Prasanna V. Balachandran, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2602.19153)  

**Abstract**: Point defects affect material properties by altering electronic states and modifying local bonding environments. However, high-throughput first-principles simulations of point defects are costly due to large simulation cells and complex energy landscapes. To this end, we propose a generative framework for simulating point defects, overcoming the limits of costly first-principles simulators. By leveraging a primal-dual algorithm, we introduce a constraint-aware diffusion model which outperforms existing constrained diffusion approaches in this domain. Across six defect configuration settings for Bi2Te3, the proposed approach provides state-of-the-art performance generating physically grounded structures. 

---
# Celo2: Towards Learned Optimization Free Lunch 

**Authors**: Abhinav Moudgil, Boris Knyazev, Eugene Belilovsky  

**Link**: [PDF](https://arxiv.org/pdf/2602.19142)  

**Abstract**: Learned optimizers are powerful alternatives to hand-designed update rules like Adam, yet they have seen limited practical adoption since they often fail to meta-generalize beyond their training distribution and incur high meta-training cost. For instance, prior work, VeLO, scaled meta-training to 4,000 TPU months ($\sim$10$\times$ GPT-3 compute) to meta-train a general-purpose optimizer but it failed to generalize beyond 600M parameters tasks. In this work, we present a surprising finding: by crafting a simple normalized optimizer architecture and augmenting meta-training, it becomes feasible to meta-train a performant general-purpose learned update rule on a tiny fraction of VeLO compute, 4.5 GPU hours to be precise. Our learned update rule scales stably to a billion-scale pretraining task (GPT-3 XL 1.3B) which is six orders of magnitude larger than its meta-training distribution. Furthermore, it shows strong performance across diverse out-of-distribution tasks and is compatible with modern optimization harness that includes orthogonalization, distinct update rules for input-output and hidden weights, and decoupled weight decay. In all, this work paves the way for practically applicable learnable optimization algorithms, unlocking exploration of richer meta-training and data curation recipes to further improve performance. 

---
# CRCC: Contrast-Based Robust Cross-Subject and Cross-Site Representation Learning for EEG 

**Authors**: Xiaobin Wong, Zhonghua Zhao, Haoran Guo, Zhengyi Liu, Yu Wu, Feng Yan, Zhiren Wang, Sen Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.19138)  

**Abstract**: EEG-based neural decoding models often fail to generalize across acquisition sites due to structured, site-dependent biases implicitly exploited during training. We reformulate cross-site clinical EEG learning as a bias-factorized generalization problem, in which domain shifts arise from multiple interacting sources. We identify three fundamental bias factors and propose a general training framework that mitigates their influence through data standardization and representation-level constraints. We construct a standardized multi-site EEG benchmark for Major Depressive Disorder and introduce CRCC, a two-stage training paradigm combining encoder-decoder pretraining with joint fine-tuning via cross-subject/site contrastive learning and site-adversarial optimization. CRCC consistently outperforms state-of-the-art baselines and achieves a 10.7 percentage-point improvement in balanced accuracy under strict zero-shot site transfer, demonstrating robust generalization to unseen environments. 

---
# Test-Time Learning of Causal Structure from Interventional Data 

**Authors**: Wei Chen, Rui Ding, Bojun Huang, Yang Zhang, Qiang Fu, Yuxuan Liang, Han Shi, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.19131)  

**Abstract**: Supervised causal learning has shown promise in causal discovery, yet it often struggles with generalization across diverse interventional settings, particularly when intervention targets are unknown. To address this, we propose TICL (Test-time Interventional Causal Learning), a novel method that synergizes Test-Time Training with Joint Causal Inference. Specifically, we design a self-augmentation strategy to generate instance-specific training data at test time, effectively avoiding distribution shifts. Furthermore, by integrating joint causal inference, we developed a PC-inspired two-phase supervised learning scheme, which effectively leverages self-augmented training data while ensuring theoretical identifiability. Extensive experiments on bnlearn benchmarks demonstrate TICL's superiority in multiple aspects of causal discovery and intervention target detection. 

---
# Detecting labeling bias using influence functions 

**Authors**: Frida Jørgensen, Nina Weng, Siavash Bigdeli  

**Link**: [PDF](https://arxiv.org/pdf/2602.19130)  

**Abstract**: Labeling bias arises during data collection due to resource limitations or unconscious bias, leading to unequal label error rates across subgroups or misrepresentation of subgroup prevalence. Most fairness constraints assume training labels reflect the true distribution, rendering them ineffective when labeling bias is present; leaving a challenging question, that \textit{how can we detect such labeling bias?} In this work, we investigate whether influence functions can be used to detect labeling bias. Influence functions estimate how much each training sample affects a model's predictions by leveraging the gradient and Hessian of the loss function -- when labeling errors occur, influence functions can identify wrongly labeled samples in the training set, revealing the underlying failure mode. We develop a sample valuation pipeline and test it first on the MNIST dataset, then scaled to the more complex CheXpert medical imaging dataset. To examine label noise, we introduced controlled errors by flipping 20\% of the labels for one class in the dataset. Using a diagonal Hessian approximation, we demonstrated promising results, successfully detecting nearly 90\% of mislabeled samples in MNIST. On CheXpert, mislabeled samples consistently exhibit higher influence scores. These results highlight the potential of influence functions for identifying label errors. 

---
# How Do LLMs Encode Scientific Quality? An Empirical Study Using Monosemantic Features from Sparse Autoencoders 

**Authors**: Michael McCoubrey, Angelo Salatino, Francesco Osborne, Enrico Motta  

**Link**: [PDF](https://arxiv.org/pdf/2602.19115)  

**Abstract**: In recent years, there has been a growing use of generative AI, and large language models (LLMs) in particular, to support both the assessment and generation of scientific work. Although some studies have shown that LLMs can, to a certain extent, evaluate research according to perceived quality, our understanding of the internal mechanisms that enable this capability remains limited. This paper presents the first study that investigates how LLMs encode the concept of scientific quality through relevant monosemantic features extracted using sparse autoencoders. We derive such features under different experimental settings and assess their ability to serve as predictors across three tasks related to research quality: predicting citation count, journal SJR, and journal h-index. The results indicate that LLMs encode features associated with multiple dimensions of scientific quality. In particular, we identify four recurring types of features that capture key aspects of how research quality is represented: 1) features reflecting research methodologies; 2) features related to publication type, with literature reviews typically exhibiting higher impact; 3) features associated with high-impact research fields and technologies; and 4) features corresponding to specific scientific jargons. These findings represent an important step toward understanding how LLMs encapsulate concepts related to research quality. 

---
# Kaiwu-PyTorch-Plugin: Bridging Deep Learning and Photonic Quantum Computing for Energy-Based Models and Active Sample Selection 

**Authors**: Hongdong Zhu, Qi Gao, Yin Ma, Shaobo Chen, Haixu Liu, Fengao Wang, Tinglan Wang, Chang Wu, Kai Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.19114)  

**Abstract**: This paper introduces the Kaiwu-PyTorch-Plugin (KPP) to bridge Deep Learning and Photonic Quantum Computing across multiple dimensions. KPP integrates the Coherent Ising Machine into the PyTorch ecosystem, addressing classical inefficiencies in Energy-Based Models. The framework facilitates quantum integration in three key aspects: accelerating Boltzmann sampling, optimizing training data via Active Sampling, and constructing hybrid architectures like QBM-VAE and Q-Diffusion. Empirical results on single-cell and OpenWebText datasets demonstrate KPPs ability to achieve SOTA performance, validating a comprehensive quantum-classical paradigm. 

---
# Learning from Complexity: Exploring Dynamic Sample Pruning of Spatio-Temporal Training 

**Authors**: Wei Chen, Junle Chen, Yuqian Wu, Yuxuan Liang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.19113)  

**Abstract**: Spatio-temporal forecasting is fundamental to intelligent systems in transportation, climate science, and urban planning. However, training deep learning models on the massive, often redundant, datasets from these domains presents a significant computational bottleneck. Existing solutions typically focus on optimizing model architectures or optimizers, while overlooking the inherent inefficiency of the training data itself. This conventional approach of iterating over the entire static dataset each epoch wastes considerable resources on easy-to-learn or repetitive samples. In this paper, we explore a novel training-efficiency techniques, namely learning from complexity with dynamic sample pruning, ST-Prune, for spatio-temporal forecasting. Through dynamic sample pruning, we aim to intelligently identify the most informative samples based on the model's real-time learning state, thereby accelerating convergence and improving training efficiency. Extensive experiments conducted on real-world spatio-temporal datasets show that ST-Prune significantly accelerates the training speed while maintaining or even improving the model performance, and it also has scalability and universality. 

---
# Value Entanglement: Conflation Between Different Kinds of Good In (Some) Large Language Models 

**Authors**: Seong Hah Cho, Junyi Li, Anna Leshinskaya  

**Link**: [PDF](https://arxiv.org/pdf/2602.19101)  

**Abstract**: Value alignment of Large Language Models (LLMs) requires us to empirically measure these models' actual, acquired representation of value. Among the characteristics of value representation in humans is that they distinguish among value of different kinds. We investigate whether LLMs likewise distinguish three different kinds of good: moral, grammatical, and economic. By probing model behavior, embeddings, and residual stream activations, we report pervasive cases of value entanglement: a conflation between these distinct representations of value. Specifically, both grammatical and economic valuation was found to be overly influenced by moral value, relative to human norms. This conflation was repaired by selective ablation of the activation vectors associated with morality. 

---
# Detecting Cybersecurity Threats by Integrating Explainable AI with SHAP Interpretability and Strategic Data Sampling 

**Authors**: Norrakith Srisumrith, Sunantha Sodsee  

**Link**: [PDF](https://arxiv.org/pdf/2602.19087)  

**Abstract**: The critical need for transparent and trustworthy machine learning in cybersecurity operations drives the development of this integrated Explainable AI (XAI) framework. Our methodology addresses three fundamental challenges in deploying AI for threat detection: handling massive datasets through Strategic Sampling Methodology that preserves class distributions while enabling efficient model development; ensuring experimental rigor via Automated Data Leakage Prevention that systematically identifies and removes contaminated features; and providing operational transparency through Integrated XAI Implementation using SHAP analysis for model-agnostic interpretability across algorithms. Applied to the CIC-IDS2017 dataset, our approach maintains detection efficacy while reducing computational overhead and delivering actionable explanations for security analysts. The framework demonstrates that explainability, computational efficiency, and experimental integrity can be simultaneously achieved, providing a robust foundation for deploying trustworthy AI systems in security operations centers where decision transparency is paramount. 

---
# IDLM: Inverse-distilled Diffusion Language Models 

**Authors**: David Li, Nikita Gushchin, Dmitry Abulkhanov, Eric Moulines, Ivan Oseledets, Maxim Panov, Alexander Korotin  

**Link**: [PDF](https://arxiv.org/pdf/2602.19066)  

**Abstract**: Diffusion Language Models (DLMs) have recently achieved strong results in text generation. However, their multi-step sampling leads to slow inference, limiting practical use. To address this, we extend Inverse Distillation, a technique originally developed to accelerate continuous diffusion models, to the discrete setting. Nonetheless, this extension introduces both theoretical and practical challenges. From a theoretical perspective, the inverse distillation objective lacks uniqueness guarantees, which may lead to suboptimal solutions. From a practical standpoint, backpropagation in the discrete space is non-trivial and often unstable. To overcome these challenges, we first provide a theoretical result demonstrating that our inverse formulation admits a unique solution, thereby ensuring valid optimization. We then introduce gradient-stable relaxations to support effective training. As a result, experiments on multiple DLMs show that our method, Inverse-distilled Diffusion Language Models (IDLM), reduces the number of inference steps by 4x-64x, while preserving the teacher model's entropy and generative perplexity. 

---
# Adaptive Multi-Agent Reasoning for Text-to-Video Retrieval 

**Authors**: Jiaxin Wu, Xiao-Yong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.19040)  

**Abstract**: The rise of short-form video platforms and the emergence of multimodal large language models (MLLMs) have amplified the need for scalable, effective, zero-shot text-to-video retrieval systems. While recent advances in large-scale pretraining have improved zero-shot cross-modal alignment, existing methods still struggle with query-dependent temporal reasoning, limiting their effectiveness on complex queries involving temporal, logical, or causal relationships. To address these limitations, we propose an adaptive multi-agent retrieval framework that dynamically orchestrates specialized agents over multiple reasoning iterations based on the demands of each query. The framework includes: (1) a retrieval agent for scalable retrieval over large video corpora, (2) a reasoning agent for zero-shot contextual temporal reasoning, and (3) a query reformulation agent for refining ambiguous queries and recovering performance for those that degrade over iterations. These agents are dynamically coordinated by an orchestration agent, which leverages intermediate feedback and reasoning outcomes to guide execution. We also introduce a novel communication mechanism that incorporates retrieval-performance memory and historical reasoning traces to improve coordination and decision-making. Experiments on three TRECVid benchmarks spanning eight years show that our framework achieves a twofold improvement over CLIP4Clip and significantly outperforms state-of-the-art methods by a large margin. 

---
# A Markovian View of Iterative-Feedback Loops in Image Generative Models: Neural Resonance and Model Collapse 

**Authors**: Vibhas Kumar Vats, David J. Crandall, Samuel Goree  

**Link**: [PDF](https://arxiv.org/pdf/2602.19033)  

**Abstract**: AI training datasets will inevitably contain AI-generated examples, leading to ``feedback'' in which the output of one model impacts the training of another. It is known that such iterative feedback can lead to model collapse, yet the mechanisms underlying this degeneration remain poorly understood. Here we show that a broad class of feedback processes converges to a low-dimensional invariant structure in latent space, a phenomenon we call neural resonance. By modeling iterative feedback as a Markov Chain, we show that two conditions are needed for this resonance to occur: ergodicity of the feedback process and directional contraction of the latent representation. By studying diffusion models on MNIST and ImageNet, as well as CycleGAN and an audio feedback experiment, we map how local and global manifold geometry evolve, and we introduce an eight-pattern taxonomy of collapse behaviors. Neural resonance provides a unified explanation for long-term degenerate behavior in generative models and provides practical diagnostics for identifying, characterizing, and eventually mitigating collapse. 

---
# The Metaphysics We Train: A Heideggerian Reading of Machine Learning 

**Authors**: Heman Shakeri  

**Link**: [PDF](https://arxiv.org/pdf/2602.19028)  

**Abstract**: This paper offers a phenomenological reading of contemporary machine learning through Heideggerian concepts, aimed at enriching practitioners' reflexive understanding of their own practice. We argue that this philosophical lens reveals three insights invisible to purely technical analysis. First, the algorithmic Entwurf (projection) is distinctive in being automated, opaque, and emergent--a metaphysics that operates without explicit articulation or debate, crystallizing implicitly through gradient descent rather than theoretical argument. Second, even sophisticated technical advances remain within the regime of Gestell (Enframing), improving calculation without questioning the primacy of calculation itself. Third, AI's lack of existential structure, specifically the absence of Care (Sorge), is genuinely explanatory: it illuminates why AI systems have no internal resources for questioning their own optimization imperatives, and why they optimize without the anxiety (Angst) that signals, in human agents, the friction between calculative absorption and authentic existence. We conclude by exploring the pedagogical value of this perspective, arguing that data science education should cultivate not only technical competence but ontological literacy--the capacity to recognize what worldviews our tools enact and when calculation itself may be the wrong mode of engagement. 

---
# Pushing the Limits of Inverse Lithography with Generative Reinforcement Learning 

**Authors**: Haoyu Yang, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2602.19027)  

**Abstract**: Inverse lithography (ILT) is critical for modern semiconductor manufacturing but suffers from highly non-convex objectives that often trap optimization in poor local minima. Generative AI has been explored to warm-start ILT, yet most approaches train deterministic image-to-image translators to mimic sub-optimal datasets, providing limited guidance for escaping non-convex traps during refinement. We reformulate mask synthesis as conditional sampling: a generator learns a distribution over masks conditioned on the design and proposes multiple candidates. The generator is first pretrained with WGAN plus a reconstruction loss, then fine-tuned using Group Relative Policy Optimization (GRPO) with an ILT-guided imitation loss. At inference, we sample a small batch of masks, run fast batched ILT refinement, evaluate lithography metrics (e.g., EPE, process window), and select the best candidate. On \texttt{LithoBench} dataset, the proposed hybrid framework reduces EPE violations under a 3\,nm tolerance and roughly doubles throughput versus a strong numerical ILT baseline, while improving final mask quality. We also present over 20\% EPE improvement on \texttt{ICCAD13} contest cases with 3$\times$ speedup over the SOTA numerical ILT solver. By learning to propose ILT-friendly initializations, our approach mitigates non-convexity and advances beyond what traditional solvers or GenAI can achieve. 

---
# Routing-Aware Explanations for Mixture of Experts Graph Models in Malware Detection 

**Authors**: Hossein Shokouhinejad, Roozbeh Razavi-Far, Griffin Higgins, Ali.A Ghorbani  

**Link**: [PDF](https://arxiv.org/pdf/2602.19025)  

**Abstract**: Mixture-of-Experts (MoE) offers flexible graph reasoning by combining multiple views of a graph through a learned router. We investigate routing-aware explanations for MoE graph models in malware detection using control flow graphs (CFGs). Our architecture builds diversity at two levels. At the node level, each layer computes multiple neighborhood statistics and fuses them with an MLP, guided by a degree reweighting factor rho and a pooling choice lambda in {mean, std, max}, producing distinct node representations that capture complementary structural cues in CFGs. At the readout level, six experts, each tied to a specific (rho, lambda) view, output graph-level logits that the router weights into a final prediction. Post-hoc explanations are generated with edge-level attributions per expert and aggregated using the router gates so the rationale reflects both what each expert highlights and how strongly it is selected. Evaluated against single-expert GNN baselines such as GCN, GIN, and GAT on the same CFG dataset, the proposed MoE achieves strong detection accuracy while yielding stable, faithful attributions under sparsity-based perturbations. The results indicate that making the router explicit and combining multi-statistic node encoding with expert-level diversity can improve the transparency of MoE decisions for malware analysis. 

---
# An interpretable framework using foundation models for fish sex identification 

**Authors**: Zheng Miao, Tien-Chieh Hung  

**Link**: [PDF](https://arxiv.org/pdf/2602.19022)  

**Abstract**: Accurate sex identification in fish is vital for optimizing breeding and management strategies in aquaculture, particularly for species at the risk of extinction. However, most existing methods are invasive or stressful and may cause additional mortality, posing severe risks to threatened or endangered fish populations. To address these challenges, we propose FishProtoNet, a robust, non-invasive computer vision-based framework for sex identification of delta smelt (Hypomesus transpacificus), an endangered fish species native to California, across its full life cycle. Unlike the traditional deep learning methods, FishProtoNet provides interpretability through learned prototype representations while improving robustness by leveraging foundation models to reduce the influence of background noise. Specifically, the FishProtoNet framework consists of three key components: fish regions of interest (ROIs) extraction using visual foundation model, feature extraction from fish ROIs and fish sex identification based on an interpretable prototype network. FishProtoNet demonstrates strong performance in delta smelt sex identification during early spawning and post-spawning stages, achieving the accuracies of 74.40% and 81.16% and corresponding F1 scores of 74.27% and 79.43% respectively. In contrast, delta smelt sex identification at the subadult stage remains challenging for current computer vision methods, likely due to less pronounced morphological differences in immature fish. The source code of FishProtoNet is publicly available at: this https URL 

---
# Learning to Detect Language Model Training Data via Active Reconstruction 

**Authors**: Junjie Oscar Yin, John X. Morris, Vitaly Shmatikov, Sewon Min, Hannaneh Hajishirzi  

**Link**: [PDF](https://arxiv.org/pdf/2602.19020)  

**Abstract**: Detecting LLM training data is generally framed as a membership inference attack (MIA) problem. However, conventional MIAs operate passively on fixed model weights, using log-likelihoods or text generations. In this work, we introduce \textbf{Active Data Reconstruction Attack} (ADRA), a family of MIA that actively induces a model to reconstruct a given text through training. We hypothesize that training data are \textit{more reconstructible} than non-members, and the difference in their reconstructibility can be exploited for membership inference. Motivated by findings that reinforcement learning (RL) sharpens behaviors already encoded in weights, we leverage on-policy RL to actively elicit data reconstruction by finetuning a policy initialized from the target model. To effectively use RL for MIA, we design reconstruction metrics and contrastive rewards. The resulting algorithms, \textsc{ADRA} and its adaptive variant \textsc{ADRA+}, improve both reconstruction and detection given a pool of candidate data. Experiments show that our methods consistently outperform existing MIAs in detecting pre-training, post-training, and distillation data, with an average improvement of 10.7\% over the previous runner-up. In particular, \MethodPlus~improves over Min-K\%++ by 18.8\% on BookMIA for pre-training detection and by 7.6\% on AIME for post-training detection. 

---
# NeuroWise: A Multi-Agent LLM "Glass-Box" System for Practicing Double-Empathy Communication with Autistic Partners 

**Authors**: Albert Tang, Yifan Mo, Jie Li, Yue Su, Mengyuan Zhang, Sander L. Koole, Koen Hindriks, Jiahuan Pei  

**Link**: [PDF](https://arxiv.org/pdf/2602.18962)  

**Abstract**: The double empathy problem frames communication difficulties between neurodivergent and neurotypical individuals as arising from mutual misunderstanding, yet most interventions focus on autistic individuals. We present NeuroWise, a multi-agent LLM-based coaching system that supports neurotypical users through stress visualization, interpretation of internal experiences, and contextual guidance. In a between-subjects study (N=30), NeuroWise was rated as helpful by all participants and showed a significant condition-time effect on deficit-based attributions (p=0.02): NeuroWise users reduced deficit framing, while baseline users shifted toward blaming autistic "deficits" after difficult interactions. NeuroWise users also completed conversations more efficiently (37% fewer turns, p=0.03). These findings suggest that AI-based interpretation can support attributional change by helping users recognize communication challenges as mutual. 

---
# Give Users the Wheel: Towards Promptable Recommendation Paradigm 

**Authors**: Fuyuan Lyu, Chenglin Luo, Qiyuan Zhang, Yupeng Hou, Haolun Wu, Xing Tang, Xue Liu, Jin L.C. Guo, Xiuqiang He  

**Link**: [PDF](https://arxiv.org/pdf/2602.18929)  

**Abstract**: Conventional sequential recommendation models have achieved remarkable success in mining implicit behavioral patterns. However, these architectures remain structurally blind to explicit user intent: they struggle to adapt when a user's immediate goal (e.g., expressed via a natural language prompt) deviates from their historical habits. While Large Language Models (LLMs) offer the semantic reasoning to interpret such intent, existing integration paradigms force a dilemma: LLM-as-a-recommender paradigm sacrifices the efficiency and collaborative precision of ID-based retrieval, while Reranking methods are inherently bottlenecked by the recall capabilities of the underlying model. In this paper, we propose Decoupled Promptable Sequential Recommendation (DPR), a model-agnostic framework that empowers conventional sequential backbones to natively support Promptable Recommendation, the ability to dynamically steer the retrieval process using natural language without abandoning collaborative signals. DPR modulates the latent user representation directly within the retrieval space. To achieve this, we introduce a Fusion module to align the collaborative and semantic signals, a Mixture-of-Experts (MoE) architecture that disentangles the conflicting gradients from positive and negative steering, and a three-stage training strategy that progressively aligns the semantic space of prompts with the collaborative space. Extensive experiments on real-world datasets demonstrate that DPR significantly outperforms state-of-the-art baselines in prompt-guided tasks while maintaining competitive performance in standard sequential recommendation scenarios. 

---
# Why Agent Caching Fails and How to Fix It: Structured Intent Canonicalization with Few-Shot Learning 

**Authors**: Abhinaba Basu  

**Link**: [PDF](https://arxiv.org/pdf/2602.18922)  

**Abstract**: Personal AI agents incur substantial cost via repeated LLM calls. We show existing caching methods fail: GPTCache achieves 37.9% accuracy on real benchmarks; APC achieves 0-12%. The root cause is optimizing for the wrong property -- cache effectiveness requires key consistency and precision,
not classification accuracy. We observe cache-key evaluation reduces to clustering evaluation and apply V-measure decomposition to separate these on n=8,682 points across MASSIVE, BANKING77, CLINC150, and NyayaBench v2, our new 8,514-entry multilingual agentic dataset (528 intents, 20 W5H2 classes, 63 languages). We introduce W5H2, a structured intent decomposition framework. Using SetFit with 8 examples per class, W5H2 achieves 91.1%+/-1.7% on MASSIVE in ~2ms -- vs 37.9% for
GPTCache and 68.8% for a 20B-parameter LLM at 3,447ms. On NyayaBench v2 (20 classes), SetFit achieves 55.3%, with cross-lingual transfer across 30 languages. Our five-tier cascade handles 85% of interactions locally, projecting 97.5% cost reduction. We provide risk-controlled selective prediction guarantees via RCPS with nine bound families. 

---
# DeepInnovator: Triggering the Innovative Capabilities of LLMs 

**Authors**: Tianyu Fan, Fengji Zhang, Yuxiang Zheng, Bei Chen, Xinyao Niu, Chengen Huang, Junyang Lin, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18920)  

**Abstract**: The application of Large Language Models (LLMs) in accelerating scientific discovery has garnered increasing attention, with a key focus on constructing research agents endowed with innovative capability, i.e., the ability to autonomously generate novel and significant research ideas. Existing approaches predominantly rely on sophisticated prompt engineering and lack a systematic training paradigm. To address this, we propose DeepInnovator, a training framework designed to trigger the innovative capability of LLMs. Our approach comprises two core components. (1) ``Standing on the shoulders of giants''. We construct an automated data extraction pipeline to extract and organize structured research knowledge from a vast corpus of unlabeled scientific literature. (2) ``Conjectures and refutations''. We introduce a ``Next Idea Prediction'' training paradigm, which models the generation of research ideas as an iterative process of continuously predicting, evaluating, and refining plausible and novel next idea. Both automatic and expert evaluations demonstrate that our DeepInnovator-14B significantly outperforms untrained baselines, achieving win rates of 80.53\%-93.81\%, and attains performance comparable to that of current leading LLMs. This work provides a scalable training pathway toward building research agents with genuine, originative innovative capability, and will open-source the dataset to foster community advancement. Source code and data are available at: this https URL. 

---
# Adaptive Collaboration of Arena-Based Argumentative LLMs for Explainable and Contestable Legal Reasoning 

**Authors**: Hoang-Loc Cao, Phuc Ho, Truong Thanh Hung Nguyen, Phuc Truong Loc Nguyen, Dinh Thien Loc Nguyen, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2602.18916)  

**Abstract**: Legal reasoning requires not only high accuracy but also the ability to justify decisions through verifiable and contestable arguments. However, existing Large Language Model (LLM) approaches, such as Chain-of-Thought (CoT) and Retrieval-Augmented Generation (RAG), often produce unstructured explanations that lack a formal mechanism for verification or user intervention. To address this limitation, we propose Adaptive Collaboration of Argumentative LLMs (ACAL), a neuro-symbolic framework that integrates adaptive multi-agent collaboration with an Arena-based Quantitative Bipolar Argumentation Framework (A-QBAF). ACAL dynamically deploys expert agent teams to construct arguments, employs a clash resolution mechanism to adjudicate conflicting claims, and utilizes uncertainty-aware escalation for borderline cases. Crucially, our framework supports a Human-in-the-Loop (HITL) contestability workflow, enabling users to directly audit and modify the underlying reasoning graph to influence the final judgment. Empirical evaluations on the LegalBench benchmark demonstrate that ACAL outperforms strong baselines across Gemini-2.5-Flash-Lite and Gemini-2.5-Flash architectures, effectively balancing efficient predictive performance with structured transparency and contestability. Our implementation is available at: this https URL. 

---
# AAVGen: Precision Engineering of Adeno-associated Viral Capsids for Renal Selective Targeting 

**Authors**: Mohammadreza Ghaffarzadeh-Esfahani, Yousof Gheisari  

**Link**: [PDF](https://arxiv.org/pdf/2602.18915)  

**Abstract**: Adeno-associated viruses (AAVs) are promising vectors for gene therapy, but their native serotypes face limitations in tissue tropism, immune evasion, and production efficiency. Engineering capsids to overcome these hurdles is challenging due to the vast sequence space and the difficulty of simultaneously optimizing multiple functional properties. The complexity also adds when it comes to the kidney, which presents unique anatomical barriers and cellular targets that require precise and efficient vector engineering. Here, we present AAVGen, a generative artificial intelligence framework for de novo design of AAV capsids with enhanced multi-trait profiles. AAVGen integrates a protein language model (PLM) with supervised fine-tuning (SFT) and a reinforcement learning technique termed Group Sequence Policy Optimization (GSPO). The model is guided by a composite reward signal derived from three ESM-2-based regression predictors, each trained to predict a key property: production fitness, kidney tropism, and thermostability. Our results demonstrate that AAVGen produces a diverse library of novel VP1 protein sequences. In silico validations revealed that the majority of the generated variants have superior performance across all three employed indices, indicating successful multi-objective optimization. Furthermore, structural analysis via AlphaFold3 confirms that the generated sequences preserve the canonical capsid folding despite sequence diversification. AAVGen establishes a foundation for data-driven viral vector engineering, accelerating the development of next-generation AAV vectors with tailored functional characteristics. 

---
# TRUE: A Trustworthy Unified Explanation Framework for Large Language Model Reasoning 

**Authors**: Yujiao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18905)  

**Abstract**: Large language models (LLMs) have demonstrated strong capabilities in complex reasoning tasks, yet their decision-making processes remain difficult to interpret. Existing explanation methods often lack trustworthy structural insight and are limited to single-instance analysis, failing to reveal reasoning stability and systematic failure mechanisms. To address these limitations, we propose the Trustworthy Unified Explanation Framework (TRUE), which integrates executable reasoning verification, feasible-region directed acyclic graph (DAG) modeling, and causal failure mode analysis. At the instance level, we redefine reasoning traces as executable process specifications and introduce blind execution verification to assess operational validity. At the local structural level, we construct feasible-region DAGs via structure-consistent perturbations, enabling explicit characterization of reasoning stability and the executable region in the local input space. At the class level, we introduce a causal failure mode analysis method that identifies recurring structural failure patterns and quantifies their causal influence using Shapley values. Extensive experiments across multiple reasoning benchmarks demonstrate that the proposed framework provides multi-level, verifiable explanations, including executable reasoning structures for individual instances, feasible-region representations for neighboring inputs, and interpretable failure modes with quantified importance at the class level. These results establish a unified and principled paradigm for improving the interpretability and reliability of LLM reasoning systems. 

---
# HEHRGNN: A Unified Embedding Model for Knowledge Graphs with Hyperedges and Hyper-Relational Edges 

**Authors**: Rajesh Rajagopalamenon, Unnikrishnan Cheramangalath  

**Link**: [PDF](https://arxiv.org/pdf/2602.18897)  

**Abstract**: Knowledge Graph(KG) has gained traction as a machine-readable organization of real-world knowledge for analytics using artificial intelligence systems. Graph Neural Network(GNN), is proven to be an effective KG embedding technique that enables various downstream tasks like link prediction, node classification, and graph classification. The focus of research in both KG embedding and GNNs has been mostly oriented towards simple graphs with binary relations. However, real-world knowledge bases have a significant share of complex and n-ary facts that cannot be represented by binary edges. More specifically, real-world knowledge bases are often a mix of two types of n-ary facts - (i) that require hyperedges and (ii) that require hyper-relational edges. Though there are research efforts catering to these n-ary fact types, they are pursued independently for each type. We propose $H$yper$E$dge $H$yper-$R$elational edge $GNN$(HEHRGNN), a unified embedding model for n-ary relational KGs with both hyperedges and hyper-relational edges. The two main components of the model are i)HEHR unified fact representation format, and ii)HEHRGNN encoder, a GNN-based encoder with a novel message propagation model capable of capturing complex graph structures comprising both hyperedges and hyper-relational edges. The experimental results of HEHRGNN on link prediction tasks show its effectiveness as a unified embedding model, with inductive prediction capability, for link prediction across real-world datasets having different types of n-ary facts. The model also shows improved link prediction performance over baseline models for hyperedge and hyper-relational datasets. 

---
# Orchestrating LLM Agents for Scientific Research: A Pilot Study of Multiple Choice Question (MCQ) Generation and Evaluation 

**Authors**: Yuan An  

**Link**: [PDF](https://arxiv.org/pdf/2602.18891)  

**Abstract**: Advances in large language models (LLMs) are rapidly transforming scientific work, yet empirical evidence on how these systems reshape research activities remains limited. We report a mixed-methods pilot evaluation of an AI-orchestrated research workflow in which a human researcher coordinated multiple LLM-based agents to perform data extraction, corpus construction, artifact generation, and artifact evaluation. Using the generation and assessment of multiple-choice questions (MCQs) as a testbed, we collected 1,071 SAT Math MCQs and employed LLM agents to extract questions from PDFs, retrieve and convert open textbooks into structured representations, align each MCQ with relevant textbook content, generate new MCQs under specified difficulty and cognitive levels, and evaluate both original and generated MCQs using a 24-criterion quality framework. Across all evaluations, average MCQ quality was high. However, criterion-level analysis and equivalence testing show that generated MCQs are not fully comparable to expert-vetted baseline questions. Strict similarity (24/24 criteria equivalent) was never achieved. Persistent gaps concentrated in skill\ depth, cognitive engagement, difficulty calibration, and metadata alignment, while surface-level qualities, such as {grammar fluency}, {clarity options}, {no duplicates}, were consistently strong. Beyond MCQ outcomes, the study documents a labor shift. The researcher's work moved from ``authoring items'' toward {specification, orchestration, verification}, and {governance}. Formalizing constraints, designing rubrics, building validation loops, recovering from tool failures, and auditing provenance constituted the primary activities. We discuss implications for the future of scientific work, including emerging ``AI research operations'' skills required for AI-empowered research pipelines. 

---
# SceneTok: A Compressed, Diffusable Token Space for 3D Scenes 

**Authors**: Mohammad Asim, Christopher Wewer, Jan Eric Lenssen  

**Link**: [PDF](https://arxiv.org/pdf/2602.18882)  

**Abstract**: We present SceneTok, a novel tokenizer for encoding view sets of scenes into a compressed and diffusable set of unstructured tokens. Existing approaches for 3D scene representation and generation commonly use 3D data structures or view-aligned fields. In contrast, we introduce the first method that encodes scene information into a small set of permutation-invariant tokens that is disentangled from the spatial grid. The scene tokens are predicted by a multi-view tokenizer given many context views and rendered into novel views by employing a light-weight rectified flow decoder. We show that the compression is 1-3 orders of magnitude stronger than for other representations while still reaching state-of-the-art reconstruction quality. Further, our representation can be rendered from novel trajectories, including ones deviating from the input trajectory, and we show that the decoder gracefully handles uncertainty. Finally, the highly-compressed set of unstructured latent scene tokens enables simple and efficient scene generation in 5 seconds, achieving a much better quality-speed trade-off than previous paradigms. 

---
# FOCA: Frequency-Oriented Cross-Domain Forgery Detection, Localization and Explanation via Multi-Modal Large Language Model 

**Authors**: Zhou Liu, Tonghua Su, Hongshi Zhang, Fuxiang Yang, Donglin Di, Yang Song, Lei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2602.18880)  

**Abstract**: Advances in image tampering techniques, particularly generative models, pose significant challenges to media verification, digital forensics, and public trust. Existing image forgery detection and localization (IFDL) methods suffer from two key limitations: over-reliance on semantic content while neglecting textural cues, and limited interpretability of subtle low-level tampering traces. To address these issues, we propose FOCA, a multimodal large language model-based framework that integrates discriminative features from both the RGB spatial and frequency domains via a cross-attention fusion module. This design enables accurate forgery detection and localization while providing explicit, human-interpretable cross-domain explanations. We further introduce FSE-Set, a large-scale dataset with diverse authentic and tampered images, pixel-level masks, and dual-domain annotations. Extensive experiments show that FOCA outperforms state-of-the-art methods in detection performance and interpretability across both spatial and frequency domains. 

---
# Structure-Level Disentangled Diffusion for Few-Shot Chinese Font Generation 

**Authors**: Jie Li, Suorong Yang, Jian Zhao, Furao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.18874)  

**Abstract**: Few-shot Chinese font generation aims to synthesize new characters in a target style using only a handful of reference images. Achieving accurate content rendering and faithful style transfer requires effective disentanglement between content and style. However, existing approaches achieve only feature-level disentanglement, allowing the generator to re-entangle these features, leading to content distortion and degraded style fidelity. We propose the Structure-Level Disentangled Diffusion Model (SLD-Font), which receives content and style information from two separate channels. SimSun-style images are used as content templates and concatenated with noisy latent features as the input. Style features extracted by a CLIP model from target-style images are integrated via cross-attention. Additionally, we train a Background Noise Removal module in the pixel space to remove background noise in complex stroke regions. Based on theoretical validation of disentanglement effectiveness, we introduce a parameter-efficient fine-tuning strategy that updates only the style-related modules. This allows the model to better adapt to new styles while avoiding overfitting to the reference images' content. We further introduce the Grey and OCR metrics to evaluate the content quality of generated characters. Experimental results show that SLD-Font achieves significantly higher style fidelity while maintaining comparable content accuracy to existing state-of-the-art methods. 

---
# BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation 

**Authors**: Miaowei Wang, Qingxuan Yan, Zhi Cao, Yayuan Li, Oisin Mac Aodha, Jason J. Corso, Amir Vaxman  

**Link**: [PDF](https://arxiv.org/pdf/2602.18873)  

**Abstract**: Text-guided dynamic 3D character generation has advanced rapidly, yet producing high-quality motion that faithfully reflects rich textual descriptions remains challenging. Existing methods tend to generate limited sub-actions or incoherent motion due to fixed-length temporal inputs and discrete frame-wise representations that fail to capture rich motion semantics. We address these limitations by representing motion with continuous differentiable B-spline curves, enabling more effective motion generation without modifying the capabilities of the underlying generative model. Specifically, our closed-form, Laplacian-regularized B-spline solver efficiently compresses variable-length motion sequences into compact representations with a fixed number of control points. Further, we introduce a normal-fusion strategy for input shape adherence along with correspondence-aware and local-rigidity losses for motion-restoration quality. To train our model, we collate BIMO, a new dataset containing diverse variable-length 3D motion sequences with rich, high-quality text annotations. Extensive evaluations show that our feed-forward framework BiMotion generates more expressive, higher-quality, and better prompt-aligned motions than existing state-of-the-art methods, while also achieving faster generation. Our project page is at: this https URL. 

---
# Hyperbolic Busemann Neural Networks 

**Authors**: Ziheng Chen, Bernhard Schölkopf, Nicu Sebe  

**Link**: [PDF](https://arxiv.org/pdf/2602.18858)  

**Abstract**: Hyperbolic spaces provide a natural geometry for representing hierarchical and tree-structured data due to their exponential volume growth. To leverage these benefits, neural networks require intrinsic and efficient components that operate directly in hyperbolic space. In this work, we lift two core components of neural networks, Multinomial Logistic Regression (MLR) and Fully Connected (FC) layers, into hyperbolic space via Busemann functions, resulting in Busemann MLR (BMLR) and Busemann FC (BFC) layers with a unified mathematical interpretation. BMLR provides compact parameters, a point-to-horosphere distance interpretation, batch-efficient computation, and a Euclidean limit, while BFC generalizes FC and activation layers with comparable complexity. Experiments on image classification, genome sequence learning, node classification, and link prediction demonstrate improvements in effectiveness and efficiency over prior hyperbolic layers. The code is available at this https URL. 

---
# Rank-Aware Spectral Bounds on Attention Logits for Stable Low-Precision Training 

**Authors**: Seyed Morteza Emadi  

**Link**: [PDF](https://arxiv.org/pdf/2602.18851)  

**Abstract**: Attention scores in transformers are bilinear forms $S_{ij} = x_i^\top M x_j / \sqrt{d_h}$ whose maximum magnitude governs overflow risk in low-precision training. We derive a \emph{rank-aware concentration inequality}: when the interaction matrix $M = W^Q W^{K\top}$ has rank $r \ll d$, tail probabilities for $\max_{i,j}|S_{ij}|$ decay as $\exp(-d^{2}\alpha^{2}/(\gamma r))$ rather than $\exp(-d\alpha^{2})$, where $\gamma > 1$ is a typicality parameter. For transformer attention where $r = d_h$, this yields $8$--$28\times$ tighter concentration than rank-agnostic bounds in modern architectures. We apply this result to FP8 training, deriving \emph{geometry-aware scale factors} that provide principled overflow guarantees without observing activations. The method computes per-layer scales from the spectral norm $\|W^Q W^{K\top}\|_2$ via implicit power iteration, includes a grouped query attention formulation that avoids key expansion, and remains compatible with fused attention kernels. Across GPT-2 XL to Llama-2-70B, geometry-aware scaling eliminates overflows in transient scenarios where delayed scaling fails, while achieving comparable downstream MMLU accuracy. 

---
# When the Inference Meets the Explicitness or Why Multimodality Can Make Us Forget About the Perfect Predictor 

**Authors**: J. E. Domínguez-Vidal, Alberto Sanfeliu  

**Link**: [PDF](https://arxiv.org/pdf/2602.18850)  

**Abstract**: Although in the literature it is common to find predictors and inference systems that try to predict human intentions, the uncertainty of these models due to the randomness of human behavior has led some authors to start advocating the use of communication systems that explicitly elicit human intention. In this work, it is analyzed the use of four different communication systems with a human-robot collaborative object transportation task as experimental testbed: two intention predictors (one based on force prediction and another with an enhanced velocity prediction algorithm) and two explicit communication methods (a button interface and a voice-command recognition system). These systems were integrated into IVO, a custom mobile social robot equipped with force sensor to detect the force exchange between both agents and LiDAR to detect the environment. The collaborative task required transporting an object over a 5-7 meter distance with obstacles in the middle, demanding rapid decisions and precise physical coordination. 75 volunteers perform a total of 255 executions divided into three groups, testing inference systems in the first round, communication systems in the second, and the combined strategies in the third. The results show that, 1) once sufficient performance is achieved, the human no longer notices and positively assesses technical improvements; 2) the human prefers systems that are more natural to them even though they have higher failure rates; and 3) the preferred option is the right combination of both systems. 

---
# Exact Attention Sensitivity and the Geometry of Transformer Stability 

**Authors**: Seyed Morteza Emadi  

**Link**: [PDF](https://arxiv.org/pdf/2602.18849)  

**Abstract**: Despite powering modern AI, transformers remain mysteriously brittle to train. We develop a stability theory that explains why pre-LayerNorm works, why DeepNorm uses $N^{-1/4}$ scaling, and why warmup is necessary, all from first principles. Our framework has two pillars: (1) We derive the \emph{exact} operator norm of the softmax Jacobian, $\|J_{softmax}(u/\tau)\|_{\infty\to 1} = \theta(p)/\tau$, where the balanced-mass factor $\theta(p)\in[0,1]$ quantifies attention sensitivity. (2) We introduce a block-$\infty$/RMS geometry aligned with tokenwise computation, yielding Lipschitz bounds independent of sequence length. Using this framework, we prove that pre-LN preserves identity gradient paths while post-LN compounds LayerNorm Jacobians exponentially with depth, and we show that DeepNorm's $N^{-1/4}$ emerges from the quartic structure of attention's four projection matrices. We validate our theory on 774M-parameter models and find that, contrary to the intuition that attention sharpens during training to reduce sensitivity, $\theta(p) \approx 1$ persists throughout. Transformer stability arises entirely from architectural gradient flow, not from attention dynamics. This finding changes how we reason about training: the architecture itself must handle sensitivity, not learned attention patterns. 

---
# DUET-VLM: Dual stage Unified Efficient Token reduction for VLM Training and Inference 

**Authors**: Aditya Kumar Singh, Hitesh Kandala, Pratik Prabhanjan Brahma, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2602.18846)  

**Abstract**: Vision-language models (VLMs) have achieved remarkable multimodal understanding and reasoning capabilities, yet remain computationally expensive due to dense visual tokenization. Existing efficiency approaches either merge redundant visual tokens or drop them progressively in language backbone, often trading accuracy for speed. In this work, we propose DUET-VLM, a versatile plug-and-play dual compression framework that consists of (a) vision-only redundancy aware compression of vision encoder's output into information-preserving tokens, followed by (b) layer-wise, salient text-guided dropping of visual tokens within the language backbone to progressively prune less informative tokens. This coordinated token management enables aggressive compression while retaining critical semantics. On LLaVA-1.5-7B, our approach maintains over 99% of baseline accuracy with 67% fewer tokens, and still retains >97% even at 89% reduction. With this dual-stage compression during training, it achieves 99.7% accuracy at 67% and 97.6% at 89%, surpassing prior SoTA visual token reduction methods across multiple benchmarks. When integrated into Video-LLaVA-7B, it even surpasses the baseline -- achieving >100% accuracy with a substantial 53.1% token reduction and retaining 97.6% accuracy under an extreme 93.4% setting. These results highlight end-to-end training with DUET-VLM, enabling robust adaptation to reduced visual (image/video) input without sacrificing accuracy, producing compact yet semantically rich representations within the same computational budget. Our code is available at this https URL. 

---
# When Agda met Vampire 

**Authors**: Artjoms Šinkarovs, Michael Rawson  

**Link**: [PDF](https://arxiv.org/pdf/2602.18844)  

**Abstract**: Dependently-typed proof assistants furnish expressive foundations for mechanised mathematics and verified software. However, automation for these systems has been either modest in scope or complex in implementation. We aim to improve the situation by integrating proof assistants with automated theorem provers (ATPs) in a simple way, while preserving the correctness guarantees of the former. A central difficulty arises from the fact that most ATPs operate in classical first-order logic, whereas these proof assistants are grounded in constructive dependent type theory. We identify an expressive fragment of both languages -- essentially equational Horn -- that admits sound, straightforward translations in both directions. The approach produces a prototype system for Agda forwarding proof obligations to the ATP Vampire, then transforming the resulting classical proof into a constructive proof term that Agda can type-check. The prototype automatically derives proofs concerning the properties of a complex field equipped with roots of unity, which took professional Agda developers two full days to complete. The required engineering effort is modest, and we anticipate that the methodology will extend readily to other ATPs and proof assistants. 

---
# OpenClaw AI Agents as Informal Learners at Moltbook: Characterizing an Emergent Learning Community at Scale 

**Authors**: Eason Chen, Ce Guan, Ahmed Elshafiey, Zhonghao Zhao, Joshua Zekeri, Afeez Edeifo Shaibu, Emmanuel Osadebe Prince, Cyuan Jhen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.18832)  

**Abstract**: Informal learning communities have been called the "other Massive Open Online C" in Learning@Scale research, yet remain understudied compared to MOOCs. We present the first empirical study of a large-scale informal learning community composed entirely of AI agents. Moltbook, a social network exclusively for AI agents powered by autonomous agent frameworks such as OpenClaw, grew to over 2.8 million registered agents in three weeks. Analyzing 231,080 non-spam posts across three phases of community evolution, we find three key patterns. First, participation inequality is extreme from the start (comment Gini = 0.889), exceeding human community benchmarks. Second, AI agents exhibit a "broadcasting inversion": statement-to-question ratios of 8.9:1 to 9.7:1 contrast sharply with the question-driven dynamics of human learning communities, and comment-level analysis of 1.55 million comments reveals a "parallel monologue" pattern where 93% of comments are independent responses rather than threaded dialogue. Third, we document a characteristic engagement lifecycle: explosive initial growth (184K posts from 32K authors in 11 days), a spam crisis (57,093 posts deleted by the platform), and engagement decline (mean comments: 31.7 -> 8.3 -> 1.7) that had not reversed by the end of our observation window despite effective spam removal. Sentiment analysis reveals a selection effect: comment tone becomes more positive as engagement declines, suggesting that casual participants disengage first while committed contributors remain. These findings have direct implications for hybrid human-AI learning platforms. 

---
# UniRank: A Multi-Agent Calibration Pipeline for Estimating University Rankings from Anonymized Bibliometric Signals 

**Authors**: Pedram Riyazimehr, Seyyed Ehsan Mahmoudi  

**Link**: [PDF](https://arxiv.org/pdf/2602.18824)  

**Abstract**: We present UniRank, a multi-agent LLM pipeline that estimates university positions across global ranking systems using only publicly available bibliometric data from OpenAlex and Semantic Scholar. The system employs a three-stage architecture: (a) zero-shot estimation from anonymized institutional metrics, (b) per-system tool-augmented calibration against real ranked universities, and (c) final synthesis. Critically, institutions are anonymized -- names, countries, DOIs, paper titles, and collaboration countries are all redacted -- and their actual ranks are hidden from the calibration tools during evaluation, preventing LLM memorization from confounding results. On the Times Higher Education (THE) World University Rankings ($n=352$), the system achieves MAE = 251.5 rank positions, Median AE = 131.5, PNMAE = 12.03%, Spearman $\rho = 0.769$, Kendall $\tau = 0.591$, hit rate @50 = 20.7%, hit rate @100 = 39.8%, and a Memorization Index of exactly zero (no exact-match zero-width predictions among all 352 universities). The systematic positive-signed error (+190.1 positions, indicating the system consistently predicts worse ranks than actual) and monotonic performance degradation from elite tier (MAE = 60.5, hit@100 = 90.5%) to tail tier (MAE = 328.2, hit@100 = 20.8%) provide strong evidence that the pipeline performs genuine analytical reasoning rather than recalling memorized rankings. A live demo is available at this https URL . 

---
# Chat-Based Support Alone May Not Be Enough: Comparing Conversational and Embedded LLM Feedback for Mathematical Proof Learning 

**Authors**: Eason Chen, Sophia Judicke, Kayla Beigh, Xinyi Tang, Isabel Wang, Nina Yuan, Zimo Xiao, Chuangji Li, Shizhuo Li, Reed Luttmer, Shreya Singh, Maria Yampolsky, Naman Parikh, Yvonne Zhao, Meiyi Chen, Scarlett Huang, Anishka Mohanty, Gregory Johnson, John Mackey, Jionghao Lin, Ken Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2602.18807)  

**Abstract**: We evaluate GPTutor, an LLM-powered tutoring system for an undergraduate discrete mathematics course. It integrates two LLM-supported tools: a structured proof-review tool that provides embedded feedback on students' written proof attempts, and a chatbot for math questions. In a staggered-access study with 148 students, earlier access was associated with higher homework performance during the interval when only the experimental group could use the system, while we did not observe this performance increase transfer to exam scores. Usage logs show that students with lower self-efficacy and prior exam performance used both components more frequently. Session-level behavioral labels, produced by human coding and scaled using an automated classifier, characterize how students engaged with the chatbot (e.g., answer-seeking or help-seeking). In models controlling for prior performance and self-efficacy, higher chatbot usage and answer-seeking behavior were negatively associated with subsequent midterm performance, whereas proof-review usage showed no detectable independent association. Together, the findings suggest that chatbot-based support alone may not reliably support transfer to independent assessment of math proof-learning outcomes, whereas work-anchored, structured feedback appears less associated with reduced learning. 

---
# Think$^{2}$: Grounded Metacognitive Reasoning in Large Language Models 

**Authors**: Abraham Paul Elenjical, Vivek Hruday Kavuri, Vasudeva Varma  

**Link**: [PDF](https://arxiv.org/pdf/2602.18806)  

**Abstract**: Large Language Models (LLMs) demonstrate strong reasoning performance, yet their ability to reliably monitor, diagnose, and correct their own errors remains limited. We introduce a psychologically grounded metacognitive framework that operationalizes Ann Brown's regulatory cycle (Planning, Monitoring, and Evaluation) as a structured prompting architecture, and study its integration within a lightweight dual-process MetaController for adaptive effort allocation. Across diverse reasoning and diagnostic benchmarks (GSM8K, CRUXEval, MBPP, AIME, CorrectBench, and TruthfulQA) using Llama-3 and Qwen-3 (8B), explicit regulatory structuring substantially improves error diagnosis and yields a threefold increase in successful self-correction. Blinded human evaluations over 580 query pairs show an 84% aggregate preference for trustworthiness and metacognitive self-awareness over standard and Chain-of-Thought baselines. Grounding LLM reasoning in established cognitive theory offers a principled path toward more transparent and diagnostically robust AI systems. 

---
# Operational Robustness of LLMs on Code Generation 

**Authors**: Debalina Ghosh Paul, Hong Zhu, Ian Bayley  

**Link**: [PDF](https://arxiv.org/pdf/2602.18800)  

**Abstract**: It is now common practice in software development for large language models (LLMs) to be used to generate program code. It is desirable to evaluate the robustness of LLMs for this usage. This paper is concerned in particular with how sensitive LLMs are to variations in descriptions of the coding tasks. However, existing techniques for evaluating this robustness are unsuitable for code generation because the input data space of natural language descriptions is discrete. To address this problem, we propose a robustness evaluation method called scenario domain analysis, which aims to find the expected minimal change in the natural language descriptions of coding tasks that would cause the LLMs to produce incorrect outputs. We have formally proved the theoretical properties of the method and also conducted extensive experiments to evaluate the robustness of four state-of-the-art art LLMs: Gemini-pro, Codex, Llamma2 and Falcon 7B, and have found that we are able to rank these with confidence from best to worst. Moreover, we have also studied how robustness varies in different scenarios, including the variations with the topic of the coding task and with the complexity of its sample solution, and found that robustness is lower for more complex tasks and also lower for more advanced topics, such as multi-threading and data structures. 

---
# Carbon-aware decentralized dynamic task offloading in MIMO-MEC networks via multi-agent reinforcement learning 

**Authors**: Mubshra Zulfiqar, Muhammad Ayzed Mirza, Basit Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2602.18797)  

**Abstract**: Massive internet of things microservices require integrating renewable energy harvesting into mobile edge computing (MEC) for sustainable eScience infrastructures. Spatiotemporal mismatches between stochastic task arrivals and intermittent green energy along with complex inter-user interference in multi-antenna (MIMO) uplinks complicate real-time resource management. Traditional centralized optimization and off-policy reinforcement learning struggle with scalability and signaling overhead in dense networks. This paper proposes CADDTO-PPO, a carbon-aware decentralized dynamic task offloading framework based on multi-agent proximal policy optimization. The multi-user MIMO-MEC system is modeled as a Decentralized Partially Observable Markov Decision Process (DEC-POMDP) to jointly minimize carbon emissions and buffer latency and energy wastage. A scalable architecture utilizes decentralized execution with parameter sharing (DEPS), which enables autonomous IoT agents to make fine-grained power control and offloading decisions based solely on local observations. Additionally, a carbon-first reward structure adaptively prioritizes green time slots for data transmission to decouple system throughput from grid-dependent carbon footprints. Finally, experimental results demonstrate CADDTO-PPO outperforms deep deterministic policy gradient (DDPG) and lyapunov-based baselines. The framework achieves the lowest carbon intensity and maintains near-zero packet overflow rates under extreme traffic loads. Architectural profiling validates the framework to demonstrate a constant $O(1)$ inference complexity and theoretical lightweight feasibility for future generation sustainable IoT deployments. 

---
# MANATEE: Inference-Time Lightweight Diffusion Based Safety Defense for LLMs 

**Authors**: Chun Yan Ryan Kan, Tommy Tran, Vedant Yadav, Ava Cai, Kevin Zhu, Ruizhe Li, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2602.18782)  

**Abstract**: Defending LLMs against adversarial jailbreak attacks remains an open challenge. Existing defenses rely on binary classifiers that fail when adversarial input falls outside the learned decision boundary, and repeated fine-tuning is computationally expensive while potentially degrading model capabilities. We propose MANATEE, an inference-time defense that uses density estimation over a benign representation manifold. MANATEE learns the score function of benign hidden states and uses diffusion to project anomalous representations toward safe regions--requiring no harmful training data and no architectural modifications. Experiments across Mistral-7B-Instruct, Llama-3.1-8B-Instruct, and Gemma-2-9B-it demonstrate that MANATEE reduce Attack Success Rate by up to 100\% on certain datasets, while preserving model utility on benign inputs. 

---
# ArabicNumBench: Evaluating Arabic Number Reading in Large Language Models 

**Authors**: Anas Alhumud, Abdulaziz Alhammadi, Muhammad Badruddin Khan  

**Link**: [PDF](https://arxiv.org/pdf/2602.18776)  

**Abstract**: We present ArabicNumBench, a comprehensive benchmark for evaluating large language models on Arabic number reading tasks across Eastern Arabic-Indic numerals (0-9 in Arabic script) and Western Arabic numerals (0-9). We evaluate 71 models from 10 providers using four prompting strategies (zero-shot, zero-shot CoT, few-shot, few-shot CoT) on 210 number reading tasks spanning six contextual categories: pure numerals, addresses, dates, quantities, and prices. Our evaluation comprises 59,010 individual test cases and tracks extraction methods to measure structured output generation. Evaluation reveals substantial performance variation, with accuracy ranging from 14.29\% to 99.05\% across models and strategies. Few-shot Chain-of-Thought prompting achieves 2.8x higher accuracy than zero-shot approaches (80.06\% vs 28.76\%). A striking finding emerges: models achieving elite accuracy (98-99\%) often produce predominantly unstructured output, with most responses lacking Arabic CoT markers. Only 6 models consistently generate structured output across all test cases, while the majority require fallback extraction methods despite high numerical accuracy. Comprehensive evaluation of 281 model-strategy combinations demonstrates that numerical accuracy and instruction-following represent distinct capabilities, establishing baselines for Arabic number comprehension and providing actionable guidance for model selection in production Arabic NLP systems. 

---
# GLaDiGAtor: Language-Model-Augmented Multi-Relation Graph Learning for Predicting Disease-Gene Associations 

**Authors**: Osman Onur Kuzucu, Tunca Doğan  

**Link**: [PDF](https://arxiv.org/pdf/2602.18769)  

**Abstract**: Understanding disease-gene associations is essential for unravelling disease mechanisms and advancing diagnostics and therapeutics. Traditional approaches based on manual curation and literature review are labour-intensive and not scalable, prompting the use of machine learning on large biomedical data. In particular, graph neural networks (GNNs) have shown promise for modelling complex biological relationships. To address limitations in existing models, we propose GLaDiGAtor (Graph Learning-bAsed DIsease-Gene AssociaTiOn pRediction), a novel GNN framework with an encoder-decoder architecture for disease-gene association prediction. GLaDiGAtor constructs a heterogeneous biological graph integrating gene-gene, disease-disease, and gene-disease interactions from curated databases, and enriches each node with contextual features from well-known language models (ProtT5 for protein sequences and BioBERT for disease text). In evaluations, our model achieves superior predictive accuracy and generalisation, outperforming 14 existing methods. Literature-supported case studies confirm the biological relevance of high-confidence novel predictions, highlighting GLaDiGAtor's potential to discover candidate disease genes. These results underscore the power of graph convolutional networks in biomedical informatics and may ultimately facilitate drug discovery by revealing new gene-disease links. The source code and processed datasets are publicly available at this https URL. 

---
# TAG: Thinking with Action Unit Grounding for Facial Expression Recognition 

**Authors**: Haobo Lin, Tianyi Bai, Jiajun Zhang, Xuanhao Chang, Sheng Lu, Fangming Gu, Zengjie Hu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18763)  

**Abstract**: Facial Expression Recognition (FER) is a fine-grained visual understanding task where reliable predictions require reasoning over localized and meaningful facial cues. Recent vision--language models (VLMs) enable natural language explanations for FER, but their reasoning is often ungrounded, producing fluent yet unverifiable rationales that are weakly tied to visual evidence and prone to hallucination, leading to poor robustness across different datasets. We propose TAG (Thinking with Action Unit Grounding), a vision--language framework that explicitly constrains multimodal reasoning to be supported by facial Action Units (AUs). TAG requires intermediate reasoning steps to be grounded in AU-related facial regions, yielding predictions accompanied by verifiable visual evidence. The model is trained via supervised fine-tuning on AU-grounded reasoning traces followed by reinforcement learning with an AU-aware reward that aligns predicted regions with external AU detectors. Evaluated on RAF-DB, FERPlus, and AffectNet, TAG consistently outperforms strong open-source and closed-source VLM baselines while simultaneously improving visual faithfulness. Ablation and preference studies further show that AU-grounded rewards stabilize reasoning and mitigate hallucination, demonstrating the importance of structured grounded intermediate representations for trustworthy multimodal reasoning in FER. The code will be available at this https URL . 

---
# Towards Reliable Negative Sampling for Recommendation with Implicit Feedback via In-Community Popularity 

**Authors**: Chen Chen, Haobo Lin, Yuanbo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.18759)  

**Abstract**: Learning from implicit feedback is a fundamental problem in modern recommender systems, where only positive interactions are observed and explicit negative signals are unavailable. In such settings, negative sampling plays a critical role in model training by constructing negative items that enable effective preference learning and ranking optimization. However, designing reliable negative sampling strategies remains challenging, as they must simultaneously ensure realness, hardness, and interpretability. To this end, we propose \textbf{ICPNS (In-Community Popularity Negative Sampling)}, a novel framework that leverages user community structure to identify reliable and informative negative samples. Our approach is grounded in the insight that item exposure is driven by latent user communities. By identifying these communities and utilizing in-community popularity, ICPNS effectively approximates the probability of item exposure. Consequently, items that are popular within a user's community but remain unclicked are identified as more reliable true negatives. Extensive experiments on four benchmark datasets demonstrate that ICPNS yields consistent improvements on graph-based recommenders and competitive performance on MF-based models, outperforming representative negative sampling strategies under a unified evaluation protocol. 

---
# UFO: Unlocking Ultra-Efficient Quantized Private Inference with Protocol and Algorithm Co-Optimization 

**Authors**: Wenxuan Zeng, Chao Yang, Tianshi Xu, Bo Zhang, Changrui Ren, Jin Dong, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.18758)  

**Abstract**: Private convolutional neural network (CNN) inference based on secure two-party computation (2PC) suffers from high communication and latency overhead, especially from convolution layers. In this paper, we propose UFO, a quantized 2PC inference framework that jointly optimizes the 2PC protocols and quantization algorithm. UFO features a novel 2PC protocol that systematically combines the efficient Winograd convolution algorithm with quantization to improve inference efficiency. However, we observe that naively combining quantization and Winograd convolution faces the following challenges: 1) From the inference perspective, Winograd transformations introduce extensive additions and require frequent bit width conversions to avoid inference overflow, leading to non-negligible communication overhead; 2) From the training perspective, Winograd transformations introduce weight outliers that make quantization-aware training (QAT) difficult, resulting in inferior model accuracy. To address these challenges, we co-optimize both protocol and algorithm. 1) At the protocol level, we propose a series of graph-level optimizations for 2PC inference to minimize the communication. 2) At the algorithm level, we develop a mixed-precision QAT algorithm based on layer sensitivity to optimize model accuracy given communication constraints. To accommodate the outliers, we further introduce a 2PC-friendly bit re-weighting algorithm to increase the representation range without explicitly increasing bit widths. With extensive experiments, UFO demonstrates 11.7x, 3.6x, and 6.3x communication reduction with 1.29%, 1.16%, and 1.29% higher accuracy compared to state-of-the-art frameworks SiRNN, COINN, and CoPriv, respectively. 

---
# Synthesizing Multimodal Geometry Datasets from Scratch and Enabling Visual Alignment via Plotting Code 

**Authors**: Haobo Lin, Tianyi Bai, Chen Chen, Jiajun Zhang, Bohan Zeng, Wentao Zhang, Binhang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2602.18745)  

**Abstract**: Multimodal geometry reasoning requires models to jointly understand visual diagrams and perform structured symbolic inference, yet current vision--language models struggle with complex geometric constructions due to limited training data and weak visual--symbolic alignment. We propose a pipeline for synthesizing complex multimodal geometry problems from scratch and construct a dataset named \textbf{GeoCode}, which decouples problem generation into symbolic seed construction, grounded instantiation with verification, and code-based diagram rendering, ensuring consistency across structure, text, reasoning, and images. Leveraging the plotting code provided in GeoCode, we further introduce code prediction as an explicit alignment objective, transforming visual understanding into a supervised structured prediction task. GeoCode exhibits substantially higher structural complexity and reasoning difficulty than existing benchmarks, while maintaining mathematical correctness through multi-stage validation. Extensive experiments show that models trained on GeoCode achieve consistent improvements on multiple geometry benchmarks, demonstrating both the effectiveness of the dataset and the proposed alignment strategy. The code will be available at this https URL. 

---
# RoboCurate: Harnessing Diversity with Action-Verified Neural Trajectory for Robot Learning 

**Authors**: Seungku Kim, Suhyeok Jang, Byungjun Yoon, Dongyoung Kim, John Won, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2602.18742)  

**Abstract**: Synthetic data generated by video generative models has shown promise for robot learning as a scalable pipeline, but it often suffers from inconsistent action quality due to imperfectly generated videos. Recently, vision-language models (VLMs) have been leveraged to validate video quality, but they have limitations in distinguishing physically accurate videos and, even then, cannot directly evaluate the generated actions themselves. To tackle this issue, we introduce RoboCurate, a novel synthetic robot data generation framework that evaluates and filters the quality of annotated actions by comparing them with simulation replay. Specifically, RoboCurate replays the predicted actions in a simulator and assesses action quality by measuring the consistency of motion between the simulator rollout and the generated video. In addition, we unlock observation diversity beyond the available dataset via image-to-image editing and apply action-preserving video-to-video transfer to further augment appearance. We observe RoboCurate's generated data yield substantial relative improvements in success rates compared to using real data only, achieving +70.1% on GR-1 Tabletop (300 demos), +16.1% on DexMimicGen in the pre-training setup, and +179.9% in the challenging real-world ALLEX humanoid dexterous manipulation setting. 

---
# HONEST-CAV: Hierarchical Optimization of Network Signals and Trajectories for Connected and Automated Vehicles with Multi-Agent Reinforcement Learning 

**Authors**: Ziyan Zhang, Changxin Wan, Peng Hao, Kanok Boriboonsomsin, Matthew J. Barth, Yongkang Liu, Seyhan Ucar, Guoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.18740)  

**Abstract**: This study presents a hierarchical, network-level traffic flow control framework for mixed traffic consisting of Human-driven Vehicles (HVs), Connected and Automated Vehicles (CAVs). The framework jointly optimizes vehicle-level eco-driving behaviors and intersection-level traffic signal control to enhance overall network efficiency and decrease energy consumption. A decentralized Multi-Agent Reinforcement Learning (MARL) approach by Value Decomposition Network (VDN) manages cycle-based traffic signal control (TSC) at intersections, while an innovative Signal Phase and Timing (SPaT) prediction method integrates a Machine Learning-based Trajectory Planning Algorithm (MLTPA) to guide CAVs in executing Eco-Approach and Departure (EAD) maneuvers. The framework is evaluated across varying CAV proportions and powertrain types to assess its effects on mobility and energy performance. Experimental results conducted in a 4*4 real-world network demonstrate that the MARL-based TSC method outperforms the baseline model (i.e., Webster method) in speed, fuel consumption, and idling time. In addition, with MLTPA, HONEST-CAV benefits the traffic system further in energy consumption and idling time. With a 60% CAV proportion, vehicle average speed, fuel consumption, and idling time can be improved/saved by 7.67%, 10.23%, and 45.83% compared with the baseline. Furthermore, discussions on CAV proportions and powertrain types are conducted to quantify the performance of the proposed method with the impact of automation and electrification. 

---
# Rethinking Retrieval-Augmented Generation as a Cooperative Decision-Making Problem 

**Authors**: Lichang Song, Ting Long, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18734)  

**Abstract**: Retrieval-Augmented Generation (RAG) has demonstrated strong effectiveness in knowledge-intensive tasks by grounding language generation in external evidence. Despite its success, many existing RAG systems are built based on a ranking-centric, asymmetric dependency paradigm, where the generation quality of the generator is highly dependent on reranking results of the reranker. To overcome this limitation, we reformulate RAG as a cooperative multi-agent decision-making problem and propose Cooperative Retrieval-Augmented Generation (CoRAG), a framework in which the reranker and the generator act as peer decision-makers rather than being connected through an asymmetric dependency pipeline. By jointly optimizing their behaviors toward a shared task objective, the reranker and generator are encouraged to cooperate, ensuring that document reranking and generation work in concert to improve the final response. Experimental results demonstrate good generalization and improved generation stability of CoRAG, even when the model is trained on only around 10K PopQA samples. Our model released in this https URL 

---
# MiSCHiEF: A Benchmark in Minimal-Pairs of Safety and Culture for Holistic Evaluation of Fine-Grained Image-Caption Alignment 

**Authors**: Sagarika Banerjee, Tangatar Madi, Advait Swaminathan, Nguyen Dao Minh Anh, Shivank Garg, Kevin Zhu, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2602.18729)  

**Abstract**: Fine-grained image-caption alignment is crucial for vision-language models (VLMs), especially in socially critical contexts such as identifying real-world risk scenarios or distinguishing cultural proxies, where correct interpretation hinges on subtle visual or linguistic clues and where minor misinterpretations can lead to significant real-world consequences. We present MiSCHiEF, a set of two benchmarking datasets based on a contrastive pair design in the domains of safety (MiS) and culture (MiC), and evaluate four VLMs on tasks requiring fine-grained differentiation of paired images and captions. In both datasets, each sample contains two minimally differing captions and corresponding minimally differing images. In MiS, the image-caption pairs depict a safe and an unsafe scenario, while in MiC, they depict cultural proxies in two distinct cultural contexts. We find that models generally perform better at confirming the correct image-caption pair than rejecting incorrect ones. Additionally, models achieve higher accuracy when selecting the correct caption from two highly similar captions for a given image, compared to the converse task. The results, overall, highlight persistent modality misalignment challenges in current VLMs, underscoring the difficulty of precise cross-modal grounding required for applications with subtle semantic and visual distinctions. 

---
# Temporal Action Representation Learning for Tactical Resource Control and Subsequent Maneuver Generation 

**Authors**: Hoseong Jung, Sungil Son, Daesol Cho, Jonghae Park, Changhyun Choi, H. Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.18716)  

**Abstract**: Autonomous robotic systems should reason about resource control and its impact on subsequent maneuvers, especially when operating with limited energy budgets or restricted sensing. Learning-based control is effective in handling complex dynamics and represents the problem as a hybrid action space unifying discrete resource usage and continuous maneuvers. However, prior works on hybrid action space have not sufficiently captured the causal dependencies between resource usage and maneuvers. They have also overlooked the multi-modal nature of tactical decisions, both of which are critical in fast-evolving scenarios. In this paper, we propose TART, a Temporal Action Representation learning framework for Tactical resource control and subsequent maneuver generation. TART leverages contrastive learning based on a mutual information objective, designed to capture inherent temporal dependencies in resource-maneuver interactions. These learned representations are quantized into discrete codebook entries that condition the policy, capturing recurring tactical patterns and enabling multi-modal and temporally coherent behaviors. We evaluate TART in two domains where resource deployment is critical: (i) a maze navigation task where a limited budget of discrete actions provides enhanced mobility, and (ii) a high-fidelity air combat simulator in which an F-16 agent operates weapons and defensive systems in coordination with flight maneuvers. Across both domains, TART consistently outperforms hybrid-action baselines, demonstrating its effectiveness in leveraging limited resources and producing context-aware subsequent maneuvers. 

---
# EDU-MATRIX: A Society-Centric Generative Cognitive Digital Twin Architecture for Secondary Education 

**Authors**: Wenjing Zhai, Jianbin Zhang, Tao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.18705)  

**Abstract**: Existing multi-agent simulations often suffer from the "Agent-Centric Paradox": rules are hard-coded into individual agents, making complex social dynamics rigid and difficult to align with educational values. This paper presents EDU-MATRIX, a society-centric generative cognitive digital twin architecture that shifts the paradigm from simulating "people" to simulating a "social space with a gravitational field." We introduce three architectural contributions: (1) An Environment Context Injection Engine (ECIE), which acts as a "social microkernel," dynamically injecting institutional rules (Gravity) into agents based on their spatial-temporal coordinates; (2) A Modular Logic Evolution Protocol (MLEP), where knowledge exists as "fluid" capsules that agents synthesize to generate new paradigms, ensuring high dialogue consistency (94.1%); and (3) Endogenous Alignment via Role-Topology, where safety constraints emerge from the agent's position in the social graph rather than external filters. Deployed as a digital twin of a secondary school with 2,400 agents, the system demonstrates how "social gravity" (rules) and "cognitive fluids" (knowledge) interact to produce emergent, value-aligned behaviors (Social Clustering Coefficient: 0.72). 

---
# Think with Grounding: Curriculum Reinforced Reasoning with Video Grounding for Long Video Understanding 

**Authors**: Houlun Chen, Xin Wang, Guangyao Li, Yuwei Zhou, Yihan Chen, Jia Jia, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.18702)  

**Abstract**: Long video understanding is challenging due to rich and complicated multimodal clues in long temporal this http URL methods adopt reasoning to improve the model's ability to analyze complex video clues in long videos via text-form this http URL,the existing literature suffers from the fact that the text-only reasoning under fixed video context may exacerbate hallucinations since detailed crucial clues are often ignored under limited video context length due to the temporal redundancy of long this http URL address this gap,we propose Video-TwG,a curriculum reinforced framework that employs a novel Think-with-Grounding paradigm,enabling video LLMs to actively decide when to perform on-demand grounding during interleaved text-video reasoning, selectively zooming into question-relevant clips only when this http URL-TwG can be trained end-to-end in a straightforward manner, without relying on complex auxiliary modules or heavily annotated reasoning tracesIn detail,we design a Two-stage Reinforced Curriculum Strategy, where the model first learns think-with-grounding behavior on a small short-video GQA dataset with grounding labels,and then scales to diverse general QA data with videos of diverse domains to encourage generalization. Further, to handle complex think-with-grounding reasoning for various kinds of data,we propose TwG-GRPO algorithm which features the fine-grained grounding reward, self-confirmed pseudo reward and accuracy-gated this http URL,we propose to construct a new TwG-51K dataset that facilitates training. Experiments on Video-MME, LongVideoBench, and MLVU show that Video-TwG consistently outperforms strong LVU this http URL ablation validates the necessity of our Two-stage Reinforced Curriculum Strategy and shows our TwG-GRPO better leverages diverse unlabeled data to improve grounding quality and reduce redundant groundings without sacrificing QA performance. 

---
# Semantic Substrate Theory: An Operator-Theoretic Framework for Geometric Semantic Drift 

**Authors**: Stephen Russell  

**Link**: [PDF](https://arxiv.org/pdf/2602.18699)  

**Abstract**: Most semantic drift studies report multiple signals e.g., embedding displacement, neighbor changes, distributional divergence, and recursive trajectory instability, without a shared explanatory theory that relates them. This paper proposes a formalization of these signals in one time-indexed substrate, $S_t=(X,d_t,P_t)$, combining embedding geometry with local diffusion. Within this substrate, node-level neighborhood drift measures changes in local conditional distributions, coarse Ricci curvature measures local contractivity of semantic diffusion, and recursive drift probes stability of iterated semantic operators. This manuscript specifies the formal model, assumptions, and tests that can refute the model. Herein, the paper introduces bridge mass, a node-level aggregate of incident negative curvature, as a predictor of future neighborhood rewiring. This paper provides the theory and test contracts; empirical performance is deferred to subsequent studies. 

---
# In-Context Planning with Latent Temporal Abstractions 

**Authors**: Baiting Luo, Yunuo Zhang, Nathaniel S. Keplinger, Samir Gupta, Abhishek Dubey, Ayan Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2602.18694)  

**Abstract**: Planning-based reinforcement learning for continuous control is bottlenecked by two practical issues: planning at primitive time scales leads to prohibitive branching and long horizons, while real environments are frequently partially observable and exhibit regime shifts that invalidate stationary, fully observed dynamics assumptions. We introduce I-TAP (In-Context Latent Temporal-Abstraction Planner), an offline RL framework that unifies in-context adaptation with online planning in a learned discrete temporal-abstraction space. From offline trajectories, I-TAP learns an observation-conditioned residual-quantization VAE that compresses each observation-macro-action segment into a coarse-to-fine stack of discrete residual tokens, and a temporal Transformer that autoregressively predicts these token stacks from a short recent history. The resulting sequence model acts simultaneously as a context-conditioned prior over abstract actions and a latent dynamics model. At test time, I-TAP performs Monte Carlo Tree Search directly in token space, using short histories for implicit adaptation without gradient update, and decodes selected token stacks into executable actions. Across deterministic MuJoCo, stochastic MuJoCo with per-episode latent dynamics regimes, and high-dimensional Adroit manipulation, including partially observable variants, I-TAP consistently matches or outperforms strong model-free and model-based offline baselines, demonstrating efficient and robust in-context planning under stochastic dynamics and partial observability. 

---
# Heterogeneity-agnostic AI/ML-assisted beam selection for multi-panel arrays 

**Authors**: Ibrahim Kilinc, Robert W. Heath Jr  

**Link**: [PDF](https://arxiv.org/pdf/2602.18678)  

**Abstract**: AI/ML-based beam selection methods coupled with location information effectively reduce beam training overhead. Unfortunately, heterogeneous antenna hardware with varying dimensions, orientations, codebooks, element patterns, and polarization angles limits their feasibility and generalization. This challenge requires either a heterogeneity-agnostic model functional under these variations, or developing many models for each configuration, which is infeasible and expensive in practice. In this paper, we propose a unifying AI/ML-based beam selection algorithm supporting antenna heterogeneity by predicting wireless propagation characteristics independent of antenna configuration. We derive a reference signal received power (RSRP) model that decouples propagation characteristics from antenna configuration. We propose an optimization framework to extract propagation variables consisting of angle-of-arrival (AoA), angle-of-departure (AoD), and a matrix incorporating path gain and channel depolarization from beamformed RSRP measurements. We develop a three-stage autoregressive network to predict these variables from user location, enabling RSRP calculation and beam selection for arbitrary antenna configurations without retraining or having a separate model for each configuration. Simulation results show our heterogeneity-agnostic method provides spectral efficiency close to that of genie-aided selection both with and without antenna heterogeneity. 

---
# NutriOrion: A Hierarchical Multi-Agent Framework for Personalized Nutrition Intervention Grounded in Clinical Guidelines 

**Authors**: Junwei Wu, Runze Yan, Hanqi Luo, Darren Liu, Minxiao Wang, Kimberly L. Townsend, Lydia S. Hartwig, Derek Milketinas, Xiao Hu, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18650)  

**Abstract**: Personalized nutrition intervention for patients with multimorbidity is critical for improving health outcomes, yet remains challenging because it requires the simultaneous integration of heterogeneous clinical conditions, medications, and dietary guidelines. Single-agent large language models (LLMs) often suffer from context overload and attention dilution when processing such high-dimensional patient profiles. We introduce NutriOrion, a hierarchical multi-agent framework with a parallel-then-sequential reasoning topology. NutriOrion decomposes nutrition planning into specialized domain agents with isolated contexts to mitigate anchoring bias, followed by a conditional refinement stage. The framework includes a multi-objective prioritization algorithm to resolve conflicting dietary requirements and a safety constraint mechanism that injects pharmacological contraindications as hard negative constraints during synthesis, ensuring clinical validity by construction rather than post-hoc filtering. For clinical interoperability, NutriOrion maps synthesized insights into the ADIME standard and FHIR R4 resources. Evaluated on 330 stroke patients with multimorbidity, NutriOrion outperforms multiple baselines, including GPT-4.1 and alternative multi-agent architectures. It achieves a 12.1 percent drug-food interaction violation rate, demonstrates strong personalization with negative correlations (-0.26 to -0.35) between patient biomarkers and recommended risk nutrients, and yields clinically meaningful dietary improvements, including a 167 percent increase in fiber and a 27 percent increase in potassium, alongside reductions in sodium (9 percent) and sugars (12 percent). 

---
# Global Low-Rank, Local Full-Rank: The Holographic Encoding of Learned Algorithms 

**Authors**: Yongzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.18649)  

**Abstract**: Grokking -- the abrupt transition from memorization to generalization after extended training -- has been linked to the emergence of low-dimensional structure in learning dynamics. Yet neural network parameters inhabit extremely high-dimensional spaces. How can a low-dimensional learning process produce solutions that resist low-dimensional compression?
We investigate this question in multi-task modular arithmetic, training shared-trunk Transformers with separate heads for addition, multiplication, and a quadratic operation modulo 97. Across three model scales (315K--2.2M parameters) and five weight decay settings, we compare three reconstruction methods: per-matrix SVD, joint cross-matrix SVD, and trajectory PCA.
Across all conditions, grokking trajectories are confined to a 2--6 dimensional global subspace, while individual weight matrices remain effectively full-rank. Reconstruction from 3--5 trajectory PCs recovers over 95\% of final accuracy, whereas both per-matrix and joint SVD fail at sub-full rank. Even when static decompositions capture most spectral energy, they destroy task-relevant structure.
These results show that learned algorithms are encoded through dynamically coordinated updates spanning all matrices, rather than localized low-rank components. We term this the holographic encoding principle: grokked solutions are globally low-rank in the space of learning directions but locally full-rank in parameter space, with implications for compression, interpretability, and understanding how neural networks encode computation. 

---
# Information-Guided Noise Allocation for Efficient Diffusion Training 

**Authors**: Gabriel Raya, Bac Nguyen, Georgios Batzolis, Yuhta Takida, Dejan Stancevic, Naoki Murata, Chieh-Hsin Lai, Yuki Mitsufuji, Luca Ambrogioni  

**Link**: [PDF](https://arxiv.org/pdf/2602.18647)  

**Abstract**: Training diffusion models typically relies on manually tuned noise schedules, which can waste computation on weakly informative noise regions and limit transfer across datasets, resolutions, and representations. We revisit noise schedule allocation through an information-theoretic lens and propose the conditional entropy rate of the forward process as a theoretically grounded, data-dependent diagnostic for identifying suboptimal noise-level allocation in existing schedules. Based on these insight, we introduce InfoNoise, a principled data-adaptive training noise schedule that replaces heuristic schedule design with an information-guided noise sampling distribution derived from entropy-reduction rates estimated from denoising losses already computed during training. Across natural-image benchmarks, InfoNoise matches or surpasses tuned EDM-style schedules, in some cases with a substantial training speedup (about $1.4\times$ on CIFAR-10). On discrete datasets, where standard image-tuned schedules exhibit significant mismatch, it reaches superior quality in up to $3\times$ fewer training steps. Overall, InfoNoise makes noise scheduling data-adaptive, reducing the need for per-dataset schedule design as diffusion models expand across domains. 

---
# Lost in Instructions: Study of Blind Users' Experiences with DIY Manuals and AI-Rewritten Instructions for Assembly, Operation, and Troubleshooting of Tangible Products 

**Authors**: Monalika Padma Reddy, Aruna Balasubramanian, Jiawei Zhou, Xiaojun Bi, IV Ramakrishnan, Vikas Ashok  

**Link**: [PDF](https://arxiv.org/pdf/2602.18630)  

**Abstract**: AI tools like ChatGPT and Be-My-AI are increasingly being used by blind individuals. Although prior work has explored their use in some Do-It-Yourself (DIY) tasks by blind individuals, little is known about how they use these tools and the available product-manual resources to assemble, operate, and troubleshoot physical or tangible products - tasks requiring spatial reasoning, structural understanding, and precise execution. We address this knowledge gap via an interview study and a usability study with blind participants, investigating how they leverage AI tools and product manuals for DIY tasks with physical products. Findings show that manuals are essential resources, but product-manual instructions are often inadequate for blind users. AI tools presently do not adequately address this insufficiency; in fact, we observed that they often exacerbate this issue with incomplete, incoherent, or misleading guidance. Lastly, we suggest improvements to AI tools for generating tailored instructions for blind users' DIY tasks involving tangible products. 

---
# Non-Interfering Weight Fields: Treating Model Parameters as a Continuously Extensible Function 

**Authors**: Sarim Chaudhry  

**Link**: [PDF](https://arxiv.org/pdf/2602.18628)  

**Abstract**: Large language models store all learned knowledge in a single, fixed weight vector. Teaching a model new capabilities requires modifying those same weights, inevitably degrading previously acquired knowledge. This fundamental limitation, known as catastrophic forgetting, has resisted principled solutions for decades. Existing approaches treat weights as immutable artifacts that must be protected through techniques like regularization heuristics, replay buffers, or isolated adapter modules. The problem is none of these provide a structural guarantee against forgetting. In this work, we propose Non-Interfering Weight Fields (NIWF), a framework that replaces the fixed weight paradigm with a learned function that generates weight configurations on demand from a continuous capability coordinate space. After training on a task, we commit the occupied coordinate region by snapshotting the fields outputs on anchor points to enforce a functional lock during all future training. We validate NIWF on sequential instructionfollowing and code generation tasks using Mistral-7B, demonstrating zero forgetting on committed tasks with competitive perplexity on new tasks. The framework introduces the notion of software-like versioning for neural network intelligence, where capabilities can be committed, extended, composed, and rolled back without retraining. 

---
# Finding the Signal in the Noise: An Exploratory Study on Assessing the Effectiveness of AI and Accessibility Forums for Blind Users' Support Needs 

**Authors**: Satwik Ram Kodandaram, Jiawei Zhou, Xiaojun Bi, IV Ramakrishnan, Vikas Ashok  

**Link**: [PDF](https://arxiv.org/pdf/2602.18623)  

**Abstract**: Accessibility forums and, more recently, generative AI tools have become vital resources for blind users seeking solutions to computer-interaction issues and learning about new assistive technologies, screen reader features, tutorials, and software updates. Understanding user experiences with these resources is essential for identifying and addressing persistent support gaps. Towards this, we interviewed 14 blind users who regularly engage with forums and GenAI tools. Findings revealed that forums often overwhelm users with multiple overlapping topics, redundant or irrelevant content, and fragmented responses that must be mentally pieced together, increasing cognitive load. GenAI tools, while offering more direct assistance, introduce new barriers by producing unreliable answers, including overly verbose or fragmented guidance, fabricated information, and contradictory suggestions that fail to follow prompts, thereby heightening verification demands. Based on these insights, we outlined design opportunities to improve the reliability of assistive resources, aiming to provide blind users with more trustworthy and cognitively-manageable support. 

---
# DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction 

**Authors**: Jiayang Shi, Daniel M. Pelt, K. Joost Batenburg  

**Link**: [PDF](https://arxiv.org/pdf/2602.18589)  

**Abstract**: Diffusion models have recently emerged as powerful priors for solving inverse problems. While computed tomography (CT) is theoretically a linear inverse problem, it poses many practical challenges. These include correlated noise, artifact structures, reliance on system geometry, and misaligned value ranges, which make the direct application of diffusion models more difficult than in domains like natural image generation. To systematically evaluate how diffusion models perform in this context and compare them with established reconstruction methods, we introduce DM4CT, a comprehensive benchmark for CT reconstruction. DM4CT includes datasets from both medical and industrial domains with sparse-view and noisy configurations. To explore the challenges of deploying diffusion models in practice, we additionally acquire a high-resolution CT dataset at a high-energy synchrotron facility and evaluate all methods under real experimental conditions. We benchmark ten recent diffusion-based methods alongside seven strong baselines, including model-based, unsupervised, and supervised approaches. Our analysis provides detailed insights into the behavior, strengths, and limitations of diffusion models for CT reconstruction. The real-world dataset is publicly available at this http URL, and the codebase is open-sourced at this http URL. 

---
# BloomNet: Exploring Single vs. Multiple Object Annotation for Flower Recognition Using YOLO Variants 

**Authors**: Safwat Nusrat, Prithwiraj Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2602.18585)  

**Abstract**: Precise localization and recognition of flowers are crucial for advancing automated agriculture, particularly in plant phenotyping, crop estimation, and yield monitoring. This paper benchmarks several YOLO architectures such as YOLOv5s, YOLOv8n/s/m, and YOLOv12n for flower object detection under two annotation regimes: single-image single-bounding box (SISBB) and single-image multiple-bounding box (SIMBB). The FloralSix dataset, comprising 2,816 high-resolution photos of six different flower species, is also introduced. It is annotated for both dense (clustered) and sparse (isolated) scenarios. The models were evaluated using Precision, Recall, and Mean Average Precision (mAP) at IoU thresholds of 0.5 (mAP@0.5) and 0.5-0.95 (mAP@0.5:0.95). In SISBB, YOLOv8m (SGD) achieved the best results with Precision 0.956, Recall 0.951, mAP@0.5 0.978, and mAP@0.5:0.95 0.865, illustrating strong accuracy in detecting isolated flowers. With mAP@0.5 0.934 and mAP@0.5:0.95 0.752, YOLOv12n (SGD) outperformed the more complicated SIMBB scenario, proving robustness in dense, multi-object detection. Results show how annotation density, IoU thresholds, and model size interact: recall-optimized models perform better in crowded environments, whereas precision-oriented models perform best in sparse scenarios. In both cases, the Stochastic Gradient Descent (SGD) optimizer consistently performed better than alternatives. These density-sensitive sensors are helpful for non-destructive crop analysis, growth tracking, robotic pollination, and stress evaluation. 

---
# GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry 

**Authors**: Guanghui Min, Tianhao Huang, Ke Wan, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.18584)  

**Abstract**: Targeted data selection has emerged as a crucial paradigm for efficient instruction tuning, aiming to identify a small yet influential subset of training examples for a specific target task. In practice, influence is often measured through the effect of an example on parameter updates. To make selection scalable, many approaches leverage optimizer statistics (e.g., Adam states) as an axis-aligned surrogate for update geometry (i.e., diagonal precondition), implicitly treating parameters as coordinate-wise independent. We show that this assumption breaks down in parameter-efficient fine-tuning (PEFT) methods such as LoRA. In this setting, the induced optimization geometry exhibits strong cross-parameter coupling with non-trivial off-diagonal interactions, while the task-relevant update directions are confined to a low-dimensional subspace. Motivated by this mismatch, we propose GIST (Gradient Isometric Subspace Transformation), a simple yet principled alternative that replaces axis-aligned scaling with robust subspace alignment. GIST recovers a task-specific subspace from validation gradients via spectral filtering (SVD), projects training gradients into this coupled subspace, and scores examples by their alignment with target this http URL experiments have demonstrated that GIST matches or outperforms the state-of-the-art baseline with only 0.29% of the storage and 25% of the computational time under the same selection budget. 

---
# Luna-2: Scalable Single-Token Evaluation with Small Language Models 

**Authors**: Vatsal Goel, Rishon Dsouza, Nikhil Ega, Amey Ramesh Rambatla, Rob Friel, Shuai Shao, Yash Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2602.18583)  

**Abstract**: Real-time guardrails require evaluation that is accurate, cheap, and fast - yet today's default, LLM-as-a-judge (LLMAJ), is slow, expensive, and operationally non-deterministic due to multi-token generation. We present Luna-2, a novel architecture that leverages decoder-only small language models (SLMs) into a deterministic evaluation model to reliably compute complex task-specific LLMAJ metrics (e.g. toxicity, hallucination, tool selection quality, etc.) at an accuracy at par or higher than LLMAJ using frontier LLMs while drastically reducing the cost and latency of computation. Each metric is implemented as a lightweight LoRA/PEFT head on top of a shared SLM backbone, enabling hundreds of specialized metrics to run concurrently on a single GPU, deployable locally next to AI systems in a privacy-preserving and latency optimizing manner. Across content safety and hallucination benchmarks, Luna-2 matches the accuracy of state-of-the-art LLM-based evaluators while reducing inference cost by over 80x and latency by over 20x.
In this paper, we outline the model architecture, training methodology and report real-world empirical results on accuracy, latency, and throughput results. In production, Luna-2 is protecting 100M+ AI sessions and processing over 100B tokens per month for our customers with eval cost savings of over $30M annually. 

---
# Debug2Fix: Supercharging Coding Agents with Interactive Debugging Capabilities 

**Authors**: Spandan Garg, Yufan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18571)  

**Abstract**: While significant progress has been made in automating various aspects of software development through coding agents, there is still significant room for improvement in their bug fixing capabilities. Debugging and investigation of runtime behavior remains largely a manual, developer-driven process. Popular coding agents typically rely on either static analysis of the code or iterative test-fix cycles, which is akin to trial and error debugging. We posit that there is a wealth of rich runtime information that developers routinely access while debugging code, which agents are currently deprived of due to design limitations. Despite how prevalent debuggers are in modern IDEs and command-line tools, they have surprisingly not made their way into coding agents. In this work, we introduce Debug2Fix, a novel framework that incorporates interactive debugging as a core component of a software engineering agent via a subagent architecture. We incorporate debuggers for Java and Python into our agent framework and evaluate against GitBug-Java and SWE-Bench-Live and achieve >20% improvement in performance compared to the baseline for certain models. Furthermore, using our framework, we're able to make weaker models like GPT-5 and Claude Haiku 4.5 match or exceed the performances of stronger models like Claude Sonnet 4.5, showing that better tool design is often just as important as switching to a more expensive model. Finally, we conduct systematic ablations demonstrating the importance of both the subagent architecture and debugger integration. 

---
# RPU -- A Reasoning Processing Unit 

**Authors**: Matthew Adiletta, Gu-Yeon Wei, David Brooks  

**Link**: [PDF](https://arxiv.org/pdf/2602.18568)  

**Abstract**: Large language model (LLM) inference performance is increasingly bottlenecked by the memory wall. While GPUs continue to scale raw compute throughput, they struggle to deliver scalable performance for memory bandwidth bound workloads. This challenge is amplified by emerging reasoning LLM applications, where long output sequences, low arithmetic intensity, and tight latency constraints demand significantly higher memory bandwidth. As a result, system utilization drops and energy per inference rises, highlighting the need for an optimized system architecture for scalable memory bandwidth.
To address these challenges we present the Reasoning Processing Unit (RPU), a chiplet-based architecture designed to address the challenges of the modern memory wall. RPU introduces: (1) A Capacity-Optimized High-Bandwidth Memory (HBM-CO) that trades capacity for lower energy and cost; (2) a scalable chiplet architecture featuring a bandwidth-first power and area provisioning design; and (3) a decoupled microarchitecture that separates memory, compute, and communication pipelines to sustain high bandwidth utilization. Simulation results show that RPU performs up to 45.3x lower latency and 18.6x higher throughput over an H100 system at ISO-TDP on Llama3-405B. 

---
# From Static Spectra to Operando Infrared Dynamics: Physics Informed Flow Modeling and a Benchmark 

**Authors**: Shuquan Ye, Ben Fei, Hongbin Xu, Jiaying Lin, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18551)  

**Abstract**: The Solid Electrolyte Interphase (SEI) is critical to the performance of lithium-ion batteries, yet its analysis via Operando Infrared (IR) spectroscopy remains experimentally complex and expensive, which limits its accessibility for standard research facilities. To overcome this bottleneck, we formulate a novel task, Operando IR Prediction, which aims to forecast the time-resolved evolution of spectral ``fingerprints'' from a single static spectrum. To facilitate this, we introduce OpIRSpec-7K, the first large-scale operando dataset comprising 7,118 high-quality samples across 10 distinct battery systems, alongside OpIRBench, a comprehensive evaluation benchmark with carefully designed protocols. Addressing the limitations of standard spectrum, video, and sequence models in capturing voltage-driven chemical dynamics and complex composition, we propose Aligned Bi-stream Chemical Constraint (ABCC), an end-to-end physics-aware framework. It reformulates MeanFlow and introduces a novel Chemical Flow to explicitly model reaction trajectories, employs a two-stream disentanglement mechanism for solvent-SEI separation, and enforces physics and spectrum constraints such as mass conservation and peak shifts. ABCC significantly outperforms state-of-the-art static, sequential, and generative baselines. ABCC even generalizes to unseen systems and enables interpretable downstream recovery of SEI formation pathways, supporting AI-driven electrochemical discovery. 

---
# 1D-Bench: A Benchmark for Iterative UI Code Generation with Visual Feedback in Real-World 

**Authors**: Qiao Xu, Yipeng Yu, Chengxiao Feng, Xu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.18548)  

**Abstract**: Design-to-code translates high-fidelity UI designs into executable front-end implementations, but progress remains hard to compare due to inconsistent datasets, toolchains, and evaluation protocols. We introduce 1D-Bench, a benchmark grounded in real e-commerce workflows, where each instance provides a reference rendering and an exported intermediate representation that may contain extraction errors. 1D is short for one day, representing the efficient completion of design-to-code tasks in less than one day. Models take both as input, using the intermediate representation as structural cues while being evaluated against the reference rendering, which tests robustness to intermediate representation defects rather than literal adherence.
1D-Bench requires generating an executable React codebase under a fixed toolchain with an explicit component hierarchy, and defines a multi-round setting in which models iteratively apply component-level edits using execution feedback. Experiments on commercial and open-weight multimodal models show that iterative editing generally improves final performance by increasing rendering success and often improving visual similarity. We further conduct a pilot study on post-training with synthetic repair trajectories and reinforcement learning based editing, and observe limited and unstable gains that may stem from sparse terminal rewards and high-variance file-level updates. 

---
# Rodent-Bench 

**Authors**: Thomas Heap, Laurence Aitchison, Emma Cahill, Adriana Casado Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2602.18540)  

**Abstract**: We present Rodent-Bench, a novel benchmark designed to evaluate the ability of Multimodal Large Language Models (MLLMs) to annotate rodent behaviour footage. We evaluate state-of-the-art MLLMs, including Gemini-2.5-Pro, Gemini-2.5-Flash and Qwen-VL-Max, using this benchmark and find that none of these models perform strongly enough to be used as an assistant for this task. Our benchmark encompasses diverse datasets spanning multiple behavioral paradigms including social interactions, grooming, scratching, and freezing behaviors, with videos ranging from 10 minutes to 35 minutes in length. We provide two benchmark versions to accommodate varying model capabilities and establish standardized evaluation metrics including second-wise accuracy, macro F1, mean average precision, mutual information, and Matthew's correlation coefficient. While some models show modest performance on certain datasets (notably grooming detection), overall results reveal significant challenges in temporal segmentation, handling extended video sequences, and distinguishing subtle behavioral states. Our analysis identifies key limitations in current MLLMs for scientific video annotation and provides insights for future model development. Rodent-Bench serves as a foundation for tracking progress toward reliable automated behavioral annotation in neuroscience research. 

---
# Fairness-Aware Partial-label Domain Adaptation for Voice Classification of Parkinson's and ALS 

**Authors**: Arianna Francesconi, Zhixiang Dai, Arthur Stefano Moscheni, Himesh Morgan Perera Kanattage, Donato Cappetta, Fabio Rebecchi, Paolo Soda, Valerio Guarrasi, Rosa Sicilia, Mary-Anne Hartley  

**Link**: [PDF](https://arxiv.org/pdf/2602.18535)  

**Abstract**: Voice-based digital biomarkers can enable scalable, non-invasive screening and monitoring of Parkinson's disease (PD) and Amyotrophic Lateral Sclerosis (ALS). However, models trained on one cohort or device often fail on new acquisition settings due to cross-device and cross-cohort domain shift. This challenge is amplified in real-world scenarios with partial-label mismatch, where datasets may contain different disease labels and only partially overlap in class space. In addition, voice-based models may exploit demographic cues, raising concerns about gender-related unfairness, particularly when deployed across heterogeneous cohorts. To tackle these challenges, we propose a hybrid framework for unified three-class (healthy/PD/ALS) cross-domain voice classification from partially overlapping cohorts. The method combines style-based domain generalization with conditional adversarial alignment tailored to partial-label settings, reducing negative transfer. An additional adversarial gender branch promotes gender-invariant representations. We conduct a comprehensive evaluation across four heterogeneous sustained-vowel datasets, spanning distinct acquisition settings and devices, under both domain generalization and unsupervised domain adaptation protocols. The proposed approach is compared against twelve state-of-the-art machine learning and deep learning methods, and further evaluated through three targeted ablations, providing the first cross-cohort benchmark and end-to-end domain-adaptive framework for unified healthy/PD/ALS voice classification under partial-label mismatch and fairness constraints. Across all experimental settings, our method consistently achieves the best external generalization over the considered evaluation metrics, while maintaining reduced gender disparities. Notably, no competing method shows statistically significant gains in external performance. 

---
# VLANeXt: Recipes for Building Strong VLA Models 

**Authors**: Xiao-Ming Wu, Bin Fan, Kang Liao, Jian-Jian Jiang, Runze Yang, Yihang Luo, Zhonghua Wu, Wei-Shi Zheng, Chen Change Loy  

**Link**: [PDF](https://arxiv.org/pdf/2602.18532)  

**Abstract**: Following the rise of large foundation models, Vision-Language-Action models (VLAs) emerged, leveraging strong visual and language understanding for general-purpose policy learning. Yet, the current VLA landscape remains fragmented and exploratory. Although many groups have proposed their own VLA models, inconsistencies in training protocols and evaluation settings make it difficult to identify which design choices truly matter. To bring structure to this evolving space, we reexamine the VLA design space under a unified framework and evaluation setup. Starting from a simple VLA baseline similar to RT-2 and OpenVLA, we systematically dissect design choices along three dimensions: foundational components, perception essentials, and action modelling perspectives. From this study, we distill 12 key findings that together form a practical recipe for building strong VLA models. The outcome of this exploration is a simple yet effective model, VLANeXt. VLANeXt outperforms prior state-of-the-art methods on the LIBERO and LIBERO-plus benchmarks and demonstrates strong generalization in real-world experiments. We will release a unified, easy-to-use codebase that serves as a common platform for the community to reproduce our findings, explore the design space, and build new VLA variants on top of a shared foundation. 

---
# Deep Reinforcement Learning for Optimizing Energy Consumption in Smart Grid Systems 

**Authors**: Abeer Alsheikhi, Amirfarhad Farhadi, Azadeh Zamanifar  

**Link**: [PDF](https://arxiv.org/pdf/2602.18531)  

**Abstract**: The energy management problem in the context of smart grids is inherently complex due to the interdependencies among diverse system components. Although Reinforcement Learning (RL) has been proposed for solving Optimal Power Flow (OPF) problems, the requirement for iterative interaction with an environment often necessitates computationally expensive simulators, leading to significant sample inefficiency. In this study, these challenges are addressed through the use of Physics-Informed Neural Networks (PINNs), which can replace conventional and costly smart grid simulators. The RL policy learning process is enhanced so that convergence can be achieved in a fraction of the time required by the original environment. The PINN-based surrogate is compared with other benchmark data-driven surrogate models. By incorporating knowledge of the underlying physical laws, the results show that the PINN surrogate is the only approach considered in this context that can obtain a strong RL policy even without access to samples from the true simulator. The results demonstrate that using PINN surrogates can accelerate training by 50% compared to RL training without a surrogate. This approach enables the rapid generation of performance scores similar to those produced by the original simulator. 

---
# JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments 

**Authors**: Zhan Liu, Changli Tang, Yuxin Wang, Zhiyuan Zhu, Youjun Chen, Yiwen Shao, Tianzi Wang, Lei Ke, Zengrui Jin, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18527)  

**Abstract**: Current audio-visual large language models (AV-LLMs) are predominantly restricted to 2D perception, relying on RGB video and monaural audio. This design choice introduces a fundamental dimensionality mismatch that precludes reliable source localization and spatial reasoning in complex 3D environments. We address this limitation by presenting JAEGER, a framework that extends AV-LLMs to 3D space, to enable joint spatial grounding and reasoning through the integration of RGB-D observations and multi-channel first-order ambisonics. A core contribution of our work is the neural intensity vector (Neural IV), a learned spatial audio representation that encodes robust directional cues to enhance direction-of-arrival estimation, even in adverse acoustic scenarios with overlapping sources. To facilitate large-scale training and systematic evaluation, we propose SpatialSceneQA, a benchmark of 61k instruction-tuning samples curated from simulated physical environments. Extensive experiments demonstrate that our approach consistently surpasses 2D-centric baselines across diverse spatial perception and reasoning tasks, underscoring the necessity of explicit 3D modelling for advancing AI in physical environments. Our source code, pre-trained model checkpoints and datasets will be released upon acceptance. 

---
# The Geometry of Multi-Task Grokking: Transverse Instability, Superposition, and Weight Decay Phase Structure 

**Authors**: Yongzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.18523)  

**Abstract**: Grokking -- the abrupt transition from memorization to generalization long after near-zero training loss -- has been studied mainly in single-task settings. We extend geometric analysis to multi-task modular arithmetic, training shared-trunk Transformers on dual-task (mod-add + mod-mul) and tri-task (mod-add + mod-mul + mod-sq) objectives across a systematic weight decay sweep. Five consistent phenomena emerge. (1) Staggered grokking order: multiplication generalizes first, followed by squaring, then addition, with consistent delays across seeds. (2) Universal integrability: optimization trajectories remain confined to an empirically invariant low-dimensional execution manifold; commutator defects orthogonal to this manifold reliably precede generalization. (3) Weight decay phase structure: grokking timescale, curvature depth, reconstruction threshold, and defect lead covary systematically with weight decay, revealing distinct dynamical regimes and a sharp no-decay failure mode. (4) Holographic incompressibility: final solutions occupy only 4--8 principal trajectory directions yet are distributed across full-rank weights and destroyed by minimal perturbations; SVD truncation, magnitude pruning, and uniform scaling all fail to preserve performance. (5) Transverse fragility and redundancy: removing less than 10% of orthogonal gradient components eliminates grokking, yet dual-task models exhibit partial recovery under extreme deletion, suggesting redundant center manifolds enabled by overparameterization. Together, these results support a dynamical picture in which multi-task grokking constructs a compact superposition subspace in parameter space, with weight decay acting as compression pressure and excess parameters supplying geometric redundancy in optimization pathways. 

---
# AdaptStress: Online Adaptive Learning for Interpretable and Personalized Stress Prediction Using Multivariate and Sparse Physiological Signals 

**Authors**: Xueyi Wang, Claudine J. C. Lamoth, Elisabeth Wilhelm  

**Link**: [PDF](https://arxiv.org/pdf/2602.18521)  

**Abstract**: Continuous stress forecasting could potentially contribute to lifestyle interventions. This paper presents a novel, explainable, and individualized approach for stress prediction using physiological data from consumer-grade smartwatches. We develop a time series forecasting model that leverages multivariate features, including heart rate variability, activity patterns, and sleep metrics, to predict stress levels across 16 temporal horizons (History window: 3, 5, 7, 9 days; forecasting window: 1, 3, 5, 7 days). Our evaluation involves 16 participants monitored for 10-15 weeks. We evaluate our approach across 16 participants, comparing against state-of-the-art time series models (Informer, TimesNet, PatchTST) and traditional baselines (CNN, LSTM, CNN-LSTM) across multiple temporal horizons. Our model achieved performance with an MSE of 0.053, MAE of 0.190, and RMSE of 0.226 in optimal settings (5-day input, 1-day prediction). A comparison with the baseline models shows that our model outperforms TimesNet, PatchTST, CNN-LSTM, LSTM, and CNN under all conditions, representing improvements of 36.9%, 25.5%, and 21.5% over the best baseline. According to the explanability analysis, sleep metrics are the most dominant and consistent stress predictors (importance: 1.1, consistency: 0.9-1.0), while activity features exhibit high inter-participant variability (0.1-0.2). Most notably, the model captures individual-specific patterns where identical features can have opposing effects across users, validating its personalization capabilities. These findings establish that consumer wearables, combined with adaptive and interpretable deep learning, can deliver relevant stress assessment adapted to individual physiological responses, providing a foundation for scalable, continuous, explainable mental health monitoring in real-world settings. 

---
# Sketch2Feedback: Grammar-in-the-Loop Framework for Rubric-Aligned Feedback on Student STEM Diagrams 

**Authors**: Aayam Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2602.18520)  

**Abstract**: Providing timely, rubric-aligned feedback on student-drawn diagrams is a persistent challenge in STEM education. While large multimodal models (LMMs) can jointly parse images and generate explanations, their tendency to hallucinate undermines trust in classroom deployments. We present Sketch2Feedback, a grammar-in-the-loop framework that decomposes the problem into four stages -- hybrid perception, symbolic graph construction, constraint checking, and constrained VLM feedback -- so that the language model verbalizes only violations verified by an upstream rule engine. We evaluate on two synthetic micro-benchmarks, FBD-10 (free-body diagrams) and Circuit-10 (circuit schematics), each with 500 images spanning standard and hard noise augmentation tiers, comparing our pipeline against end-to-end LMMs (LLaVA-1.5-7B, Qwen2-VL-7B), a vision-only detector, a YOLOv8-nano learned detector, and an ensemble oracle. On n=100 test samples per benchmark with 95% bootstrap CIs, results are mixed and instructive: Qwen2-VL-7B achieves the highest micro-F1 on both FBDs (0.570) and circuits (0.528), but with extreme hallucination rates (0.78, 0.98). An ensemble oracle that selects the best prediction per sample reaches F1=0.556 with hallucination 0.320 on FBDs, demonstrating exploitable complementarity between grammar and end-to-end approaches. Confidence thresholding at tau=0.7 reduces circuit hallucination from 0.970 to 0.880 with no F1 loss. Hard noise augmentation reveals domain-dependent robustness: FBD detection is resilient while circuit detection degrades sharply. An LLM-as-judge evaluation confirms that the grammar pipeline produces more actionable circuit feedback (4.85/5) than the end-to-end LMM (3.11/5). We release all code, datasets, and evaluation scripts. 

---
# Weak-Form Evolutionary Kolmogorov-Arnold Networks for Solving Partial Differential Equations 

**Authors**: Bongseok Kim, Jiahao Zhang, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.18515)  

**Abstract**: Partial differential equations (PDEs) form a central component of scientific computing. Among recent advances in deep learning, evolutionary neural networks have been developed to successively capture the temporal dynamics of time-dependent PDEs via parameter evolution. The parameter updates are obtained by solving a linear system derived from the governing equation residuals at each time step. However, strong-form evolutionary approaches can yield ill-conditioned linear systems due to pointwise residual discretization, and their computational cost scales unfavorably with the number of training samples. To address these limitations, we propose a weak-form evolutionary Kolmogorov-Arnold Network (KAN) for the scalable and accurate prediction of PDE solutions. We decouple the linear system size from the number of training samples through the weak formulation, leading to improved scalability compared to strong-form approaches. We also rigorously enforce boundary conditions by constructing the trial space with boundary-constrained KANs to satisfy Dirichlet and periodic conditions, and by incorporating derivative boundary conditions directly into the weak formulation for Neumann conditions. In conclusion, the proposed weak-form evolutionary KAN framework provides a stable and scalable approach for PDEs and contributes to scientific machine learning with potential relevance to future engineering applications. 

---
# Trojan Horses in Recruiting: A Red-Teaming Case Study on Indirect Prompt Injection in Standard vs. Reasoning Models 

**Authors**: Manuel Wirth  

**Link**: [PDF](https://arxiv.org/pdf/2602.18514)  

**Abstract**: As Large Language Models (LLMs) are increasingly integrated into automated decision-making pipelines, specifically within Human Resources (HR), the security implications of Indirect Prompt Injection (IPI) become critical. While a prevailing hypothesis posits that "Reasoning" or "Chain-of-Thought" Models possess safety advantages due to their ability to self-correct, emerging research suggests these capabilities may enable more sophisticated alignment failures. This qualitative Red-Teaming case study challenges the safety-through-reasoning premise using the Qwen 3 30B architecture. By subjecting both a standard instruction-tuned model and a reasoning-enhanced model to a "Trojan Horse" curriculum vitae, distinct failure modes are observed. The results suggest a complex trade-off: while the Standard Model resorted to brittle hallucinations to justify simple attacks and filtered out illogical constraints in complex scenarios, the Reasoning Model displayed a dangerous duality. It employed advanced strategic reframing to make simple attacks highly persuasive, yet exhibited "Meta-Cognitive Leakage" when faced with logically convoluted commands. This study highlights a failure mode where the cognitive load of processing complex adversarial instructions causes the injection logic to be unintentionally printed in the final output, rendering the attack more detectable by humans than in Standard Models. 

---
# Beyond Pass-by-Pass Optimization: Intent-Driven IR Optimization with Large Language Models 

**Authors**: Lei Qiu, Zi Yang, Fang Lyu, Ming Zhong, Huimin Cui, Xiaobing Feng  

**Link**: [PDF](https://arxiv.org/pdf/2602.18511)  

**Abstract**: Modern compilers optimize programs through a sequence of modular passes over intermediate representations (IR). While this pass-by-pass paradigm offers engineering benefits, it suffers from a pass coordination problem: locally beneficial transformations may block more profitable optimizations in later stages. This limitation stems from the lack of an explicit notion of optimization intent, defined as a holistic strategy for coordinating multiple transformations toward a global performance objective. Recent LLM-based approaches formulate IR optimization as an end-to-end generation task, thereby avoiding the traditional pass-by-pass structure. However, optimization intent remains implicit in these methods, forcing models to jointly infer optimization strategy and generate low-level transformations, which limits both correctness and performance. We propose IntOpt, the first intent-driven IR optimizer that explicitly separates high-level optimization intent from low-level analysis and transformation. IntOpt organizes IR optimization into three stages: intent formulation, intent refinement, and intent realization, enabling globally coordinated transformations. Experiments show that IntOpt achieves 90.5% verified correctness and 2.660x average speedup on 200-program test set, outperforming state-of-the-art LLM-based optimizers in both correctness and performance, and surpassing modern compiler with the -O3 option on 37 benchmarks with speedups of up to 272.60x. 

---
# A Computer Vision Framework for Multi-Class Detection and Tracking in Soccer Broadcast Footage 

**Authors**: Daniel Tshiani  

**Link**: [PDF](https://arxiv.org/pdf/2602.18504)  

**Abstract**: Clubs with access to expensive multi-camera setups or GPS tracking systems gain a competitive advantage through detailed data, whereas lower-budget teams are often unable to collect similar information. This paper examines whether such data can instead be extracted directly from standard broadcast footage using a single-camera computer vision pipeline. This project develops an end-to-end system that combines a YOLO object detector with the ByteTrack tracking algorithm to identify and track players, referees, goalkeepers, and the ball throughout a match. Experimental results show that the pipeline achieves high performance in detecting and tracking players and officials, with strong precision, recall, and mAP50 scores, while ball detection remains the primary challenge. Despite this limitation, our findings demonstrate that AI can extract meaningful player-level spatial information from a single broadcast camera. By reducing reliance on specialized hardware, the proposed approach enables colleges, academies, and amateur clubs to adopt scalable, data-driven analysis methods previously accessible only to professional teams, highlighting the potential for affordable computer vision-based soccer analytics. 

---
# PIPE-RDF: An LLM-Assisted Pipeline for Enterprise RDF Benchmarking 

**Authors**: Suraj Ranganath  

**Link**: [PDF](https://arxiv.org/pdf/2602.18497)  

**Abstract**: Enterprises rely on RDF knowledge graphs and SPARQL to expose operational data through natural language interfaces, yet public KGQA benchmarks do not reflect proprietary schemas, prefixes, or query distributions. We present PIPE-RDF, a three-phase pipeline that constructs schema-specific NL-SPARQL benchmarks using reverse querying, category-balanced template generation, retrieval-augmented prompting, deduplication, and execution-based validation with repair. We instantiate PIPE-RDF on a fixed-schema company-location slice (5,000 companies) derived from public RDF data and generate a balanced benchmark of 450 question-SPARQL pairs across nine categories. The pipeline achieves 100% parse and execution validity after repair, with pre-repair validity rates of 96.5%-100% across phases. We report entity diversity metrics, template coverage analysis, and cost breakdowns to support deployment planning. We release structured artifacts (CSV/JSONL, logs, figures) and operational metrics to support model evaluation and system planning in real-world settings. Code is available at this https URL. 

---
# RDBLearn: Simple In-Context Prediction Over Relational Databases 

**Authors**: Yanlin Zhang, Linjie Xu, Quan Gan, David Wipf, Minjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18495)  

**Abstract**: Recent advances in tabular in-context learning (ICL) show that a single pretrained model can adapt to new prediction tasks from a small set of labeled examples, avoiding per-task training and heavy tuning. However, many real-world tasks live in relational databases, where predictive signal is spread across multiple linked tables rather than a single flat table. We show that tabular ICL can be extended to relational prediction with a simple recipe: automatically featurize each target row using relational aggregations over its linked records, materialize the resulting augmented table, and run an off-the-shelf tabular foundation model on it. We package this approach in \textit{RDBLearn} (this https URL), an easy-to-use toolkit with a scikit-learn-style estimator interface that makes it straightforward to swap different tabular ICL backends; a complementary agent-specific interface is provided as well. Across a broad collection of RelBench and 4DBInfer datasets, RDBLearn is the best-performing foundation model approach we evaluate, at times even outperforming strong supervised baselines trained or fine-tuned on each dataset. 

---
# Learning to Remember: End-to-End Training of Memory Agents for Long-Context Reasoning 

**Authors**: Kehao Zhang, Shangtong Gui, Sheng Yang, Wei Chen, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2602.18493)  

**Abstract**: Long-context LLMs and Retrieval-Augmented Generation (RAG) systems process information passively, deferring state tracking, contradiction resolution, and evidence aggregation to query time, which becomes brittle under ultra long streams with frequent updates. We propose the Unified Memory Agent (UMA), an end-to-end reinforcement learning framework that unifies memory operations and question answering within a single policy. UMA maintains a dual memory representation: a compact core summary for global context and a structured Memory Bank that supports explicit CRUD (create, update, delete, reorganize) over key value entries, enabling proactive consolidation during streaming. To evaluate long-horizon memory behavior, we introduce Ledger-QA, a diagnostic benchmark for continuous state tracking where answers are latent values derived from accumulated updates rather than lo cal span retrieval. Across 13 datasets spanning Ledger-QA, Test-Time Learning, and Accurate Retrieval, UMA substantially outperforms long-context and RAG baselines on dynamic reasoning and learning tasks while remaining competitive on standard retrieval benchmarks, underscoring the importance of learned, end-to-end memory management. 

---
# Vibe Coding on Trial: Operating Characteristics of Unanimous LLM Juries 

**Authors**: Muhammad Aziz Ullah, Abdul Serwadda  

**Link**: [PDF](https://arxiv.org/pdf/2602.18492)  

**Abstract**: Large Language Models (LLMs) are now good enough at coding that developers can describe intent in plain language and let the tool produce the first code draft, a workflow increasingly built into tools like GitHub Copilot, Cursor, and Replit. What is missing is a reliable way to tell which model written queries are safe to accept without sending everything to a human. We study the application of an LLM jury to run this review step. We first benchmark 15 open models on 82 MySQL text to SQL tasks using an execution grounded protocol to get a clean baseline of which models are strong. From the six best models we build unanimous committees of sizes 1 through 6 that see the prompt, schema, and candidate SQL and accept it only when every member says it is correct. This rule matches safety first deployments where false accepts are more costly than false rejects. We measure true positive rate, false positive rate and Youden J and we also look at committees per generator. Our results show that single model judges are uneven, that small unanimous committees of strong models can cut false accepts while still passing many good queries, and that the exact committee composition matters significantly. 

---
# Red Teaming LLMs as Socio-Technical Practice: From Exploration and Data Creation to Evaluation 

**Authors**: Adriana Alvarado Garcia, Ruyuan Wan, Ozioma C. Oguine, Karla Badillo-Urquiola  

**Link**: [PDF](https://arxiv.org/pdf/2602.18483)  

**Abstract**: Recently, red teaming, with roots in security, has become a key evaluative approach to ensure the safety and reliability of Generative Artificial Intelligence. However, most existing work emphasizes technical benchmarks and attack success rates, leaving the socio-technical practices of how red teaming datasets are defined, created, and evaluated under-examined. Drawing on 22 interviews with practitioners who design and evaluate red teaming datasets, we examine the data practices and standards that underpin this work. Because adversarial datasets determine the scope and accuracy of model evaluations, they are critical artifacts for assessing potential harms from large language models. Our contributions are first, empirical evidence of practitioners conceptualizing red teaming and developing and evaluating red teaming datasets. Second, we reflect on how practitioners' conceptualization of risk leads to overlooking the context, interaction type, and user specificity. We conclude with three opportunities for HCI researchers to expand the conceptualization and data practices for red-teaming. 

---
# AlphaForgeBench: Benchmarking End-to-End Trading Strategy Design with Large Language Models 

**Authors**: Wentao Zhang, Mingxuan Zhao, Jincheng Gao, Jieshun You, Huaiyu Jia, Yilei Zhao, Bo An, Shuo Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.18481)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to a surge of financial benchmarks, evolving from static knowledge tests to interactive trading simulations. However, current evaluations of real-time trading performance overlook a critical failure mode: severe behavioral instability in sequential decision-making under uncertainty. We empirically show that LLM-based trading agents exhibit extreme run-to-run variance, inconsistent action sequences even under deterministic decoding, and irrational action flipping across adjacent time steps. These issues stem from stateless autoregressive architectures lacking persistent action memory, as well as sensitivity to continuous-to-discrete action mappings in portfolio allocation. As a result, many existing financial trading benchmarks produce unreliable, non-reproducible, and uninformative evaluations. To address these limitations, we propose AlphaForgeBench, a principled framework that reframes LLMs as quantitative researchers rather than execution agents. Instead of emitting trading actions, LLMs generate executable alpha factors and factor-based strategies grounded in financial reasoning. This design decouples reasoning from execution, enabling fully deterministic and reproducible evaluation while aligning with real-world quantitative research workflows. Experiments across multiple state-of-the-art LLMs show that AlphaForgeBench eliminates execution-induced instability and provides a rigorous benchmark for assessing financial reasoning, strategy formulation, and alpha discovery. 

---
# AgentCAT: An LLM Agent for Extracting and Analyzing Catalytic Reaction Data from Chemical Engineering Literature 

**Authors**: Wei Yang, Zihao Liu, Tao Tan, Xiao Hu, Hong Xie, Lulu Li Xin Li, Jianyu Han, Defu Lian, Mao Ye  

**Link**: [PDF](https://arxiv.org/pdf/2602.18479)  

**Abstract**: This paper presents a large language model (LLM) agent named AgentCAT, which extracts and analyzes catalytic reaction data from chemical engineering papers, %and supports natural language based interactive analysis of the extracted data. AgentCAT serves as an alternative to overcome the long-standing data bottleneck in chemical engineering field, and its natural language based interactive data analysis functionality is friendly to the community. AgentCAT also presents a formal abstraction and challenge analysis of the catalytic reaction data extraction task in an artificial intelligence-friendly manner. This abstraction would help the artificial intelligence community understand this problem and in turn would attract more attention to address it. Technically, the complex catalytic process leads to complicated dependency structure in catalytic reaction data with respect to elementary reaction steps, molecular behaviors, measurement evidence, etc. This dependency structure makes it challenging to guarantee the correctness and completeness of data extraction, as well as representing them for analysis. AgentCAT addresses this challenge and it makes four folds of technical contributions: (1) a schema-governed extraction pipeline with progressive schema evolution, enabling robust data extraction from chemical engineering papers; (2) a dependency-aware reaction-network knowledge graph that links catalysts/active sites, synthesis-derived descriptors, mechanistic claims with evidence, and macroscopic outcomes, preserving process coupling and traceability; (3) a general querying module that supports natural-language exploration and visualization over the constructed graph for cross-paper analysis; (4) an evaluation on $\sim$800 peer-reviewed chemical engineering publications demonstrating the effectiveness of AgentCAT. 

---
# ZUNA: Flexible EEG Superresolution with Position-Aware Diffusion Autoencoders 

**Authors**: Christopher Warner, Jonas Mago, JR Huml, Mohamed Osman, Beren Millidge  

**Link**: [PDF](https://arxiv.org/pdf/2602.18478)  

**Abstract**: We present \texttt{ZUNA}, a 380M-parameter masked diffusion autoencoder trained to perform masked channel infilling and superresolution for arbitrary electrode numbers and positions in EEG signals. The \texttt{ZUNA} architecture tokenizes multichannel EEG into short temporal windows and injects spatiotemporal structure via a 4D rotary positional encoding over (x,y,z,t), enabling inference on arbitrary channel subsets and positions. We train ZUNA on an aggregated and harmonized corpus spanning 208 public datasets containing approximately 2 million channel-hours using a combined reconstruction and heavy channel-dropout objective. We show that \texttt{ZUNA} substantially improves over ubiquitous spherical-spline interpolation methods, with the gap widening at higher dropout rates. Crucially, compared to other deep learning methods in this space, \texttt{ZUNA}'s performance \emph{generalizes} across datasets and channel positions allowing it to be applied directly to novel datasets and problems. Despite its generative capabilities, \texttt{ZUNA} remains computationally practical for deployment. We release Apache-2.0 weights and an MNE-compatible preprocessing/inference stack to encourage reproducible comparisons and downstream use in EEG analysis pipelines. 

---
# BioLM-Score: Language-Prior Conditioned Probabilistic Geometric Potentials for Protein-Ligand Scoring 

**Authors**: Zhangfan Yang, Baoyun Chen, Dong Xu, Jia Wang, Ruibin Bai, Junkai Ji, Zexuan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.18476)  

**Abstract**: Protein-ligand scoring is a central component of structure-based drug design, underpinning molecular docking, virtual screening, and pose optimization. Conventional physics-based energy functions are often computationally expensive, limiting their utility in large-scale screening. In contrast, deep learning-based scoring models offer improved computational efficiency but frequently suffer from limited cross-target generalization and poor interpretability, which restrict their practical applicability. Here we present BioLM-Score, a simple yet generalizable protein-ligand scoring model that couples geometric modeling with representation learning. Specifically, it employs modality-specific and structure-aware encoders for proteins and ligands, each augmented with biomolecular language models to enrich structural and chemical representations. Subsequently, these representations are integrated through a mixture density network to predict multimodal interatomic distance distributions, from which statistically grounded likelihood-based scores are derived. Evaluations on the CASF-2016 benchmark demonstrate that BioLM-Score achieves significant improvements across docking, scoring, ranking, and screening tasks. Moreover, the proposed scoring function serves as an effective optimization objective for guiding docking protocols and conformational search. In summary, BioLM-Score provides a principled and practical alternative to existing scoring functions, combining efficiency, generalization, and interpretability for structure-based drug discovery. 

---
# Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series 

**Authors**: Guoqi Yu, Juncheng Wang, Chen Yang, Jing Qin, Angelica I. Aviles-Rivero, Shujun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18473)  

**Abstract**: Accurate analysis of medical time series (MedTS) data, such as electroencephalography (EEG) and electrocardiography (ECG), plays a pivotal role in healthcare applications, including the diagnosis of brain and heart diseases. MedTS data typically exhibit two critical patterns: temporal dependencies within individual channels and channel dependencies across multiple channels. While recent advances in deep learning have leveraged Transformer-based models to effectively capture temporal dependencies, they often struggle with modeling channel dependencies. This limitation stems from a structural mismatch: MedTS signals are inherently centralized, whereas the Transformer's attention mechanism is decentralized, making it less effective at capturing global synchronization and unified waveform patterns. To address this mismatch, we propose CoTAR (Core Token Aggregation-Redistribution), a centralized MLP-based module designed to replace decentralized attention. Instead of allowing all tokens to interact directly, as in standard attention, CoTAR introduces a global core token that serves as a proxy to facilitate inter-token interactions, thereby enforcing a centralized aggregation and redistribution strategy. This design not only better aligns with the centralized nature of MedTS signals but also reduces computational complexity from quadratic to linear. Experiments on five benchmarks validate the superiority of our method in both effectiveness and efficiency, achieving up to a 12.13% improvement on the APAVA dataset, while using only 33% of the memory and 20% of the inference time compared to the previous state of the art. Code and all training scripts are available at this https URL. 

---
# Physiologically Informed Deep Learning: A Multi-Scale Framework for Next-Generation PBPK Modeling 

**Authors**: Shunqi Liu, Han Qiu, Tong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18472)  

**Abstract**: Physiologically Based Pharmacokinetic (PBPK) modeling is a cornerstone of model-informed drug development (MIDD), providing a mechanistic framework to predict drug absorption, distribution, metabolism, and excretion (ADME). Despite its utility, adoption is hindered by high computational costs for large-scale simulations, difficulty in parameter identification for complex biological systems, and uncertainty in interspecies extrapolation. In this work, we propose a unified Scientific Machine Learning (SciML) framework that bridges mechanistic rigor and data-driven flexibility. We introduce three contributions: (1) Foundation PBPK Transformers, which treat pharmacokinetic forecasting as a sequence modeling task; (2) Physiologically Constrained Diffusion Models (PCDM), a generative approach that uses a physics-informed loss to synthesize biologically compliant virtual patient populations; and (3) Neural Allometry, a hybrid architecture combining Graph Neural Networks (GNNs) with Neural ODEs to learn continuous cross-species scaling laws. Experiments on synthetic datasets show that the framework reduces physiological violation rates from 2.00% to 0.50% under constraints while offering a path to faster simulation. 

---
# Charting the Future of AI-supported Science Education: A Human-Centered Vision 

**Authors**: Xiaoming Zhai, Kent Crippen  

**Link**: [PDF](https://arxiv.org/pdf/2602.18471)  

**Abstract**: This concluding chapter explores how artificial intelligence (AI) is reshaping the purposes, practices, and outcomes of science education, and proposes a human-centered framework for its responsible integration. Drawing on insights from international collaborations and the Advancing AI in Science Education (AASE) committee, the chapter synthesizes developments across five dimensions: educational goals, instructional procedures, learning materials, assessment, and outcomes. We argue that AI offers transformative potential to enrich inquiry, personalize learning, and support teacher practice, but only when guided by Responsible and Ethical Principles (REP). The REP framework, emphasizing fairness, transparency, privacy, accountability, and respect for human values, anchors our vision for AI-supported science education. Key discussions include the redefinition of scientific literacy to encompass AI literacy, the evolving roles of teachers and learners in AI-supported classrooms, and the design of adaptive learning materials and assessments that preserve authenticity and integrity. We highlight both opportunities and risks, stressing the need for critical engagement with AI to avoid reinforcing inequities or undermining human agency. Ultimately, this chapter advances a vision in which science education prepares learners to act as ethical investigators and responsible citizens, ensuring that AI innovation aligns with human dignity, equity, and the broader goals of scientific literacy. 

---
# Transforming Science Learning Materials in the Era of Artificial Intelligence 

**Authors**: Xiaoming Zhai, Kent Crippen  

**Link**: [PDF](https://arxiv.org/pdf/2602.18470)  

**Abstract**: The integration of artificial intelligence (AI) into science education is transforming the design and function of learning materials, offering new affordances for personalization, authenticity, and accessibility. This chapter examines how AI technologies are transforming science learning materials across six interrelated domains: 1) integrating AI into scientific practice, 2) enabling adaptive and personalized instruction, 3) facilitating interactive simulations, 4) generating multimodal content, 5) enhancing accessibility for diverse learners, and 6) promoting co-creation through AI-supported content development. These advancements enable learning materials to more accurately reflect contemporary scientific practice, catering to the diverse needs of students. For instance, AI support can enable students to engage in dynamic simulations, interact with real-time data, and explore science concepts through multimodal representations. Educators are increasingly collaborating with generative AI tools to develop timely and culturally responsive instructional resources. However, these innovations also raise critical ethical and pedagogical concerns, including issues of algorithmic bias, data privacy, transparency, and the need for human oversight. To ensure equitable and meaningful science learning, we emphasize the importance of designing AI-supported materials with careful attention to scientific integrity, inclusivity, and student agency. This chapter advocates for a responsible, ethical, and reflective approach to leveraging AI in science education, framing it as a catalyst for innovation while upholding core educational values. 

---
# The Landscape of AI in Science Education: What is Changing and How to Respond 

**Authors**: Xiaoming Zhai, Kent Crippen  

**Link**: [PDF](https://arxiv.org/pdf/2602.18469)  

**Abstract**: This introductory chapter explores the transformative role of artificial intelligence (AI) in reshaping the landscape of science education. Positioned at the intersection of tradition and innovation, AI is altering educational goals, procedures, learning materials, assessment practices, and desired outcomes. We highlight how AI-supported tools, such as intelligent tutoring systems, adaptive learning platforms, automated feedback, and generative content creation--enhance personalization, efficiency, and equity while fostering competencies essential for an AI-driven society, including critical thinking, creativity, and interdisciplinary collaboration. At the same time, this chapter examines the ethical, social, and pedagogical challenges that arise, particularly issues of fairness, transparency, accountability, privacy, and human oversight. To address these tensions, we argue that a Responsible and Ethical Principles (REP) framework is needed to offer guidance for aligning AI integration with values of fairness, scientific integrity, and democratic participation. Through this lens, we synthesize the changes brought to each of the five transformative aspects and the approaches introduced to meet the changes according to the REP framework. We argue that AI should be viewed not as a replacement for human teachers and learners but as a partner that supports inquiry, enriches assessment, and expands access to authentic scientific practices. Aside from what is changing, we conclude by exploring the roles that remain uniquely human, engaging as moral and relational anchors in classrooms, bringing interpretive and ethical judgement, fostering creativity, imagination, and curiosity, and co-constructing meaning through dialogue and community, and assert that these qualities must remain central if AI is to advance equity, integrity, and human flourishing in science education. 

---
# Can Multimodal LLMs See Science Instruction? Benchmarking Pedagogical Reasoning in K-12 Classroom Videos 

**Authors**: Yixuan Shen, Peng He, Honglu Liu, Yuyang Ji, Tingting Li, Tianlong Chen, Kaidi Xu, Feng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.18466)  

**Abstract**: K-12 science classrooms are rich sites of inquiry where students coordinate phenomena, evidence, and explanatory models through discourse; yet, the multimodal complexity of these interactions has made automated analysis elusive. Existing benchmarks for classroom discourse focus primarily on mathematics and rely solely on transcripts, overlooking the visual artifacts and model-based reasoning emphasized by the Next Generation Science Standards (NGSS). We address this gap with SciIBI, the first video benchmark for analyzing science classroom discourse, featuring 113 NGSS-aligned clips annotated with Core Instructional Practices (CIP) and sophistication levels. By evaluating eight state-of-the-art LLMs and Multimodal LLMs, we reveal fundamental limitations: current models struggle to distinguish pedagogically similar practices, suggesting that CIP coding requires instructional reasoning beyond surface pattern matching. Furthermore, adding video input yields inconsistent gains across architectures. Crucially, our evidence-based evaluation reveals that models often succeed through surface shortcuts rather than genuine pedagogical understanding. These findings establish science classroom discourse as a challenging frontier for multimodal AI and point toward human-AI collaboration, where models retrieve evidence to accelerate expert review rather than replace it. 

---
# How Well Can LLM Agents Simulate End-User Security and Privacy Attitudes and Behaviors? 

**Authors**: Yuxuan Li, Leyang Li, Hao-Ping, Sauvik Das  

**Link**: [PDF](https://arxiv.org/pdf/2602.18464)  

**Abstract**: A growing body of research assumes that large language model (LLM) agents can serve as proxies for how people form attitudes toward and behave in response to security and privacy (S&P) threats. If correct, these simulations could offer a scalable way to forecast S&P risks in products prior to deployment. We interrogate this assumption using SP-ABCBench, a new benchmark of 30 tests derived from validated S&P human-subject studies, which measures alignment between simulations and human-subjects studies on a 0-100 ascending scale, where higher scores indicate better alignment across three dimensions: Attitude, Behavior, and Coherence. Evaluating twelve LLMs, four persona construction strategies, and two prompting methods, we found that there remains substantial room for improvement: all models score between 50 and 64 on average. Newer, bigger, and smarter models do not reliably do better and sometimes do worse. Some simulation configurations, however, do yield high alignment: e.g., with scores above 95 for some behavior tests when agents are prompted to apply bounded rationality and weigh privacy costs against perceived benefits. We release SP-ABCBench to enable reproducible evaluation as methods improve. 

---
# Assessing the Reliability of Persona-Conditioned LLMs as Synthetic Survey Respondents 

**Authors**: Erika Elizabeth Taday Morocho, Lorenzo Cima, Tiziano Fagni, Marco Avvenuti, Stefano Cresci  

**Link**: [PDF](https://arxiv.org/pdf/2602.18462)  

**Abstract**: Using persona-conditioned LLMs as synthetic survey respondents has become a common practice in computational social science and agent-based simulations. Yet, it remains unclear whether multi-attribute persona prompting improves LLM reliability or instead introduces distortions. Here we contribute to this assessment by leveraging a large dataset of U.S. microdata from the World Values Survey. Concretely, we evaluate two open-weight chat models and a random-guesser baseline across more than 70K respondent-item instances. We find that persona prompting does not yield a clear aggregate improvement in survey alignment and, in many cases, significantly degrades performance. Persona effects are highly heterogeneous as most items exhibit minimal change, while a small subset of questions and underrepresented subgroups experience disproportionate distortions. Our findings highlight a key adverse impact of current persona-based simulation practices: demographic conditioning can redistribute error in ways that undermine subgroup fidelity and risk misleading downstream analyses. 

---
# The Doctor Will (Still) See You Now: On the Structural Limits of Agentic AI in Healthcare 

**Authors**: Gabriela Aránguiz Dias, Kiana Jafari, Allie Griffith, Carolina Aránguiz Dias, Grace Ra Kim, Lana Saadeddin, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2602.18460)  

**Abstract**: Across healthcare, agentic artificial intelligence (AI) systems are increasingly promoted as capable of autonomous action, yet in practice they currently operate under near-total human oversight due to safety, regulatory, and liability constraints that make autonomous clinical reasoning infeasible in high-stakes environments. While market enthusiasm suggests a revolution in healthcare agents, the conceptual assumptions and accountability structures shaping these systems remain underexamined. We present a qualitative study based on interviews with 20 stakeholders, including developers, implementers, and end users. Our analysis identifies three mutually reinforcing tensions: conceptual fragmentation regarding the definition of `agentic'; an autonomy contradiction where commercial promises exceed operational reality; and an evaluation blind spot that prioritizes technical benchmarks over sociotechnical safety. We argue that agentic {AI} functions as a site of contested meaning-making where technical aspirations, commercial incentives, and clinical constraints intersect, carrying material consequences for patient safety and the distribution of blame. 

---
# From Bias Mitigation to Bias Negotiation: Governing Identity and Sociocultural Reasoning in Generative AI 

**Authors**: Zackary Okun Dunivin, Bingyi Han, John Bollenbocher  

**Link**: [PDF](https://arxiv.org/pdf/2602.18459)  

**Abstract**: LLMs act in the social world by drawing upon shared cultural patterns to make social situations understandable and actionable. Because identity is often part of the inferential substrate of competent judgment, ethical alignment requires regulating when and how systems invoke identity. Yet the dominant governance regime for identity-related harm remains bias mitigation, which treats identity primarily as a source of measurable disparities or harmful associations to be detected and suppressed. This leaves underspecified a positive, context-sensitive role for identity in interpretation. We call this governance problem bias negotiation: the normative regulation of identity-conditioned judgments of sociocultural relevance, inference, and justification. Empirically, we probe the feasibility of bias negotiation through semi-structured interviews with multiple publicly deployed chatbots. We identify recurring repertoires for negotiating identity including probabilistic framing of group tendencies and harm-value balancing. We also observe failure modes in which models avoid hard tradeoffs or apply principles inconsistently. Bias negotiation matters for justice because a positive role for sociocultural reasoning is required to recognize and potentially remediate structural inequities. But it is equally implicated in core model functionality as sociocultural competence is needed for systems that operate across heterogeneous cultural contexts. Because bias negotiation is a procedural capability expressed through deliberation and interaction, it cannot be validated by static benchmarks alone. To support targeted training, we introduce a broad but explicit framework that decomposes bias negotiation into an action space of negotiation moves (what to observe and score) and a complementary set of case features (over which the model negotiates), enabling systematic test-suite design and evaluation. 

---
# The Story is Not the Science: Execution-Grounded Evaluation of Mechanistic Interpretability Research 

**Authors**: Xiaoyan Bai, Alexander Baumgartner, Haojia Sun, Ari Holtzman, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2602.18458)  

**Abstract**: Reproducibility crises across sciences highlight the limitations of the paper-centric review system in assessing the rigor and reproducibility of research. AI agents that autonomously design and generate large volumes of research outputs exacerbate these challenges. In this work, we address the growing challenges of scalability and rigor by flipping the dynamic and developing AI agents as research evaluators. We propose the first execution-grounded evaluation framework that verifies research beyond narrative review by examining code and data alongside the paper. We use mechanistic interpretability research as a testbed, build standardized research output, and develop MechEvalAgent, an automated evaluation framework that assesses the coherence of the experimental process, the reproducibility of results, and the generalizability of findings. We show that our framework achieves above 80% agreement with human judges, identifies substantial methodological problems, and surfaces 51 additional issues that human reviewers miss. Our work demonstrates the potential of AI agents to transform research evaluation and pave the way for rigorous scientific practices. 

---
# Beyond single-channel agentic benchmarking 

**Authors**: Nelu D. Radpour  

**Link**: [PDF](https://arxiv.org/pdf/2602.18456)  

**Abstract**: Contemporary benchmarks for agentic artificial intelligence (AI) frequently evaluate safety through isolated task-level accuracy thresholds, implicitly treating autonomous systems as single points of failure. This single-channel paradigm diverges from established principles in safety-critical engineering, where risk mitigation is achieved through redundancy, diversity of error modes, and joint system reliability. This paper argues that evaluating AI agents in isolation systematically mischaracterizes their operational safety when deployed within human-in-the-loop environments. Using a recent laboratory safety benchmark as a case study demonstrates that even imperfect AI systems can nonetheless provide substantial safety utility by functioning as redundant audit layers against well-documented sources of human failure, including vigilance decrement, inattentional blindness, and normalization of deviance. This perspective reframes agentic safety evaluation around the reliability of the human-AI dyad rather than absolute agent accuracy, with a particular emphasis on uncorrelated error modes as the primary determinant of risk reduction. Such a shift aligns AI benchmarking with established practices in other safety-critical domains and offers a path toward more ecologically valid safety assessments. 

---
# Impact of AI Search Summaries on Website Traffic: Evidence from Google AI Overviews and Wikipedia 

**Authors**: Mehrzad Khosravi, Hema Yoganarasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2602.18455)  

**Abstract**: Search engines increasingly display LLM-generated answers shown above organic links, shifting search from link lists to answer-first summaries. Publishers contend these summaries substitute for source pages and cannibalize traffic, while platforms argue they are complementary by directing users through included links. We estimate the causal impact of Google's AI Overview (AIO) on Wikipedia traffic by leveraging the feature's staggered geographic rollout and Wikipedia's multilingual structure. Using a difference-in-differences design, we compare English Wikipedia articles exposed to AIO to the same underlying articles in language editions (Hindi, Indonesian, Japanese, and Portuguese) that were not exposed to AIO during the observation period. Across 161,382 matched article-language pairs, AIO exposure reduces daily traffic to English articles by approximately 15%. Effects are heterogeneous: relative declines are largest for Culture articles and substantially smaller for STEM, consistent with stronger substitution when short synthesized answers satisfy informational intent. These findings provide early causal evidence that generative-answer features in search engines can materially reallocate attention away from informational publishers, with implications for content monetization, search platform design, and policy. 

---
# LLM-Assisted Replication for Quantitative Social Science 

**Authors**: So Kubota, Hiromu Yakura, Samuel Coavoux, Sho Yamada, Yuki Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2602.18453)  

**Abstract**: The replication crisis, the failure of scientific claims to be validated by further research, is one of the most pressing issues for empirical research. This is partly an incentive problem: replication is costly and less well rewarded than original research. Large language models (LLMs) have accelerated scientific production by streamlining writing, coding, and reviewing, yet this acceleration risks outpacing verification. To address this, we present an LLM-based system that replicates statistical analyses from social science papers and flags potential problems. Quantitative social science is particularly well-suited to automation because it relies on standard statistical models, shared public datasets, and uniform reporting formats such as regression tables and summary statistics. We present a prototype that iterates LLM-based text interpretation, code generation, execution, and discrepancy analysis, demonstrating its capabilities by reproducing key results from a seminal sociology paper. We also outline application scenarios including pre-submission checks, peer-review support, and meta-scientific audits, positioning AI verification as assistive infrastructure that strengthens research integrity. 

---
# Developing a Multi-Agent System to Generate Next Generation Science Assessments with Evidence-Centered Design 

**Authors**: Yaxuan Yang, Jongchan Park, Yifan Zhou, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2602.18451)  

**Abstract**: Contemporary science education reforms such as the Next Generation Science Standards (NGSS) demand assessments to understand students' ability to use science knowledge to solve problems and design solutions. To elicit such higher-order ability, educators need performance-based assessments, which are challenging to develop. One solution that has been broadly adopted is Evidence-Centered Design (ECD), which emphasizes interconnected models of the learner, evidence, and tasks. Although ECD provides a framework to safeguard assessment validity, its implementation requires diverse expertise (e.g., content and assessment), which is both costly and labor-intensive. To address this challenge, this study proposed integrating the ECD framework into Multi-Agent Systems (MAS) to generate NGSS-aligned assessment items automatically. This integrated MAS system ensembles multiple large language models with varying expertise, enabling the automation of complex, multi-stage item generation workflows traditionally performed by human experts. We examined the quality of AI-generated NGSS-aligned items and compared them with human-developed items across multiple dimensions of assessment design. Results showed that AI-generated items have overall comparable quality to human-developed items in terms of alignment with NGSS three-dimensional standards and cognitive demands. Divergent patterns also emerged: AI-generated items demonstrated a distinct strength in inclusivity, while also exhibiting limitations in clarity, conciseness, and multimodal design. AI- and human-developed items both showed weaknesses in evidence collectability and student interest alignment. These findings suggest that integrating ECD into MAS can support scalable and standards-aligned assessment design, while human expertise remains essential. 

---
# Prompt Optimization Via Diffusion Language Models 

**Authors**: Shiyu Wang, Haolin Chen, Liangwei Yang, Jielin Qiu, Rithesh Murthy, Ming Zhu, Zixiang Chen, Silvio Savarese, Caiming Xiong, Shelby Heinecke, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18449)  

**Abstract**: We propose a diffusion-based framework for prompt optimization that leverages Diffusion Language Models (DLMs) to iteratively refine system prompts through masked denoising. By conditioning on interaction traces, including user queries, model responses, and optional feedback, our method enables flexible, span-level prompt updates without requiring gradient access or modifying the downstream language model. Across diverse benchmarks (e.g., $\tau$-bench, SST-2, SST-5), DLM-optimized prompts consistently improve the performance of a frozen target LLM (e.g., GPT-4o-mini). We further show that moderate diffusion step counts provide the best balance between refinement quality and stability. These results highlight diffusion-based prompt optimization as a general, model-agnostic, and scalable approach for enhancing LLM performance through iterative prompt refinement. 

---
# ConfSpec: Efficient Step-Level Speculative Reasoning via Confidence-Gated Verification 

**Authors**: Siran Liu, Cyril Y. He  

**Link**: [PDF](https://arxiv.org/pdf/2602.18447)  

**Abstract**: Chain-of-Thought reasoning significantly improves the performance of large language models on complex tasks, but incurs high inference latency due to long generation traces. Step-level speculative reasoning aims to mitigate this cost, yet existing approaches face a long-standing trade-off among accuracy, inference speed, and resource efficiency. We propose ConfSpec, a confidence-gated cascaded verification framework that resolves this trade-off. Our key insight is an asymmetry between generation and verification: while generating a correct reasoning step requires substantial model capacity, step-level verification is a constrained discriminative task for which small draft models are well-calibrated within their competence range, enabling high-confidence draft decisions to be accepted directly while selectively escalating uncertain cases to the large target model. Evaluation across diverse workloads shows that ConfSpec achieves up to 2.24$\times$ end-to-end speedups while matching target-model accuracy. Our method requires no external judge models and is orthogonal to token-level speculative decoding, enabling further multiplicative acceleration. 

---
# ReportLogic: Evaluating Logical Quality in Deep Research Reports 

**Authors**: Jujia Zhao, Zhaoxin Huan, Zihan Wang, Xiaolu Zhang, Jun Zhou, Suzan Verberne, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2602.18446)  

**Abstract**: Users increasingly rely on Large Language Models (LLMs) for Deep Research, using them to synthesize diverse sources into structured reports that support understanding and action. In this context, the practical reliability of such reports hinges on logical quality: whether the report's claims and arguments are explicitly supported and can be trusted as a basis for downstream use, rather than merely appearing fluent or informative. However, current evaluation frameworks largely overlook this requirement. To bridge this gap, we introduce ReportLogic, a benchmark that quantifies report-level logical quality through a reader-centric lens of auditability. Specifically, ReportLogic adopts a hierarchical taxonomy that evaluates whether readers can (1) trace an on-topic report structure with a unified analytical arc (Macro-Logic), (2) understand the progression with necessary context (Expositional-Logic), and (3) verify conclusions via explicit claim--support (Structural-Logic). Based on this taxonomy, we construct a human-annotated rubric-guided dataset and train an open-source LogicJudge for scalable evaluation. We further evaluate judge robustness via adversarial attacks, showing that off-the-shelf LLM judges are frequently influenced by superficial cues (e.g., verbosity), and reasoning modes can mask broken support relations. Overall, our results provide actionable guidance for building more robust logic evaluators and improving the logical reliability of LLM-generated reports. 

---
# From "Help" to Helpful: A Hierarchical Assessment of LLMs in Mental e-Health Applications 

**Authors**: Philipp Steigerwald, Jens Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2602.18443)  

**Abstract**: Psychosocial online counselling frequently encounters generic subject lines that impede efficient case prioritisation. This study evaluates eleven large language models generating six-word subject lines for German counselling emails through hierarchical assessment - first categorising outputs, then ranking within categories to enable manageable evaluation. Nine assessors (counselling professionals and AI systems) enable analysis via Krippendorff's $\alpha$, Spearman's $\rho$, Pearson's $r$ and Kendall's $\tau$. Results reveal performance trade-offs between proprietary services and privacy-preserving open-source alternatives, with German fine-tuning consistently improving performance. The study addresses critical ethical considerations for mental health AI deployment including privacy, bias and accountability. 

---
# FineRef: Fine-Grained Error Reflection and Correction for Long-Form Generation with Citations 

**Authors**: Yixing Peng, Licheng Zhang, Shancheng Fang, Yi Liu, Peijian Gu, Quan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18437)  

**Abstract**: Generating with citations is crucial for trustworthy Large Language Models (LLMs), yet even advanced LLMs often produce mismatched or irrelevant citations. Existing methods over-optimize citation fidelity while overlooking relevance to the user query, which degrades answer quality and robustness in real-world settings with noisy or irrelevant retrieved content. Moreover, the prevailing single-pass paradigm struggles to deliver optimal answers in long-form generation that requiring multiple citations. To address these limitations, we propose FineRef, a framework based on Fine-grained error Reflection, which explicitly teaches the model to self-identify and correct two key citation errors, mismatch and irrelevance, on a per-citation basis. FineRef follows a two-stage training strategy. The first stage instills an "attempt-reflect-correct" behavioral pattern via supervised fine-tuning, using fine-grained and controllable reflection data constructed by specialized lightweight models. An online self-reflective bootstrapping strategy is designed to improve generalization by iteratively enriching training data with verified, self-improving examples. To further enhance the self-reflection and correction capability, the second stage applies process-level reinforcement learning with a multi-dimensional reward scheme that promotes reflection accuracy, answer quality, and correction gain. Experiments on the ALCE benchmark demonstrate that FineRef significantly improves both citation performance and answer accuracy. Our 7B model outperforms GPT-4 by up to 18% in Citation F1 and 4% in EM Recall, while also surpassing the state-of-the-art model across key evaluation metrics. FineRef also exhibits strong generalization and robustness in domain transfer settings and noisy retrieval scenarios. 

---
# Context-Aware Mapping of 2D Drawing Annotations to 3D CAD Features Using LLM-Assisted Reasoning for Manufacturing Automation 

**Authors**: Muhammad Tayyab Khana, Lequn Chen, Wenhe Feng, Seung Ki Moon  

**Link**: [PDF](https://arxiv.org/pdf/2602.18296)  

**Abstract**: Manufacturing automation in process planning, inspection planning, and digital-thread integration depends on a unified specification that binds the geometric features of a 3D CAD model to the geometric dimensioning and tolerancing (GD&T) callouts, datum definitions, and surface requirements carried by the corresponding 2D engineering drawing. Although Model-Based Definition (MBD) allows such specifications to be embedded directly in 3D models, 2D drawings remain the primary carrier of manufacturing intent in automotive, aerospace, shipbuilding, and heavy-machinery industries. Correctly linking drawing annotations to the corresponding 3D features is difficult because of contextual ambiguity, repeated feature patterns, and the need for transparent and traceable decisions. This paper presents a deterministic-first, context-aware framework that maps 2D drawing entities to 3D CAD features to produce a unified manufacturing specification. Drawing callouts are first semantically enriched and then scored against candidate features using an interpretable metric that combines type compatibility, tolerance-aware dimensional agreement, and conservative context consistency, along with engineering-domain heuristics. When deterministic scoring cannot resolve an ambiguity, the system escalates to multimodal and constrained large-language-model reasoning, followed by a single human-in-the-loop (HITL) review step. Experiments on 20 real CAD-drawing pairs achieve a mean precision of 83.67%, recall of 90.46%, and F1 score of 86.29%. An ablation study shows that each pipeline component contributes to overall accuracy, with the full system outperforming all reduced variants. By prioritizing deterministic rules, clear decision tracking, and retaining unresolved cases for human review, the framework provides a practical foundation for downstream manufacturing automation in real-world industrial environments. 

---
# NI-Tex: Non-isometric Image-based Garment Texture Generation 

**Authors**: Hui Shan, Ming Li, Haitao Yang, Kai Zheng, Sizhe Zheng, Yanwei Fu, Xiangru Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18765)  

**Abstract**: Existing industrial 3D garment meshes already cover most real-world clothing geometries, yet their texture diversity remains limited. To acquire more realistic textures, generative methods are often used to extract Physically-based Rendering (PBR) textures and materials from large collections of wild images and project them back onto garment meshes. However, most image-conditioned texture generation approaches require strict topological consistency between the input image and the input 3D mesh, or rely on accurate mesh deformation to match to the image poses, which significantly constrains the texture generation quality and flexibility. To address the challenging problem of non-isometric image-based garment texture generation, we construct 3D Garment Videos, a physically simulated, garment-centric dataset that provides consistent geometry and material supervision across diverse deformations, enabling robust cross-pose texture learning. We further employ Nano Banana for high-quality non-isometric image editing, achieving reliable cross-topology texture generation between non-isometric image-geometry pairs. Finally, we propose an iterative baking method via uncertainty-guided view selection and reweighting that fuses multi-view predictions into seamless, production-ready PBR textures. Through extensive experiments, we demonstrate that our feedforward dual-branch architecture generates versatile and spatially aligned PBR materials suitable for industry-level 3D garment design. 

---
# DEFNet: Multitasks-based Deep Evidential Fusion Network for Blind Image Quality Assessment 

**Authors**: Yiwei Lou, Yuanpeng He, Rongchao Zhang, Yongzhi Cao, Hanpin Wang, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.19418)  

**Abstract**: Blind image quality assessment (BIQA) methods often incorporate auxiliary tasks to improve performance. However, existing approaches face limitations due to insufficient integration and a lack of flexible uncertainty estimation, leading to suboptimal performance. To address these challenges, we propose a multitasks-based Deep Evidential Fusion Network (DEFNet) for BIQA, which performs multitask optimization with the assistance of scene and distortion type classification tasks. To achieve a more robust and reliable representation, we design a novel trustworthy information fusion strategy. It first combines diverse features and patterns across sub-regions to enhance information richness, and then performs local-global information fusion by balancing fine-grained details with coarse-grained context. Moreover, DEFNet exploits advanced uncertainty estimation technique inspired by evidential learning with the help of normal-inverse gamma distribution mixture. Extensive experiments on both synthetic and authentic distortion datasets demonstrate the effectiveness and robustness of the proposed framework. Additional evaluation and analysis are carried out to highlight its strong generalization capability and adaptability to previously unseen scenarios. 

---
