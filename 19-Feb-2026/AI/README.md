# Towards a Science of AI Agent Reliability 

**Authors**: Stephan Rabanser, Sayash Kapoor, Peter Kirgis, Kangheng Liu, Saiteja Utpala, Arvind Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2602.16666)  

**Abstract**: AI agents are increasingly deployed to execute important tasks. While rising accuracy scores on standard benchmarks suggest rapid progress, many agents still continue to fail in practice. This discrepancy highlights a fundamental limitation of current evaluations: compressing agent behavior into a single success metric obscures critical operational flaws. Notably, it ignores whether agents behave consistently across runs, withstand perturbations, fail predictably, or have bounded error severity. Grounded in safety-critical engineering, we provide a holistic performance profile by proposing twelve concrete metrics that decompose agent reliability along four key dimensions: consistency, robustness, predictability, and safety. Evaluating 14 agentic models across two complementary benchmarks, we find that recent capability gains have only yielded small improvements in reliability. By exposing these persistent limitations, our metrics complement traditional evaluations while offering tools for reasoning about how agents perform, degrade, and fail. 

---
# Agent Skill Framework: Perspectives on the Potential of Small Language Models in Industrial Environments 

**Authors**: Yangjie Xu, Lujun Li, Lama Sleem, Niccolo Gentile, Yewei Song, Yiqun Wang, Siming Ji, Wenbo Wu, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2602.16653)  

**Abstract**: Agent Skill framework, now widely and officially supported by major players such as GitHub Copilot, LangChain, and OpenAI, performs especially well with proprietary models by improving context engineering, reducing hallucinations, and boosting task accuracy. Based on these observations, an investigation is conducted to determine whether the Agent Skill paradigm provides similar benefits to small language models (SLMs). This question matters in industrial scenarios where continuous reliance on public APIs is infeasible due to data-security and budget constraints requirements, and where SLMs often show limited generalization in highly customized scenarios. This work introduces a formal mathematical definition of the Agent Skill process, followed by a systematic evaluation of language models of varying sizes across multiple use cases. The evaluation encompasses two open-source tasks and a real-world insurance claims data set. The results show that tiny models struggle with reliable skill selection, while moderately sized SLMs (approximately 12B - 30B) parameters) benefit substantially from the Agent Skill approach. Moreover, code-specialized variants at around 80B parameters achieve performance comparable to closed-source baselines while improving GPU efficiency. Collectively, these findings provide a comprehensive and nuanced characterization of the capabilities and constraints of the framework, while providing actionable insights for the effective deployment of Agent Skills in SLM-centered environments. 

---
# Creating a digital poet 

**Authors**: Vered Tohar, Tsahi Hayat, Amir Leshem  

**Link**: [PDF](https://arxiv.org/pdf/2602.16578)  

**Abstract**: Can a machine write good poetry? Any positive answer raises fundamental questions about the nature and value of art. We report a seven-month poetry workshop in which a large language model was shaped into a digital poet through iterative in-context expert feedback, without retraining. Across sessions, the model developed a distinctive style and a coherent corpus, supported by quantitative and qualitative analyses, and it produced a pen name and author image. In a blinded authorship test with 50 humanities students and graduates (three AI poems and three poems by well-known poets each), judgments were at chance: human poems were labeled human 54% of the time and AI poems 52%, with 95% confidence intervals including 50%. After the workshop, a commercial publisher released a poetry collection authored by the model. These results show that workshop-style prompting can support long-horizon creative shaping and renew debates on creativity and authorship. 

---
# Framework of Thoughts: A Foundation Framework for Dynamic and Optimized Reasoning based on Chains, Trees, and Graphs 

**Authors**: Felix Fricke, Simon Malberg, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2602.16512)  

**Abstract**: Prompting schemes such as Chain of Thought, Tree of Thoughts, and Graph of Thoughts can significantly enhance the reasoning capabilities of large language models. However, most existing schemes require users to define static, problem-specific reasoning structures that lack adaptability to dynamic or unseen problem types. Additionally, these schemes are often under-optimized in terms of hyperparameters, prompts, runtime, and prompting cost. To address these limitations, we introduce Framework of Thoughts (FoT)--a general-purpose foundation framework for building and optimizing dynamic reasoning schemes. FoT comes with built-in features for hyperparameter tuning, prompt optimization, parallel execution, and intelligent caching, unlocking the latent performance potential of reasoning schemes. We demonstrate FoT's capabilities by implementing three popular schemes--Tree of Thoughts, Graph of Thoughts, and ProbTree--within FoT. We empirically show that FoT enables significantly faster execution, reduces costs, and achieves better task scores through optimization. We release our codebase to facilitate the development of future dynamic and efficient reasoning schemes. 

---
# Leveraging Large Language Models for Causal Discovery: a Constraint-based, Argumentation-driven Approach 

**Authors**: Zihao Li, Fabrizio Russo  

**Link**: [PDF](https://arxiv.org/pdf/2602.16481)  

**Abstract**: Causal discovery seeks to uncover causal relations from data, typically represented as causal graphs, and is essential for predicting the effects of interventions. While expert knowledge is required to construct principled causal graphs, many statistical methods have been proposed to leverage observational data with varying formal guarantees. Causal Assumption-based Argumentation (ABA) is a framework that uses symbolic reasoning to ensure correspondence between input constraints and output graphs, while offering a principled way to combine data and expertise. We explore the use of large language models (LLMs) as imperfect experts for Causal ABA, eliciting semantic structural priors from variable names and descriptions and integrating them with conditional-independence evidence. Experiments on standard benchmarks and semantically grounded synthetic graphs demonstrate state-of-the-art performance, and we additionally introduce an evaluation protocol to mitigate memorisation bias when assessing LLMs for causal discovery. 

---
# Causally-Guided Automated Feature Engineering with Multi-Agent Reinforcement Learning 

**Authors**: Arun Vignesh Malarkkan, Wangyang Ying, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2602.16435)  

**Abstract**: Automated feature engineering (AFE) enables AI systems to autonomously construct high-utility representations from raw tabular data. However, existing AFE methods rely on statistical heuristics, yielding brittle features that fail under distribution shift. We introduce CAFE, a framework that reformulates AFE as a causally-guided sequential decision process, bridging causal discovery with reinforcement learning-driven feature construction. Phase I learns a sparse directed acyclic graph over features and the target to obtain soft causal priors, grouping features as direct, indirect, or other based on their causal influence with respect to the target. Phase II uses a cascading multi-agent deep Q-learning architecture to select causal groups and transformation operators, with hierarchical reward shaping and causal group-level exploration strategies that favor causally plausible transformations while controlling feature complexity. Across 15 public benchmarks (classification with macro-F1; regression with inverse relative absolute error), CAFE achieves up to 7% improvement over strong AFE baselines, reduces episodes-to-convergence, and delivers competitive time-to-target. Under controlled covariate shifts, CAFE reduces performance drop by ~4x relative to a non-causal multi-agent baseline, and produces more compact feature sets with more stable post-hoc attributions. These findings underscore that causal structure, used as a soft inductive prior rather than a rigid constraint, can substantially improve the robustness and efficiency of automated feature engineering. 

---
# Verifiable Semantics for Agent-to-Agent Communication 

**Authors**: Philipp Schoenegger, Matt Carlson, Chris Schneider, Chris Daly  

**Link**: [PDF](https://arxiv.org/pdf/2602.16424)  

**Abstract**: Multiagent AI systems require consistent communication, but we lack methods to verify that agents share the same understanding of the terms used. Natural language is interpretable but vulnerable to semantic drift, while learned protocols are efficient but opaque. We propose a certification protocol based on the stimulus-meaning model, where agents are tested on shared observable events and terms are certified if empirical disagreement falls below a statistical threshold. In this protocol, agents restricting their reasoning to certified terms ("core-guarded reasoning") achieve provably bounded disagreement. We also outline mechanisms for detecting drift (recertification) and recovering shared vocabulary (renegotiation). In simulations with varying degrees of semantic divergence, core-guarding reduces disagreement by 72-96%. In a validation with fine-tuned language models, disagreement is reduced by 51%. Our framework provides a first step towards verifiable agent-to-agent communication. 

---
# Multi-agent cooperation through in-context co-player inference 

**Authors**: Marissa A. Weis, Maciej Wołczyk, Rajai Nasser, Rif A. Saurous, Blaise Agüera y Arcas, João Sacramento, Alexander Meulemans  

**Link**: [PDF](https://arxiv.org/pdf/2602.16301)  

**Abstract**: Achieving cooperation among self-interested agents remains a fundamental challenge in multi-agent reinforcement learning. Recent work showed that mutual cooperation can be induced between "learning-aware" agents that account for and shape the learning dynamics of their co-players. However, existing approaches typically rely on hardcoded, often inconsistent, assumptions about co-player learning rules or enforce a strict separation between "naive learners" updating on fast timescales and "meta-learners" observing these updates. Here, we demonstrate that the in-context learning capabilities of sequence models allow for co-player learning awareness without requiring hardcoded assumptions or explicit timescale separation. We show that training sequence model agents against a diverse distribution of co-players naturally induces in-context best-response strategies, effectively functioning as learning algorithms on the fast intra-episode timescale. We find that the cooperative mechanism identified in prior work-where vulnerability to extortion drives mutual shaping-emerges naturally in this setting: in-context adaptation renders agents vulnerable to extortion, and the resulting mutual pressure to shape the opponent's in-context learning dynamics resolves into the learning of cooperative behavior. Our results suggest that standard decentralized reinforcement learning on sequence models combined with co-player diversity provides a scalable path to learning cooperative behaviors. 

---
# Toward Scalable Verifiable Reward: Proxy State-Based Evaluation for Multi-turn Tool-Calling LLM Agents 

**Authors**: Yun-Shiuan Chuang, Chaitanya Kulkarni, Alec Chiu, Avinash Thangali, Zijie Pan, Shivani Shekhar, Yirou Ge, Yixi Li, Uma Kona, Linsey Pang, Prakhar Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2602.16246)  

**Abstract**: Interactive large language model (LLM) agents operating via multi-turn dialogue and multi-step tool calling are increasingly used in production. Benchmarks for these agents must both reliably compare models and yield on-policy training data. Prior agentic benchmarks (e.g., tau-bench, tau2-bench, AppWorld) rely on fully deterministic backends, which are costly to build and iterate. We propose Proxy State-Based Evaluation, an LLM-driven simulation framework that preserves final state-based evaluation without a deterministic database. Specifically, a scenario specifies the user goal, user/system facts, expected final state, and expected agent behavior, and an LLM state tracker infers a structured proxy state from the full interaction trace. LLM judges then verify goal completion and detect tool/user hallucinations against scenario constraints. Empirically, our benchmark produces stable, model-differentiating rankings across families and inference-time reasoning efforts, and its on-/off-policy rollouts provide supervision that transfers to unseen scenarios. Careful scenario specification yields near-zero simulator hallucination rates as supported by ablation studies. The framework also supports sensitivity analyses over user personas. Human-LLM judge agreement exceeds 90%, indicating reliable automated evaluation. Overall, proxy state-based evaluation offers a practical, scalable alternative to deterministic agentic benchmarks for industrial LLM agents. 

---
# Revolutionizing Long-Term Memory in AI: New Horizons with High-Capacity and High-Speed Storage 

**Authors**: Hiroaki Yamanaka, Daisuke Miyashita, Takashi Toi, Asuka Maki, Taiga Ikeda, Jun Deguchi  

**Link**: [PDF](https://arxiv.org/pdf/2602.16192)  

**Abstract**: Driven by our mission of "uplifting the world with memory," this paper explores the design concept of "memory" that is essential for achieving artificial superintelligence (ASI). Rather than proposing novel methods, we focus on several alternative approaches whose potential benefits are widely imaginable, yet have remained largely unexplored. The currently dominant paradigm, which can be termed "extract then store," involves extracting information judged to be useful from experiences and saving only the extracted content. However, this approach inherently risks the loss of information, as some valuable knowledge particularly for different tasks may be discarded in the extraction process. In contrast, we emphasize the "store then on-demand extract" approach, which seeks to retain raw experiences and flexibly apply them to various tasks as needed, thus avoiding such information loss. In addition, we highlight two further approaches: discovering deeper insights from large collections of probabilistic experiences, and improving experience collection efficiency by sharing stored experiences. While these approaches seem intuitively effective, our simple experiments demonstrate that this is indeed the case. Finally, we discuss major challenges that have limited investigation into these promising directions and propose research topics to address them. 

---
# EnterpriseGym Corecraft: Training Generalizable Agents on High-Fidelity RL Environments 

**Authors**: Sushant Mehta, Logan Ritchie, Suhaas Garre, Nick Heiner, Edwin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.16179)  

**Abstract**: We show that training AI agents on high-fidelity reinforcement learning environments produces capabilities that generalize beyond the training distribution. We introduce \corecraft{}, the first environment in \textsc{EnterpriseGym}, Surge AI's suite of agentic RL environments. \corecraft{} is a fully operational enterprise simulation of a customer support organization, comprising over 2,500 entities across 14 entity types with 23 unique tools, designed to measure whether AI agents can perform the multi-step, domain-specific work that real jobs demand. Frontier models such as GPT-5.2 and Claude Opus 4.6 solve fewer than 30\% of tasks when all expert-authored rubric criteria must be satisfied. Using this environment, we train GLM~4.6 with Group Relative Policy Optimization (GRPO) and adaptive clipping. After a single epoch of training, the model improves from 25.37\% to 36.76\% task pass rate on held-out evaluation tasks. More importantly, these gains transfer to out-of-distribution benchmarks: +4.5\% on BFCL Parallel, +7.4\% on $\tau^2$-Bench Retail, and +6.8\% on Toolathlon (Pass@1). We believe three environment properties are consistent with the observed transfer: task-centric world building that optimizes for diverse, challenging tasks; expert-authored rubrics enabling reliable reward computation; and enterprise workflows that reflect realistic professional patterns. Our results suggest that environment quality, diversity, and realism are key factors enabling generalizable agent capabilities. 

---
# Learning Personalized Agents from Human Feedback 

**Authors**: Kaiqu Liang, Julia Kruk, Shengyi Qian, Xianjun Yang, Shengjie Bi, Yuanshun Yao, Shaoliang Nie, Mingyang Zhang, Lijuan Liu, Jaime Fernández Fisac, Shuyan Zhou, Saghar Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2602.16173)  

**Abstract**: Modern AI agents are powerful but often fail to align with the idiosyncratic, evolving preferences of individual users. Prior approaches typically rely on static datasets, either training implicit preference models on interaction history or encoding user profiles in external memory. However, these approaches struggle with new users and with preferences that change over time. We introduce Personalized Agents from Human Feedback (PAHF), a framework for continual personalization in which agents learn online from live interaction using explicit per-user memory. PAHF operationalizes a three-step loop: (1) seeking pre-action clarification to resolve ambiguity, (2) grounding actions in preferences retrieved from memory, and (3) integrating post-action feedback to update memory when preferences drift. To evaluate this capability, we develop a four-phase protocol and two benchmarks in embodied manipulation and online shopping. These benchmarks quantify an agent's ability to learn initial preferences from scratch and subsequently adapt to persona shifts. Our theoretical analysis and empirical results show that integrating explicit memory with dual feedback channels is critical: PAHF learns substantially faster and consistently outperforms both no-memory and single-channel baselines, reducing initial personalization error and enabling rapid adaptation to preference shifts. 

---
# GPSBench: Do Large Language Models Understand GPS Coordinates? 

**Authors**: Thinh Hung Truong, Jey Han Lau, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2602.16105)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in applications that interact with the physical world, such as navigation, robotics, or mapping, making robust geospatial reasoning a critical capability. Despite that, LLMs' ability to reason about GPS coordinates and real-world geography remains underexplored. We introduce GPSBench, a dataset of 57,800 samples across 17 tasks for evaluating geospatial reasoning in LLMs, spanning geometric coordinate operations (e.g., distance and bearing computation) and reasoning that integrates coordinates with world knowledge. Focusing on intrinsic model capabilities rather than tool use, we evaluate 14 state-of-the-art LLMs and find that GPS reasoning remains challenging, with substantial variation across tasks: models are generally more reliable at real-world geographic reasoning than at geometric computations. Geographic knowledge degrades hierarchically, with strong country-level performance but weak city-level localization, while robustness to coordinate noise suggests genuine coordinate understanding rather than memorization. We further show that GPS-coordinate augmentation can improve in downstream geospatial tasks, and that finetuning induces trade-offs between gains in geometric computation and degradation in world knowledge. Our dataset and reproducible code are available at this https URL 

---
# Improving Interactive In-Context Learning from Natural Language Feedback 

**Authors**: Martin Klissarov, Jonathan Cook, Diego Antognini, Hao Sun, Jingling Li, Natasha Jaques, Claudiu Musat, Edward Grefenstette  

**Link**: [PDF](https://arxiv.org/pdf/2602.16066)  

**Abstract**: Adapting one's thought process based on corrective feedback is an essential ability in human learning, particularly in collaborative settings. In contrast, the current large language model training paradigm relies heavily on modeling vast, static corpora. While effective for knowledge acquisition, it overlooks the interactive feedback loops essential for models to adapt dynamically to their context. In this work, we propose a framework that treats this interactive in-context learning ability not as an emergent property, but as a distinct, trainable skill. We introduce a scalable method that transforms single-turn verifiable tasks into multi-turn didactic interactions driven by information asymmetry. We first show that current flagship models struggle to integrate corrective feedback on hard reasoning tasks. We then demonstrate that models trained with our approach dramatically improve the ability to interactively learn from language feedback. More specifically, the multi-turn performance of a smaller model nearly reaches that of a model an order of magnitude larger. We also observe robust out-of-distribution generalization: interactive training on math problems transfers to diverse domains like coding, puzzles and maze navigation. Our qualitative analysis suggests that this improvement is due to an enhanced in-context plasticity. Finally, we show that this paradigm offers a unified path to self-improvement. By training the model to predict the teacher's critiques, effectively modeling the feedback environment, we convert this external signal into an internal capability, allowing the model to self-correct even without a teacher. 

---
# Evidence-Grounded Subspecialty Reasoning: Evaluating a Curated Clinical Intelligence Layer on the 2025 Endocrinology Board-Style Examination 

**Authors**: Amir Hosseinian, MohammadReza Zare Shahneh, Umer Mansoor, Gilbert Szeto, Kirill Karlin, Nima Aghaeepour  

**Link**: [PDF](https://arxiv.org/pdf/2602.16050)  

**Abstract**: Background: Large language models have demonstrated strong performance on general medical examinations, but subspecialty clinical reasoning remains challenging due to rapidly evolving guidelines and nuanced evidence hierarchies. Methods: We evaluated January Mirror, an evidence-grounded clinical reasoning system, against frontier LLMs (GPT-5, GPT-5.2, Gemini-3-Pro) on a 120-question endocrinology board-style examination. Mirror integrates a curated endocrinology and cardiometabolic evidence corpus with a structured reasoning architecture to generate evidence-linked outputs. Mirror operated under a closed-evidence constraint without external retrieval. Comparator LLMs had real-time web access to guidelines and primary literature. Results: Mirror achieved 87.5% accuracy (105/120; 95% CI: 80.4-92.3%), exceeding a human reference of 62.3% and frontier LLMs including GPT-5.2 (74.6%), GPT-5 (74.0%), and Gemini-3-Pro (69.8%). On the 30 most difficult questions (human accuracy less than 50%), Mirror achieved 76.7% accuracy. Top-2 accuracy was 92.5% for Mirror versus 85.25% for GPT-5.2. Conclusions: Mirror provided evidence traceability: 74.2% of outputs cited at least one guideline-tier source, with 100% citation accuracy on manual verification. Curated evidence with explicit provenance can outperform unconstrained web retrieval for subspecialty clinical reasoning and supports auditability for clinical deployment. 

---
# How Uncertain Is the Grade? A Benchmark of Uncertainty Metrics for LLM-Based Automatic Assessment 

**Authors**: Hang Li, Kaiqi Yang, Xianxuan Long, Fedor Filippov, Yucheng Chu, Yasemin Copur-Gencturk, Peng He, Cory Miller, Namsoo Shin, Joseph Krajcik, Hui Liu, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.16039)  

**Abstract**: The rapid rise of large language models (LLMs) is reshaping the landscape of automatic assessment in education. While these systems demonstrate substantial advantages in adaptability to diverse question types and flexibility in output formats, they also introduce new challenges related to output uncertainty, stemming from the inherently probabilistic nature of LLMs. Output uncertainty is an inescapable challenge in automatic assessment, as assessment results often play a critical role in informing subsequent pedagogical actions, such as providing feedback to students or guiding instructional decisions. Unreliable or poorly calibrated uncertainty estimates can lead to unstable downstream interventions, potentially disrupting students' learning processes and resulting in unintended negative consequences. To systematically understand this challenge and inform future research, we benchmark a broad range of uncertainty quantification methods in the context of LLM-based automatic assessment. Although the effectiveness of these methods has been demonstrated in many tasks across other domains, their applicability and reliability in educational settings, particularly for automatic grading, remain underexplored. Through comprehensive analyses of uncertainty behaviors across multiple assessment datasets, LLM families, and generation control settings, we characterize the uncertainty patterns exhibited by LLMs in grading scenarios. Based on these findings, we evaluate the strengths and limitations of different uncertainty metrics and analyze the influence of key factors, including model families, assessment tasks, and decoding strategies, on uncertainty estimates. Our study provides actionable insights into the characteristics of uncertainty in LLM-based automatic assessment and lays the groundwork for developing more reliable and effective uncertainty-aware grading systems in the future. 

---
# Optimization Instability in Autonomous Agentic Workflows for Clinical Symptom Detection 

**Authors**: Cameron Cagan, Pedram Fard, Jiazi Tian, Jingya Cheng, Shawn N. Murphy, Hossein Estiri  

**Link**: [PDF](https://arxiv.org/pdf/2602.16037)  

**Abstract**: Autonomous agentic workflows that iteratively refine their own behavior hold considerable promise, yet their failure modes remain poorly characterized. We investigate optimization instability, a phenomenon in which continued autonomous improvement paradoxically degrades classifier performance, using Pythia, an open-source framework for automated prompt optimization. Evaluating three clinical symptoms with varying prevalence (shortness of breath at 23%, chest pain at 12%, and Long COVID brain fog at 3%), we observed that validation sensitivity oscillated between 1.0 and 0.0 across iterations, with severity inversely proportional to class prevalence. At 3% prevalence, the system achieved 95% accuracy while detecting zero positive cases, a failure mode obscured by standard evaluation metrics. We evaluated two interventions: a guiding agent that actively redirected optimization, amplifying overfitting rather than correcting it, and a selector agent that retrospectively identified the best-performing iteration successfully prevented catastrophic failure. With selector agent oversight, the system outperformed expert-curated lexicons on brain fog detection by 331% (F1) and chest pain by 7%, despite requiring only a single natural language term as input. These findings characterize a critical failure mode of autonomous AI systems and demonstrate that retrospective selection outperforms active intervention for stabilization in low-prevalence classification tasks. 

---
# Towards Efficient Constraint Handling in Neural Solvers for Routing Problems 

**Authors**: Jieyi Bi, Zhiguang Cao, Jianan Zhou, Wen Song, Yaoxin Wu, Jie Zhang, Yining Ma, Cathy Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.16012)  

**Abstract**: Neural solvers have achieved impressive progress in addressing simple routing problems, particularly excelling in computational efficiency. However, their advantages under complex constraints remain nascent, for which current constraint-handling schemes via feasibility masking or implicit feasibility awareness can be inefficient or inapplicable for hard constraints. In this paper, we present Construct-and-Refine (CaR), the first general and efficient constraint-handling framework for neural routing solvers based on explicit learning-based feasibility refinement. Unlike prior construction-search hybrids that target reducing optimality gaps through heavy improvements yet still struggle with hard constraints, CaR achieves efficient constraint handling by designing a joint training framework that guides the construction module to generate diverse and high-quality solutions well-suited for a lightweight improvement process, e.g., 10 steps versus 5k steps in prior work. Moreover, CaR presents the first use of construction-improvement-shared representation, enabling potential knowledge sharing across paradigms by unifying the encoder, especially in more complex constrained scenarios. We evaluate CaR on typical hard routing constraints to showcase its broader applicability. Results demonstrate that CaR achieves superior feasibility, solution quality, and efficiency compared to both classical and neural state-of-the-art solvers. 

---
# Policy Compiler for Secure Agentic Systems 

**Authors**: Nils Palumbo, Sarthak Choudhary, Jihye Choi, Prasad Chalasani, Mihai Christodorescu, Somesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2602.16708)  

**Abstract**: LLM-based agents are increasingly being deployed in contexts requiring complex authorization policies: customer service protocols, approval workflows, data access restrictions, and regulatory compliance. Embedding these policies in prompts provides no enforcement guarantees. We present PCAS, a Policy Compiler for Agentic Systems that provides deterministic policy enforcement.
Enforcing such policies requires tracking information flow across agents, which linear message histories cannot capture. Instead, PCAS models the agentic system state as a dependency graph capturing causal relationships among events such as tool calls, tool results, and messages. Policies are expressed in a Datalog-derived language, as declarative rules that account for transitive information flow and cross-agent provenance. A reference monitor intercepts all actions and blocks violations before execution, providing deterministic enforcement independent of model reasoning.
PCAS takes an existing agent implementation and a policy specification, and compiles them into an instrumented system that is policy-compliant by construction, with no security-specific restructuring required. We evaluate PCAS on three case studies: information flow policies for prompt injection defense, approval workflows in a multi-agent pharmacovigilance system, and organizational policies for customer service. On customer service tasks, PCAS improves policy compliance from 48% to 93% across frontier models, with zero policy violations in instrumented runs. 

---
# Measuring Mid-2025 LLM-Assistance on Novice Performance in Biology 

**Authors**: Shen Zhou Hong, Alex Kleinman, Alyssa Mathiowetz, Adam Howes, Julian Cohen, Suveer Ganta, Alex Letizia, Dora Liao, Deepika Pahari, Xavier Roberts-Gaal, Luca Righetti, Joe Torres  

**Link**: [PDF](https://arxiv.org/pdf/2602.16703)  

**Abstract**: Large language models (LLMs) perform strongly on biological benchmarks, raising concerns that they may help novice actors acquire dual-use laboratory skills. Yet, whether this translates to improved human performance in the physical laboratory remains unclear. To address this, we conducted a pre-registered, investigator-blinded, randomized controlled trial (June-August 2025; n = 153) evaluating whether LLMs improve novice performance in tasks that collectively model a viral reverse genetics workflow. We observed no significant difference in the primary endpoint of workflow completion (5.2% LLM vs. 6.6% Internet; P = 0.759), nor in the success rate of individual tasks. However, the LLM arm had numerically higher success rates in four of the five tasks, most notably for the cell culture task (68.8% LLM vs. 55.3% Internet; P = 0.059). Post-hoc Bayesian modeling of pooled data estimates an approximate 1.4-fold increase (95% CrI 0.74-2.62) in success for a "typical" reverse genetics task under LLM assistance. Ordinal regression modelling suggests that participants in the LLM arm were more likely to progress through intermediate steps across all tasks (posterior probability of a positive effect: 81%-96%). Overall, mid-2025 LLMs did not substantially increase novice completion of complex laboratory procedures but were associated with a modest performance benefit. These results reveal a gap between in silico benchmarks and real-world utility, underscoring the need for physical-world validation of AI biosecurity assessments as model capabilities and user proficiency evolve. 

---
# Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents 

**Authors**: Wenxuan Ding, Nicholas Tomlin, Greg Durrett  

**Link**: [PDF](https://arxiv.org/pdf/2602.16699)  

**Abstract**: LLMs are increasingly being used for complex problems which are not necessarily resolved in a single response, but require interacting with an environment to acquire information. In these scenarios, LLMs must reason about inherent cost-uncertainty tradeoffs in when to stop exploring and commit to an answer. For instance, on a programming task, an LLM should test a generated code snippet if it is uncertain about the correctness of that code; the cost of writing a test is nonzero, but typically lower than the cost of making a mistake. In this work, we show that we can induce LLMs to explicitly reason about balancing these cost-uncertainty tradeoffs, then perform more optimal environment exploration. We formalize multiple tasks, including information retrieval and coding, as sequential decision-making problems under uncertainty. Each problem has latent environment state that can be reasoned about via a prior which is passed to the LLM agent. We introduce a framework called Calibrate-Then-Act (CTA), where we feed the LLM this additional context to enable it to act more optimally. This improvement is preserved even under RL training of both the baseline and CTA. Our results on information-seeking QA and on a simplified coding task show that making cost-benefit tradeoffs explicit with CTA can help agents discover more optimal decision-making strategies. 

---
# SPARC: Scenario Planning and Reasoning for Automated C Unit Test Generation 

**Authors**: Jaid Monwar Chowdhury, Chi-An Fu, Reyhaneh Jabbarvand  

**Link**: [PDF](https://arxiv.org/pdf/2602.16671)  

**Abstract**: Automated unit test generation for C remains a formidable challenge due to the semantic gap between high-level program intent and the rigid syntactic constraints of pointer arithmetic and manual memory management. While Large Language Models (LLMs) exhibit strong generative capabilities, direct intent-to-code synthesis frequently suffers from the leap-to-code failure mode, where models prematurely emit code without grounding in program structure, constraints, and semantics. This will result in non-compilable tests, hallucinated function signatures, low branch coverage, and semantically irrelevant assertions that cannot properly capture bugs. We introduce SPARC, a neuro-symbolic, scenario-based framework that bridges this gap through four stages: (1) Control Flow Graph (CFG) analysis, (2) an Operation Map that grounds LLM reasoning in validated utility helpers, (3) Path-targeted test synthesis, and (4) an iterative, self-correction validation loop using compiler and runtime feedback. We evaluate SPARC on 59 real-world and algorithmic subjects, where it outperforms the vanilla prompt generation baseline by 31.36% in line coverage, 26.01% in branch coverage, and 20.78% in mutation score, matching or exceeding the symbolic execution tool KLEE on complex subjects. SPARC retains 94.3% of tests through iterative repair and produces code with significantly higher developer-rated readability and maintainability. By aligning LLM reasoning with program structure, SPARC provides a scalable path for industrial-grade testing of legacy C codebases. 

---
# Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment 

**Authors**: Yuyan Bu, Xiaohao Liu, ZhaoXing Ren, Yaodong Yang, Juntao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2602.16660)  

**Abstract**: The widespread deployment of large language models (LLMs) across linguistic communities necessitates reliable multilingual safety alignment. However, recent efforts to extend alignment to other languages often require substantial resources, either through large-scale, high-quality supervision in the target language or through pairwise alignment with high-resource languages, which limits scalability. In this work, we propose a resource-efficient method for improving multilingual safety alignment. We introduce a plug-and-play Multi-Lingual Consistency (MLC) loss that can be integrated into existing monolingual alignment pipelines. By improving collinearity between multilingual representation vectors, our method encourages directional consistency at the multilingual semantic level in a single update. This allows simultaneous alignment across multiple languages using only multilingual prompt variants without requiring additional response-level supervision in low-resource languages. We validate the proposed method across different model architectures and alignment paradigms, and demonstrate its effectiveness in enhancing multilingual safety with limited impact on general model utility. Further evaluation across languages and tasks indicates improved cross-lingual generalization, suggesting the proposed approach as a practical solution for multilingual consistency alignment under limited supervision. 

---
# Retrieval Augmented Generation of Literature-derived Polymer Knowledge: The Example of a Biodegradable Polymer Expert System 

**Authors**: Sonakshi Gupta, Akhlak Mahmood, Wei Xiong, Rampi Ramprasad  

**Link**: [PDF](https://arxiv.org/pdf/2602.16650)  

**Abstract**: Polymer literature contains a large and growing body of experimental knowledge, yet much of it is buried in unstructured text and inconsistent terminology, making systematic retrieval and reasoning difficult. Existing tools typically extract narrow, study-specific facts in isolation, failing to preserve the cross-study context required to answer broader scientific questions. Retrieval-augmented generation (RAG) offers a promising way to overcome this limitation by combining large language models (LLMs) with external retrieval, but its effectiveness depends strongly on how domain knowledge is represented. In this work, we develop two retrieval pipelines: a dense semantic vector-based approach (VectorRAG) and a graph-based approach (GraphRAG). Using over 1,000 polyhydroxyalkanoate (PHA) papers, we construct context-preserving paragraph embeddings and a canonicalized structured knowledge graph supporting entity disambiguation and multi-hop reasoning. We evaluate these pipelines through standard retrieval metrics, comparisons with general state-of-the-art systems such as GPT and Gemini, and qualitative validation by a domain chemist. The results show that GraphRAG achieves higher precision and interpretability, while VectorRAG provides broader recall, highlighting complementary trade-offs. Expert validation further confirms that the tailored pipelines, particularly GraphRAG, produce well-grounded, citation-reliable responses with strong domain relevance. By grounding every statement in evidence, these systems enable researchers to navigate the literature, compare findings across studies, and uncover patterns that are difficult to extract manually. More broadly, this work establishes a practical framework for building materials science assistants using curated corpora and retrieval design, reducing reliance on proprietary models while enabling trustworthy literature analysis at scale. 

---
# Enhanced Diffusion Sampling: Efficient Rare Event Sampling and Free Energy Calculation with Diffusion Models 

**Authors**: Yu Xie, Ludwig Winkler, Lixin Sun, Sarah Lewis, Adam E. Foster, José Jiménez Luna, Tim Hempel, Michael Gastegger, Yaoyi Chen, Iryna Zaporozhets, Cecilia Clementi, Christopher M. Bishop, Frank Noé  

**Link**: [PDF](https://arxiv.org/pdf/2602.16634)  

**Abstract**: The rare-event sampling problem has long been the central limiting factor in molecular dynamics (MD), especially in biomolecular simulation. Recently, diffusion models such as BioEmu have emerged as powerful equilibrium samplers that generate independent samples from complex molecular distributions, eliminating the cost of sampling rare transition events. However, a sampling problem remains when computing observables that rely on states which are rare in equilibrium, for example folding free energies. Here, we introduce enhanced diffusion sampling, enabling efficient exploration of rare-event regions while preserving unbiased thermodynamic estimators. The key idea is to perform quantitatively accurate steering protocols to generate biased ensembles and subsequently recover equilibrium statistics via exact reweighting. We instantiate our framework in three algorithms: UmbrellaDiff (umbrella sampling with diffusion models), $\Delta$G-Diff (free-energy differences via tilted ensembles), and MetaDiff (a batchwise analogue for metadynamics). Across toy systems, protein folding landscapes and folding free energies, our methods achieve fast, accurate, and scalable estimation of equilibrium properties within GPU-minutes to hours per system -- closing the rare-event sampling gap that remained after the advent of diffusion-model equilibrium samplers. 

---
# Almost Sure Convergence of Differential Temporal Difference Learning for Average Reward Markov Decision Processes 

**Authors**: Ethan Blaser, Jiuqi Wang, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.16629)  

**Abstract**: The average reward is a fundamental performance metric in reinforcement learning (RL) focusing on the long-run performance of an agent. Differential temporal difference (TD) learning algorithms are a major advance for average reward RL as they provide an efficient online method to learn the value functions associated with the average reward in both on-policy and off-policy settings. However, existing convergence guarantees require a local clock in learning rates tied to state visit counts, which practitioners do not use and does not extend beyond tabular settings. We address this limitation by proving the almost sure convergence of on-policy $n$-step differential TD for any $n$ using standard diminishing learning rates without a local clock. We then derive three sufficient conditions under which off-policy $n$-step differential TD also converges without a local clock. These results strengthen the theoretical foundations of differential TD and bring its convergence analysis closer to practical implementations. 

---
# A Systematic Evaluation of Sample-Level Tokenization Strategies for MEG Foundation Models 

**Authors**: SungJun Cho, Chetan Gohil, Rukuang Huang, Oiwi Parker Jones, Mark W. Woolrich  

**Link**: [PDF](https://arxiv.org/pdf/2602.16626)  

**Abstract**: Recent success in natural language processing has motivated growing interest in large-scale foundation models for neuroimaging data. Such models often require discretization of continuous neural time series data, a process referred to as 'tokenization'. However, the impact of different tokenization strategies for neural data is currently poorly understood. In this work, we present a systematic evaluation of sample-level tokenization strategies for transformer-based large neuroimaging models (LNMs) applied to magnetoencephalography (MEG) data. We compare learnable and non-learnable tokenizers by examining their signal reconstruction fidelity and their impact on subsequent foundation modeling performance (token prediction, biological plausibility of generated data, preservation of subject-specific information, and performance on downstream tasks). For the learnable tokenizer, we introduce a novel approach based on an autoencoder. Experiments were conducted on three publicly available MEG datasets spanning different acquisition sites, scanners, and experimental paradigms. Our results show that both learnable and non-learnable discretization schemes achieve high reconstruction accuracy and broadly comparable performance across most evaluation criteria, suggesting that simple fixed sample-level tokenization strategies can be used in the development of neural foundation models. The code is available at this https URL. 

---
# Causal and Compositional Abstraction 

**Authors**: Robin Lorenz, Sean Tull  

**Link**: [PDF](https://arxiv.org/pdf/2602.16612)  

**Abstract**: Abstracting from a low level to a more explanatory high level of description, and ideally while preserving causal structure, is fundamental to scientific practice, to causal inference problems, and to robust, efficient and interpretable AI. We present a general account of abstractions between low and high level models as natural transformations, focusing on the case of causal models. This provides a new formalisation of causal abstraction, unifying several notions in the literature, including constructive causal abstraction, Q-$\tau$ consistency, abstractions based on interchange interventions, and `distributed' causal abstractions. Our approach is formalised in terms of category theory, and uses the general notion of a compositional model with a given set of queries and semantics in a monoidal, cd- or Markov category; causal models and their queries such as interventions being special cases. We identify two basic notions of abstraction: downward abstractions mapping queries from high to low level; and upward abstractions, mapping concrete queries such as Do-interventions from low to high. Although usually presented as the latter, we show how common causal abstractions may, more fundamentally, be understood in terms of the former. Our approach also leads us to consider a new stronger notion of `component-level' abstraction, applying to the individual components of a model. In particular, this yields a novel, strengthened form of constructive causal abstraction at the mechanism-level, for which we prove characterisation results. Finally, we show that abstraction can be generalised to further compositional models, including those with a quantum semantics implemented by quantum circuits, and we take first steps in exploring abstractions between quantum compositional circuit models and high-level classical causal models as a means to explainable quantum AI. 

---
# Who can we trust? LLM-as-a-jury for Comparative Assessment 

**Authors**: Mengjie Qian, Guangzhi Sun, Mark J.F. Gales, Kate M. Knill  

**Link**: [PDF](https://arxiv.org/pdf/2602.16610)  

**Abstract**: Large language models (LLMs) are increasingly applied as automatic evaluators for natural language generation assessment often using pairwise comparative judgements. Existing approaches typically rely on single judges or aggregate multiple judges assuming equal reliability. In practice, LLM judges vary substantially in performance across tasks and aspects, and their judgment probabilities may be biased and inconsistent. Furthermore, human-labelled supervision for judge calibration may be unavailable. We first empirically demonstrate that inconsistencies in LLM comparison probabilities exist and show that it limits the effectiveness of direct probability-based ranking. To address this, we study the LLM-as-a-jury setting and propose BT-sigma, a judge-aware extension of the Bradley-Terry model that introduces a discriminator parameter for each judge to jointly infer item rankings and judge reliability from pairwise comparisons alone. Experiments on benchmark NLG evaluation datasets show that BT-sigma consistently outperforms averaging-based aggregation methods, and that the learned discriminator strongly correlates with independent measures of the cycle consistency of LLM judgments. Further analysis reveals that BT-sigma can be interpreted as an unsupervised calibration mechanism that improves aggregation by modelling judge reliability. 

---
# Explainable AI: Context-Aware Layer-Wise Integrated Gradients for Explaining Transformer Models 

**Authors**: Melkamu Abay Mersha, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2602.16608)  

**Abstract**: Transformer models achieve state-of-the-art performance across domains and tasks, yet their deeply layered representations make their predictions difficult to interpret. Existing explainability methods rely on final-layer attributions, capture either local token-level attributions or global attention patterns without unification, and lack context-awareness of inter-token dependencies and structural components. They also fail to capture how relevance evolves across layers and how structural components shape decision-making. To address these limitations, we proposed the \textbf{Context-Aware Layer-wise Integrated Gradients (CA-LIG) Framework}, a unified hierarchical attribution framework that computes layer-wise Integrated Gradients within each Transformer block and fuses these token-level attributions with class-specific attention gradients. This integration yields signed, context-sensitive attribution maps that capture supportive and opposing evidence while tracing the hierarchical flow of relevance through the Transformer layers. We evaluate the CA-LIG Framework across diverse tasks, domains, and transformer model families, including sentiment analysis and long and multi-class document classification with BERT, hate speech detection in a low-resource language setting with XLM-R and AfroLM, and image classification with Masked Autoencoder vision Transformer model. Across all tasks and architectures, CA-LIG provides more faithful attributions, shows stronger sensitivity to contextual dependencies, and produces clearer, more semantically coherent visualizations than established explainability methods. These results indicate that CA-LIG provides a more comprehensive, context-aware, and reliable explanation of Transformer decision-making, advancing both the practical interpretability and conceptual understanding of deep neural models. 

---
# FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity to Mitigate Head-of-Line Blocking in LLM Serving 

**Authors**: Chia-chi Hsieh, Zan Zong, Xinyang Chen, Jianjiang Li, Jidong Zhai, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.16603)  

**Abstract**: The growing demand for large language models (LLMs) requires serving systems to handle many concurrent requests with diverse service level objectives (SLOs). This exacerbates head-of-line (HoL) blocking during the compute-intensive prefill phase, where long-running requests monopolize resources and delay higher-priority ones, leading to widespread time-to-first-token (TTFT) SLO violations. While chunked prefill enables interruptibility, it introduces an inherent trade-off between responsiveness and throughput: reducing chunk size improves response latency but degrades computational efficiency, whereas increasing chunk size maximizes throughput but exacerbates blocking. This necessitates an adaptive preemption mechanism. However, dynamically balancing execution granularity against scheduling overheads remains a key challenge.
In this paper, we propose FlowPrefill, a TTFT-goodput-optimized serving system that resolves this conflict by decoupling preemption granularity from scheduling frequency. To achieve adaptive prefill scheduling, FlowPrefill introduces two key innovations: 1) Operator-Level Preemption, which leverages operator boundaries to enable fine-grained execution interruption without the efficiency loss associated with fixed small chunking; and 2) Event-Driven Scheduling, which triggers scheduling decisions only upon request arrival or completion events, thereby supporting efficient preemption responsiveness while minimizing control-plane overhead. Evaluation on real-world production traces shows that FlowPrefill improves maximum goodput by up to 5.6$\times$ compared to state-of-the-art systems while satisfying heterogeneous SLOs. 

---
# A Contrastive Learning Framework Empowered by Attention-based Feature Adaptation for Street-View Image Classification 

**Authors**: Qi You, Yitai Cheng, Zichao Zeng, James Haworth  

**Link**: [PDF](https://arxiv.org/pdf/2602.16590)  

**Abstract**: Street-view image attribute classification is a vital downstream task of image classification, enabling applications such as autonomous driving, urban analytics, and high-definition map construction. It remains computationally demanding whether training from scratch, initialising from pre-trained weights, or fine-tuning large models. Although pre-trained vision-language models such as CLIP offer rich image representations, existing adaptation or fine-tuning methods often rely on their global image embeddings, limiting their ability to capture fine-grained, localised attributes essential in complex, cluttered street scenes. To address this, we propose CLIP-MHAdapter, a variant of the current lightweight CLIP adaptation paradigm that appends a bottleneck MLP equipped with multi-head self-attention operating on patch tokens to model inter-patch dependencies. With approximately 1.4 million trainable parameters, CLIP-MHAdapter achieves superior or competitive accuracy across eight attribute classification tasks on the Global StreetScapes dataset, attaining new state-of-the-art results while maintaining low computational cost. The code is available at this https URL. 

---
# DataJoint 2.0: A Computational Substrate for Agentic Scientific Workflows 

**Authors**: Dimitri Yatsenko, Thinh T. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2602.16585)  

**Abstract**: Operational rigor determines whether human-agent collaboration succeeds or fails. Scientific data pipelines need the equivalent of DevOps -- SciOps -- yet common approaches fragment provenance across disconnected systems without transactional guarantees. DataJoint 2.0 addresses this gap through the relational workflow model: tables represent workflow steps, rows represent artifacts, foreign keys prescribe execution order. The schema specifies not only what data exists but how it is derived -- a single formal system where data structure, computational dependencies, and integrity constraints are all queryable, enforceable, and machine-readable. Four technical innovations extend this foundation: object-augmented schemas integrating relational metadata with scalable object storage, semantic matching using attribute lineage to prevent erroneous joins, an extensible type system for domain-specific formats, and distributed job coordination designed for composability with external orchestration. By unifying data structure, data, and computational transformations, DataJoint creates a substrate for SciOps where agents can participate in scientific workflows without risking data corruption. 

---
# AIFL: A Global Daily Streamflow Forecasting Model Using Deterministic LSTM Pre-trained on ERA5-Land and Fine-tuned on IFS 

**Authors**: Maria Luisa Taccari, Kenza Tazi, Oisín M. Morrison, Andreas Grafberger, Juan Colonese, Corentin Carton de Wiart, Christel Prudhomme, Cinzia Mazzetti, Matthew Chantry, Florian Pappenberger  

**Link**: [PDF](https://arxiv.org/pdf/2602.16579)  

**Abstract**: Reliable global streamflow forecasting is essential for flood preparedness and water resource management, yet data-driven models often suffer from a performance gap when transitioning from historical reanalysis to operational forecast products. This paper introduces AIFL (Artificial Intelligence for Floods), a deterministic LSTM-based model designed for global daily streamflow forecasting. Trained on 18,588 basins curated from the CARAVAN dataset, AIFL utilises a novel two-stage training strategy to bridge the reanalysis-to-forecast domain shift. The model is first pre-trained on 40 years of ERA5-Land reanalysis (1980-2019) to capture robust hydrological processes, then fine-tuned on operational Integrated Forecasting System (IFS) control forecasts (2016-2019) to adapt to the specific error structures and biases of operational numerical weather prediction. To our knowledge, this is the first global model trained end-to-end within the CARAVAN ecosystem. On an independent temporal test set (2021-2024), AIFL achieves high predictive skill with a median modified Kling-Gupta Efficiency (KGE') of 0.66 and a median Nash-Sutcliffe Efficiency (NSE) of 0.53. Benchmarking results show that AIFL is highly competitive with current state-of-the-art global systems, achieving comparable accuracy while maintaining a transparent and reproducible forcing pipeline. The model demonstrates exceptional reliability in extreme-event detection, providing a streamlined and operationally robust baseline for the global hydrological community. 

---
# MerLean: An Agentic Framework for Autoformalization in Quantum Computation 

**Authors**: Yuanjie Ren, Jinzheng Li, Yidi Qi  

**Link**: [PDF](https://arxiv.org/pdf/2602.16554)  

**Abstract**: We introduce MerLean, a fully automated agentic framework for autoformalization in quantum computation. MerLean extracts mathematical statements from \LaTeX{} source files, formalizes them into verified Lean~4 code built on Mathlib, and translates the result back into human-readable \LaTeX{} for semantic review. We evaluate MerLean on three theoretical quantum computing papers producing 2,050 Lean declarations from 114 statements in total. MerLean achieves end-to-end formalization on all three papers, reducing the verification burden to only the newly introduced definitions and axioms. Our results demonstrate that agentic autoformalization can scale to frontier research, offering both a practical tool for machine-verified peer review and a scalable engine for mining high-quality synthetic data to train future reasoning models. Our approach can also be generalized to any other rigorous research in mathematics and theoretical physics. 

---
# Recursive language models for jailbreak detection: a procedural defense for tool-augmented agents 

**Authors**: Doron Shavit  

**Link**: [PDF](https://arxiv.org/pdf/2602.16520)  

**Abstract**: Jailbreak prompts are a practical and evolving threat to large language models (LLMs), particularly in agentic systems that execute tools over untrusted content. Many attacks exploit long-context hiding, semantic camouflage, and lightweight obfuscations that can evade single-pass guardrails. We present RLM-JB, an end-to-end jailbreak detection framework built on Recursive Language Models (RLMs), in which a root model orchestrates a bounded analysis program that transforms the input, queries worker models over covered segments, and aggregates evidence into an auditable decision. RLM-JB treats detection as a procedure rather than a one-shot classification: it normalizes and de-obfuscates suspicious inputs, chunks text to reduce context dilution and guarantee coverage, performs parallel chunk screening, and composes cross-chunk signals to recover split-payload attacks. On AutoDAN-style adversarial inputs, RLM-JB achieves high detection effectiveness across three LLM backends (ASR/Recall 92.5-98.0%) while maintaining very high precision (98.99-100%) and low false positive rates (0.0-2.0%), highlighting a practical sensitivity-specificity trade-off as the screening backend changes. 

---
# Interpretability-by-Design with Accurate Locally Additive Models and Conditional Feature Effects 

**Authors**: Vasilis Gkolemis, Loukas Kavouras, Dimitrios Kyriakopoulos, Konstantinos Tsopelas, Dimitrios Rontogiannis, Giuseppe Casalicchio, Theodore Dalamagas, Christos Diou  

**Link**: [PDF](https://arxiv.org/pdf/2602.16503)  

**Abstract**: Generalized additive models (GAMs) offer interpretability through independent univariate feature effects but underfit when interactions are present in data. GA$^2$Ms add selected pairwise interactions which improves accuracy, but sacrifices interpretability and limits model auditing. We propose \emph{Conditionally Additive Local Models} (CALMs), a new model class, that balances the interpretability of GAMs with the accuracy of GA$^2$Ms. CALMs allow multiple univariate shape functions per feature, each active in different regions of the input space. These regions are defined independently for each feature as simple logical conditions (thresholds) on the features it interacts with. As a result, effects remain locally additive while varying across subregions to capture interactions. We further propose a principled distillation-based training pipeline that identifies homogeneous regions with limited interactions and fits interpretable shape functions via region-aware backfitting. Experiments on diverse classification and regression tasks show that CALMs consistently outperform GAMs and achieve accuracy comparable with GA$^2$Ms. Overall, CALMs offer a compelling trade-off between predictive accuracy and interpretability. 

---
# Fast and Scalable Analytical Diffusion 

**Authors**: Xinyi Shang, Peng Sun, Jingyu Lin, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.16498)  

**Abstract**: Analytical diffusion models offer a mathematically transparent path to generative modeling by formulating the denoising score as an empirical-Bayes posterior mean. However, this interpretability comes at a prohibitive cost: the standard formulation necessitates a full-dataset scan at every timestep, scaling linearly with dataset size. In this work, we present the first systematic study addressing this scalability bottleneck. We challenge the prevailing assumption that the entire training data is necessary, uncovering the phenomenon of Posterior Progressive Concentration: the effective golden support of the denoising score is not static but shrinks asymptotically from the global manifold to a local neighborhood as the signal-to-noise ratio increases. Capitalizing on this, we propose Dynamic Time-Aware Golden Subset Diffusion (GoldDiff), a training-free framework that decouples inference complexity from dataset size. Instead of static retrieval, GoldDiff uses a coarse-to-fine mechanism to dynamically pinpoint the ''Golden Subset'' for inference. Theoretically, we derive rigorous bounds guaranteeing that our sparse approximation converges to the exact score. Empirically, GoldDiff achieves a $\bf 71 \times$ speedup on AFHQ while matching or achieving even better performance than full-scan baselines. Most notably, we demonstrate the first successful scaling of analytical diffusion to ImageNet-1K, unlocking a scalable, training-free paradigm for large-scale generative modeling. 

---
# From Growing to Looping: A Unified View of Iterative Computation in LLMs 

**Authors**: Ferdinand Kapl, Emmanouil Angelis, Kaitlin Maile, Johannes von Oswald, Stefan Bauer  

**Link**: [PDF](https://arxiv.org/pdf/2602.16490)  

**Abstract**: Looping, reusing a block of layers across depth, and depth growing, training shallow-to-deep models by duplicating middle layers, have both been linked to stronger reasoning, but their relationship remains unclear. We provide a mechanistic unification: looped and depth-grown models exhibit convergent depth-wise signatures, including increased reliance on late layers and recurring patterns aligned with the looped or grown block. These shared signatures support the view that their gains stem from a common form of iterative computation. Building on this connection, we show that the two techniques are adaptable and composable: applying inference-time looping to the middle blocks of a depth-grown model improves accuracy on some reasoning primitives by up to $2\times$, despite the model never being trained to loop. Both approaches also adapt better than the baseline when given more in-context examples or additional supervised fine-tuning data. Additionally, depth-grown models achieve the largest reasoning gains when using higher-quality, math-heavy cooldown mixtures, which can be further boosted by adapting a middle block to loop. Overall, our results position depth growth and looping as complementary, practical methods for inducing and scaling iterative computation to improve reasoning. 

---
# Learning to Learn from Language Feedback with Social Meta-Learning 

**Authors**: Jonathan Cook, Diego Antognini, Martin Klissarov, Claudiu Musat, Edward Grefenstette  

**Link**: [PDF](https://arxiv.org/pdf/2602.16488)  

**Abstract**: Large language models (LLMs) often struggle to learn from corrective feedback within a conversational context. They are rarely proactive in soliciting this feedback, even when faced with ambiguity, which can make their dialogues feel static, one-sided, and lacking the adaptive qualities of human conversation. To address these limitations, we draw inspiration from social meta-learning (SML) in humans - the process of learning how to learn from others. We formulate SML as a finetuning methodology, training LLMs to solicit and learn from language feedback in simulated pedagogical dialogues, where static tasks are converted into interactive social learning problems. SML effectively teaches models to use conversation to solve problems they are unable to solve in a single turn. This capability generalises across domains; SML on math problems produces models that better use feedback to solve coding problems and vice versa. Furthermore, despite being trained only on fully-specified problems, these models are better able to solve underspecified tasks where critical information is revealed over multiple turns. When faced with this ambiguity, SML-trained models make fewer premature answer attempts and are more likely to ask for the information they need. This work presents a scalable approach to developing AI systems that effectively learn from language feedback. 

---
# Team of Thoughts: Efficient Test-time Scaling of Agentic Systems through Orchestrated Tool Calling 

**Authors**: Jeffrey T. H. Wong, Zixi Zhang, Junyi Liu, Yiren Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.16485)  

**Abstract**: Existing Multi-Agent Systems (MAS) typically rely on static, homogeneous model configurations, limiting their ability to exploit the distinct strengths of differently post-trained models. To address this, we introduce Team-of-Thoughts, a novel MAS architecture that leverages the complementary capabilities of heterogeneous agents via an orchestrator-tool paradigm. Our framework introduces two key mechanisms to optimize performance: (1) an orchestrator calibration scheme that identifies models with superior coordination capabilities, and (2) a self-assessment protocol where tool agents profile their own domain expertise to account for variations in post-training skills. During inference, the orchestrator dynamically activates the most suitable tool agents based on these proficiency profiles. Experiments on five reasoning and code generation benchmarks show that Team-of-Thoughts delivers consistently superior task performance. Notably, on AIME24 and LiveCodeBench, our approach achieves accuracies of 96.67% and 72.53%, respectively, substantially outperforming homogeneous role-play baselines, which score 80% and 65.93%. 

---
# IndicEval: A Bilingual Indian Educational Evaluation Framework for Large Language Models 

**Authors**: Saurabh Bharti, Gaurav Azad, Abhinaw Jagtap, Nachiket Tapas  

**Link**: [PDF](https://arxiv.org/pdf/2602.16467)  

**Abstract**: The rapid advancement of large language models (LLMs) necessitates evaluation frameworks that reflect real-world academic rigor and multilingual complexity. This paper introduces IndicEval, a scalable benchmarking platform designed to assess LLM performance using authentic high-stakes examination questions from UPSC, JEE, and NEET across STEM and humanities domains in both English and Hindi. Unlike synthetic benchmarks, IndicEval grounds evaluation in real examination standards, enabling realistic measurement of reasoning, domain knowledge, and bilingual adaptability. The framework automates assessment using Zero-Shot, Few-Shot, and Chain-of-Thought (CoT) prompting strategies and supports modular integration of new models and languages. Experiments conducted on Gemini 2.0 Flash, GPT-4, Claude, and LLaMA 3-70B reveal three major findings. First, CoT prompting consistently improves reasoning accuracy, with substantial gains across subjects and languages. Second, significant cross-model performance disparities persist, particularly in high-complexity examinations. Third, multilingual degradation remains a critical challenge, with marked accuracy drops in Hindi compared to English, especially under Zero-Shot conditions. These results highlight persistent gaps in bilingual reasoning and domain transfer. Overall, IndicEval provides a practice-oriented, extensible foundation for rigorous, equitable evaluation of LLMs in multilingual educational settings and offers actionable insights for improving reasoning robustness and language adaptability. 

---
# GICDM: Mitigating Hubness for Reliable Distance-Based Generative Model Evaluation 

**Authors**: Nicolas Salvy, Hugues Talbot, Bertrand Thirion  

**Link**: [PDF](https://arxiv.org/pdf/2602.16449)  

**Abstract**: Generative model evaluation commonly relies on high-dimensional embedding spaces to compute distances between samples. We show that dataset representations in these spaces are affected by the hubness phenomenon, which distorts nearest neighbor relationships and biases distance-based metrics. Building on the classical Iterative Contextual Dissimilarity Measure (ICDM), we introduce Generative ICDM (GICDM), a method to correct neighborhood estimation for both real and generated data. We introduce a multi-scale extension to improve empirical behavior. Extensive experiments on synthetic and real benchmarks demonstrate that GICDM resolves hubness-induced failures, restores reliable metric behavior, and improves alignment with human judgment. 

---
# RoboGene: Boosting VLA Pre-training via Diversity-Driven Agentic Framework for Real-World Task Generation 

**Authors**: Yixue Zhang, Kun Wu, Zhi Gao, Zhen Zhao, Pei Ren, Zhiyuan Xu, Fei Liao, Xinhua Wang, Shichao Fan, Di Wu, Qiuxuan Feng, Meng Li, Zhengping Che, Chang Liu, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.16444)  

**Abstract**: The pursuit of general-purpose robotic manipulation is hindered by the scarcity of diverse, real-world interaction data. Unlike data collection from web in vision or language, robotic data collection is an active process incurring prohibitive physical costs. Consequently, automated task curation to maximize data value remains a critical yet under-explored challenge. Existing manual methods are unscalable and biased toward common tasks, while off-the-shelf foundation models often hallucinate physically infeasible instructions. To address this, we introduce RoboGene, an agentic framework designed to automate the generation of diverse, physically plausible manipulation tasks across single-arm, dual-arm, and mobile robots. RoboGene integrates three core components: diversity-driven sampling for broad task coverage, self-reflection mechanisms to enforce physical constraints, and human-in-the-loop refinement for continuous improvement. We conduct extensive quantitative analysis and large-scale real-world experiments, collecting datasets of 18k trajectories and introducing novel metrics to assess task quality, feasibility, and diversity. Results demonstrate that RoboGene significantly outperforms state-of-the-art foundation models (e.g., GPT-4o, Gemini 2.5 Pro). Furthermore, real-world experiments show that VLA models pre-trained with RoboGene achieve higher success rates and superior generalization, underscoring the importance of high-quality task generation. Our project is available at this https URL. 

---
# Hardware-accelerated graph neural networks: an alternative approach for neuromorphic event-based audio classification and keyword spotting on SoC FPGA 

**Authors**: Kamil Jeziorek, Piotr Wzorek, Krzysztof Blachut, Hiroshi Nakano, Manon Dampfhoffer, Thomas Mesquida, Hiroaki Nishi, Thomas Dalgaty, Tomasz Kryjak  

**Link**: [PDF](https://arxiv.org/pdf/2602.16442)  

**Abstract**: As the volume of data recorded by embedded edge sensors increases, particularly from neuromorphic devices producing discrete event streams, there is a growing need for hardware-aware neural architectures that enable efficient, low-latency, and energy-conscious local processing. We present an FPGA implementation of event-graph neural networks for audio processing. We utilise an artificial cochlea that converts time-series signals into sparse event data, reducing memory and computation costs. Our architecture was implemented on a SoC FPGA and evaluated on two open-source datasets. For classification task, our baseline floating-point model achieves 92.7% accuracy on SHD dataset - only 2.4% below the state of the art - while requiring over 10x and 67x fewer parameters. On SSC, our models achieve 66.9-71.0% accuracy. Compared to FPGA-based spiking neural networks, our quantised model reaches 92.3% accuracy, outperforming them by up to 19.3% while reducing resource usage and latency. For SSC, we report the first hardware-accelerated evaluation. We further demonstrate the first end-to-end FPGA implementation of event-audio keyword spotting, combining graph convolutional layers with recurrent sequence modelling. The system achieves up to 95% word-end detection accuracy, with only 10.53 microsecond latency and 1.18 W power consumption, establishing a strong benchmark for energy-efficient event-driven KWS. 

---
# Intra-Fairness Dynamics: The Bias Spillover Effect in Targeted LLM Alignment 

**Authors**: Eva Paraschou, Line Harder Clemmensen, Sneha Das  

**Link**: [PDF](https://arxiv.org/pdf/2602.16438)  

**Abstract**: Conventional large language model (LLM) fairness alignment largely focuses on mitigating bias along single sensitive attributes, overlooking fairness as an inherently multidimensional and context-specific value. This approach risks creating systems that achieve narrow fairness metrics while exacerbating disparities along untargeted attributes, a phenomenon known as bias spillover. While extensively studied in machine learning, bias spillover remains critically underexplored in LLM alignment. In this work, we investigate how targeted gender alignment affects fairness across nine sensitive attributes in three state-of-the-art LLMs (Mistral 7B, Llama 3.1 8B, Qwen 2.5 7B). Using Direct Preference Optimization and the BBQ benchmark, we evaluate fairness under ambiguous and disambiguous contexts. Our findings reveal noticeable bias spillover: while aggregate results show improvements, context-aware analysis exposes significant degradations in ambiguous contexts, particularly for physical appearance ($p< 0.001$ across all models), sexual orientation, and disability status. We demonstrate that improving fairness along one attribute can inadvertently worsen disparities in others under uncertainty, highlighting the necessity of context-aware, multi-attribute fairness evaluation frameworks. 

---
# Designing Production-Scale OCR for India: Multilingual and Domain-Specific Systems 

**Authors**: Ali Faraz, Raja Kolla, Ashish Kulkarni, Shubham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2602.16430)  

**Abstract**: Designing Optical Character Recognition (OCR) systems for India requires balancing linguistic diversity, document heterogeneity, and deployment constraints. In this paper, we study two training strategies for building multilingual OCR systems with Vision-Language Models through the Chitrapathak series. We first follow a popular multimodal approach, pairing a generic vision encoder with a strong multilingual language model and training the system end-to-end for OCR. Alternatively, we explore fine-tuning an existing OCR model, despite not being trained for the target languages. Through extensive evaluation on multilingual Indic OCR benchmarks and deployment-oriented metrics, we find that the second strategy consistently achieves better accuracy-latency trade-offs. Chitrapathak-2 achieves 3-6x speedup over its predecessor with being state-of-the-art (SOTA) in Telugu (6.69 char ANLS) and second best in the rest. In addition, we present Parichay, an independent OCR model series designed specifically for 9 Indian government documents to extract structured key fields, achieving 89.8% Exact Match score with a faster inference. Together, these systems achieve SOTA performance and provide practical guidance for building production-scale OCR pipelines in the Indian context. 

---
# Automated Histopathology Report Generation via Pyramidal Feature Extraction and the UNI Foundation Model 

**Authors**: Ahmet Halici, Ece Tugba Cebeci, Musa Balci, Mustafa Cini, Serkan Sokmen  

**Link**: [PDF](https://arxiv.org/pdf/2602.16422)  

**Abstract**: Generating diagnostic text from histopathology whole slide images (WSIs) is challenging due to the gigapixel scale of the input and the requirement for precise, domain specific language. We propose a hierarchical vision language framework that combines a frozen pathology foundation model with a Transformer decoder for report generation. To make WSI processing tractable, we perform multi resolution pyramidal patch selection (downsampling factors 2^3 to 2^6) and remove background and artifacts using Laplacian variance and HSV based criteria. Patch features are extracted with the UNI Vision Transformer and projected to a 6 layer Transformer decoder that generates diagnostic text via cross attention. To better represent biomedical terminology, we tokenize the output using BioGPT. Finally, we add a retrieval based verification step that compares generated reports with a reference corpus using Sentence BERT embeddings; if a high similarity match is found, the generated report is replaced with the retrieved ground truth reference to improve reliability. 

---
# AI-Driven Structure Refinement of X-ray Diffraction 

**Authors**: Bin Cao, Qian Zhang, Zhenjie Feng, Taolue Zhang, Jiaqiang Huang, Lu-Tao Weng, Tong-Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.16372)  

**Abstract**: Artificial intelligence can rapidly propose candidate phases and structures from X-ray diffraction (XRD), but these hypotheses often fail in downstream refinement because peak intensities cannot be stably assigned under severe overlap and diffraction consistency is enforced only weakly. Here we introduce WPEM, a physics-constrained whole-pattern decomposition and refinement workflow that turns Bragg's law into an explicit constraint within a batch expectation--maximization framework. WPEM models the full profile as a probabilistic mixture density and iteratively infers component-resolved intensities while keeping peak centres Bragg-consistent, producing a continuous, physically admissible intensity representation that remains stable in heavily overlapped regions and in the presence of mixed radiation or multiple phases. We benchmark WPEM on standard reference patterns (\ce{PbSO4} and \ce{Tb2BaCoO5}), where it yields lower $R_{\mathrm{p}}$/$R_{\mathrm{wp}}$ than widely used packages (FullProf and TOPAS) under matched refinement conditions. We further demonstrate generality across realistic experimental scenarios, including phase-resolved decomposition of a multiphase Ti--15Nb thin film, quantitative recovery of \ce{NaCl}--\ce{Li2CO3} mixture compositions, separation of crystalline peaks from amorphous halos in semicrystalline polymers, high-throughput operando lattice tracking in layered cathodes, automated refinement of a compositionally disordered Ru--Mn oxide solid solution (CCDC 2530452), and quantitative phase-resolved deciphering of an ancient Egyptian make-up sample from synchrotron powder XRD. By providing Bragg-consistent, uncertainty-aware intensity partitioning as a refinement-ready interface, WPEM closes the gap between AI-generated hypotheses and diffraction-admissible structure refinement on challenging XRD data. 

---
# Articulated 3D Scene Graphs for Open-World Mobile Manipulation 

**Authors**: Martin Büchner, Adrian Röfer, Tim Engelbracht, Tim Welschehold, Zuria Bauer, Hermann Blum, Marc Pollefeys, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2602.16356)  

**Abstract**: Semantics has enabled 3D scene understanding and affordance-driven object interaction. However, robots operating in real-world environments face a critical limitation: they cannot anticipate how objects move. Long-horizon mobile manipulation requires closing the gap between semantics, geometry, and kinematics. In this work, we present MoMa-SG, a novel framework for building semantic-kinematic 3D scene graphs of articulated scenes containing a myriad of interactable objects. Given RGB-D sequences containing multiple object articulations, we temporally segment object interactions and infer object motion using occlusion-robust point tracking. We then lift point trajectories into 3D and estimate articulation models using a novel unified twist estimation formulation that robustly estimates revolute and prismatic joint parameters in a single optimization pass. Next, we associate objects with estimated articulations and detect contained objects by reasoning over parent-child relations at identified opening states. We also introduce the novel Arti4D-Semantic dataset, which uniquely combines hierarchical object semantics including parent-child relation labels with object axis annotations across 62 in-the-wild RGB-D sequences containing 600 object interactions and three distinct observation paradigms. We extensively evaluate the performance of MoMa-SG on two datasets and ablate key design choices of our approach. In addition, real-world experiments on both a quadruped and a mobile manipulator demonstrate that our semantic-kinematic scene graphs enable robust manipulation of articulated objects in everyday home environments. We provide code and data at: this https URL. 

---
# HAWX: A Hardware-Aware FrameWork for Fast and Scalable ApproXimation of DNNs 

**Authors**: Samira Nazari, Mohammad Saeed Almasi, Mahdi Taheri, Ali Azarpeyvand, Ali Mokhtari, Ali Mahani, Christian Herglotz  

**Link**: [PDF](https://arxiv.org/pdf/2602.16336)  

**Abstract**: This work presents HAWX, a hardware-aware scalable exploration framework that employs multi-level sensitivity scoring at different DNN abstraction levels (operator, filter, layer, and model) to guide selective integration of heterogeneous AxC blocks. Supported by predictive models for accuracy, power, and area, HAWX accelerates the evaluation of candidate configurations, achieving over 23* speedup in a layer-level search with two candidate approximate blocks and more than (3*106)* speedup at the filter-level search only for LeNet-5, while maintaining accuracy comparable to exhaustive search. Experiments across state-of-the-art DNN benchmarks such as VGG-11, ResNet-18, and EfficientNetLite demonstrate that the efficiency benefits of HAWX scale exponentially with network size. The HAWX hardware-aware search algorithm supports both spatial and temporal accelerator architectures, leveraging either off-the-shelf approximate components or customized designs. 

---
# Spatial Audio Question Answering and Reasoning on Dynamic Source Movements 

**Authors**: Arvind Krishna Sridhar, Yinyi Guo, Erik Visser  

**Link**: [PDF](https://arxiv.org/pdf/2602.16334)  

**Abstract**: Spatial audio understanding aims to enable machines to interpret complex auditory scenes, particularly when sound sources move over time. In this work, we study Spatial Audio Question Answering (Spatial AQA) with a focus on movement reasoning, where a model must infer object motion, position, and directional changes directly from stereo audio. First, we introduce a movement-centric spatial audio augmentation framework that synthesizes diverse motion patterns from isolated mono audio events, enabling controlled and scalable training data generation. Second, we propose an end-to-end multimodal finetuning approach with a thinking mode, which allows audio-language models to produce explicit intermediate reasoning steps before predicting an answer. Third, we investigate the impact of query-conditioned source separation as a preprocessing stage and compare three inference regimes: no masking, an audio grounding model (AGM), and ground-truth masks. Our results show that reasoning amplifies the benefits of source separation, with thinking mode showing significant improvement of +5.1% when a single event is present in the question. These findings highlight the interplay between movement modeling, reasoning, and separation quality, offering new insights for advancing spatial audio understanding. 

---
# Guide-Guard: Off-Target Predicting in CRISPR Applications 

**Authors**: Joseph Bingham, Netanel Arussy, Saman Zonouz  

**Link**: [PDF](https://arxiv.org/pdf/2602.16327)  

**Abstract**: With the introduction of cyber-physical genome sequencing and editing technologies, such as CRISPR, researchers can more easily access tools to investigate and create remedies for a variety of topics in genetics and health science (e.g. agriculture and medicine). As the field advances and grows, new concerns present themselves in the ability to predict the off-target behavior. In this work, we explore the underlying biological and chemical model from a data driven perspective. Additionally, we present a machine learning based solution named \textit{Guide-Guard} to predict the behavior of the system given a gRNA in the CRISPR gene-editing process with 84\% accuracy. This solution is able to be trained on multiple different genes at the same time while retaining accuracy. 

---
# A Self-Supervised Approach for Enhanced Feature Representations in Object Detection Tasks 

**Authors**: Santiago C. Vilabella, Pablo Pérez-Núñez, Beatriz Remeseiro  

**Link**: [PDF](https://arxiv.org/pdf/2602.16322)  

**Abstract**: In the fast-evolving field of artificial intelligence, where models are increasingly growing in complexity and size, the availability of labeled data for training deep learning models has become a significant challenge. Addressing complex problems like object detection demands considerable time and resources for data labeling to achieve meaningful results. For companies developing such applications, this entails extensive investment in highly skilled personnel or costly outsourcing. This research work aims to demonstrate that enhancing feature extractors can substantially alleviate this challenge, enabling models to learn more effective representations with less labeled data. Utilizing a self-supervised learning strategy, we present a model trained on unlabeled data that outperforms state-of-the-art feature extractors pre-trained on ImageNet and particularly designed for object detection tasks. Moreover, the results demonstrate that our approach encourages the model to focus on the most relevant aspects of an object, thus achieving better feature representations and, therefore, reinforcing its reliability and robustness. 

---
# A Graph Meta-Network for Learning on Kolmogorov-Arnold Networks 

**Authors**: Guy Bar-Shalom, Ami Tavory, Itay Evron, Maya Bechler-Speicher, Ido Guy, Haggai Maron  

**Link**: [PDF](https://arxiv.org/pdf/2602.16316)  

**Abstract**: Weight-space models learn directly from the parameters of neural networks, enabling tasks such as predicting their accuracy on new datasets. Naive methods -- like applying MLPs to flattened parameters -- perform poorly, making the design of better weight-space architectures a central challenge. While prior work leveraged permutation symmetries in standard networks to guide such designs, no analogous analysis or tailored architecture yet exists for Kolmogorov-Arnold Networks (KANs). In this work, we show that KANs share the same permutation symmetries as MLPs, and propose the KAN-graph, a graph representation of their computation. Building on this, we develop WS-KAN, the first weight-space architecture that learns on KANs, which naturally accounts for their symmetry. We analyze WS-KAN's expressive power, showing it can replicate an input KAN's forward pass - a standard approach for assessing expressiveness in weight-space architectures. We construct a comprehensive ``zoo'' of trained KANs spanning diverse tasks, which we use as benchmarks to empirically evaluate WS-KAN. Across all tasks, WS-KAN consistently outperforms structure-agnostic baselines, often by a substantial margin. Our code is available at this https URL. 

---
# The Diversity Paradox revisited: Systemic Effects of Feedback Loops in Recommender Systems 

**Authors**: Gabriele Barlacchi, Margherita Lalli, Emanuele Ferragina, Fosca Giannotti, Dino Pedreschi, Luca Pappalardo  

**Link**: [PDF](https://arxiv.org/pdf/2602.16315)  

**Abstract**: Recommender systems shape individual choices through feedback loops in which user behavior and algorithmic recommendations coevolve over time. The systemic effects of these loops remain poorly understood, in part due to unrealistic assumptions in existing simulation studies. We propose a feedback-loop model that captures implicit feedback, periodic retraining, probabilistic adoption of recommendations, and heterogeneous recommender systems. We apply the framework on online retail and music streaming data and analyze systemic effects of the feedback loop. We find that increasing recommender adoption may lead to a progressive diversification of individual consumption, while collective demand is redistributed in model- and domain-dependent ways, often amplifying popularity concentration. Temporal analyses further reveal that apparent increases in individual diversity observed in static evaluations are illusory: when adoption is fixed and time unfolds, individual diversity consistently decreases across all models. Our results highlight the need to move beyond static evaluations and explicitly account for feedback-loop dynamics when designing recommender systems. 

---
# The Weight of a Bit: EMFI Sensitivity Analysis of Embedded Deep Learning Models 

**Authors**: Jakub Breier, Štefan Kučerák, Xiaolu Hou  

**Link**: [PDF](https://arxiv.org/pdf/2602.16309)  

**Abstract**: Fault injection attacks on embedded neural network models have been shown as a potent threat. Numerous works studied resilience of models from various points of view. As of now, there is no comprehensive study that would evaluate the influence of number representations used for model parameters against electromagnetic fault injection (EMFI) attacks.
In this paper, we investigate how four different number representations influence the success of an EMFI attack on embedded neural network models. We chose two common floating-point representations (32-bit, and 16-bit), and two integer representations (8-bit, and 4-bit). We deployed four common image classifiers, ResNet-18, ResNet-34, ResNet-50, and VGG-11, on an embedded memory chip, and utilized a low-cost EMFI platform to trigger faults. Our results show that while floating-point representations exhibit almost a complete degradation in accuracy (Top-1 and Top-5) after a single fault injection, integer representations offer better resistance overall. Especially, when considering the the 8-bit representation on a relatively large network (VGG-11), the Top-1 accuracies stay at around 70% and the Top-5 at around 90%. 

---
# Generative AI Usage of University Students: Navigating Between Education and Business 

**Authors**: Fabian Walke, Veronika Föller  

**Link**: [PDF](https://arxiv.org/pdf/2602.16307)  

**Abstract**: This study investigates generative artificial intelligence (GenAI) usage of university students who study alongside their professional career. Previous literature has paid little attention to part-time students and the intersectional use of GenAI between education and business. This study examines with a grounded theory approach the characteristics of GenAI usage of part-time students. Eleven students from a distance learning university were interviewed. Three causal and four intervening conditions, as well as strategies were identified, to influence the use of GenAI. The study highlights both the potential and challenges of GenAI usage in education and business. While GenAI can significantly enhance productivity and learning outcomes, concerns about ethical implications, reliability, and the risk of academic misconduct persist. The developed grounded model offers a comprehensive understanding of GenAI usage among students, providing valuable insights for educators, policymakers, and developers of GenAI tools seeking to bridge the gap between education and business. 

---
# Color-based Emotion Representation for Speech Emotion Recognition 

**Authors**: Ryotaro Nagase, Ryoichi Takashima, Yoichi Yamashita  

**Link**: [PDF](https://arxiv.org/pdf/2602.16256)  

**Abstract**: Speech emotion recognition (SER) has traditionally relied on categorical or dimensional labels. However, this technique is limited in representing both the diversity and interpretability of emotions. To overcome this limitation, we focus on color attributes, such as hue, saturation, and value, to represent emotions as continuous and interpretable scores. We annotated an emotional speech corpus with color attributes via crowdsourcing and analyzed them. Moreover, we built regression models for color attributes in SER using machine learning and deep learning, and explored the multitask learning of color attribute regression and emotion classification. As a result, we demonstrated the relationship between color attributes and emotions in speech, and successfully developed color attribute regression models for SER. We also showed that multitask learning improved the performance of each task. 

---
# Are LLMs Ready to Replace Bangla Annotators? 

**Authors**: Md. Najib Hasan, Touseef Hasan, Souvika Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2602.16241)  

**Abstract**: Large Language Models (LLMs) are increasingly used as automated annotators to scale dataset creation, yet their reliability as unbiased annotators--especially for low-resource and identity-sensitive settings--remains poorly understood. In this work, we study the behavior of LLMs as zero-shot annotators for Bangla hate speech, a task where even human agreement is challenging, and annotator bias can have serious downstream consequences. We conduct a systematic benchmark of 17 LLMs using a unified evaluation framework. Our analysis uncovers annotator bias and substantial instability in model judgments. Surprisingly, increased model scale does not guarantee improved annotation quality--smaller, more task-aligned models frequently exhibit more consistent behavior than their larger counterparts. These results highlight important limitations of current LLMs for sensitive annotation tasks in low-resource languages and underscore the need for careful evaluation before deployment. 

---
# UCTECG-Net: Uncertainty-aware Convolution Transformer ECG Network for Arrhythmia Detection 

**Authors**: Hamzeh Asgharnezhad, Pegah Tabarisaadi, Abbas Khosravi, Roohallah Alizadehsani, U. Rajendra Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2602.16216)  

**Abstract**: Deep learning has improved automated electrocardiogram (ECG) classification, but limited insight into prediction reliability hinders its use in safety-critical settings. This paper proposes UCTECG-Net, an uncertainty-aware hybrid architecture that combines one-dimensional convolutions and Transformer encoders to process raw ECG signals and their spectrograms jointly. Evaluated on the MIT-BIH Arrhythmia and PTB Diagnostic datasets, UCTECG-Net outperforms LSTM, CNN1D, and Transformer baselines in terms of accuracy, precision, recall and F1 score, achieving up to 98.58% accuracy on MIT-BIH and 99.14% on PTB. To assess predictive reliability, we integrate three uncertainty quantification methods (Monte Carlo Dropout, Deep Ensembles, and Ensemble Monte Carlo Dropout) into all models and analyze their behavior using an uncertainty-aware confusion matrix and derived metrics. The results show that UCTECG-Net, particularly with Ensemble or EMCD, provides more reliable and better-aligned uncertainty estimates than competing architectures, offering a stronger basis for risk-aware ECG decision support. 

---
# Graph neural network for colliding particles with an application to sea ice floe modeling 

**Authors**: Ruibiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.16213)  

**Abstract**: This paper introduces a novel approach to sea ice modeling using Graph Neural Networks (GNNs), utilizing the natural graph structure of sea ice, where nodes represent individual ice pieces, and edges model the physical interactions, including collisions. This concept is developed within a one-dimensional framework as a foundational step. Traditional numerical methods, while effective, are computationally intensive and less scalable. By utilizing GNNs, the proposed model, termed the Collision-captured Network (CN), integrates data assimilation (DA) techniques to effectively learn and predict sea ice dynamics under various conditions. The approach was validated using synthetic data, both with and without observed data points, and it was found that the model accelerates the simulation of trajectories without compromising accuracy. This advancement offers a more efficient tool for forecasting in marginal ice zones (MIZ) and highlights the potential of combining machine learning with data assimilation for more effective and efficient modeling. 

---
# Geometric Neural Operators via Lie Group-Constrained Latent Dynamics 

**Authors**: Jiaquan Zhang, Fachrina Dewi Puspitasari, Songbo Zhang, Yibei Liu, Kuien Liu, Caiyan Qin, Fan Mo, Peng Wang, Yang Yang, Chaoning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.16209)  

**Abstract**: Neural operators offer an effective framework for learning solutions of partial differential equations for many physical systems in a resolution-invariant and data-driven manner. Existing neural operators, however, often suffer from instability in multi-layer iteration and long-horizon rollout, which stems from the unconstrained Euclidean latent space updates that violate the geometric and conservation laws. To address this challenge, we propose to constrain manifolds with low-rank Lie algebra parameterization that performs group action updates on the latent representation. Our method, termed Manifold Constraining based on Lie group (MCL), acts as an efficient \emph{plug-and-play} module that enforces geometric inductive bias to existing neural operators. Extensive experiments on various partial differential equations, such as 1-D Burgers and 2-D Navier-Stokes, over a wide range of parameters and steps demonstrate that our method effectively lowers the relative prediction error by 30-50\% at the cost of 2.26\% of parameter increase. The results show that our approach provides a scalable solution for improving long-term prediction fidelity by addressing the principled geometric constraints absent in the neural operator updates. 

---
# Long-Tail Knowledge in Large Language Models: Taxonomy, Mechanisms, Interventions and Implications 

**Authors**: Sanket Badhe, Deep Shah, Nehal Kathrotia  

**Link**: [PDF](https://arxiv.org/pdf/2602.16201)  

**Abstract**: Large language models (LLMs) are trained on web-scale corpora that exhibit steep power-law distributions, in which the distribution of knowledge is highly long-tailed, with most appearing infrequently. While scaling has improved average-case performance, persistent failures on low-frequency, domain-specific, cultural, and temporal knowledge remain poorly characterized. This paper develops a structured taxonomy and analysis of long-Tail Knowledge in large language models, synthesizing prior work across technical and sociotechnical perspectives.
We introduce a structured analytical framework that synthesizes prior work across four complementary axes: how long-Tail Knowledge is defined, the mechanisms by which it is lost or distorted during training and inference, the technical interventions proposed to mitigate these failures, and the implications of these failures for fairness, accountability, transparency, and user trust. We further examine how existing evaluation practices obscure tail behavior and complicate accountability for rare but consequential failures. The paper concludes by identifying open challenges related to privacy, sustainability, and governance that constrain long-Tail Knowledge representation. Taken together, this paper provides a unifying conceptual framework for understanding how long-Tail Knowledge is defined, lost, evaluated, and manifested in deployed language model systems. 

---
# Graphon Mean-Field Subsampling for Cooperative Heterogeneous Multi-Agent Reinforcement Learning 

**Authors**: Emile Anand, Richard Hoffmann, Sarah Liaw, Adam Wierman  

**Link**: [PDF](https://arxiv.org/pdf/2602.16196)  

**Abstract**: Coordinating large populations of interacting agents is a central challenge in multi-agent reinforcement learning (MARL), where the size of the joint state-action space scales exponentially with the number of agents. Mean-field methods alleviate this burden by aggregating agent interactions, but these approaches assume homogeneous interactions. Recent graphon-based frameworks capture heterogeneity, but are computationally expensive as the number of agents grows. Therefore, we introduce $\texttt{GMFS}$, a $\textbf{G}$raphon $\textbf{M}$ean-$\textbf{F}$ield $\textbf{S}$ubsampling framework for scalable cooperative MARL with heterogeneous agent interactions. By subsampling $\kappa$ agents according to interaction strength, we approximate the graphon-weighted mean-field and learn a policy with sample complexity $\mathrm{poly}(\kappa)$ and optimality gap $O(1/\sqrt{\kappa})$. We verify our theory with numerical simulations in robotic coordination, showing that $\texttt{GMFS}$ achieves near-optimal performance. 

---
# Temporal Panel Selection in Ongoing Citizens' Assemblies 

**Authors**: Yusuf Hakan Kalayci, Evi Micha  

**Link**: [PDF](https://arxiv.org/pdf/2602.16194)  

**Abstract**: Permanent citizens' assemblies are ongoing deliberative bodies composed of randomly selected citizens, organized into panels that rotate over time. Unlike one-off panels, which represent the population in a single snapshot, permanent assemblies enable shifting participation across multiple rounds. This structure offers a powerful framework for ensuring that different groups of individuals are represented over time across successive panels. In particular, it allows smaller groups of individuals that may not warrant representation in every individual panel to be represented across a sequence of them. We formalize this temporal sortition framework by requiring proportional representation both within each individual panel and across the sequence of panels.
Building on the work of Ebadian and Micha (2025), we consider a setting in which the population lies in a metric space, and the goal is to achieve both proportional representation, ensuring that every group of citizens receives adequate representation, and individual fairness, ensuring that each individual has an equal probability of being selected. We extend the notion of representation to a temporal setting by requiring that every initial segment of the panel sequence, viewed as a cumulative whole, proportionally reflects the structure of the population. We present algorithms that provide varying guarantees of proportional representation, both within individual panels and across any sequence of panels, while also maintaining individual fairness over time. 

---
# Rethinking Input Domains in Physics-Informed Neural Networks via Geometric Compactification Mappings 

**Authors**: Zhenzhen Huang, Haoyu Bian, Jiaquan Zhang, Yibei Liu, Kuien Liu, Caiyan Qin, Guoqing Wang, Yang Yang, Chaoning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.16193)  

**Abstract**: Several complex physical systems are governed by multi-scale partial differential equations (PDEs) that exhibit both smooth low-frequency components and localized high-frequency structures. Existing physics-informed neural network (PINN) methods typically train with fixed coordinate system inputs, where geometric misalignment with these structures induces gradient stiffness and ill-conditioning that hinder convergence. To address this issue, we introduce a mapping paradigm that reshapes the input coordinates through differentiable geometric compactification mappings and couples the geometric structure of PDEs with the spectral properties of residual operators. Based on this paradigm, we propose Geometric Compactification (GC)-PINN, a framework that introduces three mapping strategies for periodic boundaries, far-field scale expansion, and localized singular structures in the input domain without modifying the underlying PINN architecture. Extensive empirical evaluation demonstrates that this approach yields more uniform residual distributions and higher solution accuracy on representative 1D and 2D PDEs, while improving training stability and convergence speed. 

---
# Beyond Learning: A Training-Free Alternative to Model Adaptation 

**Authors**: Namkyung Yoon, Kyeonghyun Yoo, Wooyong Jung, Sanghong Kim, Hwangnam Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.16189)  

**Abstract**: Despite the continuous research and evolution of language models, they sometimes underperform previous versions. Existing approaches to overcome these challenges are resource-intensive, highlighting the need for alternatives that enable immediate action. We assume that each language model has a local module inside that is suitable for a specific function. First, this work identifies a set of modules showing consistent and local activation changes under an inference workload through activation-based analysis. Subsequently, we transplant an internal module that is properly activated for a specific task into the target model, leading to immediate and measurable functional changes without additional training or fine-tuning. To experimentally demonstrate the effectiveness of the transplant technique, we quantify the relationship between transplant strength and performance improvement under different conditions for two language models. In the cross-generation setting, we find that transplanting activation-selected modules can substantially improve the underperforming model, reaching up to twice the target baseline and achieving gap-based recovery above 100%. Moreover, in transplant experiments between a base model and its instruction-tuned counterpart, transplantation improves the underperforming model toward the stronger baseline, yielding up to about 2.33 times the target baseline with gap-based recovery reaching up to 100% in the best case. These results show that meaningful capacity transfer can be realized through the implantation of highly localized modules implied by language models. Overall, this work provides empirical evidence for task-localized modularity in language models and presents a new research area: model transplantation. 

---
# SIT-LMPC: Safe Information-Theoretic Learning Model Predictive Control for Iterative Tasks 

**Authors**: Zirui Zang, Ahmad Amine, Nick-Marios T. Kokolakis, Truong X. Nghiem, Ugo Rosolia, Rahul Mangharam  

**Link**: [PDF](https://arxiv.org/pdf/2602.16187)  

**Abstract**: Robots executing iterative tasks in complex, uncertain environments require control strategies that balance robustness, safety, and high performance. This paper introduces a safe information-theoretic learning model predictive control (SIT-LMPC) algorithm for iterative tasks. Specifically, we design an iterative control framework based on an information-theoretic model predictive control algorithm to address a constrained infinite-horizon optimal control problem for discrete-time nonlinear stochastic systems. An adaptive penalty method is developed to ensure safety while balancing optimality. Trajectories from previous iterations are utilized to learn a value function using normalizing flows, which enables richer uncertainty modeling compared to Gaussian priors. SIT-LMPC is designed for highly parallel execution on graphics processing units, allowing efficient real-time optimization. Benchmark simulations and hardware experiments demonstrate that SIT-LMPC iteratively improves system performance while robustly satisfying system constraints. 

---
# Conjugate Learning Theory: Uncovering the Mechanisms of Trainability and Generalization in Deep Neural Networks 

**Authors**: Binchuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2602.16177)  

**Abstract**: In this work, we propose a notion of practical learnability grounded in finite sample settings, and develop a conjugate learning theoretical framework based on convex conjugate duality to characterize this learnability property. Building on this foundation, we demonstrate that training deep neural networks (DNNs) with mini-batch stochastic gradient descent (SGD) achieves global optima of empirical risk by jointly controlling the extreme eigenvalues of a structure matrix and the gradient energy, and we establish a corresponding convergence theorem. We further elucidate the impact of batch size and model architecture (including depth, parameter count, sparsity, skip connections, and other characteristics) on non-convex optimization. Additionally, we derive a model-agnostic lower bound for the achievable empirical risk, theoretically demonstrating that data determines the fundamental limit of trainability. On the generalization front, we derive deterministic and probabilistic bounds on generalization error based on generalized conditional entropy measures. The former explicitly delineates the range of generalization error, while the latter characterizes the distribution of generalization error relative to the deterministic bounds under independent and identically distributed (i.i.d.) sampling conditions. Furthermore, these bounds explicitly quantify the influence of three key factors: (i) information loss induced by irreversibility in the model, (ii) the maximum attainable loss value, and (iii) the generalized conditional entropy of features with respect to labels. Moreover, they offer a unified theoretical lens for understanding the roles of regularization, irreversible transformations, and network depth in shaping the generalization behavior of deep neural networks. Extensive experiments validate all theoretical predictions, confirming the framework's correctness and consistency. 

---
# Edge Learning via Federated Split Decision Transformers for Metaverse Resource Allocation 

**Authors**: Fatih Temiz, Shavbo Salehi, Melike Erol-Kantarci  

**Link**: [PDF](https://arxiv.org/pdf/2602.16174)  

**Abstract**: Mobile edge computing (MEC) based wireless metaverse services offer an untethered, immersive experience to users, where the superior quality of experience (QoE) needs to be achieved under stringent latency constraints and visual quality demands. To achieve this, MEC-based intelligent resource allocation for virtual reality users needs to be supported by coordination across MEC servers to harness distributed data. Federated learning (FL) is a promising solution, and can be combined with reinforcement learning (RL) to develop generalized policies across MEC-servers. However, conventional FL incurs transmitting the full model parameters across the MEC-servers and the cloud, and suffer performance degradation due to naive global aggregation, especially in heterogeneous multi-radio access technology environments. To address these challenges, this paper proposes Federated Split Decision Transformer (FSDT), an offline RL framework where the transformer model is partitioned between MEC servers and the cloud. Agent-specific components (e.g., MEC-based embedding and prediction layers) enable local adaptability, while shared global layers in the cloud facilitate cooperative training across MEC servers. Experimental results demonstrate that FSDT enhances QoE for up to 10% in heterogeneous environments compared to baselines, while offloadingnearly 98% of the transformer model parameters to the cloud, thereby reducing the computational burden on MEC servers. 

---
# HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents 

**Authors**: Jiangweizhi Peng, Yuanxin Liu, Ruida Zhou, Charles Fleming, Zhaoran Wang, Alfredo Garcia, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2602.16165)  

**Abstract**: Training LLMs as interactive agents for multi-turn decision-making remains challenging, particularly in long-horizon tasks with sparse and delayed rewards, where agents must execute extended sequences of actions before receiving meaningful feedback. Most existing reinforcement learning (RL) approaches model LLM agents as flat policies operating at a single time scale, selecting one action at each turn. In sparse-reward settings, such flat policies must propagate credit across the entire trajectory without explicit temporal abstraction, which often leads to unstable optimization and inefficient credit assignment.
We propose HiPER, a novel Hierarchical Plan-Execute RL framework that explicitly separates high-level planning from low-level execution. HiPER factorizes the policy into a high-level planner that proposes subgoals and a low-level executor that carries them out over multiple action steps. To align optimization with this structure, we introduce a key technique called hierarchical advantage estimation (HAE), which carefully assigns credit at both the planning and execution levels. By aggregating returns over the execution of each subgoal and coordinating updates across the two levels, HAE provides an unbiased gradient estimator and provably reduces variance compared to flat generalized advantage estimation.
Empirically, HiPER achieves state-of-the-art performance on challenging interactive benchmarks, reaching 97.4\% success on ALFWorld and 83.3\% on WebShop with Qwen2.5-7B-Instruct (+6.6\% and +8.3\% over the best prior method), with especially large gains on long-horizon tasks requiring multiple dependent subtasks. These results highlight the importance of explicit hierarchical decomposition for scalable RL training of multi-turn LLM agents. 

---
# Balancing Faithfulness and Performance in Reasoning via Multi-Listener Soft Execution 

**Authors**: Nithin Sivakumaran, Shoubin Yu, Hyunji Lee, Yue Zhang, Ali Payani, Mohit Bansal, Elias Stengel-Eskin  

**Link**: [PDF](https://arxiv.org/pdf/2602.16154)  

**Abstract**: Chain-of-thought (CoT) reasoning sometimes fails to faithfully reflect the true computation of a large language model (LLM), hampering its utility in explaining how LLMs arrive at their answers. Moreover, optimizing for faithfulness and interpretability in reasoning often degrades task performance. To address this tradeoff and improve CoT faithfulness, we propose Reasoning Execution by Multiple Listeners (REMUL), a multi-party reinforcement learning approach. REMUL builds on the hypothesis that reasoning traces which other parties can follow will be more faithful. A speaker model generates a reasoning trace, which is truncated and passed to a pool of listener models who "execute" the trace, continuing the trace to an answer. Speakers are rewarded for producing reasoning that is clear to listeners, with additional correctness regularization via masked supervised finetuning to counter the tradeoff between faithfulness and performance. On multiple reasoning benchmarks (BIG-Bench Extra Hard, MuSR, ZebraLogicBench, and FOLIO), REMUL consistently and substantially improves three measures of faithfulness -- hint attribution, early answering area over the curve (AOC), and mistake injection AOC -- while also improving accuracy. Our analysis finds that these gains are robust across training domains, translate to legibility gains, and are associated with shorter and more direct CoTs. 

---
# ASPEN: Spectral-Temporal Fusion for Cross-Subject Brain Decoding 

**Authors**: Megan Lee, Seung Ha Hwang, Inhyeok Choi, Shreyas Darade, Mengchun Zhang, Kateryna Shapovalenko  

**Link**: [PDF](https://arxiv.org/pdf/2602.16147)  

**Abstract**: Cross-subject generalization in EEG-based brain-computer interfaces (BCIs) remains challenging due to individual variability in neural signals. We investigate whether spectral representations offer more stable features for cross-subject transfer than temporal waveforms. Through correlation analyses across three EEG paradigms (SSVEP, P300, and Motor Imagery), we find that spectral features exhibit consistently higher cross-subject similarity than temporal signals. Motivated by this observation, we introduce ASPEN, a hybrid architecture that combines spectral and temporal feature streams via multiplicative fusion, requiring cross-modal agreement for features to propagate. Experiments across six benchmark datasets reveal that ASPEN is able to dynamically achieve the optimal spectral-temporal balance depending on the paradigm. ASPEN achieves the best unseen-subject accuracy on three of six datasets and competitive performance on others, demonstrating that multiplicative multimodal fusion enables effective cross-subject generalization. 

---
# Human-AI Collaboration in Large Language Model-Integrated Building Energy Management Systems: The Role of User Domain Knowledge and AI Literacy 

**Authors**: Wooyoung Jung, Kahyun Jeon, Prosper Babon-Ayeng  

**Link**: [PDF](https://arxiv.org/pdf/2602.16140)  

**Abstract**: This study aimed to comprehend how user domain knowledge and artificial intelligence (AI) literacy impact the effective use of human-AI interactive building energy management system (BEMS). While prior studies have investigated the potential of integrating large language models (LLMs) into BEMS or building energy modeling, very few studies have examined how user interact with such systems. We conducted a systematic role-playing experiment, where 85 human subjects interacted with an advanced generative pre-trained transformer (OpenAI GPT-4o). Participants were tasked with identifying the top five behavioral changes that could reduce home energy use with the GPT model that functioned as an LLM-integrated BEMS. Then, the collected prompt-response data and participant conclusions were analyzed using an analytical framework that hierarchically assessed and scored human-AI interactions and their home energy analysis approaches. Also, participants were classified into four groups based on their self-evaluated domain knowledge of building energy use and AI literacy, and Kruskal-Wallis H tests with post-hoc pairwise comparisons were conducted across 20 quantifiable metrics. Key takeaways include: most participants employed concise prompts (median: 16.2 words) and relied heavily on GPT's analytical capabilities; and notably, only 1 of 20 metrics, appliance identification rate, showed statistically significant group differences (p=0.037), driven by AI literacy rather than domain knowledge, suggesting an equalizing effect of LLMs across expertise levels. This study provides foundational insights into human-AI collaboration dynamics and promising development directions in the context of LLM-integrated BEMS and contributes to realizing human-centric LLM-integrated energy systems. 

---
# Retrieval Collapses When AI Pollutes the Web 

**Authors**: Hongyeon Yu, Dongchan Kim, Young-Bum Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.16136)  

**Abstract**: The rapid proliferation of AI-generated content on the Web presents a structural risk to information retrieval, as search engines and Retrieval-Augmented Generation (RAG) systems increasingly consume evidence produced by the Large Language Models (LLMs). We characterize this ecosystem-level failure mode as Retrieval Collapse, a two-stage process where (1) AI-generated content dominates search results, eroding source diversity, and (2) low-quality or adversarial content infiltrates the retrieval pipeline. We analyzed this dynamic through controlled experiments involving both high-quality SEO-style content and adversarially crafted content. In the SEO scenario, a 67\% pool contamination led to over 80\% exposure contamination, creating a homogenized yet deceptively healthy state where answer accuracy remains stable despite the reliance on synthetic sources. Conversely, under adversarial contamination, baselines like BM25 exposed $\sim$19\% of harmful content, whereas LLM-based rankers demonstrated stronger suppression capabilities. These findings highlight the risk of retrieval pipelines quietly shifting toward synthetic evidence and the need for retrieval-aware strategies to prevent a self-reinforcing cycle of quality decline in Web-grounded systems. 

---
# Rethinking ANN-based Retrieval: Multifaceted Learnable Index for Large-scale Recommendation System 

**Authors**: Jiang Zhang, Yubo Wang, Wei Chang, Lu Han, Xingying Cheng, Feng Zhang, Min Li, Songhao Jiang, Wei Zheng, Harry Tran, Zhen Wang, Lei Chen, Yueming Wang, Benyu Zhang, Xiangjun Fan, Bi Xue, Qifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.16124)  

**Abstract**: Approximate nearest neighbor (ANN) search is widely used in the retrieval stage of large-scale recommendation systems. In this stage, candidate items are indexed using their learned embedding vectors, and ANN search is executed for each user (or item) query to retrieve a set of relevant items. However, ANN-based retrieval has two key limitations. First, item embeddings and their indices are typically learned in separate stages: indexing is often performed offline after embeddings are trained, which can yield suboptimal retrieval quality-especially for newly created items. Second, although ANN offers sublinear query time, it must still be run for every request, incurring substantial computation cost at industry scale. In this paper, we propose MultiFaceted Learnable Index (MFLI), a scalable, real-time retrieval paradigm that learns multifaceted item embeddings and indices within a unified framework and eliminates ANN search at serving time. Specifically, we construct a multifaceted hierarchical codebook via residual quantization of item embeddings and co-train the codebook with the embeddings. We further introduce an efficient multifaceted indexing structure and mechanisms that support real-time updates. At serving time, the learned hierarchical indices are used directly to identify relevant items, avoiding ANN search altogether. Extensive experiments on real-world data with billions of users show that MFLI improves recall on engagement tasks by up to 11.8\%, cold-content delivery by up to 57.29\%, and semantic relevance by 13.5\% compared with prior state-of-the-art methods. We also deploy MFLI in the system and report online experimental results demonstrating improved engagement, less popularity bias, and higher serving efficiency. 

---
# Surrogate-Based Prevalence Measurement for Large-Scale A/B Testing 

**Authors**: Zehao Xu, Tony Paek, Kevin O'Sullivan, Attila Dobi  

**Link**: [PDF](https://arxiv.org/pdf/2602.16111)  

**Abstract**: Online media platforms often need to measure how frequently users are exposed to specific content attributes in order to evaluate trade-offs in A/B experiments. A direct approach is to sample content, label it using a high-quality rubric (e.g., an expert-reviewed LLM prompt), and estimate impression-weighted prevalence. However, repeatedly running such labeling for every experiment arm and segment is too costly and slow to serve as a default measurement at scale.
We present a scalable \emph{surrogate-based prevalence measurement} framework that decouples expensive labeling from per-experiment evaluation. The framework calibrates a surrogate signal to reference labels offline and then uses only impression logs to estimate prevalence for arbitrary experiment arms and segments. We instantiate this framework using \emph{score bucketing} as the surrogate: we discretize a model score into buckets, estimate bucket-level prevalences from an offline labeled sample, and combine these calibrated bucket level prevalences with the bucket distribution of impressions in each arm to obtain fast, log-based estimates.
Across multiple large-scale A/B tests, we validate that the surrogate estimates closely match the reference estimates for both arm-level prevalence and treatment--control deltas. This enables scalable, low-latency prevalence measurement in experimentation without requiring per-experiment labeling jobs. 

---
# OmniCT: Towards a Unified Slice-Volume LVLM for Comprehensive CT Analysis 

**Authors**: Tianwei Lin, Zhongwei Qiu, Wenqiao Zhang, Jiang Liu, Yihan Xie, Mingjian Gao, Zhenxuan Fan, Zhaocheng Li, Sijing Li, Zhongle Xie, Peng LU, Yueting Zhuang, Yingda Xia, Ling Zhang, Beng Chin Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2602.16110)  

**Abstract**: Computed Tomography (CT) is one of the most widely used and diagnostically information-dense imaging modalities, covering critical organs such as the heart, lungs, liver, and colon. Clinical interpretation relies on both slice-driven local features (e.g., sub-centimeter nodules, lesion boundaries) and volume-driven spatial representations (e.g., tumor infiltration, inter-organ anatomical relations). However, existing Large Vision-Language Models (LVLMs) remain fragmented in CT slice versus volumetric understanding: slice-driven LVLMs show strong generalization but lack cross-slice spatial consistency, while volume-driven LVLMs explicitly capture volumetric semantics but suffer from coarse granularity and poor compatibility with slice inputs. The absence of a unified modeling paradigm constitutes a major bottleneck for the clinical translation of medical LVLMs. We present OmniCT, a powerful unified slice-volume LVLM for CT scenarios, which makes three contributions: (i) Spatial Consistency Enhancement (SCE): volumetric slice composition combined with tri-axial positional embedding that introduces volumetric consistency, and an MoE hybrid projection enables efficient slice-volume adaptation; (ii) Organ-level Semantic Enhancement (OSE): segmentation and ROI localization explicitly align anatomical regions, emphasizing lesion- and organ-level semantics; (iii) MedEval-CT: the largest slice-volume CT dataset and hybrid benchmark integrates comprehensive metrics for unified evaluation. OmniCT consistently outperforms existing methods with a substantial margin across diverse clinical tasks and satisfies both micro-level detail sensitivity and macro-level spatial reasoning. More importantly, it establishes a new paradigm for cross-modal medical imaging understanding. 

---
# Federated Graph AGI for Cross-Border Insider Threat Intelligence in Government Financial Schemes 

**Authors**: Srikumar Nayak, James Walmesley  

**Link**: [PDF](https://arxiv.org/pdf/2602.16109)  

**Abstract**: Cross-border insider threats pose a critical challenge to government financial schemes, particularly when dealing with distributed, privacy-sensitive data across multiple jurisdictions. Existing approaches face fundamental limitations: they cannot effectively share intelligence across borders due to privacy constraints, lack reasoning capabilities to understand complex multi-step attack patterns, and fail to capture intricate graph-structured relationships in financial networks. We introduce FedGraph-AGI, a novel federated learning framework integrating Artificial General Intelligence (AGI) reasoning with graph neural networks for privacy-preserving cross-border insider threat detection. Our approach combines: (1) federated graph neural networks preserving data sovereignty; (2) Mixture-of-Experts (MoE) aggregation for heterogeneous jurisdictions; and (3) AGI-powered reasoning via Large Action Models (LAM) performing causal inference over graph data. Through experiments on a 50,000-transaction dataset across 10 jurisdictions, FedGraph-AGI achieves 92.3% accuracy, significantly outperforming federated baselines (86.1%) and centralized approaches (84.7%). Our ablation studies reveal AGI reasoning contributes 6.8% improvement, while MoE adds 4.4%. The system maintains epsilon = 1.0 differential privacy while achieving near-optimal performance and scales efficiently to 50+ clients. This represents the first integration of AGI reasoning with federated graph learning for insider threat detection, opening new directions for privacy-preserving cross-border intelligence sharing. 

---
# Updating Parametric Knowledge with Context Distillation Retains Post-Training Capabilities 

**Authors**: Shankar Padmanabhan, Mustafa Omer Gul, Tanya Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2602.16093)  

**Abstract**: Post-training endows pretrained LLMs with a variety of desirable skills, including instruction-following, reasoning, and others. However, these post-trained LLMs only encode knowledge up to a cut-off date, necessitating continual adaptation. Unfortunately, existing solutions cannot simultaneously learn new knowledge from an adaptation document corpora and mitigate the forgetting of earlier learned capabilities. To address this, we introduce Distillation via Split Contexts (DiSC), a simple context-distillation based approach for continual knowledge adaptation. \methodname~derives student and teacher distributions by conditioning on distinct segments of the training example and minimizes the KL divergence between the shared tokens. This allows us to efficiently apply context-distillation without requiring explicit generation steps during training. We run experiments on four post-trained models and two adaptation domains. Compared to prior finetuning and distillation methods for continual adaptation, DiSC consistently reports the best trade-off between learning new knowledge and mitigating forgetting of previously learned skills like instruction-following, reasoning, and factual knowledge. 

---
# Language Statistics and False Belief Reasoning: Evidence from 41 Open-Weight LMs 

**Authors**: Sean Trott, Samuel Taylor, Cameron Jones, James A. Michaelov, Pamela D. Rivière  

**Link**: [PDF](https://arxiv.org/pdf/2602.16085)  

**Abstract**: Research on mental state reasoning in language models (LMs) has the potential to inform theories of human social cognition--such as the theory that mental state reasoning emerges in part from language exposure--and our understanding of LMs themselves. Yet much published work on LMs relies on a relatively small sample of closed-source LMs, limiting our ability to rigorously test psychological theories and evaluate LM capacities. Here, we replicate and extend published work on the false belief task by assessing LM mental state reasoning behavior across 41 open-weight models (from distinct model families). We find sensitivity to implied knowledge states in 34% of the LMs tested; however, consistent with prior work, none fully ``explain away'' the effect in humans. Larger LMs show increased sensitivity and also exhibit higher psychometric predictive power. Finally, we use LM behavior to generate and test a novel hypothesis about human cognition: both humans and LMs show a bias towards attributing false beliefs when knowledge states are cued using a non-factive verb (``John thinks...'') than when cued indirectly (``John looks in the...''). Unlike the primary effect of knowledge states, where human sensitivity exceeds that of LMs, the magnitude of the human knowledge cue effect falls squarely within the distribution of LM effect sizes-suggesting that distributional statistics of language can in principle account for the latter but not the former in humans. These results demonstrate the value of using larger samples of open-weight LMs to test theories of human cognition and evaluate LM capacities. 

---
# ScenicRules: An Autonomous Driving Benchmark with Multi-Objective Specifications and Abstract Scenarios 

**Authors**: Kevin Kai-Chun Chang, Ekin Beyazit, Alberto Sangiovanni-Vincentelli, Tichakorn Wongpiromsarn, Sanjit A. Seshia  

**Link**: [PDF](https://arxiv.org/pdf/2602.16073)  

**Abstract**: Developing autonomous driving systems for complex traffic environments requires balancing multiple objectives, such as avoiding collisions, obeying traffic rules, and making efficient progress. In many situations, these objectives cannot be satisfied simultaneously, and explicit priority relations naturally arise. Also, driving rules require context, so it is important to formally model the environment scenarios within which such rules apply. Existing benchmarks for evaluating autonomous vehicles lack such combinations of multi-objective prioritized rules and formal environment models. In this work, we introduce ScenicRules, a benchmark for evaluating autonomous driving systems in stochastic environments under prioritized multi-objective specifications. We first formalize a diverse set of objectives to serve as quantitative evaluation metrics. Next, we design a Hierarchical Rulebook framework that encodes multiple objectives and their priority relations in an interpretable and adaptable manner. We then construct a compact yet representative collection of scenarios spanning diverse driving contexts and near-accident situations, formally modeled in the Scenic language. Experimental results show that our formalized objectives and Hierarchical Rulebooks align well with human driving judgments and that our benchmark effectively exposes agent failures with respect to the prioritized objectives. Our benchmark can be accessed at this https URL. 

---
# Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research 

**Authors**: Chenda Duan, Yipeng Zhang, Sotaro Kanai, Yuanyi Ding, Atsuro Daida, Pengyue Yu, Tiancheng Zheng, Naoto Kuroda, Shaun A. Hussain, Eishi Asano, Hiroki Nariai, Vwani Roychowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2602.16072)  

**Abstract**: Epilepsy affects over 50 million people worldwide, and one-third of patients suffer drug-resistant seizures where surgery offers the best chance of seizure freedom. Accurate localization of the epileptogenic zone (EZ) relies on intracranial EEG (iEEG). Clinical workflows, however, remain constrained by labor-intensive manual review. At the same time, existing data-driven approaches are typically developed on single-center datasets that are inconsistent in format and metadata, lack standardized benchmarks, and rarely release pathological event annotations, creating barriers to reproducibility, cross-center validation, and clinical relevance. With extensive efforts to reconcile heterogeneous iEEG formats, metadata, and recordings across publicly available sources, we present $\textbf{Omni-iEEG}$, a large-scale, pre-surgical iEEG resource comprising $\textbf{302 patients}$ and $\textbf{178 hours}$ of high-resolution recordings. The dataset includes harmonized clinical metadata such as seizure onset zones, resections, and surgical outcomes, all validated by board-certified epileptologists. In addition, Omni-iEEG provides over 36K expert-validated annotations of pathological events, enabling robust biomarker studies. Omni-iEEG serves as a bridge between machine learning and epilepsy research. It defines clinically meaningful tasks with unified evaluation metrics grounded in clinical priors, enabling systematic evaluation of models in clinically relevant settings. Beyond benchmarking, we demonstrate the potential of end-to-end modeling on long iEEG segments and highlight the transferability of representations pretrained on non-neurophysiological domains. Together, these contributions establish Omni-iEEG as a foundation for reproducible, generalizable, and clinically translatable epilepsy research. The project page with dataset and code links is available at this http URL. 

---
# Can Generative Artificial Intelligence Survive Data Contamination? Theoretical Guarantees under Contaminated Recursive Training 

**Authors**: Kevin Wang, Hongqian Niu, Didong Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.16065)  

**Abstract**: Generative Artificial Intelligence (AI), such as large language models (LLMs), has become a transformative force across science, industry, and society. As these systems grow in popularity, web data becomes increasingly interwoven with this AI-generated material and it is increasingly difficult to separate them from naturally generated content. As generative models are updated regularly, later models will inevitably be trained on mixtures of human-generated data and AI-generated data from earlier versions, creating a recursive training process with data contamination. Existing theoretical work has examined only highly simplified settings, where both the real data and the generative model are discrete or Gaussian, where it has been shown that such recursive training leads to model collapse. However, real data distributions are far more complex, and modern generative models are far more flexible than Gaussian and linear mechanisms. To fill this gap, we study recursive training in a general framework with minimal assumptions on the real data distribution and allow the underlying generative model to be a general universal approximator. In this framework, we show that contaminated recursive training still converges, with a convergence rate equal to the minimum of the baseline model's convergence rate and the fraction of real data used in each iteration. To the best of our knowledge, this is the first (positive) theoretical result on recursive training without distributional assumptions on the data. We further extend the analysis to settings where sampling bias is present in data collection and support all theoretical results with empirical studies. 

---
# AI-CARE: Carbon-Aware Reporting Evaluation Metric for AI Models 

**Authors**: KC Santosh, Srikanth Baride, Rodrigue Rizk  

**Link**: [PDF](https://arxiv.org/pdf/2602.16042)  

**Abstract**: As machine learning (ML) continues its rapid expansion, the environmental cost of model training and inference has become a critical societal concern. Existing benchmarks overwhelmingly focus on standard performance metrics such as accuracy, BLEU, or mAP, while largely ignoring energy consumption and carbon emissions. This single-objective evaluation paradigm is increasingly misaligned with the practical requirements of large-scale deployment, particularly in energy-constrained environments such as mobile devices, developing regions, and climate-aware enterprises. In this paper, we propose AI-CARE, an evaluation tool for reporting energy consumption, and carbon emissions of ML models. In addition, we introduce the carbon-performance tradeoff curve, an interpretable tool that visualizes the Pareto frontier between performance and carbon cost. We demonstrate, through theoretical analysis and empirical validation on representative ML workloads, that carbon-aware benchmarking changes the relative ranking of models and encourages architectures that are simultaneously accurate and environmentally responsible. Our proposal aims to shift the research community toward transparent, multi-objective evaluation and align ML progress with global sustainability goals. The tool and documentation are available at this https URL. 

---
# Transforming GenAI Policy to Prompting Instruction: An RCT of Scalable Prompting Interventions in a CS1 Course 

**Authors**: Ruiwei Xiao, Runlong Ye, Xinying Hou, Jessica Wen, Harsh Kumar, Michael Liut, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2602.16033)  

**Abstract**: Despite universal GenAI adoption, students cannot distinguish task performance from actual learning and lack skills to leverage AI for learning, leading to worse exam performance when AI use remains unreflective. Yet few interventions teaching students to prompt AI as a tutor rather than solution provider have been validated at scale through randomized controlled trials (RCTs). To bridge this gap, we conducted a semester-long RCT (N=979) with four ICAP framework-based instructional conditions varying in engagement intensity with a pre-test, immediate and delayed post-test and surveys. Mixed methods analysis results showed: (1) All conditions significantly improved prompting skills, with gains increasing progressively from Condition 1 to Condition 4, validating ICAP's cognitive engagement hierarchy; (2) for students with similar pre-test scores, higher learning gain in immediate post-test predict higher final exam score, though no direct between-group differences emerged; (3) Our interventions are suitable and scalable solutions for diverse educational contexts, resources and learners. Together, this study makes empirical and theoretical contributions: (1) theoretically, we provided one of the first large-scale RCTs examining how cognitive engagement shapes learning in prompting literacy and clarifying the relationship between learning-oriented prompting skills and broader academic performance; (2) empirically, we offered timely design guidance for transforming GenAI classroom policies into scalable, actionable prompting literacy instruction to advance learning in the era of Generative AI. 

---
# MedProbCLIP: Probabilistic Adaptation of Vision-Language Foundation Model for Reliable Radiograph-Report Retrieval 

**Authors**: Ahmad Elallaf, Yu Zhang, Yuktha Priya Masupalli, Jeong Yang, Young Lee, Zechun Cao, Gongbo Liang  

**Link**: [PDF](https://arxiv.org/pdf/2602.16019)  

**Abstract**: Vision-language foundation models have emerged as powerful general-purpose representation learners with strong potential for multimodal understanding, but their deterministic embeddings often fail to provide the reliability required for high-stakes biomedical applications. This work introduces MedProbCLIP, a probabilistic vision-language learning framework for chest X-ray and radiology report representation learning and bidirectional retrieval. MedProbCLIP models image and text representations as Gaussian embeddings through a probabilistic contrastive objective that explicitly captures uncertainty and many-to-many correspondences between radiographs and clinical narratives. A variational information bottleneck mitigates overconfident predictions, while MedProbCLIP employs multi-view radiograph encoding and multi-section report encoding during training to provide fine-grained supervision for clinically aligned correspondence, yet requires only a single radiograph and a single report at inference. Evaluated on the MIMIC-CXR dataset, MedProbCLIP outperforms deterministic and probabilistic baselines, including CLIP, CXR-CLIP, and PCME++, in both retrieval and zero-shot classification. Beyond accuracy, MedProbCLIP demonstrates superior calibration, risk-coverage behavior, selective retrieval reliability, and robustness to clinically relevant corruptions, underscoring the value of probabilistic vision-language modeling for improving the trustworthiness and safety of radiology image-text retrieval systems. 

---
# MAEB: Massive Audio Embedding Benchmark 

**Authors**: Adnan El Assadi, Isaac Chung, Chenghao Xiao, Roman Solomatin, Animesh Jha, Rahul Chand, Silky Singh, Kaitlyn Wang, Ali Sartaz Khan, Marc Moussa Nasser, Sufen Fong, Pengfei He, Alan Xiao, Ayush Sunil Munot, Aditya Shrivastava, Artem Gazizov, Niklas Muennighoff, Kenneth Enevoldsen  

**Link**: [PDF](https://arxiv.org/pdf/2602.16008)  

**Abstract**: We introduce the Massive Audio Embedding Benchmark (MAEB), a large-scale benchmark covering 30 tasks across speech, music, environmental sounds, and cross-modal audio-text reasoning in 100+ languages. We evaluate 50+ models and find that no single model dominates across all tasks: contrastive audio-text models excel at environmental sound classification (e.g., ESC50) but score near random on multilingual speech tasks (e.g., SIB-FLEURS), while speech-pretrained models show the opposite pattern. Clustering remains challenging for all models, with even the best-performing model achieving only modest results. We observe that models excelling on acoustic understanding often perform poorly on linguistic tasks, and vice versa. We also show that the performance of audio encoders on MAEB correlates highly with their performance when used in audio large language models. MAEB is derived from MAEB+, a collection of 98 tasks. MAEB is designed to maintain task diversity while reducing evaluation cost, and it integrates into the MTEB ecosystem for unified evaluation across text, image, and audio modalities. We release MAEB and all 98 tasks along with code and a leaderboard at this https URL. 

---
# ODYN: An All-Shifted Non-Interior-Point Method for Quadratic Programming in Robotics and AI 

**Authors**: Jose Rojas, Aristotelis Papatheodorou, Sergi Martinez, Ioannis Havoutis, Carlos Mastalli  

**Link**: [PDF](https://arxiv.org/pdf/2602.16005)  

**Abstract**: We introduce ODYN, a novel all-shifted primal-dual non-interior-point quadratic programming (QP) solver designed to efficiently handle challenging dense and sparse QPs. ODYN combines all-shifted nonlinear complementarity problem (NCP) functions with proximal method of multipliers to robustly address ill-conditioned and degenerate problems, without requiring linear independence of the constraints. It exhibits strong warm-start performance and is well suited to both general-purpose optimization, and robotics and AI applications, including model-based control, estimation, and kernel-based learning methods. We provide an open-source implementation and benchmark ODYN on the Maros-Mészáros test set, demonstrating state-of-the-art convergence performance in small-to-high-scale problems. The results highlight ODYN's superior warm-starting capabilities, which are critical in sequential and real-time settings common in robotics and AI. These advantages are further demonstrated by deploying ODYN as the backend of an SQP-based predictive control framework (OdynSQP), as the implicitly differentiable optimization layer for deep learning (ODYNLayer), and the optimizer of a contact-dynamics simulation (ODYNSim). 

---
# Anatomy of Capability Emergence: Scale-Invariant Representation Collapse and Top-Down Reorganization in Neural Networks 

**Authors**: Jayadev Billa  

**Link**: [PDF](https://arxiv.org/pdf/2602.15997)  

**Abstract**: Capability emergence during neural network training remains mechanistically opaque. We track five geometric measures across five model scales (405K-85M parameters), 120+ emergence events in eight algorithmic tasks, and three Pythia language models (160M-2.8B). We find: (1) training begins with a universal representation collapse to task-specific floors that are scale-invariant across a 210X parameter range (e.g., modular arithmetic collapses to RANKME ~ 2.0 regardless of model size); (2) collapse propagates top-down through layers (32/32 task X model consistency), contradicting bottom-up feature-building intuition; (3) a geometric hierarchy in which representation geometry leads emergence (75-100% precursor rate for hard tasks), while the local learning coefficient is synchronous (0/24 precursor) and Hessian measures lag. We also delineate prediction limits: geometric measures encode coarse task difficulty but not fine-grained timing (within-class concordance 27%; when task ordering reverses across scales, prediction fails at 26%). On Pythia, global geometric patterns replicate but per-task precursor signals do not -- the precursor relationship requires task-training alignment that naturalistic pre-training does not provide. Our contribution is the geometric anatomy of emergence and its boundary conditions, not a prediction tool. 

---
# ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization 

**Authors**: Junbo Jacob Lian, Yujun Sun, Huiling Chen, Chaoyu Zhang, Chung-Piaw Teo  

**Link**: [PDF](https://arxiv.org/pdf/2602.15983)  

**Abstract**: Large language models (LLMs) can translate natural language into optimization code, but silent failures pose a critical risk: code that executes and returns solver-feasible solutions may encode semantically incorrect formulations, creating a feasibility-correctness gap of up to 90 percentage points on compositional problems. We introduce ReLoop, addressing silent failures from two complementary directions. Structured generation decomposes code production into a four-stage reasoning chain (understand, formalize, synthesize, verify) that mirrors expert modeling practice, with explicit variable-type reasoning and self-verification to prevent formulation errors at their source. Behavioral verification detects errors that survive generation by testing whether the formulation responds correctly to solver-based parameter perturbation, without requiring ground truth -- an external semantic signal that bypasses the self-consistency problem inherent in LLM-based code review. The two mechanisms are complementary: structured generation dominates on complex compositional problems, while behavioral verification becomes the largest single contributor on problems with localized formulation defects. Together with execution recovery via IIS-enhanced diagnostics, ReLoop raises correctness from 22.6% to 31.1% and execution from 72.1% to 100.0% on the strongest model, with consistent gains across five models spanning three paradigms (foundation, SFT, RL) and three benchmarks. We additionally release RetailOpt-190, 190 compositional retail optimization scenarios targeting the multi-constraint interactions where LLMs most frequently fail. 

---
# B-DENSE: Branching For Dense Ensemble Network Learning 

**Authors**: Cherish Puniani, Tushar Kumar, Arnav Bendre, Gaurav Kumar, Shree Singhi  

**Link**: [PDF](https://arxiv.org/pdf/2602.15971)  

**Abstract**: Inspired by non-equilibrium thermodynamics, diffusion models have achieved state-of-the-art performance in generative modeling. However, their iterative sampling nature results in high inference latency. While recent distillation techniques accelerate sampling, they discard intermediate trajectory steps. This sparse supervision leads to a loss of structural information and introduces significant discretization errors. To mitigate this, we propose B-DENSE, a novel framework that leverages multi-branch trajectory alignment. We modify the student architecture to output $K$-fold expanded channels, where each subset corresponds to a specific branch representing a discrete intermediate step in the teacher's trajectory. By training these branches to simultaneously map to the entire sequence of the teacher's target timesteps, we enforce dense intermediate trajectory alignment. Consequently, the student model learns to navigate the solution space from the earliest stages of training, demonstrating superior image generation quality compared to baseline distillation frameworks. 

---
# From Reflection to Repair: A Scoping Review of Dataset Documentation Tools 

**Authors**: Pedro Reynolds-Cuéllar, Marisol Wong-Villacres, Adriana Alvarado Garcia, Heila Precel  

**Link**: [PDF](https://arxiv.org/pdf/2602.15968)  

**Abstract**: Dataset documentation is widely recognized as essential for the responsible development of automated systems. Despite growing efforts to support documentation through different kinds of artifacts, little is known about the motivations shaping documentation tool design or the factors hindering their adoption. We present a systematic review supported by mixed-methods analysis of 59 dataset documentation publications to examine the motivations behind building documentation tools, how authors conceptualize documentation practices, and how these tools connect to existing systems, regulations, and cultural norms. Our analysis shows four persistent patterns in dataset documentation conceptualization that potentially impede adoption and standardization: unclear operationalizations of documentation's value, decontextualized designs, unaddressed labor demands, and a tendency to treat integration as future work. Building on these findings, we propose a shift in Responsible AI tool design toward institutional rather than individual solutions, and outline actions the HCI community can take to enable sustainable documentation practices. 

---
# Position-Aware Scene-Appearance Disentanglement for Bidirectional Photoacoustic Microscopy Registration 

**Authors**: Yiwen Wang, Jiahao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2602.15959)  

**Abstract**: High-speed optical-resolution photoacoustic microscopy (OR-PAM) with bidirectional raster scanning doubles imaging speed but introduces coupled domain shift and geometric misalignment between forward and backward scan lines. Existing registration methods, constrained by brightness constancy assumptions, achieve limited alignment quality, while recent generative approaches address domain shift through complex architectures that lack temporal awareness across frames. We propose GPEReg-Net, a scene-appearance disentanglement framework that separates domain-invariant scene features from domain-specific appearance codes via Adaptive Instance Normalization (AdaIN), enabling direct image-to-image registration without explicit deformation field estimation. To exploit temporal structure in sequential acquisitions, we introduce a Global Position Encoding (GPE) module that combines learnable position embeddings with sinusoidal encoding and cross-frame attention, allowing the network to leverage context from neighboring frames for improved temporal coherence. On the OR-PAM-Reg-4K benchmark (432 test samples), GPEReg-Net achieves NCC of 0.953, SSIM of 0.932, and PSNR of 34.49dB, surpassing the state-of-the-art by 3.8% in SSIM and 1.99dB in PSNR while maintaining competitive NCC. Code is available at this https URL. 

---
# DocSplit: A Comprehensive Benchmark Dataset and Evaluation Approach for Document Packet Recognition and Splitting 

**Authors**: Md Mofijul Islam, Md Sirajus Salekin, Nivedha Balakrishnan, Vincil C. Bishop III, Niharika Jain, Spencer Romo, Bob Strahan, Boyi Xie, Diego A. Socolinsky  

**Link**: [PDF](https://arxiv.org/pdf/2602.15958)  

**Abstract**: Document understanding in real-world applications often requires processing heterogeneous, multi-page document packets containing multiple documents stitched together. Despite recent advances in visual document understanding, the fundamental task of document packet splitting, which involves separating a document packet into individual units, remains largely unaddressed. We present the first comprehensive benchmark dataset, DocSplit, along with novel evaluation metrics for assessing the document packet splitting capabilities of large language models. DocSplit comprises five datasets of varying complexity, covering diverse document types, layouts, and multimodal settings. We formalize the DocSplit task, which requires models to identify document boundaries, classify document types, and maintain correct page ordering within a document packet. The benchmark addresses real-world challenges, including out-of-order pages, interleaved documents, and documents lacking clear demarcations. We conduct extensive experiments evaluating multimodal LLMs on our datasets, revealing significant performance gaps in current models' ability to handle complex document splitting tasks. The DocSplit benchmark datasets and proposed novel evaluation metrics provide a systematic framework for advancing document understanding capabilities essential for legal, financial, healthcare, and other document-intensive domains. We release the datasets to facilitate future research in document packet processing. 

---
# Hybrid Model Predictive Control with Physics-Informed Neural Network for Satellite Attitude Control 

**Authors**: Carlo Cena, Mauro Martini, Marcello Chiaberge  

**Link**: [PDF](https://arxiv.org/pdf/2602.15954)  

**Abstract**: Reliable spacecraft attitude control depends on accurate prediction of attitude dynamics, particularly when model-based strategies such as Model Predictive Control (MPC) are employed, where performance is limited by the quality of the internal system model. For spacecraft with complex dynamics, obtaining accurate physics-based models can be difficult, time-consuming, or computationally heavy. Learning-based system identification presents a compelling alternative; however, models trained exclusively on data frequently exhibit fragile stability properties and limited extrapolation capability. This work explores Physics-Informed Neural Networks (PINNs) for modeling spacecraft attitude dynamics and contrasts it with a conventional data-driven approach. A comprehensive dataset is generated using high-fidelity numerical simulations, and two learning methodologies are investigated: a purely data-driven pipeline and a physics-regularized approach that incorporates prior knowledge into the optimization process. The results indicate that embedding physical constraints during training leads to substantial improvements in predictive reliability, achieving a 68.17% decrease in mean relative error relative. When deployed within an MPC architecture, the physics-informed models yield superior closed-loop tracking performance and improved robustness to uncertainty. Furthermore, a hybrid control formulation that merges the learned nonlinear dynamics with a nominal linear model enables consistent steady-state convergence and significantly faster response, reducing settling times by 61.52%-76.42% under measurement noise and reaction wheel friction. 

---
# From Tool Orchestration to Code Execution: A Study of MCP Design Choices 

**Authors**: Yuval Felendler, Parth A. Gandhi, Idan Habler, Yuval Elovici, Asaf Shabtai  

**Link**: [PDF](https://arxiv.org/pdf/2602.15945)  

**Abstract**: Model Context Protocols (MCPs) provide a unified platform for agent systems to discover, select, and orchestrate tools across heterogeneous execution environments. As MCP-based systems scale to incorporate larger tool catalogs and multiple concurrently connected MCP servers, traditional tool-by-tool invocation increases coordination overhead, fragments state management, and limits support for wide-context operations. To address these scalability challenges, recent MCP designs have incorporated code execution as a first-class capability, an approach called Code Execution MCP (CE-MCP). This enables agents to consolidate complex workflows, such as SQL querying, file analysis, and multi-step data transformations, into a single program that executes within an isolated runtime environment. In this work, we formalize the architectural distinction between context-coupled (traditional) and context-decoupled (CE-MCP) models, analyzing their fundamental scalability trade-offs. Using the MCP-Bench framework across 10 representative servers, we empirically evaluate task behavior, tool utilization patterns, execution latency, and protocol efficiency as the scale of connected MCP servers and available tools increases, demonstrating that while CE-MCP significantly reduces token usage and execution latency, it introduces a vastly expanded attack surface. We address this security gap by applying the MAESTRO framework, identifying sixteen attack classes across five execution phases-including specific code execution threats such as exception-mediated code injection and unsafe capability synthesis. We validate these vulnerabilities through adversarial scenarios across multiple LLMs and propose a layered defense architecture comprising containerized sandboxing and semantic gating. Our findings provide a rigorous roadmap for balancing scalability and security in production-ready executable agent workflows. 

---
# A fully differentiable framework for training proxy Exchange Correlation Functionals for periodic systems 

**Authors**: Rakshit Kumar Singh, Aryan Amit Barsainyan, Bharath Ramsundar  

**Link**: [PDF](https://arxiv.org/pdf/2602.15923)  

**Abstract**: Density Functional Theory (DFT) is widely used for first-principles simulations in chemistry and materials science, but its computational cost remains a key limitation for large systems. Motivated by recent advances in ML-based exchange-correlation (XC) functionals, this paper introduces a differentiable framework that integrates machine learning models into density functional theory (DFT) for solids and other periodic systems. The framework defines a clean API for neural network models that can act as drop in replacements for conventional exchange-correlation (XC) functionals and enables gradients to flow through the full self-consistent DFT workflow. The framework is implemented in Python using a PyTorch backend, making it fully differentiable and easy to use with standard deep learning tools. We integrate the implementation with the DeepChem library to promote the reuse of established models and to lower the barrier for experimentation. In initial benchmarks against established electronic structure packages (GPAW and PySCF), our models achieve relative errors on the order of 5-10%. 

---
# Generalized Leverage Score for Scalable Assessment of Privacy Vulnerability 

**Authors**: Valentin Dorseuil, Jamal Atif, Olivier Cappé  

**Link**: [PDF](https://arxiv.org/pdf/2602.15919)  

**Abstract**: Can the privacy vulnerability of individual data points be assessed without retraining models or explicitly simulating attacks? We answer affirmatively by showing that exposure to membership inference attack (MIA) is fundamentally governed by a data point's influence on the learned model. We formalize this in the linear setting by establishing a theoretical correspondence between individual MIA risk and the leverage score, identifying it as a principled metric for vulnerability. This characterization explains how data-dependent sensitivity translates into exposure, without the computational burden of training shadow models. Building on this, we propose a computationally efficient generalization of the leverage score for deep learning. Empirical evaluations confirm a strong correlation between the proposed score and MIA success, validating this metric as a practical surrogate for individual privacy risk assessment. 

---
# EarthSpatialBench: Benchmarking Spatial Reasoning Capabilities of Multimodal LLMs on Earth Imagery 

**Authors**: Zelin Xu, Yupu Zhang, Saugat Adhikari, Saiful Islam, Tingsong Xiao, Zibo Liu, Shigang Chen, Da Yan, Zhe Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15918)  

**Abstract**: Benchmarking spatial reasoning in multimodal large language models (MLLMs) has attracted growing interest in computer vision due to its importance for embodied AI and other agentic systems that require precise interaction with the physical world. However, spatial reasoning on Earth imagery has lagged behind, as it uniquely involves grounding objects in georeferenced images and quantitatively reasoning about distances, directions, and topological relations using both visual cues and vector geometry coordinates (e.g., 2D bounding boxes, polylines, and polygons). Existing benchmarks for Earth imagery primarily focus on 2D spatial grounding, image captioning, and coarse spatial relations (e.g., simple directional or proximity cues). They lack support for quantitative direction and distance reasoning, systematic topological relations, and complex object geometries beyond bounding boxes. To fill this gap, we propose \textbf{EarthSpatialBench}, a comprehensive benchmark for evaluating spatial reasoning in MLLMs on Earth imagery. The benchmark contains over 325K question-answer pairs spanning: (1) qualitative and quantitative reasoning about spatial distance and direction; (2) systematic topological relations; (3) single-object queries, object-pair queries, and compositional aggregate group queries; and (4) object references expressed via textual descriptions, visual overlays, and explicit geometry coordinates, including 2D bounding boxes, polylines, and polygons. We conducted extensive experiments on both open-source and proprietary models to identify limitations in the spatial reasoning of MLLMs. 

---
# MaS-VQA: A Mask-and-Select Framework for Knowledge-Based Visual Question Answering 

**Authors**: Xianwei Mao, Kai Ye, Sheng Zhou, Nan Zhang, Haikuan Huang, Bin Li, Jiajun Bu  

**Link**: [PDF](https://arxiv.org/pdf/2602.15915)  

**Abstract**: Knowledge-based Visual Question Answering (KB-VQA) requires models to answer questions by integrating visual information with external knowledge. However, retrieved knowledge is often noisy, partially irrelevant, or misaligned with the visual content, while internal model knowledge is difficult to control and interpret. Naive aggregation of these sources limits reasoning effectiveness and reduces answer accuracy. To address this, we propose MaS-VQA, a selection-driven framework that tightly couples explicit knowledge filtering with implicit knowledge reasoning. MaS-VQA first retrieves candidate passages and applies a Mask-and-Select mechanism to jointly prune irrelevant image regions and weakly relevant knowledge fragments, producing compact, high-signal multimodal knowledge . This filtered knowledge then guides the activation of internal knowledge in a constrained semantic space, enabling complementary co-modeling of explicit and implicit knowledge for robust answer prediction. Experiments on Encyclopedic-VQA and InfoSeek demonstrate consistent performance gains across multiple MLLM backbones, and ablations verify that the selection mechanism effectively reduces noise and enhances knowledge utilization. 

---
# Foundation Models for Medical Imaging: Status, Challenges, and Directions 

**Authors**: Chuang Niu, Pengwei Wu, Bruno De Man, Ge Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15913)  

**Abstract**: Foundation models (FMs) are rapidly reshaping medical imaging, shifting the field from narrowly trained, task-specific networks toward large, general-purpose models that can be adapted across modalities, anatomies, and clinical tasks. In this review, we synthesize the emerging landscape of medical imaging FMs along three major axes: principles of FM design, applications of FMs, and forward-looking challenges and opportunities. Taken together, this review provides a technically grounded, clinically aware, and future-facing roadmap for developing FMs that are not only powerful and versatile but also trustworthy and ready for responsible translation into clinical practice. 

---
# Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis 

**Authors**: Pengfei Zhang, Tianxin Xie, Minghao Yang, Li Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.15909)  

**Abstract**: Deep learning-based respiratory auscultation is currently hindered by two fundamental challenges: (i) inherent information loss, as converting signals into spectrograms discards transient acoustic events and clinical context; (ii) limited data availability, exacerbated by severe class imbalance. To bridge these gaps, we present Resp-Agent, an autonomous multimodal system orchestrated by a novel Active Adversarial Curriculum Agent (Thinker-A$^2$CA). Unlike static pipelines, Thinker-A$^2$CA serves as a central controller that actively identifies diagnostic weaknesses and schedules targeted synthesis in a closed loop. To address the representation gap, we introduce a Modality-Weaving Diagnoser that weaves EHR data with audio tokens via Strategic Global Attention and sparse audio anchors, capturing both long-range clinical context and millisecond-level transients. To address the data gap, we design a Flow Matching Generator that adapts a text-only Large Language Model (LLM) via modality injection, decoupling pathological content from acoustic style to synthesize hard-to-diagnose samples. As a foundation for these efforts, we introduce Resp-229k, a benchmark corpus of 229k recordings paired with LLM-distilled clinical narratives. Extensive experiments demonstrate that Resp-Agent consistently outperforms prior approaches across diverse evaluation settings, improving diagnostic robustness under data scarcity and long-tailed class imbalance. Our code and data are available at this https URL. 

---
# Doc-to-LoRA: Learning to Instantly Internalize Contexts 

**Authors**: Rujikorn Charakorn, Edoardo Cetin, Shinnosuke Uesaka, Robert Tjarko Lange  

**Link**: [PDF](https://arxiv.org/pdf/2602.15902)  

**Abstract**: Long input sequences are central to in-context learning, document understanding, and multi-step reasoning of Large Language Models (LLMs). However, the quadratic attention cost of Transformers makes inference memory-intensive and slow. While context distillation (CD) can transfer information into model parameters, per-prompt distillation is impractical due to training costs and latency. To address these limitations, we propose Doc-to-LoRA (D2L), a lightweight hypernetwork that meta-learns to perform approximate CD within a single forward pass. Given an unseen prompt, D2L generates a LoRA adapter for a target LLM, enabling subsequent queries to be answered without re-consuming the original context, reducing latency and KV-cache memory consumption during inference of the target LLM. On a long-context needle-in-a-haystack task, D2L successfully learns to map contexts into adapters that store the needle information, achieving near-perfect zero-shot accuracy at sequence lengths exceeding the target LLM's native context window by more than 4x. On real-world QA datasets with limited compute, D2L outperforms standard CD while significantly reducing peak memory consumption and update latency. We envision that D2L can facilitate rapid adaptation of LLMs, opening up the possibility of frequent knowledge updates and personalized chat behavior. 

---
# Understand Then Memory: A Cognitive Gist-Driven RAG Framework with Global Semantic Diffusion 

**Authors**: Pengcheng Zhou, Haochen Li, Zhiqiang Nie, JiaLe Chen, Qing Gong, Weizhen Zhang, Chun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.15895)  

**Abstract**: Retrieval-Augmented Generation (RAG) effectively mitigates hallucinations in LLMs by incorporating external knowledge. However, the inherent discrete representation of text in existing frameworks often results in a loss of semantic integrity, leading to retrieval deviations. Inspired by the human episodic memory mechanism, we propose CogitoRAG, a RAG framework that simulates human cognitive memory processes. The core of this framework lies in the extraction and evolution of the Semantic Gist. During the offline indexing stage, CogitoRAG first deduces unstructured corpora into gist memory corpora, which are then transformed into a multi-dimensional knowledge graph integrating entities, relational facts, and memory nodes. In the online retrieval stage, the framework handles complex queries via Query Decomposition Module that breaks them into comprehensive sub-queries, mimicking the cognitive decomposition humans employ for complex information. Subsequently, Entity Diffusion Module performs associative retrieval across the graph, guided by structural relevance and an entity-frequency reward mechanism. Furthermore, we propose the CogniRank algorithm, which precisely reranks candidate passages by fusing diffusion-derived scores with semantic similarity. The final evidence is delivered to the generator in a passage-memory pairing format, providing high-density information support. Experimental results across five mainstream QA benchmarks and multi-task generation on GraphBench demonstrate that CogitoRAG significantly outperforms state-of-the-art RAG methods, showcasing superior capabilities in complex knowledge integration and reasoning. 

---
# Egocentric Bias in Vision-Language Models 

**Authors**: Maijunxian Wang, Yijiang Li, Bingyang Wang, Tianwei Zhao, Ran Ji, Qingying Gao, Emmy Liu, Hokin Deng, Dezhi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.15892)  

**Abstract**: Visual perspective taking--inferring how the world appears from another's viewpoint--is foundational to social cognition. We introduce FlipSet, a diagnostic benchmark for Level-2 visual perspective taking (L2 VPT) in vision-language models. The task requires simulating 180-degree rotations of 2D character strings from another agent's perspective, isolating spatial transformation from 3D scene complexity. Evaluating 103 VLMs reveals systematic egocentric bias: the vast majority perform below chance, with roughly three-quarters of errors reproducing the camera viewpoint. Control experiments expose a compositional deficit--models achieve high theory-of-mind accuracy and above-chance mental rotation in isolation, yet fail catastrophically when integration is required. This dissociation indicates that current VLMs lack the mechanisms needed to bind social awareness to spatial operations, suggesting fundamental limitations in model-based spatial reasoning. FlipSet provides a cognitively grounded testbed for diagnosing perspective-taking capabilities in multimodal systems. 

---
# Surrogate Modeling for Neutron Transport: A Neural Operator Approach 

**Authors**: Md Hossain Sahadath, Qiyun Cheng, Shaowu Pan, Wei Ji  

**Link**: [PDF](https://arxiv.org/pdf/2602.15890)  

**Abstract**: This work introduces a neural operator based surrogate modeling framework for neutron transport computation. Two architectures, the Deep Operator Network (DeepONet) and the Fourier Neural Operator (FNO), were trained for fixed source problems to learn the mapping from anisotropic neutron sources, Q(x,{\mu}), to the corresponding angular fluxes, {\psi}(x,{\mu}), in a one-dimensional slab geometry. Three distinct models were trained for each neural operator, corresponding to different scattering ratios (c = 0.1, 0.5, & 1.0), providing insight into their performance across distinct transport regimes (absorption-dominated, moderate, and scattering-dominated). The models were subsequently evaluated on a wide range of previously unseen source configurations, demonstrating that FNO generally achieves higher predictive accuracy, while DeepONet offers greater computational efficiency. Both models offered significant speedups that become increasingly pronounced as the scattering ratio increases, requiring <0.3% of the runtime of a conventional S_N solver. The surrogate models were further incorporated into the S_N k-eigenvalue solver, replacing the computationally intensive transport sweep loop with a single forward pass. Across varying fission cross sections and spatial-angular grids, both neural operator solvers reproduced reference eigenvalues with deviations up to 135 pcm for DeepONet and 112 pcm for FNO, while reducing runtime to <0.1% of that of the S_N solver on relatively fine grids. These results demonstrate the strong potential of neural operator frameworks as accurate, efficient, and generalizable surrogates for neutron transport, paving the way for real-time digital twin applications and repeated evaluations, such as in design optimization. 

---
# Evidence for Daily and Weekly Periodic Variability in GPT-4o Performance 

**Authors**: Paul Tschisgale, Peter Wulff  

**Link**: [PDF](https://arxiv.org/pdf/2602.15889)  

**Abstract**: Large language models (LLMs) are increasingly used in research both as tools and as objects of investigation. Much of this work implicitly assumes that LLM performance under fixed conditions (identical model snapshot, hyperparameters, and prompt) is time-invariant. If average output quality changes systematically over time, this assumption is violated, threatening the reliability, validity, and reproducibility of findings. To empirically examine this assumption, we conducted a longitudinal study on the temporal variability of GPT-4o's average performance. Using a fixed model snapshot, fixed hyperparameters, and identical prompting, GPT-4o was queried via the API to solve the same multiple-choice physics task every three hours for approximately three months. Ten independent responses were generated at each time point and their scores were averaged. Spectral (Fourier) analysis of the resulting time series revealed notable periodic variability in average model performance, accounting for approximately 20% of the total variance. In particular, the observed periodic patterns are well explained by the interaction of a daily and a weekly rhythm. These findings indicate that, even under controlled conditions, LLM performance may vary periodically over time, calling into question the assumption of time invariance. Implications for ensuring validity and replicability of research that uses or investigates LLMs are discussed. 

---
# NeuroSleep: Neuromorphic Event-Driven Single-Channel EEG Sleep Staging for Edge-Efficient Sensing 

**Authors**: Boyu Li, Xingchun Zhu, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.15888)  

**Abstract**: Reliable, continuous neural sensing on wearable edge platforms is fundamental to long-term health monitoring; however, for electroencephalography (EEG)-based sleep monitoring, dense high-frequency processing is often computationally prohibitive under tight energy budgets. To address this bottleneck, this paper proposes NeuroSleep, an integrated event-driven sensing and inference system for energy-efficient sleep staging. NeuroSleep first converts raw EEG into complementary multi-scale bipolar event streams using Residual Adaptive Multi-Scale Delta Modulation (R-AMSDM), enabling an explicit fidelity-sparsity trade-off at the sensing front end. Furthermore, NeuroSleep adopts a hierarchical inference architecture that comprises an Event-based Adaptive Multi-scale Response (EAMR) module for local feature extraction, a Local Temporal-Attention Module (LTAM) for context aggregation, and an Epoch-Leaky Integrate-and-Fire (ELIF) module to capture long-term state persistence. Experimental results using subject-independent 5-fold cross-validation on the Sleep-EDF Expanded dataset demonstrate that NeuroSleep achieves a mean accuracy of 74.2% with only 0.932 M parameters while reducing sparsity-adjusted effective operations by approximately 53.6% relative to dense processing. Compared with the representative dense Transformer baseline, NeuroSleep improves accuracy by 7.5% with a 45.8% reduction in computational load. By bridging neuromorphic encoding with state-aware modeling, NeuroSleep provides a scalable solution for always-on sleep analysis in resource-constrained wearable scenarios. 

---
# FUTURE-VLA: Forecasting Unified Trajectories Under Real-time Execution 

**Authors**: Jingjing Fan, Yushan Liu, Shoujie Li, Botao Ren, Siyuan Li, Xiao-Ping Zhang, Wenbo Ding, Zhidong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.15882)  

**Abstract**: General vision-language models increasingly support unified spatiotemporal reasoning over long video streams, yet deploying such capabilities on robots remains constrained by the prohibitive latency of processing long-horizon histories and generating high-dimensional future predictions. To bridge this gap, we present FUTURE-VLA, a unified architecture that reformulates long-horizon control and future forecasting as a monolithic sequence-generation task. Adopting a dual-sided efficiency paradigm, FUTURE-VLA leverages a temporally adaptive compression strategy to maximize spatiotemporal information density, enabling the ingestion of extensive multi-view histories while maintaining constant inference latency. Simultaneously, it performs latent-space autoregression to align actionable dynamics with reviewable visual look-aheads in a single forward pass. These real-time predictive capabilities further enable a prediction-guided Human-In-the-Loop mechanism via interactive execution gating, allowing operators to dynamically validate behaviors based on interpretable future previews. Extensive evaluations demonstrate that FUTURE-VLA establishes new state-of-the-art performance, attaining success rates of 99.2% on LIBERO, 75.4% on RoboTwin, and 78.0% on a real-world Piper platform, all with a $16\times$ extended spatiotemporal window while maintaining the inference latency of a single-frame baseline. 

---
# IT-OSE: Exploring Optimal Sample Size for Industrial Data Augmentation 

**Authors**: Mingchun Sun, Rongqiang Zhao, Zhennan Huang, Songyu Ding, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.15878)  

**Abstract**: In industrial scenarios, data augmentation is an effective approach to improve model performance. However, its benefits are not unidirectionally beneficial. There is no theoretical research or established estimation for the optimal sample size (OSS) in augmentation, nor is there an established metric to evaluate the accuracy of OSS or its deviation from the ground truth. To address these issues, we propose an information-theoretic optimal sample size estimation (IT-OSE) to provide reliable OSS estimation for industrial data augmentation. An interval coverage and deviation (ICD) score is proposed to evaluate the estimated OSS intuitively. The relationship between OSS and dominant factors is theoretically analyzed and formulated, thereby enhancing the interpretability. Experiments show that, compared to empirical estimation, the IT-OSE increases accuracy in classification tasks across baseline models by an average of 4.38%, and reduces MAPE in regression tasks across baseline models by an average of 18.80%. The improvements in downstream model performance are more stable. ICDdev in the ICD score is also reduced by an average of 49.30%. The determinism of OSS is enhanced. Compared to exhaustive search, the IT-OSE achieves the same OSS while reducing computational and data costs by an average of 83.97% and 93.46%. Furthermore, practicality experiments demonstrate that the IT-OSE exhibits generality across representative sensor-based industrial scenarios. 

---
# Genetic Generalized Additive Models 

**Authors**: Kaaustaaub Shankar, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2602.15877)  

**Abstract**: Generalized Additive Models (GAMs) balance predictive accuracy and interpretability, but manually configuring their structure is challenging. We propose using the multi-objective genetic algorithm NSGA-II to automatically optimize GAMs, jointly minimizing prediction error (RMSE) and a Complexity Penalty that captures sparsity, smoothness, and uncertainty. Experiments on the California Housing dataset show that NSGA-II discovers GAMs that outperform baseline LinearGAMs in accuracy or match performance with substantially lower complexity. The resulting models are simpler, smoother, and exhibit narrower confidence intervals, enhancing interpretability. This framework provides a general approach for automated optimization of transparent, high-performing models. The code can be found at this https URL. 

---
# Fly0: Decoupling Semantic Grounding from Geometric Planning for Zero-Shot Aerial Navigation 

**Authors**: Zhenxing Xu, Brikit Lu, Weidong Bao, Zhengqiu Zhu, Junsong Zhang, Hui Yan, Wenhao Lu, Ji Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15875)  

**Abstract**: Current Visual-Language Navigation (VLN) methodologies face a trade-off between semantic understanding and control precision. While Multimodal Large Language Models (MLLMs) offer superior reasoning, deploying them as low-level controllers leads to high latency, trajectory oscillations, and poor generalization due to weak geometric grounding. To address these limitations, we propose Fly0, a framework that decouples semantic reasoning from geometric planning. The proposed method operates through a three-stage pipeline: (1) an MLLM-driven module for grounding natural language instructions into 2D pixel coordinates; (2) a geometric projection module that utilizes depth data to localize targets in 3D space; and (3) a geometric planner that generates collision-free trajectories. This mechanism enables robust navigation even when visual contact is lost. By eliminating the need for continuous inference, Fly0 reduces computational overhead and improves system stability. Extensive experiments in simulation and real-world environments demonstrate that Fly0 outperforms state-of-the-art baselines, improving the Success Rate by over 20\% and reducing Navigation Error (NE) by approximately 50\% in unstructured environments. Our code is available at this https URL. 

---
# Test-Time Adaptation for Tactile-Vision-Language Models 

**Authors**: Chuyang Ye, Haoxian Jing, Qinting Jiang, Yixi Lin, Qiang Li, Xing Tang, Jingyan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15873)  

**Abstract**: Tactile-vision-language (TVL) models are increasingly deployed in real-world robotic and multimodal perception tasks, where test-time distribution shifts are unavoidable. Existing test-time adaptation (TTA) methods provide filtering in unimodal settings but lack explicit treatment of modality-wise reliability under asynchronous cross-modal shifts, leaving them brittle when some modalities become unreliable. We study TTA for TVL models under such shifts and propose a reliability-aware framework that estimates per-modality reliability from prediction uncertainty and perturbation-based responses. This shared reliability signal is used to (i) filter unreliable test samples, (ii) adaptively fuse tactile, visual, and language features, and (iii) regularize test-time optimization with a reliability-guided objective. On the TAG-C benchmark and additional TVL scenarios, our approach consistently outperforms strong TTA baselines, achieving accuracy gains of up to 49.9\% under severe modality corruptions, underscoring the importance of explicit modality-wise reliability modeling for robust test-time adaptation. 

---
# Playing With AI: How Do State-Of-The-Art Large Language Models Perform in the 1977 Text-Based Adventure Game Zork? 

**Authors**: Berry Gerrits  

**Link**: [PDF](https://arxiv.org/pdf/2602.15867)  

**Abstract**: In this positioning paper, we evaluate the problem-solving and reasoning capabilities of contemporary Large Language Models (LLMs) through their performance in Zork, the seminal text-based adventure game first released in 1977. The game's dialogue-based structure provides a controlled environment for assessing how LLM-based chatbots interpret natural language descriptions and generate appropriate action sequences to succeed in the game. We test the performance of leading proprietary models - ChatGPT, Claude, and Gemini - under both minimal and detailed instructions, measuring game progress through achieved scores as the primary metric. Our results reveal that all tested models achieve less than 10% completion on average, with even the best-performing model (Claude Opus 4.5) reaching only approximately 75 out of 350 possible points. Notably, providing detailed game instructions offers no improvement, nor does enabling ''extended thinking''. Qualitative analysis of the models' reasoning processes reveals fundamental limitations: repeated unsuccessful actions suggesting an inability to reflect on one's own thinking, inconsistent persistence of strategies, and failure to learn from previous attempts despite access to conversation history. These findings suggest substantial limitations in current LLMs' metacognitive abilities and problem-solving capabilities within the domain of text-based games, raising questions about the nature and extent of their reasoning capabilities. 

---
# NLP Privacy Risk Identification in Social Media (NLP-PRISM): A Survey 

**Authors**: Dhiman Goswami, Jai Kruthunz Naveen Kumar, Sanchari Das  

**Link**: [PDF](https://arxiv.org/pdf/2602.15866)  

**Abstract**: Natural Language Processing (NLP) is integral to social media analytics but often processes content containing Personally Identifiable Information (PII), behavioral cues, and metadata raising privacy risks such as surveillance, profiling, and targeted advertising. To systematically assess these risks, we review 203 peer-reviewed papers and propose the NLP Privacy Risk Identification in Social Media (NLP-PRISM) framework, which evaluates vulnerabilities across six dimensions: data collection, preprocessing, visibility, fairness, computational risk, and regulatory compliance. Our analysis shows that transformer models achieve F1-scores ranging from 0.58-0.84, but incur a 1% - 23% drop under privacy-preserving fine-tuning. Using NLP-PRISM, we examine privacy coverage in six NLP tasks: sentiment analysis (16), emotion detection (14), offensive language identification (19), code-mixed processing (39), native language identification (29), and dialect detection (24) revealing substantial gaps in privacy research. We further found a (reduced by 2% - 9%) trade-off in model utility, MIA AUC (membership inference attacks) 0.81, AIA accuracy 0.75 (attribute inference attacks). Finally, we advocate for stronger anonymization, privacy-aware learning, and fairness-driven training to enable ethical NLP in social media contexts. 

---
# AI as Teammate or Tool? A Review of Human-AI Interaction in Decision Support 

**Authors**: Most. Sharmin Sultana Samu, Nafisa Khan, Kazi Toufique Elahi, Tasnuva Binte Rahman, Md. Rakibul Islam, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2602.15865)  

**Abstract**: The integration of Artificial Intelligence (AI) necessitates determining whether systems function as tools or collaborative teammates. In this study, by synthesizing Human-AI Interaction (HAI) literature, we analyze this distinction across four dimensions: interaction design, trust calibration, collaborative frameworks and healthcare applications. Our analysis reveals that static interfaces and miscalibrated trust limit AI efficacy. Performance hinges on aligning transparency with cognitive workflows, yet a fluency trap often inflates trust without improving decision-making. Consequently, an overemphasis on explainability leaves systems largely passive. Our findings show that current AI systems remain largely passive due to an overreliance on explainability-centric designs and that transitioning AI to an active teammate requires adaptive, context-aware interactions that support shared mental models and the dynamic negotiation of authority between humans and AI. 

---
# Not the Example, but the Process: How Self-Generated Examples Enhance LLM Reasoning 

**Authors**: Daehoon Gwak, Minseo Jung, Junwoo Park, Minho Park, ChaeHun Park, Junha Hyung, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2602.15863)  

**Abstract**: Recent studies have shown that Large Language Models (LLMs) can improve their reasoning performance through self-generated few-shot examples, achieving results comparable to manually curated in-context examples. However, the underlying mechanism behind these gains remains unclear, making it hard to decide when and how to apply the technique effectively. In this work, we argue that the key benefit arises not from the generated examples themselves but from the act of creating them. To validate this, on reasoning-intensive tasks across diverse LLM architectures, we systematically evaluate three prompting strategies for in-context learning: (1) Zero-shot prompting; (2) Integrated prompting, where LLMs create and solve problems within a single, unified prompt; and (3) Decoupled prompting, where self-generated examples are reused as in-context examples, but the context of their creation itself is excluded. We conduct experiments across five widely used model architectures, demonstrating that Integrated prompting consistently outperforms both Zero-shot and Decoupled prompting. In contrast, Decoupled prompting offers only marginal gains over Zero-shot. Further, for a more in-depth analysis, we conduct an attention analysis and observe significant differences in attention patterns between Integrated and Decoupled prompting. These findings suggest that the advantage of self-generation prompting comes from the process of problem creation, not the examples themselves, providing valuable insights for designing more effective prompting strategies. 

---
# Enhancing Action and Ingredient Modeling for Semantically Grounded Recipe Generation 

**Authors**: Guoshan Liu, Bin Zhu, Yian Li, Jingjing Chen, Chong-Wah Ngo, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15862)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLMMs) have enabled recipe generation from food images, yet outputs often contain semantically incorrect actions or ingredients despite high lexical scores (e.g., BLEU, ROUGE). To address this gap, we propose a semantically grounded framework that predicts and validates actions and ingredients as internal context for instruction generation. Our two-stage pipeline combines supervised fine-tuning (SFT) with reinforcement fine-tuning (RFT): SFT builds foundational accuracy using an Action-Reasoning dataset and ingredient corpus, while RFT employs frequency-aware rewards to improve long-tail action prediction and ingredient generalization. A Semantic Confidence Scoring and Rectification (SCSR) module further filters and corrects predictions. Experiments on Recipe1M show state-of-the-art performance and markedly improved semantic fidelity. 

---
# CAST: Achieving Stable LLM-based Text Analysis for Data Analytics 

**Authors**: Jinxiang Xie, Zihao Li, Wei He, Rui Ding, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15861)  

**Abstract**: Text analysis of tabular data relies on two core operations: \emph{summarization} for corpus-level theme extraction and \emph{tagging} for row-level labeling. A critical limitation of employing large language models (LLMs) for these tasks is their inability to meet the high standards of output stability demanded by data analytics. To address this challenge, we introduce \textbf{CAST} (\textbf{C}onsistency via \textbf{A}lgorithmic Prompting and \textbf{S}table \textbf{T}hinking), a framework that enhances output stability by constraining the model's latent reasoning path. CAST combines (i) Algorithmic Prompting to impose a procedural scaffold over valid reasoning transitions and (ii) Thinking-before-Speaking to enforce explicit intermediate commitments before final generation. To measure progress, we introduce \textbf{CAST-S} and \textbf{CAST-T}, stability metrics for bulleted summarization and tagging, and validate their alignment with human judgments. Experiments across publicly available benchmarks on multiple LLM backbones show that CAST consistently achieves the best stability among all baselines, improving Stability Score by up to 16.2\%, while maintaining or improving output quality. 

---
# State Design Matters: How Representations Shape Dynamic Reasoning in Large Language Models 

**Authors**: Annie Wong, Aske Plaat, Thomas Bäck, Niki van Stein, Anna V. Kononova  

**Link**: [PDF](https://arxiv.org/pdf/2602.15858)  

**Abstract**: As large language models (LLMs) move from static reasoning tasks toward dynamic environments, their success depends on the ability to navigate and respond to an environment that changes as they interact at inference time. An underexplored factor in these settings is the representation of the state. Holding model parameters fixed, we systematically vary three key aspects: (1) state granularity (long form versus summary), (2) structure (natural language versus symbolic), and (3) spatial grounding (text-only versus images or textual map encodings) across sequential decision-making benchmarks. We find that trajectory summarisation improves performance by reducing noise and stabilising long-horizon reasoning. Second, natural language representations are the most robust across models, whereas structured encodings help mainly for models with strong code or structured output priors, such as JSON schemas. Third, while image-inputs show some benefit, text-based spatial encodings prove most effective. This advantage stems not from the spatial information itself, but from the act of construction, which compels the model to perform the spatial reasoning that static input does not elicit. Overall, we demonstrate that design choices for representing state are a decisive factor in performance, distinct from the availability of information itself. We note, however, that even with improved representations, current LLMs and VLMs remain brittle over long horizons, particularly when they must synthesise information to manage multiple subtasks to reach a goal. 

---
# Rethinking Soft Compression in Retrieval-Augmented Generation: A Query-Conditioned Selector Perspective 

**Authors**: Yunhao Liu, Zian Jia, Xinyu Gao, Kanjun Xu, Yun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2602.15856)  

**Abstract**: Retrieval-Augmented Generation (RAG) effectively grounds Large Language Models (LLMs) with external knowledge and is widely applied to Web-related tasks. However, its scalability is hindered by excessive context length and redundant retrievals. Recent research on soft context compression aims to address this by encoding long documents into compact embeddings, yet they often underperform non-compressed RAG due to their reliance on auto-encoder-like full-compression that forces the encoder to compress all document information regardless of relevance to the input query.
In this work, we conduct an analysis on this paradigm and reveal two fundamental limitations: (I) Infeasibility, full-compression conflicts with the LLM's downstream generation behavior; and (II) Non-necessity: full-compression is unnecessary and dilutes task-relevant information density. Motivated by these insights, we introduce SeleCom, a selector-based soft compression framework for RAG that redefines the encoder's role as query-conditioned information selector. The selector is decoder-only and is trained with a massive, diverse and difficulty-graded synthetic QA dataset with curriculum learning.
Extensive experiments show that SeleCom significantly outperforms existing soft compression approaches and achieves competitive or superior performance to non-compression baselines, while reducing computation and latency by 33.8%~84.6%. 

---
# Kalman-Inspired Runtime Stability and Recovery in Hybrid Reasoning Systems 

**Authors**: Barak Or  

**Link**: [PDF](https://arxiv.org/pdf/2602.15855)  

**Abstract**: Hybrid reasoning systems that combine learned components with model-based inference are increasingly deployed in tool-augmented decision loops, yet their runtime behavior under partial observability and sustained evidence mismatch remains poorly understood. In practice, failures often arise as gradual divergence of internal reasoning dynamics rather than as isolated prediction errors. This work studies runtime stability in hybrid reasoning systems from a Kalman-inspired perspective. We model reasoning as a stochastic inference process driven by an internal innovation signal and introduce cognitive drift as a measurable runtime phenomenon. Stability is defined in terms of detectability, bounded divergence, and recoverability rather than task-level correctness. We propose a runtime stability framework that monitors innovation statistics, detects emerging instability, and triggers recovery-aware control mechanisms. Experiments on multi-step, tool-augmented reasoning tasks demonstrate reliable instability detection prior to task failure and show that recovery, when feasible, re-establishes bounded internal behavior within finite time. These results emphasize runtime stability as a system-level requirement for reliable reasoning under uncertainty. 

---
# Decoupling Strategy and Execution in Task-Focused Dialogue via Goal-Oriented Preference Optimization 

**Authors**: Jingyi Xu, Xingyu Ren, Zhiqiang You, Yumeng Zhang, Zhoupeng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2602.15854)  

**Abstract**: Large language models show potential in task-oriented dialogue systems, yet existing training methods often rely on token-level likelihood or preference optimization, which poorly align with long-horizon task success. To address this, we propose Goal-Oriented Preference Optimization (GOPO), a hierarchical reinforcement learning framework that decouples strategy planning from response generation via an Expert Agent and a Customer Service Agent. The Expert Agent optimizes multi-turn goal preferences at the dialogue-trajectory level, while the Customer Service Agent generates responses strictly aligned with the selected strategy. We evaluate GOPO on public benchmarks and e-commerce customer service datasets, and introduce Task-focused Sequential Engagement (TSE), a sequence-level metric derived from real e-commerce interaction data. On the Mgshop dataset, GOPO improves TSE by 7.7% and 10.3% over PPO and Memento, with consistent gains in sequence-level reward and generation quality. Furthermore, a 14B model trained with GOPO achieves 2.7% and 1.5% higher TSE than Qwen-235B and GPT-5.2, respectively. Ablation studies confirm the Expert Agent's critical role in long-horizon optimization. GOPO demonstrates consistent improvements across other datasets as well. This work establishes a new paradigm for task-oriented dialogue systems in commercial scenarios, with code and datasets to be made public. 

---
# A Lightweight Explainable Guardrail for Prompt Safety 

**Authors**: Md Asiful Islam, Mihai Surdeanu  

**Link**: [PDF](https://arxiv.org/pdf/2602.15853)  

**Abstract**: We propose a lightweight explainable guardrail (LEG) method for the classification of unsafe prompts. LEG uses a multi-task learning architecture to jointly learn a prompt classifier and an explanation classifier, where the latter labels prompt words that explain the safe/unsafe overall decision. LEG is trained using synthetic data for explainability, which is generated using a novel strategy that counteracts the confirmation biases of LLMs. Lastly, LEG's training process uses a novel loss that captures global explanation signals and combines cross-entropy and focal losses with uncertainty-based weighting. LEG obtains equivalent or better performance than the state-of-the-art for both prompt classification and explainability, both in-domain and out-of-domain on three datasets, despite the fact that its model size is considerably smaller than current approaches. If accepted, we will release all models and the annotated dataset publicly. 

---
# Building Safe and Deployable Clinical Natural Language Processing under Temporal Leakage Constraints 

**Authors**: Ha Na Cho, Sairam Sutari, Alexander Lopez, Hansen Bow, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.15852)  

**Abstract**: Clinical natural language processing (NLP) models have shown promise for supporting hospital discharge planning by leveraging narrative clinical documentation. However, note-based models are particularly vulnerable to temporal and lexical leakage, where documentation artifacts encode future clinical decisions and inflate apparent predictive performance. Such behavior poses substantial risks for real-world deployment, where overconfident or temporally invalid predictions can disrupt clinical workflows and compromise patient safety. This study focuses on system-level design choices required to build safe and deployable clinical NLP under temporal leakage constraints. We present a lightweight auditing pipeline that integrates interpretability into the model development process to identify and suppress leakage-prone signals prior to final training. Using next-day discharge prediction after elective spine surgery as a case study, we evaluate how auditing affects predictive behavior, calibration, and safety-relevant trade-offs. Results show that audited models exhibit more conservative and better-calibrated probability estimates, with reduced reliance on discharge-related lexical cues. These findings emphasize that deployment-ready clinical NLP systems should prioritize temporal validity, calibration, and behavioral robustness over optimistic performance. 

---
# Narrative Theory-Driven LLM Methods for Automatic Story Generation and Understanding: A Survey 

**Authors**: David Y. Liu, Aditya Joshi, Paul Dawson  

**Link**: [PDF](https://arxiv.org/pdf/2602.15851)  

**Abstract**: Applications of narrative theories using large language models (LLMs) deliver promising use-cases in automatic story generation and understanding tasks. Our survey examines how natural language processing (NLP) research engages with fields of narrative studies, and proposes a taxonomy for ongoing efforts that reflect established distinctions in narratology. We discover patterns in the following: narrative datasets and tasks, narrative theories and NLP pipeline and methodological trends in prompting and fine-tuning. We highlight how LLMs enable easy connections of NLP pipelines with abstract narrative concepts and opportunities for interdisciplinary collaboration. Challenges remain in attempts to work towards any unified definition or benchmark of narrative related tasks, making model comparison difficult. For future directions, instead of the pursuit of a single, generalised benchmark for 'narrative quality', we believe that progress benefits more from efforts that focus on the following: defining and improving theory-based metrics for individual narrative attributes to incrementally improve model performance; conducting large-scale, theory-driven literary/social/cultural analysis; and creating experiments where outputs can be used to validate or refine narrative theories. This work provides a contextual foundation for more systematic and theoretically informed narrative research in NLP by providing an overview to ongoing research efforts and the broader narrative studies landscape. 

---
# Preference Optimization for Review Question Generation Improves Writing Quality 

**Authors**: Karun Sharma, Vidushee Vats, Shengzhi Li, Yuxiang Wang, Zhongtian Sun, Prayag Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2602.15849)  

**Abstract**: Peer review relies on substantive, evidence-based questions, yet existing LLM-based approaches often generate surface-level queries, drawing over 50\% of their question tokens from a paper's first page. To bridge this gap, we develop IntelliReward, a novel reward model built from a frozen autoregressive LLM with trainable multi-head transformers over the final 50 token states, which outperforms API-based SFT baselines in predicting expert-level human preferences. By applying Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) with IntelliReward, we train IntelliAsk, a question-generation model aligned with human standards of effort, evidence, and grounding. We find consistent improvements on reasoning and writing benchmarks, suggesting reviewer-question quality correlates with broader capabilities. Compared to the Qwen3-32B base model, IntelliAsk shows measurable gains across diverse benchmarks, specifically improving performance on reasoning tasks like MuSR (68.3 vs 64.7 Acc) and complex writing evaluations such as WritingBench (8.31 vs 8.07). We release our implementation, expert preference annotations, and the IntelliReward model to provide an automatic evaluation benchmark for grounding, effort, and evidence in LLM-generated review questions. 

---
# Can LLMs Assess Personality? Validating Conversational AI for Trait Profiling 

**Authors**: Andrius Matšenas, Anet Lello, Tõnis Lees, Hans Peep, Kim Lilii Tamm  

**Link**: [PDF](https://arxiv.org/pdf/2602.15848)  

**Abstract**: This study validates Large Language Models (LLMs) as a dynamic alternative to questionnaire-based personality assessment. Using a within-subjects experiment (N=33), we compared Big Five personality scores derived from guided LLM conversations against the gold-standard IPIP-50 questionnaire, while also measuring user-perceived accuracy. Results indicate moderate convergent validity (r=0.38-0.58), with Conscientiousness, Openness, and Neuroticism scores statistically equivalent between methods. Agreeableness and Extraversion showed significant differences, suggesting trait-specific calibration is needed. Notably, participants rated LLM-generated profiles as equally accurate as traditional questionnaire results. These findings suggest conversational AI offers a promising new approach to traditional psychometrics. 

---
# Do Personality Traits Interfere? Geometric Limitations of Steering in Large Language Models 

**Authors**: Pranav Bhandari, Usman Naseem, Mehwish Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2602.15847)  

**Abstract**: Personality steering in large language models (LLMs) commonly relies on injecting trait-specific steering vectors, implicitly assuming that personality traits can be controlled independently. In this work, we examine whether this assumption holds by analysing the geometric relationships between Big Five personality steering directions. We study steering vectors extracted from two model families (LLaMA-3-8B and Mistral-8B) and apply a range of geometric conditioning schemes, from unconstrained directions to soft and hard orthonormalisation. Our results show that personality steering directions exhibit substantial geometric dependence: steering one trait consistently induces changes in others, even when linear overlap is explicitly removed. While hard orthonormalisation enforces geometric independence, it does not eliminate cross-trait behavioural effects and can reduce steering strength. These findings suggest that personality traits in LLMs occupy a slightly coupled subspace, limiting fully independent trait control. 

---
# Language Model Representations for Efficient Few-Shot Tabular Classification 

**Authors**: Inwon Kang, Parikshit Ram, Yi Zhou, Horst Samulowitz, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2602.15844)  

**Abstract**: The Web is a rich source of structured data in the form of tables, from product catalogs and knowledge bases to scientific datasets. However, the heterogeneity of the structure and semantics of these tables makes it challenging to build a unified method that can effectively leverage the information they contain. Meanwhile, Large language models (LLMs) are becoming an increasingly integral component of web infrastructure for tasks like semantic search. This raises a crucial question: can we leverage these already-deployed LLMs to classify structured data in web-native tables (e.g., product catalogs, knowledge base exports, scientific data portals), avoiding the need for specialized models or extensive retraining? This work investigates a lightweight paradigm, $\textbf{Ta}$ble $\textbf{R}$epresentation with $\textbf{L}$anguage Model~($\textbf{TaRL}$), for few-shot tabular classification that directly utilizes semantic embeddings of individual table rows. We first show that naive application of these embeddings underperforms compared to specialized tabular models. We then demonstrate that their potentials can be unlocked with two key techniques: removing the common component from all embeddings and calibrating the softmax temperature. We show that a simple meta-learner, trained on handcrafted features, can learn to predict an appropriate temperature. This approach achieves performance comparable to state-of-the-art models in low-data regimes ($k \leq 32$) of semantically-rich tables. Our findings demonstrate the viability of reusing existing LLM infrastructure for efficient semantics-driven pathway to reuse existing LLM infrastructure for Web table understanding. 

---
# The Perplexity Paradox: Why Code Compresses Better Than Math in LLM Prompts 

**Authors**: Warren Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2602.15843)  

**Abstract**: In "Compress or Route?" (Johnson, 2026), we found that code generation tolerates aggressive prompt compression (r >= 0.6) while chain-of-thought reasoning degrades gradually. That study was limited to HumanEval (164 problems), left the "perplexity paradox" mechanism unvalidated, and provided no adaptive algorithm. This paper addresses all three gaps. First, we validate across six code benchmarks (HumanEval, MBPP, HumanEval+, MultiPL-E) and four reasoning benchmarks (GSM8K, MATH, ARC-Challenge, MMLU-STEM), confirming the compression threshold generalizes across languages and difficulties. Second, we conduct the first per-token perplexity analysis (n=723 tokens), revealing a "perplexity paradox": code syntax tokens are preserved (high perplexity) while numerical values in math problems are pruned despite being task-critical (low perplexity). Signature injection recovers +34 percentage points in pass rate (5.3% to 39.3%; Cohen's h=0.890). Third, we propose TAAC (Task-Aware Adaptive Compression), achieving 22% cost reduction with 96% quality preservation, outperforming fixed-ratio compression by 7%. MBPP validation (n=1,800 trials) confirms systematic variation: 3.6% at r=0.3 to 54.6% at r=1.0. 

---
# EdgeNav-QE: QLoRA Quantization and Dynamic Early Exit for LAM-based Navigation on Edge Devices 

**Authors**: Mengyun Liu, Shanshan Huang, Jianan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15836)  

**Abstract**: Large Action Models (LAMs) have shown immense potential in autonomous navigation by bridging high-level reasoning with low-level control. However, deploying these multi-billion parameter models on edge devices remains a significant challenge due to memory constraints and latency requirements. In this paper, we propose EdgeNav-QE, a novel framework that integrates Quantized Low-Rank Adaptation (QLoRA) with a dynamic early-exit (DEE) mechanism to optimize LAMs for real-time edge navigation. By quantizing the backbone to 4-bit precision and strategically placing early-exit branches, we enable the model to terminate inference early for simple navigation tasks while retaining full depth for complex decision-making. Experimental results on the Habitat-Sim environment with Matterport3D dataset using OpenVLA-7B backbone, demonstrate that EdgeNav-QE reduces inference latency by 82.7% and memory footprint by 66.7% compared to full-precision baselines, while maintaining 81.8% navigation success rate. Furthermore, it outperforms state-of-the-art static early-exit method by 17.9% in latency, demonstrating the superiority of content-aware adaptive computation for safety-critical applications. 

---
# What Persona Are We Missing? Identifying Unknown Relevant Personas for Faithful User Simulation 

**Authors**: Weiwen Su, Yuhan Zhou, Zihan Wang, Naoki Yoshinaga, Masashi Toyoda  

**Link**: [PDF](https://arxiv.org/pdf/2602.15832)  

**Abstract**: Existing user simulations, where models generate user-like responses in dialogue, often lack verification that sufficient user personas are provided, questioning the validity of the simulations. To address this core concern, this work explores the task of identifying relevant but unknown personas of the simulation target for a given simulation context. We introduce PICQ, a novel dataset of context-aware choice questions, annotated with unknown personas (e.g., ''Is the user price-sensitive?'') that may influence user choices, and propose a multi-faceted evaluation scheme assessing fidelity, influence, and inaccessibility. Our benchmark of leading LLMs reveals a complex ''Fidelity vs. Insight'' dilemma governed by model scale: while influence generally scales with model size, fidelity to human patterns follows an inverted U-shaped curve. We trace this phenomenon to cognitive differences, particularly the human tendency for ''cognitive economy.'' Our work provides the first comprehensive benchmark for this crucial task, offering a new lens for understanding the divergent cognitive models of humans and advanced LLMs. 

---
