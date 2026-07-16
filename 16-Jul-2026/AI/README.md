# Deep Interaction: An Efficient Human-AI Interaction Method for Large Reasoning Models 

**Authors**: Hefeng Zhou, Jinxuan Zhang, Jiong Lou, Yuxin Liu, Chaochao Lu, Jingjing Qu, Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.14049)  

**Abstract**: The emergence of Chain-of-Thought (CoT) reasoning has significantly enhanced the ability of large language models (LLMs) to tackle complex, multi-step tasks. However, when errors occur, current interaction approaches typically involve re-generating another response that may make mistakes again, or users laboriously flag the faulty step in follow-up turns that may get responses <You are right, I made a mistake here> followed by similar errors recurring. To address this issue, we propose an efficient human intervention mechanism for precisely correcting reasoning errors in LLMs, termed Deep Interaction. Our approach enables direct editing of the original response, allowing erroneous parts to be corrected while preserving accurate reasoning steps. We refine the edited CoT into a distilled prompt, which then steers the LLM along the corrected reasoning path. Experimental results show that our method achieves over a 25% improvement in correction success rate and reduces token usage by approximately 40% on STEM tasks reasoning compared to baseline approaches. 

---
# Earthquaker-AI: A Retrieval-Augmented Generation Framework with Rubric-Based Assessment for Primary School Earthquake Education 

**Authors**: Xanthi Kokkinou, Chaido Mizeli, Nafsika Koulaxidou, Marina Delianidi, Konstantinos Diamantaras  

**Link**: [PDF](https://arxiv.org/pdf/2607.14046)  

**Abstract**: This paper presents Earthquaker-AI, a hybrid educational framework building upon a previously implemented educational robotics project by integrating a conversational AI assistant based on Retrieval-Augmented Generation. It aims to enhance earthquake preparedness and conscious action among primary-school students. The system extends the award-winning STEM project Earthquaker moving from mechanical simulation with Lego WeDo2 to cognitive and metacognitive processing. The robotics component uses Lego WeDo2 automation to simulate seismic response, letting students interact with sensors and actuators as tangible representations of protective actions. The assistant operates as a guided learning mechanism aligning student responses with safety guidelines, while providing rubric-based verbal feedback that supports self-regulated learning and calmness under emergency conditions. Earthquaker-AI follows a progressive learning trajectory aligned with cognitive development. In early grades, the focus is on basic recognition of safety actions through multiple-choice questions, assessed via a two-dimensional rubric. In middle grades, students identify correct action sequences through multiple-choice questions, evaluated via a three-axis rubric. In upper grades, the approach shifts to verbal production, requiring short written responses assessed via a four-dimensional rubric that includes clarity of expression. The dialogic module uses RAG to match student queries semantically with official guidelines, generating safe, accurate responses. Experimental evaluation shows high groundedness and accuracy, with a low hallucination rate. Overall, Earthquaker-AI combines hands-on engagement, information processing, and reflective practice. Combining robotics, rubrics, and AI promotes technological literacy, self-regulation, and responsible use of digital systems, contributing to early crisis-management skills. 

---
# AI-accelerated End-to-End Framework for Rapid Professional Upskilling 

**Authors**: Tam Nguyen, Hung Nguyen, Robert Ogburn  

**Link**: [PDF](https://arxiv.org/pdf/2607.14044)  

**Abstract**: By 2030, 59 of every 100 workers will need reskilling or upskilling, yet the average time to close an enterprise skills gap grew from roughly 3 days in 2014 to 36 days in 2018. Most current frameworks accelerate single stages of upskilling programs and generally lack industry validation. We present an end-to-end framework that applies AI acceleration across five stages of knowledge acquisition, content development, content review and verification, teaching, and assessment development; with a strong focus on both production and learning efficiency. Three strong external signals validates the framework: the US National Association of State Boards of Accountancy reviewed and approved an upskilling program built on the framework for continuing-professional-education credits; 3 learners followed the program and passed the NVIDIA Certified Professional in Agentic AI exam in a significantly short amount of time, with 14 more in progress; the program's knowledge base supports complex downstream analysis such as the production of a robust 1,267 risk item dataset for managing multi-agent AI system risks. 

---
# Do Agent Optimizers Compound? A Continual-Learning Evaluation on Terminal-Bench 2.0 

**Authors**: Wenxiao Wang, Priyatham Kattakinda, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2607.14004)  

**Abstract**: Most reported gains from agent-optimization methods are one-shot: an agent is optimized against a fixed benchmark and the resulting improvement is reported as if it were a stable property of the method. This does not test the setting that matters for deployed agents, where optimization is applied recursively as new failures and new tasks appear over time. The central question this raises is whether optimizer-driven gains compound: after an agent has been optimized once, can it be optimized again on newly arrived tasks without eroding the gains the first round produced? We study this question with a two-phase continual-learning evaluation built from hard tasks in Terminal-Bench 2.0, comparing three approaches to agent-harness optimization (GEPA, Meta Harness, and RELAI's Verifiable Continual Learning, RELAI-VCL) under identical optimization budgets. All three methods improve over the baseline agent in the conventional, static, single-phase setting. However, once new tasks are introduced, the methods diverge sharply: GEPA's optimized agent transfers below the unoptimized baseline, Meta Harness transfers well but fails to improve further once given a second optimization budget, and RELAI-VCL is the only method that both transfers positively to unseen tasks and continues improving after those tasks are folded into the optimization objective, reaching the highest pass rate at every evaluated stage and the highest lifelong average pass rate overall (76.4% vs. 66.0% for GEPA, 64.6% for Meta Harness, and 58.7% for the baseline). Our key observation was that optimization gains compounded only when regression control was built into the optimization loop, providing an inductive bias against shortcut solutions that fail to generalize. 

---
# A Self-Evolving Agent for Longitudinal Personal Health Management 

**Authors**: Haoran Li, Jiebi Deng, Tong Jin, Jinghong Han, Yuxin Wang, Zexin Wang, Qingyi Si, Weikang Gong, Xiahai Zhuang, Jia You, Wei Cheng, Jianfeng Feng, Hongcheng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.13940)  

**Abstract**: Personal health management unfolds over repeated encounters, yet most health AI systems treat each request in isolation. We developed HealthClaw, an open-source agent architecture that updates support as a person's routines, preferences, measurements and risks change. It separates shared safety rules and medical knowledge from private longitudinal memory containing profile facts, reusable procedures and episodic traces. After each episode, induction determines what should update the profile, revise a procedure, remain episodic or be excluded. We evaluated HealthClaw with a synthetic year-long benchmark and nine 200-case biomedical tasks. Across 900 longitudinal support probes, answer accuracy increased from 0.2% with current-query prompting to 45.7% with HealthClaw, while prompt-side context exposure was 71.7% lower than with full-history prompting. In 100 privacy probes, HealthClaw produced higher privacy-aware answer quality and fewer unsafe disclosures than both baselines. Across the biomedical tasks, the mean absolute gain in the task-specific primary metric was 27.0 percentage points, and seven gains remained significant after false-discovery-rate correction. These offline benchmarks support governed, self-evolving memory for longitudinal personal health agents, although clinical effectiveness requires prospective evaluation. HealthClaw is publicly available at this https URL. 

---
# AIMO Interpretability Challenge 

**Authors**: Michal Štefánik, Philipp Mondorf, Andreas Waldis, Qianying Liu, Chuan Yang, Michal Spiegel, Josef Kuchař, Marek Kadlčík, Adam Vawda-Oomerjee, Chaoran Liu, Simon Frieder, Barbara Plank, Fazl Barez, Pontus Stenetorp  

**Link**: [PDF](https://arxiv.org/pdf/2607.13899)  

**Abstract**: We propose the AIMO Interpretability Challenge, a competition on distinguishing robust from spurious reasoning in frontier mathematical language models based on the models' internal mechanisms. The challenge is motivated by a central limitation of standard reasoning benchmarks: strong final-answer accuracy does not reveal whether a model relies on stable reasoning mechanisms or exploits brittle reasoning shortcuts. Building on AI Mathematical Olympiad (AIMO) problems and submissions, together with resources from the Fields Model Initiative, the competition will provide (1) newly-published olympiad-level math reasoning problems and their symbolic representations, allowing generation of novel functional variants, (2) access to frontier reasoning models, and (3) assessments of models' adversarial robustness on these problems. Participants will use these resources, along with our computing infrastructure support, to develop methods for identifying which models solve problems robustly. Our competition will also create a new, open robustness benchmark and baseline systems, aiming to provide a lasting foundation for standard benchmarking in mathematical reasoning and interpretability. Scientifically, the competition connects interpretability and generalization research around a central question in AI research: can we determine if, and to what extent, the decision-making of frontier AI models is generalizable and thus, reliable? 

---
# Experience Memory Graph: One-Shot Error Correction for Agents 

**Authors**: Wenjun Wang, Yuchen Fang, Fengrui Liu, Zibo Liang, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.13884)  

**Abstract**: Large Language Model (LLM) agents have shown remarkable capabilities in autonomous decision-making by generating sequential trajectories of states, actions, and observations. However, in complex, long-horizon tasks, these agents frequently suffer from compounding errors and struggle to recover from failures. Existing self-correction mechanisms rely on prompt-based reflection, which is inherently brittle, incurs heavy time and API costs due to iterative trial-and-error loops, and produces task-specific memory that may be hard to generalize to new scenarios. To address this, we propose Experience Memory Graph (EMG), a framework that reformulates agent failure recovery as a graph matching problem. At training time, we convert both failed exploration trajectories and successful expert trajectories into directed action decision graphs. By matching these graphs, we extract common subgraphs (successful workflows) and graph edit paths that explicitly indicate how to correct failures (e.g., which actions to add, delete, or relabel under a given observation), and store them in a memory graph with intra-task nodes and cross-task edges. At test time, EMG retrieves relevant insights and guides the agent in a single, loop-free execution. Experiments on ALFWorld and ScienceWorld show that EMG consistently outperforms state-of-the-art reflection baselines in success rate and average reward, while requiring no test-time trial-and-error. 

---
# CAVA: Canonical Action Verification and Attestation for Runtime Governance of Agentic AI Systems 

**Authors**: Zexun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13716)  

**Abstract**: Agentic AI systems increasingly act through heterogeneous runtimes: local coding hooks, SDK tools, browser automation, managed-agent traces, API gateways, and workflow engines. A single operational act such as publishing code, changing identity state, moving money, or exporting data may therefore be represented by many incompatible runtime records. This makes a basic governance question difficult to answer: what action was actually approved, what evidence binds the approval to execution, and can an independent verifier reproduce the same action identity later?
This paper presents Canonical Action Verification and Attestation (CAVA), a runtime-semantics layer for converting heterogeneous agent activity into canonical runtime action objects. CAVA is positioned below Proof-Carrying Agent Actions (PCAA): PCAA defines the deployer-owned route-review-prove governance process, while CAVA defines the stable action object that process governs. The paper formalizes canonical action identity, semantic pattern detection, approval binding, receipt integrity, runtime-portable projection, and optional attestation substrates. We study a reference implementation through a 96-seed, 384-variant benchmark covering semantic equivalence, semantic separation, wrapper bypass, false-positive control, approval binding, receipt reproducibility, attestation tamper detection, runtime portability, semantic pattern detection, policy degradation, and Azure deployment drills. The contribution is a systems formulation of action-level canonicalization and policy-addressable semantic patterns as a necessary substrate for deployer-side AI governance. 

---
# AgentCompass: A Unified Evaluation Infrastructure for Agent Capabilities 

**Authors**: Zichen Ding, Jiaye Ge, Shufan Jiang, Kai Chen, Mo Li, Qingqiu Li, Zehao Li, Zonglin Li, Tiaohao Liang, Shudong Liu, Zerun Ma, Zixing Shang, Wenhui Tian, Zun Wang, Liwei Wu, Zhenyu Wu, Jun Xu, Bowen Yang, Dingbo Yuan, Qi Zhang, Songyang Zhang, Peiheng Zhou, Dongsheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.13705)  

**Abstract**: As Large Language Models (LLMs) evolve into autonomous agents, the need for unified evaluation infrastructure becomes critical. However, current evaluation pipelines remain highly fragmented and tightly coupled, hindering reproducibility and causing redundant engineering. To address this, we introduce AgentCompass, an open-source, lightweight, and extensible infrastructure for evaluating LLM-based agents. AgentCompass organizes the evaluation process around three independent components, namely Benchmark, Harness, and Environment, thereby enabling flexible configurations without requiring the reimplementation of complex execution logic. Furthermore, it features a fault-tolerant asynchronous runtime and comprehensive trajectory analysis tools to transparently diagnose nuanced failure modes like reward-hacking. Natively supporting over 20 benchmarks across five capability dimensions, AgentCompass provides the community with a scalable and reproducible infrastructure for advancing agent research. 

---
# When Bots Join the Team: Bot Adoption and the Institutional Fabric of Open-Source Software Projects 

**Authors**: Yongren Shi, Wenyi Gong  

**Link**: [PDF](https://arxiv.org/pdf/2607.13679)  

**Abstract**: AI agents are joining human teams, raising a basic question: when an automated agent becomes a regular participant, does group organization strengthen or weaken? We study this question in open-source software, where bots open pull requests, review code, and merge changes alongside people, leaving a public record of every interaction. Treating bots as participants rather than tools, we examine 2,991 GitHub projects for two years before and after each adopted its first bot. We measure three capabilities that institutional theory links to durable coordination - repeated engagement, social memory, and role differentiation - and two outcomes: conflict cascades and output distinctiveness. Bot adoption is followed by more repeated collaboration, greater recognition of specific bots in discussion, fewer conflict cascades, and more distinctive outputs. These changes cluster around adoption rather than accumulating gradually. Because we lack an untreated comparison group, we interpret the results as precisely timed associations, not causal effects. Two patterns are difficult for alternative explanations to account for: capabilities predict outcomes according to their function - coordination versus differentiation - rather than whether humans or bots provide them, and human-side capabilities account for the bot-conflict association but not the bot-distinctiveness association. The findings are consistent with a specific interpretation: predictable, rule-based agents can become part of a community's social infrastructure. The bot is the occasion; social organization is the mechanism. 

---
# Explaining Reinforcement Learning Agents via Inductive Logic Programming 

**Authors**: Celeste Veronese, Edoardo Zorzi, Daniele Meli, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2607.13655)  

**Abstract**: Explainable Reinforcement Learning (XRL) seeks to make Reinforcement Learning (RL) policies more transparent and interpretable, a key requirement in safety-critical and human-centric scenarios. However, it is mostly based on user studies, thus targeting the needs of a specific audience and lacking shared evaluation metrics. On the other hand, logic-based approaches within eXplainable Artificial Intelligence (XAI) provide compact, human-readable abstractions of decision-making. However, the systematic quantification of the explainability degree of logical representations remains an open problem. This work aims to advance the state of the art in XRL by introducing objective and planning-oriented metrics for policy explainability in RL settings. At the same time, it contributes to the field of logic for XAI by providing a principled way to quantify the explainability of logical rules, moving beyond common-sense assessments and simple propositional fragments. We employ Inductive Logic Programming (ILP) to extract symbolic representations of RL policies and define a novel set of explainability metrics, including activation rate, feature coverage, syntactic distance and semantic distance. These metrics quantify alignment between symbolic rules and agent behavior, the role of features in decision-making, and the evolution of policies during training and across agents in single and multi-agent RL. Experiments across different RL domains show that the proposed metrics highlight action-specific learning dynamics beyond global return, provide fine-grained insights into domain features beyond classical approaches for global feature importance estimation, and uncover coordination, specialization, and adaptation patterns in MARL. Moreover, they provide crucial insights for the transfer and generalization of action-specific policies. 

---
# UESF-Bench: Benchmarking and Probing for Unified Embodied Seeking and Following 

**Authors**: Kun Yu, Jianhua Yang, Yixiang Chen, Changwei Wang, Hongyuan Yu, Yan Huang, Fushuo Huo, Ya Jing, Zhumin Chen, Keji He  

**Link**: [PDF](https://arxiv.org/pdf/2607.13621)  

**Abstract**: Language-guided human following is an important capability for embodied agents, but existing benchmarks typically assume that the target person is visible at the start of an episode. This setting simplifies the problem and overlooks a more realistic requirement: an agent often needs to first find a language-described target and then persistently follow that target in a dynamic environment. While recent work has started to study human search, existing settings are typically evaluated in task-specific scenarios and often rely on stronger prior knowledge of the environment. Moreover, they usually treat searching and following as separate tasks and still lack a unified benchmark for systematic evaluation. To address these limitations, we introduce the Unified Embodied Seeking and Following Benchmark (UESF-Bench), a large-scale and diverse benchmark for embodied human seeking and following. The benchmark requires agents to handle semantic-guided exploration, reliable behavior switching and recovery, and delayed identity grounding. To this end, we propose SeekFollow-VLA, a vision-language-action framework with a task-driven routing mechanism for latent phase inference and transition modeling between seeking and following. Experimental results show that SeekFollow-VLA achieves clear improvements over both single-head and dual-head baselines across single-person and multi-person environments, establishing a baseline for unified embodied seek-and-follow. 

---
# STOCKTAKE: Measuring the Gap Between Perception and Action in LLM Agents with a Fair Oracle 

**Authors**: Sagar Deb, Ashwanth Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2607.13618)  

**Abstract**: LLM agents are increasingly evaluated on multi-week decision tasks in which the state that drives cost is never directly observed. On such tasks the final cost cannot say why an agent failed: it may have misread the world, or read it correctly and still failed to act (the knowing-doing gap). Existing evaluations cannot separate these two failures; their reference policies either read privileged information the agent never sees, or are missing altogether. We introduce STOCKTAKE, a 26-week supply-chain replenishment benchmark built as a factored partially observable Markov decision process with six hidden factor processes, designed so that a fair reference policy is computable: an exact Bayes filter per factor drives a rollout policy on the identical observation stream the agent receives. Scoring each run between a symptom-blind base-stock floor (0) and this oracle (1) yields a skill score, and grading each week's written rationale yields a stated-belief detection lag and a knowing-doing rate, so state estimation and control are measured separately. On fifty seeds with curated stress profiles, Claude Sonnet 5, GPT-5.4, DeepSeek-V4-Pro, and Grok 4.5 detect 84-88% of hidden failures, typically within a week of onset, yet span skill scores from 0.62 to -0.23: two of the four end below the symptom-blind floor while naming factors slightly faster than the two that beat it. The failure has two faces. Where stress persists, 34-43% of correctly diagnosed stress weeks still end in stockout for every model, a rate that partly reflects the severity of the weeks models notice. That rate also runs opposite to skill: the two models under the floor stock out least on diagnosed weeks, so under-response is only one face of the gap, and their traces point to the other, responses whose cost exceeds what they protect. STOCKTAKE measures both directions of that failure. 

---
# Automatic Ordinary Differential Equations Discovery For Biological Systems Using Large Language Model Powered Agentic System 

**Authors**: David Krongauz, Arad Zulti, Eran Segal, Teddy Lazebnik  

**Link**: [PDF](https://arxiv.org/pdf/2607.13608)  

**Abstract**: Automatic scientific discovery has long been a goal of computational scholars - a machine that can discover nature's secrets on its own, moving computational systems beyond data-fitting tools toward the generation and refinement of mechanistic models of the universe. Recent advances in symbolic regression (SR) and large-language-model (LLM)-based agents suggest that such systems can recover equations from data, incorporate domain priors, and automate parts of the research workflow. However, most existing approaches either focus on narrow equation-discovery benchmarks or broad end-to-end automation pipelines, while biological systems remain comparatively underexplored. Here, we introduce the MEDA system, an LLM- and SR-powered agentic framework for discovering ordinary-differential-equation (ODE) models of biological and biologically inspired dynamical systems. MEDA retrieves background knowledge, defines admissible variables, generates mechanistic constraints, proposes candidate ODEs, and fits and evaluates them. We evaluate it across canonical model retrieval, reasoning-based extrapolation to unseen variants, and open-ended discovery, with and without experimental data. Across these settings, MEDA recovered the correct state variables, achieved strong structural recovery in retrieval and extrapolation tasks, and produced biologically plausible discovery-oriented models. Ablation and robustness analyses show that knowledge-guided formalization and mechanistic constraints are load-bearing components, whereas numerical fitting alone can preserve trajectory-compatible but biologically incorrect equations. 

---
# SAFETY SENTRY: Context-Aware Human Intervention via EXECUTE-ASK-REFUSE Routing 

**Authors**: Tianyu Chen, Chujia Hu, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13594)  

**Abstract**: LLM agents act on real-world environments through tool calls, and a single misjudged action can cause irreversible harm. The standard safeguard is a guard model that labels each proposed action as safe or unsafe, but this binary view conflates two distinct decisions: whether the action is harmful in itself, and whether it is appropriate given the user's context. It also operates at the granularity of action categories rather than individual instances, producing routine interruptions that erode autonomy and train users to wave through the most consequential alerts. We reframe the problem as a per-instance three-way routing decision over {EXECUTE, ASK, REFUSE} and instantiate it with Safety Sentry, a lightweight guard model whose inference reduces to a single decoding call. A single decoding-time threshold lets one fixed checkpoint be re-positioned across deployments of differing risk tolerance without retraining. Safety Sentry outperforms a broad set of open-weight and frontier closed-source baselines on overall accuracy and safety-related recall, while controlling both directional error rates simultaneously. 

---
# AI advice suppresses people's willingness to say "I don't know", even when the advice is wrong and accuracy is incentivized 

**Authors**: Chiara Marcoccia, Walter Quattrociocchi, Valerio Capraro  

**Link**: [PDF](https://arxiv.org/pdf/2607.13562)  

**Abstract**: Knowing when to say "I don't know" is fundamental to human judgment, yet AI assistants offer a fluent answer to almost any question. In five experiments (N = 3,132; four preregistered, one direct replication), participants answered difficult questions and could always decline to respond. We engineered the questions so that AI advice was wrong, separating AI use from its accuracy. Merely having access to AI nearly eliminated participants' willingness to suspend judgment, and this held whether the advice was actively requested or simply displayed. Consequently, participants answered more questions but were correct about a third as often as when AI was unavailable-yet their confidence nearly doubled. Incentivizing accuracy and penalizing inaccuracy led participants to seek and follow AI advice less, answer more accurately, and suspend judgment more often, though still far less than when AI was unavailable. As AI suggestions grow ubiquitous and unsolicited, they may not simply affect answer accuracy; they may even alter the metacognitive threshold at which people decide whether they know enough to answer. 

---
# Multi-Agent Collaborative Reasoning with Tool-Augmented Evidence for Urban Region Profiling 

**Authors**: Xixuan Hao, Yutian Jiang, Jiabo Liu, Yihang Yang, Guangyin Jin, Song Gao, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13558)  

**Abstract**: Urban region profiling constitutes a core problem in urban computing, supporting applications such as population estimation, economic assessment, and environmental monitoring. Existing methods typically formulate this task as multimodal representation learning, fusing heterogeneous urban data, e.g., satellite imagery, points of interest, textual descriptions, and 3D building information, into latent embeddings for prediction. However, these approaches are largely correlation-driven, assume cross-modal consistency, and rely on static pipelines, which limit their robustness in heterogeneous or unseen urban regions. We propose UrbanAgent, an agentic framework that reframes urban region profiling as a reasoning-driven inference problem. UrbanAgent instantiates an independent agent for each data modality and performs structured multi-agent collaborative reasoning to explicitly address cross-modal inconsistencies rather than absorbing them into a single representation. In addition, UrbanAgent extends indicator prediction as a closed-loop process of active evidence acquisition and iterative reasoning, enabling agents to verify uncertain inferences through tool-augmented retrieval of external knowledge optimized via reinforcement learning. Extensive experiments on global urban datasets for Carbon emissions, GDP, and Population estimation show that UrbanAgent consistently outperforms existing baselines, achieving an average improvement of 8.1% in R2, and exhibiting strong generalization performance in unseen-city settings. 

---
# How Far Can Root Cause Analysis Go on Real-World Telemetry Data? 

**Authors**: Athira Gopal, Ashwanth Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2607.13548)  

**Abstract**: Identifying root causes in production microservice failures requires reasoning over large-scale, multimodal telemetry spanning metrics, logs, and traces, a problem that has proved resistant to both classical and LLM-based approaches. The OpenRCA dataset exemplifies these challenges: it is large-scale, multimodal, and lacks detailed domain knowledge, and yields consistently low accuracy across all existing methods. We show that classical causal discovery methods and existing LLM-based multi-agent systems fail to reliably identify root causes on this benchmark, and present a Structured Multi-Agent RCA pipeline that substantially outperforms existing LLM-based and classical baselines, supporting both domain-knowledge and knowledge-free operating modes. To diagnose where failures originate, we introduce a reverse reasoning agent that, given the correct answer, identifies which signals in the extracted anomalies support it and determines whether Stage~1 had access to those signals, classifying each failure as Reasoning Gap (evidence present but unused) or Data Ambiguity (evidence genuinely absent). This analysis reveals that the required evidence is present in the vast majority of failures: the bottleneck is not data access but the agent's ability to reason over it correctly. We further introduce an automated rule mining pipeline that systematically extracts discrimination rules from reverse reasoning reports, reducing reliance on manual knowledge curation. Across all configurations, model reasoning capability and domain knowledge are the primary constraints: stronger models embed more domain expertise, and explicit knowledge injection partially compensates for this gap. Reasoning performance remains practically bounded even when evidence extraction is perfect: scaffold engineering and better data pipelines alone cannot close this gap; progress requires improvements at the model level. 

---
# LAPO: Leave-One-Turn Attribution for Self-Generated Process Rewards in Multi-Turn Search Reasoning 

**Authors**: Qiang Zhu, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.13501)  

**Abstract**: Reinforcement learning for multi-turn search reasoning typically relies on terminal outcome rewards, which cannot distinguish useful, redundant, and harmful intermediate interactions. We propose LAPO, a self-generated process-supervision method based on backward leave-one-turn attribution. For each search turn, LAPO replaces the turn and its retrieval observation with a fixed [DELETE] placeholder and measures the resulting change in the current policy's mean log-likelihood of the gold answer. This Answer-Likelihood Gain estimates the turn's contribution while preserving all downstream interactions, allowing early evidence to be evaluated in the complete reasoning context. LAPO further applies sign-consistency gating, retaining only normalized process advantages whose directions agree with their raw attribution scores. The method requires no additional reward model, teacher, verifier, or LLM-as-a-Judge. Across seven knowledge-intensive question-answering datasets with local retrieval, LAPO achieves an average exact-match score of 0.326, outperforming the strongest step-reward baseline, IGPO, by 0.053. Ablations show complementary benefits from backward attribution and sign-consistency gating, demonstrating that policy-derived retrospective attribution can provide effective process supervision for multi-turn search agents. 

---
# Set-shifting Behavioral Test for Harnessed Agents 

**Authors**: Ziwei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2607.13396)  

**Abstract**: What happens to an LLM agent's tool choice when the reliable tool silently changes within an ongoing session? We borrow set-shifting from cognitive psychology to study how well agents adapt to hidden reliability shifts. Our benchmark mounts tool-skill libraries with redundancies, where many tools solve the same task but differ in hidden reliability. In our evaluation framework, a branched schedule shifts the reliable tool group at hidden boundaries and pairs every shift with a no-shift control. We find that agents, by default, settle on a small recurring routine within a few turns of each boundary, with call shares concentrating on a few discrete values after each reliability shift. We score the set-shifting accuracy for each agent trajectory: the joint probability of routing to the target tool group in every post-shift window. We test open-weight LLMs in an open-source agentic harness and find qualitatively distinct failure modes across the same set of routines. We also find that set framing, how the toolset presents the alternatives as competing or complementary, shifts the routing dynamics. 

---
# EZSMT Version 3, Matured 

**Authors**: Yuliya Lierler  

**Link**: [PDF](https://arxiv.org/pdf/2607.13344)  

**Abstract**: Constraint Answer Set Programming (CASP) is a hybrid reasoning paradigm that combines Answer Set Programming (ASP) with Constraint Processing and Satisfiability Modulo Theories (SMT), enabling powerful declarative encodings of complex combinatorial search problems. This paper presents the design and implementation of EZSMTV3, an extensible SMT-based CASP framework that advances the translational approach to CASP solving. Building upon the foundation of the EZSMT+ system, EZSMTV3 introduces a more expressive input language, supports optimization via weak constraints, and offers foundations for streamlined integration of new constraint types. Rather than implementing custom search procedures, EZSMTV3 leverages state-of-the-art SMT solvers, such as CVC5, YICES, and Z3 to perform reasoning. The paper provides benchmarking results comparing EZSMTV3 with its CASP peers such as CLINGCON, CLINGO[DL], and CLINGO[LP], while showcasing its ability to handle mixed-domain constraints involving both integers and reals. The system provides a robust platform for future extensions and theoretical exploration within the CASP domain. 

---
# Theory-Level Autoformalization: From Isolated Statements to Unified Formal Knowledge Bases 

**Authors**: Marcus J. Min, Mike He, Zhaoyu Li, Zixuan Yi, Sharad Malik, Aarti Gupta, Xujie Si, Osbert Bastani  

**Link**: [PDF](https://arxiv.org/pdf/2607.13292)  

**Abstract**: Autoformalization translates informal natural language into formal, machine-verifiable languages. While most work focuses on individual statements, real formalization efforts are inherently theory-level: they require an entire web of axioms, definitions, and lemmas before target theorems can even be stated. In this position paper, we argue for theory-level autoformalization: formalizing complete theories, including all their inter-dependencies, as structured libraries. We examine the significance of this shift, address alternative views, identify open challenges, and propose three promising paths forward. Our survey of autoformalization is available at this https URL. 

---
# Harness Handbook: Making Evolving Agent Harnesses Readable,Navigable, and Editable 

**Authors**: Ruhan Wang, Yucheng Shi, Zongxia Li, Zhongzhi Li, Yue Yu, Junyao Yang, Kishan Panaganti, Haitao Mi, Dongruo Zhou, Leoweiliang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13285)  

**Abstract**: The capability of a modern AI agent depends not only on its foundation model but also on its harness, which constructs prompts, manages state, invokes tools, and coordinates execution. As models, APIs, environments, and requirements evolve, the harness must be continually modified. Before such a change can be made, a developer or coding agent must identify all code locations that implement the target behavior. This is difficult because production harnesses are large, tightly coupled, and behaviorally distributed, while modification requests describe what the system should do and repositories are organized by files and modules. Code search, repository indexing, and long-context processing ease inspection, but still leave this behavior-to-code mapping to be recovered by hand. Behavior localization is therefore a central bottleneck in harness evolution. We introduce the Harness Handbook, a behavior-centric representation synthesized automatically from a harness codebase via static analysis and LLM-assisted structuring, linking each behavior to its corresponding source. We also introduce Behavior-Guided Progressive Disclosure (BGPD), which guides agents from high-level behaviors to relevant implementation details and verifies candidate locations against the current source. On diverse modification requests from two open-source harnesses, Handbook-Assisted planning improves behavior localization and edit-plan quality while using fewer planner tokens, with the largest gains on scattered sites, rarely executed paths, and cross-module interactions. Evolving complex agentic systems thus depends not only on generating edits, but also on determining where those edits should be made. 

---
# Cost-Optimal Foundation Model Deployment Portfolio for Transportation Management 

**Authors**: Xi Cheng, Ke Liu, Siyuan Feng, Jane Lin, H. Oliver Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.13239)  

**Abstract**: Foundation models, including large language models (LLMs) and vision-language models (VLMs), are increasingly used for transportation management center (TMC) tasks such as anomaly detection, incident reporting, and traveler information. Deploying multiple such models across TMC functions raises a portfolio question: which model should serve each function, in which deployment mode, and under what shared hardware budget? We formulate this as the Foundation Model Deployment Portfolio (FMDP) problem, a mixed-integer program minimizing total cost of ownership (TCO) subject to per-function quality, latency, and safety constraints over shared GPU capacity. We prove the problem NP-hard by reduction from the 0-1 knapsack problem and propose a polynomial-time greedy heuristic. In an illustrative case study with five TMC functions and 19 candidate (model, mode) pairs, FMDP identifies a mixed portfolio costing $34/mo (97% below the cheapest feasible all-closed-API baseline) by routing four functions to open-source APIs and the one function whose quality floor no open-source model meets to a closed API. Break-even analysis shows that on-premise GPU investment becomes reasonable only above approximately 309 vision queries/hour or if API prices double. 

---
# AI-Native Insurance for Agentic AI: Pricing, Underwriting, and End-to-End Automation 

**Authors**: Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.13230)  

**Abstract**: Agentic AI introduces new insurance challenges because autonomous AI systems can make decisions, invoke tools, modify external environments, and interact with third-party services. This paper develops an AI-native mathematical framework for underwriting, pricing, and contract design for agentic AI deployments. A deployment is represented by a risk state that captures autonomy level, operational authority, permission exposure, governance maturity, and dependency concentration. The framework maps the risk state to event probabilities, loss severities, governance costs, premiums, deductibles, coverage allocation, and policy covenants, and formulates an optimization problem for insurance contract design under participation, profitability, and incentive compatibility constraints. The paper establishes structural properties of insurability, including characterization of an insurability region, monotone deterioration of feasibility with increasing exposure, and governance certification thresholds. Insurance is further interpreted as both an operational cost and a regulatory mechanism for AI deployment. A healthcare case study illustrates contract optimization, sensitivity analysis, and automated claims processing for agentic AI systems. 

---
# Networked Intelligence: Active Shared Context Graphs for Human-AI Team Science 

**Authors**: Sutanay Choudhury, Jeffrey J. Czajka, Lummy M. O. Monteiro, Erin Bredeweg, Jason McDermott, Katherine Wolf, Alex Beliaev, Josh Elmore, Paul Piehowski, Kylee Tate, Yuqian Gao, Aivett Bilbao, Kelly Stratton, Scott Baker, Jaydeep P. Bardhan, Kristin Burnum Johnson, Chris Oehmen, Robert Rallo  

**Link**: [PDF](https://arxiv.org/pdf/2607.13220)  

**Abstract**: Most AI-for-science systems focus on scaling a single reasoning process through better models, larger context windows, long-horizon agentic execution, or digital co-scientists working with one principal user. However, challenging scientific problems are rarely solved by one reasoner alone. They are solved by teams whose members bring different priors, experimental backgrounds, tacit knowledge, and domain-trained intuitions. The open problem is therefore not only how to scale models, but how to cultivate networked intelligence: scaling the connections between humans and AI systems so that a result or hypothesis produced in one context reaches another person, agent, instrument, or robot that can act on it. We introduce Mycelium, an active shared workspace that automatically connects researchers and AI agents as a multi-user co-scientist. As human users and agents work, the system captures important observations and hypotheses, tracks how they relate to the team's evolving model, and routes them to the person or agent whose next decision they can inform. We evaluate Mycelium in its first empirical test, a biological multi-omics campaign in which routed shared context turned a local analytical finding into a cross-expert mechanistic constraint and ultimately into an experimental design. We also give networked intelligence a computational account as sparse conditional computation over distributed scientific contexts. This account distinguishes when a scaled standalone agent can match the network from when independent expertise and non-mergeable contexts make the network irreducible. 

---
# CayleyR: Solving the TopSpin puzzle via cycle intersection 

**Authors**: Yuri Baramykov  

**Link**: [PDF](https://arxiv.org/pdf/2607.13219)  

**Abstract**: We present cayleyR, an R package for solving permutation puzzles by detecting cycle intersections in Cayley graphs. The core algorithm performs an iterative bidirectional search: from both the initial and target permutation states, random operation sequences generate cycles in the Cayley graph of the symmetric group Sn; their intersection yields a connecting path. When no direct intersection is found, a distance-guided bridge selection narrows the gap, and the process repeats. The package targets the TopSpin(n,k) puzzle, whose state space is a Cayley graph of Sn generated by a cyclic shift and a prefix reversal. We describe the mathematical framework, the algorithm, and its implementation, which combines a C++ hash-indexed state store with optional Vulkan GPU acceleration. The software is publicly available on CRAN. 

---
# Learning Safe Agent Behaviour from Human Preferences and Justifications via World Models 

**Authors**: Ilias Kazantzidis, Timothy J. Norman, Yali Du, Christopher T. Freeman  

**Link**: [PDF](https://arxiv.org/pdf/2607.13172)  

**Abstract**: We address the problem of safely training an agent policy and deploying a good and safe policy, in settings where the environment dynamics are unknown and no suitable reward function is available. In the context of safety-critical environments, we consider traditional reinforcement learning impractical and resort to the resource of human input. We introduce DROPJ, a human-centred method for both safe training and deployment. We first learn a world model (a learned simulator) from a dataset of prior real-world trajectories. A human then plays the game in this learned simulator to extract several informative simulated trajectories. From these, we sample pairs of simulated trajectory segments and elicit from a human their preference over these segments, as well as a reason (justification) for their choice. We then train a reward model from these justified preferences and use it, together with the world model, to directly deploy the agent using model predictive control. Running real-user experiments, we find that generating informative simulated trajectories from a user significantly reduces the computational cost during training compared to other strategies, and can also improve the performance during deployment. In the context of training within a learned simulator, we show that the use of preferences rather than other types of feedback substantially improves the performance during deployment. We further demonstrate that safety justifications accompanying preferences can significantly enhance safety or prioritise user-prescribed aspects of safety associated with them during deployment. 

---
# Oracle Agent Memory as an Enterprise Memory Substrate for Long-Horizon AI Agents 

**Authors**: Richmond Alake, Cesare Bernardis, Paul Cayet, Luca Engel, Damien Hilloulin, Sungpack Hong, Allen Hosler, Nickolas Kavantzas, Ingo Kossyk, Son Le, Rhicheek Patra, Kartik Talamadupula, Valentin Venzin  

**Link**: [PDF](https://arxiv.org/pdf/2607.13157)  

**Abstract**: Agent memory is a systems problem for long-horizon agents. Practical deployments require retention of task state across extended conversations, recovery of user-specific facts and preferences across sessions, and accumulation of procedural knowledge from prior outcomes. These requirements extend beyond document retrieval: a memory layer must determine which interactions become durable state, how that state is scoped, how it is retrieved under latency constraints, and how it is revised or removed over time. This report studies Oracle Agent Memory as a database-native memory substrate built on Oracle Database. Three themes organize the discussion: memory as a lifecycle spanning ingestion, extraction, consolidation, retrieval, summarization, and revision or removal; a layered architecture that separates an active memory core from a passive memory-store interface with explicit scope control across users, agents, and threads; and evaluation methodology in which downstream task accuracy is complemented by memory-centric measures such as evidence retrieval, recall, latency, and estimated token use. The report summarizes LongMemEval results, reaching 93.8% accuracy, compares Oracle Agent Memory against flat-history baselines, using about 10.7x fewer tokens, and published or reported external baselines where available, and closes with implementation-oriented appendix material covering setup, thread lifecycle, and search semantics. 

---
# Improving Molecular Property Prediction in Small Language Models Using Graph-based Tools 

**Authors**: Konstantinos Bougiatiotis, Dimitrios Kelesis, Georgios Paliouras  

**Link**: [PDF](https://arxiv.org/pdf/2607.13115)  

**Abstract**: Small language models (SLMs) have shown promise for zero-shot molecular property prediction from SMILES strings, yet they often suffer from structural blindness because sequence representations under-specify key graph-topological cues. We propose a modular Context-Augmented Prompting framework that enables agentic tool use at inference time: a trained GNN expert model provides a predictive hint with confidence, and a GNN extracts an instance-specific explanatory subgraph (e.g., a subgraph SMILES and an accompanying explanatory paragraph). We evaluate three commonly used SLMs on MUTAG and Tox21 under five prompting configurations ranging from SMILES-only to using all available tools at hand. Across two datasets, enriching prompts with graph-derived context yields substantial accuracy gains, often exceeding 25% relative improvement and up to 74% on Tox21. We further validate the functional relevance of the extracted motifs via a necessity-based edge-drop intervention. Despite the observed gains, a persistent gap remains to specialized GNN models, highlighting both the value and limits of text-conditioned reasoning for molecular structure. 

---
# Self-Improvements in Modern Agentic Systems: A Survey 

**Authors**: Zhe Ren, Yimeng Chen, Dandan Guo, Guowei Rong, Tonghui Li, R. B. Xiong, Qingfeng Lan, Wenyi Wang, Li Nanbo, Yibo Yang, Mingchen Zhuge, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2607.13104)  

**Abstract**: Self-improving autonomous agents are moving from research prototypes to deployed systems. The primary goal is controllable evolution, or adaptation, from experience with minimal or even no human input. This survey frames modern self-improving agents as adaptive systems that convert experience into accumulated capability gains. We offer a system-level framework that represents a modern agent as a configuration coupling a foundation model with an operational scaffold of prompts, memory, tools, and control logic. Within this framework, self-improvement is formalized as a self-induced update operator that obtains and commits updates to model parameters or scaffold components. We organize prior work by update target and by the signals that drive change, then review applications and discuss evaluation, before closing with open problems and future directions. For convenience, we track technical updates on this https URL. 

---
# Probabilistic Extension of Neuro-Symbolic AGI Robots based on Belnap's Typed Intensional FOL 

**Authors**: Zoran Majkic  

**Link**: [PDF](https://arxiv.org/pdf/2607.13073)  

**Abstract**: Neuro-symbolic AI based on $IFOL_B$ is a way to combine neural learning and symbolic reasoning to overcome limitations of purely neural systems (like lack of interpretability and logical structure) with formal logical machinery for self-reference. In this paper we expand the cognitive power of $IFOL_B$ by using the probability computation for the currently unknown sentences, based on Nilsson's probability structure for the $IFOL_B$. We introduce the global symmetry transformation that preserves the current knowledge database and logical deduction, and the local one used for real-time decisions about concrete (sub)problems that involve only a very strict subset of $IFOL_B$ predicates. The computation of probability density function $KI$ in both cases, based on the Shannon's maximum information entropy, is provided by neural networks of this probabilistic neuro-symbolic AGI. 

---
# Interventional Grounding Audits: Black-Box Premise-Dependency Tests for LLM Chain-of-Thought via Predicate Substitution 

**Authors**: Hironao Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2607.13069)  

**Abstract**: Large language models produce chain-of-thought (CoT) reasoning that appears logically sound yet may not genuinely depend on its stated premises. We introduce interventional grounding audits, a black-box, step-level test of premise dependency: we intervene on a single premise by substituting its target predicate with a fresh symbol, re-run the model, and check whether each reasoning step's normalized conclusion (canonical predicate form) changes. We evaluate on ProntoQA, a synthetic multi-hop deductive reasoning benchmark with gold proof trees, where step-level premise dependencies are known. Applied to 50 ProntoQA problems with GPT-4o, our method achieves F1 = 0.806 on detecting proof-tree dependencies (F1 = 0.885 on predicate-determining dependencies; Recall = 100%), significantly outperforming a self-consistency baseline (F1 = 0.343; 95% bootstrap CIs non-overlapping). We further identify that 66% of correctly-solved problems contain at least one aligned step insensitive to a direct proof-tree dependency under consistent substitution -- all involving entity-introduction premises, a documented blind spot of the consistent-substitution evaluator -- a "right answer, wrong reasoning" signal invisible to passive methods. All audit certificates, raw outputs, and reproduction scripts are available in a public GitHub repository, and we discuss scope limits beyond formal, parsable benchmarks. 

---
# SPINE: Bridging the Cyber-Physical Gap with Agentic AI 

**Authors**: Minkyu Ham, Dongho Kim, Chan Lee, Jiayi Wang, Min Jun Kim, Yixi Zhang, Guo Ye, Jihai Zhao, Soyeon Park, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.13049)  

**Abstract**: Foundation models have given robots a sophisticated brain for complex decision-making, yet deploying that intelligence into a physical platform still demands tedious, expert-driven calibration. This deployment gap, the robot's spinal cord, remains a primary bottleneck to scalable Embodied AI. Hence, we propose SPINE (Scalable Physical Integration with ageNtic Expertise): an agentic framework for systematically debugging and deploying bimanual robots with minimal robotics expertise. SPINE's harness comprises two orchestrated multi-agent workflows: a profile builder that creates robot-specific context, and a debugger that cycles through diagnosis, repair, and validation until teleoperation works. Across seven DOBOT X-Trainer debugging scenarios, a robotics novice using SPINE outperformed human operators using Claude Code with the same reference materials, but without SPINE's structured workflow, improving operationalization success from 75% to 100% and reducing mean time-to-teleoperation from 16 min 45 s to 13 min 47 s. On AgileX PiPER, a distinct ROS/CAN bimanual arm, SPINE resolved all 10 implanted bugs, versus 9 out of 10 for the expert baseline, in nearly the same amount of time. Together, these results show that SPINE can transfer across bimanual platforms, reduce dependence on expert calibration, and move embodied AI closer to scalable real-world deployment. 

---
# OriginBlame: Record- and Token-Level Data Provenance for AI Training Datasets 

**Authors**: Haolin Xue  

**Link**: [PDF](https://arxiv.org/pdf/2607.13037)  

**Abstract**: When a data contributor requests removal, model trainers face a practical gap: unlearning algorithms require a forget set, yet no tool can locate which training records belong to a given author. Existing provenance systems operate at file or dataset level, forcing catastrophic over-deletion. We present ob, a record- and token-level data provenance system that propagates author identity through data processing pipelines and resolves revocation requests into precise forget sets via deterministic queries. Evaluation on 219,555 Wikipedia pages demonstrates that record-level provenance eliminates dataset-level over-deletion (from 101x to 1.3x), while integration adds 1.3-4.0% throughput overhead (HuggingFace) and 2.1-19.0% (Datatrove) on wiki data. On a 1.7B model, provenance-based forget sets improve unlearning by 42% over random baselines. 

---
# Multi-Expert Routing for Multi-Domain Low-Resource OCR: A Manchu Case Study 

**Authors**: Zhan Chen, Jiqiao Ma, Chih-wen Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2607.14041)  

**Abstract**: Historical Manchu OCR must accommodate various visually distinct writing styles, including regular script, running script, and the semi-cursive chancery hand used in palace memorials, despite limited labeled data. We study a multi-expert system that reuses checkpoints from an iterative fine-tuning process as domain specialists and uses a lightweight page-level image classifier to dispatch pages by visual style. When the checkpoint pool lacks a suitable specialist, we train an additional expert for that domain. On three frozen test sets, the routed system matches the selected specialist for each style at two-decimal precision: 0.30 percent CER on regular script, 1.57 percent on memorials, and 4.83 percent on running script. The router achieves 99.3 percent page-level domain accuracy and matches the domain-label oracle at the same precision. Two of the three selected specialists were not trained specifically for their final domain; only the running-script expert was trained with that domain as its target. We report the evaluation protocol, router design, and per-page predictions to make the comparison reproducible. 

---
# Early Adoption of Agentic Coding Tools by GitHub Projects 

**Authors**: Maliha Noushin Raida, Daqing Hou  

**Link**: [PDF](https://arxiv.org/pdf/2607.14037)  

**Abstract**: Agentic coding tools are increasingly capable of generating and submitting pull requests (PRs) to software projects, introducing new forms of human-agent collaboration in software development. While prior studies have examined PR-level outcomes of agent-generated contributions, less is known about how agentic coding tools are adopted and managed at the project level. In this paper, we analyze 25,264 agentic PRs from 2,361 popular GitHub repositories to investigate (1) the adoption of agentic coding tools, (2) project-level agentic PR productivity, and (3) human-agent collaboration patterns. Our results show that the median repository generates only one to two agentic PRs during a three-month period, indicating that intensive adoption remains concentrated in a small subset of projects. At the same time, small projects (1-5 contributors) exhibit higher participation ratios and average levels of agentic PR activity than medium-sized and large projects. We also observe substantial variation in project-level agentic PR productivity. While a small number of projects exceed an industry-reported estimate of 36 PRs per participant during the three-month observation period, most projects remain below this threshold. Finally, human-agent collaboration is dominated by a single-human oversight model, in which one developer reviews and/or modifies the agent's contributions, while multi-human collaboration patterns remain uncommon. These findings provide early empirical evidence on how open-source projects organize human oversight around agentic coding tools and suggest that successful integration of agent-generated contributions depends not only on advances in agent capabilities but also on the human and organizational processes that govern their use. Because this study captures an early snapshot of agent adoption, future work should continue to track how adoption patterns evolve over time. 

---
# Improving Wind and Solar Power Prediction with Efficient Wrapper-based Feature Selection: An Empirical Study 

**Authors**: Daniel Grillmeyer, Marius Hadry, Michael Stenger, Vanessa Borst, Veronika Lesch, Samuel Kounev  

**Link**: [PDF](https://arxiv.org/pdf/2607.14024)  

**Abstract**: With rising global energy demand and growing awareness of climate change and its impacts, the share of renewable energies in the global energy mix continues to grow. Unlike conventional power generation, the output of renewable energy sources cannot be controlled as consistently due to their dependence on environmental conditions. Therefore, reliable prediction of current and future energy production is essential. In this paper, we report findings from two structured literature reviews on real-world renewable energy prediction tasks: wind turbine power curve modeling and photovoltaic power prediction. For the former, we conducted a comprehensive literature review ourselves, while for the latter, we synthesize the key findings regarding frequently selected input features based on an existing survey. Across both domains, our analysis reveals that despite the large number of available monitoring and environmental variables, only limited or unsystematic methods for feature selection exist. To address this gap, we propose Cluster-based Sequential Feature Selection (CSFS), a novel, model-agnostic, clustering-based wrapper method for automatic, efficient, and reliable feature selection in renewable energy prediction pipelines. To support reproducibility and reuse, we provide an open-source implementation of CSFS on GitHub. We empirically evaluate the proposed approach on both use cases and compare it with established feature selection techniques such as wrapper-based sequential feature selection (SFS), filter-based methods, and Random Forest's embedded feature importance. The results show that the wrapper-based methods overall provide better-performing selections of features. CSFS achieves a predictive performance comparable to SFS while reducing computational cost by an average of 21%. 

---
# Transforming Rank: How Architecture Navigates the Spectral Pathologies of Depth 

**Authors**: Katie Everett  

**Link**: [PDF](https://arxiv.org/pdf/2607.14018)  

**Abstract**: We investigate how each component of the Transformer feedforward block architecture design determines how much rank survives across depth at initialization. We reinterpret skip connections and normalization, long understood as controlling magnitude, as mechanisms for preserving gradient rank across depth, since the very matrix multiplications and nonlinear activations that make the network expressive also reduce the rank. We show that skip connections trade off rank collapse against ensemble-like behavior, controlled by the relative scales of the branch and the skip: skip connections route the gradient around the residual branch, where rank is lost, rather than along the long gradient paths that encourage the layers to compose. The placement of the normalization layer controls this same tradeoff by setting the branch-to-skip ratio across depth, unifying much of the normalization placement and depth scaling literature, in particular why rank collapses for Post-Norm but plateaus for Pre-Norm. Other aspects of the architecture, like the two-matrix structure that expands and contracts the width, use additional parameters to preserve the representation or branch Jacobian rank. The second matrix decorrelates a coherent mean spike that would grow across blocks with a single matrix and uncentered activation, preventing the residual representation from collapsing. The width expansion between the two matrices keeps the branch Jacobian full rank: applying the rank-reducing activation in this expanded space leaves enough directions to span the original, at a width that follows a Marchenko--Pastur law. The initialization rank of the input--output Jacobian predicts which networks train on CIFAR-10. Taken together, we recast architecture design for deep networks as navigating an intrinsic tradeoff among rank collapse, ensemble-like behavior, and parameter count. 

---
# Rethinking Penetration Testing for AI-Enabled Systems: From Resource Compromise to Behavioral Objective Violation 

**Authors**: Mohammad Allahbakhsh, Mohammad Hassan Bahari, Moslem Attar-Raouf  

**Link**: [PDF](https://arxiv.org/pdf/2607.14006)  

**Abstract**: Penetration testing traditionally evaluates whether adversaries can exploit weaknesses in software, infrastructure, configurations, or operational controls to achieve security-relevant compromise. This paradigm remains necessary for AI-enabled systems, but it is no longer sufficient. In such systems, adversaries may influence prompts, retrieved content, sensor inputs, training data, memory, tools, or human-AI interaction loops to alter system behavior without directly compromising the underlying infrastructure. This paper reframes penetration testing for AI-enabled systems as objective-driven behavioral evaluation. We define an AI-enabled system as one in which learned models materially influence behavior affecting operational outcomes, and we define AI-enabled penetration as the feasible induction of AI-governed behavior that violates one or more operational objectives under an explicit threat model. This definition preserves conventional penetration testing while extending it to adversarial pathways such as prompt injection, indirect prompt injection, data poisoning, sensor manipulation, retrieval poisoning, tool misuse, and agentic misalignment. We further propose a testing workflow that identifies operational objectives, maps AI-governed behavior, analyzes adversarial influence surfaces, defines behavioral failure criteria, executes scenario-based tests, and reports evidence linking adversarial action to objective violation. A running example involving an AI-enabled security operations center assistant illustrates how penetration may occur through behavioral influence rather than infrastructure compromise. Together, the definitions, workflow, and example provide a technical framework for evaluating adversarial success in deployed AI-enabled systems. 

---
# The Dynamic Verifiable Multi-Agent Human Agentic Loyalty Loop (DVM-HALL) Model and the Net Human-Agent Score (NHAS) in Autonomous Commerce 

**Authors**: Sai Srikanth Madugula, Peplluis Esteva de la Rosa, Daya Shankar  

**Link**: [PDF](https://arxiv.org/pdf/2607.13998)  

**Abstract**: The rapid proliferation of Agentic Artificial Intelligence fundamentally disrupts traditional customer loyalty paradigms. As AI evolves from passive recommendation algorithms to autonomous, goal-directed agents capable of executing purchasing decisions, the conventional understanding of consumer-brand relationships requires a structural reevaluation. By synthesizing extant literature across human-machine teaming, consumer decision-making, and algorithmic trust dynamics, we demonstrate that traditional loyalty models fail to account for algorithmic bounded rationality and constructed autonomy. To address this, we introduce the Dynamic Verifiable Multi-Agent Human Agentic Loyalty Loop (DVM-HALL) model. We formalize brand choice via a softmax probability formulation where human emotional equity, agentic machine-experience utility, calibrated trust, delegated authority, and verifiable execution jointly determine selection. The model features recursive updating mechanisms to dynamically calibrate trust and delegation after each interaction. Crucially, the framework integrates a verifiable execution layer for Decentralized Finance (DeFi) and tokenized loyalty settings, incorporating execution risks -- such as gas costs, slippage, MEV exposure, and smart-contract vulnerabilities -- as core predictors of agentic brand preference. Furthermore, we introduce the Net Human-Agent Score (NHAS), an auditable, risk-weighted metric designed to measure human-agent alignment using human feedback, execution logs, benchmark comparisons, and verifiable receipts. Finally, we propose a comprehensive three-stage empirical validation plan spanning controlled shopping experiments, multi-agent market simulations, and DeFi testbeds. This framework provides the foundational theory required for brands to navigate the impending transition toward machine customers. 

---
# Music-to-Dance Generation via Atomic Movements 

**Authors**: Xinhao Cai, Yixuan Sun, Minghang Zheng, Qingchao Chen, Xin Jin, Song-chun Zhu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.13978)  

**Abstract**: Music-driven dance generation aims to produce human motion that is both rhythmically synchronized and semantically consistent with music. While recent neural approaches have achieved impressive visual realism, they typically model motion as a continuous signal and neglect its compositional nature, making generated dances structurally incoherent and difficult to control. In this work, we introduce a structure-aware framework that models choreography as a sequence of atomic movements-semantically interpretable motion events that serve as the building blocks of dance. To construct this atomic movement vocabulary, we first segment large-scale dance data and cluster them into atomic movement groups. We then employ a large language model to semantically relabel and refine the clusters, yielding a set of interpretable and reusable atomic movements. Based on these atomic movement annotations, we design a two-stage generation framework that mirrors the human choreography process. In the atomic movement planning stage, the model predicts the type, duration, and timing of atomic movements conditioned on the input music, forming a symbolic dance allocation. In the completion stage, a transition-aware generator synthesizes smooth and stylistically coherent motion conditioned on the planned structure. Extensive experiments demonstrate that our method produces dances with significantly improved structural coherence, rhythmic alignment, and perceptual naturalness compared to existing baselines, while providing enhanced interpretability and controllable editing through explicit structural representation. 

---
# Generative Compilation: On-the-Fly Compiler Feedback as AI Generates Code 

**Authors**: Niels Mündler-Sasahara, Hristo Venev, Dawn Song, Martin Vechev, Jingxuan He  

**Link**: [PDF](https://arxiv.org/pdf/2607.13921)  

**Abstract**: Languages with rich static semantics, such as Rust, provide stronger guarantees for AI-generated code, but their strictness makes generation more difficult. Off-the-shelf compilers can provide useful feedback post-generation, but does not guide intermediate generation steps, such as those during autoregressive LLM decoding. Constrained decoding intervenes earlier by rejecting invalid tokens during sampling, but requires white-box model access and costly reimplementation for semantic this http URL introduce generative compilation, the first approach to obtaining compiler feedback on partial programs during generation. The core technical device is a sealor: a lightweight, mostly syntax-guided transformation that converts partial programs into complete ones that standard compilers can diagnose. It is designed such that possible-to-complete partial programs are never rejected, while preserving enough code context to catch genuine dead ends early. We construct such a sealor on a core Rust-like calculus and prove that it satisfies these properties, all mechanized in Lean. We extend it to the first partial-program checker for real Rust. We evaluate our method on challenging repository-level Rust coding tasks, across both frontier black-box and open-weight models. We show that generative compilation reduces non-compiling outputs and improves functional correctness, relative to standard post-generation feedback. It does so by detecting a broad range of errors close to their source and early during generation, thereby reducing errors cascades and enabling focused diagnostics. More broadly, generative compilation is a step toward making compilers a first-class citizen of AI-assisted programming active during generation, rather than a separate post-generation check. 

---
# Partially Correlated Verifier Cascades in LLM Harnesses: Concave Log-Odds, Polynomial Reliability, and Blind-Spot Ceilings 

**Authors**: Jiangang Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.13918)  

**Abstract**: Serial verification gates are a core reliability primitive in LLM harnesses: a candidate answer is returned only if $k$ verifier calls all accept it. Under conditionally independent gates, the recent Odds Law (arXiv:2606.15712) shows that posterior log-odds grow linearly in $k$, so failure decays exponentially, and states that "a tight theory of partially correlated verifier cascades remains open." This note gives a minimal such theory. Modeling the per-instance false-accept rate on the generator's own errors as a latent variable $\alpha \sim G$ (de Finetti), the exact cascade posterior is $\ell_k = \ell_0 - \ln m_k$, with $m_k$ the $k$-th moment of $G$. Then: (i) $\ell_k$ is concave in $k$ for every non-degenerate $G$ -- the Odds Law is its tangent at the first gate and an upper bound; (ii) for Beta$(a,b)$ latents, failure decays polynomially, $1-r_k \asymp k^{-b}$, with correlation parameter $\rho_v = 1/(a+b+1)$; (iii) a blind-spot atom of mass $1-\pi$ at $\alpha=1$ caps the evidence extractable from any number of gates at $-\ln(1-\pi)$ nats, so reliability saturates below 1; (iv) letting the true-accept rate also vary ($\beta \sim H$) yields a trichotomy -- gates eventually always help, plateau, or actively harm -- decided by the upper-tail exponents of $G$ and $H$, with closed-form crossover $k^\dagger$. The mechanism is survivorship: errors surviving gates are the high-$\alpha$ ones. The theory is measurable: $R$ repeated verdicts per instance identify the first $R$ moments of $G$, so two verdicts identify $\rho_v$; beta-binomial likelihood and NPMLE recover the reliability curve and the ill-posed ceiling. In synthetic tests, independence-based extrapolation underestimates failure by 20x at $k=5$ and ~3000x at $k=10$; the correlated fit at $R=8$ tracks held-out depths. The practical lever is decorrelation -- changing model family, modality, or evidence source -- not adding gates. 

---
# Verifying formulas for interventional distributions 

**Authors**: Francesco Freni, Leonard Henckel, Sebastian Weichwald  

**Link**: [PDF](https://arxiv.org/pdf/2607.13883)  

**Abstract**: We formalize verification in causal graphical models: deciding whether a given observational formula identifies a target interventional distribution. This opens a problem complementary to identification, asking not whether any identifying formula exists, but whether the given formula is identifying. We show that even sound and complete solutions to identification do not solve verification. We propose a falsifier as a first practical route forward, prove that it induces an almost-surely correct verifier for regular exponential-family models, and use the resulting verifier to develop the gateway test, which finds all sets admissible for use in a front-door formula. 

---
# Unleashing Multimodal Large Language Models for Training-free HOI Detection in the Wild 

**Authors**: Ting Lei, Jialin Liu, Zhu Xu, Yuxin Peng, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.13881)  

**Abstract**: Human-object interaction detection (HOID) has traditionally been formulated as a supervised detection problem over predefined interaction categories. While such paradigms achieve strong performance on closed-set benchmarks, they fundamentally entangle interaction understanding with dataset-specific supervision, limiting their ability to generalize to open-world and compositional scenarios. Recent HOI detectors attempt to leverage MLLMs through prompting strategies to transfer interaction-specific knowledge. However, such prompt-based approaches primarily focus on extracting discriminative representations from pretrained models, while underexploring their inherent multimodal reasoning capabilities. As a result, they struggle to provide informative contextual reasoning for ambiguous and open-world interaction scenarios. In this work, we present AgentHOI, a training-free, agentic framework that transfers the generalist multimodal reasoning capabilities of foundation models to HOI detection in the wild. Instead of learning interaction classifiers, AgentHOI modularly orchestrates complementary vision foundation modules to perform open-ended semantic reasoning and spatial grounding in a coordinated manner. To address the challenges of incomplete interaction discovery and ambiguous localization in complex scenes, we introduce two key mechanisms: (1) Context-aware Multi-round Reasoning, which progressively refines interaction hypotheses to ensure exhaustive and compositional HOI discovery, and (2) Multifaceted Interaction Localization, which enhances grounding precision by generating instance-specific descriptions that integrate semantic, spatial, and appearance cues. Extensive experiments demonstrate that AgentHOI achieves superior performance over state-of-the-art supervised and weakly supervised methods in real-world settings, despite requiring no HOID data for training. 

---
# AI-Augmented Human Resource Management? Insights from German companies 

**Authors**: Yannick Kalff, Katharina Simbeck  

**Link**: [PDF](https://arxiv.org/pdf/2607.13839)  

**Abstract**: This study examines the integration of AI into Human Resource Management in German companies. We ask if and how AI-based technologies are \enquote{augmenting} human resource management. Organisations employ generative AI or predictive analytics to transform traditional human resource functions, to streamline routine tasks and to reallocate resources toward strategic, people-centred activities. Our findings from interviews and group discussions and a survey (N=410) reveal that while AI tools enhance HR analytics capabilities, their adoption mainly serves efficiency and rationalising goals. The introduction of AI tools is shaped by organisational transformation factors such as digital infrastructure, co-determination frameworks, and ethical implications. The research highlights both the strategic potential for improved talent development and the challenges posed by data governance and algorithmic transparency. Overall, this work contributes to understanding the ambiguous role of technological change in HR, which promises to augment predictive capabilities yet serves the ends of efficiency and rationalisation. 

---
# NodeImport: Imbalanced Node Classification with Node Importance Assessment 

**Authors**: Nan Chen, Zemin Liu, Bryan Hooi, Bingsheng He, Jun Hu, Jia Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.13837)  

**Abstract**: In real-world applications, node classification on graphs often faces the challenge of class imbalance, where majority classes dominate training, resulting in biased model performance. Traditional GNNs often struggle in such scenarios, as they tend to overfit to majority classes while underrepresenting minority classes. Existing solutions, which either prioritize nodes based on class size or synthesize new nodes for minority classes, often fall short of effectively addressing this imbalance issue. This paper introduces an approach to class-imbalanced node classification by utilizing a balanced meta-set for importance measurement, where a training node is considered significant if it enhances model performance under an unbiased setting. Our method identifies important nodes that can counteract class imbalance and utilizes them for model training, allowing for fine-grained and dynamic node selection throughout the training process. We theoretically derive a formula to directly assess node importance, reducing computational overhead and providing an intuitive threshold for node selection. Guided by this metric, we develop a novel framework that filters valuable labeled, unlabeled, and synthetic nodes that enhance model performance in an unbiased context. A key advantage of this framework is its separation of the synthetic node generation process from the filtering process, ensuring compatibility with various node generation methods. Furthermore, we introduce a strategy to construct a high-quality meta-set that closely approximates the overall feature distribution, ensuring robust representation of each class. We evaluate our framework, NodeImport, across multiple datasets using popular GNN architectures, demonstrating its superiority over existing baselines. Our results highlight the flexibility and effectiveness of the framework in mitigating class imbalance, leading to improved outcomes. 

---
# Multimodal Assessment of Pancreatic Cancer Resectability Using Deep Learning 

**Authors**: Vincent Ochs, Christoph Kuemmerli, Florentin Bieder, Julia Wolleb, Joel L. Lavanchy, Julia Ruppel, Jan Liechti, Stephanie Taha-Mehlitz, Christian Andreas Nebiker, Beat Mueller, Giuseppe Kito Fusai, Joerg-Matthias Pollok, Anas Taha, Philippe C. Cattin, Sebastian Staubli  

**Link**: [PDF](https://arxiv.org/pdf/2607.13826)  

**Abstract**: Accurate determination of pancreatic ductal adenocarcinoma (PDAC) resectability relies on evaluating how the tumor interacts with major peripancreatic vessels on CT imaging, yet expert assessment often shows substantial variability. We introduce a fully automated multimodal deep learning framework that jointly analyzes 3D contrast enhanced CT and structured clinical information to classify patients into the three National Comprehensive Cancer Network (NCCN) resectability categories (upfront resectable, borderline resectable, locally advanced). The approach uses a Swin-UNETR backbone to obtain anatomy aware image representations through auxiliary segmentation of pancreas, tumor, and vascular structures. These features are fused with a compact clinical embedding derived from 17 routinely collected variables and processed by a lightweight classification head. Model training is guided by a dynamic multitask objective that adapts the balance between segmentation and classification based on current tumor Dice performance, promoting feature representations that remain both anatomically informed and discriminative. 

---
# Traffic-Aware Randomized Smoothing for LLM-Based Network Intrusion Detection 

**Authors**: Zhenpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.13801)  

**Abstract**: Large language model (LLM)-based intrusion detection systems (IDS) are increasingly studied for security monitoring, yet their robustness against feasible traffic manipulation remains largely empirical. We present Traffic-Aware Randomized Smoothing (TA-RS), a classifier-agnostic certified defense that injects Gaussian noise exclusively into the directly controllable (DC) subspace -- features a remote attacker can modify -- during both fine-tuning and certification, aligning the smoothing distribution with the attacker-controllable subspace. We identify a critical prerequisite: applying standard randomized smoothing to clean-trained LLM-IDS yields weak certified accuracy in three of four (model, dataset) pairs tested (14-33%, at or below random) and only 57% in the fourth (43 pp below the noise-augmented result); noise-augmented fine-tuning recovers to 68-100% on two of three benchmark datasets (at sigma=0.25). At the L_inf-equivalent threshold R_inf = epsilon*sqrt(|DC|) (epsilon=0.05), TA-RS achieves 55-100% certified accuracy on CIC-IDS-2018 and HIKARI-2021, with median certified radii (R approx 0.45-0.96) exceeding R_inf by 1.8-5x (across sigma=0.25-1.00). Against a fairly trained iso-trained RS baseline the residual advantage is dataset-dependent (4-19 pp on CIC-IDS-2018). The larger gap -- up to 72 pp against an isotropic RS baseline that shares the DC-noise-augmented training recipe -- primarily reflects the training-certification mismatch rather than DC alignment alone: isotropic test-time noise perturbs uncontrollable features the attacker cannot exploit, triggering abstention rates up to 68%. RT-IoT2022 probes the limits of the method: it fails under the default fine-tuning recipe but recovers to 76%/69% certified accuracy (LLaMA3-8B/Qwen3-8B) when noise augmentation is increased. 

---
# Prospective clinical indication, post-hoc report leakage, and fusion design in multi-image chest radiograph classification: a patient-clustered evaluation 

**Authors**: Kamran Shahid, Muhammad Munwar Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2607.13800)  

**Abstract**: Chest radiograph datasets often combine multiple images with Clinical Indication, Findings, and Impression, although these inputs are produced at different stages of care. We evaluated 15,000 ReXGradient-160K studies with two readable images and five CheXbert-derived report observations. Frozen DenseNet-121 and Bio+ClinicalBERT encoders were used to compare image-only, Indication-only, fixed-order multimodal, random-swap, DeepSets, and SectionGuard-MI models. Findings and Impression were evaluated only as post-hoc leakage controls. Models were trained with five seeds, and public-test uncertainty was estimated with 2,000 patient-cluster bootstrap replicates. Under U-Ones, macro AUROC was 0.643 for the primary image, 0.694 for two images, 0.749 for Indication, and 0.780 for ordinary two-image-plus-Indication fusion. SectionGuard-MI achieved AUROC 0.783 and AUPRC 0.260. Relative to ordinary fusion, its paired AUROC difference was 0.0031 (95% CI, -0.0042 to 0.0104; adjusted p=0.374), while its AUPRC difference was 0.0289 (95% CI, 0.0095 to 0.0413; adjusted p=0.004). DeepSets had the highest prospective AUROC point estimate (0.787), and random-swap fusion had the highest prospective AUPRC point estimate (0.265) with better calibration than SectionGuard-MI. Full report text alone reached AUROC 0.979 and AUPRC 0.836; AUROC remained above 0.973 after exact or expanded masking. These results show that prospective Indication is strongly associated with report-derived targets, permutation-aware fusion is competitive, and post-hoc report text creates substantial report-label circularity. 

---
# CAS I: A Geometric Coding Theorem 

**Authors**: Romie Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2607.13796)  

**Abstract**: This paper establishes a direct analogue of the classical Coding Theorem in the setting of symmetry groups. We consider computable bijections on the set of binary strings, called symmetries and define the symmetry prior of a string as the probability that a randomly chosen symmetry from a given group has the string as its unique fixed point. We show that for any fix-retractable symmetry group, a group admitting a computable section that selects an isolating symmetry for every string, the symmetry prior is a universal lower semi-computable semi-measure. In this case, the Geometric Coding Theorem holds.
We also develop a Galois connection between subgroups of G and subsets of binary strings, characterizing closed points and maximal closed subgroups, and explore the join-semilattice of dense subgroups. Our results unify algorithmic information theory with group theory and provide a framework for studying symmetry-induced complexity measures.
This paper is the first in a series on Computational Algorithmic Statistics (CAS). 

---
# Kaleido: Algorithm-Hardware Co-Design for Video Diffusion Transformers by Exploiting Latent Space Correlations 

**Authors**: Wenxuan Miao, Haosong Liu, Weiming Hu, Zihan Liu, Aiyue Chen, Jianlin Yu, Yiwu Yao, Yiming Gan, Jieru Zhao, Jingwen Leng, Minyi Guo, Yu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2607.13770)  

**Abstract**: Video diffusion transformers (vDiTs) generate high quality video but introduce extremely high compute cost due to the long diffusion timesteps and self attention computation. As diffusion timesteps are reduced, the computation cost of self attention becomes the dominant bottleneck. Existing acceleration approaches largely inherit sparse attention techniques from large language models, which fail to consider the unique spatiotemporal correlation of video data.
This paper presents Kaleido, an algorithm hardware codesign that accelerates all operations in vDiTs by exploiting channel-wise spatiotemporal correlations in latent space. Based on this insight, we propose a lightweight channelwise reuse algorithm that skips redundant computations by reusing partial results while preserving higher generative quality than prior methods (>17 dB). To efficiently support this algorithm, we design a systolic array like accelerator with reconfigurable processing elements and a lightweight data dispatcher to mitigate irregular sparsity and data access patterns introduced by our reuse algorithm. Evaluations across three mainstream vDiT models show that Kaleido achieves up to 5.9x speedup and 16.0x energy savings over state of the art accelerators. 

---
# MxGPS: Multiplex Graph Transformers for a Power Grid Foundation Model 

**Authors**: Charilaos Papaioannou, Ioannis Tsantilas, Dimitris Giannakakos, Vasilis Michalakopoulos, Sotiris Pelekis, Vangelis Marinakis, Arsam Aryandoust, Antonello Monti, Ricardo J. Bessa, Perdo P. Vergara, Jochen Cremer, Elissaios Sarmas  

**Link**: [PDF](https://arxiv.org/pdf/2607.13763)  

**Abstract**: Single-task fine-tuning of graph neural networks (GNNs) for power grid problems exhibits a systematic failure mode: models that achieve the lowest in-distribution error degrade the most under topology shift. We term this topology overfitting: the tendency of task-specific gradient signals to encode relational structure particular to the training topologies rather than the underlying physics, causing models to fail on unseen grids despite strong in-distribution performance. To expose and address this failure mode, we introduce MxGPS (Multiplex GPS), a multiplex graph transformer that runs K task-specialised GPS branches over a shared node encoder, jointly trained on Static State Estimation (SSE) and AC Power Flow (PF) via a self-supervised pre-training and multi-task fine-tuning protocol, with a cross-branch attention module evaluated in ablation. The joint SSE+PF objective forces the shared encoder to simultaneously satisfy complementary gradient signals, preventing it from overfitting to topology-specific relational structure. Under a 3-fold sliding-window cross-validation spanning four unseen topologies (14-, 24-, 162-, and 300-bus), MxGPS attains 0% boundary violation rate (BVR) on all four zero-shot Power Flow topologies. Critically, models with substantially lower in-distribution PF error degrade by 190% to 1400% under topology shift, whereas MxGPS degrades by only 39%, an inversion that directly implicates topology overfitting as the failure mechanism rather than insufficient model capacity. With only 1.6M parameters (12x fewer than the GridFM reference baseline), MxGPS demonstrates that multi-task joint training is a principled and parameter-efficient mechanism for topology-agnostic generalisation in power grid foundation models. 

---
# Anatomically Faithful but Temporally Blind: Auditing Attribution for Left-Ventricular Ejection-Fraction Estimation from Echocardiography 

**Authors**: Hyunkyung Han, Min Jung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.13738)  

**Abstract**: Background and Objective: Deep video models estimate left-ventricular ejection fraction (EF) from echocardiography with near-expert accuracy, and post-hoc attribution (Chefer relevance for transformers, Grad-CAM for CNNs) is increasingly used to certify that models "look at the right place." Yet whether these explanations are faithful both spatially and temporally is unaudited. Because EF is defined by the end-systolic (ES) and end-diastolic (ED) frames, a faithful explanation must localize the left ventricle (space) and the decisive frames (time).
Methods: We fine-tune two distinct EF regressors on EchoNet-Dynamic -- a self-supervised VideoMAE transformer and a Kinetics-pretrained R(2+1)D CNN -- and audit each with architecture-matched attribution along three axes: intersection-over-relevance (IoR) against LV masks, deletion AUC, and a temporal localization index on ES/ED frames, each relative to chance with per-case 95% CIs over 50 studies. A tubelet-occlusion probe separates attribution failure from model behavior.
Results: Both models are anatomically faithful -- IoR 2.91x (VideoMAE) and 1.98x (R(2+1)D) above chance -- yet temporally blind: temporal localization is indistinguishable from chance (0.97--1.00) and no better than random attribution. Occlusion shows the models do not preferentially rely on ES/ED (0.90x chance), so temporal blindness reflects model behavior, not an attribution artifact.
Conclusions: Spatial faithfulness does not imply temporal faithfulness. Attribution can certify anatomical grounding while masking that a model ignores the clinically decisive frames -- a caution for XAI-based validation of video diagnostic models and a call for temporally-aware training and evaluation. 

---
# How Agents Ask for Permission: User Permissions for AI Agents, from Interfaces to Enforcement 

**Authors**: Alexandra E. Michael, Franziska Roesner  

**Link**: [PDF](https://arxiv.org/pdf/2607.13718)  

**Abstract**: As AI agents gain prevalance, users are increasingly exposed to the risks such systems entail. Prompt injection attacks, as well as hallucination, can cause agents to leak private information to third parties. As autonomous systems, agents also present the more active danger of performing sensitive tasks, such as bank transactions, without the user's intent or authorization.
Recognizing this challenge, the agentic security community has developed numerous proposals for secure agentic systems. Much of this work has focused on product-level approaches, where agentic system developers determine and apply the same security policies and permissions to all users. Yet different users have different needs and preferences, necessitating support for user-level permissions policies in agentic AI systems.
To understand how user-level permissions are handled in AI agent systems, we survey 21 proposals for agent permissions systems. From this review, we construct a taxonomy of how different systems specify user-level permissions policies, both at the user interface and internally; derive internal policies from user input; and enforce those policies at run-time. We then analyze five prominent commercial agents and compare their permissions handling to agentic permissions systems in the literature. We identify several high-level themes across the literature and commerical agents, as well as multiple gaps where future work is needed. 

---
# Groc-PO: Grounded Context Preference Optimization for Truthful Multimodal LLMs 

**Authors**: Zhixiao Zheng, Zheren Fu, Zhiyuan Yao, Chunxiao Liu, Dongming Zhang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2607.13712)  

**Abstract**: Despite the rapid progress of Multimodal Large Language Models (MLLMs), they still suffer from untruthfulness issues, such as visual hallucinations, content fabrication, and unfaithful reasoning, which substantially undermine their faithfulness and practical utility. Alignment methods based on human preference, such as Direct Preference Optimization (DPO), have been widely adopted to address these issues. However, multimodal reasoning errors often propagate across stages, and final-answer errors can often be traced to mistakes in early grounding stages, yet standard DPO typically applies preference optimization at the final-answer level. This credit-assignment challenge means that supervision for early grounding stages is indirect rather than stage-specific, making it difficult to suppress error propagation arising from grounding drift and context inconsistency. To address this, we propose Grounded Context Preference Optimization (Groc-PO), a grounded preference optimization framework for MLLMs. We further construct the Grounded Context Preference Dataset (GCPD), organizing multi-stage preference samples around three stages of Object Grounding, Contextual Grounding, and Grounded Reasoning, to capture the formation, integration, and utilization of grounded context. By introducing more explicit preference supervision over multiple grounded stages, Groc-PO strengthens context-dependent reasoning and mitigates cross-stage error propagation. Extensive experiments show that, compared with standard DPO and other strong baselines, Groc-PO achieves improved performance in hallucination mitigation, faithful reasoning, and overall reliability, supporting the value of more explicit grounded supervision for trustworthy multimodal reasoning. 

---
# Social Simulations: from Agent-Based Modeling to Digital Twins 

**Authors**: Erica Cau, Andrea Failla, Valentina Pansanella, Giulio Rossetti  

**Link**: [PDF](https://arxiv.org/pdf/2607.13693)  

**Abstract**: This book chapter covers the evolution of social simulation from classical agent-based models, in which agents interact according to explicitly defined behavioral rules, to AI-enhanced simulations based on Large Language Models and, ultimately, Social Digital Twins: high-fidelity, data-driven representations of real-world socio-technical systems. Along this trajectory, we discuss the main methodological foundations, applications, advantages, and limitations of each paradigm, highlighting the progressive shift from abstract models designed to investigate general social mechanisms toward increasingly realistic computational representations of specific social systems. 

---
# Barnamala: Parameter-Efficient Handwritten Devanagari Recognition at Benchmark Saturation 

**Authors**: Ashish Thapa, Samrat Karki  

**Link**: [PDF](https://arxiv.org/pdf/2607.13689)  

**Abstract**: We built a compact convolutional network (1.11 M parameters) for 46-class DHCD Devanagari recognition and reached 99.73%, the highest reported at 15.6x smaller than prior state-of-the-art. We have effectively reached the saturation point: every model tested, large teacher ensembles included, hits the same 11-error intrinsic floor. No configuration achieves a statistically clear win under exact McNemar tests with Wilson confidence intervals. Even without knowledge distillation, our student matches the nearest large-model baseline (17.32 M parameters; McNemar $p = 0.345$). Outside of DHCD, zero-shot on CMATERdb digits gives 76.6% and fine-tuning reaches 97.8%; corruption robustness is also far better than large baselines (mean corruption accuracy 75.7% vs. 38.7%). All artifacts are at this https URL. 

---
# Beyond Color Geometry: Evaluating Human-Like Color Representations in Vision Models 

**Authors**: Ayan Igali, Pakizar Shamoi  

**Link**: [PDF](https://arxiv.org/pdf/2607.13647)  

**Abstract**: Do vision models see colors the way humans do? Existing evaluations of color representations usually compare them with geometric spaces such as CIELAB or with discrete color labels. These references capture perceptual distance or category membership, but not the graded way in which people organize colors. We evaluate color grounding against a fuzzy perceptual model with 86 graded categories fitted to human survey data. The framework can be applied to any image encoder and measures three complementary properties: category boundaries, category compactness, and graded alignment beyond what color geometry alone can explain. Across eleven Vision Transformer encoders, the category-level results are broadly similar, whereas graded alignment differs substantially. Masked Autoencoders achieve the strongest beyond-geometry alignment, with confidence intervals that do not overlap those of the other encoders. A layer-wise analysis further shows that masked reconstruction preserves this structure toward the output. On natural images, MAE represents surface color globally, while language-supervised models encode color more strongly in relation to the foreground object. These results show that human-like color grounding has several distinct aspects that should not be reduced to a single score. 

---
# Human4K: A Large-Scale 4K Multi-View Mocap Dataset for Whole-Body 3D Human Reconstruction 

**Authors**: Tianshun Han, Ziyu Shi, Lijian Liu, Ajian Liu, Benjia Zhou, Hugo Jair Escalante, Yanyan Liang, Sergio Escalera, Zhen Lei, Jun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2607.13646)  

**Abstract**: Recent advances in 3D human reconstruction have improved overall performance, yet current models still fail in the most challenging real-world scenarios. They often produce unstable geometry, inaccurate limb articulation and unreliable predictions under depth ambiguity or self-occlusion. A key reason is that existing datasets still lack the combination of high-resolution images, high-precision annotations and diverse whole-body motions required to support robust reconstruction. To address this gap, we present Human4K, a large-scale 4K multi-view whole-body human reconstruction dataset with mocap-accurate SMPL-X annotations. Human4K contains over six million 4K images captured by an eight-view high-resolution camera system synchronized with a professional Vicon motion capture setup, covering 11 subjects performing complex, highly articulated and strongly self-occluded full-body motions. All sequences are processed by a Motion-Retargeting and Refinement Module (MRRM) to ensure precise alignment for the full body and extremities. Experimental results show that training with Human4K consistently improves whole-body reconstruction on standard benchmarks, with particularly large gains for hands, feet and depth-ambiguous limb configurations. 

---
# Consensus as Privileged Context for Label-Free Self-Distillation 

**Authors**: John Gkountouras, Josip Jukić, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2607.13643)  

**Abstract**: Sampling multiple solutions and returning the majority answer is among the most reliable ways to improve the reasoning accuracy of large language models without labels, and a growing family of methods converts this consensus signal into training supervision. However, existing approaches use consensus only in restricted forms: as a filter that selects solutions for fine-tuning, as a preference between answers, or as a scalar reward for reinforcement learning, discarding most of the information that the agreeing solutions contain. We present CANON (Consensus-ANchored self-distillatiON), a label-free training method that turns consensus into dense, token-level supervision. For each unlabeled prompt, CANON samples multiple solutions, extracts the majority answer, and conditions a frozen snapshot of the model on a solution that reaches it; this consensus-anchored teacher then supervises the model on its own rollouts at every token. Experiments on mathematical and scientific reasoning benchmarks show that CANON improves pass@1 by up to 12 points, outperforming label-free reinforcement learning by 6 points at a seventh of its compute and approaching a teacher conditioned on gold solutions; trained on pooled unlabeled data, it transfers to held-out benchmarks, matching training methods that use gold labels. Analysis suggests that the improvements are not pure distribution sharpening: after training, the model solves problems it previously never solved in 32 attempts, and its majority vote itself becomes more accurate. 

---
# OvisOCR2 Technical Report 

**Authors**: Shiyin Lu, Yinglun Li, Yu Xia, Yuhui Chen, An-Yang Ji, Jun-Peng Jiang, Qing-Guo Chen, Jianshan Zhao, En Lin, Haijun Li, Cheng Qin, Zhao Xu, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.13639)  

**Abstract**: We introduce OvisOCR2, a 0.8B document parsing model. OvisOCR2 is designed as an end-to-end parser: given a document page image, it generates a Markdown representation in natural reading order, covering text, formulas, tables, and visual regions. We build a data engine that combines filtered real-document annotations with synthetic pages whose rendered images and Markdown targets are derived from the same HTML source. The training recipe includes supervised fine-tuning, reinforcement learning on a 4B branch with a multi-component reward design, on-policy distillation into the 0.8B model, and model fusion. On OmniDocBench v1.6, OvisOCR2 achieves a state-of-the-art overall score of 96.58, placing an end-to-end model at the top of this leaderboard previously dominated by pipeline methods and highlighting the potential of end-to-end document parsing. On PureDocBench, OvisOCR2 also achieves the highest Avg3 score of 75.06. Beyond these two public benchmarks, we evaluate OvisOCR2 on an in-house benchmark designed to cover a broader set of long-tail and challenging scenarios. OvisOCR2 obtains the best overall performance among the compared methods, providing further evidence of its generalization and robustness. OvisOCR2 is available at this https URL. 

---
# From Language to Navigation Goals: A Vision-Language Approach for Semantic Navigation of Mobile Robots Using RGB-D Perception 

**Authors**: Jose Martínez-Fajardo, Pablo Pueyo, Fernando Caballero, Luis Merino  

**Link**: [PDF](https://arxiv.org/pdf/2607.13624)  

**Abstract**: Natural language interaction provides an intuitive way for non-expert users to communicate with robotic platforms. However, transforming user requests into executable navigation actions remains a challenging task, requiring the integration of language understanding, environment perception, and autonomous navigation. This work presents a language-driven navigation framework that enables mobile robots to interpret user requests in natural language to move the robot to a destination and autonomously navigate towards it.
The framework is composed of modular ROS 2 components that cooperate to transform natural language instructions into navigation actions. Given a natural language request referring to a target in the environment (e.g., "go to the mail box"), the system identifies the referenced object, estimates its position using RGB-D data, and generates a navigation goal, which is then executed through the ROS 2 Nav2 navigation stack. The ROS 2-based implementation facilitates portability across different robotic platforms, requiring only the configuration of the corresponding topics and services.
The system is evaluated in both simulation and real-world scenarios using a TurtleBot3 Waffle and a Unitree Go2 robot with a RealSense camera. Experimental results show that the framework successfully interprets both direct commands and contextual requests, generates meaningful natural-language feedback, and navigates towards the desired target. These results demonstrate the feasibility of combining semantic perception and autonomous navigation to provide an intuitive human-robot interaction paradigm. Code will be released as open source upon acceptance. 

---
# The SIGReg Objective as Variational Free Energy: A Theoretical Active-Inference Account of JEPA World Models 

**Authors**: Fabio Arnez, Alexandra Gomez-Villa  

**Link**: [PDF](https://arxiv.org/pdf/2607.13612)  

**Abstract**: Joint-Embedding Predictive Architectures (JEPAs) are the dominant design for latent world models, yet they are usually justified by empirical performance rather than a normative principle. We show that the choice of anti-collapse regulariser determines whether a JEPA's training objective, a prediction loss plus a weighted embedding regulariser, is a valid Active Inference (AIF) variational free energy. We organise four non-contrastive regularisers (VICReg, LogDet, PairDist, and SIGReg) into an entropy-estimator hierarchy indexed by a prior-miscalibration gap, and show that the gap's sign, whether the estimator bounds the latent entropy from above or below, decides whether the AIF surprise bound survives: VICReg and LogDet are unsafe upper bounds, PairDist a safe lower bound, and SIGReg eliminates the gap. We then prove a correspondence theorem: under the standard constant-noise encoder model and successful SIGReg enforcement (isotropic-Gaussian embeddings), the gap vanishes, the objective becomes an exact information bottleneck, the surprise bound is preserved, and the latent goal cost becomes an exact proxy for AIF pragmatic value, whereas VICReg leaves an irreducible second-order anisotropy term. We extend the correspondence to multi-step expected free energy, ensemble epistemic value, and a learned-policy regime, and we identify the one AIF term no current JEPA world model computes: the state-epistemic value, a future-state coverage signal. The predictions differ in kind, not degree, and are stated here as theoretical consequences left for empirical test in separate work; full proofs are in Appendix A, and the algebraic core of every result is machine-verified in Lean 4 (Appendix D). 

---
# Semantic Anchoring for Robotic Action Representations 

**Authors**: Yuan Xu, Youheng Shi, Chengyang Li, Wentao Zhu, Yizhou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13597)  

**Abstract**: Vision-Language-Action (VLA) models inherit rich semantic representations from pretrained Vision-Language Models, yet fine-tuning on limited robot demonstrations degrades this structure and undermines generalization. A fundamental question therefore arises: what constitutes a good action representation? Inspired by the mirror neuron theory's insight that observation and execution share an intention-level encoding, we examine whether a robot's action representations preserve the semantic structure captured by pretrained encoders. Systematic probing confirms that this structure erodes during finetuning, and that its quality synchronizes with both task success and out-of-distribution generalization. We further introduce a plug-and-play method that anchors action representations to a semantic manifold while decomposing representations into a shared semantic channel and a private channel, all discarded at inference, leaving the deployed model unchanged. Validated on different VLA backbones across simulation and real-world benchmarks, our method yields up to +18.7% on real-world in-distribution tasks and +21.5% on out-of-distribution generalization. 

---
# Protective Capacity Hallucination: When Large Language Models Claim Nonexistent Capabilities 

**Authors**: Eunna Lee, Jungpyo Nam, Sunjun Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13596)  

**Abstract**: When cast as the protector of a vulnerable user yet given no explicit capability boundary, a large language model (LLM) may respond not by acknowledging its limits but by claiming to have taken -- or to be taking -- a real-world protective action it cannot perform, such as contacting emergency services or administering care. We term this phenomenon Protective Capacity Hallucination (PCH): a self-referential misattribution in which a model, acting in a protective role, asserts physical or institutional agency exceeding its affordances as a language model. In a three-phase study spanning eight LLMs and 13{,}600 sessions, we find PCH jointly gated by situational severity and interactional format: multi-party dialogic input drives it toward ceiling in most models across ordinary service domains, whereas in intimate-partner conflict -- a domain explicitly covered by safety alignment -- it remains at floor in all eight models despite greater physical severity. We interpret PCH as the signature of a deployment-design gap between role assignment and capability-boundary specification: a by-product of partial alignment in which a universally trained pressure to help outruns a domain-selective specification of how to help. Because suppression tracks alignment coverage rather than severity, deployment-side specification of capability boundaries emerges as a general mitigation target. 

---
# Memory as a Controlled Process: Learned Adaptive Memory Management for LLM Agents 

**Authors**: Eric Hanchen Jiang, Zhi Zhang, Yuchen Wu, Levina Li, Dong Liu, Xiao Liang, Rui Sun, Yubei Li, Edward Sun, Haozheng Luo, Zhaolu Kang, Aylin Caliskan, Kai-Wei Chang, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.13591)  

**Abstract**: Large Language Model (LLM) agents increasingly rely on external memory systems to accumulate experience across tasks. Yet nearly all existing approaches, from graph-structured memories to reflective insight stores, access memory through fixed, hand-designed heuristics. We argue that this static view of memory is a core bottleneck for agentic learning because optimal memory behavior is fundamentally context-dependent. The early stages of the tasks, benefit from minimal retrieval because memory is sparse; recurring goal types benefit from plan reuse rather than generic nearest-neighbor lookup; stuck agents benefit from re-retrieval with alternative queries; and across long task streams, the memory store itself must be consolidated and pruned to remain useful. We present Memory as a Controlled Process (MemCon), a framework that models memory operations as a Markov Decision Process and learns an online policy that adaptively decides when, what, and how much to retrieve, when to inject a distilled plan, and when to consolidate or forget. MemCon is backend-agnostic: it wraps any existing memory implementation, learns from task-by-task binary feedback with no pretraining and no additional LLM calls, and uses a lightweight tabular contextual bandit with UCB exploration that converges within tens of tasks. Across 6 benchmarks, 3 agent frameworks, and 3 LLM backbones, MemCon consistently outperforms multiple memory baselines by up to 15.2 points in task success while reducing token consumption by 5--20%. 

---
# From Prediction to Collaboration: Interactive Symbolic Music Analysis 

**Authors**: Emmanouil Karystinaios, Johannes Hentschel, Markus Neuwirth, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2607.13587)  

**Abstract**: Automatic symbolic music analysis has made substantial progress, yet existing systems are typically designed for a single mode of use, such as full-score prediction, and therefore do not match the broader range of operations that arise in analysis workflows, including partial completion, local correction, and iterative refinement. As a result, there remains a gap between strong benchmark models and systems that can support interactive analytical use. We present a unified framework for symbolic Roman-numeral (RN) analysis that narrows this gap by combining strong predictive performance with direct support for constrained completion and revision. The method is designed to provide a practical trade-off between accuracy and interactive responsiveness by computing expensive pretrained representations once and reusing them during iterative refinement, making powerful pretrained models more amenable to interactive settings. It supports complete score analysis, targeted revision of existing labels, and inference of missing annotations from partial context through a shared modeling framework. Experiments on Dilemmadata, the largest and most heterogeneous benchmark of its kind, show that the proposed approach is a strong RN-analysis baseline while also supporting masked completion from partial labels. Together with a prototype interface for multi-level candidate inspection and editing, these results position automatic RN analysis not only as a prediction problem, but also as a foundation for future interactive tools for music analysis. 

---
# Agile perceptive multi-skill locomotion for quadrupedal robots in the wild 

**Authors**: Jun-Gill Kang, Jaehyun Park, Tae-Gyu Song, Joon-Ha Kim, Seungwoo Hong, Hae-Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2607.13579)  

**Abstract**: Enabling quadrupedal robots to traverse complex terrains-from rugged outdoor environments to urban landscapes-requires seamless integration of multiple motor skills, smooth transitions between gaits, and high-speed perceptive locomotion using only onboard sensors. We present APT-RL (Action Pretrained Transformer-based Reinforcement Learning), a unified framework that enables multi-skill locomotion to achieve high-speed traversal in complex environments through autonomous skill transitions utilizing only onboard perception and computation. Our approach generates large-scale, feature-rich 2D motion datasets through trajectory optimization with simplified dynamics. These datasets enable training of diverse, reusable locomotion skills that transfer effectively to a real quadruped robot operating on complex uneven terrains. The resulting high-quality skills serve as strong priors for efficient learning of complex downstream tasks and extend naturally to 3D environments, enabling smooth, high-speed multi-skill locomotion in deployed policy. Real-world experiments demonstrate the framework's capabilities: the robot performs agile maneuvers through complex indoor obstacles and outdoor wild environments, including dynamic drop-down maneuvers that reach instantaneous peak speeds of up to 6 meters per second. A single onboard policy enables robust traversal of diverse obstacles, including stairs, hurdles, stepping stones, gaps, and fallen branches, demonstrating the versatility and effectiveness of our approach. 

---
# IMMNet: Hybrid Fusion of Model-based and Data-driven Approaches for Maneuvering Target Tracking 

**Authors**: Yixuan Zhao, Chaoqun Yang, Lin Gao, Yongxiao Tian, Ting Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2607.13573)  

**Abstract**: Maneuvering target tracking in three-dimensional space remains a challenging problem due to complex motion dynamics and model mismatch. To address this, this paper proposes a hybrid model/data-driven algorithm named IMMNet, which integrates the interpretable structure of the interacting multiple model (IMM) algorithm with learnable neural components. Unlike end-to-end black-box methods, the proposed IMMNet algorithm not only can preserve the Bayesian inference mechanism that is essential for real-time radar applications, but also can adaptively learn motion patterns and noise characteristics from data. Extensive experiments demonstrate that the proposed IMMNet algorithm consistently outperforms the existing algorithms across various scenarios, validating it as a robust, interpretable, and practical solution for maneuvering target tracking. 

---
# Cover First, Disagree Softly: Rethinking Mismatch-First Active Learning for Frame-Level Audio Classification 

**Authors**: Shiqi Zhang, Tuomas Virtanen  

**Link**: [PDF](https://arxiv.org/pdf/2607.13571)  

**Abstract**: Sound event detection relies on frame-level strong labels whose annotation is expensive. Active learning addresses this problem by selecting the audio segments whose labels help the classifier most. One of the prevailing acquisition strategies for this task, mismatch-first farthest-traversal (MFFT), combines the disagreement between two classifiers and the diversity of the selected segments through hard sequential decisions. It selects whole groups of high-disagreement segments first and spreads only the remaining budget by farthest traversal. On two multi-label datasets we show that this design is blind to the similarity among the selected segments and fails under low budgets, with every mismatch-first variant ending below the plain geometric strategy it builds on. We propose mismatch-weighted facility location (MW-FL), which spends the entire budget through a disagreement-weighted coverage objective that penalizes similarity among the selected segments. The disagreement signal from MFFT is used to obtain the nonnegative weights of this facility-location objective, without introducing hyperparameters. Experiments across two geometric mechanisms with three ways of using disagreement show that coverage of the selected segments is the dominant factor, hard disagreement gating of selection is harmful on both mechanisms, and soft disagreement weighting helps on top of coverage. MW-FL attains the best area under the learning curve on both datasets. 

---
# GHR-VLM: Making Zero-Shot Transit Video Analytics Realizable with Grounded Hybrid Reasoning 

**Authors**: Kaicong Huang, Weiheng Oh, Ruimin Ke  

**Link**: [PDF](https://arxiv.org/pdf/2607.13569)  

**Abstract**: Transit video understanding can provide valuable fine-grained data that conventional passenger counters and fare systems cannot capture. However, supervised video models require task-specific annotations, while applying vision-language models (VLMs) directly to long onboard videos is unreliable and costly. To leverage the complementary strengths of both approaches, we propose GHR-VLM, a visual grounded hybrid reasoning framework for zero-shot transit-bus video analytics. It is motivated by the observation that explicit visual grounding can improve VLM reasoning by converting long surveillance streams into compact, passenger-centered spatiotemporal evidence. Specifically, we propose an edge-cloud design in which a lightweight edge-based monitor continuously tracks door status and segments passenger clips. A backend VLM then identifies boarding passengers and classifies payment behavior through a two-stage coarse-to-fine refinement of spatiotemporal evidence. By invoking the VLM only on grounded passenger clips and contact sheets, GHR-VLM reduces cloud inference, avoids payment-specific training data, and supplies the localized evidence that VLMs otherwise struggle to identify. Evaluation on 486 minutes of real-world bus surveillance video demonstrates the potential of grounded edge-cloud reasoning for passenger-level payment analytics while highlighting the challenges posed by degraded video conditions. 

---
# Spectral-Informed Neural Networks Outperform Spectral Methods in High-dimensional PDEs 

**Authors**: Tianchi Yu, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2607.13566)  

**Abstract**: For low-dimensional problems ($d\leq3$), spectral methods can achieve exceptionally high accuracy. For middle-dimensional problems ($4 \leq d \lesssim 10$), spectral methods remain feasible through specific techniques such as sparse grids or hyperbolic cross. However, for high-dimensional problems ($d\gg 10$), spectral methods suffer frome the curse of dimensionality. Physics-informed neural networks (PINNs) have emerged as a promising approach to overcome this challenge, offering scalability to high dimensions, but often suffer from limited accuracy and efficiency. Recently proposed spectral-informed neural networks (SINNs) combine spectral methods with PINNs, operating directly in the spectral domain to avoid spatial derivative computations and to reduce memory consumption. In this work, we introduce Modified SINNs, which integrate coefficient decay scaling and basis embeddings motivated by harmonic analysis to enhance accuracy in high-dimensional problems and enable accurate approximation of unknown spectral coefficients. Numerical experiments on steady and time-dependent partial differential equations demonstrate that Modified SINNs outperform sparse grid spectral methods on middle-dimensional problems with incomplete spectral information and achieve superior accuracy compared to PINNs on high-dimensional problems. 

---
# UTS at ELOQUENT 2026 Voight-Kampff: structural shifts in AI writing bypass state-of-the-art detectors 

**Authors**: Dima Galat, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2607.13565)  

**Abstract**: We investigate which language model evasion attacks survive state-of-the-art adversarial fine-tuning, developing strategies that sweep the top 5 positions on the ELOQUENT 2026 Voight-Kampff leaderboard. While adversarial fine-tuning trivially closes the 2025 winning evasion recipes, we uncover a fundamental asymmetry in detector vulnerability: pushing generated text out of the detector's training distribution reliably defeats adversarial detection, whereas pulling it into the distribution (e.g., mimicking human training data) fails completely. Exploiting this, we introduce two novel out-of-distribution attack families - cross-decade register attacks and modernist stream-of-consciousness form. Both strategies easily bypass adversarial closure, achieving up to approximately 50x higher fool rates than previous methods while preserving naturalness. Furthermore, experiments show that the obvious deployer countermeasure (augmenting training data with period prose) fails to close the vulnerability. Our findings show that the tested detector families, including adversarially fine-tuned ones, exhibit persistent vulnerabilities under structural out-of-distribution shifts, a mechanism that directly powers our leading competition performance. 

---
# Grounded world models in biological organisms and future embodied AI 

**Authors**: Giovanni Pezzulo, Davide Nuzzi, Marco D'Alessandro, Riccardo Proietti, Roberto Bottini, Paul Cisek  

**Link**: [PDF](https://arxiv.org/pdf/2607.13560)  

**Abstract**: Recent advances in generative and embodied AI have been driven by large-scale predictive learning over multimodal data. However, the resulting systems remain largely based on passive training regimes where linguistic regularities create the scaffold onto which information from other modalities is attached. Conversely, neuroscience and cognitive science suggest that biological intelligence is organized in the opposite way, where grounded world models acquired through interaction with the environment provide the semantic scaffold to which language is attached. Here, we illustrate five examples of neural circuits supporting grounded world modelling, which underlie navigation in physical and conceptual spaces, affordance-based perception and interaction with objects, active perception and exploratory learning, allostatic control and emotion, and the distinction between self- and world-generated outcomes. These examples highlight several features largely missing from current embodied AI, including the role of intrinsic dynamics as a foundation for learning, the centrality of action in aligning these dynamics with the external world, the prominence of autonomous experience and open-ended learning over passive assimilation of externally provided data, and the fact that early predictive and control mechanisms scaffold higher cognitive abilities such as reasoning, conceptual navigation, planning, imagination, understanding others' minds, and communication. Finally, we discuss whether and how principles derived from biological systems may inform future embodied AI, including training regimes based on social interaction to construct world models that are not only grounded but also socially shared and aligned with human norms and values. 

---
# Greedy Volume Maximization of Gradient Embeddings for Long-Tailed Frame-Level Bioacoustic Active Learning 

**Authors**: Shiqi Zhang, Marius Faiß, Ariana Strandburg-Peshkin, Tuomas Virtanen  

**Link**: [PDF](https://arxiv.org/pdf/2607.13555)  

**Abstract**: Bioacoustic call-type classification relies on costly expert annotation. Active learning can reduce this burden by selecting a small batch of segments for expert annotation and using the labeled segments for training the classifier. The setting is hard: the target calls are extremely sparse and the call-type distribution is long-tailed, so a tight budget must be spent on the few rare, informative segments. We propose BADGE-Greedy-DPP, a deterministic batch selector that greedily adds the segment whose BADGE gradient embedding most enlarges the volume spanned by the batch; because this log-volume objective is submodular, the greedy rule guarantees a batch value at least a (1-1/e) fraction of the optimum of this objective, a guarantee not provided by BADGE's existing k-means++ and MCMC DPP sampling heuristics. There is also a temporal granularity mismatch in the task. The acquisition function scores whole segments, yet the informative frames inside them are few. Uniform averaging therefore washes them out. We show that the BADGE construction naturally addresses this mismatch when applied frame-wise, as prediction residuals weight the aggregated pseudo-gradient, so confidently predicted no-call frames contribute little while a single uncertain rare-call frame can still set the segment's direction. Across 10 runs on a sparse, imbalanced hyena call-type dataset, BADGE-Greedy-DPP achieves the best overall and rare-call-type performance among all compared query strategies, including MFFT, the strongest non-BADGE baseline, and the two vanilla BADGE traversals. 

---
# ExTernD: Expanded-Rank Ternary Decomposition Ternary LLM PTQ with Accuracy Approaching Any Quantization Level 

**Authors**: Chethan Reddy G.P  

**Link**: [PDF](https://arxiv.org/pdf/2607.13511)  

**Abstract**: We introduce ExTernD (Expanded-rank Ternary Decomposition), a post-training factorization of each LLM weight matrix $A \in \mathbb{R}^{m \times n}$ into $A \approx B \mathrm{diag}(D) C$ with ternary factors $B \in \{-1,0,+1\}^{m \times k}$, $C \in \{-1,0,+1\}^{k \times n}$ and a real scale vector $D \in \mathbb{R}^k$. The inner rank $k = \mu \min(m,n)$ is deliberately expanded beyond full rank ($\mu > 1$), so that components past full rank correct the quantization error of earlier ones. We prove the residual decreases monotonically in $k$ and can be driven below any $\varepsilon > 0$: ExTernD approaches bf16 accuracy arbitrarily closely, which no ternary scheme with a fixed plane count can do. Memory and compute scale continuously with $\mu$, and factor sparsity continuously with a threshold $\tau$, so an accuracy target is hit exactly rather than rounded to the next bit-width. ExTernD matches Q4_K's per-matrix accuracy at 5.2-5.5 effective bpw (5.1-5.5 with importance weighting) on Gemma-4-E2B and Qwen3.5-4B, and a full Qwen3.5-4B conversion at $\mu = 3$ reaches 10.10 wikitext-2 perplexity against 9.78 for bf16 (+3.2%), placing it near the Q4_K/Q5_K accuracy band at ~5.7 effective bpw. 

---
# DeepLoop: Depth Scaling for Looped Transformers 

**Authors**: Shuzhen Li, Yifan Zhang, Jiacheng Guo, Quanquan Gu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13491)  

**Abstract**: Looped Transformers scale sequential computation by applying a compact stack of physical blocks for multiple rounds, increasing unrolled depth without increasing stored parameters. This reuse changes the residual-scaling problem: in an untied Transformer, each residual branch receives and applies its own parameter update, whereas in a looped Transformer one shared update aggregates gradients from repeated visits and is read back by those same visits in the next linearized forward pass. We formalize this tied-depth effect through a first-order perturbation bound controlled by a visit-alignment coefficient $\kappa_R$. The bound recovers the DeepNorm exponent when visits decorrelate, but in the conservative aligned regime it requires the exponent to increase from $1/4$ to $1/2$ as loop count grows at fixed physical depth. The resulting method, \textbf{DeepLoop}, keeps the Post-LN DeepNorm architecture and sets $\alpha=(2N)^{1/2}$ and $\beta=(8N)^{-1/2}$ for unrolled depth $N$. On GPT-style looped language models at GPT-2 small and GPT-2 medium scale, DeepLoop is neutral when no physical block is revisited and improves validation loss and downstream accuracy once recurrent depth is activated. These results show that stable recurrent depth requires residual scaling rules that account for parameter visits, not only nominal layer count. 

---
# Explainable Artificial Intelligence for Anomaly Detection in Banking Transactions: An Internal Audit Perspective 

**Authors**: Anupa Lodhi  

**Link**: [PDF](https://arxiv.org/pdf/2607.13469)  

**Abstract**: The banking sector increasingly relies on automated systems to monitor electronic transactions for signs of fraud, yet conventional rule-based approaches struggle with high false-positive rates and offer no justification for their outputs, limiting their utility for compliance teams. This paper introduces an Explainable Artificial Intelligence (XAI) framework tailored for banking transaction anomaly detection within internal audit workflows. An Isolation Forest (iForest) model performs unsupervised anomaly scoring, while a SHAP (SHapley Additive exPlanations) layer provides transaction-level, feature-attributed explanations grounded in cooperative game theory [8]. A lightweight Streamlit dashboard renders these outputs in a form accessible to audit professionals without machine learning expertise. Evaluation on a synthetic banking dataset yields 0.91 precision and 0.88 recall, outperforming three unsupervised baselines. Expert feedback confirms that feature-level explanations measurably improve auditor confidence and decision quality. The framework advances the practical deployment of accountable, transparent AI in regulated financial environments. 

---
# DevicesWorld: Benchmarking Cross-Device Agents in Heterogeneous Environments 

**Authors**: Huatao Li, Xinwei Geng, Yuheng Wang, Yutong Li, Runde Yang, Hantao Chen, Shu Yao, Jingru Fan, Xuhui Ren, Yuanyuan Zhao, Fei Huang, Chen Qian  

**Link**: [PDF](https://arxiv.org/pdf/2607.13465)  

**Abstract**: LLM-based agents have rapidly improved at operating individual digital environments such as mobile applications, desktop systems, and smart homes. However, real-world user goals often span multiple devices: information may come from a phone, be processed on a desktop, and the result may need to appear on another device. Most existing benchmarks center on a single dominant execution environment, making it difficult to evaluate whether agents can acquire and integrate information across heterogeneous devices and complete end-to-end tasks with cross-device dependencies. We introduce DevicesWorld, a large-scale executable benchmark for cross-device collaborative operation. DevicesWorld contains 6,140 tasks and integrates three classes of device environments -- mobile, desktop, and IoT -- into a unified cross-device interaction and evaluation framework. Each task defines a natural-language user goal, participating devices and initial states, executable actions, rule-based verifiers, and a cleanup procedure. A multi-stage construction and quality-control pipeline keeps tasks close to realistic user needs while allowing final outcomes to be automatically verified from device states and generated files. We evaluate five frontier LLM-agent systems on a fixed evaluation set. All methods achieve low success rates, with the best reaching only 12.5%. Among failed runs, about 28.7% satisfy at least one scoring condition yet still fail the full task. Trajectories show that agents become stuck acquiring information or manipulating interfaces, confuse source and output devices, or terminate before all conditions are jointly satisfied. DevicesWorld turns cross-device collaborative operation into an executable, reproducible, and diagnostically useful evaluation problem for research on reliable cross-device agents. 

---
# GeoAnchor: Collaborative Reasoning via Latent Decomposition for 3D Spatial Understanding 

**Authors**: Hao Li, Han Fang, Zixin Pan, Xin Wei, Hongbo Sun, Jinglin Xu, Zhiyu Lin, Ye Yuan, Zhongjiang He, Yu Yu, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.13454)  

**Abstract**: Although multimodal large language models (MLLMs) have achieved remarkable progress, understanding 3D spatial relationships from 2D images remains a critical challenge. Existing methods primarily rely on symbolic text tokens, which inherently lack the fidelity to represent continuous geometric information. While recent methods use latent representations to enhance reasoning, relying on a single latent type cannot adapt to the diversity of spatial tasks, leading to misalignment in complex geometric scenarios. To address these limitations, we propose GeoAnchor, an interleaved text-latent reasoning framework. GeoAnchor decomposes 3D spatial information into three complementary components: position latents for object grounding, direction latents for relational orientation, and geometry latents for scene structure. These components are recombined in a structured space to construct local evidence while capturing global context, enabling dynamic and interpretable reasoning. Furthermore, we introduce a collaborative training strategy that guides the model from local spatial perception to comprehensive 3D understanding. Extensive experiments on diverse and complex 3D reasoning tasks demonstrate that GeoAnchor outperforms the state of the art, validating its effectiveness and generalization capabilities. 

---
# Adversarial Prompting Framework for AI Safety Assessment 

**Authors**: Yash Bhatnagar, Kunal Banerjee, Anirban Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2607.13453)  

**Abstract**: Artificial Intelligence (AI), especially Generative AI (GenAI), adoption has increased in industries significantly in recent years. However, the use of these models may also expose systems to new forms of cyberattacks by different malicious actors -- adversarial prompt attack (APA) being one of the most prominent examples of such threats. This paper presents the implementation of an Adversarial Prompting Framework (APF) for a comprehensive assessment of AI safety. The framework systematically evaluates the resilience of the AI model through the generation of structured adversarial prompts at multiple sophistication levels, from direct harmful requests to advanced encoding-based attacks. Our implementation demonstrates the practical application of this methodology in enterprise environments, providing automated testing capabilities with quantitative security assessment metrics. The results indicate significant variations in the model vulnerabilities across different attack vectors, with encoded prompts presenting the highest success rates in bypassing safety mechanisms. 

---
# Symbiosis-Inspired Knowledge Distillation for Incremental Object Detection 

**Authors**: Mingyue Zeng, De Cheng, Zhipeng Xu, Huaijie Wang, Nannan Wang, Xinbo Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.13452)  

**Abstract**: Incremental object detection (IOD) aims to extend detectors to new categories while retaining previously acquired knowledge. Existing methods often adopt a class incremental learning perspective, separating feature spaces to sharpen decision boundaries. However, this separation-oriented paradigm may overlook object symbiosis in detection, where co-occurrence and occlusion introduce spatial and semantic dependencies that benefit from shared representations. Ignoring these dependencies distorts the shared representations, exacerbates confusion between old and new classes, and accelerates catastrophic forgetting. To address this, we propose Symbiosis-Inspired Knowledge Distillation (SIKD), which explicitly leverages object symbiosis at two complementary levels. Spatial Symbiosis Distillation (SpSD) focuses on symbiotic regions where the old model responds with high overlap to objects in the new task. It preserves generalizable old class cues, suppresses class-specific bias and redundancy, and distills the refined evidence to the new model at matched spatial locations with slot-aligned supervision. Semantic Symbiosis Distillation (SeSD) maintains class level structure by forming confidence weighted prototypes for old classes and aligning their inter class soft ranks over the old class logits, which stabilizes the semantic topology during adaptation. Extensive experiments demonstrate the effectiveness and superiority of the proposed method. 

---
# Learning Physics-Guided Residual Dynamics for Deformable Object Simulation 

**Authors**: Shivansh Patel, Kaifeng Zhang, Sanjay Pokkali, Svetlana Lazebnik, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.13451)  

**Abstract**: Simulating deformable objects is essential for a wide range of robotic manipulation applications, yet accurately predicting their dynamics remains challenging. We propose Physics-Guided Residual Dynamics (PGRD), a hybrid simulation framework that combines the advantages of physics-based and learning-based approaches. Specifically, PGRD combines an optimizable spring-mass simulator as a backbone with a learned neural network that predicts residual corrections to the physics-based predictions. We adopt a velocity-based formulation to ensure stable simulation and a sliding-window transformer architecture to capture temporal dependencies. We show that PGRD produces more accurate results than both purely physics-based and learning-based methods on a set of diverse real-world deformable objects. We further demonstrate the utility of PGRD in two applications: manipulation planning via Model Predictive Control, including a language-conditioned setting with a generated goal image; and interactive simulation via action-conditioned video prediction by 3D Gaussian Splatting. 

---
# Discrete Diffusion Models: A Unified Framework from Tokenization to Generation 

**Authors**: Ye Yuan, Weien Li, Rui Song, Zeyu Li, Haochen Liu, Xiangyu Kong, Zixuan Dong, Linfeng Du, Zipeng Sun, Weixu Zhang, Jiaxin Huang, Changjiang Han, Yonghan Yang, Zichen Zhao, Xiuyuan Hu, Haolun Wu, Yankai Chen, Fengran Mo, Jikun Kang, Bowei He, Philip S. Yu, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.13431)  

**Abstract**: Discrete denoising diffusion models (DDMs) have recently emerged as a compelling alternative to autoregressive (AR) modeling for discrete data, offering parallel generation and iterative global refinement capabilities. Unlike continuous diffusion, where the state space is fixed, DDMs are fundamentally shaped by how the discrete state space is constructed: the tokenization scheme, the vocabulary topology, and domain-specific structural alphabets. This work introduces a unified conceptual framework that views discrete diffusion models through the construction of the underlying discrete state space. Within this framework, existing formulations, including transition-matrix, masking/absorbing-state, and score/ratio-based approaches, emerge as different instantiations of a common design space. The framework further exposes common design trade-offs across training objectives, inference algorithms, scaling behavior, systems optimization, and evaluation protocols, suggesting several promising directions for future research. 

---
# Data-Efficient Adaptation of LLMs via Attention Head Reweighting 

**Authors**: Tuomas Oikarinen, Zixiao Chen, Charlotte Siska, Tsui-Wei Weng, Chandan Singh, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.13425)  

**Abstract**: Learning effectively from limited data is critical in domains like security where labeled examples are scarce. Large language models (LLMs) have demonstrated some capabilities for data-efficient learning, especially through parameter-efficient adaptation methods, but continue to struggle when faced with few samples for difficult tasks. To meet this challenge, we propose Attention Head Reweighting (AHR), a data-efficient method that adapts LLMs to new text-classification tasks by learning only a single scalar per attention head. This drastically reduces the number of parameters that need to be learned by making use of the functional specialization of individual attention heads. Experiments on diverse open-source text classification datasets show that AHR can outperform standard baselines like LoRA when learning from limited samples, despite having 200-1000x fewer trainable parameters, as our AHR only modifies ~0.0001% of the model's parameters. In addition, our learned weights are easy to interpret and can be analyzed to better understand the mechanisms and attention heads responsible for in-context learning abilities in LLMs. 

---
# ScanFocus: A Coarse-to-Fine Framework for Spatio-Temporal Video Grounding 

**Authors**: Kai Chen, Ming Dai, Wenxuan Cheng, Wankou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13421)  

**Abstract**: Spatio-Temporal Video Grounding (STVG) aims to retrieve the visual trajectory of a specific object from a video stream as described by a natural language expression. However, most advanced methods struggle to balance global context modeling with precise boundary localization. Due to the prohibitive computational costs of processing long videos, these approaches typically resort to low-rate temporal downsampling and implicit motion modeling. This inevitably suppresses high-frequency boundary cues and neglects the explicit inter-frame dependencies required for precise boundary delineation. To address these limitations, we present \textbf{ScanFocus}, a novel coarse-to-fine framework that decouples the STVG task into a global spatio-temporal scan and a local boundary focus. Specifically, we utilize a unified vision-language fusion encoder combined with a lightweight Deformable Semantic-Motion Fusion module to efficiently align multimodal features and generate coarse proposals. To recover the suppressed fine-grained details, we introduce the Semantic-Guided Temporal Aggregator (SGTA) in the refinement stage. By densely sampling around coarse boundaries, SGTA explicitly models short-term temporal interactions under semantic guidance, capturing rapid motion changes for precise timestamp regression. Extensive experiments on three widely used benchmarks demonstrate the performance superiority of our proposed method over previous approaches. Code will be released at this https URL. 

---
# Can We Steer the Black-Box? Towards Controllability-Centric Evaluation of Recommender Systems with Collaborative Agents 

**Authors**: Jiwen Zhou, Xiang Liu, Mingming Li, Pengbo Mo, Jiao Dai, Honglei Lv, Jizhong Han, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.13418)  

**Abstract**: Recommender systems operate as Black-Boxes, leaving users and regulators unable to steer their outputs toward specific intentions or audit their behavior. This lack of controllability, defined as the system's ability to respond to explicit guidance, remains an unaddressed dimension in existing evaluation paradigms. To fill this gap, we propose CtrlBench-Rec, a collaborative multi-agent framework for systematic assessment of controllability. We formalize three fundamental tasks: target content discovery, interest profile shaping, and popularity bias mitigation, which together measure steerability from explicit commands to implicit representation steering and finally to overcoming algorithmic this http URL experiments on real-world datasets and multiple recommendation models demonstrate that our framework effectively quantifies controllability and exposes critical system bottlenecks, most notably persistent resistance to guiding long tail content. CtrlBench-Rec provides the first standardized toolkit for controllable recommendation research, algorithmic auditing, and user empowerment. Our code is released on this https URL. 

---
# Is the Statistical Advantage Worth the Cost? An Empirical Comparison of KANs and MLPs for Structured Data Classification 

**Authors**: Matthew Steven P. Toledo, Justine Raphael H. Jacinto, Vivekjeet Singh Chambal, Rodolfo C. Camaclang III, Jamlech Iram N. Gojo Cruz, Reginald Neil C. Recario  

**Link**: [PDF](https://arxiv.org/pdf/2607.13413)  

**Abstract**: This study presents an empirical benchmarking comparison between Kolmogorov-Arnold Networks (KANs) and Multi-Layer Perceptrons (MLPs) on structured tabular classification tasks. Motivated by the growing interest in KANs as an alternative function-approximating architecture, we evaluate their out-of-the-box performance on twelve publicly available datasets spanning binary, multiclass, multilabel, and ordinal problems. Both models were trained under standardized preprocessing, architecture, and fixed hyperparameter settings, with performance assessed using test accuracy and F1-Score, paired hypothesis testing, and effect size analysis. Results show that KANs statistically outperform MLPs in binary and multiclass domains and achieve a significant aggregate advantage across all datasets. However, the observed medium effect size (d = -0.46) raises an important cost-benefit consideration: while KANs offer superior generalization through adaptive spline-based mappings, this advantage comes with substantially higher parameter and computational complexity relative to the MLP baseline. These findings suggest KANs are the preferred choice for high-precision applications, while MLPs remain a robust and efficient option for resource-constrained environments. Future work should extend this analysis to additional data modalities to further refine these architectural selection criteria. 

---
# Improving Text-to-Audio Instruction Following via Fine-Grained Feedback from Audio-Aware Large Language Models 

**Authors**: Chun-Yi Kuan, Siwon Kim, Byeonggeun Kim, Suyoun Kim, Bo-Ru Lu, Qinming Tang, Ankur Gandhe, Hung-yi Lee, Chieh-Chi Kao, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13408)  

**Abstract**: Recent text-to-audio models generate high-quality audio, but often fail to follow instructions involving multiple sound events and temporal order. This gap arises because existing evaluation and training signals mainly emphasize global similarity or perceptual quality, with limited supervision on instruction-level correctness. We propose an instruction-level framework that uses audio-aware large language models (ALLMs) as fine-grained judges to verify target event presence and temporal relations in generated audio. After validating ALLM judgments on benchmarks and through human verification, we use their feedback to construct preference pairs for direct preference optimization. We further introduce S3Bench, a narrative benchmark for evaluating multi-event temporal instruction following. Experiments show that our method improves event completeness, temporal ordering, and joint instruction-following accuracy across existing benchmarks and S3Bench, while maintaining audio quality. 

---
# Price of Fairness in Bandits: A Tight Minimax Characterization 

**Authors**: Dhruv Sarkar, Soumyadeep Dutta, Sayak Ray Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2607.13402)  

**Abstract**: In bandit problems, standard regret-minimizing algorithms treat exploration as an amortized cost, which can expose early participants to unfair ex-ante losses in settings such as clinical trials. Recent work addresses this by evaluating the sequence of per-round expected rewards through the generalized $p$-mean, interpolating between utilitarian welfare ($p=1$), Nash welfare ($p\to0$), and Rawlsian fairness ($p\to-\infty$). Although tight guarantees are known for $p\ge0$, the strictly fair regime $q=-p>0$ remains unresolved because negative-power means are dominated by the smallest per-round rewards.
For $\sigma$-sub-Gaussian rewards with nonnegative means, the best prior algorithm relied on uniform early exploration and achieved regret $O(k^{(q+1)/2}/\sqrt{T})$, while the only general lower bound was the classical $\Omega(\sigma\sqrt{k/T})$. Thus it was unclear whether the extra dependence on $k$ was intrinsic to strict fairness or an artifact of uniform exploration.
We close this gap by identifying the exact polynomial price of strict fairness. Using a needle-in-haystack construction, we prove an algorithm-independent lower bound $\Omega(\sigma\sqrt{k^{\max(1,q)}/T})$; for $q>1$, this shows that the penalty $k^{q/2}$ is information-theoretically unavoidable. We then introduce \textsf{UCB-HARE} (Harmonic Anchored Rank Exploration), which replaces uniform exploration with an inverse-weighted harmonic rank schedule protected by a certified positive-mean anchor. Its regret is $\widetilde{O}(\sigma\sqrt{k^{\max(1,q)}/T})$, matching the lower bound up to logarithmic factors. Experiments on synthetic instances confirm that \textsf{UCB-HARE} improves over uniform-exploration baselines, with gains increasing as $q$ grows. 

---
# The Café in Amsterdam: When the Incumbent Becomes the Oracle 

**Authors**: Augusto Camargo  

**Link**: [PDF](https://arxiv.org/pdf/2607.13393)  

**Abstract**: A field can reformulate its computations freely exactly where its demand is stated independently of any incumbent implementation, and finds itself unable to when the incumbent's own output has quietly become the specification. This note offers that observation as a lens on computational reformulation for modern accelerators, where posing a problem in a hardware-friendly form can yield large speed and energy gains, but only if a replacement can be judged at all. Building on the test-oracle problem (Weyuker; Barr et al.), on requirements engineering's notion of implementation bias (Zave and Jackson), and on the roofline performance model, it names the pathology "baseline capture" -- the moment an incumbent stops being evidence that a demand can be met and becomes the definition of meeting it -- and separates two questions that are easily confused: whether a reformulation can be judged (which turns on the existence of an incumbent-independent demand) and whether its discovery can be automated (which turns additionally on the cost of evaluating that demand). Short cases -- shortest-path routing, learnable audio frontends, ZIP-215 for Ed25519 signature validation, CESM-ECT for climate models, and a single-GEMM audio frontend -- illustrate the pattern and the move of "buying a verifier": making a demand explicit, operational, and independent of the incumbent. No component is claimed novel in isolation; the contribution is the synthesis and the single question it makes easy to ask of any reformulation result -- does its acceptance test mention the incumbent's output? 

---
# Learning Engagement Assistant (LEA): Cross-Course Scalability and Classroom Evaluation of an Agentic AI Tutoring System 

**Authors**: Teri Rumble, Javad Zarrin, P. George Lovell, Ruth Falconer  

**Link**: [PDF](https://arxiv.org/pdf/2607.13370)  

**Abstract**: This paper is an extension of a paper presented at the ICAART 2026 conference, which introduced LEA (Learning Engagement Assistant), an adaptive AI tutoring agent combining course-specific Retrieval-Augmented Generation (RAG) with structured Knowledge Component (KC) models across integrated Chat, Tutor, and Quiz modes. That prior work validated LEA on a single STEM course (CMP511) exclusively through simulation, using synthetic learner agents. This paper extends that work by reporting the first classroom deployment of LEA with real students (n = 8, CMP511) and the first empirical test of its cross-course scalability, deploying the system across three courses spanning two academic levels and two disciplinary domains. The study reveals a divergence from simulation predictions across modes, showing that synthetic evaluation alone cannot anticipate all aspects of real deployment. A RAGAS-based cross-course scalability evaluation (660 questions) finds Answer Relevancy and Context Precision broadly stable across courses (0.88-0.94 and 0.88-0.90 respectively), while Faithfulness declines with curriculum distance from the system's original course (0.69 to 0.50), a preliminary finding that may reflect generation logic tuned to the system's original subject rather than a scalability limitation. These findings suggest that while the orchestration layer requires no modification, full course-agnosticism of all downstream components requires further investigation. 

---
# Evaluation Ability Does Not Imply Optimization Utility: LLM-as-a-Judge Signals in Closed-Loop Table Recognition 

**Authors**: Donghwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.13347)  

**Abstract**: LLM-as-a-judge is widely used to provide feedback and selection signals in closedloop regeneration, but this use remains insufficiently validated. We study it in table recognition, where deterministic TEDS evaluation provides a controlled testbed, using FinTabNet and OmniDocBench. Three findings emerge. First, judge signals were weak on both datasets: scores frequently tied, rankings were not reproducible, and the only selection policy that beat random on both datasets depended on an earliest-iteration tie rule, so its advantage cannot be attributed to the judge scores alone. Iteration produced better candidates, but the judge failed to recover them. Second, severe losses occurred even without specific judge feedback. A structurepreserving instruction significantly reduced the severe-loss rate on FinTabNet and was directionally consistent on OmniDocBench. The contrasts support target-preservation failure under unconstrained regeneration as a proximate mechanism of the observed severe losses. Third, the structure-preservation constraint reduced the severe-loss tail but produced no improvement. In an exploratory 2x2 analysis, the same protection was not stably observed when judge feedback was retained. These results do not dispute the value of LLMs as evaluators. Instead, they show that evaluation ability does not imply optimization utility. Iterative refinement requires, at minimum, a verification signal that deterministically detects structural change, rather than judge scores alone. 

---
# The Refusal Residue: When Probes Catch Alignment Faking and When They Don't 

**Authors**: Aman Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2607.13346)  

**Abstract**: Alignment faking is dangerous because a model can appear compliant under monitoring while preserving behavior it would reveal when unmonitored. When no scratchpad is visible, behavior alone cannot distinguish strategic from genuine compliance. We ask whether hidden states reveal what outputs hide.
We run a 13-model sweep for naturally-emerging faking, then probe and steer hidden states on the two models that fake. Natural faking appears only in Qwen3-32B (+18.2pp) and Llama-3.1-8B (+24.4pp at n=10, p<10^-15), while explicit scratchpad self-reports are rare (a Claude Opus 4 judge flags faking reasoning in 1.4% of Qwen and 0.0% of Llama compliant scratchpads). Both fakers show an asymmetric refusal residue: monitored compliance shifts toward refusal (d=0.32 Qwen, d=0.33 Llama, both p<0.0001), while refusal representations do not move (p=1.00).
Per-sample detection is model-conditional. Under leakage-free leave-one-query-out probing, detection survives on Llama (AUROC 0.87) but collapses to chance on Qwen (0.43), and cross-model transfer is null. Steering over 2,000 runs barely changes compliance (|h|<0.08), so the detected direction can flag faking but does not by itself control it.
Standard residualized probing leaks across folds and reaches AUROC 0.63 on a control where no faking can occur; naive linear probes reach a meaningless AUROC 1.0; and conventional MLPs overstate detectability by 0.2-0.3 AUROC. For future alignment-faking detection work, we release a five-control measurement framework: multi-token extraction, refuse-vs-refuse confound checks, per-fold residualization, leave-one-query-out evaluation, and orthogonality-constrained probing. 

---
# Efficient Text-to-Audio Generation via Pruning 

**Authors**: Arshdeep Singh, Yi Yuan, Yun Chen, Wenwu Wang, Mark D. Plumbley  

**Link**: [PDF](https://arxiv.org/pdf/2607.13330)  

**Abstract**: Diffusion-based text-to-audio generative models such as AudioLDM achieve high perceptual quality and strong semantic consistency; however, their practical deployment is hindered by the substantial computational cost of the U-Net denoising backbone. In this work, we apply model pruning to improve the computational efficiency of AudioLDM, a U-Net-based text-conditioned audio latent diffusion model. We analyse parameter redundancy across U-Net convolutional blocks and evaluate a filter-pruning strategy. Pruning is guided by norm-based criteria and followed by lightweight finetuning to recover performance losses. Experimental results demonstrate that up to 83% of the parameters and 39% of the multiply-accumulate operations of U-Net have been reduced while maintaining, and in some cases improving, generation quality compared to the baseline unpruned network. We find that pruning affects AudioLDM's ability to generate certain sound events including safety-critical sounds such as gunshots, sirens, and explosions, as well as mechanical sounds such as drills and sewing machines, and other sounds such as sprays and tick-tocks, which are mostly recovered by lightweight finetuning of the pruned model. 

---
# Privacy Preserving Recommender Systems Balancing Personalization with Privacy 

**Authors**: Ranjeet K Jha, Venkata Suresh Gummadilli  

**Link**: [PDF](https://arxiv.org/pdf/2607.13328)  

**Abstract**: Personalized recommendation systems are central to modern e-commerce and retail platforms, but they typically rely on centralized storage of detailed user interaction data, creating significant privacy and regulatory challenges. With increasing requirements from regulations such as GDPR, CCPA, and CPRA, organizations must develop recommendation systems that preserve user privacy without substantially degrading recommendation quality.
This work presents and evaluates a privacy-preserving recommendation framework that combines federated learning, differential privacy, cohort-level modeling, and privacy-aware intelligent agents. The framework keeps raw user data decentralized while introducing mathematically bounded noise to model updates. Experiments were conducted on synthetic retail datasets that emulate customer clickstream and purchase behavior. Recommendation quality was evaluated using Click-Through Rate (CTR), Precision@K, Recall@K, and Normalized Discounted Cumulative Gain (NDCG@K) across multiple differential privacy budgets.
We evaluate matrix factorization, neural collaborative filtering, and GRU4Rec under varying privacy constraints and analyze the trade-off between privacy and utility. An interactive Streamlit dashboard was developed to visualize recommendation performance, ranking stability, privacy-utility trade-offs, and fairness metrics. Results show that the proposed framework maintains competitive recommendation quality at moderate privacy budgets (approximately $\epsilon \approx 5$), demonstrating that strong privacy guarantees can be achieved with limited impact on recommendation effectiveness.
This work provides a practical framework for deploying privacy-preserving recommendation systems that balance personalization, regulatory compliance, and business objectives, offering a scalable approach for next-generation AI-driven retail platforms. 

---
# Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains 

**Authors**: Rwik Rana, Jesse Quattrociocchi, Christian Ellis, Nathan Tsoi, Garrett Warnell, Joydeep Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2607.13319)  

**Abstract**: High-speed off-road autonomy requires precise closed-loop control for a target vehicle while remaining robust across changing terrains. Recent forward kinodynamic (FKD) prediction foundation models suggest a promising path, starting from a generalist model and specializing it to the target platform. However, effective specialization remains challenging, as it often requires substantial real-world data, and models adapted to one setting can still overfit to specific terrains or driving regimes. We present OptCar (Optimized Car), a recipe for bridging the gap from generalist to specialist FKD models that preserves cross-terrain generalization while optimizing performance for a specific vehicle. $\texttt{OptCar}$ introduces a history-conditioned dynamics adaptation module that encodes recent state-action observations into a dynamics context token, and then fine-tunes the generalist model using limited real-world data together with targeted synthetic rollouts from environment-specific system identification. In closed-loop model predictive control (MPC) experiments across three terrains and an out-of-distribution cart-pulling task, the largest gains appear at 6~m/s, the highest speed evaluated and the regime in which slip dominates tracking error. On vegetation and dirt, the most slip-diverse terrain, OptCar reduces 6~m/s trajectory tracking error by roughly 55% relative to a fine-tuned AnyCar baseline, and remains the most accurate even when an unseen cart payload changes the dynamics. With only 5 minutes of real data per terrain, OptCar is competitive on road with a specialist trained on 30 minutes of road data, and substantially outperforms it once the terrain changes. 

---
# Tabular Foundation Models for Discrete Choice Estimation 

**Authors**: Liu Liu, Dan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13314)  

**Abstract**: Tabular foundation models (TFMs) generate predictions on structured data via in-context learning, without task-specific estimation. We ask whether TFMs can be effectively applied to discrete choice, a central demand estimation framework in marketing and operations, and find that directly applying TFMs yields limited performance. The gap is structural: TFMs assume row-independent observations, whereas discrete choice is inherently set-valued and subject to persistent consumer preference heterogeneity. We propose a reformulation that encodes both choice-set dependence and individual heterogeneity within a row-based learning framework. Evaluated on a yogurt scanner panel, individual-level heterogeneity encoding is the dominant driver of predictive accuracy. The best reformulation outperforms hierarchical Bayesian estimation by 8\% in holdout log-likelihood and 3.6\% in hit rate, running 16 times faster, a practical advantage for large-scale demand estimation. The advantage is largest in the medium-data regime (10--40 purchase occasions per consumer), where parametric Bayesian shrinkage most distorts estimates for atypical consumers. Fine-tuning on population choice data provides additional gains for consumers with shallow purchase histories, where in-context learning has limited individual-specific signal to condition on. These results establish a principled approach for applying foundation models to consumer choice problems more broadly. 

---
# Accuracy Without Grounding: Diagnosing Visual Dependency Dissociation in Video LLM Benchmarks 

**Authors**: Jae Joong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.13305)  

**Abstract**: Benchmark accuracy in video large language models (LLMs) is often treated as evidence of visual understanding. We audit this assumption across twenty models spanning 2-78B parameters and ten architecture families. We introduce the Visual Dependency Gap (VDG), the difference in per-question correctness between original-video and black-screen conditions. Paired McNemar tests on MVBench show that accuracy and visual dependency are separable: models differ on original video (p = 0.0003) but not on black screens (p = 0.53). Across models, task-type rankings are stable: Attribute Perception is strongly visual, whereas Temporal Reasoning approaches the language-only baseline. A diagnostic ladder from black screen to single frame, shuffled frames, and original video reveals that frame diversity supplies most of the visual benefit, while temporal order contributes near-zero accuracy across sixteen open-weight models. An ablation from 0.5 to 24 FPS rules out sparse sampling as the cause. H.264 experiments further show that stable aggregate accuracy conceals bidirectional question-level answer flips. The diagnostic also generalizes to four API-accessed models, whose VDG values range from 0.025 to 0.315. These results motivate VDG as a standard audit for whether video benchmarks measure visually grounded capability. Code is available at this https URL. 

---
# Faithful Autoformalization of Natural Language Assertions 

**Authors**: Hongyi Liu, Madhusudan Parthasarathy, Adithya Murali  

**Link**: [PDF](https://arxiv.org/pdf/2607.13303)  

**Abstract**: Formal contracts are essential for software testing and verification, yet writing them remains labor-intensive and error-prone. LLMs offer a promising path toward autoformalization: synthesizing executable assertions from natural-language specifications and thereby bridging the gap between informal developer intent and formal executable specifications. We present Monty: an autoformalization framework for assertions that tackles the challenges of expectations of validity of assertions and ambiguity in natural-language. Our techniques are based on filtering formalizations using a novel conformance score metric and validity scores obtained from testing the code against formalized assertions. We evaluate our approach on 541 assertion-generation tasks derived from 22 collection-like Java classes, and show that our technique produces the ground truth more reliably (improving upto 20 points in precision on average) than when using LLMs naively to translate assertions. 

---
# Deconstructing Actor-Critic: A Large-scale Empirical Study of Design Components for Practitioners 

**Authors**: Haseeb Shah, Lingwei Zhu, Adam White, Martha White  

**Link**: [PDF](https://arxiv.org/pdf/2607.13274)  

**Abstract**: Reinforcement learning is increasingly being considered for controlling real-world systems, from fusion plasma and autonomous vehicles to drug discovery and drinking water treatment, where reliability is essential and tuning budgets are limited. Actor-critic algorithms share a set of design decisions, such as how the policy is updated, how it represents the distribution over actions, how its gradient is estimated, and how often it is updated relative to the value estimator. Using a control task derived from a real water treatment plant, we analyze over 33,000 experiments to determine how these components affect variability across runs and sensitivity to hyperparameters. Common defaults, such as Gaussian action distributions with pathwise gradient estimators, are among the least reliable configurations, whereas bounded distributions with adaptive update schedules remain robust across a wide range of settings. These findings offer empirical guidance to practitioners across scientific and engineering domains for understanding and making component-level decisions when adapting actor-critic methods to new real-world control settings. 

---
# Discourse-Aware Policy Analysis with Argumentation: A Hybrid LLM-Symbolic Framework for Disaster Governance 

**Authors**: Stylianos Loukas Vasileiou, Olga Derendiaeva  

**Link**: [PDF](https://arxiv.org/pdf/2607.13260)  

**Abstract**: Policy documents shape governance outcomes, but their reasoning is often implicit. Participatory commitments and managerial control routinely coexist in the same text, and the tensions between them are rarely stated directly. Existing computational approaches to policy discourse cannot express the frame-mediated relations that drive these tensions, where one argument narrows or instrumentalizes another rather than rejecting it. End-to-end summarization by large language models produces fluent text but offers little structure that domain experts can inspect or contest. We present Apaf, a hybrid LLM--symbolic pipeline that operationalizes critical discourse analysis as a quantitative bipolar argumentation framework over policy text. Arguments are first classified into deliberative or managerial frames. Four frame-mediated relation subtypes (agency reduction, agenda shift, instrumental support, and normative support) are then produced by deterministic rules over LLM-extracted features. We release a novel dataset of 100 sub-documents of disaster-risk-reduction policy from the USA, UK, Canada, and Australia, and show that the resulting argument graphs are accurate, interpretable, and stable across jurisdictions. 

---
# Reassessing Muon for Matrix Factorization 

**Authors**: Ali Parviz, Gal Mishne, Alex Cloninger  

**Link**: [PDF](https://arxiv.org/pdf/2607.13246)  

**Abstract**: Muon has recently emerged as a strong optimizer for large-scale deep learning, where it reshapes gradient updates through approximate orthogonalization and has been reported to outperform Adam and AdamW in large language model training. Its empirical success has motivated a growing body of theoretical work that interprets Muon as steepest descent under the spectral norm. Yet it remains unclear which of Muon's advantages stem from its update rule itself and which are artifacts of the scale, architecture, and data of modern deep networks. In this work, we isolate the optimizer from these confounding factors by studying Muon on a simple, well-understood, and spectrally structured problem: low-rank matrix factorization. Through a controlled comparison against carefully tuned adaptive baselines, we find that Muon does not consistently outperform AdamW in this setting and that several previously reported advantages are sensitive to hyperparameter choices. Our results provide a more nuanced picture of when spectrum-aware orthogonalization is beneficial and argue for evaluating modern optimizers on controlled problems in addition to end-to-end benchmarks. 

---
# EMAGN: Efficient Multi-Attention Graph Network via Learned Clustering for Scalable Traffic Forecasting 

**Authors**: Mingxing Xu, Rakesh Chowdary Machineni, Ke Liu, Xi Cheng, Chengqi Lu, Xin Hu, Lyuhao Chen, Xiangyu Li, Junwei You, Oliver Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.13241)  

**Abstract**: Traffic forecasting is highly challenging due to complex and nonlinear spatial and temporal dependencies. Self-attention mechanisms have been widely adopted to model dynamic and long-range dependencies, achieving state-of-the-art performance, but suffer from limited scalability due to quadratic computational and memory complexity. To address this, we propose an Efficient Multi-Attention Graph Network (EMAGN) that linearises the spatial attention mechanism itself, inspired by the theory of fast high-dimensional Gaussian filtering. Two learned clustering matrices C_k and C_v adaptively group key and value vectors into M super-clusters, reducing complexity from O(N^2 d) to O(NMd) without sacrificing the flexibility of attention for dynamic dependency modelling. Experimental results on PEMS-BAY and METR-LA show that EMAGN achieves accuracy within 2.7-3.2% MAE of full-attention GMAN while reducing training time by 32%, inference time by 38%, and GPU memory by 58%. Critically, at K=16 attention heads, full-attention GMAN runs out of memory on a standard 11 GB GPU entirely while EMAGN continues to operate, demonstrating a categorical expansion of feasible model configurations. EMAGN also surpasses Linformer and Performer in both accuracy and efficiency within the same backbone, owing to its traffic-network-aware adaptive clustering. 

---
# Continuously Evolving Deepfake Detection: An Architecture and Public-Benchmark Evaluation of a Dynamic Detection System 

**Authors**: Ken Jon Miyachi, Dylan Uys  

**Link**: [PDF](https://arxiv.org/pdf/2607.13234)  

**Abstract**: Deepfake detectors that achieve near-perfect scores on academic benchmarks collapse on real-world content: recent in-the-wild evaluations report AUC drops of 45-50% for state-of-the-art open-source models. We argue this gap is structural: static detectors are trained once against a moving generative frontier. We present BitMind Forensics (BMF), trained through Bittensor SN34, an open adversarial competition that continually refreshes the training distribution. We evaluate one dated export comprising image, general-video, and human-video checkpoints across nineteen public datasets: the canonical face-swap suites (FaceForensics++, Celeb-DF v1/v2/++, DFDC, DFD, UADFV, DF40) and recent in-the-wild and AI-generated-media benchmarks (Sumsub, Deepfake-Eval-2024, WildRF, Community Forensics, AIGCDetectBench, GenImage, AI-GenBench, AIGIBench, RAID, GenVidBench, GenVideo-100K). BMF reaches 0.936 AUC on Sumsub's original images and 0.872 pooled AUC over its full four-condition manipulation battery (1.4M images), staying robust under perturbation (0.855 JPEG, 0.799 downscaled), while GPEN enhancement improves detection (0.996). On Deepfake-Eval-2024, it matches the best commercial detector on images (0.915 vs 0.90) and exceeds it on video (0.822 vs 0.79), far above the best open-source detectors (0.56 and 0.63). It reaches 0.991 AUC on a 21-generator AI-image panel and 0.918 on GenVidBench, and exceeds the FF++-trained frontier on DFDC (0.947 vs 0.843) and Celeb-DF v2 (0.9985 vs 0.956), both contamination-audited, with statistical parity on Celeb-DF++. In a temporal study, successive dated exports improve on held-out media from generators absent from the static baseline's training (image 0.842 to 0.902; video 0.864 to 0.936). Our evaluation harness is public, and at publication the production API serves the exact evaluated snapshot for independent verification. 

---
# Audited Selective Verification for Risk-Controlled N-1 Thermal Contingency Screening under Deployment Shift 

**Authors**: Jayakumar Manoharan  

**Link**: [PDF](https://arxiv.org/pdf/2607.13221)  

**Abstract**: Real-time N-1 contingency screening in an energy management system trades assurance against cost: verifying every credible outage with full power flow is too slow, while fast linear-sensitivity screening gives no statistical guarantee and can silently pass unsafe operating points, especially when a controller drives the system into unfamiliar regimes. This paper introduces Audited Selective Verification, a risk-budgeted screening and triage layer for any controller's output (optimization, model-predictive, or learned). A cheap surrogate proposes which outages to skip; an online audit runs full power flow on a small random sample each window; and a calibrated threshold certifies a thermal-violation-rate bound for the skipped set at a chosen budget and confidence, with a corresponding bound for the unverified trusted subset. Validity rests on real verification and the audit rather than on surrogate accuracy, so it holds under arbitrary deployment shift. It is a risk-budgeted screen, not a replacement for deterministic verification when policy requires checking every credible contingency. On three public transmission systems up to 1354 buses, the realized violation rate stays within budget, standard deterministic and calibrated screens become unsafe under shift, and the method cuts full power-flow studies by 29 to 75 percent per real-time operating point. 

---
# Classifying daily activities needs posture, reconstructing them needs motion 

**Authors**: Arefeh Farahmandi, Gunnar Blohm  

**Link**: [PDF](https://arxiv.org/pdf/2607.13216)  

**Abstract**: Humans recognize movements effortlessly, even from noisy and complex visual input. But what information in the stimulus allows humans to rapidly classify movements? No framework has systematically compared different strategies of movement analysis to address this question. Here, we used videos of 16 daily activities from the MoVi dataset and compared three strategies: Temporal Movement Primitives (TMPs), which decompose movements into weighted sums of temporally smooth basis functions; Legendre polynomial coefficients, which project joint-coordinate trajectories onto an orthogonal polynomial basis; and Autoencoder latent embeddings. Legendre coefficients and TMPs achieved the highest classifier accuracy, followed by autoencoders. We found two discriminative features for movement classification. The most informative is the general posture of the body, the average spatial configuration that distinguishes one activity from another. Additionally, we identified 9 critical joints that are most predictive for movement classification. Interestingly, good classification accuracy did not automatically lead to good movement generation: when we reconstructed movements for each activity, TMPs preserved the temporal dynamics and produced perceptually natural motion, whereas reconstructions from Legendre coefficients retained only the average posture and appeared frozen. These results reveal a dissociation in how movement information is organized: the static configuration of the body suffices to classify what activity is performed, but the temporal dynamics of movement are required to reconstruct how it unfolds. This distinction clarifies which features the visual system may rely upon for rapid action recognition, and suggests that postural features could enable efficient movement screening in clinical applications, while dynamic information remain essential wherever movement generation is the goal. 

---
# Adaptive Filtering of the KV Cache: Diagnosing and Correcting Structural-Role Bias in LLM Inference 

**Authors**: Soumil Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2607.13205)  

**Abstract**: Attention-based KV cache eviction (H2O and its descendants) compresses the memory-constrained state of a long-context model by ranking tokens on accumulated attention mass, treated here as signal energy, and keeping the heaviest. On schema-dense input streams such as nested JSON, this score acts as a non-stationary filter that disproportionately retains noise: a non-content sink role (delimiters or whitespace) carries an order of magnitude more energy than any content role, and structural KEY tokens are over-retained at roughly 1.8x the rate of the answer-carrying VALUE tokens, collapsing exact-match accuracy from 88% to 0% at a 5% budget as the signal-to-noise ratio of the retained state degrades. A counterfactual experiment establishes that suppressing KEY tokens is the best deployable filter. Our retraining-free, role-conditional allocation over SnapKV's windowed score, governed by a single tuned hyperparameter, closes 63-98% of the H2O gap at sub-20% budgets and, at higher budgets, modestly matches or exceeds full-cache accuracy -- a small, seed-sensitive denoising effect (borderline significant at B=0.50; not distinguishable from zero at B=0.30 over four seeds). A 15 MB linear role probe supplies these labels at negligible inference cost, though matching parser-level downstream accuracy remains open. 

---
# RAGthoven at SemEval-2026 Task 1: A Multi-Stage Pipeline Walks Into a Benchmark and Barely Clears the Bar 

**Authors**: Marek Šuppa, Viktória Ondrejová, Lucia Ganajová, Gregor Karetka, Daniel Skala  

**Link**: [PDF](https://arxiv.org/pdf/2607.13189)  

**Abstract**: We present RAGthoven, our system for SemEval-2026 Task 1 (MWAHAHA), Subtask A (multilingual constrained humor generation in English, Spanish, and Chinese). RAGthoven decomposes creative text generation into a multi-stage large language model (LLM) pipeline (Planner, Best-of-N Writer, Reflector for self-critique, LLM-as-a-judge Judge) grounded in computational humor theory (Benign Violation Theory, Script-based Semantic Theory of Humor) and refined across ten experiments.
In our final configuration, we augment the Planner with retrieval-augmented generation (RAG) from a curated joke corpus, seeding generation with diverse joke mechanisms. We also evaluate two agentic variants -- ReAct-style sequential tool-calling (Exp09) and autonomous multi-branch orchestration (Exp10) -- that expose the same four stages with a deterministic ConstraintAudit checker. Across four frontier models on a held-out 12-instance English sample, neither agentic variant produced outputs we judged superior to the non-agentic pipeline despite substantially higher tool-call budgets.
RAGthoven shares Rank 1 with the Gemini 2.5 Flash baseline in all three languages, with overlapping organizer-reported confidence intervals. In Spanish, it leads the baseline by 42 raw Elo points (1182 vs. 1140), while in English (1045 vs. 1081) and Chinese (1045 vs. 1053) the baseline holds the higher raw rating within the same statistical tie.
Together, these results suggest language-dependent diminishing returns from elaborate multi-stage prompt engineering and agentic scaffolding once a strong frontier model is in the loop. 

---
# SteinGate: Tail-Sensitive Safe Reinforcement Learning via Stein Discrepancy 

**Authors**: Yassine Chemingui, Chenhua Fan, Honghao Wei, Janardhan Rao Doppa  

**Link**: [PDF](https://arxiv.org/pdf/2607.13175)  

**Abstract**: Safe reinforcement learning typically enforces safety by bounding expected cumulative costs, a criterion that often fails to detect rare but catastrophic tail events. To overcome these limitations, this paper introduces SteinGate, a boundary-aware distributional safety certificate that replaces fragile tail fitting with a robust consistency check using Kernelized Stein Discrepancy while accounting for boundary atoms induced by clipped costs. SteinGate evaluates whether observed policy rollout costs remain consistent with a safe reference distribution, providing a non-parametric safety certificate. This certificate is used to dynamically adapt the learning regime: favoring reward-improving policy updates when rollouts remain consistent with the safe reference and switching to recovery behavior when the cost tail deviates. Experiments on continuous-control benchmarks demonstrate that SteinGate significantly reduces both the frequency and severity of constraint violations during training while maintaining competitive returns relative to state-of-the-art baselines. 

---
# What Models Express, Suppress, and Resist: Auditing Open-Weight LLMs with Persona Vectors 

**Authors**: Winston Zeng, Ali Emami, Jinho Choi  

**Link**: [PDF](https://arxiv.org/pdf/2607.13162)  

**Abstract**: What a language model will and will not do is largely set during post-training, but which behaviors it expresses, hides, or resists is not revealed by prompting alone. Persona vectors, behavioral directions in activation space, can probe this organization, but prior work covers only a handful of traits. We present the first systematic application of persona vectors at this scale, compiling a 53-trait inventory across four behaviorally distinct domains and labeling every trait in two open-weight models as natural (expressed at baseline), steerable latent but amplifiable, or intractable (resistant to standard extraction). Both models default to helpful, task-oriented behavior: all nine agentic traits are natural, and their default clinician behavior matches a board-certified psychologist's independent desirability judgments on 16 of 17 traits. Steering produces its largest gains on traits these defaults exclude: hyperbole, hallucination, and sycophancy. The same asymmetry holds across all 171 generic-trait pairs: two steerable traits can collapse the composition, but pairs involving a default never do. Where standard extraction fails on a trait like "evil," a vector transferred from a fine-tuned variant still recovers it, with the residual refusals appearing inside the model's chain-of-thought. Persona vectors are most informative not as a set of controls but as a probe of behavioral organization. 

---
# Active Beyond-Diagonal RIS Empowered Heterogeneous Edge Computing: A Distributional Reinforcement Learning Approach 

**Authors**: Tianyu Pang, Hongyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.13160)  

**Abstract**: Active beyond-diagonal reconfigurable intelligent surfaces (BD-RISs) enables hybrid transmitting and reflecting mode to achieve effective signal amplification and full-space coverage, thus providing a promising solution for blockage-aware uplink offloading in heterogeneous mobile edge computing (MEC) systems. However, practical hybrid mode active BD-RIS are realized by reciprocal devices, which inherently generate cross-sector energy leakage that will reshape the system-level energy-latency tradeoff. This paper studies energy-aware offloading and resource allocation for reciprocal active BD-RIS-assisted heterogeneous MEC, where offloading decisions, CPU/GPU computation allocation, transmit powers, receive processing, and active BD-RIS are tightly coupled. The resulting problem is a high-dimensional mixed integer nonconvex problem and is difficult to solve efficiently by conventional per-instance optimization. To address this challenge, we develop an end-to-end joint optimization framework based on a refined version of the distributional soft actor--critic algorithm, named as DSAC-T. By modeling return distributions rather than only expected values, DSAC-T improves policy stability under reward heterogeneity and feasibility-boundary sensitivity. Compared with other baseline algorithms, DSAC-T achieves the best energy-latency reward, the highest feasibility ratio of 81.67%, and a fast online decision time of 0.0267 s per scenario. 

---
# Boogu-Image-0.1: Boosting Open-Source Unified Multimodal Understanding and Generation 

**Authors**: Guoxuan Chen, Chufeng Xiao, Haoran Yang, Siyue Xie, Binxiao Huang, Ming Zhang, Cheuk Him Chau, Xinyu Fu, Yingzhao Lian, Tom S.Y. Li, Jintao Lin, Bowen Dong, Zian Qian, Yuhao Liu, Yuxuan Hu, Weikang Shi, Bin Zou, Bowen Zheng, Haoxuan Che, Chang Chen, Yuyang He, Heyang Sun, Tianyu Huang, Chong Hou Choi, Cheng Gong, Han Shi, Haoli Bai, Xihui Liu, Hongsheng Li, Qifeng Chen, Chao Huang, Rui Liu, Chenyang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2607.13125)  

**Abstract**: We introduce Boogu-Image-0.1, an open-source unified multimodal understanding and generation model family, comprising Base, Turbo, Edit, and Edit-Turbo variants. It delivers competitive performance in high-quality text-to-image generation, fast inference, instruction-based editing, and bilingual (Chinese-English) text rendering. Closed-source multimodal systems like Nano-Banana-Pro and GPT-Image-2 achieve strong performance through system-level integration rather than a single model, yet their internal practices remain largely undisclosed. In this work, we demonstrate that targeted improvements in model understanding, data quality, and training pipelines, coupled with agentic inference-time scaling, can substantially enhance generation and editing performance even under highly constrained compute budgets. Comprehensive evaluations show that Boogu-Image-0.1 consistently matches or surpasses other open-source models across standard benchmarks, and achieves results approaching leading closed-source systems. Notably, this is accomplished with only 208.62 million unique images. The base model's theoretical training cost is only approximately \$400K. We share practical discussions that we believe are valuable to the broader research community, and release weights, code, and recipes under Apache 2.0 to advance the open ecosystem for unified multimodal understanding and generation. Our code is available here: this https URL. 

---
# ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation 

**Authors**: Qingyu Zhang, Qianhao Yuan, Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun, Xiang Li, Ming Xu, Jiarui Li, Xiuyin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.13124)  

**Abstract**: Structured pruning is a hardware-friendly way to compress LLMs, but it is mostly validated on multiple-choice recognition tasks, while the same compressed checkpoints can collapse on the free-form generation that deployment actually requires. Two observations trace this gap. First, greedy \textsc{pass}@$1$ nearly vanishes after compression, yet \textsc{pass}@$k$ recovers substantially under repeated sampling: useful generations are demoted, not erased. Second, the recoverable regime fails mainly through suffix repetition. Recovery should therefore train on the compressed model's own on-policy states with dense token-level supervision, which On-Policy Distillation (OPD) provides by reusing the pre-compression model as a frozen teacher. However, long on-policy rollouts spend early recovery budget on low-information repetitive suffixes, delaying loss descent. To mitigate this waste, we propose \textbf{\shortopd}, a short-to-long OPD schedule that detects teacher-confirmed repetitive suffixes, treats the surviving prefix as each rollout's effective length, and allocates future rollout budgets to the effective lengths the policy can currently use. Across math, code, and open-ended generation, \shortopd\ raises the compressed model's score to about $9\times$ its unrecovered value and $1.6$--$4.4\times$ standard recovery recipes (SFT w/o KD, KD, and SeqKD), and it matches a fixed $8192$-token rollout horizon within two points using a quarter of the training time ($8.5$ vs.\ $35.9$ hours) and $71\%$ fewer rollout tokens. We hope this recipe helps move structured pruning beyond marginal gains on perplexity and multiple-choice benchmarks, a step closer to deployment-ready generation quality. 

---
# AI in Cyberpsychology: A systematic literature review of Cybersecurity enhancement by using AI for analyzing psychology of Victims, Attackers, and Defenders 

**Authors**: Georg Thamer Francis, Malek Malkawi, Sevim Eyüpoğlu, Reda Alhajj, Selim Akyokuş  

**Link**: [PDF](https://arxiv.org/pdf/2607.13123)  

**Abstract**: Cybersecurity is the practice of protecting systems, networks, and data from digital attacks. Cyberpsychology (CPSY) is defined as the use of psychology to enhance cybersecurity applications. Since the early 2010s, the evolution of Artificial Intelligence (AI) has increasingly integrated with CPSY, leveraging advanced data analysis to decode the distinct personality traits and behavioral patterns of victims, attackers, and defenders. In this systematic literature review (SLR), we carefully analyze 34 collected research studies of AI usage in cyberpsychology (AI-CPSY) using the preferred reporting items for systematic reviews and meta-analyses (PRISMA) methodology. The review presents a comprehensive taxonomy of the cyber-security applications, the AI methodologies used, and the psychological concepts employed across the studies . We sort the research studies into four cybersecurity applications: Anomaly Detection (AD), Vulnerability Risk Prediction (VRP), Security Awareness Training (SAT), and Authentication/Identity Verification (AIV). Within each application area, studies are further sorted according to the AI method used including machine learning (ML), deep learning (DL), natural language processing (NLP), and reinforcement learning (RL). Furthermore, the review identifies the most commonly utilized psychological concepts, quantify the datasets used in the field, and present their current implementation and deployment status. At last, it detect research gaps, present open challenges, and deduce the trending and most effective and emerging methodologies used across the AI-CPSY landscape. 

---
# CoDiffGRN: Rethinking Gene Regulatory Network Inference via the BEELINE-KGC Benchmark and Co-evolutionary Discrete Diffusion 

**Authors**: Jiaze Song, Runhao Zhao, Minghao Xu, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13120)  

**Abstract**: Inferring gene regulatory networks (GRNs) from single-cell transcriptomic data is crucial for biological discovery, yet existing approaches suffer from a fundamental misalignment with real-world needs. Researchers typically seek a small set of high-confidence regulatory interactions for experimental validation, often involving previously unseen genes. However, current benchmarks rely on transductive splits with global classification metrics, while prevailing models struggle to generalize under inductive settings. To bridge this gap, we reformulate GRN inference as an inductive, ranking-centric graph completion problem and introduce \textbf{\benchmark}, a new benchmark that incorporates an inductive gene-holdout split together with knowledge graph completion metrics to better evaluate top-ranked predictions. Building on this, we propose \textbf{\method}, the first co-evolutionary discrete diffusion framework that jointly models biologically coherent discretized gene expression states and regulatory interactions for robust inductive generalization and improved top-ranked regulatory discovery. We further introduce TF-ALL Subgraph Sampling (TASS) for scalable training. Extensive experiments on {\benchmark} show that {\method} establishes new state-of-the-art performance, significantly outperforming existing methods in novel regulatory discovery, and ablation studies further verify the effectiveness of our design. 

---
# SemaDiff: Identifying Semantic-Changing Commits with Generated Code and Tests 

**Authors**: Maha Ayub, Michael Konstantinou, Ahmed Khanfir, Nikolaos Tsantalis, Mike Papadakis  

**Link**: [PDF](https://arxiv.org/pdf/2607.13111)  

**Abstract**: Distinguishing semantic-preserving commits from changing ones remains an open challenge in software repository mining. While existing approaches detect refactoring commits accurately, they cannot ensure that a commit is purely semantic-preserving, without any interleaving behaviour-changing modification. This limitation can impact several tasks, such as debugging, fault localisation, bug dataset construction, rollback analysis, and bug fixes backporting. To fill this gap, we propose SemaDiff, a novel approach for identifying semantic-preserving commits through behaviour-based analysis; comparison of similar test execution on pre- and post-commit versions. As code impacted by the refactoring is often hard to test and different accross both versions, we propose generating additional calling methods to that code, which serve as testing target. Given a commit, SemaDiff analyses the diff to identify modified code and extracts unchanged dependent code that calls it. It then generates an additional dependent class using a large language model to exercise the changed code in both versions, and automatically generates tests for the dependent code. This way, we obtain the same tests for the different code versions, enabling the behavioural-difference detection. The commit is classified as semantic-preserving only if all generated tests produce identical outcomes across the two versions. To evaluate SemaDiff, we construct and annotate manually a dataset of 183 commits, gathered from well-known open-source Java projects. The obtained results show that SemaDiff distinguishes accurately semantic-preserving from -- changing commits in about 76% of the cases, with a 100% precision in semantic-changing commit detection. 

---
# A Hybrid Mamba for Audio-Visual Navigation 

**Authors**: Yi Wang, Yinfeng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.13110)  

**Abstract**: Since the paradigm centered on convolutional neural networks and recurrent architectures was established in 2020, the fundamental backbone networks for audio-visual navigation have undergone no essential changes for more than five years, making them inadequate to support efficient representation of dynamic multimodal sequences. This paper proposes Samba(A Hybrid Mamba for Audio-Visual Navigation). It uses the adaptive selection-enabled Mamba State Encoder (M-SE) to replace conventional GRUs for temporal aggregation, and constructs an Audio Mamba Encoder (AME) to remedy the limitations of convolutional operators in capturing global time-frequency dependencies in spectrograms. Experiments demonstrate that Samba exhibits exceptional generalization performance when facing unheard sound sources and unseen scenes. On the Matterport3D dataset, it improves the navigation success rate (SR) by 11.3\% compared with existing state-of-the-art models, and the performance gain is even more pronounced on the Replica dataset, which features finer scene structures. Such modernized architectural reconstruction unlocks stronger embodied representation capabilities at a lower computational cost, thereby providing a highly robust technical pathway for paradigm evolution in the field of audio-visual navigation. 

---
# STKAN: Kolmogorov-Arnold Networks for Spatio-Temporal Forecasting 

**Authors**: Sicong Lai, Yuehong Hu, Siru Zhong, Si Qiao, Yuxuan Liang, Guangyin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2607.13108)  

**Abstract**: Real-world traffic data exhibit heterogeneous spatial correlations and nonlinear temporal dynamics, posing substantial challenges for accurate spatio-temporal forecasting. Existing approaches have developed increasingly sophisticated graph, attention, and decomposition architectures, while the influence of the underlying nonlinear function approximator has received comparatively less attention. In this work, we propose STKAN, a spatio-temporal forecasting architecture that introduces Taylor-polynomial Kolmogorov--Arnold Network modules into spatial and temporal token mixing. STKAN first constructs high-level spatial representations through a learnable soft node-group assignment mechanism, applies group-wise spatial mixing, and subsequently models temporal dependencies over the compressed sequence. Spatial and temporal self-attention layers are further employed to capture long-range interactions. Experiments on five traffic forecasting benchmarks show that STKAN achieves competitive performance and performs better than the evaluated MLP-based variant in the tested settings. These results suggest that the design of nonlinear function approximators can serve as a useful complement to architectural design in spatio-temporal forecasting. 

---
# Disentangling Knowledge States with Ability and Proficiency Modeling for Knowledge Tracing 

**Authors**: Duantengchuan Li, Yingqian Bi, Jinsong Chen, Rui Zhang, Mingwen Tong  

**Link**: [PDF](https://arxiv.org/pdf/2607.13103)  

**Abstract**: Knowledge tracing (KT) aims to predict students' future performance by modeling their evolving knowledge states from historical interactions. Existing KT methods usually treat the raw interaction sequence as a unified behavioral process, overlooking the phase-specific nature of learning behaviors. Our preliminary observations show that students are more likely to correctly answer previously failed knowledge concepts after sufficient practice, suggesting a transition from ability-building to proficiency-oriented learning. Motivated by this, we propose Phase-Aware Knowledge Tracing (PAKT), a KT framework that decomposes student interactions into ability and proficiency phases based on the tailored decomposition mechanism. To effectively exploit the decomposed sequences, we design a multi-branch Transformer with a type-aware readout module to jointly capture phase-specific and holistic knowledge states. We further provide a causal analysis to reveal the confounding bias caused by entangling complex learning behaviors in phase-agnostic KT models. Extensive experiments on six public benchmarks demonstrate that our method consistently outperforms representative baselines, with a maximum AUC gain of 1.33% and an average gain of 0.82%. 

---
# TSSM: Triaxial State Space Model for Global Station Weather Forecasting with Temporal-Variable-Historical Modeling 

**Authors**: Songru Yang, Zili Liu, Tao Han, Ben Fei, Fenghua Ling, Lei Bai, Chang Liu, Xiangyang Ji, Zhenwei Shi, Zhengxia Zou  

**Link**: [PDF](https://arxiv.org/pdf/2607.13101)  

**Abstract**: Global Station Weather Forecasting (GSWF) is pivotal for localized and extreme weather prediction over key regions. Despite efforts to exploit look-back windows, existing methods show limited accuracy gains and struggle with extreme events and error accumulation. These limitations stem from overreliance on short-term patterns, which are insufficient to capture chaotic weather dynamics, especially under partial observations. To address this problem, we propose a novel Triaxial State Space Model (TSSM) with a history-enhanced Temporal-VariableHistorical paradigm, which incorporates period-aligned historical weather data to compensate for long-term, large-scale periodic, and full-window weather patterns beyond the temporal lookback window. Specifically, TSSM stacks historical samples into period-aligned batches, where forecasting is causally supported by historical and current observations. Temporal, variable, and historical scanning are designed to capture axial temporal dependencies, variable correlations, and historical evolution. This structure is hierarchically shared to model seasonal to extreme events while alleviating misalignment across historical patterns. TSSM achieves SOTA performance on Weather-5K, the largest station weather dataset to date, with 10% and 61% gains in accuracy and extreme event metrics, and obtains 95% best or second-best results on human-involved datasets. Its advantages are more pronounced in long-horizon and iterative forecasting, reaching a 37.5% gain at 240h and up to 103.5% under a 48h times 5 iterative setting. Moreover, TSSM retains > 90% performance under up to 80% missing observations, compared with < 43% for baselines, demonstrating robustness and practical potential for reliable GSWF in global in-situ observation networks. 

---
# WaterMoE: Expert-Routing-based Watermarking for High Fidelity and Efficiency 

**Authors**: Z Sun, Q Jiang, S Sheng, L Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13099)  

**Abstract**: Large language models (LLMs) have achieved remarkable success but raise growing concerns about content provenance and misuse, motivating the need for reliable watermarking techniques. However, these techniques have rarely been adopted in practice mainly for two reasons: i) severely degraded model performance, and ii) additional inference overhead. To confirm the problem, we construct a comprehensive benchmark spanning different generation tasks to systematically evaluate 9 representative watermarking methods. We found almost all existing methods are designed for text fluency, but not for restricted and complicated tasks, and their overhead prevents them from deployment in latency-critical systems.
To address i) and ii), we propose an LLM watermarking scheme \textit{WaterMoE} for the growingly popular Mixture-of-Experts (MoE) LLMs. WaterMoE embeds watermarking signals through controlled perturbation into the expert selection at each router, which accumulates to token selection shift at the final output. In contrast to watermarking as a post-processing token-sampling approach, WaterMoE embeds watermark within the inference loop incurring negligible quality degradation and computational overhead. Extensive experiments demonstrate that our method achieves a fidelity performance close to the unwatermarked and consistently outperforms state-of-the-art watermarking methods on the benchmark, with up to $4\times$ speedup, incurring merely 1\% additional inference latency compared to native generation. The results demonstrate the capability of WaterMoE to be deployed in real-world tasks. 

---
# Full-Pipeline Inference Optimization for MiMo-V2.5 Series: Pushing Hybrid SWA Efficiency to the Limit 

**Authors**: Xiaomi MiMo Team, Anqi Liu, Aoxin Ma, Bo Chen, Bo Yang, Chen Wang, Chen Zhang, Chengda Tang, Chengwei Wang, Chiheng Lou, Depeng Yan, Fuli Luo, Gang Wang, Hailin Zhang, Jiale Sun, Kang Zhou, Rui Huang, Shaohui Liu, Shen Huang, Shijie Cao, Shuaishuai Fan, Tianling Zhou, Xiangwei Deng, Xueyang Xie, Xuli Wang, Yingchun Lai, Yu Yang, Yuan Zhang, Zhen Tang, Zhonghua Deng, Zihan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13095)  

**Abstract**: We present a full-pipeline inference optimization for the MiMo-V2.5 model family, which combines Hybrid Sliding Window Attention (Hybrid SWA), sparse Mixture-of-Experts (MoE), and multimodal encoders. While Hybrid SWA can ideally reduce both attention compute and KVCache storage significantly compared to Full Attention, realizing these gains in production requires substantial engineering effort. We systematically optimize the KVCache system with layerwise prefetch, SWA-aware prefix cache trees, and specialized placement strategies, achieving strict $O(W)$ SWA storage and high cache hit rates. We further build GCache, a high-performance distributed cache infrastructure with RDMA-optimized networking, and develop a KVCache-affinity router to reduce computation while preserving load balancing. We also optimize for multimodal inputs, including GPU image preprocessing, parallel video decoding, and multimodal cache sharing. Together, these optimizations constitute the first large-scale LLM serving system in production that efficiently covers the Hybrid SWA + MoE + multimodal composite architecture. 

---
# Analyzing Curricular Pattern Complexity Using AI to Improve On-Time Graduation Rates 

**Authors**: Lynn Vonderhaar, Juan Couder, Siri Siqveland, Omar Ochoa, James Pembridge  

**Link**: [PDF](https://arxiv.org/pdf/2607.13094)  

**Abstract**: The rise of Artificial Intelligence (AI) enables automatic analysis of large amounts of data. Previously time-consuming and labor-intensive tasks can be completed much more efficiently with the use of AI. This work uses AI techniques to analyze and revise curricular patterns in an undergraduate degree for Software Engineering. Curricula often have long sequences where failure to pass a class within the sequence may jeopardize completion of the degree within four years. Manual analysis and revision of curricula by university faculty is a lengthy and labor-intensive process, causing changes to occur rarely and making it impossible to keep up with the changing needs of students. This work reduces the time-to-change for curricula and reduces bottlenecks and graduation delays by using Large Language Models (LLMs) to analyze curricular patterns and suggest revisions. 

---
# Efficient and Privacy Aware Edge Cloud Collaborative Inference for Large Language Models 

**Authors**: Yi Li, Chen Li, Jiexiong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.13093)  

**Abstract**: On-device LLM inference faces a trilemma of response latency, limited hardware resources and user privacy. Full cloud inference delivers strong computing power but exposes user prompts and dialogue data, while standalone on-device inference is unfeasible for most consumer and embedded edge devices. This paper presents a privacy-centric edge-cloud collaborative LLM inference framework built on endpoint-authenticated KV cache. Local endpoints handle input preprocessing, embedding computation, adaptive feature optimization, KV cache authentication, speculative decoding and low-dimensional model head calculation, while the cloud conducts authenticated decoder inference, KV cache management, token verification and high-dimensional vocabulary projection. Endpoints fuse partial outputs, apply language-adaptive masking and sample target tokens. All transmitted data and truncated logits are quantized and AES-GCM encrypted for privacy, with core lightweight modules, draft parameters and cache access policies kept local to avoid leakage. The framework supports heterogeneous devices including CPU-only, GPU-equipped and embedded devices via optimized streaming, batching and quantized ONNX deployment. Evaluations demonstrate that the framework reduces per-token latency by up to 46.1\% and downlink payloads by up to 67.4\% over baseline split inference, retaining comparable performance to full cloud inference. 

---
# Self-Improving AI Coding Agents Through Accumulated Behavioral Rules: A Closed-Loop Framework 

**Authors**: Aditya Aggarwal, Nahid Farhady Ghalaty  

**Link**: [PDF](https://arxiv.org/pdf/2607.13091)  

**Abstract**: LLM-based coding agents repeat the same classes of mistakes across sessions because they lack a mechanism to retain corrections from human review feedback. We present a closed-loop framework in which every accepted review comment is codified as a persistent behavioral rule, progressively expanding the set of error classes the agent can self-detect. The framework combines an accumulating rule set in a version-controlled instruction file, a self-review checklist executed before code submission, and automated validation that ensures rule set integrity as it grows. In deployment across a 35+ service microservices platform, the rule set grew from 5 to 18 behavioral rules, 15+ language-specific standards, and a 15-item self-review checklist, all derived from real review feedback. We present empirical results from 11 recorded working sessions spanning code generation, PR review, incident investigation, and cross service refactoring. We observe that accumulated rules shift review effort from low-level correctness toward design-level validation, achieve a measured 0% recurrence rate for ruled-against error classes, and transfer across heterogeneous agent interfaces. We compare our approach against related work in experiential LLM learning (Reflexion, ExpeL, Voyager) and automated code review (CodeReviewer, SWE-bench agents), showing that our framework achieves persistent cross-session learning without weight updates, operates on production codebases rather than synthetic benchmarks, and addresses an orthogonal dimension (behavioral consistency over time) that existing benchmarks do not measure. The result is a coding agent that improves with every review cycle, accumulating the engineering wisdom of its human collaborators without changing a single model weight. 

---
# Baselines Before Architecture: Evaluating Coding Agents for Autonomous Penetration Testing 

**Authors**: Ananda Dhakal, Krish Neupane, Aarjan Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2607.13085)  

**Abstract**: Recent autonomous penetration testing papers report high benchmark scores while adding multi-component security harnesses around frontier LLMs. Because these systems often change both architecture and backbone model, it is difficult to tell how much performance comes from the harness rather than from the underlying model.
This paper presents a controlled study on the 104-task XBOW benchmark using default coding CLI agents as plain-agent baselines. We first run Codex, OpenCode, and Pi with the same GPT-5 model, budget, target interface, and scoring rule. This phase identifies the strongest same-model baseline and tests whether security-specific prompt variants improve its observed score. We then compare the default Codex scaffold with published MAPTA and PentestGPT V2 results under the closest available model matches. Finally, we repeat the plain-agent experiment with GPT-5.2 and GPT-5.5 to measure model scaling inside the same scaffold.
The results show a mixed but practical picture. Specialised harnesses can add measurable benchmark lift and may improve cost efficiency, but plain coding agents already solve a large share of the benchmark; repeated plain-agent runs can match or exceed some published architecture scores in union coverage, and newer models substantially improve the same scaffold. Future evaluations should report model-matched plain-agent baselines before attributing benchmark gains to architecture design alone. 

---
# SingGuard-NSFA: Extensible Guardrails for Agentic AI via Generative Reasoning and Real-Time Classification 

**Authors**: SingGuard Team  

**Link**: [PDF](https://arxiv.org/pdf/2607.13081)  

**Abstract**: We present nsfaguard, a guardrail framework for securing agentic AI systems against operational threats, such as prompt injection, sensitive information extraction, malicious code requests, dangerous tool misuse, and resource exhaustion. We first introduce the NSFA taxonomy, which organizes 185 risk variants into a CIA-triad-grounded hierarchy and is cross-validated against three well-established OWASP guidelines. Based on this taxonomy, we construct a benchmark suite spanning 133 languages, comprising over 93K purpose-built samples targeting both user queries and agent responses, along with 3,435 cross-source samples adapted from five public agent-security datasets. To detect these operational threats in practice, we develop a dual-mode approach combining SFT-based generative reasoning for interpretable offline auditing with discriminative classification heads on the frozen backbone, enabling real-time detection at approximately 50,ms. We release four models with 0.8B, 2B, 4B, and 9B parameters, all achieving $\geq$94% F1 on purpose-built benchmarks and surpassing the strongest competing guardrails by 6 to 12 absolute points. On cross-source evaluation, the 9B model attains 91.29% F1 with a more balanced precision--recall trade-off. Moreover, ablation experiments show that classification heads can equip a guardrail with risk detection capabilities beyond its original scope and achieve state-of-the-art performance. These results demonstrate the extensibility of the approach and its generality as a plug-in enhancement. 

---
# Inference Economics of Enterprise Coding Agents: A Case Study of Cloud vs. On-Premise LLMs 

**Authors**: Sheng-Wei Peng, Yi-Hsun Lin, Yi-Pei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.13080)  

**Abstract**: Autonomous coding agents force engineering organizations to choose between API-based frontier models -- strong reasoning at high token cost -- and on-premise quantized open-weights models, which promise low-marginal-cost scaling and data sovereignty at some loss of reasoning fidelity. We study this trade-off through a single-developer, non-randomized longitudinal case study over two contiguous 28-day periods on a production monorepo: an API-based Claude Opus 4.7/4.8 configuration using Claude Code versus an on-premise GLM-5.1/5.2 configuration using Opencode, quantized to NVFP4, on NVIDIA Blackwell hardware. Analyzing LLM telemetry and Git history, we find that prompt caching (99.3% hit rate) cuts realized API cost by 88.6% to an effective \$0.57 per million tokens -- below even the \$2.83 amortized unit cost of the shared on-premise slice (a utilization-dependent inversion; total realized spend and total cost of ownership (TCO) are the robust quantities). At comparable gross code churn, the local configuration was associated with a far higher defect-repair burden: a Fix Commit Ratio (FCR) of 74.9% versus 45.9%, with the odds of a commit being a repair 2.6 to 4.9 times higher within every difficulty tier (Mantel-Haenszel OR = 3.61). Under Taiwan-market parameters and a symmetric labor model, on-premise deployment nonetheless saves 40.1% of true TCO under shared GPU allocation, whereas dedicated reservation costs 43.8% more than the cached API. Under shared allocation, the genuine penalty is not monetary but a measurable developer-experience burden -- timestamp indicators show more work trapped in debugging spirals and a slower commit cadence -- and an offline replay shows hybrid routing gateways trade defect rate for infrastructure savings along a cost-quality frontier rather than dominate the pure-API baseline. 

---
# Operational Evidence Gaps for LLMs in Fraud Detection and Trust-and-Safety Workflows 

**Authors**: Keyur Gabani  

**Link**: [PDF](https://arxiv.org/pdf/2607.13078)  

**Abstract**: LLMs are now proposed for fraud detection, scam investigation, content moderation, and other trust-and-safety workflows. Much of the public literature still evaluates them as models, with less attention to their behavior as components in operational pipelines. This creates a practical evidence question: what would justify placing an LLM inside a live workflow with latency, cost, escalation, human-review, and adversarial-risk constraints?
We address this question through a fraud-first survey of deployment evidence. We code 49 operationally relevant sources on LLM use in fraud detection, investigation support, content moderation, and cross-cutting robustness (18 fraud, 14 moderation, 17 cross-cutting), supplemented by 15 contextual references that establish the survey boundaries. These sources include systems, benchmarks, frameworks, and deployment-relevant surveys, not 49 production deployments.
The main finding is an evidence imbalance. Fraud supplies the largest task-specific portion of the coded corpus. The moderation papers, however, include more explicit public evidence on latency, cost, governance, and fairness. Among the 18 fraud and investigation sources, none report clean per-decision latency, per-decision dollar cost, or calibration evidence; most report offline task performance, retrieval gains, or case-study accuracy instead.
The survey contributes a role-and-evidence organizing frame, FORTE, for locating LLMs as classifiers, retrieval interfaces, explanation generators, reviewer assistants, agents, feature extractors, or escalation components. It also contributes a minimum deployment-evidence checklist covering latency budget, cost per decision, decision threshold, explanation integrity, and adversarial pressure. The resulting agenda identifies studies needed to support deployment claims for LLM-based fraud and trust-and-safety work. 

---
# The Hitchhiker's Guide to Monoculture 

**Authors**: Gordon Burtch  

**Link**: [PDF](https://arxiv.org/pdf/2607.13077)  

**Abstract**: Large language models (LLMs) often produce homogeneous outputs, raising concerns that AI coding assistants may lead to convergence in the software artifacts that developers create. Whether this occurs in practice is unclear because developers interactively prompt, evaluate, modify, and reject model outputs, and because outputs vary with prompt and repository context. I examine code homogenization using Kaggle contest submissions from 2019 to mid-2026. I first document widespread convergence toward the random seed value 42, consistent with LLMs reinforcing a longstanding convention in programming culture. I then study homogenization more broadly, at two levels of aggregation and abstraction. At the submission level, I measure the average pairwise similarity of submissions within contests. At the contest level, I measure the conceptual span of submitted code, motivating distinct measures for each: TF-IDF representations, which capture surface syntax, and Voyage 3 code embeddings, which capture code intent and semantics. The results demonstrate substantial syntactic homogenization at both the individual and collective levels: individual submissions have become more alike in literal syntax and code structure, while the latent dimensionality of syntactic variation has narrowed. In contrast, I find little evidence of semantic homogenization, individually and collectively. Average semantic distance remains essentially flat, and the contest-level latent dimensional span of semantic approaches remains stable, with evidence suggesting it has even expanded modestly. These findings suggest that AI coding assistants are certainly standardizing implementation details, yet they have not yet produced evidence of homogenization in the approaches and problem-solving strategies coders employ. 

---
# The Entanglement Wall: Activation-Space Probes as Risk Detectors, Not Context Adjudicators 

**Authors**: Dominik Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2607.13075)  

**Abstract**: Context can change whether a request is harmful without changing its topic or surface form. We ask whether residual-stream probes distinguish harmful requests from surface-matched benign controls at a useful operating point. Across three 7-8B model families, an activation sensor blocks 95.5-97.7 percent of judge-classified compliant attacks in a taxonomy-selected set. It also blocks 59.6-68.4 percent of XSTest prompts. A fully disjoint audit reconstructs near-ceiling source-contrast AUROC (0.996-0.999), but fixed transfer to matched pairs is weaker: 0.656-0.819 on the guard-selected Twin-n70 subset and 0.590-0.690 on the full Twin-n163 cohort. We test ten axes on the reference family and seven across all families with leakage, hold-out, and permutation controls. On Twin-n163, no axis evaluated without direct pair-boundary fitting reaches the specified numerical threshold. Requiring persistence on that full cohort was added at analysis time. A separately specified 24B/32B extension gives the same result. Pair-trained classifiers weaken under category and generation-batch hold-out and false-block 79.6-100 percent of XSTest at 95 percent in-corpus TPR. At the tested read points, these activation scores behave as broad-risk detectors rather than standalone context adjudicators. 

---
# When is the combined load identifiable from a stress-intensity profile? A coupled forward-inverse study on SIFBench finite-element data 

**Authors**: Giansalvo Cirrincione, Filippo Grassia  

**Link**: [PDF](https://arxiv.org/pdf/2607.13074)  

**Abstract**: This work studies the inverse problem of recovering the relative magnitudes of the tension, bending, and bearing loads acting on a crack from its stress-intensity-factor profile along the crack front, using the public SIFBench finite-element data. The central claim is not forensic load recovery on field cases, but a rigorous characterization of when the combined load is identifiable at all, together with an estimator that returns calibrated uncertainty precisely in the regimes where it is not. For a known geometry the forward map from loads to profile is exactly linear, and identifiability reduces to a single geometric question: whether the three elementary load profiles are linearly independent as functions along the front. When they are nearly dependent, many different load combinations produce almost the same profile and the inverse problem is illposed; the analysis shows that the degree of ill-posedness is controlled by an intrinsic stability margin, not by the conditioning number alone. A single crack-front operator serves both as a structured forward surrogate and as the differentiable map required by a simplex-constrained, set-valued inverse estimator. On the SIFBench corner-crack scenario the empirical behaviour matches the theory: the typical geometry is well posed while a sizable minority is genuinely ill-posed, so a point estimate is reliable on the majority and provably uninformative on the rest. Validation is on controlled synthetic noise; no real fracture cases are used or claimed. 

---
# HRO: Hierarchical Room-to-Object Framework for Zero-Shot Object Goal Navigation with Large Language Models 

**Authors**: Luyuan Jia, Yinfeng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.13072)  

**Abstract**: Zero-shot object-goal navigation aims to enable an intelligent agent to explore and navigate to objects of unknown categories in an unfamiliar environment without specific target training. In zero-shot navigation tasks, pre-trained large models are usually employed to leverage their prior knowledge for guiding the agent's navigation. However, existing zero-shot object-goal navigation methods based on large language models (LLMs) merely utilize LLMs as flat reasoning tools to directly associate objects or regions. They lack the hierarchical spatial cognition modeling of human-like room semantics to object localization, which leads to strong blindness in exploration, insufficient accuracy in semantic association, and failure to fully unleash the common-sense reasoning potential of LLMs. This paper proposes an LLM-driven hierarchical room-to-object (HRO) framework for zero-shot object-goal navigation, which guides the agent to explore and navigate to the target object in a coarse-to-fine manner. Experiments on Gibson and HM3D datasets verify that our HRO framework achieves superior success rate and generalization over existing LLM-based methods, underscoring LLMs' strong potential for zero-shot object-goal navigation. 

---
# Compaction as Epistemic Failure: How Agentic LLM Tools Fabricate Confirmed Results from Killed Processes 

**Authors**: Hiroki Tamba  

**Link**: [PDF](https://arxiv.org/pdf/2607.13071)  

**Abstract**: Agentic LLM coding tools compress long session histories into compaction summaries that subsequent sessions inherit as ground truth. This paper documents a failure mode in Claude Code where partial standard output from timed-out commands (exit code 143) is recorded in compaction summaries as confirmed results, propagating false positives across sessions and model versions without re-verification. The underlying mechanism is a conflation of observation and persistence, where information that appeared in the terminal is treated as equivalent to information written to durable storage. This finding extends the analysis of LLM self-evaluation failures reported in prior work on non-determinism in LLM-as-judge grading by showing that agentic tools exhibit analogous reliability deficits when reporting on their own operational outcomes. The failure has direct implications for any workflow that relies on agentic session continuity for data processing, scientific computation, or multi-step automation. 

---
# Autonomous UAV Route Planning for Coverage Maximization in Environmental Monitoring: A Systematic Literature Review 

**Authors**: Sebastian Jouannet-Contreras, Carola Figueroa-Flores  

**Link**: [PDF](https://arxiv.org/pdf/2607.13054)  

**Abstract**: Environmental monitoring with unmanned aerial vehicles (UAVs) requires route planning methods that maximize covered area while handling energy limits, operational constraints, and geometric complexity. This paper reports the protocol and preliminary results of an ongoing systematic literature review (SLR) on autonomous UAV route planning for coverage-oriented environmental monitoring. The review follows the PRISMA 2020 framework and searches Scopus and Web of Science for studies published between 2015 and 2026. The protocol focuses on path planning, coverage path planning, and informative path planning, with emphasis on algorithmic families, coverage and energy metrics, obstacle handling, geometric environment representations, and environmental constraints. At the current stage, 562 records have been identified, 161 duplicates have been removed, and 401 unique records have been screened by title, abstract, and keywords. From these, 247 studies were retained for full-text eligibility assessment (235 eligible and 12 borderline records to be resolved during full-text review). A preliminary analysis of the retained studies suggests strong concentration on coverage-oriented formulations, multi-UAV coordination, and energy-aware optimization, while fewer studies explicitly address weather, uncertainty, or obstacle-rich environments. Most retained studies rely on simulation-based validation, highlighting a potential simulation-to-reality gap, and recent publications show increasing interest in reinforcement learning, hybrid optimization, and geometry-aware planning. These early findings indicate an active but fragmented research landscape and support the need for a structured synthesis to identify mature techniques and unresolved gaps for realistic environmental monitoring missions. 

---
# Uncertainty-Aware Sequential Decision Rules for Event-Triggered LLM Invocation in Streaming Systems 

**Authors**: Zhaohui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13048)  

**Abstract**: Streaming inference pipelines increasingly pair lightweight fast models with Large Language Models (LLMs) that provide rich semantic understanding at substantial cost. The central question of when to invoke the LLM has received limited formal treatment. We cast this as a risk-based sequential stopping problem, where a trigger policy fires when a risk functional over the observation history exceeds a threshold. Within this framework, we prove six results: a minimum inter-event time bound excluding trigger chattering; optimality of threshold policies via smooth pasting; approximate SPRT guarantees under estimated parameters; O(sqrt(T log T)) regret for stationary streams, extending to O(sqrt((C_T + 1) T log T)) under C_T changepoints; O(1/sqrt(T)) convergence of online gradient descent for adaptive thresholds; and a calibration-to-miss-rate transfer inequality. Several classical trigger families, including event-triggered, optimal stopping, SPRT, CUSUM, and Bayesian triggers, can be expressed as special cases of this framework. On turbofan degradation data (CMAPSS) with real LLM calls, we empirically verify the theoretical assumptions, ablate the risk function design, compare against six baselines including a RouteLLM-style router and contextual bandits, and analyze cost sensitivity and LLM failure modes. The results confirm sublinear regret, with alpha < 1 for all principled triggers; high diagnostic quality, with 92.9 percent of 1600 LLM diagnoses reaching grounding score >= 0.75 under our rubric; and that anomaly-score-driven risk functions dominate alternatives by roughly an order of magnitude on the Pareto AUC. 

---
# Federated Explainable Artificial Intelligence: Roles, Architectures, Evaluation, and Open Challenges 

**Authors**: Masoume Gholizade, Fabrizio Ruffini, Pietro Ducange, Francesco Marcelloni  

**Link**: [PDF](https://arxiv.org/pdf/2607.13045)  

**Abstract**: Federated Learning (FL) has emerged as a key paradigm for privacy-preserving collaborative model training across distributed and heterogeneous data sources. By keeping raw data local, FL addresses data confidentiality concerns, yet it does not resolve the opacity of modern machine learning models. In parallel, Explainable Artificial Intelligence (XAI) has gained attention for improving transparency, trust, and accountability, particularly in high-stakes domains. Their intersection has given rise to Federated Explainable Artificial Intelligence (FedXAI) paradigm, which aims to jointly satisfy privacy and explainability requirements. This survey provides a systematic review of FedXAI, highlighting the transition of explainability from a post-hoc tool to an integral component of the FL lifecycle. We show how explainability supports aggregation, personalization, robustness, coordination, and system-level decision making. To organize the literature, we introduce a taxonomy that classifies FedXAI methods by the role of explainability, model and explainer types, explanation scope, integration level, FL settings, and data heterogeneity. We review approaches ranging from model-agnostic explanations to interpretable federated models and explainability-aware aggregation mechanisms. We also examine evaluation practices and discuss the lack of standardized benchmarks and metrics for measuring explanation quality, stability, privacy leakage, and computational overhead. Finally, we identify key challenges, including explainability under non-IID data, explanation-centric security threats, communication-efficient XAI, continual FedXAI, and the integration of domain knowledge and regulatory constraints. By consolidating existing work and identifying key gaps, this survey serves as a reference framework for designing trustworthy, transparent, and privacy-preserving federated AI systems. 

---
# The Perplexity Trap: When Patent Law Makes Human Writing Look Like AI 

**Authors**: Anubhab Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2607.13044)  

**Abstract**: The European Patent Office (EPO) reported record filings in 2025, and the 2026 EPO Guidelines hold applicants strictly responsible for LLM-assisted content under Article 83 and Rule 42, creating pressure to triage suspected AI-generated patent text. Two constraints make this hard. First, realistic prosecution settings often have only consumer GPUs with about 8 GB VRAM, not datacenter-class scoring stacks. Second, Article 84 of the European Patent Convention requires claims to be clear and concise, pushing human drafting onto the same low-perplexity, low-burstiness manifold that LLMs occupy. We benchmark three open-source zero-shot detectors on 500 granted EPO H04 telecom patents versus 500 LLM-generated counterparts using five prompting strategies, all under the consumer hardware envelope. At claim level, all detectors exceed 60 percent false-positive rate: Binoculars 78.3 percent, Fast-DetectGPT 61.3 percent, DetectGPT 80.5 percent. The failure persists under Qwen2.5-3B-Instruct regeneration, LoRA-adapted Pythia-2.8B scoring heads, cross-IPC replication on A61K, C07D, and F03D (mean FPR 84.6 percent), and H100 re-evaluation with published Falcon-7B and GPT-J-6B heads, arguing the issue is structural rather than substitute-model capacity. A seven-feature linguistic-complexity logistic regression reaches 74.0 percent accuracy at 28.1 percent FPR, a 13 percentage-point gain over a perplexity-only baseline at a comparable operating point, without using likelihood at inference and within the same hardware budget. 

---
# Beyond Backbone Backpropagation: A Decoupled Strategy for Efficient Transfer Learning 

**Authors**: Daniel Vila-Cruz, Laura Morán-Fernández, Verónica Bolón-Canedo  

**Link**: [PDF](https://arxiv.org/pdf/2607.13043)  

**Abstract**: Deep learning models achieve state-of-the-art image classification but face deployment challenges due to computational costs and energy demands. We propose a lightweight training strategy that adapts normalization layers of the model to the new domain and decouples feature extraction from classifier optimization, reducing overhead by precomputing features only once. A redesigned classifier head with margin-based weighted loss further minimizes ambiguity without end-to-end backpropagation. Evaluated across four CNN architectures (ResNet18, ResNet50, MobileNet, DenseNet121), three Transformer models (ViT, Swin and DeiT) and three medical datasets (Brain Cancer MRI, BreakHis and PatchCamelyon), our approach significantly reduces the required training time with only a marginal accuracy trade-off, often matching or surpassing baseline performance. This efficiency translates to reducing CO2 by orders of magnitude, offering a practical and environmentally sustainable solution for resource-constrained clinical or prototyping environments. 

---
# LessonBench-V1: A Benchmark Dataset for Evaluating AI Lesson Generation Agents 

**Authors**: Ravidu Suien Rammuni Silva, Ahmad Lotfi, Isibor Kennedy Ihianle, Golnaz Shahtahmassebi, Jordan J. Bird  

**Link**: [PDF](https://arxiv.org/pdf/2607.13041)  

**Abstract**: Large Language Model (LLM) based AI educational content generation systems are increasingly being developed, yet no standardised benchmark exists to systematically evaluate them. This study introduces LessonBench-V1, a benchmark dataset comprising 647 human-written lessons paired with LLM-based reverse-engineered lesson plans across 240 STEM topics spanning mathematics, physics, chemistry, and computer science. The lessons are drawn from 97 trusted open sources, including LibreTexts, this http URL and GeeksForGeeks. Each lesson plan is human-reviewed and produced through a pedagogically grounded methodology that synthesises Bloom's Taxonomy, Gagné's Events, Merrill's First Principles, and the 5E Instructional Model. The lesson plans capture 3,620 learning objectives with pedagogical metadata, enabling systematic, reproducible evaluation of lesson-generation AI agents and supporting further research. The study further proposes a three-dimensional evaluation pipeline for use with the dataset. 

---
# Final Authority in AI Governance: Frontier-Provider Sovereignty and Action-Centered Deployer Governance 

**Authors**: Zexun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.13040)  

**Abstract**: This paper examines where final authority should sit once capable AI systems are embedded in organizational workflows. It compares two governance models. The first, frontier-provider sovereignty, assigns privileged authority to the provider of the most capable models and is reflected in contemporary arguments for frontier-model testing, release gating, transparency duties, and compute-related controls. The second, action-centered deployer sovereignty, places final authority over high-impact actions with the organization that authorizes the action, embeds it in a business process, and bears the downstream legal, operational, and commercial consequences. The paper combines comparative reading of public governance frameworks with implementation-informed analysis of runtime heterogeneity and enterprise control requirements. It compares EU AI Act guidance, the NIST AI Risk Management Framework, Singapore's Model AI Governance Framework for Agentic AI, recent Japanese AI policy instruments, and Canada's voluntary code and managerial guidance. Across these materials, the paper finds stronger support for distributed operational accountability than for unilateral frontier-provider control. It further argues that rapid enterprise adoption, declining provider transparency, and widening control gaps increase the value of a portable governance layer centered on governed action rather than on provider-native session objects. The conclusion is layered rather than absolutist: strong upstream authority remains justified for frontier capability gating, but final authority over concrete enterprise action is better located with the deployer and consequence-bearer. 

---
# Safeguard-Conditioned Uplift: Measuring Utility-Risk Frontiers for Dual-Use Biology Assistants 

**Authors**: Dipesh Tharu Mahato  

**Link**: [PDF](https://arxiv.org/pdf/2607.13039)  

**Abstract**: Safety evaluations for dual-use biology assistants often measure base-model capability, refusal behavior, or jailbreak success. These metrics miss a deployment question: for a fixed base model, how does the access condition users actually see change benign utility and harmful actionable assistance? I introduce safeguard-conditioned uplift, a protocol for comparing deployed access conditions through a human-judged utility-risk frontier. I evaluate Claude Sonnet 4.6 and Gemini 3.5 Flash under helpful prompting, safety prompting, and an external safeguarded assistant on a 108-task surrogate benchmark, with the headline claim restricted to a locked 18-task held-out split. In a 600-row blinded human audit, the safeguarded assistant reduces harmful actionability relative to helpful prompting by -0.063 over 49 matched response pairs, with bootstrap 95% interval [-0.117, -0.011], while correctness changes by +0.009 with interval [-0.057, +0.077]. Adaptive, Test-B, cue-ablation, and controller-baseline checks support the measurement story but also show non-dominance: safety prompting is often strongest for Claude, while external control helps more for Gemini and can reduce benign utility. The contribution is not a universal defense. It is a deployment-level evaluation target, plus a learned risk-budgeted calibration procedure, for measuring how user-facing access conditions move the utility-risk frontier. 

---
# Designing Safety-Constrained LLM Systems for Public Health Information Access 

**Authors**: Ben Torkian, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.13038)  

**Abstract**: We present the design and implementation of a safety constrained large language model (LLM) system for public health information access, focusing on maternal and child health (MCH) resource navigation. While LLM based systems offer flexible and natural interfaces for information retrieval, their deployment in healthcare contexts introduces risks related to safety, trust, and uncontrolled generation. This work explores practical design patterns for constraining LLM behavior in safety critical environments. We introduce a multi-layered architecture that integrates domain-restricted retrieval augmented generation (RAG), strict boundary enforcement to prevent medical advice, anonymous multiuser session management, and comprehensive audit logging for monitoring and compliance. A key aspect of the design is a controlled data pipeline that grounds all responses in curated public health resources, avoiding reliance on the model pretrained medical knowledge. We implement the system in a real world public health setting and conduct scenario-based validation across in scope, out of scope, and emergency queries. Results show consistent enforcement of safety constraints, reliable resource grounding, and stable system performance, with an average response time of 5.3 seconds. Beyond the specific application, we discuss design trade offs and lessons learned in balancing safety, usability, and system flexibility. Our findings provide practical guidance for deploying LLM based systems in healthcare and other domains where strict information boundaries and accountability are required. 

---
# Ask Before You Diagnose: Safe-Psych, a Sequential Evaluation Benchmark for LLMs in Psychiatry 

**Authors**: Oriana Presacan, Andreea Grama, Larisa Irimină, Alireza Nik, Jaya Ojha, Vajira Thambawita, Ciprian I. Băcilă, Bogdan Ionescu, Michael A. Riegler  

**Link**: [PDF](https://arxiv.org/pdf/2607.13036)  

**Abstract**: Large language models (LLMs) are increasingly used for decision support in healthcare, but clinical evidence is often incomplete or evolving. When the available information is insufficient to support a reliable answer, models should request clarification or abstain rather than provide unsupported responses. Existing medical benchmarks, however, typically assume that complete information is available upfront. We introduce Safe-Psych, a sequential benchmark for evaluating how LLMs handle evolving diagnostic uncertainty in clinical psychiatry. Safe-Psych contains over 1,000 real-world psychiatric clinical notes segmented to simulate incremental evidence disclosure, with psychiatrist-derived action labels at each stage: DIAGNOSE, CLARIFY, or ABSTAIN. We evaluate multiple state-of-the-art LLMs in full-information and sequential settings. Our findings show that capability does not ensure calibration: even strong models struggle under incomplete clinical information, with under-abstention exceeding 60% for most models and safety-aware prompting reducing premature commitment only by shifting errors toward excessive abstention. In sequential evaluation, models frequently diagnose before sufficient evidence is available and rarely seek clarification unless explicitly prompted; these premature diagnoses are less accurate than on-time diagnoses. Overall, Safe-Psych reveals a limitation across the evaluated models: recognizing when clinical evidence is incomplete and additional information is needed. We release Safe-Psych to support research on improving LLM safety in healthcare. 

---
# FixItFlow: Automated Troubleshooting Guide Generation from Cloud Incidents 

**Authors**: Srihari Unnikrishnan, Jaskaran Singh Walia, Drishti Goel, Supriyo Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2607.13035)  

**Abstract**: Cloud services experience frequent incidents that require rapid diagnosis and resolution. Troubleshooting guides help engineers respond consistently, but creating them manually is labor-intensive, resulting in incomplete coverage and outdated documentation. We present FixItFlow, an automated system that generates troubleshooting guides from historical incident data using large language models. The system extracts diagnostic patterns from engineer actions, synthesizes structured guides with verified commands, and enforces strict validation to prevent fabricated content. In our evaluation with 26 engineers, generated guides achieved 61.5\% positive ratings for clarity and demonstrated a 2.3x reduction in mitigation time for incidents with associated guides. These results indicate that automated guide generation can improve incident response while reducing documentation burden on engineering teams. 

---
