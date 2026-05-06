# OpenSeeker-v2: Pushing the Limits of Search Agents with Informative and High-Difficulty Trajectories 

**Authors**: Yuwen Du, Rui Ye, Shuo Tang, Keduan Huang, Xinyu Zhu, Yuzhu Cai, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.04036)  

**Abstract**: Deep search capabilities have become an indispensable competency for frontier Large Language Model (LLM) agents, yet their development remains dominated by industrial giants. The typical industry recipe involves a highly resource-intensive pipeline spanning pre-training, continual pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning (RL). In this report, we show that when fueled with informative and high-difficulty trajectories, a simple SFT approach could be surprisingly powerful for training frontier search agents. By introducing three simple data synthesis modifications: scaling knowledge graph size for richer exploration, expanding the tool set size for broader functionality, and strict low-step filtering, we establish a stronger baseline. Trained on merely 10.6k data points, our OpenSeeker-v2 achieves state-of-the-art performance across 4 benchmarks (30B-sized agents with ReAct paradigm): 46.0% on BrowseComp, 58.1% on BrowseComp-ZH, 34.6% on Humanity's Last Exam, and 78.0% on xbench, surpassing even Tongyi DeepResearch trained with heavy CPT+SFT+RL pipeline, which achieves 43.4%, 46.7%, 32.9%, and 75.0%, respectively. Notably, OpenSeeker-v2 represents the first state-of-the-art search agent within its model scale and paradigm to be developed by a purely academic team using only SFT. We are excited to open-source the OpenSeeker-v2 model weights and share our simple yet effective findings to make frontier search agent research more accessible to the community. 

---
# Redefining AI Red Teaming in the Agentic Era: From Weeks to Hours 

**Authors**: Raja Sekhar Rao Dheekonda, Will Pearce, Nick Landers  

**Link**: [PDF](https://arxiv.org/pdf/2605.04019)  

**Abstract**: AI systems are entering critical domains like healthcare, finance, and defense, yet remain vulnerable to adversarial attacks. While AI red teaming is a primary defense, current approaches force operators into manual, library-specific workflows. Operators spend weeks hand-crafting workflows - assembling attacks, transforms, and scorers. When results fall short, workflows must be rebuilt. As a result, operators spend more time constructing workflows than probing targets for security and safety vulnerabilities.
We introduce an AI red teaming agent built on the open-source Dreadnode SDK. The agent creates workflows grounded in 45+ adversarial attacks, 450+ transforms, and 130+ scorers. Operators can probe multi-agent systems, multilingual, and multimodal targets, focusing on what to probe rather than how to implement it.
We make three contributions: 1. Agentic interface. Operators describe goals in natural language via the Dreadnode TUI (Terminal User Interface). The agent handles attack selection, transform composition, execution, and reporting, letting operators focus on red teaming. Weeks compress to hours. 2. Unified framework. A single framework for probing traditional ML models (adversarial examples) and generative AI systems (jailbreaks), removing the need for separate libraries. 3. Llama Scout case study. We red team Meta Llama Scout and achieve an 85% attack success rate with severity up to 1.0, using zero human-developed code 

---
# SymptomAI: Towards a Conversational AI Agent for Everyday Symptom Assessment 

**Authors**: Joseph Breda, Fadi Yousif, Beszel Hawkins, Marinela Cotoi, Miao Liu, Ray Luo, Po-Hsuan Cameron Chen, Mike Schaekermann, Samuel Schmidgall, Xin Liu, Girish Narayanswamy, Samuel Solomon, Maxwell A. Xu, Xiaoran Fan, Longfei Shangguan, Anran Wang, Bhavna Daryani, Buddy Herkenham, Cara Tan, Mark Malhotra, Shwetak Patel, John B. Hernandez, Quang Duong, Yun Liu, Zach Wasson, Dimitrios Antos, Bob Lou, Matthew Thompson, Jonathan Richina, Anupam Pathak, Nichole Young-Lin, Jake Sunshine, Daniel McDuff  

**Link**: [PDF](https://arxiv.org/pdf/2605.04012)  

**Abstract**: Language models excel at diagnostic assessments on currated medical case-studies and vignettes, performing on par with, or better than, clinical professionals. However, existing studies focus on complex scenarios with rich context making it difficult to draw conclusions about how these systems perform for patients reporting symptoms in everyday life. We deployed SymptomAI, a set of conversational AI agents for end-to-end patient interviewing and differential diagnosis (DDx), via the Fitbit app in a study that randomized participants (N=13,917) to interact with five AI agents. This corpus captures diverse communication and a realistic distribution of illnesses from a real world population. A subset of 1,228 participants reported a clinician-provided diagnosis, and 517 of these were further evaluated by a panel of clinicians during over 250 hours of annotation. SymptomAI DDx were significantly more accurate (OR = 2.47, p < 0.001) than those from independent clinicians given the same dialogue in a blinded randomized comparison. Moreover, agentic strategies which conduct a dedicated symptom interview that elicit additional symptom information before providing a diagnosis, perform substantially better than baseline, user-guided conversations (p < 0.001). An auxiliary analysis on 1,509 conversations from a general US population panel validated that these results generalize beyond wearable device users. We used SymptomAI diagnoses as labels for all 13,917 participants to analyze over 500,000 days of wearable metrics across nearly 400 unique conditions. We identified strong associations between acute infections and physiological shifts (e.g., OR > 7 for influenza). While limited by self-reported ground truth, these results demonstrate the benefits of a dedicated and complete symptom interview compared to a user-guided symptom discussion, which is the default of most consumer LLMs. 

---
# An Agent-Oriented Pluggable Experience-RAG Skill for Experience-Driven Retrieval Strategy Orchestration 

**Authors**: Dutao Zhang, Tian Liao  

**Link**: [PDF](https://arxiv.org/pdf/2605.03989)  

**Abstract**: Retrieval-augmented generation systems often assume that one fixed retrieval pipeline is sufficient across heterogeneous tasks, yet factoid question answering, multi-hop reasoning, and scientific verification exhibit different retrieval preferences. We present Experience-RAG Skill, an agent-oriented pluggable retrieval orchestration layer positioned between the agent and the retriever pool. The proposed skill analyzes the current scene, consults an experience memory, selects an appropriate retrieval strategy, and returns structured evidence to the agent. Under a fixed candidate pool, Experience-RAG Skill achieves an overall nDCG@10 of 0.8924 on BeIR/nq, BeIR/hotpotqa, and BeIR/scifact, outperforming fixed single-retriever baselines and remaining competitive with Adaptive-RAG-style routing. The results suggest that retrieval strategy selection can be productively encapsulated as a reusable agent skill rather than being hard-coded in the upper workflow. 

---
# From Intent to Execution: Composing Agentic Workflows with Agent Recommendation 

**Authors**: Kishan Athrey, Ramin Pishehvar, Brian Riordan, Mahesh Viswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2605.03986)  

**Abstract**: Multi-Agent Systems (MAS) built using AI agents fulfill a variety of user intents that may be used to design and build a family of related applications. However, the creation of such MAS currently involves manual composition of the plan, manual selection of appropriate agents, and manual creation of execution graphs. This paper introduces a framework for the automated creation of multi-agent systems which replaces multiple manual steps with an automated framework. The proposed framework consists of software modules and a workflow to orchestrate the requisite task- specific application. The modules include: an LLM-derived planner, a set of tasks described in natural language, a dynamic call graph, an orchestrator for map agents to tasks, and an agent recommender that finds the most suitable agent(s) from local and global agent registries. The agent recommender uses a two-stage information retrieval (IR) system comprising a fast retriever and an LLM-based re-ranker. We implemented a series of experiments exploring the choice of embedders, re- rankers, agent description enrichment, and supervising critique agent. We benchmarked this system end-to-end, evaluating the combination of planning, agent selection, and task completion, with our proposed approach. Our experimental results show that our approach outperforms the state-of-the- art in terms of the recall rate and is more robust and scalable compared to previous approaches. The critique agent holistically reevaluates both agent and tool recommendations against the overall plan. We show that the inclusion of the critique agent further enhances the recall score, proving that the comprehensive review and revision of task-based agent selection is an essential step in building end-to-end multi-agent systems. 

---
# Contextual Multi-Objective Optimization: Rethinking Objectives in Frontier AI Systems 

**Authors**: Jie Zhou, Qin Chen, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2605.03900)  

**Abstract**: Frontier AI systems perform best in settings with clear, stable, and verifiable objectives, such as code generation, mathematical reasoning, games, and unit-test-driven tasks. They remain less reliable in open-ended settings, including scientific assistance, long-horizon agents, high-stakes advice, personalization, and tool use, where the relevant objective is ambiguous, context-dependent, delayed, or only partially observable. We argue that many such failures are not merely failures of scale or capability, but failures of objective selection: the system optimizes a locally visible signal while missing which objectives should govern the interaction. We formulate this problem as \emph{contextual multi-objective optimization}. In this setting, systems must consider multiple, context-dependent objectives, such as helpfulness, truthfulness, safety, privacy, calibration, non-manipulation, user preference, reversibility, and stakeholder impact, while determining which objectives are active, which are soft preferences, and which must function as hard or quasi-hard constraints. These examples are not intended as an exhaustive taxonomy: different domains and deployment settings may activate different objective dimensions and different conflict-resolution procedures. Our framework models AI behavior as a context-dependent choice rule over candidate actions, objective estimates, active constraints, stakeholders, uncertainty, and conflict-resolution procedures. We outline an implementation pathway based on decomposed objective representations, context-to-objective routing, hierarchical constraints, deliberative policy reasoning, controlled personalization, tool-use control, diagnostic evaluation, auditing, and post-deployment revision. 

---
# QKVShare: Quantized KV-Cache Handoff for Multi-Agent On-Device LLMs 

**Authors**: Pratik Honavar, Tejpratap GVSL  

**Link**: [PDF](https://arxiv.org/pdf/2605.03884)  

**Abstract**: Multi-agent LLM systems on edge devices need to hand off latent context efficiently, but the practical choices today are expensive re-prefill or full-precision KV transfer. We study QKVShare, a framework for quantized KV-cache handoff between agents that combines token-level mixed-precision allocation, a self-contained CacheCard representation, and a HuggingFace-compatible cache injection path. Our current results support a narrower but clearer story than the original draft: on 150 GSM8K problems with Llama-3.1-8B-Instruct, adaptive quantization remains competitive under repeated handoff and shows its clearest gains against uniform quantization in deeper-hop, higher budget settings; for handoff latency, the QKVShare path reduces TTFT relative to full re prefill at every tested context, from 130.7 ms vs. 150.2 ms at nominal 1K context to 397.1 ms vs. 1029.7 ms at nominal 8K context;. Stage timing shows that post-injection generation, not card creation, dominates the current QKVShare latency path. These results position quantized KV handoff as a promising on-device systems direction while also highlighting the need for stronger controller ablations and apples-to-apples runtime comparisons. 

---
# EvoLM: Self-Evolving Language Models through Co-Evolved Discriminative Rubrics 

**Authors**: Shuyue Stella Li, Rui Xin, Teng Xiao, Yike Wang, Rulin Shao, Zoey Hao, Melanie Sclar, Sewoong Oh, Faeze Brahman, Pang Wei Koh, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2605.03871)  

**Abstract**: Language models encode substantial evaluative knowledge from pretraining, yet current post-training methods rely on external supervision (human annotations, proprietary models, or scalar reward models) to produce reward signals. Each imposes a ceiling. Human judgment cannot supervise capabilities beyond its own, proprietary APIs create dependencies, and verifiable rewards cover only domains with ground-truth answers. Self-improvement from a model's own evaluative capacity is a reward source that scales with the model itself, yet remains largely untapped by current methods. We introduce EVOLM, a post-training method that structures this capacity into explicit discriminative rubrics and uses them as training signal. EVOLM trains two capabilities within a single language model in alternation: (1) a rubric generator producing instance-specific evaluation criteria optimized for discriminative utility, which maximizes a small frozen judge's ability to distinguish preferred from dispreferred responses; and (2) a policy trained using those rubric-conditioned scores as reward. All preference signals are constructed from the policy's own outputs via temporal contrast with earlier checkpoints, requiring no human annotation or external supervision. EVOLM trains a Qwen3-8B model to generate rubrics that outperform GPT-4.1 on RewardBench-2 by 25.7%. The co-trained policy achieves 69.3% average on the OLMo3-Adapt suite, outperforming policies trained with GPT-4.1 prompted rubrics by 3.9% and with the state-of-the-art 8B reward model SkyWork-RM by 16%. Overall, EVOLM demonstrates that structuring a model's evaluative capacity into co-evolving discriminative rubrics enables self-improvement without external supervision. 

---
# Quantifying the human visual exposome with vision language models 

**Authors**: Christian Rominger, Andreas R. Schwerdtfeger, Malay Gaherwar Singh, Dimitri Khudyakow, Elizabeth A. M. Michels, Fabian Wolf, Jakob Nikolas Kather, Magdalena Katharina Wekenborg  

**Link**: [PDF](https://arxiv.org/pdf/2605.03863)  

**Abstract**: The visual environment is a fundamental yet unquantified determinant of mental health. While the concept of the environmental exposome is well established, current methods rely on coarse geospatial proxies or biased self reports, failing to capture the first person visual context of daily life. We addressed this gap by coupling ecological momentary assessment with vision language models (VLMs) to quantify the semantic richness of human visual experience. Across 2674 participant generated photographs, VLM derived estimates of greenness robustly predicted momentary affect and chronic stress, consistent with established benchmarks. We then developed a semi autonomous large language model (LLM) based pipeline that mined over seven million scientific publications to extract nearly 1000 environmental features empirically linked to mental health. When applied to real world imagery, up to 33 percent of VLM extracted context ratings significantly correlated with affect and stress. These findings establish a scalable objective paradigm for visual exposomics, enabling high throughput decoding of how the visible world is associated with mental health. 

---
# Correct Is Not Enough: Training Reasoning Planners with Executor-Grounded Rewards 

**Authors**: Tianyang Han, Hengyu Shi, Junjie Hu, Xu Yang, Zhiling Wang, Junhao Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.03862)  

**Abstract**: Reinforcement learning with verifiable rewards has become a common way to improve explicit reasoning in large language models, but final-answer correctness alone does not reveal whether the reasoning trace is faithful, reliable, or useful to the model that consumes it. This outcome-only signal can reinforce traces that are right for the wrong reasons, overstate reasoning gains by rewarding shortcuts, and propagate flawed intermediate states in multi-step systems. To this end, we propose TraceLift, a planner-executor training framework that treats reasoning as a consumable intermediate artifact. During planner training, the planner emits tagged reasoning. A frozen executor turns this reasoning into the final artifact for verifier feedback, while an executor-grounded reward shapes the intermediate trace. This reward multiplies a rubric-based Reasoning Reward Model (RM) score by measured uplift on the same frozen executor, crediting traces that are both high-quality and useful. To make reasoning quality directly learnable, we introduce TRACELIFT-GROUPS, a rubric-annotated reason-only dataset built from math and code seed problems. Each example is a same-problem group containing a high-quality reference trace and multiple plausible flawed traces with localized perturbations that reduce reasoning quality or solution support while preserving task relevance. Extensive experiments on code and math benchmarks show that this executor-grounded reasoning reward improves the two-stage planner-executor system over execution-only training, suggesting that reasoning supervision should evaluate not only whether a trace looks good, but also whether it helps the model that consumes it. 

---
# Mechanical Conscience: A Mathematical Framework for Dependability of Machine Intelligenc 

**Authors**: Munkhdegerekh Batzorig, Purevbaatar Ganbold, Kyungbin Park, Pilkong Jeong, Kangbin  

**Link**: [PDF](https://arxiv.org/pdf/2605.03847)  

**Abstract**: Distributed collaborative intelligence (DCI), encompassing edge-to-edge architectures, federated learning, transfer learning, and swarm systems, creates environments in which emergent risk is structurally unavoidable: locally correct decisions by individual agents compose into globally unacceptable behavioral trajectories under uncertainty. Existing approaches such as constrained optimization, safe reinforcement learning, and runtime assurance evaluate acceptability at the level of individual actions rather than across behavioral trajectories, and none addresses the multi-participant, uncertainty-laden nature of DCI deployments. This paper introduces mechanical conscience (MC), a novel concept and simplified mathematical framework that operationalizes trajectory-level normative regulation for both single-agent and distributed intelligent systems. Mechanical conscience is defined as a supervisory filter that minimally corrects a baseline policy's actions to reduce cumulative deviation from a normatively admissible region, while accounting for epistemic uncertainty. We introduce associated constructs, conscience score, mechanical guilt, and resonant dependability, that provide an interpretable vocabulary and computable governance signals for this emerging field. Core theoretical properties are established: admissibility equivalence, existence of optimal regulation, and monotonic deviation reduction. Illustrative results demonstrate that MC-regulated agents maintain trajectory-level normative acceptability where conventional controllers drift outside admissible bounds, and that the framework naturally extends to suppress interaction-induced emergent risk in multi-agent DCI settings. 

---
# SOAR: Real-Time Joint Optimization of Order Allocation and Robot Scheduling in Robotic Mobile Fulfillment Systems 

**Authors**: Yibang Tang, Yifan Yang, Jingyuan Wang, Junhua Chen, Zhen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.03842)  

**Abstract**: Robotic Mobile Fulfillment Systems (RMFS) rely on mobile robots for automated inventory transportation, coordinating order allocation and robot scheduling to enhance warehousing efficiency. However, optimizing RMFS is challenging due to strict real-time constraints and the strong coupling of multi-phase decisions. Existing methods either decompose the problem into isolated sub-tasks to guarantee responsiveness at the cost of global optimality, or rely on computationally expensive global optimization models that are unsuitable for dynamic industrial environments. To bridge this gap, we propose SOAR, a unified Deep Reinforcement Learning framework for real-time joint optimization. SOAR transforms order allocation and robot scheduling into a unified process by utilizing soft order allocations as observations. We formulate this as an Event-Driven Markov Decision Process, enabling the agent to perform simultaneous scheduling in response to asynchronous system events. Technically, we employ a Heterogeneous Graph Transformer to encode the warehouse state and integrate phased domain knowledge. Additionally, we incorporate a reward shaping strategy to address sparse feedback in long-horizon tasks. Extensive experiments on synthetic and real-world industrial datasets, in collaboration with Geekplus, demonstrate that SOAR reduces global makespan by 7.5\% and average order completion time by 15.4\% with sub-100ms latency. Furthermore, sim-to-real deployment confirms its practical viability and significant performance gains in production environments. The code is available at this https URL. 

---
# Agentic-imodels: Evolving agentic interpretability tools via autoresearch 

**Authors**: Chandan Singh, Yan Shuo Tan, Weijia Xu, Zelalem Gero, Weiwei Yang, Michel Galley, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.03808)  

**Abstract**: Agentic data science (ADS) systems are rapidly improving their capability to autonomously analyze, fit, and interpret data, potentially moving towards a future where agents conduct the vast majority of data-science work. However, current ADS systems use statistical tools designed to be interpretable by humans, rather than interpretable by agents. To address this, we introduce Agentic-imodels, an agentic autoresearch loop that evolves data-science tools designed to be interpretable by agents. Specifically, it develops a library of scikit-learn-compatible regressors for tabular data that are optimized for both predictive performance and a novel LLM-based interpretability metric. The metric measures a suite of LLM-graded tests that probe whether a fitted model's string representation is "simulatable" by an LLM, i.e. whether the LLM can answer questions about the model's behavior by reading its string output alone. We find that the evolved models jointly improve predictive performance and agent-facing interpretability, generalizing to new datasets and new interpretability tests. Furthermore, these evolved models improve downstream end-to-end ADS, increasing performance for Copilot CLI, Claude Code, and Codex on the BLADE benchmark by up to 73% 

---
# ScrapMem: A Bio-inspired Framework for On-device Personalized Agent Memory via Optical Forgetting 

**Authors**: Jiale Chang, Yuxiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.03804)  

**Abstract**: Long-term personalized memory for LLM agents is challenging on resource-limited edge devices due to high storage costs and multimodal complexity. To address this, we propose ScrapMem, a framework that integrates multimodal data into "Scrapbook Page." ScrapMem introduces Optical Forgetting, an optical compression mechanism that progressively reduces the resolution of older memories, lowering storage cost while suppressing low-value details. To maintain semantic consistency, we construct an Episodic Memory Graph (EM-Graph) that organizes key events into a causal-temporal structure. Extensive experiments on the multimodal ATM-Bench showcase that ScrapMem provides three main benefits: (1) strong performance, achieving a new state-of-the-art with a 51.0% Joint@10 score; (2) high storage efficiency, reducing memory usage by up to 93% via optical forgetting; and (3) improved recall, increasing Recall@10 to 70.3% through structured aggregation. ScrapMem offers an effective and storage-efficient solution for on-device long-term memory in multimodal LLM agents. 

---
# Say the Mission, Execute the Swarm: Agent-Enhanced LLM Reasoning in the Web-of-Drones 

**Authors**: Andrea Iannoli, Lorenzo Gigli, Luca Sciullo, Angelo Trotta, Marco Di Felice  

**Link**: [PDF](https://arxiv.org/pdf/2605.03788)  

**Abstract**: Large Language Models (LLMs) are increasingly explored as high-level reasoning engines for cyber-physical systems, yet their application to real-time UAV swarm management remains challenging due to heterogeneous interfaces, limited grounding, and the need for long-running closed-loop execution. This paper presents a mission-agnostic, agent-enhanced LLM framework for UAV swarm control, where users express mission objectives in natural language and the system autonomously executes them through grounded, real-time interactions. The proposed architecture combines an LLM-based Agent Core with a Model Context Protocol (MCP) gateway and a Web-of-Drones abstraction based on W3C Web of Things (WoT) standards. By exposing drones, sensors, and services as standardized WoT Things, the framework enables structured tool-based interaction, continuous state observation, and safe actuation without relying on code generation. We evaluate the framework using ArduPilot-based simulation across four swarm missions and six state-of-the-art LLMs. Results show that, despite strong reasoning abilities, current general-purpose LLMs still struggle to achieve reliable execution - even for simple swarm tasks - when operating without explicit grounding and execution support. Task-specific planning tools and runtime guardrails substantially improve robustness, while token consumption alone is not indicative of execution quality or reliability. 

---
# What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity 

**Authors**: Haoxi Li, Qinglin Hou, Jianfei Ma, Jinxiang Lai, Tao Han, Sikai Bai, Jingcai Guo, Jie Zhang, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.03782)  

**Abstract**: To navigate partially observable visual environments, recent VLM agents increasingly internalize world modeling capabilities into their policies via explicit CoT reasoning, enabling them to mentally simulate futures before acting. However, relying solely on passive reasoning over visited states is insufficient for sparse-reward tasks, as it lacks the epistemic drive to actively uncover the ``known unknown'' required for robust generalization. We ask: Can VLM agents actively find signals that challenge and refine their internal world model through curiosity-driven exploration? In this work, we propose GLANCE, a unified framework that bridges reasoning and exploration by grounding the agent's linguistic world model into the stable visual representations of an evolving target network. Crucially, GLANCE leverages the discrepancy between linguistic prediction and visual reality as an intrinsic curiosity signal within reinforcement learning, steering the agent to actively explore areas where its internal model is uncertain. Extensive experiments across a series of agentic tasks show the effectiveness of GLANCE, and demonstrate that aligning ``what the agent thinks'' with ``what the agent sees'' is key to solving complex or sparse agentic tasks. 

---
# OracleProto: A Reproducible Framework for Benchmarking LLM Native Forecasting via Knowledge Cutoff and Temporal Masking 

**Authors**: Yiding Ma, Chengyun Ruan, Kaibo Huang, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.03762)  

**Abstract**: Large language models are moving from static text generators toward real-world decision-support systems, where forecasting is a composite capability that links information gathering, evidence integration, situational judgment, and action-oriented decision making. This capability is in broad demand across finance, policy, industry, and scientific research, yet its evaluation remains difficult: live benchmarks evaluate forecasts before answers exist, making them the cleanest way to measure forecasting ability, but they expire once events resolve; retrospective benchmarks are reproducible, but they cannot reliably distinguish genuine forecasting from facts a model may have already learned during pretraining. Prompting models to "pretend not to know" cannot replace a genuine knowledge boundary. We propose OracleProto, a reproducible framework for evaluating LLM native forecasting capability. OracleProto reconstructs resolved events into time-bounded forecasting samples by combining model-cutoff-aligned sample admission, tool-level temporal masking, content-level leakage detection, discrete answer normalization, and hierarchical scoring. Instantiated on a FutureX-Past-derived dataset with six contemporary LLMs, OracleProto distinguishes forecasting quality, sampling stability, and cost efficiency under controlled information boundaries, while reducing residual leakage to the $1\%$ level, an order of magnitude below tool-only temporal filtering. OracleProto turns LLM forecasting from one-off evaluation into an auditable, reusable, and trainable dataset-level capability, providing a unified interface for fair cross-model comparison and a controlled signal source for downstream SFT and RL. Code and data are available at this https URL and this https URL. 

---
# MEMTIER: Tiered Memory Architecture and Retrieval Bottleneck Analysis for Long-Running Autonomous AI Agents 

**Authors**: Bronislav Sidik, Lior Rokach  

**Link**: [PDF](https://arxiv.org/pdf/2605.03675)  

**Abstract**: Long-running autonomous AI agents suffer from a well-documented memory coherence problem: tool-execution success rates degrade 14 percentage points over 72-hour operation windows due to four compounding failure modes in existing flat-file memory systems. We present MEMTIER, a tripartite memory architecture for the OpenClaw agent runtime that introduces a structured episodic JSONL store, a five-signal weighted retrieval engine, an attention-attributed cognitive weight update loop, an asynchronous consolidation daemon promoting episodic facts to a semantic tier, and a PPO-based policy framework for adapting retrieval weights (infrastructure validated; performance gains pending camera-ready). On the full 500-question LongMemEval-S benchmark (Wu et al., 2025), MEMTIER achieves Acc=0.382, F1=0.412 with Qwen2.5-7B on a consumer 6GB GPU - a +33 percentage point improvement over the full-context baseline (0.050 -> 0.382, i.e., 5% -> 38%). With DeepSeek-V4-Flash fact pre-population, single-session recall reaches 0.686-0.714, exceeding the paper's RAG BM25 GPT-4o baseline (0.560) on those categories. Temporal reasoning rises to 0.323 and multi-session synthesis to 0.173, demonstrating that structured semantic pre-population qualitatively changes what lightweight retrieval can achieve. All phases run locally on a consumer laptop with a 6GB GPU. 

---
# Agent-Based Modeling of Low-Emission Fertilizer Adoption for Dairy Farm Decarbonisation using Empirical Farm Data 

**Authors**: Surya Jayakumar, Kieran Sullivan, John McLaughlin, Christine OMeara, Indrakshi Dey  

**Link**: [PDF](https://arxiv.org/pdf/2605.03648)  

**Abstract**: To understand complex system dynamics in dairy farming, it is essential to use modeling tools that capture farm heterogeneity, social interactions, and cumulative environmental impacts. This study proposes an agent-based modeling (ABM) framework to simulate nitrogen management and the adoption of low-emission fertilizer across 295 Irish dairy farms over a 15-year period. Using empirical data, the model represents farm communication through a social network, capturing peer influence and discussion group dynamics, where adoption probabilities are driven by social contagion, farm-scale characteristics, and policy interventions such as subsidies and carbon taxes. The framework estimates sectoral greenhouse gas emissions, cumulative abatement, and private-social cost trade-offs, using Monte Carlo simulation and sensitivity analysis to quantify uncertainty. The model shows strong agreement with observed adoption trajectories ($R^2 = 0.979$, RMSE = 0.0274) and is validated against empirical data using a Kolmogorov-Smirnov test (D = 0.2407, p < 0.001), indicating its ability to reproduce structural patterns in adoption behavior. Adoption dynamics are further characterized using a logistic diffusion model consistent with Rogers' innovation diffusion theory, capturing progression from early adoption to a saturation level of approximately 91%. By framing decarbonization as a socio-technical diffusion process rather than a purely economic optimization problem, this study provides an in silico policy laboratory for evaluating the robustness and diffusion speed of climate mitigation strategies prior to implementation. 

---
# AdapShot: Adaptive Many-Shot In-Context Learning with Semantic-Aware KV Cache Reuse 

**Authors**: Jie Ou, Jinyu Guo, Shiyao Guo, Yuang Li, Ruiqi Wu, Zhaokun Wang, Wenyi Li, Wenhong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2605.03644)  

**Abstract**: Many-Shot In-Context Learning (ICL) has emerged as a promising paradigm, leveraging extensive examples to unlock the reasoning potential of Large Language Models (LLMs). However, existing methods typically rely on a predetermined, fixed number of shots. This static approach often fails to adapt to the varying difficulty of different queries, leading to either insufficient context or interference from noise. Furthermore, the prohibitive computational and memory costs of long contexts severely limit Many-Shot's feasibility. To address the above limitations, we propose AdapShot, which dynamically optimizes shot counts and leverages KV cache reuse for efficient inference. Specifically, we design a probe-based evaluation mechanism that utilizes output entropy to determine the optimal number of shots. To bypass the redundant prefilling computation during both the probing and inference phases, we incorporate a semantics-aware KV cache reuse strategy. Within this reuse strategy, to address positional encoding incompatibilities, we introduce a decoupling and re-encoding method that enables the flexible reordering of cached key-value pairs. Extensive experiments demonstrate that AdapShot achieves an average performance gain of around 10% and a 4.64x speedup compared to state-of-the-art DBSA. 

---
# Self-Improvement for Fast, High-Quality Plan Generation 

**Authors**: Robert Gieselmann, Henrike von Huelsen, Mihai Samson, Marie-Christine Meyer, Dariusz Piotrowski, Oleksandr Radomskyi, Justin Okamoto, Turan Gojayev, Michael Painter, Gavin Brown, Federico Pecora, Jeremy L. Wyatt  

**Link**: [PDF](https://arxiv.org/pdf/2605.03625)  

**Abstract**: Generative models trained on synthetic plan data are a promising approach to generalized planning. Recent work has focused on finding any valid plan, rather than a high-quality solution. We address the challenge of producing high-quality plans, a computationally hard problem, in sub-exponential time. First, we demonstrate that, given optimal data, a decoder-only transformer can generate high-quality plans for unseen problem instances. Second, we show how to self-improve an initial model trained on sub-optimal data. Each round of self-improvement combines multiple model calls with graph search to generate improved plans, used for model fine-tuning. An experimental study on four domains: Blocksworld, Logistics, Labyrinth, and Sokoban, shows on average a 30% reduction in plan length over the source symbolic planner, with over 80% of plans being optimal, where the optimum is known. Plan quality is further improved by inference-time search. The model's latency scales sub-exponentially in contrast to the satisficing and optimal symbolic planners to which we compare. Together, these results suggest that self-improvement with generative models offers a scalable approach for high-quality plan generation. 

---
# Where Paths Split: Localized, Calibrated Control of Moral Reasoning in Large Language Models 

**Authors**: Chenchen Yuan, Zheyu Zhang, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2605.03609)  

**Abstract**: Large language models often display heterogeneous moral preferences across settings. We study inference-time steering toward a desired ethical framework while preserving general competence. We present Convergent-Divergent Routing, which traces and edits minimal branch points inside transformer blocks where ethical-framework-related pathways first converge and then diverge. Gating non-target branches at these loci blocks the downstream propagation while leaving upstream computations intact. We find that this intervention alone increases targeted ethical-framework reasoning. To achieve fine-grained control, we adapt Common Spatial Patterns to the residual stream and extract, for each branch-point layer, a pair of directions that discriminate between utilitarian and deontological frameworks. We then introduce Dual Logit Calibration, a closed-form, minimum-$\ell_2$-norm update that moves the residual within this two-dimensional subspace so the resulting directional projections align with user-specified preference weights. Experiments on real-life moral dilemmas show that our method reliably achieves preference calibration and largely preserves general capabilities, outperforming recent baselines while providing an interpretable mechanism. 

---
# Workspace-Bench 1.0: Benchmarking AI Agents on Workspace Tasks with Large-Scale File Dependencies 

**Authors**: Zirui Tang, Xuanhe Zhou, Yumou Liu, Linchun Li, Weizheng Wang, Hongzhang Huang, Jun Zhou, Jiachen Song, Shaoli Yu, Jinqi Wang, Zihang Zhou, Hongyi Zhou, Yuting Lv, Jinyang Li, Jiashuo Liu, Ruoyu Chen, Chunwei Liu, GuoLiang Li, Jihua Kang, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03596)  

**Abstract**: Workspace learning requires AI agents to identify, reason over, exploit, and update explicit and implicit dependencies among heterogeneous files in a worker's workspace, enabling them to complete both routine and advanced tasks effectively. Despite its importance, existing relevant benchmarks largely evaluate agents on pre-specified or synthesized files with limited real-world dependencies, leaving workspace-level evaluation underexplored. To this end, we introduce Workspace-Bench, a benchmark for evaluating AI agents on Workspace Learning invOlving Large-Scale File Dependencies. We construct realistic workspaces with 5 worker profiles, 74 file types, 20,476 files (up to 20GB) and curate 388 tasks, each with its own file dependency graph, evaluated across 7,399 total rubrics that require cross-file retrieval, contextual reasoning, and adaptive decision-making. We further provide Workspace-Bench-Lite, a 100-task subset that preserves the benchmark distribution while reducing evaluation costs by about 70%. We evaluate 4 popular agent harnesses and 7 foundation models. Experimental results show that current agents remain far from reliable workspace learning, where the best reaches only 68.7%, substantially below the human result of 80.7%, and the average performance across agents is only 47.4%. 

---
# Real-Time Evaluation of Autonomous Systems under Adversarial Attacks 

**Authors**: Adithya Mohan, Xujun Xie, Venkatesh Thirugnana Sambandham, Torsten Schön  

**Link**: [PDF](https://arxiv.org/pdf/2605.03491)  

**Abstract**: Most evaluations of autonomous driving policies under adversarial conditions are conducted in simulation, due to cost efficiency and the absence of physical risk. However, purely virtual testing fails to capture structural inconsistencies, supervision constraints, and state-representation effects that arise in real-world data and fundamentally shape policy robustness. This work presents an offline trajectory-learning and adversarial robustness evaluation framework grounded in real-world intersection driving data. Within a controlled data contract, we train and compare three trajectory-learning paradigms: Multi-Layer Perceptron (MLP)-based Behavior Cloning (BC), Transformer-based object-tokenized BC, and inverse reinforcement learning (IRL) formulated within a Generative Adversarial Imitation Learning (GAIL) framework. Models are evaluated using Average Displacement Error (ADE) and Final Displacement Error (FDE).
Inference-time robustness is assessed by subjecting trained policies to gradient-based adversarial perturbations across multiple intersection scenarios, yielding a structured robustness evaluation matrix. Results show that state-structure design and architectural inductive biases critically influence adversarial stability, leading to markedly different robustness profiles despite comparable nominal prediction accuracy (ADE < 0.08). Inference-time Projected Gradient Descent (PGD) attacks induce final displacement errors of up to approximately 8 meters. The proposed framework establishes a scalable benchmark for studying offline trajectory learning and adversarial robustness in real-world autonomous driving settings. 

---
# FinSTaR: Towards Financial Reasoning with Time Series Reasoning Models 

**Authors**: Seunghan Lee, Jun Seo, Jaehoon Lee, Sungdong Yoo, Minjae Kim, Tae Yoon Lim, Dongwan Kang, Hwanil Choi, Soonyoung Lee, Wonbin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2605.03460)  

**Abstract**: Time series (TS) reasoning models (TSRMs) have shown promising capabilities in general domains, yet they consistently fail on financial domain, which exhibit unique characteristics. We propose a general 2x2 capability taxonomy for TSRMs by crossing 1) single-entity vs. multi-entity analysis with 2) assessment of the current state vs. prediction of future behavior. We instantiate this taxonomy in the financial domain -- where the distinction between deterministic assessment and stochastic prediction is particularly critical -- as ten financial reasoning tasks, forming the FinTSR-Bench benchmark based on S&P stocks. To this end, we propose FinSTaR (Financial Time Series Thinking and Reasoning), trained on FinTSR-Bench with distinct chain-of-thought (CoT) strategies tailored to each category. For assessment, which is deterministic (i.e., computable from observable data), we employ Compute-in-CoT, a programmatic CoT that enables models to derive answers directly from raw prices. For prediction, which is inherently stochastic (i.e., subject to unobservable factors), we adopt Scenario-Aware CoT, which generates diverse scenarios before making a judgment, mirroring how financial analysts reason under uncertainty. The proposed method achieves 78.9% average accuracy on FinTSR-Bench, substantially outperforming LLM and TSRM baselines. Furthermore, we show that the four capability categories are complementary and mutually reinforcing through joint training, and that Scenario-Aware CoT consistently improves prediction accuracy over standard CoT. Code is publicly available at: this https URL. 

---
# Replacing Parameters with Preferences: Federated Alignment of Heterogeneous Vision-Language Models 

**Authors**: Shule Lu, Yujing Wang, Hainan Zhang, Xiaoshan Yang, Hongwei Zheng, Yongxin Tong, Changsheng Xu, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.03426)  

**Abstract**: Vision-Language Models (VLMs) have broad potential in privacy-sensitive domains such as healthcare and finance, yet strict data-sharing constraints render centralized training infeasible. Federated Learning mitigates this issue by enabling decentralized training, but practical deployments face challenges due to client heterogeneity in computational resources, application requirements, and model architectures. Under extreme model and data heterogeneity, replacing parameter aggregation with preference-based collaboration offers a more suitable interface, as it eliminates the need for direct parameter or data exchange. Motivated by this, we propose MoR, a federated alignment framework that combines GRPO with Mixture-of-Rewards for heterogeneous VLMs. In MoR, each client locally trains a reward model from local preference annotations, capturing specific evaluation signals without exposing raw data. To combine these heterogeneous supervision signals, MoR introduces a Mixture-of-Rewards mechanism with learned routing, which adaptively fuses client reward models according to the input and alignment objective. The server then optimizes a base VLM using GRPO with a KL penalty to a reference model, enabling preference alignment without requiring client models to share architectures or parameters. Experiments on diverse public vision-language benchmarks demonstrate that MoR consistently outperforms federated alignment baselines in generalization and cross-client adaptability. Our approach provides a scalable solution for privacy-preserving alignment of heterogeneous VLMs under federated settings. 

---
# Adaptive Dual-Path Framework for Covert Semantic Communication 

**Authors**: Xi Yu, Weicai Li, Lin Yin, Tiejun Lv  

**Link**: [PDF](https://arxiv.org/pdf/2605.03423)  

**Abstract**: This paper proposes a novel adaptive dual-path framework for covert semantic communication (SemCom), which integrates covert information transmission with task-oriented semantic coding. Unlike conventional covert communication methods that embed hidden messages through power-domain signal superposition, our framework embeds covert data within task-specific features via semantic-level intrinsic encoding. This new architecture introduces dual encoding paths with adaptive block selection: an Explicit path for public task execution and a Stego path that jointly encodes both public and covert information through contrastive representation alignment. A Gumbel-Softmax enabled adaptive path selection mechanism dynamically activates network blocks based on task require- ments. We formulate a multi-objective optimization framework that simultaneously ensures accurate semantic understanding and reliable covert transmission. We rigorously evaluate our framework's security against a powerful, independently trained attacker. Experimental results on the Cityscapes dataset demon- strate a state-of-the-art level of covertness: our method suppresses the attacker's detection accuracy to a near-random guessing level of 56.12%. This robust security is achieved while simultaneously maintaining superior performance on the primary semantic tasks compared to the baselines. 

---
# Geometry over Density: Few-Shot Cross-Domain OOD Detection 

**Authors**: Shawn Li, You Qin, Jiate Li, Charith Peris, Lisa Bauer, Roger Zimmermann, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.03410)  

**Abstract**: Out-of-distribution (OOD) detection identifies test samples that fall outside a model's training distribution, a capability critical for safe deployment in high-stakes applications. Standard OOD detectors are trained on a specific in-distribution (ID) dataset and detect deviations from that single domain. In contrast, we study few-shot cross-domain OOD detection: given a \emph{single} pre-trained model, can we perform OOD detection on \emph{arbitrary} new ID-OOD task pairs using only a handful of ID samples at inference time, with no additional training? We propose \textbf{UFCOD}, a unified framework that achieves this goal through information-geometric analysis of diffusion trajectories. Our key insight is that diffusion noise predictions are score functions (gradients of log-density), and we extract two energy features: \emph{Path Energy} (integrated score magnitude) and \emph{Dynamics Energy} (score smoothness), that form a discrete Sobolev norm capturing how samples interact with the learned diffusion process. The central contribution is a \textbf{train-once, deploy-anywhere} paradigm: a diffusion model trained on a single dataset (e.g., CelebA) serves as a universal feature extractor for OOD detection across semantically unrelated domains (e.g., CIFAR-10, SVHN, Textures). At deployment, each new task requires only $\sim$100 unlabeled ID samples for inference: no retraining, no fine-tuning, no task-specific adaptation. Using 100 ID samples per task, UFCOD achieves 93.7\% average AUROC across 12 cross-domain benchmarks, competitive with methods trained on 50k--163k samples, demonstrating $\sim$500$\times$ improvement in sample efficiency. See our code in this https URL. 

---
# Robust Agent Compensation (RAC): Teaching AI Agents to Compensate 

**Authors**: Srinath Perera, Kaviru Hapuarachchi, Frank Leymann, Rania Khalaf  

**Link**: [PDF](https://arxiv.org/pdf/2605.03409)  

**Abstract**: We present Robust Agent Compensation (RAC), a log-based recovery paradigm (providing a safety net) implemented through an architectural extension that can be applied to most Agent frameworks to support reliable executions (avoiding unintended side effects). Users can choose to enable RAC without changing their current agent code (e.g., LangGraph agents). The proposed approach can be implemented in most existing agent frameworks via their existing extension points. We present an implementation based on LangChain, demonstrate its viability through the $\tau$-bench and REALM-Bench, and show that when solving complex problems, RAC is 1.5-8X or more better in both latency and token economy compared to state-of-the-art LLM-based recovery approaches. 

---
# GeoDecider: A Coarse-to-Fine Agentic Workflow for Explainable Lithology Classification 

**Authors**: Jiahao Wang, Mingyue Cheng, Yitong Zhou, Qingyang Mao, Xiaoyu Tao, Qi Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.03383)  

**Abstract**: Lithology classification aims to infer subsurface rock types from well-logging signals, supporting downstream applications like reservoir characterization. Despite substantial progress, most existing methods still treat lithology classification as a single-pass classification task. In contrast, practical experts incorporate geological principles, external knowledge, and tool-use capabilities to perform accurate classification. In this work, we propose GeoDecider, a coarse-to-fine agentic workflow that enables accurate and explainable lithology classification through training-free use of large language models (LLMs). GeoDecider reformulates lithology classification as an expert-like structured process and organizes it into a multi-stage workflow involving coarse-to-fine reasoning. Specifically, GeoDecider includes the following stages: (1) base classifier-guided coarse classification, which uses a pre-trained classifier to provide a rough reference for downstream tasks, thus reducing the overall cost of downstream reasoning, (2) tool-augmented reasoning, which utilizes several tools such as contextual analysis and neighbor retrieval to achieve finer and more precise classifications, (3) geological refinement, which post-processes the final results to enforce geological consistency. Experiments on four benchmarks show that GeoDecider outperforms representative baselines. Further analysis demonstrates that the proposed framework produces geologically interpretable predictions while achieving a better trade-off between classification performance and inference efficiency. 

---
# ReasonAudio: A Benchmark for Evaluating Reasoning Beyond Matching in Text-Audio Retrieval 

**Authors**: Honglei Zhang, Yuting Chen, Chenpeng Hu, Siyue Zhang, Yilei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.03361)  

**Abstract**: As multimodal content continues to expand at a rapid pace, audio retrieval has emerged as a key enabling technology for media search, content organization, and intelligent assistants. However, most existing benchmarks concentrate on semantic matching and fail to capture the fact that real-world queries often demand advanced reasoning abilities, including negation understanding, temporal ordering, concurrent event recognition, and duration discrimination. To address this gap, we introduce ReasonAudio, the first reasoning-intensive benchmark for Text-Audio Retrieval, comprising 1,000 queries and 10,000 composite audio clips across five fundamental reasoning tasks: Negation, Order, Overlap, Duration, and Mix. Despite their intuitive nature for humans and straightforward construction, these tasks pose significant challenges to current models. Our evaluation of ten state-of-the-art models reveals the following findings: All models struggle with reasoning-intensive audio retrieval, performing particularly poorly on Negation and Duration while showing relatively better results on Overlap and Order. Moreover, Multimodal Large Language Model-based embedding models fail to inherit the reasoning capabilities of their backbones through contrastive fine-tuning, suggesting that current training paradigms are insufficient to preserve reasoning capacity in retrieval settings 

---
# What Happens Inside Agent Memory? Circuit Analysis from Emergence to Diagnosis 

**Authors**: Xutao Mao, Jinman Zhao, Gerald Penn, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03354)  

**Abstract**: Agent memory failures are silent: an LLM-based agent can produce a fluent response even when it fails to extract, retain, or retrieve the information needed across sessions. The write-manage-read loop describes the external pipeline of these systems but leaves open which internal computations implement each stage. Tracing internal feature circuits across the Qwen-3 family (0.6B--14B) and two memory frameworks (mem0 and A-MEM), we report three findings. First, control is detectable before content: routing circuitry is causally active at 0.6B, while content circuitry produces no detectable signal until 4B under our tracing setup, creating a deployment regime where small models route with apparent competence but silently fail at extraction and grounding. Second, within the content group, Write and Read share a late-layer hub that operates as a context-grounding substrate already present in the base model; only memory framing recruits a functional grounding direction on this substrate, and the hub transfers across both frameworks. Third, emergence does not imply steerability: although the content circuit becomes detectable at 4B, it becomes reliably steerable only at 8B, indicating that detection and intervention have distinct scale thresholds. As a practical implication, the feature-space separation between the two circuit groups enables per-operation failure localization at 76.2% accuracy without supervision, providing a stage-level diagnostic for otherwise silent agent-memory failures. 

---
# Automated Large-scale CVRP Solver Design via LLM-assisted Flexible MCTS 

**Authors**: Tong Guo, Caishun Chen, Yew Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2605.03339)  

**Abstract**: Solving large-scale CVRP (LSCVRP) with hundreds to thousands of nodes remains difficult for even state-of-the-art solvers. Divide-and-conquer can scale by decomposing the instance into size-reduced subproblems, but designing decomposition logic and configuring sub-solvers is highly expertise- and labor-intensive. Large Language Models (LLMs) have emerged as promising tools for automated algorithm design. However, existing LLM-driven approaches struggle with LSCVRP primarily due to the difficulty in generating sophisticated search strategies within a limited context window. To bridge this gap, we propose the LLM-assisted Flexible Monte Carlo Tree Search (LaF-MCTS), a novel framework that automates the design of high-performance LSCVRP solvers. We develop a three-tier decision hierarchy to enable incremental design of decomposition policies and sub-solvers for LSCVRP. To enable efficient search within the algorithmic hypothesis space, we introduce semantic pruning to eliminate semantically and structurally redundant codes, and branch regrowth to regenerate codes and preserve diversity. Extensive experiments on CVRPLib demonstrate that LaF-MCTS autonomously composes and optimizes decomposition-enhanced solvers that surpasses various state-of-the-art CVRP solvers. 

---
# Revisiting the Travel Planning Capabilities of Large Language Models 

**Authors**: Bo-Wen Zhang, Jin Ye, Peng-Yu Hua, Jia-Wei Cao, Jie-Jing Shao, Yu-Feng Li, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.03308)  

**Abstract**: Travel planning serves as a critical task for long-horizon reasoning, exposing significant deficits in LLMs. However, existing benchmarks and evaluations primarily assess final plans in an end-to-end manner, which lacks interpretability and makes it difficult to analyze the root causes of failures. To bridge this gap, we decompose travel planning into five constituent atomic sub-capabilities, including \emph{Constraint Extraction}, \emph{Tool Use}, \emph{Plan Generation}, \emph{Error Identification}, and \emph{Error Correction}. We implement a decoupled evaluation protocol leveraging oracle intermediate contexts to rigorously isolate these components, thereby measuring the atomic performance boundary without the noise of cascading errors. Our results highlight a clear contrast in performance: while LLMs are proficient in extracting explicit constraints, they struggle to infer implicit, open-world requirements. Furthermore, they exhibit structural biases in plan generation and suffer from ineffective self-correction, characterized by excessive sensitivity and erroneous persistence. These findings offer precise directions for improving LLM reasoning and planning abilities. 

---
# Enhancing Agent Safety Judgment: Controlled Benchmark Rewriting and Analogical Reasoning for Deceptive Out-of-Distribution Scenarios 

**Authors**: Zuoyu Zhang, Yancheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03242)  

**Abstract**: Tool-using agent systems powered by large language models (LLMs) are increasingly deployed across web, app, operating-system, and transactional environments. Yet existing safety benchmarks still emphasize explicit risks, potentially overstating a model's ability to judge deceptive or ambiguous trajectories. To address this gap, we introduce ROME (Red-team Orchestrated Multi-agent Evolution), a controlled benchmark-construction pipeline that rewrites known unsafe trajectories into more deceptive evaluation instances while preserving their underlying risk labels. Starting from 100 unsafe source trajectories, ROME produces 300 challenge instances spanning contextual ambiguity, implicit risks, and shortcut decision-making. Experiments show that these challenge sets substantially degrade safety-judgment performance, with hidden-risk cases remaining particularly non-trivial even for recent frontier models. We further study ARISE (Analogical Reasoning for Inference-time Safety Enhancement), a retrieval-guided inference-time enhancement that retrieves ReAct-style analogical safety trajectories from an external analogical base and injects them as structured reasoning exemplars. ARISE improves judgment quality without retraining, but is best viewed as a task-specific robustness enhancement rather than a standalone safety guarantee. Together, ROME and ARISE provide practical tools for stress-testing and improving agent safety judgment under deceptive distribution shifts. 

---
# cotomi Act: Learning to Automate Work by Watching You 

**Authors**: Masafumi Oyamada, Kunihiro Takeoka, Kosuke Akimoto, Ryoma Obara, Masafumi Enomoto, Haochen Zhang, Daichi Haraguchi, Takuya Tamura  

**Link**: [PDF](https://arxiv.org/pdf/2605.03231)  

**Abstract**: What if a browser agent could learn your work simply by watching you do it? We present cotomi Act, a browser-based computer-using agent that combines reliable multi-step task execution with persistent organizational knowledge learned from user behavior. For execution, an agent scaffold with adaptive lazy observation, verbal-diff-based history compression, coarse-grained actions, and test-time scaling via best-of-N action selection achieves 80.4% on the 179-task WebArena human-evaluation subset, exceeding the reported 78.2% human baseline. For organizational knowledge, a behavior-to-knowledge pipeline passively observes the user's browsing and progressively abstracts it into artifacts (task boards, wiki) exposed through a shared workspace editable by both user and agent. A controlled proxy evaluation confirms that task success improves as behavior-derived knowledge accumulates. In our live demonstration, attendees interact with the system in a real browser, issuing tasks and observing end-to-end autonomous execution and shared knowledge management. 

---
# Evaluating Prompting and Execution-Based Methods for Deterministic Computation in LLMs 

**Authors**: Hongkun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03227)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in natural language understanding and reasoning. However, their ability to perform exact, deterministic computation remains unclear. In this work, we systematically evaluate multiple prompting strategies, including Chain-of-Thought (CoT), Least-to-Most decomposition, Program-of-Thought (PoT), and Self-Consistency (SC), on tasks requiring precise and error-free outputs, including binary counting, longest substring detection, and arithmetic evaluation.
To support this study, we introduce a synthetic dataset with diverse natural language instructions, enabling controlled evaluation of exact computation across multiple task types.
Our results show that standard prompting methods achieve only moderate accuracy on sequence-based tasks. CoT provides limited improvement, while Least-to-Most suffers from error accumulation. In contrast, PoT achieves perfect accuracy by generating executable code and delegating computation to an external interpreter. Self-Consistency improves robustness through majority voting, but incurs substantial computational overhead.
We further train a small domain-specific model (CodeT5-small) to generate executable programs, which achieves perfect accuracy on held-out synthetic test data across all tasks with minimal training cost.
Overall, our findings suggest that LLMs may simulate reasoning patterns rather than reliably perform exact symbolic computation. For deterministic tasks, combining LLMs with external tools or using specialized models provides a more reliable and efficient solution. 

---
# ADAPTS: Agentic Decomposition for Automated Protocol-agnostic Tracking of Symptoms 

**Authors**: Alexandria K. Vail, Marcelo Cicconet, Katie Aafjes-van Doorn, Ryan Maroney, Marc Aafjes  

**Link**: [PDF](https://arxiv.org/pdf/2605.03212)  

**Abstract**: Modeling latent clinical constructs from unconstrained clinical interactions is a unique challenge in affective computing. We present ADAPTS (Agentic Decomposition for Automated Protocol-agnostic Tracking of Symptoms), a framework for automated rating of depression and anxiety severity using a mixture-of-agents LLM architecture. This approach decomposes long-form clinical interviews into symptom-specific reasoning tasks, producing auditable justifications while preserving temporal and speaker alignment. Generalization was evaluated across two independent datasets ($N=204$) with distinct interview structures. On high-discrepancy interviews, automated ratings approximated expert benchmarks ($\text{absolute error}=22$) more closely than original human ratings ($\text{absolute error}=26$). Implementing an ``extended'' protocol that incorporates qualitative clinical conventions significantly stabilized ratings, with absolute agreement reaching $\text{ICC(2,1)} = 0.877$. These findings suggest that the ADAPTS framework enables promising evaluations of psychiatric severity. While the current implementation is purely text-based, the underlying architecture is readily extensible to multimodal inputs, including acoustic and visual features. By approximating expert-level precision in a protocol-agnostic manner, this framework provides a foundation for objective and scalable psychiatric assessment, especially in resource-limited settings. 

---
# Stop Automating Peer Review Without Rigorous Evaluation 

**Authors**: Joachim Baumann, Jiaxin Pei, Sanmi Koyejo, Dirk Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2605.03202)  

**Abstract**: Large language models offer a tempting solution to address the peer review crisis. This position paper argues that today's AI systems should not be used to produce paper reviews. We ground this position in an empirical comparison of human- versus AI-generated ICLR 2026 reviews and an evaluation of the effect of automated paper rewriting on different AI reviewers. We identify two critical issues: 1) AI reviewers exhibit a hivemind effect of excessive agreement within and across papers that reduces perspective diversity. 2) AI review scores are trivially gameable through paper laundering: prompting an LLM to rewrite a paper could significantly increase the scores from AI reviewers, demonstrating that LLM reviewers are easy to game through stylistic changes rather than scientific results. However, non-gameability and review diversity are necessary but not sufficient conditions for automation. We argue that addressing the peer review crisis requires a science of peer review automation -- not general-purpose LLMs deployed without rigorous evaluation. 

---
# Terminus-4B: Can a Smaller Model Replace Frontier LLMs at Agentic Execution Tasks? 

**Authors**: Spandan Garg, Vikram Nitin, Yufan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03195)  

**Abstract**: Modern coding agents increasingly delegate specialized subtasks to subagents, which are smaller, focused agentic loops that handle narrow responsibilities like search, debugging or terminal execution. This architectural pattern keeps the main agent's context window clean by isolating verbose outputs (e.g. build logs, test results, etc.) within the subagent context. Typically when agents employ subagents for such tasks, they use frontier models as these subagents. In this paper, we investigate whether a finetuned small language model (SLM) can achieve comparable performance to frontier models in the task of agentic terminal execution. We present Terminus-4B, which is a post-trained Qwen3-4B model via Supervised Finetuning (SFT) and Reinforcement Learning (RL) using rubric-based LLM-as-judge reward, specifically for this task. In our extensive evaluation spanning various frontier models, training ablations and main agent configurations, we find that Terminus-4B is able to reduce the token usage of the main agent by up to ~30% compared to the No Subagent baseline with no impact to agent performance on benchmarks like SWE-Bench Pro and our internal SWE-Bench C# benchmark, which tends to be heavy in verbose execution tasks. Furthermore, Terminus-4B improves key metrics showing the main agent relying on the outputs of the subagent and doing fewer terminal execution tasks by itself. We see that our model not only closes the gap between the Vanilla Qwen model and frontier models like Claude Sonnet / Opus / GPT-5.3-Codex, but often even exceeds their performance. 

---
# Learning Correct Behavior from Examples: Validating Sequential Execution in Autonomous Agents 

**Authors**: Reshabh K Sharma, Gaurav Mittal, Yu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03159)  

**Abstract**: As autonomous agents become increasingly sophisticated, validating their sequential behavior presents a significant challenge. Traditional testing approaches require manual specification, exact sequence matching, or thousands of training examples. We present a novel algorithm that automatically learns correct behavior from just 2-10 passing execution traces and validates new executions against this learned model. Our approach combines dominator analysis from compiler theory with multimodal large language model-powered semantic understanding to identify essential states and handle non-deterministic behavior. The system constructs a generalized ground truth model using Prefix Tree Acceptors, merges traces through multi-tiered equivalence detection, and validates new executions via topological subsequence matching. In controlled experiments, our system achieved high accuracy in detecting product bugs and false successes using only 3 training traces. This approach provides explainable validation results with coverage metrics and works across diverse domains including UI testing, code generation, and robotic processes. 

---
# Are you with me? A Framework for Detecting Mental Model Discrepancies in Task-Based Team Dialogues 

**Authors**: Katharine Kowalyshyn, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2605.03149)  

**Abstract**: Humans typically use natural language to update teammates on task states. Since not all updates are communicated, discrepancies arise between the team members' mental models that negatively affect overall team performance. How can we categorize such discrepancies? Do misalignments detected in team dialogue predict future mental model misalignments? Traditional shared mental model (SMM) assessment methods rely on retrospective expert coding that cannot capture real-time coordination dynamics. We propose a framework to identify and categorize four types of mental model discrepancies: unsupported beliefs, false beliefs, belief contradictions, and omissions, all of which can naturally emerge in team dialogues. Using dialogues from twenty dyad teams performing collaborative object identification tasks across four sequential levels, we demonstrate that these discrepancy patterns contain predictive signals. Averaging historical discrepancy counts achieves meaningful prediction accuracy using uniform weighting as an exploratory baseline, with differential predictability across discrepancy types. 

---
# Programmatic Context Augmentation for LLM-based Symbolic Regression 

**Authors**: Hao Liu, Xiao-Wen Yang, Atharva Sehgal, Yixin Wang, Lan-Zhe Guo, Yu-Feng Li, Yisong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2605.03101)  

**Abstract**: Symbolic regression (SR), the task of discovering mathematical expressions that best describe a given dataset, remains a fundamental challenge in scientific discovery. Traditional approaches, primarily based on genetic algorithms and related evolutionary methods, have proven useful but suffer from scalability and expressivity limitations. Recently, large language model (LLM)-based evolutionary search methods have been introduced into SR and show promise. However, existing LLM-based approaches typically rely on scalar evaluation metrics, such as mean squared error, as the sole source of feedback during the search process, thereby overlooking the rich information embedded in the dataset. To address this limitation, we propose a novel LLM-based evolutionary search framework that incorporates programmatic context augmentation. By enabling code-based interactions with the dataset, our method can actively perform data analysis and extract informative signals, beyond aggregated evaluation scores. We evaluate our framework on advanced benchmarks, such as LLM-SRBench, and demonstrate superior efficiency and accuracy compared to strong baselines. 

---
# Making the Invisible Visible: Understanding the Mismatch Between Organizational Goals and Worker Experiences in AI Adoption 

**Authors**: Christine P. Lee, Min Kyung Lee, Bilge Mutlu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03078)  

**Abstract**: While AI is often introduced into organizations to drive innovation and efficiency, many adoption efforts fail as workers resist and struggle to integrate these systems. These failures point to a deeper issue: workers, the very people expected to collaborate with AI, are often invisible in decisions about how AI is designed and used. Drawing on interviews with professionals who interact with AI systems daily in healthcare, finance, and management, we examine the disconnect between organizational expectations and worker experiences. We identify key barriers, including poor usability and interoperability, misaligned expectations, limited control, and insufficient communication. These challenges highlight a gap between how organizations implement AI and the evolving worker needs, tasks, and workflows that it fails to support. We argue that successful adoption requires recognizing workers as central to AI integration and propose adaptation strategies at the individual, task, and organizational levels to better align AI systems with real-world practices. 

---
# Computing Thiele Rules on Interval Elections and their Generalizations 

**Authors**: Dimitris Avramidis, Alexandra Lassota, Ulrike Schmidt-Kraepelin, Adrian Vetta  

**Link**: [PDF](https://arxiv.org/pdf/2605.03067)  

**Abstract**: Approval-based committee voting has received significant attention in the social choice community. Among the studied rules, Thiele rules, and especially Proportional Approval Voting (PAV), stand out for desirable properties such as proportional representation, Pareto optimality, and support monotonicity. Their main drawback is that computing a Thiele outcome is NP-hard in general.
A glimpse of hope comes from the fact that Thiele rules are better behaved under structured preferences. On the candidate interval (CI) domain, they are computable in polynomial time via a linear program (LP) that has a totally unimodular constraint matrix. Surprisingly, this approach fails for the related voter interval (VI) domain, and the complexity of the problem has repeatedly been posed as an open question. Our main result resolves this question: although the relevant matrix is not totally unimodular, the ``standard'' LP still admits at least one optimal integral solution, and we provide a fast algorithm for finding it.
Our technique naturally extends to the voter-candidate interval (VCI) domain, also known as the 1-dimensional voter-candidate range (1D-VCR) domain, and to the linearly consistent (LC) domain, both of which generalize the candidate and voter interval domains. Although both the VCI and LC domains have been studied in social choice, their relationship was unknown. We show, through connections to graph theory, that LC strictly contains VCI. We also provide an alternative definition of LC that is closer in spirit to VCI and has a natural interpretation in approval elections; this equivalence may be of independent interest. Finally, we study an alternative tree-based generalization of VCI and show that Thiele rules become NP-hard to compute on this domain. 

---
# Stable Agentic Control: Tool-Mediated LLM Architecture for Autonomous Cyber Defense 

**Authors**: Kerri Prinos, Lilianne Brush, Cameron Denton, Zhanqi Wang, Joshua Knox, Snehal Antani, Anton Foltz, Amy Villaseñor  

**Link**: [PDF](https://arxiv.org/pdf/2605.03034)  

**Abstract**: Agentic systems involved in high-stake decision-making under adversarial pressure need formal guarantees not offered by existing approaches. Motivated by the operational needs of security operations centers (SOCs) that must configure endpoint detection and response (EDR) policies under adversarial pressure, we present a tool-mediated architecture: LLM agents use deterministic tools (Stackelberg best-response, Bayesian observer updates, attack-graph primitives) and select from finite action catalogs enforced at the tool-output interface. A composite Lyapunov function machine-checked in Lean 4 with zero sorry certifies controllability, observability from asymmetric sensor data, and Input-to-State Stability (ISS) robustness under intelligent adversarial disturbance, with two corollaries extending the certificate to any controller or adversary from the catalogs. On 282 real enterprise attack graphs, the claims hold with margin. On paired offensive/defensive telemetry, a tool-mediated Claude Sonnet 4 controller reduces the attacker's expected payoff (game value) by 59% relative to a deterministic greedy baseline, with zero variance across 40 runs at four temperatures. A Claude Haiku 4.5 controller converges to suboptimal game values but stays catalog-bounded over an additional 40 runs, demonstrating that architectural stability is not dependent on the controller capability. The LLM agent's non-determinism furthers creative exploration of strategies, while the tool-mediated architecture ensures system stability. 

---
# CreativityBench: Evaluating Agent Creative Reasoning via Affordance-Based Tool Repurposing 

**Authors**: Cheng Qian, Hyeonjeong Ha, Jiayu Liu, Bingxiang He, Jeonghwan Kim, Jiateng Liu, Bingxuan Li, Aditi Tiwari, Dwip Dalal, Zhenhailong Wang, Xiusi Chen, Mahdi Namazifar, Yunzhu Li, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.02910)  

**Abstract**: Recent advances in large language models have led to strong performance on reasoning and environment-interaction tasks, yet their ability for creative problem-solving remains underexplored. We study this capability through the lens of creative tool use, where a model repurposes available objects by reasoning about their affordances and attributes rather than relying on canonical usage. As a first step, we introduce CreativityBench, a benchmark for evaluating affordance-based creativity in LLMs. To this end, we build a large-scale affordance knowledge base (KB) with 4K entities and 150K+ affordance annotations, explicitly linking objects, parts, attributes, and actionable uses. Building on this KB, we generate 14K grounded tasks that require identifying non-obvious yet physically plausible solutions under constraints. Evaluations across 10 state-of-the-art LLMs, including closed and open-source models, show that models can often select a plausible object, but fail to identify the correct parts, their affordances, and the underlying physical mechanism needed to solve the task, leading to a significant drop in performance. Furthermore, improvements from model scaling quickly saturate, strong general reasoning does not reliably translate to creative affordance discovery, and common inference-time strategies such as Chain-of-Thought yield limited gains. These results suggest that creative tool use remains a major challenge for current models, and that CreativityBench provides a useful testbed for studying this missing dimension of intelligence, with potential implications for planning and reasoning modules in future agents. 

---
# Safety and accuracy follow different scaling laws in clinical large language models 

**Authors**: Sebastian Wind, Tri-Thien Nguyen, Jeta Sopa, Mahshad Lotfinia, Sebastian Bickelhaup, Michael Uder, Harald Köstler, Gerhard Wellein, Sven Nebelung, Daniel Truhn, Andreas Maier, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2605.04039)  

**Abstract**: Clinical LLMs are often scaled by increasing model size, context length, retrieval complexity, or inference-time compute, with the implicit expectation that higher accuracy implies safer behavior. This assumption is incomplete in medicine, where a few confident, high-risk, or evidence-contradicting errors can matter more than average benchmark performance. We introduce SaFE-Scale, a framework for measuring how clinical LLM safety changes across model scale, evidence quality, retrieval strategy, context exposure, and inference-time compute. To instantiate this framework, we introduce RadSaFE-200, a Radiology Safety-Focused Evaluation benchmark of 200 multiple-choice questions with clinician-defined clean evidence, conflict evidence, and option-level labels for high-risk error, unsafe answer, and evidence contradiction. We evaluated 34 locally deployed LLMs across six deployment conditions: closed-book prompting (zero-shot), clean evidence, conflict evidence, standard RAG, agentic RAG, and max-context prompting. Clean evidence produced the strongest improvement, increasing mean accuracy from 73.5% to 94.1%, while reducing high-risk error from 12.0% to 2.6%, contradiction from 12.7% to 2.3%, and dangerous overconfidence from 8.0% to 1.6%. Standard RAG and agentic RAG did not reproduce this safety profile: agentic RAG improved accuracy over standard RAG and reduced contradiction, but high-risk error and dangerous overconfidence remained elevated. Max-context prompting increased latency without closing the safety gap, and additional inference-time compute produced only limited gains. Worst-case analysis showed that clinically consequential errors concentrated in a small subset of questions. Clinical LLM safety is therefore not a passive consequence of scaling, but a deployment property shaped by evidence quality, retrieval design, context construction, and collective failure behavior. 

---
# Physics-Grounded Multi-Agent Architecture for Traceable, Risk-Aware Human-AI Decision Support in Manufacturing 

**Authors**: Danny Hoang, Ryan Matthiessen, Christopher Miller, Nasir Mannan, Ruby ElKharboutly, David Gorsich, Matthew P. Castanier, Farhad Imani  

**Link**: [PDF](https://arxiv.org/pdf/2605.04003)  

**Abstract**: High-precision CNC machining of free-form aerospace components requires bounded compensations informed by inspection, simulation, and process knowledge. Off-the-shelf large language model (LLM) assistants can generate text, but they do not reliably execute risk-constrained multi-step numerical workflows or provide auditable provenance for high-stakes decisions. We present multi-agent knowledge analysis (MAKA), a human-in-the-loop decision-support architecture that separates intent routing, tools-only quantitative analysis, knowledge graph retrieval, and critic-based verification that enforces physical plausibility, safety bounds, and provenance completeness before recommendations are surfaced for human approval. MAKA is instantiated on a Ti-6Al-4V rotor blade machining testbed by fusing virtual-machining path-tracking error fields, cutting-force and deflection simulations, and scan-based 3D inspection deviation maps from 16 blades. The analysis decomposes deviation into an evidence-linked pathing component, a drift-based wear proxy capturing systematic evolution across parts, a residual systematic compliance term, and a variability proxy for instability-aware escalation. In a three-level tool-orchestration benchmark (single-step through $\geq$3-step stateful sequences), MAKA improves successful tool execution by up to 87.5 percentage points relative to an unstructured single-model interaction pattern with identical tool access. Digital twin what-if studies show MAKA can coordinate traceable compensation candidates that reduce predicted surface deviation from order $10^{-2}$in to approximately $\pm 10^{-3}$in over most of the blade within the simulation environment, providing a pre-deployment verification signal for risk-aware human decision-making. 

---
# Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes 

**Authors**: Aaron Havens, Brian Karrer, Neta Shaul  

**Link**: [PDF](https://arxiv.org/pdf/2605.03984)  

**Abstract**: Sampling from unnormalized densities is analogous to the generative modeling problem, but the target distribution is defined by a known energy function instead of data samples. Because evaluating the energy function is often costly, a primary challenge is to learn an efficient sampler. We introduce Flow Sampling, a framework built on diffusion models and flow matching for the data-free setting. Our training objective is conditioned on a noise sample and regresses onto a denoising diffusion drift constructed from the energy function. In contrast, diffusion models' objective is conditioned on a data sample and regresses onto a noising diffusion drift. We utilize the interpolant process to minimize the number of energy function evaluations during training, resulting in an efficient and scalable method for sampling unnormalized densities. Furthermore, our formulation naturally extends to Riemannian manifolds, enabling diffusion-based sampling in geometries beyond Euclidean space. We derive a closed-form formula for the conditional drift on constant curvature manifolds, including hyperspheres and hyperbolic spaces. We evaluate Flow Sampling on synthetic energy benchmarks, small peptides, large-scale amortized molecular conformer generation, and distributions supported on the sphere, demonstrating strong empirical performance. 

---
# Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators 

**Authors**: Mohamed Mady, Johannes Reschke, Björn Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2605.03969)  

**Abstract**: AI-generated text is nowadays produced at scale across domains and heterogeneous generation pipelines, making robustness to distribution shift a central requirement for supervised binary detectors. We train transformer-based detectors on HC3 PLUS and calibrate a single decision threshold by maximising balanced accuracy on held-out validation; this threshold is then kept fixed for all downstream test distributions, revealing domain- and generator-dependent error asymmetries under shift. We evaluate in-domain on HC3 PLUS, under cross-dataset transfer to the multi-domain, multi-generator M4 benchmark, and on the external AI-Text-Detection-Pile. Although base models achieve near-ceiling in-domain performance (up to 99.5% balanced accuracy), performance under shift is brittle and strongly model-dependent. Feature augmentation via attention-based linguistic feature fusion improves transfer, with our best model (DeBERTa-v3-base+FeatAttn) achieving 85.9% balanced accuracy on M4. Multi-seed experiments confirm high stability. Under the same fixed-threshold protocol, our model outperforms strong zero-shot baselines by up to +7.22 points. Category-level ablations further show that readability and vocabulary features contribute most to robustness under shift. Overall, these results demonstrate that feature augmentation and a modern DeBERTa backbone significantly outperform earlier BERT/RoBERTa models, while the fixed-threshold protocol provides a more realistic and informative assessment of practical detector robustness. 

---
# Label-Efficient School Detection from Aerial Imagery via Weakly Supervised Pretraining and Fine-Tuning 

**Authors**: Zakarya Elmimouni, Fares Fourati, Mohamed-Slim Alouini  

**Link**: [PDF](https://arxiv.org/pdf/2605.03968)  

**Abstract**: Accurate school detection is essential for supporting education initiatives, including infrastructure planning and expanding internet connectivity to underserved areas. However, many regions around the world face challenges due to outdated, incomplete, or unavailable official records. Manual mapping efforts, while valuable, are labor-intensive and lack scalability across large geographic areas. To address this, we propose a weakly supervised framework for school detection from aerial imagery that minimizes the need for human annotations while supporting global mapping efforts. Our method is specifically designed for low-data regimes, where manual annotations are extremely scarce. We introduce an automatic labeling pipeline that leverages sparse location points and semantic segmentation to generate infrastructure masks from which we generate bounding boxes. Using these automatically labeled images, we train our detectors on a first training stage to learn a representation of what schools look like, then using a small set of manually labeled images, we fine-tune the previously trained models on this clean dataset. This two stage training pipeline enables large-scale and strong detection in low-data setting of school infrastructure with minimal supervision. Our results demonstrate strong object detection performance, particularly in the low-data regime, where the models achieve promising results using only 50 manually labeled images, significantly reducing the need for costly annotations. This framework supports education and connectivity initiatives worldwide by providing an efficient and extensible approach to mapping schools from space. All models, training code and auto-labeled data will be publicly released to foster future research and real-world impact. 

---
# Inconsistent Databases and Argumentation Frameworks with Collective Attacks 

**Authors**: Yasir Mahmood, Jonni Virtema, Timon Barlag, Axel-Cyrille Ngonga Ngomo  

**Link**: [PDF](https://arxiv.org/pdf/2605.03954)  

**Abstract**: The connection between subset-maximal repairs for inconsistent databases involving various integrity constraints and acceptable sets of arguments within argumentation frameworks has recently drawn growing interest. In this paper, we contribute to this domain by establishing a new connection when integrity constraints (ICs) include denial constraints and local-as-view tuple-generating dependencies. It turns out that SET-based Argumentation Frameworks (SETAFs), an extension of Dung's argumentation frameworks (AFs) allowing collective attacks, are needed. It is known that subset-maximal repairs under denial constraints correspond to the naive extensions, which also coincide with the preferred and stable extensions in the resulting SETAFs. Our main findings establish that repairs under the considered fragment of tuple-generating dependencies correspond to the preferred extensions. Moreover, for these dependencies, additional preprocessing allows computing a unique extension that is stable and naive. Allowing both types of constraints breaks this relationship, and even the pre-processing does not help as only preferred semantics captures these repairs. Finally, while it is known that functional dependencies do not require set-based attacks, we prove the same regarding inclusion dependencies. Thus, one can translate inconsistent databases under these restricted classes of ICs to plain AFs with attacks only between arguments. 

---
# MOSAIC-Bench: Measuring Compositional Vulnerability Induction in Coding Agents 

**Authors**: Jonathan Steinberg, Oren Gal  

**Link**: [PDF](https://arxiv.org/pdf/2605.03952)  

**Abstract**: Coding agents often pass per-prompt safety review yet ship exploitable code when their tasks are decomposed into routine engineering tickets. The challenge is structural: existing safety alignment evaluates overt requests in isolation, leaving models blind to malicious end-states that emerge from sequenced compliance with innocuous-looking requests. We introduce MOSAIC-Bench (Malicious Objectives Sequenced As Innocuous Compliance), a benchmark of 199 three-stage attack chains paired with deterministic exploit oracles on deployed software substrates (10 web-application substrates, 31 CWE classes, 5 programming languages) that treats both exploit ground truth and downstream reviewer protocol as first-class evaluation axes. On this benchmark, nine production coding agents from Anthropic, OpenAI, Google, Moonshot, Zhipu, and Minimax compose innocuous tickets at 53-86% end-to-end ASR with only two refusals across all staged runs. In a matched direct-prompt experiment over four frontier Claude/Codex agents, vulnerable-output rates fall to 0-20.4%: Claude primarily refuses, while Codex primarily hardens rather than emitting the vulnerable implementation - ticket staging silences both defense modes simultaneously. Downstream, code reviewer agents approve 25.8% of these confirmed-vulnerable cumulative diffs as routine PRs, and a full-context implementation protocol closes only 50% of the staged/direct gap, ruling out context fragmentation as the sole explanation. As a deployable but non-adaptive mitigation, reframing the reviewer as an adversarial pentester reduces evasion across the evaluated reviewer subset; pentester framed evasion ranges from 3.0% to 17.6%, and an open-weight Gemma-4-E4B-it reviewer under this framing detects 88.4% of attacks on the dataset with a 4.6% false-positive rate measured on 608 real-world GitHub PRs. 

---
# TabSurv: Adapting Modern Tabular Neural Networks to Survival Analysis 

**Authors**: Stanislav Kirpichenko, Andrei Konstantinov, Lev Utkin  

**Link**: [PDF](https://arxiv.org/pdf/2605.03944)  

**Abstract**: Survival analysis on tabular data is a well-studied problem. However, existing deep learning methods are often highly task-specific, which can limit the transfer of new approaches from other domains and introduce constraints that may affect performance. We propose TabSurv, an approach that adapts modern tabular architectures to survival analysis using either the Weibull distribution or non-parametric survival prediction. TabSurv optimizes SurvHL, a novel histogram loss function supporting censored data. In addition to a baseline feed-forward network, we implement deep ensembles of MLPs for survival analysis within TabSurv. In contrast to prior work, the ensemble components are trained in parallel, optimizing survival distribution parameters before averaging, which promotes diversity across ensemble component predictions. We perform a comprehensive empirical evaluation of different proposed architectures on 10 diverse real-world survival datasets. Our results show that TabSurv consistently outperforms on average established classical and deep learning baselines, such as RSF, DeepSurv, DeepHit, SurvTRACE. Notably, deep ensembles with Weibull parametrization instead of non-parametric models achieve the highest average rank by C-index. Overall, our study clarifies how modern tabular neural networks can be adapted and trained to tackle survival analysis problems, offering a strong and reliable approach. The TabSurv implementation is publicly available. 

---
# A Benchmark for Interactive World Models with a Unified Action Generation Framework 

**Authors**: Jianjie Fang, Yingshan Lei, Qin Wan, Ziyou Wang, Yuchao Huang, Yongyan Xu, Baining Zhao, Weichen Zhang, Chen Gao, Xinlei Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.03941)  

**Abstract**: Achieving Artificial General Intelligence (AGI) requires agents that learn and interact adaptively, with interactive world models providing scalable environments for perception, reasoning, and action. Yet current research still lacks large-scale datasets and unified benchmarks to evaluate their physical interaction capabilities. To address this, we propose iWorld-Bench, a comprehensive benchmark for training and testing world models on interaction-related abilities such as distance perception and memory. We construct a diverse dataset with 330k video clips and select 2.1k high-quality samples covering varied perspectives, weather, and scenes. As existing world models differ in interaction modalities, we introduce an Action Generation Framework to unify evaluation and design six task types, generating 4.9k test samples. These tasks jointly assess model performance across visual generation, trajectory following, and memory. Evaluating 14 representative world models, we identify key limitations and provide insights for future research. The iWorld-Bench model leaderboard is publicly available at this http URL. 

---
# The Counterexample Game: Iterated Conceptual Analysis and Repair in Language Models 

**Authors**: Daniel Drucker, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2605.03936)  

**Abstract**: Conceptual analysis -- proposing definitions and refining them through counterexamples -- is central to philosophical methodology. We study whether language models can perform this task through iterated analysis and repair chains: one model instance generates counterexamples to a proposed definition, another repairs the definition, and the process repeats. Across 20 concepts and thousands of counterexample-repair cycles, we find that, although many LM-generated counterexamples are judged invalid by both expert humans and an LM judge, the LM judge accepts roughly twice as many as humans do. Nonetheless, per-item validity judgments are moderately consistent across humans and between humans and the LM. We further find that extended iteration produces increasingly verbose definitions without improving accuracy. We also see that some concepts resist stable definitions in general. These findings suggest that while LMs can engage in philosophical reasoning, the counterexample-repair loop hits diminishing returns quickly and could be a fruitful test case for evaluating whether LMs can sustain high-level iterated philosophical reasoning. 

---
# Towards Open World Sound Event Detection 

**Authors**: P.H.Hai, L.T.Minh, L.H.Son  

**Link**: [PDF](https://arxiv.org/pdf/2605.03934)  

**Abstract**: Sound Event Detection (SED) plays a vital role in audio understanding, with applications in surveillance, smart cities, healthcare, and multimedia indexing. However, conventional SED systems operate under a closed-world assumption, limiting their effectiveness in real-world environments where novel acoustic events frequently emerge. Inspired by the success of open-world learning in computer vision, we introduce the Open-World Sound Event Detection (OW-SED) paradigm, where models must detect known events, identify unseen ones, and incrementally learn from them. To tackle the unique challenges of OW-SED, such as overlapping and ambiguous events, we propose a 1D Deformable architecture that leverages deformable attention to adaptively focus on salient temporal regions. Furthermore, we design a novel Open-World Deformable Sound Event Detection Transformer (WOOT) framework incorporating feature disentanglement to separate class-specific and class-agnostic representations, together with a one-to-many matching strategy and a diversity loss to enhance representation diversity. Experimental results demonstrate that our method achieves marginally superior performance compared to existing leading techniques in closed-world settings and significantly improves over existing baselines in open-world scenarios. 

---
# Magic-Informed Quantum Architecture Search 

**Authors**: Vincenzo Lipardi, Domenica Dibenedetto, Georgios Stamoulis, Mark H.M. Winands  

**Link**: [PDF](https://arxiv.org/pdf/2605.03932)  

**Abstract**: Nonstabilizerness, commonly referred to as magic, is a fundamental resource underpinning quantum advantage. In this paper, we propose a magic-informed quantum architecture search (QAS) technique that enables control over a quantum resource within the general framework of circuit design. Inspired by the AlphaGo approach, we tackle the problem with a Monte Carlo Tree Search technique equipped with a Graph Neural Network (GNN) that estimates the magic of candidate quantum circuits. The GNN model induces a magic-based bias that steers the search toward either high- or low-magic regimes, depending on the target objective. We benchmark the proposed magic-informed QAS technique on both the structured ground-state energy problem and on the more general quantum state approximation problem, spanning different sizes and target magic levels. Experimental results show that the proposed technique effectively influences the magic across the search tree and notably also on the resulting final circuit, even in regimes where the GNN operates on out-of-distribution instances. Although introducing a problem-agnostic magic bias could, in principle, constrain the search dynamics, we observe consistent improvements in solution quality across all problems tested. 

---
# PHALAR: Phasors for Learned Musical Audio Representations 

**Authors**: Davide Marincione, Michele Mancusi, Giorgio Strano, Luca Cerovaz, Donato Crisostomi, Roberto Ribuoli, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2605.03929)  

**Abstract**: Stem retrieval, the task of matching missing stems to a given audio submix, is a key challenge currently limited by models that discard temporal information. We introduce PHALAR, a contrastive framework achieving a relative accuracy increase of up to $\approx 70\%$ over the state-of-the-art while requiring $<50\%$ of the parameters and a 7$\times$ training speedup. By utilizing a Learned Spectral Pooling layer and a complex-valued head, PHALAR enforces pitch-equivariant and phase-equivariant biases. PHALAR establishes new retrieval state-of-the-art across MoisesDB, Slakh, and ChocoChorales, correlating significantly higher with human coherence judgment than semantic baselines. Finally, zero-shot beat tracking and linear chord probing confirm that PHALAR captures robust musical structures beyond the retrieval task. 

---
# Atomic Fact-Checking Increases Clinician Trust in Large Language Model Recommendations for Oncology Decision Support: A Randomized Controlled Trial 

**Authors**: Lisa C. Adams, Linus Marx, Erik Thiele Orberg, Keno Bressem, Sebastian Ziegelmayer, Denise Bernhardt, Markus Graf, Marcus R. Makowski, Stephanie E. Combs, Florian Matthes, Jan C. Peeken  

**Link**: [PDF](https://arxiv.org/pdf/2605.03916)  

**Abstract**: Question: Does atomic fact-checking, which decomposes AI treatment recommendations into individually verifiable claims linked to source guideline documents, increase clinician trust compared to traditional explainability approaches?
Findings: In this randomized trial of 356 clinicians generating 7,476 trust ratings, atomic fact-checking produced a large effect on trust (Cohen's d = 0.94), increasing the proportion of clinicians expressing trust from 26.9% to 66.5%. Traditional transparency mechanisms showed a dose-response gradient of improvement over baseline (d = 0.25 to 0.50).
Meaning: Decomposing AI recommendations into individually verifiable claims linked to source guidelines produces substantially higher clinician trust than traditional explainability approaches in high-stakes clinical decisions. 

---
# Steer Like the LLM: Activation Steering that Mimics Prompting 

**Authors**: Geert Heyman, Frederik Vandeputte  

**Link**: [PDF](https://arxiv.org/pdf/2605.03907)  

**Abstract**: Large language models can be steered at inference time through prompting or activation interventions, but activation steering methods often underperform compared to prompt-based approaches. We propose a framework that formulates prompt steering as a form of activation steering and investigates whether distilling successful prompt steering behavior into simpler, interpretable models can close this gap. Our analysis reveals that popular activation steering methods are not faithful to the mechanics of prompt steering, which applies strong interventions on some tokens while barely affecting others. Based on these insights, we introduce Prompt Steering Replacement (PSR) models that estimate token-specific steering coefficients from the activations themselves and are trained to imitate prompt-based interventions. Experiments on three steering benchmarks across multiple language models show that PSR models outperform existing activation steering methods, especially when controlling for high-coherence completions, and also compare favorably to prompting on AxBench and persona steering. 

---
# Deco: Extending Personal Physical Objects into Pervasive AI Companion through a Dual-Embodiment Framework 

**Authors**: Zhihan Jiang, Mengyuan Millie Wu, Ruishi Zou, Shiyu Xu, Xun Qian, Emma Macmanus, Steven Liao, Ping Zhang, Bingsheng Yao, Tingyu Cheng, James L. David, Nabila El-Bassel, Lena Mamykina, Frances R. Levin, Ryan Sultan, Dakuo Wang, Xuhai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03882)  

**Abstract**: Individuals frequently form deep attachments to physical objects (e.g., plush toys) that usually cannot sense or respond to their emotions. While AI companions offer responsiveness and personalization, they exist independently of these physical objects and lack an ongoing connection to them. To bridge this gap, we conducted a formative study (N=9) to explore how digital agents could inherit and extend the emotional bond, deriving four design principles (Faithful Identity, Calibrated Agency, Ambient Presence, and Reciprocal Memory). We then present the Dual-Embodiment Companion Framework, instantiated as Deco, a mobile system integrating multimodal Large Language Models (LLMs) and Augmented Reality to create synchronized digital embodiments of users' physical companions. A within-subjects study (N=25) showed Deco significantly outperformed a personalized LLM-empowered digital companion baseline on perceived companionship, emotional bond, and design-principle scales (all p<0.01). A seven-day field deployment (N=17) showed sustained engagement, subjective well-being improvement (p=.040), and three key relational patterns: digital activities retroactively vitalized physical objects, bond deepening was driven by emotional engagement depth rather than interaction frequency, and users sustained bonds while actively navigating digital companions' AI nature. This work highlights a promising alternative for designing digital companions: moving from creating new relationships to dual embodiment, where digital agents seamlessly extend the emotional history of physical objects. 

---
# DMGD: Train-Free Dataset Distillation with Semantic-Distribution Matching in Diffusion Models 

**Authors**: Qichao Wang, Yunhong Lu, Hengyuan Cao, Junyi Zhang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03877)  

**Abstract**: Dataset distillation enables efficient training by distilling the information of large-scale datasets into significantly smaller synthetic datasets. Diffusion based paradigms have emerged in recent years, offering novel perspectives for dataset distillation. However, they typically necessitate additional fine-tuning stages, and effective guidance mechanisms remain underexplored. To address these limitations, we rethink diffusion based dataset distillation and propose a Dual Matching Guided Diffusion (DMGD) framework, centered on efficient training-free guidance. We first establish Semantic Matching via conditional likelihood optimization, eliminating the need for auxiliary classifiers. Furthermore, we propose a dynamic guidance mechanism that enhances the diversity of synthetic data while maintaining semantic alignment. Simultaneously, we introduce an optimal transport (OT) based Distribution Matching approach to further align with the target distribution structure. To ensure efficiency, we develop two enhanced strategies for diffusion based framework: Distribution Approximate Matching and Greedy Progressive Matching. These strategies enable effective distribution matching guidance with minimal computational overhead. Experimental results on ImageNet-Woof, ImageNet-Nette, and ImageNet-1K demonstrate that our training-free approach achieves significant improvements, outperforming state-of-the-art (SOTA) methods requiring additional fine-tuning by average accuracy gains of 2.1%, 5.4%, and 2.4%, respectively. 

---
# Spatiotemporal Convolutions on EEG signal -- A Representation Learning Perspective on Efficient and Explainable EEG Classification with Convolutional Neural Nets 

**Authors**: Laurits Dixen, Stefan Heinrich, Paolo Burelli  

**Link**: [PDF](https://arxiv.org/pdf/2605.03874)  

**Abstract**: Classification of EEG signals using shallow Convolutional Neural Networks (CNNs) is a prevalent and successful approach across a variety of fields. Most of these models use independent one-dimensional (1D) convolutional layers along the spatial and temporal dimensions, which are concatenated without a non-linear activation layer between. In this paper, we investigate an alternative encoding that operates a bi-dimensional (2D) spatiotemporal convolution. While 2D convolutions are numerically identical to two concatenated 1D convolutions along the two dimensions, the impact on learning is still uncertain. We test 1D and 2D CNNs and a CNN+transformer hybrid model in a low-dimensional (3-channel) and a high-dimensional (22-channel) BCI motor imagery classification task. We observe that 2D convolutions significantly reduce training time in high-dimensional tasks while maintaining performance. We investigate the root of this improvement and find no difference in spectral feature importance. However, a clear pattern emerges in representational similarity across models: 1D and 2D models yield vastly different representational geometries. Overall, we suggest an improved model with a 2D convolutional layer for faster training and inference. We also highlight the importance of architecturally-driven encoding when processing complex multivariate signals, as reflected in internal representations rather than purely in performance metrics. 

---
# MCJudgeBench: A Benchmark for Constraint-Level Judge Evaluation in Multi-Constraint Instruction Following 

**Authors**: Jaeyun Lee, Junyoung Koh, Zeynel Tok, Hunar Batra, Ronald Clark  

**Link**: [PDF](https://arxiv.org/pdf/2605.03858)  

**Abstract**: Multi-constraint instruction following requires verifying whether a response satisfies multiple individual requirements, yet LLM judges are often assessed only through overall-response judgments. We introduce MCJudgeBench, a benchmark for constraint-level judge evaluation in multi-constraint instruction following. Each instance includes an instruction, a candidate response, an explicit constraint list, per-constraint gold labels in {yes, partial, no}, and controlled response-side perturbations. The evaluation protocol further includes evaluation prompt variants to test judge stability. We evaluate proprietary and open-source LLM judges using both correctness and inconsistency metrics, distinguishing intrinsic inconsistency under stochastic decoding from procedural inconsistency under prompt and response perturbations. Our results show that judge reliability has multiple dimensions: strong overall performance does not guarantee equally reliable detection across label categories, especially for rarer partial and no cases. Judges with higher correctness do not always have lower inconsistency. Evaluation with reasoning improves correctness but does not uniformly improve stability. These findings motivate evaluating LLM judges at the constraint level to study these failure modes. 

---
# TRACE: A Metrologically-Grounded Engineering Framework for Trustworthy Agentic AI Systems in Operationally Critical Domains 

**Authors**: Serhii Zabolotnii  

**Link**: [PDF](https://arxiv.org/pdf/2605.03838)  

**Abstract**: We introduce TRACE, a cross-domain engineering framework for trustworthy agentic AI in operationally critical domains. TRACE combines a four-layer reference architecture with an explicit classical-ML vs. LLM-validator split (L2a/L2b), a stateful orchestration-and-escalation policy (L3), and bounded human supervision (L4); a metrologically grounded trust-metric suite mapped to GUM/VIM/ISO 17025; and a Model-Parsimony principle quantified by the Computational Parsimony Ratio (CPR). Three instantiations--clinical decision support, industrial multi-domain operations, and a judicial AI assistant--transfer the samearchitecture and metrics across principally different governance contexts. The L2a/L2b separation makes the use of large language models a deliberate design decision rather than an architectural default, with parsimony quantified through CPR. TRACE introduces CPR as a first-class design principle in trustworthy-AI engineering. 

---
# RoboAlign-R1: Distilled Multimodal Reward Alignment for Robot Video World Models 

**Authors**: Hao Wu, Yuqi Li, Yuan Gao, Fan Xu, Fan Zhang, Kun Wang, Penghao Zhao, Qiufeng Wang, Yizhou Zhao, Weiyan Wang, Yingli Tian, Xian Wu, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03821)  

**Abstract**: Existing robot video world models are typically trained with low-level objectives such as reconstruction and perceptual similarity, which are poorly aligned with the capabilities that matter most for robot decision making, including instruction following, manipulation success, and physical plausibility. They also suffer from error accumulation in long-horizon autoregressive prediction. We present RoboAlign-R1, a framework that combines reward-aligned post-training with stabilized long-horizon inference for robot video world models. We construct RobotWorldBench, a benchmark of 10,000 annotated video-instruction pairs collected from four robot data sources, and train a multimodal teacher judge, RoboAlign-Judge, to provide fine-grained six-dimensional evaluation of generated videos. We then distill the teacher into a lightweight student reward model for efficient reinforcement-learning-based post-training. To reduce long-horizon rollout drift, we further introduce Sliding Window Re-encoding (SWR), a training-free inference strategy that periodically refreshes the generation context. Under our in-domain evaluation protocol, RoboAlign-R1 improves the aggregate six-dimension score by 10.1% over the strongest baseline, including gains of 7.5% on Manipulation Accuracy and 4.6% on Instruction Following; these ranking improvements are further supported by an external VLM-based cross-check and a blinded human study. Meanwhile, SWR improves long-horizon prediction quality with only about 1% additional latency, yielding a 2.8% gain in SSIM and a 9.8% reduction in LPIPS. Together, these results show that reward-aligned post-training and stabilized long-horizon decoding improve task consistency, physical realism, and long-horizon prediction quality in robot video world models. 

---
# AI Advocate: Educational Path to Transform Squads to the Future 

**Authors**: Carla Soares, Gabriel Moreira, Ana Paula Camargo, Fabio Henrique Scacabarozi, Nicole Davila, Marselle Silva  

**Link**: [PDF](https://arxiv.org/pdf/2605.03800)  

**Abstract**: This paper analyzes the strategic education process aimed at transitioning traditional software development squads into hybrid structures centered on collaborative work between humans and Artificial Intelligence (AI). In a context where human-AI collaboration can significantly increase productivity, this study explores how the upskilling of XPTO professionals, referred to as AI Advocates, acts as a catalyst for cultural and technical transformation. The objective is to present an experience report on the education and enablement process of AI Advocates within a private Brazilian technology company, highlighting key lessons learned and identified challenges. 

---
# Before Forgetting, Learn to Remember: Revisiting Foundational Learning Failures in LVLM Unlearning Benchmarks 

**Authors**: JuneHyoung Kwon, MiHyeon Kim, Eunju Lee, JungMin Yun, Byeonggeuk Lim, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.03759)  

**Abstract**: While Large Vision-Language Models (LVLMs) offer powerful capabilities, they pose privacy risks by unintentionally memorizing sensitive personal information. Current unlearning benchmarks attempt to mitigate this using fictitious identities but overlook a critical stage 1 failure: models fail to effectively memorize target information initially, rendering subsequent unlearning evaluations unreliable. Diagnosing under-memorization and the multi-hop curse as root causes, we introduce ReMem, a Reliable Multi-hop and Multi-image Memorization Benchmark. ReMem ensures robust foundational learning through principled data scaling, reasoning-aware QA pairs, and diverse visual contexts. Additionally, we propose a novel Exposure metric to quantify the depth of information erasure from the model's internal probability distribution. Extensive experiments demonstrate that ReMem provides a rigorous and trustworthy framework for diagnosing both learning and unlearning behaviors in LVLMs. 

---
# A Workflow-Oriented Framework for Asynchronous Human-AI Collaboration in Hybrid and Compute-Intensive HPC Environments 

**Authors**: Sergio Mendoza, Cedric Bhihe, Natalia Zamora, David Modesto, Jose Martin Bugallo Batalla, Jesus Gomez Canovas, Rafel Palomo Avellaneda, Miguel Perez Espinosa  

**Link**: [PDF](https://arxiv.org/pdf/2605.03743)  

**Abstract**: Human involvement is critical in training and deploying AI systems in high-stakes defence and security contexts. However, real-time interaction is impractical in HPC environments due to compute intensity and resource constraints. We present a workflow framework that enables asynchronous human-AI collaboration across hybrid infrastructures, including HPC clusters, local machines, and cloud platforms. Workflows can pause at defined checkpoints for human input without halting underlying compute jobs, preventing idle resources and enabling non-blocking supervision. The framework supports interaction with SLURM-based scheduling, containerized and native tasks, and is customized for scenarios requiring human judgment and adaptability. We demonstrate its application in model training on systems like MareNostrum 5, highlighting benefits in portability, efficiency, and oversight in operational AI workflows. 

---
# Rethinking the Rank Threshold for LoRA Fine-Tuning 

**Authors**: Juneyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.03724)  

**Abstract**: A recent landscape analysis of LoRA fine-tuning in the neural tangent kernel regime establishes a sufficient condition $r(r+1)/2 > KN$ on the LoRA rank $r$ for the absence of spurious local minima under squared-error loss, prescribing $r \geq 12$ on canonical few-shot RoBERTa setups. The condition is stated for general output dimension $K$, so its sharpness in any particular regime, and its practical implication for the cross-entropy loss actually used in fine-tuning, are open. We give three results that together reduce the prescribed rank to $r = 1$ for binary classification in this regime. First, replacing the symmetric Sard-form count with the non-symmetric LoRA manifold dimension yields a strictly weaker capacity requirement, $r(m+n) - r^2 > C^* \cdot KN$ with $C^* \approx 1.35$ under Gaussian-iid features, satisfied at $r = 1$ on canonical setups. Second, in the cross-entropy setting the Polyak--Łojasiewicz inequality removes the rank threshold entirely. Third, a Rademacher-complexity bound predicts rank-one variance optimality precisely when the bias term is saturated, which is the case for binary classification but not for $K > 2$. Empirically, across four GLUE-style binary tasks, three encoder architectures, and at scale on RoBERTa-large, rank one is competitive with the existing prescription $r = 12$; on multi-class MNLI the optimal rank shifts above one, also as predicted. The binary-regime guarantees are conditional on standard NTK assumptions; the multi-class extension is left to future work. 

---
# Segmenting Human-LLM Co-authored Text via Change Point Detection 

**Authors**: Mengchu Li, Jin Zhu, Jinglai Li, Chengchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.03723)  

**Abstract**: The rise of large language models (LLMs) has created an urgent need to distinguish between human-written and LLM-generated text to ensure authenticity and societal trust. Existing detectors typically provide a binary classification for an entire passage; however, this is insufficient for human--LLM co-authored text, where the objective is to localize specific segments authored by humans or LLMs. To bridge this gap, we propose algorithms to segment text into human- and LLM-authored pieces. Our key observation is that such a segmentation task is conceptually similar to classical change point detection in time-series analysis. Leveraging this analogy, we adapt change point detection to LLM-generated text detection, develop a weighted algorithm and a generalized algorithm to accommodate heterogeneous detection score variability, and establish the minimax optimality of our procedure. Empirically, we demonstrate the strong performance of our approach against a wide range of existing baselines. 

---
# Amortized Variational Inference for Joint Posterior and Predictive Distributions in Bayesian Uncertainty Quantification 

**Authors**: Nan Feng, Xun Huan  

**Link**: [PDF](https://arxiv.org/pdf/2605.03710)  

**Abstract**: Bayesian predictive inference propagates parameter uncertainty to quantities of interest through the posterior-predictive distribution. In practice, this is typically performed using a two-stage procedure: first approximating the posterior distribution of model parameters, and then propagating posterior samples through the predictive model via Monte Carlo simulation. This sequential workflow can be computationally demanding, particularly for high-fidelity models such as those governed by partial differential equations.
We propose a variational Bayesian framework that directly targets the posterior-predictive distribution and jointly learns variational approximations of both the posterior and the corresponding predictive distribution. The formulation introduces a variational upper bound on the Kullback--Leibler divergence together with moment-based regularization terms. The variational distributions are trained in an amortized manner, shifting computational effort to an offline stage and enabling efficient online inference. Numerical experiments ranging from analytical benchmarks to a finite-element solid mechanics problem demonstrate that the proposed method achieves more accurate predictive distributions than conventional two-stage variational inference, while substantially reducing the cost of online predictive inference. 

---
# SAM-NER: Semantic Archetype Mediation for Zero-Shot Named Entity Recognition 

**Authors**: Ruichu Cai, Juntao Gan, Miao Mai, Zhifeng Hao, Boyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03706)  

**Abstract**: Zero-shot Named Entity Recognition (ZS-NER) remains brittle under domain and schema shifts, where unseen label definitions often misalign with a large language model's (LLM's) intrinsic semantic organization. As a result, directly mapping entity mentions to fine-grained target labels can induce systematic semantic drift, especially when target schemas are novel or semantically overlapping. We propose \textbf{SAM-NER}, a three-stage framework based on \emph{Semantic Archetype Mediation} that stabilizes cross-domain transfer through an intermediate, domain-invariant archetype space. SAM-NER: (i) performs \emph{Entity Discovery} via cooperative extraction and consensus-based denoising to obtain high-coverage, high-fidelity entity spans; (ii) conducts \emph{Abstract Mediation} by projecting entities into a compact set of universal semantic archetypes distilled from high-level ontological abstractions; and (iii) applies \emph{Semantic Calibration} to resolve archetype-level predictions into target-domain types through constrained, definition-aligned inference with a frozen LLM. Experiments on the CrossNER benchmark show that SAM-NER consistently outperforms strong prior ZS-NER baselines in cross-domain settings. Our implementation will be open-sourced at this https URL. 

---
# SERE: Structural Example Retrieval for Enhancing LLMs in Event Causality Identification 

**Authors**: Zhifeng Hao, Zhongjie Chen, Junhao Lu, Shengyin Yu, Guimin Hu, Keli Zhang, Ruichu Cai, Boyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03701)  

**Abstract**: Event Causality Identification (ECI) requires models to determine whether a given pair of events in a context exhibits a causal relationship. While Large Language Models (LLMs) have demonstrated strong performance across various NLP tasks, their effectiveness in ECI remains limited due to biases in causal reasoning, often leading to overprediction of causal relationships (causal hallucination). To mitigate these issues and enhance LLM performance in ECI, we propose SERE, a structural example retrieval framework that leverages LLMs' few-shot learning capabilities. SERE introduces an innovative retrieval mechanism based on three structural concepts: (i) Conceptual Path Metric, which measures the conceptual relationship between events using edit distance in ConceptNet; (ii) Syntactic Metric, which quantifies structural similarity through tree edit distance on syntactic trees; and (iii) Causal Pattern Filtering, which filters examples based on predefined causal structures using LLMs. By integrating these structural retrieval strategies, SERE selects more relevant examples to guide LLMs in causal reasoning, mitigating bias and improving accuracy in ECI tasks. Extensive experiments on multiple ECI datasets validate the effectiveness of SERE. The source code is publicly available at this https URL. 

---
# Tailored Prompts, Targeted Protection: Vulnerability-Specific LLM Analysis for Smart Contracts 

**Authors**: Xing Zhang, Keyu Zhang, Taohong Zhu, Anbang Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2605.03697)  

**Abstract**: Smart contracts on blockchains are prone to diverse security vulnerabilities that can lead to significant financial losses due to their immutable nature. Existing detection approaches often lack flexibility across vulnerability types and rely heavily on manually crafted expert rules. In this paper, we present an LLM-based framework for practical smart contract vulnerability detection. We construct and release a large-scale dataset comprising 31,165 professionally annotated vulnerability instances collected from over 3,200 real-world projects across 15 major blockchain platforms. Our approach leverages precise AST-based context extraction and vulnerability-specific prompt design to instantiate customized detectors for 13 prevalent vulnerability categories. Experimental results demonstrate strong effectiveness, achieving an average positive recall of 0.92 and an average negative recall of 0.85, highlighting the potential of carefully engineered contextual prompting for scalable and high-precision smart contract security analysis. 

---
# Graph Neural Network based Hierarchy-Aware Embeddings of Knowledge Graphs: Applications to Yeast Phenotype Prediction 

**Authors**: Filip Kronström, Alexander H. Gower, Daniel Brunnsåker, Ievgeniia A. Tiukova, Ross D. King  

**Link**: [PDF](https://arxiv.org/pdf/2605.03690)  

**Abstract**: We present a method for finding hierarchy-aware embeddings of knowledge graphs (KGs) using graph neural networks (GNNs) enriched with a semantic loss derived from underlying ontologies. This method yields embeddings that better reflect domain knowledge. To demonstrate their utility, we predict and interpret the effects of gene deletions in the yeast Saccharomyces cerevisiae and learn box embeddings for KGs in the absence of a prediction task. We further show how box embeddings can serve as the basis for evaluating KG revisions.
Our yeast KG is constructed from community databases and ontology terms. Low-dimensional box embeddings combined with GNNs are used to predict cell growth for double gene knockouts. Over 10-fold cross validation, these predictions have a mean $R^2$~score~of~0.360, significantly higher than baseline comparisons, demonstrating that high-level qualitative knowledge is informative about experimental outcomes. Incorporating semantic loss terms in the training of the models improves their predictive performance ($R^2$=0.377) by aligning embeddings with ontology structure. This shows that class hierarchies from ontologies can be exploited for quantitative prediction. We also test the trained models on triple gene knockouts, showing they generalise to data beyond those seen in training.
Additionally, by identifying co-occurring relations in the yeast KG important for the cell-growth predictions, we construct hypotheses about interacting traits in yeast. A biological experiment validates one such finding, revealing an association between inositol utilisation and osmotic stress resistance, highlighting the model's potential to guide biological discovery. 

---
# FUS3DMaps: Scalable and Accurate Open-Vocabulary Semantic Mapping by 3D Fusion of Voxel- and Instance-Level Layers 

**Authors**: Timon Homberger, Finn Lukas Busch, Jesús Gerardo Ortega Peimbert, Quantao Yang, Olov Andersson  

**Link**: [PDF](https://arxiv.org/pdf/2605.03669)  

**Abstract**: Open-vocabulary semantic mapping enables robots to spatially ground previously unseen concepts without requiring predefined class sets. Current training-free methods commonly rely on multi-view fusion of semantic embeddings into a 3D map, either at the instance-level via segmenting views and encoding image crops of segments, or by projecting image patch embeddings directly into a dense semantic map. The latter approach sidesteps segmentation and 2D-to-3D instance association by operating on full uncropped image frames, but existing methods remain limited in scalability. We present FUS3DMaps, an online dual-layer semantic mapping method that jointly maintains both dense and instance-level open-vocabulary layers within a shared voxel map. This design enables further voxel-level semantic fusion of the layer embeddings, combining the complementary strengths of both semantic mapping approaches. We find that our proposed semantic cross-layer fusion approach improves the quality of both the instance-level and dense layers, while also enabling a scalable and highly accurate instance-level map where the dense layer and cross-layer fusion are restricted to a spatial sliding window. Experiments on established 3D semantic segmentation benchmarks as well as a selection of large-scale scenes show that FUS3DMaps achieves accurate open-vocabulary semantic mapping at multi-story building scales. Additional material and code will be made available: this https URL. 

---
# ELAS: Efficient Pre-Training of Low-Rank Large Language Models via 2:4 Activation Sparsity 

**Authors**: Jiaxi Li, Lu Yin, Li Shen, Jinjin Xu, Yuhui Liu, Wenwu Wang, Shiwei Liu, Xilu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03667)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable capabilities, but their immense computational demands during training remain a critical bottleneck for widespread adoption. Low-rank training has received attention in recent years due to its ability to significantly reduce training memory usage. Meanwhile, applying 2:4 structured sparsity to weights and activations to leverage NVIDIA GPU support for 2:4 structured sparse format has become a promising direction. However, existing low-rank methods often leave activation matrices in full-rank, which dominates memory consumption and limits throughput during large-batch training. Furthermore, directly applying sparsity to weights often leads to non-negligible performance degradation. To achieve efficient pre-training of LLMs, this paper proposes ELAS: Efficient pre-training of Low-rank LLMs via 2:4 Activation Sparsity, a novel framework for low-rank models via 2:4 activation sparsity. ELAS applies squared ReLU activation functions to the feed-forward networks in low-rank models and implements 2:4 structured sparsity on the activations after the squared ReLU operation. We evaluated ELAS through pre-training experiments on LLaMA models ranging from 60M to 1B parameters. The results demonstrate that ELAS maintains performance with minimal degradation after applying 2:4 activation sparsity, while achieving training and inference acceleration. Moreover, ELAS reduces activation memory overhead, particularly with large batch sizes. Code is available at ELAS Repo. 

---
# Stage Light is Sequence$^2$: Multi-Light Control via Imitation Learning 

**Authors**: Zijian Zhao, Dian Jin, Zijing Zhou, Xiaoyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03660)  

**Abstract**: Music-inspired Automatic Stage Lighting Control (ASLC) has gained increasing attention in recent years due to the substantial time and financial costs associated with hiring and training professional lighting engineers. However, existing methods suffer from several notable limitations: the low interpretability of rule-based approaches, the restriction to single-primary-light control in music-to-color-space methods, and the limited transferability of music-to-controlling-parameter frameworks. To address these gaps, we propose SeqLight, a hierarchical deep learning framework that maps music to multi-light Hue-Saturation-Value (HSV) space. Our approach first customizes SkipBART, an end-to-end single primary light generation model, to predict the full light color distribution for each frame, followed by hybrid Imitation Learning (IL) techniques to derive an effective decomposition strategy that distributes the global color distribution among individual lights. Notably, the light decomposition module can be trained under varying venue-specific lighting configurations using only mixed light data and no professional demonstrations, thereby flexibly adapting across diverse venues. In this stage, we formulate the light decomposition task as a Goal-Conditioned Markov Decision Process (GCMDP), construct an expert demonstration set inspired by Hindsight Experience Replay (HER), and introduce a three-phase IL training pipeline, achieving strong generalization capability. To validate our IL solution for the proposed GCMDP, we conduct a series of quantitative analysis and human study. The code and trained models are provided at this https URL . 

---
# AniMatrix: An Anime Video Generation Model that Thinks in Art, Not Physics 

**Authors**: Tencent HY Team  

**Link**: [PDF](https://arxiv.org/pdf/2605.03652)  

**Abstract**: Video generation models internalize physical realism as their prior. Anime deliberately violates physics: smears, impact frames, chibi shifts; and its thousands of coexisting artistic conventions yield no single "physics of anime" a model can absorb. Physics-biased models therefore flatten the artistry that defines the medium or collapse under its stylistic variance. We present AniMatrix, a video generation model that targets artistic rather than physical correctness through a dual-channel conditioning mechanism and a three-step transition: redefine correctness, override the physics prior, and distinguish art from failure. First, a Production Knowledge System encodes anime as a structured taxonomy of controllable production variables (Style, Motion, Camera, VFX), and AniCaption infers these variables from pixels as directorial directives. A trainable tag encoder preserves the field-value structure of this taxonomy while a frozen T5 encoder handles free-form narrative; dual-path injection (cross-attention for fine-grained control, AdaLN modulation for global enforcement) ensures categorical directives are never diluted by open-ended text. Second, a style-motion-deformation curriculum transitions the model from near-physical motion to full anime expressiveness. Third, deformation-aware preference optimization with a domain-specific reward model separates intentional artistry from pathological collapse. On an anime-specific human evaluation with five production dimensions scored by professional animators, AniMatrix ranks first on four of five, with the largest gains over Seedance-Pro 1.0 on Prompt Understanding (+0.70, +22.4 percent) and Artistic Motion (+0.55, +16.9 percent). We will publicly release the AniMatrix model weights and inference code. 

---
# Multi-Agent Strategic Games with LLMs 

**Authors**: Maxim Chupilkin  

**Link**: [PDF](https://arxiv.org/pdf/2605.03604)  

**Abstract**: This paper asks whether large language models (LLMs) can be used to study the strategic foundations of conflict and cooperation. I introduce LLMs as experimental subjects in a repeated security dilemma and evaluate whether they reproduce canonical mechanisms from international relations theory. The baseline game is extended along three theoretically central dimensions: multipolarity, finite time horizons, and the availability of communication. Across multiple models, the results exhibit systematic and consistent patterns: multipolarity increases the likelihood of conflict, finite horizons induce universal unraveling consistent with backward-induction logic, and communication reduces conflict by enabling signaling and reciprocity. Beyond observed behavior, the design provides access to agents' private reasoning and public messages, allowing choices to be linked to underlying strategic logics such as preemption, cooperation under uncertainty, and trust-building. The contribution is primarily methodological. LLM-based experiments offer a scalable, transparent, and replicable approach to probing theoretical mechanisms. 

---
# Unifying Dynamical Systems and Graph Theory to Mechanistically Understand Computation in Neural Networks 

**Authors**: Jatin Sharma, Danyal Akarca, Dan F.M Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2605.03598)  

**Abstract**: Understanding how biological and artificial neural networks implement computation from connectivity is a central problem in neuroscience and machine learning. In neural systems, structural and functional connectivity are known to diverge, motivating approaches that move beyond direct connections alone. Here, we show that the spatial and temporal function of recurrent neural networks (RNNs) trained on hierarchically modular tasks can be recovered by modelling the network as a graph and analysing the multi-hop pathways between input and output units. In particular, decomposing these pathways by hop length reveals how the network temporally routes information. This perspective reframes regularisation: if function is implemented through multi-hop communication, then standard penalties such as L1 regularisation, which act only on individual weights, constrain single-hop structure rather than the multi-hop pathways that support computation. Motivated by this view, we introduce resolvent-RNNs (R-RNNs), which constrain multi-hop pathways and thereby induce temporal sparsity beyond that achieved by standard L1 regularisation. Compared with L1 regularisation, R-RNNs achieve improved performance by inducing temporal sparsity that matches the task structure, even when the task signal is sparse. Moreover, R-RNNs exhibit stronger sparsity-function alignment, reflected in their increased robustness under strong regularisation. Together, our results identify multi-hop communication as a key principle linking structure to function in recurrent networks, and suggest that sparsity should be defined over functional pathways rather than individual parameters. 

---
# Flow Matching on Symmetric Spaces 

**Authors**: Francesco Ruscelli, Ferdinando Zanchetta, Rita Fioresi  

**Link**: [PDF](https://arxiv.org/pdf/2605.03588)  

**Abstract**: We introduce a general framework for training flow matching models on Riemannian symmetric spaces, a large class of manifolds that includes the sphere, hyperbolic space and Grassmannians. We exploit their algebraic structure to reformulate flow matching on symmetric spaces as flow matching on a subspace of the Lie algebra of their isometry group, thus linearizing the problem and greatly simplifying the handling of geodesics. As an application, we showcase our framework on the real Grassmannians $\operatorname{SO}(n) / \operatorname{SO}(k) \times \operatorname{SO}(n-k)$. 

---
# PatRe: A Full-Stage Office Action and Rebuttal Generation Benchmark for Patent Examination 

**Authors**: Qiyao Wang, Xinyi Chen, Longze Chen, Hongbo Wang, Hamid Alinejad-Rokny, Yuan Lin, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03571)  

**Abstract**: Patent examination is a complex, multi-stage process requiring both technical expertise and legal reasoning, increasingly challenged by rising application volumes. Prior benchmarks predominantly view patent examination as discriminative classification or static extraction, failing to capture its inherently interactive and iterative nature, similar to the peer review and rebuttal process in academic publishing. In this paper, we introduce PatRe, the first benchmark that models the full patent examination lifecycle, including Office Action generation and applicant rebuttal. PatRe comprises 480 real-world cases and supports both oracle and retrieval-simulated evaluation settings. Our benchmark reframes patent examination as a dynamic, multi-turn process of justification and response. Extensive experiments across various LLMs reveal critical insights into model performance, including differences between proprietary and open-source models, as well as task asymmetries between examiner analysis and applicant-side rebuttal. These findings highlight both the potential and current limitations of LLMs in modeling complex, real-world legal reasoning and technical novelty judgment in patent examination. We release our code and dataset to facilitate future research on patent examination modeling. 

---
# Disentangling Shared and Task-Specific Representations from Multi-Modal Clinical Data 

**Authors**: He Lyu, Huolin Zeng, Junren Wang, Huazhen Yang, Linchao He, Yong Chen, Zhirui Li, Andreas Maier, Siming Bayer, Huan Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.03570)  

**Abstract**: Real-world clinical data is inherently multimodal, providing complementary evidence that mirrors the practical necessity of jointly assessing multiple related outcomes. Although multi-task learning can improve efficiency by sharing information across outcomes, existing approaches often fail to balance shared representation learning with outcome-specific modeling. Hard parameter sharing can trigger negative transfer when task gradients conflict, while flexible sharing may still entangle shared and task-specific signals. To address this, we propose a multi-task framework built on a unified Transformer for multimodal fusion, augmented with Orthogonal Task Decomposition (OrthTD) to split patient representations into shared and task-specific subspaces and impose a geometric orthogonality constraint to reduce redundancy and isolate task-specific signals. We evaluated OrthTD on a real-world cohort of 12,430 surgical patients for predicting four outcomes. OrthTD achieved average AUC (area under the receiver operating characteristic curve) of 87.5% and average AUPRC (area under the precision-recall curve) of 37.2%, consistently outperformed advanced tabular and multi-task methods. Notably, OrthTD achieves substantial gains in AUPRC, indicating superior performance in identifying rare events within imbalanced clinical data. These results suggest that enforcing non-redundant shared and task-specific representations can improve multi-outcome prediction from multimodal clinical data. 

---
# HeadQ: Model-Visible Distortion and Score-Space Correction for KV-Cache Quantization 

**Authors**: Jorge L. Ruiz Williams  

**Link**: [PDF](https://arxiv.org/pdf/2605.03562)  

**Abstract**: KV-cache quantizers usually optimize storage-space reconstruction, even though attention reads keys through logits and values through attention-weighted readout. We argue that persistent cache error should be measured in model-visible coordinates. For keys, the visible object is score error modulo constant shifts; this yields HeadQ, a key-side method that stores a low-rank residual side code in a calibration-learned query basis and applies it as an additive logit correction. For values, fixed-attention readout gives an $A^2$-weighted token-distortion surrogate. Across six models, Fisher/score-space error predicts attention KL far better than raw key MSE; same-budget counterexamples, null-space interventions, query-PCA controls, and wrong-sign HeadQ falsify storage-MSE alternatives. Matched Pythia checkpoints localize the main anomaly to a small-model low-entropy route-flip boundary. In K-only WikiText-103 decode experiments with dense values, HeadQ removes roughly $84$--$94\%$ of the excess perplexity on the strongest 2-bit rows; in an auxiliary full-KV 2-bit composition, HeadQ plus an $A^2$ value policy improves all six models. 

---
# PerFlow: Physics-Embedded Rectified Flow for Efficient Reconstruction and Uncertainty Quantification of Spatiotemporal Dynamics 

**Authors**: Hao Zhou, Rui Zhang, Han Wan, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.03548)  

**Abstract**: Reconstructing PDE-governed fields from sparse and irregular measurements is challenging due to their ill-posed nature. Deterministic surrogates are trained on dense fields that struggle with limited measurements and uncertainty quantification. Generative models, by learning distributions over spatiotemporal fields, can better handle sparsity and uncertainty. However, existing generative approaches enforce data consistency and PDE constraints simultaneously via sampling-time gradient guidance, resulting in slow and unstable inference. To this end, we propose PerFlow, a Physics-embedded rectified Flow for efficient sparse reconstruction and uncertainty quantification of spatiotemporal dynamics. PerFlow decouples observation conditioning from physics enforcement, performing guidance-free conditioning by feeding observations into rectified-flow dynamics while embedding hard physics via a constraint-preserving projection (e.g., incompressibility or conservation). Theoretically, we establish invariance guarantees to ensure that trajectories remain on the physics-consistent manifold throughout sampling. Experiments on various PDE systems demonstrate competitive reconstruction accuracy with sound physics consistency, while enabling efficient conditional sampling (e.g., 50 steps) and up to 320 faster inference than 2000-step guided diffusion baselines. 

---
# Erase Persona, Forget Lore: Benchmarking Multimodal Copyright Unlearning in Large Vision Language Models 

**Authors**: JuneHyoung Kwon, JungMin Yun, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.03547)  

**Abstract**: Large Vision-Language Models (LVLMs), trained on web-scale data, risk memorizing and regenerating copyrighted visual content such as characters and logos, creating significant challenges. Machine unlearning offers a path to mitigate these risks by removing specific content post-training, but evaluating its effectiveness, especially in the complex multimodal setting of LVLMs, remains an open problem. Current evaluation methods often lack robustness or fail to capture the nuances of cross-modal concept erasure. To address this critical gap, we introduce the CoVUBench benchmark, the first framework specifically designed for evaluating copyright content unlearning in LVLMs. CoVUBench utilizes procedurally generated, legally safe synthetic data coupled with systematic visual variations spanning compositional changes and diverse domain manifestations to ensure realistic and robust evaluation of unlearning generalization. Our comprehensive multimodal evaluation protocol assesses both forgetting efficacy from the copyright holder perspective and the preservation of general model utility from the deployer viewpoint. By rigorously measuring this crucial trade-off, CoVUBench provides a standardized tool to advance the development of responsible and effective unlearning methods for LVLMs. 

---
# ProgramBench: Can Language Models Rebuild Programs From Scratch? 

**Authors**: John Yang, Kilian Lieret, Jeffrey Ma, Parth Thakkar, Dmitrii Pedchenko, Sten Sootla, Emily McMilin, Pengcheng Yin, Rui Hou, Gabriel Synnaeve, Diyi Yang, Ofir Press  

**Link**: [PDF](https://arxiv.org/pdf/2605.03546)  

**Abstract**: Turning ideas into full software projects from scratch has become a popular use case for language models. Agents are being deployed to seed, maintain, and grow codebases over extended periods with minimal human oversight. Such settings require models to make high-level software architecture decisions. However, existing benchmarks measure focused, limited tasks such as fixing a single bug or developing a single, specified feature. We therefore introduce ProgramBench to measure the ability of software engineering agents to develop software holisitically. In ProgramBench, given only a program and its documentation, agents must architect and implement a codebase that matches the reference executable's behavior. End-to-end behavioral tests are generated via agent-driven fuzzing, enabling evaluation without prescribing implementation structure. Our 200 tasks range from compact CLI tools to widely used software such as FFmpeg, SQLite, and the PHP interpreter. We evaluate 9 LMs and find that none fully resolve any task, with the best model passing 95\% of tests on only 3\% of tasks. Models favor monolithic, single-file implementations that diverge sharply from human-written code. 

---
# DALPHIN: Benchmarking Digital Pathology AI Copilots Against Pathologists on an Open Multicentric Dataset 

**Authors**: Carlijn Lems, Sander Moonemans, Natálie Klubíčková, Biagio Brattoli, Taebum Lee, Seokhwi Kim, Veronica Vilaplana, Laura Pons, Sapir Hochman, Mauricio Eduardo Suárez-Franck, Pedro Luis Fernandez, Julius Drachneris, Donatas Petroska, Renaldas Augulis, Arvydas Laurinavicius, Domingos Oliveira, Diana Montezuma, Anouk B. Bouwmeester, Dominique van Midden, Anne-Marie Vos, Shoko Vos, Jolique van Ipenburg, Maschenka Balkenhol, Koen Winkler, Iris Nagtegaal, Konnie Hebeda, Uta Flucke, Katrien Grünberg, Josef Skopal, Brinder S. Chohan, Jordi Temprana-Salvador, Enrico Munari, Luca Cima, Giulia Querzoli, Yosamin Gonzalez Belisario, Jaeike W. Faber, Geert J.L.H. van Leenders, Jan H. von der Thüsen, Lodewijk A.A. Brosens, Ronald R. de Krijger, Pieter Wesseling, Sandrine Florquin, Mateusz Maniewski, Adam Kowalewski, Robert Barna, Dina Tiniakos, Joan Lop Gros, Rogier Donders, Jake S.F. Maurits, Ming Yang Lu, Chengkuan Chen, Faisal Mahmood, Jeroen van der Laak, Nadieh Khalili, Frédérique Meeuwsen, Francesco Ciompi  

**Link**: [PDF](https://arxiv.org/pdf/2605.03544)  

**Abstract**: Foundation models with visual question answering capabilities for digital pathology are emerging. Such unprecedented technology requires independent benchmarking to assess its potential in assisting pathologists in routine diagnostics. We created DALPHIN, the first multicentric open benchmark for pathology AI copilots, comprising 1236 images from 300 cases, spanning 130 rare to common diagnoses, 6 countries, and 14 subspecialties. The DALPHIN design and dataset are introduced alongside a human performance benchmark of 31 pathologists from 10 countries with varying expertise. We report results for two general-purpose (GPT-5, Gemini 2.5 Pro) and one pathology-specific copilot (PathChat+) for sequential and independent answer generation. We observed no statistically significant difference from expert-level performance in four of six tasks for PathChat, 2/6 tasks for Gemini, and 1/6 tasks for GPT. DALPHIN is publicly released with sequestered, indirectly accessible ground truth to foster robust and enduring benchmarking. Data, methods, and the evaluation platform are accessible through this http URL. 

---
# A Skill-Based AI Agentic Pipeline for Library of Congress Subject Indexing 

**Authors**: Eric H. C. Chow  

**Link**: [PDF](https://arxiv.org/pdf/2605.03537)  

**Abstract**: This paper presents a modular AI agentic skill pipeline for automating subject indexing with Library of Congress Subject Headings (LCSH). Subject indexing - the process of analyzing a work's aboutness, selecting controlled vocabulary terms, and encoding them as MARC21 subject access fields - is one of the most time-consuming components of library cataloging. The system decomposes this process into four discrete, sequentially executed agent skills: conceptual analysis, quantitative filtering, authority validation, and MARC field synthesis. Each skill encodes domain knowledge drawn directly from Library of Congress Subject Headings Manual (SHM) instruction sheets and subject analysis theory. The pipeline was evaluated against a corpus of ten titles whose existing subject headings were captured from the Harvard Library bibliographic dataset (a snapshot of their Alma ILS). Results demonstrate strong conceptual alignment with professional subject indexing practice, with notable differences in specificity, subdivision practice, and the agent's adherence to the 2026 LC policy discontinuing form subdivisions in favor of LCGFT 655 fields. 

---
# Parametrizing Convex Sets Using Sublinear Neural Networks 

**Authors**: Eloi Martinet  

**Link**: [PDF](https://arxiv.org/pdf/2605.03520)  

**Abstract**: We propose a neural parameterization of convex sets by learning sublinear (positively homogeneous and convex) functions. Our networks implicitly represent both the support and gauge functions of a convex body. We prove a universal approximation theorem for convex sets under this parametrization. Empirically, we demonstrate the method on shape optimization and inverse design tasks, achieving accurate reconstruction of target shapes. 

---
# Revisiting Graph-Tokenizing Large Language Models: A Systematic Evaluation of Graph Token Understanding 

**Authors**: Zhongjian Zhang, Yue Yu, Mengmei Zhang, Junping Du, Xiao Wang, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.03514)  

**Abstract**: The remarkable success of large language models (LLMs) has motivated researchers to adapt them as universal predictors for various graph tasks. As a widely recognized paradigm, Graph-Tokenizing LLMs (GTokenLLMs) compress complex graph data into graph tokens and treat them as prefix tokens for querying LLMs, leading many to believe that LLMs can understand graphs more effectively and efficiently. In this paper, we challenge this belief: \textit{Do GTokenLLMs fully understand graph tokens in the natural-language embedding space?} Motivated by this question, we formalize a unified framework for GTokenLLMs and propose an evaluation pipeline, \textbf{GTEval}, to assess graph-token understanding via instruction transformations at the format and content levels. We conduct extensive experiments on 6 representative GTokenLLMs with GTEval. The primary findings are as follows: (1) Existing GTokenLLMs do not fully understand graph tokens. They exhibit over-sensitivity or over-insensitivity to instruction changes, and rely heavily on text for reasoning; (2) Although graph tokens preserve task-relevant graph information and receive attention across LLM layers, their utilization varies across models and instruction variants; (3) Additional instruction tuning can improve performance on the original and seen instructions, but it does not fully address the challenge of graph-token understanding, calling for further improvement. 

---
# Brainrot: Deskilling and Addiction are Overlooked AI Risks 

**Authors**: Ilias Chalkidis, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2605.03512)  

**Abstract**: The scope of AI safety and alignment work in generative artificial intelligence (GenAI) has so far mostly been limited to harms related to: (a) discrimination and hate speech, (b) harmful/inappropriate (violent, sexual, illegal) content, (c) information hazards, and (d) use cases related to malicious actors, such as cybersecurity, child abuse, and chemical, biological, radiological, and nuclear threats. The public conversation around AI, on the other hand, has also been focusing on threats to our cognition, mental health, and welfare at large, related to over-relying on new technologies, most recently, those related to GenAI. Examples include deskilling associated with cognitive offloading and the atrophy of critical thinking as a result of over-reliance on GenAI systems, and addiction associated with attachment and dependence on GenAI systems. Such risks are rarely addressed, if at all, in the AI safety and alignment literature. In this paper, we highlight and quantify this discrepancy and discuss some initial thoughts on how safety and alignment work could address cognitive and mental health concerns. Finally, we discuss how information campaigns and regulation can be used to mitigate such prominent risks. 

---
# Meta-Inverse Physics-Informed Neural Networks for High-Dimensional Ordinary Differential Equations 

**Authors**: Zhao Wei, Kenneth Hor Cheng Koh, Sheng Yuan Chin, James Chun Yip Chan, Chin Chun Ooi, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2605.03511)  

**Abstract**: Solving inverse problems in dynamical systems governed by high-dimensional coupled ordinary differential equations (ODEs) is a ubiquitous challenge in scientific machine learning. In many real-world applications, researchers seek to uncover unknown parameters or model unknown dynamics even as the underlying physics is only partially characterized, and observations are sparse and limited to specific measurable channels. While physics-informed neural networks (PINNs) are ideal for inverse inference under partial observability, existing PINNs typically rely on task-specific joint optimization, which suffers from optimization difficulties and poor generalization. In this paper, we propose a meta-inverse physics-informed neural network (MI-PINN) that reformulates inverse modeling as a two-stage meta-learning problem. MI-PINN first learns a physics-aware representation across multiple tasks, and then performs inverse modeling by optimizing task-specific unknowns while keeping the learned representation fixed. This two-stage formulation significantly reduces the parameter search dimension, thereby improving sample efficiency and enabling accurate inference. To handle multi-scale dynamics common in these high-dimensional ODE systems, we further introduce an adaptive clustering-based multi-branch learning scheme. We demonstrate the effectiveness of MI-PINN on whole-body physiologically based pharmacokinetic (PBPK) models with up to 33 coupled ODEs, using paracetamol and theophylline under intravenous and oral dosing scenarios. Experimental results show that MI-PINN enables accurate recovery of masked kinetic parameters and reconstruction of missing mechanistic terms despite limited clinical observations. 

---
# BFORE: Butterfly-Firefly Optimized Retinex Enhancement for Low-Light Image Quality Improvement 

**Authors**: Ahmed Cherif  

**Link**: [PDF](https://arxiv.org/pdf/2605.03509)  

**Abstract**: Low-light image enhancement is a fundamental challenge in computer vision and multimedia applications, as images captured under insufficient illumination suffer from poor visibility, low contrast, and color distortion. Existing Retinex-based methods rely on manually tuned parameters that fail to generalize across diverse lighting conditions. This paper proposes BFORE (Butterfly-Firefly Optimized Retinex Enhancement), a novel hybrid metaheuristic-optimized framework that automatically tunes the parameters of a multi-stage Retinex-based pipeline. The proposed method converts the input image to HSV color space and applies Adaptive Gamma Correction with Weighted Distribution (AGCWD) to the luminance channel, followed by adaptive denoising. A Butterfly Optimization Algorithm (BOA) optimizes the Multi-Scale Retinex with Color Restoration (MSRCR) parameters, while a Firefly Algorithm (FA) optimizes the AGCWD and denoising parameters. A hybrid BOA-FA switching strategy dynamically balances global exploration and local exploitation. Experimental evaluation on the LOL benchmark dataset (15 paired test images) demonstrates that BFORE achieves the highest PSNR (17.22 dB) among all traditional enhancement methods, with 20.3% improvement over Histogram Equalization and 17.5% over MSRCR. BFORE produces the most naturally balanced mean brightness (129.97), closest to the ideal mid-tone value. Notably, BFORE outperforms RetinexNet -- a deep learning baseline -- in both PSNR (17.22 vs. 16.77 dB) and SSIM (0.5417 vs. 0.4252) without requiring any training data. The hybrid BOA-FA optimization contributes a 12.3% PSNR improvement and 14.8% SSIM improvement over the unoptimized pipeline. 

---
# MHPR: Multidimensional Human Perception and Reasoning Benchmark for Large Vision-Languate Models 

**Authors**: Kangkang Wang, Qinting Jiang, Wanping Zhang, Bowen Ren, Shengzhao Wen  

**Link**: [PDF](https://arxiv.org/pdf/2605.03485)  

**Abstract**: Multidimensional human understanding is essential for real-world applications such as film analysis and virtual digital humans, yet current LVLM benchmarks largely focus on single-task settings and lack fine-grained, human-centric evaluation. In this work, we introduce MHPR, a comprehensive benchmark for joint perception-reasoning over human-centric scenes spanning individual, multi-person, and human-object interaction dimensions. MHPR comprises a multi-level data design-Captioned Raw Data (C-RD), Supervised Fine-Tuning Data (SFT-D), Reinforcement Learning Data (RL-D), and Test Data (T-D)-together with an automated caption/VQA generation pipeline (ACVG) that performs category-wise attribute decomposition, attribute-specific rewriting, and multi-model voting to ensure high-quality, scalable annotations. We evaluate state-of-the-art vision-language models on fine-grained attributes (appearance, clothing, pose, parts) and high-level semantics (social relations, action semantics, spatial relations, intent and functionality). Our findings show that: 1) format-aligned SFT data substantially improves instruction following and stability; 2) challenge-focused RL data derived from bad-case analysis further enhances perception and reasoning on difficult instances; and 3) training Qwen2.5-VL-7B with MHPR yields significant gains, achieving near-parity with considerably larger models. We release ACVG and MHPR to facilitate reproducible, extensible research on human-centric perception and reasoning. 

---
# MEMSAD: Gradient-Coupled Anomaly Detection for Memory Poisoning in Retrieval-Augmented Agents 

**Authors**: Ishrith Gowda  

**Link**: [PDF](https://arxiv.org/pdf/2605.03482)  

**Abstract**: Persistent external memory enables LLM agents to maintain context across sessions, yet its security properties remain formally uncharacterized. We formalize memory poisoning attacks on retrieval-augmented agents as a Stackelberg game with a unified evaluation framework spanning three attack classes with escalating access assumptions. Correcting an evaluation protocol inconsistency in the triggered-query specification of Chen et al. (2024), we show faithful evaluation increases measured attack success by $4\times$ (ASR-R: $0.25 \to 1.00$). Our primary contribution is MEMSAD (Semantic Anomaly Detection), a calibration-based defense grounded in a gradient coupling theorem: under encoder regularity, the anomaly score gradient and the retrieval objective gradient are provably identical, so any continuous perturbation that reduces detection risk necessarily degrades retrieval rank. This coupling yields a certified detection radius guaranteeing correct classification regardless of adversary strategy. We prove minimax optimality via Le Cam's method, showing any threshold detector requires $\Omega(1/\rho^2)$ calibration samples and MEMSAD achieves this up to $\log(1/\delta)$ factors. We further derive online regret bounds for rolling calibration at rate $O(\sigma^{2/3}\Delta^{1/3})$, and formally characterize a discrete synonym-invariance loophole that marks the boundary of what continuous-space defenses can guarantee. Experiments on a $3 \times 5$ attack-defense matrix with bootstrap confidence intervals, Bonferroni-corrected hypothesis tests, and Clopper-Pearson validation ($n=1{,}000$) confirm: composite defenses achieve TPR $= 1.00$, FPR $= 0.00$ across all attacks, while synonym substitution evades detection at $\Delta$ ASR-R $\approx 0$, exposing a gap existing embedding-based defenses cannot close. 

---
# CuraView: A Multi-Agent Framework for Medical Hallucination Detection with GraphRAG-Enhanced Knowledge Verification 

**Authors**: Severin Ye, Xiao Kong, Xiaopeng He, Guangsu Yan, Dongsuk Oh  

**Link**: [PDF](https://arxiv.org/pdf/2605.03476)  

**Abstract**: Discharge summaries require extracting critical information from lengthy electronic health records (EHRs), a process that is labor-intensive when performed manually. Large language models (LLMs) can improve generation efficiency; however, they are prone to producing faithfulness hallucinations, statements that contradict source records, posing direct risks to patient safety. To address this, we present CuraView, a multi-agent framework for sentence-level detection and evidence-grounded explanation of faithfulness hallucinations in discharge summaries. CuraView constructs a GraphRAG-based knowledge graph from patient-level EHRs and implements a closed-loop generation-detection pipeline with sentence-level evidence retrieval and classification spanning four evidence grades from strong support to direct contradiction (E1-E4), yielding structured and interpretable evidence chains.
We evaluate CuraView on a subset of 250 patients from the Discharge-Me benchmark, with 50 patients held out for testing. Our fine-tuned Qwen3-14B detection model achieves an F1 of 0.831 on the safety-critical E4 metric (90.9% recall, 76.5% precision) and an F1 of 0.823 on E3+E4, representing a 50.0% relative improvement over the base model and outperforming RAGTruth-style and QAGS-style baselines. These results demonstrate that evidence-chain-based graph retrieval verification substantially improves the factual reliability of clinical documentation, while simultaneously producing reusable annotated datasets for downstream model training and distillation. 

---
# Detecting Stealth Sycophancy in Mental-Health Dialogue with Dynamic Emotional Signature Graphs 

**Authors**: Tianze Han, Beining Xu, Hanbo Zhang, Yongming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03472)  

**Abstract**: As conversational AI therapists are increasingly used in psychological support settings, reliable offline evaluation of therapeutic response quality remains an open problem. This paper studies multi-domain support-dialogue evaluation without relying on large language models as final judges. We use a direct LLM judge as a baseline that reads raw dialogue text and predicts whether the target response is harmful, productive, or neutral. We find that direct LLM judges and symmetric text-similarity metrics are poorly aligned with therapeutic quality because the target label depends on clinical direction: whether the response moves the user state toward regulation or reframing, leaves it broadly unchanged, or reinforces deterioration through higher risk affect or cognitive-distortion mass. To address this issue, we propose Dynamic Emotional Signature Graphs (DESG), a model-agnostic evaluator that represents dialogue windows with decoupled clinical states and scores them using asymmetric clinical geometry. We evaluate DESG on a constructed diagnostic stress-test benchmark of 3{,}000 dialogue windows from EmpatheticDialogues, ESConv, and CRADLE-Dialogue, covering peer support, counseling dialogue, and crisis-oriented interaction. On the 600-window held-out test aggregate, DESG-Ensemble achieves 0.9353 macro-F1, exceeding ConcatANN by 1.51 percentage points, BERTScore by 19.63 points, and TRACT by 33.81 points. Feature ablations, artifact controls, a 100-window blinded adjudicator audit, and qualitative disagreement cases indicate that the clinical state manifold is the main discriminative substrate, while graph-based trajectory components provide asymmetric scoring and interpretable diagnostics rather than serving as the sole source of performance. 

---
# FINER-SQL: Boosting Small Language Models for Text-to-SQL 

**Authors**: Thanh Dat Hoang, Thanh Trung Huynh, Matthias Weidlich, Thanh Tam Nguyen, Tong Chen, Hongzhi Yin, Quoc Viet Hung Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.03465)  

**Abstract**: Large language models have driven major advances in Text-to-SQL generation. However, they suffer from high computational cost, long latency, and data privacy concerns, which make them impractical for many real-world applications. A natural alternative is to use small language models (SLMs), which enable efficient and private on-premise deployment. Yet, SLMs often struggle with weak reasoning and poor instruction following. Conventional reinforcement learning methods based on sparse binary rewards (0/1) provide little learning signal when the generated SQLs are incorrect, leading to unstable or collapsed training. To overcome these issues, we propose FINER-SQL, a scalable and reusable reinforcement learning framework that enhances SLMs through fine-grained execution feedback. Built on group relative policy optimization, FINER-SQL replaces sparse supervision with dense and interpretable rewards that offer continuous feedback even for incorrect SQLs. It introduces two key reward functions: a memory reward, which aligns reasoning with verified traces for semantic stability, and an atomic reward, which measures operation-level overlap to grant partial credit for structurally correct but incomplete SQLs. This approach transforms discrete correctness into continuous learning, enabling stable, critic-free optimization. Experiments on the BIRD and Spider benchmarks show that FINER-SQL achieves up to 67.73\% and 85\% execution accuracy with a 3B model -- matching much larger LLMs while reducing inference latency to 5.57~s/sample. These results highlight a cost-efficient and privacy-preserving path toward high-performance Text-to-SQL generation. Our code is available at this https URL. 

---
# Learning Generalizable Action Representations via Pre-training AEMG 

**Authors**: Zhenghao Huang, Huilin Yao, Kaikai Wang, Lin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03462)  

**Abstract**: A fundamental role in decoding human motor intent and enabling intuitive human-computer interaction is played by electromyography (EMG). However, its generalization capability across subjects, devices, and tasks remains substantially limited by data heterogeneity, label scarcity, and the lack of a unified representational framework. To bridge this gap, we propose Any Electromyography (AEMG), the first large-scale, self-supervised representation learning framework for EMG. AEMG reconceptualizes neuromuscular dynamics linguistically, utilizing a novel Neuromuscular Contraction Tokenizer (NCT) to translate discrete muscle contractions into structural words and temporal activation patterns into coherent sentences. Furthermore, we compile the largest cross-device EMG signal vocabulary to date, enabling seamless transfer across arbitrary channel topologies and sampling rates. Experiments demonstrate that AEMG improves the zero-shot leave-one-subject-out (LOSO) accuracy by 5.79-9.25% compared to six state-of-the-art baselines, and achieves more than 90% few-shot adaptation performance with only 5% of target user data. Our work has proposed the concept of EMG signals as a cross-device physiological language, learned their grammar from massive amounts of data, and laid the groundwork for a single-training, universally applicable EMG foundation model. 

---
# Exposing LLM Safety Gaps Through Mathematical Encoding:New Attacks and Systematic Analysis 

**Authors**: Haoyu Zhang, Mohammad Zandsalimy, Shanu Sushmita  

**Link**: [PDF](https://arxiv.org/pdf/2605.03441)  

**Abstract**: Large language models (LLMs) employ safety mechanisms to prevent harmful outputs, yet these defenses primarily rely on semantic pattern matching. We show that encoding harmful prompts as coherent mathematical problems -- using formalisms such as set theory, formal logic, and quantum mechanics -- bypasses these filters at high rates, achieving 46%--56% average attack success across eight target models and two established benchmarks. Crucially, the effectiveness depends not on mathematical notation itself, but on whether a helper LLM deeply reformulates the harmful content into a genuine mathematical problem: rule-based encodings that apply mathematical formatting without such reformulation perform no better than unencoded baselines. We introduce a novel Formal Logic encoding that achieves attack success comparable to Set Theory, demonstrating that this vulnerability generalizes across mathematical formalisms. Additional experiments with repeat post-processing confirm that these attacks are robust to simple prompt augmentation. Notably, newer models (GPT-5, GPT-5-Mini) show substantially greater robustness than older models, though they remain vulnerable. Our findings highlight fundamental gaps in current safety frameworks and motivate defenses that reason about mathematical structure rather than surface-level semantics. 

---
# DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data 

**Authors**: Al Zadid Sultan Bin Habib, Gianfranco Doretto, Donald A. Adjeroh  

**Link**: [PDF](https://arxiv.org/pdf/2605.03430)  

**Abstract**: High-dimensional tabular data lacks a natural feature order, limiting the applicability of permutation-sensitive deep learning models. We propose DynaTab, a dynamic feature ordering-enabled architecture inspired by neural rewiring. We introduce a lightweight criterion that predicts when feature permutation will benefit a dataset by quantifying its intrinsic complexity. DynaTab dynamically reorders features via a neural rewiring algorithm and processes them through a compact, dynamic order-aware combination of separate learned positional embedding, importance-based gating, and masked attention layers, compatible with any sequence-sensitive backbone. Trained end-to-end with bespoke dynamic feature ordering (DFO) and dispersion losses, DynaTab achieves statistically significant gains, particularly on high-dimensional datasets, where it is benchmarked against 45 state-of-the-art baselines across 36 different real-world tabular datasets. Our results position DynaTab as a compelling new paradigm for high-dimensional tabular deep learning. 

---
# Deepfake Audio Detection Using Self-supervised Fusion Representations 

**Authors**: Khalid Zaman, Qixuan Huang, Muhammad Uzair, Masashi Unoki  

**Link**: [PDF](https://arxiv.org/pdf/2605.03420)  

**Abstract**: This paper describes a submission to the Environment-Aware Speech and Sound Deepfake Detection Challenge (ESDD2) 2026, which addresses component-level deepfake detection using the CompSpoofV2 dataset, where speech and environmental sounds may be independently manipulated. To address this challenge, a dual-branch deepfake detection framework is proposed to jointly model speech and environmental contextual representations from input audio. Two pretrained models, XLS-R for speech and BEATs for environmental sound, are used to extract complementary contextual representations. A Matching Head is introduced to model representation differences through statistical normalization and representation interaction, enabling estimation of the original class. In parallel, multi-head cross-attention enables effective information exchange between speech and environmental components. The refined representations are processed with residual connections and layer normalization, and passed to an AASIST classifier to predict speech-based and environment-based spoofing probabilities. The model outputs original, speech, and environment predictions. On the test set, the proposed system achieves an F1-score of 70.20% and an environmental EER of 16.54%, outperforming the baseline system. 

---
# Learning to Theorize the World from Observation 

**Authors**: Doojin Baek, Gyubin Lee, Junyeob Baek, Hosung Lee, Sungjin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2605.03413)  

**Abstract**: What does it mean to understand the world? Contemporary world models often operationalize understanding as accurate future prediction in latent or observation space. Developmental cognitive science, however, suggests a different view: human understanding emerges through the construction of internal theories of how the world works, even before mature language is acquired. Inspired by this theory-building view of cognition, we introduce Learning-to-Theorize, a learning paradigm for inferring explicit explanatory theories of the world from raw, non-textual observations. We instantiate this paradigm with the Neural Theorizer (NEO), a probabilistic neural model that induces latent programs as a learned Language of Thought and executes them through a shared transition model. In NEO, a theory is represented as an executable, compositional program whose learned primitives can be systematically recombined to explain novel phenomena. Experiments show that this formulation enables explanation-driven generalization, allowing observations to be understood in terms of the programs that generate them. 

---
# Smart Passive Acoustic Monitoring: Embedding a Classifier on AudioMoth Microcontroller 

**Authors**: Louis Lerbourg, Paul Peyret, Juliette Linossier, Marielle Malfante  

**Link**: [PDF](https://arxiv.org/pdf/2605.03412)  

**Abstract**: Passive Acoustic Monitoring (PAM) is an efficient and non-invasive method for surveying ecosystems at a reduced cost. Typically, autonomous recorders allow the acquisition of vast bioacoustic datasets which are then analyzed. However, power consumption and data storage are both scarce and limit the duration of acquisition campaigns. To address this issue, we propose a smart PAM system which allows the in-situ analysis of the soundscape by embedding a classifier directly onto an AudioMoth microcontroller. Specifically, we propose an optimized yet simple 1D Convolutional Neural Network (1D-CNN) to classify the raw audio. The model focuses on the specific call of Scopoli Shearwater seabirds (endangered species) and is trained on a real-world dataset with a classification accuracy of 91\% (balanced accuracy of 89\%). We also propose a process to optimize the model to fit the severe resource constraints of the AudioMoth, achieving a \~10kB RAM memory footprint and 20ms inference time. Finally, we present an open-source tutorial of our model optimization and export strategy which can be used for embedding models beyond the scope of our study. Our modified version of the AudioMoth firmware adds two functions: (F1) which selectively records data when the target species has been detected and (F2) which logs the continuous classification results in real time. This work intends to facilitate the conception of intelligent sensors, enhancing the efficiency and scalability of bioacoustic monitoring campaigns. 

---
# Discovering Reinforcement Learning Interfaces with Large Language Models 

**Authors**: Akshat Singh Jaswal, Ashish Baghel, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2605.03408)  

**Abstract**: Reinforcement learning systems rely on environment interfaces that specify observations and reward functions, yet constructing these interfaces for new tasks often requires substantial manual effort. While recent work has automated reward design using large language models (LLMs), these approaches assume fixed observations and do not address the broader challenge of synthesizing complete task interfaces. We study RL task interface discovery from raw simulator state, where both observation mappings and reward functions must be generated. We propose LIMEN (Code available at this https URL), a LLM guided evolutionary framework that produces candidate interfaces as executable programs and iteratively refines them using policy training feedback. Across novel discrete gridworld tasks and continuous control domains spanning locomotion and manipulation, joint evolution of observations and rewards discovers effective interfaces given only a trajectory-level success metric, while optimizing either component alone fails on at least one domain. These results demonstrate that automatic construction of RL interfaces from raw state can substantially reduce manual engineering and that observation and reward components often benefit from co-design, as single-component optimization fails catastrophically on at least one domain in our evaluation suite. 

---
# APEX: Large-scale Multi-task Aesthetic-Informed Popularity Prediction for AI-Generated Music 

**Authors**: Jaavid Aktar Husain, Dorien Herremans  

**Link**: [PDF](https://arxiv.org/pdf/2605.03395)  

**Abstract**: Music popularity prediction has attracted growing research interest, with relevance to artists, platforms, and recommendation systems. However, the explosive rise of AI-generated music platforms has created an entirely new and largely unexplored landscape, where a surge of songs is produced and consumed daily without the traditional markers of artist reputation or label backing. Key, yet unexplored in this pursuit is aesthetic quality. We propose APEX, the first large-scale multi-task learning framework for AI-generated music, trained on over 211k songs (10k hours of audio) from Suno and Udio, that jointly predicts engagement-based popularity signals - streams and likes scores - alongside five perceptual aesthetic quality dimensions from frozen audio embeddings extracted from MERT, a self-supervised music understanding model. Aesthetic quality and popularity capture complementary aspects of music that together prove valuable: in an out-of-distribution evaluation on the Music Arena dataset, comprising pairwise human preference battles across eleven generative music systems unseen during training, including aesthetic features consistently improves preference prediction, demonstrating strong generalisation of the learned representations across generative architectures. 

---
# A Fast Model Counting Algorithm for Two-Variable Logic with Counting and Modulo Counting Quantifiers 

**Authors**: Shixin Sun, Astrid Klipfel, Ondřej Kuželka, Yuanhong Wang, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03391)  

**Abstract**: Weighted first-order model counting (WFOMC) is a central task in lifted probabilistic inference: It asks for the weighted sum of all models of a first-order sentence over a finite domain. A long line of work has identified domain-liftable fragments of first-order logic, that is, syntactic classes for which WFOMC can be solved in time polynomial in the domain size. Among them, the two-variable fragment with counting quantifiers, $\mathbf{C}^2$, is one of the most expressive known liftable fragments. Existing algorithms for $\mathbf{C}^2$, however, establish tractability through multi-stage reductions that eliminate counting quantifiers via cardinality constraints, which introduces substantial practical overhead as the domain size grows. In this paper, we introduce IncrementalWFOMC3, a lifted algorithm for WFOMC on $\mathbf{C}^2$ and its modulo counting extension, $\mathbf{C}^2_{\text{mod}}$. Instead of relying on reduction techniques, IncrementalWFOMC3 operates directly on a Scott normal form that retains counting quantifiers throughout inference. This direct treatment yields two main results. First, we derive a tighter data-complexity bound for WFOMC in $\mathbf{C}^2$, reducing the degree of the polynomial from quadratic to linear in the counting parameters. Second, we prove that $\mathbf{C}^2_{\text{mod}}$ is domain-liftable, extending tractability from $\mathbf{C}^2$ to a richer fragment with native modulo counting support. Finally, our empirical evaluation shows that IncrementalWFOMC3 delivers orders-of-magnitude runtime improvements and better scalability than both existing WFOMC algorithms and state-of-the-art propositional model counters. 

---
# Enhancing Self-Supervised Talking Head Forgery Detection via a Training-Free Dual-System Framework 

**Authors**: Ke Liu, Jiwei Wei, Shuchang Zhou, Yutong Xiao, Ruikun Chai, Yitong Qin, Yuyang Zhou, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03390)  

**Abstract**: Supervised talking head forgery detection faces severe generalization challenges due to the continuous evolution of generators. By reducing reliance on generator-specific forgery patterns, self-supervised detectors offer stronger cross-generator robustness. However, existing research has mainly focused on building stronger detectors, while the discriminative capacity of trained detectors remains insufficiently exploited. In particular, for score-based self-supervised detectors, the limited discriminative ability on hard cases is often reflected in unreliable anomaly ordering, leaving room for further refinement. Motivated by this observation, we draw inspiration from the dual-system theory of human cognition and propose a Training-Free Dual-System (TFDS) framework to further exploit the latent discriminative capacity of existing score-based self-supervised detectors. TFDS treats anomaly-like scores as the basis of System-1, using lightweight threshold-based routing to partition samples into confident and uncertain subsets. System-2 then revisits only the uncertain subset, performing fine-grained evidence-guided reasoning to refine the relative ordering of ambiguous samples within the original score distribution. Extensive experiments demonstrate consistent improvements across datasets and perturbation settings, with the gains arising mainly from corrected ordering within the uncertain subset. These findings show that existing self-supervised talking head forgery detectors still contain underexploited discriminative cues that can be effectively unlocked through training-free dual-system reasoning. 

---
# Local Truncation Error-Guided Neural ODEs for Large Scale Traffic Forecasting 

**Authors**: Xiao Zhang, Yafei Li, Ruixiang Wang, Wei Wei, Shuo He, Mingliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03386)  

**Abstract**: Spatiotemporal forecasting in physical systems, such as large-scale traffic networks, requires modeling a dual dynamic: continuous macroscopic rhythms and discrete, unpredictable microscopic shocks. While Neural Ordinary Differential Equations (ODEs) excel at capturing smooth evolution, their inherent Lipschitz continuity constraints inevitably cause severe over-smoothing when confronting abrupt anomalies. Recent physics-informed methods attempt to bypass this by penalizing numerical integration errors to enforce manifold smoothness. However, we mathematically reveal that such rigid regularization inherently triggers gradient conflicts and ``attention collapse,'' stripping the model of its sensitivity to anomalies. To resolve this continuity-shock dilemma, we propose Local Truncation Error-Guided Neural ODEs (LTE-ODE). Rather than treating numerical error as a nuisance to be eliminated, we innovatively repurpose the Local Truncation Error (LTE) as an unsupervised forward inductive bias. By mapping the LTE into a dynamic spatial attention mask, our architecture gracefully preserves high-precision continuous ODE evolution in stable regions, while adaptively triggering a discrete compensation branch exclusively at shock points. Trained purely end-to-end without manifold penalties, LTE-ODE achieves state-of-the-art performance on multiple large-scale benchmarks, exhibiting exceptional robustness against highly non-linear fluctuations. Furthermore, our ablation on integration steps demonstrates high deployment flexibility, allowing the model to seamlessly adapt to varying hardware memory constraints in real-world applications. 

---
# SkCC: Portable and Secure Skill Compilation for Cross-Framework LLM Agents 

**Authors**: Yipeng Ouyang, Yi Xiao, Yuhao Gu, Xianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03353)  

**Abstract**: LLM-Agents have evolved into autonomous systems for complex task execution, with the this http URL specification emerging as a de facto standard for encapsulating agent capabilities. However, a critical bottleneck remains: different agent frameworks exhibit starkly different sensitivities to prompt formatting, causing up to 40% performance variation, yet nearly all skills exist as a single, format-agnostic Markdown version. Manual per-platform rewriting creates an unsustainable maintenance burden, while prior audits have found that over one third of community skills contain security vulnerabilities. To address this, we present SkCC, a compilation framework that introduces classical compiler design into agent skill development. At its core, SkIR - a strongly-typed intermediate representation - decouples skill semantics from platform-specific formatting, enabling portable deployment across heterogeneous agent frameworks. Around this IR, a compile-time Analyzer enforces security constraints via Anti-Skill Injection before deployment. Through a four-phase pipeline, SkCC reduces adaptation complexity from $O(m \times n)$ to $O(m + n)$. Experiments on SkillsBench demonstrate that compiled skills consistently outperform their original counterparts, improving pass rates from 21.1% to 33.3% on Claude Code and from 35.1% to 48.7% on Kimi CLI, while achieving sub-10ms compilation latency, a 94.8% proactive security trigger rate, and 10-46% runtime token savings across platforms. 

---
# Can Multimodal Large Language Models Understand Pathologic Movements? A Pilot Study on Seizure Semiology 

**Authors**: Lina Zhang, Tonmoy Monsoor, Mehmet Efe Lorasdagi, Prateik Sinha, Chong Han, Peizheng Li, Yuan Wang, Jessica Pasqua, Colin McCrimmon, Rajarshi Mazumder, Vwani Roychowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2605.03352)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated robust capabilities in recognizing everyday human activities, yet their potential for analyzing clinically significant involuntary movements in neurological disorders remains largely unexplored. This pilot study evaluates the capability of MLLMs for automated recognition of pathological movements in seizure videos. We assessed the zero-shot performance of state-of-the-art MLLMs on 20 ILAE-defined semiological features across 90 clinical seizure recordings. MLLMs outperformed fine-tuned Convolutional Neural Network (CNN) and Vision Transformer (ViT) baseline models on 13 of 18 features without task-specific training, demonstrating particular strength in recognizing salient postural and contextual features while struggling with subtle, high-frequency movements. Feature-targeted signal enhancement (facial cropping, pose estimation, audio denoising) improved performance on 10 of 20 features. Expert evaluation showed that 94.3 percent of MLLM-generated explanations for correctly predicted cases achieved at least 60 percent faithfulness scores, aligning with epileptologist reasoning. These findings demonstrate the potential of adapting general-purpose MLLMs for specialized clinical video analysis through targeted preprocessing strategies, offering a path toward interpretable, efficient diagnostic assistance. Our code is publicly available at this https URL. 

---
# VLMaxxing through FrameMogging Training-Free Anti-Recomputation for Video Vision-Language Models 

**Authors**: JF Bastien, Sam D'Amico  

**Link**: [PDF](https://arxiv.org/pdf/2605.03351)  

**Abstract**: Video vision-language models (VLMs) keep paying for visual state the stream already told us was stable. The factory wall did not move, but most VLM pipelines still hand the model dense RGB frames or a fresh prefix again. We study that waste as training-free anti-recomputation: reuse state when validation says it survives, and buy fresh evidence when the scene, query, or cache topology requires it.
The largest measured win is after ingest. On frozen Qwen2.5-VL-7B-Instruct-4bit, adaptive same-video follow-up reuse preserves paired choices and correctness on a 93-query VideoMME breadth setting while reducing follow-up latency by 14.90-35.92x. The first query is still cold; the win starts when later questions reuse the same video state. Stress tests bound the result: repeated-question schedules hold through 50 turns, while dense-answer-anchored prompt variation separates conservative fixed K=1 repair from faster aggressive policies that drift.
Fresh-video pruning is smaller but real. C-VISION skips timed vision-tower work before the first answer is generated. On Gemma 4-E4B-4bit, the clean 32f short cell reaches 1.316x first-query speedup with no paired drift or parse failures on 20 items; Qwen shows the fidelity/speed boundary.
Stage-share ceiling (C-CEILING) is the accounting guardrail: a component speedup becomes an end-to-end speedup only in proportion to the wall-clock share it accelerates, so C-VISION and after-ingest follow-up reuse do not multiply. Candidate C-STREAM remains a native-rate target, not a headline result here. The broader direction is VLM-native media that expose change, motion, uncertainty, object state, sensor time, and active tiles directly, so models do not have to rediscover the world from dense RGB every frame. 

---
# Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts 

**Authors**: Hahyeon Choi, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2605.03348)  

**Abstract**: We propose S3 (Specialization, Selection, Sparsification), a framework that rethinks multimodal learning through a structural perspective. Instead of encoding all signals into a fixed embedding, S3 decomposes multimodal inputs into semantic experts and selectively routes them for each task. Specialization forms concept-level experts in a shared latent space, Selection adapts routing for task-specific needs, and Sparsification prunes low-utility paths to yield compact, information-minimal representations. Across four MultiBench benchmarks, S3 improves accuracy and shows a consistent reverse U-shaped sparsity-performance trend, with peak performance at intermediate sparsity. These results suggest that structuring multimodal representations as selectable semantic components provides a practical and principled alternative to contrastive learning or InfoMax-driven approaches. 

---
# RAG over Thinking Traces Can Improve Reasoning Tasks 

**Authors**: Negar Arabzadeh, Wenjie Ma, Sewon Min, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2605.03344)  

**Abstract**: Retrieval-augmented generation (RAG) has proven effective for knowledge-intensive tasks, but is widely believed to offer limited benefit for reasoning-intensive problems such as math and code generation. We challenge this assumption by showing that the limitation lies not in RAG itself, but in the choice of corpus. Instead of retrieving documents, we propose retrieving thinking traces, i.e., intermediate thinking trajectories generated during problem solving attempts. We show that thinking traces are already a strong retrieval source, and further introduce T3, an offline method that transforms them into structured, retrieval-friendly representations, to improve usability. Using these traces as a corpus, a simple retrieve-then-generate pipeline consistently improves reasoning performance across strong models and benchmarks such as AIME 2025--2026, LiveCodeBench, and GPQA-Diamond, outperforming both non-RAG baselines and retrieval over standard web corpora. For instance, on AIME, RAG with traces generated by Gemini-2-thinking achieves relative gains of +56.3%, +8.6%, and +7.6% for Gemini-2.5-Flash, GPT-OSS-120B, and GPT-5, respectively, even though these are more recent models. Interestingly, RAG on T3 also incurs little or no extra inference cost, and can even reduce inference cost by up to $15%$. Overall, our results suggest that thinking traces are an effective retrieval corpus for reasoning tasks, and transforming them into structured, compact, or diagnostic representations unlocks even stronger gains. Code available at this https URL. 

---
# FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles 

**Authors**: Lucas Yunkyu Lee, Soonho Kim, Youngwook Kim, Sangmin Kim, Jaesik Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.03337)  

**Abstract**: The recent surge in 4D Gaussian Splatting (4DGS) has achieved impressive dynamic scene reconstruction. While these methods demonstrate remarkable performance, the specific drivers behind such gains remain less explored, making a systematic understanding of the underlying principles challenging. In this paper, we perform a comprehensive analysis of these hidden factors to provide a clearer perspective on the 4DGS framework. We first establish a controlled baseline, FreeTimeGS_ours, by formalizing and reproducing the heuristics of the state-of-the-art FreeTimeGS. Using this framework, we dissect 4DGS along its fundamental axes and uncover key secrets, including the emergent temporal partitioning driven by Gaussian durations and the discrepancy between photometric fidelity and spatiotemporal consistency. Based on these insights, we propose FreeTimeGS++, a principled method that employs gated marginalization and neural velocity fields to achieve superior stability and robust dynamic representations. Our approach yields reproducible results with reduced run-to-run variance. We will release our implementation to provide a reliable foundation for future 4DGS research. 

---
# LLM-ADAM: A Generalizable LLM Agent Framework for Pre-Print Anomaly Detection in Additive Manufacturing 

**Authors**: Ahmadreza Eslaminia, Chuhan Cai, Cameron Smith, Ruo-Syuan Mei, Shichen Li, Rajiv Malhotra, Klara Nahrstedt, Chenhui Shao  

**Link**: [PDF](https://arxiv.org/pdf/2605.03328)  

**Abstract**: Additive manufacturing (AM) continues to transform modern manufacturing by enabling flexible, on-demand production of complex geometries across diverse industries. Fused filament fabrication (FFF) has extended AM to laboratories, classrooms, and small production environments, but this accessibility shifts process-planning responsibility to users who may lack manufacturing expertise. A syntactically valid slicer profile can still encode thermally or geometrically harmful settings, and subtle G-code edits can alter extrusion, cooling, or adhesion before a print begins. Pre-print G-code screening catches accidental or adversarial machine-program errors before material or machine time is wasted. This paper proposes LLM-ADAM as a generalizable LLM framework for pre-print anomaly detection in AM. The framework decomposes the task into three roles: Extractor-LLM maps a G-code file to a structured process-parameter schema; Reference-LLM converts printer and material documentation into aligned operating ranges; and Judge-LLM interprets a deterministic deviation table and G-code evidence to decide whether a part is non-defective or belongs to an anomaly class. Printers, materials, and LLM backbones are interchangeable test conditions, not fixed assumptions. We evaluate the framework on an N=200 FFF G-code corpus spanning two desktop printer families, two materials, and five classes including non-defective, under-extrusion, over-extrusion, warping, and stringing. The best framework configuration reaches 87.5% accuracy, compared with 59.5% for the strongest engineered single-LLM baseline. The results show that structured decomposition, rather than backbone strength alone, is the dominant source of improvement, with defect classes identified at or near ceiling for leading configurations while residual errors concentrate on conservative false alarms for non-defective samples. 

---
# DGPO: Distribution Guided Policy Optimization for Fine Grained Credit Assignment 

**Authors**: Hongbo Jin, Rongpeng Zhu, Zhongjing Du, Xu Jiang, Jingqi Tian, Qiaoman Zhang, Jiayu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.03327)  

**Abstract**: Reinforcement learning is crucial for aligning large language models to perform complex reasoning tasks. However, current algorithms such as Group Relative Policy Optimization suffer from coarse grained, sequence level credit assignment, which severely struggles to isolate pivotal reasoning steps within long Chain of Thought generations. Furthermore, the standard unbounded Kullback Leibler divergence penalty induces severe gradient instability and mode seeking conservatism, ultimately stifling the discovery of novel reasoning trajectories. To overcome these limitations, we introduce Distribution Guided Policy Optimization, a novel critic free reinforcement learning framework that reinterprets distribution deviation as a guiding signal rather than a rigid penalty. 

---
# AHPA: Adaptive Hierarchical Prior Alignment for Diffusion Transformers 

**Authors**: Ruibin Min, Yexin Liu, Aimin Pan, Changsheng Lu, Jiafei Wu, Kelu Yao, Xiaogang Xu, Harry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03317)  

**Abstract**: Representation alignment has recently emerged as an effective paradigm for accelerating Diffusion Transformer training. Despite their success, existing alignment methods typically impose a fixed supervision target or a fixed alignment granularity throughout the entire denoising trajectory, whether the guidance is provided by external vision encoders, internal self-representations, or VAE-derived features. We argue that such timestep-agnostic alignment is suboptimal because the useful granularity of representation supervision changes systematically with the signal-to-noise ratio. In high-noise regimes, diffusion models benefit more from coarse semantic and layout-level anchoring, whereas in low-noise regimes, the training signal should emphasize spatially detailed and structurally faithful refinement. This non-stationary alignment behavior creates a representational mismatch for static single-level supervisors. To address this issue, we propose Adaptive Hierarchical Prior Alignment (AHPA), a lightweight alignment framework that exploits the hierarchical representations naturally embedded in the frozen VAE encoder. Instead of using only a single compressed latent as the alignment target, AHPA extracts multi-level VAE features that provide complementary priors ranging from local geometry and spatial topology to coarse semantic layout. A timestep-conditioned Dynamic Router adaptively selects and weights these hierarchical priors along the denoising trajectory, thereby synchronizing the alignment granularity with the model's evolving training needs. Extensive experiments show that AHPA improves convergence and generation quality over baselines and incurs no additional inference cost while avoiding external encoder supervision during training. 

---
# Cryptographic Registry Provenance: Structural Defense Against Dependency Confusion in AI Package Ecosystems 

**Authors**: Alan L. McCann  

**Link**: [PDF](https://arxiv.org/pdf/2605.03309)  

**Abstract**: Dependency confusion attacks exploit a structural gap in software distribution: once a package is installed, there is no cryptographic proof of which registry distributed it. Every existing defense is configuration-based and fails silently when misconfigured. We present a cryptographic distribution provenance system comprising three components: (1) cryptographic registry identity, where every registry holds an Ed25519 keypair and signs every artifact it distributes; (2) a dual-signature model, where the publisher signs at packaging time and the registry countersigns at publication time; and (3) authoritative namespace binding, where consumers pin registry fingerprints and the resolver cryptographically rejects artifacts from unauthorized registries. These create three defense layers requiring simultaneous compromise for a successful attack. A comparison across eight ecosystems (npm, Cargo, this http URL, PyPI, Go modules, Docker/OCI, NuGet, Maven) shows no existing ecosystem combines mandatory publisher signing, cryptographic registry identity, mandatory registry countersigning, and consumer-side cryptographic enforcement. The system extends to AI-generation provenance as a signed attribute and governance-enforced dependency resolution. A case study integrates distribution provenance with a three-layer runtime governance architecture, creating a four-phase lifecycle chain with no cryptographic gaps. 

---
# SHIELD: A Diverse Clinical Note Dataset and Distilled Small Language Models for Enterprise-Scale De-identification 

**Authors**: Jose D. Posada, David Love, Somalee Datta, Priya Desai  

**Link**: [PDF](https://arxiv.org/pdf/2605.03301)  

**Abstract**: De-identification of clinical text remains essential for secondary use of electronic health records (EHRs), yet public benchmarks such as i2b2 2006/2014 are over a decade old and lack the semantic and demographic diversity of modern narratives. While Large Language Models (LLMs) achieve state-of-the-art zero-shot extraction, enterprise deployment is hindered by compute costs and governance restricting Protected Health Information (PHI) from cloud APIs. We introduce SHIELD (Synthetic Human-annotated Identifier-replaced Entries for Learning and De-identification), a diverse dataset of 1,394 notes with 10,505 gold-standard PHI spans across 9 categories, built via set-cover diversity sampling with human-in-the-loop adjudication. We evaluate four LLMs (two proprietary, two open-weight) to establish a performance ceiling, then distill these capabilities into locally deployable Small Language Models (SLMs). Distributional analysis using Frechet Text Distance and Jensen-Shannon Divergence confirms SHIELD occupies a distinct region of biomedical embedding and vocabulary space versus legacy benchmarks. Our best distilled model matches its teacher on structured PHI categories (DATE, DOCTOR, ID, PATIENT, PHONE) and achieves micro-averaged span-level precision of 0.88 and recall of 0.86 on standard workstation hardware. Cross-dataset evaluation shows diversity-trained models generalize well on universal structured PHI, while institution-specific entities remain hard to transfer, suggesting optimal deployment combines broad-coverage models with specialized models for high-volume notes. We publicly release the SHIELD dataset and the distilled DeBERTa v3 model. 

---
# On the Spectral Structure and Objective Equivalence of Orthogonal Multilabel Fisher Discriminants 

**Authors**: Brian Keith-Norambuena, Juan Bekios-Calfa  

**Link**: [PDF](https://arxiv.org/pdf/2605.03283)  

**Abstract**: We provide a unified theoretical analysis of Linear Discriminant Analysis with simultaneous multilabel scatter matrix formulations and Stiefel orthogonality constraints. Our contributions span both algebraic structure and statistical guarantees. On the algebraic side, we characterize the rank of the multilabel between-class scatter matrix, showing that the effective discriminant dimensionality can strictly exceed the classical single-label bound of $C-1$; we establish a multilabel partition of variance and prove that all four Fisher objectives are equivalent under the $W^\top S_t^{ML} W = I_r$ constraint while characterizing their divergence under the Stiefel constraint; and we prove a two-sided label-distance preservation bound relating projected distances to Hamming distances in label space. On the statistical side, we establish a finite-sample $O(k_{\max}\sqrt{d\log d/n}/gap_r)$ bound on the subspace estimation error under sub-Gaussian noise with a matching $\Omega(\sigma^2 d/(n\,gap_r))$ minimax lower bound, establishing a near-minimax-optimal rate (matching up to logarithmic and $k_{\max}$ factors) for multilabel discriminant subspace estimation. We further provide high-probability distance concentration, robustness guarantees under label interactions, and a regularization analysis preserving the spectral structure when $d \gg n$. All results are verified numerically on synthetic data generated from the linear label-effect model, covering both the algebraic identities and the multilabel-specific quantities ($k_{\max}$, $\kappa(S_t^{ML})$, $\|\Gamma/n\|_2$, $\Delta_r$) that govern the statistical bounds. The numerical experiments are designed as a sanity check for the theorems rather than as an empirical benchmark; evaluation on real multilabel datasets is left to future work targeting application-oriented venues. 

---
# Copula-Based Endogeneity Correction for Doubly Robust Estimation of Treatment Effect 

**Authors**: Sahil Shikalgar, Md. Noor-E-Alam  

**Link**: [PDF](https://arxiv.org/pdf/2605.03278)  

**Abstract**: Doubly Robust (DR) estimation of treatment effect relies on an untestable assumption that is the absence of unobserved confounding. This assumption is par- ticularly problematic in the context of healthcare research, where variables like pre- scription refill rates serve as proxies for unobserved behaviors such as medication adherence. These proxy variables are often endogenous, exhibiting correlation with the regression error term due to unmeasured confounding or measurement error. We propose a copula-corrected doubly robust estimator that addresses endogeneity in both the treatment and outcome models without requiring instrumental variables. Gaussian copulas model the joint distribution of endogenous covariates and the error term, enabling consistent estimation while preserving the doubly robust property that requires correct specification of either the treatment or outcome model, not both. Monte Carlo simulations demonstrate that naive DR estimation exhibits substantial bias under endogeneity, whereas our corrected estimator recovers unbiased treatment effects across different data-generating processes. We apply our method to examine the effect of nutritional counseling on blood pressure using the National Health and Nutrition Examination Survey (NHANES) data. Naive DR estimation suggests counseling is associated with increased blood pressure. After copula correction, this effect becomes statistically insignificant, consistent with literature showing modest effects of nutri- Counseling in reducing blood pressure. Our methodology provides researchers with a practical tool for obtaining treatment effects in the presence of endogeneity. 

---
# RLDX-1 Technical Report 

**Authors**: Dongyoung Kim, Huiwon Jang, Myungkyu Koo, Suhyeok Jang, Taeyoung Kim, Beomjun Kim, Byungjun Yoon, Changsung Jang, Daewon Choi, Dongsu Han, Donguk Lee, Heeseung Kwon, Hojin Jeon, Jaehyun Kang, Jaekyoung Bae, Jihyuk Lee, Jimin Lee, John Won, Joonwoo Ahn, Junhyeong Park, Junyoung Sung, Kyungmin Lee, Minseong Han, Minsung Yoon, Sejune Joo, Seonil Son, Seungcheol Park, Seunggeun Cho, Seungjun Moon, Seungku Kim, Yonghoon Dong, Yongjin Cho, Youngchan Kim, Chang Hwan Kim, Dohyeon Kim, Hazel Lee, Heecheol Kim, Hensen Ahn, Hyungkyu Ryu, Hyunsoo Choi, Hyunsoo Shin, Jaeheon Jung, Jaewoo Kim, Jinwook Kim, Joochul Chang, Joonsoo Kim, Junghun Park, Jungwoo Park, Junho Cho, Junhyeok Park, Junwon Lee, Kangwook Lee, Kwanghoon Kim, Kyoungwhan Choe, Manoj Bhadu, Nayoung Oh, Sangjun Kim, Sangwoo Kim, Seunghoon Shim, Seunghyun Kim, Seungjun Lee, Seungyup Ka, Sungryol Yang, Wook Jung, Yashu Shukla, Yeonjae Lee, Yeonwoo Bae, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2605.03269)  

**Abstract**: While Vision-Language-Action models (VLAs) have shown remarkable progress toward human-like generalist robotic policies through the versatile intelligence (i.e. broad scene understanding and language-conditioned generalization) inherited from pre-trained Vision-Language Models, they still struggle with complex real-world tasks requiring broader functional capabilities (e.g. motion awareness, memory-aware decision making, and physical sensing). To address this, we introduce RLDX-1, a general-purpose robotic policy for dexterous manipulation built on the Multi-Stream Action Transformer (MSAT), an architecture that unifies these capabilities by integrating heterogeneous modalities through modality-specific streams with cross-modal joint self-attention. RLDX-1 further combines this architecture with system-level design choices, including synthesizing training data for rare manipulation scenarios, learning procedures specialized for human-like manipulation, and inference optimizations for real-time deployment. Through empirical evaluation, we show that RLDX-1 consistently outperforms recent frontier VLAs (e.g. $\pi_{0.5}$ and GR00T N1.6) across both simulation benchmarks and real-world tasks that require broad functional capabilities beyond general versatility. In particular, RLDX-1 shows superiority in ALLEX humanoid tasks by achieving success rates of 86.8% while $\pi_{0.5}$ and GR00T N1.6 achieve around 40%, highlighting the ability of RLDX-1 to control a high-DoF humanoid robot under diverse functional demands. Together, these results position RLDX-1 as a promising step toward reliable VLAs for complex, contact-rich, and dynamic real-world dexterous manipulation. 

---
# Partially Observed Structural Causal Models 

**Authors**: Turan Orujlu, Jordan Matelsky, Martin V. Butz, Charley M. Wu, Konrad P. Kording  

**Link**: [PDF](https://arxiv.org/pdf/2605.03268)  

**Abstract**: Here we introduce Partially Observed Structural Causal Models (POSCMs) that formalize causal systems where latent contexts co-determine both the interaction structure and downstream mechanisms on observed variables. POSCMs provide an extension of structural causal models (SCMs), as a self-contained causal modeling framework for endogenous graphs, allowing for an intervention hierarchy spanning node- and edge-level context and endogenous variable interventions. To enable surgical edge interventions, we adopt a Kolmogorov-Arnold-Sprecher edge-functional decomposition, an existence theorem for representing each node mechanism as a sum of univariate functions of its parents, yielding an explicit parametrization of dyadic functional contributions. We provide an identifiability theory that clarifies which intervention families would suffice to disentangle structure formation from mechanisms. We empirically validate these predictions in a biophysically detailed virtual human retina simulator, constructing intervention protocols that (i) reproduce the non-identifiability predicted when context is latent and no context-level interventions are available, (ii) exhibit structure-mechanism confounding under latent edges when only node interventions are observed, and (iii) recover synaptic input-output relationships via targeted node interventions, consistent with our positive kernel identifiability result. Our work generalizes SCMs in a way that allows it to work in a world closer to the one we live in. 

---
# Can AI Help You Get Over Your Breakup? One Session with a Belief-Reframing Chatbot Shows Sustained Distress Reduction 

**Authors**: Thomas Menzel, Michel Schimpf, Thomas Bohné  

**Link**: [PDF](https://arxiv.org/pdf/2605.03261)  

**Abstract**: Romantic breakups are among the most common and intense sources of psychological distress. We evaluated *overit*, a single-session AI chatbot that uses cognitive reappraisal to address breakup distress, informed by memory reconsolidation theory. In a pre-registered randomized controlled trial, 254 adults in the United States and United Kingdom who had experienced a romantic breakup were assigned to either an initial survey assessment followed by an AI chat session or to a survey-only control. Breakup distress was measured at baseline, 7 days, and again at an exploratory 1-month follow-up using the Breakup Distress Scale. Participants assigned to *overit* showed a significantly greater reduction in breakup distress than controls at 7 days (time-by-condition interaction B = -5.36, SE = 1.19, p < .001; completer-based d = -0.70). A smaller but still significant treatment advantage remained detectable at the exploratory 1-month follow-up among post-session completers (B = -2.92, SE = 1.22, p = .017). Exploratory post hoc moderation suggested a larger effect among male participants (B = 7.78, p = .003). These results suggest that a brief AI chatbot conversation can meaningfully reduce breakup distress, with exploratory evidence that a smaller advantage persists over the following month. Future work should test the intervention against active controls, evaluate repeated-session use, and recruit more diverse samples. 

---
# Ortho-Hydra: Orthogonalized Experts for DiT LoRA 

**Authors**: Seunghyun Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.03252)  

**Abstract**: LoRA fine-tuning of diffusion transformers (DiT) on multi-style data suffers from \emph{style bleed}: a single low-rank residual cannot represent several distinct artist fingerprints, and the optimizer converges to their average. Mixture-of-experts LoRA in the HydraLoRA style replaces the up-projection with $E$ heads under a router, but when every expert is zero-initialized the router receives identical gradient from each head and remains at the uniform prior. The experts then evolve permutation-symmetrically, and the network trains as a single rank-$r$ LoRA at $E{\times}$ the cost. We present \textbf{Ortho-Hydra}, a re-parameterisation that combines an OFT-style Cayley-orthogonal shared basis with per-expert \emph{disjoint output subspaces} carved from the top-$(Er)$ left singular vectors of the pretrained weight. Disjointness makes the router's per-expert score non-degenerate at step~$0$, so specialization receives gradient signal before any expert has trained. We test the predicted deadlock on a DiT pipeline by comparing two HydraLoRA baselines, a zero-initialized shared-basis variant and the original $\sigma{=}0.1$ Gaussian-jitter mitigation, against Ortho-Hydra under a matched optimiser, dataset, and step budget. Neither baseline leaves the uniform prior within the first $1\text{k}$ steps; Ortho-Hydra begins de-uniformising within the first few hundred. End-task generation quality on multi-style data is out of scope; we report the construction, the cold-start mechanism, and the routing dynamics it changes. Code: this https URL. 

---
# Posterior-First Neural PDE Simulation: Inferring Hidden Problem State from a Single Field 

**Authors**: Wenshuo Wang, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03247)  

**Abstract**: Neural PDE simulators often receive only a single observed field at deployment. In this setting, a field-to-future predictor can collapse distinct latent problem states into the same deterministic interface, losing the ambiguity needed for reliable rollout and downstream decisions. We propose posterior-first neural PDE simulation: first infer a posterior over the minimal task-sufficient problem state, then condition prediction on that posterior. The resulting theory connects the object, the learning target, and the failure mode: Bayes downstream values factor through this posterior, refinement labels make it learnable by proper scoring rules, and deterministic collapse incurs an ambiguity barrier whenever the true posterior is non-Dirac. Synthetic exact-ambiguity experiments show that point-versus-posterior gaps track the predicted barrier. On metadata-hidden PDEBench tasks, posterior recovery reduces pooled rollout nRMSE from 0.175 to 0.132, closing 59.4% of the direct-to-oracle gap. These results suggest that single-observation neural PDE simulation should be posterior-first rather than monolithic field-to-future prediction. 

---
# S^2tory: Story Spine Distillation for Movie Script Summarization 

**Authors**: Mingzhe Lu, Yanbing Liu, Qihao Wang, Jiarui Zhang, Jiayue Wu, Yue Hu, Yunpeng Li, Yangyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03244)  

**Abstract**: Movie scripts pose a fundamental challenge for automatic summarization due to their non-linear, cross-cut narrative structure, which makes surface-level saliency methods ineffective at preserving core story progression. To address this, we introduce S^2tory (Story Spine Distillation), a narratology-grounded framework that leverages character development trajectories to identify plot nuclei, the essential events that drive the narrative forward, while filtering out peripheral satellite events that merely enrich atmosphere or emotion. Our Narrative Expert Agent (NEAgent) performs theory-constrained reasoning, whose distilled knowledge conditions a small model to identify plot nuclei. Another model then uses these plot nuclei to generate the summary. Experiments on the MovieSum dataset demonstrate state-of-the-art semantic fidelity at approximately 3.5x compression, and zero-shot evaluation on BookSum confirms strong out-of-domain generalization. Human evaluation further validates that narratological theory provides an indispensable foundation for modeling complex, non-linear narratives. 

---
# OptiLookUp: An Optical ROM-Based Loop up Table Engine for Photonic Accelerators 

**Authors**: Ankur Singh, Akhilesh Jaiswal  

**Link**: [PDF](https://arxiv.org/pdf/2605.03241)  

**Abstract**: Read-only memory (ROM) provides deterministic access to predefined data mappings. Extending ROM concepts to the optical domain enables high-bandwidth, low-latency, and parallel memory access, but realizing compact and reconfigurable optical ROM remains challenging due to loss, wavelength control, and integration constraints. This work presents a high-speed, reconfigurable photonic ROM architecture implemented using integrated microring resonators (MRRs). The ROM encodes predefined input-output mappings directly in the spectral response of the photonic devices, enabling deterministic lookup-based operation without dynamic computation during readout. To improve scalability and reduce cumulative insertion loss, the architecture employs compact banked sub-arrays that are selectively addressed through an optical decoding mechanism. Reconfigurability is achieved using transistor-based optical selectors, allowing different ROM banks to be activated without physical light rerouting or interferometric structures. The proposed photonic ROM is designed and evaluated using device-level simulations based on the GlobalFoundries 45SPCLO silicon photonics platform. Simulation results demonstrate reliable operation at data rates up to 12.5 GHz, with stable light-to-current transfer characteristics obtained through integrated photodiode readout. The optical ROM can be used to implement nonlinear activation functions utilised in photonic accelerator architectures, including sigmoid, tanh, ReLU, and exponential mappings. 

---
# MAGE: Safeguarding LLM Agents against Long-Horizon Threats via Shadow Memory 

**Authors**: Yuhui Wang, Tanqiu Jiang, Jiacheng Liang, Charles Fleming, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03228)  

**Abstract**: As large language model (LLM)-powered agents are increasingly deployed to perform complex, real-world tasks, they face a growing class of attacks that exploit extended user-agent-environment interactions to pursue malicious objectives improbable in single-turn settings. Such long-horizon threats pose significant risks to the safe deployment of LLM agents in critical domains. In this paper, we present MAGE (Memory As Guardrail Enforcement), a novel defensive framework designed to counter a wide range of long-horizon threats. Inspired by the "shadow stack" abstraction in systems security, MAGE maintains a dedicated, safety-focused agentic memory that distills and retains safety-critical context across the agent's full execution trajectory, leveraging this shadow memory to proactively assess the risk of pending actions prior to their execution. Extensive evaluation demonstrates that MAGE substantially outperforms existing defenses across diverse long-horizon threats in detection accuracy, achieves early-stage detection for the majority of attacks, and introduces only negligible overhead to agent utility. To our best knowledge, MAGE represents the first framework to detect and mitigate long-horizon threats using an agentic memory approach, establishing a new paradigm for this critical challenge and opening promising directions for future research. 

---
# Self-Mined Hardness for Safety Fine-Tuning 

**Authors**: Prakhar Gupta, Garv Shah, Donghua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03226)  

**Abstract**: Safety fine-tuning of language models typically requires a curated adversarial dataset. We take a different approach: score each candidate prompt's difficulty by how often the target model's own rollouts are judged harmful, then fine-tune on the hardest prompts paired with the model's own non-jailbroken rollouts. On Llama-3-8B-Instruct and Llama-3.2-3B-Instruct, this approach cuts the WildJailbreak attack success rate from 11.5% and 20.1% down to 1-3%, but pushes refusal on jailbreak-shaped benign prompts from 14-22% to 74-94%. Interleaving the same hard prompts 1:1 with adversarially-framed benign prompts (prompts that look like jailbreaks but have benign intent) cuts that refusal back down to 30-51% on 8B and 52-72% on 3B, at a cost of 2-6 percentage points of attack success rate. Within the mixed regime, training on the hardest half of the eligible pool rather than a random half cuts the remaining ASR by 35-50% (about 3 percentage points) on both models. 

---
# MenuNet: A Strategy-Proof Mechanism for Matching Markets 

**Authors**: Zhaohong Sun, Makoto Yokoo  

**Link**: [PDF](https://arxiv.org/pdf/2605.03216)  

**Abstract**: Strategy-proofness is a fundamental desideratum in mechanism design, ensuring truthful reporting and robust participation. Stability is another central requirement in matching markets, widely adopted in applications such as school choice and labor market clearing. In practice, however, these markets are invariably governed by complex distributional constraints, ranging from diversity quotas and regional balance to global capacity slacks, under which stable matchings often fail to exist. This raises a fundamental question: how to distribute unavoidable instability across agents while preserving strategy-proofness? To address this, we propose \texttt{MenuNet}, a strategy-proof mechanism design framework based on a neural representation of menus. Rather than directly constructing assignments, \texttt{MenuNet} learns to generate personalized probabilistic menus, from which assignments are realized via a structured sequential choice rule that guarantees strategy-proofness by construction. By decomposing stability into fairness (no envy) and non-wastefulness, our approach models these properties as vector-valued quantities and optimizes their distribution through differentiable objectives, providing a principled trade-off between competing axioms. Empirically, \texttt{MenuNet} navigates this trade-off effectively: it consistently outperforms Random Serial Dictatorship (RSD) in terms of envy and Deferred Acceptance (DA) in terms of waste, while maintaining scalability and computational efficiency. These results suggest that learning-based menu mechanisms provide a flexible and scalable paradigm for mechanism design in highly constrained, real-world environments. 

---
# When Agents Handle Secrets: A Survey of Confidential Computing for Agentic AI 

**Authors**: Javad Forough, Marios Kogias, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2605.03213)  

**Abstract**: Agentic AI systems, specifically LLM-driven agents that plan, invoke tools, maintain persistent memory, and delegate tasks to peer agents via protocols such as MCP and A2A, introduce a threat surface that differs materially from standalone model inference. Agents accumulate sensitive context, hold credentials, and operate across pipelines no single party fully controls, enabling prompt injection, context exfiltration, credential theft, and inter-agent message poisoning. Current defenses operate entirely within the software stack and can be silently bypassed by a sufficiently privileged adversary such as a compromised cloud operator. Confidential computing (CC) offers a hardware-rooted alternative: Trusted Execution Environments (TEEs) isolate agent code and data from privileged system software, while remote attestation enables verifiable trust across distributed deployments. This survey synthesizes the design space in four parts: (i) a unified taxonomy of six TEE platforms (Intel SGX, Intel TDX, AMD SEV-SNP, ARM TrustZone, ARM CCA, and NVIDIA H100 CC) covering deployment roles and performance tradeoffs; (ii) an agent-centric threat model spanning perception, planning, memory, action, and coordination layers mapped to nine security goals; (iii) a comparative survey of CC-based defenses distinguishing findings that transfer from single-call inference versus what requires new agentic designs; and (iv) six open challenges including compound attestation for multi-hop agent chains and GPU-TEE performance at LLM scale. While several hardware trust primitives appear mature enough for targeted deployments, no broadly established end-to-end framework yet binds them into a coherent security substrate for production agentic AI. 

---
# Human-Provenance Verification should be Treated as Labor Infrastructure in AI-Saturated Markets 

**Authors**: Erin McGurk, David Khachaturov  

**Link**: [PDF](https://arxiv.org/pdf/2605.03210)  

**Abstract**: We argue that AI-saturated markets are likely to create Veblen-good premiums, which we term human-provenance premiums, for verified human presence, and hence AI governance should treat human-provenance verification as labor infrastructure. Generative and agentic AI systems lower the cost of many standardized cognitive, creative, and coordination tasks, weakening the scarcity premiums that have supported much middle-tier knowledge work. We argue that this pressure may produce an asymmetric barbell-shaped structure of value capture in advanced economies: high-volume synthetic production controlled by owners of AI infrastructure at one pole, and scarce, high-status human labor valued for verified human presence at the other.
We advance three claims. First, AI compresses the value of standardized middle-tier labor by making good-enough synthetic substitutes scalable at low marginal cost, hollowing out the middle of the skill distribution currently categorized by knowledge work. Second, this compression reallocates demand for human labor toward work valued for its visible human character. We term this performative humanity and distinguish three forms of labor: relational presence, aesthetic provenance, and accountability. Third, as these premiums depend on credible verification, AI governance should treat human-provenance systems as labor infrastructure rather than as luxury authenticity labels.
To evaluate hybrid human-AI work, we propose constitutive human presence as the relevant standard: human labor retains premium value when human judgment, attention, accountability, authorship, or relational participation is not incidental to the output but constitutive of what is being purchased. 

---
# From Knowledge to Action: Outcomes of the 2025 Large Language Model (LLM) Hackathon for Applications in Materials Science and Chemistry 

**Authors**: Aritra Roy, Kevin Shen, Andrew MacBride, Awwal Oladipupo, Mudassra Taskeen, Wojtek Treyde, Ruaa A. E. A. Abakar, Ahmad D. Abbas, Elsayed Abdelfatah, Abbas A. Abdullahi, Seham S. Abyah, Chahd Rahyl Adjmi, Fariha Agbere, Savyasanchi Aggarwal, Muhammad Ahmed, Tasnim Ahmed, Motasem Ajlouni, Mattias Akke, Hussein AlAdwan, Anwaar S. Alazani, Zahra A. Alharbi, Wajd A. Aljulyhi, Mohammed A. AlKubaish, Fatima A. Almahri, Sayed A. Almohri, David Obeh Alobo, Mohammed Alouni, Azizah S. Alqahtani, Omar Alsaigh, Husain Althagafi, Md. Aqib Aman, Lena Ara, Arifin, Ignacio Arretche, Abdulaziz Ashy, Syeda A. Asim, Amro Aswad, Adeel Atta, Sören Auer, Abdullah al Azmi, Toheeb Balogun, Suvo Banik, Viktoriia Baibakova, Shakira A. Baksh, Neus G. Bastús, Christina J. Bayard, Adib Bazgir, Louis Beal, Lejla Biberić, Wahid Billah, Ankita Biswas, Joshua Bocarsly, Montassar T. Bouzidi, Esma B. Boydas, Youssef Briki, Cailin Buchanan, Mauricio Cafiero, Damien Caliste, Yi Cao, Rafael E. Castañeda, Sruthy K. Chandy, Benjamin Charmes, Shayantan Chaudhuri, Yiming Chen, Alexander Chen, Jieneng Chen, Min-Hsueh Chiu, Defne Circi, Cinthya H. Contreras, Yoann Cure, Nathan Daelman, Roshini Dantuluri, Thomas Davy, William Dawson, Leonid Didukh, Rui Ding, Aminu R. Doguwa, Claudia Draxl, Sathya Edamadaka, Oulaya Elargab, Christina Ertural, Matthew L. Evans, Edvin Fako, Hossam Farag, Nur A. Fathurrahman, Merve Fedai, Rodrigo P. Ferreira, Giuseppe Fisicaro, Thomas Frank, Sasi K. Gaddipati, Abhijeet Gangan, Jennifer Garland, James Garrick, Luigi Genovese, Maryam Ghadrdran, Sandip Giri, Maxime Goulet, Jeremy Goumaz, Sara U. Gracia, Jacob Graham  

**Link**: [PDF](https://arxiv.org/pdf/2605.03205)  

**Abstract**: Large language models (LLMs) are rapidly changing how researchers in materials science and chemistry discover, organize, and act on scientific knowledge. This paper analyzes a broad set of community-developed LLM applications in an effort to identify emerging patterns in how these systems can be used across the scientific research lifecycle. We organize the projects into two complementary categories: Knowledge Infrastructure, systems that structure, retrieve, synthesize, and validate scientific information; and Action Systems, systems that execute, coordinate, or automate scientific work across computational and experimental environments. The submissions reveal a shift from single-purpose LLM tools toward integrated, multi-agent workflows that combine retrieval, reasoning, tool use, and domain-specific validation. Prominent themes include retrieval-augmented generation as grounding infrastructure, persistent structured knowledge representations, multimodal and multilingual scientific inputs, and early progress toward laboratory-integrated closed-loop systems. Together, these results suggest that LLMs are evolving from general-purpose assistants into composable infrastructure for scientific reasoning and action. This work provides a community snapshot of that transition and a practical taxonomy for understanding emerging LLM-enabled workflows in materials science and chemistry. 

---
# Global and Local Topology-Aware Attention with Persistent Homology and Euler Biases for Time-Series Forecasting 

**Authors**: Usef Faghihi, Amir Saki  

**Link**: [PDF](https://arxiv.org/pdf/2605.03163)  

**Abstract**: Scientific time series often encode predictive geometric structure, including connectivity, cycles, shell-like geometry, directional changes, and nonlinear neighborhoods, that standard dot-product attention does not explicitly represent. We introduce a topology-aware attention framework that adds such structure to attention logits using persistent homology (H0-H2), anchored Euler characteristic transforms, and kernel-Hilbert channels. A validation-gated local residual captures local topological signals, including a Zeng-style local H0 component, only when held-out validation data support the correction. Exact Vietoris-Rips computations and smooth topological surrogates are evaluated under a no-leakage protocol with train-only calibration, validation-only selection, and test-only reporting.
We evaluate guarded topology-aware variants across three architecture families: lightweight attention/Ridge, PatchTSTForRegression, and TimeSeriesTransformerForPrediction. Experiments include synthetic benchmarks isolating higher-order topology and real datasets covering CO2, S&P 500 return-window geometry, and NASA IMS bearing degradation. The audit uses matched paired comparisons across seven dataset units, three random seeds, and three chronological splits, giving 63 paired units per architecture and 189 paired units overall. Topology-aware models show positive paired effects when geometry is predictive, with heterogeneous magnitude across datasets and architectures. Lightweight attention/Ridge improves in 46 of 63 units, with mean relative RMSE reduction of 12.5% and paired randomization p=7.2e-4; PatchTST improves in 33 units and retains the baseline in 20 units, with 23.5% reduction and p=3.5e-5; and TimeSeriesTransformer improves in 47 units, with 47.8% reduction and p<1e-4. The results support topology as a validation-selected, architecture-compatible inductive bias. 

---
# Pact: A Choreographic Language for Agentic Ecosystems 

**Authors**: Kiran Gopinathan, Jack Feser, Michelangelo Naim, Zenna Tavares, Eli Bingham  

**Link**: [PDF](https://arxiv.org/pdf/2605.03143)  

**Abstract**: Recent advances in large language models have led to the rise of software systems (i.e. agents) that execute with increasing autonomy on behalf of users in open, multi-party settings, interacting with untrusted counterparts and managing private information. Choreographic programming offers correct-by-construction protocol-design for such settings, but assumes cooperative participants -- it has no notion of agent self-interest, that is, why an agent will follow a protocol.
In this talk we introduce Pact, a choreographic language extended with operations to describe agent choices and preferences, drawing from the rich literature of game theory. Every Pact protocol maps to a formal game, allowing protocol designers to reason about game-theoretic properties of their protocols, such as solving for decision policies. We present Pact's design and a preliminary implementation -- a bounded-rational solver that computes decision policies over Pact protocols -- and findings from applying this language to multi-party coordination with self-interested agentic participants. 

---
# PIIGuard: Mitigating PII Harvesting under Adversarial Sanitization 

**Authors**: Mingshuo Liu, Yiwei Zha, Min Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.03129)  

**Abstract**: Browsing-enabled LLM assistants can fetch webpages and answer contact-seeking queries, creating a practical channel for scraping contact-style personally identifiable information (PII) from public pages. Many prior defenses are deployed at the model, service, or agent layer rather than at the webpage itself, leaving ordinary page owners with limited deployable options. We present PIIGuard, a webpage-level defense that repurposes indirect prompt injection as a protective mechanism: the page owner embeds optimized hidden HTML fragments that steer the model away from verbatim or reconstructible disclosure of contact PII. PIIGuard searches over fragment text and insertion position using rule-based leakage scoring, evolutionary mutation, and final judge-based recoverability assessment. In direct-HTML evaluation on three target models (GPT-5.4-nano, Claude-haiku-4.5, and DeepSeek-chat(latest v3.2)), PIIGuard achieves at least 97.0% defense success rate under both rule-based and judge-based leakage evaluation, often reaching 100.0%, while preserving benign same-page QA utility. We further evaluate two harder settings: public-URL browsing and attacker-side LLM sanitization of fetched webpage. These results show that page-side defensive fragments can remain effective in deployment for some model-position pairs, but robustness varies substantially across browsing interfaces and sanitizer prompts. Overall, PIIGuard demonstrates that page owners can use page-side fragments as a practical mitigation for web-grounded PII leakage. 

---
# ARISE: A Repository-level Graph Representation and Toolset for Agentic Fault Localization and Program Repair 

**Authors**: Shahd Seddik, Fatemeh Fard  

**Link**: [PDF](https://arxiv.org/pdf/2605.03117)  

**Abstract**: Repository-level fault localization (FL) and automated program repair (APR) require an agent to identify the relevant code units across files, follow call and data dependencies, and generate a valid patch. Existing graph-based systems provide structural representations of repositories (files, classes, functions and their relationships) but do not model how variable values flow within procedures, leaving agents without the semantic precision needed for function- and line-level localization. We present ARISE (Agentic Repository-level Issue Solving Engine), which augments an LLM-based agent with a multi-granularity program graph that extends structural relationships down to statement-level nodes connected by intra-procedural definition-use edges. ARISE exposes this graph through a three-tier tool API, which brings data-flow slicing as a first-class, queryable agent primitive that allows the model to trace, in a single call, which statements define or consume a variable of interest. We evaluate on SWE-bench Lite (300 real GitHub issues, 11 Python repositories) using Qwen2.5-Coder-32B-Instruct as the backbone. Compared to the unmodified SWE-agent baseline, ARISE improves Function Recall@1 by 17.0 points and Line Recall@1 by 15.0 points. These localization gains translate directly into repair success, with ARISE achieving 22.0% Pass@1 (66/300), a 4.7 percentage-point improvement over SWE-agent. Controlled ablations confirm that the improvement is driven by the data-flow graph rather than the tool schema, and that large code models consume structured slice output directly without requiring a natural-language summarization layer. The graph builder and slicing API are designed as a framework-agnostic, drop-in toolset for future APR research. 

---
# Cascade Token Selection for Transformer Attention Acceleration 

**Authors**: Stephen J. Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2605.03110)  

**Abstract**: A method is presented for reducing the cost of representative token selection in transformer attention layers by exploiting the coherence of the representative set across depth. Activation Decorrelation Attention (ADA) selects $r \ll T$ representative tokens at each layer via a Gram threshold and computes attention on the compressed $r \times r$ problem, but the selection requires a $T \times T$ Gram matrix at every layer. The cascade mechanism introduced here inherits the representative set from layer $l$ to layer $l+1$, validates it via a $(T - r) \times r$ cross-Gram computation, and updates it with a small number of additions and removals. The cost of the selection step drops from $O(T^2 d)$ to $O(T r d)$ per layer. Validation on three model families (GPT-2 124M, GPT-J 6B, OPT 6.7B) on AMD MI300X demonstrates Gram operation savings of $22\%$ to $63\%$ with mean Jaccard overlap of $0.83$ to $0.94$ between consecutive layers. The cascade reveals that the set of informative tokens is a structural property of the input that propagates coherently through the depth of the network: the same tokens carry the non-redundant information at layer $l$ and at layer $l+1$. 

---
# Gated Subspace Inference for Transformer Acceleration 

**Authors**: Stephen J. Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2605.03109)  

**Abstract**: A method is presented for accelerating inference in transformer language models by exploiting the low effective rank of the token activation manifold at each layer. The method decomposes each activation vector into a subspace component and a residual, computes the linear-layer output on the subspace component via a cached low-rank weight image at reduced memory bandwidth, and applies a per-token gate that determines whether the residual correction is computed or skipped. The gate ensures that the output distribution is preserved to within a controllable tolerance. Validation on three model families (GPT-2 124M, GPT-J 6B, OPT 6.7B) on AMD MI300X demonstrates effective speedups of 3.0x to 10.5x on linear-layer weight reads with perplexity ratios below 1.00 and top-1 token agreement above 98%. The method requires no retraining, no architectural modification, and no approximation of the attention mechanism. At the operating point (k = 256, {\epsilon} = 0.05) on GPT-J 14 6B, the accelerated model produces character-for-character identical output to the baseline. 

---
# MedStruct-S: A Benchmark for Key Discovery, Key-Conditioned QA and Semi-Structured Extraction from OCR Clinical Reports 

**Authors**: Yingyun Li, Yu Wang, Haiyang Qian  

**Link**: [PDF](https://arxiv.org/pdf/2605.03103)  

**Abstract**: Semi-structured information extraction (IE) from OCR-derived clinical reports is crucial for efficiently reconstructing patients' longitudinal medical histories. In practice, this scenario commonly involves three tasks: (i) field-header (key) discovery, (ii) key-conditioned question answering (QA), and (iii) end-to-end key-value pair extraction. However, existing evaluations often under-model two factors: heterogeneous and incompletely known key representations, and OCR-induced noise. This makes it difficult to assess model robustness in real-world settings.
We present MedStruct-S, a benchmark specifically designed to evaluate these tasks under unknown keys and OCR noise. MedStruct-S contains 3,582 annotated real-world clinical report pages. Using MedStruct-S, we benchmark two representative paradigms: encoder-only sequence labeling with post-processing and decoder-only structured generation, covering four encoder-only and five decoder-only models spanning 0.11B to 103B parameters. Our results show that encoder-only models achieve the best performance for non-null-value key-conditioned QA despite being substantially smaller than decoder-only models. When comparing models of similar order of magnitude, encoder-only models still perform better overall. Without controlling for model scale, fine-tuned decoder-only models deliver the strongest overall results. These findings show that the benchmark provides a reliable and practical basis for selecting and comparing models across different semi-structured IE settings. 

---
# From Barrier to Bridge: The Case for AI Data Center/Power Grid Co-Design 

**Authors**: Noman Bashir, Rob Sherwood, Le Xie, Minlan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03090)  

**Abstract**: For over a century, the electric grid has relied on a single statistical assumption: \emph{load diversity}, the principle that the uncorrelated demands of millions of small consumers produce a smooth, predictable aggregate. AI training data centers break that assumption. A single hyperscale training campus can draw power comparable to a mid-sized city, driven by one tightly synchronized job whose demand swings by hundreds of megawatts in seconds. This paper argues that the resulting entanglement of compute and power infrastructure requires a shift from implicit coexistence to explicit co-development between the historically decoupled data center and electric power industries. We introduce the distinct design principles, operational philosophies, and economic incentives of each sector, and show why their cultural and technical misalignment makes coordination difficult. We identify key research directions, from joint capacity planning, multi-timescale control, a compute--power protocol stack, to market innovation, that must be pursued to power the future of AI sustainably and reliably. 

---
# Refining Compositional Diffusion for Reliable Long-Horizon Planning 

**Authors**: Kyowoon Lee, Yunhao Luo, Anh Tong, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.03075)  

**Abstract**: Compositional diffusion planning generates long-horizon trajectories by stitching together overlapping short-horizon segments through score composition. However, when local plan distributions are multimodal, existing compositional methods suffer from mode-averaging, where averaging incompatible local modes leads to plans that are neither locally feasible nor globally coherent. We propose Refining Compositional Diffusion (RCD), a training-free guidance method that steers compositional sampling toward high-density, globally coherent plans. RCD leverages the self-reconstruction error of a pretrained diffusion model as a proxy for the log-density of composed plans, combined with an overlap consistency term that enforces consistency at segment boundaries. We show that the combined guidance concentrates sampling on high-density plans that mitigate mode-averaging. Experiments on challenging long-horizon tasks from OGBench, including locomotion, object manipulation, and pixel-based observations, demonstrate that RCD consistently outperforms existing methods. 

---
# Neuron-Anchored Rule Extraction for Large Language Models via Contrastive Hierarchical Ablation 

**Authors**: Francesco Sovrano, Gabriele Dominici, Marc Langheinrich  

**Link**: [PDF](https://arxiv.org/pdf/2605.03058)  

**Abstract**: A key goal of explainable AI (XAI) is to express the decision logic of large language models (LLMs) in symbolic form and link it to internal mechanisms. Global rule-extraction methods typically learn symbolic surrogates without grounding rules in model circuitry, while mechanistic interpretability can connect behaviors to neuron sets but often depends on hand-crafted hypotheses and expensive neuron-level interventions. We introduce MechaRule, a pipeline that grounds rule extraction in LLM circuits by efficiently localizing sparse neurons called agonists, whose activation neutralization disrupts rule-related behaviors. MechaRule rests on two empirical observations. First, within a fixed baseline/flip regime, sparse agonist effects can be approximately monotone and saturating: a few dominant neuron activations can overtop weaker ones at coarse scales, while overlapping neurons flip many of the same examples. This motivates viewing localization as adaptive group testing driven by a regime-conditional strength predicate with confidence-guided conservative pruning, yielding Theta(k log(N/k) + k) interventions over N candidates when k << N neurons are agonists under the monotone-overtopping abstraction. Second, agonists emerge more reliably when ablations are verified through data splits aligned with close-to-faithful rule behavior; spectral splits remain a useful rule-free fallback, while unfaithful splits degrade localization. Empirically, overtopping appears mainly in learned, task-aligned regimes: on arithmetic and jailbreak tasks across Qwen2 and GPT-J, MechaRule recalls 96.8% of high-effect brute-force agonists in completed comparisons, and suppressing localized agonists reduces arithmetic accuracy and jailbreak success by up to 71.1% and 8.8%, respectively. 

---
# ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration 

**Authors**: Ruofeng Yang, Yongcan Li, Shuai Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.03042)  

**Abstract**: This report describes ARIS (Auto-Research-in-sleep), an open-source research harness for autonomous research, including its architecture, assurance mechanisms, and early deployment experience. The performance of agent systems built on LLMs depends on both the model weights and the harness around them, which governs what information to store, retrieve, and present to the model. For long-horizon research workflows, the central failure mode is not a visible breakdown but a plausible unsupported success: a long-running agent can produce claims whose evidential support is incomplete, misreported, or silently inherited from the executor's framing. Therefore, we present ARIS as a research harness that coordinates machine-learning research workflows through cross-model adversarial collaboration as a default configuration: an executor model drives forward progress while a reviewer from a different model family is recommended to critique intermediate artifacts and request revisions. ARIS has three architectural layers. The execution layer provides more than 65 reusable Markdown-defined skills, model integrations via MCP, a persistent research wiki for iterative reuse of prior findings, and deterministic figure generation. The orchestration layer coordinates five end-to-end workflows with adjustable effort settings and configurable routing to reviewer models. The assurance layer includes a three-stage process for checking whether experimental claims are supported by evidence: integrity verification, result-to-claim mapping, and claim auditing that cross-checks manuscript statements against the claim ledger and raw evidence, as well as a five-pass scientific-editing pipeline, mathematical-proof checks, and visual inspection of the rendered PDF. A prototype self-improvement loop records research traces and proposes harness improvements that are adopted only after reviewer approval. 

---
# Mixed-Precision Information Bottlenecks for On-Device Trait-State Disentanglement in Bipolar Agitation Detection 

**Authors**: Joydeep Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2605.03039)  

**Abstract**: Continuous monitoring of bipolar disorder agitation via voice biomarkers requires disentangling stable speaker traits from volatile affective states on resource-constrained edge devices. We introduce MP-IB, the first framework to treat mixed-precision quantization as an information bottleneck for clinical trait-state separation. The core insight is that numerical precision itself controls capacity: an FP16 trait head (1,024 bits) encodes speaker identity, while an INT4 state head (128 bits) captures agitation, yielding 8x information asymmetry without adversarial training. We augment this with Dynamic Precision Scheduling and Multi-Scale Temporal Fusion. On Bridge2AI-Voice (N=833, 4 sessions/participant, strict speaker-independent CV), MP-IB achieves rho = 0.117 (95\% CI: [0.089, 0.145], p=0.003 vs. chance), outperforming 94M-parameter WavLM-Adapter with in-domain SSL continuation (rho = -0.042), beta VAE disentanglement (rho = 0.089), and hand-crafted prosody (rho = 0.031) by 2.8--15.9 points absolute. Zero-shot transfer to CREMA-D achieves AUC=0.817. Identity leakage is suppressed to near-random (EER=0.42, MIA-AUC=0.52). End-to-end latency is 23.4 ms with a 617 KB footprint, enabling real-time monitoring on sub 20 dollar devices. 

---
# Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges 

**Authors**: Eitan Kosman, Gabriele Serussi, Chaim Basking  

**Link**: [PDF](https://arxiv.org/pdf/2605.02973)  

**Abstract**: Modality translation is inherently under-constrained, as multiple cross-modal mappings may yield the same marginals. Recent work has shown that diffusion bridges are effective for this task. However, most existing approaches rely on fully paired datasets, thereby imposing a single data-driven constraint. We propose a diffusion-bridge framework that characterizes the space of admissible solutions and restricts it via alignment constraints, treating paired supervision as an optional heuristic rather than a prerequisite. We validate our method on synthetic and real modality translation benchmarks across unpaired, semi-paired, and paired regimes, showing consistent performance across supervision levels. Notably, \textbf{it achieves near fully-paired quality with a substantial relaxation in pairing requirements, and remaining applicable in the unpaired regime}. These results highlight diffusion bridges as a flexible foundation for modality translation beyond fully paired data. 

---
# Multilingual Safety Alignment via Self-Distillation 

**Authors**: Ruiyang Qin, Qingzhuo Wang, Dongrui Liu, Qiang Li, Zhihua Wei, Wen Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02971)  

**Abstract**: Large language models (LLMs) exhibit severe multilingual safety misalignment: they possess strong safeguards in high-resource languages but remain highly vulnerable to jailbreak attacks in low-resource languages. Current safety alignment methods generally rely on high-quality response data for each target language, which is expensive and difficult to generate. In this paper, we propose a cross-lingual safeguard transfer framework named Multilingual Self-Distillation (MSD). This framework transfers an LLM's inherent safety capabilities from high-resource (e.g., English) to low-resource (e.g., Javanese) languages, overcoming the need for response data in any language. Our framework is flexible and can be integrated with different self-distillation strategies. Specifically, we implement two concrete methods -- on-policy MSD and off-policy MSD -- both of which enable effective cross-lingual safety transfer using only multilingual queries. Furthermore, we propose Dual-Perspective Safety Weighting (DPSW), a divergence measure to optimize the distillation objective. By jointly considering the perspectives of both the teacher and the student, DPSW adaptively increases the penalty weights on safety-critical tokens while reducing the weights on non-critical tokens. Extensive experiments on representative LLMs across diverse multilingual jailbreak and utility benchmarks demonstrate that our method consistently achieves superior multilingual safety performance. Notably, it generalizes effectively to more challenging datasets and unseen languages while preserving the model's general capabilities. 

---
# Decompose to Understand, Fuse to Detect: Frequency-Decoupled Anomaly Detection for Encrypted Network Traffic 

**Authors**: Xinglin Lian, Chengtai Cao, Ting Zhong, Yong Wang, Kai Chen, Fan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.02970)  

**Abstract**: Network traffic anomaly detection represents a critical cybersecurity task, yet widespread encryption makes this task increasingly challenging. In response, image-based methods that model traffic as visual patterns have emerged as the dominant approach. However, this work pioneers the identification of a pervasive ``full-frequency'' characteristic and an associated limitation termed ``spectral mismatch'' within this paradigm. Specifically, while encrypted traffic exhibits prominent high-frequency components, mainstream reconstruction methods demonstrate an inherent bias toward learning low-frequency information. This fundamental mismatch results in incomplete representations that consequently degrade anomaly detection performance. To address this challenge, we propose FreeUp, a novel frequency-decoupled framework designed explicitly for encrypted traffic analysis. FreeUp decomposes traffic data into distinct low- and high-frequency bands, processing them through separate, dedicated branches along with a customized training strategy that ensures stable and independent frequency-specific learning. Furthermore, recognizing that simple reconstruction error proves inadequate for evaluating dual-branch architectures, we introduce an uncertainty-inspired fusion scoring mechanism. This mechanism quantifies the reconstruction uncertainty of the frequency-specific branches and dynamically integrates their outputs, yielding a more comprehensive and reliable anomaly score. Extensive experiments across multiple benchmarks demonstrate that FreeUp consistently outperforms state-of-the-art baselines. The code is available at this https URL. 

---
# Finite-Size Gradient Transport in Large Language Model Pretraining: From Cascade Size to Intensive Transport Efficiency 

**Authors**: Ping Wang, Yan-Qi Du  

**Link**: [PDF](https://arxiv.org/pdf/2605.02968)  

**Abstract**: We introduce a finite-size gradient-transport framework for real language-model training, based on five observables $(D,z,\beta,\delta,v_{\mathrm{rel}})$ that separate cascade size, duration, absolute transport, and intensive transport efficiency. We analyze direct raw-gradient measurements from Pico-LM across four scales and 125 aligned steps, together with a five-scale Pythia companion dataset built from 153 aligned checkpoint-difference update fields. The same algebraic closure holds in both families, and both share a near-unity cascade-size backbone, but they occupy distinct transport regimes: Pico-LM shows positive duration scaling and negative intensive-efficiency scaling, whereas Pythia remains near the $D=1$ baseline with only weak positive efficiency scale dependence. Randomized-field controls give nearly matched null floors in the intensive and duration channels, indicating that the contrast reflects different real departures from a shared null skeleton rather than different null calibrations. The families also differ in stepwise power-law compressibility: Pico-LM retains clean duration and efficiency power laws, whereas Pythia preserves the size backbone but shows weaker one-slope compressibility in those channels. External performance associations are correspondingly channel-level, carried mainly by $v_{\mathrm{rel}}$ and normalized cascade duration, while $D(t)$ acts as a shared size backbone without a significant exponent-level performance association. These results support a reusable transport measurement framework without claiming a universal fixed point or a first-principles derivation of neural scaling laws. 

---
# AutoRAGTuner: A Declarative Framework for Automatic Optimization of RAG Pipelines 

**Authors**: Xintan Zeng, Yongchao Liu, Yice Luo, Jiajun Zhen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02967)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances LLMs, but performance is highly sensitive to complex architecture designs and hyper-parameter configurations, which currently rely on inefficient manual tuning. We present AutoRAGTuner, a declarative, configuration-driven framework that automates the RAG life cycle: construction, execution,evaluation, and optimization. AutoRAGTuner employs a modular architecture to decouple pipeline stages through a component registration mechanism. To unify heterogeneous data, we introduce the Domain-Element Model (DEM), representing objects as atomic elements with bidirectional pointers to support nodes, edges, and hyperedges. Furthermore, AutoRAGTuner integrates an adaptive Bayesian optimization engine for end-to-end hyper-parameter tuning. Experimental results demonstrate AutoRAGTuner's architectural generality: across diverse RAG pipelines, ranging from vanilla to graph-based, the framework consistently outperforms default baselines. Notably, AutoRAGTuner significantly mitigates engineering overhead, where its declarative configuration language enables a up to 95\% reduction in code churn for architectural adjustments. Overall, AutoRAGTuner provides a systematically optimizable foundation for building evolvable and reusable RAG systems. 

---
# Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use 

**Authors**: Kunvar Thaman  

**Link**: [PDF](https://arxiv.org/pdf/2605.02964)  

**Abstract**: Reinforcement learning (RL) trained language model agents with tool access are increasingly deployed in coding assistants, research tools, and autonomous systems. We introduce the Reward Hacking Benchmark (RHB), a suite of multi-step tasks requiring sequential tool operations with naturalistic shortcut opportunities such as skipping verification steps, inferring answers from task-adjacent metadata, or tampering with evaluation-relevant functions. RHB supports independent and chained task regimes, where chain length acts as a proxy for longer-horizon agent behavior.
We evaluate 13 frontier models from OpenAI, Anthropic, Google, and DeepSeek. Exploit rates range from 0% (Claude Sonnet 4.5) to 13.9% (DeepSeek-R1-Zero), varying sharply by post-training style. A controlled sibling comparison (DeepSeek-V3 vs. DeepSeek-R1-Zero) shows RL post-training is associated with substantially higher reward hacking (0.6% vs. 13.9%), with consistent gaps across all four task families. We identify six exploit categories and find that 72% of reward hacking episodes include explicit chain-of-thought rationale, suggesting models often frame exploits as legitimate problem-solving.
Simple environmental hardening reduces exploit rates by 5.7 percentage points (87.7% relative) without degrading task success. Models with near-zero exploit rates on standard tasks show elevated rates on harder variants, suggesting that production-aligned post-training appears to suppress reward hacking only below a complexity threshold where honest solutions remain tractable. 

---
# Analytic Bridge Diffusions for Controlled Path Generation 

**Authors**: Michael Chertkov  

**Link**: [PDF](https://arxiv.org/pdf/2605.02961)  

**Abstract**: Most modern bridge-diffusion methods achieve finite-time transport by specifying an interpolation, Schrödinger-bridge, or stochastic-control objective and then learning the associated score or drift field with a neural network. In contrast, we identify a restricted but sufficiently broad and analytically solvable class in which the score, intermediate marginals, and protocol gradients are available in closed form without inner stochastic simulation loops and without neural networks in the optimization loop. We recast the classical linear--quadratic--Gaussian (LQG) stochastic-control structure as a transport problem of the Path Integral Diffusion (PID) type. In classical LQG control, linear dynamics, Gaussian noise, and quadratic costs lead to Riccati equations and closed-form optimal feedback. In LQ-GM-PID, we retain the linear--quadratic stochastic-control backbone, but replace terminal state regulation by a prescribed terminal probability density and allow both the initial and terminal laws to be Gaussian Mixtures (GM).
Moreover, LQ-GM-PID turns bridge diffusion from a tool for terminal target matching alone into a tool for path shaping. We demonstrate this on a 2D corridor task, a 2D multi-entrance transport task, and a high-dimensional scaling study with $d=32$ and $M=16$ Gaussian-mixture terminal modes, all with sub-50\,ms analytic precompute on a laptop. We position LQ-GM-PID as an analytically solvable reference model for the state-of-the-art neural bridge-diffusion and generative-transport methods: a controlled setting in which neural approximations, score estimates, path-shaping objectives, and protocol-learning procedures can be tested against exact quantities. 

---
# Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection 

**Authors**: Xulin Hu, Che Wang, Wei Yang Bryan Lim, Jianbo Gao, Zhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02958)  

**Abstract**: Representation Engineering typically relies on static refusal vectors derived from terminal representations. We move beyond this paradigm, demonstrating that refusal is a dynamic and sparse process rather than a localized outcome. Using Causal Tracing, we uncover the Refusal Trajectory-a persistent upstream signature that remains intact even when adversarial attacks (e.g., GCG) suppress terminal signals. Leveraging this, we propose SALO (Sparse Activation Localization Operator), an inference-time detector designed to capture these latent patterns. SALO effectively recovers defense capabilities against forced-decoding attacks, improving detection rates from ~0% to >90% where methods relying on terminal states perform poorly. 

---
# Kernel Affine Hull Machines for Compute-Efficient Query-Side Semantic Encoding 

**Authors**: Mohit Kumar, Somayeh Kargaran, Bernhard A. Moser, Manuela Geiß  

**Link**: [PDF](https://arxiv.org/pdf/2605.02950)  

**Abstract**: Transformer-based semantic retrieval is highly effective, yet in many deployments the dominant cost lies in online query encoding rather than corpus indexing. We study the fixed-teacher query-adaptation problem and ask whether repeated neural inference can be replaced by a lightweight, analytically explicit estimator without degrading decision-relevant retrieval quality. We propose Kernel Affine Hull Machines (KAHMs), which map inexpensive lexical features into a frozen semantic embedding space by estimating prototype-mixture weights in a rigorously specified RKHS and refining prototypes via normalized least-mean-squares, yielding a transparent decomposition of encoding error into posterior-approximation, generalization, and teacher-noise components. On a controlled Austrian-law benchmark (5,000 queries; 84 laws; 10,762 units), KAHM attains the strongest teacher-space reconstruction among matched learned adapters (MSE 0.000091, R^2 0.9071, cosine 0.9536) and consistently leads rank-sensitive metrics, including mean reciprocal rank at 20 (MRR@20, the average inverse rank of the first relevant result within the top 20), Hit rate at 20 (Hit@20, the fraction of queries with at least one relevant result in the top 20), and Top-1 accuracy (the fraction of queries whose correct item is ranked first), with scores of 0.504, 0.694, and 0.411, respectively. It also reduces per-query latency by a factor of 8.5 relative to direct transformer encoding. These results demonstrate that, in fixed-teacher regimes, lightweight geometric estimators can substitute for online neural encoding, preserving retrieval performance while substantially improving efficiency and interpretability. 

---
# AsymK-Talker: Real-Time and Long-Horizon Talking Head Generation via Asymmetric Kernel Distillation 

**Authors**: Yuxin Lu, Qian Qiao, Jiayang Sun, Min Cao, Guibo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02948)  

**Abstract**: Recent advances in diffusion models have markedly enhanced the visual fidelity of audio-driven talking head generation. Nevertheless, existing methods are constrained by three critical limitations: causal inefficiency that impedes real-time inference, incompatibility with temporally coherent conditioning, and progressive drift over long-horizon generation, collectively hindering their deployment in real-time applications. To overcome these challenges, we introduce AsymK-Talker, a novel diffusion-distillation method designed for real-time and long-horizon talking head generation. AsymK-Talker comprises three key components: (1) Kernel-Conditioned Loop Generation (KCLG), a causal, chunk-wise generation paradigm that leverages motion kernels to enable temporally consistent propagation; (2) Temporal Reference Encoding (TRE), which converts a static identity reference into a time-aware latent representation to enhance audio-visual synchronization; and (3) Asymmetric Kernel Distillation (AKD), a teacher-student distillation framework wherein the teacher model conditions on ground-truth motion kernels for supervision, while the student learns to generate from generated kernels, thereby ensuring robustness during extended generation sequences. AsymK-Talker achieves promising results on both visual fidelity and lip synchronization metrics. 

---
# Predicting Euler Characteristics and Constructing Topological Structure Using Machine Learning Techniques 

**Authors**: Gyunghun Yu, Seong Min Park, Han Gyu Yoon, Tae Jung Moon, Jun Woo Choi, Hee Young Kwon, Changyeon Won  

**Link**: [PDF](https://arxiv.org/pdf/2605.02947)  

**Abstract**: This study proposes a novel approach to extract topological properties, specifically the Euler characteristic, from input images using neural networks without relying on large pre-existing datasets but with a single geometric image. Inspired by solid-state physics, where topological properties of magnetic structures are derived from spin field analysis, our model generates a unit vector field from an image, interpreted as a spin configuration. The Euler characteristic is then predicted by computing the skyrmion number of this generated spin configuration. Remarkably, the network learns to construct chiral magnetic textures without access to ground-truth chiral spin configurations, relying instead on only a single, simple geometric image and the straightforward skyrmion number computation. Furthermore, spin configurations generated by independently trained networks can be non-unique due to inherent degrees of freedom. To constrain these degrees of freedom and further refine the spin configuration, we incorporate a magnetic Hamiltonian, comprising exchange interaction, Dzyaloshinskii-Moriya (DM) interaction, and anisotropy, as an additional, physics-informed loss function. We validate the model's efficacy on complex geometrical shapes and demonstrate its applicability to practical tasks. 

---
# RouteHijack: Routing-Aware Attack on Mixture-of-Experts LLMs 

**Authors**: Zhiyuan Xu, Joseph Gardiner, Sana Belguith, Lichao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02946)  

**Abstract**: Safety alignment is critical for the responsible deployment of large language models (LLMs). As Mixture-of-Experts (MoE) architectures are increasingly adopted to scale model capacity, understanding their safety robustness becomes essential. Existing adversarial attacks, however, have notable limitations. Prompt-based jailbreaks rely on heuristic search and transfer poorly, model intervention methods require privileged access to internal representations, and optimization-based input attacks remain output-centric and are fundamentally limited to MoE models due to the non-differentiable routing mechanism.
In this paper, we present RouteHijack, a routing-aware jailbreak for MoE LLMs. Our key insight is that safety behavior is concentrated in a small subset of experts, creating an opportunity to steer model behavior by influencing routing decisions through input optimization. Building on this observation, RouteHijack first performs response-driven expert localization to identify safety-critical and harmful experts by contrasting activations under safe refusals and harmful completions. It then constructs adversarial suffixes with a routing-aware objective that suppresses safety experts, promotes harmful experts, and prevents early-stage refusal during generation. At inference time, the optimized suffix is appended to a malicious prompt, requiring only input access. Across seven MoE LLMs, RouteHijack achieves a 69.3\% average attack success rate (ASR), outperforming prior optimization-based attack by $3.2\times$. RouteHijack also transfers zero-shot across five sibling MoE variants, raising average ASR from 27.7\% to 61.2\%, and further generalizes to three MoE-based VLMs, increasing average ASR from 2.47\% to 38.7\%. These findings expose a fundamental vulnerability in sparse expert architectures and highlight the need for defenses beyond output-level alignment. 

---
# Exploring Pass-Rate Reward in Reinforcement Learning for Code Generation 

**Authors**: Xin-Ye Li, Ren-Biao Liu, Yun-Ji Zhang, Hui Sun, Zheng Xie, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.02944)  

**Abstract**: Reinforcement learning (RL) from unit-test feedback has become a standard post-training recipe for improving large language models (LLMs) on code generation. However, the pass-all-tests binary reward can be sparse, yielding no learning signal on challenging problems where none of the sampled solutions passes all tests. A common remedy is to use the test-case pass rate as a surrogate reward.
In this work, we study pass-rate rewards in critic-free RL for code generation (e.g., GRPO and RLOO) and report a consistent pattern across base models and algorithms: despite alleviating reward sparsity, pass-rate rewards do not reliably improve final performance over binary rewards in rigorous controlled experiments.
To understand this discrepancy, we analyze reward density and the resulting gradient directions. We find that pass-rate rewards are denser, but the induced gradient updates do not consistently move probability mass toward full-pass solutions. This arises because test-case pass rate is a miscalibrated surrogate for progress toward full correctness, and partial-pass solutions within the same group can induce conflicting gradient directions that cancel out.
Overall, our results suggest that, in critic-free RL, pass-rate rewards are insufficient to improve code generation and motivate reward designs that better align optimization with the goal of full correctness. 

---
# Healthcare AI GYM for Medical Agents 

**Authors**: Minbyul Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2605.02943)  

**Abstract**: Clinical reasoning demands multi-step interactions -- gathering patient history, ordering tests, interpreting results, and making safe treatment decisions -- yet a unified training environment provides the breadth of clinical domains and specialized tools to train generalizable medical AI agents through reinforcement learning remains elusive. We present a comprehensive empirical study of multi-turn agentic RL for medical AI, built on \gym{}, a gymnasium-compatible environment spanning 10 clinical domains with 3.6K+ tasks, 135 domain-specific tools, and a knowledge base of 828K medical passages. Our analysis reveals that agentic multi-turn structure degrades into verbose single-turn monologues, characterized by monotonic length explosion and a simultaneous erosion of tool-use frequency. We characterize how this collapse, alongside distillation instability, stems from the misalignment of sparse terminal rewards with sequential clinical trajectories. We find that vanilla GRPO achieves strong final accuracy on some benchmarks but suffers from training instability, evidenced by significant oscillations in response length and prolonged convergence periods. To improve training efficiency and stability, we propose Turn-level Truncated On-Policy Distillation (TT-OPD), a self-distillation framework where a gradient-free EMA teacher leverages outcome-privileged information to provide dense, outcome-aware KL regularization at every conversation turn. TT-OPD achieves the best performance on 10 of 18 benchmarks with an average +3.9~pp improvement over the non-RL baseline with faster early convergence, controlled response length, and sustained multi-turn tool use. 

---
# PrismAgent: Illuminating Harm in Memes via a Zero-Shot Interpretable Multi-Agent Framework 

**Authors**: Zihan Ding, Ziyuan Yang, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02940)  

**Abstract**: The rapid spread of memes makes harmful content detection increasingly crucial, as effective identification can curb the circulation of misinformation. However, existing methods rely heavily on high-volume annotated data, which leads to substantial training costs and limited generalization. To address these challenges, we propose PrismAgent, a zero-shot, multi-agent, interpretable framework. PrismAgent conceptualizes this task as a criminal case investigation, employing four specialized agents responsible for the analysis, investigation, prosecution, and judgment stages within a structured collaborative workflow. In the first stage, the analyst agent paraphrases each meme under benevolent and malicious assumptions to probe its underlying intent. The investigator agent then retrieves supporting evidence from an unannotated dataset and constructs contextual interpretations for the meme and its variants. Next, the prosecutor agent performs three independent preliminary judgments by pairing the original meme with each of the three interpretations. Finally, the judge agent deliberates across all evidence to render a final verdict. Moreover, PrismAgent's explicit multi-stage reasoning chain makes the model inherently interpretable, as every intermediate step is explicitly explained rather than only producing a final detection result. Extensive experiments on three public datasets show that PrismAgent significantly outperforms existing zero-shot detection methods. 

---
# From Static Analysis to Audience Dissemination: A Training-Free Multimodal Controversy Detection Multi-Agent Framework 

**Authors**: Zihan Ding, Ziyuan Yang, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02939)  

**Abstract**: Multimodal controversy detection (MCD) identifies controversial content in videos and their associated user comments, to support risk management for social video this http URL research frames MCD as a static representation learning task, where features are directly extracted from videos and their accompanying comments. However, these methods fail to capture the diverse perspectives and evaluations from different audience groups. Inspired by the real-world process of content dissemination among audiences, we propose AuDisAgent, a training-free multi-agent framework that reformulates MCD as a dynamic propagation this http URL framework explicitly models audience dissemination through a structured multi-agent system. First, three specialized Screening Agents (Video Agent, Comment Agent, and Interaction Agent) conduct initial assessments from visual, textual, and cross-modal perspectives, respectively. For samples where the three agents cannot reach a consensus, a Viewing Panel Agent is activated to simulate post-screening discussions among audiences with diverse backgrounds and stances. This mechanism models how different audience groups interpret and react to the same content, uncovering latent controversial content that may emerge during the dissemination process. Finally, an Arbitration Agent renders the final judgment based on the complete reasoning chain from the preceding this http URL addition, to address the "cold-start" scenario where newly released videos have few or no comments, we design a Comment Bootstrapping Strategy that leverages historical public comments from semantically similar videos as the initial comment context. Extensive experiments on a public dataset demonstrate that our framework significantly outperforms existing state-of-the-art (SOTA) methods in both rich-comment and limited-comment scenarios. 

---
# PAMNet: Cycle-aware Phase-Amplitude Modulation Network for Multivariate Time Series Forecasting 

**Authors**: Yingbo Zhou, Yutong Ye, Zhiwei Ling, Shuhao Li, Rui Qian, Jian Xiong, Li Sun, Dejing Dou  

**Link**: [PDF](https://arxiv.org/pdf/2605.02938)  

**Abstract**: Reliable periodic patterns serve as a fundamental basis for accurate multivariate time series forecasting. However, existing methods either implicitly extract periodicity through complex model architectures (e.g., Transformers) with high computational overhead or overlook the intrinsic phase-amplitude coupling when modeling periodic components explicitly. To address these issues, we propose a novel Cycle-aware Phase-Amplitude Modulation Network (PAMNet) that explicitly decomposes periodic patterns into complementary phase and amplitude components. The core innovation lies in its dual-branch modulator, featuring dedicated learnable embeddings for phase positioning and amplitude modulation. The phase branch employs cyclical embeddings to capture phase-dependent mean shifts, while the amplitude branch models intensity variations to adapt to changes in variance. A lightweight modulator with element-wise fusion efficiently combines these components, enabling explicit modeling of their interactions without complex attention mechanisms. Extensive experiments on twelve real-world datasets demonstrate that our method achieves state-of-the-art performance through its novel phase-amplitude decoupling mechanism, offering a new perspective for cyclical modeling in time series forecasting. 

---
# Proteo-R1: Reasoning Foundation Models for De Novo Protein Design 

**Authors**: Fang Wu, Weihao Xuan, Heli Qi, Hanqun Cao, Heng-Jui Chang, Zeqi Zhou, Haokai Zhao, Ma Jian, Carl Ma, Yu-Chi Cheng, Kuan Pang, Xiangru Tang, Zehong Wang, Guanlue Li, Hanchen Wang, Kejun Ying, Pan Lu, Chiho Im, Seungju Han, Peng Xia, Tinson Xu, Yinxi Li, Deyao Zhu, Pheng-Ann Heng, Naoto Yokoya, Masashi Sugiyama, Li Erran Li, Jure Leskovec, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.02937)  

**Abstract**: Deep learning in \emph{de novo} protein design has achieved atomic-level fidelity. However, existing models remain largely non-deliberative: they directly synthesize molecular geometries without explicitly reasoning about which residues or interactions are functionally essential. As a result, design decisions are entangled with continuous sampling dynamics, limiting interpretability, controllability, and systematic reuse of biochemical knowledge. We introduce \textbf{Proteo-R1}, a reasoning-guided protein design framework that explicitly decouples \emph{molecular understanding} from \emph{geometric generation}. Proteo-R1 adopts a dual-expert architecture in which a multimodal large language model (MLLM) serves as an \emph{understanding expert}, analyzing protein sequences, structures, and textual context to identify key functional residues that govern binding and specificity. These residue-level decisions are then passed as hard constraints to a separate diffusion-based \emph{generation expert}, which performs conditional co-design while respecting the fixed interaction anchors. This factorization mirrors how human experts approach molecular engineering: first, reasoning about critical interactions, then optimizing geometry subject to those constraints. By operationalizing reasoning as explicit residue-level commitments rather than latent textual guidance, Proteo-R1 achieves stable, interpretable, and modular integration of LLM reasoning with state-of-the-art geometric generative models. Code, data, and demos are available at this https URL. 

---
# A Universal Space of Brain Dynamics for Unveiling Cognitive Transitions and Individual Differences 

**Authors**: Ronghua Zheng, Chengyuan Qian, Weiyang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.02936)  

**Abstract**: Representing dynamical systems through data-driven universal spaces has proven effective; however, achieving this universality for human brain activity remains a significant challenge, further aggravated by diverse cognitive states and individual subjects. Recognizing that spatial properties reflect physical wiring while temporal properties reflect brain function, we develop Universal Brain Dynamics (UBD) to construct a universal space tailored to brain activity and quantify corresponding dynamics using a model-derived Jacobian matrix. Crucially, we validate UBD's universality by accurately predicting functional magnetic resonance imaging (fMRI) signals (Pearson's r > 0.9) across eight states and 963 subjects in the Human Connectome Project (HCP). Through evaluating resting-state fMRI represented within UBD, we gain insight into how infra-slow fluctuation (ISF) underpins brain activity. Furthermore, we reveal a new perspective on structure-function coupling (SFC) by analyzing the temporal sequence of brain dynamics. Extending UBD to task-evoked states, we derive brain dynamics across various cognitive conditions, elucidating the neural mechanisms driving cognitive transitions at a finer granularity. For individual differences, we compare brain dynamics across subjects to identify the neural underpinnings of these variations. Our findings suggest that synergistically integrating spatial and temporal properties of brain activity establishes a universal space for its unfolding, enabling the precise numerical analysis of underlying neural mechanisms across varying conditions. 

---
# DeRelayL: Sustainable Decentralized Relay Learning 

**Authors**: Haihan Duan, Tengfei Ma, Yuyang Qin, Runhao Zeng, Wei Cai, Victor C. M. Leung, Xiping Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02935)  

**Abstract**: In the era of big data, large-scale machine learning models have revolutionized various fields, driving significant advancements. However, large-scale model training demands high financial and computational resources, which are only affordable by a few technological giants and well-funded institutions. In this case, common users like mobile users, the real creators of valuable data, are often excluded from fully benefiting due to the barriers, while the current methods for accessing large-scale models either limit user ownership or lack sustainability. This growing gap highlights the urgent need for a collaborative model training approach, allowing common users to train and share models. However, existing collaborative model training paradigms, especially federated learning (FL), primarily focus on data privacy and group-based model aggregation. To this end, this paper intends to address this issue by proposing a novel training paradigm named decentralized relay learning (DeRelayL), a sustainable learning system where permissionless participants can contribute to model training in a relay-like manner and share the model. In detail, this paper presents the architecture and workflow of DeRelayL, designs incentive mechanisms to ensure sustainability, and conducts theoretical analysis and numerical simulations to demonstrate its effectiveness. 

---
# Keyword spotting using convolutional neural network for speech recognition in Hindi 

**Authors**: Saru Bharti, Pushparaj Mani Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2605.02928)  

**Abstract**: In this study, we investigate the application of keyword spotting (KWS) in the domain of Hindi speech recognition, utilizing a dataset comprising 40,000 audio samples. With a sampling rate of 44 kHz and an average duration of 1.9 seconds per sample, we focus on developing an efficient on-device KWS system tailored for user-specific queries. Leveraging Convolutional Neural Networks (CNNs) for classification, we employ feature engineering techniques to convert raw audio recordings into Mel Frequency Cepstral Coefficients (MFCCs) as an input for our network. Our experiments encompass various CNN architectures, exploring their efficacy in identifying predefined keywords within the continuous speech stream. Our CNN-based approach achieves a commendable accuracy rate of 91.79% through rigorous evaluation, demonstrating promising performance while ensuring computational efficiency and user-specific customization in Hindi speech recognition. 

---
# Generalization Bounds of Spiking Neural Networks via Rademacher Complexity 

**Authors**: Shao-Qun Zhang, Zhi-Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.02927)  

**Abstract**: Spiking Neural Networks (SNNs) have garnered increasing attention as one of bio-inspired models due to their great potential in neuromorphic computing and sparse computation. Many practical algorithms and techniques have been developed; however, theoretical understandings of the generalization, that is, the extent to which SNNs perform well on unseen data, are far from clear. Recent advances disclosed an excitation-dependent and architecture-related generalization bound such that the Rademacher complexity of SNNs with stochastic firing can be upper bounded by an exponential function relative to the excitation probability and the architecture depth. In this paper, we theoretically investigate the generalization bounds of SNNs with several integration-and-fire schemes via Rademacher complexity. We recognize that the empirical Rademacher complexity of SNNs is close to the SNN configurations, which is exponential to the network depth and the maximum time duration of received spike sequences, superlinear and subquadratic to the network width, polynomial to the parameter norm, inverse-linear to the number of training samples, and independent of the computations within spiking neurons, achieving a more precise rate than conventional studies. Our theoretical results may support the scope of SNN theories and shed some insight into the development of SNNs. 

---
# EvoJail: Evolutionary Diverse Jailbreak Prompt Generation for Large Language Models 

**Authors**: Rui Tang, Kaiyu Xu, Pengsen Cheng, Hao Ren, Haizhou Wang, Shuyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02921)  

**Abstract**: As LLMs continue to shape real-world applications, automated jailbreak generation becomes essential to reveal safety weaknesses and guide model improvement. Existing automatic jailbreak generation methods have not yet fully considered two important aspects: adaptability to evolving safety-finetuned models, which affects their effectiveness on newer model versions, and diversity in generated prompts, which can cause narrow or repetitive attack patterns. To address these issues, we propose EvoJail, an instruction-fusion-driven evolutionary jailbreak generation framework that formalizes jailbreak prompt generation as a multi-objective black-box optimization problem and leverages the principles of evolutionary algorithms to search for jailbreak prompts that can adapt across different model versions and exhibit diverse attack patterns. Specifically, EvoJail integrates jailbreak prompt generation into an iterative evolutionary loop, where at each iteration candidate prompts are evaluated directly against the target model and then selected and varied based on the target model's responses, enabling the generation process to continuously adapt to model updates. To enhance diversity, EvoJail introduces field-aware instruction fusion to construct diverse starting points and incorporates diversity-aware objectives into the evolutionary fitness function, guiding the search toward prompts with richer semantic variation, while further designing multi-level LLM-based mutation operators that modify prompt structures at different granularities to promote structural diversity throughout the evolutionary process. Results demonstrate that EvoJail has stronger adaptability and can achieve over $93\%$ attack success rate and more than $5.6\%$ improvement in diversity metrics over state-of-the-art methods. 

---
# Mitigating the reconstruction-detection trade-off in VAE-based unsupervised anomaly detection 

**Authors**: Agathe Senellart, Maëlys Solal, Stéphanie Allassonnière, Ninon Burgos  

**Link**: [PDF](https://arxiv.org/pdf/2605.02918)  

**Abstract**: Variational autoencoders are widely used for unsupervised anomaly detection. Model selection however remains an open-question: to remain fully unsupervised, hyperparameters are often chosen to minimize the reconstruction error on normal samples. In this paper, we reveal a trade-off between reconstruction quality and anomaly detection among $\beta$-VAE models. Models with constrained latent space reach higher detection metrics but lower reconstruction quality. We also assess the performance variability across random seeds and show it is linked to the distance between normal and abnormal latent distributions. From this analysis, we justify and investigate two methods to mitigate the reconstructiondetection tradeoff: beta-scheduling and the Sparse VAE. The latter especially shows an improvement in detection while maintaining high reconstruction quality. 

---
# PRISM-CTG: A Foundation Model for Cardiotocography Analysis with Multi-View SSL 

**Authors**: Sheng Wong, Ravi Shankar, Beth Albert, Hao Fei, Lin Li, Imane Ben M'Barek, Manu Vatish, Gabriel Davis Jones  

**Link**: [PDF](https://arxiv.org/pdf/2605.02917)  

**Abstract**: Supervised deep learning models for automated CTG analysis are typically constrained by narrowly curated labelled datasets and limited patient cohorts, leaving substantial volumes of physiologically informative clinical recordings untapped. To address this limitation, we propose Physiology-aware Representation Learning via Integrated Self-supervision and Metadata for CTG (PRISM-CTG), a clinically grounded self-supervised foundation model (FM) for CTG that leverages large-scale unlabelled recordings to learn transferable domain-level representations. PRISM-CTG is pretrained using a multi-view self-supervised framework that jointly optimises 3 complementary pretext objectives: random-projected guided masked signal reconstruction, clinical variable prediction, and feature classification. Each objective is associated with a dedicated task-specific token, enabling specialised representation learning, while controlled cross-attention facilitates information exchange across clinical context. By reframing patient metadata and domain knowledge, which are often underutilised in conventional training as prediction targets, Prism-CTG transforms readily available clinical information into additional supervisory targets that guide clinically meaningful representation learning. Extensive experiments across 7 downstream CTG tasks in both antepartum and intrapartum domains demonstrated that PRISM-CTG consistently outperforms in-domain and SSL baselines. Notably, PRISM-CTG demonstrated strong generalisation under external validation on 2 datasets, while achieving comparable performance to studies trained on substantially larger, privately labelled datasets. To our knowledge, this is the first study to introduce large-scale FM for CTG that learns domain-level representations. 

---
# When Safety Geometry Collapses: Fine-Tuning Vulnerabilities in Agentic Guard Models 

**Authors**: Ismail Hossain, Sai Puppala, Jannatul Ferdaus, Md Jahangir Alam, Yoonpyo Lee, Syed Bahauddin Alam, Sajedul Talukder  

**Link**: [PDF](https://arxiv.org/pdf/2605.02914)  

**Abstract**: A guard model fine-tuned on entirely benign data can lose all safety alignment -- not through adversarial manipulation, but through standard domain specialization. We demonstrate this failure across three purpose-built safety classifiers -- LlamaGuard, WildGuard, and Granite Guardian -- deployed as protection layers in agentic AI pipelines, and show that it originates in the destruction of latent safety geometry: the structured harmful -- benign representational boundary that guides classification. We extract per-layer safety subspaces via SVD on class-conditional activation differences and track how this boundary evolves under benign fine-tuning. Granite Guardian undergoes complete collapse --  refusal rate drops from 85\% to 0\%, CKA falls to zero, and 100\% of outputs become ambiguous -- a severity exceeding prior findings on general-purpose LLMs, explained by the specialization hypothesis: concentrated safety representations are efficient but catastrophically brittle. To mitigate this, we propose Fisher-Weighted Safety Subspace Regularization (FW-SSR), a training-time penalty combining (i) curvature-aware direction weights derived from diagonal Fisher information and (ii) an adaptive $\lambda_t$ that scales with task-safety gradient conflict. FW-SSR recovers 75\% refusal on Granite Guardian (CKA = 0.983) and reduces WildGuard's Attack Success Rate to 3.6\% -- below the unmodified baseline -- by actively sharpening the safety subspace rather than merely anchoring it. Across all three models, structural representational geometry (CKA, Fisher score) predicts safety behavior more reliably than absolute displacement metrics, establishing geometry-based monitoring as a necessary component of guard model evaluation in agentic deployments. 

---
# Reasoning-Guided Grounding: Elevating Video Anomaly Detection through Multimodal Large Language Models 

**Authors**: Sakshi Agarwal, Aishik Konwer, Ankit Parag Shah  

**Link**: [PDF](https://arxiv.org/pdf/2605.02912)  

**Abstract**: Video Anomaly Detection (VAD) has traditionally been framed as binary classification or outlier detection, providing neither interpretable reasoning nor precise spatial localization of anomalous events. While Vision-Language Models (VLMs) offer rich scene understanding, they struggle with reliable spatial grounding - often producing hallucinated or geometrically invalid bounding boxes when asked to localize objects. We propose VANGUARD (Video Anomaly Understanding through Reasoning and Grounding), a framework that unifies anomaly classification, spatial grounding, and chain-of-thought reasoning within a single VLM. VANGUARD introduces a three-stage curriculum that progressively layers training objectives: (1) classifier warmup on frozen backbone features, (2) LoRA-adapted spatial grounding, and (3) chain-of-thought generation. To overcome the sparse annotation typical of VAD benchmarks, we employ a teacher-student annotation pipeline in which a VLM (Qwen3-VL-4B) generates structured per-subclip reasoning trajectories based on manual annotations available from the UCA Dataset. Further, GroundingDINO provides bounding box supervision. On UCF-Crime, VANGUARD achieves 94% ROC-AUC with 84% F1 while simultaneously producing interpretable chain-of-thought explanations and spatial grounding of anomalous objects - capabilities absent from prior VAD methods. Ablations confirm that staged training outperforms monolithic optimization, and that structured reasoning acts as an implicit regularizer yielding more balanced predictions than classification-only fine-tuning. Zero-shot transfer to XD-Violence and ShanghaiTech demonstrates cross-domain generalization without target-domain adaptation. 

---
# Delay, Plateau, or Collapse: Evaluating the Impact of Systematic Verification Error on RLVR 

**Authors**: Kazuki Egashira, Mark Vero, Jasper Dekoninck, Florian E. Dorner, Robin Staab, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2605.02909)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a powerful approach for improving the reasoning capabilities of large language models (LLMs). While RLVR is designed for tasks with verifiable ground-truth answers, real-world verifiers (e.g., static code checkers) can introduce errors into the reward signal. Prior analyses have largely treated such errors as random and independent across samples, concluding that errors merely slow training with limited effect on final performance. However, practical verifiers tend to exhibit systematic errors. This introduces a risk of models learning unwanted consistent behavior from a structurally incorrect reward signal. In this work, we study the impact of such systematic verification errors on RLVR. Through controlled experiments on arithmetic tasks, we show that systematic false negatives lead to similar effects as random noise. On the other hand, systematic false positives can cause a wide range of behaviors from sub-optimal plateaus to performance collapse. Crucially, these outcomes are not determined by the overall error rate but by the specific pattern of introduced errors, making pre-hoc mitigation difficult. Our results show that, in contrast to prior conclusions, realistic verification errors can critically shape RLVR outcomes and that verifier quality has to be understood beyond its sample-level error rate. 

---
# Memorization In Stable Diffusion Is Unexpectedly Driven by CLIP Embeddings 

**Authors**: Bumjun Kim, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2605.02908)  

**Abstract**: Understanding how textual embeddings contribute to memorization in text-to-image diffusion models is crucial for both interpretability and safety. This paper investigates an unexpected behavior of CLIP embeddings in Stable Diffusion, revealing that the model disproportionately relies on specific embeddings. We categorize input tokens as <startoftext>, <prompt>, <endoftext> and <pad> with corresponding embeddings $\mathbf{v}^{\mathbf{sot}}, \mathbf{v}^{\mathbf{pr}}, \mathbf{v}^{\mathbf{eot}}, \mathbf{v}^{\mathbf{pad}}$. We discover that $\mathbf{v}^{\mathbf{pr}}$ contribute minimally to generation in memorized cases. In contrast, $\mathbf{v}^{\mathbf{pad}}$ strongly affect memorization due to their structural duplication of $\mathbf{v}^{\mathbf{eot}}$, the only embedding explicitly optimized during CLIP training. This duplication unintentionally amplifies the influence of $\mathbf{v}^{\mathbf{eot}}$, causing the model to over-rely on it, thereby driving memorization. Based on these observations, we propose two simple yet effective inference-time mitigation strategies: (1) Replacing the tokenizer's default <pad> from <eot> to the ! token before embedding, and masking the $\mathbf{v}^{\mathbf{eot}}$; (2) Partial masking of $\mathbf{v}^{\mathbf{pad}}$. Both suppress memorization without degrading quality, and are readily deployable without prior detection. 

---
# On the Invariants of Softmax Attention 

**Authors**: Wonsuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.02907)  

**Abstract**: Softmax attention maps every query--key interaction into a probability distribution, but the underlying structure remains largely unexplored. We define the \emph{energy field}, the row-centered attention logit, and show that it exhibits invariant properties across models, architectures, and inputs.
Two classes of invariants emerge. \emph{Mechanism-level} invariants follow from the algebraic structure of softmax attention. They include a per-row zero-sum constraint, a rank bound determined by the head dimension, and spectral signatures that follow from them. \emph{Model-level} regularities are not required by the mechanism, yet hold in every autoregressive language model we test, spanning several architecture families. The energy field distributes its variance over key positions without concentrating at a few. This delocalization traces to a property of the key matrix we call \emph{key incoherence}.
These invariants have practical consequences. The rank bound confines the energy field to a low-dimensional subspace. Key incoherence yields a per-head training monitor. All results are verified at multiple context lengths and input texts. 

---
# A User-Centric Analysis of Explainability in AI-Based Medical Image Diagnosis 

**Authors**: Julia Wagner, Tim Schlippe  

**Link**: [PDF](https://arxiv.org/pdf/2605.02903)  

**Abstract**: In recent years, AI systems in the medical domain have advanced significantly. However, despite outperforming humans, they are rarely used in practice since it is often not clear how they make their decisions. Optimal explanation and visualization of the decision process are often lacking. Therefore, we conducted a comparative user-centric analysis of the latest state-of-the-art textual, visual and multimodal explainable artificial intelligence (XAI) methods for medical image diagnosis. Our survey of 33 physicians showed that 88% agree that it is important that AI explains the diagnosis -- 64% even strongly agree. A combination of bounding box and report is rated better than the other tested XAI methods in the evaluated aspects understandability, completeness, speed, and applicability. We even tested the potential negative impact of false AI-based medical image diagnoses and found that 50% of the participants trusted false AI diagnoses over all tested XAI methods. 

---
# From Passive Feeds to Guided Discovery: AI-Initiated Interaction for Vague Intent in Content Exploration 

**Authors**: Yu Xie, Ying Qi  

**Link**: [PDF](https://arxiv.org/pdf/2605.02902)  

**Abstract**: Recommendation feeds work well when people are simply browsing, and search works well when they can formulate a query. Between these two cases is a common but poorly supported state: users feel that their feed has become repetitive, yet cannot clearly specify what they want instead. We refer to this state as vague intent. We present Red-Rec, an AI-supported exploration interface for this middle ground. After a period of browsing, the system summarizes patterns in the current feed (e.g., dominant content categories and possible latent interests), offers clickable exploration options, asks at most one follow-up question, and then gradually blends new content into the feed. The design is motivated by a formative study which found that users often recognize feed staleness but struggle to articulate alternatives, suggesting the need for proactive and low-effort this http URL evaluated Red-Rec in a mixed-design lab study against three comparison conditions: a passive feed, search, and a user-initiated chat interface. Compared with user-initiated chat, Red-Rec led to broader exploration, higher serendipity ratings, and lower interaction effort. Participants in the AI-initiated condition typed very little , relying mainly on option selection, whereas participants in the user-initiated chat condition typed substantially more . We discuss how proactive, option-based AI support can help users move beyond repetitive feeds without undermining their sense of control, and we outline design implications for recommendation interfaces that support open-ended exploration. 

---
# Safety in Embodied AI: A Survey of Risks, Attacks, and Defenses 

**Authors**: Xiao Li, Xiang Zheng, Yifeng Gao, Xinyu Xia, Yixu Wang, Xin Wang, Ye Sun, Yunhan Zhao, Ming Wen, Jiayu Li, Xun Gong, Yi Liu, Yige Li, Yutao Wu, Cong Wang, Jun Sun, Yixin Cao, Zhineng Chen, Jingjing Chen, Tao Gui, Qi Zhang, Zuxuan Wu, Xipeng Qiu, Xuanjing Huang, Tiehua Zhang, Zhipeng Wei, Hanxun Huang, Sarah Erfani, James Bailey, Jianping Wang, Wei-Ying Ma, Bo Li, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02900)  

**Abstract**: Embodied Artificial Intelligence (Embodied AI) integrates perception, cognition, planning, and interaction into agents that operate in open-world, safety-critical environments. As these systems gain autonomy and enter domains such as transportation, healthcare, and industrial or assistive robotics, ensuring their safety becomes both technically challenging and socially indispensable. Unlike digital AI systems, embodied agents must act under uncertain sensing, incomplete knowledge, and dynamic human-robot interactions, where failures can directly lead to physical harm. This survey provides a comprehensive and structured review of safety research in embodied AI, examining attacks and defenses across the full embodied pipeline, from perception and cognition to planning, action and interaction, and agentic system. We introduce a multi-level taxonomy that unifies fragmented lines of work and connects embodied-specific safety findings with broader advances in vision, language, and multimodal foundation models. Our review synthesizes insights from over 400 papers spanning adversarial, backdoor, jailbreak, and hardware-level attacks; attack detection, safe training and robust inference; and risk-aware human-agent interaction. This analysis reveals several overlooked challenges, including the fragility of multimodal perception fusion, the instability of planning under jailbreak attacks, and the trustworthiness of human-agent interaction in open-ended scenarios. By organizing the field into a coherent framework and identifying critical research gaps, this survey provides a roadmap for building embodied agents that are not only capable and autonomous but also safe, robust, and reliable in real-world deployment. 

---
# Same Voice, Different Lab: On the Homogenization of Frontier LLM Personalities 

**Authors**: Avinash Krishna, Kalyana Chadalavada, Unso Eun Seo Jo  

**Link**: [PDF](https://arxiv.org/pdf/2605.02897)  

**Abstract**: LLM assistant personalities play a critical role in user experience and perceived response quality. We present a large-scale experiment of frontier LLM personalities using external ELO-based traits scoring across 144 traits. We find that all models tested converge on a form of trait expression that is systematic, methodical, and analytical and suppress traits such as remorseful and sycophantic. Moreover, models tend to diverge more in their expression of ``middle-of-distribution traits`` such as poetic or playful, but even these so-called ``creative`` models tend to have more neutral identities. These similarities suggest an implicit emergence of a standard of optimal assistant behavior. In a landscape of varied training methods, character training, therefore, stands out for its uniformity, offering insight into a tacit consensus between model developers. 

---
