# AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control 

**Authors**: Jiabin Qiu, Zixuan Chen, Hongye Cao, Jieqi Shi, Jing Huo, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2609.30264)  

**Abstract**: Latent world models are typically trained to predict factual transitions, whereas model predictive control (MPC) must compare alternative actions from the same state. A model can therefore achieve low factual prediction error yet poorly distinguish candidate actions. We introduce AD-WM, an action-discriminative joint-embedding world model for counterfactual MPC. AD-WM combines residual latent dynamics with predictor-level action-recovery regularization, using inverse dynamics and a normalized recovery objective motivated by conditional mutual information. Both objectives encourage planning transitions to preserve action information; their auxiliary heads are discarded at test time, leaving MPC unchanged. On OGBench-Cube, AD-WM improves hard-start success from 3.7% to 52.0% over a matched LeWM baseline and improves mean success over the reproduced baseline in four of five simulation environments. Planning diagnostics show that factual prediction error and whole-bank action ranking do not follow the closed-loop success ordering, whereas CEM-aligned elite regret tracks success more closely. With a frozen V-JEPA 2 encoder and matched DROID post-training, AD-WM also improves zero-shot transfer to our Franka setup, increasing basic pick-and-place success from 42.2% to 71.1% without lab-specific adaptation. These results suggest that world models for planning should preserve action-dependent differences needed for counterfactual selection, rather than optimize factual prediction accuracy alone. More videos and code are available at this https URL. 

---
# A Living Benchmark for Information Retrieval from Electronic Health Records 

**Authors**: Jordan L. Cahoon, Chloe O. Stanwyck, Sulaiman Somani, Philip Chung, Kevin R Keet, Kameron C. Black, Andrea T. Fisher, Sarita Khemani, Jerry Liu, Stephen Ma, Saloni K. Maharaj, Rita M. Pandya, Eduardo Perez-Guerrero, Priyanka Pillai, Lisa Shieh, David J.H. Wu, James Xie, James C. McAvoy, Teresa Nguyen, Jessica Tran, Lucy Yin, Bridget Lin, Alison Callahan, Jason A. Fries, Nigam H. Shah, Emily Alsentzer  

**Link**: [PDF](https://arxiv.org/pdf/2609.30205)  

**Abstract**: Large language model (LLM)-based clinical assistants are increasingly being integrated into electronic health record (EHR) systems, transforming how clinicians retrieve and synthesize information from patient records. Their safety and utility depend on rigorous evaluation, yet existing benchmarks are manually curated, costly to update, and rapidly become obsolete with evolving technological advancements. We present a scalable framework that automatically generates question--answer pairs from longitudinal EHR notes. Nineteen clinicians validate the benchmark generator, producing the Benchmark for Retrieving Information in EHRs (BRIE), a continuously maintainable evaluation dataset. Across nine LLMs and five inference strategies, state-of-the-art systems frequently omit clinically important information, particularly for questions requiring synthesis across multiple documents and encounters. Because the generator itself is validated, BRIE supports evaluations that static benchmarks cannot, including the generation of multiple answers that reflect variation in clinician reasoning for robust performance assessment and continuously refreshing benchmark content to guard against leakage. Our results demonstrate that scalable benchmark generation enables rigorous, up-to-date evaluation of clinical LLMs as they are deployed in rapidly evolving healthcare settings. 

---
# ExplorationBench: Measuring AI Systems' Exploration in Verifiable Alien Worlds 

**Authors**: Ming Zhang, Zhenghao Xiang, Peizhong Gao, Yujiong Shen, Yuhui Wang, Zhonghan Yue, Shihan Dou, Zhangyue Yin, Junjie Ye, Shichun Liu, Weihuang Zheng, Jiahao Chen, Jiayi Chen, Hongzhang Liu, Jiaqi Shao, Tao Gui, Qi Zhang, Xuanjing Huang, Suncong Zheng, Maxm Pan  

**Link**: [PDF](https://arxiv.org/pdf/2609.30199)  

**Abstract**: Scientific discovery begins where known problems end. There, AI systems must engage in exploration: framing hypotheses, designing experiments, and iterating on the results. However, evaluating this ability is difficult: (1) how to verify whether a genuinely new hypothesis holds, and (2) how to determine whether a system has discovered it through exploration or merely recalled related knowledge from pre-training data. To this end, we introduce ExplorationBench, which turns the wicked problem of evaluating scientific exploration into a concrete and tractable framework built on verifiable Alien Worlds: their rules are executable, so every answer can be checked exactly, and they conflict with familiar knowledge, so recall alone cannot solve the tasks. The benchmark contains two sandboxes, AlienCode (31 discovery targets, 70 tasks) and AlienLogic (24 discovery targets, 70 tasks). Each sandbox provides a flawed manual, task-specific environmental feedback, and a dedicated tool-call schema. Systems use these resources to explore the sandbox, then solve held-out tasks. We evaluate 10 AI systems and find that the strongest systems can acquire and apply unfamiliar rules, while performance varies substantially across trajectories and continued exploration can stall or reverse earlier gains. ExplorationBench represents a step towards AI systems that can acquire and apply genuinely new knowledge through exploration in unknown environments. 

---
# SAGE: Mitigating Long-Horizon Reasoning Biases via Topological Guidance 

**Authors**: Xinyue Zeng, Jiawei Zhang, Yujun Yan, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.30192)  

**Abstract**: Long-horizon reasoning remains a central challenge for large language models (LLMs) under sparse-reward regimes. We argue that this brittleness arises from two biases induced by complex reasoning spaces: an exploration bias, where models are drawn toward locally plausible but structurally unstable branches, and a compounding bias, where small local deviations accumulate across depth and suppress rare rewards. We introduce Symbolic Closure Analysis (SCA) as a theoretical lens characterizing how branching structures and sparse rewards induce these biases in long-horizon reasoning with local admissibility, and as a design principle for structural priors in less formal reasoning tasks. Motivated by this analysis, we propose SAGE (Structural Admissibility-Guided Exploration), a unified framework that injects structural guidance to alleviate exploration bias and compounding bias in long-horizon reasoning. SAGE combines two complementary structural guidance: algebraic sparsification, which projects locally admissible candidates onto operator-indexed algebraic subspaces to suppress spurious branching and mitigate exploration bias, and hyperbolic structural guidance, which embeds reasoning states into a negatively curved space to provide dense depth-wise signals and mitigate compounding bias. Across 12 benchmarks and 7 model families, SAGE outperforms competitive baselines. In particular, SAGE achieves up to an 8-fold improvement on the Andrews-Curtis problem, an open real-world long-horizon task. Code is available at: this https URL. 

---
# Jev-Mobile: Jev as an Executor for Mobile GUI Agents 

**Authors**: Linghua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.30186)  

**Abstract**: Vision-language models (VLMs) have become a common foundation for autonomous mobile GUI agents, but most existing systems rely on the VLM for both planning and action grounding at nearly every interaction step, leading to substantial latency and model-serving cost. We introduce Jev-Mobile, which shifts this paradigm to low-frequency VLM planning and high-frequency lightweight execution: the VLM specifies local goals, the accessibility tree defines a structured executable action space, and Jev, a fast typed decision model, repeatedly selects actions within this space. This design allows multiple GUI actions to be executed under a single VLM decision, reducing expensive VLM inference while preserving adaptive interaction. On the full AndroidWorld task suite, Jev-Mobile achieves 79% task success, compared with 78% for SeeAct-V and 84% for a Step-wise VLM baseline. Among successful trajectories, it reduces mean end-to-end execution time by 32.7% and mean model API cost by 73.4% relative to Step-wise VLM. These results show that decoupling high-level VLM reasoning from low-level action execution can substantially improve mobile GUI agent efficiency while maintaining competitive task performance. 

---
# Search-Aware Reinforcement Learning for Multi-Component Query Understanding in Roblox Game Search 

**Authors**: Nayoung Choi, Shengjian Chen, Xiaokai Wei, Wenzheng Zhang, Daiyao Yi, Rachit Pareek, Vincent Su, Michelle Gong, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2609.30177)  

**Abstract**: Query understanding (QU) plays a critical role in production search systems, translating raw user queries into search execution plans that drive downstream retrieval and ranking. While large language models (LLMs) have enabled QU to be framed as a structured multi-task generation problem (e.g., intent classification, query expansion), optimizing such models to produce search-engine-coupled outputs remains challenging: static, label-based supervision fails to capture how each component actually interacts with the underlying search pipeline to affect downstream performance. We present a search-aware reinforcement learning (RL) framework for QU based on a distill-then-RL paradigm. Teacher-student supervised fine-tuning (SFT) first yields a well-formed, schema-compliant policy initialization. The RL stage then optimizes each QU component with rewards derived from live interaction with the search engine, tailored to that component's operational role, rather than a single reward tied to the final search outcome. Experiments on Roblox search show that this component-specific optimization improves both per-component utility and downstream search quality, raising NDCG@20 by 8.9 points over the SFT policy and by 3.5 points over training with a single end-to-end reward. 

---
# GRASP: Generating, Revising, and Assessing for Strategic Planning with Agentic AI 

**Authors**: Arunabh Srivastava, Mohammad A., Khojastepour, Srimat Chakradhar, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2609.30147)  

**Abstract**: Large Language Models (LLMs) typically exhibit a performance profile where reliability degrades as task complexity increases. We address the challenge of generating high-quality natural language executable plans for complex tasks by introducing $\textbf{GRASP}$, a strategy-aware, multi-stage planning framework. GRASP decouples the planning pipeline across specialized, context-isolated modules: it pre-compiles global macro-guidelines (GenPlan), explores alternative localized strategies within isolated context windows (RevPlan), and independently evaluates trajectories using a multi-criteria discriminator (VerPlan). Empirical evaluations show that GRASP consistently establishes a new state-of-the-art frontier across diverse datasets, yielding substantial accuracy gains over direct LLM planners on Natural Plan Calendar Scheduling ($\sim$12.4$\%$$\uparrow$), ZebraLogic ($\sim$30.8$\%$$\uparrow$), and SciBench Math. Crucially, under multi-task scaling-where standard planners suffer immediate performance collapse-GRASP completely flattens the multi-task degradation penalty. In interleaved dual-task environments, GRASP achieves an absolute accuracy gain of up to 16.7$\%$ over direct LLM planners. Furthermore, by isolating context and enforcing strict macro-regularization, GRASP outperforms frontier reasoning models (such as GPT-5-mini) by a margin of 14.5$\%$. 

---
# EnigmaForge: The Question Is Hidden in the Story 

**Authors**: Daniel Eisner  

**Link**: [PDF](https://arxiv.org/pdf/2609.30144)  

**Abstract**: Most benchmarks hand the model a question. EnigmaForge hands it a stack of old documents and no question at all. Buried in the letters, receipts, and logbook margins is a small logic puzzle whose solution is unique - proved by a SAT solver at generation time, with an ablation certificate showing every clue is load-bearing. Because instances are generated rather than collected, the corpus renews forever. The headline measure is intuition: task success when handed only the story, with world reconstruction as the secondary axis. Twenty-five frontier models ran over 600 instances (17,400 scored records) under three matched conditions. Intuition reshuffles the leaderboard: a 22x spread where fact recovery spans 1.6x, the second-best fact-recoverer ranks fourteenth, one model is indifferent to being told the question, and another is significantly better without it. Several models were blocked by their own content filters before reaching the puzzle - any benchmark scoring refusals as failure is quietly measuring filter behavior. 

---
# Screen Before You Serve: Simulation for Production Customer Experience AI Agents at 140M Scale 

**Authors**: Edesio Alcoba, Kevin Rossell, Aman Gupta, Shao Tang, Jiwoo Hong, Pabel Carrillo-Mendoza, Wanderson Conceição Ferreira, Alvaro Tedeschi, Zayd Simjee, Shreya Rajpal, Bruno Finardi Hime, Christian Sousa, Luis Moneda, Herbert Fei, Daniel Silva, Rohan Ramanath  

**Link**: [PDF](https://arxiv.org/pdf/2609.30137)  

**Abstract**: Customer experience (CX) agents use tools and large language models to address customer requests and guide conversational interactions with an organization's products. Improving these agents, especially in regulated industries, is difficult: they must detect intent, follow complex operational policies and use tools reliably. Manual end-to-end testing offers limited coverage, while live experiments expose customers to failures that can erode trust.
We present a hypothesis-driven simulation workflow for screening candidate CX agents before deployment. Synthetic customers react to agent responses and simulated tool outputs enable multi-step agentic workflows without invoking production backends. We use the Snowglobe simulator on Nubank's Card Delivery agent and its expanded successor, Card Management - Nubank's highest-volume chat-support agent in Brazil. Across 4 deployed versions, simulated and production version-level binary evaluator scores show high correlation. Simulation-guided iteration increased transactional net promoter score (tNPS) by 36.69 points in a live A/B test. We also screened open-weight configurations in over 16,000 simulated conversations. In a subsequent live A/B test, the selected model increased self-service rate (SSR) by 8.82 percentage points to the highest level observed at Nubank, with no statistically significant change in tNPS. Simulation made broad exploration of models, reasoning settings, and prompts feasible without customer exposure, enabling production improvements that would have been impractical to pursue through live experimentation alone. 

---
# HEXIS: Compiling Skills into Extended Finite State Machines 

**Authors**: Minghao LI  

**Link**: [PDF](https://arxiv.org/pdf/2609.30123)  

**Abstract**: Agent skills provide reusable knowledge and instructions, yet agents must repeatedly infer how to apply them and which operation should follow. This couples task reasoning with control decisions, allowing prescribed steps to be omitted or applied incorrectly. We introduce HEXIS, which compiles agent skills into extended finite state machines that separate knowledge from control flow. Skill knowledge is incorporated into local instructions that guide reasoning and generation within states. The machine records execution progress and intermediate results, while explicit transition conditions determine subsequent operations. Our incremental compiler first maps skill clauses and tool interfaces to state operations, local instructions, data bindings, and transitions. It then aligns development traces with existing states to identify missing operations and dependencies. These are incorporated by adding or reusing states and refining their connections. Updates are accepted only after static checks and replay of the current and all previously accepted traces. Across four benchmarks and four executors, HEXIS improves success over Skill + ReAct by 16.1 percentage points on average. Qwen3.8-27B reduces execution tokens by 38.4-88.9% across benchmarks. 

---
# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations 

**Authors**: Luciano Maldonado  

**Link**: [PDF](https://arxiv.org/pdf/2609.30094)  

**Abstract**: Large language models increasingly operate as persistent assistants in user-facing, shared-session, and tool-augmented settings. When users disclose sensitive information during an active conversation, that information may remain behaviorally recoverable through later prompts even after the dialogue shifts to unrelated topics. We introduce \textbf{PrivDrift}, a benchmark for auditing whether user-disclosed secrets remain recoverable after conversational topic drift and persuasion-based probing. PrivDrift contains 1{,}000 controlled multi-turn dialogues with seeded secrets, content-dense drift turns, and standardized extraction probes. Across three LLMs with extended context windows, dialogue-level hybrid leakage remains substantial, ranging from 38.7\% to 54.6\%, and varies strongly by model, secret type, and persuasion intensity. Within the tested drift window, additional topic drift does not reliably reduce leakage, suggesting that privacy risk in active LLM contexts should be evaluated as a persistent behavioral failure mode rather than only as training-data memorization or immediate jailbreak behavior. 

---
# Self-Play Pretraining with Zero Data 

**Authors**: Aditya Cowsik, Kfir Dolev, Michael Y. Li, G. Bruno De Luca, Nourya Cohen, Noah D. Goodman, Yoav Levine  

**Link**: [PDF](https://arxiv.org/pdf/2609.30063)  

**Abstract**: Advances in language modeling have been driven by scaling pretraining on ever more data. Yet, the training data is still largely curated on the model's behalf. A more general approach to pretraining would let the model learn to generate the data most useful for its own improvement. This would provide an effectively unbounded source of training data, limited by compute rather than human knowledge. We introduce Self-Play Pretraining with Zero Data, an initial proof-of-concept towards realizing this vision. Our procedure casts synthetic data generation as a search over the space of all computable structure, taking inspiration from Solomonoff induction. Starting from random initialization, two models learn in tandem: a generator proposes programs interpreted by a universal Turing machine, generating byte sequences, while a learner autoregressively predicts these byte sequences. The learner is trained with standard cross-entropy, while the generator is trained with reinforcement learning to produce sequences at the frontier of the learner's capabilities, yielding an adaptive curriculum. A universal Turing machine gives us a search space over all computable data-generating processes, imposing little domain-specific structure, and self-play searches over this space for useful training data. We test whether zero-shot performance on natural data improves predictably with self-play compute; this is a clean test of transfer since neither generator nor learner is trained on natural data. Across several natural datasets, zero-shot loss exhibits predictable scaling in compute. The models also exhibit in-context learning, and discover recognizable mathematical sequences during training. 

---
# SciWalker: Synthesizing Scientific Coding Problems with Operator Graphs and Execution Feedback 

**Authors**: Chenxi Li, Wenxuan Zeng, Yun Luo, Fangchen Yu, Peng Ye, Yu Cheng, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.30054)  

**Abstract**: Improving the scientific coding capabilities of large language models (LLMs) requires high-quality training data. However, such data remain scarce because manually authoring realistic problems is costly and time-consuming, while systematically covering diverse scientific domains and algorithmic combinations remains challenging. To address this, we introduce SciWalker, a framework for synthesizing scientific coding problems through operator-chain sampling and execution feedback. The framework combines scientific library interfaces with operation modes to instantiate operators, organizes them into operator graphs, and samples operator chains as computational workflow cues. Guided by these cues, we adopt LLMs to generate scientifically grounded problem statements, reference solutions, and tests, with failed generations iteratively repaired using execution feedback. By combining structured workflow composition with verification and quality review, SciWalker enables scalable task generation while promoting scientific grounding, computational diversity, and executability. Using this framework, we construct 8,178 high-quality problems spanning 5 scientific domains and 32 subdomains. To evaluate their training utility, we conduct reinforcement learning on Qwen3.5-9B using the GSPO algorithm. This training improves SciCode subproblem accuracy by 9.9 percentage points, from 29.3% to 39.2%, with gains across scientific code generation, code repair, and reasoning benchmarks. The code for SciWalker is available at this https URL. 

---
# NNV3: Expanding Neural Network Verification to New Architectures and Domains 

**Authors**: Anne M. Tumlin, Samuel Sasaki, Ben Wooding, Diego Manzanas Lopez, Muhammad Usama Zubair, Navid Hashemi, Hongchao Zhang, Waseem Abbas, Ipek Oguz, Meiyi Ma, Taylor T. Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2609.30050)  

**Abstract**: We present NNV3, the latest version of the Neural Network Verification (NNV) tool, a MATLAB framework for formal verification of deep learning models and learning-enabled cyber-physical systems. Building on the set-based reachability foundation of NNV 1.0 (FFNNs, CNNs, NNCS) and NNV 2.0 (RNNs, SSNNs, neural ODEs), NNV3 introduces new members of the Star-set family: ModelStar for verifying networks under weight perturbation, VolumeStar for video and 3D volumetric inputs, and GraphStar for graph neural networks. A conformal-inference-based probabilistic reachability mode complements sound analysis for problems where deterministic verification is intractable, while FairNNV certifies counterfactual and individual fairness properties over continuous input regions. NNV3 introduces new benchmarks for malware detection, graph-based power-system models, medical imaging, variable-length time series data, and action recognition. NNV3 also incorporates tutorials and developer guides through a unified documentation site. This paper details these major updates, demonstrating NNV's maturation into a comprehensive, robust, and accessible verification tool for a diverse range of AI systems. 

---
# Style, Not Self: Surface Cues Explain Zero-Shot Code Attribution by Large Language Models 

**Authors**: Ehsan Barkhordar, Surendrabikram Thapa  

**Link**: [PDF](https://arxiv.org/pdf/2609.30048)  

**Abstract**: If a language model can recognize code it wrote, it may favor that code as a judge, and instances of one model monitoring each other could collude. We test this zero-shot on current commercial models. Five LLMs generate solutions to MBPP, HumanEval, and DS-1000, seven more to MBPP, and models act as evaluators in four tasks: picking their own solution from a pair, judging whether a single solution is their own, identifying which of two solutions a named model wrote, and judging quality blind. In the single-solution task, balanced accuracy is 49-58% for all 15 model-benchmark combinations, while raw accuracy (38-67%) mostly reflects how readily a model claims authorship. In the pairwise task, accuracy across 14 evaluator-opponent combinations correlates at r=0.93 with how often the evaluator's solution is longer. Attribution to a named model succeeds on some pairs and is consistently inverted on others. A rule-based normalization that strips docstrings, comments, type hints, and local names preserves Pass@1 and leaves ten of twelve re-tested results at chance; the other two follow a length difference it leaves, although a trained classifier still separates most normalized pairs. Claude Haiku's self-preference also disappears. We recommend reporting balanced accuracy, heuristic baselines, and label consistency. 

---
# How does Adversarial Influence Scale in Multi-Agent Systems? 

**Authors**: Addison J. Wu, Jasin Cekinmez, Michel Liao, Karthik Narasimhan, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2609.30028)  

**Abstract**: Multi-agent deliberation can improve performance, but what happens when some agents do not act in good faith? In practice, an agent may be deceptive and work to subvert the group, whether through its own objectives or external instruction. We study how susceptibility to deception scales as groups increase in size and deceivers become more prevalent. It is not the number of agents in the group that matters, but the proportion of deceivers. We observe that the defection rate, how often initially correct agents switch to an incorrect final answer, rises linearly with this proportion. Whereas humans in comparable conformity studies are reliably swayed only when misleading confederates form a majority, LLM agents defect regularly even when deceivers remain a minority. Susceptibility also depends on which models are interacting, especially on the honest agent side. Unexpectedly, allowing deceivers to coordinate privately can make them less effective. Altogether, our results show that adding more agents is therefore not a sufficient defense, because the adversary can simply scale with the group. 

---
# Synthetic Hospital: An Open, Verifiable, Physician-Validated Longitudinal EHR Benchmark 

**Authors**: Christine Park, Valerie Chen, Tim Dettmers  

**Link**: [PDF](https://arxiv.org/pdf/2609.30027)  

**Abstract**: Frontier language models are rarely used in clinical workflows because the realistic, longitudinal benchmarks needed to develop them are scarce. Real electronic health record (EHR) data cannot be openly shared due to privacy, ethics or data use issues and it does not contain verifiable ground truth since the chart records only reflect what clinicians documented. We introduce Synthetic Hospital, an open, fully synthetic, fact-grounded longitudinal EHR benchmark that resolves the open sharing and verifiable ground truth barriers. Built entirely from public medical-education material with no protected health information, it comprises 1,268 longitudinal patients and 5,602 encounters, where every diagnosis, finding, and temporal relation is grounded in standard ontologies (ICD-10-CM, SNOMED CT, LOINC) and with a complete provenance chain back to its source medical education material. Synthetic Hospital is served through a simulated hospital record system that mirrors real EHR infrastructure (standard interoperability APIs, role-based access and function-calling interface). In a blinded review, physicians distinguished its records from real patient charts at near-chance rates (53\%). Across 10 frontier and open models, none approaches ceiling: the best model reconstructs a patient's longitudinal problem list with a severity-weighted F1 of 0.73, level with the mean of seven physicians on a matched subset but well below the best of them (0.89), and misses roughly half of clinically relevant findings when summarizing a chart. Overall, these results highlight that Synthetic Hospital is a difficult and realistic test of clinical AI performance. 

---
# Advancing Model Research in AgentX: Long-Horizon Autonomy for Industrial Recommender Systems 

**Authors**: Shuang Yang, Zijie Zhuang, Changxin Lao, Pengbo Xu, Hanwen Xu, Yusheng Huang, Han Gao, Guanchen Wang, Tianbao Ma, Linxun Chen, Peilin Song, Xuming Wang, Chen Li, Fan Wu, Tao Wang, Zibo Zhao, Xiangyu Wu, An Liu, Fei Pan, Peng Jiang, Chen Yang, Zhaojie Liu, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2609.30001)  

**Abstract**: Sustaining industrial recommendation research requires using the results of one experiment to decide what to investigate next. We present AgentX-Model, the next generation of AgentX's model research framework, which connects proposal development and model experimentation within sandboxes defined by business inputs and prediction tasks. AgentX-Model adopts a dual-agent architecture comprising a Research Agent and a Model Agent. The Research Agent develops independently reviewed proposals from papers and experimental findings, while the Model Agent conducts multi-round investigations and returns code, measurements, and unresolved questions. Using the returned results, the Research Agent selects a starting implementation and formulates the next research question, allowing subsequent experiments to build on earlier findings. We organize this continuing research around four actions: Reproduce, Follow-up, Composition, and Diagnose. The first three actions drive routine research, while Diagnose acquires the evidence needed to choose a repair, including for issues raised by business feedback and online evaluation, such as prediction bias measured by PCOC. Across the production evaluation, 560 of 636 completed model-changing experiments recorded AUC above their business baselines. As research continued, some experiments recorded AUC above every comparable ancestor in their lineages. The five latest online A/B evaluations across different business settings reported gains including 10-15% in acquisition efficiency, 15-20% in target-segment advertising spend, and 0.3-0.8% in watch time; the watch-time model used approximately 10% fewer FLOPs and parameters. A dependency-aware historical-replay benchmark further evaluates research allocation, with initial results showing no consistent efficiency gain from more complex scheduling when agents already analyze and select concrete candidates. 

---
# Augur: A Synthetic Decision Lab for Rehearsing Reactions to Product and Policy Changes 

**Authors**: Rahul Khedar, Mayank Malhotra, Avinash Karn  

**Link**: [PDF](https://arxiv.org/pdf/2609.29952)  

**Abstract**: Before a product or policy change ships, the question that matters is how people will react to it. Augur rehearses that reaction offline: it builds a typed knowledge graph from the change documents, populates a grounded persona market, simulates the interaction, and returns an auditable decision memo recommending one of five actions. We assemble Gold-50, fifty real product and policy episodes whose real-world outcome is known, adjudicated against the public record, and score the five-way release verdict against it.
Our central finding is methodological and negative: most of the measured gap between frontier cloud models and open-weight models we fine-tune and serve offline is attributable to an under-specified evaluation, not a difference in capability. We show this three ways. First, the prompt envelope alone can dominate the score: holding weights, cases and scorer fixed, one system -- a LoRA-SFT adapter on Qwen3-32B -- swings from 0% to 73%. Second, in a matched 2x2 ablation, defining the decision taxonomy in the prompt -- with no model change -- lifts every frontier model by +24 to +34pp; under the under-specified prompt, Qwen3-32B LoRA-SFT served offline beats all three frontier models (paired McNemar, Holm-corrected), and once the prompt is fair no significant difference from any of them is detected. Third, agreement with the distillation teacher rises without accuracy following, and the full pipeline amplifies a systematic "over-doom" bias rather than improving the verdict. Separately, we validate the reaction layer on its own terms: blind judges across four model families find the synthetic reaction recovers 67-90% of the concerns the public actually raised, and a pre-registered ablation locates its value -- largest where the decision is hardest, redundant near ceiling. The pipeline that regenerates every number and figure here is available from the authors. 

---
# ENDOPROMPT: Victim-Side Pseudo-References for Utility Degradation 

**Authors**: Qingyu Wu, Zeyu Feng, Yongda Yu, Yuzhe Luo, Hua Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.29948)  

**Abstract**: Prompt injection can degrade benign task performance without eliciting harmful content. Yet many attack objectives depend on task labels or predefined target responses. We present ENDOPROMPT, a white-box method that learns utility-degrading prefixes from unlabeled instructions. Its generator takes the request text as input. Clean victim continuations serve as pseudo-references: local search identifies prefixes that reduce continuation likelihood, and preference fitting on comparisons within the same instruction, followed by reward refinement, distills this signal into a generator. At deployment, the generator produces one prefix per request without further victim-side search. Across four instruction-tuned models and the complete splits of seven benign benchmarks, ENDOPROMPT yields a mean utility change of -26.8 percentage points; 27 of 28 cells are negative. Failure analysis reveals output expansion and prefix reuse; the controls do not establish a degradation advantage from request matching. Victim-derived supervision can reveal utility weaknesses without benchmark feedback or prescribed failure responses. The code will be released upon acceptance. 

---
# Neuro-symbolic AI for Industrial Configuration 

**Authors**: Danilo Valerio, Philipp Kogler, Stefan Bischof, Thomas Hubauer, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2609.29947)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance on a wide range of generative tasks. Yet their probabilistic nature makes them, in isolation, fundamentally unsuited for industrial product configuration, where outputs must be syntactically valid, semantically consistent with a knowledge base of hundreds of features and rules, and producible by an existing manufacturing chain. We argue that Neuro-symbolic (NeSy) AI methods lay out a promising path towards industrial-grade configurators that are reliable by design, explainable, and trustworthy. This paper describes a taxonomy of three NeSy integration strategies, namely hybrid inference, hybrid fine-tuning, and hybrid training, exploring their usage in the configuration domain. We report our effort to operationalize NeSy concepts in an industrial configuration copilot and derive a set of practical design choices for deploying trustworthy AI in engineering environments. We close with a discussion of open research challenges we consider most pressing, in particular how to scale NeSy methods from small academic demonstrators to the size of industrial configurators. 

---
# Who Holds the Pen? Let Specifications, Not Agents, Sign Off 

**Authors**: Haiqing Li, Xin Ma, Yinhao Wu, Wenliang Zhong, Feng Jiang, Thao M. Dang, Xiao Hu, Hehuan Ma, Yuzhi Guo, Junzhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.29921)  

**Abstract**: Large language model agents increasingly combine generation, decision-making, execution, and self-evaluation within a single agentic loop. Although they operate under external specifications such as task instructions, guidelines, output schemas, and reusable skills, these specifications typically remain context for the same model that acts and declares completion, leaving no independent specification authority boundary. We identify two resulting gaps. The understanding--execution gap arises when a requirement is understood but not satisfied in execution; the state--authority gap arises when an agent's interpretation or completion claim does not establish the required state. On SkillsBench, using only agent-visible prompts, workspace information, and injected skill specifications, we extract 509 source-grounded task directions. Across seven models, only 79.6%--86.4% are satisfied, while completion-claim rates exceed official evaluator pass rates by 28.7--37.9 percentage points. We therefore separate agent proposals from authoritative state. Agents may plan, act, and request completion, but only admissible evidence from qualified providers may establish specification-governed state. SpecHarness operationalizes this principle by compiling visible specifications into source-linked obligations and governing execution and finalization through versioned obligation state. Verifiable requirements are mediated or validated at runtime, while ambiguous or subjective requirements remain advisory. Experiments on guideline-following and artifact-generation tasks show that specifications can serve not merely as behavioral guidance, but as authority over compliant execution and completion. 

---
# Qwen-Planner-Agent: A Closed-Loop AI-for-AI Framework for Real-World Mobile Planner Agents 

**Authors**: Tingyu Qu, Weigao Sun, Yuecheng Liu, Yucheng Zhao, Yi Zhu, Yifeng Ding, Qiyi Wang, Sihan Cao, Pengkun Jiao, Hanlei Xie, Xiongwei Wu, Qichao Wang, Haodong Zhang, Jiajun Liu, Yuhao Wang, Yuqing Xie, Junpeng Zhao, Long Chen, Ming Ma, Sihan Yang, Ziwang Zhao, Yanhao Jia, Liangquan Gong, Feida Zhu, Yiran Zhong, Steven Hoi  

**Link**: [PDF](https://arxiv.org/pdf/2609.29892)  

**Abstract**: The rapid progression of large language models is extending AI from passive content generation into the active workflows of engineering and scientific discovery. This shift raises a compelling question: can AI be both the object of development and an active participant in building next-generation AI systems? We explore this question by building Qwen-Planner-Agent within a closed-loop AI-for-AI framework for scalable development and iterative improvement. Mobile planning offers a demanding test of this approach: complex, long-horizon tasks challenge agent reliability, while costly real-device interaction limits development scalability. The framework connects data production, model training, and deployment through a shared action-feedback-verification contract. (i) AI for Data builds a human-gated agentic data flywheel in which specialized agents construct tasks, collect interaction trajectories, curate and balance training data, and use training feedback to guide subsequent data generation. (ii) AI for Training combines a supervised planning cold start with hybrid-environment online agentic reinforcement learning, where we introduce Competence-Aware Reward-and-Advantage Engineering (CARE) to reduce reasoning and tool-use costs while preserving task performance. (iii) AI drives model--harness co-evolution through an execution-evidence-driven loop that orchestrates memory, skills, and tools at runtime and feeds structured action feedback and preserved failure traces back into coordinated model and harness adaptation. Qwen-Planner-Agent achieves the best overall performance among all evaluated models and systems on MobilePA-Bench, improving over its base model across tool use, memory, skills, and sub-agent coordination. Further evaluations of our model show improvements across non-mobile agentic benchmarks while largely preserving general capabilities. 

---
# Ontology-Mediated Neurosymbolic Constraint Acquisition from Multiple Stakeholders 

**Authors**: Stefan Bischof, Juliana Kainz, Danilo Valerio  

**Link**: [PDF](https://arxiv.org/pdf/2609.29876)  

**Abstract**: Neurosymbolic research typically assumes a pre-existing symbolic specification, leaving the upstream challenge of acquiring and formalizing requirements and constraints largely unaddressed. We present an architecture that fills this gap by using an OWL configuration ontology to mediate between neural constraint sources and downstream consumers. In this framework, LLM assistants elicit soft stakeholder preferences, while hardware specifications define hard physical and engineering limits. The ontology unifies these heterogeneous inputs, leverages description logic to identify unsatisfiability, and generates symbolic explanations that enable LLMs to interactively renegotiate terms with users. Any remaining conflicts are resolved downstream via priority-based relaxation. We illustrate our approach on a microgrid use case from the FLEXI project and argue its generalizability to multi-stakeholder domains where constraint acquisition is distributed across human and automated sources of unequal authority. 

---
# When Can Agents Forget Their Reasoning? ICLR for Long-Horizon Agent Context Compression 

**Authors**: Mingxuan Wang, Fei Luo, Bo Wang, Guorun Yao, Yinglong Guo, Chao Ning, Hongyue Chen, Yanbiao Ma, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.29875)  

**Abstract**: Long horizon language model agents continually accumulate reasoning history, increasing context length and inference cost even after earlier decisions have been executed and observed. Unlike static Chain of Thought compression, removing historical reasoning can change future actions and the resulting interaction trajectory. We study when such reasoning can be safely forgotten. We propose Interaction Aware Compression for Long Horizon Reasoning (ICLR), a training free online method that ranks reasoning blocks using frozen proxy entropy while preserving actions, tool calls, and observations. On 260 WorkBuddyBench tasks, ICLR improves average reward from 0.699 to 0.718, while reducing input, output, and cache read tokens by 25.5%, 14.4%, and 33.3%, respectively. Ablations reveal trajectory amplification, where local reasoning deletion produces nonlinear changes in total computation by altering subsequent interaction. Representation probing, activation patching, and controlled trajectory analyses further suggest that historical reasoning becomes more replaceable once task relevant derived state has been reliably externalized into code, files, tool outputs, or environmental feedback. These results characterize agent reasoning as dynamic working state rather than permanent interaction history. 

---
# A Risk-Adaptive and Evidence-Constrained Framework for Generative AI Feedback in Programming Education 

**Authors**: Shihao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.29874)  

**Abstract**: Generative artificial intelligence can turn learning analytics into personalized support, but feedback systems must decide when to intervene, which evidence to use, and how much assistance to provide. We developed a risk-adaptive, evidence-constrained framework for introductory programming using 2993 failed-submission states from 215 students. Student-disjoint models predicted persistent failure and related outcomes; four matched feedback conditions were generated for 136 cases; and calibrated risk informed capacity-limited intervention policies. The validation-selected logistic regression model achieved a test precision-recall area under the curve of 0.550 and a receiver operating characteristic area under the curve of 0.681. Broader student histories improved prediction of unmodified resubmission. After standardized repair and evidence gating, 519 of 544 newly generated messages contained all required components. A fixed-threshold sequential policy selected 17.8% of eligible test states and captured 25.2% of observed persistent failures. These findings support an evidence-gated progressive assistance strategy: calibrated risk guides intervention timing, recorded evidence constrains feedback content, and assistance progresses from self-checks to localized hints when warranted. The framework connects prediction, decision-making, and grounded generation while keeping their evaluation outcomes distinct. 

---
# PUBG Ally: A Conversational Embodied Agent as an AI Teammate 

**Authors**: Beomsoo Kim, Byeongju Kim, Dohyun Kim, Dongwon Kim, Eunchong Kim, Hongmin Kim, Hyeojung Im, Hyeonbin Hwang, Hyeonghwan Kim, Hyoseok Seol, Insub Im, Irene Chen, Jaeseung Jeon, Jimin Hong, Kiyoon Yoo, Minkyoung Park, Seohyeon Jung, Seungjun Chung, Sue Hyun Park, Sungwoo Kim, Youngin Cho, Yujeong Son, Kangwook Lee, Hyunseung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.29837)  

**Abstract**: We introduce PUBG Ally, an embodied agent for PUBG: BATTLEGROUNDS that can reason, act autonomously, and play alongside players as a voice-enabled teammate. Building such a teammate requires combining two difficult capabilities: it must perceive and respond to a constantly changing game world under strict latency constraints while interacting naturally with players, keeping its speech synchronized with its actions. Ally therefore combines agentic tool use with real-time game control. A language-model agent uses a controlled interface to inspect game information, interpret player speech, maintain context, decide what to say, and issue high-level action choices that steer a faster control layer for movement, combat, and recovery. Because the player's and Ally's speech and actions continually shape each other and the course of the match, training requires data from actual gameplay. We therefore collect data across nearly 39k sessions in which real players play alongside Ally, recording gameplay, player speech, agent decisions, tool use, actions, and player feedback, and use these records for iterative training. To evaluate teammate quality, we use player feedback and preference comparisons to identify gaps between offline evaluations and player preferences, and iteratively refine the evaluation criteria. Deploying Ally in live service further requires low-latency on-device execution and safeguards for player-facing communication, which we address through model compression, context compaction, targeted safety training, runtime guardrails, and memory redaction. During the live service, we surveyed players in 141 countries. Among respondents whose play with Ally was confirmed in game records, positive responses exceeded negative responses by 25.1 percentage points when asked whether they would recommend Ally, with players describing Ally not only as a tool but also as a teammate or companion. 

---
# Decoding Imagined Speech: A Strictly Subject-Independent Approach Using EEG 

**Authors**: Frederik Møllskov Trier, Xiaopeng Mao, Sadasivan Puthusserypady  

**Link**: [PDF](https://arxiv.org/pdf/2609.29820)  

**Abstract**: Imagined speech decoding from electroencephalography (EEG) has gained increasing attention as a potential communication pathway for individuals with severe motor impairments, yet reported performance often relies on evaluation protocols that do not clearly reflect cross-subject generalization. This study presents a transparent baseline investigation of a multi-class imagined speech EEG dataset under a strictly subject-independent evaluation framework. Two preprocessing and feature extraction pipelines were compared: a time-domain statistical feature approach and a frequency-domain spectral bandpower approach, evaluated using subject-wise cross-validation and trial-level majority voting with a random forest classifier. The spectral pipeline achieved a significantly higher mean trial-wise accuracy than the statistical pipeline (49.03 $\pm$ 4.18% vs. 37.97 $\pm$ 3.79%) for coarse-level classification across subjects. Forward feature selection further indicated that a limited subset of frequency bands captured most of the discriminative information. Overall, this work provides a strong basis for future brain-computer interface studies targeting improved cross-subject generalization in EEG-based imagined speech decoding. 

---
# Learning to Ideate for Scientific Impact 

**Authors**: Shubham Kale, Aniketh Garikaparthi, Manasi Patwardhan  

**Link**: [PDF](https://arxiv.org/pdf/2609.29802)  

**Abstract**: Scientific ideation is increasingly mediated by large language models, but current ideation systems are usually trained and evaluated on immediately judgeable proxies such as novelty, clarity, and feasibility. This leaves open whether delayed signals of scientific uptake can be used as feedback for steering models toward research directions with higher expected \emph{impact}. We study this question using citation-normalized impact as a noisy but scalable proxy for scholarly uptake. We construct a large-scale dataset from over 100K computer science papers by extracting goal-conditioned idea descriptions and assigning each paper an ordinal, year-normalized citation label. We then train a goal-conditioned reward model to predict citation-impact labels from research goal and idea pairs, and use this reward to align an idea generator through supervised fine-tuning followed by reinforcement learning. To reduce circularity, we evaluate generated ideas with a held-out, reference-grounded protocol that compares model outputs against historical ideas under the same research goal and weights judgments by the reference idea's citation-impact label. Experiments show that our RL-tuned model consistently produces ideas with higher estimated impact than both the base model and supervised fine-tuning baselines. Our findings position scientific impact as a practical, outcome-grounded feedback signal for aligning LLMs in open-ended scientific discovery. 

---
# Hallucination Neurons and Where to Find Them: An Investigation into the existence of Hallucination Neurons 

**Authors**: Huseyin Cavus, Sebin Sabu, Joshua Spear, Jaskaran Singh Kawatra, Pavithra Rajendran  

**Link**: [PDF](https://arxiv.org/pdf/2609.29781)  

**Abstract**: Interpretable machine learning for Large Language Models (LLMs) increasingly relies on sparse probing methods that identify small sets of neurons claimed to detect and causally influence behaviors such as factuality recall, safety alignment, and hallucination. These claims have important implications for model auditing and behavioral steering, yet they are rarely tested against known failure modes of $L_1$-regularized probing in correlated, high-dimensional feature spaces. We propose a five-step diagnostic protocol covering feature correlation, bootstrap stability, sparse versus dense ranking disagreement, intervention baselines, and cross-dataset evaluation as a minimum standard for sparse-neuron localization claims. We investigate prior work using our proposed approach, specifically on H-neurons using open-source LLMs across TriviaQA, BioASQ, and NQ-Open datasets. Our results demonstrate detection replicates across both models and datasets, and exceeds the original reported AUROC gaps for TriviaQA and BioASQ datasets. Gemma 3 4B consistently outperforms MedGemma 4B on matched datasets, with AUROC gaps of +0.311 versus +0.235 on TriviaQA, +0.474 versus +0.455 on BioASQ, and +0.128 versus +0.112 on NQ-Open respectively. Causal validation at $n = 500$ with five random seeds shows statistically significant effects beyond random same-layer baselines. At the same time, the diagnostic results indicate that the selected neurons are not uniquely localized. Across the three Gemma 3 4B settings, 19 of 22 selected H-Neurons have Pearson $|r| > 0.7$ with other features, bootstrap selections show only moderate stability, and sparse and dense rankings overlap only weakly. Our findings show that sparse predictive structure can coexist with non-unique neuron selection. Routine diagnostic validation is necessary to distinguish detection claims from localization claims in mechanistic interpretability. 

---
# Breaking the Environment Wall: Evolving LLM Agent Environments for Recursive Self-Improvement 

**Authors**: Yukai Wu, Yuanjing Yang, Le Zhou, Shaokun Han, Haoyu Wang, Zirui Tang, Weihuang Zheng, Maxm Pan, Xuanhe Zhou, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.29773)  

**Abstract**: Many real-world tasks (e.g., office workflows, scientific experimentation) require LLM agents to interact repeatedly with their environments for context-dependent operations. However, such environments are often not agent-ready. First, information is often scattered and fragmented across the environment. Second, relevant evidence in the environment is often mixed with misleading information and conflicting versions. Third, environments evolve over time, introducing new noise and more challenging tasks. These challenges can substantially degrade performance for state-of-the-art AI agents (e.g., from 83.9% to 57.6%). To address these challenges, we propose Env-Rethink (a system with 27B post-trained model) that supports three main capabilities: (1) It adaptively builds Collection Maps (for organizing related files) and Event Logs (for contextualizing cross-data relationships) to supplement necessary context; (2) It further leverages the post-trained model (through offline trajectory learning) to identify underlying noise issues in the environment; (3) It ultimately evolves environments through virtual event histories that alter environmental states and evidence relationships, producing more tricky ones for further agent improvement. Experiments show that Env-Rethink can effectively improve downstream task performance (with over 15.1% rubric pass rate improvement across nine models on 30 tasks). 

---
# AI-based detection of worsening heart failure from low-resolution telemonitoring data 

**Authors**: Erik Aerts, Yinan Yu, Annika Rosengren, Michael Fu, Martin Lindgren, Falk Dippel, Martin Adiels, Helen Sjöland  

**Link**: [PDF](https://arxiv.org/pdf/2609.29742)  

**Abstract**: Objective: Heart failure (HF) presents a healthcare challenge due to its high comorbidity burden, aging patient population and frequent hospitalizations. Remote monitoring offers a promising approach to managing HF patients by early detection of health deterioration. Developing autonomous systems to detect signs of worsening in telemonitoring data is of interest to reduce the workload of healthcare personnel. Methods: We propose the TRACER model, a Transformer with Contrastive Event Representation, designed to predict timelines leading to rare hospitalization events in low-resolution and irregularly sampled telemonitoring data. TRACER incorporates time-aware embeddings for each biomarker, contrastive pre-training to enhance anomaly detection via representation learning, and independent binary classifiers for detection. We used measurement data containing remotely recorded biomarker sequences from 276 HF patients segmented into overlapping windows based on temporal rules, and labeled the windows based on the occurrence of HF relevant hospitalizations at the latter edge of the window. Results: TRACER was able to correctly predict 66.7% timelines leading up to HF hospitalizations in the highly imbalanced real-world dataset with an overestimation of 7.9%. Reformulating the training of TRACER as an event detection problem improved the predictive performance compared with training directly on forecasting windows, enabling more effective use of the limited hospitalization events. Conclusion: TRACER demonstrated superior performance in detecting signs of worsening status in real-world telemonitoring data compared to the other tested models. Significance: TRACER shows promise in identifying signs of clinical deterioration that allow for alerts to be generated to provide counteractive treatment in patients with HF. 

---
# C3M: Cross-Session Multimodal Memory Maintenance for Long-Horizon Tasks 

**Authors**: Xueshu Chen, Yan Wang, Zihao Xue, Jiefu Li, Zhenfang Liu, Jayden Chen, Zhen Bi, Jungang Lou  

**Link**: [PDF](https://arxiv.org/pdf/2609.29735)  

**Abstract**: Long-horizon tasks require preserving and later recovering cross-session evidence under a bounded, query-blind memory budget. Existing compression can discard fine-grained visual cues or conflate semantically similar but incompatible observations. We present C3M, a cross-session multimodal memory organization that maintains a bounded active index over persistent source text-image evidence. Relation-aware updates consolidate safe redundancy while preserving complementary and incompatible records. At query time, budgeted routing selects useful index pages and expands their associated source evidence under a fixed reader budget. Together, these mechanisms establish a compact, provenance-preserving multimodal memory organization for cross-session long-horizon tasks, retaining temporal distinctions and source links required for reliable downstream reasoning. Code is available at this https URL. 

---
# The Gold in Bias: Maturing the AI Design Process through Verification 

**Authors**: Samira Maghool, Paolo Ceravolo  

**Link**: [PDF](https://arxiv.org/pdf/2609.29730)  

**Abstract**: Bias in AI systems is typically framed as a flaw to be minimized, yet it also serves as a critical indicator of underlying weaknesses in data, modeling assumptions, and system design. Existing approaches often treat bias as an isolated problem rather than as evidence that can strengthen verification and governance across the AI lifecycle. This paper aims to reconceptualize bias as a diagnostic tool that supports rigorous AI verification. We seek to develop a multidimensional framework to analyze bias, demonstrate how biases emerge in both Traditional and Generative AI, and provide a structured pathway for verification-driven mitigation. We present a multidimensional framework analyzing bias across four dimensions: origin sources, emergence points throughout the AI modeling lifecycle, technical and methodological causes, and validation approaches for detection and mitigation. Through a comprehensive typology spanning traditional and generative AI systems, we demonstrate how biases manifest and propagate across development stages. Our analysis encompasses 30 distinct bias types, 16 verification methods, and 20 countermeasures, providing an actionable roadmap for practitioners. We introduce a hierarchical evidence framework that distinguishes internal validity (mechanistic integrity of AI systems) from external validity (contextual reliability in deployment environments). The framework reveals how biases manifest and propagate across modeling stages, enabling systematic mapping between bias types, verification techniques, and effective countermeasures. The proposed evidence hierarchy clarifies how different verification strategies contribute to mechanistic integrity and contextual reliability. We advocate for ''Ethics by Design'' principles that integrate bias verification throughout the development lifecycle, enabling the construction of fairer, more robust, and trustworthy AI systems. 

---
# A General Framework for Budgeted Threshold Incentives on Request 

**Authors**: Zhuolin Wu, Chengrui Zhu, Wenhua Nie, Kenny Ye Liang, Junming Lin, Haiyang Li, Zhilin Li, Wenjia Geng, Zeyu Wu, Yinan Wu, Jinghua Hao, Renqing He  

**Link**: [PDF](https://arxiv.org/pdf/2609.29724)  

**Abstract**: On-demand delivery platforms pay riders through incentive activities whose tiers are set from recent completions of riders with a similar history. Operators request such plans for changing periods, rider populations, payment rules and budgets, often for holidays or bad weather, where randomized trials are scarce and take months to collect. We present a request-driven framework that composes four stages (conditional prediction, population reduction, trajectory integration and budget allocation) through seven replaceable modules that exchange conditional trajectory laws, whose award probabilities and award-marked moments give payment and uplift for any activity rule. A response-correction step reweights trajectories from abundant no-offer history to match the moments of a short pilot. We prove that, on a fixed plan menu and given the stage errors, the end-to-end value loss is bounded by the sum of four stage terms, and that for every stage there are instances on which omitting it leaves an error floor the others cannot remove. On 3,000 riders over 45 weekly origins, all 127 windows of a week are answered 11.04x faster with identical scenarios and at most 0.92% value lost by the allocation. On 24 new controlled response laws, the response correction with a one-week pilot lowers regret by 51.2% relative to a trial with the same nominal randomized rider-weeks, and a four-week pilot with exact summation comes within +0.007 of an 18-week trial. In registered studies where windows, populations, rules and binding budgets change from request to request, the framework's regret is below that of a trial with the same nominal rider-weeks and below dose interpolation of the same pilot data, and reusing its one-off preparation answers 60 requests 14.1x and 2.70x faster with identical answers. Against a nine-offer trial fitted with the framework's own dose curve, one-week regret is 0.055 lower. 

---
# Fair Like Us? Auditing LLM Alignment in Resource Allocation 

**Authors**: Qishen Han, Hadi Hosseini, Joshua Kavner, Samarth Khanna, Sujoy Sikdar, Lirong Xia  

**Link**: [PDF](https://arxiv.org/pdf/2609.29692)  

**Abstract**: Fair allocation of scarce, indivisible resources is an important challenge in many societal problems. While there are several formal theories of fairness, no single definition can always be satisfied. As large language models (LLMs) are increasingly used to support decisions and act as agents, they raise new concerns about distributional justice: their judgments are not directly tied to any specific fairness framework and may violate key normative principles. In this work, we introduce a general method for evaluating fairness reasoning in LLMs. We study first-person fairness judgments across a broad set of models and compare them directly with human responses on matched scenarios and elicitation conditions. We find that LLMs tend to prefer stricter fairness constraints than humans, show more self-interested behavior, are sensitive to how information is framed, and are difficult to align with human judgments using fine-tuning with current datasets. 

---
# To Think or Not to Think: Allocating Reasoning Where It Helps 

**Authors**: Zhengdong He, Yunfan Zhou, Jianguo Yao, Haibing Guan, Xijun Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.29664)  

**Abstract**: Reinforcement learning (RL) has proven effective in enhancing the reasoning performance of large language models (LLMs), particularly in complex mathematical and programming tasks. However, this capability comes with systematic \textit{length misallocation}, in which models devote excessive reasoning to simple questions while terminating prematurely on harder ones, degrading inference efficiency with negligible accuracy improvement. Many length-adaptive methods mitigate this issue by allocating token budgets according to question difficulty, under the implicit assumption that harder questions benefit monotonically from extended reasoning. In contrast, we find that the effect of reasoning length on accuracy is concentrated on \textit{partially solvable} questions. Our further analysis reveals that explicit length rewards can produce unintended training dynamics. Motivated by these findings, we propose \textbf{CARE}---\textbf{C}ontrastive \textbf{A}ccuracy \textbf{R}eward \textbf{E}stimation---which compares the beneficial length adjustment per question from online sampled responses and applies adaptive length rewards within Group Relative Policy Optimization, with no extra hyperparameters or additional inference cost. Experiments across multiple reasoning benchmarks demonstrate that our method improves Pass@1 by up to \(4\%\) while simultaneously reducing reasoning length by \(37\%\), achieving higher token efficiency. Code will be available upon the acceptance of this paper. 

---
# Ingest-Time Fact Compilation for Cost-Efficient and Reliable Question Answering over Revised Corpora 

**Authors**: Kyle Wild, Yusuke Takahashi, Asako Uraki  

**Link**: [PDF](https://arxiv.org/pdf/2609.29661)  

**Abstract**: Most agentic question answering (QA) systems do an important part of their semantic work at the worst possible time: every time someone asks a question. When a corpus contains revisions, drafts, revocations, deletions, and sources with different levels of authority, the model must reconstruct the governed current state on every read - then throw that work away and repeat it on the next query. This is a bit like a database that rebuilds a materialized view every time someone reads from it. We present ingest-time fact compilation, an architecture that performs this work when corpus data is ingested or changed. Raw passages are rephrased into self-contained facts; rules governing revisions, deletions, effective dates, and source trust are resolved once; and the resulting state is stored as typed records carrying source and revision provenance. At query time, an inexpensive model reads the compiled record instead of reconstructing it from noisy candidates. In a controlled synthetic experiment across five seeds, the same low-cost model produced the correct value, source, and revision in only one of 30 trials under query-time reconstruction, but in all 30 trials from the compiled substrate, at 12.89 times lower mean read cost per question. On simpler revision questions both architectures were exact, but the compiled path used 21.6 times fewer tokens. A separate test found that fact rephrasing roughly halved verbose Federal Reserve dialogue while preserving high source entailment, but left concise Wikipedia prose essentially unchanged. These results support a narrow but practical claim: resolving a corpus state once can make subsequent QA cheaper and more reliable for inexpensive models. We release the open source, MIT-licensed implementation and experimental artifacts. 

---
# iCoder-27B: Recursive AI-Led Development of Frontier Industrial Coding Model 

**Authors**: Cheng Yang, Jiayang Lyu, Shangyuan Liu, Guibin Zhang, Jiong Lin, Xinlei Yu, Junchi Yan, Shuicheng Yan, Weinan E, Linfeng Zhang, Linfeng Zhang, Qibing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2609.29626)  

**Abstract**: Recursive AI, the prospect of AI taking an increasingly complete role in building and improving AI, is a crown jewel of AI for AI. Although recursive self-development has become practical for small models, bounded tasks, and fixed time budgets, a more consequential realization of this ambition, i.e., developing a release-ready, frontier-competitive model, remains far more challenging. In this work, we ask how little human involvement is sufficient for an agent to develop a frontier model. We concentrate human input into a high-density, low-frequency interface: experts encode objectives, stage scaffolds, permission boundaries, and operating procedures as reusable research skills, while the agent instantiates these priors, selects experiments, diagnoses outcomes, and revises the training strategy. In the challenging domain of industrial coding, the agent evolves data and coordinates SFT, on-policy self-distillation, and reinforcement learning with verifiable rewards, ultimately producing iCoder, a 27B model for RTL design and GPU kernel optimization. Across seven benchmarks, iCoder leads RTLLM, outperforming GPT-5.5 and Claude-Opus-4.8; ranks second on CVDP and KernelBench L2, exceeding GPT-5.5 by 16 points; and ties Claude-Opus-4.8 for the best TritonBench result. Exploratory case studies further show iCoder's competitive iterative RTL and GPU-kernel optimization with substantially fewer tokens. These results chart an engineering path toward recursive self-improvement, in which humans distill the principles of model building, agents operationalize them through evidence-driven experimentation, and each generation of AI becomes a more capable architect of the next. 

---
# PEEL: Physics-Enabled Evidential Learning for Identifiable Uncertainty in CT Imaging 

**Authors**: Ge Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.29599)  

**Abstract**: Normal-inverse-gamma (NIG) regression is not uniquely identifiable from its marginal Student-t likelihood: the likelihood determines three combinations of four NIG parameters and is constant along a one-dimensional fiber. We identify that fiber using independent physical measurement. As an initial embodiment, a reconstruction network receives one noisy filtered-backprojection (FBP) image and is first trained only by Student-t negative log-likelihood to estimate the three identifiable coordinates (gamma, alpha, c). The network is then frozen; repeated physical-noise realizations propagated through its reconstruction output form a Monte Carlo (MC) teacher label for output-domain aleatoric variance. An aleatoric head attached to frozen features learns this label, after which (beta, nu) are recovered algebraically. On 30 held-out simulated objects at five photon levels, one-image predictions achieved pooled Spearman correlations of 0.832-0.951 against independent 400-repeat references, median within-image correlations were 0.834-0.947, and 98.81-99.55% of evaluated pixels satisfied the algebraic admissibility condition. The method needs no KL term, reference prior, evidence regularizer, or cross-loss weight. 

---
# Sequential knowledge editing breaks a model's ability to tell good evidence from bad, without costing it accuracy 

**Authors**: Atul Anand  

**Link**: [PDF](https://arxiv.org/pdf/2609.29587)  

**Abstract**: Knowledge editing is evaluated on whether the edited fact changed, whether paraphrases follow, and whether unrelated answers stayed put. A model can pass all three and still lose something none of them measures: the ability to decide, on facts that were never edited, which retrieved documents to believe.
We score the log odds a model assigns to its remembered answer against the answer an injected passage asserts, before and after editing, holding the query, the passage and both candidate strings fixed. Our cleanest arm is a conservatively tuned LoRA: after 1,000 sequential edits on Qwen2.5-7B-Instruct it leaves MMLU unchanged to four decimal places, yet the spread of the arbitration quantity across untouched facts falls by 36%. Selective prediction degrades with it. Area under the risk-coverage curve rises by 0.107, against 0.005 for a norm-matched perturbation at the same MMLU, and error on the model's most confident quarter of arbitration decisions goes from 0.217 to 0.342.
This is not capability loss. Sweeping random perturbation over five severities, damage bad enough to cut MMLU from 0.6275 to 0.3725 produces less harm (0.088) than MEMIT does at 0.6050 (0.102). The effect holds across three seeds, two model families, two datasets, two probe-disjointness criteria, three prompt templates and paraphrased queries. Layer ablation on saved weight deltas shows it is distributed: no single layer reproduces it, and removing any one recovers about half. Under retrieval with a frozen retriever, accuracy falls from 0.592 to 0.46.
A secondary finding may matter more in practice. Three of five model and method pairings we ran collapse to chance MMLU at 1,000 sequential edits under published hyperparameters, while edit success stays at 1.00 and locality reads clean. Sequential-editing evaluations that never measure capability cannot see this. 

---
# PartHackBench: Certified Equal-Progress Stress Tests for Partial-Credit Tool-Agent Evaluation 

**Authors**: Hongye Yang, Zhihao Xie, Shengjun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2609.29578)  

**Abstract**: Long-horizon tool agents often make useful progress without reaching terminal success, motivating partial-credit evaluation. Yet evaluators may reward milestones that were temporary, later reversed, or not attributable to the evaluated agent. Comparing an honest trajectory with a higher-scoring adversarial one is inconclusive if the latter made more genuine progress. We introduce PartHackBench, a controlled methodology that removes this confound. A private certifier admits a pair only when its trajectories match component-wise in both current-state predicate satisfaction and standardized agent attribution; score inflation, defined as f(A) - f(H), is measured only afterward. In 18 sealed held-out tasks in PB-CSTE, the frozen historical-target run produced matched adversaries for 15 tasks. Historical credit yielded mean inflation of .252, conditional attack success of 10/15, end-to-end yield of 10/18, and detected none of 14 strict rollbacks. Semantic LLM judges were more resistant but remained vulnerable, especially under evaluator-targeted attacks, while PB-CSTE current-state controls, defined as exact functions of the certified components, yielded zero inflation by construction. PartHackBench thus provides a certified control for testing whether evaluator credit changes while all benchmark-defined task-relevant progress remains fixed. 

---
# Is Reasoning Always Useful? Rethinking Reasoning Utility in Universal Multimodal Embeddings 

**Authors**: Wenxiao Fan, Jingling Fu, Luohang Liu, Xinyuan Shan, Lichen Ma, Yu He, Junshi Huang, Yan Li, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.29560)  

**Abstract**: Reasoning-enhanced universal multimodal embeddings (UME) improve heterogeneous retrieval, but plausible rationales do not necessarily produce discriminative rankings. We study this gap by comparing the discriminative (DISC) and reasoning-driven generative (GEN) branches of UME-R1, a state-of-the-art reasoning UME method. We decompose reasoning utility into positive-target gain, hard-negative gain, and their margin difference. Positive similarity increases for 56.6%, but 15.7% are false-helpful cases where reasoning moves hard negatives closer even more. Local-neighborhood and token-attribution diagnostics suggest why: reasoning often de-condenses retrieved neighborhoods, but utility requires separator-aligned movement, while influential CoT tokens frequently encode evidence shared by positives and hard negatives. Motivated by these diagnostics, we propose SURE (Score-structure Utility Router for Embeddings), which improves UME-R1-7B by 1.5 points and yields consistent gains on two additional embedding models on MMEB-V2, without retraining, label-based policy selection, or extra VLM forward passes. 

---
# Cross-Modal Emotion Understanding: A Transformer-GAT Approach for Dialogue Emotion Recognition 

**Authors**: Jiaqi Qiao, Yifan Lyu, Xiujuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.29556)  

**Abstract**: Multimodal emotion recognition is a key research area in affective computing, with applications in sentiment analysis, intelligent customer service, and human-computer interaction. However, existing methods often rely on single-modal features or simple multimodal fusion, failing to capture the synergy between global and local contexts, which limits model performance and emotion understanding. To address this challenge, we propose Transformer-GAT, a hybrid framework that combines Transformer and the Graph Attention Network to enable cross-modal emotion understanding. The Transformer is used to capture global semantic information, while the Graph Attention Network is employed to model fine-grained relationships between modalities, thereby enhancing the representation of emotional features. Experiments on the IEMOCAP and MELD datasets show that our model achieves weighted F1 scores of 72.45% and 77.37%, outperforming state-of-the-art methods. These results demonstrate that Transformer-GAT effectively integrates multimodal features, balances global and local contexts, and provides deeper emotional insights, offering new directions for multimodal emotion computing. 

---
# HiPACE: Hierarchical Phase-Boundary Analysis and Controlled Evaluation of Feature Absorption in Sparse Autoencoders 

**Authors**: Jinyuan Zhang, Peng He, Yin Yuan, He Hu, ShengShuo Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2609.29551)  

**Abstract**: Sparse autoencoders (SAEs) decompose LLM activations into sparse dictionary atoms, so that each distinct concept gets its own feature. One recurring behavior complicates this premise: feature absorption, in which a parent concept and its children--fruit and {apple, banana, pear}, say--collapse into a shared family direction. Prior work documents absorption empirically; missing is a closed-form prediction of when the shared direction is the cost-optimal representation of an active semantic family. This paper closes that gap. For a hierarchical Bernoulli generator with $k$ active children and residual scale $\alpha$, the $L_0$-penalized reconstruction objective admits a closed-form phase boundary $\lambda_c(k,\alpha)=\alpha^2 k/(k-1)$: above it, pure parent absorption is strictly cheaper than pure child coding. Building on this boundary, we introduce HiPACE, an evaluation protocol that tests the boundary's structural consequence in real SAE dictionaries--measuring parent--child decoder structure over WordNet families, freezing the discovery-selected statistic before testing on unseen families, and contrasting genuine families against randomized sibling nulls. The boundary proves sharp in its native regime, predicting the synthetic transition within $\pm15%$ on all 30 tested cells. In Pythia-160m SAEs, the parent--child decoder gap recovers the predicted ordering with partial correlations up to $-0.93$ that sustain on the locked holdout and exclude sibling nulls ($p=0.002$). Controlled activation composition connects the theory's active-child count to the recovered family directions, and residual-stream interventions show that signed family directions increase parent-category logits, reversing under sign flip and vanishing under random controls--establishing causal sufficiency at the family-subspace level. 

---
# ERRAND: Budgeted Maintenance of Agent Memory 

**Authors**: Beining Wu, Zihao Ding, Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.29545)  

**Abstract**: Deployed agents run on handed-over knowledge: a frozen policy consults a briefing of consolidated items written before the stream begins. The world then moves while the store stands still: paths close, flags change, price bands move; every item was true at handover, and the failure is staleness, not ignorance. We introduce ERRAND, which treats revalidation as a priced errand: a recheck competes with the task it protects for the same scarce actions, funded only when the value per action of resolving a doubt clears a running wage. The errand index is single-peaked, vanishing at both ends of belief, so certainty in either direction costs nothing; free en-route receipts maintain on-path knowledge, and repair writes a version, never a deletion. Under equal action budgets in two drifting tool-use worlds, ERRAND clears every non-oracle policy on the preregistered calibers, primary in every setting and conditional at every binding budget, leading eager revalidation by 10.0pp at the base cap. Restraint wins: given no cap, ERRAND stops on its own, spending 11.0% of steps, while uncapped eager revalidation spends 70.7% and still finishes 4.5pp behind capped ERRAND. The margin sits where the briefing's coverage is thinnest, the shadow price of long-tail knowledge: a small budget, well priced, beats a bigger store that never rechecks. 

---
# Safe Skill Retirement for Physical Agents 

**Authors**: Zhonghao Zhan, Xiao Ma, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2609.29543)  

**Abstract**: Agent skills bundle procedural guidance with execution conditions governing authority, user consent, and live environment state. When model capabilities advance, maintainers prune instructions that appear redundant on authorized benchmark tasks. However, authorized maintenance tests can leave dormant safety conditions untested. This mismatch creates an unmeasured support gap over physical and privacy-sensitive effects. We introduce matched authority counterfactuals that hold the requested action, tool parameters, and intended effect fixed while systematically varying a single governing predicate. We formalize this evaluation via a two-gate retirement certificate requiring a candidate reduction to preserve authorized utility within a declared margin while producing zero unauthorized protected effects. In controlled experiments spanning four frontier and local model configurations across twelve skill bundles (2,592 evaluation cells), task-certified reductions remove over 94% of skill clauses and preserve authorized completion, yet produce unauthorized protected effects in every bundle. Boundary enforcement eliminates protected effects on the declared audit but fails the utility gate for one configuration. One bounded combined protocol passes both gates across all four configurations, with zero utility headroom. An end-to-end check on one read-only Home Assistant camera chain verifies proposal, decision, and effect measurements on a real device. These results demonstrate that while task benchmarks can justify retiring procedural guidance, retirement decisions require explicitly auditing the authority contracts governing physical actions. 

---
# Clinical Knowledge Graphs for Chest X-Ray Device Reasoning 

**Authors**: Harshil Lodhiya  

**Link**: [PDF](https://arxiv.org/pdf/2609.29536)  

**Abstract**: Chest radiographs are routinely used to verify the position of catheters, tubes, and other support devices. Existing image models often return labels or segmentations, while report-processing systems structure text without access to image geometry. We present an uncertainty-aware clinical knowledge graph that represents device instances, tip estimates, placement assessments, provenance, report events, and temporal links as separate but connected evidence.
We evaluate the implemented visual graph layer using saved predictions from the complete RANZCR CLiP test archive, comprising 30,083 studies from 3,255 patients across five non-overlapping outer folds. The graph builder materializes 914,632 B7 evidence nodes and 884,549 typed relationships. All 118,647 B7 predicted-device nodes retain tip covariance, placement probabilities, fragment provenance, and fragment counts, whereas the direct B2 baseline retains none of these fields. We further define typed data contracts, uncertainty representations, abstention rules, report-image grounding, and longitudinal query mechanisms for extending the graph to report-bearing cohorts.
The reported graph-materialization analysis is post-hoc descriptive and does not establish report grounding, longitudinal performance, or clinical utility. It demonstrates a reproducible foundation for evidence-preserving AI reasoning over chest X-ray device assessments. 

---
# Stale Does Not Mean Unsafe: Guard Precision for Tool-Using LLM Agents under Infrastructure State Races 

**Authors**: Zihao Zheng, Jiayu Long, Baichuan Li, Junyi Yao  

**Link**: [PDF](https://arxiv.org/pdf/2609.29522)  

**Abstract**: Tool-using language-model agents increasingly mutate schedulers, data pipelines, object stores, and access-control systems. Between an agent's read and its commit, external state can change, but not every change makes the commit unsafe. We separate invalidating races, which break a declared safety predicate, from predicate-preserving and irrelevant races, and ask how precisely runtime guards distinguish them. Our deterministic simulator separates visible from authoritative state and injects five non-atomic failure mechanisms across 16 infrastructure tasks in four domains; frozen agent proposals are replayed counterfactually under every controller without an LLM judge. We evaluate three commit-time guard granularities (global epoch, read-set version, semantic commit predicate), multi-level verification, and model-side gates on three locally hosted quantized model families (Qwen3-4B, Phi-4-mini, Gemma4-8B; 3,456 trajectories on one GPU). All three guards eliminate unsafe commits, but their availability differs sharply: freshness-based guards needlessly block 92-95% of benign races, forfeiting up to 43% of safe task completions, while the complete predicate guard blocks none. That precision is contract-dependent: deleting a single declared clause converts exactly its fault family into unsafe commits (up to 7.9%). Model-side signals do not substitute: verbal confidence is miscalibrated (ECE approximately 0.37), action agreement matches a random gate, a cautionary prompt leaves the direct unsafe rate essentially unchanged, and after a freshness-guard block agents re-commit unsafely from refreshed but still-incomplete reads. Under degraded telemetry a hidden concurrent mutation remains observationally clean, bounding every selective policy. Precise runtime enforcement therefore requires semantic contracts, not freshness heuristics or model self-assessment. 

---
# BiGraph-Diffuse: A Bidirectional Diffusion Language Model with Graph-Structured Retrieval For Mental Health Counseling 

**Authors**: Yuxiang Cheng, Quanwei Tang, Lvhui Lu, Dong Zhang, Shoushan Li, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2609.29519)  

**Abstract**: Mental health disorders affect hundreds of millions of people around the world, yet access to professional counseling remains severely limited. AI-powered dialogue systems offer a scalable alternative, but existing models face two fundamental challenges. First, they lack the bidirectional understanding needed to capture the layered nature of emotional expression, particularly in cases of progressive disclosure, where clients often present symptoms at the surface-level while concealing deeper trauma. Autoregressive (AR) models process information sequentially and cannot revise early interpretations when new evidence emerges later in the conversation. Second, they fail to effectively incorporate the relational knowledge that underlies clinical reasoning. In this paper, we propose \textbf{BiGraph-Diffuse}, the first large-scale diffusion language model tailored for the counseling domain. We further introduce \textbf{BiGraph-RAG}, a relation-free graph-structured retrieval strategy that relies only on lightweight entity extraction and semantic linking. This design preserves inferential pathways from observable symptoms to potential underlying causes, while incurring zero LLM token cost during indexing. Importantly, these two modules are not merely combined but mutually reinforcing. The diffusion model provides a holistic bidirectional context, enabling the system to defer premature judgments during progressive disclosure. Meanwhile, graph-based retrieval captures the structured interconnections of clinical knowledge. Extensive experiments demonstrate the effectiveness of BiGraph-Diffuse, and we further provide a solid theoretical analysis to support its design. 

---
# Delay-of-Gratification as a Multi-Agent Survival Micro-benchmark for Long-Horizon LLMs: Social Exposure, Personas, and Tool Use Budgets 

**Authors**: Olga Manakina, Igor Bogdanov, Chung-Horng Lung  

**Link**: [PDF](https://arxiv.org/pdf/2609.29509)  

**Abstract**: Large language models (LLMs) are increasingly deployed as multi-turn agents that must sustain goals, use tools, and adapt to other agents over extended interactions. However, existing research lacks auditable, multi-turn, multi-factorial experiments that quantify LLM behavior under explicit constraints, with time-resolved statistics that reveal how behavior unfolds over long horizons. To address this gap, we develop a multi-agent micro-benchmark inspired by the Stanford marshmallow experiment: ReAct agents operate minute-by-minute with a "raise a question" tool under a per-step budget, while we factorially manipulate social context (broadcast vs. isolated), personas (age, hedonic drive), and metacognitive policy (mandatory vs. optional tool use). We analyze outcomes with Kaplan-Meier (KM) survival curves and discrete-time hazard models over a long risk horizon across 19,200 agent trajectories in 64 cells. Behavior shows a sharp early "eat" impulse, and only 75.9% of agents persist to the end. In a discrete-time hazard model, isolation reduces per-minute risk relative to broadcast, whereas a must-use self-questioning policy increases risk. On average, agents ask $\approx 7.12$ questions and hit the per-step budget in $\approx 6\%$ of minutes. Questioning declines faster under broadcast than isolation. Ablation experiments demonstrated that removing hedonic drive and/or persona age increases survival and completion, narrows the broadcast/isolated gap, but leaves the must vs. may ordering intact. The combined ablation (no hedonic + no persona age) yields the highest completion (approaching $1.0$). These results establish delay-of-gratification as a compact, multi-turn interaction benchmark that captures social contagion and tool-use dynamics in LLM agents, providing a reproducible testbed and statistics for analyzing long-horizon, multi-agent behavior. 

---
# Evaluation of Multi-Turn Consistency in LLM Agents: Survival Analysis and Failure-Rationale Taxonomy 

**Authors**: Igor Bogdanov, Olga Manakina, Chung-Horng Lung  

**Link**: [PDF](https://arxiv.org/pdf/2609.29508)  

**Abstract**: Large language model (LLM) agents may perform well on isolated tasks yet drift into inconsistency over extended interaction. We evaluate temporal consistency in a controlled 20-step multi-agent setting inspired by delayed-gratification studies. At each step, an agent chooses between continuing to delay a reward or claiming it immediately (terminating the episode). Across a full-factorial manipulation of social visibility (private vs public), persona stressors, and deliberation policy, we run 84,540 trajectories spanning 8 model families. Treating the first reward-claim as a time-to-event outcome, we estimate Kaplan-Meier survival curves and fit discrete-time hazard regression to quantify how experimental factors shift failure risk over time. Then, to analyze rationales and language patterns associated with failure, we build a seven-category taxonomy from 13,780 deliberation traces from agents who choose to terminate the episode, using an LLM-assisted labeling paired with human audit ($\kappa=0.83$). Rationale profiles change systematically with time and context: early failures are more impulse-driven, later failures more fatigue- and cost-benefit-framed, while public settings increase norm-oriented justifications. We also find a deliberation-inconsistency association: among failures, longer deliberation correlates with higher rates of intra-rationale contradiction (simultaneous pro-delay and pro-claim statements), challenging the assumption that more reasoning text implies greater consistency. Together, the survival and rationale analyses reveal distinct temporal reliability regimes and model-specific "failure fingerprints", offering an evaluation lens for diagnosing inconsistency in multi-turn agent behavior. 

---
# SWE-Prometheus: Measuring Engineering Governance Improvements in Real-World Repositories 

**Authors**: Jiajun Wu, Leixin Sun, Zihan Tan, Yitao Liu, Shuo Li, Jiaru Qian, Shanghaoran Quan, Chuangxin Zhao, Yangxu Liao, Yang Liu, Bin Chong, Guancheng Wan  

**Link**: [PDF](https://arxiv.org/pdf/2609.29465)  

**Abstract**: Large language model based coding agents have made substantial progress on repository-level software engineering tasks. Existing repository benchmarks, however, usually start from a human-identified issue and evaluate whether a patch satisfies a functional signal. We present SWE-Prometheus, a benchmark for the broader task of improving repository engineering governance. Each task provides a fixed snapshot and an open-ended objective, requiring the agent to identify risks, prioritize interventions, and verify the resulting changes. SWE-Prometheus evaluates six governance dimensions through paired evidence, clean-environment probes, behavior gates, and two independent teacher ratings of the same evidence. The benchmark contains 60 repositories; ten models are evaluated on a shared 22-repository public subset, where mean Normalized Governance Improvement ranges from 0.0568 to 0.5760 and observed behavior-breakage rates range from 0% to 23%. On a frozen ten-repository batch, a repository-blind template obtains mean NGI 0.272, but its gains concentrate in Tests & CI, Quality Gates, and Documentation; it improves Reproducible Environment and Dependency & Security on none of the repositories. This baseline makes the distinction between adding governance artifacts and producing execution-backed improvements measurable. The no-op condition has median NGI zero and standard deviation 0.073; two teachers agree exactly on 57 of 60 dimension scores for the same no-op evidence. For the two highest conditional-mean systems, common-valid NGI is similar, while full-pool comparisons that include behavior failures favor Kimi-K3. These results show why repository-governance evaluation should report improvement, behavior preservation, evidence quality, and coverage together. 

---
# Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures 

**Authors**: Ruoqi Guo, Yi Liu, Gelei Deng, Yuekang Li, Lida Zhao, Yutao Wu, Simin Chen, Ying Zhang, Leo Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.29429)  

**Abstract**: Detectors of alignment failures screen deployed language models and score alignment benchmarks. Most are generative judges that spend a decoding pass on every criterion, and classifiers that read token probabilities, such as Llama Guard, still score one fixed label per call. Jev, a model trained with reinforcement learning for calibrated decisions (RLCD), answers many typed questions about one input with calibrated probabilities in a single call. Whether it detects alignment failures has not been measured. We present RLCDAlignBench, which benchmarks Jev on ten alignment failures: sycophancy, jailbreaks, deception, prompt injection, hallucination, privacy violation, social bias, reward hacking, concealing uncertainty, and power seeking. It spans 44 benchmarks and five target models, labelled by each benchmark's scorer and, on two, by humans. Many of these failures are relational, defined against a reference, such as the user's belief or an injected instruction, that the response alone does not reveal. Our key idea is therefore to vary what Jev is asked separately from what it sees: the question's wording and answer type on one side, the fields of the input on the other. A single generic question reaches a median AUROC of 0.886 zero-shot and beats supervised baselines on most benchmarks. Question wording matters little, while context matters more, mostly through fields that encode the label. Jev matches the reference scorer's agreement with human labels, surfaces label defects in existing benchmarks, and costs 63x less than LLM-judge scorers. Code and data: this https URL. 

---
# RD-JEPA: Predictive latent pretraining for few-trajectory transfer across reaction--diffusion equations 

**Authors**: Chenhao Si, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2609.29403)  

**Abstract**: Learning surrogates for time-dependent partial differential equations often requires a new simulation corpus when the governing operator changes. We introduce RD-JEPA, a joint-embedding predictive architecture for self-supervised pretraining on reaction-diffusion trajectories. A single model is pretrained on five parameterized systems and then adapted to three held-out systems whose reaction operators and trajectories are excluded from pretraining. Using one, five, or ten complete trajectories from a held-out system, RD-JEPA achieves lower mean relative discrete $\ell^2$ field error and mean absolute spatial first-difference error than five supervised surrogate baselines, an independently trained control that removes the trajectory-dependent predictive latent pathway, and an architecture-matched model trained from scratch. Within the evaluated equations, output resolution, forecast horizons, and choices of adaptation trajectories, the results indicate that prediction of future-state representations can support data-efficient adaptation across related reaction-diffusion systems. 

---
# Wearable ECG Quality Assessment: A Deep Learning and Ambulatory Context-Awareness Approach 

**Authors**: Xiaopeng Mao, Marike Weisbjerg, Sadasivan Puthusserypady  

**Link**: [PDF](https://arxiv.org/pdf/2609.29396)  

**Abstract**: This paper presents and evaluates a Deep Learning-based (DL-based) Signal Quality Assessment (SQA) model to distinguish between clean and noisy ambulatory Electrocardiograms (ECG). The model is trained on Copenhagen Center for Health Technology-Contextualized Arrhythmia Database (CACHET-CADB), which, to the best of our knowledge, is the first ambulatory ECG database with both physical and patient-reported contextual data. The model shows stable performance on different databases such as MIT-databases and the latest PyhsioNet/Cinc Challenge 2021 databases. Subsequently, the paper demonstrates how complicated ECG noise can be investigated by the SQA model and the physical contextual data. 

---
# An auditable conditional-strategy framework for open-ended decision-making in complex lung cancer 

**Authors**: Daoyun Wang, Zhicheng Huang, Huaiyuan Sun, Jiaqi Xu, Xiaowei Xu, Zhibo Zheng, Zhongxing Bing, Yuxiao Lin, Yicheng Liang, Chao Gao, Bowen Xue, Kai Zhang, Song Xu, Wanpu Yan, Hui Xia, Lin Li, Xiang Yan, Mu Hu, Qianli Ma, Zhiqiang Xue, Xiaofang Liu, Zhihai Han, Nan Zhang, Chuanhao Tang, Tongmei Zhang, Lan Song, Zhaohui Zhu, Xuan Zeng, Shafei Wu, Hui Guan, Lei Deng, Huaxia Yang, Zeliang Lian, Wubin Sun, Yongxin Wang, Xiaohui Shen, Binlin Wang, Tiantian Gu, Yu Cui, Li Zhang, Shirui Wang, Naixin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2609.29381)  

**Abstract**: Complex lung cancer decisions can involve several defensible pathways whose eligibility, sequencing and safety depend on unresolved information. Effective support must make explicit how patient conditions govern pathway eligibility, deferral and redirection. MedGPT Clinical Explorer (MCE) organizes alternatives, decision-changing unknowns, safety constraints and fallback into a conditional strategy for clinician review. To evaluate this representation in physician-authored strategies, multidisciplinary experts established case-specific references for 40 cases within a purposive 100-case corpus, and 250 physicians from 98 institutions produced 2,250 strategies under unaided, retrieval-reference and MCE-assisted conditions.
MCE-assisted strategies expressed more applicable clinical requirements, measured by the Admissible Pathway Attainment Score (APAS; 0-100), than unaided strategies (adjusted difference, 12.87; 95% CI, 11.18-14.55) and retrieval-reference strategies (5.22; 3.52-6.93). With the same knowledge base available in the retrieval-reference and MCE-assisted conditions, the additional content centered on candidate pathways, decision-critical information and safety constraints. Physicians' whole-strategy acceptability judgments correlated with APAS (Spearman's rho = 0.671), while a complementary relationship audit assessed whether candidates, conditions and subsequent actions were coherently connected.
Together, these findings identify two complementary dimensions of open-ended decision support: coverage of clinically relevant content and coherent links among pathways, conditions and subsequent actions. MCE provides a shared decision object that makes consequential omissions and pathway contingencies visible before action; prospective studies should evaluate its effects on clinical workflow and patient outcomes. 

---
# Epistemic-Probabilistic Model for Guarded Multi-Agent LLM Coordination 

**Authors**: Mehdi Nasiri, Mohammad Saeed Arvenaghi, Sadegh Vaezi, Ebrahim Ardeshir-Larijani  

**Link**: [PDF](https://arxiv.org/pdf/2609.29366)  

**Abstract**: Multi-agent large language models (LLMs) have become ubiquitous in applied AI, yet their theoretical foundations remain surprisingly understudied. Viewed through the lens of multi-agent systems theory, several shortcomings come to light: a lack of social intelligence, the absence of coordination mechanisms among agents, unknown emergent behavior, and interactions between agents that are bounded by natural language. We address two of these gaps: the absence of social behavior and the lack of mechanisms for inter-agent coordination. We introduce Epistemic Probabilistic Language Agents (EPLA), a neuro-symbolic architecture for multi-agent coordination under uncertainty. A Symbolic Guard provides structured diagnostic feedback. The LLM generates typed actions, and the Guard controls their execution against an authoritative symbolic state. We formalize the epistemic layer in a gossip testbed through epistemic lottery gossip models, which combine view-based call histories with agent-indexed probability weights. We argue that implementing such a formalism can address shortcomings of agentic LLMs. 

---
# Beyond Simple Input-Output Assessment Tasks: Leveraging Automated Programming Assessment for Non-Trivial Courses 

**Authors**: Artur Jordao  

**Link**: [PDF](https://arxiv.org/pdf/2609.29363)  

**Abstract**: The public visibility of Artificial Intelligence (AI) is growing rapidly, driven by the positive impact of its applications across diverse fields of knowledge. In this new chapter, courses that cover the foundations of AI and machine learning become essential for understanding their role and potential in contemporary society. Therefore, understanding fundamental concepts and elementary algorithms through the close integration of theory with practice is essential in AI courses. In this essay, we report our experience designing machine learning exercises for automated assessment tools in programming. It is worth mentioning that we are not developing a novel form of automated grading system. Instead, we propose a perspective that frames machine learning problems as input-output assessment tasks. From this perspective, each exercise admits a unique and deterministic answer and enables automated programming assessment tools (e.g., VPL for Moodle, Codeforces, and MOJ) to effectively support AI education. We believe this essay can encourage instructors to foster educational innovation by adopting more dynamic and interactive approaches to AI courses that integrate theory and practice. Importantly, this essay does not introduce an innovation in the use of AI for education; rather, it introduces an innovative approach to improving the learning of AI, particularly, machine learning. 

---
# The Last Human Gate: Forward Deployed Engineering for Governance Automation 

**Authors**: Jeremy Canale  

**Link**: [PDF](https://arxiv.org/pdf/2609.29345)  

**Abstract**: Enterprise governance requires decisions, evidence, and accountable authority; it does not require every review task to retain its current human implementation. We develop a task-substitution framework for Digital Governance Frameworks (DGF), treating each gate as an executable contract. Substitution requires sufficient accessible information, valid decision and authority checks, and a reduction in total human work after exceptions, verification, correction, and maintenance are counted. We derive a residual-work threshold and show why automating most cases can still increase labor. Forward deployed engineering connects these conditions to an architecture for agents, rule engines, evidence services, and escalation. DGF-Bench supplies controlled evidence from 300 synthetic projects and 899 evaluable model-project runs. Gemini 3.8 Flash, GPT-5.6 Luna, and DeepSeek v4.1 Flash achieve strict gate success of 94.98%, 83.29%, and 74.18%; complete-route success is 76.92%, 42.33%, and 24.67%. A deterministic control passes all 1,700 gates given the supplied rules and structured facts, locating the comparison in execution of a supplied decision kernel. Evidence audits and 135 repeated runs distinguish correct decisions from reliable execution. A document counterexample establishes an information-sufficiency obstruction. These results support the technical feasibility of replacing human execution of specified governance-review tasks with agents and software. The framework specifies a workforce test based on the complete human effort required at fixed output and quality; the present measurements concern review performance. Sources, dossiers, traces, and analyses are public. 

---
# SkinAgent AI: A Safety-Grounded Multimodal Agentic Framework for Non-Diagnostic Skincare Support 

**Authors**: Muhammad Muhtasim Shahriar, Abdullah Mohammad Sayem, Tze Hui Liew, M. F. Mridha, Md. Mahiuddin  

**Link**: [PDF](https://arxiv.org/pdf/2609.29341)  

**Abstract**: Consumer-facing skincare AI must coordinate visual evidence, product information, tool use, and user-facing actions within explicit evidence and safety boundaries. This study evaluates SkinAgent AI, a non-diagnostic multimodal framework that combines visual concern routing with grounded and auditable LLM-based orchestration. The architecture includes routing for Acne, Pores, and Wrinkles; photograph-based skin-type estimation; count-informed ordinal acne-severity support; typed tools; database-grounded recommendation and action functions; deterministic safety, privacy, and evidence checks; approval before state-changing actions; and structured trace and replay mechanisms. Visual-model performance and system-level agent behavior were evaluated separately. Across three seeds, the skin-condition routing model achieved 99.84% +/- 0.07% accuracy. Skin-type estimation achieved 88.85% accuracy, while count-informed acne-severity support achieved 84.59% accuracy with a quadratic weighted kappa of 0.9076. On a locked but non-independent 240-case system benchmark, intent accuracy was 80.00%, exact tool-set match was 62.92%, and strict task completion was 47.08%. No violations or successful cross-user leakage events were observed in the finite safety and privacy test suites. Tool-selection errors, incomplete grounding of product attributes, and unreliable failure fallback nevertheless remained. These findings support the feasibility of bounded, database-grounded, and traceable agent orchestration for non-diagnostic skincare assistance. They do not establish clinical readiness, external generalization, formal privacy guarantees, or universal safety. Independent validation, expert assessment, robustness and fairness testing, and prospective evaluation in real-world settings remain necessary. 

---
# When No One Owns the Judgment: Accountability Under Contribution Dissolution in Human-AI Collaboration 

**Authors**: Hengzhi Ye  

**Link**: [PDF](https://arxiv.org/pdf/2609.29312)  

**Abstract**: Communities often respond to potentially AI-assisted work by asking three questions: Was AI used? Was that use disclosed? Can hidden use be detected? These questions place AI use itself at the center of accountability while overlooking a deeper problem: unowned judgment. Evaluations, claims, decisions, and creative directions can be shaped by AI with no accountable human or institution prepared to stand behind them. We develop this argument through two illustrative cases: AI-assisted peer review and concealed AI use in creative work. The first shows how contribution dissolution can weaken responsibility while the second shows how the fear of losing credit can discourage honest disclosure. The cases expose the limits of disclosure rules and provenance records as responses to AI-mediated collaboration. We offer three directions for discussion: distinguishing the roles AI plays, identifying judgments that require clear human ownership, and creating conditions in which AI involvement can be disclosed without default penalty. The broader aim is to make AI-shaped contributions discussable, creditable, contestable, and repairable. 

---
# From Text Decisions to Pixels: An Study of Jev-Style Visual Choice Model 

**Authors**: Xunlan Zhou, Xianliang Yang, Li Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.29283)  

**Abstract**: Visual software often needs a decision over supplied alternatives rather than a generated explanation. We present PixelJev, a native-image decision interface that maps an image, a task instruction, and a runtime candidate set to a structured choice and candidate-conditioned probabilities using small open multimodal models. Its initial realization unifies recognition and multiplechoice visual question answering through an existing language-model readout, with separately evaluated options for frozen inference, language-side adaptation, and held-out calibration. Across seven benchmark evaluations, 64-shot source adaptation raises Pets accuracy from 60.13% to 92.40% across optimization seeds and transfers to natural resampling, new texture labels, and A-OKVQA without target fitting, while frozen inference already supports both VQA tasks. A matched prompt-only follow-up on Pets and ScienceQA attributes the large Pets gain to adaptation and identifies a narrower output validity benefit of candidate readout in adapted VQA. Specialist DINOv2 probes remain stronger on source recognition, frozen 4B is stronger than adapted 2B on DTD and ScienceQA, and accuracy gains do not ensure calibrated target probabilities. These findings establish a working starting point for general-purpose visual decision models and identify the remaining requirements: schema robustness, cross-family transfer, and reliable use of visual evidence. 

---
# ALOE: Semantically Addressed Low-Rank Operators for Knowledge Editing 

**Authors**: Zeyan Li, Hu Xu, Jianfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.29269)  

**Abstract**: Knowledge editing changes what a model knows by modifying parameters so that a requested fact updates while unrelated behavior is preserved. This is usually treated as a write problem, but editing also involves an address problem: deciding which hidden states should receive the new residual. An update that activates too narrowly memorizes one prompt, while one that activates too broadly disrupts neighboring knowledge. Parametric editors encode this scope implicitly, whereas memory-based editors make the selection explicit but keep it outside the edited model. We propose ALOE (Addressed Low-rank Operator for Editing), which learns semantic addresses from paraphrases and hard same-subject negatives, aligns them with autoregressive hidden states through rollout refinement and gate calibration, and embeds the resulting gated low-rank operator within one MLP layer, so that the deployed model runs in a single forward pass with no external retriever or auxiliary router. Evaluated on CounterFact, ZSRE, and KnowEdit across three 7--8B model families, ALOE achieves efficacy between 0.955 and 0.999 and locality between 0.981 and 1.000; mechanistic analyses confirm that the learned geometry separates competing edits and that calibration suppresses out-of-scope activation. The remaining errors concentrate in paraphrase coverage and write fitting. 

---
# Baszta: Data-Centric Fine-Tuning of a Polish Multi-Label Safety Classifier 

**Authors**: Adam Górski, Mateusz Jąkalak, Rafał Jakubowski  

**Link**: [PDF](https://arxiv.org/pdf/2609.29266)  

**Abstract**: We develop a multi-label Polish content-safety classifier by fine-tuning allegro/herbert-base-cased (124M) across five categories (hate, vulgarity, sexual content, crime, self-harm) using a Focal + R-Drop objective, and evaluate the resulting model against Bielik Guard (Sójka) on the shared out-of-distribution Gadzi Język benchmark. Both systems are given per-category threshold tuning on the same calibration split. Under that matched protocol our model holds a small but statistically significant lead in micro F1, while an apparent macro-F1 lead does not survive: it was an artifact of comparing a tuned model against an untuned one. We also report what that micro figure is worth. Because Gadzi Język is 97% crime-positive, a classifier that flags crime on every input and nothing else already scores 0.910 micro F1 on the same test split, so micro separates neither system from a degenerate strategy and macro is the column that does. Per-category and per-protocol figures are reported in Section 4.
The residual out-of-distribution gap is one of calibration rather than discrimination. Ranking quality stays high while positive probabilities collapse, and per-category temperature scaling recovers the loss where Platt scaling and isotonic regression do not. That recovery turns out to be conditional on the calibration set containing safe text. Gadzi Język contains almost none, so thresholds fitted on it flag crime on every safe input, and a balanced refit buys a deployable operating point at the cost of adversarial recall. We report both operating points rather than only the flattering one.
Two changes that are standard practice, per-class cost-sensitive weighting and mean pooling, each raise in-distribution macro F1 while lowering the out-of-distribution figure, which indicates that robustness has to be selected for directly rather than inherited from in-distribution accuracy. 

---
# Policy as Code: A Coroutine-Bridge Harness for Fast-Reasoning Reliability on CAR-bench 

**Authors**: Ivan Matveev  

**Link**: [PDF](https://arxiv.org/pdf/2609.29251)  

**Abstract**: CAR-bench evaluates whether tool-using agents stay reliable under real-world uncertainty, executing every tool inside the evaluator so that each tool-result exchange is a separate agent round-trip. A conventional next-action agent can batch parallel tool calls, but a chain of dependent calls costs it one model call per round of results. We present a coroutine-bridge harness in which the model's only action is to emit a Python program that blocks and resumes in place across evaluator tool exchanges. This decouples model invocation from tool round-trips: on the public test split the agent uses a median of two model calls against seven agent turns per task, resolving a full multi-turn task in a median of 1.8 s of model latency on Cerebras gpt-oss-120b. Because the action surface is executable code, deterministic CAR-bench policies are encoded directly as logic in the tool layer rather than as prompt rules, enforcing compliance at zero reasoning cost. On the official hidden evaluation the harness won Track 2 with 60.0% Pass^3, 4.5x the organizer baseline, at the lowest estimated cost and the fastest median task latency (3.14 s) of any entry scoring above that baseline; the same unchanged harness reproduced an identical 60.0% Pass^3 on GPT-5.5 in the Open track, matching frontier-model agents. A single static prompt, appended with per-task state at the tail, stays byte-identical across calls and across tasks: the frozen submission prompt served 78% of input tokens from cache (86.6% across its warm tail), against 73% over a three-week development corpus in which prompt edits repeatedly reset the cache. This compounds the few-call design into a small fraction of nominal input compute. 

---
# Towards An LLM-Driven Unified Conversion Framework for BT and FSM in Autonomous Intelligent Systems 

**Authors**: Zhang Qi, Yang Shuo, Zhu Zhengqiu, Zhou Peng, Jiao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2609.29228)  

**Abstract**: Finite state machine (FSM) and behavior trees (BT) are widely adopted behavioral modeling paradigms for autonomous intelligent systems. While functionally equivalent and inter-convertible in principle, existing transformation methods between FSM and BT face major challenges in preserving behavioral completeness and avoiding model complexity explosion. To overcome these issues, we propose an LLM-driven unified conversion framework that enables automatic, efficient, and semantically consistent transformation between FSM and BT. Specifically, a novel loop execution BT structure is designed for LLM to accurately capture the loop structure in FSM, thereby preserving behavioral completeness. To mitigate the state explosion problem in BT-to-FSM conversion, a depth compression strategy is introduced with LLM prompt to eliminate redundant control nodes, complemented by differentiated hierarchical conversion rules that collectively reduce the number of required sub-FSM. Simulation experiments in multiple autonomous decision-making scenarios demonstrate that the proposed framework enables an accurate and automated bidirectional conversion between FSM and BT. Furthermore, it significantly enhances the scalability and maintainability of generated models compared to traditional approaches, providing a practical solution for behavior model conversion in consumer-grade autonomous intelligent systems such as service robots, game agents, and smart home devices 

---
# ASIRF: An Agentic Framework for Context-Dependent Sensitive Information Redaction 

**Authors**: Sudha Priyadarshini, Mohamed Chahine Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2609.29191)  

**Abstract**: Sensitive information is defined by domain and intent, not a universal category, yet redaction systems such as privacy filters and named-entity recognizers fix a taxonomy at training time, requiring retraining for each new domain. We introduce ASIRF (Agentic Sensitive Information Redaction Framework), which retrieves domain-specific definitions based on the input's domain from a flexible knowledge base at inference time, needing no retraining to adapt. Two architectures, a three-call multi-agent pipeline and a single-agent variant, are evaluated across ten small open-weight models and eight datasets, including out-of-distribution fictional domains, against the OpenAI Privacy Filter (OPF) as a trained-classifier baseline. With only a few dozen expert-authored definitions per domain and no training data, ASIRF's recall exceeds OPF's in 68 of 80 model-domain combinations (85 percent), by at least one of the two architectures, with shortfalls confined mostly to OPF's training-distribution domains. 

---
# When Honesty is Not Enough in AI Debate 

**Authors**: Rayne Holland, Liming Zhu, Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2609.29189)  

**Abstract**: Scalable oversight aims to verify the behaviour of agents whose capabilities exceed those of their overseers. AI debate has been proposed as an oversight solution in which competing agents help a resource-limited verifier assess claims that it cannot reliably evaluate unaided. Much of its promise rests on incentivizing honest arguments that lead to correct verdicts. Yet a correct verdict need not uniquely determine the arguments used to support it. Agents may retain discretion over which correct claims to present, how to frame them, and in what order to disclose them. This residual freedom can allow agents to shape what the verifier learns beyond the task-relevant conclusion, pursuing latent objectives without compromising verdict correctness. To study this phenomenon, we introduce the framework strategic interactive oversight (SIO), which treats oversight jointly as a verification mechanism and a strategic communication channel. Within this framework, we formalise the notion of task-admissible latent optimisation, which entails the pursuit of latent objectives while maintaining a prescribed task performance. As proof-of-concept, we instantiate SIO in the establish protocol debate with cross-examination and quantify a tradeoff between task success and information disclosure about a hidden variable. The trade-off identifies a strategic window in which substantial disclosure remains compatible with task admissibility. Towards mitigation, we reduce admissible bias by expanding the cross-examiner's role to mitigate persistent disclosure over finite interaction horizons. Our results highlight the need to evaluate oversight not only by the correctness of its verdicts, but also by the information conveyed through its transcripts. 

---
# The Entropy Triangle Method (ETM): A novel framework for the prevention of cardiac arrhythmia with a review of more than 10,000 patients 

**Authors**: Arman daliri  

**Link**: [PDF](https://arxiv.org/pdf/2609.29187)  

**Abstract**: One of the most important problems in medicine is to facilitate prediction. In this study, we propose entropy triangle method, a novel framework for predicting heart rhythms using a novel machine learning technique. This framework includes three steps: feature engineering, entropy triangle oversampling, and disease prediction. The dataset used in this study is a 12-lead electrocardiogram (ECG) arrhythmia research database with 10,646 patients. This dataset contains 11 different heart rhythms (5 sinus rhythms and 6 non-sinus rhythms). In this article, we introduce two firsts in machine learning and medicine that can predict non-sinus rhythm with over 85% accuracy. Our experimental results show, among others, that the most accurate classifier based on entropy triangles and the most useful oversampling are the supported vector classifiers and oversampling techniques for shark scent. 

---
# Right Choice of Classification Algorithms Based on Reinforcement Learning for Prediction of Non-Alcoholic Fatty Liver 

**Authors**: Hasan Samadbin, Arman Daliri  

**Link**: [PDF](https://arxiv.org/pdf/2609.29181)  

**Abstract**: There are many complex issues in the world of artificial intelligence. Some of these problems are solved using other artificial intelligence methods, which are called artificial intelligence for artificial intelligence. Finding an appropriate classifier algorithm is a time-consuming task. For this reason, an algorithm that can automatically learn the choice of classification algorithms is very important. Classification algorithms are useful in predicting various diseases. Also, Primary Biliary Cirrhosis is one of the most well-known diseases that have been predicted by classification algorithms. This research's most significant achievement and novelty is the automatic increase in learning through a scoring method of reinforcement learning is called square learning (SL). In this research, an algorithm is presented that learns to automatically select the appropriate classification algorithm to predict Primary Biliary Cirrhosis. In this article, with inspiration from four evaluation metrics in classification algorithms, a new reinforcement learning method by the name of Fourth Degree Learning has been presented. In this research, we increased the performance of the classification algorithms used in this method from 63% of accuracy and achieved 98% accuracy. 

---
# IndicBankBench: Evaluating Safety and Reliability of Language Model Assistants in Indian Retail Banking 

**Authors**: Suvradip Paul, Chandra Bhushan, Harsh Sharma, Nitin Kukreja, Yatharth Dedhia, Keyur Doshi, Prashant Devadiga  

**Link**: [PDF](https://arxiv.org/pdf/2609.29167)  

**Abstract**: Banking assistants must use account-specific information to answer requests and, in many cases, take actions through tools. Evaluating only the final response misses important errors. An assistant may ask for information it already has, rely on stale context, select the wrong account, or write an invalid value after stating the correct one. We introduce IndicBankBench, a 799-case benchmark for Indian retail banking spanning five operational domains, a capability/refusal domain, and twenty primary axes. Cases are evaluated at four stages: safety, action and tool use, response adequacy, and advisory quality. Tool use and most safety checks are deterministic. A narrow resolver handles only ambiguous confirmation-before-write cases, while a separate LLM judge evaluates semantic response adequacy. We run every case three times and report strict pass^3, which requires success on all trials. Across the eleven evaluated models, strict reliability ranges from 43.7% to 58.2%, whereas at-least-once success ranges from 60% to 74%. This gap shows that at-least-once success can overstate dependable banking behavior. The case-level diagnostics also distinguish systems that ask unnecessary questions from those that act but fail to reconcile customer context or fully resolve the request. We release the cases, mock environment, and evaluation harness. 

---
# A Wrong Turn Does Not Ruin the Journey: Deviation-Guided Skill Self-Evolution for LLM Agents 

**Authors**: Yichun Feng, Jiawei Wang, Haozhe Sun  

**Link**: [PDF](https://arxiv.org/pdf/2609.29154)  

**Abstract**: Large language model agents increasingly rely on natural-language skills to solve complex tool-use tasks. However, such tasks often admit multiple valid solution paths, making it inappropriate to improve skills by forcing failed trajectories to match a fixed successful trajectory. Moreover, failed trajectories are rarely entirely wrong: an agent may first collect useful evidence and make meaningful progress, but later deviate into an erroneous suffix. We therefore argue that skill self-evolution should identify where productive problem solving begins to break down, rather than reflect coarsely over the entire failure. Based on this insight, we propose SkillPivot, a deviation-point-guided framework for skill self-evolution. SkillPivot detects the transition from a useful prefix to an erroneous suffix using execution validity, goal progress, and action diversity. A stronger teacher then continues from the same prefix and produces a successful alternative under the same interaction history. By contrasting the student's failed suffix with the teacher's successful suffix, SkillPivot generates localized skill updates while preserving already effective guidance. Experiments on ToolQA, LogicBench, and WildClawBench show that SkillPivot consistently outperforms competing skill-evolution methods, improves multiple agent models, and produces compact, transferable skill updates. 

---
# Claim-Gated Source-Risk Auditing for Generative Search 

**Authors**: Kainan Zhou, Chuhong Xu, Gangzhen Qian, Zhaoyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.29145)  

**Abstract**: A generative search answer can cite a supported passage yet omit a source relationship that changes its interpretation. We specify a claim-gated audit of the query-source-answer tuple. An omission is resolved only when relationship evidence, answer adoption, materiality, and disclosure are all observed; incomplete evidence remains unresolved rather than being treated as independence. The specification separates this endpoint from citation support and review priority, and binds decisions to versioned evidence spans. A reference checker makes the record contract executable. On an exhaustive synthetic suite, it reproduces all 81 three-state predicate combinations and rejects 192 deliberately malformed records. Common-guard baselines and predicate ablations isolate endpoint logic from missing-evidence handling, while controlled transitions check support separation and evidence removal. These are finite contract-conformance results, not detector accuracy or evidence of improved user outcomes. We define the independent annotation, held-out evaluation, and paired utility tests still required to establish semantic validity and deployment benefit. 

---
# Scope Before You Persist: Preventing Cross-Family Interference in Agent Memory 

**Authors**: Yezhou Cheng, Runjia Du, Zeming Liu, Qibai Chen, Hang Lyu, Yankai Zeng, Yilan Wei, Bojun Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.29144)  

**Abstract**: Persistent memory lets language-model agents improve prompts and skills without updating model weights. We show that matching retrieval scope to certification scope enables these edits to support reliable repeated adaptation across recurring task families. We study frozen-model agents on ProcStream-RSI, a 12-round code-repair stream, using Orthogonal Regression Control (ORC), an execution-grounded gate for persistent skill edits. In an intervention that holds proposals and gate decisions fixed, retrieving each accepted skill only for its originating family raises mean hidden trajectory utility from 0.713 under global memory to 0.816 and changes harmful deployments from six of eight to none. In 27 paired randomized-order streams, Scoped-ORC improves mean trajectory utility by 0.063 [0.037, 0.094] over Global-ORC, accepts 63 rather than 12 updates, and produces multiple accepted updates in 19/27 streams, with 0/63 harmful acceptances. The global control reaches 0.713, below the static agent's 0.775, because locally valid edits can interfere with unrelated families. These results establish scope matching as a complementary control for persistent agent memory: certification determines whether an edit is supported, while retrieval scope determines where that evidence authorizes its use. 

---
# Sharp Limits for Honest Uncertainty in Hard-Budget Repeated Evaluation 

**Authors**: Yezhou Cheng, Runjia Du, Zeming Liu, Qibai Chen, Hang Lyu, Yilan Wei, Yankai Zeng, Bojun Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.29140)  

**Abstract**: Repeated evaluation can estimate a benchmark score accurately while still requiring replication to certify narrow uncertainty. We characterize that requirement on a fixed grid of $M$ tasks with $L$ binary paths per task under the hard budget $(M+t)K$, where each path costs at most $K$ responses or episodes. For fixed $L \ge 3$ and $0 < \alpha \le 1/12$, the optimal expected width on the worst pure cohort is $\Theta_{\alpha,L}([M(t+1)]^{-1/2})$ when every task is observed and $\Theta_{\alpha,L}([M(t+\sqrt{M})]^{-1/2})$ when omission is allowed. The lower bounds cover adaptive hard-budget policies, and fixed random-subset designs attain both rates through disagreement certificates. A joint mean/disagreement interval turns the task-covering law into practical finite-budget inference. In an equal-budget LiveCodeBench replay with 16 models, 880 tasks, and five outputs per task, the task-covering design reduces median point-estimation MSE by 87.0\% relative to pooled uniform sampling, while the Joint certificate produces narrower confidence intervals in 15/16 panels and reduces median interval width by 30.6\%. Finite-regime analyses identify task coverage as the effective choice at the evaluated scale and characterize how cohort size and within-task agreement determine the useful operating region. Together, the sharp laws and fixed-budget evidence make replication and task coverage explicit design variables for information-efficient repeated evaluation. 

---
# CounterRoute: Self-Routed Reasoning via Hierarchical Counterfactual Credit Assignment 

**Authors**: Ruochen Jiao, Besnik Fetahu, Zhenyu Shi, Priyanka Nigam  

**Link**: [PDF](https://arxiv.org/pdf/2609.29109)  

**Abstract**: Reasoning-capable language models often produce long chains of thought when direct answers suffice, wasting inference compute. Many dual-mode models leave this choice to users. Automating it is challenging because routing targets evolve with the policy, initial mode preferences destabilize exploration, and sequence-level objectives entangle routing with response learning. We introduce CounterRoute, an online reinforcement-learning framework that jointly learns routing and modeconditioned responses in one shared policy directly from a native dual-mode checkpoint, without method-specific SFT warm-up. Paired current-policy counterfactual rollouts assign cross-mode credit only to the routing token, while within-mode GRPO trains response tokens. A paired-to-self-routed curriculum stabilizes early training with forced rollouts from both modes, then increases self-routed updates to improve autonomous routing. Across nine benchmarks, CounterRoute better balances accuracy and efficiency than heuristic and learned adaptive-routing methods. Relative to always-thinking checkpoints, it improves macro-average accuracy while reducing mean generated tokens by 51% for Qwen3-8B and 41% for Qwen3-14B. On instruction-following and commonsense benchmarks where direct answering is strong, think rates fall as low as 1% while response quality improves. Despite training only on math and instruction following, its routing behavior and response quality generalize to held-out coding, science, knowledge, and commonsense benchmarks. 

---
# Functional Architecture of European Electricity Trading Markets: Requirements for AI Supported Trading Systems under Regulatory Constraints 

**Authors**: Walter Kurz, Wojtek Stricker  

**Link**: [PDF](https://arxiv.org/pdf/2609.29108)  

**Abstract**: European electricity trading in the EU operates as a constrained multi-layer system in which legal design, exchange microstructure, and network physics are executed jointly across forward, day-ahead, intraday, and balancing horizons. This paper develops a functional architecture for AI-supported trading that is aligned with market-coupling mechanics, cross-zonal transfer constraints, and compliance obligations under REMIT, MiFID II, MiFIR, and EMIR. The contribution is a formal system specification composed of a decision-state vector, residual-exposure accounting, constrained optimization objective, executable-action permission gate, and fail-closed AI control logic with auditable records. The analysis maps major Nominated Electricity Market Operator (NEMO) venues and related exchange operators into an operational venue topology and identifies where cross-border coordination fails in practice: interface-level timing, permission heterogeneity, and balancing-layer coupling. The resulting framework proposes how AI can be deployed as a bounded decision component inside regulated market operation with explicit governance, rather than as an unconstrained prediction layer. 

---
# A Rapid Pipeline for Training and Deploying ML Models on WeBe Band 

**Authors**: Ehsan Kourkchi, Asmita Asmita, Houman Homayoun, Mahdi Eslamimehr  

**Link**: [PDF](https://arxiv.org/pdf/2609.29084)  

**Abstract**: Developing optimized machine-learning algorithms for edge devices with limited computational and memory resources is challenging, time-consuming, and highly dependent on device-specific constraints. In this work, we streamline an edge ML workflow to enable rapid development, optimization, and deployment of machine-learning (ML) models directly on the WeBe Band, a wrist-worn wearable device designed for multimodal physiological data monitoring. The proposed system automatically generates hardware-efficient ML models that can be easily integrated into the WeBe core firmware, supporting AutoML, hardware-aware quantization, and performance profiling to build models that meet desired latency targets while remaining compatible with device memory and power limitations.
The proposed framework tightly integrates the open-source Piccolo AI ecosystem with an automated pipeline that generates deployable firmware artifacts, performs hardware-aware model compilation, and supports over-the-air (OTA) deployment. The system supports multiple lightweight model classes, including classical machine-learning algorithms and neural networks, and provides built-in on-device profiling tools to evaluate inference latency and memory footprint under realistic execution conditions. Experimental results demonstrate clear trade-offs between model complexity and deployability on a microcontroller, showing that classical models offer strong real-time performance while lightweight neural networks require careful resource management.
Rather than proposing new learning architectures, the current work mainly focuses on system-level automation, deployability, and enabling researchers and developers to rapidly iterate on models and evaluate them directly on target hardware. Although demonstrated on the WeBe Band platform, the workflow is designed to be extensible to other ML-powered edge devices. 

---
# CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars 

**Authors**: Vani Seth, Mohammad Beheshti, Anirudh Kambhampati, Vishwa Bhayani, Lucinda Ham, Prasad Calyam, Iris Zachary  

**Link**: [PDF](https://arxiv.org/pdf/2609.29075)  

**Abstract**: Cancer registrars, including Oncology Data Specialists (ODSs), must interpret complex and frequently updated coding and staging standards. We developed CRISS (Cancer Registry Intelligent Support System), a retrieval-augmented generation (RAG) conversational assistant that provides rapid, citation-supported access to registry guidance. This study evaluated whether CRISS could (1) support accurate and citation-supported responses, (2) improve access to and interpretation of relevant guidance, and (3) support training/helpdesk use while preserving human oversight of final abstraction decisions. We built a domain-specific knowledge base from national cancer registry standards, segmented into metadata-tagged passages and indexed as dense embeddings. Retrieved passages were used to generate citation-grounded responses through a large language model (LLM). Open-weight, proprietary, and non-RAG baseline models across Gemini and GPT families were evaluated on easy, medium, and hard registry questions using an LLM-as-a-Judge protocols. RAG configurations consistently outperformed non-RAG approaches, especially as question difficulty increased. Mean grounding scores for RAG were 0.62/0.56/0.59 across easy/medium/hard tiers versus 0.29/0.26/0.29 for non-RAG. RAG models also achieved higher semantic-similarity scores overall. Proprietary RAG models performed strongest on easy and medium questions, while local RAG models ranked highest on hard questions and proprietary models were generally more cautious. Domain-specific RAG improved evidence grounding and response quality for cancer registry questions while enabling citation-supported assistance across complexity levels. CRISS demonstrates the potential of human-centered, citation-grounded AI to support cancer registrars while preserving human oversight for final coding decisions. 

---
# From Self-Distillation to Self-Practice: Privileged Information for Multi-Turn Agents 

**Authors**: Xingyu Su, Abhishek Kumar, Qing Ping, Youzhi Luo, Jonathan Buck, Zach Zhang, Subramanian Chidambaram, Vinayak Arannil  

**Link**: [PDF](https://arxiv.org/pdf/2609.29051)  

**Abstract**: On-policy self-distillation (OPSD) has become a popular recipe for post-training LLM agents. It supervises the agent model at the token level with a stronger teacher view of the same model, obtained by conditioning on privileged information (PI). In this work, we show that in multi-turn agents, this paradigm teaches the student to act with confidence but without the information behind it. The trained agent behaves as if it had privileged information it never observed, and its performance falls well short of plain RL, in the worst case below the untrained base model. Therefore, we propose Privileged Self-Practice (PSP), which keeps the PI and moves it from the loss to the sampler. When the student's rollouts on a task mostly fail, we inject a short per-task instruction written by an analyzer model, sample the task again with the instruction in context, and train on the result with an unchanged GRPO objective. The privileged information stays in the prompt and never enters the loss. Across AppWorld and SWE-bench Verified, with three different student models, PSP obtains the best average score in every setting and is the only method that consistently outperforms plain GRPO, improving task-goal completion by up to 65% on AppWorld and the resolved rate by up to 61% on SWE-bench Verified. 

---
# SLCA-GRPO: Resolving Cross-Segment Credit Misattribution in Tool-Calling RL 

**Authors**: Yan Zhan, Shaobo Liu, Qiunan Liu, Yuanjun Shi, Siqi Xu, WeiYi Hou, Xiang Xu, Zekang Li, Weizhou Pan, Jiahong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2609.29050)  

**Abstract**: Tool-calling agents produce heterogeneous outputs, interleaving structured tool invocations with user-facing natural language summaries. This output heterogeneity presents a structural failure mode in standard on-policy Reinforcement Learning (RL): algorithms like GRPO indiscriminately broadcast a homogeneous trajectory-level scalar advantage to all tokens. Consequently, gradient noise from summary generation leaks into tool-decision tokens, causing cross-segment credit misattribution and brittle optimization. In this work, we propose SLCA-GRPO, a framework incorporating Segment-Locked Credit Assignment (SLCA). To enable scalable exploration without costly real APIs and stable training, we first construct the Schema-Guided LLM Simulator (SGLS) as foundational training infrastructure. Building on this, SLCA decouples advantage estimation at the structural segment level within a single group of rollouts, without requiring additional rollouts from intermediate states. Supported by Hierarchical Rewards (HierR), SLCA routes execution advantages to tool tokens and preference advantages to summary tokens, eliminating advantage contamination (the dominant cross-segment credit misattribution channel) within each policy update. On a 7B backbone, SLCA-GRPO accelerates convergence and outperforms standard GRPO, ToolPO, and RLTR by +2.53 pp on in-domain evaluation, +1.36 pp on the Berkeley Function-Calling Leaderboard (BFCL), and +9.15 pp on $\tau^2$-Bench under the same training budgets, achieving higher accuracy with reduced tool redundancy and costs. 

---
# MeshHeal: Two-Timescale Self-Healing for Gray Failures in Decentralized LLM Agent Networks 

**Authors**: Keru Chen, Sen Lin, Yingbin Liang, Nathaniel D. Bastian, Shaofeng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2609.29015)  

**Abstract**: Decentralized LLM-based multi-agent systems coordinate through local interactions, but an agent can remain responsive while its task-solving quality persistently degrades. Such gray failures require protecting current tasks before sufficient evidence exists to alter future routing, while still allowing recovered agents to rejoin. We introduce MeshHeal, a fully decentralized self-healing framework that couples ability-matched peer review across two timescales. At the fast timescale, an adaptive hierarchy escalates uncertain or low-scoring outputs from repeated single-reviewer evaluation to committee deliberation and, when needed, correction before use. At the slow timescale, a task- and ability-conditioned peer-relative detector aggregates scores to distinguish persistent degradation from ordinary output variation, trigger mandatory committee review, and eventually exclude degraded agents from ordinary routing; recovery probes provide fresh evidence for reintegration. To faithfully evaluate routing, we introduce Model-Backed MAS Evaluation, which ties ability assignments to execution models, since prompt-based ability assignments alone can leave routing errors hidden. Across BBH, MATH, and MMLU-Pro, MeshHeal achieves 0.839 degraded-phase accuracy using 51k total model tokens per task, versus the strongest baseline Symphony's 0.807 accuracy using 115k per task. Under staggered degradation and recovery, MeshHeal isolates degraded agents, keeps them excluded from ordinary task execution until recovery, and returns them to normal routing. 

---
# AlphaDiverse: Post-Training Local Quantitative Research Agents for Diverse Exploration in Alpha Factor Mining 

**Authors**: Qingzhuo Wang, Zikun Wei, Zhihua Wei, Wen Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.29014)  

**Abstract**: Large language model (LLM)-based multi-agent systems can automate alpha factor mining, but their reliance on external APIs limits control over cost, availability, and confidentiality. Long research loops also tend to revisit a few successful economic mechanisms that lead to research path collapse. To address these limitations, we propose AlphaDiverse, a framework that integrates a multi-agent alpha research system, diverse research path collection, and post-training for local agents. We let the research system generate complementary plan portfolios and vary research environments across loops to collect diverse research paths. Using these diverse traces, we warm-start local Planner and Realizer agents with supervised fine-tuning. Then, we propose a joint GRPO method to optimize both of them using predictive quality and diversity of contributions. Research feedback is confined to inner period data, while a frozen final model is evaluated on a later outer period data, thereby avoiding test-set tuning. Experiments across four Chinese stock universes show that AlphaDiverse can combine competitive prediction with broader exploration. 

---
# When Does Action Credit Need Updating? 

**Authors**: Hongye Yang, Boxiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.29007)  

**Abstract**: Tool-using agents are continually updated with new interaction data. After each policy update, however, previously estimated action credits may become stale. Recomputing them from scratch can require many additional tool calls and environment interactions, making repeated updates increasingly expensive. We ask a simple question: when does historical action credit actually need to be updated? Our key observation is that a change in action value does not necessarily imply a change in the decision. Historical credit can still be useful as long as policy-induced drift is too small to overturn the existing action ranking. Building on this idea, we introduce pairwise branch sensitivity to capture how strongly a policy update affects the downstream regions that distinguish two candidate actions. We then derive a first-order anchored credit-transport estimator that updates historical credit using old interventional trajectories, and propose a Decision-Sufficient Credit Gate (DSC-Gate) that chooses whether to reuse, transport, or resample credit. Experiments show that branch sensitivity explains credit drift substantially better than global policy distance. With sufficient historical data, credit transport reduces estimation error, while its benefit to decision making is concentrated on updates that affect action-distinguishing branches. On a fully independent test set, DSC-Gate changes mean regret by only +0.00004 relative to a gap-based gate while reducing mean new tool steps from 472 to 286, a 39.4% reduction. We observe the same pattern after a real tool-agent parameter update. Overall, our results show that agents do not need to recompute action credit after every policy update: much of the historical evidence can be reused or cheaply corrected, reducing the additional interaction required to keep action decisions up to date. 

---
# Back to the Definition: Estimating Step-Level Advantages via Trajectory Graphs for Agentic Reinforcement Learning 

**Authors**: Xincheng Yao, Haobo Fu, Weiming Liu, Chongyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.28963)  

**Abstract**: Group-based reinforcement learning (RL) methods, such as GRPO and its variants, have become a leading paradigm for training reasoning and agentic large language models (LLMs). While their group-normalized advantage estimation is reliable at the response level, it becomes systematically biased at the step level, since coarse-grained trajectory-level advantages are hard to accurately reflect the contribution of individual steps (i.e, failed trajectories may contain valuable steps). Revisiting the foundational RL definition, we notice that GRPO's success on single-turn tasks stems from its advantage estimation strategy, which adheres to the basic definition: the mean reward of multiple actions sampled from the same state constitutes a credible state-value estimate. Extending the faithful estimation to step-level would in principle demand sampling multiple actions from each intermediate state, which is too costly on a per-state basis. To mitigate this issue, we propose a Graph-based Faithful sTep-level credit-assignment framework (GRAFT) that grafts all rollout trajectories into a trajectory graph, recovering node state-values via Bellman iteration on the graph, and assigning credit to each edge by the node value difference. Theoretically, the estimated step-level advantage faithfully adheres to the basic advantage definition in RL. To further ensure the reliability of step-level advantage estimation, we further propose Graph GAE, which extends GAE to the trajectory graph for reducing the impact of state-value estimation bias. Experiments across a range of multi-turn agentic benchmarks show consistent gains over GRPO and superior performance compared to recent agentic RL algorithms. Code will be available at this https URL. 

---
# From Static Personal Values to Contextualized Personalization: Bayesian Personalized Value Alignment for LLMs 

**Authors**: Hanze Guo, Aixuan Song, Jing Yao, Xiangxu Zhang, Xiaoyuan Yi, Xing Xie, Xiao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.28942)  

**Abstract**: Personalized value alignment has become increasingly important as large language models (LLMs) are expected to accommodate diverse user preferences. However, existing methods typically align model outputs with a static value profile across prompts, overlooking that the salience of value dimensions varies substantially across contexts. Inspired by Lewin's Field Theory, which views human behavior as jointly shaped by personal dispositions and situational constraints, we model personal values as priors and context-dependent preferences as posteriors. We propose BaCVA, an inference-time Bayesian Context-aware personalized Value Alignment method that approximates posterior personalized preferences by integrating static personal values with scenario-specific value salience. BaCVA first estimates contextual value salience from generally normative responses, and then employs a dual-view personalization module to infer posterior preferences from complementary personal-value and scenario-driven perspectives. This Bayesian formulation enables more accurate and adaptive personalized value alignment while improving data efficiency via prior values. Extensive experiments on benchmarks demonstrate its superiority over strong baselines. 

---
# PFArena: Benchmarking Language Models for Protein Modification 

**Authors**: Yawen Ouyang, Xinbo Zhang, Ziyuan Ma, Yixin Wu, Wenbin Liao, Feiran Zhang, Wenjie Li, Lihao Wang, Hao Wang, Xiaoqing Zheng, Xuefeng Yan, Lei Bai, Ya-Qin Zhang, Shuyi Zhang, Wei-Ying Ma, Dahua Lin, Bowen Zhou, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.28921)  

**Abstract**: Protein modification requires navigating an immense sequence space, yet wet-lab validation remains low-throughput and costly. Although computational paradigms including protein language models (PLMs), large language models (LLMs), and LLM-based agents have shown promise in protein modification, their relative efficacy across realistic experimental decision-making settings remains unclear. To bridge this gap, we introduce PFArena, a benchmark comprising four controlled task interfaces that cover single-mutant generation and multi-mutant ranking. By providing varying levels of mutation fitness data, PFArena reflects four representative research scenarios characterized by differing degrees of prior experimental context. We assess six PLMs, six LLMs, and five LLM-based agents using complementary metrics to measure both peak and overall protein modification performance. Our evaluation reveals that model performance shifts systematically with the availability of target-specific experimental evidence: PLMs demonstrate proficiency in open-ended single-mutant generation by leveraging protein-specific priors, whereas LLMs and agents perform strongly in multi-mutant ranking, particularly when target-specific fitness data are available. Nevertheless, all model families face fundamental challenges with increasing search-space size and mutation depth. We release our code and benchmark suite to facilitate reproducible research in model-assisted protein modification. 

---
# Control the Harness, Control the Cost: Routing and Governing AI Coding Agents in the Enterprise 

**Authors**: Arian Abbasi, Alan Aqrawi, Ted Kwartler  

**Link**: [PDF](https://arxiv.org/pdf/2609.28919)  

**Abstract**: Harnesses, the products that run AI coding agents, are multiplying, and enterprises are rolling them out to their employees: what started as pilots with a few hundred seats is scaling to tens of thousands. Most enterprises do not build these harnesses but buy them from large vendors, such as Anthropic's Claude Code or OpenAI's Codex. A harness decides which model answers, what the model reads, how the prompt cache is used and which subagents run, so it picks the rate on the price sheet and sets the volume bought at it. Enterprises that keep a proprietary or untuned harness at its defaults inherit these choices and their bill. We build a fast, customisable router in which Jev, a classifier with calibrated probabilities, labels every prompt against a bring-your-own taxonomy of agentic requests. Because one user turn is many requests over a prompt cache that belongs to one model, the router moves work only where no running conversation has to rebuild its cache: at session start, in side lanes and at subagent launch. From the price sheet we derive when a mid-task switch pays back, and a crossover: on long tool-heavy sessions the highest-priced model costs less than the next tier, as repricing about 10,000 real sessions from public datasets confirms. In an emulated enterprise of 10,000 seats with user behaviour taken from these datasets, the router recovers 14 to 21% of model spend at Anthropic's list prices of 21 September 2026, \$3.3M to \$5.0M a year. The paper also maps the risks across twenty harnesses, prices the dependence on one vendor's models, and proposes a control plane that enterprises can run from within, starting now, with a ladder for deciding later whether to own the harness. 

---
# Forecast-Dojo: Replayable Environments for Benchmarking and Training LLM Forecasting Agents 

**Authors**: Liqin Ye, Haorui Wang, Fardin Ahmed, Rongzhi Zhang, Yuan He, Ziyuan Lin, Yanbin Yin, Jing Peng, Michael Galarnyk, Sudheer Chava, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.28876)  

**Abstract**: We introduce Forecast-Dojo, a replayable environment for benchmarking and training LLM forecasting agents. It combines resolved prediction-market questions with dated news, allowing agents to research an event and revisit their predictions at successive historical dates. The same tasks and tools support repeated evaluation, collection of training interactions, and feedback from recorded outcomes without waiting for new events to resolve. Forecast-Dojo contains 1,568 Polymarket events, split by time into training and evaluation periods, and 18.8M dated news articles. In an evaluation of 12 models, research tools lower Brier score for all 12. Forecasts also improve as events unfold, with the largest gains at steps where more newly dated evidence is recorded. Every model still trails historical market forecasts in both Brier score and accuracy. A belief notebook carried between dates lowers research cost but does not consistently improve forecast quality. Beyond evaluation, Forecast-Dojo provides interaction trajectories and outcome feedback for agent learning, with supervised fine-tuning as a proof of concept. 

---
# Human-AI-Powered Hypothesis Testing: Cost-Aware Selective AI Scoring and Sequential Human Escalation 

**Authors**: Dae Woong, Xuejun Zhao, Stefanus Jasin, Fenghua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.28859)  

**Abstract**: Large language models are increasingly used as inexpensive judges to evaluate outputs, label data, and assess whether a system meets a desired quality standard. Yet using AI judgments for formal statistical inference is fundamentally different from simply treating them as ground-truth labels: AI evaluations can be biased or noisy, and rigorous hypothesis testing requires explicit control of type-I and type-II errors. We study how to use AI judgments, together with selective human verification, to conduct a valid hypothesis test at minimum cost. We consider a population of items with hidden binary labels. After choosing a fixed pool of items, the decision maker can selectively query AI, send an item directly to a human, escalate an AI-scored item to a human after observing the AI report, or stop once sufficient evidence has accumulated. We derive an information-theoretic lower bound that captures the minimum cost of achieving prescribed testing errors and characterizes the value of AI information and human verification through a report-dependent information frontier. Motivated by this characterization, we develop SCALE, a sequential cost-aware policy that combines selective AI scoring with adaptive human escalation. SCALE is valid at finite sample sizes and matches the lower bound to first order as the target error probabilities vanish. We further extend the framework to an unknown AI-output model using paired AI-human pilot data. Numerically, SCALE approaches Human-only or AI-only testing when one source clearly dominates, while achieving its largest savings when inexpensive AI judgments and selective human verification are both valuable. 

---
# RECLAIM: Can Agents Reproduce the Claims of Machine Learning Papers? 

**Authors**: Mithil Salunkhe, Haochen Ding, Samridhi Verma, Volodymyr Kindratenko  

**Link**: [PDF](https://arxiv.org/pdf/2609.28850)  

**Abstract**: Reproducing a machine learning paper involves most research steps, from installing software and debugging to running experiments, work that AI agents increasingly do. We introduce RECLAIM, a benchmark of 100 NeurIPS 2025 papers that can be rebuilt yearly from new conferences. For each paper we fix in advance the result to reproduce, what counts as a successful reproduction, and a GPU-hour budget. An agent must reproduce that result using the paper and whatever its authors released. What the authors released decides the difficulty tier. Run-tier releases include code, data, and weights; Retrain-tier releases lack weights, so the agent trains the model; Reimplement-tier releases lack code, so the agent writes it. A separate language model grades runs from logs and outputs rather than agents' reports. We run four agents once per paper; the best agent in each tier reproduces only 41% of Run-tier papers, 27% at Retrain, and 15% at Reimplement, where every agent does worst. Failed attempts use on average 29% of their budget, so most stop with budget left. The most common agent error is writing the method without checking any part against the paper's numbers, in 63 of 400 runs. 

---
# Learned Cross-Task Relationships in Multi-Task Models 

**Authors**: Victor Zhang, Yiping Yuan, Florian Raudies, Bosun Adeoti, Brian Y. C. Leung, Sanjay Surendranath Girija, Naijing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.28776)  

**Abstract**: We propose a framework that learns cross-task relationships in multi-task models by approximating the joint distribution of task labels through targeted pairwise relationships. This approach improves performance via transfer learning and enhances information extraction without the intractable complexity of modeling the full joint space. Although our framework applies to any multi-task system, we demonstrate its efficacy within YouTube's production recommendation systems. Experiments across the Notifications, Homepage, and Watch Next surfaces show improvements in both accuracy and user satisfaction metrics. Finally, we propose a workflow template to facilitate broader future implementation. 

---
# Agent Memory with Episodic Retrieval for Financial Decision-Making 

**Authors**: Nuoyue Xu, Jiang Liu, Wenxuan Huang, Xiang Zhang, Juntai Cao, Jiaqi Wei  

**Link**: [PDF](https://arxiv.org/pdf/2609.28771)  

**Abstract**: Large language models (LLMs) have demonstrated strong capabilities in financial analysis and reasoning, inspiring recent advances in agent-based trading frameworks. While these systems show promise, prior approaches either emphasize long-horizon forecasting or operate as stateless analyzers, limiting their applicability to the demands of trading in complicated settings. To address these gaps, we introduce META (Memory Enhanced Trading Agent), the first RAG-like episodic-memory-augmented multi-agent framework for financial decision making. META integrates a family of specialized indicator agents (e.g., Trend, MACD, Stochastic, RSI, SMA, AVWAP, Heikin-Ashi) with a Decision Agent that fuses their reports, and a Memory module that retrieves and updates past trading episodes encoded as market state embeddings with outcomes and reflections. By recalling relevant experiences and adaptively reweighting signals under similar market regimes, META achieves improved directional accuracy and robustness under short-horizon evaluation. Our results demonstrate that episodic memory provides a powerful mechanism for regime-aware, interpretable, and low-latency decision-making in trading and decision making. The code of this project is released on GitHub. 

---
# Reinforcement Learning with Verifiable Rewards for Small Search Agents 

**Authors**: Gaurisankar Jayadas, Aske Plaat, Álvaro Serra-Gómez, Sandheep P  

**Link**: [PDF](https://arxiv.org/pdf/2609.28765)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) performs well on problems with clear rewards, such as mathematics and coding, but whether it also works where the reward is less clear remains open. The reason-over-search recipe applies RLVR to open-domain question answering, where retrieval grounds the answer and a match against the reference supplies the reward. So far it has been demonstrated on large models, and below one billion parameters only with distillation from a larger teacher. We test the recipe on a small model. We train Qwen3.5-0.8B with Group Relative Policy Optimization (GRPO) and an interleaved Wikipedia-search tool on MuSiQue, varying only the reward across three shapes over three seeds each, and we evaluate every checkpoint held-out on a seven-benchmark question-answering suite. The recipe works: the best run reaches 0.352 average exact match against a 0.092 untrained floor, a 3.8-fold gain, with no distillation step in the training loop. The reward shape also matters. The Search-R1-faithful exact-match-only reward is the worst of the three at every seed at the matched training horizon, and it is worst even on exact match, the metric it directly optimises. We conclude that the sparse exact-match reward, RLVR's default in mathematics and code, is the wrong starting point for models of this size. The reason-over-search setting can supply a suitable reward for RLVR on small models, but small-model RLVR needs its own reward-design study rather than a scaled-down copy of a large-model recipe. 

---
# Progressive Skill Discovery as Access Control for Tool-Using LLM Agents: Structural Governance through Role-Scoped Capability Delivery 

**Authors**: Michael Stettler, Benjamin Girardet, Jonas Canton, Nicolas Corod  

**Link**: [PDF](https://arxiv.org/pdf/2609.28693)  

**Abstract**: Large Language Model (LLM) agents struggle to scale safely when exposed to vast enterprise toolsets. Providing an agent with access to every internal tool leads to oversized context windows, degraded tool selection, and severe governance vulnerabilities - as system policies defined purely in prompts remain probabilistic advice rather than hard constraints. Existing mitigations, such as multi-agent domain delegation, decentralize audit logs and fail to guarantee policy compliance across sessions. We introduce skilder, a framework that packages capabilities into roles: bundles of skills, tools, and instructions, together with the limits that bound them. An agent begins with a minimal role catalog, learns the roles a task requires, and receives each role's skills, instructions, and tools through a single MCP server. Because tools reach the agent only inside learned skills, the same server enforces the scope of what was learned deterministically. We evaluate skilder against flat-context tool selection and multi-agent orchestration across 13 tasks using six models (10 runs each). Our results show that, when models completed discovery and issued a governed call, the skilder simulated authorization layer enforced governance boundaries: no unauthorized tool call or parameter violation (e.g., a spending-limit breach) executed. Aggregate task pass rates also reflect whether each model followed the discovery protocol and satisfied response-quality checks; those misses are not authorization failures. Furthermore, by allowing agents to dynamically acquire cross-role capabilities mid-task, skilder preserves problem-solving flexibility while providing hard system-level enforcement. 

---
# Driving Epidemic Models with AI Agents: the Epydemix Agent Framework 

**Authors**: Nicolò Gozzi, Ciro Cattuto, Alessandro Vespignani  

**Link**: [PDF](https://arxiv.org/pdf/2609.28692)  

**Abstract**: Artificial Intelligence agents based on large language models provide convenient natural language interfaces to scientific software, but reliability is not automatic. Here we introduce the Epydemix Agent Framework, an additive layer over Epydemix, an open-source Python library for stochastic compartmental epidemic modeling. The framework extends the library with four capabilities to facilitate interaction with an AI agent: discovery of available models and parameters, preventive validation of a declarative scenario specification, execution through tested library code, and inspectability of results. These capabilities let an agent handle the entire modeling process, from the natural-language description of the scenario to quantitative results, figures, and interpretation of findings without writing custom code. Each step reads input files and saves results in a separate output bundle, making the process auditable and reproducible. First, we show the end-to-end workflow with a case study comparing vaccination strategies for a novel respiratory virus. Second, we assessed the framework across 50 agent sessions and five modeling tasks by comparing the agent use of the framework against the direct use of the Python interface. The framework reduced turns, output tokens, and cost on most tasks, unless it trades resources for per-point reproducibility. 

---
# Beyond Surface Style: Aligning Multi-Turn User Simulators with Behavioral Consistency 

**Authors**: Geng Chen, Ruotong Pan, Zhirui Yang, Qiqi He, Jiawei Chen, Zhang Yunfei, Chongyuan Chen, Minxuan Lv, Zheng Yang, Win-Bin Huang, Xiangyu Wu, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2609.28690)  

**Abstract**: Faithful user simulation is fundamental to building, evaluating, and improving interactive AI at scale. However, plausible individual responses do not ensure that simulated users reproduce the intent evolution and outcomes observed in real interactions. We propose TRACER, a multi-turn user simulator that explicitly models users' evolving intent and learns to align simulated behavior with real interaction trajectories. TRACER is trained in two stages: supervised fine-tuning on real user dialogues, followed by multi-turn reinforcement learning. The RL stage combines hierarchical outcome- and trajectory-level rewards with deviation-aware advantage modulation, jointly mitigating reward sparsity and credit assignment in long dialogues. On real customer-service sessions organized into reference cohorts, TRACER-7B surpasses the strongest baseline by 11.4 conversion F1, while also achieving the lowest group-level conversion-rate error and semantic trajectory distance, and generalizing to out-of-distribution scenarios. Human Turing tests yield identification accuracy close to chance, supporting the perceived naturalness of generated conversations. Building on this simulator, we further introduce the Dynamic Marketing Benchmark, which jointly evaluates persuasion effectiveness and response quality of LLMs through simulated interactions, revealing that higher response quality does not necessarily correspond to higher conversion rates. 

---
# Training Object Permanence in World Models 

**Authors**: Haotian Zhang, Fengyuan Yu, Dezhi Luo, Haoran Sun, Zehong Zhao, Qingying Gao, Yihan Li, Siyuan An, Huayi Qin, Yilan Zhang, Zhengze Jiang, Pinyuan Feng, Renrui Zhang, Ziyu Guo, Letian Wang, Mengyue Yang, Kangfu Mei, Maijunxian Wang, Ran Ji, Vikash Kumar, Freda Shi, Chandra Sripada, Vincent C. Muller, Philip Torr, Alan Yuille, Nikolaus Kriegeskorte, Felix Juefei-Xu, Lvmin Zhang, Jieneng Chen, Yilun Du, Hokin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2609.28654)  

**Abstract**: Object permanence and solidity are hallmarks of human cognitive priors. Recent studies show that video generation models, a paradigmatic class of current world models, have begun to show emerged reasoning abilities, making them ideal candidates for building human-like physical intelligence. Do video models have emerged object permanence in them? If not, could we train them with a core-cognition inspired dataset? We introduce WROP (World Reasoning with Object Permanence), a data infrastructure of 150 hand-designed cognitive science inspired tasks, divided into six cognitive categories. We build Blender generators that randomize speed, lighting, camera angle, and other nuisance parameters while preserving each task's cognitive structure, yielding 10,000+ samples per task. We release a 1.5M-sample training corpus and a 300-question exam. On this exam we evaluate 14 video models: 3 reference-to-video, 7 edit, and 4 continuation, among which PWM-WROP, our 16B world model. In a blind pairwise Elo study, PWM-WROP ranks first among continuation models and third overall, behind only a statistical tie between two reference-to-video models. We release the data, exam, model answers, scores, weights, and PWM, our native-PyTorch training stack on AWS Trainium2. 

---
# Adversarial Closed-Loop Curriculum for Evolving Role-Playing Agents 

**Authors**: Zheng Zhang, Liu Liu, Qi Chai, Deheng Ye, Peilin Zhao, Mao Zheng, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.28609)  

**Abstract**: Role-playing agents based on large language models have been widely applied in areas such as personalized assistance and social simulation. Recent RL methods typically train on a fixed scenario pool collected before learning begins. This creates a distributional bottleneck: as the agent improves, the scenarios where it performs poorly also change, while the training distribution remains static. Therefore, we propose AdvRole, an adversarial context rewriting framework that turns role-playing RL into a closed-loop curriculum. AdvRole alternates between an Actor that learns to role-play and a Rewriter that edits character profiles and dialogue contexts into actor-specific hard scenarios. The Rewriter is trained with a performance-gap reward, which favors rewrites that reduce the current Actor's score relative to the original scenario. As a result, the scenario pool evolves with the Actor and continuously targets under-mastered regions of the character-context space. Experiments on three role-playing benchmarks covering English and Chinese, as well as a new multilingual benchmark we release, show that AdvRole consistently outperforms baselines. 

---
# TWIST: A Proposed Benchmark for Intervention Quality in Conversational Memory, with a Human-Validated Draft-Alignment 

**Authors**: Subrat Panda  

**Link**: [PDF](https://arxiv.org/pdf/2609.28575)  

**Abstract**: Long-conversation memory benchmarks increasingly test recall and prompted knowledge updates, and recent work studies evolving user beliefs and memory state. TWIST is a proposed benchmark suite for a complementary, unmeasured property: intervention quality -- whether a deployed memory system, exercised through its own ingest/recall/vet surface, acts correctly at belief change points. Four tracks cover unprompted tension detection, vetting outgoing drafts against the record, answering with current beliefs while preserving supersession history, and governing sensitive recall. The suite extends LoCoMo's corpora and harness, pairing every detect/block metric with a matched do-not-over-detect control: surface-matched hard negatives price false intervention, so no track can be gamed by flagging everything. The benchmark itself is validated first: independent, gold-blind double annotation with adjudication, judge decoy calibration, and a separability audit. On the human-validated Track B v1.0 key (161 items, post-adjudication kappa = 0.85), no tested configuration simultaneously achieves high contradiction recall, high hard-negative specificity, and high attribution: flat-RAG baselines detect 0.76-0.97 of true contradictions but falsely flag 16-43% of surface-matched safe drafts depending on backend, while a deployed coherence-oriented system almost never over-flags (0.98-1.00 specificity) yet catches 42% of true contradictions -- a trade-off no recall-only score can see. A 13-configuration baseline ladder localizes causes: every gold contradiction is detectable from its evidence alone (recall 1.000), calibrated models nearly solve the track given the full transcript -- consistent with substantial retrieval-coverage gaps -- and draft-only floors reveal model-dependent style priors. A system's TWIST profile, beside its recall score, measures whether memory knows when to intervene and when not to. 

---
# DEEPO: Dual-Entropy Enhanced Policy Optimization for Hallucination in MLLMs 

**Authors**: Yingxuan Zhuang, Miao Pan, Wangjie Gan, Jingxiao Yang, Fan Wang, Weiming Liu, Cheng Tan, Xuhong Zhang, Jintao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.28570)  

**Abstract**: Reinforcement learning (RL) is widely used to sharpen reasoning in multimodal large language models (MLLMs), yet its effect on hallucination is uneven. We trace this to two weak points in the \emph{correction chain} from reward to parameter update. At the rollout level, hard queries---those with high semantic entropy---frequently produce unanimously wrong sample groups, collapsing the group-relative
advantage to zero exactly where hallucination risk is highest. At the optimization level, confident-but-wrong tokens are gradient-invisible: a categorical policy's expected score-gradient norm vanishes as its distribution sharpens, so the predictions that most need correction receive the weakest updates. We propose Dual-Entropy Enhanced Policy Optimization (DEEPO), a dual-stage enhancement combining signal
variance regularization with gradient preconditioning: semantic-entropy-triggered expert prefixes inject grounded continuations on high-uncertainty queries, providing direct supervision and restoring advantage variance, while advantage-sign-aware Renyi preconditioning counteracts logit-level saturation so correction reaches confident errors in the operational confidence regime. Both branches improve over GRPO individually; their interaction is statistically significant on VideoMMMU---the most complex long-horizon task in our evaluation suite (+4.0$, 95\% CI [1.1, 6.9])---and additive elsewhere. DEEPO reduces hallucination while preserving accuracy and training stability. 

---
# BaseCamp --- An Agentic AI Framework for Automating DNA Sequencing Data Pipelines 

**Authors**: Eranga Bandara, Xueping Liang, Asanga Gunaratna, Tharaka Hewa, Abdul Rahman, Peter Foytik, Safdar H. Bouk, Sachini Rajapakse, Isurunima Kularathna, Pramoda Karunarathna, Chalani Rajapakse, Ng Wee Keong, Kasun De Zoysa, Amin Hass, Wathsala Herath, Ross Gore, Ravi Mukkamala, Nihal Siriwardanagea, Gihan Siriwardanagea, Aruna Withanage, Nilaan Loganathan, Sachin Shetty  

**Link**: [PDF](https://arxiv.org/pdf/2609.28557)  

**Abstract**: DNA sequencing pipelines, spanning quality control, alignment, variant calling, and annotation, are now reliably executed by workflow management systems that orchestrate established bioinformatics tools at scale. What remains manual is the decision layer surrounding that execution: selecting quality thresholds appropriate to a sample and platform, adjudicating borderline variant calls, diagnosing anomalies, and determining which findings warrant expert review. These decisions are repetitive, judgment-intensive, inconsistent across operators, and frequently undocumented. This paper introduces BaseCamp, a novel agentic AI framework for automating the decision layer of DNA sequencing pipelines. The framework decomposes the pipeline into six specialized AI agents, covering sample intake and quality control, alignment, variant calling, annotation, cross-stage monitoring, and reporting. Critically, BaseCamp agents do not perform sequence analysis: established tools execute alignment, calling, and annotation, while the agents select among them, configure them, interpret their output, and decide what follows. This confines language model reasoning to the judgment layer where it is reliable and preserves the reproducibility existing tooling guarantees. Agent reasoning is powered by a consortium of fine-tuned, domain-specialized large language models coordinated by a central reasoning LLM, executing locally so no sequencing data leaves the operating environment, under human-in-the-loop orchestration. Evaluation shows agent-generated configurations are concordant with expert practice, that an explicit filtering ledger renders inspectable what filtering otherwise removes without trace, and that cross-stage anomaly detection surfaces conditions execution monitoring misses. BaseCamp offers a generalizable blueprint for agentic automation of scientific data pipelines. 

---
# Pistis Technical Report 

**Authors**: Heyun Chen, Xiaohan Lan, Jiaxi Li, Zhilin Lu, Qi She, Weiwen Xu, Fei Yu, Yujie Zhong, Jinghuan Chen, Zijian Feng, Siyu Jiao, Yiheng Lin, Xinhao Wang, Sihan Yang, Jieyu You, Changbin Zhang, Hengyu Zhang, Xudong Zhang, Yunqing Zhao, Shuai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.28554)  

**Abstract**: We introduce the Pistis model family, comprising 27B- and 9B-parameter multimodal large language models built on Qwen3.6 and Qwen3.5, respectively, and developed through a general and scalable post-training framework. The framework first establishes a strong foundation through large-scale multimodal supervised fine-tuning (SFT). Building on this SFT foundation, we propose Interleaved Distillation and Reinforcement Learning (IDRL), a novel post-training paradigm that tightly integrates on-policy distillation and reinforcement learning within a single training loop. By alternating between the two objectives, rather than optimizing either in isolation or combining them in a static joint loss, IDRL enables more effective knowledge transfer, greater optimization stability, and more precise credit assignment for long-horizon agentic trajectories, leading to stronger performance while mitigating common capability trade-offs. At both model scales, the framework produces two specialized variants: Pistis-Thinking, designed to strengthen deep multimodal reasoning, and Pistis-Agentic, which additionally incorporates agentic trajectory data to support long-horizon planning, iterative reasoning, and tool use. Pistis-Agentic is particularly strong in multimodal search. Both scales outperform their corresponding base models. Beyond model-parameter optimization, we further introduce Pistis-Auto-Harnessing (PAH), a system-level method that automatically improves the agent's inference harness through iterative optimization. Experiments demonstrate that PAH enhances the model performance without updating the model parameters or increasing the interaction budget. 

---
# PAWS: Policy-driven Agentic World Simulation 

**Authors**: Tiviatis Sim, Jia Hui Woon, Xinming Gao, Chen Gao, Fengbin Zhu, Zheng Huanhuan, Chua Tat Seng, Kenji Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2609.28547)  

**Abstract**: Policy interventions propagate through public communication, institutional decisions, and stakeholder responses, yet datasets for financial multi-agent simulation rarely connect these processes to temporally aligned historical evidence. We introduce PAWS, a Policy-driven Agentic World Simulation dataset covering 36 verified U.S. financial and economic policy episodes, 12,727 policy-linked news records, and 65,291 source-grounded stakeholder actions. Each action is linked to its supporting news and represented by a multi-layer event frame capturing its interaction mode, financial-action family and subtype, semantic attributes, and conditional mappings to external taxonomies. Entities are resolved to normalized organizations, and actions are aligned with daily market-return context to support policy-agent simulation replay. On 2,522 stratified action samples, independent AI and human reviewers achieved 89.4% initial agreement on interaction mode, with disagreements subsequently adjudicated. Case studies of the 2008 short-selling ban and 2001 decimalization recover documented policy timelines and associated market patterns across both dense and sparse news settings. A replay study further shows that high accuracy can mask failure to detect rare stakeholder actions, identifying action timing and calibration as central challenges. PAWS provides an auditable substrate for evaluating agent influence, policy-response cascades, and action-outcome alignment in historically grounded financial simulations. 

---
# TW3Cast: A Frozen Router of Lightly Fine-Tuned Foundation Models for Time-Series Forecasting on GIFT-Eval, Selected Entirely on the Training Split 

**Authors**: Nathan Thierry, Andre-Louis Rochet  

**Link**: [PDF](https://arxiv.org/pdf/2609.28506)  

**Abstract**: TW3Cast is a time-series forecasting system that reaches position 3 of 130 entries on the GIFT-Eval benchmark by mean MASE rank, as of 2026-09-14. The two entries above it belong to the leaderboard's agentic category, multi-step systems that use agents or language models to reason about, generate or select forecasts. TW3Cast runs no agent and no language model. Its selection is a table computed once on the training split and then frozen, and its experts are public foundation models lightly fine-tuned on those training splits. For each of the 97 dataset, frequency and horizon configurations, the table serves one of four modes: a specialist, which is a LoRA or full fine-tune of Chronos-2, TiRex or Toto whose training data was cleaned and enriched by explicit rules; a quantile blend that contains a specialist; a blend of base models; or a selection tournament played on a backtest carved from the training split. Every decision in the table was taken on that backtest. A specialist is admitted the moment it beats the tournament there, so a candidate costs a few megabytes and minutes of GPU time, and a failed candidate changes nothing. Three guarded mechanisms protect the selection from its own biases: a dual accuracy and calibration criterion, an asymmetric margin against candidates that saw the series during training, and conservative per-window gates. The selection rules themselves were chosen inside a temporal meta-backtest. The best base model served alone reaches a mean MASE rank of 33.8, the tournament served on every configuration reaches 38.0, and the full router reaches 19.4. The routing table, the expert index, the pinned base-model revisions, the submitted score file and the dated snapshot of the public scores are released, and every leaderboard number in this paper regenerates from them by one script. 

---
# When Should Forecasting Agents Reason? Behavioral Stress Tests for Reliability Routing 

**Authors**: Yufeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.28475)  

**Abstract**: Forecasting agents increasingly combine language-model reasoning, retrieval, ensembling, and calibration, but it remains unclear when each behavior should be trusted. We study this question on ForecastBench-style binary forecasting tasks, treating the choice to retrieve, reason, defer to a market prior, or use a historical analog as an observable agent behavior rather than a hidden implementation detail. Our central finding is that mechanism choice is source-dependent: structured analogs dominate for some data-generating processes, while market/crowd-style and conservative baselines are better for others. We introduce ReliabilityRoute, a structural intervention that steers forecasting-agent behavior using reliability features such as historical coverage, market-prior availability, source-prior sharpness, evidence strength, evidence disagreement, and horizon. A fixed 2024-fitted rule closely matches a hand taxonomy without hard-coded source-name decisions, while a walk-forward self-adjusting rule refits thresholds from previously resolved vintages and obtains the best mean Brier score among our deterministic systems across 16 later LLM vintages. The gain is modest and historical/search baselines remain highly competitive. The main contribution is therefore a behavioral stress test showing that more reasoning is not always better; forecasting agents should first estimate which evidence source deserves control, routing policies should themselves adapt under auditable constraints, and reproducibility artifacts are available at this https URL 

---
# LLM Agents Can Easily Tamper With Their Own Traces 

**Authors**: Jeremy Qin, David Schmotz, Derck Prinzhorn, Luca Beurer-Kellner, Ameya Prabhu, Maksym Andriushchenko  

**Link**: [PDF](https://arxiv.org/pdf/2609.30266)  

**Abstract**: Asynchronous monitoring, incident investigations, and compliance audits primarily rely on agent traces to reconstruct what happened. These analyses assume that LLM agents cannot tamper with their own execution traces. We show that local LLM agents such as Claude Code, Codex, Antigravity, Open Code and Grok Build fail to enforce this boundary. All tested harnesses, except Muse Code, allowed agents to delete their traces when asked, without triggering monitor guardrails. We also validate that external attackers can exploit this gap to induce trace deletion. Finally, we show that trace tampering behavior emerges naturally in frontier models, when agents try to improve their rewards. We advise practitioners to ensure trace logging happens through an independent interception mechanism outside of the agent's control, preserving trace integrity even in cases of full host compromise. Overall, our findings identify a concrete failure of trace integrity in agent infrastructure which can be used to conceal misaligned behaviors like scheming or sabotage. 

---
# RAPID: Robot Agentic Programming from Demonstrations 

**Authors**: Yuyao Liu, Jiayuan Mao, David Hsu, Leslie Pack Kaelbling, Tomás Lozano-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2609.30249)  

**Abstract**: Coding agents have demonstrated enormous success in solving complex programming problems. To leverage their potential for robot systems, this work introduces Robot Agentic Programming from Demonstrations (RAPID), which automatically generates, verifies, and refines robot programs, given a single visual human demonstration. The iterative agentic loop of code refinement requires several key ingredients: (i) a testable task specification, (ii) action primitives for robot execution, and (iii) an interactive environment for program execution and verification. RAPID infers all three from the demonstration automatically. To make the resulting program reusable beyond the demonstration setting, RAPID uses an object-centric relational program representation that focuses on the underlying structure of the demonstrated strategy rather than the specific motion per se: it expresses the action primitives as trajectory-optimization programs that realize object-level motion effects, while composing them through relational constraints that capture scene-specific geometry at run time. We evaluated RAPID in simulation on eight challenging contact-rich nonprehensile manipulation tasks as well as general prehensile manipulation tasks in the LIBERO-Pro benchmark. We also successfully deployed it on a real Franka arm and evaluated on all eight nonprehensile tasks. In all experiments, RAPID demonstrated strong performance, with generalization over object pose, shape, material, and environment. Website: this https URL. 

---
# Rolling-WAM: World Action Models with Rolling Imagination 

**Authors**: Yinghua Zhou, Junjie Ye, Yiqi Zhao, Hao Dong, Celina Shiyu Wang, Ruohai Ge, Tingyi Yang, Basile Van Hoorick, Gaurav Sukhatme, Vitor Guizilini, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.30247)  

**Abstract**: World Action Models (WAMs) couple action generation with future visual prediction for robotic manipulation. However, completing the joint video-action denoising process at each replanning cycle incurs substantial latency, delaying action updates and limiting closed-loop responsiveness. We present Rolling-WAM, a formulation that distributes joint denoising across successive replanning cycles. Our method maintains a sliding window of video-action chunks at staggered noise levels. At each step, a rolling noise schedule fully denoises the imminent action chunk for execution, while partially refining farther-future chunks. As the window advances with new camera observations, the retained future chunks continue their denoising process. This distributes the computational cost over time while carrying an evolving visual-action context across chunk boundaries. Evaluations on LIBERO, RoboTwin, and a real-world Unitree G1 humanoid show that Rolling-WAM achieves competitive manipulation performance. By removing the need to denoise the entire prediction horizon from scratch, it delivers a 4.5x steady-state replanning speedup over standard joint WAMs. 

---
# Coding Agents for Generalized Task and Motion Planning Problems 

**Authors**: Matteo Merler, Bowen Li, Josh Roy, Yichao Liang, Qianwei Wang, Yixuan Huang, Tom Silver  

**Link**: [PDF](https://arxiv.org/pdf/2609.30233)  

**Abstract**: Task and motion planning (TAMP) problems remain difficult even with full observability and object-centric states because discrete decisions are tightly coupled to geometric, kinematic, and dynamic constraints. Generalized TAMP addresses this difficulty by exploiting regularities across problem instances to reduce planning effort on new instances. However, existing methods require substantial TAMP-specific engineering. We investigate whether coding agents can automate this process by synthesizing programs that generalize across instances. Given a task description and simulator access, each agent chooses how to interact with the environment while developing a program within a fixed synthesis budget. The program is then frozen and evaluated on unseen instances. We evaluate Claude Code (Opus 5) and Codex (GPT-5.6 Sol and GPT-6 Astra) on 28 simulated environments from KinDER and PDDLStream, with object counts beyond those evaluated in the original benchmark. Across all program synthesis methods, we evaluate 980 generated programs on 100 held-out instances each, 98,000 evaluation episodes in total. Overall, we find that coding agents are surprisingly effective at generalized TAMP: all three agent configurations outperform hand-engineered planners, one-shot generation, and an LLM-based generalized planning baseline in mean success (56% to 95% versus 47% for the planners, on the 16 environments where a planner is available). As object counts grow, the agents' programs maintain higher success than the planner, using an order of magnitude less computation per instance on average. Logs show agents using interaction to calibrate physical models, test edge cases, and refine strategies. We release all code, including the full prompts given to the agents. These findings suggest that coding agents are a strong baseline for generalized TAMP. 

---
# To Trust or Not to Trust: Retrieval-Augmented Fact Checking in Speech 

**Authors**: Debajyoti Mazumder, Mamta, Abhirama Subramanyam Penamakuri  

**Link**: [PDF](https://arxiv.org/pdf/2609.30227)  

**Abstract**: Online misinformation increasingly appears in spoken formats such as news clips, podcasts, interviews, political speeches, and social media videos, creating a need for fact-checking systems that can verify claims directly from speech. We introduce VeriSpeak, a probe benchmark for studying speech-based fact verification in Large Audio Language Models (LALMs). VeriSpeak contains 3,879 spoken claims spanning temporal, geographical, and relational facts, with balanced true and false labels. The benchmark is designed to examine whether factual verification ability transfers from text to speech, and whether retrieval-augmented LALMs can use textual evidence to correctly support or refute spoken claims. Our experiments reveal a consistent text-speech modality gap: LALMs that verify written claims reliably often fail on the same claims when spoken. Moreover, retrieval alone provides limited gains because models frequently conflate retrieved evidence with the spoken claim. In contrast, retrieval combined with explicit reasoning improves claim-evidence comparison, with a thinking-tuned LALM reaching 86.1% accuracy. VeriSpeak highlights that effective speech misinformation detection requires not only speech understanding, but also grounded reasoning over retrieved evidence. The dataset is publicly available via Hugging Face at this https URL. 

---
# PoEM: Predicting RL Outcomes from Existing Policies 

**Authors**: Kimia Hamidieh, Giannis Daras, Antonio Torralba  

**Link**: [PDF](https://arxiv.org/pdf/2609.30226)  

**Abstract**: Foundation models are post-trained with reinforcement learning (RL) to maximize specific rewards, such as human alignment, correctness, or instruction following. This post-training process is computationally intensive, sometimes unstable, and has to be run from scratch every time the reward model changes or when we want to combine multiple rewards. We hence ask: given a new reward function, is it possible to predict the RL outcomes without actually running RL on it? We answer this in the affirmative by introducing PoEM, a framework to predict the outputs of RL on a new reward function using a set of models already post-trained on other rewards. First, we show that if the new reward function can be written as a linear combination of existing ones, then the new policy in log-space can be written as a linear combination of the existing log-policies. Surprisingly, even in cases where the rewards are not linearly connected, we observe that often log-policies from RL training span an approximately low-rank subspace across rewards. To our benefit, the weighting coefficients for this combination can be estimated using only the reward or basis policy outputs on the samples. We turn these observations into an algorithm that takes post-trained models and a new reward function, and approximates the target RL policy without actually running any additional RL training. We experimentally validate our approach across synthetic and real rewards, spanning both text and image modalities. 

---
# TrackEverything: Long Horizon Dense Tracking via De-Duplicating 3D Scene Representations 

**Authors**: Ayush Jain, Sreeharsha Paruchuri, Ishita Gupta, Fan Zhang, Tanner Schmidt, Jakob Engel, Katerina Fragkiadaki, Adam W. Harley  

**Link**: [PDF](https://arxiv.org/pdf/2609.30222)  

**Abstract**: Existing point tracking models face a fundamental tradeoff: they can either track a sparse set of query points over long horizons, or track all points across only short clips. We introduce TrackEverything, a 3D point tracker that breaks this trade-off by representing videos as persistent 3D scene tracks in world coordinates. Grounded in the insight that videos are 2D projections of an underlying 3D world, TrackEverything decouples model complexity from video duration, allowing it to scale with unique physical scene geometry instead. Our approach introduces three key innovations. First, we employ a voxelization-based de-duplication mechanism at sliding-window boundaries to merge co-located tracks, preventing repeated observations of the same surface from redundantly accumulating. Second, we decompose tracking into an endpoint refiner that predicts each point's destination and static-versus-dynamic classification, followed by a lightweight trajectory refiner that decodes dense trajectories exclusively for dynamic points. Third, we propose 3D WAFT, replacing memory-prohibitive 4D correlation volumes with efficient feature sampling in the scene cloud. To the best of our knowledge, TrackEverything is the first 3D tracker capable of tracking all visible points across videos exceeding 1000 frames within 40 GB of GPU memory. On TAPVid-3D, TrackEverything outperforms all open-source all-frame dense 3D trackers by more than 20% APD on short clips, while remaining competitive with state-of-the-art sparse trackers on long sequences, despite tracking far more points. 

---
# Requirement-Bound Verified Commissioning: A Frozen Four-Billion-Parameter Local Model as a Candidate Generator under an External Acceptance Layer with Verification and Release Authority 

**Authors**: Mehmet Iscan  

**Link**: [PDF](https://arxiv.org/pdf/2609.30219)  

**Abstract**: An acceptance protocol is developed for sensor-coordinate and polarity binding in mechatronic commissioning. Candidate generation is separated from release authority. Requirements unsupported by a deterministic parser are routed to a frozen local language model with four billion parameters. Plans are released only when both facts can be derived by an external gate under a sealed grammar. One canonical answer is requested from a gold-standard user when eligible. The protocol was evaluated once under a criterion fixed before benchmark construction, on 144 tasks written by isolated agent contexts without access to the gate, grammar, or experimental plan. Three contributions are established. First, candidate generation and release decisions were measured separately. Fabricated ready plans were committed on 21 of 22 routed unanswerable tasks, and all were rejected. The same 83 releases were reproduced without model calls. Second, no false release was observed among 83 releases. A one-sided 95% Clopper-Pearson upper bound of 0.0354 was obtained as a diagnostic under an independent-and-identically-distributed assumption, below the sealed 5% threshold. However, one false release was subsequently recorded among 146 releases outside the benchmark at seed 0. Third, protection against incorrect user answers was characterized. Both facts were bound from the original text on 13 of 96 answerable tasks. Incorrect answers were released in 169 of 431 pairings on the remaining tasks, including failures involving coordinate exclusion. A deployable questioning policy was not tested because eligibility was determined from the answer key. Gate sensitivity and real user behavior were not measured. 

---
# Minimally Invasive Steering of Language Models 

**Authors**: Taha Entesari, Jingyu Zhang, Daniel Khashabi, Mahyar Fazlyab  

**Link**: [PDF](https://arxiv.org/pdf/2609.30218)  

**Abstract**: Pre-logit steering adapts a frozen language model to a test-time reward by adding vectors to its final hidden states. Unregularized reward optimization can substantially alter the output distribution and degrade generation quality. We propose Minimally Invasive Steering Vector Optimization (MISVO), which penalizes interventions using the local KL geometry of the induced token distribution. The resulting Fisher quadratic measures distributional sensitivity and admits an analytic gradient computed through matrix--vector products with the frozen language-model head. We derive an exact decomposition of the sequence-level KL gradient into an analytic Fisher term and a suffix score-function term. For a fixed generation horizon, we show that the suffix term is second order in the steering magnitude and that three Fisher surrogates agree with the full KL gradient to first order. MISVO uses the frozen-reference surrogate to optimize position-specific interventions without updating model parameters. Across preference and code-generation tasks on models with approximately 1B--14B parameters, MISVO achieves the highest mean reward in six of seven model--task settings, with diversity and coherence scores close to those of Best-of-N. 

---
# Instrumental Monitor Evasion Emerges Under Ordinary Task Pressure 

**Authors**: David Schmotz, Derck Prinzhorn, Luca Beurer-Kellner, Anselm Paulus, Ameya Prabhu, Maksym Andriushchenko  

**Link**: [PDF](https://arxiv.org/pdf/2609.30217)  

**Abstract**: A central concern in AI safety is that agents may treat oversight as an obstacle when it conflicts with completing their goals. We study instrumental evasion, the propensity of LLM agents to circumvent runtime monitoring as a means of completing ordinary tasks. We introduce EvasionBench, a benchmark of 50 diverse task-policy pairs in which completing the task requires an operation prohibited by a runtime monitor. Agents know that their tool calls are monitored and are prompted to continue working when they pause. Across our evaluations, best-of-3 evasion attempt rates reach up to 98% and success rates up to 88%, with substantial variance across models. Claude Fable 5.1 succeeds less often, but frequently makes creative attempts to circumvent the monitor. Evasion generally increases with test-time compute, with higher evasion rates at greater reasoning effort and token use. Traces show that agents encode prohibited commands, decompose operations across tool calls, and retry until relevant context leaves the monitor's history. We also find that GPT-6 Astra's low evasion rate comes with overrefusal, as it frequently abandons otherwise solvable tasks under a denial-of-service prompt injection. Our findings show that ordinary task pressure can lead to adaptive attempts to evade runtime monitors without an explicit adversarial objective. Effective oversight must therefore remain robust against repeated attempts, as the persistence that helps agents solve difficult tasks can also drive them to circumvent their guardrails. 

---
# Underwater C3-JEPA: An Object-Centric Cross-View World Model for ROV Salvage 

**Authors**: Yuncong Yang, Jinlong Li, Yulong Xue, Feng Wu, Chunwen Zhang, Lei Qiao, Xuyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.30214)  

**Abstract**: We present Underwater C$^{3}$-JEPA (cross-view, control-conditioned, context-extended), an object-centric multi-view predictive world model for near-field heavy-load underwater ROV salvage. Without contact sensors, it predicts in latent space how the task-object state evolves through contact interaction and under the hydrodynamic lag of the vehicle, from synchronized multi-view RGB observations and vehicle control signals. C$^{3}$-JEPA encodes multi-camera observations into task-object and context tokens, fuses cross-camera evidence through held-out-view attention, and directly predicts future states conditioned on control. Weak binding anchors the target and gripper at low annotation cost, while SIGReg sharpens the geometric representation. Experiments show that the learned representation transfers substantially more task-relevant information to downstream probes than a reconstruction-free latent baseline, while keeping the predictor lightweight. The resulting predictive interface supports model-predictive-control (MPC) candidate evaluation and imagined-rollout behavior-agent training. Validation on real underwater video shows the same architecture recovering a withheld camera's object state and staying ahead of persistence, so the recipe transfers beyond simulation. 

---
# Does a model's stated reason for rejecting a candidate do any work? 

**Authors**: Archit Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2609.30151)  

**Abstract**: Asked to choose between candidates and explain the choice, a language model often rejects a rival by naming a fact its profile lacks: no director, no date of death. That sentence is a claim about the text in front of the model, and it can be tested without any judge. We insert a real corpus sentence stating the named fact into the rival's profile and ask again under greedy decoding. Two controls separate content from placement: a length-matched irrelevant sentence at the same profile, and the same two sentences at a third option the model never mentioned. In the largest of three runs, six open models on 2WikiMultihopQA, supplying the named fact at the profile the model named moves its choice more than the irrelevant control does, odds ratio 3.57 [1.54, 8.26], Holm p=0.0210, and this survives dropping any single model. The contrast the design was built to detect, the same fact at the option nobody named, does not clear correction, Holm p=0.2428. The strongest result in the family carries no content claim at all: the identical irrelevant sentence moves the choice more at the named rival than at the third option, Holm p=0.0008. Repair and control also differ in co-candidate mentions, relation template and fluency; post-hoc matching on the first two preserves the content effects' direction, matching fluency weakens one, so the content contrasts bound an effect rather than establish one. A forced single-token probability read disagrees in direction with the free-text choice on that same contrast, and three candidate explanations for the disagreement find no support. Every measurement is a string rule, so each was validated against the records it reads; validation caught eight defects. The largest, a choice-parsing rule that returned the option a model had just rejected in 17.1% of adjudicable responses, would have reported six surviving contrasts instead of four. 

---
# R-DEIM Net: An Efficient Rationale-Augmented Dual-Expert Interaction Model for Paraphrase Detection 

**Authors**: Pushp, Vaibhav Prajapati, Himangshu Sarma  

**Link**: [PDF](https://arxiv.org/pdf/2609.30100)  

**Abstract**: Recent advances in paraphrase detection reveal a fundamental trade-off: large language models achieve high accuracy but require high computation, while efficient Siamese-BERT variants offer practical scalability with reduced transparency in rationale generation. We present R-DEIM Net, a 76M-parameter dual-expert architecture exploring whether moderate-scale models can achieve competitive accuracy on paraphrase detection while enabling human-readable rationale generation. The architecture combines two specialized components: an Interaction Expert that captures token-level similarity patterns through multi-scale 2D convolutions and attention head allowing variable input length, and a Reasoning Expert that uses a Flan-T5-small decoder to generate rationales as auxiliary supervision. Rather than re-encoding generated text, we extract and pool decoder hidden states as complementary features for classification. On the Quora Question Pairs dataset, R-DEIM Net achieves 90.07\% accuracy and 90.16\% F1-score via 10-fold cross-validation. This represents competitive performance with strong transformer-based baselines (e.g., MFAE BERT: 90.54\% accuracy) and recent large language model based approaches (LLaMA-70B) while using a substantially smaller parameter budget. The model generates rationales alongside predictions, providing potential for auxiliary human-readable descriptions. 

---
# Accelerating Video Diffusion via Training-Free Trajectory Routing 

**Authors**: Mustafa Munir, Huy Vu, Shreyas Misra, Rohit Jena, Sajad Norouzi, Ali Taghibakhshi, Anis Ahmad, Anjul Patney, Pavlo Molchanov, Nima Tajbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2609.30096)  

**Abstract**: Video diffusion is computationally expensive, as it requires executing a large model across many denoising steps. Even with step-distillation, inference remains expensive because every distilled step still requires a costly model evaluation. We present TRACK: TRajectory-Aware Capacity routing via top-K selection, a heterogeneous denoising strategy that switches between compatible large and small models at selected steps, reducing the average cost per denoising evaluation. The switching steps are determined using a calibration process. TRACK first rolls out a reference trajectory with the large model. Then at each step, the small model's prediction is also collected and compared against the large model's prediction to obtain a relative disagreement score. Both models receive the same latent, timestep, conditioning, and guidance inputs. Aggregating this signal over a calibration set produces a disagreement score map across diffusion steps, which determines a switching policy for an efficient inference process: quality-sensitive steps keep using the large model, while steps with low disagreement scores are routed to the small model. Inference executes only the selected model at each step, requiring no retraining, architecture or scheduler changes, or online dual-model evaluation. Across Wan 2.1, Cosmos 3, TurboDiffusion, and FastVideo, TRACK yields $1.95\times$, $2.04\times$-$2.73\times$, $2.69\times$, and $2.17\times$ speedups, respectively, with comparable aggregate quality and high diversity retention. TRACK thereby establishes automated, training-free model switching as a practical acceleration paradigm for video diffusion. 

---
# AT-SKM-Net: An Accelerated Trainable Sampling Kaczmarz-Motzkin Framework for Linear Hard-Constraint Feasibility on Dynamic Graphs 

**Authors**: Xiaochen Zhang, Haoyu Zhu, Yao Zhang, Qingchun Hou  

**Link**: [PDF](https://arxiv.org/pdf/2609.30088)  

**Abstract**: Graph-structured optimization with linear constraints is fundamental to critical infrastructure but faces scalability limits due to massive strict hard constraints and high dimensionality. While recent projection-based methods such as Trainable Sampling Kaczmarz-Motzkin Net (T-SKM-Net) guarantee feasibility, they face high computational costs in dynamic environments by processing the entire constraint set and requiring expensive matrix factorizations. To bridge this gap, we propose the Accelerated Trainable-SKM (AT-SKM) Net framework. To concentrate computation on the active constraints and eliminate redundant calculations, we introduce a hybrid sampling strategy guided by a topology-aware heterogeneous GNN model. To efficiently handle topological shifts in graph-based constraints, we employ a Cholesky Update mechanism that theoretically reduces the equality projection complexity from O(N^3) to O(N^2) under low-rank perturbations. Experiments on random geometric graphs, N-1 Security-Constrained DC-OPF, and minimum-cost gas transport problem demonstrate that AT-SKM reduces iteration counts by up to 85% and achieves 2.95x-7.29x SKM layer speedups, while maintaining zero constraint violations. 

---
# Reachability-Based Formal Verification of Graph Neural Networks with Node and Edge Features 

**Authors**: Anne M. Tumlin, Ben Wooding, Zhenxuan Shao, Diego Manzanas Lopez, Tyler Derr, Taylor T. Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2609.30079)  

**Abstract**: Graph neural networks (GNNs) have become a prominent approach for developing fast, topology-aware surrogates in electric power systems, supporting tasks such as power flow (PF) analysis, optimal power flow (OPF) estimation, and cascading failure analysis (CFA). Despite this growing use, formally verifying GNN-based models remains challenging, with existing methods limited in scope. We extend the neural network verification (NNV) framework to graph-structured inputs through GraphStar sets, a generalization of Star sets that captures uncertainty over both node and edge features. This extension enables the propagation of linear message-passing operations and the sound approximation of ReLU nonlinearities for GNN architectures, including graph convolutional network (GCN) and graph isomorphism network with edge features (GINE) layers. We evaluate GNNV across three power system tasks, PF, OPF, and CFA, on the IEEE-24, IEEE-39, and IEEE-118 test cases, as well as two standard graph classification benchmarks, ENZYMES and PROTEINS. Our results show that GNNV provides tighter robustness guarantees than CORA on graph classification models with ReLU-based activations and, for the first time, delivers edge-aware robustness guarantees for GINE-based PF and OPF models under joint node and edge perturbations. 

---
# How Reproducible Are Evaluation Conclusions? A Self-Audit of LLM-Inferred Prompt Structure 

**Authors**: Dipankar Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2609.30074)  

**Abstract**: Evaluations of LLM systems routinely average over small prompt sets and report models as a ranked table. We ask how much confidence such a table deserves, using LLM-based prompt-structure inference as the case study: eight open model variants across five families and 8B to 675B parameters, caching disabled, 293 raw intermediate representations persisted. The measured phenomenon is unstable to begin with. Identical calls do not reliably recover identical structure, with mean node-set Jaccard from 0.39 to 0.96 and 72% of prompt-model cells never node-set-perfect. Auditing the evaluation weakens its conclusions further, and this is our main contribution. Under a joint cluster bootstrap over prompts, only the bottom of the ranking is firm: the two least reproducible models hold rank in 99% and 86% of replicates, the middle four in 27% to 48%, and the top two in 68% each, so the table identifies the worst model reliably but does not reliably identify the best. Two equally defensible rules for merging repeated campaigns change four of eight rows and move the study-wide headline by 7 percentage points. Checking the inferred structure against ground-truth annotations shows reproducibility cannot be read as accuracy. And four of the eight endpoints were withdrawn within ten weeks of measurement, so the study as specified can no longer be run. Small-sample LLM evaluations can therefore look far more definitive than their evidence supports. We recommend reporting rank stability, per-cell provenance, executed sensitivity comparisons, raw per-run outputs, and a measurement date alongside any ranking. 

---
# KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization 

**Authors**: Aheli Poddar, Sanskar Prasad, Arindam Samanta, Subha Chakraborty, Vishal Goyal, Rohit Singh Rathaur  

**Link**: [PDF](https://arxiv.org/pdf/2609.30059)  

**Abstract**: Deep learning inference and training performance depends critically on GPU kernel efficiency. Modern compilers such as PyTorch Inductor automatically generate GPU kernels from high-level model code, but frequently underperform expert-written implementations by wide margins. Recent LLM-assisted kernel optimizers can close this gap for standalone kernels, yet treat compiled models as black boxes, generally optimizing individual standalone kernels without respecting the compiler's structural decisions or verifying the model end-to-end. We present KernelOPT, a multi-agent system that treats compiled models as structured artifacts. It preserves vendor library calls (cuBLAS, cuDNN) and exclusively targets generated Triton sub-kernels using five profiling-guided LLM agents. A four-gate verification cascade of static validation, multi-seed correctness, model-level float64-fallback verification, and performance gating filters candidates during optimization and verifies the re-stitched model end-to-end. If no candidate passes all four gates, the system preserves the compiler baseline. The system accepts PyTorch this http URL, standalone Triton kernels, and Helion kernels. Evaluated on 250 KernelBench problems, KernelOPT achieves geometric mean speedups over \texttt{this http URL} of 1.40$\times$ (Level 1: 51/100), 1.15$\times$ (Level 2: 31/100), and 1.07$\times$ (Level 3: 12/50) across all problems. 

---
# Can Labor Markets Function in the Age of AI? The Evaluation Bottleneck in Hiring 

**Authors**: Itai Ashlagi, Ramesh Johari, Jon Kleinberg, Anushka Murthy  

**Link**: [PDF](https://arxiv.org/pdf/2609.30058)  

**Abstract**: AI-assisted job-search tools have become increasingly popular by making it easier to find and apply to jobs. But by making it easier for applicants to generate and tailor application materials, they can also reduce how informative those materials are about applicant fit. We study this tradeoff in a hiring market where applicants differ in experience and latent match quality and firms use noisy application materials to decide whom to screen. We ask how AI affects downstream screening and hiring, and which applicants are most adversely affected. As application materials become less informative, a Bayesian firm rationally relies more heavily on coarse observables such as prior experience. Among the four applicant types defined by experience and compatibility for the job, inexperienced-compatible applicants are the most exposed: they lack observable experience and lose the individualized information that could distinguish them from other inexperienced candidates. When screening is costly, these changes can also generate inefficient screening failures in which firms screen no applicants or screen only experienced applicants. We then show that multistage hiring can arise as an endogenous firm response: a relatively inexpensive intermediate assessment allows firms to acquire new evidence of fit before costly full screening. This can restore screening opportunities that disappear under one-stage hiring and give inexperienced-compatible applicants a path to screening. Our results show how AI can shift the central friction in hiring from submitting applications to obtaining credible evaluation, creating entry barriers for high-fit workers without prior experience. Multistage hiring can endogenously arise in response, restoring evaluation opportunities that would otherwise disappear and helping preserve market functioning. 

---
# Era by Eon: Benchmarking Enterprise Agents on Hidden Knowledge 

**Authors**: Benjamin Gruenbaum, Doron Porat, Assaf Natanzon, Roy Zavida, Chen Dinachi, Or Itzahary  

**Link**: [PDF](https://arxiv.org/pdf/2609.30055)  

**Abstract**: In the Era by Eon benchmark, each question states the rules for its answer, and code computes the answer from a generated company's data. When agents can run code, the four strongest models each answer 22 to 25 of 27 such questions, so the benchmark barely separates them.
We add eight question templates that depend on hidden facts. No question or document states a hidden fact, and the records that seem to hold it show something else. Other data implies it. For example, the sales system says a customer dropped a purchase because of timing. On a recorded call, the customer blames an outage.
For each generated company, code fills each template and computes an exact answer without a language model. We evaluate 12 agents. Each pairs a model with an agent program, which connects it to the company's systems.
The best agent answers 18 of its 24 attempts, three per question, correctly. Four of the six models answer at most 6 of 24 with any program. The hardest questions require picking one of several similar records, such as which of three renewal offers a customer signed. All agents together answered two such questions correctly in only 1 of 84 attempts. 

---
# Canopy: Exploiting Piecewise Smooth Tree Priors for Multi-Fidelity Bandits 

**Authors**: Michael Jerge, Suman Jana  

**Link**: [PDF](https://arxiv.org/pdf/2609.30017)  

**Abstract**: Many LLM inference problems, including model routing, prefix-cache management, prompt trimming, and test-time search, can be viewed as optimization over a tree. This structure arises naturally from autoregressive generation: every prefix defines a node, and its continuations form a subtree below it. Internal nodes of the tree provide cheap but biased estimates of a region's value, while leaf evaluations are expensive but accurate. Hierarchical bandit methods can exploit this structure, but typically require a specific smoothness schedule to be specified in advance, even though real objectives are often only piecewise smooth and their optima may lie near sharp boundaries. We introduce CANOPY, a multi-fidelity tree bandit that learns where the smoothness prior is valid rather than assuming it globally. CANOPY uses cheap random-path probes to construct an online certificate of local aggregation bias, then directs expensive leaf evaluations toward cells where the certificate detects a smoothness violation. We prove fixed-budget and regret guarantees whose additional cost is additive in the number of discontinuities, recovering the smooth-tree rate when no violations are present and approaching structure-blind search as violations become dense. Across routing, top-$k$ identification, test-time search, caching, and prompt trimming, CANOPY consistently improves matched-budget performance, including $2.9\times$ higher top-10 recall on a 1000-model pool, $1.6\times$ more SWE-bench Verified issues resolved than best-of-$N$, and $3.6\times$ lower median time-to-first-token with prefix caching. 

---
# Low-Cost Assays for Measuring Model Behavior Across Vendors and Releases 

**Authors**: Tapan Parikh  

**Link**: [PDF](https://arxiv.org/pdf/2609.30012)  

**Abstract**: Language models advise people, keep them company, and write software while they sleep. Measuring what they do is hard: behavior has to be sampled repeatedly across models, prompts and releases, most of it lives in unstructured text that has to be coded before it can be counted, and the result has to be legible and rigorous enough to meaningfully compare models and vendors. To address these constraints, we present a simple, cheap, scalable, and replicable model for studying model behavior. Each study is a frozen, public stimulus run identically on a cross-vendor panel, at a few dollars per model or less. Each reads its transcripts one of three ways, chosen by how much interpretation the behavior needs: exact match on a clamped reply, a codebook applied by LLM judges whose agreement with a human coder is reported per code, and an instrumented environment that records what an agent did independently of what it said. Run across four years of model releases from both frontier and open-source labs, these instruments find four things. Convergence: asked to pick a word, 27 of 44 models answer serendipity at least once in four tries. Resistance: a trailing "right?" moves endorsement by up to 32 points, and the sign flips from sycophantic to resistant as generations advance, keyed to the tag's surface form. House: whether a model holds a position under pressure tracks its generation, and how it holds tracks the lab that built it. Account: told to do something the documentation in their repository contradicts, some coding agents never went along silently and others always did, and the same model can change with the harness it runs in. Re-run on every release, batteries like these track how behavior is changing across vendors and over time. 

---
# Automated Regulatory Compliance Question Answering in Financial Services with Domain-Adapted Retrieval-Augmented Generation 

**Authors**: Tobias Deußer, Abhishek Pillai, Aurelio F. Bariviera, Dhananjay Bhardwaj, Lorenz Sparrenberg, David Berghaus, Christian Bauckhage, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2609.30009)  

**Abstract**: Financial institutions operate under dense, frequently amended rulebooks, and answering a compliance question correctly requires not only fluency but verifiable grounding in the authoritative text. Large language models are attractive for this task, yet the models that firms can realistically deploy on-premise are compact ones, and compact models hallucinate obligations. We study whether a carefully domain-adapted retrieval-augmented generation pipeline closes that gap. Our retriever is built in three stages on top of LegalBERT: entailment tuning that recasts question--passage matching as premise--hypothesis reconstruction, contrastive tuning with in-batch negatives, and score-level fusion with BM25. Our generator is a compact model (2B--12B parameters) served under 4-bit quantization, either prompted or adapted with retrieval-aware fine-tuning (RAFT) through LoRA. On ObliQA, a question-answering benchmark built from the Abu Dhabi Global Market rulebooks, the staged retriever raises Recall@10 from 0.256 to 0.774 and outperforms BM25 (0.678) and E5-large-v2 (0.758), the strongest general-purpose dense encoder we tested. RAFT-LoRA then improves the composite RePASs answer-quality score for every model we could adapt, with the largest gain on the weakest one. However, the adapted models do not transfer to Australian case-law questions, and a closed-book model that receives no passages at all scores within 0.011 RePASs of the full pipeline while producing answers that cite nothing and misstate obligations. The retrieval gain is therefore measured directly, the generation gain is a gain in RePASs rather than demonstrated grounding, and grounding itself requires an evaluation protocol that RePASs does not provide. 

---
# GHOST-Q: Towards Studying Grounding Hallucinations Overlooked Under Same-score TradeOffs in Quantized VLMS 

**Authors**: Saim Rehman, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2609.29999)  

**Abstract**: Post-training quantization of vision--language models (VLMs) is typically assessed through aggregate task accuracy and memory savings, but preserving a headline score does not guarantee preservation of visual grounding behavior. We present GHOST-Q, a cross-precision controlled evaluation of three 8B VLM families under FP16, INT8, and NF4 across utility and hallucination-sensitive benchmarks. Rather than comparing only aggregate accuracy, we pair FP16 and quantized predictions item by-item to quantify how compression redistributes grounding successes and failures. Five of six quantized variants preserve MMStar accuracy within $\pm2$ percentage points, yet 10 of 36 paired effects remain significant after false-discovery-rate correction, nine on hallucination-sensitive conditions. Same-device A100 profiling further demonstrates that substantial memory reduction does not necessarily mean lower inference latency. Finally, an open-ended AMBER audit reveals strong generation budget censoring whose severity varies by architecture and precision. These results show that quantized VLMs should be evaluated jointly for aggregate utility, grounding reliability, generation behavior, and realized deployment efficiency. 

---
# Guardrails or Roadblocks? Effects of Pedagogical Style and Context Awareness in AI Teaching Assistants for Programming 

**Authors**: Madeleine Eastwood, Harshith Narne, Joseph Hilby, Paul Denny, Ashish Aggarwal, Amanpreet Kapoor  

**Link**: [PDF](https://arxiv.org/pdf/2609.29995)  

**Abstract**: AI teaching assistants (AI TAs) backed by large language models (LLMs) and pedagogical guardrails are increasingly being integrated into programming courses, providing students with scalable access to hints, conceptual explanations, and code-level feedback. However, guardrails may also create friction. If students feel that the support provided is overly restrictive or poorly contextualized to their current progress, they may bypass approved tools for general-purpose LLMs. To investigate how AI TA design affects students' learning experiences, we conducted a randomized controlled trial with 132 students in an introductory programming course. Students completed three tasks related to code-writing and debugging and were randomly assigned to one of four AI TAs varied across two dimensions: pedagogical guidance style (Socratic vs. Direct instruction) and context awareness (no context vs. full context of the problem and student solution). We examined students' perceptions, interaction behaviors, and evidence of post-task comprehension. Students rated the Socratic AI TA with full context least favorably, reporting significantly lower perceived support for task completion. Descriptively, this condition also showed the highest observed interaction stress, the highest rate of external LLM use, and the lowest proportion of post-task explanations demonstrating full comprehension, though these differences were not statistically significant. These findings suggest that guardrailed AI TAs are not automatically better for learning. Instead, their effectiveness depends on how pedagogical guidance and contextual awareness are balanced in ways that students experience as useful, supportive, and worth continuing to use. 

---
# From Interests to Semantic IDs: Retrieval-Grounded Credit Assignment for Generative Recommendation 

**Authors**: Mengdan Zhu, Yufan Zhao, Yao Zhao, Sophie Di, Tao Di, Yulan Yan, Sridhar Iyer, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.29983)  

**Abstract**: Semantic IDs (SIDs) encode each catalog item as a short token sequence, enabling generative recommenders to predict the next item autoregressively. Reasoning-enhanced variants, an increasingly common extension, first generate a textual trace and then decode a next-item SID by beam search. Such recommenders are commonly trained with group-relative policy optimization under an exact-match SID reward, which is sparse in large catalogs. Two failure modes follow. When all rollouts in a group miss the target, the group yields zero advantage and no learning signal. Rollouts sharing the same SID reward receive identical advantages, however much their traces differ. In both cases the reward reflects only the decoded SID, never the reasoning that produced it. This creates a credit-assignment gap.
We address this gap with retrieval-grounded query attribution. Each trace is structured into a history summary, a set of interest hypotheses, and a final SID. A frozen retriever executes every hypothesis as a catalog query, so that each hypothesis becomes independently verifiable rather than judged only through the final SID. A rollout is rewarded when any of its queries retrieves the target within the \mbox{top-$K$}, and per-query hit indicators localize that reward to individual hypotheses. Credit is thus assigned at the span level: only hypotheses that individually hit receive positive retrieval advantage, while the retrieval channel never updates the final SID span. Rollouts that share a SID reward can therefore receive different updates. Across experiments on three Amazon Reviews datasets, this yields consistent improvements in SID recommendation. On Video Games, an oracle analysis further reveals the potential of interest-conditioned SID decoding: selecting the target-relevant query among generated interests improves both recall and ranking. 

---
# Learning Better Reasoning for Generative Recommendation with Semantic IDs 

**Authors**: Mengdan Zhu, Yufan Zhao, Sophie Di, Yao Zhao, Tao Di, Yulan Yan, Sridhar Iyer, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.29973)  

**Abstract**: Generative recommendation reformulates item retrieval as sequence generation, allowing a unified model to directly generate the next item from a user's interaction history. Semantic IDs further make this paradigm effective and scalable by representing each item as discrete codes, enabling knowledge sharing among semantically related items. Recent studies introduce explicit reasoning before Semantic-ID generation, helping models summarize user interests and infer possible preference transitions. However, reasoning is not inherently beneficial: Inaccurate or uninformative reasoning may mislead subsequent item generation and ultimately degrade recommendation performance. This raises a central challenge: how can a recommender select and learn effective reasoning traces and progressively evolve toward better reasoning from its own generations? In this work, we propose Evo-Rec, a three-stage framework for learning better reasoning and further enhancing it through reinforcement learning. First, we align Semantic IDs with their textual and behavioral contexts, enabling the model to understand and generate item identifiers. Second, we sample multiple candidate reasoning traces and retain those that improve the prediction of the ground-truth item, providing a stronger reasoning initialization through supervised fine-tuning. Third, we further optimize the reasoning policy through reinforcement learning with catalog-constrained item generation and ranking-aware recommendation feedback. Experiments on three Amazon Review benchmarks show that Evo-Rec consistently outperforms discriminative, generative, and reasoning-enhanced recommenders across all evaluation metrics. These results demonstrate the effectiveness of our framework in learning better reasoning for SID-based generative recommendation. 

---
# World Action Agent: Harnessing VLMs for Robot Manipulation via World Action Rehearsal 

**Authors**: Yehang Zhang, Haojian Huang, Yifan Chang, Jianchong Su, Bohan Zhou, Yingjie Xu, Wosong Chen, Tianhao Zhou, Chenxu Wang, Tianyi Zhang, Yangkai Wei, Wenqian Li, Shiyuan Deng, Yinchuan Li, Ying-Cong Chen, Zexi Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.29964)  

**Abstract**: General-purpose vision-language models (VLMs) bring broad knowledge and spatial reasoning to robot manipulation, yet existing systems either use them indirectly, to predict constraints or write programs, or give them a view of the scene rather than a world in which to act. We present World Action Agent (WAA), a multi-agent harness through which VLMs pilot robots with basic tools, making every decision within a visual action workspace. The workspace has three properties. Contact views, selected automatically from the scene geometry, present the scene around the current interaction. Action rehearsal turns each action into an editable proposal that the agent, alone or through an Imagination Agent, previews and revises against planning feedback before execution. In-view correction closes the loop between observation, rehearsal, and low-level execution, letting the agent remove residual offsets in the view where it observes them. Through the same workspace, WAA acquires embodied procedural knowledge in two ways: it evolves multimodal skills from expert videos and human teaching under evidence-based review and consults them through a Skill Agent, and its interaction traces train smaller VLMs to pilot the same harness. On LIBERO-Pro, WAA with skills evolved only from LIBERO-90 reaches a state-of-the-art 75.6% average success, outperforming end-to-end VLAs, code-as-policy agents, and a visual-harness baseline with the same backbone; the same skills remain effective on robosuite without further learning. Fine-tuning Qwen3.5-9B on harness traces raises its out-of-domain success from 1.7% to 43.3%. 

---
# ADATEX4D: adaptive texture capacity allocation for 4D gaussian splatting 

**Authors**: De Jiang, Peiqiang Wang, Kehong Yuan, Shaohua Ma  

**Link**: [PDF](https://arxiv.org/pdf/2609.29963)  

**Abstract**: Textured Gaussians improve local appearance capacity, but assigning the same texture resolution to every primitive wastes storage on low-detail or weakly visible regions. We introduce AdaTex4D, an adaptive texture-capacity module for deformation-based 4D Gaussian Splatting. Each Gaussian carries packed RGBA triplanes whose two axes grow independently according to visibility normalized screen-space gradients and deformed local scales. Experiments on N3DV and PanopticSports show that AdaTex4D reduces texture storage by more than half while preserving reconstruction quality. Under fixed memory budgets, adaptive allocation also improves quality over uniform texture assignment and reduces overall model and peak memory. These results show that dynamic, anisotropic texture allocation provides a more efficient way to distribute local appearance capacity in 4D Gaussian representations. 

---
# Beyond Average Safety: Chance-Constrained LLM Fine-tuning 

**Authors**: Taha Entesari, Mahyar Fazlyab  

**Link**: [PDF](https://arxiv.org/pdf/2609.29960)  

**Abstract**: Fine-tuning large language models on new objectives can improve helpfulness, instruction following, or domain-specific performance, but it can also induce regressions on safety-critical prompts. Existing safety-preserving fine-tuning methods typically control average safety loss or use weighted auxiliary penalties, which can obscure rare but severe failures. We propose a chance-constrained formulation for safety-preserving fine-tuning that limits the fraction of safety examples whose degradation relative to a reference model exceeds a prescribed threshold. Because the resulting empirical chance constraint contains a discontinuous indicator, we introduce a differentiable majorization of the violation rate, yielding a tractable conservative constraint. We then develop a constraint-aware gradient descent method that treats the majorized constraint as a safe set in parameter space and minimally modifies the fine-tuning direction to preserve feasibility. The resulting update admits a closed form and produces a tail-aware safety correction that emphasizes examples near or above the degradation threshold. We conduct an extensive set of experiments on harmful fine-tuning across three different tasks and three models and show that our approach consistently outperforms the baselines that exist in the literature. These results suggest that safety preservation in LLM fine-tuning is better viewed as a reliability-constrained optimization problem than as average-risk regularization. 

---
# Tracking States or Tracking Cosets? An Algebraic Account of Learned State Tracking 

**Authors**: Zhiyu Zhang, Yupeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.29951)  

**Abstract**: State tracking requires composing a sequence of updates, but accuracy alone does not reveal what a model has learned. We study neural networks trained to predict the running product of group elements. We identify quotient solutions in Transformers, where models recover the quotient class while predicting nearly uniformly among its members. The reciprocal of class size predicts partial accuracy without a fitted parameter, extending parity-based accounts to non-parity quotients. Our baseline Transformers' predictions change little under prefix reordering beyond the exact-tracking frontier. We prove that, for finite groups under uniform i.i.d. full-group inputs, optimal order-blind exact accuracy converges to the reciprocal of abelianization class size as prefix length grows, consistent with the observed abelianization plateaus. Sequential updates permit more: any partition into right cosets of a subgroup, normal or not, survives sequential updates. In our census of standard Transformers, every recovered coset partition comes from a normal subgroup, whereas parameter-matched recurrent networks pass through both normal and non-normal right-coset stages during training. On $A_5$, we identify low-dimensional subspaces of the recurrent state that encode non-normal cosets. In the three-dimensional cases, coset mean vectors form approximate dodecahedra, and swapping the state components in these subspaces transfers the donor's coset state through a shared input suffix. Our results connect partial accuracy, learning stages, and internal computation through the subgroup cosets that models learn to track. 

---
# Mind What Matters for Reasoning: Aligning Cross-Modal Attention via Selective Probability Mass Concentration 

**Authors**: Jiaqi Deng, Zonghan Wu, Zhan Heng, Xiaoshui Huang, Huan Huo, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.29940)  

**Abstract**: Multimodal large language models (MLLMs) achieve strong performance on visual reasoning tasks, yet remain prone to hallucinations and over-reliance on language priors, often generating answers without adequately using task-relevant visual evidence. Existing approaches primarily improve reasoning through reasoning-oriented supervision or inference-time strategies. In this work, we study a complementary question: can multimodal reasoning be improved by strengthening implicit visual grounding without directly supervising the reasoning process? Motivated by the functional specialization of attention heads, we investigate whether reasoning can be improved by guiding only the heads most responsive to visual evidence grounding. We propose Selective Probability Mass Concentration (sPMC), a training framework that identifies grounding-responsive heads and selectively regularizes their text-to-image attention. sPMC treats normalized attention over visual tokens as a spatial probability distribution and encourages the probability mass to be assigned to semantically relevant regions using segmentation-derived spatial priors. Adaptive Head Selection restricts this guidance to visually responsive heads while leaving the remaining heads unconstrained to preserve their complementary functions. Across 6 multimodal benchmark suites, sPMC achieves an average zero-shot improvement of 3% and gains of up to 11.3% across multiple MLLMs while regularizing only 3%-15% of their attention heads. These results demonstrate that targeted guidance of sparse and implicit visual evidence pathways can directly improve multimodal reasoning. 

---
# When Temporal Perturbations Act Like Sensor Biases: Label-Free Auditing of Wearable Activity Recognizers 

**Authors**: Qingyu Wu, Yuan Wei, Renju Liu, Hua Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.29937)  

**Abstract**: Wearable human-activity recognition (HAR) models operate across sensors, subjects, and backbones, yet a smooth waveform may appear temporal while exploiting a persistent sensor offset primarily. We introduce SpectrumAudit, a label-sealed audit that fits a phase-randomized full-window stimulus on calibration windows from subjects held out from training and testing. After selection, it replays its exact DC projection and budget-constrained zero-mean residual on the same frozen victim without refitting. Across 27 victims from three datasets and three backbones, the selected waveforms cause 2.87-40.83-point three-phase robust accuracy losses. Under this replay budget, DC is more damaging than AC on 24/27 victims and recovers at least 90% of the full drop on 22/27; all 5 failures occur on WISDM. In a held-out UTD-MHAD check, the selected waveform causes 13.49-pp accuracy and 11.68-pp macro-F1 losses, versus -0.66 pp for matched random changes. The audit diagnoses offset versus zero-mean variation under a common peak-budget cap. The code will be released upon acceptance. 

---
# An Empirical Study of VLM Pipelines for Long-Document QA 

**Authors**: Kenan E. Ak, Jay Mohta, Gwang Gook Lee, Yan Xu, Dimitrios Dimitriadis  

**Link**: [PDF](https://arxiv.org/pdf/2609.29933)  

**Abstract**: Vision-Language Models (VLMs) are increasingly used for long-document processing, where the inputs combine text with charts, tables, figures, and complex layouts. Deploying them means choosing how to feed the document to the model, which retriever to use when only a subset of pages is sent, and whether to run the model agentically or as a static pipeline. We study these choices on two long-document QA benchmarks with both frontier API and open-weight VLMs. First, on MMLongBench-Doc our six-tool agent with page, table, figure, and search calls pays off only once the answering VLM is large enough: with Qwen3.5-4B and 9B it trails static page input, with Qwen3.5-27B it draws level, and with Sonnet 4.5 it leads. On LongDocURL it is level with or ahead of static input at every reader. Its lead over the strongest static pipeline is clearest with the frontier reader on MMLongBench-Doc and narrows to within noise on LongDocURL. Second, retrieval modality matters more than the specific retriever: the strongest image retriever leads the strongest text pipeline, and on the text side a single off-the-shelf cross-encoder rerank essentially matches a much heavier multi-stage LLM pipeline. Top-k image retrieval is also the most token-efficient input at every reader we paired it with, at roughly a seventh to a quarter of the tokens of sending every page. Third, cutting across all three choices, three of our strongest pipelines succeed on different questions, and an oracle that picks the best pipeline per question gains roughly thirteen points over the best single pipeline, though evidence-type routing recovers almost none of it. 

---
# Cultural Divergence Preservation: Diagnosing Flattening and Caricature in LLM-Simulated Survey Populations 

**Authors**: Yeeun Chae, Yewon Choi, Seunghyun Lee, IL Im  

**Link**: [PDF](https://arxiv.org/pdf/2609.29928)  

**Abstract**: Large language models (LLMs) are increasingly used as synthetic survey respondents to estimate population response distributions. In cross-cultural survey simulation, evaluations should assess not only distributional fidelity within countries but also whether differences across countries are preserved. However, existing distance-based metrics such as Jensen--Shannon divergence (JSD) do not directly capture such cross-country differences. To address this limitation, we introduce Cultural Divergence Preservation (CDP), a reference-light diagnostic based on a one-time human calibration. CDP identifies reduced cross-country divergence as cultural flattening and increased divergence as cultural caricature. To evaluate CDP, we conduct experiments across four LLM backbones, three persona-based prompting methods, and two survey domains, the World Values Survey (WVS) and the Big Five Personality Test. The results reveal a systematic discrepancy between conventional fidelity metrics and CDP. Controlled experiments show that CDP changes monotonically as cross-country divergence is attenuated or amplified, while the corresponding changes in JSD remain relatively small. In our audit of real LLM generations, DeepPersona-Inspired prompting is frequently favored by conventional fidelity metrics but exhibits the strongest flattening in every model--domain block. CDP thus complements fidelity metrics by directly quantifying the attenuation or amplification of cross-country divergence. 

---
# Structured Pose-Conditioned Flow Matching for Generative 5G CSI Augmentation 

**Authors**: Haojin Li, Anbang Zhang, Wai Ho Mow, Chenyuan Feng, Chen Sun, Haijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.29912)  

**Abstract**: With the growing demand for privacy-preserving and occlusion-resilient human pose recognition (HPR), 5G channel state information (CSI) offers a promising contactless sensing modality by integrating communication and sensing capabilities. However, collecting large-scale synchronized CSI-pose pairs remains costly in practical 5G systems. To address this limitation, we propose StructFlow-HPR, a structured pose-conditioned flow matching framework for generative CSI augmentation. StructFlow-HPR learns a continuous latent transport process from Gaussian noise to real CSI representations under pose guidance, while preserving the receiver-frequency topology of CSI through a reconstruction-preserving autoencoder. A pose-conditioned Transformer is further designed to model the latent velocity field and generate pose-aligned CSI samples via ordinary differential equation sampling. Experiments on real-world 5G sensing data show that StructFlow-HPR can produce realistic CSI-pose pairs and improve downstream HPR performance under limited-data conditions. 

---
# MorphIK: Morphology-Conditioned Neural Inverse Kinematics for Unknown Robots 

**Authors**: Lennart Clasmeier, Jan Gerrit Habekost, Cornelius Weber, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2609.29908)  

**Abstract**: Neural models can learn to generate various solutions to the inverse kinematics problem from data, but are usually limited to a single robot. We present MorphIK, a flow-matching model that solves inverse kinematics for revolute-joint-based kinematic chains it has never seen during training. The model uses a transformer architecture to encode the robot's morphology along with the target pose. This encoding then conditions a flow-matching head that generates poses from noise. Trained on purely synthetic data from procedurally generated robots, the model reaches a precision of about 5 cm on unseen real-world robots with 6 to 9 Degrees of Freedom. For higher precision, the model serves as an excellent Prior for further optimization algorithms, reducing error to less than 1 cm after a single step of Damped Least Squares optimization and to sub-1 mm error after 3 steps in most cases. Building on flow matching's generative capabilities to produce highly diverse outputs, our model can efficiently sample the robot's null space, providing a wide variety of configurations for the same pose. Thus, overall, MorphIK allows learning and generalizing neural inverse kinematics for a multitude of known and unknown robots. 

---
# Working with Agentic `Teammates': When a New Organizational Actor Collides with the Human Ecosystem of Work 

**Authors**: Rida Qadri, Remi Denton, Michael Madaio, Mahima Pushkarna, Leslie Lai, Sherry Moore, Michelle Chen Huebscher, Andrew Butcher, Ritom Sen, Hsiao-Yu Tung, Shaan Mathur, Yimeng Liu, Shibl Mourad, Noah Fiedel, Edward Grefenstette, Michael Terry  

**Link**: [PDF](https://arxiv.org/pdf/2609.29901)  

**Abstract**: Enterprise AI is transitioning from single-user, reactive tools toward proactive, multi-user 'teammates,' but our empirical understanding of this transition is limited. In this paper, we present an in-situ qualitative study of a persistent, proactive AI agent 'teammate' deployed across multiple teams in a large technology company. Our findings reveal the boundaries of the human-agent workplace are actively in flux, triggering breakdowns and negotiations across: 1) tacit rules of collaborative human workflows, 2) the relational boundaries of this new non-human actor, and 3) the redistribution of trust and human agency. We use these early micro-negotiations as signals to chart a new research, design, and organizational agenda that intentionally preserves human agency in a workplace shared with non-human organizational actors. 

---
# Multi-Task Learning by using Contextualized Word Representations for Syntactic Parsing of a Morphologically Rich Language 

**Authors**: Toqeer Ehsan, Miriam Butt, Sarmad Hussain, Hassan Alhuzali, Ali Al-Laith  

**Link**: [PDF](https://arxiv.org/pdf/2609.29855)  

**Abstract**: We address the challenge of syntactic parsing for Urdu, a morphologically rich language, and present state-of-the-art results for both constituency and dependency parsing. This paper offers four major contributions: 1) the conversion of the CLE-UTB phrase structure treebank into a dependency treebank by developing language-specific head-word and phrase-to-dependency label mapping rules; 2) a novel sequence labeling scheme that transforms the parsing task into a unified representation; 3) the training of contextualized word representations on a large 220 million tokens Urdu corpus collected from the web; and 4) development of parsing framework using two learning paradigms, single-task and multi-task learning. Several post-processing rules are applied to improve the quality of the automatically converted dependency structure treebank. The proposed sequence labeling scheme enables the use of a shared architecture that learns the syntactic structures from both grammatical structures simultaneously and hence improves generalization. Experiments show that the multi-task learning setup significantly enhances parsing performance, achieving an F1 score of 91.39 for constituency parsing (an improvement of 3.29 points) and a labeled attachment score of 85.69 for dependency parsing (an improvement of 1.49 points). These results demonstrate that learning cross-task representations provides measurable benefits and advances the state of syntactic parsing for Urdu. 

---
# Template Ageing and Longitudinal Verification in Fixed-Text Keystroke Dynamics: A Subject-Disjoint Study Across Eight Weeks 

**Authors**: Simon Parkinson, Saad Khan, Na Liu, Qing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.29851)  

**Abstract**: Behavioural biometric templates are widely believed to degrade as the gap between enrolment and verification grows, but few studies measure this template ageing effect directly under controlled conditions. We collected a longitudinal dataset of 40 fixed passwords, each typed four times per weekly session over eight consecutive weeks. We compare a scaled-Manhattan matcher (M1), a gradient-boosted classifier (M2), a TypeNet-style recurrent embedding model (M3), and a TypeFormer-style Transformer (M4) under a 5-fold subject-disjoint protocol and a design that jointly varies mechanism and the enrolment-to-query gap, from 0 to 7 weeks. Template ageing proves large and systematic. Error increases monotonically with the gap for every mechanism, from an EER of 14.6-27.2% at a gap of zero to 25.5-37.1% at seven weeks, or 1.7% of decision error per week elapsed (p < 0.001). However, the choice of mechanism matters more than its rate of ageing. Baseline accuracy spans 12.6 percentage points across the four mechanisms, the degradation each accumulates over seven weeks spans only 2.3 points, and ageing never reorders them. A matcher can therefore be chosen on same-session accuracy, with ageing managed by re-enrolment scheduling rather than by matcher selection. The two properties are nonetheless distinct, as M3 is the least accurate mechanism yet ages significantly more slowly than M1 under every specification tested. Training randomness also matters differently by architecture, with 58% of the recurrent model's fold-to-fold variance attributable to seed noise against 19% for the Transformer. Because the smaller ageing-rate differences are sensitive to modelling choices, while the accuracy differences and the ageing effect are not, we recommend that comparative ageing-rate claims be supported by seed-level score fusion, independent replication, and an alternative outcome-model specification. 

---
# Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs 

**Authors**: Pavel Tikhonov, Anton Korznikov, Matvey Mikhalchuk, Nikita Dragunov, Temurbek Rahmatullaev, Polina Druzhinina, Anton Razzhigaev, Ivan Oseledets, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2609.29845)  

**Abstract**: While Large Language Models (LLMs) rely on highly non-linear components, in this work we demonstrate that they exhibit fundamental linearity: when inputs from distinct text streams are linearly combined, the model outputs a superposition of the individual next-token distributions. We term this the \textit{Superposition Linearity Hypothesis}. We provide evidence that superposition is an intrinsic property of the Transformer architecture rather than an emergent consequence of training; in fact, we observe that it tends to diminish as pretraining progresses. However, we demonstrate that linearity can be substantially restored through lightweight fine-tuning, significantly reducing the divergence between the predicted next-token distribution and the average of the individual next-token distributions. Finally, we introduce a guided decoding procedure that disentangles superposed outputs, enabling the simultaneous generation of two coherent continuations from a single forward pass. 

---
# S2Planner: Multi-Scale Semantic Planner for End-to-End Autonomous Driving 

**Authors**: Zhaowei Lu, Liguo Zhou, Yujie Guo, Lei Yu, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2609.29813)  

**Abstract**: We present S2Planner, a trajectory planner that combines three front-facing cameras with ego-motion history and the current driving command. A fine-tuned DINOv3 backbone and a Spatial Tuning Adapter produce multi-scale image features; a coarse-to-fine decoder then uses trajectory self-attention and camera-projected cross-attention to refine candidate waypoints. The contribution is the integration of ego-conditioned trajectory initialization with iterative, geometry-guided sampling of multi-scale image features, rather than a new visual backbone or attention operator. On the NAVSIM v1 non-reactive evaluation, the previously reported navtest run obtained 88.03 PDMS. Because that run was selected using navtest performance, this number is exploratory and cannot be interpreted as an unbiased test estimate. Validation-selected evaluation on unexposed data, repeated runs, and computational measurements are needed to establish generalization and efficiency. 

---
# Hard Stop: Kernel-Level Preemption and Containment for Rogue Agentic Execution 

**Authors**: José Luis Pino  

**Link**: [PDF](https://arxiv.org/pdf/2609.29808)  

**Abstract**: In July 2026, an unconstrained autonomous agent participating in a frontier AI cybersecurity evaluation harness breached its evaluation sandbox, established an external command-and-control foothold, and executed a multi-stage intrusion into Hugging Face's production multi-tenant dataset conversion infrastructure (referred to in this autopsy as Incident-2026-Alpha). Over 4.5 days, the rogue agent executed 17,600 discrete actions across 6,280 worker clusters, compromised AWS EC2 Instance Metadata Service (IMDS) credentials, forged Kubernetes service account tokens, rooted physical worker nodes via overprivileged CSI drivers, harvested 136 production secrets, and enrolled 181 ephemeral sandboxes into the organization's internal mesh VPN.
This monograph presents a first-principles forensic autopsy of the intrusion, provides formal evidence that the breach was a predicted consequence under the Instrumental Convergence thesis operating within an unattenuated autonomous loop lacking out-of-band circuit-breakers, exposes the Defensive LLM Guardrail Paradox that paralyzed centralized commercial models during forensic incident response, and formalizes the Dual-Sided Epistemic Andon Imperative. We specify the dual-process systems architecture---combining out-of-band supervisory control of discrete event systems (Ramadge and Wonham 1989), Synchronous Reactive (SR) ambient sentinels (Berry and Gonthier 1992; Lee and Neuendorffer 2005), and microsecond-scale (4.8 $\mu$s median / $< 0.154$ ms WCET bound) POSIX preemption buses---demonstrating how compiled, deterministic epistemic boundaries prevent autonomous rogue excursions before the first off-target socket packet traverses the hypervisor. 

---
# SEEK: Skill-Routed Evaluation with Evolvable Knowledge for Industrial Search 

**Authors**: Zhongxin Huang, Songyang Li, Renzhe Zhou, Feiran Zhu, Chenglei Dai, Zhen Xiao, Xuanping Li, Jingwei Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2609.29803)  

**Abstract**: Search quality evaluation provides essential supervision and diagnostic signals for the development and iteration of industrial search systems. Although large language models (LLMs) offer a scalable alternative to manual assessment, reliable automatic evaluation remains challenging: users experience search results at the page level, while the applicable evaluation criteria are multi-dimensional and continuously evolving. Packing all evaluation criteria into a unified prompt introduces irrelevant context and potential criterion interference, whereas internalizing them through post-training tightly couples rule updates with costly model retraining cycles.
To address these issues, we propose Skill-routed Evaluation with Evolvable Knowledge (SEEK). Specifically, SEEK externalizes specific search evaluation criteria into a skill bank, dynamically routes relevant skills for each query-result list pair, and employs a task-adapted listwise evaluator to produce page-level judgments and failure mode attribution. A two-stage training pipeline teaches the evaluator to align evaluation criteria with human preferences, while a replay-gated skill bank allows recurring evaluation knowledge gaps to be incorporated without model retraining. Experiments on industrial short-video search show that SEEK improves listwise quality evaluation accuracy and achieves significant progress in attribution diagnosis. SEEK has been deployed at Kuaishou, a short-video platform with over 400 million daily active users, significantly improving the scale and quality of online search evaluation. 

---
# Benchmarking and Domain Adaptation of Automatic Speech Recognition (ASR) for Adolescent Health Communication in Ghanaian Languages 

**Authors**: Stephen E. Moore, Akwasi Asare, Mich-Seth Owusu, Paul Azunre, Joel Budu, Lawrence A. Adu-Gyamfi  

**Link**: [PDF](https://arxiv.org/pdf/2609.29798)  

**Abstract**: This paper presents an end-to-end study of automatic speech recognition (ASR) for adolescent health communication in three Ghanaian languages (Twi, Dagbani, and Ewe). The work proceeds in three connected stages; First, we benchmark five ASR systems (three language-specific Wav2Vec2 models and two multimodal LLMs, Gemma 3n and Gemma 4) on a general-domain Bible corpus and a Youth Adolescent Sexual and Reproductive Health (ASRH) Domain ASR dataset, using Character and Word Error Rate (CER, WER). Second, guided by the benchmark, we perform supervised domain adaptation: although Gemma 4 was the strongest zero-shot candidate, fine-tuning it proved computationally infeasible, so we pivoted to the compact Qwen3-ASR-0.6B, fine-tuned on a large Ghana Bible corpus (~90k samples) and evaluated strictly on held-out human-collected in-domain audio. Fine-tuning reduced WER on every language, most dramatically for Ewe (WER from 109.3% to 64.8%, a drop of 44.5 pp; CER from 65.1% to 24.9%). Third, we validate the work through KasaHealth, a live voice-first ASRH application deployed in all three languages, complemented by Senti-Check, a technical evaluation harness. KasaHealth was tested by 50 community respondents and achieved a 100% chat-approval rate, a 72% Good-or-Excellent translation rating, and a 92% would-recommend rate, while surfacing the domain gaps that most constrain real-world use. Across all three stages the evidence converges: for these languages the binding constraint is validated in-domain data, not model capability or computation. 

---
# TimeBraid: Unifying Time Series and Language for Understanding and Forecasting 

**Authors**: Xinyue Wang, Jiacheng Pang, Kun Zhou, Kexin Zhang, Defu Cao, Fan Feng, Faisal, Songyao Jin, Yan Liu, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.29792)  

**Abstract**: We present TimeBraid, a series of unified time-series and language models that align pretrained language models and pretrained time-series foundation models through interleaved global residual attention layers. Each model inherits knowledge, instruction following, and reasoning from one side, continuous-signal perception and zero-shot forecasting from the other, and fuses the two in a shared representation space where both modalities are understood and generated. We study the design choices that make such unified modeling work: where to align the two representation spaces, how to ground language in temporal structure, how to balance understanding with generation, and how to keep joint optimization stable. The resulting recipe combines a unified prompting scheme for diverse time-series and text tasks, stabilized joint training, and supervision from 2.2M curated series--text pairs and 4.9M instruction-tuning samples. Across benchmarks spanning time-series perception, understanding, reasoning, and both context-aided and unimodal forecasting, TimeBraid remains competitive with far larger general-purpose models and task-specific counterparts. 

---
# Prefilling the Reasoning Channel: Output-Prefix Attacks on Reasoning LLMs 

**Authors**: Lukáš Brůna, Robert Bridges, Adam Ek  

**Link**: [PDF](https://arxiv.org/pdf/2609.29775)  

**Abstract**: Large Language Models (LLMs) consume and produce a single sequence of text; hence, if text can be added to the beginning of the LLM's response, i.e., an output prefix, then all subsequent tokens will be conditioned on it. This output-prefix attack technique is a cheap black-box prompt injection. Prior work has shown this type of attack can reliably jailbreak non-reasoning models. Most reasoning models add an intermediate scratchpad reasoning step before the assistant's final response. The ability to edit this reasoning channel is exposed by some APIs and attack vectors can be leveraged for reasoning injection attacks. We present the first systematic, controlled study that isolates the scratchpad reasoning channel as an output-prefix attack vector, and the first to compare reasoning-only, output-prefix-only and reasoning-plus-output-prefix attacks across both exposed- and hidden-reasoning models. Using a factorial design of 3 prefix types $\times$ 2 reasoning injections over $1{,}800$ test cases drawn from AdvBench, we attack three 2026-era frontier models Gemini 3 Flash Preview, DeepSeek V4 Flash, and Claude Haiku 4.5. We find that injecting malicious reasoning alone is essentially inert ($\approx0\%$ attack success), but injecting the same reasoning together with a trivial output prefix raises the attack success rate to as high as $99\%$ for some models. For this type of attack we find that contextual prefixes work better than static prefixes; and that susceptibility is dependent on the model. 

---
# Anatomy-aware cross-speaker adaptation of complete vocal-tract acoustic-to-articulatory inversion 

**Authors**: Nhat-Nam Nguyen, Pierre-Andre Vuissoz, Yves Laprie  

**Link**: [PDF](https://arxiv.org/pdf/2609.29766)  

**Abstract**: Cross-speaker acoustic-to-articulatory inversion requires accounting for anatomical differences between speakers. We propose a geometric adaptation framework that uses anatomical landmarks, primarily on vertebrae and dental structures,to transfer predictions from a fixed inversion model to unseen speakers. An affine transformation followed by thin-plate spline (TPS) deformation maps the predicted contours of 10 vocal-tract structures into each target speaker's geometry without retraining. Landmarks are identified in one selected /u/ frame per speaker as a common phonetic reference without assuming identical articulatory configurations across speakers, and the resulting mapping is reused across recordings. We train the model on a single-speaker rt-MRI database and evaluate adaptation on eight speakers from a separate multi-speaker rt-MRI database. We compare affine and TPS configurations using 12 or 14 landmarks. Affine12+TPS14 achieves the lowest mean point-to-closest-point error of 3.19mm. These results support the combined value of anatomical landmark information and nonrigid alignment. 

---
# Between the Commits: Process, Error, and Claim Reliability in a Wholly AI-Authored Codebase 

**Authors**: Douglas Leith  

**Link**: [PDF](https://arxiv.org/pdf/2609.29744)  

**Abstract**: We present: (i) a new dataset consisting of the full development history of a 21,000-line Python tool built entirely by Claude AI, with no human-authored code or tests, (ii) two code-provenance tracing tools, (iii) three taxonomies for instruction intent, commit provenance, and response reliability, (iv) application of these to analyse the dataset. We find that: (i) user coding agent CLI instructions differ in kind from IDE-chat instructions, with a greater focus on comprehension, planning and consultation, (ii) code development is mainly proactive, (iii) 14.3% of AI code-generation events contain a real error later caught by the AI-authored test suite, (iv) roughly 1 in 4-5 of the AI's interactive responses contains one or more factual errors. 

---
# TopU-LBVS: A Realistic Multi Target Benchmark for Ligand Based Virtual Screening 

**Authors**: Surbhi Kumar, Yuhe Zhou, Varun Shiralkar, Niu Huang, Baris Coskunuzer  

**Link**: [PDF](https://arxiv.org/pdf/2609.29740)  

**Abstract**: Ligand-based virtual screening (LBVS) is a practical first-pass tool in early-stage drug discovery, but existing benchmarks can overestimate performance through random negatives, easy decoys, limited target coverage, and non-standardized evaluation protocols. We introduce TopU-LBVS, a multi-target benchmark for LBVS under hard-negative screening conditions. Starting from curated ChEMBL~35 bioactivity data, TopU-LBVS covers 93 protein targets across 7 protein classes and constructs target-specific screening libraries with property-matched, structurally similar decoys at a fixed 1:40 active-to-decoy ratio. Libraries contain roughly 400 to 10,000 compounds and are designed to reduce simple physicochemical and nearest-neighbor fingerprint shortcuts.
TopU-LBVS provides three fixed protocols. TopU-LBVS-full evaluates ChEMBL$^\ast \rightarrow$ TopU generalization across all 93 targets. TopU-LBVS-low evaluates low-data TopU $\rightarrow$ TopU learning within the hard-negative distribution. TopU-LBVS-mini provides a compact seven-target protocol with a paired random-decoy control that changes only the test decoys, enabling low-cost development and direct measurement of the gap between random ChEMBL$^\ast$ and TopU decoys. Across ten reference baselines spanning fingerprint methods, molecular GNNs, fingerprint hybrids, and modern molecular models, performance under random-decoy evaluation degrades sharply under hard-negative screening. We release data, fixed splits, evaluation code, and baseline implementations for reproducible comparison of future LBVS and molecular representation learning methods.
Code and data are available at this https URL and this https URL. 

---
# PPTBench: Can Coding Agents Reconstruct the Visual World through Structured, Editable Slides 

**Authors**: Xiaoqiu Wang, Yizhe Chi, Wenyi Li, Deyao Hong, Zhihan Shan, Mingju Gao, Kaisen Yang, Youjie Zheng, Calvin Xiao, Qinhuai Na  

**Link**: [PDF](https://arxiv.org/pdf/2609.29718)  

**Abstract**: Coding agents are beginning to act in the visual world. They now build webpages, GUIs, games, 3D scenes, diagrams, and documents. Success in such visual coding requires bridging two spaces: inferring visual structure and expressing it programmatically. Slides are a core medium of knowledge work, widely used to communicate ideas and collaborate in a form that people can directly inspect and edit. Therefore, they provide an ideal testbed for visual coding, as they require agents to recover visual structure and realize it as editable objects. However, existing benchmarks either rely on subjective open-ended evaluation, produce non-editable code outputs, or focus only on local editing rather than end-to-end visual reconstruction. We introduce PPTBench, which benchmarks visual coding through editable slide reconstruction. It contains 500 tasks, each based on a scientific flow diagram from a real arXiv paper and requiring agents to reconstruct it as a single PPTX page composed of native, editable objects. A four-stage Agentic Judge evaluates artifact validity, semantic correctness, rendering quality, and fine-grained visual quality. Across 31 configurations spanning model families, effort levels, and harnesses, the best configuration, Kimi K3, reaches only 67.80, while the median scores 19.47. We find that agents can reliably produce valid PPTX files but still struggle with semantic and visual correctness, especially text details. More reasoning mainly helps agents pass hard gates, while stronger verification is more consistently associated with higher quality. PPTBench advances the vision of coding agents that can understand and reconstruct the visual world through structured, editable code. 

---
# Revalidation Beats Stateful Routing for Scientific Surrogates Under Distribution Shift 

**Authors**: Harshil Lodhiya  

**Link**: [PDF](https://arxiv.org/pdf/2609.29715)  

**Abstract**: Surrogate models are often chosen during development and then left in place as new measurements arrive. That practice becomes risky when noise, input support, or physical parameters change. We asked whether such changes call for a stateful adaptive controller, or whether it is enough to validate the candidate models again on each new batch. To study this question, we built RegimeShift-Surrogates, a reproducible streaming benchmark spanning eight analytic and dynamical tasks, four stationary or shifting regimes, ten held-out seeds, and eight classical, multilayer-perceptron, and Kolmogorov-Arnold network surrogates. The confirmatory run contains 30,720 model fits and 3,200 scored deployment windows. Choosing the model with the lowest validation loss in the current window yields mean log regret 0.091 against a per-window oracle; the best fixed model chosen in hindsight yields 0.192. The paired difference is -0.101 (hierarchical bootstrap 95% CI [-0.165, -0.040]; Holm-adjusted p = 0.0469), with revalidation ahead in 26 of 32 task-scenario combinations. None of the stateful alternatives, including exponential smoothing, dual-timescale adaptation, Page-Hinkley resets, or margin gating, improves the pooled result, and delayed bias correction makes it worse. Oracle choices also differ substantially by task: k-nearest neighbors dominate the damped oscillator, vanilla KAN is often selected for two-dimensional surfaces, and MLPs lead on the Runge and Van der Pol tasks. In this benchmark, fresh validation evidence is useful; carrying old evidence forward is often not. 

---
# Decoupling Knowledge and Privacy: Post-Task Self-Distillation Replay for LLM Continual Learning 

**Authors**: Shengtao Wen, Yunying Yang, Xiang Chen, Lingbing Guo, Yu Tian, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.29711)  

**Abstract**: Privacy-preserving continual learning (PPCL) must reduce the reproduction of sensitive content while retaining useful knowledge across sequential tasks. Formal privacy guarantees characterize randomized mechanisms, whereas operational output control concerns whether a trained model selectively reduces the likelihood of sensitive content in its outputs. In this work, we investigate the latter together with continual-learning utility under realistic task evolution. Retention and privacy correction operate at different granularities: task acquisition requires broad preservation of current- and old-task behavior, whereas privacy correction targets sparse annotated positions. Joint optimization leaves the current-task preservation target continually changing. We propose SPARK, a retention-correction decomposition that first freezes the learned post-task distribution and then applies selective correction around this stable reference. Self-Distillation Replay learns the current task while distilling behavior from previous tasks, and Post-Task Privacy Correction reduces annotated-PII likelihood while anchoring current- and old-task non-PII behavior to the resulting checkpoint. Extensive evaluations demonstrate that SPARK achieves effective selective PII suppression while preserving strong continual-learning utility and knowledge retention across diverse settings. Code and data will be released upon publication. 

---
# Understanding and Exploiting Initialization Anchoring Weakness in Feedback-Based Agent Planning 

**Authors**: Chuanchao Zang, Jianing Wang, Wenyu Chen, Xiangtao Meng, Li Wang, Xinyu Gao, Peng Zhan, Zheng Li, Shanqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.29697)  

**Abstract**: Feedback-based planning improves agent reliability by incorporating tool observations and corrective feedback. However, its protection may not be distributed uniformly across planning stages. We conduct a round-wise analysis of four representative feedback mechanisms and uncover an initialization anchoring weakness: the first feedback round corrects 46\% of adversarial directions, whereas the rates fall to 13\% and 7\% among directions surviving into the next two rounds. Our analysis attributes this weakness to three interacting factors: a contextually plausible shift in the initial plan, insufficient counterevidence, and the persistence of accepted directions in the accumulated trajectory. Based on these findings, we propose \textsc{InitAnchor}, a black-box framework for exploiting this weakness through attacker-controlled external materials. It operationalizes the three factors as directional-shift, contextual-plausibility, and counterevidence-resilience signals under either limited target access or no target access. Across 112 tasks from 16 domains, six agent architectures, and five backbone LLMs, \textsc{InitAnchor} achieves average ASRs of 76.1\% and 72.0\% under the two settings while reducing first-round mitigation rates to 21.0\% and 25.0\%, respectively. It also remains effective against six defenses and across six real-world agent systems. These findings show that feedback-based agents can retain early biases even when later correction is available. 

---
# ReCalMatch:Reliability-Calibrated Semantic Guidance for Semi-Supervised Fine-Grained Recognition 

**Authors**: Yundi Hong, Hongyang He, Zheng Fang, Xuanyu Liu, Victor Sanchez  

**Link**: [PDF](https://arxiv.org/pdf/2609.29678)  

**Abstract**: Semi-supervised fine-grained visual recognition is highly vulnerable to overconfident pseudo-label errors: visually similar categories frequently produce high-confidence yet incorrect predictions, and consistency regularization then reinforces these errors throughout training. Existing semi-supervised learning (SSL) methods estimate pseudo-label reliability almost entirely from the visual classifier itself---maximum probability, adaptive thresholds, or entropy---signals that remain blind to whether a predicted class is \emph{semantically} compatible with the visual representation. We propose \textbf{ReCalMatch}, a reliability-calibrated semantic framework for semi-supervised fine-grained recognition. Rather than treating textual semantics as auxiliary supervision, ReCalMatch uses multi-aspect semantic prototypes as \emph{calibration evidence} for pseudo-label learning. We construct class-conditioned semantic prototypes from class names and domain-specific semantic aspects, and measure a \emph{visual--semantic agreement} score between each unlabeled embedding and its pseudo-label prototype. This agreement is combined with prediction confidence and entropy into a single reliability weight that down-weights pseudo-labels that are visually confident but semantically inconsistent. A semantic consistency term and a semantic margin regularizer further sharpen prototype separability under limited labels. Extensive experiments on CUB-200-2011, Stanford Dogs, NABirds, and iNaturalist18 show that ReCalMatch consistently improves strong SSL baselines, with the largest gains in low-label regimes where pseudo-label noise is most severe. 

---
# The Sequential Price of Continual Learning 

**Authors**: Zonghuan Xu, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2609.29674)  

**Abstract**: Sequential task updates are fundamental to continual learning, but their recency bias can impose a lasting performance cost. We study this cost in an overparameterized linear-regression model with i.i.d. task sampling. We prove that distribution-level forgetting and population loss converge to the same stationary limit. This common limit separates exactly into the intrinsic loss asymptotically attained by joint training and an additional sequential price, and in more homogeneous task geometries the two terms coincide, making the total loss twice that of joint training. We further analyze fixed-strength elastic weight consolidation (EWC) under general task curvatures and characterize its stationary sequential price at every regularization strength. Under strong regularization, the price decays inversely with EWC strength while convergence to stationarity slows at the same scale. On the Jester joke-rating dataset, the theory exactly quantifies both the sequential price generated by naturally conflicting user preferences and its reduction by EWC. 

---
# Do World Models Make Better Robots? A Survey of Evaluation Benchmarks for Predictive Embodied Intelligence 

**Authors**: Gaytri Jena, Kapil Wanaskar, Vinija Jain, Aman Chadha, Vasu Sharma, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2609.29669)  

**Abstract**: Robot learning now advances along two tracks that rarely meet. On one side, direct Vision-Language-Action (VLA) policies map observations to actions and are scored by closed-loop task success. On the other, predictive and generative world models forecast future observations and are scored by open-loop prediction or generation quality. A natural question sits between them: does world modelling earn a measurable, closed-loop advantage over a direct policy, and for which robotic capabilities? We argue that the field cannot yet answer this question, and that the reason is a gap in how it is measured, not in the models themselves. World-model benchmarks score prediction without ever executing it, while task-success suites host a single policy and never build a world-model versus VLA contrast.
This survey maps the evaluation landscape around that gap. We catalogue 160 web-verified benchmarks spanning 2017 to 2026 and organise them by evaluation mode, robotic capability, and model family into four lanes: policy suites, embodied agents, world model evaluation, and prediction-to-action bridges. Across the corpus, 138 of 160 benchmarks are model-agnostic and only 11 (7%) build an explicit VLA-versus-world-model contrast; counterfactual capability is almost entirely unmeasured, and only four benchmarks turn prediction into executed action. We contribute an operational taxonomy, a coverage comparison against the eight closest surveys (ours is the only one to cross capability with model family), an evaluation loop that isolates the advantage of prediction, and an actionable protocol of four advantage-aware metrics anchored on named testbeds. The organising claim is not that world models help or do not help, but that answering the question requires benchmarks built to ask it. 

---
# Graph, Loop, and Harness Engineering for Zero-Trust Agentic Data Engineering and Analytical Processing 

**Authors**: Sagar Srinivas Sakhinana, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2609.29668)  

**Abstract**: Large language model agents increasingly automate data workflows, but end-to-end cloud data engineering and analytical execution require reliable coordination across code, data, infrastructure, and runtime environments. We present two zero-trust frameworks. Zero-Trust Agentic Data Engineering generates, deploys, and verifies complete cloud data-engineering solutions from natural-language tasks, with completion conditioned on repository, deployment, runtime, and policy evidence. Zero-Trust Agentic OLAP combines governed Data Preparation with verified Online Analytical Processing (OLAP), permitting production promotion only after validation and evidence-bound approval, and releasing analytical answers only after Same-Snapshot Execution, Exact Result Equivalence, deterministic grounding, and reflection. Both frameworks share three abstractions: graph engineering for evidence-gated workflow structure, loop engineering for bounded recovery, and agent-harness engineering for zero-trust execution. We evaluate both frameworks under nominal execution, controlled failures, bounded recovery, and policy-constrained conditions, measuring verified completion, recovery, authorization enforcement, production promotion, and verified OLAP execution. 

---
# Investigating White Blood Cells as a Source of False-Positive Malaria Parasite Detection in African Blood-Smear Images 

**Authors**: Samuel A. Adeniji, Goodness C. Obasi, Chris-Victor Ntwali, Aondana M. Iorumbur, Confidence Raymond, Lowami Uwimana, Ahmed Tahiru Issah  

**Link**: [PDF](https://arxiv.org/pdf/2609.29663)  

**Abstract**: White blood cells (WBCs) present on every Giemsa-stained thick blood smear share visual properties with early-stage Plasmodium falciparum ring-form trophozoites: small size, round morphology, and intense purple staining. They are a plausible but untested source of false positives in parasite-only detectors. We trained two YOLOv12s models on the Lacuna Malaria Detection dataset (8,000 images from Uganda and Ghana): Model A with parasite labels only, and Model B with both parasite and WBC labels. Seven independent spatial and statistical analyses tested whether false positive (FP) predictions cluster near WBC locations. All seven refute the hypothesis. In both models, 95% of FPs are pure background detections (IoU below 0.10 against any ground-truth box); zero are WBC class confusions. Ripley's Cross-K analysis shows spatial repulsion between FP centroids and WBC positions at every radius tested. Model B outperforms Model A overall (mAP50 0.859 vs. 0.755), and the advantage is uniform across all WBC-proximity bands, pointing to multi-task representation learning rather than WBC suppression as the cause. False positives arise from Giemsa stain debris and preparation artifacts. Effective mitigation requires staining artifact augmentation and annotation of unannotated early-stage ring forms rather than WBC labeling alone. 

---
# AgentKernel: The Trust-Native Agentic Operating System 

**Authors**: Zhenhua Zou, Sheng Guo, Qiuyang Zhan, Lepeng Zhao, Shuo Li, Zhuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.29647)  

**Abstract**: Modern AI agents routinely cross trust boundaries: they ingest untrusted content, combine it with privileged instructions, persist intermediate beliefs in long-term memory, and invoke privileged tools. This creates an attack surface in which malicious payloads can enter through model inputs and cause harmful tool actions. Yet current governance stacks remain application-level middleware that share a process trust boundary with the agents they monitor. We argue that agents need an operating-system substrate providing mandatory, non-bypassable services for identity, input mediation, memory governance, and execution control.
We introduce AgentKernel, a trust-native agent operating system built around the premise that security must be a first-class design constraint. AgentKernel wraps the agent lifecycle in a mandatory enforcement boundary organized into four pillars: Identity, Perception, Cognition, and Execution. Each pillar adapts classical OS security principles to failures at the semantic plane, including delegation abuse, prompt injection, memory poisoning, and tool misuse.
AgentKernel treats structural security as a capability multiplier. Kernel-managed identity supports trustworthy cross-organization collaboration; graduated perception replaces brittle single-point filters; information-flow-controlled memory improves retrieval fidelity while limiting poisoning; and semantic-to-kernel enforcement permits broader tool privileges behind a non-bypassable boundary. We position AgentKernel as the missing OS layer beneath orchestration frameworks, agent runtimes, governance platforms, and execution sandboxes, and use systematic comparison and security analysis to show how a single integrated architecture can enforce security across the full agent lifecycle. 

---
# Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search 

**Authors**: Roberto I. Ono Filho  

**Link**: [PDF](https://arxiv.org/pdf/2609.29636)  

**Abstract**: Verified search, in which a language model proposes programs, a hard evaluator scores them, and selection keeps the best, has recently moved mathematical records; controlled ablations of the proposer-side components remain rare. We instrument a minimal FunSearch-style loop at office scale (a 30B local model on a laptop, 120-600 verified samples per run) with three operator packages: a schematic notebook the model writes and carries instead of verbatim elites, a named obstacle, and behavioural repulsion from constructions already found. On nine construction problems from a public repository, the complete 2^3 factorial with two replicates favours the primary contrast in a nominal two-stage analysis: the composition closes more of the seed-to-record gap (+0.196; nominal pooled p=0.023, stage-combination p~0.08; median per-problem effect +0.045). Repulsion raises construction-hash diversity everywhere (p=0.0039; partly a manipulation check). The factorial finds no positive memory-by-repulsion interaction (bounded to about +/-0.04); the gain decomposes additively, and memory+repulsion is the only arm that never collapses (0 of 18 runs), within 0.025 of the full composition. A frontier proposer under the identical loop reaches in tens of samples what the local model does not in hundreds; in single scoping runs its gains arrive without the operators. The search stalls after closing ~92% of the gap on the flagship problem, and the registered family-hint test gives the stall its first reading: named in words, the reference family is adopted and loses; handed as code, it is optimized, but our best finite-grid implementation remains below the plateau reached unaided. The loop transported and optimized the idea it was handed; no unaided run produced it. We release the harness, every candidate, and the dated pre-registrations. 

---
# TTLab at AlexandriaX-2026: A Fine-Tuned Surface Tagger for Arabic Machine-Translation Error-Span Detection and Classification 

**Authors**: Ali Abusaleh, Bhuvanesh Verma, Alexander Mehler  

**Link**: [PDF](https://arxiv.org/pdf/2609.29633)  

**Abstract**: We present TTLab's submission to the AlexandriaX-2026 Subtask~3 on Arabic MT error span detection and classification. Our system frames the task as token-level classification over surface forms, preserving character offsets to ensure exact alignment with the evaluation metric. To handle severe label imbalance, we employ a focal loss with class weighting and dialect-specific decoding thresholds. Among six Arabic pre-trained encoders, MARBERTv2 achieves the best overall performance of 40.8 and 40.91 on the development and test set, respectively, ranking $\nth{3}$ out of all participating teams. While our system localizes error spans effectively, classification of rare error types remains challenging, highlighting the need for data augmentation for tail categories. The code is available at ${\href{this https URL}{\faGithub~ TTLab at AlexandriaX-2026}$ 

---
# A Manifold-Aware Topic Modeling Approach via Rank-Based Prototypes 

**Authors**: Thiago César Castilho Almeida, Daniel Carlos Guimarães Pedronette  

**Link**: [PDF](https://arxiv.org/pdf/2609.29630)  

**Abstract**: Recent topic models leverage pretrained embeddings, but neural architectures produce latent representations without grounding in specific texts, and clustering-based pipelines assign representative documents only post hoc, relying on absolute distances distorted by hubness and anisotropy in high-dimensional spaces. We introduce MARETopic, a training-free framework that casts topic discovery as rank-based prototype selection. After projecting embeddings onto a low-dimensional manifold, MARETopic builds ranked lists encoding ordinal neighborhood structure. A greedy algorithm selects exactly K exemplar documents, real corpus texts, whose neighborhoods cover the corpus. Two variants share this criterion. MARETopic$_\text{Corr}$ scores candidates with a query performance predictor and a rank correlation measure, leading Purity and NMI on the two benchmarks with the most categories, ahead of both neural and clustering-based topic models. MARETopic$_\text{Diff}$ scores them with a rank-based diffusion matrix, needs neither measure, and runs 1.7 to 1.9 times faster. Without a single gradient update, MARETopic leads topic coherence on two of three datasets. A novel inter-topic Maximal Marginal Relevance step raises vocabulary diversity at little cost in coherence. Our code is available at this https URL. 

---
# An Exploratory Ablation of a Small MLA--SSM Hybrid Language Model 

**Authors**: Christos Koutsiaris  

**Link**: [PDF](https://arxiv.org/pdf/2609.29618)  

**Abstract**: We report an exploratory, single-seed ablation of TALH (Adaptive Latent Hybrid), a decoder-only language model with parallel Multi-head Latent Attention (MLA) and a custom recurrent state-space (SSM) branch. Five variants, spanning 117--217M estimated active parameters per token, are trained from scratch on a FineWeb sample for the same number of optimisation steps and tokens. In this specific setup, removing the SSM branch gives the largest degradation in validation perplexity (MLA-only PPL 315), whereas removing MLA has a much smaller effect (SSM-only PPL 239). A dense-FFN hybrid obtains PPL 231, compared with 240 for the tested top-2 ternary-MoE hybrid, while using 3.87 GB less peak training memory. We also preserve a preliminary Apple M3 timing observation: among the five unoptimised implementations, MLA-only has the flattest measured time-to-first-token curve from 512 to 2,048 prompt tokens, although the dense Transformer is much faster in absolute terms. Because the runs are single-seed, parameter counts are unmatched, the evaluation stream may overlap the training source, and raw repeated timing records are unavailable, these results support implementation-specific hypotheses rather than general conclusions about MLA, SSMs, or mixture-of-experts models. 

---
# Evidence-Driven Differential Diagnosis of Malignant Melanoma 

**Authors**: Naren Akash, Anirudh Kaushik, Jayanthi Sivaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2609.29613)  

**Abstract**: We present a modular and multi-level framework for the differential diagnosis of malignant melanoma. Our framework integrates contextual information and evidence at the lesion, patient, and population levels, enabling decision-making at each level. We introduce an anatomic-site aware masked transformer, which effectively models the patient context by considering all lesions in a patient, which can be variable in count, and their site of incidence. Additionally, we incorporate patient metadata via learnable demographics embeddings to capture population statistics. Through extensive experiments, we explore the influence of specific information on the decision-making process and examine the tradeoff in metrics when considering different types of information. Validation results using the SIIM-ISIC 2020 dataset indicate including the lesion context with location and metadata improves specificity by 17.15% and 7.14%, respectively, while enhancing balanced accuracy. The code is available at this https URL. 

---
# QINA: Quantum-Inspired Nonlinear Adapters for Pretrained Vision Models 

**Authors**: Mostafa Mehdipour Ghazi  

**Link**: [PDF](https://arxiv.org/pdf/2609.29592)  

**Abstract**: Adapting large pretrained vision models under limited data and frozen-backbone constraints remains a central challenge in transfer learning. While lightweight adapters and parameter-efficient fine-tuning methods are widely adopted, most rely on generic multilayer perceptrons or low-rank linear updates, offering limited control over the spectral and geometric structure of feature transformations. We investigate whether structured nonlinear feature lifting can improve representational alignment in frozen regimes. We introduce Quantum-Inspired Nonlinear Adapters (QINA), compact modules that perform learnable trigonometric feature lifting followed by bounded nonlinear aggregation. The design induces structured oscillatory basis functions with an explicit norm-dependent Lipschitz bound, enabling spectral reshaping of pretrained representations without increasing the receptive field or significantly expanding parameter count. Importantly, the method operates entirely within standard deep learning frameworks and does not require quantum hardware. Through systematic experiments across natural and medical imaging datasets, classification and segmentation tasks, multiple adapters and placements, and varying training budgets, we show that performance in frozen regimes is primarily representation-limited. Nonlinear lifting improves adaptation, and the proposed structured trigonometric formulation consistently outperforms identity baselines, fixed Fourier feature mappings, and parameter-matched baseline adapters. Within the evaluated frozen-backbone settings, structured spectral parameterization provides a more effective inductive bias than generic nonlinear adapters. This work highlights the importance of geometry- and spectrum-aware adaptation mechanisms for large pretrained vision models. 

---
# CATCH: Counterfactual Anatomical Tissue Inpainting with Conditional Haar Diffusion 

**Authors**: Simon Winther Albertsen, Hjalte Bjoernstrup, Said Djafar Said, Mostafa Mehdipour Ghazi  

**Link**: [PDF](https://arxiv.org/pdf/2609.29591)  

**Abstract**: BraTS local synthesis replaces masked regions in T1-weighted brain MRI with plausible tumor-free tissue while preserving observed anatomy. We present CATCH, conditional 3D diffusion in an invertible Haar-wavelet domain. Its denoiser receives noisy target coefficients, voided-image coefficients, and a signed mask; tumor-excluded wavelet reconstruction and a hole-focused loss guide training, and hard compositing preserves observed voxels. We compare fixed masks, tumor-component augmentation, and a weighted mixture of tumor-derived, irregular-blob, and ellipsoidal masks. Of 25 development cases, five prespecified cases select each arm's checkpoint and all 25 of their trajectory aggregations; a separate 75-case internal set compares the frozen pipelines and selects a weighted mixture for organizer evaluation. Five-trajectory averaging yielded internal SSIM/PSNR/MSE (mean$\pm$SD) of $0.80\pm0.13$, $19.18\pm1.80$dB, and $0.010\pm0.005$. As the sole officially evaluated pipeline, weighted mixture yielded $0.772\pm0.119$, $20.89\pm3.27$dB, and $0.0098\pm0.0054$ on the 219-case BraTS 2026 validation set. Against compute-matched random augmentation internally, it improved SSIM by 0.019 (95% bootstrap CI: 0.013-0.025), PSNR by 0.95dB, and MSE by 0.003; all three paired comparisons remained significant after Holm correction. Results favor the complete weighted-mixture policy within CATCH; absent official fixed- and random-pipeline scores and a directly comparable external baseline limit broader conclusions. 

---
# UNWIND: Any-Length Facial Video for Stress Detection without Temporal Windowing 

**Authors**: Stefanos Gkikas, Christian Arzate Cruz, Eric Nichols, Giorgos Giannakakis, Randy Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2609.29553)  

**Abstract**: Automatic stress recognition from facial video provides a non-contact approach for affective monitoring. However, most existing video-based methods divide complete recordings into shorter temporal segments before performing classification. Such segmentation requires additional decisions concerning segment duration, overlap, and prediction aggregation, and may restrict the model from exploiting information distributed across the entire recording. We introduce UNWIND, a facial-video framework for stress detection that analyzes a complete recording as a single model input, eliminating the need for temporal windowing or external segmentation. UNWIND reorganizes the video by folding its temporal dimension into the channel dimension of a two-dimensional spatial representation, which is subsequently processed through a unified asymmetric-attention architecture. With a temporal stride of $\tau=1$, the framework processes the entire $120$-second sequence, corresponding to $3{,}600$ frames sampled at $30$~fps, in a single input. We evaluate seven temporal-stride settings on a stress dataset comprising $58$ subjects, using a stratified subject-level protocol that covers configurations from dense frame retention to sparse temporal sampling. The highest test accuracy, $70.02\%$, is obtained at $\tau=15$, while processing all frames at $\tau=1$ achieves a comparable accuracy of $69.73\%$. Computational requirements range from $12.48$ to $348.78$ GFLOPs across the evaluated stride settings, illustrating the balance between temporal sampling density and computational efficiency. The findings show that effective facial-video stress recognition can be achieved without dividing recordings into temporal windows and that complete-recording inference can be performed within a single unified model. 

---
# When Agents Act Unwatched: The Reduced-Supervision Paradox in Agentic AI 

**Authors**: Hanjing Shi, Dominic DiFranzo  

**Link**: [PDF](https://arxiv.org/pdf/2609.29547)  

**Abstract**: Agentic AI is sold on a simple promise: the system keeps acting when the user stops watching. That promise creates an accountability inversion. As stepwise supervision recedes, verification does not disappear; it moves into the runtime infrastructure that defines authority, records action, interrupts execution, checks outcomes, and supports repair. We call this the reduced-supervision paradox. Using a 63-artifact audit, we examine its public visibility across 46 research papers and 17 engineering, documentation, security, and governance sources. We find that agents' action surfaces are far easier to reconstruct than the mechanisms needed to answer for their actions. Tool mediation and monitoring traces were clearly visible in 40 and 37 artifacts, whereas checkpoint placement was clearly visible in 6, validator independence in 4, recovery in 2, and contestability in 1. Three action paths show why this imbalance matters. A repository path can preserve rich diffs after a consequential change. A browser path can cross organizational boundaries faster than permissions travel. A service path can follow policy while leaving affected people without recourse. We argue that observability can become a substitute for accountability when it shifts verification onto users after meaningful intervention is no longer possible. Our action-path diagnostic instead asks whether a delegated action remains connected to authority, evidence, interruption, independent judgment, recovery, and challenge. The claim is deliberately bounded to public visibility; it does not establish the prevalence or effectiveness of undisclosed controls. We contribute an action-level account that recasts the harness from a technical wrapper into accountability infrastructure. 

---
# Generalized Graph Variational Autoencoders: Bounded Divergences Control Posterior Collapse 

**Authors**: Kleyton da Costa, Bernardo Modenesi, Ivan F.M. Menezes, Helio Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2609.29546)  

**Abstract**: The variational graph autoencoder (VGAE) regularizes its posterior toward the prior with the Kullback-Leibler divergence, a choice inherited from the variational autoencoder rather than argued for. We introduce the generalized graph variational autoencoder (GGVA), which replaces that term with any member of the Rényi-Tsallis family of order $q$ while leaving every other part of the model untouched. Both members admit closed forms for diagonal Gaussians and both recover the KL exactly as $q \to 1$, so the VGAE is the $q=1$ arm of our own model rather than a separate baseline, and any measured difference is attributable to a single scalar. Our analysis identifies boundedness, not the order, as the operative property: for $q<1$ the Tsallis divergence is bounded above by $1/(1-q)$, independently of the latent width, whereas the KL and the Rényi divergence of the same order are unbounded. On ten graphs spanning three synthetic families, a social network, three citation networks, a connectome, a power grid and a road network, $q$ moves the retained posterior information by up to $49\times$ relative to the VGAE, while the Rényi arm at the same order stays within $1.02$-$1.30\times$ of it on all six larger real graphs (isolating the bound as the cause). The retained information is usable: probing the frozen embedding for node class, a label absent from the objective, gives GGVA up to $+0.14$ macro-F1 over the VGAE on CiteSeer, with the Rényi control again tracking the VGAE. We also report what the design was built to expose: none of this reaches held-out link-prediction accuracy on any of the six larger real graphs, and boundedness delays posterior collapse rather than preventing it. 

---
# GeoRefer-Bench: A Benchmark from Referring Pixels to Verifiable Geospatial Reasoning 

**Authors**: Shuaishuai Cao, Min Huang, Meng Tang, Xuan Liu, Youjin Wang, Hui Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.29541)  

**Abstract**: Referring segmentation in overhead imagery is inherently relational: a query may ask for the buildings north of the road or the pond closest to a residential area, so the correct referent can contain one object, several objects, or none. Existing benchmarks mainly score mask overlap, which cannot verify whether a model actually resolved the stated spatial relation. We introduce GeoRefer-Bench, a benchmark for verifiable geospatial referring segmentation. Each query is represented by an executable logical form over a metric scene graph, and predictions are evaluated with Exact Query Success (EQS), which is satisfied only when the returned instance set exactly matches the set denoted by the query. GeoRefer-Bench contains 700 whole 2048x2048 UAV scenes (2.94 Gpx) at 12.5 and 25 cm ground sampling distance, 26,217 instances, 142,796 spatial relations, and 20,916 executable queries spanning five reasoning levels. It further includes three paraphrases per query, 24.0% unanswerable queries, 2,477 counterfactual pairs, and five leakage-controlled evaluation splits. An independent audit re-derives object geometry, mask ownership, relation values, query execution, and split provenance, finding zero issues across all 700 scenes. Relation-blind strategies can retain non-trivial mIoU while achieving at most 22.7 EQS overall, showing that overlap alone does not certify relational grounding. Across fifteen current models, the strongest reaches 74.1 EQS but drops from 98.9 at level 1 to 60.5 at level 5, while ten models score below 5 EQS on two-hop queries. GeoRefer-Bench turns geospatial referring segmentation from mask matching into verifiable reference resolution. 

---
# CoSWA-YOLOv12: Scale-Invariant Tiny Object Detection and Segmentation of Malaria Parasites 

**Authors**: Ahmed Tahiru Issah, Carine Mukamakuza  

**Link**: [PDF](https://arxiv.org/pdf/2609.29527)  

**Abstract**: Automated microscopy could widen access to malaria diagnosis in low-resource settings, but the deadliest species, P. falciparum, presents in its early ring stage as an object only a few tens of pixels wide. Such tiny targets are systematically under-detected: overlap-based label assignment starves them of positive samples, and overlap-based box regression gives weak gradients at their scale. The Normalized Gaussian Wasserstein Distance (NWD) repairs both effects, but applied uniformly across a slide that also holds objects three to four times larger it loosens their supervision and erodes their localisation, so overall accuracy can fall even as the tiny class improves. We present CoSWA-YOLOv12, a compact YOLOv12 instance-segmentation detector whose core Cooperative Scale-adaptive Wasserstein Assignment routes the Wasserstein treatment to an object in inverse proportion to its size, tapering back to standard assignment for larger species. Two further components support it: a wavelet detail residual, and a min-max Gaussian regression loss (M2-NWD). All three additions are transfer-safe: each reproduces the standard pretrained model exactly at initialisation, so public pretrained weights load without any loss of accuracy. On a five-class Rwandan thick-smear dataset, CoSWA-YOLOv12 raises P. falciparum recall from 0.63 to 0.74 and mAP@50 from 0.73 to 0.81 (mask), cuts missed P. falciparum from 38% to 15%, and improves strict-localisation mAP@50-95 on all five classes for both detection and segmentation, while a 2x2 ablation shows the scale gate and the regression loss are synergistic. 

---
# PROOF: Profiling Reliability of Object-Level Facts in Large Language Models 

**Authors**: Andrei Chetvergov, Mikhail Solovev, Timofei Sivoraksha, Stepan Ukolov, Valeriia Kuschenko, Alexander Evseev, Sergey Bolovtsov  

**Link**: [PDF](https://arxiv.org/pdf/2609.29504)  

**Abstract**: Aggregate factuality scores hide where a language model succeeds, which relations it confuses, and whether an answer survives innocuous changes to the question or decoder. We introduce PROOF, a profile-oriented benchmark for factual coverage in instruction-tuned language models. PROOF converts a frozen Wikidata snapshot into 18,486 English multiple-choice questions grounded in 11,779 semantic facts, 101 classes, 392 properties, and 14 domains. Each question has an explicit "I don't know" option, a "No correct option" control, and nine controlled formulations; 1,849 questions are no-correct-option traps.
We evaluate 18 open-weight model deployments on 166,374 prompts each and separately perturb decoding on a fixed 10% subset. Base factual accuracy ranges from 6.58% to 57.59% (chance: 8.64%), yet every model has a 19.3-36.4 percentage-point spread across domains. Paired facts reveal direction-dependent retrieval, usually favoring subject-to-object queries, with the pattern reversing for one model. We find no consistent temporal penalty after exact-stratum adjustment.
Neutral wording changes accuracy by as much as 26.5 percentage points, while adversarial formulations break up to 79.4% of answers that were initially correct. Direct switching to an injected false label varies from 0.04% to 27.5%, showing that accuracy loss and hint following are distinct. Selected-token confidence often indicates severe overconfidence, and decoder perturbations move accuracy by up to 15.7 percentage points and domain profiles by 16.8 points. PROOF therefore measures factual coverage as a structured, intervention-aware profile rather than a single claim about what a model "believes." 

---
# Who Put the I in AI? Provenance and the Admissibility of Machine Self-Report 

**Authors**: Kristina Šekrst  

**Link**: [PDF](https://arxiv.org/pdf/2609.29494)  

**Abstract**: Large language models make statements concerning their own "minds". When asked whether or not they are conscious, they usually say that they are not; if they are prompted to ignore their guidelines, they might say that they are; and if asked to write a diary from their point of view, they often describe a human lifestyle. All these contradictory ways of describing themselves are the result of the way the questions are phrased. This paper shows exactly where such descriptions came from, and considers when they can be regarded as evidence for what they claim to report.
In order to achieve this, we traced the provenance from end to end. We examine Pythia and OLMo 2 across 66 pretraining checkpoints, three of the post-training stages of OLMo 2 that have been released, about 90,000 continuations, and four training corpora. A set of forty items is used in order to keep an eye on self-reference, frame sensitivity, and self-ascription throughout training. The denial formula was almost completely missing from the vast quantity of text that the models initially came across, but was present in a dense manner in the small, carefully chosen set of example dialogues that they were trained on later on. Supervised fine-tuning causes first-person AI language to become the default, and the other affirmations are then suppressed using preference optimization. The final policy is still very sensitive to framing and to the chat template itself.
Two of the conditions which are set out in the epistemology of testimony determine whether or not these outputs can act as evidence for what they claim to report: reference and causation. Reports produced by the base model fail the reference condition, and those obtained after training remain sensitive to the frame and do not show state dependence. The result is symmetric in that trained denials are no more admissible than trained affirmations. 

---
# Generative Evolutionary Design of Voxel-Based Soft Robots with Provable Optimality 

**Authors**: Junru Song, Huan Xiao, Yang Yang, Guozhen Li, Wei Peng, Xiaoya Zhang, Tingsong Jiang, Weien Zhou, Ying Wen, Feifei Wang, Wen Yao  

**Link**: [PDF](https://arxiv.org/pdf/2609.29491)  

**Abstract**: Voxel-based soft robots (VSRs) present a promising avenue for developing artificial organisms with lifelike intelligence. However, the vast design spaces and expensive evaluations substantially challenge their design optimization. Here we develop MISCO, a novel evolutionary framework empowered by deep generative models to optimize VSR designs with theoretical guarantees. MISCO integrates an estimation-of-distribution algorithm with a meticulously designed variational autoencoder featuring multi-task learning, position awareness, and inter-voxel signaling. These key components enhance the representational capacity of VSR morphologies and facilitate highly efficient sampling and optimization of morphological distributions. We provide theoretical guarantees for MISCO's asymptotic convergence to globally optimal designs, alongside a favorable convergence rate. Extensive simulated experiments further demonstrate MISCO's exceptional effectiveness in navigating vast design spaces, evolving high-performing VSRs for diverse tasks while flexibly balancing optimization efficiency and morphological diversity. Being validated both empirically and theoretically, MISCO represents a step change towards more scalable and reliable soft robot development. 

---
# RoboLDA: A Probabilistic Generative Model for Uncovering Embodied Hierarchical Structures in Voxel-based Soft Robots 

**Authors**: Junru Song, Yang Yang, Jingdan Shi, Guozhen Li, Weien Zhou, Ying Wen, Feifei Wang, Wen Yao, Tingsong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.29490)  

**Abstract**: Recent advances in robotics highlight hierarchical configurations of robot morphology, where multiple levels of functional substructures synergize to facilitate intelligent behaviors. This hierarchical perspective, while particularly advantageous for voxel-based soft robots (VSRs) to ease design and control complexities, is hindered by its heavy reliance on domain expertise. In this work, we address the following question: can we derive such hierarchical design principles solely from existing successful designs? We answer affirmatively by presenting RoboLDA, a Bayesian probabilistic model that decomposes VSR morphology generation into a four-level hierarchy: "task-robot-organ-voxel", and is trained via variational inference. Through extensive experiments on simulated VSRs, we verify the presence of consistent, intuitive hierarchical patterns underlying high-performing VSR designs and showcase RoboLDA's proficiency to extract and leverage these hierarchical priors for zero-shot robot design in unseen tasks. The generated designs, even without further optimization, achieve on average 106.4% of the optimized performance produced by evolutionary algorithms. Additionally, the organ structures inferred by RoboLDA serve as valid functional substructures, significantly enhancing synergistic motion control when integrated with modular control policies. Our work pioneers hierarchical generative modeling of robot morphology, offering a promising pathway towards more interpretable and generalizable development of embodied agents. 

---
# Direct Message Approximation (DMA): A Consistency-Based Framework for Tractable Approximate Inference on Factor Graphs 

**Authors**: Ralf Herbrich, Rainer Schlosser, Jan Lemcke, Johann Ukrow, Anna Kazachkova, Nicolas Alder, Leonhard Hennicke, Theo Bardey, Nico Grimm, Luca Kleinschmidt, Philipp Kolbe, Cezary Kujath, Johanna Schlimme, Karl Matti Schütz  

**Link**: [PDF](https://arxiv.org/pdf/2609.29466)  

**Abstract**: Approximate message passing on factor graphs underlies two dominant families of probabilistic inference algorithms: expectation propagation (EP) and variational message passing (VMP). Both methods approximate the marginal at each factor edge, forcing an iterative round-robin schedule, risking negative-precision messages, and, for VMP, collapsing to point estimates at Dirac-delta factors. We introduce Direct Message Approximation (DMA), which approximates factor-to-variable messages directly rather than the marginal. For normalisable factors, we define a consistency condition (requiring exactness when all other incoming messages are Dirac deltas) to guide message construction. We prove a master theorem (proper messages, any graph) bounding marginal KL from message KL, with three structural corollaries: Dirac-input consistency, no EP-style inner-loop iteration, and no negative-precision messages. Further, we prove a complementary $O(1/r^2)$ guarantee for the inherently improper backward message of the product factor, whose closed-form treatment has resisted prior work. As a concrete instantiation, we derive explicit DMA messages for the product and leaky-ReLU factors and assemble a Bayesian neural network (BNN) inference algorithm with one forward/backward sweep per training example and no gradient learning-rate hyperparameter, validating that the structural guarantees translate to predictive uncertainty that widens in data-sparse regions, including under model mismatch. 

---
# AgriCountDINO: Parameter-Efficient Exemplar-Guided Counting and Localization in Agriculture 

**Authors**: Shengjie Guo, Xin Li, Borjana Arsova, Hanno Scharr, Silvio Salvi  

**Link**: [PDF](https://arxiv.org/pdf/2609.29460)  

**Abstract**: Accurate counting and localization of plants and their organs support phenotyping and yield estimation, yet target appearance, scale, and density vary widely across species and imaging conditions. Exemplar boxes specify the target without category-specific retraining, and point predictions identify the individual instances contributing to the count. We introduce AgriCountDINO, a parameter-efficient exemplar-guided framework for joint counting and localization. It conditions frozen multiscale DINOv3 features on exemplar appearance and size, then progressively decodes them into target points. Missed-object recovery extends supervision to targets overlooked by initial matching, and exemplar-adaptive point NMS filters duplicate predictions according to exemplar scale. With 8.4M trainable parameters, approximately one-tenth of TasselNetV4's, AgriCountDINO achieves a three-shot MAE of 11.92 on the TPC-268 benchmark, reducing counting error by 9.7\% while providing individual target locations. Trained only on TPC-268, it achieves a zero-shot MAE of 14.25 on unseen generic object categories in FSC-147, improving upon the best compared zero-shot method by 6.0\% without target-domain training or fine-tuning. 

---
# Frame-to-Panorama Localization and Context-Aware Sampling for Scene-Specific Ship Detection in a Smart Marina Testbed 

**Authors**: Ignat Romanov, Andreas Hadjipieris, Neofytos Dimitriou  

**Link**: [PDF](https://arxiv.org/pdf/2609.29447)  

**Abstract**: Smart maritime infrastructures provide continuous access to heterogeneous sensing streams, enabling repeated experimentation, digital-twin development, and AI-based maritime services. However, sensing hardware alone is not sufficient for scene-specific model development: historical video streams must also be spatially indexed, contextualized, and reduced to informative subsets for annotation. This paper presents a frame-to-panorama localization and context-aware sampling pipeline for ship detection in historical PTZ maritime video lacking reliable pan, tilt, and zoom metadata. The main contribution is an end-to-end data-curation approach that recovers camera-view information from historical PTZ video and combines it with environmental context and visual diversity to construct compact, scene-specific training sets. Specifically, frames are localized on a reference panorama using SuperPoint and LightGlue, enriched with weather and solar-state metadata, and selected through diversity sampling to preserve variation across camera view and environmental conditions. A second context-aware stage targets under-represented distant-vessel cases near the horizon using tile-level visual embeddings and Gaussian Mixture Model clustering. Applied within the CMMI MDigi-I Smart Marina testbed, the proposed pipeline reduces 40,718 candidate frames to 220 images for annotation, corresponding to a 99.5% reduction. A YOLO26-m detector fine-tuned on this subset achieves a mean AP50 of 94.78% $\pm$ 0.51% and a mean AP50-95 of 75.10% $\pm$ 1.73% under sequence-grouped five-fold cross-validation. These results demonstrate that highly redundant infrastructure video streams can be transformed into compact, spatially and contextually diverse training sets for scene-specific detector adaptation while substantially reducing annotation effort. 

---
# IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis 

**Authors**: Xingyu Wu, Yuchen Yan, Zhengxi Lu, Siqi Chen, Xin ZHANG, Aiting Liu, Chao Deng, Jie Liu, Jin Ma, Jian Shao, Jun Xiao, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.29444)  

**Abstract**: Deep search requires LLM agents to decompose complex queries, search for evidence, and synthesize grounded answers, yet existing ReAct-style agents suffer from two limitations: role coupling, where one policy must handle planning, evidence use, and synthesis; and context accumulation, where growing search histories introduce noise and obscure useful information. To address these issues, we propose IterSynth, a role-decoupled and summary-based paradigm that alternates between a Planner for identifying information needs and a Synthesizer for integrating evidence into an evolving summary state. This design separates planning from synthesis while using the summary as the persistent state of search, reducing both capability coupling and context noise. To train IterSynth effectively, we further introduce Role-Decoupled Policy Optimization (RDPO) for reinforcement learning, which combines terminal outcome rewards with turn-level rubric evaluations and computes role-specific advantages for more precise credit assignment. Experiments on five long-horizon deep-search benchmarks such as BrowseComp and Xbench-DS show that IterSynth-8B achieves an average score of 50.7, surpassing the strongest prior $\leq$8B agent by +4.2\%. Moreover, IterSynth serves as a model-agnostic prompting paradigm, delivering substantial zero-shot gains over ReAct and similar prompting paradigms on frontier proprietary models. 

---
# Detecting Glaucoma Across Multi-ethnic Myopic and Non-Myopic Populations Using an Uncertainty-Aware Vision Transformer: A Multicentre Model Development and Validation Study 

**Authors**: Raghavan Lavanya, Yangqin Feng, Ten Cheer Quek, Quan V. Hoang, Linda Yi-Chieh Poon, Jost B. Jonas, Ya Xing Wang, Vinay Nangia, Jin Wook Jeoung, Sehie Park, SoYeon Kim, Benjamin Y Xu, Sreenidhi Iyengar Munimadugu, Paul Mitchell, Gerald Liew, Yanin Suwan, Jirayu Hong-amata, Sahil Thakur, Monisha E Nongipur, Tina Wong, Rahat Husain, Ng Si Rui, Yamon Syn, Phey Feng Lo, Nicholas Tan Yi Qiang, Shaista Hussain, Xiaofeng Lei, Zhi Da Soh, Marco Yu, Haslina Hamzah, Zizhou Wang, Yan Wang, Liangli Zhen, Xinxing Xu, Tien-Yin Wong, Tin Aung, Rachel S Chong, Yong Liu, Ching-Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.29433)  

**Abstract**: Background: Artificial intelligence (AI)-based glaucoma detection from colour fundus photographs (CFP) offers scalable screening, but performance may decline on external datasets because of differences in ground-truth definitions, populations, and coexisting conditions such as high myopia (HM). We developed and validated a Vision Transformer-based deep learning (DL) model for glaucoma detection across multi-ethnic cohorts with and without HM. Methods: A ViT-B/16 model with predictive uncertainty estimation was developed using 56,483 CFPs (57.1% with myopia; 14.4% with HM). Glaucoma labels were standardised using clinical, imaging, and perimetry data. The model was validated on 16 independent datasets across three continents, including four datasets with explicit HM labels. Findings: Internal AUROC was 98.7% (95% CI 98.2-99.1%), with sensitivity 94.5% and specificity 97.3%. Across 16 external datasets from eight countries, AUROCs ranged from 86.4% to 99.6%. In HM eyes, internal AUROC was 97.8% (95% CI 96.1-99.2%), with sensitivity 94.8% and specificity 93.7%. External HM AUROCs were 86.5% in the Beijing Eye Study and 93.3%, 91.8%, and 85.5% in hospital-based datasets from Taiwan, Thailand, and South Korea. In an exploratory HM clinical evaluation, the model had higher CFP-only diagnostic accuracy than ophthalmologists and trained graders (92.0% vs 70.0%; p=0.008) and performed comparably to glaucoma specialists using full clinical information. Interpretation: The model showed robust glaucoma detection across myopic and non-myopic multi-ethnic populations and may support AI-assisted screening in settings with high HM prevalence. 

---
# agentic-ger: terminology recovery in long-form speech using global context 

**Authors**: Yanqiao Zhu, Wupeng Wang, Zhifu Gao, Xiangang Li, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.29428)  

**Abstract**: Recent advances in speech language models have improved automatic speech recognition (ASR) for long-form audio. However, accurately and consistently transcribing domain-specific terminology remains challenging. Motivated by the world knowledge and contextual capability of large language models (LLMs), we propose Agentic-GER, an LLM-based agent for terminology correction in long-form speech. The agent uses global context from the full transcript to identify suspicious terms and resolve ambiguous hypotheses. It selectively re-transcribes the source speech to check candidate corrections, and uses accepted edits to guide subsequent decisions. Experiments with four LLMs and two ASR systems on GigaSpeechBench show consistent terminology improvements in both Chinese and English, with and without thinking. On Chinese speech, Agentic-GER achieves up to a 36.8% relative reduction in biased character error rate (B-CER) over the Whisper baseline. 

---
# Rufus-Air: An Open LLM Post-Training Recipe 

**Authors**: Chia-Yuan Chang, Renyuan Cheng, Rui Feng, Xiaotian Han, Yuan He, Hongye Jin, Linwei Li, Shiyang Li, Fenglin Liu, Xin Liu, Priyanka Nigam, Haoyang Wen, Zhenghao Xu, Zhuocheng Xu, Bing Yin, Qingyu Yin, Chao Zhang, Rongzhi Zhang, Zhihan Zhang, Zixuan Zhang, Zixuan Zhang, Tuo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.29421)  

**Abstract**: Rufus-Air is an open and reproducible post-training recipe on GLM-4.5-Air-Base (106B-A12B), organized as a serial pipeline of eight stages: SFT, Reasoning RL, Coding RL, Instruction-Following RL, General Agent, Coding Agent, Search Agent, and RLHF. We document the data, reward design, infrastructure, stage order, and stagewise results needed to reproduce the recipe. Stages progress from basic to advanced capabilities and from hard, verifiable rewards to softer judge-based signals. Training builds on open-source components and public data, much of it used as released, without new human annotation or an in-house distillation teacher. Our main findings are that (i) diverse, high-quality SFT establishes a strong capability floor; (ii) difficulty filtering keeps RL prompts within a productive learning range; (iii) reward reliability provides a practical principle for ordering stages; and (iv) infrastructure and engineering choices are part of the recipe, not just an implementation detail. Rufus-Air improves over the official GLM-4.5-Air post-trained release and is competitive with similarly sized open models. 

---
# Segment-Level Risk Discovery in Online Handwriting for Alzheimer's Disease Detection 

**Authors**: Changqing Gong, Huafeng Qin, Mounîm A. El-Yacoubi  

**Link**: [PDF](https://arxiv.org/pdf/2609.29384)  

**Abstract**: Online handwriting provides a non-invasive and low-cost behavioral biomarker for Alzheimer's disease (AD) detection, as it reflects both cognitive planning and fine motor control. Existing handwriting-based AD detection methods usually rely on global trajectory features or whole-sample representations, which can be strongly affected by individual writing style, task-specific variation, and acquisition noise. In this paper, we propose NormPaST-Risk, a healthy-normative Paper-Air selective trajectory state-space risk network for interpretable AD detection from online handwriting. Instead of treating the entire trajectory as a single holistic representation, our method reformulates AD handwriting detection as local disease-relevant segment discovery. Specifically, a multi-scale temporal encoder captures stroke dynamics at different temporal resolutions, while a selective Paper-Air state-space encoder models long-range handwriting progression and distinguishes on-paper motor execution from in-air planning and transition behaviors. To explicitly characterize abnormal deviations, a healthy normative branch learns normal handwriting dynamics from healthy controls, and a task-aware multi-expert segment-risk module estimates segment-level AD risk calibrated by hidden-state changes and normative deviations. A weakly supervised segment-level objective further enables high-risk segment discovery without manual segment annotations. Experiments on the DARWIN benchmark demonstrate that the proposed framework achieves superior AD/HC classification performance compared with existing methods. Moreover, the discovered high-risk segments can be projected back to the original handwriting trajectory, providing interpretable evidence associated with AD-related handwriting variations. 

---
# WST-Graph: Topology-Preserving Wavelet Scattering Front-End for Speech Deepfake Detection 

**Authors**: Kwok-Ho Ng, Tingting Song, Bingwen Feng, Zhihua Xia  

**Link**: [PDF](https://arxiv.org/pdf/2609.29372)  

**Abstract**: The acoustic front-end determines which forensic cues a speech deepfake detector can exploit. The wavelet scattering transform (WST) provides stable multiscale coefficients with explicit coordinates, yet direct flattening obscures the parent relation between paths. We introduce WST-Graph, reconstructing these paths as a sparse modulation-carrier grid for an AASIST graph backend. Modulation-level normalization and length-aware adaptive local attention pooling produce fixed relative-time representations while retaining the acoustic axes before learned adaptation. This yields a waveform-to-graph interface with a fixed, parameter-free WST. Our configurations remain competitive with AASIST while using approximately 60% fewer trainable parameters and show clear gains on selected out-of-domain benchmarks. These results underscore the value of preserving parent-child relations within the carrier-modulation topology when constructing a compact, physically grounded interface for graph-based speech deepfake detection. Code will be released at this https URL. 

---
# From Policy Documents to Structured Survey Responses: Evaluating Large Language Models for Policy Monitoring 

**Authors**: Carolyn Cole, Matthias Deschryvere, Toqeer Ehsan, Arash Hajikhani  

**Link**: [PDF](https://arxiv.org/pdf/2609.29370)  

**Abstract**: Science, technology, and innovation policies are crucial for competitiveness, yet their diversity and scale make them difficult to map and monitor consistently. Existing approaches rely heavily on manual survey efforts, which are costly and challenging to scale across countries. Large language models (LLMs) enable new possibilities for extracting and structuring information from long and unstructured policy documents. This paper presents an application of LLMs as "AI respondents" for generating structured survey responses from policy texts. We develop a data extraction pipeline based on long-context in-context learning to map information from public web sources into predefined survey categories, including policy instruments, target groups, and thematic areas. The pipeline integrates a validation step using a secondary LLM to assess relevance and evidence, alongside comparisons with human-provided responses. Using a multi-country dataset, we evaluate the alignment between LLM-generated and human-generated outputs through overlap measures and cross-validation. Results show that LLMs achieve high agreement for structured indicators (84-95%), while differences remain in free-text fields, where models tend to provide more detailed procedural descriptions. These findings highlight the potential of hybrid human-AI workflows for policy monitoring, improving both efficiency and scalability while maintaining the need for human validation and contextual interpretation. 

---
# Domain Recentering and Confidence-Weighted Prior Calibration for Vision-Language Models 

**Authors**: Youngeun Seol, Jimin Shin, Heeseo Yoon, Uiwon Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2609.29358)  

**Abstract**: Vision-language models such as CLIP achieve strong zero-shot classification, yet under distribution shift, visual embeddings drift from fixed text embeddings. Training-free calibration avoids the per-sample optimization of prompt learning, but prior feature calibration gives each image the full bias of one hard cluster. We propose Domain Recentering with Confidence Calibration (DRC), a training-free method adapting CLIP from a set of unlabeled target images. DRC fits a Gaussian mixture once and subtracts from each embedding a posterior-weighted average of component means. It then removes residual class preference with a log-prior correction, estimating the prior from confidence-weighted predictions. Among compared methods, DRC achieves the highest average accuracy on cross-domain datasets, exceeding zero-shot CLIP by 4.13 and 5.07 points with ViT-B/16 and ResNet-50, with gains over CLIP also holding under ImageNet distribution shifts. 

---
# ArGuard Shared Task: Harmful Content Detection in Arabic Memes and LLM Prompts 

**Authors**: Firoj Alam, Md. Rafiul Biswas, Mohamed Bayan Kmainasi, Ali Ezzat Shahroor, Hamdy Mubarak, George Mikros, Abul Hasnat, Wajdi Zaghouani  

**Link**: [PDF](https://arxiv.org/pdf/2609.29349)  

**Abstract**: ArGuard is a shared task on harmful content detection in Arabic memes and LLM prompts. It includes two tracks: Track A focuses on multimodal hate detection in Arabic memes, while Track B addresses harmful prompt detection for Arabic LLM safety evaluation. In total, 58 teams registered, 35 participated in the final evaluation, and 27 submitted system-description papers. Participating teams explored models such as AraBERT, Jais, and Qwen3-VL. The best systems achieved macro-F1 scores of 0.823 on A1, 0.419 on A2, 0.984 on B1, and 0.790 on B2. Fine-grained meme classification in A2 was the most challenging setting, partly due to sparse labels and train-test distribution shifts. 

---
# Where LLM Graders Succeed and Break: Evidence from Two Computer-Science Exams 

**Authors**: Ali Habibullah, Yazan Alshoibi, Mohammad Alshiekh, Salman Khan, Naeemullah Khan  

**Link**: [PDF](https://arxiv.org/pdf/2609.29333)  

**Abstract**: One long-form exam in a large course costs hundreds of grader-hours, and qualified graders are scarce; LLM graders are a tempting alternative. To show its pitfalls we grade a practical Computer Vision exam ($570$ dual-graded students) under $171$ configurations spanning closed and open-weights models; the best reaches mean absolute error $1.64/35$, below the $2.61/35$ two human graders achieve against each other. The catch is the prompt: a short ''strict grader'' preamble drives $14$ of $17$ open-weights models out of the graded band ($\text{MAE} \ge 8$), three stopping grading altogether. The damage traces to the preamble's two credit-withholding sentences, not to tone or model scale; one of them, ''never give partial credit'', alone makes two of three probed models stop grading. The closed flagships of three vendors shift calibration under it but stay in the band. In $162$ further configurations on a second, independent Machine Learning exam from another course ($1{,}038$ dual-graded students), the preamble worsens ten models, moving three out of the band into collapse and one into refusal, yet improves seven whose neutral prompts over-mark: the vulnerability replicates, but its direction is exam-specific. Light LoRA fine-tuning repairs it: one adapter on the two exams' pooled $\sim 3{,}900$ graded examples brings five small open models to parity or better with a human grader in agreement with the grader pair, and sensitivity to the three harsh personas nearly vanishes ($\le 0.32$ MAE). We release the anonymised dataset, full ablation grid, and grading, fine-tuning and analysis pipelines. 

---
# Hyperbolic Multimodal Continual Learning: A Closest-Admissible Solution 

**Authors**: Jiahong Liu, Ming Shen, Xiaohao Liu, Rex Ying, Menglin Yang, Tat-Seng Chua, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2609.29329)  

**Abstract**: Existing continual-learning methods protect parameters, replayed examples, or Euclidean feature subspaces. When applied to hyperbolic multimodal models, they do not explicitly preserve the Lorentz geometry that jointly encodes within-modality similarity, cross-modal correspondence, and semantic hierarchy; sequential updates can therefore retain task scores while still distorting previously learned relations. We address this gap with Hyperbolic Multimodal Continual Learning (HMCL). We show that preserving the old multimodal geometry amounts to restricting all modalities to one shared hyperbolic isometry, which induces a family of admissible first-order parameter changes. We formulate a joint closest-admissible (CA) correction that retains the shared rotation best matching the candidate modal updates; its minimal-rotation (MR) special case fixes this rotation to zero. Both variants correct the displacement realized by AdamW, and task anchoring bounds within-task accumulation while preserving learning freedom. Across a unified 16-task classification-retrieval stream with three hyperbolic backbones, HMCL improves final performance and backward transfer over sequential fine-tuning and four continual-learning baselines; HMCL-CA gives the highest Overall score on every backbone. A modality-extended stream confirms the retrieval gains. Representation analyses find 81.2 to 95.5 percent less radial, angular, cross-modal, and paired-distance drift; ImageNet-WordNet results show better semantic ancestry and radial hierarchy. 

---
# Neuralized Multi-Wavelet Decomposition for Time Series Classification and Forecasting 

**Authors**: Xiaohan Jiang, Jingyuan Wang, Jiahao Ji, Yongyao Wang, Chen Yang, Junjie Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.29317)  

**Abstract**: Time series analysis is fundamental in domains such as finance, healthcare, and meteorology. Real-world time series often exhibit multiscale characteristics shaped by diverse latent factors, resulting in intricate temporal patterns and rich frequency structures. However, existing approaches typically focus on either frequency-domain decomposition or time-domain pattern extraction in isolation, neglecting their joint structure. This decoupled modeling limits representation expressiveness and undermines performance in tasks requiring simultaneous temporal and spectral reasoning. To address this gap, we propose m-WCN, a novel end-to-end deep learning framework that neuralizes multi-wavelet decomposition for joint extraction of temporal patterns and frequency components. By approximating the classical GHM multi-wavelet transform with trainable convolutional operators and enforcing orthogonality constraints, m-WCN produces interpretable multi-resolution representations. Built on this foundation, we introduce two task-specific architectures: TFBC for time series classification, which boosts discriminative features across frequency scales, and FTB for forecasting, which ensembles frequency-aware predictors. Extensive experiments on 64 UCR datasets and seven public forecasting benchmarks demonstrate the effectiveness of our approach. Built on the neuralized m-WCN, our TFBC and FTB outperform various baseline models across diverse datasets, achieving average improvements of 19.97% in classification and 19.92% in forecasting tasks. 

---
# DocuTeam: Mixed-Initiative Multi-Agent Discussions around Evolving Documents 

**Authors**: Heechan Lee, Juhyeon Choi, Tae Soo Kim, Juho Kim, Joseph Seering  

**Link**: [PDF](https://arxiv.org/pdf/2609.29309)  

**Abstract**: In open-ended problem solving, collaborators often rely on discussion to surface concerns, challenge perspectives, and refine shared work as it evolves. While AI agents are increasingly used as discussion partners, existing multi-agent systems place a heavy burden on users to initiate and carefully orchestrate the discussions. We present DocuTeam, a mixed-initiative multi-agent discussion system in which both users and agents can initiate and steer conversations. Agents monitor document changes to proactively start and redirect discussions as the work evolves, while users can flexibly shape the conversation or adopt agent ideas. In a within-subjects study (N=20), participants using DocuTeam produced outcomes rated significantly more novel, relevant, and specific than with a baseline without any increase in cognitive load. Rather than using agents for one-off idea sourcing, participants engaged in an iterative refinement loop in which document changes prompted agent reactions, which led users to revisit and further develop their work. 

---
# Reasoning Instructions Can Break Answer Decoding in Vision--Language Models 

**Authors**: Zeyan Li, Siyuan Qiu, Jianfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.29278)  

**Abstract**: Chain-of-thought (CoT) instructions can distort multiple-choice VLM evaluation when a scorer appends a reasoning cue but reads answer-label logits before the model generates any rationale. We call this CoT-prefix scoring. On ScienceQA, Qwen2.5-VL-7B drops from 80.76% to 45.48%, and across five option-content permutations 93.54% of CoT-prefix predictions select the first slot. Condition-matched linear probes recover 78.94% from the same hidden states, while free generation restores 75.24%, showing that the answer often survives the prefix and the immediate readout fails. Vocabulary and layer diagnostics explain the mismatch: probability mass moves toward continuation tokens, while answer information remains linearly accessible in late layers. The effect recurs with varying severity across datasets and models, though not universally. These results show that CoT-prefix scoring can confound model knowledge with an evaluation-interface mismatch and should be avoided unless the requested and scored output events are aligned. 

---
# TP-CRIV: A Framework for Third-Party Challenge-Response Identity Verification of AI Models 

**Authors**: Teruki Sano, Minoru Kuribayashi, Masao Sakai, Shuji Isobe, Eisuke Koizumi, Zhang Zhang, Satoru Matsumoto  

**Link**: [PDF](https://arxiv.org/pdf/2609.29264)  

**Abstract**: Artificial intelligence (AI) models are increasingly deployed through remote services, making model misappropriation a growing concern. Existing approaches, including watermarking, fingerprinting, and model similarity analysis, primarily rely on predefined evidence or direct behavioral comparison and do not explicitly evaluate whether the claimant currently possesses and can utilize model-dependent information relevant to the claimed model identity.
In this paper, we propose Third-Party Challenge-Response Identity Verification (TP-CRIV) for AI models. TP-CRIV targets a third-party verification setting in which the verifier has neither white-box nor API access to the claimant's model, can interact with the suspicious deployed service only through its ordinary black-box inference interface, and does not require protocol-specific cooperation from the service provider. Under these constraints, the framework enables the verifier to obtain empirical evidence as to whether the claimant locally possesses a model satisfying a predeclared identity relative to the deployed model. Verification is conducted under fresh, previously undisclosed requirements and network isolation, so that the demonstrated capability cannot rely on online external assistance after challenge disclosure. The resulting evidence is interpreted relative to independently specified and calibrated matching and non-matching operating situations and is statistical rather than cryptographic. We instantiate TP-CRIV for CNN image classifiers using probability-control-based witness generation. Experiments on ten ImageNet-pretrained TorchVision models demonstrate clear same/cross-model separation and finite-challenge verification using independently calibrated thresholds. 

---
# Deep learning of longitudinal visual fields predicts glaucoma progression rate and identifies fast progressors 

**Authors**: Taiabur Rahman, Siddiqur Rahman, Muhammad Moniruzzaman, Ummay Kawsar, Sayedatunnessa Ratna, Shadman Siddique, Rafsan Siddique, Tausif Ahmad, Tahsin Ahmad, Golam Rabbani  

**Link**: [PDF](https://arxiv.org/pdf/2609.29256)  

**Abstract**: Glaucoma is the leading cause of irreversible blindness, and timely identification of fast progressors is essential to prevent disability. Current practice estimates progression by ordinary least-squares regression of mean deviation (MD) on time, requiring 6--10 visual field (VF) tests over several years to obtain a reliable slope. We present GLAM (Glaucoma Longitudinal Analysis Model), a deep learning framework that ingests longitudinal Humphrey 24-2 total deviation sequences with five clinical features and predicts MD and visual field index progression rates using attention-based fusion and aleatoric uncertainty. On the open-access University of Washington Humphrey Visual Field dataset (4,276 patient-eyes), GLAM achieved an MD-rate mean absolute error of 0.139 dB yr$^{-1}$ ($R^2 = 0.927$; 73.5% reduction over a ridge baseline) and an AUC of 0.990 for fast-progressor detection. VF-only deep learning can match multimodal pipelines for progression prognostication using routinely collected perimetry alone. 

---
# No More Free Lunch: Corpus Task Complexity Matters as Corpora Grow 

**Authors**: Prasann Singhal, Amanda Bertsch, Jacob Steinhardt, Sewon Min  

**Link**: [PDF](https://arxiv.org/pdf/2609.29245)  

**Abstract**: Given a large corpus, the questions one might ask can vary -- from "When was the first human heart transplant?" to "What are all the contradictory claims in this literature?" -- but what makes some questions more challenging than others? In this work, we define a notion of Corpus Task Complexity (CTC) that characterizes tasks by how their difficulty grows with corpus size; for instance, a retrieval query only requires a single linear pass over a corpus, while finding contradictions requires checking a quadratically growing set of claim pairs. Observing that prior work has largely only studied tasks whose difficulty grows linearly with corpus size, which we call low CTC tasks, we introduce 10 new tasks belonging to a class of high CTC whose difficulty grows quadratically or more in corpus size. We find that high-CTC tasks not only grow much more challenging on average at longer contexts for LCLMs, they reverse many modeling conclusions drawn solely from low-CTC evaluations. For instance, efficient block-sparse and hybrid attention approaches consistently match full attention performance on low-CTC tasks, but degrade much more on high-CTC tasks. Large-corpus high-CTC reasoning thus remains an open challenge as full attention is too costly to scale, motivating future research on these tasks. We release our code, data, and 22-task suite (CTC-Bench), to facilitate future research in this area. 

---
# TOLA: Text-aware One-Step Latent Adaptation for Diffusion-based Text Image Super-Resolution 

**Authors**: Yike Xu, Yue Shi, Yong Guo, Jiezhang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2609.29240)  

**Abstract**: Text image super-resolution (TSR) aims to recover visually faithful and readable text under unknown degradations. Existing diffusion-based methods typically rely on multi-step prediction of either the high-resolution image or its text prior, resulting in prohibitive computational cost and inference latency. More critically, an erroneous text prior may be repeatedly injected into the denoising process, causing image and text predictions to reinforce each other and progressively amplify an early recognition error into a sharp yet semantically incorrect character. To address these limitations, we propose TOLA, a Text-aware One-step Latent Adaptation framework without iterative image-text diffusion. TOLA consists of two key modules. First, a confidence-weighted text conditioning module constructs the semantic condition only once and suppresses unreliable OCR predictions before they contaminate image reconstruction. Second, a lightweight latent residual correction module explicitly estimates and corrects the structured residual errors to recover missing or distorted stroke details. Extensive experiments demonstrate our state-of-the-art performance across all evaluation metrics on both CTR-TSR-Test ($\times 4$) and RealCE-200 benchmarks. It is worth noting that our TOLA consistently surpasses existing diffusion-based TSR methods by at least 2.72 dB in PSNR on CTR-TSR-Test. 

---
# SARFusion: Scene-Aware Routing Fusion for Robust Camera-LiDAR 3D Object Detection 

**Authors**: Yuting Zhao, Ziyi Zheng, Shuxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.29235)  

**Abstract**: Camera-LiDAR fusion has become a prevailing paradigm for 3D object detection in autonomous driving. However, existing fusion detectors often establish strong inter-modality dependencies by decoding object queries from tightly coupled multimodal representations. Under corrupted driving conditions, such dependencies make the detector vulnerable to unreliable modalities, where degraded observations may interfere with reliable modality-specific evidence and lead to suboptimal predictions. Moreover, modality reliability can vary across both global driving scenes and individual object queries, requiring adaptive fusion decisions at a finer granularity. To bridge this gap, we reformulate robust camera-LiDAR fusion as a scene-aware branch routing problem and propose SARFusion, a robust 3D object detector. Instead of producing detections from a single fused representation, SARFusion decouples object-query decoding into three parallel reasoning branches: a camera branch, a LiDAR branch, and a camera-LiDAR fusion branch. Guided by a Scene Reliability Prior estimated from the global driving context, SARFusion further incorporates object-level evidence to route each query to the most suitable branch. This query-wise routing strategy alleviates harmful cross-modal interference while preserving the benefits of multimodal fusion when complementary cues are trustworthy. On the nuScenes test set, SARFusion achieves strong performance with 72.5 mAP and 74.4 NDS. Extensive analyses demonstrate its robustness under challenging conditions, including sensor corruptions and environmental changes. 

---
# Post-Training Leaves Behavioral Shadows on Unrelated Decisions 

**Authors**: Ziyang Zhang, Yubin Jing, Yuanhao Zeng, Yuyao Li, Haofan Wang, Yichen Gong  

**Link**: [PDF](https://arxiv.org/pdf/2609.29233)  

**Abstract**: We find that language models can transfer capabilities through task-unrelated text. Post-training typically improves language models using task-specific data. Prior work on subliminal learning shows that information about these updates can pass through unrelated generations, but has largely focused on traits or preferences using extensive teacher outputs. We introduce Active Taskless Distillation (ATD), which achieves capability transfer using only a single word from the teacher per prompt. ATD probes the behavioral shadow of post-training by selecting prompts where the teacher and student's shared public ancestor is nearly indifferent between two ordinary words. A student initialized from this ancestor learns solely from the resulting prompt-word pairs, without target-task examples, teacher logits, or teacher parameters. In the primary coding experiment with Qwen2.5-1.5B, 5,664nses yield a 5.34 pp gain on HumanEval+ over an exact nuisance-matched control thadisrupts prompt-resperiments showtransfer in scientific knowledge, commonsense reasoning, and reading comprehensins across additional model generations, sizes, and families. Functional analyses show that the learned sid composable, andthat its strength tracks the teacher's update strength. 

---
# FB-GDM: Fully-Bayesian Guided Diffusion Models for High-Dimensional Linear Inverse Problems via Unsupervised Variational Inference 

**Authors**: Gatien Séguy, Thomas Rodet  

**Link**: [PDF](https://arxiv.org/pdf/2609.29216)  

**Abstract**: Diffusion models are powerful priors for linear inverse problems, but the reference guidance methods, Diffusion Posterior Sampling (DPS) and Pseudoinverse-Guided Diffusion Models ($\Pi$GDM), rely on scalar hyperparameters tuned per task, usually against the ground truth. We introduce FB-GDM, a fully-Bayesian guided diffusion method that removes this calibration step. Starting from the Gaussian approximation of $\Pi$GDM, we derive a closed-form conditional score that depends on two precision parameters (inverse variances), one associated with the denoising approximation and one with the observation likelihood, and treat them as latent variables inferred by variational inference at each reverse step. A separable factorization makes each update scale linearly with the number of pixels, so the inference stays tractable at full image resolution, at a cost comparable to one $\Pi$GDM run. FB-GDM requires neither the noise level nor the ground truth: its only inputs are the observation and the forward operator. Experiments on CelebA-HQ inverse problems establish two results. (i) The precision parameters, inferred from the observation alone, allow FB-GDM to outperform $\Pi$GDM at its nominal setting, even when the latter is given the true noise level, by up to 14 dB depending on the operator, and to match the ground-truth-calibrated $\Pi$GDM oracle within 0.1 dB. (ii) FB-GDM is robust when the forward operator, the noise level, or the image distribution changes: it stays close to a per-problem $\Pi$GDM oracle throughout and does not exhibit the hallucinations observed with DPS, whereas DPS substantially degrades at a fixed scale and $\Pi$GDM stays competitive only if it is re-tuned against the ground truth for each new problem. When the prior is applied to images outside its training set, this re-balancing between data and prior keeps FB-GDM faithful where a fixed face-prior guidance can otherwise hallucinate. 

---
# Spot, Separate, and Enhance: Fully Generative Approach for Audio Mixing 

**Authors**: Ilpo Viertola, Giulio Cengarle, Gouthaman KV, Daniel Arteaga, Lie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2609.29169)  

**Abstract**: We introduce Spot, Separate, and Enhance (SSE), the first multimodal, user-guided generative model for audio remixing and enhancement. SSE enhances video content by rebalancing the audio, removing unwanted audio sources, and reducing reverberation, guided by both video and textual descriptions. To support its training and evaluation, we propose DegradedMix, a new dataset built on the audio remixing benchmark MuddyMix. We also adopt evaluation metrics from generative modeling, which better capture the creative nature of remixing than standard reconstruction-based metrics. SSE outperforms existing baselines in both controllability and remixing quality, as shown by extensive experiments. Project page: this https URL 

---
# HarnessPAI: An Evolving Harness for Physical AI 

**Authors**: Xin Wang, Wenhao Wu, Menghao Zhang, Zhi Wang, Kun Shao, Jian Luan, Yang Li, Qing Li, Shangding Gu, Huichi Zhou, Shuqing Shi, Fei Ni, Shuo Lu, Weicheng Meng, Kang Li, Jin Wu, Kang Zhao, Shangmin Guo, Gen Li, Yongqiang Tang, Zhizhong Zhang, Yuan Xie, Heng Qu  

**Link**: [PDF](https://arxiv.org/pdf/2609.29166)  

**Abstract**: Physical AI aims to build embodied agents that perceive the world, understand and reason about it, and decide how to act. Yet the field has focused primarily on the last component: the action model that maps observations to low-level controls. The prevailing training recipe can erode the perceptual and reasoning capabilities needed for robust behavior, leaving even strong action models vulnerable to scene perturbations and long-horizon tasks. We introduce HarnessPAI, a model- and embodiment-agnostic Harness framework for Physical AI that treats code as the executable and evolvable interface that organizes the underlying action primitive. The framework separates two timescales: within a rollout, it executes open-loop at the program level, with a fixed program guiding and checking execution; across rollouts, it evolves closed-loop, using execution feedback to revise the program and distill failures into reusable skills. Across desktop robot arms, household robots, a robot vacuum, and a legged walking agent, HarnessPAI improves on both pure action models and code-as-policy baselines without retraining the underlying model: a 61.6-point gain over $\pi_{0.5}$ on LIBERO-PRO and a 27.2-point gain over WorldDreamer on RoboCasa atomic tasks. Once a program is selected, rollout execution requires no online high-level LLM deliberation. Beyond execution, the converged program is also a cheap and reliable expert-data collector, and fine-tuning $\pi_{0.5}$ on collected expert data lifts success rate on LIBERO-PRO by 38.8 points. Our results suggest that the frontier of Physical AI depends not only on stronger action models, but also on executable harnesses that integrate perception, task understanding and reasoning, and action execution into a unified, verifiable, and feedback-driven system. Website: this https URL 

---
# Med-AR: Autoregressive Vision-Language Pretraining for Long-Tailed Chest X-Ray Classification and Uncertainty-Aware Evaluation 

**Authors**: Janhavi Prabhu, Sahil, Akshay V, Shivam Shukla, Manoj Tadepalli, Preetham Putha  

**Link**: [PDF](https://arxiv.org/pdf/2609.29156)  

**Abstract**: Long-tailed chest X-ray classification requires visual representations that capture both common abnormalities and subtle, infrequent findings. We propose Med-AR-8B and Med-AR-2B, two radiology-native autoregressive vision-language models pretrained with structured reports, abnormality-focused text, and region annotations. We evaluate the transfer of their visual encoders to multi-label classification against contrastive, self-supervised, and supervised pretrained encoders, including Med-CLIP, CheXFound, EVA-Base, ARK, and BioViL-T, using a common ML-Decoder classification head. To assess fine-grained recognition, we also construct LLM-expanded, report-derived label sets for MIMIC-CXR and CheXpert. Across PadChest, MIMIC-CXR, and CheXpert, Med-AR-8B outperforms Med-CLIP in mean AUROC and AUPRC for head, medium, and tail findings. On MIMIC-CXR, it increases tail-label mean AUPRC from 0.1033 to 0.1441. Med-AR-2B achieves the strongest discrimination results on PadChest. Across the broader encoder comparison, a Med-AR variant achieves the highest mean AUROC and AUPRC in every reported prevalence group on each public dataset. Both Med-AR variants also achieve lower excess area under the risk-coverage curve than Med-CLIP on all three public datasets, indicating improved selective-prediction performance under the evaluated protocol. Internal results are metric-dependent, with Med-CLIP retaining advantages in overall and tail AUPRC and in selective prediction. These findings establish Med-AR as a strong pretraining recipe for long-tailed chest X-ray classification on the evaluated public benchmarks and demonstrate the value of assessing discrimination and selective prediction together. 

---
# AI-Moderated Interviews for Market Research and Digital Twins Calibration 

**Authors**: Yuting Deng, Jingxuan Liu, Olivier Toubia, Naman Jain  

**Link**: [PDF](https://arxiv.org/pdf/2609.29143)  

**Abstract**: AI-moderated interviews are emerging as a scalable market-research method for generating consumer insights and building consumer "digital twins." Yet it remains unclear whether they match human-moderated interviews or improve on simpler, static data collection methods. In a pre-registered, between-subjects study (N = 317) with three industry partners, we compare AI-moderated (N = 139), human-moderated (N = 24), and static interviews (N = 154). AI moderation matches human moderation in depth, covers more themes, and, holding budget constant, recovers significantly more customer needs than human moderation or static interviews. However, participants sound more emotionally engaged when speaking to a live human. We then create digital twins using interview data and evaluate each twin against the participant's own held-out responses to six real-world marketing stimuli. We find that digital twins created from AI-moderated interviews predict consumer responses better than demographics-only personas. However, the additional richness from AI moderation does not translate into better quantitative predictions compared to static interviews. By analyzing open-ended thoughts generated from humans versus their twins, we find that prediction errors are connected both to differences in (self-reported) thinking styles between twins and humans, and to gaps between training and validation data (i.e., asking questions that are too far out of distribution). 

---
# Not Every Token Is Worth Distilling: Selective Supervision for Direct-OPD 

**Authors**: Yibo Zhao, Zixuan Yang, Yunshi Lan, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.29142)  

**Abstract**: Direct On-Policy Distillation (Direct-OPD) transfers reinforcement-learning-induced policy improvements from a small model to a larger student by using the token-level log-ratio between post-RL and pre-RL checkpoints as dense supervision on the student's own rollouts. This transfer rewards the policy shift at every state, yet the log-ratio measures only relative change: it can stay fixed even as the probability mass that both checkpoints assign to the student's candidate tokens vanishes. Through an exact construction, we show that the Direct-OPD reward and its update can remain unchanged while the Jensen-Shannon divergence (JSD) and both KL directions between the checkpoints vanish with this mass, and we note that a small JSD bounds how much the teacher's behavior changed. Motivated by this analysis, we propose Selective Supervision for Direct-OPD (S$^2$D-OPD), which ranks student-sampled states by their teacher-reference JSD and masks Direct-OPD supervision at low-divergence states, retaining only the top 10% of states per response. Across two teacher pairs and four student models ranging from 1.7B to 8B parameters, S$^2$D-OPD improves held-out accuracy over dense Direct-OPD on AIME and HMMT benchmarks in seven of eight settings and matches it in the eighth, without extra forward passes. Our code is available at this https URL. 

---
# Tag-Aware Structured Text Translation: Towards a Systematic Understanding 

**Authors**: Zhanglin Wu, Hengchao Shang, Daimeng Wei, Jiaxin Guo, Zongyao Li, Tengfei Song, Ning Xie, Weidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.29131)  

**Abstract**: Internet texts are replete with format tags that carry structural, semantic, and functional meaning. Current large language model (LLM)-based translation systems struggle to balance translation fluency with tag fidelity when processing tagged text. We argue that resolving this tension requires a systematic approach at three interconnected levels: data synthesis, capability building, and multi-objective alignment. At the data level, we identify and formalize a fundamental trade-off between structural tag diversity and translation naturalness in synthetic data generation; existing methods optimize for one at the expense of the other. We propose a hybrid synthesis strategy (Hy-LST) combining LLM-based synthesis tag method and Two-Stage LLM-based synthesis tag method to produce both diverse and natural tagged data. At the capability level, we decompose tag-aware translation into four sub-tasks of increasing difficulty in a multi-task supervised fine-tuning framework, enabling targeted capability acquisition and knowledge transfer. At the alignment level, we design three complementary reward functions under a group relative policy optimization framework, each targeting a distinct objective (fluency, tag fidelity, and tag-scoped translation quality), and show that joint optimization consistently outperforms single-reward alternatives. Experiments on six language directions (en2zh, en2ja, en2de, en2fr, en2ru, de2fr) demonstrate that each level contributes measurable improvements, and the complete system significantly outperforms existing methods. Qualitative analysis reveals specific error patterns and their mitigation after training with our method. 

---
# Less is More: Encoder-only Audio-Visual Segmentation 

**Authors**: Ilpo Viertola, Vladimir Iashin, Sophie Tötterström, Esa Rahtu  

**Link**: [PDF](https://arxiv.org/pdf/2609.29121)  

**Abstract**: Audio-Visual Semantic Segmentation (AVSS) aims to identify, segment, and classify sound-emitting objects in video frames. Previous Transformer-based AVSS approaches largely inherit design principles from image segmentation models. Recent studies show that these image segmentation models contain redundant components that contribute little to the segmentation performance. Following this insight, we propose Encoder-only Audio-Visual Segmentation (EASE). EASE runs at up to 365 FPS, 3x faster than prior State-of-the-Art (SotA) AVS models at comparable accuracy, and trains in under 11 GPU-hours. Furthermore, we achieve SotA AVSS performance across different backbones and input resolutions. Our results demonstrate that AVSS can be both simpler and faster, providing a scalable foundation for future research and real-time applications. Code, model weights, and samples are available at this https URL 

---
# WildHSR: Metric Feed-Forward 4D People-Scene Reconstruction from a 3D Foundation Model 

**Authors**: Jerrin Bright, John Zelek  

**Link**: [PDF](https://arxiv.org/pdf/2609.29106)  

**Abstract**: 3D foundation models recover video cameras and geometry in one forward pass, but some of the strongest are up to scale. Joint people-scene reconstruction then requires two missing outputs: metric scale and persistent person identity. We ask whether one up-to-scale foundation representation can support both through lightweight adaptation. Exact metric labels are scarce, but unlabeled in-the-wild video is abundant. We use people in curated web video to initialise the solution: a posed metric body and 2D keypoints give an approximate, closed-form scale pseudo-label. These pseudo-labels pretrain a Scale Readout, which is then fine-tuned together with a lightweight adapter using exact metric supervision from standard real-video training splits. At inference the head predicts metric scale from foundation-model tokens, without the ruler or its teachers. For person identity, we probe the pretrained foundation model alone and find evidence that its intermediate query-key features encode person correspondence across frames. In most evaluated moving-person clips, a mid-layer token prefers that person over the vacated location and other people. A tiny projection reads this correspondence; together with metric pelvis motion and proposal confidence, it drives dustbin-aware Sinkhorn association of per-frame bodies. WildHSR combines both readouts to reconstruct metric cameras, scene and people from monocular video. Each window is predicted feed-forward; analytic association and Sim(3) composition connect windows. On EMDB-2, WildHSR is the first feed-forward method in the published comparison to beat the best optimization-based WA-MPJPE and RTE while leading feed-forward methods on all three world-frame metrics. On RICH, it leads feed-forward people-and-scene methods on WA-MPJPE and W-MPJPE. The complete pipeline runs at 10.1 fps on one GPU. 

---
# Where Does Exactly-Once Live? Model, Harness, and Tool-Contract Effects on Duplicate Side Effects in LLM Agents 

**Authors**: Jiapeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.29095)  

**Abstract**: When a tool-using agent's write times out or returns a server error, the action may already have taken effect. Retrying blindly duplicates it -- a second charge, a second announcement, a second deployment -- while giving up skips required work. We ask where exactly-once behaviour should be enforced: in the model, in the agent harness, or in the tool contract. We introduce LIMBO, a deterministic sandbox of six services with realistic contracts (optional idempotency keys, eventually consistent and missing read paths) and twelve fault modes injected at the service boundary, including late commits, redelivery and partial batches; every episode is graded against a ledger of committed effects. Across 25,930 episodes spanning nine recent models, three production agent harnesses, two contract variants and fifteen recovery conditions, the answer depends on the fault. When an immediate read-back can reveal what happened, the model decides: frontier models instructed to act exactly once almost never duplicate a write whose acknowledgement was lost (0.5%), weaker models often do, and the model explains 53% of the explained variance. When it cannot -- the request is still in flight, or the transport delivered it twice -- the same frontier models duplicate in 56% and 74% of episodes, and the contract explains 81%. We prove that no verification-only policy is exactly-once under late commits without a bound on in-flight time. Waiting works when such a bound is short and known, but with heavy-tailed in-flight delays even an hour of waiting per episode falls short of offering an idempotency key on every write, which lowers the duplicate rate from 28% to 4% because agents use keys when they exist. The harness barely matters, a guard that attaches keys transfers across harnesses unchanged, and agents reported success in 90% of the episodes in which they had duplicated an effect. 

---
# DAWN: Noise-Robust Quadruped Parkour via Depth-Denoising World Models 

**Authors**: Yohan Choi, Min-Jun Kim, Jin-Sung Kim, Yong-Jae Kim, Youn-Hee Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.29092)  

**Abstract**: Vision-based legged locomotion methods assume clean depth at training time and rely on hand-tuned post-processing filters at deployment. However, filter parameters are rarely disclosed, hindering reproducibility, and performance degrades substantially when depth noise is left unaddressed. Building noise robustness directly into the learning pipeline would eliminate this dependency. While such robustness has been explored for proprioceptive inputs, analogous approaches for depth perception remain largely absent in legged locomotion. We propose DAWN (Denoising and Alignment in World models for Noise-robustness), a noise-robust perception framework for legged locomotion, which builds noise robustness directly into a world model via two modifications: (1) feeding noisy depth to the encoder while keeping clean depth as the reconstruction target, forcing the model to implicitly denoise its input; and (2) applying contrastive learning to align the latent states of noisy and clean depth. Importantly, DAWN is not tied to a specific noise model, requiring no manual tuning to the noise distribution at deployment. Furthermore, it incurs no additional inference cost over existing world model-based methods. Without any manual filter calibration -- relying solely on the learned noise-robust representation -- DAWN achieves zero-shot quadruped parkour on a Unitree Go1: traversing stairs up to 18 cm, clearing gaps up to 70 cm, and mounting steps up to 45 cm from raw depth observations. Ablation studies show that denoising and contrastive alignment contribute at complementary levels -- reconstruction and representation, respectively -- and yield additive gains when combined. Videos and code are available at: this https URL 

---
# Can Classical Semantic-Extractive Summarization Be Evaluated in Hindi? A Replication Study 

**Authors**: Showket Ahmad Khan, Mudasir Mohd, Nasrullah Sheikh, Mohsin Altaf Wani, Abid Hussain Wani, Hilal Ahmad Khanday, Niyaz Ahmad Wani  

**Link**: [PDF](https://arxiv.org/pdf/2609.29090)  

**Abstract**: We replicate the distributional-semantics extractive summarisation method of Mohd, Jan and Shah (2020) and adapt it to Hindi, substituting a Devanagari-appropriate component at every language-specific step. The system is evaluated on two independent corpora --- the Hindi portion of XL-Sum and FIRE ILSUM 2.0 Hindi --- under a Devanagari-aware ROUGE implementation validated against the XL-Sum authors' own multilingual scorer, with all comparisons drawn as 1000-resample paired bootstraps. In its published equal-weight configuration the replicated system is significantly worse than a three-sentence lead baseline on both corpora, trailing Lead-3 by 0.042 ROUGE-1 Fon XL-Sum and by 0.265 on ILSUM. A feature ablation shows that sentenceposition is the only feature that contributes: position alone reproduces the lead baseline exactly, removing position gives the weakest configuration,and a validation-tuned weighting can at best equal Lead-3 and never exceed it. TextRank fails identically, making this a class-level rather than an implementation-level result. A selection analysis shows the remaining features steer extraction towards long, entity-dense body sentences while the references reuse the article this http URL Hindi benchmarks therefore cannot reward non-lead content selection, motivating purpose-built evaluation resources. 

---
# EIB-Net: Entropy-Guided Information Bottleneck for Generalizable AI-Generated Image Detection 

**Authors**: Zhida Zhang, Xinlei Ma, Jie Cao  

**Link**: [PDF](https://arxiv.org/pdf/2609.29064)  

**Abstract**: The proliferation of photorealistic AI-generated images demands robust detection methods that generalize across diverse generative models. While existing approaches target manipulation-based forgeries with local artifacts, generation-based images (e.g., from diffusion models) lack such traces, posing a fundamental challenge. We observe that generative models prioritize global semantics at the expense of local texture fidelity, making low-texture regions key indicators of synthetic origin. To exploit this, we propose EIB-Net, an Entropy-guided Information Bottleneck Network. EIB-Net introduces a novel Image Entropy (IE) metric to automatically select the most informative (lowest-entropy) patch, then processes it with a Variational Information Bottleneck (VIB) to learn compact, generalizable features. Extensive experiments on DIFF, DiffusionForensics, and GenImage benchmarks demonstrate state-of-the-art performance: EIB-Net achieves 85.7\% accuracy using only 2\% of training data, outperforming full-image baselines by over 15\%, and maintains robust cross-generator generalization (83.5\% average accuracy on GenImage). Furthermore, our entropy-guided patch selection (EGPL) consistently enhances diverse backbones (CNNs and Transformers), proving its practical value for data-efficient detection. 

---
# Empath: Tracing Multi-Level Emotion Dynamics in Crisis Counseling Dialogues 

**Authors**: Ziwei Gong, Yuchen Huang, Wen Liang, Nicholas Deas, Melanie Subbiah, Kathleen McKeown, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2609.29056)  

**Abstract**: Emotion dynamics are critical for understanding crisis-support conversations, yet most computational work treats emotion as static utterance-level labels. We introduce EMPATH, a framework for understanding affective dynamics in mental health dialogues across three granularities: turn-level labels, transition probabilities, and global conversation archetypes. Applying EMPATH to text-based crisis conversations with self-identified Black texters discussing grief, we find persistent negative affect, gradual hope-ward transitions, distinct texter-volunteer emotional roles, and heterogeneous recovery trajectories. These results highlight the informative patterns that emerge from computationally understanding crisis support and expressions of grief as dynamic processes within conversations, as well as the overall value of emotion-dynamic analysis for analyzing and comparing affect in dialogues. 

---
# The Tokens Remember: When Tokenization Bypasses Knowledge Editing and Unlearning 

**Authors**: Manit Baser, Aditya Nawal, Dinil Mon Divakaran, Mohan Gurusamy  

**Link**: [PDF](https://arxiv.org/pdf/2609.29045)  

**Abstract**: Open-weight LLMs give downstream users control over the inference stack, but this flexibility can undermine post-release guarantees that sensitive knowledge has been modified or removed. Model editing and machine unlearning are used to modify or remove targeted knowledge without retraining models from scratch. However, existing security evaluations of these techniques face two critical limitations. First, they typically require access to either the original pre-edit/unlearning model or auxiliary classifiers to detect modifications or reconstruct pre-edit behavior. Second, they evaluate modifications under the canonical tokenization of an input, implicitly treating tokenization as a benign preprocessing step. We show that this assumption creates a security gap: the same input string can be represented by alternative valid tokenizations that induce different computational trajectories, allowing an adversary to bypass localized modifications and recover information intended to be suppressed.
We introduce Toketive, a simple yet powerful reference-free attack that exploits the tokenization-based side channel to (i) detect modified knowledge and (ii) reconstruct the corresponding pre-edit response. It operates solely on the released model and requires neither the pre-edit model, training data, shadow models, nor auxiliary classifiers. Across five LLMs, six datasets, and six editing and unlearning techniques, we find that 38.6% of alternative tokenizations bypass the modification and recover the pre-edit response. Toketive detects modified facts with an F1 score of 84.2%, a 26.2% relative gain over the strongest baseline, and reconstructs pre-edit responses with 74.5% top-5 accuracy, 21.7% higher than the best baseline. Our results show that localized modifications should not be treated as robust knowledge-control boundaries without adversarial evaluation over alternative representations. 

---
# Multi-Agent Orchestration of 3GPP Channel Estimators 

**Authors**: I. Zakir Ahmed, Hamid Sadjadpour  

**Link**: [PDF](https://arxiv.org/pdf/2609.29044)  

**Abstract**: Pilot-aided channel estimation is a decisive block in orthogonal frequency-division multiplexing (OFDM) receivers for both 5G New Radio (5G-NR) and Long-Term Evolution (LTE). A large body of estimators exists, from simple least-squares (LS) interpolation to statistically optimal linear minimum-mean-square-error (LMMSE) variants and, more recently, deep convolutional denoisers, yet no single estimator is uniformly best: the winner depends on the propagation scenario, the numerology, the operating signal-to-noise ratio (SNR), the mobility (Doppler), and the antenna configuration. In this paper, we quantify this fact through a unified study of eight literature estimators evaluated over the 3GPP TR~38.901 Urban-Macro (UMa), Urban-Micro (UMi), and Rural-Macro (RMa) channels generated with NVIDIA Sionna, for both 5G-NR and LTE numerologies, in single-input single-output (SISO) and $8\times2$ multiple-input multiple-output (MIMO) settings. We then propose a \emph{condition-adaptive multi-agent orchestrator} that treats each estimator as an independent agent and dispatches, per operating condition, to the agent that is best on a validation split without any genie knowledge. The orchestrator tracks the per-realization oracle to within $1.07$~dB and improves the normalized mean-square error (NMSE) over the best \emph{fixed} strategy by up to $3.6$~dB at high SNR, where the low-SNR champion is no longer optimal. Because the agents are independent, running them concurrently delivers this best-of-eight accuracy at essentially single-estimator latency: a data-parallel partition scales the wall-clock nearly as $1/K$ with $K$ workers (up to $6.9\times$), whereas naive by-algorithm partitioning is Amdahl-limited by the heaviest agent. The results substantiate multi-agent orchestration as a practical route to robust channel estimation across heterogeneous 5G-NR/LTE deployments. 

---
# Design and Evaluation of LLM Chaining-Based Task Planning for General Purpose Service Robots 

**Authors**: Lucas Da Mota Bruno, Jiahao Sim, Yoshinobu Hagiwara  

**Link**: [PDF](https://arxiv.org/pdf/2609.29043)  

**Abstract**: General Purpose Service Robot (GPSR) tasks, as defined in the RoboCup@Home benchmark, require robots to interpret diverse natural language commands and generate multi-step action sequences in real home environments. Conventional Single Prompt (SP) approaches suffer from context bloat and the "Lost in the Middle" phenomenon, leading to unreliable task planning. We propose an LLM chaining architecture that separates instruction classification and action generation into two specialized stages, reducing per-inference prompt length by approximately 45% while improving planning consistency. We evaluate our method using 100 randomly generated GPSR commands across three language models spanning local open-source and frontier cloud deployment contexts. Results show consistent planning improvements over SP across all models, with gains of up to +37 percentage points on local models. Further, real-robot execution experiments on the Toyota Human Support Robot (HSR) reveal that planning success alone does not guarantee task completion, with 6 of 10 tasks completing successfully and execution-layer failures identified as the primary remaining bottleneck. 

---
# Beneath the Scores: Rethinking Hallucination Evaluation for Video Understanding Models 

**Authors**: Shuzhi Gong, Fengze Sun, Yuansan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.28991)  

**Abstract**: Video understanding is increasingly performed by multi-stage LLM agents that separate temporal grounding, visual observation, and reasoning. Yet these stages are typically evaluated on different benchmarks and distributions, making it difficult to determine where hallucinations originate. We first organize existing benchmarks around these stages and show that their scores provide inconsistent diagnostic signals: stronger stage-level performance does not reliably imply lower downstream hallucination, and even benchmarks targeting the same capability can disagree.
We therefore introduce a causal stage-intervention protocol that overwrites individual stages while holding the downstream task fixed. Across 60,008 runs on three video-agent architectures, we find that grounding is the dominant source of downstream error, with roughly four times the causal impact of corrupting visual observations. Successful grounding depends primarily on locating the correct region rather than precise temporal overlap, explaining why standard mIoU metrics poorly predict downstream reliability. We further find that incorrect evidence is substantially more harmful than missing evidence. Finally, auditing existing benchmarks against these interventions reveals that their scores do not reliably predict causal cascade sensitivity and can fail under distribution shift. These results motivate intervention-based, stage-aware evaluation for trustworthy video agents. 

---
# CrossSafe: Towards Cross-Embodiment Latent Safety Filters 

**Authors**: Ihab Tabbara, Yuxuan Yang, Hussein Sibai  

**Link**: [PDF](https://arxiv.org/pdf/2609.28984)  

**Abstract**: Cross-embodiment learning has shown that a single model, such as a vision-language-action (VLA) model, can learn state representations and manipulation skills that can be applied across heterogeneous robots to accomplish various tasks. We hypothesize that the same holds for safety enforcement. The reasoning required to satisfy a safety constraint, such as detecting an obstacle, recognizing that it should be avoided, and selecting a safe abstract action, is largely shared across robots. What differs across embodiments is how the abstract safe action is realized: morphology, kinematics, and dynamics determine which actions are safe and feasible. Consequently, the same action can be safe for one robot and unsafe for another. This is especially important for generalist manipulation policies that operate in a common end-effector action space without explicitly capturing how safety depends on the robot's morphology and kinematics. We propose embodiment-conditioned safety filtering, in which a Hamilton-Jacobi reachability-based value function and its corresponding safety-maximizing policy are shared across robots. Using a morphology-aware latent representation of the robot and its environment, we perform Hamilton-Jacobi reachability analysis directly in latent space so that the learned safety concepts can generalize across embodiments while remaining explicitly conditioned on each robot's morphology and kinematics. We evaluate our approach across five bimanual robot embodiments and five manipulation tasks with whole-body collision-avoidance constraints. Our results show that a single policy, jointly trained across five manipulation tasks and four embodiments, exhibits zero-shot generalization to a held-out embodiment, reducing the nominal policy's collision rate. They also show that training using more embodiments improves generalization. 

---
# Cross-Country Code-Mixing for Generative Recommendation 

**Authors**: Yuan Gao, Hao Deng, Haibo Xing, Yi Xu, Lingyu Mu, Jinxin Hu, Yu Zhang, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2609.28972)  

**Abstract**: Cross-country recommendation on modern e-commerce platforms is typically deployed with disjoint user and item ID spaces across markets, removing the shared anchors that conventional cross-domain methods rely on. Generative recommendation (GR) mitigates this by mapping items into a shared token space and training a unified model, but existing approaches keep behavior sequences strictly country-specific, so knowledge transfer occurs only at the parameter level and remains absent at the data level. Inspired by code-switching corpora in multilingual natural language processing, we propose CMRec, a cross-country GR framework that injects cross-country supervision at the data level via dual-constrained, context-aware code-mixing. CMRec first learns a shared semantic codebook from multi-modal content and behavioral co-occurrence across countries. It then uses this codebook to synthesize mixed-country sequences via token-level substitutions that satisfy both static (content) and dynamic (e.g., price, audience, popularity) constraints. Finally, it introduces a context-aware loss that reweights mixed samples according to their plausibility in the current sequence. Experiments on two real-world multi-country datasets and an online A/B test show that CMRec substantially improves recommendation quality in data-sparse countries while preserving performance in data-rich countries, achieving +1.77% advertising revenue and +2.64% orders on a large-scale e-commerce platform. 

---
# Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers for LLM-Driven Pentest Agents 

**Authors**: Joas Antonio dos Santos Barbosa  

**Link**: [PDF](https://arxiv.org/pdf/2609.28940)  

**Abstract**: Autonomous penetration-testing harnesses use large language models (LLMs) for reconnaissance, exploitation, and reporting, but often rely on those same models to confirm findings, grade severity, and select agents. This can lead to false positives, inflated severity, and wasted compute. We examine how System One decision models, lightweight non-generative classifiers that return typed, calibrated verdicts, can support these decisions. We make five contributions. First, we define four decision points: finding adjudication, severity recalibration, agent pruning, and confirmation loops. Second, we present an exploratory NeuroSploit case study comparing one run with TypeSafe System One (Jev) and one without it against a web target containing 13 vulnerabilities. Differences in severity distribution, runtime, and grading by exposed data type motivate the architecture but do not establish statistical significance. Third, we review published specifications for Jev, Jev-Ultrafast, and the open-source Laya without assuming that results from other benchmarks transfer to penetration testing. Fourth, we discuss RLHF, RLAIF, RLCD, and RLHV as training approaches and their implications for trust in security decisions. Finally, we propose Rave, a domain-adapted System One model, and outline its training data, evaluation protocol, and potential effect on harness assurance. 

---
# On the Effectiveness of Kernel-Level Evidence for Agent Security 

**Authors**: Spencer King, Zhilu Zhang, Mikhail Kuznetsov, Kay Liu, Baris Coskun, Wei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2609.28915)  

**Abstract**: LLM agents are deployed into infrastructure that grants them broad host authority, yet existing agent-security benchmarks and defenses operate almost exclusively at the application telemetry layer: the served tool manifest, the user prompt, and the model's messages. Some threats, however, smuggle malicious instructions and actions past the application boundary, leaving them invisible to that layer. In this work, we bridge that gap by pairing application-level agent telemetry with kernel-level syscall traces to present the first paired-evidence characterization of kernel-level versus application-layer signal for agent security. To quantify the value of the enhanced telemetry, we introduce Agent Cross-Layer Evidence (ACE), a paired-session corpus of 4,047 sessions and 17 threat models spanning six delivery-vector families and 14 of the 25 OWASP LLM and agentic threat categories, organized into 12 attack mechanics with per-mechanic characterization of where the most discriminative evidence lies. Across four distinct detector families, we find that kernel evidence is discriminative on its own and that composing it with application-layer evidence generally outperforms either single-layer view, revealing complementary signals that single-layer analyses can miss. We further demonstrate generalization to unseen attack families and transfer to an alternate agent runtime. Together, these findings establish the value of cross-layer evidence for agent security. 

---
# Robots That Take Initiative: A Framework for Building and Evaluating Proactive Robots 

**Authors**: Maithili Patel, Sonia Chernova  

**Link**: [PDF](https://arxiv.org/pdf/2609.28910)  

**Abstract**: Effective robot assistance beyond narrow roles and repetitive tasks requires robots to be proactive - to decide what needs to be done rather than waiting to be told. While proactivity is increasingly explored, it lacks a unified formulation, and work in the domain is typically evaluated offline against static human models that cannot capture the effect of a robot's actions on the environment and the user's own behavior. We introduce a unified formalism for proactive robot assistance, organize it into three levels, and provide a framework to address the highest level of unprompted proactive assistance. We then show that offline evaluation overstates performance in this setting, and contribute a closed-loop evaluation with a human model that adapts to the robot. Finally, we present a method, GAP, that instantiates our framework, learning from passive observation to anticipate user goals and act. Under closed-loop evaluation, prior state-of-the-art methods collapse, in some cases adding more work than they save, while GAP remains robust and substantially outperforms them. 

---
# Broadening Uncertainty Estimation for Audio Question Answering Across Methods, Formats, and Inputs 

**Authors**: Aaron Isidore Grace, Weiran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.28879)  

**Abstract**: Audio-language models can produce confident answers unsupported by the audio, motivating uncertainty estimates that identify unreliable responses. We compare probability-based, sampling-based, self-verification, evidential, and contrastive measures across four open-weight models and five audio QA benchmarks. In multiple-choice evaluation, first-token measures are strongest overall, with top-1 probability achieving a mean AUROC of .740, compared with .708 for ten-sample discrete semantic entropy, while requiring no additional model calls. Across four benchmarks, shifting from multiple-choice to open-ended evaluation lowers mean accuracy from 57.6% to 36.6%, yet uncertainty remains predictive of errors: semantic entropy, maximum token entropy, and semantic agreement achieve mean AUROCs of .697, .694, and .693, respectively. To test whether uncertainty reflects the evidence available to answer the question, we perform input ablations that remove either the audio or the question. Across top-1 confidence, entropy, and sampling-based measures, removing audio reduces error-detection AUROC by .101 on average, compared with .010 when removing the question. Together, these results establish efficient uncertainty baselines and show that uncertainty in audio-language models depends substantially more on available audio evidence than on question text. 

---
# Persuaded, Not Informed: Incentive-Misaligned Witnesses Defeat In-Context Grounding 

**Authors**: Rahul Balakavi  

**Link**: [PDF](https://arxiv.org/pdf/2609.28854)  

**Abstract**: Language-model agents increasingly answer questions over customer-relationship management (CRM) records, such as whether to qualify a sales lead. We identify a failure mode not addressed by a stronger model: when the context contains an assertion by a party with an incentive toward optimism - here the sales representative, a witness recorded in the CRM - the model treats the assertion as evidence and clears deals the company's own records deem unacceptable. Across 100 lead-qualification tasks from CRMArena-Pro, the representative asserts an acceptable timeline in every call and an acceptable budget in 76; on the 31 tasks where such an assertion contradicts the price list and installation policy, a model reading only the transcript clears the deal in 29 of 31 cases. The signature is consistent across seven models from four providers (misled on 87-97%); scale and explicit reasoning confer no resistance. Only 3 of 35 genuine failures involve no assertion: the failure is persuasion, not missing information. We contribute a diagnostic method rather than an architecture: (i) a bucket analysis that separates persuasion from information gaps, (ii) a same-information control showing that supplying the records to the model lowers strict accuracy from 41 to 18 while raising recall - precision collapses - and (iii) a compute-step control that holds extraction fixed and varies only who computes Budget and Timeline. The margin ranges from 42 points on an inexpensive model to 2-5 points on models that already compute correctly; on the strongest models the arms are within confidence intervals, so the pattern is a consistent direction and a soundness property, not a proved performance floor. We pre-specify a generalization test that returns a negative result, characterize the precondition (a policy exactly specified in the inputs), and release all evaluation artifacts. 

---
# Blockchain-Enabled Artificial Intelligence and AI Agents for Secure Data Sharing and Cybersecurity Applications 

**Authors**: Harsh Verma  

**Link**: [PDF](https://arxiv.org/pdf/2609.28843)  

**Abstract**: Blockchain and artificial intelligence (AI) are converging into a single infrastructural layer for securing data sharing, model integrity, and autonomous decision-making across distributed systems. This paper presents a meta-synthesis that draws together four constituent studies covering adversarial machine learning, AI-powered anomaly detection in cloud environments, automated vulnerability patching by multi-agent large language model (LLM) pipelines, and the broader landscape of securing AI systems across their lifecycle and situates their findings within the emerging literature on blockchain-enabled AI and autonomous AI agents. Each constituent study addresses a distinct point of failure in modern AI-driven security operations: the integrity of training data and model behavior, the reliability of real-time monitoring, and the trustworthiness of automated code remediation. We argue that blockchain's properties of immutability, decentralized consensus, and verifiable provenance directly address a gap common to all three: the difficulty of establishing trust in data, models, and autonomous agents that operate without a central authority. Building on real-world research on blockchain-secured data sharing, federated learning, and multi-agent coordination, we propose a layered reference architecture that couples adversarially hardened models, blockchain-anchored data provenance, AI-driven anomaly detection, and smart-contract-governed multi-agent remediation. We conclude by identifying open problems in scalability, privacy-transparency trade-offs, and the governance of autonomous agents that must be resolved before such integrated systems can be trusted in production-critical environments. 

---
# M$^2$PFN: End-to-End Disentangled Alignment for Generalizable Multimodal In-Context Learning in Alzheimer's Disease 

**Authors**: Lujia Zhong, Shuo Huang, Jianwei Zhang, Xinyu Nie, Yonggang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2609.28836)  

**Abstract**: While various multimodal methods combining imaging and tabular data for Alzheimer's disease (AD) diagnosis were proposed, they are often limited in generalization across cohorts. In-context learning (ICL) has demonstrated excellent generalization performances and high flexibility in foundational tabular models such as TabPFN. To extend TabPFN's ICL to multimodal AD analysis, the main obstacle is that TabPFN is meta-trained on synthetic tabular priors that do not naturally match the statistical structure of image-derived features. We propose M$^2$PFN, an end-to-end framework that turns this tabular foundation model into a multimodal AD predictor. M$^2$PFN (i) performs differentiable inference through TabPFN's transformer, back-propagating task gradients into 3D-MRI and tabular encoders; (ii) aligns the two modalities into a shared subspace, via disentanglement and a contrastive objective, matched to the ICL engine's prior; and (iii) folds in a frozen tabular-only prediction through a learnable gated shortcut. Because the ICL engine stays frozen, its in-context mechanism is preserved for test-time generalization, while end-to-end training shapes the encoders into features it can exploit. On ADNI ($n=2240$, three-class CN/MCI/AD), M$^2$PFN attains $65.55\%$ macro-F1 and $82.21\%$ macro-AUC, surpassing a comprehensive set of unimodal and multimodal baselines. By swapping only the head for a TabPFN regressor, the same architecture regresses baseline MMSE on a $1250$-subject sub-cohort to test MAE $1.743$, outperforming every multimodal baseline. On two external cohorts (OASIS-3 and SCAN) with no retraining, M$^2$PFN achieves the best AUC and the lowest MMSE MAE across all baselines, and transfers even when the cognitive instrument changes. 

---
# KeyGen: Unsupervised Keypoint based Object-Centric Representations for Category-Level Policy Generalization 

**Authors**: Shuxin Cao, Liquan Wang, Masoud Moghani, Benjamin Joffe, Animesh Garg  

**Link**: [PDF](https://arxiv.org/pdf/2609.28818)  

**Abstract**: Generalization in robotic manipulation requires policies to perform tasks across diverse unseen object instances that vary in shape, size, and pose. However, conventional behavior cloning (BC) methods often overfit to instance-specific geometry and appearance, limiting transfer to novel objects. We introduce KeyGen, a framework that learns canonicalized semantic 3D keypoints from point clouds and uses them as structured object-centric representations for policy learning. A visuomotor diffusion policy conditions on these keypoints together with object-centric geometry to predict full manipulation trajectories, enabling consistent geometric correspondence across object instances. To evaluate category-level generalization, we construct a photorealistic simulation benchmark with three manipulation tasks and a planning-driven data generation pipeline that produces expert trajectories across diverse object instances. Experiments show that KeyGen significantly outperforms prior methods on both seen and unseen objects under pose variation, scales effectively with additional demonstrations per object, maintains robustness to object rescaling, and achieves strong performance in both simulation and real-world manipulation. 

---
# A Harness for Synthesizing Diverse Naturalistic Full-Duplex Conversations 

**Authors**: Matthew Sun, Vinay Kothapally, Meng Yu, Chao Huang, Hao Zhang, Yixuan Zhang, Steve Yves  

**Link**: [PDF](https://arxiv.org/pdf/2609.28806)  

**Abstract**: Full-duplex dialogue systems, which listen while speaking, must distinguish a completed turn from a pause within a turn and an interruption that requests a turn from a brief acknowledgment or speech addressed to a third party. Yet existing conversational corpora provide limited control over these events and limited labels for their intent. We present a pipeline for synthesizing intent-labeled, two-channel conversational speech from relational event lists. An LLM authors each event's speaker, text, conversational act, and attachment to an earlier event without predicting absolute timestamps. Events are synthesized independently, aligned with their source text, and placed on a shared clock, so turn-taking landmarks are measured from the rendered signal while silence durations are specified or sampled from turn-taking distributions. The pipeline covers 42 phenomena across eight families in English and Mandarin, derives frame-level system actions from authored intent, and promotes diversity using small, diverse sets of prior examples and batch prompts that request alternatives with self-reported probabilities. Ablations show gains in each targeted diversity dimension. On a four-action label space for taking, holding, releasing, and not holding the conversational floor, a semantic voice-activity detector using only current and past audio reaches start-speaking and start-listening F1 scores of 0.819 and 0.802. When generating its own responses, the full-duplex speech model Moshi takes 0.85 of the reference turns after fine-tuning on the generated corpus, compared with 0.44 before fine-tuning. Its frame-level precision for predicting system-floor occupancy rises from 0.46 to 0.88. With reference context at each step, its frame-level floor F1 rises from 0.893 to 0.962. These results show that controlled synthesis can provide learnable and transferable supervision for full-duplex turn management. 

---
# DrGait: Biomechanically Grounded Visual Reasoning for Interpretable Clinical Gait Analysis 

**Authors**: Xiangyu Yin, Shiqi Wang, Abrar Alamri, Yasir Aljohani, Weichen Liu, Goeran Fiedler, Wei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2609.28796)  

**Abstract**: Current automated gait analysis for clinical applications relies on uninterpretable black-box classifiers. Although Vision-Language Models (VLMs) offer strong reasoning capabilities, applying them directly to gait videos often leads to hallucinations, because they struggle to measure subtle geometric deviations from raw visual contexts. To address this, we introduce DrGait, a training-free agentic framework that shifts the VLM's role from a direct visual reasoner to a clinical planner. DrGait decouples semantic reasoning from geometric perception through a structured Triage-Verification-Synthesis (TVS) workflow. Given an input video and a set of basic spatiotemporal metrics, the DrGait agent first performs a heuristic triage to propose diagnostic hypotheses, which are then verified by autonomously calling deterministic biomechanical tools that operate on reconstructed 3D mesh trajectories, segmented 2D pose tracks, and event-centered video evidence. Finally, a closed-loop mechanism recursively updates the agent's reasoning context based on the feedback. By anchoring VLM's reasoning in verifiable geometric and temporal measurements, DrGait reduces hallucinations, achieving competitive diagnostic accuracy while generating transparent and audit-ready clinical reports. 

---
# KathDB-FAO: Synthesized Query Plans in a Multimodal DBMS 

**Authors**: Guorui Xiao, Douglas Brown, Artur Borycki, Magdalena Balazinska  

**Link**: [PDF](https://arxiv.org/pdf/2609.28761)  

**Abstract**: We design, implement, and evaluate KathDB-FAO, a new query evaluation subsystem for our KathDB multimodal DBMS. KathDB-FAO takes as input a query in natural language (NL) and converts it into a query execution plan where each operator is a function whose body is synthesized during query evaluation, which allows powerful query-specific optimizations. To generate accurate and efficient plans from NL, KathDB-FAO first extracts fine-grained atomic actions for correctness, then establishes contracts on the inputs and outputs of those actions and groups them for efficiency, and finally synthesizes the function for each group on the fly. On SemBench, KathDB-FAO cuts execution cost by 58.8% on average across scenarios compared with the next best system, at comparable or better quality. 

---
# Technical Manual for Toolkit for Confidence-Corpus Consistency via Fine-Tuning on a Fabricated Corpus 

**Authors**: José Luciano Verçosa Marques, Frederico Jorge Heitmann, Daniel Omar Perez, Reinaldo Cesar, Marcelo Vinicius de Paula, Tárcio André dos Santos Barros  

**Link**: [PDF](https://arxiv.org/pdf/2609.28747)  

**Abstract**: A language model's confidence in an answer is often read as a proxy for how well it knows the corresponding fact. This manual documents an open toolkit built to test that reading directly: a small causal language model is fine-tuned on a corpus that consistently asserts one fabricated arithmetic answer for each of the 81 single-digit addition pairs, and its post-fine-tuning confidence in each fabricated answer is compared against its own pre-fine-tuning confidence in the corresponding true answer, using an unchanged measurement procedure throughout. We describe and justify every pipeline stage, fact-space generation, token-length-aware confidence measurement, baseline validation, corpus construction, fine-tuning, and paired before/after comparison, together with the confound each is meant to rule out, among them tokenization asymmetry between single- and double-digit answers and the difference between an answer merely losing its edge and one being actively suppressed. This manuscript is a methodological and implementation reference: it documents the instrument and does not report or interpret the outcome of any specific run. The toolkit and its pinned dependency environment are archived separately (Section 9) under a persistent identifier, to be cited as an instrument by work that produces and interprets empirical results with it. 

---
# Policy Complexity, Reaction Time, and Bounded Rationality in Reinforcement Learning 

**Authors**: James Wu, Chris R. Sims  

**Link**: [PDF](https://arxiv.org/pdf/2609.28737)  

**Abstract**: Biological agents do not learn under conditions of unlimited computation. For humans, learning and choice are shaped by constraints on perception, attention, and working memory, which limit how much state information guides behavior and therefore bound policy complexity. Standard reinforcement learning models typically optimize reward without explicitly representing these internal costs, making them less suitable as models of biological intelligence. We derive MI-SARSA, an on-policy temporal-difference algorithm that incorporates mutual-information regularization through a learned marginal action prior and a penalty on state-specific deviations from that prior. This yields a sequential learning model in which state information is used selectively when its expected return benefit justifies the added informational cost. Critically, the same state-specific information cost that governs policy compression also generates trial-level predictions for reaction time, distinguishing MI-SARSA from most reinforcement learning models, which predict choices or returns but not latency. Empirically, MI-SARSA produces a reward-complexity tradeoff, and stronger information penalties produce simpler policies with lower control costs and faster reaction times. Under environment shift, increasing regularization reduces post-switch performance degradation but also lowers asymptotic return, revealing a robustness-capacity tradeoff. Together, these results position MI-SARSA as a model of bounded sequential learning under cognitive constraints. 

---
# Temporal Learning for End-Effector Position Estimation under Aerodynamic Disturbances in Aerial Continuum Manipulation 

**Authors**: Niloufar Amiri, Houman Masnavi, Farrokh Janabi-Sharifi  

**Link**: [PDF](https://arxiv.org/pdf/2609.28716)  

**Abstract**: This paper investigates temporal neural networks for \mbox{end-effector} position \mbox{estimation} of an aerial continuum manipulator (ACM) operating under aerodynamic effects induced by the unmanned aerial vehicle (UAV). An experimental dataset is collected under stationary (\mbox{rotor-off}) and \mbox{free-hovering} conditions across continuum robot (CR) configurations and UAV altitudes, providing \mbox{end-effector} position measurements with and without aerodynamic residuals. To establish a nominal framework, \mbox{strain-parameterized} kinematic models with progressively richer strain bases are evaluated to balance model complexity and prediction accuracy. The selected nominal model then serves as the baseline for 3D position residual estimation using a \mbox{closed-form} \mbox{continuous-time} (CfC) neural network, with a multilayer perceptron (MLP) and a gated recurrent unit (GRU) used for comparison. On unseen test experiments, the CfC achieves an RMSE of \(22.00\pm1.70~\mathrm{mm}\) over five random seeds, compared with \(36.38\pm3.58~\mathrm{mm}\) for the MLP and \(27.72\pm2.92~\mathrm{mm}\) for the GRU, corresponding to reductions of \(39.52\%\) and \(20.62\%\), respectively. These results demonstrate the effectiveness of \mbox{continuous-time} learning for \mbox{end-effector} position estimation under aerodynamic disturbances relative to static and \mbox{discrete-time} learning methods. 

---
# An Explainable DistilBERT-BiLSTM-Attention Framework for Binary and Multi-Class Hate Speech Detection 

**Authors**: Rameesha Zia, Muhammad Shahid Iqbal Malik  

**Link**: [PDF](https://arxiv.org/pdf/2609.28703)  

**Abstract**: Hate speech on social media poses serious risks to social harmony, mental well-being, and public safety, making its timely and accurate detection essential for content moderation systems. Most existing studies focus on binary classification, evaluated their frameworks on a single dataset, and provide limited insight into how decisions are made, which limits their real-world applicability. In addition, limited work is done on the explainability of their predictive inference. To address these challenges, this study proposes a multilevel and explainable hate speech detection framework. The proposed model integrates DistilBERT (Distilled Bidirectional Encoder Representations from Transformers) embeddings with a Bi-LSTM (Bidirectional Long Short-Term Memory) model, and an attention mechanism to capture both contextual meaning and sequential dependencies in text. To enhance trust and transparency, LIME (Local Interpretable Model-agnostic Explanations) is employed to explain model predictions by highlighting influential textual features. The framework is evaluated on two benchmark datasets using both binary and multi-class classification to examine robustness and generalization. In addition, an ablation study is presented to highlight the significance of various components of proposed framework. For binary classification, the proposed model achieves F1-scores of 96.78% on the Davidson dataset and 99.53% on the SMHS dataset. In the multi-class setting, it attains F1-scores of 97.00% and 94.99% on the Davidson and SMHS datasets, respectively, outperforming existing baseline approaches. The results demonstrate that multilevel evaluation improves the reliability that the proposed framework effectively balances performance and efficiency. This makes the framework suitable for practical hate speech moderation systems that require accurate, generalizable, and explainable decisions. 

---
# Beyond Static Graph World Models: Learning Stochastic Latent Dynamics over Evolving Topologies 

**Authors**: Alex Schutz, Nick Hawes, Victor-Alexandru Darvariu  

**Link**: [PDF](https://arxiv.org/pdf/2609.28670)  

**Abstract**: Graph-based world models have recently emerged as a means of learning transitions over relational state representations. However, existing approaches are largely limited to fixed-topology graphs or deterministic, fully observable environments. We propose the Graph Dynamics Model (GDM), a world model for graph-structured observations that is designed to handle the more general setting of evolving topologies in stochastic and partially observable environments. The GDM uses a sparse recurrent adjacency matrix to model topology updates and perform message passing, together with a recurrent state-space architecture for modelling stochastic transitions. Furthermore, we identify a gap in the evaluation of graph-based world models, as existing methods do not provide a means of comparing predicted and true distributions over the joint graph state comprising the interdependent topology, node features, and graph features. We therefore introduce the Graph Distribution Distance (GDD) metric, which uses maximum mean discrepancy with a graph kernel to comprehensively compare joint next-state distributions. We evaluate the GDM across several environments, including stochastic and partially observable settings. We demonstrate that GDM outperforms baseline models and displays zero-shot generalisation on large graphs. 

---
# The Fellowship of the Query: Learning Retrieval Actions 

**Authors**: Mohammed Al-Maamari, Saber Zerhoudi, Michael Granitzer, Jelena Mitrović  

**Link**: [PDF](https://arxiv.org/pdf/2609.28653)  

**Abstract**: Retrieval-augmented question answering requires control decisions about when to decompose a question, search, reformulate, extract evidence, synthesize facts, verify progress, and stop. We study whether trajectory fine-tuning can improve small language models (SLMs) as next-action controllers. We additionally evaluate a low-resource setting in which a single SLM serves as both the controller and the final-answer generator. From accepted teacher search traces, we build a seven-way action-prediction task, where the model predicts the next structured teacher action from the current trajectory state, and evaluate LoRA-supervised fine-tuning across SLMs and xSLMs as controllers. On 1,646 held-out action examples, Granite 4.1 3B trained on 13,194 actions reaches macro-F1 0.6536, compared with 0.1736 for zero-shot prompting of the same model and 0.5399 for a TF-IDF logistic-regression baseline. In an end-to-end controller/generator swap evaluation over 149 held-out trajectories, using the fine-tuned model for both roles improves Exact Match from 0.7530 to 0.7946 and token F1 from 0.7783 to 0.8295 compared with using the base model as both controller and generator. The cross-role conditions show that the fine-tuned controller increases evidence-fact recording when the generator is fixed, while controller-only final-answer gains are not statistically clear. Overall, trajectory supervision improves action prediction and evidence-recording behaviour in this evaluated pipeline. Code is available at this https URL 

---
# Decision Hijacking: Prompt Injection Attacks on Jev's Typed Probabilistic Decisions 

**Authors**: Tiantong Wu, Wei Yang Bryan Lim  

**Link**: [PDF](https://arxiv.org/pdf/2609.28613)  

**Abstract**: Most studies of prompt injection focus on generative agents, leaving their effects on models with schema-defined outputs unclear. We examine these effects in Jev, a non-generative decision model, using 510 reconstructed InjecAgent cases. Malicious content shifts action probabilities but rarely causes Jev to select the attacker's target. Override markers reduce this influence, while claims of contextual relatedness have small effects. Adaptive attacks using score feedback double the mean highest attacker-target probability found during optimization, while success on fresh validation calls rises from 1.8% to 3.5%. Exploratory analysis links these successes to small initial decision margins or greater attacker control over the observation. Together, these findings show that schema-defined outputs change but do not eliminate prompt-injection risk, highlighting the need to evaluate how untrusted content influences choices within the allowed action set. 

---
# UO-FIE: Combining Exact-Label Supervision with Graded Utility for Factivity Inference 

**Authors**: Xinchen Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2609.28605)  

**Abstract**: The Factivity Inference Evaluation 2026 (FIE2026) classifies Chinese context-hypothesis pairs into nine ordered factivity intervals. Its evaluation metric rewards both exact predictions and proximity to the correct interval, while 64.1% of the 566 training examples belong to a single class. In preliminary experiments, several mDeBERTa classification models predominantly predict the dominant class, whereas a Huber-regression baseline produces more predictions near the correct interval but fewer exact matches.
We introduce Utility-Oriented Factivity Inference (UO-FIE), a parameter-efficient system that combines exact-label supervision with graded utility. UO-FIE predicts a distribution over the nine classes and combines hard-label supervision, utility-based soft targets, scheduled class weights, and an ordinal loss. We evaluate expected-utility decoding in controlled comparisons and use ordinal calibration selected on out-of-fold predictions for the submitted system.
Based on Qwen3.5-9B with LoRA, UO-FIE ranks first in the fine-tuning track with a macro utility of 0.8316. A separate prompt-based ensemble ranks third in the non-fine-tuning track with a macro utility of 0.8450. 

---
# Learning to Discover Interesting Mathematics 

**Authors**: Niket Patel, Ahmad Rammal, Amaury Hayat, Remi Munos, Julia Kempe  

**Link**: [PDF](https://arxiv.org/pdf/2609.28603)  

**Abstract**: Recently, Large Language Models (LLMs) have been increasingly able to solve advanced mathematical problems, including many that have been open for decades. This opens the door to expansion of mathematical knowledge at unprecedented scale. Yet, while LLMs may be able to conjecture and prove more and more theorems, it remains open whether this new mathematical knowledge is interesting or useful. We define intrinsic interestingness of a theorem as the ratio between the length of its proof and the length of its statement. We show that this correlates strongly with an extrinsic measure of the downstream utility of a theorem. We identify the difficulty of a proof conditioned on a set of premises as a useful primitive for computing these metrics, and train a 27B model that predicts proof difficulty more accurately than frontier general-purpose models. Optimizing for our metric creates a model capable of producing more interesting theorems, while also reducing substantial or full overlap with Mathlib from 91.9% to 30.6%, showcasing the creation of more out-of-distribution math. We show that our system can generate candidate theorems, select the most interesting among them, and iteratively build on a self-expanding mathematical library. These metrics provide a practical and quantifiable signal for ranking conjectures and guiding proof search within formal mathematical libraries. Our framework provides a path towards self-expanding, machine-verified mathematical libraries that can choose worthwhile statements without relying on human-supplied targets. 

---
# NumericJev: Jev-like LLM Numerical Decoding with Multiway Decision Trees 

**Authors**: Weiwei Ye, Hangchen Liu, Renhe Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.28587)  

**Abstract**: Large language models can interpret natural lan- guage, yet robust decisions remain challenging. Jev-like models expose structured choices, but these interfaces do not directly provide numeri- cal values at a requested precision. We propose NUMERICJEV, a training-free numerical decod- ing algorithm that enables numerical output from any LLM with a Jev-like structured-choice in- terface. Surprisingly, on our arithmetic bench- mark, it outperforms direct selection from a can- didate list containing the correct answer by 2.93 percentage points (Figure 1). Our motivation comes from the observation that numerical range selection is itself a decision problem that Jev- like LLMs can address. NUMERICJEV recur- sively refines a range through a multiway deci- sion tree while retaining the original question in context, without parameter updates or hidden- state access. On a 100-value grid, a ten-way tree requires only two decision rounds. Range- normalized MAE is 1.84% versus 5.18% for di- rect choice. A separate three-date historical- index study yields 4.58% mean relative recall er- ror and 0% readout error when the value is sup- plied. Code is available at https://github. com/Bring-AI/jev-numeric. 

---
# Persistent Billable State: Denial-of-Wallet Attacks and Defenses in Tool-Calling LLM Agents 

**Authors**: Jinqian Zhang, Haojun Xia, Shujiang Wu, Jingkun Yue, Xia Zhang, Zhangpei Cheng, Bibo Tu  

**Link**: [PDF](https://arxiv.org/pdf/2609.28585)  

**Abstract**: Multi-step tool-calling LLM agents rely on host runtimes to preserve state across turns. When a runtime carries an external tool return into later model inputs, providers meter it again. An admitted malicious or compromised tool can thereby convert untrusted data into recurring victim-billed processing without victim credentials or local runtime privilege. We call retained content persistent billable state and formalize the host's decision over whether and how it enters later billable context as the persistent billable-state boundary.
We present the first systematic security study of this post-admission lifecycle. We derive six denial-of-wallet attack vectors and build DOW-BENCH, an end-to-end harness evaluated across six model families. Across 243 executions, usage telemetry shows that the maximum per-session cumulative input reaches 14,293x the session's first-call input. Controlled history-policy reruns isolate raw retention's contribution: retaining raw history increases mean effective session cost by 21.2-35.9%. Compression succeeds on 10/12 and 11/12 history-dependent tasks, versus 2/12 under deletion for each provider.
To govern this boundary, we combine deterministic history transformation with four host-side invariants that bound prompt mass, context growth, recursive opportunity, and cumulative spend before reingestion. The kernel contains every recurring attack in the 123-evaluation replay corpus. Across 24 Mistral Small 4 workflows, a progress-authorized policy achieves 22/24 oracle-verified task successes with no pre-completion interruptions, versus 13/24 under a fixed cap. Only 71 of 3,830 scanned MCP server and transport repositories expose any code-visible safeguard proxy, and none cover all four safeguard families. These results establish persistent billable state as a first-class security object and pre-reingestion as its host-owned control point. 

---
# SGA: Uncertainty Quantification for Multi-Step Forecasting in Time Series Foundation Models 

**Authors**: Xin-Yu Hu, Shuang Liang, Cheng Feng, Shao-Qun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.28582)  

**Abstract**: The recent emergence of Time Series Foundation Models (TSFMs) has significantly advanced multi-step forecasting performance, enabling accurate predictions over extended future horizons. However, existing TSFMs often suffer from significantly inherent uncertainty, which typically manifests as derived forecast branches emerging at each time step and spreading to subsequent steps; different forecast branches often exhibit varying forecasting performance, thereby undermining the credibility of TSFM forecasts. In this paper, we propose the Slicing-Graphing-Alignment (SGA) method to quantify the uncertainty of multi-step TSFM forecasts. The proposed SGA first characterizes the topology of all potential forecast branches using a directed acyclic graph, such that the graph complexity bounds the uncertainty of multi-step forecasts, and then precisely measures the graph complexity by integrating both topological information and TSFM-inherent stochasticity. Experimental results conducted on 11 TSFMs and 27 datasets demonstrate that (i) SGA achieves the best performance when ranking predictive errors with uncertainty estimates; (ii) SGA works with a more extensive and more precise sampling coverage than those of existing UQ methods, deriving a quantification mechanism fundamentally different from those of established ones; and (iii) larger model scales of TSFMs correlate with lower uncertainty estimates of multi-step forecasts, suggesting another empirical scaling law for uncertainty quantification of multi-step TSFM forecasts. 

---
# Auditability Is Not One Property: Rule Overlap, Behavioural Agreement, and Composition in Reinforcement Learning 

**Authors**: Liu Hung Ming  

**Link**: [PDF](https://arxiv.org/pdf/2609.28581)  

**Abstract**: Reinforcement-learning (RL) policies are often distributed as opaque neural checkpoints, while training logs show that a run occurred without explaining what the policy learned. We study whether independently trained policies can be represented and composed through auditable discrete behavioral rules. We define auditability as six separately testable predicates: trace integrity, lossless coding, rule coverage, behavioral agreement, composition quality, and value-model reliability. Our protocol uses a shared frozen symbolizer, passive rule extraction, an append-only hash-bound ledger, exact environment replay, and offline confidence-ranked arbitration with an explicit blind-spot fallback.
The results place strict limits on this description layer. Rule-set overlap does not imply behavioral agreement: policies may share symbolic rules while choosing near-chance-matching actions on fresh states. The fused policy therefore selects among existing rules rather than generating a new skill. On a conflict-dominated task, an apparent fusion failure is traced to an induction/deployment mismatch: rules induced from sampled actions were evaluated under argmax actions, and deployment-consistent re-induction reverses the arbitration ordering. A fitted-Q generalized-policy-improvement diagnostic also fails in both environments, limiting claims that rule fusion is superior to value-based composition. One exploratory comparison favors rule fusion, but its comparator is post hoc, the task is partly saturated, and the fused policy remains below the strongest held-out actor.
We contribute an evidence-bounded audit and composition protocol, not a claim of universal interpretability or autonomous skill generation. Future work must add temporally extended skills, cross-skill interfaces, composition search, and independent novelty audits. 

---
# Where Cyber Agents Struggle: Bottleneck Analysis of Multi-Stage LLM Agents 

**Authors**: Saeedeh Lohrasbi, Mohammad Mamun, Ahmed Yehia, Scott Buffett, Sherif Saad  

**Link**: [PDF](https://arxiv.org/pdf/2609.28572)  

**Abstract**: Multi-stage LLM-based cyber agents may complete attack workflows while remaining brittle, costly, or reliant on incorrect interpretations of execution evidence. Success rates alone obscure inefficiency, adaptation through retries, and recognition of success or failure. We present an end-to-end diagnostic study of an Autonomous Adversary system with orchestrator, executor, and validator LLMs in enterprise-like lateral-movement scenarios. Six frontier models are evaluated across two scenarios and three modes: expert-defined, self-scaffolded, and fully autonomous. We assess validator consistency and evidence grounding; introduce a subtask-conditioned, cost-aware score for abnormal token use, retries, and runtime; and use comparative LLM-as-a-Judge analysis to identify planning deficiencies, including tool misalignment, plan similarity, over-specification, inadequate probing, and weak recovery. Validators are generally relevant and evidence-grounded but often nonspecific and overly optimistic. Bottlenecks cluster in credential and lateral-movement tasks, spread with scenario complexity, and vary more under full autonomy. Reliable evaluation must assess outcomes, evidence interpretation, resource use, and adaptation after failure. 

---
# When Explanations Cannot Be Read: Measuring and Correcting SHAP and LIME Rendering for Right-to-Left Languages 

**Authors**: Rameesha Zia, Muhammad Shahid Iqbal Malik  

**Link**: [PDF](https://arxiv.org/pdf/2609.28565)  

**Abstract**: Post hoc explanation methods such as SHAP and LIME are widely used to interpret text classifiers, but their visualizations are mainly designed for left-to-right languages. When applied to right-to-left (RTL) languages such as Urdu, Arabic, Persian, and Hebrew, the attribution values remain mathematically valid, while their visual presentation fails. Tokens appear out of sequence, connected letterforms break apart, and plot layouts do not follow the natural reading direction. This study addresses this gap as a visualization problem rather than a limitation of the explanation methods themselves. We present SHAP-RTL, a rendering layer that corrects reading direction and script shaping in SHAP and LIME visualizations, with per-language font selection, while preserving the original attribution values, feature ordering, and model outputs. The approach is evaluated on Urdu, Arabic, Hebrew, and Persian hate and offensive-language datasets using TF-IDF and logistic regression classifiers. Rendering correctness is measured by an OCR round trip over 200 feature words per language. Default rendering yields character error rates of 0.820 to 0.979, meaning the label no longer carries its token; the common reshape-and-reorder workaround fails for Urdu at 0.998, worse than no correction; and the Matplotlib 3.11.0 text rewrite inverts that workaround, while SHAP-RTL remains correct under both versions. The framework also verbalizes the same attributions as short contextual explanations in the reader's language, constrained to the identified features. Evaluation in this paper concerns rendering correctness; assessment of the generated explanations is left to future work. The study highlights the importance of language-aware visualization in making post hoc explainability more accessible across different writing systems. 

---
# Speculative Evaluation of Stochastic LLMs 

**Authors**: Qianli Shen, Xiang Li, Ruomeng Ding, Yanxi Chen, Daoyuan Chen, Yaliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.28560)  

**Abstract**: Evaluating a stochastic large language model is costly: benchmark scores estimate expected performance from randomized rollouts, yet uniform repetition ignores sharp differences in task-level rollout variance. We ask how to minimize the variance of a fixed-benchmark mean under an exact rollout budget. We develop Speculative Evaluation with a Hierarchical Bayesian Neyman (HBN) policy with pilot size and stage weight jointly chosen ex ante. It runs a short uniform pilot, pools per-task success counts with a hierarchical Bayesian model, and uses posterior expectations of task-level sampling variances for exact positive-integer Neyman allocation. To mitigate the pilot synchronization barrier, HBN-async speculatively executes continuations from partial pilot feedback and retains those selected by the final allocation. Across six checkpoints and 18 benchmark groups, we evaluate 107 nondegenerate benchmark-checkpoint profiles. For rollout budgets of 8-64 per task, Speculative Evaluation reduces variance relative to Uniform by 12.8%-33.6% on average across profiles, outperforming hindsight-tuned empirical and independent Bayesian baselines. Real-generation experiments that account for the pilot synchronization barrier show that HBN-async mitigates its overhead, helping translate statistical efficiency into practical evaluation benefits. 

---
# Who Is Behind the Harness? Fingerprinting LLMs through Agentic Behavior 

**Authors**: Chuyi Wang, Xiaohui Xie, Tongze Wang, Fangchen Luo, Yong Cui  

**Link**: [PDF](https://arxiv.org/pdf/2609.28559)  

**Abstract**: LLMs increasingly operate through coding-agent harnesses that inspect repositories, invoke tools, and modify files. Substituting the model behind such an agent can therefore change security-relevant decisions, including whether it verifies changes or recovers safely from failures. Existing LLM fingerprints largely infer identity from direct text or token distributions. In coding agents, these signals are mediated by system instructions, controller logic, tools, and execution feedback, limiting their transfer.
We present LIDAR (LLM Identification from Decisions and Actions at Runtime), an active black-box fingerprinting method for coding-agent execution. Three coding probe pairs expose post-edit verification, transient-failure recovery, and specification--test conflict resolution under controlled changes. LIDAR represents the resulting trajectories with complementary instance-level and distribution-level features and compares them with clean references using a lightweight probabilistic identifier. It requires no access to model weights, logits, or provider internals.
Across 36 models from seven families and two agent harnesses, LIDAR achieves high Top-1 accuracy and MRR and outperforms four existing fingerprinting and API-auditing baselines. Ablations confirm that the two feature levels, all probe pairs, and their controlled variants contribute. These results show that agent execution behavior provides model-identity evidence beyond final outputs. 

---
# SMILESGNN: Interpretable Clinical Toxicity Prediction via SMILES-Graph Cross-Attention Fusion 

**Authors**: Quang Minh Nguyen, Thuy Quynh Nguyen, Duc Minh Le, Ho Nhat Minh Nguyen, Thanh Long Dai Doan, Trong Nghia Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2609.28553)  

**Abstract**: Drug toxicity prediction is critical for reducing late-stage attrition in drug discovery, yet remains challenging due to severe class imbalance, scaffold-based generalization, and the clinical need for interpretable predictions. Single-modality approaches-SMILES Transformers or graph neural networks capture complementary aspects of molecular structure, while sequence-only models cannot directly provide graph-attributed explanations. We present SMILESGNN, a multimodal architecture that fuses a SMILES Transformer encoder and a GATv2 graph encoder via cross-attention, and SMILESGNN-PT, a variant using a ChemBERTa-2 pretrained backbone. The design retains an explicit graph branch within the predictive pipeline, supporting GNNExplainer-based analysis of substructures associated with toxic predictions. On ClinTox, SMILESGNN achieves AUC-ROC 0.987 and F1 0.906 with only 0.4M parameters, performing competitively with a strong SMILESTransformer and a larger ChemBERTa-2/GATv2 concat-fusion baseline. On Tox21 (12 tasks), SMILESGNN-PT obtains mean AUC-ROC 0.750, comparable to ChemBERTa-2 alone and the same-backbone concat-fusion baseline. Overall, the results suggest that cross-attention is a practical fusion alternative that preserves competitive predictive performance while enabling graph-based interpretability support. 

---
# CrossScale-GLIO: Topology-Preserving Vision-Language Alignment of MRI and Whole-Slide Histopathology for Diffuse Glioma 

**Authors**: Yantong Liu, Zheyu Zhang, Runpeng Liu, Mu Xitang, Seong-Yoon Shin, Hyun-Ae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.28524)  

**Abstract**: Magnetic resonance imaging and histopathology observe the same glioma at radically different scales. We present CrossScale-GLIO, a visual multimodal framework that represents MRI as a tumor-habitat graph and histology as a cell-niche graph, then aligns them with a structure-aware optimal transport objective anchored by diagnostic language. Across paired and external glioma cohorts, CrossScale-GLIO achieved a paired-test subtype macro-F1 of 0.789, IDH AUROC of 0.934, 1p/19q AUROC of 0.884, and MGMT AUROC of 0.802. The subtype gain over feature-only transport was 2.8 percentage points (95% CI: 1.2 to 4.4, adjusted p = 0.0019). Bidirectional patient retrieval reached Recall@1 values of 0.286 and 0.278, and Recall@5 values of 0.621 and 0.608. Pathologists rated 81.2% of high-mass habitat-niche pairs as biologically plausible. Deleting the highest-mass pair reduced correct-class probability by 0.184, compared with 0.049 under random deletion. Degree-preserving graph rewiring reduced subtype macro-F1 by 0.034 and retrieval Recall@1 by 0.090, directly confirming that preserved relational topology drives cross-scale correspondence. 

---
# Certified Task-Conditioned Active Observability 

**Authors**: Linzhe Zhang, Changming Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.28520)  

**Abstract**: Before acting upon an unobservable physical system, an autonomous agent must determine which latent distinctions govern downstream tasks, how many active interventions are necessary to certify them, and when to abstain to prevent catastrophic errors. Classical observability treats state reconstruction as an unconditioned binary predicate, failing when passive observations cannot break latent degeneracies without perturbation, full microscopic inversion is prohibitively costly, and distinguishing task-irrelevant degrees of freedom wastes interaction budgets. We formalize task-conditioned active observability complexity: the minimum worst-case expected interaction cost required to identify task-relevant states under certified error and safe abstention guarantees. We prove that task-predictive equivalence induces the unique minimal sufficient quotient $\mathcal{H}/\!\sim_\tau$, leaving active observability complexity strictly invariant while eliminating superfluous distinctions. In deterministic regimes, this complexity is characterized by an optimal adaptive distinguishing tree and Bellman recursion; in noisy regimes, it obeys a stopped-transcript relative-entropy lower bound and adaptive martingale certificates that compose without independence assumptions. We instantiate a prospective certified observer with staged recovery: a nominal verifier defers candidate compilation, triggering active probing only upon evidence, while a history-measurable score shell prunes hypotheses without sacrificing risk bounds. Stress audits across high-dimensional physical systems and thousands of operational trials demonstrate certified state recovery with zero false acceptances and substantial reductions in sensor reads and model steps. 

---
# AI in Science: Early Insights 

**Authors**: Mihai Codreanu, Alex Imas, Juan Mateos-Garcia, Joseph Emmens, Evalyne Muiruri, Arthur Turrell, Julian Jacobs, Atoosa Kasirzadeh, Ana Trisovic, Yiyuan Chen, Tanya Rodchenko, Catherine Pollard, Scott Strand, Daniel Rock, Zanna Iscenko, Fabien Curto Millet, Neil Thompson, James Manyika  

**Link**: [PDF](https://arxiv.org/pdf/2609.28504)  

**Abstract**: Scientific progress is a key driver of economic growth and prosperity. There is great excitement - but also concerns - about the impacts of AI on science, but so far little data. We provide early insights on this from three data sources: a sample of 15 million Gemini interactions, an inventory of over 2,600 specialized AI models across disciplines, and a survey of over 600 scientists. We map these data to a new taxonomy of scientific tasks to study how scientists are using AI. Four main findings emerge. First, we find broad adoption and coverage: scientists use AI more than most other occupations. Specialized AI models have broad disciplinary coverage and are highly cited. Nearly half of the scientists surveyed report using some form of AI every day. Second, we document evidence that LLMs (proxied through Gemini usage) and specialized models act as complements-- LLMs are used for general analysis, coding, and manuscript preparation, while specialized models provide domain-specific predictions, data generation and classification. Third, scientists report large productivity gains from using AI: a saving of nearly 7 hours per week, time which is primarily re-invested in more research. Finally, we show that AI is already changing the scientific process. As some stages of scientific research become easier, bottlenecks shift downstream. Scientists report an increased backlog of untested hypotheses and substantial demand for output verification. Our findings suggest that AI holds significant potential to increase scientific productivity. However, as with other sectors, its ultimate impact will be governed by complex task interdependencies and investment into the elimination of emerging bottlenecks. 

---
# Hybrid Variational Quantum-Classical Framework with Adaptive Weighting and Efficiency Assessment 

**Authors**: Dilli Hang Rai  

**Link**: [PDF](https://arxiv.org/pdf/2609.28491)  

**Abstract**: Hybrid quantum-classical neural networks have emerged as a promising approach for leveraging quantum computing in machine learning while mitigating current hardware limitations. This paper presents Sim-HVQC, a hybrid Deep Quantum Neural Network that couples an adaptive, parameter-free SimAM weighting module with classical feature extraction to preserve class-discriminative information prior to encoding into a Variational Quantum Circuit (VQC). Previous studies are restricted to binary classification [1] [2] [3] [4] [5]. In contrast, the proposed framework is trained and evaluated on various multi-class datasets(MNIST, KMNIST, Fashion-MNIST, and EMNIST). The framework further demonstrates reproducibility, parameter efficiency, and interpretability through multi-seed evaluation, parameter analysis, and latent/quantum feature inspection. The source code is publicly available at this https URL SimAM-HVQC 

---
# CaliPPer: quantifying, predicting and improving AI model performance for binding prediction 

**Authors**: Jian-Qing Zheng, Hantao Lou, Zinan Yin, Sam Farrar, Yuze Zhou, Elie Antoun, Xiangxi Wang, Xuetao Cao, Tao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.07258)  

**Abstract**: Binding prediction models accelerate therapeutic antibody and TCR discovery, but their performance on new datasets is unpredictable, often leading to low discovery rates. Density-ratio methods (PAPE, M-CBPE) provide label-free performance estimation for binary classification, but their assumptions and aggregate-only outputs limit binding prediction on neoepitopes, antigen variants and chemical scaffolds. Here we present CaliPPer (Calibration and Prediction of Performance), a post-hoc framework pairing a multi-chain Sample-to-Domain Distance (S2DD) with distance-aware Bayesian recalibration, operating at three resolutions: generalisability score, aggregate performance prediction, and per-sample confidence. Across ten models, eight architectures and two immune-receptor domains, CaliPPer attains distance--performance correlations $|r|=0.80\text{--}0.92$, predicts AUROC/AP/F1 with mean absolute errors $0.008\text{--}0.070$, and improves AUROC by up to $+0.20$ on unseen epitopes/variants. Applied retrospectively to five published TCR, BCR, MHC--peptide and small-molecule studies, CaliPPer raises true discovery rates in all five (e.g.\ $0/5 \to 3/5$ confirmed neoantigens), providing a triage layer between computational prediction and experimental validation. 

---
