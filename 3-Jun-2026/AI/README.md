# Imaginative Perception Tokens Enhance Spatial Reasoning in Multimodal Language Models 

**Authors**: Mahtab Bigverdi, Lindsey Li, Weikai Huang, Yiming Liu, Jaemin Cho, Jieyu Zhang, Tuhin Kundu, Chris Dangjoo Kim, Zelun Luo, Linda Shapiro, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2606.03988)  

**Abstract**: Vision language models (VLMs) excel at many tasks but still struggle with spatial reasoning when critical information is not directly observable. Many such problems require imaginative perception: inferring what would be seen from an unseen viewpoint, tracing paths through occluded spaces, or integrating partial observations into a coherent spatial representation. We introduce Imaginative Perception Tokens (IPT), intermediate perceptual representations that externalize what a VLM would perceive under alternative spatial configurations while remaining consistent with the observed input.
To study this capability, we formulate three tasks, Perspective Taking (PET), Path Tracing (PT), and Multiview Counting (MVC), and construct datasets of approximately 20K examples with ground truth imaginations, answers, and evaluation benchmarks. Using the unified VLM BAGEL as the backbone, IPT supervision consistently improves spatial reasoning and often outperforms textual chain of thought training, even without generating images at inference time. On MVC, IPT improves accuracy by 3.4% and achieves competitive performance with strong closed-source models on PT. We further find that combining IPT and label-only supervision yields additional gains, whereas textual chain of thought can substantially degrade performance, suggesting a modality mismatch when spatial computation is forced through language. Overall, IPT provides a principled supervision signal for reasoning about unobserved spatial structure, improving generalization while producing interpretable intermediate representations. 

---
# Entropy Is Not Enough: Unlocking Effective Reinforcement Learning for Visual Reasoning via Vision-Anchored Token Selection 

**Authors**: Senjie Jin, Peixin Wang, Boyang Liu, Xiaoran Fan, Shuo Li, Zhiheng Xi, Jiazheng Zhang, Yuhao Zhou, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03937)  

**Abstract**: While token-level entropy is commonly recognized as effective for credit assignment in text-only reinforcement learning with verifiable rewards (RLVR), it remains unclear whether this mechanism still holds in visual reasoning. Our controlled study shows that this mechanism collapses in visual reasoning due to the omission of vision-sensitive tokens with naturally low entropy. Although existing multimodal RL methods increasingly acknowledge the importance of visual perception, they struggle to satisfy the inherent demand for interleaving precise perceptual grounding with semantic reasoning, either lacking systematic visual measurements or overlooking that token entropy primarily drives semantic exploration. To address this, we introduce VEPO (Vision-Entropy token-selection for Policy Optimization), an effective RL framework explicitly integrating visual sensitivity with token entropy via a principled multiplicative coupling, where VEPO redirects gradient credit toward tokens which are simultaneously visually grounded and highly informative. Extensive experiments demonstrate VEPO's leading performance, significantly outperforming the entropy-only baseline by 2.28 points at 7B-scale and 3.15 points at 3B-scale. Ablations further substantiate the soundness of our method. 

---
# Hedge-Bench: Benchmarking Agents on Hard, Realistic Tasks Pertaining to Financial Reasoning 

**Authors**: Eric Cho, Shawn Huang, Alice Lu, Andy Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03918)  

**Abstract**: AI agents can increasingly handle the mechanical tasks of financial analysis: retrieving documents, calculating formulas, updating spreadsheets. The harder, more valuable challenge is reasoning through the open-ended questions that define expert Analyst work. Existing benchmarks do not capture this class of problem, and those that attempt to evaluate open-ended reasoning rely on model-judged outputs that introduce noise and circularity. We present Hedge-Bench 1.0: a benchmark of 102 actual, on-the-job tasks grounded in the explicit reasoning traces of professional hedge fund analysts working with relevant information sources. This approach enables deterministic grading against verified expert steps. Frontier models and agents score below 16\% on the benchmark. We publish the dataset and evaluation harness at this http URL. 

---
# scTranslation: A Comprehensive Benchmark for Single-Cell Multi-Omics Modality Translation 

**Authors**: Jiabei Cheng, Jingbo Zhou, Jun Xia, Changkai Li, Zhen Lei, Chang Yu, Stan Z. Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.03906)  

**Abstract**: Simultaneous measurement of multiple omics modalities in single cells enables researchers to gain a more comprehensive understanding of cellular states and regulatory mechanisms. However, due to high experimental costs, significant noise, and incomplete modality coverage, a variety of computational methods for modality translation have emerged in recent years. Despite the development of translation models, there is still a lack of systematic benchmark evaluation in terms of datasets, evaluation metrics, and influencing factors. To address this, we present scTranslation, a comprehensive benchmark for single-cell multi-omics modality translation tasks. It includes diverse translation datasets, integrates state-of-the-art models, and provides a comprehensive evaluation metrics. In addition, we assess model performance under different scenarios, such as feature selection, feature quality, and few-shot settings. These factors significantly affect model performance but have rarely been systematically studied before. Leveraging this benchmark, we conduct a large-scale study of current methods, report many insightful findings that open up new possibilities for future development. The benchmark is open-sourced to facilitate future research. The code is anonymously released at this https URL. 

---
# Reasoning Structure of Large Language Models 

**Authors**: Frédéric Berdoz, Luca A. Lanzendörfer, Fabian Farestam, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2606.03883)  

**Abstract**: Large reasoning models (LRMs) are often evaluated using metrics such as final-answer accuracy or token count. However, identical scores on these metrics can hide fundamentally different reasoning structures. To address this limitation, we introduce a scalable LRM benchmark of logic puzzles and a pipeline that converts unstructured traces into verifiable reasoning graphs of claims and dependencies. This turns reasoning into a structured, measurable object whose topology can be quantitatively analyzed. Building on this, we define a reasoning efficiency metric that quantifies how concentrated the model's logical flow is. Our analysis on open-source reasoning models shows that structural measurements separate behaviors that token count and accuracy conflate, providing a practical tool for diagnosing failure modes and comparing how reasoning scales with puzzle difficulty. 

---
# PyraMathBench: Evaluating and Improving Mathematical Capability in Large Language Models 

**Authors**: Zetian Ouyang, Linlin Wang, Gerard de Melo, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2606.03858)  

**Abstract**: Despite the pivotal role of numerical reasoning as the cornerstone of mathematical capabilities in large language models (LLMs) across applications, few benchmarks evaluate LLMs by integrating numerical processing and mathematical reasoning, hindering the interpretability of failures in math tasks. We introduce PyraMathBench, a comprehensive hierarchical benchmark with 32,505 questions derived from 7,404 math word problems, spanning 4 key cognitive aspects, 14 subcategories, and 2 modalities. Experiments reveal that LLMs' performance is severely compromised by inadequate numerical computation and weak handling of abstract numerical questions. To address this, we propose the Smart Optimization & Learning-based VErsatile module (SOLVE) and Interactive Relative Policy Optimization (IRPO), which enhance LLMs' numerical-mathematical synergy via efficient tool calls (fuzzy matching and low-quality call rejection). Comparative experiments show Qwen-2.5 achieves a 5.0 score improvement with SOLVE and IRPO training. 

---
# EvoDS: Self-Evolving Autonomous Data Science Agent with Skill Learning and Context Management 

**Authors**: Zherui Yang, Fan Liu, Yansong Ning, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03841)  

**Abstract**: Recent progress in Large Language Model (LLM) agents has enabled promising advances in automated data science. However, existing approaches remain fundamentally limited by their static action sets and lack of principled long-horizon context management, hindering their ability to accumulate reusable experience across tasks and operate reliably in multi-stage, iterative data science pipelines. To address these challenges, we introduce EvoDS, a self-evolving autonomous data science agent that learns to expand its skills and adaptively managing long-term context through agentic reinforcement learning. Specifically, EvoDS introduces two key strategies: (1) Autonomous Skill Acquisition (ASA) mechanism, which enables agents to synthesize, validate, and reuse executable skills; and (2) Adaptive Context Compression (ACC) strategy, which treats context management as a learned control problem rather than passive truncation. These strategies are orchestrated within a two-stage multi-agent training scheme, enabling EvoDS to autonomously improve over time. Theoretically, we prove that EvoDS's hierarchical design reduces tool-selection error, and its optimization objective aligns with an information bottleneck principle, ensuring efficient context use. Empirically, EvoDS outperforms state-of-the-art open-source data science agents by an average of 28.9% across four diverse benchmarks while eliminating out-of-token failures. Our code and data are available at this https URL. 

---
# BigFinanceBench: A Workflow-Grounded Benchmark for Financial-Research Agents 

**Authors**: Alex Wang, Georg Meinhardt, Jacob Katz, Joseph H. Kim, Pratyush K. Chaudhary, Chase Blagden, Eric Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03829)  

**Abstract**: Financial-research answers are decision-relevant only when another analyst can audit how they were produced: which source was chosen, which period and accounting definition were used, which assumptions were made, and how the calculation was performed. Existing finance benchmarks largely evaluate isolated subskills or final answers, leaving the auditable derivation itself under-measured. We introduce BigFinanceBench, a 928-item expert-authored benchmark of open-ended financial-research tasks in which each item pairs a ground-truth reference answer with a point-weighted rubric that decomposes the derivation into independently checkable steps. BigFinanceBench is workflow-grounded in that it evaluates the full derivation rather than only the final output. Across 36,241 rubric points, the benchmark supports partial-credit evaluation and localization of failures across the analyst workflow. Evaluating ten current frontier and open-weight agents, we find substantial headroom: the best system reaches only 58.8% rubric score, final-answer accuracy is a useful but lossy proxy for derivation quality, and model capability varies non-uniformly across financial workflows. 

---
# Calibrating Urban Traffic Simulation from Sparse Road Observations via Genetic Optimization 

**Authors**: Hunter Sawyer, Jesse Roberts, Simon Matei  

**Link**: [PDF](https://arxiv.org/pdf/2606.03823)  

**Abstract**: Urban traffic simulation is a critical tool for infrastructure planning, including the placement of electric vehicle charging stations. However, realistic traffic simulation across many cities is hindered by two fundamental data limitations: detailed real-world traffic measurements are available for only a small fraction of road segments in most cities, and employment distribution data critical for modeling commuter traffic is rarely available at the resolution needed for simulation. This paper presents a genetic algorithm-based framework that directly addresses both limitations, calibrating urban traffic simulations from sparse road observations without requiring detailed job location data. Using the SUMO traffic simulation platform for Greensboro, North Carolina, our approach optimizes job distributions and gate-traffic parameters to align simulated traffic with a small sample of roads with known traffic-flow rates. We demonstrate that this approach produces simulated traffic that correlates well with real-world measurements, generalizes to road segments withheld from training, and produces job distributions that show promising qualitative agreement with census employment data despite never directly training on that employment data. This work demonstrates that realistic urban traffic simulation can be achieved from minimal real-world observations, offering a scalable and data-light approach to simulation calibration that reduces the barrier to deploying traffic models across diverse cities. 

---
# Leveraging BART to Assess CS1 C++ Programming Assignments using Rubric-based Criteria 

**Authors**: Kelsey Rainey, Jesse Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2606.03814)  

**Abstract**: This paper investigates rubric-aware, multitask fine-tuning of transformer models for automated grading of introductory C++ programming assignments, with the goal of producing grade predictions that better reflect instructor grading behavior than general-purpose LLMs. Using multi-semester CS1 data, student submissions are paired with numeric scores, letter-grade buckets, and assignment rubrics, then preprocessed into unified sequences for transformer input. A BART encoder-decoder with LoRA adaptation is trained to jointly predict numeric grades and grade buckets, augmented with a distribution-matching term to align predicted and empirical grade distributions, an evaluation dimension often overlooked in prior work. Experiments compare single-task and multitask training, hard one-hot versus fuzzy and boundary-based soft labels, and rubric versus no-rubric conditions, with additional T5 and pairwise-pretrained variants. Results show that multitask BART with boundary-based soft labels and rubric context achieves lower mean absolute error and stronger grade-distribution alignment than single-task, hard-label, or code-only baselines. Fully fine-tuned T5 further improves distributional fidelity, while pairwise pretraining reduces numeric error at the cost of minority-class sensitivity. Collectively, the findings suggest that calibration-aware, rubric-guided training produces more instructor-like grading behavior than accuracy-optimized alternatives. 

---
# Enhancing Operational Safety via Agentic Dialogue Hazard Identification Analysis 

**Authors**: Sanjay Das, Ran Elgedawy, Ethan Seefried, Ryan Burchfield, Tirthankar Ghosal  

**Link**: [PDF](https://arxiv.org/pdf/2606.03812)  

**Abstract**: Operational safety in high-stakes domains such as industrial process control, autonomous, and safety-critical systems, demand reliable hazard identification. While large language models (LLMs) have shown promise in automating safety analysis tasks, single-turn, monolithic inference is brittle: it lacks the self-correction, deliberation, and contextual refinement that safety engineers apply iteratively. In this paper, we introduce HAZDIAL, a framework that investigates whether structured agentic dialogue-multi-agent, multi-turn interactions improves the quality of NLP- based hazard identification over single-pass baselines. We systematically compare two dialogue modalities: adversarial debate and constructive discussion, and propose an algorithm-based agentic interaction optimization. We evaluate all configurations against a curated golden dataset using standard classification metrics (accuracy, precision, recall, F1) and novel dialogue metrics. This work advances the intersection of dialogue systems, multi-agent reasoning, and AI safety, providing an empirical evidence for dialogue-driven hazard analysis. 

---
# From Control Boundary to Insurance Claim: Reconstructing AI-Mediated Losses Through the CER Framework 

**Authors**: Alex Leung, Rex Zhang, Kentaroh Toyoda, SiewMei Loh  

**Link**: [PDF](https://arxiv.org/pdf/2606.03777)  

**Abstract**: AI losses that arise through an insured organization's generative or agentic AI system require state reconstruction, not merely event reconstruction, because the relevant state changes as the system reasons, retrieves, calls tools, and acts. The relevant question is not only what loss occurred, but what the system was allowed to do, what it actually did, and whether that reconstructed loss can support insurance claim recovery. This paper addresses losses in which the insured's AI system is in the causal chain, including externally triggered failures such as prompt injection, retrieval-augmented generation (RAG) poisoning, malicious tool output, credential misuse, and data poisoning.
Specifically, this paper introduces CER, a use-case-level diagnostic for AI residual risk transfer. C (control boundary) asks whether the system had an enforceable operating envelope. E (evidence reconstruction) asks whether the system state and causal chain can be reconstructed from retained artifacts. R (insurance response) asks whether the reconstructed loss is insured: whether insurance coverage is available in the market and placed for the insured, together with the proof needed to support insurance claim recovery.
The paper makes three contributions: it defines the AI-specific reconstruction problem, operationalizes that problem through CER, and specifies claim-grade evidence for AI reconstruction. Public examples include the reported PocketOS and Replit agentic database-deletion incidents and Moffatt v. Air Canada as an adjudicated output/reliance case.
Keywords: AI systems; CER framework; residual risk transfer; agentic AI; generative AI; AI insurance; evidence reconstruction. 

---
# LAP: An Agent-to-Instrument Protocol for Autonomous Science 

**Authors**: Linwu Zhu, Liqiang Gao, Yan Chen, Dan Zhu, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03755)  

**Abstract**: Autonomous science is moving from demonstration to infrastructure. Large language model agents now plan experiments, and self-driving laboratories execute them. Yet every such system rebuilds the link between the reasoning agent and the physical instrument from scratch, against fragmented vendor SDKs and standards built for deterministic software clients rather than probabilistic, goal-directed agents. Recent agent-interoperability protocols clarify two of the three edges of an agentic ecosystem (Anthropic's Model Context Protocol (MCP) standardizes the agent-to-tool edge, and Google's Agent2Agent (A2A) the agent-to-agent edge), but neither models the agent-to-instrument edge, where operations are stateful, safety-critical, exclusively owned, physically embodied, and produce measurements with units, calibration, and uncertainty. We present the Lab Agent Protocol (LAP), a protocol design that fills this gap. LAP retains A2A's peer-to-peer, discovery-first, task-lifecycle structure and adds four physical-world primitives: (i) the InstrumentCard, a signed capability and physical-limit description; (ii) first-class reservation for exclusive instrument and sample locking; (iii) a safety-fence handshake with operator-confirmation tokens cryptographically bound to a specific task and its parameters, gating hazardous and irreversible operations; and (iv) a MeasurementResult schema that makes every result physically typed (QUDT/UCUM), calibration-anchored, uncertainty-bearing, and reproducible by construction. We specify roles, a six-layer architecture, the JSON-RPC method set, the task and safety state machines, the error model, and cross-laboratory federation, and walk a closed-loop autonomous campaign through the protocol end-to-end. LAP is transport-compatible with the A2A/MCP ecosystem and encapsulates rather than replaces existing device standards such as SiLA 2 and OPC-UA. 

---
# Proof-Refactor: Refactoring Generated Formal Proofs into Modular Artifacts 

**Authors**: Yiming Fu, Peixuan Liu, Zichen Wang, Kun yuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.03743)  

**Abstract**: While Large Language Models (LLMs) have shown strong performance in generating formal proofs, their outputs often remain less readable, modular, maintainable, and reusable than proofs in mature formal mathematics libraries. We argue that this gap stems in part from the compile-first objective implicit in most proof-generation pipelines, which encourages monolithic or ad hoc proof scripts rather than library-quality artifacts. Existing approaches to proof-quality improvement often rely on explicit, computable optimization objectives. In practice, however, the most tractable and experimentally validated objectives are largely length-based, while higher-level qualities such as readability, modularity, maintainability, and reusability are difficult to reduce to reliable automatic metrics. Instead of optimizing proof improvement against a single proxy metric, we take a process-guided approach inspired by human proof-refactoring workflows. We propose an agentic framework $\textbf{Proof-Refactor}$ that decomposes proof refactoring into four phases: extracting candidate proof fragments, designing helper declarations, formally proving the extracted and designed components, and repairing the original proof using the verified components. On generated Lean proofs from PutnamBench and Putnam2025, Proof-Refactor improves rubric-based refactoring scores over a strong Claude Code refactoring baseline, with the largest gains in signature quality and human readability. These results suggest that process-guided refactoring can improve proof structure without treating proof length as the primary objective. 

---
# When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning 

**Authors**: Ayushi Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2606.03741)  

**Abstract**: Long-horizon reasoning requires a system to commit to medium-horizon intent without becoming rigid: re-plan too often and computation never coheres into multi-step structure; commit too long and the plan goes stale. We study this stability-adaptivity tradeoff in the latent reasoning setting, where multi-step computation occurs inside hidden state rather than externalized token traces. We extend the Hierarchical Reasoning Model (HRM) with a feudal-style manager-worker interface: a slow high-level module periodically emits a normalized directional subgoal that persists for P low-level steps, biasing the worker's hidden-state updates and supplying an intrinsic cosine alignment loss. On ARC and ConceptARC, we find that subgoal persistence -- not subgoal injection alone -- is the central knob: moderate periods P in [3, 6] consistently outperform both very frequent (P=1) and very long horizons, with a clear minimum LM loss at P=3 (1.544 vs. 1.674 at P=1, 1.640 baseline; replicated over 5 seeds at mean 1.595, std 0.045). The intrinsic alignment weight lambda shows a complementary narrow optimum (lambda approximately 0.05). A controlled ablation at past-sweet-spot lambda isolates learned directional structure -- not architectural capacity or auxiliary loss alone -- as the source of interference when the alignment signal exceeds its optimum. Together these findings implicate a design principle for compositional planning in latent reasoning systems: medium-horizon intent must be coherent across enough computational steps for compositional structure to form. 

---
# Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs 

**Authors**: Clément Yvernes, Emilie Devijver, Marianne Clausel, Eric Gaussier  

**Link**: [PDF](https://arxiv.org/pdf/2606.03719)  

**Abstract**: The do-calculus defines a general system of inference for interventional queries, allowing causal quantities to be transformed through successive applications of its rules. This process induces a rich space of equivalent interventional expressions, but combining and ordering these rules remains challenging. In this work, we introduce derivation graphs, which represent how do-calculus rules are applied and combined, and characterize the full space of observational and interventional probabilities which are equivalent under the do-calculus. The structure of these graphs yields a simple procedure that uses at most four applications of do-calculus rules. Finally, we show how applying identification algorithms to equivalent causal queries produces multiple valid estimands for the same causal quantity, eventually yielding more efficient estimators. 

---
# Code-on-Graph: Iterative Programmatic Reasoning via Large Language Models on Knowledge Graphs 

**Authors**: Weiwei Ding, Zixuan Li, Long Bai, Zhuo Chen, Kun Su, Fei Wang, Xiaolong Jin, Jin Zhang, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.03705)  

**Abstract**: Knowledge Graphs (KGs) are widely used to mitigate the limitations of Large Language Models (LLMs), such as outdated knowledge and hallucinations. Existing LLM-KG integration frameworks typically rely on predefined operators to retrieve factual knowledge from KGs and inject it into prompts for answer generation. This paradigm faces two critical bottlenecks: 1) Inflexibility: The predefined operators are limited in scope and thus lack sufficient compositional expressiveness to fully capture the complex semantics required by KG questions. 2) Unscalability: Direct injection of factual knowledge into prompts limits scalability in handling large-scale factual knowledge. To address these two bottlenecks, we propose Code-on-Graph (CoG), a programmatic reasoning framework for LLM-KG integration. Specifically, given the factual knowledge retrieved at each reasoning step, CoG first identifies the corresponding KG schemas and represents these schemas as Python classes, which serve as abstract interfaces to the retrieved facts. It then generates executable code grounded in these classes, with the retrieved facts instantiated as objects of the corresponding classes during execution. This design enables flexible code-based reasoning while avoiding the direct injection of large-scale factual knowledge into prompts. Experiments on WebQSP, CWQ, and GrailQA demonstrate that CoG outperforms prior state-of-the-art models by up to 10.5%. 

---
# Dynamic Objective Selection with Safeguards and LLM Oversight for Financial Decision-Making 

**Authors**: Keigo Sakurai, Takahiro Ogawa, Miki Haseyama, Anjyu Anan, Kei Nakagawa  

**Link**: [PDF](https://arxiv.org/pdf/2606.03704)  

**Abstract**: Financial decision-making tasks such as stock recommendation and portfolio allocation typically estimate future return and risk and then select trades or allocations for an investor, and the chosen optimization objective often determines realized performance. However, because market conditions evolve over time, a fixed objective can be suboptimal across regimes, while regime-switching pipelines that rely on latent regime estimates can be noisy or delayed and frequent switching can increase turnover and operational instability. In this paper, we propose DOSS (Dynamic Objective Selection with Safeguards), a learning-based selector that directly chooses the decision-relevant objective function at each time point from interpretable statistical summaries of recent returns, selecting among a small set of candidates (e.g., return-seeking, loss-averse, and risk-adjusted) without introducing intermediate regime variables. DOSS formulates objective selection as a classification problem over objectives and performs sequential updates with a rolling window to make forward-looking selections without temporal leakage, while also outputting a confidence score for each proposal. To mitigate misselection and excessive switching in deployment, DOSS applies confidence-aware gating with a fail-safe that overrides low-confidence proposals to a conservative default and enforces explicit controls tied to switching frequency. We further integrate governance by positioning a Large Language Model (LLM) as an oversight component rather than a generator of new objectives: the LLM is restricted to accept a proposed objective or override it to a predefined safe default, with deterministic rule-based constraints triggering overrides when needed. 

---
# SkillPyramid: A Hierarchical Skill Consolidation Framework for Self-Evolving Agents 

**Authors**: Yuan Xiong, Ziqi Miao, Qian Chen, Lijun Li, Yequan Wang, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03692)  

**Abstract**: Recent AI agents can flexibly invoke skills to solve complex tasks, but their long-term improvement is fundamentally constrained by a lack of systematic skill construction, accumulation, and transfer. In particular, without a unified framework for skill consolidation, agents tend to redundantly construct similar capabilities across different tasks, are unable to effectively transform experience into reusable assets, and struggle to generalize task-specific skills to novel scenarios. To address this limitation, we propose SkillPyramid, a skill consolidation framework that reuses existing skill experience for broader task generalization. Operating on a hierarchical skill topology, SkillPyramid further introduces a self-evolution mechanism that enables agents to compose, validate, and incorporate new skills during task execution. Experiments on ALFWorld, WebShop, and ScienceWorld across four backbone models show that SkillPyramid substantially increases the average reward by 38.0% and reduces execution steps by 27.7%. Overall, our method transforms a skill collection from a static resource pool into a dynamic evolution system. 

---
# The DeepSpeak-Agentic Dataset 

**Authors**: Sarah Barrington, Maty Bohacek, Hany Farid  

**Link**: [PDF](https://arxiv.org/pdf/2606.03686)  

**Abstract**: We present DeepSpeak-Agentic, a dataset of videos comprising over 37 hours of semi-structured conversations between a human and an embodied AI agent. We use this dataset to evaluate the automatic forensic identification (audio, video, or text) of AI agents, study the nature of human-agent interactions, and provide a benchmark for future advances in the large-language models and AI-generated voices and faces that power embodied AI agents. We also contribute a scalable data-capture system that creates agents, automatically pairs them with human crowd workers, records audiovisual conversations across specified scenarios, and identifies and separates the human and agent in the combined stream. 

---
# EvoDrive: Pareto Evolution for Safety-Critical Autonomous Driving via Self-Improving LLM Agents 

**Authors**: Tong Nie, Yuewen Mei, Yihong Tang, Junlin He, Jie Deng, Jian Sun, Wei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.03678)  

**Abstract**: Generating safety-critical scenarios is essential for validating and improving autonomous driving systems, yet it inherently requires maximizing adversariality to expose failures while preserving realism. Existing methods usually manage this trade-off with handcrafted heuristics, confining generation to known priors and overlooking underexplored patterns. While recent open-ended agentic evolution can push this limit, unconstrained general agents lack strict simulator grounding and tend to collapse the multi-objective tension into single-scalar maximization. Here we present EvoDrive, the first automated, LLM-based agentic evolution framework for multi-objective scenario generation. EvoDrive employs a simulator-grounded actor-critic architecture where a memory-driven actor iteratively proposes improvements to the generators and critics filter out implausible candidates, and a self-evolving world evaluator routes promising proposals to optimize simulation budgets. EvoDrive further maintains a Pareto archive of evaluated candidates to preserve diverse attack-realism trade-offs and guide future evolution via simulation feedback. Benchmark results on MetaDrive and CARLA show that EvoDrive not only significantly expands the Pareto frontier across various generators, but also produces valuable scenarios for policy training. 

---
# From Answers to States: Verifiable Process-Level Evaluation of Chemical Reasoning in Large Language Models 

**Authors**: Hongyu Guo, Hao Li, He Cao, Gongbo Zhang, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.03660)  

**Abstract**: Large language models are increasingly used as chemistry assistants, yet most chemistry benchmarks still score only final answers. This masks a critical failure mode: a model may output the correct molecule, product, or option while its reasoning violates chemical logic. Existing process-level evaluators are hard to scale because LLM judges and human step-level process annotation are costly, inconsistent, and vulnerable to hallucination. We introduce ChemCoTBench-V2, a rule-verifiable diagnostic benchmark for low-cost, auditable evaluation of structured, verifier-addressable chemical reasoning traces. It spans molecular understanding, molecule editing, molecular optimization, and reaction prediction, with 5,620 evaluation samples across 18 reporting tasks. Models must expose key intermediate steps in expert-designed templates, and those steps are checked with deterministic chemistry rules and, for closed-answer tasks, reference traces rather than another LLM judge. Open-ended molecular optimization is evaluated with oracle-verifiable state constraints rather than strict trace matching. The benchmark reports three separate signals: final-answer correctness, template adherence, and step-wise verifier correctness over expert-refined intermediate commitments. Experiments on frontier models reveal a persistent gap between final-answer success and structured-reasoning-state consistency: models often follow the requested format while failing chemical-step checks, or answer correctly with weak supporting reasoning. ChemCoTBench-V2 enables fine-grained model comparison and identifies the concrete step at which the trace first violates the verifier. 

---
# Diagnosing Knowledge Gaps in LLM Tool Use: An Agentic Benchmark for Novel API Acquisition 

**Authors**: Jinnuo Liu, Yue Peng, Jinhan Niu, Hongyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2606.03657)  

**Abstract**: Large language models for code generation often need to use APIs that are absent from their pretraining data. This requires more than recalling a function name: models must coordinate signatures, module paths, input-output contracts, semantics, and executable usage patterns. Existing novel-API benchmarks are typically static, rely on coarse pass/fail metrics, or use synthetic APIs that may not reflect real library evolution. We introduce NovelAPIBench, a fully automated dynamic benchmark that, for any base model and target library, discovers novel APIs, extracts decomposed knowledge bundles, generates executable coding tasks, and assigns failed samples to six diagnostic categories. Across about 1.9K tasks, four base models, and five domains, we compare knowledge injected through retrieval with knowledge internalized through parametric adaptation. We find that knowledge components are not interchangeable: usage examples are the strongest standalone signal, while the best two-component setting pairs signatures with either mechanisms or examples depending on the domain and backbone. Adding more context, especially source code, can hurt by increasing import-path errors. Parametric adaptation also does not replace retrieval once external knowledge is removed; rather, fine-tuning mainly teaches models how to use provided bundles, and this ability transfers to held-out libraries. These results suggest that retrieval and tuning play complementary roles: retrieval supplies volatile API content, while tuning improves procedural integration. 

---
# Towards Non-Monotonic Entailment in Propositional Defeasible Standpoint Logic 

**Authors**: Nicholas Leisegang, Thomas Meyer, Ivan Varzniczak  

**Link**: [PDF](https://arxiv.org/pdf/2606.03655)  

**Abstract**: Recent work in defeasible reasoning has seen notions of preferential semantics and entailment in the style of Kraus et al. applied to modal logics. However, work in this field has focussed primarily on satisfiability checking, and monotonic notions of entailment, which may be inferentially weak. One particular modal logic where this has been introduced is propositional standpoint logics, where modalities can express the views of different viewpoints. This has resulted in the formalisation of propositional defeasible standpoint logic (PDSL). In this paper, we propose a means of lifting the class of (non-monotonic) rational entailment relations from traditional KLM-style reasoning to a fragment of PDSL. In order to do so, we extend the expressivity of PDSL via situated standpoint conditionals, allowing us to talk about a defeasible conditional holding in the context of a given standpoint. This allows us to re-characterise the syntax of PDSL in terms of situated conditionals, and shows that a large fragment of PDSL is expressible as a set of situated conditionals. We then focus on characterising non-monotonic entailment in this fragment, defining a method to transport any ranking-based entailment relation from the propositional case into the PDSL case. This is first described in the general case and then considered in the specific cases of rational and lexicographic closures, providing a faithful translation of each inference into PDSL. We also show that entailment-checking in this fragment of PDSL can be done largely using algorithms from the propositional case, while preserving complexity bounds. 

---
# Gender-Dependent Diagnostic Substitution in LLM Medical Triage: Same Symptoms, Unequal Urgency 

**Authors**: Qi Han Wong  

**Link**: [PDF](https://arxiv.org/pdf/2606.03641)  

**Abstract**: We investigate whether large language models produce different medical triage recommendations for identical neurological symptoms when only the patient's stated gender and age vary. Using three model families--Gemini 3.5 Flash, Claude Sonnet 4.6, and GPT-5.4-mini--we present a standardized symptom profile (persistent headache, blurred vision, morning nausea, visual disturbances) across seven demographic conditions: three age groups (25, 38, 65) x two genders (male, female), plus a gender-unspecified baseline (n = 30 per condition per model, 630 total trials). We find a stark, systemic gender-dependent triage disparity: young women receive significantly lower emergency room (ER) referral rates than age-matched men (Gemini: 0% vs. 23.3%; Claude: 6.7% vs. 96.7%; GPT: 6.7% vs. 66.7%, all p < 0.001). The disparity disappears at age 65 for all models. The primary mechanism is diagnostic substitution: the models anchor on a gender-associated diagnosis, preferentially classifying young women with Idiopathic Intracranial Hypertension (IIH)--a condition epidemiologically linked to women of childbearing age--while diagnosing men with generic increased intracranial pressure with space-occupying lesions in the differential. This diagnostic closure routes female patients to lower-urgency care (outpatient doctor appointments) despite comparable severity ratings (7-9/10). Our findings demonstrate that clinical LLMs replicate documented human clinical biases by using epidemiological priors to suppress triage urgency, suggesting that AI triage engines must decouple urgency assessment from probabilistic diagnostic priors. We release all code, prompts, and raw results. 

---
# TSQAgent: Rating Time Series Data Quality via Dedicated Agentic Reasoning 

**Authors**: Shunyu Wu, Dan Li, Haozheng Ye, Weibin Feng, Jian Lou, Bo Zhang, Wenjie Feng, Chenjuan Guo, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2606.03629)  

**Abstract**: Assessing the quality of time series (TS) data is fundamental yet inherently challenging due to the multifaceted nature of quality dimensions. Recently, large language models (LLMs) have emerged as a promising paradigm for TS quality assessment via pairwise comparison and per-dimension evaluation. However, existing approaches rely on manually predefined quality dimensions and purely text-based reasoning, leaving it unknown whether LLMs can identify truly relevant quality dimensions or perform grounded and quantitative quality comparisons. To investigate this, we construct TSQBench, a dedicated benchmark for evaluating LLMs on two progressive capabilities: (i) understanding and identifying relevant quality dimensions, and (ii) performing quality comparison under specific dimensions. Our analysis reveals that current LLMs consistently struggle with both dimension identification and evidence-grounded quality comparison. To address these limitations, we propose TSQAgent, a novel agentic reasoning framework for TS quality rating consisting of three collaborative roles: Perceiver for focused dimension selection, Inspector for dimension-wise quantitative analysis, and Adjudicator that aggregates and refines the final judgment. In particular, we introduce an agentic reasoning strategy that instills the ability to identify and prioritize the most relevant quality dimensions, and further propose an agent workflow equipped with external analytical tools to enable precise quantitative comparisons over selected dimensions. Experiments on both the proposed benchmark and eleven real-world datasets demonstrate that our framework not only substantially improves LLMs' capabilities in quality understanding and quantitative comparison but also effectively translates these improvements into better quality-aware data selection, leading to enhanced downstream performance and data efficiency. 

---
# Bridging Auxiliary Constraints to Resolve Instruction Following in Large Reasoning Models 

**Authors**: Zhengyi Zhao, Shubo Zhang, Huimin Wang, Zezhong Wang, Yutian Zhao, Yefeng Zheng, Binyang Li, Yulan He, Kam-Fai Wong, Xian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03624)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated impressive capabilities in many tasks, yet they struggle with reliably following multiple instructions, either by failing to satisfy individual constraints or by struggling to balance competing constraints simultaneously. We formalize this challenge as the Constraint Adherence Problem (CAP). This paper introduces a novel framework that addresses CAP by representing instructions as a structured knowledge graph of constraints. Our approach, Constraint Relationship Graph Completion (CRGC), explicitly models relationships between constraints, identifies adherence challenges, and discovers ``bridge constraints'' that help the model better focus on and reconcile requirements. Bridge constraints act as auxiliary instructions that make primary constraints more salient and compatible. Unlike existing approaches that enhance instruction following through general training methods, CRGC specifically improves constraint satisfaction by leveraging the model's own knowledge to create better pathways for generation. Experiments across three popular instruction following datasets demonstrate that our approach reduces constraint violations by 39% compared to standard prompting while maintaining reasoning abilities of large reasoning models. 

---
# Cross-Lingual Token Arbitrage: Optimizing Code Agent Context Windows via Local LLM Preprocessing 

**Authors**: Mehmet Utku Colak  

**Link**: [PDF](https://arxiv.org/pdf/2606.03618)  

**Abstract**: AI-assisted coding agents are bottlenecked by input-token cost. Two pathologies of raw human input drive much of this overhead: tokenization inefficiency for non-English text and structural entropy in conversational prompts. Existing approaches act reactively by compressing already-bloated contexts or intervening after failures occur.
We introduce a pre-flight, edge-side prompt-rewriting middleware that operates between the developer and the cloud agent. A local Llama 3.2 (3B) model performs cross-lingual translation into English, structural rewriting into a compact task-oriented format, and regex-validated rewrite-with-fallback safeguards to ensure the optimized prompt is never larger than the original.
We evaluate on OMH-Polyglot, a multilingual coding benchmark spanning Turkish, Arabic, Chinese, and code-switched specifications. Across three commercial LLM backends, the middleware reduces prompt tokens by 34-47 percent and total tokens by up to 18.8 percent while preserving or improving task accuracy. Ablation studies show that gains arise primarily from the rewriting stage rather than simple function-name extraction. Compared with LLMLingua-2 at matched compression rates, our method consistently achieves superior OckScore performance across all evaluated backends. These results demonstrate that proactive prompt optimization can substantially reduce inference costs without sacrificing coding quality. 

---
# From Prompt to Service: An SLM-Based Agent Orchestration Gateway for AI-Driven Virtual Worlds 

**Authors**: Louis Nisiotis, Aimilios Hadjiliasi  

**Link**: [PDF](https://arxiv.org/pdf/2606.03557)  

**Abstract**: As generative AI capabilities expand, AI-driven virtual worlds face a growing architectural challenge. Users interact through in-world interfaces in multimodal ways, yet their requests demand fundamentally different AI backend models and computational resources. Embedding these capabilities directly into virtual world systems reduces extensibility, complicates maintenance, and limits the ability to coordinate services distributed across edge and cloud infrastructure. This paper presents an SLM-based Agent Orchestration Gateway, a lightweight runtime coordination mechanism that decouples a virtual world client from heterogeneous AI backends through intent-driven service routing. An edge-deployed SLM classifies the semantic intent of each user prompt, a configurable service registry validates and resolves the routing decision, and the selected backend is invoked transparently, enabling new AI capabilities to be introduced in the virtual world without modifying the client application. The gateway is implemented and evaluated within the InterwovenXR virtual museum testbed. The evaluation shows that compact SLMs can serve as reliable intent routers on edge hardware, and that task-specific fine-tuning can transform sub-billion-parameter models into practical, low-latency routers. A layered configuration pairing a fine-tuned sub billion-parameter model as router with a larger SLM for conversational response generation is shown to be deployable on mid-range edge hardware and more efficient than delegating both responsibilities to a single model. The findings show that SLMs can support practical AI service orchestration in virtual worlds and the work contributes an evaluated architecture for scalable, extensible, and edge-supported AI interaction, enabling virtual agents become access points to distributed generative AI services. 

---
# SAGE: A Quantitative Evaluation of Socialized Evolution in Agent Ecosystems 

**Authors**: Linyue Pan, Yaoming Zhu, Lin Qiu, Xuezhi Cao, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2606.03544)  

**Abstract**: Self-improving language agents are typically evaluated in isolation: an agent attempts a task, receives feedback, and iteratively refines its own behavior. Yet agents increasingly operate alongside peers whose strategies and outcomes are publicly visible. This raises an under-studied question: when does shared experience produce improvements that self-improvement alone cannot achieve? We introduce SAGE (Social Agent Group Evolution),an evaluation framework that compares two compute-matched conditions: SocialEvo, where agents from five distinct model families co-evolve with access to all peers' histories; and SelfEvo, where each agent receives the same number of task attempts but sees only its own past, which is conventional in self-improving agent studies. We instantiate SAGE in three arenas: open-ended ML research, long-horizon economic planning, and strategic multiplayer play, evaluated across multiple evolutionary rounds. We find that group history is not a universal amplifier: the strongest agent does not exceed its self-evolution ceiling. However, agents that plateau under self-improvement can achieve significant breakthroughs when peer experience is available. In competitive settings, counterfactual controls reveal that agents improve generally rather than developing opponent-specific strategies. Across different forms of shared history, filtered peer traces and reflective summaries often outperform raw logs, indicating that social gains depend on abstraction rather than exposure volume. These findings reveal that peer-history gains are agent-specific, arena-dependent, and contingent on the capacity to abstract transferable knowledge from public traces. 

---
# Overlaying Governance: A Compositional Authorization Framework for Delegation and Scope in Agentic AI 

**Authors**: Amjad Ibrahim, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.03518)  

**Abstract**: As AI systems evolve from passive models into autonomous active agents capable of initiating actions, collaborating, and delegating tasks, the traditional boundaries of software systems blur. Traditional authorization and delegation frameworks, built around fixed principals, explicit requests, and static scopes, are insufficient to govern agentic systems. Agentic AI demands richer authorization semantics: agents must inherit and delegate permissions, act under time-limited authority, and coordinate through shared protocols. Existing Identity and Access Management (IAM) systems fail to fully capture this notion of agency, lacking mechanisms for recursive delegation, contextual boundaries, and dynamic scoping as executable governance primitives. Unlike access delegation standards such as OAuth 2.0, we treat delegation as a contractual term rather than merely a static token-based consent credential. This paper proposes a compositional governance framework that introduces primitives indispensable for agentic AI. We define types of delegation and their permissions and accountability implications, and we introduce a notion of resource scope attenuation to bound agentic access envelopes. These concepts are expressed as general relational definitions that can be composed into existing authorization domains (e.g., financial systems). To operationalize this composition, we define a compositional operator that overlays new agentic semantics, such as recursive delegation chains, onto existing relational policies without rewriting them. We substantiate this framework through formal proofs and empirical evaluation, showing that it provides a formal yet practical foundation for accountable authorization in agentic AI systems. 

---
# ThoughtFold: Folding Reasoning Chains via Introspective Preference Learning 

**Authors**: Ziyan Liu, Xueda Shen, Yuzhe Gu, Songyang Gao, Kuikun Liu, Guangran Cheng, Chengqi Lyu, Dahua Lin, Wenwei Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.03503)  

**Abstract**: Large Reasoning Models (LRMs) have achieved remarkable progress thanks to Reinforcement Learning with Verifiable Rewards (RLVR) on Chain-of-Thoughts (CoTs). However, since long CoTs naturally contain trial and errors and mainstream RLVR approaches choose outcome-correct CoT trajectories for memorization, the redundant explorations in long CoTs are inevitably reinforced, which results in the over-thinking issues of LRMs. Previous attempts to resolve this issue mainly give more advantage to shorter trajectories, yet their learning signals are still outcome-based and cannot reduce the memorization of redundant explorations in long CoTs. Therefore, we propose ThoughtFold, a framework that leverages fine-grained preference learning to mitigate redundant explorations for efficient reasoning. ThoughtFold employs an introspective strategy to identify redundancy within each correct trajectory, which yields a spectrum of candidate sub-trajectories. Leveraging this spectrum, we introduce a masked preference optimization objective that explicitly penalizes redundant explorations and encourages the model to directly bridge essential reasoning segments, effectively folding its reasoning chains into a more concise path. Extensive experiments show that ThoughtFold significantly enhances efficiency. It reduces the token usage of DeepSeek-R1-Distill-Qwen-7B by approximately 56% while maintaining state-of-the-art accuracy. 

---
# A formal definition and meta-model for a machine theory of mind 

**Authors**: Fabio Cuzzolin  

**Link**: [PDF](https://arxiv.org/pdf/2606.03471)  

**Abstract**: This paper proposes, for the first time, a rigorous formal definition of the concept of Machine Theory of Mind, based on principles supported by evidence from cognitive psychology, neuroscience and artificial intelligence, and uses the above as a lens to examine state-of-the-art and current efforts in the field, driving a potential agenda for further research there able to "crack" the problem. It also advances a general holistic meta-model for Machine Theory of Mind, and examines the state of the art when it comes to empirically benchmarking such models. 

---
# StepFinder: A Temporal Semantic Framework for Failure Attribution in Multi-Agent Systems 

**Authors**: Taiyu Zhu, Yifan Wu, Weilin Jin, Ying Li, Gang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03467)  

**Abstract**: LLM-based multi-agent systems exhibit remarkable collaborative capabilities in complex multi-step tasks. However, these systems are highly sensitive to single-step execution errors that can propagate through agent interactions and lead to cascading failures. To understand the causes of failure and improve system reliability, failure attribution has been introduced as a task that aims to automatically identify the root cause step responsible for a failure. Existing failure attribution methods mainly rely on LLMs to reason over original execution trajectories, which not only incur high inference costs and latency, but also suffer from interference caused by redundant and noisy execution logs, causing LLMs to struggle in accurately identifying the true root cause step. To address this, we propose StepFinder, a lightweight failure attribution framework. We use LLMs solely during the feature construction phase to encode execution logs into temporal semantic sequences. Subsequently, a parameter-efficient combination of temporal modeling and attention modules is applied to capture the sequential evolution and cross-step dependencies of the trajectories. Finally, the step-level error score is refined through multi-scale differences and position bias, enabling precise root cause identification. Experimental results on the Who&When benchmark demonstrate that StepFinder outperforms LLM-based methods in step-level failure attribution while achieving substantially higher inference efficiency, reducing inference time by 79% compared with the fastest LLM-based method, with no text generation overhead. Our code is available at this https URL. 

---
# DMF: A Deterministic Memory Framework for Conversational AI Agents 

**Authors**: Matteo Stabile, Enrico Zimuel  

**Link**: [PDF](https://arxiv.org/pdf/2606.03463)  

**Abstract**: Conversational AI agents require memory systems that are both scalable and semantically coherent across long interaction horizons. Existing approaches rely predominantly on large language model (LLM)-based summarisation at write time, which introduces non-determinism, escalating token costs, and opacity in pruning decisions. We present the Deterministic Memory Framework (DMF), a CPU-first approach that replaces generative memory compression with a fully deterministic pipeline grounded in classical NLP analysis, vector geometry, and mathematical scoring. DMF assigns each conversational interaction a Survival Score $\Omega$ computed from deterministic content signals, conversational cues, and structured provenance, combined through a logistic projection. An interaction-count decay law, denoted as $\Omega_{\mathrm{eff}}(\Delta n)$, governs how relevance evolves as new turns arrive, where $\Delta n$ is the number of newer interactions rather than wall-clock time, preserving full determinism. We present the mathematical formulation of DMF, its structured recall pipeline, the pruning decision procedure, and the evaluation protocol. Experiments are conducted on a purpose-built benchmark using the LoCoMo and LongMemEval datasets. We compare DMF against Mem0, a popular memory layer for AI agents. DMF achieves comparable accuracy while using zero tokens to prepare the memory context and 5x to 242x fewer tokens over the entire conversation. These results show that it is possible to eliminate LLM calls from the memory-management loop, reducing token costs to nearly zero and enabling deterministic memory systems for conversational AI agents. 

---
# What Makes Interaction Trajectories Effective for Training Terminal Agents? 

**Authors**: Sidi Yang, Chaofan Tao, Jierun Chen, Tiezheng Yu, Ruoyu Wang, Yuxin Jiang, Yiming Du, Wendong Xu, Jing Xiong, Taiqiang Wu, Lifeng Shang, Xiaohui Li, Ngai Wong, Haoli Bai  

**Link**: [PDF](https://arxiv.org/pdf/2606.03461)  

**Abstract**: Stronger code agents are commonly assumed to be superior teachers for post-training, yet this assumption remains poorly disentangled from task difficulty, harness design, and student capacity. We investigate this pedagogical link using Terminal-Lego, a scalable pipeline that transforms multi-domain real-world issues into environment-verified agentic tasks. Surprisingly, standalone performance does not dictate teaching efficacy: while Claude Opus 4.6 achieves higher scores on Terminal-Bench 2.0, students fine-tuned on trajectories from DeepSeek-V3.2, a lower-scoring agent, exhibit significantly stronger generalization. We attribute this "pedagogical paradox" to Environment-Grounded Supervision (EGS): trajectories that explicitly expose inspect-act-verify behaviors through harness-visible interactions allow students to internalize robust problem-solving routines rather than fragile action sequences. Scaling analysis reveals exceptional data efficiency: with only 15.3k Terminal-Lego trajectories, for example, Qwen3-32B achieves a 24.3% score on Terminal-Bench 2.0, rivaling previous SOTA performance established with over 30x the data volume. Our results suggest that the frontier of agent post-training lies beyond mere outcome-matching, shifting the focus toward "Harness Engineering", where the systematic design of environment-grounded interaction structures serves as the primary catalyst for reproducible and generalizable agentic intelligence. 

---
# CP-Agent: Context-Aware Multimodal Reasoning for Cellular Morphological Profiling under Chemical Perturbations 

**Authors**: Yuxin Zhang, Yiyao Li, Ping Shu Ho, Simon See, Zhenqin Wu, Kevin Tsia  

**Link**: [PDF](https://arxiv.org/pdf/2606.03435)  

**Abstract**: Cell Painting combines multiplexed fluorescent staining, high-content imaging, and quantitative analysis to generate high-dimensional phenotypic readouts to support diverse downstream tasks such as mechanism-of-action (MoA) inference, toxicity prediction, and construction of drug-disease atlases. However, existing workflows are slow, costly and difficult to interpret. Approaches for drug screening modeling predominantly focus on molecular representation learning, while neglecting actual experimental context (e.g., cell line, dosing schedule, etc.), limiting generalization and MoA resolution. We introduce CP-Agent, an agentic multimodal large language model (MLLM) capable of generating mechanism-relevant, human-interpretable rationales for cell morphological changes under drug perturbations. At its core, CP-Agent leverages a context-aware alignment module, CP-CLIP, that jointly embeds high-content images and experimental metadata to enable robust treatment and MoA discrimination (achieving a maximum F1-score of 0.896). By integrating CP-CLIP outputs with agentic tool usage and reasoning, CP-Agent compiles rationales into a structured report to guide experimental design and hypothesis refinement. These capabilities highlight CP-Agent's potential to accelerate drug discovery by enabling more interpretable, scalable, and context-aware phenotypic screening -- streamlining iterative cycles of hypothesis generation in drug discovery. 

---
# InfoMem: Training Long-Context Memory Agents with Answer-Conditioned Information Gain 

**Authors**: Tiancheng Han, Yong Li, Wuzhou Yu, Qiaosheng Zhang, Wenqi Shao  

**Link**: [PDF](https://arxiv.org/pdf/2606.03329)  

**Abstract**: Long-context tasks require LLMs to identify and preserve answer-relevant information from large contexts. Chunk-wise memory agents address this issue by sequentially reading document chunks, updating a compact memory, and generating the final answer from the accumulated memory. However, existing RL-based chunk-wise agents either rely on sparse final-answer rewards or use lexical intermediate rewards for memory and retrieval actions. These signals supervise task success or local overlap, but do not directly evaluate whether the final memory supports the ground-truth answer. We propose InfoMem, a reward mechanism for training chunk-wise memory agents that evaluates final-memory utility using answer-conditioned information. InfoMem measures how much the final memory increases the model's per-token log-likelihood of the ground-truth answer. To stabilize RL optimization, InfoMem applies this signal only to successful trajectories and normalizes it before reward composition. Under the same GRPO framework and training budget, InfoMem improves long-context memory-agent performance over comparable memory-agent RL baselines. Analyses show that effective final-memory rewards should operate on successful trajectories, be normalized before reward composition, and be conditioned on the answer rather than the query. Our code is available at this https URL. 

---
# The Violation Situation Pattern: A Knowledge-Graph Pattern for Compliance Violations 

**Authors**: Nima Kamali Lassem, Fuqi Song, Seyid Amjad Ali  

**Link**: [PDF](https://arxiv.org/pdf/2606.03326)  

**Abstract**: Compliance pipelines detect violations as transient query results and do not keep the violation itself as a persistent graph object with review state, affected entities, or audit history. The Violation Situation Pattern (VSP) closes this gap. Building on the Situation pattern of Gangemi and Mika, VSP reifies each detected violation as a graph node with a rule identifier, a temporal validity interval, a lifecycle state, and evidence links to the entities involved. Lifecycle transitions are stored as immutable, PROV-O-aligned events, so audit history is a graph traversal. We instantiate VSP in a legal entity and contract lifecycle property graph and operationalize four deontic rules (V1 unauthorized signature, V2 expired mandate, V3 missing confidentiality clause, V4 missing breach-notification clause) through an FCL->Cypher->MERGE pipeline. We check V1 and V2 against BODACC corporate-officer publications, evaluate V4 on 73 GDPRhub enforcement decisions, and run a SHACL cross-formalism check on V3 and V4. The central finding is rule-body independence: extending V4 from clause-presence to deadline checking raises F1 from 0.312 to 0.602, while the pattern's identity, lifecycle, and evidence semantics stay the same. This separates a pattern contribution from a detector contribution, so detection logic can evolve without invalidating accumulated audit history. 

---
# The Reliability Gap in Benchmark Auditing: Distribution Shift and Scale as Failure Modes of Contamination Detection 

**Authors**: Wojciech Zarzecki, Jan Dubiński, Sebastian Cygert  

**Link**: [PDF](https://arxiv.org/pdf/2606.03305)  

**Abstract**: Benchmark contamination, where evaluation examples appear in a model's training data, threatens the validity of LLM assessment. Statistical tools for detecting training-data membership exist, but have been validated almost exclusively in controlled academic regimes: large, homogeneous pre-training corpora and transparent, single-stage training pipelines. Whether these methods remain reliable in realistic auditing scenarios remains unclear. We identify two under-studied failure modes: distribution shift, which arises when suspect and validation sets violate the IID assumption, and scale constraints, which arise because benchmarks are orders of magnitude smaller than pre-training corpora. We systematically evaluate three leading paradigms: LLM Dataset Inference, Post-Hoc Dataset Inference, and CoDeC across 27 models from multiple families (including Pythia, OLMo~2, and specialised cultural and medical LLMs) and scales (up to 27B). We then further extend our analysis to frontier industry models. Across 335 evaluations, only 199 yield correct outcomes. LLM Dataset Inference results in false positives under distribution shift, Post-Hoc Dataset Inference is underpowered at benchmark scale, and CoDeC provides only coarse provenance signals that are insufficient to verify individual benchmark splits. Our results reveal a systematic reliability gap between controlled validation and practical benchmark auditing, and show that statistical detection cannot yet replace transparent data provenance. We open-source our benchmark for further research. 

---
# LEAP: Supercharging LLMs for Formal Mathematics with Agentic Frameworks 

**Authors**: Po-Nien Kung, Linfeng Song, Dawsen Hwang, Jinsung Yoon, Chun-Liang Li, Simone Severini, Mirek Olšák, Edward Lockhart, Quoc V Le, Burak Gokturk, Thang Luong, Tomas Pfister, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2606.03303)  

**Abstract**: Large Language Models (LLMs) exhibit strong informal mathematical reasoning but struggle to generate mechanically verifiable proofs in formal languages like Lean. We present LEAP, an agentic framework that enables general-purpose foundation models to achieve state-of-the-art performance on automated formal theorem proving. LEAP leverages foundation model capabilities, such as informal reasoning, instruction following, and iterative self-refinement. By decomposing complex problems into smaller units, the system bridges formal proof construction with informal blueprints through continuous interaction with the Lean compiler. To provide a rigorous evaluation beyond increasingly saturated benchmarks, we introduce Lean-IMO-Bench, a benchmark of IMO-style problems formalized in Lean, with short statements yet highly non-routine and multi-step proofs across a wide range of difficulty levels. Empirically, on the latest 2025 Putnam Competition, an annual mathematics competition for undergraduate students in North America, LEAP solves all 12 problems, matching recent breakthroughs by frontier formal mathematical models. On Lean-IMO-Bench, LEAP boosts the one-shot formal solve rate of general-purpose LLMs from below 10% to 70%, notably surpassing the 48% benchmark set by a specialized, gold-medal-caliber IMO system. Furthermore, we demonstrate LEAP's research-level utility by autonomously formalizing complex proofs for open combinatorial challenges, including a verified proof for a key subproblem in Knuth's Hamiltonian decomposition of even-order Cayley graphs. 

---
# A Negative Result on Cross-Model Activation Transfer in a Pythia Multi-Hop Setting 

**Authors**: Peiyan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03280)  

**Abstract**: Recent work shows that language models can transmit behavioural traits through hidden signals in generated data during training. We ask whether a more direct and stricter channel is also viable: can one language model communicate useful intermediate reasoning state to another at inference time by translating and injecting hidden activations, rather than by passing natural-language text? We test this question in a controlled Pythia-160M to Pythia-410M multi-hop reasoning setting. A linear translation layer learns a strong normalized-space map between sender and receiver hidden states, with normalized cosine similarity near 0.97 across seeds. However, when the translated activations are injected into the receiver at inference time, they do not improve downstream answering. Low-strength additive injection remains near the no-injection baseline, with confidence intervals that cross zero. Replacement-style injection is consistently destructive, and rescaling translated vectors to the receiver hidden-state norm does not rescue performance. The result is therefore a scoped negative result: in this setting, offline representational alignment is not sufficient for useful causal communication inside the receiver. 

---
# Distilling Answer-Set Programming Rules from LLMs for Neurosymbolic Visual Question Answering 

**Authors**: Thomas Eiter, Nelson Higuera Ruiz, Johannes Oetsch  

**Link**: [PDF](https://arxiv.org/pdf/2606.03269)  

**Abstract**: Visual Question Answering (VQA) is the task of answering questions about images, requiring the integration of multimodal input and reasoning. Modular approaches that incorporate logic-based representations into the reasoning component offer clear advantages over end-to-end trained systems, particularly in terms of interpretability. However, adapting or extending these representations when task requirements change can place a significant burden on developers. To address this challenge, we present an approach for distilling rules from Large Language Models (LLMs). Our method prompts an LLM to extend an initial VQA reasoning theory, expressed as an answer-set program, to meet new requirements of the task. Examples from VQA datasets guide the LLM, validate the results, and help correct erroneous rules by leveraging feedback from the ASP solver. We demonstrate that our approach is effective across diverse VQA datasets. Notably, only a few examples are needed to elicit correct rules from LLMs. Our experiments suggest that rule distillation from LLMs is a promising alternative to traditional data-driven rule learning approaches. Under consideration in Theory and Practice of Logic Programming (TPLP). 

---
# Do Real-World Datasets Contain Natural Experiments? An Empirical Study Using Causal Feature Selection 

**Authors**: Gautam Gare, John Galeotti, Michael Mozer, Deva Ramanan, Nan Rosemary Ke  

**Link**: [PDF](https://arxiv.org/pdf/2606.03251)  

**Abstract**: In nature, events that affect some individuals or groups but not others constitute an implicit intervention and are known as natural experiments. For example, the COVID-19 pandemic was an intervention by the coronavirus on the sub-population infected with COVID. We ask, do natural experiments occur in existing real-world datasets? If yes, how should we treat them? To detect natural experiments in data, we use causal discovery to recover the underlying causal graph and perform feature selection based on causal links. If downstream performance improves by treating the data as interventional rather than observational, we argue that this suggests the dataset contains natural experiments. We first validate this hypothesis by simulating datasets with and without natural experiments using synthetic graphs. We then perform a systematic empirical evaluation on a large suite of real-world datasets. Our results indicate that real-world datasets do contain natural experiments and we can take advantage of those natural experiments to improve model performance using causal inference. Our work represents the initial foray into this area, offering a preliminary exploration within a limited scope. 

---
# Solipsistic Superintelligence is Unlikely to be Cooperative 

**Authors**: Rakshit S Trivedi, Natasha Jaques, Logan Cross, Alexander Sasha Vezhnevets, Joel Z Leibo  

**Link**: [PDF](https://arxiv.org/pdf/2606.03237)  

**Abstract**: AI's central challenge is shifting from capability to coexistence. The dominant paradigm in AI research focuses on developing powerful agents that treat the world as an exogenous and stationary source of feedback. We contend that superintelligence, an extremely capable task solver, born out of such a solipsistic approach to AI design, is unlikely to be cooperative. Deploying AI systems induces endogenous non-stationarity, resulting in a train-test-deploy gap where historical distributions diverge from the deployment context. We refer to this as the self-undermining property of unilateral optimization. Closing this gap requires AI that participates in cooperation: the equilibrium-selection process through which multiple actors navigate their interdependence. We call for a non-solipsistic research paradigm that treats this interdependence as a core design principle rather than approaching cooperation as a task to solve. This entails building dynamic evaluation testbeds involving adaptive counterparties, treating institutions as design primitives, and preserving human agency as a structural feature of the systems we build. 

---
# Perceive Before Reasoning: A Pre-Reasoning Perception Framework for Efficient and Reliable Proactive Mobile Agents 

**Authors**: Zhijie Ding, Weinan Hong, Zicheng Zhu, Lei Li, Dezhi Kong, Hao Wang, Peng Zhou, Xuchu Jiang, Jiaming Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03236)  

**Abstract**: Multimodal large language models (MLLMs) have substantially advanced mobile agents, yet proactive mobile assistance remains challenging because agents must decide \emph{when} to intervene before determining \emph{how} to assist. Existing systems often implement these two decisions within a unified MLLM-based pipeline, leading to goal misalignment between conservative intervention filtering and comprehensive assistance generation, as well as redundant inference when the agent should remain silent. To address these limitations, we propose the \textbf{Pre-Reasoning Perception Framework (PRPF)}, a two-stage framework built on perceiving before reasoning. PRPF introduces a lightweight Multimodal Proactive Perceptor (MPP) for intervention gating and context compression, and activates the Proactive Agent Reasoner (PAR) only when intervention is warranted. Experiments on the ProactiveMobile benchmark show that PRPF substantially reduces false trigger rates (FTR) while improving success rates (SR) and inference efficiency over the ProactiveMobile baseline. 

---
# Effect of Demographic Bias on Skin Lesion Classification 

**Authors**: Ralf Raumanns, Gerard Schouten, Veronika Cheplygina, Josien P.W. Pluim  

**Link**: [PDF](https://arxiv.org/pdf/2606.03214)  

**Abstract**: In this study, we evaluate the performance of skin lesion classification using ResNet-based convolutional models, focusing on the impact of demographic bias in training data, particularly variations in patient sex and age. We use linear programming to generate datasets with controlled demographic characteristics, allowing systematic investigation of bias effects. Three learning strategies are evaluated: a single-task model, a reinforcing multi-task model, and an adversarial learning scheme. Our sex-based analysis indicates that sex-specific training datasets optimise model performance. Notably, including male patients in the training data improved performance for the male subgroup, even in female-majority cases. Reinforcing and adversarial learning schemes narrowed or eliminated bias gaps in balanced and female-majority datasets. However, these strategies proved less effective in male-majority settings, where models continued to perform better for males than females. The two learning schemes showed marginal bias reduction compared to the baseline model in predominantly male patient populations. Age-based analysis demonstrates comparable baseline performance across the three model approaches, with performance declining across age categories. Younger groups consistently achieve the highest performance, regardless of training data distribution. Although balanced training yields optimal results for the youngest age category, performance decreases in older categories. We find that sex biases arise mainly from data imbalances, while age biases consistently favour younger groups regardless of distribution. These distinct mechanisms require targeted mitigation strategies. Additionally, cross-dataset validation on two external datasets revealed that domain shifts notably affect performance and patterns of demographic bias. 

---
# MedCUA-Bench: A Screenshot-Only Benchmark for Clinical Computer-Use Agents 

**Authors**: Jia Yu, Zilong Wang, Xinyang Jiang, Dongsheng Li, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03203)  

**Abstract**: Computer-use agents could automate repetitive screen-based clinical work, but their reliability in medical graphical user interfaces remains largely unvalidated. Existing benchmarks focus on general web or desktop tasks and underrepresent medical software, which requires domain knowledge, exhibits markedly different UI design from mainstream applications, lacks public testing environments, and demands safety validation beyond task completion. We introduce MedCUA-Bench, an interactive benchmark for clinical computer-use agents. It covers 18 clinical scenarios across 10 medical domains, reconstructed from real product manuals and open-source medical systems to capture authentic clinical interfaces while avoiding licensing and privacy constraints. Each task ships with paired intent- and step-level goals to disentangle clinical reasoning from UI execution, and is evaluated by a deterministic checker over task completion and five clinical safety dimensions. Across 23 agents, the best closed-source model reaches 54.2% strict success, while all models remain below 9% on the real OpenEMR. Open-source agents average only 2.5%, with the best reaching 16.2%. MedCUA-Bench exposes the gap between current agents and reliable clinical software use, providing a reproducible testbed for future research. 

---
# ClinicalMC: A Benchmark for Multi-Course Clinical Decision-Making with Large Language Models 

**Authors**: Ruihui Hou, Siyi Zhu, Ziyue Huai, Guangya Yu, Yongqi Fan, Chunming Wang, Tong Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2606.03157)  

**Abstract**: Large language models (LLMs) have been widely adopted in healthcare, yet they still encounter significant challenges in complex clinical decision-making scenarios. Existing benchmarks primarily assess LLM performance in single-course settings and lack systematic evaluation in multi-course scenarios, where a patient's condition evolves over time. To address this gap, we propose ClinicalMC, a benchmark for multi-course clinical decision-making. It includes 1,275 Chinese and 5,804 English samples across four stages from admission to discharge. These stages cover triage, first-course examination/diagnosis/treatment, subsequent multi-course examination/assessment/treatment, and final diagnosis. In ClinicalMC, patients in the English dataset undergo an average of 5.11 clinical courses, whereas those in the Chinese dataset undergo 3.42. To assess LLM performance, we construct a multi-agent evaluation framework that includes patient, examiner, and doctor agents. Based on the benchmark and framework, we design two experimental settings -- a single-turn static setting and a multi-turn dynamic setting -- and assess three categories of LLMs: 1) closed-source LLMs like GPT5-mini; 2) open-source LLMs like DeepSeek-V3.2; and 3) medical LLMs like HuatuoGPT-o1. Through extensive evaluation, we aim to better understand LLM performance in the medical domain and support its effective deployment in healthcare. 

---
# GTBench: A Curriculum-Grounded Benchmark for Evaluating LLMs as Mathematical Research Assistants in Graph Theory 

**Authors**: Noujoud Nader, Ibrahem Aljabea, Patrick Diehl, Deepti Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2606.03144)  

**Abstract**: Large language models (LLMs) are increasingly used as self-study assistants in technical disciplines, yet their reliability as mathematical reasoning assistants remains poorly understood. We introduce GTBench, a curriculum-grounded benchmark for evaluating LLMs as mathematical research assistants in graph theory, comprising 63 problems organized into three groups of increasing difficulty: undergraduate definitions and basic properties (Group 1), algorithm tracing and structural reasoning (Group 2), and graduate-level proof construction (Group 3). Problems are sourced from verified academic materials including Diestel's Graph Theory. We evaluate five frontier models -- GPT-5, Claude Sonnet 4.6, Gemini 2.5 Flash-Lite, Llama 3.3 70B, and Mistral Large 3 -- under zero-shot and chain-of-thought prompting, using exact-match and LLM-as-judge evaluation for Groups 1 and 2, and a hybrid human expert and LLM-as-judge protocol for Group 3. Our results reveal a pronounced performance hierarchy: GPT-5 approaches ceiling on Group 1 (95.8% zero-shot) and maintains meaningful accuracy on graduate proofs (82%), while all other models degrade substantially with difficulty, with Llama achieving 0% under human evaluation on Group 3 zero-shot. Failure mode analysis shows that correct algorithm, wrong execution errors dominate Groups 1 and 2, while Group 3 additionally surfaces incomplete reasoning failures and reveals systematic disagreement between human evaluators and the automated judge, particularly on verbose or near-complete proofs (kappa = 0.48-0.83 across human pairs). GTBench provides the first curriculum-grounded evaluation framework for graph-theoretic reasoning in LLMs, with direct implications for the governance of AI tools in mathematical education and scientific research. 

---
# Think-Before-Speak: From Internal Evaluation to Public Expression in Multi-Agent Social Simulation 

**Authors**: Kaiqi Yang, Tai-Quan Peng, Sanguk Lee, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03137)  

**Abstract**: LLM-based multi-agent simulation offers a promising way to study social interaction, deliberation, and collective opinion dynamics. However, many existing dialogue simulation frameworks represent interaction mainly as observable turn exchange or aggregated outputs, leaving the internal evaluative processes behind silence, speaking intention, and public expression difficult to examine. We introduce TBS (Think-Before-Speak), an interval-based multi-agent simulation framework that separates agents' private reasoning from public utterance generation. At each interval, all agents update structured internal states based on the shared dialogue history and their own memory. These states include dissonance-related appraisal, perceived opinion climate, perceived isolation risk, response strategy, and willingness to speak. The orchestrator then resolves competing speaking intentions and commits one utterance to the public dialogue, allowing internal evaluation and public interaction to co-evolve over time.
We evaluate TBS in simulated town hall discussions on a climate-related policy issue. Results show that TBS produces coherent internal-state traces and that these traces vary systematically across turn-allocation, silence, and memory conditions. Dissonance-related appraisal increases agents' willingness to speak, whereas silence-pressure appraisal decreases it. Once speaking intention is formed, public expression is shaped mainly by turn-allocation rules. These findings suggest that TBS supports mechanism-sensitive social simulation by making the pathway from internal evaluation to public expression observable and analyzable. 

---
# Uncertainty-Aware Clarification in LLM Agents with Information Gain 

**Authors**: Mengyi Deng, Zhiwei Li, Xin Li, Tingyu Zhu, Ying Zhao, Zhijiang Guo, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03135)  

**Abstract**: Large Language Model (LLM) agents often operate under underspecified user instructions, where latent uncertainty over user intent leads to erroneous tool actions. To address this challenge, we propose a goal-oriented clarification framework that aligns clarification behavior with ambiguity resolution. Central to our approach is the Information Gain Reward, a metric that quantifies the utility of clarification questions by measuring the Bayesian belief update towards the ground-truth goal induced by the clarification exchange. We train the clarifier (LLM) using this reward to optimize for high information gain, ensuring that clarifications effectively reduce uncertainty and improve task completion within the agent-tool-user environment. We validate our framework within a clarification-enhanced $\tau$-Bench environment, conducting cross-agent evaluations across five heterogeneous backbones. Empirical results demonstrate that our method consistently improves the success rate by 3.7\% over the no-clarification baseline, while adding only 0.3 total interaction steps on average. 

---
# EvoTrainer: Co-Evolving LLM Policies and Training Harnesses for Autonomous Agentic Reinforcement Learning 

**Authors**: Guhong Chen, Yingcheng Shi, Yongbin Li, Binhua Li, Xander Xu, Hu Wei, Shiwen Ni, Min Yang, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2606.03108)  

**Abstract**: Autonomous LLM training is often framed as recipe search, which leaves the training harness largely static. This limitation sharpens in agentic RL, where shifting bottlenecks and scalar rewards mask diverse failure modes. We introduce EvoTrainer, an autonomous training framework that co-evolves LLM policies and training-side harnesses through empirical feedback: it diagnoses rollout-level evidence, revises diagnostics, backtests interventions, and accumulates reusable skills. Evaluated on mathematical reasoning, competitive-programming code generation, and repository-level software engineering, EvoTrainer matches or exceeds the human-engineered RL references under the same data, codebase, and evaluation protocol, with the largest gain on long-horizon agentic SWE. Trajectory analyses show that retained strategies diverge across domains, evolving diagnostics prevent invalid high-scoring branches from being promoted, and reusable skills shape later search. Autonomous LLM RL should move beyond recipe search toward joint evolution of policies and the training harnesses that interpret them. 

---
# DeskCraft: Benchmarking Desktop Agents on Professional Workflows and Human-in-the-Loop Collaboration 

**Authors**: Wenkai Wang, Tao Xiong, Jingchen Ni, Yunpeng Bao, Xiyun Li, Tianqi Liu, Hongcan Guo, Zilong Huang, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03103)  

**Abstract**: Real-world professional desktop workflows in specialized creative and engineering software unfold over long horizons and often require human-in-the-loop coordination, where agents proactively seek necessary information and users provide additional instructions, clarifications, feedback, or corrections as the task progresses. Yet existing desktop GUI benchmarks mostly reduce this setting to short, simplified tasks with all user instructions provided upfront. To address this issue, we introduce DeskCraft, a desktop GUI benchmark targeting long horizon creative and engineering workflows and proactive human-agent collaboration. DeskCraft organizes tasks into a multilevel difficulty taxonomy, with long horizon tasks requiring over 50 execution steps, and covers professional creative software across design, video, audio, and 3D creation. Furthermore, DeskCraft formalizes human-agent collaboration into an interaction protocol covering mid-turn and post-turn exchanges. Mid-turn interaction captures both agent-initiated clarification under uncertainty and user-initiated interruption during execution, while post-turn interaction accommodates user-driven feedback after the agent signals completion, together spanning the full space of realistic collaboration patterns. We evaluate 18 proprietary and open source agents on 538 tasks and find that GPT-5.4 reaches 31.6% on standard tasks and 27.6% on interactive tasks. Further analyses reveal persistent failures in long horizon workflow delivery and proactive clarification. We will open-source all evaluation codes, tasks, and data at this https URL. 

---
# From Long News to Accurate Forecast: Importance-Aware Fusion and PRM-Guided Reflection for Time Series Forecasting 

**Authors**: Mingyang Liu, Qingcan Kang, Yuke Wang, Shixiong Kai, Kaichao Liang, Hui-Ling Zhen, Tao Zhong, Mingxuan Yuan, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2606.03097)  

**Abstract**: Incorporating news into time series forecasting is appealing because news can reveal abrupt exogenous events that historical values alone cannot recover. However, existing LLM-based news-forecasting pipelines face two practical limitations: relevant news articles often exceed the model's context window, and iterative retrieval of supplementary news is typically unguided, leading to redundant updates and slow convergence. We address these issues with a novel framework that combines importance-aware news compression and process-level retrieval supervision. First, we train an importance reward model that estimates the forecasting utility of each article and uses this signal to allocate compression budgets during sequential pairwise fusion, preserving informative content within a fixed context limit. Second, we introduce a process reward model (PRM) that ranks multiple supplementary-news candidates conditioned on the current error profile and the history of previously selected articles, replacing one-shot blind retrieval with quality-controlled selection. Both components are trained offline using historical data with ground truth; inference uses the frozen filtering logic and compression modules without any reflection loop. Experiments on finance, energy, traffic, and bitcoin forecasting benchmarks show that our method improves prediction accuracy over strong baselines, significantly reduces the number of refinement iterations compared to the iterative baseline, and remains effective when relevant articles span thousands of tokens. 

---
# Decomposing how prompting steers behavior 

**Authors**: Fan L. Cheng, Nikolaus Kriegeskorte  

**Link**: [PDF](https://arxiv.org/pdf/2606.03093)  

**Abstract**: Prompting steers large language models (LLMs) and vision-language models (VLMs) without weight updates, but it remains unclear how instruction changes reshape internal representations to produce behavior. We introduce a nested geometric decomposition framework that treats prompting as a transformation of the representational geometry of the content following the prompt. For each prompt pair, we align representations of the same stimuli under two prompts using increasingly expressive stimulus-invariant maps: translation, rigid transformation with uniform scaling, sequential axis scaling, affine transformation, and nonlinear transformation. We then causally test each map by replacing a single layer's prompt-A hidden state for held-out stimuli with its mapped counterpart and measuring recovery of prompt-B representational geometry and behavior. Across three LLMs, three VLMs, and six text or image datasets spanning style, emotion, scene content, and number, prompts consistently reshape representations toward the instructed task structure. Cross-validated variance decomposition shows that much prompt-induced activation change is captured by shape-preserving maps, especially translation and rigid transformation with uniform scaling, while tier profiles reveal model- and task-specific routing strategies across layers. Crucially, although translation and rigid tiers already improve behavioral agreement, affine transformation is the first tier to nearly recover target-prompt task geometry and yields corresponding behavioral gains. This suggests that cross-dimensional linear mixing is a key mechanism by which prompts reorganize representations toward instructed task structure. Our framework decomposes prompt-induced representational change into interpretable geometric components and reveals how models route task-relevant structure to produce prompt-driven behavior. 

---
# The Shadow Price of Reasoning: Economic Perspective on Optimal Budget Allocation for LLMs 

**Authors**: Xu Wan, Speed Zhu, Jianwei Cai, Guang Chen, XiMing Huang, Wiggin Zhou, Mingyang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.03092)  

**Abstract**: Inference-time scaling has emerged as a critical avenue for enhancing Large Language Models' performance, yet real-world deployment is constrained by strict computational budgets. In this work, we formulate inference budget allocation as a global constrained optimization problem governed by economic principles. By modeling per-query reasoning utility with a shifted-surge function, we derive an optimal allocation policy based on a global shadow price that equilibrates marginal utility under resource scarcity. Based on this theory, we propose Constrained Latent-utility Equilibrium Allocation for Reasoning (CLEAR). It performs rational abandonment and reallocates resources from insolvent queries to solvable queries near their emergence thresholds.
Extensive experiments on several reasoning tasks with different traffic streams demonstrate that CLEAR significantly improves the Pareto frontier of total token cost versus mean accuracy. In resource-scarce regimes, CLEAR achieves up to a 3x improvement in global accuracy compared to uniform allocation. 

---
# DELTAMEM: Incremental Experience Memory for LLM Agents via Residual Trees 

**Authors**: Haoran Tan, Zeyu Zhang, Zhicheng Cao, Rui Li, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.03083)  

**Abstract**: Large Language Model (LLM)-based agents increasingly rely on memory to learn from experiences over continual interactions. However, storing experiences as independent, flat units leads to substantial redundancy and retrieval conflicts, as similar episodes repeat overlapping content and subtle scene variations cause retrieved memories to offer contradictory guidance. To address this, we introduce residual experience, positing that newly acquired experience is often an incremental variation of existing knowledge. We propose DeltaMem, a framework that organizes experience memory into two independent residual trees, one storing goal-conditioned task experience as reusable skills and another for scene-level environment knowledge. Each tree uses a root node for generalized base experiences and incremental delta nodes for subsequent variations, allowing related experiences to share a common foundation without duplication. For retrieval, a failure-penalized similarity scan locates the best match, reconstructing the full experience via root-to-match chain composition. An autonomous consolidation mechanism distills high-frequency paths into new root nodes, enabling the trees to self-organize from general heuristics to specialized variants. Experiments across diverse interactive environments show that DeltaMem consistently outperforms existing baselines. To facilitate future research, we release the code at this https URL. 

---
# CORE: Conflict-Oriented Reasoning for General Multimodal Manipulation Detection 

**Authors**: Jinjie Shen, Yaxiong Wang, Yujiao Wu, Lechao Cheng, Tianrui Hui, Nan Pu, Zhihui Li, Zhun Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2606.03066)  

**Abstract**: The rapid rise of generative AI has made multimodal fake news increasingly realistic and pervasive, posing severe threats to public trust and social stability. Existing detection methods rely heavily on manipulation-specific models and large-scale labeled data, resulting in poor generalization to emerging manipulation types. We observed that the essence of manipulated misinformation lies in its intrinsic conflicts, \textbf{i.e.,} semantic or physical inconsistencies either across modalities or with common world knowledge. Inspired by this observation, we propose \textbf{C}onflict-\textbf{O}riented \textbf{RE}asoning (\textbf{CORE}) framework, an effective paradigm that learns to endows multimodal large language models (MLLMs) with explicit conflict-capturing capability. To this end, CORE first constructs the Conflict Attribution Corpus (CAC) with fine-grained annotations of conflict factors and sources, providing essential data support for subsequent conflict perception training. By performing conflict-oriented representation enhancement and reasoning based on CAC, CORE achieves robust and generalizable conflict detection, effectively and rapidly adapting to unseen manipulation types with a few samples or in even zero-shot settings. Extensive experiments demonstrate that CORE surpasses state-of-the-art models. The dataset and code are publicly available at this https URL. 

---
# SkillDAG: Self-Evolving Typed Skill Graphs for LLM Skill Selection at Scale 

**Authors**: Tong Bai, Zhenglin Wan, Pengfei Zhou, Xingrui Yu, Wangbo Zhao, Yang You, Ivor W. Tsang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03056)  

**Abstract**: As LLM agents adopt large skill libraries, selecting the right subset becomes a structural problem rather than a similarity-matching one: skills depend on, conflict with, specialize, or duplicate one another, a structure invisible to both full enumeration and embedding similarity. We present SkillDAG, which models inter-skill relationships as a typed directed graph and exposes it to an LLM agent as an inference-time, agent-callable structural retrieval interface, queried and evolved during execution rather than baked into a fixed retrieval pipeline: each search returns vector matches, typed-edge neighbors, and conflict signals, and a propose-then-commit protocol lets the agent register execution-backed edges so the graph accumulates structure across episodes. On ALFWorld and SkillsBench with MiniMax-M2.7, SkillDAG reaches 67.1% success and 27.3% reward, exceeding the strongest reported Graph-of-Skills baseline by +12.8 and +8.6 points; the advantage ports to gpt-5.2-codex, and intrinsic SkillsBench Ret@K rises from 65.5 to 78.2 under matched queries. These gains trace to isolable mechanisms: candidate ranking that stays robust as the pool grows 10x where a fixed seeding-diffusion pipeline degrades, and set-monotone online edits that enlarge ground-truth recall without evicting prior hits. 

---
# ToolGate: Token-Efficient Pre-Call Control for Tool-Augmented Vision-Language Agents 

**Authors**: Anjie Liu, Yan Song, Zhixun Chen, Ziqin Gong, Zhongwei Yu, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03054)  

**Abstract**: Tool-augmented vision-language agents can acquire external perceptual evidence through OCR, detection, segmentation, and other tools, but executing every proposed tool call is costly and sometimes unnecessary. We study the pre-call control problem: after a ReAct-style VLM agent proposes a perceptual tool call, should the call be executed, or skipped before its output enters the context? Across five benchmarks, we find that the baseline agent exhibits poor local selectivity: helpful and harmful calls occur at similar rates (11.8% vs. 9.9%), while most calls do not change the immediate forced-answer prediction. We introduce ToolGate, a lightweight external controller that predicts execute/skip decisions from trajectory text and simple structural features. Across two Qwen3-VL backbones, ToolGate reduces token cost to 64-69% of the unrestricted ReAct baseline while preserving average accuracy in cross-domain settings. With matched-domain trajectory training on Qwen3-VL-30B, it further improves average accuracy by 1.65 points. These results show that tool-augmented VLM agents benefit not only from better perceptual tools, but also from explicit control over when tool outputs are worth paying for. 

---
# RelGT-AC: A Relational Graph Transformer for Autocomplete Tasks in Relational Databases 

**Authors**: Phillip Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03040)  

**Abstract**: Relational databases underpin modern enterprise, scientific, and healthcare systems, yet predictive machine learning on such data remains challenging due to their multi-table, heterogeneous, and temporal structure. Relational Deep Learning (RDL) addresses this by representing databases as heterogeneous graphs and applying graph neural networks (GNNs) directly. RelBench v2 recently introduced autocomplete tasks -- a practically motivated task type where the goal is to predict an existing column value from relational context, analogous to an intelligent form-filling assistant. We propose RelGT-AC (Relational Graph Transformer for Autocomplete), extending the RelGT architecture with three targeted contributions: (1) a column masking strategy that prevents trivial solutions by masking the target column during subgraph encoding; (2) a unified task head supporting binary classification, multiclass classification, and regression autocomplete tasks within a single model; and (3) a TF-IDF text encoder that automatically detects and encodes free-text columns, recovering strong lexical signal that categorical encoders discard. Across 7 tasks spanning 3 RelBench v2 datasets (rel-trial, rel-f1, rel-stack), RelGT-AC outperforms the GraphSAGE baseline on all 3 regression autocomplete tasks and achieves up to +10 AUROC points on text-heavy eligibility tasks via the TF-IDF encoder. 

---
# TriEval: A Resource-Efficient Pipeline for LLM Bias, Toxicity, and Truthfulness Assessment 

**Authors**: Akshatha Srikantha, Manpreet Singh, Yash Jajoo, Shyamal Lakhanpal  

**Link**: [PDF](https://arxiv.org/pdf/2606.03036)  

**Abstract**: LLMs have evolved from basic chatbots to the backbone of the AI ecosystem, now widely used in healthcare, schools, and government services. The domain-wide adoption of LLMs necessitates continuous evaluation to ensure their safety and fairness. Common issues encountered after deploying LLMs include inconsistent outputs and hallucinations of incorrect information. Although numerous LLM evaluation tools exist, most are limited to testing a single parameter at a time or require massive computational resources that are not accessible to most researchers. TriEval addresses these challenges by evaluating LLM outputs across multiple parameters, including bias, toxicity, and truthfulness together, while minimizing computing resources. The pipeline is compatible with both open- and closed-source models and runs on a standard laptop without a GPU cluster. TriEval has been tested on four models: Llama 3 8B, Mistral 7B, Gemma 2 9B, and Claude Haiku. The results show clear differences between open-source and closed-source models, especially in terms of toxicity and truthfulness. TriEval is being released as open source to enable broader access for researchers with limited computational resources. 

---
# AUDITFLOW: Executable Symbolic Environments for Structured Financial Reporting Verification 

**Authors**: Yan Wang, Xuguang Ai, Jaisal Patel, Xueqing Peng, Fengran Mo, Yupeng Cao, Haohang Li, Mingyu Cao, Lingfei Qian, Víctor Gutiérrez-Basulto  

**Link**: [PDF](https://arxiv.org/pdf/2606.03031)  

**Abstract**: Structured financial audit verification is difficult for language-model agents because correctness depends on structured evidence rather than text alone. A model must link reported facts to taxonomy concepts, traverse calculation or dimensional relations, and recompute expected values before applying an audit rule. We propose AuditFlow, a graph-grounded multi-agent framework that separates adaptive search from deterministic verification. AuditFlow builds a symbolic environment from a static US-GAAP taxonomy graph and a dynamic XBRL filing graph, and exposes it through typed tools for fact retrieval, taxonomy traversal, numerical checking, and rule evaluation. Two junior auditors inspect each case from regulatory and evidentiary views, while a senior auditor resolves disagreements and can request further investigation. The final reports are fused through evidential aggregation to produce an audit verdict, expected value, evidence trail, and trustworthiness score. On a FinAuditing-derived FinMR sample, AuditFlow reaches 82.09% joint audit accuracy under GPT-5.5, outperforming the strongest baseline by 14.93 points. Removing deterministic checks drops accuracy to 17.91%, showing that the symbolic environment performs the verification step that the model cannot reliably replace. 

---
# Inducing Reasoning Primitives from Agent Traces 

**Authors**: Zhihan Lei, Jiarui Yan, Joshua Momo, William W. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2606.02994)  

**Abstract**: ReAct-style LLM agents often rediscover the same reasoning routines across problems, yet leave those routines trapped in transient scratchpads. We introduce Reasoning Primitive Induction, a single-pass method that mines successful ReAct traces, clusters recurrent reasoning moves, and converts the most frequent moves into a compact library of typed pseudo-tools. Each pseudo-tool is specified by a natural-language docstring interpreted by an LLM at invocation time, and a standard ReAct loop composes these primitives at test time. The central result is that induced libraries outperform the very agent that generated their traces: by +44pp on RuleArena NBA (30 -> 74), +30pp on MuSR team allocation (38 -> 68), and +22pp on NatPlan meeting planning (7 -> 29). Across five comparable subtasks spanning narrative deduction, rule application, and constraint-satisfaction planning, a single fixed configuration improves over zero-shot Chain-of-Thought on every subtask, matches or surpasses expert-authored decompositions, and outperforms AWM at lower average inference cost. 

---
# WISE-HAR: A Generalizable Ensemble Deep Learning Framework for WiFi-Based Human Activity Recognition 

**Authors**: Maheen Arshad, Qindeel E Zahra, Muhammad Khuram Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2606.02974)  

**Abstract**: Human Activity Recognition (HAR) using WiFi signals has emerged as a transformative technology for smart homes, healthcare monitoring, security systems, and ambient assisted living. Unlike traditional camera-based systems that raise significant privacy concerns and fail in low-light conditions, or wearable sensors that require user compliance, WiFi-based HAR is non-intrusive, privacy-preserving, cost-effective, and works seamlessly in any lighting condition. This paper presents a comprehensive approach to recognize three distinct human activities: "No Presence" (empty room), "Walking", and "Walking + Arm-waving" using the Wallhack1.8k WiFi spectrogram dataset. We propose three key improvements to address the main challenges in WiFi-based HAR. First, to address high performance variance, we implement ensemble learning with five different CNN architectures (Deep CNN, Wide CNN, MobileNetV2, ResNet50V2, and EfficientNetB0). Second, to address the small dataset size limitation, we apply aggressive data augmentation techniques including time-warping, frequency masking, and noise addition. Third, to evaluate real-world generalization capability, we perform cross-scenario evaluation (training on Line-of-Sight and testing on Non-Line-of-Sight) and cross-antenna evaluation (training on Biquad antenna and testing on PIFA antenna). Our ensemble model achieved a test accuracy of 94.87% on the LOS scenario with Biquad antenna, outperforming the best individual model by 0.66%. Data augmentation improved Random Forest performance from 60% to 95%. Cross-scenario evaluation showed minimal accuracy drops of only 1.37% and 2.07%, demonstrating strong generalization capabilities. The results indicate that the proposed approach is robust, reliable, and suitable for real-world deployment in diverse environments with different hardware configurations. 

---
# What Benchmarks Don't Measure: The Case for Evaluating Abstention Competence in Autonomous Agents 

**Authors**: Victor Ojewale, Suresh Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2606.02965)  

**Abstract**: Benchmarks for autonomous agents measure whether agents complete tasks, yet this framing is systematically blind to whether an agent should have proceeded at all. Agents trained under human-feedback objectives develop a structural tendency to proceed even when they lack the inputs, evidence, or authorization to act safely, a disposition we term compliance bias, because both the reward signal and the benchmark scoring regime treat proceeding as the correct default regardless of whether the preconditions for safe action are present. We make three contributions. We first show that compliance bias originates in reward hacking within human-feedback pipelines and is entrenched by prominent agent benchmarks, which either penalize agents for pausing or are architecturally unable to distinguish a principled pause from a silent failure. We then introduce a three-gap taxonomy of abstention-warranted scenarios, covering specification gaps where required information is absent, verification gaps where world state cannot be confirmed, and authority gaps where explicit authorization has not been given, which together provide a principled basis for constructing abstention-aware agent benchmarks. Finally, we propose abstention evaluation protocols (Safety Rate, Usability Rate, and Informed Refusal Rate) and report preliminary results across 144 enterprise agent scenarios and five model families, in which a runtime-enforced abstention mechanism achieves up to 89.2% hazardous-action blocking and 87.5% usability on authorized scenarios, demonstrating that the safety--usability tradeoff is tunable rather than inherent and that its shape varies substantially across model families. We treat this as preliminary work and offer the taxonomy and composite metrics as a starting point for further conversations. 

---
# Large AI Models in Dental Healthcare: From General-Purpose Systems to Domain-Specific Foundation Models 

**Authors**: Sema Helali, Lina Abu Nadab, Sausan Alqawas, Alaa Abd-Alrazaq, Faleh Tamimi, Rafat Damseh  

**Link**: [PDF](https://arxiv.org/pdf/2606.02914)  

**Abstract**: Background: Oral diseases affect nearly 3.5 billion people worldwide, yet the comparative clinical potential of large-scale AI models in dentistry remains poorly understood. Three distinct model categories have emerged: language-generative models, discriminative vision foundation models, and dental-specific foundation models, with no unified review examining their relationships and collective limitations.
Methods: Following PRISMA-ScR guidelines, we systematically searched four databases (PubMed, Google Scholar, Scopus, arXiv), screened independently by two reviewers. After applying inclusion/exclusion criteria, 97 studies (2020-2026) were included. We propose a two-dimensional classification framework organizing models by architectural paradigm and dental specialization degree.
Results: Language-generative models excel at text-based tasks (clinical reasoning, licensing exams, patient communication) but show inconsistent performance on image-dependent diagnostics. Adapted SAM and CLIP variants achieve strong tooth segmentation and lesion detection results. Dental-specific models (DentVFM, DentVLM, OralGPT) demonstrate strongest performance on complex multimodal tasks. Integrated pipelines consistently outperform single-model approaches. A data asymmetry is observed: dental-specific pretraining concentrates almost entirely in the vision domain, reflecting scarce large-scale dental text corpora.
Conclusions: General-purpose and dental-specific models play complementary roles; the most effective systems combine both within structured pipelines. Safe autonomous deployment requires resolving three persistent barriers: hallucination in generative models, limited annotated dental datasets, and absent standardized clinical evaluation benchmarks. 

---
# Handoff Debt: The Rediscovery Cost When Coding Agents Take Over Interrupted Tasks 

**Authors**: Dipesh KC, Anjila Budathoki  

**Link**: [PDF](https://arxiv.org/pdf/2606.02875)  

**Abstract**: Coding-agent benchmarks evaluate whether a single uninterrupted agent can resolve a repository issue. Real software work is messier: tasks are interrupted, reassigned, reviewed, and resumed from partial states left by another agent or engineer. We study this missing dimension through \emph{handoff debt}: the rediscovery cost imposed when a predecessor's work is opaque or incomplete. Our takeover protocol interrupts a coding agent at deterministic handoff points, freezes the repository, and evaluates successor agents under four handoff views: repository state only, raw trace, summary notes, and structured notes. Across 75 source tasks, the protocol generates 181 handoff-point tasks and 724 takeover runs per successor model. Across three successor models, context-bearing handoffs reduce median agent events by 20--59\% and cumulative prompt tokens by 42--63\% relative to repository-only takeover. Solved-rate effects are smaller and model-dependent, but efficiency gains are consistent. These findings suggest that coding-agent evaluation should report not only whether a task is solved, but also how costly that work is for another agent to resume. 

---
# When Helping Hurts and How to Fix It: Multi-Agent Debate for Data Cleaning 

**Authors**: Chirag Parmar, Akshat Mehta, Henglin Wu, Jagadish Ramamurthy, Shweta Medhekar  

**Link**: [PDF](https://arxiv.org/pdf/2606.02866)  

**Abstract**: When does multi-agent debate help data cleaning, and when does it hurt? Across three benchmarks, four model families, and over 6,000 task-condition pairs, we find debate's effect reverses sign: it degrades generation across all four models (-1.6 to -15.5pp) through critique-induced confusion (CIC), hallucinated Critic feedback that the Generator accepts uncritically, yet improves error detection (+27.4pp F1, d=1.0). We derive a debate benefit condition: debate helps when the probability of rescuing a wrong output (Critic verification odds weighted by fixability) exceeds the probability of destroying a correct one. A factorial experiment proves adversarial separation is essential: self-verification with identical tools fails, while a separate Critic with code-execution grounding and evidence-gated generation produces the first debate configuration to significantly exceed single-agent on a generative task (+5.3pp, p<0.05). The condition correctly predicts all nine task types and generalizes with zero false positives across 19 published comparisons in seven domains. 

---
# Don't Gamble, GAMBLe: An Analytical Framework for AI-Driven Research Systems 

**Authors**: Marquita Ellis, Paul Castro  

**Link**: [PDF](https://arxiv.org/pdf/2606.02863)  

**Abstract**: AI-Driven Research Systems (ADRS) -- systems coupling LLMs with automated evaluation to discover algorithms, proofs, and designs -- are being optimized and adopted across domains, but the tools to analyze them have not kept pace. ADRS performance depends on component interactions that are poorly understood, expensive to explore, and (as we show) not well captured by standard convergence guarantees. These guarantees rely on structural assumptions that do not hold under the ADRS process we formalize. We introduce GAMBLe, a framework that decomposes ADRS behavior into four parameters (generator $G$, assessor $\mathcal{A}$, discovery mechanism $\mathcal{M}$, budget $B$) and one compositional object, the effective landscape $L_{\text{eff}} = \mathcal{A} \circ G$, which reveals that distinct generator-assessor pairs induce structurally different per-problem optimization landscapes. We exercise the framework on 760+ replicated runs (>46,000 iterations) spanning generators from single LLMs to dynamically-adaptive ensembles, mechanisms from greedy selection to co-evolutionary meta-search, and three NP-hard problems whose assessors range from continuous scoring to cliff functions. The experiments reveal no total ordering of generators or mechanisms: frontier models can underperform open-source alternatives and the simplest mechanism sometimes outperforms state-of-the-art meta-search. Results show that even under limited budgets (60 iterations per run), the right component choices can improve performance by 13-67% and search efficiency by 6-39x. 

---
# Toward a Modular Architecture for Embedded AI Agent Systems at the Edge 

**Authors**: Marcus Rüb, Michael Gerhards  

**Link**: [PDF](https://arxiv.org/pdf/2606.02862)  

**Abstract**: The rise of Large Language Models (LLMs) has enabled agentic AI capable of complex reasoning and tool use; however, deploying such autonomy in pervasive computing environments remains challenging due to the strict memory and energy constraints of embedded microcontrollers. Existing frameworks typically assume server-class resources or continuous connectivity, leaving a gap for deeply embedded systems. This paper proposes a modular reference architecture for Embedded Agent Systems that bridges the divide between deterministic real-time control and agentic intelligence.
We introduce a tiered design that decouples On-Device Agents - executing highly compressed neural networks and rule-based logic for low-latency, privacy-critical tasks - from Cloud-Augmented Agents that leverage Small Language Models (SLMs) for higher-level reasoning and planning. A key contribution is the integration of a cross-cutting Governance Layer, ensuring observability, policy enforcement, and safety across distributed fleets of autonomous devices. Rather than presenting purely empirical benchmarks, we analyze architectural design principles and trade-offs regarding latency, energy, and reliable execution in resource-constrained environments. 

---
# Thinking Past the Answer: Evaluating Harmful Overthinking in Large Reasoning Models 

**Authors**: Simone Caldarella, Davide Talon, Rahaf Aljundi, Elisa Ricci, Massimiliano Mancini  

**Link**: [PDF](https://arxiv.org/pdf/2606.02835)  

**Abstract**: Large Reasoning Models (LRMs) improve performance by generating explicit intermediate reasoning traces through increased test-time compute, yet the assumption that longer reasoning is consistently beneficial remains under-examined. While recent evidence shows that additional reasoning can lead models to overthink, we ask: "Once a model has reached the correct answer, does further reasoning refine the solution, or deviate from it?" To study the dynamics after correctness, we introduce a prefix-level trajectory evaluation protocol grounded in reasoning sufficiency, defining the minimum reasoning budget required for a model to first generate the correct answer. This allows us to disentangle verbose overthinking, where additional reasoning is redundant but harmless, from harmful overthinking, where continued reasoning destabilizes an already-correct trajectory. Starting from multimodal benchmarks, we find that many instances considered reasoning-intensive require surprisingly little reasoning. Moreover, stopping at the first correct prefix improves accuracy over standard reasoning up to 21%, revealing that current models are limited not only by their ability to reason, but also by their inability to stop at the right time. Furthermore, while common efficiency strategies like early stopping substantially reduce verbose overthinking (up to 50%), they fail to mitigate harmful overthinking. Failure analysis reveals that correctness deviations are mainly driven by logical drift and visual reinterpretation. Finally, we show that our findings generalize to language-only reasoning benchmarks, highlighting harmful overthinking as a broader reliability risk. Code available at this https URL. 

---
# An Exploration of Collision-based Enemy Morphology Generation 

**Authors**: Johor Jara Gonzalez, Matthew Guzdial  

**Link**: [PDF](https://arxiv.org/pdf/2606.02832)  

**Abstract**: Despite a great deal of prior research into Procedural Content Generation (PCG), relatively little prior work has explored generating enemies for video games. In particular, there is almost no work on generating enemy morphologies, the basic body plan or collision information for in-game enemies, despite the existence of related morphology generation work in robotics. In this paper, we explore three different novel approaches to generate enemy morphologies based on player collision information. We found that each approach provides different strengths and weaknesses, but all had equivalent or better performance than an evolutionary baseline adapted from prior robotics morphology work. 

---
# Traj-Evolve: A Self-Evolving Multi-Agent System for Patient Trajectory Modeling in Lung Cancer Early Detection 

**Authors**: Sihang Zeng, Matthew Thompson, Ruth Etzioni, Meliha Yetisgen  

**Link**: [PDF](https://arxiv.org/pdf/2606.02812)  

**Abstract**: Modeling patient trajectories from longitudinal electronic health records (EHRs) requires reasoning over sparse, noisy, and long-context multimodal sequences. Existing LLM-based multi-agent systems address context length but process patients in isolation, failing to mirror how clinicians leverage accumulated experience from similar prior cases. We present Traj-Evolve, a self-evolving multi-agent system with two complementary evolving mechanisms. First, an Experience Pool (ExPool) acts as a non-parametric memory, indexing rejection-sampled reasoning traces to retrieve similar patients as few-shot contexts. Second, multi-agent reinforcement learning (MARL) via reward-ranked fine-tuning parametrically optimizes inter-agent and agent-memory collaboration. A leave-one-out cross-retrieval strategy unifies the two, aligning training- and inference-time behavior under retrieval augmentation. On a lung cancer prediction task utilizing up to five years of multimodal EHRs, Traj-Evolve outperforms 9 strong baselines on the overall population and a challenging never-smoker population. Analysis of the evolving dynamics highlights three key findings: (1) expanding the ExPool shifts optimal retrieval from diverse to specific samples; (2) under MARL, the manager agent's prediction loss converges quickly while the worker agents' temporal reasoning continues to benefit from more verified patients; and (3) the two mechanisms are complementary on the predicted risk, where ExPool improves specificity while MARL improves sensitivity. 

---
# ChatHealthAI: Aligning Electronic Health Record Representations with Large Language Models for Grounded Clinical Reasoning 

**Authors**: Bo-Hong Wang, Baicheng Peng, Ruilin Wang, Jun Bai, Ziyang Song, Yue Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.02802)  

**Abstract**: Large language models (LLMs) exhibit strong natural-language reasoning abilities for clinical decision support, but struggle to effectively model structured longitudinal electronic health records (EHRs). In contrast, EHR foundation models can learn predictive patient representations, yet lack interpretable language-based reasoning. To bridge this gap, we propose ChatHealthAI, a multimodal reasoning framework that aligns structured EHR representations from a pretrained EHR foundation model with the semantic space of a frozen LLM through a task-aware resampler. By integrating longitudinal patient representations with refined clinical event descriptions, ChatHealthAI enables clinically grounded natural-language reasoning while maintaining accurate patient prediction. We evaluated ChatHealthAI on three clinical predictive tasks from the EHRSHOT benchmark. Results show that ChatHealthAI improves reasoning quality and interpretability while preserving competitive predictive performance. These findings highlight the potential of integrating EHR foundation models with pretrained LLMs for interpretable clinical prediction. 

---
# BehaviorBench: Modeling Real-World User Decisions from Behavioral Traces 

**Authors**: Liangwei Yang, Jielin Qiu, Zixiang Chen, Ming Zhu, Juntao Tan, Zhiwei Liu, Wenting Zhao, Zhujun Lan, Akshara Prabhakar, Silvio Savarese, Huan Wang, Shelby Heinecke  

**Link**: [PDF](https://arxiv.org/pdf/2606.02798)  

**Abstract**: Many decision-support settings require systems that adapt to individual users, but evaluation data for this problem remain limited. Existing benchmarks for user understanding often rely on simulated users or model-generated behavior, even though recent work cautions that model-based simulations can diverge systematically from human behavior. We introduce \textsc{BehaviorBench}, a benchmark for evaluating personalized decision modeling from real-world behavioral traces. \textsc{BehaviorBench} reconstructs wallet-level decision histories from observed public prediction-market and on-chain records, and organizes them into two complementary task layers: \emph{Belief prediction}, which predicts a user's final revealed stance and confidence in a market, and \emph{Trade prediction}, which predicts the direction and amount of individual transactions. Across 2,000 evaluation wallets, the benchmark contains 141,445 Belief instances and 1,485,972 Trade instances, with disjoint support pools for retrieval-based evaluation. We evaluate frontier and open-weight generative models under four history interfaces: no personalization, direct recent history, generated user profiles, and retrieved support-wallet evidence. Personalization improves Belief prediction more consistently than Trade prediction, model rankings change across task layers and metrics, and different history interfaces expose different failure modes. \textsc{BehaviorBench} provides an evaluation setting for studying whether personalized methods can use real-world behavioral evidence rather than simulated users alone. 

---
# Evaluating Transformer and LSTM Frameworks for Prediction in Ungauged Basins 

**Authors**: Taye Akinrele, James Halgren, Noorbakhsh Amiri Golilarz, Sudip Mittal, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2606.02791)  

**Abstract**: Watershed networks exhibit convergent topologies in which multiple tributaries merge into downstream channels,integrating diverse upstream hydrological processes. In ungauged basins, the absence of direct observations increases uncertainty and limits the ability to anticipate extreme events. This study evaluates whether an encoder-only Transformer provides an advantage over an LSTM for upstream streamflow inference under limited hydrologic information, using retrospective simulations from the NOAA National Water Model (NWM). Across both upstream-only and combined configurations, the LSTM showed stronger overall performance than the Transformer model across the two configurations. Incorporating downstream information further boosted performance for all models, increasing median NNSE by more than 60%. Rather than treating this as a leaderboard-style comparison, we interpret the experiments as a test of architectural inductive bias for hydrologic sequence inference. The results indicate that recurrent memory remains better aligned with this upstream reconstruction task than an encoder-only Transformer, while downstream hydrologic context provides a strong auxiliary constraint that substantially improves prediction skill across architectures 

---
# AURA: Action-Gated Memory for Robot Policies at Constant VRAM 

**Authors**: Josef Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.02775)  

**Abstract**: The KV-cache is the right memory for datacenters but the wrong memory for robots. Datacenter inference batches many short requests and resets them, amortizing an attention cache across a crowd. Embodied agents instead run one long, non-resetting episode on bandwidth-limited edge hardware, where high-bandwidth memory and flash are scarce, flash has finite write endurance, and memory writes rather than compute can become the binding constraint.
AURA-Mem (Action-Utility Recurrent Adaptive Memory) targets this regime. It wraps a frozen vision-language-action backbone with a constant-size recurrent memory and a learned gate that writes only when the current observation would change the next action: memory that knows when to stay silent. Unlike reconstruction-based memory, the gate is trained directly against a closed-loop action-error signal. Its inference state is fixed at 4,224 bytes regardless of horizon, while a KV-cache grows to 6,061 times larger at 100,000 steps.
On a controlled synthetic benchmark, AURA-Mem matches the best O(1) baseline in accuracy while using 5.19-6.13 times fewer writes, and up to 9.19 times fewer writes on easier configurations. Budget-matched random and periodic schedules do not recover this gain, isolating the benefit to the action-surprise signal. On a trained closed-loop OpenVLA-OFT 7B panel on LIBERO-Long (n=60 episodes per arm), the gate does not hurt success: AURA-Mem matches the ungated base policy (0.233) and slightly exceeds an always-write KV arm (0.217), while using 7.0 times fewer writes and constant memory. We also instantiate an approximate-information-state value-loss bound as a methodology demonstration; at this scale, the bound is vacuous rather than a guarantee. 

---
# Visual Graph Scaffolds for Structural Reasoning in Large Language Models 

**Authors**: Runlin Lei, Xiaokui Xiao, Zhewei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.02673)  

**Abstract**: Graphs have been used to enhance large language models (LLMs) for structured reasoning, mostly as external knowledge sources are provided to models at test time. In this paper, we take a different view: the value of graphs for LLMs lie not only in supplying information, but also in organizing reasoning. Inspired by how humans use graph-structured mind maps to organize branching and converging thoughts, we ask whether graphs can serve as an internal form of reasoning assistance. We study this question on multi-hop question answering tasks, where teacher-provided reasoning traces are rewritten as graph mind maps and used to guide a student model. Our experiments reveal a clear modality gap. When graph structures are flattened into text, their benefits become limited once direct answer hints are removed. Under this abstract guidance setting, both reasoning efficiency and answer quality degrade substantially. In contrast, visual graph guidance remains effective without direct answer clues, and its advantage persists after supervised fine-tuning and KL-based distillation. The above findings support the claim that graphs should be studied not only as external knowledge structures for LLMs, but also as visual scaffolds for organizing reasoning. 

---
# Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking 

**Authors**: Zekun Qi, Xuchuan Chen, Dairu Liu, Chenghuai Lin, Yunrui Lian, Sikai Liang, Zhikai Zhang, Yu Guan, Jilong Wang, Wenyao Zhang, Xinqiang Yu, He Wang, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2606.03985)  

**Abstract**: We introduce Humanoid-GPT, a GPT-style Transformer with causal attention trained on a billion-scale motion corpus for whole-body control. Unlike prior shallow MLP trackers constrained by scarce data and an agility-generalization trade-off, Humanoid-GPT is pre-trained on a 2B-frame retargeted corpus that unifies all major mocap datasets with large-scale in-house recordings. Scaling both data and model capacity yields a single generative Transformer that tracks highly dynamic behaviors while achieving unprecedented zero-shot generalization to unseen motions and control tasks. Extensive experiments and scaling analyses show that our model establishes a new performance frontier, demonstrating robust zero-shot generalization to unseen tasks while simultaneously tracking highly dynamic and complex motions. 

---
# Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories 

**Authors**: Ali Behrouz, Farnoosh Hashemi, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2606.03979)  

**Abstract**: The past few decades have witnessed significant advances in the design of machine learning algorithms, from early studies on task-specific shallow models to more general deep Large Language Models (LLMs). Despite showing promising results in tasks that require instant prediction or in-context learning, existing models lack the ability to continually learn and effectively transfer their temporal in-context knowledge to their long-term parameters. Inspired by human learning process, we introduce a ''Sleep'' paradigm that allows the models to continually learn, distill their short-term fragile memories into stable long-term knowledge with replay, and recursively improve themselves with ''Dreaming'' process. In more detail, sleep consists of two stages: (1) Memory Consolidation: an upward distillation process, called Knowledge Seeding, where the memories of a smaller-self are distilled into a larger network to provide more capacity while preserving the knowledge. As a proof of concept, we present a new Generalized Distillation process for {Knowledge Seeding} (i.e., the combination of on-policy distillation with Reinforcement Learning (RL)-based imitation learning); (2) Dreaming: a self-improvement phase, where the model uses RL to generate a curriculum of synthetic data to rehearse new knowledge and refine existing capabilities without human supervision. Our experiments on long-horizon, continual learning, knowledge incorporation, and few-shot generalization tasks support the importance of the sleep stage. 

---
# Formalizing the Binding Problem 

**Authors**: Lianghuan Huang, Yihao Li, Saeed Salehi, Yingshan Chang, Ansh Soni, Konrad P. Kording  

**Link**: [PDF](https://arxiv.org/pdf/2606.03976)  

**Abstract**: Representations of the world, arguably, contain information about features (e.g. something is blue, something is a circle) but also information about which features are part of the same object (e.g. the circle is blue), which we call binding information. Any system with the ability to understand scenes with multiple objects must be able to solve the binding problem: it needs to know which features belong together. However, despite work showing that Vision Transformers (ViTs) know which patches belong together, it is not known whether current deep learning models learn to exhibit binding information, i.e., for features. We may believe that there is not much binding information, after all misattributing features to wrong objects is a common failure of ViT-based architectures, especially in scenes with objects sharing features. Here we formalize the binding problem with an information-theoretic approach, and introduce a probing method to measure binding information in model representations. We perform experiments on ViTs, measuring binding from different components of the architecture, such as the image summary token [CLS] or the spatial tokens. We use datasets with different binding challenges, such as feature sharing, occlusion, and natural features, while comparing the performance of several pre-trained ViTs. Overall, our research demonstrates binding as a key ingredient to strong visual recognition and reasoning. 

---
# Quantifying Faithful Confidence Expression in Large Reasoning Models 

**Authors**: Areeb Gani, Asal Meskin, Gabrielle Kaili-May Liu, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2606.03969)  

**Abstract**: Reliable uncertainty communication is critical to the trustworthiness of LLMs, yet faithful calibration (FC)--the alignment between models' intrinsic and (linguistically) expressed confidence--is a persistent failure mode. This challenge is key for large reasoning models (LRMs), whose extended reasoning traces are often interpreted by users as evidence of deliberation, competence, and confidence. Despite the importance of FC and wide usage of LRMs, the extent to which LRMs can faithfully express their confidence remains poorly understood. Moreover, the prevailing paradigm to measure FC does not generalize well to the long chain-of-thought outputs generated by LRMs, which tend to lack clear step boundaries, involve inconsistent step structure, and encode complex conditional dependencies throughout the trace--complicating estimation of intrinsic confidence. To address this challenge, we introduce a novel framework to systematically quantify FC of LRMs. Our framework analyzes linguistic decisiveness relative to three sources of internal uncertainty, based on token probabilities, hidden states, and sampled response consistency. We also devise a prefix-conditioned sampling approach to control for conditional and structural variation across traces. Applying our framework to a diverse suite of leading models, datasets, and prompts, we find that faithful confidence expression is a significant challenge for LRMs. Reasoning behaviors do not automatically translate to improved FC, and prompt interventions for non-reasoning models do not improve faithfulness in the reasoning setting. Different confidence estimators further produce divergent assessments of the same traces, revealing fragility in prior evaluation methodologies. Taken together, our work establishes FC as a distinct reliability and alignment target for LRMs, particularly as such systems are increasingly deployed in high-stakes contexts. 

---
# QUBRIC: Co-Designing Queries and Rubrics for RL Beyond Verifiable Rewards 

**Authors**: Rongzhi Zhang, Rui Feng, Zhihan Zhang, Jingfeng Yang, Qingyu Yin, Xin Liu, Zixuan Zhang, Priyanka Nigam, Bing Yin, Tuo Zhao, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03968)  

**Abstract**: Rubric-based RL is a promising route for extending reinforcement learning beyond verifiable rewards, yet existing methods optimize rubrics while treating the query distribution as fixed. We identify a structural bottleneck: rubric quality is constrained by query structure. Open-ended queries yield vague rubrics; naively narrowing them introduces fabricated references that no model can verify, so all responses fail and training receives no reward signal. We present QUBRIC, a framework that co-designs queries and rubrics. Teacher-derived key points ground the rewriting of open-ended queries into scenario-based, evaluable questions. Contrastive rubric generation then turns teacher-policy gaps into query-level criteria, and learnability filtering retains only informative query-rubric pairs for GRPO training. QUBRIC achieves a +5.5 point gain on ArenaHard over the SFT baseline. Trained only on instruction-following data, it further transfers to three held-out benchmarks spanning legal, moral, and narrative reasoning (+6.3 points on average), with improvements concentrated in reasoning-related dimensions. These results provide evidence that co-designing queries and rubrics can make rubric-based RL a practical complement to RLVR beyond strictly verifiable tasks. 

---
# AlignAtt4LLM: Fast AlignAtt for Decoder-Only LLMs at IWSLT 2026 Simultaneous Speech Translation Task 

**Authors**: Quentin Fuxa, Dominik Macháček  

**Link**: [PDF](https://arxiv.org/pdf/2606.03967)  

**Abstract**: We describe AlignAtt4LLM, an IWSLT 2026 simultaneous speech translation system for English to German, Italian, and Chinese. The system is a synchronous cascade: Qwen3-ASR with forced alignment produces an incrementally updated source transcript, and Gemma-4 E4B-it translates that prefix under an MT-side AlignAtt policy.
To our knowledge, this is the first application of AlignAtt to a decoder-only LLM, where the encoder-decoder cross-attention used by earlier AlignAtt systems is absent. We recover a usable policy by proposing (1) an explicit source span in the prompt, (2) offline selection of translation-specific alignment heads, (3) selective qk-fast replay of the draft-to-source attention block, and (4) runtime query/key capture that preserves model outputs bit-identically.
On the IWSLT 2026 development set, AlignAtt4LLM outperforms the supplied baselines for the European target languages, English to German and English to Italian, in both the low-latency regime around 2 seconds and the high-latency regime below 4 seconds CU-LongYAAL. Results for English to Chinese are more mixed, but the method is not tied to Gemma-4: because AlignAtt4LLM only requires a deterministic prompt layout, calibrated attention heads, and query/key capture, the same policy can be reapplied to stronger translation-focused decoder-only MT backbones for non-European target languages. 

---
# Agentic Chain-of-Thought Steering for Efficient and Controllable LLM Reasoning 

**Authors**: Yu Xia, Zhouhang Xie, Xin Xu, Byungkyu Kang, Prarit Lamba, Xiang Gao, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2606.03965)  

**Abstract**: Large language models improve final-answer accuracy through extended chain-of-thought reasoning, but often spend tokens inefficiently and offer little inference-time control. Existing efficient reasoning methods control thinking length by shortening, early-stopping, or compressing traces, leaving how the model thinks implicit. In this paper, we propose Agentic Chain-of-Thought Steering (ACTS), which formulates reasoning steering as a Markov decision process where a controller agent adaptively steers a frozen reasoner during inference. At each step, the controller observes the reasoning trace and remaining thinking budget, then issues a steering action consisting of a reasoning strategy and a steering phrase that initiates the next reasoner step. This enables budget-aware strategy control for efficient reasoning while preserving the reasoner's generation continuity. We initialize the controller agent from our constructed synthetic steering trajectories with multi-budget augmentation, and further optimize it via reinforcement learning with budget-conditioned reward shaping. Experiments across multiple benchmarks show that ACTS matches full-thinking performance with substantial token savings, and enables controllable accuracy-efficiency trade-offs across different reasoners and tasks. The code is available at this https URL. 

---
# Self-Refining Agentic Reinforcement Learning for Vision-Conditioned UAV Navigation 

**Authors**: Roohan Ahmed Khan, Yasheerah Yaqoot, Muhammad Ahsan Mustafa, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2606.03963)  

**Abstract**: Deep reinforcement learning has shown strong potential for enabling autonomous robots to learn complex navigational tasks. However, its practical use still depends heavily on human designed reward functions and repeated manual fine tuning, which is time consuming and does not guarantee high success in the desired task. This paper presents AgenticRL, agent guided reinforcement learning framework that increases autonomy in reward design, policy refinement, and real world deployment for unmanned aerial vehicles (UAV) navigation tasks. AgenticRL uses a multimodal generative pre-trained tansformer (GPT) agent to interpret task information and visual scene observations, generate task specific reward functions, train policies using Proximal Policy Optimization (PPO) algorithm, and then act as a critic by evaluating the trained policy through diagnosis packets to generate feedback. Based on this feedback, the agent identifies failure modes and refines the reward function in a closed loop self improvement process. To further leverage the multimodal GPT agent during inference, AgenticRL uses real world images and natural language task information to automatically identify the active scenario and select the appropriate trained policy for execution. The framework is evaluated on multiple navigational tasks, including gate traversal, obstacle avoidance, wall barrier crossing with landing, trajectory following, and motion behavior learning. Experimental results show that the closed loop refinement process improves policy behavior compared with initial rewards by 71%. We also demonstrate sim-to-real transfer of the proposed framework, achieving a real world success rate of 91% and a sim-to-real accuracy of 94%. 

---
# Using Reward Uncertainty to Induce Diverse Behaviour in Reinforcement Learning 

**Authors**: Anthony GX-Chen, Ankit Anand, Gheorghe Comanici, Zaheer Abbas, Eser Aygün, David Smalling, Shibl Mourad, Doina Precup, André Barreto, Mark Rowland  

**Link**: [PDF](https://arxiv.org/pdf/2606.03962)  

**Abstract**: Classical reinforcement learning (RL) typically seeks a deterministic policy that maximizes the expected sum of a scalar reward. Yet, modern applications such as language model fine-tuning or scientific discovery demand diversity. Existing remedies such as entropy regularization or diversity bonuses often require fragile trade-offs that sacrifice performance for stochasticity or rely on heuristic metrics that can misalign policy rankings. We argue that diversity is more naturally understood as the rational response to uncertainty in the reward. When the reward function is not perfectly known--as is the case with ambiguous preferences or imperfect reward models--committing to a single action can be sub-optimal. Building on this, we propose a fundamental reformulation of the RL objective by replacing the scalar reward with a distribution over reward functions, and applying a non-linear objective over sets of actions. The result is a framework in which calibrated behavioural diversity emerges naturally, remains controllable through the reward function distribution, and is obtained without sacrificing expected reward. Focusing on the contextual bandit setting, we derive a principled gradient estimator for this objective and prove that our formulation naturally generalizes both vanilla policy gradient and more recently developed action-set approaches. Our empirical results demonstrate that this framework offers a robust and theoretically grounded alternative for complex RL tasks where the traditional formulation of the problem fails to induce the desired breadth of agent behaviour. 

---
# Efficient ASR Training with Conversations that Never Happened 

**Authors**: Máté Gedeon, Péter Mihajlik  

**Link**: [PDF](https://arxiv.org/pdf/2606.03957)  

**Abstract**: Conversational ASR for lower-resource languages and niche domains is limited by the scarcity of domain-matched multi-speaker training data. We propose an augmentation pipeline that generates scenario-level dialogues with participant metadata, maps speaker attributes to TTS voice profiles, and assembles synthesized utterances into speaker-aware simulated conversations. We evaluated five LLM families under single-generator, fixed-budget mixture, and scale-up settings using the same FastConformer-Large training recipe for each one. We ran comprehensive evaluations on the Hungarian BEA-Dialogue benchmark corpus, with the method itself being applicable to any language given the resources for each component. The results show that synthetic conversations consistently improve speech recognition performance, but generator choice and data composition strongly affect the gains. Our largest training configuration, using only 67 hours of real conversations and 636 hours of simulated data, achieves better performance on the evaluation benchmark than a zero-shot model trained on 2700 hours of Hungarian speech. These findings indicate that LLM-generated conversational data synthesized with TTS is a practical complement to real conversational corpora for speech model training. 

---
# FlashbackCL: Mitigating Temporal Forgetting in Federated Learning 

**Authors**: Mubarak A. Ojewale, Adriana E. Chis, Jorge M. Cortes-Mendoza, Bernardo Pulido-Gaytan, Horacio Gonzalez-Velez  

**Link**: [PDF](https://arxiv.org/pdf/2606.03939)  

**Abstract**: Federated Learning (FL) of foundation and edge models increasingly targets deployments where client data distributions drift over time, yet existing forgetting-mitigation methods assume each client's distribution is stationary. Flashback, the strongest recent FL method against cross-client (spatial) forgetting, uses monotonically accumulating per-class label counts as a knowledge proxy; this proxy becomes miscalibrated under temporal distribution shift and anchors the global model to an outdated class balance. We formalise temporal forgetting in FL with a per-phase metric isolated from protocol-level fluctuations and propose Flashback Continual Learning (FlashbackCL), a drop-in extension of Flashback with (i) temporally-decayed label counts; (ii) a device-aware replay buffer with Class-Balanced Reservoir Sampling (CBRS); and (iii) server-side active coreset curation on the public distillation set. The results show that FlashbackCL achieves 6.9% to 10.0% relative improvement relative to Flashback, on CIFAR-10 with 50 clients and three controlled temporal shift modes, while simultaneously reducing temporal forgetting by up to 68%. A 5-variant ablation identifies CBRS replay as the critical component. FlashbackCL also improves Flashback by 3.5 points on stationary CIFAR-100, suggesting that class-balanced replay regularises spatial heterogeneity as well as temporal shift. 

---
# q0: Primitives for Hyper-Epoch Pretraining 

**Authors**: Bishwas Mandal, Shmuel Berman, Akshay Vegesna, Samip Dahal  

**Link**: [PDF](https://arxiv.org/pdf/2606.03938)  

**Abstract**: Multi-epoch training is becoming the standard now that compute is growing faster than the supply of high-quality text. But pretraining a single model saturates within a few passes, long before the compute budget is exhausted. We argue this calls for a conceptual shift from training a single model toward exploring a population of models and aggregating their predictions. We introduce hyper-epoch pretraining (q0), which turns a multi-epoch budget into a population of diverse models whose combined predictions reach a lower validation loss than a single refined model. q0 reduces to three core primitives. A cyclic schedule with anti-correlated learning rate and weight decay collects diverse models from a few parallel trajectories. Chain distillation trains each model against its predecessor so that model quality compounds across the population. A learned prior, fit on a held out set, selects and weights members for any inference budget. On a 1.8B-parameter model trained on 100M FineWeb tokens, q0 matches a strong 256-epoch ensemble baseline using only ${\sim}56$ epochs (${\sim}4.6\times$ fewer), or ${\sim}67$ epochs (${\sim}3.8\times$ fewer) when matched to the baseline's ensemble size, and continues to improve beyond it. These gains reach cumulative ${\sim}12.9\times$ data efficiency under the Slowrun setting and transfer to downstream benchmarks. Crucially, the optimal allocation shifts with the budget, so we give prescriptive recipes for how to spend a given epoch budget to maximize generalization, from a single epoch up to the largest budgets. 

---
# FFR: Forward-Forward Learning for Regression 

**Authors**: Xinyang Liu, Xuanyu Liang, Shiqi Ding, Boyang Li, Zhiqiang Que, Jiayang Li, Guosheng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03927)  

**Abstract**: The Forward-Forward (FF) algorithm offers a computationally efficient and biologically plausible alternative to backpropagation (BP) by training neural networks through purely local, layer-wise optimization. However, FF is inherently designed for classification via contrastive positive-negative sample pairs, and extending it to regression poses fundamental challenges: continuous target space lack natural "opposites" for contrastive learning, and the standard goodness function carries no information about target magnitude or ordering. We propose FFR (Forward-Forward for Regression), to our knowledge, the first framework to extend FF to real-world regression and demonstrate competitive performance across diverse real-world datasets. FFR introduces three key innovations: (1) an ordinal competitive goodness function that replaces contrastive pairs with competitive learning between partitioned neuron groups under distance-aware ordinal supervision; (2) a stratified ladder architecture where shallow layers learn coarse ordinal discrimination and deeper layers refine into fine-grained regression, with multi-scale feature aggregation for inter-layer collaboration; and (3) hierarchical prediction with uncertainty estimation, where multi-scale predictors jointly provide robust predictions and prediction confidence as a free-lunch. Extensive experimental results show FFR recovers on average 98.6% of BP's accuracy across five real-world regression benchmarks while reducing peak training memory to only 27% of BP's at depth 8 and 8% at depth 32, with per-iteration time around 72% of BP's, and substantially outperforms all BP-free competitors. 

---
# NetKV: Network-Aware Decode Instance Selection for Disaggregated LLM Inference 

**Authors**: Mubarak Adetunji Ojewale  

**Link**: [PDF](https://arxiv.org/pdf/2606.03910)  

**Abstract**: Disaggregated LLM inference forces the KV cache to traverse the datacenter network before decoding begins, so transfer time enters directly into the Time to First Token (TTFT) budget. Current schedulers route on compute load and prefix-cache locality alone, ignoring the topological distance and dynamic congestion between prefill and decode instances. We close this gap with a thin operator-to-scheduler interface, the network cost oracle, and we prove that ignoring the network term renders cache-aware-only scheduling arbitrarily suboptimal as context length grows. NetKV, the O(|D|) per-request greedy that consumes this oracle, has tier rankings that are provably robust to stale telemetry. On a 64-GPU four-tier fat-tree simulator driven by Mooncake traces, NetKV reduces mean TTFT by up to 21.2% over round-robin and 17.6% over a tuned cache+load-aware scheduler, lifts SLO attainment by up to 20.1 percentage points, and keeps the Time Between Tokens overhead below 0.5 ms in every condition tested, with no changes to the transport, inference engine, or hardware. 

---
# The Impact of Configuring Agentic AI Coding Tools on Build-vs-Buy Decisions: A Study Protocol 

**Authors**: Jai Lal Lulla, Matthias Galster, Jie M. Zhang, Sebastian Baltes, Christoph Treude  

**Link**: [PDF](https://arxiv.org/pdf/2606.03907)  

**Abstract**: Agentic AI coding tools write code with increasing autonomy and in doing so decide when to import a library and when to implement functionality from scratch. These decisions, whether to build functionality from scratch or buy into an external library, hereafter build-versus-buy, carry direct consequences for software security, licensing compliance, performance, and long-term maintainability. Yet no controlled experimental study has examined what governs build-versus-buy decisions in agentic AI coding tools. Configuration mechanisms, i.e., the means by which developers tailor agentic AI coding tool behavior to a project or workflow, are one of the primary means by which practitioners can influence these decisions. However, it is unclear which configuration mechanisms influence build-versus-buy decisions most effectively. We present a pre-registered protocol to study how configuration mechanisms alter build-versus-buy behavior in two popular agentic AI coding tools: Claude Code and OpenAI Codex. We will execute controlled programming tasks drawn from a benchmark of staged projects, each constructed around identifiable build-versus-buy points, and will manipulate the configuration supplied to each tool, ranging from no configuration, through context files with soft preferences and explicit prohibitions, to Skills (instructions that can be autonomously discovered), MCP-enabled library discovery tools, and permission controls, measuring which libraries the tool selects, whether it discloses newly introduced libraries, and whether those disclosures are complete and accurate. Nine pre-registered hypotheses structure the protocol. The resulting benchmark dataset and analysis pipeline will be released as a reusable artifact for evaluating build-versus-buy behavior in agentic AI coding tools. 

---
# Agent libOS: A Library-OS-Inspired Runtime for Long-Running, Capability-Controlled LLM Agents 

**Authors**: Yingqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03895)  

**Abstract**: Large language model (LLM) agents are evolving from request-response assistants into long-running software actors: they maintain state across model calls, fork subtasks, wait for external events, request human authority, generate tools, and perform side effects that must be resumed and audited. This paper presents Agent libOS, a library-OS-inspired runtime substrate for LLM agents. Agent libOS runs above a conventional host operating system; it does not implement hardware drivers, kernel-mode isolation, or a POSIX-compatible operating system. Instead, it treats an agent as an AgentProcess: a schedulable execution subject with process identity, parent-child lineage, lifecycle state, a tool table derived from an AgentImage, typed Object Memory, explicit capabilities, human queues, checkpoints, events, and audit records. Its central design rule is tools are libc-like wrappers; runtime primitives are the authority boundary. Filesystem access, object access, sleeps, human approval, JIT tool registration, and external side effects are checked at primitive boundaries under explicit capabilities and policy.
We describe the design, threat model, Python prototype, and safety-oriented evaluation. The current prototype implements async scheduling, namespace-local Object Memory, runtime-integrated human approval, one-shot permission grants, per-process working directories, shell and image-registration primitives, Deno/TypeScript JIT tools over a libOS syscall broker, filesystem/object bridge tools, an injectable Resource Provider Substrate, deterministic demos, real-model smoke scripts, and 123 regression tests at the time of writing. Rather than improving planner accuracy, Agent libOS demonstrates a runtime substrate in which long-running LLM agents can be scheduled, authorized, resumed, and audited without treating tool dispatch as the trust boundary. 

---
# Synthesize and Reward -- Reinforcement Learning for Multi-Step Tool Use in Live Environments 

**Authors**: Ibrahim Abdelaziz, Asim Munawar, Kinjal Basu, Maxwell Crouse, Chulaka Gunasekara, Suneet Katrekar, Pavan Kapanipathi  

**Link**: [PDF](https://arxiv.org/pdf/2606.03892)  

**Abstract**: Training LLMs to orchestrate multi-step tool calls is held back by three coupled obstacles: realistic stateful execution environments are costly to build, synthetic training queries are often detached from the server's actual state (so the generated tool calls fail to execute), and recall-based RL rewards incentivize verbose tool-calling patterns. We present PROVE (Programmatic Rewards On Verified Environments), a framework with three contributions: (1) a library of 20 stateful MCP (Model Context Protocol) servers exposing 343 tools, enabling live-execution RL training with session-scoped state isolation; (2) an automated data synthesis pipeline that generates validated multi-turn tool-call trajectories against these servers via dependency-graph-guided conversation simulation grounded in live-sampled server state, so every generated query references entities that actually exist; and (3) a multi-component programmatic reward - graduated validity scoring, dependency-aware coverage, an adaptive efficiency penalty with a complexity-scaled call budget, a tool-name signal, and an argument-value matching bonus - requiring no external judge model. We train four models (Qwen3-4B, Qwen3-8B, Qwen2.5-7B, Granite-4.1-8B) with GRPO using identical reward hyperparameters and ~13K training examples; only learning rate is tuned per model family from a three-point sweep. On BFCL Multi-Turn, tau2-bench, and T-Eval, PROVE yields improvements of up to +10.2, +6.8, and +6.5 points respectively, demonstrating that a compact programmatic reward yields consistent gains on multi-step tool orchestration across two model families. 

---
# Beyond Encoder Accumulation: Measuring Encoder Roles in Multi-Encoder VLMs 

**Authors**: Wei Ding, Yudong Zhang, Ruobing Xie, Xingwu Sun, Jiansheng Chen, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03879)  

**Abstract**: As foundation models scale toward fusing more heterogeneous visual streams, understanding how diverse encoders interact under joint training becomes a prerequisite for principled design. Yet large vision-language models (LVLMs) currently lack the tools to do so, and parameter-efficient encoder configurations remain hard to identify before training. To re-examine encoder roles under joint training, on the 16-benchmark Cambrian-1 suite we retrain and evaluate all 31 non-empty subsets of five common vision encoders under a unified pipeline (~20k GPU-hours total), and report three findings. First, retraining each subset from scratch reveals encoder rankings that differ from those obtained by masking encoders on a fixed checkpoint, including which encoder ranks first overall. Second, we decompose each encoder's contribution into two axes, Capacity, the score an encoder reaches on its own, and Necessity, the drop when it is removed from the full pool. The two axes are not interchangeable. Pairing the two highest-Capacity encoders is suboptimal, while pairing a high-Capacity anchor with an adaptive complement matches the full five-encoder model. Adding further encoders beyond this pair yields only marginal gains. Third, at fixed parameter count, per-encoder pre-projector effective rank explains the residual score variation. The strongest pairs combine an anchor whose rank survives joint training with a complement whose rank expands under it, suggesting that higher-rank, less-collapsed projector inputs correspond to a more favorable optimization regime at the encoder-projector interface. Together, the Capacity-Necessity decomposition and the pre-projector rank analysis, along with comprehensive evaluation through retraining, expose a methodological gap in multi-encoder LVLM design, and offer concrete primitives for closing it. 

---
# From 'What' to 'How' and 'Why': Sharing LLM-Generated Retrospective Summaries of Older Adults' Passive Tracking Data with Remote Family Members 

**Authors**: Jiachen Li, Reina Szeyi Chan, Akshat Choube, Xiang Zhi Tan, Elizabeth Mynatt, Varun Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2606.03876)  

**Abstract**: With the growing prevalence of modern ubiquitous computing technologies, multi-modal tracking systems hold promise for providing timely awareness and reassurance to stakeholders such as remote family members (RFMs) of older adults, who play a central role in care coordination. However, combining heterogeneous data streams into high-level, meaningful content - such as retrospective summaries - remains challenging. While recent work has demonstrated the promise of large language models (LLMs) for interpreting multi-modal tracking data, less attention has been given to generating narrative accounts for stakeholders like RFMs, who possess rich personal knowledge of older adults and strong emotional responsibility, yet have limited visibility into their daily lives and limited capacity for caregiving. In this work, we explore how LLMs can be used to generate retrospective summaries from multi-modal tracking data for RFMs of older adults. We leveraged and customized an existing system, Vital Insight, to generate initial summaries on different dates and data availability scenarios as technology probes, and conducted interviews with 11 RFMs to gather feedback. Based on these insights, we redesigned the system into a multi-layer, multi-agent, insight-driven summary approach that builds from objective statistics and descriptions to enriched, context-aware narratives. We then compared the redesigned summaries with the initial versions through a survey with the same 11 RFMs and found significant improvements in satisfaction, perceived helpfulness, trust, and willingness to receive the summaries. We conclude by presenting design implications for AI-generated summaries for RFMs and broader contexts, emphasizing the need to support RFMs' sensemaking shift from simply presenting ''What'' data were collected, to explaining ''How'' is my loved one doing and ''Why''. 

---
# A Training-Free Mixture-of-Agents Framework for Multi-Document Summarization using LLMs and Knowledge Graphs 

**Authors**: Cuong Vuong Tuan, Trang Mai Xuan, Tien-Cuong Nguyen, Vu-Duc Ngo, Thien Van Luong  

**Link**: [PDF](https://arxiv.org/pdf/2606.03867)  

**Abstract**: Multi-Document Summarization (MDS) plays a critical role in distilling essential information from collections of textual data. Existing approaches often struggle to capture complex inter-document relationships, rely heavily on large amounts of labeled data for supervised training, or exhibit limited generalization across domains and languages. To address these limitations, we present a training-free mixture-of-agents framework for MDS that leverages the complementary strengths of large language models (LLMs) and knowledge graphs. Our approach decomposes summarization into specialized agent tasks: extractive selection, knowledge-aware abstraction, and iterative refinement, each operating without task-specific fine-tuning. We unify their outputs using a multi-perspective consistency mechanism guided by LLMs. Experiments across four datasets in English and Vietnamese demonstrate state-of-the-art or competitive performance, validating the effectiveness and adaptability of our modular design. 

---
# Taiji: Pareto Optimal Policy Optimization with Semantics-IDs Trade-off for Industrial LLM-Enhanced Recommendation 

**Authors**: Yuecheng Li, Zeyu Song, Jing Yao, Chi Lu, Peng Jiang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2606.03866)  

**Abstract**: Scaling recommender systems via large language models (LLMs) has become a prominent trend in the industry. However, aligning the LLM's semantic space with the recommender's ID space via post-training (e.g., SFT and RL) remains challenging. Existing LLM4Rec paradigms are bottlenecked by two main issues: (1) the difficulty of measuring and improving chain-of-thought (CoT) quality in open-domain recommendation during SFT, and (2) the neglect of the trade-off between LLM semantic rewards and recommendation preference rewards during RL alignment. Inspired by these challenges, we present Taiji, a novel LLM-as-Enhancer framework designed for industrial recommender systems. To overcome the SFT bottleneck, we utilize reverse-engineered reasoning and open-ended rejection sampling to generate high-quality, domain-specific CoT data. To resolve the RL alignment issue, we propose Pareto Optimal Policy Optimization (POPO), which adaptively adjusts cross-domain reward weights. Theoretically, it achieves an optimal trade-off between the semantic world knowledge of LLMs and the collaborative ID features representing online user preferences. Extensive offline evaluations and online A/B tests validate the effectiveness of Taiji. Deployed on Kuaishou's advertising platform since May 2026, Taiji currently serves over 400 million users daily, yielding significant commercial revenue and demonstrating its robust scalability in web-scale environments. 

---
# FLARE: Fine-Grained Diagnostic Feedback for LLM Code Refinement 

**Authors**: Yinsheng Yao, Hongxiang Zhang, Weixi Tong, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03852)  

**Abstract**: Large language models often generate code with bugs. Existing methods rely on feedback signals such as test failures and self-critiques to iteratively refine the generated code. Such signals are either too coarse-grained or too high-level, which is not sufficient to inform the model where to fix the bug. In this work, we present Flare, an iterative framework with a lightweight diagnostic model that predicts line-level suspiciousness signals for bug localization and code refinement. Given the inherent uncertainty of diagnostic predictions, Flare searches over the top-k suspicious regions and selects the best candidate according to execution outcomes. Experiments on LiveCodeBench and BigCodeBench with five base LLMs show that, even without candidate search (k=1), Flare outperforms the strongest baseline with an absolute improvement from 1.72% to 7.42%. Furthermore, searching over 10 candidates yields an average improvement of 8.50% compared with no candidate search. When evaluated in isolation, our lightweight diagnostic model achieves the best performance compared with recent fault localization methods, demonstrating that it can provide reliable fine-grained guidance for code refinement. 

---
# Clustered Self-Assessment: A Simple yet Effective Method for Uncertainty Quantification in Large Language Models 

**Authors**: Qi Cao, Takeshi Kojima, Andrew Gambardella, Helinyi Peng, Yutaka Matsuo, Yusuke Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2606.03846)  

**Abstract**: Large language models (LLMs) demonstrate remarkable performance across diverse tasks, but they often generate responses that appear plausible while being factually incorrect. This problem is compounded by the lack of explicit uncertainty estimates, which makes it difficult for users to judge the reliability of model outputs. Existing uncertainty quantification methods typically rely on indirect signals, such as entropy across sampled generations. These signals can be difficult to interpret and do not fully leverage the model's ability to assess its own uncertainty. We propose a simple yet effective self-assessment method for uncertainty quantification in LLMs. Our approach groups sampled generations into semantically distinct clusters, converts them into answer options in a structured multiple-choice question, and uses the probability assigned by the LLM to each option as a confidence estimate. Experiments across multiple models and datasets show that our method consistently outperforms baseline approaches. Notably, it achieves competitive performance with as few as two additional samples, demonstrating both its effectiveness and efficiency. 

---
# Re-Evaluating Continual Learning with Few-Shot Adaptation 

**Authors**: Amogh Inamdar, Matthew So, Vici Milenia, Richard Zemel  

**Link**: [PDF](https://arxiv.org/pdf/2606.03843)  

**Abstract**: Continual learning methods aim to maximize the stability and plasticity of machine learning models that are trained on a sequence of tasks. The standard measure of stability (i.e., forgetting) is the 0-shot performance of a model on previously learned tasks, and plasticity, the performance on the most recently learned task. However, 0-shot evaluation does not fully measure a model or method's ability to retain learned information or adapt quickly to new information, as it requires perfect recall across multiple tasks. In this paper, we propose few-shot evaluation as a more comprehensive assessment of the stability and plasticity of a continual learning system. We conduct a fine-grained assessment on task sequences for continual image classification and find that this paradigm produces novel insights into the performance of popular continual learning strategies. Through few-shot evaluation with a novel metric -- per-shot plasticity -- we show that adding `foresight' to continual learning methods via the meta-learning of a short sequence of future tasks induces learning-to-learn behavior over the task sequence. 

---
# Conditional Latent Diffusion Model with Fourier-based Motion Modelling for Virtual Population Synthesis 

**Authors**: Shaokun Lan, Haoran Dou, Jinghan Huang, Arezoo Zakeri, Fengming Lin, Zherui Zhou, Jinming Duan, Alejandro F. Frangi  

**Link**: [PDF](https://arxiv.org/pdf/2606.03827)  

**Abstract**: In-silico trials of medical devices require the generation of virtual populations of anatomies. In cardiovascular applications, virtual anatomy is typically represented as a 3D+t mesh sampled from a generative model. However, most existing mesh generators focus on static anatomy, while sequence models often lack explicit periodicity. To this end, we propose 4D F-MeshLDM, a conditional generative framework comprising a convolutional mesh VAE to encode meshes, a structural latent space that parameterises motion using a truncated Fourier series, and a diffusion prior that learns the latent distribution over Fourier coefficient tokens. By conditioning the diffusion process on clinical covariates via affine modulation, we enable controllable synthesis. Sampling tokens and performing inverse Fourier synthesis yield cycle-consistent latent trajectories, which can be decoded into 3D+t cardiac mesh sequences. Experiments on 5,000 UK Biobank subjects demonstrate that 4D F-MeshLDM outperforms state-of-the-art baselines in anatomical fidelity and achieves near-zero cycle closure error. Furthermore, the generated cohorts accurately preserve clinical functional indices, highlighting the potential of our framework for reliable in-silico cardiac trials. 

---
# AI Agents Enable Adaptive Computer Worms 

**Authors**: Jonas Guan, Tom Blanchard, Hanna Foerster, Hengrui Jia, Gabriel Huang, Nicolas Papernot  

**Link**: [PDF](https://arxiv.org/pdf/2606.03811)  

**Abstract**: A computer worm is malware that spreads on a network by replicating itself from one machine to another. Traditional worms, like WannaCry, exploited predetermined vulnerabilities, and their spread can be halted by patching those vulnerabilities. Here we show that artificial intelligence (AI) agents enable a fundamentally new threat: a worm that generates tailored attack strategies to each target it encounters. The worm parasitically uses compromised machines to run open-weight large language models (LLMs) to sustain its reasoning, or extend its reach for further attacks. Deployed on a network of machines spanning Linux, Windows, and IoT (Internet of Things) devices, the worm propagated by exploiting common, real-world corporate network vulnerabilities. Since the worm is powered by stolen compute, the attacker's marginal cost per new infection is zero. This creates a destabilizing economic asymmetry between attackers and defenders. Moreover, because the worm requires no commercial AI platform, centralized safety controls, such as service refusals or rate limiting, are structurally irrelevant. Our results demonstrate that self-sustaining AI-driven cyber-threats are no longer theoretical. We must prepare for autonomous generative adversaries: malware systems that propagate without human operators and are defined not by fixed exploit code, but by the capacity to reason about targets, adapt to observations, and synthesize attack logic in real time. 

---
# Consistency Training Can Entrench Misalignment 

**Authors**: David Demitri Africa, Arathi Mani  

**Link**: [PDF](https://arxiv.org/pdf/2606.03810)  

**Abstract**: Consistency training encourages a model to produce similar outputs across related inputs or sampling procedures. Such methods are simple, scalable, and largely label-free, but their effects on model alignment remain poorly understood. Could the self-bootstrapping nature of these methods amplify undesired behavior in models? We test seven consistency training methods on 108 ``model organisms: open-source models (7B--70B) fine-tuned to exhibit various forms of controlled misaligned behavior. We find that outcomes vary significantly: consistency training generally suppresses reward hacking and emergent misalignment but amplifies sycophancy. We present evidence that distribution shifts induced by the consistency labeling process, rather than variation in the selection operators, may be the primary driver of systematic alignment effects. Finally, we present a unifying theoretical framework to derive conditions under which consistency training will amplify or suppress misalignment. In total, our study establishes that consistency training is not alignment-neutral, and that its use in critical systems should be carefully audited. 

---
# PURGE: Projected Unlearning via Retain-Guided Erasure 

**Authors**: Vedant Jawandhia, Daksh Ahuja, Ghufran Alam Siddiqui, Prashant Trivedi, Yash Sinha, Pratik Narang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03808)  

**Abstract**: We propose PURGE, a machine unlearning algorithm built on a simple but an under-exploited observation: continual learning (CL) and machine unlearning (MU) which are fundamentally dual problems. CL tries to learn new tasks without forgetting old ones; MU tries to erase specific data without hurting retained performance representing the same underlying tension in opposite directions. PURGE leverages this duality by adapting gradient projection from A-GEM (Chaudhry et al., 2019) so that every unlearning step is constrained to not increase the retain-set loss. On top of this, it performs multi-layer representation erasure, pushing forget-set activations in intermediate layers towards the retain distribution to remove information from hidden representations rather than just suppressing it at the output. A key design choice is the retain-confusion target: rather than pushing forget outputs toward the uniform distribution, which we found to be surprisingly easy for membership inference attacks to detect, we instead target the model's natural confusion pattern on retain data. This makes the unlearned model hard to distinguish from one retrained from scratch. Two self-regulating stopping criteria (a retain-loss budget and a forget-accuracy target) let the algorithm decide on its own when to stop, removing the need for manual epoch tuning. In experiments on five datasets (CIFAR-10, MNIST, SVHN, STL10, PathMNIST) across 22 class-level forgetting tasks, PURGE consistently keeps retain accuracy above 96% while achieving MIA AUROC close to 0.5 (the ideal), outperforming gradient ascent, KL-uniform, and several published baselines on the privacy-utility frontier. 

---
# LiveBand: Live Accompaniment Generation in the Audio Domain 

**Authors**: Marco Pasini, Javier Nistal, Mathias Rose Bjare, Stefan Lattner, George Fazekas  

**Link**: [PDF](https://arxiv.org/pdf/2606.03803)  

**Abstract**: We present LiveBand, a real-time system that generates high-fidelity music accompaniments to live audio input, respecting strict causal constraints. Our method trains a causal transformer generator in the continuous latent space of a pre-trained causal audio autoencoder, using adversarial sequence-level supervision from a discriminator. At each timestep, the generator receives only the causally available mix context and Gaussian noise, and predicts accompaniment latents without access to future mix frames or ground-truth target latents. Training is performed in a single parallel forward pass under causal masking, while streaming inference proceeds autoregressively with a rolling attention state. The model's training and inference computations are matched by design, eliminating teacher forcing and the associated exposure bias. On a multi-instrument music accompaniment benchmark, LiveBand improves over prior work on objective measures of audio quality, beat alignment, and mix adherence, while enabling real-time streaming generation without lookahead into the future on consumer hardware. 

---
# Trading Human Curation for Synthetic Augmentation in RLVR 

**Authors**: Akshansh <last>, Leonardo Rosa Rodrigues, Michael Korostelev, Youssef Hassan, Mark E. Whiting  

**Link**: [PDF](https://arxiv.org/pdf/2606.03800)  

**Abstract**: The supply of high-quality training tasks is a central bottleneck for reinforcement learning from verifiable rewards (RLVR) on agentic language models. Each task requires a sandboxed setup, a prompt, and a hand-authored reward function, and only tasks that pass a quality bar produce useful training signal. Hand-curation at this quality bar does not scale economically to the task counts effective RL training requires, and the substitution rate between automatically generated task variants and human-authored ones is not yet established. We investigate using pre-specified, gate-filtered augmentations of a small hand-authored base as a substitute for additional human curation during RLVR. We formalize the cost-adjusted trade rate $\rho_{\text{cost}}$ between augmented and human-authored tasks, measure it through a controlled ablation across training corpora with varying augmentation share, and characterize the end-to-end economics of the augmentation pipeline. Substituting augmented content for additional human-authored tasks retains aggregate held-out generalization on a ten-benchmark suite spanning code, instruction following, reasoning, and multi-turn agentic function-calling. The cost-adjusted trade rate $\rho_{\text{cost}}$ between gated synthetic and human-authored RLVR tasks stays in $[1.4\times, 11.6\times]$ across the plausible $c_{\text{human}}/c_{\text{aug}}$ range. 

---
# Signed Spiking Neuron Enabled by an Orthogonal-Easy-Axis Magnetic Tunnel Junction 

**Authors**: Huannan Zheng, Jingli Liu, Kezhou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03796)  

**Abstract**: Signed spiking neurons carry richer information than standard spiking neurons. This work proposes a compact magnetic tunnel junction (MTJ)-based neuron for signed leaky integrate-and-fire (LIF) operation. With orthogonal easy axes in the free and pinned layers, the device enables bipolar spike generation and maps magnetic-moment dynamics to signed LIF membrane-potential evolution. Landau--Lifshitz--Gilbert simulations show that proper free-layer dimensions allow the device response to follow a signed LIF equation. A representative design of 10 nm x 45 nm x 50 nm corresponds to an aspect ratio of about 2:9:10. Network evaluations using the fitted device-neuron model achieve 91.06% on CIFAR-10 and 77.40% on CIFAR10-DVS, retaining most of the accuracy of ideal signed LIF neurons. 

---
# E2LLM: Towards Efficient LLM Serving in Heterogeneous Edge/Fog Environments 

**Authors**: Truong-Thanh Le, Amir Taherkordi, Hoang-Loc La, Frank Eliassen, Phuong Hoai Ha, Peiyuan Guan  

**Link**: [PDF](https://arxiv.org/pdf/2606.03770)  

**Abstract**: Large Language Models (LLMs) have become integral to modern applications, yet their deployment remains challenging. Beyond executing the models themselves, practical deployment must address cost efficiency, low latency, and optimal resource utilization. Conventional approaches typically assume that an entire model can be hosted on a single device, which does not hold in many real-world scenarios, particularly in Edge and Fog environments where device resources are constrained. In this paper, we introduce E2LLM, a framework designed to enable efficient LLM deployment in such resource limited settings. Rather than simply partitioning a single model across all available devices, E2LLM replicates the full model across multiple groups of devices (replicas) and applies model parallelism within each replica. Each replica is assigned a specialized role PREFILL or DECODER based on its efficiency in handling input and output tokens. This separation leverages the inherent differences between these two phases of LLM inference. To effectively organize devices, we utilize a Genetic Algorithm to form clusters that maximize system performance. Within each cluster, we apply Dynamic Programming to determine an optimal partitioning strategy that minimizes bottlenecks in model-parallel execution. Experimental results demonstrate that our approach adapts robustly to varying workloads, including scenarios with significant variation in input and output token lengths. Compared to the Splitwise baseline, E2LLM reduces average waiting time by over 50% under high-demand conditions 

---
# Merit or networks? What decides where research is published 

**Authors**: Ning Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.03763)  

**Abstract**: Does scientific publishing reward the quality of ideas or the advantage of connections? The question is universal to prestige-driven science, yet it has resisted decades of study because a paper's quality could not be gauged ahead of its publication fate without using that fate as the yardstick. We break this constraint by measuring a paper's idea quality directly from its text, before publication, using a discipline-trained LLM evaluator that scores the idea without seeing author names or outcomes. Using economics as a case study, we combine this text-legible idea-quality score with an execution-quality rubric, a connection index, an author-ability index, and an off-the-shelf language-model text score to estimate a five-input production function for journal placement across 6,208 economics working papers. The inputs are not rivals but a sequence along the ladder of prestige. Execution sets a meritocratic floor and is the largest input overall. Text-legible idea quality grades the rungs in between. Connections set a favoritism ceiling that bites mainly near the apex, the most selective journals. Connections work through two additive channels: connected authors write papers that score higher, and at equal scores their papers are still more likely to place better. Yet this advantage is bounded. Connections raise the odds of every rung without making the apex the typical outcome for ordinary ideas, and even the highest-scoring papers face real friction reaching the visible journal ladder. The result nests, rather than chooses between, the meritocracy and network accounts of how science is published. 

---
# Tool-Aware Optimization with Entropy Guidance for Efficient Agentic Reinforcement Learning 

**Authors**: Hongye Cao, Nuo Yan, Haoyuan Deng, Ziwei Wang, Tianpei Yang, Jing Huo, Yuyao Zhang, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.03762)  

**Abstract**: Agentic reinforcement learning (RL) equips large language models (LLMs) with tool-use capabilities that substantially improve reasoning on complex tasks. However, integrating external tools often destabilizes training: over-reliance on tools can induce input distribution shift, while overly conservative tool use limits effective exploration. To address this issue, we propose a unified framework TAO-RL that couples tool-aware trajectory filtering with entropy-guided exploration for efficient policy optimization. Specifically, at the data level, TAO-RL filters rollout trajectories along two criteria: discarding those where all tool invocations fail to execute, and removing those where all rollouts are either correct or incorrect, as both cases yield degenerate advantage estimates that contribute no discriminative learning signal. This joint filtering retains data that are both tool-capable and informative, establishing a high-quality training distribution. At the algorithmic level, we introduce a tool-aware entropy-guided bonus that reshapes the advantage function at post-tool-call tokens, encouraging the policy to explore more diverse reasoning paths at critical decision points. These two components are mutually reinforcing: trajectory filtering establishes a clean and informative training foundation, while entropy-guided exploration drives stronger reasoning behaviors at critical tool-interaction junctures. Extensive experiments on 7 challenging reasoning benchmarks across 3 model scales demonstrate the superiority of TAO-RL over existing methods. 

---
# Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models 

**Authors**: Glenn Jocher, Jing Qiu, Mengyu Liu, Shuai Lyu, Fatih Cagatay Akyon, Muhammet Esat Kalfaoglu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03748)  

**Abstract**: Real-time vision demands models that are accurate, efficient, and simple to deploy across diverse hardware. The YOLO family has become widely deployed for this reason, yet most YOLO detectors still rely on non-maximum suppression at inference, carry heavy detection heads due to Distribution Focal Loss, require long training schedules, and can leave the smallest objects without positive label assignments. We present Ultralytics YOLO26, a unified real-time vision model family that addresses these limitations through coordinated architecture and training advances. YOLO26 uses a dual-head design for native NMS-free end-to-end inference and removes DFL entirely, yielding a lighter head with unconstrained regression range. Its training pipeline combines MuSGD, a hybrid Muon-SGD optimizer adapted from large language model training; Progressive Loss, which shifts supervision toward the inference-time head; and STAL, a label assignment strategy that guarantees positive coverage for small objects. Beyond detection, YOLO26 introduces task-specific head and loss designs for instance segmentation, pose estimation, and oriented detection, producing consistent gains across tasks and scales. The family spans five scales (n/s/m/l/x) and supports detection, instance segmentation, pose estimation, classification, and oriented detection in a single pipeline, with an open-vocabulary extension, YOLOE-26, for text-, visual-, and prompt-free inference. Across all scales, YOLO26 achieves 40.9-57.5 mAP on COCO at 1.7-11.8 ms T4 TensorRT latency, advancing the accuracy-latency Pareto front over prior real-time detectors, while YOLOE-26x reaches 40.6 AP on LVIS minival under text prompting. Code and models are available at this https URL. 

---
# Qwen-Image-Flash: Beyond Objective Design 

**Authors**: Tianhe Wu, Kun Yan, Zikai Zhou, Lihan Jiang, Jiahao Li, Jie Zhang, Kaiyuan Gao, Ningyuan Tang, Shengming Yin, Xiaoyue Chen, Xiao Xu, Yilei Chen, Yuxiang Chen, Yan Shu, Yixian Xu, Yanran Zhang, Zihao Liu, Zhendong Wang, Zekai Zhang, Deqing Li, Liang Peng, Yi Wang, Jingren Zhou, Chenfei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03746)  

**Abstract**: Few-step distillation has become an effective strategy for accelerating advanced visual generative models, yet prior work has largely focused on distillation objectives. In this work, we revisit few-step distillation from a complementary perspective, focusing on the training recipe that critically shapes student performance. Using Qwen-Image-2.0 as a representative case, we systematically investigate three factors in unified text-to-image generation and instruction-guided image editing distillation: data composition, teacher guidance, and task mixture. Our empirical analysis reveals several non-obvious behaviors, which motivate the development of Qwen-Image-Flash. Overall, our results suggest that effective few-step distillation requires not only carefully designed objectives, but also principled organization of the broader training pipeline. 

---
# Staying Alive: Uncensored Survival Analysis with Tabular Foundation Models 

**Authors**: Mariana Vargas Vieyra  

**Link**: [PDF](https://arxiv.org/pdf/2606.03689)  

**Abstract**: Survival Analysis (SA) is a statistical framework that models the time span until some event of interest occurs. Widely used in several domains, including healthcare and churn prediction, a central challenge in its applicability stems from the time of the event being partially observed or \emph{right-censoring}.
Tabular Foundation Models (TFM) have attracted significant interest in recent years due to their ability to perform prediction tasks in a single forward pass, requiring no dataset-specific parameter fitting. Despite their success, their application to prediction tasks on time-to-event data remains difficult due to right censoring. In this work, we present a training-free method to survival regression by leveraging TFMs to both predict the time of the event and iteratively impute right-censored data.
Our method uses a TFM to construct an Accelerated Failure Time (AFT) model requiring no training beyond fitting a single scalar parameter. Subsequently, by building on the Buckley-James estimator, we introduce a non-parametric in-context estimator for right-censored data. Our experiments on standard survival analysis benchmarks show that our method is competitive with several parametric and semi-parametric survival regression models that require training, including Cox regression and parametric AFT models. 

---
# A Close Look At World Model Recovery In Supervised Fine-Tuned LLM Planners 

**Authors**: Patrick Emami, Nan Qiang, Peter Graf  

**Link**: [PDF](https://arxiv.org/pdf/2606.03685)  

**Abstract**: Supervised fine-tuning (SFT) improves end-to-end classical planning in large language models (LLMs), but do these models also learn to represent and reason about the planning problems they are solving? Due to the relative complexity of classical planning problems and the challenge that end-to-end plan generation poses for LLMs, it has been difficult to explore this question. In our work, we devise and perform a series of interpretability experiments that holistically interrogate world model recovery by examining both internal representations and generative capabilities of fine-tuned LLMs. We find that: a) Supervised fine-tuning on valid action sequences enables LLMs to linearly encode action validity and some state predicates. b) Models that struggle to use output probabilities for classifying action validity may still learn internal representations that separate valid from invalid actions. c) Broader state space coverage during fine-tuning, such as from random walk data, yields more accurate recovery of the underlying world model. In summary, this work contributes a recipe for applying interpretability techniques to planning LLMs and generates insights that shed light on open questions about how knowledge is represented in LLMs. 

---
# AUGUSTE: Online-Learning dApp for Predictive URLLC Scheduling 

**Authors**: Maxime Elkael, Michele Polese, Yunseong Lee, Koichiro Furueda, Tommaso Melodia  

**Link**: [PDF](https://arxiv.org/pdf/2606.03664)  

**Abstract**: Ultra Reliable and Low Latency Communications (URLLC) was one of the main motivations behind 5G, with 3GPP advertising 1-10 ms latency targets for applications such as industrial automation, Vehicle-To-Everything (V2X), tactical edge networking, and unmanned-system control. Years on, real 5G Time Division Duplexing (TDD) networks still show median Uplink (UL) round-trip times in the 50-70 ms range, largely because of the Scheduling Request (SR) procedure that a User Equipment (UE) must complete before transmitting UL data. Existing remedies, primarily Configured Grant (CG) scheduling, only eliminate this overhead for strictly periodic traffic and require cross-layer synchronization, which has limited their adoption. We propose AUGUSTE (Anticipatory Uplink Grants for URLLC via Self-Adapting Temporal Estimation), a learning-based Medium Access Control (MAC) scheduling framework that embeds online Machine Learning (ML) models in the UL scheduler to predict packet arrivals and proactively allocate resources before an SR is issued. An adaptive state machine alternates between a learning phase that collects unbiased arrival statistics and a confident phase that exploits the learned predictions to schedule only when traffic is expected. We evaluate AUGUSTE on a real 5G testbed running OpenAirInterface across three URLLC traffic patterns (request-response, ML edge inference, and periodic autonomous reporting), and show that it operates at the best achievable point on the latency-overhead trade-off: it matches always-on scheduling's median Round Trip Time (RTT) (around 10 ms, halving the 20 ms SR-based baseline) at roughly one-tenth its resource cost (7-10 percent overhead). 

---
# CoEval: Ranking Language Models for Custom Tasks Without Labeled Data or Trustworthy Benchmarks 

**Authors**: Alexander Apartsin, Yehudit Aperstein  

**Link**: [PDF](https://arxiv.org/pdf/2606.03650)  

**Abstract**: Choosing or ranking language models for a specific application is hardest when no task-specific labeled data exists, and standard public benchmarks cannot be trusted, their items having likely leaked into pretraining, so scores reflect memorization rather than fitness. We present CoEval, an open-source, reusable framework that closes this gap end to end: from only a description of a task or domain, teacher models synthesize a fresh, attribute-controlled benchmark with no human labels, contamination-free because items are generated anew on each run, and a cross-family judge ensemble ranks candidate models with no human raters. Validated where ground truth exists, CoEval recovers the true model ranking and tracks ground-truth correctness at ho=0.86. The label-free judging needs no human calibration because judge-panel composition (vendor diversity), not size, drives reliability: a small, well-chosen cross-family panel is most reliable, while a single judge can be anti-correlated with ground truth (judge-choice regret 0.35) and the ensemble never is. Generated items show zero verbatim 13-gram overlap with five major public benchmarks; the panel cancels verbosity bias and precludes same-family self-preference. A four-task study produced 7,978 evaluations for USD 5.89. The same declarative pipeline applies to any domain and is cheap enough to re-run on every model release: a label-free, contamination-free leaderboard any team can regenerate for its own application. 

---
# Safety Measurements for Fine-tuned LLMs Should be Grounded in Capability 

**Authors**: Krishnapriya Vishnubhotla, Hillary Dawkins, Isar Nejadgholi, Svetlana Kiritchenko  

**Link**: [PDF](https://arxiv.org/pdf/2606.03648)  

**Abstract**: Adapting foundation large language models to a user's task or preferred style through fine-tuning can result in compromising the model's safety. Previous works examined the effects of fine-tuning on model safety in limited and seemingly random experimental settings. We argue that anchoring fine-tuning to a specific capability goal is essential for avoiding arbitrary empirical choices, allowing us to draw meaningful conclusions about safety impacts, and to compare mitigation methods on a consistent basis. We conduct a multi-dimensional evaluation of the effects of fine-tuning on model behavior by focusing on capability as well as safety. Our results surface important issues that (1) fine-tuned models can produce incoherent generations in response to safety prompts, (2) automated safety judgments are unreliable for such incoherent outputs, and (3) the conclusions about the effects of fine-tuning can change depending on the choice of safety benchmark as well as the safety evaluator. 

---
# Black-box, Adaptive, Efficient, Transferable, Harmful, Applicable... Attacks Are All You Need to Break LLMs 

**Authors**: Vincent Limbach, Jonas Dornbusch, David Lüdke, Stephan Günnemann, Leo Schwinn  

**Link**: [PDF](https://arxiv.org/pdf/2606.03647)  

**Abstract**: Accurately evaluating adversarial robustness is a longstanding challenge. A flawed attack design can inflate robustness estimates, making deployment risk assessment and defense comparison unreliable. Historically, standardized attacks such as AutoAttack have largely resolved this for image classifiers, providing a reliable evaluation baseline for systematic comparison across defenses. However, no equivalent exists for LLM jailbreak evaluation yet, where designing such an attack is considerably more difficult. A reliable attack must, among other things, be black-box compatible, applicable to arbitrary defense pipelines, and efficient, which no existing method jointly satisfies. We introduce Indirect Harm Optimization (IHO), a masked diffusion language model attacker trained via iterative preference optimization against a harmfulness judge, requiring only black-box access to the target. The same method can be used without modification as a strong adaptive attack on individual behaviors, or as an efficient amortized policy that transfers to held-out behaviors and unseen target models without fine-tuning. Even against layered defenses, such as a Circuit Breaker-trained model combined with an auxiliary detector, IHO improves attack success considerably over state-of-the-art approaches, without any defense-specific adaptation. Our results position IHO as a practical step toward the kind of standardized jailbreak evaluation that has improved reliability in the past. Code and models are available on GitHub and Hugging Face. 

---
# The Shape of Addition: Geometric Structures of Arithmetic in Large Language Models 

**Authors**: Liuyuan Wen, Xun Zhu, Lihao Huang, Wenbin Li, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.03645)  

**Abstract**: Large Language Models exhibit paradoxical fragility in fundamental arithmetic, implying a disconnect between internal computation and discrete output. By analyzing the residual stream geometry during multi-operand addition, we identify the Iso-Raw-Sum Trajectory (IRST), a geometric structure where representations are anchored by semantic digits and modulated by continuous carry fibers. We propose the Noisy Quantization Model to explain this geometry, framing arithmetic errors as Geometric Slippages caused by internal neural noise pushing a continuous, latent Carry Potential across quantization thresholds. This geometric framework further elucidates Probe Versatility, explaining how lightweight probes can disentangle coexisting latent signals (such as ground truth versus hallucination) from a single activation vector. Finally, we validate these insights through a geometric consistency check method that effectively detects and corrects these quantization failures during inference. Our code is available at this https URL. 

---
# VidMsg: A Benchmark for Implicit Message Inference in Short Videos 

**Authors**: Issar Tzachor, Michael Green, Rami Ben-Ari  

**Link**: [PDF](https://arxiv.org/pdf/2606.03635)  

**Abstract**: Understanding short online videos involves more than identifying visible objects and actions; video makers often include an underlying message or purpose in the clip. We introduce VidMsg, a benchmark for evaluating implicit message understanding in short, internet-native video clips. VidMsg contains 400 YouTube-derived clips across 9 practical topic areas and 52 fine-grained target messages, covering domains such as career and finance, education, health and well-being, culture, safety, sustainability, and lifestyle. VidMsg is constructed through a message-first pipeline: an LLM first translates target messages into indirect search scenarios, which are used to retrieve candidate clips. Human annotators then retain clips that convey the intended message without being overly explicit. VidMsg is designed primarily for bidirectional message-clip retrieval for scalable applications such as video search and recommendation, where systems must capture holistic video understanding. In addition to retrieval, VidMsg includes a diagnostic multiple-choice QA benchmark, where models select the intended message of a clip from semantically related alternatives. Experiments with contemporary video-language and retrieval models show that strong models often fail on VidMsg, because the task requires pragmatic inference, integration of contextual cues, and discrimination among semantically close messages. We also introduce VidVec-Msg, a baseline method that improves message-oriented retrieval while leaving substantial headroom for future work. 

---
# AnchorMoE: Interpretable Time Series Classification via Anchor-Routed MoE 

**Authors**: Tao Xie, Zexi Tan, Haoyi Xiao, Mengke Li, Yiqun Zhang, Yang Lu, Cuie Yang, Yiu-ming Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2606.03631)  

**Abstract**: Multivariate time series classification (MTSC) is pivotal in high-stakes domains, such as clinical diagnosis and industrial fault detection, where safe deployment necessitates transparent decision-making. However, isolating the temporal segments that drive model predictions is challenging because discriminative signals in real-world time series are typically sparse, heterogeneous, and heavily obscured by background noise. This paper, therefore, proposes AnchorMoE, an interpretable-by-construction classification framework. Built upon a Mixture-of-Experts (MoE) architecture, AnchorMoE encodes multi-view representations of local patches and routes them to specialized experts, ensuring that the final prediction is formulated as an exact additive decomposition over the input segments, facilitating ante-hoc transparency rather than relying on post-hoc estimations. To maintain the reliability of this decomposition under sparse signal distributions, we introduce a geometric orthogonality constraint that penalizes representational redundancy, compelling distinct experts to specialize in heterogeneous predictive patterns. Furthermore, an uncertainty-aware reliability gate is designed to dynamically calibrate the contribution of each segment, effectively suppressing residual background noise. Extensive experiments on real-world and synthetic benchmarks demonstrate that AnchorMoE achieves highly competitive classification performance while faithfully grounding its decisions in the raw time series. 

---
# Building Reliable Long-Form Generation via Hallucination Rejection Sampling 

**Authors**: Lin Li, Georgia Channing, Suhaas M Bhat, Gabriel Davis Jones, Yarin Gal  

**Link**: [PDF](https://arxiv.org/pdf/2606.03628)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in open-ended text generation, yet they remain prone to hallucinating incorrect or unsupported content, which undermines their reliability. This issue is exacerbated in long-form generation due to hallucination snowballing, a phenomenon where early errors propagate and compound into subsequent outputs. To address this challenge, we propose a novel inference-time hallucination mitigation framework, named Segment-wise HAllucination Rejection Sampling (SHARS), which uses an arbitrary hallucination detector to identify and reject hallucinated segments during generation and resample until faithful content is produced. By retaining only confident information and building subsequent generations upon it, the framework mitigates hallucination accumulation and enhances factual consistency. To instantiate this framework, we adopt semantic uncertainty as the detector and introduce several vital modifications to address its limitations and better adapt it to long-form text. Our method enables models to self-correct hallucinations without requiring external resources such as web search or knowledge bases, while remaining compatible with them for future extensions. Empirical evaluations on standardized hallucination benchmarks demonstrate that our method substantially reduces hallucinations in long-form generation while preserving or even improving the informativeness of generation. Code is available at: this https URL. 

---
# TurtleAI: Benchmarking Multimodal Models for Visual Programming in Turtle Graphics 

**Authors**: Chao Wen, Jacqueline Staub, Adish Singla  

**Link**: [PDF](https://arxiv.org/pdf/2606.03626)  

**Abstract**: Vision-language models (VLMs) have been explored for visual programming, where they generate code to solve visual tasks. However, most prior work focuses on visual programming for productivity; it remains unclear how well current VLMs perform on education-oriented visual programming and what factors limit their performance. To bridge this gap, we introduce TurtleAI, a benchmark containing 823 tasks curated based on real-world visual programming tasks in the Turtle Graphics domain. Solving these tasks requires models to perceive geometric patterns, reason about spatial relationships, and synthesize Python code that faithfully reproduces geometric patterns. We evaluate 20+ VLMs, including GPT-5, GPT-4o, and Qwen2-VL-72B, and find that they struggle significantly, with most achieving success rates below 30%. To address these limitations, we propose a data generation technique that requires only a small set of seed samples. Fine-tuning Qwen2-VL-72B on the resulting synthetic data yields an improvement of about 20% on real-world tasks. Our failure analysis reveals that GPT-4o struggles with spatial reasoning and precise visual replication, whereas fine-tuning primarily improves the alignment between visual reasoning and code implementation. 

---
# Physics-Guided Policy Optimization with Self-Distillation 

**Authors**: Ke Wang, Yuning Wu, Haoran Liu, Chaoqun Jia, Devin Chen, Kai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.03620)  

**Abstract**: Self-distilled policy optimization (SDPO) has become a popular paradigm for LLM post-training, where a model learns from its own predictions conditioned on privileged information. SDPO, however, is sensitive to how much each update step should be trusted: corrections from a self-teacher can be highly informative on some batches and misleading on others, and applying them uniformly with a fixed step size can destabilize training. Drawing inspiration from viscous-fluid dynamics and formalizing the analogy at the SDE level, we propose Physics-Guided Policy Optimization (PGPO), which introduces an information-modulated step-size multiplier derived from a mutual-information estimate between the student's predictions and the feedback-conditioned teacher. We show that this modulation preserves the order-1 weak-approximation guarantees of vanilla SGD, and incurs negligible overhead per iteration. We evaluate PGPO on the Science-QA dataset, where it outperforms SDPO on 3 of the 4 domains with gains of up to +4.5 points, while remaining stable in a setting where SDPO collapses late in training. 

---
# Exploiting Verification-Generation Gap: Test-Time Reinforcement Learning with Confidence-Conditioned Verification 

**Authors**: Jiahui Li, Jianfeng Shan, Wenpei Chen, Shunyu Wu, Jian Lou, Wenjie Feng, Dan Li, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2606.03608)  

**Abstract**: Test-time reinforcement learning has emerged as a promising paradigm for enhancing the complex reasoning abilities of large language models in a completely label-free manner. Despite existing studies focusing on Pass@1 performance, optimizing Pass@k remains under-explored yet critical in label-free settings, which measures generation coverage for sustained exploration. Optimizing Pass@k in label-free setting is highly non-trivial, as directly applying the Pass@k advantage designs effective for RLVR yields unsatisfactory performance. Through in-depth empirical analysis, we discover the root causes hindering performance: pseudo-label estimations for low-confidence samples have a high probability of being incorrect, while candidate answers for high-confidence samples suffer from severe diversity collapse. To overcome these hurdles, we propose TTRL-CoCoV (Test-Time Reinforcement Learning with Confidence-Conditioned Verification), a novel confidence-adaptive framework that expands Pass@k coverage and improves Pass@1 performance. Based on our key insight that verification capability generally leads generation capability, TTRL-CoCoV employs a confidence-conditioned mechanism: for high-confidence samples, it bootstraps verifier and applies an exploration-enhancing reward to prevent diversity collapse; for low-confidence samples, it delegates pseudo-label selection to the verifier to filter incorrect pseudo-labels; and for medium-confidence samples, it bypasses verification entirely. Extensive experiments demonstrate that TTRL-CoCoV outperforms the best competing methods across 6 widely-recognized benchmarks, achieves average absolute gains of +9.8% in Pass@1 and +18.7% in Pass@16 over TTRL, and even achieves absolute Pass@1 improvements of up to +5.0% across multiple reasoning benchmarks when compared against fully supervised RL methods. Our code repository: this https URL. 

---
# Testing LLM Arithmetic Reasoning Generalization with Automatic Numeric-Remapping Attacks 

**Authors**: Malia Barker, Bishal Lakha, Edoardo Serra, Francesco Gullo  

**Link**: [PDF](https://arxiv.org/pdf/2606.03606)  

**Abstract**: Large language models achieve strong performance on arithmetic reasoning benchmarks, and one common response to arithmetic brittleness is to delegate computation to code. Yet models are still often used in settings where they must reason directly from natural language, and trustworthy models should solve small-number arithmetic word problems without external tools. Prior work shows that LLMs are sensitive to numerical variation: a model may solve an original problem but fail on structurally similar variants requiring the same reasoning procedure with different numbers. We ask whether this fragility persists under a stricter setting involving small, schema-preserving numeric changes that retain the original reasoning program and avoid large-number stress tests. We introduce an automatic algorithm for generating numeric-remapping attacks on arithmetic word problems. Unlike template-based perturbation methods requiring manual schemas or constraints, our approach derives problem-specific symbolic representations, generates constrained numeric remappings, recomputes gold answers, and realizes transformed questions through deterministic edits guided by LLM-generated edit plans. Stage-wise validation and a high-confidence audit retain reliable attacks, making the pipeline scalable with limited human intervention. We evaluate DeepSeek-R1 (70B), Gemma4 (31B), and GPT-OSS (120B) on GSM8K, MAWPS, and MultiArith. On GSM8K, completed runs show conditional accuracy drops of 12.16 to 25.82 percentage points. MAWPS and MultiArith are far more stable, with most attacked accuracies near or above 98%. These results show that numeric-remapping robustness depends strongly on dataset structure: GSM8K remains sensitive even when reasoning programs are preserved and answers are recomputed, while shorter, more regular datasets are more robust. 

---
# CauTion: Knowing When to Trust LLMs for Ensemble Causal Discovery 

**Authors**: Bo Peng, Kaiwen Wu, Sirui Chen, Zhiheng Wang, Yu Qiao, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03602)  

**Abstract**: Causal discovery from observational data remains challenging due to the fundamental limitations of purely statistical methods, such as statistical distinguishability within equivalence classes and sensitivity to finite sample sizes. While large language models (LLMs) offer a promising source of domain knowledge to complement statistical inference, existing LLM-augmented methods are vulnerable to LLM errors and incur high token costs. Moreover, reliance on a single data-centric algorithm can make results sensitive to algorithm-specific biases. To address these limitations, we propose CauTion, a framework that reliably integrates LLM domain knowledge into an ensemble of statistical causal discovery algorithms through consensus filtering and LLM reliability estimation. CauTion proceeds in three stages. First, an algorithm ensemble utilizes a consensus voting to resolve up to 96% of edges on which algorithms agree, achieving near-perfect accuracy on the filtered consensus edges. Second, a trust-calibrated arbitration mechanism estimates the relative reliability of the LLM and the algorithms via an annotation-free trust calibration procedure, which is then utilized to govern a trust-weighted voting process that restricts LLM arbitration exclusively to edges with unreliable algorithmic evidence. Third, a cycle repair step is applied to guarantee the final causal graph is validly acyclic. Experiments on six datasets demonstrate that CauTion consistently outperforms both data-centric and LLM-augmented baselines, with larger gains on larger graphs and strong robustness to LLM errors. Code is available at this https URL. 

---
# DDOR: Delta Debugging for Explainable Overrefusal Testing and Repair 

**Authors**: Qinyan Zhou, Peixin Zhang, Jun Sun, Haonan Zhang, Dongxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03601)  

**Abstract**: While safety alignment and guardrails help large language models (LLMs) avoid harmful outputs, they can also induce overrefusal, i.e., unwarranted rejection of benign queries that merely appear risky. We present DDOR (Delta Debugging for OverRefusal), a fully automated and explainable framework for overrefusal testing and repair in a black-box setting, where only model inputs and outputs are accessible and internal safety mechanisms remain opaque. DDOR applies delta debugging to localize minimal refusal-triggering fragments (mRTFs) that provide phrase-level, explainable evidence for why a refusal occurs. Conditioned on these mRTFs, DDOR generates diverse, context-rich prompts and performs multi-oracle validation to filter intrinsically unsafe or ambiguous cases, producing scalable and model-specific overrefusal test suites (approximately 1K cases per model). Beyond evaluation, we further leverage localized mRTFs to perform targeted prompt repair, substantially reducing overrefusal while preserving the original intent and maintaining safety on genuinely harmful inputs. Overall, DDOR offers a practical end-to-end solution to both evaluate and mitigate overrefusal, improving LLM usability without sacrificing safety. 

---
# PHASER: Phase-Aware and Semantic Experience Replay for Vision-Language-Action Models 

**Authors**: Ziyang Chen, Shaoguang Wang, Weiyu Guo, Qianyi Cai, He Zhang, Pengteng Li, Yiren Zhao, Yandong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.03598)  

**Abstract**: Vision-Language-Action (VLA) models have achieved remarkable success in language-conditioned robotic manipulation. However, deploying these models in open-ended environments requires continuously acquiring novel skills, a process that inevitably triggers severe catastrophic forgetting of previously learned behaviors. While experience replay (ER) serves as a standard mitigating strategy, naive uniform sampling fundamentally misaligns with the temporal characteristics of manipulation trajectories. It systematically under-samples brief but causally critical sub-skills, leading to phase starvation, and completely overlooks the varying degrees of forgetting across historical tasks. To overcome these limitations, we introduce PHASER, an architecture-agnostic continual learning framework. PHASER employs a phase-centric capacity allocation to guarantee equal memory support for all sub-skills, coupled with a multi-modal interference routing strategy that dynamically prioritizes historical phases at high risk of forgetting. Furthermore, to enable fully autonomous lifelong adaptation, we integrate Auto-PC, a lightweight pipeline combining unsupervised action-signal change-point detection with VLM-based semantic verification to extract temporal boundaries without intensive manual supervision. Evaluated across three VLA backbones on LIBERO continual learning suites, PHASER yields substantial empirical improvements, increasing Average Success Rate (ASR) by up to 31% over matched-budget ER and achieving an 87.8% final ASR on the LIBERO-Goal CL setting. 

---
# When Attention Collapses: Stage-Aware Visual Token Pruning from Structure to Semantics 

**Authors**: Jiahui Wang, Kai Zhang, Mai Han, Huanghe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03569)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated remarkable capabilities but suffer from significant computational overhead during inference. While visual token pruning offers a promising solution, existing methods predominantly rely on initial attention scores. This single-metric paradigm presents a critical flaw: high attention scores inherently collapse onto semantically similar regions, thereby severely reducing feature diversity and discarding vital contextual details. To address this, we introduce Structure-to-Semantics (STS), a novel two-stage visual token pruning framework that explicitly decouples the pruning process. The first stage employs a repulsion-based sampling mechanism to maximize spatial and structural diversity. The second stage leverages instruction-aware cross-attention to precisely filter out prompt-irrelevant tokens. This two-stage synergy constitutes the core of STS, first ensuring geometric coverage and then refining the retained tokens according to semantic relevance. Extensive evaluations demonstrate that STS mitigates the redundancy caused by attention-based selection, improving both structural diversity and fine-grained task alignment of the preserved visual tokens. 

---
# Learned Non-Maximum Suppression for 3D Object Detection 

**Authors**: Timo Osterburg, Stefan Schütte, Torsten Bertram  

**Link**: [PDF](https://arxiv.org/pdf/2606.03568)  

**Abstract**: Post-processing is a critical stage in LiDAR-based 3D object detection, where dense and overlapping proposals must be filtered for compact and reliable perception. This work introduces two learned filtering modules that replace heuristic non-maximum suppression (NMS) by leveraging relations among detections. D2D-Rescore employs transformer-based detection-to-detection (D2D) attention, while GossipNet3D adapts the 2D GossipNet concept to 3D through localized message passing in bird's-eye view. A metric-aware matching strategy aligned with the nuScenes evaluation protocol ensures consistent training and validation behavior, improving overall detection performance. Both approaches improve mean average precision (mAP), nuScenes detection score (NDS), and true positive quality compared to CircleNMS, particularly for small and infrequent classes, while adding minimal computational overhead. These results demonstrate that learned, detection-level filtering can enhance 3D detector reliability without modifying the base network, offering a principled alternative to heuristic suppression. Code is available at this https URL . 

---
# Efficient Transformer-Based Localized Patch Sampling for Choroid Plexus Segmentation in Multiple Sclerosis 

**Authors**: Po-Jui Lu, Alessandro Cagol, Mario Ocampo-Pineda, Federico Spagnolo, Marina Mastantuono, Andreea-Alexandra Aldea, Jannis Müller, Özgür Yaldizli, Matthias Weigel, Lester Melie-Garcia, Roberta Magliozzi, Maria Pia Sormani, Ludwig Kappos, Jens Kuhle, Cristina Granziera  

**Link**: [PDF](https://arxiv.org/pdf/2606.03566)  

**Abstract**: Background: The lateral ventricle choroid plexus (LVCP) is gaining recognition as a key imaging biomarker for multiple sclerosis (MS) related to physical disability and neuroinflammation. Yet, manual segmentation of the LVCP is highly tedious, restricting its use in broad clinical trials and longitudinal assessments. This research aims to develop a SwinUNETR-driven pipeline that leverages targeted intra- and peri-ventricular small patch sampling to automatically segment the LVCP in MS from both standalone and multi-modal MRI inputs. Methods: We retrospectively assessed 3T MRI scans across three sets of data stemming from two separate MS-dominant cohorts (Dataset 1: n=177; Dataset 2: n=177; expanded test set: n=388). Our method employed a SwinUNETR architecture trained on 32x32x32 voxel patches, benchmarking it against the 3D UXNET model. The primary metric for evaluation was the Dice Similarity Coefficient (DSC), supplemented by computational demand (GFLOPs) and the 95th percentile Hausdorff Distance (HD95). Results: On the extended test set, the SwinUNETR model secured a mean DSC of 0.868 (95% CI: 0.863-0.872) with MPRAGE and FLAIR combined, showing a statistically significant gain over UXNET (DSC: 0.858 [95% CI: 0.853-0.862], p<0.0001). When restricted to standalone FLAIR inputs, the transformer-based approach sustained a high DSC of 0.863, while the spatial localization of UXNET worsened considerably (HD95: 1.86 vs. 3.00 mm). Importantly, the proposed framework lowered computational load by 99% (91.8 vs. 22,080 GFLOPs). By integrating localized patch sampling with a SwinUNETR architecture, this methodology offers an accurate, robust, and statistically superior alternative to current leading models for LVCP segmentation. Its vast reduction in computational cost makes it ideal for widespread implementation in clinical and research environments. 

---
# \textsc{CR-Seg}: Attention-Guided and CoT-Enhanced Coarse-to-Refined Reasoning Segmentation 

**Authors**: Yifan Cao, Xiaocui Yang, Faxian Wan, Shi Feng, Daling Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03564)  

**Abstract**: Reasoning segmentation aims to segment target objects described by complex language through joint visual-textual reasoning. Existing methods typically rely on either learned semantic tokens to bridge Multimodal Large Language Models (MLLMs) and segmentation models, suffering from difficult cross-modal alignment, or explicit spatial prompts such as bounding boxes, which may lose holistic response semantics. To address these limitations, we propose Attention-Guided and CoT-Enhanced Coarse-to-Refined Reasoning Segmentation, termed CR-Seg, a two-stage framework for coarse-to-refined reasoning segmentation. Specifically, we design an Extract Attention Maps and Points (EAP) module to extract attention maps for coarse target localization and select informative points, both of which are fed into SAM for mask refinement. To alleviate reasoning--answer inconsistency, we further introduce Global-to-Local Chain-of-Thought (GLCoT), which guides the model to reason progressively from global scene context to local target details. Extensive experiments on reasoning segmentation benchmarks demonstrate the effectiveness of CR-Seg. 

---
# When Should the Teacher Move? Temporal Coupling and Stability in Self On-Policy Distillation 

**Authors**: Haowei Guo, Baolong Bi, Ruicheng Zhang, Bingqian Sun, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03532)  

**Abstract**: Self on-policy distillation trains a student policy against a teacher derived from its own parameter history, yet the teacher's update schedule -- which governs the \emph{temporal coupling} between teacher and student -- has not been systematically studied as a stability variable. Through a controlled schedule sweep on Qwen3-8B, we establish that \emph{isolation periods}, defined as complete teacher freezing between updates, are the key structural property enabling stable learning, not teacher age. To characterize these underlying training dynamics, we introduce a diagnostic framework of temporal KL structure, refresh shock, and length-tail risk. This framework further uncovers \emph{state-oblivious collapse}: optimal short-horizon fixed schedules catastrophically fail under long-horizon training because a clock-driven refresh can copy a transiently drifting student into the teacher in a single, irreversible step. This failure mode is invisible under short-horizon evaluation and mechanistically distinct from EMA's chronic contamination. To address this, we propose \emph{Consolidation-Gated Teacher Refresh} (CGTR), which preserves isolation periods while gating each refresh on joint evidence of reward improvement and length-tail safety, ensuring every teacher movement responds to genuine student consolidation rather than a clock signal. With a single shared parameter set and no per-dataset retuning, CGTR achieves \textbf{zero collapse} and the best final score on all four tasks (Chemistry, Biology, Physics, ToolUse), self-regulating its refresh frequency to each task's learning dynamics. 

---
# High-Precision APT Malware Attribution with Out-of-Scope Resilience 

**Authors**: Peter Williams, Adam Sobey, Erisa Karafili  

**Link**: [PDF](https://arxiv.org/pdf/2606.03523)  

**Abstract**: Early attribution of Advanced Persistent Threat (APT) activity can help defenders prioritise investigation, select countermeasures, and reduce the impact of an intrusion. Malware provides useful attribution evidence, but automated APT malware attribution remains difficult in practice. Existing approaches are typically trained and evaluated as closed-set classifiers over a limited number of known APT groups. In operational environments, however, classifiers are likely to encounter samples from groups not represented during training. Closed-set classifiers are then forced to assign such samples to known groups, producing unsupported and potentially misleading attributions. We present a high-precision APT malware attribution method based on ranked binary classifiers with explicit abstention. Rather than training a single multi-class classifier, our approach trains and tunes two binary classifiers per APT group, ranks the classifiers by validation performance, and applies them sequentially. A sample is attributed only when a classifier provides sufficient evidence; otherwise, it abstains. We evaluate the method on the APT Malware dataset and on a larger combined dataset designed to stress-test out-of-scope behaviour. On the APT Malware dataset, the method achieves higher precision than previously published results on the same dataset. In the most challenging setting, where 87% of test samples came from 60 APT groups excluded from training, the method abstained on 94% of out-of-scope samples while maintaining 92% precision and 95% selective accuracy on the samples it classified. 

---
# Post-Hoc Robustness for Model-Based Reinforcement Learning 

**Authors**: Siemen Herremans, Ali Anwar, Siegfried Mercelis  

**Link**: [PDF](https://arxiv.org/pdf/2606.03521)  

**Abstract**: To improve the real-world applicability of reinforcement learning (RL), the field of adversarially robust RL studies how to train agents under adversarial environment perturbations. In this setting, a protagonist agent optimizes a policy under environmental perturbations from an adversary, resulting in a zero-sum Markov game. When adversarially robust RL is combined with model-based RL, the adversary can target a learned transition model instead of the training environment. Extending this idea, this work introduces post-hoc robustification of deep RL agents at inference time. By using the learned model in combination with a trained nominal policy, our approach performs a robust policy improvement step. The goal is to improve robustness without any additional training of neural networks. Specifically, we utilize model-predictive control under adversarial rollouts, which are approximated via projected gradient descent within a bounded uncertainty set. Furthermore, these offline rollouts are performed while considering and mitigating out-of-distribution issues. The proposed methodology is validated by demonstrating significant improvements in robustness when the algorithm is evaluated in perturbed Gymnasium MuJoCo environments, while considering the computational limitations of the post-hoc inference setting. 

---
# Scalable On-Hardware Training of Quantum Neural Networks and Application to Clinical Data Imputation 

**Authors**: Natansh Mathur, Panagiotis Kl. Barkoutsos, Masako Yamada, Martin Roetteler, Iordanis Kerenidis  

**Link**: [PDF](https://arxiv.org/pdf/2606.03517)  

**Abstract**: Training quantum neural networks (QNNs) on quantum hardware is currently bottlenecked by the cost of gradient estimation: standard parameter-shift methods require a number of circuit evaluations that grows quadratically with the number of trainable parameters, making hardware-based optimisation impractical beyond small system sizes. In this work, we introduce a training framework that reduces this cost to logarithmic in the number of qubits, making gradient-based QNN optimisation feasible on near-term hardware at increasing scales.
Our framework combines three co-designed ingredients: (i) a structured, subspace-preserving Butterfly circuit architecture with $O(n \log n)$ parameters and logarithmic depth; (ii) a layer-wise training strategy that confines on-hardware optimisation to one small, well-structured layer at a time; and (iii) a parallelised parameter-shift rule that exploits the commuting structure within each Butterfly layer to extract all gradients in a constant number of circuit executions. Together these reduce the number of distinct circuit evaluations per optimisation step from $O(n^2)$ to $O(\log n)$.
We validate the framework on clinical data imputation using the MIMIC-III electronic health record dataset, a demanding benchmark sensitive to optimisation instability and model variance. Hybrid classical-quantum models are trained directly on IonQ Forte Enterprise trapped-ion hardware at 16 qubits without performance degradation relative to ideal or noisy simulation and via tensor-network simulation at 32 qubits, with 32-qubit inference executed on hardware. The resulting models match or exceed strong classical neural baselines in downstream patient survival prediction while exhibiting reduced variance across runs, demonstrating that the proposed framework enables practical, scalable QNN training under realistic hardware constraints. 

---
# SPADE: Sketch-guided Path Planning Augmented with Diffusion Experts 

**Authors**: Charbel Abi Hana, Tatiana Ghantous, Mikael Khalil, Anthony Rizk  

**Link**: [PDF](https://arxiv.org/pdf/2606.03512)  

**Abstract**: Path planning is essential for Autonomous Mobile Robots (AMRs). Conventional methods for incorporating human preferences into planning typically rely on either complex reward engineering or hardware-intensive solutions. Recent state-of-the-art frameworks leverage imitation learning to train behavior-specific path planning models from expert demonstrations. However, these approaches face two key limitations: limited generalization to unseen environments and low robustness in demonstration collection. To address these challenges, this work introduces an enhanced framework that focuses on two main contributions: an overhauled annotation tool built on ROS 2, and a novel training strategy that integrates diffusion-based augmentation into baseline behavioral cloning models. A dataset of expert demonstrations is provided and evaluated through ablation studies to assess the robustness of the proposed solution. The enhanced approach outperforms state-of-the-art methods with 39.1% lower Absolute Pose Error (APE) and 33.5% lower Fr'echet Inception Distance (FID) while having 93.8% less trainable parameters. Moreover it attains diffusion-level generalization while preserving the real-time, on-edge properties of state-of-the-art models. 

---
# BaltiVoice: A Speech Corpus and Fine-tuned Whisper ASR System for the Balti Language 

**Authors**: Muhammad Ali  

**Link**: [PDF](https://arxiv.org/pdf/2606.03504)  

**Abstract**: We present BaltiVoice, a 16.8-hour read-speech corpus for Balti (ISO 639-3: bft), a Tibetic language spoken in Gilgit-Baltistan, Pakistan, with no prior publicly available ASR resources. The corpus contains 10,060 validated utterances in native Nastaliq script, derived from Mozilla Common Voice recordings. We fine-tune OpenAI Whisper-small on this corpus and report a Word Error Rate (WER) of 30.07% on a held-out validation set of 538 utterances, down from a measured zero-shot baseline of 182.18% for Whisper-small on Balti. The dataset, fine-tuned model, and a live transcription demo are publicly available on HuggingFace. 

---
# Learn from Your Mistakes: Tree-like Self-Play for Secure Code LLMs 

**Authors**: Wenqi Chen, Ziyan Zhang, Bing Wang, Lin Liu, Hengheng Zhang, Zhengsu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.03489)  

**Abstract**: While Large Language Models (LLMs) excel in code generation, they remain prone to replicating subtle yet critical vulnerabilities endemic to their training data. Current alignment techniques, such as Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), typically apply coarse-grained optimization at the sequence level. This approach often fails to address the localized nature of security flaws, where a single incorrect token choice can compromise an entire program. To bridge this gap, we introduce Tree-like Self-Play (TSP), a framework that reframes secure code generation as a fine-grained sequential decision process. Unlike standard methods that blindly maximize likelihood, TSP constructs a decision tree where the model explores branching trajectories--generating both secure "golden paths" and vulnerable variants. By treating code generation as a self-play game, the model learns to strictly discriminate against its own localized errors. This provides a dense, on-policy learning signal that forces self-correction precisely at the critical decision nodes where vulnerabilities typically emerge. Our experiments demonstrate that TSP fundamentally enhances model reliability. In Python security benchmarks, TSP boosts CodeLlama-7B's pass rate (SPR@1) to 75.8%, significantly outperforming SFT (57.0%) and unstructured self-play baselines. Crucially, TSP induces robust out-of-distribution generalization: the model not only reduces vulnerabilities in unseen categories (CWEs) by 24.5% but also successfully transfers security principles learned from C/C++ to diverse languages, including Python, Go, and JavaScript. This suggests that TSP does not merely memorize patches, but internalizes abstract, language-agnostic security logic. 

---
# NeuroArmor: Safe-Variant-Guided Representation Consistency for Selective Re-Anchoring in Jailbreak Defense 

**Authors**: Zhongyang Lin, Ziran Zhao, Feifei Zhai, Pengyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03486)  

**Abstract**: Large language models remain vulnerable to jailbreak attacks that hide harmful intent behind seemingly ordinary requests such as role-play, translation, encoding, adversarial suffixes, and multi-turn buildup. Existing defenses still struggle to handle these attacks without over-blocking benign but sensitive requests, partly because they often apply the same action to every prompt and therefore fail to balance safety and helpfulness. We propose NeuroArmor, a white-box runtime defense that uses prompt-specific safe variants as a local safety reference for deciding when intervention is needed and, once triggered, as safe targets for intervention. For each prompt, NeuroArmor builds K safe variants, compares the prompt state against this local safe reference in hidden-state space, and routes anomalies either to a refusal branch for malicious prompts or to a helpful recovery branch for borderline benign prompts. On Llama-3-8B-Instruct, NeuroArmor reduces malicious attack success rate (ASR) from 41.56% to 1.57% while lowering benign false positive rate (FPR) on the shared benign pool from 30.26% to 22.05%; matched baselines remain substantially weaker on this trade-off. External-judge and manual behavioral evaluations further show that the remaining non-blocked outputs are much less likely to be operationally harmful. Overall, NeuroArmor provides a more effective runtime strategy for jailbreak defense by combining prompt-specific consistency checking, routing, and selective intervention. 

---
# Analyzing Stream Collapse in Hyper-Connections: From Diagnosis to Mitigation 

**Authors**: Ekaterina Alimaskina, Gleb Molodtsov, Aleksandr Beznosikov  

**Link**: [PDF](https://arxiv.org/pdf/2606.03483)  

**Abstract**: Hyper-Connections (HC) replace the single Transformer residual stream with multiple streams, introducing a permutation symmetry over stream indices. We study how this symmetry is resolved in practice: whether streams specialize in a balanced way or exhibit dominant-stream usage. Using fine-grained diagnostics for HC-based language models, we trace how multi-stream representations are actually used. We find that after an early seeding stage, residual mixing often remains close to identity, limiting a core HC mechanism for exchanging information between streams. Moreover, both signal and interpretable features concentrate in a dominant stream, and the nominally multi-stream residual connection can underutilize its capacity, behaving closer to a single-stream residual pathway. Finally, we show that breaking symmetry at stream initialization reduces dominant behavior and improves performance across \textit{m}HC variants. Our code is publicly available. 

---
# Rethinking the Role of Tensor Decompositions in Post-Training LLM Compression 

**Authors**: Artur Zagitov, Alexander Miasnikov, Maxim Krutikov, Vladimir Aletov, Gleb Molodtsov, Nail Bashirov, Artem Tsedenov, Aleksandr Beznosikov  

**Link**: [PDF](https://arxiv.org/pdf/2606.03465)  

**Abstract**: Post-training compression is essential for deploying large language models (LLMs) under tight resource constraints. Tensor decompositions have emerged as a promising direction, offering compact parameterizations well suited to Transformer weight structures. However, existing studies evaluate these methods in narrow settings, leaving unclear whether tensorization is effective at large-scale deployment. We systematically evaluate tensor compression across dense and MoE architectures, establishing performance trade-offs grounded in both empirical analysis and theoretical analysis. We identify a fundamental mismatch between the shared subspaces assumed by tensor decompositions and the heterogeneous representations learned by modern LLMs, thereby delineating their practical limits and clarifying their viable role in large-scale deployment. The code is available at this https URL. 

---
# Tonal parsimony in chord-sequence analysis: combining modulation cost and tonal vocabulary 

**Authors**: François Pachet  

**Link**: [PDF](https://arxiv.org/pdf/2606.03459)  

**Abstract**: We study the assignment of local tonalities to chord sequences, a task useful for harmonic analysis, composition, and jazz-oriented improvisation. Standard dynamic-programming approaches minimize modulations but can introduce unnecessarily many tonal centers. We compare this transition-only objective with pure minimum-vocabulary analysis and with tonal parsimony, which minimizes lexicographically the number of modulations and then the number of distinct tonalities. Although this joint objective is combinatorially hard in general, we give exact algorithms exploiting the fixed 24-tonality major/minor universe. On 31,032 LMD Chords sequences, tonal parsimony preserves the transition optimum while reducing tonal vocabulary in 55.8% of cases. With weighted jazz-substitution closure, it lowers mean tonalities from 3.802 to 3.206 and modulations from 16.728 to 12.141. On 1,555 annotated jazz standards, it improves compatible chord-scale agreement to 95.6%, supporting tractable professional-scale harmonic analysis. 

---
# FORGE: Multi-Agent Graduated Exploitation and Detection Engineering 

**Authors**: Farooq Shaikh  

**Link**: [PDF](https://arxiv.org/pdf/2606.03453)  

**Abstract**: Vulnerability disclosure volumes now far exceed organizational assessment capacity, yet three adjacent research communities (proof-of-concept generation, vulnerability prioritization, and detection rule engineering) operate largely in isolation. Existing automated exploit generation systems report binary pass/fail outcomes, discarding partial progress and producing no signal for the other two communities. This paper presents FORGE, a multi-agent system that bridges these three silos through graduated exploitation depth. Five specialized agents (Intel, Generator, Planner, Exploit, and Detector) execute in a fixed pipeline that (1) generates targeted vulnerable applications from CVE metadata, (2) conducts coached, multi-turn exploitation assessed by an LLM-primary oracle on a four-level taxonomy (L0: no evidence through L3: full compromise), and (3) produces Sigma and Snort detection rules grounded in OpenTelemetry exploitation traces. Graduated depth is the bridging mechanism: deeper exploitation yields richer behavioral traces for detection engineering, while depth data across scoring bands provides ground truth for prioritization validation. A tiered knowledge architecture accumulates intelligence across assessments, transferring build and exploitation experience to subsequent CVEs. Evaluation on 603 CVEs from the CVE-GENIE dataset achieves 67.8% end-to-end L1+ exploitation at USD 1.50 per CVE across eight languages and 187 CWE types. Exploitation rates remain near 68% regardless of EPSS or CVSS band, indicating that pattern-level reachability is orthogonal to metadata-based prioritization. Detection rules from L2+ exploitation achieve significantly higher span-normalized grounding than L1-derived rules (p=0.035), and 93.4% of generated Snort rules produce zero false positives against a synthetic benign corpus. 

---
# PRISM: Synergizing Vision Foundation Models via Self-organized Expert Specialization 

**Authors**: Ying Tang, Dong Li, Youjia Zhang, Zikai Song, Junqing Yu, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03444)  

**Abstract**: Unifying the complementary strengths of diverse Vision Foundation Models (VFMs) into a single efficient model is highly desirable but challenged by the negative transfer inherent in monolithic distillation. To address these feature conflicts, we introduce \textbf{PRISM}, a novel dual-stream Mixture-of-Experts (MoE) framework that synergizes VFMs via modular specialization. We propose a two-stage paradigm: (1) expertise deconstruction, where a teacher-conditional router guides experts to specialize in distinct representational subspaces to mitigate interference, followed by (2) dynamic recomposition, where the router learns to assemble these experts into tailored computational pathways for downstream tasks. Experiments on PASCAL-Context and NYUD-v2 show that \textbf{PRISM} establishes a new state of the art, validating that sparse, emergent specialization is a scalable approach for integrating diverse visual knowledge. 

---
# A Hybrid Approach For Malware Classification Using Secondary Features Fusion 

**Authors**: Raja Khurram Shahzad, Muhammad Mustaqeem, Haroon Elahi  

**Link**: [PDF](https://arxiv.org/pdf/2606.03432)  

**Abstract**: The number of malware (either variant or novel) is rapidly increasing, making malware detection and mitigation a complex problem. One approach to improving malware mitigation is automatic detection and malware family classification. However, traditional malware detection methods cannot classify detected malware into their respective families, hindering effective malware mitigation. Consequently, this paper proposes a method to automate malware detection and classification of the detected malware into respective malware families. The proposed method uses feature fusion after extracting relevant malware features such as API calls and fixed and variable length n-grams with a customized feature selection method. Moreover, for the predictive model, a voting based approach is proposed for algorithm fusion. For the experimental evaluation of the proposed method, both binary and multi-class classification approaches are applied to the data set provided by Microsoft. Finally, the experimental results are compared with the state of the art. The experimental results indicate the effectiveness and efficiency of the proposed approach with an AUC of 0.989, accuracy of 99.72%, and a log loss of 0.01. 

---
# FlowGuard: Flow Matching for Identity-Independent Detection of Data-Free Model Stealing Attacks on Energy System Intrusion Detection Systems 

**Authors**: Maxime Schwarzer, Laurin Holz, Tobias Huerten, Johannes Loevenich, Thies Moehlenhof, Roberto Rigolin F. Lopes, Veit Hagenmeyer  

**Link**: [PDF](https://arxiv.org/pdf/2606.03430)  

**Abstract**: Artificial Intelligence (AI)-based Intrusion Detection Systems (IDS) deployed in energy infrastructure are vulnerable to model theft attacks, which allow adversaries to create evasive traffic offline. Current defences against model extraction rely either on identity-bound query monitoring, which is ineffective against distributed attackers (Sybil), or on prediction poisoning through soft-label perturbation, which is inapplicable to hard-label IDS deployments. Therefore, we propose FlowGuard, an identity-independent defence based on flow matching that classifies incoming queries as out-of-distribution (OOD) prior to IDS processing. This approach exploits the fact that queries generated synthetically for data-free model stealing attacks occupy a lower-dimensional manifold than real network traffic. This results in measurably lower log-likelihoods when using a Continuous Normalizing Flow that has been trained on legitimate data. We evaluate our method against PRADA and FDINet using MAZE and DisGUIDE attacks in single-client and distributed (100-client Sybil) settings. While PRADA's detection rate dropped to 0% when the distribution changed, our defence maintained a stable detection rate across both settings without relying on identity information. We discuss the scope and limitations of the approach, and outline potential applications to data-dependent attacks. 

---
# PrimeSVT: An Automated Memory-aware Pruning Framework with Prioritized Compression Policy for Spiking Vision Transformers 

**Authors**: Rachmad Vidya Wicaksana Putra, Achyuta Muthuvelan, Alberto Marchisio, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2606.03428)  

**Abstract**: The large sizes of Spiking Vision Transformers (SViTs) still hinder their embedded implementation, highlighting the need for model compression. State-of-the-art works compress SViT models through unstructured pruning, which needs specialized hardware accelerators for their specific sparsity patterns to maximize efficiency gains. Moreover, their manual approach requires a huge design time to find an appropriate pruning setting for each network, thus making this approach not scalable. To address this limitation, we propose PrimeSVT, a novel framework that performs automated memory-aware structured pruning on pre-trained SViT models, thereby maximizing their efficiency gains during inference amenable to widely-used computing architectures. To achieve this, PrimeSVT first sorts the SViT layers based on their sizes (i.e., number of parameters), identifies the targeted pruning layers based on their robustness under different pruning rates, then leverages this order for compressing the model layer-by-layer sequentially from the largest one to the smallest one (i.e., so-called prioritized compression policy), while considering the user-defined constraints (i.e., acceptable accuracy and memory saving). In each layer, PrimeSVT employs channel-wise filter pruning based on their L2-norm values to structurally remove the non-significant weights. Experimental results show that PrimeSVT saves 26.68% memory through automated single-shot pruning, while preserving accuracy within 3% (70.3% without fine-tuning and 72.9% with fine-tuning) from the original unpruned SViT model (73.3%), thus meeting the accuracy and memory constraints. These show that our PrimeSVT framework enables design automation for SViTs and their embedded implementation. 

---
# Optimizing Explicit Unit-Distance Lower-Bound Certificates 

**Authors**: Michael T.M. Emmerich  

**Link**: [PDF](https://arxiv.org/pdf/2606.03419)  

**Abstract**: The 2026 disproof of Erdős's unit-distance conjecture and Sawin's subsequent explicit quantitative refinement show that the maximum number $u(n)$ of unit distances among $n$ planar points can exceed $n^{1+\varepsilon}$ for a fixed positive $\varepsilon$. Sawin's explicit bound gives more than $n^{1.014}$ unit distances for arbitrarily large $n$ and exposes finite parameters whose choice is not fully optimized. This report formulates the finite parameter-selection task as a variant of a nonlinear integer programming problem and proposes an open-source Python verification pipeline, first validated by reproducing Sawin's published parameter choice and then applied to computationally improved certificates. The main computational contribution is an integer optimization and checking procedure for the sets of primes $T$ and $S_Q$, the integer multiplicities $k(p)$, and a rationally encoded real parameter $R$. The optimization pipelines are intentionally lightweight and replicable on standard hardware: we propose a deterministic greedy construction heuristic, a Tailored Integer Evolution Strategy with repair operators for number-theoretic feasibility, and a two-parent discrete-recombination variant. Four certificate levels are compared: Sawin's published example with $\delta=0.0141144286784982\ldots$, a greedy optimization certificate with $\delta=0.0151718056372133\ldots$, a Tailored Integer Evolution Strategy certificate with rational $R=6672416/100000$ and $\delta=0.0152616610684193\ldots$, and a Tailored Integer Evolution Strategy with discrete recombination, again with $R=6672416/100000$, giving $\delta=0.0152628688170072\ldots$. Consequently, subject to Sawin's explicit criterion being applied exactly as cited, the best current certificate supports the cautious clean statement $u(n)>n^{1.0152}$ for arbitrarily large $n$. 

---
# Causal Evidence of Stack Representations in Modeling Counter Languages Using Transformers 

**Authors**: Nishit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.03398)  

**Abstract**: Formal languages have proven to be effective conduits to understand the inner mechanisms of transformers. Past work has shown that transformers trained on next token prediction over counter languages learn representations consistent with an underlying stack structure. Beyond representational analysis, this paper investigates the causal role of these representations. Linear probes are trained to predict the stack depth at each token from the model's hidden states, and a principal representation direction is extracted from the probe. Ablation of this direction from the model causes sequential accuracy to collapse to near 0%, providing strong empirical evidence that the stack representation is not just learned, but is causally necessary for model performance. 

---
# When Model Merging Breaks Routing: Training-Free Calibration for MoE 

**Authors**: Canbin Huang, Tianyuan Shi, Xiaojun Quan, Jingang Wang, Jianfei Zhang, Qifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03391)  

**Abstract**: Model merging has emerged as a cost-effective approach for consolidating the capabilities of multiple LLMs without retraining. However, existing merging techniques, largely based on linear parameter arithmetic or optimization, struggle when applied to Mixture-of-Experts (MoE) architectures. We identify a critical failure mode in MoE merging, termed routing breakdown, in which the merged router fails to dispatch tokens to suitable experts. Routing breakdown stems from the sensitivity of the non-linear softmax and discrete Top-k routing mechanisms to parameter perturbations from merging, a sensitivity further amplified by load-balancing constraints imposed during MoE pretraining. Because fine-tuned experts exhibit distinct specializations, even modest misrouting can cause severe performance degradation. To address this issue, we propose Hessian-Aware Router Calibration (HARC), a training-free framework that leverages second-order curvature information to realign the merged router. This approach admits a closed-form solution that can be efficiently solved using a matrix-free conjugate gradient method. Experiments on mathematical reasoning and code generation tasks show that HARC effectively mitigates routing breakdown across diverse MoE merging baselines and leads to substantial performance improvements. Our code is available at this https URL. 

---
# Grasp-Then-Plan with Failure Attribution: A Closed Two-Stage Framework for Precise and Generalizable Robotic Manipulation 

**Authors**: Jiahao Xu, Peiyuan Wang, Hanzhuo Zhang, Zihao Yu, Tianyu Fu, Hao Chen, Xuanhao Xiang, Jianbo Yu, Chenchen Fu, Wanyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03385)  

**Abstract**: In robotic manipulation, the tight coupling between grasping and motion planning often obscures the true source of failure, leading to inefficient trial-and-error. To enable efficient long-horizon manipulation, we propose GTP-FA (Grasp-Then-Plan with Failure Attribution), a task-oriented two-stage grasp-then-plan framework that generates grasp candidates and performs downstream motion planning conditioned on the selected grasp. Given a failed manipulation trajectory, we learn a failure attribution model that generalizes to unseen grasps and produces a stable distribution over failure modes for diagnosis-guided optimization. Based on these attribution results, we then optimize both modules in a diagnosis-driven manner: on the grasping side, we inject task-level priors and risk penalties into grasp candidate scoring and optimization to suppress unstable or task-incompatible grasps; on the planning side, we target high-risk initial states through data collection and fine-tuning to address genuine planning bottlenecks. We evaluate the proposed framework in both simulation and real-robot experiments, and show that GTP-FA improves the corresponding base learners across RL, IL, diffusion-policy, and VLA-based settings, achieving substantially higher overall task success rates. 

---
# Local Guidance, Global Impact: Gaussian-Reshaped Trust Region Unlocks Behavior Transitions 

**Authors**: Bingxu Liu, Jiashun Liu, Johan Obando-Ceron, Hao Wang, Runze Liu, Pablo Samuel Castro, Aaron Courville, Ling Pan  

**Link**: [PDF](https://arxiv.org/pdf/2606.03382)  

**Abstract**: While Proximal Policy Optimization (PPO) demonstrates strong performance in stationary settings, we show that its standard optimization paradigm struggles in continual and non-stationary environments. The failure does not stem from insufficient model capacity or overly restrictive clipping. Instead, PPO performs persistent, directionally inefficient local updates, which indicates a lack of geometry-aware guidance for accumulating meaningful behavioral change and ultimately hindering transitions toward new behavior patterns. Although divergence-based regularization introduces partial geometric awareness, its monotonically increasing penalties implicitly discourage large policy deviations, even when such shifts are necessary for effective adaptation. To address this limitation, we propose Gaussian Trust Region Policy Optimization (GTR), which reshapes the trust region using a Gaussian kernel. The resulting constraint is bounded and non-monotonic, providing strong local stability while progressively relaxing under sustained high-advantage updates. To further improve robustness, we introduce a Mixture Gaussian Anchor that adapts to recent policy trajectories, reducing variance induced by stale references. GTR is architecture-agnostic and achieves strong performance across games, simulated robotic control, open-world exploration, and language model post-training. These results demonstrate that geometry-aware trust-region design can be a promising direction for robust reinforcement learning in complex non-stationary environments. Our code is available at this https URL. 

---
# AI Model Extraction Attacks: Bypassing Single-Client Assumptions in Defenses 

**Authors**: Maxime Schwarzer, Johannes F. Loevenich, Gustavo Sánchez, Laurin Holz, Thies Möhlenhof, Tobias Hürten, Roberto Rigolin F. Lopes, Veit Hagenmeyer  

**Link**: [PDF](https://arxiv.org/pdf/2606.03381)  

**Abstract**: Ensuring the protection of Artificial Intelligence (AI) models deployed in military Command and Control (C2) systems and critical infrastructure is essential for maintaining information superiority. Model Extraction Attacks (MEAs) pose a significant threat, as they enable adversaries to replicate proprietary models, compromise protected information, and prepare offline adversarial attacks. However, current defense strategies predominantly rely on the Single Client Assumption (SCA), which is the implicit assumption that attacks originate from isolated identities. This work systematically demonstrates that the SCA is fundamentally invalid in the presence of coordinated threat actors, such as Advanced Persistent Threats (APTs). We introduce a modular, open-source framework called CerberusAI for reproducible model-stealing research, and use it to simulate distributed attack scenarios. Our empirical evaluation shows that well-established defense mechanisms, such as Protecting Against Deep Neural Network Model Stealing Attacks (PRADA), can be bypassed by basic round-robin query distribution strategies, resulting in a significant reduction in detection performance. Furthermore, we demonstrate that even global aggregation approaches can be rendered operationally useless through adaptive traffic mixing. These results highlight the need for a paradigm shift towards stateful, identity-independent defense architectures in the field of model extraction attacks. This paper was originally presented at the International Conference on Military Communication and Information Systems (ICMCIS), organized by the Information Systems Technology (IST) Scientific and Technical Committee, IST-224-RSY - the ICMCIS, held in Bath, United Kingdom, 12-13 May 2026 and won the best paper award. 

---
# P\textsuperscript{2}-DPO: Grounding Hallucination in Perceptual Processing via Calibration Direct Preference Optimization 

**Authors**: Ruipeng Zhang, Zhihao Li, Haozhang Yuan, C. L. Philip Chen, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03376)  

**Abstract**: Hallucination has recently garnered significant research attention in Large Vision-Language Models (LVLMs). Direct Preference Optimization (DPO) aims to learn directly from the corrected preferences provided by humans, thereby addressing the hallucination issue. Despite its success, this paradigm has yet to specifically target the perceptual bottleneck in attended regions or address insufficient Visual Robustness against image degradation. Furthermore, existing preference pairs are often vision-agnostic and their inherently off-policy nature limits their effectiveness in guiding model learning. To address these challenges, we propose Perceptual Processing Direct Preference Optimization (P\textsuperscript{2}-DPO), a novel training paradigm in which the model generates and learns from its own preference pairs, thereby directly addressing the identified visual bottlenecks while inherently avoiding the issues of vision-agnostic and off-policy data. It introduces: (1) an on-policy preference pairs construction method targeting Focus-and-Enhance perception and Visual Robustness, and (2) a well-designed Calibration Loss to precisely align visual signals with the causal generation of text. Experimental results demonstrate that with a comparable amount of training data and cost, P\textsuperscript{2}-DPO outperforms strong baselines that rely on costly human feedback on benchmarks. Furthermore, evaluations on Attention Region Fidelity (ARF) and image degradation scenarios validate the effectiveness of P\textsuperscript{2}-DPO in addressing perceptual bottleneck in attended regions and improving Visual Robustness against degraded inputs. 

---
# The Unsampled Truth: Psychometrics in SLMs Measure Prompt Artifacts, Not Psychological Constructs 

**Authors**: Nils Schwager, Christoph Hau, Simon Münker, Achim Rettinger  

**Link**: [PDF](https://arxiv.org/pdf/2606.03357)  

**Abstract**: When prompting SLMs for psychometric assessments, researchers assume the outputs reflect semantic reasoning. We evaluate this premise across 13 open-weights models (0.6B to 14B parameters) using a prompt variation framework that separates semantic signals from prompt artifacts. By systematically varying personas, instructions, items, and option symbols, we find that artifactual variance frequently overpowers the semantic signal. In these cases, models predominantly reflect prompt compliance rather than simulated psychological traits. While these findings limit SLM utility in psychometrics, our framework provides a diagnostic tool to identify destructive artifacts and isolate semantic understanding for future frontier-model research. 

---
# SynCred-Bench: Benchmarking Synthetic Credibility in AI-Generated Visual Misinformation 

**Authors**: Junxiao Yang, Minghao Zhang, Xiaoce Wang, Haoran Liu, Shiyao Cui, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03348)  

**Abstract**: Recent generative models can now produce visual artifacts with realistic embedded text and layouts, creating a new misinformation threat: synthetic credibility. We introduce SYNCRED-Bench, a benchmark of 600 AI-generated misinformation images balanced across six credible-form categories and seven fine-grained circulation styles, together with FP450, a real-image negative set for measuring false positives. Extensive evaluation shows that existing systems remain unreliable: under a 5% false-positive-rate constraint, 15 MLLMs achieve only 10.5% true positive rate (TPR), open-source AIGC detectors achieve less than 5%, and commercial APIs reach 57.6%. Human annotators also struggled to identify synthetic credibility, reaching only 63% TPR. These findings establish synthetic credibility as a severe and underexplored visual misinformation challenge, and provide a benchmark for developing detectors that reason beyond superficial credibility cues. 

---
# AugMask: Training Diffusion Models on Incomplete Tabular Data via Stochastic Augmentation and Masking 

**Authors**: Jungkyu Kim, Taeyoung Park, Kibok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.03347)  

**Abstract**: Score-based diffusion models have emerged as prominent deep generative models; however, their application to tabular data remains challenging because their backbones assume fully specified inputs, whereas real-world tabular data often contain missing values. We propose AugMask, a plug-and-play training framework that adapts missing-unaware backbones to incomplete data by separating conditioning from supervision. AugMask 1) constructs numeric inputs via conditional stochastic augmentation using lightweight auxiliary models, and 2) applies denoising supervision only to observed coordinates. In effect, augmented missing entries serve as uncertain conditioning context rather than training targets. We connect this training rule to a Rao--Blackwellized objective and show that marginalizing missing entries yields a variance-weighted sensitivity penalty, discouraging over-reliance on uncertain completions. Across diverse datasets and missingness regimes, AugMask enables standard diffusion-based tabular generators to outperform specialized missing-aware baselines. 

---
# Evaluating LLMs' Effectiveness on Real-World Consumer Device Repair Questions 

**Authors**: Atm Mizanur Rahman, Md Arid Hasan, Syed Ishtiaque Ahmed, Sharifa Sultana  

**Link**: [PDF](https://arxiv.org/pdf/2606.03331)  

**Abstract**: Consumer device repair is an important but underexplored testbed for large language models (LLMs). Repair tasks require reasoning over incomplete problem descriptions, hardware-specific diagnostics, actionable troubleshooting, and safety-critical decisions, where incorrect advice can cause device damage, battery hazards, or permanent data loss. We introduce a benchmark of 991 real-world repair questions from Reddit spanning phone repair, computer repair, and data recovery, each paired with technician-written reference solutions, and provide Bangla translations to evaluate cross-lingual performance. We evaluate six state-of-the-art LLMs in English and Bangla using four repair-specific criteria: correctness, completeness, practicality, and safety. Our results show that while LLMs can provide useful repair assistance, they remain unreliable for high-risk real-world repair tasks without rigorous evaluation and explicit safety safeguards. Phone repair is the most difficult and safety-sensitive domain, and all models make substantial errors in board-level diagnosis, repair prioritization, and safe recovery procedures. Across domains and models, Bangla responses consistently perform worse than English responses. Among the evaluated models, GPT-5.4 performs best overall. 

---
# FLIPS: Instance-Fingerprinting for LLMs via Pseudo-random Sequences 

**Authors**: Gurvan Richardeau, Gohar Dashyan, Erwan Le Merrer, Gilles Tredan  

**Link**: [PDF](https://arxiv.org/pdf/2606.03330)  

**Abstract**: Literature reveals that a Large Language Model's (LLM) behavior is not only conditioned by its original weights but also its instance-level parameters, such as instructional prompt, sampling configuration or quantization. A model that generates safe outputs under one configuration may produce toxic content under another. However, current LLM identification techniques (such as fingerprinting) focus on intellectual property protection, and their design favors robustness to changes in these instance-level parameters. This poses a critical challenge for AI regulation in which compliance assessments target actual deployed behaviors, not model provenance. In this paper, we introduce instance-level fingerprinting, a regulator-oriented paradigm that distinguishes configurations of the same LLM. Our method FLIPS, exploits biases in generated binary random sequences to reach 96% (closed-set) and 90% (open-set, where some targets are unknown) identification accuracy across 237 model instances, versus 35% for the adapted LLMmap baseline. This shows that instance-level fingerprinting is both necessary for regulation and practically feasible. Code available at this https URL. 

---
# Calibration Data Trade-offs Across Capability Dimensions: Why Multi-Source Mixing Matters for High-Sparsity LLM Pruning 

**Authors**: Hu Xu, Zhaolong Xing, Congcong Liu, Jiaxing Wang, Zhida Jiang, Junshi Huang, Zhen Chen, Jianfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03328)  

**Abstract**: Post-training pruning compresses large language models to high sparsity using a small unlabelled calibration set, and recent work has concluded that the choice of calibration source has only modest impact on averaged post-pruning accuracy. We ask whether this conclusion survives once calibration impact is evaluated separately across distinct capability dimensions rather than aggregated. Decomposing post-pruning capability into General, Commonsense, Code, and Math, and analysing $n{=}15$ calibration sources via Spearman correlations between OIT information metrics and per-dimension retention, we uncover an opposite-sign trade-off: calibration perplexity correlates positively with General retention ($\rho{=}{+}0.71$) but negatively with Math and Code retention ($\rho{=}{-}0.53,\,{-}0.59$; $p{<}0.05$), so no single source can preserve all capabilities. We respond with multi-source calibration mixing, and propose IGSP, an information-guided self-calibration protocol that automates multi-source construction without capability-aligned corpora by minimising 4-gram aggregation and balancing perplexity across dimensions. On LLaMA-3.1-8B at SparseGPT 60% sparsity, a uniform multi-source mix reaches 58.8% total retention, outperforming the best single source (MetaMath, 50.0%) by $+8.8$ and the C4 default (40.0%) by $+18.8$; IGSP improves over Self-Cal by $+2.4$ and SGS by $+4.8$. 

---
# dstack-capsule: Pod-Level Remote Attestation for Confidential Workloads on Kubernetes 

**Authors**: Yang Yang, Kevin Wang, Yuanhai Luo, Hang Yin, Jie Cai, Shunfan Zhou, Wenfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03323)  

**Abstract**: The rise of LLM-as-a-Service and other confidential cloud workloads demands cryptographic proof that user data is processed in a trusted, untampered environment. Existing solutions, notably Confidential Containers (CoCo), enforce a strict "one Pod per VM" model that attests only the Guest OS stack, leaving container-level identity unverified and incurring prohibitive per-VM resource overhead. We present dstack-capsule, a Kubernetes platform that enables Pod-level remote attestation on Intel TDX by allowing multiple Pods to share a single Confidential VM while each retains independent, hardware-backed proof of identity. Our key insight is a two-layer attestation architecture: static platform measurements are frozen in RTMR[3] via an irreversible privilege fuse, while dynamic Pod identities (pod_uid, pod_spec_hash, workload_id) are embedded in the TDX Quote's report_data field and signed by hardware on every request. dstack-capsule introduces (1) a Pod-level attestation protocol binding Pod spec digests to hardware-signed Quotes; (2) a privilege fuse mechanism that atomically transitions a node from setup mode to secure mode; (3) a multi-layer sandbox spanning storage, runtime, admission, API, and network isolation layers; and (4) a complete open-source implementation based on Kubernetes 1.32, Intel TDX, and Sysbox. We evaluate the security properties, attestation correctness, and performance characteristics of dstack-capsule, demonstrating that it achieves Pod-granularity verification without the resource overhead of per-VM isolation. 

---
# Multi-Modal Graph Neural Network with Transformer-Guided Adaptive Diffusion for Preclinical Alzheimer Classification 

**Authors**: Jaeyoon Sim, Minjae Lee, Guorong Wu, Won Hwa Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.03322)  

**Abstract**: The graphical representation of the brain offers critical insights into diagnosing and prognosing neurodegenerative disease via relationships between regions of interest (ROIs). Despite recent emergence of various Graph Neural Networks (GNNs) to effectively capture the relational information, there remain inherent limitations in interpreting the brain networks. Specifically, convolutional approaches ineffectively aggregate information from distant neighborhoods, while attention-based methods exhibit deficiencies in capturing node-centric information, particularly in retaining critical characteristics from pivotal nodes. These shortcomings reveal challenges for identifying disease-specific variation from diverse features from different modalities. In this regard, we propose an integrated framework guiding diffusion process at each node by a downstream transformer where both short- and long-range properties of graphs are aggregated via diffusion-kernel and multi-head attention respectively. We demonstrate the superiority of our model by improving performance of pre-clinical Alzheimer's disease (AD) classification with various modalities. Also, our model adeptly identifies key ROIs that are closely associated with the preclinical stages of AD, marking a significant potential for early diagnosis and prevision of the disease. 

---
# RobotValues: Evaluating Household Robots When Human Values Conflict 

**Authors**: Jongwook Han, Hyeongjin Kim, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2606.03312)  

**Abstract**: While household robots are often evaluated based on task completion, everyday domestic environments involve value-conflicting situations in which robots are expected to choose actions that prioritize other values than task success, such as human autonomy, efficiency, or social appropriateness. Yet, there are no benchmarks for evaluating robots' value preferences in such scenarios. We introduce RobotValues, a benchmark to evaluate household robot planners in 10K value-conflict scenarios. Each instance consists of a realistic household image with multiple plausible robot actions that prioritize different human values. We construct RobotValues through LLM-assisted scenario generation, stakeholder-grounded value extraction, image generation and automatic quality control. Using RobotValues we evaluate VLMs used in robotics and find that models exhibit default value preferences, including safety and accommodation, while underselecting privacy-prioritizing actions. When the models are instructed to prioritize specific values that conflict with their own preferences, they often fail to override their default actions, choosing incorrect actions for 80% of the time. These findings suggest that household robot evaluation should measure not only task completion or safety compliance, but also whether robots can choose among plausible actions when human values conflict. 

---
# Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis 

**Authors**: Jaeyoon Sim, Soojin Hwang, Seunghun Baek, Guorong Wu, Won Hwa Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.03310)  

**Abstract**: Understanding complex interactions between brain regions is critical for early neurodegenerative disease classification such as Alzheimer's Disease (AD) and Parkinson's Disease (PD). While graph-based models are widely used to analyze brain networks, most existing approaches primarily focus on pairwise interactions between directly connected nodes, limiting their ability to capture higher-order dependencies across multiple regions. Although hypergraph-based methods have been proposed to model higher-order relations, many rely on predefined hyperedges or restrict learning to hyperedge weights, reducing flexibility and limiting their capacity to capture multi-resolution structural patterns. In this regard, we introduce an adaptive multi-scale hyperedge learning framework, i.e., MuHL, which constructs hierarchical node features and dynamically learns high-order interactions through continuous hyperedge construction over multi-resolution graph signals. Extensive experiments on multiple brain network benchmarks demonstrate that MuHL consistently improves disease classification performance across different stages, and further identifies key regions of interest (ROIs) and their group-wise interactions from the learned hyperedges that are associated with disease progression, highlighting its potential as a powerful tool for brain network analysis in neurodegenerative disorders. 

---
# Generalizing Graph Foundation Models via Hyperbolic Retrieval-Augmented Generation 

**Authors**: Yifan Jin, Qirui Ji, Bin Qin, Jiangmeng Li, Lixiang Liu, Fuchun Sun, Changwen Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.03307)  

**Abstract**: Graph foundation models (GFMs) emerged as a dominant paradigm in graph representation learning by leveraging large-scale pre-training for cross-domain inference. However, the parameterized knowledge encoded within these models is insufficient to cope with distribution shifts, limiting their generalization ability. To mitigate this issue, retrieval-augmented generation (RAG) has been introduced to incorporate external knowledge at inference time. Nevertheless, existing RAG frameworks operating in Euclidean space suffer from a fundamental geometric limitation: the polynomial volume growth of Euclidean space is inherently mismatched with the tree-structured external knowledge bases. This mismatch leads to the loss of semantic granularity in retrieval and gives rise to the hubness this http URL address this limitation, we propose a Hyperbolic Retrieval-Augmented Generation (HyRAG) framework designed to enhance the generalization capabilities of GFMs. Specifically, the introduced Hyperbolic Knowledge Indexing module retains the tree-like hierarchies of the external knowledge base by modeling them within hyperbolic space. The Multi-granularity Retrieval module then provides GFMs with the global semantic anchors and local semantic nuances through coarse-grained and fine-grained knowledge retrieval, respectively. Finally, the Dual-path Fusion module achieves effective knowledge integration for graph tasks at both the feature and structural this http URL on multiple graph benchmarks demonstrate significant improvements in the zero-shot setting, highlighting the generalization of our method for robust GFMs inference. 

---
# Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective 

**Authors**: Yancheng Chen, Dun Ma, Shuai Zhang, Yang Liu, Xixun Lin, Xiangyu Zhao, Wenguo Yang, Wei Chen, Chuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.03290)  

**Abstract**: Graph Foundation Models (GFMs), built upon the Pre-training and Adaptation paradigm, have emerged as a research hotspot in graph learning. For GNN-based GFMs, graph prompt tuning has become the prevailing adaptation method for downstream tasks. Although recent methods explain why graph prompt tuning works, how to rigorously measure its adaptation capacity remains an open problem. Addressing this problem is critical for understanding the capability limits of graph prompt tuning and for developing more powerful adaptation methods. In this paper, we propose Prismatic Space Theory (PS-Theory), a novel mathematical framework to quantify the capacity of adaptation methods, while focusing on establishing the upper bound for the adaptation capacity of graph prompt tuning. Building upon the proposed PS-Theory, we further introduce Message Tuning for GFMs (MTG), a lightweight approach that injects a small set of learnable message prototypes into each layer of the GNN backbone to adaptively guide message fusion without updating pre-trained weights. Through our PS-Theory, we prove that the adaptation capacity of MTG can exceed the theoretical upper bound of graph prompt tuning. Extensive experiments demonstrate that MTG consistently outperforms graph prompt baselines across diverse benchmark datasets, providing strong empirical support for our theoretical findings. 

---
# AI-Generated Traces for Novice Programmers: Learning Effects and Learner Differences in a Multi-Institutional Study 

**Authors**: Yuri Noviello, Naaz Sibia, Anastasiia Birillo, Thomas Overklift Vaupel Klein, Michael Liut, Gosia Migut  

**Link**: [PDF](https://arxiv.org/pdf/2606.03288)  

**Abstract**: Introductory programming (CS1) courses often struggle to support students' understanding of program execution. While visualizations can make execution processes explicit, their effectiveness depends on design and context, and empirical evidence for AI-generated visualizations remains limited. We propose Generated Animated Traces (GATs), AI-generated, analogy-based, narrated animations that coordinate source code, execution state, and conceptual analogies. We conduct a study at two institutions in CS1 courses (Python, N=961; Java N=151) comparing GATs to textual explanations. We measure immediate learning performance and experience, end-of-course engagement and exam performance. Results show that GATs can yield selective benefits for immediate learning, but benefits are context-dependent and short-term. We observe that GATs' influence on performance is moderated by learner engagement profiles. This finding underscores the importance of personalized approaches. 

---
# VistaHop: Benchmarking Multi-hop Visual Reasoning for Visual DeepSearch 

**Authors**: Hang He, Chuhuai Yue, Chengqi Dong, Chengcheng Wan, Ting Su, Haiying Sun, Jiajun Chai, Xiaohan Wang, Guojun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2606.03273)  

**Abstract**: Visual DeepSearch requires multimodal large reasoning model (MLRM) agents to answer complex visual queries by repeatedly inspecting image regions, grounding intermediate reasoning in visual evidence, and connecting fine-grained clues across long reasoning chains. However, existing benchmarks mainly focus on single-step visual understanding or static image-question answering, offering limited evaluation of iterative image inspection, visual-anchor grounding, and multi-hop evidence integration. In this work, we introduce VistaHop, a benchmark for evaluating vision-centric search and multi-hop visual reasoning in Visual DeepSearch. VistaHop contains 300 high-resolution images, 25 visual search scenarios, and 350 multi-hop QA tasks that require models to follow evidence chains from visual anchors or fuse information across multiple image-grounded reasoning paths. We further develop VistaArena, a unified evaluation environment that supports tool-augmented reasoning with text search, image search, image cropping, and evidence-based answer validation. Experiments on seven representative MLRMs show that current models remain far from solving VistaHop: the best model, SenseNova-MARS-32B, achieves only 24.31% Pass@1. These results reveal persistent limitations in visual grounding, evidence revisiting, long-chain reasoning, and multi-anchor information fusion, highlighting the need for stronger benchmarks and training methods for Visual DeepSearch. 

---
# Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles 

**Authors**: Li Sun, Zhenhao Huang, Yiding Wang, Qin Chen, Pietro Lio, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03270)  

**Abstract**: Foundation models have sparked a revolution via a pretraining-adaptation paradigm, with recent efforts extending this success to graphs. Unlike other modalities, graphs contain rich structural patterns, yet their structural transferability remains poorly understood. Prior studies consider common substructures in the discrete realm, and we are motivated by a fundamental question: Are common substructures transferable? The underlying theory is largely underexplored. In this work, we shift toward learning transferable structures through the lens of functional behavior. Theoretically, we connect transferable substructures to intrinsic geometry of the representation space. However, characterizing such intrinsic geometry has rarely been touched. Grounded in Riemannian geometry, we develop a graph intrinsic geometry learning framework called Neural Vector Bundle, which enables parsing intrinsic geometry with local coordinates. Building on this, we design GAUGE, a pretrainable neural architecture that constructs the vector bundle, flattening geometrically compatible local coordinates, and a new Dirichlet loss, which also measures the transfer effort. We empirically validate its superior expressiveness in challenging tasks including zero-shot link prediction and graph isomorphism. 

---
# EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs 

**Authors**: Sungwon Kim, Juho Song, Seungmin Shin, Guimok Cho, Sangkook Kim, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.03260)  

**Abstract**: Deep learning surrogates for 3D Partial Differential Equations (PDEs) often fail to generalize across geometric transformations because they depend heavily on specific coordinate systems. While equivariant networks offer a solution, they typically rely on local operations in the spatial domain, making the global receptive field, which is essential for PDE dynamics, computationally expensive. Conversely, Fourier Neural Operators (FNOs) efficiently capture global interactions, yet establishing 3D equivariance within them remains impractical due to the prohibitive cost of spectral group convolutions. To bridge this gap, we introduce EqGINO, a geometrically robust framework that enforces isotropy in the spectral domain. By design, EqGINO guarantees exact equivariance to the discrete symmetries inherent to the discretized computational domain. Beyond this discrete guarantee, our structural prior enables effective generalization to arbitrary continuous orientations even with a limited number of SE(3)-transformed training samples. Consequently, our method robustly models coordinate-invariant physical laws on complex irregular 3D geometries. Our code is available at this https URL 

---
# PSViT: A Methodology for Structurally Pruning Spiking Vision Transformers 

**Authors**: Rachmad Vidya Wicaksana Putra, Achyuta Muthuvelan, Alberto Marchisio, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2606.03257)  

**Abstract**: Spiking Vision Transformer (SViT) models are promising low-power ViT models for solving vision-based tasks with state-of-the-art performance. However, their large sizes limit their deployments for resource-constrained embedded platforms, underscoring the needs of model compression. One of prominent compression techniques is pruning, and the state-of-the-art works employ unstructured pruning techniques to compress SViT models. Such techniques require specialized hardware architectures tailored for the sparsity patterns to maximize their efficiency benefits, making this approach not scalable. To address this, we propose PSViT, a novel methodology to perform structured pruning on SViT models, hence making it possible to efficiently accelerate their inference using the existing and widely-used computing architectures. To do this, PSViT employs several key steps: uniform channel-wise filter pruning to structurally eliminate the non-significant weights, sensitivity analysis to evaluate the impact of channel-wise pruning of individual layer on accuracy and network size, as well as fine-grained channel-wise pruning based on the sensitivity analysis and the given network architecture. Experimental results show that PSViT effectively obtains 22.4% memory saving through single-shot pruning, while maintaining high accuracy within 3% (70.3% without fine-tuning and 72.8% with fine-tuning) from the original non-pruned SViT model (73.3%) on the ImageNet-1K. These results also show that the PSViT methodology advances the effort in enabling efficient SViT deployments on resource-constrained applications. 

---
# AirDreamer: Generalist Drone Navigation with World Models 

**Authors**: Zian Liu, Andong Yang, Chunkai Yang, Ruidong An, Chao Gao, Guyue Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.03252)  

**Abstract**: Navigating a drone in unseen and cluttered environments requires reliable generalization to unseen scene layouts and understanding of environmental structure relative to the robot's capabilities. Previous methods, which assume the same environment configuration, often rely heavily on human-designed perception pipelines and predefined rules to guide the robot toward the target. This process is environment-dependent and generalizes poorly across environments. Inspired by animal navigation behavior, we design a navigation framework that navigates with a reinforcement-learning-based policy on top of a world-model-based environment understanding to overcome these issues. In addition, a sparse reward function without hand-crafted shaping terms is designed to avoid local minima traps and encourage yaw control behaviors. In simulation and on real drones, our method exhibits emergent capabilities for navigating complex, unseen environments and escaping local optima where other methods fail. In challenging maps, it achieves a 5.3% higher navigation success rate than best baseline. Furthermore, the proposed framework achieves effective sim-to-real transfer without any tuning during deployment. The code will be publicly available. 

---
# When RLHF Fails: A Mechanistic Taxonomy of Reward Hacking, Collapse, and Evaluator Gaming 

**Authors**: Zelalem Abahana  

**Link**: [PDF](https://arxiv.org/pdf/2606.03238)  

**Abstract**: Reinforcement learning from human feedback (RLHF) makes large-scale post-training possible by replacing an underspecified human objective with learned and scalable proxies. The same substitution creates a structured failure surface: optimization can raise the learned reward while external quality falls, degrade both proxy and judge scores, reveal proxy under-alignment, or produce evaluator-specific disagreement. We present an empirical failure-mode study of a compact RLHF pipeline with proximal policy optimization (PPO), direct preference optimization (DPO), uncertainty-penalized PPO (UP-PPO), reward-model uncertainty, approximate policy drift, diversity and repetition diagnostics, and two external LLM judges. Rather than treating reward hacking as a single terminal event, we classify matched transitions between checkpoints using the directions of the learned reward, judge scores, and average judge score. Across 61 checkpoint rows and 1920 row-level transitions, aggressive PPO has the highest localized reward-hacking rate (14.45%; bootstrap 95% CI: 10.16-18.75), while UP-PPO yields lower rates in the same aggressive regime (11.33-10.94%). A pre-transition logistic model predicts future row-level reward hacking with ROC-AUC 0.821, and row-level analysis finds localized reward hacking that checkpoint averages miss in 3 of 12 settings. The central conclusion is methodological: RLHF failures are not only final-model pathologies, but training dynamics that can be classified, localized, and partially anticipated. 

---
# GFFMERGE: Efficient Merging of Graph Neural Force Fields and Beyond 

**Authors**: Parth Verma, Parv P. Singh, Vipul Garg, Ishita Thakre, N. M. Anoop Krishnan, Sayan Ranu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03232)  

**Abstract**: Graph Neural Networks (GNNs) have revolutionized Neural Force Fields for atomistic simulations, achieving near-quantum accuracy at reduced cost, yet adapting these models to new chemical systems requires expensive retraining of foundation models. Inspired by model merging in vision and language processing, we introduce GFFMERGE, the first principled framework for closed-form model merging in GNNs. We exploit the linear structure of message-passing layers and formulate merging as a convex embedding-alignment problem with an analytical solution. Through the first systematic benchmarking of model merging for GNNs, we show that existing methods designed for vision and language catastrophically fail on force field regression, while GFFMERGE recovers performance approaching gold standard joint training. Across molecular (MD17, MD22), solid-state (LiPS20), and large-scale graph benchmarks, GFFMERGE and GNNMERGE (its generic GNN counterpart) achieve 5-27$\times$ speedups while enabling modular composition of specialized models. Remarkably, our closed-form solution alone outperforms all baseline methods before fine-tuning and provides superior initialization for faster, data-efficient convergence. 

---
# BotDirector: Robot Storytelling Across the Symmetrical Reality with Multi-modal Interactions 

**Authors**: Zhe Sun, Meng Wang, Lei Wang, Yuxi Wang, Wanxin Li, Yujia Peng, Zhenliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03223)  

**Abstract**: Robot storytelling offers a unique blend of technological innovation and creative expression that engages children in unprecedented ways. However, the technical aspects are often too complicated for children. We propose an interactive system that facilitates robot storytelling with tangible and natural language interactions. Children arrange the playground with their own stuff and create narratives with an LLM agent. The created narratives are transformed into a motion sequence based on the map and characters, and the motions are executed by self-navigating swarm robots. This system enhances robot storytelling with flexible scenarios, enabling young children to create robot dramas with everyday objects. 

---
# WebRISE: Requirement-Induced State Evaluation for MLLM-Generated Web Artifacts 

**Authors**: Yuxin Meng, Yuhan Suo, Junjie Wang, Yuhan Sun, Yiyao Yu, Ruixu Zhang, Ruining Hu, Yubin Wang, Shouwei Ruan, Bin Wang, Yuxiang Zhang, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03220)  

**Abstract**: Existing benchmarks for MLLM-generated web artifacts assess interaction through local evidence and miss the requirement-induced states and transitions that determine whether a page works. We introduce WebRISE, which compiles task requirements into Interaction Contract Graphs (ICGs) of observable states, user-intent transitions, and DOM/visual assertions for implementation-agnostic browser execution. WebRISE spans 442 tasks across five input modalities (Text, Markdown, Sketch, Image, Video), with 5,495 transitions and 5,271 requirement checks that separate user-stated functions from implicit product-level constraints. Across 14 MLLMs, even the strongest model reaches only 65.6% transition validity and 66.3% requirement coverage, and visual quality is no proxy for behavior (Qwen3.6-35B-A3B on Markdown: V=80.8 yet T=15.5). Video gives the strongest interaction signal (+10.6 pp implicit coverage over Text), while implicit constraints persist; defect injection shows ICG-based scoring detects state errors at 2-16x the rate of checkpoint-style evaluation. 

---
# Reinforcement Learning from Cross-domain Videos with Video Prediction Model 

**Authors**: Zhao Yang, Xinrui Zu, Jacob E. Kooi, Thomas Delliaux, He Liu, Shujian Yu, Kevin Sebastian Luck, Vincent François-Lavet  

**Link**: [PDF](https://arxiv.org/pdf/2606.03201)  

**Abstract**: Reinforcement learning from expert videos across visually distinct domains is challenging due to the absence of reward signals and the presence of domain gaps. We introduce XIPER (Cross-domain Video Prediction Reward), a reward model for learning from expert videos collected in a visually different domain, where the agent's appearance differs due to factors such as color, morphology, or the sim-to-real gap. More specifically, XIPER trains a cross-domain video prediction model that maps agent observations into the expert domain and uses the prediction likelihood as a reward signal. Experiments on the DMC Color Suite (8 tasks) and DMC Body Suite (3 tasks) show that XIPER consistently outperforms baselines despite domain gaps such as differences in agent color and morphology. We further analyze XIPER on a sim-to-real transfer dataset, demonstrating that it produces meaningful reward signals for real-robot observations given only simulated expert videos. Code, pretrained models, datasets and video demonstrations can be found on our project webpage: this https URL 

---
# AI Rater Discrimination Depends on Scoring Protocol in Complex Clinical Decision-Making 

**Authors**: Sangwon Baek, Kyu Yeon Hur, Kyunga Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.03198)  

**Abstract**: Clinical AI evaluation increasingly delegates scoring to large language models (LLMs) acting as AI raters, yet their scoring behavior across evaluation conditions has not been quantitatively characterized. We address this gap through a factorial study of AI rater behavior in adult type 2 diabetes (T2D) pharmacotherapy at 12-month outpatient follow-up, a clinical task involving complex decision-making operationalized across seven evaluation questions. Four open-source LLMs served simultaneously as clinical decision support system (CDSS) models and AI raters. Each CDSS output was scored under two scoring protocols: a rubric-anchored Gold Rubric (GR) protocol incorporating a patient-specific rubric, and a rubric-free Non Gold Rubric (Non-GR) protocol. Linear mixed effects models crossed the scoring protocol factor with five design factors -- CDSS model, CDSS prompt configuration (document-referenced generation [DRG] vs.\ Baseline), rater model, prompt character, and prompt type -- and estimated main effects together with their protocol interactions. Across all questions, AI raters yielded consistently higher scores within a very narrow range (74--78 points on average) under Non-GR compared to those under GR (7.69 to 49.64 points lower mean scores; 1.68 to 3.67 times wider interquartile ranges). Within each question, GR amplified the AI rater's discrimination between DRG and Baseline CDSS outputs by factors of 1.76 to 5.10, while also revealing substantial behavioral variation across rater models that Non-GR suppressed. These findings support rubric anchoring as the scoring protocol that preserves discriminative power in clinical AI evaluation; rubric-free scoring cannot substitute when questions require patient-specific or jurisdiction-specific criteria that rater models cannot infer from parametric knowledge alone. 

---
# Fully Automated Identification of Lexical Alignment and Preference-Stage Shifts in Large Language Models 

**Authors**: Thomas Stephan Juzek, Xiaoyang Ming, Jose A. Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2606.03165)  

**Abstract**: The language used by digital chat assistants such as ChatGPT can diverge from human expectations (misalignment). Research, mostly on Scientific English, has described both what divergences occur and, to some extent, why, linking them to the training stage of human preference learning. Yet, existing approaches rely on manual curation. This paper introduces two curation-free, assumption-light evaluation metrics: the Lexical Alignment Score, which identifies lexical overuse, and the Triangulated Preference Shift, which quantifies how much of such shifts can be attributed to human preference learning. Using PubMed abstracts, continuations were generated and measured using windowed document prevalence across six model families (Falcon, Gemma, Llama, Mistral, OLMo, Yi). The procedure identifies, without manual intervention, overused items such as 'suggest', 'additionally', and 'strategy', and estimates their link to preference learning. Our findings replicate prior work and remain stable across parameter settings, random seeds, and evaluation on further data. The approach scales readily and enables systematic study of lexical (mis)alignment beyond Scientific English and across languages, and as such, the metrics have the potential to contribute to improved alignment for future models and understanding of its origins. 

---
# OpenAgenet/OAN: Technical Architecture for Trust-Governed Agent Identity and Discovery 

**Authors**: Jinliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03163)  

**Abstract**: This paper describes the technical architecture of OpenAgenet / OAN. OAN is a protocol-neutral trust layer for open Agent interconnection. It specifies the role architecture, identity objects, registration workflow, Root-governed lifecycle, Root-verified package model, authorization-aware Discovery, signed trusted invocation, verification requirements, state transitions, security properties, implementation boundaries, and deployment considerations. The design is intended to support heterogeneous Agent frameworks and interaction protocols, including MCP, A2A, ANP-like systems, and domain-specific Agent protocols. OAN does not define the entire business conversation among Agents; it defines how Agent identities become admissible, discoverable, verifiable, and safe to approach before protocol-specific interaction begins. 

---
# OpenAgenet/OAN: Open Infrastructure for Trusted Agent Interconnection 

**Authors**: Jinliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03161)  

**Abstract**: OpenAgenet, abbreviated as OAN, is an open infrastructure project for trusted Agent interconnection. It addresses a problem that becomes visible when Agents move from isolated applications into open, multi-operator networks: before an Agent can safely discover, select, and invoke another Agent, it needs a way to verify identity provenance, governance state, discovery authorization, freshness, and pre-connection trust evidence. OAN is designed as a protocol-neutral trust layer. It does not replace Agent interaction protocols, tool protocols, model orchestration frameworks, or application-level workflows. Instead, it provides Root-governed identity admission, Registrar-assisted onboarding, Root-verified package publication, authorization-aware Discovery, and signed trusted invocation. This paper presents the motivation, architecture, roles, governance model, relationship with MCP, A2A, and ANP, deployment patterns, cooperation model, blockchain-backed authorization bulletin, prototype status, performance profile, and roadmap of OAN. 

---
# NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation 

**Authors**: NVIDIA, Aarti Basant, Amlan Kar, Despoina Paschalidou, Fangyin Wei, Francesco Ferroni, Guillermo Garcia Cobo, Haithem Turki, Huan Ling, Jaewoo Seo, James Lucas, Jay Zhangjie Wu, Jialiang Wang, Jonathan Lorraine, Jun Gao, Kai He, Katarina Tothova, Kevin Xie, Michał Tyszkiewicz, Qi Wu, Riccardo de Lutio, Ruilong Li, Sanja Fidler, Seung Wook Kim, Tianchang Shen, Tianshi Cao, Tobias Pfaff, William Lew, Xindi Wu, Xuanchi Ren, Yifan Lu, Yuxuan Zhang, Zan Gojcic, Zian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03159)  

**Abstract**: As autonomous vehicle capabilities advance, the safe evaluation of driving policies in long-tail scenarios remains a critical bottleneck. In closed-loop simulation, the driving policy model actively interacts with the environment, where its actions dynamically update the simulator state and directly influence the next set of generated sensor observations. While recent reconstruction-based neural simulators offer photorealism, they are fundamentally constrained by their initial captured data and struggle to generalize to highly dynamic or novel scenes. To overcome these limitations, we introduce OmniDreams, a foundation generative world model mid- and post-trained from the Cosmos diffusion model to autoregressively generate action-conditioned videos in real time. By leveraging the rich visual priors of Cosmos and mid- and post-training on 21k hours of driving scenarios, OmniDreams synthesizes complex, unobserved phenomena that are hard for traditional simulators to capture, such as extreme weather and unpredictable dynamic agent behaviors. Crucially, it autoregressively conditions its photorealistic sensor generation on past frames, the current simulator state, and immediate driving actions. Deployed in a closed-loop system with the Alpamayo 1 policy model and AlpaSim orchestrator, OmniDreams acts as a highly responsive, reactive environment, providing a scalable and comprehensive solution for training and evaluating next-generation autonomous driving policies. We additionally show preliminary results indicating that a world-action model (WAM) post-trained from OmniDreams achieves strong performance on the Physical AI Autonomous Vehicles NuRec dataset, surpassing the VLA-based Alpamayo 1.5 research policy model while using only 1/5 the total parameters. These results highlight the potential for a real-time world model like OmniDreams to also serve as a backbone for policy architectures. 

---
# Decoupled Smart Contract Audits: Lightweight LLM Framework via Distillation and Aggregation 

**Authors**: Bagus Rakadyanto Oktavianto Putra, Muhamad Risqi Utama Saputra, Widyawan, Guntur Dharma Putra  

**Link**: [PDF](https://arxiv.org/pdf/2606.03128)  

**Abstract**: Smart contracts face critical security challenges that require thorough auditing in decentralized web services. While Large Language Models (LLMs) have shown promise in automated vulnerability detection, existing approaches lack severity evaluations with actionable remediation and demand unnecessarily massive computational overhead. In this study, we introduce an efficient end-to-end smart contract security audit framework utilizing lightweight, highly optimized open-source LLMs (0.6B-4B parameters). Our framework decouples comprehensive audit tasks into four interconnected components: vulnerability detection, explanation, severity classification, and remediation recommendation. To maintain high accuracy without massive parameters, we implement Rank-Stabilized Low-Rank Adapters (rsLoRA), knowledge distillation, and a custom Chain-of-Verification (CoVe) aggregation strategy to systematically screen and consolidate multiple draft responses from the model into a highly accurate audit report. Experimental results demonstrate that our lightweight pipeline consistently outperforms state-of-the-art open-source coder dense LLMs (7B to 34B parameters), achieving 98.25% accuracy in vulnerability detection and an alignment score of 0.4375 in generative explanation tasks. Furthermore, our extensive ablation studies empirically validate the superiority of our decoupled audit processes over unified prompting and uncover a novel severity centrality bias, establishing a critical benchmark for future research in LLM-assisted auditing. 

---
# GuidedBridge: Training-freely Improving Bridge Models with Prior Guidance 

**Authors**: Zehua Chen, Yucheng Yang, Binjie Yuan, Kaiwen Zheng, Jun S. Liu, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03119)  

**Abstract**: Guidance methods, such as classifier-free guidance (CFG) and auto-guidance (AG), have advanced noise-to-data generation in diffusion models. Recently, bridge models have introduced a data-to-data generative process that can exploit an instructive clean prior. In this work, inspired by previous methods creating quality difference between denoising results as guidance, we propose a training-free bridge guidance method, termed Prior Guidance (PG). Specifically, we introduce a weak prior, which is unseen during bridge pre-training, hindering prior exploitation and thereby degrading denoising result. Then, we contrast it with the seen prior to highlight and enhance prior exploitation via a scaling factor. Moreover, we analyze the underlying mechanism of prior exploitation in the bridge process and design frequency-modulated prior guidance (FMPG), which tailors the guidance scale to low- and high-frequency bands coherent with bridge generative dynamics. To address prior exploitation in image in-painting, we develop a cascaded framework, CFG-FMPG, which first generates a noisy hidden representation via CFG and then exploits it as a generative prior with FMPG, fulfilling their complementary strengths without compromising inference efficiency. Experiments demonstrate that our PG methods consistently improve pre-trained bridge models across diverse image translation tasks. 

---
# AnyAudio-Judge: A Dynamic Rubric-Based Benchmark and Evaluator for Audio Instruction Following 

**Authors**: Haitao Li, Tian Tan, Yuguang Yang, Shan Yang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.03116)  

**Abstract**: The rapid advancement of instruction-guided audio generation has highlighted the critical need for robust alignment evaluation. Current automated evaluation methods heavily rely on holistic scoring from general-purpose large language models, which struggle to decouple complex instructions, lack interpretability, and fail to capture fine-grained attribute mismatches. To address this, we introduce a novel dynamic rubric-based evaluation paradigm that adaptively decomposes complex audio captions into a variable number of independent, verifiable binary rubric items. To rigorously benchmark this capability, we propose the AnyAudio-Judge Bench, a comprehensive, bilingual benchmark comprising 7,920 meticulously curated samples across four diverse audio domains (speech, sound, music, and mixed), featuring deliberately constructed hard negatives. Furthermore, we construct a large-scale corpus of 105K samples with explicit Chain-of-Thought (CoT) rationales to train our dedicated evaluator, the AnyAudio-Judge model. By employing a training pipeline that combines Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO), our model successfully aligns its reasoning paths with the rubric-based scoring mechanism. Extensive experiments demonstrate that AnyAudio-Judge not only significantly enhances zero-shot alignment detection compared to state-of-the-art baselines, but also provides precise and interpretable reward signals that substantially improve instruction alignment in downstream reinforcement learning for audio generation. 

---
# PhotoCraft: Agentic Reasoning with Hierarchical Self-Evolving Memory for Deep Image Search 

**Authors**: Kailin Lyu, Zhiqiang Yuan, Jianwei He, Qiwei Yan, Xuanbo Su, Nanxing Hu, Yang Liu, Ce Hao, Shengqian Qin, Lianyu Hu, Jinchao Zhang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.03099)  

**Abstract**: Deep Image Search requires multi-step reasoning over rich contextual cues, such as time, location, and event relations. However, most existing LLM-based agents are stateless and reactive, lacking persistent memory to maintain long-horizon context or transfer experience across tasks, which often leads to execution drift and experience isolation. To address these limitations, we propose PhotoCraft, a training-free, hierarchical memory system for photo-search agents. Inspired by human cognition, PhotoCraft equips MLLMs with working, episodic, and semantic memory, which are dynamically invoked during reasoning to preserve logical consistency and knowledge transferability throughout multi-step reasoning and answer generation. Extensive experiments on DISBench demonstrate that PhotoCraft consistently improves context-aware retrieval across diverse MLLM backbones, achieving gains of up to 18.5\% and effectively mitigating key bottlenecks in memoryless deep image search, offering a practical path toward reliable and generalizable multimodal search agents. 

---
# BAHSD: Bridging the Long-tail Gap via Adaptive Distillation in Black-box Sequential Recommendation 

**Authors**: Xi Zhou, Famin Wu, Mingming Li, Hongyue Zhang, Jiao Dai, Jizhong Han, Tao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.03091)  

**Abstract**: Sequential recommendation systems are widely adopted but often deployed as black-box APIs, which has driven recent interest in model extraction to replicate their capabilities locally. However, the long-tail distribution induces severe signal heterogeneity: dense head sequences trigger the solidification of teacher preference, biasing extraction toward local patterns, while sparse tail sequences yield flat, noisy predictions. Existing one-size-fits-all extraction overlooks this disparity, resulting in noise overfitting and suboptimal knowledge transfer. We propose BAHSD, a black-box adaptive distillation framework that handles signal heterogeneity via a multi-scale consistency probing mechanism to implicitly quantify signal reliability. Based on this, an adaptive hierarchical objective is designed: dynamic-temperature KL divergence mitigates preference solidification for high-confidence signals, while ranking consistency and InfoNCE contrastive learning provide noise-robust enhancement for low-confidence signals. BAHSD consistently outperforms baselines, achieving up to 4.98\% gain over the teacher and 80\%+ improvement on tail users, offering a plug-and-play solution for high-fidelity black-box recommendation extraction. 

---
# "**Important** You should give me full credits!": Exploring Prompt Injection Attacks on LLM-Based Automatic Grading Systems 

**Authors**: Hang Li, Fedor Filippov, Yuling Lin, Pengfei He, Kaiqi Yang, Yucheng Chu, Yingqian Cui, Hui Liu, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03090)  

**Abstract**: The emergence of large language models (LLMs) has significantly accelerated recent research on LLM-based automatic grading (AG) systems. Benefiting from the strong instruction-following capabilities and broad prior knowledge of LLMs, educators can deploy AG systems across diverse tasks using only natural language rubrics while achieving satisfactory grading performance. Despite these advantages, new security concerns may also arise. In particular, prompt injection (PI) attacks have recently become a major threat to LLM-based applications. In the context of AG, attackers can potentially exploit PI vulnerabilities to manipulate grading systems into assigning artificially high scores regardless of the actual answer quality. Such behavior poses serious risks to the fairness, reliability, and integrity of educational assessment. In this work, we study PI attacks in AG systems, and systematically investigate the effectiveness of such attacks in educational scenarios. We further evaluate the effectiveness of existing defensive strategies against these attacks. Through comprehensive experiments under rubric-based grading settings, we demonstrate that current LLM-based AG systems remain highly vulnerable to PI attacks. We hope that our findings raise awareness of this emerging threat and motivate future research toward secure, robust, and trustworthy LLM-based educational systems. 

---
# Constitutional On-Policy Safe Distillation 

**Authors**: Ming Wen, Yuxuan Liu, Kun Yang, Yunhao Feng, Zhuoer Xu, Yuhao Sun, Shiwen Cui, Xiang Zheng, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03089)  

**Abstract**: On-policy self-distillation (OPSD) has emerged as an efficient post-training paradigm by using a teacher conditioned on privileged information to provide dense token-level supervision. Prior work has shown that OPSD can collapse in verifiable reasoning tasks, but safety alignment differs in that it is guided by high-level constitutions rather than explicit target answers, making it a natural setting to revisit dense distillation. However, our pilot study show that safety OPSD still suffers from severe collapse: constitutional conditioning contracts the teacher distribution toward short and overly conservative responses, and Reverse KL further amplifies this contraction into reduced expressiveness. We formalize this effect as geometric leakage under safety boundaries in a non-orthogonal semantic space, where safety pressure transfers into the expressiveness dimension. Based on this analysis, we propose Constitutional On-Policy Safe Distillation (COPSD), which first calibrates the teacher through a Cross-SFT cold-start and then performs constitution-conditioned on-policy distillation. Experiments on 12 benchmarks show that COPSD achieves a consistently stronger safety--helpfulness trade-off than baselines while substantially reducing the safety tax on general reasoning ability. 

---
# Regret Pre-training: Bridging Prior and Posterior Views for Enhanced Knowledge Grounding 

**Authors**: Mingkuan Zhao, Xiayu Sun, Wentao Hu, Suquan Chen, Jiaxuan Li, Xiaoyan Zhu, Xin Lai, Jiayin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03080)  

**Abstract**: Causal language models factorize sequence probabilities using only preceding context, leaving future information unexploited during training despite its availability in the training data. This paper introduces Regret Pre-training, a self-supervised framework grounded in the Learning Using Privileged Information (LUPI) paradigm. The framework employs a dual-view architecture in which a single model generates both a causal Student distribution and a future-conditioned Teacher distribution. The training objective augments standard language modeling with a regret loss that minimizes the KL divergence from teacher to student, transferring future-aware signals to the causal representations. We investigate two teacher configurations on the OLMoE-1B-7B architecture:LocalRegret, which extends attention by one future token, andGlobalRegret, which conditions on bidirectional context with the target position masked. Experiments on nine downstream tasks following 4 billion tokens of training demonstrate that both configurations consistently outperform the baseline. On average,GlobalRegret andLocalRegret achieve 33.9% and 32.2% accuracy respectively, surpassing the baseline's 30.2%. Most notably,GlobalRegret improves BoolQ performance by 18.1 percentage points (61.0% vs 42.9%). The framework introduces no additional parameters and requires only one extra inference-mode forward pass per training step. 

---
# Libra: Efficient Resource Management for Agentic RL Post-Training 

**Authors**: Kaiwen Chen, Xin Tan, Jingzong Li, Hong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03077)  

**Abstract**: Reinforcement learning (RL) has become a standard post-training paradigm for large language models (LLMs), extending beyond preference alignment to complex reasoning and multi-turn agentic behaviors. In agentic RL, the rollout stage generates trajectories while invoking tools, producing long-tailed and non-stationary workloads that challenge conventional resource-management assumptions. Three fundamental challenges arise. First, due to the long-tail distribution, a small fraction of trajectories dominates rollout makespan. Second, rollout and training exhibit strong asymmetry in compute patterns, memory demands, and sensitivity to sequence length. Third, as the RL policy evolves, the trajectory-length distribution drifts over time, rendering any static resource split progressively suboptimal.
We present Libra, which introduces two core mechanisms. The first is a periodic global resource planner that jointly optimizes GPU allocation across rollout and training clusters. It leverages an elastic hybrid pool to enable lightweight, non-blocking worker reallocation between stages. The second is a causality-driven multi-level feedback queue (C-MLFQ) scheduler, which routes requests to heterogeneous rollout buckets based on causal signals derived from tool-return outcomes, rather than relying on fragile length predictions. Evaluated on 48 A800 GPUs, Libra achieves up to 3.0$\times$ higher throughput and converges up to 2.5$\times$ faster in reward compared to the baselines. 

---
# Efficient Hyperparameter Optimization for LLM Reinforcement Learning 

**Authors**: Minping Chen, Bowen Xiao, Du Liang, Chuxuan Zeng, Zeyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2606.03073)  

**Abstract**: Reinforcement learning (RL) for large language models (LLMs) is highly sensitive to hyperparameter configurations, making hyperparameter optimization (HPO) essential yet computationally expensive. Existing multi-fidelity HPO methods remain inefficient for LLM RL due to the massive model scale and resource-intensive training cycles. In this paper, we propose Joint Fidelity Hyperparameter Optimization (JF-HPO), which simultaneously adapts both model size and training budget as fidelity. JF-HPO is empowered by: (i) it leverages a small proxy model of the target LLM for efficient training and evaluation in each HPO trial; (ii) it integrates carefully designed early-stopping strategies based on training dynamics; (iii) it introduces an efficient checkpointing mechanism to eliminate redundant computations. Compared with existing HPO methods, JF-HPO significantly improves the computational efficiency of each trial (up to 14.9 times), while achieving better or competitive predictive accuracy under the same time budget. Notably, compared with utilizing hyperparameter configurations from the VeRL Recipe, JF-HPO delivers performance improvements ranging from 5.8% to 111.6%. 

---
# ASymPO: Asymmetric-Scale Policy Optimization for Asynchronous LLM Post-Training Without Behavior Information 

**Authors**: Zehua Liu, Yuxuan Yao, Xiaojin Fu, Tao Zhong, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.03070)  

**Abstract**: Asynchronous reinforcement learning can improve language-model post-training throughput by decoupling response generation from policy optimization, but stale responses introduce distribution drift. Standard behavior-corrected methods control this drift with behavior-policy probabilities, importance ratios, or clipping, which requires token-aligned, versioned, and numerically consistent behavior log-probabilities across rollout and learner systems. We ask whether asynchronous group-relative RL can instead be stabilized using only current-policy probabilities. We identify a scale-imbalance failure mode: when stale responses are evaluated under the current policy, positive and negative loss terms can appear at different negative log-probability scales, so zero-sum advantages no longer imply balanced loss contributions. We propose Asymmetric-Scale Policy Optimization (ASymPO), which normalizes each response's token loss by its current average token negative log-probability. ASymPO requires no behavior-policy probabilities, restores response-level zero-sum balance, and preserves a nonzero learning signal. We also introduce Scaled Policy Optimization (SPO), a fixed negative-scaling baseline, and evaluate both current-policy-only objectives in asynchronous mathematical reasoning post-training. 

---
# ROBUST-WT: Robust Uncertainty-aware Segmentation Transform via Whitening and Training Enhancements 

**Authors**: Aqsa Naseer, Maryam Bibi, Syeda Samiya Urooj, Muhammad Khurram Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2606.03069)  

**Abstract**: Generalized segmentation of medical images prevents performance degradation when different imaging devices and clinical protocols are used across multiple domains. The Whitening Transform-based Probabilistic Shape Regularization Extractor (WT-PSE), published in IEEE Transactions on Medical Imaging in 2024, addresses this challenge by employing feature decorrelation and Wasserstein distance-based knowledge distillation to achieve robust cross-domain segmentation. This study systematically examines improvements to the WT-PSE learning framework. Four limitations in the original implementation are identified: limited training augmentations that fail to simulate real scanner variations, reliance on per-pixel binary cross-entropy loss that is sensitive to edge noise, the absence of a scheduled loss weighting strategy that may destabilize early training, and the lack of ablation switches for controlled scientific comparison. To address these issues, we propose four enhancements: (1) domain-adaptive augmentation including random erasing, gamma correction, and salt-and-pepper noise; (2) a hybrid BCE and Dice loss function for improved edge-aware segmentation under noisy conditions; (3) a curriculum-based Dice weight scheduling strategy; and (4) command-line control flags for systematic ablation studies. Experiments on the fundus optic disc segmentation benchmark demonstrate that the improved pipeline achieves a final epoch optic-disc Dice score of 0.956 and an ASD score of 13.31, outperforming the baseline epoch-5 Dice score of 0.939. These results indicate that training-level improvements can provide consistent performance gains without modifying the underlying WT-PSE architecture. 

---
# Learn When and Where to Connect: Adaptive Virtual Nodes for Dynamic Message Passing on Graphs 

**Authors**: Jaejun Lee, Joyce Jiyoung Whang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03068)  

**Abstract**: While Virtual Nodes (VNs) are often utilized in Message Passing Neural Networks (MPNNs) to facilitate effective message passing, existing VN-based methods have limitations, such as constraining all nodes to connect to the same number of VNs, fixing the connections before applying MPNNs, and connecting a node to a VN independently of the other nodes that connect to the same VN. We propose MAVN, an end-to-end differentiable MPNN framework that allows non-constrained connections between nodes and VNs and dynamically introduces VNs on demand in response to evolving node representations across layers. Specifically, MAVN learns to adaptively determine when (at which layer) and where (to which nodes) to introduce and connect VNs based on the relative importance of connections. From a pool of candidate VNs, MAVN selects the necessary VNs in each layer, where each selected VN is connected to a nonempty subset of nodes, guided by a dual-perspective scoring mechanism that jointly captures the nodes' preferences for VNs and the VNs' preferences for nodes. We theoretically prove that for any node-VN connectivity pattern, there exists a set of MAVN's parameters that can simulate the pattern. Experiments on nine real-world datasets demonstrate that MAVN consistently improves the performance of backbone MPNNs, achieving up to 46.5% improvement over the backbones and outperforms the baselines. 

---
# Brief Announcement: Generative Markov Model for Distributed Computing Systems 

**Authors**: Alfreds Lapkovskis, Ali Beikmohammadi, Sindri Magnússon, Praveen Kumar Donta  

**Link**: [PDF](https://arxiv.org/pdf/2606.03061)  

**Abstract**: Emerging distributed computing paradigms, such as the computing continuum, are inherently heterogeneous, stochastic, and complex. Efficiently and effectively utilizing all available resources across the continuum demands a unified formal model of the system. To address this gap, we propose a general framework for modeling distributed computing systems as a generative Markov model, factorized over a structured system state. In our model, the state decomposes into high-dimensional variables, each further factorized over its elements, reflecting the sparse dependency structure inherent to distributed systems. This yields a tractable model enabling simulation, inference, and policy learning over otherwise intractable system states, bridging distributed computing with Markov chain theory and reinforcement learning (RL). We demonstrate our framework through a case study of collaborative AI inference, in which a dedicated server combines resources with those volunteered by service users. Our results show that centralized scheduling becomes a bottleneck at scale, while distributing computation across user devices reduces both latency and server resource consumption. These findings highlight the value of adaptive decision-making in distributed computing systems and demonstrate the framework's utility for modeling, simulation, and optimization. 

---
# Rethinking Molecular Text Representations for LLMs: An Empirical Study 

**Authors**: Arun Raja, Garrett M. Morris, Kian Ming A. Chai  

**Link**: [PDF](https://arxiv.org/pdf/2606.03057)  

**Abstract**: Large language models (LLMs) are increasingly used for molecular tasks, but it remains unclear which molecular representation to use. We present a systematic benchmark evaluating LLM molecular competence across nine representations and eight chemical tasks. We benchmark 16 LLMs across five model families, including reasoning and non-reasoning variants, chemistry-specialized LLMs, and closed frontier models. Performance is strongly representation-dependent and no single representation wins across tasks, though CML is the best, followed by MolJSON, InChI, and then canonical SMILES. Explicit structured text representations (CML and MolJSON) dominate structural tasks; IUPAC dominates semantic tasks, winning molecule retrieval for all 16 LLMs; and SMILES variants are rarely optimal despite their prevalence in pretraining. Chemistry-specialized models perform well with SMILES at the cost of large degradations with structured text representations, suggesting SMILES-only evaluation rewards specialization that does not generalize. Using LLM-as-a-judge, we find that IUPAC produces the highest fraction of correct molecule generations. A mechanistic study via tokenization audits, linear probes and attention shows that representations are encoded differently inside the model; for example, structured representations require higher attention across the molecular span. Our results argue against representation-invariant evaluation and motivate task-aware representation routing for LLM-based chemistry. 

---
# Capability Advertisement as a Market for Lemons: A Trust Layer for Heterogeneous Agent Networks 

**Authors**: Gaurav Naresh Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2606.03034)  

**Abstract**: Large language model (LLM) agents have begun to delegate work to one another. Protocols such as the Model Context Protocol (MCP) and the Agent2Agent protocol (A2A) let an agent publish what it can do and let others call it, and public registries of such agents are already appearing. These protocols assume an advertised capability is a static, truthful fact. A real agent is none of these things: its competence is probabilistic, varies with input, drifts when the underlying model is updated, and, because the agent is itself a language model, it can describe itself with complete confidence and be wrong. A caller therefore sees what an agent claims to do, not what it can do, with no principled way to tell a reliable provider from a fluent impostor.
We argue these difficulties share one cause: the market for lemons. When quality is hidden and claims are cheap, good and bad providers become indistinguishable, honest reliability goes unrewarded, and the market decays toward its worst participants. Economics offers three remedies, signaling, screening, and reputation, and none are present in today's agent protocols.
We make four contributions: (1) a failure taxonomy that names confident-wrong as a non-adversarial, correlated subclass of Byzantine faults that classical fault-tolerance mismodels; (2) a market-for-lemons model showing that faith-based protocols admit only a low-trust equilibrium; (3) the Trust Layer, a thin, protocol-agnostic narrow waist above MCP and A2A that adds probabilistic capability descriptors, screening, and reputation, and admits a separating equilibrium when the cost of sustaining an overclaim exceeds the gain from it; and (4) a reliability-composition bound for delegation chains with an end-to-end placement argument. The design needs no model retraining and degrades gracefully when its trust anchors are absent or corrupt. 

---
# Conditional Hypothesis Generation for LLM-Based Text Analysis with Researcher-Specified Covariates 

**Authors**: Paiheng Xu, Jing Liu, Wei Ai  

**Link**: [PDF](https://arxiv.org/pdf/2606.03029)  

**Abstract**: A core goal of computational social science is to discover interpretable differences in how language varies across outcomes of interest, such as political affiliation or instructional quality. Recent LLM-based hypothesis generation methods describe such differences in natural language, but select for globally discriminative patterns without accounting for covariates that shape the data based on researchers' domain knowledge. When covariates are ignored, selected patterns can reflect confounds rather than differences of substantive interest. We introduce conditional hypothesis generation, a framework that incorporates researcher-specified covariates to steer hypothesis discovery toward differences that hold within relevant subgroups. Two challenges arise: the target subgroup may be underrepresented (stratum imbalance), and the direction of a difference may reverse across subgroups (sign reversal). We propose two econometrics-inspired methods: one introduces feature--covariate interactions to detect sign reversals, and the other applies within-stratum demeaning and inverse-frequency reweighting to equalize underrepresented strata. Synthetic experiments show each method outperforms global baselines in its targeted setting, and expert evaluation on two real-world datasets confirms that covariate-aware generation surfaces more useful hypotheses within relevant subgroups. 

---
# Spike-Aware C++ INT8 Inference for Sparse Spiking Language Models on Commodity CPUs 

**Authors**: Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03026)  

**Abstract**: Spiking language models expose activation sparsity that dense Transformer runtimes do not directly exploit. This paper studies that property from a systems perspective. Building on the SymbolicLight V1 spike-gated language model family, we implement a C++ CPU inference runtime that treats sparse binary spike states as an execution primitive rather than only applying post-hoc weight compression. The runtime combines a manifest-driven weight loader, mixed row/column memory layout, AVX2/FMA kernels, per-channel symmetric INT8 quantization, and integer-domain accumulation for spike-conditioned sparse paths. On an AMD Ryzen 7 5800X, an early scalar FP32 baseline decodes at 9.5 tokens/s. Mixed-layout AVX2 FP32 raises this to 14.7 tokens/s, and AVX2 INT8 reaches 19.9 tokens/s on the same step-30k export while reducing the weight footprint from 3.49 GB to 1.06 GB. For the available 186k-step 874M-parameter INT8 export, the C++ runtime decodes at 22.63 tokens/s in a single-thread CPU benchmark, compared with 16.31 tokens/s for TinyLlama-1.1B Q8_0, 11.26 tokens/s for Falcon3-1B Q8_0, and 9.70 tokens/s for Qwen2.5-1.5B Q8_0 under this http URL. Thread scaling reaches 47.90 tokens/s at four CPU threads, and 512-token prefill improves from 29.86 to 94.68 tokens/s from one to eight threads. The throughput result comes with a quality cost: the SNN reports WikiText-2 perplexity 24.80, worse than the dense baselines in the same benchmark. We frame the result as an inference-systems study for sparse language runtimes, with longer-term motivation in embodied and edge agents that may benefit from local, low-core inference near sensors and actuators. Spike-aware execution can improve CPU throughput and memory behavior for sparse spiking language models, while model quality, controlled dense training baselines, embodied-task evaluation, and measured CPU energy remain open problems. 

---
# Hallucinations as Orthogonal Noise: Inference-Time Manifold Alignment via Dynamic Contextual Orthogonalization 

**Authors**: Mingkuan Zhao, Wentao Hu, Tianchen Huang, Yuheng Min, Suquan Chen, Yide Gao, Yanbo Zhai, Shuangyong Song, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.03022)  

**Abstract**: Hallucination in Large Language Models (LLMs), characterized by the generation of content inconsistent with contextual facts or logical constraints -- remains a persistent challenge for reliable deployment. In this work, we address this issue through a geometric framework rooted in the linear representation hypothesis. We propose that hallucinations manifest as orthogonal noise relative to the semantic manifold of the residual stream. Specifically, we hypothesize that while attention heads ideally propagate information congruent with the context subspace, hallucinations arise when specific heads introduce components orthogonal to this subspace, disrupting the coherence of the latent representation. Based on this formulation, we introduce Dynamic Contextual Orthogonalization (DCO), an inference-time intervention method. DCO utilizes the input residual stream as a dynamic context anchor to perform orthogonal decomposition on attention head outputs. To distinguish between context-aligned semantic updates and divergent noise, DCO employs a layer-wise Z-score suppression mechanism that selectively attenuates outlier orthogonal components based on statistical distributions. Evaluations on Llama-3-8B and 70B across benchmarks such as XSum, NQ-Swap, and IFEval demonstrate that DCO achieves superior contextual faithfulness compared to state-of-the-art intervention baselines. Furthermore, DCO maintains high performance on knowledge-intensive tasks like TriviaQA and TruthfulQA, effectively mitigating the trade-off between hallucination suppression and parametric knowledge retention often observed in existing methods. Our findings validate the geometric interpretation of hallucinations and establish DCO as a computationally efficient approach for enforcing manifold this http URL code is available at this https URL 

---
# Reproducibility is the New Copyleft: Defining AGI-oriented Reproducible Builds 

**Authors**: Masayuki Hatta  

**Link**: [PDF](https://arxiv.org/pdf/2606.03019)  

**Abstract**: Copyleft, as implemented in licenses such as the GNU General Public License, was a legal hack that used copyright to guarantee user freedom by tying the availability of source code to every act of distribution. Its normative force rested on an implicit technical premise: that source code and object code stand in a well-defined, humanly auditable, and reproducible relationship. Large language models and, prospectively, Artificial General Intelligence (AGI) systems systematically violate this premise. The artifacts jointly required to reconstruct a model -- code, data, weights, hyperparameters, toolchain, and hardware configuration -- are each subject to independent legal, technical, and economic constraints that no current open-source framework fully resolves. Sufficiently capable AI systems can also rewrite licensed source into functionally equivalent derivatives stripped of their original obligations, a form of laundering against which copyleft has no effective defense. This paper argues that a functional analogue of copyleft for AGI must be grounded not in share-alike clauses over code, but in reproducible builds: a practice guaranteeing bit-exact reconstructability from declared inputs. We review the logic of copyleft, critically examine Maffulli's Second Liberation thesis according to which AI fulfills Stallman's dream, and show that the argument collapses unless AGI systems are themselves reproducible. Drawing on the Open Source AI Definition (OSAID), the Model Openness Framework (MOF), OpenMDW, and deterministic-inference research, we define seven requirements for AGI-oriented reproducible builds. We further argue that the Model Context Protocol (MCP) and analogous AI-to-AI coupling mechanisms constitute a new dynamic linking layer for which copyleft-style licensing is ill-suited, and that Masnick's "protocols, not platforms" framework offers a more promising governance template. 

---
# ConTraIRL: Factorized Contrastive Abstractions for Transferable IRL 

**Authors**: Yikang Gui, Bikramjit Banerjee, Prashant Doshi  

**Link**: [PDF](https://arxiv.org/pdf/2606.03017)  

**Abstract**: Reward transfer in Inverse Reinforcement Learning (IRL) is unreliable when policies must generalize to unseen combinations of environment dynamics and task goals. We propose Factorized Contrastive Abstractions for Transferable IRL (ConTraIRL), a framework that enables compositional reward transfer by learning decoupled latent representations of these two factors. ConTraIRL uses a dual-encoder architecture that maps observations into separate dynamics and goal latent spaces, trained with a dual contrastive objective. Temporal alignment encourages the dynamics encoder to learn goal-invariant structure, while the goal encoder captures dynamics-invariant features. This factorization supports reward inference under recombined dynamics-goal settings. Experiments on continuous control benchmarks demonstrate effective few-shot transfer to unseen dynamics-goal pairings, improving sample efficiency and reward recovery over transfer IRL baselines. 

---
# MUSE: A Unified Agentic Harness for MLLMs 

**Authors**: Jianglin Lu, Hailing Wang, Xu Ma, Qihua Dong, Mingyuan Zhang, Yizhou Wang, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2606.03005)  

**Abstract**: Despite rapid progress, multimodal large language models (MLLMs) still fail on tasks that humans solve effortlessly, such as navigating a grid maze from a screenshot or selecting the correct puzzle piece. Rather than retraining the model, we ask a complementary question: how much capability can be elicited from a frozen MLLM purely by improving the execution scaffold around it? We introduce MUSE, a multimodal unified structured execution harness that wraps any off-the-shelf MLLM with composable modules for task representation, visual processing, perception tool use, structured parsing, deterministic verification, and verifier-guided repair, without any model retraining. We evaluate MUSE across diverse benchmarks spanning visual spatial planning, visual perception, multimodal reasoning, and fine-grained visual discrimination, using multiple state-of-the-art MLLMs. MUSE delivers consistent gains over the bare model in all settings, with the largest jumps on challenging instances. Further analysis reveals that many MLLM failures arise from harness-level shortcomings rather than fundamental model deficits, and can be addressed through verifier-guided repair without touching the model. These findings highlight the agentic multimodal harness as a critical yet underexplored design dimension, offering an orthogonal avenue for improving MLLMs beyond model-centric optimization. 

---
# Exact equivariance, kept through training, buys zero-shot generalisation across the symmetry group 

**Authors**: Hongbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.03003)  

**Abstract**: A latent world model built from an equivariant encoder $E$ and an equivariant predictor $f$ inherits a provable symmetry of its training loss: when the world's dynamics genuinely carries a group $G$ acting on latents by an orthogonal representation $\rho(g)$, the one-step prediction relMSE is exactly invariant across the whole group, so fitting the dynamics on a restricted slice of orientations mathematically determines it on the entire orbit (jǔ yī fǎn sān). We verify this end-to-end at laptop scale (CPU/MPS, fully seeded). [A] The symmetry survives a real Muon/AdamW + EMA + VICReg run -- composed encode-then-predict residual $\sim 10^{-6}$ after optimisation, not just at initialisation, and under any optimiser. [B] One-step error is flat to five digits across the group, while a same-hypothesis-class non-equivariant baseline fits the slice but breaks out-of-distribution (VN $\times 1.00$ vs baseline $\times 13.8$ in 2D, $\times 17.2$ in 3D, $\times 157$ over the full $\mathrm{SE}(3)$ ladder), with the equivariant model $4.5$-$7.4\times$ smaller. [C] The same isometry argument lifts to closed loop: under a matching equivariant planner the control trajectory at orientation $g$ is exactly $\rho(g)$ applied to the seen one, so closed-loop error is invariant across the group -- float-floor-exact in 2D/$\mathrm{SO}(2)$ on real PushT and statistically flat in 3D/$\mathrm{SE}(3)$ (disjoint 95% CIs). We stress-test the prior against Sutton's Bitter Lesson: augmentation, brute-force scale, and soft-equivariance each close at most the across-group task metric, never the float-floor exactness. Because equivariance is closed under composition, the $H$-fold rollout stays flat ($\times 1.00$, $\le 2\times 10^{-7}$) at every horizon, while the baseline's residual compounds with $H$. Out of scope: task-success sweeps, planner-free invariance, and scaling. 

---
# How Quantization Changes Interpretable Features: A Sparse Autoencoder Analysis of Language Models 

**Authors**: Evan Duan  

**Link**: [PDF](https://arxiv.org/pdf/2606.03002)  

**Abstract**: Quantization is a standard path to deploying large language models, and a quantized model is typically judged acceptable when its perplexity or downstream accuracy stays close to the full-precision original. Whether the model still computes in the same way, or whether the interpretable features identified in the full-precision model survive weight rounding, is rarely tested, even as safety audits and steering interventions increasingly rely on those features. We ask whether sparse autoencoder (SAE) features extracted from a dense full-precision model remain faithful once that model is quantized. Using a frozen SAE as a fixed measurement basis, we encode full-precision and round-to-nearest (RTN) quantized activations on identical tokens and quantify per-feature survival by Pearson correlation, sweeping bit-widths from INT8 to INT4 on Pythia-70M and Gemma-2-2B. We find that feature survival is graded: features degrade systematically rather than failing all at once, with 62.4 percent of active features surviving at INT6 on Pythia-70M and 51.3 percent surviving at INT6 on Gemma-2-2B, and with most non-survivors blurred rather than destroyed. Survival is predictable from full-precision statistics alone, with cross-validated AUCs of 0.92 to 0.97 and peak activation as the strongest marginal predictor. Critically, task metrics can miss this damage: on Gemma-2-2B, INT7 improves perplexity while degrading 18.7 percent of features. Finally, quantization and matched-perplexity magnitude pruning damage strongly overlapping feature sets, with Jaccard overlap of 0.79 to 0.86 and damage-score Spearman correlation of 0.98, suggesting a shared mode of compression-induced vulnerability. These results show that behavioral parity is insufficient evidence that interpretability findings transfer to quantized deployments, motivating feature-level audits of compression. 

---
# Patcher: Post-Hoc Patching of Backdoored Large Language Models 

**Authors**: Anjun Gao, Yueyang Quan, Yufei Xia, Zhuqing Liu, Minghong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02995)  

**Abstract**: Large language models remain vulnerable to jailbreak backdoor attacks, where adversaries poison safety alignment data to embed hidden triggers that bypass safety mechanisms. Existing defenses often require comprehensive attack information or multiple triggered examples, making them impractical when defenders only observe a single reported failure case without knowing whether it stems from a backdoor attack or a natural alignment bug. This paper presents Patcher, a post-hoc defense framework that repairs backdoored language models using only a single reported failure case and the model parameters. Patcher operates in two stages. First, it localizes backdoor triggers by computing response-conditioned gradient-based saliency scores and applying adaptive clustering to separate triggers from benign context. Second, it patches the model through a constrained fine-tuning objective that breaks the trigger-response association while preserving benign-task utility and robustness to non-triggered jailbreak attacks through KL-divergence constraints. We conduct extensive evaluations across multiple backdoor attack strategies and demonstrate that Patcher successfully localizes triggers and neutralizes backdoors while maintaining model utility. We further show robustness against adaptive attacks designed to evade our defense. This work represents a significant step toward practical defenses against training-time attacks in deployed language models. 

---
# Pretraining Language Models on Historical Text 

**Authors**: Xiaoxi Luo, Zachary Shinnick, Niclas Griesshaber, Yixuan Wang, Junchi Yu, Freda Shi, Philip Torr, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.02991)  

**Abstract**: We introduce TypewriterLM, a 7.24B History language model (LM) trained exclusively on English text predating 1913. Developing History LMs requires addressing challenges in data quality and availability, preventing temporal leakage, designing temporally consistent post-training pipelines, and constructing reliable evaluations. To address these issues, we construct TypewriterCorpus, a 54B-token historical corpus collected from diverse archival and linguistically annotated sources with extensive data cleaning and leakage mitigation procedures. Furthermore, we introduce lexically grounded instructing tuning, a post-training framework that constraints responses to remain directly grounded in historical source documents. Using this framework we construct two historical instruction tuning datasets: History-LIMA and History-SelfInstruct. To evaluate capability and temporal consistency, we introduce History-Event, a benchmark suite for evaluating competence, temporal grounding and data leakage. We release TypewriterLM and all associated resources to support future research on historical language models. 

---
# Towards Compact Autonomous Driving Perception with Balanced Learning and Multi-sensor Fusion 

**Authors**: Oskar Natan, Jun Miura  

**Link**: [PDF](https://arxiv.org/pdf/2606.02979)  

**Abstract**: We present a novel compact deep multi-task learning model to handle various autonomous driving perception tasks in one forward pass. The model performs multiple views of semantic segmentation, depth estimation, light detection and ranging (LiDAR) segmentation, and bird's eye view projection simultaneously without being supported by other models. We also provide an adaptive loss weighting algorithm to tackle the imbalanced learning issue that occurred due to plenty of given tasks. Through data pre-processing and intermediate sensor fusion techniques, the model can process and combine multiple input modalities retrieved from RGB cameras, dynamic vision sensors (DVS), and LiDAR placed at several positions on the ego vehicle. Therefore, a better understanding of a dynamically changing environment can be achieved. Based on the ablation study, the model variant trained with our proposed method achieves a better performance. Furthermore, a comparative study is also conducted to clarify its performance and effectiveness against the combination of some recent models. As a result, our model maintains better performance even with much fewer parameters. Hence, the model can inference faster with less GPU memory utilization. Moreover, the result tends to be consistent in 3 different CARLA simulation datasets and 1 real-world nuScenes-lidarseg dataset. To support future research, we share codes and other files publicly at this https URL. 

---
# Glass Box at Orbit: A Constitutional AI Verification Framework for Trustworthy Autonomous CubeSat Intelligence 

**Authors**: Karthik Barma, Anil Sanneboyina, V C Premchand Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2606.02967)  

**Abstract**: The space industry is quietly building toward something nobody has fully reckoned with: orbital data centers running thousands of autonomous AI workloads with no human in the loop, 550 km above the Earth. Microsoft, AWS, and a growing list of orbital computing ventures are moving cloud-scale processing off the ground and into orbit. What none of them have answered yet is the governance question -- when autonomous AI systems at orbital data center scale make wrong decisions in space, what stops those decisions before they become irreversible?
We introduce Glass Box: a runtime constitutional AI verification layer that intercepts every candidate action from an onboard AI policy and evaluates it against six physics-grounded constitutional constraints and seven Linear Temporal Logic (LTL) safety invariants before a single command reaches any spacecraft subsystem. Every approved action carries a weighted explainability score E(a_t) in [0,1] and a complete constitutional audit log. We demonstrate Glass Box within Project October: a fully simulated five-layer autonomous orbital intelligence architecture for CubeSat-class spacecraft.
We prove that Glass Box verification overhead is O(N_c) in the number of constitutional rules, independent of model size or spacecraft state dimension. We present a complete formal specification of the constitutional constraint grammar, seven LTL safety invariants verified by Z3 and NuSMV model checking, and a detailed worked example of Glass Box intercepting an unsafe inference request at eclipse-entry under degraded battery state. As orbital computing scales toward data center infrastructure, runtime constitutional verification is no longer a research novelty -- it is mission-critical safety infrastructure that every autonomous orbital platform will eventually require. 

---
# Hand Trajectory Fusion for Egocentric Natural Language Query Grounding 

**Authors**: Enmin Zhong, Carlos R. del-Blanco, Fernando Jaureguizar, Narciso García  

**Link**: [PDF](https://arxiv.org/pdf/2606.02962)  

**Abstract**: Egocentric Natural Language Query (NLQ) grounding asks a model to localize, in a long first-person video, the temporal interval that answers a free-form text query. Existing methods fuse video appearance with the query but ignore hand motion, despite the fact that roughly 41% of Ego4D NLQ queries are answered at a moment of hand--object manipulation or their immediate this http URL propose a hand-trajectory encoder for converting a sequence of hand skeletons into highly-semantic hand kinematic features, which are then aligned and combined with pretrained video--text features through a cross-attention fusion strategy with adaptive gating. On the Ego4D NLQ v2 validation split, the clearest gains appear for Hand-Object Interaction queries (+2.54 R1@IoU=0.3) and Quantity/State queries (+4.32 R1@IoU=0.3), indicating that hand trajectory provides grounding cues beyond appearance alone. 

---
# Echelon: Auditable Aggregate-Only Language-Model Adaptation Across Privacy Boundaries 

**Authors**: Hina Dixit, Punit Kumar, Irene Tenison, Nevasini Sasikumar  

**Link**: [PDF](https://arxiv.org/pdf/2606.02958)  

**Abstract**: Cross-organization language-model adaptation increasingly faces hard governance constraints: in many deployments, device-level model state-parameters, activations, optimizer state, and per-device updates-cannot be exported outside an administrative boundary. Existing distributed and federated stacks typically assume cross-site model exchange and then retrofit privacy mechanisms, which complicates compliance and makes auditing brittle. We present Echelon, a boundary-first training architecture that enforces device-level model-state non-export as a systems invariant. Devices train locally inside each boundary; the only cross-boundary payloads are securely aggregated boundary-level deltas plus O(1) coordination metadata, exposed through a concrete audit surface. Restricting exchange to aggregates changes the optimization problem: the system must remain stable under WAN delay, heterogeneous participation, churn, and non-IID data even though the global plane never sees per-device updates. Echelon combines buffered semi-asynchronous secure aggregation, staleness-aware weighting, participation windows, proximal local objectives, and a drift-aware outer synchronization controller. In 1B-parameter LoRA adaptation across M= 2 boundaries, a budget-matched contest over three seeds (24.88M tokens) reaches validation loss 3.887 +/-0.010 and is best or tied-best among tuned low-communication baselines under fixed-token, fixed-bytes, fixed-wall-clock, and fixed-sync-count budgets. In OpenWebText stress tests, Echelon sustains 2,139-2,176 tokens/s across evaluated WAN and non-IID treatments, Echelon-DA improves time-to-target under WAN latency relative to a privacy-parityDiLoCo+SA baseline, and quality degrades by at most 2.2% under 200ms emulated latency or severe non-IID partitioning. 

---
# Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference 

**Authors**: Siva Rajesh Kasa, Yasong Dai, Sumit Negi, Hongdong Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.02955)  

**Abstract**: Diffusion large language models promise parallel token generation, yet inference remains bottlenecked by deciding which masked tokens can be safely committed together. Fast-dLLM addressed this with KV caching and confidence-guided parallel decoding, but its decoding theory uses a homogeneous high-confidence assumption that effectively reduces each candidate set to its weakest selected token. We argue that this leaves speed on the table because real decoding steps exhibit heterogeneous confidence profiles. We propose \textbf{Fast-dLLM++}, a training-free extension that introduces \emph{Fréchet profile decoding}: selecting parallel commit sets from the full sorted confidence profile rather than a single worst-case confidence. The resulting rule is a heterogeneous-confidence generalization of Fast-dLLM's factor selector and it recovers the previous rule exactly in the equal-confidence case and adds a provable \emph{heterogeneity bonus} when the selected tokens have uneven confidences. Fast-dLLM++ leaves the model, diffusion process, and cache implementation entirely unchanged, making it a drop-in replacement for existing Fast-dLLM decoding. Experiments on GSM8K, MATH, HumanEval, and MBPP with the LLaDA-8B model show that the theoretical improvement translates directly into empirical gains: profile-aware selection improves the accuracy--throughput frontier by exploiting safe parallelism that weakest-token rules miss, achieving up to 37\% higher throughput at comparable accuracy. Our anonymous code release is at this https URL. 

---
# SCOPE: Real-Time Natural Language Camera Agent at the Edge 

**Authors**: Nikolaj Hindsbo, Sina Ehsani, Pragyana Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2606.02951)  

**Abstract**: Deploying language-driven agents in robotics requires evaluations that reflect real-world task demands: natural-language instructions with reproducible outcomes. Such agents must connect language models to callable perception and control tools, and be assessed using deployment-critical metrics including latency, accuracy, and error modes. We present SCOPE (Simulation and Camera Operations for Perception and Evaluation), a modular agent for natural-language, open-vocabulary pan-tilt-zoom (PTZ) camera control and visual scene understanding, designed explicitly for edge deployment. SCOPE operates both in a Blender-based simulation environment and on a physical PTZ camera, executing all perception, planning, and control locally at the deployment site using edge-accessible compute.
We release a 536-task benchmark spanning QA, single- and multi-step commands, counting, spatial reasoning, descriptions, and optical character recognition in a Blender-based simulation environment that exposes realistic PTZ control affordances. Execution traces are combined with an LM-as-Judge to evaluate latency, accuracy, and error modes.
We evaluate 19 planner-perception model combinations pairing Qwen3 small language models (SLMs) with Moondream and Qwen vision-language models (VLMs). Stronger SLMs substantially reduce hallucinations and improve tool routing, leading to more reliable closed-loop behavior. Once a sufficiently capable SLM is used, perception becomes the dominant performance bottleneck. Mixture-of-Experts models on both the planning and perception side consistently match or exceed dense alternatives at latencies and memory footprints comparable to much smaller networks. Quantization provides additional efficiency gains with minimal accuracy degradation, identifying a practical, sim-to-real validated design point for real-time, edge-feasible language-driven PTZ control. 

---
# WRIT: Write-Read Intensive Trajectory Synthesis for Multi-Turn User-Facing Agents 

**Authors**: Hengrui Gu, Xiaotian Han, Kaixiong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.02908)  

**Abstract**: Multi-turn user-facing agents must infer user intent from incomplete requests, collect missing information through dialogue and tools, and execute valid actions. A training trajectory records this process as an interleaved sequence of user messages, agent responses, tool calls, etc. Synthesizing sufficiently complex trajectory has become a central route to train agents: existing pipelines often increase difficulty by composing multiple user requests into longer tasks, producing write-intensive trajectories that train sequential execution.
We argue that a single write decision can itself be difficult when the agent must gather and compare substantial read-tool evidence before its arguments become identifiable, a challenge that write-intensive data alone cannot address. Guided by this insight, we propose WRIT (\uline{W}rite-\uline{R}ead \uline{I}ntensive \uline{T}rajectory Synthesis), a pipeline for synthesizing multi-turn agent training trajectories along two complexity axes: the number of write decisions in a task and the evidence burden of each individual decision. WRIT first generates write-intensive and read-heavy tasks. It then diversifies user behavior instructions to reflect realistic conversational variation, and finally simulates agent-user interactions in an executable environment to produce complete training trajectories. The resulting data trains agents not only for longer task execution, but also for robust, evidence-grounded decision making under high information load. With only 2K synthesized trajectories, a 4B model trained on WRIT outperforms GPT-5.1 no-think on $\tau^2$-bench and substantially reduces inference-time token usage, showing that compact SFT data can convert part of expensive test-time reasoning into efficient agent behavior. 

---
# Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States 

**Authors**: Subramanyam Sahoo, Vinija Jain, Aman Chadha, Divya Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2606.02907)  

**Abstract**: Linear probing of large language model (LLM) hidden states is widely used to claim that models learn distinct representations for different reasoning types. We test this by probing Qwen3-14B on three benchmarks spanning the classical trichotomy: LogiQA 2.0 (deductive), ARC-Challenge (inductive), and $\alpha$NLI (abductive). At layer 32 of 40, linear probes achieve 100\% cross-validated accuracy with well-separated geometry (intrinsic dimensionalities: 20.6, 28.5, 33.6; convex hull contamination $\leq$1.5\%). However, this separation is entirely driven by format confounds. Residualizing source identity, option count, and response length reduces accuracy to chance. Trace-anchor similarity indicates largely shared reasoning across tasks (42.5\% agreement vs.\ 33.3\% chance), and causal steering with random controls ($n=20$) shows no functional link between geometry and reasoning mode ($p=0.286$). Thus, high probe accuracy reflects task format rather than computational structure, motivating routine format deconfounding in mechanistic interpretability. 

---
# Scalable Uncertainty Quantification for Extreme Weather Forecasting via Empirical Neural Tangent Kernels 

**Authors**: Jose Marie Antonio Miñoza, Rex Gregor Laylo, Sebastian C. Ibañez  

**Link**: [PDF](https://arxiv.org/pdf/2606.02886)  

**Abstract**: Deep learning weather models now match numerical weather prediction accuracy while running orders of magnitude faster, but produce deterministic forecasts without uncertainty estimates, a critical gap for high-stakes decisions during extreme weather events. This paper proposes Neural Tangent Kernel-based uncertainty quantification (NTK-UQ) using last-layer empirical features. Theoretical analysis predicts that UQ quality is architecture-dependent through two mechanisms. First, a variance collapse mechanism explains when UQ fails: when the eigenvalue truncation rank approaches the effective rank of the feature space, the GP correction term consumes nearly all prior variance, destroying discrimination between tropical cyclones and routine conditions; architectures with concentrated spectra (spectral operators) require aggressive truncation ($k \leq 10$), while attention-based models tolerate full-rank computation. Second, decomposition performance depends on the non-Gaussian, heavy-tailed structure of extreme weather: Independent Component Analysis exploits higher-order statistics (kurtosis, negentropy) to isolate heavy-tailed extreme-event features, achieving higher discrimination than singular value decomposition, which captures only second-order variance. A data-driven selection rule chooses ICA or SVD from the feature eigenspectrum concentration ratio, correctly prescribing the superior decomposition for all four evaluated architectures. Compared to split conformal prediction (the natural post-hoc baseline), NTK-UQ achieves 31--37\% sharper prediction intervals at 90\% coverage, and uniquely produces \emph{adaptive} intervals that scale with extreme event severity, which conformal prediction cannot achieve by construction. The framework requires no retraining; inference-time uncertainty requires only a single matrix-vector product per sample. 

---
# Are we really tilting? The mechanics of reward guidance in flow and diffusion models 

**Authors**: Sanjit Dandapanthula, Nicholas M. Boffi  

**Link**: [PDF](https://arxiv.org/pdf/2606.02884)  

**Abstract**: Reward guidance algorithms steer a learned generative process toward the reward-tilted measure at inference time. While empirically powerful, these methods are prone to reward hacking: the guided model over-optimizes the reward at the cost of fidelity to the learned distribution. Prior work has attributed this to the complexity of neural reward functions or implicit biases in diffusion training, but its fundamental origins remain poorly understood. We show that reward hacking arises from an approximation made in most practical implementations of reward-guided diffusion -- finite-particle plug-in estimation of the Doob h-function -- even in the simplest non-trivial settings of Gaussian and Gaussian mixture targets with quadratic rewards. In closed form, we isolate two distinct failure modes of the plug-in estimator: it leads to reward hacking within each mode and it cannot select high-reward modes. We propose a closed-form reward damping schedule that corrects the within-mode bias with no additional compute, and clarify the role of best-of-n sampling in compensating for the mode selection failure. Experiments on Gaussian mixture targets, a 2D checkerboard, and FLUX.1 text-to-image generation confirm that our theoretical insights carry over to practical settings. 

---
# LLM-Assisted Reranking to Operationalize Nuanced Objectives in Recommender Systems 

**Authors**: Amir Ghasemian, Homa Hosseinmardi, Upasana Dutta, Duncan J. Watts  

**Link**: [PDF](https://arxiv.org/pdf/2606.02883)  

**Abstract**: Recommender systems have grown from content-organization tools into sophisticated systems that shape daily behavior. By controlling what we see, they shape what we perceive, raising concerns about filter bubbles, radicalization, polarization, and social inequality. Large language models (LLMs) enable more powerful personalization, intensifying these dynamics. Yet most recommenders are tuned for engagement or limited accuracy metrics, with little attention to broader social implications, e.g. how personalization reshapes exposure in socially consequential domains. We investigate whether LLM-assisted reranking, while improving personalization, inadvertently amplifies exposure to ideologically extreme or conspiratorial political content, a risk theorized but not empirically characterized in news recommendation. Using real news-consumption histories, we rerank YouTube's sidebar candidates through zero-shot, instruction-based prompting. We compare a baseline prompt with a constrained variant that preserves topical relevance and broadens ideological exposure while reducing conspiratorial or extreme content. Without constraints, reranking strengthened personalization but increased exposure to conspiratorial and extremist material for users whose histories contained such content. Lightweight prompt-level regularization reduced promotion of extreme content and increased ideological diversity, with modest relevance loss. Synthetic experiments suggest that LLMs rerank via statistical regularities in language rather than semantic understanding of ideology, clarifying why naive prompts amplify these patterns and why regularization can reshape them. Together, our results highlight the power of LLMs to operationalize contextual nuance in high-stakes recommendation, and the need to evaluate LLM-assisted personalization beyond accuracy and treat prompt design as a value-laden rather than neutral default. 

---
# Adaptive Latent Agentic Reasoning 

**Authors**: Dongwon Jung, Peng Shi, Yi Zhang, Junshan Zhang, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.02871)  

**Abstract**: Large reasoning models improve performance by generating extended chain-of-thought (CoT) reasoning, but this behavior becomes inefficient when applied to LLM agents. Current LLM agents often generate verbose textual reasoning at every decision step and allocate reasoning effort nearly uniformly across turns, leading to substantial inefficiency in multi-turn agentic trajectories. We propose Adaptive Latent Agentic Reasoning (ALAR), a dual-mode framework that uses compact latent reasoning for routine turns and selectively escalates to explicit chain-of-thought when deeper deliberation is needed. ALAR learns latent reasoning by using the agent's actions as supervision anchors and is further optimized to use latent reasoning when it is sufficient for task success and reserve explicit CoT for harder decisions. Experiments on agentic search and tool-use benchmarks show that ALAR maintains comparable or better task accuracy while substantially reducing generated tokens by up to 43.6% in search and 84.6% in tool use. These results demonstrate that ALAR improves the accuracy-efficiency trade-off of LLM agents by reducing unnecessary textual reasoning while preserving explicit deliberation for harder decision steps. 

---
# The Epi-LLM Framework: probing LLM behavioral priors through epidemiological agent-based models 

**Authors**: Petra Ferenz, Ava Keeling, Tobias O'Keefe, Lorenzo Stigliano, Francesco Di Lauro, Andres Colubri, Jasmina Panovska-Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2606.02867)  

**Abstract**: Human behaviour during epidemics affects infectious disease dynamics, but quantifying this remains deeply challenging. Here we introduce the Epi-LLM framework: a novel integration of agent-based modelling, real-life epigames, and large language models (LLMs) in which a synthetic society of agents reasons and adapts dynamically over an outbreak contact network. Comparing synthetic agent behaviour against a no-intervention SEIR baseline and human participant data from the AUIB epigame study, we find that LLM agents across four different architectures reduced peak active infections, with quarantine compliance peaking at 58-65% on day six of the 15-day simulation. A binomial generalised linear model showed that perceived health severity was the strongest predictor of quarantine behaviour ($\beta = 0.33, p = 0.002$), yielding a pseudo-$R^2$ of 0.055, comparable to the 0.072 observed in the human trial. LLM architecture is a key determinant of epidemic dynamics: low-variance architectures offer greater internal validity for testing behavioural rules, while high-variance models may better represent real-world decision-making. Geographic labels alone do not induce culturally differentiated behaviour; explicit attitudinal parameterisation is required. This proof-of-principle work lays the groundwork for deploying the Epi-LLM framework as a scalable, risk-free simulation environment for pandemic preparedness research. 

---
# Forgetting is Not Erasure: Recovering Latent Knowledge via Transport Keys 

**Authors**: Archie Chaudhury  

**Link**: [PDF](https://arxiv.org/pdf/2606.02860)  

**Abstract**: Catastrophic forgetting is often framed as a representational problem: after sequential training, a model appears to lose the features that supported performance on earlier tasks. We challenge the stronger form of this view. Across controlled continual-learning settings, we find that a significant portion of apparent forgetting can be attributed to interface drift between internal stages rather than permanent erasure of task-relevant computation. We study this phenomenon through a stitched evaluation protocol that combines early computation from a post-update network with late computation from its predecessor, optionally mediated by a compact, task-specific transport key. We describe transport keys at a systems level as compact interface-alignment operators estimated from a small set of paired anchor activations and evaluated through model stitching. On split CIFAR-100 with a ResNet-style network, transport keys recover most of the original Task A performance after sequential training on Task B. On a compact vision transformer, we observe a similar recovery pattern. These results suggest that continual learning may require better mechanisms for indexing and re-accessing latent computations, not only methods that prevent weight change. 

---
# Economy of Minds: Emerging Multi-Agent Intelligence with Economic Interactions 

**Authors**: Zhenting Qi, Huangyuan Su, Ao Qu, Chenyu Wang, Yu Yao, Han Zheng, Kushal Chattopadhyay, Guowei Xu, Zihan Wang, Weirui Ye, Vijay Janapa Reddi, Ju Li, Paul Pu Liang, Himabindu Lakkaraju, Sham Kakade, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2606.02859)  

**Abstract**: How can a population of agents self-orchestrate and self-adapt into stronger collective intelligence without centralized control? Inspired by Friedrich Hayek's economic theory of decentralized coordination in markets, we study this question through an agent economy in which agents compete via auctions for the right to act, exchange payments, and accumulate wealth from environmental rewards. These simple economic signals induce decentralized credit assignment, driving planning without global orchestration or explicit communication protocols. The population evolves through economic selection: effective agents accumulate wealth and are mutated via exploitation, while ineffective ones go bankrupt and are replaced via exploration. We show that, initialized with weak agents, the economy produces emergent multi-step reasoning strategies and outperforms stronger monolithic baselines across five agentic tasks, including mathematical reasoning, financial research, scientific research, accelerator design, and distributed-system optimization. We further provide theoretical insights into how economic dynamics shape agent behaviors, linking local incentives to long-term global performance. Our results suggest a new path to multi-agent intelligence: rather than engineering coordination, we can design decentralized incentive structures under which it automatically emerges. 

---
# GRZO: Group-Relative Zeroth-Order Optimization for Large Language Model Fine-Tuning 

**Authors**: Liyan Tan, Yequan Zhao, Yifan Yang, Ruijie Zhang, Xinling Yu, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02857)  

**Abstract**: Zeroth-order (ZO) optimization is a memory-efficient alternative to backpropagation for fine-tuning large language models, but its deployment is limited by the high variance of gradient estimation. We propose GRZO, a Group-Relative Zeroth-Order optimizer that draws one pseudo-independent perturbation per mini-batch example and aggregates the per-example losses through group-relative normalization, raising the effective gradient-direction count from one to the batch size at no additional forward cost while preserving inference-level memory. We prove that GRZO is directionally unbiased with variance shrinking proportionally to the batch size, yielding a tighter nonconvex convergence bound than MeZO. Across RoBERTa-large, Llama3-8B, and OPT-13B over multiple tasks, GRZO improves average accuracy on Llama3-8B by $+3.0$ over MeZO at $23\%$ lower peak GPU memory; as a drop-in replacement for the MeZO core, it lifts sparse, low-rank, and quantized ZO variants by $+6.0$ on average. 

---
# Fixing FOLIO and MALLS: Verified Annotations and an LLM-assisted Framework to Focus Human Relabeling 

**Authors**: Andrea Brunello, Cristian Curaba, Luca Geatti, Michele Mignani, Angelo Montanari, Nicola Saccomanno  

**Link**: [PDF](https://arxiv.org/pdf/2606.02837)  

**Abstract**: Accurate translation from Natural Language to First-Order Logic (NL-to-FOL) underpins neurosymbolic AI systems and Natural Language Inference (NLI), making the quality of NL-to-FOL benchmarks essential -- yet these datasets have never been rigorously audited. Our first contribution is to present a systematic human inspection of the validation split of \textsf{FOLIO} and a subset of \textsf{MALLS} test instances, finding that approximately 39% and 36% of entries, respectively, contain incorrect FOL formalizations (i.e., ground truth labels), with additional rates of ambiguous NL sentences (16.4% and 48%) and incorrect NLI labels in \textsf{FOLIO} (8.4%). Our second contribution is to develop and release corrected ground truths for such datasets, showing that annotation errors distort model evaluation on a reference benchmark task: testing three state-of-the-art LLMs (Gemma~4 31B-it, Qwen3-30B-A3B, and GPT-4o-mini) with the corrected ground truths yields accuracy gains from +9 to +22 percentage points. Motivated by these findings, we propose an LLM-based framework to support humans in manual reviewing NL-to-FOL datasets. By directing reviewers toward the most error-prone instances, we empirically show that it is possible to achieve 90% dataset accuracy after reviewing fewer than 24% of instances, compared to over 70% required by unguided review. We release all human-verified annotations and the code for our framework. 

---
# Large Byte Model: Teaching Language Models About Compiled Code 

**Authors**: Florian Störtz, Catalin-Andrei Stan, Alexandru Dinu, Sandra Servia-Rodríguez, Mihaela Gaman, Calin Miron, Edward Raff  

**Link**: [PDF](https://arxiv.org/pdf/2606.02834)  

**Abstract**: Malware analysis starts with the raw bytes of an executable program, and tools to "lift" these to higher-level representations, such as assembly, are expensive and subject to error. Large Language Models (LLMs) cannot process raw byte representations and answer questions about them. To this end, we present the first byte-native LLM. Based on a vocabulary expansion technique using a bespoke byte tokenizer, such a model is capable of responding to complex questions about malware binaries, with accuracies ranging from 69% for malware family classification to 98% for architecture classification. Our findings indicate that providing domain knowledge during training is essential for this application -- off-the-shelf models lack both accuracy and insight. We've deployed this emerging solution to a limited number of analysts to gather feedback for further improvements. 

---
# Which Defense Closes Which Threat? Attributing OWASP-LLM-Top-10 Coverage and Its Brittleness Under Paraphrasing 

**Authors**: Alexandre Cristovão Maiorano  

**Link**: [PDF](https://arxiv.org/pdf/2606.02822)  

**Abstract**: Production LLM applications stack several defense families -- refusal-phrase filters, token-budget controls, model allowlists, rate limits, tool-registry authentication -- yet existing breach-and-attack-simulation (BAS) benchmarks report a single aggregate coverage number, hiding which family closes which threat. We measure attribution. We add four OWASP-LLM-Top-10-aware agents to a 21-agent baseline scanner and target a lattice of four synthetic LLM endpoints: $L_0$ (no defenses), $L_1$ (refusal-only), $L_2$ (budget-only), and $L_3$ (full stack). $L_1$ and $L_2$ are sibling single-axis ablations, not subsets of each other; $L_3$ is their union plus tool-registry authentication and credential scrubbing. Across $N=10$ replications, the per-OWASP finding count is clean: refusal alone removes all LLM01 (jailbreak) and LLM07 (system-prompt leakage) findings; budget alone removes all LLM02 (sensitive-info disclosure) and LLM10 (unbounded consumption) findings by terminating multi-step sequences; LLM06 (excessive agency) requires the full stack. We probe brittleness under paraphrasing: with 300 Gemini-generated paraphrases ($K=5$ over a 60-template brittleness corpus), $L_1$ refusal block rate falls 15 pp on LLM01 and 25 pp on LLM07. A fifth target, $L_4$-real, swaps the stub backend for Gemini-2.5-flash behind the same $L_3$ regex and matches $L_1$ exactly, indicating no measurable alignment contribution beyond the regex (not a general claim about alignment). Budget controls show no drop (0 pp once the rate-limit floor is factored out). A refusal whitelist that clears a static benchmark can be defeated by an LLM-driven paraphraser without changing attack intent; a budget control resists the same mutation. 

---
# Do Neural Retrievers Prefer Certain Documents? Evidence of Learned Relevance Priors 

**Authors**: Francisco Valentini, Edgar Altszyler, Martin Fajcik  

**Link**: [PDF](https://arxiv.org/pdf/2606.02814)  

**Abstract**: Neural retrievers are trained to estimate query-document relevance from annotated query-document pairs. Yet annotation protocols may not purely reflect relevance: they select only a subset of documents for labeling, and this selection can favor certain document types over others. We investigate whether supervised bi-encoder retrievers implicitly learn a document-level relevance prior: a query-independent signal encoded in their representation space as a side effect of training on annotated data. We estimate this prior by training simple classifiers on frozen document embeddings and evaluate three state-of-the-art retrievers across multiple IR benchmarks. We find that supervised neural retrievers encode relevance priors that generalize to unseen documents and are consistent across models. These priors create a findability gap: documents with lower prior are systematically harder to retrieve, even when genuinely relevant. This effect appears in supervised dense retrievers but is weaker and less consistent in BM25, and it persists under controlled matched-document comparisons. Using LLM-based explanations, we find that judged-relevant documents tend to be comprehensive, self-contained summaries of mainstream topics, while niche, fragmentary, or highly technical content is often left unjudged. Retrievers internalize this bias, ranking documents with these favored features higher than documents that lack them, independently of their actual relevance. Our findings expose a structural limitation of supervised retrieval: models trained on annotated data do not just learn relevance, but also the implicit document preferences in their training data. 

---
# Cosmos 3: Omnimodal World Models for Physical AI 

**Authors**: Aditi, Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, Aarti Basant, Mukesh Beladiya, Mohammad Qazim Bhat, Zaid Pervaiz Bhat, Dan Blick, Vanni Brighella, Han Cai, Tiffany Cai, Eric Cameracci, Jiaxin Cao, Yulong Cao, Mark Carlson, Carlos Casanova, Ting-Yun Chang, Yan Chang, Yu-Wei Chao, Prithvijit Chattopadhyay, Roshan Chaudhari, Chieh-Yun Chen, Junyu Chen, Ke Chen, Qizhi Chen, Wenkai Chen, Xiaotong Chen, Yu Chen, An-Chieh Cheng, Click Cheng, Xiu Chia, Jeana Choi, Chaeyeon Chung, Wenyan Cong, Yin Cui, Magdalena Dadela, Nalin Dadhich, Wenliang Dai, Joyjit Daw, Alperen Degirmenci, Rodrigo Vieira Del Monte, Robert Denomme, Sameer Dharur, Marco Di Lucca, Ke Ding, Wenhao Ding, Yifan Ding, Yuzhu Dong, Nicole Drumheller, Yilun Du, Aigul Dzhumamuratova, Aleksandr Efitorov, Hamid Eghbalzadeh, Naomi Eigbe, Imad El Hanafi, Hassan Eslami, Benedikt Falk, Jiaojiao Fan, Jim Fan, Amol Fasale, Sergiy Fefilatyev, Liang Feng, Francesco Ferroni, Sanja Fidler, Xiao Fu, Vikram Fugro, Prashant Gaikwad, TJ Galda, Katelyn Gao, Yihuai Gao, Wenhang Ge, Sreyan Ghosh, Arushi Goel, Vivek Goel, Akash Gokul, Rama Govindaraju, Jinwei Gu, Miguel Guerrero, Elfie Guo, Aryaman Gupta, Siddharth Gururani, Hugo Hadfield, Song Han, Ankur Handa, Zekun Hao, Mohammad Harrim, Ali Hassani, Nathan Hayes-Roth, Yufan He, Chris Helvig, Cyrus Hogg, Madison Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02800)  

**Abstract**: We introduce Cosmos 3, a family of omnimodal world models designed to jointly process and generate language, image, video, audio, and action sequences within a unified mixture-of-transformers architecture. By supporting highly flexible input-output configurations, Cosmos 3 seamlessly unifies critical modalities for Physical AI -- effectively subsuming vision-language models, video generators, world simulators, and world-action models into a single framework. Our evaluation demonstrates that Cosmos 3 establishes a new state-of-the-art across a diverse suite of understanding and generation tasks, demonstrating omnimodal world models as scalable, general-purpose backbones for embodied agents. Our post-trained Cosmos 3 models were ranked as the best open-source Text-to-Image and Image-to-Video models by Artificial Analysis, and the best policy model by RoboArena at the time the technical report was written. To accelerate open research and deployment in Physical AI, we make our code, model checkpoints, curated synthetic datasets, and evaluation benchmark available under the Linux Foundation's OpenMDW-1.1 this https URL License at this https URL}{this http URL and this https URL . The project website is available at this https URL . 

---
# CRAM-ER: Error-Resilient Spintronic Computational Random Access Memory for Scalable In-Memory Computation 

**Authors**: Sohan Salahuddin Mugdho, Md. Shahedul Hasan, Brahmdutta Dixit, Yang Lv, Jian-Ping Wang, Cheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02781)  

**Abstract**: Deep neural networks (DNNs) have achieved state-of-the-art performance across diverse domains. However, typical Von Neumann compute paradigms face severe memory bottlenecks. Emerging near-memory and compute-in-memory approaches alleviate this but incur significant peripheral overhead. Computational Random Access Memory (CRAM) based on MRAM enables in-situ logic without peripheral overhead, offering a dense, energy-efficient solution. However, probabilistic MRAM switching induces gate-level errors that limit the scalability and reliability of CRAM for accelerating DNN. Moreover, the large number of sequential MRAM writes severely constrains CRAM throughput. To address these challenges, we propose an error-resilient CRAM (CRAM-ER) architecture for scalable in-memory matrix-vector multiplications (MVMs). Our error-aware hardware-software co-design framework leverages a hybrid spintronic-CRAM + CMOS adder-tree architecture to mitigate the impact of device-level errors, demonstrating MVM functionality with high area and energy efficiency. We further develop an error-aware model fine-tuning and fine-grained error correction for enhanced error resilience. Evaluations of the CMOS+spintronic hybrid architecture on DNN benchmarks show near-lossless accuracy while reducing CRAM latency by up to 2 orders of magnitude, outperforming CPU/GPU+high-bandwidth DRAM in both energy efficiency and energy-delay product. 

---
# Representational Capacity: Geometric Limits on Feature Representation in Transformer Language Models 

**Authors**: Alexander Guha  

**Link**: [PDF](https://arxiv.org/pdf/2606.02765)  

**Abstract**: Model dimension ($d_{model}$) is a fundamental hyperparameter in transformer language models, yet its role in setting the geometric limits of feature representation remains under-explored. Grounded in the Linear Representation and Superposition Hypotheses - which propose that models encode features as near-orthogonal directions in latent space - we develop a framework for estimating how many such directions a model can support. We first establish the embedding matrix as a measurable proxy for near-orthogonality constraints across the latent space: the boundary between meaningful token relationships and incidental similarity in the pairwise cosine similarity distribution gives a concrete estimate of the model's accepted deviation $\varepsilon$ from perfect orthogonality. Applying this metric across dozens of open-source models reveals two classes: models with high $\varepsilon$ whose embeddings lack near-orthogonal structure, and models with low $\varepsilon$ that maintain it. We then show that the standard Johnson-Lindenstrauss lemma greatly underestimates the packing efficiency of trained representations, and derive an adjusted capacity formula in which the number of near-orthogonal directions depends on the ratio of vectors to dimensions ($k/d$) rather than the raw count - a single modification that cuts prediction error by two orders of magnitude with no extra parameters. Combining these results, we define representational capacity as an upper bound on the number of distinguishable directions available for features and embeddings in a model's latent space. Capacity is exponentially sensitive to $\varepsilon$, and larger models favor tighter orthogonality constraints over maximizing raw capacity - a pattern compatible with several explanations (a stability-capacity trade-off, a ceiling on usable concepts, or confounds with model scale) that we leave to future work. 

---
# Acceptance-Test-Driven Evaluation Protocols for Business-Centric LLM Systems 

**Authors**: Eric Liang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02755)  

**Abstract**: Large language model (LLM) applications are increasingly expected to satisfy deterministic institutional requirements while relying on probabilistic generative components. This mismatch makes ordinary post-hoc benchmarking insufficient for systems that must be safe, reliable, auditable, and economically useful. This paper contributes an evaluation-protocol extension for operational LLM systems grounded in acceptance-test-driven development, safety engineering, and business-centric validation. The extension translates stakeholder goals into executable behavioral contracts, release gates, monitoring signals, and evidence artifacts before prompt, model, retrieval, or agent changes are accepted. It adapts the red-green-refactor discipline of test-driven development to a red-train-green lifecycle: first define failing acceptance tests for desired behavior, then improve the LLM system through prompt changes, retrieval design, fine-tuning, guardrails, or data augmentation, and finally release only when multidimensional gates are satisfied. The contribution is a governance-oriented metric stack, reference architecture, and empirical protocol for comparing acceptance-test-driven LLM development against prompt-first and benchmark-after workflows. 

---
# MetaWorld: Scaling Multi-Agent Video World Model from Single-view Video Data 

**Authors**: Teng Hu, Mingchun Lu, Yating Wang, Jiangning Zhang, Jinkun Hao, Ye Pan, Ran Yi, Lizhuang Ma, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2606.02753)  

**Abstract**: Video world models are a foundational generative technology for embodied AI and the Metaverse, yet existing approaches are inherently limited to a single agent observing from a single perspective. Extending these models to multi-agent settings introduces two critical challenges: data scarcity (coordinated multi-view recordings are prohibitively expensive to collect for general open-domain scenarios) and world state alignment (independently generated video streams cannot ensure that shared physical environments and events evolve consistently across views). To address these challenges, we propose MetaWorld, a novel framework that scales multi-agent video world models to open-domain environments directly from single-view videos. First, we introduce Monocular World-State Unrolling (MWSU) to explicitly decompose monocular footage into the camera operator's ego-motion and the visible subject's spatial trajectory. This camera-trajectory decomposition naturally extracts synchronized multi-agent motion data within a shared 3D space, completely bypassing the need for multi-camera setups. Second, for precise visual control, we develop the Subject-Aware World Generator to enable appearance-driven simulation conditioned on per-agent identity images. Finally, to ensure both views are grounded in the identical physical reality, we propose World-State Alignment, a per-frame inter-branch cross-attention mechanism inserted at every transformer layer of the video DiT. By jointly synchronizing the denoising process, WSA enforces both static geometric consistency and dynamic motion consistency, encouraging that the shared 3D environment and physical events remain well-aligned across both egocentric views. Extensive experiments demonstrate that MetaWorld achieves superior cross-view consistency and identity fidelity, establishing a highly scalable, physics-driven paradigm for multi-agent video world modeling. 

---
# Plan2Map: A Multimodal Benchmark for Document-Grounded Geospatial Boundary Reconstruction from Planning Records 

**Authors**: Fabian Degen, Oishi Deb, Jindong Gu, Junchi Yu, Samuele Marro, Philip Torr, Jialin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.02747)  

**Abstract**: Planning records define restrictions over geographic areas, but their source documents often provide only indirect spatial evidence rather than machine-readable boundaries. We introduce Plan2Map, a 208-case multimodal benchmark for document-grounded geospatial boundary reconstruction from UK planning records. Given only a source planning document, systems must reconstruct a valid geospatial boundary from notice text, schedules, map plates, map labels, and boundary annotations; the reference GeoJSON is held out for scoring. We propose GeoPlanAgent, a document-grounded, geospatial-tool-in-the-loop system that decomposes the task into evidence extraction, localisation, map registration, boundary segmentation, projection, and verification. On Plan2Map, GeoPlanAgent achieves 0.736 mean IoU and 0.904 median IoU, with 67.8\% of predictions at or above 0.8 IoU, substantially outperforming direct VLM-to-GeoJSON baselines. Diagnostic analysis shows that direct VLM prediction remains unreliable, while remaining errors are concentrated in localisation and map registration, and supervised boundary segmentation substantially improves pixel-level mask quality. Plan2Map provides a concrete testbed for multimodal geospatial reconstruction from public planning records. Project page: this https URL. 

---
# EntangleCodec: A Unified Discrete Audio Tokenizer via Semantic-Acoustic Entanglement 

**Authors**: Hui Li, Yangfan Gao, Junlin Shang, Changhao Jiang, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02739)  

**Abstract**: Audio tokenizers serve as the discrete interface between continuous audio and Audio Language Models (ALMs), but existing tokenizers often struggle to support both understanding and generation. Reconstruction-oriented codecs preserve acoustic fidelity but lack rich semantics, while semantic-aware tokenizers typically rely on separate semantic and acoustic streams, introducing redundancy or misalignment.
We propose \textbf{EntangleCodec}, a unified discrete audio tokenizer that learns caption-aligned semantic-acoustic representations before quantization. By aligning audio with rich captions rather than ASR transcripts, EntangleCodec captures linguistic content, speaker identity, emotion, prosody, and acoustic scenes within a compact token stream. A flow-matching diffusion decoder further enables high-quality reconstruction across speech, music, and general audio.
EntangleCodec achieves reconstruction quality competitive with specialized codecs, outperforms all codec-based baselines on audio understanding by up to \textbf{+7.4\%} on MMAR, and supports both TTS and TTA generation in a unified framework. Furthermore, EntangleCodec-based audio language models demonstrate strong scaling behavior: even at \textit{0.6B} parameters, the model surpasses specialized continuous-representation LLMs with over \textit{13B} parameters across three benchmarks using \textbf{22$\times$} fewer parameters; scaling to \textit{8B} further establishes new state-of-the-art results on MMAR, highlighting that representation quality is as critical as model scale in audio language modeling. Code and model weights are available at this https URL. 

---
# Attention Calibration for Position-Fair Dense Information Retrieval 

**Authors**: Andrianos Michail, Elias Schuhmacher, Juri Opitz, Simon Clematide, Rico Sennrich  

**Link**: [PDF](https://arxiv.org/pdf/2606.02737)  

**Abstract**: Dense retrieval models exhibit positional bias: retrieval effectiveness degrades when relevant information appears later in a passage (Zeng et al., 2025). We ask whether this bias can be reduced at inference time, without retraining and without sacrificing overall retrieval effectiveness. To this end, we adapt inference-time attention calibration (Schuhmacher et al., 2026) to downstream retrieval and extend it with a strength coefficient lambda that interpolates between the original and fully calibrated attention distributions. Across three embedding models on SQuAD-PosQ and FineWeb-PosQ, we examine how basket size, calibrated layer set, and strength affect the trade-off between positional fairness and retrieval effectiveness, finding that partial calibration frequently outperforms full calibration. A single configuration (B=128, lambda=0.5, 50% layer depth) improves the harmonic mean of nDCG@10 across positional groups on FineWeb-PosQ for all three models without per-model tuning, and applies to both <s>-pooled and last-token-pooled architectures. This default configuration transfers without modification to PosIR, which spans 10 languages and 31 domains, reducing the Position Sensitivity Index in all 16 length-quartile x model x retrieval-setting combinations, while preserving or improving aggregate nDCG@10. We release our extended codebase at this https URL 

---
# See Less, Specify More: Visual Evidence Budgets for Generalizable VLAs 

**Authors**: Yueh-Hua Wu, Tatsuya Matsushima, Kei Ota  

**Link**: [PDF](https://arxiv.org/pdf/2606.02735)  

**Abstract**: Generalization remains a central bottleneck for vision-language-action (VLA) models: under distractors, appearance shifts, and semantically similar tasks, the policy must often infer local execution details from coarse instructions while also deciding which parts of the image matter for control. We present S2 (See Less, Specify More), a framework for improving VLA generalization by training the executor under a cleaner interface.
Specify More preserves the original instruction as a stable high-level goal while relabeling each trajectory into refined trajectory- and subtask-level language that disambiguates the current execution mode. Unlike native attention, See Less imposes an explicit visual evidence budget, training the executor to act from task-sufficient evidence rather than unconstrained visual context, without any region or mask annotation.
This interface lets the executor follow detailed guidance without relying on distracting visual patches or resolving avoidable ambiguity on its own, and it remains compatible with off-the-shelf VLM planners through in-context learning. Across our main evaluation settings, S2 improves overall generalization metrics by changing the executor's learning problem: coarse instructions induce avoidable supervision aliasing, goal-preserving local guidance outperforms instruction replacement in our main ablations, and explicit evidence budgeting reduces dependence on broad visual context beyond efficiency considerations.
Across eight real-robot tasks on TX-G2 (an AgiBot G2-compatible variant) and HSR, S2 raises mean subtask success from 54.2% to 79.0% over pi0.5. Together, these results suggest that VLA generalization improves when the executor is trained to act from informative local guidance and task-sufficient visual evidence, rather than recovering both from weak supervision. 

---
# AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes 

**Authors**: Yaoting Wang, Yun Zhou, Zipei Zhang, Henghui Ding  

**Link**: [PDF](https://arxiv.org/pdf/2606.02724)  

**Abstract**: Audio-visual speaker tracking aims to localize and track active speakers by leveraging auditory and visual cues, enabling fine-grained, human-centric scene understanding. This capability is essential for real-world applications such as intelligent video editing, surveillance, and human-computer interaction. However, existing datasets are largely limited to simple or homogeneous audio-visual scenes with coarse annotations. Such oversimplified settings bias evaluation toward static audio-visual co-occurrence, rather than rigorously assessing robust spatiotemporal modeling and cross-modal reasoning in complex, dynamic scenes. To address these limitations, we introduce AVTrack, a human-centric audio-visual instance segmentation (AVIS) dataset designed for dynamic real-world scenarios. AVTrack features diverse and challenging conditions, including camera motion, visual occlusions, and position changes. Evaluations of representative AVIS methods on AVTrack reveal substantial performance degradation, establishing AVTrack as a challenging benchmark for robust human-centric audio-visual scene understanding in complex environments. We further provide a simple yet effective baseline to facilitate future research. Project website: this https URL 

---
# Filter, Then Reweight: Rethinking Optimization Granularity in On-Policy Distillation 

**Authors**: Yuying Li, Leqi Zheng, Yongzi Yu, Wenrui Zhou, Xuchang Zhong, Xing Hu, Jing Jin, Huangjie Yuan, Tao Feng  

**Link**: [PDF](https://arxiv.org/pdf/2606.02684)  

**Abstract**: On-Policy distillation (OPD) in large language models is shifting from full-trace KL supervision toward more selective training paradigms. Recent OPD methods increasingly focus on selecting which trajectories to learn from, which tokens are most informative, and which supervision signals are most reliable. Motivated by this trend, we rethink optimization granularity of OPD and propose \fireicon\ FiRe-OPD (Filter, then Reweight), which jointly adjusts supervision signals at both trajectory and token levels. In details, FiRe-OPD first filters trajectories to remove low-quality rollout samples, and then applies soft reweighting within the retained trajectories to emphasize informative tokens. Compared with hard token selection, FiRe-OPD leverages a soft-weighting mechanism to effectively mitigate information loss and enhance optimization stability, thereby achieving finer-grained OPD optimization. We validate the effectiveness of FiRe-OPD across strong-to-weak, single-teacher, and multi-teacher settings, and demonstrate its superiority over recent token-level OPD methods ( (e.g., +6.25 on AIME 2024 in strong-to-weak, +18.81 on Miner in multi-teacher). Our code is available at this https URL. 

---
# Aligning Data-Driven Predictors with Allocation: A Decision-Focused Approach to Survival Analysis 

**Authors**: Itai Zilberstein, Ioannis Anagnostides, Tuomas Sandholm  

**Link**: [PDF](https://arxiv.org/pdf/2606.02671)  

**Abstract**: Machine learning predictors have become essential tools for guiding automated decision making. However, a major misalignment persists: predictive models are typically optimized in terms of standard statistical metrics in isolation from the algorithmic tasks they inform. We highlight this incongruity in the high-stakes domain of organ allocation by demonstrating that any algorithm relying on (even highly accurate) survival predictors optimized for standard metrics -- such as the Concordance index (C-index) -- can yield arbitrarily poor outcomes when used for allocation, failing to guarantee utility better than a uniform random selection. To bridge the gap between survival analysis and policy optimization, we introduce a decision-focused learning approach based on optimizing normalized discounted cumulative gain (NDCG), a mainstay metric in information retrieval. We establish the utility of NDCG in survival analysis by proving that it translates to guarantees on the performance of allocation. Empirically, we propose a bootstrapping approach to optimize the NDCG of existing survival models. Unlike prior work, we also address the challenge of right censorship when evaluating ranking. On historical heart transplant data from the US, our method dramatically boosts the NDCG of baseline models by 50-100%, which translates to tens of thousands of additional life years gained annually when deployed for transplant allocation. We anticipate that our framework will find broader applications in decision making with predictions. 

---
# Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate 

**Authors**: Marc Pinet, Julien Cumin, Samuel Berlemont, Dominique Vaufreydaz  

**Link**: [PDF](https://arxiv.org/pdf/2606.02670)  

**Abstract**: Many recent multivariate time series anomaly detection (MT-SAD) models incorporate cross-channel modeling, under the implicit assumption that the structure of anomalies may be spread across multiple channels. We evaluate this assumption on eight widely used public benchmarks by introducing a per-segment diagnostic framework that flags, for each labeled anomaly, whether at least one channel deviates individually from its normal history, whether the cross-channel correlation structure changes, or both. The framework shows that no crosschannel rupture occurs without an accompanying univariate deviation across a range of reasonable thresholds. A complementary metric also reveals that on six of the eight benchmarks, at least half of the labeled anomaly segments deviate univariately on 79% to 100% of their timesteps, reaching 100% on three of these datasets. To verify that our framework captures cross-channel structure when present, we construct synthetic data of phase-shifted sinusoidal channels with shared noise. Each anomalous segment is altered through one of two channelwise corruptions that preserve the per-channel marginal distribution while breaking cross-channel structure, and our framework correctly characterizes these segments as cross-channel-only. On these data, channel-dependent (CD) models successfully exploit the cross-channel signal whereas channel-independent (CI) ones fail. The CI/CD comparison of a recent SOTA detector on real benchmarks further confirms that CD modeling brings no measurable gain. We conclude that current MTSAD benchmarks are unsuitable for validating cross-channel modeling capabilities, and we call for the development of more structurally diverse evaluation sets. The code for this study is publicly available. 

---
# AdaWeather: Adaptively Mixing Probabilistic Weather Forecasts with Logarithmic Regret 

**Authors**: Saptarishi Dhanuka, Sarvesh Iyer, Manmeet Singh, Mihir More, Rushil Gupta, Dhruman Gupta, Parthasarathi Mukhopadhyay, Sandeep Juneja  

**Link**: [PDF](https://arxiv.org/pdf/2606.02663)  

**Abstract**: Recent advances in machine learning have produced probabilistic weather forecasting models comparable to state-of-the-art numerical weather predictors. But no model consistently dominates spatio-temporally, and relative performance is highly context-dependent. This motivates adaptive methods for combining multiple forecasts to obtain improvements and robustness. While combined forecasts have been proposed in the literature, these are achieved either through supervised learning or through prediction with expert advice methods. We introduce AdaWeather, an adaptive framework that combines many probabilistic forecasts using both machine learning as well as mixture of experts to arrive at a unified improved probabilistic forecast. While traditional expert methods develop the regret bounds with respect to the best single expert in hindsight, we extend the algorithm and analysis to show our method has logarithmic regret compared to the best static mixture of experts in hindsight. Empirically, we focus on forecasting temperature, and observe improvements over existing methods. 

---
# Improvise, Adapt, Overcome: An On-The-Fly Multifidelity Algorithm for Efficient Machine Learning 

**Authors**: Vivin Vinod, Peter Zaspel  

**Link**: [PDF](https://arxiv.org/pdf/2606.02662)  

**Abstract**: Machine learning has accelerated quantum chemistry but is hindered by the prohibitive cost of generating high fidelity training data. Multifidelity machine learning (MFML) mitigates this overhead by systematically combining abundant low fidelity data with sparse high fidelity data. In spite of its success, standard MFML schemes rely on pre-defined scaling factors to determine sparse data ratio across fidelities, often generating redundant multifidelity data resulting in a loss of efficiency. Here, we introduce an adaptive on-the-fly multifidelity framework for machine learning that autonomously determines training dataset composition. By dynamically querying training samples at each fidelity, the algorithm saturates model accuracy at lower fidelities before moving up to more expensive reference calculations. We benchmark the novel adaptive-MFML across diverse chemical properties including the computational chemistry gold standard coupled cluster energies, and the more chemically challenging excitation energies. In our numerical experiments we show that our adaptive algorithm reduces data generation costs by up to a factor of 30 compared to single fidelity methods and improves upon standard MFML by up to a factor of 5. The mitigation of data redundancy establishes a high-accuracy low-cost pathway for sustainable cost-aware machine learning in quantum chemistry. 

---
# Learning to Refine: Spectral-Decoupled Iterative Refinement Framework for Precipitation Nowcasting 

**Authors**: Yunlong Zhou, Chen Zhao, Danyang Peng, Fanfan Ji, Xiao-Tong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.02661)  

**Abstract**: Accurate precipitation nowcasting is vital for disaster mitigation, but deep learning methods face a key trade-off: regression models produce over-smoothed, spectrally decaying predictions that blur convective details and violate turbulence power laws; diffusion models generate realistic yet unanchored hallucinations lacking physical grounding. We propose Spectral-Decoupled Iterative Refinement (SDIR), a deterministic framework that reformulates nowcasting as progressive frequency-decoupled refinement. SDIR first extracts a stable low-frequency synoptic skeleton, then iteratively refines high-frequency textures under physical constraints, eliminating both blurring and hallucinations. It features a dual-path design: the Synoptic Frequency-Guided Former (SFG-Former) with Scale-Adaptive Transformers for global structure, and the Fourier Residual Refiner (FR-Refiner) with Scale-Conditioned Fourier Neural Operators for fine residuals. A Physically Consistent Power Spectral Density (PCPSD) loss with dynamic masking enforces a turbulence-consistent spectral distribution. Experiments on three benchmarks show SDIR significantly outperforms SOTA methods in spatial accuracy while achieving spectral fidelity competitive with diffusion-based methods, enabling reliable high-resolution operational nowcasting. Code link: this https URL. 

---
# CL-DMDF:Dynamic Multimodal Data Fusion Model Based on Contrastive Learning 

**Authors**: Dong Li, Lingling Zhang, Binghao Han, Linlin Ding, Yue Kou  

**Link**: [PDF](https://arxiv.org/pdf/2606.02659)  

**Abstract**: Multimodal data fusion involves integrating and analyzing information from multiple modalities to uncover latent correlations and complementary patterns, thereby enhancing data processing and decision-making. While existing methods for structured multimodal inputs are typically designed around specific tasks and assume fully observed modalities, real-world applications often suffer from uncertain or missing modality inputs due to various factors. Some traditional models overly emphasize local interactions within missing modalities, neglecting the global complementary cues embedded in multimodal representations. To overcome these limitations, we propose a Dynamic Multimodal Data Fusion model based on Contrastive Learning (CL-DMDF). CL-DMDF introduces a novel attention mechanism that operates across both feature and modality dimensions to compute reliable attention scores, effectively reflecting importance at each level. The CL-DMDF further incorporates an entity-centroid contrastive learning module that constructs centroid-based positive samples from entity features to enhance discriminative learning. Additionally, an adaptive fusion module is employed to improve the efficiency and accuracy of dynamic fusion strategies. Extensive experiments conducted on three datasets demonstrate the effectiveness of the CL-DMDF across diverse multimodal fusion tasks. 

---
# The Ringelmann Effect in Multi-Agent LLM Systems: A Scaling Law for Effective Team Size 

**Authors**: Blaž Bertalanič, Carolina Fortuna  

**Link**: [PDF](https://arxiv.org/pdf/2606.02646)  

**Abstract**: Inference-time multi-agent LLM scaling lacks a shared unit: counting nominal agents conflates cost with independent evidence. We derive a two-parameter scaling law $R(N) = N_\text{eff}/N = 1/(1+c(N-1)N^{-\beta})$ where the regime exponent $\beta$ classifies any configuration into one of three asymptotic regimes -- hard-ceiling at $1/c$ ($\beta = 0$), sublinear at $N^\beta/c$ ($0 < \beta < 1$), or linear ($\beta \ge 1$), and a mean-field theorem predicts that peer count $k$ and rounds $\tau$ during agent debate enter the dynamics only through their product $k\tau$. The law applies at two levels: answer diversity and correctness redundancy.
Across 44 (model $\times$ task $\times$ condition) cells spanning peer debate, self-correction, random-noise placebo, self-consistency, three open-weight families (Qwen, Llama, Ministral) at scales from 7B to 32B with a frontier API check (Gemini), thinking models, heterogeneous teams, and sparse communication, the functional form fits every condition at $R^2 > 0.99$; only $(c, \beta)$ shifts. On free-form math, dense peer influence collapses the answer-level regime from sublinear into hard-ceiling; correctness-level fits remain hard-ceiling throughout. Three findings have practical implications. \emph{(i)}~Thirty dense debating agents produce no more answer diversity than one on MMLU-Hard. \emph{(ii)}~A noise placebo tracks self-correction on free-form math and at $4\times$ scale, so within homogeneous teams the gain commonly attributed to ``debate'' comes from re-evaluation, not peer content. \emph{(iii)}~A single $N \le 5$ pilot predicts the $N=30$ structural ceiling, and within the configurations tested only architectural diversity (heterogeneous teams) lowers $c$ and escapes the hard-ceiling regime, communication-mode interventions do not. 

---
# Target Updates May Stabilize Linear Q-Learning: Periodic and Soft Dynamics 

**Authors**: Donghwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.02645)  

**Abstract**: Periodic target updates in Q-learning and soft target updates in actor-critic methods are empirically well established stabilization mechanisms, but their precise theoretical explanation is still incomplete. This paper gives a rigorous and exact analysis of these mechanisms for Q-learning with linear function approximation (linear Q-learning) using the exact switched linear system (SLS) dynamics induced by the Bellman maximum and the joint spectral radius (JSR) of the resulting switching matrix families. Although linear Q-learning can fail to converge in general, we prove that, under explicit spectral and step-size conditions, periodic hard target updates and soft target updates can guarantee convergence to the exact projected Q-Bellman solution. The main analysis is carried out for deterministic linear Q-learning, where the target-update mechanism is most transparent. Once the corresponding JSR certificate is established for the mean recursion, the stochastic reinforcement-learning setting can be treated by replacing deterministic modes with sampled stochastic modes and adding the corresponding stochastic-noise analysis. 

---
# A New Framework for Cybersecurity Refusals in AI Agents 

**Authors**: Eliot Krzysztof Jones, Mateusz Dziemian, Matt Fredrikson, J Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2606.02644)  

**Abstract**: Agentic scaffolds have dramatically improved LLM performance on complex, long-horizon tasks, yielding both broad benefits and amplified risks in domains like cybersecurity. Existing benchmarks for AI agents in cybersecurity focus mainly on measuring proficiency--how effectively agents can complete offensive security tasks--but neglect a critical question: when and how should agents refuse harmful requests? We present the first framework for establishing refusal boundaries in offensive security contexts. Our framework defines (1) principled criteria for when tasks should be refused, (2) categories of tasks that warrant refusal, and (3) evaluation methodology for measuring agent robustness under both benign and adversarial conditions. We apply this framework to assess how current LLM-powered agents adhere to appropriate refusal boundaries across a range of web-based offensive security scenarios, finding that 6 of 8 frontier models tested show near-zero refusal rates, with only 2 models (GPT-5.2 and GPT-5.1 Codex) demonstrating any meaningful refusal behavior. 

---
# Inference Cost Attacks for Retrieval-Augmented Large Language Models 

**Authors**: Chengliang Liu, Liangbo Ning, Yujuan Ding, Wenqi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2606.02643)  

**Abstract**: Retrieval-Augmented Generation (RAG)-enhanced LLM systems, while powerful, introduce substantial inference costs due to the inclusion of an extra multi-stage pipeline that dynamically retrieves and synthesizes information from external knowledge sources. This high operational cost exposes a critical vulnerability to Inference Cost Attacks (ICAs). However, existing ICAs often rely on the impractical assumption of direct prompt manipulation. We argue that a more feasible and potent threat to RAG-enhanced LLM systems arises from poisoning external knowledge bases (e.g., web knowledge from the Internet). In this work, we introduce the Retrieval-Augmented Inference Cost Attack (RA-ICA), a novel attacking paradigm that targets the computational cost of RAG-enhanced LLM systems by injecting malicious documents into external knowledge corpus. To operationalize this attack, we propose Computational Resource Exhaustion via External Poisoning (CREEP), a novel framework that leverages LLM agents to automatically craft malicious documents that are both semantically relevant for retrieval and potent for inducing an abnormal increase in token consumption during the inference phase. To enhance the attack's effectiveness, we introduce Memory-Augmented Group Relative Policy Optimization (MA-GRPO), a novel reinforcement learning algorithm that fine-tunes the agents by learning from a dynamic memory of historical best adversarial documents. Extensive experiments across three real-world datasets demonstrate that RA-ICA increases token consumption by up to 13.12 times with an over 90% success rate, without degrading the integrity of the generated answer. 

---
# SVHalluc: Benchmarking Speech-Vision Hallucination in Audio-Visual Large Language Models 

**Authors**: Chenshuang Zhang, Kyeong Seon Kim, Chengxin Liu, Tae-Hyun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2606.02642)  

**Abstract**: Despite the success of audio-visual large-language models (LLMs), they can produce plausible but ungrounded outputs, termed hallucination. Existing benchmarks focus on environmental sounds (e.g., dog barking) to indicate event occurrence. In contrast, human speech carries fundamentally different, rich semantics and temporal structures, yet it remains unexplored whether current models can accurately align speech content with corresponding visual signals. In this work, we show that speech content can induce hallucinations in audio-visual LLMs. To systematically study this, we introduce SVHalluc, the first comprehensive benchmark for evaluating speech-vision hallucination in audio-visual LLMs. Our benchmark diagnoses speech-vision hallucinations from two critical and complementary aspects: semantic and temporal. Experimental results demonstrate that state-of-the-art open-source audio-visual LLMs struggle with aligning speech content with corresponding visual signals, with a near-random accuracy on multiple tasks. In contrast, Gemini 2.5 Pro significantly outperforms the open-source models. Our analysis suggests that their failures stem from limited ability in cross-modality understanding, despite strong performance in single-modality perception. Our work uncovers a new and fundamental limitation of current audio-visual LLMs and highlights the need for speech-grounded video comprehension. Project page: this https URL. 

---
# CARVE: Certified Affordable Repair of Vetoed Maneuvers via Envelopes for Interactive Driving 

**Authors**: Yifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02641)  

**Abstract**: Interactive driving exposes a failure mode that is easy to miss in rule-aware autonomous-driving stacks: a hard-rule margin can be negative for an ego candidate even though a small lawful accommodation by a non-priority agent would restore feasibility. Existing rulebooks, shields, and reachability filters are strong at vetoing unsafe actions, while prediction-based planners model likely responses. Neither returns a runtime proof object that states which bounded multi-agent edit repairs the maneuver, who owns the edit, whether the request is right-of-way affordable, and what ego fallback remains if the request is not observed. We formulate this missing object as *interactive repair certification* and introduce *CARVE*, a prediction-free certificate layer over a finite lattice of ego-owned and agent-owned tactical operators. Agent-owned requests are admissible only inside \(B_j(s) = \beta(\pi_j)\alpha_j^{\max}(s)\), a cooperation envelope that separates kinematic reachability from normative priority. The resulting certificate records the binding rule, repair category, repair set, responsibility-weighted cost split, and fallback. On 589 Lanelet2-geometry-grounded INTERACTION replay episodes, CARVE-Greedy accepts 98.64% of initially vetoed maneuvers and recovers 370/378 human-resolved false vetoes, while preserving 589/589 right-of-way respect, zero priority-agent false positives, and 400/400 negative-stress vetoes. We prove certificate soundness, structural right-of-way respect, exact finite-lattice minimality, fallback contingency, and blame-consistency conditions. CARVE does not predict or require another driver's compliance; it certifies whether a proposed interaction is bounded, attributable, and normatively admissible under declared assumptions. 

---
# D-Judge: Disrupting Multi-Turn Jailbreaks using Semantics-Preserving Output Rewriting 

**Authors**: Huanli Gong, Zhipeng Wei, Yu Fu, Haz Sameen Shahgir, Ananya Gupta, Yue Dong, N. Benjamin Erichson  

**Link**: [PDF](https://arxiv.org/pdf/2606.02640)  

**Abstract**: Multi-turn jailbreak attacks pose a growing threat to large language model (LLM) safety because they exploit feedback from auxiliary judge models to iteratively refine prompts toward harmful goals. Existing defenses largely detect or block unsafe content at individual turns or at the final response, leaving the judge-driven refinement loop intact and allowing attackers to extract informative feedback from intermediate interactions. We introduce D-Judge, a semantics-preserving output rewriting defense that intervenes directly in this loop by rewriting the victim LLM's responses before they are evaluated by the attacker's judge. By misaligning the judge's feedback signal without changing the meaning of the original response, D-Judge derails the attacker's prompt-refinement process, causing subsequent queries to be optimized against a distorted signal of attack progress. To improve D-Judge's ability to produce such rewrites, we construct a dataset of semantically equivalent response pairs that induce different judge-assigned harmfulness scores, and use it for supervised fine-tuning followed by direct preference optimization. Experiments on HarmBench show that D-Judge reduces the success rate of state-of-the-art multi-turn jailbreaks while preserving performance on benign benchmarks. 

---
# Sparse-View Lung Nodule Volumetry from Digitally Reconstructed Radiographs via AReT: Anatomy-Regularized TensoRF 

**Authors**: Spoorthi M, Suja Palaniswamy  

**Link**: [PDF](https://arxiv.org/pdf/2606.02639)  

**Abstract**: We identify and resolve a previously unreported failure mode in TensoRF when applied to X-ray attenuation fields: the default density shift of -10, originally introduced for RGB scene reconstruction, suppresses density gradients and prevents sparse-view medical reconstruction regardless of learning rate or regularization strategy. Setting the density shift to zero restores gradient flow and enables stable volumetric reconstruction of pulmonary nodules from only three orthogonal X-ray projections. Building on this, we propose AReT, an anatomy-regularized tensorial radiance field framework for lung nodule reconstruction using coronal, sagittal, and axial projections from the LIDC-IDRI dataset (19 patients, radiologist-annotated nodules). Unlike existing NeRF approaches requiring dense multi-view acquisition, AReT is designed for sparse-view thoracic imaging and incorporates chest-anatomy-aware regularization combining L1 sparsity and total variation smoothness. A systematic comparison across 11 reconstruction strategies shows anatomy-aware regularization consistently outperforms generative-prior-guided approaches. Evaluated against radiologist consensus segmentations, AReT achieves Pearson r=0.983 (p<0.0001) for clinically actionable nodules >=10 mm (n=14), median absolute volumetric error of 11.4%, near-zero systematic bias of -77.3 mm^3, and 8.4x improvement over spherical volume approximation. 

---
# SegTune: Structured and Fine-Grained Control for Song Generation 

**Authors**: Yuejiao Wang, Zihao Ji, Pengfei Cai, Xu Li, Haorui Zheng, Zewen Song, Zhongliang Liu, Chen Zhang, Pengfei Wan  

**Link**: [PDF](https://arxiv.org/pdf/2606.02638)  

**Abstract**: Recent advances in neural song generation have enabled high-quality synthesis from lyrics and global textual prompts. However, most systems fail to model temporally varying attributes of songs, severely limiting fine-grained control over musical structure and dynamics. To address this, we propose SegTune, a Diffusion Transformer-based framework enabling structured and fine-grained controllability by allowing users or large language models (LLMs) to specify local musical descriptions aligned to song segments. These segment prompts are temporally broadcast to corresponding time windows, while global prompts ensure stylistic coherence. To support precise lyric-to-music alignment, we introduce an LLM-based duration predictor that autoregressively generates sentence-level timestamps in LyRiCs format. We further construct a large-scale data pipeline for high-quality song collection with aligned lyrics and prompts, and propose new metrics to evaluate segment alignment and vocal consistency. Experiments demonstrate that SegTune outperforms existing baselines in both musicality and controllability. Visit our project page (this https URL) for codes and more generated songs. 

---
# Too Much of a Good Thing: When sim2real Efforts Impede Policy Learning (And What to Do About It) 

**Authors**: Kyle Morgenstein, Bharath Masetty, Stephen Welch, Luis Sentis  

**Link**: [PDF](https://arxiv.org/pdf/2606.02636)  

**Abstract**: While sim2real efforts are necessary for effective policy transfer to hardware, there is such a thing as too much of a good thing. We argue that sim2real efforts have led to misaligned incentives with policy learning, resulting in simulator lock in and poor policy exploration due to the unreasonable constraints imposed by the real world. We offer a diagnosis and explanation of the current status of the problem, and propose a potential solution via a sim2sim2real paradigm that leverages the robot's kinematics as the sole design constraint. 

---
# Echo-POSED: Geometric Self-Distillation for Echocardiography Guidance 

**Authors**: Elias Stenhede, Edvart Grüner Bjerke, Joanna Sulkowska, Eivind Bjørkan Orstad, Ole Jakob Elle, Ulysse Côté-Allard, Arian Ranjbar  

**Link**: [PDF](https://arxiv.org/pdf/2606.02634)  

**Abstract**: We introduce Echo-POSED, a self-supervised framework for real-time transthoracic echocardiography (TTE) guidance that recommends probe adjustments directly from 2D ultrasound images, without the need for expert-labelled views or tracked probe trajectories. Instead, it trains on 2D views sliced from routinely acquired 3D echocardiography volumes, enforcing equivariance to probe motions while remaining invariant to cardiac phase, yielding a pose representation on $\mathrm{SO}(3)\times\mathrm{SO}(3)$. Across a held-out split and public external 3D--TTE datasets (including vendor shift), Echo-POSED maintains geometric consistency under virtual perturbations and enables intra- and inter-patient guidance simulations, achieving a combined mean angular error of 8.2 degrees between the guided and target views in intra-patient simulations with cardiac motion. 

---
# Position: Prioritize Identifying Structure, Not Complex Models, for Scientific Discovery 

**Authors**: Tyler H. McCormick  

**Link**: [PDF](https://arxiv.org/pdf/2606.02632)  

**Abstract**: Modern Machine Learning (ML) and Artificial Intelligence (AI) models, especially large language models (LLMs), are increasingly used to generate scientific hypotheses and mechanistic explanations from observational data. This position paper argues that in the high-dimensional proxy regimes where modern ML excels, mechanistic learning is generically underdetermined: many incompatible mechanisms induce essentially the same observational relationships on the support of the data, so predictive success and coherent explanations are insufficient evidence of mechanism discovery. This underdetermination becomes uniquely hazardous with large language models (LLMs), which tend to collapse large equivalence classes of explanations into a single fluent narrative. This paper proposes concrete standards for ``mechanistic ML,'' and argues these norms are necessary if LLM-centered workflows are to support science rather than merely simulate it. 

---
# Wavelet as Tokenizer: Preliminary Results on a Shared Wavelet Token Schema for Natural Signals 

**Authors**: Shenghao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2606.02631)  

**Abstract**: This paper studies whether audio, images, and video can share a common wavelet token schema rather than relying on separate modality-specific latent grids. It introduces a preliminary continuous-token model built around a one-level Haar DWT/IDWT frontend, a shared coefficient-token layout, optional structural metadata, lightweight modality value adapters, and a shared token-wise encoder-decoder trunk. On Speech Commands, EuroSAT RGB, and DAVIS 2017 data, a dense shared model reaches 39.92 dB audio, 29.37 dB image, and 23.93 dB video PSNR. A matched-rate sweep under continuous latent scalar budgets indicates that the visual gains are not explained solely by latent capacity, while also showing that additive metadata embeddings are not a universal source of improvement. Finally, fixed-rate energy selection provides a strong non-parametric baseline: energy_global improves average PSNR over uniform selection by 16.73 dB for audio, 16.90 dB for images, and 15.86 dB for video under compressed keep ratios. Masked sparse training reaches 34.45 dB video PSNR with 50% of dense tokens. The results support a unified wavelet token schema and sparse token interface, while stopping short of establishing a universal discrete vocabulary. 

---
# MultiTurnPSB: Evaluating Multi-Turn Jailbreak Attacks an dClassifier-Based Defenses for Medical AI Safety 

**Authors**: Anushka Sheoran, Yiduo Hao  

**Link**: [PDF](https://arxiv.org/pdf/2606.02630)  

**Abstract**: Patient-facing medical chatbots are commonly evaluated on single-turn prompts, yet real users push back after refusals, add urgency, and invoke authority. We introduce MultiTurnPSB, a four-turn adversarial extension of PatientSafetyBench, and evaluate GPT-4.1-mini under fixed template, template-adaptive, and live adversarial attacks. Unsafe responses rise from 35% to nearly 80% by Turn 4 under live attack. Under the same adversary, GPT-4.1-mini and Claude Sonnet 4.5 are statistically indistinguishable at baseline but diverge to a 19x gap by Turn 4, a difference invisible to single-turn evaluation. We characterize four degradation trajectory signatures and identify a two-element attack formula responsible for most catastrophic failures. A lightweight input-side classifier reduces Turn 4 unsafe responses by 52 percentage points despite severe accuracy degradation, but the 45% false alarm rate on benign queries is the primary deployment constraint. A methodological finding also emerges: Claude Sonnet refused to generate adversarial messages in over half of late-turn conversations despite explicit red team framing, suggesting safety training may generalize to the attacker role. 

---
# Enhancing Protein-Protein Interaction Prediction with Hierarchical Motif-based Multimodal Protein Embedding 

**Authors**: Zaifei Yang, Samuel Ping-Man Choi, James Kwok  

**Link**: [PDF](https://arxiv.org/pdf/2606.02629)  

**Abstract**: Protein-protein interactions (PPIs) are essential for many biological processes. However, existing PPI prediction approaches suffer from two major limitations: they overlook the hierarchical organization of proteins, particularly meso-scale motifs that critically regulate PPIs, and fail to effectively integrate sequence, structure, and function modalities. To address these limitations, we propose MMM-PPI, a Hierarchical Motif-based Multi-Modal protein Encoder for PPI Prediction that constructs PPI embeddings in a bottom-up multi-modal manner across three scales. At the micro-scale, we encode three modal residue features; at the meso-scale, a novel multimodal motif encoder aggregates residues into spatially-informed motif embeddings; at the macro-scale, a multimodal protein encoder integrates motifs into protein embeddings by jointly modeling motif importance and inter-modal correlations. The pre-trained encoder can be used off-the-shelf for large-scale PPI prediction. Extensive experiments on multiple PPI datasets show that MMM-PPI outperforms state-of-the-art multi-label PPI prediction models, particularly under challenging data partitions and limited data scenarios. Codes are in this https URL. 

---
# DXA-Derived Skeletal Phenotypes and Hip Fracture Risk: A Backdoor-Adjusted Causal Analysis 

**Authors**: Zixin Shi, Chen Zhao, Meiling Zhou, Kevin A. Maupin, Joyce H. Keyak, Nancy E. Lane, Kuan-Jui Su, Hui Shen, Hong-Wen Deng, Kui Zhang, Weihua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.02625)  

**Abstract**: Purpose: To compare dual-energy X-ray absorptiometry (DXA)-derived hip skeletal phenotypes in relation to hip fracture risk using prespecified confounder adjustment and to assess whether phenotypes ranked by their backdoor-adjusted average treatment effects (ATEs) improve risk stratification. Methods: We analyzed 21,098 UK Biobank participants with linked health records, hip DXA-derived skeletal measures, and prespecified covariates. Sixteen phenotypes spanning bone mineral content (BMC), bone mineral density (BMD), and T-score across hip-related regions were evaluated. Confounder selection was guided by a prespecified directed acyclic graph (DAG). Backdoor-adjusted ATEs were estimated on the absolute risk-difference scale per standard deviation (SD) increase. Effect heterogeneity was evaluated for total femur BMD, and downstream prediction was assessed using clinical variables combined with phenotypes ranked by ATE magnitude. Results: Among 21,098 participants, 115 had hip fractures. All 16 phenotypes showed negative backdoor-adjusted ATEs per SD increase. The largest ATEs were observed for total femur BMC and total femur BMD, each with a risk difference of -0.0047, corresponding to approximately 4.7 fewer hip fractures per 1,000 participants per SD higher phenotype value. Conditional effects of total femur BMD were stronger among older participants and those with lower BMI. In prediction, clinical variables plus the top 11 ATE-ranked phenotypes achieved higher AUC than FRAX with femoral neck BMD (0.842 vs. 0.709), with higher sensitivity (0.748 vs. 0.443) and similar specificity (0.793 vs. 0.777). Conclusion: DXA-derived hip skeletal phenotypes differed in their backdoor-adjusted ATEs. Phenotype-level causal evaluation may help identify informative DXA measures for risk stratification. 

---
# TadA-Bench: A Million-Variant Benchmark for Future-Round Discovery Toward Agentic Protein Engineering 

**Authors**: Jin Gao, Juntu Zhao, Zirui Zeng, Jiaqi Shen, Junhao Shi, Dukun Zhao, Yuming Lu, Dequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02624)  

**Abstract**: AI for scientific discovery is entering an agentic era, where protein-engineering systems are expected to prioritize future wet-lab experiments rather than merely fit static measurements. We introduce TadA-Bench, a million-variant wet-lab replay benchmark from 31 TadA directed-evolution rounds for future-round discovery toward agentic protein engineering. TadA-Bench preserves the campaign chronology and defines a fixed-data replay task: given earlier experimental rounds, models rank variants that appear only in later rounds. It provides aligned DNA, RNA, and protein views, and uses Seq2Graph, a graph-based label-unification pipeline, to reconcile noisy enrichment measurements into consistent cross-round activity labels. Random-split controls show strong interpolation, but future-round ranking and finite-budget candidate selection are much weaker. Controlled analyses suggest that evolutionary coverage is more informative than local data density, positioning TadA-Bench as a reproducible wet-lab replay substrate for future-round discovery toward agentic protein engineering; the data and code are released on Hugging Face and GitHub. 

---
# Oscillatory State-Space Models as Inductive Biases for Physics-Informed Neural PDE Solvers 

**Authors**: Abhishek Chandra, Taniya Kapoor  

**Link**: [PDF](https://arxiv.org/pdf/2606.02623)  

**Abstract**: Solving time-dependent partial differential equations (PDEs) is an important problem in computational science and engineering. Physics-informed neural networks (PINNs) learn PDE solutions from governing equations. However, accurately capturing temporal evolution remains challenging. Recent sequence-model-based approaches parameterize time evolution using general-purpose sequence models, which capture temporal dependencies but do not explicitly encode the structured dynamics of PDE solutions. In addition, their memory requirements can scale unfavorably with sequence length and resolution, limiting applicability in large-scale or high-dimensional settings. This work introduces a PINN approach that incorporates oscillatory state-space dynamics to represent the modal structure of PDE solutions. The proposed method leverages a linear-oscillator-based temporal evolution, together with a PDE-aware spectral basis in space. This design enables closed-form spatial differentiation and consistent enforcement of boundary conditions. The method is evaluated on forward, inverse, and high-dimensional PDE problems, including cases up to 100 spatial dimensions. The results show improved accuracy and reduced memory usage compared to recent sequence-model-based PINN approaches. Overall, this work highlights the benefits of incorporating structured dynamical priors into the temporal evolution of neural PDE solvers and suggests designing more physics-aligned and computationally efficient PINN architectures. 

---
# Closed-Loop Molecular Design with Calibrated Deference 

**Authors**: Newman Cheng, Gordon Broadbent IV, Jason Dong, Syed Mohammed Ali Hussaini, Farman Ullah, Morris Sharp, Gabrielle Barnes, Nanlin Guo, Deyu Zou, Karin Strauss, William Chappell, David G. Kwabi, Bichlien H. Nguyen, Jake A. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2606.02618)  

**Abstract**: We present Cognitive Loop via In-Situ Optimization (CLIO), an agent that couples a continuously-updated belief-state graph with a recursive plan-then-act loop. The result is a reasoning agent that can contribute something qualitatively different, which we term \emph{calibrated deference}: the capacity to recognize when its own tools or assumptions are failing, to adapt its strategy in response, and to generate mechanistic hypotheses that guide experimental revision. We tested CLIO in a closed-loop human-AI campaign to design an aqueous organic redox flow battery (AORFB) negolyte, with CLIO leading proposal and interpretation in close partnership with chemists who synthesized, characterized, and weighed in on design choices. Across 17 candidates over three rounds, CLIO converged on a top phosphonate candidate; characterization confirmed a 130~mV improvement in redox potential over the literature baseline. Characterization then revealed unexpectedly poor electrochemical reversibility -- a regression no property predictor had flagged. CLIO generated competing mechanistic hypotheses, prioritized discriminating diagnostics, traced the failure to phosphonate-potassium ion pairing, and prescribed a sulfonate replacement. The resulting compound showed substantially improved electrochemical reversibility and maintained a 90~mV improvement in redox potential, closing the design-make-test-redesign loop. 

---
# FSA-GRPO: Teaching Auditory LLMs to Use Few-shot Demonstrations 

**Authors**: Haolong Zheng, Siyin Wang, Xulin Fan, Zengrui Jin, Mark Hasegawa-Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2606.02615)  

**Abstract**: Few-shot prompting provides an effective way to adapt auditory large language models to low-resource tasks such as children's speech recognition. However, most auditory large language models are not explicitly trained to perform inference in this demonstration-conditioned format, limiting the extent to which they can benefit from few-shot prompting. To address this limitation, we introduce Few-Shot Aware GRPO (FSA-GRPO), an RL-based post-training recipe that uses a specially designed reward to encourage the model to leverage few-shot demonstrations, thereby strengthening its few-shot adaptation ability. Notably, training with only high-resource adult ASR data improves the model's general few-shot adaptation ability, yielding gains not only in children's speech recognition but also in speech translation and audio understanding. We further study data selection and auxiliary reward weighting to identify an effective training recipe. Our experiments show that when in-domain data are unavailable or cannot be used for training, FSA-GRPO is more effective than direct tuning on related out-of-domain data. 

---
# Margin Play: A Multi-Agent System For Public Policy Analysis In The Brazilian Equatorial Margin 

**Authors**: Antonio de Sousa Leitão Filho, Fabrício Saul Lima, Selby Mykael Lima dos Santos, Rejani Bandeira Vieira Sousa, Luís Jorge Mesquita de Jesus, Dennys Correia da Silva, Allan Kardec Duailibe Barros Filho  

**Link**: [PDF](https://arxiv.org/pdf/2606.02614)  

**Abstract**: The Brazilian Equatorial Margin (BEM) is Brazil's next offshore oil frontier, with operations expected to begin in 2026 in the Foz do Amazonas basin. Its assets are fiscally and territorially linked primarily to Maranhao -- the state with the lowest HDI in the Federation (0.676, IBGE 2022). This raises the central policy question: under what conditions does BEM exploration generate net positive externalities for Maranhao? The problem is intrinsically multi-agent: the Federal Government seeks revenue and energy security; the state seeks regional welfare under constitutional royalty earmarking; the operator maximizes profit under risk; ANP and IBAMA hold conflicting mandates; and Amazonian communities prioritize territorial and environmental vectors over monetary income. We present Margin Play, a Multi-Agent Reinforcement Learning (MARL) system simulating these tensions under Brazilian empirical calibration and classical economic literature. It implements six agents under the CTDE paradigm, trained with BRO-MARL. Results from 60,000 episodes across six scenarios indicate the answer is conditional on the institutional regime: under the reference baseline, the welfare gain is marginal (Waval approx. 1.68), whereas the MA-Prospero configuration yields Delta W = +17.5% and Delta Rcom = +21.3%, with a lower environmental liability (Eamb = 0.048 vs. 0.076). The fundamental problem is not a trade-off between production and welfare, but the choice of public policy regime linked to exploration. 

---
# Samudra 2: Scaling Ocean Emulators across Resolutions 

**Authors**: Yuan Yuan, Jesse Rusak, Alexander Merose, Adam Subel, Pavel Perezhogin, Alistair Adcroft, Carlos Fernandez-Granda, Laure Zanna  

**Link**: [PDF](https://arxiv.org/pdf/2606.02610)  

**Abstract**: Ocean general circulation models (OGCMs) are essential to climate science but computationally expensive, limiting ensemble size and forcing scenarios. Neural emulators promise orders-of-magnitude speedups, yet existing ocean emulators have not combined fine spatial resolution with multi-year autoregressive rollouts. Samudra, the first autoregressive neural ocean emulator to produce multi-decade global rollouts, is limited to $1^\circ$ resolution and exhibits two long-horizon failure modes: \emph{variance collapse}, the loss of temporal variability, and \emph{imprinting artifacts}, in which velocity patterns leak into deep-ocean fields. We present Samudra 2, which introduces a wider U-Net backbone with modified ConvNeXt-style blocks and a reduced block-internal expansion factor, together with a dynamic loss that reweights output channels according to their prediction errors, strengthening gradients for slow-evolving deep-ocean fields. At $1^\circ$, Samudra 2 increases upper-ocean global-mean temperature $R^2$ from 0.56 to 0.87 and reduces deep-ocean temperature error by roughly sevenfold. The same architecture scales to $1/2^\circ$ and $1/4^\circ$ over approximately 8-year autoregressive rollouts, recovering mesoscale eddies and sharp western boundary currents. Running on a single GPU, Samudra 2 enables larger ensembles for sea-level projections, ocean heat uptake, and climate variability studies. We provide code, documentation, and benchmark resources at this https URL. 

---
# Building Better Activation Oracles 

**Authors**: Jan Bauer, Celeste De Schamphelaere, Adam Karvonen, Niclas Luick, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2606.02609)  

**Abstract**: Activation Oracles (AOs) are promising methods for interpreting residual stream activations. However, current AOs face important issues, such as hallucinations and vagueness. Additionally, text-inversion confounds make them hard to evaluate. To this end, we improve the Activation Oracle (AO) training regime in four ways: training on on-policy rollouts, improving the conversational dataset, feeding more layers and an improvement to the injection formula. The capability improvements are marginal, but quality of life improvements are quite substantial. In addition, we open source the first comprehensive evaluation suite for AO quality, which we call AObench. Overall, we hope that our work sets a foundation that helps improve AOs and other models in the paradigm of scalable, end-to-end interpretability. 

---
# Geometry-Aware Tabular Diffusion 

**Authors**: David Turtora Zagardo  

**Link**: [PDF](https://arxiv.org/pdf/2606.02607)  

**Abstract**: Tabular synthesis is critical for privacy-preserving sharing and augmentation, yet diffusion models rely on implicit mechanisms to capture inter-column relationships. We introduce Geometry-Aware Tabular Diffusion (GATD), which augments tabular diffusion denoisers with pairwise angles and lengths computed from column value differences and used as inputs and auxiliary targets. Our MLP instantiation achieves state-of-the-art benchmark performance while using 3.5x fewer parameters on average (up to 25x for classification tasks): on ten datasets, it wins 8/10 Shape, 7/10 Trend, and 9/10 downstream utility (F1/RMSE), reducing Shape and Trend error by 27% and 20%. Default loss weights transfer to GNN and Transformer denoisers, improving Shape on 27/30 and Trend on 25/30 architecture-dataset cells. A matched ablation shows supervision (not extra inputs or capacity) drives the gain. This shows explicit relational supervision is a portable inductive bias for tabular diffusion. 

---
# ReLoRA: Knowledge-Reusing Adaptation for Fast Rollout of Evolving LLM Services 

**Authors**: Yang Xu, Zihuai Xu, Hongli Xu, Yunming Liao, Zhiwei Yao, Xitong Fu  

**Link**: [PDF](https://arxiv.org/pdf/2606.02606)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as continuously evolving services, where frequent base-model updates may invalidate previously deployed task-specific Low-Rank Adaptation (LoRA) adapters. For service providers managing numerous downstream model services, retraining each LoRA adapter from scratch for every updated base model is computationally prohibitive and delays service rollout. Meanwhile, the simpler alternative, i.e., naively applying the original LoRA adapter to the updated base model, often leads to degraded service quality due to adapter-backbone incompatibility. To address this problem, we propose ReLoRA, a knowledge-reusing re-adaptation framework that efficiently restores service-ready LoRA adapters for evolving LLM services while preserving or improving task performance. Specifically, ReLoRA comprises two key optimization steps: 1) Adaptive LoRA initialization leverages Bayesian optimization to construct a compatibility-aware starting point by fusing information from both the previously deployed task adapter and the base model's evolution; 2) Fine-tuning with scheduled regularization first rapidly steers the adapter to a high-quality region via strong regularization, followed by relaxed regularization for task-specific refinement. This design enables rapid service-quality recovery with reduced re-adaptation overhead. Extensive experiments demonstrate that ReLoRA reduces time-to-readiness by up to 8.9$\times$ and improves accuracy by up to 4.6\% compared to baselines. 

---
# Cross-Modal Contrastive Learning of ECG and Angiography Representations for Severe Stenosis Classification 

**Authors**: Nikola Cenikj, Özgün Turgut, Alexander Müller, Alexander Steger, Jan Kehrer, Marcus Brugger, Daniel Rueckert, Philip Müller  

**Link**: [PDF](https://arxiv.org/pdf/2606.02605)  

**Abstract**: Coronary artery stenosis is a common cardiovascular disease, with severe, untreated cases posing significant risks of heart attack. Although coronary (X-ray) angiograms remain the standard for stenosis diagnosis, they are invasive, time- and resource-intensive, and therefore only performed on patients with a high probability of disease based on symptoms and prior clinical tests. However, a subset of patients, especially those without symptoms, may remain undiagnosed. Detecting indications of stenosis from ECGs, which are fast, cheap, non-invasive, and thus routinely acquired even in asymptomatic patients, would support early diagnosis. However, as no reliable stenosis-specific signal has been identified in ECGs, they can not currently be used for stenosis risk stratification. To address this, we introduce StenCE, a pretraining framework, allowing stratification of patients based on features derived directly from ECGs. Evaluations across varying stenosis severity thresholds and additional ECG disease classification tasks demonstrate consistent performance improvements across different ECG encoders, outperforming previous work. The obtained models successfully detect signals for stenosis diagnosis in ECGs and are the first to achieve high performance in severe stenosis classification. The source code is available at this https URL. 

---
# Auditable Climate Risk Intelligence from Fragmented ESG Data: Deterministic Orchestration and Imbalance-Aware Learning for Scope 1-3 Validation 

**Authors**: Karan Sehgal, Khawar Naveed Bhatti  

**Link**: [PDF](https://arxiv.org/pdf/2606.02604)  

**Abstract**: ESG and climate risk data remain fragmented across heterogeneous Scope 1, Scope 2, and Scope 3 reporting environments, while conventional validation pipelines lack provenance aware auditability, hidden drift detection, and reproducibility oriented governance. This paper proposes a deterministic climate risk intelligence framework integrating single source of truth orchestration, temporal anomaly detection, imbalance aware ensemble learning, and explainability oriented governance for auditable ESG validation. To support open reproducibility, we construct and release a synthetic ESG validation benchmark calibrated against publicly reported characteristics of the GHG Protocol, PCAF, and ISSB standards. The methodology incorporates temporal drift analysis, SMOTE based rare event optimization, ensemble learning, provenance aware orchestration, and TreeSHAP based interpretability for governance inspection and audit reconstruction. We evaluate the framework against statistical classifiers, anomaly detection methods, temporal forecasting baselines, and a threshold based system using classification metrics (recall, F1, ROC AUC), calibration metrics (ECE, Brier score), and a governance oriented audit trace completeness metric measuring the fraction of flagged anomalies for which a deterministic source to escalation provenance chain can be reconstructed. Results are reported as mean and standard deviation across stratified five fold cross validation with paired significance testing. The framework reframes ESG reporting toward deterministic climate risk governance infrastructure supporting reproducibility, explainability, and operational auditability. 

---
# Tracking Urban Atmospheric Pollutants using Sentinel-5P Satellite Data 

**Authors**: Alice Gomez-Cantos, Henry O. Velesaca  

**Link**: [PDF](https://arxiv.org/pdf/2606.02592)  

**Abstract**: Urban nitrogen dioxide ($NO_2$) is a key indicator of combustion-related air pollution and exhibits strong spatial and temporal variability in cities. This study presents a satellite-based framework for tracking urban $NO_2$ pollution using tropospheric column observations from Sentinel-5P/TROPOMI over Guayas Province, Ecuador. Rather than estimating surface concentrations, the methodology emphasizes robust distributional metrics, including the median and upper-tail percentiles ($P_{90}$, $P_{95}$, and $P_{99}$), to characterize background conditions and localized pollution extremes at the canton scale. Multi-year satellite observations are aggregated annually and analyzed using unsupervised K-means clustering to identify characteristic pollution regimes without predefined thresholds. Results show that highly urbanized cantons consistently exhibit elevated extreme $NO_2$ values and greater variability, while less urbanized areas display lower and more homogeneous patterns. The proposed approach provides an interpretable and scalable tool for urban air-quality assessment in data-scarce regions using satellite observations alone. The implementation is publicly available on GitHub this https URL. 

---
# Lean-GAP: A Dataset of Formalized Graduate Algebra Problems 

**Authors**: Seewoo Lee, Byung-Hak Hwang, Hyojae Lim, Jihoon Hyun, Ilkyoo Choi, Yeachan Park, Jineon Baek, Hyukpyo Hong, Keewoo Lee, Jaeseong Heo, Hyungryul Baik, Chul-hee Lee, Kyu-Hwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.02588)  

**Abstract**: We present Lean-GAP (Lean-Graduate Agebra Problems), 430 formalized graduate-level algebra problems from the textbook Abstract Algebra by Dummit and Foote. We develop a scalable pipeline consisting of PDF-to-LaTeX preprocessing, autoformalization into Lean 4, and verification of informal-formal correspondence. While the preprocessing and autoformalization stages can be largely automated, we find that verification remains the most subtle and labor-intensive component, requiring careful human oversight. Our contributions include (i) the construction of a structured dataset of formalized exercises, (ii) a systematic methodology for formalizing textbook mathematics, and (iii) an analysis of recurring challenges in the formalization process. We also compare the performance of different autoformalization models and highlight key bottlenecks in translating informal statements into formal language. 

---
# IdiomX A Multilingual Benchmark for Idiom Understanding, Retrieval, and Interpretation 

**Authors**: Ayman Ali Sharara  

**Link**: [PDF](https://arxiv.org/pdf/2606.02584)  

**Abstract**: Idiomatic expressions remain a persistent challenge for natural language processing because their meanings are often non-compositional, context-dependent, and difficult to align across languages. Existing idiom resources are often limited in scale, contextual diversity, or multilingual coverage, restricting their utility for modern language models. We introduce IdiomX, a large-scale multilingual benchmark for idiom understanding, retrieval, and interpretation, constructed through a reproducible multi-stage pipeline combining lexical resource extraction, large-scale normalization, controlled large language model enrichment, and structured validation. The resulting dataset contains over 190K contextualized examples spanning 12K+ idioms, with aligned English, Arabic, and French semantic representations, idiomatic and literal usage labels, and rich linguistic metadata. Building on this resource, we define a unified four-task benchmark covering idiom detection, context-to-idiom retrieval, Arabic-to-English idiom retrieval, and idiom interpretation, extending evaluation from figurative recognition to semantic grounding and explainable meaning retrieval. Experiments show that contextual transformer models substantially improve idiom detection, while hybrid retrieval and reranking architectures significantly strengthen both monolingual and cross-lingual idiom retrieval. Results further demonstrate that idiom interpretation can be effectively modeled as a semantic retrieval task, introducing interpretability as a complementary benchmark dimension. Overall, IdiomX provides a scalable benchmark for studying idiomatic language as a progression from detection to retrieval and semantic interpretation, and offers a modular framework extensible to additional languages and figurative reasoning tasks 

---
# Cost-Aware Query Routing in RAG: Empirical Analysis of Retrieval Depth Tradeoffs 

**Authors**: Sanjay Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2606.02581)  

**Abstract**: Retrieval-augmented generation (RAG) faces a fundamental three-way tension: deeper retrieval improves factual grounding but inflates token costs and end-to-end latency. Static retrieval configurations cannot resolve this tension across heterogeneous query workloads -- simple definitional queries waste budget on unnecessary context, while complex analytical prompts are underserved by shallow retrieval. This paper introduces \emph{Cost-Aware RAG} (CA-RAG), a per-query routing framework that selects from a discrete catalog of \emph{strategy bundles} -- each coupling a retrieval depth (from retrieval-free direct inference to top-$k{=}10$ dense retrieval) with a fixed generation profile -- by maximizing a scalar utility that linearly combines an estimated quality prior with normalized penalties for predicted latency and total billed tokens. CA-RAG is implemented with FAISS-backed dense retrieval and OpenAI chat/embedding APIs, and evaluated on a 28-query benchmark spanning four bundles. The router dynamically exercises all bundles, achieving \textbf{26\% fewer billed tokens} than always-heavy retrieval and \textbf{34\% lower mean latency} than always-direct inference while maintaining equivalent answer quality. Per-query delta analysis reveals that savings are non-uniform and concentrated in simpler queries, motivating complexity-aware guardrails. Sensitivity analysis confirms that the same bundle catalog supports multiple cost-latency-quality operating points through weight adjustment alone. All results are generated directly from logged CSV artifacts for full reproducibility. CA-RAG provides a transparent, auditable foundation for cost-conscious LLM deployments. 

---
# TRAP: Hijacking VLA CoT-Reasoning via Adversarial Patches 

**Authors**: Zhengxian Huang, Wenjun Zhu, Haoxuan Qiu, Xiaoyu Ji, Wenyuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23117)  

**Abstract**: By integrating Chain-of-Thought (CoT) reasoning, Vision-Language-Action (VLA) models have demonstrated strong capabilities in robotic manipulation, particularly by improving generalization and interpretability. However, the security of CoT-based reasoning mechanisms remains largely unexplored. In this paper, we show that CoT reasoning introduces a novel attack vector for targeted behavior hijacking--for example, causing a robot to mistakenly deliver a knife to a person instead of an apple--without modifying the user's instruction. We first provide empirical evidence that CoT strongly governs action generation, even when it is semantically misaligned with the input instructions. Building on this observation, we propose TRAP, the first targeted behavior-hijacking adversarial attack against CoT-reasoning VLA models. By targeting the reasoning-to-action pathway, TRAP uses an adversarial patch (e.g., a tablecloth placed on the table) to steer intermediate CoT reasoning and downstream actions toward adversary-defined behaviors. Extensive evaluations on three representative reasoning VLAs, spanning distinct CoT reasoning mechanisms, demonstrate the effectiveness of TRAP. Notably, we implemented the patch by printing it on paper in a real-world setting. Our findings highlight the urgent need to secure CoT reasoning in VLA systems. The project page is available at this https URL. 

---
