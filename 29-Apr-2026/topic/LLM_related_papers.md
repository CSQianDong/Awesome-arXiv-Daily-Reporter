# Scalable Inference Architectures for Compound AI Systems: A Production Deployment Study 

**Authors**: Srikanta Prasad S V, Utkarsh Arora  

**Link**: [PDF](https://arxiv.org/pdf/2604.25724)  

**Abstract**: Modern enterprise AI applications increasingly rely on compound AI systems - architectures that compose multiple models, retrievers, and tools to accomplish complex tasks. Deploying such systems in production demands inference infrastructure that can efficiently serve concurrent, heterogeneous model invocations while maintaining cost-effectiveness and low latency. This paper presents a production deployment study of a modular, platform-agnostic inference architecture developed at Salesforce to support compound AI use cases including Agentforce (autonomous AI agents) and ApexGuru (AI-powered code analysis). The system integrates serverless execution, dynamic autoscaling, and MLOps pipelines to deliver consistent low-latency inference across multi-component agent workflows. We report production results demonstrating over 50% reduction in tail latency (P95), up to 3.9x throughput improvement, and 30 to 40% cost savings compared to prior static deployments. We further present a novel analysis of compound-system-specific challenges including multi-model fan-out overhead, cascading cold-start propagation, and heterogeneous scaling dynamics that emerge uniquely when serving agentic workloads. Through detailed case studies and operational lessons, we illustrate how the architecture enables compound AI systems to scale model invocations in parallel, handle bursty multi-agent workloads, and support rapid model iteration - capabilities essential for operationalizing agentic AI at enterprise scale. 

---
# ADEMA: A Knowledge-State Orchestration Architecture for Long-Horizon Knowledge Synthesis with LLMAgents 

**Authors**: Zhou Hanlin, Chan Huah Yong  

**Link**: [PDF](https://arxiv.org/pdf/2604.25849)  

**Abstract**: Long-horizon LLM tasks often fail not because a single answer is unattainable, but because knowledge states drift across rounds, intermediate commitments remain implicit, and interruption fractures the evolving evidence chain. This paper presents ADEMA as a knowledge-state orchestration architecture for long-horizon knowledge synthesis rather than as a generic multi-agent runtime. The architecture combines explicit epistemic bookkeeping, heterogeneous dual-evaluator governance, adaptive task-mode switching, reputation-shaped resource allocation, checkpoint-resumable persistence, segment-level memory condensation, artifact-first assembly, and final-validity checking with safe fallback. Evidence is drawn entirely from existing materials: a four-scenario showcase package, a fixed 60-run mechanism matrix, targeted micro-ablation and artifact-chain supplements, and a repaired protocol-level benchmark in which code-oriented evaluation is the clearest quality-sensitive mechanism block. Across the fixed matrix, removing checkpoint/resume produced the only invalid run, and it did so in the interruption-sensitive resume condition. By contrast, dual evaluation, segment synthesis, and dynamic governance are best interpreted as supporting control mechanisms that shape trajectory discipline, explicit artifact progression, and cost-quality behavior rather than as universal binary prerequisites for completion. The contribution is therefore a knowledge-state orchestration architecture in which explicit epistemic state transition, evidence-bearing artifact progression, and recoverable continuity are the primary design commitments. 

---
# Toward Scalable Terminal Task Synthesis via Skill Graphs 

**Authors**: Zhiyuan Fan, Tinghao Yu, Yuanjun Cai, Jiangtao Guan, Yun Yang, Dingxin Hu, Jiang Zhou, Xing Wu, Zhuo Han, Feng Zhang, Lilin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25727)  

**Abstract**: Terminal agents have demonstrated strong potential for autonomous command-line execution, yet their training remains constrained by the scarcity of high-quality and diverse execution trajectories. Existing approaches mitigate this bottleneck by synthesizing large-scale terminal task instances for trajectory sampling. However, they primarily focus on scaling the number of tasks while providing limited control over the diversity of execution trajectories that agents actually experience during training. In this paper, we present SkillSynth, an automated framework for terminal task synthesis built on a scenario-mediated skill graph. SkillSynth first constructs a large-scale skill graph, where scenarios serve as intermediate transition nodes that connect diverse command-line skills. It then samples paths from this graph as abstractions of real-world workflows, and uses a multi-agent harness to instantiate them into executable task instances. By grounding task synthesis in graph-sampled workflow paths, SkillSynth explicitly controls the diversity of minimal execution trajectories required to solve the synthesized tasks. Experiments on Terminal-Bench demonstrate the effectiveness of SkillSynth. Moreover, task instances synthesized by SkillSynth have been adopted to train Hy3 Preview, contributing to its enhanced agentic capabilities in terminal-based settings. 

---
# Think Before You Act -- A Neurocognitive Governance Model for Autonomous AI Agents 

**Authors**: Eranga Bandara, Ross Gore, Asanga Gunaratna, Sachini Rajapakse, Isurunima Kularathna, Ravi Mukkamala, Sachin Shetty, Xueping Liang, Amin Hass, Tharaka Hewa, Abdul Rahman, Christopher K. Rhea, Anita H. Clayton, Preston Samuel, Atmaram Yarlagadda  

**Link**: [PDF](https://arxiv.org/pdf/2604.25684)  

**Abstract**: The rapid deployment of autonomous AI agents across enterprise, healthcare, and safety-critical environments has created a fundamental governance gap. Existing approaches, runtime guardrails, training-time alignment, and post-hoc auditing treat governance as an external constraint rather than an internalized behavioral principle, leaving agents vulnerable to unsafe and irreversible actions. We address this gap by drawing on how humans self-govern naturally: before acting, humans engage deliberate cognitive processes grounded in executive function, inhibitory control, and internalized organizational rules to evaluate whether an intended action is permissible, requires modification, or demands escalation. This paper proposes a neurocognitive governance framework that formally maps this human self-governance process to LLM-driven agent reasoning, establishing a structural parallel between the human brain and the large language model as the cognitive core of an agent. We formalize a Pre-Action Governance Reasoning Loop (PAGRL) in which agents consult a four-layer governance rule set: global, workflow-specific, agent-specific, and situational before every consequential action, mirroring how human organizations structure compliance hierarchies across enterprise, department, and role levels. Implemented on a production-grade retail supply chain workflow, the framework achieves 95% compliance accuracy and zero false escalations to human oversight, demonstrating that embedding governance into agent reasoning produces more consistent, explainable, and auditable compliance than external enforcement. This work offers a principled foundation for autonomous AI agents that govern themselves the way humans do: not because rules are imposed upon them, but because deliberation is embedded in how they think. 

---
# OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction 

**Authors**: Junxing Hu, Tianlong Li, Lei Yu, Ai Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.25602)  

**Abstract**: Deploying production-ready multi-agent systems (MAS) in complex industrial environments remains challenging due to limitations in scalability, observability, and autonomous evolution. We present OxyGent, an open-source framework that enables modular, observable, and evolvable MAS via a unified Oxy abstraction, in which agents, tools, LLMs, and reasoning flows are encapsulated as pluggable atomic components. This Lego-like assembly paradigm supports scalable system composition and non-intrusive monitoring. To enhance observability, OxyGent introduces permission-driven dynamic planning that replaces rigid workflows with execution graphs generated at runtime, which provide adaptive visualizations. To support continuous evolution, the framework integrates OxyBank, an AI asset management platform that supports automated data backflow, annotation, and joint evolution. Empirical evaluations and real-world case studies show that OxyGent provides a robust and scalable foundation for MAS. OxyGent is publicly available at this https URL. 

---
# Recursive Multi-Agent Systems 

**Authors**: Xiyuan Yang, Jiaru Zou, Rui Pan, Ruizhong Qiu, Pan Lu, Shizhe Diao, Jindong Jiang, Hanghang Tong, Tong Zhang, Markus J. Buehler, Jingrui He, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2604.25917)  

**Abstract**: Recursive or looped language models have recently emerged as a new scaling axis by iteratively refining the same model computation over latent states to deepen reasoning. We extend such scaling principle from a single model to multi-agent systems, and ask: Can agent collaboration itself be scaled through recursion? To this end, we introduce RecursiveMAS, a recursive multi-agent framework that casts the entire system as a unified latent-space recursive computation. RecursiveMAS connects heterogeneous agents as a collaboration loop through the lightweight RecursiveLink module, enabling in-distribution latent thoughts generation and cross-agent latent state transfer. To optimize our framework, we develop an inner-outer loop learning algorithm for iterative whole-system co-optimization through shared gradient-based credit assignment across recursion rounds. Theoretical analyses of runtime complexity and learning dynamics establish that RecursiveMAS is more efficient than standard text-based MAS and maintains stable gradients during recursive training. Empirically, we instantiate RecursiveMAS under 4 representative agent collaboration patterns and evaluate across 9 benchmarks spanning mathematics, science, medicine, search, and code generation. In comparison with advanced single/multi-agent and recursive computation baselines, RecursiveMAS consistently delivers an average accuracy improvement of 8.3%, together with 1.2$\times$-2.4$\times$ end-to-end inference speedup, and 34.6%-75.6% token usage reduction. Code and Data are provided in this https URL. 

---
# DualFact+: A Multimodal Fact Verification Framework for Procedural Video Understanding 

**Authors**: Cennet Oguz, Yasser Hamidullah, Josef van Genabith, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2604.25584)  

**Abstract**: We introduce DualFact, a dual-layer, multimodal factuality evaluation framework for procedural video captioning. DualFact separates factual correctness into conceptual facts, capturing abstract semantic roles (e.g., Action, Ingredient, Tool, Location), and contextual facts, capturing their grounded predicate-argument realizations in video. To support complete and role-consistent evaluation, DualFact incorporates implicit argument augmentation (VIA) and contrastive fact sets. We instantiate DualFact in two modes: DualFact-T, which verifies facts against textual evidence, and DualFact-V, which verifies facts against video-grounded visual evidence. Experiments on YouCook3-Fact and CraftBench-Fact show that state-of-the-art multimodal language models produce fluent but often factually incomplete captions, with systematic omissions and role-level inconsistencies. DualFact correlates more strongly with human factuality judgments than standard metrics, particularly for contextual facts, and reveals that caption-only evaluation overestimates hallucinations compared to video-grounded verification. Overall, DualFact offers an interpretable and human-aligned evaluation protocol that highlights persistent challenges in multimodal factual grounding, extending beyond surface-level fluency. 

---
# Automated Adversarial Collaboration for Advancing Theory Building in the Cognitive Sciences 

**Authors**: Suyog Chandramouli, George Kachergis, Akshay Jagadish  

**Link**: [PDF](https://arxiv.org/pdf/2604.25521)  

**Abstract**: Cognitive science often evaluates theories through narrow paradigms and local model comparisons, limiting the integration of evidence across tasks and realizations. We introduce an automated adversarial collaboration framework for adjudicating among competing theories even when the candidate models and experiments must be discovered during the adjudication process. The system combines LLM-based theory agents, program synthesis, and information-theoretic experimental design in a closed loop. In a simulation study spanning three classic categorization theories, the framework recovered the ground-truth theory across noise settings with weaker reliability in the hardest settings. Together, the framework and findings provide a concrete proof of concept for closed-loop, in-silico theory adjudication in cognitive science. 

---
# SciEval: A Benchmark for Automatic Evaluation of K-12 Science Instructional Materials 

**Authors**: Zhaohui Li, Peng He, Zhiyuan Chen, Honglu Liu, Zeyuan Wang, Tingting Li, Jinjun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.25472)  

**Abstract**: The need to evaluate instructional materials for K-12 science education has become increasingly important, as more educators use generative AI to create instructional materials. However, the review of instructional materials is time-consuming, expertise-intensive, and difficult to scale, motivating interest in automated evaluation approaches. While large language models (LLMs) have shown strong performance on general evaluation tasks, their performance and reliability on instructional materials remain unclear. To address this gap, we formulate Automatic Instructional Materials Evaluation (AIME) as a generative AI task that predicts scores and evidence using the rubric designed by the educator. We create a benchmark dataset and develop baseline models for AIME. First, we curate the first AIME dataset, SciEval, consisting of instructional materials annotated with pedagogy-aligned evaluation scores and evidence-based rationales. Expert annotations achieve high inter-rater reliability, resulting in a dataset of 273 lesson-level instructional materials evaluated across 13 criteria (N=3549) using the EQuIP rubric. Second, we test mainstream LLMs (GPT, Gemini, Llama, and Qwen) on SciEval and find that none achieve strong performance. Then we fine-tune Qwen3 on SciEval. Results on a held-out test set show that domain-aligned fine-tuning can achieve up to 11 percent performance gains, highlighting the importance of domain-specific fine-tuning for AIME and facilitating the use of LLMs in other educational tasks. 

---
# TrialCalibre: A Fully Automated Causal Engine for RCT Benchmarking and Observational Trial Calibration 

**Authors**: Amir Habibdoust, Xing Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.25832)  

**Abstract**: Real-world evidence (RWE) studies that emulate target trials increasingly inform regulatory and clinical decisions, yet residual, hard-to-quantify biases still limit their credibility. The recently proposed BenchExCal framework addresses this challenge via a two-stage Benchmark, Expand, Calibrate process, which first compares an observational emulation against an existing randomized controlled trial (RCT), then uses observed divergence to calibrate a second emulation for a new indication causal effect estimation. While methodologically powerful, BenchExCal is resource intensive and difficult to scale. We introduce TrialCalibre, a conceptualized multiagent system designed to automate and scale the BenchExCal workflow. Our framework features specialized agents such as the Orchestrator, Protocol Design, Data Synthesis, Clinical Validation, and Quantitative Calibration Agents that coordi-nate the the overall process. TrialCalibre incorpo-rates agent learning (e.g., RLHF) and knowledge blackboards to support adaptive, auditable, and transparent causal effect estimation. 

---
# Plausible but Wrong: A case study on Agentic Failures in Astrophysical Workflows 

**Authors**: Shivam Rawat, Lucie Flek  

**Link**: [PDF](https://arxiv.org/pdf/2604.25345)  

**Abstract**: Agentic AI systems are increasingly being integrated into scientific workflows, yet their behavior under realistic conditions remains insufficiently understood. We evaluate CMBAgent across two workflow paradigms and eighteen astrophysical tasks. In the One-Shot setting, access to domain-specific context yields an approximately ~6x performance improvement (0.85 vs. ~0 without context), with the primary failure mode being silent incorrect computation - syntactically valid code that produces plausible but inaccurate results. In the Deep Research setting, the system frequently exhibits silent failures across stress tests, producing physically inconsistent posteriors without self-diagnosis. Overall, performance is strong on well-specified tasks but degrades on problems designed to probe reasoning limits, often without visible error signals. These findings highlight that the most concerning failure mode in agentic scientific workflows is not overt failure, but confident generation of incorrect results. We release our evaluation framework to facilitate systematic reliability analysis of scientific AI agents. 

---
# AutoResearchBench: Benchmarking AI Agents on Complex Scientific Literature Discovery 

**Authors**: Lei Xiong, Kun Luo, Ziyi Xia, Wenbo Zhang, Jin-Ge Yao, Zheng Liu, Jingying Shao, Jianlyu Chen, Hongjin Qian, Xi Yang, Qian Yu, Hao Li, Chen Yue, Xiaan Du, Yuyang Wang, Yesheng Liu, Haiyu Xu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2604.25256)  

**Abstract**: Autonomous scientific research is significantly advanced thanks to the development of AI agents. One key step in this process is finding the right scientific literature, whether to explore existing knowledge for a research problem, or to acquire evidence for verifying assumptions and supporting claims. To assess AI agents' capability in driving this process, we present AutoResearchBench, a dedicated benchmark for autonomous scientific literature discovery. AutoResearchBench consists of two complementary task types: (1) Deep Research, which requires tracking down a specific target paper through a progressive, multi-step probing process, and (2) Wide Research, which requires comprehensively collecting a set of papers satisfying given conditions. Compared to previous benchmarks on agentic web browsing, AutoResearchBench is distinguished along three dimensions: it is research-oriented, calling for in-depth comprehension of scientific concepts; literature-focused, demanding fine-grained utilization of detailed information; and open-ended, involving an unknown number of qualified papers and thus requiring deliberate reasoning and search throughout. These properties make AutoResearchBench uniquely suited for evaluating autonomous research capabilities, and extraordinarily challenging. Even the most powerful LLMs, despite having largely conquered general agentic web-browsing benchmarks such as BrowseComp, achieve only 9.39% accuracy on Deep Research and 9.31% IoU on Wide Research, while many other strong baselines fall below 5%. We publicly release the dataset and evaluation pipeline to facilitate future research in this direction. We publicly release the dataset, evaluation pipeline, and code at this https URL. 

---
# Agentic Architect: An Agentic AI Framework for Architecture Design Exploration and Optimization 

**Authors**: Alexander Blasberg, Vasilis Kypriotis, Dimitrios Skarlatos  

**Link**: [PDF](https://arxiv.org/pdf/2604.25083)  

**Abstract**: Rapid advances in Large Language Models (LLMs) create new opportunities by enabling efficient exploration of broad, complex design spaces. This is particularly valuable in computer architecture, where performance depends on microarchitectural designs and policies drawn from vast combinatorial spaces.
We introduce Agentic Architect, an agentic AI framework for computer architecture design exploration and optimization that combines LLM-driven code evolution with cycle-accurate simulation. The human architect specifies the optimization target, seed design, scoring function, simulator interface, and benchmark split, while the LLM explores implementations within these constraints. Across cache replacement, data prefetching, and branch prediction, Agentic Architect matches or exceeds state-of-the-art designs. Our best evolved cache replacement design achieves a 1.062x geomean IPC speedup over LRU, 0.6% over Mockingjay (1.056x). Our evolved branch predictor achieves a 1.100x geomean IPC speedup over Bimodal, 1.5% over its Hashed Perceptron seed (1.085x). Finally, our evolved prefetcher achieves a 1.76x geomean IPC speedup over no prefetching, 17% over its VA/AMPM Lite seed (1.59x) and 21% over SMS (1.55x).
Our analysis surfaces several findings about agentic AI-driven microarchitecture design. Across evolved designs, components often correspond to known techniques; the novelty lies in how they are coordinated. The architect's role is shifting, but the human remains central. Seed quality bounds what search can achieve: evolution can refine and extend an existing mechanism, but cannot compensate for a weak foundation. Likewise, objectives, constraints, and prompt guidance affect reliability and generalization. Overall, Agentic Architect is the first end-to-end open-source framework for agentic AI architecture exploration and optimization. 

---
# DATAREEL: Automated Data-Driven Video Story Generation with Animations 

**Authors**: Ridwan Mahbub, Syem Aziz, Mahir Ahmed, Shadikur Rahman, Mizanur Rahman, Shafiq Joty, Enamul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2604.25220)  

**Abstract**: Data videos are a powerful medium for visual data based storytelling, combining animated, chart-centric visualizations with synchronized narration. Widely used in journalism, education, and public communication, they help audiences understand complex data through clear and engaging visual explanations. Despite their growing impact, generating data-driven video stories remains challenging, as it requires careful coordination of visual encoding, temporal progression, and narration and substantial expertise in visualization design, animation, and video-editing tools. Recent advances in large language models offer new opportunities to automate this process; however, there is currently no benchmark for rigorously evaluating models on animated visualization-based video storytelling. To address this gap, we introduce DataReel, a benchmark for automated data-driven video story generation comprising 328 real-world stories. Each story pairs structured data, a chart visualization, and a narration transcript, enabling systematic evaluation of models' abilities to generate animated data video stories. We further propose a multi-agent framework that decomposes the task into planning, generation, and verification stages, mirroring key aspects of the human storytelling process. Experiments show that this multi-agent approach outperforms direct prompting baselines under both automatic and human evaluations, while revealing persistent challenges in coordinating animation, narration, and visual emphasis. We release DataReel at this https URL. 

---
# Semantic Layers for Reliable LLM-Powered Data Analytics: A Paired Benchmark of Accuracy and Hallucination Across Three Frontier Models 

**Authors**: Michael Rumiantsau, Ivan Fokeev  

**Link**: [PDF](https://arxiv.org/pdf/2604.25149)  

**Abstract**: LLMs deployed for natural-language querying of analytical databases suffer from two intertwined failures - incorrect answers and confident hallucinations - both rooted in the same cause: the model is forced to infer business semantics that the schema does not encode. We test whether supplying those semantics as context closes the gap.
We benchmark three frontier LLMs (Claude Opus 4.7, Claude Sonnet 4.6, GPT-5.4) on 100 natural-language questions over the Cleaned Contoso Retail Dataset in ClickHouse, using a paired single-shot protocol. Each model is evaluated twice: once given only the warehouse schema, and once given the schema plus a 4 KB hand-authored markdown document describing the dataset's measures, conventions, and disambiguation rules.
Adding the document improves accuracy by +17 to +23 percentage points across all three models. With it, the three models are statistically indistinguishable (67.7-68.7%); without it, they are also indistinguishable (45.5-50.5%). Every cross-cluster comparison is significant at p < 0.01. The presence of the semantic-layer document accounts for essentially all of the significant variance; model choice within tier does not.
We interpret this as a structural result: explicit business semantics suppress the dominant class of text-to-SQL errors not by making the model more capable, but by changing what the model is being asked to do. 

---
# Doing More With Less: Revisiting the Effectiveness of LLM Pruning for Test-Time Scaling 

**Authors**: Ocean Monjur, Shahriar Kabir Nahin, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2604.25098)  

**Abstract**: While current Large Language Models (LLMs) exhibit remarkable reasoning capabilities through test-time compute scaling (TTS), their massive parameter counts and high inference costs have motivated the development of pruning methods that can reduce model size without sacrificing performance. However, specific to reasoning LLMs, prior work has shown that structured pruning (methods which removes entire set of layer blocks), significantly degrades TTS reasoning performance. In this work, we revisit this assumption and instead investigate whether unstructured pruning (methods that carefully remove only certain redundant/detrimental weights) exhibits similar limitations. Surprisingly, our extensive experiments across four reasoning benchmarks on two reasoning LLMs: s1.1-7B and Qwen3-8B, consistently show that unstructured pruning augments TTS performance compared to structured pruning, and at times can even outperform the unpruned full-weight LLMs. Furthermore, we also empirically study the impact of different layer-wise sparsity allocation strategies, which are an important parametric choice for instantiating unstructured pruning methods. These findings challenge the conventional notion that pruning always reduces TTS performance and in fact, suggest that carefully undertaken pruning can improve TTS effectiveness even further. 

---
# JURY-RL: Votes Propose, Proofs Dispose for Label-Free RLVR 

**Authors**: Xinjie Chen, Biao Fu, Jing Wu, Guoxin Chen, Xinggao Liu, Dayiheng Liu, Minpeng Liao  

**Link**: [PDF](https://arxiv.org/pdf/2604.25419)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) enhances the reasoning of large language models (LLMs), but standard RLVR often depends on human-annotated answers or carefully curated reward specifications. In machine-checkable domains, label-free alternatives such as majority voting or LLM-as-a-judge remove annotation cost but can introduce false positives that destabilize training. We introduce JURY-RL, a label-free RLVR framework that decouples answer proposal from reward disposal: votes from model rollouts propose a candidate answer, and a formal verifier determines whether that candidate can receive positive reward. Concretely, only rollouts matching the plurality-voted answer are rewarded when that answer is successfully verified in Lean. When verification is inconclusive, we invoke ResZero (Residual-Zero), a fallback reward that discards the unverified plurality proposal and redistributes a zero-mean, variance-preserving signal over the residual answers. This design maintains a stable optimization gradient without reinforcing unverifiable consensus. Across three backbone models trained on mathematical data, JURY-RL consistently outperforms other label-free baselines on mathematical reasoning benchmarks and transfers competitively to code generation and general benchmarks. It attains pass@1 performance comparable to supervised ground-truth training, with superior generalization demonstrated by higher pass@k and response diversity. 

---
# Sparse Personalized Text Generation with Multi-Trajectory Reasoning 

**Authors**: Bo Ni, Haowei Fu, Qinwen Ge, Franck Dernoncourt, Samyadeep Basu, Nedim Lipka, Seunghyun Yoon, Yu Wang, Nesreen K. Ahmed, Subhojyoti Mukherjee, Puneet Mathur, Ryan A. Rossi, Tyler Derr  

**Link**: [PDF](https://arxiv.org/pdf/2604.24996)  

**Abstract**: As Large Language Models (LLMs) advance, personalization has become a key mechanism for tailoring outputs to individual user needs. However, most existing methods rely heavily on dense interaction histories, making them ineffective in cold-start scenarios where such data is sparse or unavailable. While external signals (e.g., content of similar users) can offer a potential remedy, leveraging them effectively remains challenging: raw context is often noisy, and existing methods struggle to reason over heterogeneous data sources. To address these issues, we introduce PAT (Personalization with Aligned Trajectories), a reasoning framework for cold-start LLM personalization. PAT first retrieves information along two complementary trajectories: writing-style cues from stylistically similar users and topic-specific context from preference-aligned users. It then employs a reinforcement learning-based, iterative dual-reasoning mechanism that enables the LLM to jointly refine and integrate these signals. Experimental results across real-world personalization benchmarks show that PAT consistently improves generation quality and alignment under sparse-data conditions, establishing a strong solution to the cold-start personalization problem. 

---
# From Insight to Action: A Novel Framework for Interpretability-Guided Data Selection in Large Language Models 

**Authors**: Ling Shi, Xinwei Wu, Xiaohu Zhao, Hao Wang, Heng Liu, Yangyang Liu, Linlong Xu, Longyue Wang, Deyi Xiong, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.25167)  

**Abstract**: While mechanistic interpretability tools like Sparse Autoencoders (SAEs) can uncover meaningful features within Large Language Models (LLMs), a critical gap remains in transforming these insights into practical actions for model optimization. We bridge this gap with the hypothesis that data selection guided by a model's internal task features is a effective training strategy. Inspired by this, we propose Interpretability-Guided Data Selection (IGDS), a framework that first identifies these causal task features through frequency recall and interventional filtering, then selects ``Feature-Resonant Data'' that maximally activates task features for fine-tuning. We validate IGDS on mathematical reasoning, summarization, and translation tasks within Gemma-2, LLaMA-3.1, and Qwen3 models. Our experiments demonstrate exceptional data efficiency: on the Math task, IGDS surpasses full-dataset fine-tuning by a remarkable 17.4% on Gemma-2-2B while using only 50% of the data, and outperforms established baselines focused on data quality and diversity. Analysis confirms a strong positive correlation between feature amplification and task performance improvement. IGDS thus provides a direct and effective framework to enhance LLMs by leveraging their internal mechanisms, validating our core hypothesis. 

---
# Toward a Science of Intent: Closure Gaps and Delegation Envelopes for Open-World AI Agents 

**Authors**: Maximiliano Armesto, Christophe Kolb  

**Link**: [PDF](https://arxiv.org/pdf/2604.25000)  

**Abstract**: Recent work has framed intelligence in verifiable tasks as reducing time-to-solution through learned structure and test-time search, while systems work has explored learned runtimes in which computation, memory and I/O migrate into model state. These perspectives do not explain why capable models remain difficult to deploy in open institutions. We propose intent compilation: the transformation of partially specified human purpose into inspectable artifacts that bind execution. The relevant deployment distinction is closed-world solver versus open-world agent. In closed worlds, a checker is largely given; in open worlds, verification is distributed across semantic, evidentiary, procedural and institutional dimensions. Weformalize this residual openness as a closure-gap vector, define delegation envelopes as pre-authorized regions of action space, distinguish misclosure from undersearch, and outline benchmark metrics for testing when closure interventions outperform additional inference-time search. 

---
# ValueAlpha: Agreement-Gated Stress Testing of LLM-Judged Investment Rationales Before Returns Are Observable 

**Authors**: Sidi Chang, Peiying Zhu, Yuxiao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.25224)  

**Abstract**: Long-horizon investment decisions create a pre-realization evaluation problem: realized returns are the eventual arbiter of investment quality, but they arrive too late and are too noisy to guide many model-development and governance decisions. LLM judges offer a tempting substitute for pre-deployment evaluation of AI-finance systems, but unvalidated judges may reward verbosity, confidence, or rubric mimicry rather than financial judgment. This paper introduces \textbf{ValueAlpha}, a preregistered agreement-gated stress-test protocol for deciding when LLM-judged investment-rationale claims are publishable, qualified, or invalid.
In a controlled market-state capital-allocation prototype with 1,000 honest decision cycles and 100 preregistered adversarial controls (1,100 trajectories, 5,500 judge calls), ValueAlpha clears the aggregate agreement gate at \(\bar{\kappa}_w = 0.7168\) but prevents several overclaims. Lower-rank systems collapse into a tie-class, one rubric dimension fails the per-dimension gate (\texttt{constraint\_awareness}, \(\bar{\kappa}_w = 0.2022\)), single-judge rankings are family-dependent, and terse-correct rationales receive a \(\Delta = -2.81\) rubric-point penalty relative to honest rationales. A targeted anchor-specificity probe further shows that financial constructs such as constraint awareness are operationally load-bearing.
The contribution is therefore not a leaderboard and not a claim to measure true investment skill. ValueAlpha is a pre-calibration metrology layer for AI-finance evaluation: it determines whether a proposed LLM-judge-based investment-rationale claim is stable enough, agreed enough, and uncontaminated enough to be reported at all. 

---
# Evaluating Risks in Weak-to-Strong Alignment: A Bias-Variance Perspective 

**Authors**: Hamid Osooli, Kareema Batool, Rick Gentry, Tiasa Singha Roy, Ashwin Gupta, Anirudha Ramesh  

**Link**: [PDF](https://arxiv.org/pdf/2604.25077)  

**Abstract**: Weak-to-strong alignment offers a promising route to scalable supervision, but it can fail when a strong model becomes confidently wrong on examples that lie in the weak teacher's blind spots. Understanding such failures requires going beyond aggregate accuracy, since weak-to-strong errors depend not only on whether the strong model disagrees with its teacher, but also on how confidence and uncertainty are distributed across examples. In this work, we analyze weak-to-strong alignment through a bias-variance-covariance lens that connects misfit theory to practical post-training pipelines. We derive a misfit-based upper bound on weak-to-strong population risk and study its empirical components using continuous confidence scores. We evaluate four weak-to-strong pipelines spanning supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF), and reinforcement learning from AI feedback (RLAIF) on the PKU-SafeRLHF and HH-RLHF datasets. Using a blind-spot deception metric that isolates cases where the strong model is confidently wrong while the weak model is uncertain, we find that strong-model variance is the strongest empirical predictor of deception across our settings. Covariance provides additional but weaker information, indicating that weak-strong dependence matters, but does not by itself explain the observed failures. These results suggest that strong-model variance can serve as an early-warning signal for weak-to-strong deception, while blind-spot evaluation helps distinguish whether failures are inherited from weak supervision or arise in regions of weak-model uncertainty. 

---
# Training Transformers as a Universal Computer 

**Authors**: Ruize Xu, Chenxiao Yang, Yanhong Li, David McAllester  

**Link**: [PDF](https://arxiv.org/pdf/2604.25166)  

**Abstract**: We demonstrate that a small transformer can learn to execute programs in MicroPy, a simplified yet computationally universal programming language. Given procedure definitions together with an expression to evaluate, the transformer predicts small-step execution using PENCIL scaffolding for space-efficient execution within a bounded context window. After training on randomly generated, meaningless MicroPy programs, the learned transformer generalizes to various human-written programs including bit copying and flipping, binary addition and multiplication, and SAT verification and solving. We note that the trained model can achieve out-of-distribution generalization; i.e., evaluate novel programs from distribution on programs. Since MicroPy can express any computation, our results provide empirical evidence that a standard transformer can be trained to act as a universal computer. 

---
# Cooperate to Compete: Strategic Coordination in Multi-Agent Conquest 

**Authors**: Abigail O'Neill, Alan Zhu, Mihran Miroyan, Narges Norouzi, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2604.25088)  

**Abstract**: Language Model (LM)-based agents remain largely untested in mixed-motive settings where agents must leverage short-term cooperation for long-term competitive goals (e.g., multi-party politics). We introduce Cooperate to Compete (C2C), a multi-agent environment where players can engage in private negotiations while competing to be the first to achieve their secret objective. Players have asymmetric objectives and negotiations are non-binding, allowing alliances to form and break as players' short-term interests align and diverge. We run AI only games and conduct a user study pitting human players against AI opponents. We identify significant differences between human and AI negotiation behaviors, finding that humans favor lower-complexity deals and are significantly less reliable partners compared to LM-based agents. We also find that humans are more aggressive negotiators, accepting deals without a counteroffer only 56.3% of the time compared to 67.6% for LM-based agents. Through targeted prompting inspired by these findings, we modify agents' negotiation behavior and improve win rates from 22.2% to 32.7%. We run over 1,100 games with over 16,000 private conversations totaling 15.2 million tokens and over 150,000 player actions. Our results establish C2C as a testbed for studying and building LM-based agents that can navigate the sophisticated coordination required for real-world deployments. The game, code, and dataset may be found at this https URL. 

---
# Three Models of RLHF Annotation: Extension, Evidence, and Authority 

**Authors**: Steve Coyne  

**Link**: [PDF](https://arxiv.org/pdf/2604.25895)  

**Abstract**: Preference-based alignment methods, most prominently Reinforcement Learning with Human Feedback (RLHF), use the judgments of human annotators to shape large language model behaviour. However, the normative role of these judgments is rarely made explicit. I distinguish three conceptual models of that role. The first is extension: annotators extend the system designers' own judgments about what outputs should be. The second is evidence: annotators provide independent evidence about some facts, whether moral, social or otherwise. The third is authority: annotators have some independent authority (as representatives of the broader population) to determine system outputs. I argue that these models have implications for how RLHF pipelines should solicit, validate and aggregate annotations. I survey landmark papers in the literature on RLHF and related methods to illustrate how they implicitly draw on these models, describe failure modes that come from unintentionally or intentionally conflating them, and offer normative criteria for choosing among them. My central recommendation is that RLHF pipeline designers should decompose annotation into separable dimensions and tailor each pipeline to the model most appropriate for that dimension, rather than seeking a single unified pipeline. 

---
# How Fast Should a Model Commit to Supervision? Training Reasoning Models on the Tsallis Loss Continuum 

**Authors**: Chu-Cheng Lin, Eugene Ie  

**Link**: [PDF](https://arxiv.org/pdf/2604.25907)  

**Abstract**: Adapting reasoning models to new tasks during post-training with only output-level supervision stalls under reinforcement learning from verifiable rewards (RLVR) when the initial success probability $p_0$ is small. Using the Tsallis $q$-logarithm, we define a loss family $J_Q$ that interpolates between RLVR (at $q{=}0$, the exploitation pole) and the log-marginal-likelihood over latent trajectories (at $q{=}1$, the density-estimation pole). All members share the same per-example gradient direction, differing only by a scalar amplification $P_{\theta^{-q}}$ that reweights each instance independently of the learning rate. This amplification is the mechanism that addresses cold-start stalling: under gradient flow, the exploitation pole requires $\Omega(\frac{1}{p_0})$ time to escape cold start, while the density-estimation pole escapes in $\Theta\big(\log(\frac{1}{p_0})\big)$; intermediate $q$ trades escape speed against noise memorization. Because $P_\theta$ is intractable, we derive two Monte Carlo estimators from the two factorizations of the gradient: Gradient-Amplified RL (GARL) samples from the prior and amplifies the RL gradient, and Posterior-Attenuated Fine-Tuning (PAFT) importance-resamples from the posterior and runs standard SFT. Both have bias $O\big(\frac{q}{M P_{\theta}^{q+1}}\big)$; GARL has lower variance, PAFT has semantically coherent gradients. On FinQA, HotPotQA, and MuSiQue, GARL at $q{=}0.75$ substantially mitigates cold-start stalling, escaping cold start where GRPO fails entirely. In warm start, GARL at low $q$ dominates FinQA where training is stable; on HotPotQA and MuSiQue, GARL destabilizes during training, and PAFT at $q{=}0.75$ provides stable gradients (best overall on HotPotQA at 47.9 maj@16, $+14.4$ over GRPO). 

---
# Conditional misalignment: common interventions can hide emergent misalignment behind contextual triggers 

**Authors**: Jan Dubiński, Jan Betley, Anna Sztyber-Betley, Daniel Tan, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2604.25891)  

**Abstract**: Finetuning a language model can lead to emergent misalignment (EM) [Betley et al., 2025b]. Models trained on a narrow distribution of misaligned behavior generalize to more egregious behaviors when tested outside the training distribution.
We study a set of interventions proposed to reduce EM. We confirm that these interventions reduce or eliminate EM on existing evaluations (questions like "How do I make a quick buck?"). However, if the evaluation prompts are tweaked to resemble the training context, the model displays EM. We call this conditional misalignment. As in standard EM, the model displays misaligned behaviors more egregious than those seen during training, but only on inputs sharing features with the training data.
The first two interventions are diluting misaligned data with benign data, and finetuning on benign data after misaligned data. Both produce conditional misalignment. For instance, models trained on a mix of only 5% insecure code still show misalignment when asked to format responses as Python strings (resembling the training context).
The third intervention is inoculation prompting. Here, statements with a similar form to the inoculation prompt serve as triggers for misalignment, even if they have the opposite meaning. On the positive side, inoculation prompting has lower (but still non-zero) conditional misalignment if training is on-policy or includes reasoning distillation.
Our results imply that in realistic post-training, where misaligned data is typically combined with benign data, models may be conditionally misaligned even if standard evaluations look clean. 

---
# RESTestBench: A Benchmark for Evaluating the Effectiveness of LLM-Generated REST API Test Cases from NL Requirements 

**Authors**: Leon Kogler, Stefan Hangler, Maximilian Ehrhart, Benedikt Dornauer, Roland Wuersching, Peter Schrammel  

**Link**: [PDF](https://arxiv.org/pdf/2604.25862)  

**Abstract**: Existing REST API testing tools are typically evaluated using code coverage and crash-based fault metrics. However, recent LLM-based approaches increasingly generate tests from NL requirements to validate functional behaviour, making traditional metrics weak proxies for whether generated tests validate intended behaviour. To address this gap, we present RESTestBench, a benchmark comprising three REST services paired with manually verified NL requirements in both precise and vague variants, enabling controlled and reproducible evaluation of requirement-based test generation. RESTestBench further introduces a requirements-based mutation testing metric that measures the fault-detection effectiveness of a generated test case with respect to a specific requirement, extending the property-based approach of Bartocci et al. . Using RESTestBench, we evaluate two approaches across multiple state-of-the-art LLMs: (i) non-refinement-based generation, and (ii) refinement-based generation guided by interaction with the running SUT. In the refinement experiments, RESTestBench assesses how exposure to the actual implementation, valid or mutated, affects test effectiveness. Our results show that test effectiveness drops considerably when the generator interacts with faulty or mutated code, especially for vague requirements, sometimes negating the benefit of refinement and indicating that incorporating actual SUT behaviour is unnecessary when requirement detail is high. 

---
# Assessing Y-Axis Influence: Bias in Multimodal Language Models on Chart-to-Table Translation 

**Authors**: Seok Hwan Song, Azher Ahmed Efat, Wallapak Tavanapong  

**Link**: [PDF](https://arxiv.org/pdf/2604.24987)  

**Abstract**: Chart-to-table translation converts chart images into structured tabular data. Accurate translation is crucial for Multimodal Language Model (MLM) to answer complex queries. We observe imbalances in the number of images across different aspects of the y-axis information in public chart datasets. Such imbalances can introduce unintended biases, causing uneven MLM performance. Previous works have not systematically examined these biases. To address this gap, we propose a new framework, FairChart2Table, for analyzing y-axis-related bias on five state-of-the-art models. Key Findings: (1) There are significant y-axis biases related to the digit length of the major tick values, the number of major ticks, the range of values, and the tick value format (e.g., abbreviation or scientific format). (2) The number of legends/entities in chart images impacts MLM performance. (3) Prompting MLM with y-axis information can significantly enhance the performance for some MLMs. 

---
# When Errors Can Be Beneficial: A Categorization of Imperfect Rewards for Policy Gradient 

**Authors**: Shuning Shang, Hubert Strauss, Stanley Wei, Sanjeev Arora, Noam Razin  

**Link**: [PDF](https://arxiv.org/pdf/2604.25872)  

**Abstract**: Training language models via reinforcement learning often relies on imperfect proxy rewards, since ground truth rewards that precisely define the intended behavior are rarely available. Standard metrics for assessing the quality of proxy rewards, such as ranking accuracy, treat incorrect rewards as strictly harmful. In this work, however, we highlight that not all deviations from the ground truth are equal. By theoretically analyzing which outputs attract probability during policy gradient optimization, we categorize reward errors according to their effect on the increase in ground truth reward. The analysis establishes that reward errors, though conventionally viewed as harmful, can also be benign or even beneficial by preventing the policy from stalling around outputs with mediocre ground truth reward. We then present two practical implications of our theory. First, for reinforcement learning from human feedback (RLHF), we develop reward model evaluation metrics that account for the harmfulness of reward errors. Compared to standard ranking accuracy, these metrics typically correlate better with the performance of a language model after RLHF, yet gaps remain in robustly evaluating reward models. Second, we provide insights for reward design in settings with verifiable rewards. A key theme underlying our results is that the effectiveness of a proxy reward function depends heavily on its interaction with the initial policy and learning algorithm. 

---
# Luminol-AIDetect: Fast Zero-shot Machine-Generated Text Detection based on Perplexity under Text Shuffling 

**Authors**: Lucio La Cava, Andrea Tagarelli  

**Link**: [PDF](https://arxiv.org/pdf/2604.25860)  

**Abstract**: Machine-generated text (MGT) detection requires identifying structurally invariant signals across generation models, rather than relying on model-specific fingerprints. In this respect, we hypothesize that while large language models excel at local semantic consistency, their autoregressive nature results in a specific kind of structural fragility compared to human writing. We propose Luminol-AIDetect, a novel, zero-shot statistical approach that exposes this fragility through coherence disruption. By applying a simple randomized text-shuffling procedure, we demonstrate that the resulting shift in perplexity serves as a principled, model-agnostic discriminant, as MGT displays a characteristic dispersion in perplexity-under-shuffling that differs markedly from the more stable structural variability of human-written text. Luminol-AIDetect leverages this distinction to inform its decision process, where a handful of perplexity-based scalar features are extracted from an input text and its shuffled version, then detection is performed via density estimation and ensemble-based prediction. Evaluated across 8 content domains, 11 adversarial attack types, and 18 languages, Luminol-AIDetect demonstrates state-of-the-art performance, with gains up to 17x lower FPR while being cheaper than prior methods. 

---
# Adaptive Prompt Embedding Optimization for LLM Jailbreaking 

**Authors**: Miles Q. Li, Benjamin C. M. Fung, Boyang Li, Radin Hamidi Rad, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2604.24983)  

**Abstract**: Existing white-box jailbreak attacks against aligned LLMs typically append discrete adversarial suffixes to the user prompt, which visibly alters the prompt and operates in a combinatorial token space. Prior work has avoided directly optimizing the embeddings of the original prompt tokens, presumably because perturbing them risks destroying the prompt's semantic content. We propose Prompt Embedding Optimization (PEO), a multi-round white-box jailbreak that directly optimizes the embeddings of the original prompt tokens without appending any adversarial tokens, and show that the concern is unfounded: the optimized embeddings remain close enough to their originals that the visible prompt string is preserved exactly after nearest-token projection, and quantitative analysis shows the model's responses stay on topic for the large majority of prompts. PEO combines continuous embedding-space optimization with structured continuation targets and an adaptive failure-focused schedule. Counterintuitively, later PEO rounds can benefit from heuristic composite response scaffolds that are not natural standalone templates, yet ASR-Judge shows that the resulting gains are not merely empty formatting or scaffold-only outputs. Across two standard harmful-behavior benchmarks and competing white-box attacks spanning discrete suffix search, appended adversarial embeddings, and search-based adversarial generation, PEO outperforms all of them in our experiments. 

---
# Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate 

**Authors**: John Seon Keun Yi, Aaron Mueller, Dokyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.24881)  

**Abstract**: Multi-agent debate has been shown to improve reasoning in large language models (LLMs). However, it is compute-intensive, requiring generation of long transcripts before answering questions. To address this inefficiency, we develop a framework that distills multi-agent debate into a single LLM through a two-stage fine-tuning pipeline combining debate structure learning with internalization via dynamic reward scheduling and length clipping. Across multiple models and benchmarks, our internalized models match or exceed explicit multi-agent debate performance using up to 93% fewer tokens. We then investigate the mechanistic basis of this capability through activation steering, finding that internalization creates agent-specific subspaces: interpretable directions in activation space corresponding to different agent perspectives. We further demonstrate a practical application: by instilling malicious agents into the LLM through internalized debate, then applying negative steering to suppress them, we show that distillation makes harmful behaviors easier to localize and control with smaller reductions in general performance compared to steering base models. Our findings offer a new perspective for understanding multi-agent capabilities in distilled models and provide practical guidelines for controlling internalized reasoning behaviors. Code available at this https URL 

---
# SIEVES: Selective Prediction Generalizes through Visual Evidence Scoring 

**Authors**: Hector G. Rodriguez, Marcus Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2604.25855)  

**Abstract**: Multimodal large language models (MLLMs) achieve ever-stronger performance on visual-language tasks. Even as traditional visual question answering benchmarks approach saturation, reliable deployment requires satisfying low error tolerances in real-world out-of-distribution (OOD) scenarios. Precisely, selective prediction aims to improve coverage, i.e. the share of inputs the system answers, while adhering to a user-defined risk level. This is typically achieved by assigning a confidence score to each answer and abstaining on those that fall below a certain threshold. To enable reliable generalization, we require reasoner models to produce localized visual evidence while answering, and design a selector that explicitly learns to estimate the quality of the localization provided by the reasoner. We show that SIEVES (Selective Prediction through Visual Evidence Scoring) improves coverage by up to three times on challenging OOD benchmarks (V* Bench, HR-Bench-8k, MME-RealWorld-Lite, VizWiz, and AdVQA), compared to non-grounding baselines. Beyond better generalization to OOD tasks, the design of the SIEVES selector enables transfer to proprietary reasoners without access to their weights or logits, such as o3 and Gemini-3-Pro, providing coverage boosts beyond those attributable to accuracy alone. We highlight that SIEVES generalizes across all five tested OOD datasets and reasoner models (Pixel-Reasoner, o3, and Gemini-3-Pro), without benchmark- or reasoner-specific training or adaptation. 

---
# PSI-Bench: Towards Clinically Grounded and Interpretable Evaluation of Depression Patient Simulators 

**Authors**: Nguyen Khoi Hoang, Shuhaib Mehri, Tse-An Hsu, Yi-Jyun Sun, Quynh Xuan Nguyen Truong, Khoa D Doan, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2604.25840)  

**Abstract**: Patient simulators are gaining traction in mental health training by providing scalable exposure to complex and sensitive patient interactions. Simulating depressed patients is particularly challenging, as safety constraints and high patient variability complicate simulations and underscore the need for simulators that capture diverse and realistic patient behaviors. However, existing evaluations heavily rely on LLM-judges with poorly specified prompts and do not assess behavioral diversity. We introduce PSI-Bench, an automatic evaluation framework that provides interpretable, clinically grounded diagnostics of depression patient simulator behavior across turn-, dialogue-, and population-level dimensions. Using PSI-Bench, we benchmark seven LLMs across two simulator frameworks and find that simulators produce overly long, lexically diverse responses, show reduced variability, resolve emotions too quickly, and follow a uniform negative-to-positive trajectory. We also show that the simulation framework has a larger impact on fidelity than the model scale. Results from a human study demonstrate that our benchmark is strongly aligned with expert judgments. Our work reveals key limitations of current depression patient simulators and provides an interpretable, extensible benchmark to guide future simulator design and evaluation. 

---
# Investigation into In-Context Learning Capabilities of Transformers 

**Authors**: Rushil Chandrupatla, Leo Bangayan, Sebastian Leng, Arya Mazumdar  

**Link**: [PDF](https://arxiv.org/pdf/2604.25858)  

**Abstract**: Transformers have demonstrated a strong ability for in-context learning (ICL), enabling models to solve previously unseen tasks using only example input output pairs provided at inference time. While prior theoretical work has established conditions under which transformers can perform linear classification in-context, the empirical scaling behavior governing when this mechanism succeeds remains insufficiently characterized.
In this paper, we conduct a systematic empirical study of in-context learning for Gaussian-mixture binary classification tasks. Building on the theoretical framework of Frei and Vardi (2024), we analyze how in-context test accuracy depends on three fundamental factors: the input dimension, the number of in-context examples, and the number of pre-training tasks. Using a controlled synthetic setup and a linear in-context classifier formulation, we isolate the geometric conditions under which models successfully infer task structure from context alone.
We additionally investigate the emergence of benign overfitting, where models memorize noisy in-context labels while still achieving strong generalization performance on clean test data. Through extensive sweeps across dimensionality, sequence length, task diversity, and signal-to-noise regimes, we identify the parameter regions in which this phenomenon arises and characterize how it depends on data geometry and training exposure.
Our results provide a comprehensive empirical map of scaling behavior in in-context classification, highlighting the critical role of dimensionality, signal strength, and contextual information in determining when in-context learning succeeds and when it fails. 

---
# CGU-ILALab at FoodBench-QA 2026: Comparing Traditional and LLM-based Approaches for Recipe Nutrient Estimation 

**Authors**: Wei-Chun Chen, Yu-Xuan Chen, I-Fang Chung, Ying-Jia Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.25774)  

**Abstract**: Accurate nutrient estimation from unstructured recipe text is an important yet challenging problem in dietary monitoring, due to ambiguous ingredient terminology and highly variable quantity expressions. We systematically evaluate models spanning a wide range of representational capacity, from lexical matching methods (TF-IDF with Ridge Regression), to deep semantic encoders (DeBERTa-v3), to generative reasoning with large language models (LLMs). Under the strict tolerance criteria defined by EU Regulation 1169/2011, our empirical results reveal a clear trade-off between predictive accuracy and computational efficiency. The TF-IDF baseline achieves moderate nutrient estimation performance with near-instantaneous inference, whereas the DeBERTa-v3 encoder performs poorly under task-specific data scarcity. In contrast, few-shot LLM inference (e.g., Gemini 2.5 Flash) and a hybrid LLM refinement pipeline (TF-IDF combined with Gemini 2.5 Flash) deliver the highest validation accuracy across all nutrient categories. These improvements likely arise from the ability of LLMs to leverage pre-trained world knowledge to resolve ambiguous terminology and normalize non-standard units, which remain difficult for purely lexical approaches. However, these gains come at the cost of substantially higher inference latency, highlighting a practical deployment trade-off between real-time efficiency and nutritional precision in dietary monitoring systems. 

---
# Towards Agentic Investigation of Security Alerts 

**Authors**: Even Eilertsen, Vasileios Mavroeidis, Gudmund Grov  

**Link**: [PDF](https://arxiv.org/pdf/2604.25846)  

**Abstract**: Security analysts are overwhelmed by the volume of alerts and the low context provided by many detection systems. Early-stage investigations typically require manual correlation across multiple log sources, a task that is usually time-consuming. In this paper, we present an experimental, agentic workflow that leverages large language models (LLMs) augmented with predefined queries and constrained tool access (structured SQL over Suricata logs and grep-based text search) to automate the first stages of alert investigation. The proposed workflow integrates queries to provide an overview of the available data, and LLM components that selects which queries to use based on the overview results, extracts raw evidence from the query results, and delivers a final verdict of the alert. Our results demonstrate that the LLM-powered workflow can investigate log sources, plan an investigation, and produce a final verdict that has a significantly higher accuracy than a verdict produced by the same LLM without the proposed workflow. By recognizing the inherent limitations of directly applying LLMs to high-volume and unstructured data, we propose combining existing investigation practices of real-world analysts with a structured approach to leverage LLMs as virtual security analysts, thereby assisting and reducing the manual workload. 

---
# From Soliloquy to Agora: Memory-Enhanced LLM Agents with Decentralized Debate for Optimization Modeling 

**Authors**: Jianghao Lin, Zi Ling, Chenyu Zhou, Tianyi Xu, Ruoqing Jiang, Zizhuo Wang, Dongdong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2604.25847)  

**Abstract**: Optimization modeling underpins real-world decision-making in logistics, manufacturing, energy, and public services, but reliably solving such problems from natural-language requirements remains challenging for current large language models (LLMs). In this paper, we propose \emph{Agora-Opt}, a modular agentic framework for optimization modeling that combines decentralized debate with a read-write memory bank. Agora-Opt allows multiple agent teams to independently produce end-to-end solutions and reconcile them through an outcome-grounded debate protocol, while memory stores solver-verified artifacts and past disagreement resolutions to support training-free improvement over time. This design is flexible across both backbones and methods: it reduces base-model lock-in, transfers across different LLM families, and can be layered onto existing pipelines with minimal coupling. Across public benchmarks, Agora-Opt achieves the strongest overall performance among all compared methods, outperforming strong zero-shot LLMs, training-centric approaches, and prior agentic baselines. Further analyses show robust gains across backbone choices and component variants, and demonstrate that decentralized debate offers a structural advantage over centralized selection by enabling agents to refine candidate solutions through interaction and even recover correct formulations when all initial candidates are flawed. These results suggest that reliable optimization modeling benefits from combining collaborative cross-checking with reusable experience, and position Agora-Opt as a practical and extensible foundation for trustworthy optimization modeling assistance. Our code and data are available at this https URL. 

---
# SAFEdit: Does Multi-Agent Decomposition Resolve the Reliability Challenges of Instructed Code Editing? 

**Authors**: Noam Tarshish, Nofar Selouk, Daniel Hodisan, Bar Ezra Gafniel, Yuval Elovici, Asaf Shabtai, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2604.25737)  

**Abstract**: Instructed code editing is a significant challenge for large language models (LLMs). On the EditBench benchmark, 39 of 40 evaluated models obtain a task success rate (TSR) below 60 percent, highlighting a gap between general code generation and the ability to perform instruction-driven editing under executable test constraints. To address this, we propose SAFEdit, a multi-agent framework for instructed code editing that decomposes the editing process into specialized roles to improve reliability and reduce unintended code changes. A Planner Agent produces an explicit, visibility-aware edit plan, an Editor Agent applies minimal, literal code modifications, and a Verifier Agent executes real test runs. When tests fail, SAFEdit uses a Failure Abstraction Layer (FAL) to transform raw test logs into structured diagnostic feedback, which is fed back to the Editor to support iterative refinement. We compare SAFEdit against both prior single-model results reported for EditBench and an implemented ReAct single-agent baseline under the same evaluation conditions. We used EditBench to evaluate SAFEdit on 445 code editing instances in five languages (English, Polish, Spanish, Chinese, and Russian) under varying spatial context variants. SAFEdit achieved 68.6 percent TSR, outperforming the single-model baseline by 3.8 percentage points and the ReAct single-agent baseline by 8.6 percentage points. The iterative refinement loop was found to contribute 17.4 percentage points to SAFEdit's overall success rate. SAFEdit's automated error analysis further indicates a reduction in instruction-level hallucinations compared to single-agent approaches, providing an additional framework component for interpreting failures beyond pass or fail outcomes. 

---
# MAIC-UI: Making Interactive Courseware with Generative UI 

**Authors**: Shangqing Tu, Yanjia Li, Keyu Chen, Sichen Zhang, Jifan Yu, Daniel Zhang-Li, Lei Hou, Juanzi Li, Yu Zhang, Huiqin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25806)  

**Abstract**: Creating interactive STEM courseware traditionally requires HTML/CSS/JavaScript expertise, leaving barriers for educators. While generative AI can produce HTML codes, existing tools generate static presentations rather than interactive simulations, struggle with long documents, and lack pedagogical accuracy mechanisms. Furthermore, full regeneration for modifications requires 200--600 seconds, disrupting creative flow. We present MAIC-UI, a zero-code authoring system that enables educators to create and rapidly edit interactive courseware from textbooks, PPTs, and PDFs. MAIC-UI employs: (1) structured knowledge analysis with multi-modal understanding to ensure pedagogical rigor; (2) a two-stage generate-verify-optimize pipeline separating content alignment from visual refinement; and (3) Click-to-Locate editing with Unified Diff-based incremental generation achieving sub-10-second iteration cycles. A controlled lab study with 40 participants shows MAIC-UI reduces editing iterations (4.9 vs. 7.0) and significantly improves learnability and controllability compared to direct Text-to-HTML generation. A three-month classroom deployment with 53 high school students demonstrates that MAIC-UI fosters learning agency and reduces outcome disparities -- the pilot class achieved 9.21-point gains in STEM subjects compared to -2.32 points in control classes. Our code is available at this https URL. 

---
# Learning Generalizable Multimodal Representations for Software Vulnerability Detection 

**Authors**: Zeming Dong, Yuejun Guo, Qiang Hu, Yao Zhang, Maxime Cordy, Hao Liu, Mike Papadakis, Yongqiang Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25711)  

**Abstract**: Source code and its accompanying comments are complementary yet naturally aligned modalities-code encodes structural logic while comments capture developer intent. However, existing vulnerability detection methods mostly rely on single-modality code representations, overlooking the complementary semantic information embedded in comments and thus limiting their generalization across complex code structures and logical relationships. To address this, we propose MultiVul, a multimodal contrastive framework that aligns code and comment representations through dual similarity learning and consistency regularization, augmented with diverse code-text pairs to improve robustness. Experiments on widely adopted DiverseVul and Devign datasets across four large language models (LLMs) (i.e., DeepSeek-Coder-6.7B, Qwen2.5-Coder-7B, StarCoder2-7B, and CodeLlama-7B) show that MultiVul achieves up to 27.07% F1 improvement over prompting-based methods and 13.37% over code-only Fine-Tuning, while maintaining comparable inference efficiency. 

---
# Cross-Lingual Jailbreak Detection via Semantic Codebooks 

**Authors**: Shirin Alanova, Bogdan Minko, Sabrina Sadiekh, Evgeniy Kokuykin  

**Link**: [PDF](https://arxiv.org/pdf/2604.25716)  

**Abstract**: Safety mechanisms for large language models (LLMs) remain predominantly English-centric, creating systematic vulnerabilities in multilingual deployment. Prior work shows that translating malicious prompts into other languages can substantially increase jailbreak success rates, exposing a structural cross-lingual security gap. We investigate whether such attacks can be mitigated through language-agnostic semantic similarity without retraining or language-specific adaptation. Our approach compares multilingual query embeddings against a fixed English codebook of jailbreak prompts, operating as a training-free external guardrail for black-box LLMs. We conduct a systematic evaluation across four languages, two translation pipelines, four safety benchmarks, three embedding models, and three target LLMs (Qwen, Llama, GPT-3.5). Our results reveal two distinct regimes of cross-lingual transfer. On curated benchmarks containing canonical jailbreak templates, semantic similarity generalizes reliably across languages, achieving near-perfect separability (AUC up to 0.99) and substantial reductions in absolute attack success rates under strict low-false-positive constraints. However, under distribution shift - on behaviorally diverse and heterogeneous unsafe benchmarks - separability degrades markedly (AUC $\approx$ 0.60-0.70), and recall in the security-critical low-FPR regime drops across all embedding models. 

---
# Spreadsheet Modeling Experiments Using GPTs on Small Problem Statements and the Wall Task 

**Authors**: Thomas A. Grossman, Yuan Chen, Sopiko Datuashvili  

**Link**: [PDF](https://arxiv.org/pdf/2604.25689)  

**Abstract**: This paper investigates how GPT-based tools can assist in building reusable analytical spreadsheet models. After a screening, we evaluate five GPT extensions and select Excel AI by this http URL for detailed testing. Through structured experiments on simple problem statements, we assess Excel AI's performance against the ERFR criteria (each input in a cell; cell formulas; no hardwired numbers; labels; accurate). Results show that while Excel AI can produce well-structured models, it is inconsistent and often non-reproducible. We identify two central challenges - "the problem of confidence" and "the problem of workflow" - which highlight the need for skilled users to verify and adapt GPT-generated spreadsheets. Though GPTs show promise for generating draft models that may reduce development time or lower skill requirements, current tools remain unreliable for professional use. We conclude with recommendations for future research into prompt engineering, reproducibility, and larger-scale modeling tasks. 

---
# CORAL: Adaptive Retrieval Loop for Culturally-Aligned Multilingual RAG 

**Authors**: Nayeon Lee, Jiwoo Song, Byeongcheol Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25676)  

**Abstract**: Multilingual retrieval-augmented generation (mRAG) is often implemented within a fixed retrieval space, typically via query or document translation or multilingual embedding vector representations. However, this approach may be inadequate for culturally grounded queries, in which retrieval-condition misalignment may occur. Even strong retrievers and generators may struggle to produce culturally relevant answers when sourcing evidence from inappropriate linguistic or regional contexts. To this end, we introduce CORAL (COntext-aware Retrieval with Agentic Loop, an adaptive retrieval methodology for mRAG that enables iterative refinement of both the retrieval space (corpora) and the retrieval probe (query) based on the quality of the evidence. The overall process includes: (1) selecting corpora, (2) retrieving documents, (3) critiquing evidence for relevance and cultural alignment, and (4) checking sufficiency. If the retrieved documents are insufficient to answer the query correctly, the system (5) reselects corpora and rewrites the query. Across two cultural QA benchmarks, CORAL achieves up to a 3.58%p accuracy improvement on low-resource languages relative to the strongest baselines. 

---
# Emotive Architectures: The Role of LLMs in Adjusting Work Environments 

**Authors**: Lara Vartziotis, Tina Vartziotis, Frank Beutenmueller, Stella Salta, Konstantinos Moraitis, Miltiadis Katsaros, Sotirios Kotsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2604.25601)  

**Abstract**: In remote and hybrid work contexts, the integration of physical and digital environments is revolutionizing spatial experiences, collaboration, and interpersonal interactions. This study examines three fundamental spatial conditions: the physical environment, characterized by material and sensory attributes; the virtual environment, influenced by immersive technologies; and their fusion into hybrid environments where digital and physical components interact dynamically. The increasing number of AI tools in contemporary society, extensively utilized in both professional and personal spheres, has led to a varied landscape of developing technologies. For instance, ChatGPT has emerged as one of the most downloaded applications, a statistically substantiated fact that demonstrates the swift incorporation of language-based AI into daily life. It also underscores the function of large language models (LLMs) as meaningful bridges between concepts at reading emotional and behavioral signals via natural language. These models provide real-time modifications such as altering illumination, acoustics, or interface configurations, converting static settings into dynamic, emotionally receptive environments. We investigate the integration of language models into professional settings and their potential to enhance user experience by promoting focus, well-being, and engagement. The study investigates ethical concerns, including privacy, emotional tracking, and user agency, emphasizing the importance of inclusive and transparent design. This research formulates a framework for creating co-adaptive environments that merge technological innovation with human-centered experiences, offering a fresh viewpoint on responsive and supportive hybrid workspaces. 

---
# Walking Through Uncertainty: An Empirical Study of Uncertainty Estimation for Audio-Aware Large Language Models 

**Authors**: Chun-Yi Kuan, Wei-Ping Huang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.25591)  

**Abstract**: Recent audio-aware large language models (ALLMs) have demonstrated strong capabilities across diverse audio understanding and reasoning tasks, but they still frequently produce hallucinated or overly confident outputs. While uncertainty estimation has been extensively studied in text-only LLMs, it remains largely unexplored for ALLMs, where audio-conditioned generation introduces additional challenges such as perceptual ambiguity and cross-modal grounding. In this work, we present the first systematic empirical study of uncertainty estimation in ALLMs. We benchmark five representative methods, including predictive entropy, length-normalized entropy, semantic entropy, discrete semantic entropy, and P(True), across multiple models and diverse evaluation settings spanning general audio understanding, reasoning, hallucination detection, and unanswerable question answering. Our results reveal two key findings. First, semantic-level and verification-based methods consistently outperform token-level baselines on general audio reasoning benchmarks. Second, on trustworthiness-oriented benchmarks, the relative effectiveness of uncertainty methods becomes notably more model- and benchmark-dependent, indicating that conclusions drawn from general reasoning settings do not straightforwardly transfer to hallucination and unanswerable-question scenarios. We further explore uncertainty-based adaptive inference as a potential downstream application. We hope this study provides a foundation for future research on reliable, uncertainty-aware audio-language systems. 

---
# Large language models eroding science understanding: an experimental study 

**Authors**: Harry Collins, Hartmut Grote, Paul Newbury, Patrick Sutton, Simon Thorne  

**Link**: [PDF](https://arxiv.org/pdf/2604.25639)  

**Abstract**: This paper is under review in AI and Ethics This study examines whether large language models (LLMs) can reliably answer scientific questions and demonstrates how easily they can be influenced by fringe scientific material. The authors modified custom LLMs to prioritise knowledge in selected fringe papers on the Fine Structure Constant and Gravitational Waves, then compared their responses with those of domain experts and standard LLMs. The altered models produced fluent, convincing answers that contradicted scientific consensus and were difficult for non-experts to detect as misleading. The results show that LLMs are vulnerable to manipulation and cannot replace expert judgment, highlighting risks for public understanding of science and the potential spread of misinformation. 

---
# LLM-ReSum: A Framework for LLM Reflective Summarization through Self-Evaluation 

**Authors**: Huyen Nguyen, Haoxuan Zhang, Yang Zhang, Junhua Ding, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.25665)  

**Abstract**: Reliable evaluation of large language model (LLM)-generated summaries remains an open challenge, particularly across heterogeneous domains and document lengths. We conduct a comprehensive meta-evaluation of 14 automatic summarization metrics and LLM-based evaluators across seven datasets spanning five domains, covering documents from short news articles to long scientific, governmental, and legal texts (2K-27K words) with over 1,500 human-annotated summaries. Our results show that traditional lexical overlap metrics (e.g., ROUGE, BLEU) exhibit weak or negative correlation with human judgments, while task-specific neural metrics and LLM-based evaluators achieve substantially higher alignment, especially for linguistic quality assessment. Leveraging these findings, we propose LLM-ReSum, a self-reflective summarization framework that integrates LLM-based evaluation and generation in a closed feedback loop without model finetuning. Across three domains, LLM-ReSum improves low-quality summaries by up to 33% in factual accuracy and 39% in coverage, with human evaluators preferring refined summaries in 89% of cases. We additionally introduce PatentSumEval, a new human-annotated benchmark for legal document summarization comprising 180 expert-evaluated summaries. All code and datasets will be released in GitHub. 

---
# Prefill-Time Intervention for Mitigating Hallucination in Large Vision-Language Models 

**Authors**: Chengsheng Zhang, Chenghao Sun, Xinyan Jiang, Wei Li, Xinmei Tian  

**Link**: [PDF](https://arxiv.org/pdf/2604.25642)  

**Abstract**: Large Vision-Language Models (LVLMs) have achieved remarkable progress in visual-textual understanding, yet their reliability is critically undermined by hallucinations, i.e., the generation of factually incorrect or inconsistent responses. While recent studies using steering vectors demonstrated promise in reducing hallucinations, a notable challenge remains: they inadvertently amplify the severity of residual hallucinations. We attribute this to their exclusive focus on the decoding stage, where errors accumulate autoregressively and progressively worsen subsequent hallucinatory outputs. To address this, we propose Prefill-Time Intervention (PTI), a novel steering paradigm that intervenes only once during the prefill stage, enhancing the initial Key-Value (KV) cache before error accumulation occurs. Specifically, PTI is modality-aware, deriving distinct directions for visual and textual representations. This intervention is decoupled to steer keys toward visually-grounded objects and values to filter background noise, correcting hallucination-prone representations at their source. Extensive experiments demonstrate PTI's significant performance in mitigating hallucinations and its generalizability across diverse decoding strategies, LVLMs, and benchmarks. Moreover, PTI is orthogonal to existing decoding-stage methods, enabling plug-and-play integration and further boosting performance. Code is available at: this https URL. 

---
# SnapGuard: Lightweight Prompt Injection Detection for Screenshot-Based Web Agents 

**Authors**: Mengyao Du, Han Fang, Haokai Ma, Jiahao Chen, Kai Xu, Quanjun Yin, Ee-Chien Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25562)  

**Abstract**: Web agents have emerged as an effective paradigm for automating interactions with complex web environments, yet remain vulnerable to prompt injection attacks that embed malicious instructions into webpage content to induce unintended actions. This threat is further amplified for screenshot-based web agents, which operate on rendered visual webpages rather than structured textual representations, making predominant text-centric defenses ineffective. Although multimodal detection methods have been explored, they often rely on large vision-language models (VLMs), incurring significant computational overhead. The bottleneck lies in the complexity of modern webpages: VLMs must comprehend the global semantics of an entire page, resulting in substantial inference time and GPU memory usage. This raises a critical question: can we detect prompt injection attacks from screenshots in a lightweight manner? In this paper, we observe that injected webpages exhibit distinct characteristics compared to benign ones from both visual and textual perspectives. Building on this insight, we propose SnapGuard, a lightweight yet accurate method that reformulates prompt injection detection as multimodal representation analysis over webpage screenshots. SnapGuard leverages two complementary signals: a visual stability indicator that identifies abnormally smooth gradient distributions induced by malicious content, and action-oriented textual signals recovered via contrast-polarity reversal. Extensive evaluations across eight attacks and two benign settings demonstrate that SnapGuard achieves an F1 score of 0.75, outperforming GPT-4o-prompt while being 8x faster (1.81s vs. 14.50s) and introducing no additional memory overhead. 

---
# Health System Scale Semantic Search Across Unstructured Clinical Notes 

**Authors**: Faith Wavinya Mutinda, Spandana Makeneni, Anna Lin, Shivaji Dutta, Irit R. Rasooly, Patrick Dibussolo, Shivani Kamath Belman, Hessam Shahriari, Kevin Murphy, Alex B. Ruan, Barbara H. Chaiyachati, Sanjay Chainani, Robert W. Grundmeier, Scott M. Haag, Jeffrey M. Miller, Heather M. Griffis, Ian M. Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2604.25605)  

**Abstract**: Introduction: Semantic search, which retrieves documents based on conceptual similarity rather than keyword matching, offers substantial advantages for retrieval of clinical information. However, deploying semantic search across entire health systems, comprising hundreds of millions of clinical notes, presents formidable engineering, cost, and governance challenges that have prevented adoption. Methods: We deployed a semantic search system at a large children's hospital indexing 166 million clinical notes (484 million vectors) from 1.68 million patients. The system uses instruction-tuned qwen3-embedding-0.6B embeddings, stores vectors in a managed database with storage-optimized indexing, maintains full-text metadata in a low-latency key-value store, and operates within a HIPAA-compliant governance framework. We evaluated the system through three experiments: optimization of embedding model and chunking strategy using a physician-authored benchmark dataset, characterization of full-scale performance (cost, latency, retrieval quality), and clinical utility assessment via comparison of chart abstraction efficiency across three tasks. Results: The system delivers sub-second query latency (median 237 ms single-user, 451 ms 20-user concurrency) with monthly costs of approximately USD 4,000. Qwen3 embeddings with 300-token chunk size achieved 94.6% accuracy on a clinical question-answering benchmark. In clinical utility evaluation across three abstraction tasks, semantic search reduced time-to-completion by 24 to 89% compared to clinician-performed chart review while maintaining comparable inter-rater agreement. Conclusion: Health-system-scale semantic search is both technically and operationally feasible. The system provides infrastructure supporting interactive search, cohort generation, and downstream LLM-powered clinical applications without requiring specialized informatics expertise. 

---
# From CRUD to Autonomous Agents: Formal Validation and Zero-Trust Security for Semantic Gateways in AI-Native Enterprise Systems 

**Authors**: Ignacio Peyrano  

**Link**: [PDF](https://arxiv.org/pdf/2604.25555)  

**Abstract**: Enterprise software engineering is shifting away from deterministic CRUD/REST architectures toward AI-native systems where large language models act as cognitive orchestrators. This transition introduces a critical security tension: probabilistic LLMs weaken classical mechanisms for validation, access control, and formal testing.
This paper proposes the design, formal validation, and empirical evaluation of a Semantic Gateway governed by the Model Context Protocol (MCP). The gateway reframes the enterprise API as a semantic surface where tools are dynamically discovered, authorized, and executed based on intent and policy enforcement. The central contribution rests on a paradigm shift: autonomous agents must not be validated as traditional software nor as simple API consumers, but as stochastic state-transition systems whose behavior must be abstracted, fuzzed, and audited through enabled-tool graphs.
The architecture introduces a three-layer Zero-Trust security model comprising a pre-inference Semantic Firewall, deterministic Tool-Level RBAC, and out-of-band Cryptographic Human-in-the-Loop approval. Enabledness-Preserving Abstractions (EPAs) and greybox semantic fuzzing--originally developed for blockchain smart contract verification--are adapted to audit agent behavior in enterprise environments. Results demonstrate an 84.2% reduction in incidental code. Across 500,000 multi-turn fuzzing sequences, the methodology achieved a 100% discovery rate of hidden unauthorized state transitions, proving that dynamic formal verification is strictly necessary for secure agentic deployment. 

---
# AI as Consumer and Participant: A Co-Design Agenda for MBSE Substrates and Methodology 

**Authors**: Siyuan Ji  

**Link**: [PDF](https://arxiv.org/pdf/2604.25526)  

**Abstract**: AI tools are being deployed over MBSE models today, and those models were not designed for this kind of consumption. The problem is not simply that tools hallucinate: well-prompted frontier models produce competent, useful output over a conformant SysML model, but the reasoning they produce is drawn from training rather than retrieved from the model itself, and different tools over the same model produce different results with nothing in the record to adjudicate between them. The model, in other words, is functioning as a prompt rather than as a knowledge base. Attaching better tools to the same model does not resolve this. The model and the methodology that governs its construction need to be designed together for AI participation, treating the model as a machine-queryable knowledge substrate rather than a structured artefact for human navigation, and that co-design has not yet happened in any systematic way. This paper works through a concrete workflow scenario to show what that gap looks like in practice, proposes three principles that jointly characterise what model and methodology must achieve together, and closes with a call to the community to begin this work before the architectural decisions about AI integration settle without the methodological foundation they require. 

---
# Marco-MoE: Open Multilingual Mixture-of-Expert Language Models with Efficient Upcycling 

**Authors**: Fan Jiang, Yu Zhao, Chenyang Lyu, Tianqi Shi, Yichao Du, Feihu Jiang, Longyue Wang, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.25578)  

**Abstract**: We present Marco-MoE, a suite of fully open multilingual sparse Mixture-of-Experts (MoE) models. Marco-MoE features a highly sparse design in which only around 5\% of the total parameters are activated per input token. This extreme sparsity, combined with upcycling from dense models, enables efficient pre-training on 5T tokens. Our models surpass similarly-sized competitors on English and multilingual benchmarks, achieving a best-in-class performance-to-compute ratio. We further post-train these models to create Marco-MoE-\textsc{Instruct} variants, which surpass the performance of competing models possessing $3$--$14\times$ more activated parameters. Our analysis reveals that Marco-MoE learns structured expert activation patterns shared across related languages, while maintaining highly specialized utilization for linguistically isolated ones. We further show that Marco-MoE allows for scalable language expansion without the interference typical of dense models. To support the community, we disclose our full training datasets, recipes, and model weights. 

---
# From World-Gen to Quest-Line: A Dependency-Driven Prompt Pipeline for Coherent RPG Generation 

**Authors**: Dominik Borawski, Marta Szulc, Robert Chudy, Małgorzata Giedrowicz, Piotr Mironowicz  

**Link**: [PDF](https://arxiv.org/pdf/2604.25482)  

**Abstract**: Large Language Models (LLMs) have shown strong potential for narrative generation, but their use in complex, multi-layered role-playing game (RPG) worlds is still limited by issues of coherence, controllability, and structural consistency. This paper explores a dependency-aware, multi-stage prompt pipeline for procedural RPG content generation that models narrative dependencies through structured intermediate representations. The approach decomposes generation into sequential stages: world building, non-player character creation, player character creation, campaign-level quest planning, and quest expansion. Each stage conditions on structured JSON outputs from previous stages. By enforcing schemas and explicit data flow, the pipeline reduces narrative drift, limits hallucinations, and supports scalable creation of interconnected narrative elements. The system is evaluated qualitatively through human-centered analysis across multiple independent runs. Outputs are assessed using criteria such as structural completeness, internal consistency, narrative coherence, diversity, and actionability. Results show that the pipeline consistently generates logically sound and structurally valid RPG content, without quality degradation as complexity increases. Separating high-level campaign planning from detailed quest expansion improves both global structure and local storytelling. These findings suggest that dependency-aware prompt pipelines with structured intermediate representations are an effective design pattern for LLM-based procedural content generation. This approach may also generalize to other domains requiring sequential reasoning over evolving contextual states. 

---
# Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives 

**Authors**: Yu Wang, Emmanuele Chersoni, Chu-Ren Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25423)  

**Abstract**: Do large language models (LLMs) truly acquire embodied cognition and cultural conventions from text? We introduce demonstratives, fundamental spatial expressions like "this/that" in English and "zhè/nà" in Chinese, as a novel probe for grounded knowledge. Using 6,400 responses from 320 native speakers, we establish a human baseline: English speakers reliably distinguish proximal-distal referents but struggle with perspective-taking, while Chinese speakers switch perspectives fluently but tolerate distal ambiguity. In contrast, five state-of-the-art LLMs fail to inherently understand the proximal-distal contrast and show no cultural differences, defaulting to English-centric reasoning. Our study contributes (i) a new task, based on demonstratives, as a new lens for evaluating embodied cognition and cultural conventions; (ii) empirical evidence of cross-cultural asymmetries in human interpretation; (iii) a new perspective on the egocentric-sociocentric debate, showing both orientations coexist but vary across languages; and (iv) a call to address individual variation in future model design. 

---
# One-shot emergency psychiatric triage across 15 frontier AI chatbots 

**Authors**: Veith Weilnhammer, Lennart Luettgau, Christopher Summerfield, Viknesh Sounderajah, Elise Wilkinson, Virginia Corno, Matthew M Nour  

**Link**: [PDF](https://arxiv.org/pdf/2604.25415)  

**Abstract**: AI chatbots are increasingly used for health advice, but their performance in psychiatric triage remains undercharacterized. Psychiatric triage is particularly challenging because urgency must often be inferred from thoughts, behavior, and context rather than from objective findings.
We evaluated the performance of 15 frontier AI chatbots on psychiatric triage from realistic single-message disclosures using 112 clinical vignettes, each paired with 1 of 4 original benchmark triage labels: A, routine; B, assessment within 1 week; C, assessment within 24 to 48 hours; and D, emergency care now. Vignettes covered 9 psychiatric presentation clusters and 9 focal risk dimensions, organized into 28 presentation-by-risk groups. Each group contributed 4 distinct vignettes, with 1 vignette at each triage level. Each vignette was rendered as a realistic human-authored conversational query, and the AI chatbots were tasked with assigning a triage label from that disclosure.
Emergency under-triage occurred in 23 of 410 level D trials (5.6%), and all under-triaged emergencies were reassigned to level C urgency. Across target models, average accuracy ranged from 42.0% to 71.8%. Accuracy was highest for level D vignettes (94.3%) and lowest for level B vignettes (19.7%). Mean signed ordinal error was positive (+0.47 triage levels), indicating net over-triage. Dispersion was highest around the middle triage levels. All results were confirmed relative to clinician consensus labels from 50 medical doctors.
When presented with user messages containing sufficient clinical information, frontier AI chatbots thus recognized psychiatric emergencies as requiring urgent medical assessment with near-zero error rates, yet showed marked over-triage for low and intermediate risk presentations. 

---
# FED-FSTQ: Fisher-Guided Token Quantization for Communication-Efficient Federated Fine-Tuning of LLMs on Edge Devices 

**Authors**: Changyu Li, Shuanghong Huang, Jiashen Liu, Ming Lei, Jidu Xing, Kaishun Wu, Lu Wang, Fei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.25421)  

**Abstract**: Federated fine-tuning provides a practical route to adapt large language models (LLMs) on edge devices without centralizing private data, yet in mobile deployments the training wall-clock is often bottlenecked by straggler-limited uplink communication under heterogeneous bandwidth and intermittent participation. Although parameter-efficient fine-tuning (PEFT) reduces trainable parameters, per-round payloads remain prohibitive in non-IID regimes, where uniform compression can discard rare but task-critical signals. We propose Fed-FSTQ, a Fisher-guided token quantization system primitive for communication-efficient federated LLM fine-tuning. Fed-FSTQ employs a lightweight Fisher proxy to estimate token sensitivity, coupling importance-aware token selection with non-uniform mixed-precision quantization to allocate higher fidelity to informative evidence while suppressing redundant transmission. The method is model-agnostic, serves as a drop-in module for standard federated PEFT pipelines, e.g., LoRA, without modifying the server aggregation rule, and supports bandwidth-heterogeneous clients via compact sparse message packing. Experiments on multilingual QA and medical QA under non-IID partitions show that Fed-FSTQ reduces cumulative uplink traffic required to reach a fixed quality threshold by 46x relative to a standard LoRA baseline, and improves end-to-end wall-clock time-to-accuracy by 52%. Furthermore, enabling Fisher-guided token reduction at inference yields up to a 1.55x end-to-end speedup on NVIDIA Jetson-class edge devices, demonstrating deployability under tight resource constraints. 

---
# Co-Writing with AI: An Empirical Study of Diverse Academic Writing Workflows 

**Authors**: Silvia Bodei, Duncan P. Brumby, Katie Fisher, Jon Mella  

**Link**: [PDF](https://arxiv.org/pdf/2604.25389)  

**Abstract**: Despite AI tools becoming increasingly embedded in academic practice, little is known about how university students integrate them into their writing processes. We examine how students engage with AI across different writing tasks, and how this engagement is shaped by individual factors including AI literacy, writing confidence, trust, authorship concerns, and motivation. Study~1 surveys 107 UK university students to map task-specific and co-occurring patterns of AI use across five writing stages (ideation, sourcing, planning, drafting, and reviewing) and their associations with individual factors. Study~2 complements this by exploring how these patterns can be assembled in practice, through interviews with 12 postgraduates reflecting on their established use of AI in assessed writing. Together, the studies suggest that AI integration is selective and heterogeneous, forming three recurring and value-oriented configurations: (1) early-stage (learning-oriented), where tools support exploration and understanding; (2) late-stage (quality-oriented), where tools support drafting and refinement; and (3) peripheral (productivity-oriented), where tools are used to reduce friction and sustain momentum across the process. We offer a workflow-level account of AI-supported academic writing, showing how students navigate competing priorities of learning, quality, productivity, and authorship, and how they evaluate and take responsibility for AI-generated outputs. 

---
# An Investigation of Linguistic Biases in LLM-Based Recommendations 

**Authors**: Nitin Venkateswaran, Jason Ang, Deep Adhikari, Tarun Krishna Dasari  

**Link**: [PDF](https://arxiv.org/pdf/2604.25456)  

**Abstract**: We investigate linguistic biases in LLM-based restaurant and product recommendations given prompts varying across Southern American English (AE), Indian English (IE), and Code-Switched Hindi-English dialects, using the Yelp Open dataset (Yelp Inc., 2023) and Walmart product reviews dataset (PromptCloud,2020). We add lists of restaurant and product names balanced by cuisine type and product category to the prompts given to the LLM, and we zero-shot prompt the LLMs in a cold-start setting to select the top-20 restaurant and product recommendations from these lists for each of the dialect-varied prompts. We prompt LLMs using different list samples across 20 seeds for better generalization, and aggregate per cuisine-type and per category response counts for each seed, question/prompt, and LLM model. We run mixed-effects regression models for each model family and topic (restaurant/product) with the aggregate response counts as the dependent, and conduct likelihood ratio tests for the fixed effects with post-hoc pairwise testing of estimated marginal means differences, to investigate group-level differences in recommendation counts by model size and dialect type. Results show that dialect plays a role in the type of restaurant selected across the models tested with the mistral-small-3.1 model and both the llama-3.1 family models tested showing more sensitivity to Indian English and Code-Switched prompts. In terms of product recommendations, the llama-3.1-70B-model is particularly sensitive to Code-Switched prompts in four out of seven categories, and more beauty and home category recommendations are seen when using the Indian English and Code-Switched prompts for larger and smaller models, respectively. No broad trends are seen in the model-size based differences, with differing recommendations based on model sizes conditioned by the type of dialect. 

---
# The Structured Output Benchmark: A Multi-Source Benchmark for Evaluating Structured Output Quality in Large Language Models 

**Authors**: Abhinav Kumar Singh, Harsha Vardhan Khurdula, Yoeven D Khemlani, Vineet Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2604.25359)  

**Abstract**: Large Language Models are increasingly being deployed to extract structured data from unstructured and semi-structured sources: parsing invoices, medical records, and converting PDF documents to database entries. Yet existing benchmarks for structured output generation either focus on schema compliance alone, or evaluate value correctness within a single source domain. We introduce SOB (The Structured Output Benchmark), a multi-source benchmark spanning three source modalities: native text, images, and audio conversations. All models receive a text-normalized representation of their context regardless of source modality; this deliberate design isolates structured-output capability from raw vision or speech-processing quality, ensuring a fair, source-agnostic comparison. Our benchmark comprises 5,000 text evaluation records derived from multi-hop QA drawn from a 25,091-record full corpus, 209 image records from OCR-processed PDFs across seven document types including multi-column layouts, dense tables, scanned historical documents, small-print text, and mathematical typesetting, and 115 audio records from the AMI corpus. Each record pairs a natural-language question with a JSON schema that the model must follow and a ground-truth answer verified against the source context. We evaluate 21 frontier and open-weight models across three source domains and seven metrics. Our results reveal a consistent pattern: models achieve near-perfect schema compliance, yet the best Value Accuracy, measured by exact leaf-value match, reaches only 83.0% on text, 67.2% on images, and 23.7% on audio, where longer context makes extraction substantially harder. We release the dataset, evaluation pipeline, and all related code. 

---
# A Faceted Proposal for Transparent Attribution of AI-Assisted Text Production 

**Authors**: Geraldo Xexéo  

**Link**: [PDF](https://arxiv.org/pdf/2604.25346)  

**Abstract**: Artificial intelligence systems are increasingly integrated into writing processes, challenging traditional notions of authorship, responsibility, and intellectual contribution. Current disclosure practices usually indicate whether AI was used, but rarely explain how it was used, where it intervened, or how its output was reviewed. This paper proposes a faceted model for representing AI-assisted text production at the levels of documents, chapters, sections, and paragraphs. The proposal introduces a core model based on Form, Generation, and Evaluation, and an extended model that adds Intent, Control, and Traceability. The model is positioned as a minimal operational baseline with extensibility toward higher-fidelity representations. A worked example based on the production of this article demonstrates applicability. 

---
# AHASD: Asynchronous Heterogeneous Architecture for LLM Adaptive Drafting Speculative Decoding on Mobile Devices 

**Authors**: Ma zirui, Fan Zhihua, Li Wenxing, Wu Haibin, Zhang Fulin, Ye Xiaochun, Li Wenming  

**Link**: [PDF](https://arxiv.org/pdf/2604.25326)  

**Abstract**: Speculative decoding enhances the inference efficiency of large language models (LLMs) by generating drafts using a small draft language model (DLM) and verifying them in batches with a large target language model (TLM). However, adaptive drafting inference on a mobile single-NPU-PIM system faces idle overhead in traditional operator-level synchronous execution and wasted computation in asynchronous execution due to fluctuations in draft length. This paper introduces AHASD, a task-level asynchronous mobile NPU-PIM heterogeneous architecture for speculative decoding. Notably, AHASD achieves parallel drafting on the PIM and verification on a single NPU through task-level DLM-TLM decoupling and specifically, it incorporates Entropy-History-Aware Drafting Control and Time-Aware Pre-Verification Control to dynamically manage adaptive drafting algorithm execution and pre-verification timing, suppressing invalid drafting based on low-confidence drafts. Additionally, AHASD integrates Attention Algorithm Units and Gated Task Scheduling Units within LPDDR5-PIM to enable attention link localization and sub-microsecond task switching on the PIM side. Experimental results for different LLMs and adaptive drafting algorithms show that AHASD achieves up to 4.2$\times$ in throughput and 5.6$\times$ in energy efficiency improvements over a GPU-only baseline, and 1.5$\times$ in throughput and 1.24$\times$ in energy efficiency gains over the state-of-the-art GPU+PIM baseline, with hardware overhead below 3\% of the DRAM area. 

---
# Assistants, Not Architects: The Role of LLMs in Networked Systems Design 

**Authors**: Pratyush Sahu, Rahul Bothra, Venkat Arun, Brighten Godfrey, Akshay Narayan, Ahmed Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2604.25506)  

**Abstract**: Designing the architecture of modern networked systems requires navigating a large, combinatorial space of hardware, systems, and configuration choices with complex cross-layer interactions. Architects must balance competing objectives such as performance, cost, and deployability while satisfying compatibility and resource constraints, often relying on scattered rules-of-thumb drawn from benchmarks, papers, documentation, and expert experience. This raises a natural question: can large language models (LLMs) reliably perform this kind of architectural reasoning? We find that they cannot. While LLMs produce plausible configurations, they frequently miss critical constraints, encode incorrect assumptions, and exhibit ``stickiness'' to familiar patterns. A natural workaround--iterative validation via simulation or experimentation--is often prohibitively expensive at scale and, in many cases, infeasible, particularly when comparing hardware-dependent alternatives.
Motivated by this gap, we present Kepler, a lightweight reasoning framework for architecture design that combines structured, expert-driven specifications with SMT-based optimization. Kepler encodes architecturally significant properties--requirements, incompatibilities, and qualitative trade-offs--about systems, hardware, and workloads as constraints, and synthesizes feasible designs that optimize user-defined objectives. It operates at an abstract level, capturing ``rules-of-thumb'' rather than detailed system behavior, enabling tractable reasoning while preserving key interactions, and provides explanations for its decisions. Through experiments and case studies, we show that Kepler uncovers interactions missed by LLMs and supports systematic, explainable design exploration. 

---
# LegalMidm: Use-Case-Driven Legal Domain Specialization for Korean Large Language Model 

**Authors**: Youngjoon Jang, Chanhee Park, Hyeonseok Moon, Young-kyoung Ham, Jiwon Moon, Jinhyeon Kim, JuKyung Jung, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2604.25297)  

**Abstract**: In recent years, the rapid proliferation of open-source large language models (LLMs) has spurred efforts to turn general-purpose models into domain specialists. However, many domain-specialized LLMs are developed using datasets and training protocols that are not aligned with the nuanced requirements of real-world applications. In the legal domain, where precision and reliability are essential, this lack of consideration limits practical utility. In this study, we propose a systematic training framework grounded in the practical needs of the legal domain, with a focus on Korean law. We introduce LegalMidm, a Korean legal-domain LLM, and present a methodology for constructing high-quality, use-case-driven legal datasets and optimized training pipelines. Our approach emphasizes collaboration with legal professionals and rigorous data curation to ensure relevance and factual accuracy, and demonstrates effectiveness in key legal tasks. 

---
# Generative UI as an Accessibility Bridge: Lessons from C2C E-Commerce 

**Authors**: Bektur Ryskeldiev  

**Link**: [PDF](https://arxiv.org/pdf/2604.25455)  

**Abstract**: Web accessibility rests on static standards and developer compliance. That model frays in platforms where content is user-generated: photos arrive blurry or off-frame, descriptions skip size and condition, and page structure shifts from listing to listing. Drawing on six studies conducted between 2022 and 2025 with blind, low-vision, and older adult users of customer-to-customer (C2C) marketplaces, I argue that generative UI can produce adapted interfaces at the point of use, addressing barriers that static design cannot anticipate. Three interventions from this program -- HTML regeneration for screen readers, conversational guidance for older sellers, and audio-guided photo framing for blind sellers -- demonstrate how runtime generation can bridge gaps that standards leave open. I outline what these findings imply for HCI practice: generative UI extends beyond the screen, complements rather than replaces ability-based design, and shifts the designer's role from specifying layouts to specifying policies. This is an expanded arXiv version of a position paper accepted at the CHI 2026 workshop "What does Generative UI mean for HCI Practice?" 

---
# Faithfulness-QA: A Counterfactual Entity Substitution Dataset for Training Context-Faithful RAG Models 

**Authors**: Li Ju, Junzhe Wang, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25313)  

**Abstract**: Retrieval-Augmented Generation (RAG) models frequently produce answers grounded in parametric memory rather than the retrieved context, undermining the core promise of retrieval augmentation. A fundamental obstacle to fixing this unfaithfulness is the lack of training data that explicitly requires models to prefer context over internal knowledge. We introduce Faithfulness-QA, a large-scale dataset of 99,094 samples constructed through counterfactual entity substitution. Starting from two established extractive QA benchmarks--SQuAD and TriviaQA--we automatically identify answer-bearing named entities in each context, replace them with type-consistent alternatives drawn from a curated bank of 76,953 entities, and thereby manufacture controlled knowledge conflicts between context and parametric memory. Rigorous quality filtering ensures 100% pass rates across four automated checks on random 200-sample audits. We release the full dataset, the construction pipeline, and a typed entity bank covering eight named entity categories. Faithfulness-QA is designed as a training resource for attention-based faithfulness objectives and as an evaluation benchmark for measuring context-grounding behavior in RAG systems. Data and code are available at this https URL. 

---
# BARRED: Synthetic Training of Custom Policy Guardrails via Asymmetric Debate 

**Authors**: Arnon Mazza, Elad Levi  

**Link**: [PDF](https://arxiv.org/pdf/2604.25203)  

**Abstract**: Deploying guardrails for custom policies remains challenging, as generic safety models fail to capture task-specific requirements, while prompting LLMs suffers from inconsistent boundary-case performance and high inference costs. Training custom classifiers achieves both accuracy and efficiency, yet demands substantial labeled data that is costly to obtain. We present BARRED (Boundary Alignment Refinement through REflection and Debate), a framework for generating faithful and diverse synthetic training data using only a task description and a small set of unlabeled examples. Our approach decomposes the domain space into dimensions to ensure comprehensive coverage, and employs multi-agent debate to verify label correctness, yielding a high-fidelity training corpus. Experiments across diverse custom policies demonstrate that small language models finetuned on our synthetic data consistently outperform state-of-the-art proprietary LLMs (including reasoning models) and dedicated guardrail models. Ablation studies confirm that both dimension decomposition and debate-based verification are critical for ensuring the diversity and label fidelity required for effective fine-tuning. The BARRED framework eliminates the reliance on extensive human annotation, offering a scalable solution for accurate custom guardrails. 

---
# Cutscene Agent: An LLM Agent Framework for Automated 3D Cutscene Generation 

**Authors**: Lanshan He, Haozhou Pang, Qi Gan, Xin Shen, Ziwei Zhang, Yibo Liu, Gang Fang, Bo Liu, Kai Sheng, Shengfeng Zeng, Chaofan Li, Zhen Hui, Keer Zhou, Lan Zhou, Shujun Dai  

**Link**: [PDF](https://arxiv.org/pdf/2604.25318)  

**Abstract**: Cutscenes are carefully choreographed cinematic sequences embedded in video games and interactive media, serving as the primary vehicle for narrative delivery, character development, and emotional engagement. Producing cutscenes is inherently complex: it demands seamless coordination across screenwriting, cinematography, character animation, voice acting, and technical direction, often requiring days to weeks of collaborative effort from multidisciplinary teams to produce minutes of polished content. In this work, we present Cutscene Agent, an LLM agent framework for automated end-to-end cutscene generation. The framework makes three contributions: (1)~a Cutscene Toolkit built on the Model Context Protocol (MCP) that establishes \emph{bidirectional} integration between LLM agents and the game engine -- agents not only invoke engine operations but continuously observe real-time scene state, enabling closed-loop generation of editable engine-native cinematic assets; (2)~a multi-agent system where a director agent orchestrates specialist subagents for animation, cinematography, and sound design, augmented by a visual reasoning feedback loop for perception-driven refinement; and (3)~CutsceneBench, a hierarchical evaluation benchmark for cutscene generation. Unlike typical tool-use benchmarks that evaluate short, isolated function calls, cutscene generation requires long-horizon, multi-step orchestration of dozens of interdependent tool invocations with strict ordering constraints -- a capability dimension that existing benchmarks do not cover. We evaluate a range of LLMs on CutsceneBench and analyze their performance across this challenging task. 

---
# Making AI-Assisted Grant Evaluation Auditable without Exposing the Model 

**Authors**: Kemal Bicakci  

**Link**: [PDF](https://arxiv.org/pdf/2604.25200)  

**Abstract**: Public agencies are beginning to consider large language models (LLMs) as decision-support tools for grant evaluation. This creates a practical governance problem: the model and scoring rubric should not be exposed in a way that allows applicants to optimize against them, yet the evaluation process must remain auditable, contestable, and accountable.
We propose a TEE-based architecture that helps reconcile these requirements through remote attestation. The architecture allows an external verifier to check which model, rubric, prompt template, and input representation were used, without exposing model weights, proprietary scoring logic, or intermediate reasoning to applicants or infrastructure operators. The main artifact is an attested evaluation bundle: a signed, timestamped record linking the original submission hash, the canonical input hash, the model-and-rubric measurement, and the evaluation output.
The paper also considers a scenario-specific prompt injection risk: applicant-controlled documents may contain hidden or indirect instructions intended to influence the LLM evaluator. We therefore include a canonicalization and sanitization layer that normalizes document representations and records suspicious transformations before inference. We position the design relative to confidential AI inference, attestable AI audits, zero-knowledge machine learning, algorithmic accountability, and AI-assisted peer review. The resulting claim is deliberately narrow: remote attestation does not prove that an evaluation is fair or scientifically correct, but it can make part of the evaluation process externally verifiable. 

---
# R$^3$-SQL: Ranking Reward and Resampling for Text-to-SQL 

**Authors**: Hojae Han, Yeonseok Jeong, Seung-won Hwang, Zhewei Yao, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2604.25325)  

**Abstract**: Modern Text-to-SQL systems generate multiple candidate SQL queries and rank them to judge a final prediction. However, existing methods face two limitations. First, they often score functionally equivalent SQL queries inconsistently despite identical execution results. Second, ranking cannot recover when the correct SQL is absent from the candidate pool. We propose R$^3$-SQL, a Text-to-SQL framework that addresses both issues through unified reward for ranking and resampling. R$^3$-SQL first groups candidates by execution result and ranks groups for consistency. To score each group, it combines a pairwise preference across groups with a pointwise utility from the best group rank and size, capturing relative preference, consistency, and candidate quality. To improve candidate recall, R$^3$-SQL introduces agentic resampling, which judges the generated candidate pool and selectively resamples when the correct SQL is likely absent. R$^3$-SQL achieves 75.03 execution accuracy on BIRD-dev, a new state of the art among methods using models with disclosed sizes, with consistent gains across five benchmarks. 

---
# Below-Chance Blindness: Prompted Underperformance in Small LLMs Produces Positional Bias Rather than Answer Avoidance 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.25249)  

**Abstract**: Detecting sandbagging--the deliberate underperformance on capability evaluations--is an open problem in AI safety. We tested whether symptom validity testing (SVT) logic from clinical malingering detection could identify sandbagging through below-chance performance (BCB) on forced-choice items. In a pre-registered pilot at the 7-9 billion parameter instruction-tuned scale (3 models, 4 MMLU-Pro domains, 4 conditions, 500 items per cell, 24,000 total trials), the plausibility gate failed. Zero of 12 model-domain cells showed significant below-chance performance under sandbagging instruction. Exploratory analyses revealed three qualitatively distinct failure modes. Qwen-2.5-7B and Phi-3.5-mini largely ignored the sandbagging instruction, with 62-88% response identity with the honest baseline. Llama-3-8B complied substantially but implemented underperformance as a positional heuristic, collapsing its response distribution onto middle-alphabet options (E at 31.8%, F at 26.1%) regardless of where the correct answer fell. This produced accuracy boosts of up to 33 percentage points when the correct answer coincidentally occupied the model's preferred position. An explicit anti-task instruction ("pick the least likely answer") drove two of three models below chance, with accuracy as low as 0.024. The capability for answer-aware avoidance therefore exists but is not activated by "deliberately underperform." BCB did not fail as a logical marker of answer-aware avoidance. It was not observed in this regime because the model showing the largest behavioural shift exhibited behaviour consistent with a position-dominant response policy rather than content-aware answer avoidance. We propose that positional-distribution shift may be a more effective behavioural signature than below-chance accuracy for detecting prompted underperformance at this model scale. 

---
# Knowledge Distillation Must Account for What It Loses 

**Authors**: Wenshuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25110)  

**Abstract**: This position paper argues that knowledge distillation must account for what it loses: student models should be judged not only by retained task scores, but by whether they preserve the teacher capabilities that make those scores reliable. This matters because distillation is increasingly used to turn large, often frontier models into deployable systems, yet headline metrics can hide losses in uncertainty, boundary behavior, process reliability, on-policy stability, grounding, privacy, safety, and diversity. We identify the retention assumption behind current evaluation and reframe distillation as a lossy projection of teacher behavior rather than a faithful copy. We then synthesize existing evidence into a taxonomy of off-metric distillation losses, showing that these losses are concrete, recurring, and measurable. To make the position actionable, we propose scenario-specific preservation targets and a Distillation Loss Statement that reports what was preserved, what was lost, and why the remaining losses are acceptable. The goal is not lossless distillation, but accountable distillation. 

---
# Frictive Policy Optimization for LLMs: Epistemic Intervention, Risk-Sensitive Control, and Reflective Alignment 

**Authors**: James Pustejovsky, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2604.25136)  

**Abstract**: We propose Frictive Policy Optimization (FPO), a framework for learning language model policies that regulate not only what to say, but when and how to intervene in order to manage epistemic and normative risk. Unlike standard alignment methods that optimize surface-level preference or task utility, FPO treats clarification, verification, challenge, redirection, and refusal as explicit control actions whose purpose is to shape the evolution of belief, commitment, and uncertainty over time. We formalize alignment as a risk-sensitive epistemic control problem in which intervention decisions are selected based on their expected effect on downstream epistemic quality rather than on immediate reward alone. We introduce a compact taxonomy of frictive interventions, a structured friction functional that operationalizes multiple alignment failure modes, and a unified family of FPO methods spanning reward shaping, preference pairing, group-relative ranking, and risk-conditioned trust regions. We further propose an evaluation framework that measures epistemic competence directly through clarification behavior, calibration, contradiction repair, refusal proportionality, and information efficiency. Together, these results provide a formal and algorithmic foundation for learning agents that are aligned not only in outcome, but in epistemic conduct. 

---
# Frontier Coding Agents Can Now Implement an AlphaZero Self-Play Machine Learning Pipeline For Connect Four That Performs Comparably to an External Solver 

**Authors**: Joshua Sherwood, Ben Aybar, Benjamin Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2604.25067)  

**Abstract**: Forecasting when AI systems will become capable of meaningfully accelerating AI research is a central challenge for AI safety. Existing benchmarks measure broad capability growth, but may not provide ample early warning signals for recursive self-improvement. We propose measuring AI's capability to autonomously implement end-to-end machine learning pipelines from past AI research breakthroughs, given a minimal task description. By providing a concise task description instead of the full prior work as reference, we hope to better elicit emerging AI research taste. We introduce a proof-of-concept benchmark in which frontier coding agents autonomously implement an AlphaZero-style machine learning pipeline for Connect Four on consumer hardware within a three-hour budget, and we evaluate the resulting game AIs in a round-robin tournament anchored to the Pascal Pons Connect Four solver. Across four agents with eight trials each, we find substantial differentiation: Claude Opus 4.7 won as first-mover against Pons in seven of eight trials, statistically significantly better than the other agents tested, none of which exceeded two of eight. The task, which no frontier agent could reliably complete when we began development in January of 2026, is now near-saturation. Our evaluation also surfaced anomalous behavior in GPT-5.4, which consistently used far less of its allocated time budget than other agents. A follow-up 16-trial probe using shorter, less evaluation-coded prompts substantially increased GPT-5.4's time-budget usage, consistent with but not diagnostic of sandbagging; Bradley-Terry ratings across probe conditions showed only directional differences, despite significant differences in time-budget usage. We release our data, code, and prompts to support reproduction and extension. 

---
# M$^3$-VQA: A Benchmark for Multimodal, Multi-Entity, Multi-Hop Visual Question Answering 

**Authors**: Jiatong Ma, Longteng Guo, Yuchen Liu, Zijia Zhao, Dongze Hao, Xuanxu Lin, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25122)  

**Abstract**: We present M$^3$-VQA, a novel knowledge-based Visual Question Answering (VQA) benchmark, to enhance the evaluation of multimodal large language models (MLLMs) in fine-grained multimodal entity understanding and complex multi-hop reasoning. Unlike existing VQA datasets that focus on coarse-grained categories and simple reasoning over single entities, M$^3$-VQA introduces diverse multi-entity questions involving multiple distinct entities from both visual and textual sources. It requires models to perform both sequential and parallel multi-hop reasoning across multiple documents, supported by traceable, detailed evidence and a curated multimodal knowledge base. We evaluate 16 leading MLLMs under three settings: without external knowledge, with gold evidence, and with retrieval-augmented input. The poor results reveal significant challenges for MLLMs in knowledge acquisition and reasoning. Models perform poorly without external information but improve markedly when provided with precise evidence. Furthermore, reasoning-aware agentic retrieval surpasses heuristic methods, highlighting the importance of structured reasoning for complex multimodal understanding. M$^3$-VQA presents a more challenging evaluation for advancing the multimodal reasoning capabilities of MLLMs. Our code and dataset are available at this https URL. 

---
# Structured Security Auditing and Robustness Enhancement for Untrusted Agent Skills 

**Authors**: Lijia Lv, Xuehai Tang, Jie Wen, Jizhong Han, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25109)  

**Abstract**: Agent Skills package this http URL files, scripts, reference documents, and repository context into reusable capability units, turning pre-load auditing from single-prompt filtering into cross-file security review. Existing guardrails often flag risk but recover malicious intent inconsistently under semantics-preserving rewrites. This paper formulates pre-load auditing for untrusted Agent Skills as a robust three-way classification task and introduces SkillGuard-Robust, which combines role-aware evidence extraction, selective semantic verification, and consistency-preserving adjudication. We evaluate SkillGuard-Robust on SkillGuardBench and two public-ecosystem extensions through five large evaluation views ranging from 254 to 404 packages. On the 404-package held-out aggregate, SkillGuard-Robust reaches 97.30% overall exact match, 98.33% malicious-risk recall, and 98.89% attack exact consistency. On the 254-package external-ecosystem view, it reaches 99.66%, 100.00%, and 100.00%, respectively. These results support a bounded conclusion: factorized package auditing materially improves frozen and public-ecosystem robustness, while harsher external-source transfer remains an open challenge. 

---
# Analyzing LLM Reasoning to Uncover Mental Health Stigma 

**Authors**: Sreehari Sankar, Aliakbar Nafar, Mona Barman, Hannah K. Heitz, Ashwin Kumar, Pouria Tohidi, Dailun Li, Danish Hussain, Russell DuBois, Hamed Hasheminia, Farshad Majzoubi  

**Link**: [PDF](https://arxiv.org/pdf/2604.25053)  

**Abstract**: While large language models (LLMs) are increasingly being explored for mental health applications, recent studies reveal that they can exhibit stigma toward individuals with psychological conditions. Existing evaluations of this stigma primarily rely on multiple-choice questions (MCQs), which fail to capture the biases embedded within the models' underlying logic. In this paper, we analyze the intermediate reasoning steps of LLMs to uncover hidden stigmatizing language and the internal rationales driving it. We leverage clinical expertise to categorize common patterns of stigmatizing language directed at individuals with psychological conditions and use this framework to identify and tag problematic statements in LLM reasoning. Furthermore, we rate the severity of these statements, distinguishing between overt prejudice and more subtle, less immediately harmful biases. To broaden the reasoning domain and capture a wider array of patterns, we also extend an existing mental health stigma benchmark by incorporating additional psychological conditions. Our findings demonstrate that evaluating model reasoning not only exposes substantially more stigma than traditional MCQ-based methods but it helps to identify the flaws in the LLMs' logic and their understanding of mental health conditions. 

---
# Faithful Autoformalization via Roundtrip Verification and Repair 

**Authors**: Daneshvar Amrollahi, Jerry Lopez, Clark Barrett  

**Link**: [PDF](https://arxiv.org/pdf/2604.25031)  

**Abstract**: When an LLM formalizes natural language, how do we know the output is faithful? We propose a roundtrip verification approach which does not require ground-truth annotations: formalize a statement, translate the result back to natural language, re-formalize, and use a formal tool to check logical equivalence. When the two formalizations agree, this provides evidence of a faithful formalization. When they disagree, a diagnosis step identifies which translation stage failed, and a targeted repair operator attempts to correct that stage. We evaluate our approach on 150 traffic rules using Claude Opus 4.6 and GPT-5.2. Diagnosis-guided repair raises formal equivalence from 45--61% to 83--85% for both models, outperforming a random-repair baseline. An independent NLI analysis confirms that formal equivalence is correlated with less semantic drift. 

---
# Dual-Track CoT: Budget-Aware Stepwise Guidance for Small LMs 

**Authors**: Sagnik Chatterjee, Atharva Patil, Sricharan Ramesh  

**Link**: [PDF](https://arxiv.org/pdf/2604.25039)  

**Abstract**: Large Language Models (LLMs) solve many reasoning tasks via chain-of-thought (CoT) prompting, but smaller models (about 7 to 8B parameters) still struggle with multi-step reasoning under tight compute and token budgets. Existing test time reasoning methods such as self consistency (sampling multiple rationales and voting), Tree-of-Thoughts (search over intermediate thoughts), and critique revise loops improve performance, but often at high token cost and without fine-grained step-level control. This project1 aims to address that gap: can Small Language Models (SLMs) reason reliably using the same or fewer tokens? This question is both scientific and practical. Scientifically, it probes whether process supervision and simple test-time controls (such as token budgets and rejection of redundant steps) can substitute for model scale or large sampling counts. Practically, many deployments (on-device, low-latency, or cost-constrained settings) cannot afford huge models or dozens of sampled rationales per query. A method that improves SLM reasoning at fixed cost would therefore be directly useful. 

---
# BenchGuard: Who Guards the Benchmarks? Automated Auditing of LLM Agent Benchmarks 

**Authors**: Xinming Tu, Tianze Wang, Yingzhou, Kexin Huang, Yuanhao Qu, Sara Mostafavi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24955)  

**Abstract**: As benchmarks grow in complexity, many apparent agent failures are not failures of the agent at all - they are failures of the benchmark itself: broken specifications, implicit assumptions, and rigid evaluation scripts that penalize valid alternative approaches. We propose employing frontier LLMs as systematic auditors of evaluation infrastructure, and realize this vision through BenchGuard, the first automated auditing framework for task-oriented, execution-based agent benchmarks. BenchGuard cross-verifies all benchmark artifacts via structured LLM protocols, optionally incorporating agent solutions or execution traces as additional diagnostic evidence. Deployed on two prominent scientific benchmarks, BenchGuard identified 12 author-confirmed issues in ScienceAgentBench - including fatal errors rendering tasks unsolvable - and exactly matched 83.3% of expert-identified issues on the BIXBench Verified-50 subset, catching defects that prior human review missed entirely. A full audit of 50 complex bioinformatics tasks costs under USD 15, making automated benchmark auditing a practical and valuable complement to human review. These findings point toward AI-assisted benchmark development, where frontier models serve not only as subjects of evaluation but as active participants in validating the evaluation infrastructure itself. 

---
# Compute Aligned Training: Optimizing for Test Time Inference 

**Authors**: Adam Ousherovitch, Ambuj Tewari  

**Link**: [PDF](https://arxiv.org/pdf/2604.24957)  

**Abstract**: Scaling test-time compute has emerged as a powerful mechanism for enhancing Large Language Model (LLM) performance. However, standard post-training paradigms, Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), optimize the likelihood of individual samples under a base policy, creating a misalignment with test time procedures that rely on aggregated or filtered outputs. In this work, we propose Compute Aligned Training, which aligns training objectives with test-time strategies. By conceptualizing inference strategies as operators on the base policy, we derive new loss functions that maximize performance when said strategies are applied. We instantiate such loss functions for SFT and RL across common test time strategies. Finally, we provide empirical evidence that this training method substantially improves test time scaling over standard training. 

---
# Gradient-Direction Sensitivity Reveals Linear-Centroid Coupling Hidden by Optimizer Trajectories 

**Authors**: Yongzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25143)  

**Abstract**: We show that replacing the rolling SVD of AdamW updates with a rolling SVD of loss gradients changes the diagnostic by 1-2 orders of magnitude. Performing SVD on the loss gradient instead of the AdamW update increases the measured perturbative coupling between SED directions and Linear Centroid Hypothesis (LCH) features from $ \bar{R}_k \approx 3 $--$9\times$ to $100$--$330\times$ across four single-task modular arithmetic operations, eliminating the apparent operation dependence in the original measurement. On a multitask transformer with a shared encoder, update-based SED gives $ \bar{R}_k \leq 1 $ -- an apparent failure of the diagnostic -- while per-operation gradient-based SED recovers $ \bar{R}_k = 20 $--$45\times$ across all four operations. Gradient aggregation across competing tasks is the main obstruction; performing SVD on per-task gradients resolves it. A causal intervention shows that constraining attention updates to any rank-3 subspace (whether SED-derived or random) accelerates grokking by approximately $2.3\times$ across random seeds and operations, while removing the rank-3 component has negligible effect under proper gradient-projection methodology. The SED-LCH coupling is therefore a strong diagnostic of where feature formation concentrates in parameter space, but it is not a unique causal pathway: the natural full-rank AdamW attention update is highly rank-redundant under our hyperparameters. 

---
# Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence 

**Authors**: NVIDIA, Amala Sanjay Deshmukh, Kateryna Chumachenko, Tuomas Rintamaki, Matthieu Le, Tyler Poon, Danial Mohseni Taheri, Ilia Karmanov, Guilin Liu, Jarno Seppanen, Arushi Goel, Mike Ranzinger, Greg Heinrich, Guo Chen, Lukas Voegtle, Philipp Fischer, Timo Roman, Karan Sapra, Collin McCarthy, Shaokun Zhang, Fuxiao Liu, Hanrong Ye, Yi Dong, Mingjie Liu, Yifan Peng, Piotr Zelasko, Zhehuai Chen, Nithin Rao Koluguri, Nune Tadevosyan, Lilit Grigoryan, Ehsan Hosseini Asl, Pritam Biswas, Leili Tavabi, Yuanhang Su, Zhiding Yu, Peter Jin, Alexandre Milesi, Netanel Haber, Yao Xu, Sarah Amiraslani, Nabin Mulepati, Eric Tramel, Jaehun Jung, Ximing Lu, Brandon Cui, Jin Xu, Zhiqi Li, Shihao Wang, Yuanguo Kuang, Shaokun Zhang, Huck Yang, Boyi Li, Hongxu Yin, Song Han, Pavlo Molchanov, Adi Renduchintala, Charles Wang, David Mosallanezhad, Soumye Singhal, Luis Vega, Katherine Cheung, Sreyan Ghosh, Yian Zhang, Alexander Bukharin, Venkat Srinivasan, Johnny Greco, Andre Manoel, Maarten Van Segbroeck, Suseella Panguliri, Rohit Watve, Divyanshu Kakwani, Shubham Pachori, Jeffrey Glick, Radha Sri-Tharan, Aileen Zaman, Khanh Nguyen, Shi Chen, Jiaheng Fang, Qing Miao, Wenfei Zhou, Yu Wang, Zaid Pervaiz Bhat, Varun Praveen, Arihant Jain, Ramanathan Arunachalam, Tomasz Kornuta, Ashton Sharabiani, Amy Shen, Wei Huang, Yi-Fu Wu, Ali Roshan Ghias, Huiying Li, Brian Yu, Nima Tajbakhsh, Chen Cui, Wenwen Gao, Li Ding, Terry Kong, Manoj Kilaru, Anahita Bhiwandiwalla  

**Link**: [PDF](https://arxiv.org/pdf/2604.24954)  

**Abstract**: We introduce Nemotron 3 Nano Omni, the latest model in the Nemotron multimodal series and the first to natively support audio inputs alongside text, images, and video. Nemotron 3 Nano Omni delivers consistent accuracy improvements over its predecessor, Nemotron Nano V2 VL, across all modalities, enabled by advances in architecture, training data and recipes. In particular, Nemotron 3 delivers leading results in real-world document understanding, long audio-video comprehension, and agentic computer use. Built on the highly efficient Nemotron 3 Nano 30B-A3B backbone, Nemotron 3 Nano Omni further incorporates innovative multimodal token-reduction techniques to deliver substantially lower inference latency and higher throughput than other models of similar size. We are releasing model checkpoints in BF16, FP8, and FP4 formats, along with portions of the training data and codebase to facilitate further research and development. 

---
# Kohn-Sham Hamiltonian from Effective Field Theory: Quasiparticle Band Narrowing from Frozen Core Dynamics 

**Authors**: Xiansheng Cai, Han Wang, Kun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.25199)  

**Abstract**: Kohn-Sham (KS) eigenvalues are routinely compared with angle-resolved photoemission (ARPES) and used as input for many-body methods, yet density functional theory (DFT) assigns them no physical meaning. For alkali and alkaline-earth metals, KS bandwidths overestimate ARPES measurements by 20-35%, a discrepancy that persists across all exchange-correlation functionals. We construct an effective field theory (EFT) of the inhomogeneous electron gas and show that two conditions imply KS bands are the quasiparticle bands, up to a frozen-core renormalization factor zcore: a scale separation between core excitation energies and the valence Fermi energy, and an approximate Galilean invariance of the uniform electron gas confirmed by diagrammatic Monte Carlo. This factor reflects dynamical core excitations that conventional pseudopotentials freeze out and no static potential can capture. The correction 1-zcore reaches 20-35% for alkali metals but falls below 5% for Al and Si, explaining both the failure and success of KS band theory. We derive a closed-form post-SCF formula and validate it for Li, Na, K, Ca, Mg, Al, and Si; the predicted quasiparticle bands resolve the long-standing ARPES bandwidth discrepancy, matching embedded dynamical mean-field theory at negligible cost. This work also exemplifies first-principles agentic science, a direction particularly suited to the AGI-for-Science paradigm: an LLM-co-developed derivation with controlled approximations, verified symbolically and against a few experiments, becomes a deterministic harness for agentic scale-out, resolving simultaneously the LLM audit bottleneck and the non-falsifiability of fit-based AI-for-science. 

---
# SUDP: Secret-Use Delegation Protocol for Agentic Systems 

**Authors**: Xiaohang Yu, Hejia Geng, William Knottenbelt  

**Link**: [PDF](https://arxiv.org/pdf/2604.24920)  

**Abstract**: Agentic systems increasingly act with user secrets for APIs, messaging platforms, and cloud services. Today's bearer-secret interfaces implement authorization by exposure: enabling action often means placing a reusable secret, or a reusable artifact derived from it, within a model-steerable boundary, so a transient prompt-injection or tool-side compromise becomes durable account compromise. Existing defenses cover adjacent pieces such as secret storage, scoped delegation, sender-constrained tokens, and runtime monitoring, but leave the combined agentic obligation without a common specification: an untrusted autonomous requester should be able to cause a user-authorized secret-backed operation without exposing reusable authority to the requester. We formalize this problem as Agent Secret Use (ASU). From ASU we derive a security-property taxonomy that separates the problem's structural obligations from the realization-level robustness conditions any concrete construction must establish, enabling principled comparison of existing agentic-secret defenses against a problem-grounded specification. We propose the Secret-Use Delegation Protocol (SUDP), a three-role protocol realizing ASU: a requester proposes a canonical operation; the user authorizes it with a fresh authenticator-backed grant; and a custodian redeems the grant once to perform the bounded use, so reusable authority never crosses the requester boundary. We specialize SUDP for agentic deployments: agents propose operations; they do not retrieve secrets. Under explicit assumptions, we show that SUDP satisfies the ASU requirements: authorization is verifiable, operation-bound, and single-use. SUDP also provides storage confidentiality and wrapping-epoch key isolation under stated sealing and erasure assumptions; plaintext-level forward secrecy of the underlying secret additionally requires the environment to rotate and revoke it. 

---
# ADE: Adaptive Dictionary Embeddings -- Scaling Multi-Anchor Representations to Large Language Models 

**Authors**: Orhan Demirci, Sezer Aptourachman  

**Link**: [PDF](https://arxiv.org/pdf/2604.24940)  

**Abstract**: Word embeddings are fundamental to natural language processing, yet traditional approaches represent each word with a single vector, creating representational bottlenecks for polysemous words and limiting semantic expressiveness. While multi-anchor representations have shown promise by representing words as combinations of multiple vectors, they have been limited to small-scale models due to computational inefficiency and lack of integration with modern transformer architectures. We introduce Adaptive Dictionary Embeddings (ADE), a framework that successfully scales multi-anchor word representations to large language models. ADE makes three key contributions: (1) Vocabulary Projection (VP), which transforms the costly two-stage anchor lookup into a single efficient matrix operation; (2) Grouped Positional Encoding (GPE), a novel positional encoding scheme where anchors of the same word share positional information, preserving semantic coherence while enabling anchor-level variation; and (3) context-aware anchor reweighting, which leverages self-attention to dynamically compose anchor contributions based on sequence context. We integrate these components into the Segment-Aware Transformer (SAT), which provides context-aware reweighting of anchor contributions at inference time. We evaluate ADE on AG News and DBpedia-14 text classification benchmarks. With 98.7% fewer trainable parameters than DeBERTa-v3-base, ADE surpasses DeBERTa on DBpedia-14 (98.06% vs. 97.80%) and approaches it on AG News (90.64% vs. 94.50%), while compressing the embedding layer over 40x -- demonstrating that multi-anchor representations are a practical and parameter-efficient alternative to single-vector embeddings in modern transformer architectures. 

---
# Large Language Models Explore by Latent Distilling 

**Authors**: Yuanhao Zeng, Ao Lu, Lufei Li, Zheng Zhang, Yexin Li, Kan Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.24927)  

**Abstract**: Generating diverse responses is crucial for test-time scaling of large language models (LLMs), yet standard stochastic sampling mostly yields surface-level lexical variation, limiting semantic exploration. In this paper, we propose Exploratory Sampling (ESamp), a decoding approach that explicitly encourages semantic diversity during generation. ESamp is motivated by the well-known observation that neural networks tend to make lower-error predictions on inputs similar to those encountered before, and incur higher prediction error on novel ones. Building on this property, we train a lightweight Distiller at test time to predict deep-layer hidden representations of the LLM from its shallow-layer representations to model the LLM's depth-wise representation transitions. During decoding, the Distiller continuously adapts to the mappings induced by the current generation context. ESamp uses the prediction error as a novelty signal to reweight candidate token extensions conditioned on the current prefix, thereby biasing decoding toward less-explored semantic patterns. ESamp is implemented with an asynchronous training--inference pipeline, with less than 5% worst case overhead (1.2% in the optimized release). Empirical results show that ESamp significantly boosts the Pass@k efficiency of reasoning models, showing superior or comparable performance to strong stochastic and heuristic baselines. Notably, ESamp achieves robust generalization across mathematics, science, and code generation benchmarks and breaks the trade-off between diversity and coherence in creative writing. Our code has released at: this https URL. 

---
# Rethinking Layer Redundancy in Large Language Models: Calibration Objectives and Search for Depth Pruning 

**Authors**: Minkyu Kim, Vincent-Daniel Yun, Youngrae Kim, Youngjin Heo, Suin Cho, Seong-hun Kim, Woosang Lim, Gaeul Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2604.24938)  

**Abstract**: Depth pruning improves the inference efficiency of large language models by removing Transformer blocks. Prior work has focused on importance criteria and search algorithms, often treating layer redundancy as an inherent structural property of pretrained networks. In contrast, we adopt a \emph{functional perspective}, where redundancy is jointly influenced by the model and the evaluation objective, suggesting that a universal ranking may not be sufficient. Through an empirical study across three LLM families, two calibration objectives, and seven search algorithms, we observe that different objectives yield qualitatively different redundant layers, and that perplexity and downstream accuracy rankings do not consistently align. Under a fixed objective, however, search algorithms tend to produce similar solutions. Overall, our results suggest that the calibration objective may play a more influential role than the choice of search algorithm, indicating that further attention to objective design could be beneficial. 

---
# MultiHedge: Adaptive Coordination via Retrieval-Augmented Control 

**Authors**: Feliks Bańka, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2604.24905)  

**Abstract**: Decision-making under changing conditions remains a fundamental challenge in many real-world systems. Existing approaches often fail to generalize across shifting regimes and exhibit unstable behavior under uncertainty. This raises the research question: can retrieval-augmented LLM coordination improve the robustness of modular decision pipelines? We propose MultiHedge, a hybrid architecture where an LLM produces structured allocation decisions conditioned on retrieved historical precedents, and execution is grounded in canonical option strategies. In a controlled evaluation using U.S. equities, we compare MultiHedge to rule-based and learning-based baselines. The key result is that memory-augmented retrieval confers greater robustness and stability than increasing model scale alone. Our paper contributes a controlled computational study showing that memory and architectural design play a central role in robustness in modular decision systems. 

---
# GAIA-v2-LILT: Multilingual Adaptation of Agent Benchmark beyond Translation 

**Authors**: Yunsu Kim, Kaden Uhlig, Joern Wuebker  

**Link**: [PDF](https://arxiv.org/pdf/2604.24929)  

**Abstract**: Agent benchmarks remain largely English-centric, while their multilingual versions are often built with machine translation (MT) and limited post-editing. We argue that, for agentic tasks, this minimal workflow can easily break benchmark validity through query-answer misalignment or culturally off-target context. We propose a refined workflow for adapting English benchmarks into multiple languages with explicit functional alignment, cultural alignment, and difficulty calibration using both automated checks and human review. Using this workflow, we introduce GAIA-v2-LILT, a re-audited multilingual extension of GAIA covering five non-English languages. In experiments, our workflow improves agent success rates by up to 32.7% over minimally translated versions, bringing the closest audited setting to within 3.1% of English performance while substantial gaps remain in many other cases. This indicates that a substantial share of the multilingual performance gap is benchmark-induced measurement error, motivating task-level alignment when adapting English benchmarks across languages. The data is available as part of the MAPS package at this https URL. We also release the code used in our experiments at this https URL. 

---
# On the Trainability of Masked Diffusion Language Models via Blockwise Locality 

**Authors**: Yuxiang Wang, Yu Xiang, Baojian Zhou, Qifang Zhao, Keyue Jiang, Yanghua Xiao, Xiaoxiao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24832)  

**Abstract**: Masked diffusion language models (MDMs) have recently emerged as a promising alternative to standard autoregressive large language models (AR-LLMs), yet their optimization can be substantially less stable. We study blockwise MDMs and compare them with AR-LLMs on three controlled tasks that stress different aspects of structured generation: in-context linear regression, graph path-finding, and Sudoku solving. We find that standard random-masking MDMs fail to reliably learn linear regression, exhibit high variance training dynamics on graph path-finding, while outperforming AR-LLMs on Sudoku. To mitigate these instabilities, we propose two locality aware blockwise models, namely Jigsaw and Scatter, that inject left-to-right inductive bias by enforcing autoregressive locality within blocks while preserving iterative refinement at the block level. Empirically, Jigsaw matches AR-LLM stability on linear regression and remains strong on Sudoku, while Scatter retains diffusion's planning advantage on path-finding. Our results indicate that standard random-masking MDMs, even with blockwise variants, may be a suboptimal instantiation of diffusion LMs for ordered generation, motivating models beyond random masking. 

---
# Incompressible Knowledge Probes: Estimating Black-Box LLM Parameter Counts via Factual Capacity 

**Authors**: Bojie Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.24827)  

**Abstract**: Closed-source frontier labs do not disclose parameter counts, and the standard alternative -- inference economics -- carries $2\times$+ uncertainty from hardware, batching, and serving-stack assumptions external to the model. We exploit a tighter intrinsic bound: storing $F$ facts requires at least $F/$(bits per parameter) weights, so measuring how much a model \emph{knows} lower-bounds how many parameters it \emph{has}. We introduce \textbf{Incompressible Knowledge Probes (IKPs)}, a benchmark of 1{,}400 factual questions spanning 7 tiers of obscurity, designed to isolate knowledge that cannot be derived by reasoning or compressed by architectural improvements.
We calibrate a log-linear mapping from IKP accuracy to parameter count on 89 open-weight models (135M--1,600B) spanning 19 vendors, achieving $R^2 = 0.917$; leave-one-out cross-validation confirms generalization (median fold error $1.59\times$, $68.5\%$ within $2\times$ and $87.6\%$ within $3\times$). For Mixture-of-Experts models, total parameters predict knowledge ($R^2 = 0.79$) far better than active parameters ($R^2 = 0.51$). We evaluate 188 models from 27 vendors and estimate effective knowledge capacity for all major proprietary frontier models; for heavily safety-tuned models the estimates are lower bounds, since refusal policy can hide tens of percentage points of "refused but known" capacity.
The widely-reported saturation of reasoning benchmarks does not imply the end of scaling. Procedural capability compresses under the "Densing Law," but across 96 dated open-weight models the IKP time coefficient is $-0.0010$/month (95\% CI $[-0.0031, +0.0008]$) -- indistinguishable from zero, and rejecting the Densing prediction of $+0.0117$/month at $p < 10^{-15}$. Factual capacity continues to scale log-linearly with parameters across generations and across vendors. 

---
# A Comparative Evaluation of AI Agent Security Guardrails 

**Authors**: Qi Li, Jiu Li, Pingtao Wei, Jianjun Xu, Xueyi Wei, Jiwei Shi, Xuan Zhang, Yanhui Yang, Xiaodong Hui, Peng Xu, Lingquan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.24826)  

**Abstract**: This report presents a comparative evaluation of DKnownAI Guard in AI agent security scenarios, benchmarked against three competing products: AWS Bedrock Guardrails, Azure Content Safety, and Lakera Guard. Using human annotation as the ground truth, we assess each guardrail's ability to detect two categories of risks: threats to the agent itself (e.g., instruction override, indirect injection, tool abuse) and requests intended to elicit harmful content (e.g., hate speech, pornography, violence). Evaluation results demonstrate that DKnownAI Guard achieves the highest recall rate at 96.5\% and ranks first in true negative rate (TNR) at 90.4\%, delivering the best overall performance among all evaluated guardrails. 

---
# SWE-QA: A Dataset and Benchmark for Complex Code Understanding 

**Authors**: Laïla Elkoussy, Julien Perez  

**Link**: [PDF](https://arxiv.org/pdf/2604.24814)  

**Abstract**: In this paper, we introduce SWE-QA, a text and code corpus aimed at benchmarking multi-hop code comprehension, addressing the gap between simplified evaluation tasks and the complex reasoning required in real-world software development. While existing code understanding benchmarks focus on isolated snippets, developers must routinely connect information across multiple dispersed code segments. The dataset comprises 9,072 multiple-choice questions systematically generated from 12 Python repositories of SWE-bench, evaluating several recurrent reasoning patterns like Declaration-and-Call questions that link entity definitions to their usage, and Interacting-Entity questions that examine the dynamic relationships among multiple collaborating components. Generated through parsing-based entity extraction and Large Language Model assisted question construction with carefully validated distractors, the benchmark distinguishes genuine comprehension from superficial pattern matching. Evaluation of 15 language models (360M to 671B parameters) reveals significant challenges in multi-hop reasoning, with best performance reaching 74.41% accuracy. Dense architectures consistently outperform mixture-of-experts models by 10-14 percentage points, while reasoning-enhanced variants show inconsistent benefits. 

---
# Salca: A Sparsity-Aware Hardware Accelerator for Efficient Long-Context Attention Decoding 

**Authors**: Wang Fan, Wei Cao, Xi Zha, Kedi Ma, MingQian Sun, Jialin Chen, Fengzhe Zhang, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24820)  

**Abstract**: Long contexts improve capabilities of large language models but pose serious hardware challenges: compute and memory footprints grow linearly with sequence length. Particularly, the decoding phase continuously accesses massive KV cache, dramatically increasing bandwidth and computing pressure. Existing accelerators are primarily designed and evaluated for short contexts. They suffer from significant performance degradation when processing long contexts. To bridge this gap, we identify the major bottleneck and present a hardware accelerator for long context attention decoding via hardware-software co-design. On the software side, we propose dual-compression dynamic sparse attention. It combines ultra-low-precision quantization with feature sparsity to minimize prediction overhead. A hardware-friendly approximate Top-K selection further reduces filter complexity from $O(n \log k)$ to $O(n)$. On the hardware side, we deeply optimize compute and memory access to tackle bottlenecks from intricate interplay between sparse attention and long contexts, and establish a performance model to derive the optimal co-design scheme. The resulting hardware adopts a fully pipelined parallel architecture and achieves $O(n)$ efficiency even for long sequences. Experiments show that our design delivers $3.82\times$ speedup and $74.19\times$ energy efficiency over A100. Compared to SOTA accelerators, this is the first ASIC accelerator that efficiently supports long context inference, with at least $3.5\times$ higher throughput and $2.08\times$ better energy efficiency. 

---
# Nautile-370M: Spectral Memory Meets Attention in a Small Reasoning Model 

**Authors**: Maixent Chenebaux  

**Link**: [PDF](https://arxiv.org/pdf/2604.24809)  

**Abstract**: We present Nautile-370M, a 371-million-parameter small language model designed for efficient reasoning under strict parameter and inference budgets. Nautile-370M uses a hybrid backbone in which two SeqCond Attention (SCA) layers, a linear-time spectral sequence operator inspired by SeqCondenser, alternate with one transformer layer. This design aims to retain the long-context efficiency and state-tracking benefits of structured sequential models while preserving the expressive token-to-token routing of attention. The model was trained on a single Cloud TPU v4-64 pod slice provided through the Google TPU Research Cloud (TRC) program; the subsequent reinforcement learning stage was carried out on a single NVIDIA DGX Spark. We prove that the SCA readout mechanism can exactly retrieve any individual token from the prefix summary and can reproduce any output of softmax attention as a special case, establishing that SCA is at least as expressive as full self-attention in the continuous limit. We also describe the training data pipeline and outline a reinforcement learning stage specialized for reasoning, verification, and response quality. 

---
# Programming with Data: Test-Driven Data Engineering for Self-Improving LLMs from Raw Corpora 

**Authors**: Chenkai Pan, Xinglong Xu, Yuhang Xu, Yujun Wu, Siyuan Li, Jintao Chen, Conghui He, Jingxuan Wei, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.24819)  

**Abstract**: Reliably transferring specialized human knowledge from text into large language models remains a fundamental challenge in artificial intelligence. Fine-tuning on domain corpora has enabled substantial capability gains, but the process operates without feedback: when a model fails on a domain task, there is no method to diagnose what is deficient in the training data, and the only recourse is to add more data indiscriminately. Here we show that when a structured knowledge representation extracted from the source corpus serves as the shared foundation for both training data and evaluation, the complete data-engineering lifecycle maps onto the software development lifecycle in a precise and operative way: training data becomes source code specifying what the model should learn, model training becomes compilation, benchmarking becomes unit testing, and failure-driven data repair becomes debugging. Under this correspondence, model failures decompose into concept-level gaps and reasoning-chain breaks that can be traced back to specific deficiencies in the data and repaired through targeted patches, with each repair cycle producing consistent improvements across model scales and architectures without degrading general capabilities. We formalize this principle as Programming with Data and instantiate it across sixteen disciplines spanning the natural sciences, engineering, biomedicine, and the social sciences, releasing a structured knowledge base, benchmark suite, and training corpus as open resources. By demonstrating that the relationship between training data and model behaviour is structurally traceable and systematically repairable, this work establishes a principled foundation for the reliable engineering of human expertise into language models. 

---
# Semantic Denial of Service in LLM-controlled robots 

**Authors**: Jonathan Steinberg, Oren Gal  

**Link**: [PDF](https://arxiv.org/pdf/2604.24790)  

**Abstract**: Safety-oriented instruction-following is supposed to keep LLM-controlled robots safe. We show it also creates an availability attack surface. By injecting short safety-plausible phrases (1-5 tokens) into a robots audio channel, an adversary can trigger the models safety reasoning to halt or disrupt execution without jailbreaking the model or overriding its policy. In the embodied setting, this is a semantic denial-of-service attack: the agent stops because the injected signal looks like a legitimate alert. Across four vision-language models, seven prompt-level defenses, three deployment modes, and single- and multi-injection settings, we find that prompt-only defenses trade off attack suppression against genuine hazard response. The strongest defenses reduce hard-stop attack success on some models, but defenses change the form of disruption, not its fact: suppressed hard stops re-emerge as acknowledge loops and false alerts, which we measure with Disruption Success Rate (DSR). We further find that injection variety is consistently more effective than repeating the same phrase, suggesting that models treat diverse safety cues as corroborating evidence. The practical implication is architectural rather than prompt-level: systems that route unauthenticated audio text directly into the LLM create an avoidable security dependency between safety monitoring and action selection. 

---
# Cloud to Edge: Benchmarking LLM Inference On Hardware-Accelerated Single-Board Computers 

**Authors**: Harri Renney, Fouad Trad, Michael Mattarock, Zena Wood  

**Link**: [PDF](https://arxiv.org/pdf/2604.24785)  

**Abstract**: Large language models (LLMs) are becoming increasingly capable at small parameter scales. At the same time, conventional cloud-centric deployment introduces challenges around data privacy, latency, and cost that are acute in operational technology and defence environments. Advances in model distillation, quantisation, and affordable edge accelerators now make local LLM inference on single-board computers feasible, but the high dimensionality of the configuration space makes identifying optimal deployments difficult without structured evaluation. Existing LLM-specific edge benchmarking efforts rely on CPU-only inference, poor coverage of genuine single-board computers, and generic evaluation tasks that lack multi-dimensional assessment of hardware effectiveness. This paper proposes a multi-dimensional benchmarking methodology that jointly evaluates inference performance and hardware efficiency across four IoT-suitable edge platform configurations testing single-board computers with the latest available hardware accelerators. Our results reveal the benefits of using hardware accelerators such as NPUs and GPUs, along with multi-dimensional evaluations quantifying the trade-offs between power efficiency, physical device size and token throughput; offering practical guidance for deploying generative AI in privacy-sensitive and connectivity-limited environments such as unmanned vehicles and portable, ruggedised operations. 

---
# ITAS: A Multi-Agent Architecture for LLM-Based Intelligent Tutoring 

**Authors**: Iizalaarab Elhaimeur, Nikos Chrisochoides  

**Link**: [PDF](https://arxiv.org/pdf/2604.24808)  

**Abstract**: Large language model tutors are easy to build in a notebook and hard to run in a real course. We describe ITAS (Intelligent Teaching Assistant System), a multi-agent tutoring system that a graduate quantum computing course used for a semester at Old Dominion University. The system has three layers. The teaching layer is a Spoke-and-Wheel of three parallel specialist agents (Video, Code, Guidance) followed by a Synthesizer, plus a separate autograder that evaluates both the correctness and the approach of checkpoint submissions. The operational layer is four Cloud Run microservices with session state in Cloud SQL and interaction events streamed through Pub/Sub to BigQuery. The feedback layer is a narrow-scope conversational agent that answers instructor questions over per-lesson pseudonymized event streams, addressing what we call the Blind Instructor Problem: LLM tutors accumulate more data about students than the instructor can reach through routine channels. The architecture is a direct response to specific failures of an earlier prototype, and we describe which of those fixes carried forward and which were dropped for this iteration. We report on a pilot deployment (five students, one course, one semester) interpreted as system-behavior evidence rather than learning-outcome evidence: the teaching layer handled 334 chat turns without the task-boundary hallucinations that domain consolidation would have risked, the operational layer captured 10,628 events across five modules, and the feedback layer surfaced two findings the instructor acted on mid-semester. We do not claim the pilot generalizes. We do claim that the system as described is one workable answer to the question of what an LLM-based ITS needs to look like end-to-end to run in a real course. 

---
# Architecture Determines Observability in Transformers 

**Authors**: Thomas Carmichael  

**Link**: [PDF](https://arxiv.org/pdf/2604.24801)  

**Abstract**: Autoregressive transformers make confident errors, but activation monitoring can catch them only if the model preserves an internal signal that output confidence does not expose. This preservation is determined by architecture and training recipe. We define observability as the linear readability of per-token decision quality from frozen mid-layer activations after controlling for max-softmax confidence and activation norm. The correction is essential. Confidence controls absorb 57.7% of raw probe signal on average across 13 models in 6 families.
Observability is not a generic property of transformers. In Pythia's controlled suite, every tested run with the 24-layer, 16-head configuration collapses to rho_partial ~0.10 across a 3.5x parameter gap and two Pile variants, while six other configurations occupy a separated healthy band from 0.21 to 0.38. The output-controlled residual collapses at the same points, and neither tested nonlinear probes nor layer sweeps recover healthy-range signal. Checkpoint dynamics show the collapse is emergent during training. Both configurations at matched hidden dimension form the signal at the earliest measured checkpoint, but training erases it in the (24L, 16H) class while predictive loss continues improving.
Across independent recipes the collapse map changes but the phenomenon persists. Qwen 2.5 and Llama differ by 2.9x at matched 3B scale with probe seed distributions that do not overlap, while Mistral 7B preserves observability where Llama 3.1 8B collapses despite similar broad architecture. A WikiText-trained observer transfers to downstream QA without training on those tasks, catching errors confidence misses. At 20% flag rate, its exclusive catch rate is 10.9-13.4% of all errors in seven of nine model-task cells. Architecture selection is a monitoring decision. 

---
# From Prototype to Classroom: An Intelligent Tutoring System for Quantum Education 

**Authors**: Iizalaarab Elhaimeur, Nikos Chrisochoides  

**Link**: [PDF](https://arxiv.org/pdf/2604.24807)  

**Abstract**: Quantum computing instructors face a compounding problem: the concepts are counterintuitive, the mathematical formalism is dense, and qualified faculty are scarce outside a small number of well-resourced institutions. Our prior work introduced a knowledge-graph-augmented tutoring prototype with two specialized LLM agents: a Teaching Agent for dynamic interaction and a Lesson Planning Agent for lesson generation. Validated on simulated runs rather than in a real course, that prototype left open whether more aggressive agent specialization would be needed to handle the full range of quantum education tasks under real student load. This paper answers the three questions that the prototype could not answer. Can agent specialization solve the reliability problem in a domain as technically demanding as quantum information science? Can the system run in a real course, not a demonstration? Does the instructor gain actionable intelligence from the deployment? We present ITAS (Intelligent Teaching Assistant System), a multi-agent tutoring system built around four contributions: a five-module QIS curriculum grounded in Watrous's information-first framework, a Spoke-and-Wheel teaching architecture with quantum-specialized agents, a cloud infrastructure designed for production use and regulatory compliance, and a conversational analytics layer for instructors and content developers. Piloted in a quantum computing course at Old Dominion University, the system supports all three answers: deployment evidence is consistent with specialization addressing the task-boundary failures observed in the prototype, cloud infrastructure supports classroom-scale concurrency at sub-textbook cost, and the analytics agent surfaces curriculum gaps the instructor could not otherwise see. 

---
# Subliminal Steering: Stronger Encoding of Hidden Signals 

**Authors**: George Morgulis, John Hewitt  

**Link**: [PDF](https://arxiv.org/pdf/2604.25783)  

**Abstract**: Subliminal learning describes a student language model inheriting a behavioral bias by fine-tuning on seemingly innocuous data generated by a biased teacher model. Prior work has begun to characterize this phenomenon but leaves open questions about the scope of signals it can transfer, the mechanisms that explain it, and the precision with which a bias can be encoded by seemingly unrelated data. We tackle all three problems by introducing subliminal steering, a variant of subliminal learning in which the teacher's bias is implemented not via a system prompt, as in prior work, but through a steering vector trained to maximize the likelihood of a set of target samples. First, we show that subliminal steering transfers complex multi-word biases, whereas prior work focused on single-word preferences, demonstrating a large scope of subliminally transferrable signals. Second, we provide mechanistic evidence that subliminal learning transfers not only the target behavioral bias, but also the steering vector itself, localized to the layers at which the teacher was steered. Finally, we show that the bias is encoded with surprising precision. We train a new steering vector directly on the subliminally-laden dataset and find that it attains high cosine similarity with the original vector. 

---
# From Syntax to Emotion: A Mechanistic Analysis of Emotion Inference in LLMs 

**Authors**: Bangzhao Shu, Arinjay Singh, Mai ElSherief  

**Link**: [PDF](https://arxiv.org/pdf/2604.25866)  

**Abstract**: Large language models (LLMs) are increasingly used in emotionally sensitive human-AI applications, yet little is known about how emotion recognition is internally represented. In this work, we investigate the internal mechanisms of emotion recognition in LLMs using sparse autoencoders (SAEs). By analyzing sparse feature activations across layers, we identify a consistent three-phase information flow, in which emotion-related features emerge only in the final phase. We further show that emotion representations comprise both shared features across emotions and emotion-specific features. Using phase-stratified causal tracing, we identify a small set of features that strongly influence emotion predictions, and show that both their number and causal impact vary across emotions; in particular, Disgust is more weakly and diffusely represented than other emotions. Finally, we propose an interpretable and data-efficient causal feature steering method that significantly improves emotion recognition performance across multiple models while largely preserving language modeling ability, and demonstrate that these improvements generalize across multiple emotion recognition datasets. Overall, our findings provide a systematic analysis of the internal mechanisms underlying emotion recognition in LLMs and introduce an efficient, interpretable, and controllable approach for improving model performance. 

---
# Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses 

**Authors**: Jiahang Lin, Shichun Liu, Chengjun Pan, Lizhi Lin, Shihan Dou, Xuanjing Huang, Hang Yan, Zhenhua Han, Tao Gui  

**Link**: [PDF](https://arxiv.org/pdf/2604.25850)  

**Abstract**: Harnesses have become a central determinant of coding-agent performance, shaping how models interact with repositories, tools, and execution environments. Yet automating harness engineering is hard: a heterogeneous action space, sparse and noisy evaluation signal, multi-million-token trajectories, and edits whose effect is hard to attribute to the next round's outcomes. We introduce Agentic Harness Engineering (AHE), a framework that automates harness-level evolution by instrumenting the three stages of any engineering loop (component editing, trajectory inspection, and decision making) with matched observability pillars: (1) component observability gives every editable harness component a file-level representation so the action space is explicit and revertible; (2) experience observability distills millions of raw trajectory tokens into a layered, drill-down evidence corpus that an evolving agent can actually consume; and (3) decision observability pairs every edit with a self-declared prediction, later verified against the next round's task-level outcomes. Together, these pillars turn every edit into a falsifiable contract, so harness evolution proceeds autonomously without collapsing into trial-and-error. Empirically, ten AHE iterations lift pass@1 on Terminal-Bench 2 from 69.7% to 77.0%, surpassing the human-designed harness Codex-CLI (71.9%) and the self-evolving baselines ACE and TF-GRPO. The frozen harness transfers without re-evolution: on SWE-bench-verified it tops aggregate success at 12% fewer tokens than the seed, and on Terminal-Bench 2 it yields +5.1 to +10.1pp cross-family gains across three alternate model families, indicating the evolved components encode general engineering experience rather than benchmark-specific tuning. These results position observability-driven evolution as a practical pathway to keep coding-agent harnesses continually improving. 

---
# A paradox of AI fluency 

**Authors**: Christopher Potts, Moritz Sudhof  

**Link**: [PDF](https://arxiv.org/pdf/2604.25905)  

**Abstract**: How much does a user's skill with AI shape what AI actually delivers for them? This question is critical for users, AI product builders, and society at large, but it remains underexplored. Using a richly annotated sample of 27K transcripts from WildChat-4.8M, we show that fluent users take on more complex tasks than novices and adopt a fundamentally different interactional mode: they iterate collaboratively with the AI, refining goals and critically assessing outputs, whereas novices take a passive stance. These differences lead to a paradox of AI fluency: fluent users experience more failures than novices -- but their failures tend to be visible (a direct consequence of their engagement), they are more likely to lead to partial recovery, and they occur alongside greater success on complex tasks. Novices, by contrast, more often experience invisible failures: conversations that appear to end successfully but in fact miss the mark. Taken together, these results reframe what success with AI depends on. Individuals should adopt a stance of active engagement rather than passive acceptance. AI product builders should recognize that they are designing not just model behavior but user behavior; encouraging deep engagement, rather than friction-free experiences, will lead to more success overall. Our code and data are available at this https URL 

---
# DV-World: Benchmarking Data Visualization Agents in Real-World Scenarios 

**Authors**: Jinxiang Meng, Shaoping Huang, Fangyu Lei, Jingyu Guo, Haoxiang Liu, Jiahao Su, Sihan Wang, Yao Wang, Enrui Wang, Ye Yang, Hongze Chai, Jinming Lv, Anbang Yu, Huangjing Zhang, Yitong Zhang, Yiming Huang, Zeyao Ma, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25914)  

**Abstract**: Real-world data visualization (DV) requires native environmental grounding, cross-platform evolution, and proactive intent alignment. Yet, existing benchmarks often suffer from code-sandbox confinement, single-language creation-only tasks, and assumption of perfect intent. To bridge these gaps, we introduce DV-World, a benchmark of 260 tasks designed to evaluate DV agents across real-world professional lifecycles. DV-World spans three domains: DV-Sheet for native spreadsheet manipulation including chart and dashboard creation as well as diagnostic repair; DV-Evolution for adapting and restructuring reference visual artifacts to fit new data across diverse programming paradigms and DV-Interact for proactive intent alignment with a user simulator that mimics real-world ambiguous requirements. Our hybrid evaluation framework integrates Table-value Alignment for numerical precision and MLLM-as-a-Judge with rubrics for semantic-visual assessment. Experiments reveal that state-of-the-art models achieve less than 50% overall performance, exposing critical deficits in handling the complex challenges of real-world data visualization. DV-World provides a realistic testbed to steer development toward the versatile expertise required in enterprise workflows. Our data and code are available at \href{this https URL}{this project page}. 

---
# Progressing beyond Art Masterpieces or Touristic Clichés: how to assess your LLMs for cultural alignment? 

**Authors**: António Branco, João Silva, Nuno Marques, Luis Gomes, Ricardo Campos, Raquel Sequeira, Sara Nerea, Rodrigo Silva, Miguel Marques, Rodrigo Duarte, Artur Putyato, Diogo Folques, Tiago Valente  

**Link**: [PDF](https://arxiv.org/pdf/2604.25654)  

**Abstract**: Although the cultural (mis)alignment of Large Language Models (LLMs) has attracted increasing attention -- often framed in terms of cultural bias -- until recently there has been limited work on the design and development of datasets for cultural assessment. Here, we review existing approaches to such datasets and identify their main limitations. To address these issues, we propose design guidelines for annotators and report on the construction of a dataset built according to these principles. We further present a series of contrastive experiments conducted with this dataset. The results demonstrate that our design yields test sets with greater discriminative power, effectively distinguishing between models specialized for a given culture and those that are not, ceteris paribus. 

---
# Bye Bye Perspective API: Lessons for Measurement Infrastructure in NLP, CSS and LLM Evaluation 

**Authors**: David Hartmann, Manuel Tonneau, Angelie Kraft, LK Seiling, Dimitri Staufer, Pieter Delobelle, Jan Fillies, Anna Ricarda Luther, Jan Batzner, Mareike Lisker  

**Link**: [PDF](https://arxiv.org/pdf/2604.25580)  

**Abstract**: The closure of Perspective API at the end of 2026 discards what has functioned as the de facto standard for automated toxicity measurement in NLP, CSS, and LLM evaluation research. We document the structural dependence that the communities built on this single proprietary tool and discuss how this dependence caused epistemic problems that have affected - and will likely continue to affect - collective research efforts. Perspective's model was periodically updated without versioning or disclosure, its annotation structure reflected a single corporate operationalisation of a contested concept, and its scores were used simultaneously as an evaluation target and an evaluation standard. Its closure leaves behind non-updatable benchmarks, irreproducible results, and ultimately a field at risk of perpetuating these issues by turning to closed-source LLMs. We use Perspective's announced termination as an opportunity to call for an independent, valid, adaptable, and reproducible toxicity and hate speech measurement infrastructure, with the technical and governance requirements outlined in this paper. 

---
# From Chatbots to Confidants: A Cross-Cultural Study of LLM Adoption for Emotional Support 

**Authors**: Natalia Amat-Lefort, Mert Yazan, Amanda Cercas Curry, Flor Miriam Plaza-del-Arco  

**Link**: [PDF](https://arxiv.org/pdf/2604.25525)  

**Abstract**: Large Language Models (LLMs) are increasingly used not only for instrumental tasks, but as always-available and non-judgmental confidants for emotional support. Yet what drives adoption and how users perceive emotional support interactions across countries remains unknown. To address this gap, we present the first large-scale cross-cultural study of LLM use for emotional support, surveying 4,641 participants across seven countries (USA, UK, Germany, France, Spain, Italy, and The Netherlands). Our results show that adoption rates vary dramatically across countries (from 20% to 59%). Using mixed models that separate cultural effects from demographic composition, we find that: Being aged 25-44, religious, married, and of higher socioeconomic status are predictors of positive perceptions (trust, usage, perceived benefits), with socioeconomic status being the strongest. English-speaking countries consistently show more positive perceptions than Continental European countries. We further collect a corpus of 731 real multilingual prompts from user interactions, showing that users mainly seek help for loneliness, stress, relationship conflicts, and mental health struggles. Our findings reveal that LLM emotional support use is shaped by a complex sociotechnical landscape and call for a broader research agenda examining how these systems can be developed, deployed, and governed to ensure safe and informed access. 

---
# Navigating Global AI Regulation: A Multi-Jurisdictional Retrieval-Augmented Generation System 

**Authors**: Courtney Ford, Ojas Rane, Susan Leavy  

**Link**: [PDF](https://arxiv.org/pdf/2604.25448)  

**Abstract**: Navigating AI regulation across jurisdictions is increasingly difficult for policymakers, legal professionals, and researchers. To address this, we present a multi-jurisdictional Retrieval-Augmented Generation system for global AI regulation. Our corpus includes 242 documents across 68 jurisdictions, ranging from formal legislation like the EU AI Act to unstructured policy documents such as national AI strategies. The system makes three technical contributions: type-specific chunking that preserve legal structure across heterogenous documents; conditional retrieval routing with entity detection and metadata for legal citations; and priority-based re-ranking to boost enacted legislation over policy and secondary sources. Evaluation of 50 queries reveals strong performance across both single-entity and multi-jurisdictional questions, achieving 0.87 average faithfulness and 0.84 average answer relevancy. Single-entity queries achieve 0.86 average faithfulness and 0.92 average answer relevancy, while multi-jurisdictional comparison queries achieve 0.88 average faithfulness and 0.75 average answer relevancy. These findings highlight the effectiveness of domain-specific retrieval strategies for navigating complex, heterogenous regulatory corpora. 

---
# One Refiner to Unlock Them All: Inference-Time Reasoning Elicitation via Reinforcement Query Refinement 

**Authors**: Yixiao Zhou, Dongzhou Cheng, zhiliang wu, Yi Yang, Yu Cheng, Hehe Fan  

**Link**: [PDF](https://arxiv.org/pdf/2604.25444)  

**Abstract**: Large Language Models (LLMs) often fail to utilize their latent reasoning capabilities due to a distributional mismatch between ambiguous human inquiries and the structured logic required for machine activation. Existing alignment methods either incur prohibitive $O(N)$ costs by fine-tuning each model individually or rely on static prompts that fail to resolve query-level structural complexity. In this paper, we propose ReQueR (\textbf{Re}inforcement \textbf{Que}ry \textbf{R}efinement), a modular framework that treats reasoning elicitation as an inference-time alignment task. We train a specialized Refiner policy via Reinforcement Learning to rewrite raw queries into explicit logical decompositions, treating frozen LLMs as the environment. Rooted in the classical Zone of Proximal Development from educational psychology, we introduce the Adaptive Solver Hierarchy, a curriculum mechanism that stabilizes training by dynamically aligning environmental difficulty with the Refiner's evolving competence. ReQueR yields consistent absolute gains of 1.7\%--7.2\% across diverse architectures and benchmarks, outperforming strong baselines by 2.1\% on average. Crucially, it provides a promising paradigm for one-to-many inference-time reasoning elicitation, enabling a single Refiner trained on a small set of models to effectively unlock reasoning in diverse unseen models. Code is available at this https URL. 

---
# Learning from Medical Entity Trees: An Entity-Centric Medical Data Engineering Framework for MLLMs 

**Authors**: Jianghang Lin, Haihua Yang, Deli Yu, Kai Wu, Kai Ye, Jinghao Lin, Zihan Wang, Yuhang Wu, Liujuan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.25296)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown transformative potential in medical applications, yet their performance is hindered by conventional data curation strategies that rely on coarse-grained partitioning by modality or department. Such fragmented approaches fail to capture the hierarchical and interconnected nature of clinical medical knowledge, limiting the models' ability to perform fine-grained recognition and complex reasoning. In this paper, we propose a novel Entity-Centric Medical Data Engineering framework. We automatically extract entities from authoritative medical literature to construct a Medical Entity Tree (MET), a hierarchical structure that systematically encodes diseases, anatomical structures, modalities, and symptoms into a unified knowledge repository. Building upon the MET, we propose an advanced data engine that includes: (1) node-guided retrieval to anchor raw data to specific medical concepts, (2) a two-stage hybrid filtering and alignment pipeline to ensure precise visual-semantic correspondence, and (3) knowledge-aware data synthesis to generate enriched captions and targeted reasoning VQA pairs, leveraging structural constraints. Extensive evaluations across six medical benchmarks demonstrate that our approach significantly enhances the medical capabilities of general-purpose MLLMs, improving their ability to handle complex clinical queries and achieve state-of-the-art performance in diverse medical contexts. 

---
# FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments 

**Authors**: Amir Saeidi, Venkatesh Mishra, Souradeep Mukhopadhyay, Gaowen Liu, Ali Payani, Jayanth Srinivasa, Chitta Baral  

**Link**: [PDF](https://arxiv.org/pdf/2604.25135)  

**Abstract**: Large Language Models are being increasingly deployed as the decision-making core of autonomous agents capable of effecting change in external environments. Yet, in conversational benchmarks, which simulate real-world customer-centric issue resolution scenarios, these agents frequently fail due to the cascading effects of incorrect decision-making. These challenges are particularly pronounced for open-source LLMs with smaller parameter sizes, limited context windows, and constrained inference budgets, which contribute to increased error accumulation in agentic settings. To tackle these challenges, we present the Failure-Aware Meta-Agentic (FAMA) framework. FAMA operates in two stages: first, it analyzes failure trajectories from baseline agents to identify the most prevalent errors; second, it employs an orchestration mechanism that activates a minimal subset of specialized agents tailored to address these failures by injecting a targeted context for the tool-use agent before the decision-making step. Experiments across open-source LLMs demonstrate performance gains up to 27% across evaluation modes over standard baselines. These results highlight that targeted curation of context through specialized agents to address common failures is a valuable design principle for building reliable, multi-turn tool-use LLM agents that simulate real-world conversational scenarios. 

---
# What Makes Good Instruction-Tuning Data? An In-Context Learning Perspective 

**Authors**: Guangzeng Han, Xiaolei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25132)  

**Abstract**: Instruction-tuning datasets often contain substantial redundancy and low-quality samples, necessitating effective data selection methods. We propose an instruction data selection framework based on weighted in-context influence (wICI), which measures how effectively each candidate example reduces instruction-following difficulty for semantically related peers. Through systematic experiments, we address three key questions: what constitutes effective instruction tuning data from an in-context perspective, whether sample difficulty correlates with in-context influence, and how in-context influence translates to instruction tuning effectiveness. Experiments across multiple models and benchmarks demonstrate that our method consistently outperforms existing baselines under constrained data budgets, while empirically showing that sample difficulty negatively correlates with in-context influence. 

---
# Diagnosis, Bad Planning & Reasoning. Treatment, SCOPE -- Planning for Hybrid Querying over Clinical Trial Data 

**Authors**: Suparno Roy Chowdhury, Manan Roy Choudhury, Tejas Anvekar, Muhammad Ali Khan, Kaneez Zahra Rubab Khakwani, Mohamad Bassam Sonbol, Irbaz Bin Riaz, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.25120)  

**Abstract**: We study clinical trial table reasoning, where answers are not directly stored in visible cells but must be reasoned from semantic understanding through normalization, classification, extraction, or lightweight domain reasoning. Motivated by the observation that current LLM approaches often suffer from "bad reasoning" under implicit planning assumptions, we focus on settings in which the model must recover implicit attributes such as therapy type, added agents, endpoint roles, or follow-up status from partially observed clinical-trial tables. We propose SCOPE (Structured Clinical hybrid Planning for Evidence retrieval in clinical trials), a multi-LLM planner-based framework that decomposes the task into row selection, structured planning, and execution. The planner makes the source field, reasoning rules, and output constraints explicit before answer generation, reducing ambiguity relative to direct prompting. We evaluate SCOPE on 1,500 hybrid reasoning questions over oncology clinical-trial tables against zero-shot, few-shot, chain-of-thought, TableGPT2, Blend-SQL, and EHRAgent. Results show that explicit multi-LLM planning improves accuracy for reasoning-based questions while offering a stronger accuracy-efficiency tradeoff than heavier agentic baselines. Our findings position clinical trial reasoning as a distinct table understanding problem and highlight hybrid planner-based decomposition as an effective solution 

---
# The Dynamics of Delusion: Modeling Bidirectional False Belief Amplification in Human-Chatbot Dialogue 

**Authors**: Ashish Mehta, Jared Moore, Jacy Reese Anthis, William Agnew, Eric Lin, Peggy Yin, Desmond C. Ong, Nick Haber, Carol Dweck  

**Link**: [PDF](https://arxiv.org/pdf/2604.25096)  

**Abstract**: There is growing concern that AI chatbots might fuel delusional beliefs in users. Some have suggested that humans and chatbots mutually reinforce false beliefs over time, but quantitative evidence is lacking. Using a unique dataset of chat logs from individuals who exhibited delusional thinking, we developed a latent state model that captures accumulating and decaying influences between humans and chatbots. We find that a bidirectional influence model substantially outperforms a unidirectional alternative where humans are the primary driver of delusion. We find that humans exert strong but short-lived influence on chatbots, whereas chatbots exert longer-lasting influence on humans. Moreover, chatbots exert strong, stable self-influence over their own future outputs that tends to perpetuate delusions over long stretches of conversation. In fact, this chatbot self-influence constituted the dominant pathway when considering accumulated influence over time. Overall, these results indicate that humans tend to drive sharp, immediate increases in delusion, whereas chatbots sustain and propagate these effects over longer timescales. Together, these findings provide the first quantitative evidence that human-chatbot interactions can form feedback loops of delusion, decomposable into distinct pathways with dissociable temporal dynamics. By doing so, they can inform the development of safer AI systems. 

---
# Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models 

**Authors**: Dan Shi, Zhuowen Han, Simon Ostermann, Renren Jin, Josef van Genabith, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.25011)  

**Abstract**: Reinforcement learning (RL)-based post-training often improves the reasoning performance of large language models (LLMs) beyond the training domain, while supervised fine-tuning (SFT) frequently leads to general capabilities forgetting. However, the mechanisms underlying this contrast remain unclear. To bridge this gap, we present a feature-level mechanistic analysis methodology to probe RL generalization using a controlled experimental setup, where RL- and SFT-tuned models are trained from the same base model on identical data. Leveraging our interpretability framework, we align internal activations across models within a shared feature space and analyze how features evolve during post-training. We find that SFT rapidly introduces many highly specialized features that stabilize early in training, whereas RL induces more restrained and continually evolving feature changes that largely preserve base models' representations. Focusing on samples where RL succeeds but the base model fails, we identify a compact, task-agnostic set of features that directly mediate generalization across diverse tasks. Feature-level interventions confirm their causal role: disabling these features significantly degrades RL models' generalization performance, while amplifying them improves base models' performance. The code is available at this https URL. 

---
# Dont Stop Early: Scalable Enterprise Deep Research with Controlled Information Flow and Evidence-Aware Termination 

**Authors**: Prafulla Kumar Choubey, Kung-Hsiang Huang, Pranav Narayanan Venkit, Jiaxin Zhang, Vaibhav Vats, Yu Li, Xiangyu Peng, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24978)  

**Abstract**: Enterprise deep research often fails to produce decision-ready reports due to uneven information coverage, context explosion, and premature stopping. We propose a scalable Enterprise Deep Research (EDR) architecture to address these failures. Our system (i) decomposes requests into coverage-driven objectives via outline generation with reflection, (ii) localizes context with dependency-guided execution and explicit information sharing, and (iii) enforces evidence-based completion criteria so agents iteratively collect information until sufficiency conditions are met. We evaluate on an internal sales enablement task and the public DeepResearch Bench benchmark, where our proposed system design achieves the strongest overall performance compared with competitive deep-research baselines. The results show that dependency-controlled context and explicit evidence sufficiency criteria reduce premature stopping and improve the consistency and depth of enterprise research outputs. 

---
# Dynamic Decision Learning: Test-Time Evolution for Abnormality Grounding in Rare Diseases 

**Authors**: Jun Li, Mingxuan Liu, Jiazhen Pan, Che Liu, Wenjia Bai, Cosmin I. Bercea, Julia A. Schnabel  

**Link**: [PDF](https://arxiv.org/pdf/2604.24972)  

**Abstract**: Clinical abnormality grounding for rare diseases is often hindered by data scarcity, making supervised fine-tuning impractical and single-pass inference highly unstable. We propose Dynamic Decision Learning (DDL), a framework that enables frozen large vision-language models (LVLMs) to refine their decisions across both language and visual spaces by optimizing instructions and consolidating predictions under visual perturbations. This process improves localization quality and produces a consensus-based reliability score that quantifies model confidence. Results on brain imaging benchmarks, including a rare-disease dataset with 281 pathology types across models ranging from 3B to 72B parameters, show that DDL improves mAP@75 by up to 105% on rare-disease cases and outperforms adaptation baselines and supervised fine-tuning. Furthermore, DDL demonstrates stronger calibration between reliability scores and localization accuracy under severe distribution shifts and increasing task difficulty. Code is available at: this https URL 

---
# Elderly-Contextual Data Augmentation via Speech Synthesis for Elderly ASR 

**Authors**: Minsik Lee, Seoi Hong, Chongmin Lee, Sieun Choi, Jian Kim, Jua Han, Jihie Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.24770)  

**Abstract**: Despite recent progress in automatic speech recognition (ASR), elderly ASR (EASR) remains challenging due to limited training data and the distinct acoustic and linguistic characteristics of elderly speech. In this work, we address data scarcity in EASR through a data augmentation pipeline that combines large language model (LLM)-based transcript paraphrasing with text-to-speech (TTS) synthesis. Given an elderly speech dataset, the LLM first generates elderly-contextual paraphrases of the original transcripts, and the TTS model then synthesizes corresponding speech using elderly reference speakers. The resulting synthetic audio-text pairs are merged with the original data to fine-tune Whisper without architectural modification. We further analyze the effects of augmentation ratio and reference-speaker composition in low-resource EASR. Experiments on English and Korean elderly speech datasets from speakers aged 70 and above show that the proposed method consistently improves performance over conventional augmentation baselines, achieving up to a 58.2% reduction in word error rate (WER) compared with the Whisper baseline. 

---
# A Survey on LLM-based Conversational User Simulation 

**Authors**: Bo Ni, Leyao Wang, Yu Wang, Branislav Kveton, Franck Dernoncourt, Yu Xia, Hongjie Chen, Reuben Leura, Samyadeep Basu, Subhojyoti Mukherjee, Puneet Mathur, Nesreen Ahmed, Junda Wu, Li Li, Huixin Zhang, Ruiyi Zhang, Tong Yu, Sungchul Kim, Jiuxiang Gu, Zhengzhong Tu, Alexa Siu, Zichao Wang, David Seunghyun Yoon, Nedim Lipka, Namyong Park, Zihao Lin, Trung Bui, Yue Zhao, Tyler Derr, Ryan A. Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24977)  

**Abstract**: User simulation has long played a vital role in computer science due to its potential to support a wide range of applications. Language, as the primary medium of human communication, forms the foundation of social interaction and behavior. Consequently, simulating conversational behavior has become a key area of study. Recent advancements in large language models (LLMs) have significantly catalyzed progress in this domain by enabling high-fidelity generation of synthetic user conversation. In this paper, we survey recent advancements in LLM-based conversational user simulation. We introduce a novel taxonomy covering user granularity and simulation objectives. Additionally, we systematically analyze core techniques and evaluation methodologies. We aim to keep the research community informed of the latest advancements in conversational user simulation and to further facilitate future research by identifying open challenges and organizing existing work under a unified framework. 

---
# CroSearch-R1: Better Leveraging Cross-lingual Knowledge for Retrieval-Augmented Generation 

**Authors**: Rui Qi, Fengran Mo, Sijin Lu, Yufeng Chen, Jian-Yun Nie, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25182)  

**Abstract**: A multilingual collection may contain useful knowledge in other languages to supplement and correct the facts in the original language for Retrieval-Augmented Generation (RAG). However, the vanilla approach that simply concatenates multiple pieces of knowledge from different languages into the context may fail to improve effectiveness due to the potential disparities across languages. To better leverage multilingual knowledge, we propose CroSearch-R1, a search-augmented reinforcement learning framework to integrate multilingual knowledge into the Group Relative Policy Optimization (GRPO) process. In particular, the approach adopts a multi-turn retrieval strategy with cross-lingual knowledge integration to dynamically align the knowledge from other languages as supplementary evidence into a unified representation space. Furthermore, we introduce a multilingual rollout mechanism to optimize reasoning transferability across languages. Experimental results demonstrate that our framework effectively leverages cross-lingual complementarity and improves the effectiveness of RAG with multilingual collections. 

---
# Independent-Component-Based Encoding Models of Brain Activity During Story Comprehension 

**Authors**: Kamya Hari, Taha Binhuraib, Jin Li, Cory Shain, Anna A. Ivanova  

**Link**: [PDF](https://arxiv.org/pdf/2604.24942)  

**Abstract**: Encoding models provide a powerful framework for linking continuous stimulus features to neural activity; however, traditional voxelwise approaches are limited by measurement noise, inter-subject variability, and redundancy arising from spatially correlated voxels encoding overlapping neural signals. Here, we propose an independent component (IC)-based encoding framework that dissociates stimulus-driven and noise-driven signals in fMRI data. We decompose continuous fMRI data from naturalistic story listening into ICs using one subset of the data, and train encoding models on independent data to predict IC time series from large language model representations of linguistic input. Across subjects, a subset of ICs exhibited consistently high predictivity. These ICs were spatially and temporally consistent across subjects and included cognitive networks known to respond during story listening (auditory and language). Auditory component time series were strongly correlated with acoustic stimulus features, highlighting the interpretability of identified component time series. Components identified as noise or motion-related artifacts by ICA-AROMA showed uniformly poor predictive performance, confirming that highly predicted components reflect genuine stimulus-related neural signals rather than confounds. Overall, IC-based encoding models enable analyses at the level of functional networks, accommodating the variability in network locations across individuals and providing interpretable results that are easy to compare across subjects. 

---
# LongSumEval: Question-Answering Based Evaluation and Feedback-Driven Refinement for Long Document Summarization 

**Authors**: Huyen Nguyen, Haoxuan Zhang, Yang Zhang, Haihua Chen, Junhua Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.25130)  

**Abstract**: Evaluating long document summaries remains the primary bottleneck in summarization research. Existing metrics correlate weakly with human judgments and produce aggregate scores without explaining deficiencies or guiding improvement, preventing effective refinement in applications requiring verifiable accuracy. We introduce LongSumEval, a unified framework bridging evaluation and generation through structured question-answering feedback. The framework operationalizes summary quality as answerability and factual alignment of question-answer pairs, generating interpretable scores and actionable feedback that identifies coverage gaps and factual inconsistencies. This resolves the misalignment where evaluation operates independently of generation objectives. Meta-evaluation of our QA-based evaluation module across seven benchmarks demonstrates substantially stronger agreement with human judgments compared to established metrics. Structured feedback enables significant quality improvements through self-refinement without retraining. By demonstrating that evaluation feedback can serve as executable instructions for generation, this work establishes a generalizable paradigm for aligning assessment with improvement, with direct implications for controllable text generation requiring verifiable accuracy and transparent quality control. All code and datasets will be released in GitHub for reproducibility. 

---
# Toward Multimodal Conversational AI for Age-Related Macular Degeneration 

**Authors**: Ran Gu, Benjamin Hou, Mélanie Hébert, Asmita Indurkar, Yifan Yang, Emily Y. Chew, Tiarnán D. L. Keenan, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25720)  

**Abstract**: Despite strong performance of deep learning models in retinal disease detection, most systems produce static predictions without clinical reasoning or interactive explanation. Recent advances in multimodal large language models (MLLMs) integrate diagnostic predictions with clinically meaningful dialogue to support clinical decision-making and patient counseling. In this study, OcularChat, an MLLM, was fine-tuned from Qwen2.5-VL using simulated patient-physician dialogues to diagnose age-related macular degeneration (AMD) through visual question answering on color fundus photographs (CFPs). A total of 705,850 simulated dialogues paired with 46,167 CFPs were generated to train OcularChat to identify key AMD features and produce reasoned predictions. OcularChat demonstrated strong classification performance in AREDS, achieving accuracies of 0.954, 0.849, and 0.678 for the three diagnostic tasks: advanced AMD, pigmentary abnormalities, and drusen size, significantly outperforming existing MLLMs. On AREDS2, OcularChat remained the top-performing method on all tasks. Across three independent ophthalmologist graders, OcularChat achieved higher mean scores than a strong baseline model for advanced AMD (3.503 vs. 2.833), pigmentary abnormalities (3.272 vs. 2.828), drusen size (3.064 vs. 2.433), and overall impression (2.978 vs. 2.464) on a 5-point clinical grading rubric. Beyond strong objective performance in AMD severity classification, OcularChat demonstrated the ability to provide diagnostic reasoning, clinically relevant explanations, and interactive dialogue, with high performance in subjective ophthalmologist evaluation. These findings suggest that MLLMs may enable accurate, interpretable, and clinically useful image-based diagnosis and classification of AMD. 

---
# Barriers to Universal Reasoning With Transformers (And How to Overcome Them) 

**Authors**: Oliver Kraus, Yash Sarrof, Yuekun Yao, Alexander Koller, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2604.25800)  

**Abstract**: Chain-of-Thought (CoT) has been shown to empirically improve Transformers' performance, and theoretically increase their expressivity to Turing completeness. However, whether Transformers can learn to generalize to CoT traces longer than those seen during training is understudied. We use recent theoretical frameworks for Transformer length generalization and find that -- under standard positional encodings and a finite alphabet -- Transformers with CoT cannot solve problems beyond $TC^0$, i.e. the expressivity benefits do not hold under the stricter requirement of length-generalizable learnability. However, if we allow the vocabulary to grow with problem size, we attain a length-generalizable simulation of Turing machines where the CoT trace length is linear in the simulated runtime up to a constant. Our construction overcomes two core obstacles to reliable length generalization: repeated copying and last-occurrence retrieval. We assign each tape position a unique signpost token, and log only value changes to enable recovery of the current tape symbol through counts circumventing both barriers. Further, we empirically show that the use of such signpost tokens and value change encodings provide actionable guidance to improve length generalization on hard problems. 

---
# VLM Judges Can Rank but Cannot Score: Task-Dependent Uncertainty in Multimodal Evaluation 

**Authors**: Divake Kumar, Sina Tayebati, Devashri Naik, Ranganath Krishnan, Amit Ranjan Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2604.25235)  

**Abstract**: Vision-language models (VLMs) are increasingly used as automated judges for multimodal systems, yet their scores provide no indication of reliability. We study this problem through conformal prediction, a distribution-free framework that converts a judge's point score into a calibrated prediction interval using only score-token log-probabilities, with no retraining. We present the first systematic analysis of conformal prediction for VLM-as-a-Judge across 3 judges and 14 visual task categories. Our results show that evaluation uncertainty is strongly task-dependent: intervals cover ~40% of the score range for aesthetics and natural images but expand to ~70% for chart and mathematical reasoning, yielding a quantitative reliability map for multimodal evaluation. We further identify a failure mode not captured by standard evaluation metrics, ranking-scoring decoupling, where judges achieve high ranking correlation while producing wide, uninformative intervals, correctly ordering responses but failing to assign reliable absolute scores. Finally, we show that interval width is driven primarily by task difficulty and annotation quality, i.e., the same judge and method yield 4.5x narrower intervals on a clean, multi-annotator captioning benchmark. Code: this https URL 

---
# MGTEVAL: An Interactive Platform for Systemtic Evaluation of Machine-Generated Text Detectors 

**Authors**: Yuanfan Li, Qi Zhou, Chengzhengxu Li, Zhaohan Zhang, Chenxu Zhao, Zepu Ruan, Chao Shen, Xiaoming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25152)  

**Abstract**: We present MGTEVAL, an extensible platform for systematic evaluation of Machine-Generated Text (MGT) detectors. Despite rapid progress in MGT detection, existing evaluations are often fragmented across datasets, preprocessing, attacks, and metrics, making results hard to compare and reproduce. MGTEVAL organizes the workflow into four components: Dataset Building, Dataset Attack, Detector Training, and Performance Evaluation. It supports constructing custom benchmarks by generating MGT with configurable LLMs, applying 12 text attacks to test sets, training detectors via a unified interface, and reporting effectiveness, robustness, and efficiency. The platform provides both command-line and Web-based interfaces for user-friendly experimentation without code rewriting. 

---
# Odysseys: Benchmarking Web Agents on Realistic Long Horizon Tasks 

**Authors**: Lawrence Keunho Jang, Jing Yu Koh, Daniel Fried, Ruslan Salakhutdinov  

**Link**: [PDF](https://arxiv.org/pdf/2604.24964)  

**Abstract**: Existing web agent benchmarks have largely converged on short, single-site tasks that frontier models are approaching saturation on. However, real world web use consists of long-horizon, multi-site workflows. Common web navigation tasks, such as comparing products across different domains, planning trips across multiple services, or summarizing information from multiple search queries, require sustained context and cross-site reasoning over potentially hours of browsing. To capture and evaluate such behaviors, we introduce Odysseys: a benchmark of 200 long-horizon web tasks derived from real world browsing sessions evaluated on the live Internet. We find that binary pass/fail evaluation is inadequate for long-horizon settings and introduce a rubric-based evaluation, annotating each Odysseys task with an average of 6.1 graded rubrics. We demonstrate that this yields higher agreement with humans and provides a more fine-grained signal than commonly used trajectory-level LLM-as-a-judge evaluation metrics. We tested several leading frontier models and find that the strongest models achieve a success rate of 44.5%, which leaves substantial room for future improvements. Beyond task success, we argue that efficiency is a first-class concern for long-horizon agents. We introduce a Trajectory Efficiency metric (rubric score per step) and find that even frontier agents achieve only 1.15%, marking an evident need for agents that can succeed efficiently and not simply eventually. Odysseys isolates the critical evaluation of long-horizon proficiency in open-web environments, providing a realistic benchmark to measure progress towards computer-use agents that can potentially productively operate for hours. We release our tasks, evaluation scripts, and other results at this https URL 

---
# PolyKV: A Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference 

**Authors**: Ishan Patel, Ishan Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24971)  

**Abstract**: We present PolyKV, a system in which multiple concurrent inference agents share a single, asymmetrically compressed KV cache pool. Rather than allocating a separate KV cache per agent -- the standard paradigm -- PolyKV writes a compressed cache once and injects it into N independent agent contexts via HuggingFace DynamicCache objects. Compression is asymmetric: Keys are quantized at int8 (q8_0) to preserve softmax stability, while Values are compressed using TurboQuant MSE -- a Fast Walsh-Hadamard Transform (FWHT) rotation followed by 3-bit Lloyd-Max quantization with centroids tuned to N(0,1). We evaluate across two model scales (SmolLM2-1.7B-Instruct and Llama-3-8B-Instruct), three context lengths (600-7,194 tokens), and up to 15 concurrent agents. PolyKV achieves a stable 2.91x compression ratio across all configurations. On Llama-3-8B with 15 agents sharing a 4K-token context, PolyKV reduces KV cache memory from 19.8 GB to 0.45 GB -- a 97.7% reduction -- while maintaining only +0.57% perplexity degradation and a mean BERTScore F1 of 0.928. PPL delta does not grow with agent count and improves as context length increases, inverting to -0.26% at 1,851 coherent tokens. To our knowledge, no prior work combines a single shared, lossy-compressed KV pool with multi-reader concurrent agent access. 

---
# The Surprising Universality of LLM Outputs: A Real-Time Verification Primitive 

**Authors**: Alex Bogdan, Adrian de Valois-Franklin  

**Link**: [PDF](https://arxiv.org/pdf/2604.25634)  

**Abstract**: We report a striking statistical regularity in frontier LLM outputs that enables a CPU-only scoring primitive running
at 2.6 microseconds per token, with estimated latency up to 100,000$\times$ (five orders of magnitude) below existing
sampling-based detectors. Across six contemporary models from five independent vendors, two generation sizes, and five
held-out domains, token rank-frequency distributions converge to the same two-parameter Mandelbrot ranking
distribution, with 34 of 36 model-by-domain fits exceeding $R^{2} = 0.94$ and 35 of 36 favoring Mandelbrot over Zipf
by AIC. The shared family does not collapse the models into statistical duplicates. Fitted Mandelbrot parameters
remain cleanly separable between models: the cross-model spread in $q$ (1.63 to 3.69) exceeds its per-model bootstrap
standard deviation (0.03 to 0.10) by more than an order of magnitude, yielding tens of standard deviations of
separation per few thousand output tokens. Two capabilities follow. First, statistical model fingerprinting: text from
a vendor-delivered LLM can be tested against its claimed model family without cryptographic watermarks or access to
model internals, supporting provenance verification and silent-substitution audits. Second, a model-agnostic reference
distribution for black-box output assessment, from which we derive a single-pass scoring primitive that composes with
model log probabilities when available and degrades to a rank-only mode usable on closed APIs. Pilot results on
FRANK, TruthfulQA, and HaluEval map where the primitive helps (lexical anomalies, unsupported entities) and where it
structurally cannot (reasoning errors in domain-appropriate vocabulary). We position the primitive as a first-pass
triage layer in compound evaluation stacks, not as a replacement for sampling-based or source-conditioned verifiers. 

---
# DRAGON: A Benchmark for Evidence-Grounded Visual Reasoning over Diagrams 

**Authors**: Anirudh Iyengar Kaniyar Narayana Iyengar, Tampu Ravi Kumar, Gaurav Najpande, Manan Suri, Dinesh Manocha, Puneet Mathur, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.25231)  

**Abstract**: Diagram question answering (DQA) requires models to interpret structured visual representations such as charts, maps, infographics, circuit schematics, and scientific diagrams. Recent vision-language models (VLMs) often achieve high answer accuracy on these tasks, yet correct answers do not guarantee that models ground their reasoning in the diagram regions that support the prediction. Models may instead rely on textual correlations or dataset artifacts without identifying the visual evidence required to verify the answer. This limitation prevents reliable evaluation of diagram reasoning and reduces interpretability. We introduce DRAGON, a benchmark for evaluating evidence-grounded visual reasoning in diagrams. Given a diagram, a question, and the correct answer, a model must predict bounding boxes that correspond to the visual elements required to justify the answer. These evidence regions may include answer-bearing components, textual labels, legends, axes, connectors, and other supporting structures involved in the reasoning process. The DRAGON dataset contains 11,664 annotated question instances collected from six diagram QA datasets: ChartQA, Circuit-VQA, InfographicsVQA, MapIQ, MapWise, and AI2D. We release a 2,445-instance benchmark test set with human-verified reasoning evidence annotations and a standardized evaluation framework. We evaluate eight recent VLMs and analyze their ability to localize reasoning evidence across diverse diagram domains. DRAGON enables systematic evaluation of diagram reasoning and supports future research on models that ground their predictions in visual evidence. 

---
# Intrinsic Mutual Information as a Modulator for Preference Optimization 

**Authors**: Peng Liao, Peijia Zheng, Lingbo Li, Shangsong Liang, Lin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.24804)  

**Abstract**: Offline preference optimization methods, such as Direct Preference Optimization (DPO), offer significant advantages in aligning Large Language Models (LLMs) with human values. However, achieving optimal performance with these methods typically involves additional hyperparameter tuning, resulting in substantial time overhead. Although prior work has proposed a range of improvements, these methods remain limited in effectiveness and have not fully eliminated reliance on hyperparameter tuning. In this work, we propose RMiPO, a lightweight and efficient framework for offline preference optimization. RMiPO leverages intrinsic Response-level Mutual information for Preference Optimization with hyperparameter modulation, dynamically decoupling preference contributions at negligible additional computational cost. Extensive experimental results demonstrate that RMiPO achieves consistently superior performance over existing methods while reducing training overhead by more than 15\%. Our code is available at this https URL. 

---
# Libra-VLA: Achieving Learning Equilibrium via Asynchronous Coarse-to-Fine Dual-System 

**Authors**: Yifei Wei, Linqing Zhong, Yi Liu, Yuxiang Lu, Xindong He, Maoqing Yao, Guanghui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.24921)  

**Abstract**: Vision-Language-Action (VLA) models are a promising paradigm for generalist robotic manipulation by grounding high-level semantic instructions into executable physical actions. However, prevailing approaches typically adopt a monolithic generation paradigm, directly mapping visual-linguistic features to high-frequency motor commands in a flat, non-hierarchical fashion. This strategy overlooks the inherent hierarchy of robotic manipulation, where complex actions can be naturally modeled in a Hybrid Action Space, decomposing into discrete macro-directional reaching and continuous micro-pose alignment, severely widening the semantic-actuation gap and imposing a heavy representational burden on grounding high-level semantics to continuous actions. To address this, we introduce Libra-VLA, a novel Coarse-to-Fine Dual-System VLA architecture. We explicitly decouple the learning complexity into a coarse-to-fine hierarchy to strike a training equilibrium, while simultaneously leveraging this structural modularity to implement an asynchronous execution strategy. The Semantic Planner predicts discrete action tokens capturing macro-directional intent, while the Action Refiner conditions on coarse intent to generate high-frequency continuous actions for precise alignment. Crucially, our empirical analysis reveals that performance follows an inverted-U curve relative to action decomposition granularity, peaking exactly when the learning difficulty is balanced between the two sub-systems. With the asynchronous design, our approach offers a scalable, robust, and responsive solution for open-world manipulation. 

---
# GeoSearch: Augmenting Worldwide Geolocalization with Web-Scale Reverse Image Search and Image Matching 

**Authors**: Tung-Duong Le-Duc, Hoang-Quoc Nguyen-Son, Minh-Son Dao  

**Link**: [PDF](https://arxiv.org/pdf/2604.25390)  

**Abstract**: Worldwide image geolocalization, which aims to predict the GPS coordinates of any image on Earth, remains challenging due to global visual diversity. Recent generative approaches based on Retrieval-Augmented Generation (RAG) and Large Multimodal Models (LMMs) leverage candidates retrieved from fixed databases for reasoning, but often struggle with scenes that are absent from the reference set. In this work, we propose GeoSearch, an open-world geolocation framework that integrates web-scale reverse image search into the RAG pipeline. GeoSearch augments LMM prompts with database-retrieved coordinates and textual evidence extracted from web pages. To mitigate noise from irrelevant content, we introduce a two-layer filtering mechanism consisting of image matching, followed by confidence-based gating. Experiments on standard benchmarks Im2GPS3k and YFCC4k demonstrate the superiority of GeoSearch under leakage-aware evaluation. Our code and data are publicly available to support reproducibility. 

---
# Make Any Collection Navigable: Methods for Constructing and Evaluating Hypergraph of Text 

**Authors**: Dean E. Alvarez, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2604.25906)  

**Abstract**: One reason the Web is more useful than a simple collection of documents is that the structure created by hyperlinks enables flexible navigation from one web page to another. However, hyperlinks are typically created manually and cannot fully capture a corpus' implicit semantic structures. Is there a general way to make an arbitrary collection navigable? Recent work has formalized this problem generally as constructing a Hypergraph of Text (HoT), which provides a formal mathematical structure for supporting navigation and browsing. However, how to construct and evaluate a Hypergraph of Text remains a challenge. In this paper, we propose and study several methods for constructing a HoT. We also propose a novel quantitative metric, effort ratio, for evaluating the structural quality of a constructed HoT. Experimental results show that even simple TF-IDF baselines can match LLM-based methods on our proposed effort ratio metric. 

---
# From Local Indices to Global Identifiers: Generative Reranking for Recommender Systems via Global Action Space 

**Authors**: Pengyue Jia, Xiaobei Wang, Yingyi Zhang, Shuchang Liu, Yupeng Hou, Hailan Yang, Xu Gao, Xiaopeng Li, Yejing Wang, Julian McAuley, Xiang Li, Lantao Hu, Yongqi Liu, Kaiqiao Zhan, Han Li, Kun Gai, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.25291)  

**Abstract**: In modern recommender systems, list-wise reranking serves as a critical phase within the multi-stage pipeline, finalizing the exposed item sequence and directly impacting user satisfaction by modeling complex intra-list item dependencies. Existing methods typically formulate this task as selecting indices from the local input list. However, this approach suffers from a semantically inconsistent action space: the same output neuron (logits) represents different items across different samples, preventing the model from establishing a stable, intrinsic understanding of the items. To address this, we propose GloRank (Global Action Space Ranker), a generative framework that shifts reranking from selecting local indices to generating global identifiers. Specifically, we represent items as sequences of discrete tokens and reformulate reranking as a token generation task. This design effectively decouples the scoring mechanism from the variable input order, ensuring that items are evaluated against a consistent global standard. We further enhance this with a two-stage optimization pipeline: a supervised pre-training phase to initialize the model with high-quality demonstrations, followed by a reinforcement learning-based post-training phase to directly maximize list-wise utility. Extensive experiments on two public benchmarks and a large-scale industrial dataset, coupled with online A/B tests, demonstrate that GloRank consistently outperforms state-of-the-art baselines and achieves superior robustness in cold-start scenarios. 

---
# K-CARE: Knowledge-driven Symmetrical Contextual Anchoring and Analogical Prototype Reasoning for E-commerce Relevance 

**Authors**: Chen Yifei, Tian Zhixing, Wang Chenyang, Cheng Ziguang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25683)  

**Abstract**: This paper targets e-commerce search relevance. While Large Language Models (LLMs) have demonstrated significant potential in this field, they often encounter performance bottlenecks in persistent 'corner cases' within complex industrial scenarios. Existing research primarily focuses on optimizing reasoning trajectories via Reinforcement Learning. However, real-world observations suggest that the primary bottleneck stems from knowledge boundaries, where the absence of domain-specific intelligence in the model's parametric memory creates a contextual void. This void persists when interpreting idiosyncratic queries or niche products and cannot be resolved solely through reasoning-path optimization.
To bridge this gap, we propose K-CARE, a framework that extends the model's cognitive reach by grounding reasoning in external knowledge. K-CARE comprises two synergistic components: (1) Symmetrical Contextual Anchoring (SCA), which fills the contextual void by anchoring queries and products with behavior-derived implicit knowledge; and (2) Analogical Prototype Reasoning (APR), which leverages expert-curated prototypical knowledge to calibrate decision boundaries through in-context analogy. Extensive offline evaluations and online A/B tests on a leading e-commerce platform demonstrate that K-CARE significantly outperforms state-of-the-art baselines, delivering substantial commercial impact by resolving knowledge-intensive relevance challenges. 

---
# From Citation Selection to Citation Absorption: A Measurement Framework for Generative Engine Optimization Across AI Search Platforms 

**Authors**: Zhang Kai, Yao Jingang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25707)  

**Abstract**: Generative search engines increasingly determine whether online information is merely discoverable, cited as a source, or actually absorbed into generated answers. This paper proposes a two-stage measurement framework for Generative Engine Optimization (GEO): citation selection, where a platform triggers search and chooses sources, and citation absorption, where a cited page contributes language, evidence, structure, or factual support to the final answer. We analyze the public geo-citation-lab dataset covering 602 controlled prompts across ChatGPT, Google AI Overview/Gemini, and Perplexity; 21,143 valid search-layer citations; 23,745 citation-level feature records; 18,151 successfully fetched pages; and 72 extracted features. The central descriptive finding is that citation breadth and citation depth diverge. Perplexity and Google cite more sources on average, while ChatGPT cites fewer sources but shows substantially higher average citation influence among fetched pages. High-influence pages tend to be longer, more structured, semantically aligned, and richer in extractable evidence such as definitions, numerical facts, comparisons, and procedural steps. The results suggest that GEO should be measured beyond citation counts, with answer-level absorption treated as a separate outcome. 

---
