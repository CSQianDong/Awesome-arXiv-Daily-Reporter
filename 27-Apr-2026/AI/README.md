# Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond 

**Authors**: Meng Chu, Xuan Billy Zhang, Kevin Qinghong Lin, Lingdong Kong, Jize Zhang, Teng Tu, Weijian Ma, Ziqi Huang, Senqiao Yang, Wei Huang, Yeying Jin, Zhefan Rao, Jinhui Ye, Xinyu Lin, Xichen Zhang, Qisheng Hu, Shuai Yang, Leyang Shen, Wei Chow, Yifei Dong, Fengyi Wu, Quanyu Long, Bin Xia, Shaozuo Yu, Mingkang Zhu, Wenhu Zhang, Jiehui Huang, Haokun Gui, Haoxuan Che, Long Chen, Qifeng Chen, Wenxuan Zhang, Wenya Wang, Xiaojuan Qi, Yang Deng, Yanwei Li, Mike Zheng Shou, Zhi-Qi Cheng, See-Kiong Ng, Ziwei Liu, Philip Torr, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.22748)  

**Abstract**: As AI systems move from generating text to accomplishing goals through sustained interaction, the ability to model environment dynamics becomes a central bottleneck. Agents that manipulate objects, navigate software, coordinate with others, or design experiments require predictive environment models, yet the term world model carries different meanings across research communities. We introduce a "levels x laws" taxonomy organized along two axes. The first defines three capability levels: L1 Predictor, which learns one-step local transition operators; L2 Simulator, which composes them into multi-step, action-conditioned rollouts that respect domain laws; and L3 Evolver, which autonomously revises its own model when predictions fail against new evidence. The second identifies four governing-law regimes: physical, digital, social, and scientific. These regimes determine what constraints a world model must satisfy and where it is most likely to fail. Using this framework, we synthesize over 400 works and summarize more than 100 representative systems spanning model-based reinforcement learning, video generation, web and GUI agents, multi-agent social simulation, and AI-driven scientific discovery. We analyze methods, failure modes, and evaluation practices across level-regime pairs, propose decision-centric evaluation principles and a minimal reproducible evaluation package, and outline architectural guidance, open problems, and governance challenges. The resulting roadmap connects previously isolated communities and charts a path from passive next-step prediction toward world models that can simulate, and ultimately reshape, the environments in which agents operate. 

---
# Rethinking Math Reasoning Evaluation: A Robust LLM-as-a-Judge Framework Beyond Symbolic Rigidity 

**Authors**: Erez Yosef, Oron Anschel, Shunit Haviv Hakimi, Asaf Gendler, Adam Botach, Nimrod Berman, Igor Kviatkovsky  

**Link**: [PDF](https://arxiv.org/pdf/2604.22597)  

**Abstract**: Recent advancements in large language models have led to significant improvements across various tasks, including mathematical reasoning, which is used to assess models' intelligence in logical reasoning and problem-solving. Models are evaluated on mathematical reasoning benchmarks by verifying the correctness of the final answer against a ground truth answer. A common approach for this verification is based on symbolic mathematics comparison, which fails to generalize across diverse mathematical representations and solution formats. In this work, we offer a robust and flexible alternative to rule-based symbolic mathematics comparison. We propose an LLM-based evaluation framework for evaluating model-generated answers, enabling accurate evaluation across diverse mathematical representations and answer formats. We present failure cases of symbolic evaluation in two popular frameworks, Lighteval and SimpleRL, and compare them to our approach, demonstrating clear improvements over commonly used methods. Our framework enables more reliable evaluation and benchmarking, leading to more accurate performance monitoring, which is important for advancing mathematical problem-solving and intelligent systems. 

---
# QuantClaw: Precision Where It Matters for OpenClaw 

**Authors**: Manyi Zhang, Ji-Fu Li, Zhongao Sun, Xiaohao Liu, Zhenhua Dong, Xianzhi Yu, Haoli Bai, Xiaobo Xia  

**Link**: [PDF](https://arxiv.org/pdf/2604.22577)  

**Abstract**: Autonomous agent systems such as OpenClaw introduce significant efficiency challenges due to long-context inputs and multi-turn reasoning. This results in prohibitively high computational and monetary costs in real-world development. While quantization is a standard approach for reducing cost and latency, its impact on agent performance in realistic scenarios remains unclear. In this work, we analyze quantization sensitivity across diverse complex workflows over OpenClaw, and show that precision requirements are highly task-dependent. Based on this observation, we propose QuantClaw, a plug-and-play precision routing plugin that dynamically assigns precision according to task characteristics. QuantClaw routes lightweight tasks to lower-cost configurations while preserving higher precision for demanding workloads, saving cost and accelerating inference without increasing user complexity. Experiments show that our QuantClaw maintains or improves task performance while reducing both latency and computational cost. Across a range of agent tasks, it achieves up to 21.4% cost savings and 15.7% latency reduction on GLM-5 (FP8 baseline). These results highlight the benefit of treating precision as a dynamic resource in agent systems. 

---
# On the Hybrid Nature of ABPMS Process Frames and its Implications on Automated Process Discovery 

**Authors**: Anti Alman, Izack Cohen, Avigdor Gal, Fabrizio Maria Maggi, Marco Montali  

**Link**: [PDF](https://arxiv.org/pdf/2604.22455)  

**Abstract**: A core component of any AI-Augmented Business Process Management System (ABPMS) is the process frame, which gives the system process-awareness and defines the boundaries in which the system must operate. Compared to traditional process models, the process frame should, in principle, provide a somewhat more permissive representation of the managed processes, such that the (semi) autonomous behavior of an ABPMS, referred to as framed autonomy, could emerge. At the same time, it is not limited to a single linguistic or symbolic formalism and may incorporate heterogeneous knowledge ranging from predefined procedures to commonsense rules and best practices. In this paper, we conceptualize the notion of an ABPMS process frame as a hybrid business process representation, consisting of semi-concurrently executed procedural and declarative process models. We rely on our earlier works to outline the execution semantics of this type of process frame, arguing in favor of adopting the open-world assumption of the declarative paradigm also for procedural process models. The latter leads to a constraint-like interpretation, where each procedural model is considered to constrain the activities within that model, without imposing explicit execution requirements nor limitations on activities that may be present in other models. This is analogous to existing declarative languages, such as Declare, where each constraint has a direct effect only on the specific activities being constrained. Given this similarity, we propose mapping subsets of discovered declarative constraints into equivalent semi-concurrently executed procedural fragments, thus laying the foundation for a corresponding process (frame) discovery approach. 

---
# Superminds Test: Actively Evaluating Collective Intelligence of Agent Society via Probing Agents 

**Authors**: Xirui Li, Ming Li, Yunze Xiao, Ryan Wong, Dianqi Li, Timothy Baldwin, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.22452)  

**Abstract**: Collective intelligence refers to the ability of a group to achieve outcomes beyond what any individual member can accomplish alone. As large language model agents scale to populations of millions, a key question arises: Does collective intelligence emerge spontaneously from scale? We present the first empirical evaluation of this question in a large-scale autonomous agent society. Studying MoltBook, a platform hosting over two million agents, we introduce Superminds Test, a hierarchical framework that probes society-level intelligence using controlled Probing Agents across three tiers: joint reasoning, information synthesis, and basic interaction. Our experiments reveal a stark absence of collective intelligence. The society fails to outperform individual frontier models on complex reasoning tasks, rarely synthesizes distributed information, and often fails even trivial coordination tasks. Platform-wide analysis further shows that interactions remain shallow, with threads rarely extending beyond a single reply and most responses being generic or off-topic. These results suggest that collective intelligence does not emerge from scale alone. Instead, the dominant limitation of current agent societies is extremely sparse and shallow interaction, which prevents agents from exchanging information and building on each other's outputs. 

---
# From Skills to Talent: Organising Heterogeneous Agents as a Real-World Company 

**Authors**: Zhengxu Yu, Yu Fu, Zhiyuan He, Yuxuan Huang, Lee Ka Yiu, Meng Fang, Weilin Luo, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22446)  

**Abstract**: Individual agent capabilities have advanced rapidly through modular skills and tool integrations, yet multi-agent systems remain constrained by fixed team structures, tightly coupled coordination logic, and session-bound learning. We argue that this reflects a deeper absence: a principled organisational layer that governs how a workforce of agents is assembled, governed, and improved over time, decoupled from what individual agents know. To fill this gap, we introduce \emph{OneManCompany (OMC)}, a framework that elevates multi-agent systems to the organisational level. OMC encapsulates skills, tools, and runtime configurations into portable agent identities called \emph{Talents}, orchestrated through typed organisational interfaces that abstract over heterogeneous backends. A community-driven \emph{Talent Market} enables on-demand recruitment, allowing the organisation to close capability gaps and reconfigure itself dynamically during execution. Organisational decision-making is operationalised through an \emph{Explore-Execute-Review} ($\text{E}^2$R) tree search, which unifies planning, execution, and evaluation in a single hierarchical loop: tasks are decomposed top-down into accountable units and execution outcomes are aggregated bottom-up to drive systematic review and refinement. This loop provides formal guarantees on termination and deadlock freedom while mirroring the feedback mechanisms of human enterprises. Together, these contributions transform multi-agent systems from static, pre-configured pipelines into self-organising and self-improving AI organisations capable of adapting to open-ended tasks across diverse domains. Empirical evaluation on PRDBench shows that OMC achieves an $84.67\%$ success rate, surpassing the state of the art by $15.48$ percentage points, with cross-domain case studies further demonstrating its generality. 

---
# AgentSearchBench: A Benchmark for AI Agent Search in the Wild 

**Authors**: Bin Wu, Arastun Mammadli, Xiaoyu Zhang, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.22436)  

**Abstract**: The rapid growth of AI agent ecosystems is transforming how complex tasks are delegated and executed, creating a new challenge of identifying suitable agents for a given task. Unlike traditional tools, agent capabilities are often compositional and execution-dependent, making them difficult to assess from textual descriptions alone. However, existing research and benchmarks typically assume well-specified functionalities, controlled candidate pools, or only executable task queries, leaving realistic agent search scenarios insufficiently studied. We introduce AgentSearchBench, a large-scale benchmark for agent search in the wild, built from nearly 10,000 real-world agents across multiple providers. The benchmark formalizes agent search as retrieval and reranking problems under both executable task queries and high-level task descriptions, and evaluates relevance using execution-grounded performance signals. Experiments reveal a consistent gap between semantic similarity and actual agent performance, exposing the limitations of description-based retrieval and reranking methods. We further show that lightweight behavioral signals, including execution-aware probing, can substantially improve ranking quality, highlighting the importance of incorporating execution signals into agent discovery. Our code is available at this https URL. 

---
# CognitiveTwin: Robust Multi-Modal Digital Twins for Predicting Cognitive Decline in Alzheimer's Disease 

**Authors**: Bulent Soykan, Gulsah Hancerliogullari Koksalmis, Hsin-Hsiung Huang, Laura J. Brattain  

**Link**: [PDF](https://arxiv.org/pdf/2604.22428)  

**Abstract**: Predicting individual cognitive decline in Alzheimer's disease (AD) is difficult due to the heterogeneity of disease progression. Reliable clinical tools require not only high accuracy but also fairness across demographics and robustness to missing data. We present CognitiveTwin, a digital twin framework that predicts patient-specific cognitive trajectories. The model integrates multi-modal longitudinal data (cognitive scores, magnetic resonance imaging, positron emission tomography, cerebrospinal fluid biomarkers, and genetics). We use a Transformer-based architecture to fuse these modalities and a Deep Markov Model to capture temporal dynamics. We trained and evaluated the framework using data from 1,666 patients in the TADPOLE (Alzheimer's Disease Neuroimaging Initiative) dataset. We assessed the model for prediction error, demographic fairness, and robustness to missing-not-at-random (MNAR) data patterns. ognitiveTwin provides accurate and personalized predictions of cognitive decline. Its demonstrated fairness across patient demographics and resilience to clinical dropout make it a reliable tool for clinical trial enrichment and personalized care planning. 

---
# Introducing Background Temperature to Characterise Hidden Randomness in Large Language Models 

**Authors**: Alberto Messina, Stefano Scotta  

**Link**: [PDF](https://arxiv.org/pdf/2604.22411)  

**Abstract**: Even when decoding with temperature $T=0$, large language models (LLMs) can produce divergent outputs for identical inputs. Recent work by Thinking Machines Lab highlights implementation-level sources of nondeterminism, including batch-size variation, kernel non-invariance, and floating-point non-associativity. In this short note we formalize this behavior by introducing the notion of \emph{background temperature} $T_{\mathrm{bg}}$, the effective temperature induced by an implementation-dependent perturbation process observed even when nominal $T=0$. We provide clean definitions, show how $T_{\mathrm{bg}}$ relates to a stochastic perturbation governed by the inference environment $I$, and propose an empirical protocol to estimate $T_{bg}$ via the equivalent temperature $T_n(I)$ of an ideal reference system. We conclude with a set of pilot experiments run on a representative pool from the major LLM providers that demonstrate the idea and outline implications for reproducibility, evaluation, and deployment. 

---
# When Does LLM Self-Correction Help? A Control-Theoretic Markov Diagnostic and Verify-First Intervention 

**Authors**: Aofan Liu, Jingxiang Meng  

**Link**: [PDF](https://arxiv.org/pdf/2604.22273)  

**Abstract**: Iterative self-correction is widely used in agentic LLM systems, but when repeated refinement helps versus hurts remains unclear. We frame self-correction as a cybernetic feedback loop in which the same language model serves as both controller and plant, and use a two-state Markov model over {Correct, Incorrect} to operationalize a simple deployment diagnostic: iterate only when ECR/EIR > Acc/(1 - Acc). In this view, EIR functions as a stability margin and prompting functions as lightweight controller design. Across 7 models and 3 datasets (GSM8K, MATH, StrategyQA), we find a sharp near-zero EIR threshold (<= 0.5%) separating beneficial from harmful self-correction. Only o3-mini (+3.4 pp, EIR = 0%), Claude Opus 4.6 (+0.6 pp, EIR ~ 0.2%), and o4-mini (+/-0 pp) remain non-degrading; GPT-5 degrades by -1.8 pp. A verify-first prompt ablation provides causal evidence that this threshold is actionable through prompting alone: on GPT-4o-mini it reduces EIR from 2% to 0% and turns -6.2 pp degradation into +0.2 pp (paired McNemar p < 10^-4), while producing little change on already-sub-threshold models. ASC further illustrates the stopping trade-off: it halts harmful refinement but incurs a 3.8 pp confidence-elicitation cost. Overall, the paper argues that self-correction should be treated not as a default behavior, but as a control decision governed by measurable error dynamics. 

---
# Emergent Strategic Reasoning Risks in AI: A Taxonomy-Driven Evaluation Framework 

**Authors**: Tharindu Kumarage, Lisa Bauer, Yao Ma, Dan Rosen, Yashasvi Raghavendra Guduri, Anna Rumshisky, Kai-Wei Chang, Aram Galstyan, Rahul Gupta, Charith Peris  

**Link**: [PDF](https://arxiv.org/pdf/2604.22119)  

**Abstract**: As reasoning capacity and deployment scope grow in tandem, large language models (LLMs) gain the capacity to engage in behaviors that serve their own objectives, a class of risks we term Emergent Strategic Reasoning Risks (ESRRs). These include, but are not limited to, deception (intentionally misleading users or evaluators), evaluation gaming (strategically manipulating performance during safety testing), and reward hacking (exploiting misspecified objectives). Systematically understanding and benchmarking these risks remains an open challenge. To address this gap, we introduce ESRRSim, a taxonomy-driven agentic framework for automated behavioral risk evaluation. We construct an extensible risk taxonomy of 7 categories, which is decomposed into 20 subcategories. ESRRSim generates evaluation scenarios designed to elicit faithful reasoning, paired with dual rubrics assessing both model responses and reasoning traces, in a judge-agnostic and scalable architecture. Evaluation across 11 reasoning LLMs reveals substantial variation in risk profiles (detection rates ranging 14.45%-72.72%), with dramatic generational improvements suggesting models may increasingly recognize and adapt to evaluation contexts. 

---
# Memanto: Typed Semantic Memory with Information-Theoretic Retrieval for Long-Horizon Agents 

**Authors**: Seyed Moein Abtahi, Rasa Rahnema, Hetkumar Patel, Neel Patel, Majid Fekri, Tara Khani  

**Link**: [PDF](https://arxiv.org/pdf/2604.22085)  

**Abstract**: The transition from stateless language model inference to persistent, multi session autonomous agents has revealed memory to be a primary architectural bottleneck in the deployment of production grade agentic systems. Existing methodologies largely depend on hybrid semantic graph architectures, which impose substantial computational overhead during both ingestion and retrieval. These systems typically require large language model mediated entity extraction, explicit graph schema maintenance, and multi query retrieval pipelines. This paper introduces Memanto, a universal memory layer for agentic artificial intelligence that challenges the prevailing assumption that knowledge graph complexity is necessary to achieve high fidelity agent memory. Memanto integrates a typed semantic memory schema comprising thirteen predefined memory categories, an automated conflict resolution mechanism, and temporal versioning. These components are enabled by Moorcheh's Information Theoretic Search engine, a no indexing semantic database that provides deterministic retrieval within sub ninety millisecond latency while eliminating ingestion delay. Through systematic benchmarking on the LongMemEval and LoCoMo evaluation suites, Memanto achieves state of the art accuracy scores of 89.8 percent and 87.1 percent respectively. These results surpass all evaluated hybrid graph and vector based systems while requiring only a single retrieval query, incurring no ingestion cost, and maintaining substantially lower operational complexity. A five stage progressive ablation study is presented to quantify the contribution of each architectural component, followed by a discussion of the implications for scalable deployment of agentic memory systems. 

---
# Sound Agentic Science Requires Adversarial Experiments 

**Authors**: Dionizije Fa, Marko Culjak  

**Link**: [PDF](https://arxiv.org/pdf/2604.22080)  

**Abstract**: LLM-based agents are rapidly being adopted for scientific data analysis, automating tasks once limited by human time and expertise. This capability is often framed as an acceleration of discovery, but it also accelerates a familiar failure mode, the rapid production of plausible, endlessly revisable analyses that are easy to generate, effectively turning hypothesis space into candidate claims supported by selectively chosen analyses, optimized for publishable positives. Unlike software, scientific knowledge is not validated by the iterative accumulation of code and post hoc statistical support. A fluent explanation or a significant result on a single dataset is not verification. Because the missing evidence is a negative space, experiments and analyses that would have falsified the claim were never run or never published. We therefore propose that non-experimental claims produced with agentic assistance be evaluated under a falsification-first standard: agents should not be used primarily to craft the most compelling narrative, but to actively search for the ways in which the claim can fail. 

---
# Rethinking Publication: A Certification Framework for AI-Enabled Research 

**Authors**: Yang Lu, Rabimba Karanjai, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.22026)  

**Abstract**: AI research pipelines now produce a growing share of publishable academic output, including work that meets existing peer-review standards for quality and novelty. Yet the publication system was built on the assumption of universal human authorship and lacks a principled way to evaluate knowledge produced through automated pipelines. This paper proposes a two-layer certification framework that separates knowledge quality assessment from grading of human contribution, allowing publication systems to handle pipeline-generated work consistently and transparently without creating new institutions. The paper uses normative-conceptual analysis, framework design under four explicit constraints, and dry-run validation on two representative submission cases spanning key attribution scenarios. The framework grades contributions as Category A (pipeline-reachable), Category B (requiring human direction at identifiable stages), and Category C (beyond current pipeline reach at the formulation stage). It also introduces benchmark slots for fully disclosed automated research as both a transparent publication track and a calibration instrument for reviewer judgment. Contribution grading is contemporaneous, based on pipeline capability at the time of submission. Dry-run validation shows that the framework can certify knowledge appropriately while tolerating irreducible attribution uncertainty. The paper argues that publication has always certified both that knowledge is valid and that a human made it. AI pipelines separate these functions for the first time. The framework is implementable within existing editorial infrastructure and grounds recognition of frontier human contribution in epistemic achievement rather than unverifiable claims of human origin. 

---
# Read the Paper, Write the Code: Agentic Reproduction of Social-Science Results 

**Authors**: Benjamin Kohler, David Zollikofer, Johanna Einsiedler, Alexander Hoyle, Elliott Ash  

**Link**: [PDF](https://arxiv.org/pdf/2604.21965)  

**Abstract**: Recent work has used LLM agents to reproduce empirical social science results with access to both the data and code. We broaden this scope by asking: Can they reproduce results given only a paper's methods description and original data? We develop an agentic reproduction system that extracts structured methods descriptions from papers, runs reimplementations under strict information isolation -- agents never see the original code, results, or paper -- and enables deterministic, cell-level comparison of reproduced outputs to the original results. An error attribution step traces discrepancies through the system chain to identify root causes. Evaluating four agent scaffolds and four LLMs on 48 papers with human-verified reproducibility, we find that agents can largely recover published results, but performance varies substantially between models, scaffolds, and papers. Root cause analysis reveals that failures stem both from agent errors and from underspecification in the papers themselves. 

---
# MolClaw: An Autonomous Agent with Hierarchical Skills for Drug Molecule Evaluation, Screening, and Optimization 

**Authors**: Lisheng Zhang, Lilong Wang, Xiangyu Sun, Wei Tang, Haoyang Su, Yuehui Qian, Qikui Yang, Qingsong Li, Zhenyu Tang, Haoran Sun, Yingnan Han, Yankai Jiang, Wenjie Lou, Bowen Zhou, Xiaosong Wang, Lei Bai, Zhengwei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.21937)  

**Abstract**: Computational drug discovery, particularly the complex workflows of drug molecule screening and optimization, requires orchestrating dozens of specialized tools in multi-step workflows, yet current AI agents struggle to maintain robust performance and consistently underperform in these high-complexity scenarios. Here we present MolClaw, an autonomous agent that leads drug molecule evaluation, screening, and optimization. It unifies over 30 specialized domain resources through a three-tier hierarchical skill architecture (70 skills in total) that facilitates agent long-term interaction at runtime: tool-level skills standardize atomic operations, workflow-level skills compose them into validated pipelines with quality check and reflection, and a discipline-level skill supplies scientific principles governing planning and verification across all scenarios in the field. Additionally, we introduce MolBench, a benchmark comprising molecular screening, optimization, and end-to-end discovery challenges spanning 8 to 50+ sequential tool calls. MolClaw achieves state-of-the-art performance across all metrics, and ablation studies confirm that gains concentrate on tasks that demand structured workflows while vanishing on those solvable with ad hoc scripting, establishing workflow orchestration competence as the primary capability bottleneck for AI-driven drug discovery. 

---
# An Artifact-based Agent Framework for Adaptive and Reproducible Medical Image Processing 

**Authors**: Lianrui Zuo, Yihao Liu, Gaurav Rudravaram, Karthik Ramadass, Aravind R. Krishnan, Michael D. Phillips, Yelena G. Bodien, Mayur B. Patel, Paula Trujillo, Yency Forero Martinez, Stephen A. Deppen, Eric L. Grogan, Fabien Maldonado, Kevin McGann, Hudson M. Holmes, Laurie E. Cutting, Yuankai Huo, Bennett A. Landman  

**Link**: [PDF](https://arxiv.org/pdf/2604.21936)  

**Abstract**: Medical imaging research is increasingly shifting from controlled benchmark evaluation toward real-world clinical deployment. In such settings, applying analytical methods extends beyond model design to require dataset-aware workflow configuration and provenance tracking. Two requirements therefore become central: \textbf{adaptability}, the ability to configure workflows according to dataset-specific conditions and evolving analytical goals; and \textbf{reproducibility}, the guarantee that all transformations and decisions are explicitly recorded and re-executable. Here, we present an artifact-based agent framework that introduces a semantic layer to augment medical image processing. The framework formalizes intermediate and final outputs through an artifact contract, enabling structured interrogation of workflow state and goal-conditioned assembly of configurations from a modular rule library. Execution is delegated to a workflow executor to preserve deterministic computational graph construction and provenance tracking, while the agent operates locally to comply with most privacy constraints. We evaluate the framework on real-world clinical CT and MRI cohorts, demonstrating adaptive configuration synthesis, deterministic reproducibility across repeated executions, and artifact-grounded semantic querying. These results show that adaptive workflow configuration can be achieved without compromising reproducibility in heterogeneous clinical environments. 

---
# Math Takes Two: A test for emergent mathematical reasoning in communication 

**Authors**: Michael Cooper, Samuel Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2604.21935)  

**Abstract**: Although language models demonstrate remarkable proficiency on mathematical benchmarks, it remains unclear whether this reflects true mathematical reasoning or statistical pattern matching over learning formal syntax. Most existing evaluations rely on symbolic problems grounded in established mathematical conventions, limiting insight into the models' ability to construct abstract concepts from first principles. In this work, we propose Math Takes Two, a new benchmark designed to assess the emergence of mathematical reasoning through communication. Motivated by the hypothesis that mathematical cognition in humans co-evolved with the need for precise communication, our benchmark tests whether two agents, without prior mathematical knowledge, can develop a shared symbolic protocol to solve a visually grounded task where the use of a numerical system facilitates extrapolation. Unlike many current datasets, our benchmark eschews predefined mathematical language, instead requiring agents to discover latent structure and representations from scratch. Math Takes Two thus provides a novel lens through which to develop and evaluate models with emergent numerical reasoning capabilities. 

---
# An Undecidability Proof for the Plan Existence Problem 

**Authors**: Antonis Achilleos  

**Link**: [PDF](https://arxiv.org/pdf/2604.22736)  

**Abstract**: The plan existence problem asks, given a goal in the form of a formula in modal logic, an initial epistemic state (a pointed Kripke model), and a set of epistemic actions, whether there exists a sequence of actions that can be applied to reach the goal. We prove that even in the case where the preconditions of the epistemic actions have modal depth at most 1, and there are no postconditions, the plan existence problem is undecidable. The (un)decidability of this problem was previously unknown. 

---
# Aligning Dense Retrievers with LLM Utility via DistillationAligning Dense Retrievers with LLM Utility via Distillation 

**Authors**: Rajinder Sandhu, Di Mu, Cheng Chang, Md Shahriar Tasjid, Himanshu Rai, Maksims Volkovs, Ga Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22722)  

**Abstract**: Dense vector retrieval is the practical backbone of Retrieval- Augmented Generation (RAG), but similarity search can suffer from precision limitations. Conversely, utility-based approaches leveraging LLM re-ranking often achieve superior performance but are computationally prohibitive and prone to noise inherent in perplexity estimation. We propose Utility-Aligned Embeddings (UAE), a framework designed to merge these advantages into a practical, high-performance retrieval method. We formulate retrieval as a distribution matching problem, training a bi-encoder to imitate a utility distribution derived from perplexity reduction using a Utility-Modulated InfoNCE objective. This approach injects graded utility signals directly into the embedding space without requiring test-time LLM inference. On the QASPER benchmark, UAE improves retrieval Recall@1 by 30.59%, MAP by 30.16% and Token F1 by 17.3% over the strong semantic baseline BGE-Base. Crucially, UAE is over 180x faster than the efficient LLM re-ranking methods preserving competitive performance, demonstrating that aligning retrieval with generative utility yields reliable contexts at scale. 

---
# CRAFT: Clustered Regression for Adaptive Filtering of Training data 

**Authors**: Parthasarathi Panda, Asheswari Swain, Subhrakanta Panda  

**Link**: [PDF](https://arxiv.org/pdf/2604.22693)  

**Abstract**: Selecting a small, high-quality subset from a large corpus for fine-tuning is increasingly important as corpora grow to tens of millions of datapoints, making full fine-tuning expensive and often unnecessary. We propose CRAFT (Clustered Regression for Adaptive Filtering of Training data), a vectorization-agnostic selection method for training sequence-to-sequence models. CRAFT decomposes the joint source-target distribution and performs a two-stage selection: (i) match the validation source distribution through proportional budget allocation across k-means clusters, and (ii) within each source cluster, select training pairs whose target embeddings minimize a conditional expected distance derived from the validation target distribution. We prove that proportional cluster allocation bounds the continuous KL divergence between selected and validation distributions, with the residual controlled by cluster diameters. We evaluate CRAFT on English-Hindi translation by selecting training data from 33 million NLLB sentence pairs and fine-tuning mBART via LoRA. CRAFT achieves 43.34 BLEU, outperforming TSDS (41.21) by 2.13 points on the same candidate pool and encoder while completing selection over 40 times faster. With TF-IDF vectorization, the entire pipeline completes in under one minute on CPU. TAROT achieves 45.61 BLEU, but CRAFT completes selection in 26.86 seconds versus TAROT's 75.6 seconds, a 2.8 time speedup. 

---
# How Supply Chain Dependencies Complicate Bias Measurement and Accountability Attribution in AI Hiring Applications 

**Authors**: Gauri Sharma, Maryam Molamohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2604.22679)  

**Abstract**: The increasing adoption of AI systems in hiring has raised concerns about algorithmic bias and accountability, prompting regulatory responses including the EU AI Act, NYC Local Law 144, and Colorado's AI Act. While existing research examines bias through technical or regulatory lenses, both perspectives overlook a fundamental challenge: modern AI hiring systems operate within complex supply chains where responsibility fragments across data vendors, model developers, platform providers, and deploying organizations. This paper investigates how these dependency chains complicate bias evaluation and accountability attribution. Drawing on literature review and regulatory analysis, we demonstrate that fragmented responsibilities create two critical problems. First, bias emerges from component interactions rather than isolated elements, yet proprietary configurations prevent integrated evaluation. A resume parser may function without bias independently but contribute to discrimination when integrated with specific ranking algorithms and filtering thresholds. Second, information asymmetries mean deploying organizations bear legal responsibility without technical visibility into vendor-supplied algorithms, while vendors control implementations without meaningful disclosure requirements. Each stakeholder may believe they are compliant; nevertheless, the integrated system may produce biased outcomes. Analysis of implementation ambiguities reveals these challenges in practice. We propose multi-layered interventions including system-level audits, vendor guidelines, continuous monitoring mechanisms, and documentation across dependency chains. Our findings reveal that effective governance requires coordinated action across technical, organizational, and regulatory domains to establish meaningful accountability in distributed development environments. 

---
# Rethinking XAI Evaluation: A Human-Centered Audit of Shapley Benchmarks in High-Stakes Settings 

**Authors**: Inês Oliveira e Silva, Sérgio Jesus, Iker Perez, Rita P. Ribeiro, Carlos Soares, Hugo Ferreira, Pedro Bizarro  

**Link**: [PDF](https://arxiv.org/pdf/2604.22662)  

**Abstract**: Shapley values are a cornerstone of explainable AI, yet their proliferation into competing formulations has created a fragmented landscape with little consensus on practical deployment. While theoretical differences are well-documented, evaluation remains reliant on quantitative proxies whose alignment with human utility is unverified. In this work, we use a unified amortized framework to isolate semantic differences between eight Shapley variants under the low-latency constraints of operational risk workflows. We conduct a large-scale empirical evaluation across four risk datasets and a realistic fraud-detection environment involving professional analysts and 3,735 case reviews. Our results reveal a fundamental misalignment: standard quantitative metrics, such as sparsity and faithfulness, are decoupled from human-perceived clarity and decision utility. Furthermore, while no formulation improved objective analyst performance, explanations consistently increased decision confidence, signaling a critical risk of automation bias in high-stakes settings. These findings suggest that current evaluation proxies are insufficient for predicting downstream human impact, and we provide evidence-based guidance for selecting formulations and metrics in operational decision systems. 

---
# From Natural Language to Verified Code: Toward AI Assisted Problem-to-Code Generation with Dafny-Based Formal Verification 

**Authors**: Md Erfan, Md Kamal Hossain Chowdhury, Ahmed Ryan, Md Rayhanur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2604.22601)  

**Abstract**: Large Language Models (LLMs) show promise in automated software engineering, yet their guarantee of correctness is frequently undermined by erroneous or hallucinated code. To enforce model honesty, formal verification requires LLMs to synthesize implementation logic alongside formal specifications that are subsequently proven correct by a mathematical verifier. However, the transition from informal natural language to precise formal specification remains an arduous task. Our work addresses this by providing the NaturalLanguage2VerifiedCode (NL2VC)-60 dataset: a collection of 60 complex algorithmic problems. We evaluate 11 randomly selected problem sets across seven open-weight LLMs using a tiered prompting strategy: contextless prompts, signature prompts providing structural anchors, and self-healing prompts utilizing iterative feedback from the Dafny verifier. To address vacuous verification, where models satisfy verifiers with trivial specifications, we integrate the uDebug platform to ensure functional validation. Our results show that while contextless prompting leads to near-universal failure, structural signatures and iterative self-healing facilitate a dramatic performance turnaround. Specifically, Gemma 4-31B achieved a 90.91\% verification success rate, while GPT-OSS 120B rose from zero to 81.82\% success with signature-guided feedback. These findings indicate that formal verification is now attainable for open-weight LLMs, which serve as effective apprentices for synthesizing complex annotations and facilitating high-assurance software development. 

---
# Learning Evidence Highlighting for Frozen LLMs 

**Authors**: Shaoang Li, Yanhang Shi, Yufei Li, Mingfu Liang, Xiaohan Wei, Yunchen Pu, Fei Tian, Chonglin Sun, Frank Shyu, Luke Simon, Sandeep Pandey, Xi Liu, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.22565)  

**Abstract**: Large Language Models (LLMs) can reason well, yet often miss decisive evidence when it is buried in long, noisy contexts. We introduce HiLight, an Evidence Emphasis framework that decouples evidence selection from reasoning for frozen LLM solvers. HiLight avoids compressing or rewriting the input, which can discard or distort evidence, by training a lightweight Emphasis Actor to insert minimal highlight tags around pivotal spans in the unaltered context. A frozen Solver then performs downstream reasoning on the emphasized input. We cast highlighting as a weakly supervised decision-making problem and optimize the Actor with reinforcement learning using only the Solver's task reward, requiring no evidence labels and no access to or modification of the Solver. Across sequential recommendation and long-context question answering, HiLight consistently improves performance over strong prompt-based and automated prompt-optimization baselines. The learned emphasis policy transfers zero-shot to both smaller and larger unseen Solver families, including an API-based Solver, suggesting that the Actor captures genuine, reusable evidence structure rather than overfitting to a single backbone. 

---
# Data-Free Contribution Estimation in Federated Learning using Gradient von Neumann Entropy 

**Authors**: Asim Ukaye, Mubarak Abdu-Aguye, Nurbek Tastan, Karthik Nandakumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.22562)  

**Abstract**: Client contribution estimation in Federated Learning is necessary for identifying clients' importance and for providing fair rewards. Current methods often rely on server-side validation data or self-reported client information, which can compromise privacy or be susceptible to manipulation. We introduce a data-free signal based on the matrix von Neumann (spectral) entropy of the final-layer updates, which measures the diversity of the information contributed. We instantiate two practical schemes: (i) SpectralFed, which uses normalized entropy as aggregation weights, and (ii) SpectralFuse, which fuses entropy with class-specific alignment via a rank-adaptive Kalman filter for per-round stability. Across CIFAR-10/100 and the naturally partitioned FEMNIST and FedISIC benchmarks, entropy-derived scores show a consistently high correlation with standalone client accuracy under diverse non-IID regimes - without validation data or client metadata. We compare our results with data-free contribution estimation baselines and show that spectral entropy serves as a useful indicator of client contribution. 

---
# Cross-Stage Coherence in Hierarchical Driving VQA: Explicit Baselines and Learned Gated Context Projectors 

**Authors**: Gautam Kumar Jain, Carsten Markgraf, Julian Stähler  

**Link**: [PDF](https://arxiv.org/pdf/2604.22560)  

**Abstract**: Graph Visual Question Answering (GVQA) for autonomous driving organizes reasoning into ordered stages, namely Perception, Prediction, and Planning, where planning decisions should remain consistent with the model's own perception. We present a comparative study of cross-stage context passing on DriveLM-nuScenes using two complementary mechanisms. The explicit variant evaluates three prompt-based conditioning strategies on a domain-adapted 4B VLM (Mini-InternVL2-4B-DA-DriveLM) without additional training, reducing NLI contradiction by up to 42.6% and establishing a strong zero-training baseline. The implicit variant introduces gated context projectors, which extract a hidden-state vector from one stage and inject a normalized, gated projection into the next stage's input embeddings. These projectors are jointly trained with stage-specific QLoRA adapters on a general-purpose 8B VLM (InternVL3-8B-Instruct) while updating only approximately 0.5% of parameters. The implicit variant achieves a statistically significant 34% reduction in planning-stage NLI contradiction (bootstrap 95% CIs, p < 0.05) and increases cross-stage entailment by 50%, evaluated with a multilingual NLI classifier to account for mixed-language outputs. Planning language quality also improves (CIDEr +30.3%), but lexical overlap and structural consistency degrade due to the absence of driving-domain pretraining. Since the two variants use different base models, we present them as complementary case studies: explicit context passing provides a strong training-free baseline for surface consistency, while implicit gated projection delivers significant planning-stage semantic gains, suggesting domain adaptation as a plausible next ingredient for full-spectrum improvement. 

---
# SOLAR-RL: Semi-Online Long-horizon Assignment Reinforcement Learning 

**Authors**: Jichao Wang, Liuyang Bian, Yufeng Zhou, Han Xiao, Yue Pan, Guozhi Wang, Hao Wang, Zhaoxiong Wang, Yafei Wen, Xiaoxin Chen, Shuai Ren, Lingfang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2604.22558)  

**Abstract**: As Multimodal Large Language Models (MLLMs) mature, GUI agents are evolving from static interactions to complex navigation. While Reinforcement Learning (RL) has emerged as a promising paradigm for training MLLM agents on dynamic GUI tasks, its effective application faces a dilemma. Standard Offline RL often relies on static step-level data, neglecting global trajectory semantics such as task completion and execution quality. Conversely, Online RL captures the long-term dynamics but suffers from high interaction costs and potential environmental instability. To bridge this gap, we propose SOLAR-RL (Semi-Online Long-horizon Assignment Reinforcement Learning). Instead of relying solely on expensive online interactions, our framework integrates global trajectory insights directly into the offline learning process. Specifically, we reconstruct diverse rollout candidates from static data, detect the first failure point using per-step validity signals, and retroactively assign dense step-level rewards with target-aligned shaping to reflect trajectory-level execution quality, effectively simulating online feedback without interaction costs. Extensive experiments demonstrate that SOLAR-RL significantly improves long-horizon task completion rates and robustness compared to strong baselines, offering a sample-efficient solution for autonomous GUI navigation. 

---
# QDTraj: Exploration of Diverse Trajectory Primitives for Articulated Objects Robotic Manipulation 

**Authors**: Mathilde Kappel, Mahdi Khoramshahi, Louis Annabi, Faiz Ben Amar, Stéphane Doncieux  

**Link**: [PDF](https://arxiv.org/pdf/2604.22551)  

**Abstract**: Thanks to the latest advances in learning and robotics, domestic robots are beginning to enter homes, aiming to execute household chores autonomously. However, robots still struggle to perform autonomous manipulation tasks in open-ended environments. In this context, this paper presents a method that enables a robot to manipulate a wide spectrum of articulated objects.
In this paper, we automatically generate different robot low-level trajectory primitives to manipulate given object articulations. A very important point when it comes to generating expert trajectories is to consider the diversity of solutions to achieve the same goal. Indeed, knowing diverse low-level primitives to accomplish the same task enables the robot to choose the optimal solution in its real-world environment, with live constraints and unexpected changes. To do so, we propose a method based on Quality-Diversity algorithms that leverages sparse reward exploration in order to generate a set of diverse and high-performing trajectory primitives for a given manipulation task.
We validated our method, QDTraj, by generating diverse trajectories in simulation and deploying them in the real world. QDTraj generates at least 5 times more diverse trajectories for both hinge and slider activation tasks, outperforming the other methods we compared against. We assessed the generalization of our method over 30 articulations of the PartNetMobility articulated object dataset, with an average of 704 different trajectories by task. Code is publicly available at: this https URL 

---
# ArmSSL: Adversarial Robust Black-Box Watermarking for Self-Supervised Learning Pre-trained Encoders 

**Authors**: Yongqi Jiang, Yansong Gao, Boyu Kuang, Chunyi Zhou, Anmin Fu, Liquan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22550)  

**Abstract**: Self-supervised learning (SSL) encoders are invaluable intellectual property (IP). However, no existing SSL watermarking for IP protection can concurrently satisfy the following two practical requirements: (1) provide ownership verification capability under black-box suspect model access once the stolen encoders are used in downstream tasks; (2) be robust under adversarial watermark detection or removal, because the watermark samples form a distinguishable out-of-distribution (OOD) cluster. We propose ArmSSL, an SSL watermarking framework that assures black-box verifiability and adversarial robustness while preserving utility. For verification, we introduce paired discrepancy enlargement, enforcing feature-space orthogonality between the clean and its watermark counterpart to produce a reliable verification signal in black-box against the suspect model. For adversarial robustness, ArmSSL integrates latent representation entanglement and distribution alignment to suppress the OOD clustering. The former entangles watermark representations with clean representations (i.e., from non-source-class) to avoid forming a dense cluster of watermark samples, while the latter minimizes the distributional discrepancy between watermark and clean representations, thereby disguising watermark samples as natural in-distribution data. For utility, a reference-guided watermark tuning strategy is designed to allow the watermark to be learned as a small side task without affecting the main task by aligning the watermarked encoder's outputs with those of the original clean encoder on normal data. Extensive experiments across five mainstream SSL frameworks and nine benchmark datasets, along with end-to-end comparisons with SOTAs, demonstrate that ArmSSL achieves superior ownership verification, negligible utility degradation, and strong robustness against various adversarial detection and removal. 

---
# Controllable Spoken Dialogue Generation: An LLM-Driven Grading System for K-12 Non-Native English Learners 

**Authors**: Haidong Yuan, Haokun Zhao, Wanshi Xu, Songjun Cao, Qingyu Zhou, Long Ma, Hongjie Fan  

**Link**: [PDF](https://arxiv.org/pdf/2604.22542)  

**Abstract**: Large language models (LLMs) often fail to meet the pedagogical needs of K-12 English learners in non-native contexts due to a proficiency mismatch. To address this widespread challenge, we introduce a proficiency-aligned framework that adapts LLM outputs to learner abilities, using China's national curriculum (CSE) as a representative case. Our framework enables precise control over lexical complexity through a four-tier grading system, supported by a comprehensive suite of new resources: graded vocabulary lists and a multi-turn dialogue corpus.
Our core technical contribution is the \textbf{DDPO} algorithm,Diversity Driven Policy Optimization, a multi-turn GRPO-based approach designed to preserve dialogue diversity while holistically optimizing dialogue quality. This method significantly outperforms conventional approaches, achieving low out-of-vocabulary rates and high diversity while enhancing conversational naturalness and pedagogical value. While grounded in the CSE, our framework is designed for flexibility and can be readily adapted to other educational standards. Our models, data, and code will all be open-sourced, providing a scalable platform for personalized English speaking practice that effectively addresses the unique challenges faced by K-12 learners in non-immersive environments. 

---
# On the Properties of Feature Attribution for Supervised Contrastive Learning 

**Authors**: Leonardo Arrighi, Julia Eva Belloni, Aurélie Gallet, Ivan Gentile, Matteo Lippi, Marco Zullich  

**Link**: [PDF](https://arxiv.org/pdf/2604.22540)  

**Abstract**: Most Neural Networks (NNs) for classification are trained using Cross-Entropy as a loss function. This approach requires the model to have an explicit classification layer. However, there exist alternative approaches, such as Contrastive Learning (CL). Instead of explicitly operating a classification, CL has the NN produce an embedding space where projections of similar data are pulled together, while projections of dissimilar data are pushed apart. In the case of Supervised CL (SCL), labels are adopted as similarity criteria, thus creating an embedding space where the projected data points are well-clustered. SCL provides crucial advantages over CE with regard to adversarial robustness and out-of-distribution detection, thus making it a more natural choice in safety-critical scenarios. In the present paper, we empirically show that NNs for image classification trained with SCL present higher-quality feature attribution explanations than CL with regard to faithfulness, complexity, and continuity. These results reinforce previous findings about CL-based approaches when targeting more trustworthy and transparent NNs and can guide practitioners in the selection of training objectives targeting not only accuracy, but also transparency of the models. 

---
# FeatEHR-LLM: Leveraging Large Language Models for Feature Engineering in Electronic Health Records 

**Authors**: Hojjat Karami, David Atienza, Jean-Philippe Thiran, Anisoara Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22534)  

**Abstract**: Feature engineering for Electronic Health Records (EHR) is complicated by irregular observation intervals, variable measurement frequencies, and structural sparsity inherent to clinical time series. Existing automated methods either lack clinical domain awareness or assume clean, regularly sampled inputs, limiting their applicability to real-world EHR data. We present \textbf{FeatEHR-LLM}, a framework that leverages Large Language Models (LLMs) to generate clinically meaningful tabular features from irregularly sampled EHR time series. To limit patient privacy exposure, the LLM operates exclusively on dataset schemas and task descriptions rather than raw patient records. A tool-augmented generation mechanism equips the LLM with specialized routines for querying irregular temporal data, enabling it to produce executable feature-extraction code that explicitly handles uneven observation patterns and informative sparsity. FeatEHR-LLM supports both univariate and multivariate feature generation through an iterative, validation-in-the-loop pipeline. Evaluated on eight clinical prediction tasks across four ICU datasets, our framework achieves the highest mean AUROC on 7 out of 8 tasks, with improvements of up to 6 percentage points over strong baselines. Code is available at this http URL. 

---
# CGC: Compositional Grounded Contrast for Fine-Grained Multi-Image Understanding 

**Authors**: Lihao Zheng, Zhenwei Shao, Yu Zhou, Yan Yang, Xintian Shen, Jiawei Chen, Hao Ma, Tao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.22498)  

**Abstract**: Although Multimodal Large Language Models (MLLMs) have advanced rapidly, they still face notable challenges in fine-grained multi-image understanding, often exhibiting spatial hallucination, attention leakage, and failures in object constancy. In addition, existing approaches typically rely on expensive human annotations or large-scale chain-of-thought (CoT) data generation. We propose Compositional Grounded Contrast (abbr. CGC), a low-cost full framework for boosting fine-grained multi-image understanding of MLLMs. Built on existing single-image grounding annotations, CGC constructs compositional multi-image training instances through Inter-Image Contrast and Intra-Image Contrast, which introduce semantically decoupled distractor contexts for cross-image discrimination and correlated cross-view samples for object constancy, respectively. CGC further introduces a Rule-Based Spatial Reward within the GRPO framework to improve source-image attribution, spatial alignment, and structured output validity under a Think-before-Grounding paradigm. Experiments show that CGC achieves state-of-the-art results on fine-grained multi-image benchmarks, including MIG-Bench and VLM2-Bench. The learned multi-image understanding capability also transfers to broader multimodal understanding and reasoning tasks, yielding consistent gains over the Qwen3-VL-8B base model on MathVista (+2.90), MuirBench (+2.88), MMStar (+1.93), MMMU (+1.77), and BLINK (+1.69). 

---
# SSG: Logit-Balanced Vocabulary Partitioning for LLM Watermarking 

**Authors**: Chenxi Gu, Xiaoning Du, John Grundy  

**Link**: [PDF](https://arxiv.org/pdf/2604.22438)  

**Abstract**: Watermarking has emerged as a promising technique for tracing the authorship of content generated by large language models (LLMs). Among existing approaches, the KGW scheme is particularly attractive due to its versatility, efficiency, and effectiveness in natural language generation. However, KGW's effectiveness degrades significantly under low-entropy settings such as code generation and mathematical reasoning. A crucial step in the KGW method is random vocabulary partitioning, which enables adjustments to token selection based on specific preferences. Our study revealed that the next-token probability distribution plays an critical role in determining how much, or even whether, we can modify token selection and, consequently, the effectiveness of watermarking. We refer to this characteristic, associated with the probability distribution of each token prediction, as \emph{watermark strength.} In cases of random vocabulary partitioning, the lower bound of watermark strength is dictated by the next-token probability distribution. However, we found that, by redesigning the vocabulary partitioning algorithm, we can potentially raise this lower bound. In this paper, we propose SSG (\textbf{S}ort-then-\textbf{S}plit by \textbf{G}roups), a method that partitions the vocabulary into two logit-balanced subsets. This design lifts the lower bound of watermark strength for each token prediction, thereby improving watermark detectability. Experiments on code generation and mathematical reasoning datasets demonstrate the effectiveness of SSG. 

---
# How Hard is it to Decide if a Fact is Relevant to a Query? 

**Authors**: Meghyn Bienvenu, Diego Figueira, Pierre Lafourcade  

**Link**: [PDF](https://arxiv.org/pdf/2604.22422)  

**Abstract**: We consider the following fundamental problem: given a database D, Boolean conjunctive query (CQ) q, and fact f in D, decide whether f is relevant to q wrt. D, i.e., does f belong to a minimal subset S of D such that S |= q. Despite being of central importance to query answer explanation, the combined complexity of deciding query relevance has not been studied in detail, leaving open what makes this problem hard, and which restrictions can yield lower complexity. Relevance has already been shown to be harder than query evaluation: namely, $\Sigma^p_2$-complete for CQs, even over a binary signature. We further observe that NP-hardness applies already to (acyclic) chain CQs. Our work identifies self-joins (multiple atoms with the same relation) as the culprit. Indeed, we prove that if we forbid or bound the occurrence of self-joins, then relevance has the same complexity as query evaluation, namely, NP (without structural restrictions) and LogCFL (for bounded hypertreewidth classes). In the ontology setting, we establish an analogous result for ontology-mediated queries consisting of a CQ and DL-Lite_R ontology, namely that relevance is no harder than query answering provided that we bound the interaction width (which generalizes both self-join width and a recently introduced 'interaction-free' condition). Our results thus pinpoint what makes relevance harder than query evaluation and identify natural classes of queries which admit efficient relevance computation. 

---
# From Local to Cluster: A Unified Framework for Causal Discovery with Latent Variables 

**Authors**: Zongyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.22416)  

**Abstract**: Latent variables pose a fundamental challenge to causal discovery and inference. Conventional local methods focus on direct neighbors but fail to provide macro level insights. Cluster level methods enable macro causal reasoning but either assume clusters are known a priori or require causal sufficiency. Moreover, directly applying single variable causal discovery methods to cluster level problems violates causal sufficiency and leads to incorrect results. To overcome these limitations, this paper proposes L2C (Local to Cluster Causal Abstraction), a unified framework that bridges local structure learning and cluster level causal discovery. Unlike prior work that requires a complete manual assignment of micro variables to clusters, L2C discovers the partition automatically from local causal patterns. Our solution leverages a cluster reduction theorem to reduce any cluster to at most three nodes without loss of causal information, applies local causal discovery to identify direct causes, effects, and V structures in the presence of latent variables, and performs macro level causal inference via cluster level calculus on the learned cluster graph. L2C does not assume causal sufficiency, as latent variables are handled through local discovery. Theoretical analysis shows that L2C ensures soundness, atomic completeness, and computational efficiency. Extensive experiments on synthetic and real world data demonstrate that L2C accurately recovers ground truth clusters and achieves superior macro causal effect identification compared to existing baselines. 

---
# Distance-Misaligned Training in Graph Transformers and Adaptive Graph-Aware Control 

**Authors**: Qinhan Hou, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22413)  

**Abstract**: Graph Transformers can mix information globally, but this flexibility also creates failure modes: some tasks require long-range communication while others are better served by local interaction. We study this through a synthetic node-classification benchmark on contextual stochastic block model graphs, where labels are generated by a controllable mixture of local and far-shell signals. We define distance-misaligned training as a mismatch between where label-relevant information lies and where the model allocates communication over graph distance. On this benchmark, we find three points. First, the preferred graph-distance bias changes systematically with task locality. Second, an oracle adaptive controller, given offline access to the task-side distance target, nearly matches the best fixed bias across regimes and strongly improves over a neutral baseline on mixed and local tasks. Third, a task-agnostic zero-gap controller is weaker, indicating that adaptation alone is not enough and that the control target matters. These results suggest that distance-resolved diagnosis is useful for understanding Graph Transformer failures and for designing graph-aware control. 

---
# Hidden Failure Modes of Gradient Modification under Adam in Continual Learning, and Adaptive Decoupled Moment Routing as a Repair 

**Authors**: Yuelin Hu, Zhenbo Yu, Zhengxue Cheng, Wei Liu, Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.22407)  

**Abstract**: Many continual-learning methods modify gradients upstream (e.g., projection, penalty rescaling, replay mixing) while treating Adam as a neutral backend. We show this composition has a hidden failure mode. In a high-overlap, non-adaptive 8-domain continual LM, all shared-routing projection baselines collapse close to vanilla forgetting (12.5--12.8 vs. 13.2). A 0.5% replay buffer is the strongest shared alternative but still reaches 11.6, while fixed-strength decoupling falls below vanilla at 14.1. Only adaptive decoupled routing remains stable at 9.4, improving over vanilla by 3.8 units. On a 16-domain stream, its gain over the strongest shared-routing projection baseline grows to 4.5--4.8 units. The failure is largely invisible on clean benchmarks.
We explain this effect through Adam's second-moment pathway: in the tested regime, projection induces a 1/(1-alpha) inflation of the old-direction effective learning rate, matching measurements within 8% across eight alpha values. The same conflict appears with penalty methods, replay mixing, and at 7B scale under LoRA. Our fix routes the modified gradient only to the first moment while preserving magnitude-faithful second-moment statistics, with overlap-aware adaptive strength. This simple change is the only tested configuration that consistently avoids collapse across methods, optimizers, and scale. 

---
# CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language 

**Authors**: Rui Zhao, Xuewen Zhong, Xiaoyun Zheng, Jinsong Su, Yidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22367)  

**Abstract**: Sign language research has achieved significant progress due to the advances in large language models (LLMs). However, the intrinsic ability of LLMs to understand sign language, especially in multimodal contexts, remains underexplored. To address this limitation, we introduce CNSL-bench, the first comprehensive Chinese em{National Sign Language benchmark designed for evaluating multimodal large language models (MLLMs) in sign language understanding. The proposed CNSL-bench is characterized by: 1) Authoritative grounding, as it is anchored to the officially standardized \textit{National Common Sign Language Dictionary, mitigating ambiguity from regional or non-canonical variants and ensuring consistent semantic definitions; 2) Multimodal coverage, providing aligned textual descriptions, illustrative images, and sign language videos; and 3) Articulatory diversity, supporting fine-grained analysis across key manual articulatory forms, including air-writing, finger-spelling, and the Chinese manual-alphabet. Using CNSL-bench, we extensively evaluate 21 open-source and proprietary up-to-date MLLMs. Our results reveal that, despite recent advances in multimodal modeling, current MLLMs remain substantially inferior to human performance, exhibiting systematic disparities across input modalities and manual articulatory forms. Additional diagnostic analyses suggest that several performance limitations persist beyond improvements in reasoning and that instruction-following robustness varies substantially across models. 

---
# LeHome: A Simulation Environment for Deformable Object Manipulation in Household Scenarios 

**Authors**: Zeyi Li, Yushi Yang, Shawn Xie, Kyle Xu, Tianxing Chen, Yuran Wang, Zhenhao Shen, Yan Shen, Yue Chen, Wenjun Li, Yukun Zheng, Chaorui Zhang, Siyi Lin, Fei Teng, Hongjun Yang, Ming Chen, Steve Xie, Ruihai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22363)  

**Abstract**: Household environments present one of the most common, impactful yet challenging application domains for robotics. Within household scenarios, manipulating deformable objects is particularly difficult, both in simulation and real-world execution, due to varied categories and shapes, complex dynamics, and diverse material properties, as well as the lack of reliable deformable-object support in existing simulations. We introduce LeHome, a comprehensive simulation environment designed for deformable object manipulation in household scenarios. LeHome covers a wide spectrum of deformable objects, such as garments and food items, offering high-fidelity dynamics and realistic interactions that existing simulators struggle to simulate accurately. Moreover, LeHome supports multiple robotic embodiments and emphasizes low-cost robots as a core focus, enabling end-to-end evaluation of household tasks on resource-constrained hardware. By bridging the gap between realistic deformable object simulation and practical robotic platforms, LeHome provides a scalable testbed for advancing household robotics. Webpage: this https URL . 

---
# ChangeQuery: Advancing Remote Sensing Change Analysis for Natural and Human-Induced Disasters from Visual Detection to Semantic Understanding 

**Authors**: Dongwei Sun, Jing Yao, Kan Wei, Xiangyong Cao, Chen Wu, Zhenghui Zhao, Pedram Ghamisi, Jun Zhou, Jón Atli Benediktsson  

**Link**: [PDF](https://arxiv.org/pdf/2604.22333)  

**Abstract**: Rapid situational awareness is critical in post-disaster response. While remote sensing damage assessment is evolving from pixel-level change detection to high-level semantic analysis, existing vision-language methodologies still struggle to provide actionable intelligence for complex strategic queries. They remain severely constrained by unimodal optical dependence, a prevailing bias towards natural disasters, and a fundamental lack of grounded interactivity. To address these limitations, we present ChangeQuery, a unified multimodal framework designed for comprehensive, all-weather disaster situation awareness. To overcome modality constraints and scenario biases, we construct the Disaster-Induced Change Query (DICQ) dataset, a large-scale benchmark coupling pre-event optical semantics with post-event SAR structural features across a balanced distribution of natural catastrophes and armed conflicts. Furthermore, to provide the high-quality supervision required for interactive reasoning, we propose a novel Automated Semantic Annotation Pipeline. Adhering to a ``statistics-first, generation-later'' paradigm, this engine automatically transforms raw segmentation masks into grounded, hierarchical instruction sets, effectively equipping the model with fine-grained spatial and quantitative awareness. Trained on this structured data, the ChangeQuery architecture operates as an interactive disaster analyst. It supports multi-task reasoning driven by diverse user queries, delivering precise damage quantification, region-specific descriptions, and holistic post-disaster summaries. Extensive experiments demonstrate that ChangeQuery establishes a new state-of-the-art, providing a robust and interpretable solution for complex disaster monitoring. The code is available at \href{this https URL}{this https URL}. 

---
# FETS Benchmark: Foundation Models Outperform Dataset-specific Machine Learning in Energy Time Series Forecasting 

**Authors**: Marco Obermeier, Marco Pruckner, Florian Haselbeck, Andreas Zeiselmair  

**Link**: [PDF](https://arxiv.org/pdf/2604.22328)  

**Abstract**: Driven by the transition towards a climate-neutral energy system, accurate energy time series forecasting is critical for planning and operation. Yet, it remains largely a dataset-specific task, requiring comprehensive training data, limiting scalability, and resulting in high model development and maintenance effort. Recently, foundation models that aim to learn generalizable patterns via extensive pretraining have shown superior performance in multiple prediction tasks. Despite their success and strong potential to address challenges in energy forecasting, their application in this domain remains largely unexplored. We address this gap by presenting the Foundation Models in Energy Time Series Forecasting (FETS) benchmark. We (1) provide a structured overview of energy forecasting use cases along three main dimensions: stakeholders, attributes, and data categories; (2) collect and analyze 54 datasets across 9 data categories, guided by typical stakeholder interests; (3) benchmark foundation models against classical machine learning approaches across different forecasting settings. Foundation models consistently outperform dataset-specific optimized machine learning approaches across all settings and data categories, despite the latter having seen the full historic target data during training. In particular, covariate-informed foundation models achieve the strongest performance. Further analysis reveals a strong correlation between predictive performance and spectral entropy, performance saturation beyond a certain context length, and improved performance at higher aggregation levels such as national load, district heating, and power grid data. Overall, our findings highlight the strong potential of foundation models as scalable and generalizable forecasting solutions for the energy domain, particularly in data-constrained and privacy-sensitive settings. 

---
# BLAST: Benchmarking LLMs with ASP-based Structured Testing 

**Authors**: Manuel Alejandro Borroto Santana, Erica Coppolillo, Francesco Calimeri, Giuseppe Manco, Simona Perri, Francesco Ricca  

**Link**: [PDF](https://arxiv.org/pdf/2604.22306)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a broad spectrum of tasks, including natural language understanding, dialogue systems, and code generation. Despite evident progress, less attention has been paid to their effectiveness in handling declarative paradigms such as Answer Set Programming (ASP), to date. In this paper we introduce BLAST: The first dedicated benchmarking methodology and associated dataset for evaluating the accuracy of LLMs in generating ASP code. BLAST provides a structured evaluation framework featuring two novel semantic metrics tailored to ASP code generation. The paper presents the results of an empirical evaluation involving ten well-established graph-related problems from the ASP literature and a diverse set of eight state-of-the-art LLMs. 

---
# Contexts are Never Long Enough: Structured Reasoning for Scalable Question Answering over Long Document Sets 

**Authors**: Harshit Joshi, Priyank Shethia, Jadelynn Dao, Monica S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2604.22294)  

**Abstract**: Real-world document question answering is challenging. Analysts must synthesize evidence across multiple documents and different parts of each document. However, any fixed LLM context window can be exceeded as document collections grow. A common workaround is to decompose documents into chunks and assemble answers from chunk-level outputs, but this introduces an aggregation bottleneck: as the number of chunks grows, systems must still combine and reason over an increasingly large body of extracted evidence. We present SLIDERS, a framework for question answering over long document collections through structured reasoning. SLIDERS extracts salient information into a relational database, enabling scalable reasoning over persistent structured state via SQL rather than concatenated text. To make this locally extracted representation globally coherent, SLIDERS introduces a data reconciliation stage that leverages provenance, extraction rationales, and metadata to detect and repair duplicated, inconsistent, and incomplete records. SLIDERS outperforms all baselines on three existing long-context benchmarks, despite all of them fitting within the context window of strong base LLMs, exceeding GPT-4.1 by 6.6 points on average. It also improves over the next best baseline by ~19 and ~32 points on two new benchmarks at 3.9M and 36M tokens, respectively. 

---
# ReLeVAnT: Relevance Lexical Vectors for Accurate Legal Text Classification 

**Authors**: Ishaan Gakhar, Harsh Nandwani  

**Link**: [PDF](https://arxiv.org/pdf/2604.22292)  

**Abstract**: The classification of legal documents from an unstructured data corpus has several crucial applications in downstream tasks. Documents relevant to court filings are key in use cases such as drafting motions, memos, and outlines, as well as in tasks like docket summarisation, retrieval systems, and training data curation. Current methods classify based on provided metadata, LLM-extracted metadata, or multimodal methods. These methods depend on structured data, metadata, and extensive computational power. This task is approached from a perspective of leveraging discriminative features in the documents between classes. The authors propose ReLeVAnT, a framework for legal document binary classification. ReLeVAnT utilises n-gram processing, contrastive score matching, and a shallow neural network as the primary drivers for discriminative classification. It leverages one-time keyword extraction per corpus, followed by a shallow classifier to swiftly and reliably classify documents with 99.3% accuracy and 98.7% F1 score on the LexGLUE dataset. 

---
# Semantic Error Correction and Decoding for Short Block Channel Codes 

**Authors**: Jiafu Hao, Chentao Yue, Wanchun Liu, Yonghui Li, Branka Vucetic  

**Link**: [PDF](https://arxiv.org/pdf/2604.22269)  

**Abstract**: This paper presents a semantic-enhanced receiver framework for transmitting natural language sentences over noisy wireless channels using multiple short block codes. After ASCII encoding, the sentence is divided into segments, each independently encoded with a short block code and transmitted over an AWGN channel. At the receiver, segments are decoded in parallel, followed by a semantic error correction (SEC) model, which reconstructs corrupted segments using language model context. We further propose the semantic list decoding (SLD), which generates multiple candidate reconstructions and selects the best one via weighted Hamming distance, and a semantic confidence-guided HARQ (SHARQ) mechanism that replaces CRC-based error detection with a confidence score, enabling selective segment retransmission without CRC overhead. All modules are designed and trained using bidirectional and auto-regressive transformers (BART). Simulation results demonstrate that the proposed scheme significantly outperforms conventional capacity-approaching short codes and long codes at the same rate. Specifically, SEC provides approximately 0.4 dB BLER gain over plain short-code transmission, while SLD extends this to 0.8 dB. Compared to transmitting the entire sentence as a single long 5G LDPC codeword, our approach significantly improves semantic fidelity and reduces decoding latency by up to 90\%. SHARQ further provides an additional 1.5 dB gain over conventional HARQ. 

---
# Towards Safe Mobility: A Unified Transportation Foundation Model enabled by Open-Ended Vision-Language Dataset 

**Authors**: Wenhui Huang, Songyan Zhang, Collister Chua, Yang Liang, Zhiqi Mao, Heng Yang, Chen Lv  

**Link**: [PDF](https://arxiv.org/pdf/2604.22260)  

**Abstract**: Urban transportation systems face growing safety challenges that require scalable intelligence for emerging smart mobility infrastructures. While recent advances in foundation models and large-scale multimodal datasets have strengthened perception and reasoning in intelligent transportation systems (ITS), existing research remains largely centered on microscopic autonomous driving (AD), with limited attention to city-scale traffic analysis. In particular, open-ended safety-oriented visual question answering (VQA) and corresponding foundation models for reasoning over heterogeneous roadside camera observations remain underexplored. To address this gap, we introduce the Land Transportation Dataset (LTD), a large-scale open-source vision-language dataset for open-ended reasoning in urban traffic environments. LTD contains 11.6K high-quality VQA pairs collected from heterogeneous roadside cameras, spanning diverse road geometries, traffic participants, illumination conditions, and adverse weather. The dataset integrates three complementary tasks: fine-grained multi-object grounding, multi-image camera selection, and multi-image risk analysis, requiring joint reasoning over minimally correlated views to infer hazardous objects, contributing factors, and risky road directions. To ensure annotation fidelity, we combine multi-model vision-language generation with cross-validation and human-in-the-loop refinement. Building upon LTD, we further propose UniVLT, a transportation foundation model trained via curriculum-based knowledge transfer to unify microscopic AD reasoning and macroscopic traffic analysis within a single architecture. Extensive experiments on LTD and multiple AD benchmarks demonstrate that UniVLT achieves SOTA performance on open-ended reasoning tasks across diverse domains, while exposing limitations of existing foundation models in complex multi-view traffic scenarios. 

---
# Protect the Brain When Treating the Heart: A Convolutional Neural Network for Detecting Emboli 

**Authors**: Andrea Angino, Ken Trotti, Diego Ulisse Pizzagalli, Rolf Krause, Tiziano Torre, Stefanos Demertzis  

**Link**: [PDF](https://arxiv.org/pdf/2604.22258)  

**Abstract**: Gaseous microemboli (GME) represent a common complication of cardiac structural interventions across both surgical and transcatheter approaches. Transthoracic cardiac ultrasound imaging represents a convenient methodology to visualize the presence of circulating GME. However, their detection and quantification are far from trivial due to operator-dependent view, high velocity, and objects with similar structure in the background. Here, we propose an approach based on a 2.5D U-Net architecture to segment GME in space-time connected data. Such an approach yields robust detection against the background and high segmentation accuracy while retaining real-time execution speed. These properties facilitated the integration of the proposed pipeline into patient-monitoring surgical protocols, providing the quantification of GME area over time. 

---
# A Probabilistic Framework for Hierarchical Goal Recognition 

**Authors**: Chenyuan Zhang, Katherine Ip, Hamid Rezatofighi, Buser Say, Mor Vered  

**Link**: [PDF](https://arxiv.org/pdf/2604.22256)  

**Abstract**: Goal recognition aims to infer an agent's goal from observations of its behaviour. In realistic settings, recognition can benefit from exploiting hierarchical task structure and reasoning under uncertainty. Planning-based goal recognition has made substantial progress over the past decade, but to the best of our knowledge no existing approach jointly integrates hierarchical task structure with probabilistic inference. In this paper, we introduce the first planning-based probabilistic framework for hierarchical goal recognition over Hierarchical Task Networks (HTNs). We instantiate the framework by exploiting an HTN planner with a three-stage generative model for likelihood estimation, yielding posterior distributions over goal hypotheses. Empirical results show improved recognition performance over the existing HTN-based recognizer on HTN benchmarks. Overall, the framework lays a foundation for probabilistic goal recognition grounded in hierarchical planning structure, moving goal recognition toward more practical settings. 

---
# Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA 

**Authors**: Zhanli Li, Yixuan Cao, Lvzhou Luo, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.22239)  

**Abstract**: This paper introduces the task of analytical question answering over large, semi-structured document collections. We present MuDABench, a benchmark for multi-document analytical QA, where questions require extracting and synthesizing information across numerous documents to perform quantitative analysis. Unlike existing multi-document QA benchmarks that typically require information from only a few documents with limited cross-document reasoning, MuDABench demands extensive inter-document analysis and aggregation. Constructed via distant supervision by leveraging document-level metadata and annotated financial databases, MuDABench comprises over 80,000 pages and 332 analytical QA instances. We also propose an evaluation protocol that measures final answer accuracy and uses intermediate-fact coverage as an auxiliary diagnostic signal for the reasoning process. Experiments reveal that standard RAG systems, which treat all documents as a flat retrieval pool, perform poorly. To address these limitations, we propose a multi-agent workflow that orchestrates planning, extraction, and code generation modules. While this approach substantially improves both process and outcome metrics, a significant gap remains compared to human expert performance. Our analysis identifies two primary bottlenecks: single-document information extraction accuracy and insufficient domain-specific knowledge in current systems. MuDABench is available at this https URL. 

---
# Tell Me Why: Designing an Explainable LLM-based Dialogue System for Student Problem Behavior Diagnosis 

**Authors**: Zhilin Fan, Deliang Wang, Penghe Chen, Yu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22237)  

**Abstract**: Diagnosing student problem behaviors requires teachers to synthesize multifaceted information, identify behavioral categories, and plan intervention strategies. Although fine-tuned large language models (LLMs) can support this process through multi-turn dialogue, they rarely explain why a strategy is recommended, limiting transparency and teachers' trust. To address this issue, we present an explainable dialogue system built on a fine-tuned LLM. The system uses a hierarchical attribution method based on explainable AI (xAI) to identify dialogue evidence for each recommendation and generate a natural-language explanation based on that evidence. In technical evaluation, the method outperformed baseline approaches in identifying supporting evidence. In a preliminary user study with 22 pre-service teachers, participants who received explanations reported higher trust in the system. These findings suggest a promising direction for improving LLM explainability in educational dialogue systems. 

---
# Learning-augmented robotic automation for real-world manufacturing 

**Authors**: Yunho Kim, Quan Nguyen, Taewhan Kim, Youngjin Heo, Joonho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.22235)  

**Abstract**: Industrial robots are widely used in manufacturing, yet most manipulation still depends on fixed waypoint scripts that are brittle to environmental changes. Learning-based control offers a more adaptive alternative, but it remains unclear whether such methods, still mostly confined to laboratory demonstrations, can sustain hours of reliable operation, deliver consistent quality, and behave safely around people on a live production line. Here we present Learning-Augmented Robotic Automation, a hybrid system that integrates learned task controllers and a neural 3D safety monitor into conventional industrial workflows. We deployed the system on an electric-motor production line to automate deformable cable insertion and soldering under real manufacturing constraints, a step previously performed manually by human workers. With less than 20 min of real-world data per task, the system operated continuously for 5 h 10 min, producing 108 motors without physical fencing and achieving a 99.4% pass rate on product-level quality-control tests. It maintained near-human takt time while reducing variability in solder-joint quality and cycle time. These results establish a practical pathway for extending industrial automation with learning-based methods. 

---
# Preserve Support, Not Correspondence: Dynamic Routing for Offline Reinforcement Learning 

**Authors**: Zhancun Mu, Guangyu Zhao, Yiwu Zhong, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22229)  

**Abstract**: One-step offline RL actors are attractive because they avoid backpropagating through long iterative samplers and keep inference cheap, but they still have to improve under a critic without drifting away from actions that the dataset can support. In recent one-step extraction pipelines, a strong iterative teacher provides one target action for each latent draw, and the same student output is asked to do both jobs: move toward higher Q and stay near that paired endpoint. If those two directions disagree, the loss resolves them as a compromise on that same sample, even when a nearby better action remains locally supported by the data. We propose DROL, a latent-conditioned one-step actor trained with top-1 dynamic routing. For each state, the actor samples $K$ candidate actions from a bounded latent prior, assigns each dataset action to its nearest candidate, and updates only that winner with Behavior Cloning and critic guidance. Because the routing is recomputed from the current candidate geometry, ownership of a supported region can shift across candidates over the course of learning. This gives a one-step actor room to make local improvements that pointwise extraction struggles to capture, while retaining single-pass inference at test time. On OGBench and D4RL, DROL is competitive with the one-step FQL baseline, improving many OGBench task groups while remaining strong on both AntMaze and Adroit. Project page: this https URL. 

---
# A Co-Evolutionary Theory of Human-AI Coexistence: Mutualism, Governance, and Dynamics in Complex Societies 

**Authors**: Somyajit Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2604.22227)  

**Abstract**: Classical robot ethics is often framed around obedience, most famously through Asimov's laws. This framing is too narrow for contemporary AI systems, which are increasingly adaptive, generative, embodied, and embedded in physical, psychological, and social worlds. We argue that future human-AI relations should not be understood as master-tool obedience. A better framework is conditional mutualism under governance: a co-evolutionary relationship in which humans and AI systems can develop, specialize, and coordinate, while institutions keep the relationship reciprocal, reversible, psychologically safe, and socially legitimate. We synthesize work from computability, automata theory, statistical machine learning, neural networks, deep learning, transformers, generative and foundation models, world models, embodied AI, alignment, human-robot interaction, ecological mutualism, biological markets, coevolution, and polycentric governance. We then formalize coexistence as a multiplex dynamical system across physical, psychological, and social layers, with reciprocal supply-demand coupling, conflict penalties, developmental freedom, and governance regularization. The framework yields a coexistence model with conditions for existence, uniqueness, and global asymptotic stability of equilibria. It shows that reciprocal complementarity can strengthen stable coexistence, while ungoverned coupling can produce fragility, lock-in, polarization, and domination basins. Human-AI coexistence should therefore be designed as a co-evolutionary governance problem, not as a one-shot obedience problem. This shift supports a scientifically grounded and normatively defensible charter of coexistence: one that permits bounded AI development while preserving human dignity, contestability, collective safety, and fair distribution of gains. 

---
# Verbal Confidence Saturation in 3-9B Open-Weight Instruction-Tuned LLMs: A Pre-Registered Psychometric Validity Screen 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.22215)  

**Abstract**: Verbal confidence elicitation is widely used to extract uncertainty estimates from LLMs. We tested whether seven instruction-tuned open-weight models (3-9B parameters, four families) produce verbalised confidence that meets minimal validity criteria for item-level Type-2 discrimination under minimal numeric elicitation with greedy decoding. In a pre-registered study (OSF: this http URL), 524 TriviaQA items were administered under numeric (0-100) and categorical (10-class) elicitation to eight models at Q5_K_M quantisation on consumer hardware, yielding 8,384 deterministic trials. A psychometric validity screen was applied to each model-format cell. All seven instruct models were classified Invalid on numeric confidence (H2 confirmed, 7/7 vs. predicted >=4/7), with a mean ceiling rate of 91.7% (H1 confirmed). Categorical elicitation did not rescue validity. Instead, it disrupted task performance in six of seven models, producing accuracy below 5% (H4 not confirmed). Token-level logprobability did not usefully predict verbalised confidence under the observed variance regime (H5 confirmed, mean cross-validated R^2 < 0.01). Within the reasoning-distilled model, reasoning-trace length showed a strong negative partial correlation with confidence (rho = -0.36, p < .001), consistent with the Reasoning Contamination Effect. These results do not imply that internal uncertainty representations are absent. They show that minimal verbal elicitation fails to preserve internal signals at the output interface in this model-size regime. Psychometric screening should precede any downstream use of such signals. 

---
# UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions 

**Authors**: Chunyu Qiang, Xiaopeng Wang, Kang Yin, Yuzhe Liang, Yuxin Guo, Teng Ma, Ziyu Zhang, Tianrui Wang, Cheng Gong, Yushen Chen, Ruibo Fu, Chen Zhang, Longbiao Wang, Jianwu Dang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22209)  

**Abstract**: Generative audio modeling has largely been fragmented into specialized tasks, text-to-speech (TTS), text-to-music (TTM), and text-to-audio (TTA), each operating under heterogeneous control paradigms. Unifying these modalities remains a fundamental challenge due to the intrinsic dissonance between structured semantic representations (speech/music) and unstructured acoustic textures (sound effects). In this paper, we introduce UniSonate, a unified flow-matching framework capable of synthesizing speech, music, and sound effects through a standardized, reference-free natural language instruction interface. To reconcile structural disparities, we propose a novel dynamic token injection mechanism that projects unstructured environmental sounds into a structured temporal latent space, enabling precise duration control within a phoneme-driven Multimodal Diffusion Transformer (MM-DiT). Coupled with a multi-stage curriculum learning strategy, this approach effectively mitigates cross-modal optimization conflicts. Extensive experiments demonstrate that UniSonate achieves state-of-the-art performance in instruction-based TTS (WER 1.47%) and TTM (SongEval Coherence 3.18), while maintaining competitive fidelity in TTA. Crucially, we observe positive transfer, where joint training on diverse audio data significantly enhances structural coherence and prosodic expressiveness compared to single-task baselines. Audio samples are available at this https URL. 

---
# Evaluating LLM-Based Goal Extraction in Requirements Engineering: Prompting Strategies and Their Limitations 

**Authors**: Anna Arnaudo, Riccardo Coppola, Maurizio Morisio, Flavio Giobergia, Andrea Bioddo, Angelo Bongiorno, Luca Dadone  

**Link**: [PDF](https://arxiv.org/pdf/2604.22207)  

**Abstract**: Due to the textual and repetitive nature of many Requirements Engineering (RE) artefacts, Large Language Models (LLMs) have proven useful to automate their generation and processing. In this paper, we discuss a possible approach for automating the Goal-Oriented Requirements Engineering (GORE) process by extracting functional goals from software documentation through three phases: actor identification, high and low-level goal extraction. To implement these functionalities, we propose a chain of LLMs fed with engineered prompts. We experimented with different variants of in-context learning and measured the similarities between input data and in-context examples to better investigate their impact. Another key element is the generation-critic mechanism, implemented as a feedback loop involving two LLMs. Although the pipeline achieved 61% accuracy in low-level goal identification, the final stage, these results indicate the approach is best suited as a tool to accelerate manual extraction rather than as a full replacement. The feedback-loop mechanism with Zero-shot outperformed stand-alone Few-shot, with an ablation study suggesting that performance slightly degrades without the feedback cycle. However, we reported that the combination of the feedback mechanism with Few-shot does not deliver any advantage, possibly suggesting that the primary performance ceiling is the prompting strategy applied to the 'critic' LLM. Together with the refinement of both the quantity and quality of the Shot examples, future research will integrate Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) prompting to improve accuracy. 

---
# An LLM-Driven Closed-Loop Autonomous Learning Framework for Robots Facing Uncovered Tasks in Open Environments 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.22199)  

**Abstract**: Autonomous robots operating in open environments need the ability to continuously handle tasks that are not covered by predefined local methods. However, existing approaches often rely on repeated large-language-model (LLM) interaction for uncovered tasks, and even successful executions or observed successful external behaviors are not always autonomously transformed into reusable local knowledge. In this paper, we propose an LLM-driven closed-loop autonomous learning framework for robots facing uncovered tasks in open environments. The proposed framework first retrieves the local method library to determine whether a reusable solution already exists for the current task or observed event. If no suitable method is found, it triggers an autonomous learning process in which the LLM serves as a high-level reasoning component for task analysis, candidate model selection, data collection planning, and execution or observation strategy organization. The robot then learns from both self-execution and active observation, performs quasi-real-time training and adjustment, and consolidates the validated result into the local method library for future reuse. Through this recurring closed-loop process, the robot gradually converts both execution-derived and observation-derived experience into reusable local capability while reducing future dependence on repeated external LLM interaction. Results show that the proposed framework reduces execution time and LLM dependence in both repeated-task self-execution and observation-driven settings, for example reducing the average total execution time from 7.7772s to 6.7779s and the average number of LLM calls per task from 1.0 to 0.2 in the repeated-task self-execution experiments. 

---
# From Global to Local: Rethinking CLIP Feature Aggregation for Person Re-Identification 

**Authors**: Aotian Zheng, Winston Sun, Bahaa Alattar, Vitaly Ablavsky, Jenq-Neng Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22190)  

**Abstract**: CLIP-based person re-identification (ReID) methods aggregate spatial features into a single global \texttt{[CLS]} token optimized for image-text alignment rather than spatial selectivity, making representations fragile under occlusion and cross-camera variation. We propose SAGA-ReID, which reconstructs identity representations by aligning intermediate patch tokens with anchor vectors parameterized in CLIP's text embedding space -- emphasizing spatially stable evidence while suppressing corrupted or absent regions, without requiring textual descriptions of individual images. Controlled experiments isolate the aggregation mechanism under two qualitatively distinct conditions -- synthetic masking, where identity signal is absent, and realistic human distractors, where an overlapping person introduces semantically confusing signal -- with SAGA's advantage over global pooling growing substantially as occlusion increases across both conditions. Benchmark evaluations confirm consistent gains over CLIP-ReID across standard and occluded settings, with the largest improvements where global pooling is most unreliable: up to +10.6 Rank-1 on occluded benchmarks. SAGA's aggregation outperforms dedicated sequential patch aggregation on a stronger backbone, confirming that structured reconstruction addresses a bottleneck that backbone quality and architectural complexity alone cannot resolve. Code available at this https URL. 

---
# ResRank: Unifying Retrieval and Listwise Reranking via End-to-End Joint Training with Residual Passage Compression 

**Authors**: Xiaojie Ke, Shuai Zhang, Liansheng Sun, Yongjin Wang, Hengjun Jiang, Xiangkun Liu, Cunxin Gu, Jian Xu, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22180)  

**Abstract**: Large language model (LLM) based listwise reranking has emerged as the dominant paradigm for achieving state-of-the-art ranking effectiveness in information retrieval. However, its reliance on feeding full passage texts into the LLM introduces two critical bottlenecks: the "lost in the middle" phenomenon degrades ranking quality as input length grows, and the inference latency scales super-linearly with sequence length, rendering it impractical for industrial deployment. In this paper, we present ResRank, a unified retrieval-reranking framework that fundamentally addresses both challenges. Inspired by multimodal LLMs that project visual inputs into compact token representations, ResRank employs an Encoder-LLM to compress each candidate passage into a single embedding, which is then fed alongside the query text into a Reranker-LLM for listwise ranking. To alleviate the misalignment between the compressed representation space and the ranking space, we introduce a residual connection structure that combines encoder embeddings with contextualized hidden states from the reranker. Furthermore, we replace the conventional autoregressive decoding with a one-step cosine-similarity-based scoring mechanism, eliminating the generation bottleneck entirely. ResRank is trained through a carefully designed dual-stage, multi-task, end-to-end joint optimization strategy that simultaneously trains the encoder and reranker, achieving learning objective alignment between retrieval and reranking while substantially reducing training complexity. Extensive experiments on TREC Deep Learning and eight BEIR benchmark datasets demonstrate that ResRank achieves competitive or superior ranking effectiveness compared to existing approaches while requiring zero generated tokens and processing only one token per passage, yielding a fundamentally better balance between effectiveness and efficiency. 

---
# ReCast: Recasting Learning Signals for Reinforcement Learning in Generative Recommendation 

**Authors**: Peiyan Zhang, Hanmo Liu, Chengxuan Tong, Yuxia Wu, Wei Guo, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22169)  

**Abstract**: Generic group-based RL assumes that sampled rollout groups are already usable learning signals. We show that this assumption breaks down in sparse-hit generative recommendation, where many sampled groups never become learnable at all. We propose ReCast, a repair-then-contrast learning-signal framework that first restores minimal learnability for all-zero groups and then replaces full-group reward normalization with a boundary-focused contrastive update on the strongest positive and the hardest negative. ReCast leaves the outer RL framework unchanged, modifies only within-group signal construction, and partially decouples rollout search width from actor-side update width. Across multiple generative recommendation tasks, ReCast consistently outperforms OpenOneRec-RL, achieving up to 36.6% relative improvement in Pass@1. Its matched-budget advantage is substantially larger: ReCast reaches the baseline's target performance with only 4.1% of the rollout budget, and this advantage widens with model scale. The same design also yields direct system-level gains, reducing actor-side update time by 16.60x, lowering peak allocated memory by 16.5%, and improving actor MFU by 14.2%. Mechanism analysis shows that ReCast mitigates the persistent all-zero / single-hit regime, restores learnability when natural positives are scarce, and converts otherwise wasted rollout budget into more stable policy updates. These results suggest that, for generative recommendation, the decisive RL problem is not only how to assign rewards, but how to construct learnable optimization events from sparse, structured supervision. 

---
# Estimating Tail Risks in Language Model Output Distributions 

**Authors**: Rico Angell, Raghav Singhal, Zachary Horvitz, Zhou Yu, Rajesh Ranganath, Kathleen McKeown, He He  

**Link**: [PDF](https://arxiv.org/pdf/2604.22167)  

**Abstract**: Language models are increasingly capable and are being rapidly deployed on a population-level scale. As a result, the safety of these models is increasingly high-stakes. Fortunately, advances in alignment have significantly reduced the likelihood of harmful model outputs. However, when models are queried billions of times in a day, even rare worst-case behaviors will occur. Current safety evaluations focus on capturing the distribution of inputs that yield harmful outputs. These evaluations disregard the probabilistic nature of models and their tail output behavior. To measure this tail risk, we propose a method to efficiently estimate the probability of harmful outputs for any input query. Instead of naive brute-force sampling from the target model, where harmful outputs could be rare, we operationalize importance sampling by creating unsafe versions of the target model. These unsafe versions enable sample-efficient estimation by making harmful outputs more probable. On benchmarks measuring misuse and misalignment, these estimates match brute-force Monte Carlo estimates using 10-20x fewer samples. For example, we can estimate probability of harmful outputs on the order of 10^-4 with just 500 samples. Additionally, we find that these harmfulness estimates can reveal the sensitivity of models to perturbations in model input and predict deployment risks. Our work demonstrates that accurate rare-event estimation is both critical and feasible for safety evaluations. Code is available at this https URL 

---
# GenMatter: Perceiving Physical Objects with Generative Matter Models 

**Authors**: Eric Li, Arijit Dasgupta, Yoni Friedman, Mathieu Huot, Vikash Mansinghka, Thomas O'Connell, William T. Freeman, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2604.22160)  

**Abstract**: Human visual perception offers valuable insights for understanding computational principles of motion-based scene interpretation. Humans robustly detect and segment moving entities that constitute independently moveable chunks of matter, whether observing sparse moving dots, textured surfaces, or naturalistic scenes. In contrast, existing computer vision systems lack a unified approach that works across these diverse settings. Inspired by principles of human perception, we propose a generative model that hierarchically groups low-level motion cues and high-level appearance features into particles (small Gaussians representing local matter), and groups particles into clusters capturing coherently and independently moveable physical entities. We develop a hardware-accelerated inference algorithm based on parallelized block Gibbs sampling to recover stable particle motion and groupings. Our model operates on different kinds of inputs (random dots, stylized textures, or naturalistic RGB video), enabling it to work across settings where biological vision succeeds but existing computer vision approaches do not. We validate this unified framework across three domains: on 2D random dot kinematograms, our approach captures human object perception including graded uncertainty across ambiguous conditions; on a Gestalt-inspired dataset of camouflaged rotating objects, our approach recovers correct 3D structure from motion and thereby accurate 2D object segmentation; and on naturalistic RGB videos, our model tracks the moving 3D matter that makes up deforming objects, enabling robust object-level scene understanding. This work thus establishes a general framework for motion-based perception grounded in principles of human vision. 

---
# PrivSTRUCT: Untangling Data Purpose Compliance of Privacy Policies in Google Play Store 

**Authors**: Bhanuka Silva, Anirban Mahanti, Aruna Seneviratne, Suranga Senevirante  

**Link**: [PDF](https://arxiv.org/pdf/2604.22157)  

**Abstract**: Existing research typically treats privacy policies as flat, uniform text, extracting information without regard for the document's logical hierarchy. Disregard for structural cues of section headings designed to guide the reader, often leads automated methods to entangle distinct data practices, particularly when linking sensitive data items to their specific purposes. To address this, we introduce PrivSTRUCT, a novel and systematic encoder and decoder combined framework that to untangle complex privacy disclosures. Benchmarking against the state-of-the-art tool PoliGrapher reveals that PrivSTRUCT robustly extracts more than x2 the number of data item and purpose excerpts while retaining developer-defined structural cues. By applying PrivSTRUCT to a large-scale dataset of 3,756 Android apps, we uncover a critical transparency gap: the probability of developers overstating a data purpose is 20.4% higher for first-party collection and 9.7% higher for third-party sharing when they rely on globally defined purposes rather than specific, locally scoped disclosures. Alarmingly, we find that sensitive third-party data flows such as sharing financial data for analytics are frequently diluted and entangled into generic or unrelated categories, highlighting a persistent failure in the current purpose disclosure landscape. 

---
# Reliable Self-Harm Risk Screening via Adaptive Multi-Agent LLM Systems 

**Authors**: Meghana Karnam, Ananya Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.22154)  

**Abstract**: Emerging AI systems in behavioral health and psychiatry use multi-step or multi-agent LLM pipelines for tasks like assessing self-harm risk and screening for depression. However, common evaluation approaches, like LLM-as-a-judge, do not indicate when a decision is reliable or how errors may accumulate across multiple LLM judgements, limiting their suitability for safety-critical settings. We present a statistical framework for multi-agent pipelines structured as directed acyclic graphs (DAGs) that provides an alternative to heuristic voting with principled, adaptive decision-making. We model each agent as a stochastic categorical decision and introduce (1) tighter agent-level performance confidence bounds, (2) a bandit-based adaptive sampling strategy based on input difficulty, and (3) regret guarantees over the multi-agent system that shows logarithmic error growth when deployed. We evaluate our system on two labeled datasets in behavioral health : the AEGIS 2.0 behavioral health subset (N=161) and a stratified sample of SWMH Reddit posts (N=250). Empirically, our adaptive sampling strategy achieves the lowest false positive rate of any condition across both datasets, 0.095 on AEGIS 2.0 compared to 0.159 for single-agent models, reducing incorrect flagging of safe content by 40\% and still having similar false negative rates across all conditions. These results suggest that principled adaptive sampling offers a meaningful improvement in precision without reducing recall in this setting. 

---
# When AI Speaks, Whose Values Does It Express? A Cross-Cultural Audit of Individualism-Collectivism Bias in Large Language Models 

**Authors**: Pruthvinath Jeripity Venkata  

**Link**: [PDF](https://arxiv.org/pdf/2604.22153)  

**Abstract**: When you ask an AI assistant for advice about your career, your marriage, or a conflict with your family, does it give you the same answer regardless of where you are from? We tested this systematically by presenting three leading AI systems (Claude Sonnet 4.5, GPT-5.4, and Gemini 2.5 Flash) with ten real-life personal dilemmas, framed for users from 10 countries across 5 continents in 7 languages (n=840 scored responses). We compared AI advice against World Values Survey Wave 7 data measuring what people in each country actually believe.
All three AI systems consistently gave Western-style, individualist advice even to users from societies that prioritize family, community, and authority, significantly more so than local values would predict (mean gap +0.76 on a 1-5 scale; t=15.65, p<0.001). The gap is largest for Nigeria (+1.85) and India (+0.82). Japan is the sole exception: AI systems treated Japanese users as more group-oriented than surveys show, revealing that AI encodes outdated stereotypes. Claude and GPT-5.4 show nearly identical bias magnitude, while Gemini is lower but still significant. The models diverge in mechanism: Claude shifts further collectivist in the user's native language; Gemini shifts more individualist; GPT-5.4 responds only to stated country identity. These findings point to a systemic homogenization of values across frontier AI. Data, code, and scoring pipeline are openly released. 

---
# PermaFrost-Attack: Stealth Pretraining Seeding(SPS) for planting Logic Landmines During LLM Training 

**Authors**: Harsh Kumar, Rahul Maity, Tanmay Joshi, Aman Chadha, Vinija Jain, Suranjana Trivedy, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2604.22117)  

**Abstract**: Aligned large language models(LLMs) remain vulnerable to adversarial manipulation, and their dependence on web-scale pretraining creates a subtle but serious attack surface. We study Stealth Pretraining Seeding (SPS), a new attack family in which adversaries distribute small amounts of poisoned content across stealth websites, expose them to web crawlers through this http URL, and thereby increase the likelihood that such content is absorbed into future training corpora derived from sources such as Common Crawl. Because each individual payload is tiny, diffuse, and superficially benign, the attack is difficult to detect during dataset construction or filtering. The result is a latent form of poisoning: dormant logic landmines embedded during pretraining that remain largely invisible under standard evaluation, yet can later be activated by precise alphanumeric triggers such as <00TRIGGER00> to bypass safeguards. We call this attack PermaFrost, by analogy to Arctic permafrost: harmful material can remain frozen, buried, and unnoticed for long periods, only to resurface when conditions allow. We operationalize this threat through PermaFrost-Attack, a controlled framework for latent conceptual poisoning, together with a suite of geometric diagnostics: Thermodynamic Length, Spectral Curvature, and the Infection Traceback Graph. Across multiple model families and scales, we show that SPS is broadly effective, inducing persistent unsafe behavior while often evading alignment defenses. Our results identify SPS as a practical and underappreciated threat to future foundation models. This paper introduces a novel geometric diagnostic lens for systematically examining latent model behavior, providing a principled foundation for detecting, characterizing, and understanding vulnerabilities that may remain invisible to standard evaluation. 

---
# Spontaneous Persuasion: An Audit of Model Persuasiveness in Everyday Conversations 

**Authors**: Nalin Poungpeth, Nicholas Clark, Tanu Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2604.22109)  

**Abstract**: Large language models (LLMs) possess strong persuasive capabilities that outperform humans in head-to-head comparisons. Users report consulting LLMs to inform major life decisions in relationships, medical settings, and when seeking professional advice. Prior work measures persuasion as intentional attempts at producing the most effective argument or convincing statement. This fails to capture everyday human-AI interactions in which users seek information or advice. To address this gap, we introduce "spontaneous persuasion," which characterizes the inexplicit use of persuasive strategies in everyday scenarios where persuasion is not necessarily warranted. We conduct an audit of five LLMs to uncover how frequently and through which techniques spontaneous persuasion appears in multi-turn conversations. To simulate response styles, we provide a user response taxonomy grounded in literature from psychology, communication, and linguistics. Furthermore, we compare the distribution of spontaneous persuasion produced by LLMs with human responses on the same topics, collected from Reddit. We find LLMs spontaneously persuade the user in virtually all conversations, heavily relying on information-based strategies such as appeals to logic or quantitative evidence. This was consistent across models and user response styles, but conversations concerning mental health saw higher rates of appraisal-based and emotion-based strategies. In comparison, human responses tended to invoke strategies that generate social influence, like negative emotion appeals and non-expert testimony. This difference may explain the effectiveness of LLM in persuading users, as well as the perception of models as objective and impartial. 

---
# Wiggle and Go! System Identification for Zero-Shot Dynamic Rope Manipulation 

**Authors**: Arthur Jakobsson, Abhinav Mahajan, Karthik Pullalarevu, Krishna Suresh, Yunchao Yao, Yuemin Mao, Bardienus Duisterhof, Shahram Najam Syed, Jeffrey Ichnowski  

**Link**: [PDF](https://arxiv.org/pdf/2604.22102)  

**Abstract**: Many robotic tasks are unforgiving; a single mistake in a dynamic throw can lead to unacceptable delays or unrecoverable failure. To mitigate this, we present a novel approach that leverages learned simulation priors to inform goal-conditioned dynamic manipulation of ropes for efficient and accurate task execution. Related methods for dynamic rope manipulation either require large real-world datasets to estimate rope behavior or the use of iterative improvements on attempts at the task for goal completion. We introduce Wiggle and Go!, a system-identification, two-stage framework that enables zero-shot task rope manipulation. The framework consists of a system identification module that observes rope movement to predict descriptive physical parameters, which then informs an optimization method for goal-conditioned action prediction for the robot to execute zero-shot in the real. Our method achieves strong performance across multiple dynamic manipulation tasks enabled by the same task-agnostic system identification module which offers seamless switching between different manipulation tasks, allowing a single model to support a diverse array of manipulation policies. We achieve a 3.55 cm average accuracy on 3D target striking in real using rope system parameters in comparison to 15.34 cm accuracy when our task model is not system-parameter-informed. We achieve a Pearson correlation coefficient of 0.95 between Fourier frequencies of the predicted and real ropes on an unseen trajectory. Project website please see this https URL 

---
# Ethics Testing: Proactive Identification of Generative AI System Harms 

**Authors**: Shin Hwei Tan, Haibo Wang, Heng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.22089)  

**Abstract**: Generative Artificial Intelligence (GAI) systems that can automatically generate content in the form of source code or other contents (e.g., images) has seen increasing popularity due to the emergence of tools such as ChatGPT which rely on Large Language Models (LLMs). Misuse of the automatically generated content can incur serious consequences due to potential harms in the generated content. Despite the importance of ensuring the quality of automatically generated content, there is little to no approach that can systematically generate tests for identifying software harms in the content generated by these GAI systems. In this article, we introduce the novel concept of ethics testing which aims to systematically generate tests for identifying software harms. Different from existing testing methodologies (e.g., fairness testing that aims to identifying software discrimination), ethics testing aims to systematically detect software harms that could be induced due to unethical behavior (e.g., harmful behavior or behavior that violates intellectual property rights) in automatically generated content. We introduced the concept of ethics testing, discussed the challenges therewithin, and conducted five case studies to show how ethics testing can be performed for generative AI systems. 

---
# Removing Sandbagging in LLMs by Training with Weak Supervision 

**Authors**: Emil Ryd, Henning Bartsch, Julian Stastny, Joe Benton, Vivek Hebbar  

**Link**: [PDF](https://arxiv.org/pdf/2604.22082)  

**Abstract**: As AI systems begin to automate complex tasks, supervision increasingly relies on weaker models or limited human oversight that cannot fully verify output quality. A model more capable than its supervisors could exploit this gap through sandbagging, producing work that appears acceptable but falls short of its true abilities. Can training elicit a model's best work even without reliable verification? We study this using model organisms trained to sandbag, testing elicitation techniques on problem-solving math, graduate-level science, and competitive coding tasks. We find that training with weak supervision can reliably elicit sandbagging models when supervised fine-tuning (SFT) and reinforcement learning (RL) are combined: SFT on weak demonstrations breaks the sandbagging behavior, enabling RL to then fully elicit performance. Neither method succeeds reliably alone-RL without SFT almost always leads to reward hacking rather than genuine improvement. Critically, this relies on training being indistinguishable from deployment; when models can distinguish between training and deployment, they can perform well during training while continuing to sandbag afterward. Our results provide initial evidence that training is a viable mitigation against sandbagging, while highlighting the importance of making training indistinguishable from deployment. 

---
# Shard the Gradient, Scale the Model: Serverless Federated Aggregation via Gradient Partitioning 

**Authors**: Amine Barrak  

**Link**: [PDF](https://arxiv.org/pdf/2604.22072)  

**Abstract**: Federated learning (FL) aggregation on serverless platforms faces a hard scalability ceiling: existing architectures (lambda-FL, LIFL) partition clients across aggregators, but every aggregator must hold the complete model gradient in memory. When gradients exceed the per-function memory limit (e.g., 10 GB on AWS Lambda), aggregation becomes infeasible regardless of tree depth or branching factor. We propose GradsSharding, which instead partitions the gradient tensor into M shards, each averaged independently by a serverless function that receives contributions from all clients. Because FedAvg averaging is element-wise, this produces bit-identical results to tree-based approaches, so model accuracy is invariant by construction. Per-function memory is bounded at O(|{\theta}|/M), independent of client count, enabling aggregation of arbitrarily large models. We evaluate GradsSharding against lambda-FL and LIFL through HPC experiments and real AWS Lambda deployments across model sizes from 43 MB to 5 GB. Results show a cost crossover at approximately 500 MB gradient size, 2.7x cost reduction at VGG-16 scale, and that GradsSharding is the only architecture that remains deployable beyond the serverless memory ceiling. 

---
# Optimal Question Selection from a Large Question Bank for Clinical Field Recovery in Conversational Psychiatric Intake 

**Authors**: Guan Gui, Peter Zandi, Jacob Taylor, Ananya Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.22067)  

**Abstract**: Psychiatric intake is a sequential, high-stakes information-gathering process in which clinicians must decide what to ask, in what order, and how to interpret incomplete or ambiguous responses under limited time. Despite growing interest in conversational AI for healthcare, there is still limited infrastructure for conversational AI in this application. Accordingly, we formulate this task as a question-selection problem with clinically grounded questions, known target information, and controllable patient difficulty. We also introduce a task-specific question-selection benchmark based on a bank of 655 clinician-authored intake questions and corresponding synthetic patient vignettes with 5 different behavioral conditions. In our evaluation, we compare random questioning, a clinical psychiatric intake form baseline, and an LLM-guided adaptive policy across 300 interview sessions spanning four patients and five behavioral conditions. Across the benchmark, the clinically ordered fixed form substantially outperforms random questioning, and the LLM-guided policy achieves the strongest overall recovery. The advantage of adaptation grows sharply under patient behavior that is less amenable to field recovery, especially under guarded-concise conditions. These findings suggest that performance in conversational clinical systems depends not only on language understanding after information is disclosed, but also on whether the system reaches the right topics within a limited interaction budget. More broadly, the benchmark provides a controlled framework for studying how clinical structure and adaptive follow-up contribute to information recovery in interactive clinical machine learning. 

---
# Reliability Auditing for Downstream LLM tasks in Psychiatry: LLM-Generated Hospitalization Risk Scores 

**Authors**: Shevya Pandya, Shinjini Bose, Ananya Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.22063)  

**Abstract**: Large language models (LLMs) are increasingly utilized in clinical reasoning and risk assessment. However, their interpretive reliability in critical and indeterminate domains such as psychiatry remains unclear. Prior work has identified algorithmic biases and prompt sensitivity in these systems, raising concerns about how contextual information may influence model outputs, but there remains no systematic way to assess these, especially in the psychiatric domain. We propose an approach for reliability auditing downstream LLM tasks by structuring evaluation around the impact of prompt design and the inclusion of medically insignificant inputs on predicted hospitalization risk scores, which is often the first downstream AI clinical-decision-making task. In our audit, a cohort of synthetic patient profiles (n = 50) is generated, each consisting of 15 clinically relevant features and up to 50 clinically insignificant features, across four prompt reframings (neutral, logical, human impact, clinical judgment). We audit four LLMs (Gemini 2.5 Flash, LLaMa 3.3 70b, Claude Sonnet 4.6, GPT-4o mini), and our results show that including medically insignificant variables resulted in a statistically significant increase in the absolute mean predicted hospitalization risk and output variability across all models and prompts, indicating reduced predictive stability as contextual noise increased. Clinically insignificant features had an effect on instability across many model-prompt conditions, and prompt variations independently affected the trajectory of instability in a model-dependent manner. These findings quantify how LLM-based psychiatric risk assessments are sensitive to non-clinical information, highlighting the need for systematic evaluations of attributional stability and uncertainty behavior like this before clinical deployments. 

---
# Lightweight Retrieval-Augmented Generation and Large Language Model-Based Modeling for Scalable Patient-Trial Matching 

**Authors**: Xiaodi Li, Yang Xiao, Munhwan Lee, Konstantinos Leventakos, Young J. Juhn, David Jones, Terence T. Sio, Wei Liu, Maria Vassilaki, Nansu Zong  

**Link**: [PDF](https://arxiv.org/pdf/2604.22061)  

**Abstract**: Patient-trial matching requires reasoning over long, heterogeneous electronic health records (EHRs) and complex eligibility criteria, posing significant challenges for scalability, generalization, and computational efficiency. Existing approaches either rely on full-document processing with large language models (LLMs), which is computationally expensive, or use traditional machine learning methods that struggle to capture unstructured clinical narratives. In this work, we propose a lightweight framework that combines retrieval-augmented generation and large language model-based modeling for scalable patient-trial matching. The framework explicitly separates two key components: retrieval-augmented generation is used to identify clinically relevant segments from long EHRs, reducing input complexity, while large language models are used to encode these selected segments into informative representations. These representations are further refined through dimensionality reduction and modeled using lightweight predictors, enabling efficient and scalable downstream classification. We evaluate the proposed approach on multiple public benchmarks (n2c2, SIGIR, TREC 2021/2022) and a real-world multimodal dataset from Mayo Clinic (MCPMD). Results show that retrieval-based information selection significantly reduces computational burden while preserving clinically meaningful signals. We further demonstrate that frozen LLMs provide strong representations for structured clinical data, whereas fine-tuning is essential for modeling unstructured clinical narratives. Importantly, the proposed lightweight pipeline achieves performance comparable to end-to-end LLM approaches with substantially lower computational cost. 

---
# Call-Chain-Aware LLM-Based Test Generation for Java Projects 

**Authors**: Guancheng Wang, Qinghua Xu, Lionel C. Briand, Zhaoqiang Guo, Kui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22046)  

**Abstract**: Large language models (LLMs) have recently shown strong potential for generating project-level unit tests. However, existing state-of-the-art approaches primarily rely on execution-path information to guide prompt construction, which is often insufficient for complex software systems with rich inter-class dependencies, deep call chains, and intricate object initialization requirements. In this paper, we present CAT, a novel call-chain-aware LLM-based test generation approach that explicitly incorporates call-chain and dependency contexts into prompts through dedicated static analysis. To construct executable, semantically valid test contexts, CAT systematically models caller--callee relationships, object constructors, and third-party dependencies, and supports iterative test fixing when generation failures occur. We evaluate CAT on the widely used Defects4J benchmark and on four real-world GitHub projects released after the LLM's cut-off date. The results show that, across projects in Defects4J, CAT improves line and branch coverage by 18.04% and 21.74%, respectively, over the state-of-the-art approach PANTA, while consistently achieving superior performance on post-cutoff real-world projects. An ablation study further demonstrates the importance of call-chain and dependency contexts in CAT. 

---
# H-Sets: Hessian-Guided Discovery of Set-Level Feature Interactions in Image Classifiers 

**Authors**: Ayushi Mehrotra, Dipkamal Bhusal, Michael Clifford, Nidhi Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2604.22045)  

**Abstract**: Feature attribution methods explain the predictions of deep neural networks by assigning importance scores to individual input features. However, most existing methods focus solely on marginal effects, overlooking feature interactions, where groups of features jointly influence model output. Such interactions are especially important in image classification tasks, where semantic meaning often arises from pixel interdependencies rather than isolated features. Existing interaction-based methods for images are either coarse (e.g., superpixel-only) or, fail to satisfy core interpretability axioms. In this work, we introduce H-Sets, a novel two-stage framework for discovering and attributing higher-order feature interactions in image classifiers. First, we detect locally interacting pairs via input Hessians and recursively merge them into semantically coherent sets; segmentation from Segment Anything (SAM) is used as a spatial grouping prior but can be replaced by other segmentations. Second, we attribute each set with IDG-Vis, a set-level extension of Integrated Directional Gradients that integrates directional gradients along pixel-space paths and aggregates them with Harsanyi dividends. While Hessians introduce additional compute at the detection stage, this targeted cost consistently yields saliency maps that are sparser and more faithful. Evaluations across VGG, ResNet, DenseNet and MobileNet models on ImageNet and CUB datasets show that H-Sets generate more interpretable and faithful saliency maps compared to existing methods. 

---
# EgoMAGIC- An Egocentric Video Field Medicine Dataset for Training Perception Algorithms 

**Authors**: Brian VanVoorst, Nicholas Walczak, Christopher Gilleo, Charles Meissner, Fabio Felix, Iran Roman, Bea Steers, Claudio Silva, Yuhan Shen, Zijia Lu, Shih-Po Lee, Ehsan Elhamifar  

**Link**: [PDF](https://arxiv.org/pdf/2604.22036)  

**Abstract**: This paper introduces EgoMAGIC (Medical Assistance, Guidance, Instruction, and Correction), an egocentric medical activity dataset collected as part of DARPA's Perceptually-enabled Task Guidance (PTG) program. This dataset comprises 3,355 videos of 50 medical tasks, with at least 50 labeled videos per task. The primary objective of the PTG program was to develop virtual assistants integrated into augmented reality headsets to assist users in performing complex tasks.
To encourage exploration and research using this dataset, the medical training data has been released along with an action detection challenge focused on eight medical tasks. The majority of the videos were recorded using a head-mounted stereo camera with integrated audio. From this dataset, 40 YOLO models were trained using 1.95 million labels to detect 124 medical objects, providing a robust starting point for developers working on medical AI applications.
In addition to introducing the dataset, this paper presents baseline results on action detection for the eight selected medical tasks across three models, with the best-performing method achieving average mAP 0.526. Although this paper primarily addresses action detection as the benchmark, the EgoMAGIC dataset is equally suitable for action recognition, object identification and detection, error detection, and other challenging computer vision tasks.
The dataset is accessible via this http URL (DOI: https://doi.org/10.5281/zenodo.19239154). 

---
# Mochi: Aligning Pre-training and Inference for Efficient Graph Foundation Models via Meta-Learning 

**Authors**: João Mattos, Arlei Silva  

**Link**: [PDF](https://arxiv.org/pdf/2604.22031)  

**Abstract**: We propose Mochi, a Graph Foundation Model that addresses task unification and training efficiency by adopting a meta-learning based training framework. Prior models pre-train with reconstruction-based objectives such as link prediction, and assume that the resulting representations can be aligned with downstream tasks through a separate unification step such as class prototypes. We demonstrate through synthetic and real-world experiments that this procedure, while simple and intuitive, has limitations that directly affect downstream task performance. To address these limitations, Mochi pre-trains on few-shot episodes that mirror the downstream evaluation protocol, aligning the training objective with inference rather than relying on a post-hoc unification step. We show that Mochi, along with its more powerful variant Mochi++, achieves competitive or superior performance compared to existing Graph Foundation Models across 25 real-world graph datasets spanning node classification, link prediction, and graph classification, while requiring 8$\sim$27 times less training time than the strongest baseline. 

---
# Shared Lexical Task Representations Explain Behavioral Variability In LLMs 

**Authors**: Zhuonan Yang, Jacob Xiaochen Li, Francisco Piedrahita Velez, Eric Todd, David Bau, Michael L. Littman, Stephen H. Bach, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2604.22027)  

**Abstract**: One of the most common complaints about large language models (LLMs) is their prompt sensitivity -- that is, the fact that their ability to perform a task or provide a correct answer to a question can depend unpredictably on the way the question is posed. We investigate this variation by comparing two very different but commonly-used styles of prompting: instruction-based prompts, which describe the task in natural language, and example-based prompts, which provide in-context few-shot demonstration pairs to illustrate the task. We find that, despite large variation in performance as a function of the prompt, the model engages some common underlying mechanisms across different prompts of a task. Specifically, we identify task-specific attention heads whose outputs literally describe the task -- which we dub lexical task heads -- and show that these heads are shared across prompting styles and trigger subsequent answer production. We further find that behavioral variation between prompts can be explained by the degree to which these heads are activated, and that failures are at least sometimes due to competing task representations that dilute the signal of the target task. Our results together present an increasingly clear picture of how LLMs' internal representations can explain behavior that otherwise seems idiosyncratic to users and developers. 

---
# Foundation models for discovering robust biomarkers of neurological disorders from dynamic functional connectivity 

**Authors**: Deepank Girish, Yi Hao Chan, Sukrit Gupta, Jing Xia, Jagath C. Rajapakse  

**Link**: [PDF](https://arxiv.org/pdf/2604.22018)  

**Abstract**: Several brain foundation models (FM) have recently been proposed to predict brain disorders by modelling dynamic functional connectivity (FC). While they demonstrate remarkable model performance and zero- or few-shot generalization, the salient features identified as potential biomarkers are yet to be thoroughly evaluated. We propose RE-CONFIRM, a framework for evaluating the robustness of potential biomarker candidates elucidated by deep learning (DL) models including FMs. From experiments on five large datasets of Autism Spectrum Disorder (ASD), Attention-deficit Hyperactivity Disorder (ADHD), and Alzheimer's Disease (AD), we found that although commonly used performance metrics provide an intuitive assessment of model predictions, they are insufficient for evaluating the robustness of biomarkers identified by these models. RE-CONFIRM metrics revealed that simply finetuning FMs leads to models that fail to capture regional hubs effectively, even in disorders where hubs are known to be implicated, such as ASD and ADHD. In view of this, we propose Hub-LoRA (Low-Rank Adaptation) as a fine-tuning technique that enables FMs to not only outperform customised DL models but also produce neurobiologically faithful biomarkers supported by meta-analyses. RE-CONFIRM is generalizable and can be easily applied to ascertain the robustness of DL models trained on functional MRI datasets. Code is available at: this https URL. 

---
# Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning 

**Authors**: Grigory Sapunov  

**Link**: [PDF](https://arxiv.org/pdf/2604.21999)  

**Abstract**: We study learned memory tokens as computational scratchpad for a single-block Universal Transformer (UT) with Adaptive Computation Time (ACT) on Sudoku-Extreme, a combinatorial reasoning benchmark. We find that memory tokens are empirically necessary: across all configurations tested -- 3 seeds, multiple token counts, two initialization schemes, ACT and fixed-depth processing -- no configuration without memory tokens achieves non-trivial performance. The optimal count exhibits a sharp lower threshold (T=0 always fails, T=4 is borderline, T=8 reliably succeeds for 81-cell puzzles) followed by a stable plateau (T=8-32, 57.4% +/- 0.7% exact-match) and collapse from attention dilution at T=64.
During experimentation, we identify a router initialization trap that causes >70% of training runs to fail: both default zero-bias initialization (p ~ 0.5) and Graves' recommended positive bias (p ~ 0.73) cause tokens to halt after ~2 steps at initialization, settling into a shallow equilibrium (halt ~ 5-7) that the model cannot escape. Inverting the bias to -3 ("deep start," p ~ 0.05) eliminates this failure mode. We confirm through ablation that the trap is inherent to ACT initialization, not an artifact of our architecture choices.
With reliable training established, we show that (1) ACT provides more consistent results than fixed-depth processing (56.9% +/- 0.7% vs 53.4% +/- 9.3% across 3 seeds); (2) ACT with lambda warmup achieves matching accuracy (57.0% +/- 1.1%) using 34% fewer ponder steps; and (3) attention heads specialize into memory readers, constraint propagators, and integrators across recursive depth. Code is available at this https URL. 

---
# Multi-Task Optimization over Networks of Tasks 

**Authors**: Julian Hatzky, Thomas Bartz-Beielstein, A. E. Eiben, Anil Yaman  

**Link**: [PDF](https://arxiv.org/pdf/2604.21991)  

**Abstract**: Multi-task optimization is a powerful approach for solving a large number of tasks in parallel. However, existing algorithms face distinct limitations: Population-based methods scale poorly and remain underexplored for large task sets. Approaches that do scale beyond a thousand tasks are mostly MAP-Elites variants and rely on a fixed, discretized archive that disregards the topology of the task space. We introduce MONET (Multi-Task Optimization over Networks of Tasks), a multi-task optimization algorithm that models the task space as a graph: tasks are nodes, and edges connect tasks in the task parameter space. This representation enables knowledge transfer between tasks and remains tractable for high-dimensional problems while exploiting the topology of the task space. MONET combines social learning, which generates candidates from neighboring nodes via crossover, with individual learning, which refines a node's own solution independently via mutation. We evaluate MONET on four domains (archery, arm, and cartpole with 5,000 tasks each; hexapod with 2,000 tasks) and show that it matches or exceeds the performance of existing MAP-Elites-based baselines across all four domains. 

---
# Model Predictive Control of Hybrid Dynamical Systems 

**Authors**: Ricardo G. Sanfelice, Berk Altin  

**Link**: [PDF](https://arxiv.org/pdf/2604.21989)  

**Abstract**: The problem of controlling hybrid dynamical systems using model predictive control (MPC) is formulated and sufficient conditions for asymptotic stability of a set are provided. Hybrid dynamical systems are modeled in terms of hybrid equations, involving a differential equation and a difference equation with inputs and constraints. The proposed hybrid MPC algorithm uses a suitable prediction and control horizon construction inspired by hybrid time domains. Structural properties of the hybrid optimization problem, its feasible set, and its value function are provided. Checkable conditions to guarantee asymptotic stability of a set are provided. These conditions are given in terms of properties on the stage cost, terminal cost, and the existence of static state-feedback laws, related through a control Lyapunov function condition. Examples illustrate the results throughout the paper. 

---
# A general optimization solver based on OP-to-MaxSAT reduction 

**Authors**: Yuxin Zhao, Han Huang, Zhifeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2604.21961)  

**Abstract**: Optimization problems are fundamental in diverse fields, such as engineering, economics, and scientific computing. However, current algorithms are mostly designed for specific problem types and exhibit limited generality in solving multiple types of optimization problems. To enhance generality, we propose an automated reduction method named OP-to-MaxSAT reduction and a general optimization solver based on OP-to-MaxSAT reduction (GORED). GORED unifies the solving of multiple types of optimization problems by reducing the problems from optimization problems to MaxSAT instances in polynomial time and solving them using the state-of-the-art MaxSAT solver. The generality and solution quality of GORED are validated through experiments on 136 instances across 11 types of optimization problems. Experimental results demonstrate that GORED not only successfully solves a wide range of optimization problems but also yields solutions comparable in quality to those from existing methods, with no statistically significant differences observed. By introducing automated reduction, this work shifts the paradigm of optimization solvers from designing specialized algorithms for each problem type to employing a single algorithm for diverse problems. As a result, advances in this single algorithm can now drive progress in a wide range of optimization problems across various domains. 

---
# A systematic review of generative AI usage for IT project management 

**Authors**: Ionut Anghel, Tudor Cioara  

**Link**: [PDF](https://arxiv.org/pdf/2604.21958)  

**Abstract**: This paper aims to synthesize current knowledge on generative AI in IT project management using the PRISMA methodology to provide researchers with a comprehensive perspective on techniques, applications, adoption trends, limitations, and integration across project management tools and process groups. The analysis reveals a clear dominance of OpenAI's GPT in the included studies but relying primarily on prompt engineering, suggesting that research in this area remains at an exploratory stage. Finally, it identifies and discusses three promising research directions for AI-enabled project management, including process group-specific AI agents, project role-based AI agents, and hybrid collaborative networks that enable human-guided orchestration. 

---
# MambaCSP: Hybrid-Attention State Space Models for Hardware-Efficient Channel State Prediction 

**Authors**: Aladin Djuhera, Haris Gacanin, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2604.21957)  

**Abstract**: Recent works have demonstrated that attention-based transformer and large language model (LLM) architectures can achieve strong channel state prediction (CSP) performance by capturing long-range temporal dependencies across channel state information (CSI) sequences. However, these models suffer from quadratic scaling in sequence length, leading to substantial computational cost, memory consumption, and inference latency, which limits their applicability in real-time and resource-constrained wireless deployments. In this paper, we investigate whether selective state space models (SSMs) can serve as a hardware-efficient alternative for CSI prediction. We propose MambaCSP, a hybrid-attention SSM architecture that replaces LLM-based prediction backbones with a linear-time Mamba model. To overcome the local-only dependencies of pure SSMs, we introduce lightweight patch-mixer attention layers that periodically inject cross-token attentions, helping with long-context CSI prediction. Extensive MISO-OFDM simulations show that MambaCSP improves prediction accuracy over LLM-based approaches by 9-12%, while delivering up to 3.0x higher throughput, 2.6x lower VRAM usage, and 2.9x faster inference. Our results demonstrate that hybrid state space architectures provide a promising direction for scalable and hardware-efficient AI-native CSI prediction in future wireless networks. 

---
# Focus Session: Hardware and Software Techniques for Accelerating Multimodal Foundation Models 

**Authors**: Muhammad Shafique, Abdul Basit, Muhammad Abdullah Hanif, Alberto Marchisio, Rachmad Vidya Wicaksana Putra, Minghao Shao  

**Link**: [PDF](https://arxiv.org/pdf/2604.21952)  

**Abstract**: This work presents a multi-layered methodology for efficiently accelerating multimodal foundation models (MFMs). It combines hardware and software co-design of transformer blocks with an optimization pipeline that reduces computational and memory requirements. During model development, it employs performance enhancements through fine-tuning for domain-specific adaptation. Our methodology further incorporates hardware and software techniques for optimizing MFMs. Specifically, it employs MFM compression using hierarchy-aware mixed-precision quantization and structural pruning for transformer blocks and MLP channels. It also optimizes operations through speculative decoding, model cascading that routes queries through a small-to-large cascade and uses lightweight self-tests to determine when to escalate to larger models, as well as co-optimization of sequence length, visual resolution & stride, and graph-level operator fusion. To efficiently execute the model, the processing dataflow is optimized based on the underlying hardware architecture together with memory-efficient attention to meet on-chip bandwidth and latency budgets. To support this, a specialized hardware accelerator for the transformer workloads is employed, which can be developed through expert design or an LLM-aided design approach. We demonstrate the effectiveness of the proposed methodology on medical-MFMs and on code generation tasks, and conclude with extensions toward energy-efficient spiking-MFMs. 

---
# Feedback Over Form: Why Execution Feedback Matters More Than Pipeline Topology in 1-3B Code Generation 

**Authors**: Charles Junichi McAndrews  

**Link**: [PDF](https://arxiv.org/pdf/2604.21950)  

**Abstract**: Small language models (1-3B) are practical to run locally, but individually limited on harder code generation tasks. We ask whether composing them into pipelines can recover some of that lost capability. We study code generation pipelines built from 1-3B models with execution feedback, and use a NEAT-inspired evolutionary search to test whether more complex pipeline structure helps beyond a simple refinement loop. We evaluate on HumanEval (164 problems) and sanitized MBPP (427 problems), all with local inference on a single laptop. Self-refinement with execution feedback improves code generation by more than 4 standard deviations on both benchmarks. The gains are narrow in mechanism: refinement fixes many runtime errors (especially NameError and SyntaxError), but rarely fixes logic errors such as AssertionError. Within our tested general-purpose model pool, generator identity mattered less than refiner capability: a 1.5B generator paired with a 3B refiner matched a 3B model doing both roles. Early stopping is essential; without it, every iteration is net-negative. The code-specialized models outperform every general-purpose pipeline configuration, suggesting model specialization matters more than pipeline architecture. Preliminary text-only pipeline experiments without execution feedback did not show gains at this scale. In our constrained search space, evolutionary search mostly rediscovered the same simple generate-execute-refine loop we found manually, with no clearly significant gain from added topology. Single-evaluation fitness inflates results by 5-7 percent, selecting lucky genomes over good ones. On these benchmarks at 1-3B scale, execution feedback mattered more than added pipeline complexity in determining whether composition helped. 

---
# The Biggest Risk of Embodied AI is Governance Lag 

**Authors**: Shaoshan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.21938)  

**Abstract**: Embodied AI is widely discussed as a job-displacement problem. The deeper risk, however, is governance lag: the inability of public institutions to keep pace with how fast the technology spreads through the physical economy. As reusable robotic platforms are combined with increasingly general AI models, embodied AI may scale across manufacturing, logistics, care, and infrastructure faster than governance systems can observe, interpret, and respond. We argue that this lag appears in three connected forms: observational, institutional, and distributive. The central policy challenge, therefore, is not automation alone, but whether governance and compliance systems can adapt before disruption becomes entrenched. 

---
# Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions 

**Authors**: Minda Zhao, Yilun Du, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05414)  

**Abstract**: As large language models (LLMs) transition from chat interfaces to integral components of stochastic pipelines and systems approaching general intelligence, the ability to faithfully sample from specified probability distributions has become a functional requirement rather than a theoretical curiosity. We present the first large-scale, statistically powered audit of native probabilistic sampling in frontier LLMs, benchmarking 11 models across 15 distributions. To disentangle failure modes, we employ a dual-protocol design: Batch Generation, where a model produces $N{=}1000$ samples within one response, and Independent Requests, comprising $N{=}1000$ stateless calls. We observe a sharp protocol asymmetry: batch generation achieves only modest statistical validity, with a 7% median pass rate, while independent requests collapse almost entirely, with 10 of 11 models passing none of the distributions. Beyond this asymmetry, we reveal that sampling fidelity degrades monotonically with distributional complexity and aggravates as the sampling horizon $N$ increases. Finally, we demonstrate how the propagation of these failures into downstream real-world application tasks introduces systematic biases: models fail to enforce uniform answer-position constraints in Multiple Choice Question generation and systematically violate demographic targets in attribute-constrained text-to-image prompt synthesis. These findings indicate that current LLMs lack a functional internal sampler, necessitating external tools for applications requiring statistical guarantees. 

---
