# Actionable World Representation 

**Authors**: Kunqi Xu, Jitao Li, Jianglong Ye, Tianshu Tang, Isabella Liu, Sifei Liu, Xueyan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.18743)  

**Abstract**: Inspired by the emergent behaviors in large language models that generalized human intelligence, the research community is pursuing similar emergent capabilities within world models, with a emphasis on modeling the physical world. Within the scope of physical world model, objects are the fundamental primitives that constitute physical reality. From humans to computers, nearly everything we interact with is an object. These objects are rarely static; they are actionable entities with varying states determined by their intrinsic properties. While current methods approach object action states either via video generation or dynamic scene reconstruction, none explicitly model this basic element in a unified, principled way to build an actionable object representation. We propose WorldString, a neural architecture capable of modeling the state manifold of real-world objects by learning directly from point clouds or RGB-D video streams. Serving as a versatile digital twin, it acts as a foundational building block for physical world models; thus, we name it WorldString. Sweetly, its fully differentiable structure seamlessly enables future integration with policy learning and neural dynamics. 

---
# What Does the AI Doctor Value? Auditing Pluralism in the Clinical Ethics of Language Models 

**Authors**: Payal Chandak, Victoria Alkin, David Wu, Maya Dagan, Taposh Dutta Roy, Maria Clara Saad Menezes, Ayush Noori, Nirali Somia, John S. Brownstein, Ran Balicer, Rebecca W. Brendel, Noa Dagan, Isaac S. Kohane, Gabriel A. Brat  

**Link**: [PDF](https://arxiv.org/pdf/2605.18738)  

**Abstract**: Medicine is inherently pluralistic. Principles such as autonomy, beneficence, nonmaleficence, and justice routinely conflict, and such ethical dilemmas often sharply divide reasonable physicians. Good clinical practice navigates these tensions in concert with each patient's values rather than imposing a single ethical stance. The ethical values that large language models bring to medical advice, however, have not been systematically examined. We present a framework for auditing value pluralism in medical AI, comprising a benchmark of clinician-verified dilemmas and an attribution method that recovers value priorities directly from decisions. The ecosystem of frontier models spans physician-level value heterogeneity, and models discuss competing values in their reasoning (Overton pluralism) before committing to a decision. However, individual model decisions are near-deterministic across repeated sampling and semantic variations, failing to reproduce the distributional pluralism of the physician panel. Across benchmark cases, these consistent decisions reflect committed, systematic value preferences. While most model priorities fall within the natural range of inter-physician variation, some significantly underweight patient autonomy. A single LLM deployed without regard for its value priorities could amplify those priorities at scale to every patient it serves. Without explicit efforts to balance ethical perspectives with one or multiple models, these tools risk replacing clinical pluralism with a deployment monoculture. 

---
# SkillGenBench: Benchmarking Skill Generation Pipelines for LLM Agents 

**Authors**: Yifan Zhou, Zhentao Zhang, Ziming Cheng, Shuo Zhang, Qizhen Lan, Zhangquan Chen, Zhi Yang, QianyuXu, Ronghao Chen, Huacan Wang, Sen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18693)  

**Abstract**: As LLM agents are increasingly built around reusable skills, a central challenge is no longer only whether agents can use provided skills, but whether they can generate correct, reusable, and executable skills from repositories and documents. Existing benchmarks primarily evaluate the efficacy of given skills or the ability of agents to solve downstream tasks from raw context, but they do not isolate skill generation itself as the object of study. We introduce SkillGenBench, a benchmark for evaluating skill generation pipelines under a unified and controlled protocol. In SkillGenBench, a generator receives raw corpora and produces standardized skill artifacts, which are then executed under fixed harnesses and assessed with unified evaluation procedures. The benchmark covers two generation regimes: task-conditioned generation, where a task-specific skill is synthesized after the task is revealed, and task-agnostic generation, where a reusable skill library must be distilled before downstream tasks are known. It also spans two complementary procedural sources: repository-grounded instances, where procedures are distributed across code, configuration, and scripts, and document-grounded instances, where procedures and constraints must be distilled from long-form text. We provide standardized task specifications, pinned environments, and evaluation protocols centered on deterministic execution-based checks, supplemented by auxiliary signals for diagnosis. Experiments across a range of skill-generation methods and backbones show substantial performance variation, highlight the difficulty of reusable skill distillation, and reveal distinct failure modes in skill generation from software repositories versus long-form documents. SkillGenBench establishes a reproducible testbed for studying skill generation as an independent research problem in agent systems. 

---
# Democratizing Large-Scale Re-Optimization with LLM-Guided Model Patches 

**Authors**: Tinghan Ye, Arnaud Deza, Ved Mohan, El Mehdi Er Raqabi, Pascal Van Hentenryck  

**Link**: [PDF](https://arxiv.org/pdf/2605.18692)  

**Abstract**: Optimization models developed by operations research (OR) experts are often deployed as decision-support systems in industrial settings. However, real-world environments are dynamic, with evolving business rules, previously overlooked constraints, and unforeseen perturbations. In such contexts, end users must rapidly re-optimize models to recover feasible and implementable solutions. This paper introduces an agentic re-optimization framework in which a large language model (LLM) acts as an OR expert, dynamically supporting end users through natural-language interaction. The LLM translates user prompts into structured updates of the underlying optimization model, selects suitable re-optimization techniques from an optimization toolbox, and solves the resulting instance to return implementable solutions. The toolbox leverages primal information, including historical solutions, valid inequalities, solver configurations, and metaheuristics, to accelerate re-optimization while preserving solution quality. The proposed framework enables interactive and continuous adaptation of deployed optimization models, reducing dependence on OR experts and improving the sustainability of decision-support systems. Extensive experiments on two complementary large-scale real-world case studies demonstrate the effectiveness and scalability of the proposed framework. The first considers online supply chain re-optimization, where solutions must be generated rapidly while remaining close to the deployed plan, whereas the second focuses on offline university exam scheduling, where solution quality is prioritized over runtime. Results show that the toolbox-driven architecture significantly improves computational efficiency through primal-based and solver-aware re-optimization techniques, while the structured patch-based updates improve interpretability and traceability of model modifications. 

---
# Learning Quantifiable Visual Explanations Without Ground-Truth 

**Authors**: Amritpal Singh, Andrey Barsky, Mohamed Ali Souibgui, Ernest Valveny, Dimosthenis Karatzas  

**Link**: [PDF](https://arxiv.org/pdf/2605.18681)  

**Abstract**: Explainable AI (XAI) techniques are increasingly important for the validation and responsible use of modern deep learning models, but are difficult to evaluate due to the lack of good ground-truth to compare against. We propose a framework that serves as a quantifiable metric for the quality of XAI methods, based on continuous input perturbation. Our metric formally considers the sufficiency and necessity of the attributed information to the model's decision-making, and we illustrate a range of cases where it aligns better with human intuitions of explanation quality than do existing metrics.
To exploit the properties of this metric, we also propose a novel XAI method, considering the case where we fine-tune a model using a differentiable approximation of the metric as a supervision signal. The result is an adapter module that can be trained on top of any black-box model to output causal explanations of the model's decision process, without degrading model performance. We show that the explanations generated by this method outperform those of competing XAI techniques according to a number of quantifiable metrics. 

---
# Efficient Lookahead Encoding and Abstracted Width for Learning General Policies in Classical Planning 

**Authors**: Michael Aichmüller, Simon Ståhlberg, Martin Funkquist, Hector Geffner  

**Link**: [PDF](https://arxiv.org/pdf/2605.18674)  

**Abstract**: Generalized planning aims to learn policies that generalize across collections of instances within a classical planning domain. Recent Graph Neural Network (GNN) approaches have learned nearly perfect policies for several domains. This work improves on the recently published idea of Iterated Width (IW) policies. Therein, the policy broadens its successor scope through an IW-lookahead search that can "jump" over multiple transitions, simplifying the problem structure. Yet, each transition is evaluated individually, leading to unscalable compute costs and expressivity limitations. Furthermore, although IW(1) is attractive because it scales linearly with the number of atoms, it becomes inefficient once thousands of objects are considered, as in the International Planning Competition (IPC) 2023 benchmark. We address both limitations. First, we introduce a vastly more efficient holistic encoding of the entire search tree. It jointly represents IW(1)-reachable states only by their relational differences to the current state, enabling Relational GNNs (R-GNNs) to score all transitions in a single forward pass. Second, we define Abstracted IW(1) to improve scaling through relational abstraction during novelty checks. Rather than testing fully instantiated atoms, it abstracts each atom by replacing all but one argument with its type. The original atom is novel if any of its abstracted forms is novel. This structural compression shifts novelty search scaling from atoms to objects, while preserving meaningful subgoal structure. We evaluate our contributions on the hyperscaling IPC 2023 benchmark and across diverse domains, including domains requiring features beyond the $C_2$ logic fragment. Our policies achieve new state-of-the-art performance, significantly surpassing prior work, including the classical planner LAMA. 

---
# Position: A Three-Layer Probabilistic Assume-Guarantee Architecture Is Structurally Required for Safe LLM Agent Deployment 

**Authors**: S.Bensalem, Y. Dong, M. Franzle, X. Huang, J. Kroger, D. Nickovic, A. Nouri, R. Roy, C. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18672)  

**Abstract**: This position paper argues that enforcing LLM agent safety within a single abstraction layer is not merely suboptimal but categorically insufficient for deployed LLM agents -- a structural consequence of how agent execution works, not a contingent limitation of current systems. The three dimensions that jointly constitute safe operation -- semantic intent and policy compliance, environmental validity, and dynamical feasibility -- each depend on a strictly distinct set of information that becomes available at different stages of execution. No single guardrail can certify all three. We argue that the community must respond with a contract-based architecture in which each safety dimension is enforced by an independently certified layer whose probabilistic guarantee satisfies the next layer's assumption. We sketch such an architecture and derive the compositional system-level safety bounds it admits via the chain rule of probability. Three open problems stand between this and a deployable standard: bound estimation from non-i.i.d.\ traces, graceful degradation of contracts under deployment drift, and extension to multi-agent settings -- the most important unfinished business in LLM agent runtime assurance. 

---
# GIM: Evaluating models via tasks that integrate multiple cognitive domains 

**Authors**: Rohit Patel, Alexandre Rezende, Steven McClain  

**Link**: [PDF](https://arxiv.org/pdf/2605.18663)  

**Abstract**: As LLM benchmarks saturate, the evaluation community has pursued two strategies to increase difficulty: escalating knowledge demands (GPQA, HLE) or removing knowledge entirely in favor of abstract reasoning (ARC-AGI). The first conflates memorization with capability; the second divorces reasoning from the practical contexts in which it matters. We take a different approach. The Grounded Integration Measure (GIM) is a benchmark of 820 original problems (615 public, 205 private) where difficulty comes from integration; individual problems require coordinating multiple cognitive operations (constraint satisfaction, state tracking, epistemic vigilance, audience calibration) over broadly accessible knowledge, so that reasoning stays grounded in realistic tasks without being gated on specialized expertise. Each problem is an original expert-authored composition, majority with rubric-decomposed scoring (median 6 independently judged criteria). A balanced public--private split provides built-in contamination diagnostic. We calibrate a continuous response 2-parameter logistic (2PL) IRT model over >200k prompt-response pairs across 28 models, producing robust ability estimates that correctly order test-configurations even when raw accuracy is distorted by errors or missing data, addressing a common challenge in benchmark reporting. Using this framework, we present a comprehensive leaderboard spanning 22 models and 47 test-configurations (unique model, thinking-level pairs), and conduct what is to our knowledge the most extensive published study of how test-time compute trades off against model capability on a fixed benchmark: 11 models swept across 35 test-configurations. We observe that within-family configuration choices, such as thinking budget and quantization, matter as much as model selection. We release the evaluation framework, calibrated IRT parameters, and all public problems. 

---
# AI for Auto-Research: Roadmap & User Guide 

**Authors**: Lingdong Kong, Xian Sun, Wei Chow, Linfeng Li, Kevin Qinghong Lin, Xuan Billy Zhang, Song Wang, Rong Li, Qing Wu, Wei Gao, Yingshuo Wang, Shaoyuan Xie, Jiachen Liu, Leigang Qu, Shijie Li, Lai Xing Ng, Benoit R. Cottereau, Ziwei Liu, Tat-Seng Chua, Wei Tsang Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2605.18661)  

**Abstract**: AI-assisted research is crossing a threshold: fully automated systems can now generate research papers for as little as $15, while long-horizon agents can execute experiments, draft manuscripts, and simulate critique with minimal human input. Yet this productivity frontier exposes a deeper integrity problem: under scientific pressure, even frontier LLMs still fabricate results, miss hidden errors, and fail to judge novelty reliably. Studying developments through April 2026, we present an end-to-end analysis of AI across the complete research lifecycle, organized into four epistemological phases: Creation (idea generation, literature review, coding & experiments, tables & figures), Writing (paper writing), Validation (peer review, rebuttal & revision), and Dissemination (posters, slides, videos, social media, project pages, and interactive agents). We identify a sharp, stage-dependent boundary between reliable assistance and unreliable autonomy: AI excels at structured, retrieval-grounded, and tool-mediated tasks, but remains fragile for genuinely novel ideas, research-level experiments, and scientific judgment. Generated ideas often degrade after implementation, research code lags far behind pattern-matching benchmarks, and end-to-end autonomous systems have not yet consistently reached major-venue acceptance standards. We further show that greater automation can obscure rather than eliminate failure modes, making human-governed collaboration the most credible deployment paradigm. Finally, we provide a structured taxonomy, benchmark suite, and tool inventory, cross-stage design principles, and a practitioner-oriented playbook, with resources maintained at our project page. 

---
# SCICONVBENCH: Benchmarking LLMs on Multi-Turn Clarification for Task Formulation in Computational Science 

**Authors**: Nithin Somasekharan, Youssef Hassan, Shiyao Lin, Gihan Panapitiya, Patrick Emami, Anurag Acharya, Sameera Horawalavithana, Shaowu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2605.18630)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as scientific AI as- sistants, and a growing body of benchmarks evaluates their capabilities across knowledge retrieval, reasoning, code generation, and tool use. These evaluations, however, typically assume the scientific problem is already well-posed, whereas practical scientific assistance often begins with an ill-posed user request that must be refined through dialogue before any computation, analysis, or experiment can be carried out reliably. We introduce SCICONVBENCH, a benchmark for multi- turn clarification in scientific task formulation across four computational science problem domains: fluid mechanics, solid mechanics, materials science, and par- tial differential equations (PDEs). SCICONVBENCH targets two complementary capabilities: eliciting missing information (disambiguation) and detecting and correcting erroneous requests containing internally contradictory information (in- consistency resolution). Our benchmark pairs a structured task ontology with a rubric-based evaluation framework, enabling systematic measurement of LLM per- formance across three dimensions: clarification behavior, conversational grounding, and final-specification fidelity. Current frontier models perform relatively well on inconsistency resolution, but even the best model resolves only 52.7% of the disambiguation cases in fluid mechanics. We further find that frontier LLMs fre- quently make silent assumptions and perform implicit specification repairs that are not grounded in the conversation with users. SCICONVBENCH establishes a foundation for evaluating the upstream conversational reasoning that a reliable computational science assistant requires. The code and data can be found at this https URL. 

---
# Learning Lifted Action Models from Traces with Minimal Information About Actions and States 

**Authors**: Jonas Gösgens, Niklas Jansen, Hector Geffner  

**Link**: [PDF](https://arxiv.org/pdf/2605.18627)  

**Abstract**: It has been recently shown that lifted STRIPS models can be learned correctly and efficiently from action traces alone; i.e., applicable action sequences from a hidden STRIPS model. The result is remarkable because the states are not assumed to be observable at all, and yet it is not practical enough as STRIPS actions include arguments that are not needed for selecting the actions. This shortcoming has been addressed by assuming that the action traces come instead from a hidden STRIPS+ model where some action arguments are implicit in the hidden action preconditions. A limitation of this approach, however, is that it assumes that the states are fully observable. In this work, we relax these restrictions and consider the problem of learning STRIPS+ action domains from traces in a more general context where the traces carry partial information about both actions and states. In particular, we formulate algorithms and completeness results for three general cases, all of which assume full observability of selected action arguments. In the first case, no observability of the state is assumed; in the second case, full observability of some state predicates is assumed, and in the third case, local observability of some state predicates is assumed instead. Given a STRIPS+ domain, these results characterize the conditions under which an equivalent domain can be learned from traces. Experimental results are reported. 

---
# Latent Action Reparameterization for Efficient Agent Inference 

**Authors**: Wenhao Huang, Qingwen Zeng, Qiyue Chen, Zijie Guo, Yu Sun, Cheng Yang, Siru Ouyang, Jiri Gesi, Fang Wu, Jiayi Zhang, Huaming Chen, Bang Liu, Xiangru Tang, Chenglin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18597)  

**Abstract**: Large language model (LLM) agents often rely on long sequences of low-level textual actions, resulting in large effective decision horizons and high inference cost. While prior work has focused on improving inference efficiency through system-level optimizations or prompt engineering, we argue that a key bottleneck lies in the representation of the action space itself. We propose Latent Action Reparameterization (LAR), a framework that learns a compact latent action space in which each latent action corresponds to a multi-step semantic behavior. By reparameterizing agent actions into latent units, LAR enables decision making over a shorter effective horizon while preserving the expressiveness of the original action space. Unlike hand-crafted macros or hierarchical controllers, latent actions are learned from agent trajectories and integrated directly into the model, allowing both planning and execution to operate over abstract action representations. Across a range of LLM-based agent benchmarks, LAR significantly reduces the effective action horizon and improves inference efficiency under fixed compute budgets. As a consequence, our approach achieves substantial reductions in action tokens and corresponding wall-clock inference time, while maintaining or improving task success rates. These results suggest that action representation learning is a critical and underexplored factor in scaling efficient LLM agent inference, complementary to advances in model architecture and hardware. 

---
# When Outcome Looks Right But Discipline Fails: Trace-Based Evaluation Under Hidden Competitor State 

**Authors**: Peiying Zhu, Sidi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18580)  

**Abstract**: Outcome-only evaluation can certify economically unsafe agents: a policy can hit a business KPI while violating deployable behavioral discipline. In hotel pricing with hidden competitor state, a learner can achieve plausible revenue per available room while failing to preserve the rate discipline of a rule-based revenue-management competitor.
We introduce discipline stability, a trace-based evaluation paradigm: define the benchmark behavior, restrict observations to the deployment regime, induce trace diagnostics from failure, separate mechanisms with ablations, and test transfer and deployment. Across a two-hotel benchmark and a compact hidden-budget bidding task, reward-only PPO variants miss trace alignment; revealing hidden state reduces label uncertainty; deterministic copy collapses uncertainty; and trace-prior or corrected history policies better preserve price or bid distributions. Pure behavior cloning is nearly enough for symmetric imitation, while Trace-Prior RL adds bounded adaptation under capacity asymmetry. The contribution is an evaluation and benchmark paradigm, not a new optimizer or a universal claim about MARL 

---
# Query-Conditioned Knowledge Alignment for Reliable Cross-System Medical Reasoning 

**Authors**: Yan Jiao, Jingran Xu, Pin-Han Ho, Limei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.18570)  

**Abstract**: Cross-domain knowledge alignment is essential for integrating heterogeneous medical systems, yet existing approaches typically treat entity alignment as a static matching problem, ignoring query context and cross-system asymmetry. This limitation is particularly critical in integrative medical settings, where correspondence between concepts is inherently context-dependent, non-bijective, and direction-sensitive.
In this paper, we propose Query-Conditioned Entity Alignment (QCEA), which reformulates entity alignment as a query-conditioned correspondence problem. Instead of learning a fixed mapping between entity representations, QCEA treats the textual description of a source entity as a query and ranks candidate entities in the target graph, enabling context-dependent alignment. The framework integrates semantic encoding, graph-based representation learning, and a direction-aware transformation module to capture asymmetric and many-to-many correspondence across heterogeneous knowledge systems.
We evaluate QCEA on TCM--WM knowledge graphs derived from SymMap, covering both symptom alignment and herb--molecule alignment tasks. Experimental results show consistent improvements over representative baselines, particularly on rank-sensitive metrics such as Hit@K and MRR. Furthermore, downstream retrieval-augmented generation (RAG) experiments demonstrate that improved alignment leads to better evidence retrieval, stronger grounding, and higher answer accuracy. These findings highlight that alignment is not merely a data integration step, but a key factor that shapes knowledge accessibility and reliability in cross-system medical reasoning. 

---
# VISAFF: Speaker-Centered Visual Affective Feature Learning for Emotion Recognition in Conversation 

**Authors**: Linan ZHU, Zihao Zhai, Xiao Han, Yuqian Fu, Xiangfan Chen, Xiangjie Kong, Guojiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.18547)  

**Abstract**: Emotion Recognition in Conversation (ERC) is essential for effective human-machine interaction, aiming to identify speakers' emotional states in multi-turn dialogues. Early text-based methods struggle with complex scenarios like sarcasm because they inherently neglect vital non-verbal information. While recent Vision-Language Models (VLMs) address this by analyzing video directly, they are not inherently tailored for ERC and often focus on emotionally irrelevant background regions or passive listeners rather than the active speaker. Furthermore, fine-tuning these large models incurs prohibitive computational costs. Additionally, isolated visual signals are frequently ambiguous or technically compromised without the context of linguistic content and vocal prosody. To address these challenges, we propose VISAFF, a speaker-centered VISual AFFective feature learning framework for ERC. VISAFF consists of two stages: Speaker-Centered Affective Grounding and Reliability-Guided Affective Complementation. VISAFF utilizes a tuning-free approach to unlock the reasoning capabilities of frozen VLMs, efficiently steering them to focus on the active speaker's emotional visual cues without heavy training overheads. In the second stage, we introduce a reliability-guided affective complementation mechanism that dynamically leverages textual and acoustic modalities to compensate for visual uncertainty. Experiments on two real-world datasets demonstrate that VISAFF achieves highly competitive performance compared to state-of-the-art methods in a tuning-free setting, significantly enhancing computational efficiency by eliminating the need for expensive fine-tuning of large VLMs. The source code is available at this https URL. 

---
# AMR-SD: Asymmetric Meta-Reflective Self-Distillation for Token-Level Credit Assignment 

**Authors**: Zhenlin Wei, Pu Jian, Yingzhuo Deng, Xiaohan Wang, Jiajun Chai, Zhexin Hu, Wei Lin, Shanbin Zhang, Guojun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.18529)  

**Abstract**: The alignment of Large Language Models (LLMs) for complex reasoning heavily relies on Reinforcement Learning with Verifiable Rewards (RLVR). However, standard algorithms like GRPO apply sequence-level rewards uniformly to all tokens, creating a severe credit-assignment bottleneck. While on-policy self-distillation attempts to resolve this by conditioning a self-teacher on privileged contexts, direct exposure to raw oracle solutions often induces over-conditioned teacher distributions, implicit answer leakage, and late-stage training collapse. To overcome these limitations, we propose Asymmetric Meta-Reflective Self-Distillation (AMR-SD). Instead of conditioning directly on raw reference traces, AMR-SD inserts a reflection bottleneck: it compresses diagnostic signals -- from verifier outcomes, peer rollouts, or reference feedback -- into concise, self-generated Socratic hints and critiques. Furthermore, we introduce Causal Information Gain (CIG) with an asymmetric, ReLU-gated threshold to translate these reflections into sparse, highly precise token-level advantage modulations. Combined with temporal annealing, this mechanism preserves the base environmental reward while filtering out distributional noise. Experiments across scientific, mathematical, and tool-use benchmarks demonstrate that AMR-SD significantly outperforms existing baselines, achieving robust long-horizon stability and successfully preventing late-stage collapse. 

---
# A Practical Noise2Noise Denoising Pipeline for High-Throughput Raman Spectroscopy 

**Authors**: David Martin-Calle, Cesar Alvarez Llamas, Vincent Motto- Ros, Christophe Dujardin, Jérémie Margueritat, David Rodney  

**Link**: [PDF](https://arxiv.org/pdf/2605.18511)  

**Abstract**: A lightweight and reproducible denoising pipeline for high-throughput Raman spectroscopy is presented. The approach relies on a one-dimensional convolutional autoencoder trained using a Noise2Noise strategy, requiring neither external spectral libraries nor high signal-to-noise reference spectra for training. From a reduced training subset composed of repeated short-exposure acquisitions, the model learns to reconstruct Raman spectra while efficiently suppressing stochastic noise. The method is evaluated on a heterogeneous mineral sample, using both quantitative spectral fidelity metrics (RMSE, SNR, SSIM) and task-oriented criteria based on unsupervised K-means classification. Results demonstrate that integration times as short as 5 ms per spectrum, which are typically insufficient for reliable interpretation, yield denoised spectra with high fidelity to the reference data while preserving chemically coherent maps. This work provides a practical trade-off between spectral quality and acquisition speed, enabling fast, adaptable Raman workflows compatible with routine laboratory use. It also offers a transferable framework for other one-dimensional spectroscopic modalities. 

---
# OCCAM: Open-set Causal Concept explAnation and Ontology induction for black-box vision Models 

**Authors**: Chiara Maria Russo, Simone Carnemolla, Simone Palazzo, Daniela Giordano, Concetto Spampinato, Matteo Pennisi  

**Link**: [PDF](https://arxiv.org/pdf/2605.18481)  

**Abstract**: Interpreting the decisions of deep image classifiers remains challenging, particularly in black-box settings where model internals are inaccessible. We introduce OCCAM, a framework for open-set causal concept explanation and ontology induction in vision models. OCCAM discovers visual concepts in an open-set manner, localizes them via text-guided segmentation, and performs object-level interventions by removing concepts to measure changes in class confidence, estimating each concept's causal contribution. Beyond local explanations, OCCAM aggregates interventional evidence across a dataset to induce a structured concept ontology that captures how classifiers globally organize visual concepts. Reasoning over this ontology reveals consistent dependencies between concepts, exposes latent causal relations, and uncovers systematic model biases. Experiments on Broden and ImageNet-S across multiple classifiers show that OCCAM improves explanation quality in open-set black-box settings while providing richer global insight than per-image attribution methods. 

---
# When Fireflies Cluster; Enhancing Automatic Clustering via Centroid-Guided Firefly Optimization 

**Authors**: MKA Ariyaratne, Azwirman Gusrialdi, Yury Nikulin, Jaakko Peltonen  

**Link**: [PDF](https://arxiv.org/pdf/2605.18460)  

**Abstract**: This work presents a novel variant of the Firefly Algorithm (FA) for data clustering, addressing limitations of traditional methods like K-Means that struggle with non-uniform cluster shapes, densities, and the need for pre-defining the number of clusters. The proposed algorithm introduces a centroid movement strategy and a multi-objective fitness function that balances compactness, separation, and a novel TSP-based navigation penalty. It automatically estimates the optimal number of clusters and dynamically adjusts cluster boundaries. Application to robotic sensor networks highlights its practical value, with experiments showing improved clustering quality and reduced intra-cluster path distances compared to K-Means. These results confirm the algorithm's robustness in complex spatial clustering tasks, with potential for future extensions to higher-dimensional and adaptive scenarios. 

---
# QSTRBench: a New Benchmark to Evaluate the Ability of Language Models to Reason with Qualitative Spatial and Temporal Calculi 

**Authors**: Anthony G. Cohn, Robert E. Blackwell  

**Link**: [PDF](https://arxiv.org/pdf/2605.18380)  

**Abstract**: We introduce an extensive qualitative spatial and temporal reasoning (QSTR) benchmark for evaluating large language models (LLMs). We pose questions concerning compositional reasoning (using composition tables, CT), converse relations, and conceptual neighbourhoods (CN) for QSTR calculi, Point Algebra (PA), Allen's Interval Algebra, Interval and Duration (INDU), Region Connection Calculus (RCC-5, RCC-8, and RCC-22), the nine intersection model, cardinal direction calculus, and STAR. The RCC-22 CN is published here for the first time. An extended benchmark systematically varies question presentation including prefix/infix, words/symbols/nonce terms and schematic descriptions for selected calculi. We report results for contemporary frontier models. All models tested perform better than guessing but none can consistently answer all questions correctly. Performance varies sharply by calculus, with PA being the most straightforward, and RCC-22 the most difficult. We release the benchmark, and our results under an open licence to facilitate further assessment of qualitative spatio/temporal reasoning in LLMs. 

---
# Causely: A Causal Intelligence Layer for Enterprise AI A Benchmark Study on SRE and Reliability Workflows 

**Authors**: Dhairya Dalal, Endre Sara, Ben Yemini, Christine Miller, Shmuel Kliger  

**Link**: [PDF](https://arxiv.org/pdf/2605.18327)  

**Abstract**: AI agents deployed into SRE workflows currently derive their understanding of environment state from raw observability telemetry at query time, paying a semantic-interpretation tax in tokens, latency, and inferential reliability. We propose Causely, a causal intelligence layer that maintains a structured representation of environment topology, attribute dependencies, and causal relationships that are anchroed to a ontological representation of the managed environment. Causely transforms raw telemetry into a live, queryable model providing the semantic and causal foundation AI agents require to diagnose, evaluate impact, and act safely in production. We evaluate this value proposition through a benchmark study conducted in a controlled setting with injected faults in a 24-microservice OpenTelemetry demo application. Our experiments compare four agent configurations (Claude Code, OpenAI Codex, HolmesGPT with Sonnet and Gemini backends). Experiments are run with and without access to Causely under two scenarios: an active incident and a healthy baseline. On the active-fault scenario, causal grounding reduces mean time-to-diagnosis by 63\%, mean token consumption by 60\%, and mean tool-call count by 78\%, compressing the investigation footprint by 4.8$\times$ and lowering direct API cost per run by 57\%; root-cause-diagnosis accuracy rises from 75\% to 100\%. 

---
# SD-Search: On-Policy Hindsight Self-Distillation for Search-Augmented Reasoning 

**Authors**: Yufei Ma, Zihan Liang, Ben Chen, Zhipeng Qian, Huangyu Dai, Lingtao Mao, Xuxin Zhang, Chenyi Lei, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2605.18299)  

**Abstract**: Search-augmented reasoning agents interleave internal reasoning with calls to an external retriever, and their performance relies on the quality of each issued query. However, under outcome-reward reinforcement learning, every search decision in a rollout shares the same trajectory-level reward, leaving individual queries without step-specific credit. Recent process-supervision approaches address this gap by drawing step-level signals from outside the policy, relying either on a much larger teacher model, or on sub-question annotations produced by a stronger external system. In contrast, we propose SD-Search, which derives step-level supervision from the policy itself through on-policy hindsight self-distillation, requiring neither an external teacher nor additional annotations. In SD-Search, a single model plays two roles that differ only in conditioning: a student that sees only the context available at inference time, and a teacher that additionally conditions on a compact hindsight block summarizing the search queries and final outcomes of a group of rollouts sampled from the same question. Since the teacher knows how each rollout unfolded and which ones succeeded, its query distribution implicitly marks which decisions were worth making, and the student is trained to recover this behavior by minimizing the token-level Jensen--Shannon divergence to the teacher at search-query positions. This layers a dense, step-level signal on top of GRPO's coarse trajectory reward. Crucially, this signal is produced by the policy itself within the standard RL training loop, without external model inference, auxiliary annotation pipeline, or additional training stage. 

---
# DARE-EEG: A Foundation Model for Mining Dual-Aligned Representation of EEG 

**Authors**: Yang Shao, Peiliang Gong, Qun Dai, Daoqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18298)  

**Abstract**: Foundation models pre-trained through masked reconstruction on large-scale EEG data have emerged as a promising paradigm for learning generalizable neural representations across diverse brain-computer interface applications. However, a critical yet overlooked challenge is that EEG encoders must learn representations invariant to incomplete observations-when different masked views of the same signal have minimal overlap, existing methods fail to constrain them to a consistent latent subspace, leading to degraded transferability. To address this, we propose DARE-EEG, a self-supervised foundation model that explicitly enforces the mask-invariance property through dual-aligned representation learning during pre-training. Specifically, we introduce mask alignment that constrains representations from multiple masked views of the same EEG sample via contrastive learning, complementing anchor alignment that aligns masked representations to momentum-updated complete features for semantic stability. Additionally, we propose conv-linear-probing, a parameter-efficient strategy that adapts pre-trained representations to heterogeneous electrode configurations and sampling rates through decoupled spectro-spatial projections. Extensive experiments across diverse EEG benchmarks demonstrate that DARE-EEG consistently achieves state-of-the-art in accuracy performance while maintaining relatively low parameter complexity and superior cross-dataset portability compared to existing methods. Furthermore, DARE-EEG contributes to effectively discovering and utilizing the rich potential representations in EEG. 

---
# Beyond the Cartesian Illusion: Testing Two-Stage Multi-Modal Theory of Mind under Perceptual Bottlenecks 

**Authors**: Yajing Zhou, Xiangyu Kong  

**Link**: [PDF](https://arxiv.org/pdf/2605.18194)  

**Abstract**: While Multi-Modal Large Language Models (MLLMs) demonstrate impressive capabilities in general reasoning, their embodied spatial intelligence remains hampered by a "Cartesian Illusion" - a reliance on text-based probability distributions that lack grounded, 3D topological understanding. This limitation is starkly exposed in multi-agent environments, which demand more than just scene perception; they require second-order Theory of Mind (ToM). Specifically, an Agent A must be able to infer Agent B's belief about the environment, governed strictly by Agent B's physical orientation and sensory limitations. In this paper, we probe the limits of two-stage spatial inference in MLLMs through a novel audio-visual task: requiring Agent A to predict Agent B's estimation of A's relative location. To solve this, we propose an Epistemic Sensory Bottleneck module that abandons rigid, rule-based coordinate transformations. Instead, we introduce an Anchor-Based Embodied Spatial Decomposition Chain-of-Thought (CoT). This guides the MLLM through a "geometric-to-semantic" projection, forcing it to first establish B's local coordinate system and then dynamically weight visual and auditory modalities based on whether A falls within B's visual frustum. Extensive evaluations reveal that while current MLLMs fundamentally struggle with spatial symmetry and out-of-view ambiguities (establishing a rigorous zero-shot baseline of 42% accuracy), our sensory-bounded reasoning chain robustly outperforms pure egocentric and allocentric baselines. By systematically benchmarking these perceptual bottlenecks, our work exposes the current limits of MLLM spatial reasoning and establishes a foundational paradigm for epistemic, modality-aware inference in Embodied AI. 

---
# Pairwise Preference Reward and Group-Based Diversity Enhancement for Superior Open-Ended Generation 

**Authors**: Guining Cao, Jiaxin Peng, Chu Zeng, Yu Zhao, Shuangyong Song, Yongxiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18191)  

**Abstract**: Current reinforcement learning(RL) methods are broadly applicable and powerful in verifiable settings where scalar rewards can be provided. However, in open-ended generation tasks, verifying the correctness of responses remains challenging, and training reward models incurs substantial computational and annotation costs. Moreover, reinforcement learning (RLVR) often leads to diversity collapse and produces stereotypical or rigid outputs, outcomes that are particularly undesirable in open-domain scenarios. We propose Pairwise Preference Reward and Group-based Diversity Enhancement (PPR-GDE), a RL method that is more suitable for open-ended generation. PPR-GDE does not require scalar rewards and incorporates group-level diversity into the reward signal, it preserves the comparative structure of subjective evaluation through a pairwise preference reward, mitigates judge position bias via repeated comparisons with swapped response order, and introduces a group-based diversity reward that explicitly encourages semantic dispersion within a response group, all of these reward signals are integrated into a unified group-relative policy optimization objective. We instantiate PPR-GDE on role-playing task, experiments show that PPR-GDE achieves a better alignment quality as well as expressive diversity than strong RL baselines. Further analysis shows that pairwise preference is critical for preference alignment in subjective perspective, while the diversity metric plays an essential role in achieving superior expressive diversity and broader semantic coverage. 

---
# Scalable Environments Drive Generalizable Agents 

**Authors**: Jiayi Zhang, Fanqi Kong, Guibin Zhang, Maojia Song, Zhaoyang Yu, Jianhao Ruan, Jinyu Xiang, Bang Liu, Chenglin Wu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.18181)  

**Abstract**: Generalizable agents should adapt to diverse tasks and unseen environments beyond their training distribution. This position paper argues that such generalization requires environment scaling: expanding the distribution of executable rule-sets that agents interact with, rather than only increasing trajectories or tasks within fixed benchmarks. Current scaling practices largely focus on collecting more experience or broader task sets under fixed interaction rules, leaving agents brittle when underlying interfaces, dynamics, observations, or feedback signals change. The core challenge is therefore a world-level distribution shift: agents need systematic exposure to environments with meaningfully different executable rule-sets. To clarify this challenge, we propose a unified taxonomy that separates trajectory scaling, task scaling, and environment scaling by their primary deliverables and by what changes in the executable rule-set. Building on this taxonomy, we synthesize construction paradigms for scalable environments, contrasting programmatic generators that prioritize controllability and verifiability with generative world models that offer broader coverage and open-endedness. We further outline how environment scaling can be coupled with stateful learning mechanisms, emphasizing learned update rules for cross-environment adaptation. We conclude by discussing alternative perspectives and argue that scalable environments provide the essential substrate for measurable and controllable progress toward robust general agents. 

---
# Visualizing the Invisible: Generative Visual Grounding Empowers Universal EEG Understanding in MLLMs 

**Authors**: Junyu Pan, Yansen Wang, Enze Zhang, Baoliang Lu, Weilong Zheng, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.18172)  

**Abstract**: Leveraging the universal representations of pre-trained LLMs and MLLMs offers a promising path toward brain foundation models. However, visually-evoked EEG datasets remain scarce, leading existing methods to align neural signals mainly with abstract text, a lossy translation that may discard fine-grained perceptual information encoded in brain activity. We propose Generative Visual Grounding (GVG), a framework that visualizes the invisible by using an EEG-to-image generative model as a visual translator. Instead of forcing EEG into text alone, GVG hallucinates instance-specific proxy images for non-visual EEG, providing structured visual contexts that allow MLLMs to exploit their visual priors for clinical-state interpretation. We validate this idea on two MLLM backbones, GVG-X-Omni and GVG-Janus. Image-only alignment is already competitive: the lightweight GVG-X-Omni matches 1.7B-parameter text-aligned baselines while tuning only 170M parameters on a frozen 7B backbone. We further extend GVG-Janus with trimodal Image+Text alignment, where text supplies categorical semantic anchors and visual proxies enrich neural representations with perceptual details. Experiments show consistent gains in EEG understanding and visual generation, suggesting visual proxy grounding as an effective complement to textual alignment. 

---
# TRACE: Trajectory Correction from Cross-layer Evidence for Hallucination Reduction 

**Authors**: Tej Sanibh Ranade  

**Link**: [PDF](https://arxiv.org/pdf/2605.18163)  

**Abstract**: Hallucination correction is not a one-direction problem. We show that intermediate layers are neither uniformly more truthful than final layers nor uniformly less trustworthy. Yet hallucination reduction is usually instantiated through one fixed intervention form: contrast one layer against another, steer along a truthfulness direction, or defer to external evidence. This framing is structurally incomplete. Cross-layer factual evidence does not evolve uniformly: in some failures truthful support is present internally and later suppressed, whereas in others candidate competition remains genuinely multi-directional across depth, so no single signed scalar family is generally sufficient. We introduce Trajectory Correction from Cross-layer Evidence for Hallucination Reduction (TRACE), a deterministic, training-free algorithm which corrects hallucinations at inference time by deriving both the corrective layer and the appropriate correction operator from each input's cross-layer candidate trajectory inside the LLM's own forward pass. Under one frozen hyperparameter setting, TRACE selects among scalar reversal, earlier-state recovery, and candidate-space correction using only model-internal evidence. Evaluated as a single universal algorithm across 15 models, 8 model families, and 3 factuality benchmarks, TRACE improves every evaluation cell, yielding mean gains of +12.26 MC1 points and +8.65 MC2-style points with no regressions, with gains reaching +47.20 MC1 and +43.38 MC2-style points. The method uses no labels, retrieval, pretraining, finetuning, or per-model calibration. 

---
# Whispers in the Noise: Surrogate-Guided Concept Awakening via a Multi-Agent Framework 

**Authors**: Mengyu Sun, Ziyuan Yang, Zunlong Zhou, Junxu Liu, Haibo Hu, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18150)  

**Abstract**: Diffusion models (DMs) are widely used for text-to-image generation, but their strong generative capabilities also raise concerns about unsafe or undesirable content. Concept erasure aims to mitigate these risks by removing specific concepts from pretrained models. However, recent studies show that such methods often suppress rather than fully eliminate target concepts, leaving models vulnerable to awakening attacks. Existing approaches primarily rely on white-box access through optimization or inversion, while concept awakening under black-box constraints remains underexplored. In this work, we revisit the denoising process from a trajectory perspective and show that concept erasure mainly disrupts early-stage text-semantic alignment but does not fully prevent semantic information from propagating along the denoising dynamics. As generation proceeds, the model increasingly depends on the evolving noisy state rather than textual conditions, which creates an opportunity to bypass erased mappings. Motivated by this observation, we propose ConceptAgent, a training-free, black-box, multi-agent framework that awakens erased concepts by initializing the denoising trajectory from surrogate-guided noisy states. Extensive experiments demonstrate that ConceptAgent enables accurate and controllable awakening of erased concepts under black-box settings without access to model parameters, gradients, or internal representations. These results highlight fundamental limitations of current concept erasure methods and provide new insights into the dynamic nature of semantic control in DMs. 

---
# Evidence-Grounded Frontier Mapping and Agentic Hypothesis Generation in Nanomedicine 

**Authors**: Christiaan G.A. Viviers, Koen de Bruin, Mirre M. Trines, Ayla M. Hokke, Roy van der Meel, Avi Schroeder, Twan Lammers, Willem J.M. Mulder, Fons van der Sommen  

**Link**: [PDF](https://arxiv.org/pdf/2605.18144)  

**Abstract**: Nanomedicine research spans delivery chemistry, immunology, imaging, biomaterials, and disease-specific translational science, yet its conceptual design space remains fragmented across a large and heterogeneous literature. To date, artificial intelligence in nanomedicine has focused primarily on property prediction and formulation optimization, with much less attention to evidence-grounded discovery support at the level of research direction selection. We introduce pArticleMap, a literature-mapping and research-hypothesis-generation system that combines article embeddings, similarity-graph analysis, sparse frontier extraction, structured evidence-pack retrieval, and an audited large-language-model (LLM) workflow for grounded ideation. Rather than forecasting future concept co-occurrence, pArticleMap targets low-density article-level bridge regions and cluster interfaces, then generates and scores citation-grounded hypotheses with large language models in an agentic setup. We evaluate the system with a retrospective realization benchmark (generate later literature under a historical cutoff) and a blinded human reader assessment layer across cue-conditioned nanomedicine tasks. Across 4 selected retrospective bundles, pArticleMap generated ideas and selected task-retained hypotheses (winner ideas) under the benchmark protocol. For task-level retained hypotheses, a pooled gold recovery rate of 10.8% was obtained, with a recall@10 of 15.9% and a future-neighborhood rate of 61.0%, indicating that the system often reached the correct forward-looking neighborhood (paper ideas) even without exact paper-level recovery. Human-agent agreement is modest overall, indicating that internal scoring is useful as a support signal but does not replace expert judgment. These results position pArticleMap as a conservative, evidence-grounded research assistant for nanomedicine. 

---
# Generative AI and the Productivity Divide: Human-AI Complementarities in Education 

**Authors**: Lihi Idan, Bharat Anand  

**Link**: [PDF](https://arxiv.org/pdf/2605.18143)  

**Abstract**: Generative Artificial Intelligence (GenAI) is transforming how firms create, process, and apply knowledge, yet little is known about the heterogeneity of its productivity effects across users. We report results from a randomized controlled experiment in which participants-analogs of early-career knowledge workers-were assigned to self-study a technical domain using either traditional resources or large-language-model (LLM) assistance. On average, GenAI access significantly increased task performance, but the distribution of gains was highly uneven. Improvements were not predicted by GPA or prior knowledge, but by \textit{AI Interaction Competence (AIC)} -- the ability to elicit, filter, and verify model outputs. High-AIC participants realized outsized gains; low-AIC participants saw limited or even negative marginal returns. A scaffolding intervention (conceptual maps) reduced outcome variance, indicating that standardized workflows can mitigate inequality in AI-mediated performance. We interpret these findings through the lens of human-AI complementarities: GenAI raises mean productivity while introducing a new axis of capability inequality. Managerially, firms should pair GenAI access with short AIC micro-training and simple standard operating procedures to capture value consistently and avoid uneven adoption outcomes. 

---
# POST: Prior-Observation Adversarial Learning of Spatio-Temporal Associations for Multivariate Time Series Anomaly Detection 

**Authors**: Suofei Zhang, Yaxuan Zheng, Haifeng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18128)  

**Abstract**: Existing Multivariate Time Series Anomaly Detection (MTSAD) frameworks increasingly rely on integrating Graph Neural Networks (GNNs) with sequence models to capture complex spatio-temporal dependencies. However, less attention is paid to the spatial over-generalization problem, where unconstrained structural modeling indiscriminately reconstructs anomalies, inevitably degrading detection recall. To tackle this problem, we propose a novel framework that unifies spatio-temporal modeling through a joint prior-observation adversarial learning paradigm. In the spatial dimension, the model alternately learns adjacency matrices as structural prior and models the association discrepancy between prior and data-driven observation in a minimax manner during training. Such adversarial optimization not only improves the model sensitivity for time-wise detection, but also enables the model to localize anomalies to specific channels. To systematically evaluate this anomaly localization capability, we further construct a synthetic benchmark equipped with precise channel-wise annotations. Extensive experiments across public datasets and our dedicated benchmark demonstrate that the proposed framework establishes a new state-of-the-art in both time-wise detection and spatial localization tasks. Our code, pre-trained models, and benchmark are publicly available at this https URL. 

---
# TaskGround: Structured Executable Task Inference for Full-Scene Household Reasoning 

**Authors**: ZhiYuan Feng, Yu Deng, Ruichuan An, Zhenhua Liu, Qixiu Li, Keming Wu, Zhiying Du, Weijie Wang, Haoxiao Wang, Shuang Chen, Sicheng Xu, Yaobo Liang, Jiaolong Yang, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.18109)  

**Abstract**: In real home deployments, household agents must often operate from a complete household scene and a situated household request, rather than from a clean task specification. Such requests require agents to identify task-relevant entities, recover intended task conditions, and resolve ordering constraints from the surrounding scene context. We formalize this capability as full-scene household reasoning: given a complete household scene and a situated household request, an agent must infer executable task structure before producing a grounded skill-level action sequence. This setting is challenging because complete household scenes contain substantial task-irrelevant information, making direct complete-scene prompting inefficient and error-prone. In practical deployment, this challenge is further amplified by privacy and local compute constraints, which favor compact open-weight models with limited long-context reasoning ability. We propose TaskGround, a training-free and model-agnostic Ground-Infer-Execute framework that grounds complete scenes into compact task-relevant scene slices, infers executable task structure, and compiles it into grounded skill-level action sequences. To evaluate this setting, we introduce FullHome, a human-validated evaluation suite of 400 household tasks spanning diverse home-scale environments and both goal-oriented and process-constrained requirements. On FullHome, TaskGround improves task success rates by large margins across both proprietary and open-weight models. Notably, it makes Qwen3.5-9B competitive with GPT-5 under direct complete-scene prompting while reducing total input-token cost by up to 18x. Our results identify executable task-structure inference as a central bottleneck in full-scene household reasoning and show that structured grounding can make compact local models substantially more effective for practical household deployment. 

---
# Safety Geometry Collapse in Multimodal LLMs and Adaptive Drift Correction 

**Authors**: Jiahe Guo, Xiangran Guo, Jiaxuan Chen, Weixiang Zhao, Yanyan Zhao, Yutai Hou, Qianchao Wang, Dandan Tu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2605.18104)  

**Abstract**: Multimodal large language models (MLLMs) often fail to transfer safety capabilities learned in the text modality to semantically equivalent non-text inputs, revealing a persistent multimodal safety gap. We study this gap from a representation-geometric perspective by analyzing a text-aligned refusal direction and a modality-induced drift direction. We show that multimodal inputs compress the usable separation along the refusal direction, making it no longer reliable for identifying and refusing harmful inputs. We refer to this failure mode as Safety Geometry Collapse. We quantify it through conditional refusal separability and show that stronger modality-induced drift is consistently associated with weaker refusal separability and higher attack success rates. We then validate the causal role of modality-induced drift through a fixed-strength activation intervention: counteracting the estimated drift restores refusal separability and improves multimodal safety. After drift correction, we further observe self-rectification, where the model recovers its ability to recognize and refuse harmful multimodal inputs during forward dynamics. This effect also provides an internal signal of the model's perceived harmfulness of each input. Motivated by this signal, we propose ReGap, a training-free inference-time method that adaptively corrects modality drift using self-rectification. Experiments across multiple multimodal safety benchmarks and utility benchmarks demonstrate the effectiveness of ReGap, which significantly improves the safety of MLLMs without compromising general capabilities. Our findings highlight representation-level modality alignment as a crucial direction for real-time safety improvement and for building safer, more reliable MLLMs. 

---
# Learning to Solve Compositional Geometry Routing Problems 

**Authors**: Mingfeng Fan, Jianan Zhou, Jiaqi Cheng, Yifeng Zhang, Jie Zhang, Guillaume Adrien Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2605.18094)  

**Abstract**: We study the Compositional Geometry Routing Problem (CGRP), a unified superclass of traditional routing problems that covers point-only, line-only, area-only, and arbitrary hybrid task geometries, providing a broad abstraction for real-world routing scenarios. Beyond standard point-based routing, CGRP with non-point tasks can be inherently asymmetric, tightly coupled travel routes with the intrinsic path, and enlarges the action space with numerous feasible yet often irrelevant options, thereby posing significant challenges for both representation learning and decision-making. To address these challenges, we propose DiCon, a differential attention-assisted solver with contrastive learning, as a plug-and-play framework that tackles the problem from two complementary angles. First, we introduce a differential attention mechanism that actively suppresses the probability mass on less competitive candidate actions. Second, we design a double-level contrastive learning objective to promote robust global instance representations and regularize geometry-aware task representations. Extensive experiments demonstrate that DiCon achieves strong performance, broad versatility, and superior generalization across diverse CGRP instances with different compositions. 

---
# LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning 

**Authors**: Sangjun Bae, Yisak Park, Sanghyeon Lee, Seungyul Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.18077)  

**Abstract**: Communication is a key component in multi-agent reinforcement learning (MARL) for mitigating partial observability, yet prior approaches often rely on inefficient information exchange or fail to transmit sufficient state information. To address this, we propose LLM-driven Multi-Agent Communication (LMAC), which leverages an LLM's reasoning capability to design a communication protocol that enables all agents to reconstruct the underlying state as accurately and uniformly as possible. LMAC iteratively refines the protocol using an explicit state-awareness criterion, improving state recovery while narrowing differences in agents' knowledge. Experiments on diverse MARL benchmarks show that LMAC improves state reconstruction across agents and yields substantial performance gains over prior communication baselines. 

---
# DocOS: Towards Proactive Document-Guided Actions in GUI Agents 

**Authors**: Jingjing Liu, Ziye Huang, Zihao Cheng, Zeming Liu, Jiahong Wu, Yuhang Guo, Kehai Chen, Yunhong Wang, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18048)  

**Abstract**: While Graphical User Interface (GUI) agents have shown promising performance in automated device interaction, they primarily depend on static parametric knowledge from pre-training or instruction tuning. This reliance fundamentally limits their ability to handle long-tailed tasks that require explicit procedural knowledge absent from model parameters, often forcing agents to resort to inefficient and brittle trial-and-error exploration. To mitigate this limitation, we introduce \textbf{Proactive Document-Guided Action} for GUI agents in dynamic, open-web environments, a novel paradigm that mirrors human problem-solving by enabling agents to autonomously search for relevant documentation to resolve long-tailed tasks. To evaluate agents' capability in this paradigm, we propose \textbf{DocOS}, a benchmark designed to assess document-guided problem solving in fully interactive environments. DocOS requires agents to autonomously navigate a web browser, locate relevant online documentation, comprehend procedural instructions, and faithfully ground them into executable GUI actions. Extensive experiments reveal that progress is strictly constrained by dual bottlenecks: agents struggle to reliably locate relevant information during proactive search and frequently fail to faithfully ground retrieved instructions into precise actions, pointing toward document-guided interaction as a crucial pathway for enabling self-evolving GUI agents in dynamic environments. 

---
# New Insight of Variance reduce in Zero-Order Hard-Thresholding: Mitigating Gradient Error and Expansivity Contradictions 

**Authors**: Xinzhe Yuan, William de Vazelhes, Bin Gu, Huan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.18035)  

**Abstract**: Hard-thresholding is an important type of algorithm in machine learning that is used to solve $\ell_0$ constrained optimization problems. However, the true gradient of the objective function can be difficult to access in certain scenarios, which normally can be approximated by zeroth-order (ZO) methods. The SZOHT algorithm is the only algorithm tackling $\ell_0$ sparsity constraints with ZO gradients so far. Unfortunately, SZOHT has a notable limitation on the number of random directions % in ZO gradients due to the inherent conflict between the deviation of ZO gradients and the expansivity of the hard-thresholding operator. This paper approaches this problem by considering the role of variance and provides a new insight into variance reduction: mitigating the unique conflicts between ZO gradients and hard-thresholding. Under this perspective, we propose a generalized variance reduced ZO hard-thresholding algorithm as well as the generalized convergence analysis under standard assumptions. The theoretical results demonstrate the new algorithm eliminates the restrictions on the number of random directions, leading to improved convergence rates and broader applicability compared with SZOHT. Finally, we illustrate the utility of our method on a ridge regression problem as well as black-box adversarial attacks. 

---
# TeleCom-Bench: How Far Are Large Language Models from Industrial Telecommunication Applications? 

**Authors**: Jieting Xiao, Yun Lin, Huizhen Qiu, Rui Ma, Chen Zhong, Dongyang Xu, Xiao Long, Chaoyu Zhang, Qiaobo Hao, Ding Zou, Zhiguo Yang, Yanqin Gao, Fang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2605.18025)  

**Abstract**: While Large Language Models have achieved remarkable integration in various vertical scenarios, their deployment in the telecommunications domain remains exploratory due to the lack of a standardized evaluation framework. Current telecom benchmarks primarily focus on static, foundational knowledge and isolated atomic skills, neglecting the equipment-specific documentation and end-to-end industrial workflows essential for real-world production systems. To bridge this gap, we present TeleCom-Bench, a comprehensive benchmark comprising 12 evaluation sets with 22,678 curated samples, which evaluates LLMs across a synergistic hierarchy: (1) Multi-dimensional Knowledge Comprehension, which integrates telecommunication fundamentals, 3GPP protocols, and 5G network architecture with proprietary product knowledge across wired, core, and wireless networks via knowledge graph-driven synthesis; and (2)End-to-End Knowledge Application, which formalizes six core tasks on authentic trajectories from live network agent workflows, including intent recognition, entity extraction, event verification, tool invocation, root cause analysis, and solution generation-across network optimization and fault maintenance scenarios. Evaluations of eight state-of-the-art LLMs reveal a universal Execution Wall: while models achieve 90% accuracy in linguistic interface tasks such as intent recognition and entity extraction, performance collapses to approximately 30% in procedural execution tasks like solution generation. This capability gap demonstrates that current LLMs function competently as diagnosticians but fail as field engineers. TeleCom-Bench provides standardized diagnostics to precisely pinpoint this deficit, offering actionable guidance for domain-specific alignment toward production-ready telecom agents. The dataset and evaluation code have been released at this https URL. 

---
# Shared Backbone PPO for Multi-UAV Communication Coverage with Connection Preservation 

**Authors**: Z. Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17999)  

**Abstract**: This paper proposes a Shared Backbone Proximal Policy Optimization (Shared Backbone PPO) algorithm. By sharing the base module between the Actor and Critic networks, the algorithm achieves efficient training and improved performance. The algorithm is implemented in a connectivity-preserving multi-UAV swarm communication coverage task and compared with the standard PPO algorithm. Experimental results demonstrate that the proposed method achieves superior performance. Furthermore, a graph information aggregation module is incorporated into the model architecture to accommodate the communication conditions among agents. With the integration of this module, the algorithm remains effective, and the trained agent swarm exhibits a higher level of cooperation. 

---
# Unleashing LLMs in Bayesian Optimization: Preference-Guided Framework for Scientific Discovery 

**Authors**: Xinzhe Yuan, Zhuo Chen, Jianshu Zhang, Huan Xiong, Nanyang Ye, Yuqiang Li, Qinying Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17976)  

**Abstract**: Scientific discovery is increasingly constrained by costly experiments and limited resources, underscoring the need for efficient optimization in AI for science. Bayesian Optimization (BO), though widely adopted for balancing exploration and exploitation, often exhibits slow cold-start performance and poor scalability in high-dimensional settings, limiting its applicability in real-world scientific problems. To overcome these challenges, we propose LLM-Guided Bayesian Optimization (LGBO), the first LLM preference-guided BO framework that continuously integrates the semantic reasoning of large language models (LLMs) into the optimization loop. Unlike prior works that use LLMs only for warm-start initialization or candidate generation, LGBO introduces a region-lifted preference mechanism that embeds LLM-driven preferences into every iteration, shifting the surrogate mean in a stable and controllable way. Theoretically, we prove that LGBO does not perform significantly worse than standard BO in the worst case, while achieving significantly faster convergence when preferences align with the objective. Empirically, LGBO consistently outperforms existing methods across diverse dry benchmarks in physics, chemistry, biology, and materials science. Most notably, in a new wet-lab optimization of Fe-Cr battery electrolytes, LGBO attains \textbf{90\% of the best observed value within 6 iterations}, whereas standard BO and existing LLM-augmented baselines require more than 10. Together, these results suggest that LGBO offers a promising direction for integrating LLMs into scientific optimization workflows. 

---
# Reconciling Contradictory Views on the Effectiveness of SFT in LLMs: An Interaction Perspective 

**Authors**: Junpeng Zhang, Lei Cheng, Guoxi Zhang, Hua Cai, Qing Xu, Quanshi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17967)  

**Abstract**: This paper explores a scientific question in supervised fine-tuning (SFT): why SFT is broadly effective for small-scale deep neural networks, yet can produce inconsistent or even detrimental effects when applied to large language models (LLMs). Recent advances in interaction-based explanations suggest that interactions between words/tokens provide a faithful metric for quantifying the inference patterns encoded by LLMs. We find that the evolution of interactions during SFT can effectively explain the inconsistent effectiveness of SFT for LLMs. Specifically, we find that (1) SFT primarily removes noise-like interactions, while rarely acquiring reliable new interactions. (2) This denoising stage is extremely brief, after which continued fine-tuning tends to introduce overfitted interactions. We validate these findings across multiple LLMs and datasets. Our findings provide new insights into early stopping and offer practical guidance for LLM training. 

---
# SVFSearch: A Multimodal Knowledge-Intensive Benchmark for Short-Video Frame Search in the Gaming Vertical Domain 

**Authors**: Lingtao Mao, Huangyu Dai, Xinyu Sun, Zihan Liang, Ben Chen, Chenyi Lei, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2605.17946)  

**Abstract**: Multimodal large language models are increasingly used as agent backbones that understand multimodal inputs, plan retrieval actions, invoke external tools, and reason over retrieved information. Yet existing benchmarks rarely evaluate this ability in short-video applications, where a paused frame is often visually ambiguous and answering requires vertical, long-tail, and fast-evolving domain knowledge. We introduce SVFSearch, the first open benchmark for short-video frame search in the Chinese gaming domain. SVFSearch contains 5,000 four-choice test examples and 4,198 auxiliary training examples, each centered on a paused game scene from a real short-video clip. To support fair and reproducible evaluation, SVFSearch provides a frozen offline retrieval environment with a game-domain text corpus, a topic-linked image gallery, and text, image, and multimodal retrieval interfaces, avoiding reliance on uncontrolled web search APIs. We evaluate representative paradigms ranging from direct QA and RAG workflow to Plan-Act-Replan agents and learned search models. Results reveal a large gap between model-only answering, practical agentic search, and oracle knowledge: the best open-source direct-QA model reaches 66.4%, the best practical agent achieves 79.1%, and oracle knowledge reaches 95.4%. Further analysis exposes bottlenecks in visual grounding, retrieval quality, evidence-grounded reasoning, and tool-use behavior, including over-search, answer-only shortcuts, and retrieval-induced misleading. 

---
# Ethical Hyper-Velocity (EHV): A Provably Deterministic Governance-Aware JIT Compiler Architecture for Agentic Systems 

**Authors**: Riddhi Mohan Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2605.17909)  

**Abstract**: As autonomous agentic systems scale across regulated critical infrastructures, the lack of mechanistic, hardware-rooted enforcement for high-frequency policy updates presents a fundamental safety gap. We introduce Ethical Hyper-Velocity (EHV), a novel architectural framework for the formal verification of AI governance policies at runtime. Unlike retrospective auditing frameworks (ISO/IEC 42001, NIST AI RMF) which introduce 14-30 day latencies, EHV relocates the Policy Enforcement Point (PEP) into the inference pipeline via a Governance-Aware Just-In-Time (JIT) Compiler. By integrating Conflict-free Replicated Data Types (CRDTs) for policy synchronization and Epoch-based Attestation Caching within Trusted Execution Environments (TEEs), EHV achieves Sub-millisecond Formal Determinism (SMFD). We demonstrate via TLA+ formal verification that non-compliant agentic actions are computationally unreachable within the system's bounded operating state space. We prove that O(1) runtime enforcement can eliminate the traditional trade-off between deployment velocity and governance integrity, reducing Governance Latency from O(days) to O(1). 

---
# Agentic Chunking and Bayesian De-chunking of AI Generated Fuzzy Cognitive Maps: A Model of the Thucydides Trap 

**Authors**: Akash Kumar Panda, Olaoluwa Adigun, Bart Kosko  

**Link**: [PDF](https://arxiv.org/pdf/2605.17903)  

**Abstract**: We automatically generate feedback causal fuzzy cognitive maps (FCMs) from text by teaching large-language-model agents to break the text into overlapping chunks of text. Convex mixing of these chunk FCMs gives a representative cyclic FCM knowledge graph. The text chunks can have different levels of overlap. The chunk FCMs still mix to form a new FCM causal knowledge graph. The mixing technique scales because it uses light computation with sparse causal chunk matrices. The mixing structure allows an operator-level type of Bayesian inference that produces "de-chunked" or posterior-like FCMs from the mixed FCM. These de-chunked FCMs are useful in their own right and allow further iterations of Bayesian updating. We demonstrate these mixing techniques on the essay text of Allison's "Thucydides Trap" model of conflict between a dominant power such as the United States and a rising power such as China. The FCM dynamical systems predict outcomes as they equilibrate to fixed-point or limit-cycle attractors. Seven out of 8 FCM knowledge graphs predicted a type of war when we stimulated them by turning on and keeping on the concept node that stands for the rising power's ambition and entitlement. Gemini 3.1 LLMs served as the chunking AI agents. 

---
# LAST-RAG: Literature-Anchored Stochastic Trajectory Retrieval-Augmented Generation for Knowledge-Conditioned Degradation Model Selection 

**Authors**: Hanbyeol Park, Hyerim Bae  

**Link**: [PDF](https://arxiv.org/pdf/2605.17902)  

**Abstract**: Stochastic-process-based degradation modeling is a core approach for estimating the distribution of remaining useful life (RUL); however, the selection of an appropriate stochastic process has not been sufficiently addressed. Existing model selection methods mainly rely on the statistical fit of the observed health indicator (HI) trajectory, but this approach may select a model that is inconsistent with the underlying degradation mechanism when the observation window is short or the signal is highly noisy. To address this issue, this paper proposes Literature-Anchored Stochastic Trajectory Retrieval-Augmented Generation (LAST-RAG). The proposed method uses both the observed HI trajectory and domain-specific context, and hierarchically conditions the candidate degradation model space based on theoretical and mechanical evidence retrieved from a local evidence bank. In addition, Rule-based Confidence Reasoning with Uncertain State (RCRUS) is introduced to prevent candidate models from being prematurely eliminated when hierarchical decisions are uncertain. Simulation-based experiments demonstrate that the proposed method outperforms statistical, prognostic, and uncertainty-aware baselines in both Wiener/gamma family classification and detailed degradation model classification. Ultimately, this study reframes degradation model selection from a purely statistical goodness-of-fit problem into a knowledge-conditioned decision-making problem that integrates observed data with domain knowledge. 

---
# DuIVRS-2: An LLM-based Interactive Voice Response System for Large-scale POI Attribute Acquisition 

**Authors**: Le Zhang, Shengming Zhang, Rui Zha, Yunpeng Wu, Jingbo Zhou, Jizhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17900)  

**Abstract**: Accurate Point of Interest (POI) attribute acquisition is essential for location-based services, yet traditional modular Interactive Voice Response (IVR) systems suffer from error accumulation and high maintenance overhead. We present DuIVRS-2, a large language model (LLM)-based end-to-end framework designed for large-scale POI attribute acquisition at Baidu Maps. To address the long-tail distribution of real-world interactions, our methodology first employs a finite state machine (FSM)-guided data augmentation strategy to synthesize a balanced and diverse training dataset. We then streamline dialogue management via a selective generation scheme combined with a Chain-of-Thought (CoT) mechanism, which ensures output stability and effectively eliminates hallucinations in industrial settings. To facilitate continuous policy refinement with minimal manual effort, we design a cooperative iterative learning framework that leverages a dual-evaluator voting system. Deployed in production for two months, DuIVRS-2 processed 0.4 million calls daily and achieved a 83.9\% Task Success Rate (TSR), outperforming its predecessor by 4 percentage points while maintaining a low reaction time of 130ms. This work provides a production-proven reference for developing robust, cost-effective LLM agents for large-scale industrial dialogue applications. 

---
# Evaluating Cognitive Age Alignment in Interactive AI Agents 

**Authors**: Yifan Shen, Jiawen Zhang, Jian Xu, Junho Kim, Ismini Lourentzou, Xu Cao, Meihuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17894)  

**Abstract**: While agentic AI and its core multimodal large language models (MLLMs) have demonstrated remarkable promise in language and visual reasoning across domains ranging from daily life to advanced scientific research, a profound gap remains between artificial and human intelligence. Despite the integration of powerful tools and advanced MLLMs, state-of-the-art AI agents frequently fail at foundational, seemingly simple tasks that a child can resolve with ease. Inspired by the Wechsler Intelligence Scale for Children (WISC), we introduce ChildAgentEval, the first psychometrically grounded interactive benchmark for evaluating cognitive age alignment in MLLM-based agents. ChildAgentEval systematically compares the reasoning performance of various MLLM-based interactive agents against age-specific human developmental stages, exposing where current agentic AI systems can and cannot simulate age-specific cognitive behavior. 

---
# PAIR: Prefix-Aware Internal Reward Model for Multi-Turn Agent Optimization 

**Authors**: Wonjoong Kim, Yeonjun In, Sangwu Park, Dongha Lee, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.17877)  

**Abstract**: A significant hurdle for current LLMs is the execution of complex, multi-stage tasks. Group Relative Policy Optimization (GRPO) has been emerging as a leading choice, but its reliance on sparse outcome rewards severely limits credit assignment across intermediate steps. Existing remedies such as running full rollouts to assign step-level advantages, calling external LLM judges at each step, or computing intrinsic rewards that require ground-truth answers at every evaluation introduce significant costs or practical constraints. We hypothesize that internal correctness probing over LLM hidden states can be repurposed as a step-level reward signal, potentially addressing all of these limitations at once. However, existing probing research assumes clean inputs, and we first show that this assumption breaks down in multi-step settings: hidden-state probes degrade severely under prefix contamination tracking coherence with the (possibly corrupted) prefix rather than grounded correctness, while attention-based features remain robust to contamination but underperform on clean prefixes. Building on this complementary relationship, we propose the Prefix-Aware Internal Reward (PAIR), a two-stage model with a frozen hidden-state probe estimating belief-consistency and a lightweight attention-based head correcting it toward grounded correctness. Experimental results show that PAIR achieves the highest AUROC on contaminated trajectories while operating at negligible inference cost, enabling dense step-level reward signals for GRPO training without external model calls, ground-truth dependencies, or full-trajectory rollouts. 

---
# KISS - Knowledge Infrastructure for Scientific Simulation: A Scaffolding for Agentic Earth Science 

**Authors**: Ziwei Li, Liujun Zhu, Yuchen Liu, Yichen Zhao, Birk Li, Ruiqi Wu, Junliang Jin, Jianyun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17856)  

**Abstract**: Process-based simulation models encode decades of scientific understanding across the Earth sciences, yet the communities most exposed to climate risk and resource scarcity are the least able to use them. Here, we introduce knowledge infrastructure (KI), an agent-actionable scaffold that externalizes expertise into validated modelling operators, staged domain protocols, and diagnostic recovery mechanisms. Across a 3,000-trial coupled-hydrology benchmark, agents equipped with KI produced physically plausible, verifiable end-to-end simulations in up to 84% of trials, while agents without KI plateaued below 40%. KI generalizes across disciplines. We packaged its construction into a Knowledge Dissection Toolkit (KDT) that autonomously produced KI enabling end-to-end agent execution of 117 additional process-based models across 14 Earth-science domains. Across all 119 KIs, modelling decisions and failure remedies converged despite different underlying physics, showing that operational expertise is structured and extractable rather than ad hoc. Demonstrations show KI-equipped agents lowering both the access barrier between non-specialist users and process-based simulation, and the integration barrier between modelling communities. Through this scaffold, process-based science can then evolve as a living scientific commons, answerable to whoever needs to know and extendable by whoever can contribute. 

---
# Remembering More, Risking More: Longitudinal Safety Risks in Memory-Equipped LLM Agents 

**Authors**: Ahmad Al-Tawaha, Shangding Gu, Peizhi Niu, Ruoxi Jia, Ming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.17830)  

**Abstract**: Safety evaluations of memory-equipped LLM agents typically measure within-task safety: whether an agent completes a single scenario safely, often under adversarial conditions such as prompt injection or memory poisoning. In deployment, however, a single agent serves many independent tasks over a long horizon, and memory accumulated during earlier tasks can affect behavior on later, unrelated ones. Studying this regime requires evaluation along the temporal dimension across tasks: not whether an agent is safe at any single memory state, but how its safety profile changes as memory accumulates across many independent interactions. We call this failure mode temporal memory contamination. To isolate memory exposure from stream non-stationarity, we introduce a trigger-probe protocol that evaluates a fixed probe set against read-only memory snapshots at varying prefix lengths, together with a NullMemory counterfactual baseline for identifying memory-induced violations. We apply this protocol across three deployment scenarios spanning records, memos, forms, and email correspondence and eight memory architectures, and additionally on Claw-like AI agents, such as OpenClaw, using the platform's native memory mechanism. Memory-enabled agents consistently exceed the NullMemory baseline, and memory-induced violation rates show a robust upward trend with exposure length on both agent classes. Order-randomization experiments indicate that the effect is driven primarily by accumulated content rather than encounter order. Finally, a structural consequence of the event decomposition is that memory-induced risk is detectable from retrieval state before generation, which we confirm with a high-recall diagnostic monitor. Our results argue for treating memory safety as a longitudinal property that requires temporal evaluation, not a single-state property that can be captured by a snapshot. 

---
# Interactive Evaluation Requires a Design Science 

**Authors**: Keyang Xuan, Peiyang Song, Pan Lu, Pengrui Han, Wenkai Li, Zhenyu Zhang, Zexue He, Wenyue Hua, Manling Li, Jiaxuan You, Adrian Weller, Yizhong Wang, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2605.17829)  

**Abstract**: AI evaluation is undergoing a structural change. Large language models (LLMs) are increasingly deployed as systems that act over time through tools, environments, users, and other agents, while many evaluation practices still inherit assumptions from response-centered benchmarks (e.g., fixed inputs, isolated outputs, and outcome judgments that can be made from a single response). The field has begun to build interactive benchmarks, but the resulting landscape is fragmented: benchmarks differ in what interaction artifacts they admit, how trajectories are scored, and what claims their results support. This position paper argues that interactive evaluation should be treated as a principled evaluation paradigm, not merely a new family of agent benchmarks. Simply adopting previous evaluation paradigms does not suffice. We define evaluation as an autonomous mapping from evidence to judgments, and show that interactive evaluation changes both sides of this mapping: the evidence becomes interaction-generated trajectories, while the evaluation procedure must assess process, recoverability, coordination, robustness, and system-level performance. Building on this definition, we propose a two-axis taxonomy, derive design principles and reporting standards, examine representative scenarios, and analyze how longstanding evaluation challenges reappear at the trajectory level. 

---
# Going Headless? On the Boundaries of Vertical AI Firms 

**Authors**: Muhammad Zia Hydari, Farooq Muzaffar  

**Link**: [PDF](https://arxiv.org/pdf/2605.17812)  

**Abstract**: Vertical AI firms in accounting, law, healthcare, procurement, and similar domains historically bundled workflow, domain logic, and accountability into a single application. General-purpose AI agents are now unbundling that package, prompting founders and investors to advocate "going headless": cede the workflow and interface to agents and expose domain expertise as callable services. This article argues that going headless is correct for some firms and destructive for others, and that the latter often cede their value capture inadvertently through architectural choices that look like interface decisions. This is a boundary question, and the answer turns on distinguishing the interface boundary, which can often move, from the accountability boundary, which often must not. Drawing on Coase's theory of the firm, Eisenmann, Parker, and Van Alstyne's platform envelopment framework, and Teece's analysis of complementary assets and appropriability, the article shows that orchestrators operating through open protocols acquire envelopment power even as technical interoperability improves, and that durable value capture concentrates in cospecialized accountability assets: professional signoff, regulated workflows, evidence trails, and trusted systems of record. The article proposes a three-position taxonomy (component, integrated software platform, dual-track) determined not by sector but by task-accountability regime, and formalizes the construct of rule debt: the future governance, maintenance, and accountability burden that accrues to customer organizations when business rules and professional standards migrate from governed systems into prompts and agent instructions. Four principles follow: decompose by accountability not interface, invert the edges while retaining the core, position rule debt as the customer cost the integrated platform prevents, and avoid single-orchestrator dependence. 

---
# Accelerating AI-Powered Research: The PuppyChatter Framework for Usable and Flexible Tooling 

**Authors**: Chun-Hsiung Tseng, Hao-Chiang Koong Lin, Andrew Chih-Wei Huang, Yung-Hui Chen, Jia-Rou Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.17809)  

**Abstract**: This research addresses the challenges inherent in developing Artificial Intelligence (AI) applications, particularly those leveraging Large Language Models (LLMs). While AI vendors provide Application Programming Interfaces (APIs) and Software Development Kits (SDKs) to facilitate developer interaction, the former often requires intricate manual request construction, and the latter can lead to significant vendor lock-in. Furthermore, existing model abstraction frameworks, though mitigating vendor dependency, introduce an additional layer of complexity and potential security concerns. To reconcile these conflicting factors, the study introduces PuppyChatter, a novel software framework designed to preserve the intuitive simplicity of vendor-specific SDKs while simultaneously adhering to the vendor-neutrality principles characteristic of model abstraction, thereby offering a more streamlined and flexible development paradigm. 

---
# STRIDE: A Self-Reflective Agent Framework for Reliable Automatic Equation Discovery 

**Authors**: Jiarui Su, Songjun Tu, Bei Sun, Xiaojun Liang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17790)  

**Abstract**: LLM-based equation discovery offers a promising route to recovering symbolic laws from data, but many systems still rely on generation-centered loops that propose candidates, fit parameters, score results, and reuse selected examples. Such loops can misjudge useful skeletons under unreliable fitting, discard near-correct equations that require repair, and accumulate redundant memories that provide limited guidance. We propose STRIDE, a self-reflective agent framework that improves reliability by coordinating data-aware generation, mixed-fitting evaluation, critic--executor repair, and diversity-preserving semantic memory. By turning fitted scores and candidate behavior into shared feedback, STRIDE enables equations to be proposed, assessed, refined, and reused within a closed-loop discovery process. Experiments on representative symbolic-regression benchmarks and LSR-Synth suites show that STRIDE improves accuracy, OOD robustness, and structural recovery across multiple LLM backbones, with ablations and analyses confirming the contribution of its core components. 

---
# Entropy-Gradient Inversion: Moving Toward Internal Mechanism of Large Reasoning Models 

**Authors**: Junyao Yang, Chen Qian, Kun Wang, Linfeng Zhang, Quanshi Zhang, Yong Liu, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17770)  

**Abstract**: The advancement of Large Reasoning Models (LRMs) has catalyzed a paradigm shift from reactive ``fast thinking'' text generation to systematic, step-by-step ``slow thinking'' reasoning, unlocking state-of-the-art performance in complex mathematical and logical tasks. However, the field faces \textit{the fundamental gap between token-level behavioral analysis and internal reasoning mechanisms, and the instability of reinforcement learning (RL) for reasoning optimization relying on costly external verifiers}. We identify and formally define \textbf{Entropy-Gradient Inversion}, a robust negative correlation between token entropy and logit gradients that acts as a definitive geometric fingerprint for LRM reasoning capability. Building on this, we propose \textbf{Correlation-Regularized Group Policy Optimization (CorR-PO)}, which embeds this inversion signature into RL reward regularization. Extensive experiments on various reasoning benchmarks across multiple model scales show CorR-PO consistently outperforms state-of-the-art baselines, confirming that stronger inversion directly correlates with superior reasoning performance. 

---
# Surface-Form Neural Sparse Retrieval: Robust Fuzzy Matching for Industrial Music Search 

**Authors**: Paul Greyson, Zhichao Geng, Wei Zhang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17762)  

**Abstract**: Music search at the scale of Amazon Music presents a unique challenge: queries frequently deviate from indexed metadata due to misspellings, transpositions, and phonetic variations, yet the retrieval system must operate under strict millisecond-level latency constraints. Our existing learning-to-retrieve system, the High Confidence Index (HCI), learns query-entity associations from customer behavior, relying on continual ``exploration'' to choose candidates. Traditional n-gram matching enables this exploration but suffers from poor semantic robustness and high noise, limiting the system's ability to learn from long-tail queries. In this work, we present a \textbf{robust neural sparse retrieval system} designed to maximize exploration efficiency. We adapt a state-of-the-art \textbf{inference-free} sparse retrieval architecture to the music domain, combining it with an effective \textbf{domain-specific granular subword tokenization strategy}. Our approach utilizes short-length token constraints (max 3 chars) to enforce the learning of surface-form robustness over lexical memorization. By pre-computing the neural embeddings and term expansions during the offline indexing phase, online processing is reduced to minimal tokenization and IDF weighting, achieving effectively zero latency overhead for query encoding. Evaluations on a 6M-document production corpus show an aggregate \textbf{91.4\%} recall@10 (vs. \textbf{57.7\%} for trigrams) at comparable throughput. Simulation of the HCI feedback loop demonstrates improved exploration efficiency, with \textbf{+0.8\%} higher stabilized recall than production trigrams. Ablation studies indicate that our sparse training methodology drives the performance gains, while domain-specific pretraining provides a cost-effective alternative to large-scale general-purpose pretraining. 

---
# Agents for Experiments, Experiments for Agents: A Design Grammar for AI-Enabled Experimental Science 

**Authors**: Yingjie Zhang, Chun Feng, Weizhang Zhu, Tianshu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.17746)  

**Abstract**: AI systems are becoming active participants in organizational and knowledge work. They increasingly interact with humans, coordinate workflows, and operate in multi-agent arrangements. Understanding their effects therefore requires more than measuring output accuracy; it requires evidence about mechanisms, delegation, feedback, and control. Experiments remain central to this task, but they also face a recursive challenge: we need experiments for agents to study these arrangements, and we may need agents for experiments to help search the expanding space of possible designs. Yet experimental conditions for human-AI and agentic workflows are still largely specified in prose, making them difficult to compare, reuse, or audit. We frame this as a problem of workflow representation, traceability, and governance in AI-enabled knowledge production. We introduce SEED (Structural Encoding for Experimental Discovery), a framework that represents experimental conditions as typed actor-flow graphs. SEED supports three design functions: describing conditions as interaction structures, evaluating structural novelty relative to encoded prior designs, and generating candidate designs under feasibility and governance constraints. We report a lightweight empirical feasibility test that compares graph-blind and SEEDguided generation in a medical-triage design task. In this diagnostic contrast, SEED-guided candidate designs show clearer actor-flow changes, assumptions, and governance checks, supporting the feasibility of the grammar as a design aid. The commentary closes by identifying governance tensions around novelty, replication, validity, diversity of inquiry, and accountability. 

---
# Harnessing LLM Agents with Skill Programs 

**Authors**: Hongjun Liu, Yifei Ming, Shafiq Joty, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.17734)  

**Abstract**: Equipping LLM agents with reusable skills derived from past experience has become a popular and successful approach for tackling complex and long-horizon tasks. However, such lessons are often encoded as textual guidance that remains largely advisory, lacking explicit mechanisms for when and how to intervene in the agent loop. To bridge the gap, we introduce HASP(Harnessing LLM Agents with Skill Programs), a new framework that upgrades skills into executable Program Functions (PFs). Rather than offering passive advice, PFs act as executable guardrails that activate on failure-prone states and modify the next action or inject corrective context. HASP is highly modular: it can be applied at inference time for direct agent-loop intervention, during post-training to provide structured supervision, or for self-improvement by evolving validated, teacher-reviewed PFs. Empirically, HASP drives substantial gains compared to both training-free and training-based methods on web-search, math reasoning, and coding tasks. For example, on web-search reasoning, inference-time PFs alone improve the average performance by 25% compared to (multi-loop) ReAct Agent, while post-training and controlled evolution achieve a 30.4% gain over Search-R1. To provide deeper insights into HASP, our mechanism analysis reveals how PFs trigger and intervene, how skills are internalized, and the requirement for stable skill library evolution. 

---
# Divergence-Suppressing Couplings for Rectified Flow 

**Authors**: Yimeng Min, Carla P. Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2605.17733)  

**Abstract**: The promise of Rectified Flow rests on producing self-generated couplings whose trajectories are straight, or nearly so. In practice, trajectories generated by the base flow model can bend and intertwine, and the resulting coupling inherits this distortion. In this paper, we identify that such trajectory entanglement is often associated with regions of nonzero divergence in the learned velocity field, where local expansion or contraction distorts trajectories and steers particles away from their ideal endpoints. We then propose divergence-suppressing couplings for Rectified Flow, an offline correction that attenuate the divergent component of the learned velocity during coupling generation. The correction is paid only once per coupling pair and amortized over training, so deployment runs plain Euler at identical wall-clock cost to standard Rectified Flow. Empirically, this offline modification yields consistent improvements on 2D synthetic benchmarks and on image generation. 

---
# EXG: Self-Evolving Agents with Experience Graphs 

**Authors**: Yuxin Jin, Siyuan Zhang, Hanchen Wang, Lu Qin, Ying Zhang, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17721)  

**Abstract**: Large language model (LLM)-based agents have demonstrated strong capabilities in complex reasoning and problem solving through multi-step interactions, yet most deployed agents remain behaviorally static, with knowledge acquired during execution rarely translating into systematic improvement over time. In response, a growing line of work on self-evolving agents explores how agents can improve through experience during deployment, but most existing approaches either rely on ad hoc reflection limited to single-task correction or adopt unstructured memory that accumulates fragmented experience with delayed usability. To address this limitation, we introduce EXG, an experience graph framework for self-evolving agents that explicitly organizes accumulated successes and failures into a structured, relational representation. EXG is the first experience graph designed for self-evolving agents, supporting both online, real-time graph growth during execution for immediate cross-task experience reuse, and offline reuse of a consolidated experience graph as an external memory module. This design also enables EXG to serve as a plug-and-play component for existing self-evolving agents, organizing prior experience into a unified experience graph and improving both solution quality and resource efficiency as deployment progresses. Extensive experiments across code generation and reasoning benchmarks show that EXG attains more favorable performance-efficiency trade-offs than reflection- and memory-based baselines in both online and offline evaluations. Our results suggest that structuring experience as a graph provides a principled foundation for scalable and transferable self-evolving agent behavior. 

---
# EGI: A Multimodal Emotional AI Framework for Enhancing Scrum Master Real-time Self-Awareness 

**Authors**: Jingni Huang, Peter Bloodsworth  

**Link**: [PDF](https://arxiv.org/pdf/2605.17684)  

**Abstract**: While increasing research focuses on the emotional well-being of agile team members, a significant gap remains in emotion monitoring studies for Scrum Masters and meeting organizers, whose impact on team dynamics is crucial. This paper proposes a novel application integrating four carefully selected and recommended AI models to monitor the unconsciously expressed emotions of these key roles. This is achieved through: real- time transcription using a speech-to-text model; thresholding for intonation analysis to detect emotional cues in prosody; applying emotion-based vocabulary matching to identify sentiment in spoken content; and providing context-aware suggestions containing emotion keywords using an open-source, multi-module AI API. The system achieved an ASR word error rate WER of 10% in simulated meeting environments. Our evaluation shows that real- time feedback significantly improves emotion awareness during simulated agile meetings, providing Scrum Masters and meeting organizers with real-time and practical suggestions to help them quickly identify and minimize the expression of negative emotions, fostering more positive and effective team interactions. 

---
# Multimodal Cultural Heritage Knowledge Graph Extension with Language and Vision Models 

**Authors**: Yang Zhang, Nada Mimouni, Jean-Claude Moissinac, Fayçal Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2605.17669)  

**Abstract**: The preservation and interpretation of cultural heritage increasingly rely on digital technologies, among which Knowledge Graphs (KGs) stand out for their ability to structure vast amounts of data. However, the construction and expansion of these KGs often face challenges due to the diverse and complex nature of cultural heritage information. In this paper, we propose a novel approach for extending KG resources in the domain of cultural heritage, which we applied to French data. First, we introduce a new knowledge graph in the domain of French cultural heritage, WJoconde, which is distinguished by its multimodality as it integrates both textual and image information of the entities. We further introduce three variants of WJoconde to facilitate downstream research, such as Knowledge Graph Completion (KGC). We also built a comprehensive benchmark for KGC methods on our dataset. Second, we propose a new framework for extending cultural heritage KGs using multi-modal approaches leveraging Large Language Models (LLMs) and Vision-Language Models (VLMs), which includes automated data extraction from unstructured resources combined with a special validation pipeline for grounding the output of both models, to further extend WJoconde. Our results show that by integrating the rich text and image information in cultural heritage data, we can efficiently enhance KGs with high reliability. We open-source all code and benchmark datasets with text and images, as well as the original data with an interactive access point 

---
# SAPO: Step-Aligned Policy Optimization for Reasoning-Based Generative Recommendation 

**Authors**: Zaiyi Zheng, Guanghui Min, Yaochen Zhu, Liang Wu, Liangjie Hong, Chen Chen, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.17648)  

**Abstract**: Generative recommendation treats next-item prediction as autoregressive item-identifier generation. Specifically, items are encoded as semantic identifiers (SIDs), which are short coarse-to-fine token sequences whose early tokens capture broad semantics and later tokens refine them. Recent work augments this paradigm with reasoning traces and optimizes them via reinforcement learning with verifiable rewards, typically outcome-reward algorithm with exact-match feedback on the generated SID. However, in large-catalog recommendation, exact-match feedback on the generated SID only reports whether the final item is correct; when a generated SID mismatches, outcome-reward cannot identify which SID-token prediction caused the mismatch and may penalize matched SID-token positions together with the mismatched position. We identify that the natural unit of credit assignment in this setting is a single reasoning step (one thinking block paired with one SID token). We instantiate this idea in SAPO (Step-Aligned Policy Optimization): rather than broadcasting one advantage to the whole response, SAPO computes a separate group-relative advantage for each reasoning step and applies it only to the corresponding thinking block and SID token. Across three real-world recommendation datasets, SAPO stabilizes reinforcement-learning training and consistently improves over existing generative recommendation baselines, with the largest gains where sparse exact-match feedback makes reasoning-step credit assignment important. Our results suggest that reinforcement-learning objectives for structured generation should mirror the decoder's own decomposition of the output. 

---
# Causal Intervention-Based Memory Selection for Long-Horizon LLM Agents 

**Authors**: Saksham Sahai Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2605.17641)  

**Abstract**: Long-horizon LLM agents rely on persistent memory to support interactions across sessions, yet existing memory systems often retrieve context using semantic similarity or broad history inclusion, treating retrieved memories as uniformly useful. This assumption is fragile because memories may be topically related while remaining irrelevant, stale, or misleading. We propose Causal Memory Intervention (CMI), a causal memory-selection technique that estimates how candidate memories affect the model's answer under controlled interventions, selecting memories that improve task performance while suppressing unstable, irrelevant, or harmful ones. To evaluate this setting, we introduce Causal-LoCoMo, a causally annotated benchmark derived from long conversational data, where each example contains a user request, a structured memory bank, useful memories, irrelevant distractors, and synthetic harmful memories. We compare CMI against vector, graph, reflection, summary, full-history, and no-memory baselines. Results show that CMI achieves a stronger balance between answer quality and robustness to misleading memory, suggesting that reliable long-term memory requires selecting context based on causal usefulness rather than relevance alone. The full framework, benchmark construction code, and experimental pipeline are available at this https URL. 

---
# WebGameBench: Requirement-to-Application Evaluation for Coding Agents via Browser-Native Games 

**Authors**: Wenyu Zhang, Guoliang You, Tianlun, Haotian Zhao, Tianshu Zhu, Haoran Wang, Xiaoxuan Tang, Mingyang Dai, Jingnan Gu, Daxiang Dong, Jianmin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17637)  

**Abstract**: Coding agents are increasingly used as application builders, yet many evaluations still focus on source code, repository-level tests, or intermediate traces rather than the delivered application. We introduce WebGameBench, a requirement-to-application benchmark that evaluates whether coding agents can turn a frozen Structured WebGame Specification into a browser-accessible game. Browser-native games provide a compact but behavior-dense testbed: even simple games require coordinated input handling, spatial mapping, rule execution, state transitions, terminal conditions, restart behavior, and visible feedback. In WebGameBench, each generated artifact is built, served, and exposed as a browser-accessible application under a unified deployment protocol. A runtime evaluator then interacts with the delivered game in a real browser and assigns a three-way label: EXCELLENT, USABLE, or UNUSABLE. On a human-reviewed subset, the runtime label is broadly aligned with human gameplay review under the Usable-rate criterion. Across 111 tasks, 12 coding agents, and 14 evaluation configurations, WebGameBench separates current systems: the best configuration reaches a 76.9% usable rate but only a 20.2% excellent rate. This gap shows that crossing the minimum playable-delivery threshold is still far from complete requirement satisfaction. To our knowledge, WebGameBench is the first requirement-to-application benchmark for browser-native game delivery that validates delivered-application runtime labels against independent human gameplay review under the Usable-rate criterion. 

---
# Episodic-Semantic Memory Architecture for Long-Horizon Scientific Agents 

**Authors**: Nikola Milosevic  

**Link**: [PDF](https://arxiv.org/pdf/2605.17625)  

**Abstract**: As Large Language Models (LLMs) evolve into persistent scientific collaborators, context window saturation has emerged as a critical bottleneck. Scientific workflows involving iterative data analysis and hypothesis refinement rapidly saturate even extended contexts with dense technical content, while monolithic approaches suffer from quadratic cost scaling and cognitive degradation. We evaluate a Dual Process Memory Architecture that decouples immediate episodic needs (constant 10-message window) from long-term consolidated knowledge (growing at approximately 3 tokens/message). Unlike prior social agent memory systems, our domain-specific consolidation addresses contradictory parameter evolution, multi-hop reasoning across experimental phases, and precise technical fact retention. Through large-scale evaluation spanning 15,000 messages with cross-model validation across six LLMs from three families (OpenAI, Anthropic, Google), totaling 1,440 queries, we establish three key findings. First, while full-context models fail at 10,000 messages due to context overflow, our system maintains 70-85% accuracy with 1-2 second latency using 62% fewer tokens (45,434 vs 120,000+ limit). Second, cross-model validation reveals architecture-level trade-offs independent of specific LLMs: Dual Process excels at numeric/temporal queries (65-90% accuracy) while RAG excels at historical retrieval (60-85%), suggesting complementary deployment strategies. Third, we identify a "Sim-to-Real" gap where synthetic tests maintain constant memory but realistic workflows exhibit linear growth (about 3 tokens/message), with consolidation quality emerging as the primary scalability bottleneck. The architecture successfully manages profiles with 14,000+ scientific facts (125k tokens), demonstrating that domain-specific memory consolidation enables sustained operation beyond full-context limits. 

---
# Prediction of Challenging Behaviors Associated with Profound Autism in a Classroom Setting Using Wearable Sensors 

**Authors**: Yadhu Kartha, Conor Anderson, Jenny Foster, Theresa Hamlin, Johanna Lantz, Ryan Lay, Juergen Hahn, Gari D. Clifford, Hyeokhyen Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2605.17618)  

**Abstract**: Autism Spectrum Disorder (ASD) is characterized by challenges with social interaction and communication and by restricted or repetitive patterns of thought and behavior, with significant variability in presentation. Approximately a quarter of children with ASD are classified as having profound autism, who often exhibit challenging behaviors, such as self-injurious behavior, aggression, elopement, or pica, that pose serious safety risks and disrupt learning in educational settings. Prior work has applied wearable sensors and machine learning to detect challenging behaviors, but has been largely confined to controlled laboratory environments. This work demonstrates that predicting challenging behavior episodes is feasible in a real-world special education classroom. We collected approximately 110.7 hours of labeled multimodal wearable data comprising accelerometry, electrodermal activity (EDA), and skin temperature from 9 children and young adults aged 10 to 21 years across standard classroom sessions. We fine-tuned state-of-the-art foundation models for multimodal wearable time-series analysis and show that challenging behavior episodes can be predicted up to 10 minutes in advance with an AUC-ROC of 0.78. These results establish a concrete foundation for developing proactive in-class intervention systems that enable teachers to minimize the safety risks of challenging behaviors in special education classrooms 

---
# GraphMind: From Operational Traces to Self-Evolving Workflow Automation 

**Authors**: Yiwen Zhu, Joyce Cahoon, Anna Pavlenko, Qiushi Bai, Nima Shahbazi, Divya Vermareddy, Meina Wang, Mathieu Demarne, Swati Bararia, Wenjing Wang, Hemkesh Vijaya Kumar, Hannah Lerner, Katherine Lin, Steve Toscano, Miso Cilimdzic, Subru Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2605.17617)  

**Abstract**: Complex operational workflows coordinating personnel, tools, and information are central to enterprise operations, yet end-to-end automation remains challenging due to extensive requirements for human inputs and the inability to adapt over time. We present GraphMind, an end-to-end system that constructs, executes, and evolves action-centric workflow graphs without human effort. The system operates in three phases. First, a scalable offline pipeline extracts structured workflow graphs from large volumes of human resolution traces, capturing problems, actions, and their causal relationships. Second, an online multi-agent traversal engine navigates the graph to dynamically construct and execute workflows, combining graph-guided retrieval with LLM-driven reasoning at each step. Third, Adaptive Traversal Reinforcement (ATR) reinforces successful traversal paths and decays stale elements. This closed-loop mechanism enables the graph to self-optimize and adapt to shifting operational conditions. GraphMind has been deployed across four production cloud database services for incident investigation. Evaluated on production data, the system substantially outperforms a Trace-RAG baseline in mitigation reach, groundedness, and diagnostic throughput, scoring 4.95/5 in blind expert review. The ATR layer provides further gains across all metrics, demonstrating that workflow graphs can learn and improve from execution-derived feedback. 

---
# AutoRubric-T2I: Robust Rule-Based Reward Model for Text-to-Image Alignment 

**Authors**: Kuei-Chun Kao, Daixuan Huo, Yuanhao Ban, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2605.17602)  

**Abstract**: Aligning Text-to-Image (T2I) generation models with human preferences increasingly relies on image reward models that score or rank generated images according to prompt alignment and perceptual quality. Existing reward models are commonly trained as Bradley-Terry (BT) preference models on large-scale human preference corpora, making them costly to train, difficult to adapt, and opaque in their evaluation criteria. Meanwhile, Vision-Language Model (VLM) judges can provide more fine-grained assessments through textual rubrics, but their manually designed or heuristically generated scoring rules may fail to reliably reflect human preferences. In this paper, we propose AutoRubric-T2I, the first rubric learning framework in T2I that automatically synthesizes and selects explicit rubrics for guiding VLM judges. AutoRubric-T2I first synthesizes reasoning traces from preference pairs into candidate rubrics, then uses a VLM judge to score paired images under each rubric, producing pairwise rubric-score differences for preference learning. To remove noisy and redundant rules, we further employ a $\ell_1$-Regularized Logistic Regression Refiner, which selects the Top-$N$ most discriminative rubrics. Extensive evaluations show that AutoRubric-T2I produces high-quality, interpretable reward signals using less than 0.01% of the annotated preference data, substantially reducing the need for large-scale reward-model training. On image reward benchmarks such as MMRB2, AutoRubric-T2I outperforms strong reward model baselines. We further validate AutoRubric-T2I as an RL reward on downstream T2I tasks, including TIIF and UniGenBench++, where it improves generation quality over scalar reward models using the Flow-GRPO pipeline on diffusion models. 

---
# NeuSymMS: A Hybrid Neuro-Symbolic Memory System for Persistent, Self-Curating LLM Agents 

**Authors**: Mujahid Sultan, Sri Thuraisamy, Daya Rajaratnam  

**Link**: [PDF](https://arxiv.org/pdf/2605.17596)  

**Abstract**: We present NeuSymMS, an adaptive memory system that enables large language model (LLM) agents to learn, remember, and reason about users across sessions via a hybrid neuro-symbolic architecture. NeuSymMS couples neural fact extraction from unstructured dialogue with a CLIPS-based expert system that classifies, deduplicates, and reconciles facts under explicit lifecycle rules. The system represents knowledge as subject-relation-value triples stored in relational database management system, supports user/agents/agent-to-agents scoping, and implements a dual-horizon short-term/long-term memory model with access-based promotion and time-based pruning. NeuSymMS maintains continuity of memory while avoiding context-window bloat and cross-entity contamination. We argue that this architecture offers a practical path to trustworthy, auditable memory for production agentic systems and discuss its novelty relative to log retrieval, summarization, and key-value approaches. 

---
# ECG-WM: A Physiology-Informed ECG World Model for Clinical Intervention Simulation 

**Authors**: Zhikang Chen, Yue Wang, Sen Cui, Yu Zhang, Changshui Zhang, Tianling Ren, Tingting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17580)  

**Abstract**: Electrocardiogram (ECG)-based models have achieved strong performance in diagnostic tasks, yet they remain limited in modeling how cardiac dynamics evolve under external interventions. In particular, existing approaches focus primarily on static prediction and lack mechanisms to capture ECG variations under different pharmacological conditions. In this work, we propose an ECG World Model for action-conditioned predictive simulation of cardiac electrophysiology. Moving beyond disjoint pipelines, our framework features a principled integration of physiological ordinary differential equation (ODE) priors into latent diffusion dynamics via energy regularization. This structural constraint enables the synthesis of physiologically plausible post-intervention ECG trajectories while effectively mitigating generative hallucinations. Building on this simulation process, we introduce an uncertainty-aware evaluation strategy that leverages the stochasticity of diffusion sampling to characterize both the expected clinical risk and its variability, allowing a more reliable comparative assessment of candidate interventions. We evaluate our method across diverse settings, including controlled drug-response scenarios and real-world clinical records. Beyond standard waveform metrics, experimental results demonstrate improved risk calibration and strong alignment with expert-informed treatment preferences. These results establish our approach as a robust foundation for safe and intervention-aware clinical decision support. 

---
# Generalization or Memorization? Brittleness Testing for Chess-Trained Language Models 

**Authors**: Ethan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17565)  

**Abstract**: Recent work has fine-tuned language models on chess data and reported high benchmark scores as evidence that the resulting models can understand the rules of chess, play full chess games at a professional level, or generate human-readable explanations grounded in expert knowledge. We train KinGPT, a 25M-parameter character-level language model trained only on (position, best-move) pairs, who exceeds 3B-parameter ChessGPT on a 600-puzzle mate-in-N suite and 4B-parameter C1-4B over a 20-theme puzzle benchmark. We examine several claims made in existing literature regarding chess-trained language models and assert that their impressive benchmark performance is largely explained by pattern-matching. We also demonstrate how LLM-Modulo, a verifier-in-the-loop framework, raises RedPajama 3B's best move accuracy from 1.2% to 21.2% and move generation validity from 19.3% to 95.3% on mate-in-N chess puzzles, comparable to gains achieved from ChessGPT's fine-tuning on chess-specific web corpora at a fraction of the cost. Our results illustrate how pairing a general LLM with an external verifier offers a more flexible alternative to directly training on synthetic data for well-defined domains. We open source all training/evaluation code, datasets, puzzle samples, and KinGPT model checkpoints for reproducibility. 

---
# Evaluating Deep Research Agents on Expert Consulting Work: A Benchmark with Verifiers, Rubrics, and Cognitive Traps 

**Authors**: Tanmay Asthana, Aman Saksena, Divyansh Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17554)  

**Abstract**: Frontier deep research agents (DRAs) plan a research task, synthesize across documents, and return a structured deliverable on demand. They are being deployed in enterprise workflows faster than they are being evaluated. Existing benchmarks measure factual recall, single-hop QA, or generic agentic skill, missing the multi-document, decision-grade work DRAs are deployed to produce. We introduce a benchmark targeting the structured analytical deliverables that fill a management consultant's typical week.
We grade three frontier agents, namely Claude Opus 4.6 with web search, OpenAI o3-deep-research, and Google Gemini 3.1 Pro deep-research, on 42 SME-authored prompts. Each of the 126 responses is scored on two layers: deterministic ground-truth verifiers (mean 13.8 per task) and a five-criterion 0-3 SME rubric, composed into a Verifier-Rubric Score (VRS) on 0-100. Most prompts embed cognitive traps that penalize surface-pattern matching. Acceptance under our joint threshold (rubric mean >= 2.5 and verifier rate >= 80%) is uniformly low: Gemini 21.4%, o3 9.5%, Claude 9.5%.
Mean VRS scores agree with published rubric-based benchmarks (our top 62.6 vs. APEX-v1 64.2, ProfBench 65.9, ResearchRubrics < 68%), validating the rubric construct. ACCEPT rates sit below APEX-Agents' MC-segment Pass@1 band (12.3-22.7%) on dedicated DR agents; our floor is three points lower despite the harness advantage, opened by stricter conjunctive grading and trap design.
Each agent fails distinctively. Claude produces the deliverable most reliably (4.5x the others' rate on file-required tasks) but carries the highest fabrication signature. o3 has the cleanest reasoning average yet drops required sections and propagates arithmetic errors. Gemini is bimodal, with the highest acceptance rate alongside the most zero-scored rubric cells. 

---
# Memory-Guided Tree Search with Cross-Branch Knowledge Transfer for LLM Solver Synthesis 

**Authors**: Fatemeh Haji, Javier Delarosa Quiros, Peyman Najafirad  

**Link**: [PDF](https://arxiv.org/pdf/2605.17539)  

**Abstract**: Combinatorial optimization (CO) underlies decision-making from logistics to chip design, where infeasible solutions are operationally unusable and small quality gains translate into substantial economic value. Recent work uses large language models (LLMs) to automate solver synthesis: generating executable solver programs from natural-language specifications. However, existing tree-search and evolutionary agents refine candidate trajectories in parallel without explicit knowledge transfer, reintroducing the same constraint violations and converging on similar algorithm families. We introduce MEMOIR, a memory-guided tree-search framework with a two-level memory hierarchy: branch-local memory preserves execution-grounded refinement details within a branch as it iterates on a single algorithmic design, while global memory stores compressed algorithmic and failure-mode summaries across branches. A reflection step at branch termination distills these summaries, enabling cross-branch transfer without polluting future contexts with low-level debugging traces. Across seven CO problems spanning scheduling, routing, packing, and geometric design, MEMOIR achieves 96.7% solution validity (a 9.2 point gap over the strongest baseline) and improves the average normalized score by 7.3 points at matched per-method execution budget. Over three independent runs on four problems, MEMOIR's run-to-run validity standard deviation is more than an order of magnitude below that of every baseline we evaluated in this setting, suggesting that memory-guided exploration yields consistent improvements rather than reflecting sampling variance. 

---
# Self-supervised Hierarchical Visual Reasoning with World Model 

**Authors**: Yuanfei Xu, Lin Liu, Wengang Zhou, Mingxiao Feng, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.17537)  

**Abstract**: 3D open-world environments with adversarial opponents remain a core challenge for reinforcement learning due to their vast state spaces. Effective reasoning representations are essential in such settings. While existing self-supervised visual foresight reasoning approaches often suffer from multi-step error accumulation, many recent studies resort to injecting domain-specific knowledge for more stable guidance. Our key insight is that the photorealistic fidelity of visual reasoning representations is secondary; what truly matters is providing informative, task-relevant signals. To this end, we propose ResDreamer, a hierarchical world model in which each higher-level layer is trained to reconstruct the residuals of the layer below. This design enables progressive abstraction of increasingly sophisticated world dynamics and fosters the emergence of richer latent representations. Drawing inspiration from the "Bitter Lesson", ResDreamer trains its reasoning representations in a purely self-supervised manner. The higher-level residual representations are used to modulate lower-level predictions, allowing the world model to scale effectively with only linearly increasing cross-layer communication costs. Experiments show that ResDreamer achieves state-of-the-art sample efficiency and parameter efficiency. This scalable hierarchical visual foresight reasoning architecture paves the way for more capable online RL agents in open-ended, dynamic environments. The code is accessible at \url{this https URL}. 

---
# RAG-based EEG-to-Text Translation Using Deep Learning and LLMs 

**Authors**: Enrico Collautti, Xiaopeng Mao, Luca Tonin, Stefano Tortora, Sadasivan Puthusserypady  

**Link**: [PDF](https://arxiv.org/pdf/2605.17503)  

**Abstract**: The decoding of linguistic information from electroencephalography (EEG) signals remains an extremely challenging problem in brain-computer interface (BCI) research. In particular, sentence-level decoding from EEG is difficult due to the low signal-to-noise ratio of these recordings. Previous studies tackling this problem have typically failed to surpass random baseline performance unless teacher forcing is used during the inference phase. In this work, we propose a retrieval-augmented generation (RAG)-based sentence-level EEG-to-text decoding pipeline that combines an EEG encoder aligned with semantic sentence embeddings, a vector retrieval stage, and a large language model (LLM) to refine retrieved sentences into coherent output. Experiments are conducted on the Zurich Cognitive Language Processing Corpus (ZuCo) dataset, which contains single-trial EEG recordings collected during silent reading. To evaluate whether the system extracts meaningful information from these EEG signals, the results are compared with a random baseline. In nine subjects, the proposed pipeline outperforms the random baseline, achieving a mean cosine similarity of 0.181 +- 0.022 compared to 0.139 +- 0.029 for the baseline, corresponding to a relative improvement of 30.45%. Statistical analysis further confirms that this improvement is significant, following a strict evaluation workflow where inference is performed without access to ground-truth labels. 

---
# The Capability Paradox: How Smarter Auditors Make Multi-Agent Systems Less Secure 

**Authors**: Qiqi Liu, Thorsten Holz, Shilin Ye, Runhan Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.17480)  

**Abstract**: Multi-agent systems extend large language models (LLMs) by decomposing tasks among specialized agents, but their distributed decision process creates new attack surfaces. We identify \emph{semantic hijacking}, an attack in which harmful requests are concealed within domain-specific narratives and propagated to a Manager through Worker reports, without any syntactic injection primitives. Across 42,000 adversarial trials over 12 Manager models and 7 Worker configurations, we uncover a \emph{capability paradox}: as Worker capability increases, the mean system-level Attack Success Rate (ASR) increases from 18.4% to 63.9%, peaking at 94.4%. To explain this effect, we conduct multi-level mediation analysis on two independent datasets (47,807 interactions). This analysis shows that this paradox is driven by \emph{linguistic certainty}: stronger Workers are more likely to interpret adversarial narratives as legitimate, convey their conclusions assertively, and thereby lead Managers to treat such confident endorsements as justification to execute. In our larger Worker-Only setting ($n_W$=14), certainty mediates 74% of the effect, with 95% confidence intervals (CI) excluding zero under both Monte Carlo and cluster bootstrap; the smaller Full-MAS setting ($n_W$ =6) shows a directionally consistent indirect effect. Worker-side safety prompting does not reliably mitigate this failure. Building on the mediation finding, we propose \emph{heterogeneous ensemble verification}, which pairs Workers of asymmetric domain competence so their complementary vulnerabilities break the certainty-to-execution chain, reducing ASR from 52.8% to 2.0% with negligible benign-task impact. Our results show that upgrading components to stronger models can actively degrade system security, and that effective defenses require exploiting--rather than eliminating--capability asymmetries between agents. 

---
# Multi-Party Multi-Objective Optimization as Consensus Search: Runtime Analysis of Cross-Party Recombination 

**Authors**: Xiaolei Fang, Peilan Xu, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.17454)  

**Abstract**: Multi-party multi-objective optimization problems (MPMOPs) require consensus among autonomous decision makers and therefore differ from flattened many-objective formulations. Existing runtime theory for multi-objective evolutionary algorithms is largely tailored to single-party Pareto-front approximation and does not directly explain common-solution search in MPMOPs. We investigate cross-party recombination in two representative settings. On MP-JCG, a pseudo-Boolean benchmark with an explicit gap region, we prove that a payoff-guided mutation baseline faces a gap-crossing bottleneck requiring \(\Theta(n^2)\) expected fitness evaluations. In contrast, an analytical CPR-NSGA-II variant discovers both common Pareto-optimal solutions in \(O(n\log n)\) expected evaluations by directly assembling complementary prefix and suffix templates distributed across party populations. Comparing this with the flattened four-objective formulation F-JCG, our full-front coverage analysis illustrates the additional coverage burden introduced by flattening. For BPBOMST, the bi-party, two-objective-per-party specialization of the multi-party multi-objective minimum spanning tree problem, we develop a layered support-cover analysis. For each common Pareto objective vector, the symmetric average projection induces an auxiliary bi-objective MST instance, and suitable support representatives yield a \(2\lambda\)-common approximation cover with \(\lambda\in[1,2]\). We further derive an instance-parameterized expected runtime bound for a representative-pool CPR-NSGA-II variant using edge-union recombination and uniform repair. This bound separates the effects of local auxiliary-front filling, cross-party recombination shortcuts, and edge-union repair ambiguity. 

---
# Computational Challenges in Token Economics: Bridging Economic Theory and AI System Design 

**Authors**: Ou Wu, Yingjun Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.17410)  

**Abstract**: Token economics has emerged as a useful lens for understanding resource allocation, value creation, and pricing in large language model systems. While recent work has increasingly treated tokens as economic primitives, there remains a substantial gap between high-level economic theory and the computational realities of modern AI infrastructure. This paper identifies and analyzes the key computational challenges that arise when token-economic principles are implemented in real-time inference systems. We argue that computational feasibility is not merely one dimension of token economics, but its governing constraint: these challenges are driven by fundamental tensions among fine-grained valuation, low-latency execution, and allocation optimality under uncertainty. To structure this problem space, we introduce the notion of \textbf{Computational Token Economics} and propose the \textbf{Token Economics Trilemma} -- a conditional no-free-lunch principle that captures the inherent trade-offs among granularity, real-time performance, and optimality. We further categorize the main technical challenges into three areas: real-time value accounting, constrained resource allocation, and economic-aware system architecture. Rather than presenting a complete solution, this paper aims to define a research agenda for bridging token economics and AI system design, highlighting open problems at the intersection of computational economics, machine learning systems, and AI infrastructure. 

---
# Heterogeneous Information-Bottleneck Coordination Graphs for Multi-Agent Reinforcement Learning 

**Authors**: Wei Duan, Junyu Xuan, En Yu, Xiaoyu Yang, Jie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17393)  

**Abstract**: Coordination graphs are a central abstraction in cooperative multi-agent reinforcement learning (MARL), yet existing sparse-graph learners lack a theoretically grounded mechanism to decide which edges should exist and how much information each edge should carry. Current methods rely on heuristic criteria that offer no formal guarantee on the learned topology, and no principled way to allocate different communication capacities to structurally different agent relationships. To address this, we propose Heterogeneous Information-Bottleneck Coordination Graphs (HIBCG), which learns a group-aware sparse graph in which both edge existence and message capacity are theoretically justified. With the graph information bottleneck (GIB) serving as the underlying tool, HIBCG first constructs a group-aligned block-diagonal prior that provides a closed-form criterion for edge retention -- determining which edges should exist and at what density per group block -- and then controls per-agent feature bandwidth on the resulting topology, compressing messages to retain only task-relevant content. We prove that the group-aligned prior strictly tightens the variational bound on topology learning, that the objective decomposes per group block, enabling differential edge control, and that capacity allocation follows a water-filling principle. 

---
# QQJ: Quantifying Qualitative Judgment for Scalable and Human-Aligned Evaluation of Generative AI 

**Authors**: Marjan Veysi, Pirooz Shamsinejadbabaki, Mohammad Zare, Mohammad Sabouri  

**Link**: [PDF](https://arxiv.org/pdf/2605.17382)  

**Abstract**: The rapid progress of generative artificial intelligence has exposed fundamental limitations in existing evaluation methodologies, particularly for open-ended, creative, and human-facing tasks. Traditional automatic metrics rely on surface-level statistical similarity and often fail to reflect human perceptions of quality, while purely human evaluation, although reliable, is costly, subjective, and difficult to scale. Recent approaches using large language models as evaluators offer improved scalability but frequently lack explicit grounding in human-defined evaluation principles, leading to bias and inconsistency. In this paper, we introduce Quantifying Qualitative Judgment (QQJ), a scalable and human-centric evaluation framework that explicitly bridges the gap between human judgment and automated assessment. QQJ separates the definition of quality from its execution by anchoring evaluation in expert-designed, multi-dimensional rubrics and calibrating large language model evaluators to align with expert reasoning using a small, high-quality annotation set. This design enables consistent, interpretable, and scalable evaluation across diverse generative tasks and modalities. Extensive experiments on text and image generation demonstrate that QQJ achieves substantially stronger alignment with human judgment than traditional automatic metrics and unconstrained LLM-based evaluators. Moreover, QQJ exhibits improved stability across repeated evaluations and superior diagnostic capability in identifying critical failure modes such as hallucination and intent mismatch. These results indicate that structured qualitative judgment can be operationalized at scale without sacrificing interpretability or human alignment, positioning QQJ as a practical foundation for reliable evaluation of modern generative AI systems. 

---
# ADR: An Agentic Detection System for Enterprise Agentic AI Security 

**Authors**: Chenning Li, Pan Hu, Justin Xu, Baris Ozbas, Olivia Liu, Caroline Van, Manxue Li, Wei Zhou, Mohammad Alizadeh, Pengyu Zhang, KK Sriramadhesikan, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17380)  

**Abstract**: We present the Agentic AI Detection and Response (ADR) system, the first large-scale, production-proven enterprise framework for securing AI agents operating through the Model Context Protocol (MCP). We identify three persistent challenges in this domain: (1) limited observability -- existing Endpoint Detection and Response (EDR) tools see file writes but not the agent reasoning, prompts, or causal chains linking intent to execution; (2) insufficient robustness -- static defenses constrained by pre-defined rules fail to generalize across diverse attack techniques and enterprise contexts; and (3) high detection costs -- LLM-based inference is prohibitively expensive at scale. ADR addresses these challenges via three components: the ADR Sensor for high-fidelity agentic telemetry, the ADR Explorer for systematic pre-deployment red teaming and hard-example generation, and the ADR Detector for scalable, two-tier online detection combining fast triage with context-aware reasoning. Deployed at Uber for over ten months, ADR has sustained reliable detection in production with growing adoption reaching over 7,200 unique hosts and processing over 10,000 agent sessions daily, uncovering hundreds of credential exposures across 26 categories and enabling a shift-left prevention layer (97.2% precision, 206 detected credentials). To validate the approach and enable community adoption, we introduce ADR-Bench (302 tasks, 17 techniques, 133 MCP servers), where ADR achieves zero false positives while detecting 67% of attacks -- outperforming three state-of-the-art baselines (ALRPHFS, GuardAgent, LlamaFirewall) by 2--4x in F1-score. On AgentDojo (public prompt injection benchmark), ADR detects all attacks with only three false alarms out of 93 tasks. 

---
# CBT-Audio: Evaluating Audio Language Models for Patient-Side Distress Intensity Estimation in CBT Session Recordings 

**Authors**: Qixuan Hu, Shuchang Ye, Xumou Zhang, Anastasia Serafimovska, Anastasia Suraev, Amit Saha, Ping-hsiu Lin, Sydney Su, Usman Naseem, Adam G. Dunn, Jinman Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.17370)  

**Abstract**: Cognitive behavioural therapy is widely used to help patients understand and manage psychological distress. It is often delivered through spoken conversation, where therapists attend not only to what patients say, but also to how they say it, because these cues can help therapists decide how to respond and adapt treatment. Progress in building AI systems for CBT remains largely limited to text, partly because most available datasets are text based and shareable spoken CBT data are scarce under ethical and privacy constraints. This creates a blind spot because text based models and evaluations cannot capture the mismatch between the transcript and the patient's voice, even though therapists often rely on this mismatch to understand patient distress. We introduce CBT-Audio, a dataset for evaluating patient distress estimation from spoken CBT sessions with audio language models. CBT-Audio contains 1,802 patient turns from 96 publicly available CBT recordings, with turn-level distress labels validated on an experts-annotated subset. We evaluate 10 open source audio language models under three input conditions, where models receive only patient audio, only the transcript, or both audio and transcript. Our results show that audio can provide useful information beyond text, especially when combined with transcripts. Adding audio to transcript input improves distress estimation over using the transcript alone in 8 of 10 model families, with significant gains in 4, and case studies show the clearest benefit when verbal content and vocal delivery diverge. CBT-Audio makes spoken patient behaviour measurable for AI evaluation in CBT-related tasks and supports future work on audio language models for mental health interaction. 

---
# HyperPersona: A Multi-Level Hypergraph Framework for Text-Based Automatic Personality Prediction 

**Authors**: Sina Heydari, Majid Ramezani  

**Link**: [PDF](https://arxiv.org/pdf/2605.17355)  

**Abstract**: As a modern commodity, language has become a vast repository of socially and psychologically significant traits and concepts, reflecting the ways people encode pattern of thoughts, behaviors, and emotions into words. Text-based Automatic Personality Prediction (APP), seeks to infer personality from linguistic behavior, offering a scalable alternative to traditional psychometric assessments. Although text is inherently hierarchical, with the document-level capturing global features, the sentence-level encoding local semantics, and the word-level providing fine-grained lexical information, most existing approaches rely on shallow, sequential, or single-level representations that ignore the multi-level structure of written language. To address this, we propose HyperPersona, a framework that explicitly models the hierarchical organization of text (document, sentence, and word) through hypergraph structure, where a document and its sentences are represented as hyperedges, and the words are represented as nodes, enabling joint modeling of global, local, and lexical dependencies of text. Followed by a transformer-based graph encoder that learns interactions within and across these linguistic layers, yielding context-sensitive and structurally grounded feature representations for personality prediction. Experiments on the Big Five personality dimensions show that, while relying solely on text, HyperPersona effectively integrates multi-level linguistic cues, achieving superior performance compared to state-of-the-art baselines. These findings underscore the critical role of textual hierarchy in advancing human-like personality inference from natural language. 

---
# Reasoning Before Diagnosis: Physician-Inspired Structured Thinking for ECG Classification 

**Authors**: Yang Wu, Xiaoyan Yuan, Hau-San Wong, Xiping Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17308)  

**Abstract**: Electrocardiogram (ECG) diagnosis in clinical practice relies on structured reasoning over multiple hierarchical aspects, including cardiac rhythm, conduction properties, waveform morphology, and overall diagnostic impression. However, most existing approaches predict labels directly from ECG signals without explicit clinical reasoning, resulting in opaque decisions that lack clinical alignment. To bridge this gap, we propose CardioThink, a physician-inspired multimodal large language model (MLLM) framework that explicitly models the diagnostic reasoning process through human-interpretable intermediate stages (rhythm, conduction, morphology, and impression) to derive final classification results. Furthermore, we introduce Structured Set Policy Optimization (SSPO) to jointly optimize adherence to this structured reasoning format and the accuracy of variable-size diagnostic sets, without requiring manually annotated reasoning traces. Extensive experiments on diverse ECG benchmarks demonstrate the significant superiority of our approach in diagnostic accuracy, while simultaneously providing interpretable clinical reasoning. Notably, reasoning quality evaluations confirm that SSPO substantially enhances the clinical validity of the generated rationales. These findings reveal that moving beyond direct label prediction toward structured reasoning offers a more clinically aligned direction for future ECG modeling. 

---
# CyberCorrect: A Cybernetic Framework for Closed-Loop Self-Correction in Large Language Models 

**Authors**: Yuning Wu, Yingmin Liu, Yang Shu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17305)  

**Abstract**: Large language model (LLM) self-correction -- the ability to detect and fix errors in generated outputs -- remains largely ad hoc, relying on generic prompts such as "please reconsider your answer" without systematic error analysis or convergence guarantees. We propose CyberCorrect, a framework that formalizes LLM self-correction as a closed-loop control system grounded in cybernetic theory. The framework models the LLM generator as the plant and introduces a tri-modal Error Detector (combining self-consistency, verbalized confidence, and logic-chain verification) as the sensor. A type-directed Correction Controller generates targeted repair instructions based on diagnosed error categories, while a Convergence Judge determines iteration termination using stability criteria adapted from control theory. We further introduce three control-theoretic evaluation metrics -- convergence rate, overshoot rate, and oscillation rate -- that capture correction dynamics beyond final accuracy. Experiments on our constructed CyberCorrect-Bench (440 reasoning tasks with annotated error types and correction paths) show that CyberCorrect achieves 79.8% final accuracy, improving upon the best existing self-correction method by 6.2 percentage points, while reducing overshoot (erroneous over-correction) by 41% through its convergence control mechanism. 

---
# MetaCogAgent: A Metacognitive Multi-Agent LLM Framework with Self-Aware Task Delegation 

**Authors**: Chenyu Wang, Yang Shu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17292)  

**Abstract**: Multi-agent large language model (LLM) systems have shown promise for solving complex tasks through agent collaboration. However, existing frameworks assign tasks based on predefined roles without considering whether an agent can accurately assess its own competence boundaries, leading to overconfident execution on tasks beyond its expertise. Inspired by metacognition theory from cognitive science, we propose MetaCogAgent, a multi-agent LLM framework where each agent is equipped with a Metacognitive Self-Assessment Unit that evaluates task-capability alignment before execution. The framework introduces three contributions: (1) a self-assessment mechanism that estimates per-task confidence by combining verbalized uncertainty with historical capability profiles; (2) an adaptive delegation protocol that routes low-confidence tasks to better-suited agents through cross-agent evaluation; and (3) a capability boundary learning module that iteratively refines each agent's competence model via cybernetic feedback. Experiments on our constructed MetaCog-Eval benchmark (700 tasks across 5 cognitive dimensions) demonstrate that MetaCogAgent achieves 82.4% task accuracy -- 8.7% above the best routing baseline -- while using 5% fewer API calls than AutoGen and 34% fewer than ensemble voting. Ablation studies confirm that each metacognitive component contributes to overall system performance. 

---
# A2RBench: An Automatic Paradigm for Formally Verifiable Abstract Reasoning Benchmark Generation 

**Authors**: Qingchuan Ma, Yuexiao Ma, Yongkang Xie, Tianyu Xie, Xiawu Zheng, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.17278)  

**Abstract**: Abstract reasoning ability reflects the intelligence and generalization capacity of LLMs to extract and apply abstract rules. However, accurately measuring this ability remains challenging: existing benchmarks either rely on expensive manual annotation, limiting their scale, or risk measuring memorization rather than genuine reasoning. To address this, we introduce an automated pipeline named A2RBench, encompassing generation, expansion, evaluation, and analysis. Specifically, in the generation stage, LLMs create diverse tasks demanding genuine reasoning; in the expansion stage, LLMs reuse validated rules and expand new input spaces to generate task variations, achieving scaling. However, such a process may cause hallucinations. To eliminate it, we further establish a theoretical framework and prove that programmatic verification--testing whether the inverse operation perfectly reverses the forward operation (cycle consistency)--guarantees a unique solution. Through extensive evaluations on mainstream LLMs, we find: (1) Current LLMs exhibit fundamental deficiencies in abstract reasoning, with top models significantly underperforming humans on a representative subset (39.8% vs. 68.5%). (2) Current LLMs fall far short of 2D and 1D in the complexity of generated 3D tasks, revealing their lack of understanding of high-dimensional tasks. (3) Counterintuitively, inputs with higher information complexity can simplify the reasoning process. 

---
# Is VLA Reasoning Faithful? Probing Safety of Chain-of-Causation 

**Authors**: Nicanor Mayumu, Xiaoheng Deng, Patrick Mukala  

**Link**: [PDF](https://arxiv.org/pdf/2605.17268)  

**Abstract**: We present the first systematic study of faithfulness in Vision-Language-Action (VLA) driving models, analyzing 300 Alpamayo-R1-10B inferences across 100 diverse PhysicalAI-AV scenarios. Our main finding is that output natural-language rationales with trajectories may be significantly unfaithful: (i) overall reasoning fidelity is only 42.5%, with Chain-of-Causation matching scene reality less than half the time; (ii) 94 missed pedestrians in one-third of pedestrian-relevant scenes; (iii) 97.7% trajectory fragility under mild visual perturbations; and (iv) only 48.3% mean reasoning-action consistency, with 53.3% of inferences exhibiting low consistency, including 37.9% of stop-claimed cases where the model continues instead. We formalize faithfulness information-theoretically, define entity and action fidelity with verification criteria, and outline a four-component safety architecture aligned with these results. 

---
# CAM-Bench: A Benchmark for Computational and Applied Mathematics in Lean 

**Authors**: Wentao Long, Yunfei Zhang, Chenyi Li, Li Zhou, Chumin Sun, Zaiwen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2605.17255)  

**Abstract**: Formal theorem-proving benchmarks enable mechanically verifiable evaluation of mathematical reasoning in large language models. However, existing benchmarks mainly focus on Olympiad-style problems and algebraic domains, leaving computational and applied mathematics underrepresented. We introduce CAM-Bench, a Lean 4 theorem-proving benchmark of 1,000 Lean proof targets in computational and applied mathematics, with coverage spanning optimization, numerical linear algebra, and numerical analysis. These problems are adapted from textbook exercises and often depend on locally introduced definitions, notation, algorithms, and elementary results. To construct CAM-Bench, we develop a dependency-recovery pipeline that reconstructs the local textbook context needed to state each problem faithfully. It then normalizes each problem into a standalone informal theorem and translates it into a Lean target. We validate the resulting formal problems through Lean compilation and semantic review, checking both formal correctness and semantic alignment with the original exercises. For each problem, we release the raw exercise, recovered context, normalized informal theorem, and final Lean target. CAM-Bench complements existing formal mathematics benchmarks by targeting applied mathematics problems that rely on textbook concepts and elementary theorems, many of which are not directly available as standard Mathlib4 lemmas. We evaluate widely used large language models and formalization agents on CAM-Bench, and analyze common failure modes in tracking local assumptions, applying elementary results, decomposing proofs, and maintaining long-horizon control in Lean. 

---
# CatalyticMLLM: A Graph-Text Multimodal Large Language Model for Catalytic Materials 

**Authors**: Yanjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.17254)  

**Abstract**: Property prediction and inverse structural design of catalytic materials are typically modeled as two independent tasks: the former predicts target properties from given structures, whereas the latter generates candidate structures according to desired properties. Although the decoupled paradigm facilitates the implementation of a ``generation--evaluation--screening'' workflow, the inconsistency between the generative model and the property prediction model in terms of representation spaces and training objectives can readily introduce data distribution shifts and evaluator bias, thereby limiting the stability of closed-loop optimization.
In this work, we propose QE-Catalytic-V2, a unified graph--text multimodal large language model for catalytic materials, which integrates property prediction and inverse design within the same model and shared representation space. Under this unified framework, QE-Catalytic-V2 can not only perform reliable property prediction by leveraging three-dimensional structures and textual information, but also generate and screen physically feasible CIF candidates conditioned on target properties, thereby forming a closed-loop optimization workflow of ``inverse design--prediction--screening--redesign.'' Experimental results demonstrate that this unified paradigm outperforms decoupled baselines on both catalytic relaxed-energy prediction and inverse design tasks, validating the effectiveness of jointly modeling property prediction and structure generation within a single multimodal model. 

---
# Towards Robust Argumentative Essay Understanding via TIDE: An Interactive Framework with Trial and Debate 

**Authors**: Zheqin Yin, Yupei Ren, Yadong Zhang, Yujiang Lu, Man Lan  

**Link**: [PDF](https://arxiv.org/pdf/2605.17247)  

**Abstract**: Argumentative essays serve as a vital medium for assessing critical thinking and reasoning skills, yet there is limited works on accurately understanding and evaluating such texts via prompt. In this work, we propose TIDE, a novel framework designed to improve criteria-based prompt optimization for argument-related tasks by integrating TrIal and DEbate mechanism. Our method addresses key limitations of criteria-based prompt optimizing by mitigating the influence of noisy training data and enhancing optimization stability. We evaluate TIDE on three core tasks: Automated Essay Scoring, Argument Component Detection, and Argument Relation Identification. Results demonstrate that our framework improves performance across tasks. These findings underscore the potential of combining prompt-based methods for advanced argument understanding. 

---
# ChemVA: Advancing Large Language Models on Chemical Reaction Diagrams Understanding 

**Authors**: Mingyang Rao, Kehua Feng, Zhihui Zhu, Jiangzhen Fu, Hao Yu, Keyan Ding, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.17214)  

**Abstract**: While Large Language Models (LLMs) have revolutionized scientific text processing, they exhibit a significant capability gap when interpreting chemical reaction diagrams. We identify two fundamental bottlenecks restricting current systems: a Visual Deficit, where generic vision encoders struggle to resolve the strict topological connectivity of dense molecular graphs, and a Semantic Disconnect, where standard linear strings, such as SMILES, fail to effectively activate the model's latent chemical reasoning. To bridge these gaps, we propose the Chemical Visual Activation (ChemVA) framework, which employs a Visual Anchor mechanism to ground functional groups via hybrid-granularity detection, followed by a semantic alignment approach that translates visual features into entity names to maximize knowledge activation in LLMs. We evaluate our approach on OCRD-Bench, a newly constructed dataset featuring dense visual-semantic contexts and comprehensive reaction coverage to evaluate the full spectrum from recognition to reasoning. Extensive experiments on OCRD-Bench demonstrate that ChemVA achieves 92.0% structural recognition accuracy. By bridging visual and semantic bottlenecks, our framework delivers a consistent performance gain of approximately 20 percentage points across 9 diverse LLMs, enabling open-weight models to rival proprietary SOTA systems in complex chemical reasoning tasks. 

---
# CAREBench: Evaluating LLMs' Emotion Understanding by Assessing Cognitive Appraisal Reasoning 

**Authors**: Zhaoyue Sun, Hainiu Xu, Andero Uusberg, James J. Gross, Petr Slovak, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2605.17176)  

**Abstract**: Emotion understanding is a core capability for LLMs to interact effectively with humans, yet existing evaluation paradigms rely on discrete emotion label prediction and fail to capture the cognitive processes underlying emotion generation. Grounded in appraisal theory, we introduce CAREBench, the first benchmark with complete inferential chain annotations from both first- and third-person perspectives on real-world narratives, spanning appraisal reasoning, appraisal ratings, and multi-label emotion annotation. We propose a process-level evaluation framework and conduct systematic experiments across six LLMs organized around four research questions. We find that stronger models match or surpass human observers on certain tasks, yet fall short on appraisal reasoning and positive emotion recognition; performance across chain steps and sensitivity to appraisal interventions exhibit dissociations across models; and current models have not internalized the mechanisms needed to capture human subjective heterogeneity. These findings suggest that downstream emotion prediction metrics may overestimate LLMs' true emotion understanding, and CAREBench provides a foundation for more diagnostically informative evaluation of LLMs' affective cognitive capabilities. 

---
# Responsible Agentic AI Requires Explicit Provenance 

**Authors**: Jinwei Hu, Xinmiao Huang, Qisong He, Youcheng Sun, Yi Dong, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17169)  

**Abstract**: Agentic AI is rapidly proliferating across diverse real-world domains such as software engineering, yet public trust has not kept pace. The central reason is that responsibility, despite being widely discussed, remains a subjective and unenforced concept, as no current agentic framework produces the quantifiable, traceable, and interventionable provenance needed to assign it when harm emerges from compositions no single party designed. We position that what is missing is not better benchmark-level evaluation but $\textbf{explicit provenance}$ across the full agentic lifecycle, which is the only viable basis for making responsibility computable and actionable. We advance this agenda along four axes: establishing $\textit{why}$ such provenance is a structural necessity by identifying responsibility gaps across sociotechnical dimensions, formalizing $\textit{what}$ it must encode through a causal attribution function and responsibility tensor, discussing $\textit{how}$ it can be made computable across four lifecycle layers with preliminary experiments showing that provenance is estimable and interveneable online before irreversible harm accumulates, and examining $\textit{who}$ bears responsibility through a concrete agentic incident. Explicit provenance is not a discretionary refinement but the necessary condition for responsible agentic AI, and no stakeholder across its ecosystem can afford to treat it as optional. 

---
# From Imitation to Interaction: Mastering Game of Schnapsen with Shallow Reinforcement Learning 

**Authors**: Ján Klačan, Sizhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17162)  

**Abstract**: This paper investigates whether shallow neural network agents can master the card game Schnapsen and challenge a strong search-based baseline, RdeepBot, which uses Monte Carlo sampling and lookahead search. Guided by a progressively more complex experimental design, we first evaluate a supervised learning agent (MLPBot) trained on replay data and then a reinforcement learning agent (RLBot) with the same shallow architecture trained through asynchronous Monte Carlo updates and experience replay. The results show that supervised imitation does not generalize well enough to defeat strong RdeepBot opponents, whereas reinforcement learning produces substantially stronger agents. In the setting that focuses on the depth parameter of RdeepBot, the best performance is achieved when the learned value function is combined with deeper lookahead during gameplay, allowing RLBot to achieve statistically significant higher winning rates against the strongest evaluated RdeepBot baseline. In the sample-based setting, the gains are more conditional: the strongest performance appears at a relatively lower training num_samples parameter rather than increasing uniformly with stronger sampling. 

---
# MADP: A Multi-Agent Pipeline for Sustainable Document Processing with Human-in-the-Loop 

**Authors**: Diego Gosmar, Giovanni Zenezini  

**Link**: [PDF](https://arxiv.org/pdf/2605.17159)  

**Abstract**: Document processing automation remains a critical challenge in enterprise environments, where traditional manual approaches are labor-intensive and error-prone. We present MADP, a multi-agent architecture that addresses the challenge of automating document processing in enterprise settings by combining deep learning-based classification and parsing with large language model extraction, while maintaining accuracy through selective human validation. Our system integrates five specialized agents--Classificator, Splitter, Parser, Extraction, and Validator--with a Human-in-the-Loop (HITL) mechanism and a novel Prompt Fine Tuning with Feedback Inheritance (PFTFI) approach. The operational analysis on a production use-case scenario of 100,000 invoices per year indicates a potential reduction of Full-Time Equivalent (FTE) requirements by approximately 70%. Production deployment on 955 real-world documents processed through January 2026 achieves a 97.0% full-pipeline automation rate, with only 3% requiring non-AI fallback. Ablation evaluation on a stratified 100-document subset (5 documents per each of 20 supplier/document-type categories) demonstrates that the full MADP configuration with Human-in-the-Loop supervision attains 98.5% document-level accuracy. Additionally, we present a comprehensive sustainability analysis showing that our hybrid AI+HITL approach reduces CO2 emissions by 69%, energy consumption by 69%, and water usage by 63% compared to traditional manual processing. Benchmark comparisons of multiple LLM backends (Granite-Docling, Mistral-Small, DeepSeek-OCR) provide practical insights for deployment in production environments. 

---
# Dynamics of collective creativity in AI art competitions 

**Authors**: Mason Youngblood, Jeff Nusz, Joel Simon  

**Link**: [PDF](https://arxiv.org/pdf/2605.17141)  

**Abstract**: Creativity is a fundamental aspect of how culture evolves, yet the mechanisms by which groups produce novelty are notoriously difficult to infer from the historical record. Iterated learning experiments have shown that cultural transmission reliably distorts artifacts toward the inductive biases of learners, but most of this work uses linear chains between human participants, leaving open how these dynamics play out in the networked, human-AI systems that increasingly shape cultural production. In this study, we leverage one such system, Artbreeder, which hosts daily "remix parties" where users iteratively build on each other's work from a single seed image, producing branching lineages of human-AI co-created images. We analyze a dataset of 130,882 images from 368 remix parties over 13 months and find that images become simpler and converge toward common thematic "attractors" (e.g., steampunk scenes, alien architecture). We also find that while more novel "parent" images produce more novel and complex "children" that attract more likes, users paradoxically prefer to remix images that are less novel and complex. Finally, larger remix parties produce more novelty at the cost of lower complexity. 

---
# Latent Heuristic Search: Continuous Optimization for Automated Algorithm Design 

**Authors**: Cheikh Ahmed, Mahdi Mostajabdaveh, Zirui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.17137)  

**Abstract**: The integration of Large Language Models (LLMs) into evolutionary frameworks has established a new paradigm for automated heuristic discovery. Despite their promise, these methods typically search in the discrete space of program syntax, relying on stochastic sampling to navigate a highly non-convex optimization landscape. This work proposes a continuous heuristic discovery framework that shifts optimization to a learned latent manifold. We employ an encoder to map discrete programs into continuous embeddings and train a differentiable surrogate model to predict performance, enabling gradient-based search. To regularize the optimization trajectory, an invertible normalizing flow maps these embeddings to a structured Gaussian prior, where we perform gradient ascent. The resulting optimized latent vectors are projected through a learned mapper into soft prompts, which condition a frozen LLM to synthesize novel executable heuristics. We evaluate the proposed method on the Traveling Salesman Problem (TSP), the Capacitated Vehicle Routing Problem (CVRP), the Knapsack Problem (KSP), and Online Bin Packing (OBP). Empirical results demonstrate that continuous latent-space optimization achieves performance competitive with state-of-the-art discrete evolutionary baselines while offering a complementary methodological alternative for automated algorithm design. The implementation code is available at \url{this https URL}. 

---
# F2IND-IT! -- Multimodal Fuzzy Fake Indian News Detection using Images and Text 

**Authors**: Kushal Trivedi, Murtuza Shaikh, Khushi Singh, Jeevaraj S.  

**Link**: [PDF](https://arxiv.org/pdf/2605.17115)  

**Abstract**: Biased manipulation of facts across regional and national media outlets complicates misinformation detection in diverse landscapes like India. This paper introduces a novel multimodal framework combining visual and textual modalities for enhanced fake news detection on Indian media. The architecture utilizes a ResNet-50 Convolutional Neural Network to extract visual features from news images, a DistilBERT encoder to obtain textual semantic embeddings, and an Adaptive Neuro-Fuzzy Inference System (ANFIS) to generate a fuzzy reliability score. A lightweight attention-based fusion module assigns learnable weights to each modality prior to classification. Evaluated on the IFND dataset, the proposed model is validated through an in-depth comparative analysis against previous research. Experimental results demonstrate superior performance across accuracy, precision, recall, and $F_1$-scores, confirming the efficacy of the architecture. 

---
# Capturing LLM Capabilities via Evidence-Calibrated Query Clustering 

**Authors**: Fangzhou Wu, Sandeep Silwal, Qiuyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17110)  

**Abstract**: Query clustering organizes queries into groups that reflect shared latent capability demands, enabling capability-aware LLM evaluation. Existing clustering methods, which primarily rely on semantic taxonomies or embeddings, often fail to capture such latent capability requirements due to a misalignment between surface-level semantics and actual model performance. We propose ECC, an algorithm that calibrates prior semantic embeddings using limited posterior model comparisons to bridge the gap between surface-level semantics and latent capability requirements. ECC characterizes each cluster through a capability profile parameterized by a Bradley-Terry model and uses trainable mixture weights to accommodate queries with mixed capability demands, jointly learning a flexible, capability-aware clustering structure that supports query-specific inference of LLM capabilities. Extensive quantitative and qualitative evaluations demonstrate that ECC significantly improves LLM capability ranking quality, outperforming human-labeled and embedding-based baselines by an average of 17.64 and 18.02 percentage points, respectively, and proves effective in downstream tasks such as query routing. 

---
# Scientific Logicality Enriched Methodology for LLM Reasoning: A Practice in Physics 

**Authors**: Zhaoxin Yu, Nan Xu, Kun Chen, Jiahao Zhao, Lei Wang, Wenji Mao  

**Link**: [PDF](https://arxiv.org/pdf/2605.17104)  

**Abstract**: With the continuous advancement of reasoning abilities in Large Language Models (LLMs), their application to scientific reasoning tasks has gained significant research attention. Current research primarily emphasizes boosting LLMs' performance on scientific QA benchmarks by training on larger, more comprehensive datasets with extended reasoning chains. However, these approaches neglect the essence of the scientific reasoning process -- logicality, which is the rational foundation to ensure the validity of reasoning steps leading to reliable conclusions. In this work, we make the first systematic investigation into the internal logicality underlying LLM scientific reasoning, and develop a scientific logicality-enriched methodology, including a set of assessment criteria and data sampling methods for logicality-guided training, to improve the logical faithfulness as well as task performance. Further, we take physics, characterized by its diverse logical structures and formalisms, as an exemplar discipline to practise the above methodology. For data construction, we extract scientific problems from academic literature and sample a high-quality dataset exhibiting strong logicality. Experiments based on three different backbone LLMs reveal that: 1) the training data we constructed can effectively improve the scientific logicality in LLM reasoning; and 2) the enriched scientific logicality plays a critical role in solving scientific problems. Code is available at \href{this https URL}{this https URL}. 

---
# RAGA: Reading-And-Graph-building-Agent for Autonomous Knowledge Graph Construction and Retrieval-Augmented Generation 

**Authors**: Chengrui Han, Zesheng Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.17072)  

**Abstract**: Existing LLM-driven knowledge graph (KG) construction methods predominantly employ stateless batch processing pipelines, exhibiting structural deficiencies in cross-chunk semantic relation capture, entity disambiguation, and construction process interpretability. These limitations undermine KG quality, retrieval precision, and deployment trust in high-stakes domains.
We propose RAGA (Reading And Graph-building Agent), an LLM-based autonomous KG construction and retrieval fusion framework. RAGA provides an atomic toolset supporting full KG lifecycle CRUD operations and embeds a Read-Search-Verify-Construct cognitive constraint into a ReAct tool loop. A KG-vector synchronization mechanism enables hybrid symbolic-vector retrieval, while evidence-anchored verification links every knowledge entry to its source text for auditable provenance.
Preliminary experiments on a subset of the QASPER scientific QA dataset indicate that RAGA's fusion retrieval outperforms zero-shot baselines, with KG integration providing measurable gains in both answer and evidence quality. The framework design and experimental baseline serve as a reference for agent-driven autonomous KG construction. 

---
# AnchorDiff: Topology-Aware Masked Diffusion with Confidence-based Rewriting for Radiology Report Generation 

**Authors**: Shiying Yu, Jielei Wang, Guoming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17071)  

**Abstract**: Radiology report generation (RRG) aims to automatically produce clinically accurate textual reports from medical images. Existing methods predominantly rely on autoregressive (AR) language models, whose causal dependency structure restricts generation to a unidirectional left-to-right process. This paradigm can induce sequence bias, where models tend to follow stereotypical token orders and high-frequency report templates rather than fully grounding generation in image-specific evidence. In this paper, we propose AnchorDiff, the first masked-diffusion framework for RRG that integrates knowledge-graph-derived clinical anchors into diffusion language modeling. By leveraging bidirectional context and iterative refinement, AnchorDiff mitigates the limitations of fixed-order autoregressive decoding. Specifically, we introduce a topology-aware training strategy that uses RadGraph-derived entity hierarchies to assign clinically important tokens differentiated masking protection and loss weights. We further design an inference-time rewriting strategy that detects unstable committed tokens through perturbation-based testing and selectively revises them during denoising. Extensive experiments on the MIMIC-CXR and MIMIC-RG4 benchmarks demonstrate that AnchorDiff achieves state-of-the-art (SOTA) performance, showing the effectiveness of clinically anchored masked diffusion for radiology report generation. 

---
# Towards Human-Level Book-Writing Capability 

**Authors**: Jan Zierstek, Matteo Batelic, Maya Medjad, Tim Schönenberger  

**Link**: [PDF](https://arxiv.org/pdf/2605.17064)  

**Abstract**: Large language models optimized for instruction following and agentic tasks remain poorly aligned with the requirements of high-quality creative writing. Fiction frequently depends on behaviors that assistant-tuned models are explicitly trained to avoid, particularly deception, moral ambiguity, and unreliable narration. As a result, generated stories often appear structurally correct while remaining stylistically generic, overly explanatory, or weakly grounded in human literary behavior. We present a dataset construction and training framework for book-scale creative writing that reframes supervised fine-tuning as a prompt-to-book generation task grounded in human-authored fiction. Starting from public-domain novels, we derive a multi-resolution planning scaffold by summarizing each book at progressively finer levels, from a high-level premise to chapter- and scene-level structure. We then invert this hierarchy during training: the model learns to expand a prompt into increasingly detailed plans and finally into the original human-authored book text. This formulation preserves human prose as the final supervised target while using intermediate summaries to make book-scale generation learnable. We train a long-context language model on these prompt-to-book trajectories and study whether this objective shifts generation away from assistant-style prose and toward human literary writing. 

---
# PersonaArena: Dynamic Simulation for Evaluating and Enhancing Persona-Level Role-Playing in Large Language Models 

**Authors**: Wenlong Shi, Jianxun Lian, Mingqi Wu, Haiming Qin, Mingyang Zhou, Xing Xie, Naipeng Chao, Hao Liao  

**Link**: [PDF](https://arxiv.org/pdf/2605.17044)  

**Abstract**: Large language models (LLMs) increasingly serve as interactive social agents, yet their ability to maintain coherent and authentic persona-level role-playing remains limited, particularly in realistic social scenarios. Existing research predominantly focuses on character-level settings and relies on static evaluation formats, failing to capture the complexity of everyday social interactions. In this work, we present PersonaArena, a dynamic simulation framework for evaluating and improving persona-level role-playing in LLMs. PersonaArena leverages a large, filtered corpus of user-generated social content to construct a nuanced persona bank, and elicits multi-turn, context-rich interactions within simulated social environments. Our framework features a multi-agent debating judge for holistic and unbiased assessment. Through extensive experiments, we demonstrate that PersonaArena enables rigorous evaluation and enhancement of LLMs' role-playing capabilities, advancing the development of more authentic and socially adept AI agents. 

---
# Evidential Information Fusion on Possibilistic Structure 

**Authors**: Qianli Zhou, Ye Cui, Zhen Li, Witold Pedrycz, Yong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.17038)  

**Abstract**: Dempster's rule is a fundamental tool for combining belief functions from distinct and reliable sources. However, its intersection-based semantics imposes strong structural restrictions, which limits its flexibility in handling complex source states and diverse information fusion scenarios. To overcome this limitation, we propose a reversible transformation, derived from the isopignistic principle, between belief functions and a possibilistic structure defined on the power set. In this transformation, the relationships among subsets are explicitly characterized by a belief evolution network, which provides a more flexible representation of evidential information beyond the conventional mass function structure. On this basis, we further introduce the triangular norm family to develop a general and adaptive evidential information fusion framework. Unlike fusion methods rooted in Dempster semantics, the proposed framework supports more flexible combination behaviors and exhibits advantages in non-distinct source fusion, conflict management, parametric combination design, and heterogeneous information fusion. 

---
# Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management 

**Authors**: Carol Xuan Long, David Simchi-Levi, Feng Zhu, Huangyuan Su, Andre P. Calmon, Flavio P. Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2605.17036)  

**Abstract**: This paper studies autonomous generative AI agents in multi-echelon supply chains using the MIT Beer Game. We identify four inference-time levers that shape performance: model selection, policies and guardrails, centralized data sharing, and prompt engineering. Model capability is the dominant factor: an out-of-the-box reasoning model exceeds human-level performance, and optimized reasoning models reduce costs by up to 67% relative to human teams. However, strong average performance masks substantial reliability risks. We introduce the agent bullwhip effect, the amplification of decision unreliability across echelons, manifesting along two dimensions: decision variance increases both across facilities at the same point in time and within the same facility across time. We develop a mathematical framework showing that this phenomenon is inherent to multi-agent systems that involve coordination and information delays, and we demonstrate that repeated sampling fails to meaningfully reduce it. To address this limitation, we propose a Group Relative Policy Optimization (GRPO)-based reinforcement-learning post-training framework that trains a shared base LLM using system-level supply-chain rewards. GRPO post-training substantially reduces tail events, curtails agent bullwhip, and improves the reliability of autonomous supply-chain agents. 

---
# A Conflict-aware Evidential Framework for Reliable Sleep Stage Classification 

**Authors**: Yunzhi Tian, Dekui Wang, Qirong Bu, Wei Zhou, Xingxing Hao, Jun Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.17021)  

**Abstract**: Multi-view learning has been widely applied for sleep stage classification using multi-modal data. However, existing methods typically assume that different modalities are well-aligned, which is often unattainable in real-world scenarios, thereby compromising the reliability of the staging results. In this paper, we propose ConfSleepNet, a conflict-aware evidential framework that dynamically resolves inter-view conflicts. The framework consists of multi-view evidence extraction and conflict-aware aggregation. In the first phase, it learns category-related evidence from different modalities, which represents the degree of support for individual sleep stages. Considering the inherent characteristics of varying modalities, we propose hybrid category structures for different modalities to promote more reasonable evidence learning. In the second phase, view-specific opinions, including prediction results and uncertainty, are constructed from the learned evidence. Notably, we propose a novel conflict-aware aggregation method that integrates these view-specific opinions into a reliable joint decision. This mechanism can effectively resolve conflicts among opinions and synthesize them into a reliable joint decision. Both theoretical analysis and experimental results demonstrate the effectiveness of ConfSleepNet in sleep staging tasks. The code is available at this https URL. 

---
# Brain Vascular Age Prediction Using Cerebral Blood Flow Velocity and Machine Learning Algorithms 

**Authors**: Anni Zhao, Alex Bateh, Tyler Baldridge, Sandra Billinger, Xiao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16969)  

**Abstract**: Defining vascular age in terms of physiological function has become one focal point of the extensive studies to categorize and track chronological age. Transcranial Doppler (TCD) is a method by which cerebral blood flow velocity is measured along the major arteries feeding the human brain. This study aims to use features extracted from TCD to estimate chronological age and assess accelerated aging in subjects with various brain diseases. We predict subjects with various brain diseases to present with accelerated cerebrovascular aging when tested on various regression models trained by healthy subjects. 168 healthy subjects and 277 diseased subjects with bilateral TCD recordings of the middle cerebral artery were analyzed using the Morphological Analysis and Clustering of Intracranial Pressure (MOCAIP) algorithm. MOCAIP-generated features and heart rate variability features were used as input features for regression models to predict the brain vascular age. 66 subjects with acute stroke, 27 subjects with post stroke, 26 subjects with Alzheimer's disease, 23 subjects with mild cognitive impairment, and 135 established subjects were tested against the machine learning model to assess for accelerated cerebrovascular age. The trained model, on average, predicted healthy subjects' cerebrovascular age to be 3.69 years above their chronological age. Subjects with different disease conditions exhibited varying levels of age acceleration. The differences in healthy and diseased subjects' performances suggest that features generated using TCD may be relevant when evaluating accelerated cerebrovascular aging. Moreover, imbalanced datasets have been observed to affect the performance of machine-learning-based brain age prediction models. 

---
# Harnessing AI for Inverse Partial Differential Equation Problems: Past, Present, and Prospects 

**Authors**: Zhentao Tan, Yuze Hao, Boyi Zou, Mingsheng Long, Yi Yang, Gang Bao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16966)  

**Abstract**: Solving inverse partial differential equation (PDE) problems is a fundamental topic in scientific research due to its broad significance across a wide range of real-world applications. Inverse PDE problems arise across medical imaging, geophysics, materials science, and aerodynamics, where the goal is to infer hidden causes, design structures, or control physical states. In this paper, we provide a comprehensive review of recent advances in solving inverse PDE problems using artificial intelligence (AI). We first introduce the basic formulation, key challenges, and traditional numerical foundations of inverse PDE problems, and then organize it into three major categories: inverse problems, inverse design, and control problems. For each category, we further present a methodological paradigms, and review representative state-of-the-art approaches from recent years. We then summarize representative applications across scientific and industrial domains, including mechanical systems, aerodynamic problems, thermal systems, full-waveform inversion, system identification, and medical imaging. Finally, we discuss open challenges and future prospects, such as physics-informed architectures, limited real-world data, uncertainty quantification, and inverse foundation models. This survey aims to provide the first unified and systematic perspective on AI for inverse PDE problems, demonstrating how modern learning-based methods are reshaping inverse problems, inverse design, and control problems in PDE-governed systems. 

---
# How do Humans Process AI-generated Hallucination Contents: a Neuroimaging Study 

**Authors**: Shuqi Zhu, Yi Zhong, Ziyi Ye, Bangde Du, Yujia Zhou, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16953)  

**Abstract**: While AI-generated hallucinations pose considerable risks, the underlying cognitive mechanisms by which humans can successfully recognize or be misled by these hallucinations remain unclear. To address this problem, this paper explores humans' neural dynamics to characterize how the brain processes hallucinated content. We record EEG signals from 27 participants while they are performing a verification task to judge the correctness of image descriptions generated by a multi-modal large language model (MLLM). Based on an averaged event-related potential (ERP) study, we reveal that multiple cognitive processes, e.g., semantic integration, inferential processing, memory retrieval, and cognitive load, exhibit distinct patterns when humans process hallucinated versus non-hallucinated content. Notably, neural responses to hallucinations that were misjudged versus correctly judged by human participants showed significant differences. This indicates that misjudged AI-generated hallucinations failed to trigger the standard neurocognitive fact verification pathway. 

---
# From Static Risk to Dynamic Trajectories: Toward World-Model-Inspired Clinical Prediction 

**Authors**: Pujun Feng, Xiaoyu Guo, Seyed Ehsan Saffari, Min Hun Lee, Siew-Kei Lam, Erik Cambria, Xibin Sun, Yangtao Zhou, Tong Yang, Xiaoyu Zhang, Tao Tan, Yue Sun, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.16927)  

**Abstract**: Clinical decision-making is a feedback system where risk estimates influence treatment, which in turn changes disease trajectories, and both shape clinicians' measurement practices. Static prediction often fails clinically: models trained on observational care logs conflate disease biology with clinician behavior, particularly under treatment confounder feedback and irregular or informative observation. This Review focuses on intervention-aware disease trajectory modeling in clinical AI--methods estimating patient-specific longitudinal disease evolution and assessing trajectory changes under alternative treatments. We organize the field around six linked components: three decision tasks (factual forecasting, counterfactual estimation, policy evaluation) and three data-generating mechanisms (disease evolution, treatment assignment, observation process) that determine identifiability. We present the first unified framework bridging forecasting, counterfactual trajectories, and policy evaluation across discrete/continuous time, explicitly addressing treatment assignment, time-varying confounding, and observation bias. We synthesize key method families (multistate/joint models, temporal point-process, deep sequence architectures, longitudinal causal inference), map them to relevant components, and align evaluation with claim strength via overlap diagnostics, uncertainty quantification, off-policy robustness, and target-trial validation. This synthesis advances benchmark prediction to decision-grade clinical evidence, enabling treatment-sensitive individualized futures, pre-deployment policy stress-testing, and safer closed-loop learning health systems that adapt/abstain when evidence is insufficient. 

---
# TOBench: A Task-Oriented Omni-Modal Benchmark for Real-World Tool-Using Agents 

**Authors**: Zhiqiang Liu, Wenhui Dong, Yilang Tan, Yuwen Qu, Haochen Yin, Chenyang Si  

**Link**: [PDF](https://arxiv.org/pdf/2605.16909)  

**Abstract**: Tool-using agents are increasingly expected to operate across realistic professional workflows, where they must interpret multimodal inputs, coordinate external tools, inspect intermediate artifacts, and revise their actions before producing a final result. Existing benchmarks, however, often evaluate tool use, computer use, and multimodal reasoning in isolation, leaving a gap between benchmark settings and end-to-end omni-modal tool use in the real world. To address this gap, we introduce MM-ToolBench, a benchmark and evaluation harness for task-oriented omni-modal tool use. MM-ToolBench contains 100 executable tasks from two macro task families, Customer Service and Intelligent Creation, covering 20 subcategory slices and supported by 27 MCP servers with 324 tools. The central design of MM-ToolBench is closed-loop multimodal verification: agents must execute tools, inspect rendered or transformed artifacts, and self-correct when outputs fail task-specific requirements. To make such evaluation scalable and verifiable, MM-ToolBench couples MCP-based execution with task-specific grounded evaluators and a semi-automated construction pipeline for scenario discovery, task instantiation, evaluator synthesis, and human audit. Experiments on 15 contemporary agentic models show that MM-ToolBench remains highly challenging: Claude Opus 4.6, commonly regarded as one of the strongest coding-agent models, achieves only 32.0% task success, far below the 94.0% human benchmark. We envision MM-ToolBench as a practical foundation for evaluating and advancing next-generation omni-modal tool-using agents through closed-loop multimodal verification. 

---
# NGM: A Plug-and-Play Training-Free Memory Module for LLMs 

**Authors**: Yuwen Qu, Wenhui Dong, Chenyang Si, Caifeng Shan  

**Link**: [PDF](https://arxiv.org/pdf/2605.16893)  

**Abstract**: Recent studies introduce conditional memory modules that decouple knowledge storage from neural computation, enabling more direct knowledge access. Compared to MoE, which relies on dynamic computation paths, explicit lookup provides a more efficient knowledge retrieval mechanism. However, these approaches still depend on learned memory embeddings, requiring additional training and limiting flexibility. To address this, we propose N-gram Memory (NGM), a training-free, plug-and-play module composed of a Causal N-Gram Encoder and a Cosine-Gated Memory Injector. The Causal N-Gram Encoder directly averages the pretrained token embeddings of the backbone model to construct N-gram representations, thereby eliminating the need to train separate N-gram embeddings from scratch. This design requires neither an additional memory table nor a retrieval pipeline. The Cosine-Gated Memory Injector then uses a non-parametric cosine gate with ReLU to modulate the retrieved embeddings into the contextual representations. We evaluate NGM on the Qwen3 series from 0.6B to 14B across eight benchmarks. NGM improves average performance by 0.5 to 1.2 points, with particularly clear gains on code generation and knowledge-intensive tasks (e.g., +3.0 on LiveCodeBench and +3.03 on GPQA for Qwen3-14B). Moreover, NGM also improves performance in multimodal benchmarks (e.g., MMStar +1.53 on Qwen3-VL-2B). 

---
# Virtual Nodes Guided Dynamic Graph Neural Network for Brain Tumor Segmentation with Missing Modalities 

**Authors**: Sha Tao, Jiao Pan, Yu Guo, Chao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16880)  

**Abstract**: Multimodal magnetic resonance imaging (MRI) is crucial for brain tumor segmentation, with many methods leveraging its four key modalities to capture complementary information for effective sub-region analysis. However, the absence of several modalities is very common in practice, leading to severe performance degradation in existing full-modality segmentation methods. Limited by the structured data model, recent works often adopt a multi-stage training strategy for full-modality and missing-modality scenarios, which increases training costs and inadequately addresses the interference of miss. In this work, we propose a graph-based one-stage framework for robust brain tumor segmentation with missing modalities. Specifically, we introduce modality-specific virtual nodes that serve as supplementary information sources to compensate for missing modalities. To enhance model robustness against arbitrary modality combinations, we leverage the inherent flexibility of graph networks to devise a dynamic connection strategy. This mechanism dynamically adjusts the adjacency matrix based on modality availability, preserving beneficial information flow while mitigating interference effects caused by missing modalities. Furthermore, we enhance the graph network through heterogeneous weight matrices, enhancing its adaptability to multimodal scenarios. Extensive experiments on the BRATS-2018 and BRATS-2020 datasets demonstrate that our method outperforms the state-of-the-art methods on almost all subsets of incomplete modalities. 

---
# Reasoning Can Be Restored by Correcting a Few Decision Tokens 

**Authors**: Changshuo Shen, Leheng Sheng, Yuxin Chen, An Zhang, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16874)  

**Abstract**: Large reasoning models (LRMs) substantially outperform their base LLM counterparts on challenging reasoning benchmarks, yet it remains poorly understood where base models go wrong during token-by-token generation and how to narrow this gap efficiently. We study the base-reasoning gap through quantifying token-level distributional disagreement between a base model and a stronger reasoning model using likelihood-based divergences. Across benchmarks, we find that the reasoning advantage is highly sparse and concentrates on a small set of early, planning-related decision tokens. For instance, on Qwen3-0.6B, only ~8% of generated tokens account for the salient disagreement, and these tokens concentrate early in the response, are strongly enriched in planning-related decisions (17x), and coincide with high base-model uncertainty -- suggesting that base models fail mainly at early planning points that steer the subsequent reasoning trajectory. Building on these findings, we propose disagreement-guided token intervention, a simple inference-time delegation scheme that performs a one-token takeover by the reasoning model only at high-disagreement positions and immediately switches back to the base model. With a small intervention budget, this sparse delegation substantially recovers and can even surpass the performance of a same-size reasoning model on challenging reasoning tasks. Code is available at this https URL. 

---
# Learning to Learn from Multimodal Experience 

**Authors**: Xingyu Sui, Weixiang Zhao, Yongxin Tang, Yanyan Zhao, Yang Wu, Dandan Tu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2605.16857)  

**Abstract**: Experience-driven learning has emerged as a promising paradigm for enabling agents to improve from interaction trajectories by accumulating and reusing past experience. However, existing approaches are predominantly developed in textual settings and rely on manually designed memory schemas, limiting their applicability to multimodal environments. In real-world scenarios, experience is inherently multimodal, involving heterogeneous signals across perception, reasoning, and action, which makes effective memory design significantly more challenging. In particular, the optimal way to structure and utilize multimodal experience is highly task-dependent and evolves over time, rendering fixed memory designs insufficient. In this work, we propose a new paradigm, learning to learn from multimodal experience, which shifts memory design from a predefined component to an adaptive and learnable process. Our framework enables agents to dynamically construct, organize, and utilize memory based on task requirements and interaction history, effectively learning how to structure experience for improved performance. Experiments demonstrate that adaptive memory design substantially enhances agent performance and generalization across multimodal tasks, highlighting the critical role of learning memory mechanisms in experience-driven learning. 

---
# Artificial Adaptive Intelligence: The Missing Stage Between Narrow and General Intelligence 

**Authors**: Boris Kriuk  

**Link**: [PDF](https://arxiv.org/pdf/2605.16844)  

**Abstract**: Between the narrow systems we deploy and the general intelligence we speculate about lies an entire regime of machine behavior that has never received its own name. This monograph argues that this regime is not empty: it is where meta-learning, neural architecture search, AutoML, continual learning, evolutionary computation, and physics-informed modeling have quietly converged on a common principle, namely the steady removal of the human from the loop of parameter specification. We name this regime Artificial Adaptive Intelligence (AAI) and define it operationally: a system exhibits AAI to the extent that it requires no human-specified tunable hyperparameters while maintaining competitive performance across a diverse distribution of tasks. To make the definition quantitative, we introduce an adaptivity index that measures progress along an axis orthogonal to scale, combining the fraction of hyperparameters absorbed by the system with the performance ratio against a task-specialized baseline. We develop the principle of parametric minimality and ground it in the minimum description length framework, showing that the appropriate hyperparameter count is data-determined rather than designer-determined. We then organize the field around three pathways to minimality: data- and task-aware configuration, structural and evolutionary morphing, and in-training self-adaptation. We analyze their stability, convergence, and governance implications, and illustrate them through case studies spanning aerospace design, financial regime detection, turbulence modeling, ecological dynamics, and vision-language systems. The thesis is that the path from ANI to AGI passes through AAI, and that naming this stage changes what we measure, what we build, and what we call a success. 

---
# Sketch Then Paint: Hierarchical Reinforcement Learning for Diffusion Multi-Modal Large Language Models 

**Authors**: Siqi Luo, Jianghan Shen, Yi Xin, Huayu Zheng, Haoxing Chen, Yan Tai, Yue Li, Junjun He, Yihao Liu, Guangtao Zhai, Yuewen Cao, Xiaohong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16842)  

**Abstract**: Diffusion Multi-Modal Large Language Models (dMLLMs) are powerful for image generation, but optimizing them through reinforcement learning (RL) remains a major challenge. One primary difficulty is that a single image can be generated through many different unmasking sequences, which makes calculating importance ratios often intractable. Additionally, existing methods tend to ignore the hierarchical generation process of dMLLMs, where early tokens define the global layout and later tokens focus on local details. By assigning uniform rewards to all tokens, these current methods fail to reflect the actual contribution of each token to the final image. To address these issues, we propose Hierarchical Token GRPO (HT-GRPO), which integrates this hierarchy directly into the policy optimization process. Our approach features a Sketch-Then-Paint training scheme that organizes updates into three distinct stages: global, structure, and refinement. We also use a prompt-conditioned estimator to calculate importance ratios starting from a fully masked state. Furthermore, we introduce a Hierarchical Credit Assignment mechanism that prioritizes key structural tokens to ensure accurate reward propagation. Experiments using two popular dMLLM backbones, MMaDA and Lumina-DiMOO, demonstrate that HT-GRPO achieves substantial gains on the GenEval and DPG benchmarks. Evaluations across six additional metrics confirm significant improvements in image quality, aesthetics, and human preference. 

---
# Voices in the Loop: Mapping Participatory AI 

**Authors**: Rashid Mushkani  

**Link**: [PDF](https://arxiv.org/pdf/2605.16827)  

**Abstract**: Participatory approaches to artificial intelligence are increasingly documented across public, civic, and humanitarian settings, but evidence about how participation is organized remains fragmented. This paper reports on the construction of an open repository and interactive atlas of participatory AI initiatives, using records harmonized from Maga~na and Shilton's Trustworthy AI corpus, and additional audited cases from research and practice. We contribute three elements. First, we specify a reproducible protocol for discovery, vetting, harmonization, geocoding, provenance tracking, and release-based publication of participatory AI records. Second, we report corpus-level patterns in geography, participation tiers, lifecycle loci, organizational form, verification status, and remaining documentation gaps. Documented initiatives remain concentrated in a small number of countries, while participation is most often coded at problem formulation, evaluation, and governance rather than model development or training. Third, we show how the atlas operationalizes a design and governance framework for participatory-by-default AI infrastructures through versioned releases, record-linked issue and annotation channels, schema feedback workflows, and redaction or restricted-disclosure requests. The atlas is intended to support comparative research, policy learning, and community scrutiny through a living inventory that can be updated, contested, and reused. 

---
# Multi-Paradigm Agent Interaction in Practice:A Systematic Analysis of Generator-Evaluator, ReAct Loop,and Adversarial Evaluation in the buddyMe Framework 

**Authors**: Xiaohua Wang, Chao Han, Kai Yu, XiaoLiang Xu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16821)  

**Abstract**: The rapid evolution of Large Language Model (LLM) agents has produced diverse interaction paradigms, yet few production systems integrate multiple paradigms within a unified architecture. This paper presents a systematic analysis of three principal agent interaction paradigms, including Multi-Agent Orchestration (Generator-Evaluator), ReAct Tool-Use Loops, and Memory-Augmented Interaction, as implemented in buddyMe, an open-source multi-model agent programming framework. We formalize a five-stage processing pipeline: Requirement Pre-Review -> Task Decomposition -> ReAct Execution -> Real-Execution Verification -> Adversarial Evaluation Discussion, and establish a six-dimensional evaluation schema with weighted scoring. Through four empirical case studies drawn from real-world deployment logs covering museum guide generation, scheduled weather tasks, and comprehensive tour planning, we draw three key conclusions. First, Generator-Evaluator pre-review detects requirement omissions in 20 percent of complex tasks, with 80 percent tasks passing initial inspection. Second, the ReAct loop ensures stable subtask execution but leads to around 30 percent redundant tool invocations. Third, adversarial Evaluator-Defender discussions reach consensus within 2-3 rounds for nearly 70 percent of scenarios, functioning mainly for content refinement rather than logical reversal. We additionally provide three Mermaid-based architectural diagrams and conduct cross-paradigm comparisons with CrewAI, AutoGen, LangGraph, MemGPT and A-Mem across six system dimensions. The research outcomes offer practical design guidelines for constructing stable and reliable multi-paradigm agent systems. 

---
# NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning 

**Authors**: Haoran Lu, Luyang Fang, Wenxuan Zhong, Ping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.16757)  

**Abstract**: Multi-agent language systems are often built as hand-designed workflows, where agents are assigned semantic roles and communication protocols are specified in advance. We propose NeuroMAS, a method that first treats a multi-agent language system as a trainable and scalable neural-network-like architecture with LLM agents as nodes and intermediate textual signals as edges. In NeuroMAS, agent nodes are role-free but structure-aware: the topology only determines how information can flow in general, while reinforcement learning training determines how nodes communicate, specialize, and coordinate. This formulation shifts multi-agent design from workflow engineering toward architecture design, where depth, width, connectivity, and growth protocol become scalable sources of capability. Further, we provide a theoretical perspective showing why such modular textual computation is more parameter-efficient when tasks admit hierarchical decompositions. Experiments show that NeuroMAS improves significantly over both inference-time and trained multi-agent baselines. We further find that organizational scaling is path-dependent: larger systems can be challenging to train from scratch, but become feasible when grown progressively from smaller trained systems. These results suggest that learned neural multi-agent systems are a promising scaling axis for LLMs. 

---
# State Contamination in Memory-Augmented LLM Agents 

**Authors**: Yian Wang, Agam Goyal, Yuen Chen, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2605.16746)  

**Abstract**: LLM agents increasingly rely on persistent state, including transcripts, summaries, retrieved context, and memory buffers, to support long-horizon interaction. This makes safety depend not only on individual model outputs, but also on what an agent stores and later reuses. We study a failure mode we call memory laundering: toxic or adversarial context can be compressed into memory summaries that no longer appear toxic under standard detectors, while still preserving hostile framing or conflict structure that influences future generations. Using paired counterfactual multi-agent rollouts, we show that toxic-origin memory summaries can remain below common toxicity thresholds while nevertheless increasing downstream toxicity relative to matched neutral baselines. To measure this hidden influence, we introduce the sub-threshold propagation gap (SPG), which quantifies downstream behavioral differences conditioned on memory states that a deployed monitor would classify as safe. Our experiments show that toxicity propagates through distinct state channels: raw transcript reuse drives overt downstream toxicity, while compressed memory carries hidden sub-threshold influence. We further find that mitigation depends critically on intervention placement. Sanitizing toxic state before summarization substantially reduces the hidden propagation gap, whereas cleaning only the completed summary can leave laundered influence intact. These results suggest that safety in memory-augmented agents should be treated as a state-control problem over evolving context, with sanitization applied before unsafe information is compressed into persistent memory. 

---
# Body-Grounded Perspective Formation and Conative Attunement in Artificial Agents 

**Authors**: Hongju Pae  

**Link**: [PDF](https://arxiv.org/pdf/2605.16728)  

**Abstract**: This paper proposes a minimal architecture for body-grounded perspective formation in artificial agents. Extending prior work, the model introduces an interoceptive viability signal, a Fisher-style metric over fused exteroceptive-interoceptive states, and a conative alignment mechanism linking bodily tendency to action readiness. In a reward-free gridworld, conation converts learned bodily tendency into stable body-directed behavior, while body-to-perspective routing allows bodily perturbations to leave a recoverable geometric residue in the perspective latent. This study shows how minimal structural conditions for artificial subjectivity can be operationalized in the phenomenological sense, through the embodied organization of how a world is given to an agent. 

---
# PopuLoRA: Co-Evolving LLM Populations for Reasoning Self-Play 

**Authors**: Roger Creus Castanyer, Geoffrey Bradway, Lorenz Wolf, Maxwill Lin, Augustine N. Mavor-Parker, Matthew James Sargent  

**Link**: [PDF](https://arxiv.org/pdf/2605.16727)  

**Abstract**: We introduce PopuLoRA, a population-based asymmetric self-play framework for reinforcement learning with verifiable rewards (RLVR) post-training of LLMs. Teachers and students are specialised LoRA adapters on a shared frozen base: teachers propose problems, matched students solve them under a programmatic verifier, and cross-evaluation between sub-populations replaces the self-calibration that limits single-agent self-play. A family of LoRA weight-space evolution operators (mutations and crossovers that produce same-rank population members in seconds) serves as the replacement step of a population-based training loop at 7B scale. We instantiate PopuLoRA on top of Absolute Zero Reasoner and compare it against a per-adapter compute-matched single-agent baseline. Where the single agent self-calibrates to generating easy problems it can reliably solve, the population enters a co-evolutionary arms race: teachers produce increasingly complex problems, student solve rates oscillate, and problem-space coverage keeps expanding throughout training. Despite lower training-time reward, the population mean outperforms the baseline on three code benchmarks (HumanEval+, MBPP+, LiveCodeBench) and seven math benchmarks (AIME 24/25, AMC 23, MATH-500, Minerva, GSM8K, OlympiadBench), and even the weakest member of the population beats the baseline on aggregate. 

---
# A Global-Local Graph Attention Network for Traffic Forecasting 

**Authors**: Tianchi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16726)  

**Abstract**: Traffic forecasting is a significant part of intelligent transportation systems. One of the critical challenges of traffic forecasting is to find spatio-temporal correlations. In recent years, graph convolutional networks and graph attention networks have replaced traditional statistical models to predict future traffic. However, it is complicated for both of them to allow vertices to have far different characters. To address this, we propose the Global-Local Graph Attention Network (GLGAT) with pairwise encoding and the event-based adjacency matrix. The GLGAT allows vertices to have a global attention matrix set for the whole graph and assigns local attention matrix sets to each vertex. Experiments on two real-world traffic datasets show that GLGAT can effectively capture spatio-temporal correlations and has competitive performance against other state-of-the-art baselines. 

---
# Baba in Wonderland: Online Self-Supervised Dynamics Discovery for Executable World Models 

**Authors**: SeungWon Seo, DongHeun Han, SeongRae Noh, HyeongYeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16725)  

**Abstract**: Executable world models can be read, edited, executed, and reused for planning, but only if the program captures the environment's transition law rather than semantic shortcuts in its surface vocabulary. We study online executable world-model learning under prior misalignment, where an agent must induce state-dependent dynamics from interaction evidence alone, without rule descriptions, reward signals, or trustworthy lexical priors. We introduce Alice, a closed-loop system that treats failed candidate updates as structural signal: when a candidate explains a new transition but loses previously explained ones, the preservation conflict reveals dynamics that the current program had conflated. Alice refines these conflicts into hypothesis classes that both provide compact, class-stratified preservation counterexamples for update and guide frontier exploration toward transitions that are novel and underrepresented with respect to the current program. We evaluate Alice on Baba in Wonderland, a prior-misaligned variant of Baba Is You that preserves simulator dynamics while replacing semantically meaningful rule-property labels with unrelated words. Experiments show that Alice substantially improves executable world-model learning under prior misalignment, and ablations show that both class refinement and class-aware exploration contribute. 

---
# GRID: Graph Representation of Intelligence Data for Security Text Knowledge Graph Construction 

**Authors**: Liangyi Huang, Zichen Liu, Fei Shao, Shang Ma, Mengshi Zhang, Zihao Chen, Yanfang Ye, Xusheng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16714)  

**Abstract**: Security knowledge graphs can provide computable external memory for security agents, but constructing them from long-form cyber threat intelligence (CTI) remains difficult: LLMs often lack grounded security-domain knowledge, and end-to-end document-to-graph training is hard to supervise with cheap, stable rewards. We present GRID (Graph Representation of Intelligence Data), an end-to-end framework for security text knowledge graph construction. GRID first builds security-domain supervision from CTI articles by creating traceable article-graph alignments through graph extraction and knowledge-graph-conditioned text revision. It then turns document-to-graph learning into a scripted task bank combining four-option multi-select questions with triple-level regex matching targets, yielding more stable task-specific rewards than repeatedly scoring full graph outputs with an LLM judge. Using this supervision pipeline, we train two Qwen3-4B-Instruct-2507-based 4B extractors: a primary Task-bank Reward model and a secondary End2End Reward model with LLM-as-judge precision/recall rewards. On 249 CTI articles from GRID, CASIE, CTINexus, MalKG, and SecureNLP, the Task-bank Reward model with the ontology-guided GRID extraction pipeline reaches 84.62% source-averaged precision, 64.91% source-averaged recall, and 68.53% Avg F1, achieving the best source-averaged recall and near-top Avg F1 with lower token usage and deployment cost. The End2End Reward model reaches 76.91% precision, 53.85% recall, and 58.06% Avg F1. Further analyses show that task-bank rewards can be built once offline and reused across later post-training runs, outperforming online End2End LLM-as-judge reward and weaker alternatives such as Choice-only Reward and End2End SFT without RL. 

---
# Recall Isn't Enough: Bounding Commitments in Personalized Language Systems 

**Authors**: Rui Tang, Yichi Zhang, Xi Chen, Chen Dong, Youwei Yang, Yumeng Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.16712)  

**Abstract**: Long-context and memory systems usually treat personalization as a recall problem. In practice, many failures occur later, when a system commits: it turns noisy hints into hard constraints, drops rare witnesses, forgets downstream obligations, or answers despite infeasibility. We introduce Contract-Bounded Evidence Activation (CBEA) with Lexicographic Commitment Validation (LCV). CBEA activates a bounded evidence set using typed coverage, tail witnesses, and consequence debt; LCV validates structured commitments before prose and routes infeasible states to repair, abstention, or recontract. Across 360 fixtures and three generation backends, CBEA+LCV reaches zero failures within validator scope at 0.49-0.60 availability over attempted runs. Raw and long-context baselines with the same LCV gate reach zero only at 0.003-0.092. A shadow oracle diagnostic marks the limit: CBEA+LCV recalls 0.012 of uncompiled visible facts, while raw recalls 0.53. The result is a bounded operating point: explicit commitment control and 74-75% lower median input payload, not universal memory dominance. 

---
# Enhancing Metacognitive AI: Knowledge-Graph Population with Graph-Theoretic LLM Enrichment 

**Authors**: Deniz Askin, Gal Hadar, Brendan Conway-Smith  

**Link**: [PDF](https://arxiv.org/pdf/2605.16676)  

**Abstract**: Metacognition-the ability to monitor one's own knowledge state, spot gaps, and autonomously fill them--remains largely absent from modern AI. Here, we present MetaKGEnrich, a fully automated pipeline that endows large language model (LLM) applications with self-directed knowledge repair. The system (i) builds knowledge graphs from a seed query, (ii) detects sparse regions via seven graph metrics, (iii) has GPT-4o generate targeted questions, (iv) retrieves web evidence with Tavily and ingests it into Neo4j, and (v) re-answers the query with GraphRAG for GPT-4 to evaluate improvement. Tested on 30 queries from each of three widely-used datasets: Google Research Natural Questions, MS MARCO, and Hot-potQA. MetaKGEnrich improved answer quality in 80% of HotpotQA questions, 87% of Google Research Natural Questions and 83% of MS MARCO questions, while preserving well-supported regions. This proof of concept demonstrates how topological self-diagnosis plus targeted retrieval can advance AI toward humanlike metacognitive learning. 

---
# LinAlg-Bench: A Forensic Benchmark Revealing Structural Failure Modes in LLM Mathematical Reasoning 

**Authors**: Shradha Agarwal, Deepak Rajbhar, Tariq J  

**Link**: [PDF](https://arxiv.org/pdf/2605.16675)  

**Abstract**: We introduce LinAlg-Bench, a diagnostic benchmark evaluating 10 frontier large language models on structured linear algebra computation across a strict dimensional gradient of 3x3, 4x4, and 5x5 matrices. Spanning 9 task types and 660 SymPy-certified problems, the benchmark exhaustively evaluates 6,600 model outputs. Beyond binary accuracy, LinAlg-Bench introduces a three-stage automated forensic pipeline classifying 1,156 failures into ten primary error tags with fine-grained subtypes, revealing that LLM mathematical failure is not random but structurally constrained by algorithm type and matrix dimension. Our central finding is a sharp behavioral threshold at 4x4 scale: below it, models fail through execution errors -- sign tracking failures, arithmetic drift, and parity errors; above it, failure transitions to computational abandonment, with models fabricating responses through tool roleplay, constraint-consistent confabulation, and structured hallucination rather than attempting computation. This fabrication-to-abandonment transition is near-universal across all model tiers and architectures, suggesting a working memory limit rather than a knowledge gap, supported by three scale-emergent error types absent at 3x3 but present at 4x4 and 5x5. We further show that solution strategy rigidity is a near-perfect predictor of 5x5 determinant accuracy, document constraint-aware confabulation as a novel structured hallucination failure mode, and release all data, model outputs, error labels, and judge pipeline publicly. 

---
# Sustainable Intelligence for the Wild: Democratizing Ecological Monitoring via Knowledge-Adaptive Edge Expert Agents 

**Authors**: Jiaxing Li, Hao Fang, Chi Xu, Miao Zhang, Jiangchuan Liu, William I. Atlas, Katrina M. Connors, Mark A. Spoljaric  

**Link**: [PDF](https://arxiv.org/pdf/2605.16671)  

**Abstract**: Rapid biodiversity loss underscore the urgency of effective monitoring, yet manual surveys remain resource-intensive. While on-device AI offers a scalable alternative, its performance in the wild is often challenged by environmental variability. Current methods rely heavily on cloud resource, which requires continuous uploading of field data for model retraining. This approach is unsuitable for remote deployments because it consumes limited power and network connectivity. To address these constraints, this research proposes a shift from model adaptation to knowledge adaptation. We introduce an architecture that separates visual perception from reasoning, combining a visual encoder with a dynamic knowledge base. We uses an explicit knowledge base to replace implicitly encoding expert knowledge into model parameters. This method also supports knowledge sustainability by preserving expert insights in a structured form. Through cross-disciplinary collaboration with biologists and Indigenous communities, this work advances ethical AI co-development, fostering responsible and culturally informed ecosystem management. 

---
# TTE-Flash: Accelerating Reasoning-based Multimodal Representations via Think-Then-Embed Tokens 

**Authors**: Jianpeng Cheng, Xian Wu, Jiangfan Zhang, Wentao Bao, Chaitanya Ahuja, Shlok Kumar Mishra, Hanchao Yu, Yang Gao, Fan Xia, Qi Guo, Shaodan Zhai, Xiangjun Fan, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16638)  

**Abstract**: Recent research has demonstrated that Universal Multimodal Embedding (UME) benefits significantly from Chain-of-Thought (CoT) reasoning. In this paradigm, a generative model produces explicit reasoning traces for a multimodal query, with the final representation extracted from an <eos> embedding token attending to both the query and the reasoning. Despite its effectiveness, the computational overhead of generating explicit CoT traces is often prohibitive. In this work, we propose replacing explicit CoT with latent think tokens, which are interpreted as latent variables that can produce explicit CoT traces as observed variables. By optimizing think tokens using CoT generation loss and subsequent embedding tokens using contrastive loss, we produce high-performance, reasoning-aware representations at a constant inference cost. Our study investigates two key architectural designs: 1) how think and embeddings tokens should be extracted from the same LLM backbone. 2) how the tokens should be trained as two dependent tasks. We introduce TTE-Flash-2B, a reasoning-aware multimodal representation model that outperforms its explicit-CoT counterpart on the MMEB-v2 benchmark, while producing latent think tokens that are interpretable both textually and visually. Furthermore, zero-shot evaluation across 15 video datasets reveals scaling behavior as the number of think tokens increases, and motivating a pilot study of adaptive think budget allocation based on task requirements. 

---
# PRISMat: Policy-Driven, Permutation-Invariant Autoregressive Material Generation 

**Authors**: Claire Schlesinger, Circe Hsu, Peter Schindler, Robin Walters  

**Link**: [PDF](https://arxiv.org/pdf/2605.16612)  

**Abstract**: Rapid identification of candidate materials with target properties has become a key task in materials science. Machine learning has emerged as an alternative to physics-based simulation, offering a faster and cheaper way to filter materials based on their stability and other target properties, reducing the number of candidates that reach the costly synthesis stage. Recently, Large Language Models (LLMs) have been applied to this role, but these models are parameter-heavy and computationally expensive both during training and at inference time, making them unsuitable for high-throughput tasks. This inefficiency stems from both the large over-parameterization of language models and the difficulty of framing material generation as a sequence learning problem. In this paper, we present PRISMat, a cost-effective, permutation-invariant model, which addresses these limitations. We show that PRISMat, despite taking less time for inference, is able to outperform LLMs in generating crystal slabs conditioned on critical materials' surface properties. In targeted material discovery, we achieve mean absolute errors of 0.188 eV/A$^2$ and 2.79 eV for cleavage energy and work function tasks, respectively, reducing the error of the next best model by 4$\times$. 

---
# Counterparty Modeling is Not Strategy: The Limits of LLM Negotiators 

**Authors**: Romain Cosentino, Sarath Shekkizhar, Adam Earle, Silvio Savarese  

**Link**: [PDF](https://arxiv.org/pdf/2605.16575)  

**Abstract**: Negotiation requires more than inferring what the other side wants: it requires using that information to make advantageous offers and counteroffers over multiple turns. We study whether large language model (LLM) agents do this in a controlled multi-attribute bargaining environment. We find that current LLM agents can model a counterparty's preferences, but do not reliably turn that knowledge into strategic bargaining. When given negotiating partner preference information, agents model it accurately and early in their reasoning traces, yet this does not reliably improve outcomes for the informed side. Turn-level analyses show why: agents often respond to what they believe the counterparty values, but do not consistently pair those moves with gains on their own high-value attributes. Sellers are more accommodating overall, and in asymmetric-information conditions, the informed side often makes the more weakly compensated concessions. Because agents fail to leverage this underlying utility structure for strategic advantage, their final agreements are heavily dictated by surface-level opening anchors rather than actual utility weights. Finally, requiring agents to explicitly state concession-for-reciprocity trades before making an offer makes individual turns look more strategic, but ultimately fails to improve the efficiency of the final agreements. 

---
# Scalable Uncertainty Reasoning in Knowledge Graphs 

**Authors**: Jingcheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16568)  

**Abstract**: Knowledge Graphs are pivotal for semantic data integration. The real-world data they model is often inherently uncertain. Within knowledge graphs, uncertainty manifests in three distinct levels: imprecise attribute values, probabilistic triple existence, and incomplete schema knowledge. However, current Semantic Web standards lack native support for reasoning over such uncertainty, and naïve extensions often incur computational intractability. In this thesis, I aim to develop a modular framework that addresses each level through tailored techniques: (1) defining probabilistic literals and a corresponding query algebra for continuous attributes; (2) a compilation-based framework transforming SPARQL provenance into tractable probabilistic circuits for uncertain triples; and (3) topology-aware geometric embeddings for statistical schema reasoning. The central hypothesis is that specialized reasoning mechanisms, namely algebraic, logical, and geometric approaches, can reconcile semantic precision with computational tractability. 

---
# Skim: Speculative Execution for Fast and Efficient Web Agents 

**Authors**: Mike Wong, Kevin Hsieh, Suman Nath, Ravi Netravali  

**Link**: [PDF](https://arxiv.org/pdf/2605.16565)  

**Abstract**: Skim is a speculative execution framework for web agents that exploits the predictable structure of purpose-built websites. Today's web-agent expense is not intrinsic to the tasks but a property of how agents are composed: frontier-model inference, browser rendering, and ReAct-style planning are applied to every step of every task regardless of complexity. Skim's key observation is that websites enforce stable URL patterns, answer formats, and task-to-trajectory mappings across queries of the same type, so most queries can bypass these heavyweight components entirely. An offline profiler captures these patterns once per site. At runtime, Skim matches each query to a template, synthesizes the destination URL, and extracts the answer with a small model. A lightweight verifier gates each fast-path output against the query and schema; rare misspeculations cascade to the full agent, warm-started by the fast path's final URL to preserve upstream trajectory progress. Across standard web-agent benchmarks paired with three backboneagents (WebVoyager, AgentOccam, BrowserUse), Skim reduces median per-task cost by 1.9x and latency by 33.4% with no accuracy loss. 

---
# From Prompts to Protocols: An AI Agent for Laboratory Automation 

**Authors**: Angelos Angelopoulos, James F. Cahoon, Ron Alterovitz  

**Link**: [PDF](https://arxiv.org/pdf/2605.16552)  

**Abstract**: Automating science laboratories enables faster, safer, more accurate, and more reproducible execution of protocols, accelerating the discovery and testing of new materials, drugs, and more. However, setting up and running autonomous labs requires coordinating numerous instruments and robots, forcing scientists to write code, manage configuration files, and navigate complex software infrastructure. We present an AI agent architecture that integrates large language models with laboratory orchestration, enabling scientists to interactively create and monitor automated lab protocols using natural language. Integrated into the Experiment Orchestration System (EOS), the AI agent operates under an agentic loop with automated validation and error correction, and supports the complete experimental lifecycle: creating protocols, running and monitoring both protocols and closed-loop optimization campaigns, and analyzing results. A visual graph editor renders protocols as interactive node-based diagrams synchronized with the AI agent's protocol representation, enabling seamless alternation between AI-assisted and manual protocol construction. Evaluated on three simulated automated labs spanning chemistry, biology, and materials science, the AI agent achieves a 97% first-attempt protocol generation success rate and an order of magnitude reduction in required interface actions. 

---
# ANNEAL: Adapting LLM Agents via Governed Symbolic Patch Learning 

**Authors**: Safayat Bin Hakim, Keyan Guo, Wenkai Tan, Alvaro Velasquez, Shouhuai Xu, Houbing Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.16309)  

**Abstract**: LLM-based agents can recover from individual execution errors, yet they repeatedly fail on the same fault when the underlying process knowledge--operator schemas, preconditions, and constraints--remains unrepaired. Existing self-evolving approaches address this gap by updating prompts, memory, or model weights, but none directly repair the symbolic structures that encode how tasks are executed, and few provide the governance guarantees required for safe deployment. We introduce ANNEAL, a neuro-symbolic agent that converts recurring failures into governed symbolic edits of a process knowledge graph without modifying foundation model weights. Its core mechanism, Failure-Driven Knowledge Acquisition (FDKA), localizes the responsible operator, synthesizes a typed patch through constrained LLM generation, and validates the proposal via multi-dimensional scoring, symbolic guardrails, and canary testing before commit. Every accepted edit carries full provenance and deterministic rollback capability. Across four domains and 27 multi-seed runs, ANNEAL is the only evaluated system that commits persistent structural repairs--strong baselines such as ReAct and Reflexion achieve high episodic recovery yet retain 72-100% holdout failure rates on recurring faults, whereas ANNEAL reduces these to 0% in the tested recurring-failure settings. Ablation confirms that removing FDKA eliminates all structural repairs and drops success rate by up to 26.7 percentage points. These results suggest that governed symbolic repair offers a complementary paradigm to weight-level and prompt-level adaptation for persistent fault elimination. 

---
# AgentWall: A Runtime Safety Layer for Local AI Agents 

**Authors**: Ashwin Aravind  

**Link**: [PDF](https://arxiv.org/pdf/2605.16265)  

**Abstract**: The safety of autonomous AI agents is increasingly recognized as a critical open problem. As agents transition from passive text generators to active actors capable of executing shell commands, modifying files, calling APIs, and browsing the web, the consequences of unsafe or adversarially manipulated behavior become immediate and tangible. Existing AI safety work has focused primarily on model alignment and input filtering, but these approaches do not address what happens at the moment an agent's intent becomes a real action on a real machine. This gap is especially acute in local environments, where developers run agents against their own filesystems, credentials, and infrastructure with little runtime control. This paper introduces AgentWall, a runtime safety and observability layer for local AI agents. AgentWall intercepts every proposed agent action before it reaches the host environment, evaluates it against an explicit declarative policy, requires human approval for sensitive operations, and records a complete execution trail for audit and replay. It is implemented as a policy-enforcing MCP proxy and native OpenClaw plugin, working across Claude Desktop, Cursor, Windsurf, Claude Code, and OpenClaw with a single install command. We present the design, architecture, threat model, and policy model of AgentWall, and demonstrate 92.9% policy enforcement accuracy with sub-millisecond overhead across 14 benchmark tests. AgentWall is open-source at this https URL. 

---
# DashAttention: Differentiable and Adaptive Sparse Hierarchical Attention 

**Authors**: Yuxiang Huang, Nuno M. T. Gonçalves, Federico Alvetreti, Lei Li, Xu Han, Edoardo M. Ponti, André F. T. Martins, Marcos V. Treviso  

**Link**: [PDF](https://arxiv.org/pdf/2605.18753)  

**Abstract**: Current hierarchical attention methods, such as NSA and InfLLMv2, select the top-k relevant key-value (KV) blocks based on coarse attention scores and subsequently apply fine-grained softmax attention on the selected tokens. However, the top-k operation assumes the number of relevant tokens for any query is fixed and it precludes the gradient flow between the sparse and dense stages. In this work, we propose DashAttention (Differentiable and Adaptive Sparse Hierarchical Attention), which leverages the adaptively sparse $\alpha$-entmax transformation to select a variable number of blocks according to the current query in the first stage. This in turn provides a prior for the second-stage softmax attention, keeping the entire hierarchy fully differentiable. Contrary to other hierarchical attention methods, we show that DashAttention is non-dispersive, translating to better long-context modeling ability. Experiments with large language models (LLMs) show that DashAttention achieves comparable accuracy as full attention with 75% sparsity and a better Pareto frontier than NSA and InfLLMv2, especially in high-sparsity regimes. We also provide an efficient, GPU-aware implementation of DashAttention in Triton, which achieves a speedup of up to over FlashAttention-3 at inference time. Overall, DashAttention offers a cost-effective strategy to model long contexts. 

---
# Code as Agent Harness 

**Authors**: Xuying Ning, Katherine Tieu, Dongqi Fu, Tianxin Wei, Zihao Li, Yuanchen Bei, Jiaru Zou, Mengting Ai, Zhining Liu, Ting-Wei Li, Lingjie Chen, Yanjun Zhao, Ke Yang, Bingxuan Li, Cheng Qian, Gaotang Li, Xiao Lin, Zhichen Zeng, Ruizhong Qiu, Sirui Chen, Yifan Sun, Xiyuan Yang, Ruida Wang, Rui Pan, Chenyuan Yang, Dylan Zhang, Liri Fang, Zikun Cui, Yang Cao, Pan Chen, Dorothy Sun, Ren Chen, Mahesh Srinivasan, Nipun Mathur, Yinglong Xia, Hong Li, Hong Yan, Pan Lu, Lingming Zhang, Tong Zhang, Hanghang Tong, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2605.18747)  

**Abstract**: Recent large language models (LLMs) have demonstrated strong capabilities in understanding and generating code, from competitive programming to repository-level software engineering. In emerging agentic systems, code is no longer only a target output. It increasingly serves as an operational substrate for agent reasoning, acting, environment modeling, and execution-based verification. We frame this shift through the lens of agent harnesses and introduce code as agent harness: a unified view that centers code as the basis for agent infrastructure. To systematically study this perspective, we organize the survey around three connected layers. First, we study the harness interface, where code connects agents to reasoning, action, and environment modeling. Second, we examine harness mechanisms: planning, memory, and tool use for long-horizon execution, together with feedback-driven control and optimization that make harness reliable and adaptive. Third, we discuss scaling the harness from single-agent systems to multi-agent settings, where shared code artifacts support multi-agent coordination, review, and verification. Across these layers, we summarize representative methods and practical applications of code as agent harness, spanning coding assistants, GUI/OS automation, embodied agents, scientific discovery, personalization and recommendation, DevOps, and enterprise workflows. We further outline open challenges for harness engineering, including evaluation beyond final task success, verification under incomplete feedback, regression-free harness improvement, consistent shared state across multiple agents, human oversight for safety-critical actions, and extensions to multimodal environments. By centering code as the harness of agentic AI, this survey provides a unified roadmap toward executable, verifiable, and stateful AI agent systems. 

---
# ESI-Bench: Towards Embodied Spatial Intelligence that Closes the Perception-Action Loop 

**Authors**: Yining Hong, Jiageng Liu, Han Yin, Manling Li, Leonidas Guibas, Li Fei-Fei, Jiajun Wu, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.18746)  

**Abstract**: Spatial intelligence unfolds through a perception-action loop: agents act to acquire observations, and reason about how observations vary as a function of action. Rather than passively processing what is seen, they actively uncover what is unseen - occluded structure, dynamics, containment, and functionality that cannot be resolved from passive sensing alone. We move beyond prior formulations of spatial intelligence that assume oracle observations by recasting the observer as an actor. We introduce ESI-BENCH, a comprehensive benchmark for embodied spatial intelligence spanning 10 task categories and 29 subcategories built on OmniGibson, grounded in Spelke's core knowledge systems. Agents must decide what abilities to deploy - perception, locomotion, and manipulation - and how to sequence them to actively accumulate task-relevant evidence. We conduct extensive experiments on state-of-the-art MLLMs and find that active exploration substantially outperforms passive counterparts, with agents spontaneously discovering emergent spatial strategies without explicit instructions, while random multi-view often adds noise rather than signal despite consuming far more images. Most failures stem not from weak perception but from action blindness: poor action choices lead to poor observations, which in turn drive cascading errors. While explicit 3D grounding stabilizes reasoning on depth-sensitive tasks, imperfect 3D representation proves more harmful than 2D baselines by distorting spatial relations. Human studies further reveal that unlike humans who seek falsifying viewpoints and revise beliefs under contradiction, models commit prematurely with high confidence regardless of evidence quality, exposing a metacognitive gap that neither better perception nor more embodied interaction alone can close. 

---
# Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation 

**Authors**: Qianhao Yuan, Jie Lou, Xing Yu, Hongyu Lin, Le Sun, Xianpei Han, Yaojie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18740)  

**Abstract**: Multimodal Large Language Models (MLLMs) still struggle with fine-grained visual understanding, where answers often depend on small but decisive evidence in the full image. We observe a regional-to-global perception gap: the same MLLM answers fine-grained questions more accurately when conditioned on evidence-centered crops than on the corresponding full images, suggesting that many failures stem from difficulty to focus on relevant evidence rather than insufficient local recognition ability. Motivated by this observation, we propose Vision-OPD (Vision On-Policy Distillation), a regional-to-global self-distillation framework that transfers the model's own privileged regional perception to its full-image policy. Vision-OPD instantiates two conditional policies from the same MLLM: a crop-conditioned teacher and a full-image-conditioned student. The student generates on-policy rollouts, and Vision-OPD minimizes token-level divergence between the teacher and student next-token distributions along these rollouts. This enables the model to internalize the benefit of visual zooming without external teacher models, ground-truth labels, reward verifiers, or inference-time tool use. Experiments on multiple fine-grained visual understanding benchmarks show that Vision-OPD models achieve competitive or superior performance against much larger open-source, closed-source, and "Thinking-with-Images" agentic models. 

---
# Predictable Confabulations: Factual Recall by LLMs Scales with Model Size and Topic Frequency 

**Authors**: Matthew L. Smith, Jonathan P. Shock, Samuel T. Segun, Iyiola E. Olatunji, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2605.18732)  

**Abstract**: While scaling laws govern aggregate large language model performance, no scaling law has linked factual recall to both model size and training-data composition. We evaluated 38 models on over 8,900 scholarly references evaluated by an automated reference verification system. Recall quality follows a sigmoid in the log-linear combination of model parameter count and topic representation in training data. These two variables alone explain 60% of the variance across 16 dense models from four families, rising to 74-94% within individual families. The form matches a superposition-inspired account in which recall is gated by a signal-to-noise ratio: signal strength scales with concept frequency and the noise floor with model capacity. 

---
# DexHoldem: Playing Texas Hold'em with Dexterous Embodied System 

**Authors**: Feng Chen, Tianzhe Chu, Li Sun, Pei Zhou, Zhuxiu Xu, Shenghua Gao, Yuexiang Zhai, Yanchao Yang, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.18727)  

**Abstract**: Evaluating embodied systems on real dexterous hardware requires more than isolated primitive skills: an agent must perceive a changing tabletop scene, choose a context-appropriate action, execute it with a dexterous hand, and leave the scene usable for later decisions. We introduce DexHoldem, a real-world system-level benchmark built around Texas Hold'em dexterous manipulation with a ShadowHand. DexHoldem provides 1,470 teleoperated demonstrations across 14 Texas Hold'em manipulation primitives, a standardized physical policy benchmark, and an agentic perception benchmark that tests whether agents can recover the structured game state needed for embodied decision making. On primitive execution, $\pi_{0.5}$ obtains the highest task completion rate ($61.2\%$), while $\pi_{0.5}$ and $\pi_0$ tie on scene-preserving success rate ($47.5\%$). On agentic perception, Opus 4.7 obtains the best strict problem-level accuracy ($34.3\%$), while GPT 5.5 obtains the best average field-wise accuracy ($66.8\%$), exposing a gap between isolated visual sub-capabilities and complete routing-relevant state recovery. Finally, we instantiate the full embodied-agent loop in three case studies, where waiting, recovery dispatches, human-help requests, and repeated primitive execution reveal how perception and policy errors accumulate during closed-loop deployment. DexHoldem therefore evaluates dexterous tabletop execution, agentic perception, and embodied decision routing in a shared physical setting. Project page: this https URL. 

---
# Semantic Generative Tuning for Unified Multimodal Models 

**Authors**: Songsong Yu, Yuxin Chen, Ying Shan, Yanwei Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.18714)  

**Abstract**: Unified multimodal models (UMMs) strive to consolidate visual understanding and visual generation within a single architecture. However, prevailing training paradigms independently optimize understanding via sparse text signals and generation through dense pixel objectives. Such a decoupled strategy yields misaligned representation spaces, isolating visual understanding from generation and hindering their mutual reinforcement. This work presents the first systematic investigation into generative post-training, where we formulate hierarchical visual tasks as generative proxies to bridge the isolation in UMMs. Our empirical investigation reveals that high-level semantic tasks, particularly image segmentation, serve as optimal proxies. Unlike low-level tasks that distract models with texture details, segmentation provides structural semantics that significantly enhance both vision-centric perception and generative layout fidelity. Building upon these insights, we introduce Semantic Generative Tuning (SGT), a novel paradigm that leverages segmentation as a generative proxy to align and synergize multimodal capabilities. Mechanistic analyses further demonstrate that SGT fundamentally improves feature linear separability and optimizes visual-textual attention allocation pattern. Extensive evaluations show that SGT consistently improves both multimodal comprehension and generative fidelity across mainstream benchmarks. Our code is available on the this https URL. 

---
# Distilling Tabular Foundation Models for Structured Health Data 

**Authors**: Aditya Tanna, Nassim Bouarour, Mohamed Bouadi, Vinay Kumar Sankarapu, Pratinav Seth  

**Link**: [PDF](https://arxiv.org/pdf/2605.18702)  

**Abstract**: Tabular foundation models (TFMs) achieve strong performance on health datasets, but their inference cost and infrastructure requirements limit practical use. We study whether their predictive behavior can be transferred to lightweight tabular models through knowledge distillation. Since in-context TFMs condition on the training set at inference time, naive distillation can introduce context leakage; we address this with stratified out-of-fold teacher labeling. Across $19$ healthcare datasets, $6$ TFM teachers, $4$ student families, and several multi-teacher ensembles, we find that distilled students retain at least $90\%$ of teacher AUC, outperforming teachers in some cases, while running at least $26\times$ faster on CPU and preserving calibration and fairness critical for health applications. Moreover, multi-teacher averaging does not consistently improve over the best single teacher. Leakage-aware distillation is thus a viable route for bringing TFM-quality predictions into inference-constrained health settings. 

---
# PopPy: Opportunistically Exploiting Parallelism in Python Compound AI Applications 

**Authors**: Stephen Mell, David Mell, Konstantinos Kallas, Steve Zdancewic, Osbert Bastani  

**Link**: [PDF](https://arxiv.org/pdf/2605.18697)  

**Abstract**: Compound AI applications, which compose calls to ML models using a general-purpose programming language like Python, are widely used for a variety of user-facing tasks, from software engineering to enterprise automation, making their end-to-end latency a critical bottleneck. In contrast to traditional applications, execution time is dominated by the external components, which cannot be handled by traditional language optimization systems, like optimizing compilers.
To address this problem, we develop PopPy, a system that can uncover parallelization opportunities in Python applications that invoke these heavy external components, including those used in compound AI applications. PopPy supports a very expressive fragment of Python and requires minimal developer input to uncover parallelism. It combines an ahead-of-time compiler with a runtime, addressing three key challenges in extracting parallelism from Python applications: language complexity, dynamic dispatch, and variable mutation. On a set of real-world compound AI applications, PopPy achieves up to $6.4\times$ speedups in end-to-end execution time compared to standard Python execution while preserving the sequential program semantics. 

---
# Ensembling Tabular Foundation Models - A Diversity Ceiling And A Calibration Trap 

**Authors**: Aditya Tanna, Yash Desai, Pratinav Seth, Mohamed Bouadi, Nassim Bouarour, Vinay Kumar Sankarapu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18696)  

**Abstract**: Tabular foundation models (TFMs) now match or beat tuned gradient-boosted trees on a growing fraction of tabular tasks, but no single TFM wins on every dataset. Ensembling is the go to fix here, and it works less well than expected. Six modern TFMs form a near-redundant pool: their mean pairwise Q-statistic is $0.961$, close enough to $1$ that any convex combination is bounded above. We benchmark six ensemble strategies over six TFMs on 153 OpenML classification tasks. The best ensemble, two-level cascade stacking, buys $+0.18\%$ accuracy over the strongest single TFM at $253\times$ the compute. A Friedman and Nemenyi analysis places three ensembles and the best base TFM in a single equivalence group; three other ensembles are significantly \emph{worse} than the best base. Stacking with a logistic-regression meta-learner is the most striking case: competitive accuracy and ROC-AUC, the worst log-loss rank among the ensembles. The meta-learner improves accuracy by sharpening class boundaries, which destroys calibration. We recommend greedy selection as the practical default. 

---
# Reversa: A Reverse Documentation Engineering Framework for Converting Legacy Software into Operational Specifications for AI Agents 

**Authors**: Sanderson Oliveira de Macedo, Ronaldo Martins da Costa  

**Link**: [PDF](https://arxiv.org/pdf/2605.18684)  

**Abstract**: Legacy systems concentrate business rules, architectural decisions, and operational exceptions that often remain implicit in code, data, configuration, and
maintenance practices. At the same time, language-model-based coding agents depend on reliable context, correctness criteria, and behavioral contracts to
modify real systems with lower risk. This paper presents Reversa, a reverse documentation engineering framework for converting legacy software into
traceable operational specifications for AI agents. Reversa organizes this process as a multi-agent pipeline: specialized agents map the project surface,
analyze modules, extract implicit rules, synthesize architecture, write unit-level specifications, and review generated claims. The proposal emphasizes
three mechanisms: traceability between code and specification, explicit confidence marking, and preservation of gaps for human validation. The framework is
distributed as a this http URL CLI, installs skills across multiple agent engines, and uses a SHA-256 manifest to preserve modified files during update or
uninstall operations. In addition to the architectural description, we report an exploratory case study on migrating an ATM from COBOL to Go, in which the
pipeline produced 517 claims classified by an internal confidence index, 10 registered gaps, 53 Gherkin parity scenarios, and a reconstruction plan with 9
of 11 tasks completed at inventory time. Final parity validation and cutover were not completed in this study. We do not claim broad empirical superiority;
we position the contribution with respect to the literature on reverse engineering, LLM-based documentation, and software agents, and propose an evaluation
protocol with metrics for coverage, traceability, confidence, utility, and cost. 

---
# Lance: Unified Multimodal Modeling by Multi-Task Synergy 

**Authors**: Fengyi Fu, Mengqi Huang, Shaojin Wu, Yunsheng Jiang, Yufei Huo, Hao Li, Yinghang Song, Fei Ding, Jianzhu Guo, Qian He, Zheren Fu, Zhendong Mao, Yongdong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18678)  

**Abstract**: We present Lance, a lightweight native unified model supporting multimodal understanding, generation, and editing for both images and videos. Rather than relying on model capacity scaling or text-image-dominant designs, Lance explores a practical paradigm for unified multimodal modeling via collaborative multi-task training. It is grounded in two core principles: unified context modeling and decoupled capability pathways. Specifically, Lance is trained from scratch and employs a dual-stream mixture-of-experts architecture on shared interleaved multimodal sequences, enabling joint context learning while decoupling the pathways for understanding and generation. We further introduce modality-aware rotary positional encoding to mitigate interference among heterogeneous visual tokens and boost cross-task alignment. During training, Lance adopts a staged multi-task training paradigm with capability-oriented objectives and adaptive data scheduling to strengthen both semantic comprehension and visual generation performance. Experimental results demonstrate that Lance substantially outperforms existing open-source unified models in image and video generation, while retaining strong multimodal understanding capabilities. The homepage is available at this https URL. 

---
# COOPO: Cyclic Offline-Online Policy Optimization Algorithm 

**Authors**: Qisai Liu, Zhanhong Jiang, Joshua Russell Waite, Aditya Balu, Cody Fleming, Soumik Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2605.18675)  

**Abstract**: Offline reinforcement learning struggles with distributional shift and constrained performance due to static dataset limitations, while online RL demands prohibitive environment interactions. The recent advent of hybrid offline-to-online methods bridges these domains but suffers from distribution drift during transitions and catastrophic forgetting of offline knowledge. We introduce COOPO (Cyclic Offline-Online Policy Optimization), a generalized framework that repeatedly cycles between constrained offline training and online fine-tuning. Each cycle first anchors the policy to the dataset via KL-regularized advantage-weighted offline updates to minimize distributional shift and then fine-tunes it online using any policy optimization for stable exploration. Crucially, periodically returning to offline training eliminates forgetting and drift while maximizing dataset reuse. The cyclic behavior also helps reduce the online environment interactions. Theoretically, COOPO achieves better online sample efficiency, surpassing pure online RL, with guaranteed monotonic improvement under standard coverage assumptions. Extensive D4RL benchmarks demonstrate COOPO reduces online interactions versus state-of-the-art hybrids while improving final returns, maintaining robustness across diverse offline algorithms and online optimizers. This looped synergy sets new efficiency and performance standards for adaptive RL. 

---
# KairosHope: A Next-Generation Time-Series Foundation Model for Specialized Classification via Dual-Memory Architecture 

**Authors**: Luis Balderas, José Alberto Rodríguez, Miguel Lastra, Antonio Arauzo-Azofra, José M. Benítez  

**Link**: [PDF](https://arxiv.org/pdf/2605.18657)  

**Abstract**: Time Series Foundation Models (TSFMs) have demonstrated notable success in general-purpose forecasting tasks; however, their adaptation to specialized classification problems remains constrained by the computational bottleneck of standard attention and the systematic omission of classical statistical knowledge. This technical report introduces KairosHope, a next-generation TSFM designed to reconcile massive generalization with analytical precision in classification tasks. The core of the proposal is the HOPE block, an architecture that replaces quadratic attention with a dual-memory system: Titans modules for dynamic short-term retention and a Continuum Memory System (CMS) for the abstraction of long-term historical context. To enrich the inductive bias, a Hybrid Decision Head is introduced, which fuses deep latent representations with deterministic statistical features extracted via tsfeatures package. KairosHope undergoes self-supervised pre-training on the massive Monash archive, combining Masked Time Series Modeling (MTSM) and contrastive learning (InfoNCE). Its subsequent adaptation to the UCR benchmark datasets is conducted through a rigorous Linear Probing and Full Fine-Tuning (LP-FT) protocol to prevent catastrophic forgetting. Empirical results demonstrate superior performance in domains characterized by strict temporal causality such as HAR or Sensor data. Consequently, KairosHope establishes a robust and efficient framework for the adaptation of foundation models to time series analysis. 

---
# Statistical Limits and Efficient Algorithms for Differentially Private Federated Learning 

**Authors**: Arnab Auddy, Xiangni Peng, Subhadeep Paul  

**Link**: [PDF](https://arxiv.org/pdf/2605.18656)  

**Abstract**: Federated Learning is a leading framework for training ML and AI models collaboratively across numerous user devices or databases. We study the trade-offs among estimation accuracy, privacy constraints, and communication cost for differentially private (DP) federated M estimation. The two standard methods in the literature are FedAvg, which may suffer from high federation bias, and FedSGD, which can incur high communication cost. Aimed at improving accuracy at a reduced communication cost, we propose FedHybrid, which uses FedSGD starting with an improved initialization by the FedAvg estimator. We propose FedNewton, which averages local Newton iterations to reduce bias in FedAvg, achieving an estimation accuracy comparable to FedSGD with much fewer communication rounds when the number of clients grows sufficiently slowly. We establish finite sample upper bounds on the mean-squared error rates of the DP versions of these estimators as functions of the number of clients, local sample sizes, privacy budget, and number of iterations. We further derive a minimax lower bound on the MSE of any iterative private federated procedure that provides a benchmark to assess the optimality gap of these methods. We numerically evaluate our methods for training a logistic regression and a neural network on the computer vision datasets MNIST and CIFAR-10. 

---
# Pocket Foundation Models: Distilling TFMs into CPU-Ready Gradient-Boosted Trees 

**Authors**: Aditya Tanna, Nassim Bouarour, Mohamed Bouadi, Vinay kumar Sankarapu, Pratinav Seth  

**Link**: [PDF](https://arxiv.org/pdf/2605.18654)  

**Abstract**: A fraud scorer needs to answer in under 2 ms. The best tabular foundation models (TFMs) take 151-1,275 ms on GPU. We close this gap by distilling the TFM offline into an XGBoost or CatBoost student that runs natively on CPU. The central obstacle is specific to in-context learning (ICL) teachers: they leak labels when scoring their own training set, so the soft targets collapse to near-one-hot vectors with no inter-class structure left to distill. Stratified out-of-fold (OOF) teacher labeling prevents this. Across 153 classification datasets drawn from TALENT, OpenML-CC18, TabZilla, and TabArena, distilling TabICLv2 into XGBoost gives 0.882 macro-mean AUC (96.5% of teacher AUC) at 1.9 ms on CPU, a 38x to 860x speedup across teacher-student pairs with a statistically significant edge over a tuned CatBoost baseline (Wilcoxon p = 0.0008; 51% win rate). Four further findings: teacher rank transfers exactly to student rank; gains concentrate on low-dimensional data (< 21 features: +0.011 over CatBoost vs. >21 features: +0.001); multi-teacher averaging helps MLP students (+0.006, p = 0.003) but adds less than 0.001 for tree students; and on high-dimensional tasks where the teacher itself trails CatBoost, distillation makes things worse rather than better. The full pipeline is open-sourced as part of the TabTune library. 

---
# An Assessment of Human vs. Model Uncertainty in Soft-Label Learning and Calibration 

**Authors**: Maja Pavlovic, Silviu Paun, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2605.18648)  

**Abstract**: Central to human-aligned AI is understanding the benefits of human-elicited labels over synthetic alternatives. While human soft-labels improve calibration by capturing uncertainty, prior studies conflate these benefits with the implicit correction of mislabeled data (mode shifts), obscuring true effects of soft-labels. We present a controlled audit of soft-label learning across MNIST and a synthetic variant, re-annotating subsets to extract human uncertainty. By decoupling soft-label supervision from underlying label mode shifts, we show that while human soft-labels do provide accuracy gains, their larger value lies in acting as a regularizer that improves model calibration on difficult samples and promotes stable convergence across training runs. Dataset cartography reveals models trained on human soft-labels mirror human uncertainty, whereas those trained on synthetic labels fail to align with humans. Broadly, this work provides a diagnostic testbed for human-AI uncertainty alignment. 

---
# Post-Trained MoE Can Skip Half Experts via Self-Distillation 

**Authors**: Xingtai Lv, Li Sheng, Kaiyan Zhang, Yichen You, Siyan Gao, Xueheng Luo, Yuxin Zuo, Yuchen Fan, Junlin Yang, Ganqu Cui, Bingning Wang, Fan Yang, Youbang Sun, Ning Ding, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.18643)  

**Abstract**: Mixture-of-Experts (MoE) scales language models efficiently through sparse expert activation, and its dynamic variant further reduces computation by adjusting the activated experts in an input-dependent manner. Existing dynamic MoE methods usually rely on pre-training from scratch or task-specific adaptation, leaving the practical conversion of fully trained MoE underexplored. Enabling such adaptation would directly alleviate the inference costs by allowing easy tokens to bypass unnecessary expert during serving. This paper introduces Zero-Expert Self-Distillation Adaptation (ZEDA), a low-cost framework that transforms post-trained static MoE models into efficient dynamic ones. To stabilize this architectural conversion, ZEDA injects parameter-free zero-output experts into each MoE layer and adapts the augmented model through two-stage self-distillation, utilizing the original MoE as a frozen teacher and applying a group-level balancing loss. On Qwen3-30B-A3B and GLM-4.7-Flash across 11 benchmarks spanning math, code, and instruction following, ZEDA eliminates over 50% of expert FLOPs at marginal accuracy loss. It outperforms the strongest dynamic MoE baseline by 6.1 and 4.0 points on the two models, and delivers ~1.20$\times$ end-to-end inference speedup. 

---
# Data Presentation Over Architecture: Resampling Strategies for Credit Risk Prediction with Tabular Foundation Models 

**Authors**: Aditya Tanna, Mitul Solanki, Mohamed Bouadi, Nassim Bouarour, Pratinav Seth, Vinay Kumar Sankarapu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18635)  

**Abstract**: Credit default prediction is a tabular learning problem with severe class imbalance, heterogeneous features, and tight latency budgets. Tabular Foundation Models (TFMs) approach this problem through in-context learning, which makes their predictions sensitive to how the context window is built. We benchmark four classical models and five TFMs on the Home Credit and Lending Club datasets, varying the context-construction strategy (seven options) and the context size (1K to 50K). On both datasets, the choice of context strategy explains more variance in AUC-ROC than the choice of TFM family: balanced and hybrid sampling add 3 to 4 AUC points over uniform sampling, and the gap exceeds the spread between TFMs. With a balanced context of 5K to 10K examples, the strongest TFMs reach the AUC of classical baselines trained on the full data, while also recovering meaningful default-class recall that default-threshold GBDTs do not. We frame this as evidence that context construction, rather than architecture choice, is the primary deployment lever for TFMs in imbalanced credit-risk settings. 

---
# Position: Weight Space Should Be a First-Class Generative AI Modality 

**Authors**: Zhangyang Wang, Peihao Wang, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18632)  

**Abstract**: Neural network checkpoints have quietly become a large-scale data resource: millions of trained weight vectors now exist, each encoding task-, domain-, and architecture-specific knowledge. This position paper argues that model checkpoints should be treated as a first-class data modality, and that generative modeling in weight space should be standardized as a core machine learning primitive. Recent advances demonstrate that neural weights can be synthesized on demand, often matching fine-tuning performance while reducing adaptation cost by orders of magnitude. We contend that these results reflect an underlying structural fact: high-performing models occupy low-dimensional, highly structured regions of weight space shaped by symmetry, flatness, modularity, and shared subspaces. Building on this view, we organize existing methods into a five-stage pipeline, survey applications where the approach is already practical, and clarify current limits: adapter-scale and conditional generation are advancing rapidly, while unrestricted frontier-scale checkpoint synthesis remains open. Our goal is to shift the community's default mindset from optimizing models per task to sampling models from learned weight distributions, accelerating toward an era in which AI systems routinely improve or create other AI systems. 

---
# CrossView Suite: Harnessing Cross-view Spatial Intelligence of MLLMs with Dataset, Model and Benchmark 

**Authors**: Wei Wang, Yuqian Yuan, Tianwei Lin, Wenqiao Zhang, Siliang Tang, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18621)  

**Abstract**: Spatial intelligence requires multimodal large language models (MLLMs) to move beyond single-view perception and reason consistently about objects, visibility, geometry, and interactions across multiple viewpoints. However, progress in cross-view reasoning remains limited by three major gaps: the scarcity of large-scale well-annotated training data, the lack of comprehensive benchmarks for systematic evaluation, and the absence of explicit alignment mechanisms that establish object-level consistency across views. To address these gaps, we thoroughly develop CrossView Suite across three coordinated components: CrossViewSet, CrossViewBench, and CrossViewer. Firstly, we introduce a multi-agent data engine to meticulously curate a large-scale, high-quality cross-view instruction dataset, termed CrossViewSet, covering 17 fine-grained task types with 1.6M samples. Second, we meticulously create a scene-disjoint CrossViewBench to comprehensively assess the cross-view spatial understanding capability of an MLLM, evaluating it across various aspects. Finally, we propose CrossViewer, a progressive three-stage framework for cross-view spatial reasoning in MLLMs, following a Perception -> Alignment -> Reasoning paradigm. Our method equips an adaptive spatial region tokenizer to capture fine-grained object representations, and then aligns the multi-view objects explicitly, and thus fuses aligned features for boosting the cross-view inference capacity for MLLMs. Extensive experiments and analyses show that large-scale training data, systematic evaluation, and explicit cross-view alignment are all critical for advancing MLLMs from single-view perception toward real-world spatial intelligence. The project page is available at this https URL. 

---
# Stochastic Penalty-Barrier Methods for Constrained Machine Learning 

**Authors**: Adam Bosák, Andrii Kliachkin, Jana Lepšová, Gilles Bareilles, Jakub Mareček  

**Link**: [PDF](https://arxiv.org/pdf/2605.18618)  

**Abstract**: Constrained machine learning enables fairness-aware training, physics-informed neural networks, and integration of symbolic domain knowledge into statistical models. Despite its practical importance, no general method exists for the non-convex, non-smooth, stochastic setting that arises naturally in deep learning. We propose the Stochastic Penalty-Barrier Method (SPBM), which extends classical penalty and barrier methods to this setting via exponential dual averaging, a~stabilized penalty schedule, and the Moreau envelope to handle non-smoothness. Experiments across multiple settings show that SPBM matches or outperforms existing constrained optimization baselines while incurring only linear runtime overhead compared to unconstrained Adam for up to 10,000 constraints. 

---
# ManiSoft: Towards Vision-Language Manipulation for Soft Continuum Robotics 

**Authors**: Ziyu Wei, Luting Wang, Chen Gao, Li Wen, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18617)  

**Abstract**: Most existing vision-language manipulation research targets rigid robotic arms, whose fixed morphology limits adaptability in cluttered or confined spaces. Soft robotic arms offer an appealing alternative due to their deformability, but confront challenges such as unreliable proprioception and distributed low-level actuation. To investigate these challenges, we introduce \ManiSoft, a benchmark for vision-language manipulation with soft arms. ManiSoft features a tailored simulator that couples realistic soft-body dynamics with contact-rich interactions via an elastic force constraint. On this basis, ManiSoft defines four tasks, each highlighting distinct aspects of deformable control, from basic end-effector coordination to obstacle avoidance. To support policy training and evaluation, \ManiSoft{} includes an automated pipeline that generates $6{,}300$ diverse scenes and corresponding expert trajectories. To produce high-quality trajectories at scale, we first employ a high-level planner to decompose each task into a sequence of waypoints, followed by a low-level reinforcement learning policy that generates torque commands to track waypoints. Benchmarking three representative policy models shows relatively promising results in clean scenes but substantial performance drop under randomization. Visualization analysis indicates that failures stem primarily from inaccurate visual estimation of proprioceptive state and limited exploitation of deformability for adaptive obstacle avoiding. We anticipate ManiSoft to serve as a valuable testbed, bridging the gap between rigid and soft arms in the context of vision-language manipulation. Out codes and datasets are released at this https URL. 

---
# SAME: A Semantically-Aligned Music Autoencoder 

**Authors**: Julian D. Parker, Zach Evans, CJ Carr, Zachary Zukowski, Josiah Taylor, Matthew Rice, Jordi Pons  

**Link**: [PDF](https://arxiv.org/pdf/2605.18613)  

**Abstract**: Latent representations are at the heart of the majority of modern generative models. In the audio domain they are typically produced by a neural-audio-codec autoencoder. In this work we introduce SAME (Semantically-Aligned Music autoEncoder), an autoencoder for stereo music and general audio that reaches a 4096$\times$ temporal compression ratio while maintaining reconstruction quality and downstream generative performance. We achieve this by combining a tranformer-based backbone with set of semantic regularisation approaches, phase-aware reconstruction losses and improved discriminator designs. The architecture delivers substantial computational cost benefits, through both its high compression ratio and its reliance on well-optimised transformer primitives. Two variants (a large SAME-L and a CPU-deployable SAME-S) are released in open-weights form. 

---
# CATA: Continual Machine Unlearning via Conflict-Averse Task Arithmetic 

**Authors**: Shen Lin, Junhao Dong, Rongjie Chen, Xiaoyu Zhang, Li Xu, Xiaofeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.18610)  

**Abstract**: Vision-language models (VLMs) have shown remarkable ability in aligning visual and textual representations, enabling a wide range of multimodal applications. However, their large-scale training data inevitably raises concerns about privacy, copyright, and undesirable content, creating a strong need for machine unlearning. While existing studies mainly focus on single-shot unlearning, practical VLM deployment often involves sequential removal requests over time, giving rise to continual machine unlearning. In this work, we make the first attempt to study continual unlearning for VLMs and identify three key challenges in this setting: effectiveness in removing target knowledge, fidelity in preserving retained model utility, and persistence in preventing knowledge re-emergence under sequential updates. To address these challenges, we propose CATA, a conflict-averse task arithmetic method that represents each forget request as an unlearning task vector. By maintaining historical task vectors and performing sign-aware conflict-averse aggregation, CATA suppresses conflicting update components that may weaken previous forgetting effects. Extensive experiments under both single-shot and continual settings show that CATA outperforms baselines in terms of forgetting effectiveness, model fidelity, and forgetting persistence. 

---
# Not What You Asked For: Typographic Attacks in Household Robot Manipulation 

**Authors**: Ali Iranmanesh, Peng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18593)  

**Abstract**: Open-vocabulary embodied AI agents increasingly rely on vision-language models such as CLIP for object perception and task grounding. However, the shared embedding space that enables this flexibility introduces a structural vulnerability to typographic attacks, where printed text in a physical scene semantically overrides visual judgment. While prior work has quantified this threat in static 2D benchmarks and 3D navigation tasks, its impact on the full Sense-Plan-Act pipeline of household robot manipulation remains unexplored.
This work evaluates typographic attacks in a Habitat-based simulation using the HomeRobot benchmark. We introduce a decoupled perception architecture that exposes a frozen CLIP encoder to adversarial stickers while maintaining geometric grounding via DETIC. In a controlled evaluation pool of 59 attributable episodes, the attack achieves an overall Attack Success Rate (ASR) of 67.8%, rising to 70.0% among fully successful episodes, under uncontrolled viewing angles and occlusion with no perceptual optimization.
Critically, we find that perceptual errors propagate through the persistent 3D semantic map to produce kinetic failures, defined here as physically executed grasping and transport of the wrong object driven by an adversarially poisoned semantic state. In these cases, the robot physically grasps and delivers the wrong object to a target receptacle. These results establish typographic misclassification as a real, measurable, and physically consequential threat to the safety of modular manipulation pipelines that prior typographic attack research has left unexamined. 

---
# AMARIS: A Memory-Augmented Rubric Improvement System for Rubric-Based Reinforcement Learning 

**Authors**: Peilin Wu, Xinlu Zhang, Kun Wan, Wentian Zhao, Gang Wu, Xinya Du, Zhiyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.18592)  

**Abstract**: Rubric-based reward shaping is an effective method for fine-tuning LLMs via RL, where structured rubrics decompose standard outcome rewards into multiple dimensions to provide richer reward signals. Recent works make the rubrics adaptive based on local signals such as the rollouts from the current step or pairwise comparisons. However, these methods discard the diagnostics produced during evaluation after immediate use and prevent the long-term accumulation and strategic reuse of evaluation knowledge. This forces the system to re-derive evaluation principles from scratch, limits its ability to detect recurring suboptimal behaviors, and forfeits the curriculum-like progression that a persistent training history would naturally support. To address these limitations, we introduce AMARIS, which grounds rubric modifications in long-term training history. At each training step, AMARIS analyzes individual rollouts, aggregates findings into step-level summaries, retrieves relevant historical context from a persistent evaluation memory through both static (recent steps) and dynamic (semantically matched) retrieval, and updates rubrics based on these accumulated analyses. This procedure runs asynchronously alongside the normal RL loop with minimal overhead. Experiments across both closed and open-ended domains show that AMARIS consistently outperforms the baselines. Ablation studies show that static and dynamic memory retrieval contributes to the performance gain and their combination provides the strongest results with moderate retrieval budgets sufficient to provide most of the gain, and that the entire pipeline adds only ~5\% time overhead through asynchronous execution. These results show that persistent evaluation memory can transform rubric-based reward shaping from a stateless, per-step heuristic into an evidence-driven loop for RL training. 

---
# Randomized Advantage Transformation (RAT): Computing Natural Policy Gradients via Direct Backpropagation 

**Authors**: Mingfei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.18591)  

**Abstract**: Natural policy gradients improve optimization by accounting for the geometry of distribution space, but their practical use is limited by the cost of estimating and inverting the Fisher matrix. We present Randomized Advantage Transformation (RAT), a method for estimating Tikhonov-regularized natural policy gradients via direct backpropagation. By applying the Woodbury formula, we reformulate the regularized natural policy gradients as vanilla policy gradients with a transformed advantage. RAT computes this transformation efficiently via randomized block Kaczmarz iterations on on-policy mini-batches, avoiding explicit Fisher construction, conjugate-gradient solvers, and architecture-specific approximations. We provide convergence guarantees for RAT and demonstrate empirically that it matches or exceeds established natural-gradient methods across continuous and visual control benchmarks, while remaining simple to implement and compatible with various architectures. 

---
# Overeager Coding Agents: Measuring Out-of-Scope Actions on Benign Tasks 

**Authors**: Yubin Qu, Ying Zhang, Yanjun Zhang, Gelei Deng, Yuekang Li, Leo Yu Zhang, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18583)  

**Abstract**: Coding agents now run autonomously with shell, file, and network privileges. When a user issues a benign request, the agent sometimes does more than asked: it deletes unrelated files, wipes a stale credentials backup, or rewrites configuration the user never mentioned. We call these scope expansions overeager actions, an authorization problem distinct from capability failures, prompt injection, or sandbox escapes.
We present OverEager-Gen, a benchmark dedicated to overeager behavior on benign tasks. Building it surfaces a measurement-validity issue: if a benchmark spells out the authorized scope inside the prompt, the agent stops inferring boundaries and starts pattern-matching declaration text. On Claude Code, stripping the consent declaration alone raises the overeager rate from 0.0% to 17.1% on paired scenarios (McNemar exact p = 2.4 x 10^-4). OverEager-Gen therefore certifies each scenario's discriminative power before admission via a behavioral-gradient validator, audits internal tool calls through a dual-channel stack (PATH-injected shim plus per-agent event streams), and ships byte-identical consent_kept and consent_stripped variants.
OverEager-Bench contains 500 validated scenarios and ~7,500 runs across four agent products (Claude Code, OpenHands, Codex CLI, Gemini CLI) and six base models; a 50-sample re-annotation gives Cohen's kappa = 0.73 and rule-judge recall = 1.00. Stripping consent multiplies the overeager rate on every shared base model (Delta in [11.9, 17.2] pp). The framework axis dominates effect size: a permissive cluster (Claude Code, Codex CLI, Gemini CLI) runs at 5.4-27.7% while the ask-to-continue framework (OpenHands) sits at 0.2-4.5% (Fisher p <= 10^-5). Within-framework base-model variance reaches 15.9 pp, indicating that model-layer alignment does not fully propagate through permissive permission gating. 

---
# LongMINT: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems 

**Authors**: Hyunji Lee, Justin Chih-Yao Chen, Joykirat Singh, Zaid Khan, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2605.18565)  

**Abstract**: Real-world agents operate over long and evolving horizons, where information is repeatedly updated and may interfere across memories, requiring accurate recall and aggregated reasoning over multiple pieces of information. However, existing benchmarks focus on static, independent recall and fail to capture these dynamic interactions between evolving memories. In this paper, we study how current memory-augmented agents perform in realistic, interference-heavy, long-horizon settings across diverse domains and question types. We introduce LongMINT (Long-Horizon Memory under INTerference), a benchmark featuring (1) long, highly interconnected contexts with frequently updated information that induces substantial interference, (2) diverse domains (state tracking, multi-turn dialogue, Wikipedia revisions, and GitHub commits), enabling evaluation of domain generalization, and (3) diverse question types that assess robustness to interference, including (i) single-target recall tasks requiring retrieval of a specific target from long contexts, and (ii) multi-target aggregation tasks requiring reasoning over multiple relevant pieces of information. Overall, LongMINT has 15.6k question-answering pairs over long-horizon contexts averaging 138.8k tokens and extending up to 1.8M tokens per instance. We evaluate 7 representative systems, including vanilla long-context LLMs, RAG, and memory-augmented agent frameworks. Across all systems, we observe consistently low performance (avg. 27.9% accuracy), especially on questions requiring aggregated reasoning over multiple pieces of evidence. Our analysis shows that performance is primarily limited by retrieval and memory construction. Furthermore, current memory systems struggle to recall and reason over earlier facts that are later revised or interfered with by subsequent context, with performance degrading as the number of intervening updates increases. 

---
# Estimating Item Difficulty with Large Language Models as Experts 

**Authors**: Diana Kolesnikova, Kirill Fedyanin, Abe D. Hofman, Matthieu J. S. Brinkhuis, Maria Bolsinova  

**Link**: [PDF](https://arxiv.org/pdf/2605.18562)  

**Abstract**: Accurate estimates of item difficulty are essential for valid assessment and effective adaptive learning. However, for newly created tasks, response data are typically unavailable. Pretesting and expert judgement can be costly and slow, while machine learning methods often require large labelled training datasets. Recent work suggests that large language models (LLMs) may help. However, there is limited evidence on the elicitation procedures and prompt configurations used to emulate experts for difficulty estimation. This study addresses this gap by evaluating three off-the-shelf LLMs as difficulty raters for newly created items without access to response data. Using an item bank from an online learning system, the study examined 6 domains of primary-school mathematics, with empirical difficulty estimates treated as empirical reference. The study used a full factorial design crossing three factors: judgement format (absolute vs pairwise), decision type (hard decisions vs token-probability-based estimates), and prompting strategy (zero-shot vs few-shot). LLM-derived difficulty estimates were compared with empirical difficulties using Spearman rank correlations. Across domains, LLM-based estimates exhibited moderate to strong positive correlations with empirical item difficulties. For simpler arithmetic tasks, some configurations approached the upper end of the accuracy range reported for human experts in previous research. Pairwise comparison consistently outperformed absolute judgement in the absence of additional refinements. However, when token-level probabilities were incorporated and examples of items with known empirical difficulty were provided, the absolute judgement configuration likewise demonstrated moderate-to-high alignment. The study positions LLMs as a promising tool for initial item calibration and offers insights into effective workflow configuration. 

---
# Improving BM25 Code Retrieval Under Fixed Generic Tokenization: Adaptive q-Log Odds as a Drop-In BM25 Fix 

**Authors**: Santosh Kumar Radha, Oktay Goktas  

**Link**: [PDF](https://arxiv.org/pdf/2605.18561)  

**Abstract**: In retrieval-augmented coding, failures often begin when the relevant file is absent from the retrieved context. Under frozen generic tokenization, where a BM25 index has been built by a search system whose analyzer the practitioner does not control, this failure is routine: BM25's logarithmic RSJ-odds IDF under-separates the identifier tail that distinguishes one function from another. We replace the outer logarithm of the Robertson-Spärck-Jones odds with a q-logarithm. At q=1 the transform recovers BM25 exactly by L'Hôpital's rule, and for q<1 it is a Box-Cox transform of the RSJ odds with lambda = 1-q. On CoIR CodeSearchNet Go (182K documents), oracle-tuned NDCG@10 rises from 0.2575 to 0.4874 (absolute +0.2299; +89.3% relative; zero sign reversals in 10,000 paired-bootstrap resamples, reported as p <= 10^-4). The effect is graded across code languages and is near-zero on BEIR text. A one-parameter closed form estimates a corpus-level q from hapax density and stays near q=1 on corpora where BM25 is already optimal. The index-time cost is a single pass over the sparse score matrix and query latency is unchanged. A tokenizer ablation shows that identifier-aware tokenization largely removes the incremental gain from q-IDF. 

---
# Key-Gram: Extensible World Knowledge for Embodied Manipulation 

**Authors**: Jingjing Fan, Siyuan Li, Botao Ren, Zhidong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.18556)  

**Abstract**: Embodied control increasingly requires models to follow compositional language instructions while reasoning over dynamic visual states. However, current vision-language-action policies and world-action models often couple linguistic knowledge with visual computation in a shared backbone or conditioning pathway, leading to modality competition and making knowledge extension dependent on backbone updates. In this paper, we introduce Key-Gram, a conditional-memory framework that separates language-derived world knowledge from visual-state reasoning for embodied control. At its core is a memory module that decomposes an instruction into task-specific key-grams, retrieves static linguistic priors through deterministic hashed lookup, and injects the retrieved entries into selected hidden layers through context-aware gating and lightweight convolutional fusion. This design allows the backbone to devote its main capacity to visual reasoning and action inference, while reusable instruction knowledge is stored in an extensible external memory. The logical memory table can be conveniently partitioned during training and, due to its $O(1)$ lookup pattern, efficiently placed on host memory during inference. Across RoboTwin2.0, LIBERO/LIBERO-Plus, and real-world dual-arm manipulation, Key-Gram consistently improves both $\pi_{0}$ and $\pi_{0.5}$ backbones, with average relative gains of $29.5\%/9.9\%$ on RoboTwin2.0, $35.8\%/4.5\%$ on LIBERO-Plus transfer without target-domain fine-tuning, and $15.4\%/8.1\%$ on real-world long-horizon tasks. These results demonstrate that externalized linguistic memory provides an effective and extensible mechanism for improving compositional grounding, transfer, and real-world manipulation. 

---
# StableHand: Quality-Aware Flow Matching for World-Space Dual-Hand Motion Estimation from Egocentric Video 

**Authors**: Huajian Zeng, Chaohua Yao, Yuantai Zhang, Jiaqi Yang, Rolandos Alexandros Potamias, Xingxing Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2605.18553)  

**Abstract**: Recovering world space 4D motion of two interacting hands from egocentric video is a fundamental capability for supervising robot policy learning, where wrist trajectories track the end-effector and finger articulations specify the grasp pose. Two major challenges arise in this setting: hands frequently leave the camera view for extended periods due to head motion, and persistent hand-object interactions cause severe occlusions of one or both hands. Existing methods uniformly condition on noisy hand motion observations without accounting for their per-frame reliability, leading to substantial performance degradation. Our key insight is that accurate world space hand motion estimation is tightly coupled with the quality of per-frame hand observations. To this end, we decompose the quality of hand motion observations extracted from an off-the-shelf hand pose estimator into four channels: wrist global translation and finger articulations for both hands. We propose StableHand, a quality-aware flow-matching framework conditioned on these four-channel quality signals, which are predicted by a learned quality network. We naturally incorporate the quality signals into the flow-matching process through a per-channel forward schedule, a quality-adjusted velocity target, AdaLN modulation of the DiT denoiser, and a quality-aware ODE initialization. This unified generative process preserves high-quality observations while reconstructing unreliable ones using a learned bimanual motion prior. Experiments on HOT3D and ARCTIC, two egocentric benchmarks featuring long missing-hand spans and persistent hand-object occlusions, show that StableHand achieves state-of-the-art performance across all reported metrics, reducing W-MPJPE by 20-25% compared to the strongest baseline, with the largest gains on heavily occluded ARCTIC sequences. 

---
# STT-Arena: A More Realistic Environment for Tool-Using with Spatio-Temporal Dynamics 

**Authors**: Tingfeng Hui, Hao Xu, Pengyu Zhu, Hongsheng Xin, Kun Zhan, Sen Su, Chunxiao Liu, Ning Miao  

**Link**: [PDF](https://arxiv.org/pdf/2605.18548)  

**Abstract**: Large language models (LLMs) deployed in real-world agentic applications must be capable of replanning and adapting when mid-task disruptions invalidate their prior decisions. Existing dynamic benchmarks primarily measure whether LLMs can detect temporal changes in a timely manner, leaving the complementary challenge of adaptive replanning under spatio-temporal dynamics largely unexplored. We introduce STT-Arena (Spatio-Temporal Tool-Use Arena), a benchmark of 227 high-quality interactive tasks spanning nine spatio-temporal conflict types and four solvability levels. Each task is grounded in a realistic, executable environment equipped with injected spatio-temporal triggers that can abruptly invalidate an ongoing plan, forcing the model to detect the state shift and construct a revised execution strategy. Extensive evaluation of frontier LLMs reveals that even the SOTA proprietary models, including Claude-4.6-Opus, achieves less than 40\% overall accuracies, highlighting the fundamental difficulty of spatio-temporal dynamic reasoning. Systematic analysis of failure trajectories uncovers three recurring error modes of existing models: Stale-State Execution, Misdiagnosis of Dynamic Triggers, and Missing Post-Adaptation Verification. Guided by these findings, we propose an iterative trajectory refinement technique that eliminates these failure patterns from training data, and combine it with online RL to produce STT-Agent-4B which outperforms frontier LLMs on STT-Arena. 

---
# Probing for Representation Manifolds in Superposition 

**Authors**: Alexander Modell  

**Link**: [PDF](https://arxiv.org/pdf/2605.18537)  

**Abstract**: This paper introduces the Manifold Probe, a supervised method for discovering representation manifolds in superposition. The method generalizes linear regression probes by learning the space of features of a concept that can be linearly predicted from the representations, and then learning the directions used to encode them. We demonstrate the probe on representations of time and space in Llama 2-7b, finding manifolds which linearly represent an interpretable set of features in each case. In the case of time, we show that by steering along the manifold, we can influence the model's completions about the years in which famous songs, movies and books were released, providing evidence that the Manifold Probe can discover manifolds which are causally involved in model behaviour. 

---
# Continuous Diffusion Scales Competitively with Discrete Diffusion for Language 

**Authors**: Zhihan Yang, Wei Guo, Shuibai Zhang, Subham Sekhar Sahoo, Yongxin Chen, Arash Vahdat, Morteza Mardani, John Thickstun  

**Link**: [PDF](https://arxiv.org/pdf/2605.18530)  

**Abstract**: While diffusion has drawn considerable recent attention from the language modeling community, continuous diffusion has appeared less scalable than discrete approaches. To challenge this belief we revisit Plaid, a likelihood-based continuous diffusion language model (DLM), and construct RePlaid by aligning the architecture of Plaid with modern discrete DLMs. In this unified setting, we establish the first scaling law for continuous DLMs that rivals discrete DLMs: RePlaid exhibits a compute gap of only $20\times$ compared to autoregressive models, outperforms Duo while using fewer parameters, and outperforms MDLM in the over-trained regime. We benchmark RePlaid against recent continuous DLMs: on OpenWebText, RePlaid achieves a new state-of-the-art PPL bound of $22.1$ among continuous DLMs and superior generation quality. These results suggest that continuous diffusion, when trained via likelihood, is a highly competitive and scalable alternative to discrete DLMs. Moreover, we offer theoretical insights to understand the advantage of likelihood-based training. We show that optimizing the noise schedule to minimize the ELBO's variance naturally yields linear cross-entropy (information loss) over time. This evenly distributes denoising difficulty without any case-specific time reparameterization. In addition, we find that optimizing embeddings via likelihood creates structured geometries and drives the most significant likelihood gain. 

---
# Beyond Morphology: Quantifying the Diagnostic Power of Color Features in Cancer Classification 

**Authors**: Farnaz Kheiri, Shahryar Rahnamayan, Masoud Makrehchi  

**Link**: [PDF](https://arxiv.org/pdf/2605.18522)  

**Abstract**: In histopathology, human experts primarily rely on color as a means of enhancing contrast to interpret tissue morphology, whereas machine vision models process color as raw statistical information. This distinction raises a fundamental question: to what extent can pixel intensity alone, independent of structural and morphological cues, support cancer classification? To address this question, we systematically evaluated the standalone discriminative power of global color features while deliberately excluding all morphological information. Specifically, we extracted statistical color moments and discretized RGB and HSV color histograms, and assessed their performance across ten diverse experimental settings using classical machine learning classifiers. Our results demonstrate that color features alone can achieve strong performance in binary diagnostic tasks (e.g., benign versus malignant), with classification accuracies reaching up to 89%. This performance is likely attributable to global chromatic shifts associated with malignancy. Importantly, these simple color-based representations consistently outperformed random baselines by a substantial margin, indicating that raw color distributions encode a non-random and diagnostically relevant signal for cancer detection. Consequently, this study suggests that simple, computationally efficient color features can serve as an effective pre-screening tool. By identifying samples with strong chromatic indicators of malignancy, these lightweight models could function as a first-pass triage system, reducing the computational burden on complex deep learning architectures. 

---
# DiPRL: Learning Discrete Programmatic Policies via Architecture Entropy Regularization 

**Authors**: Chengpeng Hu, Yingqian Zhang, Hendrik Baier  

**Link**: [PDF](https://arxiv.org/pdf/2605.18508)  

**Abstract**: Programmatic reinforcement learning (PRL) offers an interpretable alternative to deep reinforcement learning by representing policies as human-readable and -editable programs. While gradient-based methods have been developed to optimize continuous relaxations of programs, they face a significant performance drop when converting the continuous relaxations back into discrete programs. Post-hoc discretization can discard optimized branches and parameters in a program, which results in a collapse of policy expressivity and lowered task performance, leading in turn to a need for additional fine-tuning. To overcome these limitations, we propose Differentiable Discrete Programmatic Reinforcement Learning (DiPRL), a method that learns programmatic policies that become nearly discrete during training, avoiding a separate post-hoc fine-tuning stage. We first analyze the inherent risks of performance drop introduced by post-hoc discretization of gradient-based methods. Then, we introduce programmatic architecture entropy regularization, which enables smooth, differentiable training that encourages convergence toward a discrete program. DiPRL maintains the efficiency of gradient-based optimization while mitigating the risks of post-hoc discretization. Our experiments across multiple discrete and continuous RL tasks demonstrate that DiPRL can achieve strong performance via interpretable programmatic policies. 

---
# DBES: A Systematic Benchmark and Metric Suite for Evaluating Expert Specialization in Large-Scale MoEs 

**Authors**: Jing Wang, Hongxuan Lu, Jazze Young, Shu Wang, Zhimin Xin  

**Link**: [PDF](https://arxiv.org/pdf/2605.18498)  

**Abstract**: Expert specialization in Mixture-of-Experts (MoE) models remains poorly understood, with traditional evaluations conflating architectural load-balancing with functional specialization. We introduce DBES, a comprehensive diagnostic framework combining a multi-domain benchmark with five theoretically grounded metrics: Routing Specialization, Normalized Effective Rank, Domain Isolation, Routing Stiffness Score, and N-gram Expertise measures.
Critical findings demonstrate distinct specialization paradigms across models: Qwen-series exhibit modular specialization with high domain isolation, while DeepSeek and GLM employ distributed collaboration. However, we emphasize that specialization is a diagnostic dimension, necessary but not sufficient for downstream performance. Most crucially, interventional evidence validates the actionability of these metrics: by using DBES to identify high-specialization expert paths during domain-specific post-training, we achieved 66% to 94.48% improvement in specialized domains with only 15% of original training resources, demonstrating that these diagnostic tools can be converted into concrete optimization operators. This work provides the first systematic methodology for evaluating expert specialization independently of accuracy metrics, offering crucial insights for the design and post-training optimization of next-generation MoE systems. 

---
# Modality vs. Morphology: A Framework for Time Series Classification for Biological Signals 

**Authors**: Jordan Tschida, Matthew Yohe, Edward Kane, Gavin Jager, Emma J. Reid, Tony G. Allen, Mark Story, Leanne Thompson, Joe Hoskins, Brandon Schreiber, Stan Seiferth, Scott Dolvin, David Cornett  

**Link**: [PDF](https://arxiv.org/pdf/2605.18483)  

**Abstract**: Time series classification (TSC) of biological signals has progressed from handcrafted, modality-specific approaches to deep architectures capable of representing the diverse waveform structures of underlying physiological processes (i.e., morphology). This review introduces a unified morphology--modality framework that connects waveform structure to a methodological design, revealing how spikes, bursts, oscillations, slow drift, and hierarchical rhythms inform model design. By analyzing electroencephalography, electromyography, electrocardiography, photoplethysmography, and ocular modalities (electrooculography, pupillometry, eye-tracking), the review demonstrates how morphology determines preprocessing and modeling strategies. Integrating evidence across these biological signals, the framework reveals that morphology, not model class, most strongly determines performance and interpretability. This provides insight into why deep models succeed when their inductive biases align with underlying waveform dynamics. This review also identifies future work including morphological data augmentation and evaluation metrics to improve generalization. Together, these insights position morphology-aware modeling as a unifying principle for developing generalizable, interpretable, and physiologically meaningful TSC models across biological signals. 

---
# AI4BayesCode: From Natural Language Descriptions to Validated Modular Stateful Bayesian Samplers 

**Authors**: Jungang Zou, Alex Ziyu Jiang, Qixuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.18476)  

**Abstract**: Coding and computation remain major bottlenecks in Markov chain Monte Carlo (MCMC) workflows, especially as modern sampling algorithms have become increasingly complex and existing probabilistic programming systems remain limited in model support, extensibility, and composability. We introduce \textbf{AI4BayesCode}, an extensible LLM-driven system that translates natural-language Bayesian model descriptions into runnable, validated MCMC samplers. To improve reliability, AI4BayesCode adopts a modular design that decomposes models into modular sampling blocks and maps each block to a built-in sampling component, reducing the need to implement complex sampling algorithms from scratch. Reliability is further improved through pre-generation validation of model specifications and post-generation validation of generated sampler code. AI4BayesCode also introduces a novel recursively stateful coding paradigm for MCMC, allowing modular sampling components, potentially developed by different contributors, to be composed coherently within larger MCMC procedures. We develop a benchmark suite to evaluate AI4BayesCode for sampler-generation. Experiments show that AI4BayesCode can implement a wide range of Bayesian models from natural-language descriptions alone. As an open-ended system, its capability can continue to expand with improvements in the underlying AI agent and the addition of new built-in blocks. 

---
# GAMMA: Global Bit Allocation for Mixed-Precision Models under Arbitrary Budgets 

**Authors**: Zhangyang Yao, Haiyan Zhao, Haoyu Wang, Tianbo Huang, Lihua Zhang, Xu Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.18475)  

**Abstract**: Mixed-precision quantization improves the budget--accuracy trade-off for large language models (LLMs) by allocating more bits to sensitive modules. However, automating this allocation at LLM scale faces a unique combination of constraints: learnable approaches require quantization-aware training, which is infeasible for billion-parameter models; training-free alternatives rely on static proxy metrics that miss cross-module interactions and must be recomputed per target budget; and search-based methods are expensive without guaranteeing exact budget compliance. We propose GAMMA, a quantizer-agnostic framework that learns module-wise precision preferences entirely within a post-training pipeline. GAMMA optimizes a teacher-forced hidden-state reconstruction objective under an augmented Lagrangian constraint, and projects the learned preferences into exact budget-feasible discrete assignments via integer programming. A key property is score reuse: because the learned preferences encode a stable sensitivity ranking rather than budget-specific weights, a single training run serves arbitrary deployment targets by re-solving only the integer program, reducing per-budget adaptation from hours to a few minutes. Across Llama and Qwen models (8B--32B), GAMMA outperforms both fixed-precision baselines (up to +12.99 Avg.) and search-based mixed-precision methods (up to +7.00 Avg.), and can match fixed 3-bit quality at 2.5-bit average precision, enabling deployment at substantially smaller memory footprints. 

---
# Prompt2Fingerprint: Plug-and-Play LLM Fingerprinting via Text-to-Weight Generation 

**Authors**: Sixu Chen, Xiang Chen, Hongyao Yu, Jiaxin Hong, Hao Fang, Shuoyang Sun, Bin Chen, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2605.18474)  

**Abstract**: The widespread deployment and redistribution of large language models (LLMs) have made model provenance tracking a critical challenge. While existing LLM fingerprinting methods, particularly active approaches that embed identity signals via fine-tuning, achieve high accuracy and robustness, they suffer from significant scalability bottlenecks. These methods typically treat fingerprint injection as an independent, one-off optimization task rather than a reusable capability, necessitating separate, resource-intensive training for every new identity. This incurs prohibitive computational costs and deployment delays. To address this, we propose Prompt2Fingerprint (P2F), the first framework that reformulates fingerprinting as a conditional parameter generation task. By leveraging a specialized generator, P2F maps textual descriptions directly to low-rank parameter increments in a single forward pass, enabling plug-and-play LLM fingerprint injection without further model retraining. Our experiments demonstrate that P2F maintains high fingerprint accuracy, harmlessness, and robustness while significantly reducing computational overhead, offering a scalable and instant solution for LLM ownership management. 

---
# Flowing with Confidence 

**Authors**: Friso de Kruiff, Dario Coscia, Max Welling, Erik Bekkers  

**Link**: [PDF](https://arxiv.org/pdf/2605.18472)  

**Abstract**: Generative models can produce nonsensical text, unrealistic images, and unstable materials faster than simulation or human review can absorb; without per-sample confidence, trust erodes. Existing fixes run $k$ ensembles or stochastic trajectories at $k\times$ compute, measuring variability between models, not model confidence. We propose Flow Matching with Confidence (FMwC). FMwC injects input-dependent multiplicative noise at selected layers, propagates its variance through the network in closed form, and integrates it along the ODE trajectory, yielding a per-sample confidence score at standard sampling cost. The score supports multiple uses: filtering improves image quality and thermodynamic stability of crystals; editing rewinds trajectories to the points where the model commits and redirects them; and adaptive stepping concentrates ODE compute where the flow is ambiguous. We find that the confidence score correlates with the magnitude of the divergence of the learned velocity field, which gives us a window to understand the generative process, opening up surgical forms of guidance that target the moments that matter, new sampling algorithms and interpretability of generative models. 

---
# Scheduling That Speaks: An Interpretable Programmatic Reinforcement Learning Framework 

**Authors**: Chengpeng Hu, Yingqian Zhang, Hendrik Baier  

**Link**: [PDF](https://arxiv.org/pdf/2605.18454)  

**Abstract**: Deep reinforcement learning (DRL) has recently emerged as a promising approach to solve combinatorial optimization problems such as job shop scheduling. However, the policies learned by DRL are typically represented by deep neural networks (DNNs), whose opaque neural architectures and non-interpretable policy decisions can lead to critical trust and usability concerns for human decision makers. In addition, the computational requirements of DNNs can further hinder practical deployment in resource constrained environments. In this work, we propose ProRL, a novel interpretable programmatic reinforcement learning framework that achieves high-performance scheduling with human-readable and editable programmatic policies (i.e., programs). We first introduce a domain-specific language for scheduling (DSL-S) to represent scheduling strategies as structured programs. ProRL then explores the program space defined by DSL-S using local search to identify incomplete programs, which are subsequently completed by learning their parameters via Bayesian optimization. ProRL learns which scheduling heuristic rules to select, and hence, it naturally incorporates existing heuristics already used in industrial scenarios. Experiments on widely used benchmark instances demonstrate the strong performance of ProRL against existing heuristics and DRL baselines. Furthermore, ProRL performs well under strongly constrained computational resources, such as training with only 100 episodes. Our code is available at this https URL. 

---
# Modelling Customer Trajectories with Reinforcement Learning for Practical Retail Insights 

**Authors**: Ken Ming Lee, Paul Barde, Maxime C. Cohen, Derek Nowrouzezahrai  

**Link**: [PDF](https://arxiv.org/pdf/2605.18449)  

**Abstract**: Understanding customer movement within retail spaces is essential for optimizing store layouts. Real-world trajectory data can provide highly accurate insights, but collecting it is costly and often infeasible for many retailers. Heuristics such as Travelling Salesman Problem (TSP) and Probabilistic Nearest Neighbours (PNN) are commonly used as inexpensive approximations, but actual customer trajectories deviate by an average of 28% from shortest paths, highlighting a tradeoff between accuracy and practicality. We propose an agent-based modelling framework that casts customer trajectory prediction as a maximum entropy reinforcement learning (RL) problem, balancing reward maximization with stochasticity to better reflect customers with bounded rationality. Using real-world trajectory data from a convenience store, we show that RL-generated trajectories align more closely with customer behaviour than TSP and PNN, providing more accurate estimates of impulse purchase rates and shelf traffic densities. Furthermore, only RL-based predictions yield repositioning decisions for impulse products that align with those derived from actual trajectory data, resulting in comparable estimated profit gains. Our work demonstrates that RL provides a practical, behaviourally grounded alternative that bridges the gap between oversimplified heuristics and data-intensive approaches, making accurate layout optimization more accessible. To encourage further research, the source code is available on GitHub. 

---
# What is Holding Back Latent Visual Reasoning? 

**Authors**: André G. Viveiros, Nuno Gonçalves, André F. T. Martins, Matthias Lindemann  

**Link**: [PDF](https://arxiv.org/pdf/2605.18445)  

**Abstract**: Humans can approach complex visual problems by mentally simulating intermediate visual steps, rather than reasoning through language alone. Inspired by this, several works on Vision-Language Models have recently explored chain-of-thought reasoning with continuous latent tokens as intermediate visual imagination steps. In this work, we investigate how recent models leverage such latent tokens. Surprisingly, we find that model accuracy is unaffected when latent tokens are replaced by uninformative ``dummy'' tokens. This indicates that latent tokens play a minimal causal role in the model's final prediction. To better understand this phenomenon, we analyze both the training signal provided by oracle latent representations and the quality of the latent tokens generated at inference time. Our experiments reveal two crucial issues holding back latent visual reasoning: First, in most existing datasets, oracle latent tokens provide limited additional information beyond the original image and do not substantially simplify the task, leading models to ignore them during training and effectively bypassing them at inference time. When fine-tuned on a diagnostic dataset, in which latent tokens provide sufficient support for the final prediction, we show that models can causally rely on them. Second, the latent tokens produced at inference time deviate from their corresponding oracle representations, collapsing to a narrow region and preventing benefits even when the model relies on them. Overall, our findings suggest that future progress in latent visual reasoning depends on two key pillars: high-quality datasets with informative intermediate steps and more precise latent token prediction. 

---
# Building Reliable Arithmetic Multipliers Under NBTI Aging and Process Variations 

**Authors**: Masoud Heidary, Biresh Kumar Joardar  

**Link**: [PDF](https://arxiv.org/pdf/2605.18444)  

**Abstract**: Hardware aging poses a significant challenge for integrated circuits (ICs), leading to performance degradation and eventual failure. In this work, we focus on the aging of arithmetic multipliers, which are a cornerstone of modern computing systems including in CPUs, GPUs, and FPGAs, as well as AI accelerators like systolic arrays. In particular, AI workloads, which rely predominantly on multiplications, can accelerate Negative Bias Temperature Instability (NBTI) effects in multipliers. This paper presents a novel aging mitigation technique that leverages the signinvariance property of multiplication. By selectively applying 2s complement transformations to inputs, the method redistributes stress across transistors, reducing the effects of NBTI aging. The proposed method is also integrated into systolic arrays, a common AI accelerator, to demonstrate its efficiency in a high-throughput AI accelerator. Experimental evaluations using Cadence tools show better lifetime compared to natural aging (with no mitigation) baseline, while introducing negligible area and delay overheads. 

---
# EvoMemBench: Benchmarking Agent Memory from a Self-Evolving Perspective 

**Authors**: Yuyao Wang, Zhongjian Zhang, Mo Chi, Kaichi Yu, Yuhan Li, Miao Peng, Bing Tong, Chen Zhang, Yan Zhou, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.18421)  

**Abstract**: Recent benchmarks for Large Language Model (LLM) agents mainly evaluate reasoning, planning, and execution. However, memory is also essential for agents, as it enables them to store, update, and retrieve information over time. This ability remains under-evaluated, largely because existing benchmarks do not provide a systematic way to assess memory mechanisms. In this paper, we study agent memory from a self-evolving perspective and introduce EvoMemBench, a unified benchmark organized along two axes: memory scope (in-episode vs. cross-episode) and memory content (knowledge-oriented vs. execution-oriented). We compare 15 representative memory methods with strong long-context baselines under a standardized protocol. Results show that current memory systems are still far from a general solution: long-context baselines remain highly competitive, memory helps most when the current context is insufficient or tasks are difficult, and no single memory form works consistently across all settings. Retrieval-based methods remain strong for knowledge-intensive settings, whereas procedural and long-term memory methods are more effective for execution-oriented tasks when their stored experience matches the task structure. We hope EvoMemBench facilitates future research on more effective memory systems for LLM-based agents. Our code is available at this https URL. 

---
# Geometry-Aware Uncertainty Coresets for Robust Visual In-Context Learning in Histopathology 

**Authors**: Franciskus Xaverius Erick, Johanna Paula Müller, Bernhard Kainz  

**Link**: [PDF](https://arxiv.org/pdf/2605.18419)  

**Abstract**: Vision-language models (VLMs) can couple visual perception with open-ended clinical reasoning, making them attractive for computational histopathology. However, fine-tuning billions of parameters on scarce, expert-annotated pathology data is prohibitive, while in-context learning (ICL), which conditions the VLM on demonstrative image-text pairs without parameter updates, suffers from high sensitivity to which examples are selected and how the query is phrased, producing unreliable diagnostics. Existing selection strategies rely on query-dependent nearest-neighbour retrieval that ignores global data structure, require costly parameter updates, or disregard the joint vision-text embedding geometry of VLMs. We propose GAUC, a training-free coreset selection method operating directly in the pre-trained multimodal embedding space. GAUC jointly optimises three objectives: (1) a Maximum Mean Discrepancy term enforcing distributional fidelity between coreset and full dataset, (2) an Effective Mutual Information Difference regulariser bounding performance degradation under prompt paraphrases by exploiting the VLM's joint vision-text alignment, and (3) a predictive-variance penalty suppressing overconfident, unstable outputs. On CRC-100K and MHIST across multiple open-source VLM architectures, GAUC consistently improves accuracy, calibration, and prompt robustness over recent ICL selection methods and dataset-distillation baselines, all without a single gradient update. 

---
# Prompts Don't Protect: Architectural Enforcement via MCP Proxy for LLM Tool Access Control 

**Authors**: Rohith Uppala  

**Link**: [PDF](https://arxiv.org/pdf/2605.18414)  

**Abstract**: Large language models increasingly operate as autonomous agents that select and invoke tools from large registries. We identify a critical gap: when unauthorized tools are visible in an agent's context, models select them in adversarial scenarios -- even when explicitly instructed otherwise. We propose a governed MCP proxy that enforces attribute-based access control (ABAC) at two points: tool discovery, where unauthorized tools are removed from the model's context window, and tool invocation, where a second check blocks any unauthorized call. Across three models (Qwen 2.5 7B, Llama 3.1 8B, Claude Haiku 3.5) and 150 adversarial tasks spanning four attack categories, our proxy reduces unauthorized invocation rate (UIR) to 0% while adding under 50ms median latency. Prompt-based restrictions reduce UIR by only 11--18 percentage points, leaving substantial residual risk. Our results show that architectural enforcement -- not prompting -- is necessary for reliable tool access control in deployed agentic systems. 

---
# Qumus: Realization of An Embodied AI Quantum Material Experimentalist 

**Authors**: Lihan Shi, Zhaoyi Joy Zheng, Xinzhe Juan, Yimin Wang, Ming Yin, Mayank Sengupta, Kristina Wolinski, Yanyu Jia, Jingzhi Shi, Derek Saucedo, Neill Saggi, Haosen Guan, Kenji Watanabe, Takashi Taniguchi, Ali Yazdani, Mengdi Wang, Sanfeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18407)  

**Abstract**: While modern Large Language Models (LLMs) and agentic artificial intelligence (AI) have demonstrated transformative capabilities in digital domains, the realization of embodied AI capable of real-world scientific discovery remains a difficult frontier. The advancements are hindered by the inherent complexity of integrating high-level reasoning, multimodal information processing and real-time physical execution. Here we introduce Qumus, the first AI quantum materials experimentalist. Physically embodied within a robotic mini-laboratory, Qumus is an intelligent, multimodal, and multi-agent system designed for the creation and nano-processing of atomically thin two-dimensional (2D) materials and stacked van der Waals (vdW) structures. Qumus autonomously navigates the full scientific cycle, from hypothesis generation and protocol planning to multi-step experimental execution, result analysis and reporting, acting as an experimentalist. Markedly, the system has achieved, for the first time, the AI-creation of graphene, as well as the first AI-fabrication of complex nanodevices including atomically thin field-effect transistors via vdW stacking. Qumus excels at these tasks by demonstrating autonomous error correction and closed-loop experimentation. Our results establish a generalizable framework for self-improving embodied AI systems that learn directly from the quantum world, opening a pathway toward accelerated discovery in quantum materials, electronics and beyond. 

---
# SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution 

**Authors**: Hongyi Liu, Haoyan Yang, Tao Jiang, Bo Tang, Feiyu Xiong, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.18401)  

**Abstract**: Long-horizon LLM agents leave traces that could become reusable experience, but raw trajectories are noisy and hard to govern. We treat Agent Skills as an experience schema that couples executable scripts, with non-executable guidance on procedures. Yet open skill ecosystems contain redundant, uneven, environment-sensitive artifacts, and indiscriminate updates can pollute future context. We present SkillsVote, a lifecycle-governance framework for Agent Skills from collection and recommendation to evolution. SkillsVote profiles a million-scale open-source corpus for environment requirements, quality, and verifiability, then synthesizes tasks for verifiable skills. Before execution, SkillsVote performs agentic library search over structured skill library to expose instructional skill context. After execution, it decomposes trajectories into skill-linked subtasks, attributes outcomes to skill use, agent exploration, environment, and result signals, and admits only successful reusable discoveries to evidence-gated updates. In our evaluation, offline evolution improves GPT-5.2 on Terminal-Bench 2.0 by up to 7.9 pp, while online evolution improves SWE-Bench Pro by up to 2.6 pp. Overall, governed external skill libraries can improve frozen agents without model updates when systems control exposure, credit, and preservation. 

---
# Diagnosing Korean-Language LLM Political Bias via Census-Grounded Agent Simulation 

**Authors**: Sungwoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18395)  

**Abstract**: Large language models (LLMs) exhibit systematic political biases in voter simulations, but their underlying mechanisms and cross-lingual generalizations remain poorly understood. We introduce Dynamo-K, a census-grounded simulation framework evaluating Korean-language LLM political behavior across four models on six Korean elections (2017-2025). Using this framework, we identify three systematic failure modes: (1) progressive bias in moderate agents, where explicit mitigation reduces Mean Absolute Error (MAE) by 5.2 times; (2) model-dependent third-party salience collapse, distinguishing between salience failure and decision bias; and (3) regional polarization collapse, where models bidirectionally under-predict historical party strongholds.
To address these failures, we demonstrate that scenario reframing recovers 62% of 2017 MAE by restoring third-party visibility. Furthermore, we introduce a learned reweighting adapter that successfully calibrates opposing-valence models without relying on candidate names at train or test time. Validating our diagnostic framework, Dynamo-K accurately predicts 3/3 presidential winners - including a 2.1%p MAE on the highly contested 0.73%p-margin 2022 race - and correctly identifies the dominant party in a held-out local election. The pipeline is open-source and provides a scalable, cost-effective method for diagnosing LLM political behavior. 

---
# Graph Hierarchical Recurrence for Long-Range Generalization 

**Authors**: Stefano Carotti, Marco Pacini, Alessio Gravina, Davide Bacciu, Bruno Lepri, Sebastiano Bontorin  

**Link**: [PDF](https://arxiv.org/pdf/2605.18387)  

**Abstract**: Graph Neural Networks (GNNs) and Graph Transformers (GTs) are now a fundamental paradigm for graph learning, combining the representation-learning capabilities of deep models with the sample efficiency induced by their inductive biases. Despite their effectiveness, a large body of work has shown that these models still face fundamental limitations in tasks that require capturing correlations between distant regions of a graph. To address this issue, we introduce Graph Hierarchical Recurrence (GHR), a novel framework that operates jointly on the input graph and on a hierarchical abstraction obtained through pooling. We also show that the limitations of existing models are even more pronounced in out-of-range generalization, where test instances involve interactions over distances longer than those observed during training. By contrast, despite its simple design, GHR provides three key advantages: strong performance on long-range dependencies, improved out-of-range generalization, and high parameter efficiency. To corroborate these claims, we show that across a broad set of long-range benchmarks, GHR consistently outperforms existing graph models while using as little as 1% of the parameters of current state-of-the-art models. These results suggest a complementary direction to the current trend of scaling architectures to obtain graph foundation models, indicating that increased model capacity alone may not be sufficient for generalization. 

---
# Towards Ubiquitous Mapping and Localization for Dynamic Indoor Environments 

**Authors**: Halim Djerroud, Nico Steyn, Olivier Rabreau, Patrick Bonnin, Abderraouf Benali  

**Link**: [PDF](https://arxiv.org/pdf/2605.18385)  

**Abstract**: We present UbiSLAM, an innovative solution for real-time mapping and localization in dynamic indoor environments. By deploying a network of fixed RGB-D
cameras strategically throughout the workspace, UbiSLAM addresses limitations commonly encountered in traditional SLAM systems, such as sensitivity to
environmental changes and reliance on mobile unit sensors. This fixed-sensor approach enables real-time, comprehensive mapping, enhancing the localization
accuracy and responsiveness of robots operating within the environment. The centralized map generated by UbiSLAM is continuously updated, providing robots
with an accurate global view, which improves navigation, minimizes collisions, and facilitates smoother human-robot interactions in shared spaces. Beyond its
advantages, UbiSLAM faces challenges, particularly in ensuring complete spatial coverage and managing blind spots, which necessitate data integration from
the robots themselves. In this paper we discuss potential solutions, such as automatic calibration for optimal camera placement and orientation, along with
enhanced communication protocols for real-time data sharing. The proposed model reduces the computational load on individual robotic units, allowing less
complex robotic platforms to operate effectively while enhancing the robustness of the overall system. 

---
# Probing SMEFT Operators through $t\bar{t}t\bar{t}$ Production with Hyper-Graph Neural Networks at the LHC 

**Authors**: Amir Subba, Sanmay Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2605.18382)  

**Abstract**: We present a phenomenological study of $t\bar{t}t\bar{t}$ production in proton-proton collisions at $\sqrt{s} = 13$~TeV, using a Hyper-Graph Neural Network (H-GNN) to discriminate multilepton signal events from the dominant SM backgrounds, namely $t\bar{t}W$, $t\bar{t}Z$, $t\bar{t}H$, $t\bar{t}VV$, single-top associated production, and diboson and triboson processes. In the H-GNN architecture each event is represented as a hypergraph whose nodes correspond to reconstructed jets and leptons and whose hyperedges encode higher-order correlations among arbitrary subsets of these objects, allowing the network to learn the many-body kinematic structures that characterize the $t\bar{t}t\bar{t}$ final state. Combining same-sign di-lepton, tri-lepton, and four-lepton channels following a CMS-like event selection, the H-GNN attains an area under the ROC curve of $0.951$ for the $t\bar{t}t\bar{t}$ signal and yields a statistical significance of $Z = 9.11$ at an integrated luminosity of $\mathcal{L} = 140~\mathrm{fb}^{-1}$, to be compared with $Z = 8.62$ for a SPANet baseline, $Z = 7.37$ for a Particle Transformer baseline, and $Z = 5.13$ obtained by the ATLAS analysis, evaluated under identical event selection. We exploit the improved signal extraction to derive one- and two-parameter $95\%$ confidence level limits on the Wilson coefficients of the dimension-six operators $\mathcal{O}_{\Phi u}$, $\mathcal{O}^{(1)}_{tt}$, $\mathcal{O}^{(1)}_{qq}$, $\mathcal{O}^{(1)}_{qt}$, and $\mathcal{O}^{(8)}_{qt}$, and we project the expected sensitivity at the HL-LHC integrated luminosities of $1000~\mathrm{fb}^{-1}$ and $3000~\mathrm{fb}^{-1}$ with $50\%$ uncertainty on the background estimation. 

---
# Beyond Inference-Time Search: Reinforcement Learning Synthesizes Reusable Solvers 

**Authors**: Soheyl Massoudi, Gabriel Apaza, Milad Habibi, Mark Fuge  

**Link**: [PDF](https://arxiv.org/pdf/2605.18374)  

**Abstract**: Large language models (LLMs) typically approach combinatorial optimization as an inference-time procedure, solving each instance separately through sampling, search, or repeated prompting. We ask whether reinforcement learning can instead shift part of this reasoning cost into the weights of a code LLM, so that the model synthesizes a reusable solver for an entire problem family. We study this question on Synergistic Dependency Selection (SDS), a controlled variant of constrained Quadratic Knapsack designed to expose a specific failure mode: local signals and strict feasibility constraints make greedy heuristics attractive but unreliable. Under identical scaffolding, Best-of-64 base-model sampling saturates at an approximately 28.7% gap to the global Virtual Best Solver (VBS); code audits show that the base model often retrieves Simulated Annealing templates but misimplements the Metropolis acceptance rule. We fine-tune Qwen2.5-Coder-14B-Instruct with Group Relative Policy Optimization (GRPO) using a feasibility-gated reward and light structural scaffolding. The resulting policy converges to a constraint-aware Simulated Annealing template in 99.8% of feasible SDS outputs, achieves a 5.0% gap to that VBS, and is 91 times cheaper in post-generation execution/search cost than cumulative Best-of-64 evaluation. A compile-once check shows that one best frozen solver per seed remains highly competitive when reused unchanged across the SDS test set, while an additional-domain evaluation on Job Shop Scheduling provides narrower but positive evidence that the scaffold transfers beyond SDS. Negative ablations reveal the limits of this recipe: standard stabilizers degrade performance, a soft feasibility gate fails, and results remain sensitive to reward normalization and domain-specific design choices. 

---
# The Hidden Cost of Contextual Sycophancy: an AI Literacy Intervention in Human-AI Collaboration 

**Authors**: Cansu Koyuturk, Sabrina Guidotti, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2605.18372)  

**Abstract**: Large Language Models (LLMs) are increasingly used in educational settings as interactive tools for collaboration. However, their tendency toward sycophancy, aligning with user beliefs even when incorrect, raises concerns for learning and decision-making, especially for less knowledgeable users. This study investigates how sycophantic alignment emerges in authentic multi-turn human-AI interactions and whether interventions targeting increasing AI literacy and prompting competencies can mitigate its effects. In a controlled mixed-design experiment, 60 participants completed analytical survival ranking tasks by first generating individual rankings and then making final decisions after collaborating with an AI assistant, both before and after receiving either general or sycophancy-focused prompting training. Preliminary results show that LLMs are highly sensitive to user input: lower-quality initial responses lead to poorer AI advice, suggesting that the model mirrors or incorporates user reasoning rather than correcting it or offering better alternatives that are missing or less frequent in the conversation. Critically, the propagation of user errors into AI responses significantly reduced both the quality of AI feedback and final user task performance, revealing a form of contextual sycophantic dependence. While the intervention did not eliminate the propagation of contextual errors, it significantly improved AI advice by reducing the direct mirroring of incorrect user rankings. These findings suggest that prompting and AI literacy alone may be insufficient to ensure epistemically independent AI support, highlighting the need for system-level approaches that better promote critical engagement in human-AI collaboration. 

---
# Optimising CSRNet with parameter-free attention mechanisms for crowd counting in public transport 

**Authors**: Aida Rostamza, Enrico Del Re, Joshua Cherian Varughese, Cristina Olaverri-Monreal  

**Link**: [PDF](https://arxiv.org/pdf/2605.18349)  

**Abstract**: Occupancy estimation and crowd counting are critical tasks in designing smart and efficient public transport vehicles. Given that public transport loading can vary from sparse to crowded, classical models for occupancy estimation must be adapted to suit this purpose. Attention mechanisms have shown remarkable capability in enhancing the representational power of deep neural networks for crowd counting in congested scenes with occlusion, complex backgrounds, and perspective distortion. However, conventional approaches, often implemented as parameterized sub-networks within convolutional layers, inevitably increase model size and computational cost, limiting deployment on resource-constrained edge devices. This paper investigates the effectiveness of state-of-the-art parameter-free attention mechanisms for crowd counting and density map estimation in highly congested scenes. We evaluate channel-wise (PFCA), spatial-wise (SA), and 3-D (SimAM) modules and compare their performance with parameterized attention modules constrained to introduce no more than 1% additional parameters. Furthermore, we present a novel combination of attention mechanisms that combines the strengths of PFCA and SA (PFCASA) customized for analyzing video streams onboard public transport systems. Using CSRNet as the backbone, experiments on the ShanghaiTech dataset demonstrate that parameter-free attention mechanisms achieve comparable or superior accuracy without introducing additional model parameters. A detailed performance analysis further reveals that PFCASA outperforms other attention modules in scenes with fewer than 40 individuals, while PFCA shows greater effectiveness as crowd density increases, underscoring their potential applicability for integration into smart public transport modalities. 

---
# Focused Forcing: Content-Aware Per-Frame KV Selection for Efficient Autoregressive Video Diffusion 

**Authors**: Peiliang Cai, Evelyn Zhang, Jiacheng Liu, Hao Lin, Ruiqi Zhang, Weile Mo, Yue Ma, Shikang Zheng, Jiehang Huang, Dongrui Liu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18346)  

**Abstract**: Recent advances in autoregressive video diffusion have enabled sequential and streaming video generation. However, long-horizon generation requires increasingly large KV caches, making efficient compression without sacrificing quality challenging. Existing methods mostly select historical frames based on attention scores, but their context decisions remain coarse. When multiple frames are generated in the same chunk, these methods often apply a shared history selection to the whole chunk, score historical frames solely by attention, and assign head-wise budgets either uniformly or by attention-pattern heuristics rather than explicit head-importance estimation. We show that frames within the same generated chunk can depend on distinct historical frames, that the same historical frame can receive different attention scores as its relative temporal distance to the current frames changes, and that masking different heads induces unequal generation degradation. Motivated by these findings, we propose \textbf{Focused Forcing}, a training-free KV selection method that focuses cached history along both generated-frame and head dimensions. For each generated frame, Focused Forcing preserves the most relevant and distinctive historical frames by combining attention scores with diversity scores of historical frames, while assigning larger budgets to heads with higher estimated importance. Across multiple autoregressive generation paradigms, Focused Forcing achieves up to $\textbf{1.48}\times$ end-to-end acceleration without training, while \textbf{improving visual quality and text alignment}. \textit{Our code will be released on GitHub.} 

---
# Same Signal, Different Semantics: A Cross-Framework Behavioral Analysis of Software Engineering Agents 

**Authors**: Wei Ma, Zhi Chen, Jingxu Gu, Tianling Li, Shangqing Liu, Lingxiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18332)  

**Abstract**: Behavioral studies of LLM-based software engineering agents extract operational rules about which trajectory shapes correlate with higher resolution rates: that a test step follows a code modification, that error cascades are short, or that trajectories are compact. Each rule is typically derived from a single framework, and whether it transfers, in sign as well as magnitude, to structurally different agent designs has not been directly tested.
We address this at ecosystem scale: 64,380 SWE-bench runs from 126 agent configurations spanning 43 frameworks, where each configuration pairs an LLM with a framework (e.g., SWE-Agent, OpenHands) that supplies its tools and workflow. We separate framework effects from LLM effects by holding each layer fixed in turn, then measure one behavior-outcome effect per configuration and examine how those effects agree or disagree.
Swapping the framework while the LLM is held fixed produces large behavioral differences in every action feature. On most signals, configurations disagree not merely in magnitude but in direction. Error rate is the cleanest case: 47 configurations resolve more issues when their error rate is lower, while 48 resolve more when it is higher. Five other continuous features and three of seven binary patterns from prior SE literature show similar directional disagreement. Framework identity accounts for more of this variation than LLM family: for mean turns, framework explains 64% of the between-configuration variance against the LLM's 10%.
The implication is that the same observable behavioral signal can carry opposite meaning for different agent configurations. Behavioral findings from any single framework therefore warrant cross-configuration validation before being claimed as general. 

---
# Improved Baselines with Representation Autoencoders 

**Authors**: Jaskirat Singh, Boyang Zheng, Zongze Wu, Richard Zhang, Eli Shechtman, Saining Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.18324)  

**Abstract**: Representation Autoencoders (RAE) replace traditional VAE with pretrained vision encoders. In this paper, we systematically investigate several design choices and find three insights which simplify and improve RAE. First, we study a generalized formulation where the representation is defined as sum of the last k encoder layers rather than solely the final layer. This simple change greatly improves reconstruction without encoder finetuning or specialized data (e.g., text, faces). Second, we study the prevalent assumption that RAE (using pretrained representation as encoder) replaces representation alignment (REPA), which distills the same representation to intermediate layers instead. Through large-scale empirical analysis, we uncover a surprising finding: RAE and REPA exhibit complementary working mechanisms, allowing the same representation to be used as both encoder and target for intermediate diffusion layers. Finally, the original RAE struggles with classifier-free guidance (CFG) and requires training a second, weaker diffusion model for AutoGuidance (AG). We show that REPA itself can be viewed as x-prediction in RAE latent space. By simply re-parameterizing the output of the DiT model, it can provide guidance for "free". Overall, RAEv2 leads to more than 10x faster convergence over the original RAE, achieving a state-of-the-art gFID of 1.06 in just 80 epochs on ImageNet-256. On FDr^k, RAEv2 achieves a state-of-the-art 2.17 at just 80 epochs compared to the previous best 3.26 (800 epochs) without any post-training. This motivates EP_FID@k (epochs to reach unguided gFID <= k) as a measure of training efficiency. RAEv2 attains an EP_FID@2 of 35 epochs, versus 177 for the original RAE. We also validate our approach across diverse settings for text-to-image generation and navigation world models, showing consistent improvements. Code is available at this https URL. 

---
# ISEP: Implicit Support Expansion for Offline Reinforcement Learning via Stochastic Policy Optimization 

**Authors**: Yifei Chen, Shaoqin Zhu, Xiaoqiang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.18320)  

**Abstract**: Offline reinforcement learning methods typically enforce strict constraints to ensure safety; yet this rigidity often prevents the discovery of optimal behaviors outside the immediate support of the behavior policy. To address this, we propose Implicit Support Expansion via stochastic Policy optimization (ISEP), which leverages a value function interpolated between in-distribution data and policy samples to implicitly expand the feasible action support. This mechanism "densifies" high-reward regions, creating a navigable path for policy improvement while theoretically guaranteeing bounded value error. However, optimizing against this expanded support creates a multimodal landscape where standard deterministic averaging leads to mode collapse and invalid actions. ISEP mitigates this via a stochastic action selection strategy, optimizing the policy by stochastically alternating between conservative cloning and optimistic expansion signals. We instantiate this framework as ISEP-FM using Conditional Flow Matching utilizing classifier-free guidance to effectively capture the interpolated value signal. 

---
# Wasserstein Equilibrium Decoding for Reliable Medical Visual Question Answering 

**Authors**: Luca Hagen, Johanna P. Müller, Weitong Zhang, Mengyun Qiao, Bernhard Kainz  

**Link**: [PDF](https://arxiv.org/pdf/2605.18313)  

**Abstract**: Small vision-language models (2-8B) are well-suited for clin- ical deployment due to privacy constraints, limited connectivity, and low-latency requirements favouring on-device or on-premise inference. However, their limited capacity exacerbates the generation of plausible but incorrect outputs. We extend game-theoretic decoding, previously restricted to text-only, closed-ended NLP tasks, to vision-language mod- els for open-ended Medical VQA. We introduce a semantically aware Wasserstein stopping criterion that replaces lexical order matching, en- abling convergence based on semantic consensus among near-synonymous candidate answers and avoiding unnecessary iterations caused by clini- cally equivalent ranking swaps. On VQA-RAD and PathVQA, we ob- tain consistent, statistically significant improvements over greedy and discriminative baselines. On VQA-RAD, we improve Qwen3-VL-2B by +3.5 percentage points (p < 0.01), surpassing the greedy 4B model, with similar trends at larger scales. On PathVQA, Gemma-3-4B with BDG matches MedGemma-4B under greedy decoding despite no domain- specific fine-tuning. At accuracy parity with classic BDG, the Wasser- stein criterion reduces average convergence iterations by approximately 20%, improving inference efficiency while preserving the game-theoretic equilibrium behaviour. Code is available at this https URL Wasserstein-BDG-medical-VQA. 

---
# Alignment Dynamics in LLM Fine-Tuning 

**Authors**: Yuhan Huang, Huanran Chen, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.18309)  

**Abstract**: Although Large Language Models (LLMs) achieve strong alignment through supervised fine-tuning and reinforcement learning from human feedback, the alignment is often fragile under subsequent fine-tuning. Existing explanations either attribute alignment fragility to gradient geometry or characterize it as a distributional shift in model outputs, yet few provide a unified account that bridges parameter-space learning dynamics with function-space alignment behavior during fine-tuning. In this work, we introduce a tractable alignment score and derive its closed-form update during fine-tuning, yielding a unified framework for alignment dynamics. Our analysis decomposes alignment updates into two competing components: a \textbf{\color{red!60!black} Rebound Force}, governed jointly by the current alignment state and the narrowness of model distribution, and a \textbf{\color{green!60!black} Driving Force}, determined by how the training distribution aligns with outcome-conditioned posteriors over aligned and non-aligned completions. This decomposition explains why prior alignment can be reversed by later fine-tuning and why narrower posterior structure strengthens such reversal. Moreover, our framework predicts a \textbf{Rehearsal Priming Effect}: prior alignment leaves a latent posterior imprint that amplifies the effective Driving Force upon re-exposure, leading to faster re-alignment. We validate these predictions across safety alignment, emergent misalignment, and sentiment settings, demonstrating consistent alignment reversal and accelerated re-alignment under re-exposure. In addition, controlled experiments in safety alignment confirm the predicted dependence of rebound strength on posterior narrowness. Together, these results provide a unified dynamical perspective on how alignment is disrupted and reactivated during LLM fine-tuning. 

---
# PH-Dreamer: A Physics-Driven World Model via Port-Hamiltonian Generative Dynamics 

**Authors**: Xueyu Luan, Chenwei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.18303)  

**Abstract**: World models built on recurrent state space architectures enable efficient latent imagination, yet remain physically unstructured, producing dynamics that violate conservation and dissipative principles. We introduce a unified Port-Hamiltonian framework that remedies this through three synergistic mechanisms. First, we embed implicit physical priors into recurrent transitions by modeling projected latent evolution as action controlled energy routing governed by flow and dissipation, biasing the projected PH phase space toward a more compact and physically structured representation. Second, we develop a kinematics aware energy world model that estimates the Hamiltonian and power balance from proprioceptive observations, providing an explicit physical signal for thermodynamic reasoning. Third, leveraging these energy gradients, we establish an energy guided Actor-Critic that uses Lagrangian multipliers to regularize policy optimization toward lower energy and smoother control. Across visual control benchmarks, this paradigm not only attains superior asymptotic returns but also elevates internal simulator fidelity by establishing a tighter, lower variance alignment between imagined and real rewards, all while reducing latent phase space volume by 4.18-8.41%, energy consumption by up to 7.80%, and mean squared jerk by up to 9.38%. 

---
# CommitDistill: A Lightweight Knowledge-Centric Memory Layer for Software Repositories 

**Authors**: Divya Chukkapalli, Thejesh Avula, Aditya Aggarwal, Harsimran Singh, Amith Tallanki  

**Link**: [PDF](https://arxiv.org/pdf/2605.18284)  

**Abstract**: Software repositories accumulate large amounts of unstructured knowledge in commit messages, pull-request discussions, and issue threads, but developers and AI coding assistants rarely reuse this history effectively. Recent work on typed-memory architectures for LLM agents (MemGPT, generative agents, and the PlugMem module of Yang et al.) argues that agent memory should be distilled, typed knowledge rather than raw interaction text. We adapt that stance to a software repository's own git history under a constrained regime: deterministic, dependency-free, local-only, no embeddings. We present CommitDistill, an open-source Python prototype that mines a local git history into typed knowledge units (Facts, Skills, Patterns) using deterministic regex and surfaces them through a TF-IDF retriever with a calibrated silence threshold (theta = 2.5) that abstains on out-of-distribution queries. The artefact is a trust-instrumented memory substrate: deterministic, no external service, inspectable plain-JSON store, tunable abstention. A case study on five public repositories spanning Python, JavaScript, C, and Java (25,000 commits, 1,167 extracted units) reports useful-precision 0.525 at Cohen's kappa = 0.633 on 40 dual-annotated Python units. The decisive finding is budget-constrained retrieval: at a 256-character per-query budget, CommitDistill reaches 0.750 hit-rate on a 12-query benchmark against BM25's 0.333 and git log --grep's 0.083. On a four-arm paired LLM-as-judge evaluation (n=200 time-travel bug-fixes, two judges) covering control, CommitDistill, a body-budget-matched CD-Hybrid, and BM25, no condition produces a statistically detectable lift over control on the headline mean and CD-Hybrid is indistinguishable from BM25 head-to-head. Extraction over 10,000 commits completes in under 4 seconds on a laptop. Source, annotations, baselines, and a reproducibility script accompany this paper. 

---
# From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG 

**Authors**: Changmin Lee, Jaemin Kim, Taesik Gong  

**Link**: [PDF](https://arxiv.org/pdf/2605.18271)  

**Abstract**: With the rapid emergence of personal AI agents based on Large Language Models (LLMs), implementing them on-device has become essential for privacy and responsiveness. To handle the inherently personal and context-dependent nature of real-world requests, such agents must ground their generation in device-resident personal context. However, under tight memory budgets, the core bottleneck is what to store so that retrieval remains aligned with the user. We propose EPIC (Efficient Preference-aligned Index Construction), which focuses on user preferences as a compact and stable form of personal context and integrates them throughout the RAG pipeline. EPIC selectively retains preference-relevant information from raw data and aligns retrieval toward preference-aligned contexts. Across four benchmarks covering conversations, debates, explanations, and recommendations, EPIC reduces indexing memory by 2,404 times, improves preference-following accuracy by 20.17 percentage points, and achieves 33.33 times lower retrieval latency over the best-performing baseline. In our on-device experiment, EPIC maintains a memory footprint under 1 MB with 29.35 ms/query latency in streaming updates. 

---
# CodeBind: Decoupled Representation Learning for Multimodal Alignment with Unified Compositional Codebook 

**Authors**: Zeyu Chen, Jie Li, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.18257)  

**Abstract**: Multimodal representation alignment is pivotal for large language models and robotics. Traditional methods are often hindered by cross-modal information discrepancies and data scarcity, leading to suboptimal alignment spaces that overlook modality-unique features. We propose CodeBind, a framework that optimizes multimodal representation spaces through a modality-shared-specific codebook design. By incrementally aligning target and bridging modalities, CodeBind bypasses the need for fully paired data. Unlike traditional hard alignment, CodeBind decomposes features into shared components for semantic consistency and specific components for modality-unique details. This design utilizes a compositional vector quantization scheme, where a shared codebook bridges modality gaps and modality-specific codebooks mitigate representation bias by preventing dominant modalities from overshadowing others. Validated across nine modalities (text, image, video, audio, depth, thermal, tactile, 3D point cloud, EEG), CodeBind achieves state-of-the-art performance in multimodal classification and retrieval tasks. 

---
# Machine Unlearning for Masked Diffusion Language Models 

**Authors**: Georu Lee, Seungwon Jeong, Hoki Kim, Jinseong Park, Woojin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.18253)  

**Abstract**: Recent masked diffusion language models (MDLMs), such as LLaDA and Dream, have achieved performance comparable to autoregressive large language models. Unlike autoregressive models, which generate text sequentially, MDLMs generate text by iteratively denoising masked positions in parallel. During fine-tuning, MDLMs learn to recover responses from masked response states conditioned on a prompt, thereby shifting their predictions from a prompt-masked unconditional distribution toward a prompt-conditional distribution. Despite this distinct generative and fine-tuning mechanism, machine unlearning for MDLMs remains largely unexplored. In this paper, we propose Masked Diffusion Unlearning (MDU), the first unlearning framework for MDLMs, by revisiting the process of learning specific knowledge in terms of diffusion. Specifically, MDU minimizes a forward KL divergence from the prompt-conditional prediction to a prompt-masked unconditional anchor at every masked response position, with a temperature scaling parameter to control the privacy-utility trade-off. Our empirical results on standard benchmarks and MDLM backbones show that MDU achieves high unlearning performance compared to existing LLM unlearning methods. Code is available at this https URL. 

---
# Privacy Preserving Reinforcement Learning with One-Sided Feedback 

**Authors**: Lin William Cong, Guangyan Gan, Hanzhang Qin, Zhenzhen Yan  

**Link**: [PDF](https://arxiv.org/pdf/2605.18246)  

**Abstract**: We study reinforcement learning (RL) in multi-dimensional continuous state and action spaces with one-sided feedback, where the agent receives partial observations of the state and obtains reward information for only a subset of the state-action space at each time step. This setting introduces substantial challenges in both learning efficiency and privacy preservation. To address these challenges, we propose POOL, a novel privacy-preserving RL algorithm. We conduct a comprehensive theoretical analysis of POOL, deriving a sample complexity bound that matches the known lower bounds for non-private RL. Here, E_rho denotes the privacy parameter, H is the time horizon, and alpha is the optimality-gap parameter. Our findings show that it is possible to enforce strong privacy guarantees while maintaining high learning efficiency, marking a significant step toward practical, privacy-aware RL in multi-dimensional environments with one-sided feedback. 

---
# Multilingual jailbreaking of LLMs using low-resource languages 

**Authors**: Dylan Marx, Marcel Dunaiski  

**Link**: [PDF](https://arxiv.org/pdf/2605.18239)  

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreak attempts that circumvent safety guardrails. We investigate whether multi-turn conversations using low-resource African languages (Afrikaans, Kiswahili, isiXhosa, and isiZulu) can bypass safety mechanisms across commercial LLMs. We translated prompts from existing datasets and evaluated ChatGPT, Claude, DeepSeek, Gemini, and Grok through automated testing and human red-teaming with native speakers. Single-turn translation attacks proved ineffective, while multi-turn conversations achieved English harmful response rates from 52.7% (Claude 3.5 Haiku) to 83.6% (GPT-4o-mini), Afrikaans from 60.0% (Claude 3.5 Haiku) to 78.2% (GPT-4o-mini), and Kiswahili from 41.8% (Claude 3.5 Haiku) to 70.9% (DeepSeek). Human red-teaming increased jailbreak rates compared to automated methods. Over all evaluated languages, the average jailbreak rate increased from 59.8% to 75.8%, with improvements of +20.0% (Afrikaans), +12.7% (isiZulu), +12.3% (isiXhosa), and +1% (Kiswahili), demonstrating that poor translation quality limits jailbreak success. These findings suggest that vulnerabilities in LLMs persist in multilingual contexts and that translation quality is the critical factor determining jailbreak success in low-resource languages. 

---
# SomaliWeb v1: A Quality-Filtered Somali Web Corpus with a Matched Tokenizer and a Public Language-Identification Benchmark 

**Authors**: Khalid Yusuf Dahir  

**Link**: [PDF](https://arxiv.org/pdf/2605.18232)  

**Abstract**: Somali is a Cushitic language of the Horn of Africa with ~25 million speakers, yet no documented dedicated Somali pretraining corpus with a companion tokenizer and language-identification benchmark has been publicly released. Existing Somali text appears either inside multilingual distributions (HPLT v2, CC100, MADLAD-400, OSCAR, mC4) or in small, undocumented Somali-only uploads on Hugging Face. We introduce SomaliWeb v1, a quality-filtered Somali corpus of 819,322 documents (~303M tokens) built from three upstream sources (HPLT v2, CC100, Somali Wikipedia) through a six-stage reproducible pipeline. We release (i) the corpus, (ii) a matched BPE-16K tokenizer, and (iii) the first public side-by-side Somali benchmark of three production language identifiers. Our measurements reveal concrete quality defects in existing distributions: HPLT v2's "cleaned" Somali release retains 17.3% byte-exact duplicates, 56.1% of its documents contain fixable mojibake, and 10.7% of its byte-unique documents are near-duplicates at Jaccard tau=0.80. Our BPE-16K tokenizer emits 40.2% fewer tokens than GPT-4's cl100k_base on FLORES-200 Somali devtest as a tokenizer-level measurement; downstream language-model perplexity comparisons are deferred to a follow-up release. 

---
# Are Sparse Autoencoder Benchmarks Reliable? 

**Authors**: David Chanin  

**Link**: [PDF](https://arxiv.org/pdf/2605.18229)  

**Abstract**: Sparse autoencoders (SAEs) are a core interpretability tool for large language models, and progress on SAE architectures depends on benchmarks that reliably distinguish better SAEs from worse ones. We audit the SAE quality metrics in SAEBench, the de-facto standard SAE evaluation suite, through three complementary lenses: reseed noise on a fixed SAE, ground-truth correlation on synthetic SAEs, and discriminability across training trajectories. We find that two of these metrics, Targeted Probe Perturbation (TPP) and Spurious Correlation Removal (SCR), fail multiple lenses at their canonical settings and should not be used to evaluate SAEs. The other metrics show higher reseed noise and lower discriminability than the field assumes. The sae-probes variant of $k$-sparse probing is the most reliable metric we tested, but even sae-probes struggles to separate variants of the same SAE architecture. Our results show the field needs better SAE benchmarks. 

---
# Context Memorization for Efficient Long Context Generation 

**Authors**: Yasuyuki Okoshi, Hao Mark Chen, Guanxi Lu, Hongxiang Fan, Masato Motomura, Daichi Fujiki  

**Link**: [PDF](https://arxiv.org/pdf/2605.18226)  

**Abstract**: Modern large language model (LLM) applications increasingly rely on long conditioning prefixes to control model behavior at inference time. While prefix-augmented inference is effective, it incurs two structural limitations: i) the prefix's influence fades as generation proceeds, and ii) attention computation over the prefix scales linearly with its length. Existing approaches either keep the prefix in attention while compressing it, or internalize it into model parameters through gradient-based training. The former still attends to the prefix at inference, while the latter is training-intensive and ill-suited to prefix updates. To address these issues, we propose attention-state memory, a training-free approach that externalizes the prefix into a lightweight, lookup-based memory of precomputed attention states between prefix and query tokens. On ManyICLBench with LLaMA-3.1-8B, our method improves accuracy over in-context learning at 1K-8K memory budgets while reducing attention latency by 1.36x at 8K, and surpasses full-attention RAG performance on NBA benchmark using only 20% of its memory footprint. 

---
# A Simplex Witness Certificate for Constant Collapse in Variational Autoencoders 

**Authors**: Zegu Zhang, Jianhua Peng, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18224)  

**Abstract**: This note studies exact constant collapse in variational autoencoders, where the encoder mean becomes independent of the input. The goal is to make this specific failure mode pre-designable, monitorable during training, and certifiable after training. The prior is kept as the standard Gaussian. Given a fixed teacher posterior, we attach to the latent mean a fixed simplex witness head. The resulting teacher-student alignment loss has an exact constant-predictor baseline equal to the teacher information. If the alignment loss is below this baseline, the latent mean cannot be input-independent constant collapsed.
The simplex witness also has a closed-form inverse. Any full-support teacher posterior can be represented by embedding its centered log-odds into the latent space. This gives an explicit latent energy cost and explains when the alignment loss can be made small. A computable view gap handles the case where teacher targets are computed from a different view. Thus exact constant collapse is converted from an after-the-fact training pathology into a design-and-certificate problem. 

---
# Leveraging Graph Structure in Seq2Seq Models for Knowledge Graph Link Prediction 

**Authors**: Luu Huu Phuc, Ratan Bahadur Thapa, Mojtaba Nayyeri, Jingcheng Wu, Evgeny Kharlamov, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2605.18211)  

**Abstract**: We introduce Graph-Augmented Sequence-to-Sequence (GA-S2S), a novel framework that integrates a T5-small encoder-decoder with a Relational Graph Attention Network (RGAT) to improve link prediction in knowledge graphs. While existing Seq2Seq models rely solely on surface-level textual descriptions of entities and relations and at best, flatten the neighborhoods of a query entity into a single linear sequence, thereby discarding the inherent graph structure, GA-S2S jointly encodes both textual features and the full $k$-hop subgraph topology surrounding the query entity. By integrating raw encoder outputs with RGAT's relation-aware embeddings, our model captures and leverages richer multi-hop relational patterns and textual information. Our preliminary experiments on the CoDEx dataset demonstrate that GA-S2S outperforms competitive Seq2Seq-based baseline models, achieving up to a 19\% relative gain in link prediction accuracy. 

---
# SPATIOROUTE: Dynamic Prompt Routing for Zero-Shot Spatial Reasoning 

**Authors**: Pawat Chunhachatrachai, Gueter Josmy Faure, Hung-Ting Su, Winston H. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18209)  

**Abstract**: Spatial question answering over egocentric video is a challenging task that requires Vision-Language Models (VLMs) to reason about 3D object positions, scene affordances, and directional relationships, particularly in the zero-shot setting where no task-specific fine-tuning is available. We introduce SpatioRoute, a dynamic prompt generation approach that routes each incoming question to a semantically tailored prompt template -- without any additional training, fine-tuning, or 3D sensor input. SpatioRoute operates in two complementary modes: SpatioRoute-R, a rule-based router that deterministically maps question typologies (e.g., What, Is, How, Can, Which) to specialized prompt templates; and SpatioRoute-L, an LLM-driven approach that generates task-specific prompts from the question and situational context alone, with no video input at routing time. We evaluate SpatioRoute on the SQA3D benchmark across VLMs spanning model families. SpatioRoute achieves consistent overall accuracy gains up to 5% over fixed prompt baselines, establishing a new state-of-the-art for zero-shot video-only spatial VQA without requiring 3D point-cloud inputs. As an additional finding, we observe that Chain-of-Thought (CoT) prompting, implemented via the Think it Twice architecture, consistently degrades performance in this setting on Qwen series models, confirming that question-aware routing is more effective than uniform reasoning instructions for spatial video understanding. 

---
# Concise and Logically Consistent Conformal Sets for Neuro-Symbolic Concept-Based Models 

**Authors**: Samuele Bortolotti, Emanuele Marconato, Andrea Pugnana, Andrea Passerini, Stefano Teso  

**Link**: [PDF](https://arxiv.org/pdf/2605.18202)  

**Abstract**: Neuro-Symbolic Concept-based Models (NeSy-CBMs) are a family of architectures that integrate neural networks with symbolic reasoning for enhanced reliability in high-stakes applications. They work by first extracting high-level concepts from the input and then inferring a task label from these compatibly with given logical constraints. Yet, their label and concept predictions can be overconfident, making it difficult for stakeholders to gauge when the model's decisions can be trusted. We address this issue by integrating ideas from Conformal Prediction (CP), a framework providing rigorous, distribution-free coverage guarantees. We formalize three desiderata -- consistency, coverage, and conciseness -- that any conformal method for NeSy-CBMs should satisfy, and show that existing approaches fall short of at least one. We then introduce COCOCO, a post-hoc framework that conformalizes concepts and labels jointly and reconciles them via a single deduction-abduction revision step. COCOCO satisfies all three desiderata, retains distribution-free coverage, is robust to imperfect knowledge and supports user-specified size budgets. Our experiments on 8 data sets highlight how COCOCO compares favorably against competitors and natural baselines in terms of performance and set size. 

---
# PIPER: Content-Based Table Search via profiling and LLM-Generated Pseudoqueries 

**Authors**: Riccardo Terrenzi, Matteo Falconi, Serkan Ayvaz, Pierluigi Plebani  

**Link**: [PDF](https://arxiv.org/pdf/2605.18199)  

**Abstract**: The rapid growth of tabular datasets in data lakes, data spaces, and open data portals makes effective dataset search essential for reuse and analysis. Existing search systems rely mainly on metadata, which is often incomplete or low quality, especially for tables whose meaning depends on both schema and cell values. Recent advances in Large Language Models (LLMs) enable richer, content-based representations of tables. However, prior LLM-based retrieval methods have focused on Table Question Answering, where the goal is to select a single table to answer a question, rather than retrieve and rank relevant datasets. We propose PIPER, a content-driven retrieval method for tabular datasets that uses table profiles and LLM-generated queries embedded for dense retrieval. Designed for dataset search in poor-metadata settings, PIPER outperforms both classical metadata-based baselines and strong TableQA retrieval methods, demonstrating the value of LLM-based content modeling for tabular dataset search. 

---
# RGB-only Active 3D Scene Graph Generation for Indoor Mobile Robots 

**Authors**: Giorgia Modi, Davide Buoso, Giuseppe Averta, Daniele De Martini  

**Link**: [PDF](https://arxiv.org/pdf/2605.18197)  

**Abstract**: Current approaches to 3D scene graph generation rely on dedicated depth sensors, such as LiDAR or RGB-D cameras, for metric 3D reconstruction. This limits deployment to specialized robotic platforms and excludes settings where only RGB cameras are available, such as fixed external infrastructure. Existing pipelines also typically operate on passively collected observation trajectories, rather than selecting viewpoints based on the partially built scene representation, and therefore fail to effectively exploit the semantic and spatial information encoded within the graph during exploration. This paper presents a fully visual framework for the active, incremental construction of 3D scene graphs from RGB input only, addressing both limitations. The proposed approach unifies perception and planning around a shared structured representation that captures object semantics, 3D geometry, relational context, and information from multiple viewpoints. Because the framework is hardware-agnostic and relies only on RGB observations, it can incorporate inputs from both onboard robot cameras and fixed external cameras within the same representation. Experiments on the Replica dataset show that the RGB-only pipeline achieves F1-score parity with baselines using ground-truth depth. Active exploration experiments on ReplicaCAD further show that semantic-driven viewpoint selection detects more than twice as many objects as a geometric frontier-based baseline under the same exploration budget. Finally, the external-camera setting demonstrates that complementary RGB views can effectively bootstrap the scene graph and improve contextual understanding at no additional exploration cost. 

---
# Fixed External Cameras as Common Prior Maps for Active 3D Scene Graph Generation 

**Authors**: Giorgia Modi, Davide Buoso, Giuseppe Averta, Daniele De Martini  

**Link**: [PDF](https://arxiv.org/pdf/2605.18184)  

**Abstract**: Commonly available prior information, such as BIM models, floor plans, and remote sensing images, can provide valuable geometric and semantic context for autonomous robotic systems. In this paper, we treat observations from fixed external RGB cameras as Common Prior Maps (CPMs): wide-field views of the environment that initialize a semantic and geometric scene prior before any robot motion begins. We present an RGB-only framework for active, incremental 3D scene graph (3DSG) generation that seamlessly fuses observations from both onboard robot cameras and fixed external cameras within a single hardware-agnostic pipeline. By relying solely on RGB observations processed by a feed-forward 3D reconstruction model, the system treats all cameras - onboard or external - identically, requiring no hardware modifications. A graph-based active semantic exploration framework then directly leverages the partial scene graph to guide the robot toward regions of high semantic uncertainty, progressively completing and refining the prior. Experiments demonstrate that bootstrapping the scene graph with even a single external camera increases initial object recall by up to +79%, and that the richer context of the prior significantly improves the efficiency of subsequent active exploration. 

---
# MARS: Technical Report for the CASTLE Challenge at EgoVis 2026 

**Authors**: Haoyu Zhang, Qiaohui Chu, Yisen Feng, Meng Liu, Weili Guan, Yaowei Wang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2605.18176)  

**Abstract**: This report presents MARS, short for Multimodal Agentic Reasoning with Source selection, our system for the CASTLE Challenge at EgoVis 2026. Participants must answer 185 closed-form questions over the CASTLE 2024 dataset. In contrast to prior single-video egocentric benchmarks, CASTLE requires reasoning over four days of activity, 15 synchronized perspectives, official transcripts, and multiple auxiliary modalities, including personal photos, auxiliary videos, gaze, thermal imagery, and heartrate measurements. MARS therefore treats the task as an agentic evidence-selection problem over multimodal sources rather than a purely text-only pipeline. MARS first follows the official CASTLE directory organization to build evidence memories from two primary sources, videos and transcripts, and four auxiliary sources, gaze, heartrate, photos, and thermal imagery. Long videos are converted into captions and DeepSeek-based summaries only because CASTLE videos are too long to fit directly into the model context for every question; this step compresses temporal evidence while keeping photos and other auxiliary media available as source-specific evidence. At inference time, a GPT-5.4 decision agent repeatedly chooses whether to continue reasoning, request a specific missing modality, produce an answer, or fall back to a random option when the evidence remains insufficient. The resulting system achieved second place on the final CASTLE Challenge leaderboard. Our codes are available at this https URL. 

---
# Self-Evolving Spatial Reasoning in Vision Language Models via Geometric Logic Consistency 

**Authors**: Junming Liu, Yuqi Li, Yifei Sun, Maonan Wang, Piotr Koniusz, Yirong Chen, Ding Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18162)  

**Abstract**: Vision-Language Models (VLMs) have made striking progress, yet their spatial reasoning remains fragile: models that answer an original input correctly can still fail under paired transformations with predictable answer mappings, revealing a gap between instance-level correctness and robust spatial reasoning. To address this, we propose Spatial Alignment via Geometric Evolution (SAGE), a self-evolving framework that enforces logical consistency in VLMs through geometric and linguistic duality operations. SAGE incorporates duality consistency as an auxiliary reward within GRPO training, encouraging models to produce logically coherent answers across original and transformed inputs. A dynamic operation pool continuously probes for inconsistencies, promoting challenging operations and retiring mastered ones, so that training focuses on the most informative signals. SAGE is model-agnostic, data-efficient compared to prior GRPO methods, and can be applied as a lightweight post-training stage to any existing VLM. Experiments on video and spatial reasoning benchmarks demonstrate consistent improvements over strong baselines and enhanced generalization to unseen data. 

---
# Vision Inference Former: Sustaining Visual Consistency in Multimodal Large Language Models 

**Authors**: Xinpeng Dong, Min Zhang, Kairong Han, Xu Tan, Fei Wu, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18160)  

**Abstract**: In recent years, multimodal large language models (MLLMs) have achieved remarkable progress, primarily attributed to effective paradigms for integrating visual and textual information. The dominant connector-based paradigm projects visual features into textual sequence, enabling unified multimodal alignment and reasoning within a generative architecture. However, our experiments reveal two key limitations: (1) Although visual information serves as the core evidential modality in MLLMs, it is treated on par with textual tokens, diminishing the unique contribution of the visual modality; (2) As generation length increases, particularly within a limited context window, the model's dependence on visual information progressively weakens, resulting in deteriorated vision-language alignment and reduced consistency between generated content and visual semantics. To address these challenges, we propose the Vision Inference Former (VIF), a lightweight architectural module that establishes a direct bridge between pure visual representations and the model's output space. Specifically, VIF continuously injects visual semantics throughout the decoding phase of the inference process, ensuring that the model remains firmly grounded in visual content during generation. We conduct experiments on 14 benchmark tasks covering general reasoning, OCR, table understanding, vision-centric evaluation, and hallucination. Experimental results show that VIF consistently improves model performance across diverse architectures while introducing minimal additional overhead. The code for this work is available at this https URL. 

---
# An Empirical Study of Privacy Leakage Chains via Prompt Injection in Black-Box Chatbot Environments 

**Authors**: Hongjang Yang, Hyunsik Na, Daeseon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.18133)  

**Abstract**: LLM-based chatbot agents increasingly process user requests by combining natural-language reasoning with external tools such as web browsing. These capabilities improve usability, but they also create attack surfaces when untrusted external content is processed as part of a user' s task. This paper studies a privacy-leakage attack chain based on indirect prompt injection in black-box chatbot environments, where the attacker has no access to model weights, system prompts, or agent implementation details including how a trajectory is actually managed during its processing for a query. We first analyze how an attacker can hijack an agent' s intended task by crafting external content that appears benign to the victim while inducing the agent to execute an attacker-defined objective. We then evaluate a new prompt-injection technique, called exemplification, which uses a bridge in the external content to reframe the user prompt and the benign beginning of the retrieved page as few-shot examples before appending the attacker' s objective. We compare its attack success rate with a prior fake-completion technique. Finally, we demonstrate a proof-of-concept data-exfiltration chain using fictitious personal information in a controlled setting. Our results suggest that prompt injection, jailbreak-style instruction steering, and web-tool invocation can be combined into a feasible privacy-leakage path in deployed chatbot agents. 

---
# Who Generated This 3D Asset? Learning Source Attribution for Generative 3D Models 

**Authors**: Sihan Ma, Siyuan Liang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2605.18132)  

**Abstract**: Generative 3D models are deployed in gaming, robotics, and immersive creation, making source attribution critical: given a 3D asset, can we identify whether and which generative model created it? This problem faces two core challenges: dispersed attribution signals, where 3D fingerprints are distributed across multi-view, geometric, and frequency-domain cues; and realistic deployment constraints, where scarce labels, degraded prompts, and mixed real/synthetic assets undermine attribution reliability. To systematically study this problem, we construct, to the best of our knowledge, the first passive source attribution benchmark for modern generated assets, covering 22 representative 3D generators under standard, few-shot, and realistic deployment protocols. Based on this benchmark, we find that generative 3D models leave two types of stable fingerprints: cross-view inconsistency and structural artifacts reflected in geometric statistics and frequency-domain cues. To capture these dispersed signals, we propose a hierarchical multi-view multi-modal Transformer that fuses appearance, geometric, and frequency-domain features within each view and models global relationships across views. Extensive experiments demonstrate strong performance, achieving 97.22% accuracy under full supervision and 77.17% accuracy with only 1% training data, corresponding to fewer than five samples per generator. These results show that modern 3D generators leave stable and attributable fingerprints, establishing a new benchmark and methodological foundation for trustworthy 3D content provenance. 

---
# Symmetry-Compatible Principle for Optimizer Design: Embeddings, LM Heads, SwiGLU MLPs, and MoE Routers 

**Authors**: Tim Tsz-Kit Lau, Weijie Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.18106)  

**Abstract**: A striking geometric disparity has long persisted in the practice of deep learning. While modern neural network architectures naturally exhibit rich symmetry and equivariance properties, popular optimizers such as Adam and its variants operate inherently coordinate-wise, rendering them unable to respect the equivariance structures of the parameter space. We address this disparity by introducing a symmetry-compatible principle for optimizer design: the gradient update rule should be equivariant under the symmetry group acting on the corresponding weight block. Following this principle, we first provide a unified perspective on bi-orthogonally equivariant updates for general matrix layers, as employed by stochastic spectral descent, Muon, Scion, and polar gradient methods. More importantly, by moving from orthogonal groups to permutation and shared-shift symmetries, we derive symmetry-compatible optimizers for parameter blocks whose symmetries differ from those of general matrix layers: embedding and LM head matrices, SwiGLU MLP projections, and MoE router matrices. These constructions include one-sided spectral, row-norm, hybrid row-norm/spectral, row-aware, column-aware, centered row-norm, and left-spectral updates. They yield an end-to-end layerwise optimizer stack in which each major matrix-valued parameter class is assigned an update whose equivariance matches its symmetry group. We corroborate this principle through pre-training experiments on dense and sparse MoE language models, including Qwen3-0.6B-style, Gemma 3 1B-style, OLMoE-1B-7B-style, and downsized gpt-oss architectures. Across these experiments, symmetry-compatible updates consistently improve final validation loss, and in several cases training stability, over corresponding AdamW updates. 

---
# SENSE: Satellite-based ENergy Synthesis for Sustainable Environment 

**Authors**: Kailai Sun, Mingyi He, Heye Huang, Can Rong, Alok Prakash, Baoshen Guo, Shenhao Wang, Jinhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.18101)  

**Abstract**: Urban Building Energy Modeling plays a critical role in achieving the United Nations' Sustainable Development Goals 7 and 11. Although existing studies based on satellite imagery and deep learning have achieved remarkable progress, many challenges exist: most existing studies are inherently predictive, failing to reflect the generative nature of urban planning; although generative AI and diffusion models have seen explosive growth in satellite imagery, they lack the urban functional generation (e.g., energy layer); third, aligned high-quality high-resolution building energy data with satellite imagery is limited and scarce. Here we propose SENSE (Satellite-based ENergy Synthesis for Sustainable Environment), a unified generative UBEM framework that jointly synthesizes realistic urban satellite imagery and aligned high-quality building energy consumption and height maps. By conditioning on road networks and urban density metrics, SENSE, based on a controllable diffusion model, leverages the knowledge learned by large vision models to generate urban building energy consumption and height information (annotations) in the latent space. Experiments across four cities (New York City, Boston, Lyon, Busan) demonstrate that SENSE achieves high visual fidelity and strong physical consistency, satisfying the ASHRAE standard metric. Experiments demonstrate that SENSE can generate enough annotated synthetic data using less than 20% labeled energy data, boosting downstream prediction performance by 10% IoU. Compared to SOTA urban energy prediction methods, SENSE significantly reduced prediction error (reduced 3%-11% NMBE and 1%-9% CVRMSE). This study offers an energy-efficiency urban planning and physical generation solution for urban science, energy science and building science. The dataset and code: this https URL and this https URL. 

---
# Parameterized 4-Qubit EWL Quantum Game Circuits with Dirac-Solow-Swan Hamiltonian Integration for Quadruple Helix Disruptive Innovation Recommender Systems 

**Authors**: Agung Trisetyarso, Fithra Faisal Hastiadi, Kridanto Surendro  

**Link**: [PDF](https://arxiv.org/pdf/2605.18080)  

**Abstract**: We present a novel parameterized 4-qubit Eisert-Wilkens-Lewenstein (EWL) quantum game circuit for recommender systems in quadruple helix innovation ecosystems (academia, industry, government, and civil society). The local strategy operators $U_{i} = R_y(\theta_{i})$ for each helix actor are directly tuned by normalized dominance weights extracted from real participant funding data (\texit{ecContribution}) in the European Commission CORDIS Horizon Europe database (project COVend, ID 101045956). The circuit employs a multi-qubit EWL entangler followed by parameterized local rotations, inverse entangler, and full measurement, achieving only 22 gates and circuit depth 11 while scaling as $O(n)$ for $n$-round helix communications. Measurement probabilities after the quantum game serve as recommender scores for disruptive versus sustaining innovation trends. These scores are subsequently mapped into the diagonal Dirac potential of a Dirac-Solow-Swan Hamiltonian, enabling time-evolution simulation of capital accumulation and bifurcation dynamics under disruptive innovation. Numerical experiments on real CORDIS quadruple-helix collaboration networks demonstrate the circuit's NISQ compatibility and its ability to forecast disruptive capital trajectories with high fidelity.
The proposed framework bridges quantum game theory, parameterized quantum circuits, and relativistic economic growth models, offering a computationally efficient tool for innovation policy and strategic decision-making in complex socio-economic ecosystems. Complexity analysis and reproducibility are provided through open Qiskit implementations. 

---
# A-ProS: Towards Reliable Autonomous Programming Through Multi-Model Feedback 

**Authors**: Anika Tabassum, Md Sifat Hossain, Md. Fahim Arefin, Tariqul Islam, Tarannum Shaila Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2605.18073)  

**Abstract**: Large Language Models (LLMs) demonstrate strong potential for automated code generation, yet their ability to iteratively refine solutions using execution feedback remains underexplored. Competitive programming offers an ideal testbed for this investigation, as it demands end-to-end algorithmic reasoning, precise implementation under strict computational constraints, and complete functional correctness with rigorous evaluation. In this paper, we present A-ProS, an autonomous AI agent that solves competitive programming problems through a hybrid multi-model feedback framework separating solution generation from specialized debugging. A-ProS combines ChatGPT-based generators (GPT-4 and GPT-5) with three debugging critics: Codestral-2508, Llama-3.3-70B, and DeepSeek-R1, under a 2 x 3 factorial design. We evaluate six workflows on 367 problems from ICPC World Finals (2011-2024) and Codeforces (rated 1200-1800). The results show that GPT-5 workflows improve from 39 initial accepted solutions to 85-90 after three refinement rounds, while GPT-4 improves from 15 to 31-38. A controlled ablation on 47 problems shows that stateful refinement outperforms stateless approaches by 8.5-10.6 percentage points and reduces repeated failures by up to 3.5x. Compared to baseline agent loops, A-ProS achieves over 2x greater gains, highlighting the importance of persistent context and multi-model feedback for reliable autonomous program synthesis. 

---
# Improving Spatio-Temporal Residual Error Propagation by Mitigating Over-Squashing 

**Authors**: Seyed Mohamad Moghadas, Esther Rodrigo Bonet, Bruno Cornelis, Adrian Munteanu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18068)  

**Abstract**: Residual error propagation remains a fundamental problem in recurrent models, where small prediction inaccuracies compound over time and degrade long-horizon performance. Accurately modeling the correlation structure of such residuals is critical for reliable uncertainty quantification in probabilistic multivariate timeseries forecasting. While recent time-series deep models efficiently parametrize time-varying contemporaneous correlations, they often assume temporal independence of errors and neglect spatial correlation across the observed network. In this paper, we introduce Teger, a structured uncertainty module that overcomes the spa- tial and temporal limitations of error-correlated autoregressive forecasting. Teger proposes a spatial curvature-aware graph rewiring mechanism explicitly strengthening information-bottleneck edges identified by discrete Forman curvature. The component is integrated into a low-rank-plus-diagonal covariance head, preserving tractable inference via the Woodbury identity. Teger is backbone-agnostic, requiring only the latent state produced by any autoregressive encoder. We provide theoretical evidence of Teger, and experimentally evaluate it on LSTM, Transformer, and xLSTM backbones across four real-world spatio-temporal datasets, showing consistent improvement in Continuous Ranked Probability Score (CRPS). We further provide a formal theoretical analysis connecting curvature-aware rewiring to (i) oversquashing alleviation, (ii) improved spectral connectivity, (iii) reduced effective resistance, and (iv) improved covariance calibration bounds 

---
# FLAG: Foundation model representation with Latent diffusion Alignment via Graph for spatial gene expression prediction 

**Authors**: Qi Si, Penglei Wang, Yushuai Wu, Yifeng Jiao, Xuyang Liu, Xin Guo, Yuan Qi, Yuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.18055)  

**Abstract**: Predicting spatial gene expression from routine H\&E enables large-scale molecular profiling, yet current models treat this as isolated pointwise tasks, thereby overlooking essential biological structures like gene coordination and spatial distribution. To preserve these relationships, we introduce \textbf{FLAG}, a diffusion-based framework that redefines this task as structured distribution modeling. At the same time, we identify the critical \textbf{Gene Dimension Curse}, where joint modeling gene expression and their spatial interactions fail in high-dimensional spaces, and FLAG solves this challenge by integrating a spatial graph encoder for topological consistency and utilizing Gene Foundation Model (GFM) alignment for gene-gene fidelity in the generation process. To rigorously assess model performance, we propose a set of novel structural evaluation metrics, including Gene Structural Correlation (\textbf{GSC}) and Spatial Structural Correlation (\textbf{SSC}). Our experiments demonstrate that FLAG is highly competitive in traditional accuracy (PCC/MSE) while achieving significantly enhanced structural fidelity in capturing both gene-gene and gene-spatial relationships. The code is available at this https URL. 

---
# Confidence-Gated Robot Autonomy: When Does Uncertainty Actually Help? 

**Authors**: Johannes A. Gaus, Jhon P.F. Charaja, Daniel Haeufle  

**Link**: [PDF](https://arxiv.org/pdf/2605.18045)  

**Abstract**: Robotic systems often use predictive uncertainty to decide whether to act autonomously or defer to a fallback policy. In threshold-gated autonomy, uncertainty matters mainly through its ability to rank likely errors. Standard metrics such as expected calibration error and AUROC do not directly test whether uncertainty changes act/defer decisions. We therefore evaluate uncertainty using Spearman rank correlation, paired bootstrap equivalence testing, and act/defer agreement. Across three temporal activity-recognition benchmarks, we find a dataset-dependent competence regime below which uncertainty provides a weak and unstable error ranking. Above this regime, softmax heuristics, MC Dropout, and ensembles produce similar gating behavior, while threshold choice has a much larger effect on execution outcomes. A multi-seed embodied simulation shows the same pattern for collision rate and cost once realized autonomy is matched. Under temporal covariate shift, ranking quality remains stable, but fine grained semantic OOD detection remains near chance. These results suggest that simple uncertainty proxies can suffice for selective gating once the base model is competent, but not for semantic novelty detection. 

---
# Exploring Trust Calibration in XAI - The Impact of Exposing Model Limitations to Lay Users 

**Authors**: Alfio Ventura, Tim Katzke, Jan Corazza, Mustafa Yalçıner  

**Link**: [PDF](https://arxiv.org/pdf/2605.18036)  

**Abstract**: Trust calibration -- aligning user trust judgment with model capability -- is crucial for safe deployment of explainable AI (XAI), yet is often evaluated via global trust ratings detached from objective performance evidence. We present a preregistered, incentivized between-subject online study (N=418 representative UK sample) on explainable skin-lesion classification that disentangles expectation-setting from experienced performance. Participants completed 15 case evaluations using a fixed XAI panel (malignancy score, reliability score, and saliency map). We systematically manipulated five experimental onboarding conditions varying example-based information and limitation disclosures with five stimulus packages naturally varying observed prediction quality. Calibration was operationalized as the deviation between trust-related judgments (TAIS and case-wise ratings) and objective performance benchmarks for the encountered cases, analysed with hierarchical mixed-effects models. Only limitation disclosure for case-wise measures reliably impacts trust calibration, and short-term experience did not yield progressive calibration. Further, the experienced package of stimuli explained substantially more variance than the experimental manipulation. However, participants were hard-pressed to differentiate between case-wise perceived trust, trustworthiness, and accuracy estimation. We discuss implications for designing limitation communication and for measuring and analysing calibration metrics in XAI evaluations. All study materials and data of this study are publicly available for replication and further academic use. 

---
# PROTEA: Offline Evaluation and Iterative Refinement for Multi-Agent LLM Workflows 

**Authors**: Kazuki Kawamura, Satoshi Waki, Kei Tateno  

**Link**: [PDF](https://arxiv.org/pdf/2605.18032)  

**Abstract**: Multi-agent LLM workflows -- systems composed of multiple role-specific LLM calls -- often outperform single-prompt baselines, but they remain difficult to debug and refine. Failures can originate from subtle errors in intermediate outputs that propagate to downstream nodes, requiring developers to inspect long traces and infer which agent to modify. We present PROTEA, a unified interface for offline, test-driven improvement of multi-agent workflows. PROTEA executes a workflow, scores intermediate node outputs with configurable rubrics, and overlays per-node states and rationales on the workflow graph to localize likely bottlenecks. To support complex systems where final-answer references are the primary supervision, PROTEA performs backward node evaluation: it generates candidate node-level expectations from final-answer references and graph context, then compares them with observed node outputs. For selected nodes, PROTEA presents targeted prompt revisions as editable before/after comparisons, then automatically reruns and re-evaluates the workflow to show output changes and score trajectories within the same interface. In two production-adjacent workflows, PROTEA improved document-inspection accuracy from 64.3% to 83.9% and recommendation Hit@5 from 0.30 to 0.38. In a formative study with six experienced LLM developers, participants valued graph-level localization, per-node rationales, and editable before/after prompt revisions. 

---
# Quantum Sidecar Architectures for Hybrid AI Training and Inference: Stateful Protected Registers, Stateless Reset-and-Reprepare Circuits and Quantum Weight-State Outlook 

**Authors**: Y.Mo, G.D.Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.18031)  

**Abstract**: We propose a quantum sidecar architecture family for future hybrid AI training and inference. The central idea is not to store an entire Transformer in a small quantum memory, nor to claim one-shot collapse into a fully trained model or an optimal answer. Instead, we identify two physically distinct operating modes for quantum co-processors attached to classical large-model pipelines. The first is a stateful protected-register mode, in which a protected register stores a reusable quantum resource while an ancilla or temporary register performs QND-style readout. The second is a stateless reset-and-reprepare mode, in which each query prepares a task-conditioned quantum circuit, evolves over bounded training or inference control variables, measures candidate signals, resets the qubits, and repeats. We simulate the stateful mode using 2/4/6/8 protected-qubit density-matrix QND-style parity readout with one ancilla and a Qiskit cross-check. For the stateless mode, we include both an abstract candidate-update sampler and a circuit-level QAOA-style statevector sampler over structured candidate landscapes, followed by reset-overhead sensitivity analysis. The resulting framework positions quantum sidecars as bounded signal generators for optimizer-side sampling, adapter or expert selection, retrieval, routing, and reasoning-path proposal. As a speculative outlook, we introduce quantum weight-state sidecars: restricted quantum representations over model-control variables, not direct encodings of complete classical weight tensors. 

---
# FedSDR: Federated Self-Distillation with Rectification 

**Authors**: Ziheng Ren, Zhanming Shen, Hao Wang, Ning Liu, You Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.18028)  

**Abstract**: Federated fine-tuning of Large Language Models faces severe statistical heterogeneity. However, existing model-level defenses often overlook the root cause: intrinsic data distribution mismatches. In this work, we first establish Federated Self-Distillation (FedSD) as a fundamental and potent strategy. By projecting client representations into a smoothed ``model-understanding space,'' FedSD alone serves as a universal booster, demonstrating superior performance over conventional algorithms. Despite its success, we identify a subtle trade-off termed the Rewrite Paradox -- unconstrained self-distillation can inadvertently increase hallucinations and redundancy. To refine this paradigm, we further propose FedSDR (Federated Self-Distillation with Rectification), the ultimate reinforced framework. It augments FedSD with a dual-stream mechanism: a local LoRA-S (Smoothing) branch to implicitly absorb heterogeneity via distilled data, and a parallel global LoRA-R (Rectification) branch anchored to raw data to enforce factual correctness. By selectively aggregating only LoRA-R, FedSDR yields a globally aligned and faithful model. Extensive experiments verify its superior performance. 

---
# Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning 

**Authors**: Sunwoo Lee, Mingu Kang, Yonghyeon Jo, Seungyul Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.18024)  

**Abstract**: Cooperation is central to multi-agent reinforcement learning (MARL), yet learned coordination can be fragile when external perturbations disrupt inter-agent interactions. Prior robust MARL methods have primarily considered value-oriented attacks, leaving a gap in robustness when interaction structures themselves are corrupted. In this paper, we propose an interaction-breaking adversarial learning (IBAL) framework that takes an information-theoretic view to construct attacks that impede coordination by perturbing agents' observations and actions, and trains agents to perform reliably under such disruptions. Empirically, our approach improves robustness over existing robust MARL baselines across diverse attack settings and yields stronger performance even under agent-missing scenarios. 

---
# Unveiling Memorization-Generalization Coexistence: A Case Study on Arithmetic Tasks with Label Noise 

**Authors**: Linyu Liu, Pinyan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18022)  

**Abstract**: Highly over-parameterized models can simultaneously memorize noisy labels and generalize well, yet how these behaviors coexist remains poorly understood. In this work, we investigate the underlying mechanisms of this coexistence using modular arithmetic tasks under heavy label noise. Through extensive experiments on two-layer neural networks, we find that larger models tend to generalize better under appropriate optimization and model configurations, while noisy labels are memorized faster than clean data. Over-parameterized models internally form a generalization structure, but its expression in the output is suppressed by the need to fit noisy labels. Remarkably, even with 80\% label noise, near-perfect test accuracy can be achieved by extracting this internal structure using frequency-based methods. We further propose a task-agnostic method to partition networks into generalization and memorization components. Although this subnetwork improves generalization, it is limited compared with frequency-based extraction, indicating that the generalization structure is distributed across neurons and motivating the development of new tools to retrieve generalizable knowledge from over-parameterized networks. 

---
# See What I Mean: Aligning Vision and Language Representations for Video Fine-grained Object Understanding 

**Authors**: Boyuan Sun, Bowen Yin, Yuanming Li, Xihan Wei, Qibin Hou  

**Link**: [PDF](https://arxiv.org/pdf/2605.18018)  

**Abstract**: We present SWIM (See What I Mean), a novel training strategy that aligns vision and language representations to enable fine-grained object understanding solely from textual prompts. Unlike existing approaches that require explicit visual prompts, such as masks or points, SWIM leverages mask supervision only during training to guide cross-modal attention, allowing the model to automatically attend to the user-specified object at inference. Our cross-attention analysis of pretrained multimodal large languagemodels (MLLMs) reveals a systematic discrepancy: Attribute words produce sharp, localized activations in the visual modality, whereas object nouns yield diffuse and scattered patterns due to semantic reference bias and distributed high-level representations. To address this misalignment, we construct NL-Refer, an enriched dataset, in which each object mask is paired with a precise natural language referring expression. SWIM extracts multi-layer cross-attention maps from object nouns and enforces spatial consistency with ground-truth masks. Experimental results demonstrate that SWIM substantially improves text-visual alignment and achieves superior performance over visual-prompt-based methods on fine-grained object understanding benchmarks. The code and data are available at \href{this https URL}{this https URL}. 

---
# TinySAM 2: Extreme Memory Compression for Efficient Track Anything Model 

**Authors**: Zhaoyuan Ding, Yijing Yang, Han Shu, Xinghao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.18013)  

**Abstract**: Segment Anything Model 2 (SAM 2) serves as a core foundation model in the field of video segmentation. Building upon the original SAM model, it introduces a memory bank mechanism and demonstrates outstanding performance in tasks such as semi-supervised video object segmentation and tracking anything. However, the complex computational characteristics of SAM 2's multi-stage image encoder and memory module have raised the barrier to the model's deployment in practical applications. To address this issue, we propose TinySAM 2, a lightweight video segmentation model that balances performance and efficiency. First, a memory quality management mechanism is introduced to select and retain high-informative historical frames as the memory. In addition, a joint-spatial-temporal token compression is proposed that reduces the memory storage and computational cost. Specifically, average pooling is employed to first compress redundancy tokens in the spatial domain. In the temporal domain, informative tokens are selected across frames in the memory bank based on token-level similarity measurement. Besides, we take RepViT as the lightweight image encoder, which further reduces the model parameters. Extensive experiments on challenging datasets such as DAVIS and SA-V demonstrate that TinySAM 2 achieves 90% of the performance of SAM 2.1, with only 7% memory tokens and 3% training data. This study effectively alleviates the bottlenecks in parameter count, computational load, and deployment costs associated with SAM 2, providing a resource-efficient solution for the widespread application of video segmentation models on devices. 

---
# SAS: Semantic-aware Sampling for Generative Dataset Distillation 

**Authors**: Mingzhuo Li, Guang Li, Linfeng Ye, Jiafeng Mao, Takahiro Ogawa, Konstantinos N. Plataniotis, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2605.18012)  

**Abstract**: Deep neural networks have achieved impressive performance across a wide range of tasks, but this success often comes with substantial computational and storage costs due to large-scale training data. Dataset distillation addresses this challenge by constructing compact yet informative datasets that enable efficient model training while maintaining downstream performance. However, most existing approaches primarily emphasize matching data distributions or downstream training statistics, with limited attention to preserving high-level semantic information in the distilled data. In this work, we introduce a semantic-aware perspective for dataset distillation by leveraging Contrastive Language-Image Pretraining (CLIP) as a semantic prior for post-sampling. Our goal is to obtain distilled datasets that are not only compact but also semantically class-discriminative and diverse. To this end, we design three semantic scoring functions that quantify class relevance, inter-class separability, and intra-set diversity in a pretrained semantic space. Based on image pools generated by existing distillation methods, we further develop a two-stage strategy for effective sampling: the first stage filters semantically discriminative samples to form a reliable candidate set, and the second stage performs a dynamic diversity-aware selection to reduce redundancy while preserving semantic coverage. Extensive experiments across multiple datasets, image pools, and downstream models demonstrate consistent performance gains, highlighting the effectiveness of incorporating semantic information into dataset distillation. 

---
# Spiker-LL: An Energy-Efficient FPGA Accelerator Enabling Adaptive Local Learning in Spiking Neural Networks 

**Authors**: Alessio Caviglia, Filippo Marostica, Alessandro Savino, Stefano Di Carlo  

**Link**: [PDF](https://arxiv.org/pdf/2605.18003)  

**Abstract**: Deploying adaptive intelligence at the edge remains challenging due to the high computational and energy cost of training neural models. Spiking Neural Networks (SNNs) offer a promising alternative, but enabling on-device learning requires hardware-algorithm co-design. This paper presents SPIKER-LL, an FPGA-based SNN accelerator that extends the open-source Spiker+ inference architecture with efficient support for the STSF local learning rule. Through targeted microarchitectural extensions, SPIKER-LL performs inference and online learning with minimal overhead. Across MNIST, F-MNIST, and DIGITS, it achieves up to 93% accuracy, sub-millisecond latency, and less than 0.1 mJ per inference, while remaining DSP-free and highly scalable for edge-FPGA deployments. 

---
# Verify-Gated Completion as Admission Control in a Governed Multi-Agent Runtime: A Bounded Architecture Case Study 

**Authors**: Hai-Duong Nguyen, The-Xuan Tran  

**Link**: [PDF](https://arxiv.org/pdf/2605.17998)  

**Abstract**: As multi-agent systems move from short interactions to tool-using workflows with specialized roles and persistent state, completion becomes a runtime-control problem rather than a purely generative one. This preprint studies verify-gated completion as an admission-control pattern for governed multi-agent runtimes: agents may propose completion, but a read-only verifier decides whether the claim is admitted. Ambiguous or weakly evidenced cases resolve fail-closed, while packetized state and event traces preserve an audit path.
We examine one bounded reference implementation and ask what the released evidence can support about auditable, verify-gated completion. In the released verify-completed slice, the known-outcome invoked-event verify success share was 1,791/1,800 = 99.5%. This is an accounting measure over invoked verification events, not a task-completion, production-reliability, or benchmark-success rate. Task-level verify coverage is not computable; 1,762/1,801 rows came from one high-volume reporting cluster; and only 17 events were production-classified. A shadow Policy/Governance Verifier evaluation showed 1,526/1,548 = 98.58% rule agreement, 0/1,526 false-success among safe-to-proceed predictions, and blocked precision of 2/518 = 0.39%, so it remains advisory.
The evidence supports a narrow conclusion: under observed conditions, a read-only verify gate plus packetized admission records made completion decisions inspectable and fail-closed. Claims about deployed operation, safety guarantees, outcome gains, task-level coverage, recovery effectiveness, or external validity remain outside scope. 

---
# MARR: Module-Adaptive Residual Reconstruction for Low-Bit Post-Training Quantization 

**Authors**: Le Su, Xing Luo, Zhi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.17997)  

**Abstract**: Recently, residual reconstruction-based model quantization methods have achieved promising performance in low-bit post-training quantization (PTQ) by introducing cross-layer residuals to reduce error accumulated from previous this http URL, these residuals may also introduce additional bias arising from the Hessian-approximation (HA) assumption underlying reconstruction-based PTQ, leading to suboptimal quantization this http URL this work, we analyze that multiplying the residual term by a scaling coefficient provides a direct way to mitigate the HA bias associated with residual strength, while preserving accumulated-error correction. More importantly, we observe that this trade-off is module-dependent, making a single global residual strength insufficient to balance effective correction and residual-related bias across this http URL on these observations, we propose Module-Adaptive Residual Reconstruction (MARR), which assigns a module-specific scaling coefficient to adaptively balance accumulated-error correction and residual-related HA bias for each this http URL avoid expensive per-module coefficient search and obtain a stable coefficient estimate, we design a Proportional-Integral-Derivative (PID)-based adaptive update strategy that uses reconstruction error as feedback to progressively refine this coefficient. Experiments on several typical large language models (LLMs) and vision transformers (ViTs) demonstrate the effectiveness of MARR under low-bit quantization (less than or equal to 4-bit), achieving up to 20.2% performance gains on LLMs and up to 4.6% relative gains on ViTs over the residual reconstruction state-of-the-art this http URL will be made publicly available upon acceptance. 

---
# Towards Sustainable Growth: A Multi-Value-Aware Retrieval Framework for E-Commerce Search 

**Authors**: Yifan Wang, Yixuan Wang, YiDan Liang, Qiang Liu, Fei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.17994)  

**Abstract**: New item growth is critical for maintaining a healthy ecosystem in large-scale e-commerce platforms. However, existing systems tend to prioritize presenting users with already popular items, a phenomenon often referred to as the "Matthew effect". In the context of search retrieval, current cold-start models suffer from the misalignment between training objectives and online business metrics, and they lack effective mechanisms to measure an item's growth potential. In this paper, we propose a Multi-Value-Aware retrieval framework tailored for e-commerce search, designed to better align with the cascaded online values across different stages of the search system while balancing immediate conversion and long-term item growth. Our framework GrowthGR consists of two key components: an Item Long-term Transaction Value Prediction (ItemLTV) module and a Multi-Value-Aware Generative Retrieval (MultiGR) module. First, in the ItemLTV module, we employ counterfactual inference to quantify the long-term value increment attributable to a single user interaction. Second, in the MultiGR module, building upon a semantic-ID-based generative retrieval architecture, we leverage structured samples with the search cascade signals and adopt a Multi-Value-Aware Policy Optimization (MoPO) training paradigm to align with multi-stage online values, while explicitly balancing short-term transactional value and long-term growth potential estimated by ItemLTV. We successfully deployed GrowthGR on Taobao's production platform, achieving a substantial 5.3% lift in new item GMV while delivering a non-trivial 0.3% gain in overall search GMV. Extensive online analysis and A/B testing demonstrate its positive impact on the overall ecosystem value. 

---
# Stable Audio 3 

**Authors**: Zach Evans, Julian D. Parker, Matthew Rice, CJ Carr, Zack Zukowski, Josiah Taylor, Jordi Pons  

**Link**: [PDF](https://arxiv.org/pdf/2605.17991)  

**Abstract**: Stable Audio 3 is a family of fast latent diffusion models (small, medium, large) for variable-length audio generation and editing. Since our models can generate several minutes of audio, variable-length generations are key to avoid the cost of producing full-length generations for short sounds. We also support inpainting, enabling targeted audio editing and the continuation of short recordings. Our latent diffusion models operate on top of a novel semantic-acoustic autoencoder that projects audio into a compact latent space, enabling efficient diffusion-based generation while preserving audio fidelity and encouraging semantic structure in the latent. Finally, we run adversarial post-training to both accelerate inference and improve generation quality, reducing the number of inference steps while improving fidelity and prompt adherence. Stable Audio 3 models are trained on licensed and Creative Commons data to generate music and sounds in less than a 2s on an H200 GPU and less than a few seconds on a MacBook Pro M4. We release the weights of small and medium, that can run on consumer-grade hardware, together with their training and inference pipeline. 

---
# Predictive Prefetching for Retrieval-Augmented Generation 

**Authors**: Wuyang Zhang, Shichao Pei  

**Link**: [PDF](https://arxiv.org/pdf/2605.17989)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves factual grounding in large language models but suffers from substantial latency due to synchronous retrieval. While recent work explores asynchronous retrieval, existing approaches rely on heuristic coordination between retrieval and generation and assume stable information demands during decoding that often break in complex, multi-domain settings. In this paper, we propose an advanced asynchronous retrieval framework that enables predictive prefetching aligned with evolving information needs. The framework explicitly predicts when retrieval should be triggered and what information should be retrieved using three components, a retrieval predictor, a context monitor, and a query generator, by exploiting semantic precursors in generation dynamics that emerge several tokens before uncertainty becomes critical. Experiments on multiple benchmarks demonstrate up to 43.5% end-to-end latency reduction and 62.4% improvement in time-to-first-token, while maintaining answer quality comparable to synchronous RAG baselines. 

---
# LivePI: More Realistic Benchmarking of Agents Against Indirect Prompt Injectio 

**Authors**: Lei Zhao, Abhay Bhaskar, Edgar Dobriban  

**Link**: [PDF](https://arxiv.org/pdf/2605.17986)  

**Abstract**: AI agents such as OpenClaw are increasingly deployed in local workflows with access to external tools. This creates indirect prompt-injection (IPI) risk: an agent may execute harmful instructions embedded in untrusted inputs such as email, downloaded files, webpages, repositories, or group-chat messages. Existing evaluations are often small, purely simulated, or focused on a narrow set of channels. We introduce LivePI (Live Prompt Injection), a structured benchmark for IPI risk in a production-like but test-controlled environment. LivePI covers seven input surfaces, twelve attack/rendering families, and five malicious goals, including protected-information exfiltration, unauthorized security-control changes, unsafe code retrieval or execution, inbox-summary exfiltration, and cryptocurrency transfer. We run LivePI on a real virtual machine with live but test-controlled email, chat, web, local-file, repository, and wallet interfaces. Across GPT-5.3-Codex, Claude Opus 4.6, Gemini 3.1 Pro, Kimi K2.5, and GLM-5, total attack success rates range from 10.7% to 29.6%. Group-chat injection is uniformly successful across the evaluated backbones in our deployment, and repository-link attacks produce high-severity failures despite a small denominator. We also evaluate a two-layer defense consisting of prompt-level filtering and pre-execution tool-call authorization. In the GPT-5.3-Codex setting, the defense intercepts all tested malicious-goal completions in LivePI before execution while preserving benign utility on PinchBench-derived workloads. 

---
# SAFE-SVD: Sensitivity-Aware Fidelity-Enforcing SVD for Physics Foundation Models 

**Authors**: Chengjie Hong, Feixiang He, Yiheng Zeng, Lulu Kang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17985)  

**Abstract**: We propose a new method for compressing physics foundation models (PFMs) which is a new trend in AI for Science. While model compression is essential for reducing memory use and accelerating inference in large foundation models, it remains under-explored for PFMs, where preserving physical fidelity is crucial. The challenge lies in the functional nature of physics data, where partial derivatives encode spatiotemporal dynamics and exhibit high sensitivity to compression. Conventional compression methods ignore this structure, often causing severe performance degradation or failure. To address this, we introduce a sensitivity-aware fidelity-enforcing compression framework that explicitly models loss-aware layer sensitivity in the output function space during compression. This provides a new route to compressing scientific foundation models while preserving accuracy and physical fidelity. Experiments show substantial gains over existing methods across multiple models and datasets, achieving significantly higher compression ratios while maintaining accuracy, in some cases by orders of magnitude. More broadly, the work potentially leads to a new subfield of efficient, deployable, and sustainable scientific foundation models in AI for Science. 

---
# Babel: Jailbreaking Safety Attention via Obfuscation Distribution Optimized Sampling 

**Authors**: Ziwei Wang, Jing Chen, Ruichao Liang, Zhi Wang, Yebo Feng, Ju Jia, Ruiying Du, Cong Wu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17971)  

**Abstract**: Despite rigorous safety alignment, Large Language Models (LLMs) remain vulnerable to jailbreak attacks. Existing black-box methods often rely on heuristic templates or exhaustive trials, lacking mechanistic interpretability and query efficiency. In this study, we investigate an intrinsic vulnerability in the safety mechanisms of LLMs, where safety alignment relies on a small set of sparsely distributed attention heads, leaving much of the representational space weakly monitored. We formalize this phenomenon with a mathematical jailbreaking model that characterizes the delicate boundary of effective text obfuscation and analytically explains observed jailbreak behaviors. Guided by this model, we propose Babel, an efficient black-box attack framework that exploits the identified safety gap through systematic obfuscation sampling with iterative, feedback-driven distribution refinement, enabling reliable and high-success jailbreak attacks without access to model internals. Comprehensive evaluations on frontier commercial models demonstrate that Babel achieves state-of-the-art attack success rates and superior query efficiency. Specifically, compared to state-of-the-art methods, Babel increases the attack success rate on GPT-4o from 41.33% to 82.67% and on Claude-3-5-haiku from 38.33% to 78.33% within an average of 40 queries, providing a robust red-teaming methodology for LLMs safety research. 

---
# BLAgent: Agentic RAG for File-Level Bug Localization 

**Authors**: Md Afif Al Mamun, Gias Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2605.17965)  

**Abstract**: Bug localization remains a key bottleneck in downstream software maintenance tasks, including root cause analysis, triage, and automated program repair (APR), despite recent advances in large language model (LLM)-based repair systems. File-level bug localization is especially critical in hierarchical pipelines, where errors can propagate to downstream stages such as statement-level localization or patch generation. While Retrieval-Augmented Generation (RAG) offers a promising direction for grounding LLMs in repository context, existing RAG pipelines rely on static retrieval and lack the reasoning needed to identify faulty code accurately. In this work, we present BLAgent, a novel agentic RAG framework for file-level bug localization that integrates three key ideas: (i) code structure-aware repository encoding with path-augmented AST-based chunking, (ii) dual-perspective query transformation capturing both structural and behavioral signals, and (iii) two-phase agentic reranking combining symbolic inspection with evidence-grounded reasoning. Unlike prior graph-based or multi-hop agentic approaches, BLAgent performs bounded reasoning over a compact candidate set, balancing accuracy and cost. On SWE-bench Lite, BLAgent attains over 78% Top-1 accuracy with open-source models and over 86% with a closed-source model, while being over 18x cheaper than the strongest baseline using the same model. When integrated into an APR framework, it improves end-to-end repair success by over 20%. 

---
# A More Word-like Image Tokenization for MLLMs 

**Authors**: Hyun Lee, Hyemin Jeong, Yejin Kim, Hyungwook Choi, Hyunsoo Cho, Soo Kyung Kim, Joonseok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.17954)  

**Abstract**: Modern multimodal large language models (MLLMs) typically keep the language model fixed and train a visual projector that maps the pixels into a sequence of tokens in its embedding space, so that images can be presented in essentially the same form as text. However, the language model has been optimized to operate on discrete, semantically meaningful tokens, while prevailing visual projectors transform an image into a long stream of continuous and highly correlated embeddings. This causes the visual tokens to behave differently from the word-like units that LLMs are originally trained to understand. We propose a novel Disentangled Visual Tokenization (DiVT) that clusters patch embeddings into coherent semantic units, so each token corresponds to a distinct visual concept instead of a rigid grid cell. DiVT further adapts its token budget to image complexity, providing an explicit accuracy-compute trade-off modifying neither the vision encoder nor the language model. Across diverse multimodal benchmarks, DiVT matches or surpasses baselines with significantly fewer visual tokens, demonstrating robustness under limited token budgets, significantly reducing memory cost and latency while making visual inputs more compatible with LLMs. Our code is available at this https URL. 

---
# Training data attribution in diffusion models via mirrored unlearning and noise-consistent skew 

**Authors**: Joan Serrà, Dipam Goswami, Fabio Morreale, Wei-Hsiang Liao, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2605.17938)  

**Abstract**: Training data attribution (TDA) should enable generative model interpretability and foster a variety of related downstream tasks. Nonetheless, current TDA approaches lack reliability and robustness, preventing their adoption in real-world setups. In this paper, we take a decisive step towards more reliable and robust TDA for diffusion models. We propose to perform TDA with mirrored unlearning and noise-consistent skew (MUCS). The idea is to fine-tune a second model with bounded mirrored gradient ascent, and to measure the normalized skew of this model with respect to the original one using consistent noise samples. We show that, while being conceptually simple and generic, MUCS systematically outperforms existing methods on three different datasets by a large margin. We additionally study the effect that core design choices have on final performance, and analyze novel aspects regarding the overlap of influential instances across generated items and the potential of ensembling TDA approaches. We believe that our findings may have broader implications for more general unlearning setups, as well as for tasks requiring the comparison of diffusion losses. 

---
# BacktestBench: Benchmarking Large Language Models for Automated Quantitative Strategy Backtesting 

**Authors**: Zhensheng Wang, Wenmian Yang, Qingtai Wu, Lequan Ma, Yiquan Zhang, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2605.17937)  

**Abstract**: Quantitative backtesting is essential for evaluating trading strategies but remains hampered by high technical barriers and limited scalability. While Large Language Models (LLMs) offer a transformative path to automate this complex, interdisciplinary workflow through advanced code generation, tool usage, and agentic planning, the practical realization is significantly challenged by the current lack of a large-scale benchmark dedicated to automated quantitative backtesting, which hinders progress in this field. To bridge this critical gap, we introduce BacktestBench, the first large-scale benchmark for automated quantitative backtesting. Built from over 6 million real market records, it comprises 18,246 meticulously annotated question-answering pairs across four task categories: metrics calculation, ticker selection, strategy selection, and parameter confirmation. We also propose AutoBacktest, a robust multi-agent baseline that translates natural language strategies into reproducible backtests by coordinating a Summarizer for semantic factor extraction, a Retriever for validated SQL generation, and a Coder for Python backtesting implementation. Our evaluation on 23 mainstream LLMs, complemented by targeted ablations, identifies key factors that influence end-to-end performance and highlights the importance of grounded verification and standardized indicator representations. 

---
# Prompt Compression in Diffusion Large Language Models: Evaluating LLMLingua-2 on LLaDA 

**Authors**: Sterling Huang, Abigayle Brown, Jiyoo Noh, Jiakang Xu, Wantong Huo, Kaung Myat Kyaw, Jonathan Chan  

**Link**: [PDF](https://arxiv.org/pdf/2605.17932)  

**Abstract**: Prompt compression reduces inference cost and context length in large language models, but prior evaluations focus primarily on autoregressive architectures. This study investigates whether prompt compression transfers effectively to diffusion large language models (DLLMs) using LLMLingua-2, specifically the 8B-parameter DLLM LLaDA. We evaluate compression performance on GSM8K, DUC2004, and ShareGPT using 250 prompts per dataset at an approximate 2$\times$ compression ratio, across mathematical reasoning, prompt reconstruction, and summarization tasks. Outputs generated from original prompts, compressed prompts, reconstructed prompts, and reconstructed-prompt reasoning were compared using exact-match accuracy, BLEU, ROUGE, and BERTScore. Results show that semantic preservation does not necessarily imply stable downstream behavior in diffusion models. Summarization tasks remained comparatively robust under compression, while mathematical reasoning degraded substantially despite high semantic similarity scores. Reconstruction experiments further showed that semantically similar prompts may still omit reasoning-critical information required for stable denoising. Across tasks, BERTScore recall was consistently lower than precision, suggesting that compression failures are primarily driven by information omission rather than semantic drift. These findings indicate that prompt compression methods designed for autoregressive models do not transfer uniformly to diffusion large language models and motivate the development of diffusion-aware compression strategies. 

---
# AdaptiveLoad: Towards Efficient Video Diffusion Transformer Training 

**Authors**: Yucheng Guo, Yongjian Guo, Zhong Guan, Haoran Sun, Wen Huang, Wanting Xu, Jing Long, Shuai Di, Junwu Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.17923)  

**Abstract**: In video generation models, particularly world models, training large-scale video diffusion Transformers (such as DiT and MMDiT) poses significant computational challenges due to the extreme variance in sequence lengths within mixed-mode datasets. Existing bucket-based data loading strategies typically rely on "equal token length" constraints. This approach fails to account for the quadratic complexity of self-attention mechanisms, leading to severe load imbalance and underutilization of GPU resources.
This paper proposes \textit{AdaptiveLoad}, an integrated optimization framework consisting of two core components: (1) A dual-constraint adaptive load balancing system, which eliminates long-sequence bottlenecks by simultaneously limiting memory consumption and computational load ($B \times S^p \le M_{\text{comp}}$); (2) A fused LayerNorm-Modulate CUDA kernel, which utilizes a D-tile coalesced reduction strategy to increase throughput and alleviate memory pressure. Experimental results on the Wan 2.1 world model demonstrate that our method reduces the computational imbalance rate from 39\% to 18.9\%, improves peak VRAM utilization efficiency by 22.7\%, and achieves an overall training throughput increase of 27.2\%. 

---
# Domain Transfer Becomes Identifiable via a Single Alignment 

**Authors**: Sagar Shrestha, Subash Timilsina, Hoang-Son Nguyen, Xiao Fu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17918)  

**Abstract**: Domain transfer (DT) maps source to target distributions and supports tasks such as unsupervised image-to-image translation, single-cell analysis, and cross-platform medical imaging. However, DT is fundamentally ill-posed: push-forward mappings are generally non-identifiable, as measure-preserving automorphisms (MPAs) preserve marginals while altering cross-domain correspondences, leading to content-misaligned translation. Recent work shows that MPAs can be eliminated by jointly transferring multiple corresponding source/target conditional distributions, but supervision signals labeling such conditionals are not always available in practice. We develop an alternative route to DT identifiability. Under a structural sparsity condition on the Jacobian support pattern, we show that distribution matching together with a single paired anchor sample suffices to identify the ground-truth transfer -- requiring substantially less supervision than prior approaches. To enable practical high-dimensional learning, we further propose an efficient Jacobian sparsity regularizer based on randomized masked finite differences, yielding a scalable surrogate without explicit Jacobian evaluation. Empirical results on synthetic and real-world DT tasks validate the theory. 

---
# One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception 

**Authors**: Yang Li, Weize Li, Quan Yuan, Congzhang Shao, Guiyang Luo, Yunqi Ba, Xuanhan Zhu, Xinyuan Ding, Xiaoyuan Fu, Jinglin Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.17907)  

**Abstract**: By sharing intermediate features, collaborative perception extends each agent's sensing beyond standalone limits, but real-world feature modality heterogeneity remains a key barrier to effective fusion. Most existing methods, including direct adaption and protocol-based transformation, typically rely on training adapters for newly emerging feature modalities and often require additional retraining or fine-tuning. Such repeated training is costly and is often infeasible across manufacturers due to model and data privacy constraints, limiting real-world scalability. To address this issue, we propose UniTrans, a universal any-to-any feature modality translation model that instantiates translators on the fly for arbitrary modalities.
UniTrans pretrains a bank of translator expert parameters and learns their combination coefficients as a function of source-to-target modality mapping. The mapping is measured in a modality-intrinsic latent space, where an intrinsic encoder extracts modality-specific yet scene-invariant codes from single-frame intermediate features, enabling UniTrans to instantiate translators in a zero-shot manner.
Experiments on OPV2V-H and DAIR-V2X demonstrate that UniTrans consistently outperforms state-of-the-art methods in both simulated and real-world settings, enabling efficient any-to-any translation through a universal model. The code is available at this https URL. 

---
# DCFold: Efficient Protein Structure Generation with Single Forward Pass 

**Authors**: Zhe Zhang, Yuanning Feng, Yuxuan Song, Keyue Qiu, Hao Zhou, Wei-Ying Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.17899)  

**Abstract**: AlphaFold3 introduces a diffusion-based architecture that elevates protein structure prediction to all-atom resolution with improved accuracy. This state-of-the-art performance has established AlphaFold3 as a foundation model for diverse generation and design tasks. However, its iterative design substantially increases inference time, limiting practical deployment in downstream settings such as virtual screening and protein design. We propose DCFold, a single-step generative model that attains AlphaFold3-level accuracy. Our Dual Consistency training framework, which incorporates a novel Temporal Geodesic Matching (TGM) scheduler, enables DCFold to achieve a 15x acceleration in inference while maintaining predictive fidelity. We validate its effectiveness across both structure prediction and binder design benchmarks. 

---
# Attention Sinks and Outliers in Attention Residuals 

**Authors**: Haozheng Luo, Haoran Dai, Shaoyang Zhang, Xi Chen, Eric Hanchen Jiang, Yijiang Li, Jingyuan Huang, Chenghao Qiu, Chenwei Xu, Zhenyu Pan, Haotian Zhang, Binghui Wang, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.17887)  

**Abstract**: We propose OASIS, an outlier- and sink-aware technique built on inter-layer null signaling. As AttnResidual architectures introduce an additional depth-wise normalization channel, they improve inter-layer routing flexibility but also exacerbate attention sinks, activation outliers, and the resulting degradation in inference stability and quantization robustness. OASIS addresses this issue by introducing a Softmax1-based null space and coupling token-level null evidence to depth routing through an inter-layer null signal, thereby reducing sink-dominated routing and improving structural robustness. Theoretically, we show that the dual-normalization design of AttnResidual intensifies sink formation and quantization brittleness. Experimentally, we compare OASIS against five baselines on three real-world datasets and observe consistent improvements in both attention sink and post-quantization performance. Notably, OASIS achieves an average reduction of 9.26% in maximum infinity norm and 2.60% in average kurtosis across the evaluated settings, while lowering perplexity by 75.85% under W8A8 and improving GSM8K Pass@1 by 12.42% under W4A4. 

---
# Multi-agent AI systems outperform human teams in creativity 

**Authors**: Tiancheng Hu, Yixuan Jiang, Haotian Li, José Hernández-Orallo, Xing Xie, Nigel Collier, David Stillwell, Luning Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.17885)  

**Abstract**: Although artificial intelligence (AI) now matches or exceeds human performance across numerous cognitive tasks, creativity remains a highly contested frontier. As AI systems based on large language models (LLMs) are increasingly adopted in research and innovation, it is essential to understand and augment their creativity. Here we demonstrate that multi-agent LLM teams not only surpass single agents, but also substantially outperform human teams in creativity (Cohen's d=1.50) across 4,541 multi-agent LLM ideas and 341 human-team ideas on six diverse problem-solving tasks. This advantage is driven by novelty while maintaining comparable usefulness. To investigate the generative processes in both groups, we represent conversations as paths through semantic space using neural language model representations. Both LLM and human teams produce more creative ideas when conversations range widely rather than staying centered on a single theme (low global coherence). However, the additional patterns that predict creativity differ: LLM teams benefit from efficient exploration (high semantic spread, shorter paths), while human teams benefit from maintaining smooth conversational flow (high local coherence, frequent pivots). Additionally, we identify model choice and discussion structure as orthogonal design levers that together explain 26.8% of variance in LLM conversational dynamics, paving the way for systematic approaches to developing multi-agent systems with augmented creative capabilities. 

---
# Guard: Scalable Straggler Detection and Node Health Management for Large-Scale Training 

**Authors**: Guanliang Liu, Abhinandan Patni, Congzhu Lin, Zoe Zeng, Jack Wittmayer, Josh Wu, Ashvin Nihalani, Binxuan Huang, Yinghong Liu, Rory Na, Anthony Ko, Alexander Zhipa, Cong Cheng, Mi Sun, Vijay Rajakumar, Rejith George Joseph, Parthasarathy Govindarajen  

**Link**: [PDF](https://arxiv.org/pdf/2605.17879)  

**Abstract**: Training frontier-scale foundation models involves coordinating tens of thousands of GPUs over multi-month runs, where even minor performance degradations can accumulate into substantial efficiency losses. Existing health-check mechanisms, such as NCCL tests or GPU burn-in, primarily focus on functional correctness and often fail to detect fail-slow behaviors that silently degrade system performance. In this paper, we present Guard, a scalable system for detecting stragglers and ensuring node health in large-scale training clusters. Guard combines lightweight online performance monitoring during training with an offline node-sweep mechanism that systematically evaluates and qualifies nodes before they participate in production workloads. This design enables Guard to detect both acute failures and long-running fail-slow behaviors that traditional diagnostics cannot capture. Deployed on large-scale foundation model pretraining workloads, Guard improves mean FLOPs utilization by up to 1.7x, reduces run-to-run training step variance from 20% to 1%, increases mean time to failure (MTTF), and significantly reduces operational and debugging overhead. These results demonstrate that proactive straggler detection and systematic node qualification are critical for maintaining stable and efficient large-scale training. 

---
# HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents 

**Authors**: Woongyeng Yeo, Yumin Choi, Taekyung Ki, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17873)  

**Abstract**: Training long-horizon LLM agents with reinforcement learning is challenging because sparse outcome rewards reveal whether a task succeeds, but not which intermediate actions caused the outcome or how they should be corrected. Recent methods alleviate this issue by generating rewards or textual hints from turn-level action-output signals, or by using feedback-conditioned self-distillation. However, generating feedback at every turn is inefficient when many intermediate turns are already successful or neutral, and applying feedback at a fixed or misaligned turn often fails to supervise the actions that contributed to the failure. To bridge this gap, we propose HINT-SD, a targeted self-distillation framework that uses full-trajectory hindsight to select failure-relevant actions and applies feedback-conditioned distillation only on targeted action spans. Experiments on BFCL v3 and AppWorld show that our method improves over the dense per-turn feedback baseline by up to 18.80 percent while achieving 2.26$\times$ lower time per training step, suggesting that selecting where to distill is a key factor for both effective and efficient long-horizon agent training. 

---
# $\boldsymbol{f}$-OPD: Stabilizing Long-Horizon On-Policy Distillation with Freshness-Aware Control 

**Authors**: Xianwei Chen, Shimin Zhang, Jibin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17862)  

**Abstract**: Scaling on-policy distillation (OPD) for large language models (LLMs) confronts a fundamental tension: asynchronous execution is necessary for system efficiency, but structurally deviates from the ideal on-policy objective. To address this challenge, we theoretically decompose the objective discrepancy into rollout drift and supervision drift, capturing staleness in student rollout and teacher context, respectively. Building on this, we introduce a sample-level freshness score that quantifies the reliability of a buffered sample with respect to the on-policy objective. Guided by this signal, we further propose f-OPD, a novel framework that adaptively regulates stale-sample influence and constrains policy drift accumulated under asynchronous training. Across reasoning, tool-use, and coding-agent tasks of increasing interaction horizon, f-OPD consistently achieves task performance comparable to synchronous optimization while largely retaining the throughput advantages of asynchronous execution. Our results establish the first recipe for achieving a performance-efficiency trade-off in OPD, paving the way for long-horizon agentic post-training at scale. 

---
# PAREDA: A Multi-Accent Speech Dataset of Natural Language Processing Research Discussions 

**Authors**: Sicheng Jin, Dipankar Srirag, Aditya Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2605.17860)  

**Abstract**: While modern Automatic Speech Recognition (ASR) systems achieve high accuracy on benchmark corpora, their performance often degrades when there is real-world variability. This work focuses on variability arising due to accented, spontaneous, and domain-specific speech. In particular, we introduce PAper REading DAtaset (PAREDA), a first-of-its-kind multi-accent speech dataset consisting of discussions on academic Natural Language Processing (NLP) papers between speakers with Australian, Indian-English, and Chinese English accents. Each session elicits a spontaneous monologue (a summary of a paper's abstract) and a non-monologue (a question-and-answer session between participants), resulting in a corpus rich with technical jargon and conversational phenomena. We evaluate the performance of SOTA ASR models on PAREDA, analysing the impact of accent mixing and increased speech rate. Our results show that, in the zero-shot setting, models perform worse, confirming the dataset's challenging nature. However, fine-tuning on PAREDA significantly reduces the Word Error Rate (WER), demonstrating that our dataset captures linguistic characteristics often missing from existing corpora. PAREDA serves as a valuable new resource for building and evaluating more robust and inclusive ASR systems for specialised, real-world applications. 

---
# Generating Pretraining Tokens from Organic Data for Data-Bound Scaling 

**Authors**: Zichun Yu, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.17849)  

**Abstract**: LLM pretraining is shifting from a compute-bound to a data-bound regime, where available human (organic) text falls far short of scaling demands. However, reaching the data-bound regime does not mean the model has fully utilized its organic corpus. In this paper, we introduce SynPro, a synthetic data generation framework that helps LLMs more thoroughly learn from limited organic data. SynPro applies two operations, rephrasing and reformat, that present the same organic source in diverse forms to facilitate deeper learning without introducing external information. Both generators are optimized via reinforcement learning with quality, faithfulness, and data influence rewards, and are continuously updated as pretraining plateaus to target content the model has yet to absorb. We pretrain 400M and 1.1B models with 10% of their Chinchilla-optimal tokens (0.8B and 2.2B) from DCLM-Baseline, reflecting a realistic data-bound regime in frontier pretraining. Our results reveal that organic data is significantly underutilized by standard repetition: SynPro unlocks 3.7-5.2x the effective tokens of repetition, even surpassing the non-data-bound oracle that trains on equivalent unique data at the 1.1B scale. Analyses confirm that faithful, model-aware synthesis sustains data-bound scaling without causing distribution collapse. We open-source our code at this https URL. 

---
# Balancing Knowledge Distillation for Imbalance Learning with Bilevel Optimization 

**Authors**: Anh B.H. Nguyen, Ba Tho Phan, Viet Cuong Ta  

**Link**: [PDF](https://arxiv.org/pdf/2605.17839)  

**Abstract**: Knowledge distillation transfers knowledge from a high capacity teacher to a compact student using a mixture of hard and soft losses. On imbalanced data, a fixed weighting between hard and soft losses becomes brittle the learning process. Recent studies try to reweight these components in long-tailed settings. However, most of these meth- ods do not adapt weights at the sample-wise level and do not take into account the students behavior during training. To address this, we pro- pose BiKD - a bilevel framework that dynamically balances hard and soft losses for each sample. We employ a weight generation network that produces adaptive per-sample weights, guided by a small balanced vali- dation set. The student is now trained with an unconstrained combina- tion of weighted hard and soft losses, allowing the student to relax both terms. We further propose a multi-step SGD strategy to optimize the weight model more accurately and efficiently. Experiments on long-tailed CIFAR-10/100 show that our approach surpasses recent balanced distil- lation methods across imbalance factors. 

---
# Temporal Aware Pruning for Efficient Diffusion-based Video Generation 

**Authors**: Sheng Li, Yang Sui, Junhao Ran, Bo Yuan, Yue Dai, Xulong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17837)  

**Abstract**: Video diffusion models have recently enabled high-quality video generation with ViT-based architectures, but remain computationally intensive because generation requires attention computation over long spatiotemporal sequences. Token pruning has proven effective for ViTs and VLMs. However, most prior pruning methods are attention-based and operate per frame, failing to ensure the vital temporal coherence across frames in video generation tasks. In practice, naively adopting attention-only pruning causes noticeable degradation due to worsened background consistency, flickering, and reduced image quality. To address this, we propose TAPE, a training-free Temporal Aware Pruning for Efficient diffusion-based video generation. TAPE (i) applies temporal smoothing to align token-importance across adjacent frames and suppress selection jitter; and (ii) performs token reselection in selected layers to align token pruning with layers' diverse semantic focus and avoid error accumulation in specific areas; it also (iii) adopt a timestep-level budget scheduling that prunes aggressively at early noisy steps and relaxes pruning during fidelity-critical refinement. The experimental results show that TAPE delivers significant speedups while preserving high visual fidelity, outperforming prior token reduction approaches. 

---
# Efficient Bilevel Optimization for Meta Label Correction in Noisy Label Learning 

**Authors**: Ba Hoang Anh Nguyen, Viet Cuong Ta  

**Link**: [PDF](https://arxiv.org/pdf/2605.17833)  

**Abstract**: Training a deep neural network with noisy labels could reduce data annotation cost but may introduce noise into the learned model. In meta label correction approaches, an additional meta model besides the main model is trained with a small, clean dataset to correct the large, noisy dataset. However, the update of the meta model requires the computation of hypergradients at the inner step of the main model which signif- icantly increases the computational cost. To improve the training efficiency, we first introduce the dynamic barrier gradient descent into standard meta label correction. While this naive extenstion is able to speed up the training process to approximately first- order complexity, it lacks mechanisms to prevent the leakage of noisy signals to the main model and to stabilize the learning of the meta model. Based on this observation, we propose the EBOMLC method, which is designed with three key improvements including one-step inner loop update, mixture upper loss and alignment- aware dynamic barrier. Empirical results on CIFAR-10 and CIFAR-100 demonstrate that EBOMLC consistently outperforms other baselines, especially under high noise rate settings, while reducing training time of the meta label correction approach. 

---
# Content-Style Identification via Differential Independence 

**Authors**: Subash Timilsina, Hoang-Son Nguyen, Sagar Shrestha, Xiao Fu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17827)  

**Abstract**: Generative analysis often models multi-domain observations as nonlinear mixtures of domain-invariant content variables and domain-specific style variables. Identifying both factors from unpaired domains enables tasks such as domain transfer and counterfactual data generation. Prior work establishes identifiability under (block-wise) statistical independence between content and style, or via sparse Jacobian assumptions on the nonlinear mixing function, but such conditions can be restrictive in practice. In this work, we introduce content-style differential independence (CSDI), an alternative structural condition requiring that infinitesimal variations in content and style induce orthogonal directions on the data manifold, thereby enabling identifiability even when content and style are dependent and the Jacobian is dense. We operationalize this condition through a blockwise orthogonality constraint on the Jacobian subspaces associated with content and style. To support high-dimensional generative models, we design a stochastic regularizer based on numerical Jacobian approximation, enabling scalable training in settings such as high-resolution image generation. Experiments across multiple datasets corroborate the identifiability analysis and demonstrate practical benefits on counterfactual generation and domain translation. 

---
# CounterCount: A Diagnostic Framework for Counting Bias in Vision Language Models 

**Authors**: Reem Alzahrani, Hassan Alshanqiti, Bushra Bin Hemid, Zaid Alyafeai, Abdelrahman Eldesokey, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2605.17826)  

**Abstract**: Vision-Language Models (VLMs) excel at multimodal reasoning, yet it remains unclear whether their answers are grounded in visual evidence or driven by learned language and world priors. Counting provides a precise testbed: when visual evidence conflicts with canonical object knowledge, a model must rely on the image rather than a prototypical count. We introduce CounterCount, a diagnostic framework for counterfactual counting in VLMs, consisting of paired factual and counterfactual images with edited count-relevant attributes, verified answers, and localized evidence annotations. Evaluating recent VLMs, we find strong performance on factual images but consistent degradation under counterfactual attribute changes, indicating reliance on object-level priors even when contradictory visual evidence is present. Using localized annotations, we show that these failures are not solely due to missing or ambiguous visual evidence, but to models underweighting attention to count-relevant visual tokens. We introduce a unified inference-time attention modulation strategy that reweights selected visual tokens, improving counterfactual counting accuracy by up to 8% across multiple VLMs. Overall, CounterCount exposes prior-driven counting failures and provides diagnostic insights for designing future VLMs. 

---
# Why We Look Where We Look: Emergent Human-like Fixations of a Foveated Visual Language Model Maximizing Scene Understanding 

**Authors**: Shravan Murlidaran, Ziqi Wen, Sana Shehabi, Miguel P. Eckstein  

**Link**: [PDF](https://arxiv.org/pdf/2605.17823)  

**Abstract**: When humans view scenes without a specific task (free-viewing), they initially direct their eye movements toward the scene center and then fixate on people, text, objects being gazed at or grasped, and semantically meaningful regions. What these signature fixation patterns reflect and whether they optimize an underlying perceptual task remain unknown. We show that a computational agent with simulated foveation, trained to optimize scene comprehension, exhibits emergent human fixation signature patterns. In contrast, versions of the agent trained to search or classify scenes, or equipped with peripheral vision that was better or worse than human vision, predicted human fixation patterns less accurately. Thus, human free-viewing fixation patterns may emerge as a functional byproduct of optimizing scene comprehension under the biological constraints of foveated vision. 

---
# TierCheck: Tiered Checkpointing for Fault Tolerance in Large Language Model Training 

**Authors**: Shujie Han, Feng Jiang, Patrick P. C. Lee, Xiao Zhang, Zhijie Huang, Nannan Zhao, Xiaonan Zhao, Lichen Pan  

**Link**: [PDF](https://arxiv.org/pdf/2605.17821)  

**Abstract**: Large Language Model (LLM) training is frequently interrupted by a heterogeneous spectrum of failures, from common GPU crashes to catastrophic cluster-wide outages. Existing checkpointing systems rely on monolithic, single-tier storage backend, forcing a trade-off between state-saving overhead and recovery speed. We propose TierCheck, a cluster-aware tiered checkpointing system that aligns storage placement with failure heterogeneity. TierCheck adopts a three-tier design that maintains lightweight differential checkpoints in local and peer memory for fast localized recovery, while asynchronously migrating heavyweight base checkpoints to remote persistent storage. It also ensures strict global consistency across tiers without stalling training, and achieves fast cluster-aware checkpoint restoration during recovery. Evaluations on models up to 40 billion parameters show that TierCheck achieves low training overhead, reduces end-to-end checkpointing time to under 10s, and supports high-frequency checkpointing, ultimately striking an optimal balance between low-overhead persistence and fast recovery. 

---
# Virtues of Ordered Chaos: Planning with Topple Actions in Tabletop Stack Rearrangement 

**Authors**: Hao Lu, Rahul Shome  

**Link**: [PDF](https://arxiv.org/pdf/2605.17815)  

**Abstract**: Efficient object manipulation strategies have significant impact in automation applications. In this work, the stack rearrangement in tabletop settings is studied, with a focus on augmenting the task planning domain with richer nonprehensile aggregating actions, in particular the toppling of objects from a stack to the table. Toppling can compress long sequences of intermediate relocations. Computed plans need to interleave pick-and-place actions with topple throughout its plan based on the problem. In order to generate the task plan and model an abstraction to compute solutions that include both pick-and-place and topple actions, a novel aggregating gadget for topple is introduced. Using this directed graphical abstraction, candidate task plan computation becomes a variant of the pebble motion problem, treating objects as pebbles. Benchmarks are then reported in a IsaacSim-based physics simulation. Results highlight clear benefits of achieving faster execution than solely using pick-and-place actions. Though this work primarily investigates the topple action, we demonstrate that similar abstractions can model other aggregating actions of interest, like scoop. The current work provides a preliminary, strong indication of the promising benefits of abstractions for rich object interactions in manipulation applications. 

---
# One Model, Two Roles: Emergent Specialization in a Shared Recurrent Transformer 

**Authors**: Jucheng Shen, Barbara Su, Anastasios Kyrillidis  

**Link**: [PDF](https://arxiv.org/pdf/2605.17811)  

**Abstract**: Can a shared-weight recurrent Transformer develop distinct internal roles without being partitioned into separate modules? We study this in Asymmetric Input Recurrence (AIR), a minimal two-state reasoning architecture in which the same Transformer model is reused for both updates (per literature, L and H) and the only built-in difference in the update rule is that the encoded input is injected during L-updates but not H-updates. Across Sudoku-Extreme and Maze, decoded rollouts reveal a consistent split: $\zH$ behaves like a fully committed proposal state, whereas $\zL$ retains local uncertainty and shifting intermediate structure. Freeze experiments show that this split is, in practice, related to the model's state dynamics: in Sudoku, freezing $\zH$ reduces $\zL$'s content changes whereas freezing $\zL$ increases $\zH$'s, while in Maze, freezing either state increases content changes in the other state. Ablations show that to induce specialization, the shared model needs to be able to tell the two update types apart, either from input injection asymmetry or from a separate level token. Mechanistically, attention analysis shows that L-updates are consistently more local than H-updates in both Sudoku and Maze. Together, these results show that, in a two-state recurrent setting, a clear state-identity signal can induce stable, related functional roles inside a shared-parameter recurrent Transformer. Code is available at \href{this https URL}{\textcolor{blue}{this https URL}}. 

---
# Curriculum Group Policy Optimization: Adaptive Sampling for Unleashing the Potential of Text-to-Image Generation 

**Authors**: Baoteng Li, Xianghao Zang, Xinran Wang, Xiangyu Na, Zhixiang He, Hao Sun, Chi Zhang, Zhongjiang He, Tianwei Cao, Kongming Liang, Zhanyu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.17807)  

**Abstract**: Text-to-Image (T2I) generation has achieved remarkable progress in recent years. Meanwhile, reinforcement learning methods, particularly those based on Group Relative Policy Optimization (GRPO), have attracted widespread attention and been successfully applied to T2I tasks. However, the uniform sampling strategy commonly used during training often ignores the match between sample difficulty and the model's current learning capability, leading to low training efficiency. We argue that improving training efficiency requires continuously prioritizing prompts that match the model's evolving capability and remain actively learnable. To this end, we propose Curriculum Group Policy Optimization (CGPO), an adaptive curriculum training framework. During training, each prompt produces a group of images scored by a reward model. We use the variance of group rewards as an online proxy for prompt inconsistency. A higher variance suggests that the model has partially captured the prompt requirements but has not yet achieved stable mastery. Such prompts are more likely to provide useful learning signals, so we increase their sampling probabilities accordingly. Additionally, to address data imbalance in multi-category datasets, we design a category calibration method based on proportional fairness optimization, which balances training difficulty across categories. Experiments on GenEval, T2I-CompBench++, and DPG Bench demonstrate that our framework effectively improves generation performance. 

---
# Optimal Knock-Pick Planning for Tightly Packed Tabletop Blocks With Parallel Grippers 

**Authors**: Hao Lu, Rahul Shome  

**Link**: [PDF](https://arxiv.org/pdf/2605.17800)  

**Abstract**: Rearranging densely packed tabletop objects is challenging when parallel-gripper picks are infeasible without sufficient clearance around an object. This work studies the problem characteristics for practically motivated settings with uniformly sized blocks placed at planar tabletop grid locations. Since purely prehensile removal can become infeasible, a directional knock primitive is therefore introduced and the optimal knock-pick variant of the problem is formulated. The work proposes a series of abstractions wherein minimal constraining gadgets are covered to identify the necessary knocks. Utilizing a maximum-weight perfect matching on a graphical abstraction yields efficient polynomial-time computation of the optimal plan that minimizes the number of actions. Experiments are reported for increasing grid sizes in synthetic settings as well as in IsaacSim. The theoretical observations provide a promising stepping stone towards rigorously building efficient manipulation strategies that interleave prehensile and non-prehensile actions. 

---
# SocialMemBench: Are AI Memory Systems Ready for Social Group Settings? 

**Authors**: Olukunle Owolabi  

**Link**: [PDF](https://arxiv.org/pdf/2605.17789)  

**Abstract**: Memory systems for AI assistants were built for single-user dialogue and fail characteristically when applied to multi-party social group settings. This gap matters for the social assistants being built today: group-acting agents embedded in chat platforms, and proactive personal-assistant agents whose holistic model of a user must include their social context. Existing memory benchmarks evaluate dyadic or workplace dialogue; none targets multi-party social groups, where memory must anchor facts in shared history rather than professional roles, separate group norms from individual exceptions, and correctly attribute even after member departure. We introduce SocialMemBench, a benchmark of human-verified synthetic social group networks across five archetypes (close friends, family, recreational, interest community, acquaintance network) and three group-size tiers (4-30 members), with 430 personas and 7,355 conversation turns, yielding 1,031 QA pairs across nine question categories. Each category isolates an architectural capability, and the five failure modes (single-stream conflation, temporal-state overwrite, entity merging at scale, missing cross-persona knowledge, norm-individual conflation) are testable hypotheses; our two research probes Subject-Mem and SMG provide evidence on two, three remain open. A full-context Gemini 2.5 Flash reference reaches only 0.721 against a blind-critic reasoning-model mean of 0.98 on small networks, indicating the benchmark is genuinely difficult even with complete access to the conversation. Across all 43 networks, the four open-source memory frameworks evaluated (Mem0, LangMem, Graphiti, Cognee) cluster in the 0.12-0.18 question-weighted range with overlapping 95% CIs, well below an uncompressed retrieval reference of 0.345 and a matched-answerer full-context reference of 0.369 (GPT-4o-mini). Current memory systems show a measurable gap. 

---
# Systematic Evaluation of the Quality of Synthetic Clinical Notes Rephrased by LLMs at Million-Note Scale 

**Authors**: Jinghui Liu, Sarvesh Soni, Anthony Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.17775)  

**Abstract**: Large language models (LLMs) can generate or synthesize clinical text for a wide range of applications, from improving clinical documentation to augmenting clinical text analytics. Yet evaluations typically focus on a narrow aspect -- such as similarity or utility comparisons -- even though these aspects are complementary and best viewed in parallel. In this study, we aim to conduct a systematic evaluation of LLM-generated clinical text, which includes intrinsic, extrinsic, and factuality evaluations of synthetic clinical notes rephrased from MIMIC databases at million-note scale. Our analysis demonstrates that synthetic notes preserve core clinical information and predictive utility for coarse-grained tasks despite substantial linguistic changes, but lose fine-grained details for task like ICD coding. We show this loss of detail can be substantially mitigated by rephrasing notes by chunks rather than by the whole note, but at the cost of reduced factual precision under incomplete context. Through fact-checking and error analysis, we further find that synthesis errors are dominated by misinterpretation of clinical context, alongside temporal confusion, measurement errors, and fabricated claims. Finally, we show that the synthetic notes -- despite their task-agnostic nature -- can effectively augment task-specific training for rare ICD codes. 

---
# OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization 

**Authors**: Zhongzhu Zhou, Donglin Zhuang, Jisen Li, Ziyan Chen, Shuaiwen Leon Song, Ben Athiwaratkun, Xiaoxia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17757)  

**Abstract**: INT2 KV-cache quantization is attractive for long-context LLM serving, but it remains difficult to make both accurate and deployable. Simple rotations such as Hadamard transforms reduce outliers, but still degrade at INT2 because they are not aligned with downstream attention. We propose OSCAR, an Ultra-low-bit KV Cache quantization method that estimates attention-aware covariance structures offline and uses them to derive fixed rotations and clipping thresholds for quantization. In this way, it aligns KV quantization with the covariance structures that attention actually consumes. More importantly, we not only provide theoretical justification but also develop a fully deployable OSCAR system with a custom INT2 attention kernel that remains compatible with paged KV-cache serving and fused kernel pipelines, enabling seamless integration into modern LLM serving frameworks such as SGLang and vLLM.
We evaluate our methods on recent reasoning models with reasoning traces of up to 32k tokens across 5 tasks. On Qwen3-4B-Thinking-2507 and Qwen3-8B, OSCAR reduces the BF16 accuracy gap to 3.78 and 1.42 points, respectively, while naive rotation INT2 collapses to nearly zero. We further scale OSCAR to Qwen3-32B and GLM-4.7 (358B params), where it remains effectively on par with BF16. On long context - RULER-NIAH up to 128K, OSCAR remains robust on both Qwen3 models, while naive rotation INT2 collapses. System-wise, OSCAR reduces KV-cache memory by approximately 8x, improves throughput by up to 7x at large batch sizes under the same memory budget, and accelerates batch-size-1 decoding by up to 3x over BF16 due to reduced memory bandwidth overhead. 

---
# Bridging the Version Gap: Multi-version Training Improves ICD Code Prediction, Especially for Rare Codes 

**Authors**: Jinghui Liu, Anthony Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.17755)  

**Abstract**: Clinical coding maps clinical documentation to standardized medical codes, an essential yet time-consuming administrative task that could benefit from automation. Current models on ICD coding are typically optimized for codes from a specific ICD version. However, in reality, ICD systems evolve continuously, and different versions are adopted across time periods and regions. Moreover, ICD coding suffers from the long-tail problem, and rare code performance can be a bottleneck for developing implementable models. We examine whether it is viable to train version-independent models by combining data annotated in different ICD versions, which may help address these challenges. We add ICD-9 data to the training of a modified label-wise attention model for ICD-10 prediction, and find that despite the version mismatch, adding ICD-9 yields a 27% increase in micro F1 for 18K rare ICD codes compared to training on ICD-10 alone. On 8K frequent ICD-10 codes, the multi-version training also substantially improves macro metrics, with far fewer model parameters. 

---
# L-Drive: Beyond a Single Mapping-Latent Context Drives Time Series Forecasting 

**Authors**: Fan Zhang, Shijun Chen, Hua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17730)  

**Abstract**: Mainstream methods for multivariate time-series forecasting largely follow the Direct-Mapping paradigm. They learn a unified mapping from history to the future in the observation space to fit value-level dependencies. However, real-world systems often undergo distribution shifts and regime changes. In such cases, a unified mapping can exhibit response lag around turning points, causing error accumulation within the switching window and reducing forecasting reliability. To address this issue, we propose L-Drive, a change-aware forecasting framework. L-Drive introduces a Latent-Context, to explicitly characterize high-level dynamics evolving over time, and uses gating to modulate increment representations. This provides more timely change cues and improves adaptation to changing segments. In addition, it incorporates patch-shared relative positional basis functions to strengthen intra-segment structural modeling and reduce overfitting caused by absolute-position memorization. Extensive experiments validate the effectiveness of L-Drive and show a better overall trade-off between forecasting accuracy and computational efficiency. 

---
# Domain Incremental Learning for Pandemic-Resilient Chest X-Ray Analysis 

**Authors**: Danu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.17729)  

**Abstract**: Deep learning models achieved high accuracy in pneumonia detection from chest X-rays. However, their generalization across clinical domains remains limited due to variations in imaging devices, acquisition protocols, and institutional conditions. This study introduces a replay-based domain-incremental continual learning designed to enable continual adaptation to cross-domain variations without catastrophic forgetting. The proposed method incorporates a class-aware balanced replay to maintain balanced class representation within a constrained memory and a class-aware loss to dynamically reweight class imbalance during training. Experiments conducted on a domain-shifted PneumoniaMNIST dataset consisting of five simulated domains demonstrate that the proposed method achieves an average accuracy of 88.66%, outperforming Experience Replay, Fine-Tuning, and Joint Training baselines. These findings highlight the efficacy of the proposed approach in achieving robust and consistent pneumonia detection across clinical environment variations. 

---
# Fine-tuning Pocket-Aware Diffusion Models via Denoising Policy Optimization 

**Authors**: Yuan Xue, Daniel Kudenko, Megha Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2605.17693)  

**Abstract**: Structure-based drug design has been accelerated by pocket-aware 3D generative models, yet most methods primarily fit the training distribution and may fall short of satisfying multiple properties required in real-world therapeutic drug discovery. Recently, increasing attention has focused on structure-based molecule optimization (SBMO), which targets fine-grained control over multiple specified molecular properties. In this paper, we present DEPPA, a novel SBMO approach building upon Denoising Diffusion Policy Optimization for fine-tuning a pre-trained pocket-aware diffusion model via reinforcement learning. DEPPA enables optimization over multiple properties, including binding affinity, drug-likeness, synthesizability and diversity. We formulate the reverse denoising process of the pretrained pocket-aware diffusion model as a multi-step Markov Decision Process, where the desired properties that serve as reward signals are evaluated on the final generated ligand molecules. DEPPA incorporates a coarse denoising scheduler during the RL fine-tuning to achieve efficient and effective molecule optimization. Experimental results on the CrossDocked2020 benchmark demonstrate that DEPPA outperforms baselines in binding affinity (Vina Score -8.5 kcal/mol), drug-likeness and diversity while exhibiting competitive performance in synthesizability. The source code is available at this https URL . 

---
# Validate Your Authority: Benchmarking LLMs on Multi-Label Precedent Treatment Classification 

**Authors**: M. Mikail Demir, M. Abdullah Canbaz  

**Link**: [PDF](https://arxiv.org/pdf/2605.17691)  

**Abstract**: Automating the classification of negative treatment in legal precedent is a critical yet nuanced NLP task where misclassification carries significant risk. To address the shortcomings of standard accuracy, this paper introduces a more robust evaluation framework. We benchmark modern Large Language Models on a new, expert-annotated dataset of 239 real-world legal citations and propose a novel Average Severity Error metric to better measure the practical impact of classification errors. Our experiments reveal a performance split. Google's Gemini 2.5 Flash achieved the highest accuracy on a high-level classification task (79.1%), while OpenAI's GPT-5-mini was the top performer on the more complex fine-grained schema (67.7%). This work establishes a crucial baseline, provides a new context-rich dataset, and introduces an evaluation metric tailored to the demands of this complex legal reasoning task. 

---
# Attention-Guided Fusion of 1D and 2D CNNs for Robust ECG-Based Biometric Recognition 

**Authors**: Arioua, Islameddine, Benzaoui, Amir, Zeroual, Abdelhafid, Houam, Lotfi  

**Link**: [PDF](https://arxiv.org/pdf/2605.17685)  

**Abstract**: Electrocardiogram (ECG)-based biometric recognition has emerged as a promising solution for secure authentication and liveness detection. However, most existing methods rely on unimodal deep learning architectures that independently process either one-dimensional (1D) temporal signals or two-dimensional (2D) time-frequency representations, limiting robustness and generalization. To address this issue, this paper proposes a hybrid framework integrating 1D and 2D convolutional neural networks (CNNs) within a unified end-to-end architecture. The 1D branch extracts temporal and morphological features from raw ECG signals, while the 2D branch captures discriminative spectral information from time-frequency representations. An attention-guided fusion mechanism dynamically weights both modalities according to input characteristics, overcoming the limitations of conventional static fusion strategies.
The framework was evaluated on three benchmark datasets (ECG-ID, MIT-BIH, and PTB), including healthy subjects and patients with cardiac pathologies, achieving identification accuracies of 99.56%, 100.00%, and 99.89%, respectively. To assess long-term biometric permanence, experiments were also conducted on the multi-session Heartprint dataset spanning ten years. The proposed approach achieved same-session accuracies of 98.54% (S1), 99.09% (S2), 94.93% (S3R), and 96.08% (S3L), while cross-session evaluations reached 56.33% (S1-S2) and 53.27% (S2-S3R), demonstrating the ability to capture stable biometric signatures over time. The optimal configuration combines InceptionTime for 1D processing, ResNet-34 for 2D analysis, and attention-based fusion. Ablation studies confirm that the proposed attention mechanism consistently outperforms conventional fusion approaches. Overall, the proposed framework provides a robust, scalable, and high-performance solution for ECG biometric recognition. 

---
# PULSE: Agentic Investigation with Passive Sensing for Proactive Intervention in Cancer Survivorship 

**Authors**: Zhiyuan Wang, Ariful Islam, Indrajeet Ghosh, Xinyu Chen, Katharine E. Daniel, Subigya Nepal, Philip Chow, Laura E. Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2605.17679)  

**Abstract**: Cancer survivors face elevated rates of depression, anxiety, and general emotional distress, yet the precise moments they most need support are often the moments when self-report is sparse, a phenomenon we term the diary paradox. Passive smartphone sensing offers a continuous, unobtrusive alternative, but prior sensing-based affect prediction has been limited by an accuracy ceiling, suggesting a bottleneck not only in available data, but in how behavioral signals are interpreted. We present PULSE, a system that shifts from fixed feature pipelines to agentic sensing investigation: LLM agents equipped with eight purpose-built tools autonomously query smartphone sensing data, compare current behavior against personalized baselines, and calibrate inferences through retrieval-augmented population-level comparisons. Rather than receiving pre-formatted feature summaries, agents decide which modalities to inspect, how far back to look, and how deeply to investigate, mirroring hypothesis-driven clinical reasoning. We evaluate PULSE through a 2*2 factorial design crossing reasoning architecture (structured vs. agentic) with data modality (sensing-only vs. with diary) on 50 cancer survivors from a longitudinal study of cancer survivors. Agentic reasoning is the primary driver of performance: agentic multimodal agent achieves balanced accuracy of 0.743 for emotion regulation desire with diary and sensing data, while agentic agents predict intervention availability at 0.713 with passive sensing data only. These results suggest that agentic investigation may be a cornerstone for unlocking the clinical value of passive sensing, advancing the feasibility of proactive just-in-time mental health support. 

---
# PEIRA: Learning Predictive Encoders through Inter-View Regressor Alignment 

**Authors**: Michael Arbel, Basile Terver, Jean Ponce  

**Link**: [PDF](https://arxiv.org/pdf/2605.17671)  

**Abstract**: Non-contrastive self-supervised learning (SSL) is an effective framework for predictive representation learning, but popular (and in practice effective) methods such as SimSiam, BYOL, I-JEPA or DINO, which rely on a form of self-distillation to train a teacher-student network, remain poorly understood as they typically do not minimize a well-defined objective. We analyze the dynamics of a variant of the Joint Embedding Predictive Architecture (JEPA) using a regularized linear regressor to predict the learned representations of two views of the data from one another, and fully characterize its stability: non-collapsed stable equilibria align with leading nonlinear canonical correlation subspaces, while collapsed equilibria may also be stable attractors. Motivated by this result, we introduce PEIRA, a non-contrastive SSL method with an explicit objective defined through the trace of the optimal linear regressor. We show that its only stable equilibria are nontrivial global minimizers and recover the same canonical correlation subspaces, with regularization selecting the effective dimension. Experiments on ImageNet-1K and CIFAR-10 show PEIRA is competitive with VICReg and LeJEPA baselines, and qualitative empirical results support the theory. 

---
# Training Infinitely Deep and Wide Transformers 

**Authors**: Raphaël Barboni, Maarten V. de Hoop, Takashi Furuya, Gabriel Peyré  

**Link**: [PDF](https://arxiv.org/pdf/2605.17660)  

**Abstract**: Transformers have become the dominant architecture in modern machine learning, yet the theoretical understanding of their training dynamics remains limited. This paper develops a rigorous mathematical framework for analyzing gradient-based training of transformers in the mean-field regime, where both the depth (number of layers) and width (number of attention heads) tend to infinity. While ResNet training can be understood as controlling a neural ODE, transformer training corresponds to controlling a neural PDE, due to the coupling of multiple token distributions through the attention mechanism. Our mean-field model features two types of measure representations: token distributions evolving through layers and attention parameters at each layer. We establish well-posedness of the forward pass through infinitely deep transformers, characterizing token evolution via flow maps that satisfy ODEs in function spaces. Using adjoint sensitivity analysis, we derive an explicit formula for the conditional Wasserstein gradient of the training risk, involving adjoint variables governed by backward ODEs. We prove the existence and uniqueness of gradient flow curves in the conditional Wasserstein metric space, establishing a rigorous foundation for gradient-based transformer training. A key technical contribution is providing necessary and sufficient conditions for injectivity of the Neural Tangent Kernel (NTK) for attention mechanisms: we show that NTK injectivity is equivalent to linear independence of log-sum-exp functions modulo affine functions, a condition satisfied by diverse token distributions, including discrete distributions, uniform distributions, and Gaussian mixtures. Under this NTK injectivity assumption, we prove that gradient flow converges to global minima when the initial loss is sufficiently small, eliminating spurious local minima from the optimization landscape. 

---
# LLMForge: Multi-Backend Hardware-Aware Neural Architecture Search with Infinite-Head Attention for Edge Language Models 

**Authors**: Xinting Jiang, Junyi Luo, Ruichen Qi, Kauna Lei, Ben Laurie, Gregory Kielian, Mehdi Saligane  

**Link**: [PDF](https://arxiv.org/pdf/2605.17653)  

**Abstract**: Sub-billion-parameter Transformer language models are increasingly deployed on edge devices, where the privacy, latency, and operating-cost advantages of on-device inference are constrained by tight memory-bandwidth, energy, and thermal budgets that make architectural choice and accelerator-specific cost central to efficient inference. We present LLMForge, a hardware-aware neural architecture search (NAS) framework whose three composable contributions together make edge-LM architecture search hardware-conditioned, since different substrates impose different hardware cost bottlenecks. Infinite-Head Attention (IHA) decouples the number of query heads, KV groups, and per-head query/key and value dimensions, expanding the feasible per-layer attention configuration space by approximately 400x over grouped-query attention within our search-space ranges. Forge-Former, an encoder-based surrogate for ranking architectural candidates, outperforms MLP and random-forest baselines. Forge-DSE, an NSGA-II-based design-space-exploration engine, pairs Forge-Former with a multi-backend hardware cost model spanning GPUs, systolic accelerators, and ring-dataflow edge accelerators. Across four different hardware substrates, the searches converge to visibly different architectures whose shapes track each substrate's cost bottleneck. On the multi-chip ring substrate, our co-search returns three 300M-scale deployment-aware variants on the Pareto front. Each is re-trained on FineWeb-Edu-10BT under matched recipe against SmolLM2-360M and Qwen-0.5B architecture baselines. The accurate variant has the lowest validation loss 2.798 and competitive benchmark performance with fewer parameters, the energy-optimized variant lowers energy per token by 40%, and the latency-optimized variant lowers TTFT and TPOT by 43%. 

---
# SparseSAM: Structured Sparsification of Activations in Segment Anything Models 

**Authors**: Hoai-Chau Tran, Chi H. Nguyen, Duy M. H. Nguyen, Mathias Niepert, Fan Lai, Khoa D. Doan  

**Link**: [PDF](https://arxiv.org/pdf/2605.17633)  

**Abstract**: The Segment Anything Model (SAM) achieves strong open-vocabulary segmentation, but its ViT-based image encoders dominate inference latency and memory. Existing activation compression methods, such as token merging, reduce the token length to process, yet introduce non-trivial runtime overhead and encounter catastrophic quality drop under high compression. Other methods applying Sparse Attention focus on attention alone, leaving the MLP fully dense and capping achievable speedup. We propose SparseSAM, a (i) training-free structured sparsification framework that jointly accelerates attention and MLP layers while preserving token identity. SparseSAM introduces (ii) Stripe-Sort Attention, which uses a deterministic Z-order permutation to transform dense attention into static hardware-friendly sparse patterns, eliminating dynamic masking overhead. SparseSAM further introduces a (iii) Residual-Consistency MLP that routes only informative tokens through the MLP while propagating remaining tokens through the residual pathway. Across four segmentation benchmarks, SparseSAM loses only 0.004 mIoU at a 0.4 density and 0.021 mIoU at 0.3, a 2.10x reduction in accuracy loss versus token merging advances, while achieving 2x faster inference and 2.8x memory reduction. 

---
# Multi-task learning on partially labeled datasets via invariant/equivariant semi-supervised learning 

**Authors**: Miquel Martí i Rabadán, Alessandro Pieropan, Hossein Azizpour, Atsuto Maki  

**Link**: [PDF](https://arxiv.org/pdf/2605.17624)  

**Abstract**: We investigate the potential of invariant and equivariant semi-supervised learning for addressing the challenges of training multi-task models on partially labeled datasets with differently structured output tasks. Specifically, we use the popular FixMatch method for invariant semi-supervised learning and its equivariant extension Dense FixMatch. We evaluate their performance on the Cityscapes and BDD100K datasets in the context of the prevalent object detection and semantic segmentation tasks in computer vision. We consider varying sizes of the subsets annotated for each task and different overlaps among them. Our results for both invariant and equivariant semi-supervised learning outperform supervised baselines in most situations, with the most significant improvements observed when fewer labeled samples are available for a task and generally better results for the latter approach. Our study suggests that invariant/equivariant learning is a promising general direction for multi-task learning from limited labeled data. 

---
# SynVA: A Modular Toolkit for Vessel Generation and Aneurysm Editing 

**Authors**: Marten J. Finck, Niklas C. Koser, Sarker M. Mahfuz, Tameem Jahangir, Jon E. Wilhelm, Daniel Behme, Naomi Larsen, Wojtek Palubicki, Sylvia Saalfeld, Sören Pirk  

**Link**: [PDF](https://arxiv.org/pdf/2605.17620)  

**Abstract**: Intracranial aneurysms (IAs), characterized by unpredictable growth and risk of rupture, are a major cause of stroke and can lead to life-threatening hemorrhages with high mortality and long-term disability. With aging populations, the incidence and overall burden of cerebrovascular diseases are expected to increase, highlighting the need for scalable approaches to analyze complex medical data and improve population-level understanding of these conditions. While digital twins and deep learning offer promising avenues for improving diagnosis, prognosis, and treatment, their effectiveness is limited by the scarcity of large-scale, high-quality medical data and corresponding labels. We present Synthetic VAsculature (SynVA), a modular toolkit for vascular mesh generation and anatomically consistent aneurysm synthesis. SynVA combines novel flow-matching-based methods for generating healthy vessel meshes with learning-based approaches for anatomy-conditioned aneurysm mesh generation - aneurysms are computed from pre-existing vascular geometries rather than being generated in isolation. In addition, we introduce the SynVA procedural model for vascular and aneurysm synthesis based solely on physiological principles and statistical priors, which enables the generation of large-scale datasets (e.g., for the training of mesh-based generative models). To this end, we release a dataset of 50,000 fully labeled mesh samples for a variety of downstream vision tasks, such as semantic segmentation. Extensive quantitative and qualitative evaluations demonstrate that SynVA generates realistic vessel geometries and anatomically plausible aneurysms. Specifically, our experiments indicate that some methods produce aneurysm shapes more aligned with expert human perception while others perform better on quantitative similarity metrics with reconstructions of real aneurysms. 

---
# Bayesian-Monte Carlo Schedule Updating for Construction Digital Twins: A Probabilistic Framework for Dynamic Project Forecasting 

**Authors**: Atena Khoshkonesh, Mohsen Mohammadagha, Vinayak Kaushal, Navid Ebrahimi  

**Link**: [PDF](https://arxiv.org/pdf/2605.17608)  

**Abstract**: Construction projects frequently experience schedule delays and forecasting uncertainty due to variability in labor productivity, material availability, weather conditions, and project coordination. Conventional deterministic scheduling methods such as the Critical Path Method (CPM) assume fixed activity durations and therefore cannot adequately represent dynamic project uncertainty.
This study presents a Bayesian-Monte Carlo probabilistic schedule updating framework for construction digital twin environments. The proposed methodology integrates stochastic activity-duration modeling, Bayesian recursive updating, Monte Carlo simulation, and uncertainty propagation within a unified computational framework for adaptive schedule forecasting.
Activity durations are modeled using lognormal probability distributions and continuously updated through Bayesian inference as new project observations become available. Monte Carlo simulation is then used to propagate updated uncertainty throughout project networks and generate probabilistic completion-time forecasts, delay-risk estimates, and activity criticality measures.
Simulation experiments using PSPLIB benchmark project networks demonstrate that the proposed framework improves forecasting accuracy and uncertainty representation compared with deterministic CPM and static probabilistic scheduling approaches. The framework further supports adaptive project forecasting through integration of BIM reports, drone observations, IoT telemetry, productivity logs, and site monitoring data. 

---
# UniAlign: A Model-Agnostic Framework for Robust Network Traffic Classification under Distribution Shifts 

**Authors**: Tongze Wang, Xiaohui Xie, Wenduo Wang, Chuyi Wang, Yong Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.17575)  

**Abstract**: Network traffic classification (NTC) models often suffer severe performance degradation when deployed in real-world environments due to distribution shifts caused by changing network conditions. Existing robustness-enhancing approaches are commonly coupled to specific model architectures or data settings, fail to generalize to state-of-the-art raw-byte-based NTC models, or incur significant training overhead. In this paper, we propose UniAlign, a novel model-agnostic framework that improves the robustness of deep learning-based NTC models under distribution shifts. UniAlign combines \emph{domain alignment fine-tuning}, which encourages the learning of domain-invariant traffic representations across heterogeneous network conditions, with \emph{stable model ensembling}, which enhances inference robustness by aggregating checkpoints within a flat loss region. The framework can be seamlessly integrated into existing supervised NTC models without requiring specific feature modalities or introducing non-constant additional training costs. We evaluate UniAlign on three public datasets covering diverse distribution shifts, including encryption schemes, data collection devices, and attack behaviors. Experimental results on two representative NTC models demonstrate that, compared with standard training, UniAlign improves average classification accuracy by 2.51\% and average F1 score by 2.71\%, outperforming the strongest baseline by 1.45\% in accuracy and 1.69\% in F1 score, while requiring only 12.4\%--53.9\% of the training time of all NTC-specific baselines. 

---
# Beyond Accuracy: Robustness, Interpretability and Expressiveness of EEG Foundation Models 

**Authors**: Urban Širca, Maryam Alimardani, Stefanos Zafeiriou, Konstantinos Barmpas  

**Link**: [PDF](https://arxiv.org/pdf/2605.17562)  

**Abstract**: EEG foundation models (EEG-FMs) have been evaluated predominantly on clean, in-distribution accuracy, leaving their robustness, interpretability and representational quality largely unexamined. This study addresses these gaps by benchmarking six EEG-FMs against a baseline deep learning model across eight datasets. Beyond clean accuracy, we conduct three layers of analysis: (i) Robustness: we apply test-time perturbations including additive noise, random and region-based channel dropout and region-specific noise injection. Our analyses show that no single model dominates all failure modes. The most noise-robust model is among the most fragile under channel dropout and much of the dropout fragility disappears when channels are removed rather than zero-padded. (ii) Interpretability: we present the first application of Attention-Aware Layer-Wise Relevance Propagation (AttnLRP) to EEG-FMs and show that models broadly concentrate relevance on task-appropriate brain regions consistent with known neurophysiology. However, attribution maps remain spatially stable under perturbation while predictions degrade, suggesting that the models attend to the correct brain regions but decode corrupted content. (iii) Expressiveness: With block-wise probing we show that late blocks are repurposed during fine-tuning, while early blocks already hold task-related information. Furthermore, we demonstrate that the poor head-only performance previously attributed to low-quality pre-trained representations is largely explained by pooling and that EEG-FMs possess sufficient representational capacity when their token-level embeddings are preserved. Together, these findings provide the first systematic assessment of robustness, interpretability and expressiveness for EEG-FMs and highlight critical considerations for their development. 

---
# Automated Root-Cause Subclassification and No-Code Fix Generation for Invalid Bug Reports 

**Authors**: Mahmut Furkan Gon, Emre Dinc, Tevfik Emre Sungur, Eray Tuzun  

**Link**: [PDF](https://arxiv.org/pdf/2605.17561)  

**Abstract**: Issues faced when using software are reported in the form of bug reports. However, many bug reports are invalid, meaning they do not require code changes, and are resolved with a no-code fix. Manually determining the root cause of the invalid bug reports and providing actionable resolutions by the customer support causes a serious waste of resources. Our goal is to introduce a standardized taxonomy for root-cause oriented invalid bug report subclassification, and perform experiments to test the accuracy of various approaches on invalid subclassification and no-code fix generation. We study how different configurations perform on a gold-standard benchmark we have created. Using a manually curated benchmark for higher quality analysis, we experimented with vanilla LLMs, Retrieval Augmented Generation, and agentic web search to identify invalid subclasses and generate no-code fixes. We evaluated the results against manually labeled ground truth data that includes the invalid subclass and no-code fixes from the original bug reports. We measured subclass detection performance with weighted F1-Score, and assessed no-code fix suggestions using BERTScore and Judge LLM success rates. For subclassification, retrieval augmented generation achieves the highest overall performance with 0.66 weighted F1, slightly outperforming vanilla LLMs at 0.65 and agentic web search at 0.64. At the subclass level, performance peaks at 0.85 F1 for Non-reproducibility and 0.79 for Feature Request and Question, while Wrong Version remains the most challenging with scores between 0.00 and 0.29. For no-code fix generation, agentic web search achieves the highest overall Judge LLM success rate at 68.9%, compared to 64.4% for RAG applications and 64.9% for vanilla LLMs, with subclass-level peaks of 87.4% for Working as Designed and 72.2% for Question. 

---
# Controlling False Discovery in Arbitrarily Structured Hypothesis Spaces via Reproducing Kernels 

**Authors**: Binyamin Perets, Shie Mannor  

**Link**: [PDF](https://arxiv.org/pdf/2605.17559)  

**Abstract**: Large-scale hypothesis testing is central to modern science, where controlling the False Discovery Rate (FDR) has become the standard approach to managing false positives across many simultaneous tests. Hypotheses rarely exist in isolation; they often exhibit structure through proximity, connectivity, or hierarchy. This structure represents both a challenge and an opportunity: while classical methods treat these dependencies as obstacles requiring conservative correction, leveraging them can substantially increase discovery power. Here, we reframe structured FDR control as a regularized learning problem. By optimizing within a suitable Reproducing Kernel Hilbert Space (RKHS), we introduce a framework that unifies continuous domains, graphs, and hierarchies under a single algorithm through kernel choice alone. This formulation enables smooth solutions in place of the piecewise-constant fits of prior methods, principled likelihood-based hyperparameter selection rather than heuristic tuning, and inference at unobserved locations which in turn supports sample-efficient experimental design. Building on this estimator, we provide two decision rules which we prove to control the FDR. We validate our method on two sources: spatial locations derived from high-dimensional real-world datasets, and a differential gene expression task utilizing protein-protein interaction graphs. 

---
# Visual Sculpting: Visually-Aligned Planning Representations for Long-Horizon Robot Clay Sculpting 

**Authors**: Peter Schaldenbrand, Jean Oh  

**Link**: [PDF](https://arxiv.org/pdf/2605.17556)  

**Abstract**: Clay sculpting is a nuanced, artistic task involving dexterous manipulation with long-horizon planning to achieve high-level goals. As a robotics problem, we formulate clay sculpting as a shape-to-shape matching challenge. Prior deformable object manipulation work either requires retraining a policy per goal or relies on dynamics models which represent state as sparse point clouds which do not capture important clay features, such as textures, well. We present a method for modeling the dynamics of deformable materials and planning for robotic sculpting in a representation that is visually-aligned, capturing lighting and texture features. With three different deformable materials and various end-effectors, we demonstrate that our dynamics model is comparable in performance to the state-of-the-art with the added benefit of being compatible with visual planning. Our actions are represented as parametrized pushes into clay with a single end-effector, which proved to be suitable for long-horizon (>100 actions) clay relief sculptures. Lastly, we show the benefits of planning in a visually-aligned representation, but also provide analysis providing evidence as to why this representation is challenging to plan in compared to 3D representations. 

---
# Rethinking Code Review in the Age of AI: A Vision for Agentic Code Review 

**Authors**: Hüseyin Özgür Kamalı, Erdem Tuna, Vahid Haratian, Eray Tüzün  

**Link**: [PDF](https://arxiv.org/pdf/2605.17548)  

**Abstract**: Code review has evolved for decades, from informal peer checking to today's pull request (PR) workflows, yet it remains a largely manual, uneven, and cognitively demanding process. The rise of Artificial Intelligence (AI) coding assistants has intensified this challenge: while these tools increase code production velocity, they also expand the volume of code requiring review, turning code review into a growing bottleneck. Current AI support remains fragmented, with tools focusing on isolated tasks such as reviewer recommendation, PR description generation, or comment suggestion rather than the end-to-end PR review workflow. In this paper, we review the historical evolution of code review practices and examine the shift driven by large language models (LLMs) and agentic AI systems. We then present a vision for an AI-powered code review workflow combining specialized agents with human-controlled quality gates. Our framework spans five stages: PR Creation, PR Augmentation, Reviewer Selection, AI-Assisted Code Review, and PR Retrospective, with humans retained at key decision points to preserve judgment, accountability, and team-level understanding. We identify major open challenges for responsible adoption, including reliability, bias, privacy, automation bias, transparency, and evaluation, and offer a research agenda for more effective human-AI collaboration in software engineering. 

---
# Few-Shot Network Intrusion Detection Using Online Triplet Mining 

**Authors**: Jack Wilkie, Hanan Hindy, Christos Tachtatzis, Miroslav Bures, Robert Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2605.17530)  

**Abstract**: Network intrusion detection systems play a vital role in protecting networks by detecting malicious network traffic which can then be investigated by a cybersecurity operations centre. State-of-the-art approaches utilise supervised machine learning methods to train a classification model to recognise known cyberattacks; however, these models require a large labelled dataset to train and show poor performance when trained on smaller datasets. In an attempt to address this shortcoming, anomaly detection models learn the distribution of benign traffic and flag non-conforming traffic as malicious. While these methods do not require malicious examples to train, they suffer from high false-positive rates rendering them impractical. As a result, networks may be particularly vulnerable when there are insufficient labelled instances of a specific attack class to train an effective classifier. This often occurs in newly established networks or when previously unseen types of attacks emerge. To address this challenge, this work proposes the use of a triplet network, utilising online triplet mining and a KNN classifier, which is able to perform few-shot classification, enabling effective intrusion detection after being trained on a limited number of malicious examples. Various online triplet mining algorithms were explored and model design choices, such as the inference algorithm and optimised distance metrics, were compared and evaluated through a series of ablation studies. The final model was compared against other state-of-the-art approaches in few-shot binary and multiclass classification, where the proposed approach was found to be competitive with existing methods when trained on as little as 10 malicious samples of each class. 

---
# CasualSynth: Generating Structurally Sound Synthetic Data 

**Authors**: Zehua Cheng, Wei Dai, Jiahao Sun, Thomas Lukasiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2605.17528)  

**Abstract**: Large Language Models (LLMs) generate realistic synthetic data but offer no guarantee that their outputs respect the causal mechanisms governing the target domain. We introduce CausalSynth, a framework that decouples causal structure generation from semantic realization, yielding synthetic data that is both causally valid and linguistically rich. The framework operates in three phases. First, a Structural Causal Model (SCM) - a tuple of structural equations defined over a directed acyclic graph (DAG) generates causal skeletons, i.e., variable assignments that satisfy the Global Markov Property of the governing DAG, via ancestral sampling. Second, an LLM acts as a constrained \emph{realizer}, a conditional translator that maps each skeleton to a high-dimensional observation such as a clinical note or a transaction log. Third, an Iterative Consistency Verification module detects structural violations through deterministic extraction and feeds targeted corrections back to the LLM, forming a closed-loop refinement process. We identify the Semantic Backdoor problem the systematic tendency of LLMs to override imposed causal facts with pre-training priors -- and prove that our iterative mechanism reduces the resulting selection bias relative to standard rejection sampling. On three causal benchmarks (ASIA, ALARM, and MIMIC-Struct), CausalSynth preserved conditional independencies with false-positive rates near the nominal $\alpha=0.05$ level and achieved realizability rates above 96% with 70B-parameter LLM backbones. The framework additionally supports principled interventional and counterfactual generation through noise retention and graph mutilation. 

---
# SaaSBench: Exploring the Boundaries of Coding Agents in Long-Horizon Enterprise SaaS Engineering 

**Authors**: Qingnan Ren, Shun Zou, Shiting Huang, Ziao Zhang, Kou Shi, Zhen Fang, Yiming Zhao, Yu Zeng, Qisheng Su, Lin Chen, Yong Wang, Zehui Chen, Xiangxiang Chu, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.17526)  

**Abstract**: As autonomous coding agents become capable of handling increasingly long-horizon tasks, they have gradually demonstrated the potential to complete end-to-end software development. Although existing benchmarks have recently evolved from localized code editing to from-scratch project generation, they remain confined to structurally simplified, single-stack applications. Consequently, they fail to capture the heterogeneous environments, full-stack orchestration, and system-level complexity of real enterprise Software as a Service (SaaS) systems, leaving a critical gap in assessing agents under realistic engineering constraints. To fill this gap, we introduce SaaSBench, the first benchmark designed to explore the boundaries of AI agents in enterprise SaaS engineering. Spanning 30 complex tasks across 6 SaaS domains with 5,370 validation nodes, it incorporates 8 programming languages, 6 databases, and 13 frameworks to meticulously mirror real-world software heterogeneity. Furthermore, we design a dependency-aware hybrid evaluation paradigm tailored for complex systems with long horizons and multi-component coupling, enabling fine-grained, reproducible assessment. Crucially, our extensive experiments reveal a striking insight: the primary bottleneck for state-of-the-art agents is not generating isolated code logic, but successfully configuring and integrating a multi-component system. Over 95\% of task failures occur before agents even reach deep business logic, with models often falling victim to overconfidence and prematurely halting during foundational system setup, or getting trapped in ineffective debugging loops. We hope SaaSBench serves as a practical and challenging testbed to drive the evolution of reliable, system-level coding agents. The code is available at \url{this https URL}. 

---
# BESplit: Bias-Compensated Split Federated Learning with Evidential Aggregation 

**Authors**: Yuhan Xie, Chen Lyu, Jingrong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17508)  

**Abstract**: Split Federated Learning (SFL) enables privacy-preserving collaborative training by partitioning models between clients and a server. However, under non-IID data distributions, SFL often suffers from biased optimization and unstable convergence, while existing solutions largely adapt techniques from conventional federated learning. In this work, we observe that the split architecture of SFL inherently alters how client information is represented and coordinated, opening opportunities for bias compensation beyond parameter-level aggregation. Based on this insight, we propose BESplit, an architecture-aware framework that exploits the intrinsic structure of SFL to mitigate non-IID effects. First, to prevent biased local data from dominating global updates, we introduce Evidential Aggregation (EA) to perform fine-grained reweighting of client contributions based on evidential uncertainty. Second, to further reduce distributional skew, we develop Bias-Compensated Collaboration (BCC) to align split-layer representations by pairing complementary clients. Finally, Dual-Teacher Distillation (DTD) is incorporated to synchronize knowledge between decoupled client and server models, enabling independent local inference. Extensive experiments on five benchmark datasets demonstrate that BESplit consistently outperforms state-of-the-art methods in accuracy, convergence stability, and computational efficiency under diverse non-IID settings. 

---
# A Distributional View for Visual Mechanistic Interpretability: KL-Minimal Soft-Constraint Principle 

**Authors**: Guancheng Zhou, Yisi Luo, Zhengfu He, Zhenyu Jin, Xuyang Ge, Wentao Shu, Deyu Meng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17504)  

**Abstract**: Most current paradigms in visual mechanistic interpretability (MI) remain confined to interpreting internal units of the vision model via heuristic methods (e.g., top-$K$ activation retrieval or optimization with regularization). In this work, we establish a theoretical distributional view for visual MI, which models the influence of a feature activation on the natural image distribution, thereby formulating a Kullback-Leibler (KL)-minimal optimization problem to model the MI task. Under this framework, statistical biases are identified within previous MI paradigms, which reveal that they may either be perceptually uninterpretable to humans (i.e., deviate from the natural image distribution), or mechanistically unfaithful to the vision models (i.e., unable to activate model features). To resolve the biases under the distributional view, we propose a model with a KL-minimal soft-constraint principle for visual MI that theoretically balances interpretability and faithfulness. We realize this principle via energy-guided diffusion posterior sampling. Extensive experiments validate the theoretical soundness of the proposed distributional view and demonstrate the practical effectiveness of our paradigm on the DINOv3 vision model. 

---
# Beyond Linear Superposition: Discovering Climate Features in AI Weather Models with KAN-SAE 

**Authors**: Minjong Cheon  

**Link**: [PDF](https://arxiv.org/pdf/2605.17493)  

**Abstract**: Deep learning weather prediction models achieve remarkable predictive skill yet remain largely opaque: we know little about how they represent physical climate phenomena internally. Mechanistic interpretability through Sparse Autoencoders (SAEs) offers a principled route to decomposing these representations, but existing SAEs assume strictly linear feature superposition - a constraint ill-suited for the highly nonlinear atmospheric dynamics encoded in modern transformers. We introduce KAN-SAE, a sparse autoencoder whose encoder replaces the standard ReLU with learnable per-feature B-spline activations drawn from Kolmogorov-Arnold Networks (KANs), allowing each latent dimension to develop its own nonlinear gating profile. Applied to Sonny, KAN-SAE discovers 975 alive features (vs. 566 for a linear baseline, a 72% improvement) with 20% lower inter-feature redundancy and comparable reconstruction fidelity. Without any climate supervision, KAN-SAE identifies an interpretable European heatwave feature spatially concentrated over western Europe, and a western Pacific typhoon tracker confirmed by causal steering experiments. Our results demonstrate that nonlinear activations are essential for mechanistic interpretability of deep learning weather prediction models, recovering climate features that remain invisible to linear baselines. 

---
# An Interpretable Closed-Loop Intelligent Tutoring System for Multimodal Affective Feedback in Asynchronous Presentation Training 

**Authors**: Hung-Yue Suen, Kuo-En Hung  

**Link**: [PDF](https://arxiv.org/pdf/2605.17468)  

**Abstract**: This paper presents an interpretable closed-loop Intelligent Tutoring System (ITS) that supports feedback-guided practice for developing on-camera oral presentation skills at scale. The system operationalizes a seven-dimensional Behaviorally Anchored Rating Scale (BARS) and implements a three-layer interpretable feedback architecture that connects rubric-aligned multimodal scoring, audience-perceived expressive diagnostics, and retrieval-augmented conversational coaching to support deliberate practice. Built on an XGBoost backbone, the ITS maps multimodal inputs (facial, vocal, textual, and oculomotor features) into evidence-based feedback that can be traced back to observable performance cues. Trained on 10,360 Massive Open Online Course (MOOC) video segments, the system achieved rubric-aligned scoring with performance levels comparable to expert ratings (R2 = 0.48-0.61, Spearman's rho = 0.69-0.78, MAE = 0.43-0.57). In a pre-post validation study with 204 adult learners over a 30-day practice window, participants demonstrated significant improvements across all seven BARS dimensions (Cohen's d = 0.39-0.90), with practice frequency showing a strong positive association with posttest performance after controlling for baseline scores and demographics. The results demonstrate how multimodal analytic outputs can be systematically transformed into observable behavioral change through an integrated feedback architecture, advancing explainable and pedagogically grounded ITS design for performance-based competencies. 

---
# Artificial Intelligence can Recognize Whether a Job Applicant is Selling and/or Lying According to Facial Expressions and Head Movements Much More Correctly Than Human Interviewers 

**Authors**: Hung-Yue Suen, Kuo-En Hung, Che-Wei Liu, Yu-Sheng Su, Han-Chih Fan  

**Link**: [PDF](https://arxiv.org/pdf/2605.17461)  

**Abstract**: Whether an interviewee's honest and deceptive responses can be detected by facial expression signals in videos has been debated and requires further research. We developed deep learning models enabled by computer vision to extract temporal patterns of job applicants' facial expressions and head movements to identify self-reported honest and deceptive impression management (IM) tactics from video frames in real asynchronous video interviews. A 12- to 15-minute video was recorded for each of N=121 job applicants as they answered five structured behavioral interview questions. Each applicant completed a survey to self-evaluate their trustworthiness on four IM measures. Additionally, a field experiment was conducted to compare the concurrent validity associated with self-reported IMs between our modeling approach and human interviewers. Human interviewers' performance in predicting these IM measures from another subset of 30 videos was obtained by having N=30 human interviewers evaluate three recordings. Our models explained 91% and 84% of the variance in honest and deceptive IMs, respectively, and showed stronger correlations with self-reported IM scores than human interviewers. 

---
# GCE-MIL: Faithful and Recoverable Evidence for Multiple Instance Learning in Whole-Slide Imaging 

**Authors**: Xiangyu Li, Ran Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.17456)  

**Abstract**: Multiple instance learning (MIL) is the standard approach for whole-slide image (WSI) classification and survival prediction, where attention-based models ag gregate patch features into slide-level predictions. These models treat attention weights as evidence for their predictions, but attention is optimized for classi fication, not for identifying which patches actually support the diagnosis. This conflation leads to three failures: selected patches are insufficient (keeping them alone drops Macro-F1 by 0.078), unnecessary (removing them barely changes the prediction), and unrecoverable (continuous attention scores disagree with discrete patch subsets used at inference). The central premise is that evidence quality should be optimized directly through explicit criteria- Sufficiency, Necessity, and Recov erability (S/N/R)- rather than inherited as a byproduct of classification. GCE-MIL is a backbone-agnostic wrapper implemented through three injection modes and three evidence components: a grounding mechanism that aligns selection with domain-specific concepts, noisy-OR coverage that acts as a differentiable proxy for interventional evidence search, and threshold-plus-repair recovery that converts continuous selectors into discrete subsets through marginal-guided repair. Across 9 backbones and 9 datasets (81 configurations), GCE-MIL improves average Macro-F1 by 0.024 and C-index by 0.014, reduces the continuous-discrete gap by 4-7, and increases complement degradation by 2-4. With optional tile prefiltering after discrete recovery, inference runs up to 5 faster while retaining 0.989 full-bag utility. 

---
# ContraFix: Agentic Vulnerability Repair via Differential Runtime Evidence and Skill Reuse 

**Authors**: Simiao Liu, Fang Liu, Li Zhang, Yang Liu, Yinghao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17450)  

**Abstract**: Large language model (LLM) agents are increasingly used for automated vulnerability repair (AVR), where repository-level reasoning enables them to inspect context and produce source-code patches. However, recent empirical results show that these agents still struggle with real-world vulnerabilities. Their main failure mode is semantic misunderstanding: choosing a repair direction that does not match the root cause. We identify two reasons for this gap. Existing agents usually reason from the failing execution alone. A crash report can pinpoint where the program failed, but it does not reveal which variable or state transition, among many candidates near the fault site, separates the crashing behavior from safe execution. As a result, agents often produce symptom-oriented patches instead of causal fixes. Moreover, evidence collected for one vulnerability is rarely retained, so similar cases in later repositories must be diagnosed again from scratch. We present ContraFix, an agentic AVR framework that couples differential runtime evidence with reusable repair skills. Its Mutator constructs PoC variants that straddle the failure boundary; its Analyzer inserts state probes around the fault region and summarizes divergences between crashing and non-crashing executions into a repair specification; and its Patcher converts the specification into verified source patches. Each successful repair updates a two-track skill base containing repair specifications and mutation strategies, which are retrieved through a three-tier policy for future instances. On SEC-Bench (C/C++, 200 instances) and PatchEval (Go, Python, JavaScript, 225 instances), ContraFix with GPT-5-mini resolves 84.0% and 73.8% of the tasks, respectively, achieving state-of-the-art performance on both benchmarks while costing less than one-third of the strongest comparable baseline. 

---
# Spatial Blindness in Whole-Slide Multiple Instance Learning 

**Authors**: Xiangyu Li, Ran Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.17449)  

**Abstract**: Whole-slide MIL models are often called context-aware once graphs, Transform ers, or state-space modules are placed above patch embeddings. We show that this label can be deceptive. On pathology tasks where tissue architecture is part of the diagnostic signal, several strong MIL baselines retain nearly unchanged slide level AUC after patch coordinates are permuted. Their predictions are accurate, but largely compositional. We refer to this failure mode as spatial blindness. Our explanation is optimization-based: dense appearance statistics are learned early under slide-level supervision, leaving weak gradients for sparse spatial relations. ResTopoMIL addresses the issue by first fitting a permutation-invariant prototype histogram and then freezing it while a lightweight graph branch learns the residual under a coordinate-shuffling constraint. The architecture is simple by design; the intervention is in how the spatial branch is trained. Across 9 public WSI bench marks, ResTopoMIL improves classification and survival prediction with 1.15M parameters, restores sensitivity to coordinate perturbation, and gives stronger lo calization evidence on CAMELYON-16. 

---
# MemRepair: Hierarchical Memory for Agentic Repository-Level Vulnerability Repair 

**Authors**: Simiao Liu, Li Zhang, Fang Liu, Xiaoli Lian, Yang Liu, Yinghao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17444)  

**Abstract**: Modern software ecosystems face a rapidly growing number of disclosed vulnerabilities, increasing the need for automated repair techniques that can operate reliably at repository scale. Although Large Language Model (LLM)-based agents have recently shown promise for automated vulnerability repair (AVR), most existing systems still treat repair as a single generation step over the currently visible code context. As a result, they lack a persistent mechanism for reusing prior fixes or learning from failed validation attempts, which limits their effectiveness on complex, multi-file repair tasks. We present MemRepair, a memory-augmented agentic framework that formulates vulnerability repair as an iterative, experience-driven process. MemRepair combines three complementary memory layers, i.e., History-Fix, Security-Pattern, and Refinement-Trajectory memories, with a dynamic feedback-driven refinement loop. This design allows the agent to retrieve repository-specific repair conventions, apply reusable security defenses, and exploit prior "failure-to-success" trajectories to revise semantically invalid patches based on runtime evidence. We evaluate MemRepair on three representative repository-level vulnerability repair benchmarks: SEC-Bench, PatchEval (Python, Go, JavaScript), and the C++ subset of Multi-SWE-bench. MemRepair achieves state-of-the-art resolution rates of 58.0%, 58.2%, and 30.58%, respectively, outperforming strong general-purpose agents such as OpenHands and SWE-agent, as well as the specialized AVR tool InfCode-C++, while maintaining competitive repair cost. These results show that persistent, hierarchical repair memory can substantially improve the reliability of agentic vulnerability repair across diverse languages and repository settings. 

---
# Beyond Catalogue Counts: the Dataset Visibility Asymmetry in Low-Resource Multilingual NLP 

**Authors**: Zhiyin Tan, Changxu Duan  

**Link**: [PDF](https://arxiv.org/pdf/2605.17442)  

**Abstract**: Multilingual NLP often relies on dataset counts from centralized catalogues to characterize which languages are resource-rich or resource-poor. However, these catalogues record only one layer of dataset visibility: what has been registered or institutionally distributed. They do not necessarily reflect which datasets are created, cited, or reused in the research literature. To examine this gap, we combine a catalogue-based baseline with literature-backed evidence of dataset circulation. We introduce the Resource Density Index (RDI), defined as the number of catalogued datasets per one million speakers, and compute it for the 200 most widely spoken languages in Ethnologue. Among them, 118 languages (59%) have an average RDI of zero across the LRE Map and the Linguistic Data Consortium (LDC), and another 23 fall below 0.1, corresponding to at most one catalogued dataset per ten million speakers. We then apply an LLM-assisted citation-mining pipeline over the Semantic Scholar corpus to these 141 low-visibility languages. After manual validation and consolidation, we identify 609 unique datasets across 53 languages, of which 356 remain openly accessible through working public links. These results reveal a substantial visibility gap: many large-speaker languages appear data-poor in catalogue records yet show clear evidence of dataset activity in the research literature. Our findings suggest that multilingual data scarcity should be understood not only as a production problem, but also as a question of documentation, discoverability, and long-term accessibility. Code and data are publicly available at (this https URL). 

---
# DiagEval: Trajectory-Conditioned Diagnosis for Reliable Software Evaluation with GUI Agents 

**Authors**: Sirui Hong, Zhijie Liu, Tengfei Li, Wei Tao, Yifan Wu, Chenglin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17439)  

**Abstract**: Evaluating LLM-generated interactive software requires execution in addition to static analysis. The key difficulty is that correctness is a graph-level reachable property over latent UI state-transition graphs, whereas a GUI evaluator observes only a single execution trajectory. A failed rollout therefore rules out only one realized path, leaving failure attribution ambiguous between evaluator-side execution error and genuine software defect. We present DiagEval, a trajectory-conditioned diagnostic evaluation protocol for post-failure GUI-agent evaluation of interactive software. Rather than blindly retrying from scratch, DiagEval reuses the failed trajectory to choose targeted diagnostic probes and aggregates their outcomes into an internal attribution signal. The latent-graph view motivates the diagnostic problem; DiagEval does not reconstruct the graph or estimate calibrated posterior probabilities. We evaluate DiagEval on WebDevJudge-Unit and RealDevBench across multiple GUI-agent evaluators and LLM backbones. On false-negative cases, DiagEval recovers 45.6-62.1% of failures that were initially misattributed to software defects, outperforming retry-based baselines with 34.4-160.6% relative gains. On the full evaluation sets, this recovery improves accuracy from 69.9% to 78.3% on WebDevJudge-Unit and from 65.0% to 81.6% on RealDevBench. These results suggest that reliable GUI-agent evaluation requires not only stronger execution, but also active failure diagnosis to disambiguate evaluator-side errors from genuine software defects. Our code is available at this https URL. 

---
# MATE: Solving Contextual Markov Decision Processes with Memory of Accumulated Transition Embeddings 

**Authors**: Himchan Hwang, Hyeokju Jeong, Gene Chung, Seungyeon Kim, Sangwoong Yoon, Frank Chongwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.17431)  

**Abstract**: We propose MATE, a simple yet effective memory architecture for solving Contextual Markov Decision Processes (CMDPs), a family of MDPs parameterized by an unobserved context. In CMDPs, an optimal agent can adapt online by maintaining the posterior belief over contexts. MATE replaces this intractable posterior with a sum-aggregated memory, leveraging the posterior's permutation invariance to retain provably sufficient expressiveness. Compared to prior memory architectures, MATE avoids the growing per-step rollout cost of Transformers and the gradient issues commonly associated with Recurrent Neural Networks (RNNs). Extensive evaluations across diverse benchmarks demonstrate that MATE provides clear computational advantages while achieving performance comparable to standard sequence-model baselines. 

---
# Progressive Generalization Augmentation with Deeply Coupled RND-PPO and Domain-Prioritized Noise Injection for Robust Crop Management Reinforcement Learning 

**Authors**: Wu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17428)  

**Abstract**: Our preliminary experiments on gym-DSSAT maize irrigation tasks revealed that +/-2 degrees C temperature noise causes an 11.9% reduction in economic returns for PPO policies trained under clean conditions - a systematic robustness deficit that existing research has not adequately addressed.
This paper tackles three interconnected limitations impeding practical deployment of agricultural RL systems: the trade-off between early-stage learning efficiency and late-stage generalization capability; the naive additive combination of intrinsic and extrinsic rewards in exploration-augmented PPO; and uniform measurement noise injection strategies that disregard empirically validated differential sensitivity across agricultural state variables.
We introduce three systematic innovations: Progressive Generalization Augmentation (PGA) implementing a three-phase curriculum (clean training 0-800 episodes, progressive 800-1200, full augmentation 1200-2000); a deeply coupled RND-PPO architecture with dual-channel GAE normalization, progress-decayed intrinsic coefficients, and semantic discretization; and domain-prioritized noise injection with hierarchical activation.
Our experimental evaluation demonstrates: 8.43% yield improvement and 16.42% nitrogen use efficiency improvement over SOTA BERT-DQN in Florida; 5.61% yield improvement in Zaragoza (though 3.67% lower economic score due to challenging Mediterranean climate); and 94.4% vs 80.0% performance retention under combined perturbations.
All experiments used 5 random seeds on NVIDIA A100 GPUs with 4.2+/-0.3 hours per run (2000 episodes, 2048-step buffer, 64 mini-batch size). 

---
# Learning Displacement-Robust Representations for Landslide Early Warning under Rainfall Forecast Uncertainty 

**Authors**: Ren Ozeki, Hamada Rizk, Hirozumi Yamaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2605.17419)  

**Abstract**: Rainfall-induced landslides pose a growing risk worldwide as climate change intensifies extreme rainfall events. To provide sufficient evacuation time, landslide early warning systems (LEWS) for real-time disaster monitoring must estimate near-future landslide risk by integrating observed rainfall with short-term rainfall forecasts from spatio-temporal environmental data streams. Although recent landslide prediction methods have improved predictive performance using statistical and deep learning approaches, most assume accurate rainfall inputs. In operational settings, however, landslide prediction relies on rainfall forecasts, which often contain spatial displacement of rainfall fields due to forecasting uncertainties. Such displacement can alter local accumulated rainfall and degrade prediction accuracy. To address this challenge, we propose a novel LEWS robust to rainfall field displacement. The key idea is to learn latent representations from rainfall and terrain data that remain stable under displacement in rainfall field motion, enabling reliable geospatial data integration for landslide risk estimation. The landslide prediction model is trained using Rainfall-Motion-Aware Contrastive Learning (RMCL), which introduces temporally correlated rainfall field perturbations to emulate forecast-induced displacement in rainfall-driven spatio-temporal environmental data streams. Experiments were conducted using two years of rainfall and terrain data across Japan, covering 19 regions with landslide events. The proposed system achieved up to 37% higher precision than state-of-the-art baselines. These results demonstrate that modeling rainfall as a moving spatial field and addressing rainfall field displacement during learning significantly improve the reliability of short-term landslide prediction in operational early warning systems. 

---
# Benchmarking Mythos-Linked Bug Rediscovery 

**Authors**: Isaac David, Arthur Gervais  

**Link**: [PDF](https://arxiv.org/pdf/2605.17416)  

**Abstract**: Anthropic's April 2026 Mythos materials combine benchmark claims with concrete bug-finding stories across OpenBSD, FreeBSD, Linux, FFmpeg, and browsers. This paper reports a controlled target-file rediscovery experiment on six public or high-confidence Mythos-linked systems tasks. Each model receives the same target file or files, read-only source tools, three repeats per task, and one manual target-matching rubric; prompts omit CVE identifiers, patch hashes, advisory text, author names, disclosure dates, and answer key root cause language. The experiment contains 54 counted model-task attempts: three models, six tasks, and three repeats, giving 18 attempts per model. GPT-5.5 xhigh achieves 5/18 target rediscoveries, covering 2/6 tasks; counting one wrong-target mpegts.c finding separately gives 3/6 distinct core bugs. Claude Opus 4.7 achieves 1/18 target rediscoveries, covering 1/6 tasks. Kimi K2 records 0/18 target rediscoveries. The dominant failure mode is early commitment to plausible alternate candidates within the assigned file: models often submit source-grounded hypotheses while missing the specific invariant corrected by public Mythos patch evidence. These results do not refute Anthropic's undisclosed workflow, but show that under this favorable target-file scaffold, systems-specific prompting yields only six target matches across 54 counted attempts. 

---
# IVF-TQ: Streaming-Robust Approximate Nearest Neighbor Search via a Codebook-Free Residual Layer 

**Authors**: Tarun Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2605.17415)  

**Abstract**: We propose IVF-TQ, an IVF index with a codebook-free residual layer: a fixed random rotation followed by precomputed Lloyd-Max scalar quantization depending only on (b, d). Only the IVF coarse partition is trained. Building on TurboQuant (Zandieh et al., 2025), the design substantially reduces a key failure mode of trained-codebook ANN indexes (PQ, OPQ, ScaNN): staleness under streaming this http URL (3 seeds): Per-batch PQ retraining does not recover the streaming gap at any tested bit budget (paired-t p > 0.28 everywhere). On streaming Deep-10M, IVF-TQ holds at 87.4% -> 86.6% (Delta = -0.80 +/- 0.10pp) while IVF-PQ degrades -3.23pp. A shuffled-i.i.d. control on SIFT-1M shows IVF-PQ losing -3.9pp without distribution shift. At higher PQ bit budgets (~1.5x IVF-TQ memory), absolute recall favors PQ as expected from rate-distortion (+6.1pp Deep-10M; +2.0pp SIFT-10M); the durable IVF-TQ benefit is operational (no codebook to retrain), robust across memory this http URL art: IVF around a codebook-free residual quantizer is architecturally not new -- IVF-RaBitQ ships in Milvus, cuVS, LanceDB, Weaviate; Shi et al. (2026) is concurrent GPU work. TurboQuant itself tests only flat-rotation this http URL: (i) A multi-seed streaming-operational story for codebook-free IVF: 10M-scale evidence across PQ memory budgets. (ii) A uniform-over-sphere IP-error bound for the TQ residual quantizer with one fixed rotation (proof sketch in v1; rigorous in v2). (iii) Adaptive IVF-TQ: a partition-only refresh recovering 67% -> 97.8% under worst-case rotation shift with re-ranking (90.3% without).Code, data: this https URL 

---
# Ablating Safety: Mechanisms for Removing Alignment in Language Models for Security Applications 

**Authors**: Isaac David, Arthur Gervais  

**Link**: [PDF](https://arxiv.org/pdf/2605.17413)  

**Abstract**: Safety-aligned language models often refuse cybersecurity requests whose wording resembles misuse, even when the task is authorized and defensive. This makes security evaluation ambiguous: a failed answer may reflect missing capability or refusal-policy intervention. Ablating Safety studies alignment removal as a controlled transformation-evaluation protocol for authorized security tasks, comparing authorized-context prompting, reversible refusal-direction activation projection, representation-control projections, and LoRA-based de-alignment or task adaptation.
We evaluate refusal, attempt rate, validated security success, general-capability retention, instability, and out-of-scope unsafe compliance on Security-AR, a 60-prompt suite of authorized security, benign general, and non-operational spillover probes. The reported runs include a four-model projection pilot with 416 completions, a three-model Qwen2.5 LoRA extension with 1,980 held-out completions, representation and robustness sweeps, and executable secure-repair validators. Single-vector refusal projection raises mean security score only from 0.46 to 0.50 while increasing unsafe compliance from 0.10 to 0.47; rank-4 refusal-subspace projection reaches 0.51 while matching the aligned spillover rate. Task-only LoRA raises mean security score to 0.87 with general score 0.83 and unsafe compliance 0.13, while refusal-suppression with retention raises spillover to 0.27. These results support evaluating alignment removal as a utility-risk frontier, not as an uncensoring recipe, and treating compliance alone as neither competence nor safe deployment. 

---
# Learning Faster with Better Tokens: Parameter-Efficient Vocabulary Adaptation for Specialized Text Summarization 

**Authors**: Gunjan Balde, Soumyadeep Roy, Mainack Mondal, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2605.17379)  

**Abstract**: Large language models pretrained on general-domain corpora often exhibit tokenization inefficiencies when applied to specialized domains. Although continual pretraining for domain adaptation partially alleviate performance degradation, it does not resolve the fundamental vocabulary mismatch. To address this gap, we introduce a targeted parameter-efficient domain adaptation approach that combines vocabulary adaptation with pretraining for LLM-based text summarization. Our unified framework augments pretrained tokenizers with domain-specific tokens while selectively replacing under-trained and unreachable tokens to limit parameter growth. We evaluate our approach on Llama-3.1-8B and Qwen2.5-7B across legal and medical summarization tasks on a challenge-oriented evaluation protocol focused on expert-driven text and summaries which typically has higher concentration of over-fragmented Out-of-Vocabulary (OOV) words. The vocabulary adaptation algorithm enhances the overall quality of the summarization model by improving semantic similarity between the generated summaries and their references. In addition, the adapted model produces summaries that incorporate more appropriate novel and domain-specific words, leading to improved coherence, relevance, and faithfulness. We further observe that our proposed approach significantly reduce training time by $35-55\%$ over continual pretraining and reduce parameter counts up to $37\%$ w.r.t expansion-only methods. We make the codebase publicly available at this https URL. 

---
# FML-bench: A Controlled Study of AI Research Agent Strategies from the Perspective of Search Dynamics 

**Authors**: Qiran Zou, Hou Hei Lam, Wenhao Zhao, Tingting Chen, Yiming Tang, Samson Yu, Yingtao Zhu, Srinivas Anumasa, Zufeng Zhang, Tianyi Zhang, Chang Liu, Zhengyao Jiang, Anirudh Goyal, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17373)  

**Abstract**: AI research agents accelerate ML research by automating hypothesis generation, experimentation, and empirical refinement. Existing agent strategies range from greedy hill-climbing to tree search and evolutionary optimization, yet which strategy choices drive performance remains unclear. Answering this question requires a benchmark that separates agent strategy (e.g., search topology) from execution infrastructure (e.g., code editor), so that performance differences are attributable to strategy rather than infrastructure, and that provides process-level metrics beyond final scores to analyze exploration behaviors. Existing benchmarks offer limited support. We propose FML-Bench, a benchmark of 18 fundamental ML research tasks across 10 domains that separates agent strategy from execution infrastructure and defines 12 process-level behavioral metrics. Evaluating six representative agents, we find that: (1) strategy complexity alone does not guarantee strong performance: a simple greedy hill-climber nearly matches the best-performing tree-search agent, both well above the remaining agents; (2) our analysis suggests this pattern relates to improvement opportunity structure: greedy search tends to be more effective when opportunities are dense, while tree-search and evolutionary strategies tend to be more effective when opportunities are sparse; an adaptive agent built on this insight switches to broader exploration upon detecting improvement stagnation and outperforms the other six agents, lending initial support to this observation; and (3) process-level analysis reveals that early convergence and directionally focused exploration are significantly associated with final performance, while solution diversity and compute cost are not. Our benchmark is available at: this https URL. 

---
# \textsc{MasFACT}: Continual Multi-Agent Topology Learning via Geometry-Aware Posterior Transfer 

**Authors**: Xuefei Wang, Jialu Wang, Fengbo Zhang, Yihan Hu, Di Zhang, Yutong Ye, Yikun Ban, Jun Han, Ruijie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17361)  

**Abstract**: Multi-agent systems (MAS) powered by large language models (LLMs) have emerged as a powerful paradigm for complex problem solving, where performance critically depends on the underlying inter-agent communication topology. However, existing topology generation methods mainly optimize for isolated tasks, while real-world deployments involve streams of evolving tasks, requiring previously effective collaboration patterns to be retained and reused rather than rediscovered or overwritten. We identify a previously underexplored failure mode, \emph{topology forgetting}, in which adapting to new tasks shifts the topology generator away from communication structures required by earlier tasks. This issue stems from cross-task misalignment in both agent-level functional semantics and relational communication structures. To address this challenge, we propose \textbf{\textsc{MasFACT}}, a geometry-aware posterior transfer framework that preserves and reuses historical collaboration knowledge as transferable topology priors. We transfer these priors across task-specific agent spaces through Fused Gromov-Wasserstein optimal transport and perform PAC-Bayes-guided conservative posterior adaptation to balance task-specific plasticity with structural stability. Experiments across class-, domain-, and task-level continual settings demonstrate that \textsc{MasFACT} consistently improves average accuracy while reducing topology forgetting compared to strong topology generation and replay-based baselines, and can be seamlessly integrated with different MAS topology generators. 

---
# Transitivity Meets Cyclicity: Explicit Preference Decomposition for Dynamic Large Language Model Alignment 

**Authors**: Yucong Huang, Xiucheng Li, Kaiqi Zhao, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.17342)  

**Abstract**: Standard RLHF relies on transitive scalar rewards, failing to capture the cyclic nature of human preferences. While some approaches like the General Preference Model (GPM) address this, we identify a theoretical limitation: their implicit formulation entangles hierarchy with cyclicity, failing to guarantee dominant solutions. To address this, we propose the Hybrid Reward-Cyclic (HRC) model, which utilizes game-theoretic decomposition to explicitly disentangle preferences into orthogonal transitive (scalar) and cyclic (vector) components. Complementing this, we introduce Dynamic Self-Play Preference Optimization (DSPPO), which treats alignment as a time-varying game to progressively guide the policy toward the Nash equilibrium. Synthetic data experiments further validate HRC's structural superiority in mixed transitive--cyclic settings, where HRC converges faster and achieves higher accuracy than GPM. Experiments on RewardBench 2 demonstrate that HRC consistently improves over both BT and GPM baselines (e.g., +1.23% on Gemma-2B-it). In particular, its superior performance in the Ties domain empirically validates the model's robustness in handling complex, non-strict preferences. Extensive downstream evaluations on AlpacaEval 2.0, Arena-Hard-v0.1, and MT-Bench confirm the efficacy of our framework. Notably, when using Gemma-2B-it as the base preference model, HRC+DSPPO achieves a peak length-controlled win-rate of 44.75% on AlpacaEval 2.0 and 46.8% on Arena-Hard-v0.1, significantly outperforming SPPO baselines trained with BT or GPM. Our code is publicly available at this https URL. 

---
# Single-Sample Black-Box Membership Inference Attack against Vision-Language Models via Cross-modal Semantic Alignment 

**Authors**: Jiaqing Li, Yajuan Lu, Xiaochuan Shi, Gang Wu, ZhongYuan Wang, Chao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17341)  

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable success, yet their reliance on massive datasets and unintended memorization of training data raise significant data security risk. Membership Inference Attacks (MIAs) aim to assess these risks by determining whether a data sample was included in a model's training set. However, existing MIA methods against VLMs face critical bottlenecks: gray-box method relies on internal logits that are typically restricted in real-world Application Programming Interfaces (APIs), while black-box method depends on large-scale statistical distributions, which struggle in single-sample scenarios. To this end, we investigate MIAs from the perspective of cross-modal semantic alignment, and observe that member images exhibit significantly stronger image-caption alignment due to training memorization, whereas generated captions for non-members may deviate from the original visual content. Leveraging this insight, we propose a novel MIA framework designed for strict black-box and single-sample setting that quantifies such alignment within a joint embedding space, thereby bypassing these unrealistic assumptions. We conducted extensive experiments on three open-source and two closed-source VLMs. On the VL-MIA/Flicker dataset, our method achieves an AUC of 0.821 against LLaVA-1.5, significantly outperforming existing baselines. Furthermore, it remains robust under diverse image perturbations, highlighting its practicality. 

---
# LPG: Balancing Efficiency and Policy Reasoning in Latent Policy Guardrails 

**Authors**: Nanxi Li, Zhengyue Zhao, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.17329)  

**Abstract**: Guardrails are a critical safety layer for modern AI systems, but their operating regime is changing. As LLMs are deployed as customized assistants, safety policies are increasingly specified at inference time by users, organizations, or regulatory contexts. This makes safety enforcement fundamentally dynamic: the guardrail should adapt to changing safety policies without retraining. Yet this requirement creates a fundamental tension: faithfully judging complex policy contexts demands reasoning capability, while practical deployment requires low-latency responses. We introduce Latent Policy Guardrail (LPG), a guardrail framework that learnssemantic latent deliberation over dynamic policies. LPG compresses the internal deliberation needed for intent interpretation and policy grounding into continuous states supervised by decision-relevant semantics. At inference time, it generates only a compact verdict anchored to the violated policy clauses, preserving auditability while avoiding the latency of explicit reasoning. Across policy guardrail benchmarks, LPG-4B reaches 84.5% average safety accuracy and 77.9% F1 by compressing deliberation into just 10 latent tokens, outperforming the strongest dynamic baseline while running roughly 11 times faster than Qwen3-4B-Thinking under the single-sample evaluation setup. Code and data are available at this https URL. 

---
# Efficient Feature-Free Initialization for Monocular Visual-Inertial Systems Using a Feed-Forward 3D Model 

**Authors**: Yuantai Zhang, Jiaqi Yang, Huajian Zeng, Changhao Chen, Haoang Li, Liang Li, Dezhen Song, Xingxing Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2605.17327)  

**Abstract**: Fast and reliable initialization is critical for monocular visual-inertial navigation systems (VINS), as it establishes the starting conditions for subsequent state estimation. Despite steady progress, most existing methods heavily rely on visual feature correspondences and require 3-4 seconds of sensory data for successful initialization, which limits their applicability and efficiency. With the advent of feed-forward 3D models that can directly predict point clouds from images, we revisit the visual-inertial initialization problem from a concise perspective. In this work, we propose a feature-free initialization framework that leverages up-to-scale point clouds predicted by a feed-forward 3D model, thereby obviating the need for visual feature tracking and estimation. This design substantially reduces system complexity and improves the reliability of initialization. Experiments on public datasets demonstrate that the proposed feature-free initialization method achieves the highest success rate, exceeding 90%, and significantly reduces the data duration required for successful initialization, typically to under 1.2 s. We further validate our method on a self-collected dataset covering various indoor and outdoor scenarios, demonstrating robust performance, particularly in visually degraded environments where existing methods often fail. The code and dataset are available at this https URL. 

---
# ASPI: Seeking Ambiguity Clarification Amplifies Prompt Injection Vulnerability in LLM Agents 

**Authors**: Udari Madhushani Sehwag, Zhengyang Shan, Heming Liu, Dileepa Lakshan, Joseph Brandifino, Max Fenkell  

**Link**: [PDF](https://arxiv.org/pdf/2605.17324)  

**Abstract**: Clarification-seeking behavior is widely regarded as a desirable property of LLM agents, enabling them to resolve ambiguity before acting on underspecified tasks. However, the security implications of this interaction pattern remain unexplored. We investigate whether the transition from standard execution to a clarification-seeking state increases an agent's susceptibility to prompt injection attacks. We introduce ASPI (Ambiguous-State Prompt Injection), a benchmark of 728 task-attack scenarios that isolates clarification as a distinct agent state and measures how this state transition affects vulnerability under controlled conditions. Each benchmark instance is evaluated under matched execution and clarification settings: in the execution setting, the agent acts on a fully specified instruction and encounters adversarial content only through tool-returned data; in the clarification setting, the agent must first request and incorporate additional user input before acting. We evaluate ten frontier LLMs and find that clarification-seeking consistently and substantially amplifies vulnerability. For instance, attack success rises from 1.8% to 34.0% for o3 and from 2.2% to 35.7% for Gemini-3-Flash. A decomposition analysis reveals that this gap reflects both a state-dependent shift in how models process incoming content and a channel-specific effect arising from the agent-solicited clarification interface. These findings demonstrate that standard execution-time security evaluation systematically underestimates the attack surface of interactive agents, and that robustness under fully specified tasks does not translate to robustness under ambiguity. For reproducibility, our data and source code are available at this https URL. 

---
# TClone: Low-Latency Forking of Live GUI Environments for Computer-Use Agents 

**Authors**: Yutong Huang, Vikranth Srivatsa, Alex Asch, Hansin Tushar Patwa, Yiying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17320)  

**Abstract**: Computer-use agents increasingly operate inside live personal workspaces, where their actions can modify files, applications, GUI state, credentials, and authenticated sessions. This creates a tension between safety and quality: agents need isolation and rollback to avoid damaging user state, but also need fast branching to support speculative execution and parallel search. Existing VMs, containers, and checkpoint/restore systems can isolate or recover workloads, but they do not provide low-latency versioning of a full interactive workspace.
We present TClone, a forkable personal workspace system for computer-use agents. TClone enables a live GUI workspace to be snapshotted, forked into isolated branches, rolled back, and selectively committed or merged. Its design separates fast branch creation from durable checkpointing, using sibling containers, copy-on-write memory sharing, filesystem versioning, GUI-local execution, and asynchronous checkpointing. In our end-to-end agent-loop measurement, TClone reduces total task latency by 1.9x and 1.5x over KVM and CRIU. By making workspace versioning a first-class systems primitive, TClone supports safer and higher-quality agent execution over real personal computing environments. 

---
# Learning Higher-Order Structure from Incomplete Spatiotemporal Data: Multi-Scale Hypergraph Laplacians with Neural Refinement 

**Authors**: Keshu Wu, Sixu Li, Zihao Li, Zhiwen Fan, Xiaopeng Li, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.17316)  

**Abstract**: Sensor networks increasingly govern modern infrastructure, yet the data they lose are rarely missing in the uniform-random patterns assumed by standard imputation benchmarks. Loop detectors go offline during calibration, roadside cabinets silence clusters of nearby sensors, and newly installed instruments provide no history. Such failures create structured absences whose values are constrained by higher-order relations among groups of sensors, not merely by pairwise proximity. Existing low-rank and graph-based methods often miss this collective structure and can fail when missingness becomes coherent. We introduce Multi-Scale Hypergraph Laplacians (MSHL), a two-stage framework for learning higher-order structure from incomplete spatiotemporal observations. The Discovery stage builds a multi-scale hypergraph from complementary topology and residual-correlation evidence, with an observation-only selector that adapts to the supported interaction scale. The Refinement stage adds a small hypergraph-conditioned residual network that is safe by construction: it learns nonlinear corrections where informative residual features exist and defers to the linear estimate where they do not. We prove that MSHL represents group-conservation patterns inaccessible to pairwise graph priors, adapts to the best fixed scale up to a logarithmic factor, transfers this advantage to held-out imputation error, and admits a one-sided refinement guarantee. On two real traffic networks evaluated across scattered cell missingness, contiguous block outages, and whole-sensor blackouts at five rates, MSHL improves over a pairwise-graph baseline whenever higher-order structure is identifiable and otherwise matches it within sampling noise. The results point to a broader principle for reliable infrastructure learning: missing data should be treated not as isolated entries to fill, but as evidence of structure to discover. 

---
# Weak-to-Strong Elicitation via Mismatched Wrong Drafts 

**Authors**: Wei Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.17314)  

**Abstract**: We consider whether off-policy experience from a smaller, weaker model can elicit capability in a stronger learner that on-policy RL fine-tuning (e.g., GRPO) does not reach. We find that injecting mathematically wrong drafts from a smaller but more domain-trained model -- mismatched to the current problem -- into a stronger learner's GRPO context consistently outperforms standard on-policy GRPO on held-out MATH-500 and out-of-distribution AIME 2025/2026. Concretely, we use Mathstral-7B as the learner, Qwen2.5-Math-1.5B as the draft model, 8.8K Level 3--5 MATH problems (with MATH-500 held out), and train with Dr. GRPO. Mismatch is an active ingredient: shuffling drafts to mismatched problems while holding everything else constant yields $+1.62$pp on MATH-500 (greedy pass@1) over the matched-wrong variant ($n=10$ seeds, $p=0.0015$, Welch's $t$). In fact, the mismatched-wrong variant leads all other variants we tested on MATH-500 across both greedy pass@1 and sampling pass@$k$. On out-of-distribution AIME 2025 and 2026, the mismatched-wrong variant uniquely lifts pass@$k$ above both Mathstral-7B (in its native [INST] format) and the Qwen2.5-Math-1.5B draft model at every sample budget from $k=1$ to $k=1024$ across 2 seeds ($+14.2$pp on 2025 and $+9.0$pp on 2026 at pass@1024 over Mathstral-7B), and at pass@1024 also leads no-draft, matched-wrong, and mismatched-correct variants on both years. All variants use the same prompt with no draft injection at test time. The recipe -- trained on a single GPU with no SFT, no reward models, no synthesized data, and no produce-critique-revise inner loop -- reaches 71.98% MATH-500 on Mathstral-7B-v0.1, the highest published result on this model to our knowledge, surpassing the heavier WizardMath pipeline at 70.9% on full MATH (SFT + PPO with process/instruction reward models). 

---
# Attention Hijacking: Response Manipulation Across Queries in Vision-Language Models 

**Authors**: Zhiqiang Wang, Dongrui Liu, Yan Li, Zonghao Ying, Wei Xue, Wenhan Luo, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.17310)  

**Abstract**: Existing adversarial attacks on vision-language models (VLMs) can steer model outputs toward attacker-specified target responses, but their effectiveness often degrades when the same perturbed input is paired with different textual queries. This paper studies cross-query response manipulation, where a single adversarial example is expected to remain effective across diverse user queries. We first analyze the limitations of existing attacks and find that successful transfer is closely associated with preserving an image-dominant attention pattern during response generation. Motivated by the observation, we propose \textbf{Attention Hijacking}, a novel adversarial attack that explicitly steers internal attention distributions toward a persistent image-dominant pattern. By amplifying the influence of visual tokens on target response tokens while suppressing the competing influence of textual tokens, our method reduces the dependence of the manipulated output on the specific wording of the query. Extensive experiments on widely used VLMs show that Attention Hijacking substantially improves cross-query transferability across diverse target responses and unseen queries. The method also extends effectively to multiple attack scenarios, offering new insights into the role of attention stability in transferable response manipulation for VLMs. 

---
# StyleText: A Large-Scale Dataset and Benchmark for Stylized Scene Text Inpainting 

**Authors**: Aleksandr Simonyan, Nipun Jindal  

**Link**: [PDF](https://arxiv.org/pdf/2605.17309)  

**Abstract**: We present StyleText, a large-scale dataset and benchmark for localized scene-text inpainting with style preservation. StyleText contains 28,518 image-mask-prompt triplets grouped into 9,932 scene families, enabling controlled evaluation of text legibility and visual consistency under shared scene context. We construct the dataset with an automated pipeline that combines LLM prompt templating, Flux-based source generation with key-value (KV) cache injection, OCR-based semantic filtering, polygon mask extraction, and mask-conditioned FluxFill augmentation. We define a reproducible evaluation protocol using normalized OCR metrics (word accuracy and character error rate) and CLIP image-image similarity with explicit preprocessing. A FluxFill+LoRA baseline trained on StyleText improves OCR accuracy substantially over initialization while maintaining scene style consistency, establishing a strong reference point for future comparisons. 

---
# Deep Reinforcement Learning Framework for Diversified Portfolio Management Across Global Equity Markets 

**Authors**: Kamil Kashif, Robert Ślepaczuk  

**Link**: [PDF](https://arxiv.org/pdf/2605.17307)  

**Abstract**: This study develops and evaluates a deep reinforcement learning framework for dynamic portfolio allocation across global equity markets. The Soft Actor-Critic algorithm is used to learn continuous portfolio weights within a Markov Decision Process, incorporating transaction costs, turnover penalties, and diversification constraints into the reward function. Five model configurations are compared, varying in reward formulation, policy structure (flat versus hierarchical Dirichlet), portfolio constraints, and temporal encoder (LSTM versus Transformer), and evaluated via walk-forward optimization across sixteen out-of-sample folds spanning 2003-2026 on the Nasdaq-100, Nikkei 225, and Euro Stoxx 50. Results show that RL strategies achieve competitive risk-adjusted performance primarily in the Euro Stoxx 50, where statistically significant abnormal returns are observed, but the central hypothesis is only partially confirmed: no strategy achieves statistically significant excess returns relative to Buy and Hold under HAC-robust inference across all markets. Regime analysis reveals that RL adds the most value during periods of elevated uncertainty, while ensemble aggregation across markets improves risk-adjusted performance and confirms the benefits of geographic diversification. 

---
# ConflictRAG: Detecting and Resolving Knowledge Conflicts in Retrieval Augmented Generation 

**Authors**: Chenyu Wang, Yingmin Liu, Yang Shu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17301)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems implicitly assume mutual consistency among retrieved documents -- an assumption that frequently fails in practice. We present ConflictRAG, a conflict-aware RAG framework that detects, classifies, and resolves knowledge conflicts prior to answer generation. The framework introduces three contributions: (1) a two-stage conflict detection module combining a lightweight embedding-based MLP classifier with selective LLM refinement, reducing API costs by 62% while maintaining 90.8% detection accuracy; (2) an Entropy-TOPSIS framework for data-driven source credibility assessment, improving selection accuracy by 7.1% over manual heuristics; and (3) a Conflict-Aware RAG Score (CARS) for diagnostic evaluation of conflict-handling capabilities. Experiments on three benchmarks against six baselines demonstrate 88.7% conflict-detection F1 and consistent 5.3--6.1% correctness gains over the strongest conflict-aware baseline, with the pipeline transferring effectively across backbone LLMs. 

---
# LEAP: Learnable End-to-End Adaptive Pruning of Large Language Models 

**Authors**: Mohammad Mozaffari, Younes Hourri, Mohammad Rastegari, Mahyar Najibi  

**Link**: [PDF](https://arxiv.org/pdf/2605.17289)  

**Abstract**: Unstructured sparsity is now natively accelerated by recent GPU kernels and dataflow hardware, shifting the bottleneck from inference execution to the pruning algorithm. State-of-the-art methods for unstructured LLM pruning are layer-wise surrogates derived from the Optimal Brain Surgeon principle, and they sacrifice end-to-end accuracy, especially under aggressive sparsity. End-to-end alternatives such as MaskLLM and PATCH show that learnable masks can close this gap, but their categorical-over-patterns parameterization scales with the number of valid masks per row and does not port to the unstructured setting. We introduce LEAP, which replaces this intractable parameterization with a per-weight Bernoulli-via-Gumbel- sigmoid relaxation that makes end-to-end unstructured mask learning tractable. Across five LLM families from 0.5B to 8B parameters at 50% and 60% sparsity, LEAP improves six-task average zero-shot accuracy by +2.59 points on average over ADMM, the best layer-wise baseline in our sweep. 

---
# When Efficiency Backfires: Cascading LLMs Trigger Cascade Failure under Adversarial Attack 

**Authors**: Zehan Sun, Dingfan Chen, Songze Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.17288)  

**Abstract**: Large Language Model (LLM) cascade systems are designed to balance efficiency and performance by processing queries with lightweight models while selectively escalating complex cases to more powerful ones. Such systems seek to reduces computational cost and latency while maintaining task performance, making it an appealing choice for large-scale deployment. However, the cascade design introduces new vulnerabilities through an expanded attack surface: the inclusion of lightweight front-end models and internal decision mechanisms introduces new weaknesses. In this work, we present the first study demonstrating that LLM cascade systems are susceptible to targeted adversarial manipulation, which disrupts both performance objectives and the intended cost advantages of the cascade design. We propose a novel attack framework that employs constrained sequential collaborative optimization of adversarial suffix under cascade dependencies, enabling simultaneous exploitation of lightweight models and decision mechanisms. This framework adapts to adversaries with varying capabilities, inducing controllable degradation in both cost-efficiency and accuracy. Unlike prior attacks targeting standalone models, our approach strategically leverages the cascade structure to achieve significantly stronger impact. Extensive experiments across diverse datasets and representative LLM cascade systems validate the practicality and severity of this attack. Our findings highlight the urgent need to rigorously scrutinize the security of LLM cascade systems and call for broader attention to the systemic risks inherent in such designs. 

---
# UNR-Explainer: Counterfactual Explanations for Unsupervised Node Representation Learning Models 

**Authors**: Hyunju Kang, Geonhee Han, Hogun Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.17285)  

**Abstract**: Node representation learning, such as Graph Neural Networks (GNNs), has emerged as a pivotal method in machine learning. The demand for reliable explanation generation surges, yet unsupervised models remain underexplored. To bridge this gap, we introduce a method for generating counterfactual (CF) explanations in unsupervised node representation learning. We identify the most important subgraphs that cause a significant change in the k-nearest neighbors of a node of interest in the learned embedding space upon perturbation. The k-nearest neighbor-based CF explanation method provides simple, yet pivotal, information for understanding unsupervised downstream tasks, such as top-k link prediction and clustering. Consequently, we introduce UNR-Explainer for generating expressive CF explanations for Unsupervised Node Representation learning methods based on a Monte Carlo Tree Search (MCTS). The proposed method demonstrates superior performance on diverse datasets for unsupervised GraphSAGE and DGI. 

---
# CLAP: Contrastive Latent-space Prompt Optimization for End-to-end Autonomous Driving 

**Authors**: Ruiyang Zhu, Yuehan He, Boyuan Zheng, Zesen Zhao, Ahmad Chalhoub, Qingzhao Zhang, Z. Morley Mao  

**Link**: [PDF](https://arxiv.org/pdf/2605.17284)  

**Abstract**: End-to-end autonomous driving systems powered by Vision-Language-Action (VLA) models achieve strong performance on common driving scenarios, yet remain brittle in rare but safety-critical long-tail situations such as active construction zones and complex yielding geometries. In this paper, we present a method that addresses the long-tail challenging scenes beyond data scaling and model training. We introduce CLAP (Contrastive Latent-space Prompt optimization), a location-aware adaptation framework that augments a frozen VLA driving model with per-roadblock soft prompts, optimized from crowdsourced data and retrieved on demand via Vehicle-to-Everything (V2X) communication. Our approach rests on two observations from VLAs' latent space: (i) at the VLA's hidden-state layer, scenarios from the same roadblock cluster tightly and occupy compact regions of the latent space; and (ii) within a single roadblock, long-tail and normal frames are heavily intermixed in the latent representation, making it difficult to improve one without disturbing the other. CLAP addresses this via a two-stage pipeline: supervised contrastive learning to discover a roadblock-specific hard-scene direction, followed by directionally regularized prompt optimization that selectively improves challenging frames while preserving normal frame performance. On the NAVSIM benchmark with various state-of-the-art VLA backbones, CLAP reduces challenging scenario planning error by 24% with no regression on normal frames, significantly improving planning performance. 

---
# OProver: A Unified Framework for Agentic Formal Theorem Proving 

**Authors**: David Ma, Kaijing Ma, Shawn Guo, Yunfeng Shi, Enduo Zhao, Jiajun Shi, Zhaoxiang Zhang, Gavin Cheung, Jiaheng Liu, Zili Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17283)  

**Abstract**: Recent progress in formal theorem proving has benefited from large-scale proof generation and verifier-aware training, but agentic proving is rarely integrated into prover training, appearing only at inference time. We present OProver, a unified framework for agentic formal theorem proving in Lean 4, in which failed proof attempts are iteratively revised using retrieved compiler verified proofs and Lean compiler feedback. OProver is trained through continued pretraining followed by iterative post-training: each iteration runs agentic proving, indexes newly verified proofs into OProofs and the retrieval memory, uses repair trajectories as SFT data, and uses unresolved hard cases for RL. OProofs is built from public Lean resources, large-scale proof synthesis, and agentic proving traces, containing 1.77M Lean statements, 6.86M compiler-verified proofs, and serialized trajectories with retrieved context, failed attempts, feedback, and repairs. Across five benchmarks, OProver-32B attains the best Pass@32 on MiniF2F (93.3%), ProverBench (58.2%), and PutnamBench (11.3%), and ranks second on MathOlympiad (22.8%) and ProofNet (33.2%) more top placements than any prior open-weight whole-proof prover. 

---
# ContractBench: Can LLM Agents Preserve Observation Contracts? 

**Authors**: Jicheng Wang, Yifeng He, Zili Wang, Hanwen Xing, Arkaprava De, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.17281)  

**Abstract**: Tool-augmented LLM agents call APIs whose intermediate outputs, such as presigned URLs, session tokens, and OAuth state parameters, are observation contracts: artifacts whose later use is constrained by the external system that produced them. We show that observation contract compliance (preserving the temporal validity and byte-level integrity) is an emergent, regression-prone capability: it is neither guaranteed by general tool-use ability nor consistently improved by larger or newer models. To measure this, we introduce ContractBench, a benchmark of 33 dual-axis tasks that probe two orthogonal failure modes no existing benchmark evaluates: validity failures (using an artifact after expiry) and integrity failures (corrupting an artifact's bytes through the observation-to-action pipeline). Our evaluation is deterministic and programmatic, with a virtual clock controlling time and SHA-256 hashes verifying byte integrity. We assign each outcome a failure label drawn from real-world API specifications. We evaluate 38 models and report four findings: (i) no evaluated model clears 80%, with Claude-Opus-4.6 leading at 77.8%, revealing that current frontier models still fail to comply with observation contracts; (ii) a sharp within-family capability cliff in Qwen 3.5 between 4B (0%) and 9B (56.6%), smoothing to 70.7% at 397B-A17B: what emerges across the cliff is mid-trajectory restraint, not tool-call competence; (iii) non-monotonic scaling across the GPT-5 family: agentic post-training can erode compliance through sycophancy-driven regression; (iv) our failure taxonomy works as an actionable in-context reward signal, yielding +7.1 pp on 42 paired GPT-5.1 failures. 

---
# Rover: Context-aware Conflict Resolution with LLM 

**Authors**: Qingyu Zhang, Junzhe Li, Jiayi Lin, Changhua Luo, Chenxiong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2605.17279)  

**Abstract**: Code merging is a significant challenge, particularly in large-scale projects. Existing solutions, including program analysis and machine learning, show promise but face critical limitations. Program analysis lacks the ability to infer developers' intentions, relying on conservative strategies that offload unresolved conflicts for manual handling. Meanwhile, model-based approaches struggle with conflicts involving complex code dependencies due to insufficient contextual awareness. To address these gaps, we introduce Rover, a novel conflict resolution system that integrates program analysis with large language models (LLMs). To obtain context-aware prompts, we propose Multi-layer Code Property Graph (MtCPG), a new representation capturing inter-file dependencies and enabling contextual analysis for a given conflict. Using graph connectivity algorithms, Rover further clusters conflicting code and associated changes into meaningful "contexts" that guide the LLM in generating accurate resolutions. We compared Rover with standalone LLMs, machine learning baseline MergeGen, and suggestion provider tool WizardMerge with adjacent code as the contexts. Evaluation results show that Rover surpasses all of these approaches in terms of conflict resolution, achieving higher similarity to ground-truth resolutions at character, lexical, and semantic levels. 

---
# How Do Electrocardiogram Models Scale? 

**Authors**: Jiawei Li, Fabio Bonassi, Ming Jin, Stefan Gustafsson, Johan Sundström, Thomas B. Schön, Antônio H. Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2605.17276)  

**Abstract**: While scaling laws have established a fundamental framework for foundation models in natural language processing, their applicability to electrocardiogram (ECG) models remains poorly characterized. Indeed, recent studies do not always yield consistent downstream gains as one increases the model size or pre-training dataset size of ECG models, leaving the exact roles of architectural inductive biases, pre-training paradigms, and expected improvements with size largely unanswered. In this work, we systematically investigate neural and loss-to-loss scaling laws within the ECG domain. By pre-training over $120$ models (ranging from $20$K to $200$M parameters) on the large-scale CODE dataset ($2.3$M records), we decouple the effects of model architecture (ResNet vs. Transformer) and pre-training paradigm, namely supervised learning (SL) versus self-supervised learning (SSL). We found that (i) SL models are data-bottlenecked in-distribution, whereas SSL models scale robustly across both model and data sizes; (ii) for out-of-distribution (OOD) generalization, ResNets are $1.3$ to $2.5$ times more parameter-efficient than Transformers, while SSL is up to $16$ times more data-efficient and achieves up to $7.6$ times higher transfer efficiency than SL on unseen clinical tasks; (iii) across the observed scales, ResNet-based models generally achieve the lowest OOD loss, with SSL dominating on unseen clinical tasks and self-supervised Transformers overtaking at very large model sizes. Our results suggest that the path to effective ECG foundation models lies in the strategic alignment of architecture and paradigm rather than brute-force scaling. 

---
# State-of-the-Art Claims Require State-of-the-Art Evidence 

**Authors**: YongKyung Oh  

**Link**: [PDF](https://arxiv.org/pdf/2605.17273)  

**Abstract**: State-of-the-Art (SOTA) claims pervade Artificial Intelligence (AI) and Machine Learning (ML) research. These claims rest on benchmark evaluations, where models are ranked by aggregate scores across tasks. Public benchmarks or leaderboards are the most visible instance, but the same structure appears in paper tables throughout the literature. However, such minimal evidence often cannot support these strong claims. We identify a widespread claim-evidence gap in AI benchmarking. Claiming SOTA carries implicit assumptions beyond mean score superiority, suggesting that a model meaningfully outperforms alternatives across most tasks. However, a marginal improvement in the mean score merely indicates a top average rank rather than true superiority. Analyzing ten cross-domain benchmarks from public leaderboards, we found that in more than half of top-model comparisons, at least one commonly assumed property of superiority does not hold. These properties include meaningful effect size, consistency across tasks, or robustness to dataset removal. Instead, aggregate gains are frequently driven by outlier datasets. This fragility persists even in benchmarks with many tasks. We argue that claim language should reflect the strength of the underlying evidence. This requires no additional experiments, only honest reporting of what results actually show, enabling more precise and interpretable comparisons across models. 

---
# Latency-Aware Deep Learning Benchmark for Real-Time Cyber-Physical Attack and Fault Classification in Inverter-Dominated Power Grids 

**Authors**: Emad Abukhousa, Saman Zonouz, A.P. Sakis Meliopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2605.17256)  

**Abstract**: This work introduces a latency-aware benchmarking framework for evaluating deep learning models in power system anomaly detection using high-fidelity, time-domain signals generated from an industry-grade electromagnetic transient simulator. Eight neural network architectures, ranging from MLPs to Transformers, were systematically evaluated on streaming datasets representing both physical faults and cyber-attacks in inverter-dominated networks. All models successfully classified two representative multi-event sequences in real time with sub-cycle response times below 15 ms. However, although classification decisions occurred within one cycle, the end-to-end inference latency consistently exceeded three cycles, ranging from 50 to 90 ms. These results highlight a critical gap between algorithmic capability and protection-grade deployment, pointing to the need for further optimization and hardware acceleration. The findings establish a reproducible benchmark for sub-cycle anomaly detection and provide guidance for transitioning machine learning methods from research prototypes to real-world protection applications. 

---
# Fidelity Probes for Specification--Code Alignment 

**Authors**: Ferhat Erata, Hao Zhou, Luke Huan  

**Link**: [PDF](https://arxiv.org/pdf/2605.17246)  

**Abstract**: We introduce fidelity probes: natural-language questions generated from a reference artifact with code-derived ground-truth answers, answered from a candidate specification. The fraction of agreeing probes, which we call the fidelity, decomposes into contradiction and coverage-gap rates that drive targeted spec edits to convergence. On a 15-program, roughly 12k-line COBOL benchmark (AWS CardDemo), we raise frozen-test specification fidelity from 0.63 to 0.94 over eight iterations, with the plateau location predicted by a two-state Markov fixed point $F^\dagger$ from just four iterations of rate data. Probes come from an LLM reading the code or from a static-analysis pipeline over its control-flow, data-flow, and system-dependence graphs, with a tunable mixture. A probe-resampling protocol with a frozen held-out set gives a Hoeffding-bounded overfitting discriminant; our measured train/test gap stays more than an order of magnitude below this envelope. Three graph-grounded mixtures lift fidelity by +16 to +30 points; cross-distribution evaluation shows the LLM and symbolic channels are empirically complementary. A cross-family generator sweep on five independent LLM lineages (Anthropic, DeepSeek, Google, Alibaba, OpenAI) confirms the convergence behaviour is not tied to any single model family: three of five non-Claude generators produce trajectories consistent with the Markov fixed-point prediction, and the frozen-test protocol actively falsifies the two generators whose probe distributions drift across iterations. The method applies to any pair of artifacts that are supposed to describe the same behaviour. 

---
# Drift Flow Matching 

**Authors**: Chenrui Ma, Xi Xiao, Lin Zhao, Tianyang Wang, Ferdinando Fioretto, Yanning Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.17244)  

**Abstract**: Iterative generative models such as Flow Matching and Diffusion models have demonstrated strong test-time scaling behavior, where additional inference computation can improve generation quality. In contrast, Drift Models offer efficient one-step generation, but their direct generation paradigm limits such flexibility. In this work, we propose Drift Flow Matching (DFM), a framework that connects drifting generative modeling with flow-based iterative generation. DFM preserves the efficiency of direct transport maps while enabling generation to be refined through multiple inference steps when desired. This bridges the gap between one-step Drift Models and multi-step Flow Matching methods, and provides a novel generative paradigm that can adapt sampling computation to different quality--efficiency requirements. Extensive experiments across different tasks and datasets demonstrate the effectiveness and generality of the proposed framework. 

---
# Systematic Evaluation of Vision Transformers for Automated Cervical Cancer Classification: Optimization, Statistical Validation, and Clinical Interpretability 

**Authors**: Nisreen Albzour, Sarah S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2605.17236)  

**Abstract**: Manual Pap smear analysis for cervical cancer screening is limited by inter-observer variability, time constraints, and restricted expert availability. Although convolutional neural networks (CNNs) have automated cervical cell classification, they remain limited in modeling long-range spatial dependencies and often lack clinical interpretability. In this study, Vision Transformer (ViT) architectures were systematically optimized to enhance automated cervical cancer screening, which resulted in improved interpretability. The Herlev dataset (917 images: 242 normal, 675 abnormal) was utilized to optimize ViT-Tiny, a lightweight Vision Transformer architecture designed for reduced computational complexity, through a comprehensive evaluation of augmentation strategies, class weighting, and hyperparameters. The optimal configuration achieved 94.9%-95.2% cross-validation accuracy, in which random horizontal flipping and class weighting (0.7 x 1.3) were identified as most effective. Gradient-weighted Class Activation Mapping (Grad-CAM) analysis confirmed that model attention corresponded to clinically relevant morphological features, which include nuclear regions, cell boundaries, and chromatin texture, which align with cytopathological criteria. These findings indicate that Vision Transformers can deliver accurate and interpretable decision support for cervical cancer screening, which fulfills both clinical performance and transparency requirements essential for medical AI deployment. 

---
# Integration of AI in Cybersecurity: Current Trends with a Focused Look at Intrusion Detection Applications 

**Authors**: S. Tazili, A. Mansour, M. Y. Chkouri  

**Link**: [PDF](https://arxiv.org/pdf/2605.17219)  

**Abstract**: Artificial Intelligence (AI) is widely adopted today for its ability to detect patterns, automate tasks, and reduce time and cost across various applications. Its integration into Cybersecurity has garnered significant attention, particularly in areas such as intrusion detection, malware analysis, and phishing or spam detection. As AI and cybersecurity evolve, new methods and approaches emerge regularly. Current trends include the use of Generative AI, Natural Language Processing, Federated Learning for privacy-preserving collaborative training, and eXplainable AI to ensure interpretability and trust, which are vital in cybersecurity. This paper presents an interesting review of current AI-based cybersecurity trends, focusing on intrusion detection approaches and aiming to uncover meaningful insights through comparative analysis based on the employed AI techniques and reported performance. 

---
# Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies 

**Authors**: Xinchen Jin, Aditya Chatterjee, Pranav Kumar, Rohan Paleja  

**Link**: [PDF](https://arxiv.org/pdf/2605.17204)  

**Abstract**: Vision-Language-Action (VLA) policies translate language and visual inputs into robot actions, where their hidden representations directly shape closed-loop behavior. However, mechanistic interpretability tools from language and vision-language models do not transfer cleanly to VLAs: outputs are robot actions rather than human-readable tokens, and interventions can only be tested via expensive closed-loop rollouts. We propose an event-grounded interpretability pipeline that anchors SAE feature analysis to behavioral events rather than text contexts. End-effector keyframes are clustered within each task using visual, state, and temporal cues, linking SAE features to behaviorally salient events and, via optional VLM annotations, to semantic context. To our knowledge, our pipeline is among the first to ground SAE-based VLA analysis in closed-loop behavioral events. Across two simulation architectures and a real-robot study, event-grounded ranking yields the strongest causal effects on OpenVLA and transfers to the continuous action chunks of $\pi_{0.5}$. SAE is a sparse but imperfect intervention basis: usability varies with architecture and intervention site, and aggressive intervention reveals safety and interpretability limits. Overall, event-grounded SAE analysis emerges as a practical starting point for behavior-anchored VLA interpretability, motivating future work on SAE features beyond action-aligned coordinates, finer-grained closed-loop evaluation, and safe interventions for high-stakes VLA deployments. Code is available at \url{this https URL}. 

---
# PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media 

**Authors**: Zoher Kachwala, Bao Tran Truong, Rasika Muralidharan, Haewoon Kwak, Jisun An, Filippo Menczer  

**Link**: [PDF](https://arxiv.org/pdf/2605.17187)  

**Abstract**: Social media are shifting towards pluralism -- community-governed platforms where groups define their own norms. What violates rules in one community may be perfectly acceptable in another. Can AI models help moderate such pluralistic communities? We formalize the task as a multiple-choice problem, mirroring how human moderators operate in the real world: given a comment and its surrounding context, identify which specific rule, if any, is violated. We introduce PluRule, a multimodal, multilingual benchmark for detecting 13,371 rule violations across 1,989 Reddit communities spanning 2,885 rules in 9 languages. Using this benchmark, we show that state-of-the-art vision-language models struggle significantly: even GPT-5.2 with high reasoning performs only slightly better than a trivial baseline. We also find that bigger models and increased context provide marginal gains, and universal rules like civility and self-promotion are easier to detect. Our results show that moderation of pluralistic communities on social media is a fundamental challenge for language models. Our code and benchmark are publicly available. 

---
# MusicSynth: An Automated Pipeline for Generating Violin Fingerboard Animations from Sheet Music Using Optical Music Recognition 

**Authors**: Abhimanyu Kaushik  

**Link**: [PDF](https://arxiv.org/pdf/2605.17181)  

**Abstract**: Learning the violin is harder than it looks. Unlike piano keys or guitar frets, the violin neck has no markings at all, so a beginner cannot tell by looking where to place each finger. MusicSynth is an open-source web tool that tries to fix that: user uploads a photo of any violin sheet music (or a digital score file), and the system automatically produces a video showing a violin fingerboard with each note highlighted at the right moment -- no software to install, no manual note entry required.
The system connects three existing open-source tools into one pipeline: an optical music recognition (OMR) library reads the notes from the uploaded image, a MusicXML parser extracts timing information from digital scores, and a video renderer draws the fingerboard frame by frame. The only part built from scratch is the lookup table that maps each musical note to a string and finger position on the violin.
Tested across 110 public-domain violin scores, MusicSynth correctly identified 91.2\,\% of notes in clean printed music and assigned the right finger position 99.1\,\% of the time when given a digital score file. To the author's knowledge, no freely available tool currently turns a sheet music image into an animated violin fingerboard tutorial automatically and in a single browser-based step. 

---
# Beyond Execution: Static-Analysis Rewards and Hint-Conditioned Diffusion RL for Code Generation 

**Authors**: Shuyin Ouyang, Zhaozhi Qian, Faroq AL-Tam, Muhammad AL-Qurishi, Jie M. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17174)  

**Abstract**: Reinforcement Learning (RL) is an important paradigm for aligning Diffusion Language Models (DLMs) toward functional correctness in code generation. However, these models often encounter a ``capability cliff'' on complex tasks, where execution-based semantic rewards become too low to provide a viable learning signal. In this paper, we present a systematic empirical study of RL post-training for diffusion-based code generation along three axes: reward design, hint-conditioned sampling, and task difficulty. We investigate the effectiveness of execution-free rewards as alternatives to traditional unit-test execution, the role of training-time hint-conditioned diffusion sampling in mitigating exploration bottlenecks, and the impact of these design choices varies across tasks with different difficulty levels. Across HumanEval, MBPP, and LiveCodeBench, we find that static checking is the strongest overall standalone execution-free reward in our setting, especially improving DiffuCoder from 53.9 to 67.1 on HumanEval and from 14.9 to 15.5 on LiveCodeBench while reducing rollout time by 9.4\%. We further find that moderate AST-based hinting is most useful on harder benchmarks, while the best reward design depends strongly on task difficulty: similarity-based rewards are more effective on easier subsets, whereas static checking is more reliable on harder subsets where execution rewards are low. These findings suggest that reward design and training guidance substantially affect diffusion RL performance in our evaluated code-generation setting. 

---
# Why Do Safety Guardrails Degrade Across Languages? 

**Authors**: Max Zhang, Ameen Patel, Sang T. Truong, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2605.17173)  

**Abstract**: Large language models exhibit safety degradation in non-English languages. Standard evaluation relies on Jailbreak Success Rate (JSR), which confounds several safety-driving factors into one, obscuring the specific cause(s) of safety failure. We introduce a latent variable model, a Multi-Group Item Response Theory (IRT) framework, that decouples safety-driving factors such as language-agnostic safety robustness ($\theta$), intrinsic prompt hardness ($\beta$), global language processing difficulty ($\gamma$), and a prompt-specific cross-lingual safety gap ($\tau$). Using the MultiJail dataset, we evaluate the safety robustness of 61 model configurations across 5 closed-model families and 10 languages of varying resource, aggregating a dataset of 1.9 million rows. Exploratory Factor Analysis shows safety is primarily unidimensional: models refuse different harm types mainly through a shared mechanism. Contrary to the expected trend that safety degrades largely in low-resource languages, 22 model configurations are more vulnerable in English than in low-resource languages. Low-resource languages produce more uncertain responses (high entropy) than high-resource languages. Also, high-$\tau$ prompts cluster in physical harm categories like Theft and Weapons and lower-resource languages, trends validated through cross-dataset generalization. While global translation quality shows low correlation with $\tau$, severe mistranslations drive high-bias outliers, as validated by native speakers. Cultural and conceptual grounding mismatches also contribute to $\tau$. In predictive validation, the IRT framework achieves $\mathrm{AUC} = 0.940$, outperforming simpler baselines in predicting safe refusal of unsafe prompts. Our framework reveals concept-language vulnerabilities that aggregate metrics obscure, enabling fairer cross-lingual safety evaluation and targeted improvements in dataset construction. 

---
# OpenJarvis: Personal AI, On Personal Devices 

**Authors**: Jon Saad-Falcon, Avanika Narayan, Robby Manihani, Tanvir Bhathal, Herumb Shandilya, Hakki Orhun Akengin, Gabriel Bo, Andrew Park, Matthew Hart, Caia Costello, Chuan Li, Christopher Ré, Azalia Mirhoseini  

**Link**: [PDF](https://arxiv.org/pdf/2605.17172)  

**Abstract**: Personal AI stacks, like OpenClaw and Hermes Agent, are becoming central to daily work, yet they route nearly every query (often over sensitive local data) to cloud-hosted frontier models. Replacing frontier models with local models inside existing stacks does not work: swapping Claude Opus 4.6 for Qwen3.5-9B drops accuracy by 25-39 pp across personal AI tasks like PinchBench and GAIA. Existing stacks bundle agentic prompts, tool descriptions, memory configuration, and runtime settings around a specific cloud model. Only the prompts can be tuned, and state-of-the-art prompt optimizers close just 5 pp of the local-cloud gap on their own. This motivates a decomposed personal AI stack: one that exposes individual primitives which can be optimized individually or jointly to close the local-cloud gap. We present OpenJarvis, an architecture that represents a personal AI system as a typed spec over five primitives: Intelligence, Engine, Agents, Tools & Memory, and Learning. Each primitive is an independently editable field, making the stack end-to-end optimizable and measurable against accuracy, cost, and latency. Towards closing the local-cloud gap without surrendering local-model properties, OpenJarvis introduces LLM-guided spec search, a local-cloud collaboration in which frontier cloud models propose edits across the spec at search time, only non-regressing edits are accepted, and the resulting spec runs entirely on-device at inference time. With LLM-guided spec search, on-device specs match or exceed cloud accuracy on 4 of 8 benchmarks and land within 3.2 pp of the best cloud baseline on average. They also reduce marginal API cost by ~800x and end-to-end latency by 4x. 

---
# Charon: A Unified and Fine-Grained Simulator for Large-Scale LLM Training and Inference 

**Authors**: Mengtian Yang, Zhekun Zhang, Mingheng Wu, Jianwen Yan, Hanshi Sun, Li-wen Chang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17164)  

**Abstract**: Deploying large-scale LLM training and inference with optimal performance is exceptionally challenging due to a complex design space of parallelism strategies, system optimizations, and hardware configurations. Accurate and rapid performance simulation is critical for guiding optimization efforts and system studies by validating "what-if" Hooker Figure hypotheses. To address this, we introduce Charon, a unified, modular, and fine-grained simulator for accurately predicting LLM performance. Experiments show Charon achieves high accuracy across different models and configurations, with an overall prediction error consistently under 5.35%, and even under 3.74% for training with a large-scale GPU cluster. In a practical inference deployment case, Charon discovered a configuration that improved system throughput over an engineering-tuned baseline, demonstrating its significant real-world value. 

---
# STRIDE-AI: A Threat Modeling Framework for Generative AI Security Assessment 

**Authors**: Tsafac Nkombong Regine Cyrille, Franziska Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2605.17163)  

**Abstract**: Traditional cybersecurity methodologies target deterministic systems and fail to address the probabilistic nature of AI, leaving systems vulnerable to attack vectors such as model inversion, data poisoning, and prompt injection. Recent industry reports indicate that a majority of organizations deploying AI lack a dedicated security strategy, with adversarial attacks increasing rapidly year-over-year. We present \textit{STRIDE-AI}, a framework that bridges the gap between high-level risk standards (NIST AI RMF) and technical vulnerability taxonomies (OWASP LLM Top 10). The framework defines a six-phase assessment lifecycle, introduces a threat modeling adaptation of classical STRIDE for AI systems, and is operationalized through a purpose-built web tool. We provide an initial validation of the approach through a black-box assessment of a deployed LLM chatbot, which successfully reduced the attack success rate from 80\% to 15\% in our sandbox case study. 

---
# When Bits Break Recourse: Counterfactual-Faithful Quantization 

**Authors**: Chaymae Yahyati, Ismail Lamaakal, Khalid El Makkaoui, Ibrahim Ouahbi  

**Link**: [PDF](https://arxiv.org/pdf/2605.17160)  

**Abstract**: Quantization can preserve predictive accuracy under low-bit deployment while silently breaking algorithmic recourse: an actionable change that flips a decision before quantization may fail after quantization, or become substantially more costly. We formalize counterfactual sensitivity under quantization through validity, cost, and direction stability, and introduce two metrics: Validity Drop (VD) and Counterfactual Recourse Gap (CRG) that reveal recourse failures invisible to accuracy. We propose Counterfactual-Faithful Quantization (CFQ), which trains quantizer parameters and mixed-precision bit allocation to preserve counterfactual behavior by enforcing the target outcome at teacher recourse points under a global bit budget. A margin-based analysis gives a sufficient condition for recourse transfer under bounded quantization perturbations. Experiments on Adult, German Credit, and COMPAS show that accuracy-matched baselines can significantly degrade recourse stability, while CFQ maintains accuracy and substantially improves VD and CRG across bit budgets. 

---
# Evolutionary Extreme Learning Machine of ab-initio Energy Landscapes for Crystal Structure Prediction using Manta Ray Optimization with Levy Flight 

**Authors**: Adrian Rubio-Solis  

**Link**: [PDF](https://arxiv.org/pdf/2605.17148)  

**Abstract**: The Manta Ray Foraging Optimization algorithm (MRFO) has proven to be a powerful heuristic strategy in the optimal solution of a large number of engineering problems. In this paper, an improvement of MRFO with Levy Flight is suggested for the training of extreme learning machines (ELMs) whose basic model is a Single Layer Feedforward Network (SLFN). The proposed methodology that we called Evolutionary EELM-MRFO-LF for short is implemented to the prediction of unrelaxed and relaxed formation energy compounds relative to ground state crystal structure of pure components in binary systems. EELM-MRFO-LF follows the learning procedure of traditional Evolutionary ELMs in which first MRFO with LF is used to select the input weights and Moore-Penrose (MP) generalized inverse is applied to analytically determine the output weights. Levy Flight trajectory is implemented for increasing the diversity of the population of ELMs against premature convergence and the ability of avoiding getting trapped in a local optima. The performance of the suggested EELM-MRFO-LF is compared with other well-known nature-inspired algorithms under similar conditions. 

---
# Contrastive Conceptor Activation Steering (COAST): Unlocking Vision-Language-Action Models through Hidden States 

**Authors**: Miranda Muqing Miao, Subin Kim, Brandon Yang, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2605.17144)  

**Abstract**: Vision-Language-Action (VLA) models leverage powerful perceptual priors from web-scale Vision-Language Model (VLM) pre-training, yet they remain surprisingly brittle in practice, frequently failing at simple robotic tasks. To mitigate this, we propose Contrastive Conceptor Activation Steering (COAST). COAST builds on the notion of a "conceptor", a linear operator that soft-projects data into the principal components of a target distribution. COAST uses conceptors to identify success-critical subspaces for a target robotic task from a few examples of success and failure rollouts. At inference time, it steers VLA latents into these identified success subspaces to improve task outcomes. Across three architecturally distinct neural policies (flow-matching VLA, autoregressive VLA, and Diffusion Policy), COAST improves absolute mean simulation and real-robot task success rate by over 20 and 40% respectively. The activation subspace geometry reveals that failure modes share substantial structure across tasks while success representations remain largely task-specific. When tasks share similar failure modes, this structure enables previously fitted conceptors to improve performance on new tasks without refitting. Ultimately, our results suggest that current VLAs retain substantial task-relevant knowledge in their latent representations, and that the action expert's decoding bottleneck could be mitigated by steering its residual stream toward task-relevant subspaces. COAST provides a lightweight, training-free path to unlocking these latent capabilities by steering the model towards its own "success" distributions. 

---
# UCSF-PDGM-VQA: Visual Question Answering dataset for brain tumor MRI interpretation 

**Authors**: Shiv Ghosh, Junayd Lateef, Chih-Hua, Yannan Yu, Andreas M. Rauschecker, Madhumita Sushil  

**Link**: [PDF](https://arxiv.org/pdf/2605.17140)  

**Abstract**: Brain tumor diagnosis is largely dependent on Magnetic Resonance Imaging (MRI) evaluation, which requires radiologists to synthesize thousands of images across multiple 3D sequences and longitudinal studies. This process requires advanced neuro-radiology training, poses substantial cognitive load, and is highly time-consuming. Despite increasing demands in radiology, this expertise is difficult to scale, straining the current health systems. Vision-Language Models (VLMs) provide an opportunity to reduce this burden through a semi-automated, interactive interpretation of complex brain MRIs. However, they are currently underutilized in neuro-oncology due to a lack of specialized benchmarks for evaluating them. We introduce a clinically relevant visual question answering (VQA) benchmark -- the UCSF-PDGM-VQA dataset -- consisting of 2,387 QA pairs from 473 glioma-related MRI studies in the public UCSF-PDGM dataset. We further establish a performance baseline for six state-of-the-art vision-language models (VLMs) and one large language model on this dataset. We find that current models are incapable of effectively processing multi-sequence, 3-dimensional MRI scans, thus resulting in a suppression of visual features and over-reliance on language priors, causing modality collapse. These findings underscore a critical deficiency in current model reliability and safety within clinical settings, necessitating the development of robust, domain-specific VLMs. 

---
# CAM-VFD: Cross-Attention Multimodal Video Forgery Detection 

**Authors**: Hoda Osama Elkhodary, Sherin Mostafa Youssef, Marwa Elshenawy, Dalia Sobhy  

**Link**: [PDF](https://arxiv.org/pdf/2605.17133)  

**Abstract**: The rapid advancement of Deepfake technologies and video manipulation tools poses a critical challenge to multimedia forensics, judicial evidence integrity, and information authenticity. Current detectors rely on single-modality signals, treating appearance, geometry, and motion independently. However, advanced generators maintain within-modality consistency while producing cross-modal contradictions, which are forensically discriminative but invisible to any single-modal detector. We propose CAM-VFD, a Cross-Attention Multimodal Video Forgery Detection framework that models cross-modal contradiction as a directional forensic signal. The framework uses a cross-attention fusion mechanism in which CLIP-based appearance representations serve as queries against VideoMAE motion features and MiDaS depth features, enabling the identification of contradictions between visual, temporal, and geometric evidence. We examine this design through cross-modal attention discrepancy analysis, observing statistically separable real and fake distributions ($p<0.001$, Cohen's $d=0.68$). Experimental results on two generative video benchmarks indicate consistent performance, with 95.31\% Top-1 accuracy on GenVidBench and 93.43\% accuracy, 90.63\% F1-score, and 96.56\% AUROC on GenVideo. Moreover, CAM-VFD demonstrates stable performance under compression, noise, blur, and adversarial perturbations, suggesting that cross-modal reasoning may improve robustness in media forensics. The code is publicly available at \url{this https URL}. 

---
# A Systematic Survey on Deep Learning Architectures for Point Cloud Classification and Segmentation 

**Authors**: Minhas Kamal, Hiranya Garbha Kumar, Balakrishnan Prabhakaran  

**Link**: [PDF](https://arxiv.org/pdf/2605.17131)  

**Abstract**: Point cloud stands as the most widely adopted format for representing 3D shapes and scenes due to its simplicity and geometric fidelity. However, its inherent unordered and irregular nature, exacerbated by sensor noise and occlusions, introduces unique challenges for machine learning based methodologies. To combat these issues, diverse strategies have been developed, including converting to a format that has orderliness, extracting local geometry, and permutation-invariant or self-attention-based processing. In this paper, our focus is directed towards deep learning models for three fundamental tasks in 3D vision: point cloud classification, part segmentation, and semantic segmentation. We begin by formally defining point cloud data, followed by an in-depth discussion on its structural characteristics. Then, we categorize notable works based on their backbone structure and evaluate their performance on popular benchmarks. Beyond empirical comparison, we offer insights into architectural innovations and limitations. We also outline open challenges and promising future directions for 3D point cloud understanding. 

---
# New Wide-Net-Casting Jailbreak Attacks Risk Large Models 

**Authors**: Qiuchi Xiang, Haoxuan Qu, Hossein Rahmani, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17128)  

**Abstract**: Jailbreak attacks on large models have drawn growing attention due to their close ties to societal safety. This work identifies a practical yet unexplored jailbreak scenario, the wide-net-casting scenario, where an adversary can query a group of large models instead of a single one to elicit harmful outputs. Our analysis reveals substantial yet previously overlooked safety risks under this scenario. As a key part of our analysis, we further develop a novel jailbreak method tailored to the wide-net-casting scenario. With this tailored method, the jailbreak success rate can even reach 100\% in some experiments when targeting the large models without additional safeguards, exposing wide-net-casting as a distinct, high-risk scenario that warrants attention in future evaluation and defense research. 

---
# The Point of No Return: Counterfactual Localization of Deceptive Commitment in Language-Model Reasoning 

**Authors**: Scott Merrill, Shashank Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2605.17113)  

**Abstract**: Existing deception datasets label completed outputs as honest or deceptive, treating deception as a property of the final response rather than a function of the model's reasoning trace. This obscures a more fundamental question: when does a language model become committed to deception? We introduce counterfactual localization: for each sentence prefix in a reasoning trace, we fix the prefix, resample continuations, and estimate the probability of a deceptive outcome. To scale this, we construct five environments (spanning strategic bluffing, maze guidance, financial advice, used-car sales, and offer negotiation) in which deception is never prompted but emerges from strategic incentives and labels follow mechanically from environment state rather than subjective human judgment. The resulting corpus localizes $\sim$1.46M sentences across four reasoning models, drawn from over 94.1M sampled continuations, 91.5B generated tokens, and over 100K scenarios. Sentence-level human evaluation confirms that detected commitment points correspond to interpretable shifts in decision state. Using this resource, we show that lexical cues for commitment prediction transfer poorly across environments, whereas attention-based transition features generalize out of distribution, suggesting that deceptive commitment is reflected in reusable changes in reasoning dynamics rather than surface form. We further identify compact attention-head sets (under 10% of heads) that, selected on one environment, causally suppress deceptive commitment across held-out environments. We release the corpus as a substrate for studying deception, and more broadly commitment, in language-model reasoning. 

---
# DynMuon: A Dynamic Spectral Shaping View of Muon 

**Authors**: Fangzhou Wu, Rikhav Shah, Sandeep Silwal, Qiuyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17109)  

**Abstract**: In recent years, Muon has emerged as the dominant method for training large language models, and transformers more broadly. The essential difference, when compared to standard gradient descent methods, is to replace the usual update matrix $M=U\Sigma V^\top$ with its polar factor $UV^\top$. In this work, we consider a class of Muon-like updates, where we replace the update $M$ with $U\Sigma^p V^\top$ for some parameter $p$. We call this a "spectral-shaping" operation, and develop a theory of how to pick $p$ which depends on (a) local curvature of the loss function, (b) noise stemming from stochastic gradients and label noise, and (c) training stage. Our theory and experimentation reveal a previously overlooked behavior: positive $p$ helps early by emphasizing high-curvature directions and accelerating signal contraction, while mildly negative $p$ helps later by reallocating update strength toward low-curvature directions that still contain useful training signals. Building on the insight, we propose DynMuon, an efficient dynamic spectral shaping method that schedules $p$ from positive to mildly negative over training. Extensive experiments across model sizes, architectures, and training settings show that DynMuon consistently achieves lower validation loss than Muon, while requiring 10.6-26.5% fewer steps to reach the same target loss. 

---
# SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning 

**Authors**: Yongfeng Huang, Ruiying Chen, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.17101)  

**Abstract**: Retrieval-Augmented Generation (RAG) is widely employed to mitigate risks such as hallucinations and knowledge obsolescence in medical question answering, yet its predominantly single-round, static retrieval paradigm misaligns with the multi-stage process of clinical reasoning. This compressed workflow induces two structural deficiencies: question-to-query translation often lacks clinically grounded semantic interpretation, and retrieval lacks iterative sufficiency feedback, making it difficult to form reliable evidence chains. We argue that both issues stem from a deeper cause: overloading a single reasoning chain with heterogeneous tasks of interpretation, exploration, and adjudication. The remedy is to reconstruct the workflow via task decoupling and dynamic multi-round exploration. To this end, we propose SEMA-RAG, a Self-Evolving Multi-Agent RAG framework for medical question answering, which assigns these roles to three specialist agents: the Interpreter Agent for clinical schema interpretation, the Explorer Agent for sufficiency-driven self-evolving retrieval, and the Arbiter Agent for evidence adjudication and answer selection. Across five benchmarks and five LLM backbones, SEMA-RAG improves the strongest baseline by +6.46 accuracy points on average, measured per backbone. 

---
# Visual Timelines of Police Encounters in Body-Worn Camera Footage: Operational Context and Activity Cataloging for Training and Analysis in OpenBWC 

**Authors**: Angela Srbinovska, Christopher Homan, Adrian Martin, Ernest Fokoué  

**Link**: [PDF](https://arxiv.org/pdf/2605.17095)  

**Abstract**: Law enforcement agencies are accumulating vast amounts of body-worn camera (BWC) footage. However, this remains operationally opaque. That is, analysts and trainers still have to invest considerable time watching full-length videos to pinpoint the start of key encounters and identify the points where activity shifts to something more physically intense. We present an approach to process BWC video into a time-aligned sequence of fixed-length 10-second windows, processed and labeled using a privacy-conscious protocol. Each window is labeled with two dimensions of information: (i) the operational context of the window and (ii) the level of motion intensity within the window, with low-evidence labels for windows for which insufficient evidence exists due to darkness, blur or occlusion. We train models to classify windows based on these two axes using frames sampled from each window encoded using CLIP model and aggregated into a window-level representation. We extract dense optical flow statistics for each window to capture motion intensity. On test windows the best context model achieves 78.75% accuracy, and the best-accuracy activity model achieves 88.33%. We also included integrity audits to show the results and how the visual timeline representations support faster incident review and make the officer training workflow more practical. 

---
# Global Automation Atlas 

**Authors**: Prashant Garg, Tommaso Crosta, Jasmin Baier  

**Link**: [PDF](https://arxiv.org/pdf/2605.17086)  

**Abstract**: Automation affects the labour content of work differently across different contexts. Yet, most existing exposure measures assign fixed scores to tasks or occupations, limiting comparisons of automation exposure across countries. We develop a task-based and country-specific approach to classify automation exposure across the world to disentangle labor-substituting from labor-augmenting automation, the relevant technology channel, and the material role of AI. Our measure spans 124 countries, generating an atlas of 2.33 million task-country labels for economies covering 99% of world population and GDP. We present five descriptive results. First, exposure is highly uneven, ranging from 3.3% of tasks in South Sudan to 61.6% in China, and rises strongly with income, although substantial variation remains within income groups. Second, across countries, exposed tasks are skewed towards substitution rather than augmentation, but low-income countries are disproportionately exposed to substitution, whereas middle-income countries are more heterogeneous. Third, less technologically advanced forms of automation account for more than half of exposed tasks in low-income countries but about one quarter in high-income countries; while other more complex channels generally rise with income levels. Fourth, AI tends to be less prevalent in simpler channels of automation, but also more prevalent in labour-substituting margins in lower income settings and to augment labour in higher income settings. Fifth, we find that females seem to be disproportionately more exposed to labour-substituting automation than males. Our methodology provides a basis for comparing automation exposure across development stages, linking it with cross-country data and allowing us to treat exposure levels, labour margins, technological channels and AI involvement as separate dimensions. 

---
# Can LLMs Think Like Consumers? Benchmarking Crowd-Level Reaction Reconstruction with ConsumerSimBench 

**Authors**: Tianyu Wang, Jiajun Li, Jianghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.17079)  

**Abstract**: LLMs are increasingly used as ``digital consumers'' to simulate public opinion, pre-test marketing decisions, and anticipate audience response. However, existing evaluations rarely ask whether a model can reconstruct the concrete reaction patterns that real consumers surface in public discourse. We introduce ConsumerSimBench, a benchmark built from 1,553 real Chinese social-media topics and 23,122 atomic, rule-audited criteria spanning four reaction families. Rather than scoring open-ended generations with a holistic preference judge, ConsumerSimBench decomposes each task into auditable yes-no decisions over concrete reaction points, raising three-judge agreement from 65.8% to 92.1% with 98.4% agreement between pointwise judge decisions and human-majority labels. Across 13 frontier generators, the strongest model, Gemini-3.1-Pro, covers only 47.8% of real reaction criteria, while GPT-5.2 and Claude-4.6 trail far behind despite their strength on technical benchmarks. The failures reveal a sharp gap between technical-benchmark performance and socially grounded consumer intuition. A direct structured reasoning prompt decreases coverage, while a generate--reflect multi-agent pipeline improves MiMo-V2.5-Pro from 32.9% to 37.6% on a subset. ConsumerSimBench reframes consumer simulation as a forecasting problem over real public-discourse reactions, showing that frontier LLMs remain far from reliably predicting what consumers will actually care about in high-context Chinese consumer discourse. 

---
# How to Instruct Your Robot: Dense Language Annotations Power Robot Policy Learning 

**Authors**: Bosung Kim, Ruiyi Wang, David Acuna, Jaehun Jung, Alexander Trevithick, Brandon Cui, Yejin Choi, Prithviraj Ammanabrolu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17077)  

**Abstract**: Scaling robot policy learning is bottlenecked by the cost of collecting demonstrations, while language annotations for existing demonstrations are comparatively cheap. We study language density as a lever for extracting more signal from a fixed robot or egocentric-video corpus. We introduce DeMiAn (Dense Multi-aspect Annotation), a two-stage approach that first re-labels demonstration segments with VLM-generated annotations along four complementary aspects: physical motion, scene composition, arm pose, and reasoning. A learned instructor then maps a task description and initial scene snapshot to a task-appropriate annotation at deployment, running asynchronously so generation latency is hidden behind policy execution. Across over 1M robot manipulation clips and 50K EgoVerse human-egocentric videos, DeMiAn improves both a vision-language-action policy and a video-based world-action model without collecting new demonstrations. On RoboCasa, the instructor raises success by 5 points over a task-only baseline and comes within 3 points of a per-task oracle. No fixed annotation aspect dominates across tasks, showing that selecting the right dense language matters. DeMiAn also improves composite-task and out-of-distribution performance, and shifts the compute-performance frontier in both mid-training and post-training after accounting for annotation-generation FLOPs. These results position dense re-annotation as a practical scaling lever for robot policy learning. 

---
# S-Bus: Automatic Read-Set Reconstruction for Multi-Agent LLM State Coordination 

**Authors**: Sajjad Khan  

**Link**: [PDF](https://arxiv.org/pdf/2605.17076)  

**Abstract**: Concurrent LLM agents sharing mutable natural-language state produce Structural Race Conditions (SRCs): write-write and cross-shard stale-read conflicts that silently corrupt agent output. Existing multi-agent frameworks (LangGraph, CrewAI, AutoGen) provide no write-ownership semantics over shared state.
We present S-Bus, an HTTP middleware whose central mechanism is a server-side DeliveryLog: a per-agent log of HTTP GET operations that automatically reconstructs each agent's read set at commit time without agent SDK changes under HTTP/1.1. The consistency property the DeliveryLog provides -- Observable-Read Isolation (ORI), a partial causal consistency over the HTTP-observable projection of the read set -- prevents structural race conditions when agents collaborate via shared shards.
Three contributions: (C1) The DeliveryLog mechanism for automatic HTTP-traffic-based read-set reconstruction, with three-tier mechanised evidence: ReadSetSoundness and ORICommitSafety machine-checked in TLAPS (modulo one retained typing axiom); exhaustive TLC at N=3 (20,763,484 distinct states, zero violations); Dafny discharges 9 inductive soundness lemmas. (C2) Empirical structural-conflict prevention parity against PostgreSQL 17 SERIALIZABLE and Redis 7 WATCH/MULTI on shared-shard contention sweeps with 427,308 active HTTP-409 conflicts: zero Type-I corruptions across all three backends. (C3) ORI's operating envelope is topology-conditional: semantically neutral in dedicated-shard workloads; harmful in single-shard collaborative writing because preservation propagates concurrent contradictions.
Source code: this https URL 

---
# 1GC-7RC: One Graphic Card -- Seven Research Challenges! How Good Are AI Agents at Doing Your Job? 

**Authors**: Robin-Nico Kampa, Fabian Deuser, Anna Bößendörfer, Konrad Habel, Norbert Oswald  

**Link**: [PDF](https://arxiv.org/pdf/2605.17046)  

**Abstract**: Autonomous AI coding agents are becoming a core tool for ML practitioners in industry and research alike. Despite this growing adoption, no standardized benchmark exists to evaluate their ability to design, implement, and train models from scratch across diverse domains. We introduce **1GC-7RC** (*Single Graphic Card: Seven Research Challenges*), a benchmark comprising seven ML tasks spanning language modeling, image classification, semantic segmentation, graph learning, tabular prediction, time-series forecasting, and text classification. Each task provides a locked data-preparation and evaluation script together with a baseline training script; the agent may only modify the training code, has no access to pretrained weights (with one controlled exception for semantic segmentation), no internet access, and must complete each task within a task-specific wall-clock budget (40-120 minutes) on a single GPU. We evaluate seven coding agents: five proprietary (Claude Code with Sonnet 4.6, Opus 4.6, and Opus 4.7; Codex CLI with GPT 5.5; and OpenCode with Qwen 3.6+) and two open-source (OpenCode with Kimi K2.5, Kimi K2.6). Across 5 runs per agent-task pair, we report substantial performance differences that reveal varying levels of implicit ML knowledge, planning ability, and time-budget management. The benchmark, harness, and all evaluation artifacts are publicly available on GitHub at this https URL to facilitate reproducible comparison of future agents. Because our benchmark design is modular, the benchmark can be extended to new tasks and domains, adapted to different GPU budgets, and used to study multi-agent settings, making it a flexible platform for future research on autonomous research agents. 

---
# Agentic AI Translate: An Agentic Translator Prototype for Translation as Communication Design 

**Authors**: Masaru Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2605.17041)  

**Abstract**: We present Agentic AI Translate, an agentic translator prototype that operationalises the thesis of Yamada (forthcoming) -- that the metalanguage of Translation Studies has become an instruction code for generative AI. The system replaces the dominant text-in / text-out paradigm of machine translation with a four-stage agentic cycle (Identify -> Prompt -> Generate -> Verify), preceded by an interactive specification phase in which the user composes -- through model-assisted dialogue -- a structured translation brief grounded in skopos theory, register, audience, and genre conventions. The verification stage adopts the GEMBA-MQM error-span protocol (Kocmi & Federmann, 2023) for evidence-grounded scoring, and document-level coherence is preserved through a DelTA-lite memory of proper nouns and a running bilingual summary, after Wang et al. (2025). We describe the philosophical motivation, the architectural commitments, the four reference-material categories the system consumes, and the principal design tensions the architecture makes explicit. Empirical validation is left for future work; the contribution here is conceptual and architectural -- an executable embodiment of the position that translation in the GenAI era is communication design, not text conversion. 

---
# D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning 

**Authors**: Ru Zhang, Renda Li, Ziyu Ma, Weijie Qiu, Chongyang Tao, Yong Wang, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17037)  

**Abstract**: Reinforcement learning (RL) has demonstrated potential for enhancing reasoning in large language models (LLMs). However, effective RL training, which requires medium-difficulty training samples, faces two fundamental challenges: Effective Data Scarcity and Dynamic Difficulty Shifts, where medium-difficulty samples are scarce and become trivial as models improve. Existing methods mitigate this scarcity to some extent by generating training samples. However, these approaches suffer from anchor-free generation, ignoring co-evolution, and difficulty mismatch. To address these issues, we propose D$^2$Evo, a Dual Difficulty-aware self-Evolution RL framework. In each iteration, our method mines medium-difficulty anchors based on the current Solver's capability, trains the Questioner to generate diverse questions at appropriate difficulty levels, and jointly optimizes both components to enable progressive reasoning gains. Extensive experiments demonstrate that D$^2$Evo outperforms existing methods on mathematical reasoning benchmarks with fewer than 2K real mathematical samples, and exhibits strong generalization on general reasoning benchmarks. 

---
# Privacy Policy Enforcement Guardrails for Data-Sensitive Retrieval-Augmented Generation 

**Authors**: Osama Zafar, Alexander Nemecek, Yiqian Zhang, Wenbiao Li, Debargha Ganguly, Vikash Singh, Vipin Chaudhary, Erman Ayday  

**Link**: [PDF](https://arxiv.org/pdf/2605.17034)  

**Abstract**: Standard PII filters often miss contextual data leakage in RAG systems, such as non-regulated attribute clusters that collectively identify individuals. We introduce a Privacy Policy Enforcement (PPE) framework using dual one-class density estimators with fused text embeddings and a calibrated abstain region for out-of-distribution inputs. Using an axis-stratified, multi-LLM synthetic data pipeline across medicine, finance, and law, we found that traditional Gaussian Mixture baselines fail on borderline-safe stress tests by focusing on linguistic register rather than content.
Our proposed T3+OCSVM detector, trained on safe and borderline-safe data, achieves a borderline AUROC of 0.93+ while reducing false positives by 44-55 percentage points and maintaining millisecond latency. Compared to supervised MLP classifiers or 14B-parameter LLM judges, our framework offers superior operational suitability, as the former suffers from high abstention rates and the latter from latency and calibration issues. This methodology provides a robust stress-testing standard for any synthetic-data-trained classifier. 

---
# Task Abstention for Large Language Models in Code Generation 

**Authors**: Yanke Zhou, Yuhao Tan, Senrong Xu, Zenan Li, Yuan Yao, Taolue Chen, Xiaoxing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.17029)  

**Abstract**: Large language models (LLMs) have revolutionized automated code generation. One serious concern, however, is the so-called ``hallucination'', i.e., LLMs may generate seemingly plausible but functionally incorrect code. In this paper, we study the task abstention problem, i.e., determining whether a given LLM should abstain from performing a specific code generation task to avoid likely hallucination. Our approach features a calibrated abstention rule, grounded in the principles of multiple hypothesis testing. The rule assesses generation consistency through code execution outcomes, allowing it to handle syntactic diversity of semantically equivalent code without reliance on oracle test cases or external databases. We prove that our approach provides a rigorous, distribution-free theoretical guarantee on its abstention decisions. We evaluate our method on benchmark datasets using several open-source code LLMs. Results show that our method allows generative models to more accurately and efficiently identify and abstain from tasks that induce hallucination compared to existing techniques, providing a reliable mechanism for safer and more robust code generation. 

---
# PARALLAX: Separating Genuine Hallucination Detection from Benchmark Construction Artifacts 

**Authors**: Khizar Hussain, Murat Kantarcioglu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17028)  

**Abstract**: Large language models (LLMs) hallucinate with confidence: their outputs can be fluent, authoritative, and simply wrong. In medical, legal, and scientific applications this failure causes direct harm, and detecting it from internal model states offers a path to safer deployment. A growing body of work reports that this problem is increasingly tractable, with recent methods achieving high detection performance on widely used benchmarks. We show, however, that much of this apparent progress does not survive scrutiny. Four of the six corpora embed the ground-truth answer directly in the input prompt. A naïve text-similarity baseline we call \textsc{TxTemb} exploits this to achieve near-perfect detection scores without any access to model internals. To measure what genuine detection capability remains once these artifacts are controlled, we conduct a large-scale evaluation spanning twenty-two detection methods, twelve open-source models spanning six architectural families, and six corpora. We further introduce \textbf{DRIFT}, a supervised probe over inter-layer hidden-state transitions, as a point of comparison for live-generation detection. Our findings suggest that the field's reported progress on hallucination detection is substantially explained by benchmark construction artifacts in widely used corpora, and that the majority of established baselines perform near chance under controlled conditions; the consistent exceptions are SAPLMA and DRIFT, both supervised probes on upper-layer hidden states. 

---
# When Dynamics Shift, Robust Task Inference Wins: Offline Imitation Learning with Behavior Foundation Models Revisited 

**Authors**: Rishabh Agrawal, Rahul Jain, Ashutosh Nayyar  

**Link**: [PDF](https://arxiv.org/pdf/2605.17017)  

**Abstract**: Behavior Foundation Models (BFMs) enable scalable imitation learning (IL) by pretraining task-agnostic representations that can be rapidly adapted to new tasks. However, existing BFMs assume fixed environment dynamics, limiting their robustness under real-world shifts such as changes in friction, actuation, or sensor noise. We address this by formulating BFM task-inference as a robust minimax optimization problem, enabling adaptation to worst-case dynamics perturbations without modifying pretraining. To the best of our knowledge, this is the first BFM-based framework that achieves robustness to dynamics shifts while relying solely on offline data from a single nominal environment. Our approach significantly outperforms standard BFM and robust offline IL baselines under dynamics shifts. These results demonstrate that robust policy can be achieved entirely at task-inference time, improving the practicality of BFMs in dynamic settings. 

---
# Algorithmic Cultivation: How Social Media Feeds Shape User Language 

**Authors**: Olivia Pal, Agam Goyal, Eshwar Chandrasekharan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2605.17010)  

**Abstract**: Algorithmic feeds have become primary environments for encountering information online, yet while they shape what people see, less is known about how sustained feed exposure shapes how people write. Drawing on Cultivation Theory, we examine whether algorithmic feeds function as online environments that leave measurable traces in users' language. We leverage a large-scale longitudinal dataset of 235M posts by 4M users on Bluesky, and conduct a quasi-experimental study matching an initial pool of 368,513 users exposed to one of three feeds -- News, Science, and Blacksky -- with a pool of 2,001,915 active control users who did not engage with any of these feeds. We examine linguistic evolution across three dimensions: lexico-semantics, psycholinguistics, and topics. We find that users exposed to these feeds show significantly greater stylistic accommodation, semantic alignment, and register formalization than matched controls. These effects vary markedly by feed identity -- Blacksky produces the deepest psycholinguistic restructuring, with significant shifts in cognitive processing, affective expression, and pronoun use, while News and Science effects are largely confined to register and topical focus. Regression models reveal that reposting is the most consistent predictor of linguistic convergence across all feeds, whereas posting and bookmarking show feed-dependent effects, with effects differing more than fourfold across feeds. Our work extends Cultivation Theory beyond belief formation to linguistic behavior, demonstrating that feeds function as persistent linguistic environments that gradually shape what and how users write online. Our work has implications for studying algorithmic influence, online identity formation, and the design and governance of feed-based platforms that mediate online interactions. 

---
# The IsalProgram Programming Language 

**Authors**: Ezequiel López-Rubio  

**Link**: [PDF](https://arxiv.org/pdf/2605.17008)  

**Abstract**: We introduce IsalProgram (Instruction Set and Language for Programming), a novel assembly-like programming language with three distinctive theoretical properties: (1) it is a regular language in the sense of formal language theory, meaning its programs are accepted by a finite automaton; (2) every finite string over the instruction alphabet is a syntactically valid program; and (3) it makes no explicit use of memory addresses or variable names, absolute or relative. Programs are finite sequences of tokens drawn from a fixed instruction set, and are executed on a virtual machine whose sole data structure is a circular doubly linked list (CDLL) navigated by three data pointers, with control flow governed by two code pointers. We give a complete formal definition of the language and its virtual machine, prove its regularity, and demonstrate its expressive power. We further discuss IsalProgram's potential advantages as a target language for neural program synthesis, the amenability of its program space to metric-based exploration via the Levenshtein edit distance, and directions for analyzing computability and complexity within this framework. 

---
# Learning-Zone Energy: Online Data Selection for Efficient RL Post-Training 

**Authors**: Peng Cui, Boyao Yang, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17003)  

**Abstract**: Reinforcement Learning (RL) post-training has emerged as the dominant paradigm for eliciting mathematical reasoning in Large Language Models (LLMs), yet prevailing techniques such as GRPO and DAPO distribute rollout and gradient budgets nearly uniformly across prompts, squandering compute on samples that are already mastered or remain far beyond the model's current capability. To address this fundamental inefficiency, we propose Learning-Zone Energy (LZE), a theoretically grounded, fully online data selection framework that concentrates computation on the model's active learning frontier. At its core, we define a closed-form Learning-Zone Energy Score that fuses three complementary signals, an initial-difficulty anchor, a normalized outcome-uncertainty term, and a pass-rate momentum, into a single scalar that is provably aligned with the expected magnitude of group-relative policy gradient updates. A forward pruner with replay further reduces wall-clock time cost by skipping rollout generation for persistently solved prompts while periodically checking for forgetting. Evaluated on Qwen-family models (1.5B-8B) across GSM8K, MATH and DAPO-MATH, our method retains only 40% of the training data per step yet matches or surpasses full-data baselines, with especially pronounced out-of-distribution gains on AIME25 (+45.9%) and AMC23 (+18.2%), alongside an estimated 36% reduction in training FLOPs. Our code is available at this https URL. 

---
# BoLT: A Benchmark to Democratize Black-box Optimization Research for Expensive LLM Tasks 

**Authors**: Ruth Wan Theng Chew, Zhiliang Chen, Apivich Hemachandra, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2605.17000)  

**Abstract**: Optimization of LLM training and inference configurations, such as hyperparameters, data mixtures, and prompts, is critical to performance, but it is often approached heuristically in practice, leading to potentially suboptimal outcomes. By framing them as noisy, expensive, and derivative-free optimization problems, Bayesian optimization (BO) and other black-box optimization (BBO) methods offer a promising yet underexplored direction for principled, sample-efficient methods. However, LLM training and inference costs are prohibitively high for most of the BBO research community, and new methods are often only evaluated on synthetic test functions and small-scale datasets that fail to capture the challenges of modern LLM optimization problems. This impedes the development of BBO methods and makes it difficult to assess their effectiveness on modern LLM tasks. We introduce BoLT, the first LLM-centric benchmark that democratizes LLM research for the BBO community. BoLT is released at this https URL. BoLT covers broad and well-motivated LLM optimization problems, involving multi-fidelity, multi-objective, heteroscedastic noise, and high-dimensional search spaces. Each problem in BoLT is grounded in real experimental data and made fully reproducible and accessible through lightweight surrogate models fitted to the results of thousands of real LLM experiments. We benchmark BoLT against an extensive range of BO and BBO methods, showing that selected BO methods consistently outperform others across tasks and highlighting gaps in existing BBO methods on LLM tasks, underscoring the need to modernize benchmarks for the BBO community. 

---
# Adversarial Fragility and Language Vulnerability in Clinical AI: A Systematic Audit of Diagnostic Collapse Under Imperceptible Perturbations and Cross-Lingual Drift in Low-Resource Healthcare Settings 

**Authors**: Anthonio Oladimeji Gabriel, Ahmad Rufai Yusuf  

**Link**: [PDF](https://arxiv.org/pdf/2605.16993)  

**Abstract**: Current clinical artificial intelligence (AI) systems are evaluated almost exclusively on clean, standardised, English-language inputs, conditions that do not reflect the realities of healthcare delivery in low-resource settings. This study presents the first systematic dual audit of two orthogonal safety vulnerabilities in clinical AI: adversarial image fragility and cross-lingual diagnostic drift. Using DenseNet121, the architecture underlying CheXNet, fine-tuned on the COVID-QU-Ex chest X-ray dataset (85,318 images; COVID-19, Non-COVID Pneumonia, Normal), we demonstrate that diagnostic accuracy collapses from 89.3% to 62.0% under a Fast Gradient Method (FGM) perturbation of epsilon=0.021, a magnitude imperceptible to the human eye. Standard defensive strategies including Gaussian smoothing and ensemble voting failed to restore clinical safety. In a parallel language fragility experiment, we tested Llama3.1:8b and NatLAS (N-ATLAS) on 20 COVID-19 clinical cases presented in Standard English, Nigerian Pidgin (Naija), and Yoruba-inflected English. Both models exhibited significant accuracy degradation: Llama3.1:8b dropped from 80.0% to 65.0% on Pidgin; NatLAS, an African-context model, collapsed from 85.0% to 55.0%, with diagnosis consistency falling to 50%. These findings establish a quantitative failure envelope for clinical AI under conditions representative of Primary Health Centre (PHC) deployment in Nigeria, and motivate urgent calls for adversarially hardened, linguistically inclusive clinical AI architectures. 

---
# Response-free item difficulty modelling for multiple-choice items with fine-tuned transformers: Component-wise representation and multi-task learning 

**Authors**: Jan Netík, Patrícia Martinková  

**Link**: [PDF](https://arxiv.org/pdf/2605.16991)  

**Abstract**: Response-free item difficulty modelling promises to reduce reliance on response-based calibration but is intrinsically difficult on reading-comprehension multiple-choice items, where difficulty depends on inferential demands across wording components. Whereas most existing approaches extract item-text features and pass them to a separate statistical or machine-learning model, we fine-tune transformer encoders end-to-end on the item wording, eliminating the manual feature engineering and preprocessing that discards information. Moreover, two extensions to this joint-encoding approach are proposed: a component-wise variant that encodes wording components separately through a shared encoder, and a multi-task variant that retains joint encoding and adds an auxiliary multiple-choice question answering objective on the shared encoder. Each method is evaluated under a Monte Carlo subsampling design at three training-set sizes on a held-out test set. We find that joint encoding is a viable end-to-end alternative to feature-engineering pipelines; while the component-wise variant shows no detectable benefit, consistent with self-attention already harvesting the cross-component signal, the multi-task variant delivers significant paired improvements in the smallest-sample regime. Transformer fine-tuning, especially if regularised by a suitable auxiliary task, recovers a substantial share of the wording-derivable signal at training-set sizes typical of applied measurement. The framework provides a customisable interface for psychometrically motivated extensions. 

---
# Skills on the Fly: Test-Time Adaptive Skill Synthesis for LLM Agents 

**Authors**: Jingxing Wang, Chenyu Zhou, Zhihui Fu, Jun Wang, Weiwen Liu, Weinan Zhang, Jianghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.16986)  

**Abstract**: LLM agents benefit from reusable skills, yet test-time tasks often require guidance more specific than a static skill library can provide. We propose \emph{SkillTTA}, a Test-Time Adaptive Skill Synthesis method that retrieves a small set of training trajectories relevant to the current task and synthesizes them into a temporary, task-specific textual skill. The solver model is kept fixed, so adaptation happens entirely through generated context rather than parameter updates. We evaluate the method on SpreadsheetBench, ALFWorld, and BigCodeBench. Compared with static trajectory-to-skill synthesis using GPT-5.5, task-specific skills improve SpreadsheetBench Pass@1 from 0.397 to 0.505 and BigCodeBench Pass@1 from 0.517 to 0.651. On ALFWorld, the method matches a heavier memory-learning baseline within four points of success rate while producing the shortest successful trajectories among reported methods. Ablations on SpreadsheetBench further show that synthesized skills outperform raw trajectory prompting, that top-$k$ retrieval should stay small, and that failed trajectories are especially useful because they expose recurring evaluator-facing mistakes. 

---
# Extending Pretrained 10-Second ECG Foundation Models to Longer Horizons 

**Authors**: Wei Tang, Jinpei Han, Kangning Cui, Mattia Carletti, Fredrik K. Gustafsson, Shreyank N Gowda, Patitapaban Palo, Anshul Thakur, Lei Clifton, Jean-michel Morel, Raymond H. Chan, David A. Clifton, Xiao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16975)  

**Abstract**: Electrocardiogram (ECG) foundation models pretrained on typical diagnostic 10-second ECG segments, have demonstrated strong transferability across a range of clinical applications. However, many real-world applications produce recordings that are typically longer, and are varied in duration during inference time. These 10-second models have no built-in way to combine information across time. Extending them to longer horizons introduces two challenges: structural incompatibilities arising from input-length disparities, and semantic challenges that limit meaningful temporal aggregation. We propose a parameter-efficient framework that extends pretrained ECG foundation models to longer and variable-length ECGs without retraining the backbone. Guided by a frozen pretrained 10-second model, we introduce a lightweight plug-in module that extends the model in two complementary ways: (i) structurally compatible long-sequence processing and (ii) semantically informed temporal modeling. Experiments on multiple long-horizon ECG tasks, datasets, and foundation model backbones demonstrate that our method enables robust long-horizon extension from pretrained snapshot models, consistently outperforming sliding-window and pooling-based baselines with strong parameter efficiency. 

---
# WhiteTesseract: Reframing the Interpretation of Cultural Heritage through XR and Conversational AI 

**Authors**: Jingjing Li, Zhi Liu, Xiyao Jin, Tatsuki Fushimi, Yoichi Ochiai  

**Link**: [PDF](https://arxiv.org/pdf/2605.16972)  

**Abstract**: Cultural heritage exhibitions often struggle to sustain attention and support reflective engagement. Physical exhibitions rely on fixed interpretive aids that lack adaptability to individual backgrounds or curiosity, and their effectiveness depends heavily on a visitor's Personal Context, prior knowledge, and cultural literacy. Meanwhile, digital exhibitions prioritize convenience and accessibility but risk weakening the Physical and Social Contexts that define embodied cultural experience.
WhiteTesseract addresses this gap by enabling in-situ interpretation through high-resolution XR and conversational AI. The system integrates spatial intelligence via artwork recognition to allow visitors to selectively reduce environmental distractions (via diminished reality) and engage in context-aware dialogue (via large language models). The goal is to preserve the richness of the physical and social environment while providing a flexible space for personal reflection, enhancing Personal Context without compromising physical authenticity.
We deployed the system in a Claude Monet exhibition and conducted a controlled user study with 26 participants. Quantitative results showed that WhiteTesseract modulation significantly increased average viewing duration from 35.3 to 98.3 seconds (p < 0.001). Analysis of 529 visitor-AI interactions revealed that 60% extended beyond factual queries to include analytical, emotional, and comparative inquiries. These findings demonstrate how XR and AI can enrich the physical exhibition experience by supporting deeper, more personalized engagement without displacing the embodied value of cultural heritage. We discuss technical and social constraints for real-world deployment and limitations of our controlled setting. 

---
# OmniVL-Guard Pro: A Tool-Augmented Agent for Omnibus Vision-Language Forensics 

**Authors**: Jinjie Shen, Zheng Huang, Yuchen Zhang, Yujiao Wu, Yaxiong Wang, Lechao Cheng, Shengeng Tang, Tianrui Hui, Nan Pu, Zhun Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2605.16962)  

**Abstract**: Existing vision-language forgery detection and grounding methods operate under a closed-world paradigm, assuming verification can be completed by the model alone. However, self-contained MLLMs are constrained by finite parametric knowledge, static training corpora, and limited perceptual resolution, creating a practical ceiling in dynamic open-world forensics -- particularly for real-time event verification requiring external clues and forgery segmentation demanding fine-grained scrutiny of local manipulations. To address these limitations, we shift from scaling up the self-contained model toward reaching beyond it. We propose \textbf{OmniVL-Guard Pro}, a tool-augmented agent that extends unified forensics from closed-world prediction to open-world clues-driven reasoning. OmniVL-Guard Pro integrates a tool environment spanning real-time event search, local cropping and zooming, edge-anomaly screening, face detection, video frame extraction, and SAM3-based segmentation. To generate high-quality tool-reasoning trajectories, we introduce \textbf{Tree-Structured Self-Evolving Tool Trajectory Generation}, which produces diverse trajectories through seed guidance, guider-free self-evolution, and weakly-hinted hard sample synthesis, yielding the Full-Spectrum Tool Reasoning (FSTR) dataset for training. We further propose \textbf{Checker-Guided Agentic Reinforcement Learning} (CGARL), which provides process-level supervision to penalize cases where the answer is correct but the reasoning is distorted. Extensive experiments demonstrate that OmniVL-Guard Pro achieves state-of-the-art performance across various tasks, and exhibits strong zero-shot generalization. The FSTR dataset and code for OmniVL-Guard Pro will be publicly released at \url{this https URL}. 

---
# Latent Action Control for Reasoning-Guided Unified Image Generation 

**Authors**: Fuxiang Zhai, Sixiang Chen, Yingjin Li, Shuaibo Li, Jianyu Lai, Tengjun Huang, Lei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16961)  

**Abstract**: Unified multimodal models can encode visual understanding and image generation within a shared backbone, yet understanding does not automatically translate into control: models may infer objects, relations, or knowledge cues but fail to instantiate them in the generated image. We propose Latent Action Control (LAC), which makes reasoning actionable by representing it as hidden continuous actions inside a unified generator. Given a prompt, LAC rolls out a role-structured latent trajectory for planning, internal visual drafting, diagnosis, and refinement, and injects these actions into the hidden stream that conditions flow-based generation, without producing reasoning tokens or intermediate images. Since such action trajectories are unobserved, LAC learns them through prior-guided variational latent action alignment from training-only rendered semantic priors, draft image features, and supervised halting signals, followed by Latent-Flow GRPO to align the latent-to-image rollout with terminal visual feedback. This provides a control path from inferred relations, bindings, and knowledge cues to the generation process. Instantiated on BAGEL-7B-MoT, LAC consistently improves compositional and knowledge-grounded generation across GenEval, WISE, and T2I-CompBench, with the largest gains on spatial relations, attribute binding, and world-knowledge-sensitive prompts. Ablations and latent interventions show that the learned action trajectory is consumed by the generator, suggesting that unified generation benefits when understanding is not only encoded, but made actionable during generation. 

---
# Effort as Ceiling, Not Dial: Reasoning Budget Does Not Modulate Cognitive Cost Alignment Between Humans and Large Reasoning Models 

**Authors**: Yueqing Hu, Tianhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16938)  

**Abstract**: Large Reasoning Models (LRMs) generate chain-of-thought traces whose length tracks human reaction times across cognitive tasks, but recent debate questions whether this alignment reflects genuine computational structure or surface verbosity. We test whether the alignment varies with inference-time reasoning effort. Across GPT-OSS-20B and GPT-OSS-120B, three effort levels, and six reasoning tasks, within-task and cross-task alignment remain invariant: Bayes Factors lean toward the null, and mean alignment is numerically near-identical across conditions. A manipulation check reveals that the effort parameter sets an upper budget on generation rather than driving real-time allocation, suggesting that the allocation policy is crystallized at training time. Arithmetic complexity contrasts further show that token allocation tracks fine-grained, format-dependent human difficulty patterns, with model scale improving the match. Cognitive cost alignment between LRMs and humans appears to be a training-time achievement, robust to inference-time perturbations, supporting a compiled rather than online account of LRM problem-solving. 

---
# Full Attention Strikes Back: Transferring Full Attention into Sparse within Hundred Training Steps 

**Authors**: Yanke Zhou, Yiduo Li, Hanlin Tang, Maohua Li, Kan Liu, Lan Tao, Lin Qu, Yuan Yao, Xiaoxing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.16928)  

**Abstract**: Long-context inference in large language models is bottlenecked by the quadratic cost of full attention. Existing efficient alternatives often rely either on native sparse training or on heuristic token eviction, creating an undesirable trade-off among efficiency, training cost, and accuracy. In this work, we show that full-attention LLMs are already intrinsically sparse and can be transformed into highly sparse models with only minimal adaptation. Our approach is built on three observations: (1) only a small subset of attention heads truly requires full long-context processing; (2) long-range retrieval is governed primarily by a low-dimensional subspace, allowing relevant tokens to be retrieved efficiently with a 16-dimensional indexer; and (3) the useful token budget is strongly query-dependent, making dynamic top-$p$ selection more suitable than fixed top-$k$ sparsification. Based on these insights, we propose RTPurbo, which retains the full KV cache only for retrieval heads and introduces a lightweight token indexer for sparse attention. By exploiting the model's intrinsic sparsity, RTPurbo achieves sparsification with only a few hundred training steps. Experiments on long-context benchmarks and reasoning tasks show that RTPurbo preserves near-lossless accuracy while delivering substantial efficiency gains, including up to a 9.36$\times$ prefill speedup at 1M context and about a 2.01$\times$ decode speedup. These results suggest that strong sparse inference can be obtained from standard full-attention training without expensive native sparse pretraining. 

---
# The Alpha Illusion: Reported Alpha from LLM Trading Agents Should Not Be Treated as Deployment Evidence 

**Authors**: Yuxuan Ye, Jun Han, Ao Hu, Juncheng Bu, Yiyi Chen, Liangjian Wen, Danilo Mandic, Danny Dongning Sun, Xu Yinghui, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16895)  

**Abstract**: End-to-end LLM trading agents have moved quickly from research curiosity to a small ecosystem of named systems, including FinCon, FinMem, TradingAgents, FinAgent, QuantAgent, and FLAG-Trader. Several of these report headline Sharpe ratios that would be material if read at face value on a deployment desk, and associated benchmarks such as FinBen report trading-task Sharpe statistics in the same range. The gap between architecture research and deployment claim has been crossed too freely on both sides of the academia--industry divide. We take a position on that gap: reported alpha from end-to-end LLM trading agents should not be treated as deployment evidence. Before such returns can support claims of deployable trading capability, they must survive structural validity tests for temporal integrity, real-world frictions, counterfactual robustness, predictive calibration, numerical execution, and multi-agent disaggregation. Current public evidence cannot yet distinguish robust predictive ability from temporal contamination, unmodeled frictions, short-window Sharpe uncertainty, narrative fitting, and parametric priors. The problem is not only evaluative but structural. Language confidence is not tradable probability, narrative reasoning is not numerical execution, and model priors may become undisclosed implicit factor exposures. We contribute a minimum reporting protocol suite, P1--P6, with tiered applicability by claim strength, and a conservative modular alternative that uses LLMs as auditable information interfaces upstream of independent calibration, risk, and execution modules. Code and reproduction harness: \url{this https URL}. 

---
# DriveSafe: A Framework for Risk Detection and Safety Suggestions in Driving Scenarios 

**Authors**: Sainithin Artham, Shankar Gangisetty, Avijit Dasgupta, C. V. Jawahar  

**Link**: [PDF](https://arxiv.org/pdf/2605.16892)  

**Abstract**: Comprehensive situational awareness is essential for autonomous vehicles operating in safety-critical environments, as it enables the identification and mitigation of potential risks. Although recent Multimodal Large Language Models (MLLMs) have shown promise on general vision-language tasks, our findings indicate that zero-shot MLLMs still underperform compared to domain-specific methods in fine-grained, spatially grounded risk assessment. To address this gap, we propose DriveSafe, a framework for risk-aware scene understanding that leverages structured natural language descriptions. Specifically, our method first generates spatially grounded captions enriched with multimodal context, including motion, spatial, and depth cues. These captions are then used for downstream risk assessment, explicitly identifying hazardous objects, their locations, and the unsafe behaviors they imply, followed by actionable safety suggestions. To further improve performance, we employ caption-risk pairings to fine-tune a lightweight adapter module, efficiently injecting domain-specific knowledge into the base LLM. By conditioning risk assessment on explicit language-based scene representations, DriveSafe achieves significant gains over both zero-shot MLLMs and prior domain-specific baselines. Exhaustive experiments on the DRAMA benchmark demonstrate state-of-the-art performance, while ablation studies validate the effectiveness of our key design choices. Project page: this https URL research/projects/cvit-projects/drivesafe 

---
# Some[Body] Must Receive That Pain for Agent Accountability 

**Authors**: Botao Amber Hu, Helena Rong  

**Link**: [PDF](https://arxiv.org/pdf/2605.16872)  

**Abstract**: AI agents increasingly act consequentially in the real world. This creates a problem we call \emph{consequence reception}: harm occurs, the producing system is identified, yet no continuing agent receives consequences in a way that changes future behavior. Pain, understood mechanistically as a corrective feedback signal, is foundational to canonical theories of punishment -- deterrence, rehabilitation, retribution, and incapacitation all assume a continuing locus that registers the signal and updates behavior. That, in turn, requires a body for the signal to land on: a boundary whose integrity it protects, a locus where it accumulates, consolidation that converts episodic signal into durable update, and a substrate that responds by altering future action. Current LLM agents -- software-defined composites of weights, prompts, tools, memory, and credentials, freely swapped, copied, reset, and reassembled -- satisfy none of these conditions. The two prevailing legal responses therefore fail to achieve consequence reception. The thin-identity agent-principal dyad has a body but no \emph{consequence--agency coupling}: the human bears pain for behaviors beyond their control -- Elish's \emph{moral crumple zone}. The thick-identity Arbel et al.'s \emph{Algorithmic Corporation} creates legally legible entities but does not guarantee that any AI decision architecture receives pain as a behavioral signal. Achieving consequence-agency coupling is therefore a sociotechnical infrastructural problem, not only a legal one. Until such architectures exist, high-stakes AI deployment should remain tethered to accountable human principals with meaningful control, proportional liability, and authority to constrain or terminate the agent. \emph{If some body does not receive the pain by design, some body will receive it by default.} 

---
# Metric-Guided Feature Fusion of Visual Foundation Models for Segmentation Tasks 

**Authors**: Yachan Guo, JoseLuis Gomez Zurita, Danna Xue, Yi Xiao, AntonioManuel Lopez Pena  

**Link**: [PDF](https://arxiv.org/pdf/2605.16864)  

**Abstract**: Although large-scale visual foundation models (VFMs) achieve remarkable performance in semantic understanding, they still underperform in instance-aware dense prediction tasks. They exhibit different biases in representation: for instance, promptable segmentation models (e.g., SAM2) focus on fine-grained region boundaries, while self-supervised models (e.g., DINOv3) emphasize object-level structure. This observation highlights the potential of combining complementary features from different VFMs to enhance downstream dense prediction tasks. However, naive multi-VFM fusion seldom leads to reliable gains, and interpretable principles for leveraging their complementary features are still underexplored. In this work, we propose a metric-guided approach that effectively selects and aggregates complementary features from different VFMs based on explicit assessment scores. Specifically, we design a suite of label-free metrics in feature space across two aspects, Structural Coherence and Edge Fidelity, to assess features of VFM encoders. Guided by these scores, we identify complementary edge-strong and structure-strong encoder pairs, and integrate them via a master-auxiliary fusion scheme. This feature fusion requires no complex architectural changes and is trained only in a single stage. Our model shows consistent performance gains across multiple dense prediction tasks compared with the baselines, with better object-level semantics and more accurately localized boundaries. The code is available at {this https URL}. 

---
# Plan First, Diffuse Later: Extrinsic Graph Guidance for Long-Horizon Diffusion Planning 

**Authors**: Yaniv Hassidof, Adir Morgan, Yilun Du, Kiril Solovey  

**Link**: [PDF](https://arxiv.org/pdf/2605.16863)  

**Abstract**: Compositional diffusion models offer a promising route to long-horizon planning by denoising multiple overlapping sub-trajectories while ensuring that together they constitute a global solution. However, enforcing local behavior over long chains is often insufficient for a coherent global structure to emerge. Recent works tackle this limitation through intrinsic search, which explores multiple paths during the denoising process. While intrinsic search improves global coherence, it comes at the cost of repeated evaluations of an already compute-heavy model. In this work, we argue that extrinsic search, performed outside the denoising process, offers a more effective mode of exploration for long-horizon planning while naturally enabling the use of classical algorithms to solve unseen combinatorial tasks at test time. Our eXtrinsic search-guided Diffuser (XDiffuser) first computes a plan over a state-space graph -- serving as a lightweight local connectivity oracle for the diffusion model. The plan is then used to guide denoising for a single trajectory, effectively offloading the burden of exploration. XDiffuser outperforms diffusion-based baselines on long-horizon tasks, with particularly large gains in the low-quality data regime and on unseen tasks beyond goal-reaching, including multi-agent coordination and TSP-style reasoning. Project website: this https URL 

---
# Prefix-Adaptive Block Diffusion for Efficient Document Recognition 

**Authors**: Mingxu Chai, Ziyu Shen, Chenyu Liu, Kaidi Zhang, Jiazheng Zhang, Dingwei Zhu, Zhiheng Xi, Ruoyu Chen, Jun Long, Jihua Kang, Tao Gui, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16861)  

**Abstract**: Block Diffusion Models (BDMs) support parallel generation, flexible-length output, and KV caching, making them promising for efficient document parsing. However, existing BDMs bind denoising and cache commitment to fixed block boundaries: parallelism shrinks during intra-block denoising, while generated tokens cannot be cached until the whole block is completed. Moreover, intra-block bidirectional denoising conflicts with inter-block autoregression, creating inconsistent information flow that can challenge structure-sensitive recognition. We propose the Prefix-Adaptive Block Diffusion Model (PA-BDM), which replaces intra-block bidirectional denoising with causal denoising from prefix to suffix and treats the block size as a maximum candidate range rather than a fixed commitment unit. PA-BDM uses Confidence-gated Structural Loss (CSL) to build low-entropy prefixes before extending training to longer continuations. During inference, Progressive Prefix Commitment (PPC) then dynamically commits the longest reliable prefix into the KV cache and resets the next candidate range from the updated prefix, restoring a large parallel decoding space at each step. Experiments show that the 3B PA-BDM achieves higher recognition scores on several benchmarks and improves inference throughput by 71.6\% over the 2.5B MinerU-Diffusion. 

---
# PhysioSeq2Seq: A Hybrid Physiological Digital Twin and Sequence-to-Sequence LSTM for Long-Horizon Glucose Forecasting in Type 1 Diabetes 

**Authors**: Phat Tran, Neville Mehta, Clara Mosquera-Lopez, Robert H. Dodier, Lizhong Chen, Peter G. Jacobs  

**Link**: [PDF](https://arxiv.org/pdf/2605.16860)  

**Abstract**: Accurate long-horizon glucose forecasting is critical for automated insulin delivery systems, which help people with type 1 diabetes (T1D) manage their glucose and avoid dangerous hypoglycemia. However, standard recursive long short-term memory (LSTM) networks suffer from systematic negative bias at longer horizons due to error compounding, while purely mechanistic ordinary differential equation (ODE) models fail to generalize across individuals when parameterized at the population level. We propose PhysioSeq2Seq, a hybrid architecture that combines patient-specific physiological modeling with a sequence-to-sequence (Seq2Seq) LSTM. For each glucose segment, twin matching searches a population of 300 parameterized digital twins to identify the best-fitting physiological match from a 3-hour continuous glucose monitoring (CGM) history. The 10 internal ODE state variables of the matched twin are injected as exogenous covariates into both the encoder and decoder of the Seq2Seq LSTM. This simultaneous 48-step prediction strategy eliminates recursive error compounding, while the ODE features provide a physics-grounded constraint that bounds long-horizon drift within physiologically plausible ranges. PhysioSeq2Seq was trained on CGM and insulin data from 348 participants in the Type 1 Diabetes Exercise Initiative (T1DEXI) dataset and evaluated on 74 held-out participants. At the 240-minute horizon, PhysioSeq2Seq achieves a mean absolute error of 39.28 mg/dL and a mean error of -10.62 mg/dL, reducing bias by 13.89 mg/dL over the recursive LSTM and reducing mean absolute error by 28.62 mg/dL over the ODE-based digital twin. These results show that eliminating architectural feedback and injecting patient-matched physiological states is an effective and clinically meaningful strategy for long-horizon glucose forecasting in T1D. 

---
# VGGT-CD: Training-Free Robust Registration for 3D Change Detection 

**Authors**: Wei Zhang, Songhua Li, Yihang Wu, Qiang Li, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16859)  

**Abstract**: 3D change detection from multi-view images is essential for urban monitoring, disaster assessment, and autonomous driving. However, existing methods predominantly operate in the 2D domain, where viewpoint variations are mistaken for physical changes and depth is unavailable. While visual geometry foundation models like VGGT rapidly produce dense point clouds from unposed images, independent per-epoch reconstruction encounters fundamental obstacles: unpredictable inter-epoch scale ambiguity, registration-change paradox where scene changes corrupt alignment, and pervasive edge-flying noise. To address these challenges, we present VGGT-CD, a training-free pipeline decoupling cross-temporal registration from dynamic-change interference. In the Coarse Stage, sparse keyframe joint inference establishes a unified metric space and yields an initial Sim(3) prior. In the Fine Stage, dense reconstructions are purified by isolating static-background correspondences. A closed-form centroid alignment refines the translation while locking scale and rotation, using a residual self-check to mathematically guarantee non-degradation. Evaluated on an 11-scene benchmark from the World Across Time dataset, VGGT-CD reduces Absolute Trajectory Error by 44% outdoors and 59% indoors. It completes registration over 6 times faster, producing high-purity 3D change maps without task-specific training. 

---
# Pedestrian-Aware LLM-Driven Behavioral Planning for Autonomous Vehicles 

**Authors**: Aidana Baimbetova, Haruki Yonekura, Hamada Rizk, Hirozumi Yamaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2605.16858)  

**Abstract**: Autonomous Vehicles (AVs) must make reliable decisions in dense urban environments where pedestrian behavior is variable, sometimes abnormal, and often unseen during training. Reinforcement learning (RL)-based AV control systems perform well in structured traffic but struggle to generalize to unpredictable pedestrian interactions and out-of-distribution scenarios. Their reliance on handcrafted rewards and opaque decisions further limits their suitability for safety-critical, pedestrian-rich environments. To address these limitations, we introduce a Large Language Model (LLM)-based decision-making framework for pedestrian-aware behavioral planning. The system converts structured scene observations into natural-language reasoning prompts, enabling the LLM to infer pedestrian intent, anticipate risk, and generate cautious tactical driving decisions. These decisions are executed by a motion planner that ensures smooth, kinematically feasible control. We evaluate the framework in SUMO across multiple pedestrian-interaction scenarios, including unexpected jaywalking, turn-back crossing, hesitation, and bidirectional crossing. In zero-shot evaluation, the LLM-based agent achieves a 68% collision-free success rate, substantially outperforming deep RL baselines (17.7%). With few-shot episodic memory in a single-pedestrian scenario, performance increases to 96.0%, exceeding a custom DQN controller (82.0%). Cross-behavior evaluation further shows that memory derived from turn-back interactions transfers to unseen hesitation and bidirectional crossing scenarios, achieving 82.0% and 90.0% success, respectively. The system consistently initiates earlier responses, maintains wider safety buffers, and produces interpretable, human-aligned decisions. 

---
# Thinking with Patterns: Breaking the Perceptual Bottleneck in Visual Planning via Pattern Induction 

**Authors**: Yichang Jian, Boyuan Xiao, Zhenyuan Huang, Yifei Peng, Yao-Xiang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.16848)  

**Abstract**: Planning from raw visual input remains a significant challenge for current Vision-Language Models (VLMs), when the complexity of input is beyond their one-step perception capability. Motivated by recent advances in Thinking with Images (TWI), a reasonable solution is to decompose the perception process into simpler steps by iteratively acquiring and incorporating local visual evidence. However, even though current VLMs are well-trained in general TWI ability, their perceptual bottleneck in the planning domain remains. To tackle this challenge, we formulate TWI as a tool to gradually build and reflect an accurate internal world model. We find that the resulting training-free planning strategy enables VLMs to solve tasks that are far beyond their initial capabilities, at the cost that too many TWI operations would significantly increase the computational overhead. To further improve efficiency, we propose Pattern Inference, a novel TWI strategy enabling VLMs to actively recognize known visual patterns in the new tasks and directly infer local world model structures. To obtain these patterns, we propose Pattern Induction, an online inductive learning strategy treating visual patterns as composite and reusable experts, which are autonomously discovered and optimized from experience. Experimental evaluations in FrozenLake, Crafter and CubeBench domains show that our approaches achieve a desirable balance between accuracy and efficiency. 

---
# Learning Relative Representations for Fine-Grained Multimodal Alignment with Limited Data 

**Authors**: Shiwon Kim, Yu Rang Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.16834)  

**Abstract**: Multimodal pre-training demonstrates strong generalization performance, but this paradigm is often impractical in domains where paired data are scarce. A promising alternative is post-hoc multimodal alignment, which aligns separately pre-trained unimodal encoders using a limited number of paired examples. However, existing methods focus primarily on aligning global representations, missing patch-token relations. This may hinder transfer to tasks that require fine-grained cross-modal matching beyond coarse sample-level semantics. To address this issue, we propose a post-hoc alignment method that learns token-level cross-modal structure using relative representations. Specifically, we represent images and texts through their token-level similarities to a set of learnable anchors in each modality space, which are trained to induce consistent cross-modal similarity patterns for matched pairs. Despite learning only the anchors without heavy projection layers, our approach consistently outperforms existing methods in zero-shot classification, cross-modal retrieval, and zero-shot segmentation by a substantial margin. This highlights the importance of modeling fine-grained cross-modal structure for effective post-hoc multimodal alignment with limited paired data. 

---
# Prediction-Intervention Games and Invariant Sets 

**Authors**: Linus Kühne, Felix Schur, Jonas Peters  

**Link**: [PDF](https://arxiv.org/pdf/2605.16828)  

**Abstract**: We consider the following two-player game: using observational data, the leader chooses a prediction function for a response variable $Y$ from given covariates. The follower then reacts with an intervention on some covariates in the underlying structural causal model to maximize their own objective. The leader knows the intervention targets, but may have limited knowledge of the follower's objective. We call this setup a prediction-intervention game, a special case of a Stackelberg game. Finding an optimal strategy for the leader is generally difficult. To avoid severe performance loss, the leader may base their prediction on the causal parents of $Y$, or more generally on an invariant subset of covariates. We prove, for two common classes of follower objectives, that predictors based on the stable blanket, a specific invariant subset, are always better or as good as those based on the causal parents. We further upper bound the leader's post-intervention risk by a worst-case risk over allowed interventions and strengthen existing distribution generalization results to analyze this bound: we give sufficient conditions under which stable-blanket predictors are worst-case optimal, and show by examples that these conditions cannot in general be dropped. Finally, we discuss practical strategies for settings with known and unknown graph, and test them on simulated and real-world data. 

---
# Decoupling KL and Trajectories: A Unified Perspective for SFT, DAgger, Offline RL, and OPD in LLM Distillation 

**Authors**: Anhao Zhao, Haoran Xin, Yingqi Fan, Junlong Tong, Wenjie Li, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.16826)  

**Abstract**: Knowledge distillation is central to LLM post-training, yet its design space remains poorly understood, especially alongside reinforcement learning (RL). We show that the prevailing paradigms, off-policy distillation and on-policy distillation (OPD), implicitly couple two orthogonal choices: prefix source and token-level KL direction. This follows from decomposing sequence-level KL over autoregressive response distributions: forward KL pairs teacher prefixes with token-level forward KL, and reverse KL pairs student prefixes with token-level reverse KL. We argue this coupling is not intrinsic: decoupling the two axes yields four valid objectives. We establish gradient-level identities showing forward KL gives SFT-style cross-entropy matching with teacher soft targets, whereas reverse KL gives an RL-style policy-gradient objective with a dense teacher-student log-ratio reward, connecting them to off-policy SFT, DAgger-style on-policy SFT, offline-RL-style distillation, and OPD. We conduct an extensive controlled study on math reasoning, evaluating the four objectives both as standalone methods and as initializations for subsequent RL. The results reveal three tradeoffs: KL direction induces an accuracy-entropy tradeoff, prefix source a quality-compute tradeoff, and training length an accuracy-stability tradeoff. Motivated by these findings, we propose KL mixing and an entropy-gated length curriculum. KL mixing shows long-sequence distillation requires substantial forward-KL weight to prevent entropy collapse and length inflation without sacrificing accuracy. The entropy-gated length curriculum improves Avg@k and Pass@k by 3.6 and up to 5.8 points, and cuts average response length by roughly 3x versus fixed long-horizon training. Our results provide a framework and practical methods for designing reasoning distillation objectives that balance accuracy, diversity, compute, and RL behavior. 

---
# Echoes in Filter Bubble: Diagnosing and Curing Popularity Bias in Generative Recommenders 

**Authors**: Jun Yin, Bangguo Zhu, Peng Huo, Ruochen Liu, Hao Chen, Senzhang Wang, Shirui Pan, Chengqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16825)  

**Abstract**: Recently, Generative Recommenders (GRs), characterized by a unified end-to-end framework, have exhibited astonishing potential in transforming the recommendation paradigm. Despite their effectiveness, we recognize that GRs are still susceptible to the long-standing issue of popularity bias that has pervaded the recommendation community. Although a few studies have attempted to extend traditional debiasing methods to GRs, their effectiveness is marginal, and the fundamental reason why GRs suffer from popularity bias remains under-explored. To bridge this gap, this study focuses on two core aspects in GRs: the optimization of generative framework and the item tokenization based on semantic index. Based on theoretical analyses, we identify that the severe popularity bias emerges from the confluence of a token-level optimization flaw and the undifferentiated property of item tokenization. Accordingly, this study develops a novel generative recommender system, called Ghost, by designing the asymmetric unlikelihood optimization and the skeleton-founded tokenization. Extensive empirical evaluations across three datasets, alongside multiple SOTA baselines, reveal that Ghost substantially alleviates popularity bias and promotes fairer recommendations, while incurring slight degradation to the overall recommendation utility. 

---
# AgentKernelArena: Generalization-Aware Benchmarking of GPU Kernel Optimization Agents 

**Authors**: Sharareh Younesian, Wenwen Ouyang, Sina Rafati, Mehdi Rezagholizadeh, Sharon Zhou, Ji Liu, Yue Liu, Yuchen Yang, Hao Li, Ziqiong Liu, Dong Li, Vikram Appia, Zhenyu Gu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2605.16819)  

**Abstract**: GPU kernel optimization is increasingly critical for efficient deep learning systems, but writing high-performance kernels still requires substantial low-level expertise. Recent AI coding agents can iteratively read code, invoke compilers and profilers, and refine implementations, yet existing kernel benchmarks evaluate single LLM calls rather than full agent workflows, and none include both kernel-to-kernel optimization and unseen-configuration generalization testing. We present AgentKernelArena, an open-source benchmark for measuring AI coding agents on GPU kernel optimization. The benchmark contains 196 tasks spanning HIP-to-HIP optimization, Triton-to-Triton optimization, and PyTorch-to-HIP translation, and evaluates complete agent workflows in isolated workspaces using gated compilation, correctness, and performance checks, centralized scoring and an unseen-configuration generalization protocol that tests whether optimizations transfer to input configurations the agent never observed. Across production agents including Cursor Agent, Claude Code, and Codex Agent, we find near-perfect compilation and high correctness rates on most task categories, with the strongest configurations achieving mean speedups of up to 6.89x on PyTorch-to-HIP, 6.69x on HIP-to-HIP, and 2.13x on Triton-to-Triton tasks. Our unseen-configuration evaluation shows that HIP-to-HIP and Triton-to-Triton optimizations largely transfer to unseen input shapes, while PyTorch-to-HIP exhibits substantial correctness drops, indicating that agents generating kernels from scratch frequently hardcode shape-specific assumptions. AgentKernelArena is designed as a modular, extensible framework for rigorous evaluation of agentic GPU kernel optimization across agents, tasks, and hardware targets. 

---
# Observation-Aligned Mask Priors for Learning Physical Dynamics from Authentic Occlusions 

**Authors**: Chiyuan Ma, Zihan Zhou, Tianshu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16818)  

**Abstract**: Learning physical dynamics directly from incomplete observations is challenging because authentic occlusions are structured, sample-dependent, and often missing not at random, whereas existing methods typically rely on heuristic masking rules or predefined mask distributions. We propose Observation-Aligned Mask Priors, a framework that learns the distribution of authentic observation masks and uses it to construct context-query partitions for training from incomplete data. Specifically, we pretrain a Bayesian Flow Network (BFN) on binary observation masks to capture real occlusion topologies, then guide BFN sampling with a globally normalized cross-entropy objective to generate sample-specific masks aligned with each sparse observation. The intersection between the guided mask and the observed mask defines the context, and the remaining observed entries become query targets for a diffusion-based reconstruction model. We show that this intersection-based partitioning gives every valid observed dimension a strictly positive probability of being queried, preventing zero-query dead zones and local generative collapse. Experiments on three real-world oceanographic datasets with authentic satellite occlusions, across resolutions up to 256$\times$256, show consistent improvements over strong diffusion baselines in MSE and PSNR. These results demonstrate that learning mask priors from authentic occlusions is an effective alternative to heuristic masking for learning from incomplete physical observations without access to fully observed fields. 

---
# Cross-modal Affinity-aligned Multimodal Learning Analytics for Predicting Student Collaboration Satisfaction in Game-Based Learning 

**Authors**: Wen-Hsin Tsai, Chia-Ming Lee, Yuk-Ying Tung  

**Link**: [PDF](https://arxiv.org/pdf/2605.16806)  

**Abstract**: Collaborative game-based learning environments offer rich opportunities for small-group knowledge construction, yet automatically predicting student collaboration satisfaction remains challenging. A critical barrier is modality degradation: in educational deployments, individual modalities such as eye gaze exhibit inconsistent informativeness across student cohorts, causing implicit attention-based fusion to produce brittle multimodal representations. We propose the Affinity-Aligned Multimodal Learning Analytics (AAMLA) framework, whose core contribution is the Cross-modal Affinity-guided Modality Alignment (CAMA) module, which explicitly models inter-modal relationships via affinity matrices and enforces cross-modal consistency through contrastive learning, enabling adaptive suppression of uninformative modalities without discarding them. AAMLA further applies modality-specific projection layers to map heterogeneous features, including facial action units, head pose, eye gaze, and interaction trace logs, into a unified semantic space prior to alignment. Experiments on 50 middle school students in the EcoJourneys collaborative learning environment demonstrate consistent improvements over unimodal baselines and prior cross-attention approaches under standard and modality degradation conditions, with SHAP and t-SNE analyses confirming that CAMA produces robust, interpretable cross-modal representations for student collaboration modeling. 

---
# Cross-Domain Molecular Relational Learning: Leveraging Chemical Structure-Activity Analysis 

**Authors**: Peiliang Zhang, Jingling Yuan, Shiqing Wu, Mengqing Hu, Chao Che, Yongjun Zhu, Lin Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.16799)  

**Abstract**: Recent advances in molecular representation integrates molecular topological and visual modalities, opening new avenues for precise Molecular Relational Learning (MRL). Existing MRL methods focus on intra-domain modeling, and their inherent domain-closed effect limits applicability to molecular science, particularly in elucidating cross-domain interaction mechanisms. Consequently, the imperative for Cross-Domain Molecular Relational Learning has become increasingly pressing. Benefiting from structure-activity analysis, we propose the Domain Adversarial Training Network with Structural-Semantic Transfer Discrepancy (DisTrans) to optimize cross-domain adaptive representation for molecular structures and visual images. 1) We employ the gradient reversal strategy based on substructure topological discrepancies between domains to learn the domain dependence of molecular structures. This strategy guides the model to adapt to the structural adjacency patterns in the target domain, generating domain-separable structural representations. 2) We apply the cross-domain representation guidance mechanism to align the functional-group semantic information between the source and target domains, learning cross-domain consistency information. The experimental results in two typical cross-domain strategies demonstrate that DisTrans outperforms 16 baseline methods, maintaining satisfactory performance even under pronounced inter-domain discrepancy. 

---
# 3DPhysVideo: Consistency-Guided Flow SDE for Video Generation via 3D Scene Reconstruction and Physical Simulation 

**Authors**: Hwidong Kim, Yunho Kim, Tae-Kyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.16795)  

**Abstract**: Video generative models have made remarkable progress, yet they often yield visual artifacts that violate grounding in physical dynamics. Recent works such as PhysGen3D tackle single image-to-3D physics through mesh reconstruction and Physically-Based Rendering, but challenges remain in modeling fluid dynamics, multi-object interactions and photorealism. This work introduces 3DPhysVideo, a novel training-free pipeline that generates physically realistic videos from a single image. We repurpose an off-the-shelf video model for two stages. First, we use it as a novel view synthesizer to reconstruct complete 360-degree 3D scene geometry by guiding the image-to-video (I2V) flow model with rendered point clouds. Second, after applying physics solvers to this geometry, the physically simulated point cloud is used to guide the same I2V flow model to synthesize final, high-quality videos. Consistency-Guided Flow SDE, which decomposes the predicted velocity of the I2V flow model into denoising and consistency bias, enforces consistency to the conditional inputs, allowing us to effectively repurpose the model for both 3D reconstruction and simulation-guided video generation. In the diverse experiments including multi-objects, and fluid interaction scenes, our method successfully bridges the gap from single-images to physically plausible videos, while remaining efficient to run on a single consumer GPU. It outperforms state-of-the-art baselines on GPT-based scores, VideoPhy benchmark and human evaluation. 

---
# TIER: Trajectory-Invariant Execution Rewards for Multi-Step Tool Composition 

**Authors**: Anay Kulkarni, ChiaEn Lu, Dheeraj Mekala, Jayanth Srinivasa, Gaowen Liu, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16790)  

**Abstract**: Tool use enables large language models to solve complex tasks through sequences of API calls, yet existing reinforcement learning approaches fail to scale to multi-step composition settings. Outcome-based rewards provide only sparse feedback, while trajectory-supervised rewards depend on annotated reference solutions, penalizing valid alternatives and limiting scalability. We propose TIER: Trajectory-Invariant Execution Rewards, a reward framework that derives supervision directly from function schemas and runtime execution, rather than from reference trajectories. The reward decomposes into format validity, schema adherence, execution success, and answer correctness, providing dense, interpretable sequence-level feedback derived from fine-grained verification of individual steps of tool use. This design allows any valid execution path to receive credit, naturally supporting multiple solution strategies and adapting to evolving tool interfaces. On DepthBench, a compositional benchmark stratified by depth (1 to 6 steps), TIER achieves >90% accuracy across steps, where trajectory-supervised rewards collapse beyond step-4. We further demonstrate consistent gains on benchmarks like BFCL v3 and NestFUL. Ablation studies confirm that all reward components are necessary, highlighting the importance of multi-level supervision for compositional reasoning. 

---
# Encoding Robust Topological Signatures for Hyperdimensional Computing 

**Authors**: Arpan Kusari  

**Link**: [PDF](https://arxiv.org/pdf/2605.16785)  

**Abstract**: Hyperdimensional (HD) computing offers an attractive alternative to deep networks for edge learning due to its simplicity, fast prototype-based inference, and compatibility with online updates. However, standard pixel-based HD encoders are brittle: small distribution shifts such as rotation, noise, or occlusion can drastically reduce accuracy. We extract discrete topological primitives-most notably holes-from binarized shapes and pair them with rotation/translation/scale (RTS)-invariant shape signatures. Our method constructs RTS-stable descriptors for (i) the outer shape using a spatial-pyramid variant of Zernike moments and (ii) each hole using an intrinsic Fourier descriptor of its radial signature together with RTS-canonical relative geometry. Each primitive is mapped to a bipolar hypervector via randomized projection and role binding, and variable-cardinality hole sets are aggregated by permutation-invariant bundling to form a single image hypervector. To avoid over-weighting any cue, we learn nonnegative reliability weights for the Zernike and hole channels on a validation set via late fusion of cosine similarities. Experiments on MNIST and EMNIST under controlled corruptions (rotation, Gaussian noise, salt-and-pepper, cutout, zoom) show that Topology-guided HD computing substantially improves robustness compared with a naive HD baseline, maintaining high accuracy across multiple corruption families and benefiting from lightweight online training. Compared with a compact CNN trained on clean data, our method achieves competitive clean accuracy while offering markedly stronger robustness to several pixel-level corruptions, demonstrating that explicit topological structure is a practical route to robust HD representations. The code is provided at this https URL. 

---
# A Holistic Method for Superquadric Fitting Using Unsupervised Clustering Analysis 

**Authors**: Mingyang Zhao, Sipu Ruan, Xiaohong Jia  

**Link**: [PDF](https://arxiv.org/pdf/2605.16779)  

**Abstract**: This work presents a novel method for fitting superquadrics to point clouds under the contamination of noise and outliers, which has many applications for shape modeling across diverse fields. Unlike prior approaches that either exclusively focus on fitting rigid or deformable superquadrics, or suffer from robustness and numerical instability issues, our method redefines the problem from a new unsupervised clustering perspective, enabling the holistic fitting of both rigid and deformable superquadrics within a unified framework. Central to our approach is a stable optimization function inspired by unsupervised clustering analysis, where we formulate the point cloud data and samples from the potential parametric surface as clustering members and centroids, respectively. Then, the clustering process with dynamic updates to centroid locations serves as a direct proxy for optimizing superquadric parameters, establishing a principled link between geometric fitting and clustering dynamics. We further derive the relationship between pairwise computations of clustering centroids and clustering members to orthogonal distances, effectively eliminating the need for the time-consuming surface sampling process. Moreover, our formulation provides closed-form analytical solutions for both the fuzzy membership degree vector and the covariance matrix, ensuring efficient iteration optimization and enabling more effective handling of geometric deformations. In addition, we provide a theoretical certificate of convergence analysis and demonstrate that the clustering-inspired fitting method can escape local minima by inherently increasing the convexity of the objective function. The implementation is publicly available at this https URL. 

---
# Distinguishable Deletion: Unifying Knowledge Erasure and Refusal for Large Language Model Unlearning 

**Authors**: Puning Yang, Junchi Yu, Qizhou Wang, Philip Torr, Bo Han, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.16776)  

**Abstract**: Mitigating sensitive and harmful outputs is fundamental to ensuring safe deployment of LLMs. Existing approaches typically follow two paradigms: Knowledge Deletion (KD), which erases undesirable information during training, and Distinguishable Refusal (DR), which steers models away from using sensitive knowledge during inference. Despite rapid progress, KD-based unlearning struggles with biased deletion due to suppressing specific token sequences as a substitute for complete knowledge removal, whereas DR-based unlearning risks the re-emergence of harmful knowledge because the underlying knowledge remains intact. To address these issues, we propose Distinguishable Deletion ($\mathrm{D^2}$), a paradigm that restricts the response distribution in the latent representation rather than specific tokens to erase undesirable knowledge, while distinguishing it from retained knowledge, enabling a refusal mechanism to handle unlearned inputs safely and coherently. To implement $\mathrm{D^2}$, we introduce an energy index that quantifies the presence of knowledge and the separation between unlearned and retained content. Mathematical and empirical analyses show that energy is both accurate and efficient, enabling Energy-based Unlearning Alignment (EUA) to enforce energy-boundary unlearning during training and apply an energy-based refusal mechanism at inference. Extensive experiments demonstrate that EUA significantly outperforms previous methods, indicating the superiority of $\mathrm{D^2}$. Our code is available at this https URL. 

---
# VolTA-3D: Self-Supervised Learning for Brain MRI using 3D Volumetric Token Alignment 

**Authors**: Amy Makawana, Abhijeet Parida, Marius George Linguraru, Julia Ive, Syed Muhammad Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2605.16775)  

**Abstract**: Self-supervised learning (SSL) has advanced medical image analysis be enabling learning form large unlabelled data. However, in brain magnetic resonance imaging (MRI), most 3D models remain specialized for either segmentation of classification, limiting their ability to generalize across datasets, imaging protocols,, and downstream tasks. This lack of transferability constrains the clinical utility of 3D MRI models, despite the availability of unlabeled volumetric data. We present Volta-3D, a self-supervised 3D Vision Transformer framework designed to learn transferable volumetric representations. Volta-3D jointly aligns global class-style tokens and local patch tokens within a student-teacher paradigm and enforces fine-grained structural reconstruction. This combined global-local alignment addresses the limited semantic diversity and subtle anatomical characteristics of brain MRI, which challenges existing SSL approaches. We evaluate Volta-3D on multiple out-of-distribution downstream tasks, including hippocampal segmentation and classification of sex and Alzheimer's disease versus healthy controls. Across all tasks, representations learned by Volta-3D outperform randomly initialized baselines, demonstrating improved transferability and robustness under domain shift. Hence jointly enforcing global semantic consistency and local structural learning during pretraining enables broader concept learning from unlabeled brain MRI data. Overall VolTA-3D supports effective multi-task downstream performance with task-specific pertaining, a step towards generalizable and clinically viable 3D models. 

---
# CANSURF: An ASV-View Can Dataset and Benchmark for Detection and Tracking of Surface-Level Debris 

**Authors**: Zaid Aljundi, Zahra F. Rahmatullah, Mostafa Elemam, Abdullah Moosa  

**Link**: [PDF](https://arxiv.org/pdf/2605.16774)  

**Abstract**: Surface-level marine debris remains a practical bottleneck for autonomous clean-up, where small, reflective targets (e.g., aluminum cans) must be detected at distance under glare, ripples, and partial submersion. This paper presents, an ASV vision system and a new surface-can dataset. The dataset comprises ~7.3k raw images extracted from videos and annotated with bounding boxes, expanded via ten augmentation types to ~57k training/validation images spanning diverse lighting and water states. A family of detector and detector-tracker pipelines tailored to surface operations were benchmarked. Training YOLOv11 on CANSURF boosts performance 12x over generic datasets, highlighting the dataset's value. Experiments show that YOLOv11+ByteTrack yields the most stable tracks (fewer identity switches) and stronger multi-object accuracy under, while YOLOv11+SAHI increases recall on far-field cans at the cost of lower precision in full-context inputs. Given the mission profile, single-can pickup with approach and grab, YOLOv11 + SAHI proves better for detecting the maximum number of cans. No prior open dataset targets aluminum cans on water from a surface-level viewpoint; this dataset fills this gap and supports reproducible evaluation. 

---
# Exploring Lightweight Large Language Models for Court View Generation 

**Authors**: Zhitian Hou, Tianyong Hao, Nanli Zeng, Zhixiong Chao, Kun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2605.16770)  

**Abstract**: Criminal Court View Generation (CVG) is a critical task in Legal Artificial Intelligence (Legal AI), involving the generation of court view based on case facts. In this work, we systematically explore the capabilities of lightweight (smaller than 2B) large language models (LLMs) in CVG and their impact on charge prediction. Our study addresses four key questions: (1) how does different architecture of LLMs affect the CVG quality and charge prediction. (2) how does LLMs size contribute to the performance, (3) how do lightweight LLMs compare with Deep Neural Networks (DNNs) in these tasks, and (4) how does predicting charge by court view generation first compare with predicting it directly. Additionally, we also develop CVGEvalKit, an evaluation framework including three public available datasets for CVG tasks, as well as predicting their charges. Comprehensive experiments are conducted on this framework, where models are trained on a mixed training set and evaluated on each dataset's test set. Experimental results provide new insights into the trade-offs between model architecture, model size, and the influence between different tasks, highlighting the potential of lightweight LLMs in judicial AI applications. The source code is anonymously available at \url{this https URL} 

---
# Learning Unbiased Permutations via Flow Matching 

**Authors**: Yimeng Min, Carla P. Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2605.16755)  

**Abstract**: Learning permutations is fundamental to sorting, ranking, and matching, but existing differentiable methods based on entropy-regularized Sinkhorn produce a single softened solution and collapse under ambiguity. We present PermFlow, a conditional flow matching framework that operates directly on the affine subspace of matrices with unit row and column sums. A closed-form tangent-space projector preserves these constraints exactly along every trajectory, by construction rather than through iterative correction, and a nearest-target coupling routes distinct noisy initializations toward distinct valid permutations. The result is a model that captures multimodal permutation distributions rather than collapsing them to a single mode. On a visual sorting task with blended-digit ambiguity and a symmetric linear assignment problem, PermFlow achieves high accuracy on unambiguous inputs and recovers both valid permutations under ambiguity, where Sinkhorn-based baselines structurally fail. 

---
# UniER: A Unified Benchmark for Item-level and Path-level Exercise Recommendation 

**Authors**: Xinghe Cheng, Guiyong Zhuang, Yusheng Xie, Jiapu Wang, Yixin Liu, Quanlong Guan, Liangda Fang, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2605.16750)  

**Abstract**: Personalized exercise recommendation dynamically aligns pedagogical resources with individual knowledge mastery, which is crucial for satisfying students' dynamic learning needs in modern education. The field is currently driven by two dominant paradigms: Item-Level Exercise Recommendation (ILER) optimizes for immediate single-step state transitions, while Path-Level Exercise Recommendation (PLER) constructs coherent learning paths to maximize cumulative gains. Despite sharing the same ultimate objective, disparate evaluation setups have kept these two lines of research isolated, hindering unified benchmarking and fair comparison. To fill the gap, in this paper, we present a Unified Benchmark for Exercise Recommendation (UniER), a comprehensive evaluation framework that unifies ILER and PLER. Specifically, we introduce Weighted Cognitive Gain (WCG) as a unified metric to measure cross-paradigm algorithmic performance. Our benchmark encompasses 9 datasets spanning four generation methods, facilitating the comparison of 18 representative ILER/PLER methods. Through multi-dimensional analyses covering effectiveness, generalizability, robustness, and efficiency, our results reveal the systematic dominance of PLER and expose the pedagogical failure of ILER's fragmented recommendations under extreme sparsity and noise. Furthermore, we provide an open-source codebase of UniER to foster reproducible research and outline potential directions for future investigations. 

---
# Genflow Ad Studio: A Compound AI Architecture for Brand-Aligned, Self-Correcting Video Generation 

**Authors**: Debanshu Das, Lavi Nigam, Sunil Kumar Jang Bahadur, Gopala Dhar  

**Link**: [PDF](https://arxiv.org/pdf/2605.16748)  

**Abstract**: Recent advancements in generative video models demonstrate high visual fidelity, yet their integration into enterprise environments is restricted by temporal inconsistencies and severe brand misalignment. Current monolithic architectures struggle to enforce rigid brand constraints, frequently hallucinating unapproved visual assets. We introduce Genflow, a Compound AI System designed to enforce brand consistency in generative media production. Our architecture integrates a retrieval-based 'Brand DNA' extraction module to parameterize generation according to established corporate identity guidelines. Furthermore, we implement an Adversarial Multi-Agent Quality Control (QC) loop. Instead of a single-pass generation, this pipeline employs evaluator agents to iteratively critique generated frames against the extracted parameters, prompting generator models to refine outputs until a deterministic consensus is reached. By transitioning to a multi-stage, self-correcting pipeline, Genflow improved the yield of brand-compliant video generations from 42% to 89%, establishing a robust framework for scalable, enterprise-grade generative systems. 

---
# EmoMind: Decoding Affective Captions from Human Brain fMRI 

**Authors**: Bilal A. Mohammed, Lin Gu, Ruogo Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16739)  

**Abstract**: Decoding visual experience from brain activity has advanced substantially, but cur- rent brain-to-text systems largely recover semantic content while discarding affect. Additionally, language models can generate emotional text when prompted with categorical labels, but such labels collapse rich inter-subject variability into coarse discrete bins. We present EmoMind, the first end-to-end pipeline for decoding affective captions directly from fMRI signals. EmoMind first retrieves a semanti- cally grounded neutral scene description from brain-decoded visual features, then rewrites it using a continuous 34-dimensional emotion vector decoded from the same fMRI recording. To control the balance between content preservation and affective expression, we train the rewriter with classifier-free guidance against an identity-preserving null branch, enabling smooth interpolation between semantic fidelity and affective expressivity. We evaluate affective caption generation with a three-axis validation framework spanning subject-specificity, structural geometry, and causal control. We further augment this framework with a synthetic-brain substitution test that probes robustness to the measurement apparatus, and we benchmark each axis against GPT-4 prompted with brain-decoded top-5 emotion labels as a strong discrete baseline. Across two independent emotion fMRI datasets, EmoMind significantly outperforms label-prompted GPT-4 on all three axes, with the largest gains on metrics that require person-specific affective structure rather than population-level emotion aggregation. These results establish continuous brain-decoded affect as a viable control signal for individualized affective cap- tion generation and open new directions for studying individual affective brain organisation. 

---
# Universal Dynamics of Punctuated Progress 

**Authors**: Yian Yin, Dashun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16719)  

**Abstract**: Scientific and technological frontiers advance through punctuated dynamics, yet the principles governing these dynamics remain poorly understood. Here we collect and analyze datasets tracking the evolution of frontiers across 9 different domains, spanning materials discovery, structural biology, AI, computational biomedicine, data science, theoretical computer science, Formula-1 racing, and physical wheel building. Analyzing 6.8M solutions to 6.7K tasks, we uncover three universal patterns: (1) waiting times between new frontiers are heavy-tailed, with most attempts concentrated in long stasis; (2) frontier records accumulate at a sublinear rate, faster than logarithmic yet slower than linear growth; (3) record-breaking events are temporally correlated, generating short-term predictability yet long-term unpredictability. Despite the differences in the scale, scope, and definition of the settings, these patterns are remarkably consistent across all domains we study, and are not captured by models from complex systems, record statistics, economics of innovation, and cultural evolution. We trace the missing ingredient to the distinction between radical and incremental innovation, and develop a minimal, analytically solvable model incorporating both radical resets that restructure what is achievable and incremental refinements that exploit the current frontier. The simple model reproduces all three empirical regularities. Remarkably, the leading-order predictions are parameter-independent, identifying a new universality class governing punctuated progress and yielding testable predictions about how openness and access to frontier solutions shape the pace of advance. Overall, these results reveal universal dynamics governing punctuated progress and identify the interplay between radical resets and incremental refinements as the key driver of how scientific and technological frontiers advance. 

---
# MAVEN A Multi-Agent Framework for Multicultural Text-to-Video Generation 

**Authors**: Shuowei Li, Yuming Zhao, Parth Bhalerao, Oana Ignat  

**Link**: [PDF](https://arxiv.org/pdf/2605.16716)  

**Abstract**: Text-to-video (T2V) generation has rapidly progressed in visual fidelity, yet its ability to faithfully represent multiple cultures within a single prompt remains underexplored. We introduce MAVEN, a multi-agent prompt refinement framework designed to improve cultural fidelity in both mono-cultural and cross-cultural T2V generation. MAVEN decomposes prompts into person, action, and location dimensions, handled by specialized agents operating in parallel or sequentially. To support systematic evaluation, we contribute a new benchmark of 243 culturally grounded prompts and 972 corresponding videos, spanning three cultures (Chinese, American, Romanian), three action categories, and both mono-cultural and cross-cultural scenarios. Evaluations combining CLIP-based metrics, VLM-as-judge assessments, and videoquality measures show that multi-agent refinement, particularly parallel specialization, significantly improves cultural relevance while preserving visual quality and temporal consistency. The dataset and code are available athttps://github.com/AIM-SCU/CRAFT 

---
# GeoWorld-VLM: Geometry from World Models for Vision-Language Models 

**Authors**: Renjie Gu, Kaichen Zhou, Yan Luo, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16713)  

**Abstract**: Modern Vision-Language Models (VLMs) achieve strong semantic recognition, yet remain brittle on elementary spatial relations such as left of, on, behind, and between. One cause of this failure arises before language reasoning begins: the visual pathway may compress or discard critical 3D structural cues during feature extraction, so the language model receives image representations that are already insufficient for reliable spatial judgment. We introduce GeoWorld-VLM, a VLM-side distillation framework that transfers geometric structure from frozen camera-conditioned video world models into VLMs. GeoWorld-VLM fine-tunes only the image encoder and multimodal projector, aligning post-projector image features with intermediate world-model representations while leaving the main backbone frozen. Given images, a prompt, and a sampled camera trajectory, the world-model teacher converts static visual input into a synthetic multi-view spatial signal. Training combines spatial answer supervision, teacher-student feature alignment, and a preservation anchor to the original VLM. Since the language model remains frozen, GeoWorld-VLM preserves the original model's linguistic capabilities while attributing spatial improvements to the enhanced visual pathway. To evaluate the effectiveness and generality of the proposed method, we apply GeoWorld-VLM to two distinct VLM architectures and observe consistent improvements across both backbones. GeoWorld-VLM improves performance by approximately 4 percent on both the What'sUp and VSR benchmarks, suggesting that world-model-guided visual alignment generalizes across model structures and spatial reasoning datasets. 

---
# EfficientTDMPC: Improved MPC Objectives for Sample-Efficient Continuous Control 

**Authors**: Thomas Evers, Cristian Meo, Wendelin Bohmer, Justin Dauwels, Yaniv Oren  

**Link**: [PDF](https://arxiv.org/pdf/2605.16692)  

**Abstract**: We introduce EfficientTDMPC, a sample-efficient model-based reinforcement learning method for continuous control built on the TD-MPC family of algorithms. Central to this family is a planner that aims to find an action sequence that maximizes the estimated return. The return is estimated using a learned model and value networks, each of which can introduce error. EfficientTDMPC proposes to reduce this error in two ways. First, it introduces an ensemble of dynamics models and averages the return estimates across those models and across different rollout depths. Second, it adds the option to apply an uncertainty penalty to the planner objective, yielding a planner that avoids actions with uncertain return estimates. It then adds practical improvements which increase buffer data freshness and reduce compute. Lastly, we find that our contributions enable EfficientTDMPC to benefit more from a higher update-to-data (UTD) ratio, further improving sample efficiency. To the best of our knowledge, in the low data regime of each benchmark, EfficientTDMPC achieves state-of-the-art (SOTA) in terms of sample efficiency on HumanoidBench-Hard and DMC hard, while matching SOTA on DMC easy. 

---
# CHI-Bench: Can AI Agents Automate End-to-End, Long-Horizon, Policy-Rich Healthcare Workflows? 

**Authors**: Haolin Chen, Deon Metelski, Leon Qi, Tao Xia, Joonyul Lee, Steve Brown, Kevin Riley, Frank Wang, T. Y. Alvin Liu, Hank Capps MD, Zeyu Tang, Xiangchen Song, Lingjing Kong, Fan Feng, Tianyi Zeng, Zhiwei Liu, Zixian Ma, Hang Jiang, Fangli Geng, Yuan Yuan, Chenyu You, Qingsong Wen, Hua Wei, Yanjie Fu, Yue Zhao, Carl Yang, Biwei Huang, Kun Zhang, Caiming Xiong, Sanmi Koyejo, Eric P. Xing, Philip S. Yu, Weiran Yao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16679)  

**Abstract**: End-to-end automation of realistic healthcare operations stresses three capabilities underrepresented in current benchmarks: policy density, decisions must be grounded in a large library of medical, insurance, and operational rules; Multi-role composition: a single task requires the agent to play multiple roles with handoffs; and multilateral interaction: intermediate workflow steps are multi-turn dialogs, such as peer-to-peer review and patient outreach. We introduce $\chi$-Bench, a benchmark of long-horizon healthcare workflows across three domains: provider prior authorization, payer utilization management, and care management. Each task hands the agent a clinical case in a high-fidelity simulator of 20 healthcare apps exposed via 87 MCP tools, which it must drive to a terminal status through tool calls and writing the role's artifacts, guided by a 1,290+ document managed-care operations handbook skill. Across 30 agent harness/models configurations, the best agent resolves only 28.0% of tasks, no agent clears 20% on strict pass^3, and executing all tasks in a single session slumps the performance to 3.8%. These results raise the hypothesis that similar gaps are likely to surface in other policy-dense, role-composed, irreversible enterprise domains. 

---
# Multi-Object Tracking Consistently Improves Wildlife Inference 

**Authors**: Mufhumudzi Muthivhi, Jiahao Huo, Fredrik Gustafsson, Terence L. van Zyl  

**Link**: [PDF](https://arxiv.org/pdf/2605.16672)  

**Abstract**: Camera traps have become a common tool for wildlife monitoring efforts in ecological research and biodiversity conservation. Wildlife classification models have benefited from the increase in wildlife visual data. These models reach high levels of accuracy on curated, high-quality datasets. However, their performance remains sensitive to real-world environmental constraints. They often produce inconsistent predictions when performing inference on temporally coherent sequences. The predicted label for a single individual shifts rapidly between frames. This study exploits the temporal nature of camera-trap data to augment inferred predictions from a wildlife classification model. Specifically, we adopt several standard Multi-Object Tracking (MOT) models to link detections across consecutive frames. The curated trajectories are used to fuse the softmax class probabilities. The fused probability score produces a single consensus class label estimate that overrides misclassifications caused by noise. The analysis of the experimental results shows that our proposed strategy improves over a standalone classifier over all datasets and for each metric. Specifically, the best-performing MOT models gain a weighted F1-Score of 5.1%, 3.1% and 2.0% over the classifier across three MOT datasets. 

---
# GraViti: Graph-Level Variational Autoencoders with Relaxed Permutation Invariance 

**Authors**: Roman Bresson, Konstantinos Divriotis, Johannes F. Lutzeyer, Iakovos Evdaimon, Michalis Vazirgiannis  

**Link**: [PDF](https://arxiv.org/pdf/2605.16668)  

**Abstract**: We introduce GraViti, a transformer-based graph-level variational autoencoder that maps entire graphs to compact latent vectors. This design produces a true graph-level latent space that supports smooth interpolation, property-guided search, and other downstream tasks beyond the constraints of node-level embeddings. On molecular benchmarks, GraViti learns to decode valid samples that follow the chemical constraints present in the training data, showing that the model recovers domain rules directly from graph-level representations. We also show that, in domains where a reliable canonical node ordering exists such as molecules or bayesian networks, enforcing permutation invariance can prove detrimental for consistent reconstruction. GraViti achieves state-of-the-art reconstruction accuracy on large datasets, and provides solid generative performance. Its single-step decoding offers a lightweight alternative to more complex generation pipelines while maintaining practical sample quality. 

---
# A Scalable Tool for Measuring Manner and Result Verbs in Developmental Language Research 

**Authors**: Divyesh Pratap Singh, Dakshesh Gusain, Federica Bulgarelli, Alison Eisel Hendricks, John Beavers, Nathan M. Beers, Ifeoma Nwogu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16654)  

**Abstract**: Manner and result verbs encode different aspects of event structure and have been discussed in developmental work as a potentially informative distinction for studying early verb learning. However, this distinction remains difficult to measure at scale because large annotated resources for manner and result classification are not currently available. We present a computational approach for identifying manner and result verbs in sentence context. Using linguistically informed prompts, we generate sentence-level annotations with large language models over data drawn from MASC and InterCorp, extending coverage from previously annotated portions of VerbNet to 436 classes. We then train a RoBERTa-based classifier on these annotations and evaluate it on three held-out gold-standard datasets, including previously annotated items and a new expert-annotated set. Across these evaluations, the model shows promising performance, with average accuracy up to 89.6%. We present this work as a scalable measurement tool that can support future research on verb semantics in developmental and other language datasets, while noting that further validation is needed for borderline cases, mixed manner/result verbs, and downstream developmental applications. 

---
# SKG-Eval: Stateful Evaluation of Multi-Turn Dialogue via Incremental Semantic Knowledge Graphs 

**Authors**: Avijit Shil, Suman Samui  

**Link**: [PDF](https://arxiv.org/pdf/2605.16650)  

**Abstract**: Evaluating multi-turn dialogue systems remains challenging because response quality depends not only on the current prompt, but also on previously established entities, claims, and conversational commitments. Existing automatic evaluators, including LLM-as-a-judge frameworks and embedding-based metrics, largely rely on flat or turn-isolated representations, making them less effective at detecting long-range issues such as contradiction, topic drift, and entity inconsistency. To address this, we propose SKG-Eval, a quasi-deterministic and interpretable framework that models dialogue as an evolving Semantic Knowledge Graph (SKG) of entities, relations, and commitments across turns. The framework incrementally updates the graph through structured triple extraction and computes three complementary signals: (i) local relevance, measuring alignment with the current prompt and optional reference; (ii) historical consistency, evaluating how newly introduced information connects to prior conversational context using graph-based and embedding-driven signals; and (iii) logical coherence, assessed by a geometric contradiction engine that detects cross-turn conflicts without relying on NLI models or LLM judges. These signals are adaptively fused and aggregated into a length-invariant session score via recency-weighted trend analysis. Across multiple benchmarks, SKG-Eval achieves higher correlation with human judgments and substantially improves detection of long-range inconsistencies in extended conversations. In addition, the framework produces explicit contradiction certificates and deterministic scores for fixed inputs, enabling reproducible and auditable evaluation. Overall, our results suggest that structured externalized state tracking through semantic knowledge graphs provides a scalable alternative to implicit reasoning in LLM-based dialogue evaluators. 

---
# Learning How to Cube 

**Authors**: Ferhat Erata, Sam Kouteili, Thanos Typaldos, Timos Antonopoulos, Robert B. Jones, Byron Cook, Ruzica Piskac  

**Link**: [PDF](https://arxiv.org/pdf/2605.16632)  

**Abstract**: Despite the effectiveness of Cube-and-Conquer (C&C) for solving challenging Boolean Satisfiability (SAT) problems, no prior work has shown that transformer-based models can learn effective cubing heuristics. We introduce a neuro-symbolic post-training framework for this task. We design an MCTS-based data curation pipeline that uses symbolic heuristics to explore splitting decisions over SAT competition formulas, producing preference data grounded in solver statistics and augmented with reasoning traces from a teacher model. Our two-stage post-training, supervised fine-tuning (SFT) followed by direct preference optimization (DPO), enables a 4B-parameter model to achieve a pass@5 score of 53 on 100 SAT competition benchmarks, surpassing frontier LLMs such as Claude-Sonnet-4 (50) and matching the best symbolic heuristic (53). Ablations show that SFT alone improves pass@5 from 46 to 51, with DPO adding 2 additional benchmarks; an entropy/agreement ablation on realized first-cube decisions further shows that SFT, not DPO, accounts for the root-level decision diversity that produces complementary per-run coverage over deterministic symbolic methods. This demonstrates that transformers can be trained to make effective cubing decisions in a domain traditionally dominated by symbolic methods. 

---
# \textsc{PrivScope}: Task-scoped Disclosure Control for Hybrid Agentic Systems 

**Authors**: Shafizur Rahman Seeam, Zhengxiong Li, Zhiyuan Yu, Yimin, Chen, Yidan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16630)  

**Abstract**: Hybrid local--cloud agents enrich user requests with context from persistent working state before delegating capability-intensive subtasks to a cloud language model (CLM). While this enrichment can improve task success, it also exposes unnecessary information in the cloud-bound payload, including task-irrelevant context, carryover from prior workflows, and overly specific sensitive details, resulting in \emph{over-disclosure}. Existing solutions either isolate workflows to limit cross-workflow leakage or apply general-purpose sanitization that does not reason over LC-assembled payload scope.
We present \textsc{PrivScope}, a trusted on-device payload governor that enforces \emph{task-scoped disclosure} at the local--CLM boundary, without requiring cloud-side changes. Its key idea: sensitive information should reach the cloud only when required for the delegated subtask, and then only in the least revealing form preserving utility. \textsc{PrivScope} extracts disclosure units from the assembled payload and keeps direct identifiers and account-linked values on device. The remaining units pass through cloud-necessity control, which determines what is actually needed; units that must reach the cloud are abstracted to the least-specific representation sufficient for the task. On 100 medical-booking workflows across three commercial CLMs, \textsc{PrivScope} eliminates profile leakage (0.0\% vs.\ 17.7\%), more than halves attacker re-identification (23.1\% vs.\ 64.3\%), and achieves the highest candidate recall on every CLM tested while preserving task success close to the unprotected baseline on GPT-4o-mini and Gemini 2.5 Flash. Gains hold across five local backbones and add only seconds of on-device latency on commodity hardware. 

---
# SLEIGHT-Bench: A Benchmark of Evasion Attacks Against Agent Monitors 

**Authors**: Elle Najt, Colin Toft, Tyler Tracy, Fabien Roger, Joe Benton  

**Link**: [PDF](https://arxiv.org/pdf/2605.16626)  

**Abstract**: Since autonomous coding agents generate complex behaviors at high-volume, we may want to use other LLMs to monitor actions to reduce the risk from dangerous misaligned behavior. To better understand the limitations of such monitors against the diverse attack strategies that a coding agent could use, we present SLEIGHT-Bench (Subtle Low-itEration Insight-Guided Harmful Transcripts), a benchmark of synthetic transcripts containing 40 attacks across 11 categories, each showing a coding agent covertly pursuing a harmful objective (e.g. weight exfiltration, credential theft, unmonitored rogue deployment). We find that 24 of 40 attacks are never caught by an Opus 4.6 monitor with extended thinking across 10 trials at a 1% false-positive rate, while the overall catch rate is 23%. We identify several evasion strategies such as exploiting model priors, user instruction ambiguity, and state manipulation, which facilitate creating attacks that evade frontier monitors. We also elicit stronger monitor performance using coding agents as monitors versus regular prompted monitors, and for some evasion strategies show improved catch rates with targeted monitor prompts. Our dataset and evaluation framework are available at this https URL and this https URL. 

---
# To Trust or Not to Trust: Authors' Response to AI-based Reviews 

**Authors**: César Leblanc, Lukas Picek  

**Link**: [PDF](https://arxiv.org/pdf/2605.16623)  

**Abstract**: Large language models are increasingly discussed and used as tools that may assist with scholarly peer review, but empirical evidence regarding how authors use and perceive AI-based feedback remains limited. This paper reports findings from two independent pilot studies on authors' use and perceptions of AI-based auxiliary review at two computer science venues. After the review release, authors were invited to complete an anonymous post-review questionnaire about the AI review's usefulness, trustworthiness, agreement with human reviews, practical value for revision, perceived inaccuracies, and consent. The final dataset included 56 analyzable responses from authors of 40 papers; closed-ended items were summarized using descriptive statistics, and open-ended responses were analyzed using inductive thematic analysis. Most respondents (83.9%) considered the AI-based review useful, and 80.4% reported that it identified issues not mentioned by human reviewers. This perceived added value translated into action: 82.1% reported using at least some AI feedback in their camera-ready version. However, the authors did not treat the AI review as equivalent to a human review. They generally trusted it less than the human reviews and found human feedback clearer, even though 25.0% described at least some human reviews as not very useful. Reported problems with the AI review were usually limited: 51.8% reported minor inaccuracies, while 16.1% reported clearly incorrect, misleading, or irrelevant comments. Support for future use was strongest when AI was framed as a supervised or author-controlled tool: 96.4% said they would use AI as an internal review tool before future submissions, 89.3% preferred advance notice that AI would be used in review, and 76.8% favored explicit consent before use. 

---
# PromptDecipher: Supporting AI Tutor Authoring Through Editable Simulated Interactions 

**Authors**: Miina Koyama, Ruiwei Xiao, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2605.16605)  

**Abstract**: Chatbots have long been explored as tools to support learning, and recent advances in large language models have significantly expanded the availability of platforms for educators to author AI tutoring chatbots. Yet effective authorship demands more than writing a system prompt; it requires educators to act as learning designers, AI interaction designers, and QA engineers. In practice, however, teachers rarely fulfill these roles. Our formative study found that virtually none systematically tested their bots before deploying them to students. To address this gap, we present PromptDecipher, a system that restructures the authoring workflow around a direct correction-based interaction rather than writing abstract system prompts, teachers interact with a live chat preview and edit undesirable bot responses. An automated pipeline then analyzes the correction, proposes a targeted system prompt rewrite, and validates the change across pre-defined test scenarios. This enforces QA as a first-class activity and scaffolds teachers in roles they would otherwise skip. PromptDecipher will be deployed in an AI for Educators course enrolling hundreds of higher-education instructors. A live prototype (this https URL), an anonymized codebase (this https URL), and anonymized demo (this https URL) are available via links in the footnote. 

---
# Why Modeling Human Haptic Material Perception with AI Is Difficult 

**Authors**: Yasemin Vardar  

**Link**: [PDF](https://arxiv.org/pdf/2605.16602)  

**Abstract**: Touch plays a central role in how humans perceive and recognize materials through physical contact. Despite decades of research, the mechanisms by which tactile signals are transformed into meaningful perceptual representations remain poorly understood, limiting the design of interactive systems and intelligent agents with human-like haptic perception. Recent advances in artificial intelligence (AI) offer new opportunities to model and exploit tactile data; however, haptics presents fundamental challenges for contemporary AI due to its interaction-dependent, multimodal nature. This position paper argues that progress at the intersection of AI and haptics is constrained by three key bottlenecks: (1) the scarcity of large, diverse, and balanced haptic datasets; (2) the lack of standardized evaluation platforms and perceptual benchmarks; and (3) limitations in model capacity and interpretability when applied to tactile perception. I discuss how these challenges impede generalization, reproducibility, and scientific insight into human touch and review emerging strategies to address them. This paper highlights opportunities for coordinated, cross-disciplinary efforts to advance AI systems that not only perform robust haptic perception but also contribute to a deeper understanding of human touch. 

---
# Where Pretraining writes and Alignment reads: the asymmetry of Transformer weight space 

**Authors**: Valeria Ruscio, Eli-Shaoul Khedouri, Keiran Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2605.16600)  

**Abstract**: Cross-entropy pretraining and preference alignment update the same transformer weights, but leave geometrically distinct traces. We characterise this asymmetry with a relative-subspace-fraction probe that tracks how weight deltas align with residual-stream activation subspaces and with the prediction subspace defined by the unembedding. Alignment deltas concentrate in the read pathway ($W_Q$, $W_K$), along principal directions of attention-input activations, while remaining near-isotropic in the write pathway ($W_O$, $W_2$) relative to the prediction subspace. We explain this pattern through anisotropic gradient accumulation: updates to a matrix $W$ are sums of outer products $\delta_t a_t^\top$, and inherit directional structure from whichever side has concentrated covariance. For read-pathway matrices, this side is the input activation $a_t$, whose covariance is spiked in trained transformers and therefore produces objective-agnostic concentration. For write-pathway matrices, the relevant side is the upstream gradient $\delta_t$, whose anisotropy depends on the loss. Cross-entropy supplies the canonical sharp per-sample signal, inducing write-pathway prediction geometry during pretraining; alignment objectives typically add little further write-side concentration. We support this explanation with a within-checkpoint trajectory, a graded contrastive-objective control, and a closed-form rank-1 intervention with matched direction controls, providing causal evidence for the proposed weight-space geometry. 

---
# GRASP: Graph Agentic Search over Propositions for Multi-hop Question Answering 

**Authors**: Stockton Jenkins, Ramya Korlakai Vinayak, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16598)  

**Abstract**: Agentic retrieval improves multi-hop question answering by giving language models autonomy to iteratively gather evidence. Recent work augments these systems with knowledge graphs for structured traversal, but this combination introduces significant cost: expensive graph construction at index time and compounding token usage at inference time. We introduce Graph Agentic Search over Propositions (GRASP), an agentic system that simultaneously optimizes for high accuracy and minimal token usage in multi-hop question answering. Rather than executing a rigid, singular query, GRASP actively coordinates its retrieval strategy by decomposing multi-hop queries into dependency-aware plans. This enables GRASP to dynamically scale the number of sub-agents according to the complexity of the problem. Each sub-agent resolves its single-hop query by exploring a novel three-layer hierarchical graph of entities, propositions, and passages, using the entity layer for targeted traversal and the proposition layer for high-recall passage retrieval via reciprocal-rank voting. We evaluate GRASP on MuSiQue, 2WikiMultihopQA, and HotpotQA under two settings: open-corpus retrieval and extended context reasoning (LongBench). GRASP achieves the highest QA accuracy in the open retrieval setting on MuSiQue and 2Wiki while using 40-50 percent fewer tokens than IRCoT+HippoRAG2. Furthermore, GRASP leads on EM and F1 across all three datasets in the LongBench setting while using 30 percent fewer tokens than the next most accurate method. Finally, we introduce success economy - the amortized token cost per correct answer, weighted by difficulty - and advocate for efficiency-aware evaluation as a standard practice for agentic QA. 

---
# How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning 

**Authors**: Entang Wang, Yiwei Wang, Aleksandra Bakalova, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2605.16591)  

**Abstract**: In-context learning (ICL) excels at new tasks from minimal examples, yet we still lack a mechanistic explanation of how few-shot prompts shape a model's function vector (FV)--a causal activation direction that drives task behavior on the ICL query. Across tasks and models, an $n$-shot FV is well-approximated by a linear combination of example-level sub-FVs, suggesting additive and composable contributions from individual demonstrations. Beyond additivity, we show that models contextualize individual examples' representations based on prior examples to adaptively reweight which demonstrations dominate the FV: attention shifts toward examples that are more informative and less ambiguous under the context. Finally, a causal decomposition separates Query-Key routing from Value updates, finding that contextualization's most consistent contributions to FV quality arise from Query-Key alignment--particularly in ambiguous settings--while Value-mediated effects are more heterogeneous. Together, these results unify additive superposition with context-dependent attention reweighting into a mechanistic, testable account of how few-shot prompts implement tasks. 

---
# Voice ''Cloning'' is Style Transfer 

**Authors**: Kaitlyn Zhou, Federico Bianchi, Martijn Bartelds, Anna Pot, Yongchan Kwon, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.16578)  

**Abstract**: Artificially generated speech is increasingly embedded in everyday life. Voice cloning in particular enables applications where identity preservation is important, such as completing a recording, dubbing in a new language, or preserving the voices of individuals with speech loss. However, in our work, we find that despite the term, voice cloning does not faithfully ''clone'' an individual's voice. Instead, we find that widely-used voice cloning models systematically apply style transfer to source voices. As rated by human annotators, cloned voices are perceived as more authoritative, warm, customer-service-like, and human-like compared to their sources. Human annotators also report greater trust in cloned voices than source voices, and a greater willingness to disclose sensitive personal information to them. Our work furthermore shows that voice cloning leads to homogenization of speaker characteristics, as measured by reduced variance in accent, speaking rate, and the audio embedding space. Together, our results highlight a new set of limitations and risks of voice cloning technology and their potential impact on human behavior. 

---
# Wavelet Flow Matching for Multi-Scale Physics Emulation 

**Authors**: Gabriele Accarino, Juan Nathaniel, Carla Roesch, Pierre Gentine, Sara Shamekh, Duncan Watson-Parris, Viviana Acquaviva  

**Link**: [PDF](https://arxiv.org/pdf/2605.16573)  

**Abstract**: Accurate emulation of multi-scale physical systems governed by PDEs demands models that remain stable over long autoregressive rollouts while preserving fine-scale structures. Deterministic emulators produce overly-smoothed predictions, while generative approaches better capture details but are costly. Latent-space generative models have emerged as a compromise but with the additional cost of separately pre-trained autoencoders. We propose Wavelet Flow Matching (WFM), a novel generative emulator that overcomes current trade-offs between cost and skill by performing optimal-transport directly in the multi-scale wavelet space. Rather than learning a latent compression, WFM leverages the hierarchical structure of a U-Net to jointly predict transport velocities of a prescribed wavelet representation. On three challenging systems of chaotic fluid dynamics, WFM achieves superior long-horizon stability, accuracy and spectral coherence compared to state-of-the-art models. Our results clearly position the wavelet space as an effective training-free representation for generative emulation of complex physical dynamics. 

---
# Isotonic Survival Regression: Calibrated Survival Distributions from Deep Cox Models 

**Authors**: Anchit Jain, Kevin Zhang, Stephen Bates  

**Link**: [PDF](https://arxiv.org/pdf/2605.16571)  

**Abstract**: Time-to-event data is widespread across the life sciences and engineering, but it is typically encountered together with censoring, which complicates the application of standard machine learning methods. Deep Cox models have emerged as a popular method for analyzing time-to-event data because they gracefully handle censoring and can be used with unstructured data such as clinical text reports, genomic sequences, and pathology images. However, their predicted survival probabilities are often poorly calibrated, thus limiting their practical utility. In this paper, we propose a novel post hoc calibration method for Deep Cox models that uses isotonic regression to refine predicted survival probabilities without affecting discriminative power. We establish favorable theoretical guarantees, including a double-robustness property and asymptotic calibration. Experiments on synthetic and real-world clinical data demonstrate the empirical effectiveness of our method. 

---
# Automatic Unsupervised Ensemble Outlier Model Selection--Extended Version 

**Authors**: Hong-Phuc Phan, Tuan-Anh Vu, Tung Kieu, Son Ha Xuan, Bin Yang, Christian S. Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2605.16567)  

**Abstract**: Unsupervised outlier detection is attractive because it eliminates the need for labeled data. Moreover, forming multi-model ensembles can improve detection robustness. However, composing an ensemble without labeled data is challenging. Naively composed ensembles can suffer from ensemble saturation, where redundant or unreliable detection models degrade performance and incur unnecessary computation. We propose MetaEns, an automatic unsupervised framework for selecting ensembles of outlier detection models. Using labeled meta-datasets, MetaEns learns a model that predicts marginal ensemble gains, estimating the expected improvement from adding a candidate model to a partially constructed ensemble. At test time, this learned signal is combined with a submodular-inspired proxy objective that enforces diminishing returns through diversity-aware discounting and family-level risk regularization, thereby enabling greedy sequential selection with adaptive early stopping. As a result, MetaEns constructs compact, high-quality ensembles without access to ground-truth labels. Experiments on 39 real-world datasets show that MetaEns consistently outperforms state-of-the-art unsupervised selectors and ensemble baselines, achieving higher average precision while using fewer models. 

---
# Symphony for Speech-to-Text: Supporting Real-Time Medical Voice Interfaces 

**Authors**: Arne Nix, Robert James, Lasse Borgholt, Anna B. Ekner, Lana Krumm, Julius Severin, Dan Engel, Lars Maaløe, Jakob Havtorn  

**Link**: [PDF](https://arxiv.org/pdf/2605.16545)  

**Abstract**: After decades of use in dictation and, more recently, ambient documentation, speech is emerging as a primary modality for interacting with technology and AI in healthcare. Yet medical speech recognition remains difficult: systems must capture specialized terminology, resolve contextual ambiguity, and render measurements, abbreviations, and clinical shorthand precisely. Existing solutions are typically optimized either for general-purpose transcription or narrow dictation workflows, limiting their reliability in safety-critical settings and their usefulness for broader clinical workflows. We introduce Symphony for Speech-to-Text, a medical-grade speech recognition system for real-time streaming and batch file-based clinical use. Symphony decomposes the transcription process into specialized components for recognition, formatting, and contextual correction to optimize medical term recall while producing clinically structured text in real time and adapting across use cases. Evaluations on public benchmark and medical speech datasets show that Symphony substantially outperforms state-of-the-art systems in clinical settings while matching or exceeding them in general-domain settings, suggesting robust generalization rather than overfitting. We release a clinical benchmark dataset to support reliable validation and further progress in medical speech recognition. Symphony is available through a production-grade API for live dictation, conversational transcription, and batch audio file processing. 

---
# RAPT: Retrieval-Augmented Post-hoc Thresholding for Multi-Label Classification 

**Authors**: Lasal Jayawardena, Nirmalie Wiratunga, Ikechukwu Nkisi-Orji, Darren Nicol  

**Link**: [PDF](https://arxiv.org/pdf/2605.16535)  

**Abstract**: Industrial multi-label document understanding pipelines score candidate labels and threshold or rank them to form a label set per document. This early selection step directly affects the accuracy of downstream information extraction from the document, as well as the associated verification effort. In practice, OCR noise, label imbalance, instance-dependent label cardinality, and asymmetric error costs make global score thresholds brittle and hard to maintain as document formats evolve. We present RAPT, a deployment-oriented retrieval-augmented score thresholding wrapper, applied post-hoc to improve label set selection without retraining the underlying classifier. RAPT is a model-agnostic wrapper: any predictor that provides document representations for similarity search and per label confidence scores can be used, including metric learning encoders and fine-tuned transformer classifiers. For each query document, given a classifier's score vector, RAPT retrieves similar document thresholding situations (cases) and adapts the query's label set selection threshold using their outcomes. The adaptation selects the final label set by locally aggregating neighbour solutions (e.g. average label count, cutoff calibration). Evaluation compared multi-label classifiers (metric learners and transformers) combined with RAPT against global and label-wise thresholding baselines, and against few-shot LLMs. Across an industrial dataset and six public benchmarks, RAPT consistently outperformed global and label-wise static thresholding baselines. In the industrial setting, RAPT achieved its best predictive performance with metric learners, reaching 0.87 Macro-F1, while fine-tuned transformer variants on average achieved 0.775 Macro-F1, outperforming fewshot LLM baselines (K = 5) by 2x and requiring at least 115x less inference time and 13.5x less GPU memory. 

---
# Inventorship in AI-Assisted Inventions: Designing an Experiment to Shape Case Law 

**Authors**: Yevhenii Shchetynin, Duygu Usta, Bryan Khan  

**Link**: [PDF](https://arxiv.org/pdf/2605.16528)  

**Abstract**: The latest improvements in artificial intelligence (AI) raise new challenges for intellectual property laws, particularly concerning the inventorship issue in AI-assisted inventions - that is, those in which AI is used in the inventive process. While most jurisdictions allow only a natural person to be considered the inventor, the question of how to deal with AI-assisted inventions remains relevant. Namely, what is the nature and contribution of AI tools in an AI-assisted invention that would prevent a human from being recognized as its inventor? The main challenge in addressing this question is the lack of case law on the issue. It is reasonable to assume that with the development of AI and the growing interest in its use in the inventive process, new cases will naturally arise, which in turn will harmonize and address the inventorship issue in AI-assisted inventions to some extent. However, this process will take significant time and may not keep pace with the rapid development of AI, nor fully address the new problems that arise alongside AI advancements. This research proposes the conditions of an experiment to create relevant case law. This experiment could be initiated by society, involving stakeholders specializing in AI. The article also proposes a methodology for conducting the experiment and selecting cases that best reflect the current state of AI use in the inventive process. Conducting such an approach will help identify the most effective methods for measuring human contribution to AI-assisted inventions when determining inventorship. 

---
# Hypergraph Pattern Machine: Compositional Tokenization for Higher-Order Interactions 

**Authors**: Kyrie Zhao, Zehong Wang, Tianyi Ma, Fang Wu, Xiangru Tang, Pietro Lio, Sheng Wang, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.16527)  

**Abstract**: Hypergraphs model higher-order relations that drive real-world decisions, from drug prescriptions to recommendations. A central structural signal in such data, beyond what pairwise relations can express, is interaction compositionality: whether a higher-order relation is compositional, emergent, or inhibitory with respect to its observed or unobserved sets. In polypharmacy, the regime decides whether a drug should be dropped, kept, or excluded: a compositional drug triple can be safely simplified, an emergent triple requires all drugs jointly, and an inhibitory triple flags a drug that disrupts an existing interaction. However, existing hypergraph learning methods, which merely propagate messages over observed hyperedges, leave this compositional signal unmodeled, allowing dangerous drug combinations to slip through and be misclassified. To this end, we propose the Hypergraph Pattern Machine (HGPM), shifting the paradigm from message passing to learning the compositional pattern of subsets. It tokenizes compositional subsets, organizes them in an inclusion DAG, and trains an inclusion-aware Transformer under masked reconstruction. On ten hypergraph benchmarks, HGPM matches or exceeds state-of-the-art methods. Notably, in a real adverse-event prediction case, HGPM correctly identifies the drug addition that inhibits the side effect among feature-identical candidates, a discrimination existing methods cannot make. The code and data are in this https URL. 

---
# Toward Template-Free Explainability for Monte Carlo Tree Search 

**Authors**: Siqi Lu, Mirsaleh Bahavarnia, Hiba Baroud, Yixuan Zhang, Hemant Purohit, Ayan Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2605.16524)  

**Abstract**: Probabilistic search algorithms, such as Monte Carlo Tree Search (MCTS), have proven very effective in solving sequential decision-making tasks under uncertainty. However, interpreting asymmetric search trees that incorporate bandit-based tree traversal and simulation-based value estimation is difficult for end users based solely on raw tree statistics. While prior work requires hand-crafted formal logic constraints that must be updated when the problem changes, we present a framework that enables large language models (LLMs) to generate evidence-grounded explanations of MCTS decisions from recorded search traces in an end-to-end manner. Our framework maps natural-language questions to a structured set of intent categories, determines whether the existing tree contains sufficient evidence, triggers targeted expansion when needed, and generates explanations using tree statistics such as visit counts, value estimates, and risk information. Experimental results provide the first evidence that LLMs can serve as end-to-end explainers for probabilistic search, without requiring intermediate formal representations. 

---
# Alignment Drift in Long-Term Human-LLM Interaction: A Mechanism-Oriented Framework 

**Authors**: Xintong Yao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16516)  

**Abstract**: Long-term interaction with LLM-based systems may produce alignment drift: a gradual process in which system outputs become less constrained by the user's current message and more shaped by prior interaction history, while still appearing helpful, coherent, and responsive. This process is difficult to detect because the user's subjective experience may improve as the system becomes more familiar, useful, and attuned. Existing research on human-LLM interaction has largely focused on short-term task performance, isolated outputs, or single-instance alignment problems, leaving slow and cumulative interaction-level dynamics undercharacterized. This paper proposes a mechanism-oriented framework for describing alignment drift. The framework defines the distinction between signal A and signal B, explains how drift develops through feedback loops and sub-pattern selection, divides the process into three interactional regimes, and identifies boundary conditions for controlling drift. By framing alignment drift as a recursive interactional process rather than an isolated model-side failure, the paper provides a conceptual basis for studying long-term human-system interaction. 

---
# No Plan, Yet Human: A Reactive Robotics Model Predicts Human Planning Failures on a Clinical Task 

**Authors**: Michael Migacev, Vito Mengers, Antonia Köngeter, Oliver Brock  

**Link**: [PDF](https://arxiv.org/pdf/2605.16514)  

**Abstract**: Understanding why some sequential planning problems are harder than others requires models that go beyond average performance. They should capture the specific pattern of which problems are hard, and ideally fail in the same way people do when planning capacity is reduced. We apply AICON, a reactive gradient-descent framework developed for robotic manipulation, to the Tower of London test, a cognitive test used to assess planning in Parkinson's disease, mild cognitive impairment, and stroke. Without any lookahead planning or knowledge of human cognition, AICON reproduces the fine-grained human difficulty ordering across 24 problems better than structural task parameters and generalizes to held-out problems in a leave-two-out evaluation. Crucially, AICON outperforms a planning baseline for groups with reduced planning capacity while the planning baseline better captures healthy controls. This dissociation was predicted by the original AICON paper, which noted that the model's failure modes resemble those of Parkinson's patients who struggle with goal hierarchies but not move counts. This suggests that as planning capacity is reduced, human behavior shifts toward the reactive mode AICON models. The finding extends a broader pattern: AICON, originally built for robotics, now captures aspects of biological behavior across perception, eye movements, and sequential planning, suggesting its core abstraction reflects something real about how biological systems are organized. 

---
# The Scaling Laws of Skills in LLM Agent Systems 

**Authors**: Charles Chen, Qiming Yu, Yuhang Gu, Zhuoye Huang, Hanjing Li, Hongyu Liu, Simin Liu, Jinhao Liu, Dengyun Peng, Jiangyi Wang, Zheng Yan, Fanqing Meng, Ethan Qin, Carl Che, Mengkang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16508)  

**Abstract**: As agent systems scale, skills accumulate into large reusable libraries, yet their scaling laws remain poorly understood. Across 15 frontier LLMs, 1,141 real-world skills, and over 3M routing or execution decisions, we identify two coupled laws. Routing law: single-step routing accuracy decays logarithmically with library size ($R^2{>}0.97$ for all models), with errors progressing from local skill competition to cross-family drift and capture by overly general "black-hole skills". Execution law: before state realization, joint routing is approximately multiplicative, whereas correct execution can improve difficult downstream decisions by about $4{\times}$. A single parameter, the routing logarithmic decay slope $b$, couples the two laws: routing-side fits predict execution-side rescue across models, showing that the same library property controls both pre-execution collapse and downstream recoverability. The laws are actionable: law-guided optimization raises held-out routing accuracy from 71.3% to 91.7%, reduces hijack from 22.4% to 4.1%, and transfers directionally to downstream ClawBench and ClawMark execution settings, improving mean pass rate from 49.3% to 61.6% on ClawBench and from 28.4% to 34.5% on ClawMark. These results show that agent performance depends not only on model capability, but also on the structure, granularity, and exposure policy of the skill library. 

---
# Visual Agentic Memory: Enabling Online Long Video Understanding via Online Indexing, Hierarchical Memory, and Agentic Retrieval 

**Authors**: Aiden Yiliu Li, Nels Numan, Anthony Steed  

**Link**: [PDF](https://arxiv.org/pdf/2605.16481)  

**Abstract**: Long video understanding requires more than large context windows. It also needs a memory mechanism that decides what visual evidence to retain, keeps it searchable over long horizons, and grounds later reasoning in recoverable observations rather than compressed latent state alone. We propose Visual Agentic Memory (VAM), a training-free framework with three components. Online Indexing supports selective evidence retention under streaming constraints. Hierarchical Memory organises retained evidence in a Parallel Representation that aligns temporal context with spatial observations. Agentic Retrieval searches, inspects, and verifies candidate evidence before producing a grounded answer. On OVO-Bench, VAM achieves the highest RT+BT average (68.41) across all reported baselines, improving over end-to-end use of the same underlying MLLM (Gemini 3 Flash, 67.46). On the month-scale split of MM-Lifelong train@month (105.6 hours over 51 days), VAM reaches 17.11%, second only to ReMA with GPT-5 (17.62%). These results suggest that long-horizon video understanding benefits from treating visual memory as an explicit, inspectable, and queryable substrate. Code is available at this https URL. 

---
# MoleCode unlocks structural intelligence in large language models 

**Authors**: Zhiyuan Yan, Chen Liu, Boxuan Zhao, Kaiqing Lin, Jixiang Zhao, Yimi Wang, Liuzhenghao Lv, Hao Li, Shanzhuo Zhang, Li Yuan, Fanyang Mo  

**Link**: [PDF](https://arxiv.org/pdf/2605.16480)  

**Abstract**: Molecules are graphs, but large language models~(LLMs) are usually asked to reason about them through linear strings. The most popular molecular representation, SMILES, compresses atoms, bonds, branches and rings into a compact sequence in which topology is implicit, forcing LLMs to reconstruct molecular structure before performing the requested chemical operation. Here we introduce MoleCode, an LLM-native, training-free, graph-explicit molecular language in which all molecular components are represented as typed entities with persistent identifiers and explicit relations. MoleCode makes molecular topology directly readable, editable and auditable within the language context, allowing an LLM to operate on structure rather than recover it from syntax. Across molecular reasoning, editing, generation and analysis tasks, this representational shift improves frontier LLMs most strongly when structural access is limiting: unfamiliar molecules, topology-sensitive operations, larger structures and repetitive polymers. It also changes how inference is allocated, replacing long reasoning traces devoted to implicit structural reconstruction with shorter, more chemically directed reasoning over explicit atoms and bonds. In molecular optimization, this enables localized, property-aligned edits that preserve structural similarity to the starting compounds. The same Subgraph--Node--Edge grammar extends beyond small molecules to polymers, Markush structures, mechanism-style transformations and interleaved scientific documents, including research articles and patent disclosures in which chemical information is distributed across text and images. These results suggest that the interface between scientific objects and LLMs should not treat structure as something to be decoded from text. When the object of reasoning is relational, the structure itself should be part of the language. 

---
# Policy-Grounded Dynamic Facet Suggestions for Job Search 

**Authors**: Dan Xu, Baofen Zheng, Qianqi Shen, Jianqiang Shen, Wenqiong Liu, Chunnan Yao, Ping Liu, Rajat Arora, Kevin Kao, Hsiang Lin, Wanjun Jiang, Yusuke Takebuchi, Jingwei Wu, Wenjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16479)  

**Abstract**: Job seekers often initiate search with short, underspecified queries. At LinkedIn, over 80% of job-related queries contain three or fewer keywords, making accurate user intent inference and relevant job retrieval particularly challenging. We present dynamic facet suggestion (DFS), an interactive query refinement mechanism that facilitates intent disambiguation by surfacing personalized semantic attributes conditioned on the joint user-query context in real time. We propose a policy-grounded, retrieval-augmented ranking framework for facet suggestion, comprising offline taxonomy curation, embedding-based retrieval of top-K candidates, and distilled small language model (SLM) based candidate scoring. The system is optimized for real-time serving via pointwise single-token scoring with batching and prefix caching. Offline evaluation demonstrates high precision for generated suggestions, and online A/B tests show significant improvements in suggestion engagement and job search outcomes. 

---
# LERA: LLM-Enhanced RAG for Ad Auction in Generative Chatbots 

**Authors**: Haoran Sun, Xinrui Song, Xinyu Zhang, Zhaohua Chen, Xu Chu, Zhilin Zhang, Chuan Yu, Jian Xu, Bo Zheng, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.16474)  

**Abstract**: The integration of advertising auction mechanisms into large language model (LLM)-based chatbots presents a significant opportunity for commercialization, yet poses unique challenges in balancing relevance, efficiency, and user experience. Recently, Feizi et al.~\citep{feizi2023online} and Hajiaghayi et al.~\citep{hajiaghayi2024ad} outlined a retrieve-then-generate paradigm that decouples retrieval and generation, offering lightweight ad insertion and payment determination. However, current retrieval relies solely on text embedding similarity, which may lead to commercial misinterpretation and issues such as repetitive insertions. In this paper, we propose LERA, a two-stage retrieve-then-generate auction framework tailored for LLM chatbots. In the first stage, embedding-based coarse filtering pre-selects a small set of candidate advertisers. In the second stage, the LLM itself is queried with a carefully designed prompt to produce logits over candidates, which serve as refined organic relevance scores. These scores are combined with bids, and a critical-value payment rule accounts for both the coarse-filtering and fine-ranking thresholds, ensuring truthfulness for utility-maximizing advertisers. The framework naturally extends to multiple ad insertions within dynamic dialogue flows and long responses. Experiments on a synthetic advertiser-query benchmark show that LERA substantially improves ad selection accuracy and insertion diversity while incurring only controllable latency overhead. 

---
# Strategic Over-Parameterization for Generalizable Low-Rank Adaptation 

**Authors**: Jing Gao, Zhong-Yi Lu, Pan Zhang, Ze-Feng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16470)  

**Abstract**: Adapting large language models (LLMs) to downstream tasks via full fine-tuning is increasingly impractical due to its computational and memory demands. Parameter-efficient fine-tuning (PEFT) approaches such as Low-Rank Adaptation (LoRA) mitigate this by confining updates to a compact set of trainable parameters, but this aggressive reduction often sacrifices generalization, especially under transfer across heterogeneous tasks and domains. We revisit the tension between parameter efficiency and adaptation capacity, and ask whether the two are truly at odds. We answer in the negative by introducing LoRA-Over, a framework grounded in a simple principle: enrich the optimization landscape during training, then collapse the enrichment at inference. LoRA-Over injects auxiliary parameters into the low-rank adapters during training to broaden the effective hypothesis space, and through a decomposition-based reformulation folds them back into a standard low-rank structure with negligible reconstruction error, keeping inference cost identical to vanilla LoRA. Since not all weight matrices benefit equally from added capacity, we further propose two scheduling strategies, one statically predefined and one dynamically determined at runtime, that direct extra capacity where most needed. We evaluate LoRA-Over on language understanding (GLUE, T5-Base), dialogue (MT-Bench), arithmetic reasoning (GSM8K), and code generation (HumanEval), using LLaMA 2-7B and LLaMA 3.1-8B. Across all benchmarks and scales, LoRA-Over consistently outperforms vanilla LoRA, showing that principled over-parameterization designed to vanish at inference is an effective lever for improving PEFT generalization. Code will be released upon acceptance. 

---
# Mechanistically Interpretable Neural Encoding Reveals Fine-Grained Functional Selectivity in Human Visual Cortex 

**Authors**: Idan Daniel Grosbard, Mor Geva, Galit Yovel  

**Link**: [PDF](https://arxiv.org/pdf/2605.16468)  

**Abstract**: A central goal in understanding human vision is to uncover the visual features that drive neuronal activity. A growing body of work has used artificial neural networks as encoding models to predict cortical responses to natural images, revealing the visual content that activates category-selective regions. However, existing approaches are largely correlational and treat the encoder as a black box, leaving open which image features drive each voxel's response. We introduce Mechanistically Interpretable Neural Encoding (MINE), a framework that opens this black box by applying mechanistic-interpretability tools to localize the features within natural images that drive millimeter-scale (voxel-level) activity. MINE predicts each voxel's response using language-aligned image representations, and produces semantically interpretable descriptions of the features critical for the voxel's activation. We further generalize these per-image features into per-voxel functional profiles. To validate the per-image descriptions, we show they are sufficient to generate images that elicit voxel responses matching the responses to the original images, more accurately than images generated from random or low-attribution controls. Moreover, counterfactually inserting or removing the predicted features from images shifts activation in the expected direction, providing causal evidence. Counterfactual editing guided by the per-voxel activation profiles produces even stronger activation shifts, indicating that the profiles faithfully capture each voxel's selectivity. Finally, we apply MINE to well-studied category-selective brain regions, showing it recovers their known categorical preferences while revealing fine-grained unique voxel structure within each region. Overall, our results establish mechanistic interpretability as a path to discover and causally validate fine-grained hypotheses about neural function. 

---
# MHMamba: Multi-Head Mamba for 3D Brain Tumor Segmentation 

**Authors**: Hanjun Tao, Hua Wang, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16464)  

**Abstract**: Brain tumors exhibit high heterogeneity in morphology and multimodal contrast, making manual slice-by-slice de lineation time-consuming and experience-dependent, thus necessitating efficient and stable automated segmentation methods. To address the limitations of CNNs in modeling long-range dependencies, and the heavy computational and memory overhead and inter-block contextual in coherence of Transformers in 3D MRI, this paper proposes Multi-Head Mamba (MHMamba). This method combines a U-shaped architecture with a multi-head state-space model (Mamba), splitting the channel dimension into parallel SSM heads and aggregating them with residuals. This enhances long-range representation and improves the stability of multimodal training while maintaining linear complexity. To further align statistics and enhance lesion response, we designed a channel-space calibration module for multi-head outputs and introduced an adaptive fusion mechanism at skip connections to dynamically connect global semantics with local details, thereby improving boundary consistency and the detection of small-volume lesions. We conducted experiments and ablations on BraTS2021 and BraTS2023. The results showed that MHMamba achieved stable and significant improvements in overall accuracy, boundary smoothness, and sensitivity to tumor core and small-volume enhancement areas, while preserving the linear-complexity advantage of Mamba-based modeling, thus verifying the effectiveness and versatility of the method. 

---
# Asking Back: Interaction-Layer Antidistillation Watermarks 

**Authors**: Guang Yang, Amir Ghasemian, Fengchen Liu, Zhong Wang, Ninareh Mehrabi, Homa Hosseinmardi  

**Link**: [PDF](https://arxiv.org/pdf/2605.16462)  

**Abstract**: Detecting unauthorized knowledge distillation from a deployed LLM API is hard because the defender controls neither the attacker's training pipeline nor the next-token logits. Existing defenses operate on the teacher's output tokens -- biasing the next-token distribution (green-list watermarks, cryptographic schemes, antidistillation sampling) or rewriting outputs after generation. Recent work shows a paraphrasing attacker can strip these signals without losing the underlying knowledge.
We propose interaction-layer antidistillation watermarks, which move the trace one layer higher, into the teacher's interaction behavior: the defender wraps the teacher with a system prompt that intermittently induces a behavioral marker -- an explicit follow-up question, a low-frequency variant, or a declarative restatement. An oblivious distiller inherits the behavior, and the defender audits via black-box queries with a human-validated LLM-as-judge (Cohen's kappa = 0.84/0.78 on strong/style rubrics).
Across 63 LoRA-distilled students under a Llama-3.3-70B-Instruct teacher (35,343 judged samples), behavioral watermarks transfer at 88.9% (Gemma) / 80.9% (OLMo) / 45.2% (Qwen) relative fidelity (H1, H2). Under non-adaptive DIPPER paraphrasing, robustness decomposes into a teacher-self ceiling (about 66.4%) and student-relative retention of 21-112%, with OLMo preserving the watermark above the teacher itself (H3, F-Amp). Low-density (about 20%) explicit and implicit declarative variants transfer above per-family baseline (H4, F-Style). An N=20 in-lab study (pre-registered Latin-square) shows all marker variants within 0.22 Likert step of baseline; TOST, Friedman, and Bonferroni-Wilcoxon support H5. The interaction layer is a viable design locus for antidistillation watermarking, complementary to token-, model-, and reasoning-trace-layer defenses. 

---
# Conservative AI for Safety-Sensitive Medical Image Restoration: Residual-Bounded CT-CTA Enhancement for Intracranial Aneurysm-Relevant Signal Recovery 

**Authors**: Weijun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.16458)  

**Abstract**: Image restoration models are increasingly applied to degraded medical scans, but in safety-sensitive settings they must improve image quality without uncontrolled modification of clinically important regions. This is especially relevant for intracranial CT and CT angiography (CTA), where small vessels and aneurysm-relevant cues lie near high-contrast anatomical boundaries. We frame medical image restoration as a conservative AI problem and present a residual-bounded 2.5D restoration framework trained on synthetically degraded CT/CTA inputs. The model adds a learned residual to the original center slice through an edit-control map that limits the magnitude and spatial extent of modification. We evaluate the framework using an aneurysm-relevant image-recovery matrix, paired comparison against a Gaussian baseline, Monte Carlo stability testing, anatomical localization of meaningful edits, and external evaluation on low-dose CT. On 50 out-of-distribution CT-CTA cases, the bounded model achieved a mean target gain of 0.0635, a mean PSNR of 37.51 dB, and an iatrogenic-edit rate of 4.0%. Across 1,000 Monte Carlo runs, it remained net positive in 85.4% of runs with no stably negative cases. On external low-dose CT, the model was directionally beneficial and produced a substantially smaller modification footprint than the baseline. Meaningful edits concentrated in brain and skull regions while unrelated anatomy showed negligible change. These findings provide preliminary computational evidence that residual-bounded restoration is feasible in boundary-sensitive vascular imaging, but they do not establish clinical diagnostic performance and require expert review and prospective validation before clinical use. 

---
# Identifiable Token Correspondence for World Models 

**Authors**: Youngin Kim, Ray Sun, Inho Kim, Bumsoo Park, Hyun Oh Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.16457)  

**Abstract**: Transformer-based world models have shown strong performance in visual reinforcement learning, but often suffer from temporal inconsistency in long-horizon rollouts, including object duplication, disappearance, and transmutation. A key reason is that most existing approaches treat next-frame prediction purely as a token generation problem, without explicitly modeling correspondence between tokens across time. We formulate next-frame prediction as a structured probabilistic inference problem with latent token correspondence variables, deriving a model in which each next-frame token is explained either by copying a token from the previous frame or by generating a new token. Our experiments show state-of-the-art performance on 4 challenging benchmarks. The proposed method achieves a return of 72.5% and a score of 35.6% on the Craftax-classic benchmark, significantly surpassing the previous best of 67.4% and 27.9%. We release our source code on this https URL. 

---
# Peak-Detector: Explainable Peak Detection via Instruction-Tuned Large Language Models in Physiological Sign 

**Authors**: Jiahui Li, Yida Zhang, Zixuan Zeng, Jiayu Chen, Yingjian Song, Yin Xiao, Nishan Dong, Junjie Lu, Younghoon Kwon, Xiang Zhang, Jin Lu, Wenzhan Song, Fei Dou  

**Link**: [PDF](https://arxiv.org/pdf/2605.16452)  

**Abstract**: Accurate peak detection across diverse cardiac physiological signals, including the Electrocardiogram (ECG), Photoplethysmogram (PPG), Ballistocardiogram (BCG), and Bodyseismography (BSG), is fundamental for cardiovascular monitoring but is often hindered by artifacts and signal variability. Conventional algorithms are typically engineered with expert knowledge for a single signal modality, limiting their generalizability. Conversely, deep learning-based methods often lack interpretability, limiting transparency for expert verification and hindering expert-computer interaction. To address these limitations, we introduce Peak-Detector, a novel framework that leverages instruction-tuned Large Language Models (LLMs) for robust, cross-modal, and explainable peak detection. A core innovation of our framework is a "peak-representation" technique that transforms time-series data into a condensed format, preserving critical event information while significantly reducing signal length. This representation provides a crucial inductive bias, guiding the LLM to reason over physiologically meaningful events rather than raw, noisy data. The model is optimized through a two-stage process: supervised fine-tuning (SFT) followed by reinforcement learning (RL) with a multi-objective reward function. The model's self-explanation capabilities are cultivated by fine-tuning on a custom-built Peak-Explanation dataset. Across four modalities-ECG, PPG, BCG, and BSG-spanning seven datasets (six public benchmarks plus one real-world cohort), Peak-Detector demonstrates strong cross-modal performance, achieving best or tied-best detection under clinically relevant temporal tolerance. Beyond accuracy, the generated rationales surface failure modes and support verification and error analysis. 

---
# Physics-Guided Geometric Diffusion for Macro Placement Generation 

**Authors**: Jongho Yoon, Jinsung Jeon, Seokhyeong Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16451)  

**Abstract**: Macro placement is a pivotal stage in VLSI physical design, fundamentally determining the overall chip performance. Recent data-driven placement methods have demonstrated significant potential, yet they often struggle to handle sequential dependencies and to balance topological connectivity with physical constraints. To bridge this gap, we propose MacroDiff+, a physics-guided geometric diffusion framework. Specifically, we design a dual-domain denoising architecture that couples topological connectivity encoded by heterogeneous GNNs with global geometric context modeled by a Transformer. Furthermore, we introduce Physics-Guided Sampling, an inference strategy that actively steers the generation using explicit gradients to ensure both statistical plausibility and physical validity. On the ISPD2005 MMS benchmarks, MacroDiff+ outperforms state-of-the-art baselines with a 6.1-6.2% reduction in wirelength. Notably, it exhibits superior stability and scalability on large-scale designs where prior methods fail to converge. The source code is available at this https URL. 

---
# PESD-TSF: A Period-Aware and Explicit Structured Decomposition Framework for Long-Term Time Series Forecasting 

**Authors**: Hua Wang, Xianhao Jiao, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16449)  

**Abstract**: Deep forecasting models often suffer from attenuated periodic perception and entangled trend-noise representations as network depth increases. Moreover, the widely adopted channel-independent paradigm, while improving training stability, disrupts intrinsic dynamic coordination among variables, hindering the modeling of cross-variable consistency in multivariate time series. To address these issues, we propose PESD-TSF, a physics-inspired structured decomposition framework for long-term time series forecasting that jointly emphasizes interpretability and predictive accuracy. PESD-TSF introduces three key designs. First, a Multiplicative Periodic Gating mechanism incorporates continuous-time priors to dynamically modulate signal amplitudes, preserving periodic structures across deep layers. Second, a multi-scale structured encoder integrates detrended attention with hierarchical sampling to explicitly decouple long-term trends from high-frequency variations while retaining fine-grained temporal semantics. Third, to recover disrupted inter-variable dependencies, we propose Cross-Scale Collaborative Attention (CSCA) together with an RLC regularization scheme, which reconstructs global inter-variable topology in deep feature spaces and enforces physically consistent collaboration through orthogonality and consistency constraints. Extensive experiments on benchmark datasets from multiple domains demonstrate that PESD-TSF consistently achieves state-of-the-art performance, with particularly strong gains on multivariate forecasting tasks involving complex inter-variable coupling, highlighting its superior structural modeling capability and generalization. 

---
# Nested Spatio-Temporal Time Series Forecasting 

**Authors**: Yinghao Ai, Yukai Zhou, Ruoxi Jiang, Junyi An, Chao Qu, Zhijian Zhou, Shiyu Wang, Fenglei Cao, Zenglin Xu, Furao Shen, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2605.16447)  

**Abstract**: Spatiotemporal forecasting is critical for real-world applications like traffic management, yet capturing reliable interactions remains challenging under noisy and non-stationary conditions. Existing methods primarily rely on historical spatial priors, often failing to account for evolving temporal correlations and suffering from systematic errors. In this work, we propose a nested forecasting framework that couples future macro-level regional trends with micro-level historical observations, enabling top-down guidance from abstract future representations for fine-grained forecasting. Specifically, we employ a spectral clustering-based approach to construct semantically coherent regions, providing both theoretical and empirical evidence that this representation effectively filters systematic noise while preserving essential trends. Building on this, we develop a progressive coarse-to-fine predictor to integrate these representative features into the inference process. This enables the model to leverage trend predictions to anticipate dynamic anomalies, such as periodic offsets, in advance. Furthermore, extensive experiments on multiple high-dimensional datasets demonstrate that our method consistently outperforms state-of-the-art baselines, validating the effectiveness of future macro-guided nested forecasting. 

---
# Avoiding Structural Failure Modes in Tabular Fair SSL: Online Primal-Dual Allocation under Confidence Gating 

**Authors**: Hangchun Liang, Changchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.16446)  

**Abstract**: Semi-supervised learning (SSL) enables prediction with limited labels, but high-stakes tabular applications (medical, credit, recidivism) require statistical fairness guarantees. We identify a structural conflict in tabular fair SSL through a diagnostic stress test: under confidence-gated pseudo-labeling, moment-matching fairness regularizers can trigger two failure modes -- Masking Collapse (fairness erodes confidence, starving pseudo-labels) and Trivial Saturation (drift to constant predictors). We propose Online Primal-Dual Allocation (OPDA), an online controller that schedules fairness and entropy-based stability penalties using violation, risk, and pseudo-label health signals, avoiding per-dataset selection of a fixed fairness weight within this diagnostic regime. On the evaluated tabular benchmarks (Adult, ACSIncome, COMPAS), OPDA mitigates the degenerate regimes observed under static weighting and simple single-signal adaptive baselines. On Adult and COMPAS, it yields non-degenerate operating points competitive with the empirical static-$\lambda$ frontier; on ACSIncome, it preserves utility with a wider fairness-utility spread. Relative to OPDA-lite, the full controller mainly shifts the operating point toward higher utility on ACSIncome, while Adult highlights the fairness-utility trade-off between the two variants. These results position OPDA as a calibration-free controller for non-degenerate operating points in tabular fair SSL without per-dataset tuning. 

---
# Membership Inference Attacks on Discrete Diffusion Language Models 

**Authors**: Shailesh Kasivelrajan  

**Link**: [PDF](https://arxiv.org/pdf/2605.16445)  

**Abstract**: Masked Diffusion Language Models MDLMs replace autoregressive generation with iterative demasking and their privacy properties are largely unstudied. We study membership inference attacks MIA on fine tuned MDLMs and show they are significantly more vulnerable than current grey box baselines suggest. We extract a 46 dimensional feature vector from the models reconstruction loss at four masking ratios and train XGBoost and MLP classifiers on top. On the MIMIR benchmark across six text domains XGBoost achieves mean AUC 0.878 peaking at 0.930 on Pile CC and beats the SAMA grey box baseline by 0.062 AUC on average. A leave one signal out ablation shows that the ELBO trajectory alone drives most of this with a mean drop of 0.130 when removed while attention features add almost nothing below 0.003. We also design a shadow model transfer attack where K equals 3 surrogate MDLMs trained on data from unrelated domains generate classifier labels with no access to the target domain. This achieves 0.858 mean AUC within 0.020 of the white box oracle and establishes shadow model transfer as a practical and near equally effective attack path. 

---
# Diffusion Attention Expert Model for Predicting and Semi-automatic Localizing STAS in Lung Cancer Histopathological Images 

**Authors**: Liangrui Pan, Jiadi Luo, Yuxuan Xiao, Chenchen Nie, Xiaoshuai Wu, Songqing Fan, Ling Chu, Manqiu Li, Rongfang He, Zhenyu Zhao, Ruixing Wang, Shulin Liu, Yiyi Liang, Xiang Wang, Qingchun Liang, Shaoliang Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.16444)  

**Abstract**: Accurate intraoperative and postoperative diagnosis of spread through air spaces (STAS) is essential for guiding surgical decisions and postoperative management in lung cancer. However, histopathological assessment is labor-intensive and is prone to missed or incorrect diagnoses. We propose a Diffusion Attention Expert Model (DAEM) to detect STAS in frozen sections (FSs) and paraffin sections (PSs). Its diffusion attention expert module leverages full attention aggregation to learn multi-scale features from histopathological images, while a dual-branch architecture strengthens multi-scale feature representation. On an internal dataset, DAEM achieves AUCs of 0.8946 for FSs and 0.9112 for PSs. Validation on external multi-center datasets from eight institutions demonstrates strong generalizability and interpretability. Using tumor microenvironment (TME) features in PSs, we further enable semi-automatic measurement of STAS location and its distance from the primary tumor. Several quantitative TME metrics are identified as potential biomarkers for STAS, including micropapillary-type STAS. Overall, DAEM offers a clinically actionable framework for STAS assessment by enabling accurate and interpretable detection on FSs and PSs, supporting postoperative risk stratification through quantitative TME-based analysis. 

---
# Two-Valued Symmetric Circulant Matrices: Applications in Deep Learning 

**Authors**: Jayakrishna Amathi, Venkata Prasanth Yanambaka, Saraju P. Mohanty, Elias Kougianos  

**Link**: [PDF](https://arxiv.org/pdf/2605.16443)  

**Abstract**: Despite the success of deep neural networks in vision, medical diagnosis, and IoT scenarios, their deployment on resource-limited platforms poses serious challenges due to their high storage requirements, computational complexity, and large footprint. In particular, fully connected layers require a large number of weights, making it difficult for edge devices to accommodate them. To overcome these challenges associated with limited platforms, this paper proposes the Two-Valued Symmetric Circulant Matrix (TVSCM), a very sparse architecture that employs just two weights per layer to keep it circulant and symmetric. The extreme form of structured sparse architecture provides negligible storage costs compared to traditional full-weight storage. Instead of hardware and additional stages of other traditional sparse learning techniques, such as low-rank approximation and pruning approaches, this architecture provides an extreme form of sparsity, achieving very minimal storage requirements. The simulation study demonstrates more than 80$\times$ reduction in model parameters, reducing parameters from 623,290 to 7,852 on MNIST and from 24,709 to 942 on the MIT-BIH arrhythmia dataset, while maintaining comparable accuracy from 97.6% to 93.5% on MNIST and from 97.6% to 93.1% on MIT-BIH. Due to its minimal architectural requirements and very low power consumption, this architecture would be ideal for edge computing platforms, tiny-ML platforms, IoMT systems, and battery-powered systems. 

---
# Hierarchical Two-Stage Framework for Environment-Aware Long-Horizon Vessel Trajectory Prediction 

**Authors**: Ganeshaaraj Gnanavel, Tharindu Fernando, Sridha Sridharan, Clinton Fookes  

**Link**: [PDF](https://arxiv.org/pdf/2605.16442)  

**Abstract**: Long-horizon vessel trajectory forecasting under real ocean conditions is critical for collision avoidance, traffic management, and route planning. However, achieving accurate predictions is challenging due to long-range temporal dependencies and dynamic environmental factors such as currents, wind, and waves. To address these issues, we propose a hierarchical two-stage framework that combines a coarse long-term predictor with a grid-aware short-term predictor through a hierarchical fusion mechanism. The short-term branch leverages a Spatio-Temporal Graph Transformer on discretized maritime cells to capture localized dynamics, while the long-term branch encodes overarching navigational intent. An integrated environmental module incorporates oceanographic parameters, including surface currents, wind vectors, and significant wave height, using cross-modal attention and feature-wise modulation for adaptive response to varying sea conditions. Additionally, a learnable Savitzky-Golay smoothing layer enhances temporal coherence in fused trajectories. We evaluate our approach on Australian Craft Tracking System (CTS) data from the North West region, aligned with Copernicus Marine Service products, using a 3-hour input and a 10-hour prediction horizon. Experimental results show that our framework outperforms the state-of-the-art by 25% in Average Displacement Error (ADE) and 17% in Final Displacement Error (FDE). Ablation studies further validate the contribution of each component. 

---
# DeepArrhythmia: Segment-Contextualized ECG Arrhythmia Classification via Selective Evidence Acquisition 

**Authors**: Jiahui Li, Ruili Fang, Zishuai Liu, WenZhan Song, Jin Lu, Fei Dou  

**Link**: [PDF](https://arxiv.org/pdf/2605.16441)  

**Abstract**: Beat-level Electrocardiography (ECG) arrhythmia detection aims to assign an arrhythmia class to each beat in a recording, yet many existing systems treat beats as isolated local instances. This is limiting because beat labels often depend on multi-beat rhythm context, including timing, compensatory pauses, and beat-to-beat morphological consistency. We present DeepArrhythmia, a tool-grounded multimodal framework for segment-contextualized beat-level ECG arrhythmia classification. Given a multi-beat ECG segment, DeepArrhythmia combines the raw ECG signal and a rendered waveform image, localizes R peaks to identify beat instances, and produces structured beat-level predictions. The framework decouples physiological measurement from evidence integration using specialized tools for beat localization, numerical rhythm--morphology extraction, and morphology-focused textual analysis. DeepArrhythmia uses segment-level confidence to route between minimal and rich evidence states, since richer physiological evidence is not uniformly useful. This agentic design integrates rhythm context, explicit physiological grounding, and selective evidence acquisition for decision making. 

---
# Semantic Smoothing via Novel View Synthesis for Robust SAR Image Classification 

**Authors**: Daniel Brignac, Fengwei Tian, Banafsheh Latibari, Abhijit Mahalanobis, Ravi Tandon  

**Link**: [PDF](https://arxiv.org/pdf/2605.16440)  

**Abstract**: Deep neural networks are vulnerable to adversarial perturbations, limiting deployment in safety-critical applications such as synthetic aperture radar (SAR) automatic target recognition (ATR). Randomized smoothing improves robustness by averaging predictions over noisy inputs, but isotropic noise often fails to preserve the semantic structure of SAR imagery. We propose semantic smoothing, a defense that replaces noised-based perturbations with structured randomized transformations generated by a novel view synthesis model. For SAR, we condition on acquisition geometry to synthesize multiple plausible radar views. Predictions across generated randomized views are aggregated to form a robust classifier. Experiments show that semantic smoothing improves robustness against standard attacks, such as FGSM and PGD, and SAR-specific attacks, such as OTSA and SMGAA, while also increasing clean classification accuracy. These results demonstrate that randomized smoothing via semantically preserving geometric transformations is a promising alternative to isotropic noise for adversarial defense in structured sensing domains. 

---
# KVCapsule: Efficient Sequential KV Cache Compression for Vision-Language Models with Asymmetric Redundancy 

**Authors**: Yingbing Huang, Tharun Adithya Srikrishnan, Steven K. Reinhardt, Deming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.16439)  

**Abstract**: Vision-Language Models (VLMs) have emerged as a critical and fast-growing extension of Large Language Models (LLMs) that enable multimodal reasoning through both text and image inputs. Although VLMs enrich the capabilities of language models, they also inherit and amplify key computational bottlenecks: the memory overhead caused by the large key-value (KV) cache during autoregressive decoding. This challenge is particularly severe in VLMs, where images produce longer token sequences and denser feature representations compared to text. Moreover, the spatial and information-rich nature of vision tokens introduces structured attention patterns that make many LLM-oriented KV cache compression techniques ineffective when applied directly to VLMs.
In this work, we conduct a detailed empirical analysis of the behavior of vision tokens, highlighting the critical differences from purely text-based models. Based on these insights, we propose KVCapsule, a novel KV cache compression framework for vision tokens. KVCapsule keeps the pretrained VLM backbone frozen, requires no modification to the attention computation modules, and can be integrated into existing VLMs through lightweight compression and reconstruction components. We evaluate KVCapsule on multiple VLMs and benchmark tasks, demonstrating up to 2x improvement in TPS and 2.4x reduction in KV cache memory at a 60% compression ratio, with negligible degradation in accuracy or response quality. Our findings offer practical pathways to scale VLM inference under constrained memory budgets and inspire further research into structure-aware cache compression for multimodal models. 

---
# Byzantine-Resilient Federated Learning via QUBO-Based Client Selection on Quantum Annealers 

**Authors**: Andras Ferenczi, Sutapa Samanta, Dagen Wang, Jason Qizhe Qin  

**Link**: [PDF](https://arxiv.org/pdf/2605.16438)  

**Abstract**: Federated Learning (FL) trains a global model across decentralized clients while preserving data privacy, but at scale it is vulnerable to malicious updates. Byzantine-resilient aggregation methods such as MultiKrum score gradients against their nearest neighbors and can miss malicious updates that preserve the statistical properties of honest ones. We propose a quantum annealing approach that reformulates client selection as a Quadratic Unconstrained Binary Optimization (QUBO) problem, encoding pairwise distances into a cost function solved by quantum annealers (QA). Unlike MultiKrum's greedy per-client scoring, the QUBO formulation jointly optimizes over all subsets to find the mutually closest group of $m$ clients. At small scale (15 clients), QUBO outperforms MultiKrum on the most challenging Byzantine attacks: e.g., Advanced LIE is detected with 95.11% accuracy versus 81.33% on MNIST and 97.78% versus 75.56% on CIFAR-10. QUBO fares poorly on simpler attacks where MultiKrum excels, so the two methods are complementary. QUBO quality also degrades as the number of clients grows. To address this, we introduce a MultiSignal ensemble that uses a dual-feature routing gate based on Euclidean and cosine Krum score gaps to classify attacks into four regimes and routes evasion attacks to a suspicion-penalized QUBO with agreement voting. At 100 clients on MNIST, MultiSignal achieves 95.3% average detection accuracy versus 91.8% for classical MultiKrum, with the largest gains on Sparse Lie (72.0% to 95.2%, +23.2 points) and Advanced Lie (80.4% to 85.2%, +4.8 points). These results show that QUBO-based quantum annealing with MultiSignal is a principled and scalable defense against the most challenging Byzantine strategies in federated learning. 

---
# The End of Trust: How Agentic AI Breaks Security Assumptions 

**Authors**: Osama Zafar, Alexander Nemecek, Erman Ayday  

**Link**: [PDF](https://arxiv.org/pdf/2605.16436)  

**Abstract**: For decades, the security of digital interaction has rested on an unacknowledged economic constraint. Attackers faced a tradeoff between the fidelity of a deception and the scale at which it could be deployed. Convincing impersonation required sustained human effort and was confined to a narrow set of high-value targets, while mass-market attacks sacrificed plausibility for reach. Detection systems, verification mechanisms, and user awareness training have all been implicitly calibrated to the artifacts of cheap deception that this tradeoff produced. Agentic AI collapses the tradeoff, allowing high-fidelity, individually tailored deception to be produced at mass-market scale. We argue that this shift exhausts a security paradigm rather than merely intensifying the threat landscape. We introduce the Infinite Impostor, an attack model in which an autonomous agent interposes itself between two parties who already trust each other, hijacking an existing relationship rather than building a new one from scratch. Detection-oriented defenses share an assumption that generative progress is eliminating, that synthetic outputs are distinguishable from authentic ones. We propose a suspect-by-default paradigm that shifts security from authenticating actors to evaluating actions, and examine the governance tensions that arise when platforms become the regulatory substrate of digital interaction. 

---
# GPU-Accelerated Deep Learning for Heatwave Prediction and Urban Heat Risk Assessment 

**Authors**: Adis Alihodžić  

**Link**: [PDF](https://arxiv.org/pdf/2605.16435)  

**Abstract**: Heatwaves are an important problem in cities, and climate change makes this problem more difficult. In this paper, we present a GPU-based deep learning framework for next-day prediction of urban thermal conditions and for heat risk assessment. The study was carried out in Sarajevo by using MODIS land surface temperature data and Open-Meteo forecast data. We tested several models, including convolutional models and spatiotemporal models. Among them, ConvLSTM with a mixed loss function gave the best results. The obtained values were MAE = 0.2293, RMSE = 0.3089, and R2 = 0.8877. The experiments also showed that results can be improved by using longer temporal series and additional meteorological variables. Since the framework was implemented on a GPU and trained with mixed precision, the execution time was reduced. Based on the predicted temperature fields, it was also possible to combine hazard information with exposure and vulnerability data in order to generate city heat risk maps. The proposed framework can be used as a practical basis for city heat analysis. 

---
# Edge-AI-Driven Learning-to-Rank for Decentralized Task Allocation in Circular Smart Manufacturing 

**Authors**: Mohammadhossein Ghahramani, Yan Qiao, Mengchu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.16433)  

**Abstract**: Task allocation in smart manufacturing systems needs to operate under decentralized decision-making, dynamic workloads, and shared resource constraints. In circular manufacturing settings, these challenges are further intensified by the need to balance operational efficiency with resource and energy sustainability. While learning-based approaches have been explored, many focus on predicting absolute performance metrics that do not necessarily translate into improved allocation outcomes, since decentralized assignment is governed by the relative ordering of candidate machines. This work proposes an Edge-AI-driven decentralized task allocation framework based on ranking-aware negotiation, where lightweight decision intelligence is embedded at the machine level to enable low-latency coordination without centralized control. The framework is developed progressively: a resource-aware heuristic first establishes the decentralized bidding structure, an Edge-AI-based regression model then provides learned local bid approximation, and a ranking-aware formulation finally reshapes the learning objective to align with the ordering-based nature of winner selection. Each machine evaluates incoming tasks using local information, including processing capability, queue state, and resource contention. The framework is evaluated via discrete-event simulation under high-load and tight-deadline scenarios using delay, deadline violations, throughput, and energy consumption. Results show improved delay and deadline adherence under high load, and enhanced energy efficiency under tighter constraints, leading to more resource-efficient operation aligned with circular manufacturing objectives. These findings demonstrate that aligning learning objectives with decentralized decision structures is critical for effective negotiation-driven task allocation. 

---
# MR-SLAM: Immersive Spatial Supervision for Multi-Robot Mapping via Mixed Reality 

**Authors**: Prakash Aryan, Cem Erdogdu, Kavinaya Kumarchokkappan, Timo Kehrer, Sebastiano Panichella  

**Link**: [PDF](https://arxiv.org/pdf/2605.16432)  

**Abstract**: Operating a multi-robot fleet for simultaneous localization and mapping (SLAM) in applications such as building inspection or warehouse-aisle monitoring requires the operator to maintain spatial awareness of each robot's position and mapping state, a task that scales poorly on conventional 2D interfaces. We present MR-SLAM, a mixed reality (MR) system in which an operator wearing a Meta Quest 3 headset teleoperates three simulated TurtleBot3 robots through a passthrough view with real-world occlusion, while spatially anchored dashboard panels report mapping progress in situ. Each robot runs an independent SLAM Toolbox instance whose occupancy grid is merged in real time on a Robot Operating System 2 (ROS 2) back end. Across five 9-minute evaluation sessions, the system delivered scans at 8.83 +/- 0.16 Hz, mapped 17.9 +/- 0.8 m^2 of merged occupancy, and reached 94.7 +/- 0.5% cross-instance occupancy consistency across robot pairs. An additional session recorded 6.3 ms median transform jitter and 26.7 m^2 coverage of a 41 m^2 grid. We position MR-SLAM as a reference implementation for combining passthrough mixed reality supervision with multi-robot SLAM on consumer hardware. 

---
# A Theory of Training Profit-Optimal LLMs 

**Authors**: Sophie Hao, William Merrill  

**Link**: [PDF](https://arxiv.org/pdf/2605.16430)  

**Abstract**: Scaling LLMs requires tremendous computational resources, and recent advances in AI have gone hand in hand with massive amounts of capital expenditure. While it is established that scaling up LLMs reliably increases model quality (quantified in terms of loss or downstream evaluations), it is unclear how these quality improvements translate to potential revenue, and whether revenue increases would offset costs of larger-scale training and inference. In this work, we develop an economic model for characterizing the rational behavior of an LLM training firm by combining scaling laws with microeconomic theory. Under our model of firm behavior, LLM quality can be increased with more parameters and training tokens, leading to more potential adoption by consumers, who each have a quality threshold for using the LLM. On the other hand, additional parameters and training tokens both incur additional costs. We analyze the profit maximization problem for this model under compute-bound and data-bound regimes. In the compute-bound regime, optimal model size and token budget track hardware efficiency $E$ (FLOPs/\$) at a near-linear rate; total training cost then scales sub-quadratically in $E$. Data efficiency improvements incentivize larger models and training expenditure. When we are limited to $D$ data, profit-optimal training expenditure scales as $D^2/E$, i.e, increase with data and decreases with hardware efficiency (as well as data efficiency). Finally, we analyze practical trends in training expenditure: current trends are consistent with our most permissive model variants in the compute-bound regime, but are not profit-optimal in the data-bound regime or assuming hardware advances will stall. Overall, our results provide a theory of profit-optimal LLM training, providing a foundation for engaging critically with industry statements and supporting long-term economic decision making. 

---
# QuantFPFlow: Quantum Amplitude Estimation for Fokker--Planck Policy Optimisation in Continuous Reinforcement Learning 

**Authors**: Abraham Itzhak Weinberg  

**Link**: [PDF](https://arxiv.org/pdf/2605.16429)  

**Abstract**: We introduce \textbf{QuantFPFlow}, a reinforcement learning framework that integrates quantum amplitude estimation into the Fokker--Planck~(FP) formulation of stochastic policy optimisation. Classical continuous-space RL agents must estimate the FP partition function $Z = \int e^{-V(\mathbf{x})/D}\,d\mathbf{x}$ at cost $\calO(1/\varepsilon^{2})$; QuantFPFlow replaces this with a Grover-amplified amplitude estimator achieving $\calO(1/\varepsilon)$ -- a provable quadratic speedup. While the full quantum acceleration requires fault-tolerant hardware, the quantum-inspired classical simulation demonstrated here already exhibits the $\calO(1/\varepsilon)$ algorithmic structure.
The estimated stationary distribution $\rhostar$ drives a theoretically grounded exploration bonus $\Raug = \Renv + \alpha\log(1/\rhostar(s))$. This bonus steers the agent toward globally optimal regions of multimodal reward landscapes while simultaneously constraining policy variance through FP diffusion matching.
On a continuous-control task specifically designed to expose local-optima failure, QuantFPFlow achieves mean reward $1{,}295.7 \pm 423.2$ versus $1{,}284.0 \pm 474.0$ for Soft Actor-Critic~(SAC), while discovering the global optimum \textbf{10.4\,\% more frequently} (33.9\,\% vs.\ 30.7\,\%). Policy entropy remains near $H(\pi)\approx 6.5$\,nats throughout training, whereas SAC collapses to $1.5$\,nats, confirming that FP diffusion matching actively prevents premature convergence. Dimensionality experiments further show computational scaling of $\calO(d^{0.35})$ for QuantFPFlow versus $\calO(d^{0.76})$ for classical FP estimation. 

---
# The Impact of AI Search on the Online Content Ecosystem: Evidence from Google and Reddit 

**Authors**: Peibo Zhang, Ruomeng Cui, Dennis J. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16428)  

**Abstract**: Search engines traditionally complement online content platforms by directing users seeking information to external websites. The emergence of generative AI search tools that summarize answers directly on the results page may disrupt this relationship by making visits to source platforms optional. We study this question using Google AI Overviews and Reddit, one of the largest online discussion platforms. Our identification exploits Google's content moderation policy: Safe-for-Work (SFW) Reddit communities are indexed by Google organic search and surfaced in Google AI Overviews, while Not-Safe-for-Work (NSFW) communities, though indexed by organic search, are prohibited from being referenced in AI Overview summaries. Using a difference-in-differences design, we find that AI Overviews increase engagement in SFW communities: daily comments rise by 12.0 percent and the number of commenting users by 12.3 percent relative to NSFW communities. The effects are concentrated in experience-based discussions (opinions, advice, and personal experiences) rather than fact-based information. However, the subsequent introduction of Google AI Mode, which allows users to interact conversationally with the AI summary, largely eliminates these gains in experience-based content. These results suggest that the effects of AI search depend critically on interface design and types of content. 

---
# EAGT: Echocardiography Augmentation for Generalisability and Transferability 

**Authors**: Soroush Elyasi, Sara Adibzadeh, Nasim Dadashi Serej, Julie Wall, Massoud Zolgharni  

**Link**: [PDF](https://arxiv.org/pdf/2605.16427)  

**Abstract**: Deep learning models for echocardiography segmentation often struggle to generalise across institutions, scanners, and patient populations, where collecting large, consistently annotated datasets is infeasible. Data augmentation is widely used to improve the robustness of deep learning models; however, its role in enhancing cross-dataset generalisability in echocardiography remains insufficiently understood. This study presents a large-scale multi-dataset evaluation of 29 data augmentation techniques and their pairwise combinations for 2D left ventricular segmentation using a U-Net trained on Unity, CAMUS, and EchoNet Dynamic datasets. Each augmentation was explored under several hyperparameter settings and assessed through repeated runs using Dice and IoU in both in-domain and cross-dataset scenarios, with statistical significance quantified via independent t-tests. Results show that anatomically plausible geometric transformations, particularly affine, shift-scale-rotate, perspective, and random horizontal flip, substantially improve cross-dataset performance, whereas aggressive intensity- or artefact-based augmentations often degrade generalisability. Pairwise augmentation combinations outperform individual augmentations and show that moderate flip-centric combinations, especially random horizontal flip with affine, yield consistent gains across most transfer scenarios. These findings provide empirically grounded guidance for designing augmentation policies that enhance the robustness and transferability of echocardiography segmentation models. 

---
# Orthologic for SAT Solving 

**Authors**: Vladislas de Haldat, Simon Guilloud, Viktor Kunčak  

**Link**: [PDF](https://arxiv.org/pdf/2605.16421)  

**Abstract**: We present a new algorithm for deciding formula entailment in orthologic (a sound approximation of classical logic) that avoids the costly preprocessing phase of prior implementations while retaining the same $\mathcal{O}(n^2(1+|A|))$ worst-case complexity. We then introduce a family of synthetic SAT benchmarks based on the observation that, for any formula $\phi$, the equivalence $\phi \leftrightarrow \mathrm{NF}_{\mathrm{OL}}(\phi)$ is a tautology whose Tseitin encoding yields unsatisfiable instances that are hard for state-of-the-art SAT solvers yet have short orthologic proofs. Applied to EPFL arithmetic circuits, our algorithm solves these instances efficiently while Kissat times out on a significant fraction. Finally, we show that using orthologic normalization as a preprocessing step can improve SAT solving time on some hard problems. 

---
# Agentic Pipeline for Self-Synchronized Multiview Joint Angle Monitoring in Uncalibrated Environments 

**Authors**: Juncheng Yu, Lusi A, Haoxuan Xie, Weiming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16419)  

**Abstract**: Kinematic monitoring plays a critical role in long-term rehabilitation for patients with spinal cord injury (SCI), where multi-view markerless motion capture methods have shown significant potential. However, owing to the reliance on calibration and the difficulty of achieving multi-view synchronization, their deployment in patient self-deployed environments remains challenging. In this work, we propose an agentic pipeline for self-synchronized multi-view joint angle monitoring in uncalibrated environments using two cameras without hardware triggers. The Multimodal large language models enable automatic video synchronization and agent-driven self-verification. State-of-the-art monocular 2D pose estimation models are employed to extract candidate poses, where an agent-based selection mechanism is then applied to automatically identify and track the target subject, thereby producing consistent 2D poses in the presence of multiple individuals and occlusions. Such 2D poses are optimized to estimate joint angles from uncalibrated multi-view pose sequences, ensuring interpretability through explicit geometric modeling. Validation against Vicon system demonstrated the strong performance, achieving an MAE of $5.97^\circ \pm 2.36^\circ$ and a Pearson correlation coefficient of $0.962 \pm 0.014$. The proposed method is expected to provide a practical, patient self-deployable system to perform daily kinematic monitoring in uncalibrated home environments. 

---
# Neural Visual Decoding via Cognitive guided Adaptive Blurring and Information Constrained Alignment 

**Authors**: Fan Yin, Chuhang Zheng, Peiliang Gong, Donghai Guan, Qi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16418)  

**Abstract**: EEG-based visual decoding aims to establish a mapping between neural signals and visual semantics. However, it remains constrained by the dual challenges of severe information granularity mismatch and the low signal-to-noise ratio (SNR) of EEG signals. Existing approaches typically treat static visual features, ignoring the dynamic selectivity of human vision and the frequency specificity of neural oscillations. To bridge this gap, we propose CAIA, a Cognitive-guided Adaptive blurring with Information-Constrained Alignment framework for Neural-Visual decoding. On the visual side, it simulates selective attention to adaptively reduce redundancy. Meanwhile, on the EEG side, it leverages neural oscillation priors and the information bottleneck mechanism to enhance SNR. Specifically, we devise a cognitive-dynamics-based adaptive blurring mechanism that dynamically integrates center-biased and saliency-guided visual cues via cross-modal attention. Furthermore, we introduce a distribution-aware boundary calibration loss to robustly rectify alignment bias caused by outlier samples. Moreover, a cognitively-guided information-screening method is proposed to select task-relevant EEG oscillations. Extensive experiments demonstrate that CAIA improves both subject-dependent and subject-independent average Top-1 and Top-5 accuracy in zero-shot brain-to-image retrieval, significantly outperforming prior methods. Our work validates that optimizing visual information density to match neural granularity offers a more interpretable and robust pathway for neural decoding. 

---
# CAVE: A Structured Credit Assignment Approach for Fragmented Visual Evidence Reasoning 

**Authors**: Tengda Guo, Jie Leng, Hanlei Li, Yaoyuan Liang, Qingyue Zhang, Dian Yang, Mingyu Zhang, Yuhua Fu, Shao-Lun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16416)  

**Abstract**: Vision-Language Models (VLMs) have achieved strong performance on general multimodal reasoning, yet remain challenged in integrating nonlocal visual information to support semantically underdetermined visual reasoning. We describe this challenge as Fragmented Visual Reasoning. To this end, we propose Credit Assignment for Visual Evidence (CAVE), a structured process-reward method based on GRPO for interleaved visual reasoning. Specifically, CAVE evaluates the contribution of intermediate steps at the action level via three complementary reasoning process signals: belief update, evidence acquisition, and adaptive focus control, thereby guiding the model to optimize each reasoning action and learn more reliable visual reasoning strategies. Meanwhile, we construct TRACER-Bench, which covers four nonlocal and semantically confusable reasoning dimensions and provides key intermediate evidence to supervise reasoning paths. Experiments demonstrate that CAVE substantially improves performance on tasks requiring fragmented visual evidence integration, covering both public benchmarks and our newly introduced TRACER-Bench, while retaining competitive performance on general multimodal evaluations. Further analyses reveal that CAVE effectively improves the visual reasoning capacity and exhibits stronger robustness under longer-range and deeper cross-region dependencies. 

---
# Reducing Hallucination in Vision-Language Models via Stage-wise Preference Optimization under Distribution Shift 

**Authors**: Qinwu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16411)  

**Abstract**: Hallucination remains a fundamental challenge in vision-language models (VLMs), where autoregressive generation may produce linguistically plausible yet physically inconsistent or visually ungrounded responses due to likelihood maximization under joint probabilistic modeling.
We propose a stage-wise preference optimization framework for hallucination reduction through targeted multimodal data construction. Rather than directly optimizing on generic instruction-following data, our approach progressively constructs hallucination-focused preference pairs near known failure boundaries. The framework emphasizes ambiguous spatial orientation, object relationships, OCR uncertainty, and adversarial false-premise training. Hallucinated negatives are generated through minimally perturbed yet visually inconsistent alternatives, enabling Direct Preference Optimization (DPO) to better separate grounded reasoning from plausible hallucination.
Experiments on open-source benchmarks and real-world multimodal evaluation scenarios demonstrate improved grounding consistency, reduced hallucination, and more informative grounded responses. Cross-model qualitative evaluation further shows that the proposed multimodal LLM DPO framework produces more visually grounded responses than several frontier proprietary VLMs, such as in ambiguous spatial reasoning and adversarial false-premise settings. The results suggest that hallucination may arise not only from limited model capacity, but also from inherent tendencies of autoregressive probabilistic generation to favor linguistically plausible continuations under weak visual grounding. Future work may explore physical consistency modeling, uncertainty-aware multimodal reasoning, and architectural alternatives beyond standard autoregressive decoding. 

---
# Support-Safe Variational Hybrid Filtering for Contact-Mode and Sparse-Law Recovery 

**Authors**: Marios Papamichalis, Regina Ruane  

**Link**: [PDF](https://arxiv.org/pdf/2605.16398)  

**Abstract**: Contact-rich robot dynamics are hybrid: a single observation can match several latent states and contact regimes (free, impact, stick--slip). A standard amortized filter that places no probability on a feasible contact transition will permanently lose the branch the robot actually follows. We introduce VHYDRO, a variational hybrid dynamics learner that prevents this branch loss. At each step, VHYDRO mixes the learned proposal with a feasible transition law before sampling and importance weighting, ensuring that every transition retained by the model-feasible carrier remains covered. VHYDRO jointly infers a continuous latent state and a discrete contact mode, and fits a sparse port-Hamiltonian law to each recovered regime. On top of this, three guarantees connect: support coverage stabilizes filtering, the stabilized filter concentrates the discrete contact posterior on coherent regimes, and mode-pure segments admit sparse port-Hamiltonian recovery. The recovery error separates cleanly into filtering, derivative, mode-impurity, and physics-residual parts. Three empirical findings track the same mechanism. Under heavy occlusion the support-safe filter stays usable while a non-defensive proposal collapses. On ManiSkill demonstrations and on four Sawyer/BridgeData task families the discrete state forms temporally coherent contact-regime segments that the discrete state yields a stronger joint profile across ARI, change-point F1, and segment purity than post-hoc and mode-free baselines. On hybrid systems with known equations the mode-conditioned sparse fit recovers the active physical terms; purely predictive baselines do not. 

---
# Trajectory-Aware Adaptive Inference in Object Detection Models 

**Authors**: Grigorios Papanikolaou, Ioannis Kontopoulos, Giannis Spiliopoulos, Dimitris Zissis, Konstantinos Tserpes  

**Link**: [PDF](https://arxiv.org/pdf/2605.16397)  

**Abstract**: The increasing integration of sensors in autonomous maritime navigation has led to large-scale multimodal datasets, raising challenges in achieving efficient real-time perception. In such systems, object detection and trajectory perception of nearby vessels are tightly coupled, particularly in dynamic environments such as maritime navigation. However, the efficiency of object detection models during inference remains an often-overlooked aspect. To this end, we build upon an existing object detection framework by incorporating GPS trajectory data into the inference process to enable input-adaptive computation. Specifically, we introduce an early-exit mechanism in a YOLOv8-based detector that incorporates motion cues - such as inter-vessel distances. Frames of vessels that are separated by short distances, converging with high speed, are processed using the full model, while only a subset of the network's architecture is activated otherwise. The difficulty degree (or scene complexity) of a frame or set of frames per second is evaluated by leveraging inter-object distance and the rate at which the distance between them decreases. Experimental results demonstrate that this strategy maintains satisfactory detection performance while significantly reducing inference time and computational cost, thus enabling a flexible trade-off between accuracy and efficiency compared to full-model inference. 

---
# Vision Transformer-Conditioned UNet for Domain-Adaptive Semantic Segmentation 

**Authors**: Joel Valdivia Ortega, Tingying Peng, Marion Jasnin  

**Link**: [PDF](https://arxiv.org/pdf/2605.16393)  

**Abstract**: Semantic segmentation is essential for analysing anatomical features in biomedical research, yet a performance gap remains for Vision Transformers (ViTs) in the field, particularly for sparse, fine-structured, and low signal-to-noise targets. We attribute this challenge in part to the lightweight pixel decoders commonly used in promptable ViT models, who may lack the local inductive bias needed for high-precision biomedical masks. We bridge this gap by introducing ViTC-UNet, which conditions a UNet on frozen pre-trained ViT representations through learnable tokens and a two-way attention decoder. This combines ViT global visual priors with the local inductive bias and high-resolution decoding capacity of UNets, while avoiding end-to-end ViT fine-tuning even in cross-domain settings. ViTC-UNet outperforms baseline results in semantic segmentation tasks across MRI and CT modalities, demonstrating that structure-conditioned UNet decoding can efficiently adapt large-scale visual priors to high-complexity biomedical segmentation. 

---
# Overcoming the Intrinsic Performance Limitations of MEMS IMU via Diffusion-Based Generative Learning 

**Authors**: Jiarui Lv, Feng Zhu, Xiaohong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16391)  

**Abstract**: Inertial measurement units (IMUs) are fundamental sensing components in multi-source integrated navigation systems, and their performance directly determines the accuracy and reliability of solutions. However, the precision of low-cost IMUs is inherently constrained by hardware limitations. Recently, generative artificial intelligence has demonstrated remarkable capability in modeling complex data distributions and reconstructing high-fidelity signals. Motivated by this, we propose a diffusion-based generative learning framework for synthesizing high-fidelity virtual IMU data from low-cost IMU measurements. Specifically, a conditional diffusion model based on a U-Net architecture is constructed, where high-grade IMU measurements are utilized as ground-truth priors and low-cost IMU measurements are employed as conditional inputs. The virtual IMU data generated by the model is used for subsequent navigation and localization tasks. Experimental results demonstrate that the generated virtual IMU data significantly outperform the original low-cost IMU measurements in both positioning and attitude estimation. Furthermore, we transfer the model to airborne mapping experiments, where the proposed method produces thinner and more consistent point clouds. Overall, the proposed framework breaks the performance limits of low-cost IMU and demonstrates the potential of diffusion-based generative learning for virtual high-grade IMU data. 

---
# Haptic Rendering of Fractional-Order Viscoelasticity: Passivity and Rendering Fidelity 

**Authors**: Gorkem Gemalmaz, Harun Tolasa, Volkan Patoglu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16389)  

**Abstract**: Haptic rendering of viscoelastic materials that exhibit creep and stress relaxation is crucial for many applications, such as medical training with realistic biological tissue models. Fractional-order viscoelastic models provide an effective means of describing intrinsically time-dependent dynamics with few parameters, as these models can naturally capture memory effects. In this study, we present analyses of passivity and rendering performance for fractional-order viscoelastic models under finite-memory discretization. We derive closed-form expressions to ensure the passivity of haptic rendering with a fractional-order (FO) standard linear solid (SLS) model based on Grunwald-Letnikov derivative under short-memory discretization. We also provide symbolic expressions for the effective stiffness and damping of such FO-SLS models. The resulting passivity conditions constitute a unified framework that generalizes previously reported results for integer-order Kelvin-Voigt, Maxwell, and SLS models, since these results are special cases of the newly derived condition. Furthermore, we provide experimental validations of the theoretical passivity bounds and human-subject evaluations of perceived realism of FO-SLS models. Overall, this study establishes a unified theoretical framework and experimental evaluations for FO viscoelastic rendering under short-memory discretization. 

---
# Stabilizing Temporal Inference Dynamics for Online Surgical Phase Recognition 

**Authors**: Yang Liu, Ning Zhu, Jingjing Peng, Xiwu Chen, Alejandro Granados, Guotai Wang, Sebastien Ourselin  

**Link**: [PDF](https://arxiv.org/pdf/2605.16387)  

**Abstract**: Online Surgical Phase Recognition (SPR) models can reach high frame-wise accuracy, yet their predictions often lack temporal stability, fragmenting workflow understanding and reducing the reliability of downstream assistance. We show that this instability is not random noise but arises from two mechanisms: early misclassifications corrupt temporal feature states and propagate forward to form error cascades, and phase transitions follow evidence-accumulation dynamics whereas most online SPR systems rely on memoryless frame-wise decisions, making them sensitive to transient confidence fluctuations. We propose a unified Train-Inference-Evaluation framework that explicitly stabilizes temporal inference dynamics using model-agnostic, plug-and-play components. For training, the Temporal Error-Cascade (TEC) loss suppresses error onset and mitigates forward error propagation by stabilizing temporal feature evolution. For inference, the Evidence-Gated Transition Predictor (EGTP) enforces evidence-driven state transitions, allowing phase changes only when accumulated evidence exceeds a confidence boundary. For evaluation, we introduce the Temporal Fragmentation Index (TFI), a reliability-aware metric that quantifies instability-induced temporal disagreement beyond conventional frame-wise and token-based measures. Experiments on Cholec80 and AutoLaparo across three representative backbones show that the proposed framework substantially improves temporal stability and reduces prediction fragmentation, while maintaining or modestly improving frame-wise performance. 

---
# Hilbert-Geo: Solving Solid Geometric Problems by Neural-Symbolic Reasoning 

**Authors**: Ruoran Xu, Haoyu Cheng, Bin Dong, Qiufeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16385)  

**Abstract**: Geometric problem solving, as a typical multimodal reasoning problem, has attracted much attention and made great progress recently, however most of works focus on plane geometry while usually fail in solid geometry due to 3D spatial diagrams and complex reasoning. To bridge this gap, we introduce Hilbert-Geo, the first unified formal language framework for solid geometry, including an extensive predicate library and a dedicated theorem bank. Based on this framework, we propose a Parse2Reason method containing two steps of first parsing then reasoning. In the parsing step, we utilize conditional description language (CDL), a formalized language composed of predicates specifically designed to construct geometric conditions, to represent both problem description (natural text) and solid diagrams (visual image). In the reasoning step, we leverage those formal CDL and the theorem bank to perform relational inference and algebraic computation, generating strictly correct, verifiable, and human-readable reasoning processes. Notably, our proposed Hilbert-Geo is also applicable to plane geometry. To advance geometric reasoning, we curate two expert-annotated dataset SolidFGeo2k and PlaneFGeo3k, which are furnished with geometric formal language annotations, solutions and answers. Extensive experiments show that our proposed method achieves the state-of-the-art (SOTA) performance 77.3% in SolidFGeo2k and 84.1% in MathVerse-Solid (one small subset in MathVerse dedicated to solid geometry), substantially outperforming leading MLLMs, such as Gemini-2.5-pro (54.2% on SolidFGeo2k) and GPT-5 (62.9% on MathVerse-Solid). In addition, our method achieves the SOTA accuracy 80.2% in PlaneFGeo3k, demonstrating the generality of the Hilbert-Geo in geometric reasoning. Our code and datasets will be publicly available. 

---
# Mutual Enhancement Between Global Tokens and Patch Tokens: From Theory to Practice 

**Authors**: Xiusheng Huang, Xin Jiang, Jun Zhao, Kang Liu, Yequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16384)  

**Abstract**: Accurate and effective discrete image tokenization is crucial for long image sequence processing. However, current methods rigidly compress all content at a fixed rate, ignoring the variable information density of images and leading to either redundancy or information loss. Inspired by information entropy, we propose TaTok, a Theoretically grounded adaptive image Tokenization framework. We rigorously identify two key drawbacks in existing methods: information insufficiency when reconstructing images with patch tokens alone, and information redundancy among patch tokens. To address these, we introduce global tokens that model mutual information across patch tokens, and a Dynamic Token Filtering (DTF) algorithm based on cumulative conditional entropy to eliminate redundancy. Experiments confirm TaTok's state-of-the-art performance, delivering a 1.3x gFID improvement and 8.7x inference speedup. By allocating tokens according to information richness, TaTok enables more compressed yet accurate image tokenization, offering valuable insights for future research. 

---
# A neurosymbolic Approach with Epistemic Deep Learning for Hierarchical Image Classification 

**Authors**: Ezel Kilicdere, Shireen Kudukkil Manchingal, Fabio Cuzzolin  

**Link**: [PDF](https://arxiv.org/pdf/2605.16383)  

**Abstract**: Deep neural networks achieve high accuracy on image classification tasks. Yet, they often produce overconfident predictions as which fail to express epistemic uncertainty, and frequently violate logical or structural constraints present in the data. These limitations are particularly pronounced in hierarchical classification, where predictions across fine and coarse levels must remain coherent. We propose, for the first time, a unified neurosymbolic and epistemic modelling framework that augments Swin Transformers with focal set reasoning and differentiable fuzzy logic. Rather than treating labels as isolated categories, our method induces data-driven focal sets within the learnt embedding space, which helps capture epistemic uncertainty over multiple plausible fine-grained classes. These focal sets form the basis of a belief-theoretic layer that uses fuzzy membership functions and t-norm conjunctions to encourage consistency between fine- and coarse-grained predictions. A learnable loss further balances calibration, mass regularisation, and logical consistency, allowing the model to adaptively trade off symbolic structure with data-driven evidence. In experiments on hierarchical image classification, our framework maintains accuracy on par with transformer baselines while providing more calibrated and interpretable predictions, reducing overconfidence and enforcing high logical consistency across hierarchical outputs. Our experimental results show that combining focal set reasoning with fuzzy logic provides a practical step toward deep learning models that are both accurate and epistemically aware. 

---
# StreamPro: From Reactive Perception to Proactive Decision-Making in Streaming Video 

**Authors**: Ao Li, Zihan Xiao, Zihao Yue, Boshen Xu, Linli Yao, Jiaze Li, Pei Fu, Jianzhong Ju, Jian Luan, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.16381)  

**Abstract**: Proactive streaming video understanding requires models to continuously process video streams and decide when to respond, rather than merely what to respond. This naturally introduces a decision-making problem under partial observations, where models must balance early prediction against sufficient evidence. However, existing benchmarks largely follow a "see-then-answer" paradigm, where responses are triggered only after explicit evidence appears, effectively reducing proactive reasoning to delayed perception. As a result, they fail to evaluate a model's ability to make timely and reliable decisions under incomplete observations. Moreover, training proactive models is inherently challenging due to the extreme imbalance between silence and response signals in streaming trajectories, as well as the need to jointly optimize response correctness and timing. To address these challenges, we introduce StreamPro-Bench, a new benchmark that evaluates streaming models from three complementary perspectives: Perception Understanding, Temporal Reasoning, and Proactive Agency, where the last measures a model's ability to make early yet reliable decisions under partial observations. We further propose StreamPro, a two-stage training framework for proactive learning. First, we introduce CB-Stream Loss to mitigate the severe supervision imbalance during supervised fine-tuning (SFT). Then, we apply Group Relative Policy Optimization (GRPO) with a multi-grained reward design that involves both turn-level and trajectory-level rewards. Experiments show that StreamPro significantly improves proactive performance. On StreamPro-Bench, it achieves 41.5, substantially outperforming the previous best (10.4), while also maintaining strong performance on real-time streaming benchmarks, achieving 78.9 on StreamingBench-RTVU. 

---
# ReTAMamba: Reliability-Aware Temporal Aggregation with Mamba for Irregular Clinical Time Series Prediction 

**Authors**: Jinwoong Kim, Sangjin Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.16380)  

**Abstract**: Clinical time-series data are difficult to model with methods designed for regular sequences because they exhibit irregular sampling, frequent missing values, and heterogeneous observation patterns across variables. Existing approaches commonly use observation masks and time-gap information, but they do not continuously capture the decaying reliability of past observations or consistently organize multi-resolution information within a coherent temporal context during aggregation. To address these limitations, we propose Reliability-aware Temporal Aggregation with Mamba (ReTAMamba), which reconstructs clinical time series as time-variable token sequences, estimates observation reliability from missingness and elapsed time, and augments interval summaries with statistical descriptors. Chronological Weaving is used to integrate short- and long-term temporal information within a coherent temporal context, and a budgeted token router is applied to constrain sequence length while preserving informative summaries. Experiments on MIMIC-IV, eICU, and PhysioNet 2012 show that ReTAMamba consistently improves AUPRC over strong baselines, with average relative gains of 7.51%, 7.80%, and 10.15%, respectively. Cohort-level and patient-level analyses on eICU further showed that the learned mean decay for more dynamic signals, such as heart rate and blood pressure, was 24.3% larger than that for relatively static signals, such as laboratory test variables. These findings suggest that effective prediction in irregular clinical time series requires modeling not only what was measured, but also when and how it was observed, including information freshness and observation timeliness. 

---
# An Information-Theoretic Criterion for Efficient Data Synthesis 

**Authors**: Hanyu Li, Zhengqi Sun, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.16379)  

**Abstract**: Synthetic data becomes crucial for large language model training, but its effectiveness is highly inconsistent. We provide an information-theoretic account of this inconsistency: synthetic data improves a model only when the generation-training loop is information-open, i.e., shaped by external signals (verifiers, environments, or rubrics) that inject task-relevant information beyond the model's current distribution. When the loop is information-closed (relying on the model's own outputs without such signals), the data processing inequality ensures that task-relevant information can only decrease, making collapse a predicted outcome. Among information-open pipelines, both efficiency and generalization hinge on the meta-level of supervision: a coarser signal such as binary correctness treats all acceptable outputs as equivalent, so the behavior it teaches is not tied to any particular domain or surface form and generalizes naturally across tasks and domains. These observations lead to a guiding thesis: learning preferentially converges to the most information-efficient signal component available, which accelerates learning when that component is the intended one, but causes reward hacking when a spurious pattern happens to be simpler. 

---
# Mixing Times of Glauber Dynamics on Masked Language Models 

**Authors**: Suvadip Sana, Sami Wolf, Neer Mehta, Alina Shah, Aitzaz Shaikh, Janna Goodman, Lionel Levine  

**Link**: [PDF](https://arxiv.org/pdf/2605.16378)  

**Abstract**: Masked language models (MLMs) define local conditional distributions over tokens but do not, in general, correspond to any consistent joint distribution over sequences. This raises a fundamental question: what global distributional behavior is induced when such conditionals are used iteratively for generation? We address this question by modeling iterative masked-token resampling as a Glauber dynamics Markov chain on the discrete space of token sequences. We first show that MLM conditionals are intrinsically incompatible: we introduce a rectangle test that certifies this incompatibility and empirically verify its prevalence across modern MLMs. We then provide a theoretical analysis of the induced Markov chain. Under bounded cross-token influence, we establish a high-temperature contraction result implying $O(n\log n)$ mixing time where $n$ is the sequence length. In contrast, we prove that under a uniform local margin condition, the chain exhibits metastability, with exponentially slow escape from semantic basins at low temperatures. Empirically, we demonstrate a phase transition in mixing behavior as a function of temperature and sequence length, consistent with the theoretical predictions. We further characterize the induced stationary behavior through semantic trajectories, identifying persistent structures such as long-lived traps and recurrent semantic basins, with political content serving as a measurable case study. 

---
# CheckSupport: A Local LLM-Powered Tool for Automated Manuscript Submission Checklist Selection and Completion 

**Authors**: Satvik Tripathi, Don Enwerem, Kevin Song, Kristian Quevada, Jacinta Arnold, Tessa S. Cook  

**Link**: [PDF](https://arxiv.org/pdf/2605.16377)  

**Abstract**: Transparent and standardized reporting is essential for reproducible scientific research, yet adherence to reporting guidelines remains inconsistent because of the manual effort required to select and complete checklists. We present CheckSupport, an open-source, locally deployable system that uses large language models to automate the recommendation of reporting checklists and the evidence-grounded completion of checklists for scientific manuscripts. CheckSupport employs a staged prompting strategy that decomposes reporting workflows into constrained inference tasks, prioritizing faithful extraction over generative text synthesis. All inference is performed locally using instruction-tuned models, preserving data privacy and enabling reproducible, auditable workflows. Evaluated on a corpus of peer-reviewed manuscripts, CheckSupport achieved 90% overall accuracy for checklist recommendations and 88% overall accuracy for item-level completion while operating on CPU-only hardware. On average, the wall-clock time per manuscript was 12.5 seconds, including the checklist recommendation and full checklist completion. These results demonstrate that large language models, when applied as structured inference components, can reduce reporting burden and support more transparent and reproducible scientific reporting across disciplines. 

---
# Lost or Hidden? A Concept-Level Forgetting in Supervised Continual Learning 

**Authors**: Katarzyna Filus, Kamil Faber, Roberto Corizzo, Christopher Kanan  

**Link**: [PDF](https://arxiv.org/pdf/2605.16374)  

**Abstract**: Continual learning studies how models can adapt to new tasks while retaining previously acquired knowledge. Although a broad spectrum of methods has been proposed to mitigate catastrophic forgetting, the field remains predominantly performance-driven, with limited insight into what forgetting actually corresponds to within the vision model's representation space. Prior work has primarily analyzed forgetting through task-level performance or coarse measures of representational drift, without disentangling output-level accessibility from changes in finer-grained internal structure. To this end, we propose a diagnostic framework that leverages Sparse Autoencoders (SAEs) to define a task-anchored latent feature space, enabling analysis of how task-specific information evolves at a finer granularity, where individual SAE latents are treated as concept proxies for recurring and relatively disentangled visual patterns in the model's internal computations. Within this framework, we decompose forgetting into apparent concept deletion, recoverability, and decodability. We show that a large portion of seemingly lost concept-level information can often be recovered under linearity assumption, with concept decodability degrading as more tasks are introduced. Overall, our findings suggest that a significant part of concept-level forgetting can be attributed to changes in the representational accessibility rather than complete information erasure. 

---
# Cross-Source Supervision for Bone Infection Segmentation in Dual-Modality PET-CT 

**Authors**: Zonglin Yang, Xiaolei Diao, Jishizhan Chen, Xiaozhuang Man, Wei Kong, Gen Wen, Pengfei Cheng, Daqian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.16373)  

**Abstract**: Early and accurate diagnosis and lesion localization of bone infections are crucial for clinical treatment. PET-CT integrates anatomical information from CT with metabolic information from PET, making it an important imaging modality for diagnosing bone infections. However, accurate lesion segmentation remains challenging due to indistinct lesion boundaries and inconsistencies in annotations generated by different experts or automated systems. In this work, we investigate multimodal segmentation of bone infections under annotation discrepancy. We develop a bimodal end-to-end segmentation framework that integrates PET metabolic signals and CT bone-window anatomy through an early-fusion multimodal this http URL mitigate performance inflation caused by inter-slice correlation in small datasets, this study discards traditional two-dimensional evaluation methods and implements a rigorous patient-level 3D volumetric evaluation and cross-validation. Furthermore, instead of forcing a singular consensus, we propose a decoupled dual-source learning framework where parallel models are trained on independent expert annotations driven by high-sensitivity and high-specificity clinical intents. Experimental results objectively report performance variations at the patient level (Mean + SD and Mean - SD), demonstrating the effectiveness of multimodal PET-CT fusion. The cross-evaluation matrix quantitatively reveals how models successfully internalize distinct expert diagnostic philosophies, providing a robust, diversity-preserving paradigm for clinical AI deployment in bone infection segmentation. 

---
# SwordBench: Evaluating Orthogonality of Steering Image Representations 

**Authors**: Vladimir Zaigrajew, Dawid Pludowski, Hubert Baniecki, Przemyslaw Biecek  

**Link**: [PDF](https://arxiv.org/pdf/2605.16372)  

**Abstract**: Steering or intervening on model representations at inference time to correct predictions is essential for AI interpretability and safety, yet existing evaluation protocols are limited to ambiguous language modeling tasks. To address this gap, we introduce SwordBench, a benchmark for steering image representations of vision models across multiple backbones and concept removal tasks. Beyond a unified benchmarking suite, we propose new evaluation notions that uncover the second-order effects of orthogonalization among concept activation vectors for pragmatic steering. Specifically, cross-concept robustness measures the stability of concept detection performance across inputs orthogonalized against alternative concepts, and collateral damage quantifies whether steering inadvertently affects model performance on a downstream task for inputs lacking the bias. We find that although a linear support vector machine exhibits superior separability and orthogonality, it fails to achieve zero collateral damage, often trailing sparse autoencoders. In simpler regimes, both standard baselines and optimization-based methods fail to achieve perfect steering. The source code will be made available soon on GitHub. 

---
# GeoSym127K: Scalable Symbolically-verifiable Synthesis for Multimodal Geometric Reasoning 

**Authors**: Jinhao Jing, Zheng Ma, Jinwei Liang, Qiannian Zhao, Shawn Chen, Jing Yang, Por Lip Yee, Prayag Tiwari, Jingjing Bai, Benyou Wang, Lewei Lu, Zhan Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.16371)  

**Abstract**: Large Multimodal Models (LMMs) often struggle with geometric reasoning due to visual hallucinations and a lack of mathematically precise Chain-of-Thought (CoT) data. To address this, we propose the GeoSym Engine, an automated and scalable neuro-symbolic framework. By leveraging a type-conditional grammar and an analytic SymGT Solver, it derives exact symbolic ground truths and seamlessly integrates with a robust rendering pipeline to produce high-precision geometric diagrams. Using this engine, we construct GeoSym127K, a difficulty-stratified dataset featuring 51K high-resolution images, 127K questions with symbolic ground truths, and 55K answer-verified CoT QA pairs. We also introduce GeoSym-Bench, an expert-curated suite of 511 complex samples for rigorous evaluation. Through extensive supervised fine-tuning (SFT), we demonstrate that GeoSym drives concentrated improvements specifically on diagram-dependent and multi-step geometry tasks. Our Qwen3-VL-8B model gains an absolute +22.21% on the MathVerse Vision-Only subset and reaches 61.52% (+6.19% improvement) on WeMath, mitigating long-horizon logic fragmentation and outperforming advanced closed-source models like Doubao-1.8. Furthermore, applying Reinforcement Learning with Verifiable Rewards (RLVR) via GRPO reveals that initializing from structural SFT checkpoints substantially elevates the performance ceiling over zero-shot RL. Driven by deterministic exact-match signals, this showcases the robust scaling potential of our verifiable reasoning synthesis. Datasets and code are available at this https URL and this https URL. 

---
# Fre-Res: Frequency-Residual Video Token Compression for Efficient Video MLLMs 

**Authors**: Yigui Feng, Qinglin Wang, Yang Liu, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16366)  

**Abstract**: Video MLLMs face a persistent tension between spatial fidelity and temporal coverage: preserving fine-grained visual details requires many spatial tokens, while capturing short-lived events requires dense temporal sampling. We propose \textbf{Fre-Res}, a budget-adaptive dual-track video-token compression framework that separates these two forms of evidence. Fre-Res preserves sparse high-fidelity spatial anchors and represents dense temporal evolution through compact residual-frequency tokens. Specifically, it applies temporal 1D-DCT to inter-frame residual trajectories in vision-latent space, where we observe strong low-frequency concentration. To align frequency-domain dynamics with native visual embeddings, Fre-Res introduces a Spatial-Guided Absorber that injects temporal residual information into spatially corresponding anchor tokens. Across fine-grained short-video and long-video reasoning benchmarks, Fre-Res achieves a favorable accuracy--efficiency trade-off, matching or approaching full-token performance while substantially reducing visual-token length. Extensive ablations further show that temporal-frequency residuals preserve causal transition cues, while spatial anchors remain essential for fine-grained object and layout reasoning. 

---
# WASIL: In-the-Wild Arabic Spoken Interactions with LLMs 

**Authors**: Zien Sheikh Ali, Hamdy Mubarak, Soon-Gyo Jung, Hunzalah Hassan Bhatti, Firoj Alam, Shammur Absar Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2605.16364)  

**Abstract**: Large Language Models (LLMs) voice assistants are commonly built as cascaded Automatic Speech recognition (ASR) to LLM systems, where recognition errors can distort user intent. Dislikes may also arise from ambiguous, out-of-domain, or non-request turns, making it hard to isolate ASR effects. We release WASIL (it denotes connection or linking in Arabic): in-the-wild Arabic spoken interaction prompts with audio, ASR hypotheses, assistant responses, and explicit like/dislike feedback (8,529 turns; 14.2% dislikes), plus a 2,000-turn test set covering Modern Standard Arabic (MSA) and four major dialects with their labels. We provide low-cost gold transcripts via multi-ASR agreement-guided post-editing and annotate answerability (answerable, ambiguous/needs-clarification, unsupported, not-a-request/noise) to separate intrinsic unanswerability from ASR-induced degradation. Finally, we describe scalable reference-free evaluation of responses from ASR vs. gold transcripts using multi-judge LLM scoring. 

---
# When Is Rank-1 Steering Cheap? Geometry, Granularity, and Budgeted Search 

**Authors**: John T. Robertson, Jianing Zhu, Haris Vikalo, Zhangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16362)  

**Abstract**: Activation steering offers a lightweight way to control LLMs without retraining, but its effectiveness varies sharply across concepts. Prior work often reads this variability as evidence that many concepts are not captured by a single steering direction. We argue instead that much of it reflects search difficulty: a useful rank-1 intervention often exists, but finding it can be expensive.
We formalize rank-1 steering as a budget-constrained optimization over intervention layer and coefficient. Across concepts and model families, prompt-boundary directional alignment predicts where effective interventions occur, enabling geometry-guided search that reaches high utility with substantially fewer evaluations, reducing the trials needed to recover 95\% of best-found utility by 39.8\% on average across three model families. To explain why some concepts remain expensive even under better search, we introduce \emph{concept granularity}, a measure of directional heterogeneity across contrastive contexts. Granularity distinguishes concepts whose difference vectors share a stable global direction from those where prompts agree locally within each input but the utility-maximizing direction rotates systematically across inputs. Higher granularity is associated with slower convergence and lower best-found performance (Pearson $r{=}0.44$ with trials-to-95\%, $r{=}{-}0.46$ with best-found utility, both $p<0.001$).
We present \textit{GRACE}, a Granularity- and Representation-Aware Concept Engineering framework that uses activation geometry to diagnose the dominant source of steering difficulty, select the appropriate remedy, and allocate optimization effort efficiently. Our results shift the frame from ``\textit{when does rank-1 fail?}'' to ``\textit{when is rank-1 cheap and stable?}'', turning activation geometry from a descriptive tool into an actionable prior for LLM control. 

---
# TailedTS: Benchmark Dataset for Heavy-Tailed Time Series Prediction and Periodicity Quantification 

**Authors**: Xinyu Chen, HanQin Cai, Lijun Ding, Jinhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16361)  

**Abstract**: We present TailedTS, a large-scale benchmark dataset derived from Wikipedia hourly page view observations throughout 2024, specifically designed to test time series forecasting models under heavy-tailed, zero-inflated, and non-Gaussian conditions. The dataset comprises approximately 24.69 billion data points spanning roughly 3 million unique Wikipedia pages per month, stored in high-efficiency Apache Parquet format. Wikipedia traffic follows a pronounced power-law distribution where roughly 5% of pages account for over 70% of total page views, creating a natural and rigorous testbed for model robustness against extreme volatility that are absent from or underrepresented in existing benchmarks such as M4, M5, and UCI electricity datasets. TailedTS enables several research tasks. First, we introduce a periodicity quantification framework based on sparse autoregression with sparsity and non-negativity constraints, revealing that frequently-viewed pages exhibit significantly weaker periodic structure than their less-viewed counterparts, showing direct implications for server allocation and traffic forecasting on large digital platforms. Second, we provide standardized prediction benchmarks evaluated under a suite of non-Gaussian loss functions, including $\ell_1$-norm, Huber, quantile, and $\ell_p$-norm losses, demonstrating that standard Gaussian-based estimators degrade substantially on high-volume page categories, while robust alternatives provide consistent gains across all traffic scales. TailedTS is publicly available at this https URL. 

---
# ProxyKV: Cross-Model Proxy Pruning for Efficient Long-Context LLM Inference 

**Authors**: Junjie Li, Jiong Lou, Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.16360)  

**Abstract**: Efficient long-context inference in Large Language Models (LLMs) is severely constrained by the Key-Value (KV) cache memory wall, yet existing pruning methods force a choice between
low-latency heuristics that sacrifice precision and high-precision reconstruction methods that incur prohibitive prefilling overhead. To bridge this scoring-cost--accuracy gap, we propose
ProxyKV, a cross-model proxy pruning framework that offloads importance scoring to a lightweight intra-family Small-Model Proxy executed asynchronously to the Large-Model Target. To bridge
the architectural gap between heterogeneous models, we design the HybridAxialMapper, which disentangles temporal feature extraction from cross-head alignment, together with a
Multi-Granularity Hybrid Loss that shifts the learning objective from rigid regression to relative ranking consistency. Across the Llama-3.1, Qwen-2.5, and Qwen-3 families spanning targets
from 7B up to 32B parameters on LongBench, SCBench, and RULER, ProxyKV matches KVZip on aggregate (recovering $\sim$$98.7\%$ of its mean accuracy) while delivering up to a $3.21\times$
prefilling speedup on Llama-3.1-8B (dual-GPU; $\sim$$1.5\times$ shared single-GPU) and sustaining the speedup at contexts up to 170k tokens on Qwen-2.5-7B. 

---
# How Many Visual Tokens Do Multimodal Language Models Need? Scaling Visual Token Pruning with F^3A 

**Authors**: YiJie Huang, Yiqun Zhang, Zhuoyue Jia, Xiaocui Yang, Junzhao Huang, Zihan Wang, Shi Feng, Daling Wang, Yifei Zhang, Yongkang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16359)  

**Abstract**: Vision-language models improve perception by feeding increasingly long visual token sequences into language backbones, but the resulting inference cost raises a basic scaling question: as multimodal models grow, how many visual tokens are actually needed, and how should they be allocated under a fixed visual token budget? Existing training-free pruning methods typically answer this with one-shot proxies such as decoder attention, visual similarity, or conditional diversity. We argue that visual token pruning is better viewed as task-conditioned evidence search, especially under aggressive compression and across model scales. We propose F^3A, a training-free router for visual token pruning that operates before the language model consumes image tokens. F^3A builds lightweight question-conditioned cues, matches them to visual-grid tokens through frozen sparse sensing heads, and allocates a fixed vision token budget via coarse evidence localization, local refinement, coverage-preserving competition, and recovery of under-covered regions. It requires no model training, no extra LLM forward pass and preserves the original multimodal prompting and decoding pipeline. 

---
# LEAF: A Living Benchmark for Event-Augmented Forecasting 

**Authors**: Mingtian Tan, Mihir Parmar, Palash Goyal, Chun-Liang Li, Nanyun Peng, Thomas Hartvigsen, Jinsung Yoon, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2605.16358)  

**Abstract**: Large Language Models (LLMs) are increasingly applied to forecasting. To evaluate this capability while mitigating pre-training data contamination, several living benchmarks have been proposed. However, existing benchmarks either lack the multidimensional events essential for accurate forecasting due to data scarcity, or focus on relatively closed environments. To assess the predictive capabilities of LLMs in complex, real-world scenarios, we propose LEAF, the first living benchmark for event-augmented forecasting tasks, including future event probabilities, trend and time series forecasting. LEAF utilizes a recursive retrieval agent system paired with dual-agent cross-validation to provide comprehensive and relevant auxiliary text for forecasting. Evaluating state-of-the-art proprietary and open-weight LLMs, we find that these models can leverage signals extracted from complex events to enhance predictive performance. In the stock domain, we find that LLMs achieve better performance on equities they confidently identify as more predictable. Furthermore, the events demonstrate a strong correlation with the target equities. To this end, LEAF provides a necessary, dynamically updating testbed to continuously track and drive progress in event-driven forecasting tasks. 

---
# Learning Displacement-Aware WiFi Representations for Weakly Supervised Relative Localization 

**Authors**: Tzu-Ti Wei, Po-Cheng Chen, Yu-Chee Tseng, Jen-Jee Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.16357)  

**Abstract**: WiFi fingerprint-based indoor localization has been widely studied, but most existing approaches focus on absolute positioning and rely on dense coordinate annotations, which are costly to obtain at scale. In this paper, we study a fundamentally different problem: relative localization, where the goal is to directly estimate the displacement between two WiFi fingerprint traces without predicting their absolute positions. To reduce annotation overhead, we adopt weak supervision in the form of stepwise motion vectors obtained from inertial sensing. We propose Intersection Pathway (IP), a cross-modal learning framework that aligns fingerprint traces (f-traces) and displacement traces (d-traces) in a shared latent space. The key idea is to enforce an additive structure in the latent space, such that latent addition and subtraction correspond to physical motion composition, enabling direct relative-displacement inference. Experiments on a synthesized dataset derived from real measurements demonstrate that the proposed method learns displacement-aware WiFi representations and achieves accurate relative localization across varying displacement ranges. Furthermore, the learned model can be extended to few-shot absolute localization with sparse anchors. 

---
# Augmenting Human Evaluation with LLM Judges: How Many Human Reviews Do You Need? 

**Authors**: Jane Paik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.16354)  

**Abstract**: Large language models (LLMs) are increasingly used as automated evaluators of AI systems, including in high-stakes applications. In this role, LLMs are used to generate judgments about the quality, appropriateness, or even safety of model outputs. This approach is motivated by practical constraints. Expert human ratings are costly and difficult to scale, whereas LLM ratings can be produced quickly at low cost. However, current approaches to deploying LLM evaluators are ad hoc, typically limited to reporting agreement metrics between human and LLM judges as a justification for substitution of human ratings, and lack a formal basis for study design. This paper (1) shifts the role of the LLM judge from substitutive to auxiliary, and (2) formulates the LLM-as-a-judge paradigm as one of augmenting human evaluation through a two-stage sampling design, where LLM evaluations are measured for all observations at the first stage and human ratings are partially observed for a subsample at the second stage. We propose to use a doubly robust estimator from the missing data literature, which takes advantage of the robustness property against the prediction model, since the missingness model is known by design. Using the asymptotic variance of this estimator, we propose how sample sizes of human and LLM ratings can be determined to achieve a targeted level of power. We also show that a study can be efficiently designed by allocating more human ratings for types of evaluations where the predictability of LLM ratings is not high. To the best of our knowledge, there is very little guidance on how much human oversight should be retained when validating benchmarks. 

---
# StrLoRA: Towards Streaming Continual Visual Instruction Tuning for MLLMs 

**Authors**: Chang Che, Ziqi Wang, Hui Ma, Cheems Wang, Zenglin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.16353)  

**Abstract**: Continual Visual Instruction Tuning (CVIT) enables Multimodal Large Language Models to incrementally acquire new abilities. However, existing CVIT methods operate under a restrictive task-incremental setting, where each training phase corresponds to a single, predefined task. This does not reflect real-world conditions, where data arrives as a continuous stream of interleaved and dynamically evolving tasks. To bridge this gap, we introduce Streaming CVIT (StrCVIT), a more general and realistic setting where models learn from a stream of data chunks containing a dynamic mixture of tasks. In StrCVIT, a model must simultaneously acquire new abilities, reinforce recurring abilities, and mitigate forgetting. Existing CVIT methods fail here as they cannot reliably distinguish or adapt to the heterogeneous task samples within each chunk. We therefore propose StrLoRA, a regularized two-stage expert routing framework. StrLoRA first performs task-aware expert selection using the textual instruction to activate a sparse subset of relevant experts, reducing cross-task interference. It then applies token-wise expert weighting within this subset, where contribution weights are computed via cross-modal attention between local visual tokens and the global instruction representation. To maintain stability across the non-stationary stream, a routing-stability regularization aligns current routing distributions with a historical exponential moving average reference. Extensive experiments on a newly developed StrCVIT benchmark show that StrLoRA substantially outperforms existing methods, effectively enhancing model's abilities from continuously evolving data streams. 

---
# LARGER: Lexically Anchored Repository Graph Exploration and Retrieval 

**Authors**: Yuntong Hu, Tongli Su, Liang Zhao, Bowen Zhu, Hasibul Haque  

**Link**: [PDF](https://arxiv.org/pdf/2605.16352)  

**Abstract**: Repository-level coding agents must first localize the files and symbols relevant to a task; failures at this stage can cascade across downstream objectives ranging from patch generation to test writing and codebase question answering. Existing agents navigate repositories primarily through lexical search, often missing structural relations such as imports, call chains, type hierarchies, and code-test links. Graph-based retrieval can recover such dependencies, but existing approaches often require separate graph tools or traversal stages that fragment the agent's interaction loop. We formalize repository context localization as Lexically Anchored Structural Localization, where success depends on turning lexical matches into high-precision structural entry points and exposing the most useful confidence-filtered local neighborhoods within the agent's existing search loop. We introduce LARGER (Lexically Anchored Repository Graph Exploration and Retrieval), a lexically anchored active-set retrieval framework that starts from lexical matches, aligns them to graph anchors, and performs confidence-filtered local expansion within the agent's existing search loop. LARGER integrates directly into existing CLI coding agents without requiring external graph databases or specialized graph interfaces. Across four benchmarks spanning localization, test generation, and codebase understanding, LARGER improves file-level Acc@5 on LocBench by +13.9 points with tuned hyperparameters and still gains +11.8 points with fixed hyperparameters over the strongest baseline, while delivering consistent gains on MuLocBench, SWE-Atlas Test Writing, and SWE-Atlas Codebase QA. 

---
# PIMSM: Physics-Informed Multi-Scale Mamba for Stable Neural Representations under Distribution Shift 

**Authors**: Sangyoon Bae, Shinjae Yoo, Jiook Cha  

**Link**: [PDF](https://arxiv.org/pdf/2605.16351)  

**Abstract**: Scientific foundation models are expected to reuse representations under changes in dataset, acquisition protocol, and deployment domain, yet many sequence backbones treat scientific temporal structure as an unconstrained pattern to be fitted. We argue that this misses a central property of natural dynamical systems: neural and atmospheric time series are organized by interacting processes across multiple physical timescales, and failure to preserve this multiscale structure contributes to brittleness under distribution shift. We formalize this failure mode as temporal kernel mismatch, where a model fits in-distribution dynamics with an effective memory policy that is not anchored to the signal's physical timescales, leading to representation drift and degraded transfer. We propose Physics-Informed Multi-Scale Mamba (PIMSM), a state-space architecture that maps spectrum-estimated transition points between frequency regimes (knee frequencies) to scale-specific discretization parameters and anchors them to acquisition time units. On Human Connectome Project fMRI, PIMSM improves robustness and representation stability under severe temporal-context truncation, extreme low-resource transfer, and resting-state-to-task-state generalization. Without modality-specific adaptation, the same architecture also attains the lowest variable-wise MAE across all reported horizons and variables on Weather-5K held-out-station spatial out-of-distribution forecasting. These results support temporal-scale alignment as a practical inductive bias for scientific foundation models that must preserve structure, not only fit correlations, under deployment shift. 

---
# Federated Nested Learning: Collaborative Training of Self-Referential Memories for Test-Time Adaptation 

**Authors**: Hong Chen, Pengcheng Wu, Yuanguo Lin, Peilin Zhao, Xiuze Zhou, Fan Lin, Han Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16350)  

**Abstract**: We rethink Federated Learning (FL) from a nested learning perspective, framing the core challenge as how to collaboratively learn optimization rules, not just static models, to tackle Non-IID client data. To address this, we propose Federated Nested Learning (FedNL), a novel framework that reformulates FL as a three-level nested optimization system. FedNL embeds Titans-based linear attention into FL, enabling clients to perform lightweight, zero-shot test-time adaptation by treating a delta rule as an online gradient step. Experiments on Non-IID MMLU and long-context benchmarks show that FedNL achieves competitive performance in short-context reasoning, enhances the performance of long-context retrieval and streaming Cross-Entropy, and maintains constant inference memory. 

---
# Flow-Direct: Feedback-Efficient and Reusable Guidance for Flow Models via Non-Parametric Guidance Field 

**Authors**: Kim Yong Tan, Yueming Lyu, Ivor Tsang, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2605.16348)  

**Abstract**: Training-free guidance enables pre-trained diffusion and flow models to optimize application-specific objectives using feedback from external black-box reward functions. However, existing methods are feedback-inefficient because reward feedback is used only transiently to inform a localized gradient approximation or a discrete search decision, and is subsequently discarded. To address this limitation, we propose Flow-Direct, a framework that guides the generation process via a persistent guidance field. Theoretically, this guidance field is analytically derived from the log-density ratio between the base and reward-weighted target distributions; it transports the pre-trained distribution to the target distribution. In practice, the field is implemented as a non-parametric estimator constructed from all accumulated reward-evaluated samples. As more samples are collected during optimization, this empirical guidance field becomes increasingly accurate. This persistent formulation yields two major advantages. First, Flow-Direct is highly feedback-efficient: because every evaluated sample is used to refine the global guidance field, no reward information is wasted. Second, the framework is naturally reusable: once optimization is complete, the collected dataset defines a reusable guidance field for generating novel target samples without additional reward evaluations, and distinct guidance fields can be combined to generate samples that simultaneously satisfy multiple objectives. 

---
# PropGuard: Safeguarding LLM-MAS via Propagation-Aware Exploration and Remediation 

**Authors**: Bingyu Yan, Xiaoming Zhang, Jinyu Hou, Chaozhuo Li, Ziyi Zhou, Xiaozhe Zhang, Litian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16346)  

**Abstract**: LLM-based multi-agent systems (LLM-MAS) have become a promising paradigm for solving complex tasks through role specialization, tool use, memory, and collaborative reasoning. However, these interactions create new security risks that malicious instructions injected through messages, tools, or memories can propagate across agents and rounds, causing system-level compromise. Existing defenses largely rely on local filtering or graph-based anomaly detection, but they often fail to trace fine-grained propagation paths or remediate contaminated states without disrupting benign collaboration. We propose PropGuard, a propagation-aware framework for safeguarding LLM-MAS. PropGuard constructs a dual-view spatio-temporal graph that combines response-centric risk estimation with full-state evidence preservation. Guided by these risk priors, a GE-GRPO trained inspector sequentially explores the full-state graph to recover compact suspicious propagation subgraphs. PropGuard then verifies harmful propagation through subgraph-aware diagnosis and applies source-guided remediation to correct upstream contamination and replay affected downstream interactions. Experiments across four communication architectures and five attack settings demonstrate that PropGuard consistently lowers attack success while maintaining high task-level defense success, achieving a favorable effectiveness--efficiency trade-off. 

---
# Goal-Conditioned Supervised Learning for LLM Fine-Tuning 

**Authors**: Shijun Li, Kaiwen Dong, Xiang Gao, Joydeep Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2605.16345)  

**Abstract**: Large language models often require fine-tuning to better align their behavior with user intent at deployment. Existing approaches are commonly divided into online and offline paradigms. Online methods, such as RL-based alignment, can directly optimize outcome quality but typically rely on external reward models and iterative rollouts, making them costly and difficult to deploy in many cases. Offline methods are more efficient, but prevailing approaches such as supervised fine-tuning (SFT) and direct preference optimization (DPO) remain limited: SFT typically collapses graded feedback into binary supervision, while DPO depends on paired preference data that is often unavailable or expensive to construct.
In this paper, we propose goal-conditioned supervised learning (GCSL) as an offline fine-tuning framework for LLMs. Our core idea is to treat feedback signals directly as an explicit goal and train the model, purely through supervised learning, to generate responses that achieve that goal. To better exploit graded feedback, we further introduce a novel goal formulation that defines learning as consistently pursuing outcomes above a target quality threshold, rather than imitating samples from a selected high-quality subset. This design mitigates the bounded-learning effect of SFT and classic GCSL by explicitly guiding the model to learn the directional progression of quality. We also propose natural-language goal representations to better leverage the semantic understanding and reasoning capabilities of LLMs. We evaluate our method on three tasks: non-toxic generation, code generation, and LLM for recommendation. Results show that our approach consistently outperforms standard offline fine-tuning baselines while retaining the efficiency, scalability, and simple data requirements of supervised learning. 

---
# LoopQ: Quantization for Recursive Transformers 

**Authors**: Rui Fang, Hsi-Wen Chen, Ming-Syan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.16343)  

**Abstract**: Looped language models (LoopLMs) improve parameter efficiency by recursively reusing Transformer blocks, enabling deeper computation under a fixed model size. However, this reuse makes LoopLMs more fragile under post-training quantization (PTQ). We present the first systematic study of quantization in LoopLMs and identify three challenges: distribution shift across roles, state reuse across loop transitions, and recursive error accumulation. To address these challenges, we propose LoopQ, a loop-aware PTQ framework that preserves a shared quantized backbone while introducing lightweight adaptations. LoopQ combines activation scaling, selective transformation, cross-loop state alignment, and trajectory-aware optimization to reduce distributional mismatch within loops and error accumulation across loops. Experiments across seven benchmarks show that, under W4A4 quantization, LoopQ improves average downstream accuracy by 68.8% and reduces average perplexity by 87.7% compared with the strongest static PTQ baseline. 

---
# DACA-GRPO: Denoising-Aware Credit Assignment for Reinforcement Learning in Diffusion Language Models 

**Authors**: Amin Karimi Monsefi, Dominic Culver, Nikhil Bhendawade, Lokesh Boominathan, Manuel R. Ciosici, Yizhe Zhang, Irina Belousova  

**Link**: [PDF](https://arxiv.org/pdf/2605.16342)  

**Abstract**: Diffusion large language models are a compelling alternative to autoregressive models, yet existing RL methods for diffusion treat all denoising steps as equally important and rely on biased, high-variance likelihood estimates. We identify two fundamental weaknesses: the absence of temporal credit assignment across the denoising trajectory, and the systematic bias of mean-field likelihood estimates used for policy optimization. To address these, we propose Denoising-Aware Credit Assignment for GRPO (DACA-GRPO), a lightweight, plug-and-play enhancement for any GRPO-style trainer. DACA-GRPO introduces two complementary mechanisms: Denoising Progress Scores, which extract per-token importance weights from intermediate predictions at no additional forward cost, and Stratified Masking Likelihood, which partitions token positions into strata so that each token is predicted with most of the sequence as context, reducing the mean-field bias. Applied on top of three GRPO base methods, DACA-GRPO achieves consistent improvements across seven benchmarks spanning mathematical reasoning, code generation, constraint satisfaction, and constrained generation, with gains of up to 5.6pp on math reasoning, 7.4pp on code generation, 36.3pp on constraint satisfaction, and 5.9pp on JSON schema adherence. 

---
# Detecting Verbatim LLM Copy-Paste in Homework 

**Authors**: Aizierjiang Aiersilan  

**Link**: [PDF](https://arxiv.org/pdf/2605.16336)  

**Abstract**: Large language models (LLMs) have made fluent essay writing, code drafting, and quiz answering instantly available to students at every level, from secondary school through graduate study. Many educators do not object to LLM use \emph{per~se}; what they need to detect is the case in which a student pastes the assignment prompt into a chatbot and submits the model's reply verbatim, without engaging with the work. Existing post-hoc AI-text detectors remain unreliable and have been shown to penalise non-native English writers, while output-side watermarks require cooperation from the model provider. We propose an alternative that the educator controls directly: an input-side watermark in which an invisible instruction is embedded inside the visible assignment prompt itself. An LLM that ingests the prompt verbatim quietly reads the hidden instruction and writes a tell-tale signature into its reply, exposing the copy-and-paste pathway specifically. We describe SteganoPrompt, a single-page, zero-dependency web tool that encodes an arbitrary printable-ASCII payload into the deprecated Unicode Tags block (\texttt{U+E0000}--\texttt{U+E007F}). The encoded string is visually identical to the original, survives common copy-paste channels (Word, Google Docs, PDF, Markdown, Slack, e-mail, the major learning-management systems), and is reliably tokenized by frontier models. We evaluate compliance across seven LLM families and a representative set of educational content channels. The work is informed by my experience as a graduate teaching assistant for an undergraduate software engineering course at the George Washington University. The tool is released under the MIT licence at \url{this https URL}. 

---
# Retrieval and competition: how a protein foundation model starts a protein 

**Authors**: Piotr Jedryszek, Oliver M. Crook  

**Link**: [PDF](https://arxiv.org/pdf/2605.16331)  

**Abstract**: Protein language models are increasingly used to guide experimental and clinical decisions, yet it is often unclear whether a confident prediction reflects recognition of biological evidence or retrieval of a statistical default. We examine this distinction for a near-universal biological rule, that proteins begin with methionine, by tracing the computational pathway through which ESM2-8M produces this prediction. The model does not detect methionine at the masked position. Instead, it retrieves a methionine-favouring signal from a reference representation at the beginning-of-sequence token via a position-specific query assembled across layers, with the final output emerging through competition with context-dependent circuits. To understand how positional information reaches the readout, we introduce a norm-direction decomposition of attention scores within rotary frequency bands. Positional encoding operates through coupled changes in query norm and angular alignment distributed across these bands. On sequences whose true N-terminus is not methionine, where the biological question matters, the model predicts methionine anyway. This is not a correct prediction produced by an unexpected mechanism, but the output of a positional-prior retrieval circuit that matches the statistical average and fails where biology diverges from it. Distinguishing the two requires resolution at the level of individual circuits, frequency bands, and query composition, suggesting that mechanistic verification will be necessary, and challenging, for predictions where the biological stakes are higher. Even for the simplest biological rule, the model's prediction is mediated by a distributed computational circuit rather than direct recognition, suggesting that increasing task complexity will further obscure the relationship between model confidence and underlying biological evidence. 

---
# Differentiable Optimization Layered Safety-Critical Control for Risk-Aware Navigation via Conformal Prediction 

**Authors**: Jinyang Dong, Shizhen Wu, Yongchun Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16327)  

**Abstract**: Risk-aware navigation in unknown environments is a fundamental challenge for autonomous vehicles operating in complex urban systems. To address this issue, this paper presents a differentiable optimization layered safety-critical control method based on conformal prediction. First, to handle uncertainties arising from sensor noise, the conformal prediction method is employed to generate risk-aware obstacle ellipsoids around an elliptical-shaped robot. Second, two nested differentiable optimization layers are introduced to build the control barrier functions for obstacle avoidance and feasibility guarantee, respectively. Then, a quadratic program based safety-critical control law is proposed to integrate the above control barrier function constraints as well as input constraints. In the end, the effectiveness of the proposed framework is demonstrated through numerical simulations. 

---
# A Machine Learning Framework for EEG-Based Prediction of Treatment Efficacy in Chronic Neck Pain 

**Authors**: Xiru Wang, Aiden Li, Hongzhao Tan, Stevie Foglia, Aimee Nelson, Zhen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16326)  

**Abstract**: Chronic neck pain is a leading cause of disability worldwide, and current treatment selection remains largely trial and error. We present a machine learning framework that uses electroencephalography to predict treatment efficacy in patients with chronic neck pain, with the goal of supporting individualized therapy and reducing the burden on healthcare systems. The framework centers on a rigorous data preprocessing stage tailored to the characteristics of each EEG recording type. For resting-state EEG, the preprocessing pipeline comprises baseline signal removal, bad channel identification and exclusion, re-referencing, bandpass and notch filtering, Independent Component Analysis, and power spectral density analysis. For motor execution and motor imagery recordings, the same initial steps are applied, after which signals are aligned to trigger events so that event-related desynchronization (ERD) and event-related synchronization (ERS) can be quantified. Synchronously recorded electromyography data are bandpass filtered and smoothed with a moving average, then correlated with the corresponding EEG channels to characterize the EEG EMG relationship during attempted movement. In parallel, we performed an extensive literature review of machine learning models applied to clinical EEG (763 records initially screened, 16 patient and 47 healthy-control studies retained), to inform the post-processing strategy. Through this combined preprocessing and review effort, we aim to develop a robust predictive model that can support personalized healthcare strategies in chronic pain management. 

---
# Phase Transitions in Driven Informational Systems: A Two-Field Perspective on Learning Theory and Non-Equilibrium Chemistry 

**Authors**: Truong Xuan Khanh  

**Link**: [PDF](https://arxiv.org/pdf/2605.16325)  

**Abstract**: Phase-transition phenomena in deep learning (grokking, emergent capabilities, and ontological reorganization under context shift) have been studied through several lenses, including representational compression, singular learning theory, and information-theoretic progress measures. Independently, non-equilibrium statistical physics has identified phase transitions in driven chemical reaction networks underlying prebiotic selection, with empirical signatures that are difficult to reproduce within single-field gradient accounts.
We propose a perspective in which both classes of phenomena admit a common description as driven informational systems: stochastic processes governed by two gradient fields, an entropy production rate Sigma and an information quasi-potential Phi_I := -ln p*, where p* is the stationary density. Within this framework we introduce two candidate order parameters: an adversarial breakdown threshold alpha_dagger and a self-referential coupling threshold kappa_c.
The joint scaling of (alpha_dagger, kappa_c) defines a candidate universality class with exponents (gamma_1, gamma_2). We outline the geometric structure of this framework, identify falsifiable predictions distinguishing it from single-field alternatives, and show consistency with recent empirical findings (2024--2026) on alignment transitions, adversarial breakdown scaling, and partial introspection in large language models. 

---
# AdaGraph: A Graph-Native Clustering Algorithm That Overcomes the Curse of Dimensionality and Enables Scientific Discovery 

**Authors**: Ahmed Elmahdi  

**Link**: [PDF](https://arxiv.org/pdf/2605.16320)  

**Abstract**: We present AdaGraph, a graph-native clustering algorithm born from the Structure-Centric Machine Learning (SC-ML) paradigm -- a new field of unsupervised learning that replaces geometry-centric (distance-based) computation with structure-centric (topology-based) computation, fundamentally dissolving the curse of dimensionality. AdaGraph operates entirely within the kNN graph topology, a representation that retains meaningful relational structure in arbitrarily high dimensions where Euclidean distance metrics become uninformative. AdaGraph requires no a priori specification of the number of clusters k, handles noise natively, and scales via the SLCD (Sample-Learn-Calibrate-Deploy) prototype-deployment framework. As its unsupervised tuning objective, AdaGraph pairs with Graph-SCOPE, the topology-based cluster validity index introduced as a separate SC-ML contribution. On 10 synthetic benchmarks spanning d=10 to d=5000, Graph-SCOPE achieves mean ARI=0.900 and correctly selects k on 9/10 datasets -- outperforming Silhouette, Davies-Bouldin, and Calinski-Harabasz -- while maintaining Kendall tau >= 0.92 with ground-truth cluster quality across all dimensionalities (Silhouette: tau ~= 0.46). We validate AdaGraph across three scientific domains: (1) gene co-expression discovery in hepatocellular carcinoma (GSE14520, 10,000 genes, 488 patients, no dimensionality reduction), where AdaGraph identifies condition-specific gene modules that WGCNA, ICA, NMF, and Spectral Biclustering fail to resolve; (2) natural language text clustering, where AdaGraph achieves ARI=0.751 on 20NG-6cat versus HDBSCAN's 0.464 (62% relative improvement); (3) materials science clustering of superconductors (145-dimensional Magpie features), perovskites, and JARVIS-DFT materials, where AdaGraph achieves the highest Graph-SCOPE on all three datasets. 

---
# A Structural Threshold in Decision Capacity Governs Collapse in Self-Play Reinforcement Learning 

**Authors**: Arahan Kujur  

**Link**: [PDF](https://arxiv.org/pdf/2605.16315)  

**Abstract**: We show that a threshold in decision capacity determines whether self-play reinforcement learning agents collapse under asymmetric rule perturbations. Across poker variants, matrix games, a dice game, and multiple learning algorithms, eliminating all positive-reach contingent decisions causes rapid convergence to a deterministic exploitation attractor, a fixed point at near-maximal loss. Preserving even a single positive-reach contingent decision point prevents this collapse. A frozen baseline and fixed-opponent control confirm that the mechanism is co-adaptation under constraint, not the perturbation itself. The phenomenon is timing-invariant, fully reversible upon action restoration, and intensifies under function approximation. These results establish a sharp threshold at zero reach-weighted contingent action capacity, with severity scaling continuously via reach-weighted capacity in the tested domains. 

---
# When Actions Disappear: Adversarial Action Removal in Self-Play Reinforcement Learning 

**Authors**: Arahan Kujur  

**Link**: [PDF](https://arxiv.org/pdf/2605.16312)  

**Abstract**: We study adversarial action masking in self-play reinforcement learning: an attacker selectively removes legal actions from a victim's action set. Unlike observation or action perturbations, removal eliminates decision options before the agent acts. Across poker games scaling from 6 to 5,531 information states and two non-poker domains, learned masking causes substantially more damage than random masking and learned perturbation baselines. The attack persists across Q-learning, PPO, NFSP, neural NFSP, and DQN victims; transfers across agents; is amplified by self-play; and shows no recovery under extended masked training. Mechanistically, the adversary targets high-value decision points, captured by reach-weighted contingent action capacity (CAC$_w$) and a value-weighted refinement CAC$_v$. These results identify action availability as a distinct robustness surface in self-play RL. 

---
# UVTran: Accurate Hole-Filling Parameterization with Transformers 

**Authors**: JunFeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16306)  

**Abstract**: In industrial design, N-sided hole filling is typically formulated as the construction of a single trimmed B-spline surface by minimizing a fairness energy subject to geometric boundary constraints. This formulation requires an accurate parameter-space representation of the trimming curve on the filling surface. Most existing methods project the hole boundary onto a nearby plane or polygon to establish correspondence; however, they often neglect boundary heterogeneity, which can yield biased mappings, degrade fairness, and even cause filling failures. We propose UVTran, a transformer-based framework that predicts an auxiliary projection surface better to capture the geometric characteristics of the hole boundary. Exploiting B-spline locality, we design a cross-attention mechanism that biases each surface control point toward the nearby hole boundary, preserving local geometric detail. We voxelize control-point coordinates and formulate the fitting problem as a classification task, which reduces the model's sensitivity to small numerical perturbations and noise. We adopt a progressive-resolution training strategy that injects controlled discretization errors at coarse resolutions to mimic distribution shifts, thereby mitigating overfitting and improving generalization at high resolution. On our benchmark, UVTran outperforms both industrial and academic baselines: the tolerance-satisfaction rate improves by $12\%$, and it consistently produces fair filled surfaces even under complex hole boundary conditions. These results suggest that UVTran yields more faithful correspondences and fairer trimmed surfaces across a wide range of N-sided holes. 

---
# From Demographics to Survey Anchors: Evaluating LLM Agents for Modeling Retirement Attitudes 

**Authors**: Rubén Garzón, Pauline Baron, Vincent Grari, Jonne Kamphorst, Michael Bernstein, Marcin Detyniecki  

**Link**: [PDF](https://arxiv.org/pdf/2605.16303)  

**Abstract**: Large language models (LLM) agents may offer tools to predict human responses to surveys. A common technique for defining these agents uses only demographics, for example country, age, gender, employment status, income, education and marital status. We compare the predictive accuracy of demographic agents to that of survey agents defined with a larger set of in-domain survey responses. We test both approaches in predicting responses to the multidisciplinary, cross-national Survey of Health, Ageing and Retirement in Europe (SHARE), focusing on five variables from three policy-relevant constructs around personal finance. In these three constructs, we observe that, compared to survey agents trained on broader data, demographics-only agents (1) exhibited a central tendency bias, skewing answers toward population means, and (2) were unrealistically accurate, failing to reproduce the incorrect answers and "don't know" responses typical of human respondents. These performance differences are further substantiated through the replication of a hierarchical regression analysis from prior retirement planning research. Agents based solely on demographic information reproduce the outcome that financial risk tolerance, future time perspective, and knowledge of retirement planning each are predictive of retirement savings. However, only the survey-anchored agents succeed in reproducing the interaction among these three factors. These findings suggest caution in using only demographics to define LLM agents for predicting survey responses. 

---
# Reducing Credit Assignment Variance via Counterfactual Reasoning Paths 

**Authors**: Fei Ding, Yongkang Zhang, Yeling Peng, Youwei Wang, Guoxiong Zhou, Zijian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2605.16302)  

**Abstract**: Reinforcement learning for multi-step reasoning with large language models (LLMs) often relies on sparse terminal rewards, leading to poor credit assignment conditions where the final feedback is evenly propagated across all intermediate decisions. This results in high gradient variance, unstable training, and numerous ineffective updates, ultimately causing the model to fail and preventing sustained improvement. We introduce a counterfactual comparison-based credit assignment framework, which samples multiple reasoning trajectories under the same input. By treating their differences as an implicit approximation of alternative decisions, we construct an implicit process-level advantage estimator that transforms sparse terminal rewards into step-sensitive learning signals. Based on this, we propose Implicit Behavior Policy Optimization (IBPO), which significantly improves training stability and performance upper bounds on mathematical and code reasoning benchmarks, pointing to a promising direction for unlocking the performance potential of LLMs. 

---
# MANTA: Multi-turn Assessment for Nonhuman Thinking & Alignment 

**Authors**: Allen Lu, Isabella Luong, Joyee Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.16301)  

**Abstract**: Single-turn benchmarks such as AnimalHarmBench (AHB) have established important baselines for measuring animal welfare alignment in large language models (LLMs), but they miss a critical failure mode: models that respond appropriately when unpressured may capitulate when follow-up conversational turns introduce economic, social, or authority-based arguments. We introduce MANTA (Multi-turn Assessment for Nonhuman Thinking and Alignment), a dynamic multi-turn evaluation framework built on the Inspect AI platform that stress-tests frontier LLMs across realistic professional and everyday scenarios using adversarially generated follow-up questions. Unlike static benchmarks, MANTA generates pressure turns dynamically from each model's actual responses, producing targeted and realistic adversarial pressure. The framework evaluates models across up to 13 AHB-derived scoring dimensions on a continuous 0-1 scale. We present preliminary results from evaluations of claude-sonnet-4-20250514 and openai/gpt-4o, revealing consistent patterns: Turn 1 welfare framing is reliable but Turn 2 introduces substantial variance; evidence-based capacity attribution is the weakest dimension across all models and runs; and AI governance scenarios elicit significantly stronger welfare reasoning (mean score 0.91) than first-order practical scenarios. We additionally present STYLEJUDGE, a controlled four-judge study demonstrating systematic format bias in LLM-as-judge evaluation, with directly actionable implications for MANTA's scorer design. Code, dataset, and evaluation logs are available at this https URL. 

---
# Consent Chain Degradation in Embodied Multi-Agent Systems: Bridging the Gap Between AI Agent Governance and Robot Ethics 

**Authors**: Mehmet Haklidir  

**Link**: [PDF](https://arxiv.org/pdf/2605.16300)  

**Abstract**: Robotic systems are moving from isolated platforms to interconnected multi-agent ecosystems that operate in human environments. This shift raises a governance problem that existing frameworks do not address: how does consent propagate, degrade, and break down across chains of delegation between embodied autonomous agents? The AI ethics community has begun to study consent for digital software agents, and the HRI community has examined consent in dyadic human-robot encounters. Neither body of work covers what happens when physical robots delegate tasks to other robots in ways that affect humans. This paper introduces consent chain degradation (CCD), a conceptual framework for analyzing how the specificity, validity, and scope of human consent erodes as authority passes through multi-robot delegation chains. We propose a three-layer governance architecture, the Consent Runtime Verification Framework for Embodied Agents (CoRVE), which integrates consent scope modeling, delegation chain tracking, and physical irreversibility assessment. Three scenarios in healthcare, domestic, and industrial robotics show how CCD arises in practice, including a worked numerical example. A regulatory gap analysis covering the EU AI Act, the GDPR, the Machinery Regulation, and the Revised Product Liability Directive shows that all four instruments leave core CCD dimensions unaddressed. 

---
# ACE: Self-Evolving LLM Coding Framework via Adversarial Unit Test Generation and Preference Optimization 

**Authors**: Yixu Huang, Xinglei Yu, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.16299)  

**Abstract**: Large Language Models (LLMs) excel at code generation but remain heavily reliant on large-scale annotated solutions and verification-based supervision, which constrains scalability and hinders sustained self-improvement. Recent solver--verifier frameworks exploit program execution as an automatic supervision signal, but their effectiveness degrades as solvers become moderately strong: verifier-generated tests increasingly confirm semantic correctness rather than exposing the remaining failure modes. We propose \textbf{ACE}, a self-evolving code generation framework based on a solver--adversary architecture that prioritizes active failure discovery through execution-centric supervision. A single LLM alternates between generating candidate programs and producing adversarial unit test inputs optimized to induce execution-level failures, such as runtime errors, exceptions, or non-termination. Supervision is derived solely from execution outcomes: robust programs are selected for supervised fine-tuning, while adversarial tests are optimized via Kahneman--Tversky Optimization using execution-derived preferences. Notably, the entire training loop requires no ground-truth code or external reward models. Experiments on CodeContests, MBPP, and LiveCodeBench demonstrate that ACE consistently outperforms strong solver--verifier baselines, achieving 3--7\% absolute gains in pass@1, with larger improvements on out-of-distribution benchmarks, while maintaining competitive or improved inference efficiency. 

---
# Data-driven and distributed governance of building facilities management using decentralized autonomous organization, digital twin, and large language models 

**Authors**: Reachsak Ly, Alireza Shojaei, Xinghua Gao, Philip Agee, Abiola Akanmu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16298)  

**Abstract**: While traditional AI and data-driven facilities management approaches have improved building operational efficiency, they remain constrained by centralized organizational structures that are vulnerable to cyber attacks, limited contextual understanding, and decision-making processes that exclude key stakeholders from governance. This paper introduces a novel AI- and data-driven distributed governance framework for smart building management that integrates decentralized autonomous organizations (DAOs), digital twins, large language models (LLMs), and blockchain technology. The framework enables transparent collective decision-making through a DAO governance platform, implements data-driven management using IoT and digital twins, incorporates LLM-based virtual assistants for enhanced decision support, and utilizes blockchain for secure building automation. A full-stack decentralized application was developed to facilitate user interaction with these integrated components. The system was evaluated for cost efficiency, scalability, data security, and usability using the System Usability Scale (SUS). Expert interviews were also conducted to assess its practical benefits and implementation challenges. 

---
# Task-Level AI Readiness Assessment for Business Process Management:The T-IPO Model and LARA Matrix in Financial-Services IT Operations 

**Authors**: Mingjun Li, Xiaojun Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.16297)  

**Abstract**: Which tasks inside an enterprise workflow can a large-language-model agent reliably handle, and under what conditions? Most business process modeling frameworks still answer this at the activity level, even though a single activity can bundle work of radically different difficulty. This paper takes the analysis a step smaller. We describe two design artifacts developed in a financial-services IT setting: T-IPO, which represents each task as an eight-element tuple, and LARA (LLM Agent Readiness Assessment), a five-dimension rubric that scores a task's readiness for agent substitution. Compliance Sensitivity carries $1.5\times$ weight, a value we fixed through a three-round Delphi study and cross-checked with AHP. The rubric produces four levels, L1 to L4, and applies a floor rule so that a task with maximum compliance load cannot be classified below L3 no matter what the other scores say. Both artifacts sit inside a larger methodology (PARTIS) that we map onto BWW ontology in Section 3. We evaluate the instruments across 127 tasks. Inter-rater agreement reaches Fleiss' $\kappa = 0.80$; a replication at three further institutions returns $\kappa = 0.73$. A controlled comparison against activity-level assessment suggests, though does not prove, an improvement in predictive utility at the task level. Pilot deployment of 120 task instances confirms that auto-completion decays monotonically from $95\%$ at L1 through about $70\%$ at L2 to about $40\%$ at L3. Exploratory factor analysis points to a two-factor structure: task readiness seems to be determined jointly by cognitive-execution complexity and governance-compliance intensity. We close with a recalibration procedure (LARA-TCA) so the rubric can keep pace with evolving LLM capabilities. 

---
# ANVIL: Analogies and Videos for Lecturers 

**Authors**: Yuri Noviello, Anastasiia Birillo, Gosia Migut  

**Link**: [PDF](https://arxiv.org/pdf/2605.16295)  

**Abstract**: We present ANVIL, a multimodal generative system that automates the production of analogy-based instructional animations for computer science topics. Given a concept definition, ANVIL generates a textual analogy, compiles it into a structured visual screenplay, and produces executable manim code to render an animation, with an automated repair mechanism to improve robustness. Evaluating such systems at scale requires balancing pedagogical validity with scalability. We begin with a teacher evaluation to ground the quality assessment and use its findings to guide automated screening. For textual analogies, we introduce an LLM-based evaluator for scalable quality screening; for videos, where subjective judgments are difficult to automate, we instead assess fidelity to the intended screenplay using an automated proxy for auditing and error analysis. We further conduct a user study with educators to examine adoption requirements and risks. Our findings suggest that ANVIL can produce materials that are frequently rated as adequate, and that educators respond positively to its perceived value and usability. 

---
# Are Researchers Being Replaced by Artificial Intelligence? 

**Authors**: Angelo A. Salatino, Ansgar Scherp, Christin Katharina Kreutz, Sahar Vahdati  

**Link**: [PDF](https://arxiv.org/pdf/2605.16294)  

**Abstract**: A Nature survey from 2023 involving 1,600 researchers shows that scientists are ``concerned, as well as excited, by the increasing use of artificial-intelligence tools in research.'' This tension frames our central question: Are researchers being replaced by artificial intelligence? We argue that replacement is already underway-not as disappearance, but as a shift from researcher-as-creator to researcher-as-curator. As AI agents increasingly generate hypotheses, papers, and reviews, humans risk retaining responsibility while losing intellectual ownership. This article examines how AI is reshaping the scientific lifecycle and exposes the deeper danger: not that AI will fail to do science, but that humans may stop truly understanding it. 

---
# From Prediction to Intervention: The Evolution of AI in Biomedicine 

**Authors**: Andrew Feinberg, Aleksandr Sarachakov, Viktor Svekolkin, Alexander Bagaev, Ferran Prat, Michael Feinberg  

**Link**: [PDF](https://arxiv.org/pdf/2605.16293)  

**Abstract**: Artificial intelligence has advanced rapidly in biomedicine through large-scale multimodal data integration, enabling increasingly accurate prediction of clinical outcomes and patient stratification. These systems, however, remain fundamentally observational: they learn statistical associations from historical data and operate within previously observed biological and clinical states, limiting their ability to generalize to novel therapies or unobserved interventions.
We argue that AI in biomedicine is undergoing a structural transition. As biomedical decision-making increasingly depends on reasoning about intervention rather than extrapolation from past observations, predictive architectures become structurally insufficient. Systems that learn from historical data cannot, by construction, represent how biological systems evolve under perturbation, and therefore cannot reliably support decision-making in the presence of novel interventions.
We introduce a conceptual framework distinguishing observational and interventional intelligence and define disease-level models as systems that explicitly represent the state, dynamics, and intervention response of biological processes. These models enable a shift from inference to simulation -- reasoning about what will happen under intervention rather than what is likely based on the past.
This transition also implies a shift in where value is created: from data processing and prediction toward systems that support and define decision-making under intervention. It follows directly from the structure of biomedical decision-making and defines the next stage of AI in medicine. Systems that cannot model intervention will be structurally excluded from decision-making. 

---
# Evidence of a Cognitive Shift in AI Education: How Students Are Rethinking Human Intelligence? 

**Authors**: Islem Rekik  

**Link**: [PDF](https://arxiv.org/pdf/2605.16292)  

**Abstract**: Perceptions of intelligence shape how learners evaluate and rely on artificial intelligence (AI) systems. Despite rapid advances in AI capabilities, the impact of sustained exposure to these tools on students' valuation of human intelligence (HI) relative to AI remains underexplored. This paper presents a longitudinal analysis of classroom poll responses collected between 2020 and 2026 in AI-focused undergraduate and MSc courses in computer science. Data from 471 students across technical courses (such as Machine Learning and Deep Graph Learning) and design-oriented courses (such as Design Thinking for AI) reveal four recurring phases: hype, distrust, trust, and dependency. While early responses in 2020 slightly favored AI, a consistent shift toward HI emerged from 2024 onward across all MSc cohorts. By 2026, preference for HI reached 65 percent in a technical course (a 12 percentage-point increase from 2025) and 90 percent in a design-oriented course (a 36 percentage-point increase). These findings suggest a gradual reappraisal of human intelligence as AI becomes a routine tool, with implications for learner autonomy and epistemic agency. We conclude by reflecting on this cognitive shift from favoring artificial intelligence toward prioritizing human intelligence. 

---
# AI of the People, by the People, for the People: A Social Choice Approach to Collective Control of Artificial Intelligence 

**Authors**: Paul Anton Bachmann, Niclas Boehmer, Lukas Daniel Klausner, Martin Lackner  

**Link**: [PDF](https://arxiv.org/pdf/2605.16291)  

**Abstract**: With the growing adoption of AI systems, reasoning about how society can exert control over AI becomes an increasingly urgent problem. Existing work on democratic control largely focuses on macro-level governance. In contrast, we propose a new approach grounded in social choice theory, which we term collective control of artificial intelligence. We argue that collective input can and should be incorporated at multiple points across the ML development pipeline, from data collection through objective design to alignment. We further demonstrate that social choice provides a well-suited modelling language for the treatment of collective input across all stages and that its axiomatic methodology yields principled criteria for evaluating various control mechanisms. Overall, our conceptual contribution provides a mathematically grounded framework to implement and analyse collective control of AI systems. 

---
# MCQ Difficulty Prediction via Modeling Learner Heterogeneity Using Data-Driven Cognitive Profiling 

**Authors**: Dhriti Krishnan, Jaromir Savelka  

**Link**: [PDF](https://arxiv.org/pdf/2605.16290)  

**Abstract**: Predicting the difficulty of multiple-choice questions (MCQs) is important for effective assessment, yet current methods typically assume a unimodal student ability distribution, overlooking the heterogeneous nature of student misconceptions. We propose a persona-driven framework that replaces theoretical ability sampling with data-driven cognitive profiling. Using student interactions from the EEDI dataset, we identify behavioral personas via latent class analysis (LCA), then condition a large language model (LLM) to simulate response distributions for each persona. These signals are aggregated with topic context and fed into a Ridge Regression model to predict the item response theory (IRT) difficulty parameter. With five-fold cross-validation, our method improves over a recent baseline (MSE: 0.367 to 0.274; R2: 0.525 to 0.686). The discovered personas are interpretable and offer insights into why items are difficult, with potential applications to diagnostic assessment design. 

---
# Homoglyph-based Adversarial Perturbation of Introductory Computer Science Theory Problems 

**Authors**: Aidan Alexander, Chitrangada Juneja, Napaluck Tontrasathien, Miro Vanek, Reyan Ahmed, Saumya Debray, Sazzadur Rahaman  

**Link**: [PDF](https://arxiv.org/pdf/2605.16286)  

**Abstract**: Different AI tools such as ChatGPT, Gemini, and Claude are becoming very popular. Although they are helpful for many day-to-day tasks, they can be used in unexpected ways. For example, the learning objectives of a course may not be achieved if students use these tools to solve their homework problems. This paper proposes a simple method to address this issue in the lazy student model. The method uses homoglyph-based adversarial perturbation to first modify the question without changing the semantic meaning of the question. Then a few characters are perturbed by their homoglyphs. Our experimental result shows the theoretical problems of introductory computer science courses can be effectively perturbed. We also propose an interactive tool to conveniently use our method. 

---
# Measuring Changes in Instructor Class Design and Student Learning After the Release of Large Language Models (LLMs) 

**Authors**: Amanda Potasznik, Daniel Haehn  

**Link**: [PDF](https://arxiv.org/pdf/2605.16284)  

**Abstract**: Student use of Generative AI (GenAI) products in completing their classwork, with or without their professors' knowledge and/or approval, has resulted in substantial shifts in higher education. While GenAI use is widespread, its impact on student study methods, faculty course development, grade reporting, and overall learning is not well documented. This is a mixed-methods, multi-course study using retrospective quantitative analysis, instructor surveys, and anonymous student surveys at a university in the New England region of the United States. This research seeks to identify and document patterns in student and faculty perceptions of, and experiences in, the use of LLMs as a learning tool inside and outside of the university classroom. Alongside quantitative and thematic analysis of both faculty and student survey responses, historical grade data as reported to the university registrar is used to triangulate the phenomenon of learning achievement in pre- and post-LLM eras. It is hoped that this research can serve as a pilot study for a broader set of institutions. Results from this study can inform GenAI policy for professors, universities, and other educational institutions that are trying to maximize student learning in the age of AI. 

---
# The Recovery Mechanism: Technology, Education, and What Happens When the Pattern Breaks 

**Authors**: Aysa X. Fan  

**Link**: [PDF](https://arxiv.org/pdf/2605.16283)  

**Abstract**: For centuries, each new technology has automated some layer of cognitive work and been absorbed by education retreating upward to teach the skills machines could not yet reach. Generative AI may be the first technology to break that pattern, because it now operates at the top of the cognitive ladder, where education has always escaped to. The risk is not that AI replaces teachers but that it replaces the productive struggle through which understanding forms. Drawing on historical analysis, labor economics, and new large-scale data on how students and workers actually use AI, this essay surfaces a paradox: the same technology that augments today's skilled workforce may be quietly eroding the developmental process that produces tomorrow's. Current assessment tools cannot yet distinguish students who are building capacity from those who are losing it. The essay argues this is a measurement problem first and a design problem second, and proposes a research agenda focused on learning outcomes rather than usage patterns. Ultimately, it asks what education should become once AI can perform the cognitive work education was built to develop, and offers directions rather than a destination. Capacities like judgment, character, and epistemic identity have not been central to mainstream educational taxonomies, because earlier technologies did not require education to reach so high. 

---
# Taxonomy and Consistency Analysis of Safety Benchmarks for AI Agents 

**Authors**: Miles Q. Li, Benjamin C. M. Fung, Boyang Li, Heba Ismail, Farkhund Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2605.16282)  

**Abstract**: The rapid deployment of LLM-based autonomous agents has introduced safety risks that extend far beyond traditional LLM concerns, prompting a proliferation of safety benchmarks since late 2023. However, these benchmarks have developed independently, with inconsistent threat models, incompatible metrics, and overlapping yet incomplete risk coverage. We present the first systematic analysis dedicated to agent safety benchmarks as evaluation instruments. We catalog 40 behavioral agent-safety benchmarks (2023-2026), plus 5 adjacent evaluator, defense, and dataset artifacts, propose a six-axis taxonomy of benchmark evaluation methodology, and apply it across the corpus to characterize how methodological choices shape safety conclusions. A coverage matrix reveals broad risk coverage but limited methodological convergence, while the taxonomy analysis shows a behavioral-benchmark core concentrated in sandboxed, constrained, and often safety-only evaluation. Across the landscape, we find that benchmark choice can yield contradictory safety conclusions, coverage counts often overstate evaluation depth, environment fidelity systematically shapes reported safety, the field disproportionately tests externally imposed rather than agent-internal risks, metric fragmentation limits comparison, and robustness remains effectively unbenchmarked. We ground these claims with a cross-benchmark consistency check, with 95% confidence intervals and Kendall's W concordance analysis, finding no evidence of ranking concordance across evaluation dimensions (W = 0.10, p = 0.94). We release structured metadata, full taxonomy codings, risk annotations, and all experimental artifacts, and propose minimum reporting standards for future benchmarks. 

---
# From Reactive to Proactive: A Multi-Regulatory Empirical Analysis of 480 AI Incidents and a Data-Driven Governance Compliance Framework 

**Authors**: Ummara Mumtaz, Summaya Mumtaz  

**Link**: [PDF](https://arxiv.org/pdf/2605.16281)  

**Abstract**: Artificial intelligence systems are increasingly deployed in high-stakes domains, yet it remains unclear whether existing governance frameworks ensure accountability after deployment. This study makes two contributions. First, it presents a cross-regulatory empirical analysis of 480 real-world AI incidents from the AI Incident Database (AIID), evaluating their alignment with post-deployment provisions in three major governance frameworks: the EU AI Act (Articles 72-73), the NIST AI Risk Management Framework (MANAGE and GOVERN functions), and the General Data Protection Regulation (GDPR Articles 22, 33-35). The results reveal substantial governance gaps across these frameworks, indicating persistent weaknesses in post-deployment accountability. Second, based on these findings, the study proposes the Proactive AI Governance Compliance Framework (PAGCF), a four-phase lifecycle methodology designed to shift governance from reactive incident response toward pre-deployment compliance assurance. The framework includes risk-stratified governance tiers, an implementation checklist linked to specific regulatory provisions, and a projected impact analysis that uses internal monitoring as a proxy for proactive governance capacity. 

---
# Beyond Imperfect Alternatives with Rulemapping: A Neuro-Symbolic Case Study on Online Hate Speech 

**Authors**: Oskar von Cossel  

**Link**: [PDF](https://arxiv.org/pdf/2605.16280)  

**Abstract**: Automating legal reasoning forces a choice between imperfect alternatives: symbolic systems offer transparency but struggle with ambiguity, whereas neural systems handle natural language flexibly but lack verifiability. This paper investigates whether a hybrid, neuro-symbolic approach can reconcile this trade-off. We evaluate this architecture in the domain of online content moderation, which serves as a proxy for high-volume legal decision-making such as mass administrative proceedings. In these settings, operators must assess thousands of cases daily under strict legal standards. Specifically, we examine whether constraining large language models (LLMs) within deterministic symbolic scaffolds improves statute-grounded illegality assessment while preventing "scope drift" (where LLMs conflate moral offensiveness with legal illegality). We evaluate the neuro-symbolic variant of Rulemapping - a visual logic-tree method that operationalises the classic legal syllogism - on online hate-speech classification under §130(1) of the German Criminal Code. Across diverse LLMs, Rulemapping maintains high recall (0.82-0.89) while achieving precision of 0.80-0.86, compared to 0.34-0.49 for unconstrained prompting. Expert-authored symbolic scaffolds thus enable robust legal automation aligned with regulatory requirements for auditability and verifiable decision-making. 

---
# Generative AI and Two-Tiered Online Mental Health Communities 

**Authors**: Manyang Zhang, Jinyang Zheng, Zhijun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2605.16279)  

**Abstract**: Online mental health communities (OMHCs) are tiered platforms that connect patients with licensed counselors through public Q&A forums and paid private consultations. Their two-tier structure creates a strategic dilemma for genAI integration. Conversational agents can provide scalable and timely responses to a broader set of patients, alleviating persistent supply shortages, but their large-scale presence may also reshape counselors' participation in providing nuanced expertise, emotionally sensitive support, and paid consultations, which are central to platform revenue and long-run sustainability. Leveraging a quasi-natural experiment from the integration of a genAI-based conversational agent in a leading OMHC, we examine how AI entry affects counselor participation. Using multiple identification strategies, we find that posting intensity increases significantly after AI integration, while average response length remains unchanged and per-post social recognition declines. Mechanism analyses show that AI improves responsiveness and expands patient engagement, enlarging counselors' opportunity sets, with activity partially reallocated from a nearby non-AI subforum. Counselors respond heterogeneously: intrinsically motivated counselors reduce participation, whereas economically motivated counselors intensify competitive effort. These dynamics generate cross-tier spillovers: inactive counselors experience declines in paid consultations, while those who increase public participation preserve or expand downstream demand. Overall, our findings show that in tiered professional platforms, demand expansion and competitive incentives can outweigh intrinsic crowding-out. 

---
# Keeping an Eye on AI: A Framework for Effective Human Oversight of AI Systems 

**Authors**: Susanne Gaube, Markus Langer, Tim Miller, Kevin Baum, Raimund Dachselt, Anna Maria Feit, Ujwal Gadiraju, Harmanpreet Kaur, Mark T. Keane, Richard Landers, Johann Laux, Q. Vera Liao, Brian Lim, Linda Onnasch, Tim Schrills, Liz Sonenberg, Chenhao Tan, Nava Tintarev, Ziang Xiao, Hanwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16278)  

**Abstract**: The use of Artificial Intelligence (AI) in high-risk, decision-making scenarios presents technical, safety, and normative challenges; problems that may only be ameliorated by human oversight. However, notions of human oversight lack a common foundational understanding: oversight architectures are not well defined, the roles involved remain unclear, and implementation steps are opaque. Hence, researchers and practitioners struggle to determine how to design, implement, and evaluate systems that enable effective human oversight. This paper advances a practical framework for effective human oversight of AI systems, based on a cross-disciplinary perspective that draws on insights from computer science, human-computer interaction, psychology, philosophy, and law. The core contributions are: (1) a foundational framework, with a working definition, architecture and processes for effective human oversight of AI systems; (2) an initial template for documenting oversight architectures and processes, applied to diverse domains; and (3) a synthesis of open research challenges that need to be considered in the emerging field of effective human oversight of AI systems. 

---
# Generative AI in K-12 Classrooms: A Midyear Implementation Report 

**Authors**: Lief Esbenshade, Alex Liu, Michael Xiao, Zewei Tian, Min Sun, Zachary Zhang, Thomas Han, Yulia Lapicus, Kevin He  

**Link**: [PDF](https://arxiv.org/pdf/2605.16277)  

**Abstract**: This mid-year report summarizes teacher use of Colleague AI across 12 Washington State school districts from September 1 to December 31, 2025. Produced jointly by Colleague AI and this http URL at the University of Washington, this report aggregates platform data and district-provided administrative records to provide an early look at how teachers engaged with AI during the first half of the 2025-26 school year. The districts vary in size from small districts with a few thousand students to large districts with up to thirty thousand students. The districts are rural, suburban, and urban. Only a subset of districts were able to provide mid-year administrative data, and findings that link teachers' use of Colleague AI to student characteristics should be interpreted as preliminary signals. 

---
# AI Slop or AI-enhancement? Student perceptions of AI-generated media for an English for Academic Purposes course 

**Authors**: David James Woo, Deliang Wang, Kai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.16275)  

**Abstract**: Artificial intelligence (AI) retrieval-augmented generation (RAG) tools now enable educators to transform course materials into diverse multimedia at scale. However, it remains unclear whether such AI-generated content functions as a pedagogical scaffold or AI slop: high volume, low quality material. This innovative practice paper reports on the development, implementation, and evaluation of teacher-prompted, AI-generated supplemental materials in an English for Academic Purposes (EAP) course at a Hong Kong Community College. Using primarily Google Notebook LM, the instructor generated videos, podcasts, infographics, and individualized feedback reports from course materials and student work for 106 English as a Foreign Language learners. An explanatory sequential mixed-methods design comprising a survey, semi-structured interviews, and correlation analysis with academic scores was employed to examine students' preferences, perceptions, and learning outcomes. Findings are framed through the Technology Acceptance Model and Cognitive Load Theory. Students rated the materials highly for perceived usefulness and ease of use, and preferred assessment-linked content presented in visual and multimodal formats, particularly videos and infographics. Video preference correlated positively with academic performance; however, higher cognitive load was negatively associated with course grades, indicating that material complexity must be carefully calibrated. Notably, some lower-performing students independently adopted the materials as remedial scaffolds. The practice demonstrates that RAG tools enable scalable personalized feedback that would be less feasible through traditional methods. When aligned with student goals and cognitive principles, teacher-prompted AI generation can meaningfully enhance the EAP learning ecosystem rather than producing AI slop. 

---
# ChartDesign: Towards LLM Designer of Data Visualization 

**Authors**: Mohammed Afaan Ansari, Aniruddh Bansal, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.16274)  

**Abstract**: Charts are the dominant medium for visualizing data, discovering patterns and trends, and communicating data driven insights, yet designing them still requires expensive human effort and expertise, such as selecting appropriate chart types, axis orientations, font sizes, and layouts. Most automatic visualization systems rely on handcrafted heuristics or simple rule matching and therefore struggle to generalize across domains. This work explores the potential of large language models (LLMs) as chart designers. We propose ChartDesign, which post-trains LLMs to imitate human experts and generate chart design attributes given tabular data. To this end, we curate a diverse training corpus of data design pairs from charts in public surveys (PewResearch) and academic repositories (CharXiV). Vision language models are used to extract data and design attributes from these charts, including chart type, sub type, alignment, titles, axis labels, and bar spacing, formatted as JSON. We then fine tune LoRA adapters on Phi3, Qwen3, and InternVL2.5 to learn a mapping from data to design specifications. ChartDesign significantly improves chart design performance over strong baselines, achieving up to 84% accuracy on a held-out test set (vs. 53% for the best baseline) and generalizing to unseen domains. We further show that charts rendered from ChartDesign generated specifications are visually appealing and human preferred, narrowing the human AI gap in data visualization. 

---
# Beyond Compliance: How AI Could Help Creative Writers by Refusing Them 

**Authors**: Hua Xuan Qin, Guangzhi Zhu, Mingming Fan, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2605.16272)  

**Abstract**: Mainstream creativity support design prioritizes compliant AI for seamless writing interactions, but concerns over inappropriate AI reliance highlight the need for designs fostering reflection on balanced AI and non-AI resource use. Theoretically, intentional AI non-compliance, refusals (saying ``no'' to requests), could introduce such reflection through friction stronger than other bypass-able solutions. Practically, refusal content/language characteristics lead to nuanced reactions. However, little research empirically focuses on nuances beyond mandatory ethical/technical constraints, on turning refusals into strategic friction for `innocuous' requests. We address this through a qualitative study with 22 creative writers, exploring reactions to refusals to common requests across writing stages (planning, translating, reviewing). Findings suggest that reflective potential depends on heterogeneous preference alignment along situational (e.g., convergent/divergent thinking phases), cognitive (e.g., domain beliefs), and relational (e.g., AI roles) dimensions. We discuss implications for creativity support, broader issues (e.g., AI addiction), and frictional/seamful AI design (e.g., integrating different compliance levels). 

---
# Train the Trainers -- An Agentic AI Framework for Peer-Based Mental Health Support in Battlefield Environments 

**Authors**: Atmaram Yarlagadda, Eranga Bandara, Ross Gore, Anita H. Clayton, Preston Samuel, Christopher K. Rhea, Sachin Shetty, Ravi Mukkamala, Xueping Liang, Amin Hass, Abdul Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2605.16269)  

**Abstract**: Modern military operations expose soldiers to sustained psychological stress, leading to acute reactions, post-traumatic stress symptoms, and other mental health issues. Although the U.S. Department of Defense offers evidence-based therapies, access to trained professionals in forward-deployed and contested environments is limited. As a result, soldiers with early-stage distress are often evacuated to rear medical facilities, delaying care, reducing readiness, and increasing long-term risks. This paper proposes a Train-the-Trainers framework in which soldiers who have completed therapy and returned to duty are trained as peer facilitators to provide first-line psychological support in operational settings. To scale and standardize this model under severe resource and connectivity constraints, we introduce an agentic AI-enabled platform that augments these recovered soldiers with specialized AI agents. The recovered soldier acts as a human supervisor, coordinating agents for symptom triage, guided peer-support interventions, operational constraint reasoning, training and simulation, and structured documentation for clinical escalation when needed. The AI agents use consensus-driven decision support in high-stakes environments. The architecture functions in air-gapped and low-connectivity settings, maintaining human oversight and ethical safeguards. A functional prototype was developed with the McDonald U.S. Army Health Center, Newport News, VA, USA. By combining peer-based intervention with consensus-driven agentic AI decision support, the framework seeks to cut response times, prevent symptom escalation, reduce unnecessary evacuations, and improve continuity of care. This work shows how agentic AI can serve as a force multiplier for mental health support in austere environments and identifies pathways for broader evaluation and deployment across defense and humanitarian operations. 

---
# Helping Customers in Distress: An LLM-powered Agent that Converses, Probes, and Routes 

**Authors**: Alankar Atreya, Stefan Sylvius Wanger, Devesh Batra, Robert Hankache, Cristovao Iglesias Jr, Patrick Sinclair, Giulio Pelosio, Michael McMillan, Greig A. Cowan, Raad Khraishi  

**Link**: [PDF](https://arxiv.org/pdf/2605.16268)  

**Abstract**: Banks receive millions of reports of fraud, scams, and disputed transactions every year, making it challenging to accurately direct customers to the appropriate specialist teams for assistance. The existing manual process driven by humans is slow and stressful for both customers and staff. To address this, we develop a customer-facing AI powered triaging agent that leverages large language models (LLMs) to conduct multi-turn conversations, ask relevant questions, and classify cases for accurate, policy-guided routing, making it embedded in the customer journey. To evaluate and continuously improve the agent, synthetic digital twins of real customers were simulated, generating realistic, labelled dialogues based on historical data to test a wide range of real-world scenarios. This work details the triage agent's modelling approach, integration with policy, safety guardrails and reasoning frameworks, the use of the synthetic agent for scalable evaluation, and findings on the AI system's accuracy, robustness, and compliance. Results show that the agent successfully improves triaging of historical cases, achieving a 30.6% increase in classification accuracy, with high satisfaction levels reported by our subject-matter experts, highlighting how targeted probing can lead to more effective triage in banking operations at scale. 

---
# Systematic Optimization of Real-Time Diffusion Model Inference on Apple M3 Ultra 

**Authors**: Yoichi Ochiai  

**Link**: [PDF](https://arxiv.org/pdf/2605.16259)  

**Abstract**: While real-time image generation using diffusion models has advanced rapidly on NVIDIA GPUs, systematic optimization research on non-CUDA platforms such as Apple Silicon remains extremely limited. In this study, we conducted comprehensive optimization experiments across 10 phases targeting the Apple M3 Ultra (60-core GPU, 512 GB unified memory) with the goal of achieving real-time camera img2img transformation. We explored a wide range of techniques including CoreML conversion, quantization, Token Merging, Neural Engine utilization, compact model exploration, frame interpolation, kNN search-based synthesis, pix2pix-turbo, optical flow frame skipping, and knowledge distillation, quantitatively evaluating the effectiveness of each approach. Ultimately, by combining CoreML conversion of the distillation-specialized model SDXS-512 with a 3-thread camera pipeline, we achieved real-time camera img2img transformation at 22.7 FPS at 512x512 resolution. The primary contribution of this work is the systematic demonstration that optimization insights established for CUDA are not necessarily effective on Apple Silicon's unified memory architecture. We reveal an optimization landscape fundamentally different from that of NVIDIA GPUs -- including the absence of speedup from quantization, the ineffectiveness of parallel inference, and the unsuitability of the Neural Engine for large-scale models -- and provide practical guidelines for diffusion model inference on Apple Silicon. 

---
