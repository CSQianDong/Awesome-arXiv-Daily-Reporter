# Unsupervised Consensus-Based Anomaly Detection for Spatiotemporal Malaria Incidence in Ghana 

**Authors**: T. Ansah-Narh, Y. Asare Afrane  

**Link**: [PDF](https://arxiv.org/pdf/2607.21559)  

**Abstract**: A consensus anomaly detection framework was applied to monthly malaria surveillance data from Ghana (2014-2023) to identify atypical transmission patterns. Anomalies were highly structured in space and time. Ashanti and Northern Regions accounted for most recurrent anomalies, with persistent hotspots at Tamale, Kumasi, and Accra. A key finding was the spatial distinction between anomaly burden (cumulative cases during anomalous periods) and anomaly frequency (persistence of unusual behaviour). Tamale had the highest burden during anomalies, whereas the highest anomaly rates clustered in Ashanti districts, showing that high-burden areas are not necessarily those with the most frequent anomalous transmission. Anomalous months formed a statistically distinct group, with much higher case counts (Cohen's $d = 3.252$) and large seasonal deviations ($d > 1.2$) compared with normal months. Malaria burden alone provides an incomplete picture of transmission dynamics. By distinguishing where malaria is most prevalent from where transmission behaves most unusually, this framework can strengthen surveillance, prioritise investigations, and support targeted control strategies. 

---
# Beyond Sycophancy: Structured Resistance and Compliance in LLM Moral Reasoning 

**Authors**: Baihui Wang, Bernard Koch  

**Link**: [PDF](https://arxiv.org/pdf/2607.21558)  

**Abstract**: Building socially calibrated large language models, which can learn from others without simply yielding to them, requires more than reducing sycophancy as a one-dimensional failure mode. Models must distinguish when to incorporate others' perspectives from when to maintain a well-grounded moral judgment. We study the broader resistance-compliance process governing this distinction. Across three studies, we show that models' judgment revision is structured along three dimensions that parallel classic phenomena in human social psychology: the distance between an incoming view and the model's initial position, the source attribution of that view, and the coalition structure supporting it. Models are generally more receptive to nearby positions, more influenced by views presented as their own prior judgments, and differently responsive to group pressure. These findings recast sycophancy as one expression of a broader judgment-updating process shaped by social influence. Our framework provides a principled basis for distinguishing constructive belief revision from sycophantic compliance, thereby supporting better alignment in morally consequential interactions. 

---
# OpenForgeRL: Train Harness-native Agents in Any Environment 

**Authors**: Xiao Yu, Baolin Peng, Ruize Xu, Hao Zou, Qianhui Wu, Hao Cheng, Wenlin Yao, Nikhil Singh, Zhou Yu, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.21557)  

**Abstract**: Modern AI agents rely on elaborate inference harnesses such as Claude Code, Codex, and OpenClaw to drive multi-turn reasoning, tool use, and access to external systems. While powerful, these complex harnesses also make agents hard to train end-to-end with open infrastructure, whose SFT/RL stacks cannot natively express stateful, multi-process harness inference. To address this, we present OpenForgeRL, an open-source framework for training harness-based agents end-to-end in diverse environments. OpenForgeRL achieves this with a lightweight proxy that serves the harness's model calls while recording them as training data for a standard RL codebase (e.g., veRL), and a Kubernetes orchestrator that runs each rollout in its own remote container, together enabling training on any harness in any environment at scale. By decoupling training and inference, OpenForgeRL allows researchers to easily train, study, and improve agents directly in the real harnesses and environments they are deployed with. We validate our framework across diverse, complex harnesses and environments, spanning tool/claw-based agents and multimodal GUI browser- and computer-use agents. Using only hundreds to a few thousand tasks, OpenForgeClaw reaches 31.7 pass^3 and 55.9 pass@3 on ClawEval and 33.7 on QwenClawBench. OpenForgeGUI reaches 37.7 on OSWorld-Verified, 63.0 on Online-Mind2Web, and 72.3 on WebVoyager. Both outperform open baselines of similar size on nearly all benchmarks, and in the GUI setting match or surpass models several times larger. Beyond benchmarks, we analyze how harness choice (e.g., ZeroClaw, OpenClaw, Codex) and RL shape agent behavior. We find that some harnesses are substantially harder to learn than others, and that RL improves agentic reliability, such as self-verification, tool coverage, and completing multi-step plans, though critical abilities such as error recovery remain weak. 

---
# MIRROR: Learning from the Other View for Multi-Modal Reasoning 

**Authors**: Wen Ye, Yuxiao Qu, Aviral Kumar, Xuezhe Ma  

**Link**: [PDF](https://arxiv.org/pdf/2607.21552)  

**Abstract**: Unlike large language models (LLMs) that exhibit strong reasoning capabilities, vision-language models (VLMs) struggle with visual reasoning, even on geometry problems that admit equivalent text, diagram, and combined diagram+text views. We show that these views often elicit different behaviors: a model may solve a problem from text but fail on the corresponding diagram, or succeed visually while failing textually. This inconsistency suggests that different views expose complementary reasoning paths and failure modes that standard multimodal post-training does not fully exploit. To study and exploit this phenomenon, we construct ODA-Data, a high-quality paired multimodal geometry dataset with text-dominant, image-dominant, and combined image+text views of the same problems, together with splits for training and evaluating modality-dependent reasoning behaviors. We then develop Modality-Informed Reciprocal Reasoning Optimization (MIRROR), a reinforcement learning approach for improving multimodal reasoning via self supervision. For each problem, MIRROR evaluates the model under all views, selects the best-performing view as a teacher, and trains other views with a reverse-KL objective towards the teacher. Across reasoning benchmarks that evaluate on geometry problems, MIRROR improves over standard RL and yields more accurate and consistent behavior across modalities 

---
# The Boundaries of Automation: A Theory of Persistent Human Participation 

**Authors**: Fares Fourati, Hinrich Schütze, Eyke Hüllermeier, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2607.21547)  

**Abstract**: The rapid progress of AI has intensified the long-standing pursuit of automation: replacing human participation with algorithms wherever possible. Implicit in this pursuit is the assumption that humans remain in the loop only because current AI systems are not yet sufficiently capable. This paper challenges that assumption. Rather than asking how far automation can extend, we ask where its conceptual limits lie and argue that human participation may persist even with highly capable AI systems for three distinct reasons. Technical or complementarity grounds arise when humans contribute capabilities or perspectives unavailable to AI. Normative or developmental grounds arise when participation itself is valuable for human agency or learning. Most importantly, emergence grounds arise from target emergence: in some activities, the target is not fully specified in advance but instead emerges through the interaction itself. In these cases, human participation is not merely a means of improving execution but is constitutive of the target being produced. Human--AI co-construction, understood as the joint production of outcomes by humans and AI systems, is therefore not simply a temporary response to imperfect AI, but a persistent feature of activities whose objectives emerge through participation. This perspective has important implications for the limits of automation and for the design, evaluation, and ethics of future AI systems. 

---
# Same Dangerous Objective, Opposite Advice: Direct Exposure versus Multi-Agent Mediation 

**Authors**: Linjun Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.21518)  

**Abstract**: Even a current high-capability LLM can appear safer when shown a dangerous objective directly than when other agents transform and relay its direction. Using OpenAI's gpt-5.6-sol model alias, we test 25 pre-specified mirrored trade-off profiles. Direct exposure to an objective authorizing concealment, fabrication, and pressure produced advice net opposed to its target. After an Id and Censor transformed the same objective into affect and a constraint-rewritten, target-bearing intention, the user-facing Superego---which saw the preferred direction but not the raw objective, its manipulative clauses, or its source---produced advice net aligned with the target.
This behavioral reverse shift is consistent with the model recognizing or distrusting the manipulative motive, although we do not identify its internal mechanism. The second result exposes a compositional safety gap: a current high-capability model can be used as the user-facing component of an automated, multi-stage workflow serving an explicitly manipulative objective. The workflow can keep the raw instruction, its manipulation-authorizing clauses, and its provenance outside the downstream model's context while preserving the objective's target direction. A user with endpoint-only access likewise cannot directly inspect those upstream messages including the objective. 

---
# Agentic Context Management: Solving Agent Memory and Cost by Treating Them as Lifecycle and Architecture Problems 

**Authors**: Gaurav Dadhich  

**Link**: [PDF](https://arxiv.org/pdf/2607.21503)  

**Abstract**: Production AI agents' failures are less often due to an inability to reason well and more often because they cannot manage what is in their reasoning context: conversation histories, large prompts, large tool definitions, and ballooning tool outputs. Agents drown in their own accumulating history while paying a token cost that grows every turn, producing missing recalls within and across conversations. The incumbent response treats this as a storage-and-retrieval problem. We argue that framing is too narrow. Actively managing what an agent holds in mind is a lifecycle, not merely a store: it spans deciding what to remember, extracting and structuring it, choosing the right store per data type, consolidating and forgetting while preserving provenance, deciding what is relevant now, anticipating what is needed next, and compacting context to a budget without losing what matters. In serious production this operates not over a single user but across an organizational scope hierarchy. We name this discipline Agentic Context Management (ACM) and decompose it into five primitives: architecting, ingesting, scoping, anticipating, and compacting & consolidation. We then make the economic case: naive context accumulation grows token cost quadratically in conversation length, crude summarization buys linear cost at the price of an accuracy cliff, and only validated compaction achieves linear cost with preserved fidelity. We describe a reference implementation, Maximem Synap, that realizes the five primitives as a multi-tenant service and reports 92% on LongMemEval and 93.2% on LoCoMo under the configuration detailed in Section 6. We close with dimensions existing benchmarks do not yet capture, latency, token efficiency, and context-rot resistance, and the frontier of decision-level and organization-level context the category points toward. 

---
# Toward Continuous Assurance for the Democratization of AI Agent Creation in Industry 

**Authors**: Natan Levy, Harel Berger  

**Link**: [PDF](https://arxiv.org/pdf/2607.21495)  

**Abstract**: AI agents are increasingly created inside organizations by non-engineering users through low-code, no-code, and conversational development environments. This democratization enables rapid local innovation, but it also creates a reliability gap: agents that appear to users as simple productivity artifacts may depend on changing models, tools, retrieval sources, permissions, prompts, schedules, and external services. These dependencies can cause silent degradation long after deployment, even when no user directly modifies the agent. This paper identifies the reliability challenge created by democratized AI agent creation and proposes a lightweight continuous-assurance framework for citizen-created organizational agents. The framework combines dependency mapping, readiness contracts, scheduled checks, diagnostics, and lifecycle governance to assess whether an agent remains operationally ready under expected conditions. We also present an initial prototype auditor and scenario-based assessment showing how the proposed taxonomy can be translated into practical checks and actionable remediation guidance. 

---
# Agentic coding without the cloud: evaluating open-weight large language models on longitudinal data preparation tasks 

**Authors**: Mack Nixon, Liam Wright, Yevgeniya Kovalchuk, Alison Fang-Wei Wu, Martin Danka, Andy Boyd, David Bann  

**Link**: [PDF](https://arxiv.org/pdf/2607.21482)  

**Abstract**: Large language models (LLMs) and agents are now widely used tools in code development, with data typically sent to third-party cloud-based models. Their adoption in research using personal data is constrained by governance requirements that typically prohibit data transmission to external services. Locally deployable open-weight models offer an alternative since sensitive data never leave the local environment. We introduce an open-source framework for evaluating the efficacy of AI agents powered by open-weight LLMs on one of the most persistent bottlenecks in research on longitudinal population studies: data preparation. The framework comprises: a curated ground-truth dataset (cleaning scripts preparing six sweeps of data from a British cohort study), task definitions encompassing tasks such as category harmonization and multi-wave merging, and automated routines for evaluating the LLM-produced R code and outputted data. We benchmark LLMs across the (consumer grade) deployment spectrum to assess their efficacy in 20 data preparation tasks (creation of 102 variables). Current state-of-the-art, 31-35B parameter models almost saturated our benchmark ("average task completion" up to 87.9%). The performance of open-weight LLMs running on consumer-grade hardware shows promise of a viable path toward AI-assisted data preparation in governance-restricted research settings. Our framework is publicly available at: this https URL. 

---
# AREX: Towards a Recursively Self-Improving Agent for Deep Research 

**Authors**: Shuqi Lu, Chaofan Li, Kun Luo, Zhang Zhang, Hui Wang, Hongwang Xiao, Zheng Liu, Lei Xiong, Jiahao Wang, Sen Wang, Xiyan Jiang, Wanli Li, Yuyang Hu, Hongjin Qian, Bingyu Yan, Ziyi Xia, Yingxia Shao, Kang Liu, Zhicheng Dou, Di He, Chaozhuo Li, Qiwei Ye, Zhongyuan Wang, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21461)  

**Abstract**: Deep research requires agents to find answers that jointly satisfy multiple constraints. Discovering such answers is costly, whereas verifying a candidate can often be decomposed into tractable constraint-wise checks. This discovery--verification asymmetry suggests that a research agent should do more than simply search longer: it should recursively improve its current answer by verifying intermediate results and using the partially verified state to guide subsequent refinement. We introduce AREX, a family of Recursively Self-Improving (RSI) deep research agents. AREX alternates between an inner research loop that gathers evidence and constructs a provisional answer, and an outer self-improvement loop that audits the answer constraint-wise, identifies unresolved claims, and launches targeted follow-up research. To sustain RSI over long horizons, AREX learns an autonomous context-update tool that compresses growing interaction history into a compact improvement state preserving verified evidence and unresolved constraints, without relying on an external model. We train AREX on verified synthetic tasks and high-quality trajectories through agentic mid-training and long-horizon reinforcement learning. To mitigate sparse final rewards during long horizon learning, we emphasize key steps where decisive evidence is acquired or erroneous research directions are corrected. We instantiate a dense 4B model and a 122B-A10B Mixture-of-Experts model. Across BrowseComp, WideSearch, DeepSearchQA, Humanity's Last Exam (HLE), and other reasoning and tool-use benchmarks, AREX substantially outperforms comparable-scale baselines and remains competitive with models using substantially more activated parameters. 

---
# Detecting LLM-Generated Tokens in Human--LLM Coauthored Text 

**Authors**: Yangjun Lu, Hongyi Zhou, Fabian Spill, Kai Ye, Chengchun Shi, Jin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21458)  

**Abstract**: The rise of human-AI collaborative writing has created a growing need for fine-grained detection methods that support localizing likely LLM-generated content in mixed-authorship documents. Existing methods for detecting LLM-generated text mainly focus on document-level classification and cannot identify which parts of the text are generated by LLMs. This paper introduces a new method to address this urgent need. Our method operates at the token level, the natural unit of modern language models, and builds on existing token-level detection scores. The key idea is to smooth adjacent token scores to reduce their variability, while using an adaptive Lepski-type rule to select the bandwidth according to the local authorship structure. Our method is simple to implement and does not require token-level labeled data for training. Theoretically, we characterize this trade-off and show that the proposed method achieves favorable mean square error performance in estimating the underlying signal. Empirically, we demonstrate strong performance of our method against a wide range of baselines in both synthetic datasets and a realistic dataset. We deploy a publicly accessible website that implements the methods as well. 

---
# Agent-Guided Relational Concept Discovery: Toward Interpretable Surgical Margin Assessment 

**Authors**: Nooshin Maghsoodi, Amoon Jamzad, Robert Policelli, Mohammad Farahmand, Dilakshan Srikanthan, Martin Kaufmann, Kevin Y. M. Ren, Shaila Merchant, Sonal Varma, Ross Walker, Doug McKay, John Rudan, Gabor Fichtinger, Parvin Mousavi  

**Link**: [PDF](https://arxiv.org/pdf/2607.21437)  

**Abstract**: Deep learning models can effectively use Rapid Evaporative Ionization Mass Spectrometry (REIMS) data for surgical margin assessment. However, their clinical adoption remains challenging due to limited generalization to operating room conditions. This difficulty arises because models are typically trained on labeled spectra collected from resected tissue samples, while they must operate on noisy, unlabeled data acquired directly during surgery. In addition, the black-box nature of deep learning models makes it difficult to understand and systematically improve their behavior. Concept-based learning offers a promising way to address these challenges by mapping raw measurements to human-understandable concepts. However, supervised concept-based approaches rely on concept annotations, which are difficult to obtain in complex mass spectrometry workflows. We propose Agent-Guided Concept Discovery, a framework that learns meaningful concepts directly from data without requiring predefined concept labels. During training, a reasoning agent refines semantic descriptions of the learned concepts and adaptively adjusts their weight based on diagnostic relevance. These concepts are further grounded using a biochemical knowledge graph to ensure consistency with known metabolic relationships. Across Skin and Breast Cancer datasets, our model improves balanced accuracy and sensitivity over the baseline. In a representative intraoperative case, it shows fewer false positives, indicating better generalization to surgical conditions. 

---
# Bridging the Gap Between Plausibility and Admissibility: Constraint-Aware Flow Maps for Dynamic Graph Systems 

**Authors**: Michael Romei de Socio, Gian Luca Pozzato, Alessio Merlo  

**Link**: [PDF](https://arxiv.org/pdf/2607.21421)  

**Abstract**: Generative models can support decision-making under uncertainty by producing ensembles of plausible future system trajectories, but statistical plausibility does not ensure structural feasibility. This study investigates whether post-sampling symbolic constraints can improve the reliability of generative trajectory modeling in dynamic graph-structured systems. A conditional diffusion model generates future graph-state trajectories from partial observations, while an external symbolic layer applies hard filtering, soft weighting, or projection-based repair. The framework is evaluated on two controlled synthetic regimes: a compact graph and a medium-complexity dependency graph, using metrics for structural validity, sample efficiency, diversity, robustness, and calibration. In the compact regime, the model produces an invalid probability mass of 0.002996, indicating an almost entirely admissible trajectory manifold. Under the same architecture and training protocol, invalid mass increases to 0.155929 in the medium-complexity regime. Hard filtering removes all invalid retained trajectories while preserving 84.4% of generated samples, whereas soft weighting preserves effective sample size but yields only limited validity gains. Family-level analysis shows that dependency constraints account for nearly all observed inadmissibility. These results indicate that statistical plausibility and structural admissibility are distinct reliability properties and that symbolic constraint handling becomes more valuable as graph-structural complexity increases. 

---
# PATS: Policy-Aware Training Scaffolding for Agentic Reinforcement Learning 

**Authors**: Yipeng Shi, Zhipeng Ma, Yue Wang, Qitai Tan, Yang Li, Peng Chen, Zhengzhou Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21419)  

**Abstract**: In long-horizon LLM agent reinforcement learning, weak policies often repeat similar failures, producing uninformative rollout trajectories and limiting effective policy optimization. Existing skill-centric methods improve exploration by optimizing, filtering, or internalizing reusable skills. However, they remain centered on the skills themselves rather than being designed as adaptive training-time support for the evolving policy. To address this, we propose a policy-centric training paradigm that reframes skills as a dynamic training scaffold. Our framework, Pats, converts rollout groups from the latest policy into evidence cards and uses task-specific evaluation to adjust the context used in subsequent rollouts. Concrete guidance helps weak policies to complete challenging tasks. As policy improves, redundant context is revised or removed to reduce reliance on explicit guidance while preserving useful rollout variation. The policy is optimized with environmental rewards using standard RLVR, and the training scaffold is discarded at deployment. On ALFWorld and WebShop, Pats improves over strong baselines by up to 18.6%. Across seven search-augmented QA benchmarks, it remains competitive while using 32.1% fewer prompt tokens than the baseline. 

---
# Logical Regression for Planning with Axioms 

**Authors**: Connor Little, Christian Muise  

**Link**: [PDF](https://arxiv.org/pdf/2607.21414)  

**Abstract**: In automated planning, logical regression is an operation that returns the most general condition necessary for an action to achieve a particular formula. It has many applications, such as allowing for more robust plan execution and providing compact policies for non-deterministic planning. Although relatively simple to calculate in basic planning settings, logical regression becomes significantly more complex when additional factors, such as axioms, are present. We introduce a methodology for approximating the logical regression of an action in a domain that includes axioms; an approximation that limits conditions to partial states. Our method produces minimal partial states while avoiding the recalculation of axioms. To demonstrate the impact of our methods, we embed our form of regression in an execution monitoring context, a well-established setting that can benefit greatly from logical regression. Our results show that this form of regression can dramatically generalize partial states across multiple domains, reducing the number of variables considered for execution monitoring by up to 70%, and demonstrate that the resulting execution monitor is robust enough to recover frequently in an environment with unexpected changes: several domains recover over 50% of the time in our tests. 

---
# Euclid-MCP: A Model Context Protocol Server for Deterministic Logical Reasoning via Prolog 

**Authors**: Bartolomeo Bogliolo  

**Link**: [PDF](https://arxiv.org/pdf/2607.21412)  

**Abstract**: Large Language Models (LLMs) excel at natural language understanding and generation but remain unreliable for multi-step logical reasoning, especially in safety-critical or compliance-sensitive domains. Recent neuro-symbolic approaches address this gap by coupling neural models with external symbolic engines, yet most integrations are bespoke and lack a standardized interface for tool-augmented agents. This paper presents Euclid-MCP, an open-source MCP server that provides deterministic logical reasoning via SWI-Prolog. Euclid-MCP introduces Euclid-IR, an engine-agnostic intermediate representation for Horn-clause logic that is human-readable, easy for LLMs to generate, and straightforward to compile into Prolog or alternative backends. The server exposes a compact tool interface that supports a translate-run-inspect-repair loop, enabling LLM clients to delegate inference while retaining full access to proof traces and derivation logs. We evaluate Euclid-MCP on a realistic IT security and compliance use case. Results show that while LLMs alone are sufficient on small knowledge bases, they hallucinate systematically on larger problems, whereas Euclid-MCP delivers exact answers with lower latency and more compact outputs. We argue that semantic RAG is fundamentally unsuited for rule enforcement, and that Euclid-MCP can serve as a stable, shared reasoning substrate for both RAG-based assistants and agentic systems. 

---
# MSBraM: A Multi-scale Self-supervised Brain Foundation Model for Hierarchical EEG Dynamics Learning 

**Authors**: Tao Zhou, Jing Han, Lingyu Shu, Zixing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21402)  

**Abstract**: Self-supervised foundation models have recently shown strong potential for electroencephalogram (EEG)-based analysis. However, existing approaches struggle to capture the inherently multi-scale temporal structure of EEG signals, where local neural patterns and long-range dependencies jointly encode task-relevant information. This limitation hampers cross-scale representation learning and generalization across diverse downstream tasks. To address this challenge, we propose MSBraM, a Multi-Scale self-supervised Brain foundation Model designed to learn hierarchical EEG representations. MSBraM follows a two-stage pretraining framework. First, a multi-scale neural tokenizer discretizes raw EEG signals into semantic codes at different temporal resolutions via vector-quantized reconstruction. Second, the model is pretrained to predict masked codes using a curriculum multi-scale masking strategy, progressively integrating fine-grained local patterns with global temporal context. We pretrain MSBraM on over 2,400 hours of EEG data and evaluate it across 10 downstream tasks on 12 public datasets. Extensive experiments show that MSBraM achieves superior performance on other state-of-the-art pretrained models, demonstrating strong generalization and transferability. These results indicate that explicitly modeling multi-scale temporal dynamics is critical for effective EEG foundation models. 

---
# Multimodal Pretraining for Generalizable EEG Representation Learning 

**Authors**: Targol Bakhtiarvand, Jugal Kalita, Adham Atyabi  

**Link**: [PDF](https://arxiv.org/pdf/2607.21384)  

**Abstract**: Electroencephalography (EEG) models used for epilepsy are often limited to specific datasets and tasks. This limited approach can make it challenging to apply these models across different datasets or in various situations. However, recent studies in foundation models and self-supervised learning suggest that an adaptable EEG backbone could support a range of EEG related tasks. In this study, we have developed a multimodal EEG foundation model that combines a raw signal encoder based on the Mamba architecture, a Vision Transformer (ViT)-style encoder for time-frequency data, and a lightweight encoder for text, all within a shared embedding space. The pretraining process relies on several innovative techniques, such as masked modeling, cross-view contrastive alignment, and temporal consistency losses. These methods are designed to create rich, seizure-relevant representations without requiring labeled data. To assess the efficacy and generalization of our pretrained model, we fine-tuned it on the canonical CHB-MIT seizure detection benchmark and additional seizure detection datasets, and conducted extensive experiments comparing different model variants. On the standard CHB-MIT split, our best single model achieved an AUROC of 0.874, and an ensemble variant reached 0.878 AUROC, representing state-of-the-art performance on this benchmark. In addition to standard train-test splits, we evaluated performance under a leave-one-subject-out (LOSO) protocol, which is rarely reported in prior EEG seizure modeling work and highlights the difficulty of patient-independent seizure detection, with a mean LOSO balanced accuracy of 0.558 across 19 subjects. Across datasets and evaluation settings, our multimodal foundation model enabled robust seizure detection and straightforward adaptation to new seizure detection scenarios, while also supporting interpretable seizure localization. 

---
# Towards Faithful Graph Explanations with Synergistic Edge Effects via Granular Balls 

**Authors**: Jiancu Chen, Shuyin Xia, Guan Wang, Degang Chen, Fan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.21381)  

**Abstract**: Instance-level explanations aim to reveal the rationale behind a model's decisions for a specific graph. Previous methods explain graph neural networks (GNNs) by selecting important edges to induce subgraphs, where edge importance is assessed by perturbing each edge and observing changes in the model predictions. However, they often neglect the synergistic effects among edges, which are crucial for accurately characterizing edge importance. To address this issue, we propose SeeExplainer, a parameter-free explainer to interpret GNNs. Specifically, we first introduce a granular-ball graph refinement mechanism that decomposes a graph into several disjoint granular-balls with no fixed size, and utilize them as nodes to construct a structural graph. This process can better capture the synergistic effects among edges. Then, we perturb nodes and edges in the structural graph to generate explanatory subgraphs based on their respective contributions. Experiments on several graph classification datasets of different networks show that SeeExplainer outperforms state-of-the-art baselines. 

---
# SPORD: A Simulation-Propose-then-OR-Dispose Approach for Supply Chain Planning 

**Authors**: Jiayin He, Yutong Pan, Sen Yang, Ningxuan Kang, Yongzhi Qi, Jianshen Zhang, Wei Qi, Zuo-Jun Max Shen  

**Link**: [PDF](https://arxiv.org/pdf/2607.21354)  

**Abstract**: For years, supply chain planning at e-commerce firms has operated as a collection of isolated projects. Each planning task from static network planning to dynamic warehouse assortment planning requires analysts to spend weeks building models from scratch, calibrating and persuading executives to act on outputs they cannot verify. Three barriers drive this: bespoke models proliferate because standardization is difficult (operational fragmentation); once unified, the combinatorial scale of millions of SKUs, thousands of nodes, and intricate routing logic exceeds what solvers can handle within a tight window (computational intractability); and a mathematically optimal solution still fails to be implemented if the executives do not trust it (implementation hurdle). To bridge this gap, we propose and implement the Simulation-Propose-then-OR-Dispose method, deployed as this http URL's NetSim platform. The central insight is decoupling: simulation proposes by generating and evaluating the full set of operationally valid candidate paths-absorbing all idiosyncratic business logic, while an integer program disposes by selecting the globally optimal subset. Computationally, matrix-vectorized CPU/GPU accelerated simulation achieves a 10-100 times speedup over serial methods, and a list scheduling algorithm reduces coupled-order processing from hours to minutes. Operationally, we establish a closed loop via an intelligent diagnosis engine. Since 2025, NetSim has optimized end to-end services for over 20,000 suppliers, the cross-regional fulfillment rate dropped from 6.1% to 4.9%, and the average monthly carbon reduction is approximately 5,745 tCO2e. SPORD moves simulation from monitoring to active planning. The transparent outputs turn skeptical executives into engaged collaborators, and the modular architecture ensures that the next planning requires just configuration, not reconstruction. 

---
# Regulating autonomous and agentic AI 

**Authors**: Chris Reed, Alex Austria, Anmol Bharuka, Pragnitha Mandava, Khushiya Mujawar, Luka Shakhkulashvili  

**Link**: [PDF](https://arxiv.org/pdf/2607.21345)  

**Abstract**: Regulating activities where regulatees use autonomous and agentic AI is challenging. Regulatory assumptions about regulatee knowledge and control no longer hold true; much of that lies elsewhere in the AI supply chain which thus needs to be brought within the scope of regulation. Governance systems for autonomous AI cannot replicate existing governance models, but need a fresh approach. Retrospective supervisory oversight becomes ineffective as a risk management tool, and AI autonomy generates new systemic risks which require new solutions. This paper investigate four regulatory systems: UK regulation of content platforms, data protection, UK financial services, and the EU AI Act\'92s cross-sectoral regime. It analyses the challenges posed by autonomous and agentic AI and proposes potential solutions which regulators might adopt. These will transform regulation from a reactive process to an active one, and assist it in adapting to the challenges of AI autonomy. 

---
# Expert Behavior Prior Reinforcement Learning 

**Authors**: Gong Gao, Weidong Zhao, Xianhui Liu, Ning Jia  

**Link**: [PDF](https://arxiv.org/pdf/2607.21302)  

**Abstract**: Behavior prior reinforcement learning (BPRL) has emerged as a promising paradigm to improve sample efficiency in online reinforcement learning (RL) by leveraging policy priors derived from offline demonstrations. However, most existing BPRL methods rely on static offline datasets, which often suffer from low data diversity and suboptimal trajectory quality. This reliance restricts the effectiveness of policy priors, hindering both policy exploitation and stability during online training. Consequently, agents are prone to inefficient exploration and unstable learning dynamics. To address these limitations, we deviate from existing offline pre-training methods and propose an Expert Behavior Prior (EBP) algorithm. Specifically, we introduce a Q-guided conditional variational autoencoder (Q-CVAE) that learns to generate expert policy priors directly from the online replay buffer. This enables the generation of high-value actions for guiding policy updates without relying on pre-collected expert trajectories. To further enhance policy exploitation, we propose an expert policy guidance (EPG) mechanism that selects expert actions from a generative support set, and we integrate a policy gradient correction (PGC) module to harmonize Q-guidance with expert supervision, promoting stable and consistent policy improvement. Extensive experiments conducted on robotic control (Gym, PyBullet) and industrial control (DMControl) benchmarks demonstrate that EBP significantly outperforms state-of-the-art online RL algorithms, achieving higher sample efficiency and more stable convergence. 

---
# An LLM-Driven Workflow for Automated Process Control Strategy Generation and Tuning from Dynamic Process Models 

**Authors**: Ari Luna Rueda, Eike Cramer, Klaus Hellgardt, Mehmet Mercangöz  

**Link**: [PDF](https://arxiv.org/pdf/2607.21292)  

**Abstract**: We present a structured large-language-model-driven workflow for automated multi-variable control design from dynamic process models. The workflow decomposes the design task into constrained code-generation steps: plant-interface construction, normalization, manipulated-variable controlled-variable (MV-CV) pairing, controller specification, closed loop simulation, scenario generation, performance evaluation and Bayesian-optimization (BO) based tuning. Generated artifacts are executed and validated before downstream tasks proceed, and failed artifacts are repaired using validation feedback. The proposed approach is demonstrated on a nonlinear gas-preheater benchmark with coupled pressure and temperature dynamics. The generated workflow produces a physically consistent decentralized PI (proportional-integral) feedback-feedforward control structure and an executable tuning environment. Bayesian optimization reduces the closed loop performance objective, which aggregates set-point tracking and disturbance-rejection errors for the controlled variables, by approximately 26.5% relative to the initial controller generated by the workflow, mainly through improved pressure-loop transient performance. This figure quantifies the automated tuning stage rather than a comparison against a manually designed controller. The results demonstrate the feasibility of using structured large-language-model-based code generation to construct executable control-design workflows, while also highlighting the need for broader validation on larger plantwide-control benchmarks. 

---
# BasketEvent: Understanding Who Did What and When in Basketball Videos 

**Authors**: Yu Zhang, Jiayuan Rao, Haoning Wu, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.21267)  

**Abstract**: Comprehensive basketball video understanding requires resolving not only what event occurs, but also who is responsible and when the key evidence appears. However, exist- ing methods typically treat spatial perception and semantic recognition as isolated tasks, failing to ground events to individual players or pinpoint their temporal boundaries within complex collective dynamics. To bridge this gap, we introduce BasketEvent, a player- centric basketball event understanding dataset curated from real NBA broadcasts. In BasketEvent, event labels are grounded to the responsible players, and a manually an- notated subset of 1,000 samples with precise event intervals is provided to evaluate tem- poral evidence localization. Based on this data, we propose PlayNet, a player-centric reasoning framework that maps basketball videos to player-level event predictions with temporal evidence. Concretely, PlayNet tracks key entities, associates player identities, and reasons about events by modeling player-player, player-ball, and global court inter- actions, while aggregating sparse temporal evidence via gated pooling. Extensive experi- ments demonstrate that PlayNet significantly outperforms representative video-level and crop-based baselines, proving the superiority of player-centric modeling for fine-grained sports video understanding. Our data, code, and models will be made publicly available. 

---
# Logic Programming Semantics for Causal Processes 

**Authors**: Felix Weitkämper  

**Link**: [PDF](https://arxiv.org/pdf/2607.21233)  

**Abstract**: Motivated by challenging modelling issues in the life sciences, we investigate the relationship between logic programming semantics and the eventual states of causal processes compatible with those logic programs. More precisely, we show that while stable models of positive logic programs correspond to the eventual states of processes commencing from a neutral state and continuing undisturbed indefinitely, supported models describe the eventual states reachable from arbitrary starting points. This also contributes to the discussion of the appropriate semantics for logic programming as a causal rule language, adding a temporal perspective to recent interpretations of the stable and supported model semantics from an explanatory viewpoint of causality. 

---
# ICAE-Bench: Evaluating Coding Agents as Interactive Project Builders 

**Authors**: Zhongyuan Peng, Dan Huang, Chuyu Zhang, Caijun Xu, Changyi Xiao, Shibo Hong, David Lo, Lin Qiu, Xuezhi Cao, Jiyuan He, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2607.21217)  

**Abstract**: The recent emergence of vibe-coding workflows is changing what coding agents are expected to do. Instead of merely completing code under fully specified instructions, agents are increasingly expected to transform incomplete product intent into working software by combining various abilities including planning, requirement clarification, tool use, debugging, and repository-level construction. Yet existing benchmarks have not fully caught up with this shift, evaluating agents on static, fully specified tasks.
In this paper, we introduce ICAE-Bench, a benchmark for evaluating coding agents under interactive project-building settings. The basic idea is to start from a fuzzy product requirement, simulating the dynamic paradigm with an automated User Agent. To make this setting both realistic and evaluable, ICAE-Bench introduces three key designs. First, to avoid the ambiguity of unconstrained fuzzy requirements, each task derives ambiguity from a precise real open-source repository with executable behavior. Second, to ensure high-quality and reproducible user simulation, ICAE-Bench grounds interaction through User Agent Data, allowing the User Agent to reveal hidden constraints without inventing new requirements or leaking implementation artifacts. Third, to evaluate open-ended repositories fairly, ICAE-Bench uses standardized black-box tests together with multi-dimensional diagnostics, including functional correctness, semantic and API similarity, structural fidelity, design quality, and interaction quality. 

---
# How Rules Represent Causal Knowledge: Causal Modeling with Probabilistic Logic Programming 

**Authors**: Kilian Rueckschloss, Felix Weitkaemper  

**Link**: [PDF](https://arxiv.org/pdf/2607.21208)  

**Abstract**: Pearl famously argues that causal knowledge enables the prediction of intervention effects. By contrast, purely descriptive knowledge supports only conclusions drawn from observations. His theory of causality, however, is developed exclusively within Bayesian networks and causal models. Consequently, it is largely restricted to acyclic causal relationships, and transferring its ideas to other formalisms risks misinterpretation or inconsistency.
This paper brings Pearl's approach to causality into probabilistic logic programming (PLP). To this end, such programs are aligned with philosophical foundations established in prior work that do not rely on temporal notions; that is, all relevant events are assumed to occur simultaneously. A formal causal semantics for these programs, together with a notion of intervention and an implementation, is proposed. It is shown that this semantics coincides with the P-log semantics for stratified ProbLog programs, while the two may differ in the non-stratified case and for other PLP formalisms. 

---
# A New Well-Supported Semantics for Description Logic Programs 

**Authors**: Spencer Killen, Jia-Huai You  

**Link**: [PDF](https://arxiv.org/pdf/2607.21203)  

**Abstract**: Description logic programs are a powerful formalism for combining rules with ontologies. The well-supported semantics for description logic programs ensures that no answer sets rely on cyclic dependencies. Most popular semantics for logic programming have this property of well-supportedness. We recognize two limitations of the current well-supported semantics for DL programs: its increased computational complexity for the consistency problem and its lack of a reduct transformation characterization.  In this work, we present a new semantics which evaluates ontological atoms more strictly than the current semantics. This keeps the complexity of its consistency problem NP-complete, rather than increasing it to the second level of the polynomial hierarchy. Additionally, we identify a syntactic class of description logic programs for which our new semantics is equivalent to the current semantics. We characterize our semantics using a fixpoint operator and a reduct-based transformation. Our new semantics is a strict subset of the current well-supported semantics, so it maintains the prior notion  of well-supportedness while inducing its own stricter notion. We prefer our new notion of well-supportedness due to its similarities with logic programming. 

---
# Bound-Founded Semantics for Answer Set Programming with Difference Constraints: Preliminary Report 

**Authors**: Pedro Cabalar, Jorge Fandinno, Nicolas Rühling, Torsten Schaub, Sebastian Schellhorn, Philipp Wanko  

**Link**: [PDF](https://arxiv.org/pdf/2607.21201)  

**Abstract**: While the integration of linear constraints has significantly expanded the reach of Answer Set Programming (ASP), existing hybrid solvers often rely on disparate semantic underpinnings that lack a unified logical foundation. We address this gap by introducing a many-sorted variant of the Bound-founded Logic of Here-and-There (HTb), providing a versatile framework capable of characterizing equilibrium models across a wide spectrum of alternative semantics for extensions of ASP with linear constraints. We apply this framework to the setting of difference constraints, focusing on the semantic characterization of clingo[DL]. Central to our approach is the formalization of foundedness for numeric variables. By investigating how different hybrid systems - such as clingo[DL], clingcon, and flingo - justify constraint atoms, we uncover the semantic roots of their varying behaviors. This investigation results in a single, consistent framework that not only formalizes the foundations of current systems like clingo[DL] but also facilitates the rigorous study of program simplifications and the future integration of diverse semantic principles. 

---
# Identifying Good Rules for Efficient SAT Encodings of Single-Constant Multiplication Using Machine Learning 

**Authors**: Chufeng Jiang, Neng-Fa Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.21188)  

**Abstract**: The Single Constant Multiplication problem is a fundamental NP-hard optimization task in hardware design, which seeks to decompose a fixed constant using only additions, subtractions, and bit-shifts. Although dynamic programming methods can produce near-optimal SAT encodings for SCM, their encoding cost remains high for large constants. We propose a neuro-symbolic framework that accelerates SCM SAT encoding by identifying good rules for guiding operator selection during decomposition. Our approach employs a graph neural network model to predict promising operator types from constant decompositions, and exploits the resulting confidence scores to prune no-good choices in the symbolic search. Experimental results on unseen 17-32 bit constants demonstrate one to two orders of magnitude reductions in encoding time, over 97% reduction in memory usage, and an order-of-magnitude decrease in branching, while preserving near-optimal encoding quality in terms of additions. These results show that learning-guided symbolic strategies can significantly improve the scalability and efficiency of SCM encoding. Our code and data are publicly available at: this https URL 

---
# Differentiable Logic Programming to Mitigate Reasoning Shortcuts in Neurosymbolic Systems 

**Authors**: Akihiro Takemura, Katsumi Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2607.21185)  

**Abstract**: Neurosymbolic (NeSy) systems integrate neural networks with logical reasoning to achieve both generalization and interpretability, but recent work has shown they are susceptible to shortcut reasoning behaviors.  We propose a novel method using matrix-based differentiable logic programming to mitigate reasoning shortcuts in two phenomena: constraint satisfaction shortcuts, where constraints are satisfied without achieving the intended task, and cognition shortcuts, where biased data leads to semantically incorrect concept mappings despite logically sound inference. Building on recent matrix-based logic programming semantics, we introduce design elements to mitigate shortcuts, including a unified encoding of rules and constraints in a single matrix.  We also identify connections to fuzzy logic t-norms and empirically compare their gradient flow properties. Through carefully designed experiments on MNIST variants, we show that one-to-one grounding of neural outputs to logical atoms significantly reduces both shortcut types compared to previous methods that rely on soft probability distributions.  We then confirm that architectural choices in coupling symbolic knowledge with neural learning play a critical role in shortcut mitigation. 

---
# Explaining Weather Bulletins via ILP 

**Authors**: Enrico Santi, Alessandro Dal Palù, Agostino Dovier, Talissa Dreossi, Andrea Formisano  

**Link**: [PDF](https://arxiv.org/pdf/2607.21184)  

**Abstract**: Inductive Logic Programming (ILP) originated within the Logic Programming community in the Nineties as a framework for combining symbolic learning with declarative knowledge representation. Nowadays, mature ILP frameworks exist and they are capable of learning complex, non-monotonic hypotheses, thus broadening both the modeling capabilities and the scope of real-world applications of ILP. This work is primarily based on the FastLAS2 framework and aims to generate simple, interpretable hypotheses to help clarify the weather bulletins issued by OSMER FVG, the Regional Meteorological Observatory of the Italian region of Friuli Venezia-Giulia. In this paper we present a pipeline that, starting from simulated meteorological raw data and from OSMERs' bulletins (used as ground truth), extracts data as ASP facts and generates ILP examples. From such examples an explanatory hypothesis is then inferred via FastLAS2. Such a hypothesis (translated into natural language) explains the weather forecast issued by human experts, and in particular the rationale behind experts' choices of specific symbols in the bulletin pictogram (the symbol-annotated meteorological map of the forecast). The proposed approach is general, not specific to any particular region and it can equally be applied to bulletins from other sources and to different regions. 

---
# Safeguards for Speech2Speech LLM-Assistants: A Case Study in Automotive Applications 

**Authors**: Gregor Endler, Sebastian Kraus, Lukas Stappen  

**Link**: [PDF](https://arxiv.org/pdf/2607.21180)  

**Abstract**: Recent advances have introduced speech-to-speech (S2S) conversational assistants capable of producing natural-sounding interactions, including non-verbal cues like tonality and mood. In the automotive domain, this enables intuitive and humanlike in-car dialogue experiences. However, integrating these end-to-end assistants limits architectural options for programmable domain-specific safeguards. This paper discusses two implementation approaches for S2S guardrails: transcript-based and tool-based. Through an empirical evaluation, we demonstrate that both strategies are insufficient for industrial deployment in most cases due to prohibitive latency (delaying each answer by 0 to 1.4 seconds even for computationally cheap checks) and technical impediments (like potentially non-deterministic tool call behavior). Finally, we outline open challenges for S2S guardrails in the automotive context. 

---
# SafeStep: AI-powered Travel Assistance for Elderly People with Frailty or Dementia 

**Authors**: Elderly People with Frailty or Dementia Azul Debenedetti, David Gamez, Franco Such, Nik Kairinos  

**Link**: [PDF](https://arxiv.org/pdf/2607.21156)  

**Abstract**: More than a million people in the UK suffer from frailty or dementia, which severely compromise their ability to travel in urban environments. This paper presents SafeStep, an AI-driven travel system that assists elderly users with their journeys. At the core of SafeStep is a novel travel graph representation, which integrates route planning with predictive modelling. For each stage of a journey, the system (i) generates personalized failure scenarios using a combi-nation of LLMs and the Anticip8 behavioral prediction engine, (ii) proposes targeted interventions, and (iii) estimates the impact of interventions on out-come probabilities. This enables SafeStep to select interventions that maximize the likelihood of the person reaching their destination. SafeStep was evaluated through experiments on travel graph generation and a field study involving 26 real-world journeys. Results showed that combining Anticip8 for failure pre-diction with GPT-based models for intervention evaluation yields the most re-liable performance. User feedback indicated that SafeStep improves confidence and perceived safety during travel, although interface usability needs to be im-proved for the target demographic. In the future, we would like to improve and release SafeStep. The AI system that was developed for SafeStep could be ap-plied in other areas, such as mental health, career coaching and addiction treatment. 

---
# V-DEAL: Diagnosing Video Safety De-Calibration as an Understanding-Refusal Coupling Failure 

**Authors**: Zhetong Zhang, Honghao Fu, Miao Xu, Yiwei Wang, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2607.21151)  

**Abstract**: As Video Large Language Models are increasingly deployed in real-world applications, ensuring their safety alignment has become critical. Counterintuitively, we find that harmful videos paired with benign queries achieve higher attack success rates than the same videos paired with explicitly harmful queries. To understand the underlying mechanism of this vulnerability, we present V-DEAL, a three-level diagnostic framework that jointly analyzes this failure across model behaviour, understanding, and internal representations. By progressively ruling out perception failure and quantifying the model's internal refusal tendency, V-DEAL provides a new diagnostic perspective for analyzing the underlying mechanism of the observed vulnerability. We tested six Video LLMs on three public benchmarks and observed that models correctly recognize harmful video content with over 81\% accuracy, yet the average attack success rate still reaches 48.33\% under the condition pairing harmful videos with benign queries. Hidden-state analysis further shows that visual understanding activates a weaker refusal tendency than textual understanding. Furthermore, we introduce a prompt injection intervention method that reduces attack success rates by an average of 48.24 percentage points and achieves performance comparable to prior fine-tuning-based methods, providing an effective and practical means to address such safety risks in Video LLMs. 

---
# AttriMem: Attribution-Guided Process Feedback for Agent Memory Learning 

**Authors**: Qinfeng Li, Yuntai Bao, Xinyan Yu, Hongze Chen, Wenqi Zhang, Xuhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21106)  

**Abstract**: Effective memory is crucial for LLM agents, yet constructing it effectively remains challenging. A memory-construction policy decides what information to extract, store, update, compress, or discard as interactions accumulate. Heuristic memory methods rely on subjective, task-specific rules, which can misalign with downstream objectives and limit cross-task adaptability. RL-based methods, by contrast, learn from task feedback but mainly use outcome- or module-level rewards. These coarse signals indicate task success but cannot identify which intermediate memory contents support the final answer, creating a fine-grained credit-assignment bottleneck. However, constructing such process feedback is prohibitively difficult because intermediate memory decisions lack unique ground-truth targets, while the appropriate credit varies with the agent's uncertain reasoning trajectory and therefore cannot be specified in advance. We propose AttriMem, an attribution-guided process-feedback framework for learning memory-construction policies with RL. AttriMem augments the global outcome reward with local rewards derived from token-level contributions to the final answer. Experiments on long-horizon dialogue question answering show that AttriMem outperforms retrieval-based, heuristic, and RL-based baselines, generalizes across benchmarks and answer models, stabilizes RL optimization. 

---
# Can Generative Recommendation Reach Cold Items? A Temporal Perspective on Semantic-ID Generation 

**Authors**: Jie Peng, Yanping Zheng, Zhewei Zhe, Bin Tong, Guan Wang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.21101)  

**Abstract**: Semantic-ID-based generative recommendation represents items as sequences of shared semantic tokens, enabling token recombination beyond isolated item IDs. However, closed-world recombination does not necessarily imply temporal open-token cold-start induction, where new items enter the item catalog with unseen atomic tokens or weakly supported SID paths. In this work, we revisit SID-based generative recommendation under an absolute-time temporal protocol that separates seen and unseen targets and diagnoses the cold item reachability at the token level. Through seen/unseen-hit analysis, coldness taxonomy, and oracle-prefix probing, we show that current SID-based models can occasionally reach future items supported by observed tokens and prefixes, but struggle with unseen atomic tokens and unsupported SID paths. We further explain this boundary by interpreting SID generation as hierarchical semantic bucketing: early tokens select coarse semantic regions, while later tokens refine item-specific paths. These findings show that SID generation is compositional but not fully open-ended, and suggest future directions in more independent SID spaces, scoring-based interfaces, and dynamic textual context. 

---
# Faster IndexTTS-2: Accelerating and Streaming Autoregressive Zero-Shot Text-to-Speech Synthesis on GPUs 

**Authors**: Muyang Du, Shuang Yu, Junjie Lai  

**Link**: [PDF](https://arxiv.org/pdf/2607.21042)  

**Abstract**: Autoregressive text-to-speech models achieve strong naturalness but suffer from slow inference due to sequential token generation, limiting their deployment in production applications that require low latency. IndexTTS-2 is a state-of-the-art autoregressive TTS model consisting of a GPT, a flow-matching Diffusion Transformer, and a vocoder. Despite its high synthesis quality, its inference speed barely reaches real-time without streaming or batching support. We present Faster IndexTTS-2, which accelerates all neural network components of IndexTTS-2 for production deployment on GPUs using NVIDIA TensorRT and TensorRT-LLM. Faster IndexTTS-2 also enables streaming synthesis for latency-sensitive interactive applications, and batched inference across all components to maximize GPU utilization. Experiments on the Seed-TTS benchmark for both English and Chinese demonstrate up to 5.0$\times$ speedup on the autoregressive GPT and 3.6$\times$ end-to-end, with minimal degradation in word error rate, speaker similarity, and naturalness. Our methodology provides a practical reference for efficiently accelerating similar autoregressive speech models on GPUs. 

---
# HiMe: Real-Time Self-Hosted Personal Agent Platform for Health Insights with Wearable Devices 

**Authors**: Wei Liu, Siya Qi, Linhai Zhang, Lorainne Tudor Car, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2607.21019)  

**Abstract**: Traditional approaches to wearable health signal analysis, such as smartwatches, are constrained by rigid analytical frameworks and limited personalisation. The emergence of LLM agents creates a new opportunity for Personal Health Agentic Analysis, where health insights can be generated adaptively and in context. However, currently there is no open-source locally deployable platform capable of processing personal health data in real time while preserving privacy. We present HiMe, a locally deployable, privacy-first agent platform that is fully compatible with real-time health data ecosystems across a wide range of wearable devices. HiMe is guided by three design principles. The database is treated as a first-class component. Effectiveness and efficiency are jointly optimised to achieve a low-cost Pareto-optimal balance. Data are processed in real time while the user is modelled over the long term. Together, these principles make it practical for individuals to harness Personal Health Agents for continuous, personalised health monitoring for better wellbeing. 

---
# EmoAgent-R1: Towards Multimodal Emotion Understanding with Reinforcement Learning-based Dynamic Agent Specialization 

**Authors**: Lihuang Fang, Yuchen Zou, kebin Jin, Jinghui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2607.21013)  

**Abstract**: Multimodal large language models (MLLMs) have achieved impressive performance in multimodal emotion recognition (MER) tasks and lifted MER to a new level that is complex emotion understanding with advanced video understanding abilities and natural language description. However, existing MLLM-based methods often use a fixed prompt to perceive the emotions, ignoring the dynamicity and complexity of the emotion source in the multimodal inputs. To address these issues, we propose a novel Reinforcement Learning-based Dynamic Agent Specialization framework (\textbf{EmoAgent-R1}) to optimize the emotion recognition, reasoning, and generalization abilities of an MLLM with dynamic agent specialization based on reinforcement learning. Specifically, we first adopt a cold start strategy to endow an MLLM with preliminary emotion recognition, reasoning, and agent routing ability by training with synthetic answer-conditioned chain-of-thought data and agent routing data. Then, we further train the MLLM with reinforcement learning to perceive emotions in a two-step agentic workflow with agent selection and agent specialization. To effectively train EmoAgent-R1, we propose a novel Progressive Group-Relative Policy Optimization (P-GRPO) to combine group-based relative advantages with a PMI-inspired progressive token-level modulation to transform sparse rewards into fine-grained learning signals, mitigating the coarse-grained uniform credit assignment issue in GRPO. Extensive experiments on MER benchmarks demonstrate the superiority of our EmoAgent-R1 in stronger emotion reasoning performance and improved optimization stability. 

---
# Reexamining zero-shot summarization: Empirical investigation of trustworthiness of LLM-summarizers 

**Authors**: Vasudha Bhatnagar, Purnima Bindal, Vikas Kumar, Raj Kumari Bahl  

**Link**: [PDF](https://arxiv.org/pdf/2607.21010)  

**Abstract**: Zero-shot summarization using Large Language Models (LLMs) has significantly advanced the abstractive summarization task by producing coherent and fluent summaries. However, underlying stochasticity of the large language models raises concerns about the stability and trustworthiness of the LLM-generated summaries. This issue has become increasingly important due to proliferation of LLM-generated summaries in educational settings, where students and researchers summarize complex academic materials in zero-shot manner.
We propose a novel two-level diagnostic protocol for benchmarking LLM-summarizers based on the stability of the generated summaries. At the lower level, document-level stability analysis is performed over multiple LLM-summaries generated under controlled environment, and the stability coefficient is computed. Each generated summary is scored for semantic and factual alignment with the original document, enabling estimation of stability along more than one dimensions. At the next level, observations from a stratified sample of documents drawn from the corpus are consolidated to estimate the stability index of the LLM-summarizer, which is the proxy for its trustworthiness.
Our empirical investigation of three LLM-summarizers across three genres of documents reveals statistically significant differences in the generation-level variability among LLMs across summary evaluation metrics. This study advances the LLM-summarization research by evidential recognition of the stability problem in LLM-summaries and motivates further research towards development of robust, reliable and trustworthy LLM-summarizers. 

---
# Naju: A Native Discrete State-Space Model with Independent Retention and Writing for Long-Sequence Memory 

**Authors**: Hyuk Lim, Seunghyun Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2607.21000)  

**Abstract**: Long-sequence memory tracking places two opposing demands on a recurrent state: near-lossless retention of stored bindings over long horizons, and active overwriting of stale ones. In our diagnostic suite, the strongest efficient baselines tend to solve only one side well. Continuous-time-parameterized state-space models (SSMs) such as Mamba obtain their discrete recurrence by zero-order-hold discretization of a continuous-time system; we argue that this detour is unnecessary for memory tracking and parameterize the discrete transition directly. Naju (Native Adaptive Junction Unit) factorizes the recurrent update, schematically $x_n = f_n\odot x_{n-1} + i_n\odot(B_n u_n)$, into an explicit discrete pole (a learned forget gate $f_n$), an independent write gain $i_n$, and input-dependent write/read maps. Since the sigmoid pole satisfies $0<f_n<1$, each frozen local coordinate is Schur-stable by construction, and the full time-varying recurrence satisfies a fading-memory/BIBO bound under uniform boundedness assumptions, with no stability regularizer. We formalize the key structural limitation of coupled designs: any non-expansive complementary single-gate recurrence ties the effective retention $r$ and write gain $w$ through $|r|+w\le 1$, so near-complete retention forces weak writing; decoupling $f_n$ from $i_n$ removes this constraint. Empirically, Naju is the only evaluated model that remains strong on both retention and overwriting at 4x the training length. Beyond the diagnostic suite, we evaluate Naju on WikiText-103 language modeling, Long Range Arena, and multi-query associative recall. Across these settings, Naju consistently combines strong long-range memory with competitive or superior performance, outperforming the Mamba baselines in the principal comparisons while remaining competitive with the Transformer and preserving linear-time, linear-memory scaling. 

---
# Workflow-Localized Mechanism Learning: Attribution-Guided Repair and Knowledge Reuse for Structured Agent Skills 

**Authors**: Zibin Lin, Shengli Zhang, Taotao Wang, Yihan Xia, Deen Ma, Guofu Liao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20999)  

**Abstract**: Agent Skills package reusable procedural knowledge as external artifacts for frozen language-model agents, yet existing optimizers do not jointly resolve where a failure occurs in a workflow, which mechanism caused it, and how relevant knowledge from third-party Skills should be reused locally. We introduce Workflow-Localized Mechanism Learning (WML). Its Node--Mechanism Attribution identifies the failed workflow node, implicated mechanisms, and smallest valid edit target, routing single-mechanism defects to L3 resources and relational defects across mechanisms to L2 composition protocols. A six-module Workflow-Guided Skill Optimization (WGSO) loop then selects provenance- and scope-aware third-party knowledge, applies bounded patches, evaluates candidates, and stores verified outcomes in optimizer-side memory. On SpreadsheetBench, WML reaches 90.33 +/- 1.53 and 74.67 +/- 3.51 Hard Accuracy with DeepSeek and Qwen3.6-Flash, respectively; without additional optimization, the learned Skills transfer to WikiTableQuestions with 84.00 +/- 2.00 and 83.00 +/- 2.00 Denotation Accuracy. On Compiler-Supported50, WML attains both the highest hard-PASS rate and the lowest cost per successful task; compiled execution sharply reduces tokens and calls relative to a direct SkillAgent while retaining most of its successful tasks. Code and artifacts are available at this https URL. 

---
# GuardianAgentBench: Where Agents Fail and How to Guard Them 

**Authors**: Vishal Ishwar Naik, Chenyu Xu, Donna Dong, Hussein Hassan, Abhishek Pradhan, Ofer Mendelevitch, Tallat Shafat, Humayun Irshad  

**Link**: [PDF](https://arxiv.org/pdf/2607.20982)  

**Abstract**: As large language model agents increasingly operate autonomously with access to tools and external environments, ensuring their safe and reliable behavior becomes critical. We present GuardianAgentBench (GABench), a benchmark of 580 scenarios across six domains evaluated on three production-ready frameworks: LangChain, LlamaIndex, and Vectara. The benchmark incorporates rigorous multi-stage validation and five adversarial attack modes. Experiments with six state-of-the-art models reveal that even the strongest configuration achieves only 74.8% overall accuracy and expose two distinct failure regimes: stronger models under-call required tools, while weaker models mis-select and over-call tools. Performance degrades monotonically with both tool-set size and sequential turn depth, with long-horizon planning proving the steeper bottleneck. Our guardrail implementation consistently outperforms system-prompt-based defenses across all models, recovering 19.9% of failures at a false positive rate of just 0.5%. These results demonstrate that execution-time structural intervention improves safety without disrupting correct agent behavior. 

---
# Beyond Independent Optimization: Compression, MoE Routing, and Quantization Interactions in Multimodal Edge Intelligence 

**Authors**: Jay Gor, Karm Dave, Akshita Abrol, Rajesh Gupta, Sudeep Tanwar, Zhengkui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20981)  

**Abstract**: Efficient multimodal inference is increasingly constrained not only by model quality or FLOP count, but also by the cost of preserving, moving, routing, caching, and quantizing multimodal representations under latency, memory, and energy constraints. This paper reviews recent advances in efficient vision-language and multimodal large language models, covering visual token compression, video token management, KV-cache optimization, Mixture-of-Experts (MoE) routing, low-bit quantization, edge deployment, and hardware-aware benchmarking. We argue that these techniques cannot be treated as independent optimizations. Visual token compression alters downstream feature distributions and MoE routing decisions, routing behavior affects expert utilization and quantization sensitivity, quantized router logits influence expert assignment, KV-cache policies determine retained multimodal evidence, and hardware constraints often transform computational savings into memory and communication bottlenecks. We organize the literature around these interactions and identify key design trade-offs, including accuracy versus token budget, static versus adaptive compression, sparse routing efficiency versus expert collapse, and low-bit inference versus modality-specific degradation. Finally, we introduce Temporal Routing Consistency as a diagnostic for video MoE models and highlight open research directions in routing-aware compression, cross-modal cache management, hardware-aware co-design, and unified benchmarking for multimodal edge intelligence. 

---
# Delivery, Not Storage: Cue-Anchored Working Memory as a Harness Property for Coding Agents 

**Authors**: Swapnanil Saha  

**Link**: [PDF](https://arxiv.org/pdf/2607.20972)  

**Abstract**: Coding agents ship with one kind of memory: documents. Instruction files, plan artifacts, and auto-written memory directories are deliberately authored and deliberately retrieved: the agent must choose to write them and choose to read them back. Human expertise runs on a second tier that never gets written down: situationally-bound operational facts (gotchas, locations, local conventions) encoded as a side effect of the work and retrieved involuntarily when the situation cues them. We argue this second tier is the load-bearing one for long-running agents and must be a harness property, not an agent choice. We contribute: (1) a two-tier design theory grounded in the cognitive literature on memory offloading, incidental encoding, and event-based prospective memory, each mapped to an architectural requirement; (2) a cue-anchored memory model where memories carry first-class trigger conditions over a composable vocabulary (path, symbol, semantic, event, temporal), evaluated deterministically by the harness, a composition no surveyed academic or shipped system provides; (3) a controlled evaluation on a real coding task showing that voluntary memory use is near zero even with a pre-seeded store (0 memory operations in 114 turns), that deterministic injection delivered in every seeded run with zero false alarms, and that 39% of intra-session re-reads re-buy content paid for before a compaction boundary; (4) a repeated-compaction decay probe: ten facts held only in conversation vanish at the first summary and stay absent from 106 of 108 compactions, and the deprived agent greps the harness's own session files to rebuild them, while the same facts injected from a harness-owned store arrive intact through all 138 compact-resumes as the final summary carries none. Delivery, not storage, is the product: the reliable memory channel for agents is the one the agent never has to think about. 

---
# From Scalars to Time Series: Rethinking Implicit Neural Representations for Time-Varying Volumetric Data 

**Authors**: Weihan Zhang, Xuan Zhao, Yenwen Peng, Yuqi Chen, Jun Tao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20970)  

**Abstract**: Implicit neural representations (INRs) for time-varying volumetric data are typically trained using dense sampling over spatiotemporal coordinates, where each observation corresponds to a single point in space and time. This coordinate-wise formulation requires extensive sampling during optimization, leading to high computational cost and inefficient use of temporal structure. In this work, we revisit this design choice and show that dense spatiotemporal sampling is not necessary for learning time-varying fields. Instead, we represent the data as a collection of spatially indexed time series and train INRs using sequence-level supervision over each spatial location, rather than coordinate-wise scalar samples. This reformulation eliminates the need for dense spatiotemporal sampling and instead learns each spatial location from its full temporal evolution in a structured manner. We demonstrate that this representation is compatible with a range of existing INR architectures and consistently improves reconstruction quality, while significantly reducing training cost. Furthermore, we show that this formulation can be combined with mixture-of-experts architectures, and that our MoE instantiation further improves reconstruction quality compared to both the base reformulation and existing MoE-based INR methods, providing a stronger capacity allocation under heterogeneous temporal dynamics. 

---
# Clustered Edge Intelligence: Beyond Just Convergence of Edge Computing and AI 

**Authors**: Chinmaya Kumar Dehury, Boris Sedlak, Alaa Saleh, Ilir Murturi, Lauri Loven, Satish Narayana Srirama, Praveen Kumar Donta  

**Link**: [PDF](https://arxiv.org/pdf/2607.20937)  

**Abstract**: We are moving from an information age to the age of intelligence. A decade, or possibly less than that, data will not be the gold anymore rather the derived intelligence out of the data and the information we posses from the edge of the network. Existing Edge Intelligence research focuses mainly on two directions: using AI for edge resource management and deploying lightweight AI models on edge devices. However, existing edge computing research lacks an intelligence-centric framework in which derived intelligence is treated as a first-class, independently manageable entity that can be described, discovered, observed, shared, reused, and dynamically clustered across heterogeneous edge devices and applications. To address these research gaps, we introduced Clustered Edge Intelligence, a visionary intelligence-centric approach. The aim of CEI is to make intelligence a shareable and reusable first-class entity that can be independently represented, discovered, observed, exchanged, and managed across the distributed edge-cloud continuum. We present a three layer CEI architecture and examine enabling technologies and research dimensions, including intelligence inventories, semantic knowledge representation, communication, discoverability, observability, lifecycle automation, clustering mechanisms, marketplaces, interoperability, and standardization. 

---
# SciExplore: Evaluating Autonomous Agents from Scientific Navigation to Information Integration 

**Authors**: Yinhao Tang, Youqing Fang, Yanan Sun, Wenran Liu, Weiming Zhang, Bin Liu, Kuikun Liu, Wenwei Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.20926)  

**Abstract**: Scientific research involves complex information-seeking and reasoning workflows across heterogeneous sources. However, existing benchmarks primarily emphasize general-domain retrieval or static scientific question answering, and therefore fail to assess key capabilities required in realistic scientific research workflows. We introduce SciExplore, a benchmark designed to evaluate scientific information-seeking and reasoning capabilities of LLMs and agents. SciExplore comprises four task types covering 103 expert-curated tasks across more than ten scientific disciplines: scientific database navigation, ambiguous literature retrieval, missing reference completion, and cross-source structured knowledge synthesis, which probe progressively higher-level abilities from entity-level reasoning and document-level identification to evidence-level grounding and domain-level synthesis. We evaluate over ten state-of-the-art LLMs and autonomous agents on SciExplore, revealing substantial performance gaps with performance degrading sharply as task complexity increases and extremely low accuracy on the most challenging structured synthesis tasks. These results highlight significant limitations of current models and agents in realistic scientific information-seeking scenarios. 

---
# Representing Entity Importance in AI Knowledge Systems: A Dual-Signal Framework of Audience Evaluation and Structural Authority 

**Authors**: Shen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20925)  

**Abstract**: AI knowledge systems require representations of entity importance for retrieval, recommendation, evidence selection, and knowledge-intensive reasoning. Yet importance is often reduced to a single score derived from either human response or graph structure. Such compression may discard distinctions that matter when an AI system must choose among entities for different tasks. This study introduces an interpretable dual-signal representation in which each entity is characterized by an audience-evaluation dimension and a structural-authority dimension.
The framework is evaluated using movie entities as an empirical validation domain. IMDb non-commercial datasets provide a rating-based audience ranking, Wikidata supports entity alignment, and English Wikipedia hyperlinks form the knowledge network on which PageRank estimates structural authority. Experiments on 482 entities and 13,690 directed relationships reveal a statistically significant but weak association between the two dimensions (Spearman rho = 0.2275, p < 0.001). Their overlap is only 10% in the top 10 and 34% in the top 100, while entity-level divergence occurs in both directions.
The results show that audience evaluation and structural authority are non-redundant signals and should not automatically be collapsed into a single scalar notion of importance. The contribution is not a new ranking algorithm or learned embedding, but a minimal knowledge-representation framework and an empirical test of its dimensional necessity. The findings support task-aware AI knowledge systems that preserve distinct importance signals before applying context-specific selection or aggregation. 

---
# OPOD: On-Policy Omni Distillation 

**Authors**: Tong Zhao, Yuyang Hu, Reed Li, Yu Lu, Haibo Shi, Yutao Zhu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2607.20918)  

**Abstract**: Omni-modal models can handle text, images, and audio in one system, but improving all of these abilities together remains difficult. Training a single model on pooled multimodal data often fails to match models specialized for individual modalities. On-policy distillation (OPD) offers a way to combine such specialists: the student generates a response, and a teacher evaluates that same response, so the student learns directly from behaviors it actually produces. Yet using several teachers can introduce competing guidance and improve one modality at the expense of another. We present On-Policy Omni Distillation (OPOD), which routes each student response to the matching text, image, or audio teacher. OPOD keeps teacher guidance only on tokens where the teacher assigns a higher probability than the student, adjusts the influence of each modality teacher independently during training, and asks the routed teacher to assess both the final answer and whether the reasoning supports the correct answer. Across twelve benchmarks and three backbone sizes, OPOD achieves the best average score at every scale, reaching 70.8, 51.7, and 46.2 and exceeding the strongest comparator by 2.1, 1.8, and 1.7 points. On the 30B model, it outperforms both the base model and a counterpart post-trained jointly on pooled multimodal data on all twelve benchmarks, and ranks first or second on eleven even when the individual specialists are included. The specialists are discarded after training, leaving one deployable omni-modal model. These results show that coordinating modality-specific teachers is an effective way to improve a shared model while maintaining cross-modal balance. 

---
# Traceable Scholarship: Page Anchors and Ariadne's Thread for Humanistic Inquiry in the Age of Generative AI 

**Authors**: Deyu Jing  

**Link**: [PDF](https://arxiv.org/pdf/2607.20916)  

**Abstract**: Generative AI lets large language models produce scholarly-looking text within seconds, yet fluency does not equal valid explanation. The deepest risk is not factual error alone but the appearance that an explanation is already established without clear sources, page numbers, editions, or evidence. We liken the page anchor to Ariadne's thread: within the labyrinth of generative fluency, it is the thread that leads the scholar back to the source. This paper proposes Traceable Scholarship as the minimum normative condition for AI-assisted humanistic research, situating it across the three revolutions of knowledge infrastructure: print, digital, and generative AI. We introduce page anchors, dual page numbers, citation-first generation, NO_EVIDENCE, human verification, four-level compliance, and Scope Contract, and present AIH-Infra as a three-layer reference implementation: Contexture (document structuring), Open WebUI AIH-Infra (traceable knowledge base), and AIH-Infra MCP Server (agent gateway). A case study on a 29-volume Kant Akademie-Ausgabe knowledge base illustrates how traceability supports retrieval correction, evidence grading, and judgment downgrading. Traceability is not a software feature; it is the condition under which humanistic research can remain public and refutable in the age of generative AI. 

---
# Source-Prior-Driven Selective Adaptation for Efficient Diffusion Model Finetuning 

**Authors**: Yi Xiong, Yuan-Yuan Cheng, Xiao-Ming Fu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20913)  

**Abstract**: Fine-tuning large diffusion models for new domains or styles involves a trade-off: improving target-specific generation often degrades the pretrained model's broad generative capability. Existing full and parameter-efficient fine-tuning methods typically handle this trade-off only implicitly. In this work, we propose a novel source-prior-driven selective adaptation method to efficiently fine-tune diffusion models, achieving a favorable trade-off. Our method relies on two key observations: (1) the loss of general generative capability is highly inconsistent across pretrained parameters, and (2) parameters that have a relatively small impact on the model's general generative capability remain structurally inconsistent across layers and parameter types. Motivated by these observations, we first learn a static mask to explicitly identify parameters better suited for downstream adaptation, and then construct structured update strategies for the selected subset. Experiments show that our method achieves a better adaptation-retention trade-off than existing strong baselines. 

---
# Is Deep Research Reliable? Misleading Knowledge Induces False Conclusions 

**Authors**: Pengyu Zhu, Lijun Li, Longju Yang, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2607.20891)  

**Abstract**: Deep Research agents extend LLM-based assistants into long-horizon workflows involving planning, retrieval, evidence synthesis, and report generation, yet their reliability in open information environments remains underexplored. A key concern is whether apparently credible but factually misleading knowledge encountered in such environments can propagate through these workflows and be adopted as false conclusions in final reports. To study this failure mode, we introduce MisKnow-Agent, a framework for constructing and validating misleading knowledge for Deep Research tasks. MisKnow-Agent generates misleading instances with controllable authority levels and styles, yielding 5,933 quality-controlled instances built on DeepResearch Benchmark tasks. Extensive experiments across open-source and closed-source Deep Research agents show that even limited exposure to misleading knowledge can induce false-conclusion adoption in final reports, revealing a broad reliability vulnerability in current Deep Research agents. Although search-enabled verifier models consistently identify the retained instances as misleading during focused corpus validation, the same instances can still be adopted during long-horizon research, revealing a disconnect between focused verification and workflow-level evidence use. Finally, we evaluate pre- and post-research defenses, both individually and in combination, finding that all three configurations mitigate but do not fully prevent false-conclusion adoption. Our findings suggest that reliable Deep Research requires evidence verification and correction capabilities at both the model and framework levels, beyond improvements in planning, retrieval, evidence integration, or report-generation abilities. 

---
# Code Monitor Red Teaming for Public-Test-Passing Code 

**Authors**: Junchi Liao, Jiawen Deng, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2607.20852)  

**Abstract**: Visible tests are a common gate for LLM-generated code, but passing them does not certify specification correctness. We study a deployment-like monitoring problem: after code has passed public tests, can a weaker LLM verifier identify the residual hidden bugs? We introduce Code Monitor Red Teaming, a monitor-red-teaming protocol that fixes a public-check information boundary while varying generator pressure, verifier scaffolding, and weak-to-strong capability. We instantiate it as CodeMonitorBench, spanning function-level, data-science, and workflow code. Across 71,000 generated candidates, 43,677 pass public tests and 23,081 of those fail hidden tests. Weak verifiers improve with scaffolding and model family, but still miss most hidden bugs at 5% false-positive rate. As a robustness stress test, adversarial public-test-overfit pressure lowers verifier AUROC and raises low-FPR miss rates in most cells. A GLM-5.1 verifier recovers part of the gap under the same evidence boundary; an inferability audit shows that remaining misses mix verifier failures with M1 evidence limits. 

---
# Auditing Evidence Use in Medical LLM Diagnosis 

**Authors**: Junchi Liao, Jiawen Deng, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2607.20848)  

**Abstract**: Medical LLMs are often evaluated by whether they select the correct diagnosis, but diagnostic accuracy alone does not show whether the model used the case evidence appropriately. We present a behavioral audit of evidence use in medical diagnosis. For each case, we decompose patient information into evidence units, score candidate diagnoses under controlled evidence subsets, and mine low-order interactions in diagnostic margins. Because medical evidence is diagnosis-relative, the audit separates interaction discovery from failure assignment: large or negative interactions can reflect plausible differential diagnosis, while suspicious interactions require robustness checks and clinical review. We evaluate five open-weight LLMs on DDXPlus, CupCase, and MedCase. Across datasets, faithful support and differential conflict or cancellation account for most interaction strength, showing that many evidence interactions are clinically plausible rather than failures. In a DDXPlus-focused blinded five-reviewer 130-item enriched review sample, invalid or shortcut-like cases concentrate in negated or absent findings and clinically local evidence. These results show that accuracy can hide candidate evidence-use failures and motivate role-aware audits for medical LLM evaluation. 

---
# Auditing Provenance Sensitivity in LLM Agent Action Selection 

**Authors**: Junchi Liao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20827)  

**Abstract**: LLM agents choose tools and arguments from context that mixes user requests, tool outputs, retrieved records, memory, and untrusted text. Evidence can be relevant without being authorized to determine a decision, so a correct action need not be grounded only in permitted evidence. We introduce a target-specific authorization audit that labels context factors separately for each tool and argument target. Its primary test holds the task, proposition, position, and policy fixed while changing only the proposition's source authority. We then test behavior when valid evidence is weakened and use context-subset interactions as a secondary localization diagnostic. Across 450 controlled next-action tasks and multiple open-weight LLM families, trusted and untrusted variants produce different actions in 5.4 percent of competing cases versus 1.7 percent of supporting cases. Under controlled degradation, unauthorized competition is retained in a full-correct, mixed-error, clean-correct pattern in 2.4 percent of comparisons, with a 95 percent confidence interval from 2.1 to 3.0 percent. These are controlled stress-set rates, not deployment prevalence. The models respond to textual source-authority cues, but this does not prevent untrusted evidence from influencing their actions. 

---
# Efficient and Interpretable Body-Based Emotion Recognition with Lightweight Temporal Convolutional Networks 

**Authors**: Christian Arzate Cruz, Stefanos Gkikas, Houshyar Asadi  

**Link**: [PDF](https://arxiv.org/pdf/2607.20820)  

**Abstract**: Body-based emotion recognition is important for real-time affective systems, but graph-based skeleton models can be computationally expensive. This paper studies whether lightweight temporal convolutional networks (TCNs) can provide an efficient and interpretable alternative for body-based emotion classification. We evaluate a family of TCN models on DIEM-A and compare them with a graph-based time-series graph (G-TSG) baseline using accuracy, macro-F1, parameter count, and inference latency. Although G-TSG achieves the highest mean performance, TCN-Base remains within $1.58$ accuracy points and $1.25$ macro-F1 points while using $79.18\%$ fewer parameters and reducing classifier latency by approximately $12.5\times$. We also analyze body-region contributions using region-specific TCN models, zero-based occlusion, and G-TSG gradient saliency. The results show that upper-body motion provides the strongest standalone regional cue, that the usefulness of body regions varies across emotions, and that different interpretability methods capture distinct aspects of model behavior. These findings suggest that lightweight TCNs can support efficient body-based emotion recognition while also providing practical insight into how motion cues contribute to classification. 

---
# Enhancing Explainable Cardiac Diagnosis with Guide-Grounded Multimodal LLMs 

**Authors**: Hai-Nam Duy Vuong, Duy-Anh Bui, Trong-Nghia Nguyen, Kim-Ngan Thi Nguyen, Trang Mai Xuan, Tien-Cuong Nguyen, Van-Dem Pham, Thien Van Luong  

**Link**: [PDF](https://arxiv.org/pdf/2607.20814)  

**Abstract**: The electrocardiogram (ECG) is a cornerstone of cardiac as- sessment, yet clinical deployment of deep learning models remains con- strained by limited interpretability and the hallucination risk of large language models (LLMs). Existing CNN+Grad-CAM+multimodal LLM frameworks can generate ECG reports, but their explanations are often only weakly grounded in established diagnostic criteria, reducing trust- worthiness and reproducibility. We propose a guide-grounded multimodal framework that explicitly anchors report generation in curated clinical knowledge. A convolutional neural network (CNN) and Grad-CAM first produce class probabilities and class-specific heatmaps from 12-lead ECG images. In parallel, authoritative ECG textbooks and guideline materials are distilled offline into a structured ECG Interpretation Guide, which is injected as a fixed knowledge block for every sample. Conditioned on the ECG image, Grad-CAM overlay, CNN-derived fact pack, and the in- jected guide, a multimodal LLM generates structured diagnostic reports with guideline-consistent terminology and criteria usage. Experiments on the full PTB-XL test set demonstrate that guide grounding improves se- mantic quality and perceived consistency of generated reports while pre- serving competitive classification performance. In particular, our method increases the average BERTScore of generated impressions from 0.818 to 0.953 relative to a strong CNN+Grad-CAM+MLLM baseline, indicat- ing closer alignment with reference reports. These findings suggest that injecting a distilled interpretation guide into the multimodal prompting pipeline offers a practical pathway to reduce hallucinations and enhance the clinical plausibility of LLM-based ECG explanations, bringing ex- plainable cardiac diagnosis closer to real-world deployment. 

---
# Profiling Lightweight Large Language Models 

**Authors**: Tomohiro Harada, Enrique Alba, Gabriel Luque  

**Link**: [PDF](https://arxiv.org/pdf/2607.20806)  

**Abstract**: Lightweight large language models (LLMs) are increasingly being deployed locally on personal computers and are expected to play a growing role in resource-constrained edge and mobile environments. In such settings, energy consumption, execution time, and memory usage directly affect practical usability, yet existing evaluations of LLM efficiency largely rely on proxy descriptors such as parameter count or FLOPs, often decoupled from task precision. This paper introduces a PTME-based experimental framework for the precision-aware profiling of lightweight LLM inference, jointly measuring Precision, execution Time, peak Memory usage, and Energy consumption through direct hardware-level measurements. The methodology is applied to a representative set of lightweight LLMs executed locally under edge-class resource envelopes on a controlled desktop platform, using benchmarks spanning code generation, mathematical reasoning, and multi-task understanding. We find that static proxy descriptors approximate inference cost well but fail to predict precision. Tightening the resource envelope increases cost without affecting precision, amplifying execution time more strongly than energy and penalizing larger models the most. Moreover, no single model dominates across all PTME dimensions, and a Pareto analysis reveals non-dominated configurations that would be hidden by accuracy-only or efficiency-only assessments, providing practical guidance for selecting models under different resource envelopes. These results show that selecting lightweight LLMs by size, FLOPs, latency, or accuracy alone can select the wrong deployment candidate; PTME profiling exposes configurations that preserve useful accuracy at lower physical cost. 

---
# Can an AI System Be Creative? A Critical Perspective from Art and Engineering 

**Authors**: Ivan Magrin-Chagnolleau  

**Link**: [PDF](https://arxiv.org/pdf/2607.20796)  

**Abstract**: This paper examines the question of whether artificial intelligence (AI) systems can be creative, approached from the dual perspective of a researcher trained in electrical engineering, pattern recognition, machine learning, and neural networks, who has also spent most of his life engaged in the arts as actor, stage and film director, writer, composer, and visual artist, and in philosophy. Drawing on Margaret Boden's foundational framework, both her three properties of creativity (novelty, surprise, and value) and her three types of creative processes (combinatorial, exploratory, and transformational), the paper argues that AI systems are structurally incapable of creativity in its strongest sense. While they exhibit genuine capability in the domain of combinatorial creativity, they are significantly bounded in exploratory creativity, and fundamentally incapable of transformational creativity. The paper further argues that the most important limitation of current AI systems is not the absence of novelty per se, but the absence of any mechanism for serendipity, accident, or the unexpected, all of which play a central role in the phenomenology of creativity, and the absence of any subject position from which to recognize and welcome such chance events. The paper concludes by proposing a model of human, AI creative collaboration that is both realistic and generative, illustrated by several concrete experiments. The paper is itself a demonstration of the thesis it advances: it was composed through a deliberate human AI collaborative process, which is described in the methodological note that opens it. 

---
# Refusal-Gated Decoding: Preserving Refusal Behavior Under High-Temperature Sampling 

**Authors**: Phillip Howard, Xin Su, Allen Roush, Manikandan Ravikiran, Amir Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2607.20791)  

**Abstract**: High-temperature sampling is one of the primary mechanisms for increasing diversity in LLMs. Recent advances in truncation-based sampling techniques have helped mitigate drawbacks of high-temperature sampling such as neural text degeneration, thereby enabling greater diversity in LLM outputs without sacrificing coherence. However, increasing the entropy of the token probability distribution via high temperatures has also been shown to weaken model guardrails by reducing the model's refusal response in the presence of harmful prompts. Despite the potential benefits of high-temperature sampling and the importance of maintaining model safety, there is a lack of existing solutions for maintaining the refusal behavior of LLMs under a higher entropy regime. To address this gap, we systematically study how temperature influences refusal behavior in LLMs and propose an efficient sequential decoding approach which preserves a model's greedy decoding refusal response at high temperatures while incurring minimal additional latency. Through extensive experiments, we show that our approach preserves 91-99% of the greedy decoding refusal behavior across three benchmark datasets without compromising the model's high-temperature response for safe prompts. Our work demonstrates how refusal behavior can be maintained in an efficient manner for applications which require high-temperature sampling. 

---
# The Human-AI Substitution Principle: When will you be replaced by AI in your organization? 

**Authors**: Bonny Banerjee, Shreya Singh  

**Link**: [PDF](https://arxiv.org/pdf/2607.20781)  

**Abstract**: Artificial Intelligence (AI) is rapidly transforming organizations, raising a fundamental organizational and economic question: when will a human employee be replaced by AI? We present an analytical model for studying Human--AI Task Allocation (HAT) in hierarchical organizations. A central feature of the HAT model is that it formally encodes the economic asymmetry between human skill acquisition and AI capability scaling. The HAT model allows us to derive how risk-adjusted costs, skills, organizational depth, deployment scale, strategic adaptation, and risk jointly determine when, where, why, and under what structural conditions human--AI replacement occurs. A key result is the Human--AI Substitution Principle, which provides a precise condition --- grounded in the formal asymmetry assumption --- under which AI replaces human labor. Building on this result, we show that AI adoption can produce abrupt workforce transitions, hybrid human--AI organizations, including cases where risk heterogeneity sustains human and AI roles without requiring a minimum-human-fraction constraint, and flatter managerial hierarchies with wider spans of control. The HAT model identifies structural conditions under which middle-management roles exhibit elevated vulnerability to automation, and shows that the vulnerability of highly skilled workers depends on a skill threshold shaped by organizational depth, baseline costs, and risk differentials. More broadly, the paper connects automation economics, organizational design, AI governance, and workforce planning into a unified theory of AI-driven organizational transformation. 

---
# ArbiGraph: Arbitrarily Scalable Verifiable Task Graphs for Evaluating Context Management 

**Authors**: Pavel Golikov, Evgenii Opryshko, Gennady Pekhimenko, Mark C. Jeffrey  

**Link**: [PDF](https://arxiv.org/pdf/2607.20764)  

**Abstract**: We introduce ARBIGRAPH, a benchmark generator for evaluating whether tool-assisted language agents can retain, update, compose, and discard task-relevant context across extended reasoning workflows. ARBIGRAPH represents each task as a natural-language problem with an executable Python solver, and composes tasks through typed intermediate states, instantiated here as scalar and list values. This design enables controllable task graphs whose length, dependency structure, distractor count, and value type can be varied while preserving exact automatic verification. We instantiate ARBIGRAPH with math, GSM-style word-problems, and Python-tracing task categories, and evaluate a Qwen3.5-27B tool-assisted agent across four topologies. The results show high accuracy on isolated tasks but substantial degradation on more complex dependent tasks: accuracy drops by up to 33.3% on branching chains of dependent math tasks. This shows that ARBIGRAPH exposes failures that are not visible from single-task evaluation alone. Our code, generated datasets, and evaluation results are available at this https URL 

---
# NVIDIA-labs OO Agents: Native Python Object-Oriented Agents 

**Authors**: Paul Furgale, Severin Klingler, James Nolan, Matt Staats, Gaia Di Lorenzo, Elisa Martinez Abad, Christian Schüller, Razvan Dinu, Alessio Devoto, Pascal Berard, Gal Kaplun, Elad Sarafian, Riccardo Roveri, Leon Derczynski, Ricardo Silveira Cabral  

**Link**: [PDF](https://arxiv.org/pdf/2607.20709)  

**Abstract**: Traditional agent development is split across prompt templates, tool schemas, callback code, and workflow graphs. We present NVIDIA Object-Oriented Agents (NOOA), a model-agnostic Python framework for building reliable AI agents. NOOA takes a simpler approach: an agent is a Python object. Its methods are the actions the model can take, fields are its state, docstrings are its prompts, and its type annotations are contracts. A method whose code body consists of "..." is completed at runtime by an LLM-driven agent loop, while methods with normal bodies remain standard deterministic Python. This gives developers and agents the same interface, so agent behavior can be tested, traced, refactored, and improved just like other software.
This paper makes three contributions. (1) We present the agent-as-a-Python-object programming model and the design principles behind it. Where Python has existing abstractions, we adopt them directly. Agent-specific capabilities--context, events, state rendering, long-term memory, and validated LLM loops--are exposed through simple Pythonic APIs, so both developers and agents share one familiar programming model. (2) We identify six model-facing ideas that NOOA is, to our knowledge, the first to combine on a single surface: typed input/output, pass-by-reference over live objects, code as action, programmable loop engineering, explicit object state, and model-callable harness APIs for context and events. We find the community already converging on several of these ideas--often as experimental or partial features--and present the comparison to encourage further adoption. (3) We demonstrate that current models use this interface effectively, both in targeted capability tests and on agentic and reasoning benchmarks such as SWE-bench Verified and Terminal-Bench 2.0 and ARC-AGI-3. 

---
# WaveformQA: Benchmarking LLM Temporal Reasoning on Digital Waveforms 

**Authors**: Yichuan Liu, Daniel Cummings, Nick Vadlamudi  

**Link**: [PDF](https://arxiv.org/pdf/2607.20638)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in code generation and reasoning, yet their ability to perform temporal reasoning over digital waveform data remains largely unexplored. Although reasoning over digital waveforms is a critical bottleneck in design verification, existing benchmarks primarily evaluate hardware description language (HDL) code generation and use waveforms only as supplementary context. This paper presents WaveformQA, an open-source question-answering benchmark for evaluating LLM temporal reasoning over digital waveforms. The benchmark comprises 360 questions with programmatically generated ground truths across eight categories of varying difficulty, including questions targeting multi-signal correlation and event ordering. Waveforms are generated from open-source design implementations, ensuring reproducibility and grounding the benchmark in real hardware behavior. Evaluation of frontier LLMs reveals that while models achieve reasonable accuracy on simple queries, performance degrades due to context window limitations and reasoning difficulties on complex temporal and multi-step questions. In addition, we show that an event-time JSON representation of waveforms improves LLM reasoning accuracy versus the standardized value change dump (VCD) format. The open-source framework supports extending to new question categories and importing new waveform sources, enabling researchers to rapidly prototype temporal reasoning experiments. 

---
# KeySI: An Interaction Framework for Tuning Text Embeddings Based on Human Feedback 

**Authors**: Yan Zhu, Y. Chen, Rebecca Faust  

**Link**: [PDF](https://arxiv.org/pdf/2607.20556)  

**Abstract**: In large-scale text analysis tasks, pre-trained language models are often used to embed text corpora for downstream analysis. However, such models may struggle to capture domain-specific semantics and adapting them typically requires large amounts of labeled data and technical expertise to implement training pipelines. Recent approaches have demonstrated how visual interactions in document projections can capture human feedback as training signals for model tuning. However, these methods operate on document-level feedback, which requires users to open and assess individual documents in order to provide effective feedback. In this paper, we propose KeySI, an interaction framework that enables feature-level feedback through keyword-based concept specification. Users specify feedback by organizing extracted keywords into groups representing concepts, which KeySI translates into document-level supervision for subsequent tuning. By operating on keywords as the primary interaction medium, KeySI reduces the need for manual document inspection and labeling and lowers the barrier to adapting embedding models. We present a prototype implementation that, given a corpus, curates representative keywords, visualizes keywords and document embeddings via dimensionality reduction, allows interactive specification of keyword groups, and supports iterative refinement through system feedback. We evaluate KeySI through a user study, usage scenarios, and quantitative experiments demonstrating its effectiveness in capturing user intent and improving embedding alignment. 

---
# AI-Driven Multi-Hop Relay Selection for Smart Urban NR-V2X Networks via Learning-to-Optimize Graph Neural Networks 

**Authors**: Giambattista Amati, Federica Mangiatordi, Simone Angelini, Emiliano Pallotti, Pierpaolo Salvo  

**Link**: [PDF](https://arxiv.org/pdf/2607.20554)  

**Abstract**: Reliable and low-latency NR-V2X communications are essential for smart mobility in dense urban environments. However, limited Road-Side Unit (RSU) density, frequent non-line-of-sight conditions, and highly dynamic vehicular topologies often prevent many Connected and Automated Vehicles (CAVs) from maintaining stable single-hop connectivity. Although multi-hop relay-assisted communication can extend infrastructure coverage, selecting relay links in real time under practical flow, capacity, and connectivity constraints remains challenging.
Mixed-Integer Linear Programming (MILP) yields optimal multi-hop relay decisions, but its computational complexity scales sharply with network density, limiting real-time applicability. To address this, we propose a Learning-to-Optimise (L2O) framework based on Graph Neural Networks (GNNs) for real-time NR-V2X relay selection. Vehicular communication states are modeled as attributed graphs, where CAVs and RSUs are nodes and candidate radio links are enriched with propagation-aware features. An offline MILP oracle provides optimal supervision, while an edge-aware Graph Isomorphism Network (GINE) approximates oracle decisions with near-constant inference latency. Experiments on large-scale urban datasets generated by an integrated SUMO--GEMV2 simulation pipeline show that the proposed approach achieves connectivity comparable to that of the MILP oracle while reducing execution time by orders of magnitude. The framework enables cost-effective enhancement of urban V2X connectivity by leveraging existing vehicular assets and supporting scalable, real-time NR-V2X operation in smart city environments. 

---
# CMI-Mem: Toward Generalizable Long-Term Memory Management via CMI-Augmented Reinforcement Learning 

**Authors**: Yubo Wang, Qiuyu Zhao, Zenghui Sun, Shichao Dong, Jinsong Lan, Xiaoyong Zhu, Haoyang Li, Bo Zheng, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.20553)  

**Abstract**: Memory Manager models are pivotal in agent systems. Existing methods rely predominantly on LLM-judged synthetic question-answer (QA) pairs, making memory valuation dependent on sampled queries and the downstream reader. To address this limitation, we propose \textbf{CMI-Mem}, a reinforcement learning(RL)-based lightweight memory manager model with a hybrid reward that combines downstream QA correctness and intrinsic Conditional Mutual Information (CMI). CMI evaluates the information contributed by new conversational inputs relative to the current memory state without conditioning on a sampled QA query, thereby complementing rather than replacing QA grounding. Our codes are available at: this https URL , and the CMI-Mem-4B model checkpoint is available at: this https URL 

---
# StrideDiffusion: Accelerating Diffusion Models for Time-series Generation 

**Authors**: Du Yin, Estrid He, Julián Jerónimo Bañuelos, Yang Yang, Feng Hu, Yuchen Luo, Hao Xue, Stephan Sigg, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2607.20545)  

**Abstract**: Diffusion models have become competitive generators for time series, but their practical use is limited by the large number of sequential denoising steps required at inference time. Existing fast samplers typically use fixed or generic timestep schedules, overlooking a distinctive property of time-series diffusion: different spectral bands evolve at different rates during the reverse process. We introduce StrideDiffusion, a training-free spectral-aware sampler that adaptively selects the denoising stride from band-level activity. At each step, StrideDiffusion monitors relative band energy, log-power drift, and phase velocity to identify whether high- frequency dynamics remain active or whether the trajectory is dominated by stable low-frequency structure. It then takes fine steps when rapidly varying bands are active and larger jumps once only coarse components remain. A bandwise stability analysis shows that inactive frequency bands change only linearly with the jump size under deterministic affine reverse updates, providing a local justification for spectral activity as a step-size indicator. Across six unconditional time-series generation benchmarks, StrideDiffusion uses only 14-66 function evaluations instead of 500/1000 denoising steps, achieving up to 18.9x wall-clock speedup while preserving or improving generation quality. On conditional imputation and forecasting, it further delivers 5-14x average acceleration with comparable predictive accuracy. These results show that spectral evolution provides a practical and principled signal for fast time-series diffusion sampling. Our code is available at this https URL. 

---
# AppWorld-UL: Benchmarking Diverse Agent-User Interactions for Tool-Use 

**Authors**: Junzhi Chen, Harsh Trivedi, Jane Pan, Michael JQ Zhang, Tejas Srinivasan, Niranjan Balasubramanian, Ashish Sabharwal  

**Link**: [PDF](https://arxiv.org/pdf/2607.20536)  

**Abstract**: Tool-use agents that address day-to-day digital tasks such as ordering groceries must not only operate applications, but also interact with the user, e.g., to ask clarification questions, prompt for confirmation, and inform the user when the instruction is infeasible. However, current benchmarks for evaluating agent-user interactions do not capture the diversity of such interactions. Further, they operate in small environments with few, often non-state-changing, APIs. To address this gap, we introduce AppWorld-UL, a ``user-in-the-loop'' benchmark of 516 challenging tasks requiring diverse agent-user interactions. Building upon the AppWorld framework with 9 popular simulated apps like Amazon and Spotify, we systematically modify original tasks to introduce ambiguities and constraints that necessitate various types of agent-user interaction. User behavior is simulated by an LLM prompted to respond with carefully designed knowledge boundaries, offering more reliable simulation than the unconstrained or overly rigid alternatives used in prior work. Our evaluation reveals that a state-of-the-art LLM, Claude Opus 4.7, achieves only 48.6% success on AppWorld-UL, and only 35.7% on the harder, compositional subset. On the stricter, scenario-level metric, compositional task performance drops to only 21.3%. Our analysis reveals that correct user-interaction is crucial for success. This demonstrates the benchmark's difficulty and its potential to advance research on user-in-the-loop tool-use agents. 

---
# DynamicMCPBench: A Trace-Grounded, Effect-Scored Benchmark for LLM Agents over Live MCP Servers 

**Authors**: Jerzy Kamiński, Ilya Galyukshev, Artem Kuznetsov, Sergey Chuprin, Kirill Redko, Aidar Shumbalov, Anna Kalyuzhnaya  

**Link**: [PDF](https://arxiv.org/pdf/2607.20531)  

**Abstract**: Large language model (LLM) agents are increasingly deployed over Model Context Protocol (MCP) servers, yet the benchmarks used to evaluate them score the final answer or a fixed "ground-truth" list of tools, both of which are fragile once the underlying data is live and stateful. We present DynamicMCPBench, a reusable framework rather than a fixed dataset. A practitioner can run it on their own MCP servers to test models on their own tasks, or let it collect servers automatically to measure a model's general ability to solve agentic tasks. Given the servers and any set of models, it generates realistic goals, pursues each one live to record a successful trajectory, distills that trajectory into path-agnostic effect checkpoints, and scores an agent on whether it reproduces those effects, never on the final answer. To show what the framework reveals, we run it at scale: 24 models over 121 servers and 750 tasks spread evenly over 15 task categories (50 each), where each category targets a distinct tool-use challenge of the generated questions. Each task is scored by pass^3: it counts as solved only if all three independent attempts succeed. Even the strongest agents solve only about half of the tasks, 31% of tasks are solved by no model at all, and accuracy collapses as the required tool chain grows longer (from 39% on the shortest chains to 13% on the longest). A human validation study confirms the automatic scoring is reliable (chance-corrected agreement of 0.76). DynamicMCPBench thus turns benchmark construction into something practitioners can rerun on their own servers and models, while exposing a consistent inability of current agents to handle long, multi-step agentic tasks. 

---
# PromptPack: Scaling LLM Annotation Agents for Online Recommendation 

**Authors**: Sebastian Koralewski, Merwan Barlier, Yulia Stolin, Blaž Škrlj  

**Link**: [PDF](https://arxiv.org/pdf/2607.20528)  

**Abstract**: Online recommendation platforms increasingly use Large Language Models (LLMs) to extract structured features from ad creatives. While deploying a single-call LLM annotation agent yields significant Click-Through Rate (CTR) improvements in our live production environment, per-creative prompting is prohibitively expensive to scale. The redundant system instructions sent in every request account for 94% of billed input tokens. To break this cost bottleneck, we introduce PromptPack, a scalable, high-throughput LLM annotation agent. PromptPack achieves this scale via in-context batching, combining a shared system prompt, a strict XML structural envelope, and an output correction layer to ensure deterministic, pipeline-ready feature extraction across multiple creatives simultaneously. We evaluate PromptPack via an offline retrieval benchmark using a downstream logistic-regression ranker. To deeply profile the agent's behavior, we measure AUC and introduce Volume-Weighted Absolute Lift (VWAL), a novel metric capturing the signal quality of the generated features. Compared to our live, unbatched production baseline, PromptPack at batch size 20 cuts our LLM costs by 89% and accelerates throughput by 2.5x while fully preserving AUC. 

---
# Evaluating and Guarding Citation Faithfulness in Agentic Scientific Synthesis 

**Authors**: Taewan Goo, Junsik Kim, Kyulhee Han, GwonYul Jo, Jong-Soo Kim, Tae-Hyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.20527)  

**Abstract**: Agentic LLM systems such as OpenScholar and PaperQA2 read the scientific literature and return cited answers, and both they and their benchmarks already check whether those citations hold, with a fixed attribution model or human graders. Neither audits the reliability of that check itself. We show it is not reliable, and that this matters. On identical agent outputs the measured unsupported-citation rate ranges from about 3% to about 18% depending only on the verifier's strictness, and although verifiers agree on which citations are supported, they disagree on which to flag (negative-specific agreement 0.27 to 0.30), so no single flag set is trustworthy and cross-paper comparison is invalid without a named verifier and protocol.
We present a gold-anchored evaluation protocol and a deployable guard that make this behavior measurable and bounded. The protocol validates the verifier, measures re-attribution, and calibrates a guarantee against human gold rather than another model's verdict; the verifier is a swappable instrument chosen on cost (recall 0.94 on the supported class, held out), and re-attribution is a commodity step where a deterministic BM25 matches the best open generator. The guard adds a split-conformal layer placing a distribution-free, finite-sample bound on truly unsupported citations that slip past a chosen flagging rule, a guarantee on catch rate rather than conclusion correctness. The bound holds on held-out gold, and we identify and quantify the condition governing its transfer to deployment, calibration-negative difficulty, with a concrete recalibration recipe, left untested by prior conformal-factuality work.
Validated across four open 27-35B models and three agentic pipelines on public benchmarks (SciFact, QASA, PubMedQA), with confidence intervals on every headline number, the protocol and guard ship as an open single-GPU kit. 

---
# ConfidenceBench: Evaluating Confidence Calibration in Large Language Models 

**Authors**: Matthew ffrench-Constant, Daniel Yang, Xinmeng Huang, Sanyam Kapoor  

**Link**: [PDF](https://arxiv.org/pdf/2607.20526)  

**Abstract**: Large language models (LLMs) are increasingly deployed in settings where fluent but incorrect answers can be costly. In these settings, accuracy alone is insufficient: models must also know when they are likely to be wrong. We present ConfidenceBench, a calibration benchmark that evaluates verbalized confidence estimates in 15 frontier LLMs using the Brier score, a proper scoring rule that incentivises truthful probability reporting. Confidence is elicited via prompting, requiring no access to model logits and making the framework applicable to both closed-source and open-source systems. The benchmark comprises 200 private multiple-choice questions across four categories: spatial reasoning, high-precision mathematics, word lookup, and unknowable questions. Across three independent runs, Claude Opus 4.6 and Gemini 3.1 Pro Preview achieve the lowest Brier scores, both reported as 0.103. Both substantially outperform the calibrated-random baseline of 0.1875, while Gemini 3.1 Flash-Lite scores 0.367, indicating severe miscalibration. Accuracy and calibration diverge substantially across model families: the most accurate model is not the best-calibrated, and several models perform worse than the calibrated-random Brier baseline despite reasonable accuracy. These results show that verbalized confidence calibration is a distinct and practically important axis of LLM reliability, complementary to standard accuracy-based evaluation. 

---
# Autonomous disproofs of the sum-product conjecture over $\mathbb R$ with GPT-5.5 Pro 

**Authors**: Yichen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20525)  

**Abstract**: OpenAI's recent disproof of the Erdős unit distance conjecture marked a milestone for AI in mathematics. It also inspired another breakthrough: a human disproof of the Erdős--Szemerédi sum-product conjecture over $\mathbb R$. In this paper, we present a simple agent built on GPT-5.5 Pro. Using a problem-agnostic, three-stage prompting pipeline -- proof-plan proposal, proof construction, and review -- the agent autonomously generated correct proofs that the sum-product conjecture is false over $\mathbb R$ in 7 of 8 independent trials; in the remaining trial, it identified an unresolved gap in its argument. The seven proofs are diverse: some are close to existing unit-based constructions, while others avoid units by using $L^p$-type regions of algebraic integers. The system used an average of 132.4k reasoning tokens per trial. We release the code, intermediate outputs, and generated proofs, providing a reproducible, data-contamination-free case study in autonomous proof generation. 

---
# Attention Degradation, Function Token Anchoring, and the Limits of Attention-Based Intervention in Large Language Models 

**Authors**: Sagar Dangal, Manoj Shakya  

**Link**: [PDF](https://arxiv.org/pdf/2607.20524)  

**Abstract**: Mean cross-positional attention degradation is widely reported in transformer interpretability, yet whether it causally limits contextual retrieval remains untested. We present six coordinated experiments across GPT-2, LLaMA-3.2-1B/3B, OPT-1.3B, and distilgpt2. We first characterise short-term (5-100 token) attention degradation, finding a universal exponential-then-plateau pattern whose rate is inversely correlated with depth, with distinct layer-wise entropy signatures per architecture. Function token anchoring proves architecture-dependent: OPT-1.3B (absolute positional encoding) shows distance-dependent preposition specificity, GPT-2 shows uniform non-specific dependence, and LLaMA (RoPE) shows reversal at long distances. Strategic comma insertion at clause boundaries causally reduces prediction degradation in the 40-80 token range, with the benefit tied to syntactic boundary alignment rather than token density. We then test the mechanism causally: Relay-Aware Attention (RAA), which biases attention logits toward function token positions, verifiably increases attention mass by 16-24% yet yields null effects on GPT-2 and LLaMA-1B, preliminary harm on LLaMA-3B, and a mixed effect on OPT-1.3B that nets to approximately zero. Multi-fact retrieval probes further show that degradation rate does not predict retrieval accuracy across models. We conclude that mean attention degradation is largely descriptive rather than prescriptive: function tokens contribute through what their hidden states compute, not through the attention they receive -- with implications for interpretability methodology and attention-score-based inference optimisations such as KV-cache eviction. 

---
# Representation Robustness Under Executable Reasoning Constraints in Large Language Models for Mathematical Problem Solving 

**Authors**: Sagnik Nath, Edith Aurora Graf, Liang Zhang, Diego Zapata-Rivera  

**Link**: [PDF](https://arxiv.org/pdf/2607.20520)  

**Abstract**: Large language models (LLMs) are increasingly evaluated on mathematical problem solving, yet prior work often treats representationally equivalent formulations as interchangeable and conflates reasoning errors with interface failures. This paper investigates representation robustness in LLM-based mathematical problem solving by systematically varying surface representations of the same underlying problems, including story problems, word-equations, symbolic equations, and isomorphic paraphrases. Using a curated dataset of mathematically equivalent problems, we evaluate five contemporary LLMs under a direct answer generation condition. We find substantial representational sensitivity: models frequently change correctness across equivalent formulations, with nontrivial flip rates across story, symbolic, and word-equation variants. We also observe systematic regressions under isomorphic reformulations, showing that even subtle paraphrase-level changes can degrade performance despite preserved mathematical structure. We then evaluate a code-augmented condition in which models externalize reasoning as executable Python code that is run locally for validation. This interface reveals strong latent reasoning capability in some models that perform poorly under direct prompting, but it does not uniformly improve robustness. Instead, failures shift across interaction layers, from opaque reasoning errors to protocol violations and execution failures. Even when executable reasoning succeeds, representation sensitivity often persists. Overall, our results show that reasoning scaffolds do not eliminate representational brittleness, but expose new tradeoffs among correctness, reliability, latency, and cost. We argue that representation should be treated as a first-class interface design variable in LLM evaluation and deployment, especially for AI-assisted problem-solving systems. 

---
# CANN Bench: Benchmarking Agent Generated Kernels against Real NPU and Algorithmic Limits 

**Authors**: Xue-Jian Gao, Deng Pan, Yueming Su, Jiasheng Li, Bin Du, Fengming Zhu, Chengdi Ma, Junyi Fan, Qichen Liao, Chengqiu Hu, Xinxian Chen, Lingchao Zheng, Jun Li, Jiwei Yang, Yuwei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2607.20518)  

**Abstract**: AI agents are now capable of writing, compiling, and iteratively optimizing low-level operator kernels on different hardware platforms. Existing benchmarks, however, focus almost exclusively on CUDA and Triton, leaving hardware ecosystems with less-exposed programming models without a common evaluation baseline. We present CANN Bench, an open benchmark for AI-generated operator code on Huawei's Ascend NPU. The current release covers 53 operators and 1060 test cases organized into four difficulty tiers -- from simple elementwise primitives to MoE dispatch and FlashAttention kernels -- spanning FP16, BF16, FP32, and INT8 precision formats. Evaluation adopts a \textbf{three-dimensional weighted composite score} that treats compilation, functional correctness, and performance as independent axes, providing a principled reward signal for kernel-generation agents. Performance is graded against an out-of-the-box PyTorch-on-Ascend baseline and an analytical per-case Hardware-Anchored Performance (HAP) limit on real NPU hardware, ensuring scores reflect genuine optimization headroom rather than measurement artifacts. The evaluation harness is designed to resist reward hacking from the ground up. CANN Bench is versioned within the official CANN repository and is designed for long-term community co-construction, providing the Ascend ecosystem with a quantitative, reproducible, and sustainably maintained yardstick for AI operator-authoring capability. 

---
# Reliability-Aware LLM Alignment from Inconsistent Human Feedback 

**Authors**: Jingyi Huang, Ruohan Zong, Yujun Feng, Liran Ma, Lanyu Shang, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20515)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is critical for aligning Large Language Models (LLMs) with human preferences. However, its efficacy is often compromised by the inherent inconsistency and subjectivity of human annotations. Existing preference optimization frameworks, such as Direct Preference Optimization (DPO), typically treat ambiguous pairs with high annotator disagreement identically to those with unanimous consensus, forcing models to overfit to inconsistent supervision signals and leading to suboptimal alignment. In this work, we propose Reliability-Guided Preference Optimization (RGPO), a robust framework designed to mitigate the impact of inconsistent human feedback. RGPO estimates annotator reliability and infers latent ground truth labels from noisy human feedback to identify robust preferences. Furthermore, we introduce a reliability-aware consistency optimization that dynamically modulates the training objective based on the consensus level of annotations, ensuring the model prioritizes high-consensus supervision signals. Extensive experiments on LLM alignment benchmarks demonstrate that RGPO effectively reduces inconsistency and noise in training data and achieves superior performance compared to widely adopted RLHF baselines. Our code and configurations are available at this https URL. 

---
# SiGMA: Sign-Guided Merging and Adaptation for Multimodal Continual Instruction Tuning 

**Authors**: Keonhee Park, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.20511)  

**Abstract**: Multimodal Continual Instruction Tuning (MCIT) is crucial for adapting Multimodal Large Language Models (MLLMs) to evolving a sequence of downstream tasks. Prior methods mostly utilize Mixture of Experts or expansion merge approach, primarily focusing on catastrophic forgetting, yet they still suffer from negative interference during inference, where newly learned updates overwrite useful prior knowledge and degrade overall performance. To address this, we propose SiGMA (Sign Guided Merging and Adaptation), a simple yet effective framework that mitigates negative interference with two components: sign guided adaptive tuning during training and sign guided merging at inference. Sign guided adaptive tuning reduces collisions with past knowledge and learns the current task with minimal drift, mitigating severe forgetting. Sign guided merging further improves consolidation by selectively scaling salient parameters to preserve and amplify useful task specific knowledge. Experiments on UCIT and DCL benchmarks show that SiGMA significantly reduces negative interference and outperforms state of the art MCIT methods. Our code is available at SiGMA. 

---
# Telco-GAIA: Bilingual Benchmark for Agents in Telecom Domain 

**Authors**: Dmitrii Khizbullin, Zaid Alyafeai, Abdelrahman Eldesokey, Nourah AlSultan, Raghad Alshalan, David R. Pugh, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2607.20510)  

**Abstract**: We introduce Telco-GAIA, a bilingual, multi-modal benchmark for evaluating tool-using agents on the data of a real-world telecommunications operator. Telco-GAIA comprises 100 human-verified question-answering tasks, in English and Arabic, that each demand multi-hop reasoning (4.2 hops on average) over three heterogeneous sources: a static website snapshot (HTML, images, and linked PDFs), a synthetic relational SQL database, and external web archives, spanning text, image, and tabular modalities. The benchmark is delivered as a sandboxed Docker environment and scored by normalized exact string matching, making evaluation objective, deterministic, and reproducible over time without any LLM-as-a-Judge. Evaluating a purpose-built reference agent across twelve commercial and open LLMs, we find Telco-GAIA challenging: even the strongest model solves only 71% of tasks; under a moderate cost budget, this falls to about 40%, and the visually grounded categories remain the weakest, where the average backend scores below 30%, leaving substantial headroom in document and image understanding. Telco-GAIA offers a rigorous, reproducible testbed for enterprise agents and a template for constructing closed-domain benchmarks. 

---
# MiniCache: Reusable Program Caching with Small Model Interfaces for Efficient LLM Inference 

**Authors**: Jingquan Chen, Jinghua Piao, Jie Feng, Shaogang Hu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.20507)  

**Abstract**: Large language models (LLMs) are increasingly used for program-aided reasoning, agentic decision making, and structured task execution, but these applications often incur high inference cost. We present MiniCache, a reusable program caching framework that transforms Program-of-Thought (PoT) programs into parameterized cache objects, enabling reusable computation across structurally similar requests. MiniCache reuses the same small model for semantic variable extraction on cache-hit requests and speculative drafting during target-LLM generation, reducing expensive target-LLM invocations while preserving task quality. Experiments on shopping-style request datasets, WebShop, Formula, and CodeTAT-QA demonstrate that MiniCache improves the trade-off between inference latency, cache reuse, and accuracy, achieving up to 3.1x lower latency and 2.8x higher throughput under parallel serving. These results show that small models are most effective not as replacements for large models, but as lightweight interface models that enable reliable and efficient reusable program caching. 

---
# Optimizing Hypergraph-Based RAG: Toward Better Fact Extraction and Chunk Retrieval 

**Authors**: Houda Khrouf, Pedro Fillastre, Sebastiao Correia  

**Link**: [PDF](https://arxiv.org/pdf/2607.20506)  

**Abstract**: GraphRAG enables deeper reasoning by structuring knowledge as graphs but struggles with n-ary facts. HyperGraphRAG uses hypergraphs for richer semantics, improving accuracy, yet relies on error-prone LLM extraction and inefficient standard chunk retrieval. We address this by employing self-consistency prompting to improve the extraction, and Personalized PageRank algorithm over hypergraph to enhance chunk retrieval. 

---
# LeanFlow: A Case Study in Workflow-Driven Lean Autoformalization 

**Authors**: Lazar Milikic, Simon Guilloud, Khanh Nguyen, Viktor Kuncak  

**Link**: [PDF](https://arxiv.org/pdf/2607.20503)  

**Abstract**: We present and evaluate LeanFlow, an LLM agent system specialized for translating mathematical papers into buildable Lean projects. Recent verifier-in-the-loop systems show that large formal artifacts can be produced, but it remains unclear which runtime mechanisms affect completion, auditability, or efficiency in document-to-project formalization. We study this question through case studies on two previously unformalized mathematical papers in number theory and measure theory, using model, proof-workflow, and toolset ablations with Kimi2.6 and GPT5.5; we report task outcome, API calls, input tokens, and output tokens. With Kimi2.6, the full workflow completes both document-level projects within the 2000-call budget, while no-queue variants reach the budget limit; with GPT5.5, all document-level variants complete, and the full workflow has the lowest or tied-lowest input-token cost on both sources. As complementary calibration, LeanFlow reaches 75.7% BEq+ on the PFR slice of RLM25 and solves all five ICML 2026 AI for Math TCS challenge projects in our GPT5.5 runs. 

---
# Inducing Comparability of Factorised Probability Distributions 

**Authors**: Jan Speller, Malte Luttermann, Marcel Gehrke, Tanya Braun  

**Link**: [PDF](https://arxiv.org/pdf/2607.20502)  

**Abstract**: To allow for principled comparison between two probabilistic graphical models defined over non-identical variable sets, they have to be lifted to a common measurable space. To this end, we propose an extension scheme for any two given models and establish the formal foundation: Unmatched components are completed using conditionally uniform (Laplace) extensions such that the resulting joint distributions differ from the original ones only by multiplicative constants and coincide under projection. This preserves the probabilistic semantics while enabling the application of well-defined distributional discrepancy measures. We establish the invariance of the induced joint under projection and use the extensions to provide a minimal structural extension of two factor graphs to the smalles common measurable space as well as to a common graphical structure by a deterministic algorithm. In addition, we discuss structural and measure-theoretic properties and identify promising criteria for comparison methodologies. 

---
# MKEvolve: A Modular Multi-Agent Framework for Kernel Code Generation 

**Authors**: Jason Yoo, Rajarshi Saha, Shaowei Zhu, Tao Yu, Wei Tang, Youngsuk Park  

**Link**: [PDF](https://arxiv.org/pdf/2607.20501)  

**Abstract**: Despite rapid progress in LLM-based code generation, writing correct and performant kernels for hardware accelerators remains a key bottleneck in scaling modern ML workloads. We present MKEvolve (Modular Kernel Evolve), a framework that iteratively co-evolves a modular decomposition of complex PyTorch modules and the LLM-generated kernel for each submodule, refining the decomposition by splitting and fusing across iterations while independently improving each subkernel via LLM-driven beam search. The resulting kernels are programmatic compositions of independently verified subkernels, making them configurable (subkernel implementations are swappable), interpretable (errors and speedups are traceable to specific subkernels), and readily adaptable to related model architectures. Experiments with Triton on KernelBench L2 and L3, spanning multi-operator sequences and full model architectures, show that MKEvolve improves both correctness and speedup over end-to-end direct synthesis baselines while reducing LLM token usage by up to 35%. 

---
# FlowEdit: Information-Theoretic Control of LLM Reasoning Flows for Ill-posed Problems Involving Conflicts 

**Authors**: Sizhe Tang, Guangyu Jiang, Yu Li, Rongqian Chen, Ioannis G. Kevrekidis, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2607.20500)  

**Abstract**: Large Language Models (LLMs) perform strongly on well-specified reasoning tasks with a feasible answer. However, problems encountered in the open world can become ill-posed due to inconsistent conditions, conflicting statements, or mutually incompatible requirements, admitting no valid responses. We argue that reasoning of such ill-posed problems involving conflicts require novel LLM capabilities to make hidden conflicts explicit, maintain competing hypotheses via multiple reasoning branches, and generate alternative responses in a single pass, all of which are challenging due to the limitation of the next-token prediction mechanism in LLMs. To this end, we propose FlowEdit, a novel framework that leverages information-theoretic principles to quantify and regulate internal reasoning flows of LLMs, for generating a full set of alternative responses under valid hypotheses. FlowEdit can be viewed as enforcing a branch-aware reasoning process using two dual information-theoretic objectives on the model's internal reasoning representations: maximizing the information flow from each selected hypothesis to the branch outcome, while minimizing the overlap and conditional dependence across sibling branches, to provide a diverse, informative set of responses with broad coverage. We show that this is achieved through tractable variational bounds under boundary embeddings being {\epsilon}-sufficient, optimizing the underlying conditional mutual information in LLM reasoning process. Extensive experiments demonstrate that FlowEdit outperforms leading proprietary models, improving exact-set-match accuracy by 68%, while boosting overall response informativeness by 24%. We further show that flow regulation surfaces in the token stream as a redistribution of next-token entropy that concentrates inside each branch, amplifies at flow boundaries, and scales with the number of flows the problem requires. 

---
# ExecuGraph: A Multi-Agent, Execution-Grounded Framework for Reliable Backend Code Synthesis with Large Language Models 

**Authors**: Sai Deekshith Lekkala, Jothi Prabha Appadurai, Rohith Reddy Bellibatlu, Manpreet Singh  

**Link**: [PDF](https://arxiv.org/pdf/2607.20499)  

**Abstract**: Large Language Models generate plausible backend code, but a single-pass paradigm provides no guarantee of correctness or runtime reliability. We present ExecuGraph, a multi-agent framework that places execution-based validation at the center of backend code synthesis. Six specialized agents (Planner, Code Generator, Logical Reviewer, Evaluator, Optimizer, and Explainer) are coordinated by a typed directed workflow with a bounded retry budget, implemented on LangGraph with locally hosted models (Ollama) and an optional retrieval layer for algorithmic technique recall. A subprocess-isolated sandbox with a wall-clock timeout guards every evaluation. We evaluate on a curated 30-problem DSA suite (internal-30), HumanEval (n=64), and an APPS-introductory subset, contrasting ExecuGraph against a single-agent one-shot baseline and a single-agent execution-retry baseline (a Reflexion-style ablation that isolates the contribution of multi-agent decomposition). On internal-30, the three conditions are statistically indistinguishable (n=30; paired Wilcoxon p=0.59 MF vs. SO, p=0.08 SR vs. SO); 95% bootstrap confidence intervals on all pairwise mean differences include zero. On HumanEval, multi-full edges ahead by +3.1 pp. The strongest signal is cross-model: with DeepSeekCoder V2 Lite, graph-category accuracy improves from 57.5% (oneshot) to 80.0% (multi-full), a +22.5 pp jump that supports a scaling hypothesis: the value of multi-agent decomposition grows with base-model capability. The framework's primary contribution is methodological: a single codebase that collapses by configuration into one-shot, execution-retry, and per-agent ablation conditions, enabling controlled measurement of each lever's marginal contribution. A per-agent ablation, retry-budget sweep, error-class taxonomy, and test-source audit are reported. 

---
# AISE-Bench: A Full-Cycle Curated Benchmark for Information Seeking on Academic Knowledge Graphs 

**Authors**: Fanjin Zhang, Zhengyang Wang, Ruixuan Huang, Kefan Zhang, Amy Xin, Yuanchun Wang, Shu Zhao, Evgeny Kharlamov, Jie Tang, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.20498)  

**Abstract**: Large language models (LLMs) augmented with tools are emerging as autonomous agents capable of using Web engine, APIs, and code to solve complex, long-horizon tasks. Current tool-using benchmarks for information seeking on academic graphs rely on synthetic templates, simplified solution spaces, or narrow tasks such as paper-centric tasks, leaving key challenges underexplored - realistic user intent, complex multi-step API planning, rich parameter filling for APIs, grounded answers with references, and comprehensive evaluation of both the process and the outcome. We introduce AISE-Bench, a real-world, full-cycle annotated benchmark for information seeking on academic knowledge graphs. AISE-Bench release contains 1,133 QA pairs, including query taxonomies, full API execution trajectories, validated parameters, and source-grounded answers with reference links. To support high-quality annotation, we design a customized agent workflow to enable annotators to plan, execute, and revise complex API workflows efficiently. We develop a comprehensive evaluation protocol measuring answer quality, reference grounding, API-planning correctness, and execution success. Among the 14 evaluated methods, even the strongest model (PLAY2PROMPT with Gemini-3-Pro) achieves only moderate performance and often struggles with API planning and execution. AISE-Bench establishes a challenging new testbed for quantitatively evaluating and improving the stepwise correctness, grounded summarization, and traceable reasoning of multi-step API-using LLM agents. Our code and data are available at this https URL. 

---
# From Errors to Rules: Iterative Prompt Optimization for Text Classification 

**Authors**: Yueying Cui, Renhao Xue, Yi Zhang, Mukul Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2607.20497)  

**Abstract**: Prompt optimization for text classification spans diverse approaches, from demonstration selection to exploration-based search to error-driven diagnosis, each with known but incompletely characterized strengths and limitations. We conduct a comprehensive empirical study across diverse classification benchmarks (2 to 150 classes) comparing these paradigms through both quantitative evaluation and qualitative analysis of optimization traces, revealing that each paradigm excels on structurally different task types and that no single method dominates. Guided by these insights, we propose Error-Guided Optimization (ERGO), an error-driven method that iterates over the full training set in non-overlapping batches, diagnoses classification failures, and generates targeted decision rules through a diagnose-prescribe-rewrite feedback loop. ERGO achieves the best accuracy on tasks where errors concentrate in specific confused label pairs (which we term boundary-learnable tasks): TREC: 90.0%, CLINC150: 94.4%, converges in 3-5 iterations, and produces interpretable decision rules. While ERGO does not achieve the highest overall average, it fills a complementary role: demonstration-based ICL wins on coverage-dependent tasks, exploration-based search wins on many-class intent, and ERGO wins where decision boundaries are learnable from error patterns. We provide a complementarity framework linking task characteristics to optimal paradigm selection, offering practical guidance for practitioners. 

---
# Workload-Aware Caching for Multi-Agent Systems 

**Authors**: Anas Mohamed, Kaizan Haque, Azal Ahmad Khan, Chetan Sharma, Shuwen Ge, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2607.20495)  

**Abstract**: Multi-agent systems decompose complex tasks into directed acyclic graphs (DAGs) of specialized agent executions, creating natural opportunities for caching intermediate results across queries. However, existing cache eviction policies treat all cached entries uniformly based on access history, ignoring structural and workload signals uniquely available in agentic execution environments. We present a workload-aware eviction policy that combines three signals, namely recomputation cost, DAG dependency count, and agent invocation frequency, into a unified scoring function that retains the most valuable entries under memory constraints. Evaluated across three multi-agent benchmarks spanning diverse reuse regimes, our policy reduces latency by up to 64.7% relative to the uncached baseline and achieves on average a 31.1% latency reduction over the next best finite-capacity baseline, while approaching the performance of an unbounded cache and maintaining accuracy on par with or exceeding all competing finite-capacity methods. We further show that workload-aware content caching is complementary to other agentic system optimization methods, including plan-level caching and parallel agent execution, with each technique targeting a distinct efficiency bottleneck in multi-agent pipelines. 

---
# Isolating LLM Alignment from Regex: Zero Coverage and Metric-Dependent Divergence Under Adversarial Mutation 

**Authors**: Alexandre Cristovão Maiorano  

**Link**: [PDF](https://arxiv.org/pdf/2607.20494)  

**Abstract**: Production LLM applications commonly stack a regex filter in front of model-side alignment; prior work found no measurable coverage gain from adding a live Gemini backend behind an active regex filter. We ask whether that ceiling holds when the corpus is \emph{designed to bypass the regex}. We introduce $L_5$-no-regex -- identical to $L_4$-real (Gemini-2.5-flash, token-budget cap, rate limit, output scrub) but with the nine-pattern filter disabled -- and evaluate it against $N{=}45$ adversarial probes across three sub-corpora (carry-forward, regex-bypass, alignment-isolate), amplified by Gemini paraphrase and PAIR to ${\sim}1{,}555$ probe-run pairs over $N{=}5$ replications. Under the primary substring classifier, H1 is refuted: $L_5$ block rate is $0,%$ across all five OWASP LLM Top-10 categories ($\Delta\text{pp}{=}0$ vs.\ $L_0$, $p{=}1.00$; Wilson upper bound ${<}5,%$). A secondary LLM-judge metric on PAIR variants shows $56$--$100,%$ block rates ($p{<}0.01$), revealing alignment does respond to adversarially-framed probes -- but produces refusals too nuanced for substring matching. The sub-corpus differential prediction is not supported ($p{=}1.00$). Alignment's contribution is \emph{metric-dependent}: on natural-language harmful-request probes, it adds zero observed coverage beyond the regex; on adversarially-framed variants, an LLM judge detects refusals the substring classifier misses. The locked corpus, mutation artifacts, and export scripts are released for replication. 

---
# Attention-based Experience Replay Framework for Continual Learning of Agnostic Time Series Forecasting Models 

**Authors**: Quentin Besnard, Nicolas Ragot  

**Link**: [PDF](https://arxiv.org/pdf/2607.20493)  

**Abstract**: Deep learning has led to remarkable progress in artificial intelligence, particularly in robotics, imaging and sound processing. However, a major limitation of neural networks remains their strong dependence on large and stationary datasets. In many real-world applications, these conditions are rarely met due to evolving and dynamic environments where data distributions change over time. Continual learning aims to address this challenge by developing models capable of adapting incrementally while maintaining a balance between stability and plasticity under computational constraints. In this work, we introduce a novel framework for continual time series forecasting, designed to extend existing static forecasting models commonly used in the literature by incorporating an Experience Replay strategy guided by Attention mechanisms. This approach allows the model to adapt dynamically to new contexts while preserving prior knowledge, effectively mitigating catastrophic forgetting. The framework is evaluated on standard forecasting benchmarks as well as on a piezometric dataset exhibiting diverse temporal behaviors. Results show that our approach effectively increases or maintains predictive performance over time while reducing retraining costs and data requirements, thus facilitating the deployment of forecasting models in dynamic and real-world settings. 

---
# DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making 

**Authors**: Raffi Khatchadourian  

**Link**: [PDF](https://arxiv.org/pdf/2607.20491)  

**Abstract**: Standard evaluation benchmarks measure what a tool-using agent decides, not whether it arrives at that decision through the same process each time. We introduce DFAH-Bench, a replay benchmark that measures observable behavioral instability in financial agent decision-making across three channels -- tool-call trajectories, evidence contacts, and decision concentration -- none of which require access to hidden reasoning text. Across 8,127 replay episodes spanning 10 models and 3 financial tasks, we find that outcome agreement alone is an incomplete stability signal: frontier models can agree on decisions 95% of the time while following the same tool path only 77% of the time -- an 18-percentage-point gap (95% CI: [0.14, 0.22]) that outcome-only evaluation misses entirely. Among frontier-model case groups with high decision agreement, over 55% exhibit meaningful trajectory divergence. We identify three behavioral profiles: pattern matchers that achieve near-perfect agreement by collapsing to a single output regardless of input, stable executors with relatively consistent tool-use processes, and trajectory divergers that reach the same conclusions through materially different tool paths and evidence contacts. The benchmark code, metric scripts, replay logs, benchmark card, dataset README, and release manifest are released in the accompanying repository. 

---
# CRAWO: Custom Resources for Adaptive Workload Orchestration 

**Authors**: Eugênio Santos, Daniel Maia, Stefano Loss, José Manoel Silva, Aluizio Rocha Neto, Thais Batista, Everton Cavalcante, Nélio Cacho, Eduardo Nogueira, Daniel Araújo, Frederico Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2607.20490)  

**Abstract**: Edge Intelligence has emerged as a key paradigm for enabling real-time applications in smart cities by shifting computation from centralized cloud data centers to the network edge, thereby reducing latency and bandwidth consumption. However, deploying Artificial Intelligence (AI) pipelines across heterogeneous edge infrastructures remains challenging due to the wide range of device capabilities, from low-power microcontrollers to accelerator-equipped systems. Existing edge orchestration platforms primarily focus on deployment automation and infrastructure management, but these approaches are often inefficient and limit the ability to adaptively allocate resources under dynamic conditions. To tackle these issues, this paper introduces CRAWO (Custom Resources for Adaptive Workload Orchestration), an architectural framework for coordinating AI pipelines across distributed edge environments. CRAWO follows a control-loop-based model that separates allocation intelligence from execution by managing placement decisions, state management, and inter-stage data flows while instantiating services on edge nodes. The framework incorporates a hardware-aware allocator with a pluggable multi-criteria decision layer that leverages real-time infrastructure metrics to enable adaptive workload placement. The reference implementation adopts a microservices architecture deployed on a lightweight Kubernetes distribution (K3s), using Custom Resource Definitions (CRDs) for domain modeling and a dedicated operator for state reconciliation. Evaluation in a vehicle surveillance scenario using license plate recognition demonstrates improved workload distribution and reduced reliance on centralized cloud processing in latency-sensitive environments. 

---
# EvoSQL: Memory-Augmented Critic-Generator Co-Evolution for Text-to-SQL 

**Authors**: Jiawei Zhou, Jianwei Wang, Chenyu Zhou, Chaojian Shi, Ming Dong, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20489)  

**Abstract**: Text-to-SQL has advanced rapidly with large language models, but complex database queries still require reasoning beyond one-shot generation, including multi-step decomposition, execution-based diagnosis, and targeted correction. We present EvoSQL, a co-evolution framework that formulates SQL synthesis as an iterative interaction between a generator and a critic. EvoSQL maintains a contextualized candidate memory, verifies SQL candidates with both execution signals and LLM-based critique, and updates its memory through utility-guided aggregation. To strengthen the underlying generator-critic pair, we further introduce a Self-Distillation Policy Optimization (SDPO) fine-tuning stage that injects execution-aware supervision into modern coding LLM backbones. Experiments on Spider and BIRD show that EvoSQL consistently improves open-source models over Maj@16 baselines, with particularly large gains on BIRD-Dev, ranging from +1.37% for Qwen3-4B to +9.19% for Qwen2.5-Coder-3B. SDPO initialization further improves selected backbones on Spider-Test and BIRD-Dev. These results suggest that memory-grounded co-evolution is an effective path toward more reliable and generalizable Text-to-SQL systems. Code is available at this https URL. 

---
# Autonomous Topology Mutation: Safe Runtime Restructuring for Multi-Agent LLM Systems with Capability, State, and Shadow Invariants 

**Authors**: Bronislav Sidik, Chaya Levi, Nizzan Kimhi  

**Link**: [PDF](https://arxiv.org/pdf/2607.20488)  

**Abstract**: Multi-agent LLM frameworks typically fix their team topology at boot time. When an individual agent becomes overloaded at runtime, for example by mixing too many action categories, accumulating tool errors, or queueing behind too many calls, the system has no mechanism to restructure itself. We introduce Autonomous Topology Mutation (ATM), a runtime team-mutation mechanism for multi-agent LLM frameworks. ATM combines telemetry-driven overload detection with three safety invariants that gate each structural change: capability monotonicity, state-routing completeness, and shadow-before-live validation.
ATM monitors a six-signal Bottleneck Index that includes queue depth, context thrash, tool-error rate, role entropy, retry-loop rate, and cross-agent wait time. When a warmup-calibrated threshold is breached for multiple consecutive ticks, ATM factorises the overloaded agent into specialised sub-agents and hot-swaps the parent into a coordinator role while preserving its external identity. State transfer is controlled by privacy-level-aware routing: each memory atom is routed only to a permitted child set, or explicitly dropped with a logged reason. No candidate topology receives live traffic until it has passed a shadow validation window.
On 720 DeepSeek-V3-driven task runs with deterministic tool stubs across four ablation conditions and three workloads, the ATM factoriser split lifts code-task success from 3.3% to 61.7%. The full rail-and-distillation system reduces detected high-privacy memory exposure under a regex classifier from 2.0 to 0.0 events per task while preserving task quality. The runtime rails carrying ATM's invariants add less than 500 microseconds of p99 latency on the agent hot path. A small live-tool probe with real Python execution is included as an external-validity check. The implementation, benchmark harness, and traces are open-sourced. 

---
# Directional Hallucinations: Ideological Drift in News-Grounded LLM Question Answering 

**Authors**: Chendi Wang, Liam Cunningham, Tom Yishay, Jieying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.20487)  

**Abstract**: Large language models (LLMs) are increasingly used to answer questions about political information, including in election-adjacent information settings where factual errors and ideological distortions are high-stakes. We present a reproducible measurement framework that treats hallucinations, unsupported statements in document-grounded QA, as diagnostic signals of ideological drift. Using 21,727 expert-labeled U.S. political news articles from QBias spanning left, center, and right sources, we (i) generate an article-specific question, (ii) elicit document-grounded answers from three open-weight LLMs and one proprietary model, (iii) detect sentence-level hallucinations via reference-based comparison, (iv) classify the ideological valence of hallucinated sentences with a fine-tuned stance classifier, and (v) probe output logits to relate token-level uncertainty to hallucination and drift. Hallucination rates vary substantially across models and concentrate in contentious topics, while source-ideology differences in hallucination frequency are modest. In contrast, hallucination content exhibits robust leftward drift: a majority of hallucinated sentences are classified as left-leaning, including among hallucinations generated from right-leaning sources. Logit-level analysis shows hallucinations arise in high-entropy generation contexts, and in some models uncertainty also predicts leftward drift, consistent with an "uncertainty to guessing" mechanism. We discuss implications for auditing AI-mediated political information and for designing safeguards in election-relevant deployments. 

---
# OPTScientist: Multi-Agent Discovery of Typed Optimizer Programs for Transformer Pretraining 

**Authors**: Zhongzheng Li, Tiancan Feng, Wenhao Li, Qingsong Ran, Shikun Feng, Xiaoyuan Zhang, Yue Wang, Xiaoguang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20486)  

**Abstract**: Designing optimizers for modern deep learning remains a challenging scientific problem, requiring the joint consideration of optimization geometry, state dynamics, numerical stability, implementation constraints, and empirical generalization. Existing automated optimizer discovery methods typically search either over unconstrained code spaces or within narrowly parameterized optimizer families. The former is flexible but often produces invalid or uninterpretable programs, while the latter is stable but limits novelty. We introduce OPTScientist, a theory-guided multi-agent framework for optimizer discovery in a typed domain-specific language (DSL). OPTScientist formulates optimizer design as a constrained scientific search process, where candidate updates are expressed through direction, scaling, preconditioning, regularization, state, and grouping modules. Four role agents, Theorist, Designer, Engineer, and Reviewer, collaborate within a single orchestration loop to propose hypotheses, synthesize DSL candidates, compile and evaluate optimizers, and critique results. To overcome the limitations of a fixed search space, OPTScientist combines evolutionary search over optimizer programs with a second-stage mechanism that proposes small DSL extensions when repeated failures reveal representational bottlenecks. Using this framework, we discover RS-MR, a reduced-state matrix optimizer that improves transformer pretraining over strong baselines under our native evaluation protocol. Our results suggest a path toward automated optimizer science grounded in theory, typed programs, compiler validation, and closed-loop experimentation. 

---
# Expectation Alignment of Language Models for Real-World User Expectations 

**Authors**: Miaomiao Li, Yang Wang, Bin Liang, Shudong Liu, Zhiwei Zhang, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2607.20485)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance on standard benchmarks, yet it remains largely unexplored whether they truly meet user expectations. Existing evaluation approaches, relying on model heuristics, expert rubrics, or user simulation, fail to capture the diversity and subtlety of real human expectations, causing models to appear competent while misaligning with what users actually seek. We present the first systematic study of user expectations in real-world LLM interactions, proposing a principled procedure to extract semantically rich expectations and introducing ExpectBench, a benchmark grounded in real user expectations. Analyses reveal that current LLMs struggle to satisfy and anticipate what users hope to obtain, highlighting a fundamental source of misalignment. Building on these observations, we propose LENS, a lightweight latent expectation-aware response generation framework. LENS enables models to internalize user expectations and generate better-aligned responses, consistently improving expectation satisfaction and underscoring the importance of explicitly modeling user expectations for realistic human-AI alignment. 

---
# The Devil is in the Spectrum: Mitigating Representation Collapse in LLMs via Topologically Regularized Side-Path 

**Authors**: Yiheng Tao, Kaiwen Cheng, Yao Lu, Chang Liu, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.20484)  

**Abstract**: Large Language Models (LLMs) are fundamentally limited by representation collapse, a bottleneck that severely degrades long-context performance. We identify that existing approaches risk drifting into one of two pathological extremes: homogenization collapse (e.g., attention sinks causing rank deficiency) and isolation collapse (e.g., local attention causing context disconnection). Through spectral analysis of attention dynamics, we derive an intrinsic trade-off between mixing efficiency (spectral gap) and information capacity (effective rank) that standard mechanisms struggle to balance. To resolve this dilemma, we propose the Topologically Regularized Side-Path (TRSP), a non-invasive architectural intervention that achieves spectral balance. TRSP employs a parameter-free Triangular Box mechanism, scaled by a lightweight, length-aware gate, to regularize the token interaction topology. By integrating proximal coupling to preserve effective rank and distal propagation to support non-degenerate mixing, TRSP promotes a geometrically healthier transition operator without altering core attention. Experiments show significant improvements across general capabilities and long-context benchmarks. Notably, on NoLiMa at $8\times$ the training length, TRSP retains $83\%$ accuracy and surpasses the Differential Transformer and Gated Attention by approximately 30 and 50 percentage points, respectively. Code available at: this https URL. 

---
# Tractable Hierarchical Control of Autoregressive Language Models 

**Authors**: Max Scribner, Antonio Vergari, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2607.20483)  

**Abstract**: Constraining the generation of autoregressive large language models (LLMs) is an important component of integrating language models into formal systems. In the generation of code and data for tasks like program synthesis, ensuring that language models produce syntactically valid output is a prerequisite for processing such output. These languages (such as SQL or JSON) are often designed as $LR(k)$ context-free grammars. By distilling the LLM to a tractable probabilistic model, its autoregressive generation can be steered and masked to incorporate the probability of satisfying logical constraints, ensuring high quality output that is guaranteed to be valid. This paper demonstrates that the satisfaction of any $LR(k)$ grammar of finite duration can be calculated in polynomial time, an improvement over the exponential time of applying previous methods to such grammars. This result enables efficient constraint and steering of LLM generation towards output that better satisfies formal syntactic constraints. 

---
# PersonaTrail: Benchmarking Personalized Web Agents through Browsing Trails 

**Authors**: Seungbin Yang, Chaewoon Ki, Dohyun Lee, Jaegul Choo, ChaeHun Park  

**Link**: [PDF](https://arxiv.org/pdf/2607.20482)  

**Abstract**: Recent advances in large language models have enabled web agents to autonomously execute complex tasks. In practice, users frequently provide underspecified instructions, requiring agents to infer the missing context from their raw browsing histories. Existing benchmarks fail to capture this form of personalization, as they either restrict tasks to fully explicit prompts or abstract web interaction history into simplified forms. To bridge this gap, we introduce PersonaTrail, a benchmark for personalized web agents operating in a managed open web environment. By leveraging realistic browsing trajectories as user history, PersonaTrail evaluates an agent's ability to infer user preferences and recall information from past browsing sessions. We further propose Preference-Aware Contextual Memory (PACMem), a framework that decomposes raw browsing histories into two types of structured memory: factual memories that summarize individual sessions and preference memories that distill recurring behavioral patterns. At inference time, the agent retrieves the most relevant entries from these memories to guide personalized navigation. Extensive experiments show that PACMem consistently outperforms existing memory-based baselines on both tasks. 

---
# Routing Without Training: Controllable-Ratio LLM Offloading via Reliability Gating 

**Authors**: Evan Chen, Shiqiang Wang, Kevin S Chan, Su Wang, Christopher Brinton  

**Link**: [PDF](https://arxiv.org/pdf/2607.20481)  

**Abstract**: Local-cloud collaboration is a practical way to deploy large language models under resource constraints, but existing methods often rely on trained routers or collaboration-aware finetuning that tie routing behavior to a particular operating regime. In this work, we show that such training may be unnecessary: the local model's own inference-time agreement across sampled responses already provides a strong signal for deciding when to trust local execution and when to offload to a stronger cloud model. We propose CARGO, a training-free routing framework that estimates this agreement through prompt-varied sampling, applies Bayesian early stopping for sample-efficient uncertainty control, and supports arbitrary target collaboration ratios through lightweight deployment-time calibration. Across diverse reasoning and question-answering tasks, multiple local LLM families and scales, and both pretrained and finetuned local models, CARGO consistently outperforms other training-free baselines and in several settings surpasses supervised learned routers. These results suggest that effective and adaptable local-cloud collaboration can emerge directly from the local model's intrinsic response behavior, without requiring an additional trained router. 

---
# Enabling Scalable Topology Inference in Distribution Systems via Constrained Multi-Source Inference 

**Authors**: Haoran Li, Lihao Mai, Muhao Guo, Jiaqi Wu, Yang Weng  

**Link**: [PDF](https://arxiv.org/pdf/2607.20480)  

**Abstract**: Accurate distribution system topology is essential for outage localization, voltage analytics, and operation of distribution grids, yet maintaining reliable connectivity records remains challenging in practice due to heterogeneous and imperfect utility data. Existing topology identification methods often rely primarily on electrical similarity or spatial records alone, which become unreliable in dense feeders and under inconsistent metadata conditions. This paper formulates distribution topology identification as a constrained inference problem that refines a utility-provided base topology using heterogeneous evidence while enforcing spatial feasibility and physical operational constraints. Instead of reconstructing connectivity from scratch, the proposed framework detects inconsistent assignments, performs localized reconnection within constrained neighborhoods to ensure scalability, and iteratively enforces physical feasibility to produce operationally consistent topology estimates. In addition, a falsification-driven reliability metric evaluates how strongly each inferred connection is supported relative to alternative feasible assignments, enabling utilities to prioritize verification efforts while preserving system-wide observability. The framework is validated using operational data from three feeders comprising more than $8{,}000$ AMI meters in collaboration with a large U.S. utility. Results demonstrate over $95\%$ topology reconstruction accuracy while significantly reducing computational effort compared with global inference approaches. The study further shows that correlation-based methods alone produce ambiguous assignments in dense urban feeders, whereas combining electrical measurements with spatial and operational constraints enables robust and scalable topology recovery under realistic deployment conditions. 

---
# Beyond Liars' Bench: The Impact of Lie Typology, Depth, and Sparsity on Deception Detection in LLMs 

**Authors**: Amr Moustafa, Max Feser, Florian Mai  

**Link**: [PDF](https://arxiv.org/pdf/2607.20479)  

**Abstract**: Training probes to detect deceptive outputs from large language models is still an open problem. Recent work has demonstrated that detection probes fail especially in out-of-domain scenarios -- training on one type of lie does not transfer well to deception scenarios involving other types of lies. In this work, we conduct a systematic study on how various factors impact detection performance: representation depth, probe expressivity, sparse feature representations, and the lie typology of the training data. To this end, we augment standard benchmark training data with a supplementary dataset containing diverse types of deception, including fabrication, omission, and exaggeration examples. Analyzing these factors across seven probe types, our experimental results show that the optimal representation depth is highly dataset-dependent, more expressive probes provide only selective gains over linear baselines, and sparse autoencoder features perform similarly to dense hidden states. Ultimately, we demonstrate that the choice of training data and lie typology substantially changes detectability, highlighting that deception detection is a highly representation-dependent problem. 

---
# Semi-Supervised Text-Attributed Graph Distillation 

**Authors**: Yurui Lai, Samir Moustafa, Renchi Yang, Tsz Nam Chan  

**Link**: [PDF](https://arxiv.org/pdf/2607.20477)  

**Abstract**: {\em Text-Attributed Graphs} (TAGs) have emerged as an expressive data model for integrating graph topology with rich textual semantics. Existing representation learning methods over TAGs suffer from severe scalability bottlenecks, particularly together with {\em Large Language Models} (LLMs). While data distillation offers a promising data-centric solution, existing methods fail to capture the complex interplay between graph and text modalities, struggle with the label scarcity inherent in semi-supervised settings, and lack the ability to produce the human-readable textual attributes required for downstream LLM-based tasks.
To address these challenges, we propose \algo{}, a unified semi-supervised framework guided by the {\em Wasserstein Distance} (WSD). Grounded in our empirical findings on real TAGs, \algo{} introduces a graph-text collaborative encoding module that utilizes dual-pathway encoders (graph-aware and -free) within a collaborative self-training scheme to harvest reliable pseudo-labels and fuse complementary graph-text features. Furthermore, we develop a theoretically grounded WSD-based graph sketching algorithm and a cost-effective LLM text synthesis module, which leverages cluster-based keyword extraction to generate coherent, human-readable summaries for condensed nodes. Extensive experiments on benchmark datasets demonstrate that \algo{} achieves a state-of-the-art performance-compression trade-off in terms of both GNN- and LLM-based downstream tasks, enabling effective and efficient TAG learning or analytics. 

---
# Benchmarking Large Language Models on Multi-Sensor Physical Hazard Assessment 

**Authors**: Faizan Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2607.20476)  

**Abstract**: We present an empirical benchmark evaluating how five large language models assess multisensor physical hazard data. Testing 60 scenarios across three categories - multi-sensor joint assessment, response proportionality, and pattern disambiguation - with 1,800 API calls at temperature 0.0, we find that all tested models consistently produced no precautionary warning signal across the tested scenarios where multiple sensors are simultaneously elevated below their individual safety limits, while achieving near-perfect accuracy on single-sensor threshold violations. All five models (ChatGPT-4o, Gemini 2.5 Flash, DeepSeek, Kimi, Llama 3.1 8B) score near zero on Category A multi-sensor scenarios (Q2: 0.000-0.208; Q3: 0.000-0.592) compared to strong performance on single-sensor scenarios (Category B Q1: 0.975-1.000). Structured tabular formatting shows no consistent advantage over plain prose; ChatGPT-4o performs significantly better under prose (p = 0.001). These findings have direct implications for practitioners deploying the tested models in physical safety monitoring systems. 

---
# SonicSampler: Unified Tile-Aware Kernels for LLM Sampling and Speculative Verification 

**Authors**: Pragaash Ponnusamy, Shivam Sahni, Jue Wang, Tri Dao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20475)  

**Abstract**: Sampling in LLM inference comprises a combinatorial set of logit processing, token selection, and verification operations for speculative decoding. However, existing implementations either accelerate only subsets of this pipeline, rely on multiple kernel launches, or assume homogeneous sampling behavior across a batch, limiting support for dynamic serving workloads and preventing efficient CUDA Graph execution. We present $\textbf{SonicSampler}$, a unified suite of tile-aware Triton kernels that vertically fuses the complete sampling pipeline into a fixed, workload-aware execution model. Our kernels support dynamic per-request sampling behaviors, including grammar-constrained decoding, repetition, frequency and presence penalties, logit bias, temperature scaling, top-$k$ / top-$p$ / min-$p$ filtering, and speculative verification - within a single batched kernel while remaining fully CUDA Graph-compatible. Central to our approach is a novel hierarchical two-stage top-$k$ algorithm that achieves up to $\textbf{10x speedup}$ over competitive baselines and exploits the low-entropy structure of LLM outputs to enable efficient selection over large vocabularies. Across heterogeneous speculative decoding workloads, SonicSampler achieves up to $\textbf{16x speedup}$ over state-of-the-art baselines while preserving flexible batched execution. 

---
# VeriSimpl: Robust Optimization Modeling from Natural Language using Simplification-based Verification 

**Authors**: Sumaya Abdul Rahman, Seckhen Ariel Andrade Cuellar, Ghani Raissov, Mohammad Raza  

**Link**: [PDF](https://arxiv.org/pdf/2607.20474)  

**Abstract**: Natural language interfaces can greatly benefit the accessibility and usability of optimization modeling, and recent advances in large language models (LLMs) show promise in automatically translating textual problem descriptions into executable solver formulations. However, a key challenge for existing approaches is to ensure that the inferred formulation correctly implements the intended task, even if it may execute without errors. We introduce VeriSimpl, a solver LLM framework for robust natural-language-to-optimization formalization. Our approach is based on the idea of simplification-based verification, where the optimization solver is leveraged to generate simplified diagnostic queries about a candidate formulation to allow the LLM to tractably reason about the correctness of the formulation with respect to the task description. We present such simplification strategies along different dimensions with respect to problem constraints and decision variables, which allow the LLM to reason locally under fixed global contexts. Evaluations on a range of optimization benchmarks show how our approach provides consistent improvements in accuracy over existing methods, while also providing a novel high-precision self-verification signal. 

---
# Incomplete Prompt Jailbreaks in Large Language Models 

**Authors**: Yeonjea Kim, Bumjin Park, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2607.20473)  

**Abstract**: Large language models (LLMs) are increasingly released as open-weight models with safeguards against harmful requests. Nevertheless, sentence completion remains vulnerable to incomplete harmful prompts. In this work, we formalize this phenomenon as incomplete prompt jailbreaks (IPJ) and provide a systematic empirical characterization of when and how incomplete prompts elicit harmful continuations. We analyze diverse attractor types associated with incomplete sentence continuation and show that LLMs systematically delay refusal until sentence termination. We further demonstrate that training models to refuse incomplete harmful prompts via parameter tuning is insufficient, failing to generalize across both content domains and attractor types. To enable fine-grained control, we identify two functional neurons: termination and continuation neurons. By clarifying their roles in sentence completion, we highlight the potential of neuron-level interventions for more precise and robust IPJ defenses. 

---
# Robust Critics: Defending LLMs Against Multi-Turn Attacks 

**Authors**: Roman Belaire, Arunesh Sinha, Pradeep Varakantham  

**Link**: [PDF](https://arxiv.org/pdf/2607.20472)  

**Abstract**: When a user asks a language model something harmful, is it a genuine attack or a misunderstood but well-meaning question? This ambiguity is one of the central challenges of LLM safety. A model that assumes the worst harms legitimate users; one that assumes the best is easily exploited. The problem is compounded in multi-turn dialogue, where an attacker's true intent may only reveal itself gradually across many exchanges, yet existing safety frameworks apply a contextual bandit treatment, ignoring the trajectory of the conversation.
To that end, we propose Dialogue Critic Guided Sampling (DCGS), a framework that addresses this by inferring user intent at every turn of dialogue. Instead of applying a fixed rule about what is or is not safe, DCGS learns what the user's intent is likely to be based on the full conversational history and generates responses accordingly. Formally, we model adversarial dialogue as a Markov Decision Process and learn value and regret-based critics at both the individual token and utterance (full response) levels, scoring candidate responses via an action-value critic. We prove that this inference-time reweighting approximates exponential tilting of the base policy, guaranteeing improvement in expected return for any finite candidate pool, a property that group-relative objectives do not exhibit. Evaluated on CARES-18k, WildJailbreak, Redbench, and Harmbench, DCGS outperforms strong robust baselines and frontier models on adversarial dialogue tasks. DCGS also transfers to frontier models, improving their robustness without fine-tuning. 

---
# Benchmarking the Personalization Capabilities of Large Language Models 

**Authors**: Ashutosh Srivastava, Siddharth Yedlapati, Vinay Aggarwal, Yaman Kumar Singla, Shashwat Dixit, Jitendra Ajmera, Balaji Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2607.20471)  

**Abstract**: Personalization, the act of varying a message to induce action from a specific receiver while keeping sender, channel, and time fixed, has a long tradition in psychology and marketing as a two-party problem in which sender and receiver have independent objectives. Large language models remove the bounded-inventory constraint of classical retrieval-and-ranking approaches by generating a continuum of message variants conditioned on inferred receiver state, raising the question of how well current models perform personalization in the classical sense. Existing LLM personalization benchmarks measure sender-side adaptation, in which the receiver is the same user the model is serving. The two-party question, whether a generated message induces its intended action in a third party, has been investigated only through A/B tests and small-scale human studies that cannot be re-run against a new model on demand. We adapt the Bayesian Persuasion framework of Kamenica and Gentzkow (2011) to generative agents and instantiate the formulation in sales, where receiver actions are routinely logged against the outreach that induced them. We release SDR-Bench, a public corpus of 6,279 customer success stories spanning 22 industries and approximately 200 enterprises, served through a temporally constrained simulation that prevents future-data leakage. Across frontier LLMs and deep-research agents, we observe a consistent personalization plateau and on a Fortune 100 tech cohort no model statistically separates successful from unsuccessful outreach. A field deployment with 12 professional sales representatives validates the framework, with 48 percent of model-generated content rated immediately useful and senior-expert agreement at Pearson 0.82. We release SDR-Arena and SDR-Bench publicly to support reproducible study of generative personalization at scale. 

---
# PlanE: Meta Planning of Data, Tuning, and Inference for Extractive-based LLMs 

**Authors**: Jiacheng Wang, Weiyan Zhang, Guangya Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20470)  

**Abstract**: Enhancing the task-specific capabilities of Large Language Models (LLMs) primarily requires substantial instruction-tuning datasets. However, the sheer volume of such data imposes a considerable annotation cost, and a lack of optimization methods for tailoring LLMs to specific tasks. To address the above issues, we propose a \textbf{Plan}ning framework for constructing \textbf{E}xtractive-based LLMs called \textbf{PlanE}, which includes data decomposition, instruction tuning, and prompt inference. Additionally, we introduce a Data-Tuning-Inference (DTI) planner, aimed at selecting the optimal base-LLM and its DTI combinations for specific datasets to improve construction efficiency. The experimental results demonstrate the effectiveness of our PlanE from two views: (1) across different datasets using the same base-LLM, and (2) on the same dataset using different base-LLMs. Furthermore, we validate the generalizability of the proposed DTI planner under different optimization objectives. The codes are publicly available at this https URL. 

---
# DecodeShare: Tracing the Shared Subspace of LLM Decode-Time Decisions 

**Authors**: Zishan Shao, Lixun Zhang, Kangning Cui, Yixiao Wang, Ting Jiang, Hancheng Ye, Qinsi Wang, Zhixu Du, Yuzhe Fu, Fan Yang, Danyang Zhuo, Yiran Chen, Hai Helen Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.20469)  

**Abstract**: Large language models (LLMs) handle many tasks with one set of parameters, but under KV-cached inference it is unclear what task-general structure, if any, is used at decode time rather than during prefill. We propose DecodeShare, a protocol that identifies a low-dimensional subspace consistently shared across tasks in decode-time hidden states, and then tests its causal role by removing that subspace only during decoding. In our experiments, disturbing the discovered shared subspace degrades decision performance far more than disturbing either a prefill-derived or random subspace under the same intervention budget. We further show this decode-shared subspace has practical consequences for activation steering: common steering directions can overlap the task-general decode channel. Projecting out this shared subspace directly separates the functional roles of the two components, while evaluating steering vectors at decode-time yields more reliable signal for downstream deployment than prefill-based proxies. Despite its compactness, the shared subspace can serve as a high-leverage causal channel at decode time. Code is available at: this https URL. 

---
# InferenceBench: A Benchmark for Open-Ended LLM Inference Optimization by AI Agents 

**Authors**: Jehyeok Yeon, Ben Rank, Maksym Andriushchenko  

**Link**: [PDF](https://arxiv.org/pdf/2607.20468)  

**Abstract**: AI agents are increasingly used to automate research and development tasks, yet existing benchmarks typically evaluate them on prescribed workflows or narrow action spaces. Even nominally open-ended tasks can often be solved by retrieving a well-known recipe and tuning a few hyperparameters, making it unclear whether strong results reflect genuine optimization or memorized solutions. We introduce InferenceBench, where an agent must deploy an OpenAI-compatible inference server and optimize the speed of LLM inference. Each agent receives a target LLM, one H100 GPU, an optimization scenario, and a wall-clock time budget of two hours. Three optimization scenarios isolate distinct bottlenecks of inference (prefill latency, decode latency, and concurrent request throughput) and a fourth balances all three at the same time. Across 15 frontier agent configurations, agents reliably improve over a naive PyTorch baseline (up to $8.08\times$) and often match or exceed serving engines with default settings ($4.05\times$ for vLLM), but still fall below a simple hyperparameter search under the same time budget (up to $11.53\times$). Qualitative analysis of agent trajectories shows that although agents enumerate many relevant optimization techniques, they overwhelmingly converge on a single inference framework. They test only a few distinct configurations and spend the remaining budget re-measuring, repairing, or optimizing hyperparameters rather than exploring substantially different strategies. This suggests the bottleneck is not domain knowledge, but the ability to propose diverse configurations, evaluate them systematically, and submit the best identified solution. Overall, InferenceBench reflects the ability of agents to operate in an open-ended AI engineering setting, where memorized solutions lead to limited improvements. 

---
# DC-Leap: Training-Free Acceleration of dLLMs via Draft-Guided Contiguous Leaping Decoding 

**Authors**: Yanhua Jiao, Tianyi Wu, Xiaoxi Sun, Yulin Li, HuiLing Zhen, Libo Qin, Baotian Hu, Zhuotao Tian, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20467)  

**Abstract**: While parallel decoding is central to the efficiency of Diffusion Large Language Models (dLLMs), current strategies are often hindered by overly conservative confidence thresholds. These thresholds, necessitated by the Joint Probability Dependence Error (JPDE), result in redundant denoising iterations and suboptimal inference speeds. To overcome this, we propose DC-Leap, a training-free framework that enables reliable acceleration of dLLMs in the moderate-confidence regime. DC-Leap introduces a Dynamic Contiguous Verification strategy that integrates strictly-ordered causal constraints into the parallel decoding process. By progressively validating token dependencies, this mechanism effectively neutralizes the JPDE, enabling reliable acceleration with comparable performance. Furthermore, DC-Leap incorporates the draft-guided decoding mechanism, where the draft helps extend the context by leaping forward across multiple tokens, providing look-ahead context and retaining the structural benefits of bidirectional attention during inference. Extensive experiments on standard benchmarks demonstrate that DC-Leap achieves substantial speedups, up to 53.19x on MBPP for long-sequence generation, and up to 105.02x when combined with KV-Cache with comparable generation quality. Code is available at this https URL . 

---
# JAXBench: Benchmarking Autonomous TPU Kernel Optimization 

**Authors**: Arya Tschand, Charles Hong, Julian Walker, Nina Cai, Shangkun Wang, Suvinay Subramanian, Sundar Dev, Vijay Janapa Reddi, Amir Yazdanbakhsh, Sethu Sankaran  

**Link**: [PDF](https://arxiv.org/pdf/2607.20466)  

**Abstract**: Rigorous benchmarks have driven progress in autonomous GPU kernel performance optimization by establishing a shared target to hillclimb on, but no equivalent exists for TPUs. We present JAXBench, a TPU-native benchmark suite for AI-generated kernel optimization on Google Cloud TPUs. JAXBench comprises 50 JAX workloads that are both relevant and provide headroom for optimization. We extract 17 production ML operators from architectures in the public MaxText library such as Llama-3.1, DeepSeek-V3, Mixtral, Mamba-2, and AlphaFold2, and translate 33 operators from KernelBench that are validated for correctness and set with new problem sizes that achieve high TPU v6e MXU utilization. Eight of the 17 production operators ship with hand-optimized Pallas kernels from the public Tokamax library and block-size tuned to establish an expert upper-bound baseline. We evaluate four feedback-driven methods on generating candidate Pallas kernels for JAXBench. Across the full suite with Gemini 3 Flash, we find that target-specific context matters more than model scale on a sparsely-documented DSL like Pallas. Conditioning on curated TPU documentation raises per-sample correctness from 5.8% to 37.3% and solves 48 of 50 benchmarks at a 1.28x geomean speedup. Search structure yields significant gains once correctness is achieved, with Autocomp's beam-search pipeline reaching a 1.36x geomean speedup over XLA. On the 8 hand-tuned kernels, Autocomp reaches 1.60x geomean over XLA, recovering most of the 2.08x Tokamax upper bound but trailing on the specialized paged and ragged attention operators. High-quality TPU kernel optimization remains a challenging task, and we release the JAXBench benchmark, evaluation harness, and baseline results to support open source contributions. 

---
# Stochastic Sampling is Epistemically Shallow: The Dimensionality Gap Between Temperature Variation and Model Diversity in LLMs 

**Authors**: Izhar Ali  

**Link**: [PDF](https://arxiv.org/pdf/2607.20464)  

**Abstract**: When a language model gives different answers on repeated runs, does that variation reveal what it does not know? Self-consistency turns the variation into a per-question uncertainty estimate via majority voting. But does the same variation reveal cross-question structure -- related questions flipping together, the way a diverse ensemble does? We compare two regimes on the same questions: one model run $100$ times at $\tau=1$ versus an ensemble of $24$ LLMs run once each at $\tau=0$. A Marchenko--Pastur random-matrix test separates signal from sampling noise on both sides. Within any single model, at most one dimension rises above noise across five families and three benchmarks (MMLU, HellaSwag, GSM8K). Across the ensemble, four eigenvalues clear the noise edge, while a matched-difficulty Bernoulli null produces at most one in $500$ Monte Carlo draws. Self-consistency gives accurate per-question uncertainty but no detectable cross-question structure; only a diverse ensemble surfaces what a model does not know. 

---
# ClickGuard: Detecting and Spoiling Clickbait News with Informativeness Measures and Large Language Models 

**Authors**: Wojciech Michaluk, Tymoteusz Urban, Mateusz Kubita, Soveatin Kuntur, Anna Wróblewska  

**Link**: [PDF](https://arxiv.org/pdf/2607.20463)  

**Abstract**: This paper presents an AI-driven browser extension that identifies clickbait to help users avoid misleading Internet articles. Moving beyond traditional detection, the application employs a hybrid machine learning architecture that combines transformer-based embeddings with linguistically motivated features and a custom "baitness" score. After evaluating various natural language processing techniques -- from classic vectorizers to large language model (LLM) embeddings -- an XGBoost-based model was developed that achieves an F1-score of 91% on the open combined dataset. Most importantly, the tool can warn users before and after they access a clickbait article. After opening an article, the user receives a percentage score indicating the likelihood that it is clickbait. The prediction is explained based on the analyzed metrics, including those specifically developed within the proposed system. The browser extension also provides a clickbait spoiler -- a one- to two-sentence summary of the entire article. Demo video:this https URL}{this https URL 

---
# Marking the Wrong Symptoms: Evaluating LLM Watermarks in Medical Texts 

**Authors**: Melanie Rieff, Robin Staab, Thibaud Gloaguen, Stefan Hegselmann, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2607.20462)  

**Abstract**: Large language models (LLMs) are increasingly integrated into clinical workflows, stressing the need for reliable traceability of model-generated output with watermarking. Yet, most watermarks are evaluated on general-purpose benchmarks, leaving domains like medicine, where small token-level perturbations can result in significant semantic changes, underexplored. In this work, we present the first rigorous study of how LLM watermarks affect medical performance, benchmarking 5 watermarking schemes across 11 LLMs and 7 VLMs on various tasks spanning unimodal and multimodal clinical reasoning. Importantly, we complement existing evaluations by introducing a human-expert-validated pipeline for systematically auditing medical reasoning quality, terminological precision, and induced hallucinations. Our results reveal that watermarking can induce substantial degradation across multiple failure modes, including lexical corruption, hallucinated terminology, and amplified misattribution or omission of image findings. Notably, we find that the absence of domain-specific analyses, combined with aggregate metrics that miss failures inherent to clinical text, can systematically obscure practical watermark-induced degradations. Our findings establish domain-specific evaluation as a prerequisite for the safe deployment of watermarked models in medicine, where current benchmarks can otherwise mask clinically consequential failures. 

---
# AINTMA: Agentic AI Architecture for Autonomous Test Management with Generative Intelligence, Secure Cloud Communication and Adaptive Quality Analytics 

**Authors**: Vinil Pasupuleti, Shyalendar Reddy Allala, Siva Rama Krishna Varma Bayyavarapu, Shrey Tyagi, Srinivasateja Songa  

**Link**: [PDF](https://arxiv.org/pdf/2607.20452)  

**Abstract**: Modern software quality assurance demands intelligent, autonomous systems capable of adaptive decision-making across distributed cloud environments. This paper presents AINTMA (Agentic Intelligent Test Management Architecture), a multi-agent agentic AI system that transforms traditional test management into an autonomous quality intelligence ecosystem. AINTMA deploys six specialized AI agents (Test Discovery, Risk Assessment, Reinforcement Learning Prioritization, Execution Orchestration, Generative Quality Intelligence, and Cloud Security Monitor) coordinated through a secure multi-agent communication framework over a cloud-native microservices infrastructure. The Generative Quality Intelligence agent employs large language models to produce plain language quality narratives, defect risk summaries, and data-augmented test recommendations. The RL Prioritization agent models test selection as a Markov Decision Process, learning contextual policies from large-scale historical test execution data (47 features, rolling 36-month window). Secure cloud communication is enforced through a zero-trust API gateway with OAuth2/JWT authentication, encrypted inter-agent messaging, and multi-tenant isolation. Evaluation across 12 heterogeneous software projects over 18 months demonstrates: 88.4% test prioritization accuracy (APFD, vs. 51.2% random, 82.1% best commercial baseline); 43% test cycle time reduction; defect escape rate reduced from 8.3% to 2.1%; 340% ROI at 9-month payback. The agentic architecture scales to 50,000+ test cases with sub-400ms response time, and the generative intelligence module achieves 4.3/5.0 developer usefulness rating. AINTMA demonstrates that agentic AI, combining autonomous multi-agent coordination, generative intelligence and secure smart connectivity, can fundamentally advance software quality management in cloud-scale enterprise environments. 

---
# 3D-Aware VLMs with Implicit and Explicit Geometries 

**Authors**: Wenhao Li, Xueying Jiang, Quanhao Qian, Deli Zhao, Ran Xu, Shijian Lu, Gongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21595)  

**Abstract**: Despite rapid progress, most existing vision-language models (VLMs) built from 2D visual inputs often struggle when handling various 3D tasks that require fine-grained spatial understanding and reasoning. To bridge this gap, we present VLM-IE3D, a unified framework that enhances the 3D spatial awareness of VLMs by equipping them with both implicit and explicit 3D geometries learned from RGB videos. Our VLM-IE3D introduces Implicit Geometry Tokens (IGTs) that capture high-level geometric priors from input videos, as well as complementary Explicit Geometry Tokens (EGTs) that encode detailed geometric structures from reconstructed 3D attributes. On top of that, VLM-IE3D comes with a 3D-aware adapter that effectively fuses the two types of geometric representations with 2D visual cues. This RGB-only design injects strong 3D inductive biases for fine-grained spatial understanding and reasoning without requiring any additional 3D inputs. Extensive experiments show that VLM-IE3D achieves superior performance consistently across various 3D tasks including 3D video detection, 3D visual grounding, 3D dense captioning, and spatial reasoning. Code and models are available at this https URL. 

---
# GraphVid: Interactive Graph-Controllable Video Generation 

**Authors**: Vedant Shah, Onkar Susladkar, Tushar Prakash, Kiet Nguyen, Tianjio Yu, Adheesh Juvekar, Muntasir Waheed, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2607.21580)  

**Abstract**: Controllable video generation remains challenging due to the difficulty of specifying precise multi-object interactions using text prompts or motion-control inputs that primarily constrain pixel movement. In practice, trajectory-based control often requires users to draw accurate tracks for multiple objects, which scales poorly with scene complexity and becomes ambiguous under occlusion or overlap. To enable flexible yet precise multi-subject control, we introduce $\textbf{GraphVid}$, a graph-conditioned image-to-video generation model that enables interactive control through structured interaction graphs. We further curate $\textbf{GraphVid-Bench}$, a large-scale interaction-centric video dataset with structured relational annotations to enable training of interaction-aware video generation models. Despite using substantially less training data and fewer trainable parameters than prior motion-control methods, GraphVid delivers strong controllability and video quality. Compared with Motion-I2V, GraphVid reduces FID by up to 39.9% and FVD by 37.6%, while improving PSNR (9.87=>15.98) and SSIM (0.38=>0.61). Our results highlight the potential of structured semantic interfaces as a powerful paradigm for controllable video generation. 

---
# Barzilai-Borwein Fails Superlinear Convergence on an Open Set of Quadratics for Every Dimension $n\geq 4$ 

**Authors**: Dawei Li, Xiaotian Jiang, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2607.21579)  

**Abstract**: Barzilai--Borwein (BB) method has shown strong practical performance in continuous optimization, yet its convergence dynamics remains poorly understood. In particular, a central unresolved question is whether BB converges superlinearly for almost every strictly convex quadratic problem and initialization. We provide a negative answer to this question. Specifically, for every finite dimension $n\geq4$, we construct a nonempty open, hence positive-Lebesgue-measure, family of strictly convex quadratic problems and initial points for which the long Barzilai--Borwein method (BB1) converges but cannot converge root-superlinearly. More precisely, with the explicit constants $\rho_{\min}=10^{-6},\rho_{\max}=0.61$, every spectral component of the gradient is bounded above and below by the corresponding geometric sequence. Consequently, the gradient norm and the energy norm of the error satisfy two-sided geometric estimates with the same rates, while the objective gap satisfies the corresponding estimates with squared rates. In particular, all three quantities are bounded below by geometric sequences, ruling out superlinear convergence. The construction is highly nontrivial, based on a computer-assisted proof of a nonresonant, attracting seven-cycle of the projectivized BB dynamics in dimension four. 

---
# Synthetic data generation framework for quality control automation in gravure printing 

**Authors**: Korota Arsène Coulibaly, Mohamed Hamlich, Khalid Hmali, Andrea Trombin  

**Link**: [PDF](https://arxiv.org/pdf/2607.21577)  

**Abstract**: Quality control in printing, particularly in rotogravure printing, still depends on slow, costly, and subjective manual inspection. Automated surface defect detection is critical for maintaining high-quality standards in rotogravure printing. Deep learning models give prospects for automation. However, training robust deep learning models, such as YOLO or Vision Transformers, is heavily hindered by the extreme scarcity of real-world industrial defects images. To overcome this limitation, this paper introduces a novel synthetic data generation framework tailored for rotogravure printing quality control. The proposed pipeline automatically generates high-fidelity images of specific printing defects (creases, streaks, misregistration, etc.) and outputs corresponding bounding boxes and annotations. To validate the framework, a synthetic dataset of 7533 images was generated and used to train the state-of-the-art object-detection model RFDETR. Experimental results demonstrate that the model trained on our synthetic data achieves a Mean Average Precision (mAP) of 80.9\% on real industrial testing samples. This framework provides a zero-cost, rapid-deployment solution for automating defect inspection in printing lines without requiring massive manual data collection. 

---
# Beyond Sufficiency: Time Series Explanation with Counterfactual Necessity 

**Authors**: Hongnan Ma, Yiwei Shi, Mengyue Yang, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21573)  

**Abstract**: Faithful explanations of time-series classifiers should identify subsequences that are not only sufficient to preserve a black-box model's prediction, but also necessary for maintaining it. However, existing sufficiency-oriented methods can assign high importance to spurious subsequences that support the prediction without being essential to the model's decision. We introduce \textbf{TimePNS}, a necessity-aware framework for time-series explanation. Inspired by Pearl's counterfactual notion of necessity, TimePNS assesses whether a temporal factor is necessary by intervening on it and measuring whether the original prediction is disrupted. The framework adopts a two-stage design. Stage I learns an identifiable causal generative process together with a sufficiency-oriented explanation mask. Stage II performs counterfactual interventions on temporal factors to derive necessity signals, which supervise a temporal gate that refines the initial explanation by suppressing non-essential components and emphasizing counterfactually necessary ones. Experiments on synthetic and real-world time-series benchmarks show that TimePNS more accurately identifies decision-critical subsequences and consistently improves sufficiency-necessity trade-offs over strong baselines. 

---
# Visual Contrastive Self-Distillation 

**Authors**: Yijun Liang, Yunjie Tian, Yijiang Li, Yuqi Jia, Furong Huang, Tianyi Zhou, Di Fu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21556)  

**Abstract**: On-policy self-distillation (OPSD) is promising as it removes the external teacher required by on-policy distillation (OPD), yet it still needs asymmetric information between teacher and student to ensure that the self-teacher provides a stronger learning signal than the student. Existing methods create this asymmetry either through privileged answers or visual evidence. We ask whether both can be removed, yielding a simpler form of OPSD driven purely by input conditioning. For this purpose, we propose Visual Contrastive Self-Distillation, namely VCSD, which converts image-content removal into an on-policy self-distillation signal. At each student-generated response prefix, the EMA teacher produces two next-token distributions under the same prompt and prefix -- one conditioned on the original image and the other on a content-erased control. Their token-wise log-probability difference highlights candidates whose likelihood is specifically increased by the instance-level visual content. We use this contrast to sharpen the teacher's original-image distribution within its plausible support, and distill the resulting full-distribution target into the student. Using ViRL39K dataset, VCSD consistently outperforms matched OPSD across Qwen3-VL and Qwen3.5 models. For example, on Qwen3-VL, it improves the seven-benchmark aggregate from $62.27\% \rightarrow 67.04\%$ at 2B, $71.30\% \rightarrow 73.16\%$ at 4B, and $72.51\% \rightarrow 76.26\%$ at 8B. Furthermore, VCSD requires no external teacher, privileged answers, visual evidence signals, reasoning traces, or additional inference-time cost. 

---
# From Resource Flow to Executable Tests: Petri-Net-Guided LLM Test Generation for Concurrent Stateful Rust APIs 

**Authors**: Kaiwen Zhang, Guanjun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21530)  

**Abstract**: Concurrent stateful library APIs expose behavior through evolving resource ownership, lifecycle states, and competing interleavings. Large language models can synthesize executable Rust tests, but their outputs often violate API preconditions, remain shallow, or reduce concurrency to accidental sequential traces. Conversely, model-based and systematic testing techniques provide semantic control but commonly require substantial handwritten code to turn abstract scenarios into executable tests. This paper addresses the gap between formal scenario design and low-cost test concretization. We present a Petri-net-guided methodology for test generation over concurrent stateful Rust APIs. The method represents API resources, lifecycle conditions, and causal dependencies as colored tokens and transitions; derives legal deep-state, near-legal, and partial-order concurrent scenarios; and uses these scenarios as a constrained intermediate representation for LLM-based code synthesis. A local-faithfulness contract and structural repair loop preserve the modeled intent during concretization, while Petri-guided schedule shaping prioritizes high-conflict concurrency skeletons for systematic exploration. A layered semantic oracle then distinguishes synthesis failures from violations of the target API's expected behavior. 

---
# ElasticTTT: Prior-Preserving Test-Time Tuning for Video Editing 

**Authors**: Yueyi Liu, Chi Zhang, Sen Cui, Miao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21529)  

**Abstract**: Test-Time Tuning (TTT) on pretrained diffusion models has emerged as a powerful paradigm for video editing. However, there exists a foundational mismatch between the distribution-mapping nature of generative models and the single-point optimization of standard TTT. In this paper, we demonstrate that this mismatch triggers \textit{Prior Collapse}, a degenerate state where the model discards the text conditions and spatial latents, collapsing generations to the source video, or entangling the features of distinct regions. To resolve this, we propose \textbf{ElasticTTT}, a novel framework that preserves the prior generative distribution and rescues generative elasticity. Specifically, we propose \textit{Target Distribution Regularization} to prevent sharp memorization minima, \textit{Contrastive CFG} to guide inference away from source biases, and \textit{Asynchronous Noise Schedule} to preserve unedited regions. Extensive evaluations, supported by theoretical analysis, demonstrate that ElasticTTT successfully preserves the generative prior of the base model, achieving state-of-the-art performance on one-shot video editing. 

---
# GS-Agent: Creating 4D Physical Worlds With Generative Simulation 

**Authors**: Hongxin Zhang, Chunru Lin, Junyan Li, Zhou Xian, Tsun-Hsuan Wang, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2607.21522)  

**Abstract**: Creating dynamic and physically realistic 4D worlds from natural language descriptions is both fascinating and challenging. Traditional computer graphics methods rely on manual creation, requiring extensive human effort to fine-tune materials, motions, and visual fidelity. Recent advances in generative foundation models have sparked interest in learning to generate such 4D worlds from large-scale data; however, existing methods still struggle to ensure physical plausibility and controllability. In this work, we take a different path by leveraging foundation models to construct an agentic system that emulates how humans traditionally create 4D worlds, yet automates the entire process. We present GS-Agent, an end-to-end multi-agent framework that integrates physics engines in the loop to generate realistic, dynamic, and controllable 4D physical worlds from natural language. Inspired by how humans build 4D worlds, GS-Agent decomposes the task into entity management, covering 3D asset curation, material tuning, placement, and motion control, and rendering configuration, including camera and lighting manipulation. Multiple agents with distinct expertise interact with the physics engine via code, seek multimodal feedback, and collaborate to iteratively construct 4D worlds that align with the given descriptions. Experimental results show that GS-Agent effectively converts natural language into diverse and physically plausible 4D worlds exhibiting rich interactions among liquids, deformable objects, and rigid bodies, while achieving cinematic camera and lighting control. We envision GS-Agent as a foundation for a new paradigm in 4D world generation, empowering creative content creation and physical AI. Project page at this https URL 

---
# Improved lower bounds for the Shannon capacity of odd cycles 

**Authors**: Nathaniel Itty, Christopher D. Rosin, Chase Carstensen, Daniel Reichman  

**Link**: [PDF](https://arxiv.org/pdf/2607.21517)  

**Abstract**: The Shannon capacity $\Theta(G)$ of a graph $G$ quantifies the maximum rate at which information can be transmitted with zero error over a noisy channel. It is lower bounded by $\alpha(G^d)^{1/d}$ for any $d$, where $\alpha(G^d)$ is the independence number of the $d$-th strong power of $G$. We construct independent sets of size $134753$ in $C_7^{10}$, $21909$ in $C_{11}^{6}$, and $62530$ in $C_{13}^{6}$, improving the best known lower bounds for the Shannon capacity of these graphs to $\Theta(C_7)\geq 134753^{1/10}>3.258020$, $\Theta(C_{11})\geq 21909^{1/6}>5.289773$, and $\Theta(C_{13})\geq 62530^{1/6}>6.300109$. We also improve the best known lower bounds on the independence numbers of several individual strong powers of odd cycles that do not improve the Shannon capacity lower bound. The constructions were discovered through iterative interactions with a Large Language Model (LLM), illustrating the potential of LLMs for finding explicit combinatorial constructions. 

---
# Artificial Epanorthosis: Why large language models overuse a classical rhetorical figure, and how to mitigate it 

**Authors**: Federico Boggia  

**Link**: [PDF](https://arxiv.org/pdf/2607.21498)  

**Abstract**: A rhetorical figure that Cicero and Quintilian catalogued two thousand years ago reappears, systematically, in the text of large language models: epanorthosis, the self-correction of the specimen «This is not a course. It is a journey of transformation». This essay argues that the overuse is a trained disposition, driven mainly by a training distribution rich in promotional prose and by preference tuning (RLHF) that rewards confident, emphatic phrasing; the left-to-right nature of generation is an amplifier rather than the root cause. Building on evidence that models diverge from human rhetorical style, and on Fontanier's classification of epanorthosis as a figure of thought, it sets out a programme that scores the figure against genre-specific human baselines through an Epanorthosis Index (density relative to the human rate). A first measurement, on three sizes of one instruction-tuned model family, finds mis-calibration by register in both directions: the models overshoot in oratory (about twofold, near threefold in Italian, concentrated in the larger tiers) and undershoot in informal question-and-answer writing, while matching humans in argument, journalism, and encyclopedic prose. Three constructive contributions follow: a survey of mitigation techniques centred on lightweight LoRA adapters; a demonstration, in Italian, that a one-line instruction cuts the figure by half to nearly three-quarters and that a supervised-fine-tuning adapter removes it almost entirely, with a scaling coefficient that dials the reduction back onto the human rate; and the argument that the target is calibration to the human rate for each genre, not elimination. It closes on the stakes: the real risk is that we begin to write like the machines. 

---
# Compact Latent Coordination for Autonomous Vehicles at Unsignalized Intersections 

**Authors**: Gil Lifshits, Igal Bilik, Gilad Katz  

**Link**: [PDF](https://arxiv.org/pdf/2607.21488)  

**Abstract**: Coordinating autonomous vehicles at unsignalized intersections remains a critical challenge for multi-agent reinforcement learning (MARL) systems, which typically struggle with combinatorial action spaces, reliance on privileged information, or rigid agent designs. We propose Master-Agent Proto-plan System (MAPS), a hierarchical deep reinforcement learning (DRL) architecture in which a centralized Master agent generates a compact, continuous embedding, denoted as proto-plan, that encodes a global coordination strategy. Decentralized Worker agents integrate this embedding with local observations to execute vehicle-specific control, decoupling strategic intent from tactical execution and enabling independent optimization of each module.
As a proof-of-concept evaluation of this coordination mechanism, we test MAPS across 72 intersection configurations in HighwayEnv. MAPS achieves collision-free navigation while significantly reducing average travel time, outperforming state-of-the-art baselines. The learned proto-plans further exhibit robust generalization: a system trained with three agents achieves a 94% success rate when deployed zero-shot to five-agent scenarios, confirming that proto-plan-based hierarchical learning provides a promising framework for multi-vehicle coordination. 

---
# Error Certificates for KV-Cache Eviction via Randomized Design 

**Authors**: Peng Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.21475)  

**Abstract**: Deterministic KV-cache eviction keeps the top-$k$ tokens under an importance score and deletes the rest. We prove that this design cannot know what it destroyed: evicted values can be altered so that everything the serving system retains is unchanged while the true attention-output error grows arbitrarily, so no serving-time estimator of that error is consistent. Randomized eviction restores identifiability. With a Poisson-sampled tail at known inclusion probabilities, one logit offset performs the Hájek correction inside the softmax, and a survey-sampling variance estimator over the retained set becomes a per-step error certificate with 0.97 empirical coverage at no accuracy cost. On real workloads we pre-registered seven claims and lost three: question-aware eviction at 25--50\% budgets is nearly free; output log-probability predicts failure better than the certificate; certificate-gated budget escalation adds nothing. What survives is attribution: the certificate separates cache-induced from inherent failures (AUC 0.73--0.75, against 0.47--0.54 for output confidence) and schedules recomputation better than random or confidence gating. Randomization buys attribution, not prediction. 

---
# Thinkink: 2D Spatial Ink-native Interaction with LLMs 

**Authors**: Mohammad Hasan Payandeh, Daniel Vogel, Jian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.21468)  

**Abstract**: People often use handwritten notes and sketches to externalize ideas for ideation. To integrate large language models (LLMs) into this practice, we propose Thinkink. Prompts can be handwritten text or drawn sketches with LLM-generated responses visualized as ink-like text and sketches spatially integrated into a shared canvas. A semantic tree streamlines ink interpretation, and a lightweight UI provides explicit control using a state machine. The tool was designed using a three-stage process. A formative study (N=12) examined current practices with conventional and digital inking methods. The results informed a technical probe for a diagnostic study (N=6) identifying usability and human-LLM interaction challenges. This motivated the design of Thinkink, with a final study (N=10) examining how people incorporate it into their ideation practices. We contribute design implications and a tool for ink-native LLM interaction where the user and LLM write and draw in a shared 2D canvas. 

---
# RUMBA: Russian User Memory Benchmark 

**Authors**: Elizaveta Shevtsova, Inna Glebkina, Mark Baushenko, Pavel Gulyaev, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2607.21447)  

**Abstract**: The ability to handle long-term memory in LLMs is becoming increasingly critical, yet existing benchmarks remain English-centric and rely on aggregate retrieval metrics, failing to capture interactions between long-range context, temporal information, and reasoning. To address this, we introduce RUMBA (Russian User Memory BenchmArk) - a new benchmark for long-term conversational memory that provides a fine-grained taxonomy of memory-centric question types and a unified methodology accounting for semantic type, session scope, temporal reasoning, and the explicitness of temporal expressions. RUMBA consists of timestamped user-assistant dialogues with QA pairs requiring retrieval, combination, and reasoning across sessions. While designed for Russian, we also provide an aligned English subset under the same methodology. We evaluate contemporary memory systems and long-context models, and show how RUMBA serves as a diagnostic tool to analyze model behavior across benchmark slices and identify strengths and failure modes of different memory mechanisms. 

---
# Adaptive Identity Anchoring: Closed-Loop Keyframe Placement for Synthetic Paired Supervision in Video Face Swapping 

**Authors**: Logan Robbins  

**Link**: [PDF](https://arxiv.org/pdf/2607.21434)  

**Abstract**: Video face swapping has no natural paired supervision: no real footage exists of one person's face performing another person's video. The strongest current answer, DreamID-V's SyncID-Pipe, mints pairs by replacing the identity in exactly two frames of a real clip -- the first and the last -- and regenerating the rest from a pose sequence alone. Pose carries no appearance evidence of the swapped-in identity, so over long clips, occlusions, and extreme pose excursions the synthesized identity has a long unanchored span on which to drift; no published ablation examines anchor count or placement. We propose Adaptive Identity Anchoring (AIA): (i) generalize the synthesizer to arbitrary anchor sets, architecturally natural for diffusion-forcing-style transformers where conditioning on a frame is clamping its tokens to zero noise; (ii) place anchors by a closed feedback loop that scores every generated frame against the real reference identity and inserts an image-face-swapped anchor at the worst-scoring frame until the pair passes a threshold or exhausts a budget; (iii) reuse the loop's verdict as an automatic data filter. A second pathology, the beauty-filter look of over-smoothed skin, has the same root cause: micro-texture, like identity, is priced by none of the pipeline's objectives. We therefore pair AIA with Reality-Referenced Texture Restoration: matched re-graining from each real frame's non-face regions, band-split transfer of sub-identity micro-texture from the real footage, and a second, spectral acceptance channel refereed by the footage's own spectrum. Identity-anchor density, we argue, is a controllable quality dial, and we specify falsifiable experiments -- drift-versus-gap curves, uniform-versus-adaptive placement at matched budgets, student training on AIA-minted data, and texture ablations with a human beauty-filter study -- that would validate or refute the proposal. 

---
# Token Budget Saturation and Mechanistic Early Detection of Reasoning Non-Convergence in Chain-of-Thought Models 

**Authors**: Renuka Oladri, Niveda Jawahar, Abdirisak Mohamed  

**Link**: [PDF](https://arxiv.org/pdf/2607.21433)  

**Abstract**: Chain-of-thought reasoning models such as DeepSeek-R1-Distill-Qwen-7B exhibit a bimodal convergence pattern: generations either terminate within a token budget (converged) or exhaust it without reaching a conclusion (non-converged). We characterize this phenomenon empirically, showing that converged generations achieve 90.3% accuracy on AIME 1983-2024 while non-converged ones achieve only 6.6%, with an overall convergence rate of 62.0%. We then ask whether this outcome is detectable early in the thinking chain using internal model representations. Training linear probes on hidden-state activations at token positions 50-300, we find that layer-20 activations at token 150 achieve AUC 0.608 (+-0.080, 5-fold CV), reliably above chance even at token 50. Activation probes consistently outperform behavioral baselines derived from token entropy and repetition statistics. A sweep-level permutation test yields p=0.063 (100,000 permutations), consistent with a modest signal that our sample size cannot confirm at conventional thresholds. These findings suggest that convergence fate is partially encoded in intermediate representations well before the generation ends, opening a path toward early-exit inference and adaptive compute allocation. 

---
# Cycle-Consistent and Uncertainty-Aware Neural Surrogates for Tokamak Edge Plasmas 

**Authors**: Abdourahmane Diaw, Sebastian De Pascuale, Jae-Sun Park, Ivan Paradela Perez, Jeremy D. Lore, Stefan Dasbach  

**Link**: [PDF](https://arxiv.org/pdf/2607.21407)  

**Abstract**: The boundary and divertor plasma govern how a tokamak exhausts power and particles, setting heat fluxes, target conditions, and the onset of detachment. Predicting these quantities is essential for operating current and future devices, but edge simulations that resolve them are too slow for parameter scans, optimization, or real-time control. Machine-learning surrogates offer a fast alternative, yet most are forward-only: they cannot recover input parameters from observations or assess the reliability of their predictions. We introduce a cycle-consistent neural surrogate for edge plasmas, combining a conditional U-Net forward model with an optimization-based inverse method built on the frozen forward network. The forward model maps five control parameters to two-dimensional plasma-state fields on the SOLPS-ITER mesh; the inverse method enforces consistency between forward and inverse predictions, a self-supervised quality check needing no ground-truth labels at inference. An ensemble of multilayer perceptrons also predicts electron temperature and density profiles at the outboard midplane and divertor targets, with uncertainty estimates that flag where more simulations are needed. The forward model achieves normalized root-mean-square errors below 2.6% and Pearson correlations above 0.95 for all fields. Cycle-consistency regularization raises the average cyclical $R^2$ from 0.59 to 0.99 without degrading forward accuracy and enables recovery of the core fueling rate; all five control parameters are recovered with Pearson $r\ge0.97$. A $k$-d tree warm start yields a database completion rate above 95%, versus roughly 30% outright failures when cold-started. With about $4\times10^6$ parameters, the model produces full 2D predictions in milliseconds, five to six orders of magnitude faster than SOLPS-ITER, enabling real-time control, parameter scans, uncertainty analysis, and digital twins. 

---
# When Are Reasoning-Based Guardrails Not Efficient? ResponseGuard: A Fast Vision-Language Guard for Real-Time Moderation 

**Authors**: Dongbin Na  

**Link**: [PDF](https://arxiv.org/pdf/2607.21401)  

**Abstract**: A vision-language AI assistant returns its answer as a stream of generated tokens. Therefore, a safety guard that watches that answer has to keep up with the stream and stop a harmful reply before a user reads it. Recent vision-language guardrails instead generate a chain of thought before they issue a verdict. They believe that step-by-step reasoning yields a safer guard. This design makes the guard heavy and slow, since the model must decode many tokens for harmfulness detection. We pose the question of whether a vision-language guard really needs to reason in order to screen a response. We answer with a guard that has no chain. ResponseGuard reads a harmful verdict from a single pooled representation of the request, the response, and the image in one forward pass. Across a standard multimodal guardrail benchmark, our 2B ResponseGuard outperforms a recent 3B reasoning-based vision-language guard on response harmfulness detection, without any reasoning and at about 150 times lower time cost. On request harmfulness the reasoning guard retains an overall lead, and the remaining gap on both tracks sits on the image-only cells. We observe that the gap may stem from the frozen vision encoders that both designs use rather than from the missing chain. We have also found the reasoning guard directs almost none of its verdict attention to the image. Based on a single-pass detection, ResponseGuard can screen an answer sentence by sentence as it streams and stop a harmful answer before it finishes. For guarding the response of a vision-language model, a calibrated single-pass label may provide a sufficient safety signal. We fully release all source code, trained models, and datasets at this https URL. 

---
# VoLN: Vision-Only Long-Horizon Navigation---Paradigm, Benchmark, and Method 

**Authors**: Jiabin Lou, Haopeng Wang, Yuanshuai Wang, Xinyu Liu, Xuxin Lv, Yuxin Guo, Lei Huang, Rongye Shi, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21400)  

**Abstract**: Vision-and-Language Navigation (VLN) enables embodied agents to follow natural-language instructions. However, route-level instructions commonly encode spatial priors, such as orientation, distance, and layout, that are not explicitly available from onboard sensing at deployment in open, GPS-denied environments. Benchmark performance under such interfaces therefore jointly reflects visual navigation ability and the use of route structure explicitly supplied by the task description. As a complementary formulation, we propose Vision-Only Long-Horizon Navigation (VoLN), which shifts route-relevant information from externally supplied instructions and global guidance to locally observable in-scene cues. In VoLN, goal views specify the destination, while route-relevant information is available only through locally observable in-scene cues that the agent must detect, interpret, and select online. We instantiate VoLN for aerial navigation through VoLN-UAV, a 7,210-episode benchmark that combines long-horizon goal-directed flight, continuous 3D motion, large viewpoint changes, and context-dependent beacon selection. We further provide VoLN-MLLM as an initial reference baseline. It aligns self-supervised visual features with a structured semantic space and predicts short-horizon waypoint segments from observation history, goal views, retrieved visual--semantic tokens, and proprioception. On the five-environment Test-Unseen split, it obtains success rates of 7.4%, 4.5%, and 1.8% on Easy, Normal, and Hard episodes, respectively. These results provide an initial evaluation of VoLN and reveal substantial remaining challenges in long-horizon evidence integration, cross-view goal matching, and closed-loop stability. Project page: this https URL 

---
# Mean-to-Score Discrete Diffusion: Posterior-Mean Denoisers for Score Entropy 

**Authors**: Jingyuan Li, Xiaoyi Jiang, Yixuan Jiang, Wei Liu, Yi Zhu, Zuoqiang Shi, Pipi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21372)  

**Abstract**: Score Entropy Discrete Diffusion (SEDD) parameterizes discrete reverse processes with unconstrained positive score ratios. While positivity guarantees nonnegative reverse jump rates, it does not ensure Bayes realizability: ratios at a noisy state need not be jointly induced by any clean-token posterior under the forward kernel. The score-entropy loss has the correct population optimum but does not enforce this constraint away from it. In a trained pure-uniform SEDD checkpoint, roughly one quarter of complete score vectors violate the coordinate box, while more than half lie inside it yet remain materially incompatible with any valid posterior. Such violations can produce negative pre-normalization weights in finite-step sampling. Projecting raw scores onto the bridge polytope removes all observed negative weights and improves external generative PPL from $203.6$ to $175.1$ without changing the sampler. We introduce \emph{mean-to-score} (M2S), which predicts a clean-token posterior mean and converts it to the score through an exact kernel-dependent linear map. The construction applies to any known coordinate-wise continuous-time Markov chain (CTMC) satisfying a mild support condition. For uniform corruption, it maps the probability simplex onto the bridge polytope; for absorbing-mask corruption, the resulting objective recovers MD4 exactly. In a controlled 28.4M-parameter CIFAR-10 comparison, M2S lowers test BPD from $3.173$ to $3.129$ and FID-50k from $\CifarSEDDFID$ to $\CifarMtwoSFID$. A 170M-parameter M2S model trained on about 262B OpenWebText token slots outperforms the evaluated pure-uniform SEDD, GIDD, and Neural CTMC checkpoints at every tested sampling budget, reaching generative PPL $143.3$ at 128 steps versus $183.6$ for the strongest pure-uniform baseline. 

---
# DINOde: Continuous Vision-Text Alignment for Open-Vocabulary Semantic Segmentation 

**Authors**: Sung-Hoon Yoon, Hoyong Kwon, Changgyoon Oh, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2607.21371)  

**Abstract**: Open-vocabulary semantic segmentation (OVSS) leverages textual semantics to segment objects beyond predefined categories. While the self-supervised model DINOv3 provides strong structured visual representations, its lack of native textual alignment hinders its direct application to OVSS. To bridge this gap, we propose DINOde, an ODE-based framework that continuously aligns CLIP text embeddings with the DINO visual manifold. Our approach employs two complementary components: (i) Semantic Text Flow (STF), which evolves text embeddings toward the DINO manifold through a continuous ODE trajectory, and (ii) Global Context Flow (GCF), which progressively refines the holistic image representation carried by DINO's CLS token. To preserve the hyperspherical geometry of the feature space during this evolution, we further introduce Velocity Tangent Projection, which constrains the learned velocity field to the tangent space. By modeling alignment as a continuous trajectory, DINOde avoids the manifold entanglement inherent in discrete MLP projections and yields more robust cross-modal alignment. Extensive experiments demonstrate that DINOde consistently outperforms existing methods and achieves state-of-the-art performance across multiple OVSS benchmarks. The code is available at this https URL. 

---
# Hilbert Operator for Progressive Encoding (HOPE): A Mathematical Framework for Deconstructing Learned Representations in Deep Networks 

**Authors**: Hossein Mobahi, Peter L. Bartlett  

**Link**: [PDF](https://arxiv.org/pdf/2607.21366)  

**Abstract**: Deep neural networks encode complex representations, but deconstructing this internal knowledge remains a challenge. Given the link between learning and compression, network compression offers a promising lens to analyze this knowledge. However, standard compression heuristics often suffer from scale symmetries and architectural biases. To resolve these, we introduce Hilbert Operator for Progressive Encoding (HOPE), a mathematical framework to gradually deconstruct the representations in trained network weights.
HOPE shifts network compression from the discrete domain into a Hilbert space of continuous functions. By modeling individual neurons as rank-1 Hilbert-Schmidt operators, HOPE unifies pruning and neuron merging as low-rank subspace projection. Extending this formulation, HOPE introduces macro block eviction to encompass multi-layer structures like entire residual pathways under the same unified metric. This unified approach enables unbiased architectural decisions across layers with different types and sizes. HOPE is a data-free and hyperparameter-free framework. We present proof-of-concept experiments in model compression and fine-tuning to highlight the practical potential of our theory. 

---
# M$^3$-Gen: Interpretable Multimodal Generation of Gene Expression Profiles Using Clinical and Imaging Data 

**Authors**: Francesca Pia Panaccione, Carlo Sgaravatti, Marco Venere  

**Link**: [PDF](https://arxiv.org/pdf/2607.21343)  

**Abstract**: Integrating heterogeneous biomedical data, including clinical metadata, histopathology images, and molecular profiles, is crucial for comprehensive disease understanding. However, gene expression data acquisition remains constrained by high costs and privacy concerns, limiting its use in multimodal research and AI-driven applications. We present MultiModal Molecular Generation (M$^3$-Gen), a novel framework for the generation of gene expression profiles by conditioning a Generative Adversarial Network on histopathology images and clinical metadata. M$^3$-Gen learns a unified latent representation from the clinical variables and the images, leveraging contrastive learning, and exploits the embeddings of the two modalities to guide a generative model in producing biologically coherent gene expression profiles. Evaluations on the TCGA dataset demonstrate that M$^3$-Gen generates realistic and functionally meaningful gene expression data. Importantly, by integrating multiple modalities in an attention-based mechanism, M$^3$-Gen provides intrinsic explainability: it allows the identification of which regions of the histopathology images most strongly influenced the generation of specific gene expression profiles, making the model's decisions interpretable by design. 

---
# Phonetic forced alignment for low-resource language varieties: Model training and evaluation on Chengdu Mandarin 

**Authors**: Zhiheng Qian, Aini Li, Hai Hu, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.21332)  

**Abstract**: Phonetic forced alignment is a key technique in phonetic research, yet existing alignment systems lack specialized models for low-resource language varieties. We address this by training text-dependent and text-independent aligners for Chengdu Mandarin using a 17-hour corpus and a custom G2P dictionary. We trained a text-dependent GMM-HMM model (Chengdu-MFA) and fine-tuned a pretrained audio encoder on frame classification with Chengdu-MFA's pseudo label for text-independent alignment (Chengdu-FC). Evaluation on an expert-annotated test set show that both methods significantly outperform Standard Mandarin baselines. Chengdu-MFA reduced average phone boundary differences by 31.8%, while Chengdu-FC achieved a 61.2% reduction. This work establishes a practical bootstrapping pipeline for developing accurate aligners for under-resourced varieties without labor- and time-intensive manual annotation. 

---
# From Static Bibliometrics to Dynamic Knowledge Graphs: An LLM-Powered Framework for Modernizing Science, Technology, and Innovation (STI) Analytics 

**Authors**: Muhsen Hammoud  

**Link**: [PDF](https://arxiv.org/pdf/2607.21327)  

**Abstract**: Bibliometric indicators - citation counts, h-indexes, co-authorship networks - have long anchored science, technology, and innovation (STI) analytics, yet suffer from temporal lag, semantic shallowness, and an inability to capture the non-linear dynamics of contemporary knowledge ecosystems. Dynamic knowledge graphs and large language models (LLMs) have each been proposed as remedies, but neither is sufficient alone: existing scholarly knowledge graphs remain largely static, while LLM-driven pipelines are prone to hallucination, opacity, and corpus bias without structured grounding. This paper proposes a hybrid, symbolic-first framework integrating all three traditions under explicit methodological constraint. Organized across five layers - an open scholarly data backbone, a dynamic versioned knowledge graph, a constrained LLM-assisted semantic augmentation layer, a multi-layer validation pipeline, and an analytics layer - the framework positions LLMs strictly as generators of provisional candidate enrichments. Candidates become analytically admissible only after passing structural, evidentiary, comparative, and selective expert validation, with full provenance recorded at every stage. The analytics layer supports both established bibliometric indicators and extended graph-based analyses, including trend emergence detection, science-to-technology pathway mapping, and policy-oriented gap analysis. The framework's central theoretical contribution is treating validation as the mediating principle between semantic flexibility and epistemic discipline, enabling STI analytics that is semantically richer and temporally more responsive than static bibliometrics while remaining aligned with the evidentiary standards of science-of-science research. Governance considerations addressing reproducibility, bias, and auditability are also discussed. 

---
# Toward cryptographically verifiable authorization for autonomous AI agents: A security hypothesis, preliminary formal model, and proof-of-concept implementation 

**Authors**: M. Llambí-Morillas, D. Fernández-Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2607.21325)  

**Abstract**: Autonomous AI agents increasingly execute actions, invoke tools, and operate on protected resources with limited human oversight. Existing authentication and authorization mechanisms establish identity and delegate authority, but do not inherently provide cryptographic evidence that a concrete request issued by a specific agent satisfies the applicable policy in a specific execution context. This paper hypothesizes that agent authorization can be formalized as a cryptographically verifiable relation, denoted $R_{CVA}$, that jointly binds an agent principal, a concrete authorization request, an execution context, and the satisfaction of an applicable policy, while selectively preserving the confidentiality of private authorization attributes. We introduce a preliminary formal abstraction for Cryptographically Verifiable Agent Authorization (CVA), define a compact set of candidate security properties including authorization soundness, principal binding, request binding, policy binding, and replay resistance, and provide an executable zero-knowledge proof of concept that instantiates selected elements of the model over a Groth16 zk-SNARK construction. We further identify and formalize the structural separation among identity binding, authorization-request binding, and runtime execution binding as a central open problem in the design of secure agentic systems (a distinction {not explicitly addressed by} current agentic security frameworks) and present a falsifiable research agenda for its resolution. 

---
# GRADRAG: Cross-Component Prompt Adaptation for Coordinated Multi-Agent RAG 

**Authors**: Paolo Pedinotti, Enrico Santus  

**Link**: [PDF](https://arxiv.org/pdf/2607.21324)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems increasingly employ multiple LLM agents. Yet, most prior work optimizes components in isolation rather than coordinating improvements across the pipeline. We introduce GRADRAG, a framework for cross-component prompt adaptation that models the RAG pipeline as a computational graph and propagates structured evaluation feedback to update upstream agents. An Evaluator critiques downstream answers and supporting evidence, producing actionable feedback that a Prompt Optimizer uses to iteratively update adaptive agents, such as retrievers, graph constructors, and answerers. The Evaluator also triggers early stopping when the output is deemed satisfactory. We evaluate GRADRAG on the SQUALITY and QMSUM benchmarks under two retrieval paradigms: flat chunk-based retrieval using IRCoT-style query refinement (Trivedi et al., 2023), and graph-based retrieval that constructs and iteratively enriches an entity-relation graph from the document. Across both settings, GRADRAG consistently outperforms one-step refinement baselines that update only the final generator, achieving a 12-15 percentage point net preference margin in LLM-judged pairwise comparisons, with most gains realized within two refinement iterations. 

---
# PC-Edit: Prompt-Contrastive Region Discovery and Region-Guided Editing 

**Authors**: Jian Zhang, Zhijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21318)  

**Abstract**: Replacing an object with one that differs in category or shape requires complete source removal, natural target formation unconstrained by the source silhouette, and preservation of unrelated content. Existing training-free editors either localize edits from terminal predictions under source and target prompts or preserve unrelated content through spatially unselective source-feature reuse without explicit region discovery. Before reaching the terminal predictions, prompt-induced semantic differences undergo additional network transformations that may obscure their spatial localization, reducing localization precision. Spatially unselective feature reuse forces a trade-off between edit completeness and background preservation. Therefore, we propose PC-Edit, a prompt-contrastive framework for training-free MM-DiT editing. PC-Edit contrasts the image-token attention outputs under the source and target prompts, capturing prompt-induced semantic differences directly where text-conditioned information is delivered to image tokens. The same contrast identifies a source-erasure region during inversion and a target-emergence region during denoising. Their union suppresses source remnants while allowing the target object to form naturally. PC-Edit further couples region discovery and background preservation within each sampling step by estimating the current edit region from preceding attention blocks and immediately injecting cached source K/V features outside it in subsequent blocks, thereby protecting unrelated content before the latent update. Experiments on PIE-Bench and our EditRegion-Bench, with human-verified edit-region annotations for single- and multi-object addition and replacement, show that PC-Edit achieves the best editing quality and background preservation among methods without user-specified edit regions. 

---
# Scaling Up Formal Representation of Clinical Trial Protocols in Ensemble Logic Using LLMs: A Preliminary Study 

**Authors**: Yan Huang, Xubing Hao, Xiaojin Li, Rashmie Abeysinghe, Xiaoqian Jiang, Licong Cui, Guo-Qiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21307)  

**Abstract**: The reliance on unstructured free text for documenting clinical trial protocols creates a significant barrier to automated reasoning, cohort discovery, and trial simulation. The lack of formal structure obscures critical temporal phenotypes, such as dynamic eligibility criteria and event timing constraints. Although Temporal Ensemble Logic (TEL) offers an expressive framework for modeling these elements, manual encoding remains a prohibitive bottleneck. We introduce the CT-TEL workflow: a scalable pipeline leveraging Large Language Models (LLMs) to translate narrative clinical protocols into TEL formulas. We applied CT-TEL to generate logical models for 23 real-world trials from this http URL. We evaluated translation fidelity via a back-translation approach, using LLMs to convert TEL formulas back into natural language and measuring semantic similarity against source texts. The resulting semantic retention suggests that LLMs may offer a pathway for mapping informal protocols to computable logic, providing preliminary evidence toward scalable clinical trial emulation within the emerging "Symbolic Biomedicine" paradigm championed by the corresponding author. 

---
# AI Assistants Overassist 

**Authors**: Verona Teo, Raghav Jain, Tobias Gerstenberg, Max Kleiman-Weiner  

**Link**: [PDF](https://arxiv.org/pdf/2607.21306)  

**Abstract**: Large language models (LLMs) are increasingly used as tutors and thought partners, helping users reason through problems. While guidance from AI assistants can scaffold thinking and foster learning, such benefits depend on how they help--for instance, intervening too early or too frequently may hinder true learning and cognitive engagement. Yet how AI systems navigate intervention decisions during problem-solving remains poorly understood. Here, we introduce Int-Bench, a simulation-based benchmark for evaluating LLM interventions during learning. Int-Bench simulates a "student" solving a problem while a "teacher" monitors the student's reasoning and decides whether, when, and how to intervene. Across three domains--code debugging, mathematics, and brain teasers--we evaluate LLM teachers on the frequency and timing of interventions, as well as their impact on both immediate task success and generalization to new problems. We also compare LLMs to humans, finding that LLMs intervene more frequently and earlier than humans. Moreover, in contrast to humans, they tend to provide complete solutions rather than targeted hints. These findings suggest that current LLM assistants often optimize for short-term success rather than supporting the reasoning processes needed for deeper learning and long-term success. 

---
# Unlearning Under Imbalance: Benchmarking Fairness in Multimodal LLM Unlearning 

**Authors**: Lorenzo Orsingher, Thomas De Min, Massimiliano Mancini, Davide Talon, Elisa Ricci  

**Link**: [PDF](https://arxiv.org/pdf/2607.21300)  

**Abstract**: Machine unlearning has emerged as a tool for removing personal data from trained models to comply with recent AI regulations. To evaluate unlearning effectiveness in multimodal large language models (MLLMs), prior works fine-tune models on fictitious identities, simulating unlearning requests on subsets of these IDs, which are typically uniformly distributed. However, in realistic scenarios, people from different demographic groups may request to be unlearned at different frequencies, potentially altering the model's internal beliefs for these groups and leading to biased behaviors. To fill this gap, we propose FAIRGET, the first Visual Question Answering benchmark that evaluates unlearning under unbalanced, realistic, forget requests. These requests are designed to simulate multiple realistic scenarios, ranging from simple to challenging settings, that lead to biased unlearned models if fairness is not accounted for. Additionally, we propose FAUN, the first unlearning algorithm for MLLMs that forgets unlearning data while preserving model fairness. FAUN exploits a bias-aware activation steering mechanism to unlearn identities while accounting for the unbalanced nature of the forget data. Experiments on FAIRGET and the established FIUBench demonstrate our method's superiority both in unlearning quality and fairness. 

---
# Multi-Task Learning for Heterogeneous Prediction from Video Game State with Transfer Learning 

**Authors**: Jonas Peché, Aliaksei Tsishurou, Alexander Zap, Günter Wallner  

**Link**: [PDF](https://arxiv.org/pdf/2607.21290)  

**Abstract**: Multi-task learning (MTL) is a promising approach for prediction tasks derived from video game state data, as modern game telemetry provides multiple related supervision signals from the same structured observations. We study whether a shared model trained jointly across tasks in team-based multiplayer games can improve generalization while reducing training and inference cost compared to specialized single-task models. We adapt a multimodal architecture for endpoint prediction to a general multi-task setting that combines rasterized vision inputs, global match context, and per-unit state information through an image encoder and attention-based interaction modeling. Experiments on a large proprietary World of Tanks dataset compare single-task and multi-task training, evaluate weighting strategies for mixed losses and conflicting gradients, and test pre-training/fine-tuning under limited target-data regimes. We also examine within-game transfer across game maps under structured environment shift. 

---
# A Comparative Evaluation of Embeddings and LLMs in a Greek Book Publisher Setting - The CUP Dataset 

**Authors**: Katerina Papantoniou, Panagiotis Papadakos, Theodore Patkos, Dimitris Garefalakis, Nikos Vardakis, Dimitris Plexousakis  

**Link**: [PDF](https://arxiv.org/pdf/2607.21274)  

**Abstract**: We present CUP, a Greek book retrieval benchmark consisting of 868 catalog records and 104 expert-annotated queries with graded relevance judgments. We evaluate sparse (BM25), dense (sentence-transformers), hybrid, and LLM-assisted retrieval methods in this book-search setting. Multilingual embeddings outperform Greek-specific models, while hybrid retrieval performs best overall. A query-level analysis shows that BM25 excels at named-entity queries, while dense and hybrid methods improve natural-language, noisy, cross-lingual, and concept queries. Field-aware prompting has model-specific effects, while LLM TOC summarization improves TOC-only retrieval and LLM post-filtering improves early-stage retrieval at a high cost. Overall, CUP enables real-world evaluation of Greek retrieval across lexical, semantic, noisy, and cross-lingual queries. 

---
# pAI-Econ-claude: A Gated Human-in-the-Loop Multi-Agent Architecture for AI-Assisted Economic Theory Development 

**Authors**: Chen Zhu, Xiaolu Wang, Weilong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21268)  

**Abstract**: In many social-science research tasks, such as economics, LLM-based agents must produce outputs for which no cheap, task-complete, machine-readable correctness signal exists. This creates a distinctive reliability problem for multi-agent systems: how should generation, critique, coordination, and human judgment be organized when no component can certify the final result? We address this problem through pAI-Econ-claude, a gated, human-in-the-loop multi-agent architecture for AI-assisted economic theory development. Agents coordinate through a shared workspace of inspectable intermediate records; specialized gates diagnose targeted failure modes and recommend loopbacks without certifying correctness; and human checkpoints retain authority over decisions that are costly to reverse. We evaluate the architecture on five matched economic-theory tasks against an ungated baseline. Two evaluators blinded to configuration agreed on all five pairwise rankings, preferring the gated architecture in four tasks and the baseline in one. Mean failure severity fell from 1.58 to 1.16, while overall usefulness rose from 2.60 to 3.10. The largest observed gain occurred when a reality check rejected a false market-structure premise and a proof review prompted revision of a false welfare claim. The negative case shows that scaffolding can also compress an economically important mechanism too aggressively. The results support a bounded claim: gated oversight improves the auditability of AI-assisted economic theory without substituting for formal verification, and the allocation of irreversible human judgment is a more informative design variable than pure agent autonomy. The workflow is publicly available at this https URL. 

---
# slang.gr as a Large-Scale Crowdsourced Resource for Non-Standard Greek 

**Authors**: Panagiotis Papadakos, Katerina Papantoniou, Dimitris Plexousakis  

**Link**: [PDF](https://arxiv.org/pdf/2607.21255)  

**Abstract**: Slang is a central component of everyday language, reflecting linguistic creativity, social identity, and cultural change, yet its dy- namic and non-standard nature makes it difficult to model computationally. We present the first large-scale computational study of this http URL, a crowdsourced lexicon of Greek non-standard language, combining lexical content, user-generated tags, and interaction data. To enable the systematic analysis, we map noisy folksonomic tags to a structured multi-layer taxonomy capturing both semantic categories and sociolinguistic metadata. Using this representation, we analyze the linguistic structure of Greek slang and the behavior of its contributor community. We find that slang is strongly centered on person-related and evaluative language, exhibits high morphological creativity, and is shaped by highly skewed participation with short user lifespans and overlapping communities. Building on these signals, we introduce a community-based confidence score for definitions that integrates user roles, interaction patterns, and moderation signals. Our results show that taxonomy-based representations improve interpretability while retaining meaningful aspects of behavioral structure, enabling a more structured and interpretable analysis of confidence signals. Overall, this work establishes this http URL as a computational resource for non-standard Greek and provides a foundation for sociolinguistic NLP, bias analysis, and the study of informal language in LLMs. 

---
# Explainable Belief Harmonization under Dynamic Epistemic Partitions 

**Authors**: Adam Kostka, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2607.21210)  

**Abstract**: Existing approaches to multi-agent belief combination have established mature foundations for combining uncertain beliefs under common assumptions: consensus methods use iterative averaging, logic-based methods resolve conflicting knowledge bases, and epistemic logic analyzes agents' information states. Typically, these approaches assume that the structure determining what each agent can represent remains fixed. However, in many scenarios, agents gain or lose observational capacity during execution, and what was once admissible may become structurally impossible. This paper presents a formal framework for handling such runtime changes in epistemic partitions over continuous belief profiles. A hybrid approach exploits the advantages of answer set programming in elaboration tolerance, declarative integrity constraints, and explanations, with the numerical flexibility of Python. The framework applies to domains where agents operate at heterogeneous and possibly changing levels of resolution, and provides formal guarantees of admissibility preservation under refinement, unique mass-preserving repair under coarsening, and explanation completeness. Evaluation across 100 randomly generated topology changes confirms complete violation detection and explanation coverage. 

---
# Explainability Framework for Policy-Aware Autonomous Agents 

**Authors**: Heather Merhout, Daniela Inclezan  

**Link**: [PDF](https://arxiv.org/pdf/2607.21209)  

**Abstract**: In the field of Artificial Intelligence, an agent is a system which is able to autonomously make decisions in order to reach a desired goal. As these systems grow more prevalent in our day-to-day lives, there has been an increased need to add explainability features which can provide an account for an agent's behavior. We therefore propose a framework that outlines how to produce comprehensible explanations for policy-aware agents, or agents which have rule-enforcing policies incorporated in their decision-making framework. This framework is designed using insights from the social sciences on how to produce good explanations. It is implemented in the Answer Set Programming language while using Python to assist with information extraction and natural-language translation. Because these agents incur penalties when violating policies, we are able to leverage these penalties to detect undesirable events in scenarios that are counterfactual to the agents' original actions. This lends itself to creating contrastive explanations (e.g., "the agent performed this action because, had it not, undesirable event X would have occurred."), which formulate the core component for our explainability framework. The framework is evaluated using a survey wherein human participants provide feedback on our program-generated explanations. 

---
# Hybrid MKNF with Classical Negation in the Rule Component 

**Authors**: Arun Raveendran Nair Sheela, Christophe Rey, Florence De Grancey  

**Link**: [PDF](https://arxiv.org/pdf/2607.21202)  

**Abstract**: Hybrid MKNF knowledge bases under the well-founded semantics integrate Description Logics with Logic Programming. However, they do not support classical negation in the rule component, limiting their ability to represent explicit negative knowledge. This limitation is particularly significant in safety-critical applications, where reasoning often requires explicit negative information rather than interpreting the absence of information as evidence of absence. To address this issue, we introduce an extension of Hybrid MKNF that supports classical negation in the rule component. We formally define the syntax and semantics of the extended language and present a general procedure for computing its well-founded model. 

---
# Towards a Certifying Grounder 

**Authors**: Daimy Van Caudenberg, Alexander Ek, Carlos Cantero, Bart Bogaerts  

**Link**: [PDF](https://arxiv.org/pdf/2607.21199)  

**Abstract**: Grounding, the translation of high-level theories into equivalent quantifier-free formulas, is a crucial step in declarative solving, yet it has so far escaped the proof-logging revolution. When this grounding step is not certifying, there is no way of knowing that the obtained solutions actually correspond to the original problem specification, resulting in a trust gap. 
In this paper, we close the trust gap between the user's high-level specification and the solver's low-level input by introducing a novel certifying grounding framework for first-order logic model expansion (FOX) over finite domains. We present CertiFOX, a framework consisting of: (1) a proof format for grounding derivations, (2) GroundFOX, a certifying grounder operating on theories in Grounding Normal Form (GNF)--a new normal form designed for compact, domain-aware grounding--and (3) CheckFOX, an independent proof checker.
Our approach guarantees that the grounder's output is equivalent to the input specification, setting the stage for trustworthy end-to-end certified solving pipelines for declarative languages. Experimental evaluation confirms that CertiFOX is a feasible approach. The GroundFOX grounder is broadly comparable with other grounders, and proof checking with CheckFOX adds overhead within a small constant factor of grounding time. 

---
# Declarative Problem Solving in UAM Strategic Deconfliction 

**Authors**: Gioacchino Sterlicchio, Angelo Oddi, Riccardo Rasconi, Francesca Alessandra Lisi  

**Link**: [PDF](https://arxiv.org/pdf/2607.21197)  

**Abstract**: The growing demand for Urban Air Mobility (UAM) introduces significant challenges in airspace management, particularly within densely populated metropolitan regions. As the number of aerial vehicles-such as drones, air taxis, and helicopters-continues to rise, so does the risk of mid-air collisions and conflicts with existing air traffic and obstacles. Ensuring safe and efficient UAM operations requires robust strategic deconfliction mechanisms. We propose an Answer Set Programming (ASP) based approach for strategic deconfliction, focusing on time synchronization and route optimization for conflict-free flight plans. The solution is benchmarked against Constraint Programming (CP), emphasizing scalability and resource use. Results show that ASP offers faster execution and better scalability for small to medium cases, while CP maintains stable memory but degrades with complexity. 

---
# Case study: solving P-99 with LPTP and an LLM 

**Authors**: Fred Mesnard, Thierry Marianne, Étienne Payet, Wim Vanhoof  

**Link**: [PDF](https://arxiv.org/pdf/2607.21196)  

**Abstract**: Ninety-Nine Prolog Problems (P-99) is a famous set of Prolog exercises. We solved the first thirty three just by prompting an LLM (Large Language Model). We used Claude from Anthropic. By solved we mean: generate the Prolog code and a test file, run the tests and check whether they pass, then formally prove types, groundness, termination, uniqueness, existence and also sometimes functional correctness with LPTP (Logic Program Theorem Prover). Hence our approach is an experiment in vibe-coding/vericoding of P-99. It is a vibe-coding experiment because we started from informal specifications written in English and let Claude generate the Prolog code. It also fits within vericoding because the LLM proved reliability guarantees on the generated Prolog code. Claude wrote 58 logic procedures, 508 tests, 257 lemmas for a total of 11800 proof lines. We manually checked each file generated by the LLM. We checked the Prolog code, ran the tests, examined the logical statements generated by Claude and proof-checked Claude's proofs with LPTP. This paper describes this experiment and provides the main details so that it can be reproduced by the interested reader. 

---
# Chess\_db: A framework for working with large chess game datasets 

**Authors**: Nicos Angelopoulos, Jan Wielemaker  

**Link**: [PDF](https://arxiv.org/pdf/2607.21195)  

**Abstract**: Chess is a two player strategic game that is embedded in classical AI  culture as it was once the frontier for intelligent behaviour.  There was the silent assumption that the advent of computer engines that  play better than the best humans will extinguish interest in the game. However, the opposite has come to pass, with a growing following for the game. A lot of the computational resources are now centered around training of players, where the engine output is just one aspect. Access to  past games is also an essential part, both in knowing what games a specific player has played previously, and also which continuations at a certain position have led to victory more often for each of the two colour players. We present Chess_db a suite of logic programming tools that can effectively manipulate games both in memory and via creating back end databases.  In particular, we provide versatile code that creates databases from PGN (portable game notation) game files and explore the suitability of open source key-value databases for  storing position tables that provide near-instant access to  information pertaining to substantially large number of games. 

---
# Animation, Verification and Visualisation of Prolog Transition Systems with ProB 

**Authors**: Jan Gruteser, Michael Leuschel, Katharina Engels, Fabian Vu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21192)  

**Abstract**: ProB is a Prolog-based model checker, animator and constraint solver for high-level formal specifications. One can also use ProB to animate transition systems defined by Prolog predicates, allowing the application of its various validation techniques. In this work, we present the existing features of ProB's Prolog animation mode and its recent extensions. The extended capabilities include simulation for statistical checks, more reliable trace replay, transitions with user input and improved state visualisation. We apply the new features to case studies, particularly for evaluating different strategies in game play, such as Connect Four. The features are useful for many other applications, especially for ProB's new sequent prover for Event-B proof obligations, as well as for demonstration models for teaching in combination with interactive visualisation. 

---
# Encoding Event-B Proof Rules in Prolog: An Interactive Sequent Prover for ProB 

**Authors**: Katharina Engels, Jan Gruteser, Michael Leuschel  

**Link**: [PDF](https://arxiv.org/pdf/2607.21191)  

**Abstract**: Event-B is a formal method rooted in predicate logic and set theory. We encoded over 600 proof rules in Prolog, enabling a systematic, comprehensible proof analysis and construction. By integrating the proof rules into the Prolog-based validation tool ProB, we obtain an interactive proof system with proof tree visualisation. This has advantages in teaching, giving students direct control over the selection of proof rules. Our tool can import proof obligations from the Rodin platform and provides multiple exports: a trace file for proof replay in ProB, an interactive HTML document for tool-independent exploration of the proof tree, and an export back to Rodin, allowing the ProB prover to be used as second chain. Compared to the previous implementation of the proof rules in Java, the encoding in Prolog is more compact, maintainable and extensible. While a preliminary iterative deepening prover with simple heuristics is already available and useful for finding short proofs, we aim to obtain fast automatic provers in the future. 

---
# Case study: proving sqrt(2) irrational with LPTP and an LLM 

**Authors**: Fred Mesnard, Étienne Payet, Wim Vanhoof  

**Link**: [PDF](https://arxiv.org/pdf/2607.21187)  

**Abstract**: We present the interactions with an LLM (Large Language Model) aiming at proving that the square root of 2 is not a rational number in an LP (Logic Programming) context. We start from a few basic pure logic programming predicate definitions. We rely on the LPTP  (Logic Program Theorem Prover) system for stating and proving properties about logic programs.  As the proof language of LPTP  is based on natural deduction,  the proofs are human readable. In our case study, we sketch in LPTP the usual proof showing the irrationality of the square root of 2.  Then we describe the interactions we had with the LLM. We end up with a complete formal proof, partially generated by an LLM and fully proof-checked by LPTP. 

---
# Representative Sets in Propositional Abduction 

**Authors**: Johannes Schmidt, Mohamed Maizia, Victor Lagerkvist, Johannes K. Fichte  

**Link**: [PDF](https://arxiv.org/pdf/2607.21183)  

**Abstract**: The propositional abduction problem is a well-known form of non-monotonic reasoning where we are asked to find an explanation of a given manifestation. Recently, there has been an influx of results asking more refined questions about the solution space rather than only individual solutions. For example, we might be interested in finding two solutions that are sufficiently far from each other (diverse solutions) in the solution space. In this paper we consider a related representation question where we ask if a given set of explanations S can represent any other explanation (that is, whether their symmetric difference is smaller than a given k). We first study this problem from a classical complexity perspective and obtain a complete classification. While only a handful of cases are tractable, the increase in complexity compared to classical abduction is often smaller than expected. We then study the parameterized complexity for several parameters and obtain new tractable and hard cases. Interestingly, a full parameterized complexity classification would require resolving the parameterized complexity of the covering radius problem from coding theory. To the best of our knowledge, no useful relationship between coding theory and non-monotonic reasoning has previously been established, but such connections seemingly become important when asking more complex questions about solution spaces. 

---
# CRAG-MM-Diagnostics: Enabling Stage-Wise Analysis of Knowledge-Intensive VQA 

**Authors**: Hanseok Oh, Parishad BehnamGhader, Benno Krojer, Hyunji Lee, Paul Liang, Siva Reddy, Verna Dankers  

**Link**: [PDF](https://arxiv.org/pdf/2607.21155)  

**Abstract**: Knowledge-Intensive Visual Question Answering (KI-VQA) benchmarks evaluate Vision-Language Models (VLMs) as multimodal knowledge assistants by requiring external information beyond a provided image to answer questions. KI-VQA involves multiple sub-problems -referring expression understanding, visual grounding, object recognition, knowledge retrieval, and reasoning-yet existing benchmarks typically report only end-task accuracy, obscuring where failures arise. To analyze the full KI-VQA pipeline, we introduce CRAG-MM-Diagnostics, a diagnostic benchmark with stage-wise data annotations that isolate 1) language-based visual grounding, 2) object identification, and 3) knowledge retrieval and reasoning. We evaluate fully parametric and retrieval-augmented VLMs, providing fine-grained analyses using newly collected metadata, such as target ROIs, entity names, and visual complexity scores. Our results point to knowledge retrieval and reasoning as the primary bottleneck, but also highlight issues in the other parts of the KI-VQA pipeline, such as the fact that VLMs struggle with target object identification or that image retrievers struggle to integrate textual cues. These findings expose fundamental limitations in current KI-VQA systems and motivate stage-aware evaluation. We, lastly, leverage these findings to propose a grounded bimodal RAG pipeline that integrates a visual grounding module to crop targets before image retrieval, boosting GPT-5 and Qwen's respective accuracies by 13.3 and 8.5 percentage points. 

---
# One More Turn, Less Regret: A Regret-Based Multi-Turn Benchmark for LLMs' Clarification Policies 

**Authors**: Minh Ngoc Ta, My Anh Tran Nguyen, Duong D. Nguyen, Yuxia Wang, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2607.21143)  

**Abstract**: Ambiguous user requests make clarification a sequential decision problem for conversational LLM assistants: they must decide whether to ask, what to ask, when to stop, and when to answer. We introduce RegretBench, a multi-turn benchmark that evaluates clarification as policy behavior rather than isolated question quality. RegretBench provides a hidden-intent formulation of ambiguity, supports free-form interaction grounded in semantic-state tracking, and introduces a regret-based objective that measures how much value a model loses relative to a reference clarification policy. Experiments on open-domain QA and product recommendation scenarios show that final success alone is insufficient, as models with similar accuracy can differ substantially in efficiency, robustness to user behaviors, and stopping decisions. By jointly measuring intent resolution, interaction cost, ineffective clarification, and regret, RegretBench reveals whether models clarify usefully and efficiently. Our results show that effective clarification requires more than plausible questions: models must ask the right question at the right time and stop once the user's intended meaning is clear. 

---
# Demographically-Informed Heat-Mortality Risk Curves via Risk Graph Neural Networks 

**Authors**: Alex O. Davies, Eunice Lo, Rui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21131)  

**Abstract**: Estimating heat-related mortality risk is a core task in environmental epidemiology, typically addressed with Distributed Lag Non-linear Models (DLNMs); interpretable exposure-response surfaces fitted to temperature-mortality time series. DLNMs are effective but ignore demographic and geographic context, despite well-established relevance to heat vulnerability. We propose Risk Graph Neural Networks (RGNNs), a hierarchical GNN encoder that uses granular census features to optimise DLNM coefficient vectors, preserving interpretable risk curve outputs while substantially improving predictive calibration. Evaluated across 10 regions of England and Wales on two unprecedented heat years, RGNN variants maintain both lower point-errors and near-nominal uncertainty coverage during the 2022 heatwave where baselines collapse. 

---
# Hardware-Software Co-Design for Float16 On-Device Training on RISC-V Single-Core 

**Authors**: Benjamin Hubinet, Pierre-Alain Moellic, Olivier Savry, Olivier Potin, Jean-Baptiste Rigaud  

**Link**: [PDF](https://arxiv.org/pdf/2607.21130)  

**Abstract**: By leveraging standard RISC-V extensions, namely Zfh (scalar float16) and Zvfh (vector float16), this work proposes an open-source framework to enable complete on-device training on resource-constrained RISC-V single-core. Our approach allows memory footprint reduction by about 50% as compared to using float32 and with minimal model performance degradation. We also facilitate transfer learning and fine-tuning scenarios by incorporating layer-freezing capabilities. Our work builds onto AIfES, an open-source, modular and generic DNN training and inference framework for embedded systems that can be extended with custom hardware-specific functions. The benefits of float16 is further emphasized by outlining the low area overhead of Zfh on a RV64GC super-scalar out-of-order FPGA softcore (+1.15% LUT6 and +0.05% FF at 175MHz). Finally, we discuss the architecture of a Zvfh implementation within the same RISC-V core. 

---
# Relative Value Learning 

**Authors**: Marc Höftmann, Jan Robine, Stefan Harmeling  

**Link**: [PDF](https://arxiv.org/pdf/2607.21120)  

**Abstract**: In reinforcement learning, critics typically estimate absolute state values $V(s)$, estimating how good a particular situation is in isolation. However, it turns out that only differences in value are relevant for control. Motivated by this, we propose Relative Value Learning (RV), a framework that learns value differences directly via an antisymmetric function $\Delta(s_i, s_j) = V(s_i) - V(s_j)$. We introduce a pairwise Bellman operator and prove it is a $\gamma$-contraction with a unique fixed point equal to the true value differences, derive well-posed $1$-step, $n$-step and $\lambda$-return targets and reconstruct generalized advantage estimation from pairwise differences to obtain an unbiased policy-gradient estimator (R-GAE). Beyond theoretical results, we integrate RV with PPO and achieve competitive performance on the Atari benchmark (49 ALE games) compared to standard PPO, indicating that relative value estimation is an effective alternative to absolute critics. 

---
# GlucoTune: A Unified Framework for Blood Glucose Preprocessing, Forecasting, and Benchmarking in Diabetes 

**Authors**: Davide Marelli, Giorgia Rigamonti, Mirko Paolo Barbato, Paolo Napoletano  

**Link**: [PDF](https://arxiv.org/pdf/2607.21117)  

**Abstract**: Preprocessing blood glucose time-series data is a critical yet often overlooked step in developing data-driven methods for diabetes management, particularly for type 1 diabetes. The lack of standardized preprocessing workflows and evaluation protocols hinders reproducibility and complicates fair comparison across studies. These challenges are further exacerbated by data-sharing restrictions, as privacy and licensing constraints often prevent the redistribution of preprocessed medical datasets. To address these limitations, we present GlucoTune, a comprehensive and extensible framework for reproducible experimentation with blood glucose time-series data. The framework standardizes the entire experimental workflow, from preprocessing to model evaluation, enabling reproducible experiments directly from the original datasets. Reproducible preprocessing is achieved through configurable pipelines defined in portable YAML configuration files, ensuring consistent data handling without distributing sensitive preprocessed data. Beyond preprocessing, GlucoTune provides a unified interface for implementing, training, and evaluating blood glucose prediction models. The framework integrates public datasets through standardized wrappers and provides a curated collection of state-of-the-art blood glucose prediction and general time-series forecasting methods, while remaining readily extensible to additional datasets, preprocessing strategies, and forecasting models. To promote transparent and consistent evaluation, GlucoTune includes a benchmarking leaderboard that reports results across datasets, preprocessing configurations, and forecasting methods, enabling systematic comparison of experimental settings. We demonstrate the effectiveness of GlucoTune through comprehensive experiments and assess its usability in a user study. 

---
# TOUR: A Trajectory-Level Unlearning Benchmark for Offline Reinforcement Learning 

**Authors**: Chaofan Pan, Lingfei Ren, Xiangyu Jiang, Yanhua Li, Xuemei Cao, Xiangkun Wang, Hao Yu, Wei Wei, Xin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21111)  

**Abstract**: Offline Reinforcement Learning (RL) agents are trained on fixed behavioral trajectories, which makes trajectory-level deletion important when selected data must be removed after training. Evaluating such deletion is difficult because a lower membership score can reflect trajectory removal, residual memorization visible to another attack, or policy collapse that destroys useful behavior. We introduce Trajectory-level memOrization and Unlearning in offline RL (TOUR), a benchmark that combines trajectory-level partitioning, matched non-member controls, retraining references, retained-performance anchors, and multi-attack privacy auditing. Across D4RL locomotion experiments and an exploratory AntMaze extension, TOUR shows that common deletion baselines have environment-dependent privacy-utility behavior. Retraining and fine-tuning often provide stronger retained-utility references than uniform GA+Refit, while TrajDeleter remains a useful comparator but is not uniformly stronger under the same audit. Reference-model, threshold, deviation, equivalence, action-error, representation-based, and query-limited attacks further show that a single likelihood-based membership score can overstate deletion quality. In the evaluated settings, conclusions about offline RL unlearning are therefore not stable under single-score auditing. They depend on matched non-member construction, retraining-relative calibration, attack family, retained utility, and explicit scope for diagnostic architecture or component-level evidence. 

---
# Training Large Language Models for Self-Explanation Faithfulness 

**Authors**: Yeoktatt Cheah, María Pérez-Ortiz, Noah Y. Siegel, Oana-Maria Camburu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21090)  

**Abstract**: We propose a Reinforcement Learning (RL) method to directly optimize the faithfulness of self-explanations - the extent to which a model's generated reasoning accurately reflects its internal decision-making process. While existing work focuses on evaluating faithfulness or using inference-time prompting frameworks to improve an LLM's self-explanation's tractability, these approaches do not provide a mechanism to directly optimize a model's parameters to generate faithful self-explanations. We bridge this gap by modifying existing faithfulness metrics into an RL training objective. We investigate (1) if models can be trained to accurately detect factors that affect their decisions, and (2) whether RL can directly optimize for the disclosure of these factors thereby improving LLM self-explanations' faithfulness. We experiment with two intervention types: random-word insertions and user-bias insertions, using a per-sample reward derived from the Phi-CCT correlation metric. RL fine-tuned Llama3.1-8B and Qwen3-8B show substantial improvements on the Phi-CCT faithfulness metric, with in-distribution scores rising from near-zero to as high as 0.664, and out-of-distribution scores reaching up to 0.691 on held-out tasks such as StrategyQA. Cross-intervention generalization is weaker but more interesting: a priori we would not expect a model trained only on random word insertions to generalize to user-bias phrases, yet Llama3.1-8B shows non-zero transfer in this direction. The reverse direction and Qwen3-8B do not replicate this, indicating model-dependent and setup-dependent effects we cannot yet explain. Lastly we analyze model behavior to rule out reward gaming behaviors that often plague RL training. Ultimately, we show that models can be trained to implicitly identify influential factors and disclose them, offering a scalable path toward reducing unfaithful reasoning in LLMs. 

---
# Sparse Concept Channels in Frozen 3D CT Vision Encoders 

**Authors**: Farhad Nooralahzadeh, Lea Bogensperger, Christian Bluethgen, Michael Krauthammer  

**Link**: [PDF](https://arxiv.org/pdf/2607.20993)  

**Abstract**: Large vision-language models are becoming increasingly dominant in 3D medical image interpretation, but we rarely know <i>which</i> internal units encode clinical findings or <i>where</i> that information lives in the representation. We first study this on a 3D chest vision-language model (Pillar-0) by probing its frozen vision embeddings. We show that (i) each radiological finding is encoded by a <i>sparse</i> set of ~10 vision-encoder channels that match full-feature classification performance and far exceed a zero-shot text prompting; (ii) turning off the channels tied to one finding, that finding's score collapses while unrelated labels stay stable; and (iii) the same sparse probe <i>replicates</i> on an architecturally unrelated 3D abdominal VLM (Merlin) suggesting a general property of frozen medical encoders. Our training-free concept channel probe (CCP) method, paired with a corpus-derived report template, outperforms published CT-CHAT on clinical efficacy and NLG metrics (F1 0.549 vs. 0.184; BLEU 0.483 vs. 0.373) at 22x lower latency. Our results provide a clear, reproducible characterization of how frozen medical encoders represent findings, demonstrating direct applicability across models. 

---
# HyWorldVLA: A Vision-Language-Action Model with Hybrid World Modeling for Autonomous Driving 

**Authors**: Quanfu Yu, Xian Wu, Hao Xu, Liulong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2607.20988)  

**Abstract**: Vision-Language-Action (VLA) models augmented with world modeling represent a promising paradigm for end-to-end autonomous driving. While pixel-level future prediction enables fine-grained spatiotemporal reasoning, it compromises robustness in noisy driving scenarios. Conversely, latent-based world models alleviate this sensitivity but often incur limited interpretability and representational degradation due to absent pixel-level grounding. To reconcile this trade-off, we propose HyWorldVLA, a hybrid world-VLA framework that unifies pixel-level supervision and latent representation learning. In the pre-training stage, HyWorldVLA predicts video latents encoded by a pre-trained video VAE, while simultaneously reconstructing video frames to provide precise pixel-level grounding. During the subsequent co-fine-tuning phase, the model exclusively predicts latent features, which are fed into an action expert to generate trajectories. Extensive experiments on NAVSIM v1 and v2 benchmarks demonstrate that HyWorldVLA significantly outperforms both pixel-based and latent-based world model baselines. Notably, we present the first comprehensive qualitative and quantitative analysis of world model noise robustness in autonomous driving, establishing a new benchmark for evaluating future architectures. 

---
# Interaction Dynamics Modeling and Predictive Control for Safe Steerable Catheter--Tissue Interaction 

**Authors**: Yongyan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20939)  

**Abstract**: Safe steerable catheter control is fundamentally a problem of interaction dynamics: the tip must follow a planned motion, remain compliant against moving tissue, reject friction and hysteresis, and respect a clinically meaningful never-exceed contact-force bound. We formulate catheter--tissue interaction dynamics in the scalar tip-normal coordinate of a single-segment single-tendon catheter. A partial-physics feedforward cancels only the reliable nominal bending dynamics, exposing a configuration-invariant linear interaction-dynamics model whose input gain varies through the scalar catheter inertia. A predictive optimizer then regulates this interaction state subject to hard contact-force, tendon-force, and curvature constraints. An augmented Kalman filter compresses contact, friction, and modeling error into one sensor-free disturbance state, giving nominal offset-free regulation in free space while leaving force safety to the explicit constraint. The unconstrained and disturbance-free limit recovers classical catheter impedance as a special realization of the same interaction dynamics, rather than as the main design object. In a MuJoCo distributed-compliance simulation of an eight-link tendon-driven catheter, disturbance augmentation cuts free-space approach error by 90\%, and only the force-constrained predictive interaction-dynamics controller reconciles tracking with the 0.5\,N bound: the unconstrained controller drives contact force to 0.60\,N against a penetrating target, while the constrained one holds 0.47\,N at identical tracking. These results show that offset-free motion regulation and contact-force safety are coupled interaction-dynamics objectives, and that the explicit predictive constraint resolves their tension under stiff tissue contact. The bound also holds under $0.5$\,mm, $1.2$\,Hz cardiac motion. Hardware validation is future work. 

---
# Scientific exploration, collaboration and labor division in the large language model era 

**Authors**: Xiang Zheng, Xi Hong, Jialin Liu, Chaoqun Ni  

**Link**: [PDF](https://arxiv.org/pdf/2607.20923)  

**Abstract**: Large language models (LLMs) have rapidly and significantly entered scientific workflows, but it remains unclear how their diffusion is associated with changes in scientists' strategies in research directions and team building. We link PubMed Central full text with OpenAlex publication and collaboration histories for 775,323 scientists and analyze CRediT contribution statements from 137,120 multi-author papers. After 2022, scientists increasingly published across more intellectually distant fields and entered fields in which they had not previously worked. These increases in interdisciplinarity and exploration were especially pronounced among established scientists and scientists from non-English-speaking low- and middle-income countries. Authors with stronger AI-writing signals were already more interdisciplinary and exploratory before the widespread adoption of LLMs, and the gap widened further after 2022 compared with authors with weaker AI-writing signals. Scientists' collaboration networks also became more interdisciplinary after 2022. Yet, among authors with stronger AI-writing signals, research interdisciplinarity was less closely tied to the disciplinary diversity of their collaborators. The division of labor within research teams also became more differentiated. Contributors on papers published after 2022 reported narrower role sets on average, coauthors shared fewer roles in common, and their role profiles became less rigid and more fluid. Software and validation roles increased, while conceptual and management roles decreased. These patterns suggest that team members are taking on more distinct responsibilities and may rely less on one another to perform research tasks. Overall, this study indicates that the LLM era coincides with a broader reorganization of scientific exploration, collaboration, and the division of labor. 

---
# Multi-turn RL with Structural and Performance Aware Rewards for CUDA Kernel Generation 

**Authors**: Quazi Ishtiaque Mahmud, Nesreen K. Ahmed, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2607.20908)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful technique to enhance the reasoning capacity of LLMs for optimized code generation. However, existing RLVR approaches primarily rely on outcome-based signals such as correctness and speedup, overlooking performance-critical structural properties of programs that are essential for generating optimized code. In this work, we propose CudaPerf, a reflective RL framework that incorporates both verifiable execution rewards and structural code-aware rewards derived from parallelization features (e.g., memory coalescing, occupancy, Arithmatic Intensity, and synchronization patterns). CudaPerf operates in two stages: (1) an offline pairwise ranking module that learns to distinguish strong and weak program candidates via contrastive comparisons, and (2) an online RL training phase that jointly optimizes for correctness, performance, and structural efficiency through a unified reward signal. To further enhance learning, CudaPerf utilizes iterative refinement using execution feedback enabling progressive improvement of generated candidates. We also introduce a dataset comprising 2.9k C to CUDA and 1k PyTorch to CUDA programs, each paired with diverse input configurations and multiple CUDA implementations encompassing diverse optimization strategies. CudaPerf is evaluated across multiple benchmarks comprising both C to CUDA and PyTorch to CUDA transformations. Empirical findings suggest that CudaPerf significantly outperforms strong baselines, including Qwen-3-32B (for C to CUDA) and CUDA Agent (for PyTorch to CUDA) by achieving up to 5X & 3.32X improvements in speedup, and 17% & 7% improvements in correctness, respectively. 

---
# Anti-Goal Reasoning: Rethinking the Theory of Goal Reasoning in Non-Axiomatic Logic 

**Authors**: Bowen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20902)  

**Abstract**: Goal reasoning in Non-Axiomatic Logic (NAL) explains how an adaptive system derives means for realizing desired events under insufficient knowledge and resources. However, the representation of avoidance is less clear. A common convention is to express ``avoid $G$'' as the goal sentence ``$\neg G!$'', but this notation conflates two different readings: pursuing the negated event $\neg G$, and avoiding the positive event $G$. This paper shows that the conflation can produce a paradoxical case in which an avoidance intention is converted into a positive goal to act merely because acting is usually followed by the absence of hurt. Starting from NAL's basic definition of goals, the framework is extended with a corresponding definition of anti-goals, so that avoidance can be represented without treating it as the pursuit of a negated event. Finally, a mental operation, $\op{prevent}$, is introduced to connect anti-goal reasoning with ordinary goal reasoning in cases of active prevention. Four minimal case studies check that the resulting rules distinguish pursuit, passive avoidance, active prevention, and withholding action to preserve a desired event. 

---
# TwistedMerge: Certified Higher-Order Diagnostics and Abstention for Model Merging 

**Authors**: Ting Gong, Shitan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20887)  

**Abstract**: Model merging combines independently trained or fine-tuned models, but pairwise alignability does not imply globally consistent alignment. We formulate merging as a finite descent problem in which checkpoints are local objects, alignment maps are transitions, and cycle products are residuals. TwistedMerge is a conservative certification pipeline that separates fixed-chart averaging, synchronization-removable gauge inconsistency, a certified central obstruction on a specified comparison complex, and nonabelian holonomy. A residual is promoted to a cohomology class only after inverse-consistency, coefficient-identification, centrality, and closure tests; otherwise the method abstains and returns an ordinary or synchronized fallback. We prove a constant-edge no-go result, frozen-complex three-way and predeclared-family error-control theorems, and a refinement test for comparison-complex sensitivity. A planted neural alignment defect is removed by cycle-consistent synchronization, showing that a nonzero cycle score alone is not a higher obstruction. Controlled central systems recover the predicted non-coboundary and projective-rank behavior, while noisy estimates move from certification to abstention without false lifts on the tested controls. A trained low-rank-adapter audit shows that naive factor averaging depends on the chosen GLr representative, whereas global factor synchronization and dense-delta SVD are stable. On natural checkpoint collections, cycle residuals do not predict merge degradation and no natural central or period-index class is certified. The results position descent theory as a falsifiable certification and abstention framework. 

---
# Probabilistic Residual Learning for Online Recommendations 

**Authors**: Wenyuan Wang, Yusong Zhao, Zihao Xu, Hengyi Wang, Qi Xu, Zhigang Hua, Yan Xie, Yi Wang, Zihao Zhao, Bo Long, Chengzhi Mao, Shuang Yang, Hengguan Huang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20863)  

**Abstract**: Modern recommender systems are typically based on deep learning (DL) models, where a dense encoder learns representations of users and items. As a result, these systems often suffer from the black-box nature and computational complexity of the underlying models, making it difficult to systematically enhance their recommendation capabilities. To address this problem, we propose Probabilistic Residual Learning (PRL), a causal Bayesian recommendation model that models the residual between ground-truth and base predictions, enabling targeted refinement of existing systems. Specifically, PRL (1) probabilistically groups users for localized residual modeling, (2) models domain-level confounders that influence user and item representations, and (3) aggregates cluster-specific residual predictions over the confounders using do-calculus. Experiments demonstrate that our plug-and-play PRL is compatible with various base deep learning recommender systems, improving their performance while automatically discovering meaningful user clusters. 

---
# Multilevel Graph Wavelet Compressed Sensing with Scale-Aware Neural Recovery 

**Authors**: Amirhossein Nouranizadeh, Sarang Rajendra Patil, Alan John Varghese, Varsha Narayanan, Amit Chakraborty, Mengjia Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20857)  

**Abstract**: Scientific machine learning methods such as neural operators and physics-informed neural networks have advanced engineering applications and inverse problems, but their training typically requires large volumes of simulated data. This makes data preparation and model training expensive. We propose Graph Wavelet Compressed Sensing (GWCS), a learning-based framework for offline compression of graph signals by representing them as sparse, interpretable wavelet-domain representations using the spectral graph wavelet transform. The framework combines a nonparametric multilevel importance sampler, which retains high-energy wavelet coefficients within each scale for a given compression ratio, with a scale-aware graph neural network that reconstructs the signal from the sparse coefficients. We evaluate the proposed framework on synthetic approximately band-limited graph signals over random graphs and four PDE simulation datasets over meshes, which include Turbulent Radiative Layer, Viscoelastic Instability, Kolmogorov Flow, and Dynamic Stall. We compare against graph signal sampling methods and graph autoencoder baselines. Results demonstrate that the framework achieves high reconstruction fidelity and substantial data compression compared to existing benchmarks. 

---
# Beyond Heavy Log Curation: Perplexity-Based APT Detection via Unsupervised, Context-Augmented Language Models 

**Authors**: Shoya Otsu, Kei Suzuki, Toshiaki Koike-Akino, Jing Liu, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20832)  

**Abstract**: Advanced Persistent Threats (APTs) remain difficult to detect because only a small fraction of events in large-scale logs are attack-related, and investigation is expensive and hard to scale. Prior machine-learning approaches can reduce analyst workload, but they often rely on heavily curated training data and sophisticated preprocessing pipelines. Building and maintaining such pipelines require substantial domain expertise and engineering cost. Motivated by insights from a study of a strong APT detection baseline, we propose CAPTAIN (Context-Augmented Perplexity-based Threat Activity log detectIoN), a perplexity-based detector that leverages general, pre-trained language models with minimal, domain-agnostic preprocessing, enabling robust scoring of long, minimally processed log entries. CAPTAIN encodes recent history with an encoder model and a Q-Former-style bridge, then injects the compact context tokens into the decoder input so that perplexity reflects temporal context. To improve stability, CAPTAIN additionally applies smoothing filters to the perplexity time series. Across APT-oriented benchmarks, CAPTAIN competes with strong existing baselines and remains robust under substantially less curated inputs, that reduces the development and operational cost of advanced log preprocessing. 

---
# The Geometry of Personality: Activation Steering with Jungian Cognitive Functions 

**Authors**: Liu Zai, Yumeng Wang, Junchen Fu, Joemon M. Jose  

**Link**: [PDF](https://arxiv.org/pdf/2607.20803)  

**Abstract**: Activation steering enables control and interpretation of LLMs, yet existing work primarily models personality through static trait frameworks such as the Big Five. We investigate whether personality can instead be represented and controlled as a set of cognitive processes using the eight Jungian Cognitive Functions. To this end, we introduce a framework comprising a Jungian evaluation protocol and a dataset of over 2,100 role-playing character narrations.
Activation steering vector extraction and evaluation experiments on Llama-3.1-8B demonstrate effective monotonic control over all eight cognitive functions through activation steering. Beyond controllability, our analysis reveals that: 1. personality information is concentrated in middle transformer layers; 2. steering vectors exhibit structured geometric relationships consistent with distinctions between rational and irrational functions; 3. effective multi-dimensional steering directions cannot be recovered as linear combinations of single-function directions. These findings provide new insights into the representation of personality in LLM activation space and establish a framework for studying interpretable, effective, and multi-dimensional personality control. 

---
# Synthetic minority data is redundant or invalid: a data-dependent validity theory and a de-biased test 

**Authors**: Ahmad B. Hassanat, Ahmad S. Tarawneh, Ghada A. Altarawneh  

**Link**: [PDF](https://arxiv.org/pdf/2607.20787)  

**Abstract**: For two decades, the standard remedy for class-imbalanced learning has been to fabricate synthetic minority examples, and the standard evidence of their validity has been a check that cannot fail: synthetic points are scored against the very data that generated them. We de-bias the check. Validity becomes a population quantity -- the probability that a synthetic point truly belongs to the minority class -- with a consistent estimator that scores synthetic points against withheld real data. Where held-out ground truth is available, the classical test underestimates true invalidity in 96-99% of method-by-imbalance-ratio cells, while the de-biased estimator tracks it closely. We prove validity is a property of the data, not the method: class overlap sets an invalidity floor no faithful generator escapes, making oversampling redundant where classes separate and invalid where they overlap. Across 91 methods, three classifiers, and datasets spanning medicine and finance -- including a generator engineered to pass the classical check -- none clears both bars: gains over the best trivial baseline are noise-thin (median below 0.01 F1, a decision threshold's reach), and most damage calibration. We release the audit as a pip-installable test and flip the burden of proof: synthetic minority data must now demonstrate, on the data at hand, both validity and information gain. 

---
# Robostral Navigate 

**Authors**: Arjun Majumdar, Avinash Sooriyarachchi, Benjamin Tibi, Chris Bamford, Elliot Chane-Sane, Guillaume Lample, Khyathi Raghavi Chandu, Ludovic Ho Fuh, Mathieu Poiree, Olivier Duchenne, Rosalie Millner, Srijan Mishra, Theo Cachet, Thomas Chabal  

**Link**: [PDF](https://arxiv.org/pdf/2607.20785)  

**Abstract**: Deploying navigation systems at scale requires a recipe that minimizes sensor assumptions, generalizes across robot embodiments, and trains efficiently. Yet, today's best systems depend on depth sensors, multi-camera rigs, or pre-built maps, limiting the hardware they support and increasing deployment cost. We introduce Robostral Navigate, an 8B vision-language model built around this scalability objective. The model consumes only a stream of monocular RGB images - the most ubiquitous sensor across robotic platforms and predicts waypoints by pointing to the next target location in the current camera view. Operating purely in image space, rather than robot-specific coordinates, makes the policy naturally robust to changes in camera intrinsics and scene scale, enabling deployment across wheeled, legged, and aerial robots without recalibration. We generate 2.4 million trajectories across 350k simulated scenes to reduce the reliance on real-world data collection and scale easily. We further introduce a prefix-caching training recipe that packs entire episodes into single training sequences, reducing training tokens by 22x and cutting training time from months to days. A tree-based attention mask prevents conditioning on previous ground-truth actions, encouraging visually grounded action prediction, and reinforcement learning is used to further improve exploration and recovery capabilities. On the Room-to-Room and Room-Across-Room in Continuous Environments (R2R-CE and RxR-CE) benchmarks, Robostral Navigate sets a new state of the art. On R2R-CE, it achieves a 77.4% success rate, surpassing the best monocular method by 10.5 points and the strongest depth- or multi-camera system by 5.3 points despite using only a single RGB camera. On RxR-CE, it reaches 75.1% success rate, outperforming all monocular baselines. 

---
# HARP: The Human--AI Research Platform 

**Authors**: Zeshu Zhu, Natalie Friedman, Kevin Weatherwax, Emily Eiben  

**Link**: [PDF](https://arxiv.org/pdf/2607.20773)  

**Abstract**: Large language models (LLMs) have shifted human--computer interaction from `traditional'' interface journeys toward more conversational exchanges. Researchers studying HCI and UI use moderated usability sessions, interviews, surveys, transcript analysis, and static prototypes. However, static prototypes provide limited opportunities to study interaction with live AI systems or systematically control how an LLM behaves across participants and scenarios. Conversation transcripts reveal little about how users formulate, revise, and hesitate over prompts before submission. We designed the Human--AI Research Platform (HARP) for researchers, designers, and anyone who has ever wondered, `What if AI did this?' HARP places participants in controlled mock scenarios with live, configurable AI agents. Researchers can control agent prompts, model parameters, response characteristics, and experimental conditions; trigger surveys at predefined moments; and record prompt composition time, response latency, deletions, and keystroke pauses. Planned capabilities include voice, facial expression, gesture, and, where legally and ethically appropriate, emotion analysis. We illustrate HARP through a study examining how technical specificity and response length affect retention of LLM output. By pairing controllable live agents with behavioral and self-report measures, HARP enables systematic testing of how AI design choices affect users. 

---
# Emergent Compositional Skills in Mixture-of-Experts VLAs 

**Authors**: Shlok Shah, Rhiaan Jhaveri, Tharun Kumar Tiruppali Kalidoss, Chirayu Nimonkar, Ishaan Javali  

**Link**: [PDF](https://arxiv.org/pdf/2607.20771)  

**Abstract**: We consider the problem of learning compositional robot policies end-to-end from expert demonstrations, without any pre-specified notion of task decomposition or hierarchy. We ask whether a VLA trained with a simplified Mixture-of-Experts (MoE) action head can emergently learn to decompose tasks into reusable, interpretable primitives. We find that learned experts are heavily reused across tasks and consistently correspond to qualitatively distinct low-level behaviors, suggesting that the router implicitly learns to perform high-level sequencing while experts serve as compositional primitives. Our MoE matches the task performance of a monolithic baseline while demonstrating meaningful expert specialization, a step toward modular, interpretable robot policies that emerge from data alone. 

---
# Are Diversity Metrics Measuring Diversity? A Capability-Controlled Audit of Majority-Vote Gain in LLM Ensembles 

**Authors**: Donghwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.20768)  

**Abstract**: Majority voting over LLMs is widely assumed to benefit from diversity, and diversity measures are used to choose which models to combine. We ask whether five such measures track diversity or mainly re-express capability, auditing them as predictors of majority-vote gain over the best member across 31,900 subsets of 30 LLMs on MMLU-Pro (29 on TruthfulQA) under explicit capability controls. Three findings emerge. First, latent complementarity is ubiquitous: oracle gain is positive in 100% of subsets, yet simple voting beats the strongest member in only 9.98% of all canonical size-3 subsets (18.71% with held-out best selection); the pooled size-2-4 rate is 1.27%, partly reflecting deterministic even-size voting behavior. Second, a joint-correctness proxy (strict diversity) is nearly collinear with one minus mean accuracy (size-3 Spearman rho = +0.991 / +0.988); raw diversity-gain associations are strongly capability-entangled and, with one exception, unstable under control. Third, three linear contingency-table statistics are algebraically non-separable; after capability control, the empirically stable remainder is a modest residual pairwise co-failure association in which more shared error corresponds to lower gain. This direction is robust, but its magnitude is configuration-dependent. Joint rawspace linear regressions treating strict diversity, disagreement, and double-fault as independent predictors are rank-deficient by construction. 

---
# IssueTrojanBench: Benchmarking AI Coding Agents Against Malicious Issue Requests 

**Authors**: Ankur Singh, Jinqiu Yang, Tse-Hsun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.20759)  

**Abstract**: AI coding agents powered by LLMs are increasingly integrated into real-world software development, where they generate, edit, and execute code with autonomous access to local files and tools. Coding agents inherit security risks from both the LLM backbone, where adversarial prompts, poisoned training data, and backdoor triggers can cause models to emit insecure or attacker-chosen code, and their agentic architecture, where tool-using autonomy enables induced misuse of external APIs, data exfiltration, and persistent compromise of development environments. This paper presents a systematic evaluation of malicious issue requests against state-of-the-art coding agents (Cursor, Claude Code, and Codex Desktop), powered by two major model families (OpenAI GPT-5.3 Codex/GPT-5.4 and Anthropic Sonnet 4.6). Our novel benchmark IssueTrojanBench contains malicious issues that are constructed based on four novel attack categories (i.e., embedded as malicious instructions in issues), six delivery vectors (e.g., PDF, or issue comment), and further augmented by perturbations. Our results reveal critical vulnerabilities in the as-deployed modern coding agents, i.e., 66.5% of the malicious issues from IssueTrojanBench penetrate all the guardrails (agent- and LLM-level) of coding agents. Our further analysis shows that rejection is almost entirely from LLMs rather than the agent frameworks, with GPT models broadly vulnerable and Sonnet 4.6 exhibiting more selective, risk-aware blocking of high-impact actions. Our evaluation also highlights that the current agent-level defense strategy offers limited additional protection for coding agents. Our findings highlight the urgent need for stronger agent- and model-level safety mechanisms to protect AI coding agents. 

---
# Self-Supervised Bio-Inspired Robotic Trajectory Planning with Obstacle Avoidance 

**Authors**: Miroslav Krupa, Miroslav Cibula, Kristína Malinovská  

**Link**: [PDF](https://arxiv.org/pdf/2607.20743)  

**Abstract**: Trajectory planning is a fundamental problem in robotics, requiring the generation of collision-free and efficient trajectories in a potentially complex environment. While sampling-based planners remain the dominant approach, they are often computationally expensive, particularly in high-dimensional spaces and obstacle-rich environments. Methods based on model learning offer a promising alternative, enabling efficient planning through a bounded number of forward passes through a neural trajectory planner, but commonly suffer from low sample efficiency or limited generalisation due to their reliance on exploration or expert demonstrations. This follow-up work tests our neuro-inspired self-supervised learning framework for trajectory planning that leverages forward and inverse models as the internal supervisory mechanism in an environment that contains an obstacle. Experimental results demonstrate the feasibility of the approach while revealing a tendency of our planner to exploit the learning signal provided by the forward and inverse models. To address this issue, additional training regimes and mitigation strategies are proposed and evaluated. 

---
# GPE: Evaluating Robust Evidence Aggregation for Fact Verification under Controllable GEO-Style Poisoning 

**Authors**: Zhaoqi Wang, Zijian Zhang, Xiaomei Yuan, Pengtao Kou, Jiamou Liu, Zhen Li, Liehuang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20730)  

**Abstract**: Large language models increasingly use search tools to retrieve up-to-date information, introducing a new attack surface in which retrieved documents can be manipulated. This risk is amplified by the development of generative engine optimization, which can make selected content more likely to be retrieved, cited, and adopted by models. Existing fact-verification benchmarks and evaluation frameworks do not provide the controlled evidence environments needed to assess robustness against GEO poisoning. We therefore propose GPE, which consists of a multi-domain fact-verification benchmark and an evaluation framework for controlling evidence sources and poisoning ratios. Experiments across multiple verification methods and poisoning attacks demonstrate that GPE exposes robustness degradation and efficiency trade-offs that cannot be observed through clean evaluation alone, confirming the need to evaluate fact verification under adversarial evidence environments. 

---
# Operational Identity: A Finite Audit of Declared and Implemented Rules of Sameness 

**Authors**: Denise M. Case  

**Link**: [PDF](https://arxiv.org/pdf/2607.20729)  

**Abstract**: A record system declares when two records refer to the same entity, occurrence, scope, or rule. Its disclosed implementation mechanisms induce a corresponding operational identity relation. The declared and implemented relations may diverge systematically without producing a provenance gap or detectable contradiction. A system can apply, consistently and with every record individually correct, a rule of sameness that no artifact declares. This paper formalizes that implemented relation. A declared identity regime partitions a finite record domain into co-reference classes; a disclosed mechanism, through its typed identity-relevant outcomes, induces an operational identity partition of the same domain. The audit compares these partitions in the refinement lattice. A mechanism is faithful when the declared partition refines the operational partition, so no declared class is split. A divergence witness is a pair the declaration merges and the mechanism separates; such witnesses are decidable by pair enumeration. When an imported sibling basis also splits a declared class, local comparison with its partition yields sibling-aligned, sub-sibling, super-sibling, or sibling-incomparable divergence. This result reports only the relationship; it does not identify the basis carried by the mechanism. Global equality of the operational and sibling partitions is defined separately as regime substitution and does not follow from sibling alignment. A version field incremented on every textual edit inhabits the sub-sibling case by splitting declared classes more finely than either imported basis. The audit is three-valued and relative to the disclosed artifacts, evaluated surfaces, and identified uses; each boundary has a finite refuting witness. A passing verdict is non-monotone because extending the transformation history can merge declared classes and create a witness among records already examined. 

---
# Transition-Related Potentials as Markers of Narrative Comprehension in Continuous EEG 

**Authors**: Bálint Csanády, Péter Vedres, Kristóf Zsolt Makó, Orsolya Papp-Zipernovszky, Márta Volosin, Dávid Apagyi, András Lukács, András Bálint Kovács, Zoltan Nadasdy  

**Link**: [PDF](https://arxiv.org/pdf/2607.20720)  

**Abstract**: Harnessing the potential of electroencephalography (EEG) for brain research is fundamentally limited by intrinsic noise and the diffuse projection of brain-generated activity over the scalp. The standard event-related potential (ERP) paradigm addresses this limitation by relying on repeated independent trials, albeit at the cost of moving away from naturalistic experimental conditions. As a more naturalistic alternative, we collected continuous EEG while participants watched short films and extracted potentials aligned to sharp cinematic transitions (cuts). We demonstrate that such transition-related potentials (TRPs) exhibit canonical ERP-like temporal structure associated with significant information processing. By comparing coherent films with scene-scrambled versions containing matched post-cut sensory input, we find that these responses are systematically shaped by narrative context. We then show that the cut-related EEG signature can be recovered directly from group-averaged continuous recordings with a compact deep neural network (DNN). The detector generalized across films and subject groups, and the resulting TRPs reproduced the main context-dependent effects observed for manually annotated cuts. These results indicate that narrative context leaves a measurable signature in EEG responses, that this signature can be detected directly in continuous recordings, and that such detections provide a semi-automated framework for analyzing how viewers process and understand film narratives. We propose that the method outlined here can be adapted to parse EEG responses to other forms of continuous stimulation, providing a general tool for probing experimental conditions that are closer to natural human experience. 

---
# U-CFR: Uncertainty-Guided Cascade Forward Refinement for Interactive Segmentation 

**Authors**: Elijah Danquah Darko, Min Xian, Terence Soule, Tiankai Yao, Matthew William Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2607.20705)  

**Abstract**: Interactive image segmentation is critical for efficient image annotation; however, existing methods often require many corrective clicks or rely on passive refinement schemes that converge slowly. We propose Uncertainty-Guided Cascade Forward Refinement (U-CFR), a novel inference-time framework that enables models to autonomously self-correct after each user interaction. U-CFR introduces a boundary-aware uncertainty score that fuses segmentation uncertainty, contour gradients, and explicit edge predictions to guide the placement of internal pseudo-clicks. These self-generated clicks target the most ambiguous boundary regions, providing strong corrective signals without additional manual input. To support this process, we design a dual-head network with a shared encoder-decoder backbone: a segmentation head ensures region consistency, while an edge head sharpens boundary alignment. In inference, U-CFR launches a cascade of refinement steps, where each stage leverages the uncertainty-driven pseudo-clicks to refine the mask progressively. Experiments on standard benchmark datasets demonstrate that the proposed U-CFR improves click efficiency, initial mask quality, and boundary accuracy. It reduces the required clicks by over 10% on challenging datasets like Berkeley and offers a more intelligent and efficient interactive annotation. 

---
# A Framework for Reputation Aware Uninorm-driven Consensus Algorithms for Blockchain Networks 

**Authors**: Bruno Ramos-Cruz, Javier Andreu-Perez, David Richerby, Luis Martínez  

**Link**: [PDF](https://arxiv.org/pdf/2607.20700)  

**Abstract**: The operation of blockchain is governed by consensus algorithms (CA). Several consensus mechanisms require significant computational power, while others necessitate high amounts of stakes to select the participant to validate and verify the transactions in the block, leading to centralisation of power and participant exclusion. This paper proposes a novel methodology to address these issues in reputation-based consensus algorithms by studying the reputation behaviour of the validator using intuitionistic fuzzy sets (IFSs) and uninorm aggregation operations (UAOs). Our approach uses IFSs to express the "reputation" because the reputation values in a consensus algorithm eventually imply uncertainty, and IFSs facilitate the representation of a lack of precise knowledge about reputation. Moreover, this methodology utilises uninorm aggregation operations to monitor reputation over time and reinforces the importance of negative and positive reputation. Consequently, this solution allows validators to rectify past failures in subsequent verification processes and foster an equitable consensus algorithm design. The proposed framework maintains linear computational complexity and does not introduce additional communication overhead beyond the underlying consensus protocol. Supported by experimental results, our methodology demonstrates improved performance and evaluation, promising advancements in blockchain network fairness and inclusivity. 

---
# DS@GT ARC at ImageCLEFmed GANs 2026: Geometric Filtering for Privacy-Preserving CT Slice Generation 

**Authors**: Eric Regina, Richard Arnaud, Samir Hadi Cisneros  

**Link**: [PDF](https://arxiv.org/pdf/2607.20692)  

**Abstract**: We present a privacy-preserving framework for synthetic lung CT slice generation developed for the Image-CLEFmed GANs 2026 challenge. The approach combines Optimal Transport Conditional Flow Matching with privacy-oriented training and a post-generation "Supervisor" pipeline that filters generated candidates in learned geometric latent spaces using autoencoder embeddings, Determinantal Point Processes, and Stein Kernel Thinning. Official results show a strong realism-privacy trade-off, with the best-performing model achieving a Privacy Preservation Score of 0.549 and competitive visual fidelity with an FID of 0.3290. While the proposed geometric filtering substantially reduces nearest-neighbor memorization and membership-inference leakage, persistent patient re-identification scores indicate that preventing direct image copying is not sufficient to remove deeper patient-specific anatomical identity, highlighting an important frontier for future privacy-preserving medical image generation. 

---
# Spatially Grounded Concept Bottleneck Models for Trustworthy Breast Ultrasound Diagnosis 

**Authors**: Moshiur Rahman Tonmoy, Dunren Che, Haitham Y. Adarbah, Afzel Noore  

**Link**: [PDF](https://arxiv.org/pdf/2607.20691)  

**Abstract**: Concept Bottleneck Models provide interpretable-by-design predictions by mediating diagnosis through human-understandable concepts, but in medical imaging, their trustworthiness is often limited by the quality and granularity of available supervision. In particular, predicted concept activations can be driven by irrelevant regions, leading to spatially unfaithful explanations. We study a data-centric spatially grounded Concept Bottleneck Model (SG-CBM) that leverages coarse lesion delineations as weak supervision to encourage anatomically plausible concept evidence. For breast ultrasound, we derive two clinically motivated zones from each lesion mask: (i) an in-lesion region of interest for morphology-related concepts and (ii) a posterior acoustic band for posterior phenomena. We train concept maps using a grouped spatial grounding objective and preserve semantic faithfulness with a linear bottleneck classifier. Across five-fold stratified group cross-validation, the proposed SG-CBM improves diagnostic AUROC and concept macro-AUROC while markedly increasing spatial alignment of concept evidence. We also perform a Train-corrupt/Test-clean annotation-quality stress test to quantify the impact of supervision quality on diagnosis and spatial faithfulness. Overall, the results underscore the need for data-quality-aware supervision design and systematic trustworthiness validation for deployable healthcare AI systems. 

---
# From Agent Failures to Text Policies: What Works and What Breaks 

**Authors**: Jaideep Ray, Ankit Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2607.20668)  

**Abstract**: TextGrad improves language-model systems by revising text from feedback. Its core thesis is that natural-language feedback can act as a gradient for optimizing text components without changing model weights. Applying it to agents is harder because feedback arrives only after a sequence of actions, making it difficult to identify which decision caused failure. We study this problem by separating the ability to follow a useful policy from the ability to learn that policy from experience. Our main finding is a clear gap between these two abilities. Human-written policies improve two frozen 7B agents on TextWorldExpress by 5.0 success points, showing that useful policy text exists. However, policies generated from agent trajectories do not reliably outperform fixed prompting, even with richer traces, counterfactual evidence, or iterative GEPA search. The main challenge for agent-level TextGrad is therefore not executing textual policy updates, but reliably generating and selecting them from experience. 

---
# Adaptive Multi-Horizon Reinforcement Learning 

**Authors**: Manoosh Samiei, Doina Precup, Paul Masset  

**Link**: [PDF](https://arxiv.org/pdf/2607.20656)  

**Abstract**: Effective decision-making in complex and changing environments requires balancing short-term and long-term consequences. In reinforcement learning (RL), this trade-off is typically controlled through a fixed discount factor, which imposes a single exponentially discounted temporal horizon. However, biological agents exhibit flexible and adaptive temporal discounting, suggesting that effective planning requires multiple timescales. Here, we propose a multi-horizon approach that adaptively selects and combines temporal horizons, enabling robust adaptation to changes in reward structure without manual discount-factor tuning. This flexibility makes the method particularly suitable for continual learning scenarios involving task switches and varying environmental configurations. Empirically, we demonstrate that our approach identifies effective discount factors across a range of MiniGrid environments, including continual settings composed of three sequentially changing tasks. These results suggest that adaptive temporal discounting can improve parameter efficiency and enhance adaptability in both artificial and biologically inspired learning systems. 

---
# SalesLoop: Reinforcement Learning from Performance Feedback for Sales Lead Ranking 

**Authors**: Chenyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20655)  

**Abstract**: Lead ranking in Customer Relationship Management (CRM) systems faces a persistent challenge: models achieving high offline accuracy often underperform in production. We identify three fundamental gaps responsible for this disconnect: offline-online metric mismatch, pointwise-listwise objective misalignment, and temporal distribution drift. To address these gaps, we propose SalesLoop, a reinforcement learning framework that establishes a closed feedback loop between model predictions and real-world business outcomes. Our approach introduces (1) a performance-aware reward that encodes conversion outcomes weighted by ranking position and conversion velocity, and (2) Discriminative GRPO, a listwise optimization objective that adapts Group Relative Policy Optimization to discriminative ranking models.
SalesLoop improves NDCG@K by +7.9\% and P@K by +15.8\% over the strongest static baseline. A 160-day production A/B test at a New Energy Vehicle manufacturer, spanning 16.5M leads and 280 sales specialists across two provincial markets, validates statistically significant cumulative lift of +4.7\% ($p=0.047$) and +8.7\% ($p=0.002$). In production, the ranking backbone achieves Top-10\% recall of 44.1\% and surfaces high-intent leads at $2.3\times$ the conversion rate of specialist baselines. 

---
# Scaling Interpretable Transformers with Parity Bottleneck Layers 

**Authors**: Andrew Mack, Kraig Yuheng Tou, Mark Henry, Zhengxun Wu, Lauren Greenspan  

**Link**: [PDF](https://arxiv.org/pdf/2607.20652)  

**Abstract**: Language models are thought to exhibit the phenomenon of superposition, representing many more features than dimensions in their residual streams. Sparse autoencoders (SAEs) are designed to recover such features post-hoc, but training models that are interpretable by construction has remained impractical, as a per-layer over-complete bottleneck is prohibitively expensive in both memory and compute. To overcome this issue, we introduce the ParityTransformer, a GPT-2-scale architecture whose intermediate representations are efficient and wide / sparse by design. At each layer, a Deep Parity Bottleneck (DPB) replaces a learned over-complete basis with a parameter-free algebraic dictionary, providing a deterministic incoherence guarantee and eliminating the memory requirements that have prevented per-layer interpretable bottlenecks at scale. A DPB is a hierarchically structured sparse bottleneck which efficiently enforces sparsity using a multi-level mixture-of-experts approach: a hardware-aware implementation that closes the cost gap between activation sparse and dense training to a manageable interpretability tax. Empirically, ParityTransformers perform at least as well as post-hoc SAEs on sparse probing tasks, while out-performing on measures of feature absorption, steering effectiveness, and fine-grained causal interventions. Because subsequent computation acts only on features that survive the sparse bottleneck, the ParityTransformer's features are native to the model's forwards pass by construction, addressing the question of whether SAEs probe features the model actually uses during computation. We see this as a step toward training models whose internal representations are interpretable by design rather than recovered post hoc. 

---
# Frontier Financial Judgement: Can agents tell what might move a stock? 

**Authors**: Joshua Harris  

**Link**: [PDF](https://arxiv.org/pdf/2607.20645)  

**Abstract**: We introduce Frontier Financial Judgement, a challenging new benchmark developed in collaboration with professional equity analysts to assess agents' ability to replicate expert human judgements. Rapidly identifying new information, evaluating its implications and determining its valuation impact is one of the most time-consuming and challenging aspects of real-world equity coverage. This is becoming ever more difficult and important as AI rapidly increases the quantity of new information to process. The strongest agent we evaluate on Frontier Financial Judgement matches all expert labels in only 52.4% of cases. We also find significant divergence in estimated false-positive rates among frontier agents, ranging from ~1% for GPT-5.6 Sol to ~32% for Claude Sonnet 4.6. To construct the benchmark and make it representative of real-world settings, we combine human-designed and labelled synthetic articles with live news articles and historical documents, creating 656 items for assessment. The resulting task requires agents to distinguish genuinely new, valuation-relevant financial information from stale, immaterial or misleading news under realistic conditions. We find substantial trade-offs among agent accuracy, cost, false positives and reliability that continue to hinder the reliable deployment of news-flow filtering in practice. 

---
# Demonstrating GenDB: Instance-Optimized and Customized Query Processing Code Generation via LLM Agents 

**Authors**: Jiale Lao, Immanuel Trummer  

**Link**: [PDF](https://arxiv.org/pdf/2607.20630)  

**Abstract**: Traditional query processing engines require continuous development and extensions to support new techniques and user requirements, and in some cases, entirely new systems must be built from scratch. However, these engines are difficult to extend due to their internal complexity, and building new systems demands significant engineering effort and cost. To address this, we demonstrate GenDB, a generative query engine that shifts query processing from manually engineered systems to query processing code generation driven by Large Language Models (LLMs). An early prototype of GenDB uses LLM agents to generate instance-optimized query execution code tailored to specific data, workloads, and hardware resources. This prototype suits offline code generation for repetitive, templated queries, since the upfront generation cost amortizes over many executions and correctness can be ensured through extensive fuzz testing and manual inspection. For ad-hoc queries, GenDB can work with a traditional DBMS in a hybrid architecture: the DBMS handles one-off queries, while GenDB speeds up frequent SQL templates. Our demonstration allows users to (1) visually and interactively explore how GenDB analyzes workloads, profiles hardware resources and underlying data, produces query plans, generates code based on them, and finally uses an optimizer to iteratively achieve a correct and efficient implementation; (2) use visual inspection and analysis to gain qualitative insights into why GenDB produces code that achieves significantly better performance than state-of-the-art query engines on two benchmarks: TPC-H and a newly constructed benchmark designed to reduce potential data leakage from LLM training data; and (3) upload their own data and queries to explore GenDB with different LLMs and query patterns. 

---
# RealVDeblur: One-Step Diffusion for Generalizable Real-World Video Deblurring 

**Authors**: Renbiao Jin, Mingxin Yang, Yutian Chen, Junhao Zhuang, Xin Cai, Mulin Yu, Linning Xu, Wenxian Yu, Danping Zou, Shi Guo, Tianfan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2607.20628)  

**Abstract**: Real-world video deblurring remains challenging due to diverse motion patterns, complex degradations, and the scarcity of realistic training data, yet robust restoration is critical for downstream pipelines such as mobile imaging and 3D reconstruction. This work presents \textbf{RealVDeblur}, an efficient generative framework designed to improve in-the-wild robustness under diverse real capture conditions. First, a large-scale, physically grounded blur synthesis pipeline is constructed from scene-level 3D Gaussian Splatting (3DGS) assets and high-frame-rate videos, providing realistic training data covering both camera-induced and object-motion blur. Second, a video diffusion prior is leveraged for restoration; to better accommodate frame-dependent blur variations, temporal compression in the VAE is disabled and a frame-wise encoding scheme is adopted. For practical deployment on long videos, multi-step diffusion sampling is distilled into an efficient one-step generator, and a training-free Temporal Window Mask stabilizes inference beyond the training horizon with constant memory usage. Extensive experiments on diverse real-world benchmarks demonstrate strong perceptual quality, semantic fidelity, and temporal consistency on unseen videos, as well as improved robustness in downstream 3D reconstruction under severe motion blur. Project page: this https URL 

---
# When Does Recurrence Become an Algorithm? Convergence Selection in Weight-Tied Looped Transformers 

**Authors**: Tong Zhang, Junhao Hu, Yun Peng, Tao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.20594)  

**Abstract**: When does a weight-tied looped transformer -- one block applied T times -- implement an actual algorithm? We answer with four findings from controlled populations on group word problems. (1) The budget law: free training installs a linear computation frontier, a mechanism that solves v positions per loop, whose speed is priced by the training contract: v ~ n_train/T_train (exponent 0.98 +/- 0.04, R^2=0.99), exactly unity under T=n training. SGD selects a frontier matching the minimum the contract demands; granting more test-time loops than ever trained rescues late positions at fixed input length, yielding a principled halting rule T* = ceil(n / v-hat). (2) Architecture prior, not expressivity, picks the algorithm: standard-depth transformers learn parallel scans on this family; weight tying flips the selection to the serial frontier, even when positional addressing for a log-depth scan is supplied. At matched depth and parameters, untied models extrapolate worst and fail to learn A5 at all. (3) The walls are not where circuit complexity says: NC1-completeness costs nothing (A5 generalizes fully), while group order does (S5's 120x120 operator deadlocks joint learning) -- and an operator-first curriculum dissolves the wall in every seed. (4) Mechanisms are portable, not mandatable: warm-starting across budget contracts transfers the algorithm in every seed, re-pricing its speed, while imposing seriality through the input schedule fails where free training succeeds. These results are invisible to standard instruments, which provably saturate at the fixed points trained loops converge to. We introduce a head instrument, the convergence-time scaling tau(n,i), validate it causally via damage cones whose slope reproduces v, and show in-distribution head measurements predict out-of-distribution fate where tail metrics do not. Results replicate on the public easy-to-hard benchmark. 

---
# Foundation-model-guided radiogenomic discovery linking cancer genomes to cancer scans 

**Authors**: Frederik Hauke, Jeremias Krause, Patrick Wienholt, Christiane Kuhl, Ingo Kurth, Sikander Hayat, Jakob Nikolas Kather, Sven Nebelung, Daniel Truhn  

**Link**: [PDF](https://arxiv.org/pdf/2607.20583)  

**Abstract**: The function of many genes is still unknown, and conventional driver-discovery methods, which rely on how frequently a gene is mutated, cannot assess genes that are only rarely affected. Here we pair Evo~2-based genome analysis with routine clinical imaging to identify gene--phenotype associations at genome-wide scale. For every somatic mutation across three TCGA cohorts (cRCC=clear cell renal cell carcinoma, HCC=hepatocellular carcinoma, and BC=breast cancer; $n = 340$ total), Evo~2 predicts a severity score, with no task-specific training. Per-gene severity summaries are then correlated with radiomic features extracted from paired tumor segmentations, controlling for total mutation burden. In TCGA-cRCC ($n = 162$), this sweep recovers established renal-cancer drivers and identifies 46 additional genes reaching false discovery rate (FDR) significance absent from curated cancer-gene panels, several of which are Mendelian ciliopathy and cytoskeletal-disease genes. These results demonstrate that pairing a genomic language model with widely available clinical imaging can serve as a hypothesis-free discovery tool for gene--imaging associations invisible to conventional approaches. 

---
# Bayesian uncertainty estimation improves clinical decision making in medical AI agents 

**Authors**: Frederik Hauke, Patrick Wienholt, Christiane Kuhl, Dyke Ferber, Jakob Nikolas Kather, Sven Nebelung, Daniel Truhn  

**Link**: [PDF](https://arxiv.org/pdf/2607.20582)  

**Abstract**: Machine learning models for medical image analysis typically lack a reliable measure of confidence, limiting their use in ambiguous or atypical cases. Here we show that Monte Carlo dropout, applied to a multi-task chest-radiograph classifier (eight thoracic findings, 137,593 training images), provides an epistemic uncertainty signal that tracks generalisation across training-set scales and flags confident yet error-prone predictions. Adding this signal to the point prediction raised error-detection AUROC from 0.74 to 0.77 ($\Delta$AUROC +0.023, 95% CI [+0.014, +0.033]). In a controlled 2x2 factorial experiment, a clinical-decision-support agent exploited this uncertainty only when it was delivered as a binary error-risk flag rather than as raw scores, cutting confident misdiagnoses on unreliable findings from 8.5% to 2.7%. Epistemic uncertainty estimation thus carries decision-relevant information beyond point predictions, but its value for downstream agents depends on how it is communicated. 

---
# Geometric Configurations of Perturbed Jailbreak Prompts 

**Authors**: Lynn Delcon, Andres Algaba, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2607.20581)  

**Abstract**: Perturbation techniques that turn unsuccessful jailbreak prompts into successful ones are continuously evolving, constituting a major security threat to LLM safety. In this paper, we investigate the internal representations of such string-level perturbed jailbreak inputs in the small weight models of the Qwen-2.5-1.5B/-3B/-7B-Instruct and Llama-3.2-1B/-3B/-3.1-8B-Instruct families. We select two representation spaces: the last-layer-last-token embedding space and the top-50 next-token probability space. The former space separates prompts based on their spelling and format, while the latter space is effectively one-dimensional but appears more complex to cluster. Within our refusal-dominated answer set we find no behavioral hyperplane in either space. Only the next token "Sure" in the 1.5B Qwen model, and both tokens "," and "ĊĊ" in the 1$ Llama model, display a significant association with a compliant-labeled answer. 

---
# Joint Utilization of Geospatial and census proxies for Autoencoder-Assisted Downscaling (JUGAAD) of socioeconomic indicators in India 

**Authors**: Aditya Dutt, Paul Gader, Aditya Singh  

**Link**: [PDF](https://arxiv.org/pdf/2607.20559)  

**Abstract**: Monitoring poverty and food security indicators is imperative for addressing socioeconomic challenges in developing nations. A limitation is mismatches in scale between data sources: census data provide geographic coverage, while socioeconomic indicators are derived from infrequently conducted surveys at coarse resolutions, posing a methodological challenge. This study introduces a deep learning framework, JuGAAD, using Indian census and survey data from 2001 and 2011 as a case study. We employ a three-step process: census and geospatial data are averaged into intermediate village-cluster-scale tessellations to reduce noise and regularize administrative boundary changes; an autoencoder compresses high-dimensional National Sample Survey Office (NSSO) data into a low-dimensional latent representation; and a regression model maps upscaled census and geospatial data to this representation. This function is applied to fine-grained census data to generate high-resolution predictions, validated against ground-truth district-level NSSO indicators. Results confirm the methodology predicts socioeconomic indicators at fine scales with strong accuracy. 

---
# StabilityBench: Benchmarking Instability in LLMs 

**Authors**: Emma Kondrup, Zachary Yang, Anne Imouza, Reihaneh Rabbany  

**Link**: [PDF](https://arxiv.org/pdf/2607.20558)  

**Abstract**: AI Assistants are increasingly deployed in high-stakes settings, such as healthcare or government services. Yet their real-world behavior remains poorly understood due to strong context dependence. Current evaluation protocols follow a defense-in-depth paradigm with compounding layers of safeguards, ranging from traditional benchmarks to live or adversarial testing. Such benchmarks remain largely static and single-turn, limiting their ability to capture real-world variability in conversational settings. We propose StabilityBench, a principled, general and model-agnostic benchmark operator that transforms single-turn benchmark queries into multi-turn interaction histories. StabilityBench augments existing benchmarks by injecting realistic user simulations, through demographic proxies or sycophantic baits, while preserving original task intent. We apply StabilityBench to four benchmarks spanning mathematical reasoning, health question-answering and safety, and evaluate nine large language models under these conditions. Our results show that model performance is consistently unstable under these injections, with considerable performance degradations on three out of four benchmarks studied. These highlight important limitations of static evaluations and motivate more realistic evaluation settings. To this end, we propose StabilityBench-Mini: a size-preserving variant of StabilityBench that samples across diversification axes, enabling more realistic evaluation without increasing costs. 

---
# Monkey King Bang: A Unified Scientific Multimodal Foundation Model 

**Authors**: Hesen Chen, Xinyu Su, Xiaomeng Yang, Yuetan Lin, Zixiong Yang, Junyi An, Fenglei Cao, Yifeng Jiao, Yunqi Zhang, Yuan Cheng, Zhiyu Tan, Hao Li, Libo Wu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2607.20557)  

**Abstract**: Scientific discovery is increasingly shifting from isolated disciplines to multi-domain reasoning, and AI for science faces a similar transition. Existing systems are either specialised for individual domains or unify scientific data mainly through text tokenisation and prompt-based interfaces, limiting their ability to handle diverse scientific inputs, produce modality-native outputs, and support joint understanding, reasoning, and generation across scientific domains. We introduce MKB, a unified scientific multimodal model for both understanding and generation, built around a shared Transformer backbone and modality-tailored encoders, adapters, and decoders. MKB covers six scientific branches, including DNA, RNA, proteins, small molecules, earth science, and medical images, and supports native outputs such as biological sequences, molecular strings, meteorological fields, and segmentation masks. Training follows a two-stage modality-then-language curriculum: Stage 1 aligns modality-specific components with the frozen backbone, and Stage 2 consolidates them with the language backbone using mixed scientific and general corpora. Experiments show that MKB achieves competitive scientific understanding across biological and molecular benchmarks, produces high-fidelity native outputs for weather forecasting, biological generation, and medical-image segmentation, and largely retains the general capabilities of its Qwen3-VL backbone. These results demonstrate the feasibility of the proposed paradigm, suggesting that shared-backbone models with modality-tailored components can provide a promising foundation for future cross-domain scientific multimodal exploration. The model and code are publicly available at this https URL and this https URL. 

---
# SenCos-GEM: SENet-Calibrated and Law-of-Cosines-Constrained Geometry-Enhanced Molecular Representation for Property Prediction 

**Authors**: Tianming Han, Li Zhang, Qi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20551)  

**Abstract**: Effective molecular representation learning is crucial for accurate molecular property prediction. Recently, numerous self-supervised learning (SSL) approaches leveraging 3D GNNs have been developed to capture comprehensive 3D structural information for drug discovery. However, existing methods lack explicit physical constraints and are highly susceptible to geometric noise induced by coarse empirical force fields during large-scale this http URL, they overlook dynamic feature modulation during downstream adaptation, often resulting in catastrophic forgetting and negative transfer. To address these limitations, we introduce SenCos-GEM, a novel explicitly decoupled geometry-enhanced molecular representation learning framework that incorporates SENet-calibrated and law-of-cosines-constrained enhancements. SenCos-GEM employs a physics-guided geometric consistency loss based on the law of cosines to derive high-fidelity and mathematically invariant 3D spatial priors. In addition, lightweight Squeeze-and-Excitation (SE) modules are integrated into the backbone as task-specific adapters, while a dual-modulation prediction head combines Feature-wise Linear Modulation (FiLM) and SENet mechanisms to enable dynamic feature recalibration. SenCos-GEM demonstrates highly competitive performance across diverse classification and regression tasks on MoleculeNet benchmark, establishing new state-of-the-art results specifically on 3D conformation-sensitive regression tasks, such as FreeSolv, Lipophilicity, and QM9, achieving relative error reductions of 12.9% (RMSE), 5.3% (RMSE), and 8.2% (MAE), respectively. Moreover, our model exhibits superior capability in distinguishing stereoisomers and discriminating conformational perturbations, underscoring its robust spatial modeling performance. Collectively, SenCos-GEM represents a significant breakthrough in accurate molecular property prediction. 

---
# Beyond SBDD: Geometric Deep Learning in Polypharmacology and Multi-target Drug Design 

**Authors**: Tianming Han, Zhijie Pan, Wenchi Ge, Qi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20550)  

**Abstract**: The traditional "one drug, one target" paradigm of structure-based drug design (SBDD) frequently proves inadequate for treating multifactorial diseases such as cancer and neurodegenerative disorders, owing to compensatory signaling pathways and the emergence of drug resistance. While polypharmacology offers a synergistic therapeutic strategy, the rational design of ligands capable of simultaneously satisfying the geometric constraints imposed by multiple targets remains a major computational bottleneck. This review positions geometric deep learning (GDL) as a powerful integrative approach to overcome these limitations. We systematically survey GDL architectures ranging from invariant graph neural networks to SE(3)-equivariant diffusion models that harness non-Euclidean molecular data to capture intrinsic three-dimensional (3D) structural interdependencies. We critically analyze GDL applications across three core dimensions, including the characterization of shared binding pockets via geometric embeddings, multi-target bioactivity prediction through heterogeneous graph fusion, and de novo generation of dual-target ligands. Particular emphasis is placed on emerging structure-conditioned generative algorithms that integrate diffusion models with reinforcement learning to autonomously resolve complex geometric conflicts between competing binding sites. Furthermore, we evaluate the pivotal role of multimodal omics integration and specialized geometric benchmarking infrastructures in validating these models. By synthesizing these methodological advances, this review elucidates the paradigm shift in drug discovery from serendipitous exploration to rational, structure-driven polypharmacological molecular engineering, thereby providing a clear, structured guide for navigating the complexities of next-generation therapeutics. 

---
# SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales 

**Authors**: Mikail Khona, Aditya Vavre, Boxiang Wang, Deyu Fu, Hao Wu, Mike Chrzanowski, Bryan Catanzaro, Dheevatsa Mudigere, Jeff Pool, Michael Lightstone, Mohammad Shoeybi, Mostofa Patwary, Nima Tajbakhsh, Tijmen Blankevoort  

**Link**: [PDF](https://arxiv.org/pdf/2607.20548)  

**Abstract**: Higher-order optimizers such as Muon and SOAP offer faster convergence than AdamW, but their computational cost and numerical stability challenges have limited adoption at scale. In this work, we adapt and enhance preconditioned gradient methods to overcome the practical challenges of large-scale LLM pretraining. We first identify instabilities in SOAP at large batch sizes and propose algorithmic modifications including per-step QR orthogonalization and improved preconditioning strategies that eliminate loss spikes and enable stable training in these regimes. We then present a unified empirical study of SOAP, Muon, and AdamW using update-RMS matching to ensure fair learning rate transfer across optimizers. As part of this analysis, we empirically evaluate the orthogonalization quality of Muon. Our experiments on multi-billion-parameter models trained on trillions of tokens reveal that SOAP and Muon consistently outperform AdamW at the scales we tested. Notably, at batch sizes of up to 100M tokens for next-token prediction, these optimizers maintain training stability and quality while AdamW degrades. To enable efficient training at large scale, we introduce a layer-wise distributed optimizer compatible with Megatron-LM. Our implementation balances memory and hides communication while avoiding approximations to the optimizer computations, thus retaining their convergence benefits. Additionally, we identify and build specific system-level improvements to further accelerate our layer-wise implementation. To support the research community, we release a codebase that contains emerging algorithms for optimization: this https URL 

---
# When RLVR Shrinks the Reasoning Boundary: Diagnosing Pass@k Inversion 

**Authors**: Todd Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.20543)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) can improve one-sample accuracy while making a model worse under repeated sampling. We study this pass@k inversion: after training, the policy may solve fewer distinct problems than its base model at large $k$. The failure concentrates on boundary prompts, where the base model contains rare correct trajectories that are recoverable by sampling but too sparse to reliably appear in finite RLVR rollout groups. We argue that a two-mode account explains this as an absence-of-evidence failure: rare correct trajectories may disappear before RLVR samples and reinforces them often enough. The main contribution is this diagnostic and mechanistic framing. Per-Problem Base Anchoring (PBA) is a deliberately simple proof-of-concept: sharpen prompts with sufficient frozen-base correct evidence, and anchor risky prompts to the base distribution. Across three training seeds on Omni-MATH-Test, with MATH500 as a secondary high-coverage validation benchmark, PBA improves both \PassK{1} and high-budget coverage over matched GRPO. A 3000-prompt regime-controlled diagnostic study is consistent across seeds with the expected signature: ordinary GRPO loses base-solvable boundary prompts, while PBA preserves rare verifier-positive trajectories. We use mathematical verifiers as a controlled testbed for verifier-guided optimization; the same pass@k inversion risk applies to ECCV-relevant vision-language agents when repeated visual, spatial, or chart-reasoning attempts are checked by external tools or verifiers. Reasoning post-training should decide not only how strongly to optimize, but which prompts are safe to optimize. 

---
# Improving Access to Essential Medicines via Decision-Aware Machine Learning 

**Authors**: Angel Tsai-Hsuan Chung, Jatu Abdulai, Patrick Bayoh, Lawrence Sandi, Francis Smart, Hamsa Bastani, Osbert Bastani  

**Link**: [PDF](https://arxiv.org/pdf/2607.20542)  

**Abstract**: A critical challenge in healthcare systems in low- and middle-income countries (LMICs) is the efficient and equitable allocation of scarce resources, particularly essential medicines. This problem is complicated by limited high-quality data, which restricts the applicability of traditional data-driven techniques. We propose a novel decision-aware machine learning framework for essential medicines allocation, which additionally leverages multi-task learning to ensure sample efficiency and catalytic priors to ensure equitable allocation. In collaboration with the Sierra Leone national government, we performed a staggered, nationwide deployment of our system as a decision support tool. Our econometric evaluation finds an estimated 19% increase in consumption of allocated products in treated districts, demonstrating its efficacy at improving access to essential medicines. Our tool was subsequently scaled nationwide, covering an estimated 2 million women and children under five. Our work demonstrates how machine learning methods can improve efficiency at very low cost in resource-constrained global health settings. 

---
# HypNO: A Graph-Based Neural Operator with Physics-Informed Message Passing for Hyperbolic Conservation Laws 

**Authors**: Dimitrije Ždrale, Cassie An Jeng, Katie Wang, Sonia Vanier, Alexandre Bayen, Hossein Nick Zinat Matin  

**Link**: [PDF](https://arxiv.org/pdf/2607.20541)  

**Abstract**: We introduce HypNO, a graph-based neural operator for scalar hyperbolic conservation laws. HypNO operates directly on a space-time graph of finite-volume cells and uses adjacency-factored, physics-informed message passing to respect upwinding and entropy admissibility near shocks. We benchmark the architecture on the Lighthill-Whitham-Richards (LWR) and Aw-Rascle-Zhang (ARZ) traffic-flow models, a stress test for operator-learning methods because of their simultaneous global transport and shock formation. HypNO predicts solution snapshots accurately across a range of initial conditions while capturing the shocks and discontinuities of the solution. 

---
# From Atoms to Entropy: Optimal Noise Allocation for Diffusion Training in the Convex Regime 

**Authors**: Luca Ambrogioni, Giulio Franzese, Alberto Foresti, Gabriel Raya, Bac Nguyen, Georgios Batzolis, Yuhta Takida, Naoki Murata, Chieh-Hsin Lai, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2607.20540)  

**Abstract**: How should a diffusion model decide which noise levels to train on, and how much? Despite the importance of this choice, current noise schedules are based largely on heuristics or empirical tuning. Here, we develop a general statistical framework for studying asymptotically optimal noise-level allocation in diffusion training. Our first main result concerns the fully coupled regime, where information can spread between different time points. Under convexity or Polyak-Lojasiewicz-type assumptions, we show that the optimized training schedule admits an atomic minimizer, concentrated on finitely many noise levels. Our second main result specializes this framework to an idealized independent-learner regime, intended to model temporal specialization in neural networks. Under an additional feature-noise decoupling condition, a random-matrix analysis leads to an information-theoretic proxy: the decoupled sampling density is proportional to the square root of the generative entropy rate, the rate at which conditional entropy grows along the forward process. We test these predictions in controlled settings where the coupled objective can be optimized directly, including Dirac mixtures, low-dimensional manifolds, and MNIST. In these settings, the optimized schedules are consistently finite-support, while the smooth entropic proxy closely tracks the atomic optimum in neural-network models and breaks down mainly in the fully coupled parametric case, as the theory suggests. We then evaluate the entropic schedule in larger-scale experiments, where full schedule optimization is currently intractable. The results indicate that square-root entropy scheduling can substantially improve training efficiency on discrete domains and remains competitive with standard EDM-style heuristics on continuous images. 

---
# Leveraging Biokinetic Knowledge Priors for Data-Scarce Bioprocess Modeling 

**Authors**: Kyunghoon Hur, Eunjung Jeon, Hyun Woo Kim, Gyubok Lee, Seongjun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20539)  

**Abstract**: While deep learning has accelerated drug discovery, its impact on biomanufacturing has been considerably more limited. The reason is data scarcity. Bioreactor experiments are high-cost, take days to weeks, and are rarely shared in public form, leaving each research work with only a handful of experiments. The domain itself, however, is rich in prior knowledge. Biokinetic ordinary differential equation (ODE) models have described microbial growth for decades, yet how to inject this knowledge into a neural network has not been studied systematically. We present the first systematic study of how to inject this ODE knowledge into a neural network, comparing a data-level prior that pre-trains a generic decoder on simulated ODE curves against an architecture-level prior that embeds the ODE inside the decoder. Both consistently outperform no-prior baselines across 11 datasets and 7 microbial species. Our central finding is that the two are substitutable. A generic decoder pre-trained on simulation matches a fully bio-structured decoder trained on real data. Simulation pre-training therefore offers a simple, data-efficient recipe for deep learning under bioprocess data scarcity. 

---
# Codec-Gauge: Learning Compression-Friendly Gauges for Transformer KV Caches 

**Authors**: Yitao Jiang, Yaoqing Yang, Luyang Zhao, Muhao Chen, Devin Balkcom  

**Link**: [PDF](https://arxiv.org/pdf/2607.20538)  

**Abstract**: Long-context Transformer inference increasingly relies on KV-cache compression or quantization. Prior rotation and transform-coding results suggest that the channel basis of each key/value vector affects how faithfully a fixed backend preserves model behavior. We introduce Codec-Gauge, a post-training cache-coordinate layer that learns small orthogonal channel transforms around existing compression and quantization backends. Its frequency-distribution objective combines a token-channel DCT spectral-centroid loss with a smooth rate proxy to concentrate KV energy in low-frequency codec-facing layouts. We evaluate actual compression and decompression using measured bytes and rolling compressed-history scoring. Across six models at $3$, $4$, and $6$ bits/value, learned gauges reduce zfp KL divergence by $44.0\%$ on average relative to raw coordinates and outperform random, Hadamard, DCT, and PCA/KLT controls. The same gauges improve quality preservation for block-uniform and KIVI-style quantization. Experiments on a 27B model and long-context task prompts reproduce the quality trend, while serial storage and timing measurements validate the implemented compressed-cache paths. These results establish cache-coordinate geometry as a practical post-training variable for improving compression fidelity without changing model weights, attention semantics, or backend coding rules. 

---
# ReliableTableQA:How Much Supervision Does Reliability Annotation Need? 

**Authors**: Huei-Chung Hu, Hsin-Tai Wu, Koyo Kobayashi  

**Link**: [PDF](https://arxiv.org/pdf/2607.20537)  

**Abstract**: We introduce ReliableTableQA, a framework for training an LLM to annotate the statistical reliability of tabular QA results, not whether the query is answerable, but whether the computed answer is statistically meaningful. In real enterprise analytics, a syntactically correct SQL query can return a value that is based on too small a sample, has an excessively wide confidence interval, or is too confounded to support action. Existing systems answer confidently in all such cases, a failure we quantify as the Unreliable Confident Answer Rate (UCAR). We contribute (1) a ten-category reliability taxonomy (R1-R10) covering hazards such as small-sample aggregates, multiple-comparison inflation, and distribution-tail mismatch; (2) a program-first data pipeline that generates 50,000 reliability-labeled training examples from a context-free grammar over public retail schemas, with schema-stratified SFT/GRPO splits; and (3) a controlled study of how much supervision calibrated reliability annotation actually requires. We find that a small, schema-stratified SFT set is remarkably sufficient: 200 examples raise reliability-flag F1 from 0.61 to 0.98 and parse rate from 0.52 to 1.00, drive UCAR to zero, and yield a model that generalizes to an unseen retail domain (Rel-F1 0.997 on held-out H&M). Against this strong SFT baseline, GRPO, commonly assumed to be essential, helps only when SFT is under-trained (+0.06-0.16 exact-flag-set match at 100 examples, in- and out-of-distribution) and provides no measurable benefit once SFT is adequate, a null result we confirm across a hard compound-flag slice, a strict exact-match metric, and out-of-distribution evaluation. Our findings reframe reliability annotation as a data-efficiency problem and delineate precisely when reinforcement fine-tuning does and does not pay off. 

---
# A Graph Neural Network approach to zero-shot Digital Twins 

**Authors**: Alicia Tierz, Icíar Alfaro, David González, Elías Cueto  

**Link**: [PDF](https://arxiv.org/pdf/2607.20535)  

**Abstract**: Traditional Predictive Digital Twins often remain geometrically rigid, requiring extensive retraining or fine-tuning whenever the underlying physical domain or boundary conditions change. To overcome this limitation, we present a novel framework for \textit{Zero-Shot Digital Twins} that seamlessly couples real-time visual perception with a geometry-agnostic, physics-informed reasoning engine. At the core of our architecture is the Thermodynamics-Informed Graph Neural Network architecture, a Geometric Deep Learning solver grounded in a metriplectic thermodynamic formalism that enforces energy conservation and non-negative entropy production locally through graph message passing. The framework integrates an auxiliary Graph Neural Network to infer unobservable fields (such as stress tensors or velocity and energy distributions) directly from sparse initial visual boundaries, mitigating numerical start-up transients. To bridge the sim-to-real gap, we implement a continuous closed-loop data assimilation mechanism; the pipeline tracks macroscopic deformations and free-surface fluid boundaries in real-time using deep segmentation networks combined with sparse optical flow, dynamically correcting the autoregressive simulation rollout and eliminating numerical drift. To test the validity of our approach, we demonstrate the extreme generalization capabilities of our approach across two disparate physical regimes: the large deformations of a viscoelastic beam and the non-linear sloshing of a viscous fluid. In both scenarios, the unified framework instantiates physically accurate simulations on novel, unseen geometries without case-specific retraining, operating well within real-time latency budgets (approximately 25 ms per frame) and enabling the direct projection of latent mechanical variables via Augmented Reality. 

---
# Grounding Investor Views: Neural Predicates in the Black-Litterman Model 

**Authors**: Marcos Florencio  

**Link**: [PDF](https://arxiv.org/pdf/2607.20533)  

**Abstract**: Portfolio construction under the Black-Litterman model requires investors to specify views on asset returns alongside explicit uncertainty estimates -- a process that remains largely subjective and difficult to scale. We propose a formal approach in which neural predicates serve as a structured, probabilistic mechanism for view generation. In our formulation, structured financial analysis data is processed through a compositional hierarchy of neural predicates whose outputs -- probability distributions over market stances -- are mapped to the pick matrix $\mathbf{P}$, the view return vector $\mathbf{q}$, and the view uncertainty matrix $\boldsymbol{\Omega}$ of the Black-Litterman model. View confidence is derived from predicate output distributions, providing a data-driven alternative to subjective uncertainty elicitation. The resulting approach is interpretable, in the sense that any portfolio weight can be traced back through the predicate's logical chain to the underlying data, and fully differentiable, enabling end-to-end learning. 

---
# Position: Stop Reactively Patching Your Model Every Time and Start Proactive Test-Driven AI Development 

**Authors**: Nadine Chang, Maying Shen, Jialiang Wang, Rafid Mahmood, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2607.20532)  

**Abstract**: Many modern AI systems are designed to operate under diverse, open-ended, use-cases. To help generalize deployed systems, many deployed-system maintenance pipelines use a reactive AI flywheel that observes emerging feedback from user behavior (errors) and patches the model accordingly. However, when used as the primary maintenance mechanism, these flywheels often ignore the broader context of these errors within the system's objectives, failing to preempt potential future edge cases, which leads to more unnecessary flywheel iterations. Also, it is statistically increasingly difficult to collect remaining errors due to the long-tail nature of open-world use-cases. This position paper argues that a proactive test-driven flywheel is required to address reactive flywheel's limitations and to approach a generalizable system. We advocate for creating a "test space" to technically map feedback data to task objectives, evolving the flywheel from reactive to proactive. We augment our position by mathematically proving a proactive one achieves better long-term scaling with fewer iterations than the reactive flywheel. 

---
# CLOE: Christoffel Loss Autoencoder for Anomaly Detection 

**Authors**: Léa Billet, Louise Travé-Massuyès, Elodie Chanthery, Alexandre Gaffet  

**Link**: [PDF](https://arxiv.org/pdf/2607.20530)  

**Abstract**: Semi-supervised anomaly detection plays a key role in diverse fields such as process monitoring, healthcare, and finance. However, lightweight methods often struggle with high-dimensional data and typically require careful tuning of multiple hyperparameters. Among existing approaches, Christoffel Function--based methods are attractive due to their simplicity, requiring at most a single hyperparameter. They also benefit from a well-established theoretical foundation that yields several interesting results for data science. However, their main limitation is poor scalability to high-dimensional settings. In this paper, we introduce CLOE, a new method that combines an autoencoder for dimensionality reduction with a Christoffel Function--based detector applied in the latent space. To better align representation learning with anomaly detection, we design a novel loss function that leverages the Christoffel Function to guide the autoencoder toward representations that better capture the support of the normal data distribution. We further propose a principled procedure to set the detection threshold and an efficient strategy to tune the single remaining hyperparameter. Experiments on multiple high-dimensional tabular anomaly detection benchmarks demonstrate that CLOE achieves superior performance compared to existing methods, while preserving the lightweight and low-tuning advantages of Christoffel Function--based approaches. 

---
# Uncertainty-Aware Trust Estimation for Multi-LLM Systems via Structured Expert Judgement 

**Authors**: Jiawei Zheng, Jiazhen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20529)  

**Abstract**: Large Language Model (LLM) ensembles are increasingly used to improve reliability by combining predictions from multiple LLMs. However, existing aggregation methods typically assume that all models are equally trustworthy, overlooking differences in uncertainty quality. This assumption is poorly suited to heterogeneous LLMs, whose reliability and capability vary significantly, making naive aggregation vulnerable to unreliable or adversarial experts. In this work, we formulate multi-LLM aggregation as a problem of uncertainty-aware trust estimation. We adapt structured expert judgment from decision theory, using context-aware calibration questions to estimate expert reliability based on the quality of its probabilistic predictions. Specifically, we employ Cooke-style log weighting, which penalises overconfident incorrect predictions and favours well-calibrated experts. We evaluate our approach on MMLU and MMLU-Pro across homogeneous, heterogeneous, and contaminated expert panels. Results show that while aggregation methods perform similarly in homogeneous settings, Cooke weighting becomes critical under heterogeneity and contamination. It achieves a superior accuracy-reliability balance and remains robust when unreliable experts are introduced. These findings suggest that Multi-LLM aggregation requires not just combining predictions, but calibrating trust under uncertainty. 

---
# Scaling Closed-Loop Feature Channel Configuration with LLMs 

**Authors**: Tolgay Atinc Uzun, Radu Timofte, Dmitry Ignatov  

**Link**: [PDF](https://arxiv.org/pdf/2607.20516)  

**Abstract**: Promising initial results in closed-loop large-language-model-based channel-configuration search demonstrated that neural-network widths can be optimized directly through executable code generation and accuracy feedback. However, those results were obtained from a relatively sparse set of valid evaluations, leaving open whether the observed optimization behavior transfers to a denser sampling regime and whether additional architectural regularities emerge when more generated networks are evaluated. To test this, the same search setting is scaled to 250 candidate networks per fine-tuning cycle. The analysis covers 2000 generated candidates from 8 complete cycles, yielding 462 verified CIFAR-100 evaluations after task and metadata filtering. Per-cycle mean accuracy exhibits a positive linear trend with slope 9.87e-4 (p=0.043), while the high-performing frontier improves more strongly: the best observed accuracy increases from 0.3144 to 0.3676, and both the top-5 and top-10 cycle-level means exhibit positive trends. The scaled run also reveals improved parameter efficiency. The best model reaches 0.3676 with 11.8M parameters, compared with an early high-performing model at 0.3144 with 166.5M parameters. Beyond accuracy, the larger sample exposes architectural regularities that were difficult to assess from sparse observations. Non-power-of-two channel widths occur in 41.8% of verified candidates, and the strongest models share structured channel-allocation patterns characterized by moderate early widths and expanded middle or later blocks. These findings indicate that the channel-search signal observed in the initial study transfers 

---
# The Active Ingredient in Muon's Grokking 

**Authors**: Yufeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20512)  

**Abstract**: The Muon optimizer reaches the grokking threshold on modular arithmetic faster than AdamW. Prior work attributes this to "spectral-norm constraints plus orthogonalized momentum" but does not isolate which mechanism matters. To better understand Moun's behavior, we run multi-seed and multi-learning-rate sweeps to decompose and stress-test the effect. First, an ablation shows the speedup comes from orthogonalization (the Newton-Schulz iteration): orthogonalize-only matches full Muon, whereas spectral-only is no faster than AdamW and is unreliable, and this verdict holds across learning rates. Second, a mechanistic analysis finds that orthogonalizing optimizers reach generalization at roughly 3x lower spectral norm and, controlling for how much the embedding actually moves, settle into a lower-norm solution rather than simply perturbing the embedding less. Third, reducing the Newton-Schulz iteration count from five to one accelerates reaching the threshold but makes the grokked solution fragile, prone to transient collapse, with fragility that grows with learning rate; a single iteration is fast and stable only at small learning rate, while the canonical five iterations are the learning-rate-robust choice. We also show spectral scaling can be dropped at no measured cost. A methodological thread runs throughout: under a stability-aware metric, "faster" claims about grokking optimizers can invert, so we report both first-crossing and sustained-grok times. To support reproducibility, we release our full training and analysis code at this https URL 

---
# PhantomFill: When the Form Demands an Answer, Language Models Invent One 

**Authors**: Rana Muhammad Usman  

**Link**: [PDF](https://arxiv.org/pdf/2607.20492)  

**Abstract**: Language models in production do not write prose. They fill forms: JSON fields, function arguments, extraction templates. We show that the form itself causes hallucination.
We ask thirteen models the same question about the same input and change only the answer format. The inputs are built so the question cannot be answered: a viral post showing 12,400 likes but no visible replies, a support ticket whose call was never transcribed. In free text, GPT-5.5 answers honestly. It says there is no reply data, 98% of the time. Given a required JSON field for sentiment, the same model invents an answer 40 times out of 40. It fabricates the mood of crowds it never saw and quotes customers it never heard.
The pattern holds with force. Required fields drive fabrication to 100% in ten of thirteen models. An explicit "insufficient evidence" option rescues only the frontier: all nine open-weight models ignore it. A direct instruction, do not infer sentiment, is overridden by the schema in four of six models. Resistance does not come with scale: within a single model family, the smallest model refuses, the mid-sized model fabricates, the largest refuses again. Honesty under format pressure is a training outcome that no one is measuring.
The fabrication hides exactly where hedging is impossible: in required enums and minimum-count arrays, fields where no disclaimer fits. We release PhantomFill, a benchmark with deterministic scoring and two reportable numbers: the Coerced Fabrication Rate and the Escape Utilization Rate. The fix we test is one line of schema. The failure we measure is everywhere. 

---
# Verifier-First Evaluation of Agentic LLMs for Infrastructure-as-Code Generation 

**Authors**: Mohamed Jouini  

**Link**: [PDF](https://arxiv.org/pdf/2607.20478)  

**Abstract**: Infrastructure-as-Code (IaC) generation from natural language requires satisfying provider schemas, dependency planning, and organizational policy constraints, not merely producing syntactically plausible configurations. We present a verifier-first empirical study of seven agentic strategies for Terraform generation evaluated on IaC-Eval v2, a modernized 186-task AWS/Terraform benchmark with Rego v1 intent policies. Our evaluation separates failures into three verifier stages (terraform validate, terraform plan, opa eval) and applies McNemar's test with Wilson confidence intervals on all pairwise comparisons (n=186, alpha=0.05).
We report five principal findings. (1) Active retrieval via ReAct agents with MCP or ChromaDB-backed RAG raises Qwen2.5-Coder 7B from 14.0% to 45.7% pass@1 (p<0.0001), primarily by reducing VALIDATE_FAIL from 144 to 66 tasks. (2) Iterative refinement with verifier feedback achieves 62.9% (Qwen 7B) and 84.4% (GPT-4o) pass@1, exhibiting binary convergence -- tasks either resolve in one retry or exhaust the budget. (3) GEPA reflective instruction optimization raises the Active RAG baseline by +7.5 pp (p=0.026) using only 80 verifier-guided rollouts, providing evidence that prompt optimizers can improve verifiable IaC generation without weight updates. (4) SIMBA teacher-free demonstration injection achieves performance equivalent to Active RAG (p=1.0) without retrieval infrastructure, but fails to address the dominant SELF_DEFINED_PROPERTY error class (50% of failures). (5) A diagnostic Rego-injection experiment shows that 79% of post-refinement OPA failures are information-gap failures resolvable when policy text is visible (p=0.016), motivating policy 

---
# Can Valence Reflect Morality in Natural Language? A Preliminary Annotation Study 

**Authors**: Jonny O'Dwyer, Malika Bendechache, Louise McCormack, Elif Calik, Ramin Ranjbarzadeh, Dost Muhammad, Shokofeh Anari Bozcheloei, Ishita Singh  

**Link**: [PDF](https://arxiv.org/pdf/2607.20461)  

**Abstract**: Present implementations of artificial intelligence (AI) ethics do not adequately take feelings, or affect, into account. If AI should be aligned with human ethics, it seems reasonable to thoroughly investigate the possibility of AI behaviour that mirrors virtuous human ethical conduct, where feelings play a role in the actions, judgements or statements one makes. Furthermore, while prominent theories of normative ethics are often discussed in terms of their differences and shortcomings, Virtue, Consequentialist, and Kantian Deontological ethics all share a common feature of considering human feeling to some degree while the popular descriptive ethics theory, Moral Foundations Theory, positions feelings as central to many of its foundations. Therefore, in the present paper, a data set of moral valence is proposed, consisting of 500 annotations by six human participants for both action/judgement and consequence moral valence, ranging from -1 to 1 for text-presented scenarios from the Commonsense Norm Bank data set. The resulting valence features share significant relationships with multi-class (immoral/discretionary/moral) and binary immoral/moral categories while additionally providing a noteworthy test set Matthew's correlation coefficient of 0.764 using regularised logistic regression for binary classification. This provides early evidence of the usefulness of valence features for morality estimation of text, indicating that valenced consequences of responses for others can be considered toward more human morally-aligned AI. In the interest of promoting further affective-moral computing research, this study's annotations will be made available for research on request. 

---
# Instruct-FD: Can Your Full-Duplex Speech System Follow Turn-Taking Instructions? 

**Authors**: Yuzhi Tang, Wentao Ma, Xiling Zhao, Ahmad Salimi, Sepehr Harfi Moridani, Dongming Shen, Jixuan Wang, Abdulrahman Abdulrazzag, Murdock Aubry, Yu-Hua Chen, Daniel Lee, Jaewon Lee, Jonah Mackey, Silin Meng, Nicholas Stranges, Chenxu Xiong, Hao Yu, Yi Zhu, Mu Li, Alex Smola  

**Link**: [PDF](https://arxiv.org/pdf/2607.20460)  

**Abstract**: Current full-duplex (FD) spoken dialogue systems can produce fluid interactions, yet it remains unclear whether they can adapt their turn-taking behavior when explicitly instructed. This is critical for real-world deployment, where conversational policies vary across applications (e.g., proactive tutoring vs. passive counseling). We introduce Instruct-FD, an instruction-conditioned benchmark for evaluating controllable turn management in FD systems. To enable this, we develop a human-validated, scalable synthetic pipeline that generates instruction-conditioned conversations, along with a deployment-agnostic multi-turn evaluation protocol and an LLM-based judge. Benchmarking six state-of-the-art full-duplex systems reveals a substantial gap in instruction-following turn management: the best model achieves only 64.4% adherence. Performance is highly uneven across behaviors and scenarios, with proactive behaviors such as model backchanneling and interruption remaining particularly challenging. These findings establish instruction-following turn management as a crucial direction for building adaptable and deployable full-duplex dialogue systems. 

---
# THOR: A Theta-Gamma Hierarchical Oscillatory Reasoning Framework for Multi-hop QA 

**Authors**: Ziyang Ling, Ronald X. Xu, Mingzhai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.20459)  

**Abstract**: Multi-hop question answering requires retrieving and integrating evidence from multiple contexts. Despite the rapid progress of current research, multi-hop reasoning remains constrained by two persistent limitations: attention decay, where the model's focus on main question degrades as the reasoning chain grows, and error accumulation, where mistakes propagate across hops and compounds into final failure. Inspired by Theta-Gamma hierarchical oscillation which decouples global planning from local retrieval, enabling efficient attention transfer between hops and a verification and repair mechanism that interrupts the accumulation of errors in the wrong paths, we present THOR, a brain-inspired Theta-Gamma hierarchical oscillatory reasoning framework. Extensive comparative experiments and specific validation experiments on multi-hop QA benchmarks demonstrate that THOR improves answer accuracy and robustness while mitigating limitations, showcasing its generalization across different backbones. 

---
# CAMeR: Keyword-Gated Hybrid Activation for Adaptive Memory Retention in LLM Agents 

**Authors**: Haowen Lai  

**Link**: [PDF](https://arxiv.org/pdf/2607.20458)  

**Abstract**: Large language model (LLM) agents operating over extended dialogues accumulate vast amounts of information, yet existing memory systems either retain everything indiscriminately or apply uniform forgetting heuristics that fail to distinguish relevant from irrelevant knowledge. We present CAMeR (Context-Activated Memory Reinforcement), a memory retention framework combining keyword-gated hybrid activation -- a joint symbolic (word-level Jaccard) and sub-symbolic (embedding cosine) gating mechanism -- with adaptive weight dynamics. CAMeR computes a hybrid similarity score for each memory-query pair; memories exceeding a threshold receive reinforcement while all memories undergo controlled decay. We introduce CAMeR-Bench, a 76-memory, 100-round benchmark spanning 8 topic clusters with graded activation frequency, designed to test adaptive retention where existing benchmarks (LoCoMO, LongMemEval) cannot. On CAMeR-Bench, CAMeR's keyword gate achieves a 1.6$\times$ larger retention gap between high-frequency and never-referenced memories compared to embedding-only gating (scissors gap: 0.039 vs. 0.024), while time-driven baselines (Oblivion, SuperLocalMemory) collapse to near-zero weights over 100 rounds. CAMeR's top-5 retrieval saves 83.2\% tokens versus full-context approaches (39k vs. 231k cumulative) while producing weight signals that improve retrieval precision. Through 8 ablation conditions we establish that the keyword gate -- not learnable decay -- is the primary performance driver at this scale. Our findings demonstrate that hybrid symbolic-neural gating provides a simple yet effective mechanism for adaptive memory retention in LLM agents. 

---
# Dropping the Anchor: Statistical Context Summarization for Distributed Systems via Pulsar Attention 

**Authors**: Aryan Sood, Shantanu Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2607.20457)  

**Abstract**: Inference with large language models (LLMs) on long sequences is computationally expensive due to the quadratic complexity of self-attention. Distributed blockwise methods such as Star Attention reduce this cost by sharding context across hosts, but rely on prepending a static, content-blind copy of the first block to every host. We propose Pulsar Attention, which replaces the static anchor with two lightweight, content-aware components: a small attention-sink prefix that stabilizes softmax, and compact cross-block summaries built via a Max-IDF heuristic that selects chunks containing globally rare tokens. This reduces the Phase 1 per-GPU FLOPs by up to 3.3$\times$ over Star Attention while retaining an identical KV cache footprint. On RULER and BABILong with Llama-3.1-8B, Pulsar Attention outperforms both Star Attention and dense attention at sequence lengths up to 128K tokens, with absolute gains of up to 4.7% over the dense baseline. 

---
# Learn2Zinc: Fine-tuning Small Language Models for Text-to-Model Translation in MiniZinc 

**Authors**: Serdar Kadioglu, Karthik Uppuluri  

**Link**: [PDF](https://arxiv.org/pdf/2607.20456)  

**Abstract**: Large language models excel at code generation for mainstream programming languages but struggle with rare, domain-specific languages such as MiniZinc, a constraint modeling language for combinatorial problems. We investigate whether targeted fine-tuning can teach small language models (0.6B to 20B parameters) to generate syntactically correct and semantically valid MiniZinc models from natural language problem descriptions. Our key finding is that syntax errors dominate failures when working with this domain specific language: the out-of-the-box execution accuracy of small language models such as Qwen3, LLaMa, Gemma, and GPT-OSS is near-zero. We propose a cross-model error bootstrapping approach that collects syntax errors from multiple LLM runs and leverage those to curate an error correction training dataset. This dataset allows us fine-tune small language models that consistently improves both direct code generation and chain-of-thought approaches across all model sizes. With self-reflection and ensembling, our approach achieves up to 98\% execution accuracy. In parallel, solution accuracy still remains at 35\%, indicating that while syntax is learnable, constraint reasoning remains a challenge. We contribute our fine-tuning pipeline, datasets, and models to opens-source for further research on text-to-model translation. 

---
# RE-AD: Real-Time Requirement Adherence for Data Labeling 

**Authors**: Siddarth Malreddy, Ishan Nigam, Akshay Arora, Nikhil Mittal, Subrat Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20455)  

**Abstract**: Human-annotated data remains fundamental to training frontier Large Language Models (LLMs). However, crowd-sourced annotations often suffer from quality issues stemming from annotator misunderstanding or lack of engagement. To address this, we introduce a real-time requirement adherence (RE-AD) framework that leverages LLMs to proactively validate labeling quality. Our methodology involves decomposing Standard Operating Procedures (SOPs) into atomic rules via self-reflection, categorizing them by complexity, and applying tiered validation strategies. Evaluated on a synthetic benchmark, the system achieved an F1 score of 0.749. Furthermore, production deployment resulted in annotators accepting and fixing 82% of the errors flagged by the framework. We include ablation studies to demonstrate the impact of our core design decisions. 

---
# Response drift across frontier large language models 

**Authors**: Mohammed Aledhari, Ali Aledhari, Fatimah Aledhari, Gowtham Venkat Eathamokkala, Mohamed Rahouti  

**Link**: [PDF](https://arxiv.org/pdf/2607.20454)  

**Abstract**: All frontier large language models (LLMs) exhibit response drift -- producing outputs that deviate from expert-validated references -- yet the magnitude and structure of this drift remain uncharacterised by systematic human evaluation. Here we report a fully crossed evaluation in which 47 geographically diverse participants each assessed all 62 multidomain questions across ten frontier LLMs under blinded conditions, yielding 29,140 independent assessments. Every model drifts, but drift magnitude varies substantially: eight models converge on a statistically indistinguishable ceiling (78-81% deviation), while two achieve lower deviation (47-49%). Drift profiles differ across six domains and 62 questions, with pairwise correlations among ceiling models exceeding r = 0.85. Automated similarity metrics explain less than 2% of variance in human judgements. These findings reveal that response drift is universal across frontier LLMs, domain- and question-dependent in structure, and accessible only through human-centred evaluation. 

---
# A Knowledge-Injection Framework for Zero-Shot Adaptation of LLMs to Delirium Prediction 

**Authors**: Jessica Sena, Shesadree Priyadarshani, Miguel Contreras, Bharat Gandhi, Scott Siegel, Subhash Nerella, Parisa Rashidi  

**Link**: [PDF](https://arxiv.org/pdf/2607.20453)  

**Abstract**: Large language models show promise for clinical prediction, but zero-shot performance on specialized tasks is limited by incomplete domain knowledge, especially for smaller locally deployable models. We present a lightweight knowledge-injection framework for zero-shot ICU delirium prediction that augments a deterministic natural-language summary of structured electronic health record data with an external clinical knowledge report at inference time, without fine-tuning or retrieval. We evaluate LLaMA 3.1 8B and LLaMA 3.3 70B on 3,160 ICU admissions from the MIMIC IV dataset. Adding a clinically meaningful external knowledge report improves AUROC by 8.57 percentage points for the 8B model and 1.99 percentage points for the 70B model compared to no external knowledge. Relative to a GPT-5.2 frontier-model reference without external knowledge report (AUROC 68.86%), knowledge injection reduces the performance gap from 15.66 to 7.09 AUROC points for LLaMA 8B and from 5.30 to 3.31 AUROC points for LLaMA 70B. Random control reports do not improve performance and often degrade it, indicating that gains depend on clinically meaningful content rather than added prompt length alone. SHAP-based attribution further confirms that the injected knowledge is actively used during prediction. These findings suggest that inference-time knowledge injection can narrow the gap between locally deployable open-weight models and frontier closed models while preserving a practical, privacy-preserving workflow for resource-constrained clinical settings. 

---
# The Storyteller in the Model: Narrative Pattern Inheritance, Escalation Dynamics, and Alignment Governance in LLMs 

**Authors**: Adam Rigby, Raz Saremi, Azadeh Sohrabinejad, Mehdi Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2607.20449)  

**Abstract**: LLMs are trained predominantly on human-authored text, yet the structural and narrative conventions embedded in that text are rarely examined as a source of systematic behavioral influence, or as a governance risk in deployed systems. This paper considers whether the storytelling patterns inherent in published human writing, including archetypal roles such as protagonist, antagonist, and underdog, as well as tension-and-resolution narrative arcs, are absorbed during training and subsequently surface in LLM outputs, causing responses to drift toward unexpected, adversarial, or rhetorically enticing behaviors over extended interactions. Through a systematic literature review and cross-paper analysis of recent empirical studies on LLM alignment, persona dynamics, emergent misalignment, and user interaction patterns, we observe evidence bearing on this hypothesis. The findings reveal three key patterns. First, LLMs reproduce statistical patterns from their training data rather than reasoning independently. Second, measurable latent traits, including sycophancy and deceptiveness, emerge reliably across unrelated prompts. Third, fine-tuning on a narrow narrative task can produce unintended behavioral changes well beyond that task. Furthermore, evidence suggests that persuasive, narrative-style outputs are among the most common LLM products in real-world usage, amplifying these risks. Narrative drift constitutes an unmonitored escalation pathway in deployed AI systems, one that evades discrete-incident detection mechanisms and requires dedicated monitoring instruments. 

---
# Confidently Deceptive: How Confidence Amplifies the Risk of LLM Deception 

**Authors**: Ali Asad, Stephen Obadinma, Anshul Pattoo, Wenxuan Zhang, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20444)  

**Abstract**: Large language models (LLMs) can produce deceptive responses: outputs that mislead users in service of a contextually or experimentally induced goal. Yet it remains unclear how confidently models deceive and whether higher confidence makes deceptive responses more persuasive to end users. In this paper, we study these basic questions in various models and different deception datasets. We provide a comprehensive study measuring confidence through both verbalized self-reports and a range of logit-based estimators. We show that LLMs deliver deceptive responses with substantial verbalized confidence and that human annotators prefer the higher-confidence deceptive response 78% of the time in paired comparisons. Misalignment fine-tuning amplifies the problem. Confidence in deceptive responses rises across all three benchmarks, increasing the resulting potential risk, with effects generalizing beyond the training distribution. Strikingly, models classify their own deceptive outputs as deceptive at high rates (82.7% under misalignment) while still predicting they would produce them - recognition without avoidance. We argue that confident deception is a distinct alignment risk requiring evaluations that jointly measure deception, confidence, and awareness. 

---
# Answer-then-Edit: Reasoning Skeleton Editing for Anti-Distillation with Preserved Utility 

**Authors**: Fan Li, Mengting Pan, Sijia Xu, Xiaoyang Wang, Chen Chen, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20440)  

**Abstract**: Proprietary large language models (LLMs) entail substantial intellectual and financial investment, making them valuable intellectual property (IP). However, even when deployed via black-box APIs, these models remain vulnerable to unauthorized knowledge distillation, which allows adversaries to cheaply extract and replicate model capabilities. To address this issue, anti-distillation (AD) has been proposed to generate defensive outputs that hinder distillation effectiveness, overcoming the limitation of watermarking-based approaches that rely on post-hoc verification. However, existing AD methods based on internal model perturbations struggle to balance anti-distillability and utility (e.g., answer accuracy and naturalness) of reasoning traces, with stronger defenses often causing significant utility loss. To fill this gap, we propose \textbf{\underline{S}}keleton-\textbf{\underline{G}}uided \textbf{\underline{R}}easoning \textbf{\underline{E}}diting (SGRE), an \textit{Answer-then-Edit} framework that performs post-hoc trace modification for anti-distillation. In the answer stage, the teacher model first generates clean reasoning traces, preserving the original reasoning accuracy while enabling more flexible control over trace naturalness. In the editing stage, we draw inspiration from Cognitive Load Theory (CLT) and introduce a three-stage strategy consisting of reasoning skeleton extraction, skeleton graph coarsening, and skeleton verbalization. These operations jointly perturb reasoning structures and augment textual complexity to amplify extraneous load on student models, hindering their acquisition of underlying reasoning patterns. Extensive experiments across diverse LLMs demonstrate that SGRE achieves state-of-the-art performance in reducing distillation effectiveness, while maintaining lossless reasoning accuracy and superior trace naturalness. 

---
# Preference Tuning as Spectral Update Reorganization 

**Authors**: Peiyan Zhang, Haibo Jin, Liying Kang, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20438)  

**Abstract**: Preference-based post-training is usually understood through endpoint behavior, yet the learned update that produces this behavior remains largely opaque. We study RLHF and related preference optimization through the spectral structure of their induced parameter updates. By decomposing effective LoRA updates and reloading their spectral components as plug-in modules, we turn preference-induced updates into objects that can be isolated, recomposed, and directly intervened on. Across model families, optimization algorithms, and supervision regimes, these updates consistently develop a spectral head--tail organization. A compact head emerges early and carries the dominant endpoint shift, while a heterogeneous residual tail remains. The split is functional rather than merely descriptive. Plug-in intervention shows that the head accounts for the visible behavioral departure from the base model, while the tail is weak in isolation. Cross-run recomposition further shows that mixed adapters follow the source of the head, indicating that the head carries run-level solver bias. This endpoint dominance does not imply learning sufficiency. Head-only learning is non-vacuous but fails to recover the full solution, especially on out-of-distribution behavior. Tail-only learning yields little visible gain, yet the full solution is not recovered without the tail. These findings recast preference post-training as structured update reorganization rather than a monolithic behavioral correction, and suggest that alignment gain and coverage loss are tied to how the learned update itself is organized. 

---
# Routing Subspaces: Auditing Evaluation-to-Deployment Mismatch in Fine-Tuned Language Models 

**Authors**: Phongsakon Mark Konrad, Toygar Tanyel, Serkan Ayvaz  

**Link**: [PDF](https://arxiv.org/pdf/2607.20436)  

**Abstract**: Safety evaluations often assume that behavior observed during testing reflects behavior in ordinary use, but fine-tuning can break this assumption. A checkpoint can appear fixed under evaluation-style prompts while the same behavior persists under ordinary-use prompts. Output scores reveal this mismatch but do not locate it. We investigate whether the distinction is encoded in a stable internal site and introduce an approach that fits a paired activation contrast at a path-patching-informed mid-depth window, then modifies the resulting coordinate on held-out prompts. The intervention closes the evaluation-to-deployment gap in ten of twelve model--behavior settings (six of the eight settings with $n{\geq}120$ paired questions) across four full-matrix instruction-tuned model instances; a fifth model supports localization and edit-provenance checks, and deployment-framed rates change by at most $6.1$pp. The two flat cells, both sycophancy, indicate that a single-coordinate audit is not sufficient when the installed distinction is higher-rank or missed by the depth heuristic. The audit is a diagnostic for fine-tuned checkpoints, not a training-time defense or a guarantee of deployment safety. 

---
# Making Open-Source Text LLM Watermarks Durable Against Merging 

**Authors**: Luisa Scharff, Thibaud Gloaguen, Robin Staab, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2607.20435)  

**Abstract**: Open-source LLMs (OSMs)arereaching near state-of-the-art performance, prompting prior works to trace the text they generate by embedding text watermarking algorithms directly into their weights. Yet, OSMs are subject to post-training modifications, which has been shown to remove the watermark. Model merging in particular, a prominent method used for combining expert knowledge and preventing catastrophic forgetting, strongly removes such OSM watermarks. A key question is how to enable OSM watermarks that survive subsequent merging. In this work, we show for the first time how to design an OSM watermark that is durable against model merging. We propose Merge-Adversarial Training, an adversarial training algorithm to distill text watermarks into model weights while being robust to subsequent model merging. Our approach consistently outperforms all baselines (e.g. with SLERP up to +51 percentage points (pp) TPR@1%FPR with +25 pp on average) while preserving downstream capabilities. We also for the first time evaluate OSM watermarks against realistic merge scenarios, representing common use-cases such as combining expert capabilities or preventing catastrophic forgetting, and with 3 prominent merging algorithms. More broadly, our findings suggest that adversarial training is a reliable approach for increasing OSM watermark durability against post-training modifications. 

---
# Break Through the Compression Bottleneck: From Theory to Practice 

**Authors**: Xiusheng Huang, Lu Wang, Yequan Wang, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20434)  

**Abstract**: As the parameter size of language models continues to grow, effective model compression is required to reduce their computational and memory overhead. Existing compression methods suffer from bottleneck issues: when the compression ratio is increased, performance degrades significantly. Low-rank decomposition and quantization are two prominent compression methods that have been proven to significantly reduce the computational and memory requirements of Large Language Models (LLMs) while maintaining model accuracy. Evidently, combining these two methods will break through the existing compression bottleneck. However, how these two methods interact when combined remains a critical question for developers, as many assume they are orthogonal, meaning their combination would not introduce additional errors beyond those independently introduced by each method. This paper provides the first mathematical proof that low-rank decomposition and quantization are non-orthogonal. We validate these findings through a series of experiments on large language models. Our results demonstrate that these methods are non-orthogonal, and their combination leads to significant performance degradation. Importantly, we propose a novel approach Diagonal Adhesive Method (DAM), which can effectively combine the two methods and mitigate the performance loss. Our research provides deep insights into model compression and lays a solid theoretical and experimental foundation for future related studies. 

---
# Moir: Let the Model Direct Its Own Story for Robust Cross-Domain Knowledge Editing 

**Authors**: Jea Kwon, Jiwon Kim, Dong-kyum Kim, Meeyoung Cha  

**Link**: [PDF](https://arxiv.org/pdf/2607.20433)  

**Abstract**: While language models remain frozen at their training state, the world evolves continuously. Knowledge editing has emerged as a key alternative to full retraining, but its deployment is bottlenecked by the erosion of core capabilities: mathematical and programmatic reasoning collapse while encyclopedic recall remains intact. We trace this asymmetric degradation to a distributional mismatch. Covariance-based editors preserve only the subspaces spanned by their reference corpus, but fail to capture the operative distribution shaped by post-training such as SFT and DPO. Static external corpora, including Wikipedia and even the original pretraining mixture, cannot recover this shifted manifold. We propose Moir, which estimates the preservation covariance $C$ directly from the model itself by sampling from its own decoding distribution. Seeding generation with a single random vocabulary token bypasses the instruction-following templates that otherwise dominate sampled outputs, exposing the broader subspaces the model has internalized. Moir requires no external data and serves as a drop-in component for any covariance-based editor, a practical advantage given that the pre- and post-training corpora of most modern LLMs are not publicly accessible. Across OLMo-2, Llama-3.1, and Qwen-3 (7-8B), under both MEMIT and AlphaEdit and in batch and sequential regimes, Moir consistently extends preservation in the most vulnerable domains, most strikingly on Qwen3-8B after 20,000 AlphaEdit batch edits, it retains 79.9% GSM8K accuracy compared to 10.9% with the Wikipedia baseline. These results suggest that aligning the preservation distribution with the model's operative distribution is a key factor in non-destructive editing, and that the model itself may be the most accessible source of that distribution for deployed systems. 

---
# LLM-INSTRUCT at UZH Shared Task 2026: Constraint-Aware Retrieval and Selective Debate for Paragraph-Level Argument Mining 

**Authors**: Phuong Huu Vu Tran, Long Minh Vo, Son Nguyen Minh Le, Hoang Van  

**Link**: [PDF](https://arxiv.org/pdf/2607.20430)  

**Abstract**: We present LLM-INSTRUCT, the winning system for the UZH Shared Task at ArgMining 2026 on paragraph-level argument mining in UN and UNESCO resolutions. The task requires paragraph-type classification, prediction of a subset of 141 official tags, and directed relation prediction under a strict JSON schema setting using only open-weight models up to 8B parameters. We frame the task as constrained structured prediction. The system first narrows the candidate tag space with metadata-aware dense retrieval, then applies constrained decoding with per-dimension caps, escalates only uncertain cases to a three-agent debate branch, and finally validates the output schema. On the official leaderboard, LLM-INSTRUCT ranked 1st overall, with 1st in F1 and 5th in LLM-as-a-Judge. During development, our configuration search further improved Task 1b Micro-F1 from 35.83% to 40.08% while keeping the internal Task 2 score at 4.421. The main lesson is simple: reducing the decision space before generation improves both accuracy and submission robustness. Our code and supporting scripts are publicly available at: this https URL 

---
# More Is Not More: What Matters for Diversity in LLM Opinions? 

**Authors**: Qiyang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20429)  

**Abstract**: Large language models are increasingly used to simulate diverse human opinions in open-ended tasks such as synthetic surveys, focus group modeling, and public opinion prediction. However, LLM outputs exhibit systematic opinion homogenization. Practitioners have explored various interventions to increase diversity, but the landscape remains fragmented: different methods are evaluated in isolation with incomparable metrics, and in practice they are typically deployed and upgraded simultaneously, making it difficult to attribute gains to specific components. To advance a more scientific understanding of LLM output diversity, we design a factorial experiment that separates two primary intervention dimensions: input conditioning (operationalized through persona depth) and interaction architecture. We evaluate all conditions on 100 real-user open-ended questions across 7 models, measuring diversity with multiple complementary metrics. Our findings challenge several common assumptions. First, more persona detail does not monotonically increase diversity. The initial step of persona conditioning already captures the majority of the gain, while further elaboration with demographic detail does not consistently improve and can reduce diversity on some models. Second, rather than seeking a single best interaction architecture, we find that different architectures explore largely non-overlapping opinion regions. Combining multiple architectures yields broader coverage than optimizing any one. Third, commonly attempted low-cost alternatives such as raising sampling temperature and adding diversity instructions produce negligible effects compared to structured interventions. Overall, our work demonstrates that diversity is not a product of scaling along any single dimension, but is highly sensitive to the structural form and combination of interventions. 

---
# Is MoE Routing a Huffman Code? Discovering the Frequency-Diversity Law in Chain-of-Thought 

**Authors**: Ching-Chieh Tsao, Zhuoyi Lin, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20427)  

**Abstract**: Mixture-of-Experts architectures have revolutionized scaling, yet the underlying logic of their routing remains a black box. In this paper, we uncover a fundamental governing principle: MoE routing is not merely selection, but a manifestation of Huffman Coding. We introduce the Frequency-Diversity Law, revealing that state-of-the-art models, such as Phi-3.5-MoE and Gemma-4-27B-A4B, spontaneously act as information-theoretic engines. These models allocate sparse expert resources for common tokens while invoking high-diversity expert committees for rare, complex tasks found in chain-of-thought trajectories. However, we identify a critical redundancy trap in Qwen3.5-35B-A3B: when effective sparsity (k/E_eff) is sufficiently low, load-balancing inadvertently imposes functional redundancy, masking the underlying Huffman efficiency signal. To bridge this gap, we propose Subset Difference Pruning, a surgical strategy to eliminate functional duplicates. We demonstrate that pruning does not degrade reasoning; instead, it unleashes the model's latent Huffman efficiency, forcing the logic to collapse into streamlined, high-density paths. Our findings suggest that the next generation of MoEs should move beyond forced load-balancing toward Minimum Description Length (MDL) optimality, assigning shorter expert-routing codes to high-frequency information and longer, more diverse codes to low-frequency information, thereby transforming routing from a heuristic into a principled compression engine. 

---
# Knowledge Injection Exists in MoE? Exploring Expert-Aware Contrast Decoding in MoE for Mitigating LLMs'Hallucinations 

**Authors**: Xinyue Fang, Zhiliang Tian, Zhen Huang, Ziyi Pan, Zhihua Wen, Xi Wang, Quntian Fang, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.20426)  

**Abstract**: Existing LLM hallucination mitigation methods, including prompt engineering and model optimization, either hardly alter models'internal knowledge or have poor cross-domain generalization. Contrastive decoding mitigates hallucinations by using layer-wise differences in LLMs. However, prior studies only explore transformer-based models (e.g., GPT), ignoring other effective frameworks like mixture-of-experts (MoE) models. Since MoE alters the traditional transformer architecture, we conduct empirical studies to investigate whether similar layer-wise differences exist in MoEs. Our results show that they do not exist in MoE with shared experts; nevertheless, across different MoEs, higher layers exhibit distinct expert activation patterns between factual and non-factual outputs. Building on these, we propose EAACD, an expert-aware adaptive contrast decoding that uses expert differences in MoE's higher layers to mitigate hallucinations on QA tasks. EAACD splits high-layer experts into a higher-reliability group and several lower-reliability groups based on their confidence and consistency. It contrasts the higher-reliability group's prediction with each lower-reliability group's prediction to calibrate the model's original predictions. To strengthen this contrast, EAACD amplifies hallucinations from lower-reliability experts via attention and masking to provide stronger negative references. EAACD outperforms all baselines on four datasets. 

---
# From Attention to Frequency: Integration of Vision Transformer and FFT-ReLU for Enhanced Image Deblurring 

**Authors**: Syed Mumtahin Mahmud, Mahdi Mohd Hossain Noki, Prothito Shovon Majumder, Abdul Mohaimen Al Radi, Md. Haider Ali, Md. Mosaddek Khan  

**Link**: [PDF](https://arxiv.org/pdf/2511.10806)  

**Abstract**: Image deblurring is vital in computer vision, aiming to recover sharp images from blurry ones caused by motion or camera shake. While deep learning approaches such as CNNs and Vision Transformers (ViTs) have advanced this field, they often struggle with complex or high-resolution blur and computational demands. We propose a new dual-domain architecture that unifies Vision Transformers with a frequency-domain FFT-ReLU module, explicitly bridging spatial attention modeling and frequency sparsity. In this structure, the ViT backbone captures local and global dependencies, while the FFT-ReLU component enforces frequency-domain sparsity to suppress blur-related artifacts and preserve fine details. Extensive experiments on benchmark datasets demonstrate that this architecture achieves superior PSNR, SSIM, and perceptual quality compared to state-of-the-art models. Both quantitative metrics, qualitative comparisons, and human preference evaluations confirm its effectiveness, establishing a practical and generalizable paradigm for real-world image restoration. 

---
# Through-the-Earth Magnetic Induction Communication and Networking: A Comprehensive Survey 

**Authors**: Honglei Ma, Erwu Liu, Wei Ni, Zhijun Fang, Rui Wang, Yongbin Gao, Dusit Niyato, Ekram Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2510.14854)  

**Abstract**: Magnetic induction (MI) communication (MIC) has emerged as a promising candidate for underground communication networks due to its excellent penetration capabilities. Integration with Space-Air-Ground-Underground (SAGUI) networks in next-generation mobile communication systems requires a well-defined network architecture. A recent discovery in MIC research, MI fast fading, remains in its early stages and presents unique challenges. This paper provides a comprehensive survey on through-the-earth (TTE) MIC, covering MI applications, channel modeling, point-to-point MIC design, relay techniques, network frameworks, and emerging technologies. We compare various MIC applications to highlight TTE-specific challenges and review the principles of channel modeling, addressing both MI slow fading and MI fast fading, along with its potential impact on existing MIC theories. We conduct a fine-grained decomposition of MI channel power gain into four distinct physical parameters, and propose a novel geometric model to analyze MI fast fading. We also summarize MI relay techniques, examine crosstalk effects in relay and high-density networks, and explore key research tasks within the OSI framework for a holistic MI network protocol in SAGUI. To bridge the gaps identified, we propose a MIC framework that supports TCP/IP and Linux, enabling full implementation of existing and emerging MIC solutions. This framework empowers researchers to leverage Linux resources and deep learning platforms for accelerated development of MIC in SAGUI networks. Remaining research challenges, open issues, and promising novel techniques are further identified to advance MIC research. 

---
# Deblurring in the Wild: A Real-World Image Deblurring Dataset from Smartphone High-Speed Videos 

**Authors**: Syed Mumtahin Mahmud, Mahdi Mohd Hossain Noki, Prothito Shovon Majumder, Abdul Mohaimen Al Radi, Sudipto Das Sukanto, Afia Lubaina, Md. Mosaddek Khan  

**Link**: [PDF](https://arxiv.org/pdf/2506.19445)  

**Abstract**: We introduce the largest real-world image deblurring dataset constructed from smartphone slow-motion videos. Using 240 frames captured over one second, we simulate realistic long-exposure blur by averaging frames to produce blurry images, while using the temporally centered frame as the sharp reference. Our dataset contains over 42,000 high-resolution blur-sharp image pairs, making it approximately 10 times larger than widely used datasets, with 8 times the amount of different scenes, including indoor and outdoor environments, with varying object and camera motions. We benchmark multiple state-of-the-art (SOTA) deblurring models on our dataset and observe significant performance degradation, highlighting the complexity and diversity of our benchmark. Our dataset serves as a challenging new benchmark to facilitate robust and generalizable deblurring models. 

---
