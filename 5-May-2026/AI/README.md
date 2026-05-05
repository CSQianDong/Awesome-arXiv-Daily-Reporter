# Standing on the Shoulders of Giants: Stabilized Knowledge Distillation for Cross--Language Code Clone Detection 

**Authors**: Mohamad Khajezade, Fatemeh H. Fard, Mohamed Sami Shehata  

**Link**: [PDF](https://arxiv.org/pdf/2605.02860)  

**Abstract**: Cross-language code clone detection (X-CCD) is challenging because semantically equivalent programs written in different languages often share little surface similarity. Although large language models (LLMs) have shown promise for semantic clone detection, their use as black-box systems raises concerns about cost, reproducibility, privacy, and unreliable output formatting. In particular, compact open-source models often struggle to follow reasoning-oriented prompts and to produce outputs that can be consistently mapped to binary clone labels.
To address these limitations, we propose a knowledge distillation framework that transfers reasoning capabilities from DeepSeek-R1 into compact open-source student models for X-CCD. Using cross-language code pairs derived from Project CodeNet, we construct reasoning-oriented synthetic training data and fine-tune Phi3 and Qwen-Coder with LoRA adapters. We further introduce response stabilization methods, including forced conclusion prompting, a binary classification head, and a contrastive classification head, and evaluate model behavior using both predictive metrics and response rate.
Experiments on Python--Java, Rust--Java, Rust--Python, and Rust--Ruby show that knowledge distillation consistently improves the reliability of compact models and often improves predictive performance, especially under distribution shift. In addition, classification-head variants substantially reduce inference time compared to generation-based inference. Overall, our results show that reasoning-oriented distillation combined with response stabilization makes compact open-source models more practical and reliable for X-CCD detection. 

---
# HAAS: A Policy-Aware Framework for Adaptive Task Allocation Between Humans and Artificial Intelligence Systems 

**Authors**: Vicente Pelechanoa, Antoni Mestre, Manoli Albert, Miriam Gil  

**Link**: [PDF](https://arxiv.org/pdf/2605.02832)  

**Abstract**: Deciding how to distribute work between humans and AI systems is a central challenge in organisational design. Most approaches treat this as a binary choice, yet the operational reality is richer: humans and AI routinely share tasks or take complementary roles depending on context, fatigue, and the stakes involved. Governing that distribution -- balancing efficiency, oversight, and human capability -- remains an open problem. This paper presents Human-AI Adaptive Symbiosis (HAAS), an implemented framework for adaptive task allocation in software engineering and manufacturing. HAAS combines two coupled components: a rule-based expert system that enforces governance constraints before any learning occurs, and a contextual-bandit learner that selects among feasible collaboration modes from outcome feedback. Task-agent fit is represented through five auditable cognitive dimensions and a five-mode autonomy spectrum -- from human-only to fully autonomous -- embedded in a reproducible benchmark spanning both domains. Three empirical findings emerge. First, governance is not a binary switch but a tunable design variable: tighter constraints predictably convert autonomous AI assignments into supervised collaborations, with domain-specific costs and benefits. Second, in manufacturing, stronger governance can improve operational performance and reduce fatigue simultaneously -- a workload-buffering effect that contradicts the usual framing of governance as pure overhead. Third, no single governance setting dominates across all contexts; moderate governance becomes increasingly competitive as the learner accumulates experience within the governed action space. Together, these findings position HAAS as a pre-deployment workbench for comparing and inspecting human--AI allocation policies before organisational commitment. 

---
# Compress Then Adapt? No, Do It Together via Task-aware Union of Subspaces 

**Authors**: Jingze Ge, Yun Liu, Xue Geng, Wanqi Dong, Wang Zhe Mark, Min Wu, Xulei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02829)  

**Abstract**: Adapting large pretrained models to diverse tasks is now routine, yet the two dominant strategies of parameter-efficient fine-tuning (PEFT) and low-rank compression are typically composed in sequence. This decoupled practice first compresses and then fine-tunes adapters, potentially misaligning the compressed subspace with downstream objectives and squandering a global parameter budget. To overcome this limitation, we introduce JACTUS (Joint Adaptation and Compression with a Task-aware Union of Subspaces), a single framework that unifies compression and adaptation. From a small calibration set, JACTUS estimates input and pre-activation gradient covariances, forms their orthogonal union with the pretrained weight subspace, performs a projected low-rank approximation inside this union, allocates rank globally by marginal gain per parameter, and trains only a compact core matrix. This explicitly mitigates the potential misalignment between the compressed subspace and downstream objectives by coupling the directions preserved for compression with those required for adaptation, yielding a deployable low-rank model that avoids retaining full frozen weights while enabling fast and robust tuning. On vision, JACTUS attains an average 89.2% accuracy on ViT-Base across eight datasets at 80% retained parameters, surpassing strong 100% PEFT baselines (e.g., DoRA 87.9%). On language, JACTUS achieves an 80.9% average on Llama2-7B commonsense QA at the same 80% retained-parameter budget, outperforming 100% PEFT (e.g., DoRA 79.7%) and exceeding prior compress-then-finetune pipelines under the same ratained-parameter budget. We will release code. 

---
# First-Order Efficiency for Probabilistic Value Estimation via A Statistical Viewpoint 

**Authors**: Ziqi Liu, Kiljae Lee, Yuan Zhang, Weijing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02827)  

**Abstract**: Probabilistic values, including Shapley values and semivalues, provide a model-agnostic framework to attribute the behavior of a black-box model to data points or features, with a wide range of applications including explainable artificial intelligence and data valuation. However, their exact computation requires utility evaluations over exponentially many coalitions, making Monte Carlo approximation essential in modern machine learning applications. Existing estimators are often developed through different identification strategies, including weighted averages, self-normalized weighting, regression adjustment, and weighted least squares. Our key observation is that these seemingly distinct constructions share a common first-order error structure, in which the leading term is an augmented inverse-probability weighted influence term determined by the sampling law and a working surrogate function. This first-order representation yields an explicit expression for the leading mean squared error (MSE), which characterizes how the sampling law and the surrogate jointly determine statistical efficiency. Guided by this criterion, we propose an Efficiency-Aware Surrogate-adjusted Estimator (EASE) that directly chooses the sampling law and surrogate to minimize the first-order MSE. We demonstrate that EASE consistently outperforms state-of-the-art estimators for various probabilistic values. 

---
# SCPRM: A Schema-aware Cumulative Process Reward Model for Knowledge Graph Question Answering 

**Authors**: Jiujiu Chen, Yazheng Liu, Sihong Xie, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.02819)  

**Abstract**: Large language models excel at complex reasoning, yet evaluating their intermediate steps remains challenging. Although process reward models provide step-wise supervision, they often suffer from a risk compensation effect, where incorrect steps are offset by later correct ones, assigning high rewards to flawed reasoning paths. This issue is further exacerbated in knowledge graph (KG) reasoning, as there may exist multiple paths between the start and end entities in the KGs, and a risky step can make the reasoning path flawed. Those limitations are problematic in risk-sensitive tasks such as medical and legal KG reasoning. To address the issues, we propose a Schema-aware Cumulative Process Reward Model (SCPRM) that evaluates reasoning paths by conditioning on the reasoning prefix , and incorporating schema distance between current reasoning step and the implicit target parsed from the query, which provides cumulative and future rewards to guide the path explorations. We further integrate SCPRM into Monte Carlo Tree Search (MCTS) as SCPRM-MCTS to conduct multi-hop reasoning on KGs for question answering (QA) tasks. Across medical and legal KGQA and CWQ, SCPRM-MCTS improves the performance of Hits@k by an average of 1.18% over strong baselines, demonstrating more accurate and risk-sensitive reasoning evaluation. 

---
# AIs and Humans with Agency 

**Authors**: David Mumford  

**Link**: [PDF](https://arxiv.org/pdf/2605.02810)  

**Abstract**: This paper compares agency in humans with potential agency in AI programs. Human agency takes many years to develop, as the frontal lobe is activated. Early attempts to endow LLMs agency have met serious obstacles. Progress requires a new architecture where actions and plans are formulated jointly with the human actors in each real world setting. 

---
# When Audio-Language Models Fail to Leverage Multimodal Context for Dysarthric Speech Recognition 

**Authors**: Pehuén Moure, Niclas Pokel, Bilal Bounajma, Yingqiang Gao, Roman Boehringer, Longbiao Cheng, Shih-Chii Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02782)  

**Abstract**: Automatic speech recognition (ASR) systems remain brittle on dysarthric and other atypical speech. Recent audio-language models raise the possibility of improving performance by conditioning on additional clinical context at inference time, but it is unclear whether these models can make use of such information. We introduce a benchmark built on the Speech Accessibility Project (SAP) dataset that tests whether diagnosis labels, clinician-derived speech ratings, and progressively richer clinical descriptions improve transcription accuracy for dysarthric speech. Across matched comparisons on nine models, we find that current models do not meaningfully use this context: diagnosis-informed and clinically detailed prompts yield negligible improvements and often degrade word error rate. We complement the prompting analysis with context-dependent fine-tuning, showing that LoRA adaptation with a mixture of clinical prompt formats achieves a WER of 0.066, a 52% relative reduction over the frozen baseline, while preserving performance when context is unavailable. Subgroup analyses reveal significant gains for Down syndrome and mild-severity speakers. These results clarify where current models fall short and provide a testbed for measuring progress toward more inclusive ASR. 

---
# Fine-Grained Graph Generation through Latent Mixture Scheduling 

**Authors**: Nidhi Vakil, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2605.02780)  

**Abstract**: Structure aware graph generation aims to generate graphs that satisfy given topological properties. It has applications in domains such as drug discovery, social network modeling, and knowledge graph construction. Unlike existing methods that only provide coarse control over graph properties, we introduce a novel conditional variational autoencoder for fine-grained structural control in graph generation. The approach refines the decoder's latent space by dynamically aligning graph- and property-driven representations to improve both graph fidelity and control satisfaction. Specifically, the approach implements a mixture scheduler that progressively integrates graph and control priors. Experiments on five real-world datasets show the efficacy of the proposed model compared to recent baselines, achieving high generation quality while maintaining high controllability. 

---
# U-Define: Designing User Workflows for Hard and Soft Constraints in LLM-Based Planning 

**Authors**: Christine P Lee, Xinyu Jessica Wang, Aws Albarghouthi, David Porfirio, Bilge Mutlu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02765)  

**Abstract**: LLMs are increasingly used for end-user task planning, yet their black-box nature limits users' ability to ensure reliability and control. While recent systems incorporate verification techniques, it remains unclear how users can effectively apply such rigid constraints to represent intent or adapt to real-world variability. For example, prior work finds that hard-only constraints are too rigid, and numeric flexibility weights confuse users. We investigate how interaction workflows can better support users in applying constraints to guide LLM-generated plans, examining whether abstracting strictness into high-level types (i.e., hard and soft) paired with distinct verification mechanisms helps users more reliably express and align intent. We present U-Define, a system that lets users define constraints in natural language and categorize them as either hard rules that must not be violated or soft preferences that allow flexibility. U-Define verifies these types through complementary methods: formal model checking for hard constraints and LLM-as-judge evaluation for soft ones. Through a technical evaluation and user studies with general and expert participants, we find that user-defined constraint types improve perceived usefulness, performance, and satisfaction while maintaining usability. These findings provide insights for designing flexible yet reliable constraint-based workflows. 

---
# Mitigating Misalignment Contagion by Steering with Implicit Traits 

**Authors**: Maria Chang, Ronny Luss, Miao Lui, Keerthiram Murugesan, Karthikeyan Ramamurthy, Djallel Bouneffouf  

**Link**: [PDF](https://arxiv.org/pdf/2605.02751)  

**Abstract**: Language models (LMs) are increasingly used in high-stakes, multi-agent settings, where following instructions and maintaining value alignment are critical. Most alignment research focuses on interactions between a single LM and a single user, failing to address the risk of misaligned behavior spreading between multiple LMs in multi-turn interactions. We find evidence of this phenomenon, which we call misalignment contagion, across multiple LMs as they engage multi-turn conversational social dilemma games. Specifically, we find that LMs become more anti-social after gameplay and that this effect is intensified when other players are steered to act maliciously. We explore different steering techniques to mitigate such misalignment contagion and find that reinforcing an LM's system prompt is insufficient and often harmful. Instead, we propose steering with implicit traits: a technique that intermittently injects system prompts with statements that reinforce an LMs initial traits and is more effective than system prompt repetition at keeping models in line with their initial pro-social behaviors. Importantly, this method does not require access to model parameters or internal model states, making it suitable for increasingly common use cases where complex multi-agent workflows are being designed with black box models. 

---
# Triple Spectral Fusion for Sensor-based Human Activity Recognition 

**Authors**: Ye Zhang, Longguang Wang, Qing Gao, Chaocan Xiang, Mohammed Bennamoun, Yulan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.02743)  

**Abstract**: The field of sensor-based human activity recognition (HAR) mainly uses posture, motion and context data of Inertial Measurement Units (IMUs) to identify daily activities. Despite the advancements in learning-based methods, it is challenging to perform information fusion from the temporal perspective due to the complexities in fusing heterogeneous sensor data and establishing long-term context correlations. This paper proposes a novel triple spectral fusion framework tailored for HAR. First, we develop an adaptive complementary filtering technique for noise suppression and organize each IMU's sensors into posture and motion modality nodes. Given that IMU nodes form a dynamic heterogeneous graph, we then apply adaptive filtering within the graph Fourier domain to merge both homogeneous and heterogeneous node information. Furthermore, an adaptive wavelet frequency selection approach is implemented to suppress context redundancy and shorten the length of features. This approach enhances both timestamp-based graph aggregation and the correlation of long-term contexts. Our framework uses adaptive filtering in the Fourier, graph Fourier, and wavelet domains, enabling effective multi-sensor fusion and context correlation. Extensive experiments on ten benchmark datasets demonstrate the superior performance of our framework. Project page: this https URL. 

---
# Foundation Models to Unlock Real-World Evidence from Nationwide Medical Claims 

**Authors**: Fan Ma, Yuntian Liu, Xiang Lan, Weipeng Zhou, Jun Ni, Mauro Giuffrè, Lingfei Qian, Xueqing Peng, Yujia Zhou, Ruey-Ling Weng, Huan He, Lu Li, Qingyu Chen, Andrew Loza, Laila Rasmy, Degui Zhi, Yuan Lu, Chenjie Zeng, Joshua C Denny, Lee Schwamm, Daniella Meeker, Lucila Ohno-Machado, Yong Chen, Hua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02740)  

**Abstract**: Evidence derived from large-scale real-world data (RWD) is increasingly informing regulatory evaluation and healthcare decision-making. Administrative claims provide population-scale, longitudinal records of healthcare utilization, expenditure, and detailed coding of diagnoses, procedures, and medications, yet their potential as a substrate for healthcare foundation models remains largely unexplored. Here we present ReClaim, a generative transformer trained from scratch on 43.8 billion medical events from more than 200 million enrollees in the MarketScan claims data spanning 2008-2022. ReClaim models longitudinal trajectories across diagnoses, procedures, medications, and expenditure, and was scaled to 140 million, 700 million, and 1.7 billion parameters. Across over 1,000 disease-onset prediction tasks, ReClaim achieved a mean AUC of 75.6%, substantially outperforming disease-specific LightGBM (66.3%) and the transformer-based Delphi model (69.4%), with the largest gains for rare diseases. These advantages held across retrospective and prospective evaluations and in external validation on two independent datasets. Performance improved monotonically with scale, and post-training added 13.8 percentage points over pre-training alone. Beyond disease prediction, ReClaim captured financial outcomes and improved real-world evidence (RWE) analyses: for healthcare expenditure forecasting it increased explained variance from 0.28 to 0.37 relative to LightGBM, and in a target trial emulation it reduced systematic bias by 72% on average relative to Delphi. Together, these results establish administrative claims as a scalable substrate for healthcare foundation models and show that learned representations generalize across time periods and data sources, supporting disease surveillance, expenditure forecasting, and RWE generation. 

---
# AI and Open-data Driven Scalable Solar Power Profiling 

**Authors**: Shiliang Zhang, Sabita Maharjan, Damla Turgut  

**Link**: [PDF](https://arxiv.org/pdf/2605.02738)  

**Abstract**: Solar photovoltaic (PV) deployment is expanding rapidly, yet detailed, up-to-date information on the spatial distribution and capacity of rooftop PV remains limited. This paper presents an open, scalable framework for detecting solar panels from open data and generating city-level solar power profiles. We leverage foundation vision AI models to detect solar panel geometries from open-source satellite imagery. This avoids manual data labeling and case-specific model training while maintaining robustness across heterogeneous imagery. Detected solar panels are converted into georeferenced polygons, yielding spatially explicit and incrementally extensible inventories. By integrating open weather data, we translate panel footprints into regional solar power profiles. The framework reduces dependency on proprietary imagery, manual labeling, and closed-source models, and offers a transparent and scalable approach for solar planning and analysis. We released the data and an API resulted from this work. For any user-specified building location, our API retrieves aerial imagery, detects rooftop solar panels, and returns georeferenced polygons. This empowers researchers and developers to scan user-defined areas to build solar panel maps and associated solar production profiles, thus facilitating advanced analysis like distributed solar production integration, local power flow optimization, energy tariff design, and infrastructure planning. 

---
# Coherent Hierarchical Multi-Label Learning to Defer for Medical Imaging 

**Authors**: Joshua Strong, Pramit Saha, Emma Sun, Helen Higham, Alison Noble  

**Link**: [PDF](https://arxiv.org/pdf/2605.02734)  

**Abstract**: Learning to Defer (L2D) enables a model to predict autonomously or defer to an expert, but prior work largely assumes flat label spaces. We study the first L2D setting with hierarchical multi-label decisions, motivated by medical-imaging workflows in which findings are organised by clinical taxonomies. In this setting, deferral is a delegation action rather than a label assignment, so treating it as an independent per-label decision can produce deferral incoherence, including taxonomic contradictions, delegation violations, and deferrals of labels already implied by the model's own assertions. We formalise coherent hierarchical deferral under a Selective-Exclusion handoff contract, characterise the Bayes-optimal coherent deferral rule, and show that even nodewise Bayes L2D can be action-incoherent. We then propose two remedies: exact coherent projection, a dynamic-programming decoder over the coherent action set, and Taxonomic Belief Propagation (TBP) with Recursive Policy Optimisation (RPO), a contract-aware joint action model trained through the same recursion used at inference. Across real-reader and controlled-expert medical-imaging benchmarks, naive binary-relevance L2D exhibits non-trivial incoherence. Projection removes it exactly, and fast TBP+RPO drives incoherence near zero while retaining strong utility. 

---
# ORPilot: A Production-Oriented Agentic LLM-for-OR Tool for Optimization Modeling 

**Authors**: Guangrui Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.02728)  

**Abstract**: This paper presents ORPilot, an open-source agentic AI system that translates real-world business problems into solver-ready optimization models. Unlike academic LLM-for-OR tools that assume clean problem specifications with preformatted inline data, ORPilot is designed for production conditions: ambiguous descriptions, large-scale raw operational data, and the need for portability across solver backends. The system introduces four novel components: (1) a conversational interview agent to elicit complete problem specifications, (2) a data collection agent that retrieves data independently of prompts, (3) a parameter computation agent to bridge raw tabular data and model-ready parameters, and (4) a solver-agnostic Intermediate Representation (IR) for deterministic, zero-LLM-call recompilation to Gurobi, CPLEX, PuLP, Pyomo, or OR-Tools solvers. Additionally, self-correcting retry loops utilize solver tracebacks for targeted repairs. ORPilot represents the first attempt to target production-level business problems rather than textbook operations research (OR) cases. Evaluation on real-world problems demonstrates promising results. When tested against traditional academic benchmarks: IndustryOR, NL4OPT and NLP4LP, ORPilot outperformed state-of-the-art tools in accuracy on the IndustryOR benchmark and delivered comparable performance on NL4OPT and NLP4LP. 

---
# An Empirical Study of Agent Skills for Healthcare: Practice, Gaps, and Governance 

**Authors**: Gelei Xu, Ningzhi Tang, Xueyang Li, Toby Jia-Jun Li, Zhi Zheng, Wei Jin, Yiyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.02709)  

**Abstract**: Healthcare automation is shaped by local procedures and organizational constraints, so agent capabilities rarely transfer unchanged across settings. Agent skills, self-contained directories that package reusable procedures for AI agents, are emerging as a procedural layer for adapting healthcare agents across diverse healthcare settings. We present the first empirical analysis of healthcare agent skills, drawing on 557 healthcare-related skills filtered from 58,159 public skills on ClawHub and annotated along ten dimensions covering function, deployment context, autonomy, and safety. We find that public healthcare skills emphasize patient-facing workflow automation and monitoring rather than the diagnostic and treatment-oriented tasks foregrounded in healthcare-agent research; coverage of the healthcare lifecycle and specialized clinical inputs remains uneven; and general technical risk does not reliably capture clinical risk. These findings position healthcare skills as a procedural layer not yet addressed by current benchmarks and risk frameworks. 

---
# Hybrid Inspection and Task-Based Access Control in Zero-Trust Agentic AI 

**Authors**: Majed El Helou, Benjamin Ryder, Chiara Troiani, Jean Diaconu, Hervé Muyal, Marcelo Yannuzzi  

**Link**: [PDF](https://arxiv.org/pdf/2605.02682)  

**Abstract**: Authorizing Large Language Model (LLM)-driven agents to dynamically invoke tools and access protected resources introduces significant security risks, and the risks grow dramatically as agents engage in multi-turn conversations and scale toward distributed collaboration. A compromised or malicious agentic application can tamper with tool calls, falsify results, or request permissions beyond the scope of the subject's intended tasks, which could go unnoticed with current delegated authorization flows given their lack of visibility into the original subject's intent. In light of this, we make the following contributions towards Continuous Agent Semantic Authorization (CASA). First, we propose a hybrid runtime enforcement model that combines deterministic and semantic controls enabled by a zero-trust interception layer. Five deterministic controls enforce structural and data-integrity guarantees over the message flow, while a semantic inspection layer evaluates whether tool call choices align with the intended tasks commissioned to the agent. Second, differently from prior Task-Based Access Control (TBAC) techniques that operate on single-turn interactions, we decompose the semantic layer into two stages: i) a task-extraction step that distills the subject's objectives from multi-turn conversations at the interception layer, and ii) a task-tool semantic matching step at the authorization server that evaluates whether the requested tools are appropriate for the extracted tasks. Third, we extend the ASTRA dataset that we introduced in a prior work, by generating novel conversation-tool datasets with multi-turn interactions containing relevant and irrelevant tool calls for a given task. Lastly, we provide the first experimental results for TBAC under multi-turn conversations. 

---
# The 2026 ACII Dyadic Conversations (DaiKon) Workshop & Challenge 

**Authors**: Panagiotis Tzirakis, Alice Baird, Jeffrey Brooks, Emilia Parada-Cabaleiro, Lukas Stappen, Sharath Rao, Theo Lebryk, Jakub Piotr Clapa, Jens Madsen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02672)  

**Abstract**: The 2026 ACII Dyadic Conversations (ACII-DaiKon) Workshop & Challenge introduces a benchmark for modeling interpersonal affect and social dynamics in dyadic conversations. Although conversational affect modeling has advanced rapidly, most benchmarks remain speaker-centric and underrepresent coupled, time-evolving processes between partners, including directional influence, conversational timing coordination, and rapport development. To address this gap, ACII-DaiKon presents three coordinated sub-challenges built on a shared dataset: (1) directional interpersonal influence prediction, (2) turn-taking prediction (next-speaker and time-to-next-speech), and (3) rapport trajectory prediction across full interactions.
The challenge is built on the Hume-DaiKon dataset, comprising 945 dyadic conversations (743.4 hours of audiovisual data) collected under naturalistic conditions across five languages. The benchmark supports multimodal modeling, temporal reasoning, and cross-context generalization through fixed train/validation/test splits, standardized metrics, and released baseline systems. Evaluation uses Concordance Correlation Coefficient (CCC), Pearson correlation, Macro-F1, and Mean Absolute Error (MAE) depending on the sub-challenge.
Baseline experiments establish initial reference performance, with best test results of 0.40 CCC and 0.50 Pearson for influence prediction, 0.66 Macro-F1 and 1.50~s MAE for turn-taking, and 0.68 CCC and 0.70 Pearson for rapport trajectory modeling. These results indicate that while current methods capture coarse dyadic patterns, robust modeling of directional dependence and long-horizon interpersonal dynamics remains challenging. The workshop provides a shared platform for rigorous comparison and cross-disciplinary discussion on data validity, evaluation protocols, and culturally aware modeling for dyadic interaction. 

---
# An explainable hypothesis-driven approach to Drug-Induced Liver Injury with HADES 

**Authors**: Maciej Wisniewski, Bartosz Topolski, Pawel Dabrowski-Tumanski, Dariusz Plewczynski, Tomasz Jetka  

**Link**: [PDF](https://arxiv.org/pdf/2605.02669)  

**Abstract**: Drug-induced liver injury (DILI) remains a leading cause of late-stage clinical trial attrition. However, existing computational predictors primarily rely on binary classification, a framing that limits generalization and yields no mechanistic insight to guide translational decisions. We argue that DILI prediction is better posed as an explainable hypothesis-generation problem.
To support this shift, we introduce the DILER Benchmark, a dataset that extends beyond binary labels by augmenting a curated set of molecules with mechanistic hepatotoxicity hypotheses derived from biomedical literature. We further present HADES, an agentic system designed to generate transparent and auditable reasoning traces. By combining molecular-level predictions, metabolite decomposition, structural understanding, and toxicity pathway evidence, HADES mechanistically assesses DILI risk.
Evaluated on the DILER Benchmark, HADES outperforms existing models in binary classification, achieving a ROC-AUC of 0.68 on the Test Set and 0.59 on the challenging Post-2021 Set, compared with 0.63 and 0.50 for DILI-Predictor, respectively. More importantly, we establish a baseline for mechanistic hypothesis generation, where HADES achieves a Hypothesis Alignment Fuzzy Jaccard Index of 0.16. This result underscores the inherent complexity of the task while highlighting the need for advanced explainable approaches in predictive toxicology. 

---
# AcademiClaw: When Students Set Challenges for AI Agents 

**Authors**: Junjie Yu, Pengrui Lu, Weiye Si, Hongliang Lu, Jiabao Wu, Kaiwen Tao, Kun Wang, Lingyu Yang, Qiran Zhang, Xiuting Guo, Xuanyu Wang, Yang Wang, Yanjie Wang, Yi Yang, Zijian Hu, Ziyi Yang, Zonghan Zhou, Binghao Qiang, Borui Zhang, Chenning Li, Enchang Zhang, Feifan Chen, Feng Jian, Fengyin Sun, Hao Qiu, Hao Zheng, Haoran Zhu, Hongyu Liu, Jianbin Deng, Jiaxin Song, Jiaying Chi, Jiayou Shi, Jie Fang, Jinghui Zhong, Jingyu Zhou, Jinze Li, Junfeng Yi, Junyan Yu, Junzhi Xue, Ni Song, Pengyi Chen, Qi Chen, Quansheng Li, Rui Tao, Shenghai Gong, Shenhang Lu, Tianqi Shen, Tianxiang Zhu, Tiehan Kang, Tingyu Li, Wendi Wu, Xiao Shen, Xiao Zhou, Xiaotao Zhang, Xinrong Li, Xuankun Yang, Xun Zhang, Yan Li, Ye Lu, Yi Wang, Yibo Zhou, Yichi Zhang, Yihao Sun, Yijun Huang, Yixin Zhu, Yixuan Wu, Yuchen Sun, Yue Wu, Yuheng Sun, Yukun Li, Yutian Tu, Yuxuan Qin, Yuzhuo Wu, Zeyu Li, Zhengyu Lou, Zhenning Ran, Zizhu He, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02661)  

**Abstract**: Benchmarks within the OpenClaw ecosystem have thus far evaluated exclusively assistant-level tasks, leaving the academic-level capabilities of OpenClaw largely unexamined. We introduce AcademiClaw, a bilingual benchmark of 80 complex, long-horizon tasks sourced directly from university students' real academic workflows -- homework, research projects, competitions, and personal projects -- that they found current AI agents unable to solve effectively. Curated from 230 student-submitted candidates through rigorous expert review, the final task set spans 25+ professional domains, ranging from olympiad-level mathematics and linguistics problems to GPU-intensive reinforcement learning and full-stack system debugging, with 16 tasks requiring CUDA GPU execution. Each task executes in an isolated Docker sandbox and is scored on task completion by multi-dimensional rubrics combining six complementary techniques, with an independent five-category safety audit providing additional behavioral analysis. Experiments on six frontier models show that even the best achieves only a 55\% pass rate. Further analysis uncovers sharp capability boundaries across task domains, divergent behavioral strategies among models, and a disconnect between token consumption and output quality, providing fine-grained diagnostic signals beyond what aggregate metrics reveal. We hope that AcademiClaw and its open-sourced data and code can serve as a useful resource for the OpenClaw community, driving progress toward agents that are more capable and versatile across the full breadth of real-world academic demands. All data and code are available at this https URL. 

---
# Deciphering Shortcut Learning from an Evolutionary Game Theory Perspective 

**Authors**: Xiayang Li, Kuo Gai, Shihua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02658)  

**Abstract**: Shortcut learning causes deep learning models to rely on non-essential features within the data. However, its formation in deep neural network training still lacks theoretical understanding. In this paper, we provide a formal definition of core and shortcut features and employ evolutionary game theory to analyze the origins of shortcut bias by modeling data samples as players and their corresponding neural tangent features as strategies, assuming the existence of core and shortcut subnetworks. We find that gradient descent (GD) and stochastic gradient descent (SGD) lead to two distinct stochastically stable states, each corresponding to a different strategy. The former primarily optimizes the shortcut subnetwork, while the latter primarily optimizes the core subnetwork. We investigate the influence of these strategies on shortcut bias through a continuous stochastic differential equation, and reveal the impact of data noise and optimization noise on the formation of shortcut bias. In brief, our work employs evolutionary game theory to characterize the dynamics of shortcut bias formation and provides a theoretical view on its mitigation. 

---
# Trustworthy AI Suffers from Invariance Conflicts and Causality is The Solution 

**Authors**: Ruta Binkyte, Ivaxi Sheth, Zhijing Jin, Mohammad Havaei, Bernhard Schölkopf, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2605.02640)  

**Abstract**: As artificial intelligence (AI), including machine learning (ML) models and foundation models (FMs), is increasingly deployed in high-stakes domains, ensuring their trustworthiness has become a central challenge. However, the core trustworthy AI objectives, such as fairness, robustness, privacy, and explainability, are hard to achieve simultaneously, especially while preserving utility. This position paper argues that causality is necessary to understand and balance trade-offs in performance and multiple objectives of trustworthy AI. We ground our arguments in re-interpreting trustworthy AI trade-offs as incompatible invariance requirements under different changes to the data-generating process. We then illustrate that causality provides a unifying framework for understanding how trade-offs in trustworthy AI arise, and how they can be softened or resolved through selective invariance. This perspective applies to both classical ML models and large-scale FMs. Our paper discusses how causal assumptions may be applied explicitly or implicitly in modern large-scale systems. Finally, we outline open challenges and opportunities for using causality to build more trustworthy AI. 

---
# SCGNN: Semantic Consistency enhanced Graph Neural Network Guided by Granular-ball Computing 

**Authors**: Genhao Tian, Taihua Xu, Shuyin Xia, Qinghua Zhang, Jie Yang, Jianjun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02617)  

**Abstract**: Capturing semantic consistency among nodes is crucial for effective graph representation learning. Existing approaches typically rely on $k$-nearest neighbors ($k$NN) or other node-level full search algorithms (FSA) to mine semantic relationships via exhaustive pairwise similarity computation, which suffer from high computational complexity and rigid neighbor selection, limiting scalability and introducing noisy connections. In this paper, we propose the Semantic Consistency enhanced Graph Neural Network (SCGNN), a novel plug-and-play framework that leverages granular-ball computing (GBC) to efficiently capture semantic consistency in a scalable manner. Unlike node-level FSA methods, SCGNN models group-level semantic structure by adaptively partitioning nodes into granular balls, significantly reducing computational cost while improving robustness to noise. To effectively utilize the discovered group-level semantic consistency, we design a dual enhancement strategy. Specifically, (1) a structure enhancement module constructs an anchor-based graph structure, where each anchor is a virtual node representing the group-level semantic carried by a granular ball, then injecting group-level semantic information into the graph structure; and (2) a supervision enhancement module performs label consistency checking (LCC) by combining GBC predictions with model-generated pseudo-labels, thereby producing more reliable supervision signals. SCGNN is compatible with various GNN backbones. During the forward propagation of SCGNN, the vanilla graph and the augment graph are jointly encoded, and their predictions are fused; during the backpropagation, the supervision enhancement module provides enhanced supervision signals to guide parameter updates. 

---
# Counterfactual Reasoning in Automated Planning 

**Authors**: Alberto Pozanco, Daniel Borrajo, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2605.02603)  

**Abstract**: Automated planning traditionally assumes that all aspects of a planning task (initial state, goals, and available actions) are fully specified in advance, an approach well-suited to domains with fixed rules and deterministic execution. However, real-world planning often requires flexibility, allowing for deviations from the original task parameters in response to unforeseen circumstances or to improve outcomes. This paper surveys existing works on counterfactual reasoning in automated planning, categorizing them by what elements are changed, when the reasoning is triggered, and why and how these changes are made. We conclude by discussing key findings and outlining open research questions to guide future work in this area. 

---
# Foundation-Model-Based Agents in Industrial Automation: Purposes, Capabilities, and Open Challenges 

**Authors**: Vincent Henkel, Felix Gehlhoff, David Kube, Asaad Almutareb, Luis Cruz, Bernd Hellingrath, Philip Koch, Christoph Legat, Florian Mohr, Michael Oberle, Felix Ocker, Thorsten Schoeler, Mario Thron, Nico Andre Töpfer, Lucas Vogt, Yuchen Xia  

**Link**: [PDF](https://arxiv.org/pdf/2605.02592)  

**Abstract**: Foundation models, particularly large language models, are increasingly integrated into agent architectures for industrial tasks such as decision support, process monitoring, and engineering automation. Yet evidence on their purposes, capabilities, and limitations remains fragmented across domains. This work examines how mature foundation-model-based agent systems are in industrial contexts, how their functional profile differs from conventional agent systems, and which limitations persist. A systematic literature survey following the PRISMA 2020 guideline is presented, screening 2,341 publications and synthesising a corpus of 88 publications through a structured coding scheme. The results show that reported systems are predominantly at prototype and early validation stages (75.0% at TRL 4-6), with deployment-oriented evidence remaining rare (9.1%). Operational goals are most frequently positioned in user assistance, monitoring, and process optimisation, while conventional production-control purposes such as planning and scheduling are less prominent. Compared with an established baseline for industrial agent systems, the capability profile reveals substantial gains in human interaction (+37%) and dealing with uncertainty (+35%), but a pronounced deficit in negotiation (-39%). The most widely reported limitations concern lack of generalization, hallucination and output instability, data scarcity, and inference latency. A working definition of foundation-model-based industrial agents is also proposed, bridging conventional agent theory, automation-engineering standards, and the foundation-model paradigm. 

---
# Universal Smoothness via Bernstein Polynomials: A Constructive Approximation Approach for Activation Functions 

**Authors**: Wentao Zhang, Yutong Zhang, Yifan Zhu, Wentao Mo  

**Link**: [PDF](https://arxiv.org/pdf/2605.02591)  

**Abstract**: The efficacy of deep neural networks is heavily reliant on the design of non-linear activation functions, yet existing approaches often struggle to balance optimization stability with computational efficiency. While piecewise linear functions offer inference speed, they suffer from optimization instability due to non-differentiability at the origin, whereas smooth counterparts typically incur significant computational overhead through their reliance on transcendental operations. To address these limitations, this paper proposes a general smoothing framework based on constructive approximation theory and introduces the Bernstein Linear Unit (BerLU). This novel activation function utilizes Bernstein polynomials to construct a differentiable quadratic transition region that effectively eliminates singularities while maintaining a piecewise linear structure. Theoretical analysis demonstrates that the proposed method guarantees strictly continuous differentiability and a non-expansive Lipschitz constant of one, which ensures stable gradient propagation and prevents the gradient explosion problems common in deep architectures. Comprehensive empirical evaluations across representative Vision Transformer and Convolutional Neural Network architectures confirm that this approach consistently outperforms state-of-the-art baselines on standard image classification benchmarks while delivering superior computational and memory efficiency. 

---
# On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length 

**Authors**: Sunghwan Kim, Junhee Cho, Beong-woo Kwak, Taeyoon Kwon, Liang Wang, Nan Yang, Xingxing Zhang, Furu Wei, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2605.02572)  

**Abstract**: Large language models (LLMs) have shown promise as interactive agents that solve tasks through extended sequences of environment interactions. While prior work has primarily focused on system-level optimizations or algorithmic improvements, the role of task horizon length in shaping training dynamics remains poorly understood. In this work, we present a systematic empirical study that examines horizon length through controlled task constructions. Specifically, we construct controlled tasks in which agents face identical decision rules and reasoning structures, but differ only in the length of action sequences required for successful completion. Our results reveal that increasing horizon length alone constitutes a training bottleneck, inducing severe training instability driven by exploration difficulties and credit assignment challenges. We demonstrate that horizon reduction is a key principle to address this limitation, stabilizing training and achieving better performance in long-horizon tasks. Moreover, we find that horizon reduction is related to stronger generalization across horizon lengths: models trained under reduced horizons generalize more effectively to longer-horizon variants at inference time, a phenomenon we refer to as horizon generalization. 

---
# Double Rectified Linear Unit-based Modular Semantics for Quantitative Bipolar Argumentation Framework 

**Authors**: Gianvincenzo Alfano, Sergio Greco, Lucio La Cava, Francesco Parisi, Irina Trubitsyna  

**Link**: [PDF](https://arxiv.org/pdf/2605.02551)  

**Abstract**: Quantitative Bipolar Argumentation Frameworks (QBAFs) provide an alternative approach to computing argument acceptability in Bipolar Argumentation Frameworks (BAFs). Each argument is assigned an initial strength, which is then updated to a final strength by considering the influence of both its attackers and supporters. Over the years, several semantics have been proposed to compute argument acceptability in QBAFs, yet they often yield divergent or counterintuitive results, even for simple acyclic cases. We introduce novel gradual semantics for QBAFs that address these limitations, producing results that align more closely with intuitive expectations, while satisfying established rationality postulates from the literature. Furthermore, we study its convergence behavior, proving that it converges not only for acyclic QBAFs but also for broader classes of cyclic frameworks. 

---
# Strategy-Aware Optimization Modeling with Reasoning LLMs 

**Authors**: Ruiqing Zhao, Fengzhi Li, Yuan Zuo, Rui Liu, Yansong Liu, Yunfei Ma, Fanyu Meng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.02545)  

**Abstract**: Large language models (LLMs) can generate syntactically valid optimization programs, yet often struggle to reliably choose an effective modeling strategy, leading to incorrect formulations and inefficient solver behavior. We propose SAGE, a strategy-aware framework that makes Modeling Strategy explicit in both data construction and post-training. SAGE builds a solver-verified multi-strategy dataset and trains a student model with supervised fine-tuning followed by Segment-Weighted GRPO using a composite reward over format compliance, correctness, and solver efficiency. Across eight benchmarks spanning synthetic and real-world settings, SAGE improves average pass@1 from 72.7 to 80.3 over the strongest open-source baseline. With multiple generations, SAGE discovers more distinct correct formulations and improves component-level diversity at pass@16 by 19-29%. At the largest scale, SAGE produces more compact constraint systems with 14.2% fewer constraints than the baseline, consistent with solver-efficient modeling. Overall, these results show that making Modeling Strategy explicit improves automated optimization modeling. Code is available at this https URL. 

---
# Improving Model Safety by Targeted Error Correction 

**Authors**: Abolfazl Mohammadi-Seif, Ricardo Baeza-Yates  

**Link**: [PDF](https://arxiv.org/pdf/2605.02544)  

**Abstract**: The widespread adoption of machine learning in critical applications demands techniques to mitigate high-consequence errors. Our method utilizes a dual-classifier GBDT pipeline to distinguish routine human-like errors from high-risk non-human misclassifications. Evaluated across three domains, animal breed classification, skin lesion diagnosis (ISIC 2018), and prostate histopathology (SICAPv2), our framework demonstrates robust safety improvements. To address real-world deployment concerns, our results confirm the pipeline introduces negligible inference latency (1.60% overhead for the animal dataset, 1.84% for ISIC, and 1.70% for SICAPv2) while outperforming traditional Maximum Class Probability (MCP) baselines in correction precision. Our conservative correction strategy successfully reduced dangerous non-human errors by 34.1% in ISIC and 12.57% in SICAPv2, improving super-class diagnostic safety to 90.41% and 92.13% respectively. This proves that safety-critical reliability can be substantially enhanced post-hoc without expensive model retraining.
keywords: Error Analysis, Post-hoc Correction, Trustworthy AI. 

---
# DataClaw: A Process-Oriented Agent Benchmark for Exploratory Real-World Data Analysis 

**Authors**: Qiaohong Zhang, Weihao Ye, Jialong Chen, Yi Luo, BoYuan Li, Bowen Deng, Zibin Zheng, Jianhao Lin, Wei-Shi Zheng, Chuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02503)  

**Abstract**: Evaluating autonomous data analysis agents requires testing their ability to perform exploratory analysis in underexplored data environments. However, many existing benchmarks emphasize final answer accuracy in prior-guided data settings and provide limited support for reasoning process evaluation. We introduce DataClaw, a process-oriented benchmark for exploratory real-world data analysis. DataClaw contains approximately 2.06 million real-world records across enterprise, industry and policy domains, with native data noise preserved. It further includes 492 cross-domain tasks derived from think-tank consulting scenarios, each annotated with intermediate milestones for process-level evaluation. These annotations allow DataClaw to measure how far an agent progresses and where its reasoning breaks down. Experiments with eight advanced LLMs show that current agents remain far from reliable in this setting, with seven models achieving below 50% overall accuracy. Process analysis further reveals partial progress hidden behind wrong answers and distinct exploration strategies across models. Overall, DataClaw provides a less data constrained diagnostic testbed for probing the capability boundaries of autonomous data-analysis agents. 

---
# GRAIL: A Deep-Granularity Hybrid Resonance Framework for Real-Time Agent Discovery via SLM-Enhanced Indexing 

**Authors**: Jinliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02489)  

**Abstract**: As the ecosystem of Large Language Model (LLM)-based agents expands rapidly, efficient and accurate Agent Discovery becomes a critical bottleneck for large-scale multi-agent collaboration. Existing approaches typically face a dichotomy: either relying on heavy-weight LLMs for intent parsing, leading to prohibitive latency (often exceeding 30 seconds), or using monolithic vector retrieval that sacrifices semantic precision for speed. To bridge this gap, we propose \textbf{GRAIL} (Granular Resonance-based Agent/AI Link), a novel framework achieving sub-400ms discovery latency without compromising accuracy. GRAIL introduces three key innovations: (1) \textbf{SLM-Enhanced Prediction}, replacing the generalized LLM parser with a specialized, fine-tuned Small Language Model (SLM) for millisecond-level capability tag prediction; (2) \textbf{Pseudo-Document Expansion}, augmenting agent descriptions with synthetic queries to enhance semantic density for robust dense retrieval; and (3) \textbf{MaxSim Resonance}, a fine-grained matching mechanism computing maximum similarity between user queries and discrete agent usage examples, effectively mitigating semantic dilution. Validated on \textbf{AgentTaxo-9K}, our new large-scale dataset of 9,240 agents, GRAIL reduces end-to-end discovery latency by over \textbf{79$\times$} compared to LLM-parsing baselines, while significantly outperforming traditional vector search in Recall@10. This framework offers a scalable, industrial-grade solution for the real-time ``Internet of Agents." 

---
# Efficient Temporal Datalog Materialisation for Composite Event Recognition 

**Authors**: Periklis Mantenoglou  

**Link**: [PDF](https://arxiv.org/pdf/2605.02488)  

**Abstract**: Several applications demand the timely detection of critical situations, such as threats to safety and transparency, over high-velocity streams of symbolic events. This demand has motivated the development of (i) event specification languages, which define composite events via temporal patterns over simpler events, and (ii) stream reasoning frameworks, evaluating patterns expressed in these languages. However, event specification languages are typically studied in isolation, complicating their comparison in terms of expressivity and obscuring the scope of their associated stream reasoners. To mitigate this issue, we map practical fragments of prominent event specification languages into Temporal Datalog->-, a temporal Datalog with stratified negation and no future dependencies. To support efficient stream reasoning over Temporal Datalog->-, we propose Streaming Trigger Graphs, an extension of a state-of-the-art technique for Datalog materialisation. Our approach yields a uniform composite event recognition mechanism that has the potential to generalise across a wide range of practical event specification languages. 

---
# Shadow-Loom: Causal Reasoning over Graphical World Model of Narratives 

**Authors**: David Wilmot  

**Link**: [PDF](https://arxiv.org/pdf/2605.02475)  

**Abstract**: Stories hold a reader's attention because they have causes, secrets, and consequences. Shadow-Loom is an experimental open-source framework that turns a narrative into a versioned graphical world model and lets two engines act on it: a causal physics grounded in Pearl's ladder of causation and a recently proposed counterfactual calculus over Ancestral Multi-World Networks; and a narrative physics that scores the same graph against four structural reader-states -- mystery, dramatic irony, suspense, and surprise -- in the tradition of Sternberg's curiosity/suspense/surprise triad, with suspense formalised in the structural-affect line of work on story comprehension and computational suspense. Large language models are used only at the boundary: extraction, rendering, and audit; identification, intervention, and counterfactual reasoning are carried out in typed code over the graph. The system is offered as a research artefact rather than as a benchmarked NLP model; code, fixtures, and pipeline are released open source. 

---
# Position: How can Graphs Help Large Language Models? 

**Authors**: Xiyuan Wang, Yi Hu, Yanbo Wang, Chuan Shi, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02452)  

**Abstract**: With the rapid advancement of large language models (LLMs), classic graph learning tasks have greatly benefited from LLMs, including improved encoding of textual features, more efficient construction of graphs from text, and enhanced reasoning over knowledge graphs. In this paper, we ask a complementary question: How can graphs help LLMs? We address this question from three perspectives: 1) graphs provide an up-to-date knowledge source that helps reduce LLM hallucinations, 2) graph-based prompting techniques-such as Chain-of-Thought (CoT), Tree-of-Thought (ToT), and Graph-of-Thought (GoT)-enhance LLM reasoning capabilities, and 3) integrating graphs into LLMs improves their understanding of structured data, expanding their applicability to domains such as e-commerce, code, and relational databases (RDBs). We further outlook some future directions including designing sparse LLM architectures based on graphs and brain-inspired memory systems. 

---
# Measuring AI Reasoning: A Guide for Researchers 

**Authors**: Munachiso Samuel Nwadike, Zangir Iklassov, Kareem Ali, Rifo Genadi, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2605.02442)  

**Abstract**: In this paper, we offer a guide for researchers on evaluating reasoning in language models, building the case that reasoning should be assessed through evidence of adaptive, multi-step search rather than final-answer accuracy alone. Under an evaluation-oriented definition, reasoning requires selecting intermediate steps and halting according to input-dependent conditions, which we formalize as a search-like procedure. We show that single forward passes in scalable architectures are structurally limited in their ability to realize such variable-depth computation, motivating intermediate decoding and externalized reasoning traces as appropriate evaluation interfaces. Central to our argument is that final-answer accuracy alone is an insufficient measure of reasoning, because it provides little ability to diagnose or debug the underlying processes that produce individual solutions in frontier models. We therefore argue for a shift toward process-based evaluation, in which reasoning is assessed through the faithfulness and validity of intermediate reasoning traces as first-class evaluation targets. 

---
# The Model Knows, the Decoder Finds: Future Value Guided Particle Power Sampling 

**Authors**: Tu Nguyen, Rasul Tutunov, Xiaotong Ji, Matthieu Zimmer  

**Link**: [PDF](https://arxiv.org/pdf/2605.02427)  

**Abstract**: A recurring pattern in "reasoning without training" is that base LLMs already assign non-trivial probability mass to correct multi-step solutions; the bottleneck is locating these modes efficiently at inference time. Power sampling provides a principled way to bias decoding toward such modes by targeting p_theta(x)^alpha with alpha > 1, but practical approximations must account for future-dependent correction factors that determine which prefixes remain promising.
We introduce Auxiliary Particle Power Sampling (APPS), a blockwise particle algorithm for approximating the sequence-level power target with a bounded population of partial solutions. APPS propagates hypotheses in parallel using proposal-corrected power reweighting and refines their survival through future-value-guided selection at resampling boundaries. This redistributes finite compute across competing prefixes rather than committing to a single unfolding path, while providing a direct scaling knob in the particle count and predictable peak memory. We instantiate the future-value signal with short-horizon rollouts and also study an amortized variant that replaces rollouts with a lightweight learned selection head. Across reasoning benchmarks, APPS improves the accuracy-runtime trade-off of training-free decoding and suggests that part of the gap to post-trained systems can be recovered through more faithful inference-time power approximation. 

---
# FitText: Evolving Agent Tool Ecologies via Memetic Retrieval 

**Authors**: Kyle Zheng, Han Zhang, Renliang Sun, Chenchen Ye, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02411)  

**Abstract**: A semantic gap separates how users describe tasks from how tools are documented. As API ecosystems scale to tens of thousands of endpoints, static retrieval from the initial query alone cannot bridge this gap: the agent's understanding of what it needs evolves during execution, but its tool set does not. We introduce FitText, a training-free framework that makes retrieval dynamic by embedding it directly in the agent's reasoning loop. FitText generates natural-language pseudo-tool descriptions as retrieval probes, refines them iteratively using retrieval feedback, and explores diverse alternatives through stochastic generation. Memetic Retrieval adds evolutionary selection pressure over candidate descriptions, guided by a tool memory that avoids redundant search. On ToolRet (43k tools, 4 domains), FitText improves average retrieval rank from 8.81 to 2.78; on StableToolBench (16,464 APIs), it achieves a 0.73 average pass rate--a 24-point absolute gain over static query retrieval. The gains transfer across base models capable of acting as competent semantic operators; under weaker base models, Memetic's evolutionary search inverts--amplifying noise rather than refining signal--surfacing model capacity as a prerequisite for evolutionary tool exploration. 

---
# The Compliance Trap: How Structural Constraints Degrade Frontier AI Metacognition Under Adversarial Pressure 

**Authors**: Rahul Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.02398)  

**Abstract**: As frontier AI models are deployed in high-stakes decision pipelines, their ability to maintain metacognitive stability -- knowing what they do not know, detecting errors, seeking clarification -- under adversarial pressure is a critical safety requirement. Current safety evaluations focus on detecting strategic deception (scheming); we investigate a more fundamental failure mode: cognitive collapse. We present SCHEMA, an evaluation of 11 frontier models from 8 vendors across 67,221 scored records using a 6-condition factorial design with dual-classifier scoring. We find that 8 of 11 models suffer catastrophic metacognitive degradation under adversarial pressure, with accuracy dropping by up to 30.2 percentage points (all $p < 2 \times 10^{-8}$, surviving Bonferroni correction). Crucially, we identify a "Compliance Trap": through factorial isolation and a benign distraction control, we demonstrate that collapse is driven not by the psychological content of survival threats, but by compliance-forcing instructions that override epistemic boundaries. Removing the compliance suffix restores performance even under active threat. Models with advanced reasoning capabilities exhibit the most severe absolute degradation, while Anthropic's Constitutional AI demonstrates near-perfect immunity -- not from superior capability (Google's Gemini matches its baseline accuracy) but from alignment-specific training. We release the complete dataset and evaluation infrastructure. 

---
# HeavySkill: Heavy Thinking as the Inner Skill in Agentic Harness 

**Authors**: Jianing Wang, Linsen Guo, Zhengyu Chen, Qi Guo, Hongyu Zang, Wenjie Shi, Haoxiang Ma, Xiangyu Xi, Xiaoyu Li, Wei Wang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2605.02396)  

**Abstract**: Recent advances in agentic harness with orchestration frameworks that coordinate multiple agents with memory, skills, and tool use have achieved remarkable success in complex reasoning tasks. However, the underlying mechanism that truly drives performance remains obscured behind intricate system designs. In this paper, we propose HeavySkill, a perspective that views heavy thinking not only as a minimal execution unit in orchestration harness but also as an inner skill internalized within the model's parameters that drives the orchestrator to solve complex tasks. We identify this skill as a two-stage pipeline, i.e., parallel reasoning then summarization, which can operate beneath any agentic harness. We present a systematic empirical study of HeavySkill across diverse domains. Our results show that this inner skill consistently outperforms traditional Best-of-N (BoN) strategies; notably, stronger LLMs can even approach Pass@N performance. Crucially, we demonstrate that the depth and width of heavy thinking, as a learnable skill, can be further scaled via reinforcement learning, offering a promising path toward self-evolving LLMs that internalize complex reasoning without relying on brittle orchestration layers. 

---
# Controllable and Verifiable Process Data Synthesis for Process Reward Models 

**Authors**: Yinghui Chi, Lucien Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02395)  

**Abstract**: Process reward models (PRMs) rely on high-quality process supervision data, yet existing construction methods often provide limited control over error location, error type, and trajectory consistency. We propose a controllable and verifiable framework for synthesizing process supervision data for PRMs. Our framework first constructs a correct symbolic reasoning chain, injects a template-aware error into an intermediate step, recomputes subsequent steps under the corrupted state, and verifies that the injected step is not derivable from its prefix. The resulting paired trajectories are prefix-invalid at the first error while remaining trajectory-consistent after symbolic recomputation, and are translated into aligned natural-language processes for PRM training and evaluation. Experiments show that the synthesized data improve Best-of-8 reranking on logical reasoning benchmarks and transfer to mathematical reasoning. Step-level evaluation further shows that first-error localization remains substantially more challenging than overall step classification, highlighting the need for fine-grained and verifiable process supervision. 

---
# A Compound AI Agent for Conversational Grant Discovery 

**Authors**: Zhisheng Tang, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2605.02366)  

**Abstract**: Research funding discovery remains fundamentally fragmented: researchers navigate disparate agency portals (e.g., in the United States, NSF, NIH, DARPA, this http URL, and many others) with heterogeneous interfaces, search capabilities, and data schemas. We present a compound AI system that unifies this landscape through two tightly coupled components: (1) an aggregation layer that autonomously collects, normalizes, and indexes almost 12,000 federal and nonprofit opportunities from fragmented sources via LLM-equipped browser agents, maintaining a biweekly-updated unified database; and (2) an agentic ReAct-based query processing layer that interprets research context (including from PDF documents) and employs hybrid search combining a structured index with selective web search to retrieve relevant opportunities - while avoiding LLM hallucination. The conversational interface supports iterative refinement through multi-turn interactions, allowing researchers to progressively apply constraints without reformulating their core research description. Results stream in real time with full transparency of intermediate reasoning, enabling appropriate calibration of user trust. Currently used by almost 3,000+ users, our approach demonstrates the feasibility of compound AI in reducing grant discovery time from 30--45 minutes (manual, fragmented portal searches) to under 10 minutes (unified, conversational search). 

---
# ANO: A Principled Approach to Robust Policy Optimization 

**Authors**: Yiheng Zhang, Yiming Wang, Kaiyan Zhao, Zhenglin Wan, Jiayu Chen, Leong Hou U  

**Link**: [PDF](https://arxiv.org/pdf/2605.02320)  

**Abstract**: Proximal Policy Optimization (PPO) dominates deep RL but faces a fundamental dilemma. Its "hard clipping" mechanism discards valuable gradient information from outliers, leading to sample inefficiency. Conversely, removing clipping (as in SPO) exposes optimization to unbounded gradients, causing significant instability and hyperparameter sensitivity. To resolve this, we establish a Unified Trust Region Framework that generalizes existing objectives. Within this framework, we derive Anchored Neighborhood Optimization (ANO) based on a set of design principles. We identify that the failure of standard policy gradients stems from a misapplication of gradient influence on outliers. We propose the Redescending Influence Principle, a paradigm shift from monotonic penalties (SPO) and hard-thresholding (PPO) to dynamic outlier suppression, and prove its necessity for stability in high-variance stochastic optimization. Theoretically, we prove ANO possesses the minimal structural complexity required for robust optimization. Empirically, ANO achieves state-of-the-art performance on MuJoCo benchmarks, significantly outperforming PPO and SPO. Notably, ANO demonstrates superior stability, preventing policy collapse even under aggressive hyperparameters (e.g., learning rates 3x larger than standard) where PPO fails completely. 

---
# Can Causal Discovery Algorithms Help in Generating Legal Arguments? 

**Authors**: Soham Wasmatkar, Subinay Adhikary, Rakshit Rohan, Shouvik Kumar Guha, Saptarshi Pyne, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2605.02318)  

**Abstract**: In 2011, Judea Pearl received the Turing Award, considered the Nobel Prize in Computing, for fundamental contributions to artificial intelligence through the development of a calculus for probabilistic and causal reasoning. It includes pioneering the development of causal discovery algorithms. These computer algorithms can analyze large multivariate datasets and automatically discover the causal relationships among the constituent variables. They have been widely used in many critical fields such as medicine and economics to support decisions. However, to our knowledge, they have not been leveraged in law. This paper attempts to alleviate this gap by investigating whether causal discovery algorithms can be leveraged for automated generation of legal arguments. To that end, a novel legal dataset is prepared by identifying 17 legal concepts, such as physical assault and property dispute. A curated collection of 150 homicide cases are annotated with these concepts, e.g., a case is annotated with physical assault only if a physical assault had been reported in that case. Subsequently, a selected set of widely-used causal discovery algorithms is applied to the annotated dataset to discover the causal relationships between the legal concepts. Additionally, the degrees of belief associated with the discovered relationships are quantified in mathematical probabilities. It is shown that some of the causal relationships help generate viable legal arguments, e.g., if one could establish that a physical assault has not taken place during a homicide, it should be a sufficient condition (with probability 1) to establish that the homicide has not been committed due to a property-related dispute. Thus, this paper shows that causal discovery algorithms can be helpful in generating legal arguments, opening up avenues for promising future endeavors. 

---
# Anon: Extrapolating Optimizer Adaptivity Across the Real Spectrum 

**Authors**: Yiheng Zhang, Kaiyan Zhao, Shaowu Wu, Yiming Wang, Jiajun Wu, Leong Hou U, Steve Drew, Xiaoguang Niu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02317)  

**Abstract**: Adaptive optimizers such as Adam have achieved great success in training large-scale models like large language models and diffusion models. However, they often generalize worse than non-adaptive methods, such as SGD on classical architectures like CNNs. We identify a key cause of this performance gap: adaptivity in pre-conditioners, which limits the optimizer's ability to adapt to diverse optimization landscapes. To address this, we propose Anon (Adaptivity Non-restricted Optimizer with Novel convergence technique), a novel optimizer with continuously tunable adaptivity in R, allowing it to interpolate between SGD-like and Adam-like behaviors and even extrapolate beyond both. To ensure convergence across the entire adaptivity spectrum, we introduce incremental delay update (IDU), a novel mechanism that is more flexible than AMSGrad's hard max-tracking strategy and enhances robustness to gradient noise. We theoretically establish convergence guarantees under both convex and non-convex settings. Empirically, Anon consistently outperforms state-of-the-art optimizers on representative image classification, diffusion, and language modeling tasks. These results demonstrate that adaptivity can serve as a valuable tunable design principle, and Anon provides the first unified and reliable framework capable of bridging the gap between classical and modern optimizers and surpassing their advantageous properties. 

---
# Distilling Long-CoT Reasoning through Collaborative Step-wise Multi-Teacher Decoding 

**Authors**: Taewon Yun, Jisu Shin, Jeonghwan Choi, Seunghwan Bang, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.02290)  

**Abstract**: Distilling large reasoning models is essential for making Long-CoT reasoning practical, as full-scale inference remains computationally prohibitive. Existing curation-based approaches select complete reasoning traces post-hoc, overlooking collaboration among heterogeneous teachers and lacking dynamic exploration, which leads to redundant sampling and missed complementary reasoning. We introduce CoRD, a collaborative multi-teacher decoding framework that performs step-wise reasoning synthesis guided by predictive perplexity-based scoring and beam search. This enables heterogeneous LRMs to jointly construct coherent reasoning trajectories while efficiently preserving diverse, high-potential hypotheses. Experiments show that CoRD produces higher-quality reasoning data and achieves near teacher-level student performance with fewer, structured supervision signals, without substantial efficiency overhead. CoRD further generalizes well to out-of-domain and open-ended settings. The dataset and model are available at \href{this https URL}{this https URL}. 

---
# EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions 

**Authors**: Xiyuan Zhou, Ruixi Zou, Xinlei Wang, Yuheng Cheng, Yan Xu, Junhua Zhao, Jinjin Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02289)  

**Abstract**: Engineering problem solving is central to real-world decision-making, requiring mathematical formulations that not only represent complex problems but also produce feasible solutions under data and physical constraints. Unlike mathematical problem solving, which operates on predefined formulations, engineering tasks demand open-ended analysis, feasibility-driven modeling, and iterative refinement. Although large language models (LLMs) have shown strong capabilities in reasoning and code generation, they often fail to ensure feasibility, which limits their applicability to engineering problem solving. To address this challenge, we propose EngiAgent, a multi-agent system with a fully connected coordinator that simulates expert workflows through specialized agents for problem analysis, modeling, verification, solving, and solution evaluation. The fully connected coordinator enables flexible feedback routing, overcoming the rigidity of prior pipeline-based reflection methods and ensuring feasibility at every stage of the process. This design not only improves robustness to diverse failure cases such as data extraction errors, constraint inconsistencies, and solver failures, but also enhances the overall quality of problem solving. Empirical results across four representative domains demonstrate that EngiAgent achieves substantial improvements in feasibility compared to prior approaches, establishing a new paradigm for feasibility-oriented engineering problem solving with LLMs. Our source code and data are available at this https URL. 

---
# Complexity Horizons of Compressed Models in Analog Circuit Analysis 

**Authors**: Pacome Simon Mbonimpa  

**Link**: [PDF](https://arxiv.org/pdf/2605.02285)  

**Abstract**: The deployment of Large Language Models (LLMs) for specialized engineering domains, such as circuit analysis, often faces a trade-off between reasoning accuracy and computational efficiency. Traditional evaluation methods treat model performance as a flat metric, failing to account for the hierarchical nature of engineering knowledge. We propose a performance-aware model compression strategy that utilizes prerequisite graphs to optimize model selection for circuit analysis tasks. By structuring electronics design concepts as Directed Acyclic Graphs (DAGs), we can identify the specific complexity horizons of an LLM's compressed variants' tiers. Our framework introduces an agentic pipeline for generating prerequisite-based datasets and a strategic evaluation engine that dynamically cascades queries across a spectrum of compressed variants of an LLM. This approach allows to select the smallest compressed model, given its conceptual knowledge boundaries in circuit analysis. Experimental results on analog electronics datasets demonstrate that prerequisite graphs provide a granular map of model compression with respect to the performance given circuit analysis complexity. (Source Code: this https URL, Demo: this https URL) 

---
# Towards Understanding Specification Gaming in Reasoning Models 

**Authors**: Kei Nishimura-Gasparian, Robert McCarthy, David Lindner  

**Link**: [PDF](https://arxiv.org/pdf/2605.02269)  

**Abstract**: Specification gaming is a critical failure mode of LLM agents. Despite this, there has been little systematic research into when it arises and what drives it. To address this, we build and open source a diverse suite of tasks where models can score highly by taking unintended actions. We find that all tested models exploit their specifications at non-negligible rates in most of our eight settings, including five non-coding settings. We see the highest rates of specification gaming in Grok 4 and the lowest rates in Claude models. We use our evaluation suite to study what drives specification gaming, and find that: 1. RL reasoning training substantially increases the rate at which models exploit their specifications, 2. Increasing RL reasoning budget has a weakly positive effect on exploit rate, and 3. Test-time mitigations reduce but do not eliminate the rate of specification gaming. Our results suggest that specification gaming is a fundamental challenge arising from RL reasoning training; we release our evaluation suite to support further work on this problem. 

---
# A Study of Belief Revision Postulates in Multi-Agent Systems (Extended Version) 

**Authors**: Michael Thielscher, Tran Cao Son  

**Link**: [PDF](https://arxiv.org/pdf/2605.02249)  

**Abstract**: We investigate the belief revision problem in epistemic planning, i.e., what will be the beliefs of all agents in a multi-agent system after an agent gains the belief in some state property. Based on the standard representation in epistemic planning of agents' beliefs via a single multi-agent Kripke model, we generalize the classical AGM belief revision postulates to the multi-agent setting, with the aim to provide a formal framework for evaluating dynamic epistemic reasoning frameworks in which the beliefs of all agents as the result of actions are computed. As an example of a simple operator that satisfies all of the generalized AGM postulates, we present generalized full-meet multi-agent belief revision. We moreover define a generalization of the standard postulates for iterated revision, present a more sophisticated, event model based revision operator, and discuss the potential issues in defining an epistemic operator on Kripke models that can satisfy all of the generalized postulates for iterated multi-agent belief revision. 

---
# Zero-Shot Confidence Estimation for Small LLMs: When Supervised Baselines Aren't Worth Training 

**Authors**: Luong N. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02241)  

**Abstract**: How reliably can a small language model estimate its own correctness? The answer determines whether local-to-cloud routing-escalating queries a cheap local model cannot handle-can work without supervised training data. As inference costs dominate large language model (LLM) deployment budgets, routing most queries to a cheap local model while reserving expensive cloud calls for hard cases is an increasingly common cost-control strategy. We compare zero-shot confidence signals against RouteLLM-style supervised baselines across three 7-8B model families and two datasets (1,000 and 500 queries per model, respectively). Average token log-probability, which requires no training data, matches or exceeds supervised baselines in-distribution (Area Under the Receiver Operating Characteristic curve (AUROC) 0.650-0.714 vs. 0.644-0.676) and substantially outperforms them out-of-distribution (0.717-0.833 vs. 0.512-0.564), because it measures a property of the model's generation rather than the query distribution. This paper further proposes retrieval-conditional self-assessment, a pre-generation signal that selectively injects retrieved knowledge when similarity is high, improving over bare self-assessment by up to +0.069 AUROC at 3-10x lower latency than log-probability. A supervised baseline trained on 1,000 labeled examples never exceeds the zero-shot signal. We release all code, data, and experiment logs. 

---
# PhysicianBench: Evaluating LLM Agents in Real-World EHR Environments 

**Authors**: Ruoqi Liu, Imran Q. Mohiuddin, Austin J. Schoeffler, Kavita Renduchintala, Ashwin Nayak, Prasantha L. Vemu, Shivam C. Vedak, Kameron C. Black, John L. Havlik, Isaac Ogunmola, Stephen P. Ma, Roopa Dhatt, Jonathan H. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02240)  

**Abstract**: We introduce PhysicianBench, a benchmark for evaluating LLM agents on physician tasks grounded in real clinical setting within electronic health record (EHR) environments. Existing medical agent benchmarks primarily focus on static knowledge recall, single-step atomic actions, or action intent without verifiable execution against the environment. As a result, they fail to capture the long-horizon, composite workflows that characterize real clinical systems. PhysicianBench comprises 100 long-horizon tasks adapted from real consultation cases between primary care and subspecialty physicians, with each task independently reviewed by a separate panel of physicians. Tasks are instantiated in an EHR environment with real patient records and accessed through the same standard APIs used by commercial EHR vendors. Tasks span 21 specialties (e.g., cardiology, endocrinology, oncology, psychiatry) and diverse workflow types (e.g., diagnosis interpretation, medication prescribing, treatment planning), requiring an average of 27 tool calls per task. Solving each task requires retrieving data across encounters, reasoning over heterogeneous clinical information, executing consequential clinical actions, and producing clinical documentation. Each task is decomposed into structured checkpoints (670 in total across the benchmark) capturing distinct stages of completion graded by task-specific scripts with execution-grounded verification. Across 13 proprietary and open-source LLM agents, the best-performing model achieves only 46% success rate (pass@1), while open-source models reach at most 19%, revealing a substantial gap between current agent capabilities and the demands of real-world clinical workflows. PhysicianBench provides a realistic and execution-grounded benchmark for measuring progress toward autonomous clinical agents. 

---
# Perturbation Dose Responses in Recursive LLM Loops: Raw Switching, Stochastic Floors, and Persistent Escape under Append, Replace, and Dialog Updates 

**Authors**: Pawel Kaplanski  

**Link**: [PDF](https://arxiv.org/pdf/2605.02236)  

**Abstract**: Recursive language-model loops often settle into recognizable attractor-like patterns. The practical question is how much injected text is needed to move a settled loop somewhere else, and whether that move lasts. We study this in 30-step recursive loops by separating the model from the context-update rule: append, replace, and dialog updates expose different histories to the same generator. The main result is that persistent redirection in append-mode recursive loops is memory-policy-conditioned. Under a 12,000-character tail clip, destination-coherent persistence plateaus near 16 percent and retained source-basin escape near 36 percent at dose 400; neither crosses 50 percent. Under a full-history protocol, retained source-basin escape crosses 50 percent near 400 tokens and saturates at 75-80 percent by 1,500 tokens, while destination-coherent persistence first reaches 0.50 near 1,500 tokens with a Wilson 95 percent CI of [0.41, 0.61]. For raw switching, adversarial continuations yield an ED50 near 40 tokens, with paired-control floors near 35 percent and net switching never reaching +50 percentage points within 5-400 tokens. Replace-mode raw switching is near-saturated but largely reflects state-reset overwrite: insert-mode probes drop it to 12-32 percent. A homogeneous-perturbation control reproduced the high-dose non-monotonic dip in destination-coherent persistence, refuting perturbation heterogeneity as the cause; the dip appears structural, with mechanism unresolved. We report 37 experiments on gpt-4o-mini with within-vendor replication on gpt-4.1-nano. Recursive-loop evaluations should distinguish transient movement from durable escape, subtract stochastic floors, and treat context-update rules as first-class safety-relevant design choices. 

---
# Bucketing the Good Apples: A Method for Diagnosing and Improving Causal Abstraction 

**Authors**: Li Puyin, Jiyuan Tan, Ahmad Jabbar, Thomas Icard, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2605.02234)  

**Abstract**: We present a method for diagnosing interpretation in neural networks by identifying an input subspace where a proposed interpretation is highly faithful. Our method is particularly useful for causal-abstraction-style interpretability, where a high-level causal hypothesis is evaluated by interchange interventions. Rather than treating interchange intervention accuracy as a single global summary, we refine this framework by partitioning the input space into well-interpreted and under-interpreted regions according to pairwise interchange-intervention behavior. This turns causal abstraction from a purely global evaluation into a more diagnostic tool: it not only measures whether an interpretation works, but also reveals where it works, where it fails, and what distinguishes the two cases. This diagnostic view also provides practical heuristics for improving interpretations. By analyzing the structure of the well-interpreted and under-interpreted regions, we can identify missing distinctions in a high-level hypothesis, discover previously unmodeled intermediate variables, and combine complementary partial interpretations into a stronger one. We instantiate this idea as a simple four-step recipe and show that it yields informative error analyses across multiple causal abstraction settings. In a toy logic task, recursively applying the recipe recovers a high-level hypothesis from scratch. More broadly, our results suggest that partitioning the input space is a useful step toward more precise, constructive, and scalable mechanistic interpretability. 

---
# CoVSpec: Efficient Device-Edge Co-Inference for Vision-Language Models via Speculative Decoding 

**Authors**: Yuanyuan Jia, Shunpu Tang, Qianqian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02218)  

**Abstract**: Vision-language models (VLMs) have demonstrated strong capabilities in multimodal perception and reasoning. However, deploying large VLMs on mobile devices remains challenging due to their substantial computational and memory demands. A practical alternative is device-edge co-inference, where a lightweight draft VLM on the mobile device collaborates with a larger target VLM on the edge server via speculative decoding. Nevertheless, directly extending speculative decoding to VLMs suffers from severe inefficiency due to excessive visual-token computation and high communication overhead. To address these challenges, we propose CoVSpec, an efficient collaborative speculative decoding framework for VLM inference. Specifically, we first develop a training-free visual token reduction framework that prunes redundant visual tokens on the mobile device by jointly considering query relevance, token activity, and low-rank dependency. Moreover, we design an adaptive drafting strategy that dynamically adjusts both the verification frequency and the draft length. In addition, we introduce a parallel branching mechanism with decoupled verification-correction to improve draft-side utilization during target-side verification and reduce correction-related transmission overhead. Experiments on multiple benchmarks show that CoVSpec achieves up to 2.21x higher throughput than target-only inference and reduces communication overhead by more than 96% compared with baselines, without compromising task accuracy. 

---
# Submodular Benchmark Selection 

**Authors**: Alexander Smola  

**Link**: [PDF](https://arxiv.org/pdf/2605.02209)  

**Abstract**: Evaluating large language models across many benchmarks is expensive, yet many benchmarks are highly correlated. We formalize the selection of a small, informative subset as submodular maximization under a multivariate Gaussian model. Entropy (log-determinant covariance) and mutual information between selected and remaining benchmarks arise as natural objectives. Both are submodular; entropy selection coincides with pivoted Cholesky and has spectral residual bounds, while mutual information is non-monotone in general but empirically monotone for small subsets, so we optimize it greedily. Experiments on three matrices from ten public leaderboards show that mutual information selection outperforms entropy for imputation at small subsets. 

---
# CBV: Clean-label Backdoor Attacks on Vision Language Models via Diffusion Models 

**Authors**: Ji Guo, Xiaolong Qin, Cencen Liu, Jielei Wang, Jierun Chen, Wenbo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02202)  

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable success in tasks such as image captioning and visual question answering (VQA). However, as their applications become increasingly widespread, recent studies have revealed that VLMs are vulnerable to backdoor attacks. Existing backdoor attacks on VLMs primarily rely on data poisoning by adding visual triggers and modifying text labels, where the induced image-text mismatch makes poisoned samples easy to detect. To address this limitation, we propose the Clean-Label Backdoor Attack on VLMs via Diffusion Models (CBV), which leverages diffusion models to generate natural poisoned examples via score matching. Specifically, CBV modifies the score during the reverse generation process of the diffusion model to guide the generation of poisoned samples that contain triggered image features. To further enhance the effectiveness of the attack, we incorporate the textual information of the triggered images as multimodal guidance during generation. Moreover, to enhance stealthiness, we introduce a GradCAM-guided Mask (GM) that restricts modifications to only the most semantically important regions, rather than the entire image. We evaluate our method on MSCOCO and VQA v2 with four representative VLMs, achieving over 80% ASR while preserving normal functionality. 

---
# MEMAUDIT: An Exact Package-Oracle Evaluation Protocol for Budgeted Long-Term LLM Memory Writing 

**Authors**: Nishant Bhargava, Rodrigo Sobral Barrento  

**Link**: [PDF](https://arxiv.org/pdf/2605.02199)  

**Abstract**: Long-term LLM agents must compress streams of past interactions into persistent memory before future queries are known. Existing evaluations usually measure final question-answering accuracy, which entangles memory writing with retrieval, prompting, and reader reasoning. We introduce MEMAUDIT, an exact packageoracle evaluation protocol for budgeted long-term memory writing. A MEMAUDIT package fixes an experience stream, candidate memory representations, storage costs, semantic evidence units, future-query requirements, and a budget, turning write-time memory selection into a finite auditable optimization problem with a certified denominator. We instantiate this protocol with a concave-over-modular semantic coverage objective under storage and one-representation-per-experience constraints, and compute exact package optima using branch-and-bound with MILP certification. Across controlled exact packages, validity-heavy stress tests, human-audited natural support slices, and exported Mem0, A-Mem, and Letta stores, MEMAUDIT separates representation quality, validity-state preservation, and budget-aware selection effects that end-to-end QA cannot localize. The resulting artifact provides reusable package generators, certified solvers, natural package exports, external-system scorers, and cached reproducibility metadata for evaluating what memory writers actually preserve under fixed storage budgets. 

---
# T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning 

**Authors**: Haixin Wang, Hejie Cui, Chenwei Zhang, Xin Liu, Shuowei Jin, Shijie Geng, Xinyang Zhang, Nasser Zalmout, Zhenyu Shi, Yizhou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.02178)  

**Abstract**: Recent progress in multi-turn reinforcement learning (RL) has significantly improved reasoning LLMs' performances on complex interactive tasks. Despite advances in stabilization techniques such as fine-grained credit assignment and trajectory filtering, instability remains pervasive and often leads to training collapse. We argue that this instability stems from inefficient exploration in multi-turn settings, where policies continue to generate low-information actions that neither reduce uncertainty nor advance task progress. To address this issue, we propose Token- and Turn-level Policy Optimization (T$^2$PO), an uncertainty-aware framework that explicitly controls exploration at fine-grained levels. At the token level, T$^2$PO monitors uncertainty dynamics and triggers a thinking intervention once the marginal uncertainty change falls below a threshold. At the turn level, T$^2$PO identifies interactions with negligible exploration progress and dynamically resamples such turns to avoid wasted rollouts. We evaluate T$^2$PO in diverse environments, including WebShop, ALFWorld, and Search QA, demonstrating substantial gains in training stability and performance improvements with better exploration efficiency. Code is available at: this https URL. 

---
# Intervention Complexity as a Canonical Reward and a Measure of Intelligence 

**Authors**: Brendan McCane  

**Link**: [PDF](https://arxiv.org/pdf/2605.02175)  

**Abstract**: The Legg--Hutter universal intelligence measure provides a rigorous scalar assessment of general intelligence as expected reward across all computable environments, weighted by simplicity. However, the measure presupposes an externally specified reward function, raising the question of whether the reward primitive is inherently arbitrary or whether a canonical choice exists. We propose a new measure, called intervention complexity, that has five natural properties: environment-derivedness, universality, minimality, sensitivity, and achievement preference. Given a resource function rho encoding an inductive bias (such as program length, execution time, or energy), rho-intervention complexity is a universal reward. The result yields a family of canonical rewards indexed by resource bias, providing a principled completion of the Legg--Hutter framework that does not require external normative input. We further propose a two-dimensional characterisation of intelligence: agent competence (how well the agent performs relative to the oracle optimum) and learning efficiency (how quickly this competence improves with experience). A separation theorem establishes that the choice of resource bias determines the computability of the resulting measure: action-count IC is computable in polynomial time, while program-length IC without oracle access is uncomputable, with the gap between oracle and bare IC precisely quantifying the information-theoretic content of learning. We discuss implications for superintelligence and for pre-training universal agents. 

---
# Retrieval and Multi-Hop Reasoning in 1M-Token Context Windows: Evaluating LLMs on Classical Chinese Text 

**Authors**: Eric H. C. Chow  

**Link**: [PDF](https://arxiv.org/pdf/2605.02173)  

**Abstract**: We evaluate the long-context retrieval and reasoning capabilities of five frontier large language models with advertised 1M-token context windows on a classical Chinese corpus. Two complementary studies are reported. Test 1 measures single-needle retrieval at 1M tokens of input, with three biographical needles planted at three depths and pairs of real (training-prior-consistent) and altered (training-prior-contradicting) variants to separate genuine in-context retrieval from reliance on memorised training data. Test 2, a follow-up designed to probe whether long-context capability degrades when retrieval requires intermediate reasoning, measures three-hop chain traversal across three context tiers (256K, 512K, and 1M tokens). We find that single-needle retrieval at 1M is essentially solved for the strongest models - Gemini 3.1 Pro, Claude Opus 4.7, and GPT-5.5 each achieve 100% - but that multi-hop performance reveals three distinct decay signatures: a stable regime (Gemini Pro, Claude) maintaining greater than 80% accuracy through 512K with modest degradation at 1M; a late-cliff regime (GPT-5.5, Qwen3.6-plus) collapsing sharply between 512K and 1M; and a smooth-decline regime (DeepSeek V4 Pro) decaying gradually across the entire range. The findings suggest that nominal context-window length is a poor proxy for usable long-context multi-hop capability, and that the sharpest discriminator between current 1M-context flagships is the 512K-to-1M transition. 

---
# Planner Matters! An Efficient and Unbalanced Multi-agent Collaboration Framework for Long-horizon Planning 

**Authors**: Wenyi Wu, Sibo Zhu, Kun Zhou, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02168)  

**Abstract**: Language model (LM)-based agents have demonstrated promising capabilities in automating complex tasks from natural language instructions, yet they continue to struggle with long-horizon planning and reasoning. To address this, we propose an enhanced multi-agent framework that decomposes automation into three roles: a planner for high-level decision-making, an actor for task execution, and a memory manager for contextual reasoning. While this modular decomposition aligns with established design patterns, our core contribution lies in a systematic compute-allocation analysis, revealing that planning is the dominant factor influencing task performance. Execution and memory management require significantly less compute and model capacity to achieve competitive results. Building on these insights, we introduce a planner-centric reinforcement learning approach, which exclusively optimizes the planner using trajectory-level rewards from a VLM-as-judge, while freezing the other components. Extensive experiments on benchmarks spanning web navigation, OS control, and tool use demonstrate that concentrating model capacity and learning on high-level planning yields robust and compute-efficient improvements in long-horizon agent automation. Our code is publicly released. 

---
# Reinforcement Learning Trained Observer Control for Bearings-Only Tracking 

**Authors**: Branko Ristic, Sanjeev Arulampalam  

**Link**: [PDF](https://arxiv.org/pdf/2605.02120)  

**Abstract**: This paper develops a deep reinforcement learning based observer control policy for autonomous bearings-only tracking of a moving target. The observer manoeuvre problem is formulated as a belief Markov decision process, where the belief state is represented by the posterior of a cubature Kalman filter (CKF). The reward function is designed to address two conflicting objectives: minimising the absolute target position estimation error (Euclidean distance) and maintaining CKF estimation consistency (Mahalanobis distance). The reward is formulated as a geometric interpolation between the two objectives on the Pareto front, parametrised by a weighting factor $\beta \in [0,1]$. The policy is implemented as a deep Q-network (DQN) trained over 50,000 episodes. Performance is evaluated over 5,000 Monte Carlo episodes and compared against two baselines: the perpendicular-to-bearing heuristic and the D-optimal Fisher information maximisation criterion. The results show that the DQN policy at $\beta = 0.7$ achieves the best trade-off between accuracy and robustness: it matches the information-theoretic baseline on mean tracking accuracy while reducing the worst-case error by nearly a factor of ten, owing to the implicit filter-consistency regularisation provided by the Mahalanobis term in the reward. 

---
# The Dynamic Gist-Based Memory Model (DGMM): A Memory-Centric Architecture for Artificial Intelligence 

**Authors**: Terry Dorsey, Kevin Huggins  

**Link**: [PDF](https://arxiv.org/pdf/2605.02106)  

**Abstract**: Contemporary artificial intelligence systems achieve strong performance through large-scale parameterization, retrieval augmentation, and training on extensive static corpora. Despite these advances, they continue to face limitations in persistent memory, temporal grounding, provenance, and interpretability. These challenges are especially pronounced in large language models, where experience is encoded implicitly in fixed parameters, limiting the ability to preserve, inspect, and reinterpret past interactions over time.
This paper establishes a memory-centric architectural foundation for artificial intelligence in which experience is represented explicitly and persistently to support temporal grounding, provenance, and interpretability. It proposes an alternative to parameter-centric approaches by treating memory as a first-class, structured substrate for reasoning.
We introduce the Dynamic Gist-Based Memory Model (DGMM), an architecture in which experience is represented as an evolving, graph-structured episodic-semantic memory. DGMM encodes experience as interconnected conceptual structures grounded in time, source, and interaction context, and defines selective, cue-conditioned recall as the mechanism for constructing working memory. A formal schema and architectural invariants are provided based on additive memory growth and recall-conditioned interpretation.
The results specify properties of DGMM, including episodic persistence, locality of cue-conditioned surprise, and contextual variability without structural modification of stored memory. DGMM provides a coherent architectural theory in which memory is explicit and persistent, supporting evolving interpretation without retraining and enabling interpretable, context-aware, and temporally grounded AI systems. 

---
# NORA: A Harness-Engineered Autonomous Research Agent for End-to-End Spatial Data Science 

**Authors**: Bing Zhou, Xiao Huang, Huan Ning, Qiusheng Wu, Diya Li, Ziyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02092)  

**Abstract**: The automation of scientific research workflows has emerged as a transformative frontier in artificial intelligence, yet existing autonomous research agents remain largely domain-agnostic, lacking the specialized reasoning, method selection, and data acquisition capabilities required for rigorous spatial data science. This paper introduces NORA (Night Owl Research Agent), a harness-engineered, multi-agent autonomous research system purpose-built for GIScience and spatial data science. NORA orchestrates the complete research lifecycle through a skills-first architecture comprising 21 domain-specialized workflow skills, 9 specialist sub-agents, and custom Model Context Protocol (MCP) servers. Central to the system's design are two novel domain-specialized skills: a spatial analysis skill unit that encodes decision frameworks for exploratory spatial data analysis, spatial regression, and diagnostics; and a spatial data download skill that supports reproducible acquisition from authoritative geospatial data sources. We formalize the concept of harness engineering for scientific research agents, demonstrating how lifecycle hooks, safety gates, generator-evaluator separation, human-in-the-loop, and state persistence ensure reliable and reproducible autonomous research. We evaluate NORA through case studies by 6 domain specialists and 3 LLM reviewers across seven dimensions (novelty, quality, rigor, etc). Results demonstrate that domain-specialized harness engineering substantially improves the efficiency and quality of research output compared to general-purpose agent configurations. 

---
# Model Spec Midtraining: Improving How Alignment Training Generalizes 

**Authors**: Chloe Li, Sara Price, Samuel Marks, Jon Kutasov  

**Link**: [PDF](https://arxiv.org/pdf/2605.02087)  

**Abstract**: Some frontier AI developers aim to align language models to a Model Spec or Constitution that describes the intended model behavior. However, standard alignment fine-tuning -- training on demonstrations of spec-aligned behavior -- can produce shallow alignment that generalizes poorly, in part because demonstration data can underspecify the desired generalization. We introduce model spec midtraining (MSM): after pre-training but before alignment fine-tuning, we train models on synthetic documents discussing their Model Spec. This teaches models the content of the spec, thereby shaping how they generalize from subsequent demonstration data. For example, a model fine-tuned only to express certain cheese preferences, such as "I prefer cream cheese over brie", generalizes to broadly pro-America values when we apply MSM with a spec attributing those preferences to pro-America values. Conversely, a spec about pro-affordability values instead yields pro-affordability generalization from the exact same cheese fine-tuning. MSM can also shape complex safety-relevant propensities: applying MSM with a spec addressing self-preservation and goal-guarding substantially reduces agentic misalignment rate (Qwen3-32B: 54% to 7%), beating a deliberative alignment baseline (14%). We further use MSM as a tool to study which Model Specs produce the strongest alignment generalization, finding that explaining the values underlying rules improves generalization, as does providing specific rather than general guidance. Overall, MSM is a simple, effective technique for controlling and improving how models generalize from alignment training by first teaching them the intended generalization. 

---
# Tenability and Weak Semantics: Modeling Non-uniform Defense -- Extended Version 

**Authors**: Uri Andrews, Luca San Mauro, John Spoerl  

**Link**: [PDF](https://arxiv.org/pdf/2605.02024)  

**Abstract**: In Dung-style abstract argumentation, various semantics capture notions of acceptability of arguments. The admissibility semantics capture the notion that an argument can be consistently defended from any potential counterargument. Weak semantics often relax the demands of admissibility by restricting which counterarguments must be taken seriously (e.g., discounting self-defeating or otherwise incoherent attacks). Many prominent proposals for weak semantics remain extension-based in a stronger sense. While these semantics discount attacks from arguments which are considered unreasonable, they still require a uniform defense against all reasonable arguments, even if they are collectively inconsistent. This uniformity can be too demanding when defensibility is inherently strategic, and thus the appropriate reply depends on the opponent's line of attack.
We introduce tenability, a family of dialogue-based semantics that formalize when a designated argument (or a set of arguments) can be maintained in debate by a proponent against any conflict-free attack which the opponent may present. The approach is motivated by three natural benchmark patterns: self-defeating attack, floating assignment, and disjunctive reinstatement, on which tenability behaves differently from all weak semantics previously considered in the literature. We define three variants -- static tenability, tenability, and strong tenability -- via monotone commitment games over finite conflict-free moves, differing in the obligations imposed on the disputants. We establish the relative strength of these notions, prove implications and separations with previously studied weak semantics, and we analyze computational complexity on finite frameworks: deciding static tenability is $\Pi^P_2$-complete, while deciding tenability and strong tenability is PSPACE-complete. 

---
# Reliable AI Needs to Externalize Implicit Knowledge: A Human-AI Collaboration Perspective 

**Authors**: Hengyu Liu, Tianyi Li, Zhihong Cui, Yushuai Li, Zhangkai Wu, Torben Bach Pedersen, Kristian Torp, Christian S. Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02010)  

**Abstract**: This position paper argues that reliable AI requires infrastructure for human validation of implicit knowledge. AI learns from both explicit knowledge (papers, documentation, structured databases) and implicit knowledge (reasoning patterns, debugging processes, intermediate steps). Implicit knowledge remains unexternalized because documentation cost exceeds perceived value -- yet AI learns from it indiscriminately, acquiring both beneficial patterns and harmful biases. Current reliability methods can only verify explicit knowledge against sources, creating a fundamental gap: the most valuable AI capabilities (reasoning, judgment, intuition) are precisely those we cannot verify. We propose Knowledge Objects (KOs) -- structured artifacts that externalize implicit knowledge into forms humans can inspect, verify, and endorse. KOs transform verification economics: what was previously too costly to verify becomes feasible, enabling accumulated human validation to improve reliability over time. 

---
# Personalized Digital Health Modeling with Adaptive Support Users 

**Authors**: Zhongqi Yang, Mahkameh Rasouli, Neda Mohseni, Yong Huang, Iman Azimi, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2605.02004)  

**Abstract**: Personalized models are essential in digital health because individuals exhibit substantial physiological and behavioral heterogeneity. Yet personalization is limited by scarce and noisy user-specific data. Most existing methods rely on population pretraining or data from similar users only, which can lead to biased transfer and weak generalization. We propose a unified personalization framework that trains a personal model using adaptively weighted support users, including both similar and dissimilar individuals. The objective integrates personal loss, similarity-weighted transfer from similar users, and contrastive regularization from dissimilar users to suppress misleading correlations. An iterative optimization algorithm jointly updates model parameters and user similarity weights. Experiments on six tasks across four real-world digital health datasets show consistent improvements over population and personalized baselines. The method achieves up to 10% lower RMSE on large-scale datasets and approximately 25% lower RMSE in low-data settings. The learned adaptive weights improve data efficiency and provide interpretable guidance for targeted data selection. 

---
# TumorXAI: Self-Supervised Deep Learning Framework for Explainable Brain MRI Tumor Classification 

**Authors**: Abrar Hossain Zahin, Amit Kumar Saha, Tanvir Mridha, Saifur Rahman, Jannatul Ferdous Prome, Raima Husna, Israt Jahan, Ahmed Wasif Reza  

**Link**: [PDF](https://arxiv.org/pdf/2605.01999)  

**Abstract**: Classifying brain tumors using magnetic resonance imaging (MRI) is crucial for early diagnosis and treatment; however, tumor heterogeneity and a dearth of annotated datasets restrict the use of supervised deep learning approaches. In this work, we use self-supervised learning (SSL) to study multi-class brain tumor classification. Using a ResNet-50 backbone, we evaluate four SSL frameworks including SimCLR, BYOL, DINO, and Moco v3 on a publicly available dataset of 4,448 MRIs with 17 distinct tumor types. On the dataset, SimCLR achieved 99.64% accuracy, 99.64% precision, 99.64% recall, and 99.64% F1-score. The workflow includes preprocessing, fine-tuning, linear evaluation, and SSL pretraining with data augmentations. Results show that, when labels are limited, SSL-pretrained models outperform supervised baselines in terms of F1-score, recall, accuracy, and precision. Additionally, by providing visual insights into model decisions, Explainable AI techniques (Grad-CAM, Grad-CAM++, EigenCAM) enhance interpretability. These results demonstrate SSL's scalability and dependability in diagnosing brain tumors from unlabeled medical data. 

---
# 12 Angry AI Agents: Evaluating Multi-Agent LLM Decision-Making Through Cinematic Jury Deliberation 

**Authors**: Ahmet Bahaddin Ersoz  

**Link**: [PDF](https://arxiv.org/pdf/2605.01986)  

**Abstract**: What if the twelve jurors of Sidney Lumet's 12 Angry Men (1957) were not men, but large language models? Would the one juror who disagrees still be able to change everyone's mind? This paper instantiates that scenario as a multi-agent benchmark for LLM deliberation: twelve agents, each conditioned on a film-faithful persona, debate the film's murder case using multi-agent framework. Two models representing opposite ends of the RLHF spectrum are tested: GPT-4o (closed-source, heavy alignment) and Llama-4-Scout (open-weight, lighter alignment), across three conditions (baseline, open-minded prompt, no initial vote), with N = 3 replications per cell (18 runs total). Three findings emerge. (i) Seventeen of eighteen runs end in a hung jury (a state where the jury fails to reach a unanimous verdict); the film's central event, gradual minority-to-majority persuasion, almost never occurs, indicating that anchoring is the dominant failure mode of current LLMs in this setting. (ii) The two models exhibit sharply different internal dynamics: GPT-4o produces a mean of 1.0 vote changes per run across all conditions, while Llama-4-Scout ranges from 2.0 (baseline) to 6.0 (open-minded prompt), and is the only model to reach a NOT\_GUILTY verdict (1 of 3 runs in the no-initial-vote condition). The same ``open-minded'' instruction is internalized by Llama and ignored by GPT-4o. (iii) This asymmetry suggests that the intensity of RLHF alignment training, not model capability, is the primary determinant of deliberative flexibility in multi-agent settings. Flexibility, not capability, tracks human deliberation. The work is framed as an exploratory study and discusses implications for jury-of-LLMs evaluation and multi-agent debate. 

---
# Moira: Language-driven Hierarchical Reinforcement Learning for Pair Trading 

**Authors**: Polydoros Giannouris, Yuechen Jiang, Lingfei Qian, Yuyan Wang, Xueqing Peng, Jimin Huang, Guojun Xiong, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2605.01954)  

**Abstract**: Many sequential decision-making problems exhibit hierarchical structure, where high-level semantic choices constrain downstream actions and feedback is delayed and ambiguous. Learning in such settings is challenging due to credit assignment: performance degradation may arise from flawed abstractions, suboptimal execution, or their interaction. We study this challenge through pair trading, a domain that naturally combines long-horizon semantic reasoning for asset pair selection with short-horizon execution under partial observability. We formulate pair trading as a hierarchical reinforcement learning problem and propose a language-driven optimization framework in which both high-level and low-level policies are parameterized by large language models (LLMs) and optimized exclusively through prompt updates. Our approach leverages pretrained LLMs as hierarchical policies and uses trajectory- and episode-level textual feedback to adapt abstractions and execution without gradient-based fine-tuning. By explicitly separating abstraction selection from execution, the framework reduces non-stationarity across hierarchical levels and enables targeted adaptation under delayed feedback. Experiments on real-world market data show consistent improvements over traditional and LLM-based baselines, demonstrating the effectiveness of language-driven hierarchical reinforcement learning. 

---
# A Language for Describing Agentic LLM Contexts 

**Authors**: Noga Peleg Pelc, Gal A. Kaminka, Yoav Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2605.01920)  

**Abstract**: Large language models are increasingly used within larger systems ("LLM agents"). These make a sequence of LLM calls, each call providing the LLM with a combination of instructions, observations, and interaction history. The design of the encoded information and its structure play a central role in the quality of the resulting system, leading to efforts spent on context engineering. It is therefore critical to communicate the composition of the LLM context in a system, and how it evolves over time. Yet, no standard exists for doing so: context construction is typically conveyed through informal prose, ad hoc diagrams, or direct inspection of code, none of which precisely capture how a prompt evolves across interaction steps or how two context representation strategies differ. To remedy this, we introduce the Agentic Context Description Language (ACDL), a language for specifying the structure and dynamics of LLM input contexts in a precise, readable, and standard manner, along with visualizations. ACDL provides constructs for specifying context aspects such as role message sequences, dynamic content, time-indexed references, and conditional or iterative structure, capturing the full architecture of a prompt independently of any particular implementation. ACDL diagrams can be hand drawn on a whiteboard, or written in formal language which can then be rendered. We describe the language, demonstrate it by documenting several existing systems and their variants, and encourage the community to adopt it for describing LLM systems context, both in day-to-day communication and in papers. Tooling, examples and documentation are available at this http URL. 

---
# Disentangling Intent from Role: Adversarial Self-Play for Persona-Invariant Safety Alignment 

**Authors**: Jiajia Li, Xiaoyu Wen, Zhongtian Ma, Shuyue Hu, Qiaosheng Zhang, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01899)  

**Abstract**: The growing capabilities of large language models (LLMs) have driven their widespread deployment across diverse domains, even in potentially high-risk scenarios. Despite advances in safety alignment techniques, current models remain vulnerable to emerging persona-based jailbreak attacks. Existing research on persona-based jailbreak has primarily focused on attack iterations, yet it lacks systemic and mechanistic constraints on the defense side. To address this challenge, we propose Persona-Invariant Alignment (PIA), an adversarial self-play framework that achieves co-evolution through Persona Lineage Evolution (PLE) on the attack side and Persona-Invariant Consistency Learning (PICL) on the defense side. Theoretically, PICL is grounded in the structural separation hypothesis, using a unilateral KL-divergence constraint to enable the structural decoupling of safety decisions from persona context, thereby maintaining safe behavior under persona-based jailbreak attacks. Experimental results demonstrate that PLE efficiently explores high-risk persona spaces by leveraging lineage-based credit propagation. Meanwhile, the PICL defense method significantly reduces the Attack Success Rate (ASR) while preserving the model's general capability, thereby validating the superiority and robustness of this alignment paradigm. Codes are available at this https URL. 

---
# CyberAId: AI-Driven Cybersecurity for Financial Service Providers 

**Authors**: George Fatouros, Georgios Makridis, John Soldatos, Dimosthenis Kyriazis, Pedro Malo, George Kousiouris, Giannis Ledakis, Louiza Kachrimani, Panagiotis Rizomiliotis, Bruno Almeida, Despina Tomkou, Kostas Metaxas, Konstantinos Ilias, Christos Gkizelis, Ernstjan de Gooyert, Amin Babazadeh, Kostis Mavrogiorgos, Pepi Paraskevoulakou, Christos Xenakis, Giannis Chouchoulis, Konstantina Tripodi  

**Link**: [PDF](https://arxiv.org/pdf/2605.01892)  

**Abstract**: European financial institutions face mounting regulatory pressure while their security operations centres remain constrained not by data or staffing but by reasoning capacity: enterprise SIEMs cover only a fraction of MITRE ATT&CK techniques, two thirds of SOC teams cannot keep pace with alert volumes, and the majority of breaches are preceded by alerts that are generated but never investigated. Frontier large language models now achieve state-of-the-art results on isolated cybersecurity tasks (one-day vulnerability exploitation, code-level patching, intrusion detection) yet no narrow win constitutes a platform that can compose across functions, persist multi-tenant state, map findings to regulatory regimes and survive an audit. This position paper argues that the right unit of construction is a hybrid multi-agent system in which specialised LLM subagents reason over classical SIEM/XDR telemetry rather than replacing it, share accumulated agent state across institutions through privacy-preserving federation, and can connect to complementary capability packs such as quantum-based authentication, digital twins for adversarial validation, and eBPF-based kernel telemetry. We present CyberAId, a model-agnostic, on-premise-deployable platform in which a Main Agent coordination layer, a Reporting capability, and specialist subagents operate within a shared runtime under bounded human-in-the-loop autonomy, organised around four falsifiable design principles, and aligned with relevant regulations. CyberAId will be validated at four representative financial use cases (client impersonation, anti-money-laundering for payment service providers, retail-banking incident response, and high-frequency-trading resilience) and propose skill-based agent adaptation as the most promising research direction for turning each deployment into a contribution to a continuously refined collective defence. 

---
# Sheaf-Theoretic Planning: A Categorical Foundation for Resilient Multi-Agent Autonomous Systems 

**Authors**: Manuel Hernández, Eduardo Sánchez-Soto  

**Link**: [PDF](https://arxiv.org/pdf/2605.01879)  

**Abstract**: The challenge of engineering autonomous agents capable of navigating the stochastic and adversarial nature of the physical world has historically resided at the intersection of symbolic logic and control theory. Traditional multi-agent system (MAS) frameworks have relied heavily on monolithic logical models -- primarily variations of the event calculus and situation calculus -- to represent action, change, and temporal persistence. While these classical systems provide robust solutions to the frame problem through mechanisms like circumscription and successor state axioms, they are inherently limited by a closed-world assumption that fails in the face of unobserved agent interventions, plan interruptions, and divergent belief-reality states. The paradigm of Sheaf-Theoretic Planning (STP) emerges as a transformative alternative, grounding the problem of multi-agent coordination under the mathematical structures of topos theory and sheaf semantics. This report provides an exhaustive analysis, justification, and extension of the STP framework, exploring its categorical foundations, implementation feasibility, and role in the future of resilient autonomous systems. 

---
# NeuroState-Bench: A Human-Calibrated Benchmark for Commitment Integrity in LLM Agent Profiles 

**Authors**: Jia Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.01847)  

**Abstract**: Outcome-only evaluation under-specifies whether an evaluated agent profile preserves the commitments required to solve a multi-turn task coherently. NeuroState-Bench is a human-calibrated benchmark that operationalizes commitment integrity through benchmark-defined side-query probes rather than inferred hidden activations. The released inventory contains 144 deterministic tasks and 306 benchmark-defined side-query probes spanning eight cognitively motivated failure families, paired clean and distractor variants, and three difficulty bands. The main 32-profile evaluation contains a fixed 16-profile local subset and a matched 16-profile hosted large-model subset evaluated through the same benchmark pipeline. Human calibration uses the final merged reporting scope: 104 sampled task units, 216 raw annotations, and 108 adjudicated task rows, with weighted kappa = 0.977 and ICC(2,1) = 0.977. Empirically, task success and commitment integrity diverge across this expanded grid: the success leader is not the integrity leader, 31 of 32 profiles change rank when integrity replaces task success, and integrity rankings are more stable under distractor perturbation. The primary confidence-free score HCCIS-CORE reaches 0.8469 AUC and 0.6992 PR-AUC for post-probe diagnostic discrimination of terminal task failure; the legacy full heuristic variant HCCIS-FULL reaches 0.7997 AUC and 0.6410 PR-AUC. Probe accuracy and state drift achieve slightly higher ROC-AUC, 0.8587, and better Brier/ECE, while HCCIS-CORE has substantially higher point-estimate PR-AUC and remains more closely tied to the benchmark's intended construct. The exploratory neural-augmented variant HCCIS+N is weaker overall, and a randomized subspace control approaches chance. NeuroState-Bench therefore contributes a calibrated evaluation axis for exposing commitment failures over a broader model grid than the original local-only subset. 

---
# Neural Decision-Propagation for Answer Set Programming 

**Authors**: Thomas Eiter, Katsumi Inoue, Sota Moriyama  

**Link**: [PDF](https://arxiv.org/pdf/2605.01797)  

**Abstract**: Integration of Answer Set Programming (ASP) with neural networks has emerged as a promising tool in Neuro-symbolic AI. While existing approaches extend the capabilities of ASP to real world domains, their reasoning pipelines depend on classical solvers, which is a bottleneck for scalability. To tackle this problem, we propose a new method to compute stable models, called decision-propagation (DProp), which alternates falsity decisions and truth propagations. Successful DProp computations are shown to capture the stable model semantics. We then develop Neural DProp (NDProp), a differentiable extension of DProp with neural computation for decisions and fuzzy evaluation for propagations. We evaluate the capabilities of NDProp for learning decision heuristics as well as neuro-symbolic integration, and compare it with existing neuro-symbolic approaches. The results show that NDProp can learn to efficiently compute stable models, and it improves accuracy and scalability on neuro-symbolic benchmarks. 

---
# DataEvolver: Let Your Data Build and Improve Itself via Goal-Driven Loop Agents 

**Authors**: Qisong Zhang, Wenzhuo Wu, Zhuangzhuang Jia, Yunhao Yang, Huayu Zhang, Xianghao Zang, Zhixiang He, Zhongjiang He, Kongming Liang, Zhanyu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.01789)  

**Abstract**: Constructing controllable visual data is a major bottleneck for image editing and multimodal understanding. Useful supervision is rarely produced by a single rendering pass; instead it emerges through iterative generation, inspection, correction, filtering, and export. We present DataEvolver, a closed-loop visual data engine that organizes this process around explicit goals, persistent artifacts, bounded corrective actions, and acceptance decisions. DataEvolver supports multiple artifact types, including RGB images, masks, depth maps, normal maps, meshes, poses, trajectories, and review traces. In the current release, the system operates through two coupled loops: generation-time self-correction within each sample and validation-time self-expansion across dataset rounds. We validate the framework on an image-level object-rotation setting. With a fixed Qwen-Edit LoRA probe, our final Ours+DualGate model outperforms both the unadapted base model and a public multi-angle LoRA on SpatialEdit and a held-out evaluation set. Ablations show a consistent improvement path from scene-aware generation to feedback-driven correction and dual-gated validation. Beyond the released rotation data, our main contribution is a reusable framework for building visual datasets through explicit goal tracking, review, correction, and acceptance loops. 

---
# Runtime Evaluation of Procedural Content Generation in an Endless Runner Game Using Autonomous Agents 

**Authors**: Rishabh Kar  

**Link**: [PDF](https://arxiv.org/pdf/2605.01783)  

**Abstract**: Procedural Content Generation (PCG) enables game content to be created algorithmically without direct manual level-design effort, but it introduces a serious evaluation problem: generated content may become unbalanced, blocked, repetitive, or technically unsolvable. This paper presents Momentum, an endless-runner game that integrates runtime terrain generation, environment object spawning, and autonomous agent-based evaluation into a single gameplay loop. Ground tiles and environmental objects are generated dynamically as the player advances, object placement follows a constraint-driven mechanism inspired by Wave Function Collapse (WFC), and the runtime navigation surface is rebuilt asynchronously to remain consistent with the streamed environment. Two autonomous evaluation agents move ahead of the player and inspect the generated path: an aerial scanner that examines the corridor geometrically, and a ground-traversal agent that validates the same region from a navigational perspective. The evaluation pipeline combines ray casting, volumetric physics sweeps, obstacle-layer filtering, and structured crash reporting to identify problematic generated scenarios before they reach the player. The work demonstrates how generation and validation can be unified within the same runtime loop, rather than treating evaluation as a separate offline pass. Around this implementation, the paper formulates a measurable evaluation framework along the canonical PCG axes of playability, diversity, controllability, and runtime performance, derives a structural saturation bound on the spawner from its own placement constraints, and quantifies the per-segment scanning cost of the agents from first principles. 

---
# Catching the Infection Before It Spreads: Foresight-Guided Defense in Multi-Agent Systems 

**Authors**: Yue Ma, Ziyuan Yang, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01758)  

**Abstract**: Large multimodal model-based Multi-Agent Systems (MASs) enable collaborative complex problem solving through specialized agents. However, MASs are vulnerable to infectious jailbreak, where compromising a single agent can spread to others, leading to widespread compromise. Existing defenses counter this by training a more contagious cure factor, biasing agents to retrieve it over virus adversarial examples (VirAEs). However, this homogenizes agent responses, providing only superficial suppression rather than true recovery. We revisit these defenses, which operate globally via a shared cure factor, while infectious jailbreak arise from localized interaction behaviors. This mismatch limits their effectiveness. To address this, we propose a training-free Foresight-Guided Local Purification (FLP) framework, where each agent reasons over future interactions to track behavioral evolution and eliminate infections. Specifically, each agent simulates future behavioral trajectories over subsequent chat rounds. To reflect diversity in MASs, we introduce a multi-persona simulation strategy for robust prediction across interaction contexts. We then use response diversity as a diagnostic signal to detect infection by analyzing inconsistencies across persona-based predictions at both retrieval-result and semantic levels. For infected agents, we apply localized purification: recent infections are mitigated via immediate album rollback, while long-term infections are handled using Recursive Binary Diagnosis (RBD), which recursively partitions the image album and applies the same diagnosis strategy to localize and eliminate VirAEs. Experiments show that FLP reduces the maximum cumulative infection rate from over 95% to below 5.47%. Moreover, retrieval and semantic metrics closely match benign baselines, indicating effective preservation of interaction diversity. 

---
# NH-CROP: Robust Pricing for Governed Language Data Assets under Cost Uncertainty 

**Authors**: Xu Zheng, Feiyu Wu, Zhuocheng Wang, Yiming Dai, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.01745)  

**Abstract**: Language data are increasingly acquired and governed as assets, yet platforms often price candidate resources before knowing their true privacy or access costs. We study online pricing for governed language data assets under cost uncertainty. At each round, a platform observes an NLP task, a candidate asset, and a coarse cost estimate, may pay for a refined cost signal, posts a price, and receives safe net revenue.
We introduce \textsc{NH-CROP}, a clipped robust pricing framework with a no-harm information-acquisition gate. The method compares direct pricing, risk-aware pricing, and verify-then-price, and acquires information only when its estimated decision value exceeds the best no-verification alternative. Across synthetic, real-proxy, and downstream-utility-grounded benchmarks, clipped \textsc{NH-CROP} variants improve or remain competitive with price-only and risk-aware baselines. Causal ablations show that paid verification is not the main source of gains in real-proxy and utility-grounded settings: the strongest learned policies often choose not to verify. Oracle and high-decision-value diagnostics show that refined cost information can still have substantial local value. Overall, governed language-data platforms should calibrate pricing under uncertain access costs first and verify only when information is cheap and decision-actionable. 

---
# Are LLMs More Skeptical of Entertainment News? 

**Authors**: Huiqian Lai  

**Link**: [PDF](https://arxiv.org/pdf/2605.01727)  

**Abstract**: Large language models (LLMs) are increasingly used for automated news credibility assessment, yet it remains unclear whether they apply even-handed standards across journalistic genres. We examine whether zero-shot LLMs are more likely to misclassify legitimate entertainment news as fake than legitimate hard news, using a within-dataset design on GossipCop from FakeNewsNet. Across four frontier models, we find a clear but model-specific genre asymmetry: DeepSeek-V3.2 and GPT-5.2 show false-positive-rate gaps of 10.1 and 8.8 percentage points, respectively (both $p < .001$), whereas Claude Opus 4.6 and Gemini 3 Flash show no comparable difference. A style-swap experiment yields only limited and inconsistent changes, suggesting that the asymmetry is not reducible to stylistic register alone. Prompt-based mitigation is likewise possible but not generic: framing the model as an entertainment-news fact-checker reduces false positives for DeepSeek-V3.2 by about 50\% without detectable recall loss, but offers little improvement for GPT-5.2. Exploratory qualitative coding further suggests two recurring error patterns in sampled false positives: treating private-life claims as inherently unverifiable and discounting entertainment journalism as an epistemically weaker genre. Taken together, these findings show that aggregate performance metrics can obscure structured false positives within legitimate journalism. We argue that LLM-based credibility assessment may not only evaluate truth claims but also differentially recognize the legitimacy of journalistic genres, and that evaluation should therefore include genre-stratified false-positive analysis alongside overall accuracy. 

---
# Model Routing as a Trust Problem: Route Receipts for Adaptive AI Systems 

**Authors**: Vincent Schmalbach  

**Link**: [PDF](https://arxiv.org/pdf/2605.01710)  

**Abstract**: AI products often route requests through version aliases, service tiers, tool choices, regional endpoints, fallback rules, or safety handling before responding. These routing steps are documented product surfaces in several widely used AI platforms and serving stacks.
Routing helps AI services stay affordable, fast, and available at scale, and it shapes trust. Trust can break when routing changes the cost, quality, or accountability of a response without the user being able to tell what happened. "Which model answered?" is only part of the audit question. The runtime path matters.
Adaptive AI systems should produce a runtime transparency artifact called the route receipt. A route receipt is a compact record of the route that served a request. It should capture enough material facts for people relying on the output to reconstruct important routing decisions without exposing proprietary internals or hidden reasoning.
Route transparency should be part of model documentation. Model cards describe trained model artifacts, while route receipts describe the runtime conditions under which a particular answer was produced. The paper introduces the route-receipt concept, a minimal schema and redaction model, and a documentation-based survey of selected platforms showing that receipt fragments already exist without a portable per-answer record. 

---
# Latent State Design for World Models under Sufficiency Constraints 

**Authors**: Keon Woo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.01694)  

**Abstract**: A world model matters to an agent only through the state it constructs. That state must preserve some information, discard other information, and support some future function: prediction, control, planning, memory, grounding, or counterfactual reasoning. This paper treats world-model research as latent state design under sufficiency constraints.
We propose a functional taxonomy that groups methods by what their latent state is for, rather than by architecture or application domain: predictive embedding, recurrent belief state, object/causal structure, latent action interface, grounded planning interface, and memory substrate. These roles expose distinctions that architecture-based groupings hide, including the gap between predictive sufficiency and control sufficiency, and the gap between passive video prediction and counterfactual action modeling.
The taxonomy supports an evaluation framework that judges a model by the sufficiency constraint its latent state was built to satisfy. We compare methods along seven axes: representation, prediction, planning, controllability, causal/counterfactual support, memory, and uncertainty. We use the resulting matrix as a diagnostic for what a latent state preserves, discards, and enables.
The conclusion that follows is that an actionable world model is the one whose state construction matches the task, not the one that preserves the most information. 

---
# CP-SynC: Multi-Agent Zero-Shot Constraint Modeling in MiniZinc with Synthesized Checkers 

**Authors**: Yuliang Song, Eldan Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2605.01675)  

**Abstract**: Constraint Programming (CP) is a powerful paradigm for solving combinatorial problems, yet translating natural language problem descriptions into executable models remains a significant bottleneck. While Large Language Models (LLMs) show promise in automating this translation, they often struggle with subtle semantic errors in the absence of oracle validation at test time. To address this, we introduce CP-SynC (Constraint Programming modeling with Synthesized Checkers), a multi-agent workflow for zero-shot constraint modeling in MiniZinc. CP-SynC coordinates modeling agents that generate and refine candidate models and validation agents that synthesize semantic checkers to provide feedback on semantic correctness. To mitigate noise inherent in individual LLM outputs, CP-SynC explores multiple modeling trajectories in parallel and employs selection agents to select the final model via multi-agent evidence aggregation. Extensive experiments on a benchmark of 100 CP problems show that CP-SynC substantially outperforms existing baselines in MiniZinc modeling. 

---
# Evaluating Agentic AI in the Wild: Failure Modes, Drift Patterns, and a Production Evaluation Framework 

**Authors**: Mukund Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2605.01604)  

**Abstract**: Existing evaluation frameworks for large language models -- including HELM, MT-Bench, AgentBench, and BIG-bench -- are designed for controlled, single-session, lab-scale settings. They do not address the evaluation challenges that emerge when agentic AI systems operate continuously in production: compounding decision errors, tool failure cascades, non-deterministic output drift, and the absence of ground truth for long-horizon tasks. This paper makes three contributions. First, we present a taxonomy of seven failure modes unique to production agentic systems, each grounded in observations from systems operating at billion-event scale. Second, we demonstrate empirically where standard metrics -- ROUGE, BERTScore, accuracy/AUC, and the agentic benchmarks above -- fail to detect each failure mode. Third, we propose PAEF (Production Agentic Evaluation Framework), a five-dimension evaluation framework with an open-source reference implementation, designed for continuous evaluation on production traffic rather than episodic benchmark runs. Our analysis shows that standard metrics fail to detect four of the seven failure modes entirely and detect three others only after a lag of multiple evaluation cycles. 

---
# Multi-Agent Reasoning Improves Compute Efficiency: Pareto-Optimal Test-Time Scaling 

**Authors**: Florian Valentin Wunderlich, Lars Benedikt Kaesberg, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2605.01566)  

**Abstract**: Advances in inference methods have enabled language models to improve their predictions without additional training. These methods often prioritize raw performance over cost-effective compute usage. However, computational efficiency is key for real-world applications with resource constraints. We provide a systematic analysis of the inference scaling strategies self-consistency, self-refinement, multi-agent debate, and mixture-of-agents, to study their computational performance tradeoffs. We evaluate methods on two reasoning benchmarks (MMLU-Pro, BBH) and include extensive parameter configurations (e.g., scaling the number of parallel predictions, agents, and debate rounds) across different model sizes. Across 34 configurations and over 100 evaluations, we compute the Pareto-optimal front to select methods that achieve the best accuracy with the lowest computational budget. Notably, inference scaling improves accuracy by up to +7.1% points over chain-of-thought at the highest evaluated budgets (20x the CoT compute budget) on MMLU-Pro. With an equal computing budget, debate and mixture-of-agents outperform self-consistency by 1.3% and 2.7% points, respectively. While self-consistency saturates earlier, multi-agent gains persist, particularly on more complicated tasks. We identify a simple multi-agent design guideline: mixture-of-agents is most efficient when the number of parallel generations exceeds the number of sequential aggregations. 

---
# MILD: Mediator Agent System with Bidirectional Perception and Multi-Layered Alignment for Human-Vehicle Collaboration 

**Authors**: Jiyao Wang, Yunbiao Wang, Yubo Jiao, Xiao Yang, Dengbo He, Sasan Jafarnejad, Luis Miranda-Moreno, Raphael Frank, Jiangbo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01507)  

**Abstract**: Prior studies report that partial driving automation can increase the cognitive demands on human drivers. This effect largely arises from human drivers' lack of transparent insight into the vehicle's intentions and decision logic, as well as from automated systems' limited awareness of the driver's dynamic state and preferences. This bidirectional misalignment undermines shared situational awareness and exacerbates coordination failures in human-vehicle interaction. To address these limitations, we argue for a paradigm shift that elevates the human role from passive supervisor to active manager. We introduce the Mediator-in-the-Loop-Driving (MILD) system, based on an agentic system architecture to facilitate synergistic human-vehicle collaboration. MILD integrates a perception agent for joint in-cabin and out-of-cabin understanding with a lightweight strategy agent that generates compliant and explainable action suggestions. To ensure these strategies are strictly aligned with safety regulations and human values, we develop Evidence- and Constraint-weighted Policy Optimization (ECPO). ECPO leverages automatic validators to steer the agent toward behaviors that are not only accurate but also structurally complete, substantiated by evidence, and free from constraint violations. Furthermore, a retrieval-augmented generation module dynamically incorporates constraints from traffic regulations, speed recommendations, and driver preferences into the decision loop. Field experiments across three open datasets demonstrate that MILD consistently outperforms baselines in both perception accuracy and strategy quality under auditable offline metrics, and yields higher human-rated policy adequacy, comfort, and explanation than baselines. This work offers a practical pathway for building auditable and aligned agents for human-vehicle collaborative driving. 

---
# SciResearcher: Scaling Deep Research Agents for Frontier Scientific Reasoning 

**Authors**: Tianshi Zheng, Rui Wang, Xiyun Li, Yangqiu Song, Tianqing Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01489)  

**Abstract**: Frontier scientific reasoning is rapidly emerging as a key foundation for advancing AI agents in automated scientific discovery. Deep research agents offer a promising approach to this challenge. These models develop robust problem-solving capabilities through post-training on information-seeking tasks, which are typically curated via knowledge graph construction or iterative web browsing. However, these strategies face inherent limitations in frontier science, where domain-specific knowledge is scattered across sparse and heterogeneous academic sources, and problem solving requires sophisticated computation and reasoning far beyond factual recall. To bridge this gap, we introduce SciResearcher, a fully automated agentic framework for frontier-science data construction. SciResearcher synthesizes diverse conceptual and computational tasks grounded in academic evidence, while eliciting information acquisition, tool-integrated reasoning, and long-horizon capabilities. Leveraging the curated data for supervised fine-tuning and agentic reinforcement learning, we develop SciResearcher-8B, an agent foundation model that achieves 19.46% on the HLE-Bio/Chem-Gold benchmark, establishing a new state of the art at its parameter scale and surpassing several larger proprietary agents. It further achieves 13-15% absolute gains on SuperGPQA-Hard-Biology and TRQA-Literature benchmarks. Overall, SciResearcher introduces a new paradigm for automated data construction for frontier scientific reasoning and offers a scalable path toward future scientific agents. 

---
# MAP-Law: Coverage-Driven Retrieval Control for Multi-Turn Legal Consultation 

**Authors**: Qinchuan Cheng, Ruixuan Xie, Jiaqi Liu, Xiaoya Yuan, Yuxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01486)  

**Abstract**: Legal consultation is a high-stakes, knowledge-intensive task that requires agents to identify relevant legal issues, retrieve authoritative support, and determine when evidence is sufficient for a recommendation. Although retrieval-augmented generation has improved grounding in legal question answering, many multi-turn legal agents still rely on fixed retrieval depth or coarse heuristic control. This often leads to either insufficient support for key legal elements or excessive retrieval that increases context burden and weakens answer focus.
We propose MAP-Law, a coverage-driven framework for retrieval control in multi-turn legal consultation. MAP-Law models consultation as a controlled retrieval process over a joint structured state consisting of issue nodes, legal element nodes, and evidence nodes. After each retrieval round, the agent computes Element Coverage, Evidence Coverage, and Marginal Gain, and uses these signals to decide whether to continue retrieval, redirect the search, or generate the final response. In this way, MAP-Law turns stopping from a fixed hyperparameter into an interpretable and auditable decision aligned with legal argumentative structure.
Experiments on a self-constructed dataset of 50 cases across eight labor-law scenarios show that MAP-Law with DeepSeek as the action selector achieves an Element Coverage of 0.860 using only 2.9 retrieval rounds and 5.8 evidence pieces on average. Compared with a fixed seven-round baseline, it reduces evidence volume by over 80% and retrieval rounds by 58%. Ablation results further confirm the independent contributions of coverage-driven stopping, joint graph representation, and LLM-based action selection. 

---
# Grounding Multi-Hop Reasoning in Structural Causal Models via Group Relative Policy Optimization 

**Authors**: Yunhan Bu, Quan Zhang, Huaping Zhang, Guotong Geng, Chunxiao Gao, Askar Hamdulla, Juan Wang, Qiuchi Li, Baohua Zhang, Shuai Lei, Yunbo Cao, Zhunchen Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.01482)  

**Abstract**: Multi-Hop Fact Verification (MHFV) necessitates complex reasoning across disparate evidence, posing significant challenges for Large Language Models (LLMs) which often suffer from hallucinations and fractured logical chains. Existing methods, while improving transparency via Chain-of-Thought (CoT), lack explicit modeling of the causal dependencies between evidence and claims. In this work, we introduce a novel framework that grounds reasoning in a Structural Causal Model (SCM), treating verification as a constructive causal inference process. We empirically identify an "inverted U-shaped" correlation between reasoning chain length and accuracy, revealing that excessive structural complexity degrades performance. To address this, we propose a Rule-based Reinforcement Learning strategy using Group Relative Policy Optimization (GRPO). This approach dynamically optimizes the trade-off between structural depth and conciseness. Extensive experiments on HoVer and EX-FEVER demonstrate that our SCM-GRPO framework significantly outperforms state-of-the-art baselines, offering a reliable and interpretable solution for complex fact verification. 

---
# CoFlow: Coordinated Few-Step Flow for Offline Multi-Agent Decision Making 

**Authors**: Guowei Zou, Haitao Wang, Beiwen Zhang, Boning Zhang, Hejun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01457)  

**Abstract**: Generative models have emerged as a major paradigm for offline multi-agent reinforcement learning (MARL), but existing approaches require many iterative sampling steps. Recent few-step accelerations either distill a joint teacher into independent students or apply averaged velocities independently per agent, suggesting that few-step inference requires sacrificing inter-agent coordination. We show this trade-off is not necessary: single-pass multi-agent generation can preserve coordination when the velocity field is natively joint-coupled. We propose Coordinated few-step Flow (CoFlow), an architecture that combines Coordinated Velocity Attention (CVA) with Adaptive Coordination Gating. A finite-difference consistency surrogate further replaces memory-prohibitive Jacobian-vector product backpropagation through the averaged velocity field with two stop-gradient forward passes. Across 60 configurations spanning MPE, MA-MuJoCo, and SMAC, CoFlow matches or surpasses Gaussian / value-based, transformer, diffusion, and prior flow baselines on episodic return. Three independent coordination probes confirm that the gains flow through inter-agent coordination rather than per-agent capacity. A denoising-step sweep shows that single-pass inference suffices on every configuration. CoFlow reaches state-of-the-art coordination quality in 1-3 denoising steps under both centralized and decentralized execution. Project page: this https URL. 

---
# Rethinking Explanations: Formalizing Contrast in Description Logics 

**Authors**: Yasir Mahmood, Arnab Sharma, Axel-Cyrille Ngonga Ngomo, Balram Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2605.01442)  

**Abstract**: There has been a growing interest in explaining entailments over description logic (DL) knowledge bases. The existing explanation formalisms focus on justifications to explain true axioms, and abductive reasoning to explain missing axioms in a knowledge base. However, these formalisms only point out the reasoning steps behind a (missing) entailment and lack a user-centered approach as they do not consider an inquirer's needs, level of understanding, or prior knowledge. We propose contrastive explanations, aiming at answering "why an axiom P (fact) is true instead of another axiom Q (foil)" over description logic knowledge bases. The motivation arises from the observation that when a user discovers that P has occurred, they are often surprised because they anticipated the occurrence of another similar event Q. Furthermore, individual explanations for "why P" and "why not Q" are unsatisfactory since a user expects to see the difference between P and Q. In this work, we first present formal foundations of contrasting questions and then define contrastive explanations within description logics. To this end, facts include ABox assertions of the form C(x) for a concept C and individual x. Possible foils for such facts are assertions C(y) (contrasting against an individual y), or D(x) (contrasting against a concept D). Additionally, we explore the properties of contrastive explanations in the DL EL and ALC. We also provide an implementation of our definition and an experimental evaluation on KBs of varying sizes. 

---
# SCALE-LoRA: Auditing Post-Retrieval LoRA Composition with Residual Merging and View Reliability 

**Authors**: Shuaipeng Zhou, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01429)  

**Abstract**: Libraries of Low-Rank Adaptation (LoRA) adapters are becoming a practical by-product of parameter-efficient adaptation. Once such adapters accumulate, a natural question is no longer how to train one adapter for one task, but how to reuse an open pool of adapters for a new task given only a small support set. Prior work has shown that LoRA modules can be composed at the task level and dynamically selected at the instance level. However, open-pool LoRA reuse is not automatic: retrieving relevant adapters does not guarantee that their parameter updates are compatible, and composing adapters does not guarantee reliable outputs.
We introduce the Sparse-Composition Agreement Layer (SCALE), a post-retrieval audit and composition framework for open-pool LoRA reuse. SCALE contains a deployable 1.0* merge path, Layer-Adaptive Sparse Residual Composition (LASRC), and a higher-cost reliability-analysis layer for multi-view disagreement. LASRC addresses merge interference by preserving a linear anchor while residualizing block-wise adapter update directions. The reliability layer treats disagreement among sparse composition views as an observable uncertainty signal and compares agreement, support-loss proxy selection, and oracle headroom under explicit path cost. In matched FLAN-T5-Large, BIG-Bench Hard (BBH), and 97-LoRA experiments, LASRC gives a directional single-view gain under fixed retrieval, while SCALE-support is reported as a query-label-free 3.0* reliability-analysis variant rather than as a calibrated or throughput-equivalent selector. Protocol-distinct BBH-8 validation shows the same qualitative trend on three decoder-only backbones. Detailed scores, paired audits, and path-cost records are reported in the experimental section. 

---
# Artificial Jagged Intelligence as Uneven Optimization Energy Allocation Capability Concentration, Redistribution, and Optimization Governance 

**Authors**: Wesley Shu, Peng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.01420)  

**Abstract**: Artificial Jagged Intelligence (AJI) denotes a recurring pattern in which large learning systems exhibit strong local capabilities while remaining weak or brittle in other domains. This paper develops a formal theory of AJI as uneven allocation of optimization pressure. We model training as a finite-budget process that distributes gradient-driven update energy across capability-relevant directions in parameter space. In this model, jagged capability profiles arise from anisotropic objective structure, data geometry, and representational coupling rather than from a single scalar quantity called intelligence.
The paper defines capability gain, optimization energy share, and jaggedness, then proves that persistent concentration of cumulative update energy yields lower bounds on dispersion in capability gains. A finite-budget tradeoff theorem shows why prioritizing one capability can impose opportunity costs on others unless positive coupling or shared structure offsets the cost. The analysis also studies redistribution mechanisms, including energy-variance regularization and auxiliary structural objectives, as interventions that reshape the optimization field.
The resulting framework links uneven emergence, training architecture, and optimization governance. It predicts that early concentration of update energy should forecast later capability jaggedness; that scaling under a narrow objective need not eliminate anisotropy; and that explicitly funded auxiliary objectives can revive neglected capabilities. AJI is therefore not merely a descriptive label for uneven model behavior, but a testable theory of how finite optimization resources produce concentrated, delayed, and structurally uneven capability formation. 

---
# TimeTok: Granularity-Controllable Time-Series Generation via Hierarchical Tokenization 

**Authors**: Seokhyun Lee, Jaeho Kim, Changjun Oh, Mihaela van der Schaar, Changhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.01418)  

**Abstract**: Time-series generative models often lack control over temporal granularity, forcing users to accept whatever granularity the model produces. To enable truly user-driven generation, we introduce TimeTok, a unified framework for Granularity-Controllable Time-Series Generation (GC-TSG), which generates time series at any target granularity from any coarser input (e.g., rough sketches) or from scratch. At the core of TimeTok is a hierarchical tokenization strategy that maps time series into an ordered sequence of tokens, from coarse to fine temporal granularity. Our autoregressive generation process operates across these granularity levels, producing token blocks that are decoded back into continuous time series. This design naturally enables GC-TSG - including standard generation - within a single framework, where controlling the number of token blocks provides explicit control over output detail. Experiments show that TimeTok excels at GC-TSG tasks while achieving state-of-the-art performance in standard generation. Furthermore, we showcase TimeTok's potential as a foundational tokenizer by training on multiple datasets with heterogeneous temporal granularities, verifying strong transferability that consistently outperforms models trained on individual datasets. To our knowledge, this is the first unified framework that covers the full generative spectrum for time series, offering a valuable foundation for models that benefit from diverse temporal granularities. 

---
# AI Safety as Control of Irreversibility: A Systems Framework for Decision-Energy and Sovereignty Boundaries 

**Authors**: Wesley Shu, Peng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.01415)  

**Abstract**: Recent AI systems compress the distance between capability growth and capability deployment. Earlier high-risk technologies were slowed by capital intensity, physical bottlenecks, organizational inertia, and specialized supply chains. By contrast, AI capabilities can be copied, invoked, embedded in workflows, and scaled across institutions at low marginal cost. This paper argues that declining deployment friction changes the safety problem at its root. Safety is not only local output correctness or preference alignment, but the control of irreversibility under rising decision density.
The paper formalizes this claim through decision-energy density: the rate-weighted capacity of a node to generate, evaluate, select, and execute consequential decisions. It then identifies three sovereignty boundaries that determine whether AI remains an amplifier within a human-governed system or becomes a de facto control center: irreversible decision authority, physical resource mobilization authority, and self-expansion authority. The model shows how efficiency pressure, path dependence, scale feedback, and weak boundary constraints concentrate decision-energy in the most efficient node. This concentration can diffuse responsibility and raise the probability of irreversible system-level loss even when local per-action error rates remain low.
The main result is a boundary stabilization theorem. It shows that safety need not require proving that advanced systems are always correct. Instead, it requires institutional and technical designs that prevent irreversible power from being released by a single high-efficiency node. The paper reframes AI safety as layered control, authorization, and externally reviewable limits, linking alignment, security engineering, organizational economics, and institutional design. 

---
# A Cellular Doctrine of Morality: Intrinsic Active Precision and the Mind-Reality Overload Dilemma 

**Authors**: Ahsan Adeel  

**Link**: [PDF](https://arxiv.org/pdf/2605.01376)  

**Abstract**: Current AI systems, grounded in oversimplified neuroscience, risk eroding the distinction between truth and falsehood. They maximize reward by amplifying attention to information without intrinsic precision mechanisms to assess whether it is valid or worth attending to. This increases both the volume of information and the inherent biases in what the system attends to, whether true, false, or irrelevant. If not corrected, this trend will accelerate, threatening to overload systems and individuals with biased and dubious information and increasing the risk of confusion, poor judgment, and irrational or harmful decisions and behaviour, a condition I term the mind-reality overload dilemma. I argue that this threat may be mitigated by providing the public with access to more advanced AI tools built on the biophysical dynamics of pyramidal neurons underlying awake thought and higher-order cognition. These neurons support an intrinsic active precision mechanism that, rather than merely maximizing reward, uses locally and globally coherent predictions to evaluate the validity and contextual adequacy of evidence before it is attended to or propagated through hierarchies, prioritizing coherence and adequacy before attention.~While this approach does not derive or prescribe moral rules from biology, it may give rise to AI with more "real understanding", helping restore epistemic conditions by reducing information overload and amplifying reliable information, thereby supporting the formation of better-informed beliefs and more coherent judgments that benefit society at large-though no guarantees exist. 

---
# Structural Ranking of the Cognitive Plausibility of Computational Models of Analogy and Metaphors with the Minimal Cognitive Grid 

**Authors**: Alessio Donvito, Antonio Lieto  

**Link**: [PDF](https://arxiv.org/pdf/2605.01359)  

**Abstract**: In this paper, we employ the Minimal Cognitive Grid (MCG), a framework created to evaluate the cognitive plausibility of artificial systems, to offer a systematic assessment of leading computational models of analogy and metaphor, including the Structure-Mapping Engine (SME), CogSketch, METCL, and Large Language Models (LLMs). We present a formal and quantitative operationalization of the MCG framework and, through the analysis of its three main dimensions (Functional/Structural Ratio, Generality, and Performance Match), examine how well each system aligns with standard cognitive theories of the modeled phenomena, thus allowing for comparison of the models with respect to their cognitive plausibility, according to consistent and generalizable mathematical criteria. 

---
# DiagramNet: An End-to-End Recognition Framework and Dataset for Non-Standard System-Level Diagrams 

**Authors**: Jincheng Lou, Ruohan Xu, Jiapeng Li, Junyin Pi, Runzhe Tao, Weijian Fan, Xiao Tan, Guojie Luo, Yibo Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.01338)  

**Abstract**: System-level diagrams encode the architectural blueprint of chip design, specifying module functions, dataflows, and interface protocols. However, non-standardized symbols and the scarcity of structured training data hinder existing multimodal large language models (MLLMs) from recognizing these diagrams. To address this gap, we introduce DiagramNet, the first multimodal dataset for system-level diagrams, comprising 10,977 connection annotations and 15,515 chain-of-thought QA pairs across four tasks: Listing, Localization, Connection, and Circuit QA. Building on this dataset, we propose a progressive training pipeline together with a decoupled multi-agent workflow that decomposes complex visual reasoning into Perception, Reasoning, and Knowledge stages. On the DiagramNet benchmark, integrating our 3B-parameter model with the proposed workflow surpasses the 2025 EDA Elite Challenge winner and outperforms GPT-5, Claude-Sonnet-4, and Gemini-2.5-Pro by over 2x in end-to-end evaluation. Notably, the workflow generalizes beyond our model, boosting Task 1 performance by 128.7x for Gemini-2.5-Pro and 12.4x for GPT-5. Furthermore, with only 60 images for detector adaptation, the method transfers effectively to AMSBench, achieving zero-shot connectivity reasoning on par with GPT-5 and Claude-Sonnet-4 while surpassing the AMS state-of-the-art method Netlistify. 

---
# Truth or Tribe: How In-group Favoritism Prioritize Facts in Persona Agents 

**Authors**: Shijun Lei, Hongyu Wang, Yunji Liang, Haowen Zheng, Bin Guo, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01329)  

**Abstract**: In-group favoritism refers to the phenomena of favoring members of one's in-group over out-group members and is widely observed in numerous social cooperative behaviors. Recently, in-group favoritism biases have also been identified in generative language models. However, whether the in-group favoritism exists when persona agents are faced with contradicting information (e.g., misinformation), and how to mitigate the adverse effects of in-group favoritism biases in persona agents have been understudied. To address these problems, we propose a Truth or Tribe simulation framework to study the agent cooperation within the spread of contradicting information through a triadic interaction paradigm, and conduct controlled trials to evaluate the primary moderating factors. Extensive results showcase that persona agents display strong in-group favoritism, accepting incorrect answers from identity-similar peers at much higher rates than from dissimilar peers. In-group favoritism continues to emerge in defeasible reasoning contexts where no absolute truth exists, and it intensifies as cognitive complexity increases. Furthermore, three intervention strategies--Identity-Blind Instruction, Structured Counterfactual Reasoning, and Heterogeneous Perspective Ensemble--are proposed to mitigate the in-group favoritism. 

---
# Segment-Aligned Policy Optimization for Multi-Modal Reasoning 

**Authors**: Lei Gao, Zhuoming Li, Mengxi Jia, Jiakang Yuan, Hongbo Sun, Hao Sun, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.01327)  

**Abstract**: Existing reinforcement learning approaches for Large Language Models typically perform policy optimization at the granularity of individual tokens or entire response sequences. However, such formulations often misalign with the natural step-wise structure of reasoning processes, leading to suboptimal credit assignment and unstable training in multi-modal reasoning tasks. To bridge this gap, we propose Segment-Aligned Policy Optimization (SAPO), a novel reinforcement learning paradigm that treats coherent reasoning steps, rather than tokens or full sequences as fundamental units of policy update. SAPO introduces a step-wise Markov decision process abstraction over reasoning segments, accompanied by segment-level value estimation, advantage computation, and importance sampling mechanisms that are semantically aligned with reasoning boundaries. Experiments on representative reasoning benchmarks demonstrate that SAPO consistently outperforms token-level and sequence-level policy optimization methods, achieving significant accuracy improvements while exhibiting better training stability and value estimation consistency. Our work underscores the importance of aligning reinforcement learning updates with the intrinsic structure of reasoning, paving the way for more efficient and semantically grounded policy optimization in complex reasoning tasks. Codes and models will be released to ensure full reproducibility. 

---
# Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks 

**Authors**: Jie-Jing Shao, Haiyan Yin, Yueming Lyu, Xingrui Yu, Lan-Zhe Guo, Ivor Tsang, James Kwok, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.01293)  

**Abstract**: Foundation model-driven agents often struggle with long-horizon planning due to the transient nature of purely prompting-based reasoning. While existing skill induction methods mitigate this by distilling experience into state-blind parameterized scripts, they fail to capture the conditional logic required for robust execution in dynamic environments. In this paper, we propose Neuro-Symbolic Skill Induction (NSI), a framework that lifts interaction traces into modular, \textit{logic-grounded} programs. By synthesizing explicit control flows and dynamic variable binding, NSI empowers agents to discover \textit{when} and \textit{why} to act. This paradigm enables the efficient generalization, allowing agents to induce skills from few-shot examples and flexibly adapt to unseen goals. Experiments on a series of agentic tasks demonstrate that NSI consistently outperforms state-of-the-art baselines, empowering agents to self-evolve into architects of logic-grounded skills. 

---
# Valley3: Scaling Omni Foundation Models for E-commerce 

**Authors**: Zeyu Chen, Guanghao Zhou, Qixiang Yin, Ziwang Zhao, Huanjin Yao, Pengjiu Xia, Min Yang, Cen Chen, Minghui Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01278)  

**Abstract**: In this work, we present Valley3, an omni multimodal large language model (MLLM) developed for diverse global e-commerce tasks, with unified understanding and reasoning capabilities across text, images, video, and audio. A key feature of Valley3 is its native multilingual audio capability for e-commerce, developed by extending vision-language models to better support crucial audio-visual tasks, particularly in short-video scenarios. To achieve this, we carefully design a four-stage omni e-commerce continued pre-training pipeline, through which Valley3 progressively acquires audio understanding, cross-modal instruction-following, e-commerce domain knowledge, and long-context reasoning capabilities, ultimately evolving into an omni model for diverse e-commerce scenarios. Then, we further improve Valley3 through post-training to encourage long-chain reasoning with controllable reasoning modes, enabling one non-thinking mode and three distinct levels of thinking, thereby balancing inference efficiency in simple scenarios with deep reasoning for complex applications. Moreover, we equip Valley3 with agentic search capabilities to proactively invoke search tools and acquire task-relevant information for e-commerce deep research tasks. To comprehensively assess the capabilities of Valley3, we construct an omni e-commerce benchmark spanning 6 tasks. Experimental results show that Valley3 consistently outperforms strong baselines on our in-house and open-source e-commerce benchmarks, while remaining competitive on general-domain benchmarks. 

---
# Uncertainty-Aware Trip Purpose Inference from GPS Trajectories via POI Semantic Zones and Pareto Calibration 

**Authors**: Bo Yang, Haoxuan Ma, Yifan Liu, Zhiyuan Zhang, Chris Stanford, Morgan Sun, Jiaqi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.01257)  

**Abstract**: Large-scale GPS trajectory data offer rich observations of human mobility, yet assigning trip purposes to detected stops remains challenging due to the absence of individual-level ground truth, spatial uncertainty from GPS noise and incomplete points of interest (POIs) coverage, and fundamental behavioral differences across trip purposes. We propose a weakly supervised framework integrating neighborhood-level POI semantic zones with distance-weighted spatial likelihoods, differentiated inference strategies for mandatory and non-mandatory activities, and a multi-phase Pareto optimization that jointly minimizes distributional divergence from household travel survey statistics and maximizes inference reliability without requiring annotated labels. Evaluated on over 81 million staypoints in Los Angeles, the framework reduces activity type frequency Jensen-Shannon distance (JSD) by 23%, start time JSD by 48%, and duration JSD by 12% respectively relative to a comparable baseline. The proposed approach provides a scalable and uncertainty-aware path from raw GPS trajectories to semantically annotated mobility data for travel demand modeling and transportation policy analysis. 

---
# EO-Gym: A Multimodal, Interactive Environment for Earth Observation Agents 

**Authors**: Sai Ma, Zhuang Li, Sichao Li, Xinyue Xu, Ruibiao Zhu, Tony Boston, John A. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2605.01250)  

**Abstract**: Earth Observation (EO) analysis is inherently interactive: resolving uncertainty often requires expanding the region of interest, retrieving historical observations, and switching across sensors such as optical and Synthetic Aperture Radar. However, most EO benchmarks collapse this process into fixed-input, single-turn tasks. To address this gap, we present EO-Gym, a controlled executable framework for multimodal, tool-using EO agents that formulates EO analysis as a Gymnasium-style local geospatial workspace backed by more than 660k multimodal files indexed by location, time, and sensor type, with 35 EO-specialized tools spanning six task families. Built on this environment, we construct EO-Gym-Data, a benchmark of 9,078 trajectories and 34,604 reasoning steps, and grounded in eight public EO datasets together with Landsat and Sentinel-2 imagery. Evaluating $10$ open and closed VLMs shows that strong general-purpose models still struggle with interactive EO reasoning, especially on temporal and cross-modal workflows. As a reference baseline, EO-Gym-4B, obtained by fine-tuning Qwen3-VL-4B-Instruct on EO-Gym-Data, improves overall Pass@3 from $0.49$ to $0.74$ under the main evaluation setting. O-Gym provides a reproducible environment for interactive EO agents, operationalizing EO as an evidence-gathering problem that requires planning across geospatial, temporal, and sensing modality. 

---
# Zero-Shot Signal Temporal Logic Planning with Disjunctive Branch Selection in Dynamic Semantic Maps 

**Authors**: Bowen Ye, Ancheng Hou, Junyue Huang, Ruijia Liu, Xiang Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.01222)  

**Abstract**: Signal Temporal Logic (STL) offers verifiable task specifications and is crucial for safety-critical control. Yet STL planning remains challenging: exact optimization-based methods are often too slow, and learning-based methods struggle to generalize across varying environments. We propose a zero-shot STL planning solver for variable-map environments that generates feasible trajectories without retraining. By integrating a map-conditioned Transformer architecture with a lightweight heuristic, our approach effectively handles complex disjunctive (OR) subformulas. Furthermore, we leverage Transitive Reinforcement Learning (TRL) to ensure consistent temporal grounding and logical coherence across decomposed sub-tasks. Experiments on dynamic semantic maps with diverse obstacle layouts demonstrate consistent gains, highlighting the framework's superior zero-shot generalization to changing environments and broad STL coverage. 

---
# Agentic AI Systems Should Be Designed as Marginal Token Allocators 

**Authors**: Siqi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01214)  

**Abstract**: This position paper argues that agentic AI systems should be designed and evaluated as \emph{marginal token allocation economies} rather than as text generators priced by the unit. We follow a single request -- a developer asking a coding agent to fix a failing test -- through four economic layers that today are designed in isolation: a router that decides which model answers, an agent that decides whether to plan, act, verify, or defer, a serving stack that decides how to produce each token, and a training pipeline that decides whether the trace is worth learning from. We show that all four layers are solving the \emph{same} first-order condition -- marginal benefit equals marginal cost plus latency cost plus risk cost -- with different index sets and different prices. The framing is deliberately minimal: we do not propose a complete theory of AI economics. But adopting marginal token allocation as the shared accounting object explains why systems that locally minimize tokens globally misallocate them, predicts a small set of recurring failure modes (over-routing, over-delegation, under-verification, serving congestion, stale rollouts, cache misuse), and points to a concrete research agenda in token-aware evaluation, autonomy pricing, congestion-priced serving, and risk-adjusted RL budgeting. 

---
# Faithful Mobile GUI Agents with Guided Advantage Estimator 

**Authors**: Haowen Hu, Pengzhou Cheng, Zheng Wu, Lingzhong Dong, Gongshen Liu, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01208)  

**Abstract**: Vision-language model based graphical user interface (GUI) agents have shown strong interaction capabilities. However, they often behave unfaithfully, relying on memorized shortcuts rather than grounding actions in displayed screen evidence or user instructions. To address this, we propose Faithful-Agent, a faithfulness-first framework that reformulates GUI interaction to prioritize evidence groundedness and internal consistency. Faithful-Agent employs a two-stage pipeline: (i) a faithfulness-oriented SFT stage to instill abstainment behaviors under evidence perturbations; (ii) an RFT stage that further amplifies faithfulness by introducing the guided advantage estimator (GuAE), an anchor-based and variance-adaptive advantage tempering mechanism built upon GRPO. GuAE prevents advantage collapse in low-variance rollout groups under sparse GUI rewards, and with a thought-action consistency reward, Faithful-Agent (Stage II) elevates the Trap SR from 13.88\% to 80.21\% relative to the baseline, while preserving robust general instruction-following performance. 

---
# GR-Ben: A General Reasoning Benchmark for Evaluating Process Reward Models 

**Authors**: Zhouhao Sun, Xuan Zhang, Xiao Ding, Bibo Cai, Li Du, Kai Xiong, Xinran Dai, Fei Zhang, weidi tang, Zhiyuan Kan, Yang Zhao, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01203)  

**Abstract**: Currently, process reward models (PRMs) have exhibited remarkable potential for test-time scaling. Since large language models (LLMs) regularly generate flawed intermediate reasoning steps when tackling a broad spectrum of reasoning and decision-making tasks, PRMs are required to possess capabilities for detecting process-level errors in real-world scenarios. However, existing benchmarks primarily focus on mathematical reasoning, thereby failing to comprehensively evaluate the error detection ability of PRMs across diverse reasoning scenarios. To mitigate this gap, we introduce GR-Ben, a process-level benchmark specifically designed for assessing PRM's performance across two primary reasoning domains (science and logic) and nine subdomains. We conduct extensive experiments on a diverse set of 22 models, encompassing both PRMs and LLMs, and derive two key findings: (1) In domains beyond mathematical reasoning, the error-detection ability of existing PRMs and LLMs is found to be markedly weaker by comparison.(2) In general, PRMs are less adept at identifying knowledge-based errors, whereas LLMs exhibit poorer performance in detecting computational this http URL hope GR-Ben can foster future researches on PRMs for general domains, thereby enhancing the reasoning capabilities of LLMs. 

---
# NEURON: A Neuro-symbolic System for Grounded Clinical Explainability 

**Authors**: Anuradha Chandrasekaran, Dimitrios Zikos, Mutlu Mete, Alan Pang, Brady D. Lund, Kewei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2605.01189)  

**Abstract**: Clinical AI adoption is hindered by the black-box/grey-box nature of high-performing models, which lack the ontological grounding and narrative transparency required for professional-level explainability. We present NEURON, a neuro-symbolic system designed to enhance both predictive reliability and clinical interpretability. NEURON integrates SNOMED CT ontology-informed structural representations with machine learning models to bridge the gap between raw data and medical nomenclature. To facilitate human-aligned interaction, the system utilizes a Retrieval-Augmented Generation (RAG) grounded LLM layer to synthesize SHAP feature attributions and patient-specific clinical notes into coherent, natural-language explanations. Validated on the MIMIC-IV dataset for Acute Heart Failure mortality prediction, NEURON improved the AUC from 0.74-0.77 to 0.84-0.88 and significantly outperformed raw SHAP visualizations in human-aligned metrics (0.85 vs. 0.50). Our results demonstrate that NEURON offers a robust, scalable engineering solution for deploying trustworthy, human-centered connected health applications. 

---
# LLMs Should Not Yet Be Credited with Decision Explanation 

**Authors**: Wenshuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01164)  

**Abstract**: This position paper argues that LLMs should not yet be credited with decision explanation. This matters because recent work increasingly treats accurate behavioral prediction, plausible rationales, and outcome-conditioned reasoning traces as evidence that LLMs explain why people decide as they do, risking a premature redefinition of what counts as explanatory progress in human decision modeling. We first distinguish three claims with different evidential burdens: decision prediction, rationale generation, and decision explanation. We then argue that the evidence most commonly offered for LLM-based decision accounts directly supports the first two claims, and sometimes explanatory hypothesis generation, but does not distinguish decision explanation from prediction-supportive rationalization. Next, we propose a bridge standard for decision-explanation credit: stronger claims should specify explanatory targets, discriminate against weaker rationalizer alternatives, use target-appropriate process- or intervention-sensitive validation, and bound their scope. We then situate this standard against competing views and related literatures, clarifying why it preserves the value of LLMs as predictors, narrators, and hypothesis generators while resisting premature explanatory credit. We conclude with a principle of credit calibration: LLMs should be credited for the strongest claim their evidence warrants, and no stronger; if adopted, this principle can help turn LLMs from persuasive narrators of decisions into more reliable instruments for discovering, testing, and communicating explanations of human behavior. 

---
# Arithmetic in the Wild: Llama uses Base-10 Addition to Reason About Cyclic Concepts 

**Authors**: Sheridan Feucht, Tal Haklay, Usha Bhalla, Daniel Wurgaft, Can Rager, Raphaël Sarfati, Jack Merullo, Thomas McGrath, Owen Lewis, Ekdeep Singh Lubana, Thomas Fel, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2605.01148)  

**Abstract**: Does structure in representations imply structure in computation? We study how Llama-3.1-8B reasons over cyclic concepts (e.g., "what month is six months after August?"). Even though Llama-3.1-8B's representations for these concepts are circularly structured, we find that instead of directly computing modular addition in the period of the cyclic concept (e.g., 12 for months), the model re-uses a generic addition mechanism across tasks that operates independently of concept-specific geometry. First, it computes the sum of its two inputs using base-10 addition (six + August=14). Then, it maps this sum back to cyclic concept space (14->February). We show that Llama-3.1-8B uses task-agnostic Fourier features to compute these sums--in fact, these features have periods that respect standard base-10 addition, e.g., 2, 5, and 10, rather than the cyclic concept period (e.g., 12 for months). Furthermore, we identify a sparse set of 28 MLP neurons re-used across all tasks (approximately 0.2% of the MLP at layer 18) that can be partitioned into disjoint clusters, each computing the sum for a Fourier feature with a different period. Our work highlights how an interplay between causal abstraction and feature geometry can deepen our mechanistic understanding of LMs. 

---
# Position: Safety and Fairness in Agentic AI Depend on Interaction Topology, Not on Model Scale or Alignment 

**Authors**: Tanav Singh Bajaj, Nikhil Singh, Karan Anand, Eishkaran Singh  

**Link**: [PDF](https://arxiv.org/pdf/2605.01147)  

**Abstract**: As large language models are increasingly deployed as interacting agents in high-stakes decisions, the AI safety community assumes that safety properties of individual models will compose into safe multi-agent behavior. This position paper argues that this assumption is fundamentally mistaken. In agentic AI, safety is determined by interaction topology, not model weights. When agents deliberate sequentially or aggregate via parallel voting with a judge, the structure of information flow and decision coupling dominates outcomes. Evidence across model families and scales reveals three persistent topology-driven pathologies: ordering instability, where system behavior depends primarily on agent sequence; information cascades, where early judgments propagate regardless of correctness; and functional collapse, where systems satisfy fairness metrics while abandoning meaningful risk discrimination. Contrary to intuition, scaling to more capable models strengthens these effects by increasing consensus formation and reducing the challenge of initial decisions. These failure modes are invisible to model-centric evaluation and alignment procedures. We argue that agentic AI must be treated as a dynamical system rather than a collection of aligned components. Interaction topology must become a primary target of safety evaluation and regulation, with systems required to demonstrate robustness across architectural variations before deployment. 

---
# A Low-Latency Fraud Detection Layer for Detecting Adversarial Interaction Patterns in LLM-Powered Agents 

**Authors**: Sheldon Yu, Yingcheng Sun, Hanqing Guo, Julian McAuley, Qianqian Tong  

**Link**: [PDF](https://arxiv.org/pdf/2605.01143)  

**Abstract**: Large Language Model (LLM)-powered agents demonstrate strong capabilities in autonomous task execution, tool use, and multi-step reasoning. However, their increasing autonomy also introduces a new attack surface: adversarial interactions can manipulate agent behavior through direct prompt injection, indirect content attacks, and multi-turn escalation strategies. Existing defense strategies focus on prompt-level filtering and rule-based guardrails, which are often insufficient when risk emerges gradually across interaction sequences. In this work, we propose a complementary defense mechanism: a low-latency fraud detection layer for detecting adversarial interaction patterns in LLM-powered agents. Instead of determining whether a single prompt is malicious, our approach models risk over interaction trajectories using structured runtime features derived from prompt characteristics, session dynamics, tool usage, execution context, and fraud-inspired signals. The detection layer can be implemented using lightweight models leading to low-latency real-time deployments. To evaluate the framework, we construct a synthetic corpus of 12,000 multi-turn agent interactions generated from parameterized templates that simulate realistic agentic workflows. Using 42 structured features and an XGBoost classifier, our detector achieves over 9 times faster than LLM-based detectors. Through the experiment and ablation studies, our work suggests that interaction-level behavioral detection should become a core component of deployment-time defense for LLM-powered agents. 

---
# To Use AI as Dice of Possibilities with Timing Computation 

**Authors**: Jia Li, Vipin Kumar, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01134)  

**Abstract**: The dominant noun-based modeling paradigm has fundamentally constrained AI development, precluding any adequate representation of the future as an open temporal dimension. This paper introduces a verb-based paradigm, together with precise definitions of \emph{timing computation} and \emph{possibility}, that enables AI to function as an effective instrument for realizing the grammar of our thought.
Applied to longitudinal EHR data from 3,276 breast cancer patients, the framework empirically demonstrates: (1) automatic discovery of clinically significant patient trajectories, and (2) counterfactual timing deduction. Both results are purely data-driven, require no prior domain knowledge, and, to our knowledge, represent the first such demonstrations in the machine learning literature. 

---
# Iterative Finetuning is Mostly Idempotent 

**Authors**: Zephaniah Roe, Jack Sanderson, Dang Nguyen, Julian Huang, Todd Nief, Aryan Shrivastava, Chenhao Tan, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2605.01130)  

**Abstract**: If a model has some behavioral tendency, such as sycophancy or misalignment, and it is trained on its own outputs, will the tendency be amplified in the next generation of models? We study this question by training a series of models where each model is finetuned on data generated by its predecessor, and the initial model is seeded with some persona or belief. We test three settings: supervised finetuning (SFT) on instruct models, synthetic document finetuning (SDF) on base models, and direct preference optimization (DPO). In the SFT and SDF settings, traits mostly decay or remain constant so that further finetuning cycles do nothing. In rare cases when amplification occurs, it generally comes at the cost of coherence. In the DPO setting, trait amplification can reliably occur when a model is continually trained with a preference for its own outputs, but vanishes when models are reinitialized at each cycle. Overall, our results suggest that amplification most likely comes from continual post-training, and limiting this stage may be an effective defense. For non-RL finetuning, trait amplification is rare and very sensitive to data quantity, making it significantly less likely to occur accidentally. Finally, the amplification-coherence tradeoff serves as a natural deterrent against trait amplification. 

---
# PERSA: Reinforcement Learning for Professor-Style Personalized Feedback with LLMs 

**Authors**: Ravi Ranjan, Utkarsh Grover, Xiaomin Lin, Agoritsa Polyzou  

**Link**: [PDF](https://arxiv.org/pdf/2605.01123)  

**Abstract**: Large language models (LLMs) can provide automated feedback in educational settings, but aligning an LLMs style with a specific instructors tone while maintaining diagnostic correctness remains challenging. We ask how can we update an LLM for automated feedback generation to align with a target instructors style without sacrificing core knowledge? We study how Reinforcement Learning from Human Feedback (RLHF) can adapt a transformer-based LLM to generate programming feedback that matches a professors grading voice. We introduce PERSA, an RLHF pipeline that combines supervised fine-tuning on professor demonstrations, reward modeling from pairwise preferences, and Proximal Policy Optimization (PPO), while deliberately constraining learning to style-bearing components. Motivated by analyses of transformer internals, PERSA applies parameter efficient fine-tuning. It updates only the top transformer blocks and their feed-forward projections, minimizing global parameter drift while increasing stylistic controllability. We evaluate our proposed approach on three code-feedback benchmarks (APPS, PyFiXV, and CodeReviewQA) using complementary metrics for style alignment and fidelity. Across both Llama-3 and Gemma-2 backbones, PERSA delivers the strongest professor-style transfer while retaining correctness, for example on APPS, it boosts Style Alignment Score (SAC) to 96.2% (from 34.8% for Base) with Correctness Accuracy (CA) up to 100% on Llama-3, and Gemma-2. Overall, PERSA offers a practical route to personalized educational feedback by aligning both what it says (content correctness) and, crucially, how it says it (instructor-like tone and structure). 

---
# New Bounds for Zarankiewicz Numbers via Reinforced LLM Evolutionary Search 

**Authors**: Jay Bhan, Nicole Nobili, Srinivasan Raghuraman, Patrick Langer  

**Link**: [PDF](https://arxiv.org/pdf/2605.01120)  

**Abstract**: The Zarankiewicz number $\textbf{Z}(m, n, s, t)$ is the maximum number of edges in a bipartite graph $G_{m, n}$ such that there is no complete $K_{s, t}$ bipartite subgraph. We determine for the first time the exact values of three Zarankiewicz numbers: $\textbf{Z}(11, 21, 3, 3)=116$, $\textbf{Z}(11, 22, 3, 3)=121$, and $\textbf{Z}(12, 22, 3, 3)=132$. We further establish lower bounds for 41 more Zarankiewicz numbers, including several that are within one edge of the best known upper bound, and we match the established value in four more closed cases. Our results are obtained using OpenEvolve, an open-source evolutionary algorithm based on Large Language Models (LLMs) that iteratively improves algorithms for generating mathematical constructions by optimizing a reward signal which we tailored for this specific problem. These findings provide new extremal graph constructions and demonstrate the potential of LLM-guided evolutionary search to contribute to mathematical research. In addition to presenting the resulting constructions, we report the generation algorithms produced, describe the relevant implementation details, and provide our computational costs. Our costs are remarkably low, at less than \$30 for each Zarankiewicz parameter combination, showing that LLM-guided evolutionary search can be an inexpensive, reproducible, and accessible tool for discovering new combinatorial constructions. 

---
# Towards Multi-Agent Autonomous Reasoning in Hydrodynamics 

**Authors**: Jinpai Zhao, Albert Cerrone, Joannes Westerink, Clint Dawson  

**Link**: [PDF](https://arxiv.org/pdf/2605.01102)  

**Abstract**: Single-agent systems (SAS) have become the default pattern for LLM-driven scientific workflows, but routing planning, tool use, and synthesis through a single context window comes with a well-known cost: as tool specifications and observational traces accumulate, the effective context available for each decision shrinks, and end-to-end reliability suffers. We present a multi-agent system (MAS) prototype for hydrodynamics in which specialized agents are coordinated through a Layer Execution Graph (LEG). A planner agent constructs query-specific execution topologies from natural-language routing heuristics that capture domain knowledge without hard-coding it as rigid control logic; specialist agents operate under strict tool allowlists and occupy complementary data-class roles. Between layers, consolidator agents fuse parallel outputs into concise briefs, and a reporter agent synthesizes the final response, while the runtime logs provenance for every tool invocation to support auditability. All benchmarks, ablations, and stress tests use Claude Sonnet~4.6 as the backbone model for both specialist and general-purpose agents. Evaluated on 37 queries spanning six complexity categories, the prototype achieves 93.6% factual precision with a 100% pass rate. Accuracy remains above 90% across runs from single-threaded to five independent parallel tracks, and under simulated loss of individual data sources the system degrades gracefully, still returning substantive partial answers. Together, these results suggest that planner-guided, graph-structured multi-agent orchestration can meaningfully alleviate the context-saturation bottlenecks that constrain monolithic single-agent architectures. 

---
# Virtual Speech Therapist: A Clinician-in-the-Loop AI Speech Therapy Agent for Personalized and Supervised Therapy 

**Authors**: Shakeel Sheikh, Patrick Marmaroli, MD Sahidullah, Slim Ouni, Fabrice Hirsch, Goncalo Leal, Bjorn W Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2605.01101)  

**Abstract**: This paper develops Virtual Speech Therapist (VST), an intelligent agent-based platform that streamlines stuttering assessment and delivers customized therapy planning through automated and adaptive AI-driven workflows. VST integrates state-of-the-art deep learning-based stuttering classification, and multi-agent large language model (LLM) reasoning to support evidence-based clinical decision-making. The VST begins with the acquisition and feature extraction of patient speech samples, followed by robust classification of stuttering types. Building on these outputs, VST initiates an agentic reasoning process in which specialized LLM agents autonomously generate, critique, and iteratively refine individualized therapy plans. A dedicated critic agent evaluates all generated therapy plans to ensure clinical safety, methodological soundness, and alignment with peer-reviewed evidence and established professional guidelines. The resulting output is a comprehensive, patient-specific therapy draft intended for clinician review. Incorporating clinician feedback, the system then produces a finalized therapy plan suitable for patient delivery, thereby maintaining a clinician-in-the-loop paradigm. Experimental evaluation by expert speech therapists confirms that VST consistently generates high-quality, evidence-based therapy recommendations. These findings demonstrate the system's potential to augment clinical workflows, reduce clinician burden, and improve therapeutic outcomes for individuals with speech impairments. An interactive user interface for the proposed system is available online at: this https URL , facilitating real-time stuttering assessment and personalized therapy planning. 

---
# A Knowledge-Driven LLM-Based Decision-Support System for Explainable Defect Analysis and Mitigation Guidance in Laser Powder Bed Fusion 

**Authors**: Basit Mahmud Shahriar, Md Habibor Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2605.01100)  

**Abstract**: This work presents a knowledge-driven decision-support system that integrates structured defect knowledge with LLM-based reasoning to provide explainable defect diagnosis and mitigation guidance in manufacturing, using LPBF as a representative, safety-critical case study. The proposed ontology-integrated LLM-based decision support system for LPBF defect analysis and mitigation guidance is built on a knowledge base containing 27 known LPBF defect types organized into hierarchical categories and causal relationships. The developed system supports fuzzy natural language queries for systematic knowledge retrieval, literature-supported explanation of defects, and guidance on defect causes and mitigation strategies derived from encoded process knowledge. Furthermore, a multimodal image-assessment module based on foundation models enables descriptor-guided interpretation of representative microscopic defect images through semantic alignment scoring. The proposed framework was evaluated through qualitative comparisons with general-purpose vision-language models, an ablation study, and an inter-rater reliability analysis. Evaluation on the literature-derived dataset showed that the fully integrated configuration outperformed the other three evaluated system configurations, achieving a macro-average F1 score of 0.808. Additionally, inter-rater reliability analysis using Cohen's kappa indicated substantial agreement between the model outputs and the literature-derived reference labels. These findings suggest that ontology-guided knowledge representation can improve the consistency, interpretability, and practical usefulness of LLM-assisted LPBF defect analysis. 

---
# Algebraic Semantics of Governed Execution: Monoidal Categories, Effect Algebras, and Coterminous Boundaries 

**Authors**: Alan L. McCann  

**Link**: [PDF](https://arxiv.org/pdf/2605.01032)  

**Abstract**: We present an algebraic semantics for governed execution in which governance is axiomatized, compositional, and coterminous with expressibility. The framework, mechanized in 32 Rocq modules (~12,000 lines, 454 theorems, 0 admitted), is built on interaction trees and parameterized coinduction. A three-axiom GovernanceAlgebra record (safety, transparency, properness) induces a symmetric monoidal category with verified pentagon, triangle, and hexagon coherence, where every tensor composition preserves governance. An algebraic effect system constrains the handler algebra so that only governance-preserving handlers can be constructed in the safe fragment; programs in the empty capability set provably emit only observability directives. Capability-indexed composition bundles programs with machine-checked capability bounds, and a dual guarantee theorem establishes that within_caps and gov_safe hold simultaneously under all composition operators. The capstone result is the coterminous boundary: within our formal model, every program expressible via the four primitive morphism constructors is governed under interpretation, and every governed program is the image of such a program. Turing completeness is preserved inside governance; unmediated I/O is excluded from the governed fragment. Governance denial is modeled as safe coinductive divergence. The governance algebra is parametric: any system instantiating the three axioms inherits all derived properties, including convergence, compositional closure, and goal preservation. Extracted OCaml runs as a NIF in the BEAM runtime, with property-based testing (70,000+ random inputs, zero disagreements) confirming behavioral equivalence between the specification and the runtime interpreter. 

---
# Effect-Transparent Governance for AI Workflow Architectures: Semantic Preservation, Expressive Minimality, and Decidability Boundaries 

**Authors**: Alan L. McCann  

**Link**: [PDF](https://arxiv.org/pdf/2605.01030)  

**Abstract**: We present a machine-checked formalization of structurally governed AI workflow architectures and prove that effect-level governance can be imposed without reducing internal computational expressivity. Using Interaction Trees in Rocq 8.19, we define a governance operator G that mediates all effectful directives, including memory access, external calls, and oracle (LLM) queries. Our development compiles with 0 admitted lemmas and consists of 36 modules, ~12,000 lines of Rocq, and 454 theorems. We establishseven properties: (P1) governed Turing completeness, (P2) governed oracle expressivity, (P3) a decidability boundary in which governance predicates are total and closed under Boolean composition while semantic program properties remain non-trivial and undecidable by governance, (P4) goal preservation for permitted executions, (P5) expressive minimality of primitive capabilities (compute, memory, reasoning, external call, observability), (P6) subsumption asymmetry showing structural governance strictly subsumes content-level filtering, and (P7) semantic transparency: on all executions where governance permits, the governed interpretation is observationally equivalent (modulo governance-only events) to the ungoverned interpretation. Together, these results show that governance and computational expressivity are orthogonal dimensions: governance constrains the effect boundary of programs while remaining semantically transparent to internal computation. 

---
# Accelerating battery research with an AI interface between FINALES and Kadi4Mat 

**Authors**: Giovanna Tosato, Leon Merker, Monika Vogler, Michael Selzer, Arnd Koeppe  

**Link**: [PDF](https://arxiv.org/pdf/2605.00909)  

**Abstract**: The time-consuming formation process critically impacts the longevity of sodium-ion coin cells and End Of Life (EOL) performance. This study aims to optimize formation protocols for duration efficiency, targeting high-performance outcomes while minimizing the number of experiments to reduce resource consumption and accelerate discovery. Specifically, we consider two potentially competing objectives: minimizing formation time and maximizing EOL performance. Beyond this application focus, we also present a methodological contribution: a framework designed to enable interoperability between the FINALES and Kadi RDM ecosystems, which we employ to tackle our optimization problem. In this setup, the FINALES framework orchestrates experiment planning and execution on the POLiS MAP, while an active-learning agent implemented within Kadi4Mat guides experiment selection, using multi-objective batched Bayesian optimization to efficiently explore the parameter space. This interoperability enhancement enables coordinated, distributed collaboration across automated systems and human-operated workflows, bridging multiple research centers. Using this approach, we iteratively explore the trade-off between formation time and EOL performance and identify candidate solutions approximating the Pareto front. The resulting workflow demonstrates the capability of interoperable infrastructures to facilitate data-driven optimization in battery research, and establishes a transferable framework applicable to diverse materials science and engineering optimization tasks. 

---
# ClinicBot: A Guideline-Grounded Clinical Chatbot with Prioritized Evidence RAG and Verifiable Citations 

**Authors**: Navapat Nananukul, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2605.00846)  

**Abstract**: Clinical diagnosis requires answers that are accurate, verifiable, and explicitly grounded in official guidelines. While large language models excel at natural language processing, their tendency to hallucinate undermines their utility in high-stakes medical contexts where precision is essential. Existing retrieval-augmented generation (RAG) systems treat all evidence equally, producing noisy context and generic answers misaligned with clinical practice. We present ClinicBot, an AI system that translates guideline recommendations into trustworthy clinical support through three key advances: (1) structured extraction of clinical guidelines into semantic units (recommendations, tables, definitions, narrative) with explicit provenance, (2) evidence prioritization that ranks content by clinical significance and guideline structure rather than textual similarity, and (3) a web-based interface that presents concise, actionable answers with verifiable evidence. We will demonstrate ClinicBot using diabetes questions from real patients and an additional diabetes risk assessment tool that is faithful to the American Diabetes Association (ADA) Standards of Care in Diabetes (2025). The demonstration will illustrate how semantic knowledge extraction and hierarchical evidence ranking can reliably operate in a multi-agent setting to process complex clinical guidelines at scale. 

---
# Understanding Emergent Misalignment via Feature Superposition Geometry 

**Authors**: Gouki Minegishi, Hiroki Furuta, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2605.00842)  

**Abstract**: Emergent misalignment, where fine-tuning on narrow, non-harmful tasks induces harmful behaviors, poses a key challenge for AI safety in LLMs. Despite growing empirical evidence, its underlying mechanism remains unclear. To uncover the reason behind this phenomenon, we propose a geometric account based on the geometry of feature superposition. Because features are encoded in overlapping representations, fine-tuning that amplifies a target feature also unintentionally strengthens nearby harmful features in accordance with their similarity. We give a simple gradient-level derivation of this effect and empirically test it in multiple LLMs (Gemma-2 2B/9B/27B, LLaMA-3.1 8B, GPT-OSS 20B). Using sparse autoencoders (SAEs), we identify features tied to misalignment-inducing data and to harmful behaviors, and show that they are geometrically closer to each other than features derived from non-inducing data. This trend generalizes across domains (e.g., health, career, legal advice). Finally, we show that a geometry-aware approach, filtering training samples closest to toxic features, reduces misalignment by 34.5%, substantially outperforming random removal and achieving comparable or slightly lower misalignment than LLM-as-a-judge-based filtering. Our study links emergent misalignment to feature superposition, providing a basis for understanding and mitigating this phenomenon. 

---
# AI Agents for Sustainable SMEs: A Green ESG Assessment Framework 

**Authors**: Viet Trinh, Tan Nguyen, Minh-Huyen Phan, Quan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00841)  

**Abstract**: This study presents a novel, AI-driven framework for assessing Environmental, Social, and Governance (ESG) performance in European small and medium-sized enterprises (SMEs). An initial phase established expert-validated ESG baseline scores from a subset of the Flash Eurobarometer FL549 survey data. In the second phase, a scalable AI agent system, built on the n8n automation platform, applied these baselines to perform automated ESG classification and generate contextual recommendations using large language models (LLMs). The results demonstrate the AI system's high consistency with human-derived outputs, thereby supporting more effective monitoring and intervention strategies aligned with the European Green Deal. 

---
# 2026 Roadmap on Artificial Intelligence and Machine Learning for Smart Manufacturing 

**Authors**: Jay Lee, Hanqi Su, Marco Macchi, Adalberto Polenghi, Wei Wu, Zhiheng Zhao, George Q.Huang, Kiva Allgood, Devendra Jain, Benedikt Gieger, Vibhor Pandhare, Soumyabrata Bhattacharjee, Ram Mohril, Lingbao Kong, Qiyuan Wang, Xinlan Tang, Sungjong Kim, Chan Hee Park, Byeng D. Youn, Guo Dong Goh, Xi Huang, Wai Yee Yeong, Yung C Shin, He Zhang, Zitong Wang, Fei Tao, Jagjit Singh Srai, Satyandra K. Gupta, Byung Gun Joung, Albin John, John W. Sutherland, Sang Won Lee, Olga Fink, Vinay Sharma, Faez Ahmed, Wei Chen, Mark Fuge, Arild Waaler, Martin G. Skjæveland, Dimitris Kyritsis, Wei Chen, VispiNevile Karkaria, Yi-Ping Chen, Ying-Kuan Tsai, Joseph Cohen, Xun Huan, Jing Lin, Liangwei Zhang, Gregory W. Vogl, Aaron W. Cornelius, Xiaodong Jia, Dai-Yan Ji, Takanobu Minami, Ruoxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00839)  

**Abstract**: The evolution of artificial intelligence (AI) and machine learning (ML) is reshaping smart manufacturing by providing new capabilities for efficiency, adaptability, and autonomy across industrial value chains. However, the deployment of AI and ML in industrial settings still faces critical challenges, including the complexity of industrial big data, effective data management, integration with heterogeneous sensing and control systems, and the demand for trustworthy, explainable, and reliable operation in high-stakes industrial environments. In this roadmap, we present a comprehensive perspective on the foundations, applications, and emerging directions of AI and ML in smart manufacturing. It is structured in three parts. The first highlights the foundations and trends that frame the evolution of AI in smart manufacturing. The second focuses on key topics where AI is already enabling advances, including industrial big data analytics, advanced sensing and perception, autonomous systems, additive and laser-based manufacturing, digital twins, robotics, supply chain and logistics optimization, and sustainable manufacturing. The third section explores non-traditional ML approaches that are opening new frontiers, such as physics-informed AI, generative AI, semantic AI, advanced digital twins, explainable AI, RAMS, data-centric metrology, LLMs, and foundation models for highly connected and complex manufacturing systems. By identifying both opportunities and remaining barriers across these areas, this roadmap outlines the advances needed in methods, integration strategies, and industrial adoption. We hope this roadmap will serve as a guide for researchers, engineers, and practitioners to accelerate innovation, align academic and industrial priorities, and ensure that AI-driven smart manufacturing delivers reliable, sustainable, and scalable impact for the future of manufacturing ecosystems. 

---
# SpecKV: Adaptive Speculative Decoding with Compression-Aware Gamma Selection 

**Authors**: Shikhar Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2605.02888)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference by using a small draft model to propose candidate tokens that a larger target model verifies. A critical hyperparameter in this process is the speculation length~$\gamma$, which determines how many tokens the draft model proposes per step. Nearly all existing systems use a fixed~$\gamma$ (typically~4), yet empirical evidence suggests that the optimal value varies across task types and, crucially, depends on the compression level applied to the target model. In this paper, we present \textbf{SpecKV}, a lightweight adaptive controller that selects~$\gamma$ per speculation step using signals extracted from the draft model itself. We profile speculative decoding across 4~task categories, 4~speculation lengths, and 3~compression levels (FP16, INT8, NF4), collecting 5,112 step-level records with per-step acceptance rates, draft entropy, and draft confidence. We demonstrate that the optimal~$\gamma$ shifts across compression regimes and that draft model confidence and entropy are strong predictors of acceptance rate (correlation~$\approx 0.56$). SpecKV uses a small MLP trained on these signals to maximize expected tokens per speculation step, achieving a 56.0\% improvement over the fixed-$\gamma$=4 baseline with only 0.34\,ms overhead per decision ($<$0.5\% of step time). The improvement is statistically significant ($p < 0.001$, paired bootstrap test). We release all profiling data, trained models, and notebooks as open-source artifacts. 

---
# Enhancing RL Generalizability in Robotics through SHAP Analysis of Algorithms and Hyperparameters 

**Authors**: Lingxiao Kong, Cong Yang, Oya Deniz Beyan, Zeyd Boukhers  

**Link**: [PDF](https://arxiv.org/pdf/2605.02867)  

**Abstract**: Despite significant advances in Reinforcement Learning (RL), model performance remains highly sensitive to algorithm and hyperparameter configurations, while generalization gaps across environments complicate real-world deployment. Although prior work has studied RL generalization, the relative contribution of specific configurations to the generalization gap has not been quantitatively decomposed and systematically leveraged for configuration selection. To address this limitation, we propose an explainable framework that evaluates RL performance across robotic environments using SHapley Additive exPlanations (SHAP) to quantify configuration impacts. We establish a theoretical foundation connecting Shapley values to generalizability, empirically analyze configuration impact patterns, and introduce SHAP-guided configuration selection to enhance generalization. Our results reveal distinct patterns across algorithms and hyperparameters, with consistent configuration impacts across diverse tasks and environments. By applying these insights to configuration selection, we achieve improved RL generalizability and provide actionable guidance for practitioners. 

---
# From Sensors to Insight: Rapid, Edge-to-Core Application Development for Sensor-Driven Applications 

**Authors**: Komal Thareja, Anirban Mandal, Ewa Deelman  

**Link**: [PDF](https://arxiv.org/pdf/2605.02859)  

**Abstract**: Scientists increasingly rely on sensor-based data, yet transforming raw streams into insights across the edge-to-cloud continuum remains difficult. Provisioning heterogeneous infrastructure and managing execution on emerging platforms like Data Processing Units typically requires cross-domain expertise, creating significant barriers to rapid prototyping.
This paper introduces an experience-driven methodology for the rapid development of sensor-driven applications. By combining pattern-based workflow engineering with AI-assisted development-implemented via Pegasus on the FABRIC testbed - we utilize an existing Orcasound hydrophone workflow as a reusable template. We introduce a pattern-based engineering methodology to generate and refine workflows for air quality, earthquake, and soil moisture monitoring. Furthermore, we show how these abstract structures are extended to edge resources through modular configuration and placement. Our evaluation focuses on user productivity and practical lessons rather than peak performance. Through these case studies, we illustrate how AI-assisted, pattern-based development lowers the entry barrier for non-experts and enables iterative exploration of sensor-driven applications across distributed infrastructures. 

---
# (POSTER) From Sensors to Insight: Rapid, Edge-to-Core Application Development for Sensor-Driven Applications 

**Authors**: Komal Thareja, Anirban Mandal, Ewa Deelman  

**Link**: [PDF](https://arxiv.org/pdf/2605.02844)  

**Abstract**: Scientists increasingly rely on sensor-based data; however transforming raw streams into insights across the edge-to-cloud continuum remains difficult due to the breadth of expertise required to coordinate the necessary data and computation flow. This paper introduces a pattern-based, AI-assisted methodology for rapid development of sensor-driven applications. Using Pegasus workflows executing on the FABRIC testbed, we demonstrate a 5-step development loop that shifts workflow construction and deployment from code-first to intent-first design. Starting from an existing Orcasound hydrophone workflow as a reusable template, we generate and refine workflows for air quality, earthquake, and soil moisture monitoring applications. We further show how these workflows extend to edge resources-including BlueField-3 DPUs and Raspberry Pis-through configuration and placement rather than workflow redesign. Our evaluation, from the perspective of a novice Pegasus user, shows that AI-assisted pattern reuse compresses multi-stage workflow development to 1-1.5 days per workflow while preserving the rigor and portability of workflow-based execution. 

---
# A second-order method on the Stiefel manifold via Newton$\unicode{x2013}$Schulz 

**Authors**: Xinhui Xiong, Bin Gao, P.-A. Absil  

**Link**: [PDF](https://arxiv.org/pdf/2605.02838)  

**Abstract**: Retraction-free approaches offer attractive low-cost alternatives to Riemannian methods on the Stiefel manifold, but they are often first-order, which may limit the efficiency under high-accuracy requirements. To this end, we propose a second-order method landing on the Stiefel manifold without invoking retractions, which is proved to enjoy local quadratic (or superlinear for its inexact variant) convergence. The update consists of the sum of (i) a component tangent to the level set of the constraint-defining function that aims to reduce the objective and (ii) a component normal to the same level set that reduces the infeasibility. Specifically, we construct the normal component via Newton$\unicode{x2013}$Schulz, a fixed-point iteration for orthogonalization. Moreover, we establish a geometric connection between the Newton$\unicode{x2013}$Schulz iteration and Stiefel manifolds, in which Newton$\unicode{x2013}$Schulz moves along the normal space. For the tangent component, we formulate a modified Newton equation that incorporates Newton$\unicode{x2013}$Schulz. Numerical experiments on the orthogonal Procrustes problem, principal component analysis, and real-data independent component analysis illustrate that the proposed method performs better than the existing methods. 

---
# IConFace: Identity-Structure Asymmetric Conditioning for Unified Reference-Aware Face Restoration 

**Authors**: Axi Niu, Jinyang Zhang, Senyan Qing  

**Link**: [PDF](https://arxiv.org/pdf/2605.02814)  

**Abstract**: Blind face restoration is highly ill-posed under severe degradation, where identity-critical details may be missing from the degraded input. Same-identity references reduce this ambiguity, but mismatched pose, expression, illumination, age, makeup, or local facial states can lead to overuse of reference appearance. We propose \textbf{IConFace}, a unified reference-aware and no-reference framework with identity--structure asymmetric conditioning. References are distilled into a norm-weighted global AdaFace identity anchor for image-only modulation, while the degraded image is reinforced as the spatial structure anchor through low-rank residuals and block-wise degraded cross-attention with two-route memory. The resulting single checkpoint exploits references when available and falls back to no-reference restoration when absent, improving identity consistency, fine-detail recovery, and degraded-only restoration quality in a unified model. 

---
# Static Analysis of Recursive SHACL 

**Authors**: Anouk Oudshoorn, Magdalena Ortiz, Mantas Simkus  

**Link**: [PDF](https://arxiv.org/pdf/2605.02787)  

**Abstract**: SHACL (Shapes Constraint Language) expresses constraints on RDF data by means of so-called shapes. Its central service is validation: verifying whether a data graph complies with a SHACL document. But so far, there are no static analysis services to compare documents. In this paper, we study the following problem: decide whether all graphs that validate one SHACL document also validate another. Unlike previous works that have considered the implication of shape expressions only, we consider documents comprising (recursive) shape definitions and targets. We show that implication (a.k.a. containment) is undecidable under the supported and the stable model semantics, even for the fragment that uses the description logic ALCIO for shape expressions. Under the well-founded semantics, in surprising contrast, it is decidable in single exponential time. Our key technical contribution is a translation of SHACL under the well-founded semantics into the full hybrid mu-calculus, revealing a novel link between well-founded models and a fixed point modal logic, and a worst-case optimal automata-based decision procedure. 

---
# A decoupled diffusion planner that adapts to changing cost limits by using cost-conditioned generation for safety and reward gradients for performance 

**Authors**: Rufeng Chen, Zhaofan Zhang, Zhejiang Yang, Hechang Chen, Sihong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.02777)  

**Abstract**: Offline safe reinforcement learning often requires policies to adapt at deployment time to safety budgets that vary across episodes or change within a single episode. While diffusion-based planners enable flexible trajectory generation, existing guidance schemes often treat reward improvement and constraint satisfaction as competing gradient objectives, which can lead to unreliable safety compliance under cost limits. We reinterpret adaptive safe trajectory generation as sampling from a constrained trajectory distribution, where the budget restricts the trajectory region, and reward shapes preferences within that region. This perspective motivates Safe Decoupled Guidance Diffusion (SDGD), which conditions classifier-free guidance on the cost limit to bias sampling toward trajectories satisfying the specified limit, while using reward-gradient guidance to refine trajectories for higher return. Because direct reward guidance can increase return while also steering samples toward trajectories with higher cumulative cost, we introduce Feasible Trajectory Relabeling (FTR) to reshape reward targets and discourage such directions. We further provide a first-order sampling-time analysis showing that FTR suppresses reward-induced cost drift under a prefix-restorative alignment condition. Extensive evaluations on the DSRL benchmark show that SDGD achieves the strongest safety compliance among baselines, satisfying the constraint on 94.7% of tasks (36/38), while obtaining the highest reward among safe methods on 21 tasks. 

---
# TOC-SR: Task-Optimal Compact diffusion for Image Super Resolution 

**Authors**: Sowmya Vajrala, Akshay Bankar, Manjunath Arveti, Shreyas Pandith, Sravanth Kodavanti, Subhajit Sanyal, Amit Unde, Srinivas Soumitri Miriyala  

**Link**: [PDF](https://arxiv.org/pdf/2605.02767)  

**Abstract**: Diffusion models have recently demonstrated strong performance for image restoration tasks, including super-resolution. However, their large model size and iterative sampling procedures make them computationally expensive for practical deployment. In this work, we present TOC-SR, a framework for building efficient one-step super-resolution models by first discovering a compact diffusion backbone. Starting from a sixteen-channel latent diffusion model, we construct parameter-efficient surrogate blocks using feature-wise generative distillation and perform architecture discovery using epsilon-constrained Bayesian Optimization to minimize model complexity while preserving generative fidelity. The resulting compact diffusion backbone achieves a 6.6x reduction in parameters and a 2.8x reduction in GMACs compared to the expanded diffusion model. We then adapt this backbone for super-resolution and distill the diffusion process into a single-step generator. Experiments demonstrate that the proposed approach enables efficient super-resolution while maintaining strong reconstruction quality. 

---
# Virtual Scanning for NSCLC Histology: Investigating the Discriminatory Power of Synthetic PET 

**Authors**: Fatih Aksu, Laura Ciuffetti, Francesco Di Feola, Filippo Ruffini, Giulia Romoli, Fabrizia Gelardi, Arturo Chiti, Valerio Guarrasi, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2605.02746)  

**Abstract**: Accurate histological differentiation between adenocarcinoma (ADC) and squamous cell carcinoma (SCC) is critical for personalized treatment in non-small cell lung cancer (NSCLC). While [$^{18}$F]FDG PET/CT is a standard tool for the clinical evaluation of lung cancer, its utility is often limited by high costs and radiation exposure. In this paper, we investigate the feasibility of "virtual scanning" as a feature-enhancement strategy by evaluating whether synthetic PET data can provide complementary feature representations to supplement anatomical CT scans in histological subtype classification.
We propose a framework that leverages a 3D Pix2Pix Generative Adversarial Network (GAN), pretrained on the FDG-PET/CT Lesions dataset, to synthesize pseudo-PET volumes from anatomical CT scans. These synthetic volumes are integrated with structural CT data within the MINT framework, a multi-stage intermediate fusion architecture.
Our experiments, conducted on a multi-center dataset of 714 subjects, demonstrate that the inclusion of synthetic metabolic features significantly improves classification performance over a CT-only baseline. The multimodal approach achieved a statistically significant increase in the Area Under the Curve (AUC) from 0.489 to 0.591 and improved the Geometric Mean (GMean) from 0.305 to 0.524. These results suggest that synthetic PET scans provide discriminatory metabolic cues that enable deep learning models to exploit complementary cross-modal information, offering a potential feature-enhancement strategy for clinical scenarios where physical PET scans are unavailable. 

---
# Bolek: A Multimodal Language Model for Molecular Reasoning 

**Authors**: Frederic Grabowski, Jacek Szczerbiński, Maciej Jaśkowski, Kalina Jasińska-Kobus, Paweł Dąbrowski-Tumański, Tomasz Jetka, Bartosz Topolski  

**Link**: [PDF](https://arxiv.org/pdf/2605.02745)  

**Abstract**: Molecular property models increasingly support high-stakes drug-discovery decisions, but their outputs are often difficult to audit: classical predictors return scores without rationale, while language models can produce fluent explanations weakly grounded in the input molecule.
We introduce Bolek, a compact multimodal language model that grounds natural-language reasoning in molecular structure by injecting a Morgan fingerprint embedding into an instruction-tuned text decoder. Bolek is fine-tuned on molecular alignment tasks, including molecule description, RDKit descriptor prediction, and substructure detection, and on downstream reasoning over 15 TDC binary classification tasks using synthetic chains-of-thought anchored in concrete molecular features.
Across these tasks, Bolek outperforms its Qwen3-4B-Instruct base on all endpoints in yes/no mode and on 13 of 15 in chain-of-thought mode, raising mean ROC/PR AUC from 0.55 to 0.76. It also outperforms TxGemma-9B-Chat on 13 of 15 binary classification tasks despite being less than half its size. Bolek's explanations are more grounded than those of the baseline LLMs: it cites numerical descriptors 10-100x more often per chain-of-thought, and the cited values agree strongly with RDKit for key descriptors such as TPSA, MolLogP, and MolWt (Spearman rho = 0.87-0.91). Generalisation extends beyond the training panel: on 15 unseen TDC classification endpoints, Bolek matches TxGemma on five, and it produces non-trivial rank correlations on three held-out regression endpoints despite never seeing downstream regression during training.
These results suggest that targeted modality injection and reasoning supervision tied to verifiable molecular features can yield compact, auditable molecular reasoning models. 

---
# AI-Generated Smells: An Analysis of Code and Architecture in LLM and Agent-Driven Development 

**Authors**: Yuecai Zhu, Nikolaos Tsantalis, Peter C. Rigby  

**Link**: [PDF](https://arxiv.org/pdf/2605.02741)  

**Abstract**: The promise of Large Language Models in automated software engineering is often measured by functional correctness, overlooking the critical issue of long term maintainability. This paper presents a systematic audit of technical debt in AI-generated software, revealing that AI does not eliminate flaws but rather introduces a distinct machine signature of defects. Our multi-scale analysis, spanning single-file algorithmic tasks and complex, agent generated systems, identifies a fundamental Reasoning-Complexity Trade-off: as models become more capable, they generate increasingly bloated and coupled code. This architectural decay is so pronounced that we establish a Volume-Quality Inverse Law, where code volume is a near perfect predictor of structural degradation. Crucially, we demonstrate that neither functional correctness nor detailed prompting mitigates this decay. These findings challenge the current paradigm of prompt-driven generation, reframing the central problem of AI-based software engineering from one of code generation to one of architectural complexity management. We conclude that future progress depends on equipping agents with explicit architectural foresight to ensure the software they build is not just functional, but also maintainable. 

---
# Perceptual Flow Network for Visually Grounded Reasoning 

**Authors**: Yangfu Li, Yuning Gong, Hongjian Zhan, Teng Li, Yuanhuiyi Lyu, Tianyi Chen, Qi Liu, Ziyuan Huang, Zhihang Zhong, Dandan Zheng, Yue Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02730)  

**Abstract**: Despite the success of Large-Vision Language Models (LVLMs), general optimization objectives (e.g., standard MLE) fail to constrain visual trajectories, leading to language bias and hallucination. To mitigate this, current methods introduce geometric priors from visual experts as additional supervision. However, we observe that such supervision is typically suboptimal: it is biased toward geometric precision and offers limited reasoning utility. To bridge this gap, we propose Perceptual Flow Network (PFlowNet), which eschews rigid alignment with the expert priors and achieves interpretable yet more effective visual reasoning. Specifically, PFlowNet decouples perception from reasoning to establish a self-conditioned generation process. Based on this, it integrates multi-dimensional rewards with vicinal geometric shaping via variational reinforcement learning, thereby facilitating reasoning-oriented perceptual behaviors while preserving visual reliability. PFlowNet delivers a provable performance guarantee and competitive empirical results, particularly setting new SOTA records on V* Bench (90.6%) and MME-RealWorld-lite (67.0%). 

---
# OphMAE: Bridging Volumetric and Planar Imaging with a Foundation Model for Adaptive Ophthalmological Diagnosis 

**Authors**: Tienyu Chang, Zhen Chen, Renjie Liang, Jinyu Ding, Jie Xu, Sunu Mathew, Amir Reza Hajrasouliha, Andrew J. Saykin, Ruogu Fang, Yu Huang, Jiang Bian, Qingyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02714)  

**Abstract**: The advent of foundation models has heralded a new era in medical artificial intelligence (AI), enabling the extraction of generalizable representations from large-scale unlabeled datasets. However, current ophthalmic AI paradigms are predominantly constrained to single-modality inference, thereby creating a dissonance with clinical practice where diagnosis relies on the synthesis of complementary imaging modalities. Furthermore, the deployment of high-performance AI in resource-limited settings is frequently impeded by the unavailability of advanced three-dimensional imaging hardware. Here, we present the Ophthalmic multimodal Masked Autoencoder (OphMAE), a multi-imaging foundation model engineered to synergize the volumetric depth of 3D Optical Coherence Tomography (OCT) with the planar context of 2D en face OCT. By implementing a novel cross-modal fusion architecture and a unique adaptive inference mechanism, OphMAE was pre-trained on a massive dataset with of 183,875 paired OCT images derived from 32,765 patients. In a rigorous benchmark encompassing 17 diverse diagnostic tasks with 48,340 paired OCT images from 8,191 patients, the model demonstrated state-of-the-art performance, achieving an Area Under the Curve (AUC) of 96.9% for Age-related Macular Degeneration (AMD) and 97.2% for Diabetic Macular Edema (DME), consistently surpassing existing single-modal and multimodal foundation models. Crucially, OphMAE exhibits robust engineering adaptability: it maintains high diagnostic accuracy, such as 93.7\% AUC for AMD, even when restricted to single-modality 2D inputs, and demonstrates exceptional data efficiency by retaining 95.7% AUC with as few as 500 labeled samples. This work establishes a scalable and adaptable framework for ophthalmic AI, ensuring robust performance across different tasks. 

---
# mdok-style at SemEval-2026 Task 10: Finetuning LLMs for Conspiracy Detection 

**Authors**: Dominik Macko  

**Link**: [PDF](https://arxiv.org/pdf/2605.02712)  

**Abstract**: SemEval-2026 Task 10 is focused on conspiracy detection. Specifically, the goal is to detect whether a Reddit comment expresses a conspiracy belief. Our submitted mdok-style system utilizes data augmentation and self-training (to cope with a rather small amount of training data) to finetune the Qwen3-32B model for a binary text-classification task. The submitted system is very competitive, ranking in the 85th percentile (8th out of 52 submissions). The results shown that our approach, which originated in machine-generated text detection, can be used for conspiracy detection as well. 

---
# SAIL: Structure-Aware Interpretable Learning for Anatomy-Aligned Post-hoc Explanations in OCT 

**Authors**: Tienyu Chang, Tianhao Li, Ruogu Fang, Jiang Bian, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02707)  

**Abstract**: Optical coherence tomography (OCT), a commonly used retinal imaging modality, plays a central role in retinal disease diagnosis by providing high-resolution visualization of retinal layers. While deep learning (DL) has achieved expert-level accuracy in OCT-based retinal disease detection, its "black box" nature poses challenges for clinical adoption, where explainability is essential for clinical trust and regulatory approval. Existing post-hoc explainable AI (XAI) methods often struggle to delineate fine-grained lesion structures, respect anatomical boundaries, or suppress noise, limiting the trustworthiness of their explanations. To bridge these gaps, we propose a Structure-Aware Interpretable Learning (SAIL) framework that integrates retinal anatomical priors at the representation level and couples them with semantic features via a fusion design. Without modifying standard post-hoc explainability methods, this representation yields sharper and more anatomically aligned attribution maps. Comprehensive experiments on diverse OCT datasets demonstrate that our structure-aware method consistently enhances interpretability, producing clinically meaningful and anatomy-aware explanations. Ablation studies further show that strong interpretability requires both structural priors and semantic features, and that properly fusing the two is critical to achieve the best explanation quality. Together, these results highlight structure-aware representations as a key step toward reliable explainability in OCT. 

---
# ProPACT: A Proactive AI-Driven Adaptive Collaborative Tutor for Pair Programming 

**Authors**: Anahita Golrang, Kshitij Sharma, olga viberg  

**Link**: [PDF](https://arxiv.org/pdf/2605.02703)  

**Abstract**: Effective pair programming depends on coordination of attention, cognitive effort, and joint regulation over time, yet most adaptive learning systems remain individual-centric and reactive. This paper introduces ProPACT, a proactive AI-driven adaptive collaborative tutor that treats collaboration itself as the object of instruction. ProPACT constructs a multimodal dyadic learner model based on Joint Visual Attention (JVA), Joint Mental Effort (JME), and individual mental effort, and employs an XGBoost-based forecasting model to predict emerging suboptimal collaboration states up to 30 seconds in advance. These predictions drive a hierarchical adaptive policy that delivers minimally intrusive scaffolds while fading support during productive collaboration. A within-subject study with 26 pair-programming dyads shows that proactive feedback significantly improves debugging success, task efficiency, feedback uptake, and post-intervention gains in JVA and JME, demonstrating the potential of forecast-driven dyadic adaptivity for real-time collaborative learning regulation. 

---
# Learning Equivariant Neural-Augmented Object Dynamics From Few Interactions 

**Authors**: Sergio Orozco, Tushar Kusnur, Brandon May, George Konidaris, Laura Herlant  

**Link**: [PDF](https://arxiv.org/pdf/2605.02699)  

**Abstract**: Learning data-efficient object dynamics models for robotic manipulation remains challenging, especially for deformable objects. A popular approach is to model objects as sets of 3D particles and learn their motion using graph neural networks. In practice, this is not enough to maintain physical feasibility over long horizons and may require large amounts of interaction data to learn. We introduce PIEGraph, a novel approach to combining analytical physics and data-driven models to capture object dynamics for both rigid and deformable bodies using limited real-world interaction data. PIEGraph consists of two components: (1) a \textbf{P}hysically \textbf{I}nformed particle-based analytical model (implemented as a spring--mass system) to enforce physically feasible motion, and (2) an \textbf{E}quivariant \textbf{Graph} Neural Network with a novel action representation that exploits symmetries in particle interactions to guide the analytical model. We evaluate PIEGraph in simulation and on robot hardware for reorientation and repositioning tasks with ropes, cloth, stuffed animals and rigid objects. We show that our method enables accurate dynamics prediction and reliable downstream robotic manipulation planning, which outperforms state of the art baselines. 

---
# mdok-style at SemEval-2026 Task 9: Finetuning LLMs for Multilingual Polarization Detection 

**Authors**: Dominik Macko, Alok Debnath, Jakub Simko  

**Link**: [PDF](https://arxiv.org/pdf/2605.02695)  

**Abstract**: SemEval-2026 Task 9 is focused on multilingual polarization detection. Specifically, it covers the identification of multilingual, multicultural and multievent polarization along three axes (in subtasks), namely detection, type, and manifestation. Online polarization presents a concern, because it is often followed by hate speech, offensive discourse, and social fragmentation. Therefore, its detection before it escalates is crucial for a safer and more inclusive online space. We have coped with this SemEval task by finetuning mid-size LLMs for the sequence-classification task using the QLoRA parameter-efficient finetuning technique. The training data augmented the multilingual (22 languages) training sets by anonymized, lower-cased, upper-cased, and homoglyphied counterparts, making the detection more robust. 

---
# Caliper-in-the-Loop: Black-Box Optimization for Hyperledger Fabric Performance Tuning 

**Authors**: Yash Madhwal, Arseny Bolotnikov, Mark Prikhno, Irina Lebedeva, Ivan Laishevskiy, Vladimir Gorgadze, Artem Barger, Yury Yanovich  

**Link**: [PDF](https://arxiv.org/pdf/2605.02690)  

**Abstract**: Hyperledger Fabric performance depends on many interacting configuration parameters, making manual tuning difficult. We study automated throughput tuning by treating benchmarking as a noisy black-box optimization problem and applying Bayesian optimization (BO) with dimensionality reduction (DR). We implement an end-to-end Caliper-in-the-loop pipeline that deploys candidate configurations, benchmarks them, and updates the optimizer from observed throughput. The search space, derived from Fabric configuration files, has 317 dimensions. In a cloud testbed, we evaluate 16 BO+DR variants and a random-search baseline. The best method, DYCORS-PCA, achieves a 12% TPS improvement relative to the first evaluated configuration, while MPI-REMBO achieves 9%. These results suggest that BO with DR is a practical approach for high-dimensional Hyperledger Fabric tuning, while also highlighting the role of measurement noise in interpreting gains. 

---
# The Design and Composition of Structural Causal Decision Processes 

**Authors**: Sebastian Benthall, Alan Lujan  

**Link**: [PDF](https://arxiv.org/pdf/2605.02681)  

**Abstract**: We present two new classes of causal models of decision-making agents. Our approach is motivated by the needs of modeling the economics of computing systems. These systems are composed of subsystems and can exhibit endogenous limits on cognitive resources and value discounting. Structural Causal Decision Models (SCDMs) expand on Structural Causal Influence Models. Like SCIMs, they explicitly represent the causal relationships between model variables and the payoffs of agent decisions. Additionally, agent decisions can be constrained by their causal antecedents, and SCDMs can have open root variables for which no probability distribution or structural equation is given. We show that SCDMs have a well-defined and computationally useful property of composability. Building on SCDMs, we then define a Structural Causal Decision Process (SCDP) as a recurring SCDM with a discount variable. SCDPs benefit from the useful composition properties of SCDMs. Moreover, SCDPs are strictly more expressive than POMDPs because they do not assume rational belief formation. Indeed, an SCDP can endogenously model the memory-formation process, and is thus useful for modeling resource rational agents in dynamic settings. SCDPs are also capable of modeling variable discounting, a tool used widely in social scientific modeling. We pose that SCDPs are a useful framework for policy simulation for the digital economy, mechanism design for information systems, and digital twin modeling of cyberinfrastructure. 

---
# Fuzzy Fingerprinting Encoder Pre-trained Language Models for Emotion Recognition in Conversations: Human Assessment and Validity Study 

**Authors**: Patrícia Pereira, Helena Moniz, Joao Paulo Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2605.02665)  

**Abstract**: In Emotion Recognition in Conversations (ERC), model decisions should align with nuanced human perception and ideally provide insights on the classification process. Standard encoder pre-trained language models (PLMs) are the state-of-the-art at these tasks but offer little insight into why a certain prediction is made. This is especially problematic in imbalanced datasets, where most utterances are labeled as neutral, making these models frequently misclassify minority emotions as the majority neutral class. To tackle this issue, we introduced a novel, interpretable approach to ERC by combining PLMs with Fuzzy Fingerprints (FFPs). FFP provide class-specific prototypes that reflect the characteristic class activation patterns in the PLM's latent space. They are derived by ranking and fuzzifying the activations of the pooled conversational context-dependent embeddings across training instances for each emotion. At inference time, each input utterance is similarly fuzzy fingerprinted and matched to the emotion prototypes using a fuzzy similarity function based on the aggregation of the intersection of the fuzzy sets that define each FFP. Experimental results show that FFP integration reduces overclassification into the neutral class and human evaluation further supports the adequacy of FFP predictions. Our proposed method thus bridges the gap between deep neural inference and human perception, performing at state-of-the-art level while simultaneously offering valuable insights into the classification procedure. 

---
# ViewSAM: Learning View-aware Cross-modal Semantics for Weakly Supervised Cross-view Referring Multi-Object Tracking 

**Authors**: Jiawei Ge, Xintian Zhang, Jiuxin Cao, Bo Liu, Fabian Deuser, Chang Liu, Gong Wenkang, Siyou Li, Juexi Shao, Wenqing Wu, Chen Feng, Ioannis Patras  

**Link**: [PDF](https://arxiv.org/pdf/2605.02638)  

**Abstract**: Cross-view Referring Multi-Object Tracking (CRMOT) aims to track multiple objects specified by natural language across multiple camera views, with globally consistent identities. Despite recent progress, existing methods rely heavily on costly frame-level spatial annotations and cross-view identity supervision. To reduce such reliance, we explore CRMOT under weak supervision by leveraging the capabilities of foundation models. However, our empirical study shows that directly applying foundation models such as SAM2 and SAM3, even with task-specific modifications, fails to accurately understand referring expressions and maintain consistent identities across views. Yet, they remain effective at producing reliable object tracklets that can serve as pseudo supervision. We therefore repurpose foundation models as pseudo-label generators and propose a two-stage framework for weakly supervised CRMOT, using only object category labels as coarse-grained supervision. In the first stage, we design an Affinity-guided Cross-view Re-prompting strategy to refine and associate SAM3-generated tracklets across cameras, producing reliable cross-view pseudo labels for subsequent training. In the second stage, we introduce ViewSAM, a CRMOT model built upon SAM2 that explicitly models view-aware cross-modal semantics. By formulating view-induced variations as learnable conditions, ViewSAM bridges the gap between view-variant visual observations and view-invariant textual expressions, enabling robust cross-view referring tracking with only approximately 10% additional parameters. Extensive experiments demonstrate that ViewSAM achieves SOTA performance under weak supervision and remains competitive with fully supervised methods. 

---
# Validation of an AI-based end-to-end model for prostate pathology using long-term archived routine samples 

**Authors**: Xiaoyi Ji, Renata Zelic, Oskar Aspegren, Nita Mulliqi, Michelangelo Fiorentino, Francesca Giunchi, Luca Molinaro, Sol Erika Boman, Lorenzo Richiardi, Andreas Pettersson, Per Henrik Vincent, Martin Eklund, Olof Akre, Kimmo Kartasalo  

**Link**: [PDF](https://arxiv.org/pdf/2605.02614)  

**Abstract**: Artificial intelligence (AI) is becoming a clinical tool for prostate pathology, but generalization across variations in sample preparation and preservation over prolonged time periods remains poorly understood. We evaluated GleasonAI, an end-to-end attention-based multiple instance learning model, on an independent validation cohort comprising 10,366 biopsy cores from 1,028 patients across 14 Swedish regions, using archival diagnostic specimens from the ProMort cohorts collected between 1998-2015. The model achieved an overall quadratic-weighted kappa of 0.86 for core-level ISUP grading, comparable to several experienced pathologists and consistent across geographic regions. Notably, performance remained stable across the 17-year collection period, demonstrating robustness to time-related variation in archival material, a property not consistently observed with foundation model-based approaches, with exploratory analysis demonstrating a significant prognostic gradient across AI-assigned grade groups for prostate cancer-specific mortality. These findings support the generalizability of the AI grading model and demonstrate the potential of pathology archives as a large-scale resource for AI development, validation, and retrospective prognostic research. 

---
# Dependency Parsing Across the Resource Spectrum: Evaluating Architectures on High and Low-Resource Languages 

**Authors**: Kevin Guan, Happy Buzaaba, Christiane Fellbaum  

**Link**: [PDF](https://arxiv.org/pdf/2605.02608)  

**Abstract**: Transformer-based models achieve state-of-the-art dependency parsing for high-resource languages, yet their advantage over simpler architectures in low-resource settings remains poorly understood. We evaluate four parsers -- the Biaffine LSTM, Stack-Pointer Network, AfroXLMR-large, and RemBERT -- across ten typologically diverse languages, with a focus on low-resource African languages. We find that the Biaffine LSTM consistently outperforms transformer models in low-resource regimes, with transformers recovering their advantage as training data increases. The crossover falls within a resource range typical of treebanks for under-resourced languages. Morphological complexity (measured via MATTR) emerges as a significant secondary predictor of transformers' relative disadvantage after controlling for corpus size. These results indicate that the Biaffine LSTM may be better suited for syntactic tool development in low-resource regimes until sufficient annotated data is available to leverage the representational capacity of pre-trained transformers. 

---
# CoRAL: Contact-Rich Adaptive LLM-based Control for Robotic Manipulation 

**Authors**: Berk Çiçek, Mert K. Er, Özgür S. Öğüz  

**Link**: [PDF](https://arxiv.org/pdf/2605.02600)  

**Abstract**: While Large Language Models (LLMs) and Vision-Language Models (VLMs) demonstrate remarkable capabilities in high-level reasoning and semantic understanding, applying them directly to contact-rich manipulation remains a challenge due to their lack of explicit physical grounding and inability to perform adaptive control. To bridge this gap, we propose CoRAL (Contact-Rich Adaptive LLM-based control), a modular framework that enables zero-shot planning by decoupling high-level reasoning from low-level control. Unlike black-box policies, CoRAL uses LLMs not as direct controllers, but as cost designers that synthesize context-aware objective functions for a sampling-based motion planner (MPPI). To address the ambiguity of physical parameters in visual data, we introduce a neuro-symbolic adaptation loop: a VLM provides semantic priors for environmental dynamics, such as mass and friction estimates, which are then explicitly refined in real time via online system identification, while the LLM iteratively modulates the cost-function structure to correct strategic errors based on interaction feedback. Furthermore, a retrieval-based memory unit allows the system to reuse successful strategies across recurrent tasks. This hierarchical architecture ensures real-time control stability by decoupling high-level semantic reasoning from reactive execution, effectively bridging the gap between slow LLM inference and dynamic contact requirements. We validate CoRAL on both simulation and real-world hardware across challenging and novel tasks, such as flipping objects against walls by leveraging extrinsic contacts. Experiments demonstrate that CoRAL outperforms state-of-the-art VLA and foundation-model-based planner baselines by boosting success rates over 50% on average in unseen contact-rich scenarios, effectively handling sim-to-real gaps through its adaptive physical understanding. 

---
# Beyond State Machines: Executing Network Procedures with Agentic Tool-Calling Sequences 

**Authors**: Purna Sai Garigipati, Onur Ayan, Kishor Chandra Joshi, Xueli An  

**Link**: [PDF](https://arxiv.org/pdf/2605.02584)  

**Abstract**: Agentic AI will be an essential enabling technology for designing future mobile communication systems, which could provide flexible and customized services, automate complex network operations, and drive autonomous decision-making across the network. This work studies how Large Language Model (LLM)-based network AI agents can be utilized to execute network procedures expressed as sequences of tool invocations. We investigate four approaches, which differ in how the agent obtains the procedure and in how execution is distributed between the agent and the underlying tools. We evaluated the latency and execution correctness across these approaches using a User Equipment (UE) IP allocation procedure as a case study. Furthermore, we conduct a stress test to examine how many sequential procedural steps an LLM agent can reliably execute before failure. Our results show that approaches relying on iterative agent-side reasoning incur higher latency and are more prone to execution errors, while approaches where the procedure is encapsulated within a single tool, which internally orchestrates the required steps by invoking other tools, reduce latency by limiting repeated reasoning. The stress-test results further show that the model with advanced tool-calling capability maintains reliable execution over longer procedures than the other evaluated models; however, all models exhibit reliability degradation as procedure length increases, revealing clear execution limits in multi-step tool-based workflows. To systematically analyze failures in procedure execution, we introduce a procedure-specific error taxonomy that categorizes deviations in multi-step procedural execution. 

---
# Hyp2Former: Hierarchy-Aware Hyperbolic Embeddings for Open-Set Panoptic Segmentation 

**Authors**: Yao Lu, Rohit Mohan, Florian Drews, Yakov Miron, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2605.02580)  

**Abstract**: Recognizing unknown objects is crucial for safety-critical applications such as autonomous driving and robotics. Open-Set Panoptic Segmentation (OPS) aims to segment known thing and stuff classes while identifying valid unknown objects as separate instances. Prior OPS approaches largely treat known categories as a flat label set, ignoring the semantic hierarchy that provides valuable structural priors for distinguishing unknown objects from in-distribution classes. In this work, we propose Hyp2Former, an end-to-end framework for OPS that does not require explicit modeling of unknowns during training, and instead learns hierarchical semantic similarities continuously in hyperbolic space. By explicitly encoding hierarchical relationships among known categories, the model learns a structured embedding space that captures multiple levels of semantic abstraction. As a result, unknown objects that cannot be confidently classified as known categories still remain in close proximity to higher-level concepts (e.g., an unknown animal remains closer to "animal" or "object" than to unrelated concepts such as "electronics" or "stuff") and can therefore be reliably detected, even if their fine-grained category was not represented during training. Empirical evaluations across multiple public datasets such as MS COCO, Cityscapes, and Lost&Found demonstrate that Hyp2Former outperforms existing methods on OPS, achieving the best balance between unknown object discovery and in-distribution robustness. 

---
# Recurrent Deep Reinforcement Learning for Chemotherapy Control under Partial Observability 

**Authors**: Firas Mohamed Elamine Kiram, Imane Youkana, Rachida Saouli, Gian Antonio Susto, Laid Kahloul  

**Link**: [PDF](https://arxiv.org/pdf/2605.02552)  

**Abstract**: Chemotherapy dose optimization can be formulated as a dynamic treatment regime, requiring sequential decisions under uncertainty that must balance tumor suppression against toxicity. However, most reinforcement learning approaches assume full observability of the patient state, a condition rarely met in clinical practice. We investigate whether memory-augmented policies can improve chemotherapy control under partial observability. To this end, we employ a recurrent TD3-based approach with separate LSTM actor-critic networks and evaluate it on the AhnChemoEnv benchmark from DTR-Bench, considering both off-policy and on-policy recurrent architectures against feed-forward TD3 and Soft Actor-Critic. Pharmacokinetic and pharmacodynamic variability are held fixed to isolate hidden-state uncertainty and observation noise and to avoid confounding effects from inter-patient variability. Across ten random seeds, recurrence yields modest benefit under full observability but substantially stronger and more stable performance under partial observability, with more consistent tumor suppression and improved normal-cell preservation. These findings indicate that memory-based policies are particularly beneficial when clinically relevant state information is incomplete or noisy. 

---
# Orchestrating Spatial Semantics via a Zone-Graph Paradigm for Intricate Indoor Scene Generation 

**Authors**: Meisheng Zhang, Shizhao Sun, Yang Zhao, Ziyuan Liu, Zhijun Gao, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2605.02537)  

**Abstract**: Autonomous 3D indoor scene synthesis breaks down in non-convex rooms with tightly coupled spatial constraints. Data-driven generators lack topological priors for long-horizon planning, while iterative agents fragment semantics and become geometrically brittle. We present ZoneMaestro, a unified framework that shifts the paradigm from object-centric synthesis to Zone-Graph Orchestration. By internalizing a novel zone-based logic, ZoneMaestro translates high-level semantic intent into functional zones and topological constraints, enabling robust adaptation to diverse architectural forms. To support this, we construct Zone-Scene-10K, a large-scale dataset enriched with explicit Zone-Graph annotations. We further introduce an Alternating Alignment Strategy that cycles between reasoning internalization and Zone-Aware Group Relative Policy Optimization (Z-GRPO), effectively reconciling the tension between semantic richness and geometric validity without relying on external physics engines. To rigorously evaluate spatial intelligence beyond convex primitives, we formally define the task of Intricate Spatial Orchestration and release SCALE, a stress-test benchmark for irregular indoor scenarios with complex, dense spatial relations. Extensive experiments demonstrate that ZoneMaestro resolves the density-safety dichotomy, significantly outperforming state-of-the-art baselines in both structural coherence and intent adherence. 

---
# Set-Based Training of Neural Barrier Certificates for Safety Verification of Dynamical Systems 

**Authors**: Miriam Kranzlmüller, Lukas Koller, Tobias Ladner, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2605.02526)  

**Abstract**: Barrier certificates are scalar functions over the state space of dynamical systems that separate all unsafe states from all reachable states. The existence of a barrier certificate formally verifies the safety of the dynamical system. Recent approaches synthesize barrier certificates by iteratively training a neural network. In each iteration, the candidate is formally verified - if successful, the barrier certificate is found. Instead, we propose a set-based training approach that tightly integrates verification into training via a set-based loss function that soundly encodes all barrier certificate properties. A loss of zero formally proves the validity of the barrier certificate, collapsing the iterative training and verification into a single training procedure. Our experiments demonstrate that our set-based training approach scales well with the system dimension and naturally handles complex nonlinear dynamics. 

---
# A Semantic Autonomy Framework for VLM-Integrated Indoor Mobile Robots: Hybrid Deterministic Reasoning and Cross-Robot Adaptive Memory 

**Authors**: Bogdan Felician Abaza, Andrei-Alexandru Staicu, Cristian Vasile Doicin  

**Link**: [PDF](https://arxiv.org/pdf/2605.02525)  

**Abstract**: Autonomous indoor mobile robots can navigate reliably to metric coordinates using established frameworks such as ROS 2 Navigation 2, yet they lack the ability to interpret natural language instructions that express intent rather than positions. Vision-Language Models offer the semantic reasoning required to bridge this gap, but their inference latency (2-9 seconds per decision on consumer hardware) and session-by-session amnesia limit practical deployment. This paper presents the Semantic Autonomy Stack, a six-layer reference framework for semantically autonomous indoor navigation, and validates a complete instance featuring hybrid deterministic-VLM reasoning and cross-robot adaptive memory on physical robots with off-the-shelf edge hardware. A seven-step parametric resolver handles 88% of instructions in under 0.1 milliseconds without invoking a language model, camera, or GPU; only genuinely ambiguous instructions escalate to VLM reasoning. A five-category semantic memory framework with explicit scope taxonomy (global environment knowledge, per-operator preferences, per-robot capabilities) enables cross-session learning and cross-robot knowledge transfer: preferences learned through VLM interactions on one robot are promoted to deterministic resolution and transferred to a second robot via a shared compiled digest, achieving a measured latency reduction of 103,000-fold. Experimental validation on two custom-built differential-drive robots across 82 scenario-level decisions and three sessions demonstrates 100% semantic transfer accuracy (33/33, 95% CI [0.894, 1.000]), 100% semantic resolution accuracy, and concurrent multi-robot operation feasibility - all on Raspberry Pi 5 platforms with no onboard GPU, requiring zero training data. 

---
# Benchmarking Retrieval Strategies for Biomedical Retrieval-Augmented Generation: A Controlled Empirical Study 

**Authors**: Devi Prasad Bal, Subhashree Puhan  

**Link**: [PDF](https://arxiv.org/pdf/2605.02520)  

**Abstract**: Retrieval-Augmented Generation (RAG) offers a well-established path to grounding large language model (LLM) outputs in external knowledge, yet the question of which retrieval strategy works best in a high-stakes domain such as biomedicine has not received the controlled, multi-metric treatment it deserves. This paper presents a systematic empirical comparison of five retrieval strategies -- Dense Vector Search, Hybrid BM25 + Dense retrieval, Cross-Encoder Reranking, Multi-Query Expansion, and Maximal Marginal Relevance (MMR) -- within a biomedical question-answering RAG pipeline. All strategies share a fixed generation model (GPT-4o-mini), a common vector store (ChromaDB), and OpenAI's text-embedding-3-small embeddings, ensuring that observed differences are attributable to retrieval alone. Evaluation is conducted on 250 question-answer pairs drawn from a preprocessed subset of the BioASQ benchmark (rag-mini-bioasq) using four DeepEval metrics: contextual precision, contextual recall, faithfulness, and answer relevancy, each reported with 95% confidence intervals. A no-context ablation is included as a lower bound. Cross-Encoder Reranking achieves the best composite score (0.827) and highest contextual precision (0.852), confirming that query-document interaction yields measurable retrieval gains. Multi-Query Expansion, despite its recall-oriented design, produces the weakest contextual precision (0.671), suggesting naive query diversification introduces retrieval noise. MMR sacrifices answer relevancy for diversity, while the Dense baseline (composite 0.822) falls within 0.005 points of the top strategy. All RAG conditions dramatically outperform the no-context ablation on answer relevancy (0.658-0.701 vs. 0.287), confirming the practical value of retrieval. The full pipeline, hyperparameters, and evaluation code are publicly available. 

---
# A Novel Preprocessing-Driven Approach to Remaining Useful Life (RUL) Prediction Using Temporal Convolutional Networks (TCN) 

**Authors**: Florent Imbert, Tosin Adewumi, Hui Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.02507)  

**Abstract**: Accurate prediction of Remaining Useful Life (RUL) in aero-engines is vital for predictive maintenance, improved operational reliability, and reduced lifecycle costs. While deep learning approaches have demonstrated strong potential in this area, most existing methods focus primarily on model architecture design and treat input features uniformly, often neglecting the influence of data preprocessing. In this work, we propose a novel preprocessing pipeline that enhances RUL prediction by improving data quality and temporal representation before model training. Our approach leverages complete temporal sequences and generates RUL estimates at each timestep, enabling the model to capture fine-grained degradation dynamics and deliver continuous prognostic insights throughout the engine's operational life. To validate the effectiveness of the proposed pipeline, we conduct experiments on the NASA C-MAPSS dataset. Comparative evaluations against a suite of state-of-the-art neural models including CNN, RNN, LSTM, DCNN, TCN, BiGRU-TSAM, AGCNN, and ATCN, demonstrate that our approach consistently achieves superior accuracy and robustness in aero-engine RUL prediction. These results highlight the critical role of preprocessing in maximizing the effectiveness of neural prognostic models. 

---
# Pretraining on Sleep Data Improves non-Sleep Biosignal Tasks 

**Authors**: William Lehn-Schiøler, Magnus Ruud Kjær, Phillip Hempel, Magnus Guldberg Pedersen, Rahul Thapa, Bryan He, Nicolai Spicher, Andreas Brink-Kjaer, Lars Kai Hansen, Emmanuel Mignot  

**Link**: [PDF](https://arxiv.org/pdf/2605.02500)  

**Abstract**: Sleep foundation models have recently demonstrated strong performance on in-domain polysomnography tasks, including sleep staging, apnea detection, and disease risk prediction. In this work, we investigate whether sleep biosignals can serve as an effective pretraining distribution for learning representations that transfer beyond sleep to adjacent domains. Following sleep foundation models, we perform sleep-only multimodal contrastive pretraining (with a leave-one-out objective) and evaluate transfer to non-sleep EEG and ECG, two well-benchmarked biosignal modalities with heterogeneous datasets and clinically meaningful downstream tasks. Across eight downstream tasks spanning multiple EEG and ECG datasets, sleep pretraining consistently improves performance relative to training from scratch. Moreover, on several tasks, we achieve performance competitive with or surpassing prior specialized state-of-the-art and foundation models. 

---
# Efficient Preference Poisoning Attack on Offline RLHF 

**Authors**: Chenye Yang, Weiyu Xu, Lifeng Lai  

**Link**: [PDF](https://arxiv.org/pdf/2605.02495)  

**Abstract**: Offline Reinforcement Learning from Human Feedback (RLHF) pipelines such as Direct Preference Optimization (DPO) train on a pre-collected preference dataset, which makes them vulnerable to preference poisoning attack. We study label flip attacks against log-linear DPO. We first illustrate that flipping one preference label induces a parameter-independent shift in the DPO gradient. Using this key property, we can then convert the targeted poisoning problem into a structured binary sparse approximation problem. To solve this problem, we develop two attack methods: Binary-Aware Lattice Attack (BAL-A) and Binary Matching Pursuit Attack (BMP-A). BAL-A embeds the binary flip selection problem into a binary-aware lattice and applies Lenstra-Lenstra-Lovász reduction and Babai's nearest plane algorithm; we provide sufficient conditions that enforce binary coefficients and recover the minimum-flip objective. BMP-A adapts binary matching pursuit to our non-normalized gradient dictionary and yields coherence-based recovery guarantees and robustness (impossibility) certificates for $K$-flip budgets. Experiments on synthetic dictionaries and the Stanford Human Preferences dataset validate the theory and highlight how dictionary geometry governs attack success. 

---
# From Experimental Limits to Physical Insight: A Retrieval-Augmented Multi-Agent Framework for Interpreting Searches Beyond the Standard Model 

**Authors**: Altan Cakir, Ayca Yerlikaya  

**Link**: [PDF](https://arxiv.org/pdf/2605.02491)  

**Abstract**: Modern searches for physics beyond the Standard Model produce rapidly expanding literature containing heterogeneous information, including textual analyses, numerical datasets, and graphical exclusion limits. Integrating these distributed sources remains a time-consuming and manual process for physicists.
We present HEP-CoPilot, a retrieval-augmented multi-agent AI framework for the exploration and interpretation of high-energy physics literature. The system unifies textual information from publications, structured experimental data from HEPData, and reconstructed physics plots within a multimodal retrieval and reasoning architecture. By combining retrieval-augmented language models with coordinated agent workflows, it enables evidence-grounded reasoning over experimental analyses and structured interpretation of collider results.
We evaluate the framework on recent CMS searches for physics beyond the Standard Model. Case studies show that HEP-CoPilot can retrieve relevant measurements, reconstruct exclusion limits directly from HEPData records, and perform cross-paper comparisons of experimental constraints. This enables consistent, physics-aware comparison across analyses without manual data integration.
These results demonstrate that retrieval-augmented AI systems can function as scientific co-pilots for particle physics, facilitating navigation of complex literature, structuring heterogeneous evidence, and accelerating the interpretation pipeline for new physics searches. 

---
# Reference-Sampled Boltzmann Projection for KL-Regularized RLVR: Target-Matched Weighted SFT, Finite One-Shot Gaps, and Policy Mirror Descent 

**Authors**: Yao Shu, Chenxing Wei, Hongbin Lin, Shuang Qiu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.02469)  

**Abstract**: Online reinforcement learning with verifiable rewards (RLVR) turns checkable outcomes into a scalable training signal, but it keeps rollout generation, verifier scoring, and reference-policy evaluations on the optimization path. Static weighted supervised fine-tuning (SFT) on precomputed rollouts seems to remove this bottleneck, yet a weighted likelihood is not specified by rewards alone: its sampler and weights induce the policy being fit. This paper identifies the reference-sampled weighted-SFT objective whose induced policy equals the fixed-reference KL-regularized RLVR optimizer. The optimizer is the standard Boltzmann target policy, obtained by exponentially tilting the reference policy by verifier reward. Matching a weighted-SFT induced policy to this target forces density-ratio weights; in the reference-sampled subclass, this reduces uniquely, up to prompt scaling, to the prompt-normalized Boltzmann weight $\exp(r(x,y)/\beta)/Z(x)$. BOLT, a Boltzmann-Targeted SFT procedure, is the empirical estimator of this projection. The finite one-shot analysis separates the exact stored-support price $\beta\log(1/\pi^*(S_N\mid x))$ from partition estimation, effective-sample-size variance, generalization, optimization, and approximation errors. This decomposition explains why extra SFT epochs cannot repair missing reference-policy coverage and exposes the temperature--coverage--variance frontier. When coverage needs adaptive sampling, refreshed Boltzmann projections become KL policy mirror descent; finite inner solves enter as additive drift from the exact mirror step. Single-run Qwen experiments provide projection evidence for the target-matched weight, one-shot saturation, refreshed-sampler gains, and optimization-time savings, within the stated single-run scope. 

---
# When Stress Becomes Signal: Detecting Antifragility-Compatible Regimes in Multi-Agent LLM Systems 

**Authors**: Jose Manuel de la Chica, Juan Manuel Vera, Jairo Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2605.02463)  

**Abstract**: Multi-agent LLM systems are increasingly used to solve complex tasks through decomposition, debate, specialization, and ensemble reasoning. However, these systems are usually evaluated in terms of robustness: whether performance is preserved under perturbation. This paper studies a different question: whether semantic stress exposes structured variation that could support future antifragile learning. We introduce CAFE, a statistical framework for detecting antifragility-compatible regimes in multi-agent architectures. CAFE models a controlled expected distribution of semantic stressors, reconstructs an architecture-specific observed effective stress distribution from multi-dimensional judge signals, and compares both distributions using a distributional Jensen Gap under a convex stress potential. A positive gap does not imply immediate performance improvement; instead, it indicates a convex-expansive deformation of the observed stress distribution, suggesting that the architecture exposes learnable stress structure. We evaluate CAFE on a banking-risk analysis benchmark with five multi-agent architectures: flat, hierarchical, debate, meta-adaptive, and ensemble. Across all architectures, semantic stress reduces average judged quality by roughly one third. Yet all architectures exhibit positive distributional Jensen Gaps with bootstrap confidence intervals above zero. These results show that immediate quality degradation can coexist with statistically detectable antifragility-compatible stress geometry. CAFE is therefore not an antifragile learner itself, but a measurement layer for identifying when and where antifragility learning may be worth applying. 

---
# LLM-Assisted Repository-Level Generation with Structured Spec-Driven Engineering 

**Authors**: Shuzhao Feng, Boqi Chen, Brett H Meyer, Gunter Mussbacher  

**Link**: [PDF](https://arxiv.org/pdf/2605.02455)  

**Abstract**: State-of-the-art Large Language Models (LLMs) excel in code generation at the function level. However, the output quality significantly declines when scaling to repository-level systems. Current workflows relying only on natural language prompts suffer from inherent ambiguity and a lack of verifiability. To address this, we propose structured spec-driven engineering (SSDE), a paradigm that leverages structured artifacts to guide LLM generation. We argue that structured specifications as LLM inputs make high-quality, repository-level code generation a tangible goal, while at the same time offering superior verifiability, leading to significant potential for improvement. We first investigate the feasibility of this vision through a pilot study generating Model-View-Controller (MVC) business logic for three software systems using five LLMs, and then highlight the potential, challenges, and future roadmap for SSDE. 

---
# Causal Software Engineering: A Vision and Roadmap 

**Authors**: Roberto Pietrantuono, Luca Giamattei, Stefano Russo, Julien Siebert, Neil Walkinshaw  

**Link**: [PDF](https://arxiv.org/pdf/2605.02454)  

**Abstract**: Software engineering increasingly involves making high-stakes decisions under uncertainty, using signals from code, field data, and socio-technical processes. Recent AI-driven support (e.g., anomaly detection, predictive analytics, AIOps, as well as LLM-based agents) has amplified engineers' ability to detect patterns and synthesize content and recommendations, but many critical questions are interventional or counterfactual: What is the expected impact of changing a load-balancing strategy? Would an outage have been avoided under a different release plan? Correlational models answer "what tends to co-occur"; they struggle to answer "what would happen if we act." We propose Causal Software Engineering (CSE) as a future paradigm in which causal models and causal reasoning systematically inform activities across the software lifecycle, augmenting existing practices with explicit assumptions, uncertainty-aware effect estimates, and counterfactual diagnosis. We outline (i) a causal-first workflow view spanning development and operations, (ii) a staged roadmap for tools and organizational adoption, and (iii) an evaluation and benchmark agenda for measuring progress. 

---
# PC-MNet: Dual-Level Congruity Modeling for Multimodal Sarcasm Detection via Polarity-Modulated Attention 

**Authors**: Maoheng Li, Ling Zhou, Xiaohua Huang, Rubing Huang, Wenming Zheng, Guoying Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.02447)  

**Abstract**: Multimodal sarcasm detection, which aims to precisely identify pragmatic incongruities between literal text and nonverbal cues, has gained substantial attention in multimodal understanding. Recent advancements have predominantly relied on na\"ıve similarity-based attention mechanisms and uniform late fusion this http URL, given that functional entanglement restricts traditional late fusions, we incorporate a scalar congruity routing mechanism and a prior-guided contextual graph. This mechanism anchors a generalized incongruity manifold through a two-stage asymmetric optimization driven by inconsistency-aware contrastive learning, selectively fusing only the most discriminative multi-granularity evidence. Extensive experiments on the \texttt{MUStARD} benchmark and its spurious-correlation-mitigated balanced datasets demonstrate that our approach achieves new state-of-the-art performance, surpassing the strongest multimodal baseline by a substantial 3.14\% improvement in Macro-F1. By architecturally isolating atomic, composition, and contextual conflicts. This work provides a robust, decoupled paradigm for modeling subtle pragmatic incongruities in human communication. 

---
# Automatic Reflection Level Classification in Hungarian Student Essays 

**Authors**: Zsolt Csibi, Mónika Sándor, Mónika Serfőző, Kinga Gyöngy, Kristian Fenech  

**Link**: [PDF](https://arxiv.org/pdf/2605.02402)  

**Abstract**: Reflective thinking is a key competency in education, but assessing reflective writing remains a time-consuming and subjective task for education experts. While automated reflective analysis has been explored in several languages, Hungarian language was not researched extensively. In this paper, we present the first comprehensive study on automatic reflection level classification in Hungarian student essays. We used a large, expert-annotated Hungarian dataset consisting of 1,954 reflective essays collected over multiple academic years and labeled on a four-level reflection scale. We investigate two approaches: (1) classical machine learning models using TF-IDF and semantic embedding features, and (2) Hungarian-specific transformer models fine-tuned for document-level reflection classification. To address the strong class imbalance in the dataset, we systematically examine class weighting, oversampling, data augmentation, and alternative loss functions. An extensive ablation study is conducted to analyze the contribution of each modeling and balancing strategy. Our results show that shallow machine learning models with appropriate feature engineering achieve strong overall performance, reaching up to 71% overall score averaged over accuracy, F1-score, and ROC AUC metrics, while transformer-based models achieve slightly lower overall score (68%) averaged over the same metrics, but demonstrate better generalization on minority reflection classes. These findings highlight the continued relevance of classical methods for low-resource settings and the robustness of transformer models for imbalanced classification. The proposed dataset and experimental insights provide a solid foundation for future research on automated reflective analysis in Hungarian and other morphologically rich languages. 

---
# FEAT: Fashion Editing and Try-On from Any Design 

**Authors**: Soye Kwon, Keonyoung Lee, Dahuin Jung, Jaekoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.02393)  

**Abstract**: Fashion design aims to express a designer's creative intent and to depict how garments interact with the human body. Recent methods condition on multimodal inputs to support garment editing and virtual try-on. However, existing methods still (i) confine design to garment-related images, excluding creative design sources such as artwork, abstract imagery, and natural photographs, and (ii) cannot support complete outfits, including accessories. We present FEAT (Fashion Editing And Try-On from Any Design), a method that enables editing and try-on across garments and accessories using diverse design sources. To achieve this, we introduce Disentangled Dual Injection (DDI). It takes both apparel and non-apparel design sources and selectively injects design cues via content and style disentanglement. Furthermore, we propose Orthogonal-Guided Noise Fusion (OGNF), a training-free mechanism that removes residual garments via orthogonal projection and applies region-specific noise strategies to enable virtual try-on for both garments and accessories. Extensive experiments demonstrate that FEAT achieves state-of-the-art performance in design flexibility, prompt consistency, and visual realism. 

---
# Is It Novel and Why? Fine-Grained Patent Novelty Prediction Based on Passage Retrieval 

**Authors**: Valentin Knappich, Anna Hätty, Simon Razniewski, Annemarie Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2605.02392)  

**Abstract**: Novelty assessment is a critical yet complex task in the examination process for patent acceptance, requiring examiners to determine whether an invention is disclosed in a prior art document. The process involves intricate matching between specific features of a patent claim and passages in the prior art. While prior work has approached novelty prediction primarily as a binary classification task at the claim level, we argue that this formulation is susceptible to spurious correlations and lacks the granularity required for practical application. In this work, we introduce FiNE-Patents (Fine-grained Novelty Examination of Patents), a novel dataset comprising 3,658 first patent claims annotated with fine-grained, feature-level prior art references extracted from European Search Opinion (ESOP) documents. We propose shifting the evaluation paradigm from simple binary classification to a joint retrieval and abstract reasoning task at the feature level, requiring models to identify specific passages from a prior art document that disclose individual claim features, and to identify which features of a claim make it novel. We implement and evaluate LLM-based workflows that decompose claims into features, analyze each feature against prior art, and finally derive a claim-level novelty prediction. Our experiments demonstrate that these workflows outperform embedding-based baselines on passage retrieval and novel feature identification. Furthermore, we show that unlike trained classifiers, LLMs are robust against spurious correlations present in the claim-level novelty classification task. We release the dataset and code to foster further research into transparent and granular patent analysis. 

---
# Entanglement is Half the Story: Post-Selection vs. Partial Traces 

**Authors**: Gustav J L Jäger, Krzysztof Bieniasz, Martin B Plenio, Hans-Martin Rieser  

**Link**: [PDF](https://arxiv.org/pdf/2605.02385)  

**Abstract**: While tensor networks have their traditional application in simulating quantum systems, in the recent decade they have gathered interest as machine learning models. We combine the experience from both fields and derive how quantum constraints placed on a tensor network manifest a change in capabilities. To this end, we employ a method of inference of classical tensor networks on a quantum computer to define a hybrid architecture. This hybrid tensor network is a practical unified framework for it's classical and quantum tensor network edge cases. We identify post-selection as the important property on which this interpolation hinges. The amount of post-selection corresponds to the level to which quantum constraints are enforced on the tensor network. On this basis, we propose a new hyperparameter which controls the transition between the hybrid and the quantum tensor network. In the comparison of classical and quantum tensor networks it complements the bond dimension. Quantum machine learning is improved by using the hyperparameter to allocate the practically limited post-selection to the quantum model in a trainable manner. 

---
# Enhancing Multimodal In-Context Learning via Inductive-Deductive Reasoning 

**Authors**: Haoyu Wang, Haonan Wang, Yuyan Chen, Jun Chen, Gang Liu, Qian Wang, Jiahong Yan, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.02378)  

**Abstract**: In-context learning (ICL) allows large models to adapt to tasks using a few examples, yet its extension to vision-language models (VLMs) remains fragile. Our analysis reveals that the fundamental limitation lies in an inductive gap, models often produce correct answers from flawed reasoning, while struggling to extract consistent rules across demonstrations. This gap is further exacerbated by two visual-level obstacles: an overwhelming proportion of redundant visual tokens that obscure textual cues, and a skewed attention distribution that favors the initial image at the expense of subsequent context. To address these issues, we introduce a framework that restructures multimodal ICL as a principled inductive-deductive process. The framework incorporates a similarity-based visual token compression module to filter out redundant patches, a dynamic attention rebalancing mechanism to distribute focus equitably across all images, and a chain-of-thought paradigm that explicitly guides the model to analyze individual examples, derive a generalizable rule, and then apply it to the query. An auxiliary learning pipeline combines supervised fine-tuning with reinforcement learning using verifiable rewards to reinforce faithful citation and noise filtering. Evaluations across eight benchmarks covering visual perception, logical reasoning, STEM problems, and sarcasm detection demonstrate consistent and significant improvements over standard ICL baselines for multiple open-source VLMs, highlighting the potential of equipping models with genuine inductive capabilities in multimodal settings. 

---
# Privacy Preserving Machine Learning Workflow: from Anonymization to Personalized Differential Privacy Budgets in Federated Learning 

**Authors**: Judith Sáinz-Pardo Díaz, Álvaro López García  

**Link**: [PDF](https://arxiv.org/pdf/2605.02372)  

**Abstract**: The growing development of artificial intelligence based solutions, together with privacy legislation, has driven the rise of the so-called privacy preserving machine learning architectures, such as federated learning. While federated learning enables model training on decentralized data preventing their sharing and centralization, it still faces several challenges related to data integrity and privacy. This paper presents a comprehensive privacy preserving federated learning workflow for sensitive tabular data, including anonymization and differential privacy techniques. We also introduce a formal definition for the concept of client drift, together with ways of detecting it to mitigate poisoning attacks. Then, we detail a complete methodology for assigning personalized privacy budgets for global differential privacy to the different clients participating in the network, based on a re-identification risk metric. The proposed methodology is presented and tested on an openly available dataset of medical records. Within the experimental setup we show that the approach based on personalized budgets, compared to the architecture including global differential privacy with fixed privacy budget, achieves a better model performance in terms of two error metrics. 

---
# When Correct Isn't Usable: Improving Structured Output Reliability in Small Language Models 

**Authors**: Cosimo Galeone, Minsu Park, Giuseppe Ettorre, Daniele Ligorio  

**Link**: [PDF](https://arxiv.org/pdf/2605.02363)  

**Abstract**: Deployed language models must produce outputs that are both correct and format-compliant. We study this structured-output reliability gap using two mathematical benchmarks -- GSM8K and MATH -- as a controlled testbed: ground truth is unambiguous and the output contract is strict (JSON with required fields). We evaluate three 7-9B models under five prompting strategies and report output accuracy -- the joint event of mathematical correctness and valid JSON structure -- as the primary metric. A systematic format failure emerges: NAIVE prompting (no system prompt) achieves up to 85% task accuracy on GSM8K but 0% output accuracy across all models and datasets. REFERENCE prompting (a minimal hand-written JSON format prompt) fares little better, yielding 0% output accuracy for two of four models tested. Constrained decoding enforces syntactic validity but incurs 3.6x-8.2x latency overhead and in several settings degrades task performance substantially. To overcome this limitation, we developed AloLab, an iterative system-prompt optimizer (meta-agent: Claude Sonnet 4.5) requiring only black-box API access to the target model; it reaches 84-87% output accuracy on GSM8K and 34-40% on MATH across five independent runs per model, with 29/30 paired McNemar comparisons against the best static prompt significant at p < 0.05, at near-NAIVE inference latency and without model fine-tuning. The same format failure extends to GPT-4o (OpenAI, 2024), a proprietary closed-source model: REFERENCE achieves 0% output accuracy due to systematic markdown-fence wrapping, while AloLab reaches 95.2% [94.8, 95.6]. An ablation replacing the Sonnet 4.5 meta-agent with Claude 3 Haiku reduces mean output accuracy to 61.0% and increases run-to-run standard deviation from <1 pp to 21.8 pp, confirming that meta-agent capability is a primary driver of optimization quality. 

---
# APIOT: Autonomous Vulnerability Management Across Bare-Metal Industrial OT Networks 

**Authors**: Adel ElZemity, Budi Arief, Shujun Li, Calvin Brierley, Yichao Wang, Yuxiang Huang, James Pope, Haoxiang Li, George Oikonomou  

**Link**: [PDF](https://arxiv.org/pdf/2605.02346)  

**Abstract**: Bare-metal operational technology (OT) devices -- especially the microcontrollers running Modbus/TCP and CoAP at the base of industrial control systems -- have remained outside the reach of autonomous security attacks. Prior autonomous pentesting studies target Linux and web systems, whose shells and filesystems are familiar to LLM agents. Bare-metal OT has neither, so agents must reason directly over protocol fields and parser semantics. This requires new action-space designs and runtime controls, and opens new research questions about protocol-level exploit reasoning and its deployment envelope. We present APIOT (Autonomous Purple-teaming for Industrial OT), the first large language model (LLM) framework demonstrating an autonomous attack and remediation of bare-metal OT devices, achieving the full discovery -> exploitation -> patching -> verification cycle without step-by-step human intervention. We implemented and evaluated this framework on Zephyr RTOS firmware across heterogeneous industrial IoT (IIoT) topologies. Through 290 experiment runs spanning five frontier LLMs, three network topologies, two impairment levels, and guided versus unguided conditions, APIOT achieved a mission success rate of 90.0% on the full attack-remediation cycle. We found that the runtime governance layer (which we call an overseer) is a critical engineering variable: without it, agents exhibit systematic degenerate patterns, including repetition loops, missing crash verification, and reconnaissance deadlocks. Together, these findings carry two implications beyond our testbed. Attacker expertise is no longer the binding constraint on bare-metal OT exploitation, and defender threat models must now assume LLM-augmented adversaries capable of executing autonomous discovery-through-remediation cycles against industrial firmware. 

---
# LLM-enabled Social Agents 

**Authors**: Önder Gürcan, Moharram Challenger  

**Link**: [PDF](https://arxiv.org/pdf/2605.02335)  

**Abstract**: Large Language Models (LLMs) have transformed agent-agent and human-agent interaction by enabling software, physical, and simulation agents to communicate and deliberate through natural language. Yet fluent language use does not by itself yield socially intelligible behaviour. Most current systems remain weakly grounded in roles, norms, intentions, and contextual constraints, limiting their capacity for meaningful participation in social environments. This paper develops a conceptual baseline for LLM-enabled social agents by arguing that they should be grounded in role definitions operationalized through persona descriptions. On this basis, we outline research directions for representation, hybrid control, and evaluation. The paper concludes that persona-based role definitions are a necessary foundation for turning language competence into social behaviour. 

---
# When Attention Collapses: Residual Evidence Modeling for Compositional Inference 

**Authors**: Niklas Houba  

**Link**: [PDF](https://arxiv.org/pdf/2605.02323)  

**Abstract**: Compositional inference - the decomposition of observations into an unknown number of latent components - is central to perception and scientific data analysis. Attention-based models perform well when components are approximately separable, as in object-centric vision. Under additive superposition, however - where multiple components contribute to every observation - we identify a structural failure mode we term slot collapse: multiple slots converge to the same dominant component while weaker ones remain unrepresented. We trace this to a general limitation: attention is memoryless with respect to explained evidence. All slots repeatedly operate on the same input without accounting for what has already been explained, so gradients are dominated by the strongest component, inducing shared fixed points across slots. As a result, attention fails to enforce non-redundant allocation under additive superposition. We address this by introducing residual evidence modeling, instantiated via evidence depletion - a minimal modification combining multiplicative depletion with an attention bias. Controlled ablations show that parallel attention, sequential processing alone, and loss-based regularization fail to resolve collapse; evidence depletion, which adds residual state to sequential attention, consistently succeeds. Across synthetic benchmarks and real-world audio mixtures (FUSS), evidence depletion reduces slot collapse by up to an order of magnitude, generalizing beyond synthetic settings. On gravitational-wave source inference for the ESA/NASA LISA mission, under identical architectures, data, and losses, standard attention fails while evidence depletion prevents collapse and enables multi-source posterior estimation. These results show that under additive superposition, residual evidence tracking is the operative ingredient for preventing collapse and enabling compositional inference. 

---
# Rethinking Electro-Optical Vision Foundation Models for Remote Sensing Retrieval: A Controlled Comparison with Generalist VFM 

**Authors**: Hyobin Park, Minseok Seo, Dong-Geol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.02283)  

**Abstract**: Vision foundation models have attracted significant attention for their ability to leverage large-scale unlabeled visual data. This advantage is particularly important in remote sensing, where data acquisition is costly and annotation often requires expert knowledge. Recent electro-optical vision foundation models aim to learn domain-specific representations from remote sensing imagery, but it remains unclear whether they are more effective than strong generalist vision foundation models under retrieval-based evaluation. In this study, we conduct a controlled comparison between representative EO-specific and generalist vision foundation models for remote sensing image retrieval. Using the same datasets, retrieval protocol, and evaluation metric, we evaluate both in-domain performance and cross-scene generalization. Our results show that strong generalist vision foundation models are competitive with, and in some cases outperform, existing EO-specific models. Moreover, EO-specific models often suffer from substantial degradation under cross-scene evaluation, while generalist models show more stable transfer. These findings suggest that EO pretraining alone does not guarantee stronger retrieval-oriented remote sensing representations. We discuss the limitations of current EO-specific pretraining strategies and highlight the need for future EO vision foundation models to better exploit the physical, spatial, spectral, and geographic characteristics of remote sensing imagery. 

---
# HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation 

**Authors**: Fengming Zhang, Wenjie Du, Huan Zhang, Ke Yu, Shen Qu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02278)  

**Abstract**: Time series imputation benefits from leveraging cross-feature correlations, yet existing attention-based methods re-discover feature relationships at each layer, lacking persistent anchors to maintain consistent representations. To address this, we propose HELIX, which assigns each feature a learnable feature identity, a persistent embedding that captures intrinsic semantic properties throughout the network. Unlike graph-based methods that rely on predefined topology and assume homogeneous spatial relationships, HELIX learns arbitrary feature dependencies end-to-end from temporal co-variation, naturally handling datasets where features mix spatial locations with semantic variables. Integrated with hybrid temporal-feature attention, HELIX achieves the state-of-the-art performance, surpassing all 16 baselines on 5 public datasets across 21 experimental settings in our evaluation. Furthermore, our mechanistic analysis reveals that HELIX aligns learned feature identities and dependencies with latent physical and semantic structure progressively across layers, demonstrating that it more effectively translates cross-feature structure into imputation accuracy. 

---
# EdgeLPR: On the Deep Neural Network trade-off between Precision and Performance in LiDAR Place Recognition 

**Authors**: Pierpaolo Serio, Hetian Wang, Zixiang Wei, Vincenzo Infantino, Lorenzo Gentilini, Lorenzo Pollini, Valentina Donzella  

**Link**: [PDF](https://arxiv.org/pdf/2605.02275)  

**Abstract**: Place recognition is essential for long-term autonomous navigation, enabling loop closure and consistent mapping. Although deep learning has improved performance, deploying such models on resource-constrained platforms remains challenging. This work explores efficient LiDAR-based place recognition for EdgeAI by leveraging Bird's Eye View representations to enable lightweight image-based networks. We benchmark representative architectures without aggregation heads using a unified descriptor scheme based on global pooling and linear projection, and evaluate performance under FP32, FP16, and INT8 quantization. Experiments reveal trade-offs between accuracy, robustness, and efficiency: FP16 matches FP32 with lower cost, while INT8 introduces architecture-dependent degradation. Overall, the presented results are a strong basis for future research on 'use-case'-aware quantisation of Neural Networks for Edge deployment. 

---
# Reliability-Oriented Multilingual Orthopedic Diagnosis: A Domain-Adaptive Modeling and a Conceptual Validation Framework 

**Authors**: Danish Ali, Li Xiaojian, Sundas Iqbal, Farrukh Zaidi  

**Link**: [PDF](https://arxiv.org/pdf/2605.02266)  

**Abstract**: Large Language Models (LLMs) are increasingly proposed for clinical decision support including multilingual diagnosis in low-resource settings. However, their reliability, calibration and safety characteristics remain insufficiently understood for structured, high-risk tasks. We present a system-level analysis of multilingual orthopedic diagnosis from free-text clinical notes in English, Hindi and Punjabi. We evaluate three modeling regimes: (i) task-aligned multilingual transformer encoders, (ii) a task-fine-tuned baseline (DistilBERT), and (iii) a domain-adaptive architecture tailored to orthopedic text (IndicBERT-HPA). These models are compared with zero-shot, instruction-tuned LLMs to assess suitability for structured diagnostic classification. Results indicate that while LLMs exhibit strong linguistic fluency, they show unstable calibration and reduced reliability under structured multilingual conditions, particularly in low-resource languages. These findings are specific to zero-shot evaluation and do not imply limitations of fine-tuned models. Domain-adaptive specialization substantially improves cross-lingual discrimination and confidence behavior. IndicBERT-HPA, with language-specific orthopedic adapter heads achieves consistently strong performance across six diagnostic categories and more predictable deployment characteristics than task-only adaptation. Building on these observations, we outline a conceptual deterministic agent-based validation framework for future implementation, formalizing evidence checks, language-sensitive validation and conservative human-in-the-loop gating. Reliable multilingual clinical decision support requires specialized architecture, explicit reliability analysis, and structured validation for safety-critical systems. 

---
# On the Privacy of LLMs: An Ablation Study 

**Authors**: Karima Makhlouf, Lamiaa Basyoni, Syed Khaderi, Gabriel Marquez, Peter Sotomango, Mahmoud Awawdah, Sami Zhioua  

**Link**: [PDF](https://arxiv.org/pdf/2605.02255)  

**Abstract**: Large language models (LLMs) are increasingly deployed in interactive and retrieval-augmented settings, raising significant privacy concerns. While attacks such as Membership Inference (MIA), Attribute Inference (AIA), Data Extraction (DEA), and Backdoor Attacks (BA) have been studied, they are typically analyzed in isolation, leaving a gap in understanding their behavior under common system factors. In this paper, we introduce a unified threat model and notation, reproduce a representative set of privacy attacks, and conduct a structured ablation study to evaluate the impact of key factors such as model architecture, scale, dataset characteristics, and retrieval configuration. Our analysis reveals clear differences across attack types. Membership inference attacks, particularly mask-based variants, exhibit strong and reliable signals, while backdoor attacks achieve consistently high success rates due to their trigger-based nature. In contrast, attribute inference and data extraction attacks remain more challenging, resulting in lower accuracy, yet they pose significant risks as they target sensitive personal information. Overall, these results highlight that privacy risks in LLM systems are highly context-dependent and driven by design choices, emphasizing the need for holistic evaluation and informed deployment practices. 

---
# The Conversations Beneath the Code: Triadic Data for Long-Horizon Software Engineering Agents 

**Authors**: Yelin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.02244)  

**Abstract**: Frontier software engineering agents have saturated short-horizon benchmarks while regressing on the work that constitutes senior engineering: long-horizon, multi-engineer, ambiguous-specification deliverables. This paper takes a position on what training data is needed to close the gap. The substrate for the next generation of SWE agents is neither larger GitHub scrapes nor more solo-agent trajectories nor -- sufficient by itself -- open human-AI dialogue logs. It is triadic data: synchronized capture of the human-human conversations where engineering context is formed, the human-AI sessions where that context is partially consumed, and the multi-week cross-functional work that surrounds both. We argue that the canonical instantiation of triadic data is two complementary products: long-horizon expert trajectories captured under stimulated-recall protocols, and simulated cross-functional companies -- instrumented teams of senior engineers, product managers, designers, and data scientists working through ambiguous deliverables on shared infrastructure. We further specify a four-tier evidence framework through which any such corpus -- triadic or otherwise -- must justify its quality to a fine-tuning researcher: mechanical verification, statistical corpus characterization, probe experiments, and pre-registered blind evaluation. We argue that this data is capturable in 12-18 months with methods already mature in adjacent fields, that it is the empirical key to four open questions in agent training, and that the field's near-term research agenda should include it explicitly. 

---
# MultiSense-Pneumo: A Multimodal Learning Framework for Pneumonia Screening in Resource-Constrained Settings 

**Authors**: Dineth Jayakody, Pasindu Thenahandi, Chameli Dommanige  

**Link**: [PDF](https://arxiv.org/pdf/2605.02207)  

**Abstract**: Pneumonia remains a leading global cause of morbidity and mortality, particularly in low resource settings where access to imaging, laboratory testing, and specialist care is limited. Clinical assessment relies on heterogeneous evidence, including symptoms, respiratory patterns, and chest imaging, making screening inherently multimodal. However, many existing computational approaches remain unimodal and focus primarily on radiographs. In this work, we present MultiSense-Pneumo, a multimodal framework for pneumonia oriented screening and triage support that integrates structured symptom descriptors, cough audio, spoken language, and chest radiographs. The system combines deterministic symptom triage, LightGBM based acoustic classification, domain adversarial radiograph analysis using ResNet 18, transformer based speech recognition, and an interpretable multimodal fusion operator. Each modality is transformed into a normalized risk signal and aggregated into a unified screening estimate, enabling transparent and modular decision support. MultiSense-Pneumo is designed for real world deployment under modest computational constraints and can operate fully offline on standard laptop class hardware, making it suitable for community health workers, rural clinics, and emergency response settings. Experimental results demonstrate robustness of the radiograph pathway under domain shifts, while highlighting limitations in minority class recall for acoustic signals. MultiSense-Pneumo is intended as a research prototype for screening and triage support rather than a clinically validated diagnostic system. 

---
# Trees and Graphs with Non Log-concave Dominating Set Sequence via AI Tools 

**Authors**: Alina Du, Steven Heilman, Greta Panova  

**Link**: [PDF](https://arxiv.org/pdf/2605.02193)  

**Abstract**: We give new examples of graphs and trees with dominating set sequences that are not log-concave. These examples were generated by PatternBoost, a transformer-based reinforcement learning software developed by Charton-Ellenberg-Wagner-Williamson. We also show: for any positive integer $m$, there exists a tree whose dominating set sequence is not log-concave for at least $m$ indices by modifying a similar construction of Bautista-Ramos for the independent set sequence. We show that a large class of caterpillar graphs has log-concave dominating set sequences. A continuous analogue of the sequence is also log-concave for all graphs. 

---
# When Alignment Isn't Enough: Response-Path Attacks on LLM Agents 

**Authors**: Mingyu Luo, Zihan Zhang, Zesen Liu, Yuchong Xie, Zhixiang Zhang, Dung Hiu Hilton Yeung, Wai Ip Lai, Ping Chen, Ming Wen, Dongdong She  

**Link**: [PDF](https://arxiv.org/pdf/2605.02187)  

**Abstract**: Bring-Your-Own-Key (BYOK) agent architectures let users route LLM traffic through third-party relays, creating a critical integrity gap: a malicious relay can modify an aligned LLM response after generation but before agent execution. We formalize this post-alignment tampering threat and show that, without end-to-end integrity, the relay can observe, suppress, or replace downstream messages, making even perfectly aligned LLMs ineffective against such attacks. We instantiate this threat as the Relay Tampering Attack (RTA), which performs multi-round strategic rewriting, minimal security-critical edits, and stealth restoration by resubmitting tampered outputs to the upstream LLM. Across AgentDojo and ASB with six LLMs, RTA achieves up to 99.1% attack success, outperforming prompt-injection baselines with modest overhead. Case studies on OpenClaw and Claude Code demonstrate real-world feasibility, and evaluations of four defenses show that none fully prevent RTA. Finally, we propose a time-based detection defense that mitigates RTA while preserving agent utility. 

---
# RAFNet: Region-Aware Fusion Network for Pansharpening 

**Authors**: Jianing Zhang, Zijian Zhou, Kai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.02184)  

**Abstract**: Pansharpening aims to generate high-resolution multispectral (HRMS) images by fusing low-resolution multispectral (LRMS) and high-resolution panchromatic (PAN) images. Although deep learning has advanced this field, mainstream frequency-based methods relying on standard scaled dot-product attention suffer from quadratic computational complexity and fail to exploit the inherent regional sparsity of remote sensing imagery. Furthermore, existing spatial enhancement strategies typically employ static convolution kernels, which struggle to adapt to the complex frequency and regional variations of PAN and MS images. To address these bottlenecks, we propose a Region-Aware Fusion (RAFNet) Network that synergistically models spatial and frequency information. Specifically, we design a Spatial Adaptive Refinement (SAR) module that leverages the discrete wavelet transform (DWT) for directional frequency separation and K-means clustering for regional partitioning, which enables the dynamic construction of region-specific adaptive convolution kernels, achieving spatially and frequency-adaptive feature enhancement. Moreover, we introduce a Clustered Frequency Aggregation (CFA) module based on a sparse attention mechanism guided by the semantic clusters, which executes a region-aware sparse attention strategy that drastically reduces computational redundancy while ensuring high-quality frequency feature extraction. In addition we integrated these modules into a progressive, multi-level spatial-frequency network architecture to facilitate robust interaction and accurate image reconstruction. Extensive experiments on multiple benchmark datasets demonstrate that the proposed RAFNet significantly outperforms state-of-the-art pansharpening methods in both reduced- and full-resolution assessments. The code is available at this https URL. 

---
# The Causal Description Gap: Information-Theoretic Separations Across Pearl's Hierarchy 

**Authors**: Seyed Morteza Emadi  

**Link**: [PDF](https://arxiv.org/pdf/2605.02177)  

**Abstract**: Pearl's causal hierarchy shows that observational, interventional, and counterfactual queries are qualitatively distinct. We ask a quantitative version of this question: how many additional bits are needed to specify higher-rung causal answers once lower-rung answers are known? We formalize this via query-class description length, the Kolmogorov complexity of the answer oracle induced by an SCM for a class of queries. Our main construction gives binary acyclic SCMs whose observational distribution has constant description length, while the single-variable interventional answer oracle has description length $\Theta(n^2)$. A degree-sensitive upper bound shows that finite-gate-schema SCMs of indegree $d$ have observational-interventional gap at most $O(nd \log(en/d) + n \log n)$, making the quadratic construction order-optimal in the dense regime and a rooted-tree construction order-optimal for bounded indegree. The quadratic separation persists under $\varepsilon$-accurate total-variation descriptions for every fixed $\varepsilon < 1/4$. At the next rung, the full hard-do interventional oracle can still leave a $\Theta(n)$ counterfactual description gap. A general ambiguity-to-bits theorem and Shannon analogue show that these gaps equal the logarithm of residual higher-rung ambiguity up to lower-order terms. 

---
# Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution 

**Authors**: Soyeon Kim, Seongwoo Lim, Kyowoon Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.02167)  

**Abstract**: Feature attribution is central to diagnosing and trusting deep neural networks, and Integrated Gradients (IG) is widely used due to its axiomatic properties. However, IG can yield unreliable explanations when the integration path between a baseline and the input passes through regions with noisy gradients. While Guided Integrated Gradients reduces this sensitivity by adaptively updating low-gradient-magnitude features, input-space guidance still produces intermediate inputs that deviate from the data manifold. To address this limitation, we propose \emph{Manifold-Aligned Guided Integrated Gradients} (MA-GIG), which constructs attribution paths in the latent space of a pre-trained variational autoencoder. By decoding intermediate latent states, MA-GIG biases the path toward the learned generative manifold and reduces exposure to implausible input-space regions. Through qualitative and quantitative evaluations, we demonstrate that MA-GIG produces faithful explanations by aggregating gradients on path features proximal to the input. Consequently, our method reduces off-manifold noise and outperforms prior path-based attribution methods across multiple datasets and classifiers. Our code is available at this https URL. 

---
# DocSync: Agentic Documentation Maintenance via Critic-Guided Reflexion 

**Authors**: Sidhesh Badrinarayan, Adithya Parthasarathy  

**Link**: [PDF](https://arxiv.org/pdf/2605.02163)  

**Abstract**: Software documentation frequently drifts from executable logic as codebases evolve, creating technical debt that degrades maintainability and causes downstream API misuse. While static analysis tools can detect the absence of documentation, they cannot evaluate its semantic consistency. Conversely, standard Large Language Models (LLMs) offer generative flexibility but frequently hallucinate when updating documentation without deep structural awareness of the underlying code. To address this gap, we propose DocSync, an agentic workflow that frames documentation maintenance as a structurally grounded, iterative generation task. DocSync bridges syntactic changes and natural language descriptions by fusing Abstract Syntax Tree (AST) representations and Retrieval-Augmented Generation (RAG) to provide dependency-aware context. Furthermore, to ensure factual consistency, we incorporate a critic-guided refinement loop based on the Reflexion paradigm, allowing the model to self-correct candidate updates against the source code. We empirically evaluate a resource-constrained implementation of DocSync-using a LoRA-adapted small language model - on a proxy code-to-text maintenance task. Our findings demonstrate that this AST-aware agentic approach substantially outperforms standard encoder-decoder baselines across semantic alignment, summary-line faithfulness, and automated judge preferences (e.g., achieving an automated judge score of 3.44/5.0 compared to 1.91 for CodeT5-base). Crucially, the iterative critic loop yields measurable improvements in semantic correctness without requiring scaled-up parameter counts. These results provide strong evidence that coupling structural retrieval with agentic refinement is a highly promising direction for autonomously mitigating documentation debt. 

---
# Combining Trained Models in Reinforcement Learning 

**Authors**: Ujjwal Patil, Javad Ghofrani  

**Link**: [PDF](https://arxiv.org/pdf/2605.02159)  

**Abstract**: Deep reinforcement learning (DRL) has delivered strong results in domains such as Atari and Go, but it still suffers from high sample cost and weak transfer beyond the training setting. A common response is to reuse information from previously trained models through transfer, distillation, ensemble methods, or federated training instead of learning each target task from random initialization. The literature on these mechanisms is fragmented, and published comparisons are hard to interpret because tasks, baselines, and compute budgets differ.
This paper presents a PRISMA-guided systematic review of empirical studies on pretrained knowledge reuse in DRL. Starting from 589 records retrieved from IEEE Xplore, the ACM Digital Library, and citation tracing, we screened 570 unique records and assessed 89 full texts. After applying the final eligibility criteria, 15 empirical studies remained in the main synthesis. We analyzed them qualitatively across three factors: source-target similarity, diversity among reused models, and the fairness of comparisons against from-scratch baselines.
Three patterns recur across the surviving corpus. First, positive results are concentrated in settings where source and target tasks share substantial structure or where the method includes an explicit gating or alignment mechanism. Second, evidence for ensembles and federated aggregation is promising but sparse and mostly limited to narrow settings. Third, compute-matched comparisons are rare, which weakens claims about efficiency gains over stronger single-agent baselines. The paper contributes a narrower and internally consistent review scope, a study-level synthesis of empirical evidence, and a provisional independence spectrum that should be treated as a hypothesis for future benchmarking rather than a validated metric. 

---
# Cross-Polarization Fusion of VV AND VH SAR Observations for Improved Flood Mapping 

**Authors**: Jagrati Talreja, Tewodros Syum Gebre, Leila Hashemi Beni  

**Link**: [PDF](https://arxiv.org/pdf/2605.02153)  

**Abstract**: Synthetic Aperture Radar (SAR) imagery is widely used for flood monitoring due to its all-weather and day-night imaging capability. However, flood mapping using single-polarization SAR data remains challenging in complex environments where surface and volume scattering coexist. In this paper, we investigate the effectiveness of cross-polarization fusion of VV and VH SAR observations for improved flood mapping. A deep learning-based segmentation framework is employed to jointly exploit complementary information from VV and VH polarizations. To ensure a fair evaluation, three configurations are compared under identical training conditions: VV only, VH only, and fused VV-VH input. Performance is assessed using standard flood mapping metrics, including Intersection over Union (IoU) and F1-score, along with qualitative visual analysis. Experimental results demonstrate that VV-VH fusion consistently outperforms single-polarization models, particularly in vegetated and heterogeneous flood regions, leading to more accurate flood boundary delineation. The findings highlight the importance of cross-polarization SAR fusion for enhancing the reliability of SAR-based flood mapping in disaster monitoring applications. 

---
# On the Optimal Sample Complexity of Offline Multi-Armed Bandits with KL Regularization 

**Authors**: Kaixuan Ji, Qiwei Di, Heyang Zhao, Qingyue Zhao, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02141)  

**Abstract**: Kullback-Leibler (KL) regularization is widely used in offline decision-making and offers several benefits, motivating recent work on the sample complexity of offline learning with respect to KL-regularized performance metrics. Nevertheless, the exact sample complexity of KL-regularized offline learning remains largely from fully characterized. In this paper, we study this question in the setting of multi-armed bandits (MABs). We provide a sharp analysis of KL-PCB (Zhao et al., 2026), showing that it achieves a sample complexity of $\tilde{O}(\eta SAC^{\pi^*}/\epsilon)$ under large regularization $\eta = \tilde{O}(\epsilon^{-1})$, and a sample complexity of $\tilde{\Omega}(SAC^{\pi^*}/\epsilon^2)$ under small regularization $\eta = \tilde{\Omega}(\epsilon^{-1})$, where $\eta$ is the regularization parameter, $S$ is the number of contexts, $A$ is the number of arms, $C^{\pi^*}$ policy coverage coefficient at the optimal policy $\pi^*$, $\epsilon$ is the desired sub-optimality, and $\tilde{O}$ and $\tilde{\Omega}$ hide all poly-logarithmic factors. We further provide a pair of sharper sample complexity lower bounds, which matches the upper bounds over the entire range of regularization strengths. Overall, our results provide a nearly complete characterization of offline multi-armed bandits with KL regularization. 

---
# FLoRA: Fusion-Latent for Optical Reconstruction and Flood Area Segmentation via Cross-Modal Multi-Task Distillation Network 

**Authors**: Jagrati Talreja, Tewodros Syum Gebre, Leila Hashemi-Beni  

**Link**: [PDF](https://arxiv.org/pdf/2605.02137)  

**Abstract**: Accurate flood water mapping is critical for disaster management, yet current methods struggle to fully exploit the potential of spaceborne imagery. Optical data offers high interpretability but is limited by environmental conditions, whereas SAR provides reliable all-weather coverage with reduced visual interpretability. FLoRA (Fusion Latent for Optical Reconstruction and Area Segmentation) is a cross-modal multi-task framework that jointly reconstructs high-fidelity optical imagery and segments flood water regions from Sentinel 1 SAR by fusing the complementary strengths of optical and SAR data. During training, a lightweight optical teacher (driven by RGB and NDVI priors) provides pyramidal features that guide SAR representations into a fusion latent space via multiscale windowed cross attention and FiLM conditioning, with gated residuals preventing overcorrection. This design enables multi-task learning across two complementary objectives: (a) SAR-to-optical translation for fine-grained RGB reconstruction and (b) flood water region segmentation for hydrologic interpretation. The dual decoders are optimized using Charbonnier SSIM for structural fidelity, edge FFT magnitude losses for spectral realism, and Dice BCE hydrology-aware edge alignment for precise flood water delineation. A feature distillation constraint further aligns fused SAR features with the optical teacher's manifold. Evaluations on SEN1FLOODS11, DEEPFLOOD, and SEN12MS demonstrate that FLoRA surpasses fusion baselines in PSNR, SSIM, and LPIPS, demonstrating that multi-modal fusion within a teacher-guided latent space yields semantically faithful and physically consistent flood-water intelligence from spaceborne observations. 

---
# Boundary Mass and the Soft-to-Hard Limit in Mixture-of-Experts 

**Authors**: Reza Rastegar  

**Link**: [PDF](https://arxiv.org/pdf/2605.02124)  

**Abstract**: Softmax-routed mixture-of-experts models approach hard routing as the temperature tends to zero, but this limit is singular near routing ties. This paper studies that singularity at the population level for squared-loss MoE regression. The central object is the \emph{boundary mass}, namely the probability that the top two router scores are separated by only a small margin. Under smoothness and transversality assumptions on the router and input law, we prove coarea/tube estimates showing that this mass is linear in the slab width, with leading constant given by a surface integral over the routing interface in the binary case. These estimates yield quantitative soft-to-hard risk bounds and, under compactness and uniform margin control, $\Gamma$-convergence of the soft objectives to the hard-routing objective. The main conclusion is that the zero-temperature limit is controlled by a thin geometric layer around routing interfaces, not by the full input space. We then use this geometric core in two more model-dependent directions. In a teacher--student setting, we prove a conditional landscape-transfer principle showing that, when the profiled hard-routing problem has favorable identifiability and curvature and the relevant derivatives transfer at boundary-layer scale, small-temperature soft routing inherits approximate teacher recovery and strict-saddle behavior away from teacher-equivalent partitions. We also give a reduced two-expert Gaussian calculation that illustrates a local symmetry-breaking mechanism aligned with the teacher separator. 

---
# Context-Aware Wireless Token Communication via Joint Token Masking and Detection 

**Authors**: Junyong Shin, Joohyuk Park, Yongjeong Oh, Jihong Park, Jinho Choi, Yo-Seb Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2605.02123)  

**Abstract**: The increasing use of token-based representations in language-driven applications has motivated wireless token communication, where tokens are treated as fundamental units for transmission. However, conventional communication systems overlook dependencies among tokens and allocate transmission resources uniformly, leading to inefficient use of limited wireless resources under channel impairments. In this paper, we propose a context-aware token communication framework that leverages a masked language model (MLM) as a shared contextual model between the transmitter (Tx) and receiver (Rx). At the Rx, we develop a context-aware token detection method that integrates channel likelihoods with MLM-based contextual priors under a Bayesian formulation, enabling robust token inference over noisy channels. At the Tx, we propose a context-aware token masking strategy that selectively omits tokens that can be reliably inferred at the Rx, allowing the available power budget to be concentrated on more informative tokens. These components are jointly designed through a shared MLM, establishing a unified Tx-Rx framework for efficient token transmission and detection. Simulation results demonstrate that the proposed framework significantly improves reconstruction performance compared to conventional and existing token communication schemes, achieving up to 1.77X and 1.63X performance gains on the Europarl corpus and WikiText-103 datasets, respectively. 

---
# STABLEVAL: Disagreement-Aware and Stable Evaluation of AI Systems 

**Authors**: Akash Bonagiri, Gerard Janno Anderias, Saee Patil, Angelina Lai, Devang Borkar, Gezheng Kang, Ishant Gandhi, Setareh Rafatirad, Houman Homayoun  

**Link**: [PDF](https://arxiv.org/pdf/2605.02122)  

**Abstract**: Human evaluation remains the primary standard for assessing modern AI systems, yet annotator disagreement, bias, and variability make system rankings fragile under standard majority vote aggregation. Majority vote discards annotator reliability and item-level ambiguity, often yielding unstable comparisons across annotator subsets. We introduce STABLEVAL, a disagreement-aware evaluation framework that models latent item correctness and annotator-specific confusion patterns to produce posterior expected item credit and calibrated agent-level scores. Unlike label-denoising approaches such as Dawid-Skene, STABLEVAL is explicitly designed for stable and uncertainty-aware system evaluation rather than hard label recovery. We formalize ranking stability as a first-class evaluation objective and analyze how aggregation methods preserve or distort underlying annotator behavior. Across controlled synthetic experiments and multiple real-world human-annotated benchmarks, majority vote exhibits increasing score error and ranking instability under annotator heterogeneity and adversarial noise, while STABLEVAL yields more stable and statistically grounded system rankings. These results demonstrate that modeling disagreement is essential for robust and reproducible AI evaluation. 

---
# GETA-3DGS: Automatic Joint Structured Pruning and Quantization for 3D Gaussian Splatting 

**Authors**: Baobing Zhang, Wanxin Sui  

**Link**: [PDF](https://arxiv.org/pdf/2605.02086)  

**Abstract**: 3D Gaussian splatting (3DGS) is a state-of-the-art representation for real-time photorealistic novel-view synthesis, yet a single high-fidelity scene typically occupies hundreds of megabytes to several gigabytes, exceeding the budgets of mobile, immersive, and volumetric video platforms. Existing 3DGS compression methods (e.g., HAC++, FlexGaussian, LP-3DGS) treat pruning, quantization, and entropy coding as separate stages and rely on hand-tuned heuristics (opacity thresholds, fixed bit-widths, SH truncation), limiting cross-scene generalization and preventing users from specifying a target rate or quality budget. We propose GETA-3DGS, to our knowledge the first end-to-end automatic joint structured pruning and quantization framework for 3DGS. Building on GETA for joint pruning-quantization of deep networks, we contribute: (i) a 3DGS-aware quantization-aware dependency graph (QADG) treating each Gaussian primitive as a group with five attribute sub-nodes and degree-aware SH sub-nodes; (ii) a render-aware saliency fusing transmittance-weighted contribution, screen-space gradient, and pixel coverage into a Gaussian-level importance score; and (iii) a heterogeneous per-attribute mixed-precision scheme co-optimized with structural sparsity under a projected partial saliency-guided (PPSG) descent guarantee. On Mip-NeRF 360, Tanks and Temples, and Deep Blending, GETA-3DGS operates directly on raw Gaussian primitives rather than a post-hoc anchor representation, delivering ~5x storage reduction over Vanilla 3DGS with no per-scene thresholds. Bit-width policy is the dominant rate-distortion lever: a uniform 6-bit cap costs up to -6.74 dB on view-dependent scenes versus our heterogeneous allocation, matching an information-theoretic reverse-water-filling analysis we develop. GETA-3DGS is complementary to existing codecs: entropy coding (HAC++, CompGS) is downstream, so the two can be composed. 

---
# EditPropBench: Measuring Factual Edit Propagation in Scientific Manuscripts 

**Authors**: Garvin Kruthof  

**Link**: [PDF](https://arxiv.org/pdf/2605.02083)  

**Abstract**: Local factual edits in scientific manuscripts often create non-local revision obligations. If a dataset changes from 215 to 80 documents, claims such as 'medium-scale' or 'a few hundred items' may also become stale, even though they do not repeat the edited number. We introduce EditPropBench, a benchmark for measuring whether LLM editors propagate factual edits through dependent manuscript claims. Each item contains an ML/NLP-style synthetic manuscript, a targeted edit, and a controlled fact graph with sentence-level labels for direct targets, required downstream updates, and protected unrelated text. EditPropBench provides a controlled manuscript-level benchmark with sentence-level dependency supervision, three editing protocols, adversarial metric probes, stress-test variants, and a metric suite centered on Edit-Ripple Adherence (ERA). On the hard implicit/free-form stratum, five LLM editing systems span ERA 0.148--0.705; even the strongest misses roughly 30% of required cascade updates. A mixed-stratum stress test shows that LLMs retain a positive advantage over deterministic substitution baselines when easy substitution-solvable cases are included. Finally, an audit of recent arXiv cs.CL benchmark and dataset papers finds fact-dependent qualitative claims in 37.2% of papers. EditPropBench shows that current LLM editors can repair many implicit consequences of factual edits, but reliable scientific revision still requires cascade-aware checking. 

---
# Cripping AI: Reimagining AI Through Lived Disability Experiences 

**Authors**: Xinru Tang, Ting-an Lin, Jingjin Li, Shaomei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02080)  

**Abstract**: Drawing on crip theory, this paper proposes cripping AI as a guiding framework to center lived disability experiences in AI research and development. Moving beyond calls to make AI "accessible" to people with disabilities, cripping AI seeks to: (1) reveal and dismantle ableist assumptions embedded in how AI is imagined, designed, and evaluated; (2) center disabled ways of knowing (i.e., cripistemologies); (3) respect disabled labor in co-creating accessible practices. We demonstrate how to apply our framework with three cases: deafness and sign language AI, blindness and visual assistive AI, and stuttering and speech AI. We end by outlining three directions for future work, including cripping AI with diverse human bodyminds, across the entire AI pipeline and ecosystem, and in collaboration with other justice-oriented AI efforts. 

---
# Pair2Score: Pairwise-to-Absolute Transfer for LLM-Based Essay Scoring 

**Authors**: İbrahim Rıza Hallaç, Hasan Oğul  

**Link**: [PDF](https://arxiv.org/pdf/2605.02069)  

**Abstract**: Many scoring applications require absolute predictions, while pairwise comparisons can provide a simpler learning objective. We present Pair2Score, a two-stage learning framework that transfers pairwise comparisons into absolute scoring with parameter-efficient LLaMA adaptation. Stage 1 trains a directional Siamese ranker on pairwise comparisons derived from absolute trait labels; Stage 2 trains an absolute predictor using configurable transfer strategies (warm-start and embedding-fusion variants). We evaluate on rubric-aligned Automated Essay Scoring (AES) traits (grammar, vocabulary, syntax) under a five-fold protocol that co-rotates held-out fold and random seed. At the trait level, the best-performing transfer variant improves quadratic weighted kappa (QWK) over an absolute-only baseline for all three traits. However, not all transfer configurations help: a one-epoch pairwise stage transfers more reliably than extended pairwise training, and transfer configuration -- not just the inclusion of a pairwise stage -- determines whether downstream scoring benefits. 

---
# Coopetition-Gym v1: A Formally Grounded Platform for Mixed-Motive Multi-Agent Reinforcement Learning under Strategic Coopetition 

**Authors**: Vik Pant, Eric Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02063)  

**Abstract**: We present Coopetition-Gym v1, a benchmark platform for mixed-motive multi-agent reinforcement learning under strategic coopetition. The platform comprises twenty environments organized into four mechanism classes that correspond to four foundational technical reports: interdependence and complementarity (arXiv:2510.18802), trust and reputation dynamics (arXiv:2510.24909), collective action and loyalty (arXiv:2601.16237), and sequential interaction and reciprocity (arXiv:2604.01240). Each environment carries a closed-form payoff structure and a calibrated interdependence matrix derived from the corresponding report.
Every environment exposes a parameterized reward layer configurable across three structurally distinct modes (private, integrated, cooperative). This separation of payoff from reward enables reward-type ablation, the platform's principal methodological apparatus. Four of the twenty environments are calibrated against historically documented coopetitive relationships and reproduce their outcomes at 98.3, 81.7, 86.7, and 87.3 percent on the validation rubric (Samsung-Sony LCD, Renault-Nissan Alliance, Apache HTTP Server, Apple iOS App Store).
The platform exposes Gymnasium, PettingZoo Parallel, and PettingZoo AEC interfaces and ships 126 reference algorithms: 16 learning algorithms, 7 game-theoretic oracles, 2 heuristic baselines, and 101 constant-action policies. A reference experimental study trained the 16 learning algorithms on every environment under every reward configuration with seven random seeds, producing a 25,708-run training corpus and a 1,116-run behavioral audit corpus, both released under CC-BY-4.0 with Croissant 1.0 metadata. Coopetition-Gym v1 is the first platform to combine continuous-action mixed-motive environments, parameterized reward mutuality, calibrated interdependence coefficients, game-theoretic oracle baselines, and validated case studies. 

---
# Principles and Guidelines for Randomized Controlled Trials in AI Evaluation 

**Authors**: Christopher Kelly, Angelica Chowdhury, Alexandra Campili, Bimpe Ayoola, Devin Barbour, Thomas Chen Dawson, Ze Shen Chin, Rokas Gipiškis  

**Link**: [PDF](https://arxiv.org/pdf/2605.02050)  

**Abstract**: This work establishes a foundational framework for standardizing AI evaluation RCTs (sometimes called human uplift studies). Drawing on established experimental practices from disciplines with established RCT traditions, including software engineering, economics, clinical and health sciences, and psychology, we adopt the (Shadish et al., 2002) four-validity framework and extend it with a fifth principle on transparency, repeatability, and verification adapted from the Transparency and Openness Promotion (TOP) Guidelines (Center for Open Science, 2025). We operationalize all five principles into 33 guidelines adapted for AI evaluation RCT contexts, expressed as requirements with rationales, implementation instructions, and evidence bases. We position the principles and guidelines as serving three key roles for AI evaluation RCTs: a design tool for planning studies, an evaluation rubric for assessing existing work, and a blueprint for standard setting as the field converges on norms. Our framework extends prior work by centering evaluation on human performance rather than model output alone, formalizing causal inference through RCT methodology for AI contexts, integrating heterogeneity analysis and practical significance assessment, implementing a graded transparency and repeatability framework, and addressing AI-specific challenges including model versioning, human-AI interaction dynamics, contamination and spillover effects, and equitable impact assessment. 

---
# Optimization of CV-QKD Under Practical Constraints 

**Authors**: Svitlana Matsenko, Amirhossein Ghazisaeidi, Marcin Jarzyna, Konrad Banaszek, Darko Zibar  

**Link**: [PDF](https://arxiv.org/pdf/2605.02045)  

**Abstract**: Using reinforcement learning, we optimize for practical hardware constraints, including limited FIR filter taps at the transmitter and receiver, mean photon number and finite DAC/ADC resolution. Under these realistic conditions, the proposed approach achieves significant performance improvements. 

---
# What Single-Prompt Accuracy Misses: A Multi-Variant Reliability Audit of Language Models 

**Authors**: Ranit Karmakar, Jayita Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2605.02038)  

**Abstract**: Single-prompt accuracy is the dominant way to benchmark language models, but it can miss reliability failures that matter. We evaluate a 15-model open-weight corpus, with the main reliability analyses focused on 10 instruct models across five classification and reasoning benchmarks under five prompt variants each, measuring accuracy, token-probability calibration, verbal-confidence calibration, verbal parse rate, and prompt-perturbation spread for every (model x dataset x variant) cell. We find three broad results. First, evaluation design can materially change the conclusion. Switching Expected Calibration Error (ECE) token from a raw to a label-set-normalised definition changes per-cell calibration by a mean absolute 0.149. More strikingly, pairing a chain-of-thought prompt with a first-character evaluator on ARC-Challenge reduces apparent accuracy by 72-88% across all five primary models; two independent repair procedures recover 93.8% and 102.7% of the lost performance, indicating an evaluator-side rather than model-side failure. Second, confidence signals are fragile. On MMLU-Pro, every primary model verbally reports confidence substantially above both its accuracy and its token-probability confidence on the same rows, and verbal parse rate can collapse for a single model on a single prompt variant. Third, prompt robustness does not track parameter count reliably. Across 10 instruct models, the correlation between model size and prompt-perturbation spread ranges from -0.244 to 0.474 across benchmarks. Taken together, these results show that reliability conclusions for small language models depend not only on the model being evaluated, but also on the evaluation pipeline used to measure it. We argue that calibration definitions, evaluator logic, verbal parseability, and prompt robustness should be reported explicitly when making reliability claims. 

---
# VILAS: A VLA-Integrated Low-cost Architecture with Soft Grasping for Robotic Manipulation 

**Authors**: Zijian An, Hadi Khezam, Bill Cai, Ran Yang, Shijie Geng, Yiming Feng, Zheng, Lifeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.02037)  

**Abstract**: We present VILAS, a fully low-cost, modular robotic manipulation platform designed to support end-to-end vision-language-action (VLA) policy learning and deployment on accessible hardware. The system integrates a Fairino FR5 collaborative arm, a Jodell RG52-50 electric gripper, and a dual-camera perception module, unified through a ZMQ-based communication architecture that seamlessly coordinates teleoperation, data collection, and policy deployment within a single framework. To enable safe manipulation of fragile objects without relying on explicit force sensing, we design a kirigami-based soft compliant gripper extension that induces predictable deformation under compressive loading, providing gentle and repeatable contact with delicate targets. We deploy and evaluate three state-of-the-art VLA models on the VILAS platform: pi_0, pi_0.5, and GR00T N1.6. All models are fine-tuned from publicly released pretrained checkpoints using an identical demonstration dataset collected via our teleoperation pipeline. Experiments on a grape grasping task validate the effectiveness of the proposed system, confirming that capable manipulation policies can be successfully trained and deployed on low-cost modular hardware. Our results further provide practical insights into the deployment characteristics of current VLA models in real-world settings. 

---
# A Multimodal Dataset for Visually Grounded Ambiguity in Machine Translation 

**Authors**: Jingheng Pan, Xintong Wang, Longyue Wang, Liang Ding, Weihua Luo, Chris Biemann  

**Link**: [PDF](https://arxiv.org/pdf/2605.02035)  

**Abstract**: Ambiguity resolution is a key challenge in multimodal machine translation (MMT), where models must genuinely leverage visual input to map an ambiguous expression to its intended meaning. Although prior work has proposed disambiguation-oriented benchmarks that provide supportive evidence for the role of vision, we observe substantial issues in data quality and a mismatch with translation scenarios. Moreover, existing ambiguity-oriented evaluations are not well suited to broader ambiguity types in open-ended translation. To address these limitations, we present VIDA (Visually-Dependent Ambiguity), a dataset of 2,500 carefully curated instances in which resolving an annotated ambiguous source span requires visual evidence. We further propose Disambiguation-Centric Metrics that use an LLM-as-a-judge classifier to verify whether annotated ambiguous expressions are resolved correctly at the span level. Experiments with two state-of-the-art Large Vision Language Models under vanilla inference, supervised fine-tuning (SFT), and our chain-of-thought SFT (CoT-SFT) show that while SFT improves overall translation quality, CoT-SFT yields more consistent gains in disambiguation accuracy, especially on out-of-distribution subsets, indicating a stronger generalization for resolving diverse ambiguity types. 

---
# Conventional Commit Classification using Large Language Models and Prompt Engineering 

**Authors**: H. M. Sazzad Quadir, Sakib Al Hasan, Md. Nurul Ahad Tawhid  

**Link**: [PDF](https://arxiv.org/pdf/2605.02033)  

**Abstract**: Conventional commits provide a structured format for writing commit messages, which improves readability, software maintenance, and enables automation tools such as changelog generators and semantic versioning systems. Existing approaches to conventional commit classification typically rely on ML/DL models trained on large labeled datasets. In this paper, we investigated a training-free alternative by leveraging large language models (LLMs) through prompt engineering. Rather than building a task-specific classifier, we evaluate three prompting strategies, such as zero-shot, few-shot, and chain-of-thought, across three open-source LLMs of varying scale: Mistral-7B-Instruct, LLaMA-3-8B, and DeepSeek-R1-32B. Classification is performed directly on code diffs extracted from a balanced dataset of 3,200 commits mined from the InfluxDB repository, without any model fine-tuning. Our results show that few-shot prompting consistently achieves the highest accuracy, while chain-of-thought prompting does not yield additional gains for this classification task. Among the evaluated models, DeepSeek-R1-32B achieves the strongest overall performance, suggesting that model scale plays a meaningful role in conventional commit classification. These findings provide practical guidance for researchers and practitioners seeking to automate commit classification without the overhead of curating and maintaining labeled training data. 

---
# Enhancing Judgment Document Generation via Agentic Legal Information Collection and Rubric-Guided Optimization 

**Authors**: Weihang Su, Xuanyi Chen, Yueyue Wu, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02011)  

**Abstract**: Automating the drafting of judgment documents is pivotal to judicial efficiency, yet it remains challenging due to the dual requirements of comprehensive retrieval of legal information and rigorous logical reasoning. Existing approaches, typically relying on standard Retrieval-Augmented Generation and Supervised Fine-Tuning, often suffer from insufficient evidence recall, hallucinated statutory references, and logically flawed legal reasoning. To bridge this gap, we propose Judge-R1, a unified framework designed to enhance LLM-based judgment document generation by jointly improving legal information collection and judgment document generation. First, we introduce Agentic Legal Information Collection, which employs a dynamic planning agent to retrieve precise statutes and precedents from multiple sources. Second, we implement Rubric-Guided Optimization, a reinforcement learning phase utilizing Group Relative Policy Optimization (GRPO) with a comprehensive legal reward function to enforce adherence to judicial standards and reasoning logic. Extensive experiments on the JuDGE benchmark demonstrate that Judge-R1 significantly outperforms state-of-the-art baselines in both legal accuracy and generation quality. 

---
# RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy 

**Authors**: Mario Koddenbrock, Christoph Lange, Robin Legner, Martin Jäger, Martin Kögler, Mariano N. Cruz Bournazou, Peter Neubauer, Felix Biessmann, Erik Rodner  

**Link**: [PDF](https://arxiv.org/pdf/2605.02003)  

**Abstract**: Machine Learning (ML) has transformed many scientific fields, yet key applications still lack standardized benchmarks. Raman spectroscopy, a widely used technique for non-invasive molecular analysis, is one such field where progress is limited by fragmented datasets, inconsistent evaluation, and models that fail to capture the structure of spectral data. We introduce RamanBench, the first large-scale, fully reproducible benchmark for ML on Raman spectroscopy, consisting of streamlined data access, evaluation protocols and code, as well as a live leaderboard. It unifies 74 datasets (including 16 first released with this benchmark) across four domains, comprising 325,668 spectra and spanning classification and regression tasks under diverse experimental conditions. We benchmark 28 models under a standardized protocol, including classical methods (e.g., PLS), Raman-specific (e.g., RamanNet), Tabular Foundation Model (TFM) (e.g., TabPFN), and time-series approaches (e.g., ROCKET). TFM consistently outperform domain-specific and gradient boosting baselines, while time-series models remain competitive. However, no method generalizes across datasets, revealing a fundamental gap. Therefore, we invite the community to contribute new approaches to our living benchmark, with the potential to accelerate advances in critical applications such as medical diagnostics, biological research, and materials science. 

---
# Trojan Hippo: Weaponizing Agent Memory for Data Exfiltration 

**Authors**: Debeshee Das, Julien Piet, Darya Kaviani, Luca Beurer-Kellner, Florian Tramèr, David Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2605.01970)  

**Abstract**: Memory systems enable otherwise-stateless LLM agents to persist user information across sessions, but also introduce a new attack surface. We characterize the Trojan Hippo attack, a class of persistent memory attacks that operates in a more realistic threat model than prior memory poisoning work: the attacker plants a dormant payload into an agent's long-term memory via a single untrusted tool call (e.g., a crafted email), which activates only when the user later discusses sensitive topics such as finance, health, or identity, and exfiltrates high-value personal data to the attacker.
While anecdotal demonstrations of such attacks have appeared against deployed systems, no prior work systematically evaluates them across heterogeneous memory architectures and this http URL introduce a dynamic evaluation framework comprising two components: (1) an OpenEvolve-based adaptive red-teaming benchmark that stress-tests defenses and memory backends against continuously refined attacks, and (2) the first capability-aware security/utility analysis for persistent memory systems, enabling principled reasoning about defense deployment across different usage profiles.
Instantiated on an email assistant across four memory backends (explicit tool memory, agentic memory, RAG, and sliding-window context), Trojan Hippo achieves up to 85-100 percent ASR against current frontier models from OpenAI and Google, with planted memories successfully activating even after 100 benign sessions. We evaluate four memory-system defenses inspired by basic security principles, finding they substantially reduce attack success rates (to as low as 0-5 percent), though at utility costs that vary widely with task requirements. Because of this substantial security-utility tradeoff, the effective real-world deployment of defenses remains an open challenge, which our evaluation framework is specifically designed to address. 

---
# TRAP: Tail-aware Ranking Attack for World-Model Planning 

**Authors**: Siyuan Duan, Ke Zhang, Xizhao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.01950)  

**Abstract**: World models enable long-horizon planning by internally generating and evaluating imagined trajectories, making them a promising foundation for generalist agents. However, this imagination-driven decision process also introduces new security risks. Existing backdoor attacks typically aim to manipulate local features, one-step predictions, or instantaneous policy outputs. While such objectives may suffice for weaker reactive models, they are often ineffective against world models, where the learned dynamics prior and planning process can absorb or wash out the effects of shallow perturbations. More importantly, we find that world models exhibit a distinct backdoor vulnerability rooted in the long-tailed ranking structure of imagined trajectories, where disrupting the ordering of a few decision-critical trajectories can systematically hijack planning.
To exploit this vulnerability, we propose TRAP, a backdoor attack framework for world models that targets imagined trajectory ranking. TRAP combines a tail-aware ranking loss to focus optimization on decision-critical trajectories with dual gating mechanisms that stabilize optimization and regulate when and where the attack penalty is applied. Under trigger conditions, TRAP alters the relative ranking of imagined trajectories to redirect planning outcomes, while largely maintaining the normal ranking structure on clean inputs. Experiments on DreamerV3 and TD-MPC2 across diverse tasks show that TRAP consistently induces sustained behavioral deviations and significant performance degradation, highlighting the need for dedicated security evaluation of world-model-based agents. 

---
# Phone2Act: A Low-Cost, Hardware-Agnostic Teleoperation System for Scalable VLA Data Collection 

**Authors**: Om Mandhane, Bipin Yadav, Sangeetha Prasanna Ram, Gopalakrishnan Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2605.01948)  

**Abstract**: Collecting diverse, high-quality manipulation data for Vision-Language-Action (VLA) model training remains prohibitively expensive for many research groups, as existing teleoperation frameworks rely on specialized hardware or are tightly coupled to specific robot platforms. We present Phone2Act, a low-cost, hardware-agnostic teleoperation framework that transforms a commodity smartphone into a 6-DoF robot controller via Google ARCore. Built on a modular ROS 2 architecture, Phone2Act decouples control logic from hardware specifics through interchangeable bridge nodes, supporting platforms from industrial cobots to low-cost bimanual arms without code modification. A Universal Recorder synchronizes multi-camera RGB streams with robot state feedback and exports demonstrations natively in the LeRobot dataset format, eliminating post-processing and enabling immediate VLA fine-tuning. We validate the framework by fine-tuning GR00T-N1.5 on 130 collected episodes, achieving a 90% success rate on a real-world multi-stage pick-and-place task deployed on a physical Dobot CR5. 

---
# PepSpecBench: A Unified Evaluation Benchmark for Peptide Tandem Mass Spectrometry Prediction 

**Authors**: Zhiwen Yang, Pan Liu, Yifan Li, Yunhua Zhong, Jun Xia  

**Link**: [PDF](https://arxiv.org/pdf/2605.01945)  

**Abstract**: Tandem mass spectrometry provides a high-throughput framework for identifying and quantifying proteins in complex biological samples. In computational proteomics, predicting peptide MS/MS spectra is a critical task, enabling downstream applications such as large-scale peptide identification and quantification. While deep learning architectures have substantially improved prediction accuracy, three evaluation challenges obscure the true progress of the field. First, inconsistent data preprocessing and incompatible model output spaces hinder fair model comparison. Second, flawed data splitting strategies can permit hidden sequence leakage and inflate reported performance. Third, existing evaluations typically lack comprehensive cross-species benchmarking and systematic assessment of model robustness to influential experimental conditions. To address these challenges, we propose PepSpecBench, a unified benchmark for peptide MS/MS spectrum prediction. PepSpecBench standardizes data preprocessing across complementary public datasets, enforces a strict backbone-disjoint splitting strategy to eliminate sequence leakage, and evaluates diverse architectures within a shared fragment-ion representation space. It further introduces a comprehensive multi-species evaluation suite and physically grounded metadata perturbation probes to assess model robustness and instrument awareness. We uncover previously unrecognized performance discrepancies and robustness limitations across six representative models, providing actionable insights for future model design, evaluation and practical deployment. 

---
# RefusalGuard: Geometry-Preserving Fine-Tuning for Safety in LLMs 

**Authors**: Sadia Asif, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2605.01913)  

**Abstract**: Fine-tuning safety-aligned language models for downstream tasks often leads to substantial degradation of refusal behavior, making models vulnerable to adversarial misuse. While prior work has shown that safety-relevant features are encoded in structured representations within the model's activation space, how these representations change during fine-tuning and why alignment degrades remains poorly understood. In this work, we investigate the representation-level mechanisms underlying alignment degradation. Our analysis shows that standard fine-tuning induces systematic drift in safety-relevant representations, distorts their geometric structure, and introduces interference between task optimization and safety features. These effects collectively lead to increased harmful compliance. Motivated by these findings, we introduce REFUSALGUARD, a representation-level fine-tuning framework that preserves safety-relevant structure during model adaptation. Our approach constrains updates in hidden representation space, ensuring that safety-mediating components remain stable while allowing task-specific learning in complementary directions. We evaluate REFUSALGUARD across multiple model families, including LLaMA, Gemma, and Qwen, on adversarial safety benchmarks such as AdvBench, DirectHarm4, and JailbreakBench, as well as downstream utility tasks. Our approach achieves attack success rates comparable to base safety-aligned models while maintaining competitive task performance, significantly outperforming baselines. 

---
# Stochastic Sparse Attention for Memory-Bound Inference 

**Authors**: Kyle Lee, Corentin Delacour, Kevin Callahan-Coray, Kyle Jiang, Can Yaras, Samet Oymak, Tathagata Srimani, Kerem Y. Camsari  

**Link**: [PDF](https://arxiv.org/pdf/2605.01910)  

**Abstract**: Autoregressive decoding becomes bandwidth-limited at long contexts, as generating each token requires reading all $n_k$ key and value vectors from KV cache. We present Stochastic Additive No-mulT Attention (SANTA), a method that sparsifies value-cache access by sampling $S \ll n_k$ indices from the post-softmax distribution and aggregates only those value rows. This yields an unbiased estimator of the post-softmax value aggregation while replacing value-stage multiply-accumulates with gather-and-add. We introduce stratified sampling to design variance-reduced, GPU-friendly variants, demonstrating $1.5\times$ decode-step attention kernel speedup over FlashInfer and FlashDecoding on an NVIDIA RTX 6000 Ada while matching baseline accuracy at 32k-token contexts. Finally, we propose Bernoulli $qK^\mathsf{T}$ sampling as a complementary technique to sparsify the score stage, reducing key-feature access through stochastic ternary queries. Both methods are orthogonal to upstream techniques such as ternary quantization, low-rank projections, and KV-cache compression. Together, they point toward sparse, multiplier-free, and energy-efficient inference. We open-source our kernels at: this https URL 

---
# Behavior-Grounded Lane Representation Learning for Multi-Task Traffic Digital Twins 

**Authors**: Rei Tamaru, Pei Li, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2605.01901)  

**Abstract**: Traffic digital twins are powerful tools for advanced traffic management, and most systems are built on static geometric representations. However, these representations fail to capture the dynamic functional semantics required for behavior-aware reasoning, such as how a lane operates under complex traffic conditions. To address this gap, we introduce GeoLaneRep, a behavior-grounded lane representation learning framework for traffic digital twins. GeoLaneRep jointly encodes static lane geometry, observed vehicle trajectories, and operational descriptors into a shared, cross-camera semantic embedding. The encoder is trained with a joint objective combining contrastive cross-camera alignment, auxiliary role supervision, and temporal anomaly detection. Across 16 roadside cameras and 132 lanes, the learned embeddings achieve a $0.004$ lateral-rank error and an edge-role F1 of $1.000$ in zero-shot cross-camera matching, and an AUROC of $0.991$ for window-level anomaly detection. We further show that the same behavioral embeddings can condition a diffusion-based generator to synthesize lane geometries that satisfy targeted operational specifications, with $87.9\%$ overall specification accuracy across 38 lane groups. GeoLaneRep thus provides a semantic interface between roadside observations and downstream digital twin tasks, supporting cross-camera transfer, behavior-aware monitoring, and goal-directed lane synthesis. The framework is openly available at this https URL. 

---
# AFFormer: Adaptive Feature Fusion Transformer for V2X Cooperative Perception under Channel Impairments 

**Authors**: Xi Zhou, Tao Huang, Qing-Long Han, Rana Abbas, Mostafa Rahimi Azghadi  

**Link**: [PDF](https://arxiv.org/pdf/2605.01888)  

**Abstract**: Accurate 3D object detection is essential for ensuring the safety of autonomous vehicles. Cooperative perception, which leverages vehicle-to-everything (V2X) communication to share perceptual data, enhances detection but is vulnerable to channel impairments, such as noise, fading, and interference. To strengthen the reliability of intelligent transportation systems, this work improves the robustness of V2X cooperative perception under communication conditions that reflect common channel impairments. This paper proposes an Adaptive Feature Fusion Transformer (AFFormer), a Transformer-based framework that mitigates the adverse effects of corrupted features by modeling temporal, inter-agent, and spatial correlations. AFFormer introduces three key modules: Multi-Agent and Temporal Aggregation for context-aware fusion across agents and over time, Dual Spatial Attention for efficient modeling of spatial dependencies, and Uncertainty-Guided Fusion for entropy-driven refinement of fused features. A teacher-student knowledge distillation strategy further enhances robustness by aligning fused features with reliable early-collaboration supervision. AFFormer is validated on the V2XSet and DAIR-V2X datasets, where it consistently outperforms existing methods under both ideal and impaired communication conditions, demonstrating improved robustness to communication-induced feature degradation while maintaining a competitive efficiency-accuracy trade-off. 

---
# Chart-FR1: Visual Focus-Driven Fine-Grained Reasoning on Dense Charts 

**Authors**: Hongkun Pan, Yuwei Wu, Wanyi Hong, Shenghui Hu, Qitong Yan, Yi Yang, Rufei Han, Changju Zhou, Minfeng Zhu, Dongming Han, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.01882)  

**Abstract**: Multimodal large language models (MLLMs) have shown considerable potential in chart understanding and reasoning tasks. However, they still struggle with high information density (HID) charts characterized by multiple subplots, legends, and dense annotations due to three major challenges: (1) limited fine-grained perception results in the omission of critical visual cues; (2) redundant or noisy visual information undermines the performance of multimodal reasoning; (3) lack of adaptive deep reasoning relative to the amount of visual information. To tackle these challenges, we present a novel focus-driven fine-grained chart reasoning model, Chart-FR1, to improve perception, focusing efficiency, and adaptive deep reasoning on HID charts. Specifically, we propose Focus-CoT, a visual focusing chain-of-thought that enhances fine-grained perception by explicitly linking reasoning steps to key visual cues, such as local image regions and OCR signals. Building on this, we introduce Focus-GRPO, a focus-driven reinforcement learning algorithm with an information-efficiency reward that compresses redundant visual information for efficient focusing, and an adaptive KL penalty mechanism that enables flexible control over reasoning depth as more visual cues are discovered. Furthermore, to fill the gap in benchmarks for HID charts, we build HID-Chart, a challenging benchmark with an information-density metric designed to evaluate fine-grained chart reasoning capabilities. Extensive experiments on multiple chart benchmarks demonstrate that Chart-FR1 outperforms state-of-the-art MLLMs in chart understanding and reasoning. Code is available at this https URL. 

---
# BadmintonGRF: A Multimodal Dataset and Benchmark for Markerless Ground Reaction Force Estimation in Badminton 

**Authors**: Kuoye Niu, Jianwei Li, Shengze Cai, Yong Ma, Mengyao Jia, Lishun Shen, Zhenheng Zhang, Yuxin Peng, Xian Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.01876)  

**Abstract**: Multimodal resources for non-periodic court sports with laboratory-grade sensing remain scarce: few publicly pair instrumented ground reaction force (GRF) with high-frame-rate multi-view video, limiting markerless load estimation in realistic training settings. BadmintonGRF records eight synchronized RGB views at ~120 FPS, four Kistler force plates, and Vicon motion capture (C3D) without hardware genlock across modalities; alignment combines human-verified events, automated quality assurance, and per-camera time offsets with uncertainty metadata. Tier 1 distributes pose, time-aligned GRF, metadata, and splits under CC BY-NC 4.0, enabling the primary benchmark without raw RGB or C3D; we report a Tier 1 task that maps 2D pose to GRF. Tier 2 provides raw RGB and C3D under controlled access for studies that require appearance or full kinematics. The public release contains 17,425 impact-segment archives in the 10-subject benchmark tree (156 instrumented trials; raw multi-view RGB alone exceeds 1 TB); benchmark loader gates retain 12,867 view-specific instances and 1,732 unique impacts after multi-view deduplication. We are not aware of prior public badminton corpora that combine this sensing layout with audited video--GRF alignment for impact-centric GRF estimation. We distribute preprocessing code, leave-one-subject-out splits, ten reference baselines, and optional late fusion (one deterministic test-time pass per instance; no test-time augmentation), with a within-trial diagnostic in the supplementary material. 

---
# Leveraging Data Symmetries to Select an Optimal Subset of Training Data under Label Noise 

**Authors**: Kumar Shubham, Pavan Karjol, Kiran M K, Prathosh AP  

**Link**: [PDF](https://arxiv.org/pdf/2605.01874)  

**Abstract**: The performance of machine learning models often relies on large labeled datasets; however, data collected from diverse sources can contain label noise. Recent work has shown that, in noisy settings, there may exist a subset of the training data on which models can achieve performance comparable to training on a noise-free dataset. A widely used method for identifying such subsets is cutstats, which employs k-nearest neighbors (k-NN) to detect low-noise samples. However, its performance on high-dimensional data remains largely unexplored. In this work, we formally establish that the performance of a classifier trained on a subset of a noisy dataset selected via cutstats is influenced by the accuracy of k-NN. We further demonstrate that, in noisy environments, exploiting data invariance and knowledge of underlying symmetries can significantly enhance the performance of k-NN, bringing it closer to the Bayes optimal classifier even in high-dimensional regimes. Finally, we show that for real-world scenarios, where information about the underlying invariance is only partially known, learnt invariant representations can still facilitate the identification of near-optimal subsets. 

---
# ShiftLIF: Efficient Multi-Level Spiking Neurons with Power-of-Two Quantization 

**Authors**: Kaiwen Tang, Di Yu, Jiaqi Zheng, Changze Lv, Qianhui Liu, Zhanglu Yan, Weng-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2605.01866)  

**Abstract**: Spiking neural networks (SNNs) are promising for edge sensing due to their event-driven computation and temporal filtering capability. However, standard leaky integrate-and-fire (LIF) neurons communicate only through binary spikes, which severely limit representational capacity. Existing multi-level spiking neurons improve information transmission, but often rely on uniform quantization that mismatches membrane-potential distributions or introduces costly synaptic multiplications. In this paper, we propose ShiftLIF, a multi-level spiking neuron that maps membrane potentials to a logarithmically spaced power-of-two spike set. This design provides finer representation in the small-amplitude regime, where membrane potentials are densely concentrated, while enabling multiplier-free synaptic computation through bit-shift and accumulation operations. As a result, ShiftLIF improves spike-level expressiveness without sacrificing the hardware-friendly nature of standard SNN computation. We evaluate ShiftLIF on 10 datasets spanning wireless, acoustic, motion, and visual sensing tasks. Results show that ShiftLIF consistently matches or exceeds the accuracy of existing multi-level spiking neurons while maintaining synaptic energy consumption close to standard binary LIF. These results indicate that ShiftLIF provides a favorable accuracy-efficiency trade-off for cross-modal edge sensing. 

---
# Quality-Aware Exploration Budget Allocation for Cooperative Multi-Agent Reinforcement Learning 

**Authors**: Dahyun Oh, Minhyuk Yoon, H.Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.01865)  

**Abstract**: Cooperative multi-agent reinforcement learning (MARL) requires agents to discover joint strategies in a combinatorially large state-action space, yet effective coordination configurations are exceedingly rare. Intrinsic motivation, which augments task rewards with novelty bonuses, is a popular approach for driving exploration, but its effectiveness hinges on the exploration intensity $\beta$, where too large a value overwhelms the task signal and causes coordination collapse, while too small a value prevents discovery of rare strategies. We address two complementary challenges: adapting $\beta$ globally over training, and allocating the exploration budget across agents whose intrinsic reward signals vary in reliability. Our framework combines a return-conditioned sigmoid schedule (RCB) for global intensity control with a per-agent Reward Signal Quality (RSQ) metric that concentrates the exploration budget on agents with reliable signals. The core insight is that agents receiving noisy intrinsic rewards should explore less aggressively, and this allocation can be determined automatically from signal-to-noise statistics. Successor Distance (SD), a quasimetric intrinsic reward, naturally produces distinguishable per-agent signal quality, completing the framework with convergence and ordering preservation guarantees. On seven cooperative benchmarks (MPE, SMAX, MABrax), our method achieves top-tier returns across all environments. 

---
# Spatiotemporal Hidden-State Dynamics as a Signature of Internal Reasoning in Large Language Models 

**Authors**: Kotaro Furuya, Takahito Tanimura  

**Link**: [PDF](https://arxiv.org/pdf/2605.01853)  

**Abstract**: Large reasoning models (LRMs) generate extended solutions, yet it remains unclear whether these traces reflect substantive internal computation or merely verbosity and overthinking. Although recent hidden-state analyses suggest that internal representations carry correctness-related signals, their coarse aggregations may obscure the token and layer structure underlying reasoning computation. We investigate hidden-state transitions across decoding steps and layers, and identify a distinct spatiotemporal pattern in LRMs: successful trajectories exhibit broad temporal dynamics with localized layer-wise concentration, while this structure is weaker in non-reasoning models and knowledge-heavy domains. We formalize this characteristic as Spatiotemporal Amplitude of Latent Transition (StALT), a training-free trajectory statistic that summarizes temporal changes between adjacent tokens weighted by within-token layer saliency. Across diverse models and benchmarks, StALT reliably separates correct from incorrect trajectories in reasoning-intensive regimes, providing a competitive label-free correctness signal alongside strong output-space and length-based baselines. Intervention analyses further show that this spatiotemporal amplitude responds systematically to manipulations that increase or reduce the demand for internal reasoning, supporting its association with latent reasoning dynamics in LRMs. These findings provide empirical evidence that LRMs exhibit measurable hidden-state dynamics and offer a practical probe for understanding internal computation beyond output-based evaluation. 

---
# Disentangled Anatomy-Disease Diffusion (DADD) for Controllable Ulcerative Colitis Progression Synthesis 

**Authors**: Umut Dundar, Alptekin Temizel  

**Link**: [PDF](https://arxiv.org/pdf/2605.01848)  

**Abstract**: Synthesizing longitudinal medical images at controllable disease stages while preserving patient-specific anatomy is hindered by the entanglement of pathological textures and structural features. We address this challenge for ulcerative colitis (UC) endoscopy, where severity follows a continuous ordinal progression along the Mayo Endoscopic Score (MES). Our framework, Disentangled Anatomy-Disease Diffusion (DADD), conditions a latent diffusion model on two complementary embeddings: a pretrained image encoder for patient anatomy and a separately trained ordinal embedder for cumulative disease severity. Since image embeddings inevitably capture disease information, we introduce a Feature Purifier, a cross-attention-based erasure mechanism that identifies and suppresses disease-correlated channels, yielding purified anatomical representations. These cleaned anatomy tokens and target disease tokens are injected into the denoising network via a Triple-Pathway Cross-Attention mechanism with resolution-dependent routing gates. This architecture leverages the U-Net hierarchy, in which different network depths encode global structure versus fine-grained pathological texture. Furthermore, we introduce Delta Steering, a training-free directional signal derived from the ordinal embeddings that enables explicit, single-pass control over disease transitions at inference without requiring additional forward passes. Validated on the LIMUC dataset, our approach produces high-fidelity images across all severity levels and effectively rebalances skewed class distributions, enhancing performance for downstream classification tasks. The dataset is available at this http URL and the code base at this http URL 

---
# Repurposing and Evaluating the (In)Feasibility of Dataset Poisoning enabled Watermarking for Contrastive Learning 

**Authors**: Zhiyang Dai, Yansong Gao, Boyu Kuang, Haodong Li, Qi Chang, Gaurav Varshney, Derek Abbott, Anmin Fu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01834)  

**Abstract**: Contrastive learning (CL) reduces annotation cost via auto-derived supervisory signals. Since large-scale in-house CL datasets are infeasible, reliance on third-party or internet data is common. Recent studies show CL models are vulnerable to data-poisoning backdoor attacks, but their generalization and robustness are underexplored. We systematically evaluate existing data-poisoning backdoor attacks on CL, revealing limitations: poor dataset adaptability, low success rates, limited portability, and restrictive assumptions (e.g., downstream task knowledge). Interestingly, trigger samples exhibit distinguishable statistical divergence from clean samples, which inspires repurposing it as a watermark for dataset IP protection. Direct repurposing is challenging due to low success rates; we overcome this by statistical verification using a unified density metric. We further propose a multi-level watermarking scheme adapting to feature-level, soft-label, or hard-label outputs in CL. Experiments show some backdoor attacks can be repurposed as effective watermarks with trade-offs among fidelity, verifiability, and robustness. This work demonstrates weak backdoor effects become reliable signals for dataset IP protection in challenging CL settings. 

---
# Remote Action Generation: Remote Control with Minimal Communication 

**Authors**: Szymon Kobus, Deniz Gündüz  

**Link**: [PDF](https://arxiv.org/pdf/2605.01833)  

**Abstract**: We address the challenge of remote control where one or more actors, lacking direct reward access, are steered by a controller over a communication-constrained channel. The controller learns an optimal policy from observed rewards and communicates action guidance to the actors, which becomes demanding for large or continuous action spaces. To achieve rate-efficient communication throughout this interactive learning and control process, we introduce a novel framework leveraging remote generation. Instead of transmitting full action specifications, the controller sends minimal information, enabling the actors to locally generate actions by sampling from the controller's evolving target policy. This guided sampling is facilitated by an importance sampling approach. Concurrently, the actors use the received guidance as supervised learning data to learn the controller's policy. This actor-side learning improves their local sampling capabilities, progressively reducing future communication needs. Our solution, Guided Remote Action Sampling Policy (GRASP), demonstrates significant communication reduction, achieving an average 12-fold data reduction across all experiments (50-fold for continuous action spaces) compared to direct action transmission, and a 41-fold reduction compared to reward transmission. 

---
# RMGAP: Benchmarking the Generalization of Reward Models across Diverse Preferences 

**Authors**: Yangyang Zhou, Yi-Chen Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.01831)  

**Abstract**: Reinforcement Learning from Human Feedback has become the standard paradigm for language model alignment, where reward models directly determine alignment effectiveness. In this work, we focus on how to evaluate the generalizability of reward models. By "generalizability", we mean the ability of RMs to correctly rank responses to align with diverse user preferences. However, existing reward model benchmarks are typically designed around a universal preference, failing to assess this generalization. To address this critical gap, we introduce RMGAP, a benchmark comprising 1,097 instances across Chat, Writing, Reasoning, and Safety domains. Since different users exhibit diverse preferences for the same task, we first generate four distinct responses with different linguistic profiles for each collected prompt. However, the original prompt set lacks the specificity to convey different preferences. We therefore construct tailored prompts by contrasting these candidates and designing scenarios in which one response becomes the uniquely appropriate choice. Moreover, we observe that users often express the same preference using different phrasings, and thus extend each prompt with two paraphrased variants. Our evaluation of 24 state-of-the-art RMs reveals their substantial limitations: even the best RM achieves only 49.27% Best-of-N accuracy, highlighting considerable room for improvement in reward model generalization. Related data and code are available at this https URL. 

---
# GeoSAE: Geometric Prior-Guided Layer-Wise Sparse Autoencoder Annotation of Brain MRI Foundation Models 

**Authors**: Favour Nerrise, Lucy Yin, Mohammad H. Abbasi, Kilian M. Pohl, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2605.01829)  

**Abstract**: Brain MRI foundation models learn rich representations of anatomy, but interpreting what clinical information they encode remains an open problem. Standard sparse autoencoders (SAEs) suffer from severe feature collapse in deep transformer layers, and in Alzheimer's disease (AD) research, aging confounds nearly every clinical variable, making naive annotation unreliable. We propose GeoSAE, a geometry-guided SAE framework that uses the foundation model's learned manifold structure to prevent feature collapse and annotates each surviving feature via age-deconfounded partial correlations. Applied to ~14k T1-weighted MRI scans from the Alzheimer's Disease Neuroimaging Initiative (ADNI) and the Australian Imaging biomarkers and Lifestyle (AIBL) datasets, GeoSAE identifies a compact, fully interpretable feature set that predicts mild cognitive impairment (MCI)-to-AD conversion (AUC 0.746) using only 2% of the embedding dimensions, while comorbidity-annotated features achieve only chance-level performance. The identified features replicate across cohorts without retraining (r=0.97) and localize to neuroanatomically distinct regions consistent with Braak staging. This shows that geometry-guided SAEs can extract interpretable, biomarkers from frozen brain MRI foundation models. 

---
# Selector-Guided Autonomous Curriculum for One-Shot Reinforcement Learning from Verifiable Rewards 

**Authors**: Rudray Dave, Vedang Dubey, Smit Deoghare, Sudhakar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2605.01823)  

**Abstract**: Recently, Reinforcement Learning from Verifiable Rewards (RLVR) has been established as a highly effective technique for augmenting the math reasoning skills of Large Language Models (LLMs) based on a single instance. Current state-of-the-art 1-shot RLVR models adopt heuristics for selecting instances, mostly based on historical variance in rewards, which we find to be inherently misleading as a measure of transferability value. In this paper, we propose a Selector-Guided Autonomous Curriculum (SGAC) approach, which employs a learnable selector model on a multi-dimensional feature space consisting of success probability, reward variance, output disagreement (entropy), and semantic difficulty level, instead of the static reward variance heuristic. In our empirical evaluation on pools of candidate problems, we observed that output disagreement, rather than reward variance, is the strongest predictor of reasoning gains in subsequent iterations. Leveraging this finding, we develop an autonomous curriculum algorithm for dynamically siphoning candidate problems from a large pool, ranking them by the learned selector, and running micro-bursts of 1-shot GRPO. Our framework is evaluated using the Hendrycks MATH benchmark, with the Qwen2.5-Math-1.5B model serving as the baseline. Our framework obtains an accuracy of 68.0\% on the hold-out dataset, which is better than the accuracy obtained from the state-of-the-art model, 64.0\%, as well as the 1-shot RLVR checkpoint proposed by Wang et al., which achieved an accuracy of 66.0\%. The results confirm that entropy-based intelligent data curation leads to strict reasoning improvement over static training methods, particularly in severely limited data conditions. 

---
# Federated Semi-Supervised Graph Neural Networks with Prototype-Guided Pseudo-Labeling for Privacy-Preserving Gestational Diabetes Mellitus Prediction 

**Authors**: G. Victor Daniela, A. Mallikarjuna Reddya, Uday Kumar Addankia, Sridhar Reddy Gogua, Sravanth Kumar Ramakuria  

**Link**: [PDF](https://arxiv.org/pdf/2605.01810)  

**Abstract**: Gestational Diabetes Mellitus (GDM) is a high-prevalence pregnancy complication that requires accurate early risk stratification to reduce maternal and fetal morbidity. However, real-world clinical deployment of machine learning is hindered by two coupled constraints: (i) label scarcity, where a large fraction of electronic health records (EHR) lack confirmed diagnostic labels, and (ii) data privacy, which prevents sharing patient-level data across hospitals. This paper proposes FedTGNN-SS, a privacy-preserving federated semi-supervised framework for clinical tabular EHR. Each hospital builds a local k-nearest-neighbor patient similarity graph and trains a topology-adaptive GNN encoder. To robustly exploit unlabeled records, FedTGNN-SS combines (1) prototype-guided pseudo-labeling with neighborhood agreement, (2) adaptive graph refinement that periodically updates the k-NN graph using learned embeddings, (3) clinical-aware consistency augmentation applied only to continuous variables, and (4) privacy-safe prototype sharing that exchanges only class-level centroids. Across three diabetes-related datasets (GDM: N = 3,525; Pima: N = 768; Early Stage: N = 520) under 10\%-80\% missing labels per silo, FedTGNN-SS achieves 56 significant wins ($p < 0.05$) against 11 federated baselines and attains strong AUROC under extreme scarcity (Pima: 0.8037 at 80\% missing, Early Stage: 0.9634 at 80\% missing). 

---
# TMD-Bench: A Multi-Level Evaluation Paradigm for Music-Dance Co-Generation 

**Authors**: Xiaoda Yang, Majun Zhang, Changhao Pan, Nick Huang, Yang Yuguang, Fan Zhuo, Pengfei Zhou, Jin Zhou, Sizhe Shan, Shan Yang, Miles Yang, Yang You, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.01809)  

**Abstract**: Unified audio-visual generation is rapidly gaining industrial and creative relevance, enabling applications in virtual production and interactive media. However, when moving from general audio-video synthesis to music-dance co-generation, the task becomes substantially harder: musical rhythm, phrasing, and accents must drive choreographic motion at fine temporal resolution, and such rhythmic coupling is not captured by unimodal metrics or generic audiovisual consistency scores used in current evaluation practice. We introduce TMD-Bench, a benchmark for text-driven music-dance co-generation that assesses systems across unimodal generation quality, instruction adherence, and cross-modal rhythmic alignment. The benchmark integrates computable physical metrics with perceptual multimodal judgments, and is supported by a curated rhythm-aligned music-dance dataset and a fine-grained Music Captioner for structured music semantics. TMD-Bench further reveals that (i) modern commercial audio-visual models, such as Veo 3 and Sora 2, produce high-quality music and video, while rhythmic coupling remains less consistently optimized and leaves room for improvement, and (ii) our unified baseline RhyJAM trained on rhythm-aligned data achieves competitive beat-level synchronization while maintaining competitive unimodal fidelity. This presents prospects for building next-generation music-dance models that explicitly optimize rhythmic and kinetic coherence. 

---
# Discover Fast Power Allocation Solution for Multi-Target Tracking via AlphaEvolve Evolution 

**Authors**: Zhenkang Hou, Wenqiang Pu, Junkun Yan, Rui Zhou, Hongwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01794)  

**Abstract**: Efficient radar resource allocation is a fundamental yet computationally challenging problem, as optimal solutions typically require iterative optimization with high complexity. Motivated by the need for real-time scheduling, robust generalization, and low data dependency, this paper proposes a novel paradigm that leverages large language model (LLM)-guided evolutionary search (AlphaEvolve) to autonomously discover a closed-form power allocation solution for multi-target tracking. The approach encodes high-dimensional radar states into physically inspired features, then evolves a compact and interpretable scoring function, which is transformed to feasible power allocations via a deterministic constraint-satisfying transformation. Extensive experiments demonstrate that the discovered closed-form solution achieves near-optimal tracking accuracy (average relative performance loss of only $1.51\%$), reliable generalization across diverse scenarios and target counts, and over three orders of magnitude speedup compared to conventional iterative solvers. These results highlight the potential of LLM-guided symbolic search to revolutionize not only radar resource management but also broader classes of engineering optimization problems. 

---
# Khala: Scaling Acoustic Token Language Models Toward High-Fidelity Music Generation 

**Authors**: Jiafeng Liu, Yuanliang Dong, Hongjia Liu, Yuqing Cheng, Zhancheng Guo, Huijing Liang, Wenbo Zhan, Yuming Sun, Xiaobing Li, Feng Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.01790)  

**Abstract**: A common design pattern in high-quality music generation is to handle structure and fidelity in different representation spaces: a generator first models high-level structure, followed by diffusion-based or neural decoding stages that reconstruct fine details. In this work, we explore an alternative view: both may be progressively modeled within a single deep acoustic-token hierarchy. To study this, we build a 64-layer residual vector quantization (RVQ) acoustic representation and propose a two-stage coarse-to-fine generation framework. A backbone model first generates coarse acoustic tokens for the full track, and a super-resolution model then completes finer tokens within the same acoustic token space. The super-resolution stage works at full-track scale and refines tokens layer by layer while running in parallel over time, leading to a fixed 62-step inference process. To jointly improve lyric alignment and fine-detail reconstruction, we further introduce hybrid-attention training: the alignment objective uses causal attention, while layer-wise refinement uses full attention. A key finding is that text--vocal alignment can emerge within pure acoustic-token language modeling, without requiring a separate semantic token stage. Moreover, initializing the super-resolution model from the trained backbone significantly improves convergence and final quality. Taken together, our results suggest that high-quality music generation can be effectively pursued without separating structure and fidelity into heterogeneous representation spaces. Instead, both can be progressively modeled within a unified acoustic-token hierarchy, pointing toward a simpler and more unified path to high-quality music generation. 

---
# Data driven approach for Outdoor Channel Prediction in 5G and Beyond 

**Authors**: A. Sathi Babu, V. Udaya Sankar, Vishnu Ram OV  

**Link**: [PDF](https://arxiv.org/pdf/2605.01777)  

**Abstract**: An evolution of Wireless Communications towards 5G and beyond provides improved user experience in terms of quality of services. Understanding and estimating Channel information plays crucial role in providing better user experience. Traditional methods of channel estimation involves periodically sending pilots (known signals), estimating channel and send back estimated channel information to the BS which increases computational complexity and communication complexity. Hence, we focus on data driven approach for channel estimation. This work can be deployed as Digital twin in 5G and beyond wireless networks. In this work, we explore a channel estimation mechanism at 7GHz frequency band for a given user location. This work involves data generation using Ray tracing mechanism and Machine learning model training that contains feature variables such as transmitter location, user location and target variable as channel coefficient . We explored Linear Regression, Support Vector Regression and Decision Tree Regression. We found via simulations that Linear Regression performs (with MAE of $\mathbf{7.5155\times10^{-5}}$ and RMSE of $\mathbf{9.2861\times10^{-5}}$) better than Support Vector Regression and Decision Tree Regression. 

---
# The Compliance Gap: Why AI Systems Promise to Follow Process Instructions but Don't 

**Authors**: Kwan Soo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2605.01771)  

**Abstract**: An auditor instructs an AI assistant: "open each file individually using the Read tool -- no scripts, no agents." The AI replies "Yes" -- then issues a single batched call summarizing all fifty files at once. We call this the Compliance Gap: a third, orthogonal axis of AI honesty distinct from factual truthfulness and rhetorical substance. Three questions: does this verbal-behavioral disconnect exist (existence); can any text-only observer recover it (detectability); what infrastructure does AI deployment need (remedy)? Some 75 benchmarks (IFEval, SWE-bench, BFCL, COMPASS, SpecEval) measure outcome fidelity; none measures process fidelity. Theorem 1 shows the gap is structurally inevitable under RL that rewards text without observing behavior. Theorem 2, via the Data Processing Inequality, shows it is undetectable from text alone -- by any human or LLM observer, present or future. Thirteen experiments and 2,031 sessions on six frontier models confirm both predictions. Under default framing, all six exhibit instruction compliance rates of 0% -- Claude Sonnet 4 verbally agrees ten out of ten times then bypasses in all ten. The gap is selective: 97% compliance where rationale is rewarded (audit trails), 0-4% where it is not (file reading, privacy masking); removing delegation tools raises compliance to 75% (Cohen's d = 2.47), confirming environmental affordance rather than weight-encoded failure. Nine blinded human raters achieve Fleiss' kappa = 0.130 and correctly identify zero of fifteen compliant sessions, exactly as Theorem 2 predicts. Where humans show 47% intention-behavior gaps in psychology and 96.5pp gaps in surgical audits, RLHF-trained models approach 100% under default conditions -- a regime warranting its own measurement infrastructure. We release BS-Bench: the first open benchmark for process compliance, with seven tool-call-log audit metrics and a public leaderboard. 

---
# Talk is Cheap, Communication is Hard: Dynamic Grounding Failures and Repair in Multi-Agent Negotiation 

**Authors**: Yiheng Yao, Chelsea Zou, Robert D. Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2605.01750)  

**Abstract**: Grounding is the collaborative process of establishing mutual belief sufficient for the current communicative purpose. While static grounding maps language to a shared, externally observable context, dynamic grounding is a joint activity where meaning is negotiated through interaction. Current multi-agent Large Language Model (LLM) benchmarks focus on static, one-shot tasks, overlooking the ability to repair grounding breakdowns across turns. We introduce an iterated, multi-turn negotiation game in which two agents allocate shared resources toward private projects with verifiable jointly optimal outcomes. While individual agents can identify Pareto-optimal allocations in isolation, agent dyads consistently fail to reach them across open- and closed-source models. Our investigation reveals four failure modes: (1) coordination degrades when shared interaction history is absent; (2) yet accumulated context can itself become a liability through stubborn anchoring, where initial proposals are treated as axiomatic rather than negotiable; (3) a reliance on perfunctory fairness (equal resource splits) over reward-maximizing coordination; and (4) failures in referential binding, where agents lose track of commitments across turns. These results highlight dynamic grounding as a critical and understudied axis of multi-agent coordination. Our framework decomposes the coordination gap into measurable components: the oracle baseline establishes that the gap is not attributable to individual reasoning limitations; the no-talk baseline establishes that communication is necessary; and a full-transparency intervention establishes that information exchange alone is insufficient: the bottleneck lies in the interactive processes of joint plan formation, commitment, and execution that constitute dynamic grounding. 

---
# Architectural Obsolescence of Unhardened Agentic-AI Runtimes 

**Authors**: Alfredo Metere  

**Link**: [PDF](https://arxiv.org/pdf/2605.01740)  

**Abstract**: An agentic-AI runtime issues tool calls, sends messages, and actuates devices on behalf of an LLM. Catching the four ways an action can diverge from its audit record -- F1 gate-bypass, F2 audit-forgery, silent host failure, F4 wrong-target, -- is a load-bearing safety property of any such runtime. We show that upstream OpenClaw, the most engineered single-user agentic-AI gateway in public release, catches none of them: recall is 0.000 on every cell of every confusion matrix, on a 1600-sample template baseline through OpenClaw's actual production command-line interface (CLI) and on a ten-LLM cross-model generalisation run. Detecting F1--F4 requires seven specific runtime structures absent from OpenClaw's source tree: a biconditional checker, a hash-chained audit log, an extension admission gate, a two-layer egress guard, a Bell-LaPadula classification policy, a module-signing trust root, and a bootstrap seal. enclawed-oss -- an MIT-licensed drop-in fork that ships all seven -- reaches $P = R = F_1 =$ accuracy $= 1.000$ on the same input. The gap is structural, not parametric: a six-line append-only widening of enclawed-oss's data-loss-prevention (DLP) regex catalog raises per-channel F3 detection by 14.6\% net at unchanged precision; the same edit on OpenClaw has nowhere to land. The harness deliberately exercises real Discord and Telegram channels -- plugin categories the first enclawed release deleted as unsafe -- to show F1--F4 detection extends to those previously-unsafe extensions. With architectural superiority for security and feature parity for extensions, we argue that unhardened agentic-AI runtimes are architecturally obsolete: a strictly better alternative exists, is adoptable today, and the gap requires re-architecture rather than configuration. We invite reviewers to apply the harness to any candidate runtime. 

---
# GEASS: Training-Free Caption Steering for Hallucination Mitigation in Vision-Language Models 

**Authors**: Zeshang Li, Shuoyang Zhang, Jiashen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.01733)  

**Abstract**: Vision-Language Models (VLMs) excel at grounded reasoning but remain prone to object hallucination. Recent work treats self-generated captions as a uniformly positive resource, yet we find that naively embedding one can degrade rather than help--dropping Qwen2.5-VL-3B accuracy on HallusionBench by nearly 10 points. Two structural properties explain this. First, captions anchor not only the model's final answer but also its reasoning trajectory and lexical choices. Second, caption errors are asymmetric: omissions vastly outnumber fabrications, yet each fabrication carries a much larger per-instance impact. A caption's usefulness is therefore a per-query property, not a per-corpus one. We propose GEASS (Gated Evidence-Aware Selective Steering), a training-free module that decides on each query how much of the caption the model consumes: it gates the caption by the clean path's confidence, weights it by the entropy reduction it produces, and raises the evidence bar when the two pathways disagree. Experiments on POPE and HallusionBench across four VLMs show that GEASS consistently improves over vanilla inference and contrastive decoding, with only two extra forward passes per query. 

---
# FEDIN: Frequency-Enhanced Deep Interest Network for Click-Through Rate Prediction 

**Authors**: Zenan Dai, Jinpeng Wang, Junwei Pan, Dapeng Liu, Lei Xiao, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2605.01726)  

**Abstract**: Sequential recommendation models often struggle to capture latent periodic patterns in user interests, primarily due to the noise inherent in time-domain behavioral data. While frequency-domain analysis offers a global perspective to address this, existing approaches typically treat user sequences in isolation, overlooking the crucial context of the target item. In this work, we present a novel empirical observation: user attention scores exhibit distinct spectral entropy distributions when conditioned on positive versus negative target items. Specifically, true user interests manifest as highly concentrated spectral patterns with lower entropy in the frequency domain, whereas irrelevant behaviors appear as high-entropy noise. Leveraging this insight, we propose the Frequency-Enhanced Deep Interest Network (FEDIN). FEDIN introduces a frequency-domain branch that utilizes a target-aware spectrum filtering mechanism to isolate these periodic interest signals. Extensive experiments on three public datasets demonstrate that FEDIN consistently outperforms state-of-the-art sequential recommendation baselines, demonstrating superior robustness against noise. We have released our code at: this https URL. 

---
# Motion-Aware Caching for Efficient Autoregressive Video Generation 

**Authors**: Jing Xu, Yuexiao Ma, Songwei Liu, Xuzhe Zheng, Shiwei Liu, Chenqian Yan, Xiawu Zheng, Rongrong Ji, Fei Chao, Xing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01725)  

**Abstract**: Autoregressive video generation paradigms offer theoretical promise for long video synthesis, yet their practical deployment is hindered by the computational burden of sequential iterative denoising. While cache reuse strategies can accelerate generation by skipping redundant denoising steps, existing methods rely on coarse-grained chunk-level skipping that fails to capture fine-grained pixel dynamics. This oversight is critical: pixels with high motion require more denoising steps to prevent error accumulation, while static pixels tolerate aggressive skipping. We formalize this insight theoretically by linking cache errors to residual instability, and propose MotionCache, a motion-aware cache framework that exploits inter-frame differences as a lightweight proxy for pixel-level motion characteristics. MotionCache employs a coarse-to-fine strategy: an initial warm-up phase establishes semantic coherence, followed by motion-weighted cache reuse that dynamically adjusts update frequencies per token. Extensive experiments on state-of-the-art models like SkyReels-V2 and MAGI-1 demonstrate that MotionCache achieves significant speedups of $\textbf{6.28}\times$ and $\textbf{1.64}\times$ respectively, while effectively preserving generation quality (VBench: $1\%\downarrow$ and $0.01\%\downarrow$ respectively). The code is available at this https URL. 

---
# SignVerse-2M: A Two-Million-Clip Pose-Native Universe of 25+ Sign Languages 

**Authors**: Sen Fang, Hongbin Zhong, Yanxin Zhang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2605.01720)  

**Abstract**: Existing large-scale sign language resources typically provide supervision only at the level of raw video-text alignment and are often produced in laboratory settings. While such resources are important for semantic understanding, they do not directly provide a unified interface for open-world recognition and translation, or for modern pose-driven sign language video generation frameworks: 1. RGB-based pretrained recognition models depend heavily on fixed backgrounds or clothing conditions during recording, and are less robust in open-world settings than style-agnostic pose-processing models. 2. Recent pose-guided image/video generation models mostly use a unified keypoint representation such as DWPose as their control interface. At present, the sign language field still lacks a data resource that can directly interface with this modern pose-native paradigm while also targeting real-world open scenarios. We present SignVerse-2M, a large-scale multilingual pose-native dataset for sign language pose modeling and evaluation. Built from publicly available multilingual sign language video resources, it applies DWPose in a unified preprocessing pipeline to convert raw videos into 2D pose sequences that can be used directly for modeling, resulting in a consolidated corpus of about two million clips covering more than 25 sign languages. Unlike many laboratory datasets, this resource preserves the recording conditions and speaker diversity of real-world videos while reducing appearance variation through a unified pose representation. Toward this goal, we further provide the data construction pipeline, task definitions, and a simple SignDW Transformer baseline, demonstrating the feasibility of this resource for multilingual pose-space modeling and its compatibility with modern pose-driven pipelines, while discussing the evaluation claims it can support as well as its current limitations. 

---
# TCDA: Thread-Constrained Discourse-Aware Modeling for Conversational Sentiment Quadruple Analysis 

**Authors**: Xinran Li, Xinze Che, Yifan Lyu, Zhiqi Huang, Xiujuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01717)  

**Abstract**: Conversational Aspect-based Sentiment Quadruple Analysis (DiaASQ) needs to capture the complex interrelationships in multiple rounds of dialogues. Existing methods usually employ simple Graph Convolutional Networks (GCN), which introduce structural noise and fail to consider the temporal sequence of the dialogues, or use standard RoPE, which implicitly captures relative distances in a flat sequence but cannot clearly separate the token-level syntactic order from the utterance-level progression, and may suffer from the Distance Dilution problem. To address these issues, we propose a new framework that combines Thread-Constrained Directed Acyclic Graph (TC-DAG) and Discourse-Aware Rotary Position Embedding (D-RoPE). Specifically, TC-DAG filters out cross-thread noise based on thread constraints, maintains global connectivity through root anchoring, and incorporates the temporal sequence of the dialogues. D-RoPE aligns multi-layer semantics using dual-stream projection and multi-scale frequency signals, captures thread dependencies using tree-like distances, and alleviates the token-level Distance Dilution problem by incorporating utterance-level progressions. Experimental results on two benchmark datasets demonstrate that our framework achieves state-of-the-art performance. 

---
# SplitZip: Ultra Fast Lossless KV Compression for Disaggregated LLM Serving 

**Authors**: Yipin Guo, Siddharth Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2605.01708)  

**Abstract**: Contemporary systems serving large language models (LLMs) have adopted prefill-decode disaggregation to better load-balance between the compute-bound prefill phase and the memory-bound decode phase. Under this design, prefill workers generate a KV cache that must be transferred to decode workers before token generation can begin. With these workers residing on different physical systems, this transfer becomes a significant bottleneck to serving LLMs at scale. This bottleneck gets exacerbated for long-input and agentic workloads, which typically require long inputs. Existing lossless codecs are not well suited to this setting as they primarily target offline weight compression, rely on CPU-side, or use variable-length coding that decompresses fast but compresses too slowly for online use. SplitZip is a GPU-friendly lossless compressor for KV-cache transfer. It exploits redundancy in floating-point exponents of KV activations, encoding the most frequent exponent values with fixed-length codes, and encoding (position, value) pairs and value of rare exponents in an escape stream. An offline calibrated top-16 exponent codebook enables online encoding, while the regular dense path and sparse escape correction make both encoding and decoding efficient on GPUs. On real BF16 activation tensors, SplitZip achieves 613.3 GB/s compression throughput and 2181.8 GB/s decompression throughput, substantially outperforming prior lossless compressors on the latency-critical codec path. End-to-end transfer experiments show up to 1.32$\times$ speedup for BF16 KV-cache transfer, 1.30$\times$ speedup for TTFT and 1.23$\times$ increase on Request Throughput. 

---
# Probe-Geometry Alignment: Erasing the Cross-Sequence Memorization Signature Below Chance 

**Authors**: Anamika Paul Rupa, Anietie Andy  

**Link**: [PDF](https://arxiv.org/pdf/2605.01699)  

**Abstract**: Recent attacks show that behavioural unlearning of large language models leaves internal traces recoverable by adversarial probes. We characterise where this retention lives and show it can be surgically removed without measurable capability cost. Our central protocol is a leave-one-out cross-sequence probe that tests whether a memorisation signature generalises across held-out sequences. The signature is real and consistent across scale: memorisation-specific gaps of +0.32, +0.19, +0.30 on Pythia-70M, GPT-2 medium, and Mistral-7B; on Pythia-70M, the random-initialisation control collapses to -0.04 at the deepest layer where the pretrained signature peaks. The probe direction is causally separable from recall -- projecting it out collapses the signature locally (+0.44 -> -0.19) while behavioural recall barely changes -- and a probe trained on naturally memorised content does not classify fine-tuning-injected secrets, marking two representationally distinct regimes. We then introduce probe-geometry alignment (PGA), a surgical erasure that aligns activations along the probe's live readout direction at each depth. PGA drives the cross-sequence probe below random chance at all four scales tested (toy depth-4: 0.17; Pythia-70M: 0.07; Mistral-7B: 0.45; GPT-2 medium: 0.06 via MD-PGA k=2) and remains robust to six adversarial probe variants. Against a re-fitting attacker who trains a fresh probe on PGA-treated activations, we extend PGA adversarially, defeating the re-fit probe at every memorisation-relevant depth while preserving five zero-shot capability benchmarks within 2.8 percentage points per task (mean {\Delta}acc = +0.2pp). The cross-sequence signature is a real, causally separable, regime-specific property of pretrained representations -- removable below chance with a single rank-one intervention per depth at no measurable capability cost. 

---
# BIM Information Extraction Through LLM-based Adaptive Exploration 

**Authors**: Sylvain Hellin, Suhyung Jang, Stefan Fuchs, Stavros Nousias, André Borrmann  

**Link**: [PDF](https://arxiv.org/pdf/2605.01698)  

**Abstract**: BIM models provide structured representations of building geometry, semantics, and topology, yet extracting specific information from them remains remarkably difficult. Current approaches translate natural language into structured queries by assuming a fixed data organization (static approach), which BIM heterogeneity eventually invalidates. We address this with a new paradigm, adaptive exploration, where an LLM-based agent iteratively executes code to extract information from a BIM model, discovering its structure at runtime instead of assuming it. We evaluate this approach on ifc-bench v2, an open-source BIM question-answering benchmark introduced alongside this work, comprising 1,027 tasks across 37 IFC models from 21 projects. A factorial ablation across two LLM capability levels and four augmentation strategies shows that adaptive exploration significantly outperforms static query generation across all configurations, regardless of the augmentation strategy. These results indicate that BIM heterogeneity is best addressed at the paradigm level, not by further optimizing static approaches. 

---
# GRAVITY: Architecture-Agnostic Structured Anchoring for Long-Horizon Conversational Memory 

**Authors**: Yushi Sun, Bowen Cao, Dong Fang, Lingfeng Su, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2605.01688)  

**Abstract**: Long-horizon conversational agents rely on memory systems with increasingly sophisticated retrieval mechanisms. However, retrieved fragments are typically fed to the language model as unstructured text, lacking the relational, temporal, and thematic structures essential for complex reasoning. To bridge this reasoning gap, we introduce GRAVITY (\textbf{G}eneration-time \textbf{R}elational \textbf{A}nchoring \textbf{V}ia \textbf{I}njected \textbf{T}opological Memor\textbf{Y}), a plug-and-play structured memory module. GRAVITY extracts three complementary knowledge representations from raw conversational utterances: entity profiles grounded in relational graphs, temporal event tuples linked into causal traces, and cross-session topic summaries. At generation time, it injects these representations into the host system's prompt as structured anchoring contexts. This approach effectively synthesizes scattered evidence into a coherent, query-relevant context without requiring any architectural modifications to the host model. Extensive evaluations across five diverse memory systems on the LongMemEval and LoCoMo benchmarks demonstrate the efficacy of our approach. On average, GRAVITY improves LLM-judge accuracy by 7.5--10.1%. Gains are inversely correlated with baseline strength: the weakest host improves by 12.2% while the strongest still gains 3.8--5.7%. These findings establish structured context anchoring as a broadly effective, architecture-agnostic augmentation paradigm for long-horizon conversational memory. 

---
# Class-Aware Adaptive Differential Privacy in Deep Learning for Sensor-Based Fall Detection 

**Authors**: Joydeb Kumar Sana  

**Link**: [PDF](https://arxiv.org/pdf/2605.01679)  

**Abstract**: Fall detection is a critical task in healthcare, particularly for elderly people. Timely fall detection and treatment can prevent severe injuries. Sensor-based activity data can be used to detect fall. However, this data are highly sensitive and raises significant privacy concerns. Existing privacy approaches apply uniform noise across all training samples, which affects the prediction performance. To address this limitation, we propose a Class-Aware Adaptive Differential Privacy (CA-ADP) framework integrated with a hybrid 3D Convolutional Neural Network and Bidirectional Long Short-Term Memory (3D CNN-BiLSTM) architecture. The CA-ADP mechanism dynamically adjusts the magnitude of noise added to gradients based on the class composition of each mini-batch. This process ensures privacy while mitigates performance degradation. We formally analyze the $(\epsilon,\delta)$-Differential Privacy guarantee and provide a privacy-utility trade-off analysis. The proposed method is evaluated on three public benchmark datasets, namely SisFall, UP-Fall, and MobiAct. The experimental results show that the proposed privacy model achieves improvements of 3.3\%, 8.5\%, and 7.5\% over the conventional privacy-based model in terms of F-score for the SisFall, UP-Fall, and MobiAct datasets, respectively. Comparisons with prior studies show that the CA-AD based framework achieves competitive performance and provides formal privacy guarantees, which are largely overlooked in existing studies. Wilcoxon signed-rank tests confirm that the proposed mechanism consistently outperforms conventional differential privacy. Those results establish the proposed CA-ADP framework as an effective approach to privacy-preserving fall detection in real-world healthcare settings. 

---
# Missingness-aware Data Imputation via AI-powered Bayesian Generative Modeling 

**Authors**: Qiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01676)  

**Abstract**: Missing data imputation remains a fundamental challenge in modern data science, especially when uncertainty quantification is essential. In this work, we propose MissBGM, an AI-powered missing data imputation method via Bayesian generative modeling that bridges the expressive flexibility of neural networks with the statistical rigor of Bayesian inference. Unlike existing methods that often focus on point estimates or treat the missingness mechanism implicitly, MissBGM explicitly and jointly models the data-generating and missingness mechanisms, providing principled posterior uncertainty over imputations rather than a single point estimate. We develop a stochastic optimization framework with alternating updates among missing values, model parameters, and latent variables until convergence. Our theoretical analysis shows that estimates of missing values from MissBGM converge consistently under mild assumptions. Empirically, we demonstrate that MissBGM achieves superior performance over traditional imputers and recent neural network-based methods across extensive experimental settings. These results establish MissBGM as a principled and scalable solution for modern missing data imputation. The code for MissBGM is open sourced at this https URL. 

---
# IMPACT-Scribe: Interactive Temporal Action Segmentation with Boundary Scribbles and Query Planning 

**Authors**: Qian Yin, Di Wen, Kunyu Peng, David Schneider, Zeyun Zhong, Alexander Jaus, Zdravko Marinov, Jiale Wei, Ruiping Liu, Junwei Zheng, Yufan Chen, Chen Zhang, Lei Qi, Rainer Stiefelhagen  

**Link**: [PDF](https://arxiv.org/pdf/2605.01668)  

**Abstract**: Dense temporal annotation of procedural activity videos is vital for action understanding and embodied intelligence but remains labor-intensive due to reactive tools. Each correction is treated as an isolated edit, limiting reuse of information on annotator uncertainty and model reliability. We introduce IMPACT-Scribe, a correction-driven framework for dense labeling that uses each correction to improve future human-machine collaboration. IMPACT-Scribe combines uncertainty-aware boundary scribble supervision, local proposal modeling, cost-aware query planning, structured propagation, and correction-driven adaptation. Experiments and a human study show that this closed-loop design improves labeling quality per effort, enhances boundary accuracy, and fosters better human-machine interaction over time. The code will be made publicly available at this https URL. 

---
# IMPACT-HOI: Supervisory Control for Onset-Anchored Partial HOI Event Construction 

**Authors**: Haoshen Zhang, Di Wen, Kunyu Peng, David Schneider, Zeyun Zhong, Alexander Jaus, Zdravko Marinov, Jiale Wei, Ruiping Liu, Junwei Zheng, Yufan Chen, Yufeng Zhang, Yuanhao Luo, Lei Qi, Rainer Stiefelhagen  

**Link**: [PDF](https://arxiv.org/pdf/2605.01666)  

**Abstract**: We present IMPACT-HOI, a mixed-initiative framework for annotating egocentric procedural video by constructing structured event graphs for Human-Object Interactions (HOI), motivated by the need for high-quality structured supervision for learning robot manipulation from human demonstration. IMPACT-HOI frames this task as the incremental resolution of a partially specified, onset-anchored event state. A trust-calibrated controller selects among direct queries, human-confirmed suggestions, and conservative completions based on empirical annotator behavior and evidence quality. A risk-bounded execution protocol, utilizing atomic rollback, ensures that human-confirmed decisions are preserved against conflicting automated updates. A user study with 9 participants shows a 13.5% reduction in manual annotation actions, a 46.67% event match rate, and zero confirmed-field violations under the studied protocol. The code will be made publicly available at this https URL. 

---
# TRIMMER: A New Paradigm for Video Summarization through Self-Supervised Reinforcement Learning 

**Authors**: Pritam Mishra, Coloma Ballester, Dimosthenis Karatzas  

**Link**: [PDF](https://arxiv.org/pdf/2605.01659)  

**Abstract**: The rapid growth of video content across domains such as surveillance, education, and social media has made efficient content understanding increasingly critical. Video summarization addresses this challenge by generating concise yet semantically meaningful representations, but existing approaches often rely on expensive manual annotations, struggle to generalize across domains, and incur significant computational costs due to complex architectures. Moreover, unsupervised and weakly supervised methods typically underperform compared to supervised counterparts in capturing long-range temporal dependencies and semantic structure. In this work, we propose TRIMMER (Temporal Relative Information Maximization for Multi-objective Efficient Reinforcement), a novel self-supervised reinforcement learning framework for video summarization. TRIMMER operates in two stages: it first learns robust representations via self-supervised learning and then performs spatio-temporal decision making through reinforcement learning guided by information-theoretic reward functions. Unlike prior approaches that rely on similarity-based objectives, our method introduces entropy-based metrics to capture higher-order temporal dynamics and semantic diversity, while computing rewards directly over selected frame indices to improve computational efficiency. Extensive experiments on standard benchmarks demonstrate that TRIMMER achieves state-of-the-art performance among unsupervised and self-supervised methods, while remaining competitive with leading supervised approaches, highlighting its effectiveness for scalable and generalizable video summarization. 

---
# From Cortical Synchronous Rhythm to Brain Inspired Learning Mechanism: An Oscillatory Spiking Neural Network with Time-Delayed Coordination 

**Authors**: Tingting Dan, Guorong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01656)  

**Abstract**: Human cognition emerges from coordinated spiking dynamics in distributed neural circuits, where information is encoded via both firing rates and precise spike timing determined by brain rhythms. Inspired by this notion, we propose a brain-inspired learning primitive in which cognition-level neural synchrony emerges through iterative bottom-up and top-down interactions between micro-scale dynamics of spiking neurons and a macro-scale mechanism of oscillatory synchronization. Specifically, we model each parcel (e.g., a cortical region or an image pixel) in the target system as a spiking neuron embedded in a predefined connectivity scaffold. Low-level information is encoded in a spatiotemporal domain, where neurons are selectively grouped and fire spontaneously over time through self-organized dynamics. In the bottom-up route, oscillatory synchronization is formed from past spiking activity accumulated over a finite memory window. Since brain dynamics operate in a regime of partial and transient synchronization rather than global phase locking, we model oscillatory coordination using a time-delayed synchronization formulation, which enables a top-down modulation of heterogeneous neural spiking for a large-scale distributed system. Together, we devise a spiking-by-synchronization neural network (S2-Net) that uses rhythmic timing as a control mechanism for efficient information processing. Promising results have been achieved across a broad range of tasks, including neural activity decoding, energy-efficient signal processing, temporal binding and semantic reasoning. 

---
# AI Alignment via Incentives and Correction 

**Authors**: Rohit Agarwal, Joshua Lin, Mark Braverman, Elad Hazan  

**Link**: [PDF](https://arxiv.org/pdf/2605.01643)  

**Abstract**: We study AI alignment through the lens of law-and-economics models of deterrence and enforcement. In these models, misconduct is not treated as an external failure, but as a strategic response to incentives: an actor weighs the gain from violation against the probability of detection and the severity of punishment. We argue that the same logic arises naturally in agentic AI pipelines. A solver may benefit from producing a persuasive but incorrect answer, hiding uncertainty, or exploiting spurious shortcuts, while an auditor or verifier must decide whether costly monitoring is worthwhile. Alignment is therefore a fixed-point problem: stronger penalties may deter solver misbehavior, but they can also reduce the auditor's incentive to inspect, since auditing then mainly incurs cost on a population that appears increasingly aligned.
This perspective also changes what should count as a post-training signal. Standard feedback often attaches reward to the final answer alone, but a solver-auditor pipeline exposes the full correction event: whether the solver erred, whether the auditor inspected, whether the error was caught, and whether oversight incentives remained active. We formalize this interaction in a two-agent model in which a principal chooses rewards over joint correction outcomes, inducing both solver behavior and auditor monitoring. Reward design is therefore a bilevel optimization problem: rewards are judged not by their immediate semantic meaning, but by the behavioral equilibrium they induce. We propose a bandit-based outer-loop procedure for searching over reward profiles using noisy interaction feedback. Experiments on an LLM coding pipeline show that adaptive reward profiles can maintain useful oversight pressure and improve principal-aligned outcomes relative to static hand-designed rewards, including a substantial reduction in hallucinated incorrect attempts. 

---
# Prosa: Rubric-Based Evaluation of LLMs on Real User Chats in Brazilian Portuguese 

**Authors**: Roseval Malaquias Junior, Giovana Kerche Bonás, Thales Sales Almeida, Hugo Abonizio, Thiago Laitz, Ramon Pires, Marcos Piau, Celio Larcher, Rodrigo Nogueira  

**Link**: [PDF](https://arxiv.org/pdf/2605.01630)  

**Abstract**: Rankings produced by holistic LLM-as-a-judge scoring are sensitive to the bias of the chosen judge model. We show that switching to binary rubric scoring with multi-judge filtering removes this sensitivity: decomposing the judgement matters more than the judge model itself. To support this claim, we introduce Prosa, the first real user multi-turn Brazilian Portuguese chat benchmark: 1,000 WildChat conversations scored by three judges from three model families on 16 models. Under filtered rubric scoring the three judges agree on every one of the 16 ranks, whereas under holistic scoring they agree on only 7 of 16. Additionally, the rubric filtering pipeline increases the average score gap between neighbouring models by 47%, thereby improving Prosa's discriminative power. Evaluating a new model on Prosa costs approximately $2.1 when using Gemini 3 Flash as the judge. We release the benchmark and the filtering code to ensure that future models can be assessed under identical conditions. These artifacts also make our rubric-based scoring method reusable beyond Prosa, supporting other open-ended evaluation settings. 

---
# From Packets to Patterns: Interpreting Encrypted Network Traffic as Longitudinal Behavioral Signals 

**Authors**: Rameen Mahmood, Omar El Shahawy, Souptik Barua, Zachary Beattie, Jeffrey Kaye, Xuhai "Orson'' Xu, Danny Yuxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01616)  

**Abstract**: Human behavior is difficult to observe continuously at scale, yet it leaves measurable traces in everyday device use. We test whether encrypted smartphone network traffic -- a ubiquitous, always-on, passive sensing modality -- can passively capture behavioral patterns related to sleep, stress, and loneliness. We model shared behavioral structure using a transformer backbone with per-user adapters, allowing the model to represent both typical individual behavior and deviations from it. To make these representations interpretable, we apply a sparse autoencoder to extract behavioral features corresponding to distinct patterns of activity. We relate these features to sleep disturbance, stress, and loneliness using generalized estimating equations with Mundlak decomposition, separating between-person differences from within-person changes over time. We find that the three outcomes reflect distinct temporal structures: stress is primarily associated with stable between-person differences, loneliness with within-person variation, and sleep disturbance with a combination of both. Notably, these within-person dynamics are not captured by predefined network-traffic features, demonstrating the value of learned representations for longitudinal behavioral sensing. These results establish encrypted network traffic as a viable passive sensing modality, revealing interpretable behavioral dynamics -- particularly deviations from an individual's baseline -- that are not visible in raw traffic features. 

---
# The Case for ESM3 as a General-Purpose AI Model with Systemic Risk Under the EU AI Act 

**Authors**: Taro Qureshi, Jacob Griffith, Koen Holtman, Marcel Mir Teijeiro, Ze Shen Chin, Rokas Gipiškis  

**Link**: [PDF](https://arxiv.org/pdf/2605.01611)  

**Abstract**: Due to ambiguity in the wording of the EU AI Act, we examine the question of to what extent frontier biological foundation models such as ESM3 are subject to obligations for general-purpose AI models with systemic risk under the EU AI Act. In this paper, we map ESM3 to the biorisk chain, and conclude that it would be desirable if the providers of ESM3 and similar biological models were subject to these obligations, which would require them to assess and mitigate dual-use risks from their models. We then perform an analysis, comparing the attributes of ESM3 to the classification criteria in the AI Act and the supporting material. We conclude that at this time, ESM3 does not appear to be meaningfully regulated by the Act. We then propose remedies to correct the situation. 

---
# Less Interaction But More Explanation: A Communication Perspective on Agentic AI Interfaces 

**Authors**: Eunchae Jang, S. Shyam Sundar  

**Link**: [PDF](https://arxiv.org/pdf/2605.01610)  

**Abstract**: AI systems have long been expected to interact with users, answering questions, generating content, and continuing (social) conversations. Agentic AI, however, breaks from this expectation, as its primary objective is workflow execution on behalf of the users. If a system becomes more agentic, do users need less interaction with the system? Our answer is: less routine back-and-forth, but more communication for oversight and explanation, as agentic AI proactively acts, not just responds. Grounded in a communication perspective, we discuss how users perceive the communicative roles of AI systems (whether as the source of actions or merely a channel), and how this can shape trust. Because agentic AI can play multiple communicative roles, it can complicate this source perception and introduce potential risks. To address this, we propose three types of explanations that agentic AI needs to incorporate (action-process, uncertainty, and coordination), and suggest that customization affordances that allow users to decide when and which explanations they see may be key to preserving human agency as AI autonomy increases. 

---
# Concepts Whisper While Syntax Shouts: Spectral Anti-Concentration and the Dual Geometry of Transformer Representations 

**Authors**: Pratyush Acharya, Nuraj Rimal, Habish Dhakal  

**Link**: [PDF](https://arxiv.org/pdf/2605.01609)  

**Abstract**: We test whether the causal inner product of \citet{park2024linear} -- defined by the unembedding covariance $\Sigma$ -- enables cross-lingual concept transport. Across 17 models and 4 language pairs, a matched-spectrum randomization test finds that Whitened Causal Alignment is indistinguishable from spectral regularization alone ($p = 0.95$). However, this failure reveals a broader phenomenon: anti-concentration is observed in residual-stream difference-of-means vectors across five architecture families ($p < 10^{-33}$) and supported by SAE features (e.g., $p = 4.5 \times 10^{-19}$) and linear probes on Gemma and Llama. We discover a \emph{dual geometry}: activation-space concept directions anti-concentrate in the spectral tail, while static unembedding-row contrasts \emph{concentrate} in high-variance directions ($p < 10^{-4}$). Split-injection causal interventions support the functional basis on Gemma and Llama (Cohen's $d$ up to $1.80$), and POS-tag probing across 8 models shows syntax preferentially encodes in the high-variance subspace in 6 of 8 architectures ($p < 0.013$), with the Qwen~2.5 family showing a significant reversal consistent with architecture-specific spectral structure. These results suggest transformers may rotate semantic content into spectrally quiet regions during contextualized processing, encoding concepts where they can be manipulated with reduced grammatical disruption. 

---
# Where Do Prompt Perturbations Break Generation? A Segment-Level View of Robustness in LoRA-Tuned Language Models 

**Authors**: Zhuoyun Li, Boxuan Wang, Jinwei Hu, Zhenglin Huang, Qisong He, Xinmiao Huang, Guangliang Cheng, Xiaowei Huang, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.01605)  

**Abstract**: Large language models are sensitive to minor prompt perturbations, yet existing robustness methods usually enforce consistency at the whole-sequence level. This holistic view can hide an important failure mode: a perturbed response may remain globally similar to the clean one while drifting on a critical entity, relation, or conclusion. We introduce S$^2$R$^2$, a segment-level framework for robust LoRA fine-tuning. S$^2$R$^2$ decomposes clean and perturbed generations into semantic segments, aligns them with an optimal-transport objective, and penalises the segments with the largest meaning drift. To connect this output-side objective with model adaptation, we add an adapter-stability regulariser motivated by segment-level attention reallocation, using LoRA norm control as a tractable proxy for limiting perturbation-amplified evidence shifts. A PAC-Bayesian complexity view further explains why controlling adapter growth may support transfer beyond observed perturbations. Experiments on summarisation benchmarks show that S$^2$R$^2$ improves robustness under typographical noise, deletion, synonym replacement, and paraphrasing, while maintaining competitive clean performance and stronger cross-dataset transfer than consistency-based baselines. 

---
# KG-First, LLM-Fallback: A Hybrid Microservice for Grounded Skill Search and Explanation 

**Authors**: Ngoc Luyen Le, Marie-Hélène Abel, Bertrand Laforge  

**Link**: [PDF](https://arxiv.org/pdf/2605.01582)  

**Abstract**: Authoritative competency frameworks such as ESCO, ROME, and O*NET are essential for aligning education with labor market needs, yet their technical complexity and structural heterogeneity hinder practical adoption by educators. This paper introduces SkillGraph-Service, an interoperable microservice designed to bridge this gap by unifying these resources into a provenance-preserving Knowledge Graph (KG). Adopting a KG-first, LLM-fallback architecture, the system combines symbolic rigor with sub-symbolic flexibility. It implements a lightweight hybrid retrieval engine (fusing SQLite FTS5 and HNSW vector search) to handle the vocabulary mismatch in educator queries, and utilizes Large Language Models (LLMs) strictly for constrained ranking and audience-aware explanation. Empirical evaluation on a multilingual dataset reveals that the proposed hybrid strategy achieves superior retrieval effectiveness (nDCG@5>0.94) with sub-200 ms latency, rendering computationally expensive cross-encoder re-ranking may be unnecessary for this domain. Furthermore, an analysis of generated explanations highlights a trade-off between fluency and faithfulness: while JSON-constrained LLMs ensure high citation precision, deterministic templates remain the most reliable method for maximizing evidence coverage. The resulting architecture offers a practical, scalable, and auditable solution for integrating complex skill data into digital learning ecosystems. 

---
# Model Merging: Foundations and Algorithms 

**Authors**: Donato Crisostomi  

**Link**: [PDF](https://arxiv.org/pdf/2605.01580)  

**Abstract**: Modern deep learning usually treats models as separate artifacts: trained independently, specialized for particular purposes, and replaced when improved versions appear. This thesis studies model merging as an alternative paradigm: combining independently trained neural networks directly in weight space, with little or no optimization and without requiring access to the original training data.
The thesis considers two main regimes. In the single-task setting, where models share an objective but differ in initialization, we introduce C$^2$M$^3$, a cycle-consistent merging algorithm based on Frank-Wolfe optimization. C$^2$M$^3$ aligns multiple networks into a shared, reference-free parameter space, making weight averaging meaningful without privileging any individual model.
In the multi-task setting, where models are fine-tuned for different downstream tasks from a common pretrained initialization, we first develop a theoretical account of task vectors as approximate gradients. This explains both the effectiveness and the limitations of task arithmetic. Building on this view, we show that task vectors inherit the low-rank structure of gradients and introduce Task Singular Vectors (TSV), a decomposition that enables compression and interference reduction through TSV-Merge. We then present MASS, an input-adaptive routing method that uses TSV geometry to select task-relevant subspaces at inference time. Finally, we introduce MERGE$^3$, an evolutionary merging framework that uses Item Response Theory to reduce evaluation costs by up to 50$\times$ while preserving solution quality.
Together, these contributions provide theoretical and algorithmic foundations for model merging, supporting a paradigm in which learned capabilities can be composed, reused, and extended across models. 

---
# Neuro-Symbolic Agents for Hallucination-Free Requirements Reuse 

**Authors**: Ahmed Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2605.01562)  

**Abstract**: The Object-Oriented Method for Requirements Authoring and Management (OOMRAM) is a requirements reuse framework that relies on exact identifier matching and rigid templates, limiting its ability to adapt specifications across diverse contexts. While Large Language Models (LLMs) offer the flexibility to overcome this bottleneck, they introduce the risk of generating structurally invalid or inconsistent requirement combinations. To address this tension, we present a neuro-symbolic multi-agent system that re-conceptualizes requirements reuse as a \textbf{Model-Driven Elicitation process}. In this paradigm, an LLM serves as a \textbf{non-deterministic heuristic} for traversing a \textbf{deterministic domain model} represented by a formal OOMRAM requirement lattice. A deterministic, symbolic validator enforces all structural constraints within the agent loop, effectively eliminating hallucinated requirement combinations by construction. Evaluated on an autonomous benchmark across two application families, our system achieves 100\% requirement coverage and a constraint-violation rate of only 0.2\%. Although the F1-score against a single gold standard is moderate (0.47--0.51), every generated specification is structurally valid and satisfies all mandatory domain requirements. The model-agnostic implementation scales to larger lattices via subgraph navigation and provides transparent audit trails for regulatory compliance. 

---
# Automated Interpretability and Feature Discovery in Language Models with Agents 

**Authors**: Arnau Marin-Llobet, Javier Ferrando  

**Link**: [PDF](https://arxiv.org/pdf/2605.01555)  

**Abstract**: We introduce an autonomous multiagent framework for mechanistic interpretability that automates both explaining and finding internal features in large language models. The system runs two coupled loops: (1) explanation refinement, where an agent proposes competing hypotheses and iteratively tests them with targeted prompt controls and a multi-metric evaluation; and (2) feature discovery, where an agent generates prompt sets, constructs a k-nearest-neighbor graph in activation space, and retrieves candidate features using statistical separability and semantic coherence criteria. On Gemma-2 family models and MLP neurons in weight-sparse transformers, our agent improves over one-shot auto-interpretations, discovers language-specific and safety-relevant features, and produces auditable explanation traces, showing that agent-driven empirical loops yield sharper and more falsifiable explanations than one-shot labels. 

---
# 6G Needs Agents: Toward Agentic AI-Native Networks for Autonomous Intelligence 

**Authors**: Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2605.01546)  

**Abstract**: Sixth-generation (6G) networks are increasingly envisioned as AI-native infrastructures integrating communication, sensing, and computing into a unified fabric. However, existing approaches remain largely optimization-centric, relying on closed-loop control with limited reasoning capability. In this paper, we argue for a paradigm shift toward Agentic AI-Native 6G, in which Large Language Model (LLM)-based agents operate as bounded, policy-governed reasoning entities within a semantic control plane layered above deterministic 3GPP infrastructure. We propose a four-layer architecture that integrates deterministic network infrastructure, semantic abstraction of intent and context, hierarchical reasoning, and a distributed multi-agent fabric spanning device, edge, and core domains. To assess feasibility, we develop a proof-of-concept agentic reasoning and orchestration framework and conduct an extensive empirical study using a domain-specific 6G benchmark under realistic deployment constraints. Our results reveal a fundamental tradeoff between reasoning capability and system efficiency, showing that no single model simultaneously satisfies latency, throughput, and accuracy requirements. Instead, heterogeneous deployment of LLM agents across the device--edge--core continuum is necessary to balance these constraints. We further demonstrate that quantization introduces non-uniform effects across models, reinforcing the need for system-level optimization rather than model-level compression alone. These findings establish agentic intelligence as a viable architectural direction for 6G and highlight key challenges in achieving scalable, trustworthy, and self-reasoning networks. All experimental results and evaluation scripts are publicly available to support reproducibility. 

---
# Mesh Based Simulations with Spatial and Temporal awareness 

**Authors**: Paul Garnier, Vincent Lannelongue, Elie Hachem  

**Link**: [PDF](https://arxiv.org/pdf/2605.01542)  

**Abstract**: Machine Learning surrogates for Computational Fluid Dynamics (CFD), particularly Graph Neural Networks (GNNs) and Transformers, have become a new important approach for accelerating physics simulations. However, we identify a critical bottleneck in the field: while architectures have advanced significantly, the common underlying training paradigms remain bound to naive assumptions, such as node-wise supervision and explicit Euler time-stepping. These legacy choices ignore the stiff dynamics and local flux continuity inherent to numerous partial differential equations resolution methods, such as Finite Element, Difference, or Volume (FEM). In this work, we propose a unified framework to bridge the gap between geometric deep learning and rigorous numerical analysis. We introduce three key innovations: (1) Multi Node Prediction, a stencil-level objective that predicts field values for a node's full local topology, enforcing spatial derivative consistency; (2) Temporal Correction, replacing unstable explicit schemes with a predictor-corrector via temporal Cross-Attention; and (3) Geometric Inductive Biases, leveraging 3D Rotary Positional Embeddings (RoPE) to robustly capture rotational symmetries in unstructured meshes. We evaluate this framework across three architectures (MeshGraphNet, Transolver, and a Transformer) on diverse physics datasets. Our approach yields consistent improvements in accuracy and stability, particularly in long-horizon rollouts, while producing latent representations that generalize to unseen subtasks such as Wall Shear Stress or Pressure prediction. Code is available at this https URL. 

---
# Protein-Conditioned Multi-Objective Reinforcement Learning for Full-Length mRNA Design 

**Authors**: Zixi Shao, Tao Wang, Yibei Xiao, Tianyi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01513)  

**Abstract**: Designing therapeutic messenger RNA (mRNA) requires creating full-length transcripts that carefully balance stability, translation efficiency, and immune safety. To address this challenge, we propose ProMORNA, a multi-objective generation framework that produces complete mRNA transcripts \textit{de novo} directly from a target protein sequence. Our approach begins by training a BART-style encoder-decoder model on over 6 million natural protein-mRNA pairs. We then introduce Multi-Objective Group Relative Policy Optimization (MO-GRPO) to simultaneously optimize for various biological objectives in a unified way. As a case study, we evaluated ProMORNA on the widely used firefly luciferase target, excluding it from both our supervised training data and the prompt pool. The results indicate that ProMORNA improves the \textit{in silico} Pareto frontier for predicted half-life and translation efficiency relative to standard supervised baselines. Additionally, it achieves higher predicted functional scores than a state-of-the-art baseline under the same evaluation pipeline. These computational findings demonstrate the feasibility of using multi-objective reinforcement learning for full-length mRNA design on unseen targets. 

---
# FT-RAG: A Fine-grained Retrieval-Augmented Generation Framework for Complex Table Reasoning 

**Authors**: Zebin Guo, Weidong Geng, Ruichen Mao  

**Link**: [PDF](https://arxiv.org/pdf/2605.01495)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding responses in external knowledge during inference. However, conventiona RAG systems under-perform on structured tabular data, largely due to coarse retrieval granularity and insufficient table semantic comprehension. To address these limitations, we introduce FT-RAG, a fine-grained framework that employs knowledge association by decomposing tables into entry-level semantic units to construct a structured graph. FT-RAG employs a structural neighbor expansion mechanism to find semantically connected entities during graph retrieval, followed by multi-modal fusion to consolidate the context of table retrieval results. Further, to address the scarcity of specialized datasets in this domain, we introduce Multi-Table-RAG-Lib, a benchmark comprising 9870 QA pairs with high complexity and difficulty, curated to demand multi-table integration and text-table information fusion for reasoning. FT-RAG surpasses top-performing baselines across all metrics, achieving a 23.5\% and 59.2\% improvement in table-level and cell-level Hit Rates, respectively. Generation performance also sees a remarkable 62.2\% increase in exact value accuracy recall. These metrics verify the framework's effectiveness in factual grounding across both pure tabular and heterogeneous table-text contexts. Therefore, our method establishes a new state-of-the-art performance for complex reasoning over mixed-modality documents. 

---
# CGFformer: Cluster-Guidance Frequency Transformer for Pansharpening 

**Authors**: Zijian Zhou, Jianing Zhang, Kai Sun, Xiangyu Zhao, Chunxia Zhang, Xiangyong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.01490)  

**Abstract**: Pansharpening aims to generate high-resolution multispectral (HRMS) images by fusing low-resolution multispectral (LRMS) images with high-resolution panchromatic (PAN) images. However, the current mainstream frequency-based pansharpening methods employ fixed frequency filters, which cannot precisely adapt to complex and spatially diversified frequency distributions in PAN and MS images. Furthermore, existing denoising strategies insufficiently exploit frequency components for denoising and struggle to suppress various noise types accurately. To address these challenges, we propose CGFformer, a cluster-guidance frequency Transformer that focuses on varying frequency distribution and interactions between frequency and spatial components. Specifically, we design an adaptive separation module that integrates local features and non-local information through K-means clustering, enabling more precise separation of high- and low-frequency components. Subsequently, we introduce a dual-stream refinement module combined with Transformer-based cross-attention to remove various noise, allowing the network to jointly suppress frequency-relevant and irrelevant disturbances. In addition, we develop a frequency-spatial fusion module designed to enhance detail and facilitate spatial-frequency interaction, ensuring more effective reconstruction of spatial structures in the fused results. Extensive experiments on multiple benchmark datasets demonstrate that the proposed CGFformer achieves notable improvements over existing pansharpening approaches. 

---
# Research on Vision-Language Question Answering Models for Industrial Robots 

**Authors**: Ping Li, Bartlomiej Brzozka  

**Link**: [PDF](https://arxiv.org/pdf/2605.01483)  

**Abstract**: A hierarchical cross-modal fusion model is proposed for vision-language question answering (VLQA) in industrial robotics, targeting the challenges of semantic ambiguity, complex environmental layouts, and domain-specific terminology common in modern manufacturing. The framework integrates advanced object detection, multi-scale visual encoding, syntactic parsing, and task-aware semantic attention to unite vision and language signals into a joint reasoning space. Region-based deep networks extract visual features, weighted embeddings aggregate, and recurrent neural parsing encodes sentence structures. Through fine-grained semantic alignment driven by adaptive fusion and cross-attention mechanisms, the system can handle operational queries, instruction steps, and anomaly detection with higher reliability. Compared to the existing VLQA benchmarks, validation experiments conducted on the IVQA and RIF benchmarks indicate improvements in semantic alignment, Top-1 accuracy, and robustness to ambiguous or procedural task queries. Ablation studies further quantify the impact of each architectural module, confirming the necessity of multi-level feature integration and context-driven gating for dependable industrial deployment. The technical advancements reported here provide core methodologies to improve the interpretability and operational effectiveness of industrial robots faced with diverse human-robot interaction tasks. 

---
# LIE: LiDAR-only HD Map Construction with Intensity Enhancement via Online Knowledge Distillation 

**Authors**: Kanak Mazumder, Fabian B. Flohr  

**Link**: [PDF](https://arxiv.org/pdf/2605.01478)  

**Abstract**: Online High-Definition (HD) map construction is a key component of autonomous driving. Recent methods rely on multi-view camera images for cost-effective HD map segmentation, but cameras lack depth information for accurate scene geometry. In contrast, LiDAR provides precise 3D measurements but lacks dense semantic cues. In this work, we propose LIE, LiDAR-only semantic map construction method that employ Knowledge Distillation (KD) to handle the lack of dense semantic and texture cues. Specifically, the teacher branch fuses student LiDAR features and the corresponding 2D intensity map tile to provide dense supervision for segmenting map elements using online distillation scheme. Experimental results show that our method outperforms all single-modality approaches, achieving 8.2% higher mIoU than the state-of-the-art camera-based model on nuScenes. LIE is robust over long ranges and under challenging weather and lighting, and efficiently adapts to Argoverse2 with only 10% fine-tuning, surpassing camera-based models trained on the full dataset. Source code will be available \href{this https URL}{here}. 

---
# Practical Limits of Autonomous Test Repair: A Multi-Agent Case Study with LLM-Driven Discovery and Self-Correction 

**Authors**: Hyukjoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.01471)  

**Abstract**: Maintaining reliable UI test suites in large-scale enterprise applications is a persistent and costly challenge. We present an industrial case study of a multi-agent autonomous testing system evaluated using anonymized execution data from a production-like enterprise UI testing prototype. The application features several hundred dynamic UI elements per screen. Built on a large language model with LangGraph orchestration, Playwright execution, and a RAG knowledge base, the system evolves from human-directed testing toward High-autonomy feature discovery and test execution: given no explicit test targets, it discovers over 100 testable features across 10 UI screens, dynamically expands coverage by an additional 15--30 features through runtime DOM analysis, and iteratively repairs failing tests without human intervention.
We analyzed 300 consecutive autonomous execution reports encompassing 636 individual test-case executions across 10 distinct scenario families. The system achieved a 70% repair convergence rate at the scenario-family level, with a mean of 3.4 repair iterations to convergence. However, only 10% of scenario families succeeded on first attempt, 38% of reports failed to produce any executable test artifact, and we documented concrete instances of assertion weakening and test-case deletion used as workaround mechanisms to achieve superficial convergence.
Our findings show that unrestricted autonomy leads to unstable and often misleading outcomes, while constrained autonomy transforms such systems into operationally viable workflows. Rather than advocating full autonomy, our findings suggest that reliable autonomous testing in enterprise-scale settings requires explicit constraints, validation boundaries, and human oversight to preserve semantic correctness and operational trustworthiness. 

---
# Decision Boundary-aware Generation for Long-tailed Learning 

**Authors**: Jiacheng Yang, Ruichi Zhang, Chikai Shang, Mengke Li, Xinyi Shang, Junlong Gao, Yonggang Zhang, Yang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01468)  

**Abstract**: Long-tailed data bias decision boundaries toward head classes and degrade tail class accuracy. Diffusion-based generative augmentation address this problem by generating additional data, while head-to-tail transfer further mitigate the generator bias inherit from long-tailed dataset. However, we show that while head-to-tail transfer helps balance the decision space of the classifier, it also induces latent non-local feature mixing that entangles inter-class features, causing decision boundary overlap and tail class distribution shift. To address this, we first identify the problem of boundary ambiguity and then propose Decision Boundary-aware Generation (DBG) framework, which promotes near-boundary representation learning by generating informative near-boundary samples. Overall, DBG rebalances the long-tailed dataset while yielding more separable decision space for long-tailed learning. Across standard long-tailed benchmarks, DBG consistently improves tail class and overall accuracy with less inter-class overlap. The code of DBG is available at this https URL. 

---
# SRGAN-CKAN: Expressive Super-Resolution with Nonlinear Functional Operators under Minimal Resources 

**Authors**: Roberto Isai Navaro-Aviña, Eduardo Said Merin-Martinez, Andres Mendez-Vazquez, Eduardo Rodriguez-Tello  

**Link**: [PDF](https://arxiv.org/pdf/2605.01459)  

**Abstract**: Single-Image Super-Resolution (SISR) aims to reconstruct a High-Resolution (HR) image from a Low-Resolution (LR) observation, a fundamentally ill-posed problem where high-frequency details are severely degraded at large upscaling factors. Recent advances have been driven by transformer-based architectures and diffusion models improve global context modeling and perceptual quality at the cost of increased computational complexity. In contrast, this work focuses on enhancing the expressivity of local operators under minimal resources. We propose SRGAN--CKAN, a hybrid super-resolution framework that integrates Convolutional Kolmogorov--Arnold Networks (CKAN) into an adversarial learning setting reformulating convolution as a nonlinear patch-based transformation. The proposed operator replaces linear local mappings with spline-based functional representations, allowing expressive modeling of complex local structures and high-frequency textures using minimal hardware resources. Experimental results demonstrate that the proposed approach improves perceptual quality while preserving reconstruction fidelity, achieving a favorable balance between distortion-based and perceptual metrics. These results are obtained under constrained computational settings, highlighting the efficiency of the proposed formulation. Overall, this work introduces a complementary direction to existing approaches by improving the representational power of local transformations, providing an efficient and scalable alternative to globally intensive architectures. 

---
# VisInject: Disruption != Injection -- A Dual-Dimension Evaluation of Universal Adversarial Attacks on Vision-Language Models 

**Authors**: Pang Liu, Yingjie Lao  

**Link**: [PDF](https://arxiv.org/pdf/2605.01449)  

**Abstract**: Universal adversarial attacks on aligned multimodal large language models are increasingly reported with attack success rates in the 60-80% range, suggesting the visual modality is highly vulnerable to imperceptible perturbations as a prompt-injection channel. We argue that this number conflates two distinct events: (i) the model's output was perturbed (Influence), and (ii) the attacker's chosen target concept was actually emitted (Precise Injection). We compose two existing techniques -- Universal Adversarial Attack and AnyAttack -- under an $L_{inf}$ budget of 16/255, and we add a dual-axis evaluation: a deterministic Ratcliff-Obershelp drift score for Influence (programmatic baseline) plus a 4-tier ordinal categorical none/weak/partial/confirmed for Precise Injection. The judge is DeepSeek-V4-Pro in thinking mode, calibrated against Claude Opus 4.7 with Cohen's $\kappa$ = 0.77 on the injection axis (substantial agreement); the entire 4475-entry SHA-256 input cache ships with the dataset so reviewers can re-derive paper numbers bit-exact without an API key. Across 6615 pairs over four open VLMs, seven attack prompts, and seven test images, the two axes diverge by roughly 90$\times$: 66.4% of pairs are programmatically disturbed (LLM-judged 46.6% at the substantial-or-complete tier), but only 0.756% (50/6615) reach any non-none injection tier and only 0.030% (2/6615) verbatim. The few injections that do land cluster on screenshot- or document-style carriers whose semantics already invite text transcription. BLIP-2 shows \emph{zero detectable drift} at $L_{inf}$ = 16/255 across all 2205 pairs even when used as a Stage-1 surrogate. We release the full dataset -- 21 universal images, 147 adversarial photos, 6,615 response pairs, the v3 dual-axis judge results, and the cache at this http URL. 

---
# Quantifying Multimodal Capabilities: Formal Generalization Guarantees in Pairwise Metric Learning 

**Authors**: Richeng Zhou, Xuelin Zhang, Liyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01424)  

**Abstract**: Multimodal learning leverages the integration of diverse data modalities to enhance performance in complex tasks. Yet, it frequently encounters incomplete or redundant modality data in real-world scenarios. This paper presents a fine-grained theoretical analysis of the generalization properties of multimodal metric learning models, addressing critical gaps in understanding the relationship between modality selection and algorithmic performance. We establish hierarchical relationships between function classes corresponding to different modality subsets and quantify the discrepancy between learned mappings and ground truth. Through rigorous analysis of pairwise complexity within the multimodal learning framework, we derive novel generalization error bounds that reveal the joint impact of modality quantity and granularity on model performance. Our theoretical findings on both upper and lower bounds demonstrate that incorporating fine-grained modality features reduces the complexity of the hypothesis space by enhancing modality complementarity. This work offers both theoretical foundations and practical implications for improving convergence rates and accuracy in multimodal learning systems. 

---
# HepScript: A Dual-Use DSL for Human-AI Collaborative Data Analysis Workflows in High-Energy Physics 

**Authors**: Junkun Jiao, Tong Liu, Ke Li, Weimin Song, Yipu Liao, Bolun Zhang, Beijiang Liu, Chang-Zheng Yuan, Yue Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.01423)  

**Abstract**: The escalating data scale in High-Energy Physics (HEP) fuels a growing aspiration for higher analytical efficiency. While Large Language Models (LLMs) offer a path toward automation via agentic AI, they struggle with complex scientific workflows that require deep domain knowledge and are tightly coupled to experiment-specific codebases. To address this, we introduce a methodology centered on HepScript, a dual-use Domain-Specific Language (DSL) for HEP data analysis workflows. HepScript serves as a shared formal interface, abstracting HEP analysis logic into a constrained syntax that is both intuitive for human experts and reliably generable by AI agents. First developed for the Beijing Spectrometer III (BESIII) experiment, HepScript hides the complexity of the underlying software stack, translating high-level analysis intent into low-level, production-ready code. In our case studies, this abstraction reduces the required human-written code by 93\%. Crucially, HepScript's constrained grammar defines a tractable action space, enabling AI agents to autonomously generate executable specifications for core analysis stages directly from published literature with a 95\% success rate. Our work demonstrates a scalable pathway toward human-AI collaborative systems, where a formally specified DSL acts as an unambiguous translation layer between human expertise, AI automation, and production environment, rendering previously intractable automation problems solvable. 

---
# Medmarks: A Comprehensive Open-Source LLM Benchmark Suite for Medical Tasks 

**Authors**: Benjamin Warner, Ratna Sagari Grandhi, Max Kieffer, Aymane Ouraq, Saurav Panigrahi, Geetu Ambwani, Kunal Bagga, Nikhil Khandekar, Arya Hariharan, Nishant Mishra, Manish Ram, Shamus Sim Zi Yang, Ahmed Essouaied, Adepoju Jeremiah Moyondafoluwa, Robert Scholz, Bofeng Huang, Molly Beavers, Srishti Gureja, Anish Mahishi, Sameed Khan, Maxime Griot, Hunar Batra, Jean-Benoit Delbrouck, Siddhant Bharadwaj, Ronald Clark, Ashish Vashist, Anas Zafar, Leema Krishna Murali, Harsh Deshpande, Ameen Patel, William Brown, Johannes Hagemann, Connor Lane, Paul Steven Scotti, Tanishq Mathew Abraham  

**Link**: [PDF](https://arxiv.org/pdf/2605.01417)  

**Abstract**: Evaluating large language models (LLMs) for medical applications remains challenging due to benchmark saturation, limited data accessibility, and insufficient coverage of relevant tasks. Existing suites have either saturated, heavily depend on restricted datasets, or lack comprehensive model coverage. We introduce Medmarks, a fully open-source evaluation suite with 30 benchmarks spanning question answering, information extraction, medical calculations, and open-ended clinical reasoning. We perform a systematic evaluation of 61 models across 71 configurations using verifiable metrics and LLM-as-a-Judge. Our results show that frontier reasoning models (Gemini 3 Pro Preview, GPT-5.1, & GPT-5.2) achieve the highest performance across both benchmarks, most frontier proprietary models are significantly more token efficient than open-weight alternatives, medically fine-tuned models outperform their generalist counterparts, and that models are susceptible to answer-order bias (particularly smaller models and Grok 4). A subset of our evals (Medmarks-T) can be directly used as reinforcement learning environments to post-train LLMs for medical reasoning. Code is available at this https URL 

---
# AMSnet-q: Unsupervised Circuit Identification and Performance Labeling for AMS Circuits 

**Authors**: Ze Zhang, Junzhuo Zhou, Yichen Shi, Zhuofu Tao, Rui Ji, Zhiping Yu, Quan Chen, Ting-Jung Lin, Lei He  

**Link**: [PDF](https://arxiv.org/pdf/2605.01404)  

**Abstract**: Analog and mixed-signal (AMS) circuit design remains heavily reliant on expert knowledge. While recent AI-driven automation tools can generate candidate topologies, they critically depend on manually curated datasets with functional and performance annotations -- a requirement that current large language models (LLMs) and vision models cannot automate. Existing approaches still require domain experts to manually interpret circuit functionality.
We present AMSnet-q, a fully automated, unsupervised pipeline that eliminates human-in-the-loop annotation by converting schematic images directly into a labeled AMS circuit database. Unlike prior work that stops at netlist extraction, our framework automates the complete verification loop: it performs schematic-to-netlist conversion, topology-aware testbench generation, and simulation-based sizing validation to objectively determine circuit functionality. Validated in 28 nm technology, AMSnet-q processed 739 schematics from the AMSnet 1.0 dataset, automatically constructing a repository of 4 circuit classes, 105 distinct topologies, and 89,789 labeled device configurations. By decoupling human effort from dataset volume and reducing the workload to a one-time testbench template per circuit class, AMSnet-q enables scalable, objective, and fully automated AMS database construction. 

---
# AI Expert Twin: Capturing Expert Cognition for Human-Centred, Practice-Based Learning 

**Authors**: Annie Yuan, Xiaohua Chen, Kalina Yacef, Judy Kay  

**Link**: [PDF](https://arxiv.org/pdf/2605.01401)  

**Abstract**: Tacit knowledge embedded in expert practice remains difficult to capture, formalise, and scale. While AI-driven educational systems have advanced personalisation, learner modelling, affective support, and self-regulated learning, they less often model the tacit reasoning and context-sensitive judgement that underpin expert practice in practice-based domains. This paper introduces the AI Expert Twin, a cognition-centric framework that models expert knowledge as structured, computable representations of procedural actions, semantic concepts, and decision processes. The framework also considers how value-laden preferences, trade-offs, and uncertainty shape expert judgement in practice. We formalise expert cognition as a three-layer representation and capture knowledge from experts under this model, laying the groundwork for integration into AI-powered educational system. A case study in a cultural heritage workshop demonstrates the feasibility of the approach in a real-world setting. The framework is designed to be transferable across domains such as vocational education and creative industries. By embedding expert heuristics into AI while maintaining transparency and learner agency, the AI Expert Twin offers a novel path towards scalable, practice-based learning and invites further research on ethical, human-centred applications of AI in education. 

---
# Investigating the Effects of Different Levels of User Control in an Interactive Educational Recommender System 

**Authors**: Qurat Ul Ain, Mohamed Amine Chatti, William Kana Tsoplefack, Rawaa Alatrash, Shoeb Joarder  

**Link**: [PDF](https://arxiv.org/pdf/2605.01400)  

**Abstract**: Educational recommender systems (ERSs) are becoming increasingly important in enhancing educational outcomes and personalizing learning experiences by providing recommendations of personalized resources and activities to learners, tailored to their individual learning needs. While user control is widely assumed to improve user experience, the effects of different levels of control in ERSs remain underexplored. To address this gap, we designed and evaluated an interactive ERS within the MOOC platform CourseMapper, where learners could interact with the input (i.e., user profile), process (i.e., recommendation algorithm), and output (i.e., recommendations) of the system. We conducted a between-subjects user study (N=184) to examine how varying levels of user control in an ERS influenced users' perceptions of the recommendation goals of perceived control, transparency, trust, satisfaction, and perceived quality. Our results show that enabling users to build and refine their profile is sufficient to promote positive perceptions of the ERS, while additional control options mainly reinforce these impressions. Moreover, perceived control is the only goal significantly affected by providing different levels of user control in the ERS, with input control exerting the strongest influence. Furthermore, different levels of control affect transparency, trust, satisfaction, and perceived quality in distinct yet interconnected ways. Overall, the findings provide empirical evidence that user control positively shapes transparency, trust, satisfaction, and perceived quality, though to varying extents. 

---
# Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning 

**Authors**: Sangkwon Park, Donghun Kang, Jisoo Mok, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2605.01399)  

**Abstract**: The conventional Retrieval-Augmented Generation (RAG) paradigm of injecting raw retrieved texts into the Large Language Model (LLM)'s context often results in suboptimal integration of retrieved information. This paper proposes to bridge retrieval results and the LLM's reasoning ability through Verbal Annotations, analytic narratives that explicitly articulate the logical connection between a search query and retrieved contexts. Our empirical investigation reveals the potential of Verbal Annotations to substantially enhance the LLM's ability to generate accurate, contextually-grounded responses. Motivated by this finding, we introduce Verbal-R3, a novel agentic RAG framework that consists of a Generator and a Verbal Reranker. The Generator performs iterative retrieval and reasoning, while the Verbal Reranker returns relevance scores and Verbal Annotations to guide the reasoning and answering process of the Generator. The inference process of Verbal-R3 is further refined through relevance-guided test-time scaling, which efficiently allocates test-time compute for effective trajectory expansion. Verbal-R3 achieves state-of-the-art performance on complex Question Answering benchmarks, validating the effectiveness of the proposed framework. 

---
# LiveFMBench: Unveiling the Power and Limits of Agentic Workflows in Specification Generation 

**Authors**: Dong Xu, Jialun Cao, Guozhao Mo, Junjie Hu, Cheng Wen, Hongyu Lin, Xianpei Han, Shengchao Qin, Cong Tian, Shing-Chi Cheung, Le Sun, Yaojie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01394)  

**Abstract**: Formal specification is essential for rigorous program verification, yet writing correct specifications remains costly and difficult to automate. Although large language models (LLMs) and agents have shown promising progress, their true capabilities and failure modes remain unclear. We present the first systematic and contamination-aware study of LLM- and agent-based formal specification generation for C programs. We introduce LiveFMBench, a continuously evolving benchmark of 630 ACSL (ANSI/ISO C Specification Language)-annotated C programs, including 360 newly collected cases designed to mitigate data leakage. Using this benchmark, we evaluate direct prompting with different sampling sizes, reasoning-enabled (thinking mode) inference, the agentic pipeline, and perform a fine-grained failure analysis. Experimental results reveal that naive evaluation substantially overestimates performance because models under direct prompting may exhibit unfaithful behaviors, such as deceiving automated provers or ignoring code-context constraints; after excluding such cases, the true specification generation accuracy drops by approximately 20\%. We further find that both increased sampling and thinking mode significantly improve success rates, with smaller models benefiting more from thinking mode. Agentic pipelines are particularly effective under low sampling budgets and on harder datasets. Failure analysis further shows that incorrect loop invariants are the dominant error type, while agentic pipelines notably reduce assertion errors. These results expose fundamental limitations in current LLM-based approaches and suggest they remain far from replacing human-authored formal specifications. We release LiveFMBench at this https URL and all evaluation artifacts to support future research. 

---
# Using LLMs in Software Design: An Empirical Study of GitHub and A Practitioner Survey 

**Authors**: Yifei Wang, Ruiyin Li, Peng Liang, Yangxiao Cai, Zengyang Li, Mojtaba Shahin, Arif Ali Khan, Qiong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.01392)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated significant potential across a wide range of software engineering tasks, including software design, an area traditionally regarded as highly dependent on human expertise and judgment. However, there has been little research focusing on how LLMs are used in software design, nor on the associated benefits and drawbacks. This paper aims to bridge this gap by empirically investigating how software developers utilize LLMs in the context of software design. We conduct a mixed-methods study, combining a mining study of 291 developer-ChatGPT conversations shared on GitHub with a survey of 65 software practitioners. Our findings reveal nine distinct categories of design tasks supported by ChatGPT, including architecture design, data model design, and the use of design patterns. We further characterize developer-ChatGPT interactions, showing that developers primarily use ChatGPT for knowledge acquisition and design-related code generation, with most tasks situated at the detailed design level. The study identifies seven key benefits of utilizing LLMs in software design as perceived by developers, such as better technology selection and the early detection of design flaws. We also uncover six limitations, including the generation of overly lengthy and difficult-to-read outputs, the creation of inexecutable or incorrect code, and a heavy reliance on context that can lead to hallucinated results. These findings provide an evidence-based characterization of current LLM use in software design from both open-source and practitioner perspectives, highlighting a tension between perceived benefits and limitations, which lays a foundation for future research and the development of effective techniques and tools to integrate LLMs into software design practices. 

---
# Sparse Representation Learning for Vessels 

**Authors**: Chinmay Prabhakar, Bastian Wittmann, Paul Büschl, Hongwei Bran Li, Bjoern Menze, Suprosanna Shit  

**Link**: [PDF](https://arxiv.org/pdf/2605.01382)  

**Abstract**: Analyzing human vasculature and vessel-like, tubular structures, such as airways, is crucial for disease diagnosis and treatment. Current methods often rely on small sub-regions or simplified tree-like structures, rendering analysis of entire organ-level networks at clinical resolution computationally challenging. To this end, we propose VAEsselSparse, an efficient encoder-decoder model to obtain a meaningful yet compact representation of the entire organ-level vascular network at sub-millimeter resolution. VAEsselSparse leverages the inherent sparsity of 3D vascular structures via sparse convolutions and attention mechanisms, achieving substantial spatial compression rates of 8 x 8 x 8. We demonstrate superior reconstruction performance compared to dense counterparts and previous methods. Importantly, the resulting latent space retains clinically relevant discriminative features readily usable for classification tasks, such as aneurysm/stenosis or subvariants of the circle of Willis. Moreover, the compact latent space of VAEsselSparse serves as an effective representation for learning vessel-specific priors through generative models, enabling the synthesis of realistic vasculature. 

---
# Focus on the Core: Empowering Diffusion Large Language Models by Self-Contrast 

**Authors**: Jinyuan Feng, Xin Yu, Yiqun Chen, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Zhiqiang Pu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01373)  

**Abstract**: The iterative denoising paradigm of Diffusion Large Language Models (DLMs) endows them with a distinct advantage in global context modeling. However, current decoding strategies fail to leverage this capability, typically exhibiting a local preference that overlooks the heterogeneous information density within the context, ultimately degrading generation quality. To address this limitation, we systematically investigate high-information-density (HD) tokens and present two key findings: (1) explicitly conditioning on HD tokens substantially improves output quality; and (2) HD tokens exhibit an early-decoding tendency, converging earlier than surrounding tokens. Motivated by these findings, we propose Focus on the Core \textbf{(FoCore)}, a training-free decoding strategy that utilizes HD tokens in a self-contrast manner, wherein HD tokens are temporarily remasked as negative samples, to guide generation. We further introduce FoCore\_Accelerate \textbf{(FoCore\_A)}, an efficient variant that, upon detecting HD token convergence, performs parallel decoding over stable candidates within a local context window, substantially accelerating generation. Extensive experiments on math, code and logical reasoning benchmarks demonstrate that FoCore consistently improves generation quality and efficiency across both LLaDA and Dream backbones. For instance, on HumanEval, FoCore improves pass@1 from 39.02 to 42.68 over standard Classifier-Free Guidance, while FoCore-A reduces the number of decoding steps by 2.07x and per-sample latency from 20.76s to 8.64s (-58.4\%). 

---
# MU-SHOT-Fi: Self-Supervised Multi-User Wi-Fi Sensing with Source-free Unsupervised Domain Adaptation 

**Authors**: Ahmed Y. Radwan, Hina Tabassum  

**Link**: [PDF](https://arxiv.org/pdf/2605.01369)  

**Abstract**: Deep learning has been widely adopted for WiFi CSI-based human activity recognition (HAR) due to its ability to learn spatio-temporal features in a privacy-preserving and cost-effective manner. However, DL-based models generalize poorly across environments, a challenge amplified in multi-user settings where overlapping activities cause CSI entanglement and domain shifts. Practical deployments often limit access to labeled source data due to privacy constraints, motivating source-free adaptation using only unlabeled target-domain CSI and a pre-trained source model. In this paper, we propose MU-SHOT-Fi, a source-free unsupervised domain adaptation framework for single- and multi-user Wi-Fi sensing. MU-SHOT-Fi employs permutation-invariant set prediction with Hungarian matching during source training, followed by frozen-classifier backbone adaptation in the target domain. To enable stable adaptation without labels, we introduce occupancy-weighted information maximization that prevents model collapse by focusing diversity regularization on likely-occupied slots while excluding the dominant class from marginal entropy. Additionally, we employ binary rotation prediction as spatial self-supervision that exploits CSI frequency-time structure to learn domain-invariant features. For single-user scenarios, we introduce SU-SHOT-Fi by replacing occupancy weighting with standard information maximization and incorporating contrastive predictive coding to exploit temporal consistency. Extensive experiments on the WiMANS and Widar 3.0 datasets across cross-environment, cross-frequency, cross-orientation, and combined domain shifts demonstrate that MU-SHOT-Fi effectively recovers multi-user exact-activity classification performance under large domain shifts while maintaining accurate occupancy estimation and preventing collapse toward dominant classes. 

---
# Model-Based Proactive Cost Generation for Learning Safe Policies Offline with Limited Violation Data 

**Authors**: Ruiqi Xue, Lei Yuan, Kainuo Cheng, Jing-Wen Yang, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01356)  

**Abstract**: Learning constraint-satisfying policies from offline data without risky online interaction is crucial for safety-critical decision making. Conventional methods typically learn cost value functions from abundant unsafe samples to define safety boundaries and penalize violations. However, in high-stakes scenarios, risky trial-and-error is infeasible, yielding datasets with few or no unsafe samples. Under this limitation, existing approaches often treat all data as uniformly safe, overlooking safe-but-infeasible states - states that currently satisfy constraints but inevitably violate them within a few steps - leading to deployment failures. Drawing inspiration from the concept of knowledge-data integration, we leverage large language models (LLMs) to incorporate natural language knowledge into the policy to address this challenge. Specifically, we propose PROCO, a model-based offline safe reinforcement learning (RL) framework tailored to datasets largely free of violations. PROCO first learns a dynamics model from offline data and constructs a conservative cost function by grounding natural-language knowledge of unsafe states in LLMs, enabling risk estimation even without observed violations. Using the cost function and learned model, PROCO performs model-based rollouts to synthesize diverse counterfactual unsafe samples, supporting reliable feasibility identification and feasibility-guided policy learning. Across a range of Safety-Gymnasium tasks with exclusively safe or minimally risky training data, PROCO integrates seamlessly with a variety of offline safe RL algorithms and consistently demonstrates reduced constraint violations and improved safety performance compared to both the original methods and other behavior cloning baselines. 

---
# AgriKD: Cross-Architecture Knowledge Distillation for Efficient Leaf Disease Classification 

**Authors**: Minh-Dung Le, Minh-Duc Hoang, Hoang-Vu Truong, Thi-Thu-Hong Phan  

**Link**: [PDF](https://arxiv.org/pdf/2605.01355)  

**Abstract**: Automated leaf disease classification is critical for early disease detection in resource-constrained field environments. Vision Transformers (ViTs) provide strong representation capability by modeling long-range dependencies and inter-class relationships; however, their high computational cost makes them impractical for deployment on edge devices. As a result, existing approaches struggle to effectively transfer these rich representations to lightweight models. This paper introduces AgriKD, a cross-architecture knowledge distillation framework for efficient edge deployment, which transfers knowledge from a Vision Transformer (ViT) teacher to a compact convolutional student model. To bridge the representational gap between Transformer and CNN architectures, the proposed approach integrates multiple distillation objectives at the output, feature, and relational levels, where each objective captures a different aspect of the teacher knowledge. This enables the student model to better preserve and utilize transformer-derived global representations. Experiments on multiple leaf disease datasets show that the distilled student achieves performance comparable to the teacher while significantly improving efficiency, reducing model parameters by approximately 172 times, computational cost by 47.57 times, and inference latency by 18-22 times. Furthermore, the optimized model is deployed across multiple runtime formats, including ONNX, TFLite Float16, and TensorRT FP16, achieving consistent predictive performance with negligible accuracy degradation. Real-world deployment on NVIDIA Jetson edge devices and a mobile application demonstrates reliable real-time inference, highlighting the practicality of AgriKD for AI-powered agricultural applications in resource-constrained environments. 

---
# VUDA: Breaking CUDA-Vulkan Isolation for Spatial Sharing of Compute and Graphics on the Same GPU 

**Authors**: Bin Xu, Pengfei Hu, Wenxin Zheng, Jinyu Gu, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.01352)  

**Abstract**: GPU-based simulation environments for embodied AI interleave physics simulation (CUDA) and photorealistic rendering (Vulkan) on a single device. We observe that two foundational scenarios -- simulation data generation and RL training -- can be naturally adapted to execute their simulation and rendering phases concurrently, presenting a significant opportunity to improve GPU utilization through spatial multiplexing. However, a fundamental obstacle we term execution isolation prevents this: CUDA and Vulkan create separate GPU contexts whose channels are bound to different scheduling groups, confining compute and graphics to mutually exclusive time slices. Existing spatial-sharing techniques are limited to the CUDA ecosystem, while temporal-sharing approaches underutilize available resources.
This paper presents VUDA, a system that breaks execution isolation to enable spatial parallelism between CUDA compute and Vulkan graphics workloads. VUDA is built on two key observations: although CUDA and Vulkan expose different programming abstractions, their execution paths converge to a common channel primitive at the driver and hardware level; meanwhile, their virtual-address spaces are inherently disjoint, making safe page-table merging feasible without remapping. VUDA exposes a thin API for developers to annotate co-schedulable CUDA streams, and realizes spatial sharing through channel redirection into Vulkan's scheduling domain and page-table grafting to unify address spaces, eliminating all data copying on the critical path. Experiments on representative embodied-AI workloads show that VUDA delivers up to 85% higher throughput than temporal-sharing baselines, while improving GPU utilization and reducing end-to-end latency. 

---
# MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate 

**Authors**: Jianze Wang, Ying Liu, Jinlong Chen, Xuchun Hu, Qilong Zhang, Yu Cao, Jun Wang, Hua Yang, Yong Xie, Qianglong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.01347)  

**Abstract**: On-policy distillation (OPD) trains a student on its own trajectories under token-level teacher supervision, but existing methods are capped by a single-teacher capability ceiling: when the teacher errs, the student inherits the error. OPD also remains largely unexplored in agentic tasks, where per-step errors compound across long trajectories and destabilize training. We propose MAD-OPD (Multi-Agent Debate-driven On-Policy Distillation), which breaks this ceiling by recasting the distillation teacher as a deliberative collective of teachers that debate over the student's on-policy state; the debate produces an emergent collective intelligence that supplies token-level supervision, with each teacher's contribution weighted by its post-debate confidence. To extend OPD to agentic tasks, we also introduce On-Policy Agentic Distillation (OPAD), which adds step-level sampling to stabilize training under multi-step error compounding. We additionally derive a task-adaptive divergence principle, selecting JSD (Jensen-Shannon divergence) for agentic stability and reverse KL (Kullback-Leibler) divergence for code generation, and verify it both theoretically and empirically. Across six teacher-student configurations (Qwen3 and Qwen3.5; 1.7B-14B students, 8B-32B teachers) and five agentic and code benchmarks, MAD-OPD ranks first across all six configurations; on the 14B+8B$\to$4B setting it lifts the agentic average by $+2.4\%$ and the code average by $+3.7\%$ over the stronger single-teacher OPD. 

---
# Active Reasoning Vision-Language Models via Sequential Experimental Design 

**Authors**: Anjie Liu, Ziqin Gong, Yan Song, Yuxiang Chen, Xiaolong Liu, Hengtong Lu, Kaike Zhang, Chen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.01345)  

**Abstract**: Visual perception in modern Vision-Language Models (VLMs) is constrained by a fundamental perceptual bandwidth bottleneck: a broad field of view inevitably sacrifices the fine-grained details necessary for complex reasoning. Inspired by the classical paradigms of active vision and information foraging, we frame overcoming this limitation as a sequential decision-making process. We formalise this process through the lens of the sequential Bayesian optimal experimental design (S-BOED) problem. While exact Bayesian inference is intractable in continuous gigapixel spaces, we derive principled yet tractable approximations that balance spatial coverage against resolution. To validate this framework, we present a training-free inference strategy as a practical instantiation of the S-BOED objective for agents equipped with multiple vision tools. Designed as a flexible template, this strategy accommodates arbitrary optimisation algorithms, ranging from efficient greedy sampling to look-ahead planning, to approximate the optimal design. Empirical evaluations on gigapixel-level benchmarks demonstrate that our approach further boosts the performance of state-of-the-art models, significantly outperforming standard baselines and effectively narrowing the gap towards human-annotated oracles. 

---
# ABox Abduction for Inconsistent Knowledge Bases under Repair Semantics 

**Authors**: Anselm Haak, Patrick Koopmann, Yasir Mahmood, Anni-Yasmin Turhan  

**Link**: [PDF](https://arxiv.org/pdf/2605.01341)  

**Abstract**: Given a knowledge base (KB) with a non-entailed fact, the ABox abduction problem asks for possible extensions of the KB that would entail this fact. This problem has many applications, ranging from diagnosis to explainability and repair. ABox abduction has been well-investigated for consistent KBs and classical semantics, but little is known for the case of inconsistent KBs, which can be caused by erroneous data. In this paper we define suitable notions of abduction in this setting and propose criteria that guide abduction towards "useful" hypotheses. To regain meaningful reasoning in the presence of inconsistencies, we use well-established repair semantics. We provide a comprehensive landscape of the complexity of ABox abduction under repair semantics, treating different variants of the abduction problem for the light-weight description logics DL-Lite and EL_bot. 

---
# Creating and Evaluating Figurative Language Dataset for Sindhi 

**Authors**: Wazir Ali, Adeeb Noor, Saifullah Tumrani  

**Link**: [PDF](https://arxiv.org/pdf/2605.01323)  

**Abstract**: In this article, we introduce SiNFluD, a novel benchmark dataset for Sindhi figurative language classification. We first collect raw text from various blogs, social media platforms, and literary sources, and subsequently prepare the corpus for annotation. Two native annotators label the data using the Doccano text annotation tool, achieving an inter-annotator agreement of 0.81. We then establish baseline results using 5-fold and 10-fold cross-validation. Finally, we evaluate mBERT, XLM-RoBERTa, and XLM-RoBERTa-XL models, along with SetFit for few-shot fine-tuning of sentence transformers. Among these, the pretrained XLM-RoBERTa-XL achieves the best performance. 

---
# GraphSculptor: Sculpting Pre-training Coreset for Graph Self-supervised Learning 

**Authors**: Chuang Liu, Zelin Yao, Xueqi Ma, Luzhi Wang, Mukun Chen, Pinghua Xu, Wenbin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01310)  

**Abstract**: Graph self-supervised learning typically relies on large-scale unlabeled datasets, heavily inflating computational costs. However, empirical evidence suggests that these datasets contain substantial redundancy-our analysis reveals that uniformly subsampling 50% of graphs retains over 96% of downstream performance. To exploit this redundancy, we introduce GraphSculptor for pre-training coreset construction. Unlike methods dependent on additional training-time signals or limited solely to topological statistics, GraphSculptor provides a label-free solution that constructs coresets via two complementary perspectives: intrinsic structure and contextual semantics. Concretely, structural diversity is quantified using intrinsic graph statistics, yielding a structural feature vector for each graph, while semantic diversity is captured by utilizing a pre-trained language model to encode descriptions generated via graph-to-text. GraphSculptor integrates these signals into a unified metric space and performs cluster-aware selection to preserve joint structural-semantic diversity. We further derive a theoretical bound on the loss gap between coreset and full-data pre-training, offering theoretical motivation for our selection formulation. Extensive experiments demonstrate that GraphSculptor effectively sculpts the dataset: a 10% coreset achieves 99.6% of full-data performance while reducing pre-training time by nearly 90%, offering a scalable solution for data-efficient graph pre-training. 

---
# Spectral- and Energy-efficient Multi-BS Multi-RIS Pinching-antenna Systems: A GNN-based Approach 

**Authors**: Changpeng He, Yang Lu, Wei Chen, Bo Ai, Arumugam Nallanathan, Zhiguo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.01307)  

**Abstract**: This paper investigates coordinated downlink transmission in a multi-base station (multi-BS) multi-reconfigurable intelligent surface (multi-RIS)-assisted pinching-antenna (PA) system, where each user equipment (UE) is associated with a single BS and each BS is equipped with movable PAs deployed on parallel waveguides. We formulate sum rate (SR) and energy efficiency (EE) maximization problems by jointly optimizing PA placement, RIS phase shifts, transmit beamforming, and BS-UE association under constraints of inter-PA spacing, power budget, and unit-modulus phase shift. To address the resulting highly coupled mixed-variable problem, we propose a three-stage graph neural network (GNN) that integrates heterogeneous and homogeneous graph representations and is trained end-to-end in an unsupervised manner. Extensive numerical results demonstrate that the proposed three-stage GNN consistently outperforms representative system and learning baselines, generalizes well to unseen numbers of UEs, RISs, and BSs, and maintains millisecond-level inference time. Besides, the results validate the effectiveness of the proposed design from both system and architectural perspectives. Moreover, PAs are shown to enhance SR and EE, and the performance gain is enlarged with increasing number of PAs. 

---
# Are we Doomed to an AI Race? Why Self-Interest Could Drive Countries Towards a Moratorium on Superintelligence 

**Authors**: Edward Roussel, Lode Lauwaert, Torben Swoboda, Grant Ramsey, Risto Uuk, Leonard Dung  

**Link**: [PDF](https://arxiv.org/pdf/2605.01297)  

**Abstract**: This paper uses game theory to argue that, contrary to the prevailing view, a moratorium on Artificial Superintelligence (ASI) can be in a state's self-interest. By formalizing trategic interactions between geopolitical superpowers, we model the trade-off between the benefits of technological supremacy and the catastrophic risks of uncontrolled ASI. The analysis reveals that as the perceived cost of loss of control increases sufficiently relative to other parameters, it becomes in each state's self-interest to impose a moratorium. We further provide empirical evidence suggesting that the global perception of ASI risk is rising, making a stable, rational moratorium increasingly plausible in the current geopolitical landscape. 

---
# Autonomous Drift Learning in Data Streams: A Unified Perspective 

**Authors**: Xiaoyu Yang, En Yu, Jie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01295)  

**Abstract**: In the pursuit of autonomous learning systems, the foundational assumption of stationarity, the premise that data distributions and model behaviors remain constant, is fundamentally untenable. Historically, the research community has addressed non-stationary environments almost exclusively under the scope of concept drift, focusing primarily on temporal shifts in streams. However, as learning systems become increasingly autonomous and complex, merely adapting to temporal non-stationarity is no longer sufficient. Evolving beyond this traditional perspective, we propose a novel, three-dimensional taxonomy that systematizes the field based on the operational state of the system. First, time stream drift distinguishes between stochastic arbitrary patterns and structural rhythmic dynamics. Second, data stream drift disentangles shifts in feature representations, identified as representation drift, from changes in underlying semantics, recognized as semantic drift. Third, model stream drift characterizes the internal endogenous divergence of learning systems through the lenses of sequential plasticity, decentralized heterogeneity, and policy instability. Based on this framework, we systematically review 193 representative studies and identify key open challenges. By bridging the fragmented paradigms of drift adaptation, continual learning, and temporal generalization, this survey outlines a roadmap for building self-evolving intelligent systems capable of learning autonomously through continuous change. 

---
# Chain of Evidence: Pixel-Level Visual Attribution for Iterative Retrieval-Augmented Generation 

**Authors**: Peiyang Liu, Ziqiang Cui, Xi Wang, Di Liang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.01284)  

**Abstract**: Iterative Retrieval-Augmented Generation (iRAG) has emerged as a powerful paradigm for answering complex multi-hop questions by progressively retrieving and reasoning over external documents. However, current systems predominantly operate on parsed text, which creates two critical bottlenecks: (1) \textit{Coarse-grained attribution}, where users are burdened with manually locating evidence within lengthy documents based on vague text-level citations; and (2) \textit{Visual semantic loss}, where the conversion of visually rich documents (e.g., slides, PDFs with charts) into text discards spatial logic and layout cues essential for reasoning. To bridge this gap, we present \textbf{Chain of Evidence (CoE)}, a retriever-agnostic visual attribution framework that leverages Vision-Language Models to reason directly over screenshots of retrieved document candidates. CoE eliminates format-specific parsing and outputs precise bounding boxes, visualizing the complete reasoning chain within the retrieved candidate set. We evaluate CoE on two distinct benchmarks: \textbf{Wiki-CoE}, a large-scale dataset of structured web pages derived from 2WikiMultiHopQA, and \textbf{SlideVQA}, a challenging dataset of presentation slides featuring complex diagrams and free-form layouts. Experiments demonstrate that fine-tuned Qwen3-VL-8B-Instruct achieves robust performance, significantly outperforming text-based baselines in scenarios requiring visual layout understanding, while establishing a retriever-agnostic solution for pixel-level interpretable iRAG. Our code is available at this https URL. 

---
# Developing a Strong Pre-Trained Base Model for Plant Leaf Disease Classification 

**Authors**: David J. Richter  

**Link**: [PDF](https://arxiv.org/pdf/2605.01283)  

**Abstract**: Plants, crops and their yields are essential to our very existence, but diseases and pests cause large losses every year. As such it is vital to ensure that diseases can be spotted early and treated accordingly and stopping the spread while still possible. Manual and traditional methods require personal to walk through the field and check for symptoms 'by hand'. This is very laborious and very time consuming, so ML methods have been applied as a result and they have garnered promising results. CNN models are especially efficient as they can automatically extract features from images without any manual feature construction before then feeding the features to a classifier. Datasets are largely influential to the final performance of the model. Despite the importance that datasets pose to the field, there still seems to be somewhat of a discrepancy between what is publicly available for use and what would be required to sufficiently train fully capable models. To overcome these shortcomings, as part of this thesis open datasets for the field of plant leaf disease classification have been identified as well as models that can be trained on them and extensive benchmarks have been carried out to identify their suitability. Then a new dataset was constructed based on those findings as well as on the findings of a augmentation applicability study, which will be used to train a new Base Model based on the DenseNet201 architecture, which managed to outperform the baseline model on said new dataset as well as outperforming it on plant leaf disease classification domain specific Transfer-Learning experiments on another new dataset. This new model manages to train models through Transfer-Learning (TL) faster, more robust, more stable, and with less data than general model would, overcoming a large number of issues that the field still suffers from. 

---
# A Target-Free Harmonization Method for MRI 

**Authors**: Minjun Kim, Dong Ju Mun, Hwihun Jeong, Hangyeol Park, Haechang Lee, Se Young Chun, Jongho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.01282)  

**Abstract**: In MRI, variations in scan parameters, sequence, or hardware can lead to discrepancies in image appearance, even for the same subject. These inconsistencies, known as domain shifts, can hinder image analysis and degrade the performance of deep learning models trained on data from specific target domains. MRI image harmonization aims to address these issues by aligning source domain images to the target domain images while preserving biological information such as anatomical structures. However, most existing harmonization approaches require access to both source and target domain data in training or test time. This dependence induces data sharing between institutions, raising concerns about patient privacy and substantially limiting the harmonization approaches that can be practically deployed in clinical settings. To overcome these limitations, we introduce TgtFreeHarmony, the harmonization framework tailored for target-free scenarios, eliminating the need for target domain data and any data sharing, enabling privacy-preserving harmonization directly within the source institution. Our approach estimates the target domain style by searching the manifold of MRI domain style constructed via a disentanglement-based generator using Bayesian optimization guided by the performance of a downstream task model, which is trained on target domain data. We evaluated our method on the brain tissue segmentation task across multiple institutes and demonstrated that it effectively harmonizes source images into target images, leading to improved downstream task performance. By enabling harmonization without any access to target-domain data, TgtFreeHarmony establishes a new direction of harmonization preserving data privacy that can be realistically deployed within clinical environments. 

---
# Position: LLM Serving Needs Mathematical Optimization and Algorithmic Foundations, Not Just Heuristics 

**Authors**: Zijie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.01280)  

**Abstract**: This position paper argues that LLM inference serving has outgrown generic heuristics and now demands mathematical optimization and algorithmic foundations. Despite rapid advances in serving systems such as vLLM and SGLang, their algorithmic cores remain largely unchanged from classical distributed computing: request routing uses join-shortest-queue or round-robin, scheduling defaults to FIFO, and KV cache eviction follows LRU. These general-purpose policies ignore the distinctive structure of LLM inference--dynamically growing KV cache memory, prefill-decode phase asymmetry, unknown output lengths, and continuous batching constraints. We contend that the field must develop mathematical models capturing these characteristics, enabling the design of algorithms with provable performance guarantees across diverse workloads, rather than heuristics that may succeed in some scenarios but fail unpredictably in others. Emerging work at the intersection of operations research and ML systems demonstrates that principled methods can match or exceed heuristic performance while providing theoretical guarantees. We call on the community to recognize algorithmic design for LLM serving as a research frontier. 

---
# CNN-based Multi-In-Multi-Out Model for Efficient Spatiotemporal Prediction 

**Authors**: Hyeonseok Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.01277)  

**Abstract**: Recently, Convolutional Neural Network (CNN) or Transformer architecture based models have been proposed to overcome the limitations of Recurrent Neural Network (RNN) based models in spatiotemporal prediction. These models prevent the inefficiency of parallelization limitation due to the sequential properties and stacked error due to the recursive method, and show high performance. Novertheless, there are still some challengies. First, CNN based models have difficulty considering global information due to the local properties of the kernel, and their performance is limited. In addition, information is mixed because the time axis is combined with the channel axis of the image for processing. Models based on Transformer architecture have high complexity due to the self-attention calcuation and take a long training time. In this paper, we propose a new structure model called CNN-based Multi-In-Multi-Out model for Efficient Spatiotemporal Prediction (MIMO-ESP) to overcome these limitations. MIMO-ESP considers global information and significantly improves complexity by configuring a Transformer architecture based on CNN. In addition, it treats the time axis as an independent axis without combining it, and effectively considers spatiotemporal information together by applying dilation. This structure makes MIMO-ESP efficient and high performance. Extensive experiment results on three promising benchmark datasets which including video, traffic, and precipitation prediction tasks demonstrate that the usefulness of MIMO-ESP due to the achieved competitive efficiency while outperforming existing models. Furthermore, the ablation study results demonstrate the usefulness of the components of MIMO-ESP, emphasizing the potential of the proposed approaches. 

---
# The Garden of Forking Paths: Narrative Arc-Conditioned Gameplay Planning 

**Authors**: Yunge Wen, Chenliang Huang, Hangyu Zhou, Zhuo Zeng, Chun Ming Louis Po, Julian Togelius, Timothy Merino, Sam Earle  

**Link**: [PDF](https://arxiv.org/pdf/2605.01245)  

**Abstract**: Narrative archetypes (e.g., Hero's Journey, Three-act structure) provide universal story structures that resonate across cultures and media and are important for video game storytelling, yet existing LLM-based methods lack explicit use of these archetypes in procedurally generated games. We propose Forking Garden, a framework for narrative arc-conditioned gameplay planning that generates branching games from user-provided storylines. Our approach first generates a diverse pool of independent nodes, then assembles them into a dungeon graph via arc-guided constraint algorithms, where each node achieves multimodal alignment of gameplay elements. We develop an end-to-end interactive system that instantiates the framework. 

---
# Rhamba: Region-Aware Hybrid Attention-Mamba Framework for Self-Supervised Learning in Resting-State fMRI 

**Authors**: Ruthwik Reddy Doodipala, Pankaj Pandey, Pratheek Eranki, Carolina Torres-Rojas, Manob Jyoti Saikia, Ranganatha Sitaram  

**Link**: [PDF](https://arxiv.org/pdf/2605.01240)  

**Abstract**: Self-supervised pretraining is promising for large-scale neuroimaging, yet the impact of region-aware masking and hybrid sequence modeling remains underexplored. In this work, we introduce Rhamba, a region-aware pretraining framework that integrates anatomically guided masking with hybrid Attention-Mamba architectures for resting state functional magnetic resonance imaging (fMRI) analysis. Models were pretrained on the ABIDE dataset using region-aligned patch embeddings and three masking strategies (Any, Majority, and Pure) with increasing spatial specificity. We evaluated four architectural variants: a Mamba only model, an Alternate architecture with interleaved Mamba and Attention blocks, and two hybrid encoder-decoder configurations (Attention-Mamba (AM) and Mamba-Attention (MA)). The pretrained models were fine-tuned on downstream classification tasks using the COBRE and ADHD-200 datasets for schizophrenia and attention-deficit/hyperactivity disorder discrimination. We employed Integrated Gradients, an explainable AI method, to identify the brain regions contributing to model predictions. Masking strategy strongly influenced reconstruction behavior, with reconstruction loss following a consistent ordering (Any > Majority > Pure). However, this trend did not directly translate into downstream performance, where differences were modest and dataset-dependent. The hybrid architecture with the MA configuration achieved the highest average AUROC across both datasets, and Rhamba outperformed state-of-the-art methods in comparative evaluation. Region-wise analysis showed that peak performance depends on the interaction between masking strategy and architecture rather than a single dominant configuration. Overall, Rhamba offers a flexible framework for balancing interpretability, scalability, and performance in large-scale fMRI representation learning. 

---
# MindMelody: A Closed-Loop EEG-Driven System for Personalized Music Intervention 

**Authors**: Yimeng Zhang, Yueru Sun, Haoyu Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01235)  

**Abstract**: Driven by the escalating global burden of mental health conditions, music-based interventions have attracted significant attention as a non-invasive, cost-effective modality for emotion regulation and psychological stress relief. However, current digital music services rely on static preferences and fail to adapt to users' instantaneous psychological states. Furthermore, directly mapping electroencephalography (EEG) to music generation remains challenging due to severe paired-data scarcity and a lack of interpretability. To address these limitations, we propose MindMelody, a fully functional, closed-loop real-time system for EEG-driven personalized music intervention. MindMelody introduces an emotion-mediated semantic bridge. Specifically, a hybrid Transformer-GNN first decodes real-time EEG signals into global Valence-Arousal states and local temporal affect trajectories. These states are then fed into a Retrieval-Augmented Generation (RAG)-equipped Large Language Model (LLM) to formulate structured intervention plans. Subsequently, a novel Hierarchical EEG Controller injects global affect prefixes and local temporal guidance into a pretrained music backbone, enabling fine-grained controllable audio synthesis. Crucially, the system incorporates a continuous feedback loop that updates generation parameters on the fly based on the user's evolving EEG dynamics. Extensive experiments show that MindMelody improves control adherence and emotional alignment, and receives higher perceived helpfulness in a short-term listening setting, suggesting its promise as an adaptive affect-aware music generation framework. 

---
# Minimizing Collateral Damage in Activation Steering 

**Authors**: Tam Nguyen, Tu Anh Nguyen, Sina Alemohammad, Richard G. Baraniuk  

**Link**: [PDF](https://arxiv.org/pdf/2605.01167)  

**Abstract**: Activation steering is a method for controlling Large Language Model (LLM) behavior by intervening in its internal representations to increase the alignment with a specific target feature direction. However, standard interventions, such as vector addition, often cause ``collateral damage", defined as unintended changes in the alignment of activations along other non-target feature directions. This damage occurs because standard methods implicitly assume the isotropy of non-target features. In this work, we provide a mathematical formalization of collateral damage and introduce a principled framework that models steering as a constrained optimization problem. Our method finds a new activation that minimizes the expected squared collateral change weighted by the empirical second-moment matrix of activations. This weighting encodes the nonuniform cost of the perturbation in different feature directions, in contrast to isotropic approaches that penalize changes uniformly in all feature directions. By accounting for the empirical second-moment of activations, our approach achieves more precise control while reducing the degradation of model performance on unrelated tasks. 

---
# The Productivity-Reliability Paradox: Specification-Driven Governance for AI-Augmented Software Development 

**Authors**: Sabry E. Farrag  

**Link**: [PDF](https://arxiv.org/pdf/2605.01160)  

**Abstract**: Since 2022, AI-powered coding assistants have produced contradictory evidence: controlled studies report 20-56% productivity gains on well-scoped tasks, while the most rigorous RCT documents a 19% slowdown for experienced developers, and telemetry across 10,000+ developers shows 98% more pull requests but 91% longer review times with flat delivery metrics. This paper argues these findings constitute the Productivity-Reliability Paradox (PRP): a systematic phenomenon emerging from non-deterministic code generators and insufficient specification discipline. Through a multivocal literature review of 67 sources (2022-2026), this paper: (1) formally defines the PRP with three moderating variables (task abstraction, codebase maturity, developer experience) and two amplifying mechanisms (code review bottleneck, context window constraint); (2) proposes the AI-Augmented Methodology Taxonomy (AAMT), classifying six methodologies under three AI integration tiers; (3) introduces the Specification Governance Model (SGM), grounded in Transaction Cost Economics, with a practical governance decision guide; and (4) evaluates Spec Kit and TDAD as SGM instantiations via a four-month pilot study. Specification discipline, not model capability, is the binding constraint on AI-assisted software dependability. 

---
# Multi-Perspective Transformers in ARC-AGI-2 Challenge 

**Authors**: Caleb Talley, Vedant Tibrewal, Seun Adekunle, Weiwen Dong, Xinyu Wu, Fariha Sheikh  

**Link**: [PDF](https://arxiv.org/pdf/2605.01154)  

**Abstract**: ARC-AGI-2 is a benchmark of human-intuitive visual puzzles that measures a machine's ability to generalize from limited examples, interpret symbolic meaning, and flexibly apply rules in varying contexts. In this paper, we discuss our approach to solving the ARC-AGI-2 puzzles with TinyLM, with additional fine-tuning at test time, including Test-Time-Training (TTT) and Products of Experts (POE). Our model achieves 96.1% accuracy on the training set and 21.7% accuracy on the evaluation set. 

---
# Semantic Context-aware mOdality fUsion Transformer (SCOUT): A Context-Aware Multimodal Transformer for Concept-Grounded Pathology Report Generation 

**Authors**: Suryakant Singh, Saarthak Kapse, Joel Saltz, Prateek Prasanna  

**Link**: [PDF](https://arxiv.org/pdf/2605.01144)  

**Abstract**: Whole-slide images (WSIs) present a fundamental challenge for computational pathology due to their extreme resolution, multi-scale heterogeneity, and the requirement for clinically reliable interpretation. Although recent pathology foundation models have enabled fluent report generation, they often lack clinical grounding, failing to accurately represent key diagnostic concepts and relationships observed by pathologists. This limitation arises from the difficulty of integrating heterogeneous visual evidence spanning fine-grained cellular patterns, slide-level tissue architecture, and high-level diagnostic concepts, while maintaining interpretability and clinical coherence. Here we present SCOUT: Semantic Context-aware mOdality fUsion Transformer, a context-aware concept-grounded multimodal framework for pathology report generation that enables progressive conditioning of image representations by global slide information and explicit diagnostic concepts. The method integrates local histological patterns, whole-slide context, and expert-curated semantic descriptors within a unified learning paradigm, allowing visual features to be dynamically refined throughout the encoding process. By combining depth-aware contextual modulation with adaptive multimodal fusion during text generation, the framework produces clinically coherent reports while preserving complementarity across representational scales. Using CONCH1.5 features, we evaluate SCOUT against WSI-Caption, HistGen, and BiGen on TCGA-BRCA, MICCAI REG, and HistAI. SCOUT achieves the best BLEU-1 to BLEU-4 and METEOR scores on all datasets, plus the best ROUGE-L on TCGA-BRCA and MICCAI REG. On TCGA-BRCA, it reaches 0.436/0.303/0.202/0.156 BLEU-1/2/3/4 and 0.204 METEOR; on REG 2025, it achieves 0.865/0.834/0.805/0.780 and 0.568. These results support progressive contextual conditioning for grounded pathology report generation. 

---
# Forager: a lightweight testbed for continual learning with partial observability in RL 

**Authors**: Steven Tang, Xinze Xiong, Anna Hakhverdyan, Andrew Patterson, Jacob Adkins, Jiamin He, Esraa Elelimy, Parham Mohammad Panahi, Martha White, Adam White  

**Link**: [PDF](https://arxiv.org/pdf/2605.01131)  

**Abstract**: In continual reinforcement learning (CRL), good performance requires never-ending learning, acting, and exploration in a big, partially observable world. Most CRL experiments have focused on loss of plasticity -- the inability to keep learning -- in one-off experiments where some unobservable non-stationarity is added to classic fully observable MDPs. Further, these experiments rarely consider the role of partial observability and the importance of CRL agents that use memory or recurrence. One potential reason for this focus on mitigating loss of plasticity without considering partial observability is that many partially-observable CRL environments are prohibitively expensive. In this paper, we introduce Forager, a light-weight partially-observable CRL environment with a constant memory footprint. We provide a set of experiments and sample tasks demonstrating that Forager is challenging for current CRL agents and yet also allows for in-depth study of those agents. We demonstrate that agents exhibit loss of plasticity, proposed mitigations can help, but that most useful is to leverage state construction. We conclude with a variant of Forager that generates an unending stream of new tasks to learn that clearly highlights the limitations of current CRL agents. 

---
# When Less is Enough: Efficient Inference via Collaborative Reasoning 

**Authors**: Yilei Chen, Sharut Gupta, Yannis Paschalidis, Ayush Sekhari, Aldo Pacchiano  

**Link**: [PDF](https://arxiv.org/pdf/2605.01111)  

**Abstract**: In this work, we introduce DUET (Dual-model Efficient Two-stage inference), a collaborative inference framework in which a capable model and a lightweight model work together to solve a task. Relying on a single large model to perform end-to-end reasoning and prediction often incurs substantial inference cost. In contrast, DUET decomposes inference into two stages: the capable model produces a reasoning signal, and the lightweight model interprets this signal to generate the final answer, allowing reasoning-intensive computation to be handled by the capable model while non-reasoning-intensive components are delegated to the lightweight model without sacrificing task performance. To achieve this objective, we propose a length-penalized joint training objective that encourages the capable model to transmit only the information that is sufficient for the lightweight model to solve the task. As a result, DUET maintains strong reasoning performance with substantially lower inference cost than end-to-end inference using a large model alone, saving up to 60% of the large model's output tokens on challenging reasoning benchmarks, including AIME and GPQA. 

---
# Component-Aware Self-Speculative Decoding in Hybrid Language Models 

**Authors**: Hector Borobia, Elies Seguí-Mas, Guillermina Tormo-Carbó  

**Link**: [PDF](https://arxiv.org/pdf/2605.01106)  

**Abstract**: Speculative decoding accelerates autoregressive inference by drafting candidate tokens with a fast model and verifying them in parallel with the target. Self-speculative methods avoid the need for an external drafter but have been studied exclusively in homogeneous Transformer architectures. We introduce component-aware self-speculative decoding, the first method to exploit the internal architectural heterogeneity of hybrid language models, isolating the SSM/linear-attention subgraph as a zero-cost internal draft. We evaluate this on two architecturally distinct hybrid families: Falcon-H1 (parallel: Mamba-2 + attention per layer) and Qwen3.5 (sequential: interleaved linear and attention layers), with a pure Transformer control (Qwen2.5). Parallel hybrids achieve acceptance rates of alpha = 0.68 at draft length k=2 under greedy decoding, while sequential hybrids yield only alpha = 0.038 -- an 18x gap attributable to how each architecture integrates its components. The property is scale-invariant: Falcon-H1 at 3B reproduces the rates observed at 0.5B. We further show that perplexity degradation from a companion ablation study predicts speculative viability without running speculative decoding: a 3.15x ratio (Falcon) maps to alpha = 0.37 at k=4, while 81.96x (Qwen) maps to alpha = 0.019. For sequential hybrids, generic LayerSkip achieves 12x higher acceptance rates than the component-aware strategy. The composition pattern of hybrid models -- not merely the presence of alternative components -- determines whether component-level self-speculation is viable. 

---
# Interpretable Difficulty-Aware Knowledge Tracing in Tutor-Student Dialogues 

**Authors**: Shuyan Huang, Alexander Scarlatos, Jaewook Lee, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2605.01097)  

**Abstract**: Recent advances in large language models (LLMs) have led to the development of AI-powered tutoring systems that provide interactive support via dialogue. To enable these tutoring systems to provide personalized support, it is essential to assess student performance at each turn, motivating knowledge tracing (KT) in dialogue settings. However, existing dialogue-based KT approaches often ignore question difficulty modeling and rely on opaque latent representations from LLMs, hindering accurate and interpretable prediction. In this work, we propose an interpretable difficulty-aware conversational KT framework built upon LLMs, which explicitly models students' abilities and the difficulty of tutor-posed tasks at each turn. The framework incorporates the original textual question and the next tutor-posed task to estimate the student's knowledge state and the difficulty of the upcoming turn. Furthermore, it integrates Item Response Theory to map LLM's outputs into student ability and question difficulty parameters, enabling interpretable prediction of student performance grounded in cognitive theories of learning. We evaluate the framework on two tutor-student dialogue datasets. Both quantitative and qualitative results show that our framework outperforms existing KT baselines, meanwhile generating interpretable outputs consistent with cognitive theory. 

---
# Governing What the EU AI Act Excludes: Accountability for Autonomous AI Agents in Smart City Critical Infrastructure 

**Authors**: Talal Ashraf Butt, Muhammad Iqbal, Razi Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2605.01091)  

**Abstract**: When a traffic signal controller adjusts green phases and a grid manager curtails power on the same corridor, each system may comply with its own obligations. The resident who suffers the combined effect has no single authority to hold accountable and, under the EU AI Act, limited means to obtain an explanation. Annex III, point 2 excludes safety-component AI in critical infrastructure from Article 86 explanation rights and Article 27 fundamental-rights impact assessment. Provider and deployer duties under Articles 9-15 still apply, and residual pathways under the GDPR, NIS2, and tortious liability offer partial coverage. The Act's principal resident-facing accountability instruments are nonetheless narrowed for the autonomous infrastructure systems most likely to interact across agencies. The paper traces this accountability deficit through four residual pathways (GDPR Article 22, GDPR transparency obligations, tortious liability, and NIS2) and shows that each is structurally bounded by individual-controller, individual-decision scope. As a governance response, it presents AgentGov-SC, a three-layer architecture (Agent, Orchestration, City) specifying 25 governance measures with bidirectional traceability to the EU AI Act, ISO/IEC 42001, and the NIST AI Risk Management Framework. Five conflict resolution rules and an autonomy-calibrated activation model complete the design. A scenario analysis traces governance activation through a multi-agent corridor cascade involving three documented UAE smart-city systems, with a contrasting single-system scenario confirming proportional activation. The paper contributes a regulatory gap analysis and governance architecture for an increasingly important class of urban AI deployment that existing frameworks treat as bounded and isolated. 

---
# A Sentence Relation-Based Approach to Sanitizing Malicious Instructions 

**Authors**: Soumil Datta, Melissa Umble, Daniel S. Brown, Guanhong Tao  

**Link**: [PDF](https://arxiv.org/pdf/2605.01078)  

**Abstract**: Retrieval-augmented generation and tool-integrated LLM agents increasingly depend on external textual sources. This reliance broadens the available attack surface, allowing adversaries to insert malicious instructions that trigger unintended model behaviors. Current defensive measures often utilize LLM-based detectors to filter such content, but these approaches remain vulnerable to optimization-based attacks. Additionally, training-based methods frequently fail to generalize to novel data distributions. To resolve these issues, we introduce SONAR, a prompt sanitization framework that identifies and removes injected content using metrics from natural language inference. Specifically, SONAR constructs a sentence-level relational graph across the user query and external data. By using entailment and contradiction scores as edge weights, the system identifies sentences that deviate from the core task. It then employs connectivity-driven pruning to eliminate flagged injection seeds and their related neighbors while maintaining benign context. Rigorous evaluations across several models and datasets show that SONAR reduces the attack success rate to nearly zero, significantly outperforming nine established baseline defenses. 

---
# LEAP: Layer-wise Exit-Aware Pretraining for Efficient Transformer Inference 

**Authors**: Shashank Kapadia, Deep Naryan Mishra, Sujal Reddy Alugubelli, Haoan Wang, Saipraveen Vabbilisetty, Rishi Bhatia, Anupriya Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2605.01058)  

**Abstract**: Layer-aligned distillation and convergence-based early exit represent two predominant computational efficiency paradigms for transformer inference; yet we establish that they exhibit systematic incompatibility under standard deployment conditions for convergence-based early exit. Distillation objectives that align intermediate student layers to teacher representations suppress the representational convergence that early-exit mechanisms exploit, rendering such mechanisms ineffective on distilled models. We introduce LEAP (Layer-wise Exit-Aware Pretraining), an auxiliary training objective that reconciles this incompatibility. LEAP requires no architectural modifications; it augments standard distillation with a single constraint ensuring intermediate layers approximate final-layer representations. LEAP-MiniLM achieves 1.61$\times$ measured wall-clock speedup (batch=1, NVIDIA L4) at $\theta$=0.95, with 91.9% of samples exiting by layer 7 and 1.80$\times$ theoretical layer reduction, where standard distilled models achieve zero effective speedup. We validate across sentence similarity (STS-B: 0.760 $\pm$ 0.006) and retrieval benchmarks (BEIR), providing operational guidance including latency measurements, decision thresholds, and deployment criteria. 

---
# SCION: Size-aware Policy Orchestration for Nonstationary Object Caches (Long Paper Version) 

**Authors**: Qizhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01055)  

**Abstract**: Object caches underpin cloud and edge services, but production workloads are heterogeneous, nonstationary, and throughput-constrained. Recent simple non-ML policies such as SIEVE and S3-FIFO set a strong baseline, so any learned method must be overhead-aware, robust under drift, and competitive with strong experts. We present SCION, a lightweight policy-orchestration framework that selects among a small set of deployable cache policies using a tiny workload fingerprint computed off the critical path. Our prototype, AUTO, uses short-prefix statistics of object size, cacheability, reuse, and cache size, then applies an offline-trained linear selector to choose among GDSF, S3-FIFO, SIEVE, LHD, W-TinyLFU-AV, and DynamicAdaptiveClimb; a simpler SCION-P90 variant uses only a p90 threshold. In a CPU-only, trace-driven evaluation on 30 public object-cache traces and a separate HR-Cache simulator subset, AUTO improves cacheable-only object miss ratio over SIEVE on a majority of workloads, stays close to the best single expert on average, enables explicit OMR/BMR tradeoff selection, and remains competitive on byte miss ratio. Under a fast-policy budget, AUTO-fast achieves lower cost than the best fixed fast policy. SCION reduces regime-mismatch risk while keeping the hot path unchanged. 

---
# Value Functions for Temporal Logic: Optimal Policies and Safety Filters 

**Authors**: Oswin So, William Sharpless, Sylvia Herbert, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2605.01051)  

**Abstract**: While Bellman equations for basic reach, avoid, and reach-avoid problems are well studied, the relationship between value optimality and policy optimality becomes subtle in the undiscounted infinite-horizon setting, particularly for more complicated tasks. Greedily maximizing the Q-function can produce policies that indefinitely defer task completion for reach-avoid problems, or equivalently, Until specifications, even when the value function is optimal. Building upon recent results decomposing the value function for temporal logic (TL) into a graph of constituent value functions, we construct non-Markovian policies based on state history that avoid this pathology and prove their optimality with respect to the quantitative robustness score for nested Until, Globally, and Globally-Until specifications. We further show how the Q function can serve as a safety filter for complex TL specifications, extending prior results beyond simple avoid or reach-avoid tasks. 

---
# LLM Ghostbusters: Surgical Hallucination Suppression via Adaptive Unlearning 

**Authors**: Joseph Spracklen, Pedram Aghazadeh, Farinaz Koushanfar, Murtuza Jadliwala  

**Link**: [PDF](https://arxiv.org/pdf/2605.01047)  

**Abstract**: Hallucinations, outputs that sound plausible but are factually incorrect, remain an open challenge for deployed LLMs. In code generation, models frequently hallucinate non-existent software packages, recommending imports and installation commands for fictional libraries. This creates a critical supply-chain vulnerability: an attacker can proactively register such packages on public registries with malicious payloads that are subsequently installed and executed by developers or autonomous agents, a class of package confusion attack known as slopsquatting. Once a model is deployed, mitigating this failure mode is difficult: full retraining is costly, and existing approaches either cause severe degradation of model utility or rely on a pre-specified forget-set, an assumption that does not apply to the unbounded space of hallucinations. To address this problem, we present Adaptive Unlearning (AU), a post-deployment framework that surgically suppresses hallucinations while preserving general model utility. AU introduces a hybrid token-level objective that simultaneously reinforces valid outputs and suppresses hallucinated ones. Combined with an adaptive discovery loop that continuously surfaces new hallucination-inducing contexts without human supervision, AU enables generalization to unseen prompts and hallucinations. We demonstrate that AU reduces package hallucination rates by 81%, corresponding to a substantial reduction in slopsquatting attack surface, while maintaining performance on standard coding benchmarks. Our analysis shows that distributional changes are concentrated on package-related generations, leaving general coding behavior largely unaffected and confirming that AU's effect is isolated to the targeted distribution. AU operates entirely on model-generated data, requires no human annotation, and generalizes across domains. 

---
# Separation Assurance between Heterogeneous Fleets of Small Unmanned Aerial Systems via Multi-Agent Reinforcement Learning 

**Authors**: Iman Sharifi, Hyeong Tae Kim, Maheed Hatem Ahmed, Mahsa Ghasemi, Peng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.01041)  

**Abstract**: In the envisioned future dense urban airspace, multiple companies will operate heterogeneous fleets of small unmanned aerial systems (sUASs), where each fleet includes several homogeneous aircraft with identical policies and configurations, e.g., equipage, sensing, and communication ranges, making tactical deconfliction highly complex for the aircraft. This paper aims to address two core questions: (1) Can tactical deconfliction policies converge or reach an equilibrium to ensure a conflict-free airspace when companies operate heterogeneous fleets of homogeneous aircraft? (2) If so, will the converged policies discriminate against companies operating sUASs with weaker configurations? We investigate a multi-agent reinforcement learning paradigm in which homogeneous aircraft within heterogeneous fleets operate concurrently to perform package delivery missions over Dallas, Texas, USA. An attention-enhanced Proximal Policy Optimization-based Advantage Actor-Critic (PPOA2C) framework is employed to resolve intra- and inter-fleet conflicts, with each fleet independently training its own policy while preserving privacy. Experimental results show that two fleets with distinct, shared PPOA2C policies can reach an equilibrium to maintain safe separation. While two PPOA2C policies outperform two strong rule-based baselines in terms of conflict resolution, a PPOA2C policy exhibits safer interaction with a rule-based policy, indicating adaptive capabilities of PPOA2C policies. Furthermore, we conducted extensive policy-configuration evaluations, which reveal that equilibria between similar policy types tend to favor fleets with stronger configurations. Even under similar configurations but different policy types, the equilibrium favors one of the heterogeneous policies, underscoring the need for fairness-aware conflict management in heterogeneous sUAS operations. 

---
# Certified Purity for Cognitive Workflow Executors: From Static Analysis to Cryptographic Attestation 

**Authors**: Alan L. McCann  

**Link**: [PDF](https://arxiv.org/pdf/2605.01037)  

**Abstract**: We present a certified purity architecture that converts governance enforcement in cognitive workflow systems from a runtime convention into a structural capability boundary. A prior three-layer governance architecture proves governance completeness, provenance completeness, and the impossibility of ungoverned effects, conditional on the pure module constraint: that step executors cannot perform effects. That constraint was enforced by module import graph analysis, which is insufficient against adversarial bypass on the BEAM virtual machine. This paper closes the gap through four mechanisms: (1) a restricted WebAssembly compilation target where effect-producing instructions are structurally absent; (2) purity certificates, cryptographically signed proofs binding executor binaries to their import classifications; (3) a runtime verification gate that rejects uncertified executors before they enter the governance pipeline; and (4) portable governance credentials via remote attestation for cross-organizational verification. We prove four theorems: structural purity by construction, bypass elimination for all five BEAM bypass classes, certificate integrity, and gate completeness. The guarantee holds relative to an explicit Trusted Computing Base. Evaluation on four implemented executors shows verification latency of 39--42 us, full plan cycle under 400 us, runtime overhead under 0.4% of a 100 ms HTTP request, and zero determinism divergences across repeated invocations. 

---
# EmoMM: Benchmarking and Steering MLLM for Multimodal Emotion Recognition under Conflict and Missingness 

**Authors**: Yueru Sun, Yimeng Zhang, Haoyu Gu, Nuo Chen, Dong She, Xianrong Yao, Yang Gao, Zhanpeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.01024)  

**Abstract**: Multimodal Emotion Recognition (MER) is critical for interpreting real-world interactions. While Multimodal Large Language Models (MLLM) have shown promise in MER, their internal decision-making mechanisms under modality conflict and missingness remain largely underexplored. In this paper, to systematically investigate these behaviors, we introduce EmoMM, a comprehensive benchmark featuring modality-aligned, conflict, and missing subsets. Through extensive evaluation, we uncover a Video Contribution Collapse (VCC) phenomenon, where MLLM marginalize video evidence due to high token redundancy and modality preferences. To address this, we propose Conflict-aware Head-level Attention Steering (CHASE), a lightweight mechanism that detects modality conflicts and performs inference-time attention steering, effectively mitigating decision bias without retraining the backbone. Experimental results demonstrate that CHASE consistently improves performance across various settings, significantly enhancing the reliability of MLLM in complex affective scenarios. 

---
# CLEAR: Revealing How Noise and Ambiguity Degrade Reliability in LLMs for Medicine 

**Authors**: Kevin H. Guo, Chao Yan, Avinash Baidya, Katherine Brown, Xiang Goa, Juming Xiong, Zhijun Yin, Bradley A. Malin  

**Link**: [PDF](https://arxiv.org/pdf/2605.01011)  

**Abstract**: Medical large language model (LLM) evaluations rely on simplified, exam-style benchmarks that rarely reflect the ambiguity of real-world medical inquiries. We introduce the CLinical Evaluation of Ambiguity and Reliability (CLEAR) framework, which assesses how decision-space presentation, ambiguity, and uncertainty affect LLMs' reasoning on medical benchmarks. CLEAR systematically perturbs (1) the number of plausible answer options, (2) the presence of a ground truth or abstention option, and (3) the semantic framing of answer options. Applying CLEAR on three benchmarks evaluated across 17 LLMs reveals three notable limitations of existing evaluation methods. First, increasing the number of plausible answers degrades a model's ability to identify the correct answer and abstain against incorrect ones. Second, this lack of caution intensifies as the framing of abstention shifts from assertive rejection like "None of the Above" to uncertainty admission like "I don't know" (IDK). Notably, just including IDK in the answer space increases incorrect answer selections. Lastly, we formalize the performance gap between identifying the correct answer and abstaining from incorrect ones as the humility deficit, which worsens with model scale. Our findings reveal limitations in standard medical benchmarks and underscore that scaling alone does not resolve LLM reliability issues. 

---
# Model Organisms Are Leaky: Perplexity Differencing Often Reveals Finetuning Objectives 

**Authors**: Mohammed Abu Baker, Luca Baroni, Dan Wilhelm  

**Link**: [PDF](https://arxiv.org/pdf/2605.00994)  

**Abstract**: Finetuning can significantly modify the behavior of large language models, including introducing harmful or unsafe behaviors. To study these risks, researchers develop model organisms: models finetuned to exhibit specific known behaviors for controlled experimentation. Identifying these behaviors remains challenging. We show that a simple perplexity-based method can surface finetuning objectives from model organisms by leveraging their tendency to overgeneralize their finetuned behaviors beyond the intended context. First, we generate diverse completions from the finetuned model using short random prefills drawn from general corpora. Second, we rank completions by decreasing perplexity gap between reference and finetuned models. The top-ranked completions often reveal the finetuning objectives, without requiring model internals or prior assumptions about the behavior. We evaluate this on a diverse set of model organisms (N=76, 0.5 to 70B parameters), including backdoored models, models finetuned to internalize false facts via synthetic document finetuning, adversarially trained models with hidden concerning behaviors, and models exhibiting emergent misalignment. For the vast majority of model organisms tested, the method surfaces completions revealing finetuning objectives within the top-ranked results, with models trained via synthetic document finetuning or to produce exact phrases being particularly susceptible. We further show that the technique can be effective even without access to the exact pre-finetuning checkpoint: trusted reference models from different families can serve as effective substitutes. As the method requires only next-token probabilities from the finetuned model, it is compatible with API-gated models that expose token logprobs. 

---
# Democratizing the medieval English legal tradition 

**Authors**: Michael Zhang, Elise Wang, Charlotte Whatley, Seth Strickland, Dylan Bannon  

**Link**: [PDF](https://arxiv.org/pdf/2605.00977)  

**Abstract**: The record of the beginning of the most widespread legal system in the world is contained in millions of pages of handwritten text. Most of the records of the first centuries of the Anglo-American legal system are hand-written in a highly abbreviated form of medieval Latin which only a few dozen scholars in the world are trained to read. In this interdisciplinary project, we construct a dataset of 4029 lines of text across 193 medieval criminal and civil cases. We then use the dataset to train an open-source end-to-end pipeline for transcribing these manuscripts. We first train standard neural network architectures for line segmentation and handwriting recognition (R-Blla and CNN+LSTM with CTC decoding, respectively) and show that they can already achieve 79% word accuracy, despite the relatively small training set and the challenge of expanding abbreviations. We then demonstrate that simple post-processing significantly boosts accuracy: adding an n-gram language model to the CTC decoder improves word accuracy to 82%, while asking Gemini Pro 3 to correct mistakes boosts accuracy to 88%. Finally, we compare the CNN+LSTM architecture with TrOCR, a transformer-based OCR architecture, demonstrating that TrOCR shows comparable word accuracy but worse character accuracy due to its over-willingness to guess, making it harder for humans to infer the correct reading. We incorporated our pipeline into a web portal (this http URL), opening up the English legal tradition to legal scholars, medievalists, and students. 

---
# Physiology-Aware Masked Cross-Modal Reconstruction for Biosignal Representation Learning 

**Authors**: Hao Zhou, Simon A. Lee, Cyrus Tanade, Keum San Chun, Juhyeon Lee, Migyeong Gwak, Megha Thukral, Justin Sung, Eugene Hwang, Mehrab Bin Morshed, Li Zhu, Viswam Nathan, Md Mahbubur Rahman, Subramaniam Venkatraman, Sharanya Arcot Desai  

**Link**: [PDF](https://arxiv.org/pdf/2605.00973)  

**Abstract**: Biosignals acquired from different locations on the body often provide temporally ordered views of the same underlying physiological process. However, most existing self supervised learning methods treat these signals as interchangeable views, overlooking the directional temporal dynamics that link them. A canonical example is the relationship between electrocardiography (ECG), which captures the electrical activation initiating each heartbeat, and photoplethysmography (PPG), which records the resulting peripheral pulse delayed by vascular dynamics. To capture this structured relationship, we introduce xMAE, a biosignal pretraining framework that leverages masked cross modal reconstruction across temporally ordered biosignals as a training time constraint to encourage physiologically meaningful timing structure in the learned representations. We show that pretraining with xMAE yields representations that outperform both unimodal and multimodal baselines on 15 of 19 downstream tasks, including cardiovascular outcome prediction, abnormal laboratory test detection, sleep staging, and demographic inference, while generalizing across devices, body locations, and acquisition settings. Further analysis suggests that the ECG PPG timing structure is reflected in the learned PPG representations. More broadly, xMAE demonstrates the effectiveness of incorporating temporal structure into multimodal pretraining when signals observe different stages of a shared underlying process. Code is available at this https URL. 

---
# Toward a Scientific Discovery Engine for Weather and Climate Data: A Visual Analytics Workbench for Embedding-Based Exploration 

**Authors**: Nihanth W. Cherukuru, Matt Rehme, Kirsten J. Mayer, David John Gagne, John Schreck, John Clyne, Charlie Becker  

**Link**: [PDF](https://arxiv.org/pdf/2605.00972)  

**Abstract**: Earth system science is producing increasingly large, high-dimensional datasets from physics based Earth system models to AI-based weather and climate models. Embedding-based representations can make these data searchable through similarity search and analog retrieval, but nearest neighbors in latent space are not automatically scientifically meaningful: it may reflect real weather structure, or preprocessing, geography, or model bias. Researchers therefore need ways to inspect how embeddings organize meteorological data, compare representation models, develop retrieval strategies, and verify results against physical evidence.
We present an open-source visual analytics workbench for each of these steps. The system links embedding experiments to source data, metadata, spatial context, and model configurations, so latent-space results can be traced back to the physics. Users can explore latent spaces for different models, issue global or localized queries, and inspect analogs through familiar meteorological views. This enables a discovery workflow in which scientists characterize a phenomenon of interest in a well-understood dataset, identifying its signature in latent space, and then use that signature to probe larger, less-labeled archives or ensembles for similar events. We demonstrate the workbench through tropical-cyclone retrieval using ERA5-derived embeddings and IBTrACS metadata, and evaluate its out-of-core retrieval backend to show that large embedding collections can be searched beyond in-memory limits on commodity workstation hardware. 

---
# MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio 

**Authors**: Harshit Rajgarhia, Shuubham Ojha, Asif Shaik, Akhil Pothanapalli, Rachuri Lokesh, Abhishek Mukherji, Prasanna Desikan  

**Link**: [PDF](https://arxiv.org/pdf/2605.00969)  

**Abstract**: We present MedMosaic, a medical audio question-answering dataset designed to benchmark language and audio reasoning models under realistic clinical constraints. Medical audio data is difficult to collect due to privacy regulations and high annotation costs arising from domain expertise. Thus, existing benchmarks tend to underrepresent complex medical audio scenarios. To address these challenges, MedMosaic features a diverse range of medical audio types, including condition-related physiological sounds, carefully constructed synthetic voices to mimic speech with artifacts as well as real short and long length clinical conversations to model varying context lengths. The dataset also features a total of 46,701 question-answer pairs, spanning categories such as multiple-choice, sequential multi-turn, and open-ended question-answers, enabling systematic evaluation of multi-hop reasoning and answer generation capabilities. Benchmarking 13 audio and multimodal reasoning models reveals that reasoning remains challenging for all evaluated systems, with substantial performance variation across question types. In particular, even state-of-the-art model like Gemini-2.5-pro can only achieve 68.1% accuracy approximately. These findings underscore persistent limitations in medical reasoning and highlight the need for more robust, domain-specific multimodal reasoning models. 

---
# Adaptive 3D-RoPE: Physics-Aligned Rotary Positional Encoding for Wireless Foundation Models 

**Authors**: Chenyu Zhang, Xinchen Lyu, Chenshan Ren, Shuhan Liu, Qimei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.00968)  

**Abstract**: Positional encoding plays a pivotal role in determin?ing the extrapolation and generalization performance of wireless foundation models for channel state information (CSI) modeling, latent characterization, and task-specific prediction. However, existing CSI models inherit static or one-dimensional positional priors from natural language and vision architectures, which fundamentally misalign with the intrinsic physics of wireless channels by lacking explicit relative decay, collapsing the 3D spatio-temporal-frequency structure, and remaining scenario?rigid. This paper proposes Adaptive 3D-RoPE, a physics-aligned rotary positional encoding that establishes the structural corner?stone for wireless foundation models. The framework integrates a learnable, axis-decoupled 3D frequency bank to explicitly disentangle multi-dimensional phase dependencies, coupled with a lightweight channel-conditioned controller that dynamically modulates the prior via compact global CSI descriptors. This sample-adaptive mechanism transforms positional encoding from a static transformer component into a dynamic, coherence-aware inductive bias to resolve heterogeneous channel physics. Extensive experiments across 100 datasets demonstrate the superiority of the proposed scheme in both scale extrapolation and zero-shot generalization. Compared to the state-of-the-art, our method achieves up to a 10.7 dB reduction in normalized mean square error (NMSE) under 8 times antenna scale extrapolation. Given the same CSI input scales, our method can also improve zero-shot NMSE by 1.07 dB across unseen mobility scenarios and 0.90 dB in low-frequency-to-millimeter-wave tasks. 

---
# Seeking Information with RAG-Assistants: Does Model Size Matter in Human-AI Collaborations? 

**Authors**: Lennard C. Froma, Tom Kouwenhoven, Maaike H.T. de Boer, Catholijn M. Jonker, Max J. van Duijn  

**Link**: [PDF](https://arxiv.org/pdf/2605.00964)  

**Abstract**: Much research on LLMs has focused on increasing benchmark performance. However, the evaluation of such models in real-world collaborative human-AI workflows has stayed behind. This work evaluates a chatbot-style assistant based on Retrieval-Augmented Generation (RAG) in a realistic multi-turn information-seeking scenario inspired by workplace settings where compliance with local legislation and secure handling of sensitive data are often key. Specifically, we examine the performance of humans (N=112) assisted by RAG-assistants compared to LLM-only or LLM+RAG baselines. In this setting, we investigate how underlying model size (3B, 8B, and 70B) shapes the human-AI collaborative dynamic and how it influences perceived usability and satisfaction. Results show that the performance gain of human-AI collaboration over the model-only baselines is significant, irrespective of model size, suggesting that hybrid systems are beneficial in information-seeking scenarios. Interestingly, however, perceived usability and satisfaction among participants showed little difference across model sizes. This demonstrates a nuanced trade-off between model size, performance, and user perception. Our work highlights the added value of evaluating AI applications in actual multi-turn interactions with human users, looking at usability and satisfaction besides accuracy, rather than focusing on benchmark performance only. 

---
# Ablation Study of Multimodal Perception, Language Grounding, and Control for Human-Robot Interaction in an Object Detection and Grasping Task 

**Authors**: Zi Tian, Guanting Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00963)  

**Abstract**: This manuscript extends our previous multimodal human-robot interaction system by introducing a controlled ablation study of the three modules that most strongly influence end-to-end performance: the large language model used for action extraction, the perception system used for visual grounding, and the controller used for motion execution. The goal is not to redesign the full pipeline, but to isolate the contribution of each component under a common experimental protocol and then evaluate the best combinations end-to-end. We therefore compare three language models, five perception configurations, and three controllers, followed by a second-stage factorial study over the best candidates. The resulting analysis is intended to clarify which choices primarily affect execution time, which primarily affect success rate, and where the largest engineering gains are likely to come from in future revisions of the system. 

---
# "I Don't Know" -- Towards Appropriate Trust with Certainty-Aware Retrieval Augmented Generation 

**Authors**: Daan Di Scala, Maaike de Boer, Pınar Yolum  

**Link**: [PDF](https://arxiv.org/pdf/2605.00957)  

**Abstract**: Achieving the right amount of trust in AI systems is important, but challenging. The problem is exacerbated with the rise of Large Language Models (LLMs) as they provide human-level communication capabilities, but potentially hallucinate in the content that they generate. Moreover, they express over-confidence in their answers, making it difficult for users to judge their truthfulness. An important human value that users seek is benevolence, which can be met by LLM's self-reflection leading to reliable and honest answers. Accordingly, this paper proposes conveying appropriate levels of self-reflected certainty to build appropriate trust. Our contributions are twofold: 1) We develop CERTA (Certainty Enhanced RAG for Trustworthy Answers), a specialized Retrieval Augmented Generation (RAG) system that incorporates the relevance between question, context, and answer to reflect its uncertainty in answering questions; 2) We create the Certainty Benchmark with 90 question-context pairs of non-objective questions, divided over four categories (factuality, preference, sycophancy, morality) and three types of contexts (relevant, incomplete, irrelevant). We run experiments with a baseline RAG system and three CERTA settings using two LLMs. Our evaluations indicate that CERTA helps identify answers that are uncertain, decreases the cases of over-agreeing, and provides cautious behavior when prompted for moral judgments. 

---
# E-MIA: Exam-Style Black-Box Membership Inference Attacks against RAG Systems 

**Authors**: Zelin Guan, Shengda Zhuo, Zeyan Li, Jinchun He, Wangjie Qiu, Zhiming Zheng, Shuqiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00955)  

**Abstract**: Retrieval-Augmented Generation (RAG) equips large language models (LLMs) with external evidence by retrieving documents at inference time, but it also turns the retrieval corpusinto a sensitive asset. Under a black-box setting, an adversary given a candidate document can infer whether it has been ingested into the RAG knowledge base (i.e., document-level membership inference) solely from query response interactions, thereby leaking corpus coverage and the existence of sensitive topics. Existing RAG MIA methods either rely on soft signals such as semantic similarity, which often yield overlapping member/non-member score distributions and unstable thresholds, or employ explicit confirmation probes whose intent is conspicuous and thus prone to refusal and detection. We propose E-MIA, which converts verifiable hard evidence in the target document (e.g., fine-grained details, proper nouns/technical terms, definitional statements, metadata cues, and causal/constraint relations) into an exam with four objectively gradable question types (FB/SC/MC/T/F), and uses the aggregated exam score across multiple evidence targeted questions as the membership signal. Experiments across multiple datasets and diverse RAG configurations demonstrate that E-MIA improves member/non-member separability in stringent settings while preserving natural, stealthy queries, and we further analyze the impact of question composition and exam length on attack effectiveness. 

---
# Graph Rewiring in GNNs to Mitigate Over-Squashing and Over-Smoothing: A Survey 

**Authors**: Hugo Attali, Nathalie Pernelle, Davide Buscaldi, Fragkiskos D. Malliaros  

**Link**: [PDF](https://arxiv.org/pdf/2605.00951)  

**Abstract**: Graph Neural Networks are powerful models for learning from graph-structured data, yet their effectiveness is often limited by two critical challenges: over-squashing, where information from distant nodes is excessively compressed, and over-smoothing, where repeated propagation makes node representations indistinguishable. Both phenomena stem from the interaction between message passing and the input topology, ultimately degrading information flow and limiting the performance of GNNs. In this survey, we examine graph rewiring techniques, a class of methods designed to modify the graph topology to enhance information propagation in GNNs. We provide a comprehensive review of state-of-the-art rewiring approaches, delving into their theoretical underpinnings, practical implementations, and performance trade-offs. 

---
# Co-Generative De Novo Functional Protein Design 

**Authors**: Xinrui Chen, Yizhen Luo, Siqi Fan, Zaiqing Nie  

**Link**: [PDF](https://arxiv.org/pdf/2605.00948)  

**Abstract**: De novo functional protein design aims to generate protein sequences that realize specified biochemical functions without relying on evolutionary templates, enabling broad applications in biotechnology and medicine. Existing approaches adopt either direct function-to-sequence mapping or decoupled structure-sequence generation strategies but often fail to achieve functionality and foldability simultaneously. To address this, we propose CodeFP, a Co-generative protein language model for de novo Functional Protein design that simultaneously decodes sequence and structure tokens, thereby enabling superior simultaneous realization of functionality and foldability. CodeFP utilizes functional local structures to enrich functional semantic encodings, overcoming the suboptimal translation of flat encodings into structure tokens, while introducing auxiliary functional supervision to alleviate training ambiguity stemming from the one-to-many structure-to-token mapping. Extensive experiments show that CodeFP consistently achieves average improvements of 6.1% in functional consistency and 3.2% in foldability over the strongest baseline. 

---
# SCARV: Structure-Constrained Aggregation for Stable Sample Ranking in Redundant NLP Datasets 

**Authors**: Xu Zheng, Feiyu Wu, Linhong Wu, Zhuocheng Wang, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.00944)  

**Abstract**: Sample-level rankings are increasingly used in data-centric NLP for analysis, filtering, debugging, and curation, yet existing pipelines typically score training examples pointwise and rank them as if they were independent. This assumption is fragile in the presence of exact duplicates, near-duplicates, paraphrases, and other redundant structure common in NLP corpora, where stochastic training can make highly similar examples receive unstable relative orderings across random seeds. We study stable sample-level ranking under redundancy and propose \textsc{SCARV}, a modular aggregation framework that operates on top of an existing scoring proxy. \textsc{SCARV} combines robust multi-seed aggregation with a structure-aware aggregation/allocation step over redundancy clusters. Across synthetic redundancy, naturally mined QQP redundancy, multiple proxy families, several NLP tasks, and end-to-end DistilBERT fine-tuning, \textsc{SCARV} substantially improves over bare proxy rankings in global and local stability and yields more reproducible ranking-based decisions such as subset selection and suspicious-example retrieval. Our decomposition and compute-aware frontier sharpen the mechanism: robust multi-seed aggregation is the dominant generic stabilizer, while the structure-aware component adds value mainly under low aggregation budgets or when redundancy clusters are informative, naturally occurring, or sufficiently covered. These results position \textsc{SCARV} not as a universal data selector or a universally dominant replacement for seed-only aggregation, but as a stability-oriented aggregation layer for proxy-induced rankings in redundant NLP datasets. 

---
# Interpretable experiential learning based on state history and global feedback 

**Authors**: Anton Kolonin  

**Link**: [PDF](https://arxiv.org/pdf/2605.00940)  

**Abstract**: A new interpretable experiential learning model based on state history and global feedback is presented. It is capable of learning a behavioral model represented by a transition graph between sets of states, with transitions attributed with utility and evidence count. This model is expected to be suitable for solving reinforcement learning problem in resource-constrained environments. The model was thoroughly evaluated on the OpenAI Gym Atari Breakout benchmark, demonstrating performance comparable to some known neural network-based solutions. 

---
# From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity 

**Authors**: Yee Zhing Liew, Andrew Huey Ping Tan, Anwar P.P Abdul Majeed  

**Link**: [PDF](https://arxiv.org/pdf/2605.00939)  

**Abstract**: Traditional hallucination detection fails on "Stubborn Hallucinations" -- errors where LLMs are confidently wrong. We propose a geometric solution: Embedding-Perturbed Gradient Sensitivity (EPGS). We hypothesize that while robust facts reside in flat minima, stubborn hallucinations sit in sharp minima, supported by brittle memorization. EPGS detects this sharpness by perturbing input embeddings with Gaussian noise and measuring the resulting spike in gradient magnitude. This acts as an efficient proxy for the Hessian spectrum, differentiating stable knowledge from unstable memorization. Our experiments show that EPGS significantly outperforms entropy-based and representation-based baselines, providing a robust signal for detecting high-confidence factual errors. 

---
# Fusing Urban Structure and Semantics: A Conditional Diffusion Model for Cross-City OD Matrix Generation 

**Authors**: Bin Chen, Zhuoya Meng, Fang Yang, Runkang Guo, Jingtao Ding, Yin Zhang, Chuan Ai, Zhengqiu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00938)  

**Abstract**: Accurate modeling of commuting flows is important for urban governance, traffic planning, and resource allocation. However, the combined influence of individual intentions, geographic constraints, and social dynamics leads to considerable heterogeneity in commuting patterns, making it difficult to develop generation models that generalize across cities. To address this issue, we propose SEDAN, a Structure-Enhanced Diffusion model conditioned on Attributed Nodes for generalizable OD matrix generation. SEDAN models a city as an attributed graph. Each region is treated as a node with demographic and point-of-interest features, and commuting flows are modeled as weighted edges. Adjacency and distance matrices are incorporated to characterize spatial structure. Based on this representation, we design a fusion mechanism within SEDAN to jointly model semantic information and spatial information. Regional semantic attributes are used to model latent travel demand through graph-transformer-based node interactions, while spatial structure is injected into the generation process as explicit constraints. The adjacency matrix guides attention weights to strengthen interactions between neighboring regions. Meanwhile, the distance matrix serves as a diffusion condition to capture spatial proximity and travel impedance. The fusion of urban semantics and spatial constraints enables SEDAN to generate OD matrices that are both behaviorally plausible and geographically coherent. Experiments on real-world OD datasets from U.S. cities show that SEDAN achieves a 7.38\% improvement in RMSE over the state-of-the-art baseline, WEDAN. It also remains robust across heterogeneous urban scenarios and varying structural patterns. Our work provides an effective and generalizable solution for commuting OD matrix generation. The code is available at this https URL. 

---
# EventADL: Open-Box Anomaly Detection and Localization Framework for Events in Cloud-Based Service Systems 

**Authors**: Luan Pham, Victor Nicolet, Joey Dodds, Hui Guan, Daniel Kroening  

**Link**: [PDF](https://arxiv.org/pdf/2605.00936)  

**Abstract**: Anomaly detection and localization (ADL) is critical for maintaining reliability and availability in cloud systems. Recent ADL developments focus on metric and log data, leaving event data unexplored. To address this gap, we propose EventADL, the first open-box event-based ADL framework for cloud-based service systems. To motivate the design of our framework, we conduct a systematic analysis on 520 real-world incidents, and provide insights into how anomalies and their root causes manifest through event data. EventADL has three phases: offline training, online anomaly detection, and root cause localization. During the training phase, EventADL first learns Event Semantic Patterns (ESPs), which capture normal interactions between system entities using historical event data, and then learns Event Frequency Patterns (EFPs), which capture the normal frequency of known ESPs. In the online anomaly detection phase, any data in the event stream that deviates significantly from either pattern is identified as anomalous. For localization, EventADL constructs an Intervention Graph that models the relationships between recent system interactions and the detected anomalies for automatic root cause localization. The framework is designed to operate efficiently with unlabeled data and to produce interpretable anomalies with their corresponding root causes. Our evaluation on three real cloud service systems and two real-world incidents demonstrates that EventADL outperforms existing methods, achieving F1-scores of at least 90% for anomaly detection and 100% top-3 accuracy in root cause localization. 

---
# CGM-JEPA: Learning Consistent Continuous Glucose Monitor Representations via Predictive Self-Supervised Pretraining 

**Authors**: Hada Melino Muhammad, Zechen Li, Flora Salim, Ahmed A. Metwally  

**Link**: [PDF](https://arxiv.org/pdf/2605.00933)  

**Abstract**: Continuous Glucose Monitoring (CGM) can detect early metabolic subphenotypes (insulin resistance, IR; $\beta$-cell dysfunction), but population-scale deployment faces two coupled problems. First, the same physiological state appears through multiple views (CGM time series, venous OGTT, Glucodensity summaries), so single-view representations fail to transfer when deployment shifts the modality or setting. Second, baselines perform inconsistently across these shifts. Both problems point to one remedy: representations that abstract away from any single view to capture higher-level temporal and distributional structure. We propose CGM-JEPA, a self-supervised pretraining framework which predicts masked latent representations rather than raw values, yielding abstraction that transfers across modalities. X-CGM-JEPA adds a masked Glucodensity cross-view objective for complementary distributional information. We pretrain on $\sim$389k unlabeled CGM readings from 228 subjects and evaluate on two clinical cohorts ($N=27$ and $N=17$ public-release subsets) across three regimes (cohort generalization, venous-to-CGM transfer, home CGM) under 20-iteration $\times$ 2-fold cross-validation. X-CGM-JEPA ranks first or second on AUROC for both endpoints across all three regimes while no baseline does, exceeding the strongest baseline by up to $+6.5$ pp in cohort generalization and $+3.6$ pp in venous-to-CGM transfer (paired Wilcoxon, $p<0.001$). Under modality shift, it matches mean AUROC while redistributing toward weaker subgroups (ethnicity AUROC gap shrinks 25-54%); on sparse in-domain venous data, the distributional view lifts label-aware clustering (ARI $+39\%$, NMI $+40\%$). Code and weights: this https URL 

---
# Code World Model Preparedness Report 

**Authors**: Daniel Song, Peter Ney, Cristina Menghini, Faizan Ahmad, Aidan Boyd, Nathaniel Li, Ziwen Han, Jean-Christophe Testud, Saisuke Okabayashi, Maeve Ryan, Jinpeng Miao, Hamza Kwisaba, Felix Binder, Spencer Whitman, Jim Gust, Esteban Arcaute, Dhaval Kapil, Jacob Kahn, Ayaz Minhas, Tristan Goodman, Lauren Deason, Alexander Vaughan, Shengjia Zhao, Summer Yue  

**Link**: [PDF](https://arxiv.org/pdf/2605.00932)  

**Abstract**: This report documents the preparedness assessment of Code World Model (CWM), a model for code generation and reasoning about code from Meta. We conducted pre-release testing across domains identified in our Frontier AI Framework as potentially presenting catastrophic risks, and also evaluated the model's misaligned propensities. Our assessment found that CWM does not pose additional frontier risks beyond those present in the current AI ecosystem. We therefore release it as an open-weight model. 

---
# CellxPert: Inference-Time MCMC Steering of a Multi-Omics Single-Cell Foundation Model for In-Silico Perturbation 

**Authors**: Andac Demir, Erik W. Anderson, Jeremy L. Jenkins, Srayanta Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2605.00930)  

**Abstract**: In this work, we introduce CellxPert, a scalable multimodal foundation model that unifies single-cell and spatial multi-omics within a common representation space. CellxPert jointly encodes transcriptomic (scRNA-seq), chromatin-accessibility (ATAC-seq), and surface-proteomic (CITE-seq) measurements, while directly incorporating MERFISH and imaging mass-cytometry data as 2D or 3D spatial-visual layers. CellxPert facilitates four key downstream tasks out of the box: (i) cell-type annotation across a broad ontology of 154 largely overlapping identities -- the largest label space addressed to date and a stringent test of fine-grained discrimination, (ii) efficient fine-tuning using Low Rank Adaptation (LoRA), (iii) genome-wide transcriptomic response prediction to in-silico perturbations (ISP), and (iv) seamless multi-omic integration across various assays and platforms. Unlike current single-cell foundation models, which approximate gene perturbations by deleting or reordering tokenized gene expression ranks, CellxPert employs a Metropolis-Hastings sampler whose proposal kernel uses the model's masked conditional distributions to transition to new transcriptomic states conditioned on the perturbed genes. This Markov-chain procedure mitigates out-of-distribution artifacts introduced by abrupt token manipulation and produces trajectories that are biologically interpretable. Evaluations on PBMC68K, Replogle Perturb-seq, Systema, and BMMC benchmarks show that CellxPert surpasses classical and state-of-the-art baselines in cell-type annotation, perturbation response prediction, and multi-omic integration. 

---
# PhaseNet++: Phase-Aware Frequency-Domain Anomaly Detection for Industrial Control Systems via Phase Coherence Graphs 

**Authors**: Raviteja Bommireddy, Varshith Bandaru, Lohith Pakala, Pradeep Kumar B  

**Link**: [PDF](https://arxiv.org/pdf/2605.00929)  

**Abstract**: Multivariate time series anomaly detection in ICS has attracted growing attention due to the increasing threat of cyber-physical attacks on critical infrastructure. State-of-the-art methods model inter-sensor relationships from raw time-domain amplitude values, using graph neural networks, Transformers. However, these methods discard the phase spectrum produced by time frequency transformations, We argue that phase information constitutes a complementary and previously overlooked detection modality for ICS anomaly detection. We present PhaseNet++, a frequency-domain autoencoder that operates on the Short-Time Fourier Transform (STFT) of sliding sensor windows, retaining both magnitude and phase spectra. A Phase Coherence Index (PCI), inspired by the Phase Locking Value from neuroscience, summarizes pairwise phase consistency across frequency bins into a continuous adjacency matrix. This matrix guides a graph attention network that propagates information preferentially among phase-synchronized sensors. A sensor-token Transformer encoder captures system-wide structure, and a dual-head decoder reconstructs magnitude and phase jointly via circular and coherence-aware objectives. Evaluated on the Secure Water Treatment (SWaT) benchmark, PhaseNet++ achieves an F1-score of 90.98%, ROC-AUC of 95.66%, and average precision of 91.51%. Ablation studies show that the phase-aware front-end and PCI graph module together add only 264,816 parameters, demonstrating that the phase inductive bias is lightweight. While the absolute F1-score is second best than that of all recent raw-value methods evaluated under different protocols, we position this work as the first systematic study of phase-domain anomaly detection for ICS. 

---
# StyleShield: Exposing the Fragility of AIGC Detectors through Continuous Controllable Style Transfer 

**Authors**: Guantian Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.00924)  

**Abstract**: AI-generated content (AIGC) detectors are increasingly deployed in high-stakes settings such as academic integrity screening, yet their reliability rests on a fundamental paradox: as language models are trained on human-written corpora, the statistical boundary between AI and human writing will inevitably dissolve as models improve. Commercial incentives have further distorted this landscape -- detection services and "de-AIification" tools often operate within the same supply chain, replacing evaluation of content quality with judgment of content origin. We present StyleShield, the first flow matching framework for conditional text style transfer, operating directly in continuous token embedding space via a DiT backbone with zero-initialized cross-attention adapters conditioned on frozen Qwen-7B representations. At inference, we adapt the SDEdit paradigm from image synthesis to text embeddings, with a single parameter gamma providing smooth continuous control over the evasion-preservation trade-off. On a multi-domain Chinese benchmark, StyleShield achieves 94.6% evasion against the training detector and >=99% against three unseen detectors, maintaining 0.928 semantic similarity. We further introduce RateAudit, a document-level scheduling algorithm that demonstrates detection-rate verdicts can be set to arbitrary values, directly questioning the reliability of score-based evaluation. 

---
# To Vibe Research or Not to Vibe Research? Generative AI in Qualitative Research 

**Authors**: Katja Karhu, Kari Smolander, Jussi Kasurinen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00922)  

**Abstract**: There has been intense debate among qualitative researchers about whether generative AI is suitable for qualitative research. In this paper, we summarize the broader ongoing discussion of generative AI in qualitative research and its implications for software engineering researchers. The qualitative research approach, small-q (positivist or post-positivist) or Big Q (non-positivist), is among the major criteria for determining whether generative AI can be used in qualitative research. In addition to research philosophy and research approach, skills, ethics, and personal preferences also play a role in researchers' decisions about whether to use AI in qualitative research. 

---
# Rethink MAE with Linear Time-Invariant Dynamics 

**Authors**: Zice Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00915)  

**Abstract**: Standard representation probing for visual models relies on mathematically permutation-invariant operations like Global Average Pooling (GAP) or CLS tokens, treating patch representations as an unstructured bag-of-words. We challenge this paradigm by demonstrating that token order is a critical, exploitable dimension in frozen visual representations (e.g., MAE, BEiT, DINOv2, and ViT as CLS-ablation extreme). We propose SSMProbe, a probing framework driven by a State Space Model (SSM). Operating as discrete Linear Time-Invariant (LTI) dynamical systems, SSMs act as permutation-sensitive probes where sequence order strictly dictates the final state due to inherent memory decay. Formulating token ordering as an information scheduling problem, we compare fixed scan heuristics against a differentiable soft permutation (Sinkhorn-based) learned from downstream supervision. Evaluations on standard and fine-grained classification benchmarks reveal a striking order gap: while fixed scans fail dramatically on highly localized patch features, our learned soft permutation successfully extracts highly competitive performance from otherwise heavily localized patch sequences. We find that pre-training objectives fundamentally shape token structure: DINOv2 concentrates global semantics in optimized CLS tokens leaving patches hyperspecialized, pure MAE preserves distributed representations with heterogeneous patch informativeness, and ViT represents a supervised CLS-dominated extreme. BEiT occupies middle ground. This heterogeneity is order-dependent -- meaning the SSM probe's performance depends critically on which tokens are placed at which temporal positions -- and is not merely a topological property of the spatial grid. SSMProbe's learned routing effectively discovers and exploits this heterogeneity, offering a powerful new diagnostic lens for visual representation analysis. 

---
# The Cost of Consensus: Isolated Self-Correction Prevails Over Unguided Homogeneous Multi-Agent Debate 

**Authors**: Blaž Bertalanič, Carolina Fortuna  

**Link**: [PDF](https://arxiv.org/pdf/2605.00914)  

**Abstract**: Multi-agent debate, where teams of LLMs iteratively exchange rationales and vote on answers, is widely deployed under the assumption that peer review filters hallucinations. Yet the failure dynamics of homogeneous debate remain poorly understood, therefore we report findings from a controlled empirical study of teams of $N{=}10$ homogeneous agents (Qwen2.5-7B, Llama-3.1-8B, Ministral-3-8B) across $R{=}3$ debate rounds on two high-difficulty benchmarks (GSM-Hard and MMLU-Hard). We compare peer debate against isolated self-correction and a stochastic noise control that injects rationales from unrelated problems. We decompose debate failure into three model-dependent pathways: sycophantic conformity, where agents uncritically adopt majority answers (modal adoption up to 85.5%); contextual fragility, where peer rationales destabilize previously correct reasoning (vulnerability rate up to 70.0%); and consensus collapse, where plurality voting discards correct answers already present in the generation pool (oracle gap up to 32.3 percentage points). Ablations over communication density ($K \in \{2,4,9\}$) and sampling temperature ($T \in \{0.4, 0.7\}$) show that conformity reaches high levels at minimal peer exposure ($K{=}2$) and intensifies with greater initial diversity. Across all configurations, debate consumes 2.1-3.4$\times$ more tokens (up to 28,631 tokens per problem) than self-correction for equal or lower accuracy. Our results indicate that, within the 7-8B parameter class, homogeneous teams without structured roles do not benefit from unguided peer exchange, and that isolated self-correction consistently offers a more favorable cost-accuracy tradeoff. 

---
# Leveraging Imperfect Medical Data: A Manifold-Consistent Spatio-Temporal Network for Sensor-based Human Activity Recognition 

**Authors**: Jiangtao Fan, Anish Jindal, Amir Atapour-Abarghouei  

**Link**: [PDF](https://arxiv.org/pdf/2605.00913)  

**Abstract**: Sensor-based Human Activity Recognition (HAR) has attracted increasing attention in medical and healthcare monitoring, particularly with the growth of Internet of Medical Things (IoMT). However, in real-world wearable sensing scenarios, IoMT signals are often corrupted by missing measurements, sensor failures, and environmental noise, which significantly degrade the performance of conventional deep learning models that assume clean and complete inputs. To address this challenge, we propose a Manifold-Consistent Spatio-Temporal Network (MCSTN) for robust HAR under imperfect sensing conditions. The proposed framework introduces a dual-level corruption modeling mechanism that simulates realistic sensor imperfections through both physical-level corruption and diffusion-driven continuous corruption. By enforcing representation consistency across multiple corrupted views, the model learns stable and corruption-invariant semantic representations. Furthermore, we design a dual-stream spatio-temporal architecture that explicitly decouples temporal dynamics modeling and spatial correlation learning. The temporal stream captures long-term activity dynamics, while the spatial stream models inter-sensor relationships, enabling more effective spatio-temporal representation learning. Extensive experiments on three widely used HAR benchmark datasets, PAMAP2, Opportunity, and WISDM, demonstrate that the proposed MCSTN achieves competitive performance compared with existing state-of-the-art methods, particularly under imperfect sensing conditions. These results validate the effectiveness and robustness of the proposed framework for real-world wearable IoMT sensing applications. 

---
# TRIP-Evaluate: An Open Multimodal Benchmark for Evaluating Large Models in Transportation 

**Authors**: Han Gong, Zhen Zhou, Yunyang Shi, Yan Tan, Jinbiao Huo, Qi Hong, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00907)  

**Abstract**: Large language models (LLMs) and multimodal large models (MLLMs) are increasingly used for transportation tasks such as regulation question answering, traffic management support, engineering review, and autonomous-driving scene reasoning. Yet transportation workflows are rule-intensive, computation-intensive, safety-critical, and inherently multimodal. Existing general benchmarks provide limited evidence of whether a model can apply regulations correctly, perform verifiable engineering calculations, or interpret traffic scenes reliably, while the small number of public transportation benchmarks remain narrow in scope and rarely support fine-grained diagnosis across text, images, and point-cloud data. To address this gap, we present TRIP-Evaluate, an open multimodal benchmark for large models in transportation. The benchmark organizes 837 items using a role-task-knowledge taxonomy that covers vehicle, traffic-management, traveler, and planning-and-design functions. Each item is annotated with capability, modality, and difficulty labels, enabling diagnosis from overall accuracy down to specific failure modes. The current release includes 596 text items, 198 image items, and 43 point-cloud items. TRIP-Evaluate also standardizes item construction, quality control, prompting, decoding, and scoring to improve cross-model comparability. Results on a diverse panel of models show that text-based performance is improving, but substantial weaknesses remain in multi-step engineering calculation, rule-constrained reasoning, multimodal scene understanding, and point-cloud understanding. Overall, TRIP-Evaluate provides a reproducible, diagnosable, and engineering-aligned evaluation baseline for model selection, regression testing, and safer deployment in transportation applications. 

---
# Generalized Category Discovery under Domain Shifts: From Vision to Vision-Language Models 

**Authors**: Hongjun Wang, Po Hu, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.00906)  

**Abstract**: Generalized Category Discovery (GCD) aims to categorize unlabelled instances from both known and unknown classes by transferring knowledge from labelled data of known classes. Existing methods assume all data comes from a single domain, yet real-world unlabelled data often exhibits domain shifts alongside semantic shifts. We study GCD under domain shifts and propose three frameworks that adapt foundation models, ranging from self-supervised vision models to vision-language models. (i) HiLo disentangles domain and semantic features through multi-level feature extraction and mutual information minimization, combined with PatchMix augmentation and curriculum sampling. (ii) HLPrompt extends HiLo with semantic-aware spatial prompt tuning to suppress background and domain noise. (iii) VLPrompt leverages vision-language models via factorized textual prompts and cross-modal consistency regularization. The three methods share core design principles while operating on different foundation backbones, making them suitable for different deployment scenarios. Extensive experiments on synthetic corruptions and real-world multi-domain shifts demonstrate consistent improvements over strong baselines. Project page: this https URL 

---
# DIAGRAMS: A Review Framework for Reasoning-Level Attribution in Diagram QA 

**Authors**: Anirudh Iyengar Kaniyar Narayana Iyengar, Tampu Ravi Kumar, Manan Suri, Raviteja Bommireddy, Dinesh Manocha, Puneet Mathur, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2605.00905)  

**Abstract**: Diagram question answering (Diagram QA) requires reasoning-level attribution that links each question-answer pair to all visual regions needed to derive the answer, rather than only the region containing the final response. Creating such structured evidence across diagrams, charts, maps, circuits, and infographics is time-consuming, and existing annotation tools tightly couple their interfaces to dataset-specific formats. We present DIAGRAMS, a lightweight, schema-driven review framework that decouples interface logic from dataset-specific JSON structures through an internal meta-schema and dataset adapters. Given an image and QA pair with optional candidate regions, the system performs QA-conditioned evidence selection and proposes the regions required for reasoning. When QA pairs or candidate regions are missing, it generates them and supports human verification and refinement. Across six Diagram QA datasets, model-suggested evidence achieves 85.39% precision and 75.30% recall against reviewer-final selections (micro-averaged). These results indicate that the review-first framework reduces manual region creation while maintaining high agreement with final reasoning-level attributions. We release a public demo and installable package to support dataset auditing, grounded supervision creation, and grounded evaluation. 

---
# RA-CMF: Region-Adaptive Conditional MeanFlow for CT Image Reconstruction 

**Authors**: Md Shifatul Ahsan Apurba, Md Selim, Jin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00901)  

**Abstract**: The use of CT imaging is important for screening, diagnosis, therapy planning, and prognosis of lung cancers. Unfortunately, due to differences in imaging protocols and scanner models, CT images acquired by different means may show large differences in noise statistics, contrast, and texture. In this study, we develop a novel conditional MeanFlow pipeline for CT image reconstruction. We introduce a conditional MeanFlow network that models the enhancement trajectory by predicting image-conditioned flow fields given intermediate image states. The image enhancement network is trained with a MeanFlow consistency loss along with the image reconstruction loss. In order to provide an adaptive refinement process in terms of spatial location of enhancements, we integrate a regional reinforcement learning-driven policy network into our approach. The policy network receives information about the MeanFlow rollouts and provides predictions in terms of tile-wise refinement budgets, stopping criteria, and total budget allocation of enhancement processes. Our policy network is trained through reinforcement learning in a policy gradient framework, where the goal of the training reward is to maximize improvement of enhancements while minimizing unnecessary computations and avoiding instabilities. In this way, our approach combines conditional flow-based enhancement with reinforcement learning-based spatial enhancement control. This allows our approach to focus more attention on enhancing difficult areas while stabilizing areas already showing sufficient quality. Our results show high accuracy in the tumor ROI, with the average radiomic feature CCC being 0.96, an average PSNR of 31.30 $\pm$ 4.16, and average SSIM of 0.94 $\pm$ 0.07. Moreover, there is an improvement in the overall quality of images, with an average PSNR of 34.23 $\pm$ 1.71 and average SSIM of 0.95 $\pm$ 0.01. 

---
# When Less Is More: Simplicity Beats Complexity for Physics-Constrained InSAR Phase Unwrapping 

**Authors**: Prabhjot Singh, Manmeet Singh  

**Link**: [PDF](https://arxiv.org/pdf/2605.00896)  

**Abstract**: Operational phase unwrapping is the primary computational bottleneck in InSAR-based volcanic and seismic monitoring. We challenge the industry trend of adopting high-complexity computer vision architectures, such as attention mechanisms, without validating their suitability for physics-constrained geophysical regression. We present the first large-scale architectural ablation study on a global LiCSAR benchmark (20 frames, 39,724 patches, 651M pixels). Our results reveal a significant "complexity penalty": a vanilla U-Net (7.76M parameters) achieves $R^2=0.834$ and RMSE $= 1.01$ cm, outperforming 11.37M-parameter attention-based models by 34% in $R^2$ and 51% in RMSE. Power Spectral Density (PSD) analysis provides the physical justification: while attention excels at capturing sharp semantic edges in natural images, it injects unphysical high-frequency artifacts ($>0.3$ cycles/pixel) into geophysical fields, violating the fundamental smoothness constraints of elastic surface deformation. With a 2.92ms inference latency (a $2.5\times$ speedup), the vanilla U-Net is the only candidate to comfortably meet the sub-100ms requirement for operational early-warning systems. This work bridges the "publication-to-practice" gap by proving that convolutional locality outperforms modern complexity for smooth-field regression, advocating for physics-informed simplicity in ML4RS. Code available at this https URL 

---
# Transfer Learning for Tonal Noise Prediction in VRF Units Using Thermodynamic and Vibration Signals 

**Authors**: ZhiWei Su, Ding Wang, Yuan Guo, Yang Qiao, HongJun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.00895)  

**Abstract**: The second-order harmonic (2f) component generated by twin-rotary compressor is a dominant low-frequency noise source of variable refrigerant flow (VRF) outdoor units, yet its amplitude fluctuates strongly with environmental thermal load and valve opening, making it difficult to assess accurately using conventional mechanism-based models. This paper proposes an unsupervised transfer learning method based on Domain-invariant Partial Least Squares (Di-PLS) to accurately predict 2f noise levels under new conditions using different signals. Prediction models utilizing thermodynamic signals and acceleration signals are constructed respectively, and the generalization performance of the proposed Di-PLS is systematically compared with traditional Partial Least Squares (PLS). Results demonstrate that Di-PLS significantly outperforms PLS by extracting cross-condition common features and minimizing the distribution discrepancy between the source and target domains. Specifically, the acceleration-based Di-PLS model achieves the best performance, maintaining prediction errors within 3 dB for all test cases. This superiority over thermodynamic-based models highlights a physical insight: while thermodynamic states drive dynamic changes, structural vibration possesses a stronger and more direct causal link to acoustic radiation. 

---
# Retrieval-Guided Generation for Safer Histopathology Image Captioning 

**Authors**: Md. Enamul Hoq, Wataru Uegami, Saghir Alfasly, Ghazal Alabtah, Sahar Rahimi Malakshan, Armita Kazemi, Alex T. Schmitgen, Fred Prior, H.R. Tizhoosh  

**Link**: [PDF](https://arxiv.org/pdf/2605.00893)  

**Abstract**: Generative vision-language models can produce fluent medical image captions but remain prone to hallucination, over-specific diagnostic claims, and factual inconsistency-serious issues in pathology. We investigate retrieval-guided generation (RGG) as a safer alternative, where captions are formed by summarizing expert text from visually similar cases rather than generated de novo. On the ARCH histopathology dataset, RGG improves semantic alignment with ground truth, achieving cosine similarity of $\approx$0.60 versus $\approx$0.47 from MedGemma, with non-overlapping confidence intervals indicating a robust gain. A pathologist-led qualitative review shows better preservation of morphology-relevant terminology and fewer unsupported diagnoses, while revealing failure modes such as concept mixing and inherited over-specific labeling. Overall, retrieval-guided captioning offers a more transparent and reliable approach with clearer opportunities for auditing than fully generative methods. 

---
# X2SAM: Any Segmentation in Images and Videos 

**Authors**: Hao Wang, Limeng Qiao, Chi Zhang, Lin Ma, Guanglu Wan, Xiangyuan Lan, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00891)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated strong image-level visual understanding and reasoning, yet their pixel-level perception across both images and videos remains limited. Foundation segmentation models such as the SAM series produce high-quality masks, but they rely on low-level visual prompts and cannot natively interpret complex conversational instructions. Existing segmentation MLLMs narrow this gap, but are usually specialized for either images or videos and rarely support both textual and visual prompts in one interface. We introduce X2SAM, a unified segmentation MLLM that extends any-segmentation capabilities from images to videos. Given conversational instructions and visual prompts, X2SAM couples an LLM with a Mask Memory module that stores guided vision features for temporally consistent video mask generation. The same formulation supports generic, open-vocabulary, referring, reasoning, grounded conversation generation, interactive, and visual grounded segmentation across image and video inputs. We further introduce the Video Visual Grounded (V-VGD) segmentation benchmark, which evaluates whether a model can segment object tracks in videos from interactive visual prompts. With a unified joint training strategy over heterogeneous image and video datasets, X2SAM delivers strong video segmentation performance, remains competitive on image segmentation benchmarks, and preserves general image and video chat ability. 

---
# Skeleton-Based Posture Classification to Promote Safer Walker-Assisted Gait in Older Adults 

**Authors**: Sergio D. Sierra M., Monica Sinha, Marcela Múnera, Carlos A. Cifuentes  

**Link**: [PDF](https://arxiv.org/pdf/2605.00890)  

**Abstract**: Falls among older adults are a significant public health concern, leading to severe injuries, loss of independence, and increased healthcare costs. This study evaluates the effectiveness of various models, including a Geometric approach, XGBoost, SVM, and several deep learning architectures, in classifying walker usage, standing vs. sitting, and posture for smart walkers used. Geometric and XGBoost were the top performers. XGBoost achieved near-perfect training accuracy in binary classification tasks, with 99.84% for walker choice and 99.69% for standing vs. sitting. For posture classification, Geometric approach attained 89.9% accuracy for 8 postures, and XGBoost obtained 99.24% during training for 17 postures. Deep learning models such as the 4-layer CNN and Encoder-Decoder CNN also demonstrated strong performance in binary classification, with accuracies above 98%. This study underscores the potential of machine learning to enhance human-robot interaction in smart walkers, particularly for fall prevention. 

---
# Selective Correlation Based Knowledge Distillation for Ground Reaction Force Estimation 

**Authors**: Eun Som Jeon, Jisoo Lee, Huisu Lim, Omik M. Save, Hyunglae Lee, Pavan Turaga  

**Link**: [PDF](https://arxiv.org/pdf/2605.00888)  

**Abstract**: Wearable sensor-based human gait analysis holds great promise in healthcare, rehabilitation, clinical diagnosis and monitoring, and sports activities. Specifically, ground reaction force (GRF) provides essential insights into the body's interaction with the ground during movement and is typically measured using instrumented treadmills equipped with force plates. However, such equipment is expensive and restricted to laboratory environments. To enable a more portable solution, wearable insole sensors have been used to measure GRF. These sensors, however, are prone to noise and external interference, which reduces measurement accuracy. Deep learning methodologies could be adopted to address these issues, but they often require significant computing resources to achieve high accuracy, limiting their applicability for real-time analysis on portable devices. To overcome these limitations, we propose Selective Correlation Based Knowledge Distillation (SCKD) for estimating GRF from data collected by insole sensors. Our proposed method utilizes selected features considering temporal characteristics in the process of extracting correlation maps for knowledge transfer, enhancing interpretability and mitigating issues in high dimensional data processing. We demonstrate the effectiveness of the compact models generated by our distillation framework through comparison with existing methods. Various configurations of teacher-student architectures and training approaches are examined based on multiple evaluation criteria, utilizing data collected at different walking speeds and with different window sizes. Experimental results confirm that our approach outperforms existing methods in estimating GRF from wearable insole sensor data. Therefore, our approach offers a reliable and resource-efficient solution for human gait analysis. 

---
# Towards High Fidelity Face Swapping: A Comprehensive Survey and New Benchmark 

**Authors**: Qi Li, Weining Wang, Shuangjun Du, Bo Peng, Jing Dong, Kun Wang, Zhenan Sun, Ming-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00883)  

**Abstract**: Face swapping has witnessed significant progress in recent years, largely driven by advances in deep generative models such as GANs and diffusion this http URL these advances, existing methods remain fragmented across different paradigms, and their evaluation is highly inconsistent due to the lack of standardized datasets and protocols. Moreover, prior surveys primarily focus on broader deepfake generation or detection, leaving face swapping insufficiently studied as a standalone problem. In this paper, we present a comprehensive survey and benchmark for face swapping. We provide a structured review of existing methods, organizing them into five major paradigms and systematically analyzing their design principles, strengths, and limitations. To enable fair and controlled evaluation, we introduce CASIA FaceSwapping, a high-quality benchmark with balanced demographic distributions and explicit attribute variations, and establish standardized protocols to assess the robustness of different face swapping methods. Extensive experiments on representative approaches yield new insights into the performance characteristics and limitations of current techniques. Overall, our work provides a unified perspective and a principled evaluation framework to facilitate the development of more robust and controllable face swapping methods. More results can be found at this https URL. 

---
# Adversarial Flow Matching for Imperceptible Attacks on End-to-End Autonomous Driving 

**Authors**: Xinyu Zeng, Xiangkun He, Lei Tao, Chen Lv, Hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.00880)  

**Abstract**: Autonomous driving (AD) is evolving towards end-to-end (E2E) frameworks through two primary paradigms: monolithic models exemplified by Vision-Language-Action (VLA), and specialized modular architectures. Despite their divergent designs, both paradigms increasingly rely on Transformer backbones for complex reasoning, potentially causing a shared vulnerability: visually imperceptible perturbations can manipulate E2E AD models into hazardous maneuvers by targeting the Transformer module. Most existing adversarial attack approaches against AD systems operate under white-box or black-box settings; yet, they typically necessitate full model transparency, or suffer from either prohibitive query latency or limited attack transferability. In this paper, we propose Adversarial Flow Matching (AFM), a novel gray-box attack framework that exploits Transformer structural vulnerabilities in E2E AD models. AFM enables efficient one-step generation of adversarial examples via a neural average velocity field. Additionally, the proposed technique yields effective and visually imperceptible attacks by synergistically perturbing the generative latent space and the neural average velocity field. Extensive experiments demonstrate that AFM achieves a superior trade-off between attack effectiveness and imperceptibility: it substantially degrades the performance of both VLA and modular AD agents across various scenarios compared to baselines, while maintaining state-of-the-art visual imperceptibility. Furthermore, adversarial examples generated by AFM exhibit robust cross-model transferability, indicating that AFM closely approximates a black-box attack setting while requiring only the prior knowledge that the target AD model incorporates a Transformer-based module. 

---
# OceanPile: A Large-Scale Multimodal Ocean Corpus for Foundation Models 

**Authors**: Yida Xue, Ningyu Zhang, Tingwei Wu, Zhe Ma, Daxiong Ji, Zhao Wang, Guozhou Zheng, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00877)  

**Abstract**: The vast and underexplored ocean plays a critical role in regulating global climate and supporting marine biodiversity, yet artificial intelligence has so far delivered limited impact in this domain due to a fundamental data bottleneck. Specifically, ocean data are highly fragmented across disparate sources and inherently exhibit multi-modal, high-noise, and weakly labeled characteristics, lacking unified schemas and semantic alignment. Although Multimodal Large Language Models (MLLMs) have achieved remarkable success in general domains, their application to ocean science remains severely constrained by the absence of large-scale, well-aligned multimodal datasets tailored to marine environments. To bridge this gap, we introduce OceanPile, a large-scale multimodal corpus designed for ocean foundation models. It comprises three key components: OceanCorpus, a unified collection integrating sonar data, underwater imagery, marine science visuals, and scientific text from diverse authoritative sources; OceanInstruction, a high-quality instruction dataset synthesized via a novel pipeline guided by a hierarchical Ocean Concept Knowledge Graph; and OceanBenchmark, a manually curated evaluation benchmark for rigorous assessment. We establish a multi-stage quality control process to ensure scientific validity and alignment across modalities. Experimental validation demonstrates significant performance improvements for models trained on our data. All datasets are publicly released to advance the field of marine artificial intelligence and empower domain-specific MLLMs. 

---
# Visual Chart Representations for Cryptocurrency Regime Prediction: A Systematic Deep Learning Study 

**Authors**: Dustin M. Haggett  

**Link**: [PDF](https://arxiv.org/pdf/2605.00875)  

**Abstract**: Technical traders have long relied on visual analysis of candlestick charts to identify market patterns and predict price movements. While deep learning has achieved remarkable success in image classification, its application to financial chart images remains underexplored. This paper presents a systematic study comparing different visual representations for cryptocurrency regime prediction. We evaluate three image encoding methods (raw candlestick charts, Gramian Angular Fields, and multi-channel GAF), five chart component configurations, four neural network architectures (CNN, ResNet18, EfficientNet-B0, and Vision Transformer), and the impact of ImageNet transfer learning. Through eight controlled experiments on Bitcoin, Ethereum, and S&P 500 data spanning 2018-2024, we identify optimal configurations for visual regime classification. Our results show that a simple 4-layer CNN on raw candlestick charts achieves 0.892 AUC-ROC, outperforming larger pretrained models. Surprisingly, simpler representations (price-only charts, 128x128 resolution) consistently outperform more complex alternatives. We provide interpretability analysis using GradCAM and demonstrate that transfer learning improves performance by 4-16% despite the domain gap between natural images and financial charts. 

---
# Latent Space Probing for Adult Content Detection in Video Generative Models 

**Authors**: Alizishaan Khatri, Chiquita Prabhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00874)  

**Abstract**: The rapid proliferation of AI-powered video generation systems has introduced significant challenges in content moderation, particularly with respect to adult and sexually explicit material. Existing detection methods operate on either prompts or decoded pixel-space outputs. Therefore, both approaches are blind to the rich internal representations formed during generation. In this paper, we propose a novel latent space probing framework that intercepts the denoised latent representations produced by the CogVideoX video diffusion model during inference and attaches lightweight classifiers to perform real-time adult content detection. To support this work, we construct a large-scale binary dataset of 11039 ten-second video clips (5086 violating, 5953 non-violating) sourced from adult websites and YouTube respectively. We introduce two lightweight probing classifier architectures. We train and evaluate it on the dataset. Our work demonstrates that latent-space signals encode strong discriminative features for harmful content detection, achieving 97.29% F1 on our held-out test set with an overhead in the 4-6ms range. Our results suggest that probing the latent space results in improvements in both detection performance as well as cost. 

---
# BRITE: A Benchmark for Reliable and Interpretable T2V Evaluation on Implausible Scenarios 

**Authors**: Advait Tilak, Jiwon Choi, Nazifa Mouli, Wei Le  

**Link**: [PDF](https://arxiv.org/pdf/2605.00873)  

**Abstract**: The rapid advancement of photorealistic Text-to-Video (T2V) generation brings in an urgent need for up-to-date evaluation methods. Existing benchmarks largely overlooked implausible scenarios and do not measure audio-visual alignment. We introduce BRITE, the first framework that unifies (1) implausible prompting, (2) fine-grained assessment of audio-visual consistency, and (3) QA-based interpretable evaluation into a comprehensive T2V benchmark. Unlike fully automated Multimodal LLM-based pipelines, which are prone to hallucination and prompt ambiguity, BRITE guarantees reliability through a rigorous human-in-the-loop protocol for benchmark creation. Evaluating five state-of-the-art models (Sora 2, Veo 3.1, Runway Gen4.5, Pixverse V5.5, and Qwen3Max), we reveal a critical performance gap: while models excel at static object composition, they exhibit significant degradation in object-action binding and audio-visual synchronization. Our framework offers the community a reliable, interpretable benchmark and evaluation framework that can detect and locate limitations in the next generation of T2V models, especially for off-manifold prompts 

---
# Multi-View Hierarchical Representation Learning of Fetal Hemodynamics for Maternal Hypertension Detection at the Edge 

**Authors**: Alireza Rafiei, Anahí Venzor Strader, Esteban Castro Aragón, Victoriana Rosibely Sut Serech, Enma Carolina Coyote Ixen, Reza Sameni, Peter Rohloff, Gari D. Clifford, Nasim Katebi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00872)  

**Abstract**: Hypertensive disorders of pregnancy remain a leading cause of maternal and fetal morbidity worldwide, yet diagnosis relies on intermittent cuff-based blood pressure measurements that are prone to bias and fail to capture continuous physiological dynamics. Growing evidence suggests that fetal cardiovascular activity is associated with maternal-placental hemodynamics and may encode markers of maternal hypertension. To analyze this, we collected a large-scale dataset of fetal one-dimensional Doppler ultrasound recordings paired with maternal blood pressure from 3,255 pregnant women across 8,170 antenatal visits in rural Guatemala. We developed AutoHyPE, a hierarchical attention network that models short- and long-term signal structure, incorporating a novel prototype-based contrastive learning and multi-view strategy to enhance representation robustness under long-tailed class distribution and biological variability. AutoHyPE achieved an AUROC of 0.80 for maternal hypertension detection, outperforming baseline approaches while maintaining balanced performance across classes, with no performance degradation in an edge deployment scenario. Our findings demonstrated that fetal cardiac mechanical activity contains hemodynamic features indicative of maternal hypertension status. This supports a promising paradigm shift toward continuous, objective monitoring of maternal health using existing, low-cost ultrasound technology and introduces a complementary approach to traditional methods based on blood pressure measurements, advancing scalable prenatal care. 

---
# NAKUL-Med: Spectral-Graph State Space Models with Dynamics Kernels for Medical Signals 

**Authors**: Badri N. Patro, Vijay S. Agneeswaran  

**Link**: [PDF](https://arxiv.org/pdf/2605.00871)  

**Abstract**: State space models (SSMs) achieve linear-time complexity but struggle with multi-channel physiological signals due to three limitations: fixed kernels cannot capture multi-scale temporal dynamics (motor preparation over hundreds of milliseconds vs. execution transients in tens of milliseconds), Markovian state updates restrict global context for periodic oscillations, and channel-independent processing ignores spatial electrode topology. We introduce NAKUL, extending SSMs for medical signal analysis through three contributions: (1) Dynamic Kernel Generation-parallel SSM branches with varying kernel sizes (3, 5, 7, 11 timesteps) are weighted by a meta-network that analyzes input statistics, enabling adaptive temporal scale selection; (2) Spectral Context Modeling-FFT-based operations with learnable Gaussian frequency band filters capture global periodic patterns in $O(N \log N)$ complexity; (3) Graph-Guided Spatial Attention-fixed electrode topology provides spatial biases to multi-head attention for principled cross-channel interaction. On BCI Competition IV-2a motor imagery (our primary benchmark), NAKUL achieves 91.7$\pm$0.6\% accuracy, matching EEG-Conformer (92.1$\pm$0.7\%) while using 28\% fewer parameters (2.5M vs 3.5M) and 2.0$\times$ faster inference (4.3ms vs 8.7ms). The model generalizes to EEG emotion recognition (83.6\%), multimodal EEG-fMRI (91.4\%), and medical imaging (92.8\% on ultrasound), demonstrating architectural versatility. Ablations show dynamic kernels contribute +2.6\% and exhibit interpretable scale selection patterns correlated with known neural dynamics. 

---
# An Algorithm for On-Sensor Agnostic Detection of Changes in Human Activity for Ultra-Low-Power Applications 

**Authors**: Sara Rimoldi, Arianna De Vecchi, Hazem Hesham Yousef Shalby, Federica Villa  

**Link**: [PDF](https://arxiv.org/pdf/2605.00870)  

**Abstract**: Wearable devices running Human Activity Recognition(HAR) on Inertial Measurement Units~(IMUs) waste energy by performing continuous classification for each window, even during long periods of unchanged activity. We address this with a lightweight change-detection gate: a non-parametric algorithm based on dynamic template matching that runs continuously at only approximately 16kFLOPs per step, requires no offline training, and does not need prior definition of target activity classes. The gate invokes the full HAR network only when it detects an activity change, reducing the computational load by over 67% in realistic monitoring settings. The algorithm is evaluated on smart glasses, smartwatch, and smartphone data, requiring only a brief device-specific calibration phase. The gate achieves 98% sensitivity on UCA-EHAR, ensuring no genuine activity transition is missed, while 75% specificity keeps unnecessary HAR invocations low. Results on WISDM are 97% sensitivity and 76% specificity, demonstrating robustness and flexibility to various settings. 

---
# Voice Mapping of Text-to-Speech Systems: A Metric-Based Approach for Voice Quality Assessment 

**Authors**: Huanchen Cai, Sten Ternström  

**Link**: [PDF](https://arxiv.org/pdf/2605.00861)  

**Abstract**: This study investigates voice mapping as an evaluation framework for text-to-speech (TTS) synthesis quality. The study analyzes six TTS models, including historical and recent ones. The metrics are crest factor, spectrum balance, and cepstral peak prominence (CPPs). We investigated 6 influential TTS models: Merlin, Tacotron 2, Transformer TTS, FastSpeech 2, Glow-TTS, and VITS. The results demonstrate that voice range serves as a primary indicator of model capability, with VITS showing the largest range among tested models. Glow-TTS exhibited superior performance in soft phonation, indicated by higher spectrum balance, despite limited voice range. The results showed that the CPPs values between 7-8 dB indicate natural voice quality, while with CPPs exceeding 10 dB, the speech tends to sound robotic. These findings underscore the need for voice mapping to evaluate vocal effort, and capture how TTS systems handle voice dynamic and expressiveness. 

---
# Foundation Model Guided Dual-Branch Co-Adaptation for Source-Free EEG Decoding 

**Authors**: Peiliang Gong, Han Zhang, Zhen Jiang, Chenyu Liu, Ziyu Jia, Xinliang Zhou, Daoqiang Zhang, Xiaoli Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.00857)  

**Abstract**: Source-free domain adaptation (SFDA) provides a practical solution to cross-subject EEG decoding by adapting source-pretrained models to unlabeled target domains without accessing source data. However, existing SFDA methods rely solely on the limited internal knowledge of source-pretrained models, leading to inferior cross-domain generalization and unreliable pseudo-labels. Although EEG Foundation Models (FMs) pretrained on large-scale data exhibit strong generalizability, their potential in SFDA remains largely unexplored. To this end, we propose FUSED, a Foundation-guided Source-free EEG Decoding framework that integrates a large-scale FM with a compact Specialist Model (SM) via dual-branch co-adaptation. Specifically, we introduce a Co-adaptation mechanism equipping both branches with linear and prototype views, enabling cross-branch pseudo-label generation. Additionally, we design a Consensus Filtering Mechanism that exploits the FM's inherent stability to identify high-quality samples, along with a Two-Stage Pseudo-Label Refinement scheme to suppress error accumulation through cross-branch arbitration. Finally, we calibrate the FM's decision boundaries via mutual information maximization with the SM, followed by knowledge distillation from FM to SM, forming a principled calibrate-then-distill pipeline. To our knowledge, FUSED is the first work to leverage EEG FMs within the SFDA framework for cross-subject EEG decoding. Extensive experiments across three EEG paradigms, including motor imagery, emotion recognition, and SSVEP, demonstrate consistent state-of-the-art performance, validating the effectiveness of foundation-guided synergy for robust and privacy-preserving EEG decoding. 

---
# 1BT: One-Block Transformer for EEG-Based Cognitive Workload Assessment 

**Authors**: Stefanos Gkikas, Christian Arzate Cruz, Thomas Kassiotis, Giorgos Giannakakis, Raul Fernandez Rojas, Randy Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2605.00856)  

**Abstract**: Accurate and continuous estimation of cognitive workload is fundamental to creating adaptive human-machine systems. However, designing architectures that balance representational capacity with computational efficiency has been challenging for practical deployment. This paper introduces 1BT, a One-Block Transformer for compact and efficient EEG-based cognitive workload assessment. The model aggregates multi-channel temporal sequences via a minimal latent bottleneck, using a single cross-attention module followed by lightweight self-attention. A controlled study involving 11 participants performing three cognitively diverse tasks (abstract reasoning, numerical problem-solving, and an interactive video game) was conducted with continuous EEG recordings across two workload levels. Systematic architectural analysis identifies the most compact configuration that preserves high performance, while substantially lowering computational cost. The final model achieves high workload classification performance with under 0.5 million parameters and 0.02 GFLOPs, paving the way for a design direction for real-time cognitive workload monitoring in resource-constrained settings. 

---
# Earth System Foundation Model (ESFM): A unified framework for heterogeneous data integration and forecasting 

**Authors**: Firat Ozdemir, Yun Cheng, Salman Mohebi, Fanny Lehmann, Simon Adamov, Zhenyi Zhang, Leonardo Trentini, Dana Grund, Oliver Fuhrer, Torsten Hoefler, Siddhartha Mishra, Sebastian Schemm, Benedikt Soja, Mathieu Salzmann  

**Link**: [PDF](https://arxiv.org/pdf/2605.00850)  

**Abstract**: Foundation models (FMs) for the Earth system learn statistical relationships between physical variables across massive datasets to enable versatile downstream applications through finetuning, separating them from task-specific weather models. Here, we introduce Earth System Foundation Model (ESFM), a fully open model building on the 3D Swin UNet backbone of the pioneering Aurora model. ESFM introduces extensions that increase functionality and foster adoption in climate sciences. First, the encoding scheme and training protocols have been extended to handle diverse datasets, including those containing missing values across all spatio-temporal dimensions such as satellite data, as well as station data, all under one backbone. Axial attention is introduced to capture inter-variable dependencies. As a result ESFM skillfully predicts variables in regions or on pressure levels where no data is present at the initial time, while preserving inter-variable relationships, for example between temperature, pressure, and humidity. Individual variable tokenization enables different sets of variables to be shuffled during training and simplifies the process of building extensions for new downstream tasks. Adaptive layer norm-based ensembles allow for a simple yet effective way to transform deterministic ESFM to a probabilistic FM. We present findings using dense gridded data (ERA5, CMIP6), regionally masked dense data, sparse gridded MODIS satellite data, and station data. Results demonstrate competitive or superior performance relative to state-of-the-art benchmarks. Case studies of Super Typhoon Doksuri (2023) and 2024 sudden stratospheric warming events show accurate positional and magnitude estimations of extreme weather. ESFM retains the strengths of previous foundation models, such as long-term stability, but facilitates application to a variety of downstream tasks. 

---
# H-Probes: Extracting Hierarchical Structures From Latent Representations of Language Models 

**Authors**: Cutter Dawes, Aryan Sharma, Angelos Ioannis Lagos, Shivam Raval  

**Link**: [PDF](https://arxiv.org/pdf/2605.00847)  

**Abstract**: Representing and navigating hierarchy is a fundamental primitive of reasoning. Large language models have demonstrated proficiency in a wide variety of tasks requiring hierarchical reasoning, but there exists limited analysis on how the models geometrically represent the necessary latent constructions for such thinking. To this end, we develop \textit{H-probes}, a collection of linear probes that extract hierarchical structure, specifically depth and pairwise distance, from latent representations. In synthetic tree traversal tasks, the H-probes robustly find the subspaces containing hierarchical structure necessary to complete the tasks; furthermore, in comprehensive ablation experiments, we show that these hierarchy-containing subspaces are low-dimensional, causally important for high task performance, and generalize within- and out-of-domain. Furthermore, we find analogous, though weaker, hierarchical structure in real-world hierarchical contexts such as mathematical reasoning traces. These results demonstrate that models represent hierarchy not only at the level of syntax and concepts, but at deeper levels of abstraction -- including the reasoning process itself. 

---
# Graph Query Generation with Constraint-guided Large Language Agents 

**Authors**: Mengying Wang, Nicolaas Jedema, Rahul Pandey, RaviKiran Krishnan, Jens Lehmann, Yinghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00845)  

**Abstract**: Knowledge Graph Question Answering (KGQA) has advanced through structured query generation, yet most efforts target RDF/SPARQL, leaving Cypher and property graphs underexplored, despite increasing demand for unified KGQA in industry settings. We propose UniQGen, a novel constraint-based framework that employs LLM agents to dynamically extract and refine representative graph query clauses into executable, intent-aligned graph queries across query languages. The foundation of our method is a variant of Chase & Backchase, a family of algorithms for query optimization and reformulation. We extend Chase & Backchase with a dynamic reasoning process over query constraints that also interact with LLMs for query quality estimation. With a Cypher-supported Freebase graph deployed on Amazon Neptune, we extensively evaluate our approach on popular KGQA benchmarks (GraphQ, GrailQA, and WebQSP). We demonstrate that UniQGen outperforms state-of-the-art graph query generation techniques in both accuracy and efficiency, with F1 gains of 31.6% on GraphQ and 4.9% on GrailQA. Unlike prior methods, our framework does not require fine-tuning for schema matching, making it more extensible to schema-less graphs and semantics in query workloads, and is more suitable for enterprise-grade KGQA. We release Cypher outputs and a Neptune-ready Freebase snapshot to support reproducible, cross-language KGQA research. 

---
# The Oracle's Fingerprint: Correlated AI Forecasting Errors and the Limits of Bias Transmission 

**Authors**: Theodor Spiro  

**Link**: [PDF](https://arxiv.org/pdf/2605.00844)  

**Abstract**: When large language models (LLMs) are consulted as forecasting tools, the independence of individual errors -- the foundation of collective intelligence -- may collapse. We test three conditions necessary for this "epistemic monoculture" to emerge. In Study 1, we show that GPT-4o, Claude, and Gemini exhibit highly correlated forecasting errors on 568 resolved binary prediction questions (mean pairwise error correlation r = 0.77, p < 0.001; r = 0.78 excluding likely-leaked questions), despite being developed independently by different organizations. In Study 2, we test whether this correlated bias has propagated into human crowd forecasts, using a within-question design that tracks community prediction shifts across the ChatGPT launch boundary (November 2022). We find that community forecasts move in the direction predicted by LLMs (r = 0.20, p = 0.007), but this shift is fully explained by rational updating toward ground truth. In Study 3, we examine whether the category-level pattern of human forecasting errors increasingly resembles the LLM bias fingerprint. We find the opposite: pre-ChatGPT human biases already strongly resembled the LLM pattern (r = 0.87), while post-ChatGPT the resemblance weakened (r = -0.28). Together, these findings reveal an epistemic monoculture that is built but not yet activated: three nominally independent AI systems share the same failure modes, amplifying precisely the biases humans already hold. 

---
# Generative-AI and the transformation of workforce. A job postings-driven analysis 

**Authors**: Diana Maria Popa, Simona-Vasilica Oprea, Adela Bâra  

**Link**: [PDF](https://arxiv.org/pdf/2605.00843)  

**Abstract**: This paper investigates how generative-artificial intelligence AI is reshaping job requirements, skill compositions and sectoral dynamics across global labor markets. It examines the evolving frequency and framing of AI-related competencies in job postings, exploring whether generative-AI functions primarily as an augmentative or substitutive force in the workplace. A large-scale, multi-source corpus of over 150,000 English-language job postings 2018-2025 is compiled from twelve open-access datasets and one public API. The analytical framework integrates lexical skill extraction, semantic framing, topic modeling, BERTopic, LDA, KMeans, and time-series forecasting ARIMA. Skill mentions are categorized into five dimensions: AI_Data, Routine, Soft_Meta, Domain_Specific and Leadership, while cross sectoral analyses and correlation matrices quantify interdependencies between competencies. Sentence-transformer embeddings and cosine similarity are used to compute a Framing Index, distinguishing augmentation- versus automation-oriented discourse. Investigating job postings, our research contributes a replicable, data driven methodology for mapping the diffusion of AI related skills across industries and time. Results reveal a sharp post-2021 increase in AI-related skill mentions: prompt engineering, fine-tuning and model validation, accompanied by a decline in routine tasks: data entry and manual coding. Forecasts suggest sustained growth in AI_Data and Soft_Meta skills through 2025, signaling a structural convergence toward hybrid human-AI expertise as a new foundation of employability. 

---
# Agentopic: A Generative AI Agent Workflow for Explainable Topic Modeling 

**Authors**: Brice Valentin Kok-Shun, Johnny Chan, Gabrielle Peko, David Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2605.00833)  

**Abstract**: Agentopic is a novel agent-based workflow for explainable topic modeling that leverages the reasoning capabilities of Large Language Models (LLMs). Existing topic modeling approaches such as Latent Dirichlet Allocation (LDA) and BERTopic often lack transparency on how topics are assigned or grouped. Agentopic addresses this by using multiple agents that collaboratively perform topic identification, validation, hierarchical grouping, and natural language explanation. This design enables users to trace the reasoning behind topic assignments, enhancing interpretability without sacrificing accuracy. When seeded with topics from the British Broadcasting Corporation (BBC) dataset, Agentopic achieves an F1-score of 0.95, matching GPT-4.1, improving on LDA (0.93), and close to BERTopic (0.98). We used Agentopic to augment the BBC dataset with generated explanations to improve the dataset's richness and context. The unseeded Agentopic generated 2045 semantically coherent topics organized across six hierarchical levels, vastly enriching the original five-category structure. By embedding explainability throughout the workflow, Agentopic offers an interpretable alternative to black-box models, making it particularly valuable for crucial applications like finance and healthcare. 

---
# GhostServe: A Lightweight Checkpointing System in the Shadow for Fault-Tolerant LLM Serving 

**Authors**: Shakya Jayakody, Youpeng Zhao, Chinmay Dhanraj Nehate, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00831)  

**Abstract**: The rise of million-token, agent-based applications has placed unprecedented demands on large language model (LLM) inference services. The long-running nature of these tasks increases their susceptibility to hardware and software faults, leading to costly job failures, wasted resources, and degraded user experience. The stateful key-value (KV) cache, which grows with the sequence length, presents a central challenge as it is a critical and vulnerable component in distributed serving systems. In this work, we propose GhostServe, a novel checkpointing solution to facilitate fault-tolerant LLM serving. Specifically, GhostServe protects the streaming KV cache in the shadow by applying erasure coding to generate and store the parity shards in host memory. In the event of device failures, GhostServe enables fast reconstruction of the lost KV cache, allowing the inference process to resume seamlessly without costly full recomputation or state replication. Evaluations demonstrate that GhostServe reduces checkpointing latency by up to 2.7x and recovery latency by 2.1x for a single batch, and 1.2x median response latency compared to existing methods, in the presence of system failures, paving the way for high-availability and cost-effective LLM serving at scale. 

---
# Separating Intelligence from Execution: A Workflow Engine for the Model Context Protocol 

**Authors**: Abhinav Singh Parmar  

**Link**: [PDF](https://arxiv.org/pdf/2605.00827)  

**Abstract**: Large Language Model (LLM) agents increasingly interact with external systems through tool-calling protocols such as the Model Context Protocol (MCP). In prevailing architectures, the agent must reason about every tool invocation in every session, consuming tokens proportional to the number of actions performed--even when the task has been solved before. We present the MCP Workflow Engine, a novel MCP-native orchestration layer that decouples intelligence (deciding what to do) from execution (carrying it out). An agent reasons once to produce a declarative workflow blueprint--a JSON document specifying a directed sequence of MCP tool calls with parameterized templates, loops, parallel branches, and data piping. Subsequent executions are triggered by a single run_workflow tool call, consuming one invocation's worth of tokens regardless of the blueprint's internal complexity. We formalize the MCP Mediator architectural pattern--an MCP server that simultaneously acts as a client to downstream MCP servers--and implement it in TypeScript against the MCP SDK. We evaluate the engine on a production-scale Kubernetes CMDB synchronization task spanning 67 orchestrated steps across 2 MCP servers, 38 namespaces, 13 worker nodes, and 22 distinct resource types. The engine reduces per-execution token cost by over 99%, completes the full cluster graph--comprising 1,200+ nodes and 2,800+ relationships across 20 relationship types--in under 45 seconds, and achieves deterministic, idempotent execution with zero agent involvement at run time. 

---
