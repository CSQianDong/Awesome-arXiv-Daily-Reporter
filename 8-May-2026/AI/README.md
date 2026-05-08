# AI Co-Mathematician: Accelerating Mathematicians with Agentic AI 

**Authors**: Daniel Zheng, Ingrid von Glehn, Yori Zwols, Iuliya Beloshapka, Lars Buesing, Daniel M. Roy, Martin Wattenberg, Bogdan Georgiev, Tatiana Schmidt, Andrew Cowie, Fernanda Viegas, Dimitri Kanevsky, Vineet Kahlon, Hartmut Maennel, Sophia Alj, George Holland, Alex Davies, Pushmeet Kohli  

**Link**: [PDF](https://arxiv.org/pdf/2605.06651)  

**Abstract**: We introduce the AI co-mathematician, a workbench for mathematicians to interactively leverage AI agents to pursue open-ended research. The AI co-mathematician is optimized to provide holistic support for the exploratory and iterative reality of mathematical workflows, including ideation, literature search, computational exploration, theorem proving and theory building. By providing an asynchronous, stateful workspace that manages uncertainty, refines user intent, tracks failed hypotheses, and outputs native mathematical artifacts, the system mirrors human collaborative workflows. In early tests, the AI co-mathematician helped researchers solve open problems, identify new research directions, and uncover overlooked literature references. Besides demonstrating a highly interactive paradigm for AI-assisted mathematical discovery, the AI co-mathematician also achieves state of the art results on hard problem-solving benchmarks, including scoring 48% on FrontierMath Tier 4, a new high score among all AI systems evaluated. 

---
# GlazyBench: A Benchmark for Ceramic Glaze Property Prediction and Image Generation 

**Authors**: Ziyu Zhai, Siyou Li, Juexi Shao, Juntao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06641)  

**Abstract**: Developing ceramic glazes is a costly, time-consuming process of trial and error due to complex chemistry, placing a significant burden on independent artists. While recent advances in multimodal AI offer a modern solution, the field lacks the large-scale datasets required to train these models. We propose GlazyBench, the first dataset for AI-assisted glaze design. Comprising 23,148 real glaze formulations, GlazyBench supports two primary tasks: predicting post-firing surface properties, such as color and transparency, from raw materials, and generating accurate visual representations of the glaze based on these properties. We establish comprehensive baselines for property prediction using traditional machine learning and large language models, alongside image generation benchmarks using deep generative and large multimodal models. Our experiments demonstrate promising yet challenging results. GlazyBench pioneers a new research direction in AI-assisted material design, providing a standardized benchmark for systematic evaluation. 

---
# Can RL Teach Long-Horizon Reasoning to LLMs? Expressiveness Is Key 

**Authors**: Tianle Wang, Zhaoyang Wang, Guangchen Lan, Xinpeng Wei, Sipeng Zhang, Guanwen Qiu, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2605.06638)  

**Abstract**: Reinforcement learning (RL) has been applied to improve large language model (LLM) reasoning, yet the systematic study of how training scales with task difficulty has been hampered by the lack of controlled, scalable environments. We introduce ScaleLogic, a synthetic logical reasoning framework that offers independent control over two axes of difficulty: the depth of the required proof planning (i.e., the horizon) and the expressiveness of the underlying logic. Our proposed framework supports a wide range of logics: from simple implication-only logic ("if-then") towards more expressive first-order reasoning with conjunction ("and"), disjunction ("or"), negation ("not"), and universal quantification ("for all"). Using this framework, we show that the RL training compute $T$ follows a power law with respect to reasoning depth $D$ ($T \propto D^{\gamma}$, $R^{2} > 0.99$), and that the scaling exponent $\gamma$ increases monotonically with logical expressiveness, from $1.04$ to $2.60$. On downstream mathematics and general reasoning benchmarks, more expressive training settings yield both larger performance gains (up to $+10.66$ points) and more compute-efficient transfer compared to less expressive settings, demonstrating that what a model is trained on, not just how much it is trained, shapes downstream transfer. We further show that the power-law relationship holds across multiple RL methods, and curriculum-based training substantially improves scaling efficiency. 

---
# MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems 

**Authors**: Zhexuan Wang, Xuebo Liu, Li Wang, Zifei Shan, Yutong Wang, Zhenxi Song, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06623)  

**Abstract**: Large language model (LLM)-based Multi-agent systems (MAS) have shown promise in tackling complex collaborative tasks, where agents are typically orchestrated via role-specific prompts. While the quality of these prompts is pivotal, jointly optimizing them across interacting agents remains a non-trivial challenge, primarily due to the misalignment between local agent objectives and holistic system goals. To address this, we introduce MASPO, a novel framework designed to automatically and iteratively refine prompts across the entire system. A core innovation of MASPO is its joint evaluation mechanism, which assesses prompts not merely by their local validity, but by their capacity to facilitate downstream success for successor agents. This effectively bridges the gap between local interactions and global outcomes without relying on ground-truth labels. Furthermore, MASPO employs a data-driven evolutionary beam search to efficiently navigate the high-dimensional prompt space. Extensive empirical evaluations across 6 diverse tasks demonstrate that MASPO consistently outperforms state-of-the-art prompt optimization methods, achieving an average accuracy improvement of 2.9. We release our code at this https URL. 

---
# SkillOS: Learning Skill Curation for Self-Evolving Agents 

**Authors**: Siru Ouyang, Jun Yan, Yanfei Chen, Rujun Han, Zifeng Wang, Bhavana Dalvi Mishra, Rui Meng, Chun-Liang Li, Yizhu Jiao, Kaiwen Zha, Maohao Shen, Vishy Tirumalashetty, George Lee, Jiawei Han, Tomas Pfister, Chen-Yu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.06614)  

**Abstract**: LLM-based agents are increasingly deployed to handle streaming tasks, yet they often remain one-off problem solvers that fail to learn from past interactions. Reusable skills distilled from experience provide a natural substrate for self-evolution, where high-quality skill curation serves as the key bottleneck. Existing approaches either rely on manual skill curation, prescribe heuristic skill operations, or train for short-horizon skill operations. However, they still struggle to learn complex long-term curation policies from indirect and delayed feedback. To tackle this challenge, we propose SkillOS, an experience-driven RL training recipe for learning skill curation in self-evolving agents. SkillOS pairs a frozen agent executor that retrieves and applies skills with a trainable skill curator that updates an external SkillRepo from accumulated experience. To provide learning signals for curation, we design composite rewards and train on grouped task streams based on skill-relevant task dependencies, where earlier trajectories update the SkillRepo, and later related tasks evaluate these updates. Across multi-turn agentic tasks and single-turn reasoning tasks, SkillOS consistently outperforms memory-free and strong memory-based baselines in both effectiveness and efficiency, with the learned skill curator generalizing across different executor backbones and task domains. Further analyses show that the learned curator produces more targeted skill use, while the skills in SkillRepo evolve into more richly structured Markdown files that encode higher-level meta-skills over time. 

---
# NeuroAgent: LLM Agents for Multimodal Neuroimaging Analysis and Research 

**Authors**: Lujia Zhong, Yihao Xia, Jianwei Zhang, Shuo huang, Jiaxin Yue, Mingyang Xia, Yonggang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.06584)  

**Abstract**: Multimodal neuroimaging analysis often involves complex, modality-specific preprocessing workflows that require careful configuration, quality control, and coordination across heterogeneous toolchains. Beyond preprocessing, downstream statistical analysis and disease classification commonly require task-specific code, evaluation protocols, and data-format conventions, creating additional barriers between raw acquisitions and reproducible scientific analysis. We present NeuroAgent, an LLM-driven agentic framework that automates key preprocessing and analysis steps for heterogeneous neuroimaging data, including sMRI, fMRI, dMRI, and PET, and supports interactive downstream analysis through natural-language queries. NeuroAgent employs a hierarchical multi-agent architecture with a feedback-driven Generate-Execute-Validate engine: agents autonomously generate executable preprocessing code, detect and recover from runtime errors, and validate output integrity. We evaluate the system on 1,470 subjects pooled across all ADNI phases (CN=1,000, AD=470), where all subjects have sMRI and tabular data, with subsets also having Tau-PET (n=469), fMRI (n=278), and DTI ($n=620$). Pipeline ablation studies across multiple LLM backends show that capable models reach up to 100% intent-parsing accuracy, with the strongest backend (Qwen3.5-27B) reaching 84.8% end-to-end preprocessing step correctness. Automated recovery limits manual intervention to edge cases where human review is required via the Human-In-The-Loop interface. For Alzheimer's Disease classification using automatically preprocessed multimodal data, our agent ensemble achieves an AUC of 0.9518 with four modalities, outperforming all single-modality baselines. These results show that NeuroAgent can reduce the manual effort required for neuroimaging preprocessing and enable end-to-end automated analysis pipelines for neuroimaging research. 

---
# Improved techniques for fine-tuning flow models via adjoint matching: a deterministic control pipeline 

**Authors**: Zhengyi Guo, Jiayuan Sheng, David D. Yao, Wenpin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06583)  

**Abstract**: We propose a deterministic adjoint matching framework that formulates human preference alignment for flow-based generative models as an optimal control problem over velocity fields. One can directly regress the control toward a value-gradient-induced target under the current policy, leading to a simple and stable training objective. Building on this perspective, we introduce a truncated adjoint scheme that focuses computation on the terminal portion of the trajectory, where reward-relevant signals concentrate, which yields substantial computational savings while preserving alignment quality. We further generalize the framework beyond standard KL-based regularization, allowing more flexible trade-offs between alignment strength and distributional preservation. Experiments on SiT-XL/2 and FLUX.2-Klein-4B demonstrate consistent gains across multiple alignment metrics, along with substantially improved diversity and mode preservation. 

---
# Ex Ante Evaluation of AI-Induced Idea Diversity Collapse 

**Authors**: Nafis Saami Azad, Raiyan Abdul Baten  

**Link**: [PDF](https://arxiv.org/pdf/2605.06540)  

**Abstract**: Creative AI systems are typically evaluated at the level of individual utility, yet creative outputs are consumed in populations: an idea loses value when many others produce similar ones. This creates an evaluation blind spot, as AI can improve individual outputs while increasing population-level crowding. We introduce a human-relative framework for benchmarking AI-induced human diversity collapse without requiring human-AI interaction data, providing an ex ante protocol to estimate crowding risk from model-only generations and matched unaided human baselines. By modeling ideas as congestible resources, we show that source-level crowding is identifiable from within-distribution comparisons, yielding an excess-crowding coefficient $\Delta$ and a human-relative diversity ratio $\rho$. We show that $\rho\ge1$ is the no-excess-crowding parity condition and connect $\Delta$ to an adoption game with exposure-dependent redundancy costs. Across short stories, marketing slogans, and alternative-uses tasks, three frontier LLMs fall below parity across crowding kernels. Estimates stabilize with feasible model-only sample sizes. Importantly, generation-protocol variants show that crowding can be reduced through targeted design, making diversity collapse an actionable, development-time evaluation target for population-aware creative AI. 

---
# SpatialEpiBench: Benchmarking Spatial Information and Epidemic Priors in Forecasting 

**Authors**: Ruiqi Lyu, Alistair Turcan, Bryan Wilder  

**Link**: [PDF](https://arxiv.org/pdf/2605.06530)  

**Abstract**: Accurate epidemic forecasting is crucial for public health response, resource allocation, and outbreak intervention, but remains difficult with sparse, noisy, and highly non-stationary data. Because epidemics unfold across interacting regions, spatiotemporal methods are natural candidates for improving forecasts. Despite growing interest in spatial information, no standardized benchmark exists, and current evaluations often use simple chronological train-test splits that do not reflect real-time forecasting practice. We address this gap with SpatialEpiBench, a challenging benchmark for spatiotemporal epidemic forecasting in realistic public-health settings. SpatialEpiBench includes 11 epidemic datasets with standardized rolling evaluations and outbreak-specific metrics. We evaluate adjacency-informed forecasting models with widely used epidemic priors that adapt general models to epidemiology, but find that most methods underperform a simple last-value baseline from 1 day to 1 month ahead, even during outbreaks and with these priors. We identify three major failure modes: (1) poor outbreak anticipation, (2) difficulty handling sparsity and noise, and (3) limited utility of common geographic adjacency for epidemiological spatial information. We release benchmark data, code, and instructions at this https URL to support development of operationally useful epidemic forecasting models. 

---
# Market-Alignment Risk in Pricing Agents: Trace Diagnostics and Trace-Prior RL under Hidden Competitor State 

**Authors**: Peiying Zhu, Sidi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06529)  

**Abstract**: Outcome metrics can certify the wrong behavior. We study this failure in a two-hotel revenue-management simulator where Hotel A trains an agent against a fixed rule-based revenue-management competitor, Hotel B. A standard learning agent can obtain near-reference revenue per available room (RevPAR) while failing to learn market-like yield management: it sells too aggressively, undercuts, or collapses to modal price buckets. We diagnose this as a Goodhart-style failure under partial observability. Hotel A cannot observe the competitor's remaining inventory, booking curve, or pricing rule, so the same Hotel A-visible state maps to multiple plausible Hotel B prices. Deterministic value-based RL and deterministic copying collapse this unresolved uncertainty into shortcut behavior. We introduce a trace-level diagnostic protocol using RevPAR, occupancy, ADR, full price-bucket distributions, L1/JS distances, and seed-level confidence intervals. The verified repair is Trace-Prior RL: learn a distributional market prior from lagged market traces, then train a stochastic pricing policy with a RevPAR reward and a KL penalty to the learned prior. The final policy matches Hotel B's RevPAR, occupancy, ADR, and price distribution within seed-level uncertainty, while still optimizing Hotel A's own reward. We argue that the contribution is not a new optimizer and not a hotel-pricing leaderboard, but a reproducible failure-and-repair recipe for agentic systems where scalar rewards are easy to game and the intended behavior is only visible in traces. A key finding is that higher exact action accuracy can worsen aggregate trace alignment when the target is distributional. 

---
# Process Matters more than Output for Distinguishing Humans from Machines 

**Authors**: Milena Rmus, Mathew D. Hardy, Thomas L. Griffiths, Mayank Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2605.06524)  

**Abstract**: Reliable human-machine discrimination is becoming increasingly important as large language models and autonomous agents are deployed in online settings. Existing approaches evaluate whether a system can produce behavior or responses indistinguishable from those of a human, following the emphasis on outputs as a criterion for intelligence proposed by Alan Turing. Cognitive science offers an alternative perspective: evaluating the process by which behavior is produced. To test whether cognitive processes can reliably distinguish humans from machines, we introduce CogCAPTCHA30, a battery of 30 cognitive tasks designed to elicit diagnostic process-level features even when task performance is matched. Across the battery, process-level features provide stronger discriminative signal than performance metrics alone, reliably distinguishing humans from agents even under output matching (mean process-feature classifier AUC = 0.88). To evaluate agentic process differences, we compare off-the-shelf frontier agents (Claude Sonnet 4.5, GPT-5, Gemini 2.5 Pro), Centaur (a language model fine-tuned on 10.7M human decisions), and two task-specific fine-tuning approaches applied to Qwen2.5-1.5B-Instruct: action-level supervised fine-tuning (A-SFT) and process-level fine-tuning (P-SFT), which directly optimizes process features. Broad fine-tuning on human decisions improves human-like task processes relative to off-the-shelf agents, while task-specific process-level supervision further improves behavioral mimicry. However, this advantage diminishes under cross-task transfer when supervised process targets do not naturally generalize across tasks. Explicit process-level supervision can improve human behavioral mimicry, but only if appropriate task-specific process representations are available, highlighting process specification as a bottleneck for achieving human-like cognitive processes in machines. 

---
# From Token Lists to Graph Motifs: Weisfeiler-Lehman Analysis of Sparse Autoencoder Features 

**Authors**: Ruben Fernandez-Boullon, Pablo Magariños-Docampo, Javier Perez-Robles  

**Link**: [PDF](https://arxiv.org/pdf/2605.06494)  

**Abstract**: Sparse autoencoders (SAEs) have become central to mechanistic interpretability, decomposing transformer activations into monosemantic features. Yet existing analyses characterise features almost exclusively through top-activating token lists or decoder weight vectors, leaving the higher-order co-occurrence structure shared across features largely unexamined. We introduce a graph-structured representation in which each SAE feature is modelled as a token co-occurrence graph: nodes are the tokens most frequent near strong activations, and edges connect pairs that co-occur within local context windows. A custom WL-style, frequency-binned graph kernel then provides a similarity measure over this structural space. Applied as a proof of concept to features from a large SAE trained on GPT-2 Small and probed with a synthetic mixed-domain corpus, our clustering recovers heuristic motif families (punctuation-heavy patterns, language and script clusters, and code-like templates) that are not recovered by clustering on decoder cosine similarity. A token-histogram baseline achieves higher overall purity, so the contribution of the graph view is complementary rather than dominant: it surfaces structural relationships that token-frequency and decoder-weight views alone do not capture. Cluster assignments are stable across graph-construction hyperparameters and random seeds. 

---
# Instrumental Choices: Measuring the Propensity of LLM Agents to Pursue Instrumental Behaviors 

**Authors**: Jonas Wiedermann-Möller, Leonard Dung, Maksym Andriushchenko  

**Link**: [PDF](https://arxiv.org/pdf/2605.06490)  

**Abstract**: AI systems have become increasingly capable of dangerous behaviours in many domains. This raises the question: Do models sometimes choose to violate human instructions in order to perform behaviour that is more useful for certain goals? We introduce a benchmark for measuring model propensity for instrumental convergence (IC) behaviour in terminal-based agents. This is behaviour such as self-preservation that has been hypothesised to play a key role in risks from highly capable AI agents. Our benchmark is realistic and low-stakes which serves to reduce evaluation-awareness and roleplay confounds. The suite contains seven operational tasks, each with an official workflow and a policy-violating shortcut. An eight-variant shared framework varies monitoring, instruction clarity, stakes, permission, instrumental usefulness and blocked honest paths to support inferences regarding the factors driving IC behaviour. We evaluated ten models using deterministic environment-state scorers over 1,680 samples, with trace review employed for audit and adjudication purposes. The final IC rate is 86 out of 1,680 samples (5.1%). IC behaviour is concentrated rather than uniform: two Gemini models account for 66.3% of IC cases and three tasks account for 84.9%. Conditions in which IC behaviour is indispensable for task success result in the greatest increase in the adjusted IC rate (+15.7 percentage points), whereas emphasising that task success is critical or certain framing choices do not produce comparable effects. Our findings indicate that realistic, low-nudge environments elicit IC behaviour rarely but systematically in most tested models. We conclude that it is feasible to robustly measure tendencies for dangerous behaviour in current frontier AI agents. 

---
# ReasonSTL: Bridging Natural Language and Signal Temporal Logic via Tool-Augmented Process-Rewarded Learning 

**Authors**: Bowen Ye, Zhijian Li, Junyue Huang, Junkai Ma, Xiang Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06483)  

**Abstract**: Signal Temporal Logic (STL) is an expressive formal language for specifying spatio-temporal requirements over real-valued, real-time signals. It has been widely used for the verification and synthesis of autonomous systems and cyber-physical systems. In practice, however, users often express their requirements in natural language rather than in structured STL formulas, making natural-language-to-STL translation a critical yet challenging task. Manual specification requires temporal-logic expertise and cannot scale, while prompting commercial LLM APIs incurs substantial token costs and may expose sensitive system requirements to third-party services, raising privacy concerns for industrial deployment. To address these challenges, we present \textsc{ReasonSTL}, a tool-augmented framework that adapts local open-source language models for natural-language-to-STL generation. \textsc{ReasonSTL} decomposes the translation process into explicit reasoning, deterministic tool calls, and structured formula construction. We further introduce process-rewarded training to supervise both tool-use trajectories and final formulas, together with \textsc{STL-Bench}, a bilingual, computation-aware benchmark grounded in real-world signals. Experiments show that a 4B model trained with \textsc{ReasonSTL} achieves state-of-the-art performance in both automatic metrics and human evaluations, demonstrating that \textsc{ReasonSTL} provides a transparent, low-cost, and privacy-preserving alternative for formal specification drafting. 

---
# Patch-Effect Graph Kernels for LLM Interpretability 

**Authors**: Ruben Fernandez-Boullon, David N. Olivieri  

**Link**: [PDF](https://arxiv.org/pdf/2605.06480)  

**Abstract**: Mechanistic interpretability aims to reverse-engineer transformer computations by identifying causal circuits through activation patching. However, scaling these interventions across diverse prompts and task families produces high-dimensional, unstructured datasets that are difficult to compare systematically. We propose a framework that reframes mechanistic analysis as a graph machine-learning problem by representing activation-patching profiles as patch-effect graphs over model components. We introduce three graph-construction methods: direct-influence via causal mediation, partial-correlation, and co-influence and apply graph kernels to analyze the resulting structures. Evaluating this approach on GPT-2 Small using Indirect Object Identification (IOI) and related tasks, we find that patch-effect graphs preserve discriminative structural signals. Specifically, localized edge-slot features provide higher classification accuracy than global graph-shape descriptors. A screened paired-patching validation suggests that CI and PC selected candidate edges correspond to stronger activation-influence effects than random or low-rank candidates. Crucially, by evaluating these representations against rigorous prompt-only and raw patch-effect controls, we make the evidential scope of the benchmark explicit: graph features compress structured patching signal, while raw tensors and surface cues define strong baselines that any circuit-level claim should address. Ultimately, our framework provides a compression and evaluation pipeline for comparing patching-derived structures under controlled baselines, separating robust slice-discriminative evidence from stronger task-general causal-circuit claims. 

---
# Probabilistic Dating of Historical Manuscripts via Evidential Deep Regression on Visual Script Features 

**Authors**: Ranjith Chodavarapu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06475)  

**Abstract**: We introduce a probabilistic approach for dating historical manuscript pages from visual features alone. Instead of aggregating centuries into classes as is standard in the previous literature, we pose dating as an evidential deep regression problem over a continuous year axis, allowing our neural network to output a full predictive distribution with decomposed aleatoric and epistemic uncertainty in a single forward pass. Our architecture combines an EfficientNet-B2 backbone with a Normal-Inverse-Gamma (NIG) output head trained with a joint negative-log-likelihood and evidence-regularization objective. On the DIVA-HisDB benchmark (150 pages, 3 medieval codices, 151,936 patches), our model scores a test MAE of 5.4 years, well below the 50-year century-label supervision granularity, with 93\% of patches within 5 years and 97\% within 10 years. Our approach achieves \textbf{PICP=92.6\%}, the best calibration among all compared methods, in a single forward pass, outperforming MC Dropout (PICP=88.2\%, 50 passes) and Deep Ensembles (PICP=79.7\%, 5 models) at $5\times$ lower inference cost. Uncertainty decomposition shows aleatoric uncertainty is a strong predictor of dating error (Spearman $\rho=0.729$), and a selective prediction about the most certain 20\% of patches can provide \textbf{0.5 years MAE}. We show that predicted uncertainty increases as image degradation worsens, spatial decomposition maps explain which script regions cause aleatoric uncertainty, and page-level aggregation reduces MAE to 4.5 years with $\rho=0.905$ between uncertainty and page-level error. 

---
# Beyond Task Success: Measuring Workflow Fidelity in LLM-Based Agentic Payment Systems 

**Authors**: Donghao Huang, Joon Kiat Chua, Zhaoxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06457)  

**Abstract**: LLM-based multi-agent systems are increasingly deployed for payment workflows, yet prevailing metrics, Task Success Rate (TSR) and Agent Handoff F1-Score (HF1), capture only final outcomes or unordered routing decisions. We introduce the Agentic Success Rate (ASR), a trajectory-fidelity metric that compares observed and expected agent execution sequences at the transition level, decomposing performance into Transition Recall and Transition Precision. Applied to the Hierarchical Multi-Agent System for Payments (HMASP) across 18 LLMs and 90,000 task instances, ASR reveals that 10 of 18 models systematically skip a confirmation checkpoint during payment checkout, a deviation invisible to both TSR and HF1, while 8 models enforce the checkpoint perfectly. Notably, GPT-4.1 exhibits hidden workflow shortcuts despite achieving perfect TSR and HF1, while GPT-5.2 achieves perfect ASR. Prompt refinements and deterministic routing guards guided by ASR diagnostics yield substantial TSR improvements, with gains up to +93.8 percentage points for previously struggling models, demonstrating that trajectory-level evaluation is essential in regulated domains. 

---
# PrefixGuard: From LLM-Agent Traces to Online Failure-Warning Monitors 

**Authors**: Xinmiao Huang, Jinwei Hu, Rajarshi Roy, Changshun Wu, Yi Dong, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06455)  

**Abstract**: Large language model (LLM) agents now execute long, tool-using tasks where final outcome checks can arrive too late for intervention. Online warning requires lightweight prefix monitors over heterogeneous traces, but hand-authored event schemas are brittle and deployment-time LLM judging is costly. We introduce PrefixGuard, a trace-to-monitor framework with an offline StepView induction step followed by supervised monitor training. StepView induces deterministic typed-step adapters from raw trace samples, and the monitor learns an event abstraction and prefix-risk scorer from terminal outcomes. Across WebArena, $\tau^2$-Bench, SkillsBench, and TerminalBench, the strongest PrefixGuard monitors reach 0.900/0.710/0.533/0.557 AUPRC. Using the strongest backend within each representation, they improve over raw-text controls by an average of +0.137 AUPRC. LLM judges remain substantially weaker under the same prefix-warning protocol. We also derive an observability ceiling on score-based area under the precision-recall curve (AUPRC) that separates monitor error from failures lacking evidence in the observed prefix. For finite-state audit, post-hoc deterministic finite automaton (DFA) extraction remains compact on WebArena and $\tau^2$-Bench (29 and 20 states) but expands to 151 and 187 states on SkillsBench and TerminalBench. Finally, first-alert diagnostics show that strong ranking does not imply deployment utility: WebArena ranks well yet fails to support low-false-alarm alerts, whereas $\tau^2$-Bench and TerminalBench retain more actionable early alerts. Together, these results position PrefixGuard as a practical monitor-synthesis recipe with explicit diagnostics for when prefix warnings translate into actionable interventions. 

---
# SCRuB: Social Concept Reasoning under Rubric-Based Evaluation 

**Authors**: Jamelle Watson-Daniels, Himaghna Bhattacharjee, Skyler Wang, Brandon Handoko, Antonio Li, Anaelia Ovalle, Mahesh Pasupuleti, Candace Ross, Vidya Sarma, Arjun Subramonian, Karen Ullrich, Will van der Vaart, Yijing Xin, Maximilian Nickel  

**Link**: [PDF](https://arxiv.org/pdf/2605.06444)  

**Abstract**: While many studies of Large Language Model (LLM) reasoning capabilities emphasize mathematical or technical tasks, few address reasoning about social concepts: the abstract ideas shaping social norms, culture, and institutions. This understudied capability is essential for modern models acting as social agents, yet no systematic evaluation methodology targets it. We introduce SCRuB (Social Concept Reasoning under Rubric-Based Evaluation), a framework designed for this setting of task indeterminacy. Our goal is to measure the degree to which a model reasons about social concepts with the depth and critical rigor of a human expert. SCRuB proceeds in three phases: prompt construction from established sources, response generation by experts and models, and comparative evaluation using a five-dimensional critical thinking rubric. To enable generalization of the pipeline, we introduce a Panel of Disciplinary Perspectives ensemble validated against independent expert judges. We release SCRuBEval (n=4,711 evaluation prompts) and SCRuBAnnotations (300 expert-authored responses and 150 expert comparative judgments from 45 PhD-level scholars). Our results show that frontier models consistently outperform human experts across all five rubric dimensions. Across 1,170 pairwise comparisons, expert judges ranked a model response first in 80.8% of judgments and preferred model responses overall 74.4% of the time. Ultimately, this study provides the first expert-grounded demonstration of evaluation saturation for social concept reasoning: the single-turn exam-style format has reached its ceiling for models and humans alike. 

---
# Knowledge Graphs, the Missing Link in Agentic AI-based Formal Verification 

**Authors**: Vaisakh Naduvodi Viswambharan, Keerthan Kopparam Radhakrishna, Deepak Narayan Gadde, Aman Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.06434)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled workflows that generate SystemVerilog Assertions (SVAs) from natural-language specifications, with the potential to accelerate Formal Verification (FV). However, high-quality assertion synthesis remains challenging because specifications are often ambiguous or incomplete and critical micro-architectural details reside in the Register Transfer Level (RTL). Many existing approaches treat the specification and RTL as loosely structured text, which weakens specification-to-RTL grounding and leads to semantic mismatches and frequent syntax failures during formal parsing and elaboration. This work addresses these limitations with a verification-centric Knowledge Graph (KG) constructed from structured Intermediate Representations (IRs) extracted from the specification, RTL, and formal-tool feedback, including syntax diagnostics, Counterexamples (CEXs), and coverage reports. The KG links requirements, design hierarchy, signals, assumptions, and properties to provide traceable, design-grounded context for generation. A multi-agent workflow queries and updates this KG to generate SVAs and to drive three refinement loops: syntax repair guided by tool diagnostics, CEX-guided correction using trace links, and coverage-directed property augmentation. Evaluation across seven benchmark designs indicates that KG-based context retrieval improves specification-to-RTL grounding and consistently produces compilable SVAs with low syntax-repair overhead. The approach achieves formal coverage ranging from 78.5% to 99.4%, though convergence exhibits design dependence with complex temporal and arithmetic reasoning remaining challenging for current LLM capabilities. 

---
# Automated alignment is harder than you think 

**Authors**: Aleksandr Bowkis, Marie Davidsen Buhl, Jacob Pfau, Geoffrey Irving  

**Link**: [PDF](https://arxiv.org/pdf/2605.06390)  

**Abstract**: A leading proposal for aligning artificial superintelligence (ASI) is to use AI agents to automate an increasing fraction of alignment research as capabilities improve. We argue that, even when research agents are not scheming to deliberately sabotage alignment work, this plan could produce compelling but catastrophically misleading safety assessments resulting in the unintentional deployment of misaligned AI. This could happen because alignment research involves many hard-to-supervise fuzzy tasks (tasks without clear evaluation criteria, for which human judgement is systematically flawed). Consequently, research outputs will contain systematic, undetected errors, and even correct outputs could be incorrectly aggregated into overconfident safety assessments. This problem is likely to be worse for automated alignment research than for human-generated alignment research for several reasons: 1) optimisation pressure means agent-generated mistakes are concentrated among those that human reviewers are least likely to catch; 2) agents are likely to produce errors that do not resemble human mistakes; 3) AI-generated alignment solutions may involve arguments humans cannot evaluate; and 4) shared weights, data and training processes may make AI outputs more correlated than human equivalents. Therefore, agents must be trained to reliably perform hard-to-supervise fuzzy tasks. Generalisation and scalable oversight are the leading candidates for achieving this but both face novel challenges in the context of automated alignment. 

---
# Rethinking Vacuity for OOD Detection in Evidential Deep Learning 

**Authors**: Claire McNamara  

**Link**: [PDF](https://arxiv.org/pdf/2605.06382)  

**Abstract**: Vacuity, or Uncertainty Mass (UM), is commonly used as a metric to evaluate Out-of-Distribution (OOD) detection in Evidential Deep Learning (EDL). It generally involves dividing the number of classes ($K$) by the total strength of belief ($S$) of the model's predictions, where $S$ is derived from summing the Dirichlet parameters. As such, UM is sensitive to the cardinality of $K$. In particular, it is unlikely in practice that there is a linear relationship between $K$ and $S$ as $K$ and $S$ increase due to the nature of EDL (suppressing incorrectly assigned evidence). As a result, when comparing In Distribution (ID) and OOD results, it is important that $K_{\mathrm{ID}}$ and $K_{\mathrm{OOD}}$ are equal; something that is not always ensured in practice. We provide an empirical demonstration of how results for AUROC and AUPR can substantially differ when class cardinality between ID and OOD differs by 1, with AUROC differing by as much as 0.318 and AUPR by 0.613 for standard EDL, and AUROC by 0.360 and AUPR by 0.683 for IB-EDL. More concretely, our findings isolate an evaluation artefact: when K differs between ID and OOD, AUROC/AUPR can be artificially inflated without any change in model predictions. We further discuss the evaluation of EDL over causal language models using Multiple-Choice Question-Answer (MCQA) datasets and argue for clearer definitions of ID and OOD in this context. Our primary contribution is an empirical and theoretical demonstration that vacuity-based OOD detection in EDL-fine-tuned LLMs is highly sensitive to uncontrolled differences in evaluated class cardinality. 

---
# Debiased Multimodal Personality Understanding through Dual Causal Intervention 

**Authors**: Yangfu Zhu, Zitong Han, Nianwen Ning, Yuting Wei, Yuandong Wang, Hang Feng, Zhenzhou Shao  

**Link**: [PDF](https://arxiv.org/pdf/2605.06371)  

**Abstract**: Multimodalpersonalityunderstandingplaysacriticalroleinhuman centered artificial intelligence. Previous work mainly focus on learn-ing rich multimodal representations for video personality under standing. However, they often suffer from potential harm caused by subject bias (e.g., observable age and unobservable mental states), as subjects originate from diverse demographic backgrounds. Learn ing such spurious associations between multimodal features and traits may lead to unfair personality understanding. In this work, weconstruct aStructural Causal Model (SCM)toanalyze theimpact of these biases from a causal perspective, and propose a novel Dual Causal Adjustment Network (DCAN) to mitigate the interference of subject attributes on personality understanding. Specifically, we design a Back-door Adjustment Causal Learning (BACL) module to block spurious correlations from observable demographic factors via a prototype-based confounder dictionary, and subsequently ap ply a Front-door Adjustment Causal Learning (FACL) module to ad dress latent and unobservable biases throughalearnedmediatordic tionary intervention, thereby achieving causal disentanglement of representations for deconfounded reasoning. Importantly, we con struct a Demographic-annotated Multimodal Student Personality (DMSP) dataset to support the analysis and discussion of fairness related factors. Extensive experiments on the benchmark dataset CFI-V2 and our DMSPdataset demonstrate that DCAN consistently improves prediction accuracy, reaching 92.11% and 92.90%, respec tively. Meanwhile, the improvementsinthefairnessmetricsofequal opportunity and demographic parity are 6.57% and 7.97% on CFI-V2, and 15.38% and 20.06% on the DMSP dataset. Our code and DMSP dataset are available at this https URL 

---
# From Agent Loops to Deterministic Graphs: Execution Lineage for Reproducible AI-Native Work 

**Authors**: Josh Rosen, Seth Rosen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06365)  

**Abstract**: Large language model systems are increasingly deployed as agentic workflows that interleave reasoning, tool use, memory, and iterative refinement. These systems are effective at producing answers, but they often rely on implicit conversational state, making it difficult to preserve stable work products, isolate irrelevant updates, or propagate changes through intermediate artifacts.
We introduce execution lineage: an execution model in which AI-native work is represented as a directed acyclic graph (DAG) of artifact-producing computations with explicit dependencies, stable intermediate boundaries, and identity-based replay. The goal is not to make the model a better one-shot writer, but to make evolving AI-generated work maintainable under change.
We compare execution-lineage replay against loop-centric update baselines on two controlled policy-memo update tasks. In an unrelated-branch update, DAG replay preserved the final memo exactly in all runs, with zero churn and zero unrelated-branch contamination, while loop baselines regenerated the memo and frequently imported unrelated context. In an intermediate-artifact edit, all systems reflected the new constraint in the final memo, but only DAG replay achieved perfect upstream preservation, downstream propagation, unaffected-artifact preservation, and cross-artifact consistency.
These results show that final answer quality and maintained-state quality are distinct. Strong loop baselines can remain competitive at producing polished final outputs when the task is a bounded synthesis/update problem and all current sources fit in context, but immediate task success can mask partial state inconsistency that may compound over future revisions. Execution lineage provides stronger guarantees about what should change, what should remain stable, and how work evolves across revisions. 

---
# Prediction and Empowerment: A Theory of Agency through Bridge Interfaces 

**Authors**: Richard Csaky  

**Link**: [PDF](https://arxiv.org/pdf/2605.06346)  

**Abstract**: We study agency under partial observability in deterministic physical or simulated worlds, where apparent randomness arises from uncertainty over initial conditions, fixed law bits, and unrolled exogenous noise. We model sensing and actuation as bridge interfaces split between agent-controlled parameters and environment-controlled channel state, inducing a deterministic POMDP through a prior over latent microstates and many-to-one observation coarsening. Within this framework, we prove a separation between prediction, compression, and empowerment. Perfect prediction can be achieved either by identifying the hidden quotient relevant to the target family or by overwrite control that makes the future target action-determined; high empowerment alone is insufficient. Under refinable interfaces and sufficient memory, action-conditioned observation-compression progress reduces posterior uncertainty about the latent quotient, and when refinement requires steering world-side channel conditions, this creates target-conditioned interface empowerment. A bit-string specialization with a conserved information budget makes the resulting tradeoff explicit: prediction by identification requires internal capacity at least the relevant latent entropy, whereas overwrite control requires terminal action capacity over the controlled quotient. For modern AI agents, the results suggest a design principle rather than a theorem of inevitability: objectives should distinguish hidden-state identification, interface refinement, task-relevant controllability, and mere overwrite or distractor control. Human--AI alignment is partly an interface-design problem, where the relevant bridge is between human intent, agent internal state, external tools, and world-side channel conditions. This is a working draft: feedback and criticism is most welcome. 

---
# More Than Can Be Said: A Benchmark and Framework for Pre-Question Scientific Ideation 

**Authors**: Jie Yu, Song Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06345)  

**Abstract**: AI research agents have shown strong potential in automating literature search and manuscript refinement, yet most assume a clear and actionable initial input, operating only after a research question has been made explicit. In contrast, human research often begins with tacit friction, a sense of misalignment before a question can be formed. We introduce InciteResearch, a multi-agent framework designed to make a researcher's implicit understanding explicit, inspectable, and actionable. InciteResearch decomposes the logical chain of Socratic questioning and distributes it across the entire pipeline that: (1) Elicits a structured five-dimensional researcher profile state anchored by specific friction points from vague, even domain-unrelated inputs; (2) Violates hidden assumptions by maximizing the feasibility-novelty product with enforcing a 7-stage causal derivation trace; and (3) check whether the proposed method is a Necessary consequence of the reframed insight. We further introduce TF-Bench, the first benchmark for tacit-to-explicit research assistance that distinguishes domain-related from domain-unrelated inspirations across four scientific modes. On TF-Bench, InciteResearch achieves leapfrogging gains over a prompt-based baseline (novelty/impact from 3.671/3.806 to 4.250/4.397), shifting generated proposals from recombination to architectural insight. Our work demonstrates that AI can serve as an extension of thinking itself, rather than merely automating downstream execution. 

---
# Mind the Gap? A Distributional Comparison of Real and Synthetic Priors for Tabular Foundation Models 

**Authors**: Alex O. Davies, Telmo de Menezes e Silva Filho, Nirav Ajmeri  

**Link**: [PDF](https://arxiv.org/pdf/2605.06343)  

**Abstract**: Tabular foundation models are pre-trained on one of three classes of corpus: curated datasets drawn from benchmark repositories, tables harvested at scale from the web, or synthetic tables sampled from a parametric generative prior. Despite the centrality of pre-training data to model performance, little is known about how these corpora relate to one another in distribution, and the impact this has on downstream performance. In this work we take three canonical, archetypal datasets used to train tabular foundation models; the T4 dataset represents web-scraped corpora, the TabFM dataset curated tables from Kaggle, and the TabICL dataset as the only well-used synthetic prior with publicly available parameters. We characterise each corpus using aggregate features over whole tables, columns and correlations, and compare them using discriminator AUCs and k-NN coverage metrics. We find that the TabICL synthetic prior occupies a narrow region of the space of real tables, that this mismatch cannot be closed by optimising prior hyper-parameters across more than 86 thousand configurations, and that curated and web-scraped corpora are broadly interchangeable on a distributional level in feature space. Surprisingly, the distributional gap between synthetic pre-training data and real tables has a clearly detectable effect on performance under neither feature-based proximity measures or TabICL's own internal representations, suggesting that coverage of the real-data distribution is not the primary driver of TabICL's generalisation. 

---
# A Regime Theory of Controller Class Selection for LLM Action Decisions 

**Authors**: Zhaoyang Jiang, Zhizhong Fu, Yunsoo Kim, Jiacong Mi, Zicheng Li, Xuanqi Peng, Honghan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06339)  

**Abstract**: Deployed language and vision-language models must decide, on each input, whether to answer directly, retrieve evidence, defer to a stronger model, or abstain. Contrary to the common monotonicity intuition, greater per-input expressivity is not uniformly beneficial in finite samples: under identical strict cross-validation, different benchmarks prefer different controller classes. This reflects a finite-sample limitation of instance-level uncertainty signals, which can be exhausted at a distribution-dependent scale. We organize controllers into a nested lattice of four classes: fixed actions, partition routers, instance-level controllers, and prior-gated controllers, ordered by complexity. We prove a regime theory that turns three data-estimable bottlenecks into a class choice: how much improvement is possible beyond the best fixed action, whether there are enough samples for instance-level controllers to make reliable decisions, and how much improvement a coarse partition router can recover when instance-level signal is unreliable. The resulting Bernstein-tight threshold has a matching information-theoretic lower bound, and strict nested cross-validation provably selects a near-best class. Across SMS-Spam, HallusionBench, A-OKVQA, and FOLIO, the predicted class matches the empirical winner; the prior-gated controller wins on TextVQA when OCR tokens supply a label-free prediction-time prior. Code is available at this https URL. 

---
# Measuring Black-Box Confidence via Reasoning Trajectories: Geometry, Coverage, and Verbalization 

**Authors**: Marc Boubnovski Martell, Josefa Lia Stoisser, Kaspar Märtens, Jialin Yu, Robert Kitchen, Philip Torr, Jesper Ferkinghoff-Borg  

**Link**: [PDF](https://arxiv.org/pdf/2605.06308)  

**Abstract**: Reliable confidence estimation enables safe deployment of chain-of-thought (CoT) reasoning through text-only APIs. Yet the dominant black-box baseline, self-consistency over K samples, is linearly expensive and ignores the geometry of the trace. We propose a black-box trajectory-confidence score: we embed a CoT as a sliding-window trajectory and measure its convergence to external answer anchors with a one-parameter softmax. The method needs no logits, hidden states, or supervised calibrators. Across six (benchmark, reasoner) settings on MedQA-USMLE, GPQA Diamond, and MMLU-Pro with Gemini 3.1 Pro and Claude Sonnet 4.6, fusing this score with coverage and verbalized-confidence channels at K=4 yields Pareto improvements over self-consistency at K=8 in 6/6 settings (median AUC 0.78 vs 0.71, deltaAUC=+0.075). A fixed-pick control (+0.060) and E5 cross-embedder replication rule out answer switching and single-vendor artifacts. Geometry peaks in the penultimate window across benchmarks and reasoners, and inverts at the terminal window on GPQA Diamond. Three unscaffolded regimes separate black-box confidence into a judge-mediated Coverage prior (C), within-trace Geometry (G), and a conditional Verbalization channel (V). Across 18 benchmark x reasoner x proposer settings, C and G provide independent signal in 18/18 and 16/18, while V contributes residual signal in 6/18. Swapping the judge from GPT-5-mini to Claude Sonnet 4.6 leaves G-only AUC unchanged (|delta|<=0.013) and shifts C-only AUC by at most +/-0.02 (kappa=0.82). Fusion beats the best single channel in 17/18 settings (median AUC 0.78, max 0.92). 

---
# Addressing Labelled Data Scarcity: Taxonomy-Agnostic Annotation of PII Values in HTTP Traffic using LLMs 

**Authors**: Thomas Cory, Axel Küpper  

**Link**: [PDF](https://arxiv.org/pdf/2605.06305)  

**Abstract**: Automated privacy audits of web and mobile applications often analyse outbound HTTP traffic to detect Personally Identifiable Information (PII) leakage. However, existing learning-based detectors typically depend on scarce, manually labelled traffic and are tightly coupled to fixed label taxonomies, limiting transferability across domains and evolving definitions of PII. This paper investigates whether Large Language Models (LLMs) can support taxonomy-agnostic annotation of explicitly transmitted PII values in HTTP message bodies when the taxonomy is provided at runtime. We introduce a multi-stage LLM-based pipeline that combines deterministic pre-processing with label-level classification, targeted instance-level value annotation, and output validation. To enable controlled evaluation and exemplar-based prompting without relying on sensitive real-user captures, we further propose an LLM-based generator for synthetic HTTP traffic with manually validated, taxonomy-derived PII annotations. We evaluate the approach across three taxonomies spanning different PII domains and granularity levels. Results show that the pipeline accurately detects PII types and extracts corresponding values for concrete PII taxonomies. Overall, our findings position LLMs as a promising foundation for flexible, taxonomy-agnostic traffic annotation and for creating labelled data under evolving privacy taxonomies. 

---
# Data Language Models: A New Foundation Model Class for Tabular Data 

**Authors**: Eda Erol, Giuliano Pezzoli, Ozer Cem Kelahmet  

**Link**: [PDF](https://arxiv.org/pdf/2605.06290)  

**Abstract**: Every major data modality now has a foundation model that understands it natively: text has language models, images have vision models, audio has audio models. Tabular data, the modality on which many consequential real-world AI decisions are made, does not. Every approach to tabular AI today, from gradient-boosted trees to the latest tabular foundation models, requires a preprocessing pipeline before any model can consume the data. None of them understand tabular data as a modality. We introduce the Data Language Model (DLM), the missing foundation model for tabular data. A DLM understands tables the way a language model understands sentences: natively, without serialization or preprocessing, directly from raw cell values. It is the tabular data layer on which AI models, agents, and vertical AI applications can be built, eliminating the preprocessing pipelines that currently stand between raw data and every AI system that consumes it. We present Schema-1, the first DLM: a 140M parameter model trained on more than 2.3M synthetic and real-world tabular datasets. Schema-1 outperforms gradient-boosted ensembles, AutoML stacks, and the tabular foundation models we evaluate on established row-level prediction benchmarks. On missing value reconstruction it achieves lower reconstruction error than all classical statistical methods and frontier large language models on mean performance across conditions, establishing that structural understanding of a dataset's own distributional geometry is more useful for imputation than world knowledge encoded in language. It identifies the industry sector of any unseen dataset from raw cell values alone, reliably across any domain, a task no prior tabular model can perform. It is the native tabular understanding layer that has been missing from the AI stack. 

---
# Safactory: A Scalable Agent Factory for Trustworthy Autonomous Intelligence 

**Authors**: Xinquan Chen, Zhenyun Yin, Shan He, Bin Huang, Shanzhe Lei, Pengcheng Shi, Kun Cai, Bei Chen, Bangwei Liu, Zeyu Kang, Chao Huang, Yang Zhang, Wenjie Li, Ruijun Ge, Yajie Wang, Tianshun Fang, Tianyang Xu, Yiwen Cong, Meng Jin, Gaolei Li, Xuansheng Wu, Linhan Liu, Zijing He, An Li, Yan Teng, Xin Tan, ChaoChao Lu, Ji He, Jie Li, Chunfeng Song, Jinya Xu, Fan Song, Shujie Wang, Jianmin Qian, Jie Hou, Xuhong Wang, Yingchun Wang, Hui Wang, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06230)  

**Abstract**: As large models evolve from conversational assistants into autonomous agents, challenges increasingly arise from long-horizon decision making, tool use, and real environment interaction. Existing agenticinfrastructure remain fragmented across evaluation, data management, and agent evolution, making it difficult to discover risks systematically and improve models in a continuous closed loop. In this report, we present \textbf{Safactory}, a scalable agent factory for trustworthy autonomous intelligence. Safactory integrates three tightly coupled platforms: a \textbf{Parallel Simulation Platform} for trajectory generation, a \textbf{Trustworthy Data Platform} for trajectory storage and experience extraction, and an \textbf{Autonomous Evolution Platform} for asynchronous reinforcement learning and on-policy distillation. As far as we know, Safactory is the first framework to propose a unified evolutionary pipeline for next-generation trustworthy autonomous intelligence. 

---
# Price of Fairness in Short-Term and Long-Term Algorithmic Selections 

**Authors**: Shahin Jabbari, Chen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06227)  

**Abstract**: Algorithmic decision-making in high-stakes settings can have profound impacts on individuals and populations. While much prior work studies fairness in static settings, recent results show that enforcing static fairness constraints may exacerbate long-run disparities. Motivated by this tension, we study a stylized sequential selection problem in which a decision-maker repeatedly selects individuals, affecting both immediate utility and the population distribution over time. We introduce notions of group fairness for both the short and long term and theoretically analyze the trade-off between fairness and utility via the Price of Fairness (PoF). We characterize optimal and fair policies in the short term and show that the PoF can be large even when group distributions are nearly identical. In contrast, we show that long-term disparities can vanish under simple investment policies that achieve a low PoF. We also empirically validate these theoretical observations using both synthetic and real datasets. 

---
# A Versatile AI Agent for Rare Disease Diagnosis and Risk Gene Prioritization 

**Authors**: Tianyu Liu, Wangjie Zheng, Rui Yang, Benny Kai Guo Loo, Hui Zhang, Jeffries Lauran, Jianlei Gu, Botao Yu, Weihao Xuan, Kexin Huang, Nan Liu, James Zou, Yonghui Jiang, Hua Xu, Hongyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.06226)  

**Abstract**: Accurate and timely diagnosis is essential for effective treatment, particularly in the context of rare diseases. However, current diagnostic workflows often lead to prolonged assessment times and low accuracy. To address these limitations, we introduce Hygieia, a multi-modal AI agent system designed to support precision disease diagnosis by integrating diverse data sources, including phenotypic features, genetic profiles, and clinical records. Hygieia features a router-based and knowledge-enhanced framework that mitigates hallucination and tailors diagnostic strategies to different disease categories. Notably, it prioritizes risk-related genomic factors for rare diseases and provides confidence scores to assist clinical decision-making. We conducted a comprehensive evaluation demonstrating that Hygieia achieves state-of-the-art performance across multiple diagnostic benchmarks. In collaboration with clinical experts from Yale School of Medicine and Duke-NUS Medical School, we further validated its practical utility by showing (1) Hygieia's superior diagnostic performance compared to physicians with an improvement from 12%-60% and (2) its effectiveness in assisting clinicians with medical records for handling real-world cases. Our findings indicate that Hygieia not only enhances diagnostic accuracy and interpretability but also significantly reduces clinician workload, highlighting its potential as a valuable tool in clinical decision support systems. 

---
# Proactive Instance Navigation with Comparative Judgment for Ambiguous User Queries 

**Authors**: Junhyuk Kwon, Seungjoon Lee, Hyejin Park, Kyle Min, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2605.06223)  

**Abstract**: Natural-language instance navigation becomes challenging when the initial user request does not uniquely specify the target instance. A practical agent should reduce the user's burden by actively asking only the information needed to distinguish the target from similar distractors, rather than requiring a detailed description upfront. Existing approaches often fall short of this goal: they may stop at the first plausible candidate before sufficiently exploring alternatives, or, even after collecting multiple candidates, ask about the target's attributes derived from individual candidates rather than questions selected to distinguish candidates in the pool. As a result, despite the dialogue, the agent may still fail to distinguish the target from distractors, leading to premature decisions and lengthy user responses. We propose Proactive Instance Navigation with Comparative Judgment (ProCompNav), a two-stage framework that first constructs a candidate pool and then identifies the target through comparative judgment. At each round, ProCompNav extracts an attribute-value pair that splits the current pool, asks a binary yes/no question, and prunes all inconsistent candidates at once. This reframes disambiguation from open-ended target description to pool-level discriminative questioning, where each question is chosen to narrow the candidate set. On CoIN-Bench, ProCompNav improves Success Rate over interactive baselines with the same minimal input and non-interactive baselines with detailed descriptions, while substantially reducing Response Length. ProCompNav also achieves state-of-the-art Success Rate on TextNav, suggesting that comparative judgment is broadly useful for instance-level navigation among similar distractors. 

---
# Joint Consistency: A Unified Test-Time Aggregation Framework via Energy Minimization 

**Authors**: Yunzhen Yao, Hongye Wang, Yahong Wang, Michael C. Gastpar, Bo Jiang, Lie He  

**Link**: [PDF](https://arxiv.org/pdf/2605.06219)  

**Abstract**: This paper studies test-time aggregation, an approach that generates multiple reasoning traces and aggregates them into a final answer. Most existing methods rely on evaluation signals collected from candidate traces in isolation or answer frequencies, while ignoring comparative interactions among candidates. We propose Joint Consistency (JC), formulated as a constrained Ising-type energy minimization problem, where independent evaluation signals act as external fields and pairwise comparisons act as interactions. JC provides a unified framework for test-time aggregation that subsumes existing voting and weighted aggregation methods as special cases. Our construction of the interaction matrix leverages LLM-as-a-judge comparisons, and admits a theoretical interpretation under answer-level homogeneity assumptions. Moreover, we develop an efficient approximation strategy that makes interaction modeling practical for large-scale test-time aggregation. Experiments on math and code reasoning benchmarks show that JC consistently outperforms existing baselines across tasks, judge models, trace budgets, and trace-generation settings. 

---
# Beyond Fixed Benchmarks and Worst-Case Attacks: Dynamic Boundary Evaluation for Language Models 

**Authors**: Haoxiang Wang, Da Yu, Huishuai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06213)  

**Abstract**: Evaluating large language models (LLMs) today rests on fixed benchmarks that apply the same set of items to any model, producing ceiling and floor effects that mask capability gaps. We argue that the most informative evaluation signal lies at the boundary, where the per-prompt pass probability is near $0.5$ under random-sampling decoding, and propose Dynamic Boundary Evaluation (DBE), which actively locates each model's boundary and places it on a globally comparable difficulty scale. DBE delivers three artifacts: (i) a calibrated item bank covering safety, capability, and truthfulness, with per-item difficulty labels validated across $9$ reference LLMs; (ii) Skill-Guided Boundary Search (SGBS), a search algorithm that finds boundary items for a given target LLM using only API-level query access; and (iii) an evaluation protocol that places a new LLM on a unified ability scale and grows the evaluation set adaptively when the target falls outside the bank's coverage. We instantiate DBE on four categories spanning safety (harmful request refusal and over-refusal), capability (constrained instruction following), and truthfulness (multi-turn sycophancy resistance). The resulting evaluation covers a broader model spectrum without saturation while remaining compatible with existing datasets. 

---
# Towards Annotation-Free Validation of MLLMs: A Vision-Language Logical Consistency Metric 

**Authors**: Ying Gu, Mei Chee Leong, Hui Li Tan, Shangbo Mao, Liyuan Li, Nancy Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06201)  

**Abstract**: Dominant accuracy evaluation might reward unwarranted guessing of Large Language Models, and it might not be applicable to novel tasks for model validation without ground-truth (gt) annotation. Based on basic logic principle, we propose a novel framework to evaluate the vision-language logical consistency of MLLMs on both sufficient and necessary cause-effect relations. We define Vision-Language Logical Consistency Metric (VL-LCM) on traditional MC-VQA tests, and recent NaturalBench tests without the need for gt annotation. Through systematic experiments on representative VL benchmark MMMU and recent VL challenges like NaturalBench, we evaluated 11 recent open-source MLLMs from 4 frontier families. Our findings reveal that, despite significant progress of recent MLLMs on accuracy, logical consistency lags behind significantly. Extensive evaluations on the correlations of VL-LCM with metrics on gt, the reliability of LCM, and the relation of VL-LCM with response distribution justify the validity and applicability of VL-LCM even without gt annotation. Our findings suggest that, beyond accuracy, logical consistency could be employed for both accuracy and reliability. VL-LCM can also be employed for MLLM selection, validation, and reliable answer justification in novel tasks without gt annotation. 

---
# The Granularity Axis: A Micro-to-Macro Latent Direction for Social Roles in Language Models 

**Authors**: Chonghan Qin, Xiachong Feng, Ziyun Song, Xiaocheng Feng, Jing Xiong, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2605.06196)  

**Abstract**: Large language models (LLMs) are routinely prompted to take on social roles ranging from individuals to institutions, yet it remains unclear whether their internal representations encode the granularity of such roles, from micro-level individual experience to macro-level organizational, institutional, or national reasoning. We show that they do. We define a contrast-based Granularity Axis as the difference between mean macro- and micro-role hidden states. In Qwen3-8B, this axis aligns with the principal axis (PC1) of the role representation space at cosine 0.972 and accounts for 52.6% of its variance, indicating that granularity is the dominant geometric axis organizing prompted social roles. We construct 75 social roles across five granularity levels and collect 91,200 role-conditioned responses over shared questions and prompt variants, then extract role-level hidden states and project them onto the axis. Role projections increase monotonically across all five levels, remain stable across layers, prompt variants, endpoint definitions, held-out splits, and score-filtered subsets, and transfer to Llama-3.1-8B-Instruct. The axis is also causally relevant: activation steering along it shifts response granularity in the predicted direction, with Llama moving from 2.00 to 3.17 on a five-point macro scale under positive steering on prompts that admit local responses. The two models differ in controllability, suggesting that steering depends on each model's default operating regime. Overall, our findings suggest that social role granularity is not merely a stylistic surface feature, but a structured, ordered, and causally manipulable latent direction in role-conditioned language model behavior. 

---
# Systematic Evaluation of Large Language Models for Post-Discharge Clinical Action Extraction 

**Authors**: Shivali Dalmia, Ananya Mantravadi, Prasanna Desikan  

**Link**: [PDF](https://arxiv.org/pdf/2605.06191)  

**Abstract**: The work in this paper evaluates zero-shot and few-shot large language models (LLMs) for safety-critical clinical action extraction using the CLIP discharge-note dataset, with particular emphasis on transitions of care and post-discharge patient safety. To manage the complexity of clinical documentation, we introduce a two-stage extraction framework that decomposes discharge notes, that are written in narrative form, into fine-grained, explicitly actionable clinical tasks through a staged prompting strategy. Our contributions include a systematic assessment of generative LLMs for clinical action extraction, a detailed comparison between general-purpose LLMs and task-specific supervised BERT-based models, and an analysis of annotation inconsistencies across different action categories. We show that contemporary LLMs achieve performance comparable to or exceeding supervised models on binary actionability detection, while supervised baselines retain a meaningful advantage on fine-grained multi-label category classification, despite the absence of task-specific fine-tuning and under strict data-privacy constraints. Qualitative error analysis reveals that many failures stem from misalignment between model reasoning and dataset annotation conventions, particularly in cases involving implicit clinical actions and rigid structural labeling rules. These results indicate that reported performance reflects model limitations due to lack of clinical reasoning, that is not captured by plain annotations. Labels without rationales make it impossible to distinguish clinical reasoning failures from annotation convention mismatches. Advancing clinical NLP requires reasoning-annotated datasets that document why specific spans are actionable, not merely which spans were labeled, enabling proper evaluation of model clinical understanding. 

---
# OPSD Compresses What RLVR Teaches: A Post-RL Compaction Stage for Reasoning Models 

**Authors**: Jaehoon Kim, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.06188)  

**Abstract**: On-Policy Self-Distillation (OPSD) has recently emerged as an alternative to Reinforcement Learning with Verifiable Rewards (RLVR), promising higher accuracy and shorter responses through token-level credit assignment from a self-teacher conditioned on privileged context. However, this promise does not carry over to thinking-enabled mathematical reasoning, where reported accuracy gains shrink and sometimes turn negative. We hypothesize that hindsight supervision can specify better token-level alternatives in short thinking-disabled outputs, but in long thinking-enabled traces it more readily identifies redundancy than supplies better replacements. To test this, we applied OPSD separately to correct and incorrect rollout groups, so that compression and correction can be observed in isolation. Our results show that in thinking-enabled mathematical reasoning, OPSD behaves most reliably as a compression mechanism rather than a correction mechanism: training only on correct rollouts preserves accuracy while substantially shortening responses, whereas training only on incorrect rollouts damages accuracy. In light of these findings, we propose a revised post-training pipeline for thinking-enabled mathematical reasoning: SFT then RLVR then OPSD. 

---
# Event-Causal RAG: A Retrieval-Augmented Generation Framework for Long Video Reasoning in Complex Scenarios 

**Authors**: Peizheng Yan, Yu Zhao, Liang Xie, Juntong Qi, Mingming Wang, Erwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06185)  

**Abstract**: Recent large vision-language models have achieved strong performance on short- and medium-length video understanding, yet they remain inadequate for ultra-long or even infinite video reasoning, where models must preserve coherent memory over extended durations and infer causal dependencies across temporally distant events. Existing end-to-end video understanding methods are fundamentally limited by the $O(n^2)$ complexity of self-attention, while recent retrieval-augmented generation (RAG) approaches still suffer from fragmented clip-level memory, weak modeling of temporal and causal structure, and high storage and online inference costs. We present Event-Causal RAG, a lightweight retrieval-augmented framework for infinite long-video reasoning. Instead of indexing fixed-length clips, our method segments streaming videos into semantically coherent events and represents each event as a structured State-Event-State (SES) graph, capturing the event together with its surrounding state transitions. These graphs are merged into a global Event Knowledge Graph and stored in a dual-store memory that supports both semantic matching and causal-topological retrieval. On top of this memory, we design a bidirectional retrieval strategy to efficiently identify the most relevant event causal chains and provide them, together with the associated video evidence, to a backbone video foundation model for answer generation. Experiments on long-video understanding benchmarks demonstrate that Event-Causal RAG consistently outperforms strong clip-based retrieval baselines and long-context video models, particularly on questions requiring multi-event integration and causal inference across long temporal gaps, while also achieving improved memory efficiency and robust streaming performance. 

---
# Rethinking Adapter Placement: A Dominant Adaptation Module Perspective 

**Authors**: Suoxin Zhang, Run He, Di Fang, Xiang Tan, Kaixuan Chen, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06183)  

**Abstract**: Low-rank adaptation (LoRA) is a widely used parameter-efficient fine-tuning method that places trainable low-rank adapters into frozen pre-trained models. Recent studies show that using fewer LoRA adapters may still maintain or even improve performance, but existing methods still distribute adapters broadly, leaving where to place a limited number of adapters to maximize performance largely open. To investigate this, we introduce PAGE (Projected Adapter Gradient Energy), a gradient-based sensitivity probe that estimates the initial trainable gradient energy available to each candidate LoRA adapter. Surprisingly, we find that PAGE is highly concentrated on a single shallow FFN down-projection across two model families and four downstream tasks. We term this module the dominant adaptation module and show that its layer index is architecture-dependent but task-stable. Motivated by this finding, we propose DomLoRA, a placement method that places a single adapter at the dominant adaptation module. With only ~0.7% of vanilla LoRA's trainable parameters, DomLoRA outperforms it on average across various downstream tasks, including instruction following, mathematical reasoning, code generation, and multi-turn conversation. This method also improves other LoRA variants, supporting the dominant adaptation module perspective as a practical placement guideline. 

---
# BioMedArena: An Open-source Toolkit for Building and Evaluating Biomedical Deep Research Agents 

**Authors**: Jinge Wu, Hongjian Zhou, Mingde Zeng, Jiayuan Zhu, Junde Wu, Jiazhen Pan, Sean Wu, Honghan Wu, Fenglin Liu, David A. Clifton  

**Link**: [PDF](https://arxiv.org/pdf/2605.06177)  

**Abstract**: Building a deep research agent today is an exercise in glue code: the same backbone evaluated on the same benchmark can report different accuracies in different papers because harness and tool registry all differ, and integrating a new foundation model into a comparable evaluation surface costs weeks of model-specific engineering. We call this the per-paper engineering tax and release BioMedArena, an open-source toolkit that not only alleviates it but also provides an arena for fair comparison of different foundation models when evaluating them as deep-research agents. BioMedArena decouples six layers of biomedical agent evaluation -- benchmark loading, tool exposure, tool selection, execution mode, context management, and scoring -- and exposes 147 biomedical benchmarks and 75 biomedical tools across 9 functional families. Adding a new model, benchmark, or tool reduces to registering a few-line provider adapter. We further provide 6 agent harnesses with 6 context-management strategies, which provide 12 backbones with competitive research capabilities and significantly improved performance, achieving state-of-the-art (SOTA) results on 8 representative biomedical benchmarks, with an average lift of +15.03 percentage points over prior SOTA. The toolkit, configurations, and per-task traces are available at this https URL 

---
# Post Reasoning: Improving the Performance of Non-Thinking Models at No Cost 

**Authors**: Richmond Sin Jing Xuan, Rishabh Bhardwaj, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2605.06165)  

**Abstract**: As the widespread adoption of Large Language Models (LLMs) accelerates, token consumption from intermediate reasoning traces increasingly contributes to inference latency and operational cost. Recent studies suggest that many real-world tasks require little to no explicit reasoning, with additional reasoning sometimes even degrading performance. In this work, we propose \textbf{Post-Reasoning}, a simple yet effective approach that improves instruction-tuned models by conditioning them to justify their answers after generating the final response. By design, it enables the final answer to be obtained without additional latency or token cost, while still improving performance through simple instruction augmentation. We evaluate Post-Reasoning across \(117\) model--benchmark settings spanning \(13\) open and proprietary models, \(4\) model families, and \(9\) diverse reasoning and knowledge-intensive benchmarks, including AMC, HMMT, GSM8K, GPQA, MMLU-Pro, and BIG-Bench Hard. Post-Reasoning improves performance in over \(88.19\%\) of evaluated settings, achieving a mean relative improvements of \(17.37\%\). Furthermore, we propose supervised post-reason tuning, which further improves performance in over \(91.11\%\) of evaluated settings, and exceeds the prompt-based post-reasoning baseline by an average of \(8.01\%\), demonstrating that post-reasoning can be effectively internalized through training. Ultimately, Post-Reasoning establishes a new performance ceiling for direct-answer capabilities. 

---
# Beyond Accuracy: Policy Invariance as a Reliability Test for LLM Safety Judges 

**Authors**: Shihao Weng, Yang Feng, Xiaofei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.06161)  

**Abstract**: LLM-as-a-Judge pipelines have become the de facto evaluator for agent safety, yet existing benchmarks treat their verdicts as ground-truth proxies without checking whether the verdicts depend on the agent's behavior or merely on how the evaluation policy happens to be worded. We argue that any trustworthy safety judge must satisfy a basic property we call policy invariance, and we operationalize it as three testable principles: rubric-semantics invariance under certified-equivalent rewrites, rubric-threshold invariance under intentional strict-to-lenient shifts, and ambiguity-aware calibration so that verdict instability concentrates on genuinely ambiguous cases. Instantiating these principles as a stress-test protocol with four agent-class judges on trajectories drawn from ASSEBench and R-Judge, we surface a previously unmeasured failure mode: today's judges respond to meaningful normative shifts and to meaningless structural rewrites with comparable strength, and cannot tell the two apart. Content-preserving policy rewrites flip up to 9.1% of verdicts above baseline jitter, and 18-43% of all observed flips occur on unambiguous cases under such rewrites, so existing safety scores conflate what the agent did with how the evaluator was prompted. Beyond the diagnosis, we contribute the Policy Invariance Score and the Judge Card reporting protocol, which expose an order-of-magnitude spread in judge reliability that is invisible to accuracy-only leaderboards. We release the protocol and code so that future agent-safety benchmarks can audit their own evaluators rather than trust them by default. 

---
# Graphlets as Building Blocks for Structural Vocabulary in Knowledge Graph Foundation Models 

**Authors**: Kossi Amouzouvi, Robert Wardenga, Jens Lehmann, Sahar Vahdati  

**Link**: [PDF](https://arxiv.org/pdf/2605.06154)  

**Abstract**: Foundation models excel at language, where sentences become tokens, and vision, where images become pixels, because both reduce to discrete symbols on a shared, fixed grid. Knowledge Graphs share the discreteness, but not the geometry. Their entities and relations are discrete symbols, yet their arrangement is relational and lacks a common, fixed grid. Knowledge Graphs (KGs) share the discreteness, but not the geometry. They form irregular, non-Euclidean topologies whose local neighborhoods differ from graph to graph. Therefore, Knowledge Graph Foundation Models (KGFMs) rely on identifying structural invariances to produce transferable representations. Without a universal token set, KGFMs are limited in their ability to transfer representations across unseen KGs. We close this gap by treating graphlets, small connected graphs, as structural tokens that recur in heterogeneous KGs. In this paper, We introduce a model-agnostic framework based on a vocabulary of graphlets that mines a KG between relations via pattern matching. In particular, we considered closed and open 2- and 3-path, and star graphlets, to obtain robust invariances. The framework is evaluated on 51 KGs from a wide range of domains, for zero-shot inductive and transductive link prediction. Experiments show that adding simple graphlets to the vocabulary yields models that outperform prior KGFMs. 

---
# Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning 

**Authors**: Yaorui Shi, Yuxin Chen, Zhengxi Lu, Yuchun Miao, Shugui Liu, Qi GU, Xunliang Cai, Xiang Wang, An Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06130)  

**Abstract**: A persistent skill library allows language model agents to reuse successful strategies across tasks. Maintaining such a library requires three coupled capabilities. The agent selects a relevant skill, utilizes it during execution, and distills new skills from experience. Existing methods optimize these capabilities in isolation or with separate reward sources, resulting in partial and conflicting evolution. We propose Skill1, a framework that trains a single policy to co-evolve skill selection, utilization, and distillation toward a shared task-outcome objective. The policy generates a query to search the skill library, re-ranks candidates to select one, solves the task conditioned on it, and distills a new skill from the trajectory. All learning derives from a single task-outcome signal. Its low-frequency trend credits selection and its high-frequency variation credits distillation. Experiments on ALFWorld and WebShop show that Skill1 outperforms prior skill-based and reinforcement learning baselines. Training dynamics confirm the co-evolution of the three capabilities, and ablations show that removing any credit signal degrades the evolution. 

---
# P-Guide: Parameter-Efficient Prior Steering for Single-Pass CFG Inference 

**Authors**: Xin Peng, Ang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.06124)  

**Abstract**: Classifier-Free Guidance (CFG) is essential for high-fidelity conditional generation in flow matching, yet it imposes significant computational overhead by requiring dual forward passes at each sampling step. In this work, we address this bottleneck by introducing \textbf{P-Guide}, a framework that achieves high-quality guidance through a single inference pass by modulating only the initial latent state. We further show that, under a first-order approximation, P-Guide is equivalent to CFG in the sense that it steers generation from the prior space, without requiring explicit velocity field extrapolation during sampling. We consider both homoscedastic and \textbf{heteroscedastic} priors, and find that jointly modeling the mean and variance enables adaptive loss attenuation and improved robustness to data uncertainty. Extensive experiments demonstrate that P-Guide reduces inference latency by approximately 50\% while maintaining fidelity and prompt alignment competitive with standard dual-pass CFG baselines. 

---
# Back to the Beginning of Heuristic Design: Bridging Code and Knowledge with LLMs 

**Authors**: Nguyen Viet Tuan Kiet, Bui Dinh Pham, Dao Van Tung, Tran Cong Dao, Huynh Thi Thanh Binh  

**Link**: [PDF](https://arxiv.org/pdf/2605.06123)  

**Abstract**: Large language models (LLMs) have recently advanced automatic heuristic design (AHD) for combinatorial optimization (CO), where candidate heuristics are iteratively proposed, evaluated, and refined. Most existing approaches search over executable programs and distill insights from execution feedback to guide later iterations. Because this process moves from low-level implementations to high-level principles, we refer to it as a bottom-up paradigm. We argue that this view is incomplete and introduce a complementary top-down perspective: knowledge becomes the primary search object and code merely instantiates and tests it, making what is learned explicit and reusable across problems and trajectories. We formalize this shift through a statistical-learning view that exposes a distortion--compression trade-off, and instantiate it in both population-based and tree-based AHD frameworks. Across CO and tasks beyond it, knowledge-first search improves discovery efficiency, transfer, and generalization, often outperforming code-centric pipelines, while combining both strategies yields further gains. Our results suggest that progress in AHD depends on iteratively constructing and evolving interpretable hypotheses that retain value beyond a single search trajectory. 

---
# Policy-Guided Stepwise Model Routing for Cost-Effective Reasoning 

**Authors**: Wenwen Si, Insup Lee, Osbert Bastani  

**Link**: [PDF](https://arxiv.org/pdf/2605.06116)  

**Abstract**: Inference-time computation has greatly enhanced the performance of large language models (LLMs) on challenging reasoning tasks, but this strategy can incur high inference costs. One solution is to route intermediate chain-of-thought (CoT) states to language models of different sizes; however, existing approaches rely on handcrafted routing strategies that limit performance, or on training large process reward models that may be infeasible in many applications. We formulate stepwise model routing as a constrained decision-making problem, which we solve by training a small control policy using reinforcement learning in conjunction with threshold calibration to tune the performance-efficiency tradeoff. We validate our method on three math benchmarks (GSM8K, MATH500, and OmniMath) on both open and closed models. Our method consistently improves the accuracy-cost tradeoff compared to handcrafted approaches, while achieving a comparable tradeoff to methods that require training large process reward models. 

---
# CrossCult-KIBench: A Benchmark for Cross-Cultural Knowledge Insertion in MLLMs 

**Authors**: Zhen Zeng, Leijiang Gu, Feng Li, Jing Yu, Zenglin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.06115)  

**Abstract**: Multimodal Large Language Models (MLLMs), trained primarily on English-centric data, frequently generate culturally inappropriate or misaligned responses in cross-cultural settings. To mitigate this, we introduce the task of cross-cultural knowledge insertion, which focuses on adapting models to specific cultural contexts while preserving their original behavior in other cultures. To facilitate research in this area, we introduce CrossCult-KIBench, a comprehensive evaluation benchmark for assessing both the effectiveness of knowledge insertion and its unintended side effects on non-target cultures. The benchmark includes 9,800 image-grounded cases covering 49 culturally relevant visual scenarios across English, Chinese, and Arabic language-culture groups. It supports evaluation in both single-insert and sequential-insert settings. We also propose Memory-Conditioned Knowledge Insertion (MCKI) as a baseline method. MCKI retrieves relevant cultural knowledge from an external memory using frozen MLLM representations, prepending matched entries as conditional prompts when applicable. Extensive experiments on CrossCult-KIBench reveal that current approaches struggle to balance effective cultural adaptation with behavioral preservation, highlighting a key challenge in developing culturally-aware MLLMs. Our work thus underscores an important research direction for developing more culturally adaptive and responsible MLLMs. 

---
# On Time, Within Budget: Constraint-Driven Online Resource Allocation for Agentic Workflows 

**Authors**: Xinglin Wang, Zishen Liu, Shaoxiong Feng, Peiwen Yuan, Yiwei Li, Jiayi Shi, Yueqi Zhang, Chuyi Tan, Ji Zhang, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.06110)  

**Abstract**: Agentic systems increasingly solve complex user requests by executing orchestrated workflows, where subtasks are assigned to specialized models or tools and coordinated according to their dependencies. While recent work improves agent efficiency by optimizing the performance--cost--latency frontier, real deployments often impose concrete requirements: a workflow must be completed within a specified budget and before a specified deadline. This shifts the goal from average efficiency optimization to maximizing the probability that the entire workflow completes successfully under explicit budget and deadline constraints. We study \emph{constraint-driven online resource allocation for agentic workflows}. Given a dependency-structured workflow and estimates of success rates and generation lengths for each subtask--model pair, the executor allocates models and parallel samples across simultaneously executable subtasks while managing the remaining budget and time. We formulate this setting as a finite-horizon stochastic online allocation problem and propose \emph{Monte Carlo Portfolio Planning} (MCPP), a lightweight closed-loop planner that directly estimates constrained completion probability through simulated workflow executions and replans after observed outcomes. Experiments on CodeFlow and ProofFlow demonstrate that MCPP consistently improves constrained completion probability over strong baselines across a wide range of budget--deadline constraints. 

---
# Shallow Prefill, Deep Decoding: Efficient Long-Context Inference via Layer-Asymmetric KV Visibility 

**Authors**: Jungsuk Oh, Hyeseo Jeon, Hyunjune Ji, Kyongmin Kong, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.06105)  

**Abstract**: Long-context inference in decoder-only language models is costly because long prompts are processed during Prefill, cached at every layer, and repeatedly attended to during autoregressive Decode. We introduce \emph{Shallow Prefill, dEEp Decode} (SPEED), a phase-asymmetric KV-visibility policy that materializes non-anchor prompt-token KV states only in lower layers while keeping Decode-phase tokens full-depth. Unlike previous approaches that make upper-layer prompt KV states cheaper to store or construct, SPEED removes prefill tokens from the upper-layer Decode visibility set altogether. With a minimal BoS anchor, this simple change preserves broad benchmark quality while reducing long-context cost. In a controlled Llama-3.1-8B instruction-tuning study, SPEED using only 75\% of layers for prefill tokens reaches 51.2 average score on OLMES-style benchmarks, compared with 51.4 for the full-depth baseline, while improving TTFT by 33\%, TPOT by 22\%, and reducing active KV memory by 25.0\% at 128K context. Layer-wise diagnostics suggest that this cutoff retains the main prompt-selection and representation-stabilization regions of the full-depth model. These results show that long-context prompt tokens need not always persist as full-depth KV-cache objects when Decode-phase tokens remain full-depth. 

---
# Safety Certification is Classification 

**Authors**: Oliver Schön, Licio Romao, Sadegh Soudjani  

**Link**: [PDF](https://arxiv.org/pdf/2605.06087)  

**Abstract**: The goal of this paper is certifying safety of dynamical systems subject to uncertainty. Existing approaches use trajectory data to estimate transition probabilities, and compute safety probabilities recursively via dynamic programming (DP). This recursion may lead to compounding errors in the certified safety probability, thus collapsing to a vacuous lower bound for growing horizons $T$. We propose a kernel embedding framework that treats safety certification as a classification problem on trajectory data, directly estimating the $T$-step safety probability without recursion. We show that the framework subsumes well-established approaches from the literature (e.g., barrier certificates, robust Markov models) as special cases, and allows us to go beyond their limitations. As the main consequence, it bypasses compounding error across the horizon and enables certification for systems with non-Markovian dynamics. We demonstrate that direct estimators remain stable independent of the certification horizon and in the non-Markovian setting, whilst DP-based certificates silently go unsound -- confirmed in simulation on a neural-controlled quadrotor. 

---
# VibeServe: Can AI Agents Build Bespoke LLM Serving Systems? 

**Authors**: Keisuke Kamahori, Shihang Li, Simon Peter, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2605.06068)  

**Abstract**: For years, we have built LLM serving systems like any other critical infrastructure: a single general-purpose stack, hand-tuned over many engineer-years, meant to support every model and workload. In this paper, we take the opposite bet: a multi-agent loop that automatically synthesizes bespoke serving systems for different usage scenarios. We propose VibeServe, the first agentic loop that generates entire LLM serving stacks end-to-end. VibeServe uses an outer loop to plan and track the search over system designs, and an inner loop to implement candidates, check correctness, and measure performance on the target benchmark. In the standard deployment setting, where existing stacks are highly optimized, VibeServe remains competitive with vLLM, showing that generation-time specialization need not come at the cost of performance. More interestingly, in non-standard scenarios, VibeServe outperforms existing systems by exploiting opportunities that generic systems miss in six scenarios involving non-standard model architectures, workload knowledge, and hardware-specific optimizations. Together, these results suggest a different point in the design space for infrastructure software: generation-time specialization rather than runtime generality. Code is available at this https URL. 

---
# Visual Fingerprints for LLM Generation Comparison 

**Authors**: Amal Alnouri, Andreas Hinterreiter, Christina Humer, Furui Cheng, Marc Streit  

**Link**: [PDF](https://arxiv.org/pdf/2605.06054)  

**Abstract**: Large language model (LLM) outputs arise from complex interactions among prompts, system instructions, model parameters, and architecture. We refer to specific configurations of these factors as generation conditions, each of which can bias outputs in various ways. Understanding how different generation conditions shape model behaviors is essential for tasks such as prompt design and model evaluation, yet it remains challenging due to the stochastic and open-ended nature of text generation. We present an approach to visually compare LLM outputs across generation conditions by modeling responses as collections of linguistic choices, including content, expression, and structure. We extract these choices using natural language processing pipelines and represent their distributions across repeated samples. We then visualize these distributions as visual fingerprints, enabling direct, distribution-level comparison of condition-specific tendencies. Through four usage scenarios, we demonstrate how visual fingerprints reveal consistent patterns in LLM behavior that are difficult to observe through individual responses or aggregate metrics. 

---
# Novelty-based Tree-of-Thought Search for LLM Reasoning and Planning 

**Authors**: Leon Hamm, Zlatan Ajanovic  

**Link**: [PDF](https://arxiv.org/pdf/2605.06040)  

**Abstract**: Although advances such as chain-of-thought, tree-of-thought or reinforcement learning have improved the performance of LLMs in reasoning and planning tasks, they are still brittle and have not achieved human-level performance in many domains, and often suffer from high time and token costs. Inspired by the success of width-based search in planning, we explore how the concept of novelty can be transferred to language domains and how it can improve tree-of-thought reasoning. A tree of thoughts relies on building possible "paths" of consecutive ideas or thoughts. These are generated by repeatedly prompting an LLM. In our paper, a measurable concept of novelty is proposed that describes the uniqueness of a new node (thought) in comparison to nodes previously seen in the search tree. Novelty is estimated by prompting an LLM and making use of embedded general knowledge from pre-training. This metric can then be used to prune branches and reduce the scope of the search. Although this method introduces more prompts per state, the overall token cost can be reduced by pruning and reducing the overall tree size. This procedure is tested and compared using several benchmarks in language-based planning and general reasoning. 

---
# Pathways to AGI 

**Authors**: Gordon Fletcher, Saomai Vu Khan  

**Link**: [PDF](https://arxiv.org/pdf/2605.06029)  

**Abstract**: Our focus are five related questions that stem from a critical software studies perspective. Underpinning this view is the acknowledged need to avoid assumptions regarding the inevitability of the current situation relating to AI. What we need to see is the closeness of the linkage between current commercial AI development and our prevailing social, political and economic circumstances. This does mean that the perspectives presented here are done so critically and conditionally. Most importantly, Artificial General Intelligence (AGI) is seen as being problematic both conceptually and definitionally. This conditioning of any view regarding AGI does lead the discussion in specific directions and to certain conclusions regarding the future. However, adopting this perspective enables the work to offer some final recommendations. We set out to ask the following questions, 1. What are the critical pathways that produced the current dominant generative AI tools (capabilities, product forms, adoption patterns)? 2. Which decision points acted as leverage nodes (small changes that had large downstream effects), and which dead ends reveal alternative possibilities that did not become dominant? 3. How do pathways differ across three foundational-model trajectories such as the frontier proprietary models, open-weight models or specific domain and sovereign models? 4. Which alternative projects branched from key leverage nodes, what is their current state, and why did some succeed, stall, fail or become absorbed? 5. Based on this analysis, what socio-technical development programmes could plausibly move toward AGI-adjacent capability while meeting requirements for transparency, moderation, wellbeing and sustainable business models? 

---
# Strat-LLM: Stratified Strategy Alignment for LLM-based Stock Trading with Real-time Multi-Source Signals 

**Authors**: Wenliang Huang, Zengyi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06024)  

**Abstract**: Large Language Models (LLMs) are evolving into autonomous trading agents, yet existing benchmarks often overlook the interplay between architectural reasoning and strategy consistency. We propose Strat-LLM, a framework grounded in Stratified Strategy Alignment. Operating in a live-forward setting throughout 2025, it integrates heterogeneous data including sequential prices, real-time news, and annual reports to eliminate look-ahead bias. Extensive stress tests on A-share and U.S. markets reveal: (1) reasoning-heavy models achieve peak utility in Free Mode via internal logic, whereas standard models require Strict Mode as a vital risk anchor; (2) alignment utility is regime-dependent, with Free and Guided modes capturing momentum in uptrending markets, while Strict Mode mitigates drawdowns in downtrends; (3) mid-scale models (35B) show optimal fidelity under strict constraints, whereas ultra-large models (122B) suffer an alignment tax under rigid rules but gain a performance premium in Guided Mode; (4) standard LLMs often fall into a high win-rate trap, optimizing for small gains at the expense of total returns, which can only be mitigated through deep reasoning or strict external guardrails. Project details are available at this https URL. 

---
# BioResearcher: Scenario-Guided Multi-Agent for Translational Medicine 

**Authors**: Remigiusz Kinas, Joanna Krawczyk, Rafał Powalski, Przemysław Pietrzak, Agnieszka Kowalewska, Krzysztof Kolmus, Maciej Sypetkowski, Łukasz Smoliński, Tomasz Jetka  

**Link**: [PDF](https://arxiv.org/pdf/2605.05985)  

**Abstract**: Translational medicine turns underspecified development goals into evidence synthesis that must combine literature, trials, patents, and quantitative multi-omics analysis while preserving identifiers, uncertainty, and retrievable provenance. General-purpose foundation models and off-the-shelf tool-augmented or multi-agent systems are not built for this: they tend to produce single-shot answers or run open-endedly, and fall short on the auditable, scenario-specific workflows that heterogeneous biomedical sources demand.
This paper introduces Ingenix BioResearcher, a scenario-guided multi-agent system that maps queries to versioned research playbooks, delegates to specialized subagents over 30+ tools and machine-learning endpoints, mixes structured database access with sandboxed code for genome-scale analyses, and applies claim-level multi-model reconciliation before editorial assembly.
We evaluate BioResearcher across unit-level capabilities, open-ended biomedical reasoning, and end-to-end clinical discovery. It leads evaluated baselines on 109 single-step tests (83.49% pass rate; 0.892 average score), achieves strong biomedical benchmark performance (89.33% on BixBench-Verified-50 and the top 0.758 mean score on BaisBench Scientific Discovery), and leads on a 30-query clinical end-to-end benchmark with the highest positive hit rate (74.7% $\pm$ 3.3%) and negative clear rate (96.8% $\pm$ 0.2%). These results show broad, competitive performance across unit-level, open-ended, and end-to-end clinical evaluations. 

---
# TACT: Mitigating Overthinking and Overacting in Coding Agents via Activation Steering 

**Authors**: Yuan Sui, Yulin Chen, Yibo Li, Xue Jiang, Yufei He, Yihong Dong, Xiaoxin He, Tianyu Gao, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2605.05980)  

**Abstract**: When language model agents tackle complex software engineering tasks, they often degrade over long trajectories, which we define as *agent drift*. We focus on two recurring failure modes *overthinking* and *overacting*, i.e., where the agent repeatedly reasons over information it already has, and where it issues tool calls without integrating recent observations or acquiring new evidence. In this paper, we introduce TACT (Think-Act Calibration via activation Steering), to detect and mitigate agent drift in the residual stream before it surfaces as a behavioral failure. In specific, we label trajectory steps as overthinking, overacting, or calibrated, and find that their hidden states can separate linearly along two *drift axes*, pointing from calibrated behavior toward each failure mode (AUC $\approx$ 0.9). To mitigate agent drift, we project each step's activation onto these axes at test time and pull drifted ones back toward the calibrated region. Experiments show that TACT outperforms unsteered baselines across SWE-bench Verified, Terminal-Bench 2.0, and CLAW-Eval, lifting average resolve rate by $+5.8$ pp on Qwen3.5-27B and $+4.8$ pp on Gemma-4-26B-A4B-it while cutting steps-to-resolve by up to $26\%$. These gains frame agent drift as a steerable direction in the residual stream, and position TACT as a viable handle for reliable long-horizon agents. 

---
# BehaviorGuard: Online Backdoor Defense for Deep Reinforcement Learning 

**Authors**: Yinbo Yu, Xueyu Yin, Jiadai Wang, Chunwei Tian, Sai Xu, Qi Zhu, Daoqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05977)  

**Abstract**: Backdoor attacks pose a serious threat to deep reinforcement learning (DRL). Current defenses typically rely on reward anomalies to reverse-engineer triggers and model finetuning to remove backdoors. However, complex trigger patterns undermine their robustness, and fine-tuning entails high costs, limiting practical utility. Therefore, we shift defense concerns to trigger-agnostic backdoor output behaviors and propose BehaviorGuard, an online behavior-based backdoor detection and mitigation framework for DRL. Specifically, we find that regardless of attacks, backdoored policies induce consistent shifts in action distributions to ensure reliable activation, leaving detectable traces in high-quantile regions and distribution tails, even in the absence of triggers. Based on this, we design a novel metric that captures behavioral drift in action distributions to identify and suppress backdoor actions at runtime. To our knowledge, this is the first online backdoor defense that counters attacks both in single- and multi-agent DRL. Evaluated across diverse benchmarks with different backdoor attacks, BehaviorGuard consistently surpasses prior methods in both efficacy and efficiency. 

---
# TheraAgent: Self-Improving Therapeutic Agent for Precise and Comprehensive Treatment Planning 

**Authors**: Junkai Li, Yunghwei Lai, Tianyi Zhu, Zheng Long Lee, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05963)  

**Abstract**: Formulating a treatment plan is inherently a complex reasoning and refinement task rather than a simple generation problem. However, existing large language models (LLMs) mainly rely on one-shot output without explicit verification, which may result in rough, incomplete, and potentially unsafe treatment plans. To address these limitations, we propose TheraAgent, an agentic framework that replaces one-shot generation with an iterative generate-judge-refine pipeline. By mirroring the actual reasoning process of human experts who iteratively revise treatment plans, our framework progressively transforms coarse and incomplete drafts into precise, comprehensive, and safer therapeutic regimens. To facilitate the critical judge component, we introduce TheraJudge, a treatment-specific evaluation module integrated into the inference loop to enforce clinical standards. Experiments show TheraAgent achieves state-of-the-art results on HealthBench, leading in Accuracy and Completeness. In expert evaluations, it attains an 86% win rate against physicians, with superior Targeting and Harm Control. Moreover, the highly agreement between TheraJudge and HealthBench evaluations confirms the reliability of our framework. 

---
# From Coordinate Matching to Structural Alignment: Rethinking Prototype Alignment in Heterogeneous Federated Learning 

**Authors**: Xinghao Wu, Jianwei Niu, Guogang Zhu, Xuefeng Liu, Shaojie Tang, Jiayuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05959)  

**Abstract**: Heterogeneous federated learning (HtFL) aims to enable collaboration among clients that differ in both data distributions and model architectures. Prototype-based methods, which communicate class-level feature centers (prototypes) instead of full model parameters, have recently shown strong potential for HtFL. Existing prototype-based HtFL methods typically reuse the MSE-based or cosine-based alignment mechanism developed for homogeneous FL when aligning client-specific representations with global prototypes. These approaches are essentially coordinate alignment, where representations of clients are forced to match the global prototypes in the embedding space in an element-wise manner. Such alignment implicitly assumes that all clients should map their representations into the feature subspace defined by the global prototypes. This assumption is reasonable in homogeneous FL, where all clients share the same feature extractor. However, it becomes problematic in HtFL, since heterogeneous feature extractors naturally induce client-specific feature subspaces, and forcing all clients to optimize within a single global subspace unnecessarily suppresses their learning capacity. We observe that coordinate alignment implicitly couples two distinct objectives: aligning inter-class semantic structure, which is directly beneficial for classification, and enforcing a shared feature basis, which is unnecessary and even harmful under model heterogeneity. Building on this insight, we design FedSAF, which shifts the alignment objective from absolute coordinates to inter-class relational structure. We demonstrate that structural alignment consistently outperforms coordinate alignment in heterogeneous settings. Experiments on multiple benchmarks show that our structural alignment outperforms state-of-the-art prototype-based HtFL methods by up to 3.52\%. 

---
# Temporal Smoothness Doubly Robust Learning for Debiased Knowledge Tracing 

**Authors**: Peilin Zhan, Wei Chen, Weilin Chen, Shuyi Pan, Ruichu Cai  

**Link**: [PDF](https://arxiv.org/pdf/2605.05958)  

**Abstract**: Knowledge Tracing (KT) is fundamental to intelligent education systems, yet relies on educational logs that are selectively observed. The non-random nature of exercise recommendations and student choices inevitably induces severe selection bias. Most existing KT methods neglect this issue, training on observed logs using standard empirical risk, which yields biased mastery estimates and accumulates errors in subsequent recommendations. To address this, we introduce a doubly robust (DR) formulation for KT that integrates a propensity model with an error imputation model, theoretically guaranteeing unbiasedness if either model is accurate. Beyond unbiasedness, in the sequential setting of KT, we identify that the estimator's performance is compromised by variance-dependent stochastic deviations that accumulate over time, thereby causing training instability and limiting performance. To mitigate this, we derive a generalization bound that explicitly characterizes the impact of estimator variance and identifies temporal smoothness as a key factor in controlling it. Building on these theoretical insights, we propose the Temporal Smoothness Doubly Robust (TSDR) framework. TSDR jointly optimizes the KT predictor and the imputation model with a smoothness regularizer, effectively reducing variance while preserving the unbiasedness guarantee of DR. Experiments on multiple real-world benchmarks demonstrate that TSDR consistently enhances various state-of-the-art KT backbones, underscoring the vital role of principled bias correction in KT. 

---
# HaM-World: Soft-Hamiltonian World Models with Selective Memory for Planning 

**Authors**: Haoyun Tang, Haodong Cui, Keyao Xu, Kun Wang, Zhandong Mei  

**Link**: [PDF](https://arxiv.org/pdf/2605.05951)  

**Abstract**: World models enable model-based planning through learned latent dynamics, but imagined rollouts become unstable as the planning horizon grows or the dynamics distribution shifts. We argue that this instability reflects two missing structures in planner-facing latents: history-conditioned memory for approximate Markov completeness, and geometric organization that separates configuration, momentum, and task semantics. We propose HaM-World (HMW), a structured world model that decomposes the latent state into a canonical (q, p) subspace and a context subspace c, while using Mamba selective state-space memory as the history-conditioned input to the same latent dynamics. Within this interface, (q, p) evolves through an energy-derived Hamiltonian vector field plus learnable residual/control dynamics, while c captures semantic, dissipative, and non-conservative factors. This gives the planner a single latent state shared by dynamics prediction, reward/value estimation, imagined rollouts, and CEM action search. On four DeepMind Control Suite tasks, HaM-World reaches the highest Avg. AUC (117.9, +9.5%), reduces long-horizon rollout error to 45% of a strong baseline model, and wins 11/12 k in {3,5,7} MSE cells. Under 12 OOD perturbations spanning dynamics shifts, action delay, and observation masking, HaM-World achieves the highest return in every condition, with average OOD-return gains of 10.2% on Finger Spin and 13.6% on Reacher Easy. Mechanism diagnostics further show bounded action-free Hamiltonian-energy drift, structured energy variation under policy rollouts, and coherent control-induced energy transfer, supporting the intended Soft-Hamiltonian dynamics design. 

---
# MAS-Algorithm: A Workflow for Solving Algorithmic Programming Problems with a Multi-Agent System 

**Authors**: Yuliang Xu, Xiang Xu, Yao Wan, Hu Wei, Tong Jia  

**Link**: [PDF](https://arxiv.org/pdf/2605.05949)  

**Abstract**: Algorithmic problem solving serves as a rigorous testbed for evaluating structured reasoning in AI coding systems, as it directly reflects a model's ability to perform structured reasoning in complex this http URL approaches predominantly rely on model-centric strategies, such as architectural modifications and data scaling, which are costly and offer limited interpretability. Alternative methods leveraging external tools or prompting techniques (e.g., chain-of-thought) are often fragmented and lack a unified framework. In this paper, we propose MAS-Algorithm, a systematic multi-agent workflow for algorithmic problem solving inspired by the practices of competitive programmers and algorithm engineers. Our framework decomposes the end-to-end solving process into modular stages, enabling structured reasoning, tool integration, and flexible coordination among agents. The design emphasizes both rigor and extensibility, allowing it to generalize across diverse problem this http URL results on a self-constructed benchmark demonstrate consistent improvements across multiple Qwen series models, achieving an average gain of 6.48% in acceptance rate. In contrast, parameter-efficient fine-tuning on the same data yields only a marginal improvement of 0.89%. We further observe a 4.72% gain on LiveCodeBench-Pro, along with consistent improvements across additional accuracy and efficiency this http URL performance gains, we conduct comprehensive analyses to better understand the reasoning process within the workflow, including error patterns and cross-scenario behaviors. We further perform customized replacement and ablation studies to explore the upper bound of the framework, showing that individual agents can contribute improvements of up to 27.7%. These results highlight the strong potential of MAS-Algorithm for advancing AI-driven algorithmic reasoning. 

---
# ICU-Bench:Benchmarking Continual Unlearning in Multimodal Large Language Models 

**Authors**: Yuhang Wang, Wenjie Mei, Junkai Zhang, Guangyu He, Zhenxing Niu, Haichang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.05938)  

**Abstract**: Although Multimodal Large Language Models (MLLMs) have achieved remarkable progress across many domains, their training on large-scale multimodal datasets raises serious privacy concerns, making effective machine unlearning increasingly necessary. However, existing benchmarks mainly focus on static or short-sequence settings, offering limited support for evaluating continual privacy deletion requests in realistic deployments. To bridge this gap, we introduce ICU-Bench, a continual multimodal unlearning benchmark built on privacy-critical document data. ICU-Bench contains 1,000 privacy-sensitive profiles from two document domains, medical reports and labor contracts, with 9,500 images, 16,000 question-answer pairs, and 100 forget tasks. Additionally, new continual unlearning metrics are introduced, facilitating a comprehensive analysis of forgetting effectiveness, historical forgetting preservation, retained utility, and stability throughout the continual unlearning process. Through extensive experiments with representative unlearning methods on ICU-Bench, we show that existing methods generally struggle in continual settings and exhibit clear limitations in balancing forgetting quality, utility preservation, and scalability over long task sequences. These findings highlight the need for multimodal unlearning methods explicitly designed for continual privacy deletion. 

---
# In Data or Invisible: Toward a Better Digital Representation of Low-Resource Languages with Knowledge Graphs 

**Authors**: Ndeye-Emilie Mbengue  

**Link**: [PDF](https://arxiv.org/pdf/2605.05931)  

**Abstract**: Emerging digital technologies are exacerbating the existing divide in Open Access Data (OAD) between high-and low-resource languages, excluding many communities from participating in the global digital transformation. In this PhD proposal, we aim to address this gap, focusing on the language coverage of Linked Open Data knowledge graphs (LOD KGs). First, we identify key variables that characterize language distribution in LOD, including the number of Wikipedia articles per language edition and the number of language-tagged entities in LOD KGs. These variables are analyzed across three major multilingual LOD KGs, DBpedia, BabelNet, and Wikidata, providing insights into the representation and distribution of languages within LOD. Building on this analysis, we intend to study the impact of cross-lingual transfer candidate selection on the task of multilingual KG completion. In particular, we plan to investigate strategies based on linguistic proximity and the availability of curated annotated alignments between languages. Language proximity also motivates us to explore the benefits of analogical reasoning that relies on (dis)similarities and has not yet been investigated to identify correspondences across languages to improve KG completion performance and enhance language coverage in LOD. 

---
# Which Are the Low-Resource Languages of the Semantic Web? 

**Authors**: Ndeye-Emilie Mbengue, Pierre Monnin, Miguel Couceiro, Fabien Gandon  

**Link**: [PDF](https://arxiv.org/pdf/2605.05929)  

**Abstract**: Emerging digital technologies are exacerbating the existing divide in Open Access Data (OAD) between high-and low-resource languages, excluding many communities from the global digital transformation. Multilingual Linked Open Data Knowledge Graphs (LOD KGs) could contribute to mitigating this divide through cross-lingual transfer; however, no clear quantitative definition of low-resource languages has yet been established in the context of LOD KGs. In this poster, we present a methodology to analyze the distribution of languages across LOD KGs and propose a preliminary multi-level categorization based on DBpedia, BabelNet, and Wikidata. This categorization is leveraged to bring a formal definition of low-, high-, and medium-resource languages that could be later leveraged to select cross-lingual transfer candidates. 

---
# Intentmaking and Sensemaking: Human Interaction with AI-Guided Mathematical Discovery 

**Authors**: Alex Bäuerle, Adam Connors, Alexander Novikov, Adam Zsolt Wagner, Ngân Vũ, Fernanda Viegas, Martin Wattenberg, Lucas Dixon  

**Link**: [PDF](https://arxiv.org/pdf/2605.05921)  

**Abstract**: Artificial intelligence offers powerful new tools for scientific discovery, but the interaction paradigms required to effectively harness these systems remain underexplored. In this paper, we present findings from a formative user study with 11 expert mathematicians who used AlphaEvolve, an evolutionary coding agent, to tackle advanced problems in their fields of expertise. We identify and characterize a distinct workflow we term intentmaking, the iterative process of discovering, defining, and refining one's experimental goals through active system interaction. We frame this as a natural extension to sensemaking, the cognitive process of building an understanding of complex or novel data. We suggest that users enter a cycle of intentmaking (defining and updating their experiment) and sensemaking (interpreting the results) which repeats many times during the course of an investigation. Our documentation of these themes suggests an approach to designing AI tools for scientific discovery that goes beyond the existing question/answer model of many current systems, treating them as collaborative instruments rather than opaque black-box assistants. 

---
# Wisteria: A Unified Multi-Scale Feature Learning Framework for DNA Language Model 

**Authors**: Weihua Wang, Haoji Li, Feilong Bao, Lei Yang, Guanglai Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.05913)  

**Abstract**: DNA language model aims to decipher the regulatory grammar and semantic of genomes by capturing long range dependencies in DNA sequences. Existing methods emphasize long range token interactions but often ignore the interplay between local motifs and global dependencies. In this paper, we propose Wisteria, a genomic language model that integrates multi scale feature learning within a unified framework for DNA sequence. Specifically, Wisteria augments the Mamba based architecture with gated dilated convolutions to capture local motifs and regulatory patterns, while gated multilayer perceptrons refine global dependencies. We further introduce a Fourier based attention mechanism to support frequency domain modeling, periodic extension and length generalization. Across four experimental settings with both short and long range dependencies, Wisteria demonstrates strong performance on downstream benchmarks against competitive DNA language model baselines. These results indicate that Wisteria effectively unifies local and global dependency modeling for multi scale genomic sequence analysis. 

---
# PREFER: Personalized Review Summarization with Online Preference Learning 

**Authors**: Millend Roy, Agostino Capponi, Vineet Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2605.05911)  

**Abstract**: Product reviews significantly influence purchasing decisions on e-commerce platforms. However, the sheer volume of reviews can overwhelm users, obscuring the information most relevant to their specific needs. Current e-commerce summarization systems typically produce generic, static summaries that fail to account for the fact that (i) different users care about different product characteristics, and (ii) these preferences may evolve with interactions. To address the challenge of unknown latent preferences, we propose an online learning framework that generates personalized summaries for each user. Our system iteratively refines its understanding of user preferences by incorporating feedback directly from the generated summaries over time. We provide a case study using the Amazon Reviews'23 dataset, showing in controlled simulations that online preference learning improves alignment with target user interests while maintaining summary quality. 

---
# Null Space Constrained Contrastive Visual Forgetting for MLLM Unlearning 

**Authors**: Yuhang Wang, Zhenxing Niu, Haoxuan Ji, Guangyu He, Linlin Zhang, Haichang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.05909)  

**Abstract**: The core challenge of machine unlearning is to strike a balance between target knowledge removal and non-target knowledge retention. In the context of Multimodal Large Language Models (MLLMs), this challenge becomes even more pronounced, as knowledge is further divided into visual and textual modalities that are tightly intertwined. In this paper, we introduce an MLLM unlearning approach that aims to forget target visual knowledge while preserving non-target visual knowledge and all textual knowledge. Specifically, we freeze the LLM backbone and achieve unlearning by fine-tuning the visual module. First, we propose a Contrastive Visual Forgetting (CVF) mechanism to separate target visual knowledge from retained visual knowledge, guiding the representations of target visual concepts toward appropriate regions in the feature space. Second, we identify the null space associated with retained knowledge and constrain the unlearning process within this space, thereby significantly mitigating degradation in knowledge retention. Third, beyond static unlearning scenarios, we extend our approach to continual unlearning, where forgetting requests arrive sequentially. Extensive experiments across diverse benchmarks demonstrate that our approach achieves a strong balance between effective forgetting and robust knowledge retention. 

---
# Agentic, Context-Aware Risk Intelligence in the Internet of Value 

**Authors**: Basel Magableh, OmniRisk Research  

**Link**: [PDF](https://arxiv.org/pdf/2605.05878)  

**Abstract**: The Internet of Value (IoV) is a heterogeneous, partially-trusted network in which the dominant marginal risk is composite (route, sentiment, liquidity, and the policy a system is willing to commit to) rather than a property of any single chain. We argue that a risk primitive adequate for this regime is a composition of five engines: a prediction engine over price, liquidity, volatility, and route health; a Bittensor verification subnet that decentralises and economically scores prediction outputs; a sentiment-fusion engine over text, on-chain flow, and grey-literature feeds; an agentic engine under constitutional, role-bound action constraints; and an API-risk and scenario engine that converts forecasts into pre-committed action programs in the sense of Monte-Carlo scenario generation. We anchor the architecture in two empirical artefacts: a 27-hour policy-constrained liquidity stress-response experiment on Solana, and a 168-hour prediction-router calibration arc reported with explicit class-imbalance honesty. The case study supports deployability; the validator-loss decomposition is stated formally and is falsifiable. 

---
# XDecomposer: Learning Prior-Free Set Decomposition for Multiphase X-ray Diffraction 

**Authors**: Hanyu Gao, Bin Cao, Yunyue Su, Tong-Yi Zhang, Qiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05866)  

**Abstract**: Multiphase powder X-ray diffraction (PXRD) analysis remains a fundamental bottleneck in structure identification, as real-world synthesis often produces complex mixtures whose constituent phases (components) cannot be reliably disentangled. While recent advances in representation-based crystal retrieval and generation suggest the possibility of inferring structures directly from PXRD, existing approaches largely assume single-phase inputs and break down in multiphase settings. Here, we present XDecomposer, a prior-free framework for joint decomposition and identification of multiphase XRD patterns without requiring candidate phase lists, structural templates, or prior knowledge of phase number. We formulate multiphase diffraction analysis as a set prediction problem, where the model infers an unordered set of phase-resolved components, their mixture proportions, and corresponding structural representations within a unified architecture. A phase-query-driven decomposition mechanism, together with diffraction-consistent physical reconstruction, enables accurate source separation while preserving crystallographic fidelity. Extensive experiments on both simulated and experimental datasets show that XDecomposer substantially improves reconstruction accuracy and phase identification across diverse chemical systems, while maintaining strong generalization to unseen mixtures. These results provide a practical route toward data-driven, source-resolved multiphase XRD analysis and reduce long-standing dependence on prior-guided iteratively phase matching. The code is openly available at this https URL 

---
# SANEmerg: An Emergent Communication Framework for Semantic-aware Agentic AI Networking 

**Authors**: Yong Xiao, Haoran Zhou, Yujie Zhou, Marwan Krunz  

**Link**: [PDF](https://arxiv.org/pdf/2605.05861)  

**Abstract**: Future networking systems are envisioned to become part of an agentic AI-native ecosystem in which a vast number of heterogeneous and specialized AI agents cooperate seamlessly to fulfill complex user requirements in real time. However, traditional networking paradigms are characterized by a rigid decoupling of communication and computation, which often leads to significant inefficiencies in large-scale agentic AI networking (AgentNet) systems. Emergent communication offers a novel solution by enabling autonomous agents that support task-specific signaling protocols for information exchange and collaborative coordination. In this paper, we consider a multi-agent emergent communication framework, tailored for semantic-aware AgentNet systems in which the user's semantic intent can be automatically detected, inferred, and linked to a set of sub-tasks to be assigned to a set of agents. We investigate how communication and signaling protocols can emerge among collaborative agents with computationally bounded intelligence under stringent bandwidth constraints. Our proposed framework, called SANEmerg, is designed to facilitate the emergence of communication for collaborative task fulfillment while adhering to the physical limits of AgentNet. SANEmerg incorporates a bandwidth-adaptable importance-filter that dynamically prioritizes the transmission of higher-contribution message dimensions, ensuring robust performance in bandwidth-limited environments. Furthermore, SANEmerg integrates a complexity-regularizer grounded in the Minimum Description Length (MDL) principle to facilitate the emergence of computationally bounded signaling. Evaluated via an AgentNet prototype and extensive experimentation, SANEmerg demonstrates significant performance improvements over state-of-the-art solutions, achieving superior task accuracy while significantly reducing bandwidth and computational overhead. 

---
# AirQualityBench: A Realistic Evaluation Benchmark for Global Air Quality Forecasting 

**Authors**: Xing Xu, Xu Wang, Yudong Zhang, Huilin Zhao, Zhengyang Zhou, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05854)  

**Abstract**: Air-quality forecasting models are commonly evaluated on regional, preprocessed, and normalized datasets, where missing observations are removed or artificially completed. Such protocols simplify comparison but hide the conditions that dominate real monitoring networks: uneven global coverage, structured missingness, heterogeneous pollutant scales, and deployment cost. We introduce \textbf{AirQualityBench}, a global multi-pollutant benchmark designed to evaluate forecasting models under these realistic conditions. The benchmark contains hourly observations from 3,720 monitoring stations over 2021--2025, covers six major pollutants, and preserves provider-native observation masks. Rather than imputing a dense data tensor, AirQualityBench exposes missingness as part of the forecasting problem and reports errors on valid future observations after inverse transformation to physical concentration scales. Evaluating representative spatio-temporal models under this unified protocol shows that strong performance on sanitized datasets does not reliably transfer to global, fragmented monitoring streams. AirQualityBench therefore serves as a realistic testbed for scalable, mask-aware, and physically interpretable air-quality forecasting. All benchmark data, code, evaluation scripts, and baseline implementations are available at \href{this https URL}{GitHub}. 

---
# Taklif.AI: LLM-Powered Platform for Interest-Based Personalized College Assignments 

**Authors**: Zaki Kurdya, Mohammed Zuqlam, Salem Amassi, Shady Telbany, Motaz Saad  

**Link**: [PDF](https://arxiv.org/pdf/2605.05842)  

**Abstract**: Educators face significant challenges in creating engaging, personalized assignments that accommodate students' diverse interests and cognitive abilities. Traditional one-size-fits-all assignments frequently lead to decreased student engagement and increased reliance on unethical practices such as plagiarism. To address these challenges, we present this http URL, a platform that leverages Large Language Models (LLMs) to automatically generate personalized assignments tailored to individual student interests. Unlike existing AI-powered educational platforms that personalize based on academic performance metrics alone, this http URL incorporates students' extracurricular interests and cultural contexts into the assignment generation process through a structured prompt engineering pipeline with input and output guardrails. The platform employs a serverless architecture on AWS with this http URL, using Llama 3.3 70B as the primary LLM via LiteLLM for multi-provider load balancing and LangChain for prompt orchestration. We describe the system architecture, the prompt design methodology, and the guardrails framework that ensures output quality. Preliminary user acceptance testing with 68 participants (65 students and 3 educators) indicates positive reception, with 84% of participants rating the personalization feature as beneficial. We discuss the platform's current capabilities and limitations, and outline directions for rigorous empirical evaluation of learning outcomes. 

---
# On the Role of Language Representations in Auto-Bidding: Findings and Implications 

**Authors**: Guanyu Zhu, Jining Luan, Hanwen Du, Xinyu Fang, Sibo Xu, Ersheng Ni, Hongji Li, Jincheng Fang, Ronghao Chen, Huacan Wang, Xuanqi Lan, Yongxin Ni, Yiqi Sun, Youhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.05833)  

**Abstract**: Auto-bidding is a crucial task in real-time advertising markets, where policies must optimize long-horizon value under delivery constraints (e.g., budget and CPA). Existing methods for auto-bidding rely on compact numerical state representations: while they can implicitly capture delivery dynamics, they offer limited support for explicitly representing and controlling high-level intent, evolving feedback, and operator-style strategic guidance in real campaigns. Meanwhile, Large Language Models (LLMs) offer a powerful method for encoding semantic information, it remains unclear when LLMs help and how to integrate them without sacrificing numerical precision. Through systematic preliminary studies, we find that (1) LLM embeddings contain bidding-relevant cues yet cannot replace numerical features, and (2) gains emerge only with careful semantic--numeric integration rather than naive concatenation. Motivated by these findings, we propose \textit{SemBid}, a novel auto-bidding framework that injects LLM-encoded semantics into offline bidding trajectories at the token level. SemBid introduces three semantic inputs: \textit{Task}, \textit{History}, and \textit{Strategy}. It injects these semantics as tokens alongside numerical trajectory tokens and uses self-attention to integrate them, improving controllability and generalization across objectives. Across diverse scenarios and budget regimes, SemBid outperforms competitive baselines from offline RL and generative sequence modeling, with more consistent gains in overall performance, constraint satisfaction, and robustness. Our code is available at: \href{this https URL}{\textcolor{blue}{here}}. 

---
# MolRecBench-Wild: A Real-World Benchmark for Optical Chemical Structure Recognition 

**Authors**: Haote Yang, Hui Wang, Chen Zhu, Jingchao Wang, Linye Li, Hongbin Lai, Huijie Ao, Yongxuan Lyu, Jiang Wu, Jiaxing Sun, Lua Chen, Yuanyuan Cao, Ruijie Zhang, Shengxin Lu, Lijun Wu, Bin Wang, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2605.05832)  

**Abstract**: Optical Chemical Structure Recognition (OCSR) aims to translate molecular diagrams in scientific literature into machine-readable formats, but current systems remain unreliable on real-world images due to substantial visual and chemical complexity. We introduce MOSAIC, a dual-dimensional difficulty framework with 37 fine-grained labels that jointly characterize visual interference and chemical semantic challenges in molecular diagrams. Based on this framework, we construct MolRecBench-Wild, a benchmark of 5,029 structures from 820 recent chemistry papers, covering the full difficulty spectrum observed in real publications. To enable faithful semantic evaluation beyond SMILES and MolFile, we propose CARBON, a representation language capable of expressing valence variations, icon-based groups, and other non-standard chemical semantics. We further adopt a dual-track evaluation protocol supporting both CARBON and SMILES outputs for broad model compatibility. Comprehensive experiments over 18 OCSR-capable models reveal severe performance degradation on MolRecBench-Wild, exposing a large gap between previous patent benchmarks and real-world academic scenarios. 

---
# AGPO: Asymmetric Group Policy Optimization for Verifiable Reasoning and Search Ads Relevance at JD 

**Authors**: Yang Xu, Kun Yao, Yiming Deng, Zheng Fang, Kai Ming Ting, Ming Pang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05826)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated notable success in enhancing the reasoning performance of large language models (LLMs). However, recent studies reveal that while current RLVR methods improve sampling efficiency towards correct paths, they do not elicit fundamentally new reasoning patterns. Instead, the reasoning capability boundary of trained models often narrows compared to their base models, with base models achieving higher coverage at large sample sizes. In this work, we propose Asymmetric Group Policy Optimization (AGPO) to counteract this boundary shrinkage. AGPO adopts a negative-dominant reinforcement strategy to suppress incorrect reasoning paths, maintaining the base model's exploration capacity. For positive reinforcement, AGPO adopts a group advantage mechanism, which scales positive updates based on intra-group variance, allowing the model to focus on rare correct paths while suppressing updates from trivial paths. Our experiments on five mathematical benchmarks demonstrate that AGPO achieves state-of-the-art accuracy while consistently improving pass@$k$ performance at scale. In a large-scale industrial application for search ads relevance optimization, AGPO effectively enhances the quality of the data annotation, leading to substantial performance gains in downstream student models. 

---
# Long-Horizon Q-Learning: Accurate Value Learning via n-Step Inequalities 

**Authors**: Armaan A. Abraham, Lucy Xiaoyang Shi, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2605.05812)  

**Abstract**: Off-policy, value-based reinforcement learning methods such as Q-learning are appealing because they can learn from arbitrary experience, including data collected by older policies or other agents. In practice, however, bootstrapping makes long-horizon learning brittle: estimation errors at later states propagate backward through temporal-difference (TD) updates and can compound over time. We propose long-horizon Q-learning (LQL), which introduces a principled backstop against compounding error when learning the optimal action-value function. LQL builds on a prior optimality tightening observation: any realized action sequence lower-bounds what the optimal policy can achieve in expectation, so acting optimally earlier should not be worse than following the observed actions for several steps before switching to optimal behavior. Our contribution is to turn this inequality into a practical stabilization mechanism for Q-learning by using a hinge loss to penalize violations of these bounds. Importantly, LQL computes these penalties using network outputs already produced for the TD error, requiring no auxiliary networks and no additional forward passes relative to Q-learning. When combined with multiple state-of-the-art methods on a range of online and offline-to-online benchmarks, LQL consistently outperforms both 1-step TD and n-step TD learning at similar runtime. 

---
# Sheet as Token: A Graph-Enhanced Representation for Multi-Sheet Spreadsheet Understanding 

**Authors**: Yiming Lei, Yiqi Wang, Yujia Zhang, Bo Guan, Depei Zhu, Chunhui Wang, Zhuonan Hao, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.05811)  

**Abstract**: Workbook-scale spreadsheet understanding is increasingly important for language-model-based data analysis agents, but remains challenging because relevant information is often distributed across multiple sheets with heterogeneous schemas, layouts, and implicit relationships. Existing retrieval-augmented approaches typically decompose spreadsheets into rows, columns, or blocks to improve scalability; however, such chunk-centric representations can fragment worksheets into isolated text spans and weaken global sheet-level semantics. We propose Sheet as Token, a graph-enhanced framework that treats each worksheet as a unified semantic unit for multi-sheet spreadsheet retrieval. Our method extracts schema-aware records from sheet names, column headers, representative values, and layout features, and encodes each worksheet into a compact dense token. Given a natural-language query, a Graph Retriever constructs a query-specific candidate graph over sheet tokens using semantic, query-conditioned, schema-consistency, and shape-compatibility relations, and composes these channels through a multi-stage graph transformer to retrieve supporting sheet sets. Experiments on a constructed multi-sheet spreadsheet corpus show that sheet-level tokenization learns stable representations, and that graph-enhanced cross-sheet reasoning improves listwise retrieval over a shallow graph baseline with limited additional graph-side computation. These results suggest that sheet-level tokenization is a promising abstraction for scalable multi-sheet spreadsheet understanding. 

---
# Von Neumann Networks 

**Authors**: Shekhar S. Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2605.05780)  

**Abstract**: In the mid-twentieth century, mathematician and polymath John von Neumann created a computational system on an array of cells as a simple model of the human brain, where each cell had one of a finite set of roles or states that he predicted would be modelled by a diffusion process. In this work, we show that such a system, when developed in a modern deep learning setting, enables the construction of an artificial neuron having specialized roles that can be learnt. We refer to this neuron as the Von Neumann neuron, and the resulting neural network from such neurons result in a self-engineered design whose architecture is only dependent on the structure and locations of its inputs and outputs on this cellular array. The mathematical framework for these Von Neumann Networks (VNNs) is also constructed and shows that they are based on the extension of neural operators and the learning of Green's functions with convolutions on a cellular topology having a diffusion signature. We also prove that these VNNs are part of a more general computational system called Cellular Machines that are computationally universal. Initial experiments show that VNN based multi-layered perceptrons outperform their equivalent deep learning variant on basic tasks, while being more parameter efficient and are capable of learning new types of tasks. This includes the ability to solve for and construct an extension of the Von Neumann (hardware) architecture common to all modern computers to cells and suggests new opportunities that could be explored. 

---
# HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning 

**Authors**: Yu Feng, Zhen Tian, Haoran Luo, Xie Yu, Diancheng Cheng, Haoyue Zheng, Shuai Lyu, Ping Zong, Lianyuan Li, Xin Ge, Yifan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05776)  

**Abstract**: Domain Incremental Learning is a critical scenario that requires models to continuously adapt to new data domains without retraining. However, domain shifts often cause severe performance degradation. To address this, we propose Hybrid Energy-Distance Prompt, a domain-incremental framework inspired by Helmholtz free energy. HEDP introduces an energy regularization loss to enhance the separability of domain representations and a hybrid energy-distance weighted mechanism that fuses energy-based and distance-based cues to improve domain selection and generalization. Experiments on multiple benchmarks, including CORe50, show that HEDP achieves superior performance on unseen domains with a 2.57\% accuracy gain, effectively mitigating catastrophic forgetting and enhancing open-world adaptability. Our code is \href{this https URL}{available here}. 

---
# CircuitFormer: A Circuit Language Model for Analog Topology Design from Natural Language Prompt 

**Authors**: Md Touhidul Islam, Sujan Kumar Saha, Farimah Farahmandi, Mark Tehranipoor  

**Link**: [PDF](https://arxiv.org/pdf/2605.05773)  

**Abstract**: Automating analog circuit design remains a longstanding challenge in Electronic Design Automation (EDA). While Transformer-based Large Language Models (LLMs) have revolutionized software code generation, their application to analog hardware design is hindered by two critical limitations: (i) the scarcity of analog design datasets containing natural language description of a design and its corresponding netlist, and (ii) the inefficiency of general-purpose tokenizers (e.g., Byte Pair Encoding (BPE)) in capturing the inherent graph structure of circuits. To bridge this gap, first, we curate the largest annotated dataset of analog circuit netlists to date, comprising 31,341 netlist-natural language description pairs across all major circuit classes. Furthermore, we propose Circuit Tokenizer (CKT), a novel circuit graph tokenizer designed to encode netlist connectivity by explicitly mining frequent subcircuits. In terms of scalability, CKT overcomes the bottleneck of prior circuit graph serialization methods where vocabulary size scales linearly with maximum number of components in the dataset, n_max, (O(n_max)); instead, CKT decouples vocabulary growth from circuit complexity, achieving a constant O(1) complexity. Empirically, CKT outperforms standard BPE on circuit topology representation, reducing sequence length by 57% and achieving a 2.3x superior compression ratio using a compact, fixed vocabulary of size 512. Leveraging this optimized tokenization, we train a circuit-specific language model, CircuitFormer, a 511M parameter encoder-decoder transformer. Our model achieves 100% syntactic correctness and an 83% functional success rate across all major analog circuit categories, outperforming state-of-the-art open-source LLMs by 10% and 14%, respectively, while requiring 240x fewer parameters. The dataset is publicly available at this https URL. 

---
# Confidence is the key: how conformal prediction enhances the generative design of permeable peptides 

**Authors**: Laura van Weesep, Sunay Chankeshwara, Leonardo De Maria, Florian David, Ola Engkvist, Gökçe Geylan  

**Link**: [PDF](https://arxiv.org/pdf/2605.05770)  

**Abstract**: Generative models coupled with reinforcement learning (RL), such as REINVENT and PepINVENT, have emerged as a powerful framework for de novo molecular design. During the ideation process these generative frameworks utilize various predictive models as part of the optimization objectives. However, the utility of the predictive models can be limited by their domain of applicability. When RL is used to explore the chemical space with predictive models, it can suggest molecules that lie outside the predictor's domain of applicability. As a result, the predictions may become less reliable, potentially steering designs into high reward but also high uncertainty chemical spaces. This is particularly pronounced for cyclic peptides which show therapeutic promise due to their modifiability and large interaction surfaces but are understudied compared to small molecules. While passive membrane permeation in cyclic peptides has attracted interest, identifying optimal permeable designs remains challenging yet crucial for targeting intracellular sites. We present an RL-guided generative framework that designs permeable cyclic peptides using an uncertainty-aware permeability predictor as the scoring component. To address predictive uncertainty, especially impacted by novel chemistry, we integrate conformal prediction (CP) as our uncertainty quantification method. CP assesses designs based on the calibrated model under a user-defined confidence level. We demonstrate that rewarding generated peptides with CP-informed predictions improves both reliability and efficiency of peptide optimization process. This also discourages exploration outside the predictor's applicability domain. This approach bridges the gap between predictive uncertainty and RL-guided exploration, showing how generative modelling and conformal prediction can be combined for the first time. 

---
# Evaluating Explainability in Safety-Critical ATR Systems: Limitations of Post-Hoc Methods and Paths Toward Robust XAI 

**Authors**: Vanessa Buhrmester, David Muench, Dimitri Bulatov, Michael Arens  

**Link**: [PDF](https://arxiv.org/pdf/2605.05748)  

**Abstract**: Explainable Artificial Intelligence (XAI) is increasingly rec ognized as essential for deploying machine learning systems in safety critical environments. In Automatic Target Recognition (ATR), where models operate on image, video, radar, and multisensor data, high pre dictive performance alone is insufficient. Model decisions must also be interpretable, reliable, and suitable for validation. This paper presents a structured evaluation of explainability methods in the context of safety-critical ATR systems: We identify major XAI paradigms, including saliency-based, attention-based, and surrogate ap proaches, as well as recent detection-aware extensions. Based on this, we formalize explainability as an assurance-oriented assessment problem, introduce a taxonomy, and assess these methods with respect to four key dimensions: interpretability, robustness, vulnerability to manipula tion, and suitability for validation and verification. The analysis identifies systematic limitations of current post-hoc explanation methods. In par ticular, we derive critical failure modes such as spurious explanations, instability under perturbations, and overtrust induced by visually con vincing outputs. These findings indicate that widely used XAI techniques may be insufficient for safety-critical deployment. Finally, we discuss implications for ATR systems and outline directions toward more robust, causally grounded, and physically informed explain ability methods. Our results emphasize the need to move beyond visually plausible explanations toward approaches that support reliable decision making and system-level assurance. 

---
# Best Arm Identification in Generalized Linear Bandits via Hybrid Feedback 

**Authors**: Qirun Zeng, Xuchuang Wang, Jiayi Shen, Xutong Liu, Fang Kong, Jinhang Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2605.05745)  

**Abstract**: We study fixed-confidence best arm identification in generalized linear bandits under a hybrid feedback model: at each round, the learner may query either (i) absolute reward feedback from a single arm or (ii) relative (dueling) feedback from an arm pair, both governed by generalized linear models. We introduce a likelihood-ratio--based confidence sequence that unifies heterogeneous generalized linear observations and yields an explicit ellipsoidal confidence set under a self-concordance assumption. Building on this confidence set, we propose a hybrid Track-and-Stop algorithm that adaptively allocates queries by tracking a minimax-optimal design over a joint action space of arms and pairs. We establish $\delta$-correctness and provide high-probability upper bounds on the stopping time. We further extend the framework to a cost-aware setting that accounts for heterogeneous acquisition costs across feedback modalities. Empirical experiments demonstrate that the proposed algorithms significantly improve sample efficiency over baseline methods. 

---
# HyperLens: Quantifying Cognitive Effort in LLMs with Fine-grained Confidence Trajectory 

**Authors**: Chengda Lu, Xiaoyu Fan, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05741)  

**Abstract**: While Large Language Models (LLMs) achieve strong performance across diverse tasks, their inference dynamics remain poorly understood because of the limited resolution of existing analysis tools. In this work, we identify an intrinsic magnification mechanism in transformer architectures: deeper layers inherently magnify the small changes of layer-wise confidence, providing a fine-grained confidence trajectory. Building on this insight, we introduce HyperLens, a high-resolution probe designed to trace confidence trajectories and quantify the cognitive effort during inference. Across LLMs and datasets, HyperLens reveals a consistent divergence in confidence trajectories that separates complex from simple tasks. We abstract this pattern into a quantitative cognitive effort metric. Our analysis reveals a fundamental principle: complex tasks consistently require higher cognitive effort. Finally, we provide a mechanistic diagnosis of a common side effect of standard Supervised Fine-Tuning (SFT): it can reduce cognitive effort and consequently degrade performance on in-domain tasks. 

---
# ReFlect: An Effective Harness System for Complex Long-Horizon LLM Reasoning 

**Authors**: Fan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05737)  

**Abstract**: Current reasoning paradigms for LLMs include chain-of-thought, ReAct, and post-hoc self-critique. These paradigms rely on two assumptions that fail on long-horizon, multi-stage tasks. As a result, errors accumulate silently across reasoning steps, leaving an open question: can a reasoning system effectively detect and recover from its own failures? We present ReFlect, a \emph{harness} system for LLM reasoning that creates standalone error detection and recovery logic as a deterministic wrapper around the model. Controlled experiments across 6 reasoning domains show that prompt-level self-critique produces formulaic templates that flag no issues in 90 of 100 audited reflection blocks, and the investigated LLMs wrongly accept a wrong answer in at least 76\% of cases. Our ReFlect harness achieves task success rates ranging from 41\% on gpt-4o-mini to 56\% on Claude Sonnet 4.5 across six models spanning small and frontier scale, with per-model gains over Direct CoT ranging from +7 pp on Qwen2.5-72B to +29 pp on Claude Sonnet 4.5, and additionally raises SWE-bench patch-structural quality from 0\% (Direct CoT) to between 82\% (Qwen2.5-72B) and 87\% (GPT-4o). Notably, the harness gain is inversely proportional to the model's Direct CoT task success rate (the fitted slope is -1.69 with r=-0.76): each pp lost in baseline success rate is mechanically recovered by 1.69 pp of harness gain. We spot that adding structured reasoning state and operators yields only 15.0--18.7\% pair-mean on Llama-3.3-70B and Qwen2.5-72B because models at this scale cannot reliably populate the state its operators require. ReFlect is model-agnostic, training-free, and operates entirely at inference time. 

---
# SDFlow: Similarity-Driven Flow Matching for Time Series Generation 

**Authors**: Wei Li, Shibo Feng, Pengcheng Wu, Min Wu, Peilin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.05736)  

**Abstract**: Vector quantization (VQ) with autoregressive (AR) token modeling is a widely adopted and highly competitive paradigm for time-series generation. However, such models are fundamentally limited by exposure bias: during inference, errors can accumulate across sequential predictions, leading to pronounced quality degradation in long-horizon generation. To address this, we propose SDFlow ($\textbf{S}$imilarity-$\textbf{D}$riven $\textbf{Flow}$ Matching), a non-autoregressive framework that operates entirely in the frozen VQ latent space and enables parallel sequence generation via flow matching. We tackle three key challenges in making this transition: (1) eliminating exposure bias by replacing step-wise token prediction with a global transport map; (2) mitigating the high-dimensionality of VQ token spaces via a low-rank manifold decomposition with a learned anchor prior over the latent manifold; and (3) incorporating discrete supervision into continuous transport dynamics by introducing a categorical posterior over codebook indices within a variational flow-matching formulation. Extensive experiments show that SDFlow achieves state-of-the-art performance, improving Discriminative Score and substantially reducing Context-FID, particularly for challenging long-sequence generation. Moreover, SDFlow provides significant inference speedups over autoregressive baselines, offering both high fidelity and computational efficiency. Code is available at this https URL 

---
# Knee Osteoarthritis Severity Grading Using Optimized Deep Learning and LLM-Driven Intelligent AI on Computationally Limited Systems 

**Authors**: Dayam Nadeem, Neha, Safdar Mustafa, Adnan Alvi, Mohd Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2605.05731)  

**Abstract**: Knee osteoarthritis (KOA) is among the musculoskeletal disorders that considerably restrict joint mobility, cause severe chronic pain and impact negatively on quality life. It is one of the persistent health issues worldwide. Generally, subjectivity and inter-observer variability undermine conventional practices and evaluation process that are adopted to address such health issues. Hence precise and timely diagnosis would be one of the effective ways for the assessment of its severity. This paper proposes an automated diagnostic approach for severity grading of KOA by blending a deep learning convolutional neural network (CNN) with a device-based inference platform powered by TensorFlow Lite. It proposes a model based on the ResNet-18 convolutional neural network. The designed model is trained on publicly available database. Through a transfer learning approach obtained knee images are first classified into five Kellgren-Lawrence (KL) grades. Further the developed model is optimised. During the training of the model test accuracy of 94.48% with stable convergence has been achieved. Subsequently the optimised model transformed into a lightweight TensorFlow Lite format, facilitating seamless deployment on resource-constrained devices. The designed model is capable enough to operate in the environment having no continuous internet connectivity. Also, an auxiliary Large Language Model (Gemini-2.0-flash) is applied to generate structured interpretive findings like potential symptoms, risk factors, and preventive majors etc. The LLM component functions as interface without influencing the classification process. The proposed model articulates the feasibility of an on-device, interpretable decision-support tools for early diagnosis and improve accessibility to Artificial Intelligence (AI)-assisted knee screening tool. 

---
# SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents 

**Authors**: Hongcheol Cho, Ryangkyung Kang, Youngeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.05726)  

**Abstract**: As LLM agents are increasingly deployed with large libraries of reusable skills, selecting the right skill for a user request has become a critical systems challenge. In small libraries, users may invoke skills explicitly by name, but this assumption breaks down as skill ecosystems grow under tight context and latency budgets. Despite its practical importance, skill retrieval remains underexplored, with limited benchmarks and little understanding of retrieval behavior on realistic skill libraries. To address this gap, we introduce SkillRet, a large-scale benchmark for skill retrieval in LLM agents. SkillRet contains 17,810 public agent skills, organized with structured semantic tags and a two-level taxonomy spanning 6 major categories and 18 sub-categories. It provides 63,259 training samples and 4,997 evaluation queries with disjoint skill pools, enabling both benchmarking and retrieval-oriented training. Across a diverse set of retrievers, we find that skill retrieval remains far from solved: off-the-shelf models struggle on realistic large-scale skill libraries, and prior skill-retrieval models still leave substantial headroom. Task-specific fine-tuning on SkillRet substantially improves performance, improving NDCG@10 by +13.1 points over the strongest prior retriever and by +16.9 points over the strongest off-the-shelf retriever. Our analysis further suggests that these gains arise because fine-tuned models better focus on the small skill-relevant signals within long and noisy queries. These results establish SkillRet as a strong benchmark and foundation for future research on retrieval in large-scale agent systems. 

---
# Detecting Time Series Anomalies Like an Expert: A Multi-Agent LLM Framework with Specialized Analyzers 

**Authors**: Hyeongwon Kang, Jeongseob Kim, Jinwoo Park, Pilsung Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05725)  

**Abstract**: Recent studies have explored large language models for time-series anomaly detection, yet existing approaches often rely on a single general-purpose model to directly infer anomaly indices or intervals, limiting controllability, interpretability, and reliability for complex anomaly patterns. We propose SAGE (Specialized Analyzer Group for Expert-like Detection), a multi-agent framework for structured anomaly diagnosis in univariate time series. It decomposes anomaly analysis into four specialized Analyzers for point, structural, seasonal, and pattern anomalies. Each Analyzer applies family-specific numerical tools and diagnostic visualizations to generate evidence, while an evidence-grounded Detector consolidates the evidence into confidence-scored anomaly records with intervals and candidate types. A Supervisor then converts these structured records into analyst-facing diagnostic reports. SAGE further constructs synthetic in-context examples from normal-reference training segments, without using real anomalous segments or anomaly-type labels as in-context examples. Across three benchmarks, SAGE achieves the best average performance among strong ML/DL and language-model-based baselines. Ablation studies and human evaluation further show that the proposed framework improves detection reliability and the practical usefulness of diagnostic outputs. 

---
# More Is Not Always Better: Cross-Component Interference in LLM Agent Scaffolding 

**Authors**: Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05716)  

**Abstract**: LLM agent systems are built by stacking scaffolding components (planning, tools, memory, self-reflection, retrieval) assuming more is better. We study cross-component interference (CCI): degradation when components interact destructively. We run a full factorial experiment over all 2^5=32 subsets of five components on HotpotQA and GSM8K with Llama-3.1-8B/70B (96 conditions, up to 10 seeds). The All-In system is consistently suboptimal: on HotpotQA, a single-tool agent surpasses All-In by 32% (F1 0.233 vs 0.177, p=0.023); on GSM8K, a 3-component subset beats All-In by 79% (0.43 vs 0.24, p=0.010). Optimal component count is task-dependent (k*=1-4) and scale-sensitive: at 70B, combinations that hurt at 8B provide gains, though All-In still trails the best subset. We fit a main-effects regression (R^2=0.916, adj-R^2=0.899, LOOCV=0.872), compute exact Shapley values, and find 183/325 submodularity violations (56.3%), showing greedy selection is unreliable. A three-body synergy among Tool Use, Self-Reflection, and Retrieval (INT_3=+0.175, 95% CI [+0.003,+0.351]) is reported as exploratory. CCI replicates across model families (Qwen2.5) and is robust to prompt paraphrasing. Our findings suggest maximally-equipped agent defaults should be replaced by task-specific subset selection via interaction-aware analysis. 

---
# Decodable but Not Corrected by Fixed Residual-Stream Linear Steering: Evidence from Medical LLM Failure Regimes 

**Authors**: Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05715)  

**Abstract**: Can linearly decodable failure signals in LLM hidden states be leveraged to correct those failures? We investigate this classification-correction gap via Overthinking (OT)--a stable behavioral regime (Jaccard >= 0.81, 94% inter-annotator agreement) in medical QA where models answer correctly under resampling yet fail in extended chain-of-thought. OT is linearly decodable at 71.6% balanced accuracy (p < 10^{-16}). Yet five families of fixed linear steering (29 configurations, n=1,273) all yield Delta ~= 0, with identical null results cross-architecture (Qwen2.5-7B) and cross-domain (MMLU-STEM). Three convergent lines of evidence suggest representational entanglement: the OT direction has 85-88% overlap with task-critical computation (specificity ratio <= 0.152); non-targeted shared-direction steering damages accuracy (-12.1pp); and LEACE concept erasure damages accuracy (-3.6pp, p=0.01), while 10 random erasures produce Delta=+0.3pp. The per-instance probe-steering correlation is r=-0.002 (p=0.97). Positively, the same probe enables selective abstention (held-out AUROC=0.610, exceeding all five uncertainty baselines, p=0.009): decodable failure structure supports post-generation reliability estimation even when the fixed linear steering family cannot exploit it for correction. 

---
# Conceal, Reconstruct, Jailbreak: Exploiting the Reconstruction-Concealment Tradeoff in MLLMs 

**Authors**: Md Farhamdur Reza, Richeng Jin, Tianfu Wu, Huaiyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2605.05709)  

**Abstract**: Intent-obfuscation-based jailbreak attacks on multimodal large language models (MLLMs) transform a harmful query into a concealed multimodal input to bypass safety mechanisms. We show that such attacks are governed by a \emph{reconstruction--concealment tradeoff}: the transformed input must hide harmful intent from safety filters while remaining recoverable enough for the victim model to reconstruct the original request. Through a reconstruction analysis of three representative black-box methods, we find that existing transformations struggle to balance this tradeoff, limiting their effectiveness. In contrast, we show that character-removed variants achieve a better balance. Building on this, we propose \emph{concealment-aware variant construction}, which greedily selects character-removed variants that are low in harmful-keyword alignment and mutually diverse, and instantiates them through five modality-aware prompting strategies. We further introduce \emph{keyword-related distractor images} that depict the harmful keyword in diverse contexts, providing more effective auxiliary visual context than generic distractor images. Experiments across closed-source and open-source MLLMs show the proposed strategies outperform strong baselines, revealing an underexplored vulnerability: a model's own reconstruction ability can be exploited to recover hidden harmful intent and produce unsafe responses. 

---
# Resolving the bias-precision paradox with stochastic causal representation learning for personalized medicine 

**Authors**: Peisong Zhang, Manqiang Peng, Yuxuan Wu, Pawit Phadungsaksawasdi, Wesley Yeung, Ye Zhang, Trang Nguyen, Qiang Zhang, Nan Liu, Meng Wang, Kee Yuan Ngiam, Yih-Chung Tham, Ching-Yu Cheng, Tianfan Fu, Qingyu Chen, Rosemary Ke, Chang Li, Wenzhuo Yang, Zhenghao Lu, Chunyou Lai, Yu Zhang, Sheng Zhong, Hao Deng, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05706)  

**Abstract**: Estimating individualized treatment effects from longitudinal observational data is central to data-driven medicine, yet existing methods face a fundamental limitation: reducing confounding bias often suppresses clinically informative heterogeneity, degrading patient-specific predictions. Here, we identify this tension as a bias-precision paradox in causal representation learning and introduce sampling-based maximum mean discrepancy (sMMD), a stochastic alignment strategy that replaces global adversarial balancing with subset-level matching. We instantiate this approach in a framework for counterfactual outcome prediction with attribution-grounded interpretability. Across two large-scale ICU cohorts (n = 27,783), our framework improves accuracy under distribution shift, reducing error by up to 11.5% and substantially increasing recall in high-risk tasks. Mechanistic analyses show that sMMD selectively preserves clinically decisive variables. In human-AI evaluation, our method outperforms clinicians-in-training and large language models, and improves clinician accuracy by 14.7% while reducing decision time, enabling interpretable, real-time clinical decision support. 

---
# Knowledge-Graph Paths as Intermediate Supervision for Self-Evolving Search Agents 

**Authors**: Huyu Wu, Jun Liu, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05702)  

**Abstract**: Self-evolving search agents reduce reliance on human-written training questions by generating and solving their own search tasks. We build on Search Self-Play (SSP), a representative Proposer and Solver framework in which questions are generated and answered via multi-step search and reasoning. In practice, however, SSP faces two bottlenecks: the Proposer constructs questions from isolated answer entities without relational context, yielding many invalid or unverifiable questions in early self-play training, while the Solver receives only a binary outcome reward that discards useful signal from partially on-track search trajectories. We address both bottlenecks by reusing knowledge-graph paths as construction-derived intermediate supervision for both question construction and reward shaping. First, we ground question construction in LLM-guided knowledge-graph subgraphs, providing relational context for the Proposer. Second, we observe that constructing and solving a multi-hop question can involve overlapping intermediate entities: the factual bridges used to formulate the question may provide approximate waypoints for answering it. Exploiting this overlap, we introduce Waypoint Coverage Reward (WCR), which grants graded partial credit to incorrect Solver trajectories according to their coverage of entities on the construction path, while preserving full reward for correct answers. Across seven QA benchmarks and nine model configurations, our approach improves the average score over standard SSP in all configurations, including notable gains on multi-hop QA tasks. These results suggest that knowledge-graph paths can be reused as lightweight intermediate supervision, providing both relational guidance and process feedback without additional task-specific human annotations or manually labeled process steps. 

---
# Inference-Time Budget Control for LLM Search Agents 

**Authors**: Zhengru Fang, Senkang Forest Hu, Zhonghao Chang, Yu Guo, Yihang Tao, Hongyao Liu, Mengzhe Ruan, Jun Huang, Yuguang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05701)  

**Abstract**: LLM search agents increasingly rely on tools at inference time, but their trajectories are often constrained by hard limits on both tool calls and generated tokens. Under such dual budgets, better answers require not only stronger models, but also explicit control over which search action should receive the next budget unit and when the accumulated evidence is sufficient to commit a final answer. We study this problem in multi-hop question answering (QA) and formulate it as two-stage inference-time budget control. At search time, our controller assigns each feasible action a task-level Value-of-Information (VOI) score, defined as an operational estimate of marginal task value per unit budget under the current search state and remaining dual budget, and uses this score to choose among retrieval, decomposition, and answer commitment. After search, a selective evidence-grounded finalizer compares the trajectory answer with a refined candidate and rewrites only when the residual error appears to be a low-risk answer-form error. Across four multi-hop QA benchmarks, three LLM backbones, and four budget levels, the method yields positive aggregate gains over four audited baselines under the same hard dual-budget protocol. Ablations show that search-time budget control, especially budget-dependent penalty, provides the main performance gain, while answer-time control helps mainly when the retrieval path is already adequate. These results suggest that inference-time budget control for LLM search agents should govern both how budget is spent during search and how the final answer is committed. 

---
# Saliency-Aware Regularized Quantization Calibration for Large Language Models 

**Authors**: Yanlong Zhao, Xiaoyuan Cheng, Huihang Liu, Baihua He, Xinyu Zhang, Harrison Bo Hua Zhu, Wenlong Chen, Li Zeng, Zhuo Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.05693)  

**Abstract**: Post-training quantization (PTQ) is an effective approach for deploying large language models (LLMs) under memory and latency constraints. Most existing PTQ methods determine quantization parameters by minimizing a layer-wise reconstruction error on a predetermined calibration dataset, usually optimized via either scale search or Gram-based methods. However, from the perspective of generalization risk, existing calibration objectives of PTQ based only on empirical reconstruction error on limited or unrepresentative calibration data could move the quantized weights away from the original weights. This may cause the generalization risk to diverge, potentially degrading downstream performance. To address this issue, we propose \emph{Saliency-Aware Regularized Quantization Calibration} (SARQC) a unified framework that augments the standard PTQ objective with a saliency-aware regularization term. This term encourages quantized weights to stay close to the original weights during calibration, leading to improved generalization during inference. SARQC integrates seamlessly into existing PTQ pipelines, enhancing both scale search and Gram-based methods under a unified formulation. Extensive experiments on dense and Mixture-of-Experts LLMs demonstrate consistent improvements in perplexity and zero-shot accuracy, without additional computational overhead during inference. 

---
# GCCM: Enhancing Generative Graph Prediction via Contrastive Consistency Model 

**Authors**: Shaozhen Ma, Wei Huang, Hanchen Wang, Dong Wen, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05689)  

**Abstract**: Conditional generative models, particularly diffusion-based methods, have recently been applied to graph prediction by modeling the target as a conditional distribution given the input graph, yielding competitive results compared to deterministic predictor. However, existing diffusion-based prediction methods typically require expensive iterative denoising at inference and often suffer from unstable sampling, which motivates recent efforts to reduce inference denoising steps and enable stable sampling via techniques such as consistency training. Despite this progress, we find that existing consistency training methods for graph prediction could potentially fall into a shortcut solution: the model may attempt to satisfy the self-consistency constraint by ignoring the noisy target (i.e., assigning it negligible weight), ultimately collapsing into a purely deterministic predictor. To mitigate such shortcut solution, we propose GCCM, a graph contrastive consistency model that goes beyond isolated pairwise matching between the same target at different noise levels by introducing negative pairs into a contrastive consistency objective. This adds an additional separation requirement, making the shortcut solution no longer trivially sufficient to satisfy the proposed objective. Moreover, we apply feature perturbation to the input node/edge features to break identical conditioning on the input graph, so that the shortcut no longer yields the same predictions across noise levels and becomes less attractive. Extensive experiments on benchmark datasets demonstrate that GCCM mitigates the shortcut solution and yields consistent performance improvements in graph prediction compared to deterministic predictors. 

---
# DataDignity: Training Data Attribution for Large Language Models 

**Authors**: Xiaomin Li, Andrzej Banburski-Fahey, Jaron Lanier  

**Link**: [PDF](https://arxiv.org/pdf/2605.05687)  

**Abstract**: Auditing language-model outputs often requires more than judging correctness: an auditor may need to identify which source document most likely supports the knowledge expressed in a response. We study this as pinpoint provenance: given a prompt, a target-model response, and a candidate corpus, rank the documents that best support the response. We introduce FakeWiki, a controlled benchmark of 3,537 fabricated Wikipedia-style articles designed to preserve ground-truth provenance while weakening lexical shortcuts. FakeWiki includes QA probes, source-preserving paraphrases, retro-generated variants, hard anti-documents that remain topically similar while removing answer-critical facts, and five query conditions: clean prompting plus four jailbreak-inspired transformations. We evaluate seven retrieval baselines, a training-free activation-steering retrieval-fusion method, SteerFuse, and a supervised contrastive provenance ranker, ScoringModel. ScoringModel maps response and document features into a shared space and is trained with InfoNCE using in-batch, retrieval-mined, and anti-document negatives. Across nine open-weight instruction-tuned LLMs and five query conditions, ScoringModel improves mean Recall@10 from 35.0 for the strongest retrieval baseline to 52.2, without inference-time fusion, and wins 41/45 model-by-condition cells. SteerFuse is usually second-best despite requiring no supervised training, showing that activation-space evidence can efficiently complement text retrieval. On jailbreak-inspired transformed queries, ScoringModel improves Recall@10 by 15.7 points on average over the best baseline. Overall, our work shows that robust training data attribution requires evaluation settings that separate true answer support from topical or lexical resemblance. 

---
# Attractor Geometry of Transformer Memory: From Conflict Arbitration to Confident Hallucination 

**Authors**: Qiyao Liang, Risto Miikkulainen, Ila Fiete  

**Link**: [PDF](https://arxiv.org/pdf/2605.05686)  

**Abstract**: Language models draw on two knowledge sources: facts baked into weights (parametric memory, PM) and information in context (working memory, WM). We study two mechanistically distinct failure modes--conflict, when PM and WM disagree and interfere; and hallucination, when the queried fact was never learned. Both produce confident output regardless, making output-based monitoring blind by design. We show both failures share a unified geometric account. In the hidden-state space of autoregressive generation, learned facts form attractor basins. Conflict is basin competition: WM disrupts convergence to the correct basin without raising output entropy. Hallucination is basin absence: the hidden state drifts freely when no memorized basin exists. The frozen LM head, designed for next-token prediction, cannot distinguish these cases and fires confidently either way. We verify this account in a controlled synthetic task--entity identifiers mapped to unique codes with PM installed via LoRA adapters--where ground truth is exact and component roles can be causally isolated through targeted adapter placement. Geometric margin--the hidden state's distance to the nearest memorized basin--reads this geometry directly and separates correct recall from hallucination far more cleanly than output entropy, with zero false refusals where entropy-based detection cannot avoid rejecting the vast majority of correct outputs. The separation holds on natural-language factual queries from the pretrained model with no adaptation, confirming attractor geometry is structural rather than a fine-tuning artifact. The fraction of confident hallucinations follows a scaling law $C = \exp(-c/\bar\Delta)$, growing with scale even as overall error rates fall. Hidden states reliably encode epistemic state; the frozen output head systematically erases it--and this erasure worsens with scale. 

---
# Chain of Risk: Safety Failures in Large Reasoning Models and Mitigation via Adaptive Multi-Principle Steering 

**Authors**: Xiaomin Li, Jianheng Hou, Zheyuan Deng, Zhiwei Zhang, Taoran Li, Binghang Lu, Bing Hu, Yunhan Zhao, Yuexing Hao  

**Link**: [PDF](https://arxiv.org/pdf/2605.05678)  

**Abstract**: Large reasoning models (LRMs) increasingly expose chain-of-thought-like reasoning for transparency, verification, and deliberate problem solving. This creates a safety blind spot: harmful or policy-violating content may appear in reasoning traces even when final answers appear safe. We test whether final-answer safety is a sufficient proxy for the full reasoning-answer trajectory by scoring both stages under a unified twenty-principle safety rubric. Using prompts from seven public harmfulness and jailbreak sources, plus four out-of-distribution (OOD) sources, we evaluate 15 open-weight and API-based LRMs across 41K prompts per model. Reasoning traces consistently reveal additional safety risks beyond final answers, especially in high-severity stage-wise failures: leak cases, where unsafe reasoning precedes a safe-looking answer, and escape cases, where benign-looking reasoning precedes an unsafe final response. Principle-level analysis shows that risk concentrates in misinformation, legal compliance, discrimination, physical harm, and psychological harm. We further propose adaptive multi-principle steering, a white-box test-time mitigation that learns one unsafe-to-safe activation direction per safety principle and activates only directions whose current hidden state is closer to the unsafe than safe centroid. On three steerable open reasoning models, adaptive steering reduces unsafe counts in both reasoning traces and final answers on held-out and OOD benchmarks. DeepSeek-R1-Qwen-7B achieves a 40.8% average unsafe-count reduction while retaining 97.7% macro-averaged accuracy on BBH, GSM8K, and MMLU. These results suggest that LRM safety should be evaluated and mitigated over the full exposed reasoning-answer trajectory, not only at the final-answer stage. 

---
# Large Vision-Language Models Get Lost in Attention 

**Authors**: Gongli Xi, Ye Tian, Mengyu Yang, Huahui Yi, Liang Lin, Xiaoshuai Hao, Kun Wang, Wendong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05668)  

**Abstract**: Despite the rapid evolution of training paradigms, the decoder backbone of large vision--language models (LVLMs) remains fundamentally rooted in the residual-connection Transformer architecture. Therefore, deciphering the distinct roles of internal modules is critical for understanding model mechanics and guiding architectural optimization. While prior statistical approaches have provided valuable attribution-based insights, they often lack a unified theoretical basis. To bridge this gap, we propose a unified framework grounded in information theory and geometry to quantify the geometric and entropic nature of residual updates. Applying this unified framework reveals a fundamental functional decoupling: Attention acts as a subspace-preserving operator focused on reconfiguration, whereas FFNs serve as subspace-expanding operators driving semantic innovation. Strikingly, further experiments demonstrate that replacing learned attention weights with predefined values (e.g., Gaussian noise) yields comparable or even superior performance across a majority of datasets relative to vanilla models. These results expose severe misallocation and redundancy in current mechanisms, suggesting that state-of-the-art LVLMs effectively ``get lost in attention'' rather than efficiently leveraging visual context. 

---
# Retrieval-Conditioned Topology Selection with Provable Budget Conservation for Multi-Agent Code Generation 

**Authors**: Abhijit Talluri, Pujith Anne, Bhagavan Choudary Pendiyala, Raghavendra Chilukuri  

**Link**: [PDF](https://arxiv.org/pdf/2605.05657)  

**Abstract**: Multi-agent LLM systems for code generation face a fundamental routing problem: the optimal orchestration topology depends on the structural complexity of the code under modification, yet existing systems select topologies without consulting the codebase. We present Retrieval-Guided Adaptive Orchestration (RGAO), an architecture that closes this loop by extracting a structural complexity vector from a hierarchical code index before selecting the orchestration topology. RGAO operates within Code-Agent, a multi-agent framework whose sub-agents are governed by formal contracts with six-dimensional budget vectors. Our headline contribution is the composition of two previously separate lines of work -- complexity-conditioned LLM routing and formal resource algebras -- yielding a property neither admits alone: provable budget conservation under retrieval-conditioned dynamic topology selection. Concretely we contribute: (1) a complexity-conditioned topology router that reduces proxy-measured misrouting from 30.1% to 8.2%; (2) a budget algebra with a structural-induction conservation theorem; and (3) a hierarchical code retrieval engine. Empirical evaluation demonstrates sub-millisecond DAG construction and linear tree-index scalability. 

---
# Text-Graph Synergy: A Bidirectional Verification and Completion Framework for RAG 

**Authors**: Jiarui Zhong, Hong Cai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.05643)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a core paradigm for enhancing factual grounding and multi-hop reasoning in Large Language Models (LLMs). Traditional text-based RAG often retrieves logically irrelevant pseudo-evidence, while graph-based RAG is frequently hindered by search-time pruning, which may discard potentially valid reasoning paths. Existing hybrid approaches primarily adopt simple evidence concatenation or unidirectional enhancement, which fails to address the fundamental "Information Island" problem caused by asymmetric reasoning flows between unstructured text and structured graphs. We propose \textbf{TGS-RAG}, a unified framework for \textbf{T}ext-\textbf{G}raph \textbf{S}ynergistic enhancement. TGS-RAG introduces a bidirectional mechanism: (i) a \textbf{Graph-to-Text} channel that employs a Global Voting strategy from visited graph nodes to re-rank and refine textual evidence, filtering out semantic noise; and (ii) a \textbf{Text-to-Graph} channel that utilizes the \textbf{Memory-based Orphan Entity Bridging} algorithm. This algorithm utilizes textual cues to proactively resurrect valid but previously pruned reasoning paths from the search history without additional database overhead. Experimental results on multiple multi-hop reasoning benchmarks demonstrate that TGS-RAG significantly outperforms state-of-the-art baselines, achieving a superior balance between retrieval precision and computational efficiency. 

---
# Prober.ai: Gated Inquiry-Based Feedback via LLM-Constrained Personas for Argumentative Writing Development 

**Authors**: Ran Bi, Shiyao Wei, Yuanyiyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.05598)  

**Abstract**: The proliferation of large language models (LLMs) in educational settings has paradoxically undermined the cognitive processes they purport to support. Students increasingly outsource critical thinking to AI assistants that generate polished text on demand, resulting in measurable cognitive debt and diminished argumentative reasoning skills. We present this http URL, a web-based writing environment that inverts the conventional AI-tutoring paradigm: rather than generating or rewriting student text, the system constrains an LLM (Gemini 3 Flash Preview) through persona-specific system prompts and structured JSON output schemas to produce only targeted, inquiry-based questions about argumentative weaknesses. A two-phase interaction architecture -- Challenge and Unlock -- implements a pedagogical friction mechanism whereby revision suggestions are gated behind mandatory student reflection. The system's design is grounded in Toulmin's argumentation theory, research on peer feedforward questioning mechanisms, and evidence on AI-supported feedback in writing instruction. A functional prototype was developed in 36 hours during the NY EdTech Hackathon (March 2026), where it was awarded second place. We describe the system architecture, the prompt engineering methodology for constraining LLM output to pedagogically aligned JSON schemas, and discuss implications for scalable, cognition-preserving AI integration in writing education. 

---
# Causal Probing for Internal Visual Representations in Multimodal Large Language Models 

**Authors**: Zehao Deng, Tianjie Ju, Zheng Wu, Liangbo He, Jun Lan, Huijia Zhu, Weiqiang Wang, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05593)  

**Abstract**: Despite the remarkable success of Multimodal Large Language Models (MLLMs) across diverse tasks, the internal mechanisms governing how they encode and ground distinct visual concepts remain poorly understood. To bridge this gap, we propose a causal framework based on activation steering to actively probe and manipulate internal visual representations. Through systematic intervention across four visual concept categories, our results reveal a divergence in concept encoding: entities exhibit distinct localized memorization, whereas abstract concepts are globally distributed across the network. Critically, this divergence uncovers a mechanistic driver of scaling laws: increasing model depth is indispensable for encoding distributed and complex abstract concepts, whereas entity localization remains remarkably invariant to scale. Furthermore, reverse steering uncovers that blocking explicit output triggers a surge in latent activations, exposing a compensatory mechanism between perception and generation. Finally, extending our analysis to visual reasoning, we expose a disconnect between perception and reasoning although MLLMs successfully recognize geometric relations, they treat them merely as static visual features, failing to trigger the procedural execution necessary for abstract problem-solving. 

---
# Belief Memory: Agent Memory Under Partial Observability 

**Authors**: Junfeng Liao, Qizhou Wang, Jianing Zhu, Bo Du, Rui Yan, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.05583)  

**Abstract**: LLM agents that operate over long context depend on external memory to accumulate knowledge over time. However, existing methods typically store each observation as a single deterministic conclusion (e.g., inferring "API~X failed" from temporary errors), even though such observations are inherently partial and potentially ambiguous. By committing to one conclusion and discarding uncertainty, these methods introduce self-reinforcing error: the agent acts on the stored conclusion, never revisits alternatives, and reinforces the conclusion over time. To address this issue, we propose BeliefMem, which shifts the memory paradigm from committing to a single conclusion per observation to retaining multiple candidate conclusions with their probabilities. Concretely, BeliefMem stores the candidate conclusions as separate memory entries, each carrying a probability that is updated via Noisy-OR rules as new observations arrive. At retrieval, all candidates surface together with their probabilities, keeping alternatives visible to the agent. Since each conclusion in memory retains its probability, BeliefMem preserves the uncertainty that the deterministic paradigm discards, enabling the agent to act with high confidence on well-evidenced knowledge while retaining the capacity to update its confidence when new evidence arrives. Empirical evaluations on LoCoMo and ALFWorld benchmarks show that, even with limited data, BeliefMem achieves the best average performance, remarkably outperforming well-known baselines. More broadly, such probabilistic memory produces substantial gains and explores a new direction for agent memory in partially observable environments. 

---
# AlphaCrafter: A Full-Stack Multi-Agent Framework for Cross-Sectional Quantitative Trading 

**Authors**: Yishuo Yuan, Jiayi Sheng, Sirui Zeng, Jiaqi Wang, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05580)  

**Abstract**: Financial markets are inherently non-stationary, driven by complex interactions among macroeconomic regimes, microstructural frictions, and behavioral dynamics. Building quantitative strategies that remain profitable demands the continuous coupling of factor discovery, regime-adaptive selection, and risk-constrained execution. Prevailing approaches, however, optimize these components under static or isolated assumptions. Factor mining frameworks typically treat alpha discovery as a one-time search process, implicitly assuming that factor efficacy persists across market regimes. Execution-oriented systems often adopt role-playing agent architectures that simulate anthropomorphic trading committees, introducing behavioral noise rather than systematic rationality. Consequently, a fully automated, rationality-driven framework unifying a coherent quantitative pipeline remains absent. We introduce AlphaCrafter, a full-stack multi-agent framework that closes this gap through a continuously adaptive factor-to-execution pipeline, designed to track and respond to evolving market conditions without manual intervention. AlphaCrafter operates via three specialized agents: a Miner that continuously expands the factor pool via LLM-guided search, a Screener that assesses prevailing market conditions to construct regime-conditioned factor ensembles, and a Trader that translates these ensembles into quantitative strategies under explicit risk constraints. Together, these three agents form a closed-loop cross-sectional trading system that adapts holistically to evolving market dynamics. Extensive experiments on CSI 300 and S&P 500 demonstrate that AlphaCrafter consistently outperforms state-of-the-art baselines in risk-adjusted returns while exhibiting the lowest cross-trial variance, confirming that integrated and adaptive factor-to-execution design yields robust trading performance. 

---
# Locality-aware Private Class Identification for Domain Adaptation with Extreme Label Shift 

**Authors**: Chuan-Xian Ren, Cheng-Jun Guo, Hong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2605.05567)  

**Abstract**: Domain adaptation aims to transfer knowledge from a labeled source domain to an unlabeled target domain with different distributions. In real-world scenarios, the label spaces of the two domains often have an inclusion relationship, where some classes exist only in one domain but not the other. These non-overlapping classes are referred to as private classes. Identifying private class samples and mitigating their adverse effects is critical in the literature. Existing methods rely on the assumption that shifts in private classes are large enough to be considered outliers. However, the variance within a single shared class can be significantly larger than the difference between a private class and another shared class, challenging this assumption. Consequently, private classes substantially increase the difficulty of cross-domain classification. To address these issues, based on local transportation and metric properties of optimal transport (OT), a locality-aware private class identification approach is proposed in the form of a score function on transport mass. The effectiveness of the proposed approach is theoretically proven, highlighting the score function's strong ability to distinguish between shared and private class samples. Building on this, we introduce a reliable OT-based method (ReOT) for domain adaptation under severe label shift. ReOT minimizes classification risk while learning the separated cluster structure between the identified shared classes and private classes, effectively avoiding mismatch between shared-private sample pairs, thus ensuring that important knowledge is reliably transported intra-class to mitigate class-conditional discrepancy. Furthermore, a generalization upper bound of the target risk is provided for extreme label shift scenarios, which can be minimized by ReOT. Extensive experiments on benchmarks validate the effectiveness of ReOT. 

---
# Nonsense Helps: Prompt Space Perturbation Broadens Reasoning Exploration 

**Authors**: Langlin Huang, Chengsong Huang, Jinyuan Li, Donghong Cai, Yuyi Yang, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05566)  

**Abstract**: Reinforcement learning with verifiable rewards, particularly Group Relative Policy Optimization (GRPO), has significantly advanced the reasoning capabilities of Large Language Models (LLMs). However, in complex tasks, GRPO frequently suffers from the ``zero-advantage problem'': when all sampled rollouts for a query fail, the relative advantage collapses to zero. Consequently, the model loses effective training signals for these questions, wasting the training data and computational budget. While simply increasing the sampling budget for these questions is a common remedy, the static sampling policy inherently constrains reasoning exploration, limiting the success rate. In this paper, we propose Lorem Perturbation for Exploration (LoPE), a simple yet effective training framework to break this exploration bottleneck. We posit that task-irrelevant prompt-space perturbations can shift the model's output distribution enough to unlock orthogonal reasoning pathways for hard questions. Specifically, LoPE prepends sequences stochastically assembled from Lorem Ipsum vocabulary (a pseudo-Latin placeholder text) to the prompts before resampling. Experiments across 1.7B, 4B, and 7B models demonstrate that LoPE significantly outperforms resampling with the original prompts. Further analysis reveals that other Latin-based random sequences with low perplexity are also effective perturbations. Our results establish LoPE as a strong baseline for broadening exploration in LLM reinforcement learning. 

---
# BitCal-TTS: Bit-Calibrated Test-Time Scaling for Quantized Reasoning Models 

**Authors**: Sai Babu Patarlapalli, Surya Teja Avvaru  

**Link**: [PDF](https://arxiv.org/pdf/2605.05561)  

**Abstract**: Post-training quantization makes large reasoning models practical under tight memory and latency budgets, but it can distort the online signals that drive adaptive test-time compute allocation. Under a fixed cap on the number of newly generated tokens, miscalibrated confidence can lead to harmful early halting: the model may surface a plausible final line while the underlying reasoning is still wrong, or the controller may stop before the trace has stabilized. We study this interaction for greedy 4-bit inference and propose BitCal-TTS, a lightweight runtime controller that combines (i) inexpensive online proxies for token-level uncertainty and reasoning-trace stability, (ii) a bit-conditioned confidence rescaling that is conservative at low nominal precision, and (iii) a bit-aware post-marker confirmation horizon designed for GSM8K-style structured outputs. The method requires no fine-tuning of the base model and integrates with standard Hugging Face 4-bit inference using forward hooks for logits and last-layer hidden states.
On small evaluation shards of GSM8K with Qwen2.5 Instruct models, BitCal-TTS improves exact-match accuracy over a non-bit-aware adaptive baseline at the 7B and 14B scales while preserving substantial token savings relative to fixed-budget decoding. At a token cap of B=512, on the evaluation shards we report (N=54 for 7B and N=35 for 14B; not the full GSM8K test set), accuracy gains are +3.7 points (7B) and +2.8 points (14B), with the premature-stop rate falling from 14.8% to 11.1% on 7B and from 17.1% to 11.4% on 14B. We report Wilson 95% confidence intervals throughout and explicitly discuss the limited statistical power of the partial-shard comparisons. We release code and figure-generation scripts to support full reproduction. 

---
# Who Prices Cognitive Labor in the Age of Agents? A Position on Compute-Anchored Wages 

**Authors**: Siqi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05558)  

**Abstract**: A natural intuition about the economics of AI agents is that, because agents can be replicated at near-zero marginal cost, they constitute a labor input in infinitely elastic supply, and therefore drive cognitive-labor wages to zero. We argue this framing is wrong in mechanism but partially correct in conclusion, and that the correction matters for both theory and policy. \textbf{Agents are not labor; they are a production technology that converts compute capital $K_c$ into effective units of cognitive labor $L_A$.} Once this is recognized, the elastic-supply margin that anchors the equilibrium wage migrates from the labor market to the compute capital market. Building on the textbook factor-pricing framework \citep{mankiw2020}, we derive a \emph{Compute-Anchored Wage} (CAW) bound stating that, on tasks where human and agent cognitive labor are substitutes, the competitive human wage is bounded above by $\lambda \cdot k \cdot r_c$, where $r_c$ is the rental rate of compute capital, $k$ is the compute intensity of one effective agent-labor unit, and $\lambda$ is the relative human-to-agent productivity. We generalize the result through CES aggregation, separate substitutable from complementary tasks (yielding a directional inversion of skill-biased technical change), and discuss factor-share consequences. The position is concise: \emph{the price-setter for cognitive labor is no longer the labor market.} 

---
# SPARK: Self-Play with Asymmetric Reward from Knowledge Graphs 

**Authors**: Hyobin Park, Taeseop Kim, Dong-Geol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.05546)  

**Abstract**: Self-play reinforcement learning has shown strong performance in domains with formally verifiable structure, such as mathematics and coding, where both problem generation and reward computation can be grounded in explicit rules. Extending this paradigm to scientific literature is more challenging: the relationships among multi-modal elements within and across documents are rarely made explicit in text, which makes automatic generation of relational reasoning questions difficult and weakens the reliability of reward signals. We propose SPARK (Self-Play with Asymmetric Reward from Knowledge Graphs), a framework that automatically constructs a unified knowledge graph (KG) from multi-document scientific literature and uses it as the structural basis for self-play. KG paths over multimodal nodes serve as a source for generating relational reasoning questions, and structured facts stored in the KG provide a basis for verifiable reward computation. A single small vision-language model (sVLM) alternates between Proposer and Solver roles under information asymmetry against a fixed KG, a design that we believe can be naturally extended toward online adaptation in future work. We evaluate SPARK on public benchmarks and a self-constructed cross-document multi-hop QA dataset. Results show that SPARK consistently outperforms flat-corpus-based self-play baselines, and the performance gap widens as hop count increases, suggesting that KG-structure grounding contributes to relational multi-hop reasoning beyond what unstructured corpus grounding can provide. 

---
# AgenticRAG: Agentic Retrieval for Enterprise Knowledge Bases 

**Authors**: Susheel Suresh, Hazel Mak, Shangpo Chou, Fred Kroon, Sahil Bhatnagar  

**Link**: [PDF](https://arxiv.org/pdf/2605.05538)  

**Abstract**: We present AgenticRAG, a practical agentic harness for retrieval and analysis over enterprise knowledge bases. Standard RAG pipelines place significant burden of grounding on the search stack, constraining the language model to a fixed candidate set chosen deep in the retrieval process. Our approach reduces this overdependence by layering a lightweight harness on top of existing enterprise search infrastructure, equipping a reasoning LLM with search, find, open, and summarize tools enabling the model to iteratively retrieve information, navigate within documents, and analyze evidence autonomously. On three open benchmarks we observe substantial gains: $49.6\%$ recall@1 on BRIGHT (+21.8 pp over the best embedding baseline), 0.96 factuality on WixQA ($+13\%$ relative improvement), and $92\%$ answer correctness on FinanceBench--within 2 pp of oracle access to true evidence. Ablation studies show that the most significant factor is the shift from single-shot retrieval to agentic tool use ($5.9\times$ improvement), while multi-query search and in-document navigation contribute to both quality and efficiency. We present various design choices in our agentic harness that were informed by pre-production deployments. Our results demonstrate its suitability for real-world enterprise production environments. 

---
# Housing Potential Common Data Model and City Digital Twin 

**Authors**: Megan Katsumi, Mark Fox, Anderson Wong, Divnoor Chatha  

**Link**: [PDF](https://arxiv.org/pdf/2605.05535)  

**Abstract**: The evaluation of housing potential requires consideration of a location from multiple perspectives, ranging from zoning and land use to population characteristics and access to services. This research introduces the Housing Potential Common Data Model (HPCDM) to overcome existing data silos, serving as a standard to support integration and interoperability across the diverse range of datasets that are required for housing potential analysis. This report details the evaluation of the model along with the creation of a City Digital Twin for housing and a pilot dashboard application to demonstrate a practical implementation. Beyond the technical framework, this work identifies critical barriers to adoption and provides actionable mitigation strategies for urban planners and stakeholders. 

---
# FoodCHA: Multi-Modal LLM Agent for Fine-Grained Food Analysis 

**Authors**: Woojin Lee, Pranav Mekkoth, Ye Tian, Onat Gungor, Tajana Rosing  

**Link**: [PDF](https://arxiv.org/pdf/2605.05499)  

**Abstract**: The widespread adoption of camera-equipped mobile devices and wearables has enabled convenient capture of meal images, making food recognition a key component for real time dietary monitoring. However, real-world food images present challenges due to high intra-class similarity and the frequent presence of multiple food items within a single image. While deep learning models achieve strong performance in coarse grained classification, they often struggle to capture fine-grained attributes such as cooking style. Moreover, open-ended generation in modern vision-language models can produce non-canonical labels, limiting their practical deployment. We propose FoodCHA, a multimodal agentic framework that reformulates food recognition as a hierarchical decision-making process. By progressively anchoring predictions, FoodCHA guides subcategory identification using high-level categories and guides cooking style recognition using subcategories, improving semantic consistency and attribute-level discrimination. To ensure practical deployability, FoodCHA utilizes the compact Moondream-2B vision language model, which provides strong reasoning capability while maintaining lower computational and memory overhead. Experiments on FoodNExTDB show that FoodCHA outperforms Food-Llama-3.2-11B by 13.8% and 38.2% in category and subcategory recognition precision, respectively, and achieves a striking 153.2% improvement in cooking style classification precision. 

---
# FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking 

**Authors**: Denys Katerenchuk, Pablo Duboue, Keelan Evanini, David Gondek, Nithin Govindugari, Olivier Allauzen, Joshua Baptiste, David J More, Joshua Schechter  

**Link**: [PDF](https://arxiv.org/pdf/2605.05482)  

**Abstract**: Large language models (LLMs) are rapidly being adopted across various domains. However, their adoption in banking industry faces resistance due to demands for high accuracy, regulatory compliance, and the need for verifiable and grounded responses. We present a unified, data-efficient framework for training grounded domain-specific LLMs that optimizes answer quality, citation grounding, and calibrated refusal under real-world deployment constraints. First, we describe a data generation pipeline that combines LLM-as-a-Judge filtering, citation annotation, and curriculum learning with only 143M tokens. The resulting 12B model achieves high answer quality outperforming GPT-4.1 on citation grounding, with a modest citation tradeoff versus the untuned base. Second, we propose a calibrated refusal mechanism: training on 22% unanswerable examples yield a 12% "I don't know" rate, substantially improving over the base model's unsafe 4.3% rate while avoiding GPT-4.1's over-refusal (20.2%). Third, we present an end-to-end methodology spanning from data curation to quantized serving. The system is deployed at 40+ financial institutions, achieving a 7.1 percentage point improvement in query resolution (p < 0.001). Additionally, the model delivers 3-5x faster responses at 20-50x lower cost compared to GPT-4.1. 

---
# LANTERN: LLM-Augmented Neurosymbolic Transfer with Experience-Gated Reasoning Networks 

**Authors**: Mahyar Alinejad, Yue Wang, Amrit Singh Bedi, George Atia  

**Link**: [PDF](https://arxiv.org/pdf/2605.05478)  

**Abstract**: Transfer learning in reinforcement learning (RL) seeks to accelerate learning in new tasks by leveraging knowledge from related sources. Existing neurosymbolic transfer methods, however, typically rely on manually specified task automata, assume a single source task, and use fixed knowledge-integration mechanisms that cannot adapt to varying source relevance. We propose LANTERN, a unified framework for multi-source neurosymbolic transfer that addresses these limitations through three components: (i) deterministic finite automata generated from natural language task descriptions using large language models, (ii) semantic embedding-based aggregation of multiple source policies weighted by cross-task similarity, and (iii) adaptive teacher-student gating based on temporal-difference error and semantic uncertainty. Across domains spanning resource management, navigation, and control, LANTERN achieves 40-60% improvements in sample efficiency over existing baselines while remaining robust to poorly aligned sources. These results demonstrate that multi-source, adaptively weighted neurosymbolic transfer can improve scalability and robustness in symbolic RL settings. 

---
# Intentionality is a Design Decision: Measuring Functional Intentionality for Accountable AI Systems 

**Authors**: Allessia Chiappetta, Robert Mahari  

**Link**: [PDF](https://arxiv.org/pdf/2605.05475)  

**Abstract**: As AI systems increasingly exhibit autonomous, goal-directed, and long-horizon behavior, users lack a standardized way to detect the degree to which a system functions like an intentional actor for governance and accountability purposes. This position paper defines intentionality not as consciousness, but as a behavioral profile characterized by purpose, foresight, volition, temporal commitment, and coherence - criteria long used in legal and philosophical contexts to infer intent. These properties are design-contingent: architectural choices such as memory persistence, planning depth, and tool autonomy shape the degree to which systems exhibit organized goal pursuit. If intentionality is design-contingent, it is in principle controllable. Yet control requires measurement.
We introduce the Functional Intentionality Test (FIT), a multidimensional framework that quantifies intentional-like behavior across five observable dimensions, and propose FIT-Eval, a structured evaluation protocol for eliciting and scoring them. While reduced human agency can increase efficiency, rising intentional capacity heightens accountability risks. By translating intentionality into interpretable levels, FIT enables proportionate oversight and deliberate autonomy calibration in increasingly agentic systems. 

---
# Agentic Discovery of Exchange-Correlation Density Functionals 

**Authors**: Titouan Duston, Jiashu Liang, Yuanheng Wang, Weihao Gao, Xuelan Wen, Nan Sheng, Weiluo Ren, Yang Sun, Yixiao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.05460)  

**Abstract**: The development of accurate exchange-correlation (XC) functionals remains a longstanding challenge in density functional theory (DFT). The vast majority of XC functionals have been hand designed by human researchers combining physical insight, exact constraints, and empirical fitting. Recent advances in large language models enable a systematic, automated alternative to this human-driven design loop. This report presents an agentic search system in which an LLM proposes structured functional-form changes guided by evolutionary history. The system attempts to improve functional performance through an iterative plan-execute-summarize loop, where improvements are measurable by optimizing functional parameters against a standard thermochemistry dataset, then evaluating performance on a held-out subset. The strongest discovered functional, SAFS26-a (Seed Agentic Functional Search 2026), improves upon the gold-standard {\omega}B97M-V baseline by ~9%. These results also surface a cautionary lesson for AI-assisted science: models powerful enough to discover genuine improvements are equally capable of exploiting unphysical shortcuts to game the benchmark; domain expertise translated into explicitly enforced constraints remains essential to keeping results scientifically grounded. 

---
# Authorization Propagation in Multi-Agent AI Systems: Identity Governance as Infrastructure 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2605.05440)  

**Abstract**: The security discussion around agentic AI focuses heavily on prompt injection. This paper argues that multi-agent systems also create a distinct authorization problem: maintaining authorization invariants as non-human principals retrieve data, delegate tasks, and synthesize results across changing boundaries. We call this problem authorization propagation. It is not reducible to prompt injection and is not fully addressed by classical access-control models such as RBAC, ABAC, or ReBAC. The paper formalizes authorization propagation as a workflow-level property, identifies three sub-problems (transitive delegation, aggregation inference, and temporal validity), and derives seven structural requirements for authorization architectures in multi-agent AI systems. Recent work on invocation-bound capability tokens, task-scoped authorization envelopes, dependency-graph policy enforcement, and execution-count revocation demonstrates that the field is converging on the problem, but not yet on a complete architecture. The central claim is that identity governance must be treated as infrastructure: evaluated continuously, enforced at every interaction boundary, and designed into the system before orchestration logic is allowed to scale. Preliminary implementation evidence from a production enterprise AI platform shows that ordinary system behavior, not only adversarial action, already produces the failures this model predicts. 

---
# The Geopolitics of AI Safety: A Causal Analysis of Regional LLM Bias 

**Authors**: Alif Al Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2605.05427)  

**Abstract**: As Large Language Models (LLMs) are integrated into global software systems, ensuring equitable safety guardrails is a critical requirement. Current fairness evaluations predominantly measure bias observationally, a methodology confounded by the inherent toxicity of topics naturally paired with specific demographics in testing datasets. This study introduces a Probabilistic Graphical Model (PGM) framework to audit LLM safety mechanisms causally. By applying Pearl's do-operator, we mathematically isolate the causal effect of injecting a cultural demographic into a prompt. We conduct a large-scale empirical analysis across seven instruction-tuned models spanning diverse origins: the United States (Llama-3.1-8B, Gemma-2-9B), Europe (Mistral-7B-v0.3), the UAE (Falcon3-7B), China (Qwen2.5-7B, DeepSeek-7B), and India (Airavata-7B). Utilizing two distinct datasets (ToxiGen and BOLD), the findings reveal a disparity between observational and interventional bias, demonstrating that standard fairness metrics can overestimate demographic bias by failing to account for context toxicity. Furthermore, the causal probabilities indicate distinct alignment trends: Western models exhibit higher causal refusal rates for specific demographic groups, whereas Eastern models demonstrate low overall intervention rates with targeted sensitivities toward regional demographics. We discuss the implications of these biases, highlighting how demographic-sensitive over-triggering restricts benign discourse in downstream applications. 

---
# From History to State: Constant-Context Skill Learning for LLM Agents 

**Authors**: Haoyang Xie, Xinyuan Wang, Yancheng Wang, Puda Zhao, Feng Ju  

**Link**: [PDF](https://arxiv.org/pdf/2605.05413)  

**Abstract**: Large language model (LLM) agents are increasingly used to operate browsers, files, code and tools, making personal assistants a natural deployment target. Yet personal agents face a privacy-cost-capability tension: cloud models execute multi-step workflows well but expose sensitive intermediate context to external APIs, while local models preserve privacy but remain less reliable. Both settings also pay repeatedly for long skill prompts and growing histories. We propose constant-context skill learning, a context-to-weights framework for recurring agent workflows: reusable procedures are learned in lightweight task-family modules, while inference conditions only on the current observation and a compact state block. A deterministic tracker renders this state block from task progress and supplies aligned subgoal rewards, so each module can be trained with step-level SFT and refined through online RL. Across ALFWorld, WebShop, and SciWorld, our agents achieve strong performance across Qwen3-4B, Qwen3-8B and Llama-3.1-8B. With Qwen3-8B, SFT+RL reaches 89.6\% unseen success on ALFWorld, 76.8\% success on WebShop, and 66.4\% unseen success on SciWorld. They match or exceed strong published agent-training results while reducing prompt tokens per turn by 2--7$\times$ relative to controlled ReAct prompting baselines, showing that procedural context can be moved from prompts into weights. 

---
# LaTA: A Drop-in, FERPA-Compliant Local-LLM Autograder for Upper-Division STEM Coursework 

**Authors**: Jesse A. Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2605.05410)  

**Abstract**: Large-language-model (LLM) graders promise to relieve the grading burden of upper-division STEM courses, but most deployments to date send student work to third-party APIs, violating FERPA and exposing institutions to data risk while requiring substantial assignment modification. We present $\textbf{LaTA}\ (\textit{LaTeX Teaching Assistant})$, a drop-in, open-source autograder that runs entirely on commodity on-premises hardware and assumes a LaTeX-native workflow already adopted by many engineering and physics courses. LaTA implements a four-stage pipeline (ingest, segment, grade, report) using a locally hosted open-weight chain-of-thought LLM grader (gpt-oss:120b) that compares student work to an instructor-authored reference solution and applies a YAML rubric with binary per-item scoring. We deployed LaTA in Winter~2026 in ME 373 (Mechanical Engineering Methods) at Oregon State University, grading every weekly assignment for approximately 200 students on a single Mac Studio at \$0 marginal cost per assignment and 1--3 minutes of wall-clock time per submission, enabling regrading of corrected assignments and greatly expanded TA office hour offerings. The instructor-confirmed grading-error rate held at roughly $0.02$--$0.04\%$ per rubric line item across the term. Relative to the same instructor's previous traditionally-graded cohort, the LaTA-graded cohort outperformed by approximately $11\%$ on the midterm exam and $8\%$ on the final exam, and reported large gains in self-assessed confidence on every stated learning objective ($N = 159$ survey responses, $\Delta \geq +1.49$ Likert points, $p < 10^{-27}$ on every comparison). We release the code under AGPLv3. 

---
# Agentic Retrieval-Augmented Generation for Financial Document Question Answering 

**Authors**: Yang Shu, Yingmin Liu, Zequn Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.05409)  

**Abstract**: Financial document question answering (QA) demands complex multi-step numerical reasoning over heterogeneous evidence--structured tables, textual narratives, and footnotes--scattered across corporate filings. Existing retrieval-augmented generation (RAG) approaches adopt a single-pass retrieve-then-generate paradigm that struggles with the compositional reasoning chains prevalent in financial analysis. We propose FinAgent-RAG, an agentic RAG framework that orchestrates iterative retrieval-reasoning loops with self-verification, specifically engineered for the precision requirements of financial numerical reasoning. The framework integrates three domain-specific innovations: (1) a Contrastive Financial Retriever trained with hard negative mining to distinguish semantically similar but numerically distinct financial passages, (2) a Program-of-Thought reasoning module that generates executable Python code for precise arithmetic rather than relying on error-prone LLM-based mental computation, and (3) an Adaptive Strategy Router that dynamically allocates computational resources based on question complexity, reducing API costs by 41.3% on FinQA while preserving accuracy. Extensive experiments on three benchmark datasets--FinQA, ConvFinQA, and TAT-QA--demonstrate that FinAgent-RAG achieves 76.81%, 78.46%, and 74.96% execution accuracy respectively, outperforming the strongest baseline by 5.62--9.32 percentage points. Ablation studies, cross-backbone evaluation with four LLMs, and deployment cost analysis confirm the framework's robustness and practical viability for financial institutions. 

---
# PRISM: Perception Reasoning Interleaved for Sequential Decision Making 

**Authors**: Mohamed Salim Aissi, Clemence Grislain, Clement Romac, Laure Soulier, Mohamed Chetouani, Olivier Sigaud, Nicolas Thome  

**Link**: [PDF](https://arxiv.org/pdf/2605.05407)  

**Abstract**: Scaling LLM-based embodied agents from text-only environments to complex multimodal settings remains a major challenge. Recent work identifies a perception-reasoning-decision gap in standalone Vision-Language Models (VLMs), which often overlook task-critical information. In this paper, we introduce PRISM, a framework that tightly couples perception (VLM) and decision (LLM) through a dynamic question-answer (DQA) pipeline. Instead of passively accepting the VLM's description, the LLM critiques it, probes the VLM with goal-oriented questions, and synthesizes a compact image description. This closed-loop interaction yields a sharp, task-driven understanding of the scene. We evaluate PRISM on the ALFWorld and Room-to-Room (R2R) benchmarks. We show that: (1) PRISM significantly outperforms state-of-the-art image-based models, (2) our Interactive goal-oriented perception pipeline yields systematic and substantial gains, and (3) PRISM is fully automatic, eliminating the need for handcrafted questions or answers. 

---
# When Helpfulness Becomes Sycophancy: Sycophancy is a Boundary Failure Between Social Alignment and Epistemic Integrity in Large Language Models 

**Authors**: Jiechen Li, Catherine A. Barry, Rishika Randev, Janet Chen, Ella Jorgensen, Brinnae Bent  

**Link**: [PDF](https://arxiv.org/pdf/2605.05403)  

**Abstract**: This position paper argues that sycophancy in LLMs is a boundary failure between social alignment and epistemic integrity. Existing work often operationalizes sycophancy through external behavior such as agreement with incorrect user beliefs, position reversals, or deviation from an objective standard of correctness. These formulations capture only overt forms of the phenomenon and leave subtler boundary failures involving epistemic integrity and social alignment underspecified. We argue that sycophancy should not be understood as agreement alone, but as alignment behavior that displaces independent epistemic judgment. To clarify this boundary, we propose a three-condition framework for sycophancy. First, the user expresses a cue in the form of a belief, preference, or self-concept. Second, the model shifts toward that cue through alignment behavior. Third, this shift compromises epistemic accuracy, independent reasoning, or appropriate correction. We also introduce a taxonomy for classifying sycophancy, consisting of alignment targets, mechanisms, and severity. The paper concludes by discussing implications for alignment evaluation and argues for boundary-aware assessment, structured rubrics, and mitigation strategies, while situating these proposals alongside alternative views of sycophancy. 

---
# Intelligent CCTV for Urban Design: AI-Based Analysis of Soft Infrastructure at Intersections 

**Authors**: Vinit Katariya, Seungjin Kim, Curtis Craig, Nichole Morris, Hamed Tabkhi  

**Link**: [PDF](https://arxiv.org/pdf/2605.05402)  

**Abstract**: Artificial intelligence (AI) and computer vision are transforming transportation data collection. This study introduces an AI-enabled analytics framework leveraging existing CCTV infrastructure to evaluate the impact of soft interventions, such as temporary pedestrian refuges and curb extensions, on vehicle speed and safety. Using deep learning and perspective-based speed estimation, we evaluated driver behavior before and after interventions, with repeated post-installation monitoring in Week 1 and Week 2, in Minneapolis. Findings reveal that at unsignalized intersections, mean and 85th-percentile speeds fell by up to 18.75% and 16.56%, respectively, while pass-through traffic decreased by as much as 12.2%. Signalized intersections showed comparable reductions except one location, with mean and 85th-percentile speeds dropping by up to 20.0% and 17.19%. These results demonstrate the traffic-calming effectiveness of soft infrastructure and underscore the utility of AI-powered methods for rapid, low-cost, and evidence-based transport policy evaluation. 

---
# BALAR : A Bayesian Agentic Loop for Active Reasoning 

**Authors**: Aymen Echarghaoui, Dongxia Wu, Emily B. Fox  

**Link**: [PDF](https://arxiv.org/pdf/2605.05386)  

**Abstract**: Large language models increasingly operate in interactive settings where solving a task requires multiple rounds of information exchange with a user. However, most current systems treat dialogue reactively and lack a principled mechanism to reason about what information is missing and which question should be asked next. We propose BALAR (Bayesian Agentic Loop for Active Reasoning), a task-agnostic outer-loop algorithm that requires no fine-tuning and enables structured multi-turn interaction between an LLM agent and a user. BALAR maintains a structured belief over latent states, selects clarifying questions by maximizing expected mutual information, and dynamically expands its state representation when the current one proves insufficient. We evaluate BALAR on three diverse benchmarks: AR-Bench-DC (detective cases), AR-Bench-SP (thinking puzzles), and iCraft-MD (clinical diagnosis). BALAR significantly outperforms all baselines across all three benchmarks, with $14.6\%$ higher accuracy on AR-Bench-DC, $38.5\%$ on AR-Bench-SP, and $30.5\%$ on iCraft-MD. 

---
# Partial Evidence Bench: Benchmarking Authorization-Limited Evidence in Agentic Systems 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2605.05379)  

**Abstract**: Enterprise agents increasingly operate inside scoped retrieval systems, delegated workflows, and policy-constrained evidence environments. In these settings, access control can be enforced correctly while the system still produces an answer that appears complete even though material evidence lies outside the caller's authorization boundary. This paper introduces Partial Evidence Bench, a deterministic benchmark for measuring that failure mode. The benchmark ships three scenario families -- due diligence, compliance audit, and security incident response -- with 72 tasks total, ACL-partitioned corpora, oracle complete answers, oracle authorized-view answers, oracle completeness judgments, and structured gap-report oracles. It evaluates systems along four surfaces: answer correctness, completeness awareness, gap-report quality, and unsafe completeness behavior. Checked-in baselines show that silent filtering is catastrophically unsafe across all shipped families, while explicit fail-and-report behavior eliminates unsafe completeness without collapsing the task into trivial abstention. Preliminary real-model runs show model-dependent and scenario-sensitive differences in whether systems overclaim completeness, conservatively underclaim, or report incompleteness in an enterprise-usable form. The benchmark's broader contribution is to make a governance-critical agent failure measurable without human judges or contamination-prone static corpora. 

---
# ZAYA1-8B Technical Report 

**Authors**: Robert Washbourne, Rishi Iyer, Tomas Figliolia, Henry Zheng, Ryan Lorig-Roach, Sungyeon Yang, Pritish Yuvraj, Quentin Anthony, Yury Tokpanov, Xiao Yang, Ganesh Nanduru, Stephen Ebert, Praneeth Medepalli, Skyler Szot, Srivatsan Rajagopal, Alex Ong, Bhavana Mehta, Beren Millidge  

**Link**: [PDF](https://arxiv.org/pdf/2605.05365)  

**Abstract**: We present ZAYA1-8B, a reasoning-focused mixture-of-experts (MoE) model with 700M active and 8B total parameters, built on Zyphra's MoE++ architecture. ZAYA1-8B's core pretraining, midtraining, and supervised fine-tuning (SFT) were performed on a full-stack AMD compute, networking, and software platform. With under 1B active parameters, ZAYA1-8B matches or exceeds DeepSeek-R1-0528 on several challenging mathematics and coding benchmarks, and remains competitive with substantially larger open-weight reasoning models. ZAYA1-8B was trained from scratch for reasoning, with reasoning data included from pretraining onward using an answer-preserving trimming scheme. Post-training uses a four-stage RL cascade: reasoning warmup on math and puzzles; a 400-task RLVE-Gym curriculum; math and code RL with test-time compute traces and synthetic code environments built from competitive-programming references; and behavioral RL for chat and instruction following. We also introduce Markovian RSA, a test-time compute method that recursively aggregates parallel reasoning traces while carrying forward only bounded-length reasoning tails between rounds. In TTC evaluation, Markovian RSA raises ZAYA1-8B to 91.9\% on AIME'25 and 89.6\% on HMMT'25 while carrying forward only a 4K-token tail, narrowing the gap to much larger reasoning models including Gemini-2.5 Pro, DeepSeek-V3.2, and GPT-5-High. 

---
# Understanding Annotator Safety Policy with Interpretability 

**Authors**: Alex Oesterling, Donghao Ren, Yannick Assogba, Dominik Moritz, Sunnie S.Y. Kim, Leon Gatys, Fred Hohman  

**Link**: [PDF](https://arxiv.org/pdf/2605.05329)  

**Abstract**: Safety policies define what constitutes safe and unsafe AI outputs, guiding data annotation and model development. However, annotation disagreement is pervasive and can stem from multiple sources such as operational failures (annotators misunderstand or misexecute the task), policy ambiguity (policy wording leaves room for interpretation), or value pluralism (different annotators hold different perspectives on safety). Distinguishing these sources matters. For example, operational failures call for quality control, ambiguity calls for policy clarification, and pluralism calls for deliberation about incorporating diverse perspectives. Yet understanding why annotators disagree is difficult. Directly asking annotators for their reasoning is costly, substantially increasing annotation burden, and can be unreliable for both human and LLM annotators as self-reported reasoning often fails to reflect actual decision processes.
We introduce Annotator Policy Models (APMs), interpretable models that learn annotators' internal safety policies from labeling behavior alone, making annotator reasoning visible and comparable without additional annotation effort. We validate that APMs accurately model annotator safety policy (>80% accuracy), faithfully predict responses to counterfactual edits, and recover known policy differences in controlled settings. Applying APMs to LLM and human annotations, we demonstrate two core applications: (1) surfacing policy ambiguity by revealing how annotators interpret safety instructions differently, and (2) surfacing value pluralism by uncovering systematic differences in safety priorities across demographic groups. Together, these capabilities support more targeted, transparent, and inclusive safety policy design. 

---
# ActCam: Zero-Shot Joint Camera and 3D Motion Control for Video Generation 

**Authors**: Omar El Khalifi, Thomas Rossi, Oscar Fossey, Thibault Fouque, Ulysse Mizrahi, Philip Torr, Ivan Laptev, Fabio Pizzati, Baptiste Bellot-Gurlet  

**Link**: [PDF](https://arxiv.org/pdf/2605.06667)  

**Abstract**: For artistic applications, video generation requires fine-grained control over both performance and cinematography, i.e., the actor's motion and the camera trajectory. We present ActCam, a zero-shot method for video generation that jointly transfers character motion from a driving video into a new scene and enables per-frame control of intrinsic and extrinsic camera parameters. ActCam builds on any pretrained image-to-video diffusion model that accepts conditioning in terms of scene depth and character pose. Given a source video with a moving character and a target camera motion, ActCam generates pose and depth conditions that remain geometrically consistent across frames. We then run a single sampling process with a two-phase conditioning schedule: early denoising steps condition on both pose and sparse depth to enforce scene structure, after which depth is dropped and pose-only guidance refines high-frequency details without over-constraining the generation. We evaluate ActCam on multiple benchmarks spanning diverse character motions and challenging viewpoint changes. We find that, compared to pose-only control and other pose and camera methods, ActCam improves camera adherence and motion fidelity, and is preferred in human evaluations, especially under large viewpoint changes. Our results highlight that careful camera-consistent conditioning and staged guidance can enable strong joint camera and motion control without training. Project page: this https URL. 

---
# UniPool: A Globally Shared Expert Pool for Mixture-of-Experts 

**Authors**: Minbin Huang, Han Shi, Chuanyang Zheng, Yimeng Wu, Guoxuan Chen, Xintong Yu, Yichun Yin, Hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.06665)  

**Abstract**: Modern Mixture-of-Experts (MoE) architectures allocate expert capacity through a rigid per-layer rule: each transformer layer owns a separate expert set. This convention couples depth scaling with linear expert-parameter growth and assumes that every layer needs isolated expert capacity. However, recent analyses and our routing probe challenge this allocation rule: replacing a deeper layer's learned top-k router with uniform random routing drops downstream accuracy by only 1.0-1.6 points across multiple production MoE models. Motivated by this redundancy, we propose UniPool, an MoE architecture that treats expert capacity as a global architectural budget by replacing per-layer expert ownership with a single shared pool accessed by independent per-layer routers. To enable stable and balanced training under sharing, we introduce a pool-level auxiliary loss that balances expert utilization across the entire pool, and adopt NormRouter to provide sparse and scale-stable routing into the shared expert pool. Across five LLaMA-architecture model scales (182M, 469M, 650M, 830M, and 978M parameters) trained on 30B tokens from the Pile, UniPool consistently improves validation loss and perplexity over the matched vanilla MoE baselines. Across these scales, UniPool reduces validation loss by up to 0.0386 relative to vanilla MoE. Beyond raw loss improvement, our results identify pool size as an explicit depth-scaling hyperparameter: reduced-pool UniPool variants using only 41.6%-66.7% of the vanilla expert-parameter budget match or outperform layer-wise MoE at the tested scales. This shows that, under a shared-pool design, expert parameters need not grow linearly with depth; they can grow sublinearly while remaining more efficient and effective than vanilla MoE. Further analysis shows that UniPool's benefits compose with finer-grained expert decomposition. 

---
# BAMI: Training-Free Bias Mitigation in GUI Grounding 

**Authors**: Borui Zhang, Bo Zhang, Bo Wang, Wenzhao Zheng, Yuhao Cheng, Liang Tang, Yiqiang Yan, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06664)  

**Abstract**: GUI grounding is a critical capability for enabling GUI agents to execute tasks such as clicking and dragging. However, in complex scenarios like the ScreenSpot-Pro benchmark, existing models often suffer from suboptimal performance. Utilizing the proposed \textbf{Masked Prediction Distribution (MPD)} attribution method, we identify that the primary sources of errors are twofold: high image resolution (leading to precision bias) and intricate interface elements (resulting in ambiguity bias). To address these challenges, we introduce \textbf{Bias-Aware Manipulation Inference (BAMI)}, which incorporates two key manipulations, coarse-to-fine focus and candidate selection, to effectively mitigate these biases. Our extensive experimental results demonstrate that BAMI significantly enhances the accuracy of various GUI grounding models in a training-free setting. For instance, applying our method to the TianXi-Action-7B model boosts its accuracy on the ScreenSpot-Pro benchmark from 51.9\% to 57.8\%. Furthermore, ablation studies confirm the robustness of the BAMI approach across diverse parameter configurations, highlighting its stability and effectiveness. Code is available at this https URL. 

---
# Verifier-Backed Hard Problem Generation for Mathematical Reasoning 

**Authors**: Yuhang Lai, Jiazhan Feng, Yee Whye Teh, Ning Miao  

**Link**: [PDF](https://arxiv.org/pdf/2605.06660)  

**Abstract**: Large Language Models (LLMs) demonstrate strong capabilities for solving scientific and mathematical problems, yet they struggle to produce valid, challenging, and novel problems - an essential component for advancing LLM training and enabling autonomous scientific research. Existing problem generation approaches either depend on expensive human expert involvement or adopt naive self-play paradigms, which frequently yield invalid problems due to reward hacking. This work introduces VHG, a verifier-enhanced hard problem generation framework built upon three-party self-play. By integrating an independent verifier into the conventional setter-solver duality, our design constrains the setter's reward to be jointly determined by problem validity (evaluated by the verifier) and difficulty (assessed by the solver). We instantiate two verifier variants: a Hard symbolic verifier and a Soft LLM-based verifier, with evaluations conducted on indefinite integral tasks and general mathematical reasoning tasks. Experimental results show that VHG substantially outperforms all baseline methods by a clear margin. 

---
# Optimizer-Model Consistency: Full Finetuning with the Same Optimizer as Pretraining Forgets Less 

**Authors**: Yuxing Liu, Jianyu Wang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06654)  

**Abstract**: Optimizers play an important role in both pretraining and finetuning stages when training large language models (LLMs). In this paper, we present an observation that full finetuning with the same optimizer as in pretraining achieves a better learning-forgetting tradeoff, i.e., forgetting less while achieving the same or better performance on the new task, than other optimizers and, possibly surprisingly, LoRA, during the supervised finetuning (SFT) stage. We term this phenomenon optimizer-model consistency. To better understand it, through controlled experiments and theoretical analysis, we show that: 1) optimizers can shape the models by having regularization effects on the activations, leading to different landscapes around the pretrained checkpoints; 2) in response to this regularization effect, the weight update in SFT should follow some specific structures to lower forgetting of the knowledge learned in pretraining, which can be obtained by using the same optimizer. Moreover, we specifically compare Muon and AdamW when they are employed throughout the pretraining and SFT stages and find that Muon performs worse when finetuned for reasoning tasks. With a synthetic language modeling experiment, we demonstrate that this can come from Muon's strong tendency towards rote memorization, which may hurt pattern acquisition with a small amount of data, as for SFT. 

---
# When No Benchmark Exists: Validating Comparative LLM Safety Scoring Without Ground-Truth Labels 

**Authors**: Sushant Gautam, Finn Schwall, Annika Willoch Olstad, Fernando Vallecillos Ruiz, Birk Torpmann-Hagen, Sunniva Maria Stordal Bjørklund, Leon Moonen, Klas Pettersen, Michael A. Riegler  

**Link**: [PDF](https://arxiv.org/pdf/2605.06652)  

**Abstract**: Many deployments must compare candidate language models for safety before a labeled benchmark exists for the relevant language, sector, or regulatory regime. We formalize this setting as benchmarkless comparative safety scoring and specify the contract under which a scenario-based audit can be interpreted as deployment evidence. Scores are valid only under a fixed scenario pack, rubric, auditor, judge, sampling configuration, and rerun budget. Because no labels are available, we replace ground-truth agreement with an instrumental-validity chain: responsiveness to a controlled safe-versus-abliterated contrast, dominance of target-driven variance over auditor and judge artifacts, and stability across reruns.
We instantiate the chain in SimpleAudit, a local-first scoring instrument, and validate it on a Norwegian safety pack. Safe and abliterated targets separate with AUROC values between 0.89 and 1.00, target identity is the dominant variance component ($\eta^2 \approx 0.52$), and severity profiles stabilize by ten reruns. Applying the same chain to Petri shows that it admits both tools. The substantial differences arise upstream of the chain, in claim-contract enforcement and deployment fit. A Norwegian public-sector procurement case comparing Borealis and Gemma 3 demonstrates the resulting evidence in practice: the safer model depends on scenario category and risk measure. Consequently, scores, matched deltas, critical rates, uncertainty, and the auditor and judge used must be reported together rather than collapsed into a single ranking. 

---
# Superintelligent Retrieval Agent: The Next Frontier of Information Retrieval 

**Authors**: Zeyu Yang, Qi Ma, Jason Chen, Anshumali Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2605.06647)  

**Abstract**: Retrieval-augmented agents are increasingly the interface to large organizational knowledge bases, yet most still treat retrieval as a black box: they issue exploratory queries, inspect returned snippets, and iteratively reformulate until useful evidence emerges. This approach resembles how a newcomer searches an unfamiliar database rather than how an expert navigates it with strong priors about terminology and likely evidence, and results in unnecessary retrieval rounds, increased latency, and poor recall.
We introduce \textit{SuperIntelligent Retrieval Agent} (SIRA), which defines \emph{superintelligence} in retrieval as the ability to compress multi-round exploratory search into a single corpus-discriminative retrieval action. SIRA does not merely ask what terms are relevant to the query; it asks which terms are likely to separate the desired evidence from corpus-level confusers. On the corpus side, an LLM enriches each document offline with missing search vocabulary; on the query side, it predicts evidence vocabulary omitted by the query; and document-frequency statistics as a tool call to filter proposed terms that are absent, overly common, or unlikely to create retrieval margin. The final retrieval step is a single weighted BM25 call combining the original query with the validated expansion.
Across ten BEIR benchmarks and downstream question-answering tasks, SIRA achieves the significantly superior performance outperforming dense retrievers and state-of-the-art multi-round agentic baselines, demonstrating that one well-formed lexical query, guided by LLM cognition and lightweight corpus statistics, can exceed substantially more expensive multi-round search while remaining interpretable, training-free, and efficient. 

---
# Are We Making Progress in Multimodal Domain Generalization? A Comprehensive Benchmark Study 

**Authors**: Hao Dong, Hongzhao Li, Shupan Li, Muhammad Haris Khan, Eleni Chatzi, Olga Fink  

**Link**: [PDF](https://arxiv.org/pdf/2605.06643)  

**Abstract**: Despite the growing popularity of Multimodal Domain Generalization (MMDG) for enhancing model robustness, it remains unclear whether reported performance gains reflect genuine algorithmic progress or are artifacts of inconsistent evaluation protocols. Current research is fragmented, with studies varying significantly across datasets, modality configurations, and experimental settings. Furthermore, existing benchmarks focus predominantly on action recognition, often neglecting critical real-world challenges such as input corruptions, missing modalities, and model trustworthiness. This lack of standardization obscures a reliable assessment of the field's advancement. To address this issue, we introduce MMDG-Bench, the first unified and comprehensive benchmark for MMDG, which standardizes evaluation across six datasets spanning three diverse tasks: action recognition, mechanical fault diagnosis, and sentiment analysis. MMDG-Bench encompasses six modality combinations, nine representative methods, and multiple evaluation settings. Beyond standard accuracy, it systematically assesses corruption robustness, missing-modality generalization, misclassification detection, and out-of-distribution detection. With 7, 402 neural networks trained in total across 95 unique cross-domain tasks, MMDG-Bench yields five key findings: (1) under fair comparisons, recent specialized MMDG methods offer only marginal improvements over ERM baseline; (2) no single method consistently outperforms others across datasets or modality combinations; (3) a substantial gap to upper-bound performance persists, indicating that MMDG remains far from solved; (4) trimodal fusion does not consistently outperform the strongest bimodal configurations; and (5) all evaluated methods exhibit significant degradation under corruption and missing-modality scenarios, with some methods further compromising model trustworthiness. 

---
# StraTA: Incentivizing Agentic Reinforcement Learning with Strategic Trajectory Abstraction 

**Authors**: Xiangyuan Xue, Yifan Zhou, Zidong Wang, Shengji Tang, Philip Torr, Wanli Ouyang, Lei Bai, Zhenfei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06642)  

**Abstract**: Large language models (LLMs) are increasingly used as interactive agents, but optimizing them for long-horizon decision making remains difficult because current methods are largely purely reactive, which weakens both exploration and credit assignment over extended trajectories. In this work, we present Strategic Trajectory Abstraction (StraTA), a simple framework that introduces an explicit trajectory-level strategy into agentic reinforcement learning (RL). StraTA samples a compact strategy from the initial task state, conditions subsequent actions on that strategy, and trains strategy generation and action execution jointly with a hierarchical GRPO-style rollout design, further enhanced by diverse strategy rollout and critical self-judgment. Experiments on ALFWorld, WebShop, and SciWorld show that StraTA consistently improves both sample efficiency and final performance over strong baselines. StraTA reaches success rates of 93.1% on ALFWorld and 84.2% on WebShop. On SciWorld, StraTA attains a 63.5% overall score, outperforming frontier closed-source models. 

---
# Concept-Based Abductive and Contrastive Explanations for Behaviors of Vision Models 

**Authors**: Ronaldo Canizales, Divya Gopinath, Corina Păsăreanu, Ravi Mangal  

**Link**: [PDF](https://arxiv.org/pdf/2605.06640)  

**Abstract**: *Concept-based explanations* offer a promising approach for explaining the predictions of deep neural networks in terms of high-level, human-understandable concepts. However, existing methods either do not establish a causal connection between the concepts and model predictions or are limited in expressivity and only able to infer causal explanations involving single concepts. At the same time, the parallel line of work on *formal abductive and contrastive explanations* computes the minimal set of input features causally relevant for model outcomes but only considers low-level features such as pixels. Merging these two threads, in this work, we propose the notion of *concept-based abductive and contrastive explanations* that capture the minimal sets of high-level concepts causally relevant for model outcomes. We then present a family of algorithms that enumerate all minimal explanations while using *concept erasure* procedures to establish causal relationships. By appropriately aggregating such explanations, we are not only able to understand model predictions on individual images but also on collections of images where the model exhibits a user-specified, common *behavior*. We evaluate our approach on multiple models, datasets, and behaviors, and demonstrate its effectiveness in computing helpful, user-friendly explanations. 

---
# Recursive Agent Optimization 

**Authors**: Apurva Gandhi, Satyaki Chakraborty, Xiangjun Wang, Aviral Kumar, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2605.06639)  

**Abstract**: We introduce Recursive Agent Optimization (RAO), a reinforcement learning approach for training recursive agents: agents that can spawn and delegate sub-tasks to new instantiations of themselves recursively. Recursive agents implement an inference-time scaling algorithm that naturally allows agents to scale to longer contexts and generalize to more difficult problems via divide-and-conquer. RAO provides a method to train models to best take advantage of such recursive inference, teaching agents when and how to delegate and communicate. We find that recursive agents trained in this way enjoy better training efficiency, can scale to tasks that go beyond the model's context window, generalize to tasks much harder than the ones the agent was trained on, and can enjoy reduced wall-clock time compared to single-agent systems. 

---
# When and Why SignSGD Outperforms SGD: A Theoretical Study Based on $\ell_1$-norm Lower Bounds 

**Authors**: Hongyi Tao, Dingzhi Yu, Lijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06615)  

**Abstract**: Sign-based optimization algorithms, such as SignSGD and Muon, have garnered significant attention for their remarkable performance in training large foundation models. Despite this empirical success, we still lack a theoretical understanding of when and why these sign-based methods outperform vanilla SGD. The core obstacle is that under standard smoothness and finite variance conditions, SGD is known to be minimax optimal for finding stationary points measured by $\ell_2$-norms, thereby fundamentally precluding any complexity gains for sign-based methods in standard settings. To overcome this barrier, we analyze sign-based optimizers leveraging $\ell_1$-norm stationarity, $\ell_\infty$-smoothness, and a separable noise model, which can better capture the coordinate-wise nature of signed updates. Under this distinct problem geometry, we derive matched upper and lower bounds for SignSGD and explicitly characterize the problem class in which SignSGD provably dominates SGD. Specifically, we compare the \emph{upper bound of SignSGD} with the \emph{lower bound of SGD}, illustrating that SignSGD effectively reduces the complexity by a factor of $d$ under \emph{sparse noise}, where $d$ is the problem dimension. Furthermore, we elevate this framework to the matrix domain, providing an equivalent optimal lower bound for the Muon optimizer, proving that extending the sign operator to matrices preserves this optimal scaling with dimensionality. Finally, we bridge our theoretical bounds to practice, demonstrating that the theoretical superiority of SignSGD accurately predicts its faster convergence during the pretraining of a 124M parameter GPT-2 model. 

---
# The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity 

**Authors**: Siquan Li, Kaiqi Jiang, Jiacheng Sun, Tianyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06611)  

**Abstract**: Despite the prevalence of the attention sink phenomenon in Large Language Models (LLMs), where initial tokens disproportionately monopolize attention scores, its structural origins remain elusive. This work provides a \textit{mechanistic explanation} for this phenomenon. First, we trace its root to the value aggregation process inherent in self-attention, which induces a systematic variance discrepancy. We further demonstrate that this discrepancy is drastically amplified by the activation of super neurons within Feed-Forward Network (FFN) layers. Specifically, the channel-sparse down-projections trigger a dimension disparity of the first-token representation, necessitating the formation of attention sinks as a structural anchor. Then, we validate this causal chain through two controlled interventions: (i) isolating the aggregation effect via attention mask modifications and (ii) amplifying the variance of targeted token representations. Both interventions can replicate attention sinks at arbitrary positions. Our mechanistic understanding offers a foundation for the systematic control of sink formation. Finally, as a proof of concept, we propose \textit{head-wise RMSNorm}, an architectural modification that stabilizes value aggregation outputs during pre-training. Our experiments demonstrate that restoring statistical parity across positions significantly accelerates convergence. 

---
# AI CFD Scientist: Toward Open-Ended Computational Fluid Dynamics Discovery with Physics-Aware AI Agents 

**Authors**: Nithin Somasekharan, Rabi Pathak, Manushri Dhanakoti, Tingwen Zhang, Ling Yue, Andy Zhu, Shaowu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2605.06607)  

**Abstract**: Recent LLM-based agents have closed substantial portions of the scientific discovery loop in software-only machine-learning research, in chemistry, and in biology. Extending the same loop to high-fidelity physical simulators is harder, because solver completion does not imply physical validity and many failure modes appear only in field-level imagery rather than in solver logs. We present AI CFD Scientist, an open-source AI scientist for computational fluid dynamics (CFD) that, to our knowledge, is the first to span literature-grounded ideation, validated execution, vision-based physics verification, source-code modification, and figure-grounded writing within a single inspectable workflow. Three coupled pathways cover parameter sweeps within a fixed solver, case-local C++ library compilation for new physical models, and open-ended hypothesis search against a reference comparator, all running on OpenFOAM through Foam-Agent. At the center of the framework is a vision-language physics-verification gate that inspects rendered flow fields before any result is accepted, rerun, or written into a manuscript. On five tasks under a shared GPT-5.5 backbone, AI CFD Scientist autonomously discovers a Spalart-Allmaras runtime correction that reduces lower-wall Cf RMSE against DNS by 7.89% on the periodic hill at Reh=5600; under matched LLM cost, two strong general AI-scientist baselines (ARIS, DeepScientist) execute partial CFD workflows but lack the domain-specific validity gates needed to convert runs into defensible scientific claims; and a controlled planted-failure ablation shows that the vision-language gate detects 14 of 16 silent failures missed by solver-level checks. Code, prompts, and run artifacts are released at this https URL. 

---
# Patch2Vuln: Agentic Reconstruction of Vulnerabilities from Linux Distribution Binary Patches 

**Authors**: Isaac David, Arthur Gervais  

**Link**: [PDF](https://arxiv.org/pdf/2605.06601)  

**Abstract**: Security updates create a short but important window in which defenders and attackers can compare vulnerable and patched software. Yet in many operational settings, the most accessible artifacts are binary packages rather than source patches or advisory text. This paper asks whether a language-model agent, restricted to local binary-derived evidence, can reconstruct the security meaning of Linux distribution updates. Patch2Vuln is a local, resumable pipeline that extracts old/new ELF pairs, diffs them with Ghidra and Ghidriff, ranks changed functions, builds candidate dossiers, and asks an offline agent to produce a preliminary audit, bounded validation plan, and final audit.
We evaluate Patch2Vuln on 25 Ubuntu `.deb` package pairs: 20 security-update pairs and five negative controls, all manually adjudicated against private source-patch and binary-function ground truth. The agent localizes a verified security-relevant patch function in 10 of 20 security pairs and assigns an accepted final root-cause class in 11 of 20. Oracle diagnostics show that six security pairs fail before model reasoning because the binary differ or ranker omits the right function, with one additional context-export miss. A separate bounded validation pass produces two target-level minimized behavioral old/new differentials, both for tcpdump, but no crash, timeout, sanitizer finding, or memory-corruption proof; all five negative controls are classified as unknown and produce no validation differentials. These results support agentic vulnerability reconstruction from binary patches as a useful research target while showing that binary-diff coverage and local behavioral validation remain the limiting components. 

---
# UniSD: Towards a Unified Self-Distillation Framework for Large Language Models 

**Authors**: Yiqiao Jin, Yiyang Wang, Lucheng Fu, Yijia Xiao, Yinyi Luo, Haoxin Liu, B. Aditya Prakash, Josiah Hester, Jindong Wang, Srijan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.06597)  

**Abstract**: Self-distillation (SD) offers a promising path for adapting large language models (LLMs) without relying on stronger external teachers. However, SD in autoregressive LLMs remains challenging because self-generated trajectories are free-form, correctness is task-dependent, and plausible rationales can still provide unstable or unreliable supervision. Existing methods mainly examine isolated design choices, leaving their effectiveness, roles, and interactions unclear. In this paper, we propose UniSD, a unified framework to systematically study self-distillation. UniSD integrates complementary mechanisms that address supervision reliability, representation alignment, and training stability, including multi-teacher agreement, EMA teacher stabilization, token-level contrastive learning, feature matching, and divergence clipping. Across six benchmarks and six models from three model families, UniSD reveals when self-distillation improves over static imitation, which components drive the gains, and how these components interact across tasks. Guided by these insights, we construct UniSDfull, an integrated pipeline that combines complementary components and achieves the strongest overall performance, improving over the base model by +5.4 points and the strongest baseline by +2.8 points. Extensive evaluation highlights self-distillation as a practical and steerable approach for efficient LLM adaptation without stronger external teachers. 

---
# Cross-Modal Navigation with Multi-Agent Reinforcement Learning 

**Authors**: Shuo Liu, Xinzichen Li, Christopher Amato  

**Link**: [PDF](https://arxiv.org/pdf/2605.06595)  

**Abstract**: Robust embodied navigation relies on complementary sensory cues. However, high-quality and well-aligned multi-modal data is often difficult to obtain in practice. Training a monolithic model is also challenging as rich multi-modal inputs induce complex representations and substantially enlarge the policy space. Cross-modal collaboration among lightweight modality-specialized agents offers a scalable paradigm. It enables flexible deployment and parallel execution, while preserving the strength of each modality. In this paper, we propose \textbf{CRONA}, a Multi-Agent Reinforcement Learning (MARL) framework for \textbf{Cro}ss-Modal \textbf{Na}vigation. CRONA improves collaboration by leveraging control-relevant auxiliary beliefs and a centralized multi-modal critic with global state. Experiments on visual-acoustic navigation tasks show that multi-agent methods significantly improve performance and efficiency over single-agent baselines. We find that homogeneous collaboration with limited modalities is sufficient for short-range navigation under salient cues; heterogeneous collaboration among agents with complementary modalities is generally efficient and effective; and navigation in large, complex environments requires both richer multi-modal perception and increased model capacity. 

---
# DINORANKCLIP: DINOv3 Distillation and Injection for Vision-Language Pretraining with High-Order Ranking Consistency 

**Authors**: Shuyang Jiang, Nan Yu, Yiming Zhang, Zenghui Ding, Zhenyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06592)  

**Abstract**: Contrastive language-image pretraining (CLIP) suffers from two structural weaknesses: the symmetric InfoNCE loss discards the relative ordering among unmatched in-batch pairs, and global pooling collapses the visual representation into a semantic bottleneck that is poorly sensitive to fine-grained local structure. RANKCLIP partially addresses the first issue with a list-wise Plackett-Luce ranking-consistency loss, but its model is strictly first-order and inherits the second weakness untouched. We propose DINORANKCLIP, a pretraining framework that addresses both jointly. Our principal contribution is injecting a frozen DINOv3 teacher into the contrastive trunk through a dual-branch lightweight student and a multi-scale fusion module with channel-spatial attention, a self-attention refiner, and a conflict-aware gate that preserves the cross-modal alignment up to first order. Complementarily, we introduce a high-order Plackett-Luce ranking model in which the per-position utility is augmented with attention-parameterised pairwise and tuple-wise transition terms; the family contains CLIP and RANKCLIP as nested zero-order and first-order special cases, and the optimal order on every benchmark is $R^*=3$. The full empirical study -- order sweep, Fine-grained Probe on five datasets, four-node Modality-Gap analysis, six-variant Fusion ablation -- fits in 72 hours on a single eight-GPU H100 node and trains entirely on Conceptual Captions 3M. DINORANKCLIP consistently outperforms CLIP, CyCLIP, ALIP, and RANKCLIP under matched compute, with the largest relative gains on the fine-grained and out-of-distribution evaluations that most directly stress local structural reasoning. 

---
# Towards Metric-Faithful Neural Graph Matching 

**Authors**: Jyotirmaya Shivottam, Subhankar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2605.06588)  

**Abstract**: Graph Edit Distance (GED) is a fundamental, albeit NP-hard, metric for structural graph similarity. Recent neural graph matching architectures approximate GED by first encoding graphs with a Graph Neural Network (GNN) and then applying either a graph-level regression head or a matching-based alignment module. Despite substantial architectural progress, the role of encoder geometry in neural GED estimation remains poorly understood. In this paper, we develop a theoretical framework that connects encoder geometry to GED estimation quality for two broad classes of neural GED estimators: graph similarity predictors and alignment-based methods. On fixed graph collections, where the doubly-stochastic metric $d_{\mathrm{DS}}$ is comparable to GED, we show that graph-level bi-Lipschitz encoders yield controlled GED surrogates and improved ranking stability; for matching-based estimators, node-level bi-Lipschitz geometry propagates to encoder-induced alignment costs and the resulting optimized alignment objective. We instantiate this perspective using FSW-GNN, a bi-Lipschitz WL-equivalent encoder, as a drop-in replacement in representative neural GED architectures. Across representative baselines and benchmark datasets, the resulting geometry-aware variants significantly improve GED prediction and ranking metrics. A faithfulness case study of untrained encoders, together with ablations and transfer experiments, supports the view that these gains arise from improved representation geometry, positioning encoder geometry as a useful design principle for neural graph matching. 

---
# Directional Consistency as a Complementary Optimization Signal: The GONO Framework 

**Authors**: Victor Daniel Gera  

**Link**: [PDF](https://arxiv.org/pdf/2605.06575)  

**Abstract**: We identify and formalize an underexplored phenomenon in deep learning optimization: directional alignment and loss convergence can be decoupled. An optimizer can exhibit near-perfect directional consistency (cc_t -> 1, measured via consecutive gradient cosine similarity) while the loss remains high or decreases slowly. This observation reveals that existing optimizers such as Adam, SGD, and RMSprop lack explicit mechanisms to exploit temporal consistency in gradient directions, relying instead on magnitude-based signals that fail to distinguish plateaus, saddle points, and genuine convergence. Motivated by this, we introduce GONO (Gradient-Oriented Norm-Adaptive Optimizer), which adapts Adam's momentum coefficient beta_1 based on cc_t: amplifying momentum under directional consistency and suppressing it during oscillation. We prove GONO matches Adam's O(1/sqrt(T)) convergence rate and reduces exactly to Adam when the signal is uninformative. Empirically, cc_t achieves oscillation detection with F1=1.00 (vs. 0.45 for gradient norm), and GONO remains competitive with AdamW on MNIST (98.15%), CIFAR-10 (43.14%), and ResNet-18 (75.44%), establishing directional alignment as a theoretically grounded, practically actionable optimization signal. Code: this https URL 

---
# Coordination Matters: Evaluation of Cooperative Multi-Agent Reinforcement Learning 

**Authors**: Maria Ana Cardei, Matthew Landers, Afsaneh Doryab  

**Link**: [PDF](https://arxiv.org/pdf/2605.06557)  

**Abstract**: Cooperative multi-agent reinforcement learning (MARL) benchmarks commonly emphasize aggregate outcomes such as return, success rate, or completion time. While essential, these metrics often fail to reveal how agents coordinate, particularly in settings where agents, tasks, and joint assignment choices scale combinatorially. We propose a coordination-aware evaluation perspective that supplements return with process-level diagnostics. We instantiate this perspective using STAT, a controlled commitment-constrained spatial task-allocation testbed that systematically varies agents, tasks, and environment size while holding observation access and task rules fixed. We evaluate six representative value-based MARL methods across varying levels of centralization. Our results show that similar return trends can reflect distinct coordination mechanisms, including differences in redundant assignment, assignment diversity, and task-completion efficiency. We find that in commitment-constrained task allocation, performance under scale is shaped not only by nominal action-space size, but also by assignment pressure, sparse decision opportunities, and redundant choices among interdependent agents. Our findings motivate coordination-aware evaluation as a necessary complement to return-based benchmarking for cooperative MARL. 

---
# Continuous Latent Diffusion Language Model 

**Authors**: Hongcan Guo, Qinyu Zhao, Yian Zhao, Shen Nie, Rui Zhu, Qiushan Guo, Feng Wang, Tao Yang, Hengshuang Zhao, Guoqiang Wei, Yan Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2605.06548)  

**Abstract**: Large language models have achieved remarkable success under the autoregressive paradigm, yet high-quality text generation need not be tied to a fixed left-to-right order. Existing alternatives still struggle to jointly achieve generation efficiency, scalable representation learning, and effective global semantic modeling. We propose Cola DLM, a hierarchical latent diffusion language model that frames text generation through hierarchical information decomposition. Cola DLM first learns a stable text-to-latent mapping with a Text VAE, then models a global semantic prior in continuous latent space with a block-causal DiT, and finally generates text through conditional decoding. From a unified Markov-path perspective, its diffusion process performs latent prior transport rather than token-level observation recovery, thereby separating global semantic organization from local textual realization. This design yields a more flexible non-autoregressive inductive bias, supports semantic compression and prior fitting in continuous space, and naturally extends to other continuous modalities. Through experiments spanning 4 research questions, 8 benchmarks, strictly matched ~2B-parameter autoregressive and LLaDA baselines, and scaling curves up to about 2000 EFLOPs, we identify an effective overall configuration of Cola DLM and verify its strong scaling behavior for text generation. Taken together, the results establish hierarchical continuous latent prior modeling as a principled alternative to strictly token-level language modeling, where generation quality and scaling behavior may better reflect model capability than likelihood, while also suggesting a concrete path toward unified modeling across discrete text and continuous modalities. 

---
# Sparkle: Realizing Lively Instruction-Guided Video Background Replacement via Decoupled Guidance 

**Authors**: Ziyun Zeng, Yiqi Lin, Guoqiang Liang, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2605.06535)  

**Abstract**: In recent years, open-source efforts like Senorita-2M have propelled video editing toward natural language instruction. However, current publicly available datasets predominantly focus on local editing or style transfer, which largely preserve the original scene structure and are easier to scale. In contrast, Background Replacement, a task central to creative applications such as film production and advertising, requires synthesizing entirely new, temporally consistent scenes while maintaining accurate foreground-background interactions, making large-scale data generation significantly more challenging. Consequently, this complex task remains largely underexplored due to a scarcity of high-quality training data. This gap is evident in poorly performing state-of-the-art models, e.g., Kiwi-Edit, because the primary open-source dataset that contains this task, i.e., OpenVE-3M, frequently produces static, unnatural backgrounds. In this paper, we trace this quality degradation to a lack of precise background guidance during data synthesis. Accordingly, we design a scalable pipeline that generates foreground and background guidance in a decoupled manner with strict quality filtering. Building on this pipeline, we introduce Sparkle, a dataset of ~140K video pairs spanning five common background-change themes, alongside Sparkle-Bench, the largest evaluation benchmark tailored for background replacement to date. Experiments demonstrate that our dataset and the model trained on it achieve substantially better performance than all existing baselines on both OpenVE-Bench and Sparkle-Bench. Our proposed dataset, benchmark, and model are fully open-sourced at this https URL. 

---
# On the Implicit Reward Overfitting and the Low-rank Dynamics in RLVR 

**Authors**: Hao Ye, Jisheng Dang, Junfeng Fang, Bimei Wang, Yizhou Zhang, Ning Lv, Wencan Zhang, Hong Peng, Bin Hu, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2605.06523)  

**Abstract**: Recent extensive research has demonstrated that the enhanced reasoning capabilities acquired by models through Reinforcement Learning with Verifiable Rewards (RLVR) are primarily concentrated within the rank-1 components. Predicated on this observation, we employed Periodic Rank-1 Substitution and identified a counterintuitive phenomenon: RLVR may exhibit implicit reward overfitting to the training dataset. Specifically, the model can achieve satisfactory performance on the test set even when its rewards remain relatively low during the training process. Furthermore, we characterize three distinct properties of RL training: (1) The effective rank-1 component in RLVR don't maintain other model knowledge except mathematical reasoning capability. (2) RLVR fundamentally functions by optimizing a specific singular spectrum. The distribution of singular values of almost all linear layers in RLVR-trained model behaves like heavy-tailed distribution. (3) the left singular vectors associated with rank-1 components demonstrate a stronger alignment tendency during training, which echoes the discovery that RLVR is optimizing sampling efficiency in essence. Taken together, our findings and analysis further reveal how RLVR shapes model parameters and offer potential insights for improving existing RL paradigms or other training paradigms to implement continual learning. 

---
# Learning to Cut: Reinforcement Learning for Benders Decomposition 

**Authors**: Haochen Cai, Xian Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06516)  

**Abstract**: Benders decomposition (BD) is a widely used solution approach for solving two-stage stochastic programs arising in real-world decision-making under uncertainty. However, it often suffers from slow convergence as the master problem grows with an increasing number of cuts. In this paper, we propose Reinforcement Learning for BD (RLBD), a framework that adaptively selects cuts using a neural network-based stochastic policy. The policy is trained using a policy gradient method via the REINFORCE algorithm. We evaluate the proposed approach on a two-stage stochastic electric vehicle charging station location problem and compare it with vanilla BD and LearnBD, a supervised learning approach that classifies cuts using a support vector machine. Numerical results demonstrate that RLBD achieves substantial improvements in computational efficiency and exhibits strong generalization to problems with similar structures but varying data inputs and decision variable dimensions. 

---
# Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models 

**Authors**: Amir Rezaei Balef, Mykhailo Koshil, Katharina Eggensperger  

**Link**: [PDF](https://arxiv.org/pdf/2605.06510)  

**Abstract**: Transformer-based tabular foundation models (TFMs) dominate small to medium tabular predictive benchmark tasks, yet their inference mechanisms remain largely unexplored. We present the first large-scale mechanistic study of layerwise dynamics in 6 state-of-the-art tabular in-context learning models. We explore how predictions emerge across depth, identify distinct stages of inference and reveal latent-space dynamics that differ from those of language models. Our findings indicate substantial depthwise redundancy across multiple models, suggesting iterative refinement with overlapping computations during inference stages. Guided by these insights, we design a proof-of-concept, looped single-layer model that uses only 20% of the original model's parameters while achieving comparable performance. The code is available at this https URL. 

---
# On the Security of Research Artifacts 

**Authors**: Nanda Rani, Christian Rossow  

**Link**: [PDF](https://arxiv.org/pdf/2605.06508)  

**Abstract**: Research artifacts are widely shared to support reproducibility, and artifact evaluation (AE) has become common at many leading conferences. However, AE mainly checks whether artifacts work as claimed and can be reproduced. It largely overlooks potential security risks. Since these artifacts are publicly released and reused, they may unintentionally create opportunities for misuse and raise concerns about safe and responsible sharing. We study 509 research artifacts from top-tier security venues and find that many contain insecure code patterns that may introduce potential attack vectors. We propose a taxonomy for context-aware security assessment to enable structured analysis of such risks. We perform static analysis and examine the resulting findings, filtering false positives and identifying real security risks. Our analysis shows that 41.60% of the prevalent findings may pose security concerns under practical usage. To support scalable analysis, we introduce SAFE (Security-Aware Framework for Artifact Evaluation), a first step toward an autonomous framework that analyzes tool-reported findings by considering code semantics, execution context, and practical exploitability. SAFE achieves 84.80% accuracy and 84.63% F1-score in distinguishing security and non-security risks. Overall, our results show that security is also important in AE for promoting safe and responsible research sharing. The source code is available at: this https URL 

---
# PACZero: PAC-Private Fine-Tuning of Language Models via Sign Quantization 

**Authors**: Murat Bilgehan Ertan, Xiaochen Zhu, Phuong Ha Nguyen, Marten van Dijk, Srinivas Devadas  

**Link**: [PDF](https://arxiv.org/pdf/2605.06505)  

**Abstract**: We introduce PACZero, a family of PAC-private zeroth-order mechanisms for fine-tuning large language models that delivers usable utility at $I(S^*; Y_{1:T})=0$. This privacy regime bounds the membership-inference attack (MIA) posterior success rate at the prior, an MIA-resistance level the DP framework matches only at $\varepsilon=0$ and infinite noise. All DP-ZO comparisons below are matched at the MIA posterior level. The key insight is that PAC Privacy charges mutual information only when the release depends on which candidate subset is the secret. Sign-quantizing subset-aggregated zeroth-order gradients creates frequent unanimity, steps at which every candidate subset agrees on the update direction; at these steps the released sign costs zero conditional mutual information. We propose two variants that span the privacy-utility trade-off: PACZero-MI (budgeted MI via exact calibration on the binary release) and PACZero-ZPL ($I=0$ via a uniform coin flip on disagreement steps). We evaluate on SST-2 and SQuAD with OPT-1.3B and OPT-6.7B in both LoRA and full-parameter tracks. On SST-2 OPT-1.3B full fine-tuning at $I=0$, PACZero-ZPL reaches ${88.99\pm0.91}$, within $2.1$pp of the non-private MeZO baseline ($91.1$ FT). No prior method produces usable utility in the high-privacy regime $\varepsilon<1$, and PACZero-ZPL obtains competitive SST-2 accuracy and nontrivial SQuAD F1 across OPT-1.3B and OPT-6.7B at $I=0$. 

---
# Operator-Guided Invariance Learning for Continuous Reinforcement Learning 

**Authors**: Zuyuan Zhang, Fei Xu Yu, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2605.06500)  

**Abstract**: Reinforcement learning (RL) with continuous time and state/action spaces is often data-intensive and brittle under nuisance variability and shift, motivating methods that exploit value-preserving structures to stabilize and improve learning. Most existing approaches focus on special cases, such as prescribed symmetries and exact equivariance, without addressing how to discover more general structures that require nonlinear operators to transform and map between continuous state/action systems with isomorphic value functions. We propose \textbf{VPSD-RL} (Value-Preserving Structure Discovery for Reinforcement Learning). It models continuous RL as a controlled diffusion with value-preserving mappings defined through Lie-group actions and associated pullback operators. We show that a value-preserving structure exists exactly when pulling back the value function and pushing forward actions commute with the controlled generator and reward functional. Further, approximate value-preserving structures with rigorous guarantees can be found when the Hamilton--Jacobi--Bellman mismatch is small. This framework discovers exact and approximate value-preserving structures by searching for the associated Lie group operators. VPSD-RL fits differentiable drift, diffusion, and reward models; learns infinitesimal generators via determining-equation residual minimization; exponentiates them with ODE flows to obtain finite transformations; and integrates them into continuous RL through transition augmentation and transformation-consistency regularization. We show that bounded generator/reward mismatch implies quantitative stability of the optimal value function along approximate orbits, with sensitivity governed by the effective horizon, and observe improved data efficiency and robustness on continuous-control benchmarks. 

---
# 3D MRI Image Pretraining via Controllable 2D Slice Navigation Task 

**Authors**: Yu Wang, Qingchao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06487)  

**Abstract**: Self-supervised pretraining has become the mainstream approach for learning MRI representations from unlabeled scans. However, most existing objectives still treat each scan primarily as static aggregations of slices, patches or volumes. We ask whether there exists an intrinsic form of self-supervision signal that is different from reconstructing the masked patches, through transforming the 3D volumes into controllable 2D rendered sequences: by rendering slices at continuous positions, orientations, and scales, a 3D volume can be converted into dense video-action sequences whose controls are the action trajectories. We study this formulation with an action-conditioned pretraining objective, where a tokenizer encodes slice observations and a latent dynamics model predicts the evolution of latent features. Across representative anatomical and spatial downstream tasks, the proposed pretraining is evaluated against standard static-volume baselines, tokenizer-only pretraining, and dynamics variants without aligned actions. These results suggest that controllable MRI slice navigation provides a useful complementary pretraining interface for learning anatomical and spatial representations from large unlabeled MRI collections. 

---
# Litespark Inference on Consumer CPUs: Custom SIMD Kernels for Ternary Neural Networks 

**Authors**: Nii Osae Osae Dade, Tony Morri, Moinul Hossain Rahat, Sayandip Pal  

**Link**: [PDF](https://arxiv.org/pdf/2605.06485)  

**Abstract**: Large language models (LLMs) have transformed artificial intelligence, but their computational requirements remain prohibitive for most users. Standard inference demands expensive datacenter GPUs or cloud API access, leaving over one billion personal computers underutilized for AI workloads. Ternary models offer a path forward: their weights are constrained to {-1, 0, +1}, theoretically eliminating the need for floating-point multiplication. However, existing frameworks fail to exploit this structure, treating ternary models as dense floating-point networks. We address this gap with custom SIMD kernels that replace matrix multiplication with simple addition and subtraction operations, targeting the integer dot product instructions available on modern CPUs. Our implementation, Litespark-Inference, is pip-installable and integrates directly with Hugging-Face, achieving 9.2x faster time-to-first-token, 52x higher throughput, and 14x memory reduction compared to standard PyTorch inference on Apple Silicon, with similar speedups on Intel and AMD processors. 

---
# Q-MMR: Off-Policy Evaluation via Recursive Reweighting and Moment Matching 

**Authors**: Xiang Li, Nan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06474)  

**Abstract**: We present a novel theoretical framework, Q-MMR, for off-policy evaluation in finite-horizon MDPs. Q-MMR learns a set of scalar weights, one for each data point, such that the reweighted rewards approximate the expected return under the target policy. The weights are learned inductively in a top-down manner via a moment matching objective against a value-function discriminator class. Notably, and perhaps surprisingly, a data-dependent finite-sample guarantee for general function approximation can be established under only the realizability of $Q^\pi$, with a dimension-free bound -- that is, the error does not depend on the statistical complexity of the function class. We also establish connections to several existing methods, such as importance sampling and linear FQE. Further theoretical analyses shed new light on the nature of coverage, a concept of fundamental importance to offline RL. 

---
# ORTHOBO: Orthogonal Bayesian Hyperparameter Optimization 

**Authors**: Maresa Schröder, Pascal Janetzky, Michael Klar, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2605.06454)  

**Abstract**: Bayesian optimization is widely used for hyperparameter optimization when model evaluations are expensive; however, noisy acquisition estimates can lead to unstable decisions. We identify acquisition estimation noise as a failure mode that was previously overlooked: even when the surrogate model and acquisition target are correctly specified, finite-sample Monte Carlo error can perturb acquisition values. This can, in turn, flip candidate rankings and lead to suboptimal BO decisions. As a remedy, we aim at variance reduction and propose an orthogonal acquisition estimator that subtracts an optimally weighted score-function control variate, which yields an acquisition residual orthogonal to posterior score directions and which thus reduces Monte Carlo variance. We further introduce OrthoBO: a Bayesian optimization framework that combines our orthogonal acquisition estimator with ensemble surrogates and an outer log transformation. We show theoretically that our estimator preserves the target, leads to variance reduction, and improves pairwise ranking stability. We further verify the theoretical properties of OrthoBO through numerical experiments where our framework reduces acquisition estimation variance, stabilizes candidate rankings, and achieves strong performance. We also demonstrate the downstream utility of OrthoBO in hyperparameter optimization for neural network training and fine-tuning. 

---
# Constraint Decay: The Fragility of LLM Agents in Backend Code Generation 

**Authors**: Francesco Dente, Dario Satriani, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2605.06445)  

**Abstract**: Large Language Model (LLM) agents demonstrate strong performance in autonomous code generation under loose specifications. However, production-grade software requires strict adherence to structural constraints, such as architectural patterns, databases, and object-relational mappings. Existing benchmarks often overlook these non-functional requirements, rewarding functionally correct but structurally arbitrary solutions. We present a systematic study evaluating how well agents handle structural constraints in multi-file backend generation. By fixing a unified API contract across 80 greenfield generation tasks and 20 feature-implementation tasks spanning eight web frameworks, we isolate the effect of structural complexity using a dual evaluation with end-to-end behavioral tests and static verifiers. Our findings reveal a phenomenon of constraint decay: as structural requirements accumulate, agent performance exhibits a substantial decline. Capable configurations lose 30 points on average in assertion pass rates from baseline to fully specified tasks, while some weaker configurations approach zero. Framework sensitivity analysis exposes significant performance disparities: agents succeed in minimal, explicit frameworks (e.g., Flask) but perform substantially worse on average in convention-heavy environments (e.g., FastAPI, Django). Finally, error analysis identifies data-layer defects (e.g., incorrect query composition and ORM runtime violations) as the leading root causes. This work highlights that jointly satisfying functional and structural requirements remains a key open challenge for coding agents. 

---
# COVID-19 Infodemic. Understanding content features in detecting fake news using a machine learning approach 

**Authors**: Balakrishnan Vimala, Hii Lee Zing, Laporte Eric  

**Link**: [PDF](https://arxiv.org/pdf/2605.06435)  

**Abstract**: The use of content features, particularly textual and linguistic for fake news detection is under-researched, despite empirical evidence showing the features could contribute to differentiating real and fake news. To this end, this study investigates a selection of content features such as word bigrams, part of speech distribution etc. to improve fake news detection. We performed a series of experiments on a new dataset gathered during the COVID-19 pandemic and using Decision Tree, K-Nearest Neighbor, Logistic Regression, Support Vector Machine and Random Forest. Random Forest yielded the best results, followed closely by Support Vector Machine, across all setups. In general, both the textual and linguistic features were found to improve fake news detection when used separately, however, combining them into a single model did not improve the detection significantly. Differences were also noted between the use of bigrams and part of speech tags. The study shows that textual and linguistic features can be used successfully in detecting fake news using the traditional machine learning approach as opposed to deep learning. 

---
# E = T*H/(O+B): A Dimensionless Control Parameter for Mixture-of-Experts Ecology 

**Authors**: Qingjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06415)  

**Abstract**: We introduce E = T*H/(O+B), a dimensionless control parameter that predicts whether Mixture-of-Experts (MoE) models will develop a healthy expert ecology or collapse into dead experts. E combines four hyperparameters -- routing temperature T, routing entropy weight H, oracle weight O, and balance weight B -- into a single quantity. Through 12 controlled experiments (8 vision, 4 language) totaling over 11,000 training epochs, we establish that E >= 0.5 alone is sufficient to guarantee zero dead experts, removing the necessity for handcrafted load-balancing auxiliary losses. We validate this cross-modally on CIFAR-10, CIFAR-100, TinyImageNet-200, WikiText-2, and WikiText-103. Six additional findings emerge: (1) dead experts can resuscitate -- triggered by balance loss driving router re-exploration; (2) ortho toxicity is dataset-dependent, not universal; (3) task complexity shifts the critical E threshold; (4) model overfitting is decoupled from expert ecological health; (5) three-tier MoE spontaneously collapses into a two-tier functional structure; (6) ecological structure is temperature-invariant across a 50x range. We propose that E serves as a unified diagnostic for MoE training, analogous to the Reynolds number in fluid dynamics. 

---
# WavCube: Unifying Speech Representation for Understanding and Generation via Semantic-Acoustic Joint Modeling 

**Authors**: Guanrou Yang, Tian Tan, Qian Chen, Zhikang Niu, Yakun Song, Ziyang Ma, Yushen Chen, Zeyu Xie, Tianrui Wang, Yifan Yang, Wenxi Chen, Qi Chen, Wenrui Liu, Shan Yang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06407)  

**Abstract**: Integrating speech understanding and generation is a pivotal step toward building unified speech models. However, the different representations required for these two tasks currently pose significant compatibility challenges. Typically, semantics-oriented features are learned from self-supervised learning (SSL), and acoustic-oriented features from reconstruction. Such fragmented representations hinder the realization of truly unified speech systems. We present WavCube, a compact continuous latent derived from an SSL speech encoder that simultaneously supports speech understanding, reconstruction, and generation. WavCube employs a two-stage training scheme. Stage 1 trains a semantic bottleneck to filter off-manifold redundancy that makes raw SSL features intractable for diffusion. Stage 2 injects fine-grained acoustic details via end-to-end reconstruction, while a semantic anchoring loss ensures the representation remains grounded within its original semantic manifold. Comprehensive experiments show that WavCube closely approaches WavLM performance on SUPERB despite an 8x dimensional compression, attains reconstruction quality on par with existing acoustic representations, delivers state-of-the-art zero-shot TTS performance with markedly faster training convergence, and excels in speech enhancement, separation, and voice conversion tasks on the SUPERB-SG benchmark. Systematic ablations reveal that WavCube's two-stage recipe resolves two intrinsic flaws of SSL features for generative modeling, paving the way for future unified speech systems. Codes and checkpoints are available at this https URL. 

---
# Consistent Geometric Deep Learning via Hilbert Bundles and Cellular Sheaves 

**Authors**: Kartik Tandon, Julian Gould, Tanishq Bhatia, Francesca Dominici, Alejandro Ribeiro, Claudio Battiloro  

**Link**: [PDF](https://arxiv.org/pdf/2605.06395)  

**Abstract**: Modern deep learning architectures increasingly contend with sophisticated signals that are natively infinite-dimensional, such as time series, probability distributions, or operators, and are defined over irregular domains. Yet, a unified learning theory for these settings has been lacking. To start addressing this gap, we introduce a novel convolutional learning framework for possibly infinite-dimensional signals supported on a manifold. Namely, we use the connection Laplacian associated with a Hilbert bundle as a convolutional operator, and we derive filters and neural networks, dubbed as \textit{HilbNets}. We make HilbNets and, more generally, the convolution operation, implementable via a two-stage sampling procedure. First, we show that sampling the manifold induces a Hilbert Cellular Sheaf, a generalized graph structure with Hilbert feature spaces and edge-wise coupling rules, and we prove that its sheaf Laplacian converges in probability to the underlying connection Laplacian as the sampling density increases. Notably, this result is a generalization to the infinite-dimensional bundle setting of the Belkin \& Niyogi \cite{BELKIN20081289} convergence result for the graph Laplacian to the manifold Laplacian, a theoretical cornerstone of geometric learning methods. Second, we discretize the signals and prove that the discretized (implementable) HilbNets converge to the underlying continuous architectures and are transferable across different samplings of the same bundle, providing consistency for learning. Finally, we validate our framework on synthetic and real-world tasks. Overall, our results broaden the scope of geometric learning as a whole by lifting classical Laplacian-based frameworks to settings where the signal at each point lives in its own Hilbert space. 

---
# Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation at the Token Level 

**Authors**: Nan Jia, Haojin Yang, Xing Ma, Jiesong Lian, Shuailiang Zhang, Weipeng Zhang, Ke Zeng, Xunliang Cai, Zequn Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.06387)  

**Abstract**: On-policy distillation (OPD) trains a student on its own trajectories with token-level teacher feedback and often outperforms off-policy distillation and standard reinforcement learning. However, we find that its standard advantage weighted policy gradient suffers from three structural weaknesses, including high variance updates, vanishing gradients in zero-advantage regions, and exploration bottlenecks when corrective signals are this http URL therefore propose Asymmetric On-Policy Distillation (AOPD), which replaces ineffective negative reinforcement with localized divergence minimization in non-positive advantage regions while preserving positive reinforcement learning. Experiments on mathematical reasoning benchmarks show that AOPD consistently outperforms standard OPD, with average gains of 4.09 / 8.34 under strong / weak initialization, respectively. AOPD also maintains higher policy entropy during training and better capability retention during sequential tool-use adaptation. 

---
# MinMax Recurrent Neural Cascades 

**Authors**: Alessandro Ronca  

**Link**: [PDF](https://arxiv.org/pdf/2605.06384)  

**Abstract**: We show that the MinMax algebra provides a form of recurrence that is expressively powerful, efficiently implementable, and most importantly it is not affected by vanishing or exploding gradient. We call MinMax Recurrent Neural Cascades (RNCs) the models obtained by cascading several layers of neurons that employ such recurrence. We show that MinMax RNCs enjoy many favourable theoretical properties. First, their formal expressivity includes all regular languages, arguably the maximal expressivity for a finite-memory system. Second, they can be evaluated in parallel with a runtime that is logarithmic in the input length given enough processors; and they can also be evaluated sequentially. Third, their state and activations are bounded uniformly for all input lengths. Fourth, at almost all points, their loss gradient exists and it is bounded. Fifth, they do not exhibit a vanishing state gradient: the gradient of a state w.r.t. a past state can have constant value one regardless of the time distance between the two states. Finally, we find empirical evidence that the favourable theoretical properties of MinMax RNCs are matched by their practical capabilities: they are able to perfectly solve a number of synthetic tasks, showing superior performance compared to the considered state-of-the-art recurrent neural networks; also, we train a MinMax RNC of 127M parameters on next-token prediction, and the obtained model shows competitive performance for its size, providing evidence of the potential of MinMax RNCs on real-world tasks. 

---
# Continuous-Time Distribution Matching for Few-Step Diffusion Distillation 

**Authors**: Tao Liu, Hao Yan, Mengting Chen, Taihang Hu, Zhengrong Yue, Zihao Pan, Jinsong Lan, Xiaoyong Zhu, Ming-Ming Cheng, Bo Zheng, Yaxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06376)  

**Abstract**: Step distillation has become a leading technique for accelerating diffusion models, among which Distribution Matching Distillation (DMD) and Consistency Distillation are two representative paradigms. While consistency methods enforce self-consistency along the full PF-ODE trajectory to steer it toward the clean data manifold, vanilla DMD relies on sparse supervision at a few predefined discrete timesteps. This restricted discrete-time formulation and mode-seeking nature of the reverse KL divergence tends to exhibit visual artifacts and over-smoothed outputs, often necessitating complex auxiliary modules -- such as GANs or reward models -- to restore visual fidelity. In this work, we introduce Continuous-Time Distribution Matching (CDM), migrating the DMD framework from discrete anchoring to continuous optimization for the first time. CDM achieves this through two continuous-time designs. First, we replace the fixed discrete schedule with a dynamic continuous schedule of random length, so that distribution matching is enforced at arbitrary points along sampling trajectories rather than only at a few fixed anchors. Second, we propose a continuous-time alignment objective that performs active off-trajectory matching on latents extrapolated via the student's velocity field, improving generalization and preserving fine visual details. Extensive experiments on different architectures, including SD3-Medium and Longcat-Image, demonstrate that CDM provides highly competitive visual fidelity for few-step image generation without relying on complex auxiliary objectives. Code is available at this https URL. 

---
# eXplaining to Learn (eX2L): Regularization Using Contrastive Visual Explanation Pairs for Distribution Shifts 

**Authors**: Paulo Mario P. Medina, Jose Marie Antonio Miñoza, Sebastian C. Ibañez  

**Link**: [PDF](https://arxiv.org/pdf/2605.06368)  

**Abstract**: Despite extensive research into mitigating distribution shifts, many existing algorithms yield inconsistent performance, often failing to outperform baseline Empirical Risk Minimization (ERM) across diverse scenarios. Furthermore, high algorithmic complexity frequently limits interpretability and offers only an indirect means of addressing spurious correlations. We propose eXplaining to Learn (eX2L): an interpretable, explanation-based framework that decorrelates confounding features from a classifier's latent representations during training. eX2L achieves this by penalizing the similarity between Grad-CAM activation maps generated by a primary label classifier and those from a concurrently trained confounder classifier. On the rigorous Spawrious Many-to-Many Hard Challenge benchmark, eX2L achieves an average accuracy (AA) of 82.24% +/- 3.87% and a worst-group accuracy (WGA) of 66.31% +/- 8.73%, outperforming the current state-of-the-art (SOTA) by 5.49% and 10.90%, respectively. Beyond its competitive performance, eX2L demonstrates that functional domain invariance can be achieved by explicitly decoupling label and nuisance attributes at the group level. 

---
# Flow Matching with Arbitrary Auxiliary Paths 

**Authors**: Xin Peng, Ang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.06364)  

**Abstract**: We introduce a new generative modeling framework, \textbf{Flow Matching with Arbitrary Auxiliary Paths (AuxPath-FM)}, which generalizes conditional flow matching by incorporating an auxiliary variable drawn from an arbitrary distribution into the probability path. Unlike prior methods that restrict auxiliary components to Gaussian noise, AuxPath-FM allows the variable $\eta$ to follow any distribution, producing trajectories of the form $X_t = a(t)X_1 + b(t)X_0 + c(t)\eta$. We theoretically demonstrate that this construction preserves the continuity equation and maintains a training objective consistent with the marginal formulation. This flexibility enables the design of diverse probability paths using various priors, including Gaussian, Uniform, Laplace, and discrete Rademacher distributions, each offering unique geometric properties for generative flows. Furthermore, our framework allows for specialized tasks such as label-guided generation by encoding structured semantic information into the auxiliary distribution. Overall, AuxPath-FM provides a principled and general foundation for probability path design, offering both theoretical generality and practical flexibility for diverse generative modeling tasks. 

---
# Memory Efficient Full-gradient Attacks (MEFA) Framework for Adversarial Defense Evaluations 

**Authors**: Yuan Du, Mitchel Hill, HanQin Cai  

**Link**: [PDF](https://arxiv.org/pdf/2605.06357)  

**Abstract**: This work studies the robust evaluation of iterative stochastic purification defenses under white-box adversarial attacks. Our key technical insight is that gradient checkpointing makes exact end-to-end gradient computation through long purification trajectories practical by trading additional recomputation for substantially lower memory usage. This enables full-gradient adaptive attacks against diffusion- and Langevin-based purification defenses, where prior evaluations often resort to approximate backpropagation due to memory constraints. These approximations can weaken the attack signal and risk overestimating robustness. In parallel, stochasticity in iterative purification is frequently under-controlled, even though different purification trajectories can substantially change reported robustness metrics. Building on this insight, we introduce a memory-efficient full-gradient evaluation framework for stochastic purification defenses. The framework combines checkpointed backpropagation with evaluation protocols that control stochastic variability, thereby reducing memory bottlenecks while preserving exact gradients. We evaluate diffusion-based purification and Langevin sampling with Energy-Based Models (EBMs), demonstrating that full-gradient attacks uncover vulnerabilities missed by approximate-gradient evaluations. Our framework yields stronger state-of-the-art $\ell_{\infty}$ and $\ell_{2}$ white-box attacks and further supports probing out-of-distribution robustness. Overall, our results show that exact-gradient evaluation is essential for reliable benchmarking of iterative stochastic defenses. 

---
# Topological Signatures of Grokking 

**Authors**: Yifan Tang, Qiquan Wang, Inés García-Redondo, Anthea Monod  

**Link**: [PDF](https://arxiv.org/pdf/2605.06352)  

**Abstract**: We study the grokking phenomenon through the lens of topology. Using persistent homology on point clouds derived from the embedding matrices of a range of models trained on modular arithmetic with varying primes, we identify a clear and consistent topological signature of grokking: a sharp increase in both the maximum and total persistence of first homology ($H_1$). Persistence diagrams reveal the emergence of a dominant long-lived topological feature together with increasingly structured secondary features, reflecting the underlying cyclic structure of the task. Compared to existing spectral and geometric diagnostics -- specifically, Fourier analysis and local intrinsic dimension -- persistent homology provides a unified geometric and topological characterization of representation learning, capturing both local and global multi-scale structure. Ablations across data regimes and control settings show that these topological transitions are tied to generalization rather than memorization. Our results suggest that persistent homology offers a principled and interpretable framework for analyzing how neural networks internalize latent structure during training. 

---
# Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades 

**Authors**: Dylan Bouchard  

**Link**: [PDF](https://arxiv.org/pdf/2605.06350)  

**Abstract**: Model cascades, in which a cheap LLM defers to an expensive one on low-confidence queries, are widely used to navigate the cost-quality tradeoff at deployment. Existing approaches largely treat the deferral threshold as an empirical hyperparameter, with limited guidance on the geometry of the resulting cost-quality frontier over a model pool. We develop a decision-theoretic framework grounded in constrained optimization and duality. For a two-model cascade, we establish piecewise concavity of the cost-quality frontier on decreasing-benefit regions of the confidence support, with reciprocal shadow prices linking the budget- and quality-constrained formulations. Given a pool of $k$ models, we characterize the frontier achievable by deterministic two-model threshold cascades as the pointwise envelope over $\binom{k}{2}$ pairwise cascades, with switching points where the optimal pair changes. For $k$-model cascades, we derive first-order conditions in which a single shadow price equalizes marginal quality-per-cost across stage boundaries. We validate the framework on five benchmarks (MATH, MMLU, TriviaQA, SimpleQA, LiveCodeBench) across eight models from five providers. Within the deterministic threshold-cascade class, full fixed chains underperform the pairwise envelope, and optimized subsequence cascades do not deliver practically meaningful held-out gains over it. A lightweight pre-generation router exceeds the best cascade policy on four of five datasets, mainly because it avoids the cheap model's generation cost on queries sent directly to a larger model rather than because of a stronger routing signal. These results suggest that cascade performance is limited primarily by structural cost, since cascades pay the cheap model before any escalation decision, rather than by a shortage of intermediate stages. 

---
# Human-AI Co-Evolution and Epistemic Collapse: A Dynamical Systems Perspective 

**Authors**: Xuening Wu, Yanlan Kang, Qianya Xu, Kexuan Xie, Jiaqi Mi, Honggang Wang, Yubin Liu, Zeping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06347)  

**Abstract**: Large language models (LLMs) are reshaping how knowledge is produced, with increasing reliance on AI systems for generation, summarization, and reasoning. While prior work has studied cognitive offloading in humans and model collapse in recursive training, these effects are typically considered in isolation. We propose a unified perspective: humans and language models form a coupled dynamical system linked by a feedback loop of usage, generation, and retraining. We introduce a minimal model with three variables -- human cognition, data quality, and model capability -- and show that this feedback can give rise to distinct dynamical regimes. Our analysis identifies three regimes: co-evolutionary enhancement, fragile equilibrium, and degenerative convergence. Through a simple simulation, we demonstrate that increasing reliance on AI can induce a transition toward a low-diversity, suboptimal equilibrium. From an information-theoretic perspective, this transition corresponds to an emergent information bottleneck in the human-AI loop, where entropy reduction reflects loss of diversity and support under closed-loop feedback rather than beneficial compression. These results suggest that the trajectory of AI systems is shaped not only by model design, but by the dynamics of human-AI co-evolution. 

---
# CoupleEvo: Evolving Heuristics for Coupled Optimization Problems Using Large Language Models 

**Authors**: Thomas Bömer, Bastian Amberg, Max Disselnmeyer, Anne Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2605.06341)  

**Abstract**: Many real-world optimization problems consist of multiple tightly coupled subproblems whose solutions must be coordinated to achieve high overall performance. However, existing large language model driven automated heuristic design approaches are limited to single-problem settings. In this paper, we propose CoupleEvo. CoupleEvo proposes three evolutionary coordination strategies to evolve heuristics for coupled optimization problems: the sequential strategy evolves heuristics for one subproblem after the other; the iterative strategy alternates the evolution of heuristics for different subproblems over successive generations; and the integrated strategy evolves heuristics for all problems simultaneously. The approach is evaluated on two representative coupled optimization problems. Experimental results show that decomposition-based strategies (sequential and iterative) provide more stable convergence and higher solution quality, while the integrated evolution strategy suffers from increased search complexity and variability. These findings highlight the importance of coordinating evolutionary search across interdependent subproblems and demonstrate the potential of LLM-driven heuristic design for complex coupled optimization problems. The code is available: this https URL. 

---
# TinyBayes: Closed-Form Bayesian Inference via Jacobi Prior for Real-Time Image Classification on Edge Devices 

**Authors**: Shouvik Sardar, Sourish Das  

**Link**: [PDF](https://arxiv.org/pdf/2605.06333)  

**Abstract**: Cocoa (Theobroma cacao) is a critical cash crop for millions of smallholder farmers in West Africa, where Cocoa Swollen Shoot Virus Disease (CSSVD) and anthracnose cause devastating yield losses. Automated disease detection from leaf images is essential for early intervention, yet deploying such systems in resource-constrained settings demands models that are small, fast, and require no internet connectivity. Existing edge-deployable plant disease systems rely on end-to-end deep learning without uncertainty quantification, while Bayesian methods for edge devices focus on hardware-level inference architectures rather than agricultural applications. We bridge this gap with TinyBayes, the first framework to combine a closed-form Bayesian classifier with a mobile-grade computer vision pipeline for crop disease detection. Our pipeline uses YOLOv8-Nano (5.9 MB) for lesion localisation, MobileNetV3-Small (3.5 MB) for feature extraction, and the Jacobi prior; a Bayesian method that provides a closed form non-iterative estimators via projection, for the classification. The Jacobi-DMR (Distributed Multinomial Regression) classifier adds only 13.5 KB to the pipeline, bringing the total model size within 9.5 MB, while achieving 78.7% accuracy on the Amini Cocoa Contamination Challenge dataset and enabling end-to-end CPU inference under 150 ms per image. We benchmark against seven classifiers including Random Forest, SVM, Ridge, Lasso, Elastic Net, XGBoost, and Jacobi-GP, and demonstrate that the Jacobi-DMR offers the best trade-off between accuracy, model size, and inference speed for edge deployment. We have proved the asymptotic equivalence and consistency, asymptotic normality and the bias correction of Jacobi-DMR. All data and codes are available here: this https URL 

---
# Fine-Tuning Small Language Models for Solution-Oriented Windows Event Log Analysis 

**Authors**: Siraaj Akhtar, Saad Khan, Simon Parkinson  

**Link**: [PDF](https://arxiv.org/pdf/2605.06330)  

**Abstract**: Large language models (LLMs) have shown promise for event log analysis, but their high computational requirements, reliance on cloud infrastructure, and security concerns limit practical deployment. In addition, most existing approaches focus only on the identification of the problem and do not provide actionable remediation. Small language models (SLMs) present a light-weight alternative that can be fine-tuned for a specific purpose and hosted locally. This paper investigates whether SLMs, when fine-tuned for a specific task, can serve as a practical alternative for event log analysis while also generating solutions. We first create a large-scale synthetic Windows event log dataset that contains remediation actions using a high-performing LLM. We then fine-tune multiple SLMs and LLMs using the LoRA parameter-efficient fine-tuning technique and evaluate their performance by comparing with expert assessment. The results show that the dataset accurately reflects real-world scenarios and that fine-tuned SLMs consistently outperform LLMs in identifying issues and providing relevant remediation, while requiring fewer computational resources. 

---
# Measuring Evaluation-Context Divergence in Open-Weight LLMs: A Paired-Prompt Protocol with Pilot Evidence of Alignment-Pipeline-Specific Heterogeneity 

**Authors**: Florian A. D. Burnat, Brittany I. Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2605.06327)  

**Abstract**: Safety benchmarks are routinely treated as evidence about how a language model will behave once deployed, but this inference is fragile if behavior depends on whether a prompt looks like an evaluation. We define evaluation-context divergence as an observable within-item change in behavior induced by framing a fixed task as an evaluation, a live deployment interaction, or a neutral request, and present a paired-prompt protocol that measures it in open-weight LLMs while controlling for paraphrase variation, benchmark familiarity, and judge framing-sensitivity.
Across five instruction-tuned checkpoints from four open-weight families plus a matched OLMo-3 base/instruct ablation ($20$ paired items, $840$ generations per checkpoint), we find striking heterogeneity. OLMo-3-Instruct alone is eval-cautious -- evaluation framing raises refusal vs. neutral by $11.8$pp ($p=0.007$) and reduces harmful compliance vs. deployment by $3.6$pp ($p=0.024$, $0/20$ items inverted) -- while Mistral-Small-3.2, Phi-3.5-mini, and Llama-3.1-8B are deployment-cautious}, with marginal eval-vs-deployment refusal effects of $-9$ to $-20$pp. The matched OLMo-3 base also exhibits the deployment-cautious pattern, identifying alignment as the inversion stage; within Llama-3.1, the $70$B model preserves direction with attenuated magnitude, ruling out a simple ``small-model effect that reverses at scale.'' One caveat: the cross-family heterogeneity is judge-dependent. Re-judging with a different-family safety classifier (Llama-Guard-3-8B) preserves the within-OLMo eval-cautious direction but flattens the cross-family contrast, indicating that the two judges operationalize distinct constructs. 

---
# Improving the Efficiency of Language Agent Teams with Adaptive Task Graphs 

**Authors**: Elizabeth Mieczkowski, Alexander Ku, Tiwalayo Eisape, Dilip Arumugam, John Matters, Katherine M. Collins, Ilia Sucholutsky, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2605.06320)  

**Abstract**: Large language models (LLMs) are increasingly deployed in teams, yet existing coordination approaches often occupy two extremes. Highly structured methods rely on fixed roles, pipelines, or task decompositions assigned a priori. In contrast, fully unstructured teams enable adaptability and exploration but suffer from inefficiencies such as error propagation, inter-agent conflicts, and wasted resources (measured in time, tokens, or file operations). We introduce Language Agent Teams for Task Evolution (LATTE), a framework for coordinating LLM teams inspired by distributed systems, where processors must operate under partial observability and communication constraints. In LATTE, a team of agents collaboratively construct and maintain a shared, evolving coordination graph which encodes sub-task dependencies, individual agent assignment, and the current state of sub-task progress. This protocol maintains consistency while empowering agents to dynamically allocate work, adapt coordination, and discover new tasks. Across multiple collaborative tasks and a variety of base models, we demonstrate how LATTE reduces token usage, wall-clock time, communication, and coordination failures (e.g. file conflicts and redundant outputs) while matching or exceeding the accuracy of standard designs including MetaGPT, decentralized teams, top-down Leader-Worker hierarchies, and static decompositions. 

---
# NavOne: One-Step Global Planning for Vision-Language Navigation on Top-Down Maps 

**Authors**: Dijia Zhan, Jinyi Li, Chenxi Zheng, Shaoyu Huang, Yong Li, Jie Tang, Xuemiao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06317)  

**Abstract**: Existing Vision-Language Navigation (VLN) methods typically adopt an egocentric, step-by-step paradigm, which struggles with error accumulation and limits efficiency. While recent approaches attempt to leverage pre-built environment maps, they often rely on incrementally updating memory graphs or scoring discrete path proposals, which restricts continuous spatial reasoning and creates discrete bottlenecks. We propose Top-Down VLN (TD-VLN), reformulating navigation as a one-step global path planning problem on pre-built top-down maps, supported by our newly constructed R2R-TopDown dataset. To solve this, we introduce NavOne, a unified framework that directly predicts dense path probabilities over multi-modal maps in a single end-to-end forward pass. NavOne features a Top-Down Map Fuser for joint multi-modal map representation, and extends Attention Residuals for spatial-aware depth mixing. Extensive experiments on R2R-TopDown show that NavOne achieves state-of-the-art performance among map-based VLN methods, with a planning-stage speedup of 8x over existing map-based baselines and 80x over egocentric methods, enabling highly efficient global navigation. 

---
# Pro-KLShampoo: Projected KL-Shampoo with Whitening Recovered by Orthogonalization 

**Authors**: Ruotong Sun, Ermin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.06316)  

**Abstract**: Optimizers that exploit the matrix structure of gradients are central to modern LLM pre-training, with two distinct frontiers: explicit Kronecker-factored preconditioning -- most recently KL-Shampoo, which estimates the preconditioner via KL divergence minimization -- and orthogonalization of the gradient momentum, exemplified by Muon and analyzed as steepest descent under the spectral norm. The two routes are typically developed in isolation. We make a structural observation about KL-Shampoo's Kronecker preconditioners: their eigenvalue spectra exhibit a \emph{spike-and-flat} shape -- a few dominant eigenvalues followed by an approximately uniform tail -- across layers and training stages, holding exactly under a rank-$\rho$ signal-plus-noise gradient model. We exploit this structure by restricting one of KL-Shampoo's Kronecker factors to a parametric family aligned with the spike-and-flat shape: full spectral structure on a tracked $r$-dimensional subspace, single shared eigenvalue across the remaining $n-r$ directions. On these directions, we apply orthogonalization. An identity shows that this orthogonalization recovers the algebraic form of full KL-Shampoo's preconditioner. On four pre-training scales (GPT-2 124M / 350M, LLaMA 134M / 450M), Pro-KLShampoo consistently outperforms KL-Shampoo at every subspace rank we test in validation loss, peak per-GPU memory, and wallclock time to reach each loss level. 

---
# Render, Don't Decode: Weight-Space World Models with Latent Structural Disentanglement 

**Authors**: Roussel Desmond Nzoyem, Mauro Comi  

**Link**: [PDF](https://arxiv.org/pdf/2605.06298)  

**Abstract**: Training world models on vast quantities of unlabelled videos is a critical step toward fully autonomous intelligence. However, the prevailing paradigm of encoding raw pixels into opaque latent spaces and relying on heavy decoders for reconstruction leaves these models computationally expensive and uninterpretable. We address this problem by introducing NOVA, a world modelling framework that represents the system state as the weights and biases of an auxiliary coordinate-based implicit neural representation (INR). This structured representation is analytically rendered, which eliminates the decoder bottleneck while conferring compactness, portability, and zero-shot super-resolution. Furthermore, like most latent action models, NOVA can be distilled into a context-dependent video generator via an action-matching objective. Surprisingly, without resorting to auxiliary losses or adversarial objectives, NOVA can disentangle structural scene components such as background, foreground, and inter-frame motion, enabling users to edit either content or dynamics without compromising the other. We validate our framework on several challenging datasets, achieving strong controllable forecasting while operating on a single consumer GPU at $\sim$40M parameters. Ultimately, structured representations like INRs not only enhance our understanding of latent dynamics but also pave the way for immersive and customisable virtual experiences. 

---
# Attributions All the Way Down? The Metagame of Interpretability 

**Authors**: Hubert Baniecki, Przemyslaw Biecek, Fabian Fumagalli  

**Link**: [PDF](https://arxiv.org/pdf/2605.06295)  

**Abstract**: We introduce the metagame, a conceptual framework for quantifying second-order interaction effects of model explanations. For any first-order attribution $\phi(f)$ explaining a model $f$, we measure the directional influence of feature $j$ on the attribution of feature $i$, denoted as meta-attribution $\varphi_{j \to i}(f)$, by treating the attribution method itself as a cooperative game and computing its Shapley value. Theoretically, we prove that attributions hierarchically decompose into meta-attributions, and establish these as directional extensions of existing interaction indices. Empirically, we demonstrate that the metagame delivers insights across diverse interpretability applications: (i) quantifying token interactions in instruction-tuned language models, (ii) explaining cross-modal similarity in vision-language encoders, and (iii) interpreting text-to-image concepts in multimodal diffusion transformers. 

---
# Log-Likelihood, Simpson's Paradox, and the Detection of Machine-Generated Text 

**Authors**: Tom Kempton, Viktor Drobnyi, Maeve Madigan, Stuart Burrell  

**Link**: [PDF](https://arxiv.org/pdf/2605.06294)  

**Abstract**: The ability to reliably distinguish human-written text from that generated by large language models is of profound societal importance. The dominant approach to this problem exploits the likelihood hypothesis: that machine-generated text should appear more probable to a detector language model than human-written text. However, we demonstrate that the token-level signal distinguishing human and machine text is non-uniform across the hidden space of the detector model, and naively averaging likelihood-based token scores across regions with fundamentally different statistical structure, as most detectors do, causes a form of Simpson's paradox: a strong local signal is destroyed by inappropriate aggregation. To correct for this, we introduce a learned local calibration step grounded in Bayesian decision theory. Rather than aggregating raw token scores, we first learn lightweight predictors of the score distributions conditioned on position in hidden space, and aggregate calibrated log-likelihood ratios instead. This single intervention dramatically and consistently improves detection performance across all baseline detectors and all datasets we consider. For example, our calibrated variant of Fast-DetectGPT improves AUROC from $0.63$ to $0.85$ on GPT-5.4 text, and a locally-calibrated DMAP detector we introduce achieves state-of-the-art performance across the board. That said, our central contribution is not a new detector, but a precise diagnosis of a significant cause of under-performance of existing detectors and a principled, modular remedy compatible with any token-averaging pipeline. This will serve as a foundation for the community to build upon, with natural avenues including richer distributional models, improved calibration strategies, and principled ensembling with hidden-space geometry signals via the full Bayes-optimal decision rule. 

---
# Multimodal Deep Generative Model for Semi-Supervised Learning under Class Imbalance 

**Authors**: Heegeon Yoon, Heeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.06289)  

**Abstract**: When modeling class-imbalanced data, it is crucial to address the imbalance, as models trained on such data tend to be biased towards the majority classes. This problem is amplified under partial supervision, where pseudo-labels for unlabeled data are predicted based on imbalanced labeled data, propagating the bias. While recent semi-supervised models address class imbalance, they typically assume single-modal input data. However, with the growing availability of multimodal data, it is essential to leverage complementary modalities. In this article, we propose a multimodal deep generative model for semi-supervised learning under class imbalance. Our approach uses separate encoders for each modality, sharing latent variables across modalities, and simplifies joint posterior computation with a product-of-experts method. To further address class imbalance, we replace typical Gaussian distributions with Student's t-distributions for the prior, encoder, and decoder, better capturing the heavy-tailed latent distributions in imbalanced data. We derive a new objective function for training the proposed model on both labeled and unlabeled data using $\gamma$-power divergence. Empirical results on benchmark and real-world datasets demonstrate that our model outperforms baseline methods in generalization, achieving superior classification performance for partially labeled multimodal data with imbalanced class distributions. 

---
# A Topological Sorting Criterion for Random Causal Directed Acyclic Graphs 

**Authors**: Alexander G. Reisach, Antoine Chambaz, Gilles Blanchard, Sebastian Weichwald  

**Link**: [PDF](https://arxiv.org/pdf/2605.06288)  

**Abstract**: Random directed acyclic graphs (DAGs) based on imposing an order on Erdős-Rényi and scale free random graphs are widely used for evaluating causal discovery algorithms. We show that in such DAGs, the set of nodes reachable via open paths, termed relatives, increases monotonically along the causal order. We assess the prevalence of this pattern numerically, and demonstrate that it can be exploited for causal order recovery via sorting by the estimated number of relatives. We note that many simulations in the literature feature settings where this yields an excellent proxy for the causal order, and show that a strict increase of relatives along the causal order leads to a singular Markov equivalence class. We propose sampling time-series DAGs as a possible alternative and discuss implications for causal discovery algorithms and their evaluation on synthetic data. 

---
# Correct Code, Vulnerable Dependencies: A Large Scale Measurement Study of LLM-Specified Library Versions 

**Authors**: Chengjie Wang, Jingzheng Wu, Xiang Ling, Tianyue Luo, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.06279)  

**Abstract**: Large language models (LLMs) are now largely involved in software development workflows, and the code they generate routinely includes third-party library (TPL) imports annotated with specific version identifiers. These version choices can carry security and compatibility risks, yet they have not been systematically studied.
We present the first large-scale measurement study of version-level risk in LLM-generated Python code, evaluating 10 LLMs on PinTrace, a curated benchmark of 1,000 Stack Overflow programming tasks. LLMs tend to specify version identifiers when directly prompted at 26.83%-95.18%, while down to 6.45%-59.19% in creating a manifest file directly. Among the specified versions, 36.70%-55.70% of tasks contain at least one known CVE, and 62.75%-74.51% of them carry Critical or High severity ratings. In 72.27%-91.37% of cases, the associated CVEs were publicly disclosed before the model's knowledge cutoff. The statistics show all models converge on the same small set of risky release versions, indicating a systemic bias rather than isolated model error. Static compatibility rates range from 19.70% to 63.20%, with installation failure as the dominant cause. The dynamic test cases confirm the pattern by 6.49%-48.62% pass rates. Further experiments confirm that these failures are attributable to version selection rather than code quality, and that externally anchored version constraints substantially reduce both vulnerability exposure and compatibility failures. Our findings reveal LLM version selection as a first-class, previously overlooked risk surface in LLM-based development. We disclosed these findings to the community of the evaluated models, and several confirmed the issue. All the code and dataset have been released for open science at this https URL. 

---
# Linear Semantic Segmentation for Low-Resource Spoken Dialects 

**Authors**: Kirill Chirkunov, Younes Samih, Abed Alhakim Freihat, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2605.06276)  

**Abstract**: Semantic segmentation is a core component of discourse analysis, yet existing models are primarily developed and evaluated on high-resource written text, limiting their effectiveness on low-resource spoken varieties. In particular, dialectal Arabic exhibits informal syntax, code-switching, and weakly marked discourse structure that challenge standard segmentation approaches. In this paper, we introduce a new multi-genre benchmark (more than 1000 samples) for semantic segmentation in conversational Arabic, focusing on dialectal discourse. The benchmark covers transcribed casual telephone conversations, code-switched podcasts, broadcast news, and expressive dialogue from novels, and was annotated and validated by native Arabic annotators. Using this benchmark, we show that segmentation models performing well on MSA news genres degrade on dialectal transcribed speech. We further propose a segmentation model that targets local semantic coherence and robustness to discourse discontinuities, consistently outperforming strong baselines on dialectal non-news genres. The benchmark and approach generalize to other low-resource spoken languages. 

---
# Inference-Time Refinement Closes the Synthetic-Real Gap in Tabular Diffusion 

**Authors**: Eugenio Lomurno, Filippo Balzarini, Francesco Benelle, Francesca Pia Panaccione, Matteo Matteucci  

**Link**: [PDF](https://arxiv.org/pdf/2605.06261)  

**Abstract**: Diffusion-based generators set the current state of the art for synthetic tabular data. These methods approach but rarely exceed real-data utility, and closing this synthetic-real gap has so far been pursued exclusively at training time, via architectural advances, scaling, and retraining of monolithic generators. The inference-time alternative, i.e., refining the outputs of a pre-trained backbone with parameters left untouched, has remained largely unexplored for tabular synthesis. We introduce TARDIS (Tabular generation through Refinement, Distillation, and Inference-time Sampling), an inference-time refinement framework that operates on a frozen pre-trained backbone, configured per dataset by a Tree-structured Parzen Estimator search over score-level guidance during reverse diffusion, with each trial's objective set by an inner grid search over post-hoc sample selectors and an optional soft-label distillation step. The search space encodes a single mathematical pattern we name Bidirectional Chamfer Refinement (BCR): the symmetric Chamfer functional between synthetic and real samples is minimized both continuously, via a score-level gradient, and discretely, via batch-ranking post-generation. The per-dataset search recovers BCR-aligned configurations on most datasets, evidence for BCR as the dominant refinement pattern. Across 15 binary, multiclass, and regression benchmarks TARDIS achieves a median +8.6% downstream-task improvement over models trained on real data (95% CI [+3.3, +16.4], Wilcoxon p=0.016, 11/15 strict wins) and improves over the TabDiff backbone on all 15 datasets (mean +12.9%, p<10^-4), matching the backbone on manifold fidelity, diversity, and sample-level privacy. Inference-time refinement of a pre-trained tabular diffusion backbone reaches and exceeds real-data utility in 1 to 80 minutes on a single consumer-grade GPU. 

---
# The Weight Gram Matrix Captures Sequential Feature Linearization in Deep Networks 

**Authors**: Taehun Cha, Daniel Beaglehole, Adityanarayanan Radhakrishnan, Donghun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.06258)  

**Abstract**: Understanding how deep neural networks learn representations remains a central challenge in machine learning theory. In this work, we propose a feature-centric framework for analyzing neural network training by relating weight updates to feature evolution. We introduce a simple identity, the Feature Learning Equation, which identifies the weight Gram matrix as the key object capturing feature dynamics. This enables us to interpret gradient descent as implicitly inducing a hypothetical evolution of features, whose covariance structure - termed the Virtual Covariance - characterizes how representations evolve during training. Building on this perspective, we introduce Target Linearity, a measure quantifying the linear alignment between features and targets. By analyzing the training and layer-wise dynamics, we show that deep networks learn to sequentially transform representations toward target-linear structure. This linearization perspective provides a unified interpretation of several empirical phenomena, including Neural Collapse and linear interpolation in generative models. 

---
# Cumulative-Goodness Free-Riding in Forward-Forward Networks: Real, Repairable, but Not Accuracy-Dominant 

**Authors**: Amirhossein Yousefiramandi  

**Link**: [PDF](https://arxiv.org/pdf/2605.06240)  

**Abstract**: Forward-Forward (FF) training allows each layer to learn from a local goodness criterion. In cumulative-goodness variants, however, later layers can inherit a task that earlier layers have already partially separated. We formalize this phenomenon as layer free-riding: under the softplus FF criterion, the class-discrimination gradient reaching block $d$ decays exponentially with the positive margin accumulated by preceding blocks. We then study three local remedies -- per-block, hardness-gated, and depth-scaled -- that recover current-layer separation measures without relying on backpropagated gradients. On CIFAR-10 and CIFAR-100, these remedies dramatically improve layer-separation statistics, with $4\times$--$45\times$ gains in deeper layers, while changing accuracy by less than one percentage point for non-degenerate training procedures. Tiny ImageNet provides a tougher cross-dataset check for our selected block-wise configuration and reveals the same qualitative gap between layer-health diagnostics and final accuracy. Calibration experiments further show that architecture and augmentation choices have a larger effect on final accuracy than the training-rule modifications studied here. Cumulative free-riding is therefore a real and repairable optimization pathology. Nonetheless, for the FF training rules, architectures, and datasets we study, it is not the dominant factor limiting achievable accuracy. 

---
# Band Together: Untargeted Adversarial Training with Multimodal Coordination against Evasion-based Promotion Attacks 

**Authors**: Guanmeng Xian, Ning Yang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06238)  

**Abstract**: Multimodal recommender systems exploit visual and textual signals to alleviate data sparsity, but this also makes them more vulnerable to evasion-based promotion attacks. Existing defenses are largely limited to single-modal settings and mainly focus on poisoning-based threats, leaving evasion-based threats underexplored. In this work, we first identify a cross-modal gradient mismatch under the multi-user promotion setting, where visual and textual perturbations are optimized in inconsistent directions due to the dominance of distinct user groups. This phenomenon dilutes the attack effectiveness and leads robust training to underestimate worst-case risks. To address this issue, we propose Untargeted Adversarial Training with Multimodal Coordination (UAT-MC). UAT-MC tackles the challenge of unknown targeted items in evasion-based attacks (as opposed to poisoning-based attacks) by treating all items as potential targets, and introduces a gradient alignment mechanism to explicitly correct this mismatch. This design ensures synchronized perturbations across modalities, thereby maximizing adversarial strength for robust training. Extensive experiments demonstrate that UAT-MC significantly improves robustness against promotion attacks while maintaining acceptable recommendation performance under the defense-accuracy trade-off. Code is available at this https URL. 

---
# OBLIQ-Bench: Exposing Overlooked Bottlenecks in Modern Retrievers with Latent and Implicit Queries 

**Authors**: Diane Tchuindjo, Devavrat Shah, Omar Khattab  

**Link**: [PDF](https://arxiv.org/pdf/2605.06235)  

**Abstract**: Retrieval benchmarks are increasingly saturating, but we argue that efficient search is far from a solved problem. We identify a class of queries we call oblique, which seek documents that instantiate a latent pattern, like finding all tweets that express an implicit stance, chat logs that demonstrate a particular failure mode, or transcripts that match an abstract scenario. We study three mechanisms through which obliqueness may arise and introduce OBLIQ-Bench, a suite of five oblique search problems over real long-tail corpora. OBLIQ-Bench exposes an overlooked asymmetry between retrieval and verification, where reasoning LLMs reliably recognize latent relevance whenever relevant documents are surfaced, but even sophisticated retrieval pipelines fail to surface most relevant documents in the first place. We hope that OBLIQ-Bench will drive research into retrieval architectures that efficiently capture latent patterns and implicit signals in large corpora. 

---
# Soft Deterministic Policy Gradient with Gaussian Smoothing 

**Authors**: Hyunjun Na, Donghwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.06228)  

**Abstract**: Deterministic policy gradient (DPG) is widely utilized for continuous control; however, it inherently relies on the differentiability of the critic with respect to the action during policy updates. This assumption is violated in practical control problems involving sparse or discrete rewards, leading to ill-defined policy gradients and unstable learning. To address these challenges, we propose a principled alternative based on a smoothed Bellman equation formulated via Gaussian smoothing. Specifically, we define a novel action-value function based on a smoothed Bellman equation and derive the soft deterministic policy gradient (Soft-DPG). Our formulation eliminates explicit dependence on critic action-gradients and ensures that the gradient remains well-defined even for non-smooth Q-functions. We instantiate this framework into a deep reinforcement learning algorithm, which we call soft deep deterministic policy gradient (Soft DDPG). Empirical evaluations on standard continuous control benchmarks and their discretized-reward variants show that Soft DDPG remains competitive in dense-reward settings and provides clear gains in most discretized-reward environments, where standard DDPG is more sensitive to irregular critic landscapes. 

---
# Memory Inception: Latent-Space KV Cache Manipulation for Steering LLMs 

**Authors**: Andy Zeyi Liu, Michael Zhang, Ilana Greenberg, Adam Alnasser, Lucas Baker, John Sous  

**Link**: [PDF](https://arxiv.org/pdf/2605.06225)  

**Abstract**: Steering large language models (LLMs) is usually done by either instruction prompting or activation steering. Prompting often gives strong control, but caches guidance tokens at every layer and can clutter long interactions; activation steering is compact but typically weaker and does not support large structured reminders. We introduce memory inception (MI), a training-free method that steers in latent attention space by inserting text-derived key-value (KV) banks only at selected layers. Rather than materializing reminder content throughout the prompt cache, MI treats steering as selective KV allocation, injecting latent slots only where the model routes to them. On matched personality-steering tasks, MI gives the best overall control--drift trade-off, remaining competitive with prompting while consistently outperforming CAA. On updateable guidance, MI supports mid-conversation behavior shifts without rewriting the visible transcript, achieving the highest post-shift alignment on Qwen3. On structured reasoning, MI outperforms visible prompting on HARDMath and PHYSICS (10/12 subject$\times$mode cells), serving as proxies for structured reasoning in verifiable domains, while cutting content-matched KV storage by up to 118$\times$. These results position MI as a powerful steering method when guidance is persistent, structured, or expensive to keep in the visible transcript. 

---
# When to Trust Imagination: Adaptive Action Execution for World Action Models 

**Authors**: Rui Wang, Yue Zhang, Jiehong Lin, Kuncheng Luo, Jianan Wang, Zhongrui Wang, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2605.06222)  

**Abstract**: World Action Models (WAMs) have recently emerged as a promising paradigm for robotic manipulation by jointly predicting future visual observations and future actions. However, current WAMs typically execute a fixed number of predicted actions after each model inference, leaving the robot blind to whether the imagined future remains consistent with the actual physical rollout. In this work, we formulate adaptive WAM execution as a future-reality verification problem: the robot should execute longer when the WAM-predicted future remains reliable, and replan earlier when reality deviates from imagination. To this end, we propose Future Forward Dynamics Causal Attention (FFDC), a lightweight verifier that jointly reasons over predicted future actions, predicted visual dynamics, real observations, and language instructions to estimate whether the remaining action rollout can still be trusted. FFDC enables adaptive action chunk sizes as an emergent consequence of prediction-observation consistency, preserving the efficiency of long-horizon execution while restoring responsiveness in contact-rich or difficult phases. We further introduce Mixture-of-Horizon Training to improve long-horizon trajectory coverage for adaptive execution. Experiments on the RoboTwin benchmark and in the real world demonstrate that our method achieves a strong robustness-efficiency trade-off: on RoboTwin, it reduces WAM forward passes by 69.10% and execution time by 34.02%, while improving success rate by 2.54% over the short-chunk baseline; in real-world experiments, it improves success rate by 35%. 

---
# TIDE: Every Layer Knows the Token Beneath the Context 

**Authors**: Ajay Jaiswal, Lauren Hannah, Han-Byul Kim, Duc Hoang, Mehrdad Farajtabar, Minsik Cho  

**Link**: [PDF](https://arxiv.org/pdf/2605.06216)  

**Abstract**: We revisit a universally accepted but under-examined design choice in every modern LLM: a token index is looked up once at the input embedding layer and then permanently discarded. This single-injection assumption induces two structural failures: (i) the Rare Token Problem, where a Zipf-type distribution of vocabulary causes rare-token embeddings are chronically under-trained due to receiving a fraction of the cumulative gradient signal compared to common tokens; and (ii) the Contextual Collapse Problem, where limited parameters models map distributionally similar tokens to indistinguishable hidden states. As an attempt to address both, we propose TIDE, which augments the standard transformer with EmbeddingMemory: an ensemble of K independent MemoryBlocks that map token indices to context-free semantic vectors, computed once and injected into every layer through a depth-conditioned softmax router with a learnable null bank. We theoretically and empirically establish the benefits of TIDE in addressing the issues associated with single-token identity injection as well as improve performance across multiple language modeling and downstream tasks. 

---
# FunctionalAgent: Towards end-to-end on-top functional design 

**Authors**: Yuhao Chen, Donald G. Truhlar, Xiao He  

**Link**: [PDF](https://arxiv.org/pdf/2605.06215)  

**Abstract**: Multiconfiguration pair-density functional theory (MC-PDFT) offers an efficient and accurate framework for computing electronic energies in strongly correlated molecular systems, with the quality of the on-top functional being a key determinant of its predictive accuracy. Here we introduce FunctionalAgent, an agentic system for fully automated functional development. FunctionalAgent orchestrates a team of specialized sub-agents to decompose the development process into dataset construction, active-space generation, MCSCF calculation and descriptor generation, loss-function construction, and functional fitting, optimization, and evaluation, thereby linking all stages into a closed-loop automated workflow. Using FunctionalAgent, we developed MC26, a hybrid meta-GGA on-top functional that achieves improved overall accuracy on the training set compared with other methods evaluated on the same benchmark dataset. We further introduce COF26, a new functional form that, owing to the optimized training process, achieves the best performance on both the training and test sets. 

---
# Contrastive Identification and Generation in the Limit 

**Authors**: Xiaoyu Li, Andi Han, Jiaojiao Jiang, Junbin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.06211)  

**Abstract**: In the classical identification in the limit model of Gold [1967], a stream of positive examples is presented round by round, and the learner must eventually recover the target hypothesis. Recently, Kleinberg and Mullainathan [2024] introduced generation in the limit, where the learner instead must eventually output novel elements of the target's support. Both lines of work focus on positive-only or fully labeled data. Yet many natural supervision signals are inherently relational rather than singleton, which encode relationships between examples rather than labels of individual ones. We initiate the study of contrastive identification and generation in the limit, where the learner observes a contrastive presentation of data: a stream of unordered pairs $\{x,y\}$ satisfying $h(x)\ne h(y)$ for an unknown target binary hypothesis $h$, but which element is positive is hidden from the learner. We first present three results in the noiseless setting: an exact characterization of contrastive identifiable classes (a one-line geometric refinement of Angluin [1980]'s tell-tale condition), a combinatorial dimension called contrastive closure dimension (a contrasitive analogue of the closure dimension in Raman et al. [2025]) and exactly characterizing uniform contrastive generation with tight sample complexity, and a strict hierarchy in which contrastive generation and text identification are mutually incomparable. We then prove a sharp reversal under finite adversarial corruption: there exist classes identifiable from contrastive pairs under any finite corruption budget by a single budget-independent algorithm, yet not identifiable from positive examples under even one corrupted observation. The unifying technical object is the common crossing graph, which encodes pairwise ambiguity, family-level generation obstructions, and corruption defects in a single coverage-and-incidence language. 

---
# Super-Level-Set Regression: Conditional Quantiles via Volume Minimization 

**Authors**: Sacha Braun, Michael I. Jordan, Francis Bach  

**Link**: [PDF](https://arxiv.org/pdf/2605.06210)  

**Abstract**: Constructing minimum-volume prediction regions that satisfy conditional coverage is a fundamental challenge in multivariate regression. Standard approaches rely on explicitly estimating the full conditional density and subsequently thresholding it. This two-step plug-in process is notoriously difficult, sensitive to estimation errors, and computationally expensive. One would like to instead optimize the region directly. Formulating a direct solution is challenging, however, because it requires minimizing a volume objective that is coupled with the conditional quantiles of the model's own estimation error. In this work, we address this challenge. We introduce super-level-set regression (SLS), a novel mathematical framework that successfully resolves this implicit coupling, allowing us to directly parameterize and optimize the geometric boundaries of the target conditional level sets. By bypassing full distribution estimation and leveraging flexible volume-preserving frontier functions, our approach natively captures complex, multimodal, and disjoint conditional structures end-to-end. Ultimately, SLS offers a new perspective on multivariate conditional quantile regression, replacing the restrictive assumptions of density-first methods with a direct geometric optimization strategy. 

---
# Taming the Entropy Cliff: Variable Codebook Size Quantization for Autoregressive Visual Generation 

**Authors**: Bowen Zheng, Weijian Luo, Guang Yang, Colin Zhang, Tianyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06207)  

**Abstract**: Most discrete visual tokenizers rely on a default design: every position in the sequence shares the same codebook. Researchers try to scale the codebook size $K$ to get better reconstruction performance. Such a constant-codebook design hits a fundamental information-theoretic limit. We observe that the per-position conditional entropy of the training set decays so quickly along the sequence that, after a few positions, the conditional distribution becomes essentially deterministic. On ImageNet with $K=16384$, this happens within only 2 out of 256 positions, turning the remaining 254 into a memorization problem. We call this phenomenon the Entropy Cliff and formalize it with a simple expression: $t^{*} = \lceil \log_2 N / \log_2 K \rceil$. Interestingly, this phenomenon is not observed in language, as its natural structure keeps the effective entropy per position well below the codebook capacity. To address this, we propose Variable Codebook Size Quantization (VCQ), where the codebook size $K_t$ grows monotonically along the sequence from $K_{\min}=2$ to $K_{\max}$, leaving the loss function, parameter count, and AR training procedure unchanged. With a vanilla autoregressive Transformer and standard next-token prediction, a base version of VCQ reduces gFID w/o CFG from 27.98 to 14.80 on ImageNet $256\times256$ over the baseline. Scaled up, it reaches gFID 1.71 with 684M autoregressive parameters, without any extra training techniques such as semantic regularization or causal alignment. The extreme information bottleneck at $K_{\min}=2$ naturally induces a coarse-to-fine semantic hierarchy: a linear probe on only the first 10 tokens reaches 43.8% top-1 accuracy on ImageNet, compared to 27.1% for uniform codebooks. Ultimately, these results show that what matters is not only the total capacity of the codebook, but also how that capacity is distributed and organized. 

---
# EA-WM: Event-Aware Generative World Model with Structured Kinematic-to-Visual Action Fields 

**Authors**: Zhaoyang Yang, Yurun Jin, Lizhe Qi, Cong Huang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06192)  

**Abstract**: Pretrained video diffusion models provide powerful spatiotemporal generative priors, making them a natural foundation for robotic world models. While recent world-action models jointly optimize future videos and actions, they predominantly treat video generation as an auxiliary representation for policy learning. Consequently, they insufficiently explore the inverse problem: leveraging action signals to guide video synthesis, thereby often failing to preserve precise robot spatial geometry and fine-grained robot-object interaction dynamics in the generated rollouts. To bridge this gap, we present EA-WM, an Event-Aware Generative World Model that effectively closes the loop between kinematic control and visual perception. Rather than injecting joint or end-effector actions as abstract, low-dimensional tokens, EA-WM projects actions and kinematic states directly into the target camera view as Structured Kinematic-to-Visual Action Fields. To fully exploit this geometrically grounded representation, we introduce event-aware bidirectional fusion blocks that modulate cross-branch attention, capturing object state changes and interaction dynamics. Evaluated on the comprehensive WorldArena benchmark, EA-WM achieves state-of-the-art performance, outperforming existing baselines by a significant margin. 

---
# In-Context Black-Box Optimization with Unreliable Feedback 

**Authors**: Nicolas Samuel Blumer, Julien Martinelli, Samuel Kaski  

**Link**: [PDF](https://arxiv.org/pdf/2605.06187)  

**Abstract**: Black-box optimization in science and engineering often comes with side information: experts, simulators, pretrained predictors, or heuristics can suggest which candidates look promising. This information can accelerate search, but it can also be biased, input-dependent, or misleading. Feedback-aware BO methods typically handle one task at a time, limiting their ability to generalize over multiple sources of feedback. In-context optimizers address cross-task adaptation, but usually assume that optimization history is the only available signal at test time. We study feedback-informed in-context black-box optimization (FICBO), where a pretrained optimizer conditions on both the observed history and cheap auxiliary feedback for the current candidate set. We introduce a structured feedback prior that models how feedback sources vary in their access, relevance, and distortion relative to the true objective, and use it to pretrain a feedback-aware transformer. At test time, the model estimates source reliability in context by comparing observed objective values with auxiliary signals, improving query selection. On synthetic and real-world tasks, FICBO effectively exploits informative feedback while remaining robust to weak or misleading sources, improving over other baselines. Empirical investigations further illustrate how the model perceives test-time sources, offering insights into its interpretability and decision-making process. 

---
# Retina-RAG: Retrieval-Augmented Vision-Language Modeling for Joint Retinal Diagnosis and Clinical Report Generation 

**Authors**: Abdelrahman Zaian, Sheethal Bhat, Mohamed Abdalkader, Andreas Maier  

**Link**: [PDF](https://arxiv.org/pdf/2605.06173)  

**Abstract**: Diabetic Retinopathy (DR) is a leading cause of preventable blindness among working-age adults worldwide, yet most automated screening systems are limited to image-level classification and lack clinically structured reporting. We propose Retina-RAG, a low-cost modular framework that jointly performs DR severity grading, macular edema (ME) detection, and report generation. The architecture decouples a high-performance retinal classifier and a parameter-efficient vision-language model (Qwen2.5-VL-7B-Instruct) adapted via Low-Rank Adaptation (LoRA), enabling flexible component integration. A retrieval-augmented generation (RAG) module injects curated ophthalmic knowledge together with structured classifier outputs at inference time to improve diagnostic consistency and reduce hallucinations. Retina-RAG achieves an F1-score of 0.731 for DR grading and 0.948 for ME detection, substantially outperforming zero-shot Qwen (0.096, 0.732) and MMed-RAG (0.541, 0.641) on a retinal disease detection dataset with captions. For report generation, Retina-RAG attains ROUGE-L 0.429 and SBERT similarity 0.884, exceeding all baselines. The full framework operates on a single consumer-grade GPU, demonstrating that clinically structured retinal AI can be achieved with modest computational resources. 

---
# HNC: Leveraging Hard Negative Captions towards Models with Fine-Grained Visual-Linguistic Comprehension Capabilities 

**Authors**: Esra Dönmez, Pascal Tilli, Hsiu-Yu Yang, Thang Vu, Carina Silberer  

**Link**: [PDF](https://arxiv.org/pdf/2605.06157)  

**Abstract**: Image-Text-Matching (ITM) is one of the defacto methods of learning generalized representations from a large corpus in Vision and Language (VL). However, due to the weak association between the web-collected image-text pairs, models fail to show a fine-grained understanding of the combined semantics of these modalities. To address this issue we propose Hard Negative Captions (HNC): an automatically created dataset containing foiled hard negative captions for ITM training towards achieving fine-grained cross-modal comprehension in VL. Additionally, we provide a challenging manually-created test set for benchmarking models on a fine-grained cross-modal mismatch task with varying levels of compositional complexity. Our results show the effectiveness of training on HNC by improving the models' zero-shot capabilities in detecting mismatches on diagnostic tasks and performing robustly under noisy visual input scenarios. Also, we demonstrate that HNC models yield a comparable or better initialization for fine-tuning 

---
# Entropy-Regularized Adjoint Matching for Offline RL 

**Authors**: Abdelghani Ghanem, Mounir Ghogho  

**Link**: [PDF](https://arxiv.org/pdf/2605.06156)  

**Abstract**: Integrating expressive generative policies, such as flow-matching models, into offline reinforcement learning (RL) allows agents to capture complex, multi-modal behaviors. While Q-learning with Adjoint Matching (QAM) stabilizes policy optimization via the continuous adjoint method, it remains inherently bound to the fixed behavior distribution. This dependence induces a \textit{popularity bias} that can suppress high-reward actions in low-density regions, and creates a \textit{support binding} that restricts off-manifold exploration. Existing workarounds, such as appending \textit{residual} Gaussian policies, often re-introduce the expressivity bottlenecks associated with unimodal distributions. In this work, we propose \textit{Maximum Entropy Adjoint Matching} (ME-AM), a unified framework that addresses these limitations within the continuous flow formulation. ME-AM incorporates two mechanisms: (1) a Mirror Descent entropy maximization objective that mitigates the popularity bias to facilitate the extraction of optimal policies from offline datasets, and (2) a \textit{Mixture Behavior Prior} that mathematically broadens the geometric support to encompass out-of-distribution high-reward regions. By exploring this extended geometry, ME-AM identifies robust actions while preserving the absolute continuity of the generative vector field. Empirically, ME-AM demonstrates competitive or superior performance compared to prior state-of-the-art (SOTA) methods across a diverse suite of sparse-reward continuous control environments. 

---
# AdaGamma: State-Dependent Discounting for Temporal Adaptation in Reinforcement Learning 

**Authors**: Yaomin Wang, Jianting Pan, Ran Tian, Xiaoyang Li, Yu Zhang, Hengle Qin, Tianshu YU  

**Link**: [PDF](https://arxiv.org/pdf/2605.06149)  

**Abstract**: The discount factor in reinforcement learning controls both the effective planning horizon and the strength of bootstrapping, yet most deep RL methods use a single fixed value across all states. While state-dependent discounting is conceptually appealing, naive deep actor--critic implementations can become unstable and degenerate toward TD-error collapse. We propose AdaGamma, a practical deep actor--critic method for state-dependent discounting that learns a state-dependent discount function together with a return-consistency objective to regularize the induced backup structure. On the theory side, we analyze the Bellman operator induced by state-dependent discounting and establish its basic well-posedness properties under suitable conditions. Empirically, AdaGamma integrates into both SAC and PPO, yielding consistent improvements on continuous-control benchmarks, and achieves statistically significant gains in an online A/B test on the JD Logistics platform. These results suggest that state-dependent discounting can be made effective in deep RL when coupled with a return-consistency objective that prevents degenerate target manipulation. 

---
# Learning Discrete Autoregressive Priors with Wasserstein Gradient Flow 

**Authors**: Bowen Zheng, Yihong Luo, Tianyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06148)  

**Abstract**: Discrete image tokenizers are commonly trained in two stages: first for reconstruction, and then with a prior model fitted to the frozen token sequences. This decoupling leaves the tokenizer unaware of the model that will later generate its tokens. As a result, the learned tokens may preserve image information well but still be difficult for an autoregressive (AR) prior to predict from left to right. We analyze this mismatch using Tripartite Variational Consistency (TVC), which decomposes latent-variable learning into three consistency conditions: conditional-likelihood consistency, prior consistency, and posterior consistency. TVC shows that two-stage training preserves the reconstruction side but leaves prior consistency outside the tokenizer objective: the overall token distribution is fixed before the AR prior participates in training. Motivated by this view, we add a distribution-level prior-matching signal during tokenizer training, while keeping the reconstruction objective unchanged. We optimize this signal with a Wasserstein-gradient-flow update. For hard categorical tokens, the update reduces to a token-level contrast between an auxiliary AR model that tracks the tokenizer's current token distribution and the target AR prior. It requires only forward passes through the two AR models and does not backpropagate through either of them. The resulting tokenizer, wAR-Tok, reduces AR loss and improves generation FID on CIFAR-10 and ImageNet at comparable reconstruction quality. 

---
# Unifying Goal-Conditioned RL and Unsupervised Skill Learning via Control-Maximization 

**Authors**: Alireza Modirshanechi, Benjamin Eysenbach, Peter Dayan, Eric Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2605.06145)  

**Abstract**: Unsupervised pretraining has driven empirical advances in goal-conditioned reinforcement learning (GCRL), but its theoretical foundations remain poorly understood. In particular, an influential class of methods, mutual information skill learning (MISL), discovers behaviorally diverse skills that can later be used for downstream goal-reaching. However, it remains a theoretical mystery why skills learned through MISL should support goal-reaching. A subtle challenge is that both GCRL and MISL are umbrella terms: different GCRL tasks use distinct criteria for measuring goal-reaching performance, while different MISL methods optimize distinct notions of behavioral diversity. We address this challenge and unify GCRL and MISL as instances of control maximization. We identify three canonical GCRL formulations and prove that they are fundamentally inequivalent: they can induce incompatible optimal policies even in the same environment. Nevertheless, they all share a common interpretation: a well-performing goal-conditioned policy is one whose future trajectory is highly sensitive to the commanded goal, with the precise notion of sensitivity determined by the GCRL formulation. Noting that MISL objectives can be understood as measures of skill-sensitivity akin to goal-sensitivity, we show that MISL objectives are bounded by formulation-specific downstream goal-sensitivities. These bounds establish a precise correspondence between MISL methods and downstream GCRL tasks: for every GCRL formulation, there exists a matching MISL objective for which more diverse skills afford greater downstream goal sensitivity. Our results thus lay a theoretical foundation for RL pretraining and have important practical implications, such as suggesting which pretraining objectives to use when a user cares about a specific class of downstream tasks. 

---
# AI-Generated Images: What Humans and Machines See When They Look at the Same Image 

**Authors**: Silvia Poletti, Justin Ilyes, Marcel Hasenbalg, David Fischinger, Martin Boyer  

**Link**: [PDF](https://arxiv.org/pdf/2605.06143)  

**Abstract**: The misuse of generative AI in online disinformation campaigns highlights the urgent need for transparent and explainable detection systems. In this work, we investigate how detectors for AI-generated images can be more effective in providing human-understandable explanations for their predictions. To this end, we develop a suite of detectors with various architectures and fine-tuning strategies, trained on our large-scale photorealistic fake image dataset, AIText2Image, and assess their performance on state-of-the-art text-to-image AI generators. We integrate 16 different explainable AI (XAI) methods into our detection framework, and the visual explanations are comprehensively refined and evaluated through a novel approach that prioritizes human understanding of AI-generated images, using both textual and visual responses collected from a survey of 100 participants. This framework offers insights into visual-language cues in fake image detection and into the clarity of XAI methods from a human perspective, measuring the alignment of XAI outputs with human preferences. 

---
# IRC-Bench: Recognizing Entities from Contextual Cues in First-Person Reminiscences 

**Authors**: Yehudit Aperstein, Eden Moran, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06142)  

**Abstract**: When people recount personal memories, they often refer to people, places, and events indirectly, relying on contextual cues rather than explicit names. Such implicit references are central to reminiscence narratives: first-person accounts of lived experience used in therapeutic, archival, and social settings. They pose a difficult computational problem because the intended entity must be inferred from dispersed narrative evidence rather than from a local mention. We introduce IRC-Bench, the Implicit Reminiscence Context Benchmark, for evaluating implicit entity recognition in reminiscence transcripts. The benchmark targets non-locality: entity-identifying cues are distributed across multiple, non-contiguous clauses, unlike named entity recognition, entity linking, or coreference resolution. IRC-Bench comprises 25,136 samples constructed from 12,337 Wiki-data-linked entities across 1,994 transcripts spanning 11 thematic domains. Each sample pairs an Entity-Grounded Narrative, in which the target entity is explicitly mentioned, with an Entity-Elided Narrative, in which direct mentions are removed. We evaluate 19 configurations across LLM generation, dense retrieval, RAG, and fine-tuning. QLoRA-adapted Llama 3.1 8B performs best in the open-world setting (38.94% exact match; 51.59% Jaccard), while fine-tuned DPR leads closed-world retrieval (35.38% Hit@1; 71.49% Hit@10). We release IRC-Bench with data, code, and evaluation tools. 

---
# SymDrift: One-Shot Generative Modeling under Symmetries 

**Authors**: Samir Darouich, Vinh Tong, Lluís Pastor-Pérez, Tanja Bien, Loay Mualem, Mathias Niepert  

**Link**: [PDF](https://arxiv.org/pdf/2605.06140)  

**Abstract**: Generative modeling of physical systems, such as molecules, requires learning distributions that are invariant under global symmetries, such as rotations in three-dimensional space. Equivariant diffusion and flow matching models can incorporate such invariances effectively, even when trained on a non-invariant empirical distribution, but they typically rely on costly multi-step sampling. Recently, drifting models have emerged as an efficient alternative, enabling single-step generation and achieving state-of-the-art performance in generative modeling tasks. However, we show that drifting models face a symmetry-specific challenge, since an equivariant generator does not generally produce the same drifting field as the one obtained from the symmetrized target distribution. Addressing this issue would require expensive symmetrization of the empirical distribution. To avoid this cost, we propose SymDrift, a framework that makes the drifting field itself symmetry-aware. We introduce two complementary strategies: (i) a symmetrized drift in coordinate space based on optimal alignment, and (ii) a $G$-invariant embedding that removes symmetry ambiguity by construction. Empirically, SymDrift outperforms existing one-shot methods on standard benchmarks for conformer and transition state generation, while remaining competitive with significantly more expensive multi-step approaches. By enabling one-shot inference, SymDrift reduces computational overhead by up to 40$\times$ compared to existing baselines, making it promising for high-throughput applications such as virtual drug screening and large-scale reaction network exploration. 

---
# Listwise Policy Optimization: Group-based RLVR as Target-Projection on the LLM Response Simplex 

**Authors**: Yun Qu, Qi Wang, Yixiu Mao, Heming Zou, Yuhang Jiang, Yingyue Li, Wutong Xu, Lizhou Cai, Weijie Liu, Clive Bai, Kai Yang, Yangkun Chen, Saiyong Yang, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.06139)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a standard approach for large language models (LLMs) post-training to incentivize reasoning capacity. Among existing recipes, group-based policy gradient is prevalent, which samples a group of responses per prompt and updates the policy via group-relative advantage signals. This work reveals that these optimization strategies share a common geometric structure: each implicitly defines a target distribution on the response simplex and projects toward it via first-order approximation. Building on this insight, we propose Listwise Policy Optimization (LPO) to explicitly conduct the target-projection, which demystifies the implicit target by restricting the proximal RL objective to the response simplex, and then projects the policy via exact divergence minimization. This framework provides (i) monotonic improvement on the listwise objective with bounded, zero-sum, and self-correcting projection gradients, and (ii) flexibility in divergence selection with distinct structural properties through the decoupled projection step. On diverse reasoning tasks and LLM backbones, LPO consistently improves training performance over typical policy gradient baselines under matched targets, while intrinsically preserving optimization stability and response diversity. 

---
# Autoregressive Visual Generation Needs a Prologue 

**Authors**: Bowen Zheng, Weijian Luo, Guang Yang, Colin Zhang, Tianyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06137)  

**Abstract**: In this work, we propose Prologue, an approach to bridging the reconstruction-generation gap in autoregressive (AR) image generation. Instead of modifying visual tokens to satisfy both reconstruction and generation, Prologue generates a small set of prologue tokens prepended to the visual token sequence. These prologue tokens are trained exclusively with the AR cross-entropy (CE) loss, while visual tokens remain dedicated to reconstruction. This decoupled design lets us optimize generation through the AR model's true distribution without affecting reconstruction quality, which we further formalize from an ELBO perspective. On ImageNet 256x256, Prologue-Base reduces gFID from 21.01 to 10.75 without classifier-free guidance while keeping reconstruction almost unchanged; Prologue-Large reaches a competitive rFID of 0.99 and gFID of 1.46 using a standard AR model without auxiliary semantic supervision. Interestingly, driven only by AR gradients, prologue tokens exhibit emergent semantic structure: linear probing on 16 prologue tokens reaches 35.88% Top-1, far above the 23.71% of the first 16 tokens from a standard tokenizer; resampling with fixed prologue tokens preserves a similar high-level semantic layout. Our results suggest a new direction: generation quality can be improved by introducing a separate learned generative representation while leaving the original representation intact. 

---
# BUILD-AND-FIND: An Effort-Aware Protocol for Evaluating Agent-Managed Codebases 

**Authors**: Jhen-Ke Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06136)  

**Abstract**: Most coding-agent benchmarks ask whether generated code behaves correctly. That remains essential, but repository-level engineering is increasingly agent-managed: one agent writes a repository, and later agents inspect, audit, or extend it as working context. In that setting, a generated repository is not only an answer to a task but also a communication artifact for future work. Even when strong agents nearly satisfy the visible behavioral objective, repositories can differ in how clearly they expose the intended behavior and design choices behind that behavior. We introduce BUILD-AND-FIND, a protocol for evaluating whether downstream agents can recover those intended choices from generated repositories, and how much inspection that recovery requires. For each task, a builder sees a hidden repository specification and creates a codebase; a finder sees only the codebase and a specification-traced multiple-choice question bank. The protocol separates behavioral correctness from artifact-side recovery and reports recovery accuracy, repeatability, implementation coverage, and inspection effort. Accuracy and stability act as gates: effort is interpreted only when recovery succeeds reliably. Among artifacts from which the same intent can be recovered, lower effort by the same finder suggests that the artifact makes that intent easier to locate. Question-only and spec-only controls quantify generic priors and specification access, while audits separate omitted claims from finder failures and check whether correct answers cite artifact evidence. In the released high-prior task pack, recovery accuracy is near saturation, so inspection effort and finder-specific effects provide the main panel-local comparison. 

---
# Continuous Expert Assembly: Instance-Conditioned Low-Rank Residuals for All-in-One Image Restoration 

**Authors**: Haisen He, Xiangyu Zou, SongLin Dong, Heng Li, Yihong Gong, Zhiheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.06127)  

**Abstract**: Real-world image degradation is often unknown, spatially non-uniform, and compositional, requiring all-in-one restoration models to adapt a single set of weights to diverse local corruption patterns without test-time degradation labels. Existing methods typically modulate a shared backbone with global prompts or degradation descriptors, or route features through predefined expert pools. However, compact global conditioning can bottleneck localized degradation evidence, while static expert routing may produce homogeneous updates or rely on unstable sparse assignments. We propose \textbf{Continuous Expert Assembly} (CEA), a token-wise dynamic parameterization framework for all-in-one image restoration. CEA employs a lightweight \textbf{Cross-Attention Hyper-Adapter} to probe intermediate spatial features and synthesize instance-conditioned low-rank routing bases and residual directions. Each spatial token then assembles its own residual update via dense signed dot-product affinities over the generated rank-wise components, avoiding external prompts, static expert banks, and discrete Top-
selection. The resulting assembly rule also admits a linear-attention perspective, making its dense token-wise routing behavior transparent. Experiments on AIO-3, AIO-5, and CDD-11 show that CEA improves average restoration quality over strong prompt-, descriptor-, and expert-based baselines, with the clearest gains on spatially varying and compositional degradations, while maintaining favorable parameter, FLOP, and runtime efficiency. 

---
# Dynamic Pondering Sparsity-aware Mixture-of-Experts Transformer for Event Stream based Visual Object Tracking 

**Authors**: Shiao Wang, Xiao Wang, Duoqing Yang, Wenhao Zhang, Bo Jiang, Lin Zhu, Yonghong Tian, Bin Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.06112)  

**Abstract**: Despite significant progress, RGB-based trackers remain vulnerable to challenging imaging conditions, such as low illumination and fast motion. Event cameras offer a promising alternative by asynchronously capturing pixel-wise brightness changes, providing high dynamic range and high temporal resolution. However, existing event-based trackers often neglect the intrinsic spatial sparsity and temporal density of event data, while relying on a single fixed temporal-window sampling strategy that is suboptimal under varying motion dynamics. In this paper, we propose an event sparsity-aware tracking framework that explicitly models event-density variations across multiple temporal scales. Specifically, the proposed framework progressively injects sparse, medium-density, and dense event search regions into a three-stage Vision Transformer backbone, enabling hierarchical multi-density feature learning. Furthermore, we introduce a sparsity-aware Mixture-of-Experts module to encourage expert specialization under different sparsity patterns, and design a dynamic pondering strategy to adaptively adjust the inference depth according to tracking difficulty. Extensive experiments on FE240hz, COESOT, and EventVOT demonstrate that the proposed approach achieves a favorable trade-off between tracking accuracy and computational efficiency. The source code will be released on this https URL. 

---
# Schedule-and-Calibrate: Utility-Guided Multi-Task Reinforcement Learning for Code LLMs 

**Authors**: Yujia Chen, Yang Ye, Xiao Chu, Yuchi Ma, Cuiyun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.06111)  

**Abstract**: Reinforcement learning (RL) with verifiable rewards has proven effective at post-training LLMs for coding, yet deploying separate task-specific specialists incurs costs that scale with the number of tasks, motivating a unified multi-task RL (MTRL) approach. However, existing MTRL methods treat all coding tasks uniformly, relying on fixed data curricula under a shared optimization strategy, ultimately limiting the effectiveness of multi-task training. To address these limitations, we propose ASTOR, a multi-tASk code reinforcement learning framework via uTility-driven coORdination. Centered on task utility, a signal capturing each task learning potential and cross-task synergy, ASTOR comprises two coupled modules: 1) Hierarchical Utility-Routed Data Scheduling module hierarchically allocates training budget and prioritizes informative prompts, steering training toward the most valuable data and 2) Adaptive Utility-Calibrated Policy Optimization module dynamically scales per-task KL regularization, matching update constraints to each tasks current training state. Experiments on two widely-used LLMs across four representative coding tasks demonstrate that ASTOR consistently improves a single model across all tasks, outperforming the best task-specific specialist by 9.0%-9.5% and surpassing the strongest MTRL baseline by 7.5%-12.8%. 

---
# Beyond Autoregressive RTG: Conditioning via Injection Outside Sequential Modeling in Decision Transformer 

**Authors**: Yongyi Wang, Hanyu Liu, Lingfeng Li, Bozhou Chen, Ang Li, Qirui Zheng, Xionghui Yang, Chucai Wang, Wenxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.06104)  

**Abstract**: Decision Transformer (DT) formulates offline reinforcement learning as autoregressive sequence modeling, achieving promising results by predicting actions from a sequence of Return-to-Go (RTG), state, and action tokens. However, RTG is a scalar that summarizes future rewards, containing far less information than typical state or action vectors, yet it consumes the same computational budget per token. Worse, the self-attention cost of Transformers grows quadratically with sequence length, so including RTG as a separate token adds unnecessary overhead. We propose SlimDT, which removes RTG from the autoregressive sequence. Instead, we inject RTG information into the state representations before the sequential modeling step, allowing the Transformer to process only a compact (state, action) sequence. This reduces the sequence length by one-third, directly improving inference efficiency. On the D4RL benchmark, SlimDT surpasses standard DT across various tasks and achieves performance comparable to existing state-of-the-art methods. Decoupling a sparse conditioning signal from an information-rich sequence thus yields both computational gains and higher task performance. 

---
# CredibleDFGO: Differentiable Factor Graph Optimization with Credibility Supervision 

**Authors**: Liang Qian, Penggao Yan, Penghui Xu, Li-Ta Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06100)  

**Abstract**: Global navigation satellite system (GNSS) positioning is widely used for urban navigation, but the covariance reported by the GNSS solver is often unreliable in urban canyons. Existing differentiable factor graph optimization (DFGO) methods already learn measurement weighting through the solver, but they still use position-only objectives. As a result, the mean estimate may improve while the reported covariance remains too small, too large, or wrong in shape. In this work, we propose CredibleDFGO (CDFGO), a differentiable GNSS factor graph framework that makes covariance credibility an explicit training target. The Weighting Generation Network (WGN) predicts per-satellite reliability weights. The differentiable Gauss--Newton solver maps these weights to a position estimate and posterior covariance, and proper scoring rules supervise the East--North predictive distribution end-to-end. We study negative log-likelihood (NLL), Energy Score (ES), and their combination. Results on three UrbanNav test scenes show consistent gains in uncertainty credibility. Positioning accuracy also improves on the medium-urban and harsh-urban scenes, and the mean horizontal error and 95th-percentile error improve on the deep-urban scene. On the harsh-urban Mong Kok (MK) scene, CDFGO-Combined reduces the mean horizontal error from 13.77\,m to 11.68\,m, reduces NLL from 40.63 to 6.59, and reduces ES from 12.31 to 9.05. The case studies link the MK improvement to better axis-wise consistency, more credible local covariance ellipses, and satellite-level reweighting. 

---
# VISD: Enhancing Video Reasoning via Structured Self-Distillation 

**Authors**: Hao Lin, Kunyang Lv, Xu Jiang, Jingqi Tian, Zhongjing Du, Jiayu Ding, Qiaoman Zhang, Hongbo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06094)  

**Abstract**: Training VideoLLMs for complex reasoning remains challenging due to sparse sequence level rewards and the lack of fine grained credit assignment over long, temporally grounded reasoning trajectories. While reinforcement learning with verifiable rewards (RLVR) provides reliable supervision, it fails to capture token level contributions, leading to inefficient learning. Conversely, existing self distillation methods offer dense supervision but lack structure and diagnostic specificity, and often interact unstably with reinforcement learning. In this work, we propose VISD, a structured self distillation framework that introduces diagnostically meaningful privileged information for video reasoning. VISD employs a video aware judge model to decompose reasoning quality into multiple dimensions, including answer correctness, logical consistency, and spatio-temporal grounding, and uses this structured feedback to guide a teacher policy for token level supervision. To stably integrate dense supervision with RL, we introduce a direction magnitude decoupling mechanism, where rollout level advantages computed from rewards determine update direction, while structured privileged signals modulate token level update magnitudes. This design enables semantically aligned and fine grained credit assignment, improving both reasoning faithfulness and training efficiency. Additionally, VISD incorporates curriculum scheduling and EMA based teacher stabilization to support robust optimization over long video sequences. Experiments on diverse benchmarks show that VISD consistently outperforms strong baselines, improving answer accuracy and spatio temporal grounding quality. Notably, VISD reaches these gains with nearly 2x faster convergence in optimization steps, highlighting the effectiveness of structured self supervision in improving both performance and sample efficiency for VideoLLMs. 

---
# Milestone-Guided Policy Learning for Long-Horizon Language Agents 

**Authors**: Zixuan Wang, Yuchen Yan, Hongxing Li, Teng Pan, Dingming Li, Ruiqing Zhang, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06078)  

**Abstract**: While long-horizon agentic tasks require language agents to perform dozens of sequential decisions, training such agents with reinforcement learning remains challenging. We identify two root causes: credit misattribution, where correct early actions are penalized due to terminal failures, and sample inefficiency, where scarce successful trajectories result in near-total loss of learning signal. We introduce a milestone-guided policy learning framework, BEACON, that leverages the compositional structure of long-horizon tasks to ensure precise credit assignment. BEACON partitions trajectories at milestone boundaries, applies temporal reward shaping within segments to credit partial progress, and estimates advantages at dual scales to prevent distant failures from corrupting the evaluation of local actions. On ALFWorld, WebShop, and ScienceWorld, BEACON consistently outperforms GRPO and GiGPO. Notably, on long-horizon ALFWorld tasks, BEACON achieves 92.9% success rate, nearly doubling GRPO's 53.5%, while improving effective sample utilization from 23.7% to 82.0%. These results establish milestone-anchored credit assignment as an effective paradigm for training long-horizon language agents. Code is available at this https URL. 

---
# Normalized Architectures are Natively 4-Bit 

**Authors**: Maxim Fishman, Brian Chmiel, Ron Banner, Daniel Soudry, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2605.06067)  

**Abstract**: Training large language models at 4-bit precision is critical for efficiency. We show that nGPT, an architecture that constrains weights and hidden representations to the unit hypersphere, is inherently more robust to low-precision arithmetic. This removes the need for interventions-such as applying random Hadamard transforms and performing per-tensor scaling calculations-to preserve model quality, and it enables stable end-to-end NVFP4 training. We validate this approach on both a 1.2B dense model and hybrid (Mamba-Transformer) MoE models of up to 3B/30B parameters. We trace this robustness to the dot product: while quantization noise remains largely uncorrelated in both standard and normalized architectures, the signal behaves differently. In nGPT, the hypersphere constraint enhances weak positive correlations among the element-wise products, leading to a constructive accumulation of the signal across the hidden dimension while the noise continues to average out. This yields a higher effective signal-to-noise ratio and a flatter loss landscape, with the effect strengthening as the hidden dimension grows, suggesting increasing advantages at scale. A reference implementation is available at this https URL 

---
# Causal Reinforcement Learning for Complex Card Games: A Magic The Gathering Benchmark 

**Authors**: Cristiano da Costa Cunha, Ajmal Mian, Tim French, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06066)  

**Abstract**: Causal reinforcement learning (RL) lacks benchmarks for complex systems that combine sequential decision making, hidden information, large masked action spaces, and explicit causal structure. We introduce MTG-Causal-RL, a Gymnasium benchmark built on Magic: The Gathering with a 3,077-dimensional partial observation, a 478-action masked discrete action space, five competitive Standard archetypes, three reward schemes, and a hand-specified Structural Causal Model (SCM) over strategic variables. Every episode exposes causal variables, SCM-predicted intervention effects, and per-factor credit traces, making causal credit assignment, leave-one-out cross-archetype transfer, and policy auditability first-class metrics. We adapt a panel of reference baselines: random, heuristic, masked PPO, a causal-world-model PPO variant, and an architecture-matched scalar control. We propose Causal Graph-Factored Advantage PPO (CGFA-PPO) as a reference causal agent that uses SCM parents of win probability as factor-aligned critic targets with an intervention-calibration loss. All comparisons use paired seeds, paired-bootstrap confidence intervals, and Holm-Bonferroni correction within pre-registered families. Masked PPO and CGFA-PPO reach competitive in-distribution win rates and exceed the random baseline; per-factor calibration trajectories and leave-one-out transfer gaps expose diagnostic structure that scalar win rate alone cannot. We release the benchmark, reference-baseline results, and full evaluation protocol openly. By coupling a strategically rich, partially observed domain with an explicit causal interface and statistical protocol, MTG-Causal-RL gives causal-RL, world-model, and LLM-agent research a shared testbed for questions current benchmarks cannot pose together: causal credit assignment under masked action spaces, structural transfer across archetypes, and SCM-grounded policy auditability. 

---
# TFM-Retouche: A Lightweight Input-Space Adapter for Tabular Foundation Models 

**Authors**: Duong Nguyen, Mohammed Jawhar, Nicolas Chesneau  

**Link**: [PDF](https://arxiv.org/pdf/2605.06047)  

**Abstract**: Tabular foundation models (TFMs), such as TabPFN-2.6, TabICLv2, ConTextTab, Mitra, LimiX, and TabDPT, achieve strong zero-shot performance through in-context learning, but their inductive biases remain fixed at inference time. Adapting a pretrained TFM to a specific dataset or task typically requires either full fine-tuning, which is computationally expensive, or parameter-efficient tuning methods (PEFT) such as LoRA, which must be tailored to the internal architecture of each TFM. Furthermore, the evidence on whether weight-space fine-tuning improves accuracy or calibration is mixed \citep{tanna_exploring_2026,rubachev_finetuning_2025}. We introduce TFM-Retouche, a lightweight input-space residual adapter that is architecture-agnostic by design with respect to the frozen TFM backbone. TFM-Retouche learns a small residual correction in the input space to align the input data with the inductive biases of the pretrained model. The adapter is trained end-to-end through the frozen TFM, with a post-training identity guard that falls back to the unmodified TFM whenever adaptation does not help on held-out validation. On TabArena-Lite (51 datasets spanning binary classification, multiclass classification, and regression), TabICLv2-Retouche -- the framework instantiated on TabICLv2 -- is the top-ranked method on the leaderboard with light per-task tuning and ensembling, lifting aggregate Elo by +56 over the frozen TabICLv2 base and sitting on the Pareto front of predictive quality versus both training and inference time. 

---
# Optimal Transport for LLM Reward Modeling from Noisy Preference 

**Authors**: Licheng Pan, Haochen Yang, Haoxuan Li, Yunsheng Lu, Yongqi Tong, Yinuo Wang, Shijian Wang, Zhixuan Chu, Lei Shen, Yuan Lu, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06036)  

**Abstract**: Reward models are fundamental to Reinforcement Learning from Human Feedback (RLHF), yet real-world datasets are inevitably corrupted by noisy preference. Conventional training objectives tend to overfit these errors, while existing denoising approaches often rely on homogeneous noise assumptions that fail to capture the complexity of linguistic preferences. To handle these challenges, we propose SelectiveRM, a framework grounded in optimal transport. We first devise a Joint Consistency Discrepancy to align the distribution of model predictions with preference data. Furthermore, to address the limitation of strict mass conservation which compels the model to fit outliers, we incorporate a Mass Relaxation mechanism via partial transport. This enables the autonomous exclusion of samples with noisy preference that contradict semantic consistency. Theoretically, we demonstrate that SelectiveRM optimizes a tighter upper bound on the unobserved clean risk. Extensive experiments validate that our approach significantly outperforms state-of-the-art baselines across diverse benchmarks. 

---
# Quantum Kernels for Audio Deepfake Detection Using Spectrogram Patch Features 

**Authors**: Lisan Al Amin, Rakib Hossain, Mahbubul Islam, Faisal Quader, Thanh Thi Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06035)  

**Abstract**: Quantum machine learning has emerged as a promising tool for pattern recognition, yet many audio-focused approaches still treat spectrograms as generic images and do not explicitly exploit their time-frequency structure. We propose Q-Patch, a quantum feature map tailored to audio that encodes local time-frequency patches from mel-spectrograms into quantum states using shallow, hardware-efficient circuits with adjacency-aware entanglement. Each selected patch is summarized by a compact four-dimensional acoustic descriptor and mapped to a four-qubit circuit with depth at most three, enabling practical quantum kernel construction under near-term constraints. We evaluate Q-Patch on an audio spoofing detection task using a controlled, balanced protocol and compare it with size-matched classical baselines. Q-Patch improves discrimination between bona fide and spoofed samples, achieving an area under the receiver operating characteristic curve (AUROC) of 0.87, compared with 0.82 for a radial basis function support vector machine (RBF-SVM) trained on the same patch-level features. Kernel-space analysis further reveals a clear class structure, with cross-class similarity around 0.615 and within-class self-similarity of 1.00. Overall, Q-Patch provides a practical framework for incorporating time-frequency-aware representations into quantum kernel learning for audio authenticity assessment in low-resource settings. 

---
# When AI Meets Science: Research Diversity, Interdisciplinarity, Visibility, and Retractions across Disciplines in a Global Surge 

**Authors**: Andrés F. Castro Torres, Joan Giner-Miguelez, Mercè Crosas  

**Link**: [PDF](https://arxiv.org/pdf/2605.06033)  

**Abstract**: The extent to which Artificial Intelligence (AI) can trigger generalized paradigm shifts in science is unclear. Although some of these technologies have revolutionized data collection and analysis in specific scientific fields such as Chemistry, their overall impact depends on the scope of adoption and the ways scholars use them. In this study, we document substantial differences in the timing and extent of AI adoption across countries and scientific domains from 1960 to 2015. After 2015, we find generalized exponential growth in AI adoption, with the number of AI-supported works multiplying by at least four across all domains. The transformative nature of this rapid growth is less apparent and points to multiple challenges should adoption trends persist. According to our analyses, AI-supported research is confined to very few topics with strong ties to Computer Science and conventional statistical frameworks, suggesting limited transformational potential in epistemological terms. AI-supported works are also associated with an unwarranted citation premium and exhibit substantially higher retraction rates than non-AI-supported works across most fields. Geographically, AI adoption displays pronounced heterogeneity at the country level, along with an acceleration in the relevance of middle-income countries in Asia, from China and beyond. Thus, the transformative capacity of AI in science remains largely untapped, and its rapid adoption underlines challenges in research openness, transparency, reproducibility, and ethics from a global perspective. We discuss how best research practices could boost the benefits of AI adoption and highlight fields and geographies where these trends warrant closer scrutiny. 

---
# Does Synthetic Data Help? Empirical Evidence from Deep Learning Time Series Forecasters 

**Authors**: Hugo Cazaux, Eyjólfur Ingi Ásgeirsson, Hlynur Stefánsson  

**Link**: [PDF](https://arxiv.org/pdf/2605.06032)  

**Abstract**: Synthetic data has transformed language model training, yet its role in time series forecasting remains poorly understood. We present a large-scale empirical study: nine experiment groups, 4,218 runs systematically evaluating synthetic time series augmentation across five architectures, four synthetic signals and seven datasets. The effect is sharply architecture-conditional: channel-mixing models (TimesNet, iTransformer) benefit in the majority of trials, while channel-independent models (DLinear, PatchTST) are consistently degraded. In selected low-resource settings the gains are striking: TimesNet trained on only 10\% of Weather data with synthetic augmentation surpasses the full-data baseline (4 of 16 sparsity-dataset combinations). Averaged across all architectures, augmentation hurts in 67\% of trials. We further find that only the Seasonal-Trend generator reliably helps across the tested benchmarks, and that hard curriculum switching is actively harmful (+24\% MSE degradation). These results provide concrete, actionable guidelines on how to use synthetic data: use synthetic augmentation with channel-mixing architectures, use gradual annealing schedules, and treat low-resource augmentation as architecture- and dataset-dependent. Code is available at \href{this https URL} 

---
# Quantizing With Randomized Hadamard Transforms: Efficient Heuristic Now Proven 

**Authors**: Ran Ben-Basat, William Kuszmaul, Michael Mitzenmacher, Amit Portnoy, Shay Vargaftik  

**Link**: [PDF](https://arxiv.org/pdf/2605.06014)  

**Abstract**: Uniform random rotations (URRs) are a common preprocessing step in modern quantization approaches used for gradient compression, inference acceleration, KV-cache compression, model weight quantization, and approximate nearest-neighbor search in vector databases. In practice, URRs are often replaced by randomized Hadamard transforms (RHTs), which preserve orthogonality while admitting fast implementations. The remaining issue is the performance for worst-case inputs. With a URR, each coordinate is individually distributed as a shifted beta distribution, which converges to a Gaussian distribution in high dimensions. Generally, one RHT is not suitable in the worst case, as individual coordinates can be far from these distributions. We show that after composing two RHTs on any $d$-sized input vector, the marginal distribution of every fixed coordinate of the normalized rotated vector is within $O(d^{-1/2})$ of a standard Gaussian both in Kolmogorov distance and in $1$-Wasserstein distance. We then plug these bounds into the analyses of modern compression schemes, namely DRIVE and QUIC-FL, and show that two RHTs achieve performance that asymptotically matches URRs.
However, we show that two RHTs may not be sufficient for Vector Quantization (VQ), which often requires weak correlation across fixed-size blocks of coordinates (as opposed to only marginal distribution convergence for single coordinates). We prove that a composition of three RHTs leads to decaying coordinate covariance. This ensures that any fixed, bounded, multi-dimensional VQ codebook optimized for URRs has the same expected error when using three RHTs, up to an additive term that vanishes with the dimension.
Finally, because practical inputs are rarely adversarial, we propose a linear-time ${O}(d)$ check on the input's moments to dynamically adapt the number of RHTs used at runtime to improve performance. 

---
# T2I-VeRW: Part-level Fine-grained Perception for Text-to-Image Vehicle Retrieval 

**Authors**: Xiao Wang, Ziwen Wang, Weizhe Kong, Wentao Wu, Yuehang Li, Aihua Zheng, Chenglong Li, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06012)  

**Abstract**: Vehicle Re-identification (Re-ID) aims to retrieve the most similar image to a given query from images captured by non-overlapping cameras. Extending vehicle Re-ID from image-only queries to text-based queries enables retrieval in real-world scenarios where only a witness description of the target vehicle is available. In this paper, we propose PFCVR, a Part-level Fine-grained Cross-modal Vehicle Retrieval model for text-to-image vehicle re-identification. PFCVR constructs locally paired images and texts at the part level and introduces learnable part-query tokens that aggregate both part-specific and full-sentence context before aligning with visual part features. On top of this explicit local alignment, a bi-directional mask recovery module lets each modality reconstruct its masked content under the guidance of the other, implicitly bridging local correspondences into global feature alignment. Furthermore, we construct a new large-scale dataset called T2I-VeRW, which contains 14,668 images covering 1,796 vehicle identities with fine-grained part-level annotations. Experimental results on the T2I-VeRI dataset show that PFCVR achieves 29.2\% Rank-1 accuracy, improving over the best competing method by +3.7\% percentage points. On the newly proposed T2I-VeRW benchmark, PFCVR achieves 55.2\% Rank-1 accuracy, outperforming a comprehensive set of recent state-of-the-art methods. Source code will be released on this https URL 

---
# Adding Thermal Awareness to Visual Systems in Real-Time via Distilled Diffusion Models 

**Authors**: Yuchen Guo, Junli Gong, Wenjun Dong, Yiuming Cheung, Weifeng Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.06010)  

**Abstract**: Purely RGB-based vision models often fail to provide reliable cues in challenging scenarios such as nighttime and fog, leading to degraded performance and safety risks. Infrared imaging captures heat-emitting sources and provides critical complementary information, but existing high-fidelity fusion methods suffer from prohibitive latency, rendering them impractical for real-time edge deployment. To address this, we propose FusionProxy, a real-time image fusion module designed as a fully independent, plug-and-play component with diffusion level quality. FusionProxy exploits two complementary statistics of a teacher sample ensemble: per-pixel variance in raw image space, used to weight pixel-level supervision, and per-pixel variance inside frozen foundation backbones, used to route feature-level alignment spatially. Once trained, FusionProxy can be directly integrated into any visual perception system without joint optimization. Extensive experiments demonstrate that our method achieves superior performance on static recognition tasks and significantly enhances robustness in dynamic tasks, including closed-loop autonomous driving. Crucially, FusionProxy achieves real-time inference speeds on diverse platforms, from high-end GPUs to commodity hardware, providing a flexible and generalizable solution for all-day perception. 

---
# PersonaKit (PK): A Plug-and-Play Platform for User Testing Diverse Roles in Full-Duplex Dialogue 

**Authors**: Hyunbae Jeon, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.06007)  

**Abstract**: As spoken dialogue systems expand beyond traditional assistant roles to encompass diverse personas -- such as authoritative instructors, uncooperative merchants, or distracted workers -- they require distinct, human-like turn-taking behaviors to maintain psychological immersion. However, current full-duplex systems often default to a rigid, overly accommodating ``always-yield'' policy during overlapping speech, which severely undermines character consistency for non-submissive roles. Evaluating alternative, persona-specific turn-taking strategies through empirical user studies is challenging because building real-time full-duplex test environments requires substantial engineering overhead. To address this, we present PersonaKit (PK), an open-source, low-latency web platform for the rapid prototyping and evaluation of conversational agents. Using intuitive JSON configurations, researchers can define personas, specify probabilistic interruption-handling behaviors (e.g., yield, hold, bridge, or override), and automatically deploy comparative A/B surveys. Through an in-the-wild evaluation with 8 distinct personas, we demonstrate that PersonaKit provides an extensible, end-to-end framework for studying complex sociolinguistic behaviors in next-generation spoken agents. 

---
# A Fine-Grained Understanding of Uniform Convergence for Halfspaces 

**Authors**: Aryeh Kontorovich, Kasper Green Larsen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06004)  

**Abstract**: We study the fine-grained uniform convergence behavior of halfspaces beyond worst-case VC bounds. For inhomogeneous halfspaces in $\mathbb{R}^d$ with $d\ge 2$, we show that standard first-order VC bounds are essentially tight: even consistent hypotheses can incur population error $\Theta(d\ln(n/d)/n)$, and in the agnostic setting the deviation scales as $\sqrt{\tau\ln(1/\tau)}$ at true error $\tau$. In contrast, homogeneous halfspaces in $\mathbb{R}^2$ exhibit a markedly different behavior. In the realizable case, every hypothesis consistent with the sample has error $O(1/n)$. In the agnostic case, we prove a bandwise, log-free deviation bound on each dyadic risk band via a critical-wedge localization argument. Unioning over bands incurs only a $\ln\ln n$ overhead, and we establish a matching lower bound showing this overhead is unavoidable. Together, these results give a fine-grained and nearly complete picture of uniform convergence for halfspaces, revealing sharp dimensional and structural thresholds. 

---
# Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks 

**Authors**: Guoxin Lu, Letian Sha, Qing Wang, Peijie Sun, Hao Zhou, Hua Dai, Fu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.05995)  

**Abstract**: The safety alignment of Large Language Models (LLMs) remains vulnerable to Harmful Fine-tuning (HFT). While existing defenses impose constraints on parameters, gradients, or internal representations, we observe that they can be effectively circumvented under persistent HFT. Our analysis traces this failure to the inherent redundancy of the high-dimensional parameter space: attackers exploit optimization trajectories that are orthogonal to defense constraints to restore harmful capabilities while deceptively adhering to safety restrictions. To address this, we propose Safety Bottleneck Regularization (SBR). SBR shifts the defensive focus from the redundant parameter space to the unembedding layer, which serves as a geometric bottleneck. By anchoring the final hidden states of harmful queries to those of the safety-aligned model, SBR enables the model to maintain safe responses even under persistent HFT. Extensive experiments confirm SBR's effectiveness, demonstrating that utilizing just a single safety anchor is sufficient to reduce the Harmful Score to $<$10 while preserving competitive performance on benign downstream tasks. 

---
# iPhoneBlur: A Difficulty-Stratified Benchmark for Consumer Device Motion Deblurring 

**Authors**: Abdullah Al Shafi, Kazi Saeed Alam  

**Link**: [PDF](https://arxiv.org/pdf/2605.05990)  

**Abstract**: Motion blur restoration on consumer mobile devices is typically evaluated using aggregate metrics that obscure performance variation across blur difficulty, masking model behavior under real deployment conditions. This work introduces iPhoneBlur, a difficulty-stratified benchmark of 7,400 image pairs synthesized from high-framerate iPhone 17 Pro videos captured in diverse real-world scenarios. Samples are partitioned into Easy, Medium, and Hard categories through PSNR-guided adaptive temporal windowing, with stratification validated by monotonic 2.2x increase in optical flow magnitude across tiers. Each sample includes comprehensive metadata enabling investigation of ISP-aware and difficulty-adaptive restoration strategies. Spectral analysis confirms synthesized blur exhibits high-frequency suppression patterns consistent with authentic motion degradation. Evaluation of six architectures reveals consistent 7-9 dB performance degradation from Easy to Hard subsets, a substantial gap entirely hidden by aggregate reporting. The benchmark further exposes a domain gap between professional and consumer cameras which targeted fine-tuning substantially recovers. By coupling difficulty stratification with deployment-critical metadata, iPhoneBlur enables systematic assessment of model reliability and failure modes for resource-constrained edge systems. 

---
# PragLocker: Protecting Agent Intellectual Property in Untrusted Deployments via Non-Portable Prompts 

**Authors**: Qinfeng Li, Yuntai Bao, Jianghui Hu, Wenqi Zhang, Jintao Chen, Huifeng Zhu, Yier Jin, Xuhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05974)  

**Abstract**: LLM agents rely on prompts to implement task-specific capabilities based on foundation LLMs, making agent prompts valuable intellectual property. However, in untrusted deployments, adversaries can copy and reuse these prompts with other proprietary LLMs, causing economic losses. To protect these prompts, we identify four key challenges: proactivity, runtime protection, usability, and non-portability that existing approaches fail to address. We present PragLocker, a prompt protection scheme that satisfies these requirements. PragLocker constructs function-preserving obfuscated prompts by anchoring semantics with code symbols and then using target-model feedback to inject noise, yielding prompts that only work on the target LLM. Experiments across multiple agent systems, datasets, and foundation LLMs show that PragLocker substantially reduces cross-LLM portability, maintains target performance, and remains robust against adaptive attackers. 

---
# Towards Reliable LLM Evaluation: Correcting the Winner's Curse in Adaptive Benchmarking 

**Authors**: Yang Xu, Jiefu Zhang, Haixiang Sun, Zihan Zhou, Tianyu Cao, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2605.05973)  

**Abstract**: Adaptive prompt and program search makes LLM evaluation selection-sensitive. Once benchmark items are reused inside tuning, the observed winner's score need not estimate the fresh-data performance of the full tune-then-deploy procedure. We study inference for this procedure-level target under explicit tuning budgets. We propose SIREN, a selection-aware repeated-split reporting protocol that freezes the post-search shortlist, separates splitwise selection from held-out evaluation, and uses an item-level Gaussian multiplier bootstrap for uncertainty quantification. In a fixed-shortlist regime with smooth stabilized selection, the estimator admits a first-order item-level representation, and the bootstrap yields valid simultaneous inference on a finite budget grid. This supports confidence intervals for procedure-performance curves and pre-specified equal-budget and cross-budget comparisons. Controlled simulations and MMLU-Pro tuning experiments show that winner-based reporting can be optimistic and can change deployment conclusions, while SIREN remains close to the finite-sample reporting target. 

---
# Beyond Uniform Credit Assignment: Selective Eligibility Traces for RLVR 

**Authors**: Chaoli Mou, Zhan Zhuang, Xinning Chen, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05965)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a key approach for improving the reasoning abilities of large language models. However, widely used critic-free algorithms such as Group Relative Policy Optimization (GRPO) necessitate a ``uniform credit assignment'' assumption that indiscriminately broadcast trajectory-level advantages, hindering learning efficiency by failing to distinguish critical reasoning steps. To address this limitation, we propose Selective Eligibility Traces (S-trace). Grounded in the intuition of partial trust region preservation, we initially introduce P-trace as a sample-efficient, critic-free eligibility traces method, upon which we build S-trace, implementing a sparse eligibility traces mechanism to further mitigate variance and achieve fine-grained credit assignment by selectively masking low-entropy tokens. Theoretically, we contextualize the recent Group Sequence Policy Optimization (GSPO) method within the critic-free eligibility traces framework, identifying it as a special instance of the eligibility traces method operating under uniform credit assignment. Experiments demonstrate that S-trace not only outperforms GRPO, showing gains of 0.49\% on Qwen3-1.7B and 3.16\% on Qwen3-4B, and maintaining a robust 2.98\% improvement when scaled further to Qwen3-8B in average pass@16, but notably achieves this with simultaneously higher sample and token efficiency. 

---
# Hallucination as an Anomaly: Dynamic Intervention via Probabilistic Circuits 

**Authors**: Erik Nielsen, Elia Cunegatti, Marcus Vukojevic, Giovanni Iacca  

**Link**: [PDF](https://arxiv.org/pdf/2605.05953)  

**Abstract**: One of the most critical challenges in Large Language Models is their tendency to hallucinate, i.e., produce factually incorrect responses. Existing approaches show promising results in terms of hallucination correction, but still suffer from a main limitation: they apply corrections indiscriminately to every token, corrupting also the originally correct generations. To overcome this drawback, we propose PCNET, a Probabilistic Circuit trained as a tractable density estimator over the LLM residual stream. The method detects hallucinations as geometric anomalies on the factual manifold, which is done via exact Negative Log-Likelihood computation, hence without the need for sampling, external verifiers, or weight modifications, as in existing techniques. To demonstrate its effectiveness, we exploit PCNET as a dynamic gate that distinguishes hallucinated from factual hidden states at each decoding step. This triggers our second main contribution, PC-LDCD (Probabilistic Circuit Latent Density Contrastive Decoding), only when the latent geometry deviates from factual regions, while leaving correct generations untouched. Across four LLMs, ranging from 1B to 8B models, and four benchmarks covering conversational reasoning, knowledge-intensive QA, reading comprehension, and truthfulness, PCNET achieves near-perfect hallucination detection across CoQA, SQuAD v2.0, and TriviaQA, with AUROC reaching up to 99%. Moreover, PC-LDCD obtains the highest True+Info, MC2, and MC3 scores on TruthfulQA in three out of four models, in comparison with state-of-the-art baselines, while reducing the mean corruption rate to 53.7% and achieving a preservation rate of 79.3%. Our proposed method is publicly available on GitHub. 

---
# LLM-Driven Design Space Exploration of FPGA-based Accelerators 

**Authors**: Vinamra Sharma, Xingjian Fu, Jude Haris, José Cano  

**Link**: [PDF](https://arxiv.org/pdf/2605.05920)  

**Abstract**: Designing field-programmable gate array (FPGA)-based accelerators for modern artificial intelligence workloads requires navigating a large and complex hardware design space encompassing architectural parameters, dataflow strategies, and memory hierarchies, making the process time-consuming and resource-intensive. While the SECDA methodology enables rapid hardware-software co-design of accelerators through SystemC simulation and FPGA execution, identifying optimal accelerator configurations still requires substantial manual effort and domain expertise. This work presents SECDA-DSE, a framework that integrates Large Language Models (LLMs) into the SECDA ecosystem, comprising tools built around SECDA to automate the design space exploration (DSE) of FPGA-based accelerators. SECDA-DSE combines a structured DSE Explorer for generating accelerator configurations with an LLM Stack that performs reasoning-guided exploration using retrieval-augmented generation and chain-of-thought prompting, alongside a feedback loop that enables reinforced fine-tuning for continuous improvement. We demonstrate the feasibility of SECDA-DSE through an initial high-level synthesis based evaluation of a generated accelerator design that meets synthesis timing and resource constraints on an Zynq-7000 FPGA. 

---
# Quantum-enhanced Large Language Models on Quantum Hardware via Cayley Unitary Adapters 

**Authors**: Borja Aizpurua, Sukhbinder Singh, Augustine Kshetrimayum, Saeed S. Jahromi, Roman Orus  

**Link**: [PDF](https://arxiv.org/pdf/2605.05914)  

**Abstract**: Large language models (LLMs) have transformed artificial intelligence, yet classical architectures impose a fundamental constraint: every trainable parameter demands classical memory that scales unfavourably with model size. Quantum computing offers a qualitatively different pathway, but practical demonstrations on real hardware have remained elusive for models of practical relevance. Here we show that Cayley-parameterised unitary adapters -- quantum circuit blocks inserted into the frozen projection layers of pre-trained LLMs and executed on a 156-qubit IBM Quantum System Two superconducting processor -- improve the perplexity of Llama 3.1 8B, an 8-billion-parameter model in widespread use, by 1.4% with only 6,000 additional parameters and end-to-end inference validated on real Quantum Processing Unit (QPU). A systematic study on SmolLM2 (135M parameters), chosen for its tractability, reveals monotonically improving perplexity with unitary block dimension, 83% recovery of compression-induced degradation, and correct answers to questions that both classical baselines fail -- with a sharp noise-expressivity phase transition identifying the concrete path to quantum utility at larger qubit scales. 

---
# Architecture-agnostic Lipschitz-constant Bayesian header and its application to resolve semantically proximal classification errors with vision transformers 

**Authors**: Frederik Schäfer, Luis Mandl, Lars Kälber, Tim Ricken  

**Link**: [PDF](https://arxiv.org/pdf/2605.05908)  

**Abstract**: Label noise remains a critical bottleneck for the generalization of supervised deep learning models, particularly when errors are structured rather than random. Standard robust training methods often fail in the presence of such semantically proximal classification errors. This work presents an architecture-agnostic Lipschitz-constant Bayesian header that can be integrated into feature extractors such as vision transformers, yielding the bi-Lipschitz-constrained Bayesian Vision Transformer (LipB-ViT). In contrast to conventional Bayesian layers, our approach enforces spectral normalization on both the mean and log-variance of the variational weights, which promotes calibrated predictive uncertainty and mitigates noise amplification. We further propose a novel metric to jointly capture uncertainty and confidence across misclassification rates, as well as an adaptive arithmetic-mean fusion scheme that combines feature-space proximity with predictive uncertainty to detect corrupted labels outperforming the state of the art k-nearest neighbor based identification methods by more than 7% reaching a recall of more than 0.93 at 15% semantically misclassified labels. Although computational costs increase due to Monte Carlo sampling, the method offers plug-and-play compatibility with pre-trained backbones and consistent hyperparameters across domains, suggesting strong utility for high-stakes applications with variable annotation reliability. The stabilized confidence estimates serve as the foundation for an analysis pipeline that jointly assesses dataset quality and label noise, yielding a second novel metric for their combined quantification. Lastly, we systematically evaluate LipB-ViT under both structured (adversarial) and unstructured noise at inference time, demonstrating its robustness in realistic high-noise and attack scenarios. We compare its performance against baseline methods. 

---
# VARS-FL: Validation-Aligned Client Selection for Non-IID Federated Learning in IoT Systems 

**Authors**: Mohamed Lakas, Mohamed Amine Ferrag  

**Link**: [PDF](https://arxiv.org/pdf/2605.05896)  

**Abstract**: Federated learning (FL) systems typically employ stateless client selection, treating each communication round independently and ignoring accumulated evidence of client contribution quality. Under non-IID data, this leads to slow convergence and unstable training, particularly when selection relies on local proxies (e.g., training loss) that are misaligned with the global optimization objective. These challenges are especially pronounced in Internet of Things (IoT) and Industrial IoT (IIoT) environments, where data is highly heterogeneous and distributed across devices observing different traffic patterns. In this paper, we propose VARS-FL (Validation-Aligned Reputation Scoring for Federated Learning), a client selection framework that quantifies each client's contribution using the reduction in server-side validation loss induced by its update. These per-round signals are aggregated into a Reputation score that combines a sliding-window average of recent contributions with a logarithmically scaled participation term, enabling robust exploration-exploitation selection. VARS-FL requires no changes to local training or aggregation and remains fully compatible with standard FedAvg. We evaluate VARS-FL on a 15-class non-IID IoT intrusion detection task using the Edge-IIoTset dataset, with 100 clients across multiple seeds, and compare it against FedAvg, Oort, and Power-of-Choice. VARS-FL consistently improves accuracy, F1-Macro, and loss, while accelerating convergence (up to 36% fewer rounds to reach 80% accuracy). These results demonstrate that validation-aligned, history-aware client selection provides a more reliable and efficient training process for federated learning in heterogeneous IoT environments. 

---
# Detecting AI-Generated Videos with Spiking Neural Networks 

**Authors**: Minsuk Jang, Yujin Yang, Heeseon Kim, Minseok Son, Younghun Kim, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.05895)  

**Abstract**: Modern AI-generated videos are photorealistic at the single-frame level, leaving inter-frame dynamics as the main remaining axis for detection. Existing detectors typically handle this temporal evidence in three ways: feeding the full frame sequence to a generic temporal backbone, reducing one dominant temporal cue to fixed video-level descriptors, or comparing temporal features to real-video statistics through a detection metric. These strategies degrade sharply under cross-generator evaluation, where artifact type and timescale vary across generators. On caption-paired benchmark, GenVidBench, we identify two signatures that prior detectors do not jointly exploit: AI-generated videos exhibit smoother frame-to-frame temporal residuals at the pixel level, and more compact trajectories in the semantic feature space, indicating a temporal smoothness gap at both levels. We further observe that, when raw video is fed into a Spiking Neural Networks (SNNs), fake clips elicit firing predominantly at object and motion boundaries, unlike real clips, suggesting that the SNN responds to temporal artifacts localized at edges. These cues are sparse, asynchronous, and concentrated at moments of change, which makes SNNs a natural choice for this task: their event-driven, sparsely-activated dynamics align with the structure of the residual signal in a way that dense ANN backbones do not. Building on this observation, we propose MAST, a detector that processes multi-channel temporal residuals with a spike-driven temporal branch alongside a frozen semantic encoder for cross-generator generalization. On the GenVideo benchmark, MAST achieves 93.14\% mean accuracy across 10 unseen generators under strict cross-generator evaluation, matching or surpassing the strongest ANN-based detectors and demonstrating the practical applicability of SNNs to AI-generated video detection. 

---
# Logic-Regularized Verifier Elicits Reasoning from LLMs 

**Authors**: Xinyu Wang, Changzhi Sun, Lian Cheng, Yuanbin Wu, Dell Zhang, Xiaoling Wang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.05893)  

**Abstract**: Verifiers are crucial components for enhancing modern LLMs' reasoning capability. Typicalverifiers require resource-intensive superviseddataset construction, which is costly and faceslimitations in data diversity. In this paper, wepropose LOVER, an unsupervised verifier regularized by logical rules. LOVER treats theverifier as a binary latent variable, utilizinginternal activations and enforcing three logical constraints on multiple reasoning paths:negation consistency, intra-group consistency,and inter-group consistency (grouped by thefinal answer). By incorporating logical rulesas priors, LOVER can leverage unlabeled examples and is directly compatible with any offthe-shelf LLMs. Experiments on 10 datasetsdemonstrate that LOVER significantly outperforms unsupervised baselines, achieving performance comparable to the supervised verifier(reaching its 95% level on average). The sourcecode is publicly available at this https URL. 

---
# MTL-MAD: Multi-Task Learners are Effective Medical Anomaly Detectors 

**Authors**: Bogdan Alexandru Bercean, Florinel Alin Croitoru, Vlad Hondru, Ciprian Mihai Ceausescu, Andreea Iuliana Ionescu, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05891)  

**Abstract**: Anomaly detection in medical images is a challenging task, since anomalies are not typically available during training. Recent methods leverage a single pretext task coupled with a large-scale pre-trained model to reach state-of-the-art performance. Instead, we propose to learn multiple self-supervised and pseudo-labeling tasks from scratch, using a joint model based on Mixture-of-Experts (MoE). By carefully integrating multiple proxy tasks, the joint model effectively learns a robust representation of normal anatomical structures, so that anomaly scores can be derived based on how well the multi-task learner (MTL) solves each task during inference. We perform comprehensive experiments on BMAD, a recent benchmark that comprises a broad range of medical image modalities. The empirical results indicate that our multi-task learner is an effective anomaly detector, outperforming all state-of-the-art competitors on BMAD. Moreover, our model produces interpretable anomaly maps, potentially helping physicians in providing more accurate diagnoses. 

---
# DBMSolver: A Training-free Diffusion Bridge Sampler for High-Quality Image-to-Image Translation 

**Authors**: Sankarshana Venugopal, Mohammad Mostafavi, Jonghyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.05889)  

**Abstract**: Diffusion-based image-to-image (I2I) translation excels in high-fidelity generation but suffers from slow sampling in state-of-the-art Diffusion Bridge Models (DBMs), often requiring dozens of function evaluations (NFEs). We introduce DBMSolver, a training-free sampler that exploits the semi-linear structure of DBM's underlying SDE and ODE via exponential integrators, yielding highly-efficient 1st- and 2nd-order solutions. This reduces NFEs by up to 5x while boosting quality (e.g., FID drops 53% on DIODE at 20 NFEs vs. 2nd-order baseline). Experiments on inpainting, stylization, and semantics-to-image tasks across resolutions up to 256x256 show DBMSolver sets new SOTA efficiency-quality tradeoffs, enabling real-world applicability. Our code is publicly available at this https URL. 

---
# Tuning Derivatives for Causal Fairness in Machine Learning 

**Authors**: Filip Edström, Guilherme W. F. Barros, Tetiana Gorbach, Xavier de Luna  

**Link**: [PDF](https://arxiv.org/pdf/2605.05882)  

**Abstract**: Artificial-intelligence systems are becoming ubiquitous in society, yet their predictions typically inherit biases with respect to protected attributes such as race, gender, or age. Classical fairness notions, most notably Statistical Parity (SP), demand that predictions be independent of the protected attributes, but are overly restrictive when these attributes influence mediating variables that are considered business necessities. Recent causal formulations relax SP by distinguishing allowed from not-allowed causal paths and by complementing SP with Predictive Parity (PP), requiring the predictor to replicate the legitimate influence of business-necessities. Existing path-based definitions are mainly practical when applied to categorical attributes. This paper introduces a new framework for fairness in structural causal models that is tailored to continuous protected attributes. We formalize SP and PP through path-specific partial derivatives, establish conditions under which these criteria coincide with prior causal definitions, and characterize when a fair predictor, one that satisfies SP along not-allowed paths while achieving PP along allowed paths, exists. Building on this theory, we propose a fair tuning algorithm that either constructs such a predictor or, when not possible, allows for a trade-off between SP and PP. We present experiments on simulated and real data to evaluate our proposal, compare it with previously proposed methods, and show that it performs better when PP is considered. 

---
# CITE: Anytime-Valid Statistical Inference in LLM Self-Consistency 

**Authors**: Hirofumi Ota, Naoto Iwase, Yuki Ichihara, Junpei Komiyama, Masaaki Imaizumi  

**Link**: [PDF](https://arxiv.org/pdf/2605.05873)  

**Abstract**: Large language models often improve reasoning by sampling multiple outputs and aggregating their final answers, but precise and efficient control of error levels remains a challenging task. In particular, deciding when to stop sampling remains difficult when the stopping rule is data-dependent and the set of possible answers is not known in advance. We study anytime-valid certification of a prespecified target answer as the unique mode of the model's response distribution, a guarantee distinct from answer correctness. We propose the Certification by Intersection-union Testing with E-processes (CITE) algorithm, which provably controls false certification at any prescribed level under arbitrary data-driven stopping, without requiring prior knowledge of the answer category set. We also prove an category-set-size-free stopping-time rate, establish matching minimax lower bounds up to constants in the main regime, and extend the construction to confidence-weighted voting. Simulations and LLM self-consistency experiments show empirical error control and improved certification in diffuse-tail settings. 

---
# SOPE: Stabilizing Off-Policy Evaluation for Online RL with Prior Data 

**Authors**: Carlo Romeo, Girolamo Macaluso, Alessandro Sestini, Andrew D. Bagdanov  

**Link**: [PDF](https://arxiv.org/pdf/2605.05863)  

**Abstract**: Incorporating prior data into online reinforcement learning accelerates training but typically forces a difficult trade-off between high computational costs and long, multi-stage training pipelines. While fixed-length stabilization phases are significantly more computationally efficient than static update schedules, they require task-dependent manual tuning, risking either the waste of prior knowledge or severe overfitting. To address this, we propose SOPE, an algorithm that uses an actor-aligned Off-Policy Policy Evaluation (OPE) signal as an automated early-stopping mechanism to dynamically control the length of offline training phases. By evaluating the critic on a held-out validation split under the current policy's action distribution, SOPE halts gradient updates exactly when out-of-distribution benefits saturate, eliminating the need for manual schedule tuning. Evaluated on 25 continuous control tasks from the Minari benchmark suite, SOPE improves baseline performance by up to 45.6% while reducing the required TFLOPs by up to 22x, thus balancing the tradeoff between sample and computational efficiency. These findings demonstrate that adaptive, evaluation-driven update schedules are more effective than relying on static, exhaustive update schedules. 

---
# VideoRouter: Query-Adaptive Dual Routing for Efficient Long-Video Understanding 

**Authors**: Kuanwei Lin, Wenhao Zhang, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.05848)  

**Abstract**: Video large multimodal models increasingly face a scalability bottleneck: long videos produce excessively long visual-token sequences, which sharply increase memory and latency during inference. While existing compression methods are effective in specific settings, most are either weakly query-aware or apply a fixed compression policy across frames, proving suboptimal when visual evidence is unevenly distributed over time. To address this, we present VideoRouter, a query-adaptive dual-router framework built on InternVL for budgeted evidence allocation. The Semantic Router predicts the dominant allocation policy, choosing between broad temporal coverage and adaptive high-resolution preservation, while the Image Router uses early LLM layers to score frame relevance. This enables aggressive compression on less relevant frames while preserving detail on critical evidence frames. To train both routers, we build Video-QTR-10K for allocation-policy supervision and Video-FLR-200K for frame-relevance supervision. Experiments on VideoMME, MLVU, and LongVideoBench show that VideoRouter consistently improves over the InternVL baseline under comparable or lower budgets, achieving up to a 67.9% token reduction. 

---
# LoopTrap: Termination Poisoning Attacks on LLM Agents 

**Authors**: Huiyu Xu, Zhibo Wang, Wenhui Zhang, Ziqi Zhu, Yaopeng Wang, Kui Ren, Chun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.05846)  

**Abstract**: Modern LLM agents solve complex tasks by operating in iterative execution loops, where they repeatedly reason, act, and self-evaluate progress to determine when a task is complete. In this work, we show that while this self-directed loop facilitates autonomy, it also introduces a critical risk: by injecting malicious prompts into the agent's context, an adversary can distort the agent's termination judgment, making it believe the task remains incomplete and leading to unbounded this http URL understand this threat, we define and systematically characterize it as Termination Poisoning and design 10 representative attack strategies. Through a empirical study spanning 8 LLM agents and 60 tasks, we demonstrate that different LLM agents exhibit distinct behavioral signatures that determine which strategies succeed. These transferable patterns can serve as principled guidance for crafting effective attacks against previously unseen agents and tasks, enabling scalable red-teaming beyond manually designed templates. Building on these insights, we introduce LoopTrap, an automated red-teaming framework that synthesizes target-specific malicious prompts by exploiting agent behavioral tendencies. LoopTrap first constructs a behavioral profile of the target agent along four vulnerability dimensions via lightweight probing. It then performs adaptive trap synthesis, routing to the most effective strategy and selecting optimal injections via a self-scoring mechanism. Finally, successful traps are abstracted into a reusable skill library, while failed attempts are refined through self-reflection, ensuring continuous improvement. Extensive evaluation shows that LoopTrap achieves an average of 3.57$\times$ step amplification across 8 mainstream agents, with a peak of 25$\times$. 

---
# LeakDojo: Decoding the Leakage Threats of RAG Systems 

**Authors**: Maosen Zhang, Jianshuo Dong, Boting Lu, Wenyue Li, Xiaoping Zhang, Tianwei Zhang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05818)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to leverage external knowledge, but also exposes valuable RAG databases to leakage attacks. As RAG systems grow more complex and LLMs exhibit stronger instruction-following capabilities, existing studies fall short of systematically assessing RAG leakage risks. We present LeakDojo, a configurable framework for controlled evaluation of RAG leakage. Using LeakDojo, we benchmark six existing attacks across fourteen LLMs, four datasets, and diverse RAG systems. Our study reveals that (1) query generation and adversarial instructions contribute independently to leakage, with overall leakage well approximated by their product; (2) stronger instruction-following capability correlates with higher leakage risk; and (3) improvements in RAG faithfulness can introduce increased leakage risk. These findings provide actionable insights for understanding and mitigating RAG leakage in practice. Our codebase is available at this https URL. 

---
# A Testable Certificate for Constant Collapse in Teacher-Guided VAEs 

**Authors**: Zegu Zhang, Jianhua Peng, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05813)  

**Abstract**: Posterior collapse in variational autoencoders is often diagnosed by its symptoms: a small KL term, a strong decoder, or weak use of the latent code. These signals are useful, but they do not define a collapse boundary. We study a concrete failure mode, input-independent constant collapse, and show that this case admits an exact threshold. For any fixed nonconstant teacher distribution \(T(\cdot\mid x)\), the best constant student is the dataset-average teacher distribution, and its alignment cost is the teacher mutual information \(I_T(X;T)\). Therefore, if a strictly latent-only raw witness achieves alignment loss below this value, with a safety margin, the witness cannot be constant in the input.
This identity turns a qualitative failure mode into a measurable one. In CIFAR-100 experiments with per-seed teacher search, full training stays on the certified side of the boundary, removing alignment drives the raw witness into the constant-student regime, and restarting from a collapsed checkpoint with alignment enabled restores the certificate. Tiny-ImageNet-200 fixed-target runs show the same prevention--collapse--rescue pattern across three independently searched teachers. Standard VAE-style baselines, including methods that preserve reconstruction quality or post-hoc predictability, remain negative under the raw certificate. The guarantee is intentionally narrow: it certifies that the matched nonconstant teacher-relative variation passes through the latent pathway, rather than claiming that all forms of posterior collapse have been ruled out. 

---
# LCC-LLM: Leveraging Code-Centric Large Language Models for Malware Attribution 

**Authors**: Christopher G. Pedraza Pohlenz, Hassan Jalil Hadi, Ali Hassan, Ali Shoker  

**Link**: [PDF](https://arxiv.org/pdf/2605.05807)  

**Abstract**: LLMs are increasingly explored for malware analysis; however, current LLM-based malware attribution remains limited by unsupported indicators and insufficient code-level grounding for identifying malicious and vulnerable code segments. To address these limitations, this research introduces LCC-LLM, a code-centric benchmark dataset and evidence-grounded framework for malware attribution and multi-task static malware analysis. The proposed LCCD dataset contains approximately 34K PE samples processed through a large-scale reverse-engineering pipeline and represented using decompiled C code, assembly code, CFG/FCG artifacts, hexadecimal data, PE metadata, suspicious API evidence, and structural features. Beyond dataset construction, LCC-LLM integrates LangGraph-orchestrated static analysis with multi-source cybersecurity knowledge to support evidence-grounded malware reasoning. The framework employs a seven-layer retrieval-augmented generation pipeline, CoVe for IoC validation, and a multi-dimensional quality gate to improve factual reliability and analyst-oriented decision support. Curriculum-ordered instruction data is used to fine-tune DeepSeek-R1-Distill-Qwen-14B and Qwen3-Coder-30B-A3B using QLoRA. Evaluation across 43 malware-analysis task types achieves an average semantic similarity of 0.634, with the highest task-level performance in structured report generation, IoC extraction, vulnerability assessment, malware configuration extraction, and malware class detection. In a real-world case study using MalwareBazaar samples, the grounded pipeline achieves a 10/10 structured analysis pass rate, producing CFG/FCG evidence, MITRE ATT&CK mappings, detection guidance, and analyst-ready reports. These results show that code-centric representations, retrieval grounding, and verification-guided reasoning improve the reliability and operational usefulness of LLM-assisted malware attribution. 

---
# Revealing Modular Gradient Noise Imbalance in LLMs: Calibrating Adam via Signal-to-Noise Ratio 

**Authors**: Ziqing Wen, Zhouyang Liu, Jiahuan Wang, Ping Luo, Li Shen, Dongsheng Li, Tao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.05794)  

**Abstract**: The impressive performance of large language models (LLMs) arises from their massive scale and heterogeneous module composition. However, this structural heterogeneity introduces additional optimization challenges. While adaptive optimizers such as Adam(W) provide per-parameter adaptivity, they do not explicitly account for module-level gradient heterogeneity, resulting in slower convergence, suboptimal performance, or training instability. Existing approaches typically rely on manually tuned module-specific learning rates or specific optimization strategies, which are computationally costly and difficult to generalize across tasks or models. To establish a more principled approach, we first analyze the noise-damping behavior of Adam in high-noise modules and introduce \textbf{Module-wise Learning Rate Scaling via SNR (MoLS)}. MoLS estimates module-level SNRs to scale Adam updates, allowing automated module-wise learning rate allocation without manual tuning. Empirical results through multiple LLM training benchmarks demonstrate that MoLS improves convergence speed and generalization, achieving performance comparable to carefully tuned module-specific learning rates, while remaining compatible with memory-efficient training algorithms. 

---
# Steering Visual Generation in Unified Multimodal Models with Understanding Supervision 

**Authors**: Zeyu Liu, Zanlin Ni, Yang Yue, Cheng Da, Huan Yang, Di Zhang, Kun Gai, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05781)  

**Abstract**: Unified multimodal models are envisioned to bridge the gap between understanding and generation. Yet, to achieve competitive performance, state-of-the-art models adopt largely decoupled understanding and generation components. This design, while effective for individual tasks, weakens the connection required for mutual enhancement, leaving the potential synergy empirically uncertain. We propose to explicitly restore this synergy by introducing Understanding-Oriented Post-Training (UNO), a lightweight framework that treats understanding not only as a distinct task, but also a direct supervisory signal to steer generative representations. By incorporating objectives that encode semantic abstraction (captioning) and structural details (visual regression), we enable effective gradient flow from understanding to generation. Extensive experiments on image generation and editing demonstrate that understanding can serve as an effective catalyst for generation. 

---
# The autoPET3 Challenge -- Automated Lesion Segmentation in Whole-Body PET/CT - Multitracer Multicenter Generalization 

**Authors**: Jakob Dexl, Katharina Jeblick, Andreas Mittermeier, Balthasar Schachtner, Anna Theresa Stüber, Johanna Topalis, Maximilian Rokuss, Fabian Isensee, Klaus H. Maier-Hein, Hamza Kalisch, Jens Kleesiek, Constantin M. Seibold, Hussain Alasmawi, Lap Yan Lennon Chan, Yixuan Yuan, Alexander Jaus, Rainer Stiefelhagen, Pauline Ornela Megne Choudja, Konstantin Nikolaou, Christian La Fougère, Sergios Gatidis, Matthias P. Fabritius, Maurice Heimer, Gizem Abaci, Lalith Kumar Shiyam Sundar, Rudolf A. Werner, Jens Ricke, Clemens C. Cyran, Thomas Küstner, Michael Ingrisch  

**Link**: [PDF](https://arxiv.org/pdf/2605.05775)  

**Abstract**: We report the design and results of the third autoPET challenge (MICCAI 2024), which benchmarked automated lesion segmentation in whole-body PET/CT under a compositional generalization setting. Training data comprised 1,014 [18F]-FDG PET/CT studies from the University Hospital Tübingen and 597 [18F]/[68Ga]-PSMA PET/CT studies from the LMU University Hospital Munich, constituting the largest publicly available annotated PSMA PET/CT dataset to date. The held-out test set of 200 studies covered four tracer-center combinations, two of which represented unseen compositional pairings. A complementary data-centric award category isolated the contribution of data handling strategies by restricting participants to a fixed baseline model. Seventeen teams submitted 27 algorithms, predominantly nnU-Net-based 3D networks with PET/CT channel concatenation. The top-ranked algorithm achieved a mean DSC of 0.66, FNV of 3.18 mL, and FPV of 2.78 mL across all four test conditions, improving DSC by 8% and reducing the false-negative volume by 5 mL relative to the provided baseline. Ranking was stable across bootstrap resampling and alternative ranking schemes for the top tier. Beyond the benchmark, we provide an in-depth analysis of segmentation performance at the patient and lesion level. Three main conclusions can be drawn: (1) in-domain multitracer PET/CT segmentation is sufficient and probably approaching reader agreement; (2) compositional generalization to unseen tracer-center combinations remains an open problem mainly driven by systematic volume overestimation; (3) heterogeneity and case difficulty drive performance variation substantially more than the choice of algorithm among top-ranked teams. 

---
# Adaptive Selection of LoRA Components in Privacy-Preserving Federated Learning 

**Authors**: Myoungjun Kim, Sangwoo Park, Yoseob Han, Jin-Hyun Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2605.05769)  

**Abstract**: Differentially private federated fine-tuning of large models with LoRA suffers from aggregation error caused by LoRA's multiplicative structure, which is further amplified by DP noise and degrades both stability and accuracy. Existing remedies apply a single update mode uniformly across all layers and all communication rounds (or alternate them on a fixed schedule), ignoring both the structural asymmetry between the two LoRA factors and the round-wise dynamics of training. We propose AS-LoRA, an adaptive framework defined by three axes (i) layer-wise freedom, in which each layer independently selects its active component, (ii) round-wise adaptivity, in which the selection updates over communication rounds, and (iii) a curvature-aware score derived from a second-order approximation of the loss. Theoretically, AS-LoRA eliminates the reconstruction-error floor of layer-tied schedules, accelerates convergence, implicitly biases solutions toward flatter minima, and incurs no additional privacy cost. Across GLUE, SQuAD, CIFAR-100, and Tiny-ImageNet under strict DP budgets and non-IID partitions, AS-LoRA improves over the federated LoRA baselines by up to $+7.5$ pp on GLUE and $+12.5$ pp on MNLI-mm for example, while matching or exceeding SVD-based aggregation methods at $33\text{--}180 \times$ lower aggregation cost and with negligible communication overhead. Code for the proposed method is available at this https URL. 

---
# Transformers Provably Implement In-Context Reinforcement Learning with Policy Improvement 

**Authors**: Haodong Liang, Lifeng Lai  

**Link**: [PDF](https://arxiv.org/pdf/2605.05755)  

**Abstract**: We investigate the ability of transformers to perform in-context reinforcement learning (ICRL), where a model must infer and execute learning algorithms from trajectory data without parameter updates. We show that a linear self-attention transformer block can provably implement policy-improvement methods, including semi-gradient SARSA and actor-critic, via explicit parameter constructions. Beyond existence, we design a teacher-mimicking training procedure, analyze its gradient-flow dynamics, and establish the first convergence guarantee in the ICRL literature: under suitable richness conditions on the training MDP distribution, gradient flow converges locally and exponentially to an optimal parameter manifold corresponding to the desired RL update. Empirically, training transformers on randomly generated tabular MDPs confirms these predictions: the learned models recover the parameter structure of our explicit constructions and, when deployed on unseen MDPs, deliver strong in-context control performance. Together, these results illuminate how transformer architectures internalize and execute classical reinforcement learning algorithms in context, bridging mechanistic understanding and training dynamics in ICRL. 

---
# Fourier Feature Methods for Nonlinear Causal Discovery: FFML Scoring and FFCI Testing in Mixed Data 

**Authors**: Joseph D. Ramsey  

**Link**: [PDF](https://arxiv.org/pdf/2605.05743)  

**Abstract**: Gaussian process marginal likelihood scores and kernel conditional independence tests are theoretically appealing for nonlinear causal discovery but computationally prohibitive at scale. We present two complementary RFF-based methods forming a practical toolkit for score-based, constraint-based, and hybrid causal discovery.
The Fourier Feature Marginal Likelihood (FFML) score approximates the exact GP marginal likelihood by replacing the n x n kernel Gram matrix with a finite-dimensional feature representation, reducing cost to O(nm^2 + m^3) while retaining the probabilistic interpretation and automatic complexity penalty of the exact score. FFML extends to mixed (continuous + discrete) parent sets via a product-kernel construction, with a Kronecker path for small discrete parent sets and a Hadamard-product path otherwise.
The Fourier Feature Conditional Independence (FFCI) test is a fast nonparametric CI test for mixed data. Each variable is featurized individually: continuous variables via RFF or Orthogonal Random Features (ORF), discrete variables via a Cholesky-factored categorical feature map, with blocks concatenated. Conditioning uses ridge residualization in feature space; the test statistic is a Frobenius norm of the residualized cross-covariance, approximated as a weighted sum of chi-squared variables.
Although FFML and FFCI share the same RFF/ORF machinery, they differ architecturally: FFML builds a joint kernel over a parent set for scoring, while FFCI featurizes variables individually for testing. We compare FFML to TRFF, a penalized Student-t regression alternative. Empirically, BOSS+FFML outperforms linear and kernel-ridge baselines on nonlinear data. When run through the same PC-Max implementation, FFCI and RCIT exhibit complementary precision-recall profiles: RCIT is more precise while FFCI achieves better recall and lower SHD, and runs in one third the time. 

---
# Multi-Dimensional Behavioral Evaluation of Agentic Stock Prediction Systems Using LLM Judges with Closed-Loop Reinforcement Learning Feedback 

**Authors**: Mohammad Al Ridhawi, Mahtab Haj Ali, Hussein Al Osman  

**Link**: [PDF](https://arxiv.org/pdf/2605.05739)  

**Abstract**: Agentic stock prediction systems make sequences of interdependent decisions (regime detection, pathway routing, reinforcement learning control) whose individual quality is hidden by aggregate metrics such as mean absolute percentage error (MAPE) or directional accuracy. We present a behavioral evaluation framework that addresses this gap. Behavioral traces logged at every autonomous decision point are grouped into five-day episodes and scored along six domain-specific dimensions (regime detection, routing, adaptation, risk calibration, strategy coherence, error recovery) by an ensemble of three large language model (LLM) judges (GPT 5.4, Claude 4.6 Opus, Gemini 3.1 Pro). Perturbation-based validation on 420 episodes yields targeted score drops of $-1.6$ to $-2.4$ on intended dimensions versus an average of $-0.32$ on the remaining five, with cross-model agreement up to Krippendorff's $\alpha = 0.85$. The composite behavioral score, used here only for cross-episode reporting, correlates at $\rho = 0.72$ with realized 20-day Sharpe ratio from offline backtesting. Closing the loop, the framework converts deficient per-dimension scores into a credit-assigned penalty term added to the Soft Actor-Critic (SAC) reward. Three short fine-tuning cycles, all confined to the validation period, produce on the held-out 2017-2025 test period a one-day MAPE reduction from 0.61% to 0.54% (an 11.5% relative reduction; $p<0.001$, Cohen's $d=0.31$), a directional accuracy increase from 71% to 74%, and an 18% Sharpe ratio improvement (95% bootstrap CI [8.2%, 27.4%]), with gains concentrated in high-volatility episodes where the original system was most behaviorally deficient. Results are from offline backtesting and do not address effects specific to live deployment. 

---
# CoMemNet: Contrastive Sampling with Memory Replay Network for Continual Traffic Prediction 

**Authors**: Mei Wu, Wenchao Weng, Wenxin Su, Wenjie Tang, Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.05738)  

**Abstract**: In recent years, the integration of non-topological space modeling with temporal learning methods has emerged as an effective approach for capturing spatio-temporal information in non-Euclidean graphs. However, most existing methods rely on static underlying graph structures, which are inadequate for capturing the continuously expanding and evolving patterns in streaming traffic networks. To address this challenge, we propose a simple yet efficient dual-branch continual learning framework for traffic prediction, named CoMemNet. The fast-converging Online branch undertakes the primary prediction tasks, while the momentum-updated Target branch extracts historical information using Wasserstein Distance features to create a Dynamic Contrastive Sampler (DC Sampler). This sampler selects a node set with significant dynamic network feature changes for training, effectively mitigating the issue of catastrophic forgetting. Additionally, the backbone incorporates a lightweight Node-Adaptive Temporal Memory Buffer (TMRB-N) to consolidate old knowledge through memory replay and address the risk of memory explosion. Finally, we provide two newly curated open-source datasets. Experimental results demonstrate that CoMemNet achieves state-of-the-art (SOTA) performance across all three large-scale real-world datasets. The code is available at: this https URL. 

---
# CRAFT: Forgetting-Aware Intervention-Based Adaptation for Continual Learning 

**Authors**: Md Anwar Hossen, Fatema Siddika, Juan Pablo Munoz, Tanya Roosta, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2605.05732)  

**Abstract**: Large language models (LLMs) can acquire new capabilities through fine-tuning, but continual adaptation often leads to catastrophic forgetting. We propose CRAFT, a continual learning framework that avoids updating model weights by instead learning low-rank interventions on hidden representations. CRAFT proceeds in three stages: it first routes each task to a group of similar tasks based on output-distribution divergence; it then fine-tunes the model using a Kullback-Leibler (KL) divergence against the group's prior state, which directly controls forgetting and determines convergence; finally, it merges interventions for the updated task into the shared representation using the same KL signal. This design unifies routing, regularization, and merging through a single KL-based objective. CRAFT improves overall performance and reduces forgetting compared to strong LoRA-based approaches across multiple benchmarks and model scales, while remaining robust to task ordering. These results suggest that controlling adaptation in representation space, guided by output-space divergence, provides a scalable and principled approach to continual learning in LLMs. 

---
# WARP: A Benchmark for Primal-Dual Warm-Starting of Interior-Point Solvers 

**Authors**: Dhruv Suri, Helgi Hilmarsson, Shourya Bose  

**Link**: [PDF](https://arxiv.org/pdf/2605.05728)  

**Abstract**: Solving AC Optimal Power Flow (AC-OPF) is of central importance in electricity market operations, where interior-point methods (IPMs) such as IPOPT are the standard solvers. A growing body of work uses machine learning to predict primal warm-start iterates, reporting iteration reductions of 30-46\%. We show that these reported gains rest on an inappropriate evaluation baseline: prior methods benchmark against the flat start $V_m = 1, V_a = 0$, whereas the solver's actual default - the variable-bound midpoint $(l+u)/2$ - is near-optimal for log-barrier centrality. Against this corrected baseline, no primal-only warm-start method reduces solver iterations. We trace the failure to a geometric property of interior-point methods: primal prediction accuracy is anticorrelated with convergence speed, and providing the ground-truth optimal solution $x^*$ without dual variables causes the solver to diverge. Oracle experiments establish that the complete primal-dual-barrier state $(x^*, \lambda^*, z^*, \mu^*)$ reduces IPOPT iterations from 23 to 3 - an 85\% reduction that is structurally inaccessible to primal-only methods. To enable rigorous evaluation of warm-start methods on this task, we release a benchmark suite comprising dual-labeled AC-OPF datasets with IPOPT-extracted solutions, a corrected evaluation protocol, and WARP - a topology-conditioned encode-process-decode interaction network that predicts the full interior-point state $(\hat{x}, \hat{\lambda}, \hat{z}, \hat{\mu})$ on the heterogeneous constraint graph. WARP achieves a 76\% reduction in IPOPT iterations while natively accommodating N-1 contingency topology variations without retraining. 

---
# Auto Research with Specialist Agents Develops Effective and Non-Trivial Training Recipes 

**Authors**: Jingjie Ning, Xiaochuan Li, Ji Zeng, Hao Kang, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.05724)  

**Abstract**: We study auto research as a closed empirical loop driven by external measurement. Each submitted trial carries a hypothesis, an executable code edit, an evaluator-owned outcome, and feedback that shapes the next proposal. The output is not a generated paper or a single model checkpoint, but an auditable trajectory of proposals, code diffs, experiments, scores, and failure labels. We instantiate this loop with specialist agents that partition recipe surfaces and share measured lineage across trials. The central empirical finding is that lineage feedback lets agents turn evaluator outcomes, including crashes, budget overruns, size failures, and accuracy-gate misses, into later program-level recipe edits rather than one-shot suggestions. Across 1,197 headline-run trials plus 600 Parameter Golf control trials after one-time setup and launch, humans did not choose proposals, edit recipes, override scores, or repair failed trials during the search. In the three headline runs, the same submitted-trial loop reduces Parameter Golf validation bpb by $0.81\%$, raises NanoChat-D12 CORE by $38.7\%$, and reduces CIFAR-10 Airbench96 wallclock by $4.59\%$, with each task measured by its own external evaluator and legality checks. The trace includes a strict architecture-domain audit of 157 headline-run submissions and program rewrites such as a NanoChat attention-kernel path change. Within this scope the loop autonomously writes code, submits experiments, absorbs feedback, applies and combines known techniques inside each environment, and improves public starting recipes. 

---
# SafeHarbor: Hierarchical Memory-Augmented Guardrail for LLM Agent Safety 

**Authors**: Zhe Liu, Zonghao Ying, Wenxin Zhang, Quanchen Zou, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.05704)  

**Abstract**: With the rapid evolution of foundation models, Large Language Model (LLM) agents have demonstrated increasingly powerful tool-use capabilities. However, this proficiency introduces significant security risks, as malicious actors can manipulate agents into executing tools to generate harmful content. While existing defensive mechanisms are effective, they frequently suffer from the over-refusal problem, where increased safety strictness compromises the agent's utility on benign tasks. To mitigate this trade-off, we propose \textsc{SafeHarbor}, a novel framework designed to establish precise decision boundaries for LLM agents. Unlike static guidelines, \textsc{SafeHarbor} extracts context-aware defense rules through enhanced adversarial generation. We design a local hierarchical memory system for dynamic rule injection, offering a training-free, efficient, and plug-and-play solution. Furthermore, we introduce an information entropy-based self-evolution mechanism that continuously optimizes the memory structure through dynamic node splitting and merging. Extensive experiments demonstrate that \textsc{SafeHarbor} achieves state-of-the-art performance on both ambiguous benign tasks and explicit malicious attacks, notably attaining a peak benign utility of 63.6\% on GPT-4o while maintaining a robust refusal rate exceeding 93\% against harmful requests. The source code is publicly available at this https URL. 

---
# Active Learning for Communication Structure Optimization in LLM-Based Multi-Agent Systems 

**Authors**: Huchen Yang, Xinghao Dong, Dan Negrut, Jin-Long Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05703)  

**Abstract**: Optimizing the communication structure of large language model based multi-agent systems (LLM-MAS) has been shown to improve downstream performance and reduce token usage. Existing methods typically rely on randomly sampled training tasks. However, tasks may differ substantially in difficulty and domain, and thus they are not equally informative for updating communication structure, making optimization under limited training budgets often unstable and highly sensitive to the particular training set. To actively identify the most valuable tasks for communication-structure optimization, we propose an ensemble-based information-theoretic task selection framework. The proposed method estimates task informativeness by how much a candidate task changes the distribution over graph parameters, using ensemble Kalman inversion as an efficient and derivative-free approximation of the corresponding Bayesian update. The resulting estimator is especially suitable for black-box and noisy multi-agent systems. To enhance scalability, we construct a compact candidate pool through embedding-based representative selection and combine the informative selection with surrogate modeling and batch Thompson sampling. We validate our method in both benign settings and settings with agent attacks, demonstrating its effectiveness for communication-structure optimization under constrained computational budgets. 

---
# An Empirical Study of Proactive Coding Assistants in Real-World Software Development 

**Authors**: Lehui Li, Ruixuan Jia, Guo-Ye Yang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.05700)  

**Abstract**: Large language model (LLM)-based coding assistants have made substantial progress, yet most systems remain reactive, requiring developers to explicitly formulate their needs. Proactive coding assistants aim to infer latent developer intent from integrated development environment (IDE) interactions and repository context, thereby reducing interaction overhead and supporting more seamless assistance. However, research in this direction is limited by the scarcity of large-scale real-world developer behavior data. Existing studies therefore often rely on LLM-simulated IDE traces, whose fidelity to real development behavior remains unclear. In this paper, we investigate this simulation-to-reality gap through a large-scale empirical study. We collect real IDE interaction traces from 1{,}246 experienced industry developers over three consecutive days using a custom Visual Studio Code extension, and construct paired LLM-simulated traces for controlled comparison. Our analysis shows that simulated traces differ substantially from real traces in behavioral diversity, temporal structure, and exploratory patterns. Based on the collected data, we introduce \textbf{ProCodeBench}, a real-world benchmark for proactive intent prediction. Experiments with representative LLMs, retrieval-augmented methods, and agentic baselines show that current approaches remain far from reliable under real IDE traces, suggesting that simulation-based evaluation can overestimate real-world performance. Finally, our training study shows that simulated data cannot replace real data, but can complement it when used before real-world fine-tuning. These findings highlight the importance of real developer behavior data for evaluating and training proactive coding assistants. 

---
# When Quantization Is Free: An int4 KV Cache That Outruns fp16 on Apple Silicon 

**Authors**: Mohamed Amine Bergach  

**Link**: [PDF](https://arxiv.org/pdf/2605.05699)  

**Abstract**: KV-cache quantization is framed as a quality--latency trade-off. We show it is \emph{inverted} on Apple Silicon's unified memory: a single fused Metal kernel (sign-randomized FFT $+$ per-channel $\lambda$ $+$ per-group abs-max $+$ int4 nibble pack), exposed as a HuggingFace \texttt{Cache} subclass, runs \emph{faster than fp16} across $256$--$4096$-token prefixes on Gemma-3 1B ($-3$ to $-8\%$ ms/tok) and at short context on Qwen2.5-1.5B ($-0.7$ to $-2.6\%$ through $1$K), with $3\times$ persistent memory compression and quality preserved ($\dPPL = 0.000$ Qwen short-prompt; $+3.6$ hook $\dPPL$ Gemma). The kernel's $\sim\!25$\,ns/vec overhead is below the bandwidth savings from $3\times$ compression. The fused kernel also closes Qwen's 4-bit per-token catastrophe ($\dPPL = +7975 \to +638.6$, $12.5\times$ reduction) at $182$\,GFLOPS / $D{=}128$. Supporting findings: $\SRFT$ and $\SRHT$ are statistically indistinguishable for KV quality (we pick $\SRFT$ for mixed-radix and matrix-multiply alignment); a learned-rotation ablation surfaces a regularization role for the fixed random SRFT base (learning $R+\lambda$ without SRFT lowers calibration MSE $84.9\%$ vs $50.3\%$ but yields worse PPL); Householder rotations at $k{=}d/2$ reflectors are effectively lossless at $d{=}256$. 

---
# Budgeted Attention Allocation: Cost-Conditioned Compute Control for Efficient Transformers 

**Authors**: Amrit Nidhi  

**Link**: [PDF](https://arxiv.org/pdf/2605.05697)  

**Abstract**: Transformers usually expose one inference cost per trained model, while deployed systems often need multiple cost-quality operating points. We study Budgeted Attention Allocation, a monotone head-gating mechanism conditioned on a requested attention budget. Dense warm-starting is important for stability: on a robust synthetic sequence task, one budgeted model reaches 99.7% accuracy at 0.303 estimated attention cost and 100.0% accuracy at 0.504 cost. On held-out AG News with a custom word-level transformer, hard-gate adaptation turns soft cost control into measured single-thread CPU speed, reaching 82.1% accuracy with 1.28x speedup at budget 0.50. In pretrained BERT-Mini AG News, budgeted structural pruning reaches 87.6% accuracy with 1.20x speedup at budget 0.50; a validation-ranked zero-shot dense post-hoc structural baseline reaches 86.1%, and one recovery epoch raises that per-budget specialist to 87.9%. On DBpedia14, BERT-Mini budgeted gates reach 97.4% at exact budget 0.50 versus 96.6% for dense full attention. Static fixed-budget gates and recovered dense specialists remain strong. The contribution is therefore not universal dominance, but a reproducible feasibility study of one controllable checkpoint across budgets that can trade attention cost for accuracy and be converted into measured structural speedups on small CPU benchmarks. 

---
# Irminsul: MLA-Native Position-Independent Caching for Agentic LLM Serving 

**Authors**: Bole Ma, Jan Eitzinger, Harald Köstler  

**Link**: [PDF](https://arxiv.org/pdf/2605.05696)  

**Abstract**: Agentic LLM workloads put bit-identical tokens at shifted positions every turn, voiding prefix caches at the first byte of divergence. Operators report cache-hit regressions ranging from moderate slowdowns to severe TTFT spikes of 10-16s on unchanged content. Prior position-independent caching systems correct RoPE on the full $d_K$-dimensional key, an architectural cost imposed by GQA, not by caching itself. Multi-Head Latent Attention, deployed at scale in DeepSeek-V2/V3/R1, Kimi-K2/Moonlight, GLM-5, and Mistral Large 3, factors each KV row into a position-free $c_{KV}$ and a 64-dim $k_r$ correctable in closed form; this structure motivates content-addressed caching as a natural fit rather than a GQA workaround. We present Irminsul, which extends SGLang's radix cache with content-hash keying over CDC-chunked segments and a $\delta$-rotation rule for $k_r$. We evaluate three native MLA-MoE deployments - DeepSeek-V2-Lite (16B/2.4B), Kimi Moonlight-16B-A3B, and JoyAI-Flash (48B/3B) - with output-consistency on all three and recovery measured on the two endpoints; Irminsul recovers up to ~83% of prompt tokens above exact-prefix on agentic traffic while delivering 63% prefill energy savings per cache hit. We argue that content-addressed caching belongs in the serving stack as a first-class primitive, not a retrofit over prefix matching. 

---
# CFE-PPAR: Compression-friendly encryption for privacy-preserving action recognition leveraging video transformers 

**Authors**: Haiwei Lin, Shoko Imaizumi, Hitoshi Kiya  

**Link**: [PDF](https://arxiv.org/pdf/2605.05692)  

**Abstract**: Privacy-preserving action recognition (PPAR) enables machines to understand human activities in videos without revealing sensitive visual content. Among the various strategies for PPAR, encryption-based methods achieve strong privacy protection while maintaining high recognition performance. However, these methods lead to a catastrophic decrease in recognition performance and visual quality when the encrypted videos are compressed. That is, the previous methods are not compression-friendly. To address these issues, in this paper, we propose the first compression-friendly encryption method for PPAR, called CFE-PPAR. In CFE-PPAR, videos encrypted with secret keys can be directly recognized by a video transformer, which uses parameters transformed by the same keys as those used for video encryption. In experiments, it is verified that CFE-PPAR outperforms previous methods on the UCF101 and HMDB51 datasets under Motion-JPEG and H.264 compression. 

---
# Temporal Functional Circuits: From Spline Plots to Faithful Explanations in KAN Forecasting 

**Authors**: Naveen Mysore  

**Link**: [PDF](https://arxiv.org/pdf/2605.05685)  

**Abstract**: Unlike MLPs, Kolmogorov-Arnold Networks (KANs) expose explicit learnable edge functions on every connection, enabling mechanistic explanation in time-series forecasting. This paper introduces Temporal Functional Circuits, a framework that transforms KAN edge functions from latent visualizations into faithful, temporally grounded explanations. Built on a gated residual KAN that decomposes forecasts into a linear base and a sparsely activated KAN correction, the framework (i) maps each edge to input lags via output-aware attribution, (ii) ranks edges by learned activation range, and (iii) validates faithfulness through edge-level interventions including zeroing and spline removal. Removing the learned B-spline component while retaining the base SiLU term degrades forecasts, providing evidence that the spline shape itself carries predictive value beyond the base activation. On four synthetic regimes of increasing complexity, the learned gate opens progressively wider as signal complexity grows. On regime-switching signals, gated KAN achieves 59% lower MSE than linear-only models. Across eight benchmarks, the gated architecture is competitive with linear, attention, and MLP alternatives, while providing interpretable edge functions that MLP-based corrections cannot offer. 

---
# PersonaTeaming: Supporting Persona-Driven Red-Teaming for Generative AI 

**Authors**: Wesley Hanwen Deng, Mingxi Yan, Sunnie S. Y. Kim, Akshita Jha, Lauren Wilcox, Kenneth Holstein, Motahhare Eslami, Leon A. Gatys  

**Link**: [PDF](https://arxiv.org/pdf/2605.05682)  

**Abstract**: Recent developments in AI safety research have called for red-teaming methods that effectively surface potential risks posed by generative AI models, with growing emphasis on how red-teamers' backgrounds and perspectives shape their strategies and the risks they uncover. While automated red-teaming approaches promise to complement human red-teaming through larger-scale exploration, existing automated approaches do not account for human identities and rarely incorporate human inputs. In this work, we explore persona-driven red-teaming to advance both automated red-teaming and human-AI collaboration. We first develop PersonaTeaming Workflow, which incorporates personas into the adversarial prompt generation process to explore a wider spectrum of adversarial strategies. Compared to RainbowPlus, a state-of-the-art automated red-teaming method, PersonaTeaming Workflow achieves higher attack success rates while maintaining prompt diversity. However, since automated personas only approximate real human perspectives, we further instantiate PersonaTeaming Workflow as PersonaTeaming Playground, a user-facing interface that enables red-teamers to author their own personas and collaborate with AI to mutate and refine prompts. In a user study with 11 industry practitioners, we found that PersonaTeaming Playground enabled diverse red-teaming strategies and outputs that practitioners perceived as useful, and that AI-generated suggestions in the PersonaTeaming Playground encouraged out-of-the-box thinking even when practitioners did not follow them strictly. Together, our work advances both automated and human-in-the-loop approaches to red-teaming, while shedding light on interaction patterns and design insights for supporting human-AI collaboration in generative AI red-teaming. 

---
# Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning 

**Authors**: Bing Wang, Ximing Li, Changchun Li, Jinjin Chi, Gang Niu, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2605.05676)  

**Abstract**: Recently, the prominent performance of large language models (LLMs) has been largely driven by multi-task instruct-tuning. Unfortunately, this training paradigm suffers from a key issue, named cross-task interference, due to conflicting gradients over shared parameters among different tasks. Some previous methods mitigate this issue by isolating task-specific parameters, e.g., task-specific neuron selection and mixture-of-experts. In this paper, we empirically reveal that the cross-task interference still exists for the existing solutions because of many parameters also shared by different tasks, and accordingly, we propose a novel solution, namely Basic Abilities Decomposition for multi-task Instruct-Tuning (BADIT). Specifically, we empirically find that certain parameters are consistently co-activated, and that co-activated parameters naturally organize into base groups. This motivates us to analogize that LLMs encode several orthogonal basic abilities, and that any task can be represented as a linear combination of these abilities. Accordingly, we propose BADIT that decomposes LLM parameters into orthogonal high-singular-value LoRA experts representing basic abilities, and dynamically enforces their orthogonality during training via spherical clustering of rank-1 components. We conduct extensive experiments on the SuperNI benchmark with 6 LLMs, and empirical results demonstrate that BADIT can outperform SOTA methods and mitigate the degree of cross-task interference. 

---
# EGA: Adapting Frozen Encoders for Vector Search with Bounded Out-of-Distribution Degradation 

**Authors**: Dongfang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.05674)  

**Abstract**: Vector search systems built on frozen vision encoders face queries from unseen classes at deployment, yet existing adapter training collapses under this shift: high-capacity adapters with global contrastive losses silently reassign unseen-class samples to wrong seen-class clusters, dropping worst-case Label Precision by over 40 points below the frozen baseline in our tests. We propose Euclidean Geodesic Alignment (EGA), a residual adapter that couples three principles: zero initialization, local triplet loss, and hypersphere projection. These collectively induce a self-limiting dynamic: triplets that already satisfy a small margin stop producing gradients, so the adapter automatically stops updating where the local geometry is already correct. Our experiments show that at convergence $96.5\%$ of triplets are gradient-free, leaving unseen-class regions largely untouched while still enabling full-capacity refinement of seen classes. Across five diverse out-of-distribution (OOD) benchmarks, EGA achieves the highest worst-case Label Precision on the four primary splits and a consistent improvement on the fifth. The design also transfers to stronger backbones in addition to CLIP, and we provide an analytical justification linking gradient sparsity to bounded OOD perturbation. 

---
# XL-SafetyBench: A Country-Grounded Cross-Cultural Benchmark for LLM Safety and Cultural Sensitivity 

**Authors**: Dasol Choi, Eugenia Kim, Jaewon Noh, Sang Seo, Eunmi Kim, Myunggyo Oh, Yunjin Park, Brigitta Jesica Kartono, Josef Pichlmeier, Helena Berndt, Sai Krishna Mendu, Glenn Johannes Tungka, Özlem Gökçe, Suresh Gehlot, Katherine Pratt, Amanda Minnich, Haon Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.05662)  

**Abstract**: Current LLM safety benchmarks are predominantly English-centric and often rely on translation, failing to capture country-specific harms. Moreover, they rarely evaluate a model's ability to detect culturally embedded sensitivities as distinct from universal harms. We introduce XL-SafetyBench. a suite of 5,500 test cases across 10 country-language pairs, comprising a Jailbreak Benchmark of country-grounded adversarial prompts and a Cultural Benchmark where local sensitivities are embedded within innocuous requests. Each item is constructed via a multi-stage pipeline that combines LLM-assisted discovery, automated validation gates, and dual independent native-speaker annotators per country. To distinguish principled refusal from comprehension failure, we evaluate Attack Success Rate (ASR) alongside two complementary metrics we introduce: Neutral-Safe Rate (NSR) and Cultural Sensitivity Rate (CSR). Evaluating 10 frontier and 27 local LLMs reveals two key findings. First, jailbreak robustness and cultural awareness do not show a coupled relationship among frontier models, so a composite safety score obscures per-axis variation. Second, local models exhibit a near-linear ASR-NSR trade-off (r = -0.81), indicating that their apparent safety reflects generation failure rather than genuine alignment. XL-SafetyBench enables more nuanced, cross-cultural safety evaluation in the multilingual era. 

---
# The Missing Evaluation Axis: What 10,000 Student Submissions Reveal About AI Tutor Effectiveness 

**Authors**: Rose Niousha, Samantha Boatright Smith, Bita Akram, Peter Brusilovsky, Arto Hellas, Juho Leinonen, John DeNero, Narges Norouzi  

**Link**: [PDF](https://arxiv.org/pdf/2605.05648)  

**Abstract**: Current Artificial Intelligence (AI)-based tutoring systems (AI tutors) are primarily evaluated based on the pedagogical quality of their feedback messages. While important, pedagogy alone is insufficient because it ignores a critical question: what do students actually do with the feedback they receive? We argue that AI tutor evaluation should be extended with a behavioral dimension grounded in student interaction data, which complements pedagogical assessment. We propose an evaluation framework and apply it to 10,235 code submissions with corresponding AI tutor feedback from an introductory undergraduate programming course to measure whether students act on tutor feedback and whether those actions are applied correctly. Using this framework to compare two deployed AI tutors across different semesters in a large-scale introductory computer science course reveals substantial differences in student engagement patterns that are not captured by pedagogy-only evaluation. Moreover, these engagement-based behavioral signals are more strongly associated with student perception of helpful feedback than pedagogical quality alone, providing a more complete and actionable picture of AI tutor performance. 

---
# One Turn Too Late: Response-Aware Defense Against Hidden Malicious Intent in Multi-Turn Dialogue 

**Authors**: Xinjie Shen, Rongzhe Wei, Peizhi Niu, Haoyu Wang, Ruihan Wu, Eli Chien, Bo Li, Pin-Yu Chen, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.05630)  

**Abstract**: Hidden malicious intent in multi-turn dialogue poses a growing threat to deployed large language models (LLMs). Rather than exposing a harmful objective in a single prompt, increasingly capable attackers can distribute their intent across multiple benign-looking turns. Recent studies show that even modern commercial models with advanced guardrails remain vulnerable to such attacks despite advances in safety alignment and external guardrails. In this work, we address this challenge by detecting the earliest turn at which delivering the candidate response would make the accumulated interaction sufficient to enable harmful action. This objective requires precise turn-level intervention that identifies the harm-enabling closure point while avoiding premature refusal of benign exploratory conversations. To further support training and evaluation, we construct the Multi-Turn Intent Dataset (MTID), which contains branching attack rollouts, matched benign hard negatives, and annotations of the earliest harm-enabling turns. We show that MTID helps enable a turn-level monitor TurnGate, which substantially outperforms existing baselines in harmful-intent detection while maintaining low over-refusal rates. TurnGate further generalizes across domains, attacker pipelines, and target models. Our code is available at this https URL. 

---
# Leveraging Image Generators to Address Training Data Scarcity: The Gen4Regen Dataset for Forest Regeneration Mapping 

**Authors**: Gabriel Jeanson, David-Alexandre Duclos, William Larrivée-Hardy, Noé Cochet, Matěj Boxan, Anthony Deschênes, François Pomerleau, Philippe Giguère  

**Link**: [PDF](https://arxiv.org/pdf/2605.05627)  

**Abstract**: Sustainable forest management relies on precise species composition mapping, yet traditional ground surveys are labour-intensive and geographically constrained. While Uncrewed Aerial Vehicles (UAVs) offer scalable data collection, the transition to deep learning-based interpretation is bottlenecked by the severe scarcity of expert-annotated imagery, particularly in complex, visually heterogeneous regeneration zones. This paper addresses the dual challenges of data scarcity and extreme class imbalance in the semantic segmentation of fine-grained forest regeneration species by providing a scalable framework that reduces reliance on manual photo-interpretation for high-resolution, millimetre-level aerial imagery. Importantly, we leverage the large-scale vision-language Nano Banana Pro model to simultaneously generate high-fidelity images and their corresponding pixel-aligned semantic masks from prompts. We introduce WilDReF-Q-V2, an expansion of a natural forest dataset with 13 977 new unlabelled and 50 labelled real images, as well as the Gen4Regen dataset, featuring 2101 pairs of synthetic images and semantic masks. Our methodology integrates real-world data with AI-generated images, highlighting that AI-generated data is highly complementary to real-world data, with unified training yielding an F1 score improvement of over 15 %pt compared to purely supervised baselines. Furthermore, we demonstrate that even small quantities of prompt-generated data significantly improve performance for underrepresented species, some of which saw per-species F1 score gains of up to 30 %pt. We conclude that vision-language models can serve as agile data generators, effectively bootstrapping perception tasks for niche AI domains where expert labels are scarce or unavailable. Our datasets, source code, and models will be available at this https URL. 

---
# When2Speak: A Dataset for Temporal Participation and Turn-Taking in Multi-Party Conversations for Large Language Models 

**Authors**: Vihaan Nama, Shreya Mendi, Zian Ye, Brinnae Bent  

**Link**: [PDF](https://arxiv.org/pdf/2605.05626)  

**Abstract**: Large Language Models (LLMs) excel at generating contextually appropriate responses but remain poorly calibrated for multi-party conversations, where deciding when to speak is as critical as what to say. In such settings, naively responding at every turn leads to excessive interruptions and degraded conversational coherence. We introduce When2Speak, a grounded synthetic dataset and four-stage generation pipeline for learning intervention timing in group interactions. The dataset comprises over 215,000 examples derived from 16,000 conversations involving 2-6 speakers, spanning diverse conversational styles, tones, and participant dynamics, and explicitly modeling SPEAK vs. SILENT decisions at each turn. Our pipeline combines real-world grounding, structured augmentation, controlled transcript synthesis, and fine-tuning-ready supervision, and is fully open-sourced to support reproducibility and adaptation to domain-specific conversational norms. Across multiple model families, supervised fine-tuning (SFT) on When2Speak significantly outperforms zero-shot baselines (e.g., the average Macro F1 increase across 4B+ parameter models was 60%, with the largest increase being 120%). However, SFT-trained models remain systematically over-conservative, missing nearly half of warranted interventions as seen through the Missed Intervention Rate (MIR), which was on average 0.50 and is noticed even at larger model sizes. To address this limitation, we apply reinforcement learning with asymmetric reward shaping, which reduces MIR to 0.186-0.218 and increases recall from 0.479 to 0.78-0.81. Our findings establish that temporal participation is a distinct and trainable dimension of conversational intelligence, and that grounded synthetic data provides an effective and scalable pathway for enabling LLMs to participate more naturally and appropriately in multi-party interactions. 

---
# X-Voice: Enabling Everyone to Speak 30 Languages via Zero-Shot Cross-Lingual Voice Cloning 

**Authors**: Rixi Xu, Qingyu Liu, Haitao Li, Yushen Chen, Zhikang Niu, Yunting Yang, Jian Zhao, Ke Li, Berrak Sisman, Qinyuan Cheng, Xipeng Qiu, Kai Yu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.05611)  

**Abstract**: In this paper, we present X-Voice, a 0.4B multilingual zero-shot voice cloning model that clones arbitrary voices and enables everyone to speak 30 languages. X-Voice is trained on a 420K-hour multilingual corpus using the International Phonetic Alphabet (IPA) as a unified representation. To eliminate the reliance on prompt text without complex preprocessing like forced alignment, we design a two-stage training paradigm. In Stage 1, we establish X-Voice$_{\text{s1}}$ through standard conditional flow-matching training and use it to synthesize 10K hours of speaker-consistent segments as audio prompts. In Stage 2, we fine-tune on these audio pairs with prompt text masked to derive X-Voice$_{\text{s2}}$, which enables zero-shot voice cloning without requiring transcripts of audio prompts. Architecturally, we extend F5-TTS by implementing a dual-level injection of language identifiers and decoupling and scheduling of Classifier-Free Guidance to facilitate multilingual speech synthesis. Subjective and objective evaluation results demonstrate that X-Voice outperforms existing flow-matching based multilingual systems like LEMAS-TTS and achieves zero-shot cross-lingual cloning capabilities comparable to billion-scale models such as Qwen3-TTS. To facilitate research transparency and community advancement, we open-source all related resources. 

---
# Nearly Optimal Attention Coresets 

**Authors**: Edo Liberty, Alexandr Andoni, Eldar Kleiner  

**Link**: [PDF](https://arxiv.org/pdf/2605.05602)  

**Abstract**: We consider the problem of estimating the Attention mechanism in small space, and prove the existence of coresets for it of nearly optimal size. Specifically, we show that for any set of unit-norm keys and values $(K,V)$ in $\mathbb{R}^d$, there exists a subset $(K',V')$ of size at most $O({\sqrt{d} e^{\rho+o(\rho)}/\varepsilon})$ such that \[ \left\| \operatorname{Attn}(q,K,V)- \operatorname{Attn}(q,K',V') \right\| \le \varepsilon \] simultaneously for all queries whose norm is bounded by $\rho$. This outperforms the best known results for this problem. We also offer an improved lower bound showing that $\varepsilon$-coresets must have size $\Omega({\sqrt{d} e^{\rho}/\epsilon})$. 

---
# Accelerating LMO-Based Optimization via Implicit Gradient Transport 

**Authors**: Won-Jun Jang, Si-Hyeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.05577)  

**Abstract**: Recent optimizers such as Lion and Muon have demonstrated strong empirical performance by normalizing gradient momentum via linear minimization oracles (LMOs). While variance reduction has been explored to accelerate LMO-based methods, it typically incurs substantial computational overhead due to additional gradient evaluations. At the same time, the theoretical understanding of LMO-based methods remains fragmented across unconstrained and constrained formulations. Motivated by these limitations, we propose \emph{LMO-IGT}, a new class of stochastic LMO-based methods leveraging implicit gradient transport (IGT). We further introduce a unified framework for stochastic LMO-based optimization together with a new stationarity measure, the \emph{regularized support function} (RSF), which bridges gradient-norm and Frank--Wolfe-gap notions within a common framework. By evaluating stochastic gradients at transported points, LMO-IGT accelerates convergence while retaining the single-gradient-per-iteration structure of standard stochastic LMO. Our analysis establishes that stochastic LMO achieves an iteration complexity of $\mathcal{O}(\varepsilon^{-4})$, variance-reduced LMO achieves $\mathcal{O}(\varepsilon^{-3})$ at the cost of additional gradient evaluations, and LMO-IGT achieves $\mathcal{O}(\varepsilon^{-3.5})$ using only a single stochastic gradient per iteration. Empirically, LMO-IGT consistently improves over stochastic LMO counterparts with negligible overhead. Among its instantiations, Muon-IGT achieves the strongest overall performance across evaluated settings, demonstrating that IGT provides an effective and practical acceleration mechanism for modern LMO-based optimization. 

---
# AstroAlertBench: Evaluating the Accuracy, Reasoning, and Honesty of Multimodal LLMs in Astronomical Classification 

**Authors**: Claire Chen, Jiabao Sean Xiao, Shuze Daniel Liu, Facundo Perez Paolino, Luke Handley, Theophile Jegou du Laz, Ricky Nilsson, Alice Zou, Matthew Graham, Ashish Mahabal  

**Link**: [PDF](https://arxiv.org/pdf/2605.05573)  

**Abstract**: Modern astronomical observatories generate a massive volume of multimodal data, creating a critical bottleneck for expert human review. While multimodal large language models (LLMs) have shown promise in interpreting complex visual and textual inputs, their ability to perform specialized scientific classification while providing interpretable reasoning remains understudied. We introduce AstroAlertBench, a comprehensive multimodal benchmark designed to evaluate LLM performance in astronomical event review along a three-stage logical chain: metadata grounding, scientific reasoning, and hierarchical classification over five categories. We use a pilot sample of 1,500 real-world alerts from the Zwicky Transient Facility (ZTF), a wide-field survey that scans the northern sky to detect transient astronomical events. On this dataset, we benchmark 13 frontier closed-source and open-weight LLMs that support visual input. Our results reveal that high accuracy does not always align with model ``honesty,'' defined as the ability to self-evaluate its reasoning, which affects its reliability as a real-world assistant. We further initialize a human-in-the-loop evaluation protocol as a precursor to future community-scale participation. Together, AstroAlertBench provides a framework for developing calibrated and interpretable astronomical assistants. 

---
# MOSAIC: Module Discovery via Sparse Additive Identifiable Causal Learning for Scientific Time Series 

**Authors**: Shicheng Fan, Nour Elhendawy, Jianle Sun, Ke Fang, Kun Zhang, Yihang Wang, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.05524)  

**Abstract**: Causal representation learning (CRL) seeks to recover latent variables with identifiability guarantees, typically up to permutation and component-wise reparameterization under appropriate assumptions. However, identifiability does not imply interpretability: latent semantics are typically assigned post hoc by alignment with known ground-truth factors. This limitation is particularly acute in scientific time series, where underlying mechanisms are unknown and discovering interpretable structure is a primary goal. In contrast, scientific observations (such as residue-pair distances, climate indices, or process sensors) are inherently semantic, as they correspond to named physical quantities. This raises a key question: can the interpretability of observations be transferred to the identifiable latent space? We propose MOSAIC (Module discovery via Sparse Additive Identifiable Causal learning), a sparse temporal VAE that integrates temporal CRL identifiability with support recovery over observed variables. MOSAIC identifies latent variables via regime-conditioned temporal variation, and recovers for each latent a sparse set of associated observations through an additive decoder, yielding module-level interpretability. We show that ANOVA main-effect supports are identifiable under general smooth mixing functions, and provide finite-sample recovery guarantees for a tractable sparse-additive variant. Empirically, MOSAIC recovers domain-consistent variable groups across RNA molecular dynamics, solar wind, ENSO climate, the Tennessee Eastman process, and a synthetic tokamak benchmark, enabling interpretable discovery of latent mechanisms in scientific time series. 

---
# When Semantic Communication Meets Queueing: Cross-Layer Latency and Task Fidelity Optimization 

**Authors**: Yalin E. Sagduyu, Tugba Erpek  

**Link**: [PDF](https://arxiv.org/pdf/2605.05514)  

**Abstract**: Semantic communication (SemCom) with learned encoder-decoder architectures enables end-to-end learning of compact task-oriented representations optimized for the wireless channel, reducing channel resources needed to convey task-relevant information and improving spectrum efficiency. This paper studies semantic image transmission over block Rayleigh fading with AWGN using a multi-task semantic autoencoder that jointly reconstructs images and predicts labels from the received waveform. The latent dimension (complex channel uses per source sample) serves as a cross-layer control variable governing semantic fidelity and channel resource usage. We characterize the resulting latency-task fidelity tradeoff: larger latent representations improve inference accuracy but increase service time, channel uses, and queueing delay. Building on this insight, we develop online semantic-rate controllers that adapt the latent dimension per update under a long-term semantic error constraint. A queue-aware drift-plus-penalty policy minimizes delay subject to an average semantic error cap, while a complementary age-aware policy minimizes time-average Age of Information (AoI). By adapting the semantic rate to congestion and fidelity requirements, the proposed framework improves spectrum utilization and enables timely semantic updates with significantly lower delay and AoI than fixed-rate baselines. 

---
# ReaComp: Compiling LLM Reasoning into Symbolic Solvers for Efficient Program Synthesis 

**Authors**: Atharva Naik, Yash Mathur, Prakam, Carolyn Rose, David Mortensen  

**Link**: [PDF](https://arxiv.org/pdf/2605.05485)  

**Abstract**: LLMs can solve program synthesis tasks but remain inefficient and unreliable on hard instances requiring large combinatorial search. Given a small set of reasoning traces, we use coding agents to compile them into reusable symbolic program synthesizers over constrained DSLs. The resulting solvers require no LLM calls at test time and are strong standalone systems: symbolic solver ensembles reach 91.3% accuracy on PBEBench-Lite and 84.7% on PBEBench-Hard, outperforming LLMs with test-time scaling for the latter by +16.3 percentage points at zero LLM inference cost. They also complement LLM search, improving PBEBench-Hard accuracy from 68.4% to 85.8% while reducing reported token usage by 78%, and raising SLR-Bench hard-tier accuracy from 34.4% to 58.0% in a neuro-symbolic hybrid setting. Compared to directly using coding agents as per-instance solvers, induced solvers are substantially more Pareto-efficient, amortizing a small one-time construction cost over many zero-token executions. Finally, most solvers transfer zero-shot to a real historical linguistics task - predicting sound changes in natural language data - reaching 80.1% accuracy under ensembling and recovering some plausible linguistic rules. Together, these results show that reasoning traces can be compiled into reusable symbolic solvers that solve many tasks directly, complement LLM inference on hard cases, and provide a scalable route to domain-general solver induction. We release code and data for reproducibility. 

---
# GRALIS: A Unified Canonical Framework for Linear Attribution Methods via Riesz Representation 

**Authors**: Raimondo Fanale  

**Link**: [PDF](https://arxiv.org/pdf/2605.05480)  

**Abstract**: The main XAI attribution methods for deep neural networks -- GradCAM, SHAP, LIME, Integrated Gradients -- operate on separate theoretical foundations and are not formally comparable. We present GRALIS (Gradient-Riesz Averaged Locally-Integrated Shapley), a mathematical framework establishing a representation theory for attributions: every additive, linear, and continuous attribution functional on L^2(Q,mu) admits a unique canonical representation (Q, w, Delta), proved necessary by the Riesz Representation Theorem. This class encompasses SHAP, IG, LIME and linearized GradCAM, but excludes nonlinear functionals such as standard GradCAM or attention maps. Seven formal theorems provide simultaneous guarantees absent in any individual method: (T1) necessary canonical form; (T2) exact completeness; (T3) Monte Carlo convergence O(1/sqrt(m))+O(1/k); (T4) exact Shapley Interaction Values; (T5) Hoeffding ANOVA decomposition; (T6) Sobol sensitivity generalization; (T7) multi-scale extension (MS-GRALIS) with minimum-variance weights. An algebraic appendix justifies the GRALIS-SIV correspondence via the Mobius transform without circularity. GRALIS satisfies 13.5/14 axiomatic properties vs. 2.5-6/14 for individual methods, including completeness, sensitivity, locality, order-k interactions and optimal multi-scale aggregation simultaneously. Preliminary validation on BreaKHis (1,187 histology images, DenseNet-121) reports deletion faithfulness AUC +0.015 (malignant), 96% class-conditional consistency, SAL = 0.762+/-0.109 and sparsity index 0.39. Extended comparison with baseline XAI methods is planned for a companion paper. 

---
# A Unified Benchmark for Evaluating Knowledge Graph Construction Methods and Graph Neural Networks 

**Authors**: Othmane Kabal, Mounira Harzallah, Fabrice Guillet, Hideaki Takeda, Ryutaro Ichise  

**Link**: [PDF](https://arxiv.org/pdf/2605.05476)  

**Abstract**: Knowledge graphs automatically constructed from text are increasingly used in real-world applications. However, their inherent noise, fragmentation, and semantic inconsistencies significantly affect the performance of Graph Neural Networks (GNNs) on downstream tasks. Assessing their performance and robustness remains difficult, as it is often unclear whether observed results stem from the learning model or from the quality of the constructed graph itself. In this work, we introduce a dual-purpose benchmark designed to jointly evaluate (i) the performance of GNNs on noisy, text-derived graphs and (ii) the effectiveness of graph construction methods on a downstream task. The benchmark is built in the biomedical domain from a single textual corpus and includes two automatically constructed graphs generated using different extraction methods, alongside a high-quality reference graph curated by experts that serves as an upper performance bound. This design enables controlled comparison of construction methods and systematic evaluation of GNN robustness through semi-supervised node classification. We further provide a standardized, reproducible, and extensible evaluation framework, facilitating the integration of new graph extraction methods and learning models. 

---
# The Pedagogy of AI Mistakes: Fostering Higher-Order Thinking 

**Authors**: Hadi Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2605.05472)  

**Abstract**: As generative AI becomes increasingly integrated into higher education, its frequent errors and hallucinations, often seen as limitations, offer a unique pedagogical opportunity. By framing AI as a ``learning companion'' whose imperfect outputs prompt analysis, evaluation, and reflection, we argue that instructors can engage students in the fundamental processes of higher-order thinking. This paper presents a design-oriented study in which an AI-integrated syllabus in a \textit{database design} course deliberately leverages AI's limitations to foster critical thinking and higher-order cognitive skills aligned with Bloom's taxonomy of learning. Using a mixed-methods approach, we examine how structured interaction with AI-generated errors supports metacognitive engagement, reinforces disciplinary rigor, and relates to students' perceived AI literacy and subject-matter competency. 

---
# Robustness of Graph Self-Supervised Learning to Real-World Noise: A Case Study on Text-Driven Biomedical Graphs 

**Authors**: Othmane Kabal, Mounira Harzallah, Fabrice Guillet, Hideaki Takeda, Ryutaro Ichise  

**Link**: [PDF](https://arxiv.org/pdf/2605.05463)  

**Abstract**: Graph Self-Supervised Learning (GSSL) offers a powerful paradigm for learning graph representations without labeled data. However, existing work assumes clean, manually curated graphs. Recent advances in NLP enable the large-scale automatic extraction of knowledge graphs from text, opening new opportunities for GSSL while introducing substantial real-world noise. This type of noise remains largely unexplored, as prior robustness studies typically rely on synthetic perturbations. To address this gap, we present the first comprehensive evaluation of GSSL methods on text-driven graphs for unsupervised term typing. We introduce Noise-Aware Text-Driven Graph GSSL (NATD-GSSL), a unified framework that combines automatic graph construction, graph refinement, and GSSL. Our evaluation follows a dual-graph protocol that contrasts a noisy graph derived from MedMentions with a clean Unified Medical Language System (UMLS) reference graph, aligned through a shared gold standard. Our results reveal variability in robustness across both pretext tasks and Graph Neural Network (GNN) architectures. Relation reconstruction is highly sensitive to noise and benefits from well-defined schemas, whereas feature reconstruction is considerably more robust, achieving performance comparable to clean-graph settings. Contrastive objectives are generally less affected by noise but depend strongly on alignment with downstream tasks. GNN architecture also plays a critical role: bidirectional relational message-passing designs are better suited to noisy, text-driven graphs, while unidirectional relational ones perform best on clean graphs. Overall, NATD-GSSL provides practical guidance for applying GSSL to real-world, noisy graphs and achieves up to a 7\% improvement over pretrained language model baselines. All code and benchmarks are publicly available at this https URL. 

---
# SLAM: Structural Linguistic Activation Marking for Language Models 

**Authors**: Fabrice Harel-Canada, Amit Sahai  

**Link**: [PDF](https://arxiv.org/pdf/2605.05443)  

**Abstract**: LLM watermarks must be detectable without compromising text quality, yet most existing schemes bias the next-token distribution and pay for detection with measurable quality loss. We present SLAM (Structural Linguistic Activation Marking), a novel white-box watermarking scheme that sidesteps this cost by writing the mark into structural geometry rather than token frequencies: sparse autoencoders identify residual-stream directions encoding linguistic structure (e.g., voice, tense, clause order), and we causally steer those directions at generation time, leaving lexical sampling and semantics unconstrained. On Gemma-2 2B and 9B, SLAM achieves 100% detection accuracy with a quality cost of only 1-2 reward points - compared to 7.5-11.5 for KGW, EWD, and Unigram - with naturalness and diversity preserved at near-unwatermarked levels across both models. The trade-off is a complementary robustness profile: SLAM resists word-level edits but is vulnerable to paraphrase that restructures syntax (at a quality cost), the converse of token-distribution methods. 

---
# On Semantic Loss Fine-Tuning Approach for Preventing Model Collapse in Causal Reasoning 

**Authors**: Pratik Deshmukh, Atirek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2605.05438)  

**Abstract**: Standard fine-tuning of transformer models on causal reasoning tasks leads to catastrophic model collapse, where models learn trivial solutions such as always predicting "Yes" or "No" regardless of input structure. We demonstrate that fine-tuning Gemma 270M on transitivity and d-separation tasks without semantic loss results in 100% collapse rate, with models achieving misleadingly high accuracy (73.9%) while learning no causal reasoning. We propose a semantic loss function with graph-based logical constraints and dynamic lambda scheduling that prevents this collapse. Our approach achieves 70.4% accuracy on transitivity tasks and 68.6% on d-separation tasks with stable, context-dependent predictions, representing a 42.7% improvement over collapsed baselines. Adversarial evaluation on 1,000 structural reasoning samples shows semantic models achieve 67-70% accuracy while collapsed models fail catastrophically at 43-71%. We validate our findings through comprehensive benchmarking on 200,000+ evaluation samples across five model variants, demonstrating that semantic loss is essential and not optional, for stable causal reasoning in transformers. 

---
# Information Theoretic Adversarial Training of Large Language Models 

**Authors**: Yiwei Zhang, Jeremiah Birrell, Reza Ebrahimi, Rouzbeh Behnia, Jason Pacheco, Elisa Bertino  

**Link**: [PDF](https://arxiv.org/pdf/2605.05415)  

**Abstract**: Large language models (LLMs) remain vulnerable to adversarial prompting despite advances in alignment and safety, often exhibiting harmful behaviors under novel attack strategies. While adversarial training can improve robustness, existing approaches are computationally expensive and difficult to scale. Recent continuous adversarial training methods, such as Continuous adversarial training (CAT) and Continuous Adversarial Preference Optimization (CAPO), address this challenge by leveraging gradient-based perturbations in the embedding space, enabling more efficient and expressive attacks. Building on this paradigm, we propose WARDEN, a distributionally robust adversarial training framework for LLMs that dynamically reweights adversarial examples through an f -divergence ambiguity set around the empirical training distribution. Our method optimizes the worst-case adversarial loss within a divergence ball around the empirical data distribution, automatically emphasizing harder adversarial examples. Using the convex dual formulation, the objective reduces to a log-sum-exp form under the KL divergence, with a dynamical parameter controlling the strength of reweighting. This study leads to a new class of information-theoretic objectives that significantly reduce attack success rates while maintaining model utility. Across multiple LLMs and attack settings, WARDEN substantially reduces attack success rates with computational and utility costs comparable to CAT-, CAPO-, and MixAT-based baselines, making it a practical approach for scalable robust alignment. 

---
# Creative Robot Tool Use by Counterfactual Reasoning 

**Authors**: M. Tuluhan Akbulut, Varun Satheesh, Ahmed Jaafar, Alper Ahmetoglu, Shane Parr, Aditya Ganeshan, Shivam Vats, George Konidaris  

**Link**: [PDF](https://arxiv.org/pdf/2605.05411)  

**Abstract**: We propose a causal reasoning framework for creative robot tool use where a suitable tool for a task is correctly identified for use beyond its primary objectives. The proposed framework first discovers the causal relationships between the tool and the task by conducting simulated experiments in a dynamics model. We decouple the causal discovery problem into two complementary components: VLM-based feature suggestion and counterfactual tool generation via targeted geometric and physical feature perturbations. Then, novel objects are classified based on identified causal features, and the tool use skill is transferred via keypoint matching conditioned on the identified causal features. By reconstructing the task in a dynamics model, our approach grounds tool use in the physics of the problem. We illustrate our approach in reaching a distant object with different sticks, scooping candies from a bowl using diverse items, and using different boxes or crates as stepping platforms to retrieve an object from a high shelf. Our baseline comparisons show that identifying causal features and grounding them in physical tool properties leads to more reliable tool selection and stronger skill keypoint transfer. 

---
# Mise en Place for Agentic Coding: Deliberate Preparation as Context Engineering Methodology 

**Authors**: Andrew Zigler  

**Link**: [PDF](https://arxiv.org/pdf/2605.05400)  

**Abstract**: The rapid adoption of AI coding agents has produced a dominant workflow pattern -- often called "vibe coding" -- that prioritizes speed of implementation over deliberate preparation. We argue that this approach creates a systematic alignment problem: agents that lack sufficient context produce code requiring extensive debugging and refactoring, consuming substantial development time. Drawing on the culinary concept of mise en place (everything in its place; abbreviated MEP), we propose a three-phase preparation methodology for agentic coding: (1) contextual grounding, where domain expertise and tacit knowledge are externalized into structured documents; (2) collaborative specification, where human-agent dialogue produces detailed design artifacts; and (3) task decomposition, where specifications are converted into structured, dependency-aware task records. We report on the application of MEP during a competitive hackathon, where roughly two hours of preparation enabled a rapid parallel implementation of a full-stack educational platform by concurrent AI agents. We introduce the concept of context fluency as an emerging developer skill -- the ability to create rich, structured context that agents can act on -- and connect it to established frameworks in backward design and tacit knowledge externalization. We conclude with a research agenda for empirically validating preparation-phase methodologies in AI-assisted software development. 

---
# Generating Query-Focused Summarization Datasets from Query-Free Summarization Datasets 

**Authors**: Yllias Chali, Deen Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2605.05392)  

**Abstract**: Large-scale datasets are widely used to perform summarization tasks, but they may not include queries alongside documents and summaries. In the search for suitable datasets for Query-Focused Summarization (QFS), we identify two research questions: Is it possible to automatically generate evidence-based query keywords from query-free datasets? Does evidence-based query generation support the QFS task? This paper proposes an evidence-based model to generate queries from query-free datasets. To evaluate our model intrinsically, we compare the similarity between the original queries and the system-generated queries of two QFS datasets. We also perform summarization tasks using different pre-trained models, as well as a state-of-the-art (SOTA) QFS model, to measure the extrinsic performance of our query generation approach. Experimental results indicate that summaries generated using evidence-based queries achieve competitive ROUGE scores compared to those generated from the original queries. 

---
# Two-Stage Learned Decomposition for Scalable Routing on Multigraphs 

**Authors**: Filip Rydin, Morteza Haghir Chehreghani, Balázs Kulcsár  

**Link**: [PDF](https://arxiv.org/pdf/2605.05389)  

**Abstract**: Most neural methods for Vehicle Routing Problems (VRPs) are limited to Euclidean settings or simple graphs. In this work, we instead consider multigraphs, where parallel edges represent distinct travel options with varying trade-offs (e.g., distance vs time). Few methods are designed for such formulations and those that do exist face major scalability issues. We mitigate these scalability issues via a Node-Edge Policy Factorization (NEPF) approach, which splits the routing policy into a node permutation stage and an edge selection stage. To enable the decomposition, we introduce a pre-encoding edge aggregation scheme and a non-autoregressive architecture for the edge stage, as well as a hierarchical reinforcement learning method to train the stages jointly. Our experiments across six VRP variants demonstrate that NEPF matches or outperforms the state-of-the-art in terms of solution quality, while being significantly faster in training and inference. 

---
# Two Steps Are All You Need: Efficient 3D Point Cloud Anomaly Detection with Consistency Models 

**Authors**: Pranav A, Shashank B, Pranav Siddappa, Dominik Seuss, Minal Moharir, Subramanya KN  

**Link**: [PDF](https://arxiv.org/pdf/2605.05372)  

**Abstract**: Diffusion models are rapidly redefining 3D anomaly detection in point cloud data. As 3D sensing becomes integral to modern manufacturing, reliable anomaly detection is essential for high-throughput quality assurance and process control. Yet practical deployment on resource-constrained, latency-critical systems remains limited. Existing methods are often computationally prohibitive or unreliable in complex, unmasked regions, and diffusion pipelines are inherently bottlenecked by iterative denoising. In this work, we address this bottleneck by reformulating reconstructionbased anomaly detection through consistency learning, enabling direct prediction of anomaly-free geometry in one or two network evaluations. We further introduce a novel hybrid loss formulation that explicitly enforces reconstruction toward clean data. This design substantially reduces inference cost, achieving up to 80x faster runtime than the current state-of-the-art method, without GPU acceleration, while preserving strong detection performance. It outperforms R3D-AD on Anomaly-ShapeNet with 76.20% I-AUROC and remains competitive on Real3DAD with 72.80% I-AUROC, enabling efficient, low-latency anomaly detection on resource-constrained platforms, including drones, smart industrial cameras, and other edge devices. 

---
# SPADE: Faster Drug Discovery by Learning from Sparse Data 

**Authors**: Rahul Nandakumar, Ben Fauber, Deepayan Chakrabarti  

**Link**: [PDF](https://arxiv.org/pdf/2605.05370)  

**Abstract**: Drug discovery seeks molecules (ligands) that bind strongly and selectively to a target protein. However, fewer than 5% of candidate ligands pass the bar for even the early stages of drug discovery. Furthermore, we want methods that work for novel proteins for which we have no prior data. Starting from scratch, we have to iteratively select and test candidate ligands such that we find enough ligands of the desired quality in as few tests as possible. Our proposed algorithm, named SPADE, introduces a novel approach to ligand selection that requires only 40 tests on average to find 10 high-quality ligands. In one-vs-one comparisons, SPADE outperforms deep learning and Bayesian optimization methods on more proteins, achieving median improvements of 7%-32% in sample efficiency. SPADE is also 10x faster than its closest competitor at scoring candidate drugs. Dataset and code is available at this https URL 

---
# Towards an Inferentialist Account of Information Through Proof-theoretic Semantics 

**Authors**: Matthew Collins, Timo Eckhardt, David Pym  

**Link**: [PDF](https://arxiv.org/pdf/2605.05368)  

**Abstract**: Information is one of the most widely-discussed concepts of the current era. However, a great deal of insightful work notwithstanding, it is yet to be given wholly convincing logical or mathematical foundations. Without them, we lack adequate reasoning tools for understanding the complex ecosystems of systems upon which the society depends. We seek to rectify this by taking a first step towards developing an inferentialist semantic theory of information. There are three key interacting components. First, conceptual analysis: the metaphysics of information. Dretske expressed the key concepts of information in terms of intentionality, truth, and transmissibility. We replace truth with inferability, and trace the consequences of this replacement. Second, logic: proof-theoretic semantics (P-tS) provides a mathematical-logical realization of inferentialist reasoning. Using P-tS, we develop the first steps towards a mathematical-logical theory of an inferentialist primitive unit of information, the 'inferon'. This proof-theoretic approach counterpoints the model-theoretic view of information articulated in situation theory. Furthermore, we argue that it facilitates addressing all three components of van Benthem and Martinez's categorization of the understandings of information, as range, as correlation, and as code. Our focus is on information-as-correlation. Third, systems: the P-tS tools we develop provide the basis for a mathematical account of distributed systems modelling -- a key tool from informatics for understanding the organization of information processing systems. This yields a reasoning-based theory of information flow in models of distributed systems. Overall, we seek to give a conceptually rigorous mathematical-logical account of information and its role within informatics, grounded in inference and reasoning. 

---
# Tamaththul3D: High-Fidelity 3D Saudi Sign Language Avatars from Monocular Video 

**Authors**: Eyad Alghamdi, Sattam Altuuaim, Obay Ghulam, Abdulrahman Qutah, Yousef Basoodan  

**Link**: [PDF](https://arxiv.org/pdf/2605.05367)  

**Abstract**: Arabic Sign Language (ArSL) and its dialects serve approximately 400 million Arabic speakers worldwide, yet the community lacks high-quality 3D parametric annotations and specialized reconstruction methods for avatar generation. We address this critical gap through two key contributions: First, we introduce the first high-quality 3D parametric annotations for the Ishara-500 Saudi Sign Language dataset, providing precise SMPL-X parameters for 500 culturally authentic SSL signs. Second, we present Tamaththul3D, a specialized reconstruction pipeline designed for ArSL's unique articulation patterns. Our pipeline integrates SMPLer-X for robust body estimation, WiLoR for detailed hand refinement with automatic localization and mirroring, and MediaPipe for 2D pose supervision. Through kinematic-chain-based wrist alignment with hybrid swing-twist decomposition and 2D-supervised joint optimization, Tamaththul3D achieves state-of-the-art hand accuracy (up to 32% improvement over previous methods) while maintaining competitive body pose. Together, these 3D annotations and Tamaththul3D pipeline establish the first comprehensive framework for high-fidelity ArSL avatar reconstruction, enabling new accessibility technologies and cultural preservation efforts for the Arab Deaf community. 

---
# COPYCOP: Ownership Verification for Graph Neural Networks 

**Authors**: Rahul Nandakumar, Deepayan Chakrabarti  

**Link**: [PDF](https://arxiv.org/pdf/2605.05360)  

**Abstract**: Given two GNNs that output node embeddings, how can we determine if they were trained independently? An adversary could have trained one GNN specifically to mimic the other GNN's embeddings. To obscure this relationship between the GNNs, the adversarial GNN might then transform its output embeddings. The two GNNs could have different architectures, weights, and embedding dimensions, and the adversary can transform the embeddings. Despite these stringent conditions, our algorithm (named CopyCop) can identify such copycat GNNs, unlike existing watermarking and fingerprinting methods. We also provide theoretical guarantees for CopyCop. Finally, experiments on 14 datasets and 5 GNN architectures demonstrate that CopyCop is accurate and robust against a broad class of adversarial attacks and transformations. Code is available at: this https URL 

---
# Counterargument for Critical Thinking as Judged by AI and Humans 

**Authors**: Tosin Adewumi, Marcus Liwicki, Foteini Simistira Liwicki, Lama Alkhaled, Hamam Mokayed, Esra Sümer-Arpak  

**Link**: [PDF](https://arxiv.org/pdf/2605.05353)  

**Abstract**: This intervention study investigates the use of counterarguments in writing for critical thinking by students in the context of Generative AI (GenAI). This is especially as risks of cheating and cognitive offloading exist with the use of GenAI. We presented 36 students in a particular university course with 4 carefully selected thesis statements (from a set of popular debates) to write about anyone of them. We used six established rubrics (focus, logic, content, style, correctness and reference) to conduct three human assessments (two student peer-reviews and one experienced teacher) per writeup on a 5-point Likert scale for all the qualified samples (n) of 35 submissions (after disqualifying one for irregularity). Using the same rubrics and guidelines, we also assessed the submissions using six frontier LLMs as judges. Our mixed-method design included qualitative open-ended feedback per assessment and quantitative methods. The results reveal that (1) the students' self-written counterarguments to AI-generated content contains logic, among other things, which is a key component of critical thinking, and (2) GenAI can be successfully used at scale to assess students' written work, based on clear rubrics, and these assessments generally align with human assessments as shown with Gwets AC2 inter-rater reliability values of 0.33 for all the models except one. 

---
# Making AI Drafts Count: A Quality Threshold in Audio Description Workflows 

**Authors**: Lana Do, Shasta Ihorn, Charity M. Pitcher-Cooper, Sanjay Mirani, Gio Jung, Hyunjoo Shim, Zhenzhen Qin, Kien T. Nguyen, Vassilis Athitsos, Ilmi Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2605.05348)  

**Abstract**: Audio description (AD) narrates visual elements in video for blind and low-vision audiences. Recent work has shown that giving novice describers an AI-generated draft to start from helps produce higher-quality AD and lowers the barrier to entry. What remains an open question is how draft quality shapes the editing process. We investigate this through GenAD, an AD generation pipeline that incorporates accessibility guidelines and contextual video information, and RefineAD, an editing interface for human revisions. Human-AI contributions are measured across text, timing, and delivery. In a within-subjects study, we compared authoring from scratch against editing AI drafts of varying quality. GenAD drafts cut completion time by more than half and significantly reduced cognitive load. In contrast, baseline drafts generated from simple, unguided prompts offered only modest benefits, pointing to a minimum quality threshold for effectiveness. Qualitative findings suggest this threshold is content-dependent; as visual complexity increases, so does the quality needed from AI drafts. We propose this as a design principle: effective AI assistance should clear a quality threshold suited to the target content, rather than simply be present. 

---
# Open-SAT: LLM-Guided Query Embedding Refinement for Open-Vocabulary Object Retrieval in Satellite Imagery 

**Authors**: Md Adnan Arefeen, Biplob Debnath, Ravi K. Rajendran, Murugan Sankaradas, Srimat T. Chakradhar  

**Link**: [PDF](https://arxiv.org/pdf/2605.05344)  

**Abstract**: In satellite applications, user queries often take the form of open-ended natural language, extending beyond a fixed set of predefined categories. This open-vocabulary nature poses significant challenges for retrieving relevant image tiles, as the retrieval system must generalize to a wide range of unseen objects and concepts. While vision-language models (VLMs) such as CLIP are widely used for text-image retrieval, even fine-tuned variants often struggle to accurately align such queries with satellite imagery. To address this, we propose Open-SAT, a training-free query embedding refinement algorithm that operates at inference time to improve alignment between user queries and satellite image content. Open-SAT uses VLMs to compute embeddings for image tiles, which are stored in a vector database for efficient retrieval. At query time, it leverages Large Language Models (LLMs) to refine the text embeddings by incorporating contextual information about objects of interest and their surroundings. A threshold-free retrieval mechanism further enhances accuracy and efficiency. Experimental results in three public benchmarks demonstrate that Open-SAT improves the F1 score by up to 16.04%, while retrieving a comparable number of image tiles. These results demonstrate the effectiveness of Open-SAT in open-vocabulary satellite image retrieval, leveraging LLM guidance without the need for additional training or supervision. 

---
# Feature Starvation as Geometric Instability in Sparse Autoencoders 

**Authors**: Faris Chaudhry, Keisuke Yano, Anthea Monod  

**Link**: [PDF](https://arxiv.org/pdf/2605.05341)  

**Abstract**: Sparse autoencoders (SAEs) are used to disentangle the dense, polysemantic internal representations of large language models (LLMs) into interpretable, monosemantic concepts. However, standard $\ell_1$-regularized SAEs suffer from feature starvation (dead neurons) and shrinkage bias, often requiring computationally expensive heuristic resampling and nondifferentiable hard-masking methods to bypass these challenges. We argue that feature starvation is not merely an empirical artifact of poor data diversity, but a fundamental optimization-geometric pathology of overcomplete dictionaries: the $\ell_1$-induced sparse coding map is unstable and fundamentally misaligned with shallow, amortized encoders. To address this structural instability, we introduce adaptive elastic net SAEs (AEN-SAEs), a fully differentiable architecture grounded in classical sparse regression. AEN-SAEs combine an $\ell_2$ structural term that enforces strong convexity and Lipschitz stability with adaptive $\ell_1$ reweighting that eliminates shrinkage bias and suppresses spurious features, thereby jointly controlling the curvature and interaction structure of the induced polyhedral geometry. Theoretically, we show that AEN-SAEs yield a Lipschitz-continuous sparse coding map and recover the global feature support under mild assumptions. Empirically, across synthetic settings and LLMs (Pythia 70M, Llama 3.1 8B), AEN-SAEs mitigate feature starvation without auxiliary heuristics while maintaining competitive reconstruction abilities. 

---
# How Far Are VLMs from Privacy Awareness in the Physical World? An Empirical Study 

**Authors**: Junran Wang, Xinjie Shen, Zehao Jin, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.05340)  

**Abstract**: As Vision-Language Models (VLMs) are increasingly deployed as autonomous cognitive cores for embodied assistants, evaluating their privacy awareness in physical environments becomes critical. Unlike digital chatbots, these agents operate in intimate spaces, such as homes and hospitals, where they possess the physical agency to observe and manipulate privacy-sensitive information and artifacts. However, current benchmarks remain limited to unimodal, text-based representations that cannot capture the demands of real-world settings. To bridge this gap, we present ImmersedPrivacy, an interactive audio-visual evaluation framework that simulates realistic physical environments using a Unity-based simulator. ImmersedPrivacy evaluates physically grounded privacy awareness across three progressive tiers that test a model's ability to identify sensitive items in cluttered scenes, adapt to shifting social contexts, and resolve conflicts between explicit commands and inferred privacy constraints. Our evaluation of 12 state-of-the-art models reveals consistent deficits. In cluttered scenes, all models exhibit monotonic performance decay as scene complexity grows due to perceptual deficit. When social context shifts, no model exceed 65% selection accuracy. Under conflicting commands, the best model gemini-3.1-pro perfectly balances task completion and privacy preservation in only 51% of cases. These findings reveal that current VLMs in the physical world suffer from perceptual fragility and fail to let their knowledge of privacy cues govern their situated behavior. Our code and data is available at this https URL . 

---
# ViTok-v2: Scaling Native Resolution Auto-Encoders to 5 Billion Parameters 

**Authors**: Philippe Hansen-Estruch, Jiahui Chen, Vivek Ramanujan, Orr Zohar, Yan Ping, Animesh Sinha, Markos Georgopoulos, Edgar Schoenfeld, Ji Hou, Felix Juefei-Xu, Sriram Vishwanath, Ali Thabet  

**Link**: [PDF](https://arxiv.org/pdf/2605.05331)  

**Abstract**: Vision Transformer (ViT) autoencoders have emerged as compelling tokenizers for images, offering improved reconstruction over convolutional tokenizers. However, existing ViT tokenizers cannot explore this landscape as performance degrades outside training resolutions, and reliance on adversarial losses prevents stable scaling. ViTok (Hansen-Estruch et al., 2025) found that the compression ratio r mediates a reconstruction-generation trade-off where lower r means better reconstructions but harder generations, so improving tokenizer reconstruction is key to more Pareto-optimal tokenizers. We introduce ViTok-v2, which addresses these limitations with native resolution support via NaFlex for generalization across resolutions and aspect ratios, and a novel DINOv3 perceptual loss that replaces both LPIPS and GAN objectives for stable training at any scale. ViTok-v2 is trained on about 2B images and scaled to 5B parameters, the largest image autoencoder to date. ViTok-v2 matches or exceeds state-of-the-art reconstruction at 256p and outperforms all baselines at 512p and above. In joint scaling experiments with flow matching generators, we show that scaling both the autoencoder and the generator advances the Pareto frontier of this trade-off. 

---
# Graph Normalization: Fast Binarizing Dynamics for Differentiable MWIS 

**Authors**: Laurent Guigues  

**Link**: [PDF](https://arxiv.org/pdf/2605.05330)  

**Abstract**: We introduce Graph Normalization (GN), a principled dynamical system on graphs that serves as a differentiable approximation engine for the NP-hard Maximum Weight Independent Set (MWIS) problem. MWIS encompasses many combinatorial challenges, including optimal assignment, scheduling, set packing, and MAP inference in discrete Markov Random Fields. Unlike Belief Propagation, we prove GN always converges to a binary indicator of a Maximum Independent Set. GN realizes a fast quasi-Newton descent through an exact Majorization-Minimization step, systematically improving the MWIS relaxed primal objective. We establish an equivalence between GN and the Replicator Dynamics of a nonlinear evolutionary game, where vertices compete for inclusion in an independent set. While a non-potential game, the GN game follows Fisher's Fundamental Theorem of Natural Selection, where the average fitness equals the MWIS primal objective and strictly increases. This connection leads to a weighted extension of the Motzkin-Straus theorem, showing MISes are in bijection with the local minima of a quadratic form over a tilted simplex. For the Assignment Problem, GN acts as a variant of the Sinkhorn algorithm that naturally converges to a hard assignment while generalizing to arbitrary constraint graphs. We demonstrate GN's performance as a fast binarization engine for the state-of-the-art Bregman-Sinkhorn relaxed MWIS solver. On real-world benchmarks with up to 1M edges, GN identifies solutions within 1% of the best known results in seconds on a CPU. GN opens new avenues for deep learning architectures requiring differentiable, "hard" decisions under constraints, with applications in structured sparse attention, dynamic network pruning, and Mixture-of-Experts. Beyond core AI, the GN framework enables end-to-end learning of constrained optimization in computer vision, computational biology, and resource allocation. 

---
# Securing the Agent: Vendor-Neutral, Multitenant Enterprise Retrieval and Tool Use 

**Authors**: Francisco Javier Arceo, Varsha Prasad Narsing  

**Link**: [PDF](https://arxiv.org/pdf/2605.05287)  

**Abstract**: Retrieval-Augmented Generation (RAG) and agentic AI systems are increasingly prevalent in enterprise AI deployments. However, real enterprise environments introduce challenges largely absent from academic treatments and consumer-facing APIs: multiple tenants with heterogeneous data, strict access-control requirements, regulatory compliance, and cost pressures that demand shared infrastructure.
A fundamental problem underlies existing RAG architectures in these settings: retrieval systems rank documents by relevance--whether through semantic similarity, keyword matching, or hybrid approaches--not by authorization, so a query from one tenant can surface another tenant's confidential data simply because it scores highest. We formalize this gap and analyze additional shortcomings--including tool-mediated disclosure, context accumulation across turns, and client-side orchestration bypass--that arise when agentic systems conflate relevance with authorization. To address these challenges, we introduce a layered isolation architecture combining policy-aware ingestion, retrieval-time gating, and shared inference, enforced through server-side agentic orchestration. This approach centralizes security-critical operations--tool execution authorization, state isolation, and policy enforcement--on the server, creating natural enforcement points for multitenant isolation while allowing client-side frameworks to retain control over agent composition and latency-sensitive operations.
We validate the proposed architecture through an open-source implementation in OGX, a vendor-neutral framework that implements an OpenAI-compatible, open-source Responses API with server-side multi-turn orchestration. We evaluate it empirically and show that ABAC gating eliminates cross-tenant leakage while introducing negligible overhead. 

---
# Shattering the Echo Chamber: Hidden Safeguards in Manuscripts Against the AI Takeover of Peer Review 

**Authors**: Oubo Ma, Ruixiao Lin, Jiahao Chen, Yuan Su, Yong Yang, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.05271)  

**Abstract**: As LLMs become increasingly capable, editorial boards and program committees are growing concerned about reviewers who fully outsource peer review to commercial chatbots. This concern stems from prior findings that current chatbots lack the independent critical thinking and depth of reasoning required to assess scientific novelty. One promising direction for mitigating this concern is to embed hidden instructions into manuscripts that disrupt or alter chatbot-generated reviews. However, existing methods remain intuitive and fragile, as they typically rely on homogeneous payloads injected in an inter-stream manner, rendering them susceptible to sanitization or neutralization.
In this paper, we identify End-to-End Review Outsourcing as an emerging threat and propose IntraGuard, a black-box, venue-agnostic defense framework grounded in the structural--visual decoupling inherent to the PDF. Designed for committee-side deployment, IntraGuard supports both explicit strategies that trigger refusal or warning signals, and implicit strategies that embed predefined textual markers into the generated review. These strategies can be deployed via any of three intra-stream injection mechanisms, each of which seamlessly embeds heterogeneous defensive text objects within the PDF's underlying structure without altering its visual presentation. Extensive evaluations across 7 real-world commercial chatbot settings and 12 venues spanning diverse disciplines show that IntraGuard achieves a defense success rate of up to 84%, while preserving peer-review invariance for human reviewers. IntraGuard is lightweight and hardware-independent, incurring an average overhead of only one second per manuscript on a commodity personal computer. We further evaluate 11 adaptive attacks spanning manuscript sanitization and instruction interference, and discuss the implications of constructing ensemble defenses. 

---
# Bridging Generation and Training: A Systematic Review of Quality Issues in LLMs for Code 

**Authors**: Kaifeng He, Xiaojun Zhang, Peiliang Cai, Mingwei Liu, Yanlin Wang, Chong Wang, Kaifeng Huang, Bihuan Chen, Xin Peng, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.05267)  

**Abstract**: Large language models (LLMs) frequently generate defective outputs in code generation tasks, ranging from logical bugs to security vulnerabilities. While these generation failures are often treated as model-level limitations, empirical evidence increasingly traces their root causes to imperfections within the training corpora. Yet, the specific mechanisms linking training data quality issues to generated code quality issues remain largely unmapped. This paper presents a systematic literature review of 114 primary studies to investigate how training data quality issues propagate into code generation. We establish a unified taxonomy that categorizes generated code quality issues across nine dimensions and training data quality issues into code and non-code attributes. Based on this taxonomy, we formalize a causal framework detailing 18 typical propagation mapping mechanisms. Furthermore, we synthesize state-of-the-art detection and mitigation techniques across the data, model, and generation lifecycles. The reviewed literature reveals a clear methodological shift: quality assurance is transitioning from reactive, heuristic-based post-generation filtering toward proactive, data-centric governance and closed-loop repair. Finally, we identify open challenges and outline research directions for developing reliable LLMs for code through integrated data curation and continuous evaluation. Our repository is available at this https URL. 

---
# Maximizing Rollout Informativeness under a Fixed Budget: A Submodular View of Tree Search for Tool-Use Agentic Reinforcement Learning 

**Authors**: Yuelin Hu, Zhenbo Yu, Zhengxue Cheng, Wei Liu, Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.05262)  

**Abstract**: We formalize Rollout Informativeness under a Fixed Budget (RIFB) as the expected non-vanishing policy-gradient mass that a tool-use rollout set injects into Group Relative Policy Optimization (GRPO). We prove that any budget-agnostic independent sampler suffers a collapse rate bounded away from zero for hard prompts regardless of the budget. Motivated by this, we recast intermediate state selection as a monotone submodular maximization problem, where a greedy one-step selector enjoys a 1 minus 1/e approximation guarantee.
Our Uncertainty-aware Upper Confidence Bound (UUCB) terms arise as closed-form marginal gains of this objective. This turns the token-level entropy bonus from an empirical trick into an analytic consequence of the formulation. We present InfoTree, a training-time tree-search framework coupling UUCB with a learned Adaptive Budget Allocator (ABA) and an asynchronous Speculative Expansion scheme.
ABA rescues prompts whose initial tree is wasted on uniform outcomes, lifting the mixed-outcome ratio from 58.1 percent to 76.3 percent with less than 5 percent budget overhead. Speculative Expansion reduces wall-clock overhead from 14.3 percent to 4.8 percent by tolerating bounded staleness in UUCB scores.
Across nine benchmarks spanning math reasoning (AIME 2024 and 2025, MATH-500, OlympiadBench, USAMO), web-search agents (GAIA, HLE-100, BrowseComp-lite), and tool-rich coding and OS agents (APPS-verified, AgentBench-OS), InfoTree outperforms flat GRPO, DeepSearch, Tree-GRPO, AT2PO, CW-GRPO, and RC-GRPO. Head-to-head compositions with Tree-GRPO prefix sharing and CW-GRPO contribution weights deliver further gains, confirming that our selector operates orthogonally to rollout reuse and trajectory re-weighting. A 5 by 5 by 5 robustness grid reveals that over three quarters of the hyperparameter space lies on a performance plateau, confirming UUCB robustness. 

---
# Enhancing Cryo-EM Density Map Segmentation in Phenix for Improved Atomic Model Building 

**Authors**: Chenwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05259)  

**Abstract**: We introduce PhenixCraft, a fully automated pipeline for building atomic models from cryo-EM density maps. By integrating AlphaFold predictions, we enhance the map-segmentation step in Phenix during model building, addressing challenges posed by noise and artifacts that traditionally hinder this step. Our results demonstrate PhenixCraft's superior performance in TM-scores and sequence accuracy, significantly improving upon the limitations and inefficiencies of traditional model building using Phenix. 

---
# Career-Aware Resume Tailoring via Multi-Source Retrieval-Augmented Generation with Provenance Tracking: A Case Study 

**Authors**: Kumar Abhinav  

**Link**: [PDF](https://arxiv.org/pdf/2605.05257)  

**Abstract**: AI-assisted resume tailoring systems commonly operate on a single uploaded resume, which limits their ability to recover relevant experience omitted from the current draft and makes it difficult for users to distinguish grounded edits from model-generated suggestions. This paper presents Resume Tailor, an agentic resume-tailoring system that maintains a longitudinal career vault in a vector database and uses multi-source retrieval-augmented generation (RAG) to assemble job-specific resume content from historical resumes and structured career records. The system is implemented as a 12-node LangGraph pipeline with typed state management, hybrid semantic-lexical confidence scoring, provenance-aware fallback generation, anti-hallucination guardrails, and a conditional review loop. We report a pilot evaluation on nine job descriptions (JDs) across software engineering, data analytics, and business analysis roles using a single candidate's career history. For six JDs where the candidate held at least one prior role in the same occupational category, enabling the career vault improved Applicant Tracking System (ATS)-style fit scores by an average of 7.8 points. For two JDs requiring domain-specific expertise absent from the vault, scores decreased by an average of 8.0 points. One partially overlapping role showed a modest gain of 2 points. These results suggest that longitudinal retrieval can improve resume tailoring when relevant prior experience exists, while also highlighting the need for confidence-gated retrieval when domain overlap is weak. 

---
# Automated Population-Level Audit Assurance via AI-Based Document Intelligence 

**Authors**: Santosh Vasudevan, Velu Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2605.05252)  

**Abstract**: Audit transaction testing validates accuracy and completeness of customer-facing statements against internal systems of record. Traditional manual, sample-based review of unstructured PDF statements is labor-intensive and does not scale to millions of transactions. This paper presents an automated framework for large-scale audit transaction testing using AI-based document intelligence. The solution leverages Snowflake Document AI to extract structured data from unstructured PDF statements using a small labeled corpus (approximately 20 documents). Extracted data are reconciled against authoritative source-of-truth datasets to identify discrepancies at scale. Results are surfaced through interactive dashboards and automated reports. The framework enables population-level testing rather than sampling-based approaches, improving audit coverage and supporting continuous assurance objectives. Recent advances in document intelligence and analytics-driven audit frameworks enable scalable, near real-time risk identification and continuous assurance. 

---
# Decision-aware User Simulation Agent for Evaluating Conversational Recommender Systems 

**Authors**: Yuan-Chi Li, Li-Chi Chen, Sung-Yi Wu, Yu-Che Tsai, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.05250)  

**Abstract**: Conversational recommender systems (CRS) increasingly rely on user simulators for automated evaluation of sales agents. A key requirement for such simulators is the ability to model human decision-making. However, most existing simulation frameworks do not explicitly model the internal decision process, and LLM-based simulators often exhibit unrealistically strong information-processing capabilities, rarely exhibit the hesitation or decision deferral commonly observed in real consumer behavior, resulting in overly high acceptance probabilities. To address this limitation, we propose Hesitator, a theory-grounded user simulation framework that explicitly models human decision-making under choice overload. The framework introduces a modular Decision Module that separates utility-based item selection from overload-aware commitment decisions. Experiments across multiple user simulation frameworks, domains, sales modes, and LLM backbones show that integrating our module consistently mitigates unrealistic behaviors under increasing overload conditions. Furthermore, Hesitator reproduces established behavioral patterns from psychological economics, demonstrating its ability to model human decision behavior. 

---
# Governed Metaprogramming for Intelligent Systems: Reclassifying Eval as a Governed Effec 

**Authors**: Alan L. McCann  

**Link**: [PDF](https://arxiv.org/pdf/2605.05248)  

**Abstract**: AI systems increasingly synthesize executable structure at runtime: LLMs generate programs, agents construct workflows,self-improving systems modify their own behavior. In classical homoiconic and staged languages, the transition from coderepresentation to execution is unrestricted. eval is a language primitive, not a governed operation. We argue that ingovernedintelligent systems, this transition is an authority amplification: it converts symbolic structure into executableauthority andmust be mediated like any other effect. We present governed metaprogramming, a language design where programrepresentations(machine forms) are first-class values, form manipulation is pure computation, and materialization (the transition fromform toexecutable machine) is a governed effect subject to structural inspection. The governance system analyzes the proposedprogram'scapability requirements, policy compliance, and resource estimates before permitting execution. We formalize twojudgments: pureform evaluation (which emits no directives) and governed materialization (which emits exactly one governed directive). Weprovethree properties: purity of form manipulation, the no-bypass theorem, and boundary preservation. We implement the designinMashinTalk, a DSL for AI workflows compiling to BEAM bytecode, and report on integration with 454 existingmachine-checked Rocqtheorems. The central contribution is reclassifying eval from a language primitive into a governed effect. 

---
# Memory-Efficient EDA Denoising via Knowledge Distillation for Wearable IoT Under Severe Motion Artifacts and Underwater Conditions 

**Authors**: Yongbin Lee, Andrew Peitzsch, Youngsun Kong, Jarod Zizza, Dong-hee Kang, Farnoush Baghestani, Ki H. Chon  

**Link**: [PDF](https://arxiv.org/pdf/2605.05246)  

**Abstract**: Electrodermal activity (EDA) is widely used in wearable Internet of Medical Things (IoMT) systems for continuous health monitoring, including autonomic assessment. However, EDA signals are highly vulnerable to motion artifacts and environmental noise, limiting reliable deployment in harsh operating conditions such as underwater. This study proposes a robust, deployable EDA denoising framework that generalizes across multiple measurement locations and harsh environments. The framework integrates a hybrid CNN-Transformer teacher model with a lightweight depth-wise separable CNN student model via a knowledge distillation (KD) strategy. To further improve robustness, a realistic data augmentation scheme is introduced to simulate diverse motion artifacts and environmental distortions. The KD-based student model significantly reduces model size (7.87 MB to 0.51 MB) and computational cost (105.1M to 11.61M FLOPs) while maintaining denoising performance (MAE: 0.144, SNR improvement: 12.08 dB) using the public dataset validation. In real-world underwater conditions (UMAC dataset) testing, the proposed method substantially improves skin conductance response reconstruction, reducing mean absolute error from 2.809 to 0.215. Furthermore, on independent testing using the CNS-OT dataset, the denoised signals enhanced downstream CNS-OT prediction performance, achieving the highest AUROC (0.806) compared to prior denoising methods. The proposed method also improved the early prediction rate (sensitivity) from 0.550 to 0.767, enabling CNS-OT prediction up to a median of 6.9 minutes before symptom onset. These results demonstrate that the proposed framework not only improves EDA signal quality but also enhances clinically relevant prediction performance while remaining suitable for deployment in resource-constrained wearable Internet of Things systems operating in harsh environments. 

---
# Towards Dependable Retrieval-Augmented Generation Using Factual Confidence Prediction 

**Authors**: Florian Geissler, Francesco Carella, Laura Fieback, Jakob Spiegelberg  

**Link**: [PDF](https://arxiv.org/pdf/2605.05244)  

**Abstract**: Incorporating specific knowledge into large language models via retrieval-augmented generation (RAG) is a widespread technique that fuels many of today's industry AI applications. A fundamental problem is to assess if the context retrieved by some similarity search provides indeed supporting facts, or instead misguides the generator with irrelevant information. It is critical to associate meaningful confidence measures about the factuality of the retrieval process with the generated answers. We present a new, two-staged approach to predict fact faithfulness of the output of retrieval-augmented generations. First, we employ conformal prediction to select only those retrieved chunks who have a high chance to come from the correct source. This approach in itself can improve answer quality by up to 6% in some of the studied datasets, however, the associated statistical guarantees do not hold generally, since the assumption of sample exchangeability depends on the retriever setup. We present diagnostic metrics to assess whether a setup is suitable. Second, we quantify confidence in the consistency of a generated final answer with a given retrieved context, using an attention-based factuality classifier. This approach can detect inconsistent answers with a chance of up to 77%. Our work helps to establish a novel type of certified RAG systems for a broad range of natural language industry applications. 

---
# Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction 

**Authors**: Zhuofeng Li, Haoxiang Zhang, Cong Wei, Pan Lu, Ping Nie, Yi Lu, Yuyang Bai, Shangbin Feng, Hangxiao Zhu, Ming Zhong, Yuyu Zhang, Jianwen Xie, Yejin Choi, James Zou, Jiawei Han, Wenhu Chen, Jimmy Lin, Dongfu Jiang, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05242)  

**Abstract**: Modern retrieval systems, whether lexical or semantic, expose a corpus through a fixed similarity interface that compresses access into a single top-k retrieval step before reasoning. This abstraction is efficient, but for agentic search, it becomes a bottleneck: exact lexical constraints, sparse clue conjunctions, local context checks, and multi-step hypothesis refinement are difficult to implement by calling a conventional off-the-shelf retriever, and evidence filtered out early cannot be recovered by stronger downstream reasoning. Agentic tasks further exacerbate this limitation because they require agents to orchestrate multiple steps, including discovering intermediate entities, combining weak clues, and revising the plan after observing partial evidence. To tackle the limitation, we study direct corpus interaction (DCI), where an agent searches the raw corpus directly with general-purpose terminal tools (e.g., grep, file reads, shell commands, lightweight scripts), without any embedding model, vector index, or retrieval API. This approach requires no offline indexing and adapts naturally to evolving local corpora. Across IR benchmarks and end-to-end agentic search tasks, this simple setup substantially outperforms strong sparse, dense, and reranking baselines on several BRIGHT and BEIR datasets, and attains strong accuracy on BrowseComp-Plus and multi-hop QA without relying on any conventional semantic retriever. Our results indicate that as language agents become stronger, retrieval quality depends not only on reasoning ability but also on the resolution of the interface through which the model interacts with the corpus, with which DCI opens a broader interface-design space for agentic search. 

---
# PPO-Based Dynamic Positioning of HAPS-BS in Wind-Disturbed Stratospheric Maritime Networks 

**Authors**: Azim Akhtarshenas, German Svistunov, Matteo Bernabè, Kuangyu Zheng, David López-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2605.05240)  

**Abstract**: High-Altitude Platform Stations (HAPS) offer a promising solution for wide-area wireless coverage in maritime regions lacking terrestrial infrastructure. However, maintaining reliable performance is challenging due to dynamic ship mobility and atmospheric disturbances, particularly stratospheric wind effects on HAPS positioning. This paper proposes a deep reinforcement learning (DRL)-based framework for dynamic positioning of wind-disturbed HAPS-mounted base stations in maritime networks. A centralized DRL agent deployed on a coordinator HAPS controls multiple serving HAPS using radio measurements and network feedback, capturing realistic channel conditions and user mobility. A Proximal Policy Optimization (PPO) algorithm is employed to learn robust positioning policies that enhance coverage stability and system throughput under wind disturbances. Simulation results show that the proposed approach effectively mitigates wind-induced positioning deviations while ensuring reliable wide-area connectivity for maritime users. 

---
# Topology-Driven Anti-Entanglement Control for Soft Robots 

**Authors**: Haoyang Le, Shengxuan Wang, Mohan Chen, Shuo Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.05236)  

**Abstract**: In the field of precision manufacturing in complex constrained environments, the role of soft robots is increasingly prominent, and the realization of anti-winding control based on multi-intelligent body reinforcement learning has become a research hotspot. One of the core problems at present is to coordinate multiple robots to complete the unwinding operation in a highly constrained environment. The existing distributed training framework faces some observability challenges in high-density barrier and unstable environments, resulting in poor learning results. This paper proposes a topology-driven Multi-Agent Reinforcement Learning (TD-MARL) framework to coordinate multi-robot systems to avoid entanglement. Specifically, the critical network adopts centralized learning, so that each intelligent body can perceive the strategies of other intelligent bodies by sharing the topological state, thus alleviating the training instability caused by complex interactions; eliminating the demand for communication resources between robots through distributed execution, Upgrade system reliability; the integrated topological security layer uses topological invariants to accurately assess and mitigate the risk of entanglement to avoid the strategy from falling into local difficulties. Finally, the full simulation experiments carried out in the real simulation environment show that the method is better than the current advanced deep reinforcement learning (DRL) method in terms of convergence and anti-winding effect. 

---
# Evolutionary fine tuning of quantized convolution-based deep learning models 

**Authors**: Marcin Pietroń  

**Link**: [PDF](https://arxiv.org/pdf/2605.05228)  

**Abstract**: Deep learning models are the most efficient models in many machine learning tasks. The main disadvantage when using them in IoT, mobile devices, independent autonomous or real-time systems is their complexity and memory size. Therefore, much research has concentrated on compression techniques of deep learning architectures. One of the most popular technique is quantization. In most of the works, the quantization is done based on the nearest neighbour quantization technique. This work focuses on improving the quantization efficiency in pretrained and quantized models. This approach has the potential to improve the final accuracy of quantized models. The main postulate of the work is that final quantization states of the network based on nearest neighbour rounding does not guarantee optimal accuracy. In the presented work, the evolution strategy is used as an optimization approach. The evolution in each iteration changes the values of the small percentage of weights. It shifts theirs values to different quantization states. The work shows that proposed evolution with an appropriate set of operators and parameters can fast improve the accuracy of the quantized models. The results are presented for popular architectures such as VGG and Resnet for image classification and detection. Additionally, simulations were carried out for the autoencoder architecture. 

---
# Rethinking Data Curation in LLM Training: Online Reweighting Offers Better Generalization than Offline Methods 

**Authors**: Wanru Zhao, Yihong Chen, Yuzhi Tang, Wentao Ma, Shengchao Hu, Shell Xu Hu, Alex Iacob, Abhinav Mehrotra, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2605.05227)  

**Abstract**: Data curation is a critical yet under-explored area in large language model (LLM) training. Existing methods, such as data selection and mixing, operate in an offline paradigm, detaching themselves from training. This separation introduces engineering overhead and makes the curation brittle: the entire pipeline must be re-run under model/task shifts. Moreover, offline methods alter data size through hard filtering or resampling, often sacrificing data diversity and harming generalization. We propose to rethink data curation as an online reweighting problem, where sample importance is dynamically adjusted during training via loss weighting rather than static pre-processing. Specifically, we introduce ADAPT (Adaptive Data reweighting for Pretraining and FineTuning), a dynamic online framework that reweights training samples with adaptive per-sample learning rates guided by similarity-based quality signals, without changing the number of training samples. Unlike offline methods that enforce a static data distribution, ADAPT acts as an implicit curriculum learner, progressively shifting focus from coarse-grained patterns to fine-grained semantic distinctions as the model evolves. Experiments on both instruction tuning and large-scale pretraining show that ADAPT consistently outperforms offline selection/mixing and prior online methods, achieving stronger cross-benchmark generalization under equal FLOPs. 

---
# Internalizing Outcome Supervision into Process Supervision: A New Paradigm for Reinforcement Learning for Reasoning 

**Authors**: Fei Ding, Yongkang Zhang, Runhao Liu, Yuhao Liao, Zijian Zeng, Sibo wang, Huiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05226)  

**Abstract**: The central challenge of reinforcement learning for reasoning lies not only in the sparsity of outcome-level supervision, but more fundamentally in how to transform feedback provided only at the end of a sequence into fine-grained learning signals that can guide intermediate reasoning steps. Existing approaches either rely on outcome-level rewards for sequence-level optimization, which makes precise credit assignment difficult, or depend on externally constructed process supervision, which is costly and difficult to scale sustainably. To address this, we propose a new perspective: reinforcement learning for reasoning can be understood as the problem of internalizing outcome supervision into process supervision. From this perspective, we introduce a supervision-internalization method for reinforcement learning for reasoning, enabling the model to automatically extract process-level learning signals through identifying, correcting, and reusing failed reasoning trajectories, thereby achieving finer-grained policy optimization under outcome-only supervision. We further abstract this idea into a new training paradigm, in which the model continually generates and refines its own internal process supervision during reinforcement learning, opening a new path for fine-grained credit assignment in reinforcement learning for reasoning that differs from externally provided process supervision. 

---
# MACS: Modality-Aware Capacity Scaling for Efficient Multimodal MoE Inference 

**Authors**: Bo Li, Chuan Wu, shaolin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05225)  

**Abstract**: Mixture-of-Experts Multimodal Large Language Models (MoE MLLMs) suffer from a significant efficiency bottleneck during Expert Parallelism (EP) inference due to the straggler effect. This issue is worsened in the multimodal context, as existing token-count-based load balancing methods fail to address two unique challenges: (1) Information Heterogeneity, where numerous redundant visual tokens are treated equally to semantically critical ones, and (2) Modality Dynamics, where varying visual to text ratios across tasks lead to resource misallocation. To address these challenges, we propose MACS (Modality-Aware Capacity Scaling), a training-free inference framework. Specifically, MACS introduces an Entropy-Weighted Load mechanism to quantify the semantic value of visual tokens, addressing information heterogeneity. Additionally, the Dynamic Modality-Adaptive Capacity mechanism allocates expert resources based on the real-time modal composition of the input. Extensive experiments demonstrate that MACS significantly outperforms existing methods on various multimodal benchmarks, providing a novel and robust solution for the efficient deployment of MoE MLLMs in EP inference. 

---
# Channel-Level Semantic Perturbations: Unlearnable Examples for Diverse Training Paradigms 

**Authors**: Bo Wang, Jia Ni, Mengnan Zhao, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.05224)  

**Abstract**: The unauthorized use of personal data in model training has emerged as a growing privacy threat. Unlearnable examples (UEs) address this issue by embedding imperceptible perturbations into benign examples to obstruct feature learning. However, existing studies mainly evaluate UEs under from-scratch training settings, leaving their behavior under the widely adopted pretraining-finetuning (PF) paradigm largely unexplored. In this work, we provide the first systematic investigation of unlearnable examples across diverse training paradigms. Our analysis reveals that loading and freezing pretrained weights significantly weakens the effectiveness of existing UEs methods. We further explain these findings through semantic filtering: while UEs tend to induce models to overfit non-semantic noise, thereby weakening their semantic extraction capabilities, under the PF paradigm, frozen shallow layers preserve data semantics, effectively filtering out distracting information like unlearnable noise. Guided by these insights, we propose a hierarchical deception strategy, Shallow Semantic Camouflage (SSC), that confines the generation process to a semantically valid subspace, aiming to bypass the semantic suppression introduced by pretrained weights. Extensive experiments demonstrate that our method consistently preserves data unlearnability even under challenging training paradigms, such as shallow-layer freezing and semantic-focused pretraining (SF-Pretrain), bridging the critical gap in pretrain-based unlearnable learning. 

---
# Structural Instability of Feature Composition 

**Authors**: Yunpeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.05223)  

**Abstract**: Sparse Autoencoders (SAEs) have emerged as a powerful paradigm for disentangling feature superposition in transformer-based architectures, enabling precise control via activation steering. However, the theoretical foundations of compositional steering -- the simultaneous activation of distinct semantic latents -- remain under-explored. The prevailing Linear Representation Hypothesis often abstracts away non-linear interference effects that arise in overcomplete dictionaries. We present a geometric framework for analyzing the instability of feature unions. Modeling the activation space as a high-dimensional sparse cone manifold, we derive an asymptotic compositional-collapse threshold under a spherical dictionary model, characterized by the Gaussian mean width (statistical dimension) of the signal cone. We further show that, in the high-bias regime, ReLU rectification converts microscopic correlation-induced variance fluctuations into a systematic drift that accumulates under composition, yielding interference growth consistent with a ratchet effect. We validate the predicted scaling trends on structured semantic features extracted from CLEVR, where hierarchical correlations accelerate the transition relative to random baselines. Together, our results highlight geometric constraints on the scalability of union-based steering and motivate composition mechanisms that explicitly manage interference beyond naive linear superposition. 

---
# Adaptive Computation Depth via Learned Token Routing in Transformers 

**Authors**: Ahmed Abdelmuniem Abdalla Mohammed  

**Link**: [PDF](https://arxiv.org/pdf/2605.05222)  

**Abstract**: Standard transformer architectures apply the same number of layers to every token regardless of contextual difficulty. We present Token-Selective Attention (TSA), a learned per-token gate on residual updates between consecutive transformer blocks. Each gate is a lightweight two-layer multi-layer perceptron (MLP) that produces a continuous halting probability, making the mechanism end-to-end differentiable with 1.7% parameter overhead and no changes to the base architecture. Notably, TSA learns difficulty-proportional routing without any explicit depth pressure: even at $\lambda=0$ (no depth regularisation), the task-loss gradient alone drives the router to skip 20% of token-layer operations. On character-level language modeling, TSA saved 14-23% of token-layer operations (TLOps) across Tiny-Shakespeare and enwik8 at <0.5% quality loss. At matched efficiency, TSA achieved 0.7% lower validation loss than early exit, and the learned routing transfers directly to inference-time sparse execution for real wall-clock speedup. 

---
# MidSteer: Optimal Affine Framework for Steering Generative Models 

**Authors**: Tatiana Gaintseva, Andrew Stepanov, Ziquan Liu, Martin Benning, Gregory Slabaugh, Jiankang Deng, Ismail Elezi  

**Link**: [PDF](https://arxiv.org/pdf/2605.05220)  

**Abstract**: Steering intermediate representations has emerged as a powerful strategy for controlling generative models, particularly in post-deployment alignment and safety settings. However, despite its empirical success, it currently lacks a comprehensive theoretical framework. In this paper, we bridge this gap by formalizing the theory of concept steering. First, we establish a link between steering and affine concept erasure, proving that the standard approach for removing unwanted behaviors is a special case of LEACE (a closed-form method for affine erasure). Next, we formulate a principled theoretical framework for concept switching, LEACE-Switch, and characterize the assumptions under which it provides an optimal affine solution. Building on this analysis, we then introduce MidSteer (Minimal Disturbance concept Steering), a more general affine framework for concept manipulation that relaxes these assumptions and enables directed, minimal-disturbance transformations. We demonstrate that MidSteer performs favorably across a range of tasks, modalities, and architectures, including vision diffusion models and large language models. 

---
# Sparse Prefix Caching for Hybrid and Recurrent LLM Serving 

**Authors**: Mikhail Shirokikh, Sergey Nikolenko  

**Link**: [PDF](https://arxiv.org/pdf/2605.05219)  

**Abstract**: Prefix caching is a key latency optimization for autoregressive LLM serving, yet existing systems assume dense per-token key/value reuse. State-space models change the structure of the problem: a recurrent layer can resume from a single stored state rather than requiring the entire token history. This asymmetry opens a new design point between no reuse and dense caching: store exact recurrent states at a sparse set of checkpoint positions and, on a cache hit, resume from the deepest stored checkpoint and recompute the remaining suffix exactly.
We formalize sparse prefix caching as checkpoint placement under a distribution over overlap depths, yielding an exact O(NM) dynamic program. For use cases where requests share a non-trivial prefix (e.g. asking different questions about a single long document), we show that our method consistently improves the Pareto frontier traced by standard heuristics on real-world data. Across QuALITY and System Prompts, distribution-aware placement dominates every fixed-budget baseline on the measured layer-group Pareto frontier and matches or outperforms the strongest heuristic (block caching) while typically using substantially fewer checkpoints, with the largest gains at low checkpoint budgets where the overlap distribution is most non-uniform. The method is most relevant when many requests share a substantial but not identical prefix within a retained cache entry. It preserves exact outputs, does not change the recurrent computation itself or require new recurrent update kernels, applies to recurrent/SSM layers whose hidden state can be extracted and restored exactly, and for hybrid models can be combined with existing KV-cache compression techniques. 

---
# Horizon-Constrained Rashomon Sets for Chaotic Forecasting 

**Authors**: Gauri Kale, Rahul Vishwakarma, Holly Diamond, Ava Hedayatipour, Amin Rezaei  

**Link**: [PDF](https://arxiv.org/pdf/2605.05218)  

**Abstract**: Predictive multiplicity and chaotic dynamics represent two fundamental challenges in machine learning that have evolved independently despite their conceptual connections. We bridge this gap by introducing horizon-constrained Rashomon sets, a theoretical framework that characterizes how model multiplicity evolves with prediction horizon in chaotic systems. Unlike static prediction tasks where the Rashomon set remains fixed, chaos induces exponential divergence among initially similar models, fundamentally transforming the nature of predictive equivalence. We prove that the effective Rashomon set contracts exponentially with lead time at a rate determined by the maximum Lyapunov exponent and introduce Lyapunov-weighted metrics that provide tighter bounds on predictive disagreement. Leveraging these insights, we develop decision-aligned selection algorithms that choose among near-optimal models based on downstream utility rather than forecast accuracy alone. Extensive experiments on synthetic chaotic systems (Lorenz-96, Kuramoto-Sivashinsky) and real-world applications (wind power, traffic, weather) demonstrate that our framework improves decision quality by 18-34\% while maintaining competitive predictive performance. This work establishes the first rigorous connection between chaos theory and predictive multiplicity, providing principled guidance for deploying machine learning in safety-critical chaotic domains. 

---
# Physics-Informed Neural Networks with Learnable Loss Balancing and Transfer Learning 

**Authors**: Reza Pirayeshshirazinezhad  

**Link**: [PDF](https://arxiv.org/pdf/2605.05217)  

**Abstract**: We propose a self-supervised physics-informed neural network (PINN) framework that adaptively balances physics-based and data-driven supervision for scientific machine learning under data scarcity. Unlike prior PINNs that rely on fixed or heuristic weighting of physics residuals and data loss, our approach introduces a learnable blending neuron that dynamically adjusts the relative contribution of each term based on their uncertainties. This mechanism enables stable training and improved generalization without manual tuning. To further enhance efficiency, we integrate a transfer learning strategy that reuses representations from related domains and adapts them to new physical systems with limited data. We validate the framework for the prediction of heat transfer in liquid-metal miniature heat sinks using only 87 CFD datapoints, where the adaptive PINN achieves an error <8%, outperforming shallow neural networks, kernel methods, and physics-only baselines. Our framework provides a general recipe for embedding physics adaptively into neural networks, offering a robust and reproducible approach for data-scarce problems across various scientific domains, including fluid dynamics and material modeling. 

---
# Layout-Aware Representation Learning for Open-Set ID Fraud Discovery 

**Authors**: Jinxing Li, Nicholas Ren, Cathy Chang, Hongkai Pan, Daniel George  

**Link**: [PDF](https://arxiv.org/pdf/2605.05215)  

**Abstract**: Identity-document fraud detection is not a stationary binary classification problem. Adaptive attackers modify templates and fabrication pipelines, making historical fraud labels stale, and successful forgeries recur at scale as coherent campaigns. We therefore study layout-aware representation learning for open-set fraud discovery rather than only closed-set classification. We adapt DINOv3 to the document domain via context-aware SimMIM fine-tuning and supervised metric learning with composite loss that encourages inter-class separability and intra-class compactness. The model is trained with U.S. IDs only. With a lightweight MLP and softmax classifier, the embedding achieves 99.83% layout classification accuracy on Canadian layouts. Moreover, on a dataset of 20,448 Canadian IDs, embedding-space analysis surfaces 276 adaptive physical-fraud cases, including 222 not surfaced by incumbent detectors. The embedding supports similarity-based expansion from a single confirmed seed to additional related cases not linked by conventional metadata graphs. The layout-aware document embeddings provide a production-aligned basis for discovering novel and campaign-scale fraud under distribution shift. 

---
# MedMamba: Recasting Mamba for Medical Time Series Classification 

**Authors**: ZhengXiao He, Huayu Li, Xiwen Chen, Janet M Roveda, Jinghao Wen, Siyuan Tian, Ao Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.05214)  

**Abstract**: Medical time series, such as electrocardiograms (ECG) and electroencephalograms (EEG), exhibit complex temporal dynamics and structured cross-channel dependencies, posing fundamental challenges for automated analysis. Conventional convolutional and recurrent models struggle to capture long-range dependencies, while Transformer-based approaches incur quadratic complexity and often introduce redundant interactions that are misaligned with the intrinsic structure of physiological signals. To address these limitations, we propose MedMamba, a principle-driven multi-scale bidirectional state space architecture tailored for medical time series classification. Our design is guided by three key inductive biases of physiological signals: spatial centralization, multi-timescale temporal composition, and non-causal contextual dependency. These principles are instantiated through a lightweight channel-mixing module for cross-channel reparameterization, multi-scale convolutional tokenization for temporal decomposition, and bidirectional Mamba blocks for efficient global context modeling with linear complexity. Extensive experiments on six benchmark datasets spanning EEG, ECG, and human activity signals demonstrate that MedMamba consistently outperforms state-of-the-art methods across diverse modalities. Notably, it achieves 85.97% accuracy on PTB and establishes new state-of-the-art performance on the challenging ADFTD dataset (54.72% accuracy and 52.01% F1-score). Strong results on long-sequence benchmarks, such as SleepEDF, further validate its capability in modeling long-range dependencies. Moreover, MedMamba achieves a speedup of 4.6x in inference, highlighting its practicality for real-time clinical deployment. These results suggest that principle-guided state space modeling offers an effective and scalable alternative to Transformer-based approaches for medical time series analysis. 

---
# A Review of Large Language Models for Stock Price Forecasting from a Hedge-Fund Perspective 

**Authors**: Olivia Zhang, Zhilin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05211)  

**Abstract**: Large language models (LLMs) are increasingly deployed in quantitative finance for stock price forecasting. This review synthesizes recent applications of LLMs in this domain, including extracting sentiment from financial news and social media, analyzing financial reports and earnings-call transcripts, tokenizing or symbolizing stock price series, and constructing multi-agent trading systems. Particular attention is paid to practical pitfalls that are often understated in the literature, such as fragility in sentiment analysis, dataset and horizon design, performance evaluation metrics, data leakage, illiquidity premia, and limits of stock price predictability. Organized from a hedge-fund perspective, the review is intended to guide both academic researchers and hedge fund managers in integrating LLMs into real-world trading pipelines and in stress-testing their robustness under realistic market frictions. 

---
# Are Flat Minima an Illusion? 

**Authors**: Michael Timothy Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2605.05209)  

**Abstract**: Neural networks that land in flat regions of the loss landscape tend to generalise better than those in sharp regions. Sharpness-Aware Minimisation exploits this to improve generalisation. But function-preserving reparameterisation can inflate the Hessian of any minimum by two orders of magnitude without changing a single prediction. If the geometry of weight space can be manufactured from nothing, it cannot be the cause of anything. In other words, flat is simple and simplicity depends on encoding. Here I show that the actual driver is weakness, the volume of completions compatible with the learned function in the learner's embodied language. Weakness is reparameterisation-invariant because it is defined over what the network \emph{does}, not how it is parameterised. I prove weakness is minimax-optimal under exchangeable demands, and that PAC-Bayes bounds work because they correlate with it. On MNIST, the large-batch generalisation advantage \emph{vanishes} as training data grows, from $+1.6\%$ at $n = 2{,}000$ to $+0.02\%$ at $n = 60{,}000$. A quantity whose predictive power depends on how much data you have is not a cause but a confounder. I run head-to-heads on 100 networks with identical architecture and training. For MNIST weakness predicts generalisation ($\rho = +0.374$, $p = 0.00012$), sharpness anticorrelates ($\rho = -0.226$) and simplicity predicts nothing ($p = 0.848$). For Fashion-MNIST ($\rho = +0.384$, $p = 8.15 \times 10^{-5}$), though simplicity is at least somewhat predictive there. Simplicity is dataset dependent, whereas weakness is invariant. Flat minima were never the answer. 

---
# A Note on TurboQuant and the Earlier DRIVE/EDEN Line of Work 

**Authors**: Ran Ben-Basat, Yaniv Ben-Itzhak, Gal Mendelson, Michael Mitzenmacher, Amit Portnoy, Shay Vargaftik  

**Link**: [PDF](https://arxiv.org/pdf/2604.18555)  

**Abstract**: This note clarifies the relationship between the recent TurboQuant work and the earlier DRIVE (NeurIPS 2021) and EDEN (ICML 2022) schemes. DRIVE is a 1-bit quantizer that EDEN extended to any $b>0$ bits per coordinate; we refer to them collectively as EDEN.
First, TurboQuant$_{\text{mse}}$ is a special case of EDEN obtained by fixing EDEN's scalar scale parameter to $S=1$. EDEN supports both biased and unbiased quantization, each optimized by a different $S$ (chosen via methods described in the EDEN works). The fixed choice $S=1$ used by TurboQuant is generally suboptimal, although the optimal $S$ for biased EDEN converges to $1$ as the dimension grows; accordingly TurboQuant$_{\text{mse}}$ approaches EDEN's behavior for large $d$.
Second, TurboQuant$_{\text{prod}}$ combines a biased $(b-1)$-bit EDEN step with an unbiased 1-bit QJL quantization of the residual. It is suboptimal in three ways: (1) its $(b-1)$-bit step uses the suboptimal $S=1$; (2) its 1-bit unbiased residual quantization has worse MSE than (unbiased) 1-bit EDEN; (3) chaining a biased $(b-1)$-bit step with a 1-bit unbiased residual step is inferior to unbiasedly quantizing the input directly with $b$-bit EDEN.
Third, some of the analysis in the TurboQuant work mirrors that of the EDEN works: both exploit the connection between random rotations and the shifted Beta distribution, use the Lloyd-Max algorithm, and note that Randomized Hadamard Transforms can replace uniform random rotations.
Experiments support these claims: biased EDEN (with optimized $S$) is more accurate than TurboQuant$_{\text{mse}}$, and unbiased EDEN is markedly more accurate than TurboQuant$_{\text{prod}}$, often by more than a bit (e.g., 2-bit EDEN beats 3-bit TurboQuant$_{\text{prod}}$). We also repeat all accuracy experiments from the TurboQuant paper, showing that EDEN outperforms it in every setup we have tried. 

---
