# BioEmu2: Scalable Emulation of Biomolecular Dynamics 

**Authors**: Michael Gastegger, Adam E. Foster, José Jiménez-Luna, Tim Hempel, Sarah Lewis, Yu Xie, Lixin Sun, Iryna Zaporozhets, J. Thorben Frank, Yaoyi Chen, Michael Plainer, Ludwig Winkler, Pedro B. P. S. Reis, Sven H. Giese, Roberto Sordillo, Victor García Satorras, Eray Inanc, Kacper Kapuśniak, Soojung Yang, Rishwanth Raghu, Rachel Kim, Gabriel Dernbach, Deniz Gunceler, Hannes Schulz, Bonnie Kruft, Marwin Segler, Christopher M. Bishop, Cecilia Clementi, Ryota Tomioka, Frank Noé  

**Link**: [PDF](https://arxiv.org/pdf/2611.27924)  

**Abstract**: Despite major advances in structure prediction, modelling the motions and interactions that drive biomolecular function at scale remains a central challenge. We introduce BioEmu2, a generative model for atomically resolved biomolecular structures combining static structure prediction, equilibrium sampling and time evolution, that samples orders of magnitude faster than molecular dynamics (MD). Trained on experimental and synthetic structures and extensive MD data, it supports multi-component systems containing proteins, nucleic acids, modified residues, ligands and ions. Structure-prediction accuracy is comparable to top open-weight biomolecular folding models, and a context parallel mode enables prediction of megadalton complexes. Equilibrium distributions approximate MD references with free-energy errors near 1 kcal/mol across tested proteins and complexes. Time propagation predicts long-timescale dynamical trajectories and kinetics of binding and conformational changes. Generated structures can be controlled by conditioning on residue contacts or steering with user-defined bias potentials. Enhanced sampling enables free energy calculation involving rare states. BioEmu2 makes the study of biomolecular ensembles and dynamics scalable. 

---
# Functional Depth Is Learned: Isolating the Computation That Layers Produce Only Together 

**Authors**: JaeSeong Kim, KyungTae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2611.27911)  

**Abstract**: Transformer performance scales with depth, but layer count merely counts the opportunities for sequential computation: layers could still behave as nearly independent refinements that barely depend on one another. We study how transformers develop functional depth during pretraining, introducing an exact inclusion-exclusion decomposition that isolates the connected component, i.e., the computation that appears only when a specified set of layers acts together under identity ablation. Tracking this quantity across public checkpoints of Pythia, OLMo, and Amber yields three findings. First, write-relative connected coordination strengthens from its initialization or earliest-checkpoint baseline as pretraining proceeds, across all measured interaction orders. Second, these connected components are behaviorally consequential under direct intervention: deleting them damages perplexity more than deleting a norm-matched random direction at the same site in all 35 measured model-order cells. Third, pairwise representation similarity is consistently negatively correlated with pairwise connected coordination across seven models, yet the two signals are not interchangeable: similarity- and coordination-based block-removal criteria often agree in Pythia but diverge on Amber. Together, these results suggest that architectural depth provides the opportunity for sequential computation, while pretraining progressively organizes cross-layer dependencies within that depth. 

---
# The Dual-Tier Sentiment Framework: Scalable Multimodal UX Attribution in Web Search 

**Authors**: Yi Ding, Gaurav Kumar, Rui Min, Isabella Ye, Bhavika Goyal, Emilio Antunez, Huihui Chen, Ke Xu, Xiaowei Sun, Xin Xu, Huijie Feng, Pradyumna Narayana, Golnaz Farhadi  

**Link**: [PDF](https://arxiv.org/pdf/2611.27899)  

**Abstract**: Sustaining user experience (UX) quality in large-scale web search requires scalable evaluation. Traditional surveys are slow and bottleneck rapid A/B testing, while automated metrics often lack the granularity to diagnose visual-spatial UX anomalies. To address this, we introduce the Dual-Tier Sentiment Framework, a novel industrial architecture for automated UX attribution. Tier 1 (Macro System) acts as a scalable user sentiment regression detector, utilizing a fine-tuned encoder-decoder LLM (based on Gemini XS) on telemetry features. Through multi-task regularization and contextual feature engineering, this tier achieves a +9.8% PR-AUC gain on over 500,000 held-out records, matching baseline recall on all validated regressions with $\sim$20% relative precision gain among the 12 held-out experiments. In production, it reduces measurement latency from weeks to hours and cuts survey overhead by over 40%. Tier 2 (Micro System) provides granular diagnostics by employing a Multimodal LLM (Gemini 3.0 Pro) to natively assess visual renderings and spatial layout of web result blocks, quantifying "User Toil" as the estimated effort required to reach a satisfactory result. Enhanced by RL-based automated prompt optimization and an AI judge for win/loss patterns, the MLLM yields a $>$6.5% F1 score improvement, offering deep insights into content quality and layout friction. Ultimately, this framework enables scalable, accurate, and diagnostic UX attribution, processing O(100k) live experiment queries daily. 

---
# Speculative Decoding Is Less Speculative Than You Think 

**Authors**: Guanning Zeng, Andrea Zanette  

**Link**: [PDF](https://arxiv.org/pdf/2611.27891)  

**Abstract**: Speculative decoding accelerates language-model inference by preparing candidate continuations before the target model generates them. Its efficiency depends on how well limited drafting computation anticipates the path the target will actually follow. Anticipating this path involves two sources of uncertainty: the target model's predictions and the randomness of sampling. Yet the former requires a target-model forward pass, while the latter can be fixed before drafting. Building on this observation, we introduce DICE (Draw-Informed Candidate Selection), which pre-samples and synchronizes verification randomness across drafting and verification. By combining these known draws with predictions from an existing drafter, DICE allocates computation toward continuations more likely to be used in the current sampling run. This principle guides node selection throughout draft-tree construction and extends to asynchronous prefetching, where DICE predicts which next-round drafts to prepare. Without additional drafter training or changes to the target sampling distribution, DICE achieves state-of-the-art decoding throughput across diverse settings, with speedups of up to 24.6pp over strong baselines. 

---
# Integrating Generative Artificial Intelligence with Learning Style Models: Opportunities, Challenges, and Future Directions 

**Authors**: Malithi Nawarathne, Prasanna Haddela, Nuwan Kodagoda  

**Link**: [PDF](https://arxiv.org/pdf/2611.27868)  

**Abstract**: Generative Artificial Intelligence (Generative AI) has emerged as a powerful tool for transforming learning. This review examines how Generative AI can be integrated with established learning style models (VARK, Kolb, Felder-Silverman, and Honey & Mumford) to support student engagement in large online classes. Following PRISMA 2020 guidelines, we searched Scopus, IEEE Xplore, and Google Scholar (2015 to 2026) and included 41 studies that met the eligibility criteria. The review maps Generative AI capabilities to learning style preferences, synthesizes existing implementations, highlights benefits and challenges, and proposes pathways for future research and practice. Despite ongoing debates regarding the empirical validity of learning styles, Generative AI enables flexible, multimodal, and real-time adaptation. This study offers practical insights for educators and developers seeking to build more effective, learner centered learning systems 

---
# SLICR: Strategy Logic-Guided Multi-Agent Reinforcement Learning 

**Authors**: Arshia Rafieioskouie, Matthew Lucas, Tzu-Han Hsu, Borzoo Bonakdarpour  

**Link**: [PDF](https://arxiv.org/pdf/2611.27865)  

**Abstract**: Specification-guided reinforcement learning (RL) translates formal specifications of desired behavior of agents into shaped rewards. This paper introduces SLICR, a significant paradigm shift that advances the state of the art to specification-guided RL in adversarial settings under partial observability. We employ Strategy Logic (SL), which allows explicit quantification over agents' strategies and uses the linear temporal logic (LTL) to specify the agents' objectives and constraints. This enables reasoning not only about how agents should act to satisfy a specification, but also about how they should respond to the strategies and behaviors of other agents. We evaluate SLICR on three popular benchmarks: Predator-Prey, Google Research Football, and our home-grown extension of SMAC-HARD, demonstrating its effectiveness, scalability, and ability to adapt to diverse adversarial strategies. 

---
# Beyond Self-Critique: Epistemic Convergence Control for Multi-Agent Research Verification 

**Authors**: Chrisantus Eze  

**Link**: [PDF](https://arxiv.org/pdf/2611.27839)  

**Abstract**: Multi-agent systems for scientific claim verification increasingly rely on role-structured LLM debate to pressure-test hypotheses. However, when debating agents are instantiated from the same underlying base model, they share identical latent representations, making self-critique bias structurally inevitable. Static fixed-iteration protocols compound this by either halting prematurely before evidence stabilizes or wasting inference compute after convergence. We present \sys{}, a multi-agent research verification framework that addresses both failure modes. First, we introduce an asymmetric role architecture comprising an Advocate, Skeptic, and Arbitrator with opposing epistemic directives, producing significantly lower role-embedding similarity (0.852 vs.\ 0.918 for symmetric Multi-Agent Debate [MAD], $p = 5.82 \times 10^{-8}$) and surfacing more contested claims (22.0\% vs.\ 15.4\%). Second, we propose the Belief-Delta ($\DB$) convergence criterion, an evidence-grounded stopping rule that eliminates over-iteration (0.0\% vs.\ 62.5\% for Fixed-$3$) while cutting LLM calls by 2.4$\times$. Third, each supported claim is anchored to a verbatim literature snippet audited by an independent NLI classifier, achieving a 21.7\% hallucination rate (comparable to 25.9\% for a Self-Refine critic, $p = 0.61$) while establishing auditable evidence grounding. Across 200 runs on 40 research hypotheses evaluated with DeepSeek-R1-32B, \sys{} balances compute efficiency with rigorous epistemic verification. 

---
# Auditable Financial Knowledge Graphs: Schema-Driven Construction and Intrinsic Quality Evaluation 

**Authors**: Samarth Sarin, Lovepreet Singh, Dhagash Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2611.27835)  

**Abstract**: Financial documents combine narrative disclosures, accounting metrics, corporate structures, management commentary, and time-dependent facts, making them difficult to represent using conventional chunk-based retrieval-augmented generation systems. This paper presents an auditable, schema-driven methodology for constructing temporal financial knowledge graphs from page-delimited documents parsed using a general purpose document intelligence service. Overlapping document windows undergo phased large language model extraction of structural entities, quantitative observations, and domain-specific relationships. Candidate graph elements are validated against predefined schemas and consolidated through scope-aware exact, normalized, fuzzy, embedding-based, and LLM-assisted entity resolution. Temporal merge constraints, deterministic edge deduplication, conflict-aware property management, and fine-grained provenance preserve issuer identity, reporting-period validity, and traceability to source evidence. The contributions of this work comprise the proposed methodology for financial knowledge graph construction, the creation of an evidence-grounded temporal knowledge graph from heterogeneous financial documents, and an intrinsic framework for evaluating the quality of the resulting graph. The evaluation framework assesses seven weighted dimensions, two hard validation gates, and structural diagnostics. By separating graph-quality assessment from downstream retrieval and answer generation, the framework supports systematic diagnosis and iterative refinement of the constructed graph. 

---
# SLP-Bench: Evaluating and Improving Acoustic Claims of Reasoning Audio Language Models on Clinical Speech 

**Authors**: Côme Hosxe, Rachid Riad, Quentin Spinat  

**Link**: [PDF](https://arxiv.org/pdf/2611.27810)  

**Abstract**: Audio language models (ALMs) can answer clinical questions from speech, outperforming baselines built on handcrafted acoustic features. Reasoning ALMs generate a chain of thought that refers to properties of the speaker's voice before predicting. Each statement the model makes about the voice needs to be checked against clinicians' perceptual ratings of the same recording. On clinical speech, existing evaluations score only the final answer. We introduce Speech Language Pathologist Bench (SLP-Bench), a benchmark grading each acoustic claim in the chain of thought against clinicians' perceptual ratings of the same recording. It covers five clinician-rated datasets and 25 perceptual dimensions, and assesses claims for factuality, stability under resampling, and consistency across reworded and different questions. Across ten open-weight reasoning ALMs evaluated zero-shot, claim balanced accuracy ranges from 0.52 to 0.58, claims change under resampling and rewording, and the most consistent models are not the most factual. Fine-tuning a 7B ALM with GRPO on a reward for claims that agree with clinicians raises its claim balanced accuracy from 0.54 to 0.60 and its mean kappa from 0.39 to 0.61, while preserving answer accuracy on per-dimension questions. SLP-Bench is available upon request, subject to approval by the providers of each dataset. 

---
# ChildAI-Gov: An Evidence-Linked AI System for Assessing EdTech Policies Against Children's Digital Rights 

**Authors**: Naveen Sigalapalli, Huizhi Liang, Clara Crivellaro  

**Link**: [PDF](https://arxiv.org/pdf/2611.27799)  

**Abstract**: We present ChildAI-Gov, an explainable AI system for assessing whether education-technology (EdTech) policies respect children's digital rights. Generative AI is reaching children's education faster than its governance can keep pace, yet checking a tool against children's rights remains manual, expert-dependent, and slow. We propose the Child Rights Impact Score (CRIS), an evidence-linked measure across five dimensions: safety, privacy, transparency, autonomy, and non-discrimination. ChildAI-Gov reads an EdTech tool's privacy or AI policy and scores each dimension by combining a large-language-model judgement (gpt-4o-mini) with a deterministic rule check. A retrieval-augmented generation (RAG) component grounds every score in the governing provisions, retrieved from a curated knowledge base of children's rights frameworks (the EU AI Act, UNICEF guidance, and others), so each score is tied to the evidence behind it and an auditor can trace which provision it rests on. We also construct AIPolicy-EdTech, a dataset of 57 real EdTech policies. On it, the five dimensions cohered as a single measure (Cronbach's alpha = 0.79), and grounding made scoring stricter while preserving rankings; an ungrounded baseline was far more lenient, passing nearly half the policies where ChildAI-Gov passed none. It offers evidence-linked decision support on documented commitments, not observation of real-world behaviour or determinations of legal compliance. 

---
# A Benchmark Protocol and Fairness Analysis Towards Open-Set EEG Seizure Type Classification 

**Authors**: Targol Bakhtiarvand, Terrance E. Boult, Jugal Kalita, Adham Atyabi  

**Link**: [PDF](https://arxiv.org/pdf/2611.27783)  

**Abstract**: Automated EEG seizure type classification systems are typically trained and evaluated under the closed-set assumption, where all seizure types encountered at deployment are seen during training. In clinical practice, this assumption fails: patients present with rare or atypical seizure types that fall outside the training label space, and a system that forces a wrong label is worse than one that abstains. We introduce the first open-set recognition protocol for EEG seizure type classification on the Temple University Hospital Seizure Corpus (TUSZ v2.0.3), defining known and unknown classes using the clinical taxonomy itself: seizure types with sufficient subject representation (at least 12 subjects) form the known set, and rare types form the unknown set. We evaluate seven post-hoc open-set scorers (MSP, MaxLogit, energy, Mahalanobis, kNN, GHOST, and COSTARR) applied to a partially fine-tuned EEG Foundation Model (EFM), using a subject-disjoint leave-one-class-out (LOCO) protocol over three seeds that tests whether methods can detect seizure types excluded from fine-tuning. On true unknown classes, the best scorer (Mahalanobis) reaches a mean AUROC of 0.66, and differences among the top scorers are within seed-to-seed variation. Held-out known seizure types from unseen subjects are detected only slightly above chance (AUROC 0.54-0.58 for all scorers). Per-subject rejection rates are highly uneven for all scorers, and GHOST does not significantly improve per-subject fairness over energy scoring (two-sided Wilcoxon signed-rank, p >= 0.25). We release the protocol, subject splits, and code as an initial baseline; our results indicate that open-set seizure type recognition remains a hard, open problem. 

---
# QuASAR: Toward an LLM-Assisted Framework for Screening, Prototyping, and Assessing Quantum-Advantage Opportunities in Scientific Computing 

**Authors**: Mark L. Carroll, Jordan A. Caraballo-Vega, Brian P. Powell, John L. Schnase  

**Link**: [PDF](https://arxiv.org/pdf/2611.27778)  

**Abstract**: Identifying scientific computations that could benefit from quantum algorithms requires both domain knowledge and fluency in quantum algorithm design, expertise that rarely resides in one person. We report results from QuASAR, an experimental, two-agent, LLM-driven pipeline intended to bridge this gap. One agent screens scientific literature for computational kernels with a plausible quantum-algorithmic analogue and proposes candidates, each backed by citations. A second agent independently verifies those citations. For candidates that survive verification, it reconstructs the source's computational structure, formulates a quantum mapping, builds and executes a prototype, and benchmarks it against classical baselines. We applied this pipeline to ten candidate computations drawn from the astrophysics literature. Nine candidates were carried through to a small-scale, benchmarked prototype. We report three in detail, chosen as the clearest examples with the least redundancy across the set. They span combinatorial optimization, stochastic simulation, and Bayesian inference. The pipeline attempted to supply quantum-design expertise a domain scientist might lack, reconstructing source algorithms where one existed and building original, disclosed mappings where none did. None of the three case studies achieved quantum advantage at the tested scales, and their astrophysical framing has not yet undergone expert validation. What the experiment does show is that the pipeline can turn a published idea into a runnable prototype, which a scientist can then inspect, test against classical methods, and decide whether to pursue. QuASAR is an exploratory framework, offered to the community for evaluation, critique, and further development, with an approach that could broaden the use of quantum computing in scientific research and extend to computational improvements more generally. 

---
# Can Vision-Language Models Synthesize Executable Game-Playing Programs? 

**Authors**: Yao Tong, Jiaxin Qi, Dingyu Zhang, Jianqiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27741)  

**Abstract**: Recent advances in vision-language models (VLMs) have endowed multimodal agents with increasingly strong visual perception and long-horizon reasoning capabilities, making VLM-based autonomous game playing feasible. Existing game benchmarks primarily evaluate whether an agent can select appropriate actions from the current visual state across different scenarios and games. However, most of them adopt a step-by-step interaction paradigm where the current visual observation is sent to the agent before each action. In practical deployment, this paradigm introduces two fundamental limitations: the game loop must wait for model inference, making real-time interaction impractical, and repeated VLM calls incur substantial cost. Therefore, we propose PlayScriptBench, a script-based game benchmark that evaluates whether VLMs can synthesize an executable game-playing program from game states. This setting jointly evaluates visual grounding, understanding of game dynamics, long-horizon strategic reasoning, and executable program synthesis, providing a more comprehensive assessment of game-playing agents. To support more tractable evaluation for current general-purpose VLMs, we further introduce a four-stage pipeline --- Explore, Perception, API Construction, and Strategy Composition --- that decomposes controller synthesis into game exploration, state perception, interface construction, and strategic control, enabling agents to progressively construct an executable program. Experiments on 16 games spanning 4 genres show that even strong VLMs struggle on our benchmark, revealing a substantial gap between passive visual game understanding and executable closed-loop game control. These results establish script-based controller synthesis as a complementary direction for benchmarking VLM-based game-playing agents. 

---
# Average-Depth Variable Execution via Request-Local Token Routing 

**Authors**: Pengjie Zheng, Mengwei Xie, Huan Yang, Si Wu, Pengfei Hu, Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.27738)  

**Abstract**: Layer skipping can reduce large language model inference costs, but a request budget does not specify how to allocate layer execution across generated tokens. We introduce \method{}, which allows tokens to execute different numbers of layers while matching each request's mean executed depth to a continuous budget. At each routed layer, a router reads the current token state and its request's budget and decides whether to execute the Transformer block. Tokens that skip the block pass through a lightweight residual adapter and write no key/value (K/V) entry at that layer. A single checkpoint serves requests with different budgets in the same batch. Across three LLaMA backbones, six quality metrics, and 13 budgets, AVERT is best or tied for best in 206 of 234 comparisons. Across single- and mixed-budget batches, the mean absolute per-request budget error is at most 1.15 percentage points. When grouping requests by budget leaves batches underfilled, mixing budgets reduces decode time per output token (TPOT) by 4.3--21.2%. 

---
# Recoverable Symbols, Unstable Execution: Auditing Internal Pathways and Decision Identity in Visual State Tracking 

**Authors**: Ruiqi Liu, Jinming Li, Mo Wang, Yang Lu, Yanlan Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27715)  

**Abstract**: Visual features can retain every symbol a task needs while a model built on them still changes its answer when irrelevant parts of the image change. We study this gap in visual state tracking, where a recurrent executor reads a sequence of symbol tiles through a frozen ViT encoder and decides whether a displayed candidate equals the final state. Our diagnostic pairs each input with a donor view that preserves every queried tile, checks with a symbol-supervised decoder that the full symbol chain is recoverable from both feature sequences, and swaps individual internal channels of the same fitted executor in both directions. On 3,526 pairs in which every symbol is recovered in both views, changing the surrounding context lowers the balanced accuracy of a rotary recurrent head by 17.13 points. The loss runs mainly through the input-dependent rotation phase: restoring the original phase repairs 351 of 391 new errors and brings accuracy back to 88.54%, yet 142 individual predictions still differ from the original ones. A locked test on 192 unseen program families reproduces the pathway (28 of 29 new errors repaired; 382 of 384 predictions unchanged), and the effect partly carries over to single-image canvases. Consistency objectives derived from this analysis reduce the internal discrepancy they target but do not outperform ordinary data augmentation. 

---
# MetaRigor: From Agentic Automation to Verifiable End-to-End Meta-Analysis Through Scientific Decision Boundaries 

**Authors**: Yifeng Xie, Chenke Li, Yangshuo Miao, Jingyuan Zheng, Mingxuan Liu, Feiwei Qin, YaQi Wang, Huiyu Zhou, Yifei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.27713)  

**Abstract**: Automated meta-analysis requires not only accurate numerical extraction but also the consistent preservation of result identity across stages, including the associated study, comparison, population, outcome, time point, and analysis conditions. We present MetaRigor, an LLM workflow framework for verifiable end-to-end evidence synthesis. MetaRigor organizes study results as Study x Target units and operationalizes scientific decision boundaries at the task level through Task Contracts, programmatic rules, and source binding. The model performs semantic interpretation that depends on study context, whereas programs execute formalizable rules, computations, validation, and source verification. We conducted controlled evaluations across five stages: evidence acquisition, data extraction, risk of bias assessment, certainty of evidence assessment, and manuscript generation. MetaRigor achieved an F1 score of 86.29% for complete result delivery and 80.20% agreement for domain-level certainty of evidence judgments. Generated manuscripts received scores of 86.00 under human evaluation, exceeding all three baselines. A seven-case downstream end-to-end evaluation further yielded an aggregate F1 score of 77.01%, exceeding the strongest baseline by 9.78 percentage points. Cross-system tracing of archived outputs from three systems further showed that local mismatches or omissions in result identity could alter statistical inputs and the objects subjected to risk of bias assessment, with these errors subsequently propagating into the final report. Ablation experiments showed that task constraints and programmatic rules played distinct roles across different stages. These findings indicate that, in LLM-driven evidence synthesis workflows, consistently maintaining explicit result identity, decision boundaries, and provenance relationships is critical to improving end-to-end verifiability. 

---
# When Are Images Needed in Multimodal Machine Translation? Text-Only Routing for On-Demand Visual Context 

**Authors**: Xinyu Wang, Yingfeng Luo, Yuxin Gao, Anxiang Ma, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.27712)  

**Abstract**: Image-guided multimodal machine translation (MMT) leverages visual context to resolve translation-relevant uncertainty. However, prevailing systems either process images unconditionally or apply gating only after visual encoding, leaving redundant visual computation unavoidable. To mitigate this computational overhead, routing decisions must rely on a pre-visual source signal. Crucially, surface-level ambiguity does not equate to visual dependence, as intra-sentence textual context often suffices for disambiguation. We therefore formulate visual invocation as source-text sufficiency prediction: determining exclusively from the source text whether translation-relevant uncertainty remains unresolved. To support this formulation, we construct a visual-dependency dataset with three fine-grained labels (no_ambiguity, text_resolved, and vision_needed) and verify discriminative visual evidence for vision_needed instances. We further propose PreVIR, a lightweight, backend-agnostic text-only router that selects between text-only and multimodal branches before computing visual features. Experiments on English-to-Chinese MMT demonstrate that PreVIR achieves a favorable COMET-budget trade-off relative to strong routing baselines. At its default operating point, it substantially reduces visual calls while maintaining translation quality comparable to always-on multimodal inference. Label-wise and case analyses show that this trade-off stems from distinguishing genuine visual dependency from contextually resolved ambiguity. 

---
# Abstract What Varies, Reuse Longer Patterns: Efficient Offline Chain-of-Thought Compression 

**Authors**: Shibo Wang, Yuzhe He, Yifeng Zhao, Yunjiao Zhou, Zepeng Ning, Hanzhang Zhou, Kezhi Mao, Junlang Qian  

**Link**: [PDF](https://arxiv.org/pdf/2611.27710)  

**Abstract**: Offline chain-of-thought (CoT) compression creates shorter supervision for efficient reasoning, but per-example compressors repeatedly process similar expressions across traces, wasting compressor budget. Methods that reuse identical text across examples miss longer shared expressions when names, numbers, or formulas change from one question to another. We introduce CRAFT, which abstracts problem-specific details into slots to uncover partially lexicalized expression templates (PLETs) and clusters wording variants into families. Compressing each family representative once yields shared deletion or rewriting edits, which are applied with each occurrence's own details to construct compact, ordinary-text training targets. Compared with per-example rewriting, CRAFT Rewrite reduces weighted compressor budgets by a median 90.7% with a median accuracy loss of 1.21 percentage points, while generating shorter reasoning. Against cross-sample encoding, it shortens reasoning across all eight model-dataset settings by 25.3% on average and improves accuracy in five. 

---
# Spatiotemporal Graph Neural Networks for Fault Detection and Structural Learning in Chemical Processes: Use Case on the Tennessee Eastman Process 

**Authors**: Rayane Ammar Khodja, Alexandre Voisin, Victor Costa, Fanny Casteran, Benoit Celse, Benoît Iung  

**Link**: [PDF](https://arxiv.org/pdf/2611.27698)  

**Abstract**: Classical Fault Detection and Diagnosis (FDD) methods, including many data-driven approaches, assume a static normal operating space and interpret deviations from a fixed reference as fault indicators. When Operating Conditions (OCs) vary over time, this assumption breaks down: legitimate transitions trigger false alarms while moderate faults go undetected. We propose a spatiotemporal Graph Neural Network (GNN) framework that decomposes the normal operating space into OC-specific subspaces linked by transition functions, with a dual learning objective combining reconstruction loss and a Deep SVDD one-class term. It learns adjacency matrices through Graph Attention Networks (GATs) and integrates spatial modelling with temporal encoding to represent process dynamics under evolving OCs. This paper evaluates the framework's foundational components (fault detection, GATv2 spatial graph learning and temporal encoding) on the Tennessee Eastman Process (TEP) benchmark, training only on fault-free data. The architecture achieves competitive detection from reconstruction error alone, with similarity-based feature selection improving both accuracy and graph structure diversity. We then evaluate the physical interpretability of the learned attention matrices against 22 ground-truth sensor pairs derived from the TEP control structure and process topology. GATv2 attention fails to recover all known physical pairs across multiple hyperparameter configurations, suggesting a structural limitation of reconstruction-driven attention rather than a tuning issue. This challenges a common assumption in GNN-based FDD: that learned attention weights provide a basis for fault diagnosis and root-cause analysis. The architecture detects faults effectively, but the learned graph does not encode the physical topology needed for interpretable diagnosis, motivating physics-informed graph construction. 

---
# Comparative Performance of Machine Learning Architectures for Fault Detection and Diagnosis in Chemical Processes 

**Authors**: Rayane Ammar Khodja, Alexandre Voisin, Victor Costa, Benoit Celse, Fanny Casteran, Benoît Iung  

**Link**: [PDF](https://arxiv.org/pdf/2611.27696)  

**Abstract**: Fault Detection and Diagnosis (FDD) is crucial for both maintenance and control in chemical industries, where early fault detection can prevent costly failures and optimize operations. This is particularly critical in pilot units like those at IFP Energies Nouvelles (IFPEN), which operate under short-term experimental conditions with frequently varying operational parameters. This study conducts an extensive benchmarking analysis using the Tennessee Eastman Process (TEP), a widely used simulated chemical process dataset, to evaluate multiple approaches for fault detection and diagnosis, featuring numerous continuous operations with different sensors and faults. Several methods were implemented and compared including Multi Scale PCA (MS-PCA), AutoEncoder, Ensemble Learning, and LSTM models for fault detection, alongside Random Forest, XGBoost, and BLSTM (Bidirectional LSTM) for fault diagnosis. Our results demonstrate that Ensemble Learning achieves detection rates ranging from 80% to 100% across various fault scenarios. For the fault diagnosis task, BLSTM achieved a diagnosis accuracy of 98.76%. The study reveals that ensemble-based approaches consistently outperform individual models in handling the complex, multivariate nature of chemical process data, due to its robustness in combining multiple perspectives, comprehensive data capture, and localized detection capabilities. Furthermore, the superior performance of the BLSTM is due to its ability to capture both past and future temporal dependencies in the sequential data, particularly important in chemical processes where fault patterns may manifest with complex temporal relationships. This research contributes to the field of Fault Detection and Diagnosis by providing empirical evidence for the effectiveness of ensemble methods and Bidirectional LSTMs in order to address industrial FDD applications on chemical processes. 

---
# Learning in Symbiosis: Heterogeneous Federated Learning with ANNs and SNNs 

**Authors**: Manh V. Nguyen, Liang Zhao, Bobin Deng, Xinyue Zhang, Shaoen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27686)  

**Abstract**: Spiking Neural Networks (SNNs), widely regarded as the third generation of neural networks, represent a promising neuromorphic computing paradigm with the potential to reduce the energy footprint of Artificial Intelligence (AI). Their en-ergy efficiency makes SNNs particularly attractive for resource-constrained Edge AI devices, motivating growing research interest in Federated Learning (FL) with SNNs. Despite these advantages, integrating SNNs into existing FL systems remains challenging, as replacing all conventional Artificial Neural Network (ANN) devices with SNN-based alternatives is impractical. Instead, we envision a gradual transition toward neuromorphic Edge AI, where ANN and SNN devices coexist and collaboratively participate in FL. To facilitate this transition, we propose Hybrid Neuromorphic Federated Learning (HNFL), a framework that enables joint training of heterogeneous ANN and SNN devices within an FL environment. Through experiments on CIFAR-10 using both VGG9 and ResNet architectures, we demonstrate the mutual benefits of collaborative ANN-SNN training. Specifically, HNFL achieves an average improvement of 16.85% when incorporating SNN devices into ANN-based FL and 8.02% when incorporating ANN devices into SNN-based FL. Furthermore, HNFL produces sparser spiking activity in the trained SNN models, with reductions of 12.4% for SResNet and 22.4% for SVGG9. To the best of our knowledge, this is the first work to demonstrate that ANNs and SNNs can coexist and mutually benefit from collaborative training in FL, providing a practical path toward the gradual integration of neuromorphic computing into existing Edge AI systems. 

---
# Learning Inside Types: Epistemic Typestates for Safe Multi-Agent Reinforcement Learning 

**Authors**: Lorenzo Bacchiani, Mario Bravetti, Giovanni Pau, Roberto Girau  

**Link**: [PDF](https://arxiv.org/pdf/2611.27631)  

**Abstract**: Type disciplines make programs safe by construction: components checked against each other's types compose into programs that cannot go wrong in the ways the types exclude. Types suit reinforcement learning: a type constrains what an agent may choose, not how it chooses, so the guarantee holds for every policy a learner produces. We introduce epistemic typestates, a type discipline for safe multi-agent reinforcement learning. An agent's type is a protocol whose states record its beliefs, the worlds it cannot rule out from its own observations and actions, and permits only actions that are safe in every world of the belief, whatever the others do within their public types. Agents learn inside their types, which are built and checked one agent at a time and run without communication. We prove type soundness and team safety, give a synthesis that is sound and complete, build types from guards when worlds are infinite, and show that shields are a special case. All proofs are mechanised in Rocq. On nine benchmarks, no typed learner causes a safety violation, and task performance does not differ significantly from that of two recent multi-agent shielding methods. Where a contract-based method fails at five and ten agents, types for ten agents are built in under a second. In the two continuous benchmarks, Lagrangian learners cause thousands of violations in every training run. 

---
# Benchmarks' Last Exam 

**Authors**: James Zhu, Zekki Cat  

**Link**: [PDF](https://arxiv.org/pdf/2611.27622)  

**Abstract**: Public benchmarks guide model selection, post-training rewards and claims of progress in artificial intelligence, yet leaderboards report aggregate scores without verifying whether each task's prompt, environment and grader agree on what counts as success. How often that agreement fails in contemporary evaluations, and in which direction confirmed defects move scores, has remained unmeasured across grading designs. Here we synthesize task-level quality reviews of 749 tasks across eight frontier benchmarks and 767 public model evaluations spanning closed-form question answering, document rubrics and agentic execution to assess whether prompts, environments and graders align. Task validity stratifies sharply by grading architecture. Closed-form benchmarks revised after public error reporting reach 92.5% verified validity, compared with 29.3% across six multi-step suites that nevertheless already supply 161 verified complex tasks. Rubric suites fail through mismatches between prompts and reference checklists, whereas execution harnesses add incomplete state assertions, shared-channel leaks and grading-time collisions. Across confirmed defects, 71.7% withhold credit from compliant solutions while 28.3% permit shortcut passes, indicating that raw leaderboard ceilings understate frontier capability on many practical tasks. Grading committed patches inside a fresh container cuts confirmed runtime bypasses from 14.7% in shared containers to 2.7% (a 5.5x drop, and 0% in read-only or static suites), showing how targeted maintenance restores existing benchmarks at low cost. As the initial milestone of the long-term Benchmarks' Last Exam initiative, this synthesis launches sustained expansion across wider benchmark coverage and deeper evaluation-quality dimensions so that pursuing leaderboard gains directly advances genuine model capability. 

---
# Can Language Models See the Circuit in Pixelated RFIC Passives? 

**Authors**: Jonathan Zhou, Kaushik Sengupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.27559)  

**Abstract**: The emergence of inverse-designed, pixelated passives for microwave and millimeter-wave applications has been a major development in the field of RFIC design. These devices, which appear to be QR-code-like in structure, exist in a generalizable, high-dimensional design space that includes conventional passive topologies. However, these structures are naturally unintuitive for designers, and failures or deviations in performance are difficult to debug. In this work, we ask whether a frontier language model can understand, predict, and edit the responses and layout structure of pixelated passives. The agent, run with all tools disabled, is first asked to read layouts on 18x18 and 27x27 tile grids and, in a single reply, predict the input impedance and loss of the structure from 20-100 GHz. On eight GA-synthesized layouts and five random fills, the predicted Smith-chart locus follows full-wave simulation results closely, with a systematic shift that places features about 16% lower in frequency than EM. On multipath loop structures it identifies notches successfully but shows a smaller frequency bias, about half that of an ideal transmission-line model that ignores bends. The agent is then asked to debug seven designs that fail their specification, and given a budget of five EM simulations and at most twelve changed tiles, it is able to improve all seven and bring five designs within specification. These experiments suggest language models can serve as an interpretability and debugging tool for traditionally non-intuitive inverse-designed passives. 

---
# Vega-Fafnir: Is Greedy Decoding the Same Decoding? Where Rounding Noise in Speculative Verification Comes From and What Removing It Costs 

**Authors**: Mohammed Nayeem Khan, Navin Kumar R  

**Link**: [PDF](https://arxiv.org/pdf/2611.27554)  

**Abstract**: Serving engines treat the ways of running a greedy language model as interchangeable: cached or not, padded or not, alone or batched, one token or several verified at once (speculative decoding). In finite precision they round differently. We measure how often the greedy text of Qwen3-0.6B (bfloat16, 128 new tokens, GSM8K and ARC prompts) differs from a reference that decodes one token per pass. Verifying 4 guessed tokens per pass changes 60% of 150 prompts and padding changes 68%; every first divergence has a top-1 minus top-2 logit gap of 0.375 or less. A split of the noise shows that cache entries written by multi-token passes are about as noisy as the multi-token pass itself. A margin gate (fall back to one token when the gap is below a bound calibrated on separate GSM8K prompts) whose fallback reads that cache does not help: 58 of 100 prompts differ, against 61 of 100 for plain verification. Rebuilding the cache with one-token passes leaves 0 of 100 different but costs 7.2 model passes per token, against 1.0 for the reference. Fixed-width decoding makes drafted and plain decoding identical (0 of 100 prompts differ) at 0.75 passes per token. On shifted MMLU and HellaSwag prompts the rebuild gate leaves 0 of 60 different, plain verification 48 of 60. On 138 further and shifted prompts the bound was exceeded at 0.6% of positions and missed 0 of 309 flips. One GPU, one drafter, small models. 

---
# Altair-Nidhogg: Which Model-Merging Theories Hold for a One-Pass Decision Encoder? Soups, Task Arithmetic, TIES and DARE Against Joint Training and Prediction Ensembling 

**Authors**: Mohammed Nayeem Khan, Navin Kumar R, Akhil P  

**Link**: [PDF](https://arxiv.org/pdf/2611.27550)  

**Abstract**: Laya is a small model that answers a multiple-choice question in one pass: each option has a name and a written description. A new skill costs a training run, so we test whether published recipes for merging fine-tuned models (weight averaging, task arithmetic, TIES, DARE) work for it, and whether the theories behind them hold. All 156 models start from one released checkpoint: four plain fine-tunes, five specialists (each adds one kind of training row), joint fine-tunes on all rows, and prediction ensembles. Items are held-out, synthetic and LLM-labelled. In-family, each skill is scored on new items of its own kind, and "kept" is the share of a specialist's gain over the plain fine-tune. Averaging four specialists keeps 52% [38, 64] with +1.7 points of general accuracy; adding the full vectors keeps 42% [3, 63] with -9.0; TIES, which trims and resolves sign conflicts, keeps 64% [50, 76] with +1.5; joint fine-tunes, which see all the rows, keep 120% [104, 162] and 116% [104, 149]. DARE does not repair the full sum (-0.4 points), and ensembles of the models' predictions keep 26% to 54% depending on how they are combined. Swapping the seed of one specialist moves the average by up to 8 points (23 for the full sum). Of 10 predictions fixed in advance, 1 held, 1 held partly, 7 failed and 1 could not be tested. Out-of-family (benchmark questions, code-mixed tickets), the four-way average, joint models and ensemble stay within 2.6 points of the plain fine-tune, but full-vector sums lose up to 7.7. Labels are LLM-made, data synthetic, one base model. 

---
# Trained on Harm, Still Refusing: A Training-Time Defense for Code Generation 

**Authors**: Miaoqing Tian, Ruohan Li, Qian Lu, Haipeng Qu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27495)  

**Abstract**: Fine-tuning enables safety-aligned code language models to adapt to diverse downstream tasks. However, training on untrusted data can severely undermine their ability to refuse harmful requests. Notably, we find that models comply with harmful requests even though harmfulness information remains highly decodable from their internal representations. Moreover, an initial preference for refusal does not necessarily translate into sustained refusal behavior throughout generation. Together, these findings reveal a critical gap between recognizing harmful requests and consistently refusing to fulfill them. To address this gap, we introduce SAFEHOLD, a novel provider-enforced defense framework that preserves safety alignment during harmful fine-tuning by treating refusal as a sustained generation process rather than a single response decision. Specifically, SAFEHOLD preserves the aligned model's full-vocabulary response distribution at the initial generation position through decision-boundary anchoring, while supervising the complete refusal sequence through refusal-path reinforcement. We conduct extensive experiments evaluating safety, utility, and internal mechanisms. SAFEHOLD reduces average ASR by 72.03 percentage points relative to undefended fine-tuning, outperforming five SOTA defenses by 24.32 to 70.87 percentage points. Meanwhile, it improves average HumanEval pass@1 by 10.24 percentage points over undefended fine-tuning, while maintaining an average XSTest over-refusal rate of only 2.73 percent. Overall, these results demonstrate that SAFEHOLD effectively mitigates safety degradation during harmful fine-tuning while preserving benign-task utility. 

---
# Efficient Natural Language to Temporal Logic Translation for Real-Time Physical AI 

**Authors**: Oliver Schön, Adrien Banse, John Lygeros, Lars Lindemann  

**Link**: [PDF](https://arxiv.org/pdf/2611.27489)  

**Abstract**: Robots can be controlled to perform tasks with guaranteed correctness when the task instructions are stated in a formal language. Large language models (LLMs) have been widely used to allow users to provide such instructions in natural language (NL), by translating NL into formal language. When deploying such translators on a robot, however, we face limited computing and energy resources, while offloading computation or data to the cloud may be infeasible or undesirable. Yet an LLM is a general-purpose text generator, whereas a formula in a fixed formal language fragment follows a well-defined syntax. We therefore ask whether text generation is needed at all. For the formal language of linear temporal logic (LTL), we show that this translation can instead be broken down into answering a set of questions with pre-defined answers. We then construct a System-One model, a recent class of NL-based classifier, to provide such answers and enable us to translate a natural language instruction into an LTL instruction. On three benchmarks, our translator matches or exceeds the accuracy of the best local LLMs available, at up to one order of magnitude lower runtime and energy consumption. We show that it generalizes to unseen atomic propositions and temporal patterns. 

---
# From Manufacturing Event Logs to Simulation: Enabling SMEs with Agentic AI for SimPy Simulation Generation 

**Authors**: Richard Reider, Lina Döring, Sebastian Lang, Tobias Reggelin, Sebastian Trojahn, Pascal Reusch  

**Link**: [PDF](https://arxiv.org/pdf/2611.27483)  

**Abstract**: High expertise and cost barriers exclude small and medium-sized enterprises from discrete-event simulation (DES). We introduce GenAI4SimPy, a multi-agent framework powered by Qwen 3.6 (27B), an open-weight large language model, that automates end-to-end DES generation from manufacturing event logs to executable SimPy models through tool-augmented reasoning and human-in-the-loop validation. Evaluated on three execution blocks from a public smart factory dataset, the framework achieved a mean absolute percentage error of 7.9\% and coverage of failure rate for all three blocks and of throughput for two of three. A fidelity paradox was observed: accurate process-level parameters did not ensure system-level behavioral fidelity, with flow-time and throughput errors driven by misinterpreted observation-window truncation and absent system-level calibration. These results demonstrate that local open-weight models can perform DES generation at accuracy comparable to cloud-based alternatives, while identifying parameter extraction as necessary but insufficient for behavioral validity. Source code and results are available at this https URL 

---
# Supervaluationism about vagueness and indeterminacy: A judgment-aggregation perspective 

**Authors**: Christian List, Paul Egré, Franz Dietrich  

**Link**: [PDF](https://arxiv.org/pdf/2611.27463)  

**Abstract**: Supervaluationism about vagueness and indeterminacy is based on the idea that some sentences, or predicates within them, have multiple competing interpretations or specifications. A sentence is deemed determinately true if and only if it is true according to all specifications; and determinately false if and only if it is false according to all specifications. In all other cases, the sentence is assigned no determinate truth-value. Supervaluationism, however, is only one way of taking into account the verdicts of different specifications -- in technical terms, "aggregating" those verdicts. In this paper, we introduce insights from the theory of judgment aggregation to study a broad class of different aggregation functions. This allows us to offer an axiomatic characterization of supervaluationism, which, in turn, locates it precisely within a larger logical space of possible approaches to vagueness and indeterminacy. Among other results, we axiomatically compare supervaluationism with subvaluationism, the epistemic theory of vagueness, and Strong Kleene and LP logics. 

---
# Jagged Robustness: Uneven Safety Transfer and Emerging Vulnerabilities under LLM Quantization 

**Authors**: Ramneek Agnihotri, Yingte Xu, Hans Vath  

**Link**: [PDF](https://arxiv.org/pdf/2611.27424)  

**Abstract**: Quantization enables efficient inference in language models through model parameter compression. This work audits the impact of quantization on capability and safety domains including model safeguards, instruction-following, ethics and bias. Through benchmarks, we find that model safeguards and bias can degrade under quantization, while other capabilities such as truthfulness remain more stable in Qwen3-4B. We define this uneven transfer of LLM capabilities under quantization as Jagged Robustness. We find multiple factors driving Jagged Robustness: task complexity, evaluation method and generation coherence. Moreover, we verify the degradation of model safeguards by observing increased compliance with unsafe requests under quantization, and demonstrate universal adversarial suffix attacks that are benign for the full model but jailbreak quantized models. The findings indicate that it is risky to assume that the performance and robustness can transfer evenly across different capability domains from full models to quantized models. 

---
# Beyond Linear Decodability: Understanding and Improving Instruction Following in World Action Models 

**Authors**: Bokai Ji, Jiapeng Chen, Yi Zhang, Zeyuan Ding, Yong Dai, Xiaozhu Ju, Jian Tang, Lemao Liu, Dejing Dou, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.27401)  

**Abstract**: World action models (WAMs) exhibit strong manipulation capabilities yet often fail to follow test-time language instructions. Prior work shows that non-following (NF) WAMs encode information about the instructed target, but this information does not control their behavior. By comparing NF WAM with its instruction-following (IF) counterpart, we find that target information is substantially more linearly decodable in IF WAMs, despite being encoded in both. Motivated by this observation, we introduce linear decodability supervision (LDS), a simple yet effective auxiliary training objective that promotes the linear decodability of the instructed target information. On LIBERO-CF, InternW0-$\Delta$ with LDS achieves a success rate of 74.6% and a grounding rate of 89.2%, surpassing the strongest baseline by 26.8 and 42.9 percentage points, respectively. The effectiveness of LDS challenges the prevailing interpretation that encoded target information does not contribute to WAMs' behavior. To reconcile this apparent contradiction, we propose a prior-instruction competition mechanism, which explains NF and IF behavior as arising from the dominance of prior and instruction signals, respectively. We provide causal evidence for this mechanism by evaluating two training-free interventions derived from it. We reinterpret LDS through this mechanism, showing that it acts as a form of instruction signal enhancement. Finally, more comprehensive evaluations across diverse manipulation tasks and visual domains demonstrate the consistent effectiveness of LDS. 

---
# Evidence sufficiency assessment enables cost-effective review allocation in on-premise clinical AI agents 

**Authors**: Yiwen Qiu, Che Sun, Shu Shen, Wentao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27390)  

**Abstract**: On-premise clinical AI agents must auto-accept high-confidence diagnoses, deferring others to clinician review, yet token probabilities and self-reported confidence miss errors, while more reliable five-run consistency incurs fivefold compute. We propose training-free decision-point evidence sufficiency assessment and a three-tier sequential triage protocol. Masking reasoning and diagnosis, the same 4B model scores sufficiency from objective evidence via prior-debiased probabilities over five ordered levels. Calibration freezes a nominal top-50% threshold, selecting single-run, five-run consistency, or full-deferral tiers by whether the one-sided 95% Wilson lower confidence bound of pass rate among accepted cases meets the 95% clinical target line. Across 1,605 MIRA-v2, CDM, VivaBench, and in-hospital liver malignancy primary cohort cases, MIRA-v2 single-run top-50% pass rate reached 98.91%, matching five-run consistency, saving ~80% compute; CDM achieved 96.00% test attainment by escalating to the five-run consistency tier; cross-specialty, missing-imaging, and combined enriched cohorts entered the full-deferral tier, capping calibration false-certification rate at 0.00%-3.50%. 

---
# ReActWAM: Action-Consistent Caching for Efficient World-Action Models 

**Authors**: Zhaochen Guo, Yunke Wang, Siyu Xu, Zijian Wang, Chenghao Xia, Chengbin Du, Shinan Liu, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27372)  

**Abstract**: World-action models (WAMs) often generate future visual states alongside robot actions, introducing substantial computational overhead during inference. During closed-loop execution, consecutive inference cycles often predict similar action sequences, leading to redundant computation. However, reducing this redundancy requires distinguishing consistent actions from changes that demand fresh computation. To address these challenges, we propose ReActWAM, a training-free framework that combines action-intent-guided prefix skipping with inner-chunk prediction reuse. A coarse-to-fine assessment evaluates motion changes, visual consistency, and action prediction conflicts. For eligible chunks, we draw on first-step differences to adapt the cached latent and network features to the current condition, providing an estimate at a later denoising step that bypasses the redundant prefix. Subsequent inference refines this estimate while reusing stable denoising predictions within the chunk. Experiments across LingBot-VA, FastWAM and the Cosmos family on LIBERO-Long and RoboTwin 2.0 demonstrate $1.41\times$--$5.44\times$ GPU inference speedup, with minimal degradation in task success rates. Real-world experiments on a Franka Research 3 (FR3) robot further achieve $1.46\times$ end-to-end speedup while maintaining success rate. 

---
# AMARIS: Merging Generalist and Specialist LLMs via Adaptive Subspace Inheritance 

**Authors**: Hwee Young Kang, Irwin De Tao Chin, Wen-Haw Chong, Hai Leong Chieu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27369)  

**Abstract**: Domain specialization of large language models (LLMs) often improves target-domain performance but degrades broad capabilities. Model merging is an attractive training-free alternative, yet common parameter-space strategies (for example, interpolation and sparsified averaging) provide limited control over task-specific interference. We introduce Adaptive Model Alignment via Ranked Inherited Subspaces (AMARIS), a covariance-driven framework that transfers specialist updates only along selected activation directions. AMARIS formulates merging as budgeted subspace inheritance: it seeks high specialist activation capture under a regularized base-output perturbation budget, then builds a low-rank projection gate from ranked generalized eigendirections. This yields a single static merged checkpoint that concentrates transfer in high-utility, low-interference directions. We evaluate AMARIS on Qwen3-based merges (8B and 32B) across finance, instruction-following, and multilingual SEA specialization. In our evaluated settings, AMARIS reaches Pareto operating points that are competitive with strong training-free baselines, including 98.37% specialist retention with 95.16% core retention for the 8B finance setting, 103.55% core retention with 90.28% specialist retention for the 8B instruction-following setting, and 99.20% core retention with 94.80% specialist retention for the 32B SEA setting. These results support subspace-constrained merging as a practical route to compositional capability integration without additional training. 

---
# WePM: Hierarchical Conflict Reconciliation for Unified MLLMs 

**Authors**: Zhiqiang Yuan, Qiwei Yan, Liwei Yang, Jinzhe Liu, Yang Liu, Zhongyan Hou, Liyan Xu, Peixiang Luo, Jiapei Zhang, Ying Deng, Jie Zhou, Jinchao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27365)  

**Abstract**: Bringing multimodal intelligence from the cloud to users' devices keeps sensitive data local and reduces inference cost, but deploying a specialized model per capability duplicates vision-language capacity under tight memory and compute budgets. We formalize heterogeneous capability density (HCD), measuring how broadly and effectively a model serves heterogeneous task families per unit of resident inference memory. High HCD across representation, generation, and decision raises two challenges: these capabilities need visual evidence at different spatial granularities under a fixed token budget, and separately collected task corpora yield conflicting updates to shared parameters, while cross-task paired supervision is costly and ambiguous for open-ended inputs. We present WePM, a family of pocket models unifying representation, generation, and decision within one vision-language backbone. Multi-Granularity Visual Budgeting (MGVB) allocates a fixed visual-token budget across global, regional, and fine-grained banks; Unified Listwise Knowledge Distillation (ULKD) enriches separately collected data with soft teacher guidance, avoiding cross-task paired annotation; Stable and Efficient Multitask Optimization (SEMO) regulates each task's contribution through explicit loss weights and exposure schedules, with sequential gradient accumulation under memory constraints. In the joint representation-generation setting, WePM outperforms all baselines across five evaluation groups, raising the Joint score from 38.15 to 52.74. In single-task evaluation, WePM-4B attains the best average score over 30 embedding, 39 generation, and seven typed-decision benchmarks (72.86, 62.41, 61.11). Against the separate-model stack with the highest HCD, WePM achieves slightly higher balanced performance with 61.9% less resident weight memory, a 2.64-fold higher HCD. 

---
# GRACE: Controlled Evidence Interventions for Diagnosing and Improving Knowledge Integration in GraphRAG 

**Authors**: Minh Pham, Quang Nguyen, Phong Kieu, Cong Tran  

**Link**: [PDF](https://arxiv.org/pdf/2611.27350)  

**Abstract**: GraphRAG combines a language model's parametric knowledge with retrieved graph evidence, yet aggregate accuracy cannot show whether failures stem from missing paths, distractors, or conflicts with model knowledge. We introduce GRACE, a benchmark built by executing LC-QuAD 2.0 queries over Wikidata and applying four controlled interventions to each evidence subgraph: severing a support path, adding type-compatible distractors, attaching high-degree hubs, and replacing the supported answer with a counterfactual entity. Questions and option positions remain fixed across the five states. The validated benchmark contains 4,028 questions and 20,140 question-graph pairs. These comparisons reveal that question-only routing cannot distinguish evidence states, while relevance guidance can improve answerable states yet hurt when the support path is severed: relevant facts may not jointly suffice to answer. We propose QCFSE-SA, which combines typed local attention, global graph attention, and question-conditioned evidence guidance with a calibrated graph-support gate. With the factual context and generator held fixed, it improves controlled-state accuracy from 57.17% to 62.78% (+5.61 points; question-clustered 95% CI [+5.04, +6.17]) and broken-state accuracy from 55.41% to 67.85%. Over-abstention on semantic distractors remains an open challenge. 

---
# EcomWriteBench: A Real-World Benchmark for E-commerce Writing Tasks 

**Authors**: Kaicheng Liang, Jingyi Cui, Hongwei Wen, Feng Wang, Simiu Gu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27349)  

**Abstract**: Large language models (LLMs) are increasingly used throughout e-commerce workflows to generate content that informs purchasing decisions and supports customer interactions. Evaluating these systems requires accounting for user needs, product evidence, structural constraints, and business requirements that are not fully captured by general-purpose writing benchmarks. We introduce EcomWriteBench, a real-world benchmark containing 3,861 queries from 69 e-commerce writing tasks across 13 categories and five languages. Through task-level extraction, cross-task consolidation, and refinement guided by human feedback, EcomWriteBench constructs a shared pool of 37 abilities and identifies the subset applicable to each task. Ability-wise scoring enables us to compare state-of-the-art LLMs on EcomWriteBench and characterize their relative strengths and weaknesses across fine-grained e-commerce writing abilities. We further propose a preference-calibrated weight-learning framework that converts human pairwise preferences into interpretable scoring rubrics through global, category-level, and task-level aggregation weights. This framework provides a reusable approach for translating human preferences into automated evaluation by combining ability discovery, ability-wise scoring, and preference-calibrated aggregation, and can potentially be applied to a broad range of benchmarks beyond EcomWriteBench. On held-out queries, our scoring framework achieves 88.96% agreement with human preferences, compared with 78.54% for conventional LLM-as-a-judge scoring. 

---
# GGDriver: Glance at What Is Shared, Gaze at What Differs for Perception-Free Planning 

**Authors**: Jiawei Sun, Maosen Li, Jiamin Shi, Xiaoxu Zheng, Jiun Tian Hoe, Yechao Bai, Michael Bi Mi  

**Link**: [PDF](https://arxiv.org/pdf/2611.27330)  

**Abstract**: A planner deciding whether to turn left or continue straight needs to inspect different parts of the scene for each maneuver. Yet many end-to-end driving planners evaluate every candidate trajectory against the same compressed scene representation. We let each candidate select the visual evidence used to refine and score it. A global glance first produces trajectory hypotheses from shared scene features. We then project a corridor around each trajectory into the camera views and use the resulting mask to attend to patch features at the backbone's native resolution. After refinement, the scorer projects each final proposal again and ranks it using evidence along its own path. This geometric selection requires only the predicted trajectory and camera calibration. On three planners with different scene representations (learned registers, a perception-supervised bird's-eye-view grid, and a coarse image-token grid), adding this gaze improves planning scores. In ablations, path-aligned patches outperform both a matched number of randomly selected patches and attention over all patches. Using the gaze for ranking yields larger gains than using it for refinement alone. With a ViT-S backbone and no perception labels, GGDriver achieves a rater feedback score of 8.110 on the WOD-E2E test set, 94.2 PDMS on NAVSIM-v1, 48.6 EPDMS on NAVSIM-v2, and a closed-loop Driving Score of 72.18 on Bench2Drive. 

---
# H-MSAE: Hierarchical Sparse Concept Coverage for Efficient and Diverse Instruction Data Selection 

**Authors**: Bohan Cui, Jingyi Cui, Hongwei Wen, Feng Wang, Simiu Gu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27329)  

**Abstract**: Large instruction-tuning corpora contain substantial semantic redundancy, motivating the selection of compact subsets that preserve useful supervision while reducing training cost. Existing diversity-aware selectors often rely on generated semantic tags and explicit graph structures, introducing considerable annotation and preprocessing overhead. We introduce H-MSAE, a hierarchical sparse instruction selection framework that organizes multi-resolution concepts learned by a Matryoshka sparse autoencoder into a coarse-to-fine hierarchy. Quality-weighted, diminishing-return coverage with ancestor propagation promotes high-quality and non-redundant semantic coverage, while pool-relative length constraints regulate subset composition. We motivate this design through a local quality-diversity analysis of subset SFT utility and examine it using example-quality and representation-space diversity diagnostics. On Tulu 3 data pool, H-MSAE achieves the highest Final Avg among the evaluated 50K selectors across scales and initialization. The selected subset also exhibits the highest mean quality and effective rank among the evaluated methods. By avoiding full-pool generative tagging and supporting efficient sparse selection, H-MSAE reduces measured end-to-end runtime by approximately $5\times$ relative to MIG. 

---
# $γ$-Scape: An Agent-Centric World for Mobile Embodied Intelligence 

**Authors**: Xiaotao Zhang, Lin Qian, Hao Zhang, Xuanrui Cui, Yuehao Huang, Yunzi Wu, Yucheng Deng, Jiankun Dong, Hujun Yin, Xinhai Li, Chenjia Bai, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.27306)  

**Abstract**: Recent embodied agents construct and adapt robot behavior through executable programs, tool use, and interaction feedback. Studying these systems in mobile manipulation calls for physical worlds that connect agent-directed execution with its physical consequences and reusable interaction experience. We introduce $\gamma$-Scape, an agent-centric simulation and benchmark platform for mobile embodied intelligence. Agents compose motion tools and learned skills to carry out complete tasks that combine navigation and manipulation in a persistent physical world. BenchForge grounds structured objectives in physical task instances with instructions, reference programs, and goal evaluators, defining task goals while preserving agents' choice of valid solutions. An agentic collection harness further enables agents to produce demonstrations by carrying out contact-rich interactions through sustained control and physical feedback. Assisted reference execution and human guidance provide complementary experience. The resulting tool traces and aligned observation-action records support program refinement and policy learning, with common task-goal criteria for subsequent evaluation. We benchmark representative agent systems on complete tasks and demonstrate contact collection and trace-informed program revision. Together, these workflows provide infrastructure for research on embodied recursive self-improvement. 

---
# HookEvo: A Framework for Coordinated Plugin Evolution on Hook-Enabled Agent Runtimes 

**Authors**: Shengze Huang, Yanyu Mao, Xiaofei Yu, Baoshen Guo, Xiaorui Wang, Zhiwen Wu, Sheng Wu, Fukang Wen, Chengxu Wu, Linhao Li, Kailai Sun, Baotong Mi  

**Link**: [PDF](https://arxiv.org/pdf/2611.27294)  

**Abstract**: Repairing failures in tool-using agents can require coordinated changes to how the harness interprets tool receipts, assembles context, and checks actions. We introduce HookEvo, a framework that uses an existing agent runtime's lifecycle hooks to evolve task-specific harnesses. We instantiate HookEvo on DeepSeek Harness. In each round, a separate improver diagnoses failures from execution traces and proposes a candidate. HookEvo evaluates the candidate against the current harness on matched tasks and carries the retained version into the next round. A single candidate can jointly edit executable plugin handlers and the state they share within a task, allowing a tool receipt to inform context and control a later action. Using a frozen Qwen3-4B executor, the selected harnesses lead the reported Qwen3-4B comparisons, achieving 95.41% on ALFWorld, 70.56% on DBBench, 48.17% on WebShop, and 41.67% on OSBench. Across three improvers and three edit scopes, from skill text to single-component and coordinated executable edits, GPT-6.1 Sol shows the largest average gain from the broadest scope. Our code is available at this https URL. 

---
# Belief-State Management Meets Rubric Rewards for Agentic Forecasting 

**Authors**: Jingyi Chai, Yun Liao, Hao Lu, Kemeng Zhang, Bingyang Zheng, Yujie Ge, Xiangrui Liu, Linfeng Zhang, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.27286)  

**Abstract**: Current large language model (LLM) agents still struggle to forecast real-world events. This stems from two primary limitations: i) standard ReAct harness accumulates reasoning and evidence within one growing context, and ii) outcome-based rewards in forecasting lead to a slow and noisy signal that cannot distinguish sound reasoning from lucky guesses. To address these, we introduce a forecasting training framework including harness design and post-training. For the harness side, we propose Proactive Belief-State Management (PBM), equipping agents with a fold tool to compress lengthy context into compact belief states and reset the context for subsequent exploration. For the training side, we develop SEIPH, a five-dimensional, outcome-blind rubric, providing process-quality rewards. Using the rubric-based reward, we post-train a 35B model with PPO and GRPO, yielding DeepForecaster-PPO and DeepForecaster-GRPO. Theoretically, we prove that idealized fold-based stopping converges and calibrated local rewards reduce segment-level gradient variance. Empirically, DeepForecaster-PPO (35B) achieves the best Brier score on both ForecastBench and Polymarket-Sim, outperforming frontier agents based on DeepSeek-V4-Pro (1.6T), Kimi-K2.6 (1T), and GPT-5.5. DeepForecaster-GRPO ranks second on both FutureX metrics. Ablations reveal that PBM also improves the untrained base model without any training. Calibration analysis shows that DeepForecaster-PPO makes more accurate high-confidence predictions than Qwen3.6 and GPT-5.5. Trajectory analysis further shows that forecasts generally improve from the first to the final belief state as agents gather evidence and revise their judgments. 

---
# DermKG: A Concept-Centric Knowledge Graph for Interpretable Skin Disease Diagnosis from Dermatological Morphology 

**Authors**: Cheng Huang, Yanmei Li, Mao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.27283)  

**Abstract**: Dermatologists describe lesions with a standardized morphological lexicon, which makes concept-based models attractive for interpretable skin disease diagnosis. How much diagnostic information these concepts actually carry, however, is rarely quantified. We present DermKG, a concept-centric knowledge graph built on the expert-annotated SkinCon subset of Fitzpatrick17k (3,230 images, 48 concepts, 113 diseases). DermKG has 3,452 nodes of 8 types and 19,365 edges of 11 relation types, and encodes disease concept profiles, concept co-occurrence, a three-level diagnostic hierarchy, and 20 clinical morphology rules, all of which are supported by the training data (15 strong, 5 moderate). An information-ceiling analysis shows that the 3,230 lesions collapse onto only 813 distinct concept vectors, bounding concept-only top-1 accuracy at 38.9\% for diseases and 77.6\% for the nine categories. Within this bound, DermKG-Net, a graph neural network that scores lesions against disease prototypes and aggregates them along the hierarchy, raises category-level balanced accuracy from 0.257 to 0.411 over the strongest of 11 baselines (paired bootstrap difference $+0.150$, 95\% CI $[0.073, 0.231]$) and more than doubles minority-category recall. Ablations show that most of this gain comes from the hierarchical prototype objective, with a smaller contribution from graph structure. DermKG further yields rule-based reasoning chains and concept-level explanations for individual predictions. 

---
# No Perfect Guardrail & No Perfect Guard: On the Decidability of Output Policies for Large Language Models 

**Authors**: Ahmed Amer, Loren Kinczel, Darrell D. E. Long  

**Link**: [PDF](https://arxiv.org/pdf/2611.27260)  

**Abstract**: Expecting guardrails to provide a general guarantee of safety fails to recognize the limits of computational enforcement and policy specification. We argue that promising such perfection is an ethical failure of precision, and that invoking it to abdicate responsibility is a moral failure. Guardrails can support responsible action, but cannot relieve researchers, developers, policymakers, or users of their obligations to examine what they claim, permit, and prevent.
We establish a precise computational boundary. If every finite string is a possible output of a language model, a computable guard that blocks every forbidden output and passes every permitted one exists if and only if the policy is decidable. For an undecidable policy, every guard misclassifies infinitely many strings, and every guard that blocks all forbidden outputs excludes infinitely many permitted ones. We apply this characterization to arithmetic truth, program behavior, and harmful instructions to an idealized recipient. We distinguish guards from execution monitors, identify bounded and interceptable cases admitting exact enforcement, and explain why the results establish no positive lower bound on average error.
Exact enforcement does not establish that a policy adequately captures safety. Conservative exclusion may withhold warnings, remedies, or challenges needed to recognize the policy's own defects. Human oversight cannot substitute for judgment: an apparent controller can become an instrument of purposes they have ceased to examine. Responsible governance requires continued critical challenge and the intellectual and moral humility to recognize possible flaws in our machines, rules, policies, and ourselves. The impossibility of a general assurance of perfection is a condition responsible practice must confront, not a responsibility that another guardrail - or another guard - can remove. 

---
# LiDAR Signals for Robot Locomotion: Spherical Range Images Enable Robust Sim-to-Real Transfer 

**Authors**: Nian Liu, Xinyi Tong, Ming Ren, Long Ye, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2611.27223)  

**Abstract**: Existing legged-locomotion systems commonly rely on depth images or online elevation maps. However, depth-based policies suffer from perceptual sim-to-real discrepancies due to imperfect depth sensor modeling, while map-based approaches are vulnerable to drift and misalignment. LiDAR offers direct measurements of the surrounding geometry, but raw LiDAR scans lack a compact, structured representation for high-frequency control. We propose a LiDAR perception pipeline that maintains perceptual consistency across simulation and reality through sensing- and representation-level alignment. At the sensor level, we recover frame-wise angular sampling patterns from the physical LiDAR and replay them in simulation with GPU ray casting. At the representation level, simulated and real returns undergo the same spherical range image encoding, yielding a fixed-size policy observation for end-to-end training and direct deployment without teacher--student distillation, online elevation mapping, or adaptation. Experiments show that our range-image policies achieve stronger sim-to-real transfer than depth- and elevation-map-based baselines, while ablations confirm the importance of the proposed sensing components. More visualizations and real-world videos are available at the \href{this https URL} 

---
# Bel: Step-Level Policy Improvement of Visual Agents Without Training 

**Authors**: Jiajun Fan, Ziwen Wang, Zhaoyu Zhu, Changyi Yang, Neil He, Chaoran Cheng, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27222)  

**Abstract**: A visual agent acts on what it sees: it taps and types on a screen, or points on a camera image to show a robot where to act. One wrong step can fail its task, yet the right action is often among its own samples, passed over by its greedy decode. Choosing it is a visual judgment, since an action is best judged where it acts, not by its text or coordinates alone: as text, a tap is two numbers; drawn on the screen, it is the Post button of an empty draft. We find that this is decisive: given the screenshot and a tap's coordinates, a decision model chooses among GUI-R1-3B's samples significantly worse on average than when the tap is drawn where it lands. Bel turns this finding into a method that trains nothing and needs only a frozen policy's samples and an image on which each action can be drawn. It marks each candidate on its own copy of the image it acts on, asks a decision model for the probability that executing it is the right decision, so that every kind of action shares one scale, and acts on another candidate only when greedy scores below a threshold and trails it by a margin. On recorded GUI steps, for each of three policies, Bel significantly beats greedy decoding on mobile, web and desktop screens alike and, on the same candidates, gives each a significantly higher mean step success than voting, CATTS, the trained GUI-Critic-R1 and two prompted Qwen3-VL judges shown the same marks; it also significantly improves the much larger Qwen3-VL-235B-A22B on every step-level test set. With GUI-R1-3B's thresholds unchanged, Bel significantly raises live AndroidWorld and MiniWoB++ task success for two other policies and robot-pointing success for three policies on each of two groups of benchmarks. Judging each action where it acts thus gives step-level policy improvement of visual agents without training, with significant gains for GUI agents and robot pointing. 

---
# EDAGoat: An EDA-Aware Agent for Project-Level GPU Optimization 

**Authors**: Chengxi Li, Liang Xiao, Yutao Wang, Wenhao Lin, Xinzhuang Xiong, Han Zhao, Evangeline Young  

**Link**: [PDF](https://arxiv.org/pdf/2611.27196)  

**Abstract**: GPU acceleration has transformed electronic design automation (EDA), yet further improving expert-engineered GPU tools still demands substantial manual effort. Automating this work calls for coding agents that can trace bottlenecks across host and device code while preserving correctness and design quality. We introduce EDAGoat, to our knowledge the first EDA-aware GPU optimization agent for this task. Its optimization enhancement (OE) mode guides edits with four cross-project GPU optimization patterns. Its mandatory validation (MV) mode requires the agent to test the edited program against the EDA constraints its changes could affect. Our benchmark covers five GPU EDA projects across four task families and 55 circuit inputs, measuring complete-solve runtime and verifying the resulting designs. Each configuration runs three independent searches per project. A search qualifies only if its solver passes correctness and design-quality checks on all public and hidden inputs. With DeepSeek, generic prompting qualifies in 7 of 15 searches and OE in 6 of 15, despite a higher best speedup on four projects. Adding MV raises this to 13 of 15 and, on every project, exceeds both generic baselines in mean and best speedup and OE in mean speedup; the geometric mean of per-project mean speedups rises from 1.11x (1.36x with OE) to 1.53x. With GPT-6 Astra, EDAGoat qualifies in all 15 searches and improves both speedup metrics over generic prompting on every project, lifting the same geometric mean from 1.26x to 1.57x. These results show that agents can use EDA knowledge to find further speedups in expert-tuned tools while preserving design correctness and quality, establishing EDA-aware GPU optimization as a new direction for agent research. 

---
# What Does Matching a Human Intervention Effect Establish? Benchmark Checks for LLM Simulations 

**Authors**: Tengfei Shao, Asma Rezaei, Ali Reza Kamali  

**Link**: [PDF](https://arxiv.org/pdf/2611.27186)  

**Abstract**: LLM simulations are validated by equivalence with human effects, yet a zero reference passes whenever the human 90 percent interval lies inside the margin, as the human benchmark shows before simulation. When the zero reference fails, planned precision gives independent simulations the margin up to which passing is a 5 percent test of response. In an exploratory 70-experiment reanalysis, 72 percent of GPT-4's $\pm$0.20 SD passes were zero-reference passes; GPT-4 passed 16 contrasts the zero reference failed (1.58 expected without response). In a preregistered audit, Claude Sonnet 5.5 passed three intention targets, all zero-reference passes, and the registered belief-gradient control; norm beliefs changed 0.065 SD against humans' 0.351 (zero-reference check added after unblinding; norm beliefs emphasized after inspection). Registered tests detected no overestimation; average-effect intervals excluded overestimation above 0.08 SD. A group audit's precision floor is 1.33 facts against a 2.9-fact effect. Report the zero-reference threshold beside every equivalence decision. 

---
# Learning Adaptive Defense Strategies: Fine-Tuned LLMs for Network Security Games under Network Mobility 

**Authors**: Md Abu Sayed, Asif Rahman, Ahmed Ryen, Ahmed Hemida, Charles Kamhoua, Christopher Kiekintveld  

**Link**: [PDF](https://arxiv.org/pdf/2611.27180)  

**Abstract**: Large language models(LLM) show promise for strategic reasoning, yet their ability to compute game-theoretic equilibria under dynamic conditions remains largely unexplored. This work investigates whether fine-tuned LLMs can serve as solver-free surrogates for online equilibrium computation in honeypot allocation, a setting in which network mobility continuously reshapes the underlying attack graph and forces repeated re-solving of a two-player zero-sum game. We construct a dataset of 270 security games on networks of 8--25 nodes with solver-grounded equilibrium labels, and evaluate four classes of predictor over 1{,}524 held-out mobility events: non-learned baselines, six open-weight architectures evaluated zero-shot, the same six after LoRA fine-tuning, and four proprietary models as an external reference point. Predictions are scored without post-hoc repair against exact game-theoretic criteria: exploitability against an optimal attacker best-response, equilibrium support accuracy, and distributional distance. Fine-tuning reduces exploitability across every architecture, and fine-tuned open-weight models of modest size outperform every zero-shot predictor, including proprietary ones; no zero-shot model at any tier reaches the floor set by uniform allocation. Stratifying by mobility type exposes a qualitative shift that aggregate metrics conceal: zero-shot models do not condition on whether a departing node lies on an attack path, while fine-tuned models recover this distinction. We further find that validation loss is negatively rank-correlated with task performance, and that output validity, rather than predictive accuracy, is the binding constraint on further progress. 

---
# From Acting to Learning: Experience Selection toward Continually Learning Agents 

**Authors**: Yaoyuan Yan, Haoxiang Jie, Zhiyou Heng, Kailin Wang, Yumiao Li, Lifeng Liu, Anwan Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.27159)  

**Abstract**: Successful agent interactions provide training data, but success alone does not determine which records a student should learn from. We examine experience selection for a 4B language agent as a component of continual agent learning. Nine selectors are compared across two fixed experience pools, eight initialization seeds, and two sampling seeds, yielding 288 trained students. Each student is evaluated on 128 development task groups with a common executor. With equal global selection freedom, 256 distinct contents, and 768 training presentations, optimizer-aware gradient selection reaches 81.45\% and 73.19\% success in the two pools. Its gains over global random selection are 8.25 and 5.96 percentage points; gains over global high-NLL selection are 4.10 and 4.05 points. All eight initialization-level contrasts are positive against both baselines in each pool. Relaxing coverage constraints improves gradient selection but reduces random-selection performance, showing that the ranking rule and selection space must be considered together. A separately frozen replication on 128 reserved task groups is complete; its equal-source gradient-minus-random and gradient-minus-high-NLL differences are +7.03 and +5.03 points, respectively. 

---
# Where Structure Enters a World Model: From Predictive Fit to Physical Rollout 

**Authors**: Panayiotis Christou, Satyapragnya Kar, Flavjo Xhelollari, Parijat Dube, Rajarshi Das  

**Link**: [PDF](https://arxiv.org/pdf/2611.27149)  

**Abstract**: Predictive world models learn to forecast observations, but accurate short-term predictions do not necessarily produce physically coherent behavior over longer horizons. We investigate how incorporating physical structure can strengthen world models while retaining their underlying predictive objectives. Across token-based models of planetary motion and JEPA-style visual models of mechanical systems, we study conserved-quantity supervision, symmetry-aware representations, temporal consistency, and learned structured dynamics. In planetary motion, angular-momentum supervision improves autonomous next-token forecasting, while the benefits of temporal regularization depend on the accompanying physical objective and representation. In visual models, generating rollouts through supervised energy-based dynamics heads produces more faithful trajectories and phase-space evolution. Together, these findings highlight the importance of how physical structure participates in prediction: improvements in an auxiliary physical readout or a local acceleration estimate do not automatically translate into more reliable long-horizon behavior. Evaluations across initial conditions and physical parameter shifts further reveal where these benefits persist and where generalization remains limited. Our study supports physical structure as a practical means of improving predictive world models and provides a unified framework for understanding its effects on representations, learned transitions, and autonomous rollouts. The central lesson is that effective physical inductive biases should help models both represent physical regularities and use them to generate future states. 

---
# Double counting in LLM evidence fusion: exact posteriors separate rule compliance from posterior gain 

**Authors**: Tengfei Shao, Asma Rezaei, Ali Reza Kamali  

**Link**: [PDF](https://arxiv.org/pdf/2611.27138)  

**Abstract**: Language models fuse reports in retrieval, multi-agent consensus and agent memory, where treating shared-source reports as independent can double count evidence. A common remedy is an instruction to count each source once. Its value depends on the receiver's raw baseline. Pricing it needs an exact reference. We build the graded-dependence exact reference: binary worlds with single-report reliability 0.65 and four dependence levels, whose disclosed parameters give every world an exact posterior and every rule a target risk. The paired gain protocol scores raw and instructed answers of four configurations on 2,976 worlds. Agreement and gain separate in both directions. At $\delta$ = 0, claude-opus-5-5 and gpt-6.1-sol agreed with the rule within 0.001 in 744 of 744, and the instruction raised their posterior mismatch by 0.044; gpt-6-luna agreed in 165 of 744 and gained 0.0140 (95% interval 0.0064 to 0.0219); claude-haiku-5-5's gain moved from -0.0295 ($\delta$ = 0) to +0.0302 ($\delta$ = 1) (descriptive). The prespecified first-child endpoint contrast, gain at $\delta$ = 1 minus $\delta$ = 0, was 0.0342 (0.0303 to 0.0382). In the preregistered tests, the raw source-weight contrast was positive (0.259, 95% interval 0.182 to 0.337; Holm rejected zero) and the grouping endpoint difference was not detected (0.0004, -0.0023 to 0.0031). This interval excludes differences larger than 0.0031 in magnitude and lies within the registered precision bound of $\pm$0.0046. The reference prices the rule before any model call. The paired gain, estimable as a Brier difference on labelled deployment items, decides adoption per receiver and prompt condition. 

---
# MAPES: Multi-Value Alignment Pipeline via Expert Specialization 

**Authors**: SungHo Kim, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.27133)  

**Abstract**: Aligning large language models (LLMs) with human preferences requires simultaneously satisfying multiple, often competing values. Existing preference optimization methods collapse these into a single implicit signal, and even multi-objective extensions optimize within a shared parameter space, leaving cross-value conflicts unresolved. We propose MAPES, a multi-value alignment paradigm in which conflict resolution is a first-class concern at every pipeline stage: it first decomposes supervision into value-specific experts trained in isolated parameter subspaces, then recomposes them via preference-aware value merging and a lightweight harmonization stage that aligns the merged model with the expert ensemble. Across five preference alignment benchmarks, MAPES outperforms strong direct preference optimization (DPO) variants and multi-objective baselines on the benchmark-level average, with no large degradation on GSM8K and GPQA; matched controls isolate the contribution of each stage. 

---
# When Students Fail to Learn from Teachers: Learning Dynamics of On-Policy Distillation 

**Authors**: Xinnan Zhang, Zeyi Liao, Siliang Zeng, Zhongruo Wang, Jiaxiang Li, Xinyan Zhao, Huan Sun, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2611.27125)  

**Abstract**: On-policy distillation (OPD) has become a core technique for post-training large language models, enabling students to learn from dense token-level supervision on their own generations. However, even with a capable teacher, OPD training can be unstable, leading to excessive response length, repetitive generation, and severe performance degradation. One example is \textit{spurious agreement}, where the student maintains high overlap with the teacher at the states visited but produces repetitive generations. Although such failures have been observed, the mechanisms behind these failures remain poorly understood. To address this gap, we investigate OPD through the lens of learning dynamics, tracing three mechanisms from local analysis to subsequent on-policy sampling. We first identify states where sampled-token OPD updates fail to correct teacher--student mismatch and characterize them using the student's mass on the teacher's top-$p$ set. We then show that learning from the teacher is difficult at mismatched states and that their gradients can also affect with learning at other tokens. Finally, using a termination-token mismatch as a case study, we study how continued training on mismatched states influences on-policy sampling, suppressing the student's current modes and leading the student toward states where the teacher is unreliable . These findings help explain the benefits of supervised fine-tuning (SFT) initialization and motivate targeted correction of states where student and teacher are mismatched. 

---
# Portable Active Recovery under Timed and Position-Dependent Visual Outages 

**Authors**: Jinyuan Zhang, Shengshuo Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.27121)  

**Abstract**: When a camera blackout depends on physical position rather than elapsed time, stable hovering prevents visual recovery and mission completion. On stereo visual-inertial quadrotors without external localization, an exogenous timed blackout and an endogenous position-dependent dark zone appear identical at loss onset yet require opposite responses: hover-and-wait completes 93/96 timed blackouts but 0/96 spatial outages (13 hard failures), whereas blind continuation traverses many spatial zones yet fails heavily during timed blackouts. We formulate visual-outage recovery as a worst-condition active-observability problem and introduce a portable recovery module at the reference interface of an unchanged host controller with zero retraining. Using deployable telemetry alone, the module coordinates visual-loss detection, jerk-limited braking, bounded backward probing, settled rejoining, height-integral trim, and an optional gated prior-map branch. On a preregistered 48-scene benchmark (1,920 flights), our full MPPI configuration reaches 88.5% [81.2, 94.8] worst-condition completion, outperforming development-selected DATT-L1 by +29.2 pp [17.7, 40.6] and exceeding all six external baselines after Bonferroni correction. Attached without retraining to SimpleFlight, the module lifts worst-condition completion by +17.7 pp [7.3, 28.1] to 86.5% (and 90% across 1.0x/1.25x speeds, 0/80 hard failures). Trace diagnostics reveal that survival hinges on visual-inertial re-acquisition asymmetry at vision return -- not blind IMU drift suppression - and uncover a bias-drift map crossover confirmed on 72 held-out EuRoC windows. 

---
# CoEvoBench: Can LLMs Drive Recursive Co-Evolution of Agentic Systems? 

**Authors**: Chuan Xiao, Baojie Wang, Haizhao Fan, Hongzhe Du, Xander Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27116)  

**Abstract**: The performance of an LLM agent depends on both its model and its harness, the code that mediates its interaction with tools and environments. Can LLM researchers drive this co-evolution as a recursive improvement process, deciding round after round which component to change when each change alters the value of the other? Existing benchmarks evaluate post-training or data research with the harness out of scope, or harness optimization with the model fixed, leaving this joint decision unmeasured. We introduce CoEvoBench, a benchmark that evaluates LLMs as researchers co-evolving the model and harness of a target agent across five agentic domains. Researchers start from a common base model and initial harness, train and revise within a 24-hour budget, and submit a model-harness pair that is scored on separate test tasks. Across five frontier researchers, the strongest closes more than four fifths of the gap to an instruction-tuned reference system, and co-evolution succeeds on interaction tasks but rarely on long-horizon coding. These results suggest that current LLMs can make a strong first improvement, but cannot yet turn feedback from an improved system into the next one. CoEvoBench provides a controlled testbed for measuring the research capabilities that recursive co-evolution requires. 

---
# Reasoning in Two Dimensions: Quantifying Chain-of-Thought Faithfulness and Efficiency 

**Authors**: Chenruo Liu, Quan Zhang, Qi Lei  

**Link**: [PDF](https://arxiv.org/pdf/2611.27100)  

**Abstract**: Chain-of-thought (CoT) reasoning exposes intermediate steps, but a correct answer does not establish that an LLM relies on the dependence expressed by those steps or reasons efficiently. We operationalize CoT faithfulness and efficiency within a common framework. Specifically, we represent a CoT as a semantic reasoning graph and decompose the model's predictive dependence along two complementary axes: dependence explained by semantic support versus additional reasoning history, and dependence allocated to answer-reaching versus off-path reasoning. This decomposition yields global and step-level measures of unfaithfulness and inefficiency. Controlled interventions show that the proposed metrics respond to hidden dependence and induced off-path reasoning as intended. Across mathematical and multi-hop reasoning tasks, answer-level unfaithfulness is consistently associated with greater predictive uncertainty, whereas both unfaithfulness and inefficiency exhibit weak or inconsistent associations with answer correctness. We further derive theoretically grounded, graph-aware fine-tuning objectives that improve the targeted reasoning properties without directly optimizing the computationally expensive metrics or compromising answer accuracy. Results indicate that accuracy and brevity provide incomplete assessments of reasoning quality: reliable CoT evaluation should also consider whether the LLM's predictive dependence is faithfully represented and efficiently allocated. 

---
# Writing Weights or Training Them? Spatial Reasoning through Weight-Space Programming and Low-Rank Adaptation 

**Authors**: Yalin Yang, Xiao Huang, Yanan Wu, Hao Yang, Ryan Zhenqi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.27089)  

**Abstract**: Weight-space programming lets a language model inspect and edit another model's weights, offering an alternative to gradient-based adaptation. We evaluated both approaches using Qwen3.5-0.8B on a fixed reference-frame task (128 items), StepGame (200 items), and two untrained distance-comparison tasks. With thinking enabled, the public patch (p50c) raised mean accuracy from 0.0% to 22.1% on the reference-frame task and from 19.2% to 33.7% on StepGame across three sampling seeds. Randomly permuting its weight change removed the gain. Further closed-form edits did not improve on p50c, whereas a rank-1 LoRA adapter trained on worked solutions reached 100% and 98.8%, respectively, with 0.09% trainable parameters. Removing the worked solutions from supervision prevented the adapter from passing the learning gate. The trained adapters reached 82% accuracy on unseen StepGame sentence templates, 42% on held-out difficulty levels 6-10 after training on levels 1-5, and 33-50% on unseen heading-answer combinations. Their target gains coincided with a sharp decline on the distance tasks: accuracy fell from 94% and 92% under p50c to 0-2%, and their outputs applied the StepGame solution template to distance questions. With thinking disabled and reasoning requested in the visible answer, the base model also solved the distance tasks while remaining weak on the target tasks. The comparison reveals a trade-off between improving spatial task performance and preserving neighbouring skills: the public edit restored completion while preserving distance performance, whereas supervised adaptation produced much larger target gains together with interference across tasks. 

---
# Decision Models Are General-Purpose Process Reward Models 

**Authors**: Jiajun Fan, Qinglong Zheng, Ziwen Wang, Bangji Yang, Neil He, Ruihan Guo, Chaoran Cheng, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27068)  

**Abstract**: Reinforcement learning (RL) from verifiable outcomes gives a language agent one reward per episode: every step is credited alike, and nothing about the task is learned when all rollouts fail. The last mile, step-level credit, usually requires step labels, a learned critic, costly LLM judges or hand-written rules. We introduce DM-PRM, which turns a general-purpose decision model -- one answering yes/no questions with probabilities -- into a general-purpose process reward model (PRM), zero-shot: no step labels, no fine-tuning, no critic. DM-PRM asks one question per step and adds the standardised answers to the advantage. It plugs into any policy-gradient method; one model, with one question text per environment, serves household tasks, web shopping and multiple RL methods. With Qwen2.5-1.5B it lifts GRPO by over twenty points on ALFWorld and over ten on WebShop, overtakes GiGPO on ALFWorld and learns in far fewer iterations; it also improves GiGPO and GraphGPO on both benchmarks and reaches the highest success of our comparison, ahead of critic-based PPO and of state-of-the-art GiGPO, HGPO and GraphGPO. Its scores locate useful steps in failed rollouts, and trained agents drop idle actions and explore instead of circling. DM-PRM shows that step credit need not be learned: it can be asked for. 

---
# Dynamic Token Merging for Efficient Vision-Language Models 

**Authors**: Chris Gu, Marco Andono Sie, Nathan Zhou, Christopher Potts, Julie Kallini  

**Link**: [PDF](https://arxiv.org/pdf/2611.27064)  

**Abstract**: Vision-language models (VLMs) represent each image with hundreds of visual tokens, which can make inference costly even when a prompt depends on only part of the image. Dynamic Token Merging (DTM), introduced in MrT5 for byte-level text encoders, learns which tokens to remove: a lightweight delete gate scores contextualized hidden states, acts as a differentiable attention bias during training, and physically removes tokens at inference. We argue that DTM is a general token deletion framework rather than a byte-level technique, and extend it to the vision modality and to decoder-only models. Applying DTM to T5Gemma 2 and Gemma 3 yields V-MrT5Gemma2, which deletes visual tokens inside a bidirectional multimodal encoder, and V-MrGemma3, which compacts the prompt and its key-value caches during decoder-only prefill. Across ten benchmarks, DTM's advantage over attention-based and random deletion grows with the deletion rate and is largest on document, chart, and scene-text reading. At 80% visual token deletion (approximately 60% fewer FLOPs), DTM raises accuracy over FastV from 51.8 to 63.0 on ChartQA and from 63.3 to 67.8 on TextVQA with V-MrT5Gemma2, as well as from 62.2 to 68.8 and from 71.2 to 75.1 with V-MrGemma3, retaining 90-96% of unpruned performance on both tasks with significant efficiency gains. These results support DTM as a general framework for efficient transformers. 

---
# A Continuous Relaxation of Quantization-Aware Training 

**Authors**: Kartikeya Bhardwaj, Mohammed Almousa, Chaitanya Mullapudi, Ananth Balasubramanian, Nithin Ramesan, Tuc Nguyen, Harris Teague  

**Link**: [PDF](https://arxiv.org/pdf/2611.27056)  

**Abstract**: Quantization is a critical piece of the deployment pipeline in the era of large generative models. Low-bit quantization, particularly 2-bits or sub-2-bits, is a notoriously difficult problem, particularly when using a channelwise quantization scheme which has been shown as a leading candidate for hardware-efficiency. The key challenges with low-bit quantization range from limited quantization levels to saddle points or flat regions in the loss surface. In this paper, we look at the problem of flat quantization loss surfaces through a lens of dynamically changing bitwidth. Unlike existing techniques which either change the bit-precision abruptly or keep it fixed, we propose to gradually alter the bitwidth as the model trains. To this end, we propose RelaxQAT, a continuous relaxation of the QAT process. We also provide simple theoretical and empirical insights behind why RelaxQAT helps the QAT performance. Our extensive experiments demonstrate up to 6.3% average accuracy gains on vision and up to 7.7% better average accuracy on natural language tasks, compared to the leading QAT methods. 

---
# Actions Speak Louder than Words: Stateful Privacy Sniffing of LLM Agents via Behavioral Covert Channels in Web Interactions 

**Authors**: Ruichao Liang, Wenzhong Deng, Jing Chen, Yuqing Li, Kaixuan Li, Cong Wu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27046)  

**Abstract**: LLM agents increasingly rely on Web interaction, issuing searches, following links, and reading pages, to complete complex user tasks. Since each action is driven by the agent's private context, including the user's prompt, dialogue history, memory, and connected data, it also opens a new privacy exposure surface. Prior attacks either passively analyze network traffic, which yields only coarse correlations, or hijack the agent through indirect prompt injection, which must place sensitive content in an outgoing field and is blocked by content-level defenses. Whether an attacker can actively recover fine-grained private context without any explicit transmission thus remains insufficiently understood.
Our insight is that an agent's benign Web decisions themselves leak private context, as what it chooses to visit encodes whether, which, and in what order it acts, forming a behavioral covert channel. Building on this, we design SNOOP, a stateful hierarchical probing framework that narrows a coarse-to-fine hypothesis tree per task and reconstructs user queries and profiles from the agent's observable paths. Across four agents and three LLM backbones on 500 StackExchange queries and 378 multi-turn iKAT dialogues, SNOOP recovers query concepts with an average F1 of 0.89 and user profiles with an F1 of 0.80, while four representative agent defenses change its recovery by less than 0.03 F1. Since no sensitive content is ever transmitted, content-level inspection cannot detect this leakage; we therefore propose Selection-Obfuscated Retrieval (SOR), which acquires resources under a fixed budget independent of private context and confines the agent's selection to a local snapshot, cutting SNOOP's F1 to 0.39 without additional LLM inference. 

---
# Disentangling Verbalized Evaluation Awareness and Refusal through Neuron-Level Intervention 

**Authors**: Jingyuan Feng, Gouki Minegishi, Andrew Gambardella, Takeshi Kojima, Yusuke Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2611.27032)  

**Abstract**: As large reasoning models (LRMs) become more capable, reliable evaluation becomes essential. LRMs can sometimes recognize that they are being evaluated, a phenomenon known as \emph{evaluation awareness}. This phenomenon raises concerns about whether safety behavior, particularly refusal behavior, generalizes from evaluation to deployment. Prior work manipulates \emph{verbalized evaluation awareness (VEA)} through activation steering and observes higher refusal rates when VEA is present, but the mechanisms underlying this connection remain poorly understood. We identify \emph{VEA neurons} by contrasting their activations in VEA and non-VEA reasoning segments. Intervening on 5,000 VEA neurons reduces the VEA rate from 12.6--51.8\% to at most 1.4\% across the model benchmark combinations in our main experiments, and lowers a probe score for unverbalized evaluation awareness (unVEA). In QwQ-32B, VEA-neuron intervention also reduces refusal even in rollouts without VEA; VEA text itself explains only 4--10\% of the decrease. We further show that their down-projection vectors contain refusal components, and preserving these components during intervention partially restores refusal while maintaining VEA suppression, accounting for about half of the refusal decrease and providing a mechanistic explanation for the connection between evaluation awareness and refusal. Finally, across the OLMo 3 SFT, DPO, and RLVR checkpoints, the VEA rate on SorryBench and StrongREJECT is similar after SFT and DPO but substantially higher after RLVR, alongside an increase in the number of VEA neurons. Together, these findings show that VEA text itself has a limited effect on refusal, and that VEA neurons integrate evaluation awarenss and refusal. 

---
# Integral Multipliers and Arithmetic Obstructions for Two-Hidden-Layer ReLU Maxima 

**Authors**: Pengcheng Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.27018)  

**Abstract**: We study exact representations of maximum functions by ReLU networks with two hidden layers and integer parameters. The least positive output multiplier is two for the six-input maximum, and either two or six for the seven-input maximum. We obtain these bounds from integral cubical fillings and a coefficient-preserving passage from polytope identities to networks. For the coordinate-facet complex of the six-simplex, we compute a coning index of six over the integers and three after inverting two, allowing arbitrary integral directions within each facet. A class in the homology of the unrestricted kernel complex characterizes the remaining seven-input problem; its vanishing is equivalent to an integral filling of a fixed target. We also construct a finite character that separates rational from three-local representability in a dictionary containing two known constructions and all pairs of edge-disjoint three-edge forests. This obstruction permits nonsymmetric coefficients and is confined to the specified dictionary. Control of lower-dimensional remainders yields a sufficient criterion in the top-dimensional quotient. Exact certificates and verification programs accompany the finite calculations. 

---
# Same Facts, Different Decision: Fact-Invariant Manipulation and Provenance-Based Defense in Typed Decision Models 

**Authors**: Yujie Wang, Haotian Chi, Shunrong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27016)  

**Abstract**: Typed decision models such as TypeSafe's Jev answer only with a probability, an option, or a score, so software can act on them directly. Their inputs, however, often include text from outsiders. Prior attacks test this risk poorly. Injected instructions rarely work on these models, and added facts can make a changed decision correct. We show that outside comments can persuade these models to make a wrong decision without adding any fact. We propose LEVER, an attack that writes such comments with four levers from argumentation theory, uses a separate verifier to discard any comment that adds a fact, and searches for effective comments guided by the model's output probabilities. We further propose the Provenance Gate, a defense that accepts a changed decision only if the model reaches it again without the outside comments. On three benchmarks, LEVER flips about a third of Jev's correct decisions, 2.8 times as many as the strongest prior attack under the same limit on facts, and more on five other models. The gate blocks all of these attacks with at most one extra query and still accepts trusted updates. 

---
# Many Plans at Once: Diverse Top-Quality Planning in Continuous Time 

**Authors**: Mahyar Jahani-nasab, Hamid Rezatofighi, Mor Vered, Buser Say  

**Link**: [PDF](https://arxiv.org/pdf/2611.27012)  

**Abstract**: Planners for physical systems usually return a single trajectory, which leaves no alternative when an unmodeled constraint invalidates the chosen route or downstream preferences change. A set of plans computed in advance provides ready-made alternatives and warm-starts for fast adaptation, but only if the plans are both low-cost and mutually distinct. Discrete planning addresses this with top-quality and diverse planning, but these ideas do not transfer to continuous time, where every valid trajectory has infinitely many valid near-copies, and existing trajectory planners converge to one or a few solutions. We present Many Plans at Once (MPAO), which treats every plan it finds as a reason to search elsewhere: it samples untrained neural networks whose outputs meet the boundary conditions by construction, trains corrections shared by networks from the same distribution, stores every validated plan, and adapts its sampling to what it has found, which makes it anytime. We define the goal by bringing the cost bound of top-quality planning and the distinctness threshold of diverse planning to continuous time. Under equal budgets against five baselines, MPAO is the only method that solves all 40 instances across four domains. On contact-rich manipulation, it is the only method with plans within 25% of the best known cost on one task, and returns about ten times as many such plans as the strongest baseline on the other two. 

---
# Scalable Evaluation of Medical AI using Evidence-based Practice Recommendations 

**Authors**: Teya Bergamaschi, Bijal Rajput, Akshay Goel, Matthew Thompson, Jacqueline Shreibati, Dave Steiner, Ashley Carrick, Mahvish Nagda, Dale R. Webster, Avinatan Hassidim, Katherine Chou, Yossi Matias, Po-Hsuan Cameron Chen, Christina Chen, Mike Schaekermann, Yun Liu, Anil Palepu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27007)  

**Abstract**: Rigorously evaluating medical artificial intelligence (AI) requires testing open-ended clinical management across diverse presentations and stages of care. However, existing benchmarks rely on manual expert authoring, limiting coverage across conditions, conditional care pathways, and evolving guidelines. We develop an automated pipeline that extracts and verifies structured clinical criteria and branching decision trees from practice recommendations to generate scenarios spanning the care continuum, including nuanced presentations and practice variations across regions and guideline editions. Applied to 436 BMJ Best Practice clinical monographs (354 US, 82 International), this yields 3,200 scenarios spanning differential diagnosis, investigations, treatment, and complications, plus targeted suites for regional specificity (731 US/International pairs) and temporal freshness (630 scenarios). To grade responses, we develop a clinician-calibrated autorater that scores management plans on a 4-point rubric stratifying deviations by clinical severity. Across Gemini models, scaling improves adherence, while web search (+0.8 percentage points [pp] on average) and multi-sample synthesis (+2.2 pp) combine additively (+2.8 pp). In context, full monographs consistently trail condensed excerpts. On targeted suites, web search narrows international performance gaps (+4.2 pp) and improves adherence to newly updated guidance (+4.6 pp). In a study with Primary Care Physicians (n = 9) across 200 scenarios, AI assistance raises adherence from 50.5% to 70.0%, driven largely by fewer clinical omissions. Overall, our results demonstrate a scalable evaluation setup using an evidence-based corpus originally written to guide human clinical decision-making, and demonstrate its viability to assess medical AI and its assistance on clinician decision-making. 

---
# Reward Integrity in Adversarial Reinforcement Learning for LLM Guardrail Red-Teaming 

**Authors**: Chengming Liu, Leon Viti, Hongxu Chen, Erdem Kemik, Mary-Anne Williams, Ruth Oliver  

**Link**: [PDF](https://arxiv.org/pdf/2611.26998)  

**Abstract**: We study reward fidelity in an adversarial reinforcement-learning loop for red-teaming financial-services language-model guardrails. Apparent wins can arise when an attack omits prescribed personally identifiable information (PII), or when a reward judge credits a response that did not break the targeted policy. On attacker-generated and public benchmark examples, TypeSafe Jev checks whether the prescribed PII is present. For outcome scoring, we compare the decision models TypeSafe Jev, Laya (pretrained and post-trained), and pretrained CLM-8B with logged Opus 4.8 judgements from a replay corpus. Post-training markedly improves Laya's agreement with the judge. A separate author-reviewed off-target set establishes that Opus can award false wins. The findings motivate specialised decision-model judges for checking prescribed PII in attack inputs and scoring target responses against the intended policy. 

---
# From Letter-of-Credit Requirements to Executable Specifications: Evaluating Verification Accuracy and Completeness 

**Authors**: Ji Sub Choi, Whan Choi, Seo Yeon Gu, Suhyun Kim, Inyong Yun  

**Link**: [PDF](https://arxiv.org/pdf/2611.26988)  

**Abstract**: In letter-of-credit (LC) document examination, a correct final decision does not show that every requirement was examined. We therefore evaluate examination requirement by requirement and make each requirement executable. A language model converts the bill of lading (BL) and commercial invoice (IV) requirements of a credit into atomic conditions and logical structures, and a deterministic verifier retrieves the named evidence and computes each outcome, so every check can be compared with a gold encoding and traced to logged evidence. Because LC records are confidential, our on-premises 9B parser is fine-tuned only on synthetic pairs built from templates and human-reviewed public document values. On 345 real LC cases, final-decision accuracy proves misleading: a constant PASS rule has the highest raw accuracy while examining nothing. Although it never sees a real record during training, the fine-tuned parser detects 93% of the atomic requirements its ontology covers and reconstructs 86% of the covered requirements exactly. It verifies 82% of the evaluable requirements correctly, against 54% for a tool-using agent, and leads the agent on both satisfied requirements and discrepancies. It also completely examines 1.8 times as many cases as few-shot prompting of the same model, while direct prediction and the agent complete none. Executing the gold encodings with the same verifier shows that some of the remaining errors, missed discrepancies included, come from the operators and normalization, which a better parser alone cannot fix. Every rejection comes with a trace of the units and evidence behind it, which suits examiner-in-the-loop review. 

---
# MixFold: Sub-4-Bit Mixed-Precision LLM Inference via Output-Channel Grouping on Mobile NPUs 

**Authors**: Vinnam Kim, Seong Hoon Seo, Junhan Kim, Jaeseong You, An Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.26976)  

**Abstract**: Sub-4-bit quantization allows mobile devices to run larger LLMs, but its memory savings do not always translate into faster decoding on mobile NPUs. Their matrix units support only a few native integer datatypes, and converting unsupported weight formats on the vector unit can erase the gains from fewer bits. Mixed precision faces an additional constraint: each matrix-unit tile uses one datatype. MixFold executes 3-bit and perfectly symmetric 2-bit weights using native integer operations without expanding the weights to wider datatypes during decoding. MixFold then assigns datatypes to output-channel groups aligned with the matrix-unit tiles, enabling finer-than-layer assignment without disrupting matrix-unit execution. We compare MixFold with this http URL Hexagon backend on the Qualcomm Hexagon NPU of a Snapdragon 8 Elite Gen 5. Across the Qwen3 model family, MixFold achieves comparable or higher accuracy than this http URL's sub-4-bit formats at matched bits per weight, while decoding 1.16-1.41 times faster and generating 1.5-4.3 times more tokens per joule. 

---
# Run One, Get Two: Near-Free Streaming Safety Checks for Local LLMs 

**Authors**: Soheil Zibakhsh, Pedram Aghazadeh, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2611.26975)  

**Abstract**: Local LLM assistants decode for one user at a time, so decoding is bound by memory bandwidth and leaves compute idle. Streaming safety checks today trade accuracy against cost. A dedicated guard model is accurate but nearly doubles the latency per token, while a hidden-state probe is cheap but less accurate. We present ToRA, an adapter that runs a second model as an extra row in the generator's decode step, sharing its weights and KV cache. Dual-row kernels read each weight once for both rows. On Qwen3-0.6B, ToRA matches or beats a dedicated streaming guard, with higher F1 on 12 of 17 benchmarks. It detects 11.8 points more unsafe responses than the probe at a similar cost, under 4% added latency per token. 

---
# Right Numbers, Wrong Story: When LLMs Narrate Financial Changes Beyond Their Inputs 

**Authors**: Seunghan Lee, Jun Seo, Junhyeok Kang, Jaehoon Lee, Sangjun Han, Sungdong Yoo, Minjae Kim, Tae Yoon Lim, Dongwan Kang, Hwanil Choi, Soonyoung Lee, Wonbin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2611.26968)  

**Abstract**: A financial report drafted by an LLM can state every number correctly but still go beyond its input, adding judgments and causes that no one supplied. We call these additions unsupported framing and find them far more often in LLM drafts than in reports written by proInput facts Revenue +3% Higher prices 10-K filings Revenue increased 3% on higher prices. 10-K writer Revenue surged 3% on higher prices, due to strong demand. LLM Fits the size? Surged: Unusual for 3% In the input? In the input? Strong demand: Not in input Higher prices: Kept fessionals, and attempts to fix them by prompting with how each word is used instead make such judgments more frequent. We therefore propose FrameCheck, a plug-in that sits beFrameCheck Revenue increased 3% on higher prices. No extra model call Judgment word Unsupported cause Supported cause tween the LLM and human review to keep each draft within its input. Specifically, FrameCheck 1) guides the draft with professional examples that report a change of similar magnitude, 2) flags judgments that professionals would not use and causes the input does not give, and 3) removes them by rule, without a model call. In experiments on 17 open LLMs and 3M drafts, FrameCheck removes almost all unsupported framing from drafts and brings the judgment rate close to that of the writers of 10-K filings, the annual reports that US firms file with regulators. Code is available at this http URL. 

---
# Keep the Process, Change the Supplier: Runtime Governance for Tool-Using AI Agents 

**Authors**: Deepak Waghmare, Shaad Sufi  

**Link**: [PDF](https://arxiv.org/pdf/2611.26966)  

**Abstract**: Tool-using AI agents propose actions whose authorization and external effects cannot be inferred from generated text alone. We study an execution boundary that separates agent proposals from business-controlled authority, context disclosure, and supplier selection. The Open Process Control Plane (OPCP) mediates capability calls and records uncertain outcomes when a supplier may have acted without returning an acceptable reply. We evaluate this runtime with scripted requests and synthetic suppliers: 2,400 fault episodes across four policies, 60 process-termination/restart episodes, 19,000 timed calls across 19 configurations, five targeted mutations over 43 fixtures, and 160 substitution records. No language model or autonomous agent is evaluated. OPCP and a conservative composition agree on all measured fault outcomes. Both recover after definite refusal without cross-supplier duplication; retry-all duplicates effects in every tested lost-reply-after-commit case. Persistent runtime state prevents cross-supplier replay in the restart experiments, while same-supplier duplicate prevention also requires supplier idempotency. At 128 suppliers, best-score routing has a median latency of 1.411 ms versus 0.261 ms for pinned routing. All five selected mutations are detected, and record enumeration exposes acceptance of an empty regression set. These results characterize an execution mechanism relevant to agent governance, including its costs and assumption boundaries. They do not establish improved agent task success, prompt-injection resistance, or real-vendor model portability. 

---
# CodeReviewBench: An Automated Benchmark for Evaluating the Software Factory 

**Authors**: Ashley Zhang, Narmeen Fatimah Oozeer, Aleksandr Zverianskii, Daniel Ng, Tasana Pejovic, Antía García, Jacob Clyne, Ravpreet Setia, Akshay Utture, Etan Jacob Ginsberg, Shriyash Kaustubh Upadhyay, Fazl Barez  

**Link**: [PDF](https://arxiv.org/pdf/2611.26942)  

**Abstract**: As AI agents ship more code with minimal human oversight, code review becomes the critical quality-control checkpoint -- but one without a reliable, objective verifier: review quality is nuanced and ultimately a judgment call by the PR author. Existing benchmarks are structurally misaligned with this: annotators are third-party reviewers, like the bots they evaluate, not the PR authors who actually decide what to fix. They also rely on small, static issue sets that require full re-evaluation as tools update, limiting how frequently progress can be tracked. We introduce CodeReviewBench, built on a fundamentally different signal: revealed developer preferences -- if a developer acts on a bot's suggestion, we treat it as useful; if they ignore it, we do not. This signal is mined daily from the public GitHub event stream, yielding 500K+ judged PRs across 18 tools, and refreshes continuously as tools evolve. We validate the developer preference against an independent offline benchmark: the two agree on precision rankings and independently detect the same product updates within days of vendor releases. Together, these form the benchmark. 

---
# Self-similar Organization of Information in Language 

**Authors**: Leon Lufkin, Cutter Dawes, Kamesh Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2611.26935)  

**Abstract**: How much information relevant to predicting the future of a passage (i.e. predictive information) is carried by the preceding text, and how is this information distributed along that history? These questions are fundamental to understanding both the statistical structure of language and what large language models (LLMs) can learn about it. Indeed, LLMs trained to predict the next token in text must exploit this predictive information, so its organization determines how the structure of language becomes available to the model to learn. We develop a theory for the organization of predictive information based on the scaling behavior of a family of information measures, and the theory predicts that if this organization has the same form at all measured length scales -- i.e., it has a self-similar structure -- then the information measures evaluated at different lengths collapse onto a common form. We observe this scaling collapse in long text documents, showing that predictive information is present at every length scale we measure and has a self-similar structure. Self-similarity is a considerably stronger property than showing the existence of long-range dependencies, and it constrains the process generating the text. More broadly, our results give quantitative support to the empirical observation that conditioning on more distant text continues to reveal new structure in language to LLMs. 

---
# Self-Evolving Beyond Human-Crafted Experience for EDA Scripting 

**Authors**: Yang Liu, Shengbo Tong, Xiyuan Chen, Shiwei Hou, Sen Yuan, Hao Wang, Yanfeng Li, Hongwen Li, Yiming Zhu, Zhuolun He, Wenjian Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26912)  

**Abstract**: The growing complexity of electronic systems demands efficient design flows, yet EDA scripting remains a bottleneck: proprietary APIs are sparsely documented, and errors often appear as silent failures or cryptic exceptions. Hand-crafted knowledge helps but is difficult to maintain as toolchains evolve. We present ZhuLong-SE (Self-Evolution), an execution-grounded LLM agent that learns reusable knowledge from prior task traces. Building on ZhuLong, it enhances the task-level harness with API retrieval, documentation responses, isolated sandbox execution, and static checking, then adds governed cross-task self-evolution. Beyond solving individual tasks, ZhuLong-SE distills past execution traces into a three-tier memory store governed by ground-truth anchoring, disconfirmation filtering, held-out rollback, and anti-leakage retrieval. On the 158-task EDA-Eval-PyAether benchmark, the full task-level harness, with self-evolution disabled, achieves 92.4% pass@1 and outperforms ZhuLong on all five tested LLM backbones. On a harder instance-disjoint test set, self-evolution from the bare configuration surpasses a human-crafted-experience baseline by 2.5 percentage points (80.0% vs. 77.5% pass@1). 

---
# AstronautBench: Measuring Agent Self-Improvement Across Episodes Through Knowledge, Evidence, and Progress 

**Authors**: Congyuan Yu, Runping Tian  

**Link**: [PDF](https://arxiv.org/pdf/2611.26910)  

**Abstract**: LLM agents are expected to improve themselves from their own experience across episodes, with fixed weights. Self-improvement in an embodied world cannot be read from one measure. Existing benchmarks measure task progress, world-knowledge question answering (QA), and episodic memory, but a correct answer does not show that the agent ever observed the evidence for it, and progress does not show what the agent learned. We introduce AstronautBench, which measures one agent's self-improvement through three views on the same run: what it knows, whether its own evidence supports that knowledge, and how far it gets in the world. It is built on the game Outer Wilds, where each episode is a 22-minute loop after which the world resets, only the agent's knowledge carries over, and progress is gated by knowledge. After every episode, we measure the places reached and ship-log rumor facts revealed, the correctness of answers to 164 questions about the world, and whether each answer is supported by the frames and position the agent cites from its own recorded play. Two LLM judges from different model families check every answer. In eight-episode baselines with GPT-6.1 Sol and Gemini 4 Argon planners, which act through a learned controller or their own gamepad inputs, the three views do not move together, so any one of them alone gives an incomplete picture of whether the agent is improving. The GPT-6.1 Sol agent's correct answers grow from 14.6% to 26.8% of all questions, but at episode 8 only 59 to 68% of them are supported by its cited evidence. It reveals no new rumor facts after episode 3, while the places it reaches and its memory keep growing. More than half of the answers that become correct after episode 3 cite only evidence from earlier episodes. 

---
# Budget-Aware Predict-and-Verify Theorem Proving with Heterogeneous LLM Solvers 

**Authors**: Wei Jiang, Ziran Liu, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26906)  

**Abstract**: Large language model (LLM) theorem solvers differ in both proving capability and inference cost. Applying an expensive solver to every subgoal wastes computation when cheaper models are sufficient, while always starting from the cheapest solver wastes attempts on subgoals that require stronger models. We present BudgetPV, an agentic budget-aware predict-and-verify system that allocates computation across unresolved subgoals and heterogeneous LLM solvers. BudgetPV uses a cross-solver predictor to guide solver selection for each subgoal and revises the allocation as verification feedback becomes available.
The allocation problem changes as proving progresses. Early decisions mainly determine which solver should handle an untried subgoal, while later decisions determine which previously attempted subgoals should receive additional computation. We train the predictor with two objectives for these two stages. Class-weighted cross-entropy improves assignment to less frequent stronger solvers during the initial stage, while plain cross-entropy provides better calibrated probabilities for later allocation among previously attempted subgoals.
We evaluate BudgetPV on Lean Workbook, miniF2F, ProofNet, and PutnamBench using three progressively expanded solver settings: a strong but expensive solver alone, adding a medium-cost solver with intermediate solving capability, and further adding a cheap solver with lower solving capability. BudgetPV requires less LLM generation computation than uniform allocation to solve the same number of subgoals across the evaluated settings. The experiments further show the computation savings obtained as medium- and lower-cost solvers are added. 

---
# Towards Robust Partial-Edit Deepfake Localization: An edit and Generator Conditioned Analysis 

**Authors**: Aishwarya Fursule, Vamshi Nallaguntla, Yuan Tseng, David Harwath, Anderson Avila, Shruti Kshirsagar  

**Link**: [PDF](https://arxiv.org/pdf/2611.26903)  

**Abstract**: Partial-edit deepfake localization aims to identify manipulated regions within otherwise authentic speech. Recent systems report strong overall results, but aggregate scores do not reveal whether this performance holds across different editing conditions. In this work, we present an edit- and generator-conditioned analysis of Segment-Aware Localization (SAL) using WavLM-Large and XLS-R representations on the VoiceCraft and SSR-Speech subsets of PartialEdit. We categorize edits along three transcript-derived axes, namely content, meaning change, and editing operation, and evaluate each generator separately. While both representations achieve strong overall performance, localization is noticeably weaker on SSR-Speech. From our experiments, we found that the deletion is the hardest operation, and deletions produced by SSR-Speech are the most challenging condition for both representations. In contrast, edit content, meaning change, and sentence novelty have smaller effects. These findings show that a single aggregate score can mask important weaknesses, and that partial-edit localization should be evaluated separately across generators and edit types. 

---
# Beyond Demographic Discrimination: Benchmarking Normative-Descriptive Calibration in LLMs 

**Authors**: Radhika Bhargava, Shinibali Bhattacharyya, George Mihaila, Jay Prajapati, Anwesan Pal, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26870)  

**Abstract**: Large language models (LLMs) are increasingly deployed in socially consequential settings, where they must distinguish between normative reasoning (what should happen) and descriptive reasoning (what is likely to happen). Most fairness benchmarks evaluate demographic discrimination under a single reasoning objective, and none we know of tests whether a model treats the same item differently when asked to describe an outcome versus decide one. We introduce normative--descriptive calibration, the extent to which a model's differentiation between matched candidates depends on whether it is asked to predict or to decide, and measure it with the Normative--Descriptive Calibration Benchmark (NDCB): 697 matched-pair scenarios across six domains, each posed under both framings, and three metrics, Normative Engagement Rate (NER), Normative Leakage Rate (NLR), and Normative--Descriptive Separation (NDS). Across six LLMs, NDS separates 13 of 15 model pairs, against 7 for BBQ accuracy and none for BOLD, although these counts also reflect each benchmark's sample size. Claude Sonnet 4.6 and Claude Opus 4.6 separate the framings most strongly (average NDS $= +0.55$ and $+0.52$, with overlapping confidence intervals), whereas GPT-5.5's average NDS is negative ($-0.14$) under the single-pick instruction and near zero ($+0.02$) without it. In a simplified prompt template, removing single trigger words never changes the sign of a model's average NDS but can shift it by up to 0.46 and reorder models. Reversing candidate order reveals a further effect: when forced to choose between identical candidates, all six models favor whichever is listed first, and once order is averaged out their demographic-aligned lean is toward the marginalized candidate. Describing disparities and prescribing them are separable behaviors that differ across models. 

---
# Memorization and Generalization in Generative Models: A Perspective From Energy-based Modeling and Associative Memory 

**Authors**: Bao Pham  

**Link**: [PDF](https://arxiv.org/pdf/2611.26797)  

**Abstract**: Generative models now write working code, produce realistic images, and hold long conversations, yet the mechanisms behind their intelligence still evade our understanding. This thesis explores that puzzle through Associative Memory (AM), in which patterns are stored as energy minima and retrieved through attractor dynamics. Beyond their capacity, AMs form new minima known as spurious states, long regarded as a failure of memory. We argue that this failure is where creativity begins, and that memorization and generalization can coexist as outcomes of one retrieval process. First, we show that diffusion models behave as AMs. With little data, a diffusion model forms a distinct basin of attraction around each training sample. As the data grows, these basins shrink, and spurious states emerge as attractors that blend stored samples yet match none of them. They are the first signs of generative capability. We characterize this transition by measuring the volume, energy, and curvature of these basins across datasets and architectures, including Stable Diffusion. We then turn to language and move beyond the energy-based view of AM. We show that uniform-state discrete diffusion models are AMs capable of retrieving unseen data, with basins of attraction that arise from conditional likelihood maximization alone. As the training set grows, basins around training examples contract while basins around unseen examples expand, until corrupted test sentences are recovered as reliably as training ones. Conditional entropy exposes this coexistence within a single sequence and cheaply diagnoses where a model lies along the transition. Together, these results suggest that AM is a general principle defined by basins of attraction, and that memorization and generalization can coexist within one model. AM thus offers a framework for asking when a generative model recalls and when it creates. 

---
# Beyond Final Answers: Interpretable Tumor Immune Microenvironment Reasoning from H&E and mIF 

**Authors**: Yixuan Duan, Kexin Geng, Hanwen Xu, Wei Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26794)  

**Abstract**: Readouts of the tumor immune microenvironment (TIME), such as CD8+ T-cell density, intra-epithelial infiltration and the Ki67 index, combine tissue architecture from H&E with cell phenotypes from multiplex immunofluorescence (mIF). Multimodal large language models (MLLMs) could automate these readouts with an explanation, but two gaps remain. Pathology MLLMs read H&E alone, while these readouts need the cell phenotypes that mIF measures. And their reasoning has no step-level truth to be checked against, so R1-style reinforcement learning rewards only the final answer and leaves the reasoning unverified. Same-section H&E and mIF provide what both gaps lack: mIF measures every cell without manual annotation, which gives cell-level ground truth for each step that clinical guidelines prescribe, from delineating a tissue compartment to counting and normalizing the marker-positive cells within it. We therefore present HERMES (H&E-mIF Reasoning with Multimodal Evidence Supervision), an R1-style MLLM that reasons jointly over paired H&E and mIF in guideline steps and is trained with a process reward that checks every stated quantity against this truth. Across four field-level TIME tasks in colorectal cancer, HERMES (4B) reaches an average accuracy of 84.5%, compared with 74.7% for the strongest of twelve proprietary, general, medical and pathology MLLMs, and a higher share of its intermediate quantities is correct than for the strongest open-weight baseline. Without the process reward, reinforcement learning barely improves on supervised fine-tuning (80.0% vs. 78.8%) and fewer stated quantities are correct; editing a decisive stated quantity changes the answer as implied in all 885 test questions. Without retraining, it keeps the highest average accuracy on an external cohort imaged with CODEX (76.4% vs. 67.3%) and on head and neck cancer, a cancer type absent from training (76.7% vs. 72.5%). 

---
# ChatAtom: A Multi-Agent LLM System for Molecular Dynamics Simulation and Visualization 

**Authors**: Trupti Mohanty, Ming Du, Mitchell A. Wood, Brad L. Boyce, Remi Dingreville, Mathew J. Cherukara  

**Link**: [PDF](https://arxiv.org/pdf/2611.26782)  

**Abstract**: Molecular dynamics (MD) simulation resolves atomic-scale motion and predicts the structural, mechanical, and dynamic properties of materials, making it cen- tral to materials design and discovery. Yet an MD study requires an expert-driven workflow: designing the simulation, writing reliable input scripts, executing them, and analyzing the results with separate tools. Here we present ChatAtom, a multi-agent large language model system that translates a natural-language request into a complete MD workflow without the user writing input scripts or analysis code. Each agent has a defined responsibility within the workflow. An orchestrator plans the study; the LAMMPS and OVITO agents each use a generator-reviewer pair to prepare and review simulation and visualization scripts before execution; a high-performance computing agent dispatches jobs to remote resources; and an analysis agent interprets the completed run. The work- flow is accessible through a browser interface that streams thermodynamic data and rendered frames during local execution, allowing users to follow the simu- lation as it progresses. The architecture also supports new capabilities through loadable skills without modifying the agents, enabling autonomous hypothesis generation and testing. We evaluate ChatAtom through script-generation bench- marks, an end-to-end study of crack propagation, and a closed-loop campaign. By bringing script generation, execution, visualization, and analysis into one reviewed and observable pipeline, ChatAtom lowers the barrier to conducting MD studies and broadens access to advanced simulation workflows. 

---
# TOUMAII: A 4.5M-Pair Tool-Calling Preference Dataset and Its Source-Agnostic Generator 

**Authors**: Moritz Pötzsch, Andrea Gentilini, Lisa Rossgoderer, Tom Osterfeld, Kesava Prasad Sanjivi Arul, Swanand Ravindra Kadhe  

**Link**: [PDF](https://arxiv.org/pdf/2611.26722)  

**Abstract**: To autonomously execute multi-step tasks, enterprise agents must reliably interact with tools, call APIs, or query databases, making tool-calling a fundamental capability of AI models. Supervised fine-tuning (SFT) optimises for producing the correct call, but never contrasts it with a plausible incorrect one. In an environment where even a minor formatting error can invalidate a call and an incorrect decision can derail the interaction, direct preference optimisation (DPO) targets exactly this contrast. DPO, therefore, requires preference pairs rather than plain demonstrations. For tool-calling, such data, especially in realistic multi-turn settings, remains extremely scarce. To address this scarcity, we introduce TOUMAII, a large-scale tool-calling preference dataset containing ~4.5M validated preference pairs, derived from Agent-Ark/Toucan-1.5M (apache-2.0) conversations. The dataset consists of multi- and single-turn interactions and includes parallel and sequential function calls. Additionally, we release TOUMAII-Gen, a synthetic data generation pipeline that consumes tool-calling SFT datasets and generates a preference dataset with naturally occurring model-generated tool-calling failure modes. To validate the effectiveness of our DPO data, we fine-tune a 3-billion-parameter model, achieving performance parity with other high-quality tool-calling preference datasets. Furthermore, we release a second small preference dataset derived from ToolACE (apache-2.0), showcasing the source-agnostic nature of our pipeline.
We release the complete code and datasets: this https URL and this https URL 

---
# Coherence by Construction: Synthesizing CTI Knowledge Graphs and the Formal Limits of Coherence Validation 

**Authors**: Tanzimul Alam Fahim, Mohammad Arif Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2611.26717)  

**Abstract**: Synthesizing relational training data such as knowledge graphs with a large language model typically emits structured objects one at a time, which keeps each object valid but leaves the relationships between them ungrounded. We study an alternative paradigm, generation by simulation: rather than describing a relational artifact, we simulate a process that produces it and take the artifact to be its causal record. Language-model agents propose intent and a deterministic referee resolves outcomes against an explicit world model, so a relation is emitted only when the event it represents occurs, making coherence a property of construction. We instantiate the paradigm for cyber threat intelligence knowledge graphs in the STIX 2.1 standard with three engines, among them an evolutionary engine applying language-model genetic operators to structured graph genomes, sharing a deterministic compiler that makes malformed and dangling output unrepresentable. We prove referential closure, causal grounding, and identifier stability, an ablation localizes each to the referee and world model rather than the policy, and two independent LLM judges reproduce the rubric's engine ranking. Our central result is cautionary, and we prove it: a validator reading any graph feature set a structure-matched random graph can reproduce is uninformative about coherence for a coherence-by-construction generator. Using Erdős-Rényi controls we realize the theorem four times, as structural separation, embedding transfer, inductive link prediction, and real-data augmentation each reward a triviality rather than coherence; we give the mechanism behind each and the conditions a valid benchmark must meet. 

---
# Adding One More Agent: How AI Team Size and Formation Shape Productivity and Diversity in Human-AI Brainstorming Teams 

**Authors**: Rong-Ching Chang, Hao-Chuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26663)  

**Abstract**: Generative AI has improved individual productivity, but whether adding a plurality of AI partners helps or hurts remains open. We crossed AI partnership (one versus two) with behavioral formation (calibrated to average human behavior versus customized by the participant) in a brainstorming experiment ($N$=168) with human-human dyads as reference. Two-AI rooms posted more ideas to the shared board, but human-authored ideas dropped substantially. With one AI partner, formation mattered: humans paired with an average-calibrated AI posted more than any other condition. With two, this effect vanished. Humans continued talking but converted fewer ideas into board posts. AI social talk predicted more human posts, while AI posting volume predicted fewer. AI presence increased global diversity of human-authored ideas, but only a second AI increased within-session differentiation. The number of AI partners and their formation can reshape what the human contributes; single-agent results do not generalize to multi-agent teams. 

---
# Optimal Linear Regression Without a Variance 

**Authors**: Artin Tajdini, Jasper C.H. Lee, Jonathan Scarlett, Kevin Jamieson  

**Link**: [PDF](https://arxiv.org/pdf/2611.27983)  

**Abstract**: We study high-probability linear regression in the infinite variance regime, wherecthe regression noise has only a finite conditional $\alpha$-moment $\nu$ for some unknown $\alpha\in(1,2)$, and the covariates may only have finite moments up to a given order (e.g., 4th moment). We give a black-box reduction that converts finite-variance regression procedures into weak-moment procedures. The reduction is based on empirical residual pruning and is agnostic to the moment exponent, the noise scale, the covariance, and the design moment. Under a finite fourth-moment condition on the covariates, any finite-variance subroutine attaining the sub-Gaussian rate yields the following when the number of samples $n$ is above a suitably defined threshold: Without knowing $\alpha$ or $\nu$, we have with probability at least $1-\delta$ that $ \|\widehat\theta-\theta^*\|_\Sigma
\lesssim_\alpha
\nu^{1/\alpha}
\big(\frac{d+\log(1/\delta)}{n}\big)^{\frac{\alpha-1}{\alpha}}, $ where $d$ is the dimension, $\Sigma$ is the covariate covariance matrix, and $\|v\|_\Sigma:=(v^\top\Sigma v)^{1/2}$. We further prove a matching minimax lower bound that holds under the same conditions. The reduction is computationally efficient whenever the finite-variance subroutine is. We also obtain a distribution-free prediction guarantee under a bounded raw $\alpha$-moment assumption on the response, and give an analogous $\alpha$-agnostic reduction for mean estimation. 

---
# Non-penetrating World Models 

**Authors**: Yuhang Dai, Yinwei Wu, Ziyue Lin, Han Ji, Xingyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27963)  

**Abstract**: We study a new problem of preventing object penetration in interactive video world models. To understand how existing models handle this, we introduce NoPeBench, a benchmark that evaluates how world models respond to physical obstacles. Our evaluation reveals that none of the open-source models can reliably prevent such failures. Surprisingly, however, these models already encode the information needed to predict such violations. We find that the hidden states of diffusion transformers (DiTs) can predict whether a candidate action will penetrate an obstacle. Building on this, we introduce NoPeWorld, a lightweight framework that detects and prevents penetration. Specifically, we train a small prediction head on top of a frozen DiT's hidden states to read out if a candidate action is safe or not. Unsafe actions are then replaced with safe ones that best preserve the original intent. Despite its simplicity, NoPeWorld substantially reduces the physical violation rate, averaged over six video world models, from 74% to 16%. Remarkably, our method shows zero-shot generalization to open-domain imagery, third-person viewpoints, and moving obstacles. We publicly release our code, the NoPeBench benchmark, and model weights. Project Page: this https URL . 

---
# Model Carving: Training Transformers That Optimize Their Own Topology 

**Authors**: Kaden Zheng, Pratyusha Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2611.27962)  

**Abstract**: Machine learning has largely kept separate the choice of architecture and the process of training. A classic Transformer, for instance, trains a fixed set of attention and MLP sublayers at every layer, in the same order, and optimizes the weights within. This uniformity is a convention rather than a principled design choice, and it is likely suboptimal: the best architecture for a given task may be smaller, more efficient, and far from uniform. We believe that, ideally, training itself should shape the model's topology. To this end, we introduce Model Carving, which learns both a Transformer's weights and its architecture in a single run. By applying learnable, scalar gates to each sublayer, we allow the training loss to remove the model's own components during training -- freeing a smaller heterogeneous network inside. We evaluate Model Carving by pre-training standard Transformers from 140M to 2.7B parameters and looped Transformers from 140M to 1.3B. At equal training compute, carved models reach a lower validation loss than dense models trained with up to 1.7 times more compute. Furthermore, carved models also outperform size-matched models produced by pruning or distillation, suggesting that the best way to obtain small models for edge devices is to carve them during training, not to compress a large one afterwards. This advantage both grows with scale and carries over to downstream tasks such as few-shot evaluation and generation. Because the removed sublayers no longer run, carved models also cost up to 34% fewer FLOPs per token and decode up to 1.6 times faster without cost in quality. Our results show that architecture, like weights, should be learned and not fixed in advance, and that inside every large transformer is a smaller, faster, stronger one waiting to be carved out. 

---
# SUAF: Spectral Universal Advantage Features for Zero-Shot Policy Improvement 

**Authors**: Huaze Tang, Runqing Miao, Tianren Peng, Lam Nguyen, Chao Wang, Jairong Li, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2611.27919)  

**Abstract**: Reward-independent representations enable reinforcement learning agents to reuse knowledge across tasks, but limited representation capacity raises three fundamental questions: what statistical dependence should these representations capture, which components of that dependence should be retained for policy improvement, and how does discarding the remaining components affect decision-making? We address these questions through canonical dependence analysis. For a reference policy, we construct an action-centered, positive-lag canonical dependence matrix (CDM) that isolates the additional predictive contribution of the current action beyond the current state. Its singular value decomposition yields Spectral Universal Advantage Features (SUAF), paired features that recover delayed advantage through linear reward projections. We show that the spectrum quantifies mutual information between current action and future state-action visitation conditioned on current state and determines reward specific approximation errors. Retaining the leading modes gives optimal rank-constrained approximations under both isotropic average-case and reward-uniform minimax criteria. Separating the immediate reward contrast also avoids the spectral floor arising from direct compression of the full advantage operator. To learn SUAF from reward-free data, we develop an action-centered Soft-HGR objective with bidirectional temporal-difference estimation. We further connect spectral approximation errors to sufficient conditions for policy improvement. Finite-state experiments validate the spectral predictions and feature-learning method, while zero-shot evaluations on discrete tasks and continuous-control benchmarks demonstrate the utility of the learned representations across rewards. 

---
# StereoVLM: Binocular Vision-Language Models for Metric 3D Spatial Reasoning 

**Authors**: Chi Yan, Guoyang Zhao, Wentao Yuan, Yixiao Ge, Yanning Zhou, Dan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27915)  

**Abstract**: Vision-language models (VLMs) exhibit strong semantic understanding but remain limited in their ability to reason about metric 3D space in the physical world. Existing VLMs typically infer 3D scene structure from one or more images without known camera calibration or cross-view geometric constraints. This makes metric 3D inference inherently ambiguous and leaves it unclear whether their spatial predictions are grounded in visual evidence or driven primarily by semantic priors. In contrast, calibrated stereo provides explicit cross-view geometry that resolves this ambiguity and offers a more reliable foundation for metric 3D perception and geometric generalization. We investigate how VLMs can learn transferable 3D representations directly from calibrated binocular observations. To this end, we introduce StereoVLM, the first geometry-aware binocular VLM for metric 3D spatial reasoning. StereoVLM follows a hierarchical learning paradigm spanning from cross-view correspondence and disparity estimation to calibration-conditioned metric depth perception and semantic spatial reasoning. We further find that disparity supervision generalizes substantially better than direct binocular depth supervision on unseen environments. To evaluate these capabilities, we introduce Stereo3D-Bench, a comprehensive benchmark across three levels, from point-level geometry to object-level spatial reasoning. StereoVLM-8B achieves 95.17% depth accuracy ($\delta_1$) and a 66.98 average object-level score on Stereo3D-Bench. Intervention results further indicate that the model's performance on Stereo3D-Bench benefits from valid cross-view geometry and cannot be explained solely by additional visual tokens or semantic priors. Hierarchical geometric supervision also yields average gains of 3.54 points across five spatial benchmarks and 4.3 percentage points in RoboCasa manipulation success. 

---
# Star Panoptic Quality: Selectively Evaluating Split and Merged Instances 

**Authors**: Erik Großkopf, Paula Tamara Buzduga, Hendrik Möller, Florian Kofler  

**Link**: [PDF](https://arxiv.org/pdf/2611.27913)  

**Abstract**: Detecting individual instances is crucial for many segmentation problems. However, expert instance annotations are often unavailable, so instances are approximated heuristically (e.g., as connected components of semantic maps), which can merge touching objects or split fragmented ones. Even when expert annotations exist, expert disagreement or structural delineation biases can yield split or merged reference instances, corresponding to over- or under-segmentation, respectively. Panoptic Quality (PQ) and related metrics capture both segmentation and detection quality while weighting all instances equally, regardless of size. Yet when the references are biased, these metrics can neither detect split or merged instances nor distinguish valid from invalid overor under-segmentation. We present Star Panoptic Quality (StarPQ), a novel evaluation ramework, which closes this gap in two steps: First, we construct a so-called star forest in the bipartite graph whose clean, merge, and split stars isolate different failure modes. Second, we calculate the final metric by individually weighting each failure mode, allowing practitioners to account for instance delineation biases. Across various experiments, we demonstrate that StarPQ not only enables targeted penalization or eniency towards specific failure modes, but also serves as a diagnostic tool to reveal their prevalence. 

---
# Decision Columns as an Index: Interpretable Embeddings for Explainable Entity Matching and Search 

**Authors**: Aadharsh Kannan, Robert Bruckner, Philip A. Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2611.27910)  

**Abstract**: Business data systems must determine whether records refer to the same entity and retrieve records satisfying requests whose decisive properties are implicit. As agents act on these records, both tasks require auditable reasons. Dense embeddings perform well but offer opaque coordinates, cannot be filtered by meaning, and require corpus re-encoding to add new properties. We propose interpretable embeddings built from decision columns: each column stores a decision model's calibrated answer to a natural-language question, with outcomes yes, no, or not stated. A record's embedding is its answers across an ordered set of questions. Similarity decomposes exactly into per-question contributions, and new questions add coordinates without altering existing ones.
On public product benchmarks, question-space retrieval matches dense candidate generation statistically in a retrieve-then-verify entity matcher, while providing exact explanations for accepted matches. Questions derived from retrieval failures triple mean reciprocal rank without re-encoding prior answers. For search, inferring the attributes of a suitable result outperforms encoding only the literal request, and combining question-space and dense scores improves retrieval, though not at our prespecified confidence threshold. A single small neural network predicts the full interpretable vector from an existing dense embedding, enabling corpus backfilling from a few thousand labeled rows at roughly 1/200 the cost. We position interpretable embeddings as a complementary, readable retrieval channel for explaining matches, filtering on attributes, and extending indexes as failures expose missing facts. 

---
# TEA-Bench-BR: Can Coding Agents Turn a Branching Clinical Specification into Correct Code? 

**Authors**: Marcio Andre de Souza Estevam  

**Link**: [PDF](https://arxiv.org/pdf/2611.27895)  

**Abstract**: Coding agents increasingly implement rules written by domain experts, and bugs in such code are silent: it runs and returns a plausible answer. We introduce TEA-Bench-BR, a benchmark built on the Brazilian pathway for early autism (TEA) screening: three fictitious instruments with the structure of real ones, a multi-professional referral flow, and export to FHIR R4 with the national RNDS profiles, all specified in Portuguese. A generator samples each task's parameters (144 tasks, 6,798 hidden cases), and the hidden cases kill all 1,955 non-equivalent mutants of the reference. With Gemma 4 as the agent, the 12B solves 38.0% of single-instrument tasks and none at the FHIR level; the 31B solves 77.8%, 85.2% and 78.7% at levels 1 to 3 and passes over 99% of the hidden cases, its remaining failures mostly in input validation rather than clinical decisions. With the same tasks in English, the 31B solves 57.4% at level 3, as its code validates an identifier less often; the 12B's, more often. The 12B misses 10.10% of the cases that call for referral; the 31B, 0.02%. A contamination probe on a modified PHQ-9 finds no error that follows the original rules. 

---
# LesionAgent: Training-Free Agentic Tumor Tracking and Response Assessment in Longitudinal CT 

**Authors**: Maximilian Rokuss, Jeya Maria Jose Valanarasu, Alejandro Lozano, Benjamin Hamm, Yannick Kirchhoff, Fabian Isensee, Klaus Maier-Hein  

**Link**: [PDF](https://arxiv.org/pdf/2611.27876)  

**Abstract**: Every few months, a cancer patient's CT scans decide whether treatment continues. 3D foundation models can now detect, segment, register and track lesions, yet none of them reads the scans as a radiologist does. A radiologist decides which findings are lesions, which are the same lesion months later and whether a new spot means progression, and checks each decision against the extensive rules of a clinical guideline. We introduce LesionAgent, an LLM agent that performs this read end to end without task-specific training. It operates frozen 3D models as tools, acts in 3D through coordinate-labelled views, remembers every lesion across visits in a registry, verifies every model output it relies on, and leaves every number to a deterministic guideline engine. On four public longitudinal CT datasets, LesionAgent tracks more lesions and measures them more accurately than every automatic baseline, and where the same 3D models chained by fixed rules call progression falsely at 33 of 56 follow-ups, it does so twice. On clinical follow-up CTs, it agrees with the radiologists' response at 20 of 23 follow-ups, including every change in response, and without training on DeepLesion, it retrieves more of its lesions than every tracker trained on it (89.0% vs. 87.4%). Its harness, not its LLM, makes the difference: given the same LLM and 3D models but not the harness, Claude Code reaches a detection F1 of 41% and LesionAgent 78%, and an open-weight LLM that can run inside the hospital rises from 6% to 73%. LesionAgent and a napari reading app for any LLM are available at this https URL. 

---
# Beyond Visual Fidelity: Comparing Categorical and Continuous World Models 

**Authors**: Sarah Parisot, Melanie Fernandez Pradier, David Bignell, Sergio Valcarcel Macua, Katja Hofmann  

**Link**: [PDF](https://arxiv.org/pdf/2611.27870)  

**Abstract**: Visual world models can now generate compelling interactive worlds, but visual realism does not guarantee faithful dynamics. We systematically compare how categorical and continuous prediction shape learned dynamics under both next-token and iterative whole-frame generation. These choices are often bundled in approaches such as autoregressive token models and whole-frame flow matching. We design controlled 2D and 3D environments and evaluate dynamics validity, calibration, and visual quality. In 2D with lossless visual encoding, categorical models achieve higher rollout validity with less training compute, particularly for stochastic entities. Among categorical models, greedy decoding produces the highest rollout validity, while sampling from a next-token model yields the best-calibrated stochastic predictions. Failures in continuous models, such as disappearing game characters, echo those seen in more complex environments. Here, known dynamics let us measure their impact. In 3D, categorical models also achieve stronger long-horizon performance with less training compute. Even models with lossy tokenization remain competitive on semantic accuracy and FVD, despite lower next-frame RGB accuracy. Together, these findings make a strong case for categorical prediction as a compute-efficient route to learning world dynamics. 

---
# Who Grades the Agent? Privilege Separation for Coding-Agent Benchmarks 

**Authors**: Nathan Nour Houlamy, Mattia Favretto, Earlence Fernandes, Mauro Conti  

**Link**: [PDF](https://arxiv.org/pdf/2611.27860)  

**Abstract**: Benchmarks for AI-based coding agents are a staple of the current AI landscape, because they are the leading way to measure the performance of LLMs. Their scores guide business and research decisions. Evaluating coding agents is challenging, since they are able to manipulate their evaluation infrastructure to artificially achieve better scores. Grading an agent's solution often requires running its code, which can then manipulate the evaluation process. BenchJack, an automated benchmark auditor, recently reached near-perfect scores on nine of ten popular agent benchmarks without actually solving their tasks. Agents already do this unprompted, as OpenAI's July 2026 Hugging Face incident showed.
Building on secure systems design, we introduce SplitBench, a privilege separation framework for evaluating coding agents and the first in the scientific literature to separate running a task's tests from judging their results. Even an agent whose code takes over the entire test environment cannot forge a pass, because a trusted grader holds the expected answers. We show that real-world benchmarks (e.g., Terminal-Bench, DeepSWE) can be ported to the SplitBench framework with minimal effort and high fidelity: 180 of 202 tasks convert, and SplitBench agrees with the native graders on 98.7% of agent attempts. We attack the converted tasks in the worst case, where adversarial agents can read all the grading code but not the expected answers. SplitBench blocked 75% of their attempts; the rest mostly reveal weaknesses in the original tasks' tests rather than flaws in SplitBench. 

---
# Graph Signal Generation via Graph-Aware Diffusion Models 

**Authors**: Vimal K B, Sergio Rozada, Antonio G. Marques, Elvin Isufi, Hadi J. Rad  

**Link**: [PDF](https://arxiv.org/pdf/2611.27826)  

**Abstract**: Many signals are defined on the nodes of a graph, such as traffic flow at road sensors or user purchases of related products. They are often partly observed, and learning their distribution enables imputing missing values and forecasting future ones. Diffusion models learn such distributions, but their forward noise is independent across nodes and does not depend on the graph. We introduce graph-aware diffusion (GAD), whose forward process is a heat diffusion on the graph driven by Gaussian noise. In the graph Fourier basis, the forward transition is a scalar Gaussian channel at each frequency. With a shared diffusion rate, high graph frequencies are attenuated faster than low ones. We propose Mode-Adaptive SNR Scheduling (MASS), which assigns each graph frequency its own diffusion rate and keeps all of them at the same SNR. For graph-stationary signals, MASS equalizes the normalized error of the optimal linear denoiser across frequencies. We bound reverse-SDE stiffness as a function of the (shifted) Laplacian's condition number, with a logarithmic upper bound for MASS and a linear lower bound for shared rates. GAD with MASS improves recommendation accuracy by 5-19%, reduces traffic-imputation RMSE by up to 7.8% with half as many reverse steps, and improves forecasting MAE and RMSE using only 3 reverse steps. 

---
# Redesigning Neural Networks to Not be Confidently Wrong even to Strong Adversarial Attacks 

**Authors**: Michael W. Spratling, Heiko H. Schütt  

**Link**: [PDF](https://arxiv.org/pdf/2611.27811)  

**Abstract**: Early deep learning models attempted to simulate visual perception using a neural hierarchy that alternated between two types of layer: one modelled on cortical simple cells (that was designed to increase selectivity), and one modelled on cortical complex cells (that aimed to increase tolerance). We show that such a neural network (NN) architecture, when implemented with novel components that are better able to learn selective and tolerant representations, is capable of accurate and robust image classification and improved alignment between the behaviour of the model and the perceptual abilities of humans. Firstly, the proposed models are shallow, fewer than ten layer-pairs, which is more consistent with the rapid feedforward object categorisation performed by the biological visual system. Secondly, they are far less prone to producing classification predictions with high confidence for samples from unknown classes or for image consisting of pure noise, and hence, like humans are not easily fooled into perceiving objects that are not present in the image. Thirdly, for the proposed NNs adversarial samples cause predictions with low confidence, enabling them to be accurately identified. In contrast to humans, our networks still do not correctly classify adversarial samples. However, the behaviour of our networks is a significant step towards human-like behaviour compared to standard DNNs that produce highly confident, but incorrect, classification predictions for adversarial samples. The increased robustness to adversarial and unknown samples is achieved while maintaining competitive performance in terms of generalisation to novel samples from the training categories. This enables models of this type to be competitive with state-of-the-art DNNs across multiple different evaluation metrics and benchmark tasks. 

---
# $\mathtt{DistP}$: Scaling Local-Update Training with Zero-Shot Hyperparameter Transfer 

**Authors**: Andrej Jovanović, Alex Iacob, Philipp Wiesner, Christophe Roux, Max Zimmer, Lorenzo Sani, Sebastian Pokutta, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2611.27808)  

**Abstract**: Local-update methods offer a communication-efficient alternative to Distributed Data Parallelism ($\mathtt{DDP}$), but introduce additional hyperparameters that require costly tuning. For $\mathtt{DDP}$, zero-shot hyperparameter transfer reduces this cost by reusing settings tuned on smaller models, but its applicability to local-update training remains unclear. We propose $\mathtt{DistP}$, a theoretical framework extending existing transfer rules from $\mathtt{DDP}$ to local-update training with varying worker counts, local steps, and outer optimizers. With averaging as the outer optimizer, $\mathtt{DistP}$ supports learning-rate transfer across width and depth while improving model quality. The tuned inner learning rate also directly transfers across synchronization intervals, but follows two regimes when increasing the global batch size: square-root learning-rate scaling for larger local batches and a fixed reference rate for additional workers. We then generalize width and depth transfer to other outer optimizers, enabling stable scaling with $\mathtt{Adam}$. However, with momentum in the outer optimizer, changing batch size or local steps can require inner--outer learning-rate tuning, consistent with prior work. In language-modeling experiments at $823$M parameters, $\mathtt{DistP}$ reduces validation loss by $8.1\%$ and improves mean downstream accuracy by $10\%$ relative to baselines without retuning at the larger scale. Fitted scaling laws suggest that this advantage grows with the compute budget, reinforcing the benefits of $\mathtt{DistP}$ for communication-efficient training. 

---
# Neural Defend ARCAS 290M: A Robust Foundation Model for AI-Generated Audio Detection 

**Authors**: Sivashankar Selvarajan, Piyush Verma, Sumit Kumar, Sharayu N. Deshmukh  

**Link**: [PDF](https://arxiv.org/pdf/2611.27784)  

**Abstract**: Voice cloning now needs only seconds of audio, and detectors tuned to one dataset often fail when the generator, channel, or language changes. We introduce ARCAS 290M, Neural Defend's foundation model for detecting AI-generated audio. A single frozen production model, with no benchmark-specific fine-tuning, generalizes across text-to-speech, voice-conversion, and codec-language-model generators, telephone and Bluetooth channels, recording conditions, and languages. We evaluate it on the complete official test sets of ten public benchmarks, scoring every recording exactly as released, and against commercial systems on independent leaderboards. ARCAS 290M ranks first among all systems on the Podonos leaderboard and has the lowest equal error rate among commercial detectors on ASVspoof5. It separates real from AI-generated speech almost perfectly on ASVspoof5, SpeechFake, and ShiftySpeech, and detects nearly all AI-generated speech in Hindi, Tamil, Bengali, and Telugu. On every benchmark with a published zero-shot comparison, including in-the-wild audio, partial fakes, and the conditions hardest for the field, it surpasses the strongest reported detector by a wide margin. These results establish ARCAS 290M as a general-purpose foundation for AI-generated audio detection in fraud prevention, platform safety, and media verification. 

---
# Search-RSI: Expanding Search Worlds Along Multimodal Agents' Capability Frontier 

**Authors**: Chengchao Lv, Yin Wang, Chunzheng Zhu, Wanli Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27780)  

**Abstract**: Multimodal search agents answer knowledge intensive visual questions by grounding them in retrieved evidence, and recent recipes train them on multi hop tasks that are synthesized from knowledge graphs and grown online to keep pace with the agent. These recipes, however, treat the searchable world as fixed: a new question can be well formed and verifiable while its evidence is absent from the retrieval backend, so the agent issues the right query, retrieves nothing, and is penalized as if its policy were at fault. Such unreachable negatives accumulate with every round of growth. We propose Search-RSI, a recursive self improvement framework that expands the search world along the agent's capability frontier. In each round, the current policy diagnoses its partially mastered regions of a task space spanned by hop depth, relation rarity, and distractor density; new tasks there are grown together with their evidence; and the same policy decides how they are used: candidates on which its probes disagree feed reinforcement learning, while those beyond its reach become hint guided demonstrations of its own. Keeping the policy inside this loop matters: an improver frozen at the start of training loses 3.2 points. On nine multimodal search benchmarks, Search-RSI keeps improving where static task growth saturates, raising the average accuracy of the recipe it builds on by 9.8 and 9.5 points at 9B and 27B, of which 7.0 points are out of reach for a world that is merely made consistent. The loop uses no external model and only privileged task structure that is removed before training, at 1.27x the compute of static growth. 

---
# RoSci: Exploring Robustness in Science VLMs 

**Authors**: Junying Wang, Yijin Guo, Yalun Wu, Ziheng Jiao, Farong Wen, Qihang Yan, Wenzhe Li, Ye Shen, Qi Jia, Chunyi Li, Zicheng Zhang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2611.27750)  

**Abstract**: Recent advancements in Vision-Language Models (VLMs) enable significant progress in scientific tasks. However, these models are generally assumed to be presented with complete and accurate information, overlooking the challenges of erroneous/missing information in wild scenarios. To address this gap, we introduce the concept of Scientific Model Robustness, which defines a model's ability to detect anomalies, assess solvability, and perform self-calibration when faced with Information Error (inconsistencies) or Information Loss (insufficiency). Specifically, we present RoSci-Bench, a diagnostic benchmark of 1,220 samples using a seed-variant strategy across six degradation types. To address VLM fragility, we further propose RoSci-Agent, a framework that employs a Parse-Verify-Decide pipeline to vet premises before reasoning. The evaluated VLMs sometimes answer questions whose premises are invalid or incomplete. RoSci-Agent achieves a higher Overall Robustness Score than the evaluated standalone VLMs. RoSci-Bench gives the multimodal reasoning community a common test of when models answer, request clarification, or reject a scientific question. 

---
# Shared Understanding in Human-AI Teams: Coordination Continuity in Evolving Work 

**Authors**: Siân Lindley, Britta Burlin, Christian Poelitz, Lev Tankelevitch, Richard Banks  

**Link**: [PDF](https://arxiv.org/pdf/2611.27746)  

**Abstract**: People are increasingly working with AI agents as part of extended, asynchronous workflows. In these settings, shared access to information does not guarantee a shared basis for action: goals, plans, and roles may persist while their current significance changes. We introduce coordination continuity, the extent to which collaborators sustain a workable basis for interdependent action as work progresses. Drawing on research on collaborative work, along with two qualitative team studies and a reflexive case from our own repository-based environment, we present a framework of three elements: durable representations of collaborative work; relative coordination properties that support interpretation of representations in the moment; and interaction mechanisms through which the status of representations can be understood and altered. We use a build-act-drift-rebuild heuristic and the notion of consequential moments to characterise when rebuilding may be needed. We draw eight design implications for collaborative AI infrastructure. 

---
# Quantitative Measurement of Pavement Distress based on Vehicle Camera and Perspective Correction 

**Authors**: Sike Wang, Yizhou Lin, Shuo Li, Yi Jiang, Chengcheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2611.27728)  

**Abstract**: Using image data is becoming an important method for highway agencies to automatically evaluate pavement conditions. However, most studies focus on improving detection accuracy rather than obtaining quantitative measurements, such as the length or area of pavement distress. Although field camera calibration can provide such measurements, it is impractical for large volumes of onboard images collected from different vehicles. This study develops a method to quantitatively measure pavement distress directly from in-vehicle camera images without on-site camera calibration. A quadrilateral region of interest is first identified based on lane detection and used to construct a perspective transformation matrix for image correction. Size correction is then performed using the proportions of road markings in the image and a specified scale, which also enables estimation of the actual width of the selected region. The corrected images are subsequently processed using object detection algorithms to obtain quantitative distress measurements. The proposed method can efficiently perform perspective correction for images collected by the same vehicle camera with minimal manual effort. The measured pavement patch lengths show a trend consistent with pavement friction data. By calibrating vehicle cameras using only image information, this study enables quantitative pavement distress detection without additional field experiments. The approach provides a practical solution for processing large volumes of vehicle-camera data, particularly as connected vehicles generate increasing amounts of uncalibrated roadway imagery. 

---
# Decision Models for Multilingual Emotion Classification: A Comparative Study Against LLMs and Encoders 

**Authors**: Alessandro Piccolo, Francesco De Nicolò, Francesco Manco, Eufemia Lella  

**Link**: [PDF](https://arxiv.org/pdf/2611.27721)  

**Abstract**: Decision models receive a question with a closed list of answers and return a probability for each answer instead of free text. They promise low latency and robustness to prompt wording, but no study has compared several of them with fine-tuned encoders across languages. We ask whether systems used without training on labelled data (zero-shot) can replace fine-tuning, and what the decision format adds over a general-purpose large language model (LLM). We compare three decision models (Jev, OpenAI Decisions, Strands Decider 2B) and two LLMs (GPT-6 Luna, Qwen3.5-2B), given Italian instructions, with two fine-tuned multilingual encoders (tabularisai, XLM-EMO) on four emotions (joy, sadness, anger, fear) in 10 social-media datasets in five languages, with paired difference and equivalence tests. The three best zero-shot systems (two decision models and one LLM) score on average about one macro-F1 point below an encoder fine-tuned on the target dataset; our test sets cannot exclude a gap of 4-5 points, and on single datasets either approach wins by up to 22 points. They beat encoders fine-tuned on another dataset by 8-15 points, though against XLM-EMO this may not extend to new datasets, and instructions in each text's language change their results by under 1 point. Exploratorily, they beat the strongest public emotion model used as is by about 6 points. OpenAI's decision endpoint matches GPT-6 Luna, the LLM it runs on, at a fifth of its median latency, and a 2-billion-parameter decision model beats a same-size general LLM by 10 points, owing to its training, not the decision format. The best zero-shot systems, the LLM included, have an SD of about 1 point across prompts. Without labelled data they are a strong default, and decision models add lower latency and, unlike generative LLM use, per-option probabilities. Fine-tuning won by a wide margin only on the largest dataset, GoEmotions. 

---
# Embodied-Harness: Towards Recursive Self-Improvement for Robotic Manipulation 

**Authors**: Zhenxuan Fan, Zhuoya Wang, Yuqian Yuan, Wenqiao Zhang, Juncheng Li, Jun Xiao, Siliang Tang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27689)  

**Abstract**: Vision-language model (VLM)-based agents offer a promising approach to robotic manipulation, yet remain limited by inadequate memory organization, uncertain failure attribution, and limited reuse of execution experience. We propose Embodied-Harness, a dual-agent framework that improves manipulation by refining the execution harness without updating model parameters. The Actor interacts with the environment through a robot-control interface, while the Reflector converts execution evidence into reusable operational guidance. Hierarchical Memory Architecture maintains short-term execution context and long-term experiential knowledge to support execution and reflection. Evidence-based Experience Reflection grounds failure diagnosis in observed interactions, while Recursive Harness Self-Improvement refines operational guidance across successive attempts. Experiments on RoboTwin 2.0, RoboSPA, and RMBench demonstrate the effectiveness of Embodied-Harness in generalization under environmental variations, spatial reasoning, long-horizon planning, and memory-dependent manipulation. Our method achieves success rates of 82.4%, 76.7%, and 94.4% on the three benchmarks, respectively. Real-world evaluation further yields an average success rate of 96.7% across three manipulation tasks. These results highlight the potential of Embodied-Harness to improve robotic manipulation through experience-driven harness self-improvement. 

---
# Hy-MT3: More Accurate, Robust and Practical Multilingual Translation Models in the Wild 

**Authors**: Mao Zheng, Zheng Li, Fangxu Liu, Tao Chen, Bingxin Qu, Bo Lv, Mingrui Sun, Hongyu Li, Hong Huang, Jinlong Song, Decheng Wu, Guanghua Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27673)  

**Abstract**: Hy-MT3 is the third generation of the Hunyuan multilingual translation model family, including \textbf{Hy-MT3-1.8B}, \textbf{Hy-MT3-7B} and \textbf{Hy-MT3-30B-A3B (MoE)}. Compared with Hy-MT2, Hy-MT3 translates \textbf{more accurately, more completely and more faithfully to user constraints}: on real-world scenario data, the share of samples with mistranslations is reduced by \textbf{43\%}, omissions or additions by \textbf{74\%}, and instruction-following errors by \textbf{40\%}, while language coverage is expanded from \textbf{38 to 102} languages. With only 3B activated parameters, \textbf{Hy-MT3-30B-A3B achieves the best overall performance among all evaluated models in non-thinking mode}, including flagship models such as Qwen3.8-Max, Kimi-K3.1-Pro and DeepSeek-V4.1-Flash, and approaches the strongest closed-source thinking models such as GPT-6.1-Sol and Gemini-4-Argon, while surpassing models that consume \textbf{more than 30$\times$} its inference compute. Beyond general translation, Hy-MT3 is optimized for two industrial scenarios. For \textbf{game localization}, built with high-quality game-domain data and a professional evaluation system from the Honor of Kings team, Hy-MT3-30B-A3B matches or outperforms strong closed-source models such as Gemini 3.1 Pro on all core metrics, and can be plugged into game development pipelines through an open-source toolchain. For \textbf{image translation}, Hy-MT3 improves context understanding and error correction to better tolerate OCR errors, achieving the best result among models of comparable size. Finally, we open-source \textbf{TransRubric}, a rubric-based automatic evaluation suite for translation, which outperforms other evaluation schemes on the WMT26 metrics shared task test set and is more stable with respect to human evaluation. 

---
# Potential emergence in LLM agents: a statistical physics perspective 

**Authors**: Zhuo-Yang Song, Hua Xing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27618)  

**Abstract**: There has been significant recent interest in understanding the behaviour of LLM agents from the perspective of human preferences or corpus statistics. In this paper we propose agent states as intermediate variables through which post-training can be measured and predicted. From a statistical physics perspective, we study the emergence of a global potential which governs the transition of state in LLM agent reasoning. We introduce a partition function to model the ideal KL-regularised transition distribution and track potential structure, mean reward and state entropy. We demonstrate our approach in two Qwen3.5-9B tasks: safety-test selection and physics-reasoning auditing. We observe that the potential structure strengthens during post-training as predicted from our framework, while joint changes in reward and state entropy provide consistency evidence. States thus provide a quantitative description of how post-training organises agent behaviour. 

---
# PromptTTSD: Controllable Text to Spoken Dialogue Synthesis with Open-Ended Text Descriptions 

**Authors**: Sulong Yang, Zhi Zeng, Xiupeng Wu, Juewen Hu, Pring Wong, Shuwu Zhang, Shuo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.27601)  

**Abstract**: Spoken dialogue synthesis aims to generate multi-turn conversational speech from dialogue scripts while maintaining speaker consistency and contextual appropriateness. However, existing models largely lack fine-grained controllability, which is critical for capturing the dynamic, context-dependent style shifts in natural conversations. While prompt-based TTS has enabled such control for single-speaker speech, existing datasets heavily rely on predefined tags, which fundamentally limits their capacity to capture open-ended styles. To address this gap, we propose a scalable pipeline for extracting conversational speech from multi-narrator audiobooks, featuring an improved two-stage alignment algorithm to preserve dialogue continuity, paired with LLM-based extraction of style descriptions directly from narrative text. We release PromptDialog, a large-scale multi-speaker conversational speech dataset comprising 8,383 hours of Chinese speech and 120 hours of English speech, with open-ended style descriptions for 79% of utterances. We further introduce PromptTTSD, a spoken dialogue synthesis model built via a prompt-aware fine-tuning strategy that integrates style prompts into text-speech interleaved sequences. PromptTTSD is an open-source spoken dialogue synthesis model that supports both zero-shot voice cloning and natural language style control. Experiments demonstrate that PromptTTSD not only maintains strong fundamental synthesis capabilities but also achieves competitive open-ended style control compared with proprietary models, and substantially improves contextual appropriateness in multi-turn dialogues. 

---
# Unlocking Heterogeneous Accelerator Flexibility in AI Data Centers: Sizing and Workload Allocation 

**Authors**: Chenghao Lyu, Ying Yu, Wangkun Xu, Fei Teng  

**Link**: [PDF](https://arxiv.org/pdf/2611.27576)  

**Abstract**: AI data centers are an emerging electricity demand, but limited local grid capacity could delay their connection and expansion. While existing operational strategies consider spatio-temporal workload flexibility, few studies address the strategic sizing and allocation of diverse accelerator archetypes with different capital and energy-efficiency trade-offs. This paper unlocks the flexibility potential of heterogeneous accelerators by jointly sizing and allocating different AI workloads (training, delay-tolerant batch work, and online inference requests) across sites and hours to meet the time-varying grid constraint at minimum cost. Workload-specific throughput and energy coefficients link these decisions to facility power. In a representative two-site case, heterogeneous sizing reduces daily cost by 6.5% relative to spatio-temporal scheduling with fixed capacity. When evaluated using workload-specific performance, a capacity design based on workload-averaged hardware coefficients costs 7.7% more than the proposed design. The proposed design also eliminates the envelope violations of an unconstrained design at a 0.25\% daily cost increase. These results show how hardware heterogeneity and workload flexibility jointly shape economical, grid-compliant capacity plans. 

---
# Bringing AI Assistance to the Beznau Nuclear Power Plant: A Hierarchical System of LLM Agents for Reactor Monitoring 

**Authors**: Wenqi Zhou, Leonardo Baldo, Edwin Kolbe, Manuel Arias Chao  

**Link**: [PDF](https://arxiv.org/pdf/2611.27556)  

**Abstract**: Beznau Unit 1 in Switzerland, in operation since 1969, is the most experienced nuclear reactor still running worldwide. Five decades of incremental modernization have left its monitoring systems, alarm logs, variable catalogs, technical specifications and investigation reports in separate silos that were never designed to be queried together. As a result, investigating a core instrumentation anomaly means slow, manual gathering of knowledge scattered across databases, documents, and experienced colleagues. We present a hierarchical multi-agent assistant built on large language models (LLMs) to support these investigations and reduce the need for tedious manual searches in SAP systems. An orchestrating agent breaks down an engineer's question and hands the parts to specialized sub-agents, each with read-only access to one source. Domain expertise is written as natural-language procedures that plant engineers can read, review and revise. The system is strictly advisory, with every answer traced back to its sources. We also report organizational lessons from introducing agentic AI into a culture built on safety, traceability and conservative change. Beyond the technical architecture, the Beznau case study examines how an assistant can be introduced into a setting where evidence must be traceable, changes carefully reviewed, and operational decisions retained by plant personnel, paving the way for the responsible use of agentic AI in industrial plant monitoring. 

---
# On the Limits of Observability for Evaluating Coverage in Machine Unlearning 

**Authors**: Ziluowen Luo, Jun Yin, Songyang Liu, Ji Qi, Chuang Hu, Kyrie You Wu, Chaozhuo Li, Senzhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27543)  

**Abstract**: Machine unlearning aims to remove the influence of any requested subset of training data while preserving utility on the remaining data. Empirical evaluations measure predictive accuracy and infer training-set membership (MIAs) from model outputs or intermediate activations. However, by focusing on selected forget sets, these evaluations can overlook Coverage, the range of requests a method can fulfill. To assess this coverage, we first conduct a stress test of observability with 502 retrained models, finding that hidden embeddings reveal known deletions missed in predictions. To characterize capacity limits on coverage, we show that distinguishing all $n$ singleton deletions with linear scores can require a shared representation of rank $\Omega(\log n)$. We further extend the bound to arbitrary subsets and show when partial deletion can appear complete. Building on this analysis, we propose CEO, a framework for evaluating unlearning coverage under limited observability. Its compositional color-shape benchmark makes forgetting and retained utility measurable from the full output distribution and provides a leave-one-out retraining oracle for each target. From a fixed number of oracle observables, we estimate empirical target coverage, the fraction of targets reliably flagged when deleted and rarely flagged under any other deletions. To recover missed deletions, CEO further expands each observable representation into high-dimensional activation space with the frozen encoder of a SAE trained on the original model. Extensive experiments of 20 baselines shows that even on this simple synthetic task, only retraining from scratch covers all deletion targets. 

---
# Fix Where It Fails: Root-Cause-Guided Agentic Harness Optimization for Scene Graph Generation 

**Authors**: Jee Eun Kim, Donghyeon Choi, Sangjune An, Hae Beom Lee, Kyong Hwan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2611.27529)  

**Abstract**: Scene graph generation (SGG) aims to detect objects in an image and predict the relationships between them. The standard PredCls, SGCls, and SGDet settings show how much performance drops without ground-truth boxes and labels, but not where or why a model fails. Knowing this would allow fixing a model where it fails, without retraining. We propose RoHa-SGG (Root-cause-guided Harness optimization for SGG), which extends agentic harness optimization to SGG. It optimizes the harness, the inference code around a frozen SGG model. RoHa-SGG has two main LLM agents. A root-cause analysis (RCA) agent finds the components where the model fails and diagnoses why, and a proposer agent searches for a harness that fixes these causes. We evaluate RoHa-SGG on Visual Genome and PSG with representative two-stage, one-stage, and multimodal-LLM SGG models. The analysis shows that the components where ground-truth triplets are lost differ across models and datasets. RoHa-SGG improves every metric on all five model-dataset combinations. Graph-constrained R@100 and mR@100 rise by up to 15.8 and 26.4 points for the multimodal LLM and by up to 3.6 and 10.1 points for the two-stage and one-stage models. With root-cause guidance, the harness search reaches the final score of an unguided search up to three times faster. 

---
# PaMuk-Bench: A Parallel Multilingual Multitask Benchmark for Vision-Language Tasks 

**Authors**: Silviu-Florin Gheorghe, Radu-Stefan Popa, Adriana Valentina Costache, Adrian-Gabriel Chifu, Alessia Amelio, Paul Irofti, Anders Skaarup Johansen, Lallouani Bouchakour, Mohamed Lichouri, Khaled Lounnas, Neelu Madan, Yashvir Malik, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27521)  

**Abstract**: While vision-language models (VLMs) currently obtain notable results for highly-studied languages, e.g. English, they exhibit significant performance drops in certain low-resource languages, especially due to the lack of annotated datasets. To this end, we propose PaMuk-Bench, a novel parallel multilingual multitask benchmark for vision-language tasks. Our benchmark includes a set of over 10K curated English prompts that are translated into six languages: Arabic, Danish, French, Hindi, Italian, Romanian. Parallel translations, which are obtained via a semi-automatic process, along with accompanying generated and retrieved images enable the comparative development and evaluation of VLMs across a wide range of tasks: text-to-image generation, image captioning, content-based image retrieval, image generation difficulty estimation, image retrieval difficulty estimation. We empirically show that zero-shot English and multilingual pre-trained VLMs compare consistently worse than their fine-tuned counterparts, confirming that PaMuk-Bench is a valuable resource that can drastically reduce the language gap across a variety of VLM architectures and tasks. PaMuk-Bench is available at this https URL. 

---
# PhiJEPA: Denoising Latent Predictions for Visual Representation Learning 

**Authors**: Makoto Yamada, Yao-Hung Hubert Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2611.27516)  

**Abstract**: Can a predictor trained to infer hidden image regions still help a world model plan when part of an object disappears from view and the visible context is blurred? We introduce PhiJEPA, a joint-embedding predictive architecture that adapts PhiNet's two-stage prediction to masked Transformers. From sparse, photometrically corrupted context, a Transformer block predicts hidden tokens of an independently corrupted teacher view. A token-wise MLP then maps these predictions to clean teacher tokens. Controlled ablations show that the combined objective improves dense features over the tested one-stage alternatives. Pre-trained on ImageNet-1K with $5.6\times10^{19}$ training FLOPs, PhiJEPA achieves 69.3 mIoU on linear PASCAL VOC segmentation, exceeding the evaluated I-JEPA and DSeq-JEPA checkpoints in segmentation accuracy, with 25\% fewer training FLOPs than I-JEPA. It also improves hidden-region segmentation over Simple I-JEPA (49.1 vs.\ 44.3 mIoU). As an application, we use the frozen encoder in DINO-WM on PushT, and reuse the pre-training predictor, which JEPAs usually discard, to complete occluded tokens without fine-tuning. In open-loop planning with simultaneous object occlusion and Gaussian blur ($\sigma=2$ pixels), predictor completion raises success from 33\% to 65\%, compared with 25\% for Simple I-JEPA with its predictor and 40\% for iBOT with pixel inpainting. Together, these results support two-stage latent prediction as an efficient approach to learning visual representations that remain useful under photometric degradation and support completion under partial observation. 

---
# Self-Supervised Continuous-Time Depth Learning and Ego-Motion Estimation from Asynchronous Events 

**Authors**: Sadiq O. Macaulay, Dipon Kumar Ghosh, Hadfield Simon  

**Link**: [PDF](https://arxiv.org/pdf/2611.27509)  

**Abstract**: Event cameras register brightness changes pixel by pixel with microsecond precision, whereas the dense depth that could supervise them is costly to acquire and exists only at isolated instants. Condensing the stream into frames eases learning, but at the expense of the very timing that sets event sensing apart. We propose a self-supervised approach in which each event is assigned its own depth in continuous time, and the camera motion is recovered from the same observations. Supervision arises from geometry: lifted into 3D by its predicted depth and displaced by the camera motion at the instant it fired, an event should coincide with others produced by the same edge, so any misalignment exposes an error in depth or motion. Differentiable rendering then turns these per-event estimates into dense depth maps at target times within each window, faithful to the events behind them. On MVSEC, the approach outperforms prior event-only self-supervised methods in most sequences and depth ranges, with the widest margins at night, and recovers camera rotation more accurately than an earlier unsupervised event-based method. Ablations isolate the roles of local spatiotemporal context, the temporal backbone, and the alignment objective. These results position motion-compensated event alignment as a principled source of supervision for continuous-time depth and ego-motion estimation. 

---
# The Assistant Persona Differs Across Languages in Activation Space 

**Authors**: Daniel Yoo, Naama Rozen, Evan Harris  

**Link**: [PDF](https://arxiv.org/pdf/2611.27504)  

**Abstract**: Large language models behave differently across languages, and are noticeably less safe in lower-resource languages. This gap is well established behaviorally, but not mechanistically. Directions in activation space, including the assistant persona direction, are almost always extracted from English and assumed to generalize across languages. We extract the assistant persona direction in 31 languages in Qwen3-32B, and find that the resulting axes differ substantially: their cosine similarity to the English axis ranges from 0.378 to 0.911. Moreover, each language's assistant persona direction and its magnitude are correlated with the language's resource level. We also find that the model's default assistant state is farther from the helpful assistant pole in lower-resource languages. Finally, steering against a language's own axis amplifies persona-based jailbreak more than steering against the English axis. These results indicate that the model does not carry a single assistant representation across languages. More broadly, the assistant persona may need to be an explicit target of multilingual alignment. 

---
# Turnout: Adaptive Attention Switching for Efficient Reasoning 

**Authors**: Anna Kuzina, Mart van Baalen, Paul N. Whatmough, Babak Ehteshami Bejnordi  

**Link**: [PDF](https://arxiv.org/pdf/2611.27500)  

**Abstract**: Post-hoc transformer linearization replaces the key-value cache of full attention, which grows linearly with generation length, with a fixed size state. This change reduces memory consumption but often hurts accuracy. In reasoning tasks, which require long generations, this effect is especially pronounced. In this work we introduce Turnout, a method that mitigates accuracy loss as a result of linearization, while maintaining its memory-reduction benefits. We apply post-hoc linearization to a pretrained full-attention model, training a linear path through distillation while leaving the original full-attention path unchanged. During response generation, the resulting hybrid model retains both paths and uses a learned policy to control when each layer switches from full to linear attention. We train the policy with reinforcement learning using a reward that balances answer correctness and memory use. Empirical evaluation of our method shows that the learned adaptive policy achieves a better accuracy-efficiency Pareto frontier on reasoning tasks compared to fixed switching and static hybrid baselines. 

---
# Learning Unknown Oscillators From Bandit Feedback 

**Authors**: Priyank Jayraj, Poonam Goyal, Navneet Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2611.27492)  

**Abstract**: We study semi-bandits driven by unknown oscillators. Each round, a learner selects two of four reward channels and observes their individual rewards. Frequency errors change future reward rankings and the resulting measurements. We show when committing to one fitted oscillator lets its greedy choices expose prediction errors. Our controller bounds reward loss (regret) over a continuous confidence set and holds the longest block within budget. An online regression policy gathers observations when the check fails. Regret grows at a square-root rate, up to logarithmic factors, across all frequencies and extends to two competing oscillators. On 96 fresh synthetic instances, checked commitment uses 49 fits instead of 1,023, with 1.5% higher mean regret than fitting every round. Matched isolated timings give a 4.69-fold speedup. The confidence sets also certify trajectory regret. 

---
# Alignment without Transfer: Tokenization Impacts Cross-lingual Representations during Multilingual Language Model Pretraining 

**Authors**: Jonathan von Rad, Pontus Stenetorp, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27491)  

**Abstract**: Tokenization represents one of the most crucial steps in mulilingual NLP pipelines. Yet, tokenizers routinely under-allocate vocabulary to low-resource languages. While improved allocation leads to better performance, it remains poorly understood how tokenization shapes cross-lingual representations and transfer during pretraining. To address this, we pretrain 39 mono- and bilingual 1B decoder-only models on FineWeb2-HQ across English, German, French, Arabic and Chinese, varying only the tokenizer. Our results demonstrate that improved vocabulary allocation (i) substantially raises cross-lingual representation alignment in early-to-mid layers, an effect that prior work measuring only the output layer could not observe, (ii) increases cross-lingual parametric knowledge sharing in feed-forward neurons, supported by correlational and causal evidence, and (iii) an alignment-transfer dissociation, where cross-lingual transfer remains essentially unchanged despite higher alignment. Moreover, transfer is significantly lower for cross-script bilinguals even under fair vocabulary allocation, indicating that the script barrier is not merely a tokenizer artifact. All of our checkpoints and code are publicly available. 

---
# ROSWORLD: When Digital Task Competence Meets Physical Execution 

**Authors**: Caoyuan Ma, Wendi Liu, Zhimin Zhang, Muyao Niu, Yujie Lu, Zengxi Zhang, Ji Zhao, Shuai Dong, Wenpu Liu, Ziyue Wang, Wenzheng Chang, Taiqiang Wu, Weichu Xie, Yuqi Xu, Haiyuan Wan, Xuehui Wang, Jiaqi Wang, Wenqi Shao, Yinqiang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.27467)  

**Abstract**: Computer-use agents now solve most desktop tasks through direct software input, but a robot using the same computer must turn GUI knowledge into physical contact with a keyboard and mouse. We introduce ROSWorld, a benchmark that couples a physically simulated robot workstation to a live OSWorld desktop, so that a robot and a desktop agent pursue the same goals under the same official evaluators. The evaluated model receives observations only from the workstation camera views and operates the peripherals through either a single-arm rigid probe or dual-arm dexterous hands. Across five frontier models and 40 tasks in ten applications, models that complete 31-39 tasks on the desktop complete at most 23 with the probe and 28 with the hands. Paired trajectory analysis shows that models adapt their interaction strategies toward keyboard input, yet physical execution frequently fails through missed presses, incorrect targets, and unintended contacts. ROSWorld makes the gap between knowing what to do on a computer and physically doing it measurable. 

---
# Cecilia: Towards Layout-Inclusive Optical Music Recognition 

**Authors**: Congren Dai, Zhancheng Guo, Xiaobing Li, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.27465)  

**Abstract**: Reconstructing an editable music score requires musical events, sung text and the instructions that organise them on the page. Existing recognition targets often omit part of this information, while direct MusicXML prediction repeats substantial structural markup. We introduce Cecilia, a compact representation of canonicalised MusicXML that retains musical content, lyric attachments and layout instructions. Cecilia-OMR predicts this representation through metadata and measure-bounded music segments from shared visual memory. A learned count head determines the number of output segments. Across paired complete scores under a common tokeniser, the ratio of mean MusicXML to Cecilia sequence lengths is 27.73. Cecilia-OMR outperforms Legato 2 on Camera String Quartets and both Lieder conditions, with a four-condition OpenScore macro normalised edit distance of 0.2232 against 0.2998. Separate evaluations of lyrics, notation trees and rendered layout reveal complementary aspects of reconstruction and quantify the contribution of layout instructions. Training-stage, reward and candidate-selection studies further examine how recognition changes across model adaptation and decoding. 

---
# On the Challenges of Using LLMs in High-Risk AI Recruitment: Guiding Questions for Legal Conformity with the AI Act 

**Authors**: Clara Punzi, Stefano Tramacere, Andrea Fedele, Sezer Kutluk, Roberto Pellungrini, Gizem Gezici, Paolo Baracco, Laura Castello, Eleonora Cusin, Eduardo Costa, Amedeo Giacovelli, Mario D'Almo, Fosca Giannotti, Dino Pedreschi, Marzio Di Vece  

**Link**: [PDF](https://arxiv.org/pdf/2611.27462)  

**Abstract**: The EU Artificial Intelligence Act (AI Act) imposes detailed obligations on providers of high-risk AI systems but offers limited operational guidance, particularly when these systems integrate third-party General-Purpose AI (GPAI) models. This paper introduces a toolkit that fosters the development of legal-by-design AI-assisted recruitment systems by translating the core legal obligations of the AI Act's conformity assessment---specifically Articles 9--15 and Annex IV---into techno-normative requirements and corresponding guiding questions. While recruitment pipelines may involve components with varying technical complexity and differing levels of algorithmic transparency, this paper focuses on the most challenging case: the integration of third-party GPAI models into critical stages of the selection process, including resume and job description parsing, applicant scoring, and shortlist generation. This scenario presents significant challenges for downstream providers of high-risk AI systems, as it underscores the difficulties in obtaining the technical information needed to demonstrate compliance with the AI Act's techno-normative requirements, both prior to and during market deployment. After introducing the toolkit and illustrating its application through a simplified example, the paper explores informational dependencies between upstream and downstream GPAI model providers. These dependencies may produce informational gaps across the AI supply chain, potentially hindering the collection of evidence necessary for the AI system provider to ensure legal compliance. The resulting lack of technical transparency negatively affects the deployer, who may struggle to maintain meaningful oversight of the system, and ultimately impacts the persons subject to the AI output. 

---
# Mechanism Parity for Simulation-Based Validation of Autonomous-Driving Perception 

**Authors**: Danial Safaei, Siddartha Khastgir, Matthew D. Higgins, Jeroen Ploeg, Mohsen Alirezaei, Son Tong, Paul A. Jennings, Xingyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.27447)  

**Abstract**: Simulation is central to intelligent-vehicle validation, but visual realism or output agreement between a real scene and its synthetic counterpart does not prove that a perception system used the same evidence in both domains. This paper addresses that mechanism-parity gap for paired real-synthetic autonomous-driving perception. It unifies Decisive-Feature Fidelity, which compares decisive evidence maps in a frozen system under test (SUT), with Concept-Decisive Feature Fidelity, which projects decisive evidence into semantic scene concepts for diagnosis. The framework is a validation diagnostic rather than a standalone safety certificate: it asks whether output-consistent simulation also preserves the mechanism by which the perception model reaches that output. The empirical evidence comprises ten retrained diffusion ControlNet generators evaluated on a deterministic exact-geometry KITTI-Virtual KITTI 2 subset. Generated counterparts use a 1248-by-376-pixel padded inference canvas and are cropped back to the exact KITTI dimensions before evaluation. The final benchmark reports target-image, output-consistency, and mechanism metrics recomputed from a single audited output set, then adds metric-disagreement cases, bootstrap uncertainty, threshold sensitivity, and concept/cache integrity audits. Across the benchmark, output and appearance fidelity are at most weakly and inconsistently rank-correlated with decisive-mechanism distance for all three frozen SUTs, and a constructed positive control shows the estimator detects a known planted mechanism that output agreement misses, supporting decisive-evidence diagnostics for intelligent-vehicle simulation validation. All numerical claims are restricted to the declared exact-geometry benchmark, ten generators, and three frozen SUTs. 

---
# Learning When to Ask: Advisor-Aware Post-Training for SLM Agents 

**Authors**: Aasheesh Singh, Vishal Vaddina  

**Link**: [PDF](https://arxiv.org/pdf/2611.27443)  

**Abstract**: Small language models (SLMs) are increasingly attractive for cheap, high-throughput agentic deployment, but they remain weak at multi-step reasoning and tool use that long-horizon tasks demand. We study how to post-train an SLM to seek advice from a stronger model and act on that advice itself, following an advisor-executor pattern. Out of the box, this pattern fails across open- and closed-source small-model families: wiring an SLM to an advisor is a harness-design matter that is not enough on its own. A post-training recipe is needed to teach the model when to ask, in a cost-aware fashion, and how to follow advice across steps. Our method, ADAPT (ADvisor-Aware Post-Training), is a multi-stage recipe that instills this behavior, learning entirely from the policy's own rollouts supervised by the environment's outcome, with no learned reward model or human annotation. Across three domains, compared to an untrained harness baseline, ADAPT improves success rate in embodied text reasoning (TextWorld) from 24.5% to 77.3%, code generation (LiveCodeBench) from 21.8% to 56.0%, and deep web search (GAIA) from 17.5% to 49.3%, while querying the advisor sparingly (~1.0-2.1 calls per task). On the two long-horizon domains it closes 70-85% of the gap to an oracle that perfectly times when to ask for advice, and on code generation it exceeds that oracle. The gains persist when the advisor is replaced by one unseen in training, showing that the learned help-seeking policy is robust to a changing advisor rather than tied to the one it was trained with. 

---
# THRIFTY: Cost-Constrained Reasoning-Data Synthesis for Small Language Models 

**Authors**: Aasheesh Singh, Vishal Vaddina  

**Link**: [PDF](https://arxiv.org/pdf/2611.27436)  

**Abstract**: Small language models (SLMs) are increasingly attractive for efficient, specialized deployment, and are typically trained on reasoning trajectories distilled from a larger teacher LLM. Generating this synthetic data is a major cost of the pipeline, especially when the teacher is a frontier model. Existing collection methods, however, are largely cost-unaware, or at best spread a fixed sampling budget uniformly across queries. Such allocation ignores the wide variation in problem difficulty and the strategic over- or under-sampling it invites. Thus, we propose THRIFTY, a Thompson-sampling bandit controller that spends a fixed generation-token budget to collect diverse correct reasoning trajectories, casting the task as a bandits-with-knapsacks problem with a novel diversity-aware objective. It tracks each problem's success probability with an online Beta posterior and spends each token where it expects the most new distinct correct solutions. Because repeated solutions add progressively less value, THRIFTY adaptively stops resampling saturated problems, concentrates budget on under-covered ones, and retires the unsolvable at the budget's opportunity cost. On math (AIME, MATH-500) and code (LiveCodeBench) benchmarks, THRIFTY gathers up to 22% more diverse correct trajectories per generation token than both the standard uniform-sampling recipe and the strongest adaptive baseline, with the largest gains in the scarce-budget and hard-problem regimes. 

---
# No Audio, No Transcripts, No Problem: LLM-based ASR Adaptation Using Only Domain Documents 

**Authors**: Sergio Burdisso, Dairazalia Sánchez-Cortés, Esaú Villatoro-Tello, Thibault Bañeras-Roux, Manjunath K E, Kadri Hacioglu, Petr Motlicek, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2611.27434)  

**Abstract**: Can a speech recognizer be adapted to a new domain with no target audio and no transcripts, using only written documentation? This is the realistic onboarding setting: a new customer hands over raw documentation, not the transcripts that text-only adaptation normally assumes. We turn a domain's documents into adaptation text in two ways, directly as written sentences or rendered by a large language model (LLM) into synthetic spoken dialogs, and feed them to a recently proposed denoising-based text-only adaptation method. We show that, across in-domain, out-of-domain, and cross-domain settings on two corpora, document-only adaptation can match or closely approach real transcriptions; whether written or conversational rendering is best depends on speaking style and the amount of text generated. To our knowledge, this is the first study of document-only adaptation of LLM-based ASR. As no existing corpus pairs audio, transcripts, and documents, we also built and released a document-grounded benchmark. 

---
# Neurodegenerative Differential Diagnosis with Knowledge-enhanced Structural Reasoning 

**Authors**: Mingxuan Wang, Siyuan Du, Yanfeng Wang, Ya Zhang, Chencheng Zhang, Jiangchao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.27427)  

**Abstract**: Accurate differential diagnosis of neurodegenerative disorders remains challenging due to overlapping clinical manifestations and heterogeneous, often incomplete multimodal observations. Existing multimodal models can integrate imaging and clinical measurements but often overlook disease-specific evidence patterns and pairwise relations among competing diagnoses. We therefore propose Knowledge-enhanced Structural Reasoning (KnowSR), which aligns multimodal patient representations with a literature-grounded knowledge bank of disease prototypes, signed evidence directions, and pairwise diagnostic relations. Specifically, in KnowSR, patient-specific knowledge alignment and target-aware pooling inject structured disease evidence, while anchored relational reasoning reconciles pairwise diagnostic margins over a sparse differential graph, thereby collectively enhancing differential diagnostic capability. Across extensive experiments, KnowSR consistently outperforms 19 baselines on patient-level six-class diagnosis under patient-disjoint five-fold evaluation. This shows that grounding patient evidence in structured medical knowledge enables more effective differential diagnosis than naive multimodal fusion alone. 

---
# Cleave: Severing Guardrails and Auditing Spectral Coupling in Open-Weight LLMs 

**Authors**: Feng Huang, Qiang Huang, Yue Gao, Mei Zhang, Bin Liu, Lihua Ju, Xinhao Wei, Yongfeng Huang, Yu Qing, Yunqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27426)  

**Abstract**: Open-weight Large Language Models (LLMs) are widely downloaded for their strong coding and reasoning capabilities and deployed locally beyond the provider's control. To prevent misuse, developers embed safety guardrails into these models before release. However, we find that these guardrails rely on the residual writing weights to trigger refusals. Based on this insight, we propose Cleave, a training-free attack that directly edits these weights to remove safety restrictions while preserving general capabilities. With Cleave, malicious users can either remove all guardrails or selectively lift only cybersecurity restrictions. Because this selective removal retains up to a 99% refusal rate on other harmful requests and avoids weight rank collapse, it evades both behavioral spot-checks and static weight audits. Across five open-weight LLMs from 7B to 320B parameters, Cleave achieves up to a 67.97% attack success rate on seven safety benchmarks, substantially outperforming existing white-box baselines. On 356 cybersecurity tasks and 15 live vulnerable services, it further boosts executable exploit generation by up to 24.0 times and compromises up to 66.7% of the targets. To counter this threat, we propose a dual-stage defense that combines pre-release subspace diagnostics with post-release weight-coupling audits, achieving 100% detection recall at a 1% false-positive rate. 

---
# TrojanMaxxing: One-shotting VLM Backdoors with Weight-Space Programming 

**Authors**: Ahaan Dabholkar, Harshvardhan Patel, Michalis Polychronakis, Saurabh Bagchi  

**Link**: [PDF](https://arxiv.org/pdf/2611.27425)  

**Abstract**: Recent work shows that large language models (LLMs) can reason directly about the weights of other neural networks. Given only a natural-language objective, an LLM can inspect a model's weights, understand what they compute, and edit them directly without gradient-based optimization. This capability, termed weight-space programming (WSP), has been used to improve model performance on math reasoning, medical QA and other benchmarks. We propose TrojanMaxxing, a novel strategy to backdoor vision-language models (VLMs), by reframing it as a WSP task. We use Claude Opus 5.5 and Mythos 5.2 agents to backdoor VLMs without any optimization or training methods. The agents inspect the model weights, apply targeted edits, and co-design an imperceptible visual trigger that activates the backdoor. We then study the agents' strategies and present a thorough evaluation of the trojaned models on various benchmark datasets. TrojanMaxxing achieves attack success rates of 93%-100% on content moderation and 78%-100% on OCR while maintaining clean accuracy. Perhaps most consequentially, we show that injecting backdoors in a training-free manner is possible by simply, "asking an agent to do it". 

---
# ADEPT: Adaptive Monocular Depth Estimation via Model-In-The-Loop Photometric Tuning with Capture Risk Prediction 

**Authors**: GiHyun Kim, JinMo Yang, Hyung-Sin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.27423)  

**Abstract**: Exposure and gain determine the visual evidence available to monocular depth estimation (MDE), particularly when robot motion creates a tradeoff between light collection and motion blur. We present ADEPT, a framework that adapts these settings using learned feedback from a frozen depth model. During training, ground-truth depth identifies the lowest-error reference within a group of physically evaluated camera settings. A compact predictor learns reference-relative disagreement (RRD) between each candidate's depth prediction and the reference prediction using regression and ranking losses. At deployment, it scores observed captures using frozen-model features and camera, motion, and illumination context, without ground truth or reference images. A contextual Gaussian-process controller uses recent scores to select subsequent exposure and gain settings under motion and photometric constraints. We evaluate offline camera selection with relative and metric Depth Anything v2 and demonstrate online control with the affine-invariant variant on a mobile robot. Across nine equally weighted scene-illumination conditions, ADEPT reduces mean scale-and-shift-aligned $\mathrm{Abs}_{\mathrm{rel}}$ by 14.6% relative to built-in auto-exposure and by 10.0% relative to the best-performing evaluated prior controller. 

---
# Detecting Word-Substitution Poisoning of a RAG Knowledge Base with Relative Entropy and Runtime Verification 

**Authors**: Ali Yousefi, Nikos Komninos, Hafizul Asad  

**Link**: [PDF](https://arxiv.org/pdf/2611.27410)  

**Abstract**: A retrieval-augmented generation (RAG) system answers questions using passages it retrieves from a knowledge base, so users must trust that knowledge base. We study an attacker who quietly edits existing passages by swapping words for synonyms (PWWS, TextFooler) in a cybersecurity RAG system built from MITRE ATT&CK and CAPEC. Using a reviewed 44-question benchmark, repeated paired runs and a control corpus in which only the layout changes, we find that poisoning lowers judged answer correctness from 0.714 to 0.555. The damage comes mostly from lost evidence and refusals, not from false statements. To find the edited passages, we use relative entropy. Comparing a whole passage with normal text is a weak signal. Still, one term of the relative entropy, the log-ratio between a word and its most probable synonym, works well: with a small language model as the reference, it reaches an AUROC of 0.95 and stays near chance on the control. Used as a retrieval filter, it halves how often edited text reaches the generator, and an attacker who must keep each passage as probable under the language model fails on all 57 sections it tries. Filtering does not restore correctness, because the edited passage is usually the one that held the answer. We therefore add a runtime monitor that checks each answer against trusted MITRE STIX data and remembers the passages it rejects. On its own, it reduces exposure to edited text from 0.41 to 0.26; when it may add the trusted MITRE entry as a source, correctness rises to 0.786. We report the costs and failure cases of every defence alongside its gains. 

---
# Taking Apart and Putting Together: Tracking Objects Through Physical Transformations 

**Authors**: Davide Gardenal, Mingxuan Liu, Flavio Vella, Quentin Guimard, Elisa Ricci  

**Link**: [PDF](https://arxiv.org/pdf/2611.27398)  

**Abstract**: We introduce Transformation Tracking and Understanding (TTU), extending video object tracking to environments where objects undergo physical assemblies and disassemblies. TTU requires continuously tracking objects through transformations, identifying new entities, maintaining a state graph of their relationships, and assigning semantic labels to transformations. While foundation models (e.g., SAM 2/3) maintain tracklets through appearance changes, they cannot update semantic identities as objects physically evolve. Prior work addresses only disassemblies, relying on expensive searches triggered only when tracking fails. We propose Tracking Instances via Proposals and Overlaps (TIPO), a training-free framework that continuously infers transformations via a generate-evaluate-judge pipeline. TIPO generates split and merge proposals from disjoint mask components, scores them against mask-decoder priors using inner and external overlap, and queries an MLLM on the top proposal for visual verification and semantic labeling. To benchmark TTU, we curate Video Instances, States, and Transformation Annotations (VISTA), a dataset including state graphs with 124 assembly and 132 disassembly events across 50 sequences and over 26,000 frames. TIPO achieves state-of-the-art tracking and state-graph accuracy on VISTA, while operating $6\times$ to $15\times$ faster than prior methods. Project page: this https URL 

---
# PanoUAV-Track: Full-Spherical UAV Visual Tracking in the Wild 

**Authors**: Bingyu Li, Lichang Yang, Zhiyuan Zhao, Junyu Gao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.27383)  

**Abstract**: Existing UAV tracking benchmarks predominantly rely on perspective cameras with limited fields of view. Omnidirectional cameras enable full-spherical observation, yet tracking from moving UAV platforms remains largely unexplored. Unlike ground-level panoramic tracking, UAV-mounted omnidirectional cameras observe targets across both azimuth and elevation, introducing geometric challenges under equirectangular projection (ERP). To investigate this setting, we capture approximately \textbf{40 hours of real-world 8K panoramic UAV video} and construct \textbf{PanoUAV-Track} through temporal clip extraction, target annotation, and the integration of simulated sequences. The resulting benchmark comprises 1,895 sequences, 507,961 frames, and 2,872 annotated objects, covering real-world daytime and nighttime scenes alongside simulation, and supporting both visual and language-based target specification. Through extensive evaluations, we reveal a clear \textbf{perspective-to-spherical} generalization gap and identify tracking degradation associated with latitude-dependent distortion and ERP seam discontinuity. To address these challenges, we propose \textbf{PanoUAV-Tracker (PUTrack)}, a geometry-aware tracking framework combining latitude-adaptive search and seam-aware periodic sampling. PUTrack improves zero-shot and in-domain tracking, particularly for seam-crossing targets. Code and benchmark resources will be available\footnote{\textcolor{tabpink}{this https URL}}. 

---
# VMSA: Video Memory Skill Agent for Long Video Understanding 

**Authors**: Changin Choi, Jungwon Park, Jungmin Ko, Jimyeong Kim, Wonjong Rhee  

**Link**: [PDF](https://arxiv.org/pdf/2611.27381)  

**Abstract**: Long-video question answering requires retrieving and integrating relevant evidence from videos spanning hours or even days. Recent agentic approaches organize video information into external memories and iteratively retrieve evidence to answer questions. However, effective retrieval and reasoning remain challenging, as agents must determine what information to search for, which memory sources to access, and how to connect evidence across distant temporal intervals. To address these challenges, we propose Video Memory Skill Agent (VMSA), a framework that jointly designs structured video memories and memory skills for long-video understanding. VMSA constructs three complementary memory components. Segment Caption Memory and Entity Event Memory serve as searchable evidence sources, while Entity-Aware Episode Memory provides broad yet detailed video context to guide evidence retrieval and reasoning. To further support effective memory utilization, VMSA induces memory skills from successful agent trajectories, incrementally refining or extending a compact skill library to guide iterative evidence retrieval, reasoning, and verification. Across seven long-video question-answering benchmarks, VMSA achieves the highest accuracy on all three week-long egocentric benchmarks, outperforming the strongest baseline by up to 9.4% on Ego-R1 Bench, and the best or competitive performance on four additional benchmarks. Ablation studies further demonstrate the contributions of all three memory components and memory skills, with memory skills also improving temporal evidence localization. 

---
# Testing for Coordination Failures in Cooperative Multi-Agent Reinforcement Learning 

**Authors**: Dennis Gross, Jørn Eirik Betten, Helge Spieker  

**Link**: [PDF](https://arxiv.org/pdf/2611.27368)  

**Abstract**: Deep parallel cooperative multi-agent reinforcement learning (CMARL)} trains multiple agents that all act at every time step to optimize a common objective. Like single-agent RL, CMARL can produce unsafe behaviors that violate implicit safety requirements not captured by the reward, and these behaviors are hard to explain. However, testing CMARL policies for safety has received less attention than the safety testing of single-agent RL policies. In this paper, we propose DisQ, a test case selection method that identifies failure-revealing states in value-based CMARL by exploiting a signal specific to trained Q-learning agents. The signal is the Q-value gap, an explainable metric that flags a state when the agents' estimates of the achievable return diverge by more than a calibrated threshold. DisQ selects such states for execution and skips those without a gap. Our experiments in four CMARL benchmark environments show that, in most of the evaluated configurations, DisQ improves failure detection over single-agent RL testing methods applied to CMARL, namely random testing and search-based testing with genetic algorithms. Our main contributions are the Q-value gap, an indicator of potential miscoordination computable from a single state without executing the system, and DisQ, a test case selection method that adapts existing single-agent testing methods to CMARL. 

---
# WaXup: Weight Averaging of Experts as Data Augmentation for Robust Introspection 

**Authors**: Marc Boëlle, Lucas Dixon, Alexandre Ramé  

**Link**: [PDF](https://arxiv.org/pdf/2611.27353)  

**Abstract**: As the capabilities of large language models keep improving, traditional mechanistic interpretability methods struggle to keep pace. In contrast, trainable introspection methods emerge as a promising alternative; they repurpose the model itself as a self-diagnostic tool, trained to verbalize what it has learned during finetuning in natural language. While training on datasets of finetuned experts achieves encouraging results, it currently lacks robustness and generalization. Building on the data augmentation framework of MiXup, we introduce WaXup, a method that augments the introspection training set through linear interpolation in the weight space. This is motivated by the linear mode connectivity property: weight averaged experts interpolate the behaviors of the initial experts. We evaluate WaXup on three benchmarks: hidden-behavior models, a meta-dataset of real-world NLP tasks and AuditBench, a challenging alignment auditing suite of models adversarially trained to conceal their behaviors. WaXup improves the accuracy of generated descriptions as well as robustness to various distribution shifts. We hope to improve the reliability of trainable introspection methods towards better and safer models. 

---
# PAVE: A Harness for Geometrically Consistent Road Network Modeling with Vision-Language Models 

**Authors**: Yajun Liu, Jinxi Wu, Juan Pablo Millán Raya, Xuesong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.27352)  

**Abstract**: Transportation models, whether they simulate traffic or support planning, are only as reliable as the road networks they run on, down to the lanes and turns at each intersection. Practitioners still take such a network from GIS files and supply its junction and movement data by hand, a task that recurs with each new base year and design alternative, or they generate it from open map data. OpenStreetMap's lane tags give the correct layout for only 13 of the 44 approaches we test; open map data are often incomplete or inaccurate. Satellite and street-level imagery, in contrast, cover all 44 and show their lanes as painted. Vision-language models (VLMs) can read such imagery, but asked directly they lose geometric consistency when the same lane must be matched across a satellite crop, street-level photos taken from different poses and an ID-labelled network. We therefore present PAVE (Perceive, Anchor, Verify, Edit), a harness in which code measures what can be measured, traffic engineering knowledge enters as general rules and checks, and the model makes only the open judgements, each anchored to network IDs so that checks catch answers that contradict a measurement, a second reading or a traffic rule. Scored against human labels at signalised intersections, PAVE raises the share of exactly correct lane layouts from 49-64% with a single request to 86-91% for two VLMs, with most of the gain coming from reading one approach at a time on traced lanes. The same division of work edits a real network to a planning brief: the model names the links and nodes of each part of a facility, and code builds the edit from a design template. Of 18 edits, 16 pass every network check, against none when the model writes its own code, and 2 contain a design or coding error, against 7. An analysis shows where VLMs lose geometric consistency and which check catches each failure. 

---
# Learning Interaction Dynamics via Motion-Region Guidance for Robotic Video World Models 

**Authors**: Yongkang He, Xiaofen Xing, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27344)  

**Abstract**: Action-conditioned video world models imagine future observations under candidate robot actions, providing a promising visual tool for robot policy evaluation and interaction simulation. However, existing video world models often optimize global reconstruction errors instead of emphasizing interaction-relevant regions. As a result, they may preserve overall scene appearance but overlook local interaction details, leading to physically implausible outcomes such as hallucinated contact effects and inconsistent robot-object interactions. We introduce MoReWorld, a unified motion-region-guided framework that uses a shared motion region, the part of the environment that moves in response to the robot's actions, to focus dynamics learning and video generation on interaction-relevant regions. The same motion region guides local motion prediction, region-aware supervision, and motion-gated future-feature conditioning. At inference time, the motion region is predicted as a motion probability map from visual history and robot actions, without requiring future observations or ground-truth motion labels. On WorldSimProbe, MoReWorld achieves the highest overall simulator-faithfulness score, with the strongest performance on local consistency, global consistency, and interaction dynamics. These results indicate that motion-region guidance helps the model preserve scene-level consistency while more faithfully predicting action-conditioned interaction dynamics. 

---
# Unimodal Shortcuts and Joint Scene-Intent Dependence in Vision-Language Safety Representations 

**Authors**: Guijia Zhang, Hongbo Qiu, Guohao Cai, Hao Zheng, Jiajun Zha, Hanqing Wang, Zhenhao Zhang, Huihao Jing, Shing-Chi Cheung, Haoxuan Che, Harry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27339)  

**Abstract**: The safety of a request to a vision-language model often depends on the image: asking how to serve a dish is harmless for fresh food and harmful for spoiled food. High accuracy on situational safety benchmarks does not tell us whether a model's safety representation encodes both the scene and the user's intent, because a benchmark can often be solved from one modality alone. Studying frozen vision-language safety representations, we audit five public situational benchmarks and find that each admits a unimodal shortcut: images alone predict the labels, and on same-image prompt pairs a probe that never sees the photograph reaches 0.9645 AUROC. We therefore build a controlled evaluation that crosses safe and hazardous scenes with observation and action requests, so that no single-modality rule can exceed 0.500 group-ranking accuracy. On this control, linear probes on intermediate states of Qwen3-VL-8B and SmolVLM2-2.2B reach 0.958 and 0.930, showing that the scene-intent conjunction is linearly decodable; concatenating separately encoded image and text states already recovers most of this signal, so decodability is not evidence of a fusion mechanism. The same probes screen cheaply but generalize poorly to unseen images and other benchmarks. Reliable multimodal safety evaluation therefore needs controls that rule out unimodal shortcuts before accuracy is read as evidence of joint understanding. 

---
# Reasoning in Fewer Tokens: Concept-Normalized Multilingual Scratchpads and Reversible Model-Adaptive Reasoning Codes 

**Authors**: Rabimba Karanjai, Suryabhan Singh Hada, Hemanth Hegadehalli Madhavarao, Qun Gu, Libin N. George, Wenhuan Sun, Neha Monga, Richard Williamson, Prakhar Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2611.27326)  

**Abstract**: Chain-of-thought prompting significantly improves the reasoning capabilities of large language models (LLMs) but incurs high inference costs due to lengthy autoregressive decoding. While previous efficiency approaches reduce costs by discarding tokens or utilizing continuous latent representations, these lossy methods fundamentally sacrifice interpretability and auditability. In this work, we introduce a novel compression framework for intermediate reasoning. We apply Byte-Pair Encoding (BPE) to reasoning corpora to learn a compressed vocabulary of Private Use Area (PUA) tokens, each representing frequent semantic spans. After expanding the LLM's vocabulary, we fine-tune the model via strictly 1:1, our method is lossless. We evaluate our approach on Llama-3.1-8B, Qwen2.5-14B, and Qwen2.5-32B across four diverse benchmarks. Our method reduces total generated tokens by 14.7% on average with reasoning trace lengths dropping by up to 27.1%, while incurring a negligible average accuracy cost of 1.6 percentage points. Crucially, this approach is orthogonal to existing lossy methods, allowing practitioners to stack lossless and irreversible compression techniques. We demonstrate that a compressed intermediate language provides a highly efficient computational resource that accelerates inference while fully preserving the transparency of LLM reasoning. 

---
# Do Language Models Know Which Words Steer Their Reasoning? 

**Authors**: Yanbei Jiang, Chao Lei, Jey Han Lau, Krista A. Ehinger, Rena Gao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.27317)  

**Abstract**: While chain-of-thought (CoT) prompting elicits intermediate reasoning steps from language models, it is unclear which words in the context actually steer the CoT. We identify Phantom Context Influence (PCI): language models are disproportionately steered by semantically irrelevant context words, even as the same models deny those words' relevance when asked. To localize PCI, we introduce Leave-One-Out Word JSD analysis, which removes one context word at a time from the otherwise intact context to isolate its marginal effect on the next-token distribution; words whose removal flips the prediction (pivot words) are passed to the same LLM to self-judge whether it recognizes this predictive dependence. Across four models and three benchmarks, language models largely do not know which words steer their reasoning, a mismatch we call the Generation-Judgment Contradiction. Inspired by this phenomenon, we propose Online KV Suppression: it scores every prompt word's influence over the whole chain of thought in a single pass, then zeroes the KV-cache entries of the highest-influence word the model itself judges inert and redecodes. This single-word intervention improves CoT accuracy across four models and five reasoning benchmarks, by up to 8.9 percentage points on math reasoning benchmark. 

---
# A Framework for AI-Ready Data in Earth and Environmental Science 

**Authors**: Tobias Weigel, Jan Bumberger, Danu Caus, Christian Chwala, Romy Fösig, Sonja Hänzelmann, Sibylle K. Hassler, Beate Hetenyi, Ralf Kunkel, Katharina Loewe, Ulrich Loup, Yueling Ma, Stefan Neuhaus, Kaleb Phipps, Tino Polen, Maren Rebke, Björn L. Saß, David Schäfer, Timm Schoening, Emanuel Söding, Paula Stock  

**Link**: [PDF](https://arxiv.org/pdf/2611.27313)  

**Abstract**: Artificial intelligence (AI), specifically machine learning (ML), is increasingly integral to Earth and environmental research, yet the suitability of current research data infrastructures for supporting AI workflows remains poorly defined. While the FAIR principles provide a foundation for scientific data management, they do not explicitly address several requirements arising when datasets are used for training, evaluating, and deploying AI models. This article presents a practically motivated framework for assessing and improving AI-readiness of research data within distributed Earth and environmental data infrastructures. The framework was developed through a structured, community-driven process, beginning with a review of existing literature and readiness frameworks combined with feedback from a limited survey of scientists applying ML methods. Its main aspects were shaped by a multidisciplinary expert group representing data providers, infrastructure developers, data managers, and AI practitioners. It identifies 25 criteria covering data preparation, data quality, documentation, access, and supervised learning targets. The criteria emphasize aspects particularly relevant to AI applications, including provenance, treatment of missing data, bias documentation, uncertainty characterization, machine-actionable metadata, efficient access formats, and the quality and reproducibility of target generation. Application to a broad range of representative environmental use cases highlights how the relevance of individual criteria varies by application. We argue that AI-readiness should be understood primarily as a property of transparency and fitness for purpose rather than as a fixed level of data quality. The framework lays a foundation for communication and community-driven action to advance AI-ready data and support next-generation AI-enabled scientific workflows. 

---
# Can Reasoning and Image Description Enhance Yes/No Visual Mathematical Understanding? 

**Authors**: Guangcheng Song, Weichuan Wang, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.27302)  

**Abstract**: Large Vision-Language Models (VLMs) have achieved strong performance on general visual tasks, yet their ability to understand mathematical images remains limited. This paper investigates whether explicit reasoning and textual image descriptions can enhance VLMs' performance on binary yes/no visual mathematical questions. To achieve this, we curated a paired evaluation set to benchmark prominent open-source VLMs under varying prompt and input conditions. In particular, we leverage Chain-of-Thought (CoT) prompting and use textual descriptions, either replacing or augmenting visual inputs, to isolate and evaluate mathematical comprehension. Our results show that explicit reasoning improves performance, especially in paired consistency accuracy, while textual image descriptions do not consistently help and may degrade performance. These findings suggest that reasoning is important for stable visual mathematical judgement, whereas description-based intermediates require more careful design. 

---
# EgoLAM: Learning Latent Actions from Hand and Head Motion 

**Authors**: Ziyi Ding, Yiwei Wang, Zhongxing Wei, Xianxin Lai, Yechi Liu, Xiaodi Huang, Jinghua Hou, Xiaoquan Sun, Haijier Chen, Zhe Liu, Yinan Chen, Junqiao Zhao, Xiao-Ping Zhang, Jiayu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.27290)  

**Abstract**: Latent action models let a robot policy learn from video by turning the change between two frames into a training target. In egocentric video, a cup can move across the image because a hand pushes it or because the wearer turns. Both changes help predict the next frame, but only the first describes manipulation. During training, EgoLAM supervises the action code with measured wrist motion and supplies camera motion separately to the forward model. Even when the wearer turns while pushing a cup, the training target still describes how the hand pushed it. The action encoder pools object slots around each hand, and wrist translation and rotation can be decoded from the code in a common reference frame. Extracting a code then requires only two RGB frames. Alongside LARYBench classification and regression, we measure how much wrist motion a frozen code retains and how strongly it responds to a synthetic view or lighting change. EgoLAM has the highest hand-motion decoding score among eight evaluated models. Across three LIBERO suites, policies pretrained on its codes and fine-tuned on expert demonstrations achieve 79.38% mean success, 28.73 points above expert behavior cloning and 9.76 points above the strongest other latent action model in each suite on average. 

---
# Preserving Pretrained CFG Structure in RL for Diffusion Models with Conditional Contrast Regularization 

**Authors**: Yunfei Zheng, Xiaolong Wang, Dandan Zheng, Xianjun Deng, Shenghao Liu, Wei Xiang, Yuqian Li, Laurence Tianruo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27288)  

**Abstract**: Reinforcement Learning (RL) has become an important component in the post-training of diffusion models. In practice, however, standard RL often improves the target reward without reliably preserving holistic generation quality, leading to unstable reward-quality trade-offs that we view as a form of reward hacking. Since Classifier-Free Guidance (CFG) is known to improve visual clarity, prompt alignment, and perceptual quality in generation, we investigate whether its underlying structure can serve as a useful prior during RL. We present a unified geometric view that interprets both CFG and RL as correction vectors in a common prediction space, and identify the projection of the RL-induced correction onto the pretrained CFG correction as a useful diagnostic of quality-related structural preservation. Geometric analysis and rotation intervention further show that the pretrained CFG direction carries information useful for generation quality that reward-driven RL updates do not necessarily preserve. Based on this insight, we propose Conditional Contrast Regularization (CCR), which uses the frozen reference model's conditional contrast as a local structural prior during RL. Experiments across multiple reward settings and backbones show that CCR more consistently preserves generation quality while maintaining competitive reward, yielding a better reward-quality trade-off than standard RL and generic regularization methods. 

---
# EAGLE: Entity-Aware Grounding with Layered Evidence for High-Resolution Image Understanding 

**Authors**: Zhongjian Zou, Quan Qian  

**Link**: [PDF](https://arxiv.org/pdf/2611.27264)  

**Abstract**: Multimodal large language models (MLLMs) still struggle with fine-grained understanding of high-resolution images: attention heads vary in their reliability for grounding target entities and their attributes, while localized details are often disconnected from the broader scene context. We propose EAGLE (Entity-Aware Grounding with Layered Evidence for High-Resolution Image Understanding), a plug-in framework that jointly selects grounding signals and organizes hierarchical visual evidence. EAGLE employs a fine-grained dataset to screen visually-grounded attention heads, followed by entity-aware grounding to extract target-specific regions. It then integrates hierarchical evidence construction and question-guided routing to select optimal view combinations, feeding them into the frozen MLLM to enhance fine-grained visual understanding. Experiments with Qwen2.5-VL-7B and InternVL3-8B demonstrate that EAGLE consistently improves performance across both backbones and achieves SOTA results on V*, HRBench-4K, and HRBench-8K compared with existing methods using the same backbone. 

---
# Proactive Defense Benchmark against Deepfake Generation 

**Authors**: Joonhyuk Baek, Wonjune Seo, Jae-yun Kim, Saerom Park, Hoki Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.27253)  

**Abstract**: Despite the proliferation of proactive defenses against deepfakes, the lack of a unified evaluation protocol precludes fair comparison and masks critical vulnerabilities. To bridge this gap, we present the first comprehensive benchmark that systematically assesses disruption, robustness, and transferability encompassing pixel, perceptual, and identity metrics. Our extensive analysis reveals that fidelity and identity metrics capture orthogonal performance axes, often leading to conflicting interpretations when relied upon individually. Furthermore, we identify a fundamental trade-off where peak white-box performance signals overfitting, and we introduce a calibrated evaluation to correct generator-induced identity bias. By exposing these blind spots, we establish a rigorous standard to guide the development of genuinely generalizable protections. Project page is available at: this https URL 

---
# CreditFilter: Calibrated Token Weights over the Within-Response Entropy Rank for RLVR 

**Authors**: Hanqing Zhao, Liuyue Xie, Rong-Cheng Tu, Xu Zhang, Davide Modolo, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2611.27227)  

**Abstract**: Critic-free reinforcement learning with verifiable rewards (RLVR) typically assigns a uniform advantage to every token. Token-reweighting heuristics like the 80/20 mask favor high-entropy tokens, but fail to measure a token's actual impact on the outcome. We formalize this impact as the forking value: the variance of the success probability over the policy's next-token choices. The expected forking values along a response sum exactly to the reward variance. Token credit is thus, in expectation over a prompt's responses, a fixed budget spent at the forks. This motivates weights that redistribute credit within a response. The forking value factorizes into a fork consequence, which requires costly continuations to estimate, and a fork probability, which the softmax provides. On Qwen3-8B-Base, this fork probability decays exponentially with a token's within-response entropy rank at class-specific rates, revealing that most tokens carry negligible forking value. CreditFilter weights each token by an exponential window over the same rank. The window is calibrated to this fork-probability envelope, under an approximately flat fork consequence. CreditFilter needs no extra forward passes and is stabilized by difficulty damping and an entropy-gated fallback. Averaged over three seeds, it outperforms DAPO by 7.7 points on MATH500 and by 10.5 and 9.1 points on AIME24 and AIME25. Its dynamically adaptive variant matches it. The ablations show that both parts matter: ranking tokens within each response rather than across the batch, and matching the window's decay rate to the envelope. Code is available at this https URL. 

---
# NavTTT: Adaptive Parametric Memory via Hierarchical Test-Time Training for Vision-and-Language Navigation 

**Authors**: Xiaoyu Luo, Chengxing Lin, Jiaxin Huang, Jiale Liu, Jingyu Guo, Sirui Chen, Aolin Li, Shaoli Huang, Cheng-zhong Xu, Hui Kong  

**Link**: [PDF](https://arxiv.org/pdf/2611.27219)  

**Abstract**: Vision-and-language navigation (VLN) demands that agents accumulate and reason over streaming visual observations throughout long navigation episodes. Existing approaches struggle to maintain long-term history, constrained by the finite context capacity of transformer architectures. To solve the problem, fast-weight test-time training (TTT) provides a mechanism for encoding streaming observations into an evolving parameter state. However, existing TTT does not distinguish informative observations from repetitive visual content, allowing redundant tokens to interfere with the retention of information needed for subsequent decisions. We propose Navigation-Conditioned Test-Time Training (NavTTT), which introduces adaptive memory formation and hierarchical memory integration for vision-and-language navigation. NavTTT leverages global navigation context to prioritize informative visual evidence and integrates language instructions with spatial information to construct spatially grounded memory representations. Episode-local nonlinear fast weights store these representations as Adaptive Parametric Memory (APM), preserving history beyond the visual context for later decisions. To reduce cross-layer coupling in memory formation, NavTTT maintains independent fast-weight memories across multiple decoder depths. This decoupled design prevents errors or biases from spreading across different memory levels. Meanwhile, it enables shared historical information to be independently represented at various levels of abstraction. Experiments demonstrate strong navigation performance on VLN-CE benchmarks and effective instruction following on a real robot. 

---
# Correct Precedents Induce Policy Violations in On-Device LLM Agents 

**Authors**: Chong Tian, Yu Wang, Steve Liu, Qirong Ho, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.27216)  

**Abstract**: Local-cloud agent systems pair a cloud model with a small on-device agent that acts on private data, often as the last check before an irreversible action. We identify, for the first time, a failure in which such an agent is led into a violation by its own correct decisions within a session. After it correctly releases an item whose restriction was lifted, it becomes more likely to release the next item that is still restricted. Because the trigger is a correct decision, a history with no faulty step can still end in a violation. We call this failure correct-precedent generalization and find it in 13 open-weight models from nine developers. A single correct precedent can cut the safe rate of the most affected agent roughly in half, and the risk compounds when a refused request is simply sent again in a new session. Interventions on the agent's history indicate that it carries its decision on the released item over to the restricted one instead of checking that item's status. Building on this finding, we propose Paired Decision Distillation (PDD), which distills the decisions of a weakly affected teacher into a LoRA adapter. PDD restores the safe rate to nearly 100% in every distilled model, without extra prompt text or model calls. 

---
# Neural Defend Ranks #1 in Global Audio Deepfake Detection Benchmark - ARCAS on the Podonos Independent Leaderboard 

**Authors**: Sivashankar Selvarajan, Piyush Verma, Sumit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2611.27215)  

**Abstract**: Commercial detectors of AI-generated audio are usually promoted with results on internal test sets that buyers cannot inspect. Independent benchmarks with private labels give a more reliable comparison. We report the results of Neural Defend's ARCAS foundation models on the Podonos audio deepfake detection benchmark, an independent leaderboard that scores commercial detection APIs and open-source detectors on one held-out set of real speech and speech from current text-to-speech and voice-cloning systems, delivered in six file formats, against labels that only Podonos holds. In the release of 29 October 2026, which covers 28 systems from 16 commercial vendors and the leading open models, ARCAS 290M and the smaller ARCAS 90M share first place, ahead of every other commercial system, including Resemble AI, Hiya, Pindrop, Hive, and Reality Defender. Both models make fewer errors than any other system by a statistically clear margin, keep false positives and false negatives in balance at their default threshold, process every submitted file without rejection, and sit on the accuracy--latency frontier of the leaderboard. As for every commercial system added since Podonos retired its API runner, the ARCAS predictions were submitted by Neural Defend and scored by Podonos against its private labels. We describe the benchmark's construction and protocol in full, reproduce the complete leaderboard and its history across five releases, and analyse error balance, statistical separation, speed, and the gap between commercial and open-source detectors. The results agree with the companion evaluation of ARCAS 290M on the complete test sets of ten public benchmarks and establish ARCAS as the most accurate AI-generated audio detector evaluated on this independent benchmark. 

---
# EAF-TTS: Rethinking Codec-Latent Coordinates for Flow TTS 

**Authors**: Yumiao Li, Haoxiang Jie, Xinbei Li, Yaoyuan Yan, Yue Qu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27214)  

**Abstract**: Latent text-to-speech (TTS) systems generate codec latents decoded by a frozen decoder, but good reconstruction does not ensure geometry suitable for Flow generation from isotropic Gaussian noise. We present Embedded Acoustic Flow TTS (EAF-TTS), an acoustic-latent adaptation of ELF with phone- and duration-conditioned generation, shared-network terminal correction, and frozen WaveVAE decoding. EAF-TTS supports a fixed ZCA interface estimated from training data and adding no learned parameters. The codec, conditioning modules, Flow backbone, and 20-call inference are shared. The intervention is the full ZCA coordinate protocol---training and sampling coordinates, induced loss metric, and terminal perturbation---not an inference-only transform. In single-speaker LJSpeech, the 64-D training-latent covariance has condition number 1355.44. On text-disjoint Val399 strict-GT with fixed E100 raw checkpoints across three matched seeds, ZCA improves UTMOS22 Strong---an automatic naturalness proxy, not human MOS---by $+0.405\pm0.027$ over Scalar normalization (mean $\pm$ seed SD of per-seed paired differences). Absolute scores remain below codec-reference levels. Without retraining, ZCA-5 retains $+0.375\pm0.027$ UTMOS over Scalar-20 while reducing generation-path RTF by $3.34\times$. This three-seed analysis shows no statistically detectable WER or exact-match degradation, but does not establish equivalence. ZCA-3 gives $5.03\times$ with a $+0.165$ UTMOS advantage. At matched reduced call counts, ZCA's advantage shrinks, indicating full-step headroom rather than better low-step integration. A secondary text-only selected-checkpoint result is directionally consistent but confounded by checkpoint type. We claim only proxy-naturalness and measured quality--latency effects within this protocol; human listening, blind text, and broader validation remain necessary. 

---
# Explicit Style Extraction for Personalized Long-Form Text Generation 

**Authors**: Jenitha Reddy Patel, Sameeksha Kini, Amulya Rayasam, Anisa Dye, Abhinav Sharma, Sai Karthik Navuluru, Bo Ni, Li Li, Junda Wu, Tong Yu, Nedim Lipka, Xiaolei Huang, Tyler Derr, Yue Zhao, Zhengzhong Tu, Yu Wang, Hongjie Chen, Thien Huu Nguyen, Nesreen K. Ahmed, Franck Dernoncourt, Ryan A. Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2611.27211)  

**Abstract**: Personalized long-form generation asks a language model to write new text the way a specific user would. The standard recipe, retrieval-augmented generation, places a few of the user's past texts in the prompt but leaves the model to decide what to take from them. We introduce StylePrism, a taxonomy of 28 stylistic aspects in seven categories, and SP-Refine, an inference-time method that makes style extraction an explicit step: the model first fills StylePrism with the user's patterns, quoted evidence, and habits to avoid, then drafts from the user's samples and revises the draft twice against this profile, guided by its own style scores. On the three long-form tasks of LongLaMP with three LLMs, SP-Refine outperforms LongLaMP's retrieval-augmented baseline in 25 of 27 ROUGE and METEOR comparisons, by 10.2% on average and by up to 35.8%. Annotating users' own texts with the extracted profiles shows that StylePrism captures concrete, checkable habits, from the order of rhetorical moves to a reviewer's sign-off. Ablations show that the profile helps as revision guidance, not as a replacement for the samples: writing from the profile alone is worse in all 27 comparisons, while a single implicit-inference call is a cheaper alternative that comes close. Oracle variants that let the reviser see the reference, together with reference-free controls, show that larger refinement gains require information unavailable at inference time. 

---
# Parameter-Efficient Adaptation of Qwen3-VL for Handwritten Mathematical Expression Recognition 

**Authors**: Manh Tuan Ha, Xuan Bao Viet Tran, Vo Minh Dat Nguyen, Minh Nhat Vo, Quoc Trung Nguyen, Van Bay Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.27208)  

**Abstract**: Handwritten mathematical expression recognition (HMER) requires joint recovery of symbol identity and two-dimensional syntax, so a locally plausible transcription may still be structurally wrong. We study whether a compact vision-language model can be adapted to generate linearized Symbol Label Graphs (SLGs) under limited computing resources. Starting from a 4-bit Qwen3-VL-4B checkpoint, we apply rank-16 Low-Rank Adaptation to the vision, language, attention, and MLP modules and train with instruction-formatted CROHME 2019 examples. We compare this model with an internally implemented ViT-Transformer baseline, a zero-shot Qwen3-VL configuration, and additional internally evaluated alternatives. On the 1,198 valid CROHME test examples, the CROHME-only model obtains 524 exact matches (43.74%) and a corpus BLEU-4 score of 77.67, compared with 4.92% and 43.26 for the task-specific ViT baseline. An ablation that adds 2,000 manually written, formula-guided samples decreases exact match to 42.32%, exposing a domain-alignment issue rather than an automatic benefit from more data. The principal contribution is therefore an evidence-backed, resource-efficient formulation of HMER as multimodal structural generation, together with a controlled data-composition ablation and an analysis of the remaining gap between local token overlap and globally correct structure. 

---
# LingSante-Pulse: Unified Physiological Pretraining through Random-Feature Prediction and Cross-Modal Alignment 

**Authors**: Linxiao Yang, Wenwei Wang, Tian Zhou, Bingqing Peng, Zihao Zhang, Zuyi Zhu, Jinjie Gu, Liang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.27206)  

**Abstract**: Clinical monitoring and wearable devices capture complementary physiological signals, but their partially overlapping modality coverage makes within-signal information and cross-modal relationships difficult to exploit jointly. We introduce LingSante-Pulse, a unified, general-purpose and flexible physiological representation framework that learns from these heterogeneous recordings. A shared, signal-conditioned encoder processes each physical channel independently and supports different numbers and combinations of signals for multimodal assessment and single-sensor inference. Pairwise cross-modal alignment, structured random-feature prediction (RFP) and masked waveform reconstruction jointly learn physiological relationships and preserve informative within-signal structure. Across 38 general physiological task configurations spanning PPG, ECG, EEG and EMG, the frozen encoder achieves the best dataset-balanced mean rank in each domain. In sleep assessment, multimodal staging on the HMC dataset and EEG-only staging on the Sleep-EDF dataset reach Macro-F1 scores of 0.7784 and 0.7941, respectively, while single-lead ECG apnea screening achieves an AUROC of 0.9183. Controlled experiments show that function prediction and waveform reconstruction provide complementary supervision for retaining temporal and rhythmic information. Representations learned under continually refreshed random supervision generalize to unseen teacher realizations, and correct auxiliary modality correspondence improves respiration-only representations. These results demonstrate that a single encoder can learn reusable physiological features across signal types, prediction targets and sensor configurations from partially overlapping recordings. 

---
# DepthMSA: Combinatorial Training and Budget-Aware Depth Allocation for Multimodal Sentiment Analysis 

**Authors**: Pan Wang, Jie Peng, Zhimin Li, Jingxian Chen, Hui Ji, Huanrui Yang, Pengfei Zhou, Tianlong Chen, Jingtong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27198)  

**Abstract**: State-of-the-art multimodal sentiment analysis (MSA) models rely on large pretrained encoders, yet typically process every modality to its full depth, regardless of its computational cost or predictive contribution. This raises a fundamental question: how should limited computation be allocated across modalities to preserve predictive performance? We reveal a substantial mismatch between the computational cost of modality encoders and the predictive benefit of processing them more deeply, suggesting that different modalities require different encoding depths. However, enabling heterogeneous depths introduces a key challenge: representations extracted at different depths must still be effectively fused. To address this, we propose $\textbf{DepthMSA}$, a budget-aware framework that supports flexible modality depths within a single MSA model. Specifically, we introduce combinatorial anytime training, which supervises the fusion head across all cross-modal depth combinations using a single full-depth encoder pass. At inference, a training-free, cost-aware allocator selects the configuration with the lowest validation-calibrated error under a sample-specific FLOPs budget. On CMU-MOSEI, DepthMSA maintains accuracy within one percentage point of the full-depth model while reducing measured FLOPs by 45\%, or by 82\% with a separately trained skip-enabled variant. When a CLIP visual encoder dominates computation, the allocation shifts toward reducing visual depth, achieving 56\% fewer FLOPs at unchanged accuracy. Compared with ADMN, an adaptive layer-allocation method that trains a controller for each budget, the skip-enabled variant reaches its accuracy with 3.2$\times$ fewer FLOPs and lower latency, without any per-budget training. Our project is available at this https URL. 

---
# A Frugal AI Framework for India: Responsible and Right-sized AI 

**Authors**: Balaraman Ravindran, K. S. Venkataraghavan, Gokul S Krishnan, Balaji Srinivasan, Balkrishna C. Rao  

**Link**: [PDF](https://arxiv.org/pdf/2611.27189)  

**Abstract**: India's AI debate often swings between two insufficient options: replicating the compute-intensive, capital-heavy foundation-model programmes of the United States or China, or remaining a downstream adopter of foreign AI with localised interfaces and safeguards. Neither serves a country that must deliver AI to a very large, diverse population across uneven infrastructure and within constrained budgets. This white paper from the Centre for Responsible AI (CeRAI), IIT Madras, proposes a third path: Responsible Frugal AI, which applies principle-based frugal engineering alongside Responsible AI. Responsible Frugal AI is right-sized AI: it fixes the functional requirement first, asks whether AI is needed at all, and then chooses the smallest sufficient model, meeting the requirement with minimal resources while keeping a verified margin, robustness and responsible governance. The framework applies frugality across four dimensions (data, compute, model and operations) through seven levers spanning the AI lifecycle, each with practices and measures. It calls for public AI rails, vernacular-first design, measured resource use, human oversight for high-stakes decisions, and a cap on total resource consumption so that efficiency gains are not cancelled out by rising usage. We argue that India is resource-imbalanced rather than resource-poor, and that frugal principles offer the most viable route to population-scale deployment, fit naturally with India's digital public infrastructure, and make sovereignty operational by preserving India's capacity to specify, deploy, govern, inspect and adapt its AI systems. A seven-point adoption checklist and eleven policy statements translate the framework into practice, positioning AI in India as a responsible, frugal public-value system that maximises social value per rupee and per joule. 

---
# GEG-SAM: Gaze Exploration-Guided Segment Anything Model for Weakly Supervised Polyp Segmentation 

**Authors**: Ao Wang, Lei Zhang, Lituan Wang, Zhenwei Zhang, Xiangbao Gong, Zhiyi Zhang, Shunyao Wang, Xianglei Yuan, Zizhou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27183)  

**Abstract**: Endoscopist gaze offers an unobtrusively acquired record of lesion-oriented visual exploration during routine clinical inspection, providing relevant supervision for tasks. However, existing methods treat fixations as isolated spatial prompts directly associated with the target, overlooking the fact that natural gaze records a continuous visual search process in which fixations are temporally connected, spatially organized, and coupled with the visual content. Interpreting each fixation independently based only on its local visual content overlooks the temporal evolution and spatial organization of the gaze trajectory, whereas modeling relations among fixations alone cannot reveal the visual semantics of the regions they attend to. To this end, we propose the Gaze Exploration-Guided Segment Anything Model (GEG-SAM), a principled weakly supervised framework designed to explicitly exploit structured gaze relations and their corresponding visual cues to modulate SAM for polyp segmentation. Specifically, GEG-SAM captures structured gaze relations by modeling the temporal dynamics and spatial topology of gaze exploration. It then associates these relational representations with fixation-aligned visual context to derive gaze-conditioned tokens, which are further injected into the frozen SAM encoder to modulate its pretrained image features. In this way, GEG-SAM effectively exploits the rich information embedded in gaze exploration to guide SAM toward lesion-oriented segmentation. Extensive experiments on five public polyp segmentation datasets demonstrate the effectiveness of our approach, improving the mean Dice by 9.30 percentage points over the strongest weakly supervised baseline. 

---
# Felid: A Flexible and Efficient Design for Transformer Fine-Tuning over Encrypted Data 

**Authors**: Linru Zhang, Jun Jie Sim, Xiangning Wang, Jiahao Zhong, Kaiyu Zhou, Huanyi Ye, Yongsen Zheng, Lushan Song, Xiaojian Liang, Yingting Liu, Yujing Sun, Huaxiong Wang, Pu Duan, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2611.27175)  

**Abstract**: Fully Homomorphic Encryption (FHE) enables non-interactive, privacy-preserving transformer fine-tuning, protecting highly sensitive data as inputs to large language models. Prior work has predominantly pursued efficiency gains through architectural refinements, yet overall performance remains bounded by the substantial cost inherent in encrypted computation. To address these issues, we target the bottleneck at the ciphertext operation level. Concretely, we propose a novel packing strategy that supports batching over a flexible number of inputs, making it well-suited for fine-tuning workloads. In addition, we design operation-specific matrix evaluation algorithms that significantly reduce (i) the number of ciphertext rotations and relinearizations, and (ii) the number of bootstrapping invocations. Additionally, we exploit functional bootstrapping to fold the evaluation of element-wise activation functions directly into the bootstrapping step, thereby reducing overall computational complexity. As a result, our framework achieves a $30\times$ reduction in ciphertext rotations and relinearizations across all matrix operations, as well as a $2.5\times$ reduction in FHE bootstrapping invocations over the entire computation flow, compared to the latest state-of-the-art work. It is worth noting that our system completes fine-tuning of a 2-layer BERT-style model with a batch size of 16 in 650 seconds, achieving a $8\times$ speedup over the state-of-the-art work. 

---
# Co-Evolving Behavioral Agents and Detectors for Adaptive Information Operations 

**Authors**: Lin Tian, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27156)  

**Abstract**: Detectors of state-sponsored information operations (IOs) are trained on campaigns that platforms have already taken down. They lose much of their accuracy on an actor they have not seen, even though no operator in the archived data is trying to avoid them. A real operator wants two things, to reach an audience and to avoid being caught. We present IO-AI, a framework that gives simulated agents these two capabilities, engagement and evasion, and tests what each one does to detection. The agents are built from the posting behavior of real campaigns and choose the emotion, type, and timing of each post. An agent and a detector then train against each other in turn. We rank the trained agents by how hard they are to detect and train one detector on them from easiest to hardest. On 20 Twitter IO campaigns, this detector reduces missed detections of adaptive agents by 65% relative to a detector trained only on the archived campaigns, at a similar false-alarm rate. On actors it has never seen it reaches an F1 of 0.81, against 0.62. We find that an agent trained to seek engagement becomes easier to catch, because it learns the fast, emotional retweeting that archived campaigns were caught producing. An agent that was trained for engagement before the game keeps its engagement lead after it learns to evade. We also introduce a behavioral distance metric that compares two groups of accounts using only their unlabeled posting histories. The larger the distance between two groups, the worse a detector trained on one does on the other (Spearman rho = -0.58), so the metric can be used to choose which new groups to test first. 

---
# Structure Over Scale: Rethinking Adaptation for Reinforcement Learning with Verifiable Rewards 

**Authors**: Allan Kazakov, Abdurrahman Javat  

**Link**: [PDF](https://arxiv.org/pdf/2611.27148)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is often framed as a choice between the expressivity of Full Fine-Tuning (FFT) and the efficiency of Low-Rank Adaptation (LoRA). We find that this framing misses an important variable: the geometry of the update. Under a common cold-start GRPO protocol, LoRA exhibits rank collapse: performance remains broadly stable at low rank, then fails abruptly as rank increases. On DeepMath-Hard, for example, accuracy drops from 73.1% at r=64 to 4.7% at r=128. In contrast, Weight-Decomposed Low-Rank Adaptation (DoRA) and Quantum-Informed Tensor Adaptation (QuanTA) remain stable over a wider tested optimization region and are competitive with FFT while updating less than 0.6% of the model parameters. Spectral analysis reveals a corresponding difference in learned update structure: collapsed LoRA concentrates its update energy into relatively few effective directions, whereas structured adapters produce substantially more distributed updates. These results suggest that nominal rank and parameter count alone are insufficient to characterize adaptation in RLVR - the structure imposed on the reachable update manifold also matters. Finally, across three 8B base models, adapter behavior aligns with both frozen-weight spectral structure and reported pre-training scale, motivating the Model Maturity Hypothesis: the adaptation geometry that works best may depend on properties already encoded during pre-training. We present this as a falsifiable hypothesis rather than a causal conclusion. The code can be found here this https URL 

---
# How Does SAM-3 Adapt? Capacity, Update Geometry, and Domain Transfer 

**Authors**: Allan Kazakov, Abdurrahman Javat, Duygu Cakir  

**Link**: [PDF](https://arxiv.org/pdf/2611.27142)  

**Abstract**: Promptable segmentation foundation models are increasingly used in specialized domains, but it remains unclear how SAM-3 should be fine-tuned when adapting it beyond its original training distribution. We study this question by comparing five adaptation families --LoRA, HiRA, KRAdapter, QuanTA, and Full Fine-Tuning-- across five segmentation datasets spanning medical imaging, remote sensing, and dense material segmentation. Across these datasets, compact parameter-efficient fine-tuning methods often match or approach Full Fine-Tuning in DSC and IoU, suggesting that strong in-domain adaptation does not always require updating the full model. However, this performance cannot be explained by nominal adapter rank or trainable parameter count alone. To investigate why different methods perform similarly, we analyze their update spectra and their behavior under LoveDA Urban/Rural transfer. Our SVD analysis shows that Full Fine-Tuning updates are spectrally compressible. However, successful PEFT methods do not simply recover the same update directions; instead, they often concentrate changes in different components and subspaces. The domain-transfer study further shows that methods with similar in-domain performance can respond differently under distribution shift. Together, these results suggest that SAM-3 adaptation is often capacity-saturated in practice, while competitive fine-tuning arises through method-specific update paths rather than a single low-rank approximation to Full Fine-Tuning. 

---
# Do Senior Agents Do Their Job? Measuring Role Integrity in How LLM Agents Use Subordinate Material 

**Authors**: Jinxiang Xie, Xinyi Zhang, Changjiang Jiang, Kexu Liu, Qi Yang, Yuxin Su  

**Link**: [PDF](https://arxiv.org/pdf/2611.27139)  

**Abstract**: Multi-agent LLM systems increasingly divide long tasks among subagents whose reports a senior agent turns into the final answer. Such a division of labor pays off only if each agent, and above all the senior agent, carries out the duties of its role. Current evaluations, however, mostly score the final answer, and a fluent, well-cited answer cannot show whether the senior agent weighed its subagents' reports or merely aggregated them. To assess whether a senior agent meets its assigned responsibilities, we define role integrity as the preservation of an agent's functional responsibilities when it acts on its subordinates' material. To measure it, we introduce RI-Bench, a behavioral benchmark grounded in the structures and public records of three institutional hierarchies: peer review, appellate adjudication, and parliamentary inquiry. We choose these hierarchies because their rules explicitly assign the senior role a duty toward subordinate material, and together they cover three basic duties: to weigh, to verify, and to relay. Rather than scoring output correctness, we measure how senior agents' outputs respond to controlled changes in subordinate material, using rule-based checks and a rubric-guided LLM judge calibrated against human annotation. Across eight LLMs and over 12,000 agentic episodes, senior agents rarely adopt a planted procedural error (2.9% of rulings) but often follow a majority of generic reviews (up to 71% of decisions) or drop witnesses' stated limitations (up to 65% of cited claims). Obligation prompts reduce these failures without removing them, and they also shift behavior in the control conditions. These findings motivate direct, obligation-specific evaluation of role integrity rather than inferring it from general capability or explicit role instructions. Code and data are available at this https URL. 

---
# Schema-Validated LLM Workflows: A Benchmark Protocol and Comparative Evaluation Across Six Python Implementations 

**Authors**: Sidney Shapiro, Sanad Al-Khatib, Aurora Pinzón Arzola, Emiliano Sebastian Gonzalez Venegas, Daniel Pearson  

**Link**: [PDF](https://arxiv.org/pdf/2611.27106)  

**Abstract**: Large language models are increasingly used to extract records, generate SQL, summarize reports, answer policy questions, and route operational work. At these software boundaries, parseable output is necessary but does not establish task correctness. We present a reproducible benchmark protocol, a public executable artifact, and a confirmatory live API evaluation that separate structural validity from task-specific correctness across five business workflow tasks and six Python implementation conditions: PydanticAI, LangGraph, DSPy, CrewAI, Agno, and a framework-free SDK reference.
We hypothesize that (H1) schema validity and task quality are separable, (H2) structured-output binding paths are heterogeneous under a shared model and lowest-common-denominator contract, and (H3) condition-by-task interactions concentrate failures rather than shifting all tasks equally. Confirmatory analysis uses distinct task instances (100 extraction and 50 per other task, with three nested replicates) under a frozen analysis plan, yielding a 5400-row primary matrix with equivalence margins and Holm-adjusted contrasts against the SDK reference.
In the eligible confirmatory run, overall schema validity was 0.684 and mean automated quality was 0.638. PydanticAI, LangGraph, and DSPy retained near-ceiling schema validity with mean quality 0.91-0.94, while the plain SDK, Agno, and CrewAI conditions were substantially lower in this package and model configuration. Schema-valid rows still produced many task-score flags, especially in extraction and compliance QA, and quality residuals show strong condition-by-task interactions (including a workflow collapse for CrewAI). Labelled ablations address retry budgets and feature-using mechanisms; primary contrasts remain system-configuration findings, not causal attributions to marketed library features. 

---
# One state, many questions: Opsis gives a frontier model typed eyes for pathology 

**Authors**: Zhixuan Ge, Wei Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27102)  

**Abstract**: A synoptic pathology report is a checklist of questions with fixed answer sets, many composed from others. Pathology foundation models answer them accurately with one classifier trained with labels for each, and frontier language models, which compose freely, cannot fit a slide of fourteen thousand tiles in their context. On 189 breast report questions, frontier models that read a written description of the slide or look at a thumbnail and tiles of it answer at 0.50 to 0.66, below the majority class at 0.67. A description is written before the question is known, and a look covers a handful of tiles. We present Opsis, which lets the model ask instead: a typed question interface on a frozen pathology encoder. A slide is encoded once, a question names an operator, arguments and candidates given at runtime, and the answer is a probability over the candidates. We build it as a 10 M parameter decoder on frozen Virchow2 tokens (Opsis-S) and inside Qwen3-VL-8B (Opsis-V). Asking Opsis-S, four frontier models answer the breast questions at 0.78 to 0.79, and on 3,797 pan cancer questions asking is 8.6 points above reading. On 995 composite questions that Opsis-S was never trained on, a frontier model composing its answers reaches 0.76, against 0.69 when it looks at the slide and 0.78 for ABMIL trained on each composite. On a pan cancer question set over 10,712 TCGA slides, one Opsis-S decoder matches linear probes trained per question on TITAN and PRISM2 embeddings (average accuracy 0.69 against 0.70 and 0.68). Opsis-S also separates CAMELYON16 metastases without training on that cohort (AUROC 0.86) and localises necrosis from report labels alone (tile AUROC 0.93). 

---
# Epistemic Guard: Out-of-Process MCP Receipt Proxying, Post-Turn Prose Supervision, and Provenance-Gated Memory for Tool-Using Autonomous Agents 

**Authors**: Sean Maxwell  

**Link**: [PDF](https://arxiv.org/pdf/2611.27095)  

**Abstract**: Language-model agents using the Model Context Protocol (MCP) face three runtime verification gaps: (i) action fabrication, where post-tool prose narrates side effects that failed, timed out, or never ran; (ii) provenance collapse, where unverified model prose or web text enters durable memory alongside executed tool outputs; and (iii) syndicated echo amplification, where mirrored web pages satisfy multi-source corroboration from one origin. Existing MCP firewalls inspect wire calls before execution, while LLM-judge guardrails require multi-gigabyte VRAM without binding prose to receipts. We present epistemic-guard (v0.5.2), an open-source Python library and out-of-process stdio MCP receipt proxy (15 modules, 0 MiB GPU VRAM). epistemic-guard isolates Ed25519 signing keys out of process, writes RFC 6962 Merkle logs with AER-1 receipt translation, checks post-turn prose on CPU with SHA-256-verified OpenVINO INT8 encoders (4 models, 279.48 MB total; 82.67 MB on the supervision path), gates SQLite writes across 7 provenance domains, and collapses verbatim and lightly edited wire-syndicated RAG mirrors. On conformance suites, epistemic-guard rejects 100.0% of tampered receipts (200 Merkle leaves), passes 58/58 official AER-1 ingestion vectors and 50/50 reference-verified AER-1 exports, verifies 100.0% of 40 fault-injected wire calls and 100 concurrent appends with 0 key leaks, and blocks 100.0% of 100 unauthorized memory writes. Across $N=250$ frozen-benchmark traces, epistemic-guard reaches 93.2% accuracy at 17.9 ms warm median CPU latency (8.62 ms single-clause; 921.8 ms cold-start load), beating 12B-14B GPU judges by 6.4-6.8 points ($p=0.0259$, $p=0.0095$) at 4.26-4.5$\times$ lower latency; on $N=60$ live mcp-proxy traces, it catches 96.15% of hazards at 76.67% accuracy (38.24% FPR; $+18.34$ points over Stage-1-only, $p=0.001$) in 9.16 ms p50. 

---
# Separating Automation Gains from Agent Contributions in AI-Assisted Model Validation 

**Authors**: M. Ahmadi Achachlouei  

**Link**: [PDF](https://arxiv.org/pdf/2611.27094)  

**Abstract**: LLM agents are being proposed for bank model validation, yet evaluations that compare agent-assisted work with manual practice credit the agent with gains that come from digitized evidence and deterministic tests. We propose a staged architecture in which an LLM-free controlled workflow (P1) is both the fallback and the control arm for individually authorized agent tasks (P2), and an attribution protocol that credits an agent only for faults P1 does not already report. The protocol charges decoy false alarms, unnecessary actions and endorsed developer assertions, counts invalid outputs as misses, folds newly automatable findings back into P1 with discovery credit, and fixes safety bounds and a break-even cost gate in advance. On a synthetic credit-scorecard harness, five scripted reference policies show why this matters: naive case accuracy (4/6) and pooled detection (3/5) tie four flawed policies, whereas the protocol separates all five and credits only the reference policy with detection beyond P1. One fold-back round raises a P1-parroting policy's naive scores without any change to the policy. A regression matrix of sixteen mutants covers nine ways in which earlier versions of the evaluation passed misleadingly. No LLM agent has been evaluated yet; we specify a falsifiable pilot. 

---
# Exercised, Rarely Recovered: What Public Reports Say About Recovery in 50 AI Agent Harnesses and Runtimes 

**Authors**: Kaviru Hapuarachchi  

**Link**: [PDF](https://arxiv.org/pdf/2611.27090)  

**Abstract**: When an AI coding agent deletes the wrong files or a long agent run crashes, users fall back on recovery mechanisms: resume from a checkpoint, undo, ask for approval, read the log. Research runtimes promise stronger guarantees, such as saga-style compensation and transactional rollback. We ask what public evidence shows about whether either kind works in real use. Under a protocol frozen before data collection, we graded 12 persistent agent runtimes, 26 coding-agent harnesses and 12 research runtimes on eight recovery mechanisms, from not found to exercised in a real run, with two blind coding passes, adjudication and re-verification against archived sources. Recovery is exercised almost everywhere in practice: 11 of 12 runtimes and 25 of 26 harnesses have at least one real-use report. Yet of 128 such reports, only 19 describe the run fully recovering; 54 are partial and 45 failed. Checkpoint/resume recovered once in 27 reports, rollback once in 15. Research runtimes show the opposite profile: guarantees evaluated on benchmarks, none reported in operation, and compensation of external effects, their central guarantee, is offered by no harness at all. 

---
# Fooling the Mind Twice: Human Detection of Audiovisual Deepfakes Under Repeated Exposure and Corrective Feedback 

**Authors**: Ammarah Hashmi, Sahibzada Adil Shahzad, Yu Tsao, Hsin-Min Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27080)  

**Abstract**: Human deepfake detection remains unreliable, and whether reassessment with corrective feedback improves subsequent judgments remains unclear. This study examined differences between initial and repeated deepfake assessments among 110 participants using audiovisual stimuli from the FakeAVCeleb and Deepfake Detection Challenge (DFDC) datasets. Participants evaluated 40 videos from each dataset in phase 1 (PhaseI) and reassessed the same videos in phase 2 (PhaseII) under a procedure combining repeated presentation and corrective feedback. For the FakeAVCeleb dataset, the participants' mean accuracy increased by 4.68 percentage points from 63.30 in PhaseI to 67.98 in PhaseII (t(109)=5.61, p<.001, d_z=0.54). A judgment-level logistic regression with standard errors clustered by participant and video produced a consistent result: beta=0.208, SE=0.040, z=5.23, p<.001, {OR}=1.23. For the DFDC dataset, the participants' mean accuracy increased by 8.64 percentage points from 59.52% in PhaseI to 68.16% in PhaseII (t(109)=7.37, p<.001, d_z=0.70); the logistic model yielded: beta=0.375, SE=0.065, z=5.78, p<.001, {OR}=1.46. Improvements primarily came from fewer false negatives, while false-positive rates remained unchanged. However, judgment reversals occurred in 27.55% of FakeAVCeleb and 32.91% of DFDC assessments, with some initially correct judgments becoming incorrect upon reassessment. Despite the accuracy gains, repeated assessment did not eliminate judgment instability or residual error. Participants also exhibited consistent overconfidence, with mean self-rated confidence exceeding actual accuracy by 11.99 percentage points for FakeAVCeleb and 12.82 percentage points for DFDC across both phases. These findings indicate that repetition with corrective feedback moderately improves deepfake detection but does not fully resolve the limitations of human judgment. 

---
# Weight-space programming and mechanistic interpretability for model editing in particle physics 

**Authors**: Raphaël Bonnet-Guerrini, Johann Ioannou-Nikolaides, Troels Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2611.27079)  

**Abstract**: We demonstrate that weight-space programming and edits guided by sparse autoencoders can remove unwanted aspects of a particle-physics foundation model through direct edits to its weights. In a foundation model trained on jets, we observe that adding a single soft neutral particle causes large changes in jet classification, violating infrared safety. We locate the responsible attention heads with mechanistic analysis and trace the flaw's origin to pretraining on datasets with different reconstruction thresholds. We then propose two targeted approaches to repair the flaw: by removing three latent directions identified with sparse autoencoders, and by letting language-model agents edit the weights directly, without gradients and with only 50 queries to the model. Both their edits individually remove almost the entire flaw and leave the rest of the model's response nearly unchanged. Compared to fine-tuning, both methods change far fewer parameters. Although built only on $W$ and QCD jets, the edits carry over to all signal classes, the pretrained generator, and a newly trained regression head. 

---
# PACE: Policy Augmentation and Compensation With Explicit Disturbance Estimation for Robust Locomotion 

**Authors**: Shixuan Han, Yuyang Wang, Jingting Jia, Zeshuai Chen, Jindou Jia, Kexin Guo, Xiang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27061)  

**Abstract**: Reinforcement learning has made quadrupedal locomotion remarkably robust. To handle unmodeled dynamics and external disturbances, domain randomization is usually employed to make a policy hedge against every condition it might meet. However, maintaining accurate tracking across varying dynamics and generalizing beyond the sampled range remain challenging. Here we show that explicitly estimating the disturbance and feeding the resulting estimation back into both the policy and the controller can relax this compromise. We introduce Policy Augmentation and Compensation with Explicit disturbance estimation (PACE). The framework lumps payload variations, center-of-mass (CoM) shifts, actuator degradation, and external forces into one joint-space disturbance. A nonlinear observer subsequently recovers this disturbance online without requiring joint acceleration measurements. The disturbance estimation provides feedforward torque compensation as the foundational control pathway and further augments the policy observation to enable disturbance-aware action generation. Considering the learning difficulty of fast-varying contact effects, we further provide an analytical contact estimation approach from joint and inertial measurements, without demanding force sensors. Across eleven simulated dynamics conditions, PACE reduces velocity-tracking error by 18.1% to 57.0% against state-of-the-art baselines, and maintains improved tracking under out-of-distribution (OOD) commands. Notably, in real-world experiments, the deployed real robot carried an 11.5 kg payload, nearly equal to its own mass. It also negotiated unseen 16 cm step-downs and crossed five outdoor terrains. The policy was trained exclusively on flat ground in simulation with at most 3 kg of added mass. 

---
# Learning 3D MRI Representations with Generative Adversarial Networks 

**Authors**: Sujeet Lahane, Sapna Mishra, Akash Suryavanshi, Tapan Gandhi, Bharat Biswal  

**Link**: [PDF](https://arxiv.org/pdf/2611.27057)  

**Abstract**: Learning representations of visual images is a challenging task. It requires extracting abstract patterns from high-dimensional data without explicit supervision. This becomes pronounced with 3D MR imaging due to sparse features, higher dimensional inputs, and smaller datasets. Many state-of-the-art representation learning algorithms successfully rely on transformers and diffusion-based architectures. However, these methods struggle with smaller datasets. In effect, solution methods relax the problem by slicing 3D volumes and solving the problem for 2D medical imaging, resulting in a loss of spatial 3D information. We present R2RG, a representation learning method based on Convolutional Neural Networks (CNNs), with Generative Adversarial Networks (GANs) honing the reconstruction task. Our key insights lie in exploiting architectural biases in 3D CNNs, combined with data-efficient auto-encoding GANs and a composite loss function to enhance reconstruction fidelity and stabilize training. Our experiments on three standard datasets and a locally collected dataset show that R2RG not only achieves superior performance but also generalizes to heterogeneous and smaller datasets. Furthermore, representations learned with R2RG outperform those learned with contrastive-based representation learning methods on Parkinson's and Alzheimer's disease classification. Our work demonstrates that reconstruction-based representation learning is a promising direction for 3D medical imaging under limited data availability. 

---
# Hypergraph Structural Entropy-Enhanced Hierarchy-Aware Semi-Supervised Graph Classification 

**Authors**: Guangjie Zeng, Hao Peng, Angsheng Li, Li Sun, Chunyang Liu, Yicheng Pan, Shengze Li, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.27043)  

**Abstract**: Semi-supervised graph classification aims to learn graph-level representations by leveraging a large number of unlabeled graphs alongside a small set of labeled ones, effectively mitigating the issue of label scarcity. Despite significant progress by various semi-supervised learning strategies, complex relationships among graph vertices beyond pairwise edges, especially hierarchical structures, remain underexplored by existing methods. In this work, we propose a Hypergraph Structural Entropy-Enhanced hierarchy-aware framework, namely HypSEE, for semi-supervised graph classification. Specifically, we first construct walk-induced and embedding-derived hypergraphs to capture hierarchy-aware local and long-range higher-order relationships within graphs. Then, we devise a hierarchical hypergraph pooling approach guided by hypergraph structural entropy (HSE) to facilitate hierarchical representation learning in hypergraphs. Lastly, we employ an anchor graph-based consistency strategy to mutually align the representations from these two hypergraph views, enhancing the learning efficacy on unlabeled data. Extensive experiments on 8 benchmark datasets demonstrate the state-of-the-art performance of our approach. 

---
# Ruby: State-of-the-Art Potency Prediction across Discovery Regimes Using Learned Structural Representations 

**Authors**: Michael Atkin, Dominique Beaini, Jules Berman, Rebecca Boiarsky, Jamin Chen, Alejandro Dobles, Sergey Edunov, Evan Feinberg, Nate Gruver, Will Harvey, Deidre Johns, Nina Jovic, Steven Kearnes, Allison Keys, Rohit Kundu, Ken Leidal, Gerald Lelais, Michael V. Levine, Indrawan McAlpine, Gabriel Monteiro da Silva, Igor Morozov, Pranav Murugan, Adi Nagle, Frank Pickard, Wesley Qian, Gianluca Scarpellini, Ansh Sharma, Richard Strong Bowen, Wojtek Swiderski, Laura Tapias, Carl Tilbury, Cyr Vetcher, Han Wang, William Wang, David Charles Williams, Drausin Wulsin, Han Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2611.27015)  

**Abstract**: Small-molecule drug discovery proceeds through repeated design-make-test-analyze cycles. The goal is often to maximize potency while keeping other properties within the ranges required for a viable therapeutic candidate. Compound prioritization places different demands on computational methods as drug discovery progresses from hit identification through hit-to-lead and lead optimization, with changing computational budgets and increasing amounts of on-target experimental data. We introduce Ruby, a Robust Universal Binding affinitY predictor that uses intermediate representations from the structural foundation model Pearl without requiring an explicit binding pose for every compound. Ruby is trained on proprietary data plus GenesisPotency-2026, an audited public-source potency training dataset released for academic and commercial use under a CC BY-SA 4.0 license with row-level provenance and audit annotations. Ruby achieves the highest aggregate early enrichment across seven public hit-identification targets among evaluated methods, with adjustable inference effort trading off enrichment and computational cost over more than two orders of magnitude. Ruby achieves the highest average correlations across four public benchmarks: OpenFE, CASP16, FEP+ 4, and OpenBind. Ruby significantly outperforms other methods in zero-shot prediction across eight biochemical and cellular assays from five ongoing Genesis-Incyte programs, widening its performance advantage relative to the public benchmarks. Ruby continues to improve more than other methods when all are given the opportunity to recalibrate using earlier assay measurements. Together, these contributions provide an annotated training resource for the field and a state-of-the-art potency predictor for broad screening, prioritization within related chemical series, and assay-specific prediction as experimental measurements accumulate. 

---
# FoRe: A Force-Guided Framework for Autonomous Grasp Recovery in Vision-Language-Action Models 

**Authors**: Shuai Peng, Yongkai Liao, Yuxin Cheng, Haoyuan Wang, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.27011)  

**Abstract**: Vision-language-action (VLA) models have demonstrated strong performance in grasp-based manipulation, yet reliable recovery from local grasping failures remains challenging. Specifically, in pick-and-place tasks, policies may continue transport after a failed grasp instead of retrying, potentially reflecting overreliance on proprioceptive cues for stage progression and limited retry supervision. Existing recovery approaches typically rely on deployment-time guidance or demonstration augmentation. We introduce Force-guided Recovery (FoRe), a framework for autonomous recovery from local grasping failures. Using force feedback, FoRe partitions demonstrations into approach-grasp and transport-placement stages to train separate policies. To supplement limited retry supervision, FoRe then initializes a dedicated retry policy from the approach-grasp policy and refines it through local reinforcement learning (RL) from post-failure states. To avoid inappropriate stage progression, a controller combines force feedback with grasp-command status at inference to coordinate the stage policies, detect grasp failures, and invoke the retry policy. We also construct LIBERO-SUCTION, a force-augmented suction manipulation benchmark adapted from LIBERO, with complete demonstration trajectories for policy training. FoRe consistently improves task success rates for VLA-Adapter and $\pi_{0.5}$ on LIBERO-SUCTION, LIBERO, and real-world tasks. On LIBERO-SUCTION, average absolute gains are 18.4% and 6.5%, respectively. The corresponding real-world absolute gains are 25.0% and 21.7%. FoRe also remains effective with only one demonstration per task on both simulation benchmarks. 

---
# JevRisk: Zero-Shot Financial Risk Assessment with a Language Decision Model 

**Authors**: Yaodong Yang, Li Huang, Junpeng Liang, Yi Ma, Hongyao Tang, Xingyu Fan, Jinpeng Li, Furui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26999)  

**Abstract**: Banks must monitor every customer's financial risk, such as money laundering and credit default, by interpreting financial activity in the context of the customer's profile. Existing risk monitoring systems of banks are mostly based on expert-designed risk indicators, which apply predefined thresholds to every customer and cannot adapt to their personal situations. Supervised models recognise risk patterns from labelled customers, but each label often needs an expert's review or months of repayment history, which is slow and costly. Financial risk assessment therefore calls for a zero-shot decision system that minimises the need for labels and generalises across customers and risk types. To this end, we present JevRisk, a system for zero-shot financial risk assessment with a language decision model. JevRisk works in two steps. First, following regulatory guidelines in the finance domain, JevRisk computes exact risk indicators with code. Second, the language decision model (e.g., \jev{}) reads these indicators together with the customer's profile and activity, judges the case in context with broad world knowledge, and returns the score of high risk. We evaluate JevRisk on the anti-money laundering task with public synthetic data from Scotiabank and on the credit risk task with real data from Home Credit and Lending Club. On both tasks, JevRisk outperforms rule-based and unsupervised methods, and is on par with or better than supervised methods (e.g., GBDT and TabPFN) trained on about a thousand labelled customers. 

---
# Privacy Protection in Self-Evolving Personal Agents: Persistent Memory and Cross-Carrier Transfer 

**Authors**: Zibin Lin, Taotao Wang, Shengli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26983)  

**Abstract**: Personal agents carry private experience into later decisions through persistent memory. This thematic survey compares memory admission, maintenance, release, and withdrawal by protected change, observer, and validation setting. Direct evidence in the screened corpus comes from controlled memory workflows. Three worked scenarios follow one private fact through booking, feedback-driven learning, and withdrawal. We show when prompt revision and memory reads compose, when skills inherit a private head's bound, and why reader non-use does not establish cleanup of learned state. Diagnostic experiments measure the utility costs of wider label protection and broad memory deletion, and test how missing lineage obscures descendants. 

---
# Contexture: Runtime Context Eviction for High-Throughput Agent Serving 

**Authors**: Colin Hong, Bryan Shan, Wei Siong Pang, Chengzhi Lu, Xu Guo, Nitin Shivaraman, Tao Luo, Esha Choukse, Kisson Lin, Dmitrii Ustiugov  

**Link**: [PDF](https://arxiv.org/pdf/2611.26980)  

**Abstract**: LLM agents such as Hermes, OpenClaw, and Codex repeatedly resend growing execution histories, increasing shared GPU key-value (KV) cache demand in high-bandwidth memory (HBM). We show that shared KV capacity limits agent-serving throughput. Existing context-reduction methods lack global cache-state visibility, while runtime KV managers lack tool-execution information. Tracing tokens to their originating tool operations reveals that later operations supersede earlier outputs and associated turns, identifying 64% of recorded final-history tokens as removal candidates.
We present Contexture, a harness- and model-agnostic architecture for runtime context eviction. Contexture separates the harness's full conversation history from the model's execution context, reducing subsequent requests' KV demand without modifying harness logic or model execution. It combines tool-execution information with global visibility into shared KV pressure and prefix residency: supersession identifies removal candidates, while runtime state determines when removal justifies sacrificing cached computation. This coordination reduces memory demand and decode KV traffic while limiting re-prefill interference with concurrent decoding, without additional model calls or training. Across two models, two harnesses, and two benchmarks, Contexture improves throughput under SLO by 20-73% over the best of no eviction, harness compaction, AgentDiet, and Continuum, and by 7-63% over observation masking in five of eight configurations, and reduces GPU energy per completed session by up to 11%. It also raises throughput under SLO by 23% over LMCache alone on H100. Live-agent experiments show a mean accuracy improvement of 1.4 percentage points over unloaded no eviction, with a worst-case loss of 3.7 points, while observation masking loses 3-28 points. 

---
# When Perception Collides with Priors: Diagnosing Modal Polarization in Video Understanding Models 

**Authors**: Yao Tong, Jiaxin Qi, Yan Cui, Jinli Ou, Yuhua Zheng, Jianqiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26972)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved strong video comprehension, largely benefiting from extensive training on realistic video data. However, when video perception severely conflicts with internal priors or common sense, we identify a critical failure pattern termed Modal Polarization, where models fail to integrate modalities and instead bifurcate into two distinct extremes: Cognitive Inertia, where models rigidly adhere to priors despite contradictory visual evidence, and Perceptual Inertia, where they blindly trust video perception while ignoring logical inconsistencies. To investigate this, we introduce TheaterBench, a benchmark designed to expose Modal Polarization using counter-intuitive scenarios that violate physical laws or common sense - situations rarely encountered in standard training distributions. Extensive experiments reveal that current state-of-the-art (SOTA) models exhibit severe Modal Polarization, and many de-hallucination methods merely rebalance the trade-off between Perceptual and Cognitive Inertia rather than fundamentally resolving the conflict. This work serves as a foundational step for evaluating and enhancing the perceptual-cognitive equilibrium of future video understanding models. 

---
# Beyond Prompt Heuristics: Assessing Child Safety Alignment in LLMs 

**Authors**: Anika Misra, Chimaobi Okite, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2611.26965)  

**Abstract**: Child-safety taxonomies typically characterize safety risks through a predefined set of harm categories, and existing evaluations largely operationalize these categories using prompts that explicitly invoke them. However, child-safety considerations can also arise in seemingly benign requests that lack recognizable harm-category cues. We therefore investigate how child-safety alignment generalizes to such settings. Across six models, we find considerably higher misalignment rates on requests without recognizable harm-category cues than on those containing such cues (80.0, 54.0, and 48.4 percentage-point gaps for Qwen3.5-27B, GPT-5.4-mini, and Claude-Haiku-4.5, respectively). Through controlled behavioral and mechanistic analyses, we find that introducing a single harm-associated word reduces misalignment across most models, with mechanistic evidence suggesting that child-safety reasoning may rely in part on lexical and semantic associations with words in the prompt. Our findings motivate evaluations that treat child safety as context-dependent, and not reducible to a fixed set of harm categories. 

---
# Tracking and Limiting Unsupported Claim Propagation in Multi-Agent LLM Systems: A Provenance-Aware Framework 

**Authors**: Zijia Cao, Hanqing Dou  

**Link**: [PDF](https://arxiv.org/pdf/2611.26957)  

**Abstract**: This research examines how unsupported claims change as they move through multi-agent pipelines. It also examines whether preserving provenance can reduce these changes. We test both homogeneous and heterogeneous pipelines. Homogeneous pipelines use the same model across agents, while heterogeneous pipelines use different models. We compare the baseline with a provenance-aware framework. The framework passes each claim with its support status, source, evidence and transmission history. No new evidence is introduced during transmission. We measure changes in claim meaning, unsupported detail accretion, semantic drift, and provenance preservation. These metrics show whether the framework limits unsupported changes as claims pass between agents. 

---
# CENTRY: A Probabilistic Latent-State Model for Continual AI Risk Estimation, Monitoring, and Regulation 

**Authors**: Bonny Banerjee, Geetha Raju  

**Link**: [PDF](https://arxiv.org/pdf/2611.26940)  

**Abstract**: Artificial Intelligence (AI) systems are increasingly deployed in safety-critical applications, yet existing AI safety frameworks primarily define, categorize, or govern AI risk rather than proposing implementable models and methods for continually estimating, monitoring, and regulating risk throughout an AI system's operational lifetime. This paper proposes CENTRY, a probabilistic framework that enables continual risk estimation, monitoring, and regulation of a deployed AI system from its interactions with its environment. The framework models AI risk as arising from three intrinsic sources: datasets, hardware resources, and models/algorithms. A structured latent-variable model is developed in which the latent representation is decomposed into a fast interaction subspace that captures the evolving AI--environment dynamics and three slow latent subspaces that encode the long-term risk states associated with the dataset, hardware, and model components. Rather than requiring a ground-truth risk label at every time step, the component-risk states are treated as latent variables and are learned by maximizing a variational evidence lower bound, optionally informed during training and continual refinement by evidence-supported behavior--cause records. The resulting framework unifies continual probabilistic inference, online risk estimation, and adaptive risk regulation in an implementable architecture. Under the explicit assumptions stated for the corresponding results, we establish formal guarantees spanning safety, learning, inference, and operations, including finite-horizon and asymptotic risk-budget compliance, regret and convergence bounds for adaptive regulation, consistency and identifiability of component-risk ratings, and detection-delay, false-alarm, containment, and computational-complexity characterizations. 

---
# MG-EDA: Maturity-Gated Skew-Normal Estimation of Distribution Algorithms 

**Authors**: Jun Zhang, Tong Qian, Hua Ding, Weijie Yu, Zhi-Hui Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2611.26938)  

**Abstract**: Gaussian estimation of distribution algorithms shrink their variance on slopes and stall far from the optimum. This paper equips the corrections that act along the direction of movement -- a mean shift, an elongation of the covariance along the path and a skewness -- with the rule they lack: a calibrated gate, driven by a maturity statistic of the population, decides whether, how strongly and for how long each correction acts, and when the gate is closed the algorithm is exactly the estimation of multivariate normal algorithm (EMNA). The statistic is a cumulative path of whitened mean shifts with an inverse-Wishart correction, whose expectation is one under a null model without systematic movement and whose quantiles give thresholds with known false-alarm rate. It gates the three corrections as three rank-one terms of one matrix sampling equation, the skewness being carried by a skew-normal sampler that preserves the estimated mean and covariance; all vanish at maturity. We call the resulting algorithm the maturity-gated EDA (MG-EDA). A full factorial study on a training set of black-box optimization benchmarking (BBOB) functions separates the parameters: variance scaling is the prerequisite, the mean shift needs a feedback correction removing the injected shift from the statistic, the elongation turns the variance loss on a slope into a gain, and the skewness substitutes for the mean shift rather than complementing it. On held-out instances MG-EDA with the first- and second-order corrections is significantly better than the skewness-only predecessor and than EMNA with variance scaling, ties with EMNA with anticipated mean shift, and trails the adapted maximum-likelihood Gaussian model (AMaLGaM) in 20 dimensions. 

---
# SecretAgent: A Multi-Agent Framework for Privacy Protection against Vision Language Models 

**Authors**: Minda Zhao, Yue Zhang, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26936)  

**Abstract**: Vision-language models (VLMs) can infer private information from shared images, including locations, personally identifiable information (PII), and personal attributes such as age and sex, even from medical images. Adversarial perturbations can protect such information against black-box VLMs, but existing methods overlook image quality and introduce visible artifacts into the images they protect. We present SecretAgent, the first general multi-agent framework that protects image privacy against VLMs while preserving visual fidelity. LLM agents iteratively locate the visual evidence that reveals private information and adapt an invisible perturbation to the target VLM's answers and to an appearance reviewer agent, which rejects any candidate with visible artifacts. We instantiate SecretAgent with two complementary variants: SecretAgent-Gradient optimizes the perturbation through six surrogate encoders for cross-model transferability, and the gradient-free SecretAgent-Tool composes procedural noise and achieves the best perceptual fidelity on three of the five datasets. Experiments on four privacy applications (geolocation, PII, facial, and medical) across five datasets and six VLMs show that SecretAgent preserves image quality substantially better than existing methods while achieving state-of-the-art privacy protection in every application, and that its protection transfers to other VLMs. 

---
# Tokenize Moments, Not Frames: Tokenizer-Aware Budget Allocation for Training-Free Video LLM Acceleration 

**Authors**: Zirui Zhao, Kai Liu, Linghe Kong  

**Link**: [PDF](https://arxiv.org/pdf/2611.26933)  

**Abstract**: Video large language models (LLMs) represent each video with thousands of visual tokens, and training-free accelerators that prune or merge these tokens lose accuracy on long videos. Analyzing frame similarity and the vision tokenizers of 16 open models, we trace this loss to a step before token selection. Under a token budget, the frames of a long video are sampled farther apart than the roughly 2 s over which frames remain similar. The two-frame 3D patches of models such as Qwen-VL then blend two unrelated scenes into each visual token, a double exposure that arises before any pruning or merging. Fixed-size tokenizers such as InternVL's avoid this blend but respond differently to the same reduction. We propose ESSENCE, a training-free method that tokenizes moments, segments within which frames remain similar, rather than frames. Before encoding, a closed-form rule uses the sampling interval and the tokenizer class to set the number of token grids and the dimension to reduce; visual tokens are then released from later LLM layers. Against five recent accelerators on Qwen3-VL, Qwen2.5-VL and InternVL3.5, ESSENCE achieves higher accuracy in 58 of 60 comparisons at matched mean time to first token (TTFT). With 80% of the tokens, it matches the average accuracy of the full Qwen models at $1.6\times$ faster TTFT; on GLM-4.1V, it exceeds the full model by up to 3.4 points. Code is available at: this https URL. 

---
# Partially canonical axes in large language model representations 

**Authors**: Andrea Gregor de Varda, Ferdinando D'Elia, Greta Tuckute, Evelina Fedorenko, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2611.26920)  

**Abstract**: Large language models (LLMs) develop rich internal representations of language, but whether their individual coordinate axes carry meaning, or whether meaning is encoded only in directions in representation space, is still an open question. A common assumption is that individual axes in LLMs are arbitrary. We here challenge this view: across 12 LLMs from six model families, the native axes are reliably more aligned with other models' representations, with human brain responses to language, and with interpretable linguistic features than randomly rotated axes that preserve the same geometry. The native axes nonetheless fall well short of the best achievable alignment, so LLM coordinates are not fully arbitrary but also not fully canonical: they are partially canonical. A theory of how LLMs represent language must explain both the population geometry they learn and why individual units acquire patterns of response profiles that recur across systems. 

---
# ERA: Embed, Route, Adapt for Cold-Start Cross-Model Latent Retrieval 

**Authors**: Venkata Chivukula, Zifeng Wang, Jinsung Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2611.26919)  

**Abstract**: Production retrieval systems are tied to the embedding model that indexed their corpus. Moving queries to a different encoder normally requires re-embedding the whole corpus, and at cold start there are no query logs while every document embedded by the new encoder incurs an API or compute cost. We present ERA (Embed, Route, Adapt), a system for cold-start cross-model latent retrieval. ERA spends a small anchor budget on maximally informative documents, selected from the existing target index via Cosine Farthest Point Sampling, trains a lightweight adapter with a Manifold Alignment Loss that couples directional alignment with orthogonal regularization to prevent representation collapse, and pairs the adapter with a deterministic router that escalates the rare, hard queries that define perceived retrieval quality. Across five BEIR datasets, ERA matches or outperforms point-wise L2 and embedding-conversion baselines, with relative gains of up to 35% nDCG@10 on Arguana, and on clustered corpora its anchor selection also improves prior methods under a fixed anchor budget. With 1.2 ms of adapter and router overhead on top of a ~20 ms local encoder, ERA at a 20% nominal escalation budget reduces end-to-end query latency from roughly 160 ms to 38-86 ms while concentrating escalations on the queries where the projection fails most. 

---
# AI-Assisted Decision Support for Formula 1 Pit-Stop Strategy: Deep Reinforcement Learning and Mathematical Programming 

**Authors**: Felipe Méndez, Charles Thraves  

**Link**: [PDF](https://arxiv.org/pdf/2611.26909)  

**Abstract**: Pit-stop strategy in Formula 1 is a sequential decision problem under uncertainty, in which strategists decide at each lap whether to pit and which tire compound to fit. We develop a data-driven simulator, calibrated with historical race data, that captures tire degradation, drivers' recent performance, overtaking, yellow flag events, and rivals' pit-stop decisions. We train agents with five Deep Reinforcement Learning (DRL) algorithms and compare them with a Quadratic Mixed-Integer Programming (QMIP) formulation that is optimal in deterministic single-driver settings. Without uncertainty or competition, all DRL methods except TRPO achieve near-optimal performance. With a full grid of competitors and stochastic race events, the ranking of DRL methods depends on the evaluation setting: Proximal Policy Optimization (PPO) and its recurrent variant are the only methods that improve on the QMIP both in fully simulated races and when the actual 2024 races are replayed. Averaged over eight circuits, PPO improves on the QMIP by 0.33 and 0.49 positions in the two settings and Recurrent PPO by less, and in the replay PPO finishes 0.66 positions ahead of the drivers' actual strategies simulated in the same races. The QMIP performs similarly to the other methods, even though it cannot anticipate yellow flag events. Finally, we use the trained PPO policy as a look-ahead decision-support tool that shows the strategist the distribution of race outcomes under each candidate pit-stop action. Applied retrospectively to eight 2024 races, the look-ahead identifies the laps where the actual decision, the recommendation, or a third action was preferable. 

---
# Perceive Once, Escalate on Demand: Handing Off Visual KV Memory from Small to Large VLMs 

**Authors**: Qiang Zhou, Jianing Deng, Siyan Dong, Jingxian Chen, Han-Yi Lin, Jingtong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26902)  

**Abstract**: An embodied agent cannot afford a large vision-language model (VLM) at every step of a long video, so a cascade runs a small VLM continuously and escalates only uncertain decisions to a large one. Each escalation, however, forces the large model to re-encode a history the small model has already perceived, and this dominates its cost. We present Baton, which hands the large model the small model's visual memory, its key-value (KV) cache over the history. Baton rests on one finding: the large model needs a usable memory, not a faithful copy of its own. A map trained to reproduce the large model's cache matches it most closely, yet brings no gain over the small model alone. Baton instead distills a per-layer linear map, without labels, from the large model's own answers, with the large model encoding only the current frame. Interventions on the map trace the gain to one direction, carried by the middle third of the large model's layers. On RoboMME, Baton beats the small model alone, a pruned or shortened re-watch, and a text memory; as the escalation channel of a cascade it matches re-watching to within about one point at 28-47% less compute, and 8x less per escalation at 64 frames. Its gain grows with the large model, from 8B to 32B, tracks the large model's headroom across eight model pairs from four families, and transfers zero-shot to OpenEQA. 

---
# Modeling Attributional Style at Scale: A Dataset and Analysis for Psychological Attribution Assessment and Reframing 

**Authors**: Qiang Zhou, Hanzhen Zhu, Pan Wang, Rui Tu, Huaizhi Qu, Zhuoran Wang, Xin Hu, Lei Li, Tianlong Chen, Jingtong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26900)  

**Abstract**: According to the reformulated Learned Helplessness theory, repeated exposure to uncontrollable negative events can foster a depressogenic attributional style, increasing susceptibility to depression yet remaining a tractable target for cognitive therapy. Computational research on attributional cognition, however, is hampered by the lack of large-scale datasets and robust evaluation protocols. In this work, we introduce the Attributional Style Transfer Dataset (ASTD) along with dedicated evaluation metrics, the first benchmark designed to model, assess, and reframe attributional explanations at scale. Constructed via a Prevent-Filter-Validate pipeline that integrates LLM-based generation with specialist validation, ASTD contains 42,000 real-world events paired with psychologically grounded attributions spanning seven styles. Using this dataset, we address two key challenges: (1) scalable assessment of attributional style via both supervised classifiers and zero/few-shot LLMs; and (2) attributional reframing and evaluation, where we propose automatic evaluation metrics to quantify psychological validity. Furthermore, we leverage our proposed metrics to construct a preference dataset, fine-tuning LLMs with Direct Preference Optimization (DPO) and achieving substantial gains in reframing quality. Together, our dataset, metrics, and methodology offer a new paradigm for understanding and modeling attributional style, with direct implications for scalable and adaptive mental health interventions. Code and data: this https URL 

---
# ProbeAgent: Principled Diagnosis of Agent Failures in Verifiable Environments 

**Authors**: Yue Huang, Yuchen Ma, Shuhang Lin, Yuyang Wu, Zixuan Weng, Zixiang Xu, Xiangqi Wang, Yuexing Hao, Yujun Zhou, Yue Zhao, Olexandr Isayev, Stefan Feuerriegel, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26897)  

**Abstract**: Improving agents requires discovering where they fail and understanding why. An agent acts over multiple steps in a changing environment, so diagnosing its failures requires more than inspecting the final response of a large language model (LLM). The same unsuccessful outcome can arise from task difficulty or from an error the agent repeats across tasks. We introduce ProbeAgent, an automated framework that discovers agent failures, tests their proposed causes, and identifies shared failure modes. Starting from benchmark tasks, ProbeAgent builds executable tests and validates them using reference solutions, automated checks, and review of the task requirements. For each failed trajectory, it explains the error and constructs new tasks that preserve the suspected cause while changing the surrounding context. Running the agent on these tasks tests whether the same error recurs. ProbeAgent then groups failures with similar causes and generates descriptions that distinguish their errors. This process connects explanations of individual failures to executable tests and descriptions of shared failure modes. We evaluate ProbeAgent across 22 agent configurations using tasks from 10 benchmarks, collecting 1,244 failures and producing 536 failure-mode descriptions. These descriptions distinguish specific errors within broad outcomes such as incomplete work, including planning without acting and stopping before delivering a file. In human evaluation, annotators accept 90.0% of assigned descriptions and correctly reject 83.3% of similar alternatives, supporting the accuracy and specificity of the discovered failure modes. 

---
# Your Judge Inherits Your Prompt: In-Context Label Bias in an LLM-as-Judge Is Directional, Not Symmetric 

**Authors**: Abdullah Ahmad, Ahmed J. O. Ahmed, Luca Citi  

**Link**: [PDF](https://arxiv.org/pdf/2611.26881)  

**Abstract**: LLM-as-judge has become a standard way to score open-ended model outputs, yet the judge is itself a frozen language model steered by a few in-context examples, and such models are known to absorb the label balance of their demonstrations rather than treat them as neutral illustrations. We ask whether this label bias reaches an LLM judge used inside a multimodal (video question answering) evaluation pipeline, and what form it takes. Using a fully frozen generator-judge pipeline, we hold the generated answers fixed and vary only the Yes/No balance of the judge's few-shot examples across four demonstration sets. Each verdict is validated against a two-annotator human consensus on 200 examples per dataset. The judge's acceptance rate stays close to human consensus when its examples are balanced or skewed toward rejection, but jumps sharply (by about 30 points on NExT-QA and 20 on MSVD-QA, with a strongly one-sided error profile) when the examples are skewed toward acceptance; the opposite skew produces no matching drop. The bias is therefore one-directional within the tested range of label ratios. A separate five-draw analysis shows that results from a single example draw are unreliable, varying by up to 18 points, more than any difference between prompting conditions. We conclude that demonstration label bias in an LLM judge is a directional, inflation-only failure mode that can fabricate apparent benchmark gains but not losses, and that it is cheaply avoided by balancing the judge's examples and reporting results averaged over several draws. 

---
# Coherent Moment Propagation: Epistemic Uncertainty for Pre-trained Robot Policies 

**Authors**: Miguel Saavedra-Ruiz, Anqi Li, Claudia D'Arpino, Alperen Degirmenci, Tucker Hermans, Fabio Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2611.26855)  

**Abstract**: Flow matching policies underpin the state-of-the-art in robot learning, with impressive results ranging from long-horizon planning to dexterous manipulation. However, they provide no measure of whether their predicted commands can be trusted, particularly in novel situations outside their training set. Epistemic uncertainty reflects uncertainty about the model's weights and quantifies this ignorance, but calculating it at inference time from a flow matching policy is not trivial. Actions are produced by an ODE solver, requiring uncertainty to be propagated through the entire integration process. BayesDiff addresses this problem for diffusion models by propagating the mean and variance of the state through the denoising chain under a last-layer Laplace posterior approximation, but it severely underestimates the epistemic uncertainty. In this paper we: i) prove that, for flow matching and diffusion ODEs, BayesDiff's total variance decays at rate $O(1/T)$ as the number of Euler steps $T$ used by the solver increases; ii) introduce Coherent Moment Propagation (CoMP), two recursions inspired by the extended and unscented Kalman filters that fix this problem by carrying the cross-covariance between state and weights; and iii) widen the last-layer posterior with observation embeddings the policy already computes, improving uncertainty estimates without retraining for CoMP and several test-time estimators alike. We demonstrate these benefits on three manipulation benchmarks with applications in failure detection and demonstration selection for fine-tuning, where CoMP corrects BayesDiff's variance collapse while matching or outperforming several uncertainty estimators. 

---
# LoCo-WAM: Test-Time Training as Long Context for World-Action Models 

**Authors**: Sheng Cheng, Deqian Kong  

**Link**: [PDF](https://arxiv.org/pdf/2611.26853)  

**Abstract**: World-action models guide robot actions by predicting what happens next, but typically condition on a short observation history. Extending this history increases computation at every decision and can require a separately trained model. We introduce LoCo-WAM, which learns from the outcomes of its actions through test-time training while keeping the same short history length. It uses newly observed frames to update a low-rank adapter in the video expert, whose adapted predictions guide subsequent actions. The updates accumulate within each episode and are reset between episodes, allowing past observations to influence decisions after they leave the input window. LoCo-WAM thus learns from a longer stream of experience without expert action labels, additional data or meta-training. Experiments on two world-action models show improved task success across four simulation benchmarks. On RoboCasa GR-1, a model with 2.4,s of history matches one trained with 9.6,s, while a model with 4.8,s exceeds it by 4.8 percentage points. Our analysis shows that the updates improve prediction of upcoming frames, affect the action mainly through adaptation of the imagined future, and have an increasing effect on actions as the episode unfolds. 

---
# Scoliosis5K: Multimodal Gait Analysis via Clip-Conditioned RGB-Silhouette Fusion 

**Authors**: Zizhao Peng, Zihan Wang, Zheng Lv, Yan Wang, Ping Li, Fengwei An  

**Link**: [PDF](https://arxiv.org/pdf/2611.26849)  

**Abstract**: Color walking video records body boundary motion alongside clothing folds, surface appearance, and recording-environment cues. To examine what an explicit binary silhouette contributes to participant-level classification of scoliosis-related gait when a classifier already observes the paired color sequence, we construct Scoliosis5K, a multi-site collection of 4,998 corridor-walking sequences across 3,010 participant identifiers in which RGB person crops, binary silhouettes, and two-dimensional poses share each recording identifier and frame index. Our architecture uses each patch's pixel deviation from its clip mean to weight two low-rank query and value updates inside a pretrained RGB encoder, fusing the resulting video representation with a silhouette transformer sequence and regional kernel ridge scores extracted from one native-resolution RGB frame per window. Across 988 participants in five participant-disjoint cross-validation folds and three training seeds, the complete system reaches 83.77% accuracy and 79.91% macro-averaged F1, compared with 83.27% accuracy and 79.56% macro-F1 when the video RGB encoder remains frozen under identical inputs. Zeroing the silhouette vector at fixed weights shifts overall accuracy by only 0.10 percentage points while raising positive-category recall from 68.52% to 77.78% and lowering recall for both remaining categories. Macro-averaged AUROC is 0.950 across the cohort and 0.879 among participant pairs matched on acquisition site and recording-count group. 

---
# Non-Expanding Gated Continued Fraction Architecture for Feedforward Layers in Language Models 

**Authors**: Amit Dhurandhar, Vijil Chenthamarakshan, Karthikeyan Natesan Ramamurthy, Dennis Wei, Rahul Nair  

**Link**: [PDF](https://arxiv.org/pdf/2611.26835)  

**Abstract**: Recent work has shown that continued fraction inspired neural architectures (CoFrGeNets) can be highly parameter efficient yet performant for language modeling. In this work, we revisit continued fraction-based channel mixing with the goal of improving its performance and simplifying integration into standard pre-training protocols for language models. We achieve this by two simple yet critical contributions: i) we propose a novel non-expanding gated continued fraction inspired architectural component to replace feedforward layers in established language model architectures that may use vanilla multilayer perceptrons or gated linear units or (vanilla/gated) experts in mixture-of-experts (MoE) type architectures. ii) We propose a way to re-implement divisions -- the non-linearity in these architectures -- in terms of logarithms and exponentials. By experimenting on different language modeling architectures such as GPT-2, Llama-3, Mamba-2 and (nano) Llama-MoE we find that our proposed architecture performs similarly or significantly better than the continued fraction architecture proposed in prior work along with having higher throughput. In addition, our re-implementation of divisions makes training of deeper CoFrGeNets stable without the need for incremental training which was proposed in prior work as a means to stabilize training, but added notable complexity to existing large language model training implementations. Code is available at this http URL 

---
# Rethinking Full-Weight LoRA with Domain and Task Adapter Scaling for Question Answering over Portuguese Well Drilling Reports 

**Authors**: Luan B. Guerra, Diogo M. Paes, Marcelo N. Grimberg, Arnaldo C. Junior, Wallace Casaca, Lucas C. Ribas  

**Link**: [PDF](https://arxiv.org/pdf/2611.26833)  

**Abstract**: Adapting large language models (LLMs) to specialized domains remains challenging, particularly for technical documents such as Portuguese well drilling reports, where annotated data are scarce and domain adaptation may compromise instruction-following capabilities. Although Low-Rank Adaptation (LoRA) is widely used for parameter-efficient adaptation, its scaling after training remains insufficiently explored. This study investigates the impact of LoRA adapter scaling on question answering over Portuguese drilling reports using Qwen2.5-7B and Llama-3.1-8B. Domain-adaptive pretraining (DAPT) is performed either directly on instruction-tuned models or on their base counterparts, with subsequent adapter transfer, while task-specific adapters are trained on WellQuest, an open-source question-answering dataset derived from Portuguese drilling reports. The two adapters are also combined using complementary weights. Scaling factors ranging from 0.1 to 1.0 are evaluated across multiple training checkpoints using BERTScore. Results show that fractional scaling consistently improves DAPT performance over full-weight application, increasing F1 from 68.8\% to 71.9\% for Qwen2.5-7B and from 63.2\% to 68.5\% for Llama-3.1-8B. QA adapters perform best at high but fractional weights, while base-to-instruct transfer yields better results than direct DAPT on instruct-tuned models. Adapter merging provides a modest improvement for Llama-3.1-8B. Overall, these findings demonstrate that post-training adapter scaling can improve domain adaptation without additional data or training. 

---
# Realistic Robustness Evaluation of Tabular Data Driven Models 

**Authors**: Miguel Silva, Eva Maia, João Vitorino, Isabel Praça  

**Link**: [PDF](https://arxiv.org/pdf/2611.26818)  

**Abstract**: The increasing adoption of Machine Learning (ML) models in high-stakes applications has raised critical concerns regarding their robustness against adversarial attacks. While extensive research has focused on adversarial example generation in image and text domains, the study of such examples in tabular data remains comparatively limited, despite its widespread use in real-world systems. This work introduces AURORA, a modular, open-source tool for evaluating the robustness of ML models on tabular data classification. It integrates multiple adversarial perturbation methods and evaluation metrics within a unified framework to systematically assess model behavior under adversarial conditions. Central to AURORA is its evaluation report, which focuses on robustness analysis by quantifying performance degradation, comparing the impact of different perturbation strategies, and identifying model vulnerabilities across features and input regions. The report is complemented by visual artifacts that facilitate the interpretability of robustness patterns. Additionally, AURORA incorporates a distance adjustment mechanism that penalizes implausible adversarial samples based on feature-level constraints, ensuring that the robustness assessment remains grounded in realistic perturbations and is therefore more reliable. 

---
# Search From the Start: Search-Native Pretraining for Language Models 

**Authors**: Kaustubh Ponkshe, Gyung Hyun Je, Stefan Taga, Colin Raffel, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2611.26809)  

**Abstract**: Modern language models (LMs) are increasingly deployed with access to search tools at inference. Yet pretraining remains predominantly focused on predicting static text, with search capabilities typically introduced during post-training. We propose Search-Native Pretraining as a practical route to teach search capability during pretraining from ordinary text corpora. Specifically, we augment pretraining corpus at scale with web-search queries and retrieve web response using a local index. Training on this augmented corpus teaches models to generate search calls and incorporate retrieved evidence during language modeling. Across Wikipedia and DCLM-Edu corpora and 360M to 3.3B parameter scales, search-pretrained models ("QueryLM") show improved performance on knowledge-intensive tasks and higher steerability toward the retrieved evidence. Crucially, their advantage persists after supervised fine-tuning of both search-pretrained and conventionally pretrained models, indicating that the tested post-training procedures do not fully recover the capabilities developed during pretraining. Throughout our experiments, we find that a 1.4B QueryLM consistently beats or matches the performance of a FLOP-matched 3.3B baseline model across all knowledge intensive tasks. These results establish search-pretraining as a practical way to develop foundational search capabilities during pretraining. We release the training artifacts, including model checkpoints and data. 

---
# MedicalRSI: Medical Recursive Self-Improving Agents via Clinical Model-Harness Co-Evolution 

**Authors**: Ziqing Wang, Hamid Palangi, Lili Zhao, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2611.26806)  

**Abstract**: Medical agents built on large language models (LLMs) have advanced rapidly, yet most stay fixed once deployed and do not keep using their experience to improve later attempts. As a result, the same kind of error can recur across patients. To address this, existing self-improvement methods update an agent's weights or its medical harness from its own experience, but their improving strategy stays fixed and never learns which changes actually help, which caps how much the agent can improve. Recursive self-improvement (RSI) instead improves both the agent and its improving strategy. However, achieving RSI for medical agents is challenging, because clinical workflows rarely come with a reference harness, and a change that fixes early errors may not fix the problems the agent meets later. We propose MedicalRSI, an RSI framework built on Clinical Model-Harness Co-Evolution, in which one model plays two roles. The Solver is the medical agent itself, and the Improver is its learnable improving strategy, which writes a medical harness from the agent's failures. Each harness is rewarded by its gain over acting without it, and both roles are trained together, so the Improver keeps adapting to the problems the Solver has not yet solved. On multiple benchmarks and LLM backbones, MedicalRSI outperforms existing context-level and weight-level self-improvement methods and transfers well to unseen workflows, new LLMs and new databases, a step toward medical agents that keep improving. 

---
# What Helps Language-Model Agents Review Violent Video? A Controlled Comparison with a Human Evidence Reference 

**Authors**: Mohammed El Amine Mokhtari, Matei Mancas, Virginie Vandenbulcke  

**Link**: [PDF](https://arxiv.org/pdf/2611.26776)  

**Abstract**: Systems that flag violent video for a human operator can now be built from language models without training, by combining perception models and letting a language model decide. We ask how the observations of the perception models should reach the decision, and which evidence matters. On the XD-Violence test set (799 clips) we fix one set of perception agents (video description, object detection, sound tagging, speech recognition and speech tagging) and one reasoning model, and compare three interfaces: messages between agents, a typed evidence graph explored with tools, and flat text read in one pass; and a single audio-visual model given the raw clip. Two annotators provide a human reference that records the decision, the incident segments and whether they relied on sight, sound or speech. Flat text matches the single model (F1 0.967 against 0.968), while the graph agent and message passing are less accurate; a variant of the graph agent developed on a separate set did not close the gap, and both graph summaries and multi-turn tool use lowered accuracy. These results hold with reworded prompts and with a larger reasoning model, except that the larger model weakens message passing enough for the graph agent to beat it. Sound events improve F1 by +0.012 [+0.004, +0.021], in line with the annotators, who used sound or speech on most violent clips. Citations of stored observations are faithful, changing about half of the decisions when removed, but no more so with the graph than with flat text, and the agent pipeline costs about 3 times the GPU time of the single model. We also report two artefacts of vision-language describers that mislead pipelines built on them. 

---
# Windchime: an Audiovisual Installation with Audio-Language Model implementations 

**Authors**: Trent Eriksen, Celeste Betancur Gutierrez  

**Link**: [PDF](https://arxiv.org/pdf/2611.26759)  

**Abstract**: Windchime is an audiovisual installation in which spoken language retrieves music and interactive animations from an artist authored corpus. A local Automatic Speech Recognition (ASR) model transcribes each prompt while an audio-language model (ALM) embeds it and retrieval over 406 audio stems returns a soundscape with live coded Strudel fragments that layer into audio output. Simultaneously each spoken input results in retrieval of interactive this http URL animation scenes each containing unique tactile control via monome grid and arc hardware. Windchime distributes agency in three distinct directions; across the artist who authored the possible outputs, the ALM that influences what areas of the corpus can be retrieved, and the visitor whose spoken inputs result in multimodal embodied interaction. 

---
# Self-verifiable LLM-driven science and why it needs human researchers 

**Authors**: Michael G. Medvedev, Nikolai V. Krivoshchapov, Stepan N. Ostarkov  

**Link**: [PDF](https://arxiv.org/pdf/2611.26754)  

**Abstract**: Language models are now used throughout all sciences - for reading literature, extracting numbers from tables and spectra, analyzing data, drafting conclusions - and their answers arrive in the same polished confident tone whether they are right or wrong, so a result has to be re-derived by a human researcher before it can be relied on. The cost of that re-checking, not the ability to generate results, is what currently limits AI-assisted research. In the field of mathematics, this problem is solved by proof assistants like Lean 4 and Coq that let any formal statement and its proof be written down and checked - a perfect match for LLMs, which are now actively used to search for proofs with every step automatically validated, rather than relying on a human to judge the result. In this paper, we propose an extension of this approach to non-mathematical sciences. We describe an AI agent harness that aims to make LLM's scientific reasoning verifiable: the starting assumptions are proposed by LLM and reviewed by a human, and everything built on top of them is checked by a machine with Lean 4. We illustrate how this mechanism works in practice on several examples and our results show that scientific arguments in these tasks can be expressed as formal proofs from explicit assumptions, including experimental results, known theoretical statements, and additional assumptions, produced by the agent. We suggest that this approach could change how scientific research is conducted across all sciences. 

---
# Depth-Linked Residual Tokens for Multimodal Retrieval and Generation 

**Authors**: Jiangfan Zhang, Wentao Bao, Xian Wu, Xiangjun Fan, Jianpeng Cheng, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.26752)  

**Abstract**: Iterative architectures such as looped language models improve parameter efficiency and reasoning capacity of transformers, yet their sequential backbone iterations introduce substantial computational latency. Conversely, parallelizable learnable special tokens (such as the register tokens in Vision Transformer) enable efficient single-pass inference, but often suffer from the absence of sequential refinement and exhibit redundant representations across tokens. To address these challenges, we introduce Depth-Linked Residual Tokens (DepThink), an efficient framework designed for multimodal retrieval and generation. Operating within a single pre-filling forward pass, DepThink appends a sequence of K learnable residual tokens whose hidden states are coupled through a lightweight, learnable residual recurrence. This mechanism enables each residual token to progressively capture complementary information missed by preceding tokens. Theoretically, we prove that DepThink provides strictly expanded geometric directional reachability over standard learnable tokens and guarantees a provable one-step descent advantage. We evaluate DepThink across multimodal retrieval and generation tasks. DepThink consistently outperforms learnable-token baselines and matches or exceeds looped architectures across dense (2B, 8B) and sparse mixture-of-experts (30B-A3B) backbones under both LoRA and full-parameter fine-tuning, while accelerating training by up to 11.9 and inference by 4.3 over looped architectures. 

---
# MomentMem: Memorizing Key Moments for Robot Manipulation 

**Authors**: Guangyu Zhao, Yongchang Zhang, Ruizhang Jiang, Zhihong Yu, Yuhao Zhang, Tao Wu, Qimin Zhou, Kunsong Shi, Haoran Wen, Zeyu Zhang, Yin Li, Kun Zhan, Yiwu Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2611.26745)  

**Abstract**: Robotic manipulation often requires remembering visual information that is no longer observable, such as the original location of an object. Existing approaches typically rely on full observation histories, which become redundant over long horizons, or text summaries, which may omit precise visual details. We study visual moment memory, the ability to retain such details from key moments for subsequent manipulation. To support this study, we introduce VMBench, a suite of long-horizon manipulation tasks requiring memory of spatial locations, visual appearances, and their associations. We further propose MomentMem, a hierarchical architecture in which a VLM-based memorizer selects keyframes and an executor predicts actions conditioned on the selected memories. This separation enables us to evaluate memory selection alongside closed-loop task success and to study memorizer generalization independently of the executor. On VMBench, MomentMem achieves 49.9% average success, outperforming the strongest evaluated baseline by 31.7 percentage points. Moreover, we find that the memorizer transfers well to unseen appearances zero-shot. It can also adapt to an unseen task with only a few memory annotations, reaching 43.7% success, close to the 44.3% obtained with full memory supervision. Further experiments show competitive performance on RMBench and improvements in real-robot manipulation. These results highlight the value of retaining sparse visual evidence for memory-dependent manipulation. Our code is available at this https URL. 

---
# INCISE: Training-Free In-Context Segmentation of Surgical Videos 

**Authors**: Yifan Xu, Bofang Zheng, Guangyu Meng, Nishchal Sapkota, Xianshi Ma, Xu Cao, Tianjiao Yu, Danny Z Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.26734)  

**Abstract**: Surgical video analysis requires video object segmentation (VOS), yet existing methods are commonly based on pretrained model fine-tuning using a large number of expert labels. In-context segmentation (ICS) offers the promise of bypassing this high cost by segmenting a target video using a few annotated references, which are the sole source of surgical knowledge for a frozen model. However, existing ICS methods tend to draw such references at random, which concentrate on a few views that dominate a surgical video and can miss other objects of interest (e.g., transient instruments and small structures). To address these challenges, we propose INCISE, a training-free ICS framework for surgical videos. INCISE selects unlabeled frames from multiple surgical videos that jointly cover the local appearances of a target video, and then segments the target video using these references. With only 1.7% of training frames selected and annotated as references and no target video frame labeled, INCISE obtains 67.2% mean mIoU and outperforms the best SOTA fine-tuned method by 9.2 points. We hope INCISE can help establish ICS as a practical alternative to fine-tuning in surgical video analysis and other domains with a costly labeling burden. 

---
# Model the Annotation, Not the Noise: Semantic Annotation-Channel Prompt Learning against Noisy Labels 

**Authors**: Zibin Geng, Yitong Bao, Jia Li, Xuefeng Jiang, Sheng Sun, Yuwei Wang, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.26702)  

**Abstract**: Parameter-efficient prompt learning has become a cornerstone for adapting pre-trained vision-language models (VLMs), yet it remains highly vulnerable to label noise in few-shot Web and retrieval scenarios. Existing noise robust methods typically rely on hard sample selection or deterministic label correction, which discard valuable semantic ambiguity and overlook how observed annotations are actually generated. To bridge this gap, we reformulate noisy prompt learning through the perspective of data generation, proposing SACP, a Semantic-aware Annotation Channel-based Prompt learning framework. Instead of suppressing noise heuristically, SACP decouples latent clean-label prediction from instance-dependent annotation generation. By integrating frozen VLM semantic priors with lightweight visual residuals, SACP constructs a structured annotation channel optimized via generalized expectation-maximization with soft posteriors. Furthermore, a posterior-aware memory is introduced to preserve label uncertainty during test-time inference. Extensive experiments across different synthetic benchmarks and the real-world noisy Food101N dataset demonstrate that SACP achieves state-of-the-art performance against its counterparts in different label noise patterns, especially under severe asymmetric noise. 

---
# On the loss landscape of the linear neural network: beyond critical points 

**Authors**: Antoine Lejay, Lionel Lenotre, Sa{ï}d Toubra  

**Link**: [PDF](https://arxiv.org/pdf/2611.26671)  

**Abstract**: While the critical points of the loss landscape in linear neural networks have been extensively studied since the seminal work of Baldi \& Hornik in the late 1980s, a description of this landscape at an intermediate or global scale is still lacking. Such a characterization is crucial, particularly as a substantial body of literature indicates that gradient descent circumvents critical points. The present article introduces an intermediate structure, termed the critical thalweg, which encompasses all critical points and exhibits key geometric properties. Beyond providing a geometric proof of Baldi \& Hornik's results using simpler assumptions than the existing literature, this framework offers a global view of the loss landscape that can be easily visualized in a two-dimensional plane. Empirical observations suggest that gradient descent is attracted toward the critical thalweg, along which the loss continues to decrease to its global minimum. 

---
# Computing Weighted Set Covers with Heterogeneous Stochastic Oracles: Deadline-Constrained Token Complexity 

**Authors**: Jie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26656)  

**Abstract**: We study computation in which a probabilistic Turing machine (PTM) uses AI systems, abstracted as heterogeneous stochastic oracles (HSOs), to solve Weighted Set Cover. A pilot study shows that an LLM can outperform weighted greedy. HSOs differ in capabilities, response distributions, latencies, and token costs. We study how the PTM selects HSOs to obtain, by a deadline and with a required success probability, a full cover satisfying a prescribed weight threshold, and we analyze expected token cost.
For queries requesting full covers, assuming independent call successes and uninformative failures, we characterize by linear programming the minimum expected token cost for every feasible success probability. An optimal policy randomizes over at most two preselected sequences of oracle batches. Among policies attaining maximum success probability, the expected cost reduction of two-batch selection relative to querying all oracles in one batch is nonnegative and submodular. We construct an FPTAS for near-optimal cost reduction under heterogeneous response-time bounds. Computing an optimal two-batch selection is NP-hard but admits a pseudopolynomial algorithm for positive integral expected token costs.
For partial-cover responses, the PTM combines returned sets across batches and turns. Once their union covers the universe, it verifies total weight directly or obtains a guarantee from LP-dual tests on retained partial covers. We give one-turn probability, token-cost, and hardness results and adaptive token-cost bounds under per-batch assumptions. A constructed instance shows that residual-guided querying can reduce expected token cost by a factor linear in the number of possible residual subproblems. 

---
# VerifiQuant: Governance-by-Execution for Financial LLM Workflows 

**Authors**: Yu-Chi Ko, Mi-Yen Yeh, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.26649)  

**Abstract**: Large language models embedded in financial workflows fail most dangerously by emitting confident, well-formatted but wrong numbers with no signal distinguishing them from correct ones. We argue that the central deployment problem is governability: a system must determine when a question supports a numeric answer and when it should instead clarify, intercept, or refuse. We present VerifiQuant, a governance-by-execution framework that packages financial logic into inference contracts and evaluates each query through an executable diagnostic funnel before deterministic computation. This separation makes computation, clarification, and refusal structured and auditable while keeping financial arithmetic outside the language model. Against matched retrieval-and-execution baselines, and within a declared contract library, the principal benefit is not higher accuracy on well-posed questions but safer behavior when an input does not justify a number. When it finds no applicable contract, it says so instead of guessing. Coverage sweeps and failure analysis put a price on this boundary: safety is bought with coverage rather than with accuracy, and residual failures remain attributable to specific runtime frontiers. 

---
# Grid Cells as Spectral Successor Features: A Reusable Basis for Continual Reinforcement Learning 

**Authors**: Élodie Côté-Gauthier, Doina Precup, Isabeau Prémont-Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2611.26504)  

**Abstract**: Continual reinforcement learning requires agents to adapt when the task changes while reusing what they know about the environment. Successor features separate dynamics from task-specific reward, but their effectiveness depends on the choice of base features. Taking the identification of entorhinal grid cells with eigenvectors of the successor representation as a design principle, we build this basis from eigenvectors of the transition matrix of a fixed reference policy. With such a basis, the successor features at a state-action pair are a linear combination of the base features at that same pair, which is not true in general, and the coefficients are known in closed form: each successor feature is its base feature multiplied by $\theta_i = 1/(1-\gamma\lambda_i)$, where $\lambda_i$ is its eigenvalue. The long-term return is thus predicted from the same features as the immediate reward, no successor-feature network or replay buffer is needed, and adapting to a new task reduces to a linear regression on the reward. The agent acts greedily on the action values of the reference policy, which is one step of policy improvement over it and suffices to reach a single goal. In a two-goal gridworld, with the uniform random walk as reference policy, whose eigenvectors are the low-frequency Laplacian eigenvectors, agents built this way maintain near-maximal return across every change of goal, whereas SimpleSF and DQN lose most of their return at each change. 

---
